from __future__ import annotations

import json
import gc
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:  # pragma: no cover - optional runtime dependency
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    ensure_image_paths,
    ensure_object_names,
    get_rex_omni_backend,
    get_sam2_predictor,
    get_sam2_video_predictor,
    get_tool_settings,
    get_vggt_backend,
    load_pil_image,
    load_rgb_array,
    map_point_to_preprocessed,
    resolve_device,
    save_bbox_overlay,
)
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.video_counting_3d_utils import (
    constrained_greedy_cluster,
    median_point,
    normalize_rex_bbox_predictions,
    sample_bbox_pixels,
)


class CountVideoObjects3DTool(BaseSpatialTool):
    name = "CountVideoObjects3D"
    description = (
        "Count unique object instances in a video using ViSRA-style 3D object detection: "
        "Rex-Omni 2D views, optional SAM2 masks, VGGT 3D lifting, and constrained greedy clustering."
    )
    args_schema = {
        "type": "object",
        "properties": {
            "images": {"type": "array", "items": {"type": "string"}},
            "objects": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
        },
        "required": ["objects"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config

    def invoke(self, **kwargs) -> Dict[str, Any]:
        image_paths = ensure_image_paths(kwargs.get("images") or kwargs.get("image"))
        objects = ensure_object_names(kwargs.get("objects"))
        if not image_paths:
            return self.error("CountVideoObjects3D requires sampled video frame paths.")
        if not objects:
            return self.error("CountVideoObjects3D requires one or more object names.")

        settings = get_tool_settings(self.config, self.name, aliases=["video_counting_3d", "visra_counting"])
        max_frames = int(settings.get("num_frames", 64))
        selected_paths = _uniform_sample(image_paths, max_frames) if max_frames > 0 else image_paths
        device = resolve_device(settings.get("device"))
        preprocess_mode = str(settings.get("preprocess_mode", "pad"))
        cg_threshold = float(settings.get("cg_distance_threshold", 0.35))
        bbox_stride = int(settings.get("bbox_point_stride", 8))
        use_sam_masks = bool(settings.get("use_sam_masks", True))
        use_tracking = bool(settings.get("use_tracking", True))
        tracking_max_frames = int(settings.get("tracking_max_frames", 50))
        tracking_absent_patience = int(settings.get("tracking_absent_patience", 2))
        max_detections_per_frame = int(settings.get("max_detections_per_frame", 20))

        try:  # pragma: no cover - dependency-heavy runtime path
            rex_settings = get_tool_settings(self.config, "CountObjects", aliases=["counting", "count"])
            rex_backend = get_rex_omni_backend(
                model_path=str(
                    settings.get("rex_model_path")
                    or rex_settings.get("model_path")
                    or rex_settings.get("model_id")
                    or "IDEA-Research/Rex-Omni"
                ),
                backend=str(settings.get("rex_backend") or rex_settings.get("backend", "transformers")),
                device=device,
                repo_path=str(settings.get("rex_repo_path") or rex_settings.get("repo_path")) if (settings.get("rex_repo_path") or rex_settings.get("repo_path")) else None,
                quantization=str(settings.get("rex_quantization") or rex_settings.get("quantization")) if (settings.get("rex_quantization") or rex_settings.get("quantization")) else None,
                attn_implementation=str(settings.get("attn_implementation") or rex_settings.get("attn_implementation", "sdpa")),
                device_map=str(settings.get("device_map") or rex_settings.get("device_map", "auto")),
                max_tokens=int(settings.get("max_tokens") or rex_settings.get("max_tokens", 2048)),
                temperature=float(settings.get("temperature") or rex_settings.get("temperature", 0.0)),
                top_p=float(settings.get("top_p") or rex_settings.get("top_p", 0.05)),
                top_k=int(settings.get("top_k") or rex_settings.get("top_k", 1)),
                repetition_penalty=float(settings.get("repetition_penalty") or rex_settings.get("repetition_penalty", 1.05)),
            )

            detection_records: List[Dict[str, Any]] = []
            frame_detections: List[Dict[str, Any]] = []
            for frame_index, image_path in enumerate(selected_paths):
                image = load_pil_image(image_path)
                detections = _detect_frame_rex(
                    rex_backend["wrapper"],
                    image,
                    objects,
                    max_detections=max_detections_per_frame,
                )
                frame_detections.append(
                    {
                        "image": f"image-{frame_index}",
                        "image_path": image_path,
                        "candidate_count": len(detections),
                    }
                )
                for det_index, det in enumerate(detections):
                    detection_records.append(
                        {
                            "frame_index": frame_index,
                            "det_index": det_index,
                            "image_path": image_path,
                            "image_size": image.size,
                            "object": det["object"],
                            "bbox": det["bbox"],
                            "score": det.get("score"),
                        }
                    )
            del rex_backend
            _clear_cached_backend(get_rex_omni_backend)
            torch = _import_torch()
            _empty_torch_cache(torch)

            vggt_backend = get_vggt_backend(
                model_id=str(settings.get("vggt_model_id", "facebook/VGGT-1B")),
                checkpoint_path=settings.get("vggt_checkpoint_path"),
                device=device,
            )
            torch = vggt_backend["torch"]
            images_tensor = vggt_backend["load_and_preprocess_images"](selected_paths, mode=preprocess_mode).to(device)
            with torch.no_grad():
                predictions = vggt_backend["model"](images_tensor)
            world_points = _extract_world_points(predictions)
            if world_points is None:
                return self.error(
                    "VGGT predictions did not include world_points/world_points_cam; cannot lift detections to 3D.",
                    payload={"prediction_keys": sorted(str(key) for key in predictions.keys())},
                )
            world_points_np = world_points[0].detach().cpu().numpy()
            del predictions, images_tensor, world_points, vggt_backend
            _clear_cached_backend(get_vggt_backend)
            _empty_torch_cache(torch)

            sam_predictor = None
            sam_video_predictor = None
            if use_sam_masks or use_tracking:
                mask_settings = get_tool_settings(self.config, "GetObjectMask", aliases=["mask", "sam2"])
                if use_sam_masks:
                    try:
                        sam_predictor = get_sam2_predictor(
                            model_id=str(settings.get("sam2_model_id") or mask_settings.get("model_id", "facebook/sam2.1-hiera-large")),
                            checkpoint_path=settings.get("sam2_checkpoint_path") or mask_settings.get("checkpoint_path"),
                            config_path=settings.get("sam2_config_path") or mask_settings.get("config_path"),
                            device=device,
                        )
                    except Exception:
                        sam_predictor = None
                if use_tracking:
                    try:
                        sam_video_predictor = get_sam2_video_predictor(
                            model_id=str(settings.get("sam2_model_id") or mask_settings.get("model_id", "facebook/sam2.1-hiera-large")),
                            checkpoint_path=settings.get("sam2_checkpoint_path") or mask_settings.get("checkpoint_path"),
                            config_path=settings.get("sam2_config_path") or mask_settings.get("config_path"),
                            device=device,
                        )
                    except Exception:
                        sam_video_predictor = None

            views: List[Dict[str, Any]] = []
            for record in detection_records:
                frame_index = int(record["frame_index"])
                det_index = int(record["det_index"])
                center_3d = _lift_detection_to_3d(
                    image_path=str(record["image_path"]),
                    image_size=tuple(record["image_size"]),
                    frame_world_points=world_points_np[frame_index],
                    bbox=record["bbox"],
                    bbox_stride=bbox_stride,
                    sam_predictor=sam_predictor,
                    preprocess_mode=preprocess_mode,
                )
                if center_3d is None:
                    continue
                views.append(
                    {
                        "view_id": f"f{frame_index:03d}_d{det_index:03d}",
                        "object": record["object"],
                        "frame_index": frame_index,
                        "image_path": record["image_path"],
                        "bbox": record["bbox"],
                        "score": record.get("score"),
                        "center_3d": center_3d,
                    }
                )

            track_summaries: List[Dict[str, Any]] = []
            if use_tracking and sam_video_predictor is not None and views:
                track_summaries = _attach_sam2_tracks(
                    sam_video_predictor=sam_video_predictor,
                    frame_paths=selected_paths,
                    views=views,
                    max_frames=tracking_max_frames,
                    absent_patience=tracking_absent_patience,
                    device=device,
                )

            instances = []
            for object_name in objects:
                object_views = [view for view in views if view["object"] == object_name]
                instances.extend(constrained_greedy_cluster(object_views, distance_threshold=cg_threshold))

            instances.sort(key=lambda item: (-int(item["member_count"]), item["instance_id"]))
            artifact_dir = artifact_dir_for_tool(self.config, self.name)
            manifest_path = artifact_dir / "visra_3d_counting.json"
            manifest = {
                "instance_count": len(instances),
                "objects": objects,
                "instances": instances,
                "views": views,
                "tracks": track_summaries,
                "frame_summaries": frame_detections,
                "pipeline_stats": {
                    "requested_frame_count": len(image_paths),
                    "sampled_frame_count": len(selected_paths),
                    "view_count": len(views),
                    "track_count": len(track_summaries),
                    "cg_distance_threshold": cg_threshold,
                    "bbox_point_stride": bbox_stride,
                    "use_sam_masks": use_sam_masks,
                    "sam_masks_available": sam_predictor is not None,
                    "use_tracking": use_tracking,
                    "sam_tracking_available": sam_video_predictor is not None,
                    "tracking_max_frames": tracking_max_frames,
                    "tracking_absent_patience": tracking_absent_patience,
                    "preprocess_mode": preprocess_mode,
                },
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

            overlay_path = artifact_dir / "visra_2d_views.png"
            overlay_artifact = _save_views_overlay(selected_paths, views, overlay_path)
            artifacts = [str(manifest_path)]
            if overlay_artifact:
                artifacts.append(overlay_artifact)

            return self.success(
                payload={
                    "instance_count": len(instances),
                    "instances": instances,
                    "tracks": track_summaries,
                    "frame_summaries": frame_detections,
                    "pipeline_stats": manifest["pipeline_stats"],
                    "backend": f"visra_3d:rex_omni+sam2+vggt+cg",
                    "artifact_descriptions": [
                        {
                            "path": str(manifest_path),
                            "kind": "visra_3d_manifest",
                            "description": "3D views and CG-clustered physical object instances.",
                        },
                        {
                            "path": str(overlay_path),
                            "kind": "2d_view_overlay",
                            "description": "Rex-Omni 2D views used for VGGT lifting and CG clustering.",
                        },
                    ],
                },
                artifacts=artifacts,
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"ViSRA-style 3D video counting failed to initialize or run: {exc}")


def _uniform_sample(paths: Sequence[str], count: int) -> List[str]:
    if len(paths) <= count:
        return list(paths)
    if count <= 1:
        return [paths[0]]
    indices = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[index] for index in indices]


def _detect_frame_rex(wrapper: Any, image: Any, objects: Sequence[str], max_detections: int) -> List[Dict[str, Any]]:
    outputs = wrapper.inference(images=image, task="detection", categories=list(objects))
    detections = normalize_rex_bbox_predictions(outputs, objects, image.size)
    detections.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return detections[:max(1, max_detections)]


def _extract_world_points(predictions: Dict[str, Any]) -> Any | None:
    for key in ("world_points", "world_points_cam", "points3D", "points_3d"):
        if key in predictions:
            return predictions[key]
    return None


def _clear_cached_backend(factory: Any) -> None:
    cache_clear = getattr(factory, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    gc.collect()


def _empty_torch_cache(torch: Any) -> None:
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _lift_detection_to_3d(
    image_path: str,
    image_size: Tuple[int, int],
    frame_world_points: Any,
    bbox: Sequence[float],
    bbox_stride: int,
    sam_predictor: Any | None,
    preprocess_mode: str,
) -> List[float] | None:
    if np is None:
        return None
    pixels: List[Tuple[int, int]]
    if sam_predictor is not None:
        try:
            image_rgb = load_rgb_array(image_path)
            sam_predictor.set_image(image_rgb)
            box = np.asarray(bbox, dtype=np.float32)
            masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
            best_index = int(np.argmax(scores))
            mask = masks[best_index].astype(bool)
            ys, xs = np.where(mask)
            if xs.size > 0:
                stride = max(1, bbox_stride)
                pixels = list(zip(xs[::stride].tolist(), ys[::stride].tolist()))
            else:
                pixels = sample_bbox_pixels(bbox, image_size, bbox_stride)
        except Exception:
            pixels = sample_bbox_pixels(bbox, image_size, bbox_stride)
    else:
        pixels = sample_bbox_pixels(bbox, image_size, bbox_stride)

    h, w = frame_world_points.shape[:2]
    points = []
    for x, y in pixels:
        mapped_x, mapped_y = map_point_to_preprocessed(
            (float(x), float(y)),
            image_size,
            mode=preprocess_mode,
            target_size=max(h, w),
        )
        px = int(round(max(0.0, min(float(w - 1), mapped_x))))
        py = int(round(max(0.0, min(float(h - 1), mapped_y))))
        point = frame_world_points[py, px]
        if not np.all(np.isfinite(point[:3])):
            continue
        points.append([float(point[0]), float(point[1]), float(point[2])])
    return median_point(points)


def _attach_sam2_tracks(
    sam_video_predictor: Any,
    frame_paths: Sequence[str],
    views: List[Dict[str, Any]],
    max_frames: int,
    absent_patience: int,
    device: str,
) -> List[Dict[str, Any]]:
    if np is None:
        return []
    torch = _import_torch()
    track_summaries: List[Dict[str, Any]] = []
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for view in views:
        by_frame.setdefault(int(view["frame_index"]), []).append(view)

    temp_dir = tempfile.mkdtemp(prefix="visra_sam2_frames_")
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            shutil.copyfile(frame_path, str(Path(temp_dir) / f"{frame_index:05d}.jpg"))

        next_obj_id = 1
        for start_frame in sorted(by_frame):
            seeded_views = [view for view in by_frame[start_frame] if not view.get("track_id")]
            if not seeded_views:
                continue

            inference_state = sam_video_predictor.init_state(video_path=temp_dir)
            if hasattr(sam_video_predictor, "reset_state"):
                sam_video_predictor.reset_state(inference_state)

            obj_to_view: Dict[int, Dict[str, Any]] = {}
            for view in seeded_views:
                obj_id = next_obj_id
                next_obj_id += 1
                obj_to_view[obj_id] = view
                box = torch.tensor(view["bbox"], dtype=torch.float32, device=device)
                sam_video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=start_frame,
                    obj_id=obj_id,
                    box=box,
                )

            seen_frames: Dict[int, List[int]] = {obj_id: [] for obj_id in obj_to_view}
            last_seen: Dict[int, int] = {obj_id: start_frame for obj_id in obj_to_view}
            stop_frame = min(len(frame_paths) - 1, start_frame + max(1, max_frames) - 1)
            for out_frame_idx, out_obj_ids, out_mask_logits in sam_video_predictor.propagate_in_video(inference_state):
                frame_index = int(out_frame_idx)
                if frame_index < start_frame:
                    continue
                if frame_index > stop_frame:
                    break
                for local_index, obj_id in enumerate(out_obj_ids):
                    obj_id = int(obj_id)
                    if obj_id not in obj_to_view:
                        continue
                    mask = (out_mask_logits[local_index] > 0.0).detach().cpu().numpy().squeeze()
                    if not mask.any():
                        continue
                    seen_frames[obj_id].append(frame_index)
                    last_seen[obj_id] = frame_index
                    seed_view = obj_to_view[obj_id]
                    _assign_track_to_overlapping_views(
                        views=by_frame.get(frame_index, []),
                        object_name=str(seed_view["object"]),
                        mask=mask,
                        track_id=f"sam2_{obj_id}",
                    )

                if absent_patience > 0 and all(frame_index - last_seen[obj_id] >= absent_patience for obj_id in obj_to_view):
                    break

            for obj_id, seed_view in obj_to_view.items():
                frames = sorted(set(seen_frames.get(obj_id, [])))
                if not frames:
                    frames = [start_frame]
                track_id = f"sam2_{obj_id}"
                seed_view["track_id"] = seed_view.get("track_id") or track_id
                track_summaries.append(
                    {
                        "track_id": track_id,
                        "object": seed_view["object"],
                        "seed_view_id": seed_view["view_id"],
                        "start_frame": start_frame,
                        "frames": frames,
                        "member_view_ids": sorted(
                            view["view_id"] for view in views if view.get("track_id") == track_id
                        ),
                    }
                )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return track_summaries


def _assign_track_to_overlapping_views(
    views: Sequence[Dict[str, Any]],
    object_name: str,
    mask: Any,
    track_id: str,
) -> None:
    best_view = None
    best_overlap = 0.0
    for view in views:
        if view.get("track_id") or view.get("object") != object_name:
            continue
        overlap = _mask_bbox_overlap(mask, view["bbox"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_view = view
    if best_view is not None and best_overlap > 0.0:
        best_view["track_id"] = track_id


def _mask_bbox_overlap(mask: Any, bbox: Sequence[float]) -> float:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return 0.0
    mx1, my1, mx2, my2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    ix1, iy1 = max(mx1, x1), max(my1, y1)
    ix2, iy2 = min(mx2, x2), min(my2, y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
    return intersection / bbox_area


def _import_torch() -> Any:
    import torch  # pragma: no cover - optional runtime dependency

    return torch


def _save_views_overlay(frame_paths: Sequence[str], views: Sequence[Dict[str, Any]], output_path: Path) -> str | None:
    if not frame_paths or not views:
        return None
    selected_views = []
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for view in views:
        by_frame.setdefault(int(view["frame_index"]), []).append(
            {
                "label": f"{view['object']}:{view['view_id']}",
                "bbox": view["bbox"],
                "score": view.get("score"),
            }
        )
    rendered_paths = []
    for frame_index, frame_views in by_frame.items():
        if 0 <= frame_index < len(frame_paths):
            selected_views.extend(frame_views)
    if not selected_views:
        return None
    # Use the first frame as a compact smoke artifact; full per-frame details live in the manifest.
    first_frame = min(by_frame)
    image = load_pil_image(frame_paths[first_frame])
    save_bbox_overlay(image, by_frame[first_frame], output_path)
    rendered_paths.append(str(output_path))
    return rendered_paths[0]
