from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

try:  # pragma: no cover - optional runtime dependency
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    ensure_image_paths,
    ensure_object_names,
    get_sam2_predictor,
    get_tool_settings,
    get_vggt_backend,
    load_pil_image,
    load_rgb_array,
    map_point_to_preprocessed,
    resolve_device,
    save_bbox_overlay,
)
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.localization import LocalizeObjectsTool


ADAPTIVE_P05_P90_RATIO_MAX = 3.692
ADAPTIVE_P05_P90_SPREAD_MIN = 0.471


class EstimateObjectDistance3DTool(BaseSpatialTool):
    name = "EstimateObjectDistance3D"
    description = (
        "Estimate the absolute 3D distance in meters between two objects in a video "
        "using multi-frame localization, SAM2 masks, VGGT reconstruction, and object point-cloud statistics."
    )
    args_schema = {
        "type": "object",
        "properties": {
            "images": {"type": "array", "items": {"type": "string"}},
            "objects": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
            "object_1": {"type": "string"},
            "object_2": {"type": "string"},
        },
        "required": ["images"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config
        self.localizer = LocalizeObjectsTool(config)

    def invoke(self, **kwargs) -> Dict[str, Any]:
        image_paths = ensure_image_paths(kwargs.get("images") or kwargs.get("image"))
        objects = _parse_object_pair(kwargs)
        if len(image_paths) < 1:
            return self.error("EstimateObjectDistance3D requires video frame images.")
        if not objects:
            return self.error("EstimateObjectDistance3D requires exactly two object names.")
        if np is None:
            return self.unavailable("NumPy is required for EstimateObjectDistance3D.")

        settings = get_tool_settings(
            self.config,
            self.name,
            aliases=["object_distance_3d", "video_distance_3d", "distance"],
        )
        camera_settings = get_tool_settings(self.config, "GetCameraParametersVGGT", aliases=["camera", "vggt"])
        mask_settings = get_tool_settings(self.config, "GetObjectMask", aliases=["mask", "sam2"])
        top_frames = int(settings.get("top_distance_frames", 6))
        max_points = int(settings.get("mask_max_points", 128))
        aggregate = str(settings.get("pointcloud_aggregate", "adaptive_p05_p90"))
        preprocess_mode = str(settings.get("preprocess_mode") or camera_settings.get("preprocess_mode", "pad"))
        device = resolve_device(settings.get("device") or camera_settings.get("device") or mask_settings.get("device"))

        try:  # pragma: no cover - dependency-heavy runtime path
            frame_records = self._localize_frames(image_paths, objects)
            selected_records = _select_top_same_frame_records(frame_records, top_frames)
            if not selected_records:
                return self.error(
                    "No sampled frame localized both objects.",
                    payload={"objects": list(objects), "frame_results": frame_records},
                )

            predictor = get_sam2_predictor(
                model_id=str(mask_settings.get("model_id", "facebook/sam2.1-hiera-large")),
                checkpoint_path=mask_settings.get("checkpoint_path"),
                config_path=mask_settings.get("config_path"),
                device=device,
            )
            backend = get_vggt_backend(
                model_id=str(
                    settings.get("hf_model_id")
                    or settings.get("vggt_model_id")
                    or camera_settings.get("hf_model_id")
                    or "facebook/VGGT-1B"
                ),
                checkpoint_path=(
                    settings.get("checkpoint_path")
                    or settings.get("vggt_checkpoint_path")
                    or camera_settings.get("checkpoint_path")
                ),
                device=device,
            )
            selected_paths = [image_paths[int(record["frame_position"])] for record in selected_records]
            images = backend["load_and_preprocess_images"](selected_paths, mode=preprocess_mode).to(device)
            with backend["torch"].no_grad():
                predictions = backend["model"](images)
            world_points = _extract_world_points_sequence(predictions)
            if world_points is None:
                return self.error(
                    "VGGT predictions did not include world_points/world_points_cam; cannot calculate object distance.",
                    payload={"prediction_keys": sorted(str(key) for key in predictions.keys())},
                )

            object_1_points = []
            object_2_points = []
            frame_summaries = []
            for local_index, record in enumerate(selected_records):
                image_path = image_paths[int(record["frame_position"])]
                image = load_pil_image(image_path)
                predictor.set_image(load_rgb_array(image_path))
                mask_1 = _predict_sam_mask(predictor, record["region_1"]["bbox"])
                mask_2 = _predict_sam_mask(predictor, record["region_2"]["bbox"])
                points_1 = _mask_pixels_to_3d(
                    mask=mask_1,
                    image_size=image.size,
                    frame_points=world_points[local_index],
                    preprocess_mode=preprocess_mode,
                    max_points=max_points,
                )
                points_2 = _mask_pixels_to_3d(
                    mask=mask_2,
                    image_size=image.size,
                    frame_points=world_points[local_index],
                    preprocess_mode=preprocess_mode,
                    max_points=max_points,
                )
                if len(points_1):
                    object_1_points.append(points_1)
                if len(points_2):
                    object_2_points.append(points_2)
                frame_summaries.append(
                    {
                        "frame_position": record["frame_position"],
                        "image": image_path,
                        "localization_quality": record.get("localization_quality"),
                        "pointcloud_sizes": [int(len(points_1)), int(len(points_2))],
                        "mask_pixels": [int(mask_1.sum()), int(mask_2.sum())],
                    }
                )

            if not object_1_points or not object_2_points:
                return self.error(
                    "No finite mask pointclouds for one or both objects.",
                    payload={"objects": list(objects), "frames": frame_summaries},
                )
            points_1_all = np.concatenate(object_1_points, axis=0)
            points_2_all = np.concatenate(object_2_points, axis=0)
            distances = np.linalg.norm(points_1_all[:, None, :] - points_2_all[None, :, :], axis=2).reshape(-1)
            distances = distances[np.isfinite(distances)]
            if distances.size == 0:
                return self.error("No finite pairwise point-cloud distances.")

            aggregate_policy = _aggregate_policy(distances, aggregate)
            distance_meters = aggregate_policy["value"]
            return self.success(
                payload={
                    "distance_meters": float(distance_meters),
                    "unit": "meters",
                    "objects": list(objects),
                    "backend": "grounding_dino+sam2+vggt_object_pointcloud",
                    "method": "mask_pointcloud_multiframe",
                    "top_distance_frames": top_frames,
                    "selected_frame_positions": [int(record["frame_position"]) for record in selected_records],
                    "aggregate": aggregate,
                    "selected_aggregate": aggregate_policy["selected_aggregate"],
                    "aggregate_policy": aggregate_policy,
                    "pointcloud_sizes": [int(len(points_1_all)), int(len(points_2_all))],
                    "frames": frame_summaries,
                    "distance_stats": _distance_stats(distances),
                    "frame_results": frame_records,
                },
                artifacts=[artifact for item in frame_records for artifact in item.get("artifacts", [])],
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"Object 3D distance estimation failed: {exc}")

    def _localize_frames(self, image_paths: Sequence[str], objects: Tuple[str, str]) -> List[Dict[str, Any]]:
        records = []
        for index, image_path in enumerate(image_paths):
            localization = self.localizer.invoke(image=image_path, objects=list(objects))
            record: Dict[str, Any] = {
                "frame_position": index,
                "image": image_path,
                "localization_status": localization.get("status"),
                "artifacts": localization.get("artifacts") or [],
            }
            if localization.get("status") != "success":
                record.update({"status": "error", "error": localization.get("error")})
                records.append(record)
                continue
            regions = localization.get("payload", {}).get("regions", [])
            region_1 = _best_region(regions, objects[0])
            region_2 = _best_region(regions, objects[1])
            record.update({"region_1": region_1, "region_2": region_2, "regions": regions})
            if not region_1 or not region_2:
                record.update({"status": "error", "error": "Failed to localize both objects."})
            else:
                quality = min(float(region_1.get("score") or 0.0), float(region_2.get("score") or 0.0))
                record.update({"status": "localized", "localization_quality": quality})
            records.append(record)
        return records


class CompareObjectDistance3DTool(BaseSpatialTool):
    name = "CompareObjectDistance3D"
    description = (
        "Compare 3D distances from a reference object to candidate objects in a video, "
        "using EstimateObjectDistance3D as the geometric primitive."
    )
    args_schema = {
        "type": "object",
        "properties": {
            "images": {"type": "array", "items": {"type": "string"}},
            "reference_object": {"type": "string"},
            "candidate_objects": {"type": "array", "items": {"type": "string"}},
            "mode": {"type": "string", "enum": ["closest", "farthest"]},
        },
        "required": ["images", "reference_object", "candidate_objects"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config
        self.localizer = LocalizeObjectsTool(config)

    def invoke(self, **kwargs) -> Dict[str, Any]:
        image_paths = ensure_image_paths(kwargs.get("images") or kwargs.get("image"))
        reference_object = str(kwargs.get("reference_object") or "").strip()
        candidate_objects = ensure_object_names(kwargs.get("candidate_objects") or kwargs.get("objects"))
        mode = str(kwargs.get("mode") or "closest").strip().lower()
        if mode not in {"closest", "farthest"}:
            return self.error("CompareObjectDistance3D mode must be 'closest' or 'farthest'.")
        if not image_paths:
            return self.error("CompareObjectDistance3D requires video frame images.")
        if not reference_object:
            return self.error("CompareObjectDistance3D requires a reference_object.")
        if not candidate_objects:
            return self.error("CompareObjectDistance3D requires candidate_objects.")
        if np is None:
            return self.unavailable("NumPy is required for CompareObjectDistance3D.")

        settings = get_tool_settings(
            self.config,
            self.name,
            aliases=["compare_object_distance_3d", "object_rel_distance", "object_distance_3d", "distance"],
        )
        camera_settings = get_tool_settings(self.config, "GetCameraParametersVGGT", aliases=["camera", "vggt"])
        mask_settings = get_tool_settings(self.config, "GetObjectMask", aliases=["mask", "sam2"])
        frames_per_candidate = int(settings.get("frames_per_candidate", 2))
        max_frames = int(settings.get("max_compare_frames", 8))
        max_points = int(settings.get("mask_max_points", 128))
        min_mask_pixels = int(settings.get("min_mask_pixels", 1000))
        duplicate_iou_threshold = float(settings.get("duplicate_bbox_iou_threshold", 0.95))
        aggregate = str(settings.get("pointcloud_aggregate", "p90"))
        preprocess_mode = str(settings.get("preprocess_mode") or camera_settings.get("preprocess_mode", "pad"))
        device = resolve_device(settings.get("device") or camera_settings.get("device") or mask_settings.get("device"))

        try:  # pragma: no cover - dependency-heavy runtime path
            objects = _dedupe_objects([reference_object, *candidate_objects])
            frame_records = self._localize_frames(image_paths, objects, reference_object, candidate_objects)
            selected_records = _select_shared_compare_records(
                frame_records=frame_records,
                reference_object=reference_object,
                candidate_objects=candidate_objects,
                frames_per_candidate=frames_per_candidate,
                max_frames=max_frames,
            )
            if not selected_records:
                return self.error(
                    "No frames localized the reference object with any candidate object.",
                    payload={"reference_object": reference_object, "candidate_objects": candidate_objects, "frame_results": frame_records},
                )

            predictor = get_sam2_predictor(
                model_id=str(mask_settings.get("model_id", "facebook/sam2.1-hiera-large")),
                checkpoint_path=mask_settings.get("checkpoint_path"),
                config_path=mask_settings.get("config_path"),
                device=device,
            )
            backend = get_vggt_backend(
                model_id=str(
                    settings.get("hf_model_id")
                    or settings.get("vggt_model_id")
                    or camera_settings.get("hf_model_id")
                    or "facebook/VGGT-1B"
                ),
                checkpoint_path=(
                    settings.get("checkpoint_path")
                    or settings.get("vggt_checkpoint_path")
                    or camera_settings.get("checkpoint_path")
                ),
                device=device,
            )
            selected_paths = [image_paths[int(record["frame_position"])] for record in selected_records]
            images = backend["load_and_preprocess_images"](selected_paths, mode=preprocess_mode).to(device)
            with backend["torch"].no_grad():
                predictions = backend["model"](images)
            world_points = _extract_world_points_sequence(predictions)
            if world_points is None:
                return self.error("VGGT predictions did not include world_points/world_points_cam.")

            points_by_object: Dict[str, List[Any]] = {name: [] for name in objects}
            reference_instances: List[Dict[str, Any]] = []
            frame_summaries = []
            for local_index, record in enumerate(selected_records):
                image_path = image_paths[int(record["frame_position"])]
                image = load_pil_image(image_path)
                predictor.set_image(load_rgb_array(image_path))
                duplicate_objects = _duplicate_bbox_objects(
                    record.get("regions_by_object") or {},
                    objects,
                    duplicate_iou_threshold,
                )
                frame_summary: Dict[str, Any] = {
                    "frame_position": record["frame_position"],
                    "image": image_path,
                    "objects": {},
                }
                for object_name in objects:
                    region = (record.get("regions_by_object") or {}).get(object_name)
                    if not region:
                        continue
                    if object_name in duplicate_objects:
                        frame_summary["objects"][object_name] = {
                            "score": region.get("score"),
                            "pointcloud_size": 0,
                            "mask_pixels": 0,
                            "skipped": "duplicate_bbox",
                        }
                        continue
                    mask = _predict_sam_mask(predictor, region["bbox"])
                    mask_pixels = int(mask.sum())
                    if mask_pixels < min_mask_pixels:
                        frame_summary["objects"][object_name] = {
                            "score": region.get("score"),
                            "pointcloud_size": 0,
                            "mask_pixels": mask_pixels,
                            "skipped": "small_mask",
                        }
                        continue
                    points = _mask_pixels_to_3d(
                        mask=mask,
                        image_size=image.size,
                        frame_points=world_points[local_index],
                        preprocess_mode=preprocess_mode,
                        max_points=max_points,
                    )
                    if len(points):
                        if object_name == reference_object:
                            reference_instances.append(
                                {
                                    "points": points,
                                    "frame_position": int(record["frame_position"]),
                                    "score": float(region.get("score") or 0.0),
                                    "mask_pixels": mask_pixels,
                                }
                            )
                        else:
                            points_by_object[object_name].append(points)
                    frame_summary["objects"][object_name] = {
                        "score": region.get("score"),
                        "pointcloud_size": int(len(points)),
                        "mask_pixels": mask_pixels,
                    }
                frame_summaries.append(frame_summary)

            reference_instance = _select_reference_instance(reference_instances)
            if reference_instance is None:
                return self.error(
                    "No finite point cloud for reference object.",
                    payload={"reference_object": reference_object, "frames": frame_summaries},
                )
            reference_points = reference_instance["points"]
            contact_shortcut = _select_bbox_contact_shortcut(
                selected_records=selected_records,
                reference_instance=reference_instance,
                reference_object=reference_object,
                candidate_objects=candidate_objects,
            )

            comparisons = []
            for candidate in candidate_objects:
                candidate_points = _concat_points(points_by_object.get(candidate) or [])
                if candidate_points is None:
                    comparisons.append(
                        {
                            "candidate_object": candidate,
                            "status": "error",
                            "error": "No finite point cloud for candidate object.",
                            "distance_meters": None,
                        }
                    )
                    continue
                distances = np.linalg.norm(reference_points[:, None, :] - candidate_points[None, :, :], axis=2).reshape(-1)
                distances = distances[np.isfinite(distances)]
                if distances.size == 0:
                    comparisons.append(
                        {
                            "candidate_object": candidate,
                            "status": "error",
                            "error": "No finite pairwise distances.",
                            "distance_meters": None,
                        }
                    )
                    continue
                aggregate_policy = _aggregate_policy(distances, aggregate)
                value = aggregate_policy["value"]
                comparisons.append(
                    {
                        "candidate_object": candidate,
                        "status": "success",
                        "error": None,
                        "distance_meters": float(value),
                        "selected_aggregate": aggregate_policy["selected_aggregate"],
                        "aggregate_policy": aggregate_policy,
                        "distance_stats": _distance_stats(distances),
                        "pointcloud_sizes": [int(len(reference_points)), int(len(candidate_points))],
                    }
                )

            valid = [item for item in comparisons if isinstance(item.get("distance_meters"), (int, float))]
            if not valid:
                return self.error(
                    "No candidate object produced a valid shared 3D distance.",
                    payload={"reference_object": reference_object, "comparisons": comparisons, "frames": frame_summaries},
                )
            selected = min(valid, key=lambda item: float(item["distance_meters"]))
            if mode == "farthest":
                selected = max(valid, key=lambda item: float(item["distance_meters"]))

            if mode == "closest" and contact_shortcut:
                contact_shortcut = _validate_bbox_contact_with_3d(contact_shortcut, valid)
                if contact_shortcut.get("applied"):
                    contact_selected = next(
                        item for item in valid if item.get("candidate_object") == contact_shortcut["candidate_object"]
                    )
                    selected = {**contact_selected, "selection_reason": "bbox_contact_3d_supported"}
                    comparisons = [
                        {**item, "selection_reason": "bbox_contact_3d_supported"}
                        if item.get("candidate_object") == selected["candidate_object"]
                        else item
                        for item in comparisons
                    ]

            compare_artifacts = [artifact for item in frame_records for artifact in item.get("artifacts", [])]
            compare_overlay = _save_compare_detection_overlay(
                frame_records=frame_records,
                selected_positions=[int(record["frame_position"]) for record in selected_records],
                output_path=artifact_dir_for_tool(self.config, self.name) / "compare_selected_bboxes.png",
            )
            if compare_overlay:
                compare_artifacts.append(compare_overlay)

            return self.success(
                payload={
                    "reference_object": reference_object,
                    "mode": mode,
                    "selected_object": selected["candidate_object"],
                    "selected_distance_meters": float(selected["distance_meters"]),
                    "comparisons": comparisons,
                    "selected_frame_positions": [int(record["frame_position"]) for record in selected_records],
                    "reference_selection": {
                        "strategy": "highest_confidence_single_instance",
                        "frame_position": int(reference_instance["frame_position"]),
                        "score": float(reference_instance["score"]),
                        "mask_pixels": int(reference_instance["mask_pixels"]),
                        "pointcloud_size": int(len(reference_points)),
                    },
                    "shortcut": contact_shortcut,
                    "frames": frame_summaries,
                    "frame_results": frame_records,
                    "aggregate": aggregate,
                    "backend": "shared_grounding_dino+sam2+vggt_object_pointcloud",
                },
                artifacts=compare_artifacts,
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"Shared object distance comparison failed: {exc}")

    def _localize_frames(
        self,
        image_paths: Sequence[str],
        objects: Sequence[str],
        reference_object: str,
        candidate_objects: Sequence[str],
    ) -> List[Dict[str, Any]]:
        records = []
        for index, image_path in enumerate(image_paths):
            localization = self.localizer.invoke(image=image_path, objects=list(objects))
            record: Dict[str, Any] = {
                "frame_position": index,
                "image": image_path,
                "localization_status": localization.get("status"),
                "artifacts": localization.get("artifacts") or [],
            }
            if localization.get("status") != "success":
                record.update({"status": "error", "error": localization.get("error")})
                records.append(record)
                continue
            regions = localization.get("payload", {}).get("regions", [])
            regions_by_object = {object_name: _best_region(regions, object_name) for object_name in objects}
            reference_region = regions_by_object.get(reference_object)
            candidate_quality = {}
            for candidate in candidate_objects:
                candidate_region = regions_by_object.get(candidate)
                if reference_region and candidate_region:
                    candidate_quality[candidate] = min(
                        float(reference_region.get("score") or 0.0),
                        float(candidate_region.get("score") or 0.0),
                    )
            record.update(
                {
                    "regions": regions,
                    "regions_by_object": regions_by_object,
                    "candidate_quality": candidate_quality,
                    "status": "localized" if candidate_quality else "error",
                }
            )
            if not candidate_quality:
                record["error"] = "Failed to localize reference with any candidate."
            records.append(record)
        return records


def _parse_object_pair(kwargs: Dict[str, Any]) -> Tuple[str, str] | None:
    if kwargs.get("object_1") and kwargs.get("object_2"):
        return str(kwargs["object_1"]), str(kwargs["object_2"])
    objects = ensure_object_names(kwargs.get("objects"))
    if len(objects) < 2:
        return None
    return objects[0], objects[1]


def _best_region(regions: Sequence[Dict[str, Any]], object_name: str) -> Dict[str, Any] | None:
    candidates = [region for region in regions if str(region.get("label", "")).lower() == object_name.lower()]
    if not candidates:
        candidates = [
            region
            for region in regions
            if object_name.lower() in str(region.get("label", "")).lower()
            or str(region.get("label", "")).lower() in object_name.lower()
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("score") or 0.0))


def _select_top_same_frame_records(frame_records: Sequence[Dict[str, Any]], top_frames: int) -> List[Dict[str, Any]]:
    localized = [item for item in frame_records if item.get("status") == "localized"]
    localized = sorted(localized, key=lambda item: float(item.get("localization_quality") or 0.0), reverse=True)
    if top_frames <= 0:
        return localized
    return localized[:top_frames]


def _select_shared_compare_records(
    *,
    frame_records: Sequence[Dict[str, Any]],
    reference_object: str,
    candidate_objects: Sequence[str],
    frames_per_candidate: int,
    max_frames: int,
) -> List[Dict[str, Any]]:
    selected: Dict[int, Dict[str, Any]] = {}
    per_candidate = max(1, frames_per_candidate)
    for candidate in candidate_objects:
        candidates = [
            item
            for item in frame_records
            if isinstance((item.get("candidate_quality") or {}).get(candidate), (int, float))
        ]
        candidates = sorted(
            candidates,
            key=lambda item: float((item.get("candidate_quality") or {}).get(candidate) or 0.0),
            reverse=True,
        )
        for item in candidates[:per_candidate]:
            selected[int(item["frame_position"])] = item

    if not selected:
        return []

    selected_items = [selected[position] for position in sorted(selected)]
    if max_frames > 0 and len(selected_items) > max_frames:
        scored = []
        for index, item in enumerate(selected_items):
            qualities = list((item.get("candidate_quality") or {}).values())
            score = max(float(value) for value in qualities) if qualities else 0.0
            coverage = len(qualities)
            scored.append((coverage, score, index, item))
        keep = {index for _coverage, _score, index, _item in sorted(scored, reverse=True)[:max_frames]}
        selected_items = [item for index, item in enumerate(selected_items) if index in keep]

    # Keep frames where the reference object is available; candidate-only frames do not help ranking.
    return [
        item
        for item in selected_items
        if (item.get("regions_by_object") or {}).get(reference_object) is not None
    ]


def _dedupe_objects(objects: Sequence[str]) -> List[str]:
    deduped = []
    seen = set()
    for item in objects:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(str(item).strip())
    return deduped


def _save_compare_detection_overlay(
    *,
    frame_records: Sequence[Dict[str, Any]],
    selected_positions: Sequence[int],
    output_path: Path,
) -> str | None:
    selected = set(selected_positions)
    selected_records = [record for record in frame_records if int(record.get("frame_position", -1)) in selected]
    if not selected_records:
        return None

    images = []
    for record in selected_records:
        regions = [region for region in (record.get("regions_by_object") or {}).values() if region]
        if not regions:
            continue
        image = load_pil_image(record["image"])
        frame_path = Path(record["image"])
        tmp_path = output_path.parent / f"{frame_path.stem}_compare_payload_bbox.png"
        save_bbox_overlay(image, regions, tmp_path)
        images.append(load_pil_image(str(tmp_path)).convert("RGB"))

    if not images:
        return None

    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    grid = Image.new("RGB", (width, height), "white")
    y_offset = 0
    for image in images:
        grid.paste(image, (0, y_offset))
        y_offset += image.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return str(output_path)


def _concat_points(chunks: Sequence[Any]) -> Any | None:
    valid = [chunk for chunk in chunks if len(chunk)]
    if not valid:
        return None
    return np.concatenate(valid, axis=0)


def _select_reference_instance(instances: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    valid = [item for item in instances if item.get("points") is not None and len(item["points"])]
    if not valid:
        return None
    return max(valid, key=lambda item: (float(item.get("score") or 0.0), int(item.get("mask_pixels") or 0)))


def _select_bbox_contact_shortcut(
    *,
    selected_records: Sequence[Dict[str, Any]],
    reference_instance: Dict[str, Any],
    reference_object: str,
    candidate_objects: Sequence[str],
) -> Dict[str, Any] | None:
    reference_frame_position = int(reference_instance["frame_position"])
    record = next((item for item in selected_records if int(item.get("frame_position", -1)) == reference_frame_position), None)
    if not record:
        return None
    regions = record.get("regions_by_object") or {}
    reference_region = regions.get(reference_object)
    if not reference_region:
        return None
    candidates = []
    for candidate in candidate_objects:
        candidate_region = regions.get(candidate)
        if not candidate_region:
            continue
        metrics = _bbox_contact_metrics(reference_region.get("bbox") or [], candidate_region.get("bbox") or [])
        reference_score = float(reference_region.get("score") or 0.0)
        candidate_score = float(candidate_region.get("score") or 0.0)
        if not _is_strong_bbox_contact(metrics, reference_score=reference_score, candidate_score=candidate_score):
            continue
        candidates.append(
            {
                "candidate_object": candidate,
                "reference_frame_position": reference_frame_position,
                "reference_score": reference_score,
                "candidate_score": candidate_score,
                **metrics,
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["iou"]),
            float(item["overlap_min_area_ratio"]),
            -float(item["edge_gap_ratio"]),
            float(item["candidate_score"]),
        ),
    )


def _bbox_contact_metrics(reference_bbox: Sequence[float], candidate_bbox: Sequence[float]) -> Dict[str, float]:
    if len(reference_bbox) < 4 or len(candidate_bbox) < 4:
        return {"iou": 0.0, "overlap_min_area_ratio": 0.0, "edge_gap_ratio": 1.0, "x_overlap_ratio": 0.0, "y_overlap_ratio": 0.0}
    x11, y11, x12, y12 = [float(value) for value in reference_bbox[:4]]
    x21, y21, x22, y22 = [float(value) for value in candidate_bbox[:4]]
    area_1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area_2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    ix = max(0.0, min(x12, x22) - max(x11, x21))
    iy = max(0.0, min(y12, y22) - max(y11, y21))
    intersection = ix * iy
    union = area_1 + area_2 - intersection
    iou = intersection / union if union > 0 else 0.0
    min_area = max(1.0, min(area_1, area_2))
    overlap_min_area_ratio = intersection / min_area
    x_overlap_ratio = ix / max(1.0, min(x12 - x11, x22 - x21))
    y_overlap_ratio = iy / max(1.0, min(y12 - y11, y22 - y21))
    x_gap = max(0.0, max(x11, x21) - min(x12, x22))
    y_gap = max(0.0, max(y11, y21) - min(y12, y22))
    diag = ((max(x12 - x11, x22 - x21) ** 2) + (max(y12 - y11, y22 - y21) ** 2)) ** 0.5
    edge_gap_ratio = ((x_gap**2 + y_gap**2) ** 0.5) / max(1.0, diag)
    return {
        "iou": float(iou),
        "overlap_min_area_ratio": float(overlap_min_area_ratio),
        "edge_gap_ratio": float(edge_gap_ratio),
        "x_overlap_ratio": float(x_overlap_ratio),
        "y_overlap_ratio": float(y_overlap_ratio),
    }


def _is_strong_bbox_contact(metrics: Dict[str, float], *, reference_score: float = 1.0, candidate_score: float = 1.0) -> bool:
    if reference_score < 0.40 or candidate_score < 0.40:
        return False
    if metrics["iou"] >= 0.10 and metrics["overlap_min_area_ratio"] >= 0.18:
        return True
    if metrics["iou"] >= 0.06 and metrics["overlap_min_area_ratio"] >= 0.35:
        return True
    return False


def _validate_bbox_contact_with_3d(contact: Dict[str, Any], comparisons: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidate = next((item for item in comparisons if item.get("candidate_object") == contact.get("candidate_object")), None)
    if not candidate:
        return {**contact, "applied": False, "rejected_reason": "no_3d_distance"}

    candidate_near = _near_contact_distance(candidate)
    if candidate_near is None:
        return {**contact, "applied": False, "rejected_reason": "no_3d_near_distance"}

    other_near_values = [
        value
        for item in comparisons
        if item.get("candidate_object") != contact.get("candidate_object")
        for value in [_near_contact_distance(item)]
        if value is not None
    ]
    best_other_near = min(other_near_values) if other_near_values else None
    aggregate_best = min(comparisons, key=lambda item: float(item["distance_meters"]))
    aggregate_winner = aggregate_best.get("candidate_object") == contact.get("candidate_object")

    absolute_near = candidate_near <= 0.25
    relatively_near = best_other_near is None or candidate_near <= best_other_near + 0.05 or candidate_near <= best_other_near * 0.90
    if aggregate_winner or (absolute_near and relatively_near):
        return {
            **contact,
            "applied": True,
            "candidate_near_distance_meters": float(candidate_near),
            "best_other_near_distance_meters": None if best_other_near is None else float(best_other_near),
            "aggregate_winner": bool(aggregate_winner),
        }

    return {
        **contact,
        "applied": False,
        "rejected_reason": "3d_distance_not_nearest",
        "candidate_near_distance_meters": float(candidate_near),
        "best_other_near_distance_meters": None if best_other_near is None else float(best_other_near),
        "aggregate_winner": bool(aggregate_winner),
    }


def _near_contact_distance(comparison: Dict[str, Any]) -> float | None:
    stats = comparison.get("distance_stats") or {}
    for key in ("p05", "p10", "min"):
        value = stats.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    value = comparison.get("distance_meters")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _duplicate_bbox_objects(
    regions_by_object: Dict[str, Dict[str, Any] | None],
    objects: Sequence[str],
    iou_threshold: float,
) -> set[str]:
    duplicate_objects: set[str] = set()
    if iou_threshold <= 0:
        return duplicate_objects
    for index, object_1 in enumerate(objects):
        region_1 = regions_by_object.get(object_1)
        if not region_1:
            continue
        for object_2 in objects[index + 1 :]:
            region_2 = regions_by_object.get(object_2)
            if not region_2:
                continue
            if _bbox_iou(region_1.get("bbox") or [], region_2.get("bbox") or []) >= iou_threshold:
                duplicate_objects.add(object_1)
                duplicate_objects.add(object_2)
    return duplicate_objects


def _bbox_iou(bbox_1: Sequence[float], bbox_2: Sequence[float]) -> float:
    if len(bbox_1) < 4 or len(bbox_2) < 4:
        return 0.0
    x11, y11, x12, y12 = [float(value) for value in bbox_1[:4]]
    x21, y21, x22, y22 = [float(value) for value in bbox_2[:4]]
    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area_2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    union = area_1 + area_2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _predict_sam_mask(predictor: Any, bbox: Sequence[float]) -> Any:
    masks, scores, _ = predictor.predict(box=np.asarray(bbox, dtype=np.float32), multimask_output=True)
    if len(masks) == 0:
        return np.zeros((0, 0), dtype=bool)
    return masks[int(np.argmax(scores))].astype(bool)


def _mask_pixels_to_3d(
    *,
    mask: Any,
    image_size: Tuple[int, int],
    frame_points: Any,
    preprocess_mode: str,
    max_points: int,
) -> Any:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=float)
    if xs.size > max_points:
        indices = np.linspace(0, xs.size - 1, max_points).round().astype(int)
        xs = xs[indices]
        ys = ys[indices]
    h, w = frame_points.shape[:2]
    sampled = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        mapped_x, mapped_y = map_point_to_preprocessed(
            (float(x), float(y)),
            image_size,
            mode=preprocess_mode,
            target_size=max(h, w),
        )
        px = int(round(max(0.0, min(float(w - 1), mapped_x))))
        py = int(round(max(0.0, min(float(h - 1), mapped_y))))
        point = frame_points[py, px]
        if np.all(np.isfinite(point[:3])):
            sampled.append([float(point[0]), float(point[1]), float(point[2])])
    if not sampled:
        return np.zeros((0, 3), dtype=float)
    points = np.asarray(sampled, dtype=float)
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center[None, :], axis=1)
    keep = radii <= np.quantile(radii, 0.90)
    return points[keep]


def _extract_world_points_sequence(predictions: Dict[str, Any]) -> Any | None:
    for key in ("world_points", "world_points_cam", "points3D", "points_3d"):
        if key not in predictions:
            continue
        points = predictions[key]
        if hasattr(points, "detach"):
            points = points.detach().cpu().numpy()
        if points.ndim == 5:
            return points[0]
        if points.ndim == 4:
            return points
    return None


def _aggregate_distances(distances: Any, mode: str) -> float:
    if mode == "min":
        return float(np.min(distances))
    if mode == "p05":
        return float(np.quantile(distances, 0.05))
    if mode == "p10":
        return float(np.quantile(distances, 0.10))
    if mode == "p25":
        return float(np.quantile(distances, 0.25))
    if mode == "median":
        return float(np.median(distances))
    if mode == "p75":
        return float(np.quantile(distances, 0.75))
    if mode == "p80":
        return float(np.quantile(distances, 0.80))
    if mode == "p85":
        return float(np.quantile(distances, 0.85))
    if mode == "p90":
        return float(np.quantile(distances, 0.90))
    if mode == "iqm_75_90":
        p75 = float(np.quantile(distances, 0.75))
        p90 = float(np.quantile(distances, 0.90))
        trimmed = distances[(distances >= p75) & (distances <= p90)]
        return float(np.mean(trimmed)) if trimmed.size else float((p75 + p90) / 2.0)
    if mode == "blend_p75_p90_25":
        p75 = float(np.quantile(distances, 0.75))
        p90 = float(np.quantile(distances, 0.90))
        return float(0.25 * p90 + 0.75 * p75)
    if mode == "blend_p75_p90_50":
        p75 = float(np.quantile(distances, 0.75))
        p90 = float(np.quantile(distances, 0.90))
        return float(0.50 * p90 + 0.50 * p75)
    if mode == "blend_p75_p90_75":
        p75 = float(np.quantile(distances, 0.75))
        p90 = float(np.quantile(distances, 0.90))
        return float(0.75 * p90 + 0.25 * p75)
    if mode == "adaptive_p05_p90":
        policy = _adaptive_p05_p90_policy(distances)
        return float(policy["value"])
    raise ValueError(f"Unknown pointcloud aggregate: {mode}")


def _adaptive_p05_p90_policy(distances: Any) -> Dict[str, Any]:
    p05 = float(np.quantile(distances, 0.05))
    p90 = float(np.quantile(distances, 0.90))
    ratio = p90 / max(p05, 1e-9)
    spread = p90 - p05
    selected = "p05" if ratio <= ADAPTIVE_P05_P90_RATIO_MAX and spread >= ADAPTIVE_P05_P90_SPREAD_MIN else "p90"
    return {
        "strategy": "adaptive_p05_p90",
        "selected_aggregate": selected,
        "value": p05 if selected == "p05" else p90,
        "p05": p05,
        "p90": p90,
        "ratio": ratio,
        "spread": spread,
        "ratio_max": ADAPTIVE_P05_P90_RATIO_MAX,
        "spread_min": ADAPTIVE_P05_P90_SPREAD_MIN,
    }


def _aggregate_policy(distances: Any, mode: str) -> Dict[str, Any]:
    if mode == "adaptive_p05_p90":
        return _adaptive_p05_p90_policy(distances)
    return {
        "strategy": "fixed",
        "selected_aggregate": mode,
        "value": _aggregate_distances(distances, mode),
    }


def _distance_stats(distances: Any) -> Dict[str, float]:
    return {
        "min": float(np.min(distances)),
        "p05": float(np.quantile(distances, 0.05)),
        "p10": float(np.quantile(distances, 0.10)),
        "p25": float(np.quantile(distances, 0.25)),
        "median": float(np.median(distances)),
        "p75": float(np.quantile(distances, 0.75)),
        "p80": float(np.quantile(distances, 0.80)),
        "p85": float(np.quantile(distances, 0.85)),
        "p90": float(np.quantile(distances, 0.90)),
    }
