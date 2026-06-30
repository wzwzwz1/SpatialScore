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
    resolve_device,
    save_bbox_overlay,
)
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.localization import LocalizeObjectsTool
from spatial_agent.tools.object_distance_3d import (
    _aggregate_policy,
    _best_region,
    _concat_points,
    _extract_world_points_sequence,
    _mask_pixels_to_3d,
    _predict_sam_mask,
)


class EstimateObjectSize3DTool(BaseSpatialTool):
    name = "EstimateObjectSize3D"
    description = (
        "Estimate the longest physical dimension of one object in a video, in centimeters, "
        "using multi-frame localization, SAM2 masks, VGGT reconstruction, and robust object point-cloud extents."
    )
    args_schema = {
        "type": "object",
        "properties": {
            "images": {"type": "array", "items": {"type": "string"}},
            "object": {"type": "string"},
            "objects": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
        },
        "required": ["images"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config
        self.localizer = LocalizeObjectsTool(config)

    def invoke(self, **kwargs) -> Dict[str, Any]:
        image_paths = ensure_image_paths(kwargs.get("images") or kwargs.get("image"))
        object_name = _parse_object_name(kwargs)
        if not image_paths:
            return self.error("EstimateObjectSize3D requires video frame images.")
        if not object_name:
            return self.error("EstimateObjectSize3D requires an object name.")
        if np is None:
            return self.unavailable("NumPy is required for EstimateObjectSize3D.")

        settings = get_tool_settings(
            self.config,
            self.name,
            aliases=["object_size_3d", "object_size_estimation", "size"],
        )
        camera_settings = get_tool_settings(self.config, "GetCameraParametersVGGT", aliases=["camera", "vggt"])
        mask_settings = get_tool_settings(self.config, "GetObjectMask", aliases=["mask", "sam2"])
        top_frames = int(settings.get("top_size_frames", 6))
        max_points = int(settings.get("mask_max_points", 256))
        min_mask_pixels = int(settings.get("min_mask_pixels", 1000))
        aggregate = str(settings.get("size_aggregate", "p90"))
        extent_policy = str(settings.get("extent_policy", "p05_p95"))
        lower_q, upper_q = _parse_extent_policy(extent_policy)
        preprocess_mode = str(settings.get("preprocess_mode") or camera_settings.get("preprocess_mode", "pad"))
        device = resolve_device(settings.get("device") or camera_settings.get("device") or mask_settings.get("device"))

        try:  # pragma: no cover - dependency-heavy runtime path
            frame_records = self._localize_frames(image_paths, object_name)
            selected_records = _select_size_records(frame_records, top_frames)
            if not selected_records:
                return self.error(
                    "No sampled frame localized the target object.",
                    payload={"object": object_name, "frame_results": frame_records},
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

            point_chunks = []
            frame_summaries = []
            per_frame_size_cm = []
            for local_index, record in enumerate(selected_records):
                image_path = image_paths[int(record["frame_position"])]
                image = load_pil_image(image_path)
                predictor.set_image(load_rgb_array(image_path))
                region = record["region"]
                mask = _predict_sam_mask(predictor, region["bbox"])
                mask_pixels = int(mask.sum())
                summary: Dict[str, Any] = {
                    "frame_position": int(record["frame_position"]),
                    "image": image_path,
                    "score": region.get("score"),
                    "bbox": region.get("bbox"),
                    "bbox_area_ratio": record.get("bbox_area_ratio"),
                    "edge_contact_ratio": record.get("edge_contact_ratio"),
                    "mask_pixels": mask_pixels,
                }
                if mask_pixels < min_mask_pixels:
                    summary.update({"status": "small_mask", "pointcloud_size": 0})
                    frame_summaries.append(summary)
                    continue

                points = _mask_pixels_to_3d(
                    mask=mask,
                    image_size=image.size,
                    frame_points=world_points[local_index],
                    preprocess_mode=preprocess_mode,
                    max_points=max_points,
                )
                summary["pointcloud_size"] = int(len(points))
                if len(points):
                    extent = _robust_extent(points, lower_q=lower_q, upper_q=upper_q)
                    max_extent_cm = float(extent["max_extent_m"] * 100.0)
                    summary.update(
                        {
                            "status": "success",
                            "extent_m": extent,
                            "max_extent_cm": max_extent_cm,
                        }
                    )
                    per_frame_size_cm.append(max_extent_cm)
                    point_chunks.append(points)
                else:
                    summary["status"] = "empty_pointcloud"
                frame_summaries.append(summary)

            if not per_frame_size_cm:
                return self.error(
                    "No finite object point cloud was extracted from selected frames.",
                    payload={"object": object_name, "frames": frame_summaries, "frame_results": frame_records},
                )

            size_values = np.asarray(per_frame_size_cm, dtype=float)
            aggregate_policy = _aggregate_policy(size_values, aggregate)
            combined_points = _concat_points(point_chunks)
            combined_extent = (
                _robust_extent(combined_points, lower_q=lower_q, upper_q=upper_q) if combined_points is not None else None
            )
            artifacts = [artifact for item in frame_records for artifact in item.get("artifacts", [])]
            overlay = _save_size_detection_overlay(
                frame_records=frame_records,
                selected_positions=[int(record["frame_position"]) for record in selected_records],
                output_path=artifact_dir_for_tool(self.config, self.name) / "size_selected_bboxes.png",
            )
            if overlay:
                artifacts.append(overlay)

            return self.success(
                payload={
                    "object": object_name,
                    "size_centimeters": float(aggregate_policy["value"]),
                    "unit": "centimeters",
                    "target_dimension": "longest_dimension",
                    "method": "mask_pointcloud_multiframe_extent",
                    "selected_frame_positions": [int(record["frame_position"]) for record in selected_records],
                    "aggregate": aggregate,
                    "selected_aggregate": aggregate_policy["selected_aggregate"],
                    "aggregate_policy": aggregate_policy,
                    "extent_policy": extent_policy,
                    "per_frame_size_cm": per_frame_size_cm,
                    "combined_extent_m": combined_extent,
                    "frames": frame_summaries,
                    "frame_results": frame_records,
                    "backend": "grounding_dino+sam2+vggt_object_extent",
                },
                artifacts=artifacts,
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"Object 3D size estimation failed: {exc}")

    def _localize_frames(self, image_paths: Sequence[str], object_name: str) -> List[Dict[str, Any]]:
        records = []
        for index, image_path in enumerate(image_paths):
            localization = self.localizer.invoke(image=image_path, objects=[object_name])
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
            region = _best_region(regions, object_name)
            record.update({"region": region, "regions": regions})
            if not region:
                record.update({"status": "error", "error": "Failed to localize target object."})
            else:
                _attach_bbox_quality(record, region, image_path)
                quality = _size_frame_quality(record)
                record.update({"status": "localized", "localization_quality": quality})
            records.append(record)
        return records


def _parse_object_name(kwargs: Dict[str, Any]) -> str | None:
    if kwargs.get("object"):
        return str(kwargs["object"]).strip()
    objects = ensure_object_names(kwargs.get("objects"))
    if not objects:
        return None
    return objects[0]


def _parse_extent_policy(policy: str) -> Tuple[float, float]:
    normalized = policy.strip().lower()
    if normalized == "p02_p98":
        return 0.02, 0.98
    if normalized == "p05_p95":
        return 0.05, 0.95
    if normalized == "p10_p90":
        return 0.10, 0.90
    raise ValueError(f"Unsupported extent_policy: {policy}")


def _attach_bbox_quality(record: Dict[str, Any], region: Dict[str, Any], image_path: str) -> None:
    try:
        image = load_pil_image(image_path)
        width, height = image.size
    except Exception:
        width, height = 1, 1
    x1, y1, x2, y2 = [float(value) for value in (region.get("bbox") or [0, 0, 0, 0])[:4]]
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)
    area_ratio = (bbox_width * bbox_height) / max(1.0, float(width * height))
    edge_margin = min(max(0.0, x1), max(0.0, y1), max(0.0, width - x2), max(0.0, height - y2))
    edge_contact_ratio = 1.0 - min(1.0, edge_margin / max(1.0, min(width, height) * 0.08))
    record["bbox_area_ratio"] = float(area_ratio)
    record["edge_contact_ratio"] = float(edge_contact_ratio)


def _size_frame_quality(record: Dict[str, Any]) -> float:
    region = record.get("region") or {}
    score = float(region.get("score") or 0.0)
    area_bonus = min(1.0, float(record.get("bbox_area_ratio") or 0.0) * 8.0)
    edge_penalty = 0.25 * float(record.get("edge_contact_ratio") or 0.0)
    return float(score + area_bonus - edge_penalty)


def _select_size_records(frame_records: Sequence[Dict[str, Any]], top_frames: int) -> List[Dict[str, Any]]:
    localized = [item for item in frame_records if item.get("status") == "localized" and item.get("region")]
    localized = sorted(
        localized,
        key=lambda item: (
            float(item.get("localization_quality") or 0.0),
            float(item.get("bbox_area_ratio") or 0.0),
            float((item.get("region") or {}).get("score") or 0.0),
        ),
        reverse=True,
    )
    if top_frames <= 0:
        return localized
    return localized[:top_frames]


def _robust_extent(points: Any, *, lower_q: float, upper_q: float) -> Dict[str, Any]:
    lower = np.quantile(points, lower_q, axis=0)
    upper = np.quantile(points, upper_q, axis=0)
    extent = np.maximum(0.0, upper - lower)
    return {
        "lower_quantile": float(lower_q),
        "upper_quantile": float(upper_q),
        "axis_extent_m": [float(value) for value in extent.tolist()],
        "max_extent_m": float(np.max(extent)),
        "point_count": int(len(points)),
    }


def _save_size_detection_overlay(
    *,
    frame_records: Sequence[Dict[str, Any]],
    selected_positions: Sequence[int],
    output_path: Path,
) -> str | None:
    selected = set(selected_positions)
    images = []
    for record in frame_records:
        if int(record.get("frame_position", -1)) not in selected:
            continue
        region = record.get("region")
        if not region:
            continue
        image = load_pil_image(record["image"])
        tmp_path = output_path.parent / f"{Path(record['image']).stem}_size_bbox.png"
        save_bbox_overlay(image, [region], tmp_path)
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
