from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

try:  # pragma: no cover - optional runtime dependency
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

from PIL import ImageDraw

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    ensure_image_paths,
    get_tool_settings,
    get_vggt_backend,
    load_pil_image,
    map_point_to_preprocessed,
    resolve_device,
)
from spatial_agent.tools.base import BaseSpatialTool


class Get3DDistanceTool(BaseSpatialTool):
    name = "Get3DDistance"
    description = (
        "Calculate the absolute 3D spatial distance in meters between two pixel points "
        "in an image using VGGT reconstruction."
    )
    args_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string"},
            "point_1": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "point_2": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        },
        "required": ["image", "point_1", "point_2"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config

    def invoke(self, **kwargs) -> Dict[str, Any]:
        image_paths = ensure_image_paths(kwargs.get("image") or kwargs.get("images"))
        if not image_paths:
            return self.error("Get3DDistance requires an image path.")
        try:
            point_1 = _parse_point(kwargs.get("point_1"), "point_1")
            point_2 = _parse_point(kwargs.get("point_2"), "point_2")
        except ValueError as exc:
            return self.error(str(exc))

        settings = get_tool_settings(self.config, self.name, aliases=["distance", "3d_distance"])
        camera_settings = get_tool_settings(self.config, "GetCameraParametersVGGT", aliases=["camera", "vggt"])
        preprocess_mode = str(settings.get("preprocess_mode") or camera_settings.get("preprocess_mode", "pad"))
        neighbor_radius = int(settings.get("neighbor_search_radius", 3))

        try:  # pragma: no cover - dependency-heavy runtime path
            if np is None:
                return self.unavailable("NumPy is required for Get3DDistance.")

            image_path = image_paths[0]
            image = load_pil_image(image_path)
            device = resolve_device(settings.get("device") or camera_settings.get("device"))
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
            torch = backend["torch"]
            images = backend["load_and_preprocess_images"]([image_path], mode=preprocess_mode).to(device)
            with torch.no_grad():
                predictions = backend["model"](images)

            frame_points = _single_frame_world_points(predictions)
            if frame_points is None:
                return self.error(
                    "VGGT predictions did not include world_points/world_points_cam; cannot calculate 3D distance.",
                    payload={"prediction_keys": sorted(str(key) for key in predictions.keys())},
                )

            point_1_3d = _lookup_point_3d(
                frame_points=frame_points,
                point_xy=point_1,
                image_size=image.size,
                preprocess_mode=preprocess_mode,
                neighbor_radius=neighbor_radius,
            )
            point_2_3d = _lookup_point_3d(
                frame_points=frame_points,
                point_xy=point_2,
                image_size=image.size,
                preprocess_mode=preprocess_mode,
                neighbor_radius=neighbor_radius,
            )
            if point_1_3d is None or point_2_3d is None:
                return self.error(
                    "Could not find finite reconstructed 3D points near both input pixels.",
                    payload={"point_1_3d": point_1_3d, "point_2_3d": point_2_3d},
                )

            distance_meters = math.sqrt(sum((a - b) ** 2 for a, b in zip(point_1_3d, point_2_3d)))
            artifact_path = artifact_dir_for_tool(self.config, self.name) / f"{Path(image_path).stem}_3d_distance.png"
            _save_distance_overlay(image, point_1, point_2, artifact_path)

            return self.success(
                payload={
                    "distance_meters": float(distance_meters),
                    "unit": "meters",
                    "point_1": [float(point_1[0]), float(point_1[1])],
                    "point_2": [float(point_2[0]), float(point_2[1])],
                    "point_1_3d": point_1_3d,
                    "point_2_3d": point_2_3d,
                    "backend": "vggt",
                    "preprocess_mode": preprocess_mode,
                    "artifact_descriptions": [
                        {
                            "path": str(artifact_path),
                            "kind": "distance_point_overlay",
                            "description": "Input image with the two distance query points and connecting line.",
                        }
                    ],
                },
                artifacts=[str(artifact_path)],
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"VGGT 3D distance estimation is not available or failed to run: {exc}")


def _parse_point(value: Any, name: str) -> Tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError(f"{name} must be a [x, y] pixel coordinate.")
    return float(value[0]), float(value[1])


def _single_frame_world_points(predictions: Dict[str, Any]) -> Any | None:
    for key in ("world_points", "world_points_cam", "points3D", "points_3d"):
        if key not in predictions:
            continue
        points = _to_numpy(predictions[key])
        if points.ndim == 5:
            return points[0, 0]
        if points.ndim == 4:
            return points[0]
        if points.ndim == 3:
            return points
    return None


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _lookup_point_3d(
    *,
    frame_points: Any,
    point_xy: Tuple[float, float],
    image_size: Tuple[int, int],
    preprocess_mode: str,
    neighbor_radius: int,
) -> list[float] | None:
    h, w = frame_points.shape[:2]
    mapped_x, mapped_y = map_point_to_preprocessed(
        point_xy,
        image_size,
        mode=preprocess_mode,
        target_size=max(h, w),
    )
    px = int(round(max(0.0, min(float(w - 1), mapped_x))))
    py = int(round(max(0.0, min(float(h - 1), mapped_y))))

    candidates = [(0, px, py)]
    radius = max(0, neighbor_radius)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = px + dx
            y = py + dy
            if x < 0 or y < 0 or x >= w or y >= h or (x == px and y == py):
                continue
            candidates.append((dx * dx + dy * dy, x, y))
    candidates.sort(key=lambda item: item[0])

    for _distance, x, y in candidates:
        point = frame_points[y, x]
        if np.all(np.isfinite(point[:3])):
            return [float(point[0]), float(point[1]), float(point[2])]
    return None


def _save_distance_overlay(image: Any, point_1: Tuple[float, float], point_2: Tuple[float, float], output_path: Path) -> str:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    p1 = (float(point_1[0]), float(point_1[1]))
    p2 = (float(point_2[0]), float(point_2[1]))
    draw.line((p1, p2), fill=(255, 210, 90), width=3)
    for label, point, color in [
        ("P1", p1, (255, 90, 90)),
        ("P2", p2, (90, 200, 255)),
    ]:
        x, y = point
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="black", width=1)
        draw.text((x + 8, y - 8), label, fill=color)
    canvas.save(output_path)
    return str(output_path)
