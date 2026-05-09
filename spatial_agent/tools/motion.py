from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - optional runtime dependency
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    bbox_center,
    ensure_image_paths,
    ensure_object_names,
    get_tool_settings,
    get_vggt_backend,
    load_pil_image,
    map_point_to_preprocessed,
    resolve_device,
    save_track_visualization,
    unmap_point_from_preprocessed,
)
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.localization import LocalizeObjectsTool


def _motion_direction(dx: float, dy: float, threshold: float = 5.0) -> str:
    horizontal = ""
    vertical = ""
    if abs(dx) > threshold:
        horizontal = "right" if dx > 0 else "left"
    if abs(dy) > threshold:
        vertical = "down" if dy > 0 else "up"
    if horizontal and vertical:
        return f"{vertical}-{horizontal}"
    return horizontal or vertical or "stationary"


class EstimateObjectMotionTool(BaseSpatialTool):
    name = "EstimateObjectMotion"
    description = "Estimate object motion across frames."
    args_schema = {
        "type": "object",
        "properties": {
            "images": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "objects": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
        },
        "required": ["images", "objects"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config
        self.localizer = LocalizeObjectsTool(config)

    def invoke(self, **kwargs):
        image_paths = ensure_image_paths(kwargs.get("images") or kwargs.get("image"))
        objects = ensure_object_names(kwargs.get("objects"))
        if len(image_paths) < 2:
            return self.error("EstimateObjectMotion requires at least two image paths.")
        if not objects:
            return self.error("EstimateObjectMotion requires one or more object names.")

        settings = get_tool_settings(self.config, self.name, aliases=["motion", "vggt_motion"])
        preprocess_mode = str(settings.get("preprocess_mode", "pad"))
        try:  # pragma: no cover - dependency-heavy runtime path
            device = resolve_device(settings.get("device"))
            backend = get_vggt_backend(
                model_id=str(settings.get("hf_model_id", "facebook/VGGT-1B")),
                checkpoint_path=settings.get("checkpoint_path"),
                device=device,
            )
            load_and_preprocess_images = backend["load_and_preprocess_images"]
            torch = backend["torch"]
            model = backend["model"]

            first_image = load_pil_image(image_paths[0])
            localization = self.localizer.invoke(image=image_paths[0], objects=objects)
            if localization["status"] != "success":
                return self.error("Motion estimation depends on object localization in the first frame.", payload={"localization": localization})

            regions = localization["payload"].get("regions", [])
            by_label: Dict[str, List[Dict[str, object]]] = {}
            for region in regions:
                by_label.setdefault(str(region["label"]).lower(), []).append(region)

            query_points = []
            tracked_objects = []
            for object_name in objects:
                candidates = by_label.get(object_name.lower(), [])
                if not candidates:
                    for label, label_regions in by_label.items():
                        if object_name.lower() in label or label in object_name.lower():
                            candidates.extend(label_regions)
                if not candidates:
                    continue
                region = max(candidates, key=lambda item: float(item.get("score", 0.0)))
                center = bbox_center(region["bbox"])
                mapped_center = map_point_to_preprocessed(center, first_image.size, mode=preprocess_mode)
                query_points.append(mapped_center)
                tracked_objects.append({"object": object_name, "bbox": region["bbox"]})

            if not query_points:
                return self.error("No objects could be localized for motion tracking.", payload={"localization": localization["payload"]})

            images = load_and_preprocess_images(image_paths, mode=preprocess_mode).to(device)
            query_tensor = torch.tensor(query_points, dtype=torch.float32, device=device)
            with torch.no_grad():
                predictions = model(images, query_points=query_tensor)

            tracks = predictions["track"][0].detach().cpu().numpy()
            vis = predictions.get("vis")
            conf = predictions.get("conf")

            results = []
            track_points_original = np.zeros_like(tracks)
            for frame_index, image_path in enumerate(image_paths):
                frame_size = load_pil_image(image_path).size
                for track_index in range(tracks.shape[1]):
                    track_points_original[frame_index, track_index] = unmap_point_from_preprocessed(
                        tuple(float(value) for value in tracks[frame_index, track_index]),
                        frame_size,
                        mode=preprocess_mode,
                    )

            for track_index, tracked in enumerate(tracked_objects):
                object_track = track_points_original[:, track_index, :]
                start = object_track[0]
                end = object_track[-1]
                dx = float(end[0] - start[0])
                dy = float(end[1] - start[1])
                displacement = float(np.linalg.norm(end - start))
                results.append(
                    {
                        "object": tracked["object"],
                        "start_point": [float(start[0]), float(start[1])],
                        "end_point": [float(end[0]), float(end[1])],
                        "delta_xy": [dx, dy],
                        "displacement": displacement,
                        "direction": _motion_direction(dx, dy),
                        "trajectory": object_track.tolist(),
                        "visibility_mean": float(vis[0, :, track_index].mean().detach().cpu()) if vis is not None else None,
                        "confidence_mean": float(conf[0, :, track_index].mean().detach().cpu()) if conf is not None else None,
                        "bbox": tracked["bbox"],
                    }
                )

            artifact_path = artifact_dir_for_tool(self.config, self.name) / f"{Path(image_paths[0]).stem}_motion.png"
            artifact = save_track_visualization(image_paths, track_points_original, artifact_path)
            return self.success(
                payload={
                    "results": results,
                    "backend": f"vggt_track:{settings.get('hf_model_id', 'facebook/VGGT-1B')}",
                },
                artifacts=[artifact],
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"VGGT motion tracking is not available or failed to initialize: {exc}")
