from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - optional runtime dependency
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    build_depth_model,
    clamp_bbox,
    ensure_image_paths,
    ensure_object_names,
    get_tool_settings,
)
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.localization import LocalizeObjectsTool


class EstimateObjectDepthTool(BaseSpatialTool):
    name = "EstimateObjectDepth"
    description = "Estimate object depth values for one or more named objects."
    args_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string"},
            "objects": {"type": "array", "items": {"type": "string"}},
            "indoor_or_outdoor": {"type": "string"},
        },
        "required": ["image", "objects"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config
        self.localizer = LocalizeObjectsTool(config)

    def invoke(self, **kwargs):
        image_paths = ensure_image_paths(kwargs.get("image"))
        objects = ensure_object_names(kwargs.get("objects"))
        if not image_paths:
            return self.error("EstimateObjectDepth requires an image path.")
        if not objects:
            return self.error("EstimateObjectDepth requires one or more object names.")

        settings = get_tool_settings(self.config, self.name, aliases=["depth"])
        try:  # pragma: no cover - dependency-heavy runtime path
            import cv2
            from PIL import Image

            image_path = image_paths[0]
            raw_bgr = cv2.imread(image_path)
            if raw_bgr is None:
                return self.error(f"Failed to read image for depth estimation: {image_path}")

            model, _device = build_depth_model(settings)
            input_size = int(settings.get("input_size", 518))
            depth_map = model.infer_image(raw_bgr, input_size)

            localization = self.localizer.invoke(image=image_path, objects=objects)
            if localization["status"] != "success":
                return self.error("Depth estimation depends on object localization.", payload={"localization": localization})

            regions = localization["payload"].get("regions", [])
            image_height, image_width = depth_map.shape[:2]
            by_label: Dict[str, List[Dict[str, object]]] = {}
            for region in regions:
                by_label.setdefault(str(region["label"]).lower(), []).append(region)

            results = []
            for object_name in objects:
                candidates = by_label.get(object_name.lower(), [])
                if not candidates:
                    for label, label_regions in by_label.items():
                        if object_name.lower() in label or label in object_name.lower():
                            candidates.extend(label_regions)
                if not candidates:
                    results.append({"object": object_name, "depth": None, "bbox": None, "error": "Object was not localized."})
                    continue

                region = max(candidates, key=lambda item: float(item.get("score", 0.0)))
                bbox = clamp_bbox(region["bbox"], image_width, image_height)
                x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                roi = depth_map[y1 : max(y1 + 1, y2 + 1), x1 : max(x1 + 1, x2 + 1)]
                if roi.size == 0:
                    results.append({"object": object_name, "depth": None, "bbox": bbox, "error": "Localized region is empty."})
                    continue

                depth_value = float(np.median(roi))
                results.append(
                    {
                        "object": object_name,
                        "depth": depth_value,
                        "bbox": bbox,
                        "score": float(region.get("score", 0.0)),
                        "error": None,
                    }
                )

            normalized = depth_map - float(depth_map.min())
            if float(normalized.max()) > 0:
                normalized = normalized / float(normalized.max())
            artifact_path = artifact_dir_for_tool(self.config, self.name) / f"{Path(image_path).stem}_depth.png"
            Image.fromarray((normalized * 255).astype(np.uint8)).save(artifact_path)

            return self.success(
                payload={
                    "results": results,
                    "scene_type_hint": kwargs.get("indoor_or_outdoor"),
                    "backend": f"depth_anything_v2_metric:{settings.get('encoder', 'vitl')}",
                },
                artifacts=[str(artifact_path)],
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"DepthAnything depth estimation is not available or failed to initialize: {exc}")
