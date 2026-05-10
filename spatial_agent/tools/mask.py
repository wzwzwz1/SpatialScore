from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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
    load_pil_image,
    load_rgb_array,
    resolve_device,
    save_mask_overlay,
)
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.localization import LocalizeObjectsTool


class GetObjectMaskTool(BaseSpatialTool):
    name = "GetObjectMask"
    description = "Estimate segmentation masks and areas for named objects."
    args_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string"},
            "objects": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["objects"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config
        self.localizer = LocalizeObjectsTool(config)

    def invoke(self, **kwargs):
        image_paths = ensure_image_paths(kwargs.get("image"))
        objects = ensure_object_names(kwargs.get("objects"))
        if not image_paths:
            return self.error("GetObjectMask requires an image path.")
        if not objects:
            return self.error("GetObjectMask requires one or more object names.")

        settings = get_tool_settings(self.config, self.name, aliases=["mask", "sam2"])
        try:  # pragma: no cover - dependency-heavy runtime path
            device = resolve_device(settings.get("device"))
            predictor = get_sam2_predictor(
                model_id=str(settings.get("model_id", "facebook/sam2.1-hiera-large")),
                checkpoint_path=settings.get("checkpoint_path"),
                config_path=settings.get("config_path"),
                device=device,
            )

            image_path = image_paths[0]
            image_rgb = load_rgb_array(image_path)
            image_pil = load_pil_image(image_path)
            predictor.set_image(image_rgb)

            localization = self.localizer.invoke(image=image_path, objects=objects)
            if localization["status"] != "success":
                return self.error("Mask estimation depends on object localization.", payload={"localization": localization})

            regions = localization["payload"].get("regions", [])
            by_label: Dict[str, List[Dict[str, object]]] = {}
            for region in regions:
                by_label.setdefault(str(region["label"]).lower(), []).append(region)

            results = []
            artifacts: List[str] = []
            for object_name in objects:
                candidates = by_label.get(object_name.lower(), [])
                if not candidates:
                    for label, label_regions in by_label.items():
                        if object_name.lower() in label or label in object_name.lower():
                            candidates.extend(label_regions)
                if not candidates:
                    results.append({"object": object_name, "mask_area": None, "bbox": None, "error": "Object was not localized."})
                    continue

                region = max(candidates, key=lambda item: float(item.get("score", 0.0)))
                box = np.asarray(region["bbox"], dtype=np.float32)
                masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                best_index = int(np.argmax(scores))
                mask = masks[best_index].astype(bool)
                ys, xs = np.where(mask)
                if xs.size == 0 or ys.size == 0:
                    results.append({"object": object_name, "mask_area": 0.0, "bbox": region["bbox"], "error": "SAM2 returned an empty mask."})
                    continue

                bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                mask_area_fraction = float(mask.mean())
                artifact_path = artifact_dir_for_tool(self.config, self.name) / f"{Path(image_path).stem}_{object_name.replace(' ', '_')}_mask.png"
                artifacts.append(save_mask_overlay(image_pil, mask, bbox, artifact_path))

                results.append(
                    {
                        "object": object_name,
                        "mask_area": mask_area_fraction,
                        "mask_area_pixels": int(mask.sum()),
                        "bbox": bbox,
                        "score": float(scores[best_index]),
                        "error": None,
                    }
                )

            return self.success(
                payload={
                    "results": results,
                    "backend": f"sam2:{settings.get('model_id', 'facebook/sam2.1-hiera-large')}",
                    "instance_count": sum(1 for item in results if not item.get("error") and item.get("bbox") is not None),
                    "artifact_descriptions": [
                        {
                            "path": path,
                            "kind": "mask_overlay",
                            "description": "Object mask overlay produced by SAM2.",
                        }
                        for path in artifacts
                    ],
                },
                artifacts=artifacts,
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"SAM2 object masking is not available or failed to initialize: {exc}")
