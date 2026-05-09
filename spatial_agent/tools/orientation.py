from __future__ import annotations

from typing import Dict, List

try:  # pragma: no cover - optional runtime dependency
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

from spatial_agent.tools.backends import (
    crop_from_bbox,
    ensure_image_paths,
    ensure_object_names,
    get_orientation_backend,
    get_tool_settings,
    load_pil_image,
    resolve_device,
)
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.localization import LocalizeObjectsTool


class GetObjectOrientationTool(BaseSpatialTool):
    name = "GetObjectOrientation"
    description = "Estimate azimuth/polar/rotation orientation for a named object."
    args_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string"},
            "objects": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
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
            return self.error("GetObjectOrientation requires an image path.")
        if not objects:
            return self.error("GetObjectOrientation requires one or more object names.")

        settings = get_tool_settings(self.config, self.name, aliases=["orientation", "orientanything"])
        try:  # pragma: no cover - dependency-heavy runtime path
            import torch.nn.functional as F

            device = resolve_device(settings.get("device"))
            backend = get_orientation_backend(
                checkpoint_path=settings.get("checkpoint_path"),
                checkpoint_repo_id=str(settings.get("checkpoint_repo_id", "Viglong/Orient-Anything")),
                checkpoint_filename=str(settings.get("checkpoint_filename", "croplargeEX2/dino_weight.pt")),
                dino_mode=str(settings.get("dino_mode", "large")),
                device=device,
            )
            model = backend["model"]
            processor = backend["processor"]
            utils = backend["utils"]
            torch = backend["torch"]

            image_path = image_paths[0]
            image = load_pil_image(image_path)
            localization = self.localizer.invoke(image=image_path, objects=objects)
            localized_regions = localization["payload"].get("regions", []) if localization["status"] == "success" else []
            by_label: Dict[str, List[Dict[str, object]]] = {}
            for region in localized_regions:
                by_label.setdefault(str(region["label"]).lower(), []).append(region)

            results = []
            for object_name in objects:
                candidates = by_label.get(object_name.lower(), [])
                if not candidates:
                    for label, label_regions in by_label.items():
                        if object_name.lower() in label or label in object_name.lower():
                            candidates.extend(label_regions)
                region = max(candidates, key=lambda item: float(item.get("score", 0.0))) if candidates else None
                object_crop = crop_from_bbox(image, region["bbox"]) if region else image
                processed = utils.background_preprocess(object_crop, bool(settings.get("remove_background", True)))
                crop_images = utils.get_crop_images(object_crop, num=3) + utils.get_crop_images(processed, num=3)
                inputs = processor(images=crop_images, return_tensors="np")
                pixel_values = torch.from_numpy(np.array(inputs["pixel_values"])).to(device)
                with torch.no_grad():
                    predictions = model({"pixel_values": pixel_values})

                azimuth = torch.argmax(predictions[:, 0:360], dim=-1).to(torch.float32)
                polar = torch.argmax(predictions[:, 360 : 360 + 180], dim=-1).to(torch.float32)
                rotation = torch.argmax(predictions[:, 360 + 180 : 360 + 180 + 180], dim=-1).to(torch.float32)
                azimuth = utils.remove_outliers_and_average_circular(azimuth)
                polar = utils.remove_outliers_and_average(polar) - 90.0
                rotation = utils.remove_outliers_and_average(rotation) - 90.0
                confidence = float(torch.mean(F.softmax(predictions[:, -2:], dim=-1), dim=0)[0].detach().cpu())

                results.append(
                    {
                        "object": object_name,
                        "bbox": region["bbox"] if region else None,
                        "angle_data": {
                            "azimuth": float(azimuth),
                            "polar": float(polar),
                            "rotation": float(rotation),
                            "confidence": confidence,
                        },
                        "error": None,
                    }
                )

            return self.success(
                payload={
                    "results": results,
                    "backend": f"orient_anything:{settings.get('checkpoint_filename', 'croplargeEX2/dino_weight.pt')}",
                }
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"OrientAnything object orientation is not available or failed to initialize: {exc}")
