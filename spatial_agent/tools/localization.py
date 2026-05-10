from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    clamp_bbox,
    ensure_image_paths,
    ensure_object_names,
    get_grounding_backend,
    get_ram_backend,
    get_tool_settings,
    load_pil_image,
    resolve_device,
    save_bbox_overlay,
)
from spatial_agent.tools.base import BaseSpatialTool


class LocalizeObjectsTool(BaseSpatialTool):
    name = "LocalizeObjects"
    description = "Localize named objects in an image."
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
        "required": ["objects"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config

    def invoke(self, **kwargs):
        image_paths = ensure_image_paths(kwargs.get("image"))
        objects = ensure_object_names(kwargs.get("objects"))
        if not image_paths:
            return self.error("LocalizeObjects requires at least one image path.")
        if not objects:
            return self.error("LocalizeObjects requires one or more object names.")

        image_path = image_paths[0]
        settings = get_tool_settings(self.config, self.name, aliases=["localization", "grounding"])
        try:  # pragma: no cover - dependency-heavy runtime path
            device = resolve_device(settings.get("device"))
            model_id = str(settings.get("model_id", "IDEA-Research/grounding-dino-base"))
            box_threshold = float(settings.get("box_threshold", 0.30))
            text_threshold = float(settings.get("text_threshold", 0.25))
            backend = get_grounding_backend(model_id, device)
            image = load_pil_image(image_path)
            prompt = ". ".join(objects) + "."
            processor = backend["processor"]
            model = backend["model"]
            torch = backend["torch"]

            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)

            processed = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]

            width, height = image.size
            normalized_objects = {name.lower(): name for name in objects}
            per_object: Dict[str, List[Dict[str, Any]]] = {name: [] for name in objects}
            regions: List[Dict[str, Any]] = []

            for box, score, label in zip(processed["boxes"], processed["scores"], processed["labels"]):
                label_text = str(label)
                bbox = clamp_bbox(box.tolist(), width, height)
                region = {"label": label_text, "bbox": bbox, "score": float(score)}
                matched = False
                for object_name in objects:
                    if object_name.lower() in label_text.lower() or label_text.lower() in object_name.lower():
                        per_object[object_name].append(region)
                        matched = True
                if matched or label_text.lower() in normalized_objects:
                    regions.append(region)

            for object_name in objects:
                if per_object[object_name]:
                    best_region = max(per_object[object_name], key=lambda item: item["score"])
                    regions.append({**best_region, "label": object_name})

            deduped: List[Dict[str, Any]] = []
            seen = set()
            for region in sorted(regions, key=lambda item: item["score"], reverse=True):
                key = (region["label"], tuple(round(value, 2) for value in region["bbox"]))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(region)

            payload: Dict[str, Any] = {
                "regions": deduped,
                "missing_objects": [name for name in objects if not any(region["label"] == name for region in deduped)],
                "backend": f"grounding_dino:{model_id}",
                "instance_count": len(deduped),
            }

            if settings.get("enable_ram_tags"):
                ram_settings = get_tool_settings(self.config, "ram", aliases=["localizeobjects_ram"])
                checkpoint_path = ram_settings.get("checkpoint_path")
                if checkpoint_path:
                    try:
                        ram_backend = get_ram_backend(
                            checkpoint_path=str(checkpoint_path),
                            vit=str(ram_settings.get("vit", "swin_l")),
                            image_size=int(ram_settings.get("image_size", 384)),
                            device=device,
                        )
                        transform = ram_backend["transform"]
                        tensor = transform(image).unsqueeze(0).to(device)
                        tags, tags_zh = ram_backend["inference_ram"](tensor, ram_backend["model"])
                        payload["ram_tags"] = [tag.strip() for tag in tags.split("|") if tag.strip()]
                        payload["ram_tags_zh"] = [tag.strip() for tag in tags_zh.split("|") if tag.strip()]
                    except Exception as exc:  # pragma: no cover - best effort enrichment
                        payload["ram_error"] = str(exc)

            if not deduped:
                return self.error("Object grounding produced no candidate regions.", payload=payload)
            artifact_path = artifact_dir_for_tool(self.config, self.name) / f"{Path(image_path).stem}_bbox.png"
            artifact = save_bbox_overlay(image, deduped, artifact_path)
            payload["artifact_descriptions"] = [
                {
                    "path": artifact,
                    "kind": "bbox_overlay",
                    "description": "Object localization bounding boxes with label and confidence.",
                }
            ]
            return self.success(payload=payload, artifacts=[artifact])
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"Object localization backend is not available or failed to initialize: {exc}")
