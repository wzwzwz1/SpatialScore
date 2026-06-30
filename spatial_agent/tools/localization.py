from __future__ import annotations

import hashlib
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


OBJECT_ALIASES: Dict[str, List[str]] = {
    "trash bin": [
        "trash bin",
        "trash can",
        "garbage can",
        "garbage bin",
        "waste bin",
        "wastebasket",
        "dustbin",
        "bin",
        "recycling bin",
    ],
    "trash can": [
        "trash can",
        "trash bin",
        "garbage can",
        "garbage bin",
        "waste bin",
        "wastebasket",
        "dustbin",
        "bin",
        "recycling bin",
    ],
    "couch": ["couch", "sofa"],
    "sofa": ["sofa", "couch"],
    "tv": ["tv", "television", "monitor", "screen"],
    "television": ["television", "tv", "monitor", "screen"],
    "plant": ["plant", "potted plant", "houseplant"],
}


def _aliases_for_object(object_name: str) -> List[str]:
    aliases = OBJECT_ALIASES.get(object_name.lower(), [object_name])
    values: List[str] = []
    seen = set()
    for alias in [object_name, *aliases]:
        key = alias.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        values.append(alias.strip())
    return values


def _label_matches_query(label_text: str, query: str) -> bool:
    label = label_text.lower().strip()
    target = query.lower().strip()
    return bool(label and target) and (target in label or label in target)


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
            query_to_object: Dict[str, str] = {name.lower(): name for name in objects}
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
            per_object: Dict[str, List[Dict[str, Any]]] = {name: [] for name in objects}
            regions: List[Dict[str, Any]] = []

            for box, score, label in zip(processed["boxes"], processed["scores"], processed["labels"]):
                label_text = str(label)
                bbox = clamp_bbox(box.tolist(), width, height)
                region = {"label": label_text, "bbox": bbox, "score": float(score)}
                matched = False
                for query, object_name in query_to_object.items():
                    if _label_matches_query(label_text, query):
                        per_object[object_name].append({**region, "matched_alias": query, "detector_label": label_text})
                        matched = True
                if matched:
                    regions.append(region)

            alias_queries_by_object = {
                object_name: [
                    alias
                    for alias in _aliases_for_object(object_name)
                    if alias.lower() != object_name.lower()
                ]
                for object_name in objects
            }
            for object_name in objects:
                if per_object[object_name] or not alias_queries_by_object[object_name]:
                    continue
                alias_prompt = ". ".join(alias_queries_by_object[object_name]) + "."
                alias_inputs = processor(images=image, text=alias_prompt, return_tensors="pt")
                alias_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in alias_inputs.items()}
                with torch.no_grad():
                    alias_outputs = model(**alias_inputs)
                alias_processed = processor.post_process_grounded_object_detection(
                    alias_outputs,
                    alias_inputs["input_ids"],
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[image.size[::-1]],
                )[0]
                for box, score, label in zip(alias_processed["boxes"], alias_processed["scores"], alias_processed["labels"]):
                    label_text = str(label)
                    for alias in alias_queries_by_object[object_name]:
                        if not _label_matches_query(label_text, alias):
                            continue
                        bbox = clamp_bbox(box.tolist(), width, height)
                        region = {
                            "label": label_text,
                            "bbox": bbox,
                            "score": float(score),
                            "matched_alias": alias.lower(),
                            "detector_label": label_text,
                        }
                        per_object[object_name].append(region)
                        regions.append(region)
                        break

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
            prompt_hash = hashlib.sha1("|".join(name.lower() for name in objects).encode("utf-8")).hexdigest()[:10]
            artifact_path = artifact_dir_for_tool(self.config, self.name) / f"{Path(image_path).stem}_{prompt_hash}_bbox.png"
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
