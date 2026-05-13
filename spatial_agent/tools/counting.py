from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    ensure_image_paths,
    ensure_object_names,
    get_rex_omni_backend,
    get_tool_settings,
    load_pil_image,
    resolve_device,
    save_point_overlay,
)
from spatial_agent.tools.base import BaseSpatialTool


class CountObjectsTool(BaseSpatialTool):
    name = "CountObjects"
    description = "Count target objects in an image and return normalized instance points."
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
            return self.error("CountObjects requires an image path.")
        if not objects:
            return self.error("CountObjects requires one or more object names.")

        image_path = image_paths[0]
        settings = get_tool_settings(self.config, self.name, aliases=["counting", "count"])
        device = resolve_device(settings.get("device"))
        model_path = str(
            settings.get("model_path")
            or settings.get("model_id")
            or settings.get("checkpoint_path")
            or "IDEA-Research/Rex-Omni"
        )
        backend_name = str(settings.get("backend", "transformers"))
        repo_path = settings.get("repo_path")
        quantization = settings.get("quantization")

        try:  # pragma: no cover - dependency-heavy runtime path
            backend = get_rex_omni_backend(
                model_path=model_path,
                backend=backend_name,
                device=device,
                repo_path=str(repo_path) if repo_path else None,
                quantization=str(quantization) if quantization else None,
                max_tokens=int(settings.get("max_tokens", 2048)),
                temperature=float(settings.get("temperature", 0.0)),
                top_p=float(settings.get("top_p", 0.05)),
                top_k=int(settings.get("top_k", 1)),
                repetition_penalty=float(settings.get("repetition_penalty", 1.05)),
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"Rex-Omni counting backend is not available or failed to initialize: {exc}")

        image = load_pil_image(image_path)
        width, height = image.size

        try:  # pragma: no cover - dependency-heavy runtime path
            outputs = backend["wrapper"].inference(images=image, task="pointing", categories=objects)
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.error(f"Rex-Omni counting inference failed: {exc}")

        first_output = outputs[0] if isinstance(outputs, list) and outputs else outputs
        predictions = {}
        if isinstance(first_output, dict):
            predictions = first_output.get("extracted_predictions") or {}

        normalized_points: Dict[str, List[List[float]]] = {}
        for object_name in objects:
            raw_points = predictions.get(object_name, [])
            points: List[List[float]] = []
            for point in raw_points:
                if not isinstance(point, dict):
                    continue
                coords = point.get("coords") or []
                if point.get("type") != "point" or len(coords) < 2:
                    continue
                x_abs = float(coords[0])
                y_abs = float(coords[1])
                points.append(
                    [
                        round(max(0.0, min(1.0, x_abs / width)), 6),
                        round(max(0.0, min(1.0, y_abs / height)), 6),
                    ]
                )
            normalized_points[object_name] = points

        artifact_path = artifact_dir_for_tool(self.config, self.name) / f"{Path(image_path).stem}_count_points.png"
        artifact = save_point_overlay(image, normalized_points, artifact_path)
        instance_count = sum(len(points) for points in normalized_points.values())
        payload = {
            "points": normalized_points,
            "instance_count": instance_count,
            "backend": backend["backend_label"],
            "artifact_descriptions": [
                {
                    "path": artifact,
                    "kind": "point_overlay",
                    "description": "Rex-Omni pointing results used for object counting.",
                }
            ],
        }
        return self.success(payload=payload, artifacts=[artifact])
