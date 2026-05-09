from __future__ import annotations

from typing import List

from spatial_agent.tools.backends import ensure_image_paths, get_tool_settings, get_vggt_backend, load_pil_image, resolve_device
from spatial_agent.tools.base import BaseSpatialTool


class GetCameraParametersVGGTTool(BaseSpatialTool):
    name = "GetCameraParametersVGGT"
    description = "Estimate camera intrinsic and extrinsic parameters from image inputs."
    args_schema = {
        "type": "object",
        "properties": {
            "image": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            }
        },
        "required": ["image"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config

    def invoke(self, **kwargs):
        image_paths = ensure_image_paths(kwargs.get("image") or kwargs.get("images"))
        if not image_paths:
            return self.error("GetCameraParametersVGGT requires one or more image paths.")

        settings = get_tool_settings(self.config, self.name, aliases=["camera", "vggt"])
        try:  # pragma: no cover - dependency-heavy runtime path
            device = resolve_device(settings.get("device"))
            backend = get_vggt_backend(
                model_id=str(settings.get("hf_model_id", "facebook/VGGT-1B")),
                checkpoint_path=settings.get("checkpoint_path"),
                device=device,
            )
            load_and_preprocess_images = backend["load_and_preprocess_images"]
            torch = backend["torch"]
            images = load_and_preprocess_images(image_paths, mode=str(settings.get("preprocess_mode", "pad"))).to(device)
            model = backend["model"]

            with torch.no_grad():
                predictions = model(images)

            pose_encoding_to_extri_intri = backend["pose_encoding_to_extri_intri"]
            image_size_hw = tuple(int(value) for value in images.shape[-2:])
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                image_size_hw=image_size_hw,
                build_intrinsics=True,
            )

            output: List[dict] = []
            depth_conf = predictions.get("depth_conf")
            point_conf = predictions.get("world_points_conf")
            for image_index, image_path in enumerate(image_paths):
                original = load_pil_image(image_path)
                output.append(
                    {
                        "image_index": image_index,
                        "image_path": image_path,
                        "original_size_wh": list(original.size),
                        "model_image_size_hw": list(image_size_hw),
                        "extrinsic": extrinsics[0, image_index].detach().cpu().tolist(),
                        "intrinsic": intrinsics[0, image_index].detach().cpu().tolist(),
                        "pose_encoding": predictions["pose_enc"][0, image_index].detach().cpu().tolist(),
                        "depth_confidence_mean": float(depth_conf[0, image_index].mean().detach().cpu()) if depth_conf is not None else None,
                        "world_points_confidence_mean": float(point_conf[0, image_index].mean().detach().cpu()) if point_conf is not None else None,
                    }
                )

            return self.success(
                payload={
                    "output": output,
                    "backend": f"vggt:{settings.get('hf_model_id', 'facebook/VGGT-1B')}",
                }
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"VGGT camera estimation is not available or failed to initialize: {exc}")
