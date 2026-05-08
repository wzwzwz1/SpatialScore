from __future__ import annotations

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
        return self.unavailable(
            "VGGT-backed camera estimation requires torch/checkpoints and is not configured in the current environment."
        )

