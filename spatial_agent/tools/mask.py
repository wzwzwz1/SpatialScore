from __future__ import annotations

from spatial_agent.tools.base import BaseSpatialTool


class GetObjectMaskTool(BaseSpatialTool):
    name = "GetObjectMask"
    description = "Estimate segmentation masks and areas for named objects."
    args_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "string"},
            "objects": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["image", "objects"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config

    def invoke(self, **kwargs):
        return self.unavailable(
            "SAM2-backed object masking requires torch/checkpoints and is not configured in the current environment."
        )

