from __future__ import annotations

from spatial_agent.tools.base import BaseSpatialTool


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

    def invoke(self, **kwargs):
        return self.unavailable(
            "DepthAnything-backed object depth estimation requires torch/checkpoints and is not configured in the current environment."
        )

