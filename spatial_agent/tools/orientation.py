from __future__ import annotations

from spatial_agent.tools.base import BaseSpatialTool


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

    def invoke(self, **kwargs):
        return self.unavailable(
            "OrientAnything-backed orientation estimation requires torch/checkpoints and is not configured in the current environment."
        )

