from __future__ import annotations

from spatial_agent.tools.base import BaseSpatialTool


class LocalizeObjectsTool(BaseSpatialTool):
    name = "LocalizeObjects"
    description = "Localize named objects in an image."
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
            "Object localization is not cleanly derivable from the current repo assets without an added grounding stack."
        )
