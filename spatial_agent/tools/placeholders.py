from __future__ import annotations

from spatial_agent.tools.base import BaseSpatialTool


class PlaceholderTool(BaseSpatialTool):
    def __init__(self, name: str, description: str, reason: str) -> None:
        self.name = name
        self.description = description
        self.reason = reason
        self.args_schema = {"type": "object", "properties": {}}
        self.returns_schema = {"type": "object"}

    def invoke(self, **kwargs):
        return self.unavailable(self.reason)

