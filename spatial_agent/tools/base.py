from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseSpatialTool(ABC):
    name = ""
    description = ""
    args_schema: Dict[str, Any] = {"type": "object", "properties": {}}
    returns_schema: Dict[str, Any] = {"type": "object"}

    @abstractmethod
    def invoke(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
            "returns_schema": self.returns_schema,
        }

    def success(self, payload: Dict[str, Any], artifacts: List[str] = None) -> Dict[str, Any]:
        return {
            "status": "success",
            "tool_name": self.name,
            "payload": payload,
            "artifacts": artifacts or [],
            "error": None,
        }

    def error(self, message: str, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        return {
            "status": "error",
            "tool_name": self.name,
            "payload": payload or {},
            "artifacts": [],
            "error": message,
        }

    def unavailable(self, reason: str) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "tool_name": self.name,
            "payload": {},
            "artifacts": [],
            "error": reason,
        }

