from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping


class AdapterResponseError(ValueError):
    """Raised when an adapter cannot return a valid ReAct payload."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


class LLMAdapter(ABC):
    """Graph-facing abstraction for ReAct-capable multimodal language models."""

    last_raw_output: str = ""

    @abstractmethod
    def generate(self, state: Mapping[str, Any], available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a structured ReAct payload for the current state."""
