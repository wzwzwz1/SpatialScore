from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

from spatial_agent.adapters.base import AdapterResponseError, LLMAdapter


class MockLLMAdapter(LLMAdapter):
    """Simple deterministic adapter for graph tests."""

    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.last_raw_output = ""

    def generate(self, state: Mapping[str, Any], available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self._responses:
            raise AdapterResponseError("Mock adapter has no more queued responses.")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            self.last_raw_output = ""
            raise AdapterResponseError(str(response))
        if isinstance(response, str):
            self.last_raw_output = response
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError as exc:
                raise AdapterResponseError("Malformed JSON response from mock adapter.", raw_output=response) from exc
            return parsed
        if isinstance(response, dict):
            self.last_raw_output = json.dumps(response, ensure_ascii=False)
            return response
        self.last_raw_output = str(response)
        raise AdapterResponseError(f"Unsupported mock response type: {type(response)!r}")
