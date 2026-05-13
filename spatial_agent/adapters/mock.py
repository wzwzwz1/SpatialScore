from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

from spatial_agent.adapters.base import AdapterResponseError, LLMAdapter
from spatial_agent.adapters.react_decisions import parse_react_decisions


class MockLLMAdapter(LLMAdapter):
    """Simple deterministic adapter for graph tests."""

    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.last_raw_output = ""
        self.last_parsed_decisions = []
        self.last_parse_summary = None

    def generate(self, state: Mapping[str, Any], available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.last_parsed_decisions = []
        self.last_parse_summary = None
        if not self._responses:
            raise AdapterResponseError("Mock adapter has no more queued responses.")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            self.last_raw_output = ""
            raise AdapterResponseError(str(response))
        if isinstance(response, str):
            self.last_raw_output = response
            try:
                parsed = parse_react_decisions(response)
                self.last_parsed_decisions = list(parsed.accepted_steps)
                self.last_parse_summary = {
                    "parsed_step_count": parsed.parsed_step_count,
                    "accepted_step_count": parsed.accepted_step_count,
                    "dropped_step_count": parsed.dropped_step_count,
                    "dropped_steps": parsed.dropped_steps,
                }
            except json.JSONDecodeError as exc:
                raise AdapterResponseError("Malformed JSON response from mock adapter.", raw_output=response) from exc
            return parsed.accepted_steps[0]
        if isinstance(response, dict):
            self.last_raw_output = json.dumps(response, ensure_ascii=False)
            self.last_parsed_decisions = [response]
            self.last_parse_summary = {
                "parsed_step_count": 1,
                "accepted_step_count": 1,
                "dropped_step_count": 0,
                "dropped_steps": [],
            }
            return response
        self.last_raw_output = str(response)
        raise AdapterResponseError(f"Unsupported mock response type: {type(response)!r}")
