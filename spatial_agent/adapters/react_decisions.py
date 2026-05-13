from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ParsedReActDecisions:
    accepted_steps: List[Dict[str, Any]]
    parsed_step_count: int
    dropped_steps: List[Dict[str, Any]]

    @property
    def accepted_step_count(self) -> int:
        return len(self.accepted_steps)

    @property
    def dropped_step_count(self) -> int:
        return len(self.dropped_steps)


def parse_react_decisions(raw_output: str, max_steps: int = 5) -> ParsedReActDecisions:
    text = raw_output.strip()
    if not text:
        raise json.JSONDecodeError("Empty output", raw_output, 0)

    decoder = json.JSONDecoder()
    parsed_objects: List[Any] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        parsed, next_index = decoder.raw_decode(text, index)
        parsed_objects.append(parsed)
        index = next_index

    accepted_steps: List[Dict[str, Any]] = []
    dropped_steps: List[Dict[str, Any]] = []
    finish_seen = False

    for position, parsed in enumerate(parsed_objects, start=1):
        if finish_seen:
            dropped_steps.append({"index": position, "reason": "after_finish", "step": parsed})
            continue
        if len(accepted_steps) >= max_steps:
            dropped_steps.append({"index": position, "reason": "step_limit_exceeded", "step": parsed})
            continue
        if not _is_valid_react_step(parsed):
            dropped_steps.append({"index": position, "reason": "invalid_step", "step": parsed})
            continue

        accepted_steps.append(parsed)
        finish = parsed.get("finish")
        if isinstance(finish, dict) and finish:
            finish_seen = True

    if not accepted_steps:
        raise json.JSONDecodeError("No valid ReAct decision objects found.", raw_output, 0)

    return ParsedReActDecisions(
        accepted_steps=accepted_steps,
        parsed_step_count=len(parsed_objects),
        dropped_steps=dropped_steps,
    )


def _is_valid_react_step(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    if "thought" not in step:
        return False

    action = step.get("action")
    finish = step.get("finish")

    valid_action = isinstance(action, dict) and isinstance(action.get("name"), str) and bool(action.get("name").strip())
    valid_finish = isinstance(finish, dict) and bool(finish)
    return valid_action or valid_finish
