from __future__ import annotations

import json

from spatial_agent.adapters.base import AdapterResponseError


def reason_node(runtime):
    def _reason_node(state):
        if isinstance(state.get("pending_decision"), dict):
            return state

        llm_attempt = len(state.get("llm_raw_outputs", [])) + 1
        if state["step_count"] >= state["max_steps"]:
            state["status"] = "max_steps"
            state["error"] = "Maximum reasoning steps exceeded."
            state["pending_route"] = "fail"
            return state

        if state["repair_count"] > state["max_repairs"]:
            state["status"] = "failed"
            state["error"] = "Maximum repair attempts exceeded."
            state["pending_route"] = "fail"
            return state

        if state["tool_fail_count"] > state["max_tool_fails"]:
            state["status"] = "failed"
            state["error"] = "Maximum tool failures exceeded."
            state["pending_route"] = "fail"
            return state

        try:
            decision = runtime.llm_adapter.generate(state=state, available_tools=state["available_tools"])
        except AdapterResponseError as exc:
            state.setdefault("llm_raw_outputs", []).append(
                {
                    "attempt": llm_attempt,
                    "status": "error",
                    "raw_output": getattr(exc, "raw_output", ""),
                    "error": str(exc),
                }
            )
            state["pending_decision"] = None
            state["pending_repair_message"] = str(exc)
            state["reasoning_trace"].append(
                {
                    "stage": "repair",
                    "error": str(exc),
                    "raw_output": getattr(exc, "raw_output", ""),
                }
            )
            state["pending_route"] = "repair"
            return state

        if not isinstance(decision, dict):
            state.setdefault("llm_raw_outputs", []).append(
                {
                    "attempt": llm_attempt,
                    "status": "invalid_decision",
                    "raw_output": getattr(runtime.llm_adapter, "last_raw_output", ""),
                    "error": "Model output was not a JSON object.",
                }
            )
            state["pending_decision"] = None
            state["pending_repair_message"] = (
                "Model output must be a JSON object with keys `thought`, `action`, and `finish`."
            )
            state["reasoning_trace"].append(
                {
                    "stage": "repair",
                    "error": "Model output was not a JSON object.",
                    "raw_output": decision,
                }
            )
            state["pending_route"] = "repair"
            return state

        state.setdefault("llm_raw_outputs", []).append(
            {
                "attempt": llm_attempt,
                "status": "success",
                "raw_output": getattr(runtime.llm_adapter, "last_raw_output", json.dumps(decision, ensure_ascii=False)),
                "parsed_step_count": (getattr(runtime.llm_adapter, "last_parse_summary", None) or {}).get("parsed_step_count", 1),
                "accepted_step_count": (getattr(runtime.llm_adapter, "last_parse_summary", None) or {}).get("accepted_step_count", 1),
                "dropped_step_count": (getattr(runtime.llm_adapter, "last_parse_summary", None) or {}).get("dropped_step_count", 0),
                "dropped_steps": (getattr(runtime.llm_adapter, "last_parse_summary", None) or {}).get("dropped_steps", []),
                "multi_step_queue": len(getattr(runtime.llm_adapter, "last_parsed_decisions", [])) > 1,
            }
        )
        parsed_decisions = list(getattr(runtime.llm_adapter, "last_parsed_decisions", []))
        state["pending_decision"] = decision
        state["pending_decision_queue"] = parsed_decisions[1:] if len(parsed_decisions) > 1 else []
        state["last_thought"] = decision.get("thought")
        state["reasoning_trace"].append({"stage": "reason", "decision": decision})
        return state

    return _reason_node
