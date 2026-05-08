from __future__ import annotations

from spatial_agent.adapters.base import AdapterResponseError


def reason_node(runtime):
    def _reason_node(state):
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

        state["pending_decision"] = decision
        state["last_thought"] = decision.get("thought")
        state["reasoning_trace"].append({"stage": "reason", "decision": decision})
        return state

    return _reason_node

