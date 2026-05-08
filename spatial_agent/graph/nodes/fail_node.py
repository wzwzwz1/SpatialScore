from __future__ import annotations


def fail_node(runtime):
    def _fail_node(state):
        if state.get("status") == "running":
            state["status"] = "failed"
        if not state.get("error"):
            state["error"] = "Agent terminated without producing a final answer."
        state["reasoning_trace"].append({"stage": "fail", "error": state["error"]})
        return state

    return _fail_node

