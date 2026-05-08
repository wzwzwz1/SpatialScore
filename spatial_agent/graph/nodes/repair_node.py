from __future__ import annotations


def repair_node(runtime):
    def _repair_node(state):
        state["repair_count"] += 1
        state["messages"].append(
            {
                "role": "system",
                "content": state.get("pending_repair_message") or "Please return valid ReAct JSON.",
            }
        )
        state["pending_repair_message"] = None
        state["pending_route"] = None
        state["pending_decision"] = None
        return state

    return _repair_node

