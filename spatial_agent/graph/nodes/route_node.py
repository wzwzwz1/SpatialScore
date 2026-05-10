from __future__ import annotations

from spatial_agent.graph.tool_args import normalize_tool_arguments


def route_node(runtime):
    def _route_node(state):
        if state.get("pending_route") == "repair":
            return state
        if state.get("status") in {"failed", "max_steps"}:
            state["pending_route"] = "fail"
            return state

        decision = state.get("pending_decision") or {}
        finish = decision.get("finish")
        action = decision.get("action")

        if finish:
            state["pending_route"] = "finalize"
            return state

        if action and action.get("name"):
            state["selected_tool"] = action["name"]
            state["selected_args"] = normalize_tool_arguments(
                state=state,
                tool_name=action["name"],
                arguments=action.get("arguments", {}),
            )
            state["pending_route"] = "tool"
            return state

        state["pending_repair_message"] = "Decision did not contain either a finish payload or a valid action."
        state["pending_route"] = "repair"
        return state

    return _route_node
