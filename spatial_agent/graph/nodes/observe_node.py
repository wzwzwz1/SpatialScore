from __future__ import annotations

from spatial_agent.graph.state import summarize_observation


def observe_node(runtime):
    def _observe_node(state):
        result = state["last_tool_result"]
        state["step_count"] += 1
        state["tool_calls"].append(
            {
                "tool_name": result.get("tool_name"),
                "arguments": state.get("selected_args") or {},
            }
        )
        state["tool_observations"].append(result)
        state["scratchpad"].append(
            {
                "thought": state.get("last_thought"),
                "action": state.get("selected_tool"),
                "arguments": state.get("selected_args") or {},
                "observation": result,
            }
        )
        if result.get("status") != "success":
            state["tool_fail_count"] += 1
            state["tool_errors"].append(result)

        state["messages"].append({"role": "tool", "content": summarize_observation(result)})
        state["reasoning_trace"].append({"stage": "observe", "observation": result})
        state["selected_tool"] = None
        state["selected_args"] = None
        state["pending_decision"] = None
        state["pending_route"] = None
        return state

    return _observe_node

