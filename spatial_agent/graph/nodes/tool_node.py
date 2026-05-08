from __future__ import annotations


def tool_node(runtime):
    def _tool_node(state):
        tool_name = state.get("selected_tool")
        tool = runtime.tool_registry.get(tool_name)
        if tool is None:
            result = {
                "status": "unavailable",
                "tool_name": tool_name,
                "payload": {},
                "artifacts": [],
                "error": f"Tool `{tool_name}` is not registered.",
            }
        else:
            try:
                result = tool.invoke(**(state.get("selected_args") or {}))
            except Exception as exc:  # pragma: no cover - defensive runtime behavior
                result = {
                    "status": "error",
                    "tool_name": tool_name,
                    "payload": {},
                    "artifacts": [],
                    "error": str(exc),
                }
        state["last_tool_result"] = result
        return state

    return _tool_node

