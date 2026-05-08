from __future__ import annotations

from spatial_agent.graph.state import build_initial_state


def init_state(runtime):
    def _init_state(task_input):
        return build_initial_state(task_input, runtime.tool_registry.to_metadata(), runtime.config)

    return _init_state

