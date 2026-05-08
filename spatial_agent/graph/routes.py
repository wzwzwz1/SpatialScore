from __future__ import annotations

from spatial_agent.graph.state import SpatialAgentState


def route_from_reason(state: SpatialAgentState) -> str:
    return state.get("pending_route", "fail")

