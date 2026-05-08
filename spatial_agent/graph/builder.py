from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from spatial_agent.graph.routes import route_from_reason
from spatial_agent.graph.state import SpatialAgentState
from spatial_agent.graph.nodes.fail_node import fail_node
from spatial_agent.graph.nodes.finalize_node import finalize_node
from spatial_agent.graph.nodes.init_node import init_state
from spatial_agent.graph.nodes.observe_node import observe_node
from spatial_agent.graph.nodes.reason_node import reason_node
from spatial_agent.graph.nodes.repair_node import repair_node
from spatial_agent.graph.nodes.route_node import route_node
from spatial_agent.graph.nodes.tool_node import tool_node


def build_graph(runtime):
    graph = StateGraph(dict)
    graph.add_node("init_state", init_state(runtime))
    graph.add_node("reason_node", reason_node(runtime))
    graph.add_node("route_node", route_node(runtime))
    graph.add_node("tool_node", tool_node(runtime))
    graph.add_node("observe_node", observe_node(runtime))
    graph.add_node("repair_node", repair_node(runtime))
    graph.add_node("finalize_node", finalize_node(runtime))
    graph.add_node("fail_node", fail_node(runtime))

    graph.add_edge(START, "init_state")
    graph.add_edge("init_state", "reason_node")
    graph.add_edge("reason_node", "route_node")
    graph.add_conditional_edges(
        "route_node",
        route_from_reason,
        {
            "tool": "tool_node",
            "repair": "repair_node",
            "finalize": "finalize_node",
            "fail": "fail_node",
        },
    )
    graph.add_edge("tool_node", "observe_node")
    graph.add_edge("observe_node", "reason_node")
    graph.add_edge("repair_node", "reason_node")
    graph.add_edge("finalize_node", END)
    graph.add_edge("fail_node", END)
    return graph.compile()

