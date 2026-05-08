from __future__ import annotations

from spatial_agent.adapters.factory import create_llm_adapter
from spatial_agent.agent import SpatialAgent
from spatial_agent.tools.registry import build_default_tool_registry


def build_spatial_agent(config):
    adapter = create_llm_adapter(config)
    registry = build_default_tool_registry(config)
    return SpatialAgent(llm_adapter=adapter, tool_registry=registry, config=config)
