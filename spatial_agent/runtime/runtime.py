from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphRuntime:
    llm_adapter: object
    tool_registry: object
    config: object

