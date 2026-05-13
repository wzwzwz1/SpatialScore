from __future__ import annotations

from typing import Any, Dict, Optional

from spatial_agent.graph.builder import build_graph
from spatial_agent.runtime.config import SpatialAgentConfig
from spatial_agent.runtime.runtime import GraphRuntime
from spatial_agent.runtime.tracing import write_debug_dump, write_trace
from spatial_agent.tools.registry import ToolRegistry


class SpatialAgent:
    """Thin runtime wrapper around the compiled LangGraph graph."""

    def __init__(
        self,
        llm_adapter,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[SpatialAgentConfig] = None,
    ) -> None:
        self.config = config or SpatialAgentConfig()
        self.tool_registry = tool_registry or ToolRegistry()
        self.runtime = GraphRuntime(
            llm_adapter=llm_adapter,
            tool_registry=self.tool_registry,
            config=self.config,
        )
        self.graph = build_graph(self.runtime)

    def invoke(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run the SpatialAgent graph on a single task input."""
        result = self.graph.invoke(task_input)
        trace = {
            "task_id": result.get("task_id", task_input.get("task_id", "task")),
            "question": result.get("question", task_input.get("question")),
            "image_paths": result.get("image_paths", task_input.get("image_paths", [])),
            "status": result.get("status"),
            "error": result.get("error"),
            "final_answer": result.get("final_answer"),
            "tool_calls": result.get("tool_calls", []),
            "tool_observations": result.get("tool_observations", []),
            "reasoning_trace": result.get("reasoning_trace", []),
            "llm_raw_outputs": result.get("llm_raw_outputs", []),
        }
        result["trace_path"] = write_trace(
            trace=trace,
            artifact_dir=self.config.artifact_dir,
            task_id=trace["task_id"],
        )
        result["llm_raw_outputs_path"] = write_debug_dump(
            payload={
                "task_id": trace["task_id"],
                "question": trace["question"],
                "status": trace["status"],
                "llm_raw_outputs": result.get("llm_raw_outputs", []),
            },
            artifact_dir=self.config.artifact_dir,
            filename=f"{trace['task_id']}_llm_raw_outputs.json",
        )
        return result
