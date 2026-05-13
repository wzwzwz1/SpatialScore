from __future__ import annotations

from typing import Any, Dict, List, Optional


SpatialAgentState = Dict[str, Any]


def build_initial_state(task_input: Dict[str, Any], available_tools: List[Dict[str, Any]], config) -> SpatialAgentState:
    image_paths = list(task_input.get("image_paths", []))
    options = task_input.get("options")
    return {
        "task_id": task_input.get("task_id", "task"),
        "question": task_input["question"],
        "question_type": task_input.get("question_type", "open_ended"),
        "input_modality": task_input.get("input_modality", "single_image"),
        "image_paths": image_paths,
        "options": options,
        "metadata": task_input.get("metadata", {}),
        "step_count": 0,
        "max_steps": config.max_steps,
        "repair_count": 0,
        "max_repairs": config.max_repairs,
        "tool_fail_count": 0,
        "max_tool_fails": config.max_tool_fails,
        "messages": [
            {"role": "user", "content": task_input["question"]},
        ],
        "llm_raw_outputs": [],
        "scratchpad": [],
        "last_thought": None,
        "selected_tool": None,
        "selected_args": None,
        "last_tool_result": None,
        "available_tools": available_tools,
        "tool_calls": [],
        "tool_observations": [],
        "tool_errors": [],
        "final_answer": None,
        "pending_decision": None,
        "pending_decision_queue": [],
        "pending_route": None,
        "pending_repair_message": None,
        "executed_decision_queue_steps": [],
        "reasoning_trace": [],
        "status": "running",
        "error": None,
    }


def summarize_observation(result: Dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    tool_name = result.get("tool_name", "unknown")
    payload = result.get("payload", {})
    if status == "success":
        summary_parts = []
        instance_count = payload.get("instance_count")
        if isinstance(instance_count, int):
            summary_parts.append(f"instance_count={instance_count}")
        results = payload.get("results")
        if isinstance(results, list):
            summary_parts.append(f"result_items={len(results)}")
        summary_prefix = f"{tool_name} succeeded"
        if summary_parts:
            summary_prefix += " (" + ", ".join(summary_parts) + ")"
        return f"{summary_prefix} with payload: {payload}"
    return f"{tool_name} returned status={status} error={result.get('error')}"
