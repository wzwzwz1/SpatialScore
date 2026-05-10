from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


KNOWN_SAMPLE_FIELDS = {
    "doc_id",
    "doc",
    "target",
    "arguments",
    "resps",
    "filtered_resps",
    "doc_hash",
    "prompt_hash",
    "target_hash",
}


def infer_vsibench_task_id(doc_id: int, task_name: str = "vsibench", split: str = "test") -> str:
    return f"{task_name}___{split}___{doc_id}"


def _extract_prediction(filtered_resps: Any) -> str:
    if isinstance(filtered_resps, list):
        if not filtered_resps:
            return ""
        first = filtered_resps[0]
        if isinstance(first, list):
            return str(first[0]).strip() if first else ""
        return str(first).strip()
    if filtered_resps is None:
        return ""
    return str(filtered_resps).strip()


def _extract_score_fields(sample: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in sample.items():
        if key in KNOWN_SAMPLE_FIELDS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _normalize_sample(sample: Mapping[str, Any], task_name: str, split: str) -> Dict[str, Any]:
    doc = dict(sample.get("doc", {}))
    doc_id = int(sample["doc_id"])
    return {
        "task_id": infer_vsibench_task_id(doc_id=doc_id, task_name=task_name, split=split),
        "doc_id": doc_id,
        "question": doc.get("question", ""),
        "question_type": doc.get("question_type", "unknown"),
        "ground_truth": doc.get("ground_truth", sample.get("target")),
        "target": sample.get("target"),
        "prediction": _extract_prediction(sample.get("filtered_resps")),
        "scene_name": doc.get("scene_name"),
        "dataset": doc.get("dataset"),
        "score_fields": _extract_score_fields(sample),
        "doc": doc,
        "raw_sample": dict(sample),
    }


def _discover_sample_file(path: Path, task_name: str) -> Path:
    if path.is_file():
        return path

    candidates = sorted(
        list(path.rglob(f"samples_{task_name}_*.jsonl"))
        + list(path.rglob(f"samples_{task_name}_*.json"))
        + list(path.rglob(f"{task_name}.json"))
        + list(path.rglob(f"{task_name}.jsonl")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No sample log file found under {path} for task '{task_name}'.")
    return candidates[0]


def load_lmms_samples(path: str | Path, task_name: str = "vsibench", split: str = "test") -> List[Dict[str, Any]]:
    sample_path = _discover_sample_file(Path(path), task_name=task_name)
    if sample_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        bundle = json.loads(sample_path.read_text(encoding="utf-8"))
        rows = bundle.get("logs", [])
    return [_normalize_sample(row, task_name=task_name, split=split) for row in rows]


def load_spatial_traces(trace_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    traces: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(trace_dir).rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        task_id = payload.get("task_id")
        if task_id and "reasoning_trace" in payload:
            traces[task_id] = payload
    return traces


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_runs(samples: List[Dict[str, Any]], traces: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    question_type_stats: Dict[str, Dict[str, Any]] = {}
    tool_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "success": 0, "error": 0, "unavailable": 0})
    merged_samples: List[Dict[str, Any]] = []
    artifact_rows: List[Dict[str, str]] = []

    trace_hits = 0
    step_counts: List[int] = []
    tool_call_counts: List[int] = []
    status_counts: Dict[str, int] = defaultdict(int)

    for sample in samples:
        trace = traces.get(sample["task_id"])
        trace_found = trace is not None
        if trace_found:
            trace_hits += 1

        tool_calls = list((trace or {}).get("tool_calls", []))
        tool_observations = list((trace or {}).get("tool_observations", []))
        reasoning_trace = list((trace or {}).get("reasoning_trace", []))

        status = (trace or {}).get("status", "missing")
        status_counts[status] += 1
        reasoning_steps = len(reasoning_trace)
        tool_call_count = len(tool_calls)
        artifact_paths: List[str] = []
        tool_names = [call.get("tool_name", "unknown") for call in tool_calls]

        for tool_name in tool_names:
            tool_stats[tool_name]["calls"] += 1

        for observation in tool_observations:
            tool_name = observation.get("tool_name", "unknown")
            observation_status = observation.get("status", "unknown")
            if observation_status in tool_stats[tool_name]:
                tool_stats[tool_name][observation_status] += 1
            artifacts = observation.get("artifacts", []) or []
            for artifact_path in artifacts:
                artifact_paths.append(artifact_path)
                artifact_rows.append(
                    {
                        "sample_task_id": sample["task_id"],
                        "tool_name": tool_name,
                        "path": artifact_path,
                    }
                )

        if trace_found:
            step_counts.append(reasoning_steps)
            tool_call_counts.append(tool_call_count)

        question_type = sample["question_type"]
        q_stat = question_type_stats.setdefault(question_type, {"count": 0, "trace_found": 0, "metrics": defaultdict(list)})
        q_stat["count"] += 1
        q_stat["trace_found"] += int(trace_found)
        for metric_name, value in sample["score_fields"].items():
            q_stat["metrics"][metric_name].append(value)

        merged_samples.append(
            {
                "task_id": sample["task_id"],
                "doc_id": sample["doc_id"],
                "question": sample["question"],
                "question_type": question_type,
                "scene_name": sample["scene_name"],
                "dataset": sample["dataset"],
                "ground_truth": sample["ground_truth"],
                "prediction": sample["prediction"],
                "trace_found": trace_found,
                "status": status,
                "error": (trace or {}).get("error"),
                "tool_names": tool_names,
                "tool_call_count": tool_call_count,
                "reasoning_steps": reasoning_steps,
                "artifact_count": len(artifact_paths),
                "artifact_paths": artifact_paths,
                "score_fields": sample["score_fields"],
                "final_answer": (trace or {}).get("final_answer"),
            }
        )

    normalized_question_types: Dict[str, Dict[str, Any]] = {}
    for question_type, stats in question_type_stats.items():
        normalized_question_types[question_type] = {
            "count": stats["count"],
            "trace_found": stats["trace_found"],
            "metrics": {metric_name: _mean(values) for metric_name, values in stats["metrics"].items()},
        }

    success_count = sum(1 for sample in merged_samples if sample["status"] == "success")

    return {
        "summary": {
            "sample_count": len(samples),
            "trace_count": len(traces),
            "trace_coverage": trace_hits / len(samples) if samples else 0.0,
            "average_reasoning_steps": _mean(step_counts),
            "average_tool_calls": _mean(tool_call_counts),
            "success_rate": success_count / len(samples) if samples else 0.0,
            "status_counts": dict(status_counts),
        },
        "question_types": normalized_question_types,
        "tools": dict(sorted(tool_stats.items())),
        "samples": merged_samples,
        "artifacts": artifact_rows,
    }
