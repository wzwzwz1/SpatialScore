from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_agent.io.video_sampling import sample_video_frames
from spatial_agent.io.vsibench_runner import build_vsibench_video_path, resolve_vsibench_cache_dir
from spatial_agent.runtime.config import SpatialAgentConfig, load_tool_config
from spatial_agent.tools.object_size_3d import EstimateObjectSize3DTool


def _load_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    cache_dir: str,
    token: bool | str,
    dataset_arrow_file: str | None = None,
):
    if dataset_arrow_file:
        from datasets import Dataset

        return Dataset.from_file(dataset_arrow_file)
    import datasets

    if dataset_config:
        return datasets.load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir, token=token)
    return datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir, token=token)


def _select_doc_ids(dataset, limit: int, question_type: str) -> List[int]:
    doc_ids = []
    for index, doc in enumerate(dataset):
        if doc.get("question_type") != question_type:
            continue
        doc_ids.append(index)
        if len(doc_ids) >= limit:
            break
    return doc_ids


def _parse_doc_ids(value: str) -> List[int] | None:
    if not value.strip():
        return None
    return [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]


def _parse_doc_ids_file(path: str) -> List[int] | None:
    if not path.strip():
        return None
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.extend(int(chunk.strip()) for chunk in line.split(",") if chunk.strip())
    return values


def _parse_object(question: str) -> str | None:
    text = " ".join(question.strip().split())
    match = re.search(
        r"longest dimension\s*\(length,\s*width,\s*or\s*height\)\s+of the\s+(.+?),\s+measured in centimeters\?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" .?")
    return None


def _parse_float(value: Any) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _mean_relative_accuracy(prediction: float | None, ground_truth: float | None) -> float:
    if prediction is None or ground_truth is None:
        return 0.0
    if not math.isfinite(float(prediction)) or not math.isfinite(float(ground_truth)):
        return 0.0
    if float(ground_truth) == 0.0:
        return 1.0 if float(prediction) == 0.0 else 0.0
    rel_error = abs(float(prediction) - float(ground_truth)) / abs(float(ground_truth))
    confidences = [0.50 + 0.05 * index for index in range(10)]
    passed = sum(1 for confidence in confidences if rel_error <= 1.0 - confidence)
    return passed / len(confidences)


def _score(prediction: float | None, ground_truth: float | None) -> Dict[str, Any]:
    if prediction is None or ground_truth is None:
        return {
            "abs_error_cm": None,
            "rel_error": None,
            "mra": 0.0,
            "vsibench_metric": "MRA:.5:.95:.05",
            "within_25pct": False,
            "within_10cm": False,
        }
    abs_error = abs(float(prediction) - float(ground_truth))
    rel_error = abs_error / abs(float(ground_truth)) if float(ground_truth) != 0 else None
    return {
        "abs_error_cm": abs_error,
        "rel_error": rel_error,
        "mra": _mean_relative_accuracy(prediction, ground_truth),
        "vsibench_metric": "MRA:.5:.95:.05",
        "within_25pct": bool(rel_error is not None and rel_error <= 0.25),
        "within_10cm": abs_error <= 10.0,
    }


def _run_doc(
    *,
    doc_id: int,
    doc: Dict[str, Any],
    args: argparse.Namespace,
    config: SpatialAgentConfig,
    cache_dir: str,
    tool: EstimateObjectSize3DTool,
) -> Dict[str, Any]:
    doc_dir = Path(args.output_dir) / f"doc_{doc_id}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    result_path = doc_dir / "result.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    question = str(doc.get("question") or "")
    object_name = _parse_object(question)
    gt = _parse_float(doc.get("ground_truth"))
    record: Dict[str, Any] = {
        "doc_id": doc_id,
        "question": question,
        "question_type": doc.get("question_type"),
        "ground_truth": gt,
        "raw_ground_truth": doc.get("ground_truth"),
        "scene_name": doc.get("scene_name"),
        "dataset": doc.get("dataset"),
        "object": object_name,
    }
    if not object_name:
        record.update({"status": "error", "error": "Failed to parse target object from question."})
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    video_path = build_vsibench_video_path(doc["dataset"], doc["scene_name"], cache_dir)
    record["video_path"] = video_path
    try:
        frames = sample_video_frames(video_path=video_path, output_dir=str(doc_dir / "sampled_frames"), num_frames=args.num_frames)
        result = tool.invoke(images=frames, object=object_name)
    except Exception as exc:
        record.update(
            {
                "status": "error",
                "error": str(exc),
                "prediction": None,
                "payload": {},
                **_score(None, gt),
            }
        )
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    payload = result.get("payload") or {}
    pred = payload.get("size_centimeters")
    record.update(
        {
            "status": result.get("status"),
            "error": result.get("error"),
            "prediction": float(pred) if isinstance(pred, (int, float)) else None,
            "payload": payload,
            "artifacts": result.get("artifacts") or [],
            **_score(float(pred) if isinstance(pred, (int, float)) else None, gt),
        }
    )
    result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _write_summary(records: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    evaluated = [item for item in records if isinstance(item.get("prediction"), (int, float))]
    mra_scores = [float(item.get("mra", 0.0)) for item in records]
    abs_errors = [float(item["abs_error_cm"]) for item in evaluated if isinstance(item.get("abs_error_cm"), (int, float))]
    summary = {
        "count": len(records),
        "evaluated": len(evaluated),
        "success": sum(1 for item in records if item.get("status") == "success"),
        "vsibench_metric": "MRA:.5:.95:.05",
        "mra": sum(mra_scores) / len(mra_scores) if mra_scores else None,
        "mra_percent": (sum(mra_scores) / len(mra_scores) * 100.0) if mra_scores else None,
        "mae_cm": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "rmse_cm": math.sqrt(sum(err * err for err in abs_errors) / len(abs_errors)) if abs_errors else None,
        "within_25pct": sum(1 for item in evaluated if item.get("within_25pct")),
        "within_10cm": sum(1 for item in evaluated if item.get("within_10cm")),
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EstimateObjectSize3D on VSI-Bench object_size_estimation samples.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-name", default="nyu-visionx/VSI-Bench")
    parser.add_argument("--dataset-config", default="full", help="VSI-Bench config: full, pruned, or debiased.")
    parser.add_argument("--dataset-cache-dir", default="/disk/wangzhe/VSI-Bench")
    parser.add_argument("--dataset-arrow-file", default="")
    parser.add_argument("--tool-config-path", default="configs/tool_config.server.json")
    parser.add_argument("--output-dir", default="/tmp/spatial_agent_vsibench_object_size")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--doc-ids", default="")
    parser.add_argument("--doc-ids-file", default="")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--question-type", default="object_size_estimation")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--hf-token", default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cache_dir = resolve_vsibench_cache_dir(args.dataset_cache_dir)
    dataset = _load_dataset(
        args.dataset_name,
        args.dataset_config,
        args.split,
        cache_dir,
        args.hf_token,
        args.dataset_arrow_file or None,
    )
    doc_ids = _parse_doc_ids_file(args.doc_ids_file) or _parse_doc_ids(args.doc_ids) or _select_doc_ids(
        dataset, args.limit, args.question_type
    )

    if args.list_only:
        rows = []
        for doc_id in doc_ids:
            doc = dict(dataset[doc_id])
            video_path = build_vsibench_video_path(doc["dataset"], doc["scene_name"], cache_dir)
            rows.append(
                {
                    "doc_id": doc_id,
                    "question_type": doc.get("question_type"),
                    "video_path": video_path,
                    "video_exists": Path(video_path).exists(),
                    "object": _parse_object(str(doc.get("question") or "")),
                    "ground_truth_cm": _parse_float(doc.get("ground_truth")),
                    "question": doc.get("question"),
                }
            )
        print(json.dumps({"count": len(rows), "docs": rows}, indent=2, ensure_ascii=False))
        return 0

    config = SpatialAgentConfig(artifact_dir=args.output_dir, tool_config=load_tool_config(args.tool_config_path))
    tool = EstimateObjectSize3DTool(config)
    records = []
    for position, doc_id in enumerate(doc_ids, start=1):
        print(f"[{position}/{len(doc_ids)}] doc_id={doc_id} {dataset[doc_id].get('question')}", flush=True)
        record = _run_doc(doc_id=doc_id, doc=dict(dataset[doc_id]), args=args, config=config, cache_dir=cache_dir, tool=tool)
        print(
            f"  status={record.get('status')} pred={record.get('prediction')}cm gt={record.get('ground_truth')}cm "
            f"mra={record.get('mra')} error={record.get('error')}",
            flush=True,
        )
        records.append(record)
    print(json.dumps(_write_summary(records, args.output_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
