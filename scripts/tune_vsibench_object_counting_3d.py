from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_agent.io.video_sampling import sample_video_frames
from spatial_agent.io.vsibench_runner import build_vsibench_video_path, resolve_vsibench_cache_dir
from spatial_agent.runtime.config import SpatialAgentConfig, load_tool_config
from spatial_agent.tools.video_counting_3d import CountVideoObjects3DTool
from spatial_agent.tools.video_counting_3d_utils import constrained_greedy_cluster


def _load_dataset(dataset_name: str, split: str, cache_dir: str, token: bool | str, dataset_arrow_file: str | None = None):
    if dataset_arrow_file:
        from datasets import Dataset

        return Dataset.from_file(dataset_arrow_file)
    try:
        import datasets
    except Exception as exc:
        raise RuntimeError("This script requires the `datasets` package.") from exc
    return datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir, token=token)


def _object_from_question(question: str) -> str:
    match = re.search(r"How many\s+(.+?)\(s\)\s+are", question, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"How many\s+(.+?)s?\s+are", question, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "object"


def _select_doc_ids(dataset, limit: int, question_type: str) -> List[int]:
    doc_ids: List[int] = []
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
    return _parse_doc_ids(Path(path).read_text(encoding="utf-8"))


def _parse_ground_truth(value: Any) -> int | None:
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _mean_relative_accuracy(prediction: float | None, ground_truth: float | None) -> float:
    if prediction is None or ground_truth is None:
        return 0.0
    if float(ground_truth) == 0.0:
        return 1.0 if float(prediction) == 0.0 else 0.0
    rel_error = abs(float(prediction) - float(ground_truth)) / abs(float(ground_truth))
    confidences = [0.50 + 0.05 * index for index in range(10)]
    passed = sum(1 for confidence in confidences if rel_error <= 1.0 - confidence)
    return passed / len(confidences)


def _run_doc(
    *,
    doc_id: int,
    doc: Dict[str, Any],
    args: argparse.Namespace,
    tool_config: Dict[str, Any],
    cache_dir: str,
) -> Dict[str, Any]:
    doc_dir = Path(args.output_dir) / f"doc_{doc_id}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    result_path = doc_dir / "tool_result.json"
    manifest_path = doc_dir / "tools" / "CountVideoObjects3D" / "visra_3d_counting.json"

    if args.resume and result_path.exists() and manifest_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    video_path = build_vsibench_video_path(
        dataset=doc["dataset"],
        scene_name=doc["scene_name"],
        cache_dir=cache_dir,
    )
    target_object = _object_from_question(str(doc.get("question") or ""))
    frame_dir = doc_dir / "sampled_frames"

    try:
        frames = sample_video_frames(video_path=video_path, output_dir=str(frame_dir), num_frames=args.num_frames)
        config = SpatialAgentConfig(
            artifact_dir=str(doc_dir),
            tool_config=tool_config,
        )
        result = CountVideoObjects3DTool(config).invoke(images=frames, objects=[target_object])
    except Exception as exc:
        record = {
            "doc_id": doc_id,
            "question": doc.get("question"),
            "question_type": doc.get("question_type"),
            "scene_name": doc.get("scene_name"),
            "dataset": doc.get("dataset"),
            "ground_truth": doc.get("ground_truth"),
            "target_object": target_object,
            "frame_count": 0,
            "status": "error",
            "error": str(exc),
            "instance_count": None,
            "manifest_path": None,
            "artifacts": [],
            "video_path": video_path,
        }
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    record = {
        "doc_id": doc_id,
        "question": doc.get("question"),
        "question_type": doc.get("question_type"),
        "scene_name": doc.get("scene_name"),
        "dataset": doc.get("dataset"),
        "ground_truth": doc.get("ground_truth"),
        "target_object": target_object,
        "frame_count": len(frames),
        "status": result.get("status"),
        "error": result.get("error"),
        "instance_count": (result.get("payload") or {}).get("instance_count"),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "artifacts": result.get("artifacts") or [],
    }
    result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _scan_thresholds(records: List[Dict[str, Any]], thresholds: List[float]) -> List[Dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        exact = 0
        absolute_errors: List[int] = []
        mra_scores: List[float] = []
        evaluated = 0
        per_doc = []
        for record in records:
            manifest_path = record.get("manifest_path")
            gt = _parse_ground_truth(record.get("ground_truth"))
            if not manifest_path or gt is None or not Path(manifest_path).exists():
                continue
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            views = [view for view in manifest.get("views", []) if view.get("object") == record.get("target_object")]
            instances = constrained_greedy_cluster(views, distance_threshold=threshold)
            pred = len(instances)
            err = abs(pred - gt)
            mra = _mean_relative_accuracy(pred, gt)
            evaluated += 1
            exact += int(pred == gt)
            absolute_errors.append(err)
            mra_scores.append(mra)
            per_doc.append({"doc_id": record["doc_id"], "prediction": pred, "ground_truth": gt, "abs_error": err, "mra": mra})
        mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        rows.append(
            {
                "threshold": threshold,
                "evaluated": evaluated,
                "vsibench_metric": "MRA:.5:.95:.05",
                "mra": sum(mra_scores) / evaluated if evaluated else None,
                "mra_percent": (sum(mra_scores) / evaluated * 100.0) if evaluated else None,
                "exact": exact,
                "accuracy": exact / evaluated if evaluated else None,
                "mae": mae,
                "per_doc": per_doc,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune CountVideoObjects3D CG threshold on VSI-Bench object-counting samples.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-name", default="nyu-visionx/VSI-Bench")
    parser.add_argument("--dataset-cache-dir", default="/disk/wangzhe/VSI-Bench")
    parser.add_argument("--dataset-arrow-file", default="")
    parser.add_argument("--tool-config-path", default="configs/tool_config.server.json")
    parser.add_argument("--output-dir", default="/tmp/spatial_agent_vsibench_object_tune")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--doc-ids", default="")
    parser.add_argument("--doc-ids-file", default="")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--question-type", default="object_counting")
    parser.add_argument("--thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--hf-token", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    cache_dir = resolve_vsibench_cache_dir(args.dataset_cache_dir)
    dataset = _load_dataset(args.dataset_name, args.split, cache_dir, args.hf_token, args.dataset_arrow_file or None)
    doc_ids = _parse_doc_ids_file(args.doc_ids_file) or _parse_doc_ids(args.doc_ids) or _select_doc_ids(dataset, args.limit, args.question_type)
    tool_config = load_tool_config(args.tool_config_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.json"
    if args.scan_only:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    else:
        records = []
        for position, doc_id in enumerate(doc_ids, start=1):
            print(f"[{position}/{len(doc_ids)}] doc_id={doc_id} {dataset[doc_id].get('question')}", flush=True)
            records.append(
                _run_doc(
                    doc_id=doc_id,
                    doc=dict(dataset[doc_id]),
                    args=args,
                    tool_config=tool_config,
                    cache_dir=cache_dir,
                )
            )
            print(
                f"  status={records[-1]['status']} count={records[-1]['instance_count']} gt={records[-1]['ground_truth']}",
                flush=True,
            )
        records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    scan = _scan_thresholds(records, thresholds)
    scan_path = output_dir / "threshold_scan.json"
    scan_path.write_text(json.dumps(scan, indent=2, ensure_ascii=False), encoding="utf-8")

    print("threshold,evaluated,mra,mra_percent,exact,accuracy,mae")
    for row in scan:
        print(f"{row['threshold']},{row['evaluated']},{row['mra']},{row['mra_percent']},{row['exact']},{row['accuracy']},{row['mae']}")
    print(f"records_json: {records_path}")
    print(f"threshold_scan_json: {scan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
