from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_agent.io.video_sampling import sample_video_frames
from spatial_agent.io.vsibench_runner import build_vsibench_video_path, resolve_vsibench_cache_dir
from spatial_agent.runtime.config import SpatialAgentConfig, load_tool_config
from spatial_agent.tools.object_distance_3d import CompareObjectDistance3DTool


def _load_dataset(dataset_name: str, split: str, cache_dir: str, token: bool | str, dataset_arrow_file: str | None = None):
    if dataset_arrow_file:
        from datasets import Dataset

        return Dataset.from_file(dataset_arrow_file)
    import datasets

    return datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir, token=token)


def _select_doc_ids(dataset, limit: int) -> List[int]:
    doc_ids = []
    for index, doc in enumerate(dataset):
        if doc.get("question_type") != "object_rel_distance":
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


def _parse_question(question: str) -> Tuple[List[str], str, str] | None:
    text = " ".join(question.strip().split())
    match = re.search(
        r"which of these objects\s*\((.+?)\)\s+is the\s+(closest|farthest)\s+to the\s+(.+?)\?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    candidates = [item.strip() for item in match.group(1).split(",") if item.strip()]
    mode = match.group(2).lower()
    reference = match.group(3).strip(" .?")
    return candidates, reference, mode


def _option_object(option: str) -> Tuple[str, str] | None:
    match = re.match(r"\s*([A-Z])\.\s*(.+?)\s*$", option)
    if not match:
        return None
    return match.group(1), match.group(2)


def _run_doc(
    *,
    doc_id: int,
    doc: Dict[str, Any],
    args: argparse.Namespace,
    config: SpatialAgentConfig,
    cache_dir: str,
    tool: CompareObjectDistance3DTool,
) -> Dict[str, Any]:
    doc_dir = Path(args.output_dir) / f"doc_{doc_id}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    result_path = doc_dir / "result.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    parsed = _parse_question(str(doc.get("question") or ""))
    record: Dict[str, Any] = {
        "doc_id": doc_id,
        "question": doc.get("question"),
        "ground_truth": doc.get("ground_truth"),
        "options": doc.get("options"),
        "scene_name": doc.get("scene_name"),
        "dataset": doc.get("dataset"),
    }
    if not parsed:
        record.update({"status": "error", "error": "Failed to parse relative distance question."})
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    question_candidates, reference, mode = parsed
    option_map = dict(filter(None, (_option_object(option) for option in doc.get("options", []))))
    candidates = [option_map.get(letter) for letter in sorted(option_map)]
    if not candidates:
        candidates = question_candidates

    video_path = build_vsibench_video_path(doc["dataset"], doc["scene_name"], cache_dir)
    try:
        frames = sample_video_frames(video_path=video_path, output_dir=str(doc_dir / "sampled_frames"), num_frames=args.num_frames)
        result = tool.invoke(images=frames, reference_object=reference, candidate_objects=candidates, mode=mode)
    except Exception as exc:
        record.update(
            {
                "status": "error",
                "error": str(exc),
                "reference_object": reference,
                "candidate_objects": candidates,
                "mode": mode,
                "selected_object": None,
                "prediction": None,
                "correct": False,
                "payload": {"video_path": video_path},
            }
        )
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record
    payload = result.get("payload") or {}
    selected_object = payload.get("selected_object")
    prediction = None
    for letter, object_name in option_map.items():
        if selected_object and object_name.lower() == str(selected_object).lower():
            prediction = letter
            break

    record.update(
        {
            "status": result.get("status"),
            "error": result.get("error"),
            "reference_object": reference,
            "candidate_objects": candidates,
            "mode": mode,
            "selected_object": selected_object,
            "prediction": prediction,
            "correct": prediction == doc.get("ground_truth"),
            "payload": payload,
        }
    )
    result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _write_summary(records: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    evaluated = [item for item in records if item.get("prediction")]
    summary = {
        "count": len(records),
        "evaluated": len(evaluated),
        "success": sum(1 for item in records if item.get("status") == "success"),
        "correct": sum(1 for item in evaluated if item.get("correct")),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CompareObjectDistance3D on VSI-Bench object_rel_distance samples.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-name", default="nyu-visionx/VSI-Bench")
    parser.add_argument("--dataset-cache-dir", default="/disk/wangzhe/VSI-Bench")
    parser.add_argument("--dataset-arrow-file", default="")
    parser.add_argument("--tool-config-path", default="configs/tool_config.server.json")
    parser.add_argument("--output-dir", default="/tmp/spatial_agent_vsibench_rel_distance")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--doc-ids", default="")
    parser.add_argument("--doc-ids-file", default="")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--hf-token", default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cache_dir = resolve_vsibench_cache_dir(args.dataset_cache_dir)
    dataset = _load_dataset(args.dataset_name, args.split, cache_dir, args.hf_token, args.dataset_arrow_file or None)
    doc_ids = _parse_doc_ids_file(args.doc_ids_file) or _parse_doc_ids(args.doc_ids) or _select_doc_ids(dataset, args.limit)
    config = SpatialAgentConfig(artifact_dir=args.output_dir, tool_config=load_tool_config(args.tool_config_path))
    tool = CompareObjectDistance3DTool(config)

    records = []
    for position, doc_id in enumerate(doc_ids, start=1):
        print(f"[{position}/{len(doc_ids)}] doc_id={doc_id} {dataset[doc_id].get('question')}", flush=True)
        record = _run_doc(doc_id=doc_id, doc=dataset[doc_id], args=args, config=config, cache_dir=cache_dir, tool=tool)
        print(
            f"  status={record.get('status')} pred={record.get('prediction')} gt={record.get('ground_truth')} "
            f"correct={record.get('correct')} selected={record.get('selected_object')} error={record.get('error')}",
            flush=True,
        )
        records.append(record)
    print(json.dumps(_write_summary(records, args.output_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
