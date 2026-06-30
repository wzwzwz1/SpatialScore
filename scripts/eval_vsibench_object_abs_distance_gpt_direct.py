from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_agent.io.video_sampling import sample_video_frames
from spatial_agent.io.vsibench_runner import build_vsibench_video_path, resolve_vsibench_cache_dir


def _load_dataset(dataset_name: str, split: str, cache_dir: str, token: bool | str):
    try:
        import datasets
    except Exception as exc:
        raise RuntimeError("This script requires the `datasets` package.") from exc
    return datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir, token=token)


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
            "abs_error": None,
            "rel_error": None,
            "mra": 0.0,
            "vsibench_metric": "MRA:.5:.95:.05",
            "within_25pct": False,
            "within_05m": False,
        }
    abs_error = abs(prediction - ground_truth)
    rel_error = abs_error / max(abs(ground_truth), 1e-6)
    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "mra": _mean_relative_accuracy(prediction, ground_truth),
        "vsibench_metric": "MRA:.5:.95:.05",
        "within_25pct": rel_error <= 0.25,
        "within_05m": abs_error <= 0.5,
    }


def _encode_image(path: str, *, max_side: int, jpeg_quality: int) -> str:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("This script requires Pillow for image resizing.") from exc

    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, float(max_side) / float(max(width, height)))
        if scale < 1.0:
            image = image.resize((int(width * scale), int(height * scale)))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_messages(question: str, frames: List[str], args: argparse.Namespace) -> List[Dict[str, Any]]:
    system_prompt = (
        "You answer VSI-Bench video spatial questions directly from images. "
        "Do not call tools. Estimate the metric distance visually from the sampled video frames. "
        "Return only valid JSON."
    )
    user_text = (
        f"Question: {question}\n\n"
        "The following images are uniformly sampled frames from the same video, in temporal order. "
        "Estimate the answer in meters. Output JSON exactly like:\n"
        "{\"answer_meters\": 1.23, \"reason\": \"short visual reason\"}\n"
        "The answer_meters value must be a number, not a range."
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for index, frame in enumerate(frames):
        encoded = _encode_image(frame, max_side=args.image_max_side, jpeg_quality=args.jpeg_quality)
        content.append({"type": "text", "text": f"frame_index={index}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                    "detail": args.image_detail,
                },
            }
        )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def _extract_prediction(text: str) -> float | None:
    try:
        parsed = json.loads(text)
        for key in ("answer_meters", "answer", "distance_meters", "distance"):
            if key in parsed:
                return _parse_float(parsed[key])
    except Exception:
        pass
    return _parse_float(text)


def _run_doc(doc_id: int, doc: Dict[str, Any], args: argparse.Namespace, cache_dir: str, client) -> Dict[str, Any]:
    doc_dir = Path(args.output_dir) / f"doc_{doc_id}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    result_path = doc_dir / "result.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    question = str(doc.get("question") or "")
    gt = _parse_float(doc.get("ground_truth"))
    record: Dict[str, Any] = {
        "doc_id": doc_id,
        "question": question,
        "ground_truth": gt,
        "raw_ground_truth": doc.get("ground_truth"),
        "scene_name": doc.get("scene_name"),
        "dataset": doc.get("dataset"),
        "model": args.model,
        "num_frames": args.num_frames,
        "mode": "gpt_direct_no_tools",
    }

    reuse_dir = Path(args.reuse_frame_results_dir) / f"doc_{doc_id}" / "result.json" if args.reuse_frame_results_dir else None
    if reuse_dir and reuse_dir.exists():
        reused = json.loads(reuse_dir.read_text(encoding="utf-8"))
        frame_paths = []
        seen = set()
        for item in reused.get("frame_results") or []:
            path = item.get("image_path")
            if path and path not in seen:
                seen.add(path)
                frame_paths.append(path)
        frame_paths = frame_paths[: args.num_frames]
    else:
        video_path = build_vsibench_video_path(doc["dataset"], doc["scene_name"], cache_dir)
        frame_paths = sample_video_frames(
            video_path=video_path,
            output_dir=str(doc_dir / "sampled_frames"),
            num_frames=args.num_frames,
        )

    record["frame_paths"] = frame_paths
    if not frame_paths:
        record.update({"status": "error", "error": "No frames available."})
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=_build_messages(question, frame_paths, args),
            temperature=0,
            max_tokens=args.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        prediction = _extract_prediction(raw)
        record.update(
            {
                "status": "success" if prediction is not None else "parse_error",
                "prediction": prediction,
                "raw_response": raw,
                **_score(prediction, gt),
            }
        )
    except Exception as exc:
        record.update({"status": "error", "error": str(exc), "prediction": None, **_score(None, gt)})

    result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct GPT-only baseline for VSI-Bench object_abs_distance.")
    parser.add_argument("--dataset-name", default="nyu-visionx/VSI-Bench")
    parser.add_argument("--dataset-cache-dir", default="/disk/wangzhe/VSI-Bench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="/tmp/vsibench_abs_distance_gpt_direct")
    parser.add_argument("--doc-ids", default="")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--model", default=os.getenv("GPT_DIRECT_MODEL", "gpt-5.5"))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--image-max-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--image-detail", default="low", choices=["low", "high", "auto"])
    parser.add_argument("--reuse-frame-results-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--token", default="true")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("This script requires the openai package.") from exc

    token: bool | str = args.token
    if str(args.token).lower() == "true":
        token = True
    elif str(args.token).lower() == "false":
        token = False

    cache_dir = resolve_vsibench_cache_dir(args.dataset_cache_dir)
    dataset = _load_dataset(args.dataset_name, args.split, cache_dir, token)
    doc_ids = _parse_doc_ids(args.doc_ids) or _select_doc_ids(dataset, args.limit, "object_abs_distance")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    client_kwargs: Dict[str, Any] = {"api_key": os.getenv("OPENAI_API_KEY"), "timeout": 600}
    if os.getenv("OPENAI_API_BASE_URL"):
        client_kwargs["base_url"] = os.getenv("OPENAI_API_BASE_URL")
    client = OpenAI(**client_kwargs)

    records: List[Dict[str, Any]] = []
    for offset, doc_id in enumerate(doc_ids, start=1):
        doc = dict(dataset[int(doc_id)])
        print(f"[{offset}/{len(doc_ids)}] doc_id={doc_id} {doc.get('question')}", flush=True)
        record = _run_doc(doc_id, doc, args, cache_dir, client)
        records.append(record)
        print(
            f"  status={record.get('status')} pred={record.get('prediction')} "
            f"gt={record.get('ground_truth')} mra={record.get('mra')} error={record.get('error')}",
            flush=True,
        )

    summary = {
        "count": len(records),
        "success": sum(1 for item in records if item.get("status") == "success"),
        "mean_mra": sum(float(item.get("mra") or 0.0) for item in records) / max(len(records), 1),
        "within_25pct": sum(1 for item in records if item.get("within_25pct")),
        "within_05m": sum(1 for item in records if item.get("within_05m")),
        "records": [
            {
                "doc_id": item.get("doc_id"),
                "status": item.get("status"),
                "prediction": item.get("prediction"),
                "ground_truth": item.get("ground_truth"),
                "rel_error": item.get("rel_error"),
                "mra": item.get("mra"),
                "error": item.get("error"),
            }
            for item in records
        ],
    }
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
