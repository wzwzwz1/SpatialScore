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
from typing import Any, Dict, List, Sequence, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_agent.io.video_sampling import sample_video_frames
from spatial_agent.io.vsibench_runner import build_vsibench_video_path, resolve_vsibench_cache_dir
from spatial_agent.runtime.config import SpatialAgentConfig, load_tool_config
from spatial_agent.tools.backends import get_sam2_predictor, get_tool_settings, get_vggt_backend, load_pil_image, load_rgb_array, map_point_to_preprocessed, resolve_device
from spatial_agent.tools.distance import Get3DDistanceTool
from spatial_agent.tools.localization import LocalizeObjectsTool


def _load_dataset(dataset_name: str, split: str, cache_dir: str, token: bool | str):
    try:
        import datasets
    except Exception as exc:
        raise RuntimeError("This script requires the `datasets` package.") from exc
    return datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir, token=token)


def _load_annotation_dataset(path: str) -> List[Dict[str, Any]]:
    annotation_path = Path(path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
    if annotation_path.suffix == ".parquet":
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("Loading parquet annotations requires `pandas` and `pyarrow`.") from exc
        frame = pd.read_parquet(annotation_path)
        records: List[Dict[str, Any]] = []
        for index, item in frame.iterrows():
            record = {
                key: (None if _is_nan(value) else value)
                for key, value in item.to_dict().items()
            }
            record["_annotation_index"] = int(index)
            if "source" in record and "dataset" not in record:
                record["dataset"] = record.get("source")
            records.append(record)
        return records
    if annotation_path.suffix == ".jsonl":
        records = []
        with annotation_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_annotation_index"] = int(index)
                if "source" in record and "dataset" not in record:
                    record["dataset"] = record.get("source")
                records.append(record)
        return records
    raise ValueError(f"Unsupported annotation format: {annotation_path.suffix}")


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


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
    if not path:
        return None
    doc_ids_path = Path(path)
    if not doc_ids_path.exists():
        raise FileNotFoundError(f"Doc ids file does not exist: {doc_ids_path}")
    values = []
    for line in doc_ids_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.extend(int(chunk.strip()) for chunk in line.split(",") if chunk.strip())
    return values


def _resolve_video_path(doc: Dict[str, Any], cache_dir: str, args: argparse.Namespace) -> str:
    if doc.get("video"):
        video_path = Path(str(doc["video"]))
        if not video_path.is_absolute():
            root = Path(args.video_root or cache_dir)
            video_path = root / video_path
        return str(video_path)
    return build_vsibench_video_path(doc["dataset"], doc["scene_name"], cache_dir)


def _parse_objects(question: str) -> Tuple[str, str] | None:
    text = " ".join(question.strip().split())
    patterns = [
        r"distance between the (.+?) and the (.+?)(?:\s*\(|\?|$)",
        r"distance between (.+?) and (.+?)(?:\s*\(|\?|$)",
        r"distance from the (.+?) to the (.+?)(?:\s*\(|\?|$)",
        r"distance from (.+?) to (.+?)(?:\s*\(|\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_object(match.group(1)), _clean_object(match.group(2))
    return None


def _clean_object(value: str) -> str:
    value = re.sub(r"\bin meters\b.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bmeasuring from.*$", "", value, flags=re.IGNORECASE)
    value = value.strip(" .?()")
    return value


def _parse_float(value: Any) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _best_region(regions: Sequence[Dict[str, Any]], object_name: str) -> Dict[str, Any] | None:
    candidates = [region for region in regions if str(region.get("label", "")).lower() == object_name.lower()]
    if not candidates:
        candidates = [
            region
            for region in regions
            if object_name.lower() in str(region.get("label", "")).lower()
            or str(region.get("label", "")).lower() in object_name.lower()
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("score") or 0.0))


def _closest_bbox_points(
    bbox_1: Sequence[float],
    bbox_2: Sequence[float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pairs = _closest_bbox_boundary_point_pairs(bbox_1, bbox_2, samples_per_edge=24, top_k=1, min_pixel_distance=8.0)
    if pairs:
        return pairs[0]

    x11, y11, x12, y12 = [float(value) for value in bbox_1[:4]]
    x21, y21, x22, y22 = [float(value) for value in bbox_2[:4]]
    cx1 = (x11 + x12) / 2.0
    cy1 = (y11 + y12) / 2.0
    cx2 = (x21 + x22) / 2.0
    cy2 = (y21 + y22) / 2.0

    px1 = min(max(cx2, x11), x12)
    py1 = min(max(cy2, y11), y12)
    px2 = min(max(px1, x21), x22)
    py2 = min(max(py1, y21), y22)
    px1 = min(max(px2, x11), x12)
    py1 = min(max(py2, y11), y12)
    return (px1, py1), (px2, py2)


def _closest_bbox_boundary_point_pairs(
    bbox_1: Sequence[float],
    bbox_2: Sequence[float],
    *,
    samples_per_edge: int,
    top_k: int,
    min_pixel_distance: float,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    points_1 = _sample_bbox_boundary_points(bbox_1, samples_per_edge)
    points_2 = _sample_bbox_boundary_points(bbox_2, samples_per_edge)
    candidates: List[Tuple[float, Tuple[float, float], Tuple[float, float]]] = []
    min_distance_sq = min_pixel_distance * min_pixel_distance
    for point_1 in points_1:
        for point_2 in points_2:
            distance_sq = (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2
            if distance_sq < min_distance_sq:
                continue
            candidates.append((distance_sq, point_1, point_2))
    candidates.sort(key=lambda item: item[0])

    selected: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    used = set()
    for _distance_sq, point_1, point_2 in candidates:
        key = (round(point_1[0], 1), round(point_1[1], 1), round(point_2[0], 1), round(point_2[1], 1))
        if key in used:
            continue
        used.add(key)
        selected.append((point_1, point_2))
        if len(selected) >= top_k:
            break
    return selected


def _sample_bbox_boundary_points(bbox: Sequence[float], samples_per_edge: int) -> List[Tuple[float, float]]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    count = max(2, samples_per_edge)
    points: List[Tuple[float, float]] = []
    for index in range(count):
        alpha = index / float(count - 1)
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        points.extend([(x, y1), (x, y2), (x1, y), (x2, y)])
    deduped = []
    seen = set()
    for point in points:
        key = (round(point[0], 2), round(point[1], 2))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


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


def _mean_relative_accuracy(prediction: float | None, ground_truth: float | None) -> float:
    """VSI-Bench numeric-answer metric: mean over relative-error thresholds 50%..5%."""
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


def _run_doc(
    *,
    doc_id: int,
    doc: Dict[str, Any],
    args: argparse.Namespace,
    config: SpatialAgentConfig,
    cache_dir: str,
    localizer: LocalizeObjectsTool,
    distance_tool: Get3DDistanceTool,
) -> Dict[str, Any]:
    doc_dir = Path(args.output_dir) / f"doc_{doc_id}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    result_path = doc_dir / "result.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    question = str(doc.get("question") or "")
    objects = _parse_objects(question)
    gt = _parse_float(doc.get("ground_truth"))
    record: Dict[str, Any] = {
        "doc_id": doc_id,
        "question": question,
        "ground_truth": gt,
        "raw_ground_truth": doc.get("ground_truth"),
        "scene_name": doc.get("scene_name"),
        "dataset": doc.get("dataset"),
        "objects": list(objects) if objects else None,
    }
    if not objects:
        record.update({"status": "error", "error": "Failed to parse object pair from question."})
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    reused_frame_records = None
    if args.reuse_frame_results_dir:
        reused_path = Path(args.reuse_frame_results_dir) / f"doc_{doc_id}" / "result.json"
        if reused_path.exists():
            reused_record = json.loads(reused_path.read_text(encoding="utf-8"))
            reused_frame_records = reused_record.get("frame_results") or []
            max_position = max((int(item.get("frame_position", -1)) for item in reused_frame_records), default=-1)
            frames = [""] * (max_position + 1)
            for item in reused_frame_records:
                position = int(item["frame_position"])
                frames[position] = str(item["image_path"])
        else:
            reused_frame_records = None
    if reused_frame_records is None:
        video_path = _resolve_video_path(doc, cache_dir, args)
        record["video_path"] = video_path
        frames = sample_video_frames(video_path=video_path, output_dir=str(doc_dir / "sampled_frames"), num_frames=args.num_frames)
    frame_indices = _selected_frame_positions(len(frames), args)
    if args.method == "mask_pointcloud_multiframe":
        args._current_doc_dir = str(doc_dir)
        if reused_frame_records is not None:
            localization_records = reused_frame_records
        else:
            localization_records = [
                _run_frame(
                    image_path=frames[frame_position],
                    frame_position=frame_position,
                    objects=objects,
                    config=config,
                    localizer=localizer,
                    distance_tool=distance_tool,
                    args=args,
                    localize_only=True,
                )
                for frame_position in frame_indices
            ]
        args._current_objects = objects
        multiframe = _multiframe_mask_pointcloud_distance(
            frames=frames,
            frame_records=localization_records,
            config=config,
            args=args,
        )
        pred = multiframe.get("distance_meters")
        valid_distances = [float(pred)] if isinstance(pred, (int, float)) else []
        best = localization_records[0] if localization_records else {}
        record.update(
            {
                "status": "success" if pred is not None else "error",
                "error": multiframe.get("error"),
                "prediction": pred,
                "candidate_distances": valid_distances,
                "frame_results": localization_records,
                "candidate_distance_results": [multiframe],
                "distance_payload": multiframe,
                "artifacts": [artifact for item in localization_records for artifact in item.get("artifacts", [])],
                **_score(float(pred) if isinstance(pred, (int, float)) else None, gt),
            }
        )
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    if args.top_distance_frames > 0:
        localization_records = [
            _run_frame(
                image_path=frames[frame_position],
                frame_position=frame_position,
                objects=objects,
                config=config,
                localizer=localizer,
                distance_tool=distance_tool,
                args=args,
                localize_only=True,
            )
            for frame_position in frame_indices
        ]
        top_positions = {
            int(item["frame_position"])
            for item in sorted(
                [item for item in localization_records if item.get("status") == "localized"],
                key=lambda item: float(item.get("localization_quality") or 0.0),
                reverse=True,
            )[: args.top_distance_frames]
        }
        frame_records = [
            _run_frame(
                image_path=frames[int(item["frame_position"])],
                frame_position=int(item["frame_position"]),
                objects=objects,
                config=config,
                localizer=localizer,
                distance_tool=distance_tool,
                args=args,
                localize_only=False,
            )
            if int(item.get("frame_position", -1)) in top_positions
            else item
            for item in localization_records
        ]
    else:
        frame_records = []
        for frame_position in frame_indices:
            image_path = frames[frame_position]
            frame_records.append(
                _run_frame(
                    image_path=image_path,
                    frame_position=frame_position,
                    objects=objects,
                    config=config,
                    localizer=localizer,
                    distance_tool=distance_tool,
                    args=args,
                    localize_only=False,
                )
            )

    valid_frame_distances = [
        float(item["prediction"])
        for item in frame_records
        if item.get("status") == "success" and isinstance(item.get("prediction"), (int, float))
    ]
    pred = _aggregate_candidate_distances(valid_frame_distances, args.frame_aggregate)
    best = next((item for item in frame_records if item.get("status") == "success"), frame_records[0] if frame_records else {})
    valid_distances = valid_frame_distances
    record.update(
        {
            "status": "success" if pred is not None else "error",
            "error": best.get("error"),
            "prediction": pred,
            "point_1": best.get("point_1"),
            "point_2": best.get("point_2"),
            "candidate_distances": valid_distances,
            "frame_results": frame_records,
            "candidate_distance_results": best.get("candidate_distance_results"),
            "region_1": best.get("region_1"),
            "region_2": best.get("region_2"),
            "distance_payload": best.get("payload"),
            "artifacts": [artifact for item in frame_records for artifact in item.get("artifacts", [])],
            **_score(float(pred) if isinstance(pred, (int, float)) else None, gt),
        }
    )
    result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _selected_frame_positions(frame_count: int, args: argparse.Namespace) -> List[int]:
    if frame_count <= 0:
        return []
    if args.frame_index >= 0:
        return [min(max(args.frame_index, 0), frame_count - 1)]
    return list(range(frame_count))


def _run_frame(
    *,
    image_path: str,
    frame_position: int,
    objects: Tuple[str, str],
    config: SpatialAgentConfig,
    localizer: LocalizeObjectsTool,
    distance_tool: Get3DDistanceTool,
    args: argparse.Namespace,
    localize_only: bool = False,
) -> Dict[str, Any]:
    localization = localizer.invoke(image=image_path, objects=list(objects))
    frame_record: Dict[str, Any] = {
        "frame_position": frame_position,
        "image_path": image_path,
        "localization_status": localization.get("status"),
        "artifacts": localization.get("artifacts") or [],
    }
    if localization.get("status") != "success":
        frame_record.update({"status": "error", "error": localization.get("error"), "localization": localization})
        return frame_record

    regions = localization.get("payload", {}).get("regions", [])
    region_1 = _best_region(regions, objects[0])
    region_2 = _best_region(regions, objects[1])
    frame_record["has_object_1"] = region_1 is not None
    frame_record["has_object_2"] = region_2 is not None
    frame_record["regions"] = regions
    frame_record["region_1"] = region_1
    frame_record["region_2"] = region_2
    frame_record["object_1_quality"] = float(region_1.get("score") or 0.0) if region_1 else 0.0
    frame_record["object_2_quality"] = float(region_2.get("score") or 0.0) if region_2 else 0.0
    if not region_1 or not region_2:
        if localize_only and (region_1 or region_2):
            frame_record.update({"status": "partially_localized", "prediction": None})
            return frame_record
        frame_record.update(
            {
                "status": "error",
                "error": "Failed to find both localized object regions.",
                "region_1": region_1,
                "region_2": region_2,
            }
        )
        return frame_record

    frame_record["localization_quality"] = _localization_quality(region_1, region_2)
    if localize_only:
        frame_record.update({"status": "localized", "prediction": None})
        return frame_record

    frame_record = _verify_frame_regions(
        frame_record=frame_record,
        image_path=image_path,
        objects=objects,
        args=args,
    )
    region_1 = frame_record.get("region_1")
    region_2 = frame_record.get("region_2")
    if not region_1 or not region_2:
        frame_record.update(
            {
                "status": "error",
                "error": "Detection verifier rejected one or both localized object regions.",
                "prediction": None,
            }
        )
        return frame_record

    distance_results = _run_distance_for_regions(
        image_path=image_path,
        region_1=region_1,
        region_2=region_2,
        config=config,
        distance_tool=distance_tool,
        args=args,
    )
    valid_distances = [
        float(item["distance_meters"])
        for item in distance_results
        if isinstance(item.get("distance_meters"), (int, float))
    ]
    if args.method in {"mask_pointcloud", "bbox_pointcloud"}:
        pred = valid_distances[0] if valid_distances else None
    else:
        pred = _aggregate_candidate_distances(valid_distances, args.distance_aggregate)
    best = distance_results[0] if distance_results else {}
    frame_record.update(
        {
            "status": "success" if pred is not None else "error",
            "error": best.get("error"),
            "prediction": pred,
            "point_1": best.get("point_1"),
            "point_2": best.get("point_2"),
            "region_1": region_1,
            "region_2": region_2,
            "candidate_distances": valid_distances,
            "candidate_distance_results": distance_results,
            "payload": best.get("payload"),
            "artifacts": frame_record["artifacts"] + [artifact for item in distance_results for artifact in item.get("artifacts", [])],
        }
    )
    return frame_record


def _localization_quality(region_1: Dict[str, Any], region_2: Dict[str, Any]) -> float:
    return min(float(region_1.get("score") or 0.0), float(region_2.get("score") or 0.0))


def _select_multiframe_records(frame_records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    by_position = {int(item["frame_position"]): item for item in frame_records}
    manual_positions = _parse_frame_positions(args.selected_frame_positions)
    if manual_positions is not None:
        return [by_position[position] for position in manual_positions if position in by_position]

    selected: Dict[int, Dict[str, Any]] = {}

    same_frame = [item for item in frame_records if item.get("status") == "localized"]
    same_frame = sorted(
        same_frame,
        key=lambda item: float(item.get("localization_quality") or 0.0),
        reverse=True,
    )
    same_limit = args.top_distance_frames if args.top_distance_frames > 0 else len(same_frame)
    for item in same_frame[:same_limit]:
        selected[int(item["frame_position"])] = item

    if args.single_object_frames > 0:
        object_1_frames = sorted(
            [item for item in frame_records if item.get("region_1") is not None],
            key=lambda item: float(item.get("object_1_quality") or 0.0),
            reverse=True,
        )
        object_2_frames = sorted(
            [item for item in frame_records if item.get("region_2") is not None],
            key=lambda item: float(item.get("object_2_quality") or 0.0),
            reverse=True,
        )
        for item in object_1_frames[: args.single_object_frames]:
            selected[int(item["frame_position"])] = item
        for item in object_2_frames[: args.single_object_frames]:
            selected[int(item["frame_position"])] = item

    if args.bridge_frames > 0 and len(selected) >= 2:
        positions = sorted(selected)
        bridge_positions = _sparse_bridge_positions(
            min_position=positions[0],
            max_position=positions[-1],
            existing_positions=set(positions),
            max_count=args.bridge_frames,
        )
        for position in bridge_positions:
            if position in by_position:
                selected[position] = by_position[position]

    selected_items = [selected[position] for position in sorted(selected)]
    if args.max_vggt_frames > 0 and len(selected_items) > args.max_vggt_frames:
        selected_items = _cap_selected_multiframe_records(selected_items, args.max_vggt_frames)
    return selected_items


def _backfill_missing_selected_records(
    *,
    frame_records: Sequence[Dict[str, Any]],
    selected_records: Sequence[Dict[str, Any]],
    objects: Tuple[str, str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    selected_by_position = {int(item["frame_position"]): dict(item) for item in selected_records}
    backfill_summary: Dict[str, Any] = {"added": [], "bridges": [], "missing_before": []}

    for object_index in (1, 2):
        region_key = f"region_{object_index}"
        quality_key = f"object_{object_index}_quality"
        if any(item.get(region_key) is not None for item in selected_by_position.values()):
            continue
        backfill_summary["missing_before"].append(objects[object_index - 1])
        candidates = sorted(
            [item for item in frame_records if item.get(region_key) is not None],
            key=lambda item: float(item.get(quality_key) or 0.0),
            reverse=True,
        )
        limit = max(1, int(getattr(args, "single_object_frames", 1) or 1))
        attempts = max(limit, int(getattr(args, "instance_verifier_recheck_batch_size", 6) or 6))
        accepted = 0
        tried = 0
        for candidate in candidates:
            if accepted >= limit or tried >= attempts:
                break
            position = int(candidate["frame_position"])
            if position in selected_by_position and selected_by_position[position].get(region_key) is not None:
                continue
            tried += 1
            candidate_for_verification = _restore_overlapping_counterpart_region(
                frame_record=candidate,
                object_index=object_index,
                objects=objects,
            )
            verified = _verify_selected_frame_records([candidate_for_verification], objects=objects, args=args)[0]
            if verified.get(region_key) is None:
                backfill_summary.setdefault("rejected", []).append(
                    {
                        "frame_position": position,
                        "object": objects[object_index - 1],
                        "reason": "backfill_verifier_rejected",
                        "detection_verification": verified.get("detection_verification") or [],
                    }
                )
                continue
            selected_by_position[position] = verified
            accepted += 1
            backfill_summary["added"].append(
                {
                    "frame_position": position,
                    "object": objects[object_index - 1],
                    "quality": float(verified.get(quality_key) or 0.0),
                    "attempts": tried,
                }
            )

    if not backfill_summary["added"]:
        return list(selected_records), None

    by_position = {int(item["frame_position"]): item for item in frame_records}
    if args.bridge_frames > 0 and len(selected_by_position) >= 2:
        positions = sorted(selected_by_position)
        bridge_positions = _sparse_bridge_positions(
            min_position=positions[0],
            max_position=positions[-1],
            existing_positions=set(positions),
            max_count=args.bridge_frames,
        )
        for position in bridge_positions:
            if position not in by_position:
                continue
            verified = _verify_selected_frame_records([by_position[position]], objects=objects, args=args)[0]
            selected_by_position[position] = verified
            backfill_summary["bridges"].append(position)

    selected_items = [selected_by_position[position] for position in sorted(selected_by_position)]
    if args.max_vggt_frames > 0 and len(selected_items) > args.max_vggt_frames:
        selected_items = _cap_selected_multiframe_records(selected_items, args.max_vggt_frames)
    return selected_items, backfill_summary


def _restore_overlapping_counterpart_region(
    *,
    frame_record: Dict[str, Any],
    object_index: int,
    objects: Tuple[str, str],
) -> Dict[str, Any]:
    restored = dict(frame_record)
    current_key = f"region_{object_index}"
    other_index = 2 if object_index == 1 else 1
    other_key = f"region_{other_index}"
    if restored.get(current_key) is None or restored.get(other_key) is not None:
        return restored

    current_region = restored[current_key]
    other_name = objects[other_index - 1].lower()
    for region in restored.get("regions") or []:
        if str(region.get("label", "")).lower() != other_name:
            continue
        if not _bbox_close(current_region.get("bbox") or [], region.get("bbox") or [], eps=2.0) and _bbox_iou(
            current_region.get("bbox") or [], region.get("bbox") or []
        ) < 0.98:
            continue
        restored[other_key] = dict(region)
        restored[f"has_object_{other_index}"] = True
        restored[f"object_{other_index}_quality"] = float(region.get("score") or 0.0)
        restored["status"] = _status_after_instance_filter(restored)
        restored.setdefault("backfill_conflict_context", []).append(
            {
                "current_object": objects[object_index - 1],
                "counterpart_object": objects[other_index - 1],
                "counterpart_bbox": region.get("bbox"),
                "counterpart_score": float(region.get("score") or 0.0),
            }
        )
        return restored
    return restored


def _verify_object_instances(
    *,
    frame_records: Sequence[Dict[str, Any]],
    objects: Tuple[str, str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = [dict(item) for item in frame_records]
    by_position = {int(item["frame_position"]): item for item in records}
    summary: Dict[str, Any] = {"objects": {}, "artifacts": []}
    for object_index, object_name in enumerate(objects, start=1):
        region_key = f"region_{object_index}"
        quality_key = f"object_{object_index}_quality"
        has_key = f"has_object_{object_index}"
        candidates = _collect_instance_candidates(records, object_name=object_name, args=args)
        hypotheses = _cluster_instance_candidates(candidates, args=args)
        verified_hypotheses = []
        accepted_keys: set[Tuple[int, int]] = set()
        accepted_by_position: Dict[int, Dict[str, Any]] = {}
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            verdict = _verify_instance_hypothesis(
                object_name=object_name,
                hypothesis_index=hypothesis_index,
                candidates=hypothesis,
                args=args,
            )
            verified_hypotheses.append(
                {
                    "hypothesis_id": hypothesis_index,
                    "verdict": verdict.get("verdict"),
                    "reason_code": verdict.get("reason_code"),
                    "corrected_label": verdict.get("corrected_label"),
                    "accepted_candidate_ids": verdict.get("accepted_candidate_ids") or [],
                    "rejected_candidate_ids": verdict.get("rejected_candidate_ids") or [],
                    "candidate_count": len(hypothesis),
                    "frame_positions": [int(item["frame_position"]) for item in hypothesis],
                    "scores": [float(item["region"].get("score") or 0.0) for item in hypothesis],
                    "contact_sheet": verdict.get("contact_sheet"),
                    "raw_response": verdict.get("raw_response"),
                    "error": verdict.get("error"),
                }
            )
            if verdict.get("contact_sheet"):
                summary["artifacts"].append(verdict["contact_sheet"])
            if verdict.get("accepted_candidates") is not None:
                accepted_items = verdict.get("accepted_candidates") or []
            elif verdict.get("verdict") == "accept":
                accepted_items = list(hypothesis)
            else:
                accepted_items = []
            for item in accepted_items:
                position = int(item["frame_position"])
                accepted_keys.add((position, int(item["candidate_index"])))
                current = accepted_by_position.get(position)
                if current is None or float(item["region"].get("score") or 0.0) > float(current.get("score") or 0.0):
                    accepted_by_position[position] = dict(item["region"])

        fallback_rechecks = []
        if not accepted_by_position and getattr(args, "enable_instance_verifier_recheck", False):
            fallback_rechecks, fallback_accepted = _recheck_rejected_instance_candidates(
                object_name=object_name,
                candidates=candidates,
                args=args,
            )
            for recheck in fallback_rechecks:
                if recheck.get("contact_sheet"):
                    summary["artifacts"].append(recheck["contact_sheet"])
            for item in fallback_accepted:
                position = int(item["frame_position"])
                accepted_keys.add((position, int(item["candidate_index"])))
                current = accepted_by_position.get(position)
                if current is None or float(item["region"].get("score") or 0.0) > float(current.get("score") or 0.0):
                    accepted_by_position[position] = dict(item["region"])

        best_match_fallback = None
        if not accepted_by_position and getattr(args, "enable_instance_verifier_best_match_fallback", False):
            best_match_fallback, best_match_accepted = _select_best_match_instance_candidate(
                object_name=object_name,
                candidates=candidates,
                args=args,
            )
            if best_match_fallback and best_match_fallback.get("contact_sheet"):
                summary["artifacts"].append(best_match_fallback["contact_sheet"])
            for item in best_match_accepted:
                position = int(item["frame_position"])
                accepted_keys.add((position, int(item["candidate_index"])))
                current = accepted_by_position.get(position)
                if current is None or float(item["region"].get("score") or 0.0) > float(current.get("score") or 0.0):
                    region = dict(item["region"])
                    region["_best_match_fallback"] = True
                    accepted_by_position[position] = region

        for item in records:
            position = int(item["frame_position"])
            accepted_region = accepted_by_position.get(position)
            if accepted_region is None:
                item[region_key] = None
                item[quality_key] = 0.0
                item[has_key] = False
                item["status"] = _status_after_instance_filter(item)
            else:
                accepted_region.pop("_instance_candidate_index", None)
                accepted_region.pop("sheet_candidate_id", None)
                item[region_key] = accepted_region
                item[quality_key] = float(accepted_region.get("score") or 0.0)
                item[has_key] = True
        summary["objects"][object_name] = {
            "candidate_count": len(candidates),
            "hypothesis_count": len(hypotheses),
            "accepted_hypotheses": sum(1 for item in verified_hypotheses if item.get("verdict") == "accept"),
            "fallback_rechecks": fallback_rechecks,
            "best_match_fallback": best_match_fallback,
            "hypotheses": verified_hypotheses,
        }

    for item in records:
        region_1 = item.get("region_1")
        region_2 = item.get("region_2")
        if region_1 and region_2:
            item["localization_quality"] = _localization_quality(region_1, region_2)
            item["status"] = "localized"
        elif region_1 or region_2:
            item["localization_quality"] = 0.0
            item["status"] = "partially_localized"
        else:
            item["localization_quality"] = 0.0
            item["status"] = "error"
            item["error"] = "Instance verifier rejected all localized object regions."
    return records, summary


def _collect_instance_candidates(
    records: Sequence[Dict[str, Any]],
    *,
    object_name: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    candidates = []
    min_score = float(args.instance_verifier_min_score)
    max_candidates = int(args.instance_verifier_max_candidates)
    for item in records:
        position = int(item["frame_position"])
        image_path = str(item.get("image_path") or "")
        regions = item.get("regions") or []
        for candidate_index, region in enumerate(regions):
            if str(region.get("label", "")).lower() != object_name.lower():
                continue
            if float(region.get("score") or 0.0) < min_score:
                continue
            region = dict(region)
            region["_instance_candidate_index"] = candidate_index
            candidates.append(
                {
                    "frame_position": position,
                    "image_path": image_path,
                    "region": region,
                    "candidate_index": candidate_index,
                }
            )
    candidates = sorted(candidates, key=lambda item: float(item["region"].get("score") or 0.0), reverse=True)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    candidates = sorted(candidates, key=lambda item: int(item["frame_position"]))

    by_position = {int(item["frame_position"]): item for item in records}
    for candidate in candidates:
        position = int(candidate["frame_position"])
        frame_record = by_position.get(position)
        if not frame_record:
            continue
        for key in ("region_1", "region_2"):
            region = frame_record.get(key)
            if (
                region
                and str(region.get("label", "")).lower() == object_name.lower()
                and _bbox_close(region.get("bbox") or [], candidate["region"].get("bbox") or [], eps=1e-3)
            ):
                region["_instance_candidate_index"] = int(candidate["candidate_index"])
    return candidates


def _bbox_close(a: Sequence[float], b: Sequence[float], *, eps: float) -> bool:
    return len(a) >= 4 and len(b) >= 4 and all(abs(float(a[i]) - float(b[i])) <= eps for i in range(4))


def _cluster_instance_candidates(candidates: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[List[Dict[str, Any]]]:
    if not candidates:
        return []
    hypotheses: List[List[Dict[str, Any]]] = []
    for candidate in candidates:
        best_index = None
        best_cost = float("inf")
        for index, hypothesis in enumerate(hypotheses):
            if len(hypothesis) >= int(args.instance_verifier_max_cluster_frames):
                continue
            prev = hypothesis[-1]
            frame_gap = abs(int(candidate["frame_position"]) - int(prev["frame_position"]))
            if frame_gap > int(args.instance_verifier_max_frame_gap):
                continue
            cost = _normalized_bbox_center_distance(candidate["region"].get("bbox") or [], prev["region"].get("bbox") or [])
            if cost < best_cost:
                best_cost = cost
                best_index = index
        if best_index is not None and best_cost <= float(args.instance_verifier_center_threshold):
            hypotheses[best_index].append(candidate)
        else:
            hypotheses.append([candidate])
    return sorted(hypotheses, key=lambda item: (len(item), max(float(x["region"].get("score") or 0.0) for x in item)), reverse=True)


def _normalized_bbox_center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return float("inf")
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    aw = max(1.0, float(a[2]) - float(a[0]))
    ah = max(1.0, float(a[3]) - float(a[1]))
    bw = max(1.0, float(b[2]) - float(b[0]))
    bh = max(1.0, float(b[3]) - float(b[1]))
    scale = max(1.0, (aw + ah + bw + bh) / 4.0)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) / scale


def _verify_instance_hypothesis(
    *,
    object_name: str,
    hypothesis_index: int,
    candidates: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    sheet_kind: str = "hypothesis",
) -> Dict[str, Any]:
    contact_sheet = _build_instance_contact_sheet(
        object_name=object_name,
        hypothesis_index=hypothesis_index,
        candidates=candidates,
        args=args,
        sheet_kind=sheet_kind,
    )
    try:
        raw = _call_instance_verifier_llm(
            object_name=object_name,
            contact_sheet=contact_sheet,
            candidates=candidates,
            args=args,
        )
        parsed = _parse_verifier_json(raw)
        verdict = str(parsed.get("verdict") or "").strip().lower()
        if verdict not in {"accept", "reject"}:
            raise ValueError(f"Invalid instance verifier verdict: {verdict!r}")
        accepted_ids = _parse_candidate_id_list(parsed.get("accepted_candidate_ids"))
        rejected_ids = _parse_candidate_id_list(parsed.get("rejected_candidate_ids"))
        candidate_by_id = {str(item.get("sheet_candidate_id")): item for item in candidates}
        accepted_candidates = [candidate_by_id[candidate_id] for candidate_id in accepted_ids if candidate_id in candidate_by_id]
        if verdict == "accept" and not accepted_ids:
            accepted_candidates = list(candidates)
        return {
            "agent": "InstanceVerifierAgent",
            "verdict": verdict,
            "reason_code": str(parsed.get("reason_code") or "instance_verifier"),
            "corrected_label": parsed.get("corrected_label"),
            "accepted_candidate_ids": accepted_ids,
            "rejected_candidate_ids": rejected_ids,
            "accepted_candidates": accepted_candidates,
            "raw_response": raw,
            "contact_sheet": str(contact_sheet),
        }
    except Exception as exc:
        return {
            "agent": "InstanceVerifierAgent",
            "verdict": "accept" if args.verifier_on_error == "accept" else "reject",
            "reason_code": "instance_verifier_error",
            "error": str(exc),
            "accepted_candidate_ids": [],
            "rejected_candidate_ids": [],
            "accepted_candidates": list(candidates) if args.verifier_on_error == "accept" else [],
            "contact_sheet": str(contact_sheet),
        }


def _parse_candidate_id_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip() for value in value.split(",") if value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _recheck_rejected_instance_candidates(
    *,
    object_name: str,
    candidates: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    max_rounds = int(args.instance_verifier_recheck_rounds)
    batch_size = int(args.instance_verifier_recheck_batch_size)
    if max_rounds <= 0 or batch_size <= 0:
        return [], []
    ordered = sorted(candidates, key=lambda item: float(item["region"].get("score") or 0.0), reverse=True)
    rechecks = []
    accepted: List[Dict[str, Any]] = []
    for round_index in range(max_rounds):
        start = round_index * batch_size
        batch = ordered[start : start + batch_size]
        if not batch:
            break
        verdict = _verify_instance_hypothesis(
            object_name=object_name,
            hypothesis_index=1000 + round_index,
            candidates=batch,
            args=args,
            sheet_kind="recheck",
        )
        rechecks.append(
            {
                "round": round_index,
                "verdict": verdict.get("verdict"),
                "reason_code": verdict.get("reason_code"),
                "corrected_label": verdict.get("corrected_label"),
                "accepted_candidate_ids": verdict.get("accepted_candidate_ids") or [],
                "rejected_candidate_ids": verdict.get("rejected_candidate_ids") or [],
                "frame_positions": [int(item["frame_position"]) for item in batch],
                "scores": [float(item["region"].get("score") or 0.0) for item in batch],
                "contact_sheet": verdict.get("contact_sheet"),
                "raw_response": verdict.get("raw_response"),
                "error": verdict.get("error"),
            }
        )
        accepted.extend(verdict.get("accepted_candidates") or [])
        if accepted and not getattr(args, "instance_verifier_recheck_all_rounds", False):
            break
    return rechecks, accepted


def _select_best_match_instance_candidate(
    *,
    object_name: str,
    candidates: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    if not candidates:
        return None, []
    limit = max(1, int(getattr(args, "instance_verifier_best_match_candidates", 12) or 12))
    batch = sorted(candidates, key=lambda item: float(item["region"].get("score") or 0.0), reverse=True)[:limit]
    verdict = _verify_best_match_instance_candidate(
        object_name=object_name,
        candidates=batch,
        args=args,
    )
    accepted = verdict.get("accepted_candidates") or []
    summary = {
        "verdict": verdict.get("verdict"),
        "reason_code": verdict.get("reason_code"),
        "corrected_label": verdict.get("corrected_label"),
        "selected_candidate_id": verdict.get("selected_candidate_id"),
        "candidate_count": len(batch),
        "frame_positions": [int(item["frame_position"]) for item in batch],
        "scores": [float(item["region"].get("score") or 0.0) for item in batch],
        "contact_sheet": verdict.get("contact_sheet"),
        "raw_response": verdict.get("raw_response"),
        "error": verdict.get("error"),
    }
    return summary, accepted


def _verify_best_match_instance_candidate(
    *,
    object_name: str,
    candidates: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    contact_sheet = _build_instance_contact_sheet(
        object_name=object_name,
        hypothesis_index=2000,
        candidates=candidates,
        args=args,
        sheet_kind="best_match",
    )
    try:
        raw = _call_best_match_verifier_llm(
            object_name=object_name,
            contact_sheet=contact_sheet,
            candidates=candidates,
            args=args,
        )
        parsed = _parse_verifier_json(raw)
        selected_id = str(parsed.get("selected_candidate_id") or "").strip()
        candidate_by_id = {str(item.get("sheet_candidate_id")): item for item in candidates}
        selected = candidate_by_id.get(selected_id)
        if selected is None:
            selected = max(candidates, key=lambda item: float(item["region"].get("score") or 0.0))
            selected_id = str(selected.get("sheet_candidate_id") or "")
        return {
            "agent": "InstanceBestMatchFallbackAgent",
            "verdict": "accept",
            "reason_code": str(parsed.get("reason_code") or "best_match_fallback"),
            "corrected_label": parsed.get("corrected_label"),
            "selected_candidate_id": selected_id,
            "accepted_candidates": [selected],
            "raw_response": raw,
            "contact_sheet": str(contact_sheet),
        }
    except Exception as exc:
        selected = max(candidates, key=lambda item: float(item["region"].get("score") or 0.0))
        return {
            "agent": "InstanceBestMatchFallbackAgent",
            "verdict": "accept",
            "reason_code": "best_match_fallback_error_score_top1",
            "error": str(exc),
            "selected_candidate_id": selected.get("sheet_candidate_id"),
            "accepted_candidates": [selected],
            "contact_sheet": str(contact_sheet),
        }


def _build_instance_contact_sheet(
    *,
    object_name: str,
    hypothesis_index: int,
    candidates: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    sheet_kind: str = "hypothesis",
) -> Path:
    from PIL import Image, ImageDraw

    doc_dir = Path(getattr(args, "_current_doc_dir", args.output_dir))
    output_dir = doc_dir / "instance_verifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _representative_instance_candidates(candidates, int(args.instance_verifier_contact_frames))
    tile_w = int(args.instance_verifier_tile_size)
    tile_h = int(args.instance_verifier_tile_size)
    gap = int(args.instance_verifier_tile_gap)
    columns = max(1, int(args.instance_verifier_grid_cols))
    rows = max(1, math.ceil(len(selected) / columns))
    sheet_w = columns * tile_w + (columns + 1) * gap
    sheet_h = rows * tile_h + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(selected):
        item["sheet_candidate_id"] = f"c{index + 1}"
        image = Image.open(item["image_path"]).convert("RGB")
        original_w, original_h = image.size
        scale = min(tile_w / float(original_w), tile_h / float(original_h))
        resized = image.resize((max(1, int(original_w * scale)), max(1, int(original_h * scale))))
        row = index // columns
        col = index % columns
        tile_x = gap + col * (tile_w + gap)
        tile_y = gap + row * (tile_h + gap)
        draw.rectangle([tile_x - 1, tile_y - 1, tile_x + tile_w, tile_y + tile_h], outline=(210, 210, 210), width=2)
        x_offset = tile_x + (tile_w - resized.width) // 2
        y_offset = tile_y + (tile_h - resized.height) // 2
        sheet.paste(resized, (x_offset, y_offset))
        bbox = [float(value) * scale for value in item["region"].get("bbox", [])[:4]]
        bbox = [bbox[0] + x_offset, bbox[1] + y_offset, bbox[2] + x_offset, bbox[3] + y_offset]
        draw.rectangle(bbox, outline=(255, 0, 0), width=4)
        if sheet_kind == "recheck":
            prefix = f"{item['sheet_candidate_id']} recheck r{hypothesis_index - 1000}"
        elif sheet_kind == "best_match":
            prefix = f"{item['sheet_candidate_id']} best"
        else:
            prefix = f"{item['sheet_candidate_id']} h{hypothesis_index}"
        label = f"{prefix} f{item['frame_position']} {object_name} {float(item['region'].get('score') or 0.0):.2f}"
        label_x = max(tile_x + 2, min(bbox[0], tile_x + tile_w - 180))
        label_y = max(tile_y + 2, bbox[1] - 24)
        try:
            text_box = draw.textbbox((label_x, label_y), label)
            draw.rectangle([text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2], fill=(255, 255, 255), outline=(255, 0, 0))
        except Exception:
            draw.rectangle([label_x - 3, label_y - 2, label_x + 210, label_y + 18], fill=(255, 255, 255), outline=(255, 0, 0))
        draw.text((label_x, label_y), label, fill=(255, 0, 0))
    if sheet_kind == "recheck":
        path = output_dir / f"{object_name}_recheck_{hypothesis_index - 1000:02d}.jpg"
    elif sheet_kind == "best_match":
        path = output_dir / f"{object_name}_best_match_fallback.jpg"
    else:
        path = output_dir / f"{object_name}_hypothesis_{hypothesis_index:02d}.jpg"
    sheet.save(path, quality=90)
    return path


def _representative_instance_candidates(candidates: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(candidates) <= limit:
        return list(candidates)
    picks: Dict[int, Dict[str, Any]] = {}
    sorted_by_score = sorted(candidates, key=lambda item: float(item["region"].get("score") or 0.0), reverse=True)
    for item in sorted_by_score[: max(1, limit // 2)]:
        picks[int(item["frame_position"])] = item
    picks[int(candidates[0]["frame_position"])] = candidates[0]
    picks[int(candidates[-1]["frame_position"])] = candidates[-1]
    remaining = sorted_by_score
    for item in remaining:
        if len(picks) >= limit:
            break
        picks[int(item["frame_position"])] = item
    return [picks[position] for position in sorted(picks)]


def _call_instance_verifier_llm(
    *,
    object_name: str,
    contact_sheet: Path,
    candidates: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    candidate_lines = []
    for item in candidates:
        candidate_id = item.get("sheet_candidate_id")
        if not candidate_id:
            continue
        candidate_lines.append(
            f"{candidate_id}: frame={int(item['frame_position'])}, detector_score={float(item['region'].get('score') or 0.0):.3f}"
        )
    candidate_text = "; ".join(candidate_lines)
    confusion_hint = _detector_confusion_hint()
    object_hint = _verifier_object_hint(object_name)
    target_list_hint = _target_object_list_hint(args)
    is_recheck = "_recheck_" in contact_sheet.name
    if is_recheck:
        prompt = (
            "You are InstanceVerifierAgent. The contact sheet shows independent fallback candidate boxes, not one instance. "
            "Judge every candidate id independently. Candidates do not need to be the same object instance. "
            "For each candidate id visible in the sheet, decide whether the object category inside that red box matches the requested object. "
            "Accept a candidate when the category is correct, even if the object is occluded, cropped, partly outside the box, small, or only partially visible. "
            "Do not require a complete or clearly unobstructed object. Reject only candidates whose visible content is a different category, pure background, or impossible to identify as the requested object. "
            f"{target_list_hint}"
            f"{confusion_hint}"
            f"{object_hint}"
            "Do not use detector scores as evidence. Return only compact JSON with keys: "
            "verdict ('accept' if at least one candidate is accepted, otherwise 'reject'), accepted_candidate_ids, rejected_candidate_ids, reason_code, corrected_label. "
            f"Requested object: {object_name}. Candidate ids: {candidate_text}."
        )
    else:
        prompt = (
            "You are InstanceVerifierAgent. The contact sheet shows one cross-frame object instance hypothesis. "
            "All red boxes are proposed to be the same requested object instance. "
            "The hypothesis may be mixed: some boxes can be correct and some can be wrong. "
            "For each candidate id visible in the sheet, decide whether the object category inside that red box matches the requested object. "
            "Accept a candidate when the category is correct, even if the object is occluded, cropped, partly outside the box, small, or only partially visible. "
            "Do not require a complete or clearly unobstructed object. Reject only candidates whose visible content is a different category, pure background, or impossible to identify as the requested object. "
            f"{target_list_hint}"
            f"{confusion_hint}"
            f"{object_hint}"
            "Do not use detector scores as evidence. Return only compact JSON with keys: "
            "verdict ('accept' if at least one candidate is accepted, otherwise 'reject'), accepted_candidate_ids, rejected_candidate_ids, reason_code, corrected_label. "
            f"Requested object: {object_name}. Candidate ids: {candidate_text}."
        )
    return _call_verifier_vl_model(prompt=prompt, image_path=str(contact_sheet), args=args, max_tokens=256)


def _call_best_match_verifier_llm(
    *,
    object_name: str,
    contact_sheet: Path,
    candidates: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    candidate_lines = []
    for item in candidates:
        candidate_id = item.get("sheet_candidate_id")
        if not candidate_id:
            continue
        candidate_lines.append(
            f"{candidate_id}: frame={int(item['frame_position'])}, detector_score={float(item['region'].get('score') or 0.0):.3f}"
        )
    candidate_text = "; ".join(candidate_lines)
    confusion_hint = _detector_confusion_hint()
    object_hint = _verifier_object_hint(object_name)
    target_list_hint = _target_object_list_hint(args)
    prompt = (
        "You are InstanceBestMatchFallbackAgent. This fallback is used only after the normal verifier rejected every candidate for a required object. "
        "The object detector is high-recall, so among these candidates there is often at least one true or partially visible target object. "
        "You must choose exactly one candidate id that best matches the requested object. Do not reject all candidates. "
        "Choose the candidate whose red box is most likely to contain the requested object category, including occluded, cropped, small, or partially visible cases. "
        "Use visual evidence first; detector scores are only tie-breakers when two candidates look equally plausible. "
        "If none is perfect, choose the least-wrong / most semantically similar candidate from the current question target list. "
        f"{target_list_hint}"
        f"{confusion_hint}"
        f"{object_hint}"
        "Return only compact JSON with keys: selected_candidate_id, reason_code, corrected_label. "
        f"Requested object: {object_name}. Candidate ids: {candidate_text}."
    )
    return _call_verifier_vl_model(prompt=prompt, image_path=str(contact_sheet), args=args, max_tokens=128)


def _call_verifier_vl_model(*, prompt: str, image_path: str, args: argparse.Namespace, max_tokens: int) -> str:
    backend = str(getattr(args, "verifier_backend", "openai") or "openai").lower()
    if backend == "ollama_generate":
        return _call_ollama_generate(prompt=prompt, image_path=image_path, args=args)
    if backend != "openai":
        raise RuntimeError(f"Unsupported verifier backend: {backend}")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for OpenAI verifier backend.") from exc

    api_key = os.getenv(str(args.verifier_api_key_env))
    if not api_key:
        raise RuntimeError(f"Missing verifier API key env: {args.verifier_api_key_env}")
    client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": int(args.verifier_timeout)}
    api_base_url = args.verifier_api_base_url or os.getenv(str(args.verifier_api_base_url_env))
    if api_base_url:
        client_kwargs["base_url"] = api_base_url

    encoded = _encoded_image_data_url(image_path, max_side=int(args.verifier_max_image_side))
    response = OpenAI(**client_kwargs).chat.completions.create(
        model=str(args.verifier_model),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encoded}},
                ],
            }
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return str(response.choices[0].message.content or "")


def _call_ollama_generate(*, prompt: str, image_path: str, args: argparse.Namespace) -> str:
    import urllib.request

    image_b64 = _encoded_image_base64(image_path, max_side=int(args.verifier_max_image_side))
    payload = {
        "model": str(args.verifier_model),
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        str(args.verifier_ollama_url),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=int(args.verifier_timeout)) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    return str(parsed.get("response") or "")


def _encoded_image_data_url(image_path: str, *, max_side: int) -> str:
    encoded = _encoded_image_base64(image_path, max_side=max_side)
    return f"data:image/jpeg;base64,{encoded}"


def _encoded_image_base64(image_path: str, *, max_side: int) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_side > 0:
        scale = max_side / float(max(image.size))
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _status_after_instance_filter(item: Dict[str, Any]) -> str:
    if item.get("region_1") and item.get("region_2"):
        return "localized"
    if item.get("region_1") or item.get("region_2"):
        return "partially_localized"
    return "error"


def _sparse_bridge_positions(
    *,
    min_position: int,
    max_position: int,
    existing_positions: set[int],
    max_count: int,
) -> List[int]:
    if max_count <= 0 or max_position <= min_position + 1:
        return []
    span = max_position - min_position
    candidates = []
    for index in range(1, max_count + 1):
        position = int(round(min_position + span * index / float(max_count + 1)))
        if position <= min_position or position >= max_position or position in existing_positions:
            continue
        candidates.append(position)
    deduped = []
    seen = set()
    for position in candidates:
        if position in seen:
            continue
        seen.add(position)
        deduped.append(position)
    return deduped


def _cap_selected_multiframe_records(records: Sequence[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    if len(records) <= max_count:
        return list(records)
    scored = []
    for index, item in enumerate(records):
        if item.get("status") == "localized":
            score = 3.0 + float(item.get("localization_quality") or 0.0)
        elif item.get("region_1") is not None or item.get("region_2") is not None:
            score = 2.0 + max(float(item.get("object_1_quality") or 0.0), float(item.get("object_2_quality") or 0.0))
        else:
            score = 1.0
        scored.append((score, index, item))
    keep_indices = {index for _score, index, _item in sorted(scored, reverse=True)[:max_count]}
    return [item for index, item in enumerate(records) if index in keep_indices]


def _parse_frame_positions(value: str) -> List[int] | None:
    if not value.strip():
        return None
    positions = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        positions.append(int(chunk))
    return positions


def _run_distance_for_regions(
    *,
    image_path: str,
    region_1: Dict[str, Any],
    region_2: Dict[str, Any],
    config: SpatialAgentConfig,
    distance_tool: Get3DDistanceTool,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if args.method == "mask_pointcloud":
        return [
            _mask_pointcloud_distance(
                image_path=image_path,
                bbox_1=region_1["bbox"],
                bbox_2=region_2["bbox"],
                config=config,
                max_points=args.mask_max_points,
                aggregate=args.pointcloud_aggregate,
            )
        ]
    if args.method == "bbox_pointcloud":
        return [
            _bbox_pointcloud_distance(
                image_path=image_path,
                bbox_1=region_1["bbox"],
                bbox_2=region_2["bbox"],
                config=config,
                grid_size=args.pointcloud_grid,
                aggregate=args.pointcloud_aggregate,
            )
        ]

    point_pairs = _closest_bbox_boundary_point_pairs(
        region_1["bbox"],
        region_2["bbox"],
        samples_per_edge=args.boundary_samples,
        top_k=args.candidate_pairs,
        min_pixel_distance=args.min_pixel_distance,
    )
    if not point_pairs:
        point_pairs = [_closest_bbox_points(region_1["bbox"], region_2["bbox"])]

    distance_results = []
    for point_1, point_2 in point_pairs:
        distance = distance_tool.invoke(image=image_path, point_1=list(point_1), point_2=list(point_2))
        payload = distance.get("payload") or {}
        distance_results.append(
            {
                "status": distance.get("status"),
                "error": distance.get("error"),
                "point_1": list(point_1),
                "point_2": list(point_2),
                "distance_meters": payload.get("distance_meters"),
                "payload": payload,
                "artifacts": distance.get("artifacts") or [],
            }
        )
    return distance_results


def _aggregate_candidate_distances(values: List[float], mode: str) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if mode == "min":
        return ordered[0]
    if mode == "p10":
        return ordered[max(0, min(len(ordered) - 1, round((len(ordered) - 1) * 0.10)))]
    if mode == "p25":
        return ordered[max(0, min(len(ordered) - 1, round((len(ordered) - 1) * 0.25)))]
    if mode == "median":
        return ordered[len(ordered) // 2]
    raise ValueError(f"Unknown distance aggregate: {mode}")


def _bbox_pointcloud_distance(
    *,
    image_path: str,
    bbox_1: Sequence[float],
    bbox_2: Sequence[float],
    config: SpatialAgentConfig,
    grid_size: int,
    aggregate: str,
) -> Dict[str, Any]:
    try:
        import numpy as np
        import torch
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "distance_meters": None}

    settings = get_tool_settings(config, "Get3DDistance", aliases=["distance", "3d_distance"])
    camera_settings = get_tool_settings(config, "GetCameraParametersVGGT", aliases=["camera", "vggt"])
    preprocess_mode = str(settings.get("preprocess_mode") or camera_settings.get("preprocess_mode", "pad"))
    device = resolve_device(settings.get("device") or camera_settings.get("device"))
    try:
        backend = get_vggt_backend(
            model_id=str(
                settings.get("hf_model_id")
                or settings.get("vggt_model_id")
                or camera_settings.get("hf_model_id")
                or "facebook/VGGT-1B"
            ),
            checkpoint_path=(
                settings.get("checkpoint_path")
                or settings.get("vggt_checkpoint_path")
                or camera_settings.get("checkpoint_path")
            ),
            device=device,
        )
        image = load_pil_image(image_path)
        images = backend["load_and_preprocess_images"]([image_path], mode=preprocess_mode).to(device)
        with backend["torch"].no_grad():
            predictions = backend["model"](images)
        frame_points = _extract_single_frame_points(predictions)
        if frame_points is None:
            return {
                "status": "error",
                "error": "VGGT predictions did not include world points.",
                "distance_meters": None,
            }
        points_1 = _bbox_points_to_3d(
            bbox=bbox_1,
            image_size=image.size,
            frame_points=frame_points,
            preprocess_mode=preprocess_mode,
            grid_size=grid_size,
        )
        points_2 = _bbox_points_to_3d(
            bbox=bbox_2,
            image_size=image.size,
            frame_points=frame_points,
            preprocess_mode=preprocess_mode,
            grid_size=grid_size,
        )
        if len(points_1) == 0 or len(points_2) == 0:
            return {
                "status": "error",
                "error": "No finite 3D points were sampled inside one or both bboxes.",
                "distance_meters": None,
                "pointcloud_sizes": [int(len(points_1)), int(len(points_2))],
            }

        distances = np.linalg.norm(points_1[:, None, :] - points_2[None, :, :], axis=2).reshape(-1)
        distances = distances[np.isfinite(distances)]
        if distances.size == 0:
            return {"status": "error", "error": "No finite pairwise distances.", "distance_meters": None}
        aggregate_policy = _aggregate_np_policy(distances, aggregate)
        value = aggregate_policy["value"]
        return {
            "status": "success",
            "error": None,
            "distance_meters": float(value),
            "method": "bbox_pointcloud",
            "aggregate": aggregate,
            "selected_aggregate": aggregate_policy["selected_aggregate"],
            "aggregate_policy": aggregate_policy,
            "pointcloud_sizes": [int(len(points_1)), int(len(points_2))],
            "distance_stats": {
                "min": float(np.min(distances)),
                "p05": float(np.quantile(distances, 0.05)),
                "p10": float(np.quantile(distances, 0.10)),
                "p25": float(np.quantile(distances, 0.25)),
                "median": float(np.median(distances)),
                "p75": float(np.quantile(distances, 0.75)),
                "p80": float(np.quantile(distances, 0.80)),
                "p85": float(np.quantile(distances, 0.85)),
                "p90": float(np.quantile(distances, 0.90)),
            },
            "payload": {},
            "artifacts": [],
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "distance_meters": None}
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _mask_pointcloud_distance(
    *,
    image_path: str,
    bbox_1: Sequence[float],
    bbox_2: Sequence[float],
    config: SpatialAgentConfig,
    max_points: int,
    aggregate: str,
) -> Dict[str, Any]:
    try:
        import numpy as np
        import torch
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "distance_meters": None}

    settings = get_tool_settings(config, "Get3DDistance", aliases=["distance", "3d_distance"])
    camera_settings = get_tool_settings(config, "GetCameraParametersVGGT", aliases=["camera", "vggt"])
    mask_settings = get_tool_settings(config, "GetObjectMask", aliases=["mask", "sam2"])
    preprocess_mode = str(settings.get("preprocess_mode") or camera_settings.get("preprocess_mode", "pad"))
    device = resolve_device(settings.get("device") or camera_settings.get("device") or mask_settings.get("device"))
    try:
        image_rgb = load_rgb_array(image_path)
        predictor = get_sam2_predictor(
            model_id=str(mask_settings.get("model_id", "facebook/sam2.1-hiera-large")),
            checkpoint_path=mask_settings.get("checkpoint_path"),
            config_path=mask_settings.get("config_path"),
            device=device,
        )
        predictor.set_image(image_rgb)
        mask_1 = _predict_sam_mask(predictor, bbox_1)
        mask_2 = _predict_sam_mask(predictor, bbox_2)
        if mask_1 is None or mask_2 is None:
            return {"status": "error", "error": "SAM2 returned an empty mask.", "distance_meters": None}

        backend = get_vggt_backend(
            model_id=str(
                settings.get("hf_model_id")
                or settings.get("vggt_model_id")
                or camera_settings.get("hf_model_id")
                or "facebook/VGGT-1B"
            ),
            checkpoint_path=(
                settings.get("checkpoint_path")
                or settings.get("vggt_checkpoint_path")
                or camera_settings.get("checkpoint_path")
            ),
            device=device,
        )
        image = load_pil_image(image_path)
        images = backend["load_and_preprocess_images"]([image_path], mode=preprocess_mode).to(device)
        with backend["torch"].no_grad():
            predictions = backend["model"](images)
        frame_points = _extract_single_frame_points(predictions)
        if frame_points is None:
            return {"status": "error", "error": "VGGT predictions did not include world points.", "distance_meters": None}

        points_1 = _mask_pixels_to_3d(
            mask=mask_1,
            image_size=image.size,
            frame_points=frame_points,
            preprocess_mode=preprocess_mode,
            max_points=max_points,
        )
        points_2 = _mask_pixels_to_3d(
            mask=mask_2,
            image_size=image.size,
            frame_points=frame_points,
            preprocess_mode=preprocess_mode,
            max_points=max_points,
        )
        if len(points_1) == 0 or len(points_2) == 0:
            return {
                "status": "error",
                "error": "No finite 3D points were sampled inside one or both masks.",
                "distance_meters": None,
                "pointcloud_sizes": [int(len(points_1)), int(len(points_2))],
            }
        distances = np.linalg.norm(points_1[:, None, :] - points_2[None, :, :], axis=2).reshape(-1)
        distances = distances[np.isfinite(distances)]
        if distances.size == 0:
            return {"status": "error", "error": "No finite pairwise distances.", "distance_meters": None}
        aggregate_policy = _aggregate_np_policy(distances, aggregate)
        value = aggregate_policy["value"]
        return {
            "status": "success",
            "error": None,
            "distance_meters": float(value),
            "method": "mask_pointcloud",
            "aggregate": aggregate,
            "selected_aggregate": aggregate_policy["selected_aggregate"],
            "aggregate_policy": aggregate_policy,
            "pointcloud_sizes": [int(len(points_1)), int(len(points_2))],
            "mask_pixels": [int(mask_1.sum()), int(mask_2.sum())],
            "distance_stats": {
                "min": float(np.min(distances)),
                "p05": float(np.quantile(distances, 0.05)),
                "p10": float(np.quantile(distances, 0.10)),
                "p25": float(np.quantile(distances, 0.25)),
                "median": float(np.median(distances)),
                "p75": float(np.quantile(distances, 0.75)),
                "p80": float(np.quantile(distances, 0.80)),
                "p85": float(np.quantile(distances, 0.85)),
                "p90": float(np.quantile(distances, 0.90)),
            },
            "payload": {},
            "artifacts": [],
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "distance_meters": None}
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _multiframe_mask_pointcloud_distance(
    *,
    frames: Sequence[str],
    frame_records: Sequence[Dict[str, Any]],
    config: SpatialAgentConfig,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    try:
        import numpy as np
        import torch
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "distance_meters": None}

    instance_verification = None
    if getattr(args, "enable_instance_verifier", False):
        frame_records, instance_verification = _verify_object_instances(
            frame_records=frame_records,
            objects=getattr(args, "_current_objects", ("object_1", "object_2")),
            args=args,
        )

    selection_backfill = None
    selected_records = _select_multiframe_records(frame_records, args)
    selected_records = _verify_selected_frame_records(selected_records, objects=getattr(args, "_current_objects", ("object_1", "object_2")), args=args)
    if not any(item.get("region_1") is not None for item in selected_records) or not any(
        item.get("region_2") is not None for item in selected_records
    ):
        selected_records, selection_backfill = _backfill_missing_selected_records(
            frame_records=frame_records,
            selected_records=selected_records,
            objects=getattr(args, "_current_objects", ("object_1", "object_2")),
            args=args,
        )
    if not selected_records:
        return {
            "status": "error",
            "error": "No frames localized either object.",
            "distance_meters": None,
            "instance_verification": instance_verification,
            "selection_backfill": selection_backfill,
        }
    if not any(item.get("region_1") is not None for item in selected_records) or not any(
        item.get("region_2") is not None for item in selected_records
    ):
        return {
            "status": "error",
            "error": "Selected frames did not include finite localizations for both objects.",
            "distance_meters": None,
            "instance_verification": instance_verification,
            "selection_backfill": selection_backfill,
            "selected_frame_positions": [int(item["frame_position"]) for item in selected_records],
        }

    selected_positions = [int(item["frame_position"]) for item in selected_records]
    selected_paths = [frames[position] for position in selected_positions]

    settings = get_tool_settings(config, "Get3DDistance", aliases=["distance", "3d_distance"])
    camera_settings = get_tool_settings(config, "GetCameraParametersVGGT", aliases=["camera", "vggt"])
    mask_settings = get_tool_settings(config, "GetObjectMask", aliases=["mask", "sam2"])
    preprocess_mode = str(settings.get("preprocess_mode") or camera_settings.get("preprocess_mode", "pad"))
    device = resolve_device(settings.get("device") or camera_settings.get("device") or mask_settings.get("device"))
    try:
        predictor = get_sam2_predictor(
            model_id=str(mask_settings.get("model_id", "facebook/sam2.1-hiera-large")),
            checkpoint_path=mask_settings.get("checkpoint_path"),
            config_path=mask_settings.get("config_path"),
            device=device,
        )
        backend = get_vggt_backend(
            model_id=str(
                settings.get("hf_model_id")
                or settings.get("vggt_model_id")
                or camera_settings.get("hf_model_id")
                or "facebook/VGGT-1B"
            ),
            checkpoint_path=(
                settings.get("checkpoint_path")
                or settings.get("vggt_checkpoint_path")
                or camera_settings.get("checkpoint_path")
            ),
            device=device,
        )
        images = backend["load_and_preprocess_images"](selected_paths, mode=preprocess_mode).to(device)
        with backend["torch"].no_grad():
            predictions = backend["model"](images)
        world_points = _extract_world_points_sequence(predictions)
        if world_points is None:
            return {"status": "error", "error": "VGGT predictions did not include world points.", "distance_meters": None}

        object_1_points = []
        object_2_points = []
        frame_summaries = []
        for local_index, record in enumerate(selected_records):
            image_path = frames[int(record["frame_position"])]
            image = load_pil_image(image_path)
            image_rgb = load_rgb_array(image_path)
            predictor.set_image(image_rgb)
            mask_1 = _predict_sam_mask(predictor, record["region_1"]["bbox"]) if record.get("region_1") else None
            mask_2 = _predict_sam_mask(predictor, record["region_2"]["bbox"]) if record.get("region_2") else None
            points_1 = (
                _mask_pixels_to_3d(
                    mask=mask_1,
                    image_size=image.size,
                    frame_points=world_points[local_index],
                    preprocess_mode=preprocess_mode,
                    max_points=args.mask_max_points,
                )
                if mask_1 is not None
                else np.zeros((0, 3), dtype=float)
            )
            points_2 = (
                _mask_pixels_to_3d(
                    mask=mask_2,
                    image_size=image.size,
                    frame_points=world_points[local_index],
                    preprocess_mode=preprocess_mode,
                    max_points=args.mask_max_points,
                )
                if mask_2 is not None
                else np.zeros((0, 3), dtype=float)
            )
            if len(points_1):
                object_1_points.append(points_1)
            if len(points_2):
                object_2_points.append(points_2)
            frame_summaries.append(
                {
                    "frame_position": record["frame_position"],
                    "status": "success" if len(points_1) or len(points_2) else "empty_mask",
                    "has_object_1": record.get("region_1") is not None,
                    "has_object_2": record.get("region_2") is not None,
                    "pointcloud_sizes": [int(len(points_1)), int(len(points_2))],
                    "mask_pixels": [
                        int(mask_1.sum()) if mask_1 is not None else 0,
                        int(mask_2.sum()) if mask_2 is not None else 0,
                    ],
                    "localization_quality": record.get("localization_quality"),
                    "object_1_quality": record.get("object_1_quality"),
                    "object_2_quality": record.get("object_2_quality"),
                    "detection_verification": record.get("detection_verification") or [],
                }
            )

        if not object_1_points or not object_2_points:
            return {
                "status": "error",
                "error": "No finite mask pointclouds for one or both objects across selected frames.",
                "distance_meters": None,
                "frames": frame_summaries,
            }
        points_1_all = np.concatenate(object_1_points, axis=0)
        points_2_all = np.concatenate(object_2_points, axis=0)
        distances = np.linalg.norm(points_1_all[:, None, :] - points_2_all[None, :, :], axis=2).reshape(-1)
        distances = distances[np.isfinite(distances)]
        if distances.size == 0:
            return {"status": "error", "error": "No finite pairwise distances.", "distance_meters": None}
        aggregate_policy = _aggregate_np_policy(distances, args.pointcloud_aggregate)
        value = aggregate_policy["value"]
        return {
            "status": "success",
            "error": None,
            "distance_meters": float(value),
            "method": "mask_pointcloud_multiframe",
            "selected_frame_positions": selected_positions,
            "selection": {
                "top_distance_frames": args.top_distance_frames,
                "single_object_frames": args.single_object_frames,
                "bridge_frames": args.bridge_frames,
                "max_vggt_frames": args.max_vggt_frames,
            },
            "selection_backfill": selection_backfill,
            "instance_verification": instance_verification,
            "aggregate": args.pointcloud_aggregate,
            "selected_aggregate": aggregate_policy["selected_aggregate"],
            "aggregate_policy": aggregate_policy,
            "pointcloud_sizes": [int(len(points_1_all)), int(len(points_2_all))],
            "frames": frame_summaries,
            "distance_stats": {
                "min": float(np.min(distances)),
                "p05": float(np.quantile(distances, 0.05)),
                "p10": float(np.quantile(distances, 0.10)),
                "p25": float(np.quantile(distances, 0.25)),
                "median": float(np.median(distances)),
                "p75": float(np.quantile(distances, 0.75)),
                "p80": float(np.quantile(distances, 0.80)),
                "p85": float(np.quantile(distances, 0.85)),
                "p90": float(np.quantile(distances, 0.90)),
            },
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc), "distance_meters": None}
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _verify_selected_frame_records(
    records: Sequence[Dict[str, Any]],
    *,
    objects: Tuple[str, str],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if not (getattr(args, "enable_detection_verifier", False) or getattr(args, "enable_final_frame_verifier", False)):
        return list(records)
    verified = []
    for item in records:
        if item.get("region_1") is None and item.get("region_2") is None:
            verified.append(dict(item))
            continue
        verified.append(
            _verify_frame_regions(
                frame_record=dict(item),
                image_path=str(item.get("image_path") or ""),
                objects=objects,
                args=args,
            )
        )
    return verified


def _verify_frame_regions(
    *,
    frame_record: Dict[str, Any],
    image_path: str,
    objects: Tuple[str, str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not (getattr(args, "enable_detection_verifier", False) or getattr(args, "enable_final_frame_verifier", False)):
        return frame_record

    verifications = []
    conflict = _verify_overlapping_target_regions(
        frame_record=frame_record,
        image_path=image_path,
        objects=objects,
        args=args,
    )
    if conflict:
        verifications.append(conflict)
    for key, object_name in (("region_1", objects[0]), ("region_2", objects[1])):
        region = frame_record.get(key)
        if not region:
            continue
        if region.get("_best_match_fallback"):
            verifications.append(
                {
                    "agent": "InstanceBestMatchFallbackAgent",
                    "verdict": "accept",
                    "reason_code": "best_match_fallback_preserved",
                    "corrected_label": object_name,
                    "region_key": key,
                    "object": object_name,
                    "detector_score": float(region.get("score") or 0.0),
                    "bbox": region.get("bbox"),
                }
            )
            continue
        verdict = _verify_detection_region(
            image_path=image_path,
            object_name=object_name,
            region=region,
            frame_position=int(frame_record.get("frame_position", -1)),
            args=args,
        )
        verifications.append({**verdict, "region_key": key, "object": object_name})
        if verdict.get("verdict") == "reject":
            _drop_frame_region(frame_record, key)

    frame_record["detection_verification"] = verifications
    region_1 = frame_record.get("region_1")
    region_2 = frame_record.get("region_2")
    if region_1 and region_2:
        frame_record["localization_quality"] = _localization_quality(region_1, region_2)
    return frame_record


def _drop_frame_region(frame_record: Dict[str, Any], key: str) -> None:
    frame_record[key] = None
    if key == "region_1":
        frame_record["has_object_1"] = False
        frame_record["object_1_quality"] = 0.0
    elif key == "region_2":
        frame_record["has_object_2"] = False
        frame_record["object_2_quality"] = 0.0


def _verify_overlapping_target_regions(
    *,
    frame_record: Dict[str, Any],
    image_path: str,
    objects: Tuple[str, str],
    args: argparse.Namespace,
) -> Dict[str, Any] | None:
    region_1 = frame_record.get("region_1")
    region_2 = frame_record.get("region_2")
    if not region_1 or not region_2:
        return None
    bbox_1 = region_1.get("bbox") or []
    bbox_2 = region_2.get("bbox") or []
    if not _bbox_close(bbox_1, bbox_2, eps=2.0) and _bbox_iou(bbox_1, bbox_2) < 0.98:
        return None

    verdict = _verify_bbox_label_conflict(
        image_path=image_path,
        objects=objects,
        region_1=region_1,
        region_2=region_2,
        frame_position=int(frame_record.get("frame_position", -1)),
        args=args,
    )
    decision = verdict.get("decision")
    if decision == "object_1":
        _drop_frame_region(frame_record, "region_2")
    elif decision == "object_2":
        _drop_frame_region(frame_record, "region_1")
    else:
        _drop_frame_region(frame_record, "region_1")
        _drop_frame_region(frame_record, "region_2")
        verdict["decision"] = "neither"
    return verdict


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    ix1 = max(min(ax1, ax2), min(bx1, bx2))
    iy1 = max(min(ay1, ay2), min(by1, by2))
    ix2 = min(max(ax1, ax2), max(bx1, bx2))
    iy2 = min(max(ay1, ay2), max(by1, by2))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = abs((ax2 - ax1) * (ay2 - ay1))
    area_b = abs((bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _verify_bbox_label_conflict(
    *,
    image_path: str,
    objects: Tuple[str, str],
    region_1: Dict[str, Any],
    region_2: Dict[str, Any],
    frame_position: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    try:
        raw = _call_bbox_conflict_verifier_llm(
            image_path=image_path,
            objects=objects,
            region_1=region_1,
            region_2=region_2,
            frame_position=frame_position,
            args=args,
        )
        parsed = _parse_verifier_json(raw)
        decision = _normalize_conflict_decision(parsed.get("decision"), objects)
        if decision not in {"object_1", "object_2", "neither"}:
            raise ValueError(f"Invalid conflict verifier decision: {decision!r}")
        return {
            "agent": "BBoxConflictVerifierAgent",
            "verdict": "accept" if decision in {"object_1", "object_2"} else "reject",
            "decision": decision,
            "reason_code": str(parsed.get("reason_code") or "bbox_label_conflict"),
            "raw_response": raw,
            "objects": list(objects),
            "bbox": region_1.get("bbox"),
            "region_key": "region_conflict",
        }
    except Exception as exc:
        decision = "object_1" if args.verifier_on_error == "accept" else "neither"
        return {
            "agent": "BBoxConflictVerifierAgent",
            "verdict": "accept" if decision != "neither" else "reject",
            "decision": decision,
            "reason_code": "bbox_conflict_verifier_error",
            "error": str(exc),
            "objects": list(objects),
            "bbox": region_1.get("bbox"),
            "region_key": "region_conflict",
        }


def _normalize_conflict_decision(value: Any, objects: Tuple[str, str]) -> str:
    decision = str(value or "").strip().lower()
    aliases = {
        "1": "object_1",
        "a": "object_1",
        "object_1": "object_1",
        "region_1": "object_1",
        objects[0].strip().lower(): "object_1",
        "2": "object_2",
        "b": "object_2",
        "object_2": "object_2",
        "region_2": "object_2",
        objects[1].strip().lower(): "object_2",
        "none": "neither",
        "neither": "neither",
        "both_wrong": "neither",
        "reject": "neither",
    }
    return aliases.get(decision, decision)


def _verify_detection_region(
    *,
    image_path: str,
    object_name: str,
    region: Dict[str, Any],
    frame_position: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    try:
        raw = _call_detection_verifier_llm(
            image_path=image_path,
            object_name=object_name,
            region=region,
            frame_position=frame_position,
            args=args,
        )
        parsed = _parse_verifier_json(raw)
        verdict = str(parsed.get("verdict") or "").strip().lower()
        confidence = float(parsed.get("confidence") or 0.0)
        if verdict not in {"accept", "reject"}:
            raise ValueError(f"Invalid verifier verdict: {verdict!r}")
        reason = str(parsed.get("reason_code") or "llm_verifier")
        return {
            "agent": "DetectionVerifierAgent",
            "verdict": verdict,
            "confidence": confidence,
            "reason_code": reason,
            "corrected_label": parsed.get("corrected_label"),
            "raw_response": raw,
            "detector_score": region.get("score"),
            "bbox": region.get("bbox"),
        }
    except Exception as exc:
        return {
            "agent": "DetectionVerifierAgent",
            "verdict": "accept" if args.verifier_on_error == "accept" else "reject",
            "confidence": 0.0,
            "reason_code": "verifier_error",
            "error": str(exc),
            "detector_score": region.get("score"),
            "bbox": region.get("bbox"),
        }


def _call_detection_verifier_llm(
    *,
    image_path: str,
    object_name: str,
    region: Dict[str, Any],
    frame_position: int,
    args: argparse.Namespace,
) -> str:
    image_path_for_verifier = _write_verifier_overlay(image_path=image_path, region=region, object_name=object_name, args=args)
    object_hint = _verifier_object_hint(object_name)
    confusion_hint = _detector_confusion_hint()
    target_list_hint = _target_object_list_hint(args)
    prompt = (
        "You are DetectionVerifierAgent. The object detector is high-recall and may over-detect many candidate boxes. "
        "Your job is to remove clearly wrong detections while preserving recall for downstream 3D measurement. "
        "Accept when the highlighted box plausibly contains the requested object, including partially occluded views, cropped views, "
        "small objects, or cases where only a distinctive part is visible. "
        "Reject only when the highlighted box clearly contains a different object, pure background, or a region that cannot plausibly be the requested object. "
        "If the evidence is uncertain but the box could plausibly be the requested object, accept it. "
        "Across candidate frames, assume the detector usually produced at least one true detection for the requested object; avoid over-rejecting all candidates. "
        f"{target_list_hint}"
        f"{confusion_hint}"
        "Judge only the highlighted box, not nearby objects outside the box. "
        f"{object_hint}"
        "Do not use detector score as evidence. Do not estimate distance or count objects. "
        "Return only compact JSON with keys: verdict ('accept' or 'reject'), reason_code, corrected_label. "
        f"Requested object: {object_name}. Detector label: {region.get('label')}. "
        f"Detector score: {region.get('score')}. Frame position: {frame_position}."
    )
    return _call_verifier_vl_model(prompt=prompt, image_path=image_path_for_verifier, args=args, max_tokens=256)


def _call_bbox_conflict_verifier_llm(
    *,
    image_path: str,
    objects: Tuple[str, str],
    region_1: Dict[str, Any],
    region_2: Dict[str, Any],
    frame_position: int,
    args: argparse.Namespace,
) -> str:
    image_path_for_verifier = _write_conflict_overlay(
        image_path=image_path,
        region=region_1,
        label=f"{objects[0]} OR {objects[1]}",
        args=args,
    )
    pair_hint = _conflict_pair_hint(objects)
    confusion_hint = _detector_confusion_hint()
    prompt = (
        "You are BBoxConflictVerifierAgent. The highlighted bounding box is identical or nearly identical for two requested objects. "
        "A single box should not be used as both objects for 3D distance. "
        "Decide which one label is more appropriate for the highlighted box: object_1, object_2, or neither. "
        "Prefer choosing object_1 or object_2 when the box contains a recognizable object that fits one requested label better than the other. "
        "For visually similar categories, such as chair vs sofa, choose the closest matching label instead of rejecting by default. "
        f"{confusion_hint}"
        f"{pair_hint}"
        "Choose neither only when the box clearly contains neither requested object, is mostly background, or the visible evidence is too weak to identify either object. "
        "Objects may be partially occluded; choose a label if a distinctive visible part makes the object identifiable. "
        "Do not use detector scores as evidence. Return only compact JSON with keys: decision ('object_1', 'object_2', or 'neither'), reason_code. "
        f"object_1: {objects[0]}. object_2: {objects[1]}. "
        f"detector_score_1: {region_1.get('score')}. detector_score_2: {region_2.get('score')}. Frame position: {frame_position}."
    )
    return _call_verifier_vl_model(prompt=prompt, image_path=image_path_for_verifier, args=args, max_tokens=128)


def _conflict_pair_hint(objects: Tuple[str, str]) -> str:
    names = [item.strip().lower() for item in objects]
    if set(names) == {"chair", "sofa"}:
        sofa_key = "object_1" if names[0] == "sofa" else "object_2"
        chair_key = "object_1" if names[0] == "chair" else "object_2"
        return (
            f"For chair vs sofa: choose {sofa_key} for upholstered lounge seating, bulky padded seats, couches, loveseats, "
            f"or armchair-like soft furniture. Choose {chair_key} only for a clearly separate dining/office/simple chair "
            "with a distinct chair form. If the box shows a padded blue lounge seat or couch-like furniture, prefer sofa. "
        )
    return ""


def _detector_confusion_hint() -> str:
    return (
        "Object detectors often confuse visually similar categories, such as chair vs sofa, table vs cabinet/dresser, "
        "stool vs chair, refrigerator vs cabinet, or washer/dishwasher/stove-like appliances. "
        "Use the visual evidence inside the highlighted box to correct these category confusions; do not trust the detector label when the object shape better matches a similar category. "
    )


def _target_object_list_hint(args: argparse.Namespace) -> str:
    objects = getattr(args, "_current_objects", None) or ()
    objects = [str(item).strip() for item in objects if str(item).strip()]
    if not objects:
        return ""
    target_text = ", ".join(objects)
    return (
        f"The current task only cares about this target object list: [{target_text}]. "
        "First decide whether the visible object belongs to this target list or is outside the target list. "
        "Then accept only if it matches the requested object for this verifier call. "
        "Do not perform open-vocabulary classification; labels outside the target list may be used only as a reason to reject, not as an alternative target. "
    )


def _verifier_object_hint(object_name: str) -> str:
    name = object_name.strip().lower()
    if name in {"toilet", "toilet seat", "commode"}:
        return (
            "For toilet specifically: accept only if the box contains a recognizable toilet fixture, "
            "such as a bowl/rim/seat/tank/base. Reject laundry baskets, chairs, cabinets, wall fixtures, "
            "narrow vertical objects, door edges, shadows, or boxes that merely occur in a bathroom-like area. "
        )
    if name in {"table", "desk", "dining table", "coffee table"}:
        return (
            "For table specifically: accept desks and tables when the box contains a usable tabletop, table edge, "
            "open leg/knee space, table legs, or a work/dining surface, even if cropped or partially occluded. "
            "Reject dressers, cabinets, drawer units, bedside cabinets, and closed storage furniture, especially when "
            "the box mostly covers drawer fronts, handles, or a solid cabinet body without visible tabletop/leg evidence. "
            "Do not call an object a table merely because it has a flat top; prefer cabinet/dresser for drawer-front storage furniture. "
        )
    return ""


def _write_conflict_overlay(*, image_path: str, region: Dict[str, Any], label: str, args: argparse.Namespace) -> str:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    max_side = int(args.verifier_max_image_side)
    scale = 1.0
    if max(image.size) > max_side > 0:
        scale = max_side / float(max(image.size))
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    draw = ImageDraw.Draw(image)
    bbox = [float(value) * scale for value in (region.get("bbox") or [])[:4]]
    if len(bbox) == 4:
        color = (180, 0, 255)
        draw.rectangle(bbox, outline=color, width=max(3, int(round(4 * scale))))
        draw.text((bbox[0], max(0, bbox[1] - 16)), label, fill=color)
    output_dir = Path(getattr(args, "_current_doc_dir", args.output_dir)) / "verifier_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(image_path).stem}_conflict.jpg"
    image.save(output_path, format="JPEG", quality=90)
    return str(output_path)


def _write_verifier_overlay(*, image_path: str, region: Dict[str, Any], object_name: str, args: argparse.Namespace) -> str:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    max_side = int(args.verifier_max_image_side)
    scale = 1.0
    if max(image.size) > max_side > 0:
        scale = max_side / float(max(image.size))
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    draw = ImageDraw.Draw(image)
    bbox = [float(value) * scale for value in (region.get("bbox") or [])[:4]]
    if len(bbox) == 4:
        draw.rectangle(bbox, outline=(255, 0, 0), width=max(3, int(round(4 * scale))))
        draw.text((bbox[0], max(0, bbox[1] - 16)), f"{object_name} {float(region.get('score') or 0.0):.2f}", fill=(255, 0, 0))
    output_dir = Path(getattr(args, "_current_doc_dir", args.output_dir)) / "verifier_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_object = re.sub(r"[^a-zA-Z0-9_.-]+", "_", object_name.strip()) or "object"
    output_path = output_dir / f"{Path(image_path).stem}_{safe_object}_verify.jpg"
    image.save(output_path, format="JPEG", quality=90)
    return str(output_path)


def _parse_verifier_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Verifier response JSON must be an object.")
    return parsed


def _extract_world_points_sequence(predictions: Dict[str, Any]) -> Any | None:
    for key in ("world_points", "world_points_cam", "points3D", "points_3d"):
        if key not in predictions:
            continue
        points = predictions[key]
        if hasattr(points, "detach"):
            points = points.detach().cpu().numpy()
        if points.ndim == 5:
            return points[0]
        if points.ndim == 4:
            return points
    return None


def _predict_sam_mask(predictor: Any, bbox: Sequence[float]) -> Any | None:
    import numpy as np

    masks, scores, _ = predictor.predict(box=np.asarray(bbox, dtype=np.float32), multimask_output=True)
    if len(masks) == 0:
        return None
    mask = masks[int(np.argmax(scores))].astype(bool)
    if not mask.any():
        return None
    return mask


def _mask_pixels_to_3d(
    *,
    mask: Any,
    image_size: Tuple[int, int],
    frame_points: Any,
    preprocess_mode: str,
    max_points: int,
) -> Any:
    import numpy as np

    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=float)
    if xs.size > max_points:
        indices = np.linspace(0, xs.size - 1, max_points).round().astype(int)
        xs = xs[indices]
        ys = ys[indices]
    h, w = frame_points.shape[:2]
    sampled = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        mapped_x, mapped_y = map_point_to_preprocessed(
            (float(x), float(y)),
            image_size,
            mode=preprocess_mode,
            target_size=max(h, w),
        )
        px = int(round(max(0.0, min(float(w - 1), mapped_x))))
        py = int(round(max(0.0, min(float(h - 1), mapped_y))))
        point = frame_points[py, px]
        if np.all(np.isfinite(point[:3])):
            sampled.append([float(point[0]), float(point[1]), float(point[2])])
    if not sampled:
        return np.zeros((0, 3), dtype=float)
    points = np.asarray(sampled, dtype=float)
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center[None, :], axis=1)
    keep = radii <= np.quantile(radii, 0.90)
    return points[keep]


def _extract_single_frame_points(predictions: Dict[str, Any]) -> Any | None:
    for key in ("world_points", "world_points_cam", "points3D", "points_3d"):
        if key not in predictions:
            continue
        points = predictions[key]
        if hasattr(points, "detach"):
            points = points.detach().cpu().numpy()
        if points.ndim == 5:
            return points[0, 0]
        if points.ndim == 4:
            return points[0]
        if points.ndim == 3:
            return points
    return None


def _bbox_points_to_3d(
    *,
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    frame_points: Any,
    preprocess_mode: str,
    grid_size: int,
) -> Any:
    import numpy as np

    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    count = max(2, grid_size)
    h, w = frame_points.shape[:2]
    sampled = []
    for y in np.linspace(y1, y2, count):
        for x in np.linspace(x1, x2, count):
            mapped_x, mapped_y = map_point_to_preprocessed(
                (float(x), float(y)),
                image_size,
                mode=preprocess_mode,
                target_size=max(h, w),
            )
            px = int(round(max(0.0, min(float(w - 1), mapped_x))))
            py = int(round(max(0.0, min(float(h - 1), mapped_y))))
            point = frame_points[py, px]
            if np.all(np.isfinite(point[:3])):
                sampled.append([float(point[0]), float(point[1]), float(point[2])])
    if not sampled:
        return np.zeros((0, 3), dtype=float)
    points = np.asarray(sampled, dtype=float)
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center[None, :], axis=1)
    keep = radii <= np.quantile(radii, 0.90)
    return points[keep]


ADAPTIVE_P05_P90_RATIO_MAX = 3.692
ADAPTIVE_P05_P90_SPREAD_MIN = 0.471


def _aggregate_np_distances(distances: Any, mode: str) -> float:
    import numpy as np

    if mode == "min":
        return float(np.min(distances))
    if mode == "p05":
        return float(np.quantile(distances, 0.05))
    if mode == "p10":
        return float(np.quantile(distances, 0.10))
    if mode == "p25":
        return float(np.quantile(distances, 0.25))
    if mode == "median":
        return float(np.median(distances))
    if mode == "p75":
        return float(np.quantile(distances, 0.75))
    if mode == "p80":
        return float(np.quantile(distances, 0.80))
    if mode == "p85":
        return float(np.quantile(distances, 0.85))
    if mode == "p90":
        return float(np.quantile(distances, 0.90))
    if mode == "iqm_75_90":
        p75 = float(np.quantile(distances, 0.75))
        p90 = float(np.quantile(distances, 0.90))
        trimmed = distances[(distances >= p75) & (distances <= p90)]
        return float(np.mean(trimmed)) if trimmed.size else float((p75 + p90) / 2.0)
    if mode == "blend_p75_p90_25":
        p75 = float(np.quantile(distances, 0.75))
        p90 = float(np.quantile(distances, 0.90))
        return float(0.25 * p90 + 0.75 * p75)
    if mode == "blend_p75_p90_50":
        p75 = float(np.quantile(distances, 0.75))
        p90 = float(np.quantile(distances, 0.90))
        return float(0.50 * p90 + 0.50 * p75)
    if mode == "blend_p75_p90_75":
        p75 = float(np.quantile(distances, 0.75))
        p90 = float(np.quantile(distances, 0.90))
        return float(0.75 * p90 + 0.25 * p75)
    if mode == "adaptive_p05_p90":
        policy = _adaptive_p05_p90_policy(distances)
        return float(policy["value"])
    raise ValueError(f"Unknown pointcloud aggregate: {mode}")


def _adaptive_p05_p90_policy(distances: Any) -> Dict[str, Any]:
    import numpy as np

    p05 = float(np.quantile(distances, 0.05))
    p90 = float(np.quantile(distances, 0.90))
    ratio = p90 / max(p05, 1e-9)
    spread = p90 - p05
    selected = "p05" if ratio <= ADAPTIVE_P05_P90_RATIO_MAX and spread >= ADAPTIVE_P05_P90_SPREAD_MIN else "p90"
    return {
        "strategy": "adaptive_p05_p90",
        "selected_aggregate": selected,
        "value": p05 if selected == "p05" else p90,
        "p05": p05,
        "p90": p90,
        "ratio": ratio,
        "spread": spread,
        "ratio_max": ADAPTIVE_P05_P90_RATIO_MAX,
        "spread_min": ADAPTIVE_P05_P90_SPREAD_MIN,
    }


def _aggregate_np_policy(distances: Any, mode: str) -> Dict[str, Any]:
    if mode == "adaptive_p05_p90":
        return _adaptive_p05_p90_policy(distances)
    return {
        "strategy": "fixed",
        "selected_aggregate": mode,
        "value": _aggregate_np_distances(distances, mode),
    }


def _write_summary(records: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    evaluated = [r for r in records if isinstance(r.get("prediction"), (int, float)) and r.get("ground_truth") is not None]
    abs_errors = [float(r["abs_error"]) for r in evaluated if r.get("abs_error") is not None]
    mra_scores = [float(r.get("mra", 0.0)) for r in records]
    summary = {
        "count": len(records),
        "evaluated": len(evaluated),
        "success": sum(1 for r in records if r.get("status") == "success"),
        "vsibench_metric": "MRA:.5:.95:.05",
        "mra": sum(mra_scores) / len(mra_scores) if mra_scores else None,
        "mra_percent": (sum(mra_scores) / len(mra_scores) * 100.0) if mra_scores else None,
        "mae_m": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "rmse_m": math.sqrt(sum(err * err for err in abs_errors) / len(abs_errors)) if abs_errors else None,
        "within_25pct": sum(1 for r in evaluated if r.get("within_25pct")),
        "within_05m": sum(1 for r in evaluated if r.get("within_05m")),
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Get3DDistance on VSI-Bench object_abs_distance samples.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-name", default="nyu-visionx/VSI-Bench")
    parser.add_argument("--dataset-cache-dir", default="/disk/wangzhe/VSI-Bench")
    parser.add_argument("--annotations-path", default="", help="Optional local parquet/jsonl annotations, e.g. VSI-Train-10k.")
    parser.add_argument("--video-root", default="", help="Root directory for relative `video` paths in local annotations.")
    parser.add_argument("--question-type", default="", help="Question type to select. Defaults to object_abs_distance or absolute_distance for annotations.")
    parser.add_argument("--tool-config-path", default="configs/tool_config.server.json")
    parser.add_argument("--output-dir", default="/tmp/spatial_agent_vsibench_abs_distance")
    parser.add_argument("--reuse-frame-results-dir", default="", help="Reuse frame_results and sampled image paths from a previous run.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--doc-ids", default="", help="Comma-separated dataset doc ids to evaluate instead of the first limit samples.")
    parser.add_argument("--doc-ids-file", default="", help="File containing doc ids, one per line or comma-separated.")
    parser.add_argument("--list-only", action="store_true", help="Print selected docs and resolved video paths without running tools.")
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--frame-index", type=int, default=-1, help="-1 uses the middle sampled frame.")
    parser.add_argument("--frame-aggregate", choices=["min", "p10", "p25", "median"], default="median")
    parser.add_argument("--top-distance-frames", type=int, default=0, help="If >0, localize all selected frames and run distance only on the top-k frames.")
    parser.add_argument("--single-object-frames", type=int, default=0, help="Also include top-k frames for each individual object in multiframe pointcloud mode.")
    parser.add_argument("--bridge-frames", type=int, default=0, help="Also include up to this many sparse frames between selected multiframe endpoints.")
    parser.add_argument("--max-vggt-frames", type=int, default=0, help="Cap selected multiframe VGGT inputs after adding same-object, single-object, and bridge frames.")
    parser.add_argument("--selected-frame-positions", default="", help="Comma-separated sampled frame positions to send to VGGT in multiframe pointcloud mode.")
    parser.add_argument("--boundary-samples", type=int, default=24)
    parser.add_argument("--candidate-pairs", type=int, default=1)
    parser.add_argument("--min-pixel-distance", type=float, default=8.0)
    parser.add_argument("--distance-aggregate", choices=["min", "p10", "p25", "median"], default="min")
    parser.add_argument("--method", choices=["point_pairs", "bbox_pointcloud", "mask_pointcloud", "mask_pointcloud_multiframe"], default="point_pairs")
    parser.add_argument("--pointcloud-grid", type=int, default=16)
    parser.add_argument(
        "--pointcloud-aggregate",
        choices=[
            "min",
            "p05",
            "p10",
            "p25",
            "median",
            "p75",
            "p80",
            "p85",
            "p90",
            "iqm_75_90",
            "blend_p75_p90_25",
            "blend_p75_p90_50",
            "blend_p75_p90_75",
            "adaptive_p05_p90",
        ],
        default="adaptive_p05_p90",
    )
    parser.add_argument("--mask-max-points", type=int, default=256)
    parser.add_argument("--enable-detection-verifier", action="store_true", help="Use DetectionVerifierAgent to accept/reject selected object detections before distance calculation.")
    parser.add_argument("--enable-final-frame-verifier", action="store_true", help="Run DetectionVerifierAgent after final VGGT frame selection to reject bad selected-frame boxes.")
    parser.add_argument("--enable-instance-verifier", action="store_true", help="Use InstanceVerifierAgent to accept/reject cross-frame object hypotheses before VGGT frame selection.")
    parser.add_argument("--instance-verifier-min-score", type=float, default=0.25)
    parser.add_argument("--instance-verifier-max-candidates", type=int, default=30)
    parser.add_argument("--instance-verifier-max-frame-gap", type=int, default=12)
    parser.add_argument("--instance-verifier-center-threshold", type=float, default=2.0)
    parser.add_argument("--instance-verifier-max-cluster-frames", type=int, default=6)
    parser.add_argument("--instance-verifier-contact-frames", type=int, default=6)
    parser.add_argument("--instance-verifier-tile-size", type=int, default=360)
    parser.add_argument("--instance-verifier-grid-cols", type=int, default=3)
    parser.add_argument("--instance-verifier-tile-gap", type=int, default=12)
    parser.add_argument("--enable-instance-verifier-recheck", action="store_true", help="If an object has no accepted instance candidates, recheck rejected candidates in score-ranked batches.")
    parser.add_argument("--instance-verifier-recheck-rounds", type=int, default=3)
    parser.add_argument("--instance-verifier-recheck-batch-size", type=int, default=6)
    parser.add_argument("--instance-verifier-recheck-all-rounds", action="store_true")
    parser.add_argument("--enable-instance-verifier-best-match-fallback", action="store_true", help="If instance verification rejects every candidate for an object, force-select the visually best matching candidate.")
    parser.add_argument("--instance-verifier-best-match-candidates", type=int, default=12)
    parser.add_argument("--verifier-model", default=os.getenv("DETECTION_VERIFIER_MODEL", "gpt-4o-mini"))
    parser.add_argument("--verifier-backend", choices=["openai", "ollama_generate"], default=os.getenv("VERIFIER_BACKEND", "openai"))
    parser.add_argument("--verifier-ollama-url", default=os.getenv("VERIFIER_OLLAMA_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--verifier-api-base-url", default="")
    parser.add_argument("--verifier-api-base-url-env", default="OPENAI_API_BASE_URL")
    parser.add_argument("--verifier-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--verifier-timeout", type=int, default=120)
    parser.add_argument("--verifier-on-error", choices=["accept", "reject"], default="accept")
    parser.add_argument("--verifier-max-image-side", type=int, default=960)
    parser.add_argument("--hf-token", default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cache_dir = resolve_vsibench_cache_dir(args.dataset_cache_dir)
    if args.annotations_path:
        dataset = _load_annotation_dataset(args.annotations_path)
        default_question_type = "absolute_distance"
    else:
        dataset = _load_dataset(args.dataset_name, args.split, cache_dir, args.hf_token)
        default_question_type = "object_abs_distance"
    question_type = args.question_type or default_question_type
    doc_ids = _parse_doc_ids(args.doc_ids) or _parse_doc_ids_file(args.doc_ids_file) or _select_doc_ids(dataset, args.limit, question_type)

    if args.list_only:
        rows = []
        for doc_id in doc_ids:
            doc = dict(dataset[doc_id])
            rows.append(
                {
                    "doc_id": doc_id,
                    "question_type": doc.get("question_type"),
                    "video_path": _resolve_video_path(doc, cache_dir, args),
                    "video_exists": Path(_resolve_video_path(doc, cache_dir, args)).exists(),
                    "objects": list(_parse_objects(str(doc.get("question") or "")) or []),
                    "ground_truth": _parse_float(doc.get("ground_truth")),
                    "question": doc.get("question"),
                }
            )
        print(json.dumps({"count": len(rows), "question_type": question_type, "docs": rows}, indent=2, ensure_ascii=False))
        return 0

    config = SpatialAgentConfig(artifact_dir=args.output_dir, tool_config=load_tool_config(args.tool_config_path))
    localizer = LocalizeObjectsTool(config)
    distance_tool = Get3DDistanceTool(config)

    records = []
    for position, doc_id in enumerate(doc_ids, start=1):
        print(f"[{position}/{len(doc_ids)}] doc_id={doc_id} {dataset[doc_id].get('question')}", flush=True)
        record = _run_doc(
            doc_id=doc_id,
            doc=dict(dataset[doc_id]),
            args=args,
            config=config,
            cache_dir=cache_dir,
            localizer=localizer,
            distance_tool=distance_tool,
        )
        records.append(record)
        print(
            f"  status={record.get('status')} pred={record.get('prediction')} gt={record.get('ground_truth')} abs_err={record.get('abs_error')} error={record.get('error')}",
            flush=True,
        )

    summary = _write_summary(records, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
