from __future__ import annotations

import argparse
import csv
import contextlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


QUANTILES = ["min", "p05", "p10", "p25", "median", "p75", "p90"]
MAIN_QUANTILES = ["p25", "median", "p75", "p90"]
EPS = 1e-9


@dataclass(frozen=True)
class Row:
    doc_id: int
    ground_truth: float
    predictions: dict[str, float]
    scores: dict[str, float]
    extra: dict[str, float]

    @property
    def p90(self) -> float:
        return self.predictions["p90"]

    @property
    def spread_p90_p25(self) -> float:
        return (self.predictions["p90"] - self.predictions["p25"]) / max(self.predictions["p90"], EPS)

    @property
    def lower_tail(self) -> float:
        return (self.predictions["p25"] - self.predictions["p05"]) / max(self.predictions["p90"], EPS)

    @property
    def skew(self) -> float:
        return (self.predictions["p90"] - self.predictions["median"]) / max(
            self.predictions["median"] - self.predictions["p25"], EPS
        )

    @property
    def ratio_p90_median(self) -> float:
        return self.predictions["p90"] / max(self.predictions["median"], EPS)

    @property
    def iqr_norm(self) -> float:
        return (self.predictions["p75"] - self.predictions["p25"]) / max(self.predictions["p90"], EPS)


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            predictions = {key: float(item[f"pred_{key}"]) for key in QUANTILES}
            scores = {key: float(item[f"mra_{key}"]) for key in QUANTILES}
            rows.append(
                Row(
                    doc_id=int(item["doc_id"]),
                    ground_truth=float(item["ground_truth"]),
                    predictions=predictions,
                    scores=scores,
                    extra={},
                )
            )
    return rows


def _safe_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else 0.0


def _safe_min(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return min(values) if values else 0.0


def _safe_max(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else 0.0


def extract_extra_features(result: dict[str, object]) -> dict[str, float]:
    payload = result.get("distance_payload") or {}
    if not isinstance(payload, dict):
        return {}
    frames = payload.get("frames") or []
    if not isinstance(frames, list):
        frames = []
    selected = payload.get("selected_frame_positions") or []
    if not isinstance(selected, list):
        selected = []
    pointcloud_sizes = payload.get("pointcloud_sizes") or [0, 0]
    if not isinstance(pointcloud_sizes, list):
        pointcloud_sizes = [0, 0]

    obj1_points = [float((frame.get("pointcloud_sizes") or [0, 0])[0]) for frame in frames if isinstance(frame, dict)]
    obj2_points = [float((frame.get("pointcloud_sizes") or [0, 0])[1]) for frame in frames if isinstance(frame, dict)]
    obj1_masks = [float((frame.get("mask_pixels") or [0, 0])[0]) for frame in frames if isinstance(frame, dict)]
    obj2_masks = [float((frame.get("mask_pixels") or [0, 0])[1]) for frame in frames if isinstance(frame, dict)]
    qualities = [float(frame.get("localization_quality") or 0.0) for frame in frames if isinstance(frame, dict)]
    obj1_quality = [float(frame.get("object_1_quality") or 0.0) for frame in frames if isinstance(frame, dict)]
    obj2_quality = [float(frame.get("object_2_quality") or 0.0) for frame in frames if isinstance(frame, dict)]

    det_scores: list[float] = []
    verifier_accept = 0
    verifier_reject = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for verify in frame.get("detection_verification") or []:
            if not isinstance(verify, dict):
                continue
            if verify.get("verdict") == "accept":
                verifier_accept += 1
            elif verify.get("verdict") == "reject":
                verifier_reject += 1
            score = verify.get("detector_score")
            if score is not None:
                det_scores.append(float(score))

    instance = payload.get("instance_verification") or {}
    objects = instance.get("objects") if isinstance(instance, dict) else {}
    candidate_counts: list[float] = []
    hypothesis_counts: list[float] = []
    accepted_hyp_counts: list[float] = []
    accepted_candidate_counts: list[float] = []
    rejected_candidate_counts: list[float] = []
    fallback_count = 0
    if isinstance(objects, dict):
        for info in objects.values():
            if not isinstance(info, dict):
                continue
            candidate_counts.append(float(info.get("candidate_count") or 0))
            hypothesis_counts.append(float(info.get("hypothesis_count") or 0))
            accepted_hyp_counts.append(float(info.get("accepted_hypotheses") or 0))
            if info.get("best_match_fallback"):
                fallback_count += 1
                if info.get("fallback_rechecks"):
                    fallback_count += len(info.get("fallback_rechecks") or [])
            for hyp in info.get("hypotheses") or []:
                if not isinstance(hyp, dict):
                    continue
                accepted_candidate_counts.append(float(len(hyp.get("accepted_candidate_ids") or [])))
                rejected_candidate_counts.append(float(len(hyp.get("rejected_candidate_ids") or [])))

    selected_count = float(len(selected))
    selected_span = float(max(selected) - min(selected)) if selected else 0.0
    frame_results = result.get("frame_results") or []
    if not isinstance(frame_results, list):
        frame_results = []
    frame_statuses = [str(frame.get("status") or "") for frame in frame_results if isinstance(frame, dict)]
    localized_count = sum(1 for status in frame_statuses if status == "localized")
    partial_count = sum(1 for status in frame_statuses if status == "partially_localized")
    error_count = sum(1 for status in frame_statuses if status == "error")
    predictions = [float(frame.get("prediction")) for frame in frame_results if isinstance(frame, dict) and frame.get("prediction") is not None]
    localized_quality = [float(frame.get("localization_quality") or 0.0) for frame in frame_results if isinstance(frame, dict)]
    object1_quality = [float(frame.get("object_1_quality") or 0.0) for frame in frame_results if isinstance(frame, dict)]
    object2_quality = [float(frame.get("object_2_quality") or 0.0) for frame in frame_results if isinstance(frame, dict)]
    pc1_frame_values = [float((frame.get("pointcloud_sizes") or [0, 0])[0]) for frame in frame_results if isinstance(frame, dict)]
    pc2_frame_values = [float((frame.get("pointcloud_sizes") or [0, 0])[1]) for frame in frame_results if isinstance(frame, dict)]
    both_object_frame_ratio = sum(
        1 for frame in frame_results if isinstance(frame, dict) and frame.get("has_object_1") and frame.get("has_object_2")
    ) / max(len(frame_results), 1)
    obj1_present_ratio = sum(1 for frame in frame_results if isinstance(frame, dict) and frame.get("has_object_1")) / max(len(frame_results), 1)
    obj2_present_ratio = sum(1 for frame in frame_results if isinstance(frame, dict) and frame.get("has_object_2")) / max(len(frame_results), 1)
    selected_frame_results = [frame for frame in frame_results if isinstance(frame, dict) and frame.get("frame_position") in set(selected)]
    selected_quality = [float(frame.get("localization_quality") or 0.0) for frame in selected_frame_results]
    selected_localized = sum(1 for frame in selected_frame_results if frame.get("status") == "localized")
    selected_partial = sum(1 for frame in selected_frame_results if frame.get("status") == "partially_localized")
    selected_error = sum(1 for frame in selected_frame_results if frame.get("status") == "error")
    selected_pc1 = [float((frame.get("pointcloud_sizes") or [0, 0])[0]) for frame in selected_frame_results]
    selected_pc2 = [float((frame.get("pointcloud_sizes") or [0, 0])[1]) for frame in selected_frame_results]
    both_frames = sum(1 for frame in frames if isinstance(frame, dict) and frame.get("has_object_1") and frame.get("has_object_2"))
    obj1_frames = sum(1 for frame in frames if isinstance(frame, dict) and frame.get("has_object_1"))
    obj2_frames = sum(1 for frame in frames if isinstance(frame, dict) and frame.get("has_object_2"))
    total_pc_1 = float(pointcloud_sizes[0]) if len(pointcloud_sizes) > 0 else 0.0
    total_pc_2 = float(pointcloud_sizes[1]) if len(pointcloud_sizes) > 1 else 0.0
    min_total_pc = min(total_pc_1, total_pc_2)
    max_total_pc = max(total_pc_1, total_pc_2)

    return {
        "selected_count": selected_count,
        "selected_span": selected_span,
        "selected_density": selected_count / max(selected_span, 1.0),
        "selected_frame_count": float(len(selected_frame_results)),
        "selected_localized_ratio": selected_localized / max(len(selected_frame_results), 1),
        "selected_partial_ratio": selected_partial / max(len(selected_frame_results), 1),
        "selected_error_ratio": selected_error / max(len(selected_frame_results), 1),
        "selected_quality_mean": _safe_mean(selected_quality),
        "selected_quality_min": _safe_min(selected_quality),
        "selected_pc1_mean": _safe_mean(selected_pc1),
        "selected_pc2_mean": _safe_mean(selected_pc2),
        "both_frame_ratio": both_frames / max(len(frames), 1),
        "obj1_frame_ratio": obj1_frames / max(len(frames), 1),
        "obj2_frame_ratio": obj2_frames / max(len(frames), 1),
        "both_object_frame_ratio": both_object_frame_ratio,
        "obj1_present_ratio": obj1_present_ratio,
        "obj2_present_ratio": obj2_present_ratio,
        "single_object_frame_ratio": (obj1_frames + obj2_frames - 2 * both_frames) / max(len(frames), 1),
        "total_pc_min": min_total_pc,
        "total_pc_max": max_total_pc,
        "total_pc_ratio": min_total_pc / max(max_total_pc, EPS),
        "frame_pc1_mean": _safe_mean(obj1_points),
        "frame_pc2_mean": _safe_mean(obj2_points),
        "frame_pc_min_mean": min(_safe_mean(obj1_points), _safe_mean(obj2_points)),
        "frame_pc_ratio_mean": min(_safe_mean(obj1_points), _safe_mean(obj2_points)) / max(max(_safe_mean(obj1_points), _safe_mean(obj2_points)), EPS),
        "frame_pc1_std": math.sqrt(_safe_mean([(x - _safe_mean(obj1_points)) ** 2 for x in obj1_points])) if obj1_points else 0.0,
        "frame_pc2_std": math.sqrt(_safe_mean([(x - _safe_mean(obj2_points)) ** 2 for x in obj2_points])) if obj2_points else 0.0,
        "frame_prediction_std": math.sqrt(_safe_mean([(x - _safe_mean(predictions)) ** 2 for x in predictions])) if predictions else 0.0,
        "mask1_mean": _safe_mean(obj1_masks),
        "mask2_mean": _safe_mean(obj2_masks),
        "mask_min_mean": min(_safe_mean(obj1_masks), _safe_mean(obj2_masks)),
        "mask_ratio_mean": min(_safe_mean(obj1_masks), _safe_mean(obj2_masks)) / max(max(_safe_mean(obj1_masks), _safe_mean(obj2_masks)), EPS),
        "quality_mean": _safe_mean(qualities),
        "quality_min": _safe_min(qualities),
        "quality_max": _safe_max(qualities),
        "obj_quality_min_mean": min(_safe_mean(obj1_quality), _safe_mean(obj2_quality)),
        "frame_quality_std": math.sqrt(_safe_mean([(x - _safe_mean(qualities)) ** 2 for x in qualities])) if qualities else 0.0,
        "frame_localized_ratio": localized_count / max(len(frame_results), 1),
        "frame_partial_ratio": partial_count / max(len(frame_results), 1),
        "frame_error_ratio": error_count / max(len(frame_results), 1),
        "frame_localized_quality_mean": _safe_mean(localized_quality),
        "det_score_mean": _safe_mean(det_scores),
        "det_score_min": _safe_min(det_scores),
        "verifier_accept_count": float(verifier_accept),
        "verifier_reject_count": float(verifier_reject),
        "verifier_reject_ratio": verifier_reject / max(verifier_accept + verifier_reject, 1),
        "candidate_count_mean": _safe_mean(candidate_counts),
        "candidate_count_max": _safe_max(candidate_counts),
        "hypothesis_count_mean": _safe_mean(hypothesis_counts),
        "accepted_hypothesis_ratio": _safe_mean(accepted_hyp_counts) / max(_safe_mean(hypothesis_counts), EPS),
        "accepted_candidate_mean": _safe_mean(accepted_candidate_counts),
        "rejected_candidate_mean": _safe_mean(rejected_candidate_counts),
        "fallback_count": float(fallback_count),
        "has_backfill": 1.0 if payload.get("selection_backfill") else 0.0,
    }


def attach_extra_features(rows: list[Row], run_dir: Path | None) -> list[Row]:
    if run_dir is None:
        return rows
    by_doc: dict[int, dict[str, float]] = {}
    for path in run_dir.glob("doc_*/result.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if result.get("status") != "success":
            continue
        by_doc[int(result["doc_id"])] = extract_extra_features(result)
    return [
        Row(
            doc_id=row.doc_id,
            ground_truth=row.ground_truth,
            predictions=row.predictions,
            scores=row.scores,
            extra=by_doc.get(row.doc_id, {}),
        )
        for row in rows
    ]


def mean_score(rows: Iterable[Row], choose: Callable[[Row], str]) -> float:
    values = [row.scores[choose(row)] for row in rows]
    return sum(values) / len(values) if values else 0.0


def candidate_thresholds(values: list[float], max_candidates: int = 80) -> list[float]:
    unique = sorted({value for value in values if math.isfinite(value)})
    if not unique:
        return []
    if len(unique) <= max_candidates:
        mids = [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
        return [unique[0] - EPS, *mids, unique[-1] + EPS]
    thresholds = []
    for index in range(max_candidates):
        q = index / (max_candidates - 1)
        pos = q * (len(unique) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        value = unique[lo] if lo == hi else unique[lo] * (hi - pos) + unique[hi] * (pos - lo)
        thresholds.append(value)
    return sorted(set(thresholds))


def split_folds(rows: list[Row], folds: int, seed: int) -> list[list[Row]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    buckets = [[] for _ in range(folds)]
    for index, row in enumerate(shuffled):
        buckets[index % folds].append(row)
    return buckets


def tune_near_gate(train: list[Row]) -> dict[str, object]:
    thresholds = candidate_thresholds([row.p90 for row in train])
    best_score = -1.0
    best: dict[str, object] = {}
    for near_quantile in ["p05", "p10", "p25", "median", "p75"]:
        for threshold in thresholds:
            def choose(row: Row, q: str = near_quantile, t: float = threshold) -> str:
                return q if row.p90 < t else "p90"

            score = mean_score(train, choose)
            if score > best_score:
                best_score = score
                best = {
                    "family": "near_gate",
                    "train_mra": score,
                    "near_quantile": near_quantile,
                    "threshold_p90": threshold,
                }
    return best


def tune_spread_gate(train: list[Row]) -> dict[str, object]:
    p90_thresholds = candidate_thresholds([row.p90 for row in train], max_candidates=50)
    spread_thresholds = candidate_thresholds([row.spread_p90_p25 for row in train], max_candidates=50)
    best_score = -1.0
    best: dict[str, object] = {}
    tight_quantiles = ["p10", "p25", "median"]
    loose_quantiles = ["p25", "median", "p75", "p90"]
    for p90_threshold in p90_thresholds:
        for spread_threshold in spread_thresholds:
            for tight_q in tight_quantiles:
                for loose_q in loose_quantiles:
                    def choose(
                        row: Row,
                        t_dist: float = p90_threshold,
                        t_spread: float = spread_threshold,
                        q_tight: str = tight_q,
                        q_loose: str = loose_q,
                    ) -> str:
                        if row.p90 >= t_dist:
                            return "p90"
                        return q_tight if row.spread_p90_p25 < t_spread else q_loose

                    score = mean_score(train, choose)
                    if score > best_score:
                        best_score = score
                        best = {
                            "family": "spread_gate",
                            "train_mra": score,
                            "threshold_p90": p90_threshold,
                            "threshold_spread": spread_threshold,
                            "tight_quantile": tight_q,
                            "loose_quantile": loose_q,
                        }
    return best


FEATURES: dict[str, Callable[[Row], float]] = {
    "p90": lambda row: row.p90,
    "median": lambda row: row.predictions["median"],
    "spread_p90_p25": lambda row: row.spread_p90_p25,
    "lower_tail": lambda row: row.lower_tail,
    "skew": lambda row: row.skew,
    "ratio_p90_median": lambda row: row.ratio_p90_median,
    "iqr_norm": lambda row: row.iqr_norm,
}

EXTRA_FEATURE_NAMES = [
    "selected_count",
    "selected_span",
    "selected_density",
    "selected_frame_count",
    "selected_localized_ratio",
    "selected_partial_ratio",
    "selected_error_ratio",
    "selected_quality_mean",
    "selected_quality_min",
    "selected_pc1_mean",
    "selected_pc2_mean",
    "both_frame_ratio",
    "obj1_frame_ratio",
    "obj2_frame_ratio",
    "both_object_frame_ratio",
    "obj1_present_ratio",
    "obj2_present_ratio",
    "single_object_frame_ratio",
    "total_pc_min",
    "total_pc_max",
    "total_pc_ratio",
    "frame_pc_min_mean",
    "frame_pc_ratio_mean",
    "frame_pc1_std",
    "frame_pc2_std",
    "frame_prediction_std",
    "mask_min_mean",
    "mask_ratio_mean",
    "quality_mean",
    "quality_min",
    "quality_max",
    "obj_quality_min_mean",
    "frame_quality_std",
    "frame_localized_ratio",
    "frame_partial_ratio",
    "frame_error_ratio",
    "frame_localized_quality_mean",
    "det_score_mean",
    "det_score_min",
    "verifier_accept_count",
    "verifier_reject_count",
    "verifier_reject_ratio",
    "candidate_count_mean",
    "candidate_count_max",
    "hypothesis_count_mean",
    "accepted_hypothesis_ratio",
    "accepted_candidate_mean",
    "rejected_candidate_mean",
    "fallback_count",
    "has_backfill",
]
for _feature_name in EXTRA_FEATURE_NAMES:
    FEATURES[_feature_name] = lambda row, name=_feature_name: row.extra.get(name, 0.0)

ACTIVE_FEATURES: set[str] | None = None


def iter_active_features() -> Iterable[tuple[str, Callable[[Row], float]]]:
    for name, getter in FEATURES.items():
        if ACTIVE_FEATURES is None or name in ACTIVE_FEATURES:
            yield name, getter


@contextlib.contextmanager
def active_features(names: set[str] | None):
    global ACTIVE_FEATURES
    old = ACTIVE_FEATURES
    ACTIVE_FEATURES = names
    try:
        yield
    finally:
        ACTIVE_FEATURES = old


def tune_stump(train: list[Row]) -> dict[str, object]:
    best_score = -1.0
    best: dict[str, object] = {}
    for feature, getter in iter_active_features():
        thresholds = candidate_thresholds([getter(row) for row in train], max_candidates=80)
        for threshold in thresholds:
            left_rows = [row for row in train if getter(row) < threshold]
            right_rows = [row for row in train if getter(row) >= threshold]
            if not left_rows or not right_rows:
                continue
            left_q = max(MAIN_QUANTILES, key=lambda key: mean_score(left_rows, lambda _row, q=key: q))
            right_q = max(MAIN_QUANTILES, key=lambda key: mean_score(right_rows, lambda _row, q=key: q))

            def choose(row: Row, f: Callable[[Row], float] = getter, t: float = threshold, lq: str = left_q, rq: str = right_q) -> str:
                return lq if f(row) < t else rq

            score = mean_score(train, choose)
            if score > best_score:
                best_score = score
                best = {
                    "family": "stump",
                    "train_mra": score,
                    "feature": feature,
                    "threshold": threshold,
                    "left_quantile": left_q,
                    "right_quantile": right_q,
                }
    return best


def _best_leaf_quantile(rows: list[Row], allowed_quantiles: list[str] = MAIN_QUANTILES) -> str:
    return max(allowed_quantiles, key=lambda key: mean_score(rows, lambda _row, q=key: q))


def _split_rows(rows: list[Row], feature: str, threshold: float) -> tuple[list[Row], list[Row]]:
    getter = FEATURES[feature]
    left = [row for row in rows if getter(row) < threshold]
    right = [row for row in rows if getter(row) >= threshold]
    return left, right


def _leaf_score(rows: list[Row], quantile: str) -> float:
    return sum(row.scores[quantile] for row in rows)


def _best_split(rows: list[Row], *, min_leaf: int, max_candidates: int = 36) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    best_total = -1.0
    for feature, getter in iter_active_features():
        thresholds = candidate_thresholds([getter(row) for row in rows], max_candidates=max_candidates)
        for threshold in thresholds:
            left, right = _split_rows(rows, feature, threshold)
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            left_q = _best_leaf_quantile(left)
            right_q = _best_leaf_quantile(right)
            total = _leaf_score(left, left_q) + _leaf_score(right, right_q)
            if total > best_total:
                best_total = total
                best = {
                    "feature": feature,
                    "threshold": threshold,
                    "left_quantile": left_q,
                    "right_quantile": right_q,
                    "total_score": total,
                    "left_count": len(left),
                    "right_count": len(right),
                }
    return best


def _build_depth2_node(rows: list[Row], *, min_leaf: int, depth: int) -> dict[str, object]:
    if depth == 0 or len(rows) < min_leaf * 2:
        q = _best_leaf_quantile(rows)
        return {"type": "leaf", "quantile": q, "count": len(rows), "train_mra": mean_score(rows, lambda _row, q=q: q)}
    split = _best_split(rows, min_leaf=min_leaf)
    if split is None:
        q = _best_leaf_quantile(rows)
        return {"type": "leaf", "quantile": q, "count": len(rows), "train_mra": mean_score(rows, lambda _row, q=q: q)}
    left, right = _split_rows(rows, str(split["feature"]), float(split["threshold"]))
    left_node = _build_depth2_node(left, min_leaf=min_leaf, depth=depth - 1)
    right_node = _build_depth2_node(right, min_leaf=min_leaf, depth=depth - 1)
    return {
        "type": "node",
        "feature": split["feature"],
        "threshold": split["threshold"],
        "count": len(rows),
        "left": left_node,
        "right": right_node,
    }


def _choose_tree(row: Row, node: dict[str, object]) -> str:
    while node.get("type") != "leaf":
        feature = str(node["feature"])
        threshold = float(node["threshold"])
        node = node["left"] if FEATURES[feature](row) < threshold else node["right"]
    return str(node["quantile"])


def tune_tree(train: list[Row], *, depth: int, min_leaf: int) -> dict[str, object]:
    tree = _build_depth2_node(train, min_leaf=min_leaf, depth=depth)
    choose = lambda row: _choose_tree(row, tree)
    return {
        "family": f"tree_depth{depth}",
        "train_mra": mean_score(train, choose),
        "min_leaf": min_leaf,
        "tree": tree,
    }


def make_chooser(rule: dict[str, object]) -> Callable[[Row], str]:
    family = rule["family"]
    if family == "near_gate":
        threshold = float(rule["threshold_p90"])
        near_q = str(rule["near_quantile"])
        return lambda row: near_q if row.p90 < threshold else "p90"
    if family == "spread_gate":
        threshold_p90 = float(rule["threshold_p90"])
        threshold_spread = float(rule["threshold_spread"])
        tight_q = str(rule["tight_quantile"])
        loose_q = str(rule["loose_quantile"])
        return lambda row: "p90" if row.p90 >= threshold_p90 else (
            tight_q if row.spread_p90_p25 < threshold_spread else loose_q
        )
    if family == "stump":
        feature = str(rule["feature"])
        threshold = float(rule["threshold"])
        left_q = str(rule["left_quantile"])
        right_q = str(rule["right_quantile"])
        getter = FEATURES[feature]
        return lambda row: left_q if getter(row) < threshold else right_q
    if str(family).startswith("tree_depth"):
        tree = rule["tree"]
        return lambda row: _choose_tree(row, tree)  # type: ignore[arg-type]
    raise ValueError(f"Unknown rule family: {family}")


def quantile_counts(rows: list[Row], choose: Callable[[Row], str]) -> dict[str, int]:
    counts = {key: 0 for key in QUANTILES}
    for row in rows:
        counts[choose(row)] += 1
    return {key: value for key, value in counts.items() if value}


def cross_validate(rows: list[Row], folds: int, seed: int, tuner: Callable[[list[Row]], dict[str, object]]) -> dict[str, object]:
    fold_rows = split_folds(rows, folds=folds, seed=seed)
    records = []
    all_val_scores = []
    all_counts = {key: 0 for key in QUANTILES}
    for fold_index in range(folds):
        val = fold_rows[fold_index]
        train = [row for index, fold in enumerate(fold_rows) if index != fold_index for row in fold]
        rule = tuner(train)
        choose = make_chooser(rule)
        val_mra = mean_score(val, choose)
        train_mra = mean_score(train, choose)
        counts = quantile_counts(val, choose)
        for key, value in counts.items():
            all_counts[key] += value
        all_val_scores.extend(row.scores[choose(row)] for row in val)
        records.append(
            {
                "fold": fold_index,
                "train_count": len(train),
                "val_count": len(val),
                "rule": rule,
                "train_mra": train_mra,
                "val_mra": val_mra,
                "val_mra_percent": val_mra * 100.0,
                "val_quantile_counts": counts,
            }
        )
    mean_val = sum(all_val_scores) / len(all_val_scores)
    fold_values = [record["val_mra"] for record in records]
    std = math.sqrt(sum((value - mean_val) ** 2 for value in fold_values) / len(fold_values))
    return {
        "folds": records,
        "cv_mra": mean_val,
        "cv_mra_percent": mean_val * 100.0,
        "fold_mra_std_percent": std * 100.0,
        "cv_quantile_counts": {key: value for key, value in all_counts.items() if value},
    }


def write_html(summary: dict[str, object], path: Path) -> None:
    methods = summary["methods"]
    rows = "\n".join(
        f"<tr><td>{name}</td><td>{info['mra_percent']:.2f}%</td><td>{info.get('std_percent', '')}</td><td>{info.get('counts', '')}</td></tr>"
        for name, info in methods.items()
    )
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>DADP Experiment</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
th {{ background: #f3f3f3; }}
pre {{ background: #f7f7f7; padding: 12px; overflow: auto; }}
</style>
<h1>DADP 离线规则实验</h1>
<table>
<tr><th>方法</th><th>MRA</th><th>fold std</th><th>选择计数</th></tr>
{rows}
</table>
<h2>完整 JSON</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline DADP gate experiments for object_abs_distance.")
    parser.add_argument(
        "--input-csv",
        default="docs/tasks/object_abs_distance/distribution_analysis_full_20260615/per_doc_quantile_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/tasks/object_abs_distance/dadp_experiment_full_20260617",
    )
    parser.add_argument(
        "--run-dir",
        default="/disk/wangzhe/SpatialScore/runs/object_abs_distance_full_qwen25vl_gpu3_20260615_160000",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--tree-min-leaf", type=int, default=30)
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated feature names for tree/stump search. Empty means all features.",
    )
    parser.add_argument(
        "--exclude-features",
        default="",
        help="Comma-separated feature names to remove from tree/stump search.",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.input_csv))
    rows = attach_extra_features(rows, Path(args.run_dir) if args.run_dir else None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_set: set[str] | None = None
    if args.features.strip():
        feature_set = {name.strip() for name in args.features.split(",") if name.strip()}
    if args.exclude_features.strip():
        excluded = {name.strip() for name in args.exclude_features.split(",") if name.strip()}
        feature_set = (set(FEATURES) if feature_set is None else feature_set) - excluded

    fixed = {
        key: {
            "mra": mean_score(rows, lambda _row, q=key: q),
            "mra_percent": mean_score(rows, lambda _row, q=key: q) * 100.0,
        }
        for key in QUANTILES
    }
    oracle = sum(max(row.scores.values()) for row in rows) / len(rows)

    with active_features(feature_set):
        experiments = {
            "near_gate_cv": cross_validate(rows, args.folds, args.seed, tune_near_gate),
            "spread_gate_cv": cross_validate(rows, args.folds, args.seed, tune_spread_gate),
            "stump_cv": cross_validate(rows, args.folds, args.seed, tune_stump),
            f"tree_depth{args.tree_depth}_cv": cross_validate(
                rows,
                args.folds,
                args.seed,
                lambda train: tune_tree(train, depth=args.tree_depth, min_leaf=args.tree_min_leaf),
            ),
        }

    methods: dict[str, dict[str, object]] = {
        f"fixed_{key}": {"mra_percent": value["mra_percent"], "counts": {key: len(rows)}}
        for key, value in fixed.items()
    }
    methods["oracle"] = {"mra_percent": oracle * 100.0, "counts": "per-sample best"}
    for name, result in experiments.items():
        methods[name] = {
            "mra_percent": result["cv_mra_percent"],
            "std_percent": round(float(result["fold_mra_std_percent"]), 3),
            "counts": result["cv_quantile_counts"],
        }

    summary = {
        "input_csv": str(args.input_csv),
        "count": len(rows),
        "folds": args.folds,
        "seed": args.seed,
        "active_features": sorted(feature_set) if feature_set is not None else sorted(FEATURES),
        "active_feature_count": len(feature_set) if feature_set is not None else len(FEATURES),
        "fixed": fixed,
        "oracle_mra": oracle,
        "oracle_mra_percent": oracle * 100.0,
        "experiments": experiments,
        "methods": methods,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "folds.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["experiment", "fold", "train_mra_percent", "val_mra_percent", "rule_json", "val_quantile_counts_json"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for experiment_name, result in experiments.items():
            for record in result["folds"]:
                writer.writerow(
                    {
                        "experiment": experiment_name,
                        "fold": record["fold"],
                        "train_mra_percent": record["train_mra"] * 100.0,
                        "val_mra_percent": record["val_mra_percent"],
                        "rule_json": json.dumps(record["rule"], ensure_ascii=False),
                        "val_quantile_counts_json": json.dumps(record["val_quantile_counts"], ensure_ascii=False),
                    }
                )
    write_html(summary, output_dir / "dadp_experiment.html")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
