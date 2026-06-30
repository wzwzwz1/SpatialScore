from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.experiment_dadp_abs_distance as dadp


EPS = 1e-9
FEATURE_NAMES = [
    "skew",
    "accepted_candidate_mean",
    "p90",
    "median",
    "iqr_norm",
    "lower_tail",
]


@dataclass(frozen=True)
class RuleResult:
    name: str
    rule: dict[str, object]
    val_mra: float
    counts: dict[str, int]
    switch_outcomes: dict[str, int]
    switch_delta_sum: float


def _feature(row: dadp.Row, name: str) -> float:
    if name == "skew":
        return row.skew
    if name == "p90":
        return row.p90
    if name == "median":
        return row.predictions["median"]
    if name == "iqr_norm":
        return row.iqr_norm
    if name == "lower_tail":
        return row.lower_tail
    return float(row.extra.get(name, 0.0))


def _delta(row: dadp.Row) -> float:
    return row.scores["p75"] - row.scores["p90"]


def _eval(rows: list[dadp.Row], choose: Callable[[dadp.Row], str]) -> tuple[float, dict[str, int], dict[str, int], float]:
    scores = []
    counts = {"p75": 0, "p90": 0}
    outcomes = {"benefit": 0, "tie": 0, "harm": 0}
    delta_sum = 0.0
    for row in rows:
        q = choose(row)
        scores.append(row.scores[q])
        counts[q] += 1
        if q == "p75":
            delta = _delta(row)
            delta_sum += delta
            if delta > EPS:
                outcomes["benefit"] += 1
            elif delta < -EPS:
                outcomes["harm"] += 1
            else:
                outcomes["tie"] += 1
    return sum(scores) / len(scores), counts, outcomes, delta_sum


def _thresholds(rows: list[dadp.Row], feature: str, max_candidates: int) -> list[float]:
    return dadp.candidate_thresholds([_feature(row, feature) for row in rows], max_candidates=max_candidates)


def tune_core2(train: list[dadp.Row], *, max_candidates: int = 40) -> dict[str, object]:
    best: dict[str, object] = {}
    best_score = -1.0
    for skew_t in _thresholds(train, "skew", max_candidates):
        for acc_t in _thresholds(train, "accepted_candidate_mean", max_candidates):
            def choose(row: dadp.Row, st: float = skew_t, at: float = acc_t) -> str:
                if row.skew < st:
                    return "p90" if _feature(row, "accepted_candidate_mean") < at else "p75"
                return "p75"

            score, _, _, _ = _eval(train, choose)
            if score > best_score:
                best_score = score
                best = {
                    "family": "core2",
                    "skew_threshold": skew_t,
                    "accepted_candidate_mean_threshold": acc_t,
                    "train_mra": score,
                }
    return best


def choose_core2(row: dadp.Row, rule: dict[str, object]) -> str:
    if row.skew < float(rule["skew_threshold"]):
        return "p90" if _feature(row, "accepted_candidate_mean") < float(rule["accepted_candidate_mean_threshold"]) else "p75"
    return "p75"


def tune_core2_far_guard(train: list[dadp.Row], *, proxy: str, max_candidates: int = 26) -> dict[str, object]:
    best: dict[str, object] = {}
    best_score = -1.0
    far_thresholds = _thresholds(train, proxy, max_candidates)
    skew_thresholds = _thresholds(train, "skew", max_candidates)
    acc_thresholds = _thresholds(train, "accepted_candidate_mean", max_candidates)
    for far_t in far_thresholds:
        for skew_t in skew_thresholds:
            for acc_t in acc_thresholds:
                def choose(row: dadp.Row, ft: float = far_t, st: float = skew_t, at: float = acc_t) -> str:
                    if _feature(row, proxy) >= ft:
                        return "p90"
                    if row.skew >= st:
                        return "p75"
                    if _feature(row, "accepted_candidate_mean") >= at:
                        return "p75"
                    return "p90"

                score, _, _, _ = _eval(train, choose)
                if score > best_score:
                    best_score = score
                    best = {
                        "family": "core2_far_guard",
                        "proxy": proxy,
                        "far_threshold": far_t,
                        "skew_threshold": skew_t,
                        "accepted_candidate_mean_threshold": acc_t,
                        "train_mra": score,
                    }
    return best


def choose_core2_far_guard(row: dadp.Row, rule: dict[str, object]) -> str:
    proxy = str(rule["proxy"])
    if _feature(row, proxy) >= float(rule["far_threshold"]):
        return "p90"
    if row.skew >= float(rule["skew_threshold"]):
        return "p75"
    if _feature(row, "accepted_candidate_mean") >= float(rule["accepted_candidate_mean_threshold"]):
        return "p75"
    return "p90"


def _weighted_mean_delta(rows: list[dadp.Row], *, margin: float) -> float:
    weighted_sum = 0.0
    weight_total = 0.0
    for row in rows:
        delta = _delta(row)
        weight = abs(delta)
        if weight <= margin:
            continue
        weighted_sum += delta * weight
        weight_total += weight
    if weight_total <= EPS:
        return 0.0
    return weighted_sum / weight_total


def _weighted_sse(rows: list[dadp.Row], *, margin: float) -> float:
    mean = _weighted_mean_delta(rows, margin=margin)
    total = 0.0
    for row in rows:
        delta = _delta(row)
        weight = abs(delta)
        if weight <= margin:
            continue
        total += weight * (delta - mean) ** 2
    return total


def _build_delta_tree(
    rows: list[dadp.Row],
    *,
    depth: int,
    min_leaf: int,
    margin: float,
    max_candidates: int,
    features: list[str],
) -> dict[str, object]:
    prediction = _weighted_mean_delta(rows, margin=margin)
    if depth == 0 or len(rows) < min_leaf * 2:
        return {"type": "leaf", "count": len(rows), "predicted_delta": prediction}

    base = _weighted_sse(rows, margin=margin)
    best: dict[str, object] | None = None
    best_loss = base
    for feature in features:
        for threshold in _thresholds(rows, feature, max_candidates):
            left = [row for row in rows if _feature(row, feature) < threshold]
            right = [row for row in rows if _feature(row, feature) >= threshold]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            loss = _weighted_sse(left, margin=margin) + _weighted_sse(right, margin=margin)
            if loss < best_loss:
                best_loss = loss
                best = {"feature": feature, "threshold": threshold, "left": left, "right": right}
    if best is None:
        return {"type": "leaf", "count": len(rows), "predicted_delta": prediction}
    return {
        "type": "node",
        "feature": best["feature"],
        "threshold": best["threshold"],
        "count": len(rows),
        "left": _build_delta_tree(
            best["left"],
            depth=depth - 1,
            min_leaf=min_leaf,
            margin=margin,
            max_candidates=max_candidates,
            features=features,
        ),
        "right": _build_delta_tree(
            best["right"],
            depth=depth - 1,
            min_leaf=min_leaf,
            margin=margin,
            max_candidates=max_candidates,
            features=features,
        ),
    }


def _predict_delta(row: dadp.Row, node: dict[str, object]) -> float:
    while node.get("type") != "leaf":
        feature = str(node["feature"])
        threshold = float(node["threshold"])
        node = node["left"] if _feature(row, feature) < threshold else node["right"]  # type: ignore[index]
    return float(node["predicted_delta"])


def tune_delta_tree(
    train: list[dadp.Row],
    *,
    margin: float,
    switch_margin: float,
    far_proxy: str | None = None,
    max_candidates: int = 24,
) -> dict[str, object]:
    best: dict[str, object] = {}
    best_score = -1.0
    far_thresholds = [math.inf]
    if far_proxy:
        far_thresholds = _thresholds(train, far_proxy, max_candidates)
    for far_t in far_thresholds:
        tree_rows = [row for row in train if not far_proxy or _feature(row, far_proxy) < far_t]
        if len(tree_rows) < 80:
            continue
        tree = _build_delta_tree(
            tree_rows,
            depth=2,
            min_leaf=25,
            margin=margin,
            max_candidates=max_candidates,
            features=FEATURE_NAMES,
        )

        def choose(row: dadp.Row, ft: float = far_t, tree_node: dict[str, object] = tree) -> str:
            if far_proxy and _feature(row, far_proxy) >= ft:
                return "p90"
            return "p75" if _predict_delta(row, tree_node) > switch_margin else "p90"

        score, _, _, _ = _eval(train, choose)
        if score > best_score:
            best_score = score
            best = {
                "family": "delta_tree",
                "margin": margin,
                "switch_margin": switch_margin,
                "far_proxy": far_proxy,
                "far_threshold": far_t,
                "tree": tree,
                "train_mra": score,
            }
    return best


def choose_delta_tree(row: dadp.Row, rule: dict[str, object]) -> str:
    far_proxy = rule.get("far_proxy")
    if far_proxy and _feature(row, str(far_proxy)) >= float(rule["far_threshold"]):
        return "p90"
    return "p75" if _predict_delta(row, rule["tree"]) > float(rule["switch_margin"]) else "p90"  # type: ignore[arg-type]


def cross_validate(
    rows: list[dadp.Row],
    *,
    name: str,
    tuner: Callable[[list[dadp.Row]], dict[str, object]],
    chooser: Callable[[dadp.Row, dict[str, object]], str],
    folds: int,
    seed: int,
) -> dict[str, object]:
    fold_rows = dadp.split_folds(rows, folds=folds, seed=seed)
    records = []
    all_scores = []
    all_counts = {"p75": 0, "p90": 0}
    all_outcomes = {"benefit": 0, "tie": 0, "harm": 0}
    delta_sum = 0.0
    for fold_index in range(folds):
        val = fold_rows[fold_index]
        train = [row for index, fold in enumerate(fold_rows) if index != fold_index for row in fold]
        rule = tuner(train)

        def choose(row: dadp.Row, current_rule: dict[str, object] = rule) -> str:
            return chooser(row, current_rule)

        val_mra, counts, outcomes, fold_delta = _eval(val, choose)
        for key, value in counts.items():
            all_counts[key] += value
        for key, value in outcomes.items():
            all_outcomes[key] += value
        delta_sum += fold_delta
        all_scores.extend(row.scores[choose(row)] for row in val)
        records.append(
            {
                "fold": fold_index,
                "rule": rule,
                "val_mra_percent": val_mra * 100.0,
                "counts": counts,
                "switch_outcomes": outcomes,
                "switch_delta_sum": fold_delta,
            }
        )
    return {
        "name": name,
        "cv_mra_percent": sum(all_scores) / len(all_scores) * 100.0,
        "counts": all_counts,
        "switch_outcomes": all_outcomes,
        "switch_delta_sum": delta_sum,
        "folds": records,
    }


def load_questions(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["doc_id"]): row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="docs/tasks/object_abs_distance/distribution_analysis_full_20260615/per_doc_quantile_scores.csv")
    parser.add_argument("--run-dir", default="runs/object_abs_distance_full_qwen25vl_gpu3_20260615_160000")
    parser.add_argument("--output-dir", default="docs/tasks/object_abs_distance/dadp_v2_delta_aware")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    rows = dadp.load_rows(input_csv)
    rows = dadp.attach_extra_features(rows, Path(args.run_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_p90 = sum(row.scores["p90"] for row in rows) / len(rows) * 100.0
    fixed_p75 = sum(row.scores["p75"] for row in rows) / len(rows) * 100.0
    oracle = sum(max(row.scores["p75"], row.scores["p90"]) for row in rows) / len(rows) * 100.0

    experiments = [
        cross_validate(
            rows,
            name="v1_core2",
            tuner=lambda train: tune_core2(train),
            chooser=choose_core2,
            folds=args.folds,
            seed=args.seed,
        ),
        cross_validate(
            rows,
            name="v2_core2_far_guard_p90",
            tuner=lambda train: tune_core2_far_guard(train, proxy="p90"),
            chooser=choose_core2_far_guard,
            folds=args.folds,
            seed=args.seed,
        ),
        cross_validate(
            rows,
            name="v2_core2_far_guard_median",
            tuner=lambda train: tune_core2_far_guard(train, proxy="median"),
            chooser=choose_core2_far_guard,
            folds=args.folds,
            seed=args.seed,
        ),
        cross_validate(
            rows,
            name="v3_delta_tree",
            tuner=lambda train: tune_delta_tree(train, margin=0.1, switch_margin=0.0),
            chooser=choose_delta_tree,
            folds=args.folds,
            seed=args.seed,
        ),
        cross_validate(
            rows,
            name="v4_delta_tree_far_guard_p90",
            tuner=lambda train: tune_delta_tree(train, margin=0.1, switch_margin=0.0, far_proxy="p90"),
            chooser=choose_delta_tree,
            folds=args.folds,
            seed=args.seed,
        ),
        cross_validate(
            rows,
            name="v4_delta_tree_far_guard_median",
            tuner=lambda train: tune_delta_tree(train, margin=0.1, switch_margin=0.0, far_proxy="median"),
            chooser=choose_delta_tree,
            folds=args.folds,
            seed=args.seed,
        ),
    ]

    summary = {
        "count": len(rows),
        "folds": args.folds,
        "seed": args.seed,
        "fixed": {"p90_mra_percent": fixed_p90, "p75_mra_percent": fixed_p75},
        "p75_p90_oracle_percent": oracle,
        "experiments": experiments,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_dir / "summary_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "cv_mra_percent", "p75_count", "p90_count", "benefit", "tie", "harm", "switch_delta_sum"],
        )
        writer.writeheader()
        for exp in experiments:
            writer.writerow(
                {
                    "name": exp["name"],
                    "cv_mra_percent": exp["cv_mra_percent"],
                    "p75_count": exp["counts"]["p75"],
                    "p90_count": exp["counts"]["p90"],
                    "benefit": exp["switch_outcomes"]["benefit"],
                    "tie": exp["switch_outcomes"]["tie"],
                    "harm": exp["switch_outcomes"]["harm"],
                    "switch_delta_sum": exp["switch_delta_sum"],
                }
            )
    print(json.dumps({"output_dir": str(output_dir), "fixed_p90": fixed_p90, "experiments": experiments}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
