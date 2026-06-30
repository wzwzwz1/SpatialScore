from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import scripts.experiment_dadp_abs_distance as dadp
import scripts.experiment_dadp_v2_abs_distance as dadp_v2


BASE_FEATURES = [
    "p90",
    "median",
    "p75",
    "p25",
    "p10",
    "p05",
    "spread_p90_p25",
    "lower_tail",
    "skew",
    "ratio_p90_median",
    "iqr_norm",
    "p90_minus_p75",
    "p75_minus_median",
    "median_minus_p25",
    "gt_proxy_bucket_p90_lt1",
    "gt_proxy_bucket_p90_1_3",
    "gt_proxy_bucket_p90_ge3",
]

EXTRA_FEATURES = [
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


def _delta(row: dadp.Row) -> float:
    return row.scores["p75"] - row.scores["p90"]


def _feature(row: dadp.Row, name: str) -> float:
    if name == "p90":
        return row.predictions["p90"]
    if name == "median":
        return row.predictions["median"]
    if name == "p75":
        return row.predictions["p75"]
    if name == "p25":
        return row.predictions["p25"]
    if name == "p10":
        return row.predictions["p10"]
    if name == "p05":
        return row.predictions["p05"]
    if name == "spread_p90_p25":
        return row.spread_p90_p25
    if name == "lower_tail":
        return row.lower_tail
    if name == "skew":
        return row.skew
    if name == "ratio_p90_median":
        return row.ratio_p90_median
    if name == "iqr_norm":
        return row.iqr_norm
    if name == "p90_minus_p75":
        return row.predictions["p90"] - row.predictions["p75"]
    if name == "p75_minus_median":
        return row.predictions["p75"] - row.predictions["median"]
    if name == "median_minus_p25":
        return row.predictions["median"] - row.predictions["p25"]
    if name == "gt_proxy_bucket_p90_lt1":
        return 1.0 if row.predictions["p90"] < 1.0 else 0.0
    if name == "gt_proxy_bucket_p90_1_3":
        return 1.0 if 1.0 <= row.predictions["p90"] < 3.0 else 0.0
    if name == "gt_proxy_bucket_p90_ge3":
        return 1.0 if row.predictions["p90"] >= 3.0 else 0.0
    return float(row.extra.get(name, 0.0))


def _matrix(rows: list[dadp.Row], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[_feature(row, name) for name in feature_names] for row in rows], dtype=np.float64)


def _score_rows(rows: list[dadp.Row], choices: list[str]) -> dict[str, Any]:
    scores = [row.scores[q] for row, q in zip(rows, choices)]
    outcomes = {"benefit": 0, "tie": 0, "harm": 0}
    switch_delta = 0.0
    counts = {"p75": 0, "p90": 0}
    for row, q in zip(rows, choices):
        counts[q] += 1
        if q != "p75":
            continue
        delta = _delta(row)
        switch_delta += delta
        if delta > dadp.EPS:
            outcomes["benefit"] += 1
        elif delta < -dadp.EPS:
            outcomes["harm"] += 1
        else:
            outcomes["tie"] += 1
    return {
        "mra_percent": float(np.mean(scores) * 100.0) if scores else 0.0,
        "counts": counts,
        "switch_outcomes": outcomes,
        "switch_delta_sum": switch_delta,
    }


def _apply_far_guard(rows: list[dadp.Row], choices: list[str], *, proxy: str | None, threshold: float | None) -> list[str]:
    if proxy is None or threshold is None:
        return choices
    guarded: list[str] = []
    for row, choice in zip(rows, choices):
        guarded.append("p90" if _feature(row, proxy) >= threshold else choice)
    return guarded


def _threshold_candidates(values: np.ndarray, max_count: int = 80) -> list[float]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return [0.0]
    unique = np.unique(finite)
    if unique.size <= max_count:
        mids = [(float(left) + float(right)) / 2.0 for left, right in zip(unique[:-1], unique[1:])]
        return [float(unique[0]) - dadp.EPS, *mids, float(unique[-1]) + dadp.EPS]
    qs = np.linspace(0.0, 1.0, max_count)
    return sorted({float(np.quantile(finite, q)) for q in qs})


def _tune_threshold(train_rows: list[dadp.Row], predictions: np.ndarray, *, max_count: int = 80) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.0
    best_summary: dict[str, Any] = {}
    best_score = -1.0
    for threshold in _threshold_candidates(predictions, max_count=max_count):
        choices = ["p75" if value > threshold else "p90" for value in predictions]
        summary = _score_rows(train_rows, choices)
        if summary["mra_percent"] > best_score:
            best_score = summary["mra_percent"]
            best_threshold = threshold
            best_summary = summary
    return best_threshold, best_summary


def _tune_threshold_and_far_guard(
    train_rows: list[dadp.Row],
    predictions: np.ndarray,
    *,
    max_count: int = 80,
) -> tuple[float, str | None, float | None, dict[str, Any]]:
    best_threshold = 0.0
    best_proxy: str | None = None
    best_far_threshold: float | None = None
    best_summary: dict[str, Any] = {}
    best_score = -1.0
    far_options: list[tuple[str | None, float | None]] = [(None, None)]
    for proxy in ["p90", "p75", "median"]:
        for threshold in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
            far_options.append((proxy, threshold))
    for threshold in _threshold_candidates(predictions, max_count=max_count):
        base_choices = ["p75" if value > threshold else "p90" for value in predictions]
        for proxy, far_threshold in far_options:
            choices = _apply_far_guard(train_rows, base_choices, proxy=proxy, threshold=far_threshold)
            summary = _score_rows(train_rows, choices)
            if summary["mra_percent"] > best_score:
                best_score = summary["mra_percent"]
                best_threshold = threshold
                best_proxy = proxy
                best_far_threshold = far_threshold
                best_summary = summary
    return best_threshold, best_proxy, best_far_threshold, best_summary


def _clean_rows(rows: list[dadp.Row], *, mode: str, margin: float) -> list[dadp.Row]:
    if mode == "none":
        return rows
    if mode == "non_tie":
        return [row for row in rows if abs(_delta(row)) > margin]
    if mode == "switch_relevant":
        return [row for row in rows if abs(_delta(row)) > margin and max(row.scores["p75"], row.scores["p90"]) > 0.0]
    if mode == "drop_far_proxy":
        return [row for row in rows if row.predictions["p90"] < 3.0]
    if mode == "quality_loose":
        return [row for row in rows if _passes_quality(row, strict=False)]
    if mode == "quality_strict":
        return [row for row in rows if _passes_quality(row, strict=True)]
    if mode == "non_tie_quality_loose":
        return [row for row in rows if abs(_delta(row)) > margin and _passes_quality(row, strict=False)]
    if mode == "non_tie_quality_strict":
        return [row for row in rows if abs(_delta(row)) > margin and _passes_quality(row, strict=True)]
    raise ValueError(f"Unknown cleaning mode: {mode}")


def _passes_quality(row: dadp.Row, *, strict: bool) -> bool:
    selected_localized_ratio = float(row.extra.get("selected_localized_ratio", 0.0))
    both_object_frame_ratio = float(row.extra.get("both_object_frame_ratio", 0.0))
    selected_error_ratio = float(row.extra.get("selected_error_ratio", 1.0))
    selected_quality_mean = float(row.extra.get("selected_quality_mean", 0.0))
    if strict:
        return (
            selected_localized_ratio >= 0.45
            and both_object_frame_ratio >= 0.20
            and selected_error_ratio <= 0.15
            and selected_quality_mean >= 0.12
        )
    return (
        selected_localized_ratio >= 0.35
        and both_object_frame_ratio >= 0.15
        and selected_error_ratio <= 0.25
        and selected_quality_mean >= 0.08
    )


def _quality_weight(row: dadp.Row) -> float:
    selected_localized_ratio = max(0.0, min(1.0, float(row.extra.get("selected_localized_ratio", 0.0))))
    both_object_frame_ratio = max(0.0, min(1.0, float(row.extra.get("both_object_frame_ratio", 0.0))))
    selected_error_ratio = max(0.0, min(1.0, float(row.extra.get("selected_error_ratio", 1.0))))
    selected_quality_mean = max(0.0, min(1.0, float(row.extra.get("selected_quality_mean", 0.0))))
    det_score_mean = max(0.0, min(1.0, float(row.extra.get("det_score_mean", 0.0))))
    return max(
        0.05,
        0.30 * selected_localized_ratio
        + 0.25 * both_object_frame_ratio
        + 0.20 * (1.0 - selected_error_ratio)
        + 0.15 * selected_quality_mean
        + 0.10 * det_score_mean,
    )


def _label_stats(rows: list[dadp.Row], *, margin: float) -> dict[str, Any]:
    deltas = [_delta(row) for row in rows]
    benefit = sum(1 for value in deltas if value > margin)
    harm = sum(1 for value in deltas if value < -margin)
    tie = len(rows) - benefit - harm
    all_quantile_tie = sum(1 for row in rows if len({row.scores[key] for key in dadp.QUANTILES}) == 1)
    p90_zero = sum(1 for row in rows if row.scores["p90"] == 0.0)
    p75_zero = sum(1 for row in rows if row.scores["p75"] == 0.0)
    return {
        "count": len(rows),
        "benefit_p75_over_p90": benefit,
        "harm_p75_under_p90": harm,
        "tie_abs_delta_le_margin": tie,
        "margin": margin,
        "delta_sum": float(sum(deltas)),
        "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "delta_abs_mean": float(np.mean(np.abs(deltas))) if deltas else 0.0,
        "all_quantile_mra_tie": all_quantile_tie,
        "p90_zero_mra": p90_zero,
        "p75_zero_mra": p75_zero,
    }


def _model_specs(seed: int) -> list[tuple[str, Any, str]]:
    specs: list[tuple[str, Any, str]] = [
        (
            "ridge_delta",
            Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            "regression",
        ),
        (
            "logistic_switch",
            Pipeline(
                [
                    ("imputer", SimpleImputer()),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed)),
                ]
            ),
            "classification",
        ),
        (
            "decision_tree_delta_d2",
            DecisionTreeRegressor(max_depth=2, min_samples_leaf=35, random_state=seed),
            "regression",
        ),
        (
            "decision_tree_delta_d3",
            DecisionTreeRegressor(max_depth=3, min_samples_leaf=35, random_state=seed),
            "regression",
        ),
        (
            "random_forest_delta",
            RandomForestRegressor(n_estimators=160, max_depth=3, min_samples_leaf=25, random_state=seed, n_jobs=4),
            "regression",
        ),
        (
            "gbrt_delta",
            GradientBoostingRegressor(n_estimators=80, learning_rate=0.04, max_depth=2, min_samples_leaf=25, random_state=seed),
            "regression",
        ),
    ]
    try:
        from xgboost import XGBRegressor

        specs.append(
            (
                "xgb_delta",
                XGBRegressor(
                    n_estimators=80,
                    max_depth=2,
                    learning_rate=0.04,
                    min_child_weight=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=5.0,
                    objective="reg:squarederror",
                    random_state=seed,
                    n_jobs=4,
                ),
                "regression",
            )
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        specs.append(
            (
                "lgbm_delta",
                LGBMRegressor(
                    n_estimators=100,
                    max_depth=2,
                    learning_rate=0.04,
                    min_child_samples=35,
                    reg_lambda=5.0,
                    random_state=seed,
                    n_jobs=4,
                    verbose=-1,
                ),
                "regression",
            )
        )
    except Exception:
        pass
    return specs


def _fit_predict(
    *,
    name: str,
    model: Any,
    model_type: str,
    train_rows_for_fit: list[dadp.Row],
    train_rows_for_tune: list[dadp.Row],
    test_rows: list[dadp.Row],
    feature_names: list[str],
    margin: float,
    enable_far_guard: bool,
    weight_mode: str,
) -> dict[str, Any]:
    x_fit = _matrix(train_rows_for_fit, feature_names)
    x_tune = _matrix(train_rows_for_tune, feature_names)
    x_test = _matrix(test_rows, feature_names)
    deltas_fit = np.asarray([_delta(row) for row in train_rows_for_fit], dtype=np.float64)

    if model_type == "classification":
        y_fit = (deltas_fit > margin).astype(int)
        if len(set(y_fit.tolist())) < 2:
            return {"name": name, "status": "skipped_one_class"}
        sample_weight = _sample_weights(train_rows_for_fit, deltas_fit, mode=weight_mode)
        model.fit(x_fit, y_fit, model__sample_weight=sample_weight) if isinstance(model, Pipeline) else model.fit(x_fit, y_fit, sample_weight=sample_weight)
        tune_pred = model.predict_proba(x_tune)[:, 1]
        test_pred = model.predict_proba(x_test)[:, 1]
        try:
            train_auc = roc_auc_score((np.asarray([_delta(row) for row in train_rows_for_tune]) > margin).astype(int), tune_pred)
        except Exception:
            train_auc = None
    else:
        sample_weight = _sample_weights(train_rows_for_fit, deltas_fit, mode=weight_mode)
        model.fit(x_fit, deltas_fit, model__sample_weight=sample_weight) if isinstance(model, Pipeline) else model.fit(x_fit, deltas_fit, sample_weight=sample_weight)
        tune_pred = np.asarray(model.predict(x_tune), dtype=np.float64)
        test_pred = np.asarray(model.predict(x_test), dtype=np.float64)
        train_auc = None

    if enable_far_guard:
        threshold, far_proxy, far_threshold, train_summary = _tune_threshold_and_far_guard(train_rows_for_tune, tune_pred)
    else:
        threshold, train_summary = _tune_threshold(train_rows_for_tune, tune_pred)
        far_proxy, far_threshold = None, None
    test_choices = ["p75" if value > threshold else "p90" for value in test_pred]
    test_choices = _apply_far_guard(test_rows, test_choices, proxy=far_proxy, threshold=far_threshold)
    test_summary = _score_rows(test_rows, test_choices)
    return {
        "name": name,
        "status": "ok",
        "model_type": model_type,
        "threshold": threshold,
        "far_guard": {"proxy": far_proxy, "threshold": far_threshold},
        "train": train_summary,
        "test": test_summary,
        "train_auc": train_auc,
        "weight_mode": weight_mode,
    }


def _sample_weights(rows: list[dadp.Row], deltas: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "delta":
        return np.maximum(np.abs(deltas), 0.05)
    if mode == "delta_quality":
        quality = np.asarray([_quality_weight(row) for row in rows], dtype=np.float64)
        return np.maximum(np.abs(deltas), 0.05) * quality
    if mode == "quality":
        return np.asarray([_quality_weight(row) for row in rows], dtype=np.float64)
    raise ValueError(f"Unknown weight mode: {mode}")


def _fixed(rows: list[dadp.Row], quantile: str) -> float:
    return float(np.mean([row.scores[quantile] for row in rows]) * 100.0) if rows else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep DADP p90/p75 model families and basic train-data cleaning options.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--train-run-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--test-run-dir", required=True)
    parser.add_argument("--output-dir", default="docs/tasks/object_abs_distance/dadp_model_sweep")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = dadp.attach_extra_features(dadp.load_rows(Path(args.train_csv)), Path(args.train_run_dir))
    test_rows = dadp.attach_extra_features(dadp.load_rows(Path(args.test_csv)), Path(args.test_run_dir))
    feature_names = BASE_FEATURES + EXTRA_FEATURES

    data_quality = {
        "train": _label_stats(train_rows, margin=args.margin),
        "test": _label_stats(test_rows, margin=args.margin),
        "cleaning_modes": {},
    }

    experiments = []
    for clean_mode in [
        "none",
        "non_tie",
        "switch_relevant",
        "drop_far_proxy",
        "quality_loose",
        "quality_strict",
        "non_tie_quality_loose",
        "non_tie_quality_strict",
    ]:
        fit_rows = _clean_rows(train_rows, mode=clean_mode, margin=args.margin)
        data_quality["cleaning_modes"][clean_mode] = _label_stats(fit_rows, margin=args.margin)
        if len(fit_rows) < 80:
            continue
        for model_name, model, model_type in _model_specs(args.seed):
            for enable_far_guard in [False, True]:
                for weight_mode in ["delta", "delta_quality", "quality"]:
                    suffix = ":far_guard" if enable_far_guard else ""
                    if weight_mode != "delta":
                        suffix = f"{suffix}:{weight_mode}"
                    result = _fit_predict(
                        name=f"{clean_mode}:{model_name}{suffix}",
                        model=model,
                        model_type=model_type,
                        train_rows_for_fit=fit_rows,
                        train_rows_for_tune=train_rows,
                        test_rows=test_rows,
                        feature_names=feature_names,
                        margin=args.margin,
                        enable_far_guard=enable_far_guard,
                        weight_mode=weight_mode,
                    )
                    result["clean_mode"] = clean_mode
                    result["fit_count"] = len(fit_rows)
                    result["enable_far_guard"] = enable_far_guard
                    experiments.append(result)

    summary = {
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "features": feature_names,
        "fixed": {
            "train_p90": _fixed(train_rows, "p90"),
            "train_p75": _fixed(train_rows, "p75"),
            "test_p90": _fixed(test_rows, "p90"),
            "test_p75": _fixed(test_rows, "p75"),
            "train_p75_p90_oracle": float(np.mean([max(row.scores["p75"], row.scores["p90"]) for row in train_rows]) * 100.0),
            "test_p75_p90_oracle": float(np.mean([max(row.scores["p75"], row.scores["p90"]) for row in test_rows]) * 100.0),
        },
        "data_quality": data_quality,
        "experiments": sorted(
            experiments,
            key=lambda item: item.get("test", {}).get("mra_percent", -1.0) if item.get("status") == "ok" else -1.0,
            reverse=True,
        ),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "summary_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "clean_mode",
                "fit_count",
                "status",
                "weight_mode",
                "test_mra_percent",
                "train_mra_percent",
                "test_p75_count",
                "test_benefit",
                "test_tie",
                "test_harm",
                "test_switch_delta_sum",
                "threshold",
                "far_guard_proxy",
                "far_guard_threshold",
            ],
        )
        writer.writeheader()
        for item in summary["experiments"]:
            test = item.get("test") or {}
            train = item.get("train") or {}
            writer.writerow(
                {
                    "name": item.get("name"),
                    "clean_mode": item.get("clean_mode"),
                    "fit_count": item.get("fit_count"),
                    "status": item.get("status"),
                    "weight_mode": item.get("weight_mode"),
                    "test_mra_percent": test.get("mra_percent"),
                    "train_mra_percent": train.get("mra_percent"),
                    "test_p75_count": (test.get("counts") or {}).get("p75"),
                    "test_benefit": (test.get("switch_outcomes") or {}).get("benefit"),
                    "test_tie": (test.get("switch_outcomes") or {}).get("tie"),
                    "test_harm": (test.get("switch_outcomes") or {}).get("harm"),
                    "test_switch_delta_sum": test.get("switch_delta_sum"),
                    "threshold": item.get("threshold"),
                    "far_guard_proxy": (item.get("far_guard") or {}).get("proxy"),
                    "far_guard_threshold": (item.get("far_guard") or {}).get("threshold"),
                }
            )

    print(json.dumps({"output_dir": str(output_dir), "best": summary["experiments"][:8], "data_quality": data_quality}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
