from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.experiment_dadp_abs_distance as dadp
import scripts.experiment_dadp_v2_abs_distance as dadp_v2


def _load_rows(csv_path: str, run_dir: str) -> list[dadp.Row]:
    rows = dadp.load_rows(Path(csv_path))
    if run_dir:
        rows = dadp.attach_extra_features(rows, Path(run_dir))
    return rows


def _fixed_mra(rows: list[dadp.Row], quantile: str) -> float:
    return sum(row.scores[quantile] for row in rows) / len(rows) * 100.0 if rows else 0.0


def _oracle_mra(rows: list[dadp.Row]) -> float:
    return sum(max(row.scores["p75"], row.scores["p90"]) for row in rows) / len(rows) * 100.0 if rows else 0.0


def _evaluate(
    rows: list[dadp.Row],
    *,
    name: str,
    rule: dict[str, object],
    chooser: Callable[[dadp.Row, dict[str, object]], str],
) -> dict[str, object]:
    score, counts, outcomes, switch_delta = dadp_v2._eval(rows, lambda row: chooser(row, rule))
    return {
        "name": name,
        "mra_percent": score * 100.0,
        "counts": counts,
        "switch_outcomes": outcomes,
        "switch_delta_sum": switch_delta,
        "rule": rule,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DADP p90->p75 gate on train records and evaluate on held-out records.")
    parser.add_argument("--train-csv", required=True, help="Train per_doc_quantile_scores.csv.")
    parser.add_argument("--train-run-dir", required=True, help="Train run directory containing doc_*/result.json.")
    parser.add_argument(
        "--test-csv",
        default="docs/tasks/object_abs_distance/distribution_analysis_alias_fix_merged_20260617/per_doc_quantile_scores.csv",
    )
    parser.add_argument(
        "--test-run-dir",
        default="runs/object_abs_distance_full_alias_fix_merged_20260617_docs",
    )
    parser.add_argument("--output-dir", default="docs/tasks/object_abs_distance/dadp_train_classifier")
    parser.add_argument("--max-candidates", type=int, default=26)
    parser.add_argument("--delta-margin", type=float, default=0.1)
    parser.add_argument("--switch-margin", type=float, default=0.0)
    args = parser.parse_args()

    train_rows = _load_rows(args.train_csv, args.train_run_dir)
    test_rows = _load_rows(args.test_csv, args.test_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rules = {
        "core2": (
            dadp_v2.tune_core2(train_rows, max_candidates=args.max_candidates),
            dadp_v2.choose_core2,
        ),
        "core2_far_guard_p90": (
            dadp_v2.tune_core2_far_guard(train_rows, proxy="p90", max_candidates=args.max_candidates),
            dadp_v2.choose_core2_far_guard,
        ),
        "core2_far_guard_median": (
            dadp_v2.tune_core2_far_guard(train_rows, proxy="median", max_candidates=args.max_candidates),
            dadp_v2.choose_core2_far_guard,
        ),
        "delta_tree_far_guard_p90": (
            dadp_v2.tune_delta_tree(
                train_rows,
                margin=args.delta_margin,
                switch_margin=args.switch_margin,
                far_proxy="p90",
                max_candidates=args.max_candidates,
            ),
            dadp_v2.choose_delta_tree,
        ),
    }

    experiments = []
    for name, (rule, chooser) in rules.items():
        experiments.append(
            {
                "name": name,
                "train": _evaluate(train_rows, name=name, rule=rule, chooser=chooser),
                "test": _evaluate(test_rows, name=name, rule=rule, chooser=chooser),
            }
        )

    summary = {
        "train": {
            "count": len(train_rows),
            "fixed_p90_mra_percent": _fixed_mra(train_rows, "p90"),
            "fixed_p75_mra_percent": _fixed_mra(train_rows, "p75"),
            "p75_p90_oracle_percent": _oracle_mra(train_rows),
        },
        "test": {
            "count": len(test_rows),
            "fixed_p90_mra_percent": _fixed_mra(test_rows, "p90"),
            "fixed_p75_mra_percent": _fixed_mra(test_rows, "p75"),
            "p75_p90_oracle_percent": _oracle_mra(test_rows),
        },
        "experiments": experiments,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
