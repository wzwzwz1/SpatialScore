from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


QUANTILES = ["min", "p05", "p10", "p25", "median", "p75", "p90"]


def mean_relative_accuracy(prediction: float | None, ground_truth: float | None) -> float:
    if prediction is None or ground_truth is None:
        return 0.0
    if not math.isfinite(float(prediction)) or not math.isfinite(float(ground_truth)):
        return 0.0
    if float(ground_truth) == 0.0:
        return 1.0 if float(prediction) == 0.0 else 0.0
    rel_error = abs(float(prediction) - float(ground_truth)) / abs(float(ground_truth))
    confidences = [0.50 + 0.05 * index for index in range(10)]
    return sum(1 for confidence in confidences if rel_error <= 1.0 - confidence) / len(confidences)


def load_records(pattern: str) -> list[dict[str, Any]]:
    by_doc: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(pattern)):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            doc_id = int(record["doc_id"])
            old = by_doc.get(doc_id)
            if old is None or (old.get("status") != "success" and record.get("status") == "success"):
                by_doc[doc_id] = record
    return [by_doc[key] for key in sorted(by_doc)]


def gt_bucket(value: float) -> str:
    if value < 1.0:
        return "<1m"
    if value < 3.0:
        return "1-3m"
    return ">=3m"


def analyze(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixed_scores: dict[str, list[float]] = {key: [] for key in QUANTILES}
    fixed_errors: dict[str, list[float]] = {key: [] for key in QUANTILES}

    for record in records:
        stats = (record.get("distance_payload") or {}).get("distance_stats") or {}
        if record.get("status") != "success" or not all(key in stats for key in QUANTILES):
            continue
        gt = float(record["ground_truth"])
        scores = {key: mean_relative_accuracy(float(stats[key]), gt) for key in QUANTILES}
        errors = {key: abs(float(stats[key]) - gt) for key in QUANTILES}
        for key in QUANTILES:
            fixed_scores[key].append(scores[key])
            fixed_errors[key].append(errors[key])

        best_mra = max(scores.values())
        best_mra_quantiles = [key for key in QUANTILES if scores[key] == best_mra]
        best_tiebreak = min(best_mra_quantiles, key=lambda key: errors[key])
        best_abs = min(QUANTILES, key=lambda key: errors[key])
        p05 = float(stats["p05"])
        p25 = float(stats["p25"])
        p75 = float(stats["p75"])
        p90 = float(stats["p90"])
        median = float(stats["median"])
        rows.append(
            {
                "doc_id": int(record["doc_id"]),
                "question": record.get("question", ""),
                "objects": record.get("objects", []),
                "ground_truth": gt,
                "gt_bucket": gt_bucket(gt),
                "best_mra": best_mra,
                "best_mra_quantiles": best_mra_quantiles,
                "best_tiebreak_quantile": best_tiebreak,
                "best_abs_quantile": best_abs,
                "best_abs_error": errors[best_abs],
                "p90_p05_ratio": p90 / max(p05, 1e-9),
                "p90_p05_spread": p90 - p05,
                "iqr_p75_p25": p75 - p25,
                "median": median,
                "scores": scores,
                "errors": errors,
                "predictions": {key: float(stats[key]) for key in QUANTILES},
            }
        )

    count = len(rows)
    fixed = {
        key: {
            "mra": sum(fixed_scores[key]) / count if count else 0.0,
            "mra_percent": (sum(fixed_scores[key]) / count * 100.0) if count else 0.0,
            "mae": sum(fixed_errors[key]) / count if count else None,
        }
        for key in QUANTILES
    }
    best_tiebreak_counts = Counter(row["best_tiebreak_quantile"] for row in rows)
    best_abs_counts = Counter(row["best_abs_quantile"] for row in rows)
    tie_counts = Counter()
    for row in rows:
        for key in row["best_mra_quantiles"]:
            tie_counts[key] += 1

    by_bucket: dict[str, dict[str, Any]] = {}
    for bucket in ["<1m", "1-3m", ">=3m"]:
        bucket_rows = [row for row in rows if row["gt_bucket"] == bucket]
        if not bucket_rows:
            continue
        by_bucket[bucket] = {
            "count": len(bucket_rows),
            "best_tiebreak_counts": dict(Counter(row["best_tiebreak_quantile"] for row in bucket_rows)),
            "fixed_mra_percent": {
                key: sum(row["scores"][key] for row in bucket_rows) / len(bucket_rows) * 100.0
                for key in QUANTILES
            },
        }

    summary = {
        "count": count,
        "quantiles": QUANTILES,
        "fixed": fixed,
        "oracle_mra_percent": (sum(row["best_mra"] for row in rows) / count * 100.0) if count else 0.0,
        "best_mra_tie_counts": dict(tie_counts),
        "best_tiebreak_counts": dict(best_tiebreak_counts),
        "best_abs_counts": dict(best_abs_counts),
        "by_ground_truth_bucket": by_bucket,
    }
    return rows, summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "doc_id",
        "ground_truth",
        "gt_bucket",
        "best_mra",
        "best_mra_quantiles",
        "best_tiebreak_quantile",
        "best_abs_quantile",
        "best_abs_error",
        "p90_p05_ratio",
        "p90_p05_spread",
        "iqr_p75_p25",
        "median",
        "question",
    ]
    for key in QUANTILES:
        fieldnames.extend([f"pred_{key}", f"mra_{key}", f"abs_err_{key}"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fieldnames if key in row}
            flat["best_mra_quantiles"] = "|".join(row["best_mra_quantiles"])
            for key in QUANTILES:
                flat[f"pred_{key}"] = row["predictions"][key]
                flat[f"mra_{key}"] = row["scores"][key]
                flat[f"abs_err_{key}"] = row["errors"][key]
            writer.writerow(flat)


def bar_svg(values: dict[str, float], *, width: int = 780, height: int = 260, suffix: str = "") -> str:
    left, right, top, bottom = 54, 18, 24, 42
    max_value = max(values.values()) if values else 1.0
    max_value = max(max_value, 1e-9)
    bar_gap = 12
    inner_w = width - left - right
    inner_h = height - top - bottom
    bar_w = (inner_w - bar_gap * (len(values) - 1)) / max(len(values), 1)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<line x1="{left}" y1="{top + inner_h}" x2="{width - right}" y2="{top + inner_h}" stroke="#333"/>',
    ]
    for index, (label, value) in enumerate(values.items()):
        x = left + index * (bar_w + bar_gap)
        bar_h = value / max_value * inner_h
        y = top + inner_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#4C78A8"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="12">{value:.1f}{suffix}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="12">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def heat_color(score: float) -> str:
    # White to green.
    score = max(0.0, min(1.0, score))
    red = int(245 - 120 * score)
    green = int(245 - 60 * (1 - score))
    blue = int(245 - 140 * score)
    return f"rgb({red},{green},{blue})"


def write_html(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    fixed_mra = {key: summary["fixed"][key]["mra_percent"] for key in QUANTILES}
    best_counts = {key: summary["best_tiebreak_counts"].get(key, 0) for key in QUANTILES}
    best_abs_counts = {key: summary["best_abs_counts"].get(key, 0) for key in QUANTILES}
    rows_html = []
    for row in rows:
        cells = []
        for key in QUANTILES:
            score = row["scores"][key]
            pred = row["predictions"][key]
            mark = "*" if key == row["best_tiebreak_quantile"] else ""
            cells.append(
                f'<td style="background:{heat_color(score)}" title="pred={pred:.3f}, err={row["errors"][key]:.3f}">'
                f"{score:.1f}{mark}</td>"
            )
        rows_html.append(
            "<tr>"
            f"<td>{row['doc_id']}</td>"
            f"<td>{row['ground_truth']:.2f}</td>"
            f"<td>{html.escape(row['gt_bucket'])}</td>"
            f"<td>{html.escape(row['best_tiebreak_quantile'])}</td>"
            f"<td>{html.escape(row['best_abs_quantile'])}</td>"
            f"<td>{row['p90_p05_ratio']:.2f}</td>"
            f"<td>{row['p90_p05_spread']:.2f}</td>"
            + "".join(cells)
            + f"<td>{html.escape(row['question'])}</td>"
            "</tr>"
        )

    bucket_rows = []
    for bucket, info in summary["by_ground_truth_bucket"].items():
        counts = ", ".join(f"{key}:{info['best_tiebreak_counts'].get(key, 0)}" for key in QUANTILES)
        best_fixed = max(info["fixed_mra_percent"], key=info["fixed_mra_percent"].get)
        bucket_rows.append(
            f"<tr><td>{bucket}</td><td>{info['count']}</td><td>{best_fixed}</td>"
            f"<td>{info['fixed_mra_percent'][best_fixed]:.1f}%</td><td>{counts}</td></tr>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>Object Absolute Distance Distribution Analysis</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; }}
h1, h2 {{ margin: 18px 0 10px; }}
table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
th, td {{ border: 1px solid #d0d0d0; padding: 5px 7px; vertical-align: top; }}
th {{ background: #f3f3f3; position: sticky; top: 0; }}
.note {{ max-width: 980px; line-height: 1.5; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; max-width: 900px; }}
</style>
<h1>绝对距离：点云距离分布取值分析</h1>
<p class="note">
样本数：{summary['count']}。表格中数值是 VSI-Bench MRA 单样本分数，星号表示该样本在 MRA 打平后按绝对误差最小选出的分位数。
Oracle MRA：{summary['oracle_mra_percent']:.2f}%。
</p>
<div class="grid">
<section><h2>固定分位数 MRA</h2>{bar_svg(fixed_mra, suffix="%")}</section>
<section><h2>每题最优分位数计数（MRA tie-break by abs error）</h2>{bar_svg(best_counts)}</section>
<section><h2>每题绝对误差最小分位数计数</h2>{bar_svg(best_abs_counts)}</section>
</div>
<h2>按真实距离分桶</h2>
<table>
<tr><th>GT bucket</th><th>count</th><th>best fixed</th><th>best fixed MRA</th><th>best per-answer counts</th></tr>
{''.join(bucket_rows)}
</table>
<h2>样本级热力表</h2>
<table>
<tr><th>doc</th><th>GT</th><th>bucket</th><th>best MRA</th><th>best abs</th><th>p90/p05</th><th>p90-p05</th>{''.join(f'<th>{key}</th>' for key in QUANTILES)}<th>question</th></tr>
{''.join(rows_html)}
</table>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Object Absolute Distance Distribution Analysis",
        "",
        "This report quantifies which point-cloud distance distribution statistic works best per answer.",
        "",
        f"- Samples: `{summary['count']}`",
        f"- Per-answer oracle MRA: `{summary['oracle_mra_percent']:.2f}%`",
        "",
        "## Fixed Aggregates",
        "",
        "| aggregate | MRA | MAE |",
        "| --- | ---: | ---: |",
    ]
    for key in QUANTILES:
        item = summary["fixed"][key]
        lines.append(f"| `{key}` | {item['mra_percent']:.2f}% | {item['mae']:.3f} |")
    lines.extend(
        [
            "",
            "## Best Aggregate Per Answer",
            "",
            "`best_tiebreak_counts` uses VSI-Bench MRA first and absolute error as the tie-breaker.",
            "",
            "```json",
            json.dumps(summary["best_tiebreak_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Interpretation",
            "",
            "- The best statistic is not fixed across samples.",
            "- `median` is the strongest single fixed statistic on the current 40-sample run.",
            "- Low quantiles help on some near/contact cases, but are harmful if used globally.",
            "- This supports an adaptive distribution-selection module as a separable research contribution.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze quantile choices for object_abs_distance point-cloud distances.")
    parser.add_argument(
        "--records-glob",
        default="runs/object_abs_distance_qwen25vl_full40_20260610_211122/*/records.json",
        help="Glob for records.json files containing distance_payload.distance_stats.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/tasks/object_abs_distance/distribution_analysis",
        help="Directory for CSV/JSON/HTML analysis artifacts.",
    )
    args = parser.parse_args()

    records = load_records(args.records_glob)
    rows, summary = analyze(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "per_doc_quantile_scores.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "per_doc_quantile_scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(rows, summary, output_dir / "distribution_analysis.html")
    write_markdown(summary, output_dir / "README.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
