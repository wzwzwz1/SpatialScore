from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _text_height() -> int:
    return 14


def _save_barh_chart(labels: List[str], values: List[float], title: str, output_path: Path, color: str, max_value: float | None = None) -> str:
    width = 1200
    row_height = 44
    margin_left = 280
    margin_right = 40
    margin_top = 70
    margin_bottom = 30
    height = max(220, margin_top + margin_bottom + max(1, len(labels)) * row_height)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((24, 20), title, fill="black", font=font)

    chart_width = width - margin_left - margin_right
    max_val = max_value if max_value is not None else max(values or [1.0])
    max_val = max(max_val, 1e-6)

    for index, label in enumerate(labels):
        y = margin_top + index * row_height
        value = values[index]
        bar_length = int((value / max_val) * chart_width)
        draw.text((20, y + 8), label, fill="black", font=font)
        draw.rectangle([margin_left, y + 6, margin_left + bar_length, y + 28], fill=color, outline="#999999")
        draw.text((margin_left + bar_length + 8, y + 8), f"{value:.3f}" if isinstance(value, float) else str(value), fill="black", font=font)

    image.save(output_path)
    return str(output_path.name)


def _save_stacked_barh_chart(labels: List[str], series: List[List[float]], series_names: List[str], colors: List[str], title: str, output_path: Path) -> str:
    width = 1200
    row_height = 44
    margin_left = 280
    margin_right = 40
    margin_top = 90
    margin_bottom = 30
    height = max(240, margin_top + margin_bottom + max(1, len(labels)) * row_height)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((24, 20), title, fill="black", font=font)

    legend_x = 24
    for index, (name, color) in enumerate(zip(series_names, colors)):
        y = 46 + index * 18
        draw.rectangle([legend_x, y, legend_x + 12, y + 12], fill=color, outline="#888888")
        draw.text((legend_x + 18, y - 1), name, fill="black", font=font)

    chart_width = width - margin_left - margin_right
    totals = [sum(values[index] for values in series) for index in range(len(labels))]
    max_total = max(totals or [1.0])
    max_total = max(max_total, 1e-6)

    for index, label in enumerate(labels):
        y = margin_top + index * row_height
        draw.text((20, y + 8), label, fill="black", font=font)
        cursor = margin_left
        for values, color in zip(series, colors):
            value = values[index]
            length = int((value / max_total) * chart_width)
            if length > 0:
                draw.rectangle([cursor, y + 6, cursor + length, y + 28], fill=color, outline="#999999")
            cursor += length
        draw.text((cursor + 8, y + 8), str(totals[index]), fill="black", font=font)

    image.save(output_path)
    return str(output_path.name)


def _save_histogram(values: List[int], title: str, output_path: Path) -> str:
    width = 1000
    height = 420
    margin_left = 60
    margin_bottom = 50
    margin_top = 40
    margin_right = 30
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((20, 12), title, fill="black", font=font)

    buckets: Dict[int, int] = {}
    for value in values:
        buckets[value] = buckets.get(value, 0) + 1
    keys = sorted(buckets.keys()) or [0]
    max_count = max(buckets.values()) if buckets else 1

    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    bar_width = max(30, chart_width // max(1, len(keys)))

    for i, key in enumerate(keys):
        count = buckets[key]
        x0 = margin_left + i * bar_width
        bar_height = int((count / max_count) * (chart_height - 20))
        y0 = margin_top + chart_height - bar_height
        draw.rectangle([x0, y0, x0 + bar_width - 8, margin_top + chart_height], fill="#B07AA1", outline="#666666")
        draw.text((x0 + 6, margin_top + chart_height + 6), str(key), fill="black", font=font)
        draw.text((x0 + 6, y0 - 16), str(count), fill="black", font=font)

    image.save(output_path)
    return str(output_path.name)


def _save_heatmap(row_labels: List[str], col_labels: List[str], matrix: List[List[int]], title: str, output_path: Path) -> str:
    cell_w = 70
    cell_h = 34
    margin_left = 260
    margin_top = 120
    width = margin_left + max(1, len(col_labels)) * cell_w + 40
    height = margin_top + max(1, len(row_labels)) * cell_h + 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((24, 20), title, fill="black", font=font)

    max_val = max((value for row in matrix for value in row), default=1)
    max_val = max(max_val, 1)
    for col, label in enumerate(col_labels):
        draw.text((margin_left + col * cell_w, 70), label[:10], fill="black", font=font)

    for row, label in enumerate(row_labels):
        y = margin_top + row * cell_h
        draw.text((20, y + 8), label[:28], fill="black", font=font)
        for col, value in enumerate(matrix[row]):
            x = margin_left + col * cell_w
            intensity = int(255 - (value / max_val) * 170)
            color = (intensity, intensity, 255)
            draw.rectangle([x, y, x + cell_w - 4, y + cell_h - 4], fill=color, outline="#999999")
            draw.text((x + 10, y + 8), str(value), fill="black", font=font)

    image.save(output_path)
    return str(output_path.name)


def _materialize_artifacts(report: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, str] = {}

    for row in report.get("artifacts", []):
        source = Path(row["path"])
        if not source.exists():
            continue
        filename = _safe_filename(source.name)
        destination = artifact_dir / filename
        if destination.exists() and destination.resolve() != source.resolve():
            filename = _safe_filename(f"{row['sample_task_id']}__{row['tool_name']}__{source.name}")
            destination = artifact_dir / filename
        if not destination.exists():
            try:
                destination.symlink_to(source)
            except OSError:
                shutil.copy2(source, destination)
        mapping[str(source)] = str(destination.relative_to(output_dir))
    return mapping


def _write_csv(samples: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "task_id",
        "doc_id",
        "question_type",
        "dataset",
        "scene_name",
        "ground_truth",
        "prediction",
        "trace_found",
        "status",
        "error",
        "tool_call_count",
        "reasoning_steps",
        "artifact_count",
        "tool_names",
        "tool_execution_details",
        "score_fields",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = dict(sample)
            row["tool_names"] = "|".join(sample.get("tool_names", []))
            row["tool_execution_details"] = json.dumps(sample.get("tool_execution_details", []), ensure_ascii=False)
            row["score_fields"] = json.dumps(sample.get("score_fields", {}), ensure_ascii=False)
            writer.writerow({key: row.get(key) for key in fieldnames})


def _plot_question_type_counts(report: Dict[str, Any], chart_dir: Path) -> str:
    items = sorted(report["question_types"].items())
    labels = [key for key, _ in items]
    counts = [value["count"] for _, value in items]
    path = chart_dir / "question_type_counts.png"
    return _save_barh_chart(labels, counts, "Sample Count by Question Type", path, "#4C78A8")


def _plot_question_type_scores(report: Dict[str, Any], chart_dir: Path) -> str:
    labels = []
    scores = []
    for question_type, payload in sorted(report["question_types"].items()):
        metric_values = list(payload.get("metrics", {}).values())
        if not metric_values:
            continue
        labels.append(question_type)
        scores.append(metric_values[0])
    path = chart_dir / "question_type_scores.png"
    return _save_barh_chart(labels, scores, "Primary Score by Question Type", path, "#59A14F", max_value=1.0)


def _plot_tool_calls(report: Dict[str, Any], chart_dir: Path) -> str:
    items = sorted(report["tools"].items(), key=lambda item: item[1]["calls"], reverse=True)
    labels = [key for key, _ in items]
    counts = [value["calls"] for _, value in items]
    path = chart_dir / "tool_call_counts.png"
    return _save_barh_chart(labels, counts, "Tool Call Frequency", path, "#F28E2B")


def _plot_tool_status(report: Dict[str, Any], chart_dir: Path) -> str:
    items = sorted(report["tools"].items(), key=lambda item: item[1]["calls"], reverse=True)
    labels = [key for key, _ in items]
    success = [value["success"] for _, value in items]
    error = [value["error"] for _, value in items]
    unavailable = [value["unavailable"] for _, value in items]
    path = chart_dir / "tool_status_stacked.png"
    return _save_stacked_barh_chart(
        labels,
        [success, unavailable, error],
        ["success", "unavailable", "error"],
        ["#59A14F", "#EDC948", "#E15759"],
        "Tool Observation Status",
        path,
    )


def _plot_reasoning_steps(report: Dict[str, Any], chart_dir: Path) -> str:
    steps = [sample["reasoning_steps"] for sample in report["samples"] if sample.get("trace_found")]
    path = chart_dir / "reasoning_steps_hist.png"
    return _save_histogram(steps, "Reasoning Step Distribution", path)


def _plot_question_tool_heatmap(report: Dict[str, Any], chart_dir: Path) -> str:
    question_types = sorted(report["question_types"].keys())
    tool_names = sorted(report["tools"].keys())
    matrix = [[0 for _ in tool_names] for _ in question_types]

    for sample in report["samples"]:
        for tool_name in sample.get("tool_names", []):
            if sample["question_type"] in question_types and tool_name in tool_names:
                row = question_types.index(sample["question_type"])
                col = tool_names.index(tool_name)
                matrix[row][col] += 1

    path = chart_dir / "question_type_tool_heatmap.png"
    return _save_heatmap(question_types, tool_names, matrix, "Question Type x Tool Usage", path)


def _build_case_rows(report: Dict[str, Any], artifact_mapping: Dict[str, str], max_cases: int) -> List[Dict[str, Any]]:
    def rank(sample: Dict[str, Any]) -> tuple[int, int, int]:
        primary_score = next(iter(sample.get("score_fields", {}).values()), 0.0)
        return (int(sample.get("trace_found", False)), int(primary_score == 1.0), -sample.get("artifact_count", 0))

    samples = sorted(report["samples"], key=rank)
    case_rows: List[Dict[str, Any]] = []
    for sample in samples[:max_cases]:
        case_rows.append(
            {
                **sample,
                "artifact_report_paths": [artifact_mapping.get(path, path) for path in sample.get("artifact_paths", [])],
            }
        )
    return case_rows


def _status_counts_line(summary: Dict[str, Any]) -> str:
    counts = summary.get("status_counts", {})
    if not counts:
        return "(none)"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _format_metric_map(metric_map: Dict[str, float]) -> str:
    if not metric_map:
        return "(none)"
    return ", ".join(f"{key}={value:.3f}" for key, value in metric_map.items())


def _format_json_block(payload: Any) -> str:
    if payload in (None, {}, []):
        return "(none)"
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _derive_findings(report: Dict[str, Any]) -> List[str]:
    findings: List[str] = []
    summary = report["summary"]
    if summary.get("trace_coverage", 0.0) < 0.8:
        findings.append("trace 覆盖率偏低，说明有一部分样本没有成功落盘到 SpatialAgent trace，建议先排查 artifact_dir 和 task_id 映射。")
    else:
        findings.append("trace 覆盖率较高，说明 benchmark 样本与 SpatialAgent 执行轨迹已经可以稳定对齐。")

    if summary.get("average_tool_calls", 0.0) < 0.5:
        findings.append("平均 tool 调用次数偏低，agent 可能主要在裸答，没有充分利用空间工具。")
    else:
        findings.append("平均每题至少发生了一次 tool 调用，说明 agent 已经在真实依赖工具信息而不是纯语言猜测。")

    unavailable_tools = [name for name, payload in report["tools"].items() if payload.get("unavailable", 0) > 0]
    if unavailable_tools:
        findings.append(f"以下工具存在 unavailable 情况：{', '.join(unavailable_tools)}。优先检查 tool_config、checkpoint 路径和依赖安装。")

    counting_samples = [sample for sample in report["samples"] if sample.get("question_type") == "object_counting"]
    natural_language_counting = [
        sample
        for sample in counting_samples
        if sample.get("prediction") and not str(sample["prediction"]).strip().isdigit()
    ]
    if natural_language_counting:
        findings.append("计数题存在自然语言长句输出，建议把数值题最终答案收紧为纯数字，避免评测时因为格式问题丢分。")

    if not findings:
        findings.append("当前样本量较小，建议把 `--limit` 扩到 20 或更多，再观察题型分数和 tool 使用分布。")

    return findings


def _write_markdown(
    report: Dict[str, Any],
    chart_files: Dict[str, str],
    case_rows: List[Dict[str, Any]],
    output_dir: Path,
    summary_path: Path,
    csv_path: Path,
    html_path: Path,
) -> Path:
    summary = report["summary"]
    findings = _derive_findings(report)
    lines = [
        "# VSI-Bench SpatialAgent 中文分析报告",
        "",
        "## 文件索引",
        "",
        f"- summary_json: {summary_path.name}",
        f"- samples_csv: {csv_path.name}",
        f"- report_markdown: report.md",
        f"- report_html: {html_path.name}",
        f"- charts_dir: charts/",
        f"- artifacts_dir: artifacts/",
        "",
        "## 总览结论",
        "",
        f"- 样本数: {summary['sample_count']}",
        f"- trace 覆盖率: {summary['trace_coverage']:.3f}",
        f"- agent 执行成功率: {summary['success_rate']:.3f}",
        f"- 平均推理步数: {summary['average_reasoning_steps']:.3f}",
        f"- 平均 tool 调用次数: {summary['average_tool_calls']:.3f}",
        f"- status 分布: {_status_counts_line(summary)}",
        "",
        "## 自动观察",
        "",
    ]
    for finding in findings:
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## 题型统计",
            "",
            "| 题型 | 样本数 | trace 命中数 | 指标均值 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for question_type, payload in sorted(report["question_types"].items()):
        lines.append(
            f"| {question_type} | {payload['count']} | {payload.get('trace_found', 0)} | {_format_metric_map(payload.get('metrics', {}))} |"
        )

    lines.extend(
        [
            "",
            "## Tool 统计",
            "",
            "| Tool | 调用次数 | success | unavailable | error |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for tool_name, payload in sorted(report["tools"].items(), key=lambda item: item[1]["calls"], reverse=True):
        lines.append(
            f"| {tool_name} | {payload.get('calls', 0)} | {payload.get('success', 0)} | {payload.get('unavailable', 0)} | {payload.get('error', 0)} |"
        )

    lines.extend(
        [
            "",
            "## 图表总览",
            "",
        ]
    )
    for title, filename in chart_files.items():
        lines.extend([f"### {title}", "", f"![{title}](charts/{filename})", ""])

    lines.extend(
        [
            "## 样本级明细",
            "",
        ]
    )
    for sample in case_rows:
        lines.extend(
            [
                f"### {sample['task_id']}",
                "",
                f"- 题型: {sample['question_type']}",
                f"- 执行状态: {sample['status']}",
                f"- ground_truth: {sample['ground_truth']}",
                f"- prediction: {sample['prediction']}",
                f"- tool 列表: {', '.join(sample['tool_names']) if sample['tool_names'] else '(none)'}",
                f"- tool 调用次数: {sample['tool_call_count']}",
                f"- 推理步数: {sample['reasoning_steps']}",
                f"- 指标: {_format_metric_map(sample.get('score_fields', {}))}",
                f"- error: {sample['error'] or '(none)'}",
                "",
                "问题：",
                "",
                sample["question"],
                "",
            ]
        )
        if sample["artifact_report_paths"]:
            lines.extend(["中间视觉产物：", ""])
            for artifact_path in sample["artifact_report_paths"]:
                lines.extend([f"![artifact]({artifact_path})", ""])
        else:
            lines.extend(["中间视觉产物：", "", "(none)", ""])

        lines.extend(["### Tool 执行明细", ""])
        if sample.get("tool_execution_details"):
            for execution in sample["tool_execution_details"]:
                lines.extend(
                    [
                        f"#### 第 {execution['step']} 步：{execution['tool_name']}",
                        "",
                        f"- status: {execution['status']}",
                        f"- error: {execution['error'] or '(none)'}",
                        "",
                        "arguments:",
                        "",
                        "```json",
                        _format_json_block(execution.get("arguments", {})),
                        "```",
                        "",
                        "payload:",
                        "",
                        "```json",
                        _format_json_block(execution.get("payload", {})),
                        "```",
                        "",
                    ]
                )
                artifacts = execution.get("artifacts", [])
                if artifacts:
                    lines.extend(["artifacts:", ""])
                    for artifact_path in artifacts:
                        lines.extend([f"- {artifact_path}", ""])
                else:
                    lines.extend(["artifacts:", "", "(none)", ""])
        else:
            lines.extend(["(none)", ""])

    lines.extend(
        [
            "## 后续建议",
            "",
            "1. 先看 `Tool 统计` 和 `图表总览`，判断当前掉分是来自工具不可用、工具误用，还是最终答案格式问题。",
            "2. 再看 `样本级明细` 里的 artifact 图片，确认分割、深度、光流、轨迹这些中间结果是否真的贴合场景。",
            "3. 如果当前样本量很小，优先把 `--limit` 扩大到 20 或更多，再决定是改 tool_config、Prompt 还是工具实现。",
        ]
    )

    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_html(report: Dict[str, Any], chart_files: Dict[str, str], case_rows: List[Dict[str, Any]], output_dir: Path) -> Path:
    summary = report["summary"]
    findings = _derive_findings(report)
    chart_html = "\n".join(
        f"<section><h2>{title}</h2><img src=\"charts/{filename}\" alt=\"{title}\" style=\"max-width: 100%; border: 1px solid #ddd; border-radius: 8px;\" /></section>"
        for title, filename in chart_files.items()
    )

    question_type_rows = "".join(
        f"<tr><td>{question_type}</td><td>{payload['count']}</td><td>{payload.get('trace_found', 0)}</td><td>{_format_metric_map(payload.get('metrics', {}))}</td></tr>"
        for question_type, payload in sorted(report["question_types"].items())
    )
    tool_rows = "".join(
        f"<tr><td>{tool_name}</td><td>{payload.get('calls', 0)}</td><td>{payload.get('success', 0)}</td><td>{payload.get('unavailable', 0)}</td><td>{payload.get('error', 0)}</td></tr>"
        for tool_name, payload in sorted(report["tools"].items(), key=lambda item: item[1]["calls"], reverse=True)
    )

    case_html_parts = []
    for sample in case_rows:
        artifact_html = "".join(
            f"<img src=\"{artifact_path}\" alt=\"artifact\" style=\"max-width: 360px; margin: 8px 8px 8px 0; border: 1px solid #ddd; border-radius: 6px;\" />"
            for artifact_path in sample["artifact_report_paths"]
        )
        tool_detail_html = []
        for execution in sample.get("tool_execution_details", []):
            artifact_list = execution.get("artifacts", [])
            artifact_detail = "".join(f"<li>{artifact}</li>" for artifact in artifact_list) if artifact_list else "<li>(none)</li>"
            tool_detail_html.append(
                f"""
                <section style="margin-top: 12px; padding: 12px; background: #fafafa; border-radius: 8px;">
                  <h4>第 {execution['step']} 步：{execution['tool_name']}</h4>
                  <p><strong>status:</strong> {execution['status']} | <strong>error:</strong> {execution['error'] or '(none)'}</p>
                  <p><strong>arguments</strong></p>
                  <pre style="white-space: pre-wrap; background: white; border: 1px solid #ddd; padding: 10px;">{_format_json_block(execution.get('arguments', {}))}</pre>
                  <p><strong>payload</strong></p>
                  <pre style="white-space: pre-wrap; background: white; border: 1px solid #ddd; padding: 10px;">{_format_json_block(execution.get('payload', {}))}</pre>
                  <p><strong>artifacts</strong></p>
                  <ul>{artifact_detail}</ul>
                </section>
                """
            )
        case_html_parts.append(
            f"""
            <article style="padding: 16px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 16px;">
              <h3>{sample['task_id']}</h3>
              <p><strong>题型:</strong> {sample['question_type']} | <strong>执行状态:</strong> {sample['status']}</p>
              <p><strong>ground_truth:</strong> {sample['ground_truth']} | <strong>prediction:</strong> {sample['prediction']}</p>
              <p><strong>tools:</strong> {", ".join(sample['tool_names']) if sample['tool_names'] else "(none)"}</p>
              <p><strong>推理步数:</strong> {sample['reasoning_steps']} | <strong>tool 调用次数:</strong> {sample['tool_call_count']}</p>
              <p><strong>指标:</strong> {_format_metric_map(sample.get('score_fields', {}))}</p>
              <p>{sample['question']}</p>
              <div>{artifact_html if artifact_html else "(none)"}</div>
              <h4>Tool 执行明细</h4>
              {''.join(tool_detail_html) if tool_detail_html else "<p>(none)</p>"}
            </article>
            """
        )

    finding_html = "".join(f"<li>{finding}</li>" for finding in findings)

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>VSI-Bench SpatialAgent 中文分析报告</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
          .summary-grid {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 12px; }}
          .summary-card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; background: #fafafa; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }}
          th {{ background: #f5f5f5; }}
        </style>
      </head>
      <body>
        <h1>VSI-Bench SpatialAgent 中文分析报告</h1>
        <div class="summary-grid">
          <div class="summary-card"><strong>样本数</strong><br />{summary['sample_count']}</div>
          <div class="summary-card"><strong>trace 覆盖率</strong><br />{summary['trace_coverage']:.3f}</div>
          <div class="summary-card"><strong>执行成功率</strong><br />{summary['success_rate']:.3f}</div>
          <div class="summary-card"><strong>平均推理步数</strong><br />{summary['average_reasoning_steps']:.3f}</div>
          <div class="summary-card"><strong>平均 tool 调用</strong><br />{summary['average_tool_calls']:.3f}</div>
        </div>
        <section>
          <h2>自动观察</h2>
          <ul>{finding_html}</ul>
        </section>
        <section>
          <h2>题型统计</h2>
          <table>
            <thead><tr><th>题型</th><th>样本数</th><th>trace 命中数</th><th>指标均值</th></tr></thead>
            <tbody>{question_type_rows}</tbody>
          </table>
        </section>
        <section>
          <h2>Tool 统计</h2>
          <table>
            <thead><tr><th>Tool</th><th>调用次数</th><th>success</th><th>unavailable</th><th>error</th></tr></thead>
            <tbody>{tool_rows}</tbody>
          </table>
        </section>
        {chart_html}
        <section>
          <h2>样本级明细</h2>
          {''.join(case_html_parts)}
        </section>
      </body>
    </html>
    """
    path = output_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def write_analysis_report(report: Dict[str, Any], output_dir: str | Path, max_cases: int = 24) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    artifact_mapping = _materialize_artifacts(report, output_dir)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "samples.csv"
    _write_csv(report["samples"], csv_path)

    chart_files = {
        "Question Type Count": _plot_question_type_counts(report, chart_dir),
        "Question Type Score": _plot_question_type_scores(report, chart_dir),
        "Tool Call Count": _plot_tool_calls(report, chart_dir),
        "Tool Status": _plot_tool_status(report, chart_dir),
        "Reasoning Steps": _plot_reasoning_steps(report, chart_dir),
        "Question Type x Tool": _plot_question_tool_heatmap(report, chart_dir),
    }

    case_rows = _build_case_rows(report, artifact_mapping=artifact_mapping, max_cases=max_cases)
    html_path = output_dir / "report.html"
    markdown_path = _write_markdown(report, chart_files, case_rows, output_dir, summary_path, csv_path, html_path)
    html_path = _write_html(report, chart_files, case_rows, output_dir)

    return {
        "summary_json": str(summary_path),
        "samples_csv": str(csv_path),
        "report_markdown": str(markdown_path),
        "report_html": str(html_path),
    }
