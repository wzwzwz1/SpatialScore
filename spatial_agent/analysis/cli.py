from __future__ import annotations

import argparse

from spatial_agent.analysis.analyzer import aggregate_runs, load_lmms_samples, load_spatial_traces
from spatial_agent.analysis.report import write_analysis_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze VSI-Bench lmms_eval outputs and SpatialAgent traces.")
    parser.add_argument("--samples-path", required=True, help="lmms_eval samples file or output directory.")
    parser.add_argument("--trace-dir", required=True, help="SpatialAgent trace directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to write charts and reports.")
    parser.add_argument("--task-name", default="vsibench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-cases", type=int, default=24)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    samples = load_lmms_samples(args.samples_path, task_name=args.task_name, split=args.split)
    traces = load_spatial_traces(args.trace_dir)
    report = aggregate_runs(samples=samples, traces=traces)
    outputs = write_analysis_report(report, args.output_dir, max_cases=args.max_cases)

    print("Analysis complete.")
    for name, value in outputs.items():
        print(f"{name}: {value}")
    return 0
