from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_agent.factory import build_spatial_agent
from spatial_agent.io.vsibench_runner import resolve_vsibench_cache_dir, run_vsibench_sample, write_vsibench_run_outputs
from spatial_agent.analysis.analyzer import aggregate_runs, load_spatial_traces
from spatial_agent.analysis.report import write_analysis_report
from spatial_agent.runtime.config import SpatialAgentConfig, load_tool_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one VSI-Bench sample directly from the SpatialAgent project.")
    parser.add_argument("--doc-id", type=int, required=True, help="VSI-Bench sample index in the chosen split.")
    parser.add_argument("--split", default="test", help="Dataset split to use.")
    parser.add_argument("--dataset-name", default="nyu-visionx/VSI-Bench", help="Hugging Face dataset name.")
    parser.add_argument("--dataset-cache-dir", help="Resolved VSI-Bench cache directory. Defaults to $HF_HOME/vsibench.")
    parser.add_argument("--hf-token", default=True, help="Hugging Face token forwarding mode or explicit token.")
    parser.add_argument("--llm-backend", default="hf", choices=["hf", "openai_compatible"], help="LLM backend used by SpatialAgent.")
    parser.add_argument("--model-path", help="Local HuggingFace Qwen VL checkpoint path.")
    parser.add_argument("--model-name", help="Remote API model name for OpenAI-compatible backends.")
    parser.add_argument("--api-base-url", help="Base URL for OpenAI-compatible APIs.")
    parser.add_argument("--api-base-url-env", default="OPENAI_API_BASE_URL", help="Environment variable name for API base URL lookup.")
    parser.add_argument("--api-key", help="API key for OpenAI-compatible APIs.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable name for API key lookup.")
    parser.add_argument("--api-timeout", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--video-num-frames", type=int, default=16)
    parser.add_argument("--video-frame-dir", help="Optional directory for sampled video frames.")
    parser.add_argument("--keep-video-frames", action="store_true", help="Keep sampled video frames on disk.")
    parser.add_argument("--artifact-dir", default=".artifacts/spatial_agent")
    parser.add_argument("--tool-config-path", help="JSON file path for tool_config.")
    parser.add_argument(
        "--output-dir",
        help="Directory to store run.json and vsibench.json. Defaults to <artifact_dir>/single_runs/<split>/<doc_id>.",
    )
    parser.add_argument("--run-analysis", action="store_true", help="Run the existing analysis reporter after saving files.")
    args = parser.parse_args()

    config = SpatialAgentConfig(
        llm_backend=args.llm_backend,
        qwen_model_path=args.model_path,
        api_model_name=args.model_name,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        api_base_url_env=args.api_base_url_env,
        api_key_env=args.api_key_env,
        api_timeout=args.api_timeout,
        max_steps=args.max_steps,
        video_num_frames=args.video_num_frames,
        video_frame_dir=args.video_frame_dir,
        keep_video_frames=args.keep_video_frames,
        artifact_dir=args.artifact_dir,
        tool_config=load_tool_config(args.tool_config_path),
    )
    agent = build_spatial_agent(config)
    payload = run_vsibench_sample(
        agent=agent,
        dataset_split=args.split,
        doc_id=args.doc_id,
        num_frames=config.video_num_frames,
        artifact_dir=config.artifact_dir,
        keep_video_frames=config.keep_video_frames,
        dataset_cache_dir=resolve_vsibench_cache_dir(args.dataset_cache_dir),
        dataset_name=args.dataset_name,
        video_frame_dir=config.video_frame_dir,
        token=args.hf_token,
    )
    output_dir = args.output_dir or str(Path(config.artifact_dir) / "single_runs" / args.split / str(args.doc_id))
    output_paths = write_vsibench_run_outputs(
        output_dir=output_dir,
        doc_id=args.doc_id,
        payload=payload,
    )

    print("Run complete.")
    print(f"run_json: {output_paths['run_json']}")
    print(f"samples_json: {output_paths['samples_json']}")
    print(f"trace_path: {payload['result'].get('trace_path')}")

    if args.run_analysis:
        report = aggregate_runs(
            samples=[{
                "task_id": payload["task_input"]["task_id"],
                "doc_id": args.doc_id,
                "question": payload["doc"].get("question", ""),
                "question_type": payload["doc"].get("question_type", "unknown"),
                "ground_truth": payload["doc"].get("ground_truth"),
                "target": payload["doc"].get("ground_truth"),
                "prediction": payload["result"].get("final_answer") or "",
                "scene_name": payload["doc"].get("scene_name"),
                "dataset": payload["doc"].get("dataset"),
                "score_fields": {},
                "doc": dict(payload["doc"]),
                "raw_sample": {},
            }],
            traces=load_spatial_traces(config.artifact_dir),
        )
        analysis_dir = str(Path(output_dir) / "analysis")
        analysis_paths = write_analysis_report(report, analysis_dir)
        print(f"analysis_dir: {analysis_dir}")
        for name, value in analysis_paths.items():
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
