from __future__ import annotations

import argparse
import json
from typing import List, Optional

from spatial_agent.factory import build_spatial_agent
from spatial_agent.runtime.config import SpatialAgentConfig, load_tool_config


def _parse_options(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LangGraph SpatialAgent ReAct loop.")
    parser.add_argument("--question", required=True, help="Question to answer.")
    parser.add_argument(
        "--image-path",
        action="append",
        dest="image_paths",
        default=[],
        help="Image path(s) associated with the question. Repeat for multiple images.",
    )
    parser.add_argument(
        "--question-type",
        default="open_ended",
        choices=["multi_choice", "judgment", "open_ended"],
        help="Question type.",
    )
    parser.add_argument(
        "--input-modality",
        default="single_image",
        choices=["single_image", "multi_image", "video"],
        help="Input modality for the task.",
    )
    parser.add_argument("--options", help="Comma-separated answer options for multi-choice tasks.")
    parser.add_argument(
        "--llm-backend",
        default="hf",
        choices=["hf", "openai_compatible"],
        help="LLM backend used by SpatialAgent.",
    )
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
    result = agent.invoke(
        {
            "task_id": "cli-task",
            "question": args.question,
            "question_type": args.question_type,
            "input_modality": args.input_modality,
            "image_paths": args.image_paths,
            "options": _parse_options(args.options),
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
