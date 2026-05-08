# SpatialAgent (LangGraph ReAct)

This package contains a new independent LangGraph-based SpatialAgent implementation that follows the paper's ReAct-style tool-augmented reasoning flow without reusing the old AutoGen control loop from `version_0/SpatialAgent`.

## What is implemented

- ReAct-only reasoning loop
- LangGraph state graph with repair, tool, observe, finalize, and fail paths
- LLM adapter abstraction
- Default local HuggingFace Qwen VL adapter
- OpenAI-compatible API adapter
- Tool registry with normalized `ToolResult`
- Structured trace writing per run
- VSI-Bench bridge utilities for same-host `lmms-eval` integration

## Current tool status

Implemented with optional real runtime support when dependencies are available:

- `EstimateOpticalFlow`
- `EstimateHomographyMatrix`

Wrapped but currently dependency/checkpoint-gated:

- `EstimateObjectDepth`
- `GetObjectMask`
- `GetCameraParametersVGGT`
- `GetObjectOrientation`

Declared but intentionally unavailable:

- `LocalizeObjects`
- `EstimateObjectMotion`

## Run from CLI

Local HuggingFace backend:

```bash
python3 -m spatial_agent \
  --question "Which option is correct?" \
  --question-type multi_choice \
  --input-modality single_image \
  --options A,B,C,D \
  --image-path /path/to/image.jpg \
  --llm-backend hf \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct
```

OpenAI-compatible backend:

```bash
python3 -m spatial_agent \
  --question "Which option is correct?" \
  --question-type multi_choice \
  --input-modality single_image \
  --options A,B,C,D \
  --image-path /path/to/image.jpg \
  --llm-backend openai_compatible \
  --model-name gpt-4o-mini \
  --api-base-url https://api.openai.com/v1
```

The adapters and some tools use lazy imports. If `torch`, `transformers`, OpenCV, checkpoints, or model paths are unavailable, the graph remains runnable, but affected tools will return structured `unavailable` observations.

## VSI-Bench same-host integration

This repository now includes a lightweight `lmms-eval` plugin package at:

```text
lmms_eval_spatialagent_plugin/
```

To use it with the `thinking-in-space` benchmark on the same server without modifying benchmark internals:

```bash
export PYTHONPATH=/Users/wz/code/SpatialScore:$PYTHONPATH
export LMMS_EVAL_PLUGINS=lmms_eval_spatialagent_plugin

cd /Users/wz/code/thinking-in-space
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,api_base_url=https://api.openai.com/v1,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs \
  --tasks vsibench \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/vsibench
```

For a local model backend, swap `llm_backend` and pass `model_path=/path/to/Qwen2.5-VL-7B-Instruct`.

For a full step-by-step guide, including smoke tests, local/API commands, output locations, and recommended `model_args`, see [`/Users/wz/code/SpatialScore/docs/spatial_agent_vsibench.md`](/Users/wz/code/SpatialScore/docs/spatial_agent_vsibench.md).

## Trace artifacts

Each run writes a JSON trace to:

```text
.artifacts/spatial_agent/<task_id>.json
```

The trace includes:

- tool calls
- tool observations
- final answer
- status / error
- reasoning trace
