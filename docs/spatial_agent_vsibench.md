# SpatialAgent + VSI-Bench

This document explains how to evaluate the new `spatial_agent` package on [`VSI-Bench`](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) when:

- the benchmark code lives at `/Users/wz/code/thinking-in-space`
- the agent code lives at `/Users/wz/code/SpatialScore`
- both run on the same server

The integration is designed to avoid modifying `thinking-in-space` internals. Instead, `lmms-eval` discovers a plugin model from this repository through `LMMS_EVAL_PLUGINS`.

## 1. What was added

The integration pieces live in this repository:

- `spatial_agent/`
  - ReAct graph
  - local HuggingFace backend
  - OpenAI-compatible API backend
  - video-to-frame bridge
- `lmms_eval_spatialagent_plugin/`
  - `lmms-eval` plugin model named `spatial_agent_api`

`thinking-in-space` keeps using its own `vsibench` task and metrics.

## 2. Environment setup

Use one Python environment that can import both repositories.

At minimum, make sure:

1. `thinking-in-space` is installed or runnable in the active environment.
2. `SpatialScore` dependencies are also available in the same environment.
3. The benchmark dataset and model resources are already available on the server.

If you prefer not to install `SpatialScore` as a package, the documented path uses `PYTHONPATH`.

## 3. Required environment variables

From the shell where you will run `VSI-Bench`:

```bash
export PYTHONPATH=/Users/wz/code/SpatialScore:$PYTHONPATH
export LMMS_EVAL_PLUGINS=lmms_eval_spatialagent_plugin
```

For API-backed inference, also export your key if you do not want to pass it in `model_args`:

```bash
export OPENAI_API_KEY=your_api_key_here
```

If you use a different variable name, pass it through `api_key_env=...` in `model_args`.

## 4. Smoke test first

Before a full run, do a tiny dry run with `--limit 2`.

Change into the benchmark repo:

```bash
cd /Users/wz/code/thinking-in-space
```

### 4.1 API backend

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,api_base_url=https://api.openai.com/v1,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs \
  --tasks vsibench \
  --batch_size 1 \
  --limit 2 \
  --log_samples \
  --output_path ./logs/vsibench_smoke
```

### 4.2 Local model backend

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=hf,model_path=/path/to/Qwen2.5-VL-7B-Instruct,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs \
  --tasks vsibench \
  --batch_size 1 \
  --limit 2 \
  --log_samples \
  --output_path ./logs/vsibench_smoke
```

If the smoke test works, remove `--limit 2` and run the full benchmark.

## 5. Full VSI-Bench run

### 5.1 API backend

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,api_base_url=https://api.openai.com/v1,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs \
  --tasks vsibench \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/vsibench
```

### 5.2 Local backend

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=hf,model_path=/path/to/Qwen2.5-VL-7B-Instruct,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs \
  --tasks vsibench \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/vsibench
```

## 6. Supported `model_args`

The plugin model is:

```text
--model spatial_agent_api
```

Supported `model_args`:

- `llm_backend=hf|openai_compatible`
- `model_path=/path/to/local/model`
- `model_name=<remote model name>`
- `api_base_url=<OpenAI-compatible base URL>`
- `api_key=<optional inline key>`
- `api_key_env=<env var name for the key>`
- `api_timeout=<seconds>`
- `max_steps=<reasoning steps>`
- `video_num_frames=<uniformly sampled frames per video>`
- `video_frame_dir=<optional explicit frame cache directory>`
- `artifact_dir=<trace and temp artifact root>`
- `keep_video_frames=true|false`

Example with preserved sampled frames:

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,api_base_url=https://api.openai.com/v1,video_num_frames=24,video_frame_dir=/tmp/vsi_frames,artifact_dir=/tmp/spatial_agent_runs,keep_video_frames=true \
  --tasks vsibench \
  --batch_size 1 \
  --limit 2 \
  --log_samples \
  --output_path ./logs/vsibench_debug
```

## 7. What the plugin does internally

For each VSI-Bench sample:

1. `thinking-in-space` resolves the video path from the dataset cache.
2. `spatial_agent_api` samples the video into frames.
3. The frames are mapped into a `SpatialAgent` task input.
4. `SpatialAgent` runs the ReAct loop and returns `final_answer`.
5. `lmms-eval` scores that answer using the original `vsibench` task metric code.

## 8. Outputs to inspect

### Benchmark outputs

`thinking-in-space` writes benchmark outputs under the `--output_path` you provide, for example:

```text
/Users/wz/code/thinking-in-space/logs/vsibench/
```

That directory will contain:

- aggregated `results.json`
- per-task sample logs when `--log_samples` is enabled

### SpatialAgent traces

`SpatialAgent` writes per-sample traces under:

```text
<artifact_dir>/<task_id>.json
```

For example:

```text
/tmp/spatial_agent_runs/vsibench___test___42.json
```

These traces include:

- final answer
- status and error
- tool calls
- tool observations
- reasoning trace

## 9. Notes

- `VSI-Bench` is video-based, so the bridge samples frames before calling the agent.
- `batch_size 1` is the safe default because each sample may trigger multi-step reasoning and tool use.
- If `keep_video_frames=false`, sampled frames are deleted after each sample finishes.
- If you want to inspect inputs, set `keep_video_frames=true`.
