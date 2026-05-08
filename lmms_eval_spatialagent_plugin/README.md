# lmms-eval SpatialAgent Plugin

This package lets the `thinking-in-space` VSI-Bench evaluator call `SpatialAgent` directly on the same host through `LMMS_EVAL_PLUGINS`, without modifying benchmark internals.

## Usage

```bash
export PYTHONPATH=/Users/wz/code/SpatialScore:$PYTHONPATH
export LMMS_EVAL_PLUGINS=lmms_eval_spatialagent_plugin
```

Then run from `/Users/wz/code/thinking-in-space`:

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,api_base_url=https://api.openai.com/v1,video_num_frames=16 \
  --tasks vsibench \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/vsibench
```

For local model inference, use:

```text
llm_backend=hf,model_path=/path/to/Qwen2.5-VL-7B-Instruct
```

The complete operational guide is here:

- [`/Users/wz/code/SpatialScore/docs/spatial_agent_vsibench.md`](/Users/wz/code/SpatialScore/docs/spatial_agent_vsibench.md)
