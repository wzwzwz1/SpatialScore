# lmms-eval SpatialAgent 插件

这个插件让 `thinking-in-space` 里的 `VSI-Bench` 评测器，可以通过 `LMMS_EVAL_PLUGINS` 在同一台服务器上直接调用 `SpatialAgent`，而不需要修改 benchmark 内部逻辑。

## 基本用法

先设置环境变量：

```bash
export PYTHONPATH=/disk/wangzhe/SpatialScore:$PYTHONPATH
export LMMS_EVAL_PLUGINS=lmms_eval_spatialagent_plugin
export OPENAI_API_KEY=your_api_key_here
export OPENAI_API_BASE_URL=https://yunwu.ai/v1
```

然后进入 `/Users/wz/code/thinking-in-space` 执行：

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,video_num_frames=16,tool_config_path=/disk/wangzhe/SpatialScore/docs/tool_config.server.template.json \
  --tasks vsibench \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/vsibench
```

如果你走本地模型，把 `model_args` 换成：

```text
llm_backend=hf,model_path=/path/to/Qwen2.5-VL-7B-Instruct
```

完整操作说明请看：

- [`/disk/wangzhe/SpatialScore/docs/spatial_agent_vsibench.md`](/disk/wangzhe/SpatialScore/docs/spatial_agent_vsibench.md)
