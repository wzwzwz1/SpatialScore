# SpatialAgent + VSI-Bench 使用说明

这份文档说明如何在同一台服务器上，把新的 `spatial_agent` 接到 [`VSI-Bench`](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) 做评测。

适用前提：

- benchmark 仓库在 `/Users/wz/code/thinking-in-space`
- agent 仓库在 `/Users/wz/code/SpatialScore`
- 两者运行在同一台服务器

这个接法的目标是：**尽量不修改 `thinking-in-space` 内部代码**。  
具体做法是通过 `LMMS_EVAL_PLUGINS`，让 `lmms-eval` 从当前仓库发现并加载一个插件模型。

## 1. 这套集成里新增了什么

新增内容主要在当前仓库：

- `spatial_agent/`
  - ReAct 图
  - 本地 HuggingFace 后端
  - OpenAI-compatible API 后端
  - 视频转帧桥接
- `lmms_eval_spatialagent_plugin/`
  - 一个 `lmms-eval` 插件模型：`spatial_agent_api`

`thinking-in-space` 仍然继续使用它自己的 `vsibench` task 和 metric。

## 2. 环境准备

建议让两个仓库共用同一个 Python 环境。

至少需要满足：

1. `thinking-in-space` 能在当前环境中运行。
2. `SpatialScore` 的依赖也在同一个环境里可用。
3. benchmark 数据和模型资源已经在服务器上准备好。

如果你不想把 `SpatialScore` 安装成 package，也没关系，下面的方式会用 `PYTHONPATH` 解决导入。

## 3. 必要环境变量

在你运行 `VSI-Bench` 的 shell 里先执行：

```bash
export PYTHONPATH=/Users/wz/code/SpatialScore:$PYTHONPATH
export LMMS_EVAL_PLUGINS=lmms_eval_spatialagent_plugin
```

如果你走 API 推理，再补一个：

```bash
export OPENAI_API_KEY=your_api_key_here
```

如果你用的不是 `OPENAI_API_KEY` 这个环境变量名，也可以在 `model_args` 里传：

```text
api_key_env=你的变量名
```

## 4. 先做 smoke test

正式全量跑之前，建议先用 `--limit 2` 做一个小样本 dry run。

先进入 benchmark 仓库：

```bash
cd /Users/wz/code/thinking-in-space
```

### 4.1 API 后端

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,api_base_url=https://api.openai.com/v1,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs,tool_config_path=/Users/wz/code/SpatialScore/docs/tool_config.server.template.json \
  --tasks vsibench \
  --batch_size 1 \
  --limit 2 \
  --log_samples \
  --output_path ./logs/vsibench_smoke
```

### 4.2 本地模型后端

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=hf,model_path=/path/to/Qwen2.5-VL-7B-Instruct,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs,tool_config_path=/Users/wz/code/SpatialScore/docs/tool_config.server.template.json \
  --tasks vsibench \
  --batch_size 1 \
  --limit 2 \
  --log_samples \
  --output_path ./logs/vsibench_smoke
```

如果 smoke test 没问题，再去掉 `--limit 2` 跑全量。

## 5. 全量 VSI-Bench 评测

### 5.1 API 后端

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,api_base_url=https://api.openai.com/v1,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs,tool_config_path=/Users/wz/code/SpatialScore/docs/tool_config.server.template.json \
  --tasks vsibench \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/vsibench
```

### 5.2 本地模型后端

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=hf,model_path=/path/to/Qwen2.5-VL-7B-Instruct,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs,tool_config_path=/Users/wz/code/SpatialScore/docs/tool_config.server.template.json \
  --tasks vsibench \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/vsibench
```

## 6. 支持的 `model_args`

插件模型名固定是：

```text
--model spatial_agent_api
```

支持的 `model_args` 包括：

- `llm_backend=hf|openai_compatible`
- `model_path=/path/to/local/model`
- `model_name=<远端模型名>`
- `api_base_url=<OpenAI-compatible base URL>`
- `api_key=<可选，直接传 key>`
- `api_key_env=<从哪个环境变量读取 key>`
- `api_timeout=<秒>`
- `max_steps=<推理步数>`
- `video_num_frames=<每个视频均匀采样多少帧>`
- `video_frame_dir=<可选，显式指定帧缓存目录>`
- `artifact_dir=<trace 和临时产物目录>`
- `keep_video_frames=true|false`
- `tool_config_path=<JSON 配置文件路径>`

例如，下面这个命令会保留采样后的帧，便于排查问题：

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,api_base_url=https://api.openai.com/v1,video_num_frames=24,video_frame_dir=/tmp/vsi_frames,artifact_dir=/tmp/spatial_agent_runs,keep_video_frames=true,tool_config_path=/Users/wz/code/SpatialScore/docs/tool_config.server.template.json \
  --tasks vsibench \
  --batch_size 1 \
  --limit 2 \
  --log_samples \
  --output_path ./logs/vsibench_debug
```

## 7. 插件内部做了什么

对每一个 VSI-Bench 样本，流程是：

1. `thinking-in-space` 从数据集缓存里解析出视频路径
2. `spatial_agent_api` 把视频均匀采样成多帧
3. 这些帧被映射成 `SpatialAgent` 的输入任务
4. `SpatialAgent` 跑 ReAct 推理并返回 `final_answer`
5. `lmms-eval` 继续使用原始 `vsibench` metric 对答案打分

## 8. 你应该看哪些输出

### 8.1 Benchmark 输出

`thinking-in-space` 会把评测结果写到你传入的 `--output_path` 下，例如：

```text
/Users/wz/code/thinking-in-space/logs/vsibench/
```

通常这里会有：

- 聚合结果 `results.json`
- 如果开启了 `--log_samples`，还会有逐样本日志

### 8.2 SpatialAgent trace

`SpatialAgent` 会把每个样本的 trace 写到：

```text
<artifact_dir>/<task_id>.json
```

例如：

```text
/tmp/spatial_agent_runs/vsibench___test___42.json
```

这些 trace 里会包含：

- final answer
- status / error
- tool calls
- tool observations
- reasoning trace

## 9. 额外说明

- `VSI-Bench` 是视频任务，所以桥接层会先采样帧，再调用 agent
- `batch_size 1` 是最稳妥的默认值，因为每个样本都可能触发多步推理和 tool 调用
- 如果 `keep_video_frames=false`，每个样本结束后采样帧会被清理掉
- 如果你想检查 agent 的实际输入帧，建议设成 `keep_video_frames=true`
- 如果你还需要把空间工具链一起配好，可以继续看：
  - [`/Users/wz/code/SpatialScore/docs/spatial_agent_tool_config_template.md`](/Users/wz/code/SpatialScore/docs/spatial_agent_tool_config_template.md)
