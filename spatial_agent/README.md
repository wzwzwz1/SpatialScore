# SpatialAgent（LangGraph ReAct 版）

这个目录是一个新的、独立的 `LangGraph` 版 `SpatialAgent` 实现。  
它遵循论文里的 **ReAct + 空间工具调用** 思路，但**不复用** `version_0/SpatialAgent` 里的 AutoGen 控制流。

## 当前已经实现的能力

- 基于 LangGraph 的 ReAct 推理主图
- `repair / tool / observe / finalize / fail` 节点
- LLM adapter 抽象
- 本地 HuggingFace Qwen-VL 后端
- OpenAI-compatible API 后端
- 统一的工具注册表与 `ToolResult`
- 每次运行的 trace JSON 落盘
- 同机 `lmms-eval` / `VSI-Bench` 桥接

## 当前工具状态

### 已接入真实后端的工具

下面这些工具已经接上了真实推理链路；只要服务器上的依赖、权重、路径配置正确，就会真正执行：

- `CountObjects`
  - 后端：**Rex-Omni pointing**
- `EstimateOpticalFlow`
  - 后端：**RAFT**
- `EstimateHomographyMatrix`
  - 后端：OpenCV ORB + RANSAC
- `EstimateObjectDepth`
  - 后端：**Depth Anything V2 metric depth**
- `GetObjectMask`
  - 后端：**SAM2**
- `GetCameraParametersVGGT`
  - 后端：**VGGT**
- `GetObjectOrientation`
  - 后端：**Orient Anything**
- `EstimateObjectMotion`
  - 后端：**VGGT track head**
- `LocalizeObjects`
  - 后端：grounding bridge，支持可选 `RAM tags`

### 需要服务器侧配置的工具

下面这些工具虽然已经接好了代码，但如果缺少依赖、checkpoint 或 `tool_config`，会返回结构化的 `unavailable` 结果，而不会直接把 agent 跑崩：

- `EstimateOpticalFlow`
- `CountObjects`
- `EstimateObjectDepth`
- `GetObjectMask`
- `GetCameraParametersVGGT`
- `GetObjectOrientation`
- `LocalizeObjects`
- `EstimateObjectMotion`

## CLI 运行方式

### 1. 本地 HuggingFace 模型

```bash
python3 -m spatial_agent \
  --question "Which option is correct?" \
  --question-type multi_choice \
  --input-modality single_image \
  --options A,B,C,D \
  --image-path /path/to/image.jpg \
  --llm-backend hf \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --tool-config-path /disk/wangzhe/SpatialScore/docs/tool_config.server.template.json
```

### 2. OpenAI-compatible API

先设置环境变量：

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_API_BASE_URL=https://yunwu.ai/v1
```

```bash
python3 -m spatial_agent \
  --question "Which option is correct?" \
  --question-type multi_choice \
  --input-modality single_image \
  --options A,B,C,D \
  --image-path /path/to/image.jpg \
  --llm-backend openai_compatible \
  --model-name gpt-4o-mini \
  --tool-config-path /disk/wangzhe/SpatialScore/docs/tool_config.server.template.json
```

如果你不想使用默认环境变量名，也可以显式传：

```bash
python3 -m spatial_agent \
  --question "Which option is correct?" \
  --image-path /path/to/image.jpg \
  --llm-backend openai_compatible \
  --model-name gpt-4o-mini \
  --api-base-url-env OPENAI_API_BASE_URL \
  --api-key-env OPENAI_API_KEY
```

这些 adapter 和部分工具都使用懒加载。如果 `torch`、`transformers`、OpenCV、checkpoint 或模型路径没有准备好，主图仍然可以运行，只是相关工具会返回结构化的 `unavailable` observation。

## 服务器侧 `tool_config`

所有工具的模型路径和运行参数，统一从 `SpatialAgentConfig.tool_config` 读取。

### 常用配置键

- `CountObjects` / `counting`
  - `model_path`（默认 `IDEA-Research/Rex-Omni`）
  - `backend`（`transformers|vllm`）
  - `repo_path`（可选，本地 `Rex-Omni` 仓库路径）
  - `quantization`（可选，例如 `awq`）
  - `attn_implementation`（推荐 `sdpa`，避免强依赖 `flash-attn`）
  - `device_map`
  - `max_tokens`
  - `temperature`
  - `top_p`
  - `top_k`
  - `repetition_penalty`
- `EstimateOpticalFlow` / `raft`
  - `checkpoint_path`
  - `small`
  - `mixed_precision`
  - `alternate_corr`
  - `iters`
- `EstimateObjectDepth` / `depth`
  - `checkpoint_path`
  - `encoder`（`vits|vitb|vitl|vitg`）
  - `max_depth`
  - `input_size`
- `GetObjectMask` / `mask`
  - `model_id`（默认 `facebook/sam2.1-hiera-large`）
  - 或 `checkpoint_path` + `config_path`
- `GetCameraParametersVGGT` / `camera`
  - `hf_model_id`（默认 `facebook/VGGT-1B`）
  - 或 `checkpoint_path`
  - `preprocess_mode`（`pad|crop`）
- `GetObjectOrientation` / `orientation`
  - `checkpoint_repo_id`（默认 `Viglong/Orient-Anything`）
  - `checkpoint_filename`（默认 `croplargeEX2/dino_weight.pt`）
  - `dino_mode`（`small|base|large|giant`）
- `LocalizeObjects` / `localization`
  - `model_id`（默认 `IDEA-Research/grounding-dino-base`）
  - `box_threshold`
  - `text_threshold`
  - `enable_ram_tags`
- `EstimateObjectMotion` / `motion`
  - `hf_model_id` / `checkpoint_path`
  - `preprocess_mode`

### 配置示例

```python
from spatial_agent.runtime.config import SpatialAgentConfig

config = SpatialAgentConfig(
    llm_backend="hf",
    qwen_model_path="/models/Qwen2.5-VL-7B-Instruct",
    tool_config={
        "counting": {
            "model_path": "IDEA-Research/Rex-Omni",
            "backend": "transformers",
            "attn_implementation": "sdpa",
            "device_map": "auto",
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 0.05,
            "top_k": 1,
            "repetition_penalty": 1.05,
        },
        "raft": {
            "checkpoint_path": "/models/raft/raft-things.pth",
            "small": False,
            "mixed_precision": True,
            "alternate_corr": False,
            "iters": 20,
        },
        "depth": {
            "checkpoint_path": "/models/depth_anything/depth_anything_v2_metric_hypersim_vitl.pth",
            "encoder": "vitl",
            "max_depth": 20,
        },
        "mask": {
            "model_id": "facebook/sam2.1-hiera-large",
        },
        "camera": {
            "hf_model_id": "facebook/VGGT-1B",
            "preprocess_mode": "pad",
        },
        "orientation": {
            "checkpoint_repo_id": "Viglong/Orient-Anything",
            "checkpoint_filename": "croplargeEX2/dino_weight.pt",
            "dino_mode": "large",
        },
        "localization": {
            "model_id": "IDEA-Research/grounding-dino-base",
            "box_threshold": 0.3,
            "text_threshold": 0.25,
            "enable_ram_tags": False,
        },
        "motion": {
            "hf_model_id": "facebook/VGGT-1B",
            "preprocess_mode": "pad",
        },
    },
)
```

## Counting 行为

当前 ReAct 编排已经按论文思路切到 `CountObjects` 优先：

- counting 类问题默认优先调用 `CountObjects`
- `CountObjects` 使用 **Rex-Omni** 的 `pointing` 任务来返回实例点位
- 最终答案会基于返回点数收紧为纯数字
- 默认会把 `attn_implementation` 设成 `sdpa`，尽量避免因为缺少 `flash-attn` 而阻塞加载
- 如果 Rex-Omni 不可用，系统不会自动回退到 `GetObjectMask` 或 `LocalizeObjects`

如果你想直接抄服务器模板，用下面这份专门文档：

- [`/disk/wangzhe/SpatialScore/docs/spatial_agent_tool_config_template.md`](/disk/wangzhe/SpatialScore/docs/spatial_agent_tool_config_template.md)
- [`/disk/wangzhe/SpatialScore/docs/tool_config.server.template.json`](/disk/wangzhe/SpatialScore/docs/tool_config.server.template.json)

## VSI-Bench 同机评测

仓库里已经带了一个轻量的 `lmms-eval` 插件包：

```text
lmms_eval_spatialagent_plugin/
```

如果你要在同一台服务器上，用 `thinking-in-space` 的 `VSI-Bench` 直接调这个 agent，而不修改 benchmark 内部实现，请看：

- [`/disk/wangzhe/SpatialScore/docs/spatial_agent_vsibench.md`](/disk/wangzhe/SpatialScore/docs/spatial_agent_vsibench.md)

## Trace 输出

每次运行会把 trace 写到：

```text
.artifacts/spatial_agent/<task_id>.json
```

trace 中包含：

- tool calls
- tool observations
- final answer
- status / error
- reasoning trace

## VSI-Bench 结果分析

如果你已经通过 `lmms_eval` 跑出了 `VSI-Bench` 结果，并且保留了 `SpatialAgent` trace，可以直接用内置分析器生成图表和报告：

```bash
python -m spatial_agent.analysis \
  --samples-path /disk/wangzhe/thinking-in-space/logs/vsibench \
  --trace-dir /tmp/spatial_agent_runs \
  --output-dir /disk/wangzhe/SpatialScore/analysis/vsibench
```

输出包括：

- `summary.json`
- `samples.csv`
- `report.md`
- `report.html`
- `charts/*.png`
- `artifacts/`

分析器会把 `lmms_eval` 的题目级结果，与 `SpatialAgent` 的：

- `tool_calls`
- `tool_observations`
- `reasoning_trace`
- tool artifacts

拼在一起，方便你分析：

- 哪些题型掉分最多
- 哪些 tool 调用频繁
- 哪些 tool 经常 `unavailable` 或 `error`
- 分割 / 深度 / 光流 / 运动轨迹等中间结果是否合理

详细说明见：

- [`/disk/wangzhe/SpatialScore/docs/spatial_agent_vsibench_analysis.md`](/disk/wangzhe/SpatialScore/docs/spatial_agent_vsibench_analysis.md)
