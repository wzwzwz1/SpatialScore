# SpatialScore 运行笔记

## 环境

- Python 环境: `conda activate SpatialScore`
- GPU: RTX 5090, CUDA 13.0
- HF 镜像: `export HF_ENDPOINT=https://hf-mirror.com`（直连不通，必须设）
- API: `export OPENAI_API_BASE_URL=https://yunwu.ai/v1`

## 权重位置

| 模型 | 路径 |
|------|------|
| VGGT-1B | `/disk/wangzhe/models/VGGT-1B/model.pt` |
| RAFT | `/disk/wangzhe/models/raft/raft-things.pth` |
| Depth Anything V2 | `/disk/wangzhe/models/depth_anything/depth_anything_v2_metric_hypersim_vitl.pth` |
| SAM2.1 | HF cache (自动) |
| Rex-Omni 模型 | HF cache (自动) |
| Rex-Omni 代码 | `/disk/wangzhe/Rex-Omni/` |
| VSI-Bench 视频 | `/disk/wangzhe/VSI-Bench/arkitscenes.zip` |

## 运行命令

```bash
cd /disk/wangzhe/SpatialScore
conda activate SpatialScore

export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<your_token>
export OPENAI_API_KEY=<key>
export OPENAI_API_BASE_URL=https://yunwu.ai/v1

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m spatial_agent.vsibench_cli \
  --split test --doc-id 0 \
  --video-num-frames 16 \
  --artifact-dir /tmp/spatial_agent_runs_visra \
  --tool-config-path configs/tool_config.server.json \
  --dataset-cache-dir /disk/wangzhe/VSI-Bench \
  --keep-video-frames --max-steps 15 \
  --hf-token $HF_TOKEN \
  --llm-backend openai_compatible \
  --api-base-url $OPENAI_API_BASE_URL \
  --api-key $OPENAI_API_KEY \
  --model-name gpt-4o
```

2 帧 smoke test（验证管线）:
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -c "
from pathlib import Path
from spatial_agent.runtime.config import SpatialAgentConfig, load_tool_config
from spatial_agent.tools.video_counting_3d import CountVideoObjects3DTool

frames = sorted(str(p) for p in Path('/tmp/spatial_agent_runs_visra/sampled_frames/vsibench/test/0').glob('*.jpg'))[:2]
config = SpatialAgentConfig(artifact_dir='/tmp/spatial_agent_smoke',
    tool_config=load_tool_config('configs/tool_config.server.json'))
config.tool_config['video_counting_3d'] = dict(config.tool_config['video_counting_3d'])
config.tool_config['video_counting_3d'].update({
    'num_frames': 2, 'use_sam_masks': False, 'use_tracking': False,
    'max_detections_per_frame': 5, 'bbox_point_stride': 16, 'max_tokens': 1024,
})
result = CountVideoObjects3DTool(config).invoke(images=frames, objects=['table'])
print(result['status'], result.get('error'))
print(result.get('payload',{}).get('instance_count'))
"
```

## 发现的问题

### 1. SAM2 tracking 时 device mismatch (cuda:0 vs cpu)

`CountVideoObjects3D` 开启 `use_sam_masks`/`use_tracking` 时，SAM2 内部某处 tensor 在 CPU 而非 CUDA，导致 `torch.cat` 报错。**临时修复**: config 中关闭 `use_sam_masks` 和 `use_tracking`。

### 2. Agent repair loop — 视频计数答案被反复驳回

`route_node.py:28` 对视频计数任务强制逐帧检查后才允许输出最终答案，即使 `CountVideoObjects3D` 已完成 3D 全帧分析。导致 LLM 每次 `finish` 都被驳回 → repair → 再次 finish → 最终 "Maximum repair attempts exceeded"。

**位置**: `spatial_agent/graph/nodes/route_node.py` 第 28-37 行，`is_video_counting_task` 判断未区分"已调用过 3D 计数工具"和"从未调用"的场景。

### 3. 帧数越多过计数越严重

| 帧数 | 3D 计数 | GT |
|------|--------|----|
| 16 | 9 | 4 |
| 32 | 11 | 4 |

聚类阈值 `cg_distance_threshold`(0.35) 可能需要按场景调大。

### 4. 依赖安装: 3 个本地包不在 PyPI

`multiscaledeformableattention`、`rex-omni`、`sam-2` 需跳过 pip 安装，运行时通过 `sys.path` 动态加载。其中 `multiscaledeformableattention` 仅 CountVid fallback 需要，当前不用。

### 5. VSI-Bench 视频下载

数据集视频以 zip 形式存储（LFS），hf-mirror 不支持 LFS 解析，需通过代理从 huggingface.co 下载: `curl -x http://127.0.0.1:17898 ...`
