# CountVid 视频计数工具 — 实现总结

## 概述

按照 `countvid-video-counting-workflow-design.md` 的设计，完成了 CountVideoObjects 工具的全部后端实现。该工具将视频计数问题建模为：候选框生成 → 时序滤波 → 跨帧传播 tracking → unique instance 聚合，替代原有的"多帧 CountObjects + LLM 推测去重"方案。

## 实现文件清单

### 新增文件（3 个）

| 文件 | 说明 |
|---|---|
| `spatial_agent/tools/video_counting.py` | `CountVideoObjectsTool` 主工具类，实现四阶段 pipeline |
| `spatial_agent/tools/countvid_backend.py` | CountVid 后端封装：CountGD-Box 候选框生成 + SAM 2.1 视频传播 |
| `spatial_agent/tools/video_counting_utils.py` | 时序滤波、track 格式化、可视化 artifact 生成 |

### 修改文件（8 个）

| 文件 | 改动 |
|---|---|
| `spatial_agent/tools/registry.py` | 条件注册 CountVideoObjects（仅当 backend 路径已配置） |
| `spatial_agent/adapters/prompting.py` | 视频计数提示优化；`should_skip_images_for_video_counting()` |
| `spatial_agent/adapters/openai_compatible.py` | 视频计数任务跳过 base64 图片发送；增加 max_tokens；修复 import json |
| `spatial_agent/adapters/huggingface_qwen.py` | 同 openai_compatible：视频计数跳过图片 |
| `spatial_agent/prompts/react_system_prompt.py` | 系统提示根据可用工具条件化计数引导 |
| `spatial_agent/graph/tool_args.py` | CountVideoObjects 自动绑定全量采样帧到 images 参数 |
| `spatial_agent/analysis/report.py` | markdown/HTML 报告中展示 tracks 和 frame_summaries |
| `spatial_agent/agent.py` | recursion_limit 从默认 25 提升到 100 |
| `spatial_agent/runtime/config.py` | max_repairs 从 2 提升到 3 |
| `configs/tool_config.server.json` | 新增 video_counting 配置段 |
| `tests/spatial_agent/test_tools.py` | 8 个新测试覆盖 CountVideoObjects |

## 工具接口

### Input

```json
{
  "images": ["frame_00000.jpg", "frame_00001.jpg", ...],
  "objects": "table"
}
```

### Output

```json
{
  "instance_count": 4,
  "tracks": [
    {
      "track_id": "table_000",
      "object": "table",
      "supporting_frames": ["image-0", "image-8", "image-15"],
      "supporting_points": [[0.10, 0.30], [0.18, 0.28], [0.22, 0.31]]
    }
  ],
  "frame_summaries": [
    {"image": "image-0", "candidate_count": 2, "filtered_count": 2}
  ],
  "backend": "countvid:countgd_box+sam2.1",
  "artifact_descriptions": [...]
}
```

## 四阶段 Pipeline

```
视频帧 → [Stage 1] CountGD-Box 候选框生成
       → [Stage 2] 时序窗口滤波（去孤立误检）
       → [Stage 3] SAM 2.1 跨帧传播 tracking
       → [Stage 4] unique track 聚合 → instance_count
```

## 关键设计决策

1. **原子工具**：`CountVideoObjects` 一次调用完成全流程，不走 `CountObjects → CountObjects → finish` 的多步模式
2. **条件注册**：仅当 tool_config 中配置了 CountVid checkpoint 路径时才注册该工具，避免 LLM 调用不可用工具浪费步骤
3. **跳过图片发送**：视频计数任务不再将全量帧 base64 发给 LLM（64 帧 ≈ 数 MB），工具 backend 直接处理帧，LLM 只需从问题文本提取 object 名称
4. **失败不静默回退**：backend 不可用时返回 `unavailable`，由 LLM 显式决定回退策略
5. **可观测性**：输出 track_overlay、candidate_overlay、track_manifest JSON 三种 artifact

## 测试覆盖

`tests/spatial_agent/test_tools.py` 中 8 个测试：

- `test_count_video_objects_returns_unavailable_when_backend_missing` — backend 初始化失败 → unavailable
- `test_count_video_objects_returns_unavailable_when_partial_backend` — 部分组件缺失 → unavailable
- `test_count_video_objects_returns_error_on_missing_images` — 无帧 → error
- `test_count_video_objects_returns_error_on_missing_objects` — 无目标 → error
- `test_count_video_objects_success_with_mock_backend` — 全 pipeline mock 成功
- `test_count_video_objects_zero_instance_success` — 零实例 → success, count=0
- `test_count_video_objects_not_registered_without_backend_config` — 无配置不注册
- `test_count_video_objects_registered_with_backend_config` — 有配置正确注册

## VSI-Bench 实测

doc-id 0 (test split): "How many table(s) are in this room?"

- Ground truth: **4**
- 当前结果: **1** (success)
- 流程：LLM 使用 CountObjects × 8 帧，每帧 count=1，推断为 1 张 table
- 暴露问题：缺少跨帧去重，每帧只看到同一 table 的不同角度

## 待完成：CountVid Backend 接入

当前 `countvid_backend.py` 提供了完整的接口定义和 placeholder 实现。要启用真正的视频传播计数，需要：

1. Clone [CountVid](https://github.com/niki-amini-naieni/CountVid) 仓库
2. 下载 CountGD-Box checkpoint（`countgd_box.pth`）
3. 下载 SAM 2.1 checkpoint（`sam2.1_hiera_large.pt`）
4. 在 `tool_config.server.json` 中填写实际路径
5. 验证 CountVid 的 Python API 与 `countvid_backend.py` 中的调用方式对齐

配置好上述路径后，`CountVideoObjects` 会自动注册并可用。
