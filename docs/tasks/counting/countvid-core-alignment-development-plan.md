# CountVid 核心对齐开发文档

## 1. 文档目标

本文档用于在当前 `SpatialScore` 仓库内，规划如何把已经接入的 `CountVideoObjects` 工具，从“可运行的工程骨架”推进到“与 CountVid 核心方法对齐”的视频计数实现。

本文档关注的是 **CountVid 核心对齐**，不是泛泛而谈的视频计数愿景，因此会明确回答下面四个问题：

1. 原论文在 count 任务上的原始编排策略是什么
2. CountVid 论文真正新增了哪些关键能力
3. 当前仓库已经实现了什么、还缺什么
4. 下一步应按什么顺序推进，哪些文件要改，什么算完成

---

## 2. 参考资料

### 2.1 SpatialScore 原论文与整理文档

- `docs/references/papers/SpatialScore - Wu et al. 2026.pdf`
- `docs/guides/spatial_agent_paper_tooling_and_prompts.md`

### 2.2 CountVid 论文与开源实现

- CountVid 论文：`Open-World Object Counting in Videos`
- CountVid 代码仓库：`https://github.com/niki-amini-naieni/CountVid`
- CountVid 项目页：`https://www.robots.ox.ac.uk/~vgg/research/countvid/`

### 2.2.1 本文档与 CountVid 的关系说明

本文档讨论的 **视频 count workflow**，核心方法论是参考 CountVid，而不是声称这部分已经由 SpatialScore 原论文完整给出。

更准确地说：

- **SpatialScore 原论文** 提供的是：
  - agent 框架
  - ReAct / Plan-Execute 编排
  - 单图 `CountObjects` 工具口径
- **CountVid** 提供的是：
  - 视频级唯一实例计数的方法论
  - per-frame candidate generation
  - temporal filtering
  - propagation / tracking
  - unique instance aggregation

因此，本仓库后续的 count 路线应理解为：

- **单图 count**：继续沿用 SpatialScore 原始 `CountObjects` 路线
- **视频 count**：在 SpatialScore agent 框架中，引入 CountVid 风格的专用 workflow

这是一种“**SpatialScore 作为 agent 容器，CountVid 作为 video counting 方法参考**”的组合，而不是把二者混为同一篇论文中的同一套原始实现。

### 2.3 当前仓库内与 CountVid 相关的实现

- `spatial_agent/tools/video_counting.py`
- `spatial_agent/tools/countvid_backend.py`
- `spatial_agent/tools/video_counting_utils.py`
- `spatial_agent/tools/registry.py`
- `spatial_agent/adapters/prompting.py`
- `docs/tasks/counting/countvid-video-counting-workflow-design.md`
- `docs/tasks/counting/countvid-implementation-summary.md`

---

## 3. 原论文与当前扩展的关系

## 3.1 SpatialScore 原论文中的 count 编排

SpatialScore 原论文对 count 任务的核心策略非常简单：

1. count 相关问题优先使用 `CountObjects`
2. `CountObjects` 是 **单图工具**
3. 工具返回的是实例点，不是直接返回 count
4. 点的数量就是数量
5. ReAct 一次只允许一个 action

这意味着原论文原始链路更接近：

```text
LLM -> CountObjects(image-k, objects) -> points -> LLM 统计点数 -> Terminate
```

原论文没有系统性描述以下内容：

- 视频采样多少帧
- 视频中的 count 是否需要跨帧去重
- 如何建立跨帧实例 identity
- 如何把多个 frame-level count 合成 video-level unique count

因此，**视频级唯一实例计数并不是 SpatialScore 原论文里已经解决的问题**。

## 3.2 为什么需要 CountVid 风格的 workflow

当前仓库已经把视频计数从“单帧一次调用就结束”推进到：

- 多步 ReAct
- 多帧顺序调用
- 多 step JSON 队列串行执行

但只靠：

```text
CountObjects(frame_1)
CountObjects(frame_2)
CountObjects(frame_3)
-> LLM 总结
```

仍然无法稳定得到 room-level unique count。根因是：

1. 单帧 `CountObjects` 看到的是“当前可见实例”
2. 不同帧看到的可能是同一实例的不同视角
3. 也可能是不同实例在不同帧分别出现
4. LLM 不适合隐式完成实例级跨帧去重

CountVid 之所以相关，是因为它把视频计数建模成：

1. per-frame candidate generation
2. temporal filtering
3. video propagation / tracking
4. unique instance aggregation

即：**不要把跨帧去重留给 LLM，而要把它做成显式的视频实例流程。**

---

## 4. 当前仓库现状评估

## 4.1 已完成的部分

当前实现已经具备以下基础设施：

1. 新增了原子工具 `CountVideoObjects`
2. 对 `video + counting` 场景提供了专门路由入口
3. 对视频计数任务支持“跳过把全量帧 base64 发给 LLM”
4. 输出中已经包含：
   - `instance_count`
   - `tracks`
   - `frame_summaries`
   - 调试 artifact
5. 测试中已经覆盖：
   - 工具注册
   - backend 缺失时的 unavailable
   - mock 成功路径

这说明 **工具接入层、调用协议层、调试层已经搭好骨架**。

## 4.2 当前与 CountVid 核心对齐的差距

下面这些差距是当前实现尚未真正对齐 CountVid 的关键原因。

### Gap A: Stage 1 还未对齐真实 CountGD-Box 接口

当前 `generate_candidates_countgd(...)` 假设本地存在：

- `countgd.models.countgd_box.build_countgd_box`

并假设可以直接：

```python
outputs = model(image, objects)
```

但这只是一个本地封装假设，不等于已经对齐 CountVid 官方代码的真实调用路径、输入格式、文本 prompt 方式、阈值控制方式和输出结构。

换句话说：

- 当前有“接口壳”
- 还没有“官方实现对齐”

### Gap B: Stage 2 的 temporal filtering 还不是真正的时序去噪

当前 `temporal_window_filter(...)` 的行为更接近：

- 如果当前帧本身有候选，就保留
- 如果邻帧有候选，就也保留当前帧

这并不能真正去掉“只出现一帧的孤立误检”。  
它更像一个轻量标记器，不是 CountVid 风格的时序筛选。

### Gap C: Stage 3 的 SAM 2 propagation 还没有形成稳定 identity

当前 `run_sam2_propagation(...)` 存在几个结构性问题：

1. 每个 frame candidate 都单独起一个 track
2. `track_id = f"{obj_name}_{cand_idx:03d}"` 会在不同起始帧重复
3. 没有看到明确的 prompt seed / video state 初始化
4. 没有真实的 forward + backward propagation 管线
5. 没有显式的 track merge / collision resolution

所以当前“track”更像候选占位结构，不是视频级唯一实例轨迹。

### Gap D: Stage 4 没有真正做 unique instance aggregation

当前 `aggregate_unique_tracks(...)` 只是按 `min_track_support` 过滤一次。

它没有完成：

- 同一实例跨帧合并
- 同一实例多次初始化的去重
- 相似轨迹冲突消解
- 多候选映射到一个唯一对象的聚合

因此，当前的 `instance_count = len(accepted_tracks)` 在语义上仍然不足以代表 CountVid 式 unique count。

### Gap E: 坐标语义还未统一

当前代码里，`point` 既可能被当成：

- 像素坐标
- 归一化坐标

overlay 又按归一化坐标乘宽高画图。  
如果不先统一坐标约定，后续：

- track merge
- artifact 调试
- frame-level / video-level compare

都会变得不可靠。

### Gap F: 工具注册条件过宽

当前 `CountVideoObjects` 的注册条件只检查：

- `countgd_checkpoint_path`
- `sam2_checkpoint_path`

是否为非空字符串，没有检查：

- 路径是否真实存在
- repo 是否存在
- sam2 config 是否存在
- backend 是否可导入

这会导致：

- 工具在配置占位路径时就提前暴露给 LLM
- 提示词还会引导优先使用它
- 但 backend 实际不可用

这会放大调试噪音。

---

## 5. CountVid 核心对齐目标

本项目中的“CountVid 核心对齐”定义如下：

## 5.1 功能目标

对 `video + object_counting` 任务，系统应做到：

1. 输入为按时间排序的视频采样帧
2. 根据文本目标生成 per-frame candidates
3. 过滤短时不稳定候选
4. 把候选作为 prompt 送入视频传播模型
5. 生成视频级实例轨迹
6. 在轨迹层做 unique instance aggregation
7. 输出 video-level unique count

## 5.2 非目标

本轮不要求：

1. 训练或微调 CountVid 模型
2. 支持 exemplar-conditioned counting
3. 完全复现 CountVid 论文所有实验
4. 在第一版就支持任意长视频的全量高效处理

---

## 6. 推荐总体方案

## 6.1 保持新工具，不回退到 LLM 多工具编排

推荐继续保留：

- `CountObjects`: 单图 count
- `CountVideoObjects`: 视频级 count

而不是把 CountVid 拆成多个零散工具再交给 LLM 编排。

原因：

1. CountVid 的关键价值就在于紧耦合的视频实例流程
2. LLM 不适合承担 identity-level merge
3. 原子工具更容易测试、调试和评估

## 6.2 推荐最终形态

`CountVideoObjects` 应成为以下 pipeline 的代码化实现：

```text
ordered sampled frames
-> CountGD-Box candidates
-> temporal filtering
-> SAM 2.1 promptable video propagation
-> track canonicalization
-> unique track aggregation
-> instance_count
```

---

## 7. 模块拆分设计

## 7.1 文件职责

### `spatial_agent/tools/video_counting.py`

职责：

- 保持工具输入输出协议稳定
- 负责四阶段 pipeline 的调用编排
- 做错误转换、artifact 收集、payload 组装

不应承载：

- 复杂 track merge 算法
- CountGD / SAM2 具体适配细节

### `spatial_agent/tools/countvid_backend.py`

职责：

- 对齐 CountVid 官方上游模块
- 封装：
  - CountGD-Box candidate generation
  - SAM 2.1 propagation initialization
  - propagation state 管理
  - 原始 track 生成

这是 CountVid 对齐的核心文件。

### `spatial_agent/tools/video_counting_utils.py`

职责：

- 纯 Python 的中间处理与后处理
- temporal filtering
- track canonicalization
- unique aggregation
- artifact visualization
- manifest 生成

这里应该放“算法可解释、易测试”的聚合规则。

### `spatial_agent/tools/registry.py`

职责：

- 严格控制工具何时注册
- 避免未就绪 backend 误暴露给 LLM

### `spatial_agent/adapters/prompting.py`

职责：

- 只负责在视频计数任务中引导 LLM 优先选择 `CountVideoObjects`
- 不负责任何跨帧去重逻辑

---

## 8. 分阶段实现计划

## Phase 0：基线修正与能力边界收紧

### 目标

先把当前骨架里的“误导性完成态”收紧，保证接下来的对齐工作站在稳定基线上。

### 要做的事

1. 收紧工具注册条件
   - 必须检查：
     - `countgd_repo_path` 存在
     - `countgd_checkpoint_path` 存在
     - `sam2_checkpoint_path` 存在
     - `sam2_config_name` 存在
2. 统一坐标语义
   - 明确所有中间结构中的 `point` 是：
     - 像素坐标，或
     - 归一化坐标
   - 推荐统一成 **像素坐标内部表示**
   - 对外输出时再显式转换
3. 修正文档和 payload 说明
   - 明确当前版本不是完整 CountVid 对齐
4. 为 backend 失败补充更可诊断的错误信息

### 涉及文件

- `spatial_agent/tools/registry.py`
- `spatial_agent/tools/countvid_backend.py`
- `spatial_agent/tools/video_counting_utils.py`
- `docs/tasks/counting/countvid-implementation-summary.md`

### 完成标准

- 未配置真实 CountVid 依赖时，`CountVideoObjects` 不注册
- 点坐标在代码中只有一种内部语义
- 当前实现状态的文档表述准确

---

## Phase 1：对齐 CountGD-Box 候选生成

### 目标

让 Stage 1 使用 **CountVid 官方 CountGD-Box 调用方式**，而不是本地猜测式包装。

### 要做的事

1. 阅读并确认 CountVid 官方仓库中：
   - CountGD-Box 的构建函数
   - 推理入口
   - 文本 prompt 格式
   - 阈值与输出结构
2. 在 `countvid_backend.py` 中实现正式 adapter：
   - `load_countgd_box_backend(...)`
   - `infer_frame_candidates(...)`
3. 规范候选结构，统一输出：

```python
{
  "bbox_xyxy": [x1, y1, x2, y2],
  "point_xy": [cx, cy],
  "score": float,
  "label": str,
}
```

4. 增加阈值配置：
   - detection threshold
   - max candidates per frame

### 涉及文件

- `spatial_agent/tools/countvid_backend.py`
- `configs/tool_config.server.json`
- `tests/spatial_agent/test_tools.py`

### 完成标准

- Stage 1 能在真实 CountVid 依赖下产出稳定的 per-frame candidates
- 候选结构统一、可序列化、可测试

---

## Phase 2：实现真正的 temporal filtering

### 目标

把当前“邻帧存在候选就保留”的弱逻辑，替换为真正能抑制单帧 spike 的时序过滤。

### 推荐策略

第一版不要过于激进，采用简单但可解释的规则：

1. 对每个 frame candidate，先建立局部邻域支持关系
2. 只有当候选在时间窗口内获得足够支持时，才进入 propagation seed
3. 孤立单帧候选在没有强分数支撑时被过滤

### 一个可落地的 V1 规则

对于某个候选 `c_t`：

- 在 `t-1` 或 `t+1` 中存在空间接近且类别一致的候选，则保留
- 否则只有在 `score >= high_conf_threshold` 时保留

这样可以兼顾：

- 避免纯单帧假阳性
- 又不至于错杀短暂可见的真实对象

### 涉及文件

- `spatial_agent/tools/video_counting_utils.py`
- `tests/spatial_agent/test_tools.py`

### 完成标准

- 单帧孤立误检在测试中能被过滤
- 高置信孤立候选可按规则保留
- `frame_summaries` 能反映过滤前后数量

---

## Phase 3：对齐 SAM 2.1 propagation 与 track identity

### 目标

让 Stage 3 真正形成视频级实例轨迹，而不是“每帧候选列表的松散串联”。

### 要做的事

1. 按 CountVid 官方实现确认 SAM 2.1 video predictor 的真实工作流：
   - video state 初始化
   - prompt seed 注入方式
   - forward / backward propagation
   - mask / points / boxes 的返回结构
2. 为每个 seed 候选生成 **全局唯一 seed_id**
   - 推荐格式：
     - `"{object}_{start_frame_idx}_{seed_idx}"`
3. 保持 raw track 级别结构：

```python
{
  "seed_id": "...",
  "object": "table",
  "start_frame_idx": 12,
  "support_frames": [...],
  "points_xy": [...],
  "masks": [... optional ...],
  "scores": [... optional ...],
}
```

4. 支持 forward / backward propagation 后合并成一个 canonical track

### 注意点

- 不要把 track_id 建立在单个 `cand_idx` 上
- 不要把“同一 object 的不同 seed”直接当最终唯一实例
- raw track 与 final unique track 必须分层

### 涉及文件

- `spatial_agent/tools/countvid_backend.py`
- `spatial_agent/tools/video_counting_utils.py`
- `tests/spatial_agent/test_tools.py`

### 完成标准

- 同一个 seed 在 propagation 后得到稳定的跨帧支持集合
- raw track 不再发生跨起始帧 ID 冲突
- tracks 中的 frame support 和 point/mask evidence 可稳定落盘

---

## Phase 4：实现 unique instance aggregation

### 目标

这是 CountVid 核心对齐最关键的一步：  
把多个 raw tracks 聚合成 **视频级唯一实例集合**。

### 设计原则

最终计数对象不应是：

- raw detections
- raw seeds
- raw propagation runs

而应是：

- **canonical unique instances**

### 推荐 V1 聚合策略

对同一 `object` 下的 raw tracks，按以下信号做合并：

1. **temporal overlap**
   - 在共同帧中是否有稳定支持
2. **spatial proximity**
   - 共同帧上的点或 mask centroid 是否接近
3. **mask / bbox overlap**
   - 如果拿得到 mask，优先用 mask IoU
4. **track consistency**
   - propagation 路径是否连贯

### 推荐流程

1. 构建 raw track 相似图
2. 边条件满足则连边
3. 对图做连通分量或聚类
4. 每个 cluster 形成一个 canonical unique track

输出示例：

```python
{
  "track_id": "table_uq_002",
  "object": "table",
  "member_seed_ids": [...],
  "supporting_frames": [...],
  "supporting_points": [...],
}
```

最终：

```python
instance_count = len(unique_tracks)
```

### 涉及文件

- `spatial_agent/tools/video_counting_utils.py`
- `spatial_agent/tools/video_counting.py`
- `tests/spatial_agent/test_tools.py`

### 完成标准

- 同一实例由多个 raw track 触发时，会被合并成一个 unique track
- `instance_count` 基于 unique tracks，而不是 raw tracks
- 调试输出可追溯一个 unique track 由哪些 raw seeds 合成

---

## Phase 5：完善 artifact、报告与可观测性

### 目标

让 CountVid 对齐版可调试、可解释、可评估。

### 必须补充的 artifact

1. `candidate_overlay`
   - 显示 Stage 1 候选与过滤结果
2. `seed_overlay`
   - 显示进入 propagation 的 seeds
3. `track_overlay`
   - 显示 raw tracks
4. `unique_track_overlay`
   - 显示最终 unique instances
5. `countvid_tracks.json`
   - 原始结构化轨迹清单
6. `countvid_unique_tracks.json`
   - 聚合后的唯一实例清单

### 报告中建议新增字段

- `backend_status`
- `frame_candidate_stats`
- `seed_count`
- `raw_track_count`
- `unique_track_count`
- `dropped_candidates`
- `dropped_tracks`

### 涉及文件

- `spatial_agent/tools/video_counting.py`
- `spatial_agent/tools/video_counting_utils.py`
- `spatial_agent/analysis/report.py`

### 完成标准

- 一次失败或错误结果可以从 artifact 中定位到是：
  - candidate generation 问题
  - temporal filtering 问题
  - propagation 问题
  - aggregation 问题

---

## Phase 6：Agent 路由与回退策略收口

### 目标

让 `CountVideoObjects` 成为视频 count 的正式首选路径，同时保留可控回退。

### 路由策略

对于：

- `input_modality == video`
- count 相关问题

优先使用：

- `CountVideoObjects`

### 回退策略

第一版推荐保守策略：

1. backend 不可用：
   - 返回 `unavailable`
   - 不静默回退
2. backend 可用但 pipeline 内部 error：
   - 返回 `error`
   - 由 LLM 或上层显式决定是否回退到 `CountObjects`

不建议在 CountVid tool 内部自动偷偷回退到：

- representative-frame `CountObjects`

因为这样会掩盖 CountVid pipeline 的真实问题。

### 涉及文件

- `spatial_agent/tools/registry.py`
- `spatial_agent/adapters/prompting.py`
- `spatial_agent/adapters/openai_compatible.py`
- `spatial_agent/adapters/huggingface_qwen.py`

### 完成标准

- 视频计数默认优先 `CountVideoObjects`
- backend 未就绪时行为明确、可见、可调试

---

## 9. 配置设计建议

建议把 `video_counting` 配置统一整理为：

```json
{
  "video_counting": {
    "countgd_repo_path": "...",
    "countgd_checkpoint_path": "...",
    "countgd_box_threshold": 0.25,
    "countgd_max_candidates_per_frame": 32,
    "sam2_checkpoint_path": "...",
    "sam2_config_name": "...",
    "window_size": 3,
    "high_conf_threshold": 0.5,
    "min_track_support": 1,
    "track_merge_point_threshold_px": 24,
    "track_merge_iou_threshold": 0.4,
    "device": "cuda"
  }
}
```

说明：

- `point_threshold_px` 和 `iou_threshold` 是 unique aggregation 的关键超参
- 所有 threshold 都应在 manifest 中回显，方便实验对比

---

## 10. 测试计划

## 10.1 单元测试

### Stage 1

- CountGD backend 缺失 -> unavailable
- 真实 backend adapter 返回统一 candidate 结构

### Stage 2

- 单帧孤立误检会被过滤
- 高置信孤立候选按规则保留

### Stage 3

- seed_id 全局唯一
- propagation 结果能形成 raw tracks

### Stage 4

- 两个 raw tracks 属于同一实例时被合并
- 不同实例不会被误合并
- `instance_count` 基于 unique tracks

## 10.2 集成测试

### Tool-level

- `CountVideoObjects` 在 mock CountGD + mock SAM2 下返回：
  - `instance_count`
  - `tracks`
  - `frame_summaries`
  - artifact

### Routing-level

- 视频 count 问题优先使用 `CountVideoObjects`
- 图片 count 问题仍使用 `CountObjects`

## 10.3 回归测试

- 现有 `CountObjects` 行为不受影响
- 非视频任务 prompt 不受影响
- 未配置 CountVid backend 时工具不误注册

---

## 11. 验收标准

当下面条件都满足时，可以认为“CountVid 核心对齐”第一版完成：

1. `CountVideoObjects` 使用真实 CountVid 依赖，而不是 placeholder 假设
2. temporal filtering 能实际滤掉孤立误检
3. propagation 能形成稳定 raw tracks
4. unique aggregation 能把多个 raw tracks 合并为唯一实例
5. `instance_count` 语义等于 video-level unique instance count
6. 工具注册条件严格，不误暴露不可用工具
7. artifact 足够支持错误定位
8. 在至少一组 VSI-Bench / 本地视频样例上，相比“多帧 CountObjects + LLM 总结”有可解释的提升

---

## 12. 推荐开发顺序

为了减少返工，建议严格按下面顺序推进：

1. **Phase 0**
   - 收紧注册
   - 统一坐标
   - 修正文档状态
2. **Phase 1**
   - CountGD-Box 正式 adapter
3. **Phase 2**
   - temporal filtering 真正生效
4. **Phase 3**
   - SAM2 propagation 对齐
5. **Phase 4**
   - unique aggregation
6. **Phase 5**
   - artifact / report 完善
7. **Phase 6**
   - route 与 fallback 收口

不要先做：

- 花哨的 LLM prompt 微调
- 更复杂的 multi-step ReAct 视频 count 编排
- 隐式跨帧推理增强

因为 CountVid 核心问题不在 LLM，而在 **显式的视频实例流程**。

---

## 13. 结论

当前仓库已经完成了 CountVid 的：

- 工具入口
- 路由骨架
- payload 结构
- 测试骨架
- 调试落盘

但这还不是 CountVid 核心对齐完成。

真正的 CountVid 核心对齐，必须完成下面三件事：

1. **真实 CountGD-Box 候选生成对齐**
2. **真实 SAM 2.1 propagation 与 track identity**
3. **真实 unique instance aggregation**

因此，后续开发的重心不应继续放在：

- LLM 如何多问几轮
- ReAct 如何再多看几帧

而应明确转向：

- **把视频级唯一实例计数做成代码层的显式 pipeline**

这才是 CountVid 对齐能够真正提升 `video + object_counting` 表现的关键。
