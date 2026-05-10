# VSI-Bench 结果分析与可视化

这份文档说明如何把 `lmms_eval` 的 `VSI-Bench` 输出，和 `SpatialAgent` 的 trace 合并起来做分析。

适用前提：

- `lmms_eval` 开启了 `--log_samples`
- `SpatialAgent` 运行时保留了 trace
- 你已经能正常跑出 `VSI-Bench` 结果

## 1. 输入来源

分析器需要两路输入：

1. `lmms_eval` 的样本日志
2. `SpatialAgent` 的 trace 目录

典型路径如下：

- 样本日志目录：
  - `/disk/wangzhe/thinking-in-space/logs/vsibench_smoke`
  - 或 `/disk/wangzhe/thinking-in-space/logs/vsibench`
- trace 目录：
  - `/tmp/spatial_agent_runs`

其中 `lmms_eval` 的 `log_samples` 目录里通常会包含：

- `samples_vsibench_*.jsonl`
  - 每条样本的题目、目标答案、模型输出、指标字段
- 有些运行方式还会额外生成一个 JSON bundle

`SpatialAgent` trace 目录里每个样本通常对应一个：

- `<task_id>.json`

里面包含：

- `tool_calls`
- `tool_observations`
- `reasoning_trace`
- `final_answer`
- `status`
- `error`

## 2. 一条命令生成分析报告

```bash
python -m spatial_agent.analysis \
  --samples-path /disk/wangzhe/thinking-in-space/logs/vsibench_smoke \
  --trace-dir /tmp/spatial_agent_runs \
  --output-dir /disk/wangzhe/SpatialScore/analysis/vsibench_smoke
```

如果你跑的是正式评测，把路径替换成正式日志目录即可：

```bash
python -m spatial_agent.analysis \
  --samples-path /disk/wangzhe/thinking-in-space/logs/vsibench \
  --trace-dir /tmp/spatial_agent_runs \
  --output-dir /disk/wangzhe/SpatialScore/analysis/vsibench_full
```

可选参数：

- `--task-name vsibench`
- `--split test`
- `--max-cases 24`

## 3. 会产出什么

分析器会在 `--output-dir` 下生成这些文件：

- `summary.json`
  - 汇总统计，适合程序化读取
- `samples.csv`
  - 每题一行，适合筛选、排序、手工查看
- `report.md`
  - 适合终端、Markdown 查看器或 Codex 内查看
- `report.html`
  - 适合浏览器打开
- `charts/`
  - 多张 PNG 图表
- `artifacts/`
  - 从 trace 里收集并物化的关键可视化产物

如果某些 tool 在 trace 中留下了 artifact，报告里会尽量带上这些图片，例如：

- 分割 mask overlay
- 深度图
- 光流可视化
- 运动轨迹图

## 4. 当前会生成哪些图

默认包含：

- `question_type_counts.png`
  - 各题型样本数量
- `question_type_scores.png`
  - 各题型主指标均值
- `tool_call_counts.png`
  - tool 调用次数
- `tool_status_stacked.png`
  - tool 成功 / unavailable / error 分布
- `reasoning_steps_hist.png`
  - 推理步数分布
- `question_type_tool_heatmap.png`
  - 题型和 tool 的对应关系

## 5. 怎么读这些结果

建议按下面顺序看：

### 第一步：看整体是不是“真在用工具”

先看 `summary.json` 里的：

- `trace_coverage`
- `average_tool_calls`
- `success_rate`

如果：

- `trace_coverage` 很低
  - 说明大量样本没正确写 trace，先查运行链路
- `average_tool_calls` 接近 `0`
  - 说明 agent 主要在裸答
- `success_rate` 很低
  - 说明 agent 自身经常在失败或降级

### 第二步：看哪些题型掉分最严重

打开：

- `charts/question_type_scores.png`

重点看：

- `object_counting`
- `object_abs_distance`
- `object_size_estimation`
- `room_size_estimation`
- `object_rel_distance`
- `object_rel_direction`
- `route_planning`
- `obj_appearance_order`

这一步回答的是：

- 哪些能力最弱
- 优先该补哪条工具链

### 第三步：看是不是工具没起作用

打开：

- `charts/tool_status_stacked.png`

如果你看到很多：

- `unavailable`
  - 多半是 `tool_config`、checkpoint、依赖没接好
- `error`
  - 多半是工具执行异常或输入格式不匹配

如果 `calls` 很多但 `success` 不高，优先排查工具本身。  
如果 `calls` 很少，优先排查 prompt / tool selection。

### 第四步：看场景理解的中间结果

打开：

- `report.html`

重点看每条 case 下的 artifact 图片。  
这些图片最适合判断：

- 分割是否把目标物体找对了
- 深度估计是否有明显偏差
- 光流是否抓到了运动方向
- 运动轨迹是否合理

这一步最接近你说的“看 agent 对场景的分割和建模”。

## 6. 推荐的排查顺序

如果结果差，建议按这个顺序排：

1. `summary.json`
   - 确认 trace 覆盖率和工具调用不是空的
2. `tool_status_stacked.png`
   - 判断是不是工具没接上
3. `question_type_scores.png`
   - 找最弱题型
4. `report.html`
   - 看失败案例的中间视觉产物
5. `samples.csv`
   - 筛选某类题型或某个 tool 的样本

## 7. 一条典型工作流

1. 跑一次 `--limit 20`
2. 生成分析报告
3. 看 `tool_status_stacked.png`
4. 看 `question_type_scores.png`
5. 在 `report.html` 里抽 5 到 10 条失败样本看 artifact
6. 再决定是改：
   - `tool_config`
   - tool 实现
   - agent prompt
   - tool selection
   - LLM backend

## 8. 建议先跑的小规模分析

第一次建议这样：

```bash
python -m lmms_eval \
  --model spatial_agent_api \
  --model_args llm_backend=openai_compatible,model_name=gpt-4o-mini,video_num_frames=16,artifact_dir=/tmp/spatial_agent_runs,tool_config_path=/disk/wangzhe/SpatialScore/configs/tool_config.server.json \
  --tasks vsibench \
  --batch_size 1 \
  --limit 20 \
  --log_samples \
  --output_path /disk/wangzhe/thinking-in-space/logs/vsibench_debug
```

然后：

```bash
python -m spatial_agent.analysis \
  --samples-path /disk/wangzhe/thinking-in-space/logs/vsibench_debug \
  --trace-dir /tmp/spatial_agent_runs \
  --output-dir /disk/wangzhe/SpatialScore/analysis/vsibench_debug
```

这样最容易快速看出：

- trace 是否正常
- tool 是否真实执行
- 视觉中间产物是否靠谱
