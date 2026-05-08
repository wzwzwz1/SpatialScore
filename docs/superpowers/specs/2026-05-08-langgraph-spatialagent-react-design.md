# LangGraph SpatialAgent ReAct 重构设计

> 面向独立研究型项目的设计文档，不绑定 Spring Boot / React 产品架构。

## 1. 背景

当前 `version_0/SpatialAgent` 公开代码提供了以下基础能力：

- AutoGen 风格的 `UserAgent` 对话壳
- JSON action 解析器
- 基于 action name 的执行器
- 若干空间感知模型目录，如 `DepthAnythingV2`、`sam2`、`RAFT`、`VGGT`、`OrientAnything`

但公开代码并未提供论文中完整的 SpatialAgent 推理链路，尤其缺失：

- 完整的 SpatialAgent inference code
- 多智能体编排逻辑
- Plan-Execute / ReAct 的正式运行入口
- 完整的 tool registry
- 工具包装层与统一输入输出协议

论文与项目页给出的目标是：SpatialAgent 作为一个带有 specialized spatial perception tools 的系统，通过工具增强的方式提升空间推理能力。考虑到当前公开仓库的完整度，以及实际可运行性的优先级，本设计聚焦于 **ReAct 范式** 的 LangGraph 重构版本，不考虑 Plan-Execute。

## 2. 设计目标

本次重构的目标如下：

1. 使用 LangGraph 重建 SpatialAgent 的主推理图。
2. 仅实现 ReAct 推理范式，不实现 PE。
3. 保持“问题驱动、动态调用工具”的论文思路。
4. 对已知工具提供标准化封装接口。
5. 对论文提及但公开实现不足的工具，允许先以占位 tool 形式注册。
6. 让系统具备后续继续补齐论文工具、替换底层模型、接入不同核心 VLM 的能力。

非目标：

- 不做统一场景图 / world model 先验建模。
- 不做 Web 产品集成设计。
- 不要求第一阶段完整复现论文全部实验结果。
- 不要求第一阶段严格保留 AutoGen 对话式编排。

## 3. 方案选择

本次重构对比了三种方案：

### 方案 1：多 Agent 复刻版

使用多个 planner / reasoner / tool agents 互发消息，形式上更接近论文所说的 “multi-agent system”。

优点：

- 更接近论文叙事
- 便于未来扩展为 PE + ReAct 双范式

缺点：

- 当前公开代码无法支撑完整复刻
- 状态流难调试
- 首版运行成本高

### 方案 2：单图 ReAct 主图 + 工具节点

使用一个核心 reasoning node 作为 agent brain，输出结构化 ReAct action，由 LangGraph 负责工具路由、执行和 observation 回灌。

优点：

- 与当前公开代码最贴近
- 易于调试和验证
- 最有利于先跑通

缺点：

- 形式上不是多 agent 对话协作

### 方案 3：混合版

主图采用单 agent ReAct，部分复杂工具封装为子 agent。

优点：

- 保留扩展性

缺点：

- 第一阶段收益不高
- 架构复杂度明显提高

### 推荐方案

采用 **方案 2：单图 ReAct 主图 + 工具节点**。

原因：

- 符合论文的核心方法论：工具增强的迭代推理
- 保留动态工具调用特征
- 最适合当前仓库的开放程度
- 最容易逐步补齐未公开工具

## 4. 总体架构

系统由一个 LangGraph 主图驱动，其内部不做多 agent 对话，而是通过结构化 ReAct 循环完成推理。

高层流程如下：

1. 初始化输入状态与 tool schema
2. 核心推理节点生成：
   - 简短 thought
   - 一个 tool action
   - 或 finish answer
3. LangGraph 按 action 路由到工具节点
4. 工具节点执行底层模型包装器
5. 结果统一转换为 observation
6. observation 追加回 state 与消息历史
7. 再次进入推理节点
8. 达成足够证据后输出最终答案

这一设计保留论文思路中的“按需感知、按需计算、按需推理”，而不是先构造完整场景结构再统一求解。

## 5. 输入输出设计

### 5.1 输入

LangGraph 主图入口输入统一定义为：

```python
{
    "task_id": str,
    "question": str,
    "question_type": "multi_choice" | "judgment" | "open_ended",
    "input_modality": "single_image" | "multi_image" | "video",
    "image_paths": list[str],
    "options": list[str] | None,
    "metadata": dict | None
}
```

说明：

- `image_paths` 支持单图、多图、视频抽帧
- `options` 对多选题可选
- `metadata` 允许未来透传数据集、帧号、camera id、hint 等上下文

### 5.2 输出

主图输出统一定义为：

```python
{
    "final_answer": str | None,
    "reasoning_trace": list[dict],
    "tool_calls": list[dict],
    "tool_observations": list[dict],
    "status": "success" | "failed" | "max_steps" | "tool_unavailable",
    "error": str | None
}
```

说明：

- `reasoning_trace` 主要用于调试、复现实验、错误分析
- `final_answer` 由 `finalize_node` 做统一规范
- `status` 用于区分失败原因

## 6. LangGraph 状态设计

建议状态结构如下：

```python
SpatialAgentState = {
    "task_id": str,
    "question": str,
    "question_type": str,
    "input_modality": str,
    "image_paths": list[str],
    "options": list[str] | None,

    "step_count": int,
    "max_steps": int,
    "repair_count": int,
    "max_repairs": int,
    "tool_fail_count": int,
    "max_tool_fails": int,

    "messages": list,
    "scratchpad": list[dict],
    "last_thought": str | None,

    "selected_tool": str | None,
    "selected_args": dict | None,
    "last_tool_result": dict | None,

    "available_tools": list[str],
    "tool_errors": list[dict],

    "final_answer": str | None,
    "status": str,
    "error": str | None
}
```

设计原则：

- `messages` 服务于 LLM 上下文
- `scratchpad` 服务于可观测性与回放
- `selected_tool` / `selected_args` 作为路由中间态
- `tool_errors` 用于累积错误并决定是否降级

推荐阈值：

- `max_steps = 8`
- `max_repairs = 2`
- `max_tool_fails = 3`

## 7. 节点设计

### 7.1 `init_state`

职责：

- 规范化输入
- 注入系统 prompt
- 注入当前可用工具清单
- 初始化步数与空历史

输入：

- 原始 task input

输出：

- 完整 `SpatialAgentState`

### 7.2 `reason_node`

职责：

- 执行核心 ReAct 推理
- 输出一个结构化结果：
  - thought
  - action
  - 或 finish

要求：

- 一轮最多选一个工具
- 若已有足够证据，可直接 finish
- 不允许臆造工具输出

### 7.3 `route_node`

职责：

- 判断 `reason_node` 输出属于哪类分支：
  - finish
  - valid tool call
  - invalid format

### 7.4 `tool_node`

职责：

- 调用 tool registry
- 执行真实工具或占位工具
- 返回标准化 `ToolResult`

### 7.5 `observe_node`

职责：

- 将工具结果转换为 observation
- 回写 `messages`、`scratchpad`、`tool_calls`
- 更新步数

### 7.6 `repair_node`

职责：

- 修复以下错误：
  - 非法 JSON
  - 缺失 action name
  - arguments 不匹配
  - 工具名不在 registry

策略：

- 使用简短纠错 prompt
- 不改变问题本身，只纠正输出格式和动作合法性

### 7.7 `finalize_node`

职责：

- 规范最终答案格式
- 对不同题型执行输出约束

示例：

- multi-choice 仅输出选项
- numeric 保留单位
- judgment 输出 yes/no

### 7.8 `fail_node`

职责：

- 在以下条件触发：
  - 超过最大步数
  - 连续修复失败
  - 连续工具失败
  - 不可恢复异常

输出：

- `status = failed | max_steps`
- 错误信息
- 可选保底答案

## 8. 路由设计

主图路由如下：

```text
init_state
  -> reason_node
  -> route_node

route_node
  -> finalize_node      (finish)
  -> tool_node          (valid_tool_call)
  -> repair_node        (invalid_format)
  -> fail_node          (fatal_error / max_steps)

tool_node
  -> observe_node
  -> reason_node

repair_node
  -> reason_node
```

终止条件：

- `final_answer` 已生成
- `step_count >= max_steps`
- `repair_count >= max_repairs`
- `tool_fail_count >= max_tool_fails`

## 9. 工具体系设计

### 9.1 设计原则

工具层必须与 LangGraph 主图解耦。主图只知道：

- 工具名
- 参数 schema
- 返回值 schema

底层模型替换不应影响主图逻辑。

### 9.2 工具统一接口

建议统一为：

```python
class BaseSpatialTool(Protocol):
    name: str
    description: str
    args_schema: dict
    returns_schema: dict

    def invoke(self, **kwargs) -> ToolResult:
        ...
```

返回值：

```python
ToolResult = {
    "status": "success" | "error" | "unavailable",
    "tool_name": str,
    "payload": dict,
    "artifacts": list[str],
    "error": str | None
}
```

### 9.3 第一阶段优先工具

这些工具在当前公开代码中有比较明确的模型基础，适合作为第一阶段优先接入对象：

1. `EstimateObjectDepth`
2. `GetObjectMask`
3. `EstimateOpticalFlow`
4. `GetCameraParametersVGGT`
5. `GetObjectOrientation`
6. `EstimateHomographyMatrix`

### 9.4 第二阶段工具

1. `LocalizeObjects`
2. `SelfReasoning`

其中 `SelfReasoning` 可以实现为 pseudo-tool，也可以保留在 `reason_node` 内部，不一定需要真实 registry entry。

### 9.5 占位工具策略

对于论文中提及但当前仓库无法明确实现细节的工具：

- 保留工具名和 schema
- 使用 placeholder tool 返回：

```python
{
    "status": "unavailable",
    "tool_name": "...",
    "payload": {},
    "artifacts": [],
    "error": "Tool is declared but not implemented in the current release."
}
```

这样既能对齐论文方法，又不会阻塞主图运行。

## 10. Prompt 设计

### 10.1 System Prompt

核心 system prompt 应约束模型只做两种事：

1. 调用一个工具
2. 结束并给答案

关键要求：

- 每一步只能调用一个工具
- 只能使用 `AVAILABLE_TOOLS`
- 工具失败时应调整策略
- 不允许虚构 observation
- thought 保持简短、操作性强

### 10.2 输出格式

统一要求输出 JSON：

```json
{
  "thought": "...",
  "action": {
    "name": "ToolName",
    "arguments": {}
  },
  "finish": null
}
```

或：

```json
{
  "thought": "...",
  "action": null,
  "finish": {
    "answer": "..."
  }
}
```

### 10.3 Repair Prompt

Repair prompt 只解决输出错误，不重新解释任务本身。修复范围包括：

- JSON 语法错误
- 字段缺失
- 工具名非法
- 参数不匹配

## 11. 错误处理与降级

错误处理原则：

- **工具失败不是进程失败，而是 observation**
- **不可恢复错误才终止主图**

处理规则：

1. JSON 输出非法 -> `repair_node`
2. 工具不存在 -> `unavailable observation`
3. 工具参数错误 -> `error observation`
4. 工具执行异常 -> `tool_fail_count + 1`
5. 连续失败过多 -> `fail_node`
6. 超步数 -> `fail_node`

## 12. 推荐目录结构

```text
spatial_agent/
  graph/
    state.py
    builder.py
    routes.py
    nodes/
      init_node.py
      reason_node.py
      route_node.py
      tool_node.py
      observe_node.py
      repair_node.py
      finalize_node.py
      fail_node.py

  prompts/
    react_system_prompt.py
    repair_prompt.py
    output_schema.py

  tools/
    base.py
    registry.py
    depth_tool.py
    mask_tool.py
    optical_flow_tool.py
    camera_tool.py
    orientation_tool.py
    homography_tool.py
    placeholders/
      empty_tool.py

  adapters/
    depth_anything_adapter.py
    sam2_adapter.py
    raft_adapter.py
    vggt_adapter.py
    orientanything_adapter.py

  runtime/
    config.py
    logging.py
    artifacts.py
```

## 13. 与当前公开代码的映射

### 可复用部分

- `prompt` 的“单步动作”思想
- `parser` 的 JSON action 约束思路
- `executor` 的 observation 回灌思路
- `DepthAnythingV2`、`sam2`、`RAFT`、`VGGT`、`OrientAnything` 目录中的底层模型代码

### 不建议直接沿用部分

- AutoGen 对话编排壳
- 当前不完整的 action registry 依赖
- 非结构化 observation 拼接方式

## 14. 第一阶段落地顺序

建议分三轮推进：

### 第一轮：主图跑通

- 搭建 LangGraph 主图
- 实现 `init/reason/route/finalize/fail`
- 暂时不接真实工具，只接 placeholder tools

### 第二轮：核心工具接入

- 接入 `EstimateObjectDepth`
- 接入 `GetObjectMask`
- 统一 `ToolResult`
- 实现 `repair_node`

### 第三轮：扩展工具与论文对齐

- 接入 RAFT / VGGT / SAM2 / OrientAnything
- 对齐论文工具表
- 增加缺失工具占位项
- 加入 trace 保存与评测脚本

## 15. 风险与已知差距

### 15.1 论文与公开代码存在发布落差

论文与项目页描述的是完整 SpatialAgent，而当前公开仓库 `version_0` 明确说明：

- 主要发布了 evaluation code
- SpatialAgent 仅有 basic codes
- inference code 尚未完整放出

因此本设计是：

- 遵循论文方法论
- 面向当前公开仓库做工程重建
- 允许未知工具先空置

### 15.2 底层模型权重与路径问题

当前公开代码存在：

- checkpoint 未随仓库完整提供
- 个别路径硬编码
- 个别模型依赖外部下载资源

因此重构后必须把所有模型依赖外置到统一配置层。

## 16. 结论

本设计建议采用 **LangGraph 单图 ReAct 主图 + 标准化工具节点** 的重构路径，以最小偏离论文核心思路的方式实现一个可运行、可扩展、可逐步补齐的 SpatialAgent。

该方案的核心价值在于：

- 保留论文中的工具增强式空间推理思路
- 避免被未公开的多 agent 推理代码阻塞
- 为后续接入更多空间感知工具与实验复现打下清晰结构基础

后续如需进入实现阶段，应在此设计基础上继续生成：

- LangGraph 节点函数签名
- state schema 类型定义
- tool registry 接口定义
- placeholder tool 清单
- 最小可运行实现计划
