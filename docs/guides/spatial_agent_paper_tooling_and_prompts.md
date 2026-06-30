# SpatialAgent 论文原版工具设计与提示词设计整理

## 1. 文档目的

这份文档用于把原论文 **SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence** 中与 `SpatialAgent` 直接相关的两部分内容整理出来，方便后续与当前仓库实现做对齐：

1. 工具设计（toolbox / tool specifications）
2. 提示词设计（Plan-Execute / ReAct prompt templates）

本文档以原论文附录中的描述为准，尽量保留原版英文提示词与工具接口说明，同时补充中文解释，方便工程侧查阅。

说明：

- 工具规范和 prompt 模板主要来自论文附录页 `31-44`
- 个别引号、连字符、特殊符号可能带有 PDF 提取或 OCR 痕迹；若出现这类小问题，应以论文 PDF 原页为最终依据

## 2. 论文来源

- 论文文件：[/Users/wz/Zotero/storage/8M55G2J4/Wu 等 - 2026 - SpatialScore Towards Comprehensive Evaluation for Spatial Intelligence.pdf](/Users/wz/Zotero/storage/8M55G2J4/Wu%20%E7%AD%89%20-%202026%20-%20SpatialScore%20Towards%20Comprehensive%20Evaluation%20for%20Spatial%20Intelligence.pdf)
- 关键章节：
  - Sec. 3.3 SpatialAgent
  - Sec. 3.4 Toolbox
  - Appendix B.3 SpatialAgent Development
  - Appendix B.4 Toolbox Specifications

## 3. 总体架构摘录

### 3.1 原论文对 SpatialAgent 的定位

论文将 `SpatialAgent` 描述为一种 **training-free** 的 agent-based spatial reasoning 框架：

- 以开源或闭源 MLLM 作为 agent core
- 通过精心设计的 prompts 引导 agent 执行空间推理
- 同时支持两种范式：
  - **Plan-Execute (PE)**
  - **ReAct**
- 结合一组空间感知工具来弥补纯语言视觉模型在 spatial intelligence 上的不足

### 3.2 原论文对工具箱的分组

论文正文（Sec. 3.4）将工具箱分成四类：

1. **General Perception**
2. **Motion & Transformation**
3. **Pose & Geometry**
4. **Auxiliary Tools**

论文正文里写的是 **12 specialized spatial perception tools**；但附录 `B.4 Toolbox Specifications` 中实际展示的 callable modules 比 12 个更多，因为还额外列出了：

- `Terminate`
- `SelfThinking`
- 某些组合式工具（例如 `EstimateObjectGeometryProperties`）
- 某些能力拆成了多个可调用函数（例如 `EstimateRegionDepth` / `EstimateObjectDepth`）

因此，做工程对齐时建议区分两层：

- **正文口径**：12 个 specialized spatial perception tools
- **附录实现口径**：一个更完整的 callable toolbox，包含 specialized tools 与 auxiliary tools

## 4. 论文原版工具设计

> 说明：以下内容按论文 `Appendix B.4` 整理。每个工具都尽量保留原版的 `description`、`args_spec`、`rets_spec` 和 `examples`。

---

## 4.1 General Perception

### 4.1.1 `LocalizeObjects`

**Backend（论文描述）**：Rex-Omni

**原版描述**

```text
description = """
Localize specific objects in an image.
Returns bounding boxes for target categories, optionally visualizing them.
"""
args_spec = {
"image": "The image to analyze.",
"objects": "A list of object categories to detect."
}
rets_spec = {"regions": "List of detected regions with label, bbox"}
examples = [{
"name": "LocalizeObjects",
"arguments": {"image": "image-0", "objects": ["dog", "cat"]}
}]
```

**中文理解**

- 功能：在图像中定位指定类别的物体
- 输入：单张图像 + 类别列表
- 输出：目标类别的 2D 检测框

---

### 4.1.2 `CountObjects`

**Backend（论文描述）**：Rex-Omni

**原版描述**

```text
description = """
Count target objects in an image. Returns the coordinates of each detected target as
points.
"""
args_spec = {
"image": "The image to analyze.",
"objects": "List of object categories to count."
}
rets_spec = {"points": "Dictionary {category: [points...]}, points in normalized
coordinates.}
examples = [{"name": "CountObjects", "arguments": {"image": "image-0", "objects":
["bed"]}]
```

**中文理解**

- 功能：统计图像中指定类别的物体数量
- 输出不是单纯标量，而是 **每个实例对应的点**
- 论文在 ReAct prompt 中明确写了：**counting-related problems 要优先使用这个工具**

---

### 4.1.3 `GetObjectMask`

**Backend（论文描述）**：Rex-Omni + SAM2

**原版描述**

```text
description = """
Generate pixel-level segmentation masks for specified objects.
Returns mask area ratios and bounding boxes for each detected object.
Suitable for analyzing object shapes, sizes, and coverage.
"""
args_spec = {
"image": "Image file to process.",
"objects": "List of object descriptions to localize and segment."
}
rets_spec = {
"results": "List of dicts with mask area ratio, bounding box, and optional error:
[{’object’: str, ’mask_area’: float, ’bbox’: [left, top, right, bottom], ’error’: str
or None}]"
}
examples = [
{"name": "GetObjectMask", "arguments": {"image": "image-0", "objects": ["coffee
mug", "microwave"]}}
]
```

**中文理解**

- 功能：对指定目标做像素级实例分割
- 输出：每个实例的 mask 面积比例、bbox 和错误信息
- 适合：物体形状、遮挡、区域覆盖、相对大小等分析

---

### 4.1.4 `Detect3DObjects`

**Backend（论文描述）**：DetAny3D

**原版描述**

```text
description = """
Detect specific objects in an image and estimate their 3D bounding boxes.
Returns 3D bounding box parameters in the following format:
x, y, z -> object center in camera coordinates (meters);
width, height, length -> physical size (width, height, length) in meters;
yaw -> heading angle around vertical axis (radians).
"""
args_spec = {
"image": "Path to the input image."
"objects": "List of object categories to detect (or a single string)."
}
rets_spec = {
"objects": "List of dicts with {label: str, bbox_3d: {x:float, y:float, z:float,
width:float, height:float, length:float, yaw:float}}"
}
examples = [
{"name": "Detect3DObjects", "arguments": {"image": ["image-1"], "objects": ["dog",
"rabbit"]}
]
```

**中文理解**

- 功能：目标检测 + 3D bounding box 估计
- 输出：中心点、长宽高、yaw 等 3D 参数

---

## 4.2 Motion & Transformation

### 4.2.1 `EstimateOpticalFlow`

**Backend（论文描述）**：RAFT

**原版描述**

```text
description = """
Estimate optical flow between two images to measure motion in pixels.
Returns average displacement in horizontal (x) and vertical (y) directions.
First image is earlier in time; second is later.
- mean_flow_x > 0: objects move left / camera moves right.
- mean_flow_x < 0: objects move right / camera moves left.
- mean_flow_y > 0: objects move up / camera moves down.
- mean_flow_y < 0: objects move down / camera moves up.
Useful for analyzing camera motion, object movement, and 3D spatial reasoning.
"""
args_spec = {
"image": "A list of exactly two image paths to compute optical flow between. First
image is earlier in time."
}
rets_spec = {
"output": "Dictionary containing ’mean_flow_x’ (average horizontal pixel
displacement) and ’mean_flow_y’ (average vertical pixel displacement)."
}
examples = [{"name": "EstimateOpticalFlow", "arguments": {"image": ["image-1",
"image-3"]}]
```

**中文理解**

- 功能：估计两帧之间的平均光流
- 用途：相机运动、目标运动、视频时序推理

---

### 4.2.2 `MatchImagesSIFT`

**Backend（论文描述）**：OpenCV SIFT

**原版描述**

```text
description = """
Match keypoints between two images using SIFT.
Detects distinctive features and returns matched coordinate pairs for tasks like
alignment or recognition.
"""
args_spec = {
"image": "List of two image paths.",
"num_keypoints": "Max keypoints per image (default: 1200).",
"ratio_th": "Ratio test threshold for matching (default: 0.75)."
}
rets_spec = {
"matches": "List of matched coordinate pairs: [[x1, y1], [x2, y2]].",
"num_matches": "Total number of matches found."
}
examples = [
{"name": "MatchImagesSIFT", "arguments": {"image": ["image-0", "image-1"],
"num_keypoints": 1200, "ratio_th": 0.75}}
]
```

**中文理解**

- 功能：图像间关键点匹配
- 用途：对齐、几何估计、单应矩阵求解前置步骤

---

### 4.2.3 `EstimateHomographyMatrix`

**Backend（论文描述）**：OpenCV + SIFT + RANSAC

**原版描述**

```text
description = """
Compute a 3*3 homography matrix between two images using SIFT features and RANSAC.
Useful for alignment, perspective correction, and planar transformations.
"""
args_spec = {
"image": "List of two image paths.",
"num_keypoints": "Max keypoints per image (default: 1200).",
"ratio_th": "Ratio test threshold (default: 0.75).",
"ransac_reproj_threshold": "Max reprojection error in RANSAC (default: 5.0)."
}
rets_spec = {
"homography_matrix": "3*3 matrix mapping points from first image to second.",
"inliers_count": "Number of inlier matches used.",
"total_matches": "Total matches found.",
"status": "Success or failure."
}
examples = [
{"name": "EstimateHomographyMatrix", "arguments": {"image": ["image-0", "image-1"],
"num_keypoints": 1200, "ratio_th": 0.75, "ransac_reproj_threshold": 5.0}}
]
```

**中文理解**

- 功能：估计两张图间的 3x3 单应矩阵
- 用途：透视变换、平面配准、几何推理

---

## 4.3 Pose & Geometry

### 4.3.1 `GetCameraParametersVGGT`

**Backend（论文描述）**：VGGT

**原版描述**

```text
description = """
Extract camera extrinsic (3*4, relative to first image) and intrinsic (3*3) parameters
from images using VGGT.
Useful for 3D reconstruction, novel view synthesis, and geometric analysis.
"""
args_spec = {"image": "List of image paths (at least one)."}
rets_spec = {
"output": "List of dicts with image_index (int), extrinsic (3*4 matrix), and
intrinsic (3*3 matrix)."
}
examples = [
{"name": "GetCameraParametersVGGT", "arguments": {"image": ["image-0", "image-1"]}}
]
```

**中文理解**

- 功能：从单帧或多帧估计相机内参、外参
- 用途：3D reconstruction、camera reasoning、view reasoning

---

### 4.3.2 `EstimateObjectGeometryProperties`

**Backend（论文描述）**：SAM2 + Depth-Anything-V2 + VGGT

**原版描述**

```text
description = """
Analyze objects in an image to obtain bounding boxes, mask areas, depth (m), and
camera parameters.
Camera parameters include intrinsic (3*3) and extrinsic (3*4) matrices for 3D geometry
tasks.
"""
args_spec = {
"image": "Image file path to analyze.",
"object_descs": "List of object descriptions (e.g., [’dog’, ’cat’])."
}
rets_spec = {
"results": "List of dicts with object, bbox, mask_area, depth (m), and optional
error.",
"camera_parameters": "Dict with intrinsic (3*3) and extrinsic (3*4) matrices."
}
examples = [
{"name": "EstimateObjectGeometryProperties", "arguments": {"image": "image-0",
"object_descs": ["coffee cup", "keyboard"]}}
]
```

**中文理解**

- 功能：组合式几何分析工具
- 一次性产出：bbox、mask area、depth、camera params

---

### 4.3.3 `EstimateRegionDepth`

**Backend（论文描述）**：Depth-Anything-V2

**原版描述**

```text
description = """
Estimate metric depth (in meters) of specified regions in an image.
Supports indoor (0-20m) and outdoor (0-80m) scenes.
Works with single or multiple bounding boxes in pixel coordinates.
Depth is distance from camera to object, not between objects or object size.
"""
args_spec = {
"image": "Image to analyze.",
"bboxes": "Bounding box or list of boxes in pixel coordinates: [left, top, right,
bottom] or [[...], ...].",
"indoor_or_outdoor": "Scene type (’indoor’ or ’outdoor’).",
"mode": "Depth calculation: ’mean’ (average) or ’center’ (center point). Default:
’mean’."
}
rets_spec = {
"depths": "List of dicts with bbox, depth (m), and optional error: [’bbox’: list,
’depth’: float, ’error’: str or None]",
"unit": "Always ’meters’."
}
examples = [
{"name": "EstimateRegionDepth", "arguments": {"image": "image-0", "bboxes": [100,
50, 200, 150], [150, 100, 250, 200] "indoor_or_outdoor": "indoor"}}
]
```

**中文理解**

- 功能：给定一个或多个区域框，估计该区域的 metric depth
- 和 `EstimateObjectDepth` 的区别：
  - `EstimateRegionDepth` 输入 bbox
  - `EstimateObjectDepth` 输入 object description

---

### 4.3.4 `EstimateObjectDepth`

**Backend（论文描述）**：Rex-Omni + Depth-Anything-V2

**原版描述**

```text
description = """
Estimate object depth (in meters) from an image.
Supports indoor (0-20m) and outdoor (0-80m) scenes.
Depth indicates distance from camera to object, not between objects or object size.
"""
args_spec = {
"image": "Image to analyze.",
"objects": "List of object descriptions to measure distance to (e.g., [’dog’,
’cat’]).",
"indoor_or_outdoor": "Scene type (’indoor’ or ’outdoor’)."
}
rets_spec = {
"results": "List of dicts with object description, depth (m), and optional error:
[’object’: str, ’depth’: float, ’error’: str or None]"
}
examples = [
{"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "objects": ["the
red car", "dog"], "indoor_or_outdoor": "outdoor"}}
]
```

**中文理解**

- 功能：按物体语义描述估计其深度

---

### 4.3.5 `GetObjectOrientation`

**Backend（论文描述）**：OrientAnything

**原版描述**

```text
description = """
Estimate 3D orientation of objects in an image using Orient-Anything.
Measures:
- Azimuth: Horizontal rotation (0-360°clockwise)
- Polar: Vertical inclination (0-180°)
- Rotation: In-plane rotation (-180°to +180°)
- Confidence: Reliability score
Useful for 3D understanding, pose estimation, and spatial reasoning.
"""
args_spec = {
"image": "Image to analyze.",
"objects": "Object description(s) to analyze; string or list."
}
rets_spec = {
"results": "List of dicts with object orientation data: [{’object’: str,
’angle_data’: {’azimuth’: float, ’polar’: float, ’rotation’: float, ’confidence’:
float}, ’error’: str or None}]"
}
examples = [
{"name": "GetObjectOrientation", "arguments": {"image": "image-0", "objects": "a
red car"}}
]
```

**中文理解**

- 功能：估计物体朝向
- 输出：azimuth / polar / rotation / confidence

---

### 4.3.6 `Get3DDistance`

**Backend（论文描述）**：MapAnything

**原版描述**

```text
description = """
Calculates the absolute 3D spatial distance (in meters) between two pixel points (x,
y) in an image.
Note: this tool should be used in outdoor scenes.
Returns the calculated distance (in meters).
"""
args_spec = {
"image": "Path to the input image.",
"point_1": "List of [x, y] pixel coordinates for the first point."
"point_2": "List of [x, y] pixel coordinates for the second point."
}
rets_spec = {
"distance_meters": "The calculated 3D distance (float, in meters)."
}
examples = [
{"name": "Get3DDistance", "arguments": {"image": "image-0", "point_1": [100, 100],
"point_2": [1000, 1000]}}
]
```

**中文理解**

- 功能：计算图中两个像素点在 3D 空间中的真实距离
- 论文中明确提示：更适合 outdoor scene

---

## 4.4 Auxiliary Tools

### 4.4.1 `Terminate`

**原版描述**

```text
description = """
Use this function ONLY when you are completely confident in your final answer.
For multiple-choice questions: Specify the letter of the correct option.
For numerical answers: Include both the specific value and appropriate unit of
measurement (e.g., meter or centimeter).
For yes/no questions: Clearly state ’Yes’ or ’No’.
DO NOT call this function if you are uncertain or need to perform additional analysis.
Double-check your answer before terminating!
"""
args_spec = {
"answer": "The final answer with proper formatting. For multiple choice: include
letter (e.g., ’A. explanation’ or ’(B)’). For numerical answers: include units (e.g.,
’3.25 meters’)."
}
rets_spec = {
"answer": "The final answer that will be submitted."
}
examples = [
{"name": "Terminate", "arguments": {"answer": "A. Yes."}},
{"name": "Terminate", "arguments": {"answer": "(B)."}},
{"name": "Terminate", "arguments": {"answer": "B. 3.25 meters."}},
{"name": "Terminate", "arguments": {"answer": "(A) 2 inches."}},
{"name": "Terminate", "arguments": {"answer": "47.3 centimeters."}},
{"name": "Terminate", "arguments": {"answer": "38.2 degrees."}}
]
```

**中文理解**

- 功能：显式结束推理并返回最终答案
- 工程意义：不是普通函数，而是 workflow completion action

---

### 4.4.2 `SelfThinking`

**原版描述**

```text
description = """
Modes:
1. Text-only: Provide ’query’ for pure language tasks.
2. Vision+Language: Provide ’images’ + ’query’ for visual analysis.
Suitable for: Scene understanding, OCR, object/color recognition, classification, and
concept-level Q&A.
"""
args_spec = {
"query": "Text question or instruction (REQUIRED).",
"image": "List of image paths. If omitted, the model performs text-only reasoning.",
}
rets_spec = {"response": "Model’s response string."}
examples = [
{"name": "SelfThinking", "arguments": {"query": "Summarize the image content.",
"image": "image-0"}}
]
```

**中文理解**

- 功能：不调用专门感知工具，直接让 agent core 自己推理
- 论文在 ReAct prompt 里明确说了：**优先级较低，仅在其他工具无法解决时使用**

---

## 5. 论文原版提示词设计

> 说明：
> - 以下 prompt 按论文 `Appendix B.3 SpatialAgent Development` 整理
> - 为了方便后续工程对齐，保留英文原文
> - 文中变量占位如 `{action_details}`、`{tool_plan}`、`{ALL_OBSERVATION}` 也按论文原样保留

---

## 5.1 Plan-Execute Paradigm

### 5.1.1 Planner Prompt（原版）

```text
[BEGIN OF GOAL]
Generate a JSON-formatted tool-calling plan to solve spatial understanding questions about
given images or videos.
[END OF GOAL]
[BEGIN OF TOOLBOX]
{action_details}
[END OF TOOLBOX]
[BEGIN OF TASK INSTRUCTIONS]
Generate a step-by-step plan to answer the given spatial understanding question about
given images or videos.
***Use ONLY the tools listed in the TOOLBOX section (e.g., GetObjectOrientation,
EstimateObjectGeometryProperties, LocalizeObjects, EstimateObjectDepth)***
***Follow their argument specifications EXACTLY as defined in the toolbox, and try to give
detailed and comprehensive instructions in queries.***
Do NOT invent new tools or modify the existing tool interfaces.
The plan should strictly follow what these tools can and cannot do.
[END OF TASK INSTRUCTIONS]
[BEGIN OF FORMAT INSTRUCTIONS]
You are a helpful assistant tasked with solving spatial reasoning questions. Think step
by step.
***
Return a JSON list of tool calls inside "'json"' tags, where each call is a dictionary
with 'name' and 'arguments'.
The 'name' MUST match exactly one of the tool names provided in the toolbox.
The 'arguments' MUST include ALL required parameters for that specific tool with EXACT
parameter names.
The 'images' or 'image' argument must be specified as 'image-0', 'image-1', and 'image-2',
to refer to the provided images.
Do not answer the question directly, and do not use absolute paths for the 'images' or
'image' argument.
***
[END OF FORMAT INSTRUCTIONS]
[BEGIN OF EXAMPLES]
Example for 'Which is closer to the camera, the dog or the cat?':
"'json' [
{"name": "LocalizeObjects", "arguments": {"image": "image-0", "objects": ["dog",
"cat"]}},
{"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "objects":
["dog", "cat"], "indoor_or_outdoor": "outdoor"}},
]
"'
[END OF EXAMPLES]
***
Do not answer the question directly. Instead, think step-by-step, and output the
tool-calling plan inside "'json"' tags.
***
```

### 5.1.2 Executor Prompt（原版）

```text
[BEGIN OF GOAL]
Generate a Chain of Thought (CoT) reasoning process using the provided tool execution
results.
[END OF GOAL]
[BEGIN OF TASK INSTRUCTIONS]
You are a helpful assistant tasked with solving spatial reasoning questions. Analyze the
given question and tool execution results. Think step by step.
Generate a step-by-step reasoning process that shows how the tools contribute to solving
the question.
Use ONLY the tools and results provided, following their specifications STRICTLY.
The results of tool calls can sometimes be incomplete or incorrect, so please be critical
and decide how to make use of them.
If a tool failed, note the failure and proceed with your prior knowledge and reasoning.
Repeat for each tool result in order.
[END OF TASK INSTRUCTIONS]
[BEGIN OF FORMAT INSTRUCTIONS]
***
Output a CoT with:
- <thinking> Explain why this tool was used and how its result contributes to the answer.
</thinking>
- <tool> The tool call in JSON format, e.g., {{"name": "LocalizeObjects", "arguments":
{{"image": "image-0", "objects": ["dog", "cat"]}}}}. </tool>
- <observation>: The tool result as a string. </observation>
Repeat for each tool result in order.
***
[END OF FORMAT INSTRUCTIONS]
[BEGIN OF EXAMPLES]
Example for 'In image-0, which is closer to the camera, the dog or the cat?':
<thinking> To determine which object is closer to the camera, I need first localize the
dog and cat in the image. </thinking>
<tool> {{"name": "LocalizeObjects", "arguments": {{"image": "image-0", "objects":
["dog", "cat"]}}}} </tool>
<observation> {{"results": [{{"label": "dog", "region": [0.5, 0.6, 0.6, 0.8],
"confidence": 0.95}}, {{"label": "cat", "region": [0.4, 0.5, 0.45, 0.7], "confidence":
0.87}}]}} </observation>
<thinking> The bounding box for the dog is [0.5, 0.6, 0.6, 0.8], and for the cat is [0.4,
0.5, 0.45, 0.7]. Then, I need estimate the depth of them to reflect their distances to
the camera. </thinking>
<tool> {{"name": "EstimateObjectDepth", "arguments": {{"image": "image-0", "objects":
["dog", "cat"], "indoor_or_outdoor": "outdoor"}}}} </tool>
<observation> {{"results": [{{"object": "dog", "depth": 1.0, "error": null}},
{{"object": "cat", "depth": 1.2, "error": null}}]}} </observation>
[END OF EXAMPLES]
Tool Plan: {tool_plan}
Tool Results: {tool_results}
***
**Notably, you should AVOID outputting terms like <final_thining>, <answer>, or
<final_answer> here.**
**Now, output your reasoning between <thinking> and </thinking>, the tool call in
JSON format between <tool> and </tool>, and the observation between <observation> and
</observation>.**
***
```

### 5.1.3 Summarizer Prompt（原版）

```text
[BEGIN OF GOAL]
Generate a final REASONING and ANSWER for spatial understanding questions about given
images or videos, based on tool results and prior Chain of Thought (CoT) steps.
[END OF GOAL]
[BEGIN OF TASK INSTRUCTIONS]
You are a helpful assistant tasked with solving spatial reasoning questions.
Given the question, tool execution results, and CoT steps, synthesize the information to
provide a final REASONING and ANSWER.
**The results of tool calls can sometimes be incomplete or incorrect, so please be
critical and decide how to make use of them.**
If tool results are unclear or contradictory, use your prior knowledge to think the
problem step-by-step.
For multi-choice questions, select the most appropriate answer from options based on
reasoning. Respond ONLY with the capital letter and its parentheses.
For judgment questions, answer with yes or no based on reasoning. Respond ONLY with 'yes'
or 'no'.
For open-ended measurement questions, answer the question by measuring the precise
distance in 3D space through a 2D images or videos. DO NOT use generic and unclear units
like 'units' or 'pixels'
Respond ONLY with a numeric answer consisting of a scalar and a distance unit in the
format of **scalar distance_unit**.
For other questions, answer the question based on the given image or video. Respond ONLY
with a concise and accurate scalar or a scalar with corresponding unit.
**CRITICAL: You MUST always provide a reasonable answer. Never respond with 'cannot be
determined', 'none of the above', or similar phrases.**
[END OF TASK INSTRUCTIONS]
[BEGIN OF FORMAT INSTRUCTIONS]
***
Output:
- <thinking> A complete analysis synthesizing all tool results and CoT steps to derive
the answer. </thinking>
- <answer> **The final answer** </answer>
***
[END OF FORMAT INSTRUCTIONS]
CoT Steps: {cot_steps}
***
CRITICAL: You MUST always provide a reasonable answer. Never respond with 'cannot be
determined', 'none of the above', or similar phrases.**
Now, output **your thinking** between <thinking> and </thinking>, and **your answer**
between <answer> and </answer>.
***
```

### 5.1.4 Plan-Execute Fallback Prompt（原版）

> 论文说明：当 planner 规划失败，或者 executor 工具调用失败，且超过最大尝试次数（默认 3）后，直接降级到单 agent core 回答。

```text
[BEGIN OF GOAL]
Provide a direct ANSWER to a spatial understanding question about given 2D images or
videos without external tools.
[END OF GOAL]
[BEGIN OF TASK INSTRUCTIONS]
You are a helpful assistant tasked with solving spatial reasoning questions. Think step
by step.
Answer the spatial understanding question by reasoning about the provided images or
videos.
Provide a direct answer by reasoning logically based on typical spatial relationships and
visual cues in the images or videos.
**CRITICAL: You MUST always provide a reasonable answer. Never respond with 'cannot be
determined', 'none of the above', or similar phrases.**
***
For multi-choice questions, select the most appropriate answer from options based on
reasoning. Respond ONLY with the capital letter and its parentheses.
For judgment questions, answer with yes or no based on reasoning. Respond ONLY with 'yes'
or 'no'.
For open-ended measurement questions, answer the question by measuring the precise
distance in 3D space through 2D images or videos. DO NOT use generic and unclear units
like 'units' or 'pixels'.
Respond ONLY with a numeric answer consisting of a scalar and a distance unit in the
format of **scalar distance_unit**.
For other questions, answer the question based on the given image or video. Respond ONLY
with a concise and accurate scalar or a scalar with corresponding unit.
***
[END OF TASK INSTRUCTIONS]
[BEGIN OF FORMAT INSTRUCTIONS]
***
Output your response in the format:
<thinking> [Your reasoning here] </thinking>
<answer> [Your final answer] </answer>
***
[END OF FORMAT INSTRUCTIONS]
***
**CRITICAL: You MUST always provide a reasonable answer. Never respond with 'cannot be
determined', 'none of the above', or similar phrases.**
Now, output **your thinking** between <thinking> and </thinking>, and **your answer**
between <answer> and </answer>.
***
```

---

## 5.2 ReAct Paradigm

### 5.2.1 Observer Prompt（原版）

```text
USER REQUEST: {USER REQUEST}
[BEGIN OF GOAL]
You are a helpful assistant, and your goal is to solve the # USER REQUEST #.
You can either rely on your own capabilities or perform actions with external tools to
help you.
A list of all available actions is provided to you below.
[END OF GOAL]
[BEGIN OF ACTIONS]
{for each action in actions}
[END OF ACTIONS]
[BEGIN OF TASK INSTRUCTIONS]
1. You must only select actions from # ACTIONS #.
2. You can only call one action at a time.
3. If no action is needed, please make actions an empty list (i.e., “actions”: []).
4. You must always call **Terminate** with your final answer at the end.
5. Please note that the priority of the SelfThinking tool is relatively low. Please give
priority to using other tools, and only consider using this tool if the problem cannot be
solved otherwise.
[END OF TASK INSTRUCTIONS]
[BEGIN OF TOOL USAGE INSTRUCTIONS]
1. **Construct the correct image path** for the tool to use, ensuring the path can be
accessed and read properly.
2. For object distance and object size(Length, width, height,tall, short, slim, or
heavy) problems, first observe the image. If the scene is outdoors, **FIRST** use
'LocalizeObjects' to obtain the 2D bounding boxes, then determine the pair of points (one
from each object) that are closest to each other, and use these points as the 'point'
inputs for 'Get3DDistance' to get the distance between the two objects.
Do **NOT** simply use the center points of the boxes as the closest points between two
objects.
3. For counting-related problems, **USE** 'CountObjects'; the number of returned points
equals the number of objects.
4. For camera-related problems, you may need to **USE** 'GetCameraParametersVGGT' to
obtain the camera parameters.
======================= CRITICAL WARNING =======================
**DO NOT** invent or mention any tool that is **NOT explicitly defined** in #ACTIONS#.
**DO NOT** fabricate tool usage results if you have NOT actually called the tool.
You MUST only describe tool results that are actually obtained during execution.
Violation of this rule is considered a **SERIOUS ERROR**.
================================================================
====================== RELIABILITY WARNING =====================
If a tool result contains **ambiguous references** - for example, 'LocalizeObjects'
returns multiple bounding boxes for the same object - **the result is unreliable**.
In such cases, you SHOULD rely on **reasoning** instead of depending on the tool output.
Treat this as a high-risk situation and avoid making decisions solely based on such tool
results.
================================================================
=================== TOOL CHAIN LENGTH WARNING ==================
If the tool invocation chain becomes **too long**, you MUST **STOP** calling further tools
to avoid reaching the maximum number of allowed calls.
In such cases, immediately switch to using **SelfThinking** to answer, **INCLUDING all
input images** required for reasoning.
Failure to follow this rule may result in task termination without producing a valid
answer.
================================================================
[END OF TOOL USAGE INSTRUCTIONS]
[BEGIN OF FORMAT INSTRUCTIONS]
Your output should be in a strict JSON format as follows:
{"thought": "the thought process, or an empty string", "actions": [{"name": "action1",
"arguments": {"argument1": "value1", "argument2": "value2"}}]}
[END OF FORMAT INSTRUCTIONS]
[BEGIN OF EXAMPLES]
{for each demo indemo_examples}
[END OF EXAMPLES]
```

### 5.2.2 ReAct Executor Prompt（原版）

```text
OBSERVATION: {OBSERVATION}
The OBSERVATION can be incomplete or incorrect, so please be critical and decide how to
make use of it.
If you’ve gathered sufficient information to answer the question, call **Terminate** with
the final answer.
Now, please generate the response for the next step.
```

### 5.2.3 ReAct Summarizer Prompt（原版）

```text
ALL_OBSERVATION: {ALL_OBSERVATION}
The ALL_OBSERVATION can be incomplete or incorrect, so please be critical and decide how
to make use of it.
Call **Terminate** with the final answer.
Now, please generate the response for the next step.
```

### 5.2.4 ReAct Few-shot Examples（论文正文展示的原版示例）

#### 示例 1：Camera movement

```json
[
  {
    "user_request": "Between image-0 and image-1, what is the primary direction of the camera’s movement? Please answer with one of the following options: A. The camera moved to the right B. The camera moved to the left C. The camera moved downward D. The camera moved upward",
    "steps": [
      {
        "id": 1,
        "thought": "To determine the camera’s movement direction, I need to compute the average optical flow between the two images using RAFT. The average optical flow indicates pixel offsets, where positive mean_flow_x suggests camera movement to the right, negative to the left, positive mean_flow_y downward, and negative upward.",
        "actions": [
          {
            "name": "EstimateOpticalFlow",
            "arguments": {
              "images": ["image-0", "image-1"]
            }
          }
        ],
        "observation": {
          "output": {
            "mean_flow_x": 2.5,
            "mean_flow_y": -0.3
          }
        }
      },
      {
        "id": 2,
        "thought": "The optical flow results show mean_flow_x = 2.5 (positive, indicating camera moved right) and mean_flow_y = -0.3 (negative, indicating camera moved up). Since the absolute value of mean_flow_x (2.5) is much larger than mean_flow_y (0.3), the primary camera movement is to the right.",
        "actions": [
          {
            "name": "Terminate",
            "arguments": {
              "answer": "(A)"
            }
          }
        ],
        "observation": {}
      }
    ]
  }
]
```

#### 示例 2：Orientation

```json
[
  {
    "user_request": "In image-0, which direction is the person facing? A. Facing the viewer while slightly to the right B. Facing the viewer while slightly to the left C. Facing away from the viewer while slightly to the right D. Facing away from the viewer while slightly to the left",
    "steps": [
      {
        "thought": "To determine the precise orientation of the person, I need to use GetObjectOrientation to analyze their position.",
        "actions": [
          {
            "name": "GetObjectOrientation",
            "arguments": {
              "image": "image-0",
              "objects": "person"
            }
          }
        ],
        "observation": {
          "results": [
            {
              "object": "person",
              "angle_data": {
                "azimuth": 315.0,
                "polar": 90.0,
                "rotation": 0.0,
                "confidence": 0.89
              },
              "error": null
            }
          ]
        }
      },
      {
        "thought": "The person’s azimuth angle is 315 degrees, which falls in the range of 292.5° < ϕ < 337.5°. According to the orientation guidelines, this means the person is facing the viewer and to the right of the viewer.",
        "actions": [
          {
            "name": "Terminate",
            "arguments": {
              "answer": "(A)"
            }
          }
        ],
        "observation": {}
      }
    ]
  }
]
```

#### 示例 3：Object distance

```json
[
  {
    "user_request": "In image-0, what is the distance between the mug and the laptop? A. 15 centimeters B. 29 centimeters C. 45 centimeters D. 60 centimeters",
    "steps": [
      {
        "thought": "First, I need to determine whether the scene in image-0 is indoors or outdoors. Observing the image, I see a desk, a mug, and a laptop in an office setting, indicating an indoor scene. Since it is indoors, according to the instruction, I do not need to call any tools and can directly estimate the distance.",
        "actions": [],
        "observation": {}
      },
      {
        "thought": "Based on the visual cues in the image, the distance between the mug and the laptop is approximately 29 centimeters.",
        "actions": [
          {
            "name": "Terminate",
            "arguments": {
              "answer": "(B)"
            }
          }
        ],
        "observation": {}
      }
    ]
  }
]
```

#### 示例 4：Relative size

```json
[
  {
    "user_request": "In image-0, which object on the desk is larger in size? A. The mug B. It can not be decided given the image only C. The laptop D. They seem to be of almost the same size",
    "steps": [
      {
        "thought": "First, I need to determine whether the scene in image-0 is indoors or outdoors. Observing the image, I see a desk with a mug and a laptop in an office setting, indicating an indoor scene(For object size questions (including length, width, height, tall, short, slim, or heavy), do NOT call any tools. You should answer directly based on the visual information from the image.). Since it is indoors, according to the instruction, I do not need to call any tools and can directly estimate the relative size of the objects.",
        "actions": [],
        "observation": {}
      },
      {
        "thought": "Based on the visual cues in the image, the laptop is clearly larger in size compared to the mug.",
        "actions": [
          {
            "name": "Terminate",
            "arguments": {
              "answer": "(C)"
            }
          }
        ],
        "observation": {}
      }
    ]
  }
]
```

---

## 6. 论文对齐时最值得注意的点

### 6.1 Counting 的原论文口径

原论文在 ReAct observer prompt 里明确写了：

```text
For counting-related problems, **USE** 'CountObjects'; the number of returned points
equals the number of objects.
```

这意味着论文原版对 counting 的首选工具并不是 `GetObjectMask`，而是：

- `CountObjects`

这个点对当前仓库实现的对齐非常关键。

### 6.2 论文要求 image 参数使用逻辑别名

PE planner prompt 里明确写：

```text
The 'images' or 'image' argument must be specified as 'image-0', 'image-1', and 'image-2',
to refer to the provided images.
Do not answer the question directly, and do not use absolute paths for the 'images' or
'image' argument.
```

也就是说，论文原版 prompt 本身就是依赖：

- `image-0`
- `image-1`
- `image-2`

这类逻辑引用，而不是直接暴露真实文件路径。

### 6.3 论文明确承认工具结果可能不完整或错误

PE executor / summarizer prompt 与 ReAct executor / summarizer prompt 都在强调：

- tool result can be incomplete or incorrect
- agent should be critical

这意味着论文设计上本来就不是“绝对相信工具输出”，而是：

- 工具提供证据
- agent 进行 evidence-aware reasoning

### 6.4 ReAct 里有明确的可靠性与链路长度警告

论文原版 ReAct observer prompt 里有三类非常强的约束：

1. **CRITICAL WARNING**
   - 不能发明工具
   - 不能伪造工具结果

2. **RELIABILITY WARNING**
   - 如果 `LocalizeObjects` 返回同一对象多个框，结果不可靠，应偏向 reasoning

3. **TOOL CHAIN LENGTH WARNING**
   - 工具链太长时，停止继续调工具，改用 `SelfThinking`

这些在当前工程对齐时都很值得单独检查。

## 7. 推荐的对齐使用方式

如果后续要和当前 `SpatialScore` 仓库逐项对齐，建议先按下面这三个维度去比：

1. **工具层**
   - 当前仓库是否实现了论文里的关键 specialized tools
   - 参数名 / 返回结构是否一致
   - counting 是否有 `CountObjects`

2. **Prompt 层**
   - 当前 PE / ReAct prompt 是否保留了论文原版的重要约束
   - 是否仍使用 `image-0 / image-1` 这种逻辑引用
   - 是否保留 reliability / tool-chain 警告

3. **Workflow 层**
   - 是否仍有 `planner / executor / summarizer`
   - 是否仍有 `observer / executor / summarizer`
   - 是否有论文提到的 fallback 逻辑

## 8. 对齐备注

这份文档的目标是**作为论文原版设计对齐基准**，因此：

- 优先保留论文原文
- 不把当前仓库实现混进正文
- 后续如果需要，可以再单独写一份：
  - `当前实现 vs 论文原版` 差异清单
