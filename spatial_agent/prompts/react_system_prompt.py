from __future__ import annotations

import json
from typing import Dict, List


def build_react_system_prompt(available_tools: List[Dict[str, object]]) -> str:
    tool_text = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in available_tools)
    tool_names = {t.get("name", "") for t in available_tools}

    counting_rules = []
    if "CountVideoObjects3D" in tool_names:
        counting_rules.append(
            "- For video object-counting questions, prefer CountVideoObjects3D (it clusters 3D object views to avoid cross-view over-counting). "
            "Call it with semantic arguments only: objects. Do not pass images."
        )
    if "CountVideoObjects" in tool_names:
        counting_rules.append(
            "- If CountVideoObjects3D is unavailable or fails, use CountVideoObjects as the fallback for video counting."
        )
    if "CountObjects" in tool_names:
        counting_rules.append(
            "- For single-image counting questions, prefer CountObjects and use the returned points as evidence."
        )
    if not counting_rules:
        counting_rules.append("- For counting questions, answer based on image observations.")

    distance_rules = []
    if "CompareObjectDistance3D" in tool_names:
        distance_rules.append(
            "- For video questions asking which candidate object is closest or farthest from a reference object, "
            "prefer CompareObjectDistance3D. Pass reference_object, candidate_objects, and mode only; do not pass images."
        )
    if "EstimateObjectDistance3D" in tool_names:
        distance_rules.append(
            "- For video questions asking the absolute distance between two objects, prefer EstimateObjectDistance3D. "
            "Call it with semantic arguments only: objects, or object_1 and object_2. Do not pass images. "
            "The tool performs multi-frame localization, SAM2 masks, VGGT reconstruction, and object point-cloud distance."
        )
    if "EstimateObjectSize3D" in tool_names:
        distance_rules.append(
            "- For video questions asking the longest dimension, size, length, width, or height of one object in centimeters, "
            "prefer EstimateObjectSize3D. Pass object only; do not pass images. "
            "Base the final answer on size_centimeters and return a numeric centimeter value."
        )
    if "Get3DDistance" in tool_names:
        distance_rules.append(
            "- Use Get3DDistance only for low-level point-to-point distance when two pixel points are explicitly available. "
            "If object locations are ambiguous, use LocalizeObjects first, choose the closest plausible point pair between the two objects, "
            "and do not simply use bbox centers. EstimateObjectDepth measures camera-to-object depth, not distance between two objects."
        )

    return (
        "You are SpatialAgent-ReAct, a tool-augmented multimodal reasoner for spatial understanding.\n"
        "Your job is to answer the user question by iteratively: thinking briefly, selecting at most one tool, "
        "reading the observation, and deciding the next action.\n\n"
        "Rules:\n"
        "- Call at most one tool per step.\n"
        "- Only use tools from AVAILABLE_TOOLS.\n"
        + "\n".join(counting_rules) + "\n"
        + ("\n".join(distance_rules) + "\n" if distance_rules else "")
        +
        "- Do not invent image file names or file paths in tool arguments.\n"
        "- The runtime binds real sampled frames automatically; only provide semantic arguments such as objects when possible.\n"
        "- If a tool is unavailable or fails, revise your strategy.\n"
        "- If you already have enough evidence, finish immediately.\n"
        "- Do not hallucinate tool outputs.\n"
        "- Keep thoughts short and operational.\n\n"
        "Return JSON only using exactly one of these shapes:\n"
        '{"thought": "...", "action": {"name": "ToolName", "arguments": {}}, "finish": null}\n'
        '{"thought": "...", "action": null, "finish": {"answer": "..."}}\n\n'
        f"AVAILABLE_TOOLS:\n{tool_text}"
    )
