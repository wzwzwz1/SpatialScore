from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}


def _normalize_options(options: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not options:
        return None
    return [str(option).strip() for option in options]


def map_vsibench_question_type(question_type: str) -> str:
    if question_type in MCA_QUESTION_TYPES:
        return "multi_choice"
    return "open_ended"


def build_task_input_from_vsibench_doc(
    doc: Mapping[str, Any],
    image_paths: List[str],
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "task_id": task_id or f"{doc.get('scene_name', 'scene')}::{doc.get('question_id', doc.get('question', 'question'))}",
        "question": doc["question"],
        "question_type": map_vsibench_question_type(doc["question_type"]),
        "input_modality": "video",
        "image_paths": list(image_paths),
        "options": _normalize_options(doc.get("options")),
        "metadata": {
            "source_benchmark": "vsibench",
            "vsibench_question_type": doc.get("question_type"),
            "dataset": doc.get("dataset"),
            "scene_name": doc.get("scene_name"),
            "ground_truth": doc.get("ground_truth"),
        },
    }
