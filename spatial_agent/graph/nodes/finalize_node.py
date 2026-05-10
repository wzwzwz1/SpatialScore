from __future__ import annotations

import re


_COUNT_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def finalize_node(runtime):
    def _finalize_node(state):
        finish = (state.get("pending_decision") or {}).get("finish") or {}
        answer = finish.get("answer", "")
        state["final_answer"] = _normalize_answer(
            answer,
            state.get("question_type"),
            state.get("options"),
            state.get("metadata"),
        )
        state["status"] = "success"
        state["reasoning_trace"].append({"stage": "finalize", "answer": state["final_answer"]})
        return state

    return _finalize_node


def _normalize_answer(answer, question_type, options, metadata=None):
    if answer is None:
        return None
    normalized_text = str(answer).strip()
    if _is_vsibench_object_counting(metadata):
        return _normalize_counting_answer(normalized_text)
    if question_type == "multi_choice":
        normalized = normalized_text
        if normalized.startswith("(") and normalized.endswith(")"):
            return normalized[1:-1]
        return normalized
    if question_type == "judgment":
        return normalized_text.lower()
    return normalized_text


def _is_vsibench_object_counting(options):
    metadata = options if isinstance(options, dict) else None
    if not metadata:
        return False
    return metadata.get("source_benchmark") == "vsibench" and metadata.get("vsibench_question_type") == "object_counting"


def _normalize_counting_answer(answer: str) -> str:
    lowered = answer.strip().lower()
    number_match = re.search(r"\d+", lowered)
    if number_match:
        return number_match.group(0)

    tokens = re.findall(r"[a-z]+", lowered)
    for token in tokens:
        if token in _COUNT_WORDS:
            return _COUNT_WORDS[token]
    return answer.strip()
