from __future__ import annotations


def finalize_node(runtime):
    def _finalize_node(state):
        finish = (state.get("pending_decision") or {}).get("finish") or {}
        answer = finish.get("answer", "")
        state["final_answer"] = _normalize_answer(answer, state.get("question_type"), state.get("options"))
        state["status"] = "success"
        state["reasoning_trace"].append({"stage": "finalize", "answer": state["final_answer"]})
        return state

    return _finalize_node


def _normalize_answer(answer, question_type, options):
    if answer is None:
        return None
    if question_type == "multi_choice":
        normalized = str(answer).strip()
        if normalized.startswith("(") and normalized.endswith(")"):
            return normalized[1:-1]
        return normalized
    if question_type == "judgment":
        return str(answer).strip().lower()
    return str(answer).strip()

