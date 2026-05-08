from __future__ import annotations


def build_repair_prompt(raw_output: str) -> str:
    return (
        "The previous model output was not valid ReAct JSON.\n"
        "Return valid JSON using exactly one allowed schema and do not add markdown fences.\n"
        f"Raw output:\n{raw_output}"
    )

