from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping


def build_text_prompt_from_state(state: Mapping[str, Any]) -> str:
    sections: List[str] = [
        f"Question: {state.get('question', '')}",
        f"Question type: {state.get('question_type', 'open_ended')}",
        f"Input modality: {state.get('input_modality', 'single_image')}",
    ]

    options = state.get("options") or []
    if options:
        sections.append("Options: " + ", ".join(str(option) for option in options))

    scratchpad = state.get("scratchpad") or []
    if scratchpad:
        scratchpad_lines = []
        for index, item in enumerate(scratchpad, start=1):
            scratchpad_lines.append(
                f"Step {index}: thought={item.get('thought')!r}; "
                f"action={item.get('action')!r}; "
                f"arguments={json.dumps(item.get('arguments') or {}, ensure_ascii=False)}; "
                f"observation={json.dumps(item.get('observation') or {}, ensure_ascii=False)}"
            )
        sections.append("Previous tool interaction history:\n" + "\n".join(scratchpad_lines))

    trailing_messages = []
    for message in state.get("messages", [])[1:]:
        content = message.get("content")
        if isinstance(content, str):
            trailing_messages.append(f"{message.get('role', 'unknown')}: {content}")
    if trailing_messages:
        sections.append("Recent conversation context:\n" + "\n".join(trailing_messages[-6:]))

    return "\n\n".join(sections)


def normalize_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                maybe_text = item.get("text") or item.get("content")
                if isinstance(maybe_text, str):
                    text_parts.append(maybe_text)
        return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()
    raise TypeError(f"Unsupported response content type: {type(content)!r}")


def build_openai_image_content(image_paths: List[str]) -> List[Dict[str, Any]]:
    import base64
    import mimetypes
    from pathlib import Path

    content: List[Dict[str, Any]] = []
    for image_path in image_paths:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )
    return content
