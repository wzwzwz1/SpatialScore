from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional

from spatial_agent.adapters.base import AdapterResponseError, LLMAdapter
from spatial_agent.adapters.prompting import (
    build_openai_image_content,
    build_text_prompt_from_state,
    normalize_response_text,
)
from spatial_agent.prompts.react_system_prompt import build_react_system_prompt
from spatial_agent.prompts.repair_prompt import build_repair_prompt


class OpenAICompatibleAdapter(LLMAdapter):
    """OpenAI-compatible multimodal adapter for remote API inference."""

    def __init__(
        self,
        model_name: str,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url_env: str = "OPENAI_API_BASE_URL",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.api_base_url_env = api_base_url_env
        self.api_key_env = api_key_env
        self.timeout = timeout
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        resolved_key = self.api_key or os.getenv(self.api_key_env)
        resolved_base_url = self.api_base_url or os.getenv(self.api_base_url_env)
        if not resolved_key:
            raise AdapterResponseError(
                f"No API key configured. Set SpatialAgentConfig.api_key or environment variable {self.api_key_env}."
            )

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise AdapterResponseError(
                "OpenAI-compatible adapter dependencies are unavailable. Install the openai package."
            ) from exc

        client_kwargs = {"api_key": resolved_key, "timeout": self.timeout}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self._client = OpenAI(**client_kwargs)
        return self._client

    def _build_messages(self, state: Mapping[str, Any], available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        base_prompt = build_react_system_prompt(available_tools)
        text_prompt = build_text_prompt_from_state(state)
        content = build_openai_image_content(list(state.get("image_paths", [])))
        content.append({"type": "text", "text": text_prompt})
        return [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": content},
        ]

    def generate(self, state: Mapping[str, Any], available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        client = self._ensure_client()
        messages = self._build_messages(state, available_tools)

        try:  # pragma: no cover - depends on remote API
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=1024,
            )
            raw_output = normalize_response_text(response.choices[0].message.content)
        except Exception as exc:
            raise AdapterResponseError("OpenAI-compatible adapter inference failed.") from exc

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise AdapterResponseError(build_repair_prompt(raw_output), raw_output=raw_output) from exc
