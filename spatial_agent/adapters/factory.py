from __future__ import annotations

from spatial_agent.adapters.huggingface_qwen import HuggingFaceQwenAdapter
from spatial_agent.adapters.openai_compatible import OpenAICompatibleAdapter


def create_llm_adapter(config):
    backend = (config.llm_backend or "hf").lower()

    if backend == "hf":
        return HuggingFaceQwenAdapter(model_path=config.qwen_model_path)

    if backend in {"openai", "openai_compatible"}:
        if not config.api_model_name:
            raise ValueError("SpatialAgentConfig.api_model_name is required for the OpenAI-compatible backend.")
        return OpenAICompatibleAdapter(
            model_name=config.api_model_name,
            api_base_url=config.api_base_url,
            api_key=config.api_key,
            api_base_url_env=config.api_base_url_env,
            api_key_env=config.api_key_env,
            timeout=config.api_timeout,
        )

    raise ValueError(f"Unsupported llm backend: {config.llm_backend}")
