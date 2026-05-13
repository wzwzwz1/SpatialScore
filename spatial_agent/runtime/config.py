from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SpatialAgentConfig:
    llm_backend: str = "hf"
    max_steps: int = 10
    max_repairs: int = 2
    max_tool_fails: int = 3
    artifact_dir: str = ".artifacts/spatial_agent"
    qwen_model_path: Optional[str] = None
    api_model_name: Optional[str] = None
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url_env: str = "OPENAI_API_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    api_timeout: int = 120
    video_num_frames: int = 16
    video_frame_dir: Optional[str] = None
    keep_video_frames: bool = False
    tool_config: Dict[str, Dict[str, object]] = field(default_factory=dict)


def load_tool_config(path: Optional[str]) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("tool_config file must contain a JSON object at the top level.")
    return content
