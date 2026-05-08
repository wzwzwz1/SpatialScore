from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SpatialAgentConfig:
    llm_backend: str = "hf"
    max_steps: int = 8
    max_repairs: int = 2
    max_tool_fails: int = 3
    artifact_dir: str = ".artifacts/spatial_agent"
    qwen_model_path: Optional[str] = None
    api_model_name: Optional[str] = None
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    api_timeout: int = 120
    video_num_frames: int = 16
    video_frame_dir: Optional[str] = None
    keep_video_frames: bool = False
    tool_config: Dict[str, Dict[str, object]] = field(default_factory=dict)
