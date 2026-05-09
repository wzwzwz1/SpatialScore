from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

from spatial_agent.factory import build_spatial_agent
from spatial_agent.io.lmms_bridge import build_task_input_from_vsibench_doc
from spatial_agent.io.video_sampling import sample_video_frames
from spatial_agent.runtime.config import SpatialAgentConfig, load_tool_config

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("spatial_agent_api")
class SpatialAgentAPI(lmms):
    """lmms-eval model wrapper that runs SpatialAgent on same-host VSI-Bench inputs."""

    def __init__(
        self,
        llm_backend: str = "hf",
        model_path: str = "",
        model_name: str = "",
        api_base_url: str = "",
        api_key: str = "",
        api_base_url_env: str = "OPENAI_API_BASE_URL",
        api_key_env: str = "OPENAI_API_KEY",
        api_timeout: int = 120,
        max_steps: int = 8,
        video_num_frames: int = 16,
        video_frame_dir: str = "",
        artifact_dir: str = ".artifacts/spatial_agent",
        keep_video_frames: bool = False,
        tool_config_path: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = SpatialAgentConfig(
            llm_backend=llm_backend,
            qwen_model_path=model_path or None,
            api_model_name=model_name or None,
            api_base_url=api_base_url or None,
            api_key=api_key or None,
            api_base_url_env=api_base_url_env,
            api_key_env=api_key_env,
            api_timeout=api_timeout,
            max_steps=max_steps,
            artifact_dir=artifact_dir,
            video_num_frames=video_num_frames,
            video_frame_dir=video_frame_dir or None,
            keep_video_frames=keep_video_frames,
            tool_config=load_tool_config(tool_config_path or None),
        )
        self.agent = build_spatial_agent(self.config)

    def _flatten(self, items):
        flattened = []
        for item in items:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        return flattened

    def _sample_visuals(self, task: str, split: str, doc_id: int, visuals: List[str]) -> Tuple[List[str], Path]:
        frame_root = Path(self.config.video_frame_dir or self.config.artifact_dir) / "sampled_frames" / task / split / str(doc_id)
        frame_paths = sample_video_frames(
            video_path=str(visuals[0]),
            output_dir=str(frame_root),
            num_frames=self.config.video_num_frames,
        )
        return frame_paths, frame_root

    def generate_until(self, requests) -> List[str]:
        responses: List[str] = []

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc = self.task_dict[task][split][doc_id]
            visuals = self._flatten([doc_to_visual(doc)])
            if not visuals:
                raise ValueError(f"No visual input found for task={task} split={split} doc_id={doc_id}.")

            frame_paths, frame_root = self._sample_visuals(task=task, split=split, doc_id=doc_id, visuals=visuals)
            task_input = build_task_input_from_vsibench_doc(
                doc=doc,
                image_paths=frame_paths,
                task_id=f"{task}___{split}___{doc_id}",
            )

            result = self.agent.invoke(task_input)
            responses.append((result.get("final_answer") or "").strip())

            if not self.config.keep_video_frames:
                shutil.rmtree(frame_root, ignore_errors=True)

        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise AssertionError("SpatialAgentAPI does not support loglikelihood evaluation.")
