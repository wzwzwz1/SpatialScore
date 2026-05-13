from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from spatial_agent.adapters.base import AdapterResponseError, LLMAdapter
from spatial_agent.adapters.prompting import build_text_prompt_from_state
from spatial_agent.adapters.react_decisions import parse_react_decisions
from spatial_agent.prompts.react_system_prompt import build_react_system_prompt
from spatial_agent.prompts.repair_prompt import build_repair_prompt


class HuggingFaceQwenAdapter(LLMAdapter):
    """Default local HuggingFace Qwen VL adapter with lazy imports."""

    def __init__(self, model_path: Optional[str] = None, device_map: str = "auto") -> None:
        self.model_path = model_path
        self.device_map = device_map
        self._model = None
        self._processor = None
        self._process_vision_info = None
        self.last_raw_output = ""
        self.last_parsed_decisions = []
        self.last_parse_summary = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        if not self.model_path:
            raise AdapterResponseError(
                "No Qwen model path configured. Set SpatialAgentConfig.qwen_model_path or pass --model-path."
            )

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise AdapterResponseError(
                "HuggingFace Qwen adapter dependencies are unavailable. Install transformers, torch, and qwen-vl-utils.",
            ) from exc

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=self.device_map,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
        self._process_vision_info = process_vision_info

    def _build_messages(self, state: Mapping[str, Any], available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        base_prompt = build_react_system_prompt(available_tools)
        text_prompt = build_text_prompt_from_state(state)
        image_content = [{"type": "image", "image": image_path} for image_path in state.get("image_paths", [])]
        return [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": image_content + [{"type": "text", "text": text_prompt}]},
        ]

    def generate(self, state: Mapping[str, Any], available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._ensure_loaded()
        self.last_parsed_decisions = []
        self.last_parse_summary = None

        messages = self._build_messages(state, available_tools)
        try:  # pragma: no cover - depends on optional runtime deps
            input_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self._processor(
                text=[input_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)
            output_ids = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            raw_output = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            self.last_raw_output = raw_output
        except Exception as exc:
            self.last_raw_output = ""
            raise AdapterResponseError("Qwen adapter inference failed.") from exc

        try:
            parsed = parse_react_decisions(raw_output)
            self.last_parsed_decisions = list(parsed.accepted_steps)
            self.last_parse_summary = {
                "parsed_step_count": parsed.parsed_step_count,
                "accepted_step_count": parsed.accepted_step_count,
                "dropped_step_count": parsed.dropped_step_count,
                "dropped_steps": parsed.dropped_steps,
            }
            return parsed.accepted_steps[0]
        except json.JSONDecodeError as exc:
            raise AdapterResponseError(build_repair_prompt(raw_output), raw_output=raw_output) from exc
