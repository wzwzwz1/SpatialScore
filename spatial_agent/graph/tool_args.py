from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping


_INDEXED_IMAGE_PATTERN = re.compile(r"^(?:image|frame)[_\- ]?(\d+)(?:\.[a-z0-9]+)?$", re.IGNORECASE)
_FIRST_IMAGE_PATTERN = re.compile(r"^(?:first|current|input)[_\- ]?image(?:\.[a-z0-9]+)?$", re.IGNORECASE)
_GENERIC_IMAGE_TOKENS = {
    "image",
    "images",
    "image_path",
    "image_paths",
    "image_data",
    "frame",
    "frames",
}


def get_image_reference_map(image_paths: List[str]) -> Dict[str, str]:
    reference_map: Dict[str, str] = {}
    for index, image_path in enumerate(image_paths):
        normalized_path = str(Path(image_path))
        basename = Path(normalized_path).name
        zero_based_aliases = {
            f"image-{index}",
            f"image_{index}",
            f"image {index}",
        }
        one_based_index = index + 1
        one_based_frame_aliases = {
            f"frame-{one_based_index}",
            f"frame_{one_based_index}",
            f"frame {one_based_index}",
            f"frame{one_based_index}",
            f"frame{one_based_index}.jpg",
            f"frame{one_based_index}.jpeg",
            f"frame{one_based_index}.png",
        }
        aliases = {normalized_path.lower(), basename.lower(), *zero_based_aliases, *one_based_frame_aliases}
        for alias in aliases:
            reference_map.setdefault(alias, normalized_path)
    return reference_map


def _looks_like_placeholder_image(value: str) -> bool:
    token = value.strip().lower()
    if token in _GENERIC_IMAGE_TOKENS:
        return True
    if _INDEXED_IMAGE_PATTERN.match(token):
        return True
    if _FIRST_IMAGE_PATTERN.match(token):
        return True
    if any(sep in token for sep in ("/", "\\")):
        return False
    return token.startswith("image") or token.startswith("frame")


def _resolve_single_image_token(token: str, image_paths: List[str]) -> str:
    if not image_paths:
        return token
    stripped = token.strip()
    reference_map = get_image_reference_map(image_paths)
    direct_match = reference_map.get(stripped.lower())
    if direct_match:
        return direct_match

    lowered = stripped.lower()
    if lowered in _GENERIC_IMAGE_TOKENS or _FIRST_IMAGE_PATTERN.match(lowered):
        return image_paths[0]

    match = _INDEXED_IMAGE_PATTERN.match(lowered)
    if match:
        raw_index = int(match.group(1))
        if lowered.startswith("image"):
            has_separator = any(separator in lowered for separator in ("-", "_", " "))
            index = max(0, raw_index if has_separator else raw_index - 1)
        else:
            index = max(0, raw_index - 1)
        return image_paths[min(index, len(image_paths) - 1)]
    return image_paths[0]


def _normalize_image_value(key: str, value: Any, image_paths: List[str]) -> Any:
    if not image_paths:
        return value

    reference_map = get_image_reference_map(image_paths)
    if isinstance(value, str):
        direct_match = reference_map.get(value.strip().lower())
        if direct_match:
            return direct_match
        if _looks_like_placeholder_image(value) and not Path(value).exists():
            return _resolve_single_image_token(value, image_paths)
        return value

    if isinstance(value, list) and key in {"images", "other_images", "image_paths"}:
        if not value:
            return image_paths
        string_values = [str(item) for item in value]
        resolved_values = []
        all_resolved = True
        for item in string_values:
            direct_match = reference_map.get(item.strip().lower())
            if direct_match:
                resolved_values.append(direct_match)
                continue
            if _looks_like_placeholder_image(item) and not Path(item).exists():
                resolved_values.append(_resolve_single_image_token(item, image_paths))
                continue
            all_resolved = False
            break
        if all_resolved:
            if resolved_values:
                return resolved_values
            return image_paths
        return value

    return value


def is_video_counting_task(state: Mapping[str, Any]) -> bool:
    metadata = state.get("metadata") or {}
    benchmark_type = str(metadata.get("vsibench_question_type") or "").lower()
    if "count" not in benchmark_type:
        question = str(state.get("question") or "").strip().lower()
        if "how many" not in question and "number of" not in question:
            return False
    return str(state.get("input_modality") or "").lower() == "video"


def get_representative_counting_frames(state: Mapping[str, Any]) -> List[str]:
    image_paths = [str(path) for path in state.get("image_paths", [])]
    if len(image_paths) <= 3:
        return image_paths

    candidate_indices = [0, len(image_paths) // 2, len(image_paths) - 1]
    ordered_unique_indices: List[int] = []
    for index in candidate_indices:
        if index not in ordered_unique_indices:
            ordered_unique_indices.append(index)
    return [image_paths[index] for index in ordered_unique_indices]


def get_observed_counting_frames(state: Mapping[str, Any]) -> List[str]:
    observed: List[str] = []
    for call in state.get("tool_calls", []) or []:
        if call.get("tool_name") != "CountObjects":
            continue
        arguments = call.get("arguments") or {}
        image = arguments.get("image")
        if isinstance(image, str) and image not in observed:
            observed.append(image)
    return observed


def get_next_counting_frame(state: Mapping[str, Any]) -> str | None:
    representative_frames = get_representative_counting_frames(state)
    if len(representative_frames) <= 1:
        return None
    observed_frames = set(get_observed_counting_frames(state))
    for frame_path in representative_frames:
        if frame_path not in observed_frames:
            return frame_path
    return None


def normalize_tool_arguments(state: Mapping[str, Any], tool_name: str | None, arguments: Dict[str, Any] | None) -> Dict[str, Any]:
    args = dict(arguments or {})
    image_paths = [str(path) for path in state.get("image_paths", [])]
    if not image_paths:
        return args

    for key in list(args.keys()):
        if key in {"image", "images", "other_images", "image_paths"}:
            args[key] = _normalize_image_value(key, args[key], image_paths)

    if tool_name in {"CountObjects", "LocalizeObjects", "GetObjectMask", "EstimateObjectDepth", "GetObjectOrientation"}:
        if "image" not in args or (isinstance(args.get("image"), str) and _looks_like_placeholder_image(str(args["image"]))):
            if tool_name == "CountObjects" and is_video_counting_task(state):
                args["image"] = get_next_counting_frame(state) or image_paths[0]
            else:
                args["image"] = image_paths[0]

    if tool_name in {"EstimateOpticalFlow", "EstimateObjectMotion"}:
        if "images" not in args:
            args["images"] = image_paths

    if tool_name == "EstimateHomographyMatrix":
        if "image" not in args or not isinstance(args.get("image"), list):
            args["image"] = image_paths[:2]

    if tool_name == "GetCameraParametersVGGT":
        if "image" not in args and "images" not in args:
            args["image"] = image_paths

    return args
