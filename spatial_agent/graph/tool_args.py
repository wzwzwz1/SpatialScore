from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping


_INDEXED_IMAGE_PATTERN = re.compile(r"^(?:image|frame)[_\- ]?(\d+)$", re.IGNORECASE)
_GENERIC_IMAGE_TOKENS = {
    "image",
    "images",
    "image_path",
    "image_paths",
    "image_data",
    "frame",
    "frames",
}


def _looks_like_placeholder_image(value: str) -> bool:
    token = value.strip().lower()
    if token in _GENERIC_IMAGE_TOKENS:
        return True
    if _INDEXED_IMAGE_PATTERN.match(token):
        return True
    if any(sep in token for sep in ("/", "\\")) or "." in token:
        return False
    return token.startswith("image") or token.startswith("frame")


def _resolve_single_image_token(token: str, image_paths: List[str]) -> str:
    if not image_paths:
        return token
    match = _INDEXED_IMAGE_PATTERN.match(token.strip().lower())
    if match:
        index = max(0, int(match.group(1)) - 1)
        return image_paths[min(index, len(image_paths) - 1)]
    return image_paths[0]


def _normalize_image_value(key: str, value: Any, image_paths: List[str]) -> Any:
    if not image_paths:
        return value

    if isinstance(value, str):
        if _looks_like_placeholder_image(value) and not Path(value).exists():
            return _resolve_single_image_token(value, image_paths)
        return value

    if isinstance(value, list) and key in {"images", "other_images", "image_paths"}:
        if not value:
            return image_paths
        string_values = [str(item) for item in value]
        if all(_looks_like_placeholder_image(item) and not Path(item).exists() for item in string_values):
            indexed = [_INDEXED_IMAGE_PATTERN.match(item.strip().lower()) for item in string_values]
            if any(match is not None for match in indexed):
                return [_resolve_single_image_token(item, image_paths) for item in string_values]
            return image_paths
        return value

    return value


def normalize_tool_arguments(state: Mapping[str, Any], tool_name: str | None, arguments: Dict[str, Any] | None) -> Dict[str, Any]:
    args = dict(arguments or {})
    image_paths = [str(path) for path in state.get("image_paths", [])]
    if not image_paths:
        return args

    for key in list(args.keys()):
        if key in {"image", "images", "other_images", "image_paths"}:
            args[key] = _normalize_image_value(key, args[key], image_paths)

    if tool_name in {"LocalizeObjects", "GetObjectMask", "EstimateObjectDepth", "GetObjectOrientation"}:
        if "image" not in args or (isinstance(args.get("image"), str) and _looks_like_placeholder_image(str(args["image"]))):
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
