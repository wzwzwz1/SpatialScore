from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw

try:  # pragma: no cover - optional runtime dependency
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None


ROOT_DIR = Path(__file__).resolve().parents[2]
LEGACY_AGENT_DIR = ROOT_DIR / "version_0" / "SpatialAgent"


@contextmanager
def prepend_sys_path(path: Path):
    text_path = str(path)
    sys.path.insert(0, text_path)
    try:
        yield
    finally:
        try:
            sys.path.remove(text_path)
        except ValueError:
            pass


def get_tool_settings(config: Any, tool_name: str, aliases: Sequence[str] | None = None) -> Dict[str, Any]:
    tool_config = getattr(config, "tool_config", {}) or {}
    keys = [tool_name, tool_name.lower(), tool_name.replace("Tool", ""), tool_name.replace("Tool", "").lower()]
    if aliases:
        keys.extend(aliases)
    for key in keys:
        if key in tool_config and isinstance(tool_config[key], dict):
            return dict(tool_config[key])
    return {}


def resolve_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device

    try:  # pragma: no cover - optional runtime dependency
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def artifact_dir_for_tool(config: Any, tool_name: str) -> Path:
    base_dir = Path(getattr(config, "artifact_dir", ".artifacts/spatial_agent"))
    path = base_dir / "tools" / tool_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_image_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def ensure_object_names(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def load_pil_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_rgb_array(path: str) -> np.ndarray:
    return np.array(load_pil_image(path))


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def clamp_bbox(bbox: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(0.0, min(float(width - 1), float(x2)))
    y2 = max(0.0, min(float(height - 1), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def crop_from_bbox(image: Image.Image, bbox: Sequence[float], expand_ratio: float = 0.08) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * expand_ratio
    pad_y = box_h * expand_ratio
    crop_box = (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    )
    return image.crop(crop_box)


def infer_preprocessed_hw(image_size_wh: Tuple[int, int], mode: str = "crop", target_size: int = 518) -> Tuple[int, int]:
    width, height = image_size_wh
    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be either 'crop' or 'pad'")
    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        return target_size, target_size
    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14
    if new_height > target_size:
        new_height = target_size
    return new_height, new_width


def map_point_to_preprocessed(
    point_xy: Tuple[float, float],
    image_size_wh: Tuple[int, int],
    mode: str = "crop",
    target_size: int = 518,
) -> Tuple[float, float]:
    x, y = point_xy
    width, height = image_size_wh
    if mode == "pad":
        if width >= height:
            resized_width = target_size
            resized_height = round(height * (resized_width / width) / 14) * 14
        else:
            resized_height = target_size
            resized_width = round(width * (resized_height / height) / 14) * 14
        x = x * resized_width / width
        y = y * resized_height / height
        pad_left = max(0, (target_size - resized_width) // 2)
        pad_top = max(0, (target_size - resized_height) // 2)
        return float(x + pad_left), float(y + pad_top)

    resized_width = target_size
    resized_height = round(height * (resized_width / width) / 14) * 14
    x = x * resized_width / width
    y = y * resized_height / height
    if resized_height > target_size:
        crop_top = (resized_height - target_size) / 2.0
        y = y - crop_top
    return float(x), float(y)


def unmap_point_from_preprocessed(
    point_xy: Tuple[float, float],
    image_size_wh: Tuple[int, int],
    mode: str = "crop",
    target_size: int = 518,
) -> Tuple[float, float]:
    x, y = point_xy
    width, height = image_size_wh
    if mode == "pad":
        if width >= height:
            resized_width = target_size
            resized_height = round(height * (resized_width / width) / 14) * 14
        else:
            resized_height = target_size
            resized_width = round(width * (resized_height / height) / 14) * 14
        pad_left = max(0, (target_size - resized_width) // 2)
        pad_top = max(0, (target_size - resized_height) // 2)
        x = (x - pad_left) * width / resized_width
        y = (y - pad_top) * height / resized_height
        return float(x), float(y)

    resized_width = target_size
    resized_height = round(height * (resized_width / width) / 14) * 14
    if resized_height > target_size:
        crop_top = (resized_height - target_size) / 2.0
        y = y + crop_top
    x = x * width / resized_width
    y = y * height / resized_height
    return float(x), float(y)


def save_mask_overlay(image: Image.Image, mask: np.ndarray, bbox: Sequence[float], output_path: Path) -> str:
    overlay = image.copy().convert("RGBA")
    mask_uint8 = (mask.astype(np.uint8) * 120)
    red = Image.new("RGBA", image.size, (255, 64, 64, 0))
    red.putalpha(Image.fromarray(mask_uint8))
    overlay = Image.alpha_composite(overlay, red)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(tuple(bbox), outline=(255, 255, 0, 255), width=3)
    overlay.convert("RGB").save(output_path)
    return str(output_path)


def save_bbox_overlay(
    image: Image.Image,
    regions: Sequence[Dict[str, Any]],
    output_path: Path,
) -> str:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    palette = [
        (255, 90, 90),
        (90, 200, 255),
        (100, 255, 120),
        (255, 210, 90),
        (190, 130, 255),
    ]
    for index, region in enumerate(regions):
        bbox = region.get("bbox")
        if not bbox:
            continue
        color = palette[index % len(palette)]
        x1, y1, x2, y2 = bbox
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = str(region.get("label", "object"))
        score = region.get("score")
        text = f"{label} ({float(score):.2f})" if isinstance(score, (int, float)) else label
        text_x = max(0, int(round(x1)))
        text_y = max(0, int(round(y1)) - 14)
        draw.rectangle((text_x, text_y, text_x + max(48, len(text) * 7), text_y + 14), fill=color)
        draw.text((text_x + 2, text_y + 1), text, fill="black")
    canvas.save(output_path)
    return str(output_path)


def save_track_visualization(frame_paths: Sequence[str], tracks: np.ndarray, output_path: Path) -> str:
    frames = [load_pil_image(path) for path in frame_paths]
    colors = [(255, 90, 90), (90, 200, 255), (100, 255, 120), (255, 210, 90)]
    rendered = []
    for index, frame in enumerate(frames):
        canvas = frame.copy()
        draw = ImageDraw.Draw(canvas)
        for track_index in range(tracks.shape[1]):
            color = colors[track_index % len(colors)]
            x, y = tracks[index, track_index]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            if index > 0:
                px, py = tracks[index - 1, track_index]
                draw.line((px, py, x, y), fill=color, width=2)
        rendered.append(canvas)

    total_width = sum(frame.width for frame in rendered)
    max_height = max(frame.height for frame in rendered)
    grid = Image.new("RGB", (total_width, max_height))
    cursor = 0
    for frame in rendered:
        grid.paste(frame, (cursor, 0))
        cursor += frame.width
    grid.save(output_path)
    return str(output_path)


def save_optical_flow_visualization(flow: np.ndarray, output_path: Path) -> str:
    backend = get_raft_visualization_backend()
    flow_image = backend["flow_to_image"](flow)
    Image.fromarray(flow_image).save(output_path)
    return str(output_path)


@lru_cache(maxsize=1)
def get_depth_backend(cache_key: str = "default") -> Dict[str, Any]:
    import torch  # pragma: no cover - optional runtime dependency

    metric_dir = LEGACY_AGENT_DIR / "DepthAnythingV2" / "metric_depth"
    with prepend_sys_path(metric_dir):
        dpt_module = importlib.import_module("depth_anything_v2.dpt")
    return {"torch": torch, "DepthAnythingV2": dpt_module.DepthAnythingV2}


def build_depth_model(settings: Dict[str, Any]) -> Tuple[Any, str]:
    backend = get_depth_backend(settings.get("cache_key", "default"))
    torch = backend["torch"]
    device = resolve_device(settings.get("device"))
    encoder = settings.get("encoder", "vitl")
    max_depth = float(settings.get("max_depth", 20.0))
    checkpoint = settings.get("checkpoint_path") or settings.get("load_from")
    if not checkpoint:
        default_name = f"depth_anything_v2_metric_hypersim_{encoder}.pth"
        checkpoint = str(LEGACY_AGENT_DIR / "DepthAnythingV2" / "metric_depth" / "checkpoints" / default_name)

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    model = backend["DepthAnythingV2"](**{**model_configs[encoder], "max_depth": max_depth})
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, device


@lru_cache(maxsize=2)
def get_sam2_predictor(
    model_id: str,
    checkpoint_path: str | None,
    config_path: str | None,
    device: str,
) -> Any:
    with prepend_sys_path(LEGACY_AGENT_DIR):
        predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        build_module = importlib.import_module("sam2.build_sam")
    if checkpoint_path and config_path:
        sam_model = build_module.build_sam2(config_path, ckpt_path=checkpoint_path, device=device)
        return predictor_module.SAM2ImagePredictor(sam_model)
    return predictor_module.SAM2ImagePredictor.from_pretrained(model_id, device=device)


@lru_cache(maxsize=2)
def get_vggt_backend(model_id: str, checkpoint_path: str | None, device: str) -> Dict[str, Any]:
    import torch  # pragma: no cover - optional runtime dependency

    with prepend_sys_path(LEGACY_AGENT_DIR):
        vggt_module = importlib.import_module("vggt.models.vggt")
        load_fn_module = importlib.import_module("vggt.utils.load_fn")
        pose_module = importlib.import_module("vggt.utils.pose_enc")

    if checkpoint_path:
        model = vggt_module.VGGT()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    else:
        model = vggt_module.VGGT.from_pretrained(model_id)
    model = model.to(device).eval()
    return {
        "model": model,
        "torch": torch,
        "load_and_preprocess_images": load_fn_module.load_and_preprocess_images,
        "pose_encoding_to_extri_intri": pose_module.pose_encoding_to_extri_intri,
    }


@lru_cache(maxsize=2)
def get_orientation_backend(
    checkpoint_path: str | None,
    checkpoint_repo_id: str,
    checkpoint_filename: str,
    dino_mode: str,
    device: str,
) -> Dict[str, Any]:
    import torch  # pragma: no cover - optional runtime dependency
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor

    orient_dir = LEGACY_AGENT_DIR / "OrientAnything"
    with prepend_sys_path(orient_dir):
        vision_module = importlib.import_module("vision_tower")
        utils_module = importlib.import_module("utils")
    if dino_mode == "large":
        vision_module.DINO_LARGE = "facebook/dinov2-large"

    if not checkpoint_path:
        checkpoint_path = hf_hub_download(repo_id=checkpoint_repo_id, filename=checkpoint_filename, repo_type="model")

    model = vision_module.DINOv2_MLP(
        dino_mode=dino_mode,
        in_dim=int({"small": 384, "base": 768, "large": 1024, "giant": 1536}[dino_mode]),
        out_dim=360 + 180 + 180 + 2,
        evaluate=True,
        mask_dino=False,
        frozen_back=True,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    processor_name = {"small": "facebook/dinov2-small", "base": "facebook/dinov2-base", "large": "facebook/dinov2-large", "giant": "facebook/dinov2-giant"}[dino_mode]
    processor = AutoImageProcessor.from_pretrained(processor_name)
    return {
        "model": model,
        "processor": processor,
        "torch": torch,
        "utils": utils_module,
    }


@lru_cache(maxsize=2)
def get_ram_backend(checkpoint_path: str | None, vit: str, image_size: int, device: str) -> Dict[str, Any]:
    import torch  # pragma: no cover - optional runtime dependency

    with prepend_sys_path(LEGACY_AGENT_DIR):
        inference_module = importlib.import_module("ram.inference")
        transform_module = importlib.import_module("ram.transform")
        model_module = importlib.import_module("ram.models")
    model = model_module.ram(pretrained=checkpoint_path or "", vit=vit, image_size=image_size)
    model = model.to(device).eval()
    transform = transform_module.get_transform(image_size=image_size)
    return {
        "model": model,
        "transform": transform,
        "inference_ram": inference_module.inference_ram,
        "torch": torch,
    }


@lru_cache(maxsize=2)
def get_grounding_backend(model_id: str, device: str) -> Dict[str, Any]:
    import torch  # pragma: no cover - optional runtime dependency
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return {"processor": processor, "model": model, "torch": torch}


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


@lru_cache(maxsize=1)
def get_raft_visualization_backend() -> Dict[str, Any]:
    with prepend_sys_path(LEGACY_AGENT_DIR):
        flow_viz_module = importlib.import_module("RAFT.core.utils.flow_viz")
    return {"flow_to_image": flow_viz_module.flow_to_image}


@lru_cache(maxsize=2)
def get_raft_backend(checkpoint_path: str, small: bool, mixed_precision: bool, alternate_corr: bool, device: str) -> Dict[str, Any]:
    import torch  # pragma: no cover - optional runtime dependency

    with prepend_sys_path(LEGACY_AGENT_DIR):
        raft_module = importlib.import_module("RAFT.core.raft")
        utils_module = importlib.import_module("RAFT.core.utils.utils")

    args = AttrDict(
        small=small,
        mixed_precision=mixed_precision,
        alternate_corr=alternate_corr,
        dropout=0,
    )
    model = raft_module.RAFT(args)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state[key[len("module.") :]] = value
        else:
            cleaned_state[key] = value
    model.load_state_dict(cleaned_state, strict=True)
    model = model.to(device).eval()
    return {"model": model, "torch": torch, "InputPadder": utils_module.InputPadder}
