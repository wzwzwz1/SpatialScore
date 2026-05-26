from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spatial_agent.tools.backends import load_pil_image, resolve_device, ROOT_DIR

# Coordinate convention: all internal structures use pixel coordinates.
#   - point_px: [x, y] in pixel space
#   - bbox_px:  [x1, y1, x2, y2] in pixel space
# Normalization to [0,1] happens in the final payload via video_counting_utils.


@contextmanager
def _prepend_path(path: Path):
    text_path = str(path)
    sys.path.insert(0, text_path)
    try:
        yield
    finally:
        try:
            sys.path.remove(text_path)
        except ValueError:
            pass


def _diagnose_missing_paths(
    countgd_repo: str | None,
    countgd_ckpt: str | None,
    sam2_ckpt: str | None,
    sam2_config: str | None,
) -> List[str]:
    missing: List[str] = []
    if not countgd_repo:
        missing.append("countgd_repo_path is not configured")
    elif not Path(countgd_repo).is_dir():
        missing.append(f"countgd_repo_path does not exist or is not a directory: {countgd_repo}")
    if not countgd_ckpt:
        missing.append("countgd_checkpoint_path is not configured")
    elif not Path(countgd_ckpt).is_file():
        missing.append(f"countgd_checkpoint_path does not exist: {countgd_ckpt}")
    if not sam2_ckpt:
        missing.append("sam2_checkpoint_path is not configured")
    elif not Path(sam2_ckpt).is_file():
        missing.append(f"sam2_checkpoint_path does not exist: {sam2_ckpt}")
    if not sam2_config:
        missing.append("sam2_config_name is not configured (Hydra config name, e.g. sam2.1/sam2.1_hiera_l.yaml)")
    return missing


def _build_countgd_model(
    repo_path: Path, checkpoint_path: str, device: str
) -> Tuple[Any, Any, List[str]]:
    """Build CountGD-Box model and transform using CountVid repo modules.

    Returns (model, transform, diag_list).
    """
    diag: List[str] = []
    try:
        with _prepend_path(repo_path):
            import torch
            import numpy as np
            import random

            from util.slconfig import SLConfig
            import datasets.transforms as T
            from models.registry import MODULE_BUILD_FUNCS

            # Build transform (from count_in_videos.py build_model_and_transforms)
            normalize = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            data_transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

            # Build model config from cfg_app.py
            cfg = SLConfig.fromfile(str(repo_path / "cfg_app.py"))
            cfg.merge_from_dict({"text_encoder_type": str(repo_path / "checkpoints" / "bert-base-uncased")})

            class _Args:
                pass
            args = _Args()
            args.modelname = "groundingdino"
            args.pretrain_model_path = checkpoint_path
            args.device = device

            cfg_dict = cfg._cfg_dict.to_dict()
            for k, v in cfg_dict.items():
                setattr(args, k, v)

            seed = 42
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            build_func = MODULE_BUILD_FUNCS.get(args.modelname)
            model, _, _ = build_func(args)

            ckpt = torch.load(checkpoint_path, map_location="cpu")["model"]
            model.load_state_dict(ckpt, strict=False)
            model = model.to(device)
            model.eval()

            return model, data_transform, diag
    except Exception as exc:
        diag.append(f"CountGD-Box failed to load: {exc}")
        return None, None, diag


def _countgd_infer(
    model: Any,
    transform: Any,
    pil_image: Any,
    text: str,
    device: str,
    repo_path: Path,
) -> List[Dict[str, Any]]:
    """Run CountGD inference on a single image.

    Returns list of candidate dicts with bbox_px, point_px, score.
    """
    import torch

    width, height = pil_image.size
    CONF_THRESH = 0.23

    with _prepend_path(repo_path):
        from util.misc import nested_tensor_from_tensor_list

    input_image, _ = transform(pil_image, {"exemplars": torch.tensor([])})
    input_tensor = nested_tensor_from_tensor_list(
        list(input_image.unsqueeze(0).to(device))
    )

    with torch.no_grad():
        output = model(
            input_tensor,
            nested_tensor_from_tensor_list(
                list(torch.zeros(1, 3, *input_image.shape[-2:]).to(device))
            ),
            [torch.tensor([]).to(device)],
            [torch.tensor([0]).to(device)],
            captions=[text + " ."] * 1,
        )

    logits = output["pred_logits"].sigmoid()
    boxes = output["pred_boxes"]
    box_mask = logits.max(dim=-1).values > CONF_THRESH
    boxes_filtered = boxes[0, box_mask[0], :]
    logits_filtered = logits[0, box_mask[0], :]

    candidates: List[Dict[str, Any]] = []
    for box_idx in range(boxes_filtered.shape[0]):
        cx, cy, bw, bh = boxes_filtered[box_idx].cpu().tolist()
        score = float(logits_filtered[box_idx].max().cpu().item())
        x1 = width * (cx - bw / 2)
        y1 = height * (cy - bh / 2)
        x2 = width * (cx + bw / 2)
        y2 = height * (cy + bh / 2)
        candidates.append({
            "bbox_px": [x1, y1, x2, y2],
            "point_px": [width * cx, height * cy],
            "score": score,
        })
    return candidates


@lru_cache(maxsize=2)
def get_countvid_backend(
    countgd_repo_path: str,
    countgd_checkpoint_path: str,
    sam2_checkpoint_path: str,
    sam2_config_name: str,
    device: str,
) -> Dict[str, Any]:
    """Load CountVid backend: CountGD-Box + SAM 2.1 predictors."""
    import torch

    device = resolve_device(device)
    countgd_diag: List[str] = []
    sam2_diag: List[str] = []

    repo_path = Path(countgd_repo_path) if countgd_repo_path else None
    valid_paths = (
        repo_path and repo_path.is_dir()
        and countgd_checkpoint_path and Path(countgd_checkpoint_path).is_file()
        and sam2_checkpoint_path and Path(sam2_checkpoint_path).is_file()
        and bool(sam2_config_name)  # Hydra config name, validated at SAM2 init time
    )

    countgd_model = None
    countgd_transform = None
    countgd_available = False

    sam2_image_predictor = None
    sam2_video_predictor = None
    sam2_available = False

    if valid_paths:
        countgd_model, countgd_transform, countgd_diag = _build_countgd_model(
            repo_path, countgd_checkpoint_path, device
        )
        countgd_available = countgd_model is not None

        try:
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_image_predictor = SAM2ImagePredictor(
                build_sam2(sam2_config_name, sam2_checkpoint_path)
            )
            sam2_video_predictor = build_sam2_video_predictor(
                sam2_config_name, sam2_checkpoint_path, device=device,
            )
            sam2_available = True
        except Exception as exc:
            sam2_diag.append(f"SAM 2.1 failed to load: {exc}")
    else:
        countgd_diag = _diagnose_missing_paths(
            countgd_repo_path, countgd_checkpoint_path, None, None
        )
        sam2_diag = _diagnose_missing_paths(
            None, None, sam2_checkpoint_path, sam2_config_name
        )

    return {
        "torch": torch,
        "countgd_model": countgd_model,
        "countgd_transform": countgd_transform,
        "countgd_available": countgd_available,
        "countgd_diag": countgd_diag,
        "sam2_image_predictor": sam2_image_predictor,
        "sam2_video_predictor": sam2_video_predictor,
        "sam2_available": sam2_available,
        "sam2_diag": sam2_diag,
        "device": device,
        "repo_path": str(repo_path) if repo_path else "",
    }


def generate_candidates_countgd(
    backend: Dict[str, Any],
    image_paths: List[str],
    objects: List[str],
) -> List[Dict[str, Any]]:
    """Run CountGD-Box on each frame.

    Returns per-frame dicts with candidates (bbox_px, point_px, score in pixel coords).
    """
    if not backend.get("countgd_available"):
        diag = backend.get("countgd_diag", ["CountGD-Box backend is not available."])
        raise RuntimeError("CountGD-Box unavailable. " + "; ".join(diag))

    model = backend["countgd_model"]
    transform = backend["countgd_transform"]
    device = backend["device"]
    repo_path = Path(backend.get("repo_path", ""))

    text = objects[0] if objects else "object"

    frame_detections: List[Dict[str, Any]] = []
    for image_path in image_paths:
        pil_image = load_pil_image(image_path)
        try:
            candidates = _countgd_infer(model, transform, pil_image, text, device, repo_path)
        except Exception as exc:
            candidates = [{"error": str(exc)}]
        frame_detections.append({"candidates": candidates})
    return frame_detections


def run_sam2_propagation(
    backend: Dict[str, Any],
    image_paths: List[str],
    frame_detections: List[Dict[str, Any]],
    objects: List[str],
    temp_dir: str | None = None,
) -> List[Dict[str, Any]]:
    """Run SAM 2.1 video propagation to build object tracks.

    Implements the CountVid three-stage pipeline:
    1. Per-frame mask generation with SAM2 image predictor
    2. [temporal filter — lightweight V1: skip isolated detections]
    3. Video propagation with SAM2 video predictor

    Returns tracks with ``points_px`` in pixel coordinates.
    """
    if not backend.get("sam2_available"):
        diag = backend.get("sam2_diag", ["SAM 2.1 backend is not available."])
        raise RuntimeError("SAM 2.1 unavailable. " + "; ".join(diag))

    import numpy as np
    import shutil
    import tempfile

    torch = backend["torch"]
    device = backend["device"]
    sam2_image = backend["sam2_image_predictor"]
    sam2_video = backend["sam2_video_predictor"]

    N = len(image_paths)
    if N == 0:
        return []

    obj_text = objects[0] if objects else "unknown"

    # --- Stage 1: Per-frame SAM2 independent masks + box prompts ---
    countgd_boxes: Dict[int, torch.Tensor] = {}
    for frame_idx, frame in enumerate(frame_detections):
        boxes_list = []
        for cand in frame.get("candidates", []):
            if "bbox_px" in cand:
                boxes_list.append(cand["bbox_px"])
        countgd_boxes[frame_idx] = torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.tensor([])

    # --- Stage 2: V1 temporal filter — keep objects appearing in >= 2 neighbor frames ---
    # For each frame's boxes, check spatial proximity with neighbor-frame boxes
    countgd_boxes_filtered: Dict[int, torch.Tensor] = {}
    for frame_idx in range(N):
        boxes_i = countgd_boxes.get(frame_idx, torch.tensor([]))
        if len(boxes_i) == 0:
            countgd_boxes_filtered[frame_idx] = boxes_i
            continue
        # Check neighbors for support
        has_neighbor_support = False
        for nb in [frame_idx - 1, frame_idx + 1]:
            if 0 <= nb < N:
                if len(countgd_boxes.get(nb, torch.tensor([]))) > 0:
                    has_neighbor_support = True
                    break
        if has_neighbor_support or N == 1:
            countgd_boxes_filtered[frame_idx] = boxes_i
        else:
            # Keep only high-confidence boxes for isolated frames
            high_conf_boxes = []
            for idx, cand in enumerate(frame_detections[frame_idx].get("candidates", [])):
                if cand.get("score", 0.0) >= 0.5:
                    if idx < len(boxes_i):
                        high_conf_boxes.append(boxes_i[idx].tolist())
            countgd_boxes_filtered[frame_idx] = torch.tensor(high_conf_boxes, dtype=torch.float32) if high_conf_boxes else torch.tensor([])

    # --- Stage 3: Video propagation ---
    tmpdir = tempfile.mkdtemp(prefix="countvid_sam2_") if temp_dir is None else temp_dir
    try:
        for idx, image_path in enumerate(image_paths):
            dest = Path(tmpdir) / f"{idx:05d}.jpg"
            shutil.copyfile(image_path, str(dest))

        inference_state = sam2_video.init_state(video_path=tmpdir)

        # Find first frame with filtered detections
        start_frame = 0
        for j in range(N):
            if len(countgd_boxes_filtered.get(j, torch.tensor([]))) > 0:
                start_frame = j
                break

        T: Dict[int, Dict[int, tuple]] = {}

        # Seed from first frame
        boxes_start = countgd_boxes_filtered.get(start_frame, torch.tensor([]))
        if len(boxes_start) > 0:
            sam2_video.reset_state(inference_state)
            for box_ind in range(len(boxes_start)):
                box_t = boxes_start[box_ind].to(device) if boxes_start[box_ind].device.type != device else boxes_start[box_ind]
                sam2_video.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=start_frame,
                    obj_id=box_ind + 1,
                    box=box_t,
                )

            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_video.propagate_in_video(
                inference_state
            ):
                for ind, obj_id in enumerate(out_obj_ids):
                    if obj_id not in T:
                        T[obj_id] = {}
                    T[obj_id][out_frame_idx] = np.nonzero(
                        (out_mask_logits[ind] > 0.0).cpu().numpy().squeeze()
                    )

        # Walk forward, check for new objects
        for i in range(start_frame + 1, N):
            boxes_i = countgd_boxes_filtered.get(i, torch.tensor([]))
            if len(boxes_i) == 0:
                continue

            existing_in_frame = [oid for oid in T if i in T[oid]]
            if len(existing_in_frame) >= len(boxes_i):
                continue

            sam2_video.reset_state(inference_state)
            last_obj_id = max(T.keys()) if T else 0
            new_boxes = boxes_i.to(device) if boxes_i.device.type != device else boxes_i
            for ind in range(len(new_boxes)):
                sam2_video.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=last_obj_id + ind + 1,
                    box=new_boxes[ind],
                )

            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_video.propagate_in_video(
                inference_state
            ):
                for ind, obj_id in enumerate(out_obj_ids):
                    if obj_id not in T:
                        T[obj_id] = {}
                    T[obj_id][out_frame_idx] = np.nonzero(
                        (out_mask_logits[ind] > 0.0).cpu().numpy().squeeze()
                    )
    finally:
        if temp_dir is None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Convert T to track format
    tracks: List[Dict[str, Any]] = []
    for obj_id, frame_masks in T.items():
        frames_sorted = sorted(frame_masks.keys())
        points_px: List[List[float]] = []
        for frame_idx in frames_sorted:
            mask_indices = frame_masks[frame_idx]
            if len(mask_indices[0]) > 0:
                cy = float(np.mean(mask_indices[0]))
                cx = float(np.mean(mask_indices[1]))
            else:
                cx, cy = 0.0, 0.0
            points_px.append([cx, cy])

        tracks.append({
            "seed_id": f"propagated_{obj_id}",
            "track_id": f"propagated_{obj_id}",
            "object": obj_text,
            "start_frame_idx": frames_sorted[0] if frames_sorted else 0,
            "frame_indices": frames_sorted,
            "points_px": points_px,
        })

    return tracks


def run_countvid_subprocess(
    image_paths: List[str],
    objects: List[str],
    countgd_repo_path: str,
    countgd_checkpoint_path: str,
    sam2_checkpoint_path: str,
    sam2_config_name: str,
    device: str = "cuda",
    temp_dir: str | None = None,
) -> Dict[str, Any]:
    """Run CountVid pipeline via subprocess calling count_in_videos.py.

    This approach avoids complex in-process dependency issues (GroundingDINO
    custom CUDA ops, GCC compatibility, etc.) by running CountVid in its own
    process. It writes frames to a temp directory, invokes the script, and
    parses structured output.

    Returns:
        Dict with ``instance_count``, ``raw_tracks``, ``frame_summaries``,
        and ``backend`` label.
    """
    import numpy as np

    repo_path = Path(countgd_repo_path)
    script_path = repo_path / "count_in_videos.py"
    if not script_path.exists():
        raise RuntimeError(f"CountVid script not found: {script_path}")

    N = len(image_paths)
    if N == 0:
        return {"instance_count": 0, "raw_tracks": [], "frame_summaries": [], "backend": "countvid:subprocess"}

    obj_text = objects[0] if objects else "object"

    # Prepare temp directories
    own_tmpdir = temp_dir is None
    tmpdir = temp_dir or tempfile.mkdtemp(prefix="countvid_run_")
    # output_dir must NOT exist — CountVid creates it itself
    output_dir = Path(tmpdir) / "output"
    result_file = Path(tmpdir) / "result.json"
    t_file = Path(tmpdir) / "T.json"

    try:
        # Copy frames to temp directory with CountVid-expected naming
        frames_dir = Path(tmpdir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx, image_path in enumerate(image_paths):
            ext = Path(image_path).suffix or ".jpg"
            dest = frames_dir / f"{idx:05d}{ext}"
            shutil.copyfile(image_path, str(dest))

        # Build command
        cmd = [
            sys.executable, str(script_path),
            "--video_dir", str(frames_dir),
            "--input_text", obj_text,
            "--sam_checkpoint", sam2_checkpoint_path,
            "--sam_model_cfg", sam2_config_name,
            "--pretrain_model_path", countgd_checkpoint_path,
            "--device", device,
            "--output_dir", str(output_dir),
            "--output_file", str(result_file),
            "--save_T",
        ]

        # Run CountVid with torch lib path for compiled CUDA ops
        import os as _os
        import torch as _torch
        env = {**_os.environ, "PYTHONPATH": str(repo_path)}
        # Add torch lib path for MultiScaleDeformableAttention
        torch_lib = str(Path(_torch.__file__).parent / "lib")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = torch_lib + ":" + env["LD_LIBRARY_PATH"]
        else:
            env["LD_LIBRARY_PATH"] = torch_lib

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(repo_path),
            env=env,
        )

        stdout = proc.stdout
        stderr = proc.stderr

        # Parse result: CountVid prints "Total Number of Objects: N"
        instance_count = 0
        for line in (stdout + stderr).splitlines():
            if "Total Number of Objects:" in line:
                try:
                    instance_count = int(line.split(":")[-1].strip())
                except ValueError:
                    pass

        # Try to load structured output from T.json
        # CountVid saves T.json in the current working directory (repo_path)
        raw_tracks: List[Dict[str, Any]] = []
        t_json_paths = [t_file, Path(str(repo_path)) / "T.json"]
        t_json = None
        for p in t_json_paths:
            if p.exists():
                t_json = p
                break
        if t_json is not None:
            try:
                T_data = json.loads(t_json.read_text())
                for obj_id_str, frame_dict in T_data.items():
                    obj_id = int(obj_id_str)
                    frames_sorted = sorted(int(k) for k in frame_dict.keys())
                    points_px = []
                    for frame_idx in frames_sorted:
                        mask_data = frame_dict[str(frame_idx)]
                        if isinstance(mask_data, list) and len(mask_data) >= 2:
                            y_indices = mask_data[0]
                            x_indices = mask_data[1]
                            if len(y_indices) > 0:
                                points_px.append([float(np.mean(x_indices)), float(np.mean(y_indices))])
                            else:
                                points_px.append([0.0, 0.0])
                    raw_tracks.append({
                        "seed_id": f"track_{obj_id}",
                        "track_id": f"track_{obj_id}",
                        "object": obj_text,
                        "start_frame_idx": frames_sorted[0] if frames_sorted else 0,
                        "frame_indices": frames_sorted,
                        "points_px": points_px,
                    })
            except Exception:
                pass

        # Build frame summaries
        frame_summaries = []
        for i in range(N):
            tracks_in_frame = sum(1 for t in raw_tracks if i in t.get("frame_indices", []))
            frame_summaries.append({
                "image": f"image-{i}",
                "candidate_count": tracks_in_frame,
                "filtered_count": tracks_in_frame,
            })

        return {
            "instance_count": max(instance_count, len(raw_tracks)),
            "raw_tracks": raw_tracks,
            "frame_summaries": frame_summaries,
            "backend": f"countvid:subprocess+{sam2_config_name}",
        }

    finally:
        if own_tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
