from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spatial_agent.tools.backends import load_pil_image, resolve_device, ROOT_DIR

# Coordinate convention: all internal structures use pixel coordinates.
#   - point_px: [x, y] in pixel space (float for sub-pixel precision)
#   - bbox_px:  [x1, y1, x2, y2] in pixel space
# Normalization to [0,1] happens only in the final payload via video_counting_utils.


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
    """Return a list of human-readable diagnostic messages for missing paths."""
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
        missing.append("sam2_config_name is not configured")
    elif not Path(sam2_config).is_file():
        missing.append(f"sam2_config_name does not exist: {sam2_config}")
    return missing


@lru_cache(maxsize=2)
def get_countvid_backend(
    countgd_repo_path: str,
    countgd_checkpoint_path: str,
    sam2_checkpoint_path: str,
    sam2_config_name: str,
    device: str,
) -> Dict[str, Any]:
    """Load CountVid backend components (CountGD-Box + SAM 2.1).

    Returns a dict with:
      - countgd_model, countgd_available, countgd_diag
      - sam2_predictor, sam2_available, sam2_diag
      - torch, device
    """
    import torch

    device = resolve_device(device)
    countgd_diag: List[str] = []
    sam2_diag: List[str] = []

    # --- CountGD-Box ---
    countgd_model = None
    countgd_available = False
    repo_path = Path(countgd_repo_path) if countgd_repo_path else None
    if repo_path and repo_path.is_dir() and countgd_checkpoint_path and Path(countgd_checkpoint_path).is_file():
        try:
            with _prepend_path(repo_path):
                from countgd.models.countgd_box import build_countgd_box
            countgd_model = build_countgd_box(countgd_checkpoint_path, device=device)
            countgd_model.eval()
            countgd_available = True
        except Exception as exc:
            countgd_diag.append(f"CountGD-Box failed to load: {exc}")
    else:
        countgd_diag = _diagnose_missing_paths(
            countgd_repo_path, countgd_checkpoint_path, None, None
        )

    # --- SAM 2.1 ---
    sam2_predictor = None
    sam2_available = False
    sam2_config_path = Path(sam2_config_name) if sam2_config_name else None
    if sam2_checkpoint_path and sam2_config_path and sam2_config_path.is_file() and Path(sam2_checkpoint_path).is_file():
        try:
            legacy_dir = ROOT_DIR / "version_0" / "SpatialAgent"
            with _prepend_path(legacy_dir):
                from sam2.build_sam import build_sam2_video_predictor
            sam2_predictor = build_sam2_video_predictor(
                str(sam2_config_path),
                sam2_checkpoint_path,
                device=device,
            )
            sam2_available = True
        except Exception as exc:
            sam2_diag.append(f"SAM 2.1 failed to load: {exc}")
    else:
        sam2_diag = _diagnose_missing_paths(
            None, None, sam2_checkpoint_path, sam2_config_name
        )

    return {
        "torch": torch,
        "countgd_model": countgd_model,
        "countgd_available": countgd_available,
        "countgd_diag": countgd_diag,
        "sam2_predictor": sam2_predictor,
        "sam2_available": sam2_available,
        "sam2_diag": sam2_diag,
        "device": device,
    }


def generate_candidates_countgd(
    backend: Dict[str, Any],
    image_paths: List[str],
    objects: List[str],
) -> List[Dict[str, Any]]:
    """Run CountGD-Box on each frame to produce per-frame candidates.

    Returns:
        List of per-frame dicts with ``candidates``, each containing:
          - ``bbox_px``: [x1, y1, x2, y2] in pixel coordinates
          - ``point_px``: [cx, cy] in pixel coordinates (center)
          - ``score``: confidence score in [0, 1]
    """
    if not backend.get("countgd_available"):
        diag = backend.get("countgd_diag", ["CountGD-Box backend is not available."])
        raise RuntimeError("CountGD-Box unavailable. " + "; ".join(diag))

    model = backend["countgd_model"]
    torch = backend["torch"]

    frame_detections: List[Dict[str, Any]] = []
    for image_path in image_paths:
        image = load_pil_image(image_path)
        candidates: List[Dict[str, Any]] = []
        try:
            with torch.no_grad():
                outputs = model(image, objects)
            if isinstance(outputs, list):
                for det in outputs:
                    cand: Dict[str, Any] = {}
                    # bbox in pixel coords
                    if hasattr(det, "bbox") or (isinstance(det, dict) and "bbox" in det):
                        bbox = det["bbox"] if isinstance(det, dict) else det.bbox
                        cand["bbox_px"] = [float(v) for v in bbox]
                    # point in pixel coords
                    if hasattr(det, "point") or (isinstance(det, dict) and "point" in det):
                        point = det["point"] if isinstance(det, dict) else det.point
                        cand["point_px"] = [float(v) for v in point]
                    elif "bbox_px" in cand:
                        x1, y1, x2, y2 = cand["bbox_px"]
                        cand["point_px"] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                    if hasattr(det, "score") or (isinstance(det, dict) and "score" in det):
                        score = det["score"] if isinstance(det, dict) else det.score
                        cand["score"] = float(score)
                    if cand:
                        candidates.append(cand)
        except Exception as exc:
            # Per-frame errors are diagnostic; don't kill the whole pipeline
            candidates.append({"error": str(exc)})
        frame_detections.append({"candidates": candidates})
    return frame_detections


def run_sam2_propagation(
    backend: Dict[str, Any],
    image_paths: List[str],
    frame_detections: List[Dict[str, Any]],
    objects: List[str],
) -> List[Dict[str, Any]]:
    """Run SAM 2.1 video propagation to build object tracks.

    Each seed gets a globally unique ``seed_id`` of the form
    ``{object}_{start_frame_idx}_{seed_idx}``.
    Points are stored as ``points_px`` in pixel coordinates.
    """
    if not backend.get("sam2_available"):
        diag = backend.get("sam2_diag", ["SAM 2.1 backend is not available."])
        raise RuntimeError("SAM 2.1 unavailable. " + "; ".join(diag))

    predictor = backend["sam2_predictor"]
    torch = backend["torch"]

    tracks: List[Dict[str, Any]] = []
    # Per-object seed counters to guarantee globally unique seed_ids
    seed_counters: Dict[str, int] = {}

    for obj_name in objects:
        if obj_name not in seed_counters:
            seed_counters[obj_name] = 0
        for frame_idx, frame in enumerate(frame_detections):
            candidates = frame.get("candidates", [])
            for cand in candidates:
                if "point_px" in cand:
                    px, py = float(cand["point_px"][0]), float(cand["point_px"][1])
                elif "bbox_px" in cand:
                    x1, y1, x2, y2 = cand["bbox_px"]
                    px = (x1 + x2) / 2.0
                    py = (y1 + y2) / 2.0
                else:
                    continue

                seed_idx = seed_counters[obj_name]
                seed_counters[obj_name] += 1
                seed_id = f"{obj_name}_{frame_idx}_{seed_idx}"

                points_px: List[List[float]] = [[px, py]]
                frame_indices: List[int] = [frame_idx]

                # Forward propagation through remaining frames
                try:
                    with torch.no_grad():
                        for t in range(frame_idx + 1, len(image_paths)):
                            out = predictor.propagate_in_video(None, t)
                            if out is not None and "points" in out:
                                pts = out["points"]
                                if isinstance(pts, torch.Tensor):
                                    pts = pts.cpu().tolist()
                                if len(pts) > 0 and len(pts[0]) >= 2:
                                    points_px.append([float(pts[0][0]), float(pts[0][1])])
                                    frame_indices.append(t)
                except Exception:
                    pass

                tracks.append(
                    {
                        "seed_id": seed_id,
                        "track_id": seed_id,
                        "object": obj_name,
                        "start_frame_idx": frame_idx,
                        "frame_indices": frame_indices,
                        "points_px": points_px,
                    }
                )
    return tracks
