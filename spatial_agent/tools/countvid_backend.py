from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spatial_agent.tools.backends import load_pil_image, resolve_device, ROOT_DIR


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


@lru_cache(maxsize=2)
def get_countvid_backend(
    countgd_repo_path: str,
    countgd_checkpoint_path: str,
    sam2_checkpoint_path: str,
    sam2_config_name: str,
    device: str,
) -> Dict[str, Any]:
    """Load CountVid backend components (CountGD-Box + SAM 2.1).

    Returns a dict with callables for per-frame counting and video propagation.
    """
    import torch

    device = resolve_device(device)

    # --- CountGD-Box (frame-level counting/detection) ---
    CountGDWrapper = None
    countgd_model = None
    countgd_available = False
    repo_path = Path(countgd_repo_path) if countgd_repo_path else None
    if repo_path and repo_path.exists():
        try:
            with _prepend_path(repo_path):
                from countgd.models.countgd_box import build_countgd_box
            countgd_model = build_countgd_box(countgd_checkpoint_path, device=device)
            countgd_model.eval()
            countgd_available = True
        except Exception:
            countgd_available = False

    # --- SAM 2.1 (video propagation) ---
    sam2_predictor = None
    sam2_available = False
    sam2_config_path = Path(sam2_config_name) if sam2_config_name else None
    if sam2_checkpoint_path and sam2_config_path and sam2_config_path.exists():
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
        except Exception:
            sam2_available = False

    return {
        "torch": torch,
        "countgd_model": countgd_model,
        "countgd_available": countgd_available,
        "sam2_predictor": sam2_predictor,
        "sam2_available": sam2_available,
        "device": device,
    }


def generate_candidates_countgd(
    backend: Dict[str, Any],
    image_paths: List[str],
    objects: List[str],
) -> List[Dict[str, Any]]:
    """Run CountGD-Box on each frame to produce per-frame candidates."""
    if not backend.get("countgd_available"):
        raise RuntimeError("CountGD-Box backend is not available.")

    model = backend["countgd_model"]
    torch = backend["torch"]
    device = backend["device"]

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
                    if hasattr(det, "bbox") or (isinstance(det, dict) and "bbox" in det):
                        bbox = det["bbox"] if isinstance(det, dict) else det.bbox
                        cand["bbox"] = [float(v) for v in bbox]
                    if hasattr(det, "point") or (isinstance(det, dict) and "point" in det):
                        point = det["point"] if isinstance(det, dict) else det.point
                        cand["point"] = [float(v) for v in point]
                    if hasattr(det, "score") or (isinstance(det, dict) and "score" in det):
                        score = det["score"] if isinstance(det, dict) else det.score
                        cand["score"] = float(score)
                    if cand:
                        candidates.append(cand)
        except Exception:
            pass
        frame_detections.append({"candidates": candidates})
    return frame_detections


def run_sam2_propagation(
    backend: Dict[str, Any],
    image_paths: List[str],
    frame_detections: List[Dict[str, Any]],
    objects: List[str],
) -> List[Dict[str, Any]]:
    """Run SAM 2.1 video propagation to build object tracks."""
    if not backend.get("sam2_available"):
        raise RuntimeError("SAM 2.1 backend is not available.")

    predictor = backend["sam2_predictor"]
    torch = backend["torch"]
    device = backend["device"]

    tracks: List[Dict[str, Any]] = []
    for obj_idx, obj_name in enumerate(objects):
        for frame_idx, frame in enumerate(frame_detections):
            candidates = frame.get("candidates", [])
            for cand_idx, cand in enumerate(candidates):
                track_id = f"{obj_name}_{cand_idx:03d}"
                if "point" in cand:
                    px, py = cand["point"]
                elif "bbox" in cand:
                    x1, y1, x2, y2 = cand["bbox"]
                    px = (x1 + x2) / 2.0
                    py = (y1 + y2) / 2.0
                else:
                    continue

                # Propagate from this frame forward/backward
                frame_indices: List[int] = [frame_idx]
                points: List[List[float]] = [[float(px), float(py)]]
                supporting_frames: List[int] = [frame_idx]

                # Forward propagation
                try:
                    with torch.no_grad():
                        for t in range(frame_idx + 1, len(image_paths)):
                            out = predictor.propagate_in_video(None, t)
                            if out is not None and "points" in out:
                                pts = out["points"]
                                if isinstance(pts, torch.Tensor):
                                    pts = pts.cpu().tolist()
                                if len(pts) > 0 and len(pts[0]) >= 2:
                                    points.append([float(pts[0][0]), float(pts[0][1])])
                                    frame_indices.append(t)
                                    supporting_frames.append(t)
                except Exception:
                    pass

                tracks.append(
                    {
                        "track_id": track_id,
                        "object": obj_name,
                        "frame_indices": supporting_frames,
                        "points": points,
                    }
                )
    return tracks
