from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from spatial_agent.tools.backends import load_pil_image

# Coordinate convention (Phase 0 unified):
#   Internal (point_px, bbox_px): pixel coordinates relative to the original image.
#     - point_px: [x, y] where x ∈ [0, width), y ∈ [0, height)
#     - bbox_px:   [x1, y1, x2, y2] in pixel space
#   Output (supporting_points): normalized [0, 1] for LLM consumption.
#   All overlay/drawing functions accept pixel coordinates directly.


def temporal_window_filter(
    frame_detections: List[Dict[str, Any]],
    window_size: int = 3,
) -> List[Dict[str, Any]]:
    """Remove one-frame spikes: keep detections with neighbor-frame support."""
    if not frame_detections:
        return []
    num_frames = len(frame_detections)
    half = window_size // 2
    filtered: List[Dict[str, Any]] = []
    for i, frame in enumerate(frame_detections):
        start = max(0, i - half)
        end = min(num_frames, i + half + 1)
        neighbor_has_detection = any(
            len(frame_detections[j].get("candidates", [])) > 0
            for j in range(start, end)
            if j != i
        )
        candidates = frame.get("candidates", [])
        if candidates or neighbor_has_detection:
            filtered.append({**frame, "candidates": candidates})
        else:
            filtered.append({**frame, "candidates": [], "filtered": True})
    return filtered


def build_track_payload(
    tracks: List[Dict[str, Any]],
    image_aliases: List[str],
    image_sizes: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Format raw tracks (pixel coords) into the standardized output schema (normalized).

    Args:
        tracks: Raw tracks with ``points_px`` in pixel coordinates.
        image_aliases: Ordered list of ``"image-N"`` aliases.
        image_sizes: Ordered list of ``(width, height)`` per frame, for normalization.

    Returns:
        Tracks with ``supporting_points`` in normalized [0, 1] coordinates.
    """
    formatted: List[Dict[str, Any]] = []
    for index, track in enumerate(tracks):
        supporting_frames: List[str] = []
        supporting_points: List[List[float]] = []
        for frame_idx in track.get("frame_indices", []):
            if 0 <= frame_idx < len(image_aliases):
                supporting_frames.append(image_aliases[frame_idx])
        for pt_idx, point in enumerate(track.get("points_px", [])):
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                frame_idx = track.get("frame_indices", [pt_idx])[0] if pt_idx < len(track.get("frame_indices", [])) else 0
                w, h = (1, 1)
                if 0 <= frame_idx < len(image_sizes):
                    w, h = image_sizes[frame_idx]
                x_norm = round(max(0.0, min(1.0, float(point[0]) / max(1, w))), 6)
                y_norm = round(max(0.0, min(1.0, float(point[1]) / max(1, h))), 6)
                supporting_points.append([x_norm, y_norm])
        formatted.append(
            {
                "track_id": track.get("track_id", f"object_{index:03d}"),
                "object": track.get("object", "unknown"),
                "supporting_frames": supporting_frames,
                "supporting_points": supporting_points,
            }
        )
    return formatted


def build_frame_summaries(
    frame_detections: List[Dict[str, Any]],
    image_aliases: List[str],
) -> List[Dict[str, Any]]:
    """Build per-frame summary dicts for the output payload."""
    summaries: List[Dict[str, Any]] = []
    for i, frame in enumerate(frame_detections):
        candidates = frame.get("candidates", [])
        alias = image_aliases[i] if i < len(image_aliases) else f"frame-{i}"
        summaries.append(
            {
                "image": alias,
                "candidate_count": len(candidates),
                "filtered_count": len(candidates) if not frame.get("filtered") else 0,
            }
        )
    return summaries


def aggregate_unique_tracks(
    tracks: List[Dict[str, Any]],
    min_track_support: int = 1,
) -> List[Dict[str, Any]]:
    """Filter tracks by minimum frame support threshold."""
    return [
        track
        for track in tracks
        if len(track.get("frame_indices", [])) >= min_track_support
    ]


def save_track_overlay(
    frame_paths: List[str],
    tracks: List[Dict[str, Any]],
    output_path: Path,
) -> str:
    """Visualize propagated tracks on sampled frames.

    Expects ``points_px`` in pixel coordinates on each track.
    """
    if not frame_paths or not tracks:
        return str(output_path)

    frames = [load_pil_image(p) for p in frame_paths]
    palette = [
        (255, 90, 90),
        (90, 200, 255),
        (100, 255, 120),
        (255, 210, 90),
        (190, 130, 255),
        (255, 150, 200),
        (150, 255, 200),
        (200, 180, 255),
    ]

    rendered = []
    for frame_idx, frame in enumerate(frames):
        canvas = frame.copy()
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        for track_idx, track in enumerate(tracks):
            color = palette[track_idx % len(palette)]
            frame_indices = track.get("frame_indices", [])
            points_px = track.get("points_px", [])
            if frame_idx in frame_indices:
                pt_idx = frame_indices.index(frame_idx)
                if pt_idx < len(points_px):
                    x = max(0.0, min(float(w - 1), float(points_px[pt_idx][0])))
                    y = max(0.0, min(float(h - 1), float(points_px[pt_idx][1])))
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="black", width=1)
                    label = track.get("track_id", f"T{track_idx}")
                    draw.text((x + 8, y - 8), label, fill=color)
        rendered.append(canvas)

    total_width = sum(f.width for f in rendered)
    max_height = max(f.height for f in rendered)
    grid = Image.new("RGB", (total_width, max_height))
    cursor = 0
    for f in rendered:
        grid.paste(f, (cursor, 0))
        cursor += f.width
    grid.save(output_path)
    return str(output_path)


def save_candidate_overlay(
    frame_paths: List[str],
    frame_detections: List[Dict[str, Any]],
    output_path: Path,
) -> str:
    """Visualize per-frame candidate detections.

    Expects candidates with ``bbox_px`` (pixel) and ``point_px`` (pixel).
    """
    if not frame_paths:
        return str(output_path)

    frames = [load_pil_image(p) for p in frame_paths]
    palette = [
        (255, 90, 90),
        (90, 200, 255),
        (100, 255, 120),
        (255, 210, 90),
    ]

    rendered = []
    for frame_idx, frame in enumerate(frames):
        canvas = frame.copy()
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        detections = frame_detections[frame_idx] if frame_idx < len(frame_detections) else {}
        candidates = detections.get("candidates", [])
        for ci, cand in enumerate(candidates):
            color = palette[ci % len(palette)]
            if "bbox_px" in cand:
                x1, y1, x2, y2 = cand["bbox_px"]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            if "point_px" in cand:
                px = max(0.0, min(float(w - 1), float(cand["point_px"][0])))
                py = max(0.0, min(float(h - 1), float(cand["point_px"][1])))
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline="black", width=1)
        label = f"frame {frame_idx}: {len(candidates)} candidates"
        draw.rectangle((4, 4, 4 + len(label) * 7, 22), fill=(255, 255, 255))
        draw.text((8, 6), label, fill="black")
        rendered.append(canvas)

    total_width = sum(f.width for f in rendered)
    max_height = max(f.height for f in rendered)
    grid = Image.new("RGB", (total_width, max_height))
    cursor = 0
    for f in rendered:
        grid.paste(f, (cursor, 0))
        cursor += f.width
    grid.save(output_path)
    return str(output_path)
