from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from spatial_agent.tools.backends import load_pil_image


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
) -> List[Dict[str, Any]]:
    """Format raw tracks into the standardized output schema."""
    formatted: List[Dict[str, Any]] = []
    for index, track in enumerate(tracks):
        supporting_frames = []
        supporting_points = []
        for frame_idx in track.get("frame_indices", []):
            if 0 <= frame_idx < len(image_aliases):
                supporting_frames.append(image_aliases[frame_idx])
        for point in track.get("points", []):
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                supporting_points.append([round(float(point[0]), 6), round(float(point[1]), 6)])
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
    """Visualize propagated tracks on sampled frames."""
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
            points = track.get("points", [])
            if frame_idx in frame_indices:
                pt_idx = frame_indices.index(frame_idx)
                if pt_idx < len(points):
                    x_norm, y_norm = float(points[pt_idx][0]), float(points[pt_idx][1])
                    x = max(0.0, min(float(w - 1), x_norm * w))
                    y = max(0.0, min(float(h - 1), y_norm * h))
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
    """Visualize per-frame candidate detections."""
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
            if "bbox" in cand:
                x1, y1, x2, y2 = cand["bbox"]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            if "point" in cand:
                px_norm, py_norm = float(cand["point"][0]), float(cand["point"][1])
                px = max(0.0, min(float(w - 1), px_norm * w))
                py = max(0.0, min(float(h - 1), py_norm * h))
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline="black", width=1)
        # summary label
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
