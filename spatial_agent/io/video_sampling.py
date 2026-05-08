from __future__ import annotations

from pathlib import Path
from typing import List


def select_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if total_frames <= num_frames:
        return list(range(total_frames))
    if num_frames == 1:
        return [0]

    stride = (total_frames - 1) / float(num_frames - 1)
    indices = []
    for index in range(num_frames):
        candidate = round(index * stride)
        if not indices or candidate != indices[-1]:
            indices.append(candidate)
    if indices[-1] != total_frames - 1:
        indices[-1] = total_frames - 1
    return indices


def sample_video_frames(video_path: str, output_dir: str, num_frames: int) -> List[str]:
    try:  # pragma: no cover - optional runtime dependency
        import cv2
    except Exception as exc:  # pragma: no cover - runtime-specific
        raise RuntimeError("Video frame sampling requires OpenCV (cv2) to be installed.") from exc

    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Unable to read frame count from video: {video_path}")

    selected_indices = set(select_frame_indices(total_frames, num_frames))
    sampled_paths: List[str] = []
    current_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if current_index in selected_indices:
            output_path = destination / f"{source.stem}_frame_{current_index:05d}.jpg"
            cv2.imwrite(str(output_path), frame)
            sampled_paths.append(str(output_path))
        current_index += 1

    capture.release()

    if not sampled_paths:
        raise RuntimeError(f"No frames were sampled from video: {video_path}")
    return sampled_paths
