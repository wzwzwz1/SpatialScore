from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

from spatial_agent.io.lmms_bridge import build_task_input_from_vsibench_doc
from spatial_agent.io.video_sampling import sample_video_frames


DEFAULT_VSIBENCH_DATASET = "nyu-visionx/VSI-Bench"


def _load_datasets_module():
    try:
        import datasets
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Running a VSI-Bench sample requires the `datasets` package to be installed.") from exc
    return datasets


def build_vsibench_video_path(dataset: str, scene_name: str, cache_dir: str) -> str:
    return str(Path(cache_dir) / dataset / f"{scene_name}.mp4")


def build_vsibench_visual_task_input(
    doc: Dict[str, Any],
    video_path: str,
    task_id: str,
    artifact_dir: str,
    split: str,
    doc_id: int,
    num_frames: int,
    video_frame_dir: str | None,
) -> Tuple[Dict[str, Any], Path]:
    frame_root = Path(video_frame_dir or artifact_dir) / "sampled_frames" / "vsibench" / split / str(doc_id)
    frame_paths = sample_video_frames(
        video_path=video_path,
        output_dir=str(frame_root),
        num_frames=num_frames,
    )
    task_input = build_task_input_from_vsibench_doc(
        doc=doc,
        image_paths=frame_paths,
        task_id=task_id,
    )
    return task_input, frame_root


def run_vsibench_sample(
    agent,
    dataset_split: str,
    doc_id: int,
    num_frames: int,
    artifact_dir: str,
    keep_video_frames: bool,
    dataset_cache_dir: str,
    dataset_name: str = DEFAULT_VSIBENCH_DATASET,
    video_frame_dir: str | None = None,
    token: bool | str = True,
) -> Dict[str, Any]:
    datasets = _load_datasets_module()
    dataset = datasets.load_dataset(
        dataset_name,
        split=dataset_split,
        token=token,
        cache_dir=dataset_cache_dir,
    )
    doc = dict(dataset[int(doc_id)])
    task_id = f"vsibench___{dataset_split}___{doc_id}"
    video_path = build_vsibench_video_path(
        dataset=doc["dataset"],
        scene_name=doc["scene_name"],
        cache_dir=dataset_cache_dir,
    )
    if not Path(video_path).exists():
        raise FileNotFoundError(f"VSI-Bench video does not exist: {video_path}")

    task_input, frame_root = build_vsibench_visual_task_input(
        doc=doc,
        video_path=video_path,
        task_id=task_id,
        artifact_dir=artifact_dir,
        split=dataset_split,
        doc_id=doc_id,
        num_frames=num_frames,
        video_frame_dir=video_frame_dir,
    )
    result = agent.invoke(task_input)

    if not keep_video_frames:
        shutil.rmtree(frame_root, ignore_errors=True)

    return {
        "doc": doc,
        "video_path": video_path,
        "task_input": task_input,
        "result": result,
    }


def resolve_vsibench_cache_dir(explicit_cache_dir: str | None) -> str:
    if explicit_cache_dir:
        return explicit_cache_dir
    hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
    base_cache_dir = os.path.expanduser(hf_home)
    return os.path.join(base_cache_dir, "vsibench")
