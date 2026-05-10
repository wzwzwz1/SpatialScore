"""Input/output bridges for SpatialAgent integrations."""

from spatial_agent.io.lmms_bridge import build_task_input_from_vsibench_doc
from spatial_agent.io.video_sampling import sample_video_frames, select_frame_indices
from spatial_agent.io.vsibench_runner import (
    build_vsibench_video_path,
    build_vsibench_visual_task_input,
    run_vsibench_sample,
)

__all__ = [
    "build_task_input_from_vsibench_doc",
    "sample_video_frames",
    "select_frame_indices",
    "build_vsibench_video_path",
    "build_vsibench_visual_task_input",
    "run_vsibench_sample",
]
