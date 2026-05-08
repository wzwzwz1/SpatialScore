"""Input/output bridges for SpatialAgent integrations."""

from spatial_agent.io.lmms_bridge import build_task_input_from_vsibench_doc
from spatial_agent.io.video_sampling import sample_video_frames, select_frame_indices

__all__ = [
    "build_task_input_from_vsibench_doc",
    "sample_video_frames",
    "select_frame_indices",
]
