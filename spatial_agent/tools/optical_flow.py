from __future__ import annotations

from spatial_agent.tools.base import BaseSpatialTool


class EstimateOpticalFlowTool(BaseSpatialTool):
    name = "EstimateOpticalFlow"
    description = "Estimate optical flow statistics between two images."
    args_schema = {
        "type": "object",
        "properties": {
            "images": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        },
        "required": ["images"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config

    def invoke(self, **kwargs):
        images = kwargs.get("images") or []
        if len(images) < 2:
            return self.error("EstimateOpticalFlow requires at least two image paths.")

        try:  # pragma: no cover - optional runtime dependency
            import cv2
            import numpy as np
        except Exception:
            return self.unavailable(
                "Optical flow requires OpenCV or RAFT dependencies, which are not available in the current environment."
            )

        first = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(images[1], cv2.IMREAD_GRAYSCALE)
        if first is None or second is None:
            return self.error("Failed to load one or more input images for optical flow estimation.")

        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mean_flow_x = float(np.mean(flow[..., 0]))
        mean_flow_y = float(np.mean(flow[..., 1]))
        magnitude = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        return self.success(
            payload={
                "mean_flow_x": mean_flow_x,
                "mean_flow_y": mean_flow_y,
                "mean_magnitude": magnitude,
                "backend": "opencv_farneback",
            }
        )
