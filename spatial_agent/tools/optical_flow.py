from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - optional runtime dependency
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    ensure_image_paths,
    get_raft_backend,
    get_tool_settings,
    load_rgb_array,
    resolve_device,
    save_optical_flow_visualization,
)
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
        image_paths = ensure_image_paths(kwargs.get("images") or kwargs.get("image"))
        if len(image_paths) < 2:
            return self.error("EstimateOpticalFlow requires at least two image paths.")

        settings = get_tool_settings(self.config, self.name, aliases=["optical_flow", "raft"])
        try:  # pragma: no cover - dependency-heavy runtime path
            device = resolve_device(settings.get("device"))
            checkpoint_path = settings.get("checkpoint_path")
            if not checkpoint_path:
                return self.unavailable("RAFT optical flow requires a configured checkpoint_path in tool_config.")

            backend = get_raft_backend(
                checkpoint_path=str(checkpoint_path),
                small=bool(settings.get("small", False)),
                mixed_precision=bool(settings.get("mixed_precision", False)),
                alternate_corr=bool(settings.get("alternate_corr", False)),
                device=device,
            )
            torch = backend["torch"]
            model = backend["model"]
            InputPadder = backend["InputPadder"]

            first = load_rgb_array(image_paths[0])
            second = load_rgb_array(image_paths[1])
            image1 = torch.from_numpy(first).permute(2, 0, 1).float()[None].to(device)
            image2 = torch.from_numpy(second).permute(2, 0, 1).float()[None].to(device)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            with torch.no_grad():
                _, flow_up = model(
                    image1,
                    image2,
                    iters=int(settings.get("iters", 20)),
                    test_mode=True,
                )

            flow = padder.unpad(flow_up[0]).permute(1, 2, 0).detach().cpu().numpy()
            mean_flow_x = float(np.mean(flow[..., 0]))
            mean_flow_y = float(np.mean(flow[..., 1]))
            magnitude = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            max_magnitude = float(np.max(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))

            artifact_path = artifact_dir_for_tool(self.config, self.name) / f"{Path(image_paths[0]).stem}_to_{Path(image_paths[1]).stem}_raft_flow.png"
            artifact = save_optical_flow_visualization(flow, artifact_path)

            return self.success(
                payload={
                    "mean_flow_x": mean_flow_x,
                    "mean_flow_y": mean_flow_y,
                    "mean_magnitude": magnitude,
                    "max_magnitude": max_magnitude,
                    "flow_shape_hw": [int(flow.shape[0]), int(flow.shape[1])],
                    "backend": "raft",
                },
                artifacts=[artifact],
            )
        except Exception as exc:  # pragma: no cover - dependency-heavy runtime path
            return self.unavailable(f"RAFT optical flow is not available or failed to initialize: {exc}")
