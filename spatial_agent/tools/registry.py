from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.camera import GetCameraParametersVGGTTool
from spatial_agent.tools.counting import CountObjectsTool
from spatial_agent.tools.depth import EstimateObjectDepthTool
from spatial_agent.tools.homography import EstimateHomographyMatrixTool
from spatial_agent.tools.localization import LocalizeObjectsTool
from spatial_agent.tools.mask import GetObjectMaskTool
from spatial_agent.tools.motion import EstimateObjectMotionTool
from spatial_agent.tools.optical_flow import EstimateOpticalFlowTool
from spatial_agent.tools.orientation import GetObjectOrientationTool
from spatial_agent.tools.video_counting_3d import CountVideoObjects3DTool
from spatial_agent.tools.video_counting import CountVideoObjectsTool


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[BaseSpatialTool]] = None) -> None:
        self._tools: Dict[str, BaseSpatialTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseSpatialTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: Optional[str]) -> Optional[BaseSpatialTool]:
        if not name:
            return None
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        return sorted(self._tools)

    def to_metadata(self) -> List[Dict[str, object]]:
        return [self._tools[name].to_metadata() for name in self.list_names()]


def _video_counting_configured(config) -> bool:
    """Only register CountVideoObjects if all CountVid backend paths exist on disk."""
    from pathlib import Path
    from spatial_agent.tools.backends import get_tool_settings

    settings = get_tool_settings(config, "CountVideoObjects", aliases=["video_counting", "countvid"])
    countgd_repo = settings.get("countgd_repo_path")
    countgd_ckpt = settings.get("countgd_checkpoint_path")
    sam2_ckpt = settings.get("sam2_checkpoint_path")
    sam2_config = settings.get("sam2_config_name")

    # countgd_repo, countgd_ckpt, sam2_ckpt must exist on disk.
    # sam2_config is a Hydra config name (e.g. "sam2.1/sam2.1_hiera_l.yaml"),
    # not a file path — only check it is a non-empty string.
    if not all([countgd_repo, countgd_ckpt, sam2_ckpt, sam2_config]):
        return False
    if not Path(str(countgd_repo)).is_dir():
        return False
    if not Path(str(countgd_ckpt)).is_file():
        return False
    if not Path(str(sam2_ckpt)).is_file():
        return False
    return True


def _video_counting_3d_configured(config) -> bool:
    from spatial_agent.tools.backends import get_tool_settings

    settings = get_tool_settings(config, "CountVideoObjects3D", aliases=["video_counting_3d", "visra_counting"])
    return bool(settings)


def build_default_tool_registry(config) -> ToolRegistry:
    tools = [
        CountObjectsTool(config),
        EstimateObjectDepthTool(config),
        GetObjectMaskTool(config),
        EstimateOpticalFlowTool(config),
        GetCameraParametersVGGTTool(config),
        GetObjectOrientationTool(config),
        EstimateHomographyMatrixTool(config),
        LocalizeObjectsTool(config),
        EstimateObjectMotionTool(config),
    ]
    if _video_counting_3d_configured(config):
        tools.insert(1, CountVideoObjects3DTool(config))
    if _video_counting_configured(config):
        insert_at = 2 if _video_counting_3d_configured(config) else 1
        tools.insert(insert_at, CountVideoObjectsTool(config))
    return ToolRegistry(tools)
