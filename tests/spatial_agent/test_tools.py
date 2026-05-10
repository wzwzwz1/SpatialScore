from pathlib import Path

from PIL import Image

from spatial_agent.runtime.config import SpatialAgentConfig
from spatial_agent.tools.localization import LocalizeObjectsTool
from spatial_agent.tools.placeholders import PlaceholderTool
from spatial_agent.tools.registry import build_default_tool_registry


def test_placeholder_tool_returns_unavailable_status():
    tool = PlaceholderTool(
        name="MissingTool",
        description="Not implemented yet.",
        reason="Tool is not available in the current release.",
    )

    result = tool.invoke()

    assert result["status"] == "unavailable"
    assert result["tool_name"] == "MissingTool"
    assert "current release" in result["error"]


def test_default_registry_tools_return_structured_results():
    registry = build_default_tool_registry(SpatialAgentConfig())

    invocations = {
        "EstimateObjectDepth": {"image": "missing.jpg", "objects": ["cat"]},
        "GetObjectMask": {"image": "missing.jpg", "objects": ["cat"]},
        "EstimateOpticalFlow": {"images": ["missing-a.jpg", "missing-b.jpg"]},
        "GetCameraParametersVGGT": {"image": ["missing.jpg"]},
        "GetObjectOrientation": {"image": "missing.jpg", "objects": "person"},
        "EstimateHomographyMatrix": {"image": ["missing-a.jpg", "missing-b.jpg"]},
        "LocalizeObjects": {"image": "missing.jpg", "objects": ["cat"]},
        "EstimateObjectMotion": {"images": ["missing-a.jpg", "missing-b.jpg"], "objects": ["cat"]},
    }

    for tool_name in registry.list_names():
        tool = registry.get(tool_name)
        result = tool.invoke(**invocations[tool_name])
        assert result["tool_name"] == tool_name
        assert result["status"] in {"success", "error", "unavailable"}
        assert "payload" in result
        assert "artifacts" in result
        assert "error" in result


def test_localize_objects_returns_instance_count_and_bbox_artifact(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 48), "white").save(image_path)

    class DummyInputs(dict):
        def to(self, _device):
            return self

    class DummyTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class DummyScore(float):
        def __new__(cls, value):
            return float.__new__(cls, value)

    class DummyProcessor:
        def __call__(self, **kwargs):
            return DummyInputs({"input_ids": [1, 2, 3]})

        def post_process_grounded_object_detection(self, outputs, input_ids, box_threshold, text_threshold, target_sizes):
            return [
                {
                    "boxes": [DummyTensor([5, 6, 30, 36]), DummyTensor([32, 8, 58, 42])],
                    "scores": [DummyScore(0.9), DummyScore(0.8)],
                    "labels": ["chair", "chair"],
                }
            ]

    class DummyModel:
        def __call__(self, **kwargs):
            return object()

    class DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyTorch:
        def no_grad(self):
            return DummyNoGrad()

    monkeypatch.setattr(
        "spatial_agent.tools.localization.get_grounding_backend",
        lambda model_id, device: {
            "processor": DummyProcessor(),
            "model": DummyModel(),
            "torch": DummyTorch(),
        },
    )
    monkeypatch.setattr("spatial_agent.tools.localization.resolve_device", lambda device=None: "cpu")

    config = SpatialAgentConfig(artifact_dir=str(tmp_path))
    tool = LocalizeObjectsTool(config)

    result = tool.invoke(image=str(image_path), objects=["chair"])

    assert result["status"] == "success"
    assert result["payload"]["instance_count"] == 2
    assert len(result["artifacts"]) == 1
    assert Path(result["artifacts"][0]).exists()
