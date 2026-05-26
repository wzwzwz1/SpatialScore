from pathlib import Path

from PIL import Image

from spatial_agent.runtime.config import SpatialAgentConfig
from spatial_agent.tools.counting import CountObjectsTool
from spatial_agent.tools.localization import LocalizeObjectsTool
from spatial_agent.tools.placeholders import PlaceholderTool
from spatial_agent.tools.registry import build_default_tool_registry
from spatial_agent.tools.video_counting import CountVideoObjectsTool


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
    assert "CountObjects" in registry.list_names()

    invocations = {
        "CountObjects": {"image": "missing.jpg", "objects": ["cat"]},
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


def test_count_objects_returns_points_and_artifact(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 80), "white").save(image_path)
    captured_kwargs = {}

    class DummyRex:
        def inference(self, *, images, task, categories):
            assert task == "pointing"
            assert categories == ["chair"]
            assert images.size == (100, 80)
            return [
                {
                    "extracted_predictions": {
                        "chair": [
                            {"type": "point", "coords": [10, 20]},
                            {"type": "point", "coords": [50, 60]},
                        ]
                    }
                }
            ]

    monkeypatch.setattr(
        "spatial_agent.tools.counting.get_rex_omni_backend",
        lambda **kwargs: captured_kwargs.update(kwargs) or {"wrapper": DummyRex(), "backend_label": "rex_omni:IDEA-Research/Rex-Omni"},
    )

    tool = CountObjectsTool(SpatialAgentConfig(artifact_dir=str(tmp_path)))
    result = tool.invoke(image=str(image_path), objects=["chair"])

    assert result["status"] == "success"
    assert captured_kwargs["attn_implementation"] == "sdpa"
    assert result["payload"]["instance_count"] == 2
    assert result["payload"]["points"] == {"chair": [[0.1, 0.25], [0.5, 0.75]]}
    assert len(result["artifacts"]) == 1
    assert Path(result["artifacts"][0]).exists()


def test_count_objects_returns_unavailable_when_rex_omni_missing(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 24), "white").save(image_path)

    def _raise_backend(**kwargs):
        raise ModuleNotFoundError("No module named 'rex_omni'")

    monkeypatch.setattr("spatial_agent.tools.counting.get_rex_omni_backend", _raise_backend)

    tool = CountObjectsTool(SpatialAgentConfig(artifact_dir=str(tmp_path)))
    result = tool.invoke(image=str(image_path), objects=["chair"])

    assert result["status"] == "unavailable"
    assert "rex_omni" in result["error"].lower()


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


def test_count_video_objects_returns_unavailable_when_backend_missing(tmp_path, monkeypatch):
    """CountVideoObjects should return unavailable when CountVid script not found."""
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 24), "white").save(image_path)

    def _raise_not_found(**kwargs):
        raise FileNotFoundError("No such file: count_in_videos.py")

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.run_countvid_subprocess",
        _raise_not_found,
    )

    config = SpatialAgentConfig(
        artifact_dir=str(tmp_path),
        tool_config={
            "video_counting": {
                "countgd_repo_path": "/nonexistent",
                "countgd_checkpoint_path": "/nonexistent.pth",
                "sam2_checkpoint_path": "/nonexistent.pt",
                "sam2_config_name": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "device": "cpu",
            }
        },
    )
    tool = CountVideoObjectsTool(config)
    result = tool.invoke(images=[str(image_path)], objects=["table"])

    assert result["status"] == "unavailable"
    assert "not found" in result["error"].lower()


def test_count_video_objects_returns_unavailable_when_partial_backend(tmp_path, monkeypatch):
    """CountVideoObjects should return error when subprocess fails."""
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 24), "white").save(image_path)

    def _raise_error(**kwargs):
        raise RuntimeError("CountVid subprocess failed: SAM 2.1 not available")

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.run_countvid_subprocess",
        _raise_error,
    )

    config = SpatialAgentConfig(
        artifact_dir=str(tmp_path),
        tool_config={
            "video_counting": {
                "countgd_repo_path": "/nonexistent",
                "countgd_checkpoint_path": "/nonexistent.pth",
                "sam2_checkpoint_path": "/nonexistent.pt",
                "sam2_config_name": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "device": "cpu",
            }
        },
    )
    tool = CountVideoObjectsTool(config)
    result = tool.invoke(images=[str(image_path)], objects=["table"])

    assert result["status"] == "error"
    assert "SAM 2.1" in result["error"]


def test_count_video_objects_returns_error_on_missing_images():
    tool = CountVideoObjectsTool(SpatialAgentConfig())
    result = tool.invoke(objects=["table"])
    assert result["status"] == "error"
    assert "image" in result["error"].lower()


def test_count_video_objects_returns_error_on_missing_objects():
    tool = CountVideoObjectsTool(SpatialAgentConfig())
    result = tool.invoke(images=["frame.jpg"])
    assert result["status"] == "error"
    assert "object" in result["error"].lower()


def test_count_video_objects_success_with_mock_backend(tmp_path, monkeypatch):
    """Full pipeline test with mocked subprocess backend."""
    frame_paths = []
    for i in range(3):
        p = tmp_path / f"frame_{i}.jpg"
        Image.new("RGB", (64, 48), "white").save(p)
        frame_paths.append(str(p))

    def _mock_subprocess(**kwargs):
        return {
            "instance_count": 1,
            "raw_tracks": [
                {
                    "track_id": "track_1",
                    "seed_id": "track_1",
                    "object": "table",
                    "start_frame_idx": 0,
                    "frame_indices": [0, 1],
                    "points_px": [[6.4, 9.6], [7.68, 10.56]],
                }
            ],
            "frame_summaries": [
                {"image": "image-0", "candidate_count": 1, "filtered_count": 1},
                {"image": "image-1", "candidate_count": 1, "filtered_count": 1},
                {"image": "image-2", "candidate_count": 0, "filtered_count": 0},
            ],
            "backend": "countvid:subprocess+test",
        }

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.run_countvid_subprocess",
        _mock_subprocess,
    )

    config = SpatialAgentConfig(
        artifact_dir=str(tmp_path),
        tool_config={
            "video_counting": {
                "countgd_repo_path": "/tmp/fake",
                "countgd_checkpoint_path": "/tmp/fake.pth",
                "sam2_checkpoint_path": "/tmp/fake.pt",
                "sam2_config_name": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "device": "cpu",
            }
        },
    )
    tool = CountVideoObjectsTool(config)
    result = tool.invoke(images=frame_paths, objects=["table"])

    assert result["status"] == "success"
    assert result["payload"]["instance_count"] == 1
    assert len(result["payload"]["tracks"]) == 1
    assert result["payload"]["tracks"][0]["track_id"] == "track_1"
    assert result["payload"]["tracks"][0]["object"] == "table"
    assert "image-0" in result["payload"]["tracks"][0]["supporting_frames"]
    assert "image-1" in result["payload"]["tracks"][0]["supporting_frames"]
    assert len(result["payload"]["frame_summaries"]) == 3
    assert "countvid" in result["payload"]["backend"]
    assert len(result["artifacts"]) >= 1


def test_count_video_objects_zero_instance_success(tmp_path, monkeypatch):
    """CountVideoObjects should return success with instance_count=0 when no objects found."""
    frame_paths = []
    for i in range(2):
        p = tmp_path / f"frame_{i}.jpg"
        Image.new("RGB", (64, 48), "white").save(p)
        frame_paths.append(str(p))

    def _mock_subprocess(**kwargs):
        return {
            "instance_count": 0,
            "raw_tracks": [],
            "frame_summaries": [
                {"image": "image-0", "candidate_count": 0, "filtered_count": 0},
                {"image": "image-1", "candidate_count": 0, "filtered_count": 0},
            ],
            "backend": "countvid:subprocess+test",
        }

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.run_countvid_subprocess",
        _mock_subprocess,
    )

    config = SpatialAgentConfig(
        artifact_dir=str(tmp_path),
        tool_config={
            "video_counting": {
                "countgd_repo_path": "/tmp/fake",
                "countgd_checkpoint_path": "/tmp/fake.pth",
                "sam2_checkpoint_path": "/tmp/fake.pt",
                "sam2_config_name": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "device": "cpu",
            }
        },
    )
    tool = CountVideoObjectsTool(config)
    result = tool.invoke(images=frame_paths, objects=["table"])

    assert result["status"] == "success"
    assert result["payload"]["instance_count"] == 0
    assert result["payload"]["tracks"] == []


def test_count_video_objects_not_registered_without_backend_config():
    """CountVideoObjects should be omitted when CountVid backend is not configured."""
    registry = build_default_tool_registry(SpatialAgentConfig())
    assert "CountVideoObjects" not in registry.list_names()


def test_count_video_objects_registered_with_backend_config(tmp_path):
    """CountVideoObjects should be registered when CountVid paths exist on disk."""
    repo_dir = tmp_path / "CountVid"
    repo_dir.mkdir()
    ckpt_file = tmp_path / "countgd_box.pth"
    ckpt_file.write_text("dummy")
    sam2_ckpt = tmp_path / "sam2.1_hiera_large.pt"
    sam2_ckpt.write_text("dummy")

    config = SpatialAgentConfig(
        tool_config={
            "video_counting": {
                "countgd_repo_path": str(repo_dir),
                "countgd_checkpoint_path": str(ckpt_file),
                "sam2_checkpoint_path": str(sam2_ckpt),
                "sam2_config_name": "sam2.1/sam2.1_hiera_l.yaml",
            }
        }
    )
    registry = build_default_tool_registry(config)
    assert "CountVideoObjects" in registry.list_names()
    tool = registry.get("CountVideoObjects")
    assert tool is not None
    assert tool.name == "CountVideoObjects"
    names = registry.list_names()
    assert names[0] == "CountObjects"
    assert names[1] == "CountVideoObjects"


def test_count_video_objects_not_registered_when_repo_missing(tmp_path):
    """CountVideoObjects should not register when countgd_repo_path doesn't exist."""
    ckpt = tmp_path / "ckpt.pth"
    ckpt.write_text("dummy")
    sam2_ckpt = tmp_path / "sam2.pt"
    sam2_ckpt.write_text("dummy")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("dummy")

    config = SpatialAgentConfig(
        tool_config={
            "video_counting": {
                "countgd_repo_path": str(tmp_path / "nonexistent_repo"),
                "countgd_checkpoint_path": str(ckpt),
                "sam2_checkpoint_path": str(sam2_ckpt),
                "sam2_config_name": "sam2.1/sam2.1_hiera_l.yaml",
            }
        }
    )
    registry = build_default_tool_registry(config)
    assert "CountVideoObjects" not in registry.list_names()


def test_count_video_objects_not_registered_when_checkpoint_missing(tmp_path):
    """CountVideoObjects should not register when a checkpoint file doesn't exist."""
    repo = tmp_path / "repo"
    repo.mkdir()
    sam2_ckpt = tmp_path / "sam2.pt"
    sam2_ckpt.write_text("dummy")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("dummy")

    config = SpatialAgentConfig(
        tool_config={
            "video_counting": {
                "countgd_repo_path": str(repo),
                "countgd_checkpoint_path": str(tmp_path / "missing.pth"),
                "sam2_checkpoint_path": str(sam2_ckpt),
                "sam2_config_name": "sam2.1/sam2.1_hiera_l.yaml",
            }
        }
    )
    registry = build_default_tool_registry(config)
    assert "CountVideoObjects" not in registry.list_names()
