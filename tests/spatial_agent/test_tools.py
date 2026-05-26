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
    """CountVideoObjects should return unavailable when backend init fails."""
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 24), "white").save(image_path)

    def _raise_backend(**kwargs):
        raise ModuleNotFoundError("No module named 'countgd'")

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.get_countvid_backend",
        _raise_backend,
    )

    config = SpatialAgentConfig(
        artifact_dir=str(tmp_path),
        tool_config={
            "video_counting": {
                "countgd_repo_path": "/nonexistent",
                "countgd_checkpoint_path": "/nonexistent.pth",
                "sam2_checkpoint_path": "/nonexistent.pt",
                "sam2_config_name": "/nonexistent.yaml",
                "device": "cpu",
            }
        },
    )
    tool = CountVideoObjectsTool(config)
    result = tool.invoke(images=[str(image_path)], objects=["table"])

    assert result["status"] == "unavailable"
    assert "No module named 'countgd'" in result["error"]


def test_count_video_objects_returns_unavailable_when_partial_backend(tmp_path, monkeypatch):
    """CountVideoObjects should return unavailable when one component is missing."""
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 24), "white").save(image_path)

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.get_countvid_backend",
        lambda **kwargs: {
            "countgd_available": True,
            "sam2_available": False,
            "device": "cpu",
        },
    )

    config = SpatialAgentConfig(
        artifact_dir=str(tmp_path),
        tool_config={
            "video_counting": {
                "countgd_repo_path": "/nonexistent",
                "countgd_checkpoint_path": "/nonexistent.pth",
                "sam2_checkpoint_path": "/nonexistent.pt",
                "sam2_config_name": "/nonexistent.yaml",
                "device": "cpu",
            }
        },
    )
    tool = CountVideoObjectsTool(config)
    result = tool.invoke(images=[str(image_path)], objects=["table"])

    assert result["status"] == "unavailable"
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
    """Full pipeline test with mocked backend components."""
    # Create dummy frames
    frame_paths = []
    for i in range(3):
        p = tmp_path / f"frame_{i}.jpg"
        Image.new("RGB", (64, 48), "white").save(p)
        frame_paths.append(str(p))

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.get_countvid_backend",
        lambda **kwargs: {
            "countgd_model": None,
            "countgd_available": True,
            "sam2_predictor": None,
            "sam2_available": True,
            "torch": type("DummyTorch", (), {"no_grad": lambda: type("DummyCtx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()})(),
            "device": "cpu",
        },
    )

    # Mock candidate generation (pixel coords, image is 64x48)
    def _mock_generate(backend, image_paths, objects):
        return [
            {"candidates": [{"point_px": [6.4, 9.6], "score": 0.9}]},
            {"candidates": [{"point_px": [7.68, 10.56], "score": 0.85}]},
            {"candidates": []},
        ]

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.generate_candidates_countgd",
        _mock_generate,
    )

    # Mock SAM propagation (returns points_px in pixel coords)
    def _mock_propagate(backend, image_paths, frame_detections, objects):
        return [
            {
                "track_id": "table_000",
                "seed_id": "table_0_0",
                "object": "table",
                "start_frame_idx": 0,
                "frame_indices": [0, 1],
                "points_px": [[6.4, 9.6], [7.68, 10.56]],
            }
        ]

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.run_sam2_propagation",
        _mock_propagate,
    )

    config = SpatialAgentConfig(
        artifact_dir=str(tmp_path),
        tool_config={
            "video_counting": {
                "countgd_repo_path": "/tmp/fake",
                "countgd_checkpoint_path": "/tmp/fake.pth",
                "sam2_checkpoint_path": "/tmp/fake.pt",
                "sam2_config_name": "/tmp/fake.yaml",
                "device": "cpu",
            }
        },
    )
    tool = CountVideoObjectsTool(config)
    result = tool.invoke(images=frame_paths, objects=["table"])

    assert result["status"] == "success"
    assert result["payload"]["instance_count"] == 1
    assert len(result["payload"]["tracks"]) == 1
    assert result["payload"]["tracks"][0]["track_id"] == "table_000"
    assert result["payload"]["tracks"][0]["object"] == "table"
    assert "image-0" in result["payload"]["tracks"][0]["supporting_frames"]
    assert "image-1" in result["payload"]["tracks"][0]["supporting_frames"]
    assert len(result["payload"]["frame_summaries"]) == 3
    assert result["payload"]["backend"] == "countvid:countgd_box+sam2.1"
    assert len(result["artifacts"]) >= 1


def test_count_video_objects_zero_instance_success(tmp_path, monkeypatch):
    """CountVideoObjects should return success with instance_count=0 when no objects found."""
    frame_paths = []
    for i in range(2):
        p = tmp_path / f"frame_{i}.jpg"
        Image.new("RGB", (64, 48), "white").save(p)
        frame_paths.append(str(p))

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.get_countvid_backend",
        lambda **kwargs: {
            "countgd_model": None,
            "countgd_available": True,
            "sam2_predictor": None,
            "sam2_available": True,
            "torch": type("DummyTorch", (), {"no_grad": lambda: type("DummyCtx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()})(),
            "device": "cpu",
        },
    )
    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.generate_candidates_countgd",
        lambda backend, image_paths, objects: [{"candidates": []}, {"candidates": []}],
    )
    monkeypatch.setattr(
        "spatial_agent.tools.video_counting.run_sam2_propagation",
        lambda backend, image_paths, frame_detections, objects: [],
    )

    config = SpatialAgentConfig(
        artifact_dir=str(tmp_path),
        tool_config={
            "video_counting": {
                "countgd_repo_path": "/tmp/fake",
                "countgd_checkpoint_path": "/tmp/fake.pth",
                "sam2_checkpoint_path": "/tmp/fake.pt",
                "sam2_config_name": "/tmp/fake.yaml",
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
    """CountVideoObjects should be registered when all CountVid paths exist on disk."""
    repo_dir = tmp_path / "CountVid"
    repo_dir.mkdir()
    ckpt_file = tmp_path / "countgd_box.pth"
    ckpt_file.write_text("dummy")
    sam2_ckpt = tmp_path / "sam2.1_hiera_large.pt"
    sam2_ckpt.write_text("dummy")
    sam2_config = tmp_path / "sam2.1_hiera_l.yaml"
    sam2_config.write_text("dummy")

    config = SpatialAgentConfig(
        tool_config={
            "video_counting": {
                "countgd_repo_path": str(repo_dir),
                "countgd_checkpoint_path": str(ckpt_file),
                "sam2_checkpoint_path": str(sam2_ckpt),
                "sam2_config_name": str(sam2_config),
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
                "sam2_config_name": str(cfg),
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
                "sam2_config_name": str(cfg),
            }
        }
    )
    registry = build_default_tool_registry(config)
    assert "CountVideoObjects" not in registry.list_names()
