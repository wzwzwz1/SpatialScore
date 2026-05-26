import json
from pathlib import Path

from PIL import Image

from spatial_agent.runtime.config import SpatialAgentConfig
from spatial_agent.tools.counting import CountObjectsTool
from spatial_agent.tools.localization import LocalizeObjectsTool
from spatial_agent.tools.placeholders import PlaceholderTool
from spatial_agent.tools.registry import build_default_tool_registry
from spatial_agent.tools.countvid_backend import run_countvid_subprocess
from spatial_agent.tools.video_counting_3d import CountVideoObjects3DTool
from spatial_agent.tools.video_counting_3d_utils import constrained_greedy_cluster
from spatial_agent.tools.video_counting import CountVideoObjectsTool
from spatial_agent.tools.video_counting_utils import aggregate_unique_tracks


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
        "CountVideoObjects3D": {"images": ["missing.jpg"], "objects": ["cat"]},
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
    assert result["payload"]["pipeline_stats"]["official_pipeline"] is False
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


def test_count_video_objects_3d_success_with_mock_backends(tmp_path, monkeypatch):
    frame_paths = []
    for i in range(3):
        p = tmp_path / f"frame_{i}.jpg"
        Image.new("RGB", (20, 20), "white").save(p)
        frame_paths.append(str(p))

    class DummyRex:
        def inference(self, *, images, task, categories):
            assert task == "detection"
            return [
                {
                    "extracted_predictions": {
                        "table": [
                            {"type": "box", "coords": [4, 4, 10, 10], "score": 0.9},
                        ]
                    }
                }
            ]

    class DummyTensor:
        def __init__(self, array):
            self.array = array

        @property
        def shape(self):
            return self.array.shape

        def to(self, _device):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

        def __getitem__(self, item):
            return DummyTensor(self.array[item])

    class DummyTorch:
        class no_grad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

    class DummyVGGT:
        def __call__(self, images):
            import numpy as np

            points = np.zeros((1, 3, 20, 20, 3), dtype=float)
            points[0, 0, :, :, :] = [0.0, 0.0, 0.0]
            points[0, 1, :, :, :] = [0.1, 0.0, 0.0]
            points[0, 2, :, :, :] = [2.0, 0.0, 0.0]
            return {"world_points": DummyTensor(points)}

    def _load_images(paths, mode="pad"):
        import numpy as np

        return DummyTensor(np.zeros((len(paths), 3, 20, 20), dtype=float))

    monkeypatch.setattr(
        "spatial_agent.tools.video_counting_3d.get_rex_omni_backend",
        lambda **kwargs: {"wrapper": DummyRex(), "backend_label": "rex"},
    )
    monkeypatch.setattr(
        "spatial_agent.tools.video_counting_3d.get_vggt_backend",
        lambda **kwargs: {
            "torch": DummyTorch(),
            "model": DummyVGGT(),
            "load_and_preprocess_images": _load_images,
        },
    )

    config = SpatialAgentConfig(
        artifact_dir=str(tmp_path),
        tool_config={
            "video_counting_3d": {
                "num_frames": 3,
                "device": "cpu",
                "use_sam_masks": False,
                "use_tracking": False,
                "cg_distance_threshold": 0.35,
                "bbox_point_stride": 4,
            }
        },
    )
    result = CountVideoObjects3DTool(config).invoke(images=frame_paths, objects=["table"])

    assert result["status"] == "success"
    assert result["payload"]["instance_count"] == 2
    assert result["payload"]["pipeline_stats"]["sampled_frame_count"] == 3
    assert result["payload"]["pipeline_stats"]["view_count"] == 3
    assert Path(result["artifacts"][0]).exists()


def test_constrained_greedy_cluster_respects_frame_disjoint_constraint():
    views = [
        {"view_id": "a", "object": "chair", "frame_index": 0, "bbox": [0, 0, 1, 1], "center_3d": [0, 0, 0]},
        {"view_id": "b", "object": "chair", "frame_index": 1, "bbox": [0, 0, 1, 1], "center_3d": [0.1, 0, 0]},
        {"view_id": "c", "object": "chair", "frame_index": 1, "bbox": [2, 2, 3, 3], "center_3d": [0.12, 0, 0]},
    ]

    result = constrained_greedy_cluster(views, distance_threshold=0.2)

    assert len(result) == 2
    assert sorted(instance["member_count"] for instance in result) == [1, 2]


def test_constrained_greedy_cluster_uses_tracking_prior():
    views = [
        {"view_id": "a", "object": "chair", "frame_index": 0, "bbox": [0, 0, 1, 1], "center_3d": [0.0, 0, 0], "track_id": "t1"},
        {"view_id": "b", "object": "chair", "frame_index": 1, "bbox": [0, 0, 1, 1], "center_3d": [0.2, 0, 0], "track_id": "t1"},
        {"view_id": "c", "object": "chair", "frame_index": 2, "bbox": [0, 0, 1, 1], "center_3d": [0.12, 0, 0]},
    ]

    result = constrained_greedy_cluster(views, distance_threshold=0.05)

    assert len(result) == 1
    assert result[0]["member_count"] == 3
    assert sorted(view["track_id"] for view in result[0]["views"] if view["track_id"]) == ["t1", "t1"]


def test_count_video_objects_3d_registered_when_configured():
    registry = build_default_tool_registry(
        SpatialAgentConfig(tool_config={"video_counting_3d": {"num_frames": 64}})
    )

    names = registry.list_names()
    assert "CountVideoObjects3D" in names
    assert registry.get("CountVideoObjects3D").name == "CountVideoObjects3D"


def test_count_video_objects_defaults_to_official_count(tmp_path, monkeypatch):
    frame_paths = []
    for i in range(2):
        p = tmp_path / f"frame_{i}.jpg"
        Image.new("RGB", (64, 48), "white").save(p)
        frame_paths.append(str(p))

    def _mock_subprocess(**kwargs):
        return {
            "instance_count": 4,
            "raw_tracks": [
                {
                    "track_id": "track_1",
                    "seed_id": "track_1",
                    "object": "table",
                    "start_frame_idx": 0,
                    "frame_indices": [0, 1],
                    "points_px": [[10, 10], [11, 11]],
                },
                {
                    "track_id": "track_2",
                    "seed_id": "track_2",
                    "object": "table",
                    "start_frame_idx": 0,
                    "frame_indices": [0, 1],
                    "points_px": [[12, 12], [13, 13]],
                },
            ],
            "frame_summaries": [
                {"image": "image-0", "candidate_count": 2, "filtered_count": 2},
                {"image": "image-1", "candidate_count": 2, "filtered_count": 2},
            ],
            "backend": "countvid:subprocess+test",
            "countvid_stats": {"official_pipeline": True},
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
                "track_merge_point_threshold_px": 100,
            }
        },
    )
    result = CountVideoObjectsTool(config).invoke(images=frame_paths, objects=["table"])

    assert result["status"] == "success"
    assert result["payload"]["instance_count"] == 4
    assert result["payload"]["pipeline_stats"]["unique_track_count"] == 4
    assert result["payload"]["pipeline_stats"]["apply_spatialscore_aggregation"] is False


def test_countvid_subprocess_uses_full_official_pipeline(tmp_path, monkeypatch):
    frame_paths = []
    for i in range(2):
        p = tmp_path / f"frame_{i}.jpg"
        Image.new("RGB", (64, 48), "white").save(p)
        frame_paths.append(str(p))

    repo = tmp_path / "CountVid"
    repo.mkdir()
    script = repo / "count_in_videos.py"
    script.write_text("# fake", encoding="utf-8")
    countgd_ckpt = tmp_path / "countgd.pth"
    countgd_ckpt.write_text("x", encoding="utf-8")
    sam_ckpt = tmp_path / "sam.pt"
    sam_ckpt.write_text("x", encoding="utf-8")

    captured = {}

    class _Proc:
        returncode = 0
        stdout = (
            "original number of frames: 2\n"
            "new number of frames: 2\n"
            "time stage 1: 1.5\n"
            "time stage 2: 0.5\n"
            "time stage 3: 2.0\n"
            "Total Number of Objects: 1\n"
        )
        stderr = ""

    def _run(cmd, capture_output, text, timeout, cwd, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        output_idx = cmd.index("--output_dir") + 1
        Path(cmd[output_idx]).mkdir(parents=True)
        Path(cmd[output_idx], "final-video.mp4").write_text("video", encoding="utf-8")
        output_file = Path(cmd[cmd.index("--output_file") + 1])
        frames_dir = cmd[cmd.index("--video_dir") + 1]
        output_file.write_text(json.dumps({frames_dir: {"table": 1}}), encoding="utf-8")
        (repo / "T.json").write_text(
            json.dumps({"1": {"0": [[10, 11], [20, 22]], "1": [[12], [24]]}}),
            encoding="utf-8",
        )
        return _Proc()

    monkeypatch.setattr("spatial_agent.tools.countvid_backend.subprocess.run", _run)

    result = run_countvid_subprocess(
        image_paths=frame_paths,
        objects=["table"],
        countgd_repo_path=str(repo),
        countgd_checkpoint_path=str(countgd_ckpt),
        sam2_checkpoint_path=str(sam_ckpt),
        sam2_config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
        device="cpu",
        window_size=5,
        temporal_filter=True,
        obj_batch_size=7,
        obj_batch_size_filter=11,
        img_batch_size=3,
        min_obj_area=13,
        downsample_factor=2,
        sample_frames=2,
        convert_to_rgb=True,
        save_final_video=True,
        temp_dir=str(tmp_path / "run"),
    )

    cmd = captured["cmd"]
    assert "--temporal_filter" in cmd
    assert cmd[cmd.index("--w") + 1] == "5"
    assert cmd[cmd.index("--obj_batch_size") + 1] == "7"
    assert cmd[cmd.index("--obj_batch_size_filter") + 1] == "11"
    assert cmd[cmd.index("--img_batch_size") + 1] == "3"
    assert cmd[cmd.index("--min_obj_area") + 1] == "13"
    assert cmd[cmd.index("--downsample_factor") + 1] == "2.0"
    assert cmd[cmd.index("--sample_frames") + 1] == "2"
    assert "--convert_to_rgb" in cmd
    assert "--save_final_video" in cmd
    assert "--save_T" in cmd

    assert result["instance_count"] == 1
    assert len(result["raw_tracks"]) == 1
    assert result["raw_tracks"][0]["points_px"] == [[21.0, 10.5], [24.0, 12.0]]
    assert result["countvid_stats"]["official_pipeline"] is True
    assert result["countvid_stats"]["temporal_filter"] is True
    assert result["countvid_stats"]["time_stage_1"] == 1.5
    assert len(result["countvid_artifacts"]) == 1


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


# --- Phase 4: Unique Instance Aggregation ---

def test_aggregate_merges_similar_tracks():
    tracks = [
        {"track_id": "A", "object": "table",
         "frame_indices": [0, 1, 2, 3], "points_px": [[100, 200], [102, 198], [105, 202], [108, 199]]},
        {"track_id": "B", "object": "table",
         "frame_indices": [2, 3, 4, 5], "points_px": [[103, 201], [107, 200], [110, 198], [112, 197]]},
    ]
    result = aggregate_unique_tracks(tracks, point_threshold_px=50, merge_overlap_min_frames=2)
    assert len(result) == 1
    assert set(result[0]["frame_indices"]) == {0, 1, 2, 3, 4, 5}


def test_aggregate_keeps_distant_tracks_separate():
    tracks = [
        {"track_id": "A", "object": "table",
         "frame_indices": [0, 1, 2], "points_px": [[100, 200], [102, 198], [105, 202]]},
        {"track_id": "B", "object": "table",
         "frame_indices": [0, 1, 2], "points_px": [[500, 600], [502, 598], [505, 602]]},
    ]
    result = aggregate_unique_tracks(tracks, point_threshold_px=50, merge_overlap_min_frames=2)
    assert len(result) == 2


def test_aggregate_keeps_non_overlapping_separate():
    tracks = [
        {"track_id": "A", "object": "table",
         "frame_indices": [0, 1], "points_px": [[100, 200], [102, 198]]},
        {"track_id": "B", "object": "table",
         "frame_indices": [3, 4], "points_px": [[100, 200], [102, 198]]},
    ]
    result = aggregate_unique_tracks(tracks, point_threshold_px=50, merge_overlap_min_frames=2)
    assert len(result) == 2


def test_aggregate_filters_by_min_support():
    tracks = [
        {"track_id": "A", "object": "table",
         "frame_indices": [0], "points_px": [[100, 200]]},
        {"track_id": "B", "object": "table",
         "frame_indices": [0, 1, 2], "points_px": [[500, 600], [502, 598], [505, 602]]},
    ]
    result = aggregate_unique_tracks(tracks, min_track_support=2, point_threshold_px=50)
    assert len(result) == 1
    assert result[0]["track_id"] == "B"


def test_aggregate_empty_returns_empty():
    assert aggregate_unique_tracks([]) == []


def test_aggregate_single_track_passes_through():
    tracks = [
        {"track_id": "A", "object": "table",
         "frame_indices": [0, 1, 2], "points_px": [[100, 200], [102, 198], [105, 202]]},
    ]
    result = aggregate_unique_tracks(tracks)
    assert len(result) == 1


def test_aggregate_multi_component_clustering():
    """A+B merge, C stays alone → 2 unique instances."""
    tracks = [
        {"track_id": "A", "object": "table",
         "frame_indices": [0, 1, 2, 3], "points_px": [[100, 200], [102, 198], [105, 202], [107, 201]]},
        {"track_id": "B", "object": "table",
         "frame_indices": [2, 3, 4, 5], "points_px": [[103, 201], [106, 200], [108, 199], [110, 198]]},
        {"track_id": "C", "object": "table",
         "frame_indices": [0, 1, 2, 3], "points_px": [[500, 600], [502, 598], [505, 602], [507, 601]]},
    ]
    result = aggregate_unique_tracks(tracks, point_threshold_px=50, merge_overlap_min_frames=2)
    assert len(result) == 2
