from spatial_agent.adapters.factory import create_llm_adapter
from spatial_agent.adapters.huggingface_qwen import HuggingFaceQwenAdapter
from spatial_agent.adapters.openai_compatible import OpenAICompatibleAdapter
from spatial_agent.io.lmms_bridge import build_task_input_from_vsibench_doc
from spatial_agent.io.vsibench_runner import (
    build_vsibench_video_path,
    build_vsibench_visual_task_input,
    run_vsibench_sample,
)
from spatial_agent.io.video_sampling import select_frame_indices
from spatial_agent.runtime.config import SpatialAgentConfig


def test_create_llm_adapter_returns_hf_adapter():
    adapter = create_llm_adapter(SpatialAgentConfig(llm_backend="hf", qwen_model_path="/tmp/qwen"))
    assert isinstance(adapter, HuggingFaceQwenAdapter)


def test_create_llm_adapter_returns_openai_compatible_adapter():
    adapter = create_llm_adapter(
        SpatialAgentConfig(
            llm_backend="openai_compatible",
            api_model_name="gpt-4o-mini",
            api_base_url="https://example.com/v1",
        )
    )
    assert isinstance(adapter, OpenAICompatibleAdapter)


def test_build_task_input_from_vsibench_doc_maps_mca_to_multi_choice():
    doc = {
        "question": "Where is the sofa relative to the TV?",
        "question_type": "object_rel_distance",
        "options": ["A. left", "B. right"],
        "dataset": "scannet",
        "scene_name": "scene0001",
        "ground_truth": "A",
    }

    task_input = build_task_input_from_vsibench_doc(
        doc=doc,
        image_paths=["/tmp/f1.jpg", "/tmp/f2.jpg"],
        task_id="vsibench___test___1",
    )

    assert task_input["task_id"] == "vsibench___test___1"
    assert task_input["question_type"] == "multi_choice"
    assert task_input["input_modality"] == "video"
    assert task_input["options"] == ["A. left", "B. right"]
    assert task_input["metadata"]["vsibench_question_type"] == "object_rel_distance"


def test_build_task_input_from_vsibench_doc_maps_numeric_to_open_ended():
    doc = {
        "question": "How many chairs are visible?",
        "question_type": "object_counting",
        "options": None,
        "dataset": "arkitscenes",
        "scene_name": "scene0002",
        "ground_truth": "4",
    }

    task_input = build_task_input_from_vsibench_doc(
        doc=doc,
        image_paths=["/tmp/f1.jpg"],
    )

    assert task_input["question_type"] == "open_ended"
    assert task_input["options"] is None


def test_select_frame_indices_spans_video_evenly():
    indices = select_frame_indices(total_frames=10, num_frames=4)
    assert indices[0] == 0
    assert indices[-1] == 9
    assert len(indices) == 4


def test_select_frame_indices_returns_all_frames_when_short_video():
    indices = select_frame_indices(total_frames=3, num_frames=8)
    assert indices == [0, 1, 2]


def test_build_vsibench_video_path_uses_cache_layout(tmp_path):
    video_path = build_vsibench_video_path(
        dataset="scannet",
        scene_name="scene0001",
        cache_dir=str(tmp_path / "vsibench"),
    )

    assert video_path == str(tmp_path / "vsibench" / "scannet" / "scene0001.mp4")


def test_build_vsibench_visual_task_input_samples_frames_and_maps_doc(monkeypatch, tmp_path):
    doc = {
        "question": "How many chairs are visible?",
        "question_type": "object_counting",
        "dataset": "scannet",
        "scene_name": "scene0001",
        "ground_truth": "4",
    }
    video_path = str(tmp_path / "scene0001.mp4")

    monkeypatch.setattr(
        "spatial_agent.io.vsibench_runner.sample_video_frames",
        lambda video_path, output_dir, num_frames: [str(tmp_path / "frame0.jpg"), str(tmp_path / "frame1.jpg")],
    )

    task_input, frame_root = build_vsibench_visual_task_input(
        doc=doc,
        video_path=video_path,
        task_id="vsibench___test___2",
        artifact_dir=str(tmp_path / "artifacts"),
        split="test",
        doc_id=2,
        num_frames=16,
        video_frame_dir=None,
    )

    assert task_input["task_id"] == "vsibench___test___2"
    assert task_input["image_paths"] == [str(tmp_path / "frame0.jpg"), str(tmp_path / "frame1.jpg")]
    assert task_input["metadata"]["vsibench_question_type"] == "object_counting"
    assert str(frame_root).endswith("sampled_frames/vsibench/test/2")


def test_run_vsibench_sample_loads_doc_and_invokes_agent(monkeypatch, tmp_path):
    class DummySplit(list):
        pass

    loaded = {"question": "How many tables are visible?", "question_type": "object_counting", "dataset": "scannet", "scene_name": "scene0001", "ground_truth": "4"}
    dummy_dataset = DummySplit([loaded])
    (tmp_path / "scene0001.mp4").write_bytes(b"video")

    monkeypatch.setattr(
        "spatial_agent.io.vsibench_runner._load_datasets_module",
        lambda: type("DummyDatasets", (), {"load_dataset": staticmethod(lambda *args, **kwargs: dummy_dataset)})(),
    )
    monkeypatch.setattr(
        "spatial_agent.io.vsibench_runner.build_vsibench_video_path",
        lambda dataset, scene_name, cache_dir: str(tmp_path / "scene0001.mp4"),
    )
    monkeypatch.setattr(
        "spatial_agent.io.vsibench_runner.build_vsibench_visual_task_input",
        lambda **kwargs: (
            {
                "task_id": "vsibench___test___0",
                "question": loaded["question"],
                "question_type": "open_ended",
                "input_modality": "video",
                "image_paths": [str(tmp_path / "frame0.jpg")],
                "metadata": {"vsibench_question_type": "object_counting"},
            },
            tmp_path / "frames",
        ),
    )

    class DummyAgent:
        def invoke(self, task_input):
            return {"status": "success", "final_answer": "4", "task_id": task_input["task_id"]}

    result = run_vsibench_sample(
        agent=DummyAgent(),
        dataset_split="test",
        doc_id=0,
        num_frames=16,
        artifact_dir=str(tmp_path / "artifacts"),
        keep_video_frames=False,
        dataset_cache_dir=str(tmp_path / "vsibench"),
    )

    assert result["doc"]["scene_name"] == "scene0001"
    assert result["task_input"]["task_id"] == "vsibench___test___0"
    assert result["result"]["final_answer"] == "4"


def test_run_vsibench_sample_removes_sampled_frames_when_not_kept(monkeypatch, tmp_path):
    class DummySplit(list):
        pass

    loaded = {"question": "How many tables are visible?", "question_type": "object_counting", "dataset": "scannet", "scene_name": "scene0001", "ground_truth": "4"}
    dummy_dataset = DummySplit([loaded])
    (tmp_path / "scene0001.mp4").write_bytes(b"video")
    frame_root = tmp_path / "frames"
    frame_root.mkdir()

    monkeypatch.setattr(
        "spatial_agent.io.vsibench_runner._load_datasets_module",
        lambda: type("DummyDatasets", (), {"load_dataset": staticmethod(lambda *args, **kwargs: dummy_dataset)})(),
    )
    monkeypatch.setattr("spatial_agent.io.vsibench_runner.build_vsibench_video_path", lambda dataset, scene_name, cache_dir: str(tmp_path / "scene0001.mp4"))
    monkeypatch.setattr(
        "spatial_agent.io.vsibench_runner.build_vsibench_visual_task_input",
        lambda **kwargs: (
            {
                "task_id": "vsibench___test___0",
                "question": loaded["question"],
                "question_type": "open_ended",
                "input_modality": "video",
                "image_paths": [str(tmp_path / "frame0.jpg")],
                "metadata": {"vsibench_question_type": "object_counting"},
            },
            frame_root,
        ),
    )

    class DummyAgent:
        def invoke(self, task_input):
            return {"status": "success", "final_answer": "4", "task_id": task_input["task_id"]}

    run_vsibench_sample(
        agent=DummyAgent(),
        dataset_split="test",
        doc_id=0,
        num_frames=16,
        artifact_dir=str(tmp_path / "artifacts"),
        keep_video_frames=False,
        dataset_cache_dir=str(tmp_path / "vsibench"),
    )

    assert not frame_root.exists()
