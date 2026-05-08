from spatial_agent.adapters.factory import create_llm_adapter
from spatial_agent.adapters.huggingface_qwen import HuggingFaceQwenAdapter
from spatial_agent.adapters.openai_compatible import OpenAICompatibleAdapter
from spatial_agent.io.lmms_bridge import build_task_input_from_vsibench_doc
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
