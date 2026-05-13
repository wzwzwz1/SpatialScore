from spatial_agent.adapters.mock import MockLLMAdapter
from spatial_agent.agent import SpatialAgent
from spatial_agent.graph.tool_args import normalize_tool_arguments
from spatial_agent.runtime.config import SpatialAgentConfig
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.registry import ToolRegistry


class EchoTool(BaseSpatialTool):
    name = "EchoTool"
    description = "Returns the input payload for testing."
    args_schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}
    returns_schema = {"type": "object"}

    def invoke(self, **kwargs):
        return self.success(payload={"echo": kwargs["value"]})


class CaptureImageTool(BaseSpatialTool):
    name = "CaptureImageTool"
    description = "Captures normalized image arguments for testing."
    args_schema = {"type": "object", "properties": {"image": {"type": "string"}}, "required": ["image"]}
    returns_schema = {"type": "object"}

    def invoke(self, **kwargs):
        return self.success(payload={"image": kwargs.get("image"), "images": kwargs.get("images")})


def test_graph_finish_path_returns_answer():
    adapter = MockLLMAdapter(
        responses=[
            {"thought": "Enough evidence.", "action": None, "finish": {"answer": "A"}},
        ]
    )
    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=ToolRegistry(),
        config=SpatialAgentConfig(),
    )

    result = agent.invoke(
        {
            "task_id": "task-1",
            "question": "Which option is correct?",
            "question_type": "multi_choice",
            "input_modality": "single_image",
            "image_paths": [],
            "options": ["A", "B"],
        }
    )

    assert result["status"] == "success"
    assert result["final_answer"] == "A"
    assert result["tool_calls"] == []


def test_graph_tool_round_trip_records_observation():
    adapter = MockLLMAdapter(
        responses=[
            {
                "thought": "Need to inspect an observation first.",
                "action": {"name": "EchoTool", "arguments": {"value": "hello"}},
                "finish": None,
            },
            {"thought": "Now I know enough.", "action": None, "finish": {"answer": "done"}},
        ]
    )

    registry = ToolRegistry()
    registry.register(EchoTool())

    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=registry,
        config=SpatialAgentConfig(),
    )

    result = agent.invoke(
        {
            "task_id": "task-2",
            "question": "Use a tool and finish.",
            "question_type": "open_ended",
            "input_modality": "single_image",
            "image_paths": [],
        }
    )

    assert result["status"] == "success"
    assert result["final_answer"] == "done"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool_name"] == "EchoTool"
    assert result["tool_observations"][0]["payload"]["echo"] == "hello"


def test_graph_repair_path_recovers_after_malformed_response():
    adapter = MockLLMAdapter(
        responses=[
            '{"thought": "bad json"',
            {"thought": "Recovered.", "action": None, "finish": {"answer": "yes"}},
        ]
    )

    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=ToolRegistry(),
        config=SpatialAgentConfig(),
    )

    result = agent.invoke(
        {
            "task_id": "task-3",
            "question": "Recover from malformed output.",
            "question_type": "judgment",
            "input_modality": "single_image",
            "image_paths": [],
        }
    )

    assert result["status"] == "success"
    assert result["final_answer"] == "yes"
    assert result["reasoning_trace"][0]["stage"] == "repair"


def test_graph_writes_trace_file(tmp_path):
    adapter = MockLLMAdapter(
        responses=[
            {"thought": "Enough evidence.", "action": None, "finish": {"answer": "B"}},
        ]
    )
    config = SpatialAgentConfig(artifact_dir=str(tmp_path))
    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=ToolRegistry(),
        config=config,
    )

    result = agent.invoke(
        {
            "task_id": "task-trace",
            "question": "Persist the trace.",
            "question_type": "multi_choice",
            "input_modality": "single_image",
            "image_paths": [],
            "options": ["A", "B"],
        }
    )

    assert result["status"] == "success"
    assert result["trace_path"].endswith("task-trace.json")


def test_graph_fails_after_exceeding_repair_limit():
    adapter = MockLLMAdapter(
        responses=[
            '{"thought": "bad json"',
            '{"thought": "still bad json"',
            '{"thought": "again bad json"',
        ]
    )
    config = SpatialAgentConfig(max_repairs=1)
    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=ToolRegistry(),
        config=config,
    )

    result = agent.invoke(
        {
            "task_id": "task-4",
            "question": "Fail after too many malformed outputs.",
            "question_type": "open_ended",
            "input_modality": "single_image",
            "image_paths": [],
        }
    )

    assert result["status"] == "failed"
    assert "repair" in result["error"].lower()


def test_graph_normalizes_placeholder_single_image_argument():
    adapter = MockLLMAdapter(
        responses=[
            {
                "thought": "Need to inspect the first frame.",
                "action": {"name": "CaptureImageTool", "arguments": {"image": "image1"}},
                "finish": None,
            },
            {"thought": "Done.", "action": None, "finish": {"answer": "ok"}},
        ]
    )
    registry = ToolRegistry()
    registry.register(CaptureImageTool())
    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=registry,
        config=SpatialAgentConfig(),
    )

    result = agent.invoke(
        {
            "task_id": "task-image-normalization",
            "question": "Use the first image.",
            "question_type": "open_ended",
            "input_modality": "video",
            "image_paths": ["/tmp/frame0.jpg", "/tmp/frame1.jpg"],
        }
    )

    assert result["status"] == "success"
    assert result["tool_calls"][0]["arguments"]["image"] == "/tmp/frame0.jpg"
    assert result["tool_observations"][0]["payload"]["image"] == "/tmp/frame0.jpg"


def test_graph_normalizes_placeholder_multi_image_argument():
    adapter = MockLLMAdapter(
        responses=[
            {
                "thought": "Need all frames.",
                "action": {"name": "CaptureImageTool", "arguments": {"images": ["image1", "image2"]}},
                "finish": None,
            },
            {"thought": "Done.", "action": None, "finish": {"answer": "ok"}},
        ]
    )
    registry = ToolRegistry()
    registry.register(CaptureImageTool())
    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=registry,
        config=SpatialAgentConfig(),
    )

    result = agent.invoke(
        {
            "task_id": "task-images-normalization",
            "question": "Use all images.",
            "question_type": "open_ended",
            "input_modality": "video",
            "image_paths": ["/tmp/frame0.jpg", "/tmp/frame1.jpg", "/tmp/frame2.jpg"],
        }
    )

    assert result["status"] == "success"
    assert result["tool_calls"][0]["arguments"]["images"] == ["/tmp/frame0.jpg", "/tmp/frame1.jpg"]
    assert result["tool_observations"][0]["payload"]["images"] == ["/tmp/frame0.jpg", "/tmp/frame1.jpg"]


def test_graph_normalizes_placeholder_image_with_extension_argument():
    adapter = MockLLMAdapter(
        responses=[
            {
                "thought": "Need the first frame.",
                "action": {"name": "CaptureImageTool", "arguments": {"image": "image1.jpg"}},
                "finish": None,
            },
            {"thought": "Done.", "action": None, "finish": {"answer": "ok"}},
        ]
    )
    registry = ToolRegistry()
    registry.register(CaptureImageTool())
    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=registry,
        config=SpatialAgentConfig(),
    )

    result = agent.invoke(
        {
            "task_id": "task-image-extension-normalization",
            "question": "Use the first image.",
            "question_type": "open_ended",
            "input_modality": "video",
            "image_paths": ["/tmp/frame0.jpg", "/tmp/frame1.jpg"],
        }
    )

    assert result["status"] == "success"
    assert result["tool_calls"][0]["arguments"]["image"] == "/tmp/frame0.jpg"


def test_normalize_tool_arguments_injects_default_single_image_for_target_tools():
    state = {"image_paths": ["/tmp/frame0.jpg", "/tmp/frame1.jpg"]}

    normalized = normalize_tool_arguments(
        state=state,
        tool_name="LocalizeObjects",
        arguments={"objects": ["chair"]},
    )

    assert normalized["image"] == "/tmp/frame0.jpg"
    assert normalized["objects"] == ["chair"]


def test_normalize_tool_arguments_injects_default_single_image_for_count_objects():
    state = {"image_paths": ["/tmp/frame0.jpg", "/tmp/frame1.jpg"]}

    normalized = normalize_tool_arguments(
        state=state,
        tool_name="CountObjects",
        arguments={"objects": ["chair"]},
    )

    assert normalized["image"] == "/tmp/frame0.jpg"
    assert normalized["objects"] == ["chair"]


def test_default_runtime_config_aligns_react_max_steps_to_paper():
    config = SpatialAgentConfig()

    assert config.max_steps == 10


def test_graph_normalizes_vsibench_counting_answer_to_digits():
    adapter = MockLLMAdapter(
        responses=[
            {"thought": "Enough evidence.", "action": None, "finish": {"answer": "There are at least 3 chairs in the room."}},
        ]
    )
    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=ToolRegistry(),
        config=SpatialAgentConfig(),
    )

    result = agent.invoke(
        {
            "task_id": "task-counting-normalization",
            "question": "How many chair(s) are in this room?",
            "question_type": "open_ended",
            "input_modality": "video",
            "image_paths": ["/tmp/frame0.jpg"],
            "metadata": {
                "source_benchmark": "vsibench",
                "vsibench_question_type": "object_counting",
            },
        }
    )

    assert result["status"] == "success"
    assert result["final_answer"] == "3"


def test_graph_normalizes_vsibench_counting_word_answer_to_digits():
    adapter = MockLLMAdapter(
        responses=[
            {"thought": "Enough evidence.", "action": None, "finish": {"answer": "two"}},
        ]
    )
    agent = SpatialAgent(
        llm_adapter=adapter,
        tool_registry=ToolRegistry(),
        config=SpatialAgentConfig(),
    )

    result = agent.invoke(
        {
            "task_id": "task-counting-word-normalization",
            "question": "How many sofa(s) are in this room?",
            "question_type": "open_ended",
            "input_modality": "video",
            "image_paths": ["/tmp/frame0.jpg"],
            "metadata": {
                "source_benchmark": "vsibench",
                "vsibench_question_type": "object_counting",
            },
        }
    )

    assert result["status"] == "success"
    assert result["final_answer"] == "2"
