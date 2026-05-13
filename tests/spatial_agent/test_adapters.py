from spatial_agent.adapters.huggingface_qwen import HuggingFaceQwenAdapter
from spatial_agent.adapters.react_decisions import parse_react_decisions
from spatial_agent.prompts.react_system_prompt import build_react_system_prompt


def test_qwen_adapter_builds_prompt_with_history_and_options():
    adapter = HuggingFaceQwenAdapter(model_path="/tmp/qwen")
    state = {
        "question": "Is the cat in front of the dog?",
        "question_type": "multi_choice",
        "input_modality": "multi_image",
        "image_paths": ["/tmp/a.jpg", "/tmp/b.jpg"],
        "options": ["A", "B", "C"],
        "messages": [
            {"role": "user", "content": "Is the cat in front of the dog?"},
            {"role": "tool", "content": "EstimateObjectDepth succeeded with payload: {'cat': 1.2, 'dog': 2.0}"},
        ],
        "scratchpad": [
            {
                "thought": "Need depth.",
                "action": "EstimateObjectDepth",
                "arguments": {"objects": ["cat", "dog"]},
                "observation": {"status": "success", "payload": {"cat": 1.2, "dog": 2.0}},
            }
        ],
        "metadata": {
            "source_benchmark": "vsibench",
            "vsibench_question_type": "object_counting",
        },
    }
    available_tools = [{"name": "CountObjects"}]

    messages = adapter._build_messages(state, available_tools)

    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "image"
    prompt_text = user_content[-1]["text"]
    assert "Options: A, B, C" in prompt_text
    assert "Previous tool interaction history" in prompt_text
    assert "EstimateObjectDepth" in prompt_text
    assert "Recent conversation context" in prompt_text
    assert "Counting rule:" in prompt_text
    assert "CountObjects" in prompt_text
    assert "returned points" in prompt_text
    assert "pure Arabic numeral only" in prompt_text


def test_react_system_prompt_forbids_invented_image_paths():
    prompt = build_react_system_prompt([{"name": "CountObjects"}])

    assert "Do not invent image file names or file paths" in prompt
    assert "runtime binds real sampled frames automatically" in prompt
    assert "CountObjects" in prompt


def test_parse_react_decisions_accepts_multiple_concatenated_objects():
    raw_output = (
        '{"thought":"step1","action":{"name":"CountObjects","arguments":{"objects":"table"}},"finish":null}\n'
        '{"thought":"step2","action":{"name":"CountObjects","arguments":{"objects":"table"}},"finish":null}\n'
        '{"thought":"done","action":null,"finish":{"answer":"2"}}'
    )

    parsed = parse_react_decisions(raw_output)

    assert parsed.parsed_step_count == 3
    assert parsed.accepted_step_count == 3
    assert parsed.dropped_step_count == 0
    assert parsed.accepted_steps[0]["thought"] == "step1"
    assert parsed.accepted_steps[-1]["finish"]["answer"] == "2"


def test_parse_react_decisions_drops_invalid_steps_but_keeps_valid_ones():
    raw_output = (
        '{"thought":"step1","action":{"name":"CountObjects","arguments":{"objects":"table"}},"finish":null}\n'
        '[]\n'
        '{"thought":"done","action":null,"finish":{"answer":"2"}}'
    )

    parsed = parse_react_decisions(raw_output)

    assert parsed.parsed_step_count == 3
    assert parsed.accepted_step_count == 2
    assert parsed.dropped_step_count == 1
    assert parsed.dropped_steps[0]["reason"] == "invalid_step"
