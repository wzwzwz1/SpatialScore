from spatial_agent.adapters.huggingface_qwen import HuggingFaceQwenAdapter
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
    available_tools = [{"name": "EstimateObjectDepth"}]

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
    assert "pure Arabic numeral only" in prompt_text


def test_react_system_prompt_forbids_invented_image_paths():
    prompt = build_react_system_prompt([{"name": "LocalizeObjects"}])

    assert "Do not invent image file names or file paths" in prompt
    assert "runtime binds real sampled frames automatically" in prompt
