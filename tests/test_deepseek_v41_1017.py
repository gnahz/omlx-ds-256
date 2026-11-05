"""Migrated 1017 output arithmetic and media ordering checks on stable oMLX."""
from dataclasses import asdict
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from omlx.patches.deepseek_v41 import encoding, language
from omlx.patches.deepseek_v41.config import ModelConfig
from test_deepseek_v41 import tiny


@pytest.fixture(autouse=True)
def cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_checkpoint_metadata_selects_new_attention_arithmetic():
    raw = asdict(tiny())
    assert not ModelConfig.from_dict(raw).attention_output_quantization
    raw["omlx_deepseek_v41"] = {"attention_output_quantization": True}
    assert ModelConfig.from_dict(raw).attention_output_quantization
    raw["omlx_deepseek_v41"]["attention_output_quantization"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        ModelConfig.from_dict(raw)


def test_official_attention_amax_floor_and_signed_zero_are_preserved():
    values = np.tile(np.array([1e-12, -1e-12, 0.0, -0.0], np.float32), 8).reshape(1, 1, 1, 32)
    actual = language.Attention._output_heads(None, mx.array(values))
    expected = np.copysign(np.zeros_like(values), values)
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint8)), expected.view(np.uint8))


@pytest.mark.parametrize("rows", [1, 3, 9])
def test_attention_output_rounding_is_per_head_before_group_projection(rows):
    from omlx.patches.deepseek_v41.quantization import quantize_attention_output
    x = mx.array(np.random.default_rng(17).normal(size=(1, rows, 2, 32)).astype(np.float32))
    actual = language.Attention._output_heads(None, x)
    separate = mx.concatenate([quantize_attention_output(x[:, i:i+1]) for i in range(rows)], axis=1)
    np.testing.assert_array_equal(actual, separate)


def tool_call(identifier):
    return {"id": identifier, "type": "function", "function": {"name": "read", "arguments": "{}"}}


def image(url):
    return {"type": "image_url", "image_url": {"url": url}}


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("use_context", [False, True])
def test_tool_images_follow_tool_call_prompt_order(structured, use_context):
    history = [{"role": "user", "content": "Read both"},
               {"role": "assistant", "tool_calls": [tool_call("a"), tool_call("b")]}]
    results = []
    for identifier in ("b", "a"):
        msg = {"role": "tool", "tool_call_id": identifier}
        msg["content_blocks" if structured else "content"] = [
            {"type": "text", "text": "result-" + identifier}, image(identifier + ".png")]
        results.append(msg)
    prompt, media = encoding.encode_messages(
        results if use_context else history + results,
        context=history if use_context else None,
        thinking_mode="thinking", return_multi_modal_data=True,
    )
    assert prompt.index("result-a") < prompt.index("result-b")
    assert prompt.count(encoding.IMAGE_PLACEHOLDER) == 2
    assert [item["url"] for item in media["images"]] == ["a.png", "b.png"]
    assert "Unsupported" not in prompt


def test_tool_results_keep_current_turn_reasoning_until_next_user():
    messages = [{"role": "user", "content": "Read"},
                {"role": "assistant", "reasoning_content": "current-plan", "tool_calls": [tool_call("a")]},
                {"role": "tool", "tool_call_id": "a", "content": "ok"}]
    prompt = encoding.encode_messages(messages, thinking_mode="thinking")
    assert "current-plan" in prompt
    merged = encoding.merge_tool_messages(messages)
    assert encoding.find_last_user_index(merged) == 0
    messages.append({"role": "user", "content": "Next question"})
    prompt = encoding.encode_messages(messages, thinking_mode="thinking")
    assert "current-plan" not in prompt
    assert "Next question" in prompt


def test_user_image_list_survives_merge_and_mid_system_starts_turn():
    messages = [{"role": "user", "content": [{"type": "text", "text": "Look"}, image("x.png")]},
                {"role": "assistant", "content": "ok", "reasoning_content": "old-plan"},
                {"role": "system", "content": "New instruction"}]
    prompt, media = encoding.encode_messages(messages, thinking_mode="thinking", return_multi_modal_data=True)
    assert "Look" in prompt and "Unsupported" not in prompt
    assert [item["url"] for item in media["images"]] == ["x.png"]
    assert "old-plan" not in prompt
    assert encoding.find_last_user_index(encoding.merge_tool_messages(messages)) == 2


@pytest.mark.parametrize("role", ["user", "tool"])
def test_placeholder_is_validated_before_merge_can_discard_fields(role):
    messages = [{"role": role, "content": encoding.IMAGE_PLACEHOLDER,
                 "content_blocks": [{"type": "text", "text": "safe"}]}]
    with pytest.raises(ValueError, match="image special token"):
        encoding.encode_messages(messages, thinking_mode="chat")


def data_image(color):
    import base64
    from io import BytesIO
    from PIL import Image

    stream = BytesIO()
    Image.new("RGB", (2, 2), color).save(stream, format="PNG")
    return image("data:image/png;base64," + base64.b64encode(stream.getvalue()).decode())


@pytest.mark.parametrize("structured", [False, True])
def test_vlm_adapter_keeps_decoded_tool_pixels_in_sorted_prompt_order(structured):
    from omlx.patches.deepseek_v41.processing import format_messages, prepare_media_messages

    messages = [{"role": "user", "content": "Read both"},
                {"role": "assistant", "tool_calls": [tool_call("a"), tool_call("b")]}]
    for identifier, color in [("b", "blue"), ("a", "red")]:
        messages.append({"role": "tool", "tool_call_id": identifier,
                         "content_blocks" if structured else "content": [
                             {"type": "text", "text": "result-" + identifier}, data_image(color)]})
    prepared, text_messages, images, audio = prepare_media_messages(messages)
    assert not audio
    assert [item.getpixel((0, 0)) for item in images] == [(255, 0, 0), (0, 0, 255)]
    text_prompt = encoding.encode_messages(text_messages, thinking_mode="chat")
    assert encoding.IMAGE_PLACEHOLDER not in text_prompt
    formatted, ranges = format_messages(prepared, len(images))
    prompt = encoding.encode_messages(formatted, thinking_mode="chat")
    assert prompt.index("result-a") < prompt.index("result-b")
    assert prompt.count(encoding.IMAGE_PLACEHOLDER) == len(images)
    assert ranges == [(2, 2)]


def test_vlm_request_path_uses_sorted_tool_images(monkeypatch):
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.patches.deepseek_v41.processing import format_messages

    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._vlm_model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v41"))
    engine._processor = SimpleNamespace()
    engine._apply_ocr_prompt = lambda messages: messages
    captured = []

    def prepare(messages, images, **kwargs):
        formatted, _ = format_messages(messages, len(images))
        captured.append((encoding.encode_messages(formatted, thinking_mode="chat"), images))
        return [], None, None, None, 0, []

    engine._prepare_vision_inputs = prepare
    monkeypatch.setattr(mx, "synchronize", lambda: None)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    messages = [{"role": "user", "content": "Read both"},
                {"role": "assistant", "tool_calls": [tool_call("a"), tool_call("b")]},
                {"role": "tool", "tool_call_id": "b", "content": [data_image("blue")]},
                {"role": "tool", "tool_call_id": "a", "content": [data_image("red")]}]
    engine._process_chat_messages(messages, None, {})
    assert len(captured) == 1
    prompt, images = captured[0]
    assert prompt.count(encoding.IMAGE_PLACEHOLDER) == 2
    assert [item.getpixel((0, 0)) for item in images] == [(255, 0, 0), (0, 0, 255)]
