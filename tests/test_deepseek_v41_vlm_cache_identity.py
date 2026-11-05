"""Image KV identities use the final processor layout without loading weights."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from PIL import Image

from omlx.cache.paged_cache import resolve_block_extra_keys
from omlx.engine.vlm import (
    VLMBatchedEngine,
    _deepseek_v41_image_cache_key_ranges,
)
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.encoding import IMAGE_PLACEHOLDER
from omlx.patches.deepseek_v41.processing import Processor
from omlx.request import Request, SamplingParams


class TinyTokenizer:
    """Preserve real image markers while assigning one token per text character."""

    def __init__(self, image_token_id):
        self.image_token_id = image_token_id

    def encode(self, text, add_special_tokens=False):
        pieces = text.split(IMAGE_PLACEHOLDER)
        ids = []
        for i, piece in enumerate(pieces):
            if i:
                ids.append(self.image_token_id)
            ids.extend(ord(char) + 1_000_000 for char in piece)
        return ids


def configuration():
    return ModelConfig(
        vision_patch_size=2,
        vision_downsample_ratio=2,
        vision_min_pixels=0,
        vision_max_n_token=1024,
    )


def prepared(images, *, config=None, prefix="lead", separators=None):
    config = config or configuration()
    processor = Processor(TinyTokenizer(config.image_token_id), config)
    separators = separators if separators is not None else ["gap"] * len(images)
    prompt = prefix + "".join(
        IMAGE_PLACEHOLDER + separator for separator in separators
    )
    inputs = processor(text=prompt, images=images)
    tokens = inputs["input_ids"][0].tolist()
    ranges = _deepseek_v41_image_cache_key_ranges(tokens, inputs, images, config)
    return tokens, inputs, ranges


def request_for(tokens, ranges):
    return Request(
        request_id="image-prefix-test",
        prompt=tokens,
        sampling_params=SamplingParams(),
        vlm_image_hash=ranges[-1][1],
        vlm_cache_key_start=ranges[0][0],
        vlm_cache_key_ranges=ranges,
    )


def salt(request, end):
    return resolve_block_extra_keys(
        end,
        extra_keys=request.vlm_extra_keys_for_cache,
        extra_key_token_start=request.vlm_extra_key_token_start_for_cache,
        extra_key_ranges=request.vlm_extra_key_ranges_for_cache,
    )


def test_appending_image_preserves_every_prior_identity_and_vlm_text_domain():
    red = Image.new("RGB", (4, 4), "red")
    blue = Image.new("RGB", (8, 4), "blue")
    tokens_a, inputs_a, ranges_a = prepared([red])
    tokens_b, inputs_b, ranges_b = prepared([red, blue])
    assert tokens_b[: len(tokens_a)] == tokens_a
    assert ranges_b[: len(ranges_a)] == ranges_a
    assert [start for start, _ in ranges_b] == [
        0,
        *[start for start, _ in inputs_b["image_spans"]],
    ]
    a, b = request_for(tokens_a, ranges_a), request_for(tokens_b, ranges_b)
    for end in range(1, len(tokens_a) + 1):
        assert salt(a, end) == salt(b, end)
    assert salt(a, 1) is not None  # VLM text cannot alias a text-only prefix.
    assert a.vlm_extra_key_token_start_for_cache == 0
    assert a.vlm_extra_keys_for_cache == (ranges_a[-1][1],)
    assert len(ranges_b) == 3
    assert all(len(key) == 64 and int(key, 16) >= 0 for _, key in ranges_b)


def test_same_size_different_pixels_diverge_at_exact_second_image_start():
    red = Image.new("RGB", (4, 4), "red")
    blue = Image.new("RGB", (4, 4), "blue")
    green = Image.new("RGB", (4, 4), "green")
    tokens, inputs, original = prepared([red, blue])
    changed_tokens, _, changed = prepared([red, green])
    assert changed_tokens == tokens  # Placeholder token IDs cannot detect this.
    boundary = inputs["image_spans"][1][0]
    a, b = request_for(tokens, original), request_for(tokens, changed)
    assert original[:2] == changed[:2]
    assert salt(a, boundary) == salt(b, boundary)
    assert salt(a, boundary + 1) != salt(b, boundary + 1)


def test_image_order_and_equal_length_different_grid_layouts_change_identity():
    red = Image.new("RGB", (4, 4), "red")
    blue = Image.new("RGB", (4, 4), "blue")
    tokens, _, forward = prepared([red, blue], separators=["", ""])
    reverse_tokens, _, reverse = prepared([blue, red], separators=["", ""])
    assert tokens == reverse_tokens
    assert forward[0] == reverse[0]
    assert forward[1:] != reverse[1:]

    # 1x5 and 2x2 aligned grids both cost eight tokens, but have different
    # row separators and positional geometry. Size-only token keys collide.
    wide_tokens, wide_inputs, wide = prepared([Image.new("RGB", (20, 4), "red")])
    square_tokens, square_inputs, square = prepared([Image.new("RGB", (8, 8), "red")])
    assert wide_tokens == square_tokens
    assert wide_inputs["image_types"] != square_inputs["image_types"]
    assert wide[0] == square[0]
    assert wide[-1] != square[-1]


def test_preprocessing_configuration_is_in_the_vlm_domain_even_if_grid_is_equal():
    images = [Image.new("RGB", (4, 4), "red")]
    config = configuration()
    tokens, inputs, ranges = prepared(images, config=config)
    changed = replace(config, vision_max_n_token=512)
    other = _deepseek_v41_image_cache_key_ranges(tokens, inputs, images, changed)
    assert [start for start, _ in ranges] == [start for start, _ in other]
    assert all(a != b for (_, a), (_, b) in zip(ranges, other))


def test_first_image_at_zero_merges_domain_and_pixel_identity_without_duplicate_start():
    image = Image.new("RGB", (4, 4), "red")
    tokens, inputs, ranges = prepared([image], prefix="")
    assert inputs["image_spans"][0][0] == 0
    assert len(ranges) == 1
    assert ranges[0][0] == 0
    assert salt(request_for(tokens, ranges), 1) == (ranges[0][1],)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda inputs: inputs.pop("image_spans"),
        lambda inputs: inputs["image_spans"].append((0, 1)),
        lambda inputs: inputs["image_spans"].__setitem__(0, (-1, 4)),
        lambda inputs: inputs["image_spans"].__setitem__(0, (4, 3)),
        lambda inputs: inputs["image_grids"].__setitem__(0, (0, 2)),
        lambda inputs: inputs["image_types"][0].__setitem__(1, 2),
        lambda inputs: inputs["image_spans"].__setitem__(1, (4, 4)),
    ],
)
def test_invalid_span_metadata_fails_closed(corrupt):
    images = [Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")]
    tokens, inputs, _ = prepared(images)
    corrupt(inputs)
    with pytest.raises(ValueError, match="DeepSeek V4.1"):
        _deepseek_v41_image_cache_key_ranges(tokens, inputs, images, configuration())


def test_unspanned_placeholder_or_non_image_token_inside_span_fails_closed():
    images = [Image.new("RGB", (4, 4), "red")]
    tokens, inputs, _ = prepared(images)
    tokens[0] = configuration().image_token_id
    with pytest.raises(ValueError, match="unkeyed image tokens"):
        _deepseek_v41_image_cache_key_ranges(tokens, inputs, images, configuration())
    start, _ = inputs["image_spans"][0]
    tokens[0] = 1_000_000
    tokens[start] = 1_000_000
    with pytest.raises(ValueError, match="layout does not match"):
        _deepseek_v41_image_cache_key_ranges(tokens, inputs, images, configuration())


def test_real_processor_spans_skip_prefix_rerenders_and_vision_feature_cache():
    config = configuration()
    processor = Processor(TinyTokenizer(config.image_token_id), config)
    processor.apply_chat_template = MagicMock(wraps=processor.apply_chat_template)
    engine = VLMBatchedEngine(model_name="tiny-v41-cache-metadata")
    engine._processor = processor
    engine._vlm_model = SimpleNamespace(
        config=config,
        language_model=SimpleNamespace(),
        get_input_embeddings=MagicMock(
            side_effect=lambda ids, *args, **kwargs: SimpleNamespace(
                inputs_embeds=mx.zeros((*ids.shape, 1))
            )
        ),
    )
    # An available feature cache must still remain unused on this model path.
    engine._vision_cache = MagicMock()
    images = [Image.new("RGB", (4, 4), "red"), Image.new("RGB", (8, 4), "blue")]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [
            {"type": "text", "text": "before"},
            {"type": "image", "url": "unused"},
            {"type": "text", "text": "between"},
            {"type": "image", "url": "unused"},
            {"type": "text", "text": "after"},
        ]},
    ]
    captured = {}

    def prepare(processor, images=None, prompts=None, **kwargs):
        inputs = processor(text=prompts[0], images=images)
        captured.update(inputs)
        return inputs

    with patch("mlx_vlm.utils.prepare_inputs", side_effect=prepare) as prepare_mock:
        result = engine._prepare_vision_inputs(messages, images)

    tokens, _, _, image_hash, key_start, ranges = result
    assert prepare_mock.call_count == 1
    assert processor.apply_chat_template.call_count == 1
    assert [start for start, _ in ranges] == [
        0,
        *[start for start, _ in captured["image_spans"]],
    ]
    assert key_start == 0
    assert image_hash == ranges[-1][1]
    assert tokens == captured["input_ids"][0].tolist()
    assert engine._vision_cache.mock_calls == []
    assert (
        "cached_image_features"
        not in engine._vlm_model.get_input_embeddings.call_args.kwargs
    )
