"""Image wire keys are CPU-only; processor integration needs an optional local tokenizer."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from omlx.patches.deepseek_v41 import og_images as G


def test_image_digest_and_span_keys_match_wire_test_vector():
    digest = G.image_digest(6, 9, bytes(range(256)) * 4)
    assert digest == "ca566a533a9e8cea032edbb18b02c205173b3fa69dd450129e9babf366abad35"
    assert [int(key) for key in G.span_keys(digest, 3)] == [
        16016058611342194163, 15360604927821645464, 15651555166149315402
    ]


def test_image_keys_and_wire_preserve_text_and_payload():
    data = bytes(range(256)) * 4
    image = G.Image(1, 3, 6, 9, data, G.image_digest(6, 9, data))
    keys = G.prompt_keys([17, 99, 99, 99, 23], [image])
    assert int(keys[0]) == 17 and int(keys[-1]) == 23
    assert all(int(key) >> 63 for key in keys[1:4])
    np.testing.assert_array_equal(keys[1:4], G.span_keys(image.digest, image.length))
    specs, tail = G.wire([image])
    assert specs == [{"start": 1, "length": 3, "grid": [6, 9], "sha256": image.digest}]
    assert tail == data


def test_registry_returns_each_placeholder_once():
    registry = G.Registry()
    placeholder, images = object(), [object()]
    registry.put(placeholder, images)
    assert registry.take(placeholder) is images
    assert registry.take(placeholder) is None


def test_processor_builds_image_payload_with_local_tokenizer():
    model_directory = os.environ.get("DS41_TEST_MODEL_DIR")
    if not model_directory:
        pytest.skip("Set DS41_TEST_MODEL_DIR to a local DeepSeek V4.1 tokenizer directory")
    model = Path(model_directory).expanduser()
    if not (model / "tokenizer.json").is_file() or not (model / "config.json").is_file():
        pytest.skip("DS41_TEST_MODEL_DIR needs tokenizer.json and config.json")
    mx = pytest.importorskip("mlx.core", reason="Processor integration requires MLX CPU arrays")
    transformers = pytest.importorskip("transformers")
    image_module = pytest.importorskip("PIL.Image")
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.processing import Processor, format_messages

    config = ModelConfig.from_dict(json.loads((model / "config.json").read_text()))
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        str(model), local_files_only=True
    )
    processor = Processor(tokenizer, config)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image."}
    ]}]
    formatted, _ = format_messages(messages, 1)
    prompt = processor.apply_chat_template(formatted, tokenize=False)
    with mx.stream(mx.cpu):
        result = processor(text=prompt, images=[image_module.new("RGB", (84, 56), "orange")])
        ids = result["input_ids"]
        images = G.build(
            ids, result["pixel_values"], config.image_token_id,
            result["image_grids"], result["image_spans"], result["image_types"]
        )
        assert len(images) == 1
        image = images[0]
        keys = G.prompt_keys(np.asarray(ids).reshape(-1)[:-1].tolist(), images)
        assert all(int(key) >> 63 for key in keys[image.start:image.start + image.length])
        assert int(keys[0]) == int(np.asarray(ids)[0, 0])
        specs, tail = G.wire(images)
        assert len(tail) == image.vit_h * image.vit_w * G.PATCH_VALUES * 2
        assert specs[0]["sha256"] == image.digest
