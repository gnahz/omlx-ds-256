"""Plain DeepSeek completions must preserve the bytes delivered by streaming.

CPU-only fake request cores exercise the real VLM wrappers and SSE renderer;
no checkpoint, service or GPU work is involved.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.engine.vlm import VLMBatchedEngine


class Core:
    def __init__(self, chunks):
        self.chunks = chunks
        self.aborted = []

    def output(self, index):
        finished = index == len(self.chunks) - 1
        return SimpleNamespace(
            output_text="".join(self.chunks[:index + 1]),
            new_text=self.chunks[index],
            prompt_tokens=8193,
            completion_tokens=index + 1,
            finished=finished,
            finish_reason="stop" if finished else None,
            tool_calls=None,
            cached_tokens=8192,
            first_token_at=None,
        )

    async def generate(self, **kwargs):
        return self.output(len(self.chunks) - 1)

    async def add_request(self, **kwargs):
        return "whitespace-request"

    async def stream_outputs(self, request_id):
        for index in range(len(self.chunks)):
            yield self.output(index)

    async def abort_request(self, request_id):
        self.aborted.append(request_id)


def engine(chunks, model_type="deepseek_v41"):
    result = VLMBatchedEngine(model_name="whitespace-unit-test")
    result._vlm_model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    result._loaded = True
    result._engine = Core(chunks)
    return result


@pytest.mark.parametrize("chunks", [
    ("that", "\n"),
    ("  ", "indented", "\n"),
    ("\n", "\t", "\n"),
    ("", "\r\n", "end", "\r\n"),
    ("\tvalue", " "),
    ("plain",),
])
def test_deepseek_complete_text_matches_each_stream_prefix_and_exact_deltas(chunks):
    async def run():
        model = engine(chunks)
        complete = await model.generate("continue", temperature=0)
        outputs = [output async for output in model.stream_generate("continue", temperature=0)]
        expected = "".join(chunks)
        assert complete.text == "".join(output.new_text for output in outputs) == expected
        assert [output.text for output in outputs] == ["".join(chunks[:i + 1]) for i in range(len(chunks))]
        assert complete.text == outputs[-1].text
        assert complete.finish_reason == outputs[-1].finish_reason == "stop"
        assert complete.prompt_tokens == outputs[-1].prompt_tokens == 8193
        assert complete.completion_tokens == outputs[-1].completion_tokens == len(chunks)
        assert complete.cached_tokens == outputs[-1].cached_tokens == 8192
        assert model._engine.aborted == []

    asyncio.run(run())


@pytest.mark.parametrize("model_type", ["qwen2_vl", "test-vlm"])
def test_other_models_retain_existing_complete_and_cumulative_cleanup(model_type):
    async def run():
        chunks = ("  ", "value", "\n")
        model = engine(chunks, model_type=model_type)
        complete = await model.generate("continue")
        outputs = [output async for output in model.stream_generate("continue")]
        assert complete.text == outputs[-1].text == "value"
        assert [output.text for output in outputs] == ["", "value", "value"]
        assert "".join(output.new_text for output in outputs) == "  value\n"

    asyncio.run(run())


@pytest.mark.parametrize("raw,expected", [
    ("<s> \tthat\n</s>", " \tthat\n"),
    ("<|im_start|>\nbody \t<|im_end|>", "\nbody \t"),
    ("<think>reason</think>\nanswer\n", "<think>reason</think>\nanswer\n"),
])
def test_deepseek_still_removes_special_tokens_without_trimming_body(raw, expected):
    async def run():
        model = engine((raw,))
        complete = await model.generate("continue")
        outputs = [output async for output in model.stream_generate("continue")]
        assert complete.text == outputs[-1].text == expected

    asyncio.run(run())


def test_actual_completion_sse_keeps_terminal_newline_and_matches_nonstream(monkeypatch):
    from omlx import server
    from omlx.api.openai_models import CompletionRequest

    monkeypatch.setattr(server, "get_server_metrics", lambda: MagicMock())
    monkeypatch.setattr(server, "get_sampling_params", lambda *a, **kw: (0.0, 1.0, 0, 1.0, 0.0, 0.0, 0.0, 16, 0.0, 0.1))

    async def run():
        model = engine(("that", "\n"))
        complete = await model.generate("continue", temperature=0)
        request = CompletionRequest(model="deepseek-v41-lsq", prompt="continue", stream=True,
                                    stream_options={"include_usage": True}, temperature=0, max_tokens=16)
        chunks = [chunk async for chunk in server.stream_completion(
            model, "continue", request, prompt_token_ids=[], resolved_model="deepseek-v41-lsq")]
        assert chunks[-1] == "data: [DONE]\n\n"
        events = [json.loads(chunk[6:]) for chunk in chunks[:-1]]
        assert not any(event.get("error") for event in events)
        text = "".join(choice.get("text", "") for event in events for choice in event.get("choices", []))
        assert text == complete.text == "that\n"
        terminal = [choice for event in events for choice in event.get("choices", []) if choice.get("finish_reason")]
        assert len(terminal) == 1 and terminal[0]["finish_reason"] == "stop"
        usage = next(event["usage"] for event in events if "usage" in event)
        assert usage["completion_tokens"] == 2
        assert usage["prompt_tokens"] == 8193
        assert usage["prompt_tokens_details"]["cached_tokens"] == 8192

    asyncio.run(run())
