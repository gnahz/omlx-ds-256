"""Terminal delivery across the actual executor/collector boundary; no forwards."""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from omlx.engine_core import EngineConfig, EngineCore
from omlx.output_collector import RequestOutputCollector
from omlx.request import RequestOutput
from omlx.scheduler import SchedulerOutput


def bare_engine(step, pending=None):
    engine = EngineCore.__new__(EngineCore)
    engine.config = EngineConfig(
        decode_burst_max_steps=4,
        decode_burst_budget_s=10,
        decode_burst_budget_single_s=10,
    )
    engine.scheduler = SimpleNamespace(
        step=step, running={}, waiting=pending or deque(), has_requests=lambda: True
    )
    return engine


@pytest.mark.parametrize("finished_ids,finished", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("tokens", [[], [71]])
def test_terminal_fields_end_burst_without_starting_pending_prefill(finished_ids, finished, tokens):
    terminal = SchedulerOutput(
        has_work=True,
        finished_request_ids={"A"} if finished_ids else set(),
        outputs=[RequestOutput(request_id="A", new_token_ids=tokens,
                               completion_tokens=380, finished=finished)],
    )
    events = []

    def step():
        events.append("A-terminal")
        assert len(events) == 1, "Pending B prefill ran before terminal delivery"
        return terminal

    pending = deque(["B", "C"])
    engine = bare_engine(step, pending)
    assert engine._step_burst() == [terminal]
    assert events == ["A-terminal"]
    assert list(engine.scheduler.waiting) == ["B", "C"]


def test_nonterminal_later_chunks_continue_burst_in_order():
    steps = [SchedulerOutput(
        has_work=True, outputs=[RequestOutput(request_id="A", new_token_ids=[i],
                                              completion_tokens=i)])
        for i in range(2, 6)]
    pending = deque(steps)
    engine = bare_engine(pending.popleft)
    assert engine._step_burst() == steps
    assert not pending


def test_nonterminal_empty_control_output_keeps_burst():
    step = SchedulerOutput(has_work=True, outputs=[RequestOutput(request_id="A")])
    assert bare_engine(lambda: step)._step_burst() == [step] * 4


def test_completion_keeps_other_decode_outputs_and_pending_requests():
    other = RequestOutput(request_id="C", new_token_ids=[31], completion_tokens=12)
    terminal = SchedulerOutput(has_work=True, finished_request_ids={"A"}, outputs=[
        RequestOutput(request_id="A", completion_tokens=380, finished=True), other])
    following = SchedulerOutput(has_work=False, outputs=[
        RequestOutput(request_id="C", new_token_ids=[32], completion_tokens=13)])
    steps = deque([terminal, following])
    pending = deque(["B"])
    engine = bare_engine(steps.popleft, pending)
    engine.scheduler.running = {"C": object()}
    assert engine._step_burst() == [terminal]
    assert terminal.outputs[1] is other
    assert list(pending) == ["B"]
    assert engine._step_burst() == [following]


def test_finished_id_without_token_outputs_is_a_delivery_barrier():
    terminal = SchedulerOutput(has_work=True, finished_request_ids={"A"})
    assert bare_engine(lambda: terminal)._step_burst() == [terminal]


@pytest.mark.parametrize("finished_ids,finished", [(True, False), (False, True), (True, True)])
def test_live_disabled_barrier_restores_original_burst(finished_ids, finished):
    terminal = SchedulerOutput(
        has_work=True, finished_request_ids={"A"} if finished_ids else set(),
        outputs=[RequestOutput(request_id="A", completion_tokens=380, finished=finished)])
    pending = deque(["B"])
    engine = bare_engine(lambda: terminal, pending)
    engine._terminal_response_barrier_enabled = False
    assert engine._step_burst() == [terminal] * 4
    assert list(pending) == ["B"]
    engine._terminal_response_barrier_enabled = True
    assert engine._step_burst() == [terminal]
    assert list(pending) == ["B"]


def test_terminal_reaches_collector_before_next_prefill_handoff(mock_model, mock_tokenizer):
    """Use the real async engine loop and executor, with prefill replaced by a marker."""
    async def scenario():
        loop = asyncio.get_running_loop()
        with patch("omlx.engine_core.get_registry") as registry:
            registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer,
                config=EngineConfig(decode_burst_max_steps=4,
                                    decode_burst_budget_s=10,
                                    decode_burst_budget_single_s=10))
        assert engine._terminal_response_barrier_enabled is True
        collector = engine._output_collectors["A"] = RequestOutputCollector()
        finished = engine._finished_events["A"] = asyncio.Event()
        prefill_entered = asyncio.Event()
        pending = deque(["A-finish", "B-prefill"])
        observed = []
        terminal = RequestOutput(request_id="A", new_text="done",
                                 completion_tokens=380, finished=True, finish_reason="stop")

        def step():
            phase = pending.popleft()
            if phase == "A-finish":
                return SchedulerOutput(has_work=True, outputs=[terminal], finished_request_ids={"A"})
            observed.append(collector.output is terminal and finished.is_set())
            loop.call_soon_threadsafe(prefill_entered.set)
            return SchedulerOutput(has_work=False)

        try:
            with patch.object(engine.scheduler, "step", side_effect=step), \
                 patch.object(engine.scheduler, "has_requests", side_effect=lambda: bool(pending)):
                await engine.start()
                await asyncio.wait_for(finished.wait(), timeout=2)
                await asyncio.wait_for(prefill_entered.wait(), timeout=2)
                assert observed == [True]
                assert collector.get_nowait() is terminal
                assert not pending
        finally:
            await engine.stop()
            engine.close()

    asyncio.run(scenario())
