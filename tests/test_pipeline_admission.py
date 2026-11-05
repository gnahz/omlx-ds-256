"""CPU-only entrance ownership and four-protocol route contention."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from omlx import server, request_admission, request_preparation
from omlx.request_admission import RequestAdmission
from test_request_preparation import preparation_route
from test_active_models_visibility import FakePool, EmptyPrefillTracker
from omlx.admin import routes as admin_routes


@pytest.fixture
def admission(monkeypatch):
    value = RequestAdmission()
    monkeypatch.setattr(request_admission, "_admission", value)
    return value


@pytest.mark.asyncio
async def test_fifo_limit_spans_http_worker_and_scheduler(admission):
    for i in range(4):
        await admission.acquire(str(i), "model")
    tasks = [asyncio.create_task(admission.acquire(str(i), "model")) for i in range(4, 9)]
    await asyncio.sleep(0)
    assert admission.snapshot("model")["admitted_requests"] == 4
    assert admission.snapshot("model")["queued_requests"] == 5
    admission.retain("0", "scheduler")
    admission.retain("0", "worker")
    admission.release("0")
    admission.release("0", "worker")
    assert not tasks[0].done()
    admission.release("0", "scheduler")
    await asyncio.wait_for(tasks[0], 1)
    assert [row["request_id"] for row in admission.snapshot("model")["queued"]] == ["5", "6", "7", "8"]
    assert admission.snapshot("model")["admitted_requests"] == 4
    for task in tasks[1:]:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for i in range(1, 5):
        admission.release(str(i))
    assert admission.snapshot("model")["admitted_requests"] == 0
    assert admission.snapshot("model")["queued_requests"] == 0


@pytest.mark.asyncio
async def test_disconnect_of_queue_head_does_not_block_next(admission):
    await admission.acquire("running", "m", limit=1)
    disconnected = False

    async def check():
        return disconnected

    first = asyncio.create_task(admission.acquire("gone", "m", limit=1, disconnected=check))
    second = asyncio.create_task(admission.acquire("next", "m", limit=1))
    await asyncio.sleep(0)
    disconnected = True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, 1)
    assert admission.snapshot("m")["queued_requests"] == 1
    admission.release("running")
    await asyncio.wait_for(second, 1)
    admission.release("next")


@pytest.mark.asyncio
async def test_real_cancelled_sync_preparation_holds_pipeline_slot(admission):
    tracker = request_preparation.RequestPreparationTracker()
    await admission.acquire("old", "m", limit=1)
    tracker.begin("old", "m")
    started, finish = threading.Event(), threading.Event()

    def prepare():
        started.set()
        assert finish.wait(3)

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = asyncio.create_task(tracker.prepare_prompt("old", executor, prepare))
        assert await asyncio.to_thread(started.wait, 1)
        queued = asyncio.create_task(admission.acquire("next", "m", limit=1))
        await asyncio.sleep(0)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        admission.release("old")
        assert admission.snapshot("m")["queued_requests"] == 1
        assert not queued.done()
        finish.set()
        await asyncio.wait_for(queued, 1)
    admission.release("next")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("handler,request_class,data", [
    ("create_completion", "CompletionRequest", {"prompt": "Hello"}),
    ("create_chat_completion", "ChatCompletionRequest", {"messages": [{"role": "user", "content": "Hello"}]}),
    ("create_anthropic_message", "AnthropicMessagesRequest", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2}),
    ("create_response", "ResponsesRequest", {"input": "Hello", "store": False}),
])
async def test_six_routes_only_four_reach_preflight(admission, preparation_route, handler, request_class, data, stream):
    fixture = preparation_route
    reached = []

    async def preflight(*args, **kwargs):
        reached.append(1)
        await asyncio.Event().wait()

    fixture.engine.preflight_chat = preflight
    fixture.engine.preflight_completion = preflight
    tasks = []
    for _ in range(6):
        request = getattr(server, request_class)(model="alias", stream=stream, **data)
        http = SimpleNamespace(headers={}, scope={}, is_disconnected=AsyncMock(return_value=False))
        tasks.append(asyncio.create_task(getattr(server, handler)(request, http)))
    try:
        for _ in range(100):
            if len(reached) == 4 and admission.snapshot("canonical-model")["queued_requests"] == 2:
                break
            await asyncio.sleep(0.01)
        assert len(reached) == 4
        assert len(fixture.tracker.get_model_activities("canonical-model")) == 4
        state = admission.snapshot("canonical-model")
        assert state["admitted_requests"] == 4 and state["queued_requests"] == 2
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert admission.snapshot("canonical-model")["admitted_requests"] == 0
    assert admission.snapshot("canonical-model")["queued_requests"] == 0


@pytest.mark.asyncio
async def test_stream_owns_slot_until_generator_closes(admission, preparation_route, monkeypatch):
    fixture = preparation_route
    fixture.allow_preflight.set()

    async def stream(*args, **kwargs):
        yield "data: test\n\n"
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "stream_completion", stream)
    http = SimpleNamespace(headers={}, scope={}, is_disconnected=AsyncMock(return_value=False))
    response = await server.create_completion(server.CompletionRequest(model="alias", prompt="hello", stream=True), http)
    assert admission.snapshot("canonical-model")["admitted_requests"] == 1
    await anext(response.body_iterator)
    await response.body_iterator.aclose()
    assert admission.snapshot("canonical-model")["admitted_requests"] == 0


@pytest.mark.asyncio
async def test_admin_entrance_count_does_not_inflate_processing_total(admission):
    tracker = request_preparation.RequestPreparationTracker()
    for i in range(4):
        await admission.acquire(str(i), "model-a")
        tracker.begin(str(i), "model-a")
    queued = [asyncio.create_task(admission.acquire(str(i), "model-a")) for i in (4, 5)]
    await asyncio.sleep(0)
    scheduler = SimpleNamespace(snapshot_for_admin=lambda: {"running_by_id": {}, "waiting": []})
    try:
        with (
            patch.object(admin_routes, "_get_engine_pool", return_value=FakePool(scheduler)),
            patch.object(admin_routes, "_get_server_state", return_value=None),
            patch.object(admin_routes, "_get_settings_manager", return_value=None),
            patch.object(admin_routes, "_get_global_settings", return_value=None),
            patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
            patch("omlx.request_preparation.get_request_preparation_tracker", return_value=tracker),
        ):
            data = admin_routes._build_active_models_data()
        model = data["models"][0]
        assert data["total_active_requests"] == data["total_admitted_requests"] == 4
        assert data["total_queued_requests"] == model["queued_requests"] == 2
        assert model["request_limit"] == 4
        assert len(model["activities"]) == 4
        assert len(model["queued"]) == 2
    finally:
        for task in queued:
            task.cancel()
        await asyncio.gather(*queued, return_exceptions=True)
        for i in range(4):
            admission.release(str(i))


@pytest.mark.asyncio
async def test_decode_pause_keeps_new_preparation_at_visible_entrance(admission):
    await admission.acquire('old','model',limit=4)
    assert admission.model_ids(['old'])=={'model'}
    admission.pause('model')
    task=asyncio.create_task(admission.acquire('next','model',limit=4))
    await asyncio.sleep(.01)
    admission.release('old')
    await asyncio.sleep(.01)
    assert not task.done()
    snapshot=admission.snapshot('model')
    assert snapshot['queued_requests']==1 and snapshot['admitted_requests']==0
    admission.pause('model',False)
    await asyncio.wait_for(task,1)
    admission.release('next')
