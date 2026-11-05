import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from omlx import engine_core, server
from omlx.engine.base import GenerationOutput
from omlx.request_preparation import RequestPreparationTracker


@pytest.mark.asyncio
@pytest.mark.parametrize("running", [False, True])
async def test_preparation_remains_visible_only_while_cancelled_worker_drains(running):
    tracker = RequestPreparationTracker()
    tracker.begin("request-1", "model")
    started = threading.Event()
    allow_finish = threading.Event()
    ran = threading.Event()

    def block():
        started.set()
        assert allow_finish.wait(5)

    def prepare():
        ran.set()
        if running:
            block()

    with ThreadPoolExecutor(max_workers=1) as executor:
        if not running:
            executor.submit(block)
            assert await asyncio.to_thread(started.wait, 2)
        task = asyncio.create_task(
            tracker.prepare_prompt("request-1", executor, prepare)
        )
        try:
            if running:
                assert await asyncio.to_thread(started.wait, 2)
            else:
                await asyncio.sleep(0)
            (row,) = tracker.get_model_activities("model")
            assert row["phase"] == (
                "preparing_prompt" if running else "waiting_for_executor"
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert [row["phase"] for row in tracker.get_model_activities("model")] == (
                ["cancelling"] if running else []
            )
        finally:
            allow_finish.set()

    assert ran.is_set() is running
    assert tracker.get_model_activities("model") == []


@pytest.fixture
def preparation_route(monkeypatch):
    tracker = RequestPreparationTracker()
    preflight_started = asyncio.Event()
    allow_preflight = asyncio.Event()
    pool = SimpleNamespace(
        release_engine=AsyncMock(),
        get_entry=lambda _: None,
        resolve_model_id=lambda model, _settings: model,
    )

    async def preflight(*_args, **_kwargs):
        preflight_started.set()
        await allow_preflight.wait()

    engine = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text: [0] * len(text)),
        model_type="llama",
        preflight_chat=preflight,
        preflight_completion=preflight,
        count_chat_tokens=lambda *_args, **_kwargs: 5,
    )

    async def get_engine(_model, *, lease):
        lease.model_id = "canonical-model"
        return engine

    monkeypatch.setattr(server, "get_request_preparation_tracker", lambda: tracker)
    monkeypatch.setattr(server, "get_engine_for_model", get_engine)
    monkeypatch.setattr(server, "get_server_metrics", Mock(return_value=Mock()))
    monkeypatch.setattr(server._server_state, "engine_pool", pool)
    for name in ("settings_manager", "mcp_manager", "oq_manager", "global_settings"):
        monkeypatch.setattr(server._server_state, name, None)
    return SimpleNamespace(
        tracker=tracker,
        preflight_started=preflight_started,
        allow_preflight=allow_preflight,
        engine=engine,
        pool=pool,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "request_class", "data"),
    [
        ("create_completion", "CompletionRequest", {"prompt": "Hello"}),
        (
            "create_chat_completion",
            "ChatCompletionRequest",
            {"messages": [{"role": "user", "content": "Hello"}]},
        ),
        (
            "create_anthropic_message",
            "AnthropicMessagesRequest",
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2},
        ),
        ("create_response", "ResponsesRequest", {"input": "Hello", "store": False}),
    ],
)
async def test_preflight_uses_eventual_inference_id_and_cleans_if_stream_never_starts(
    preparation_route, handler, request_class, data
):
    fixture = preparation_route
    response_ids = []
    scope = {"type": "http"}

    async def app(scope, receive, send):
        request = getattr(server, request_class)(model="alias", stream=True, **data)
        http_request = SimpleNamespace(
            headers={}, scope=scope, is_disconnected=AsyncMock(return_value=False)
        )
        response = await getattr(server, handler)(request, http_request)
        response_ids.append(response.headers[server._REQUEST_ID_HEADER])
        await send({"type": "http.response.start"})

    async def send(_message):
        raise BrokenPipeError

    app_task = asyncio.create_task(
        server.ClientDisconnectTrackingMiddleware(app)(scope, AsyncMock(), send)
    )
    await asyncio.wait_for(fixture.preflight_started.wait(), 2)
    (row,) = fixture.tracker.get_model_activities("canonical-model")
    assert fixture.tracker.get_model_activities("alias") == []
    assert row["phase"] == "preparing_request"
    assert row["prompt_tokens"] == 5
    fixture.allow_preflight.set()
    with pytest.raises(BrokenPipeError):
        await app_task
    assert response_ids == [row["request_id"]]
    assert fixture.tracker.get_model_activities("canonical-model") == []


@pytest.mark.asyncio
async def test_preflight_cancellation_releases_lease_and_clears_preparation(
    preparation_route,
):
    fixture = preparation_route
    http_request = SimpleNamespace(headers={})
    request = server.CompletionRequest(model="alias", prompt="Hello")
    task = asyncio.create_task(server.create_completion(request, http_request))
    await asyncio.wait_for(fixture.preflight_started.wait(), 2)
    assert fixture.tracker.get_model_activities("canonical-model")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fixture.pool.release_engine.assert_awaited_once_with("canonical-model")
    assert fixture.tracker.get_model_activities("canonical-model") == []


@pytest.mark.asyncio
async def test_batch_completion_preparations_have_distinct_ids_without_batch_header(
    preparation_route,
):
    fixture = preparation_route
    fixture.allow_preflight.set()
    request_ids = []

    async def generate(prompt, _request_id, **_kwargs):
        (row,) = fixture.tracker.get_model_activities("canonical-model")
        assert row["request_id"] == _request_id
        assert row["prompt_tokens"] == len(prompt)
        request_ids.append(_request_id)
        # Simulate scheduler accepting the request.
        fixture.tracker.close(_request_id)
        return GenerationOutput(text="x", prompt_tokens=len(prompt), completion_tokens=1)

    fixture.engine.generate = generate
    http_request = SimpleNamespace(headers={}, scope={})
    request = server.CompletionRequest(model="alias", prompt=["first", "second"])
    response = await server.create_completion(request, http_request)

    assert server._REQUEST_ID_HEADER not in response.headers
    assert len(set(request_ids)) == 2
    fixture.pool.release_engine.assert_awaited_once_with("canonical-model")
    assert fixture.tracker.get_model_activities("canonical-model") == []


@pytest.mark.asyncio
async def test_engine_core_ends_preparation_when_scheduler_accepts_request(monkeypatch):
    tracker = RequestPreparationTracker()
    tracker.begin("request-1", "model")
    monkeypatch.setattr(engine_core, "get_request_preparation_tracker", lambda: tracker)
    adding_request = threading.Event()
    allow_add = threading.Event()

    def add_request(request):
        assert request.request_id == "request-1"
        adding_request.set()
        assert allow_add.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        core = SimpleNamespace(
            scheduler=SimpleNamespace(
                add_request=add_request, _specprefill_draft_model=None
            ),
            config=SimpleNamespace(stream_interval=1),
            _output_collectors={},
            _stream_states={},
            _finished_events={},
            _mlx_executor=executor,
            _wake_engine_loop=Mock(),
        )
        task = asyncio.create_task(
            engine_core.EngineCore.add_request(core, [0], request_id="request-1")
        )
        try:
            assert await asyncio.to_thread(adding_request.wait, 2)
            (row,) = tracker.get_model_activities("model")
            assert row["phase"] == "waiting_for_scheduler"
        finally:
            allow_add.set()

        assert await task == "request-1"
        assert tracker.get_model_activities("model") == []
        core._wake_engine_loop.assert_called_once_with()
