"""Messages must survive the streaming and non-streaming error envelopes."""
import ast
import asyncio
import json
from pathlib import Path

import pytest
from omlx.request_memory_budget import TransientMemoryPressureError
from omlx.server import _with_sse_keepalive, _with_json_keepalive

SOURCE = Path(__file__).parents[1] / 'omlx/scheduler.py'
MESSAGE = next(n.value for n in ast.walk(ast.parse(SOURCE.read_text()))
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value.startswith('Rate limit: temporary physical memory pressure prevented this request'))


PREFILL_MESSAGE = next(n.value for n in ast.walk(ast.parse(SOURCE.read_text()))
                       if isinstance(n, ast.Constant) and isinstance(n.value,str)
                       and n.value.startswith('Rate limit: temporary physical memory pressure prevented this prefill'))

@pytest.mark.asyncio
@pytest.mark.parametrize('message',[MESSAGE,PREFILL_MESSAGE])
async def test_rate_limit_survives_partial_stream(message):
    async def upstream():
        yield 'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
        raise TransientMemoryPressureError(message)
    frames = [item async for item in _with_sse_keepalive(upstream())]
    errors = [json.loads(frame[6:])['error'] for frame in frames
              if frame.startswith('data: {') and '"error"' in frame]
    assert errors[0]['message'] == message
    assert frames[-1] == 'data: [DONE]\n\n'


@pytest.mark.asyncio
@pytest.mark.parametrize('message',[MESSAGE,PREFILL_MESSAGE])
async def test_rate_limit_survives_json_keepalive(message):
    result = asyncio.get_running_loop().create_future()
    stream = _with_json_keepalive(None, result)
    assert await anext(stream) == ' '
    result.set_exception(TransientMemoryPressureError(message))
    data = json.loads(''.join([part async for part in stream]))
    assert data['error']['message'] == message

@pytest.mark.parametrize('code',['decode_memory_budget_exceeded','memory_admission_stalled','prefill_memory_budget_exceeded'])
def test_engine_retains_retryable_memory_error_type(code):
    from omlx.engine_core import _raise_request_output_error
    from omlx.request import RequestOutput
    with pytest.raises(TransientMemoryPressureError, match='Rate limit:'):
        _raise_request_output_error(RequestOutput(request_id='test',finished=True,error=MESSAGE,error_code=code))


def test_other_runtime_errors_do_not_become_rate_limits():
    from omlx.engine_core import _raise_request_output_error
    from omlx.request import RequestOutput
    with pytest.raises(RuntimeError) as exc:
        _raise_request_output_error(RequestOutput(request_id='test',finished=True,error='Invalid input',error_code='other'))
    assert not isinstance(exc.value,TransientMemoryPressureError)


@pytest.mark.asyncio
async def test_prefill_pressure_returns_http_429_with_retry_after():
    from omlx.server import transient_memory_pressure_handler
    response=await transient_memory_pressure_handler(None,TransientMemoryPressureError(PREFILL_MESSAGE))
    assert response.status_code==429 and response.headers['Retry-After']=='1'
    assert json.loads(response.body)['error']['message']==PREFILL_MESSAGE
