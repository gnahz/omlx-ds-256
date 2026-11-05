"""Request-ID archives without TRACE logging or per-token event-loop disk I/O.

Each request has request.json, response.json, metadata.json, and response.sse
for streaming responses. SSE captures emitted bytes, not proof of client receipt.
The writer retains partial responses on disconnect/cancellation. No HTTP auth
headers are stored. All archives are private to the service account.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import time
import uuid
from zoneinfo import ZoneInfo

LOG = logging.getLogger(__name__)
CURRENT = ContextVar('omlx_request_archive', default=None)
BLOCK = 64 * 1024
ENDPOINTS = {'/v1/chat/completions', '/v1/completions', '/v1/responses', '/v1/messages'}


def private_open(path, mode='wb'):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if mode == 'ab' else os.O_TRUNC), 0o600)
    return os.fdopen(fd, mode, buffering=128 * 1024)


def write_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    with private_open(temporary) as stream:
        stream.write((json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode())
    os.replace(temporary, path)


class StreamSummary:
    """Readable OpenAI SSE reconstruction; raw SSE retains other protocol data."""
    def __init__(self):
        self.pending = b''
        self.data = []
        self.choices = {}
        self.response_ids = []
        self.usage = None
        self.done = False
        self.error = None
        self.completed_response = None
        self.parse_errors = 0

    def feed(self, body):
        self.pending += body
        while b'\n' in self.pending:
            line, self.pending = self.pending.split(b'\n', 1)
            line = line.rstrip(b'\r')
            if not line:
                self.event()
            elif line.startswith(b'data:'):
                self.data.append(line[5:].lstrip(b' '))

    def event(self):
        if not self.data:
            return
        payload, self.data = b'\n'.join(self.data), []
        if payload == b'[DONE]':
            self.done = True
            return
        try:
            event = json.loads(payload)
            if not isinstance(event, dict):
                return
        except (ValueError, UnicodeError):
            self.parse_errors += 1
            return
        response_id = event.get('id')
        if response_id and response_id not in self.response_ids:
            self.response_ids.append(response_id)
        if event.get('usage') is not None:
            self.usage = event['usage']
        if event.get('error') is not None:
            self.error = event['error']
        if event.get('type') in ('response.completed', 'response.failed', 'response.incomplete'):
            self.completed_response = event.get('response')
            self.done = True
        if event.get('type') == 'message_stop':
            self.done = True
        for choice in event.get('choices', []):
            index = choice.get('index', 0)
            row = self.choices.setdefault(index, {'index': index, 'role': 'assistant', 'content': [],
                'reasoning_content': [], 'tool_calls': {}, 'finish_reason': None})
            delta = choice.get('delta') or {}
            if delta.get('role'):
                row['role'] = delta['role']
            row['content'].append(delta.get('content') or choice.get('text') or '')
            row['reasoning_content'].append(delta.get('reasoning_content') or delta.get('reasoning') or '')
            if choice.get('finish_reason') is not None:
                row['finish_reason'] = choice['finish_reason']
            for tool in delta.get('tool_calls') or []:
                item = row['tool_calls'].setdefault(tool.get('index', 0), {'id': '', 'type': 'function',
                    'function': {'name': '', 'arguments': ''}})
                for key in ('id', 'type'):
                    if tool.get(key):
                        item[key] = tool[key]
                for key in ('name', 'arguments'):
                    item['function'][key] += tool.get('function', {}).get(key) or ''

    def result(self):
        choices = []
        for index in sorted(self.choices):
            row = self.choices[index]
            choices.append({'index': index, 'message': {'role': row['role'],
                'content': ''.join(row['content']), 'reasoning_content': ''.join(row['reasoning_content']),
                'tool_calls': [row['tool_calls'][key] for key in sorted(row['tool_calls'])]},
                'finish_reason': row['finish_reason']})
        return {'response_ids': self.response_ids, 'choices': choices, 'usage': self.usage,
                'sse_terminal_seen': self.done, 'error': self.error,
                'completed_response': self.completed_response,
                'unparsed_complete_events': self.parse_errors,
                'scope': 'OpenAI choice deltas are assembled here. Other protocol bodies remain complete in response.sse.'}


@dataclass
class RequestContext:
    request_id: str
    writer: object
    inference_ids: list = field(default_factory=list)


class ArchiveWriter:
    def __init__(self, directory, retention_days=10, cleanup_interval_seconds=3600,
                 flush_interval_seconds=1, queue_blocks=128, clock=time.time):
        if type(retention_days) is not int or not 1 <= retention_days <= 365:
            raise ValueError('retention_days must be an integer from 1 to 365')
        if not 1 <= queue_blocks <= 4096 or cleanup_interval_seconds <= 0 or flush_interval_seconds <= 0:
            raise ValueError('Invalid archive writer bounds')
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory.chmod(0o700)
        self.retention_days, self.clock = retention_days, clock
        self.cleanup_interval = cleanup_interval_seconds
        self.flush_interval = flush_interval_seconds
        self.queue = queue.Queue(maxsize=queue_blocks)
        self.records = {}
        self.failures = 0
        self.last_error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='request-archive-writer', daemon=True)
        self.thread.start()

    async def emit(self, operation, request_id=None, value=None):
        if self.closed:
            return
        if isinstance(value, bytes) and len(value) > BLOCK:
            for offset in range(0, len(value), BLOCK):
                await self.emit(operation, request_id, value[offset:offset + BLOCK])
            return
        item = (operation, request_id, value)
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            pending = asyncio.create_task(asyncio.to_thread(self.queue.put, item))
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                # Keep this queued chunk ahead of the final cancellation record.
                await asyncio.shield(pending)
                raise

    def snapshot(self):
        return {'enabled': not self.closed, 'directory': str(self.directory),
                'retention_days': self.retention_days, 'queue_blocks_pending': self.queue.qsize(),
                'writer_failures': self.failures, 'last_error_type': self.last_error,
                'retention_scope': 'Completed requests expire by finished time; interrupted requests by started time. Cleanup runs at startup and hourly, and skips active requests.',
                'stream_scope': 'Emitted response bytes, including partial responses; not proof of client receipt.'}

    def _open(self, request_id, value):
        assert str(uuid.UUID(request_id)) == request_id
        day = datetime.fromtimestamp(value['started_unix'], ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
        directory = self.directory / day / request_id
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        directory.parent.chmod(0o700)
        row = {'directory': directory, 'metadata': dict(value, request_id=request_id, status='active'),
               'request': private_open(directory / 'request.json'), 'response': None,
               'request_sha': hashlib.sha256(), 'response_sha': hashlib.sha256(),
               'parser': None, 'request_bytes': 0, 'response_bytes': 0}
        self.records[request_id] = row
        write_json(directory / 'metadata.json', row['metadata'])

    def _finish(self, request_id, value):
        row = self.records.pop(request_id, None)
        if row is None:
            return
        close_error = None
        for key in ('request', 'response'):
            if row[key] is not None:
                try:
                    row[key].close()
                except OSError as error:
                    close_error = error
        if close_error is not None:
            raise close_error
        metadata = row['metadata']
        metadata.update(value, finished_unix=self.clock(), request_bytes=row['request_bytes'],
                        response_bytes=row['response_bytes'], request_sha256=row['request_sha'].hexdigest(),
                        response_sha256=row['response_sha'].hexdigest())
        if row['parser'] is not None:
            result = row['parser'].result()
            result['request_id'] = request_id
            result['transport_complete'] = value.get('transport_complete', False)
            write_json(row['directory'] / 'response.json', result)
            metadata['sse_terminal_seen'] = result['sse_terminal_seen']
        write_json(row['directory'] / 'metadata.json', metadata)

    def _note_failure(self, error):
        self.failures += 1
        self.last_error = type(error).__name__
        LOG.error('Request archive write failed (%s); inference continues', self.last_error)

    def _handle(self, operation, request_id, value):
        if operation == 'open':
            return self._open(request_id, value)
        row = self.records.get(request_id)
        if row is None:
            return
        if operation == 'request':
            row['request'].write(value)
            row['request_sha'].update(value)
            row['request_bytes'] += len(value)
        elif operation == 'start':
            row['metadata'].update(value)
            streaming = value.get('response_content_type', '').startswith('text/event-stream')
            row['parser'] = StreamSummary() if streaming else None
            name = 'response.sse' if streaming else 'response.json'
            row['response'] = private_open(row['directory'] / name)
        elif operation == 'response':
            if row['response'] is None:
                raise RuntimeError('Response body preceded response headers')
            row['response'].write(value)
            row['response_sha'].update(value)
            row['response_bytes'] += len(value)
            if row['parser'] is not None:
                row['parser'].feed(value)
        elif operation == 'finish':
            self._finish(request_id, value)

    def prune(self):
        cutoff = self.clock() - self.retention_days * 86400
        for day in self.directory.iterdir():
            if day.is_symlink() or not day.is_dir() or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day.name):
                continue
            for directory in day.iterdir():
                if directory.is_symlink() or not directory.is_dir() or directory.name in self.records:
                    continue
                try:
                    if str(uuid.UUID(directory.name)) != directory.name:
                        continue
                    metadata = json.loads((directory / 'metadata.json').read_text())
                    timestamp = metadata.get('finished_unix', metadata['started_unix'])
                    if isinstance(timestamp, (int, float)) and timestamp < cutoff:
                        shutil.rmtree(directory)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            try:
                day.rmdir()
            except OSError:
                pass

    def _run(self):
        flushed, cleaned = 0.0, 0.0
        while True:
            item = None
            try:
                now = time.monotonic()
                if now - cleaned >= self.cleanup_interval:
                    self.prune()
                    cleaned = now
                if now - flushed >= self.flush_interval:
                    for row in tuple(self.records.values()):
                        for key in ('request', 'response'):
                            if row[key] is not None and not row[key].closed:
                                row[key].flush()
                    flushed = now
                try:
                    item = self.queue.get(timeout=min(self.flush_interval, 1))
                except queue.Empty:
                    continue
                operation, request_id, value = item
                if operation == 'close':
                    for key in tuple(self.records):
                        try:
                            self._finish(key, {'status': 'interrupted', 'transport_complete': False})
                        except Exception as error:
                            self._note_failure(error)
                    return
                self._handle(operation, request_id, value)
            except Exception as error:
                self._note_failure(error)
            finally:
                if item is not None:
                    self.queue.task_done()

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(('close', None, None))
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError('Request archive writer did not finish shutdown')


class RequestArchiveMiddleware:
    def __init__(self, app, writer):
        self.app, self.writer = app, writer

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope.get('path') not in ENDPOINTS:
            return await self.app(scope, receive, send)
        request_id = str(uuid.uuid4())
        context = RequestContext(request_id, self.writer)
        headers = dict(scope.get('headers', []))
        await self.writer.emit('open', request_id, {'started_unix': time.time(),
            'method': scope['method'], 'path': scope['path'],
            'request_content_type': headers.get(b'content-type', b'').decode('latin-1'),
            'client_request_id': headers.get(b'x-request-id', b'').decode('latin-1')[:512]})
        token = CURRENT.set(context)
        complete, disconnected, request_complete, status = False, False, False, None
        outcome, error_type = 'incomplete', None

        async def logged_receive():
            nonlocal disconnected, request_complete
            message = await receive()
            if message['type'] == 'http.request':
                await self.writer.emit('request', request_id, message.get('body', b''))
                if not message.get('more_body', False):
                    request_complete = True
            elif message['type'] == 'http.disconnect':
                disconnected = True
            return message

        async def logged_send(message):
            nonlocal complete, status
            if message['type'] == 'http.response.start':
                status = message['status']
                response_headers = [(k, v) for k, v in message.get('headers', []) if k.lower() != b'x-request-id']
                response_headers.append((b'x-request-id', request_id.encode()))
                message = dict(message, headers=response_headers)
                content_type = dict(response_headers).get(b'content-type', b'').decode('latin-1')
                await self.writer.emit('start', request_id, {'http_status': status, 'response_content_type': content_type})
            elif message['type'] == 'http.response.body':
                await self.writer.emit('response', request_id, message.get('body', b''))
            await send(message)
            if message['type'] == 'http.response.body' and not message.get('more_body', False):
                complete = True

        try:
            await self.app(scope, logged_receive, logged_send)
            outcome = 'completed' if complete else 'incomplete'
            if status is not None and status >= 400:
                outcome = 'http_error'
        except asyncio.CancelledError:
            outcome = 'cancelled'
            raise
        except BaseException as error:
            outcome, error_type = 'error', type(error).__name__
            raise
        finally:
            if disconnected and not complete:
                outcome = 'disconnected'
            try:
                await asyncio.shield(self.writer.emit('finish', request_id, {'status': outcome,
                    'transport_complete': complete, 'client_disconnected': disconnected,
                    'request_body_complete': request_complete,
                    'engine_request_ids': list(context.inference_ids), 'error_type': error_type}))
            finally:
                CURRENT.reset(token)


def install_request_id_hooks(server, engine_class):
    """Keep archive IDs identical to scheduler IDs, including lazy SSE tasks."""
    original = engine_class.add_request
    if getattr(original, '_request_archive_hook', False):
        raise RuntimeError('Request ID hook is already installed')
    signature = inspect.signature(original)

    @functools.wraps(original)
    async def add_request(*args, **kwargs):
        context = CURRENT.get()
        if context is None:
            return await original(*args, **kwargs)
        bound = signature.bind_partial(*args, **kwargs)
        if bound.arguments.get('request_id') is None:
            bound.arguments['request_id'] = context.request_id if not context.inference_ids else str(uuid.uuid4())
        request_id = await original(*bound.args, **bound.kwargs)
        context.inference_ids.append(request_id)
        return request_id

    add_request._request_archive_hook = True
    engine_class.add_request = add_request
    abort_id = getattr(server, '_request_abort_id', None)
    if callable(abort_id):
        @functools.wraps(abort_id)
        def scoped_abort_id(engine):
            value = abort_id(engine)
            context = CURRENT.get()
            return context.request_id if value is not None and context is not None else value
        server._request_abort_id = scoped_abort_id
