"""Optional private request archives; disabled until explicitly configured."""
from __future__ import annotations

import json
from pathlib import Path
import sys



def load_config(state):
    path = state / "request-archive.json"
    if not path.exists():
        return {"enabled": False}
    config = json.loads(path.read_text())
    enabled = config.pop('enabled', None)
    if type(enabled) is not bool:
        raise ValueError('Request archive enabled must be a boolean')
    if set(config) != {'directory', 'retention_days', 'cleanup_interval_seconds',
                       'flush_interval_seconds', 'queue_blocks'}:
        raise ValueError('Unexpected request archive configuration fields')
    if type(config['retention_days']) is not int or not 1 <= config['retention_days'] <= 365:
        raise ValueError('retention_days must be an integer from 1 to 365')
    if type(config['queue_blocks']) is not int or not 1 <= config['queue_blocks'] <= 4096:
        raise ValueError('queue_blocks must be an integer from 1 to 4096')
    if any(type(config[k]) not in (int, float) or not 0 < config[k] <= 86400
           for k in ('cleanup_interval_seconds', 'flush_interval_seconds')):
        raise ValueError('Archive intervals must be positive and at most one day')
    directory = (state / config['directory']).resolve()
    if not directory.is_relative_to((state / 'logs').resolve()):
        raise ValueError('Archive directory must be under this service state logs')
    config['directory'] = str(directory)
    return dict(config, enabled=enabled)


def attach_before_run(service, config):
    import uvicorn
    original_run = uvicorn.Server.run

    def run(instance, *args, **kwargs):
        # The original launcher has applied all import-time model flags by now.
        from omlx import server
        from omlx.engine_core import EngineCore
        from omlx_request_archive import ArchiveWriter, RequestArchiveMiddleware, install_request_id_hooks
        if server.app.middleware_stack is not None:
            raise RuntimeError('Request archive must install before server startup')
        writer = ArchiveWriter(**config)
        try:
            install_request_id_hooks(server, EngineCore)
            server.app.add_middleware(RequestArchiveMiddleware, writer=writer)
            from fastapi import Depends
            @server.app.get('/api/v41-request-archive', dependencies=[Depends(server.verify_api_key)])
            def archive_status():
                return writer.snapshot()
            print('Request archive enabled: ' + json.dumps(writer.snapshot()), flush=True)
            return original_run(instance, *args, **kwargs)
        finally:
            writer.close()

    uvicorn.Server.run = run
