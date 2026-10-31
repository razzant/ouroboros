"""Real dependency-bearing child responses retain ASGI semantics and lifetime."""
import asyncio
import json
import pathlib
import sys

import pytest

from ouroboros import extension_loader
from ouroboros.extension_process_runner import dispatch_extension_route_subprocess
from tests._shared import clean_extension_runtime_state
from tests.test_extension_loader import _prepare_extension, _add_fake_native_dep, _mark_isolated_deps_installed
from tests.test_widget_stream_download_ui import widget_server as widget_server


@pytest.fixture(autouse=True)
def clean_runtime(monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def prepare_route(tmp_path, plugin, *, name="stream_skill"):
    loaded, skills_root, drive_root = _prepare_extension(tmp_path, name, plugin,
        permissions=["route"], extra_frontmatter="dependencies:\n  - dummy_pkg\n")
    _add_fake_native_dep(loaded)
    _mark_isolated_deps_installed(drive_root, loaded)
    assert extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root) is None
    spec = extension_loader.list_routes()[f"/api/extensions/{name}/stream"]
    return loaded, spec, drive_root


def route_response(spec, drive_root, *, method="GET", headers=()):
    return dispatch_extension_route_subprocess(spec, {
        "method": method, "path": spec["path"], "headers": list(headers), "body_b64": "",
    }, drive_root=drive_root, repo_dir=pathlib.Path(__file__).resolve().parents[1])


async def collect_response(response, *, method="GET", observer=None):
    events = []
    async def receive():
        await asyncio.Event().wait()
    async def send(message):
        events.append(message)
        if observer is not None:
            await observer(message)
    await response({"type": "http", "method": method}, receive, send)
    return events


def test_response_streams_late_dependency_import_and_keeps_duplicate_headers(tmp_path):
    plugin = '''from starlette.responses import StreamingResponse
async def chunks():
    import dummy_pkg
    yield dummy_pkg.VALUE.encode()
    yield b'-tail'
def stream(request):
    r = StreamingResponse(chunks(), media_type='text/plain')
    r.raw_headers.extend([(b'x-item', b'one'), (b'x-item', b'two')])
    return r
def register(api):
    api.register_route('stream', stream)
'''
    _, spec, root = prepare_route(tmp_path, plugin)
    response = route_response(spec, root)
    events = asyncio.run(collect_response(response))
    assert events[0]["status"] == 200, events
    assert events[0]["headers"][-2:] == [(b"x-item", b"one"), (b"x-item", b"two")]
    assert b"".join(e.get("body", b"") for e in events) == b"isolated-native-risk-tail"
    assert events[-1]["more_body"] is False
    assert list((root / 'state/skills/stream_skill/extension_calls').glob('*.result.json')) == []


def test_child_failure_before_headers_is_502(tmp_path):
    _, spec, root = prepare_route(tmp_path, '''def stream(request):
    raise RuntimeError('before-headers')
def register(api):
    api.register_route('stream', stream)
''')
    events = asyncio.run(collect_response(route_response(spec, root)))
    assert events[0]["status"] == 502
    assert "before-headers" in json.loads(events[-1]["body"])["error"]


@pytest.mark.parametrize(('method', 'range_header', 'status', 'expected'), [
    ('GET', '', 200, None), ('HEAD', '', 200, b''),
    ('GET', 'bytes=2-8', 206, b'2345678'), ('HEAD', 'bytes=2-8', 206, b''),
    ('GET', 'bytes=99999999-', 416, b''),
])
def test_file_response_preserves_large_bytes_head_and_range(tmp_path, method, range_header, status, expected):
    plugin = '''from starlette.responses import FileResponse
def stream(request):
    return FileResponse(request.app.state.drive_root / 'large.bin', filename='export.bin')
def register(api):
    api.register_route('stream', stream)
'''
    _, spec, root = prepare_route(tmp_path, plugin)
    content = b'0123456789' * (128 * 1024)
    (root / 'large.bin').write_bytes(content)
    headers = [('range', range_header)] if range_header else []
    events = asyncio.run(collect_response(route_response(spec, root, method=method, headers=headers), method=method))
    assert events[0]['status'] == status, events
    body = b''.join(e.get('body', b'') for e in events)
    assert body == (content if expected is None else expected)
    response_headers = dict(events[0]['headers'])
    if status == 200:
        assert response_headers[b'content-length'] == str(len(content)).encode()
        assert b'export.bin' in response_headers[b'content-disposition']
    elif status == 206:
        assert response_headers[b'content-range'] == f'bytes 2-8/{len(content)}'.encode()
        assert response_headers[b'content-length'] == b'7'


def test_first_bytes_precede_producer_completion(tmp_path):
    plugin = '''import asyncio
from starlette.responses import StreamingResponse
def stream(request):
    async def chunks():
        yield b'first'
        while not (request.app.state.drive_root / 'release').exists():
            await asyncio.sleep(.01)
        yield b'last'
    return StreamingResponse(chunks())
def register(api):
    api.register_route('stream', stream)
'''
    _, spec, root = prepare_route(tmp_path, plugin)
    from ouroboros.tools.shell import _active_subprocesses
    async def observe(message):
        if message.get('body') == b'first':
            assert any(proc.poll() is None for proc in _active_subprocesses)
            assert not (root / 'release').exists()
            (root / 'release').touch()
    events = asyncio.run(collect_response(route_response(spec, root), observer=observe))
    assert b''.join(e.get('body', b'') for e in events) == b'firstlast'
    assert not _active_subprocesses


@pytest.mark.parametrize('failure', [False, True])
def test_background_work_survives_normal_body_completion(tmp_path, failure, caplog):
    plugin = '''import asyncio
from starlette.responses import Response
from starlette.background import BackgroundTask
def stream(request):
    async def after():
        import dummy_pkg
        await asyncio.sleep(.05)
        (request.app.state.drive_root / 'background.done').write_text(dummy_pkg.VALUE)
        FAIL
    return Response(b'delivered', background=BackgroundTask(after))
def register(api):
    api.register_route('stream', stream)
'''.replace('FAIL', "raise RuntimeError('cleanup-failed')" if failure else 'pass')
    _, spec, root = prepare_route(tmp_path, plugin)
    response = route_response(spec, root)
    async def run():
        finished = asyncio.Event()
        events = []
        async def receive():
            await finished.wait()
            return {'type': 'http.disconnect'}
        async def send(message):
            events.append(message)
            if message['type'] == 'http.response.body' and not message.get('more_body'):
                finished.set()
        await response({'type': 'http', 'method': 'GET'}, receive, send)
        return events
    events = asyncio.run(run())
    assert events[0]['status'] == 200
    assert events[-1]['body'] == b'delivered'
    assert (root / 'background.done').read_text() == 'isolated-native-risk'
    if failure:
        assert 'cleanup-failed' in caplog.text


def test_midbody_error_aborts_instead_of_success(tmp_path):
    from ouroboros.extension_process_runner import ExtensionProcessError
    _, spec, root = prepare_route(tmp_path, '''from starlette.responses import StreamingResponse
async def chunks():
    yield b'partial'
    raise RuntimeError('midbody-failed')
def stream(request):
    return StreamingResponse(chunks())
def register(api):
    api.register_route('stream', stream)
''')
    response = route_response(spec, root)
    with pytest.raises(ExtensionProcessError, match='midbody-failed'):
        asyncio.run(collect_response(response))
    assert response.body_complete is False


def test_child_uses_selected_roots(tmp_path, monkeypatch):
    selected_app = tmp_path / 'explicit-app-root'
    monkeypatch.setenv('OUROBOROS_APP_ROOT', str(selected_app))
    _, spec, root = prepare_route(tmp_path, '''import os, sys, pathlib, ouroboros
def stream(request):
    return {'finder': any('__editable___ouroboros_' in getattr(f, '__module__', '') for f in sys.meta_path),
            'module': ouroboros.__file__, 'python': sys.executable,
            'roots': {k: os.environ[k] for k in ('OUROBOROS_APP_ROOT','OUROBOROS_REPO_DIR','OUROBOROS_DATA_DIR','OUROBOROS_SETTINGS_PATH')}}
def register(api):
    api.register_route('stream', stream)
''')
    events = asyncio.run(collect_response(route_response(spec, root)))
    body = json.loads(b''.join(e.get('body', b'') for e in events))
    assert body['python'] == sys.executable
    print('CHILD_BINDING', json.dumps(body, sort_keys=True))
    assert pathlib.Path(body['module']).resolve().is_relative_to(pathlib.Path(__file__).resolve().parents[1])
    assert body['roots']['OUROBOROS_DATA_DIR'] == str(root)
    assert body['roots']['OUROBOROS_SETTINGS_PATH'] == str(root / 'settings.json')
    assert body['roots']['OUROBOROS_APP_ROOT'] == str(selected_app), 'a separately selected data root does not replace APP_ROOT'


def test_disconnect_before_headers_reaps_the_child(tmp_path):
    _, spec, root = prepare_route(tmp_path, '''import asyncio
async def stream(request):
    await asyncio.sleep(90)
def register(api):
    api.register_route('stream', stream)
''')
    from ouroboros.tools.shell import _active_subprocesses
    response = route_response(spec, root)
    async def run():
        async def receive():
            while not _active_subprocesses:
                await asyncio.sleep(.01)
            return {'type': 'http.disconnect'}
        async def send(_message):
            pytest.fail('disconnected request should not publish headers')
        await response({'type': 'http', 'method': 'GET'}, receive, send)
        assert response.client_disconnected is True
    asyncio.run(run())
    assert not _active_subprocesses


def test_unload_cancels_only_the_observed_skill_instance(tmp_path):
    plugin = '''import asyncio
from starlette.responses import StreamingResponse
def stream(request):
    async def chunks():
        yield b'first'
        await asyncio.sleep(90)
    return StreamingResponse(chunks())
def register(api):
    api.register_route('stream', stream)
'''
    one_root = tmp_path / 'one'
    two_root = tmp_path / 'two'
    one_root.mkdir()
    two_root.mkdir()
    _, one, root_one = prepare_route(one_root, plugin, name='one')
    _, two, root_two = prepare_route(two_root, plugin, name='two')
    first = route_response(one, root_one)
    second = route_response(two, root_two)
    async def run():
        first_ready, second_ready = asyncio.Event(), asyncio.Event()
        async def observe_one(message):
            if message.get('body') == b'first':
                first_ready.set()
        async def observe_two(message):
            if message.get('body') == b'first':
                second_ready.set()
        tasks = [asyncio.create_task(collect_response(first, observer=observe_one)),
                 asyncio.create_task(collect_response(second, observer=observe_two))]
        try:
            await asyncio.wait_for(asyncio.gather(first_ready.wait(), second_ready.wait()), 20)
            assert extension_loader.unload_extension('one', expected_generation='not-this-generation') is False
            assert not first.cancelled and not second.cancelled
            assert extension_loader.unload_extension('one', expected_generation=one['extension_generation']) is True
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert not second.cancelled and not tasks[1].done()
        finally:
            second.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.run(run())
    from ouroboros.tools.shell import _active_subprocesses
    assert not _active_subprocesses


@pytest.mark.integration
def test_slow_reader_outlives_the_retired_sixty_second_limit(tmp_path):
    """A real child blocked by delivery backpressure remains alive beyond 60s."""
    plugin = '''from starlette.responses import FileResponse
def stream(request):
    return FileResponse(request.app.state.drive_root / 'large.bin')
def register(api):
    api.register_route('stream', stream)
'''
    _, spec, root = prepare_route(tmp_path, plugin)
    (root / 'large.bin').write_bytes(b'x' * (2 * 1024 * 1024))
    from ouroboros.tools.shell import _active_subprocesses
    paused = False
    async def slow(message):
        nonlocal paused
        if message.get('body') and not paused:
            paused = True
            await asyncio.sleep(65)
            assert any(proc.poll() is None for proc in _active_subprocesses)
    events = asyncio.run(collect_response(route_response(spec, root), observer=slow))
    assert paused
    assert sum(len(e.get('body', b'')) for e in events) == 2 * 1024 * 1024
    assert not _active_subprocesses


def test_failure_sending_final_body_is_not_cleanup_success(tmp_path):
    _, spec, root = prepare_route(tmp_path, "def stream(request):\n    return 'final'\ndef register(api):\n    api.register_route('stream', stream)\n")
    response = route_response(spec, root)
    async def fail_send(message):
        if message['type'] == 'http.response.body':
            raise OSError('client disconnected during final send')
    asyncio.run(collect_response(response, observer=fail_send))
    assert response.client_disconnected is True
    assert response.body_complete is False


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
def test_content_length_close_preserves_file_background(tmp_path, method):
    _, spec, root = prepare_route(tmp_path, """from starlette.responses import FileResponse
from starlette.background import BackgroundTask
def stream(request):
    def after():
        (request.app.state.drive_root / 'file-background').write_text('finished')
    return FileResponse(request.app.state.drive_root / 'large.bin', background=BackgroundTask(after))
def register(api):
    api.register_route('stream', stream)
""")
    size = 64 * 1024
    (root / 'large.bin').write_bytes(b'x' * size)
    response = route_response(spec, root, method=method)
    async def run():
        disconnected, sent = asyncio.Event(), 0
        async def receive():
            await disconnected.wait()
            return {'type':'http.disconnect'}
        async def send(message):
            nonlocal sent
            if message['type'] == 'http.response.start' and method == 'HEAD':
                disconnected.set()
                await asyncio.sleep(.01)
            elif message['type'] == 'http.response.body':
                if disconnected.is_set():
                    raise OSError('client closed after promised body')
                sent += len(message.get('body', b''))
                if sent == size:
                    assert message['more_body'] is True, 'exact FileResponse block ends with a later empty body'
                    disconnected.set()
                    await asyncio.sleep(.01)
        await response({'type':'http','method':method}, receive, send)
        assert sent == (0 if method == 'HEAD' else size)
    asyncio.run(run())
    assert response.body_complete is True
    assert (root / 'file-background').read_text() == 'finished'


def test_disconnect_during_body_runs_child_generator_cleanup(tmp_path):
    _, spec, root = prepare_route(tmp_path, """import asyncio
from starlette.responses import StreamingResponse
def stream(request):
    async def chunks():
        try:
            yield b'first'
            await asyncio.sleep(90)
        finally:
            (request.app.state.drive_root / 'cancel-cleanup').write_text('done')
    return StreamingResponse(chunks())
def register(api):
    api.register_route('stream', stream)
""")
    response = route_response(spec, root)
    async def run():
        first = asyncio.Event()
        async def receive():
            await first.wait()
            return {'type':'http.disconnect'}
        async def send(message):
            if message.get('body') == b'first':
                first.set()
        await response({'type':'http','method':'GET'}, receive, send)
    asyncio.run(run())
    assert response.client_disconnected and not response.body_complete
    assert (root / 'cancel-cleanup').read_text() == 'done'
    from ouroboros.tools.shell import _active_subprocesses
    assert not _active_subprocesses


def test_frame_writer_handles_partial_pipe_writes():
    import io
    from ouroboros.extension_route_stream import _write_frame, _read_frame
    class PartialPipe(io.BytesIO):
        def write(self, data):
            return super().write(data[:3])
    pipe = PartialPipe()
    _write_frame(pipe, b'B', b'bytes across several writes')
    pipe.seek(0)
    assert _read_frame(pipe) == (b'B', b'bytes across several writes')


def test_reload_does_not_let_old_response_cancel_new_generation(tmp_path):
    loaded, spec, root = prepare_route(tmp_path, '''import asyncio
from starlette.responses import StreamingResponse
def stream(request):
    async def chunks():
        yield b'first'
        await asyncio.sleep(90)
    return StreamingResponse(chunks())
def register(api):
    api.register_route('stream', stream)
''')
    old = route_response(spec, root)
    async def run():
        ready = asyncio.Event()
        async def observe(message):
            if message.get('body') == b'first':
                ready.set()
        old_task = asyncio.create_task(collect_response(old, observer=observe))
        await asyncio.wait_for(ready.wait(), 20)
        extension_loader.unload_extension(loaded.name, expected_generation=spec['extension_generation'])
        with pytest.raises(asyncio.CancelledError):
            await old_task
        assert await asyncio.to_thread(extension_loader.load_extension, loaded, lambda: {}, drive_root=root) is None
        newer = extension_loader.list_routes()[spec['path']]
        assert newer['extension_generation'] != spec['extension_generation']
        current = route_response(newer, root)
        ready.clear()
        current_task = asyncio.create_task(collect_response(current, observer=observe))
        try:
            await asyncio.wait_for(ready.wait(), 20)
            old.cancel()
            assert extension_loader.unload_extension(loaded.name, expected_generation=spec['extension_generation']) is False
            assert not current.cancelled and not current_task.done()
        finally:
            current.cancel()
            await asyncio.gather(current_task, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.serial
def test_real_http_second_route_and_module_respond_during_child_startup(widget_server, monkeypatch):
    import threading
    import httpx
    from ouroboros import process_custody
    from ouroboros.tools.shell import _active_subprocesses

    from ouroboros.extension_route_stream import RouteStreamResponse

    settled = threading.Event()
    completed = 0
    original_response = RouteStreamResponse.__call__
    async def observe_completion(self, *args):
        nonlocal completed
        try:
            return await original_response(self, *args)
        finally:
            completed += 1
            if completed == 2:
                settled.set()
    monkeypatch.setattr(RouteStreamResponse, "__call__", observe_completion)
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = process_custody.record_process

    def held_record(*args, **kwargs):
        calls.append(kwargs["pid"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(15), "test did not release its startup barrier"
        return original(*args, **kwargs)

    monkeypatch.setattr(process_custody, "record_process", held_record)

    async def run():
        async with httpx.AsyncClient(base_url=widget_server["url"], timeout=10) as client:
            first = asyncio.create_task(client.get("/api/extensions/export_widget/stream"))
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                module = await client.get("/api/extensions/export_widget/module/widget.js")
                assert module.status_code == 200 and b"OuroborosWidget" in module.content
                assert len(calls) == 1, "captured static module sources must not spawn children"
                second = await client.get("/api/extensions/export_widget/export")
                assert second.status_code == 200
                assert second.content == widget_server["expected"]["widget-large.bin"]
                assert len(calls) == 2 and not first.done()
            finally:
                release.set()
                (widget_server["root"] / "release-stream").touch()
                result = await first
            assert result.status_code == 200 and result.content == b"firstlast"
    asyncio.run(run())
    # Receiving the final HTTP byte precedes the permitted child/background cleanup.
    assert settled.wait(10), "both actual response handles must finish cleanup"
    assert not _active_subprocesses


@pytest.mark.serial
@pytest.mark.parametrize("startup_fails", [False, True])
def test_real_http_startup_cancel_retains_spawned_child_until_cleanup(widget_server, monkeypatch, startup_fails):
    import threading
    import httpx
    from ouroboros import process_custody, extension_registry_state
    from ouroboros.tools.shell import _active_subprocesses

    entered, release = threading.Event(), threading.Event()
    children = []
    original = process_custody.record_process

    def held_record(*args, **kwargs):
        children.extend(proc for proc in _active_subprocesses if proc.pid == kwargs["pid"])
        entered.set()
        assert release.wait(15), "test did not release its startup barrier"
        result = original(*args, **kwargs)
        if startup_fails:
            raise RuntimeError("controlled failure after real process registration")
        return result

    monkeypatch.setattr(process_custody, "record_process", held_record)

    async def run():
        async with httpx.AsyncClient(base_url=widget_server["url"], timeout=10) as client:
            request = asyncio.create_task(client.get("/api/extensions/export_widget/stream"))
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                assert len(children) == 1 and children[0].poll() is None
                with extension_registry_state._lock:
                    bundle = extension_registry_state._extensions["export_widget"]
                    work, = bundle.supervised_futures
                work.cancel()
                work.loop.call_soon_threadsafe(work.task.cancel)  # repeated cancellation
                module = await client.get("/api/extensions/export_widget/module/widget.js")
                assert module.status_code == 200
                assert not work.task.done(), "startup worker still owns an unreturned child"
            finally:
                release.set()
                try:
                    response = await request
                    assert response.status_code == 500  # cancelled before ASGI headers
                except httpx.RemoteProtocolError:
                    pass  # uvicorn may close a cancelled unstarted response
            for _ in range(100):
                if work.task.done():
                    break
                await asyncio.sleep(.01)
            assert work.task.done()
            assert all(proc.poll() is not None for proc in children)
            assert not _active_subprocesses
            assert work not in bundle.supervised_futures
            calls_dir = widget_server["root"] / "state/skills/export_widget/extension_calls"
            assert list(calls_dir.iterdir()) == []
    asyncio.run(run())


@pytest.mark.parametrize("kind,limit", [(b"B", "body"), (b"S", "metadata")])
def test_stream_frame_allocation_is_bounded_without_limiting_response_length(kind, limit):
    import io
    import struct
    from ouroboros.config import EXTENSION_STREAM_CHUNK_BYTES, EXTENSION_STREAM_METADATA_BYTES
    from ouroboros.extension_route_stream import _read_frame

    bound = EXTENSION_STREAM_CHUNK_BYTES + 1 if limit == "body" else EXTENSION_STREAM_METADATA_BYTES
    class Observed(io.BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.read_sizes = []
        def read(self, size=-1):
            self.read_sizes.append(size)
            return super().read(size)
    over = Observed(struct.pack("!I", bound + 2) + kind)
    with pytest.raises(ValueError, match="channel bound"):
        _read_frame(over)
    assert over.read_sizes == [4, 1], "oversized payload must be refused before allocation"
    huge = Observed(struct.pack("!I", 0xffffffff) + kind)
    with pytest.raises(ValueError, match="channel bound"):
        _read_frame(huge)
    assert huge.read_sizes == [4, 1]
    # Multiple legal frames remain readable; there is no cumulative body cap.
    frame = struct.pack("!I", bound + 1) + kind + b"x" * bound
    valid = Observed(frame * 3)
    for _ in range(3):
        assert _read_frame(valid) == (kind, b"x" * bound)


def test_native_route_crash_retains_drained_stderr_and_exit_code(tmp_path, caplog):
    import logging
    from ouroboros.tools.shell import _active_subprocesses

    _, spec, drive = prepare_route(tmp_path, """import os, sys
def stream(request):
    sys.stderr.write('CONTROLLED_NATIVE_CRASH\\n')
    sys.stderr.flush()
    os._exit(7)
def register(api):
    api.register_route('stream', stream)
""")
    with caplog.at_level(logging.WARNING, logger="ouroboros.extension_route_stream"):
        messages = asyncio.run(collect_response(route_response(spec, drive)))
    assert messages[0]["status"] == 502
    assert "CONTROLLED_NATIVE_CRASH" in caplog.text and "returncode=7" in caplog.text
    assert not _active_subprocesses
