"""Supervised extension work runs on the host extension loop, not the HTTP loop."""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import threading
import time
import urllib.request
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from ouroboros import event_bus, extension_companion, extension_plugin_api, server_boot
from ouroboros.extension_registry_state import _StagedSupervisedTask


@pytest.fixture
def restored_host():
    saved_bus = event_bus._GLOBAL_BUS
    saved_supervisor = extension_companion._GLOBAL_SUPERVISOR
    saved_pid = extension_companion._SERVER_PROCESS_PID
    saved_env = os.environ.get("OUROBOROS_SERVER_PROCESS_PID")
    yield
    current = event_bus._GLOBAL_BUS
    if current is not None and current is not saved_bus:
        current.stop_loop(4)
    event_bus._GLOBAL_BUS = saved_bus
    extension_companion._GLOBAL_SUPERVISOR = saved_supervisor
    extension_companion._SERVER_PROCESS_PID = saved_pid
    if saved_env is None:
        os.environ.pop("OUROBOROS_SERVER_PROCESS_PID", None)
    else:
        os.environ["OUROBOROS_SERVER_PROCESS_PID"] = saved_env


@contextlib.contextmanager
def _quiet_host_bind(monkeypatch):
    @contextlib.contextmanager
    def bound(*_args, **_kwargs):
        yield object()

    class _Server:
        def __init__(self, _config):
            self.should_exit = False

        async def serve(self, sockets=None):
            await asyncio.Future()

    monkeypatch.setattr(server_boot, "bound_service_socket", bound)
    monkeypatch.setattr(server_boot, "_embedded_uvicorn_server", lambda config: _Server(config))
    yield


def _host():
    return SimpleNamespace(_skill="demo", _disclose_model_capable_dispatch=lambda *_a, **_k: None)


def _spec(factory):
    return _StagedSupervisedTask(
        name="worker", factory=factory, restart_policy="never", max_restarts=0, backoff_seconds=0.01,
    )


async def _answer_ok(reader, writer):
    await reader.read(1024)
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
    await writer.drain()
    writer.close()


def test_supervised_sleep_does_not_delay_an_http_request(tmp_path, monkeypatch, restored_host):
    """A supervised time.sleep stays off the loop that is serving HTTP."""

    async def main():
        http = await asyncio.start_server(_answer_ok, "127.0.0.1", 0)
        port = http.sockets[0].getsockname()[1]
        boot = SimpleNamespace(
            host_service_task=None, host_service_server=None, host_service_listener=ExitStack(),
        )
        try:
            await server_boot.serve_host_service(tmp_path, boot)
            http_loop = asyncio.get_running_loop()
            bus = event_bus.get_global_event_bus()
            assert bus._loop is not http_loop and bus._loop.is_running()
            entered = threading.Event()

            async def block():
                entered.set()
                time.sleep(2.0)

            future = extension_plugin_api.PluginAPIImpl._start_supervised_task(_host(), _spec(block))
            assert future is not None
            assert await asyncio.to_thread(entered.wait, 2)
            started = time.monotonic()

            def request():
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    return response.read()

            assert await asyncio.to_thread(request) == b"ok"
            assert time.monotonic() - started < 1.0
        finally:
            task = boot.host_service_task
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            boot.host_service_listener.close()
            http.close()
            await http.wait_closed()

    with _quiet_host_bind(monkeypatch):
        asyncio.run(main())


def test_stop_loop_runs_a_supervised_tasks_finally(monkeypatch):
    bus = event_bus.EventBus()
    bus.start_loop()
    started = threading.Event()
    finished = threading.Event()

    async def work():
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            finished.set()

    monkeypatch.setattr(extension_plugin_api, "get_global_event_bus", lambda: bus)
    monkeypatch.setattr(extension_plugin_api, "is_server_process", lambda: True)
    try:
        future = extension_plugin_api.PluginAPIImpl._start_supervised_task(_host(), _spec(work))
        assert future is not None and started.wait(2)
        bus.stop_loop(2)
    finally:
        if bus._owns_loop:
            bus.stop_loop(1)
    assert finished.is_set() and future.done()
    assert not any(thread.name == "extension-loop" and thread.is_alive() for thread in threading.enumerate())


def test_supervised_task_starts_when_host_service_precedes_extensions(tmp_path, monkeypatch, restored_host):
    """The extensions step runs as soon as host_service returns, and the loop is already running."""
    started = threading.Event()

    async def work():
        started.set()
        await asyncio.sleep(30)

    async def main():
        boot = SimpleNamespace(
            host_service_task=None, host_service_server=None, host_service_listener=ExitStack(),
        )
        try:
            await server_boot.serve_host_service(tmp_path, boot)
            bus = event_bus.get_global_event_bus()
            again = bus.start_loop()
            assert again is bus._loop and again.is_running()
            assert sum(1 for thread in threading.enumerate() if thread.name == "extension-loop") == 1
            future = extension_plugin_api.PluginAPIImpl._start_supervised_task(_host(), _spec(work))
            assert future is not None
            assert await asyncio.to_thread(started.wait, 2)
        finally:
            event_bus.get_global_event_bus().stop_loop(2)
            task = boot.host_service_task
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            boot.host_service_listener.close()

    with _quiet_host_bind(monkeypatch):
        asyncio.run(main())
    assert started.is_set()


def test_lifespan_teardown_stops_the_extension_loop():
    text = (pathlib.Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8")
    start = text.index("stop_owned_work(lifespan_drive_root)")
    assert "server_boot.stop_extension_loop()" in text[start:start + 200]


def test_set_loop_keeps_running_when_stop_loop_is_called():
    async def main():
        bus = event_bus.EventBus()
        loop = asyncio.get_running_loop()
        bus.set_loop(loop)
        bus.stop_loop(0.2)
        assert loop.is_running()
        received = []

        async def handler(payload):
            received.append(payload["text"])

        bus.subscribe("skill", event_bus.CHAT_OUTBOUND, handler)
        bus.publish(event_bus.CHAT_OUTBOUND, {"text": "hello"})
        await asyncio.sleep(0.05)
        assert received == ["hello"]

    asyncio.run(main())
