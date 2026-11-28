"""Cost law L10: the WebSocket fan-out encodes only for open sockets and keeps one writer each.

Every ``log`` frame used to be JSON-encoded on the event loop before anyone looked for a
client, each frame was its own coroutine scheduled across threads, and a failed send appended
its diagnostic row synchronously on the loop. Now a frame with no open socket is dropped
unencoded and wakes nothing; with sockets open it is encoded once, a burst reaches the loop in
one flush, and each socket's single writer sends the identical bytes in order. E3: the frames
the producer pushes are identical with 0 and 2 clients — only the encode and the sends differ.

Each socket's one disconnect line says what is known about why it closed: the ASGI close code as
received (none is ``unknown``), the browser watchdog's own allowlisted reason fields and nothing else
of a reason, and what the connection kept (last send and kind, a send still awaited, outbox depth,
the largest loop lateness it lived through, a write error or the server's own close). The lifecycle
lines leave the loop through one ordered handoff, so a slow log sink never holds a socket's frames.
"""
from __future__ import annotations

import asyncio
import pathlib
import re
import threading
import time

import logging

import pytest
from starlette.websockets import WebSocketDisconnect

from ouroboros.gateway import ws


class _Socket:
    def __init__(self, fail: bool = False) -> None:
        self.sent, self.fail = [], fail

    async def send_text(self, data):
        if self.fail:
            raise RuntimeError("broken pipe")
        self.sent.append(data)


@pytest.fixture
def server_loop(monkeypatch):
    """A running loop on its own thread, as the server's; the broadcasts come from this thread."""
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    monkeypatch.setattr(ws, "_event_loop", loop)
    monkeypatch.setattr(ws, "_ws_clients", [])
    monkeypatch.setattr(ws, "_pending", ws.collections.deque())
    monkeypatch.setattr(ws, "_flush_scheduled", False)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    runner.join(timeout=5)
    loop.close()


def _on(loop, fn, *args):
    async def call():
        return fn(*args)
    return asyncio.run_coroutine_threadsafe(call(), loop).result(timeout=5)


def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _counting(monkeypatch, loop):
    counts = {"encode": 0, "wake": 0}
    encode, wake = ws._encode, loop.call_soon_threadsafe

    def counted_encode(msg):
        counts["encode"] += 1
        return encode(msg)

    def counted_wake(*args, **kwargs):
        counts["wake"] += 1
        return wake(*args, **kwargs)

    monkeypatch.setattr(ws, "_encode", counted_encode)
    monkeypatch.setattr(loop, "call_soon_threadsafe", counted_wake)
    return counts


def _frames(n):
    return [{"type": "log", "data": {"type": "task_heartbeat", "task_id": "t", "n": i}, "chat_id": 0} for i in range(n)]


def test_with_no_open_socket_a_frame_is_neither_encoded_nor_wakes_the_loop(server_loop, monkeypatch):
    counts = _counting(monkeypatch, server_loop)
    assert all(ws.broadcast_ws_sync(frame) is True for frame in _frames(100)), "a running loop took them"
    assert counts == {"encode": 0, "wake": 0}
    assert list(ws._pending) == []


def test_two_sockets_receive_identical_bytes_encoded_once_in_order_by_one_writer_each(server_loop, monkeypatch):
    sockets = [_Socket(), _Socket()]
    connections = [_on(server_loop, ws._register, socket) for socket in sockets]
    tasks_before = _on(server_loop, lambda: len(asyncio.all_tasks()))
    counts = _counting(monkeypatch, server_loop)
    frames = _frames(50)
    assert all(ws.broadcast_ws_sync(frame) is True for frame in frames)
    _until(lambda: all(len(socket.sent) == 50 for socket in sockets))
    assert sockets[0].sent == sockets[1].sent == [ws.json.dumps(frame, ensure_ascii=False, default=str)
                                                  for frame in frames], "the same bytes, in order"
    assert counts["encode"] == 50, "each frame encoded once, not once per socket"
    assert 1 <= counts["wake"] <= 50, "a burst wakes the loop at most once per frame, never per socket"
    assert _on(server_loop, lambda: len(asyncio.all_tasks())) == tasks_before, "no coroutine per frame"
    for connection in connections:
        _on(server_loop, ws._unregister, connection)


def test_a_failed_send_retires_only_its_socket_and_writes_its_row_off_the_loop(server_loop, monkeypatch, tmp_path):
    monkeypatch.setattr(ws, "DATA_DIR", tmp_path)
    written = []
    monkeypatch.setattr(ws, "_record_failure", lambda root, row: written.append((threading.get_ident(), row)))
    good, bad = _Socket(), _Socket(fail=True)
    _on(server_loop, ws._register, bad)
    kept = _on(server_loop, ws._register, good)
    loop_thread = _on(server_loop, threading.get_ident)
    ws.broadcast_ws_sync({"type": "log", "data": {"type": "x"}})
    ws.broadcast_ws_sync({"type": "log", "data": {"type": "y"}})
    _until(lambda: len(good.sent) == 2 and written)
    assert ws._ws_clients == [kept]
    [(thread, row)] = written
    assert thread != loop_thread, "the diagnostic row is written by the executor, never on the loop"
    assert (row["type"], row["msg_type"], row["dead_clients"], row["total_clients"]) == (
        "broadcast_partial_failure", "log", 1, 2)
    _on(server_loop, ws._unregister, kept)


def test_an_async_broadcast_on_the_loop_keeps_the_order_of_frames_queued_before_it(server_loop):
    socket = _Socket()
    connection = _on(server_loop, ws._register, socket)

    async def mixed():
        ws._queue_frame({"type": "first"})  # as a supervisor-thread frame whose flush is still scheduled
        await ws.broadcast_ws({"type": "second"})

    asyncio.run_coroutine_threadsafe(mixed(), server_loop).result(timeout=5)
    _until(lambda: len(socket.sent) == 2)
    assert [ws.json.loads(text)["type"] for text in socket.sent] == ["first", "second"]
    _on(server_loop, ws._unregister, connection)


class _EndpointSocket(_Socket):
    """A socket for ``ws_endpoint``: its receive loop ends the way ``ending`` says once ``end`` is set."""

    app = None

    def __init__(self, ending: BaseException, fail: bool = False) -> None:
        super().__init__(fail)
        self.ending, self.end, self.closed = ending, asyncio.Event(), []

    async def accept(self):
        return None

    async def receive_text(self):
        await self.end.wait()
        raise self.ending

    async def close(self, code=1000, reason=""):
        self.closed.append(code)


def _disconnect_lines(caplog, count: int = 1) -> list[str]:
    def lines():
        return [record.getMessage() for record in caplog.records
                if record.getMessage().startswith("WebSocket client disconnected")]
    _until(lambda: len(lines()) >= count)
    return lines()


def _serve(socket, during=None) -> None:
    """Run ``ws_endpoint`` for ``socket`` on a fresh loop; ``during(connection)`` runs while it is open."""
    async def main():
        task = asyncio.create_task(ws.ws_endpoint(socket))
        while not ws._ws_clients:
            await asyncio.sleep(0)
        if during is not None:
            await during(ws._ws_clients[0])
        socket.end.set()
        await task

    asyncio.run(main())


@pytest.fixture
def isolated_registry(monkeypatch, caplog):
    monkeypatch.setattr(ws, "_ws_clients", [])
    monkeypatch.setattr(ws, "_pending", ws.collections.deque())
    monkeypatch.setattr(ws, "_flush_scheduled", False)
    caplog.set_level(logging.INFO, logger=ws.log.name)


def test_a_watchdog_close_logs_its_allowlisted_fields_and_drops_everything_else(isolated_registry, caplog):
    reason = "silent_ms=46012 tick_gap_ms=93004 hidden=1 note=owner-words silent_ms=1 tick_gap_ms=99999999999 x"
    _serve(_EndpointSocket(WebSocketDisconnect(ws.WATCHDOG_CLOSE_CODE, reason)))
    [line] = _disconnect_lines(caplog)
    assert " code=4408 silent_ms=46012 tick_gap_ms=93004 hidden=1 age=" in line
    for dropped in ("note", "owner-words", "99999999999", "silent_ms=1 "):
        assert dropped not in line, line
    assert "last_send=none in_flight=none outbox=0 beat_late_max=0.0s" in line


@pytest.mark.parametrize(("ending", "expected"), [
    (RuntimeError("receive failed"), " code=unknown "),
    (WebSocketDisconnect(1006, "silent_ms=5 hidden=1"), " code=1006 "),
    (WebSocketDisconnect(1005), " code=1005 "),
    (WebSocketDisconnect(True), " code=unknown "),  # a flag is no close code
])
def test_a_disconnect_without_our_code_logs_its_code_or_unknown_and_never_a_reason(isolated_registry, caplog,
                                                                                    ending, expected):
    _serve(_EndpointSocket(ending))
    [line] = _disconnect_lines(caplog)
    assert expected in line and "silent_ms" not in line and "hidden" not in line, line


def test_the_line_carries_what_the_connection_kept(isolated_registry, caplog):
    release = asyncio.Event()

    class _Held(_EndpointSocket):
        async def send_text(self, data):
            if '"type": "held"' in data:
                await release.wait()
            self.sent.append(data)

    socket = _Held(WebSocketDisconnect(1001))

    async def during(connection):
        await ws.broadcast_ws({"type": "heartbeat"})
        while connection.sent is None:
            await asyncio.sleep(0)
        for kind in ("held", "queued", "queued"):
            await ws.broadcast_ws({"type": kind})
        ws.note_loop_lateness(2.5)
        ws.note_loop_lateness(0.2)
        while connection.sending is None:
            await asyncio.sleep(0)

    try:
        _serve(socket, during)
    finally:
        release.set()
    [line] = _disconnect_lines(caplog)
    assert " code=1001 " in line and " last_send=heartbeat:" in line, line
    assert " in_flight=0." in line and " outbox=2 " in line and " beat_late_max=2.5s" in line, line


def test_a_writer_failure_and_the_servers_own_close_are_named_in_the_line(isolated_registry, caplog, monkeypatch):
    monkeypatch.setattr(ws, "_record_failure", lambda root, row: None)
    failing = _EndpointSocket(WebSocketDisconnect(1006), fail=True)

    async def broken(connection):
        await ws.broadcast_ws({"type": "log"})
        while ws._ws_clients:  # the failed writer retired it
            await asyncio.sleep(0)

    _serve(failing, broken)
    restarting = _EndpointSocket(WebSocketDisconnect(1012, "Server restarting"))

    async def restart(connection):
        await ws.close_all_ws()

    _serve(restarting, restart)
    closing = _EndpointSocket(WebSocketDisconnect(1012, "Server restarting"), fail=True)

    async def close_then_fail(connection):
        await ws.close_all_ws()
        await ws.broadcast_ws({"type": "log"})  # a send after the server's own close fails
        while ws._ws_clients:
            await asyncio.sleep(0)

    _serve(closing, close_then_fail)
    first, second, third = _disconnect_lines(caplog, 3)
    assert " code=1006 " in first and first.endswith(" cause=write_error:RuntimeError"), first
    assert restarting.closed == [1012]
    assert " code=1012 " in second and second.endswith(" cause=server_close:1012"), second
    assert third.endswith(" cause=server_close:1012"), "a later write error does not replace the server's close"


def test_the_browser_watchdog_closes_with_the_code_and_fields_this_module_parses():
    """Cross-language twins: ``web/modules/ws.js`` sends what ``watchdog_reason_fields`` keeps."""
    source = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "ws.js").read_text(encoding="utf-8")
    assert re.search(r"export const WATCHDOG_CLOSE_CODE = (\d+);", source).group(1) == str(ws.WATCHDOG_CLOSE_CODE)
    builder = re.search(r"export function watchdogCloseReason\(.*?\n}\n", source, re.S).group(0)
    assert re.findall(r"(\w+)=\$\{", builder) == list(ws._WATCHDOG_REASON_FIELDS)


class _HeldSink(logging.Handler):
    """A log sink that waits for ``release`` before every write, and notes the registry at that moment."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.release = threading.Event()
        self.lines: list[str] = []
        self.registry: list[list] = []

    def emit(self, record):
        assert self.release.wait(10), "the test never released the sink"
        self.lines.append(record.getMessage())
        self.registry.append(list(ws._ws_clients))


def test_a_held_log_sink_never_holds_the_loop_and_a_retired_client_has_left_before_its_line(server_loop, caplog,
                                                                                             monkeypatch):
    """The lifecycle lines leave the loop through one handoff: while the sink waits, the loop goes on handing
    frames (the heartbeat among them) to a healthy socket; once released, the send-failure line says the count
    after the dead client left the registry, as the registry itself does when the line is written."""
    monkeypatch.setattr(ws, "_record_failure", lambda root, row: None)
    caplog.set_level(logging.INFO, logger=ws.log.name)
    sink = _HeldSink()
    ws.log.addHandler(sink)
    try:
        dead = _on(server_loop, ws._register, _Socket(fail=True))
        good = _Socket()
        kept = _on(server_loop, ws._register, good)
        ws.broadcast_ws_sync({"type": "log", "data": {"type": "first"}})
        _until(lambda: dead not in ws._ws_clients)
        for frame in ({"type": "heartbeat"}, {"type": "log", "data": {"type": "later"}}):
            ws.broadcast_ws_sync(frame)
        _until(lambda: len(good.sent) == 3)
        assert sink.lines == [], "the loop served on while its failure line waited for the sink"
        sink.release.set()
        _until(lambda: sink.lines)
    finally:
        sink.release.set()
        ws.log.removeHandler(sink)
    [line] = sink.lines
    assert line == f"WebSocket send failed for msg type=log; dropped client {dead.serial} (RuntimeError) (total: 1)"
    assert sink.registry == [[kept]]
    _on(server_loop, ws._unregister, kept)
