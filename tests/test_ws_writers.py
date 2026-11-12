"""Cost law L10: the WebSocket fan-out encodes only for open sockets and keeps one writer each.

Every ``log`` frame used to be JSON-encoded on the event loop before anyone looked for a
client, each frame was its own coroutine scheduled across threads, and a failed send appended
its diagnostic row synchronously on the loop. Now a frame with no open socket is dropped
unencoded and wakes nothing; with sockets open it is encoded once, a burst reaches the loop in
one flush, and each socket's single writer sends the identical bytes in order. E3: the frames
the producer pushes are identical with 0 and 2 clients — only the encode and the sends differ.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

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
