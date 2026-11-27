"""The ASGI loop's own beat (``ouroboros/asgi_beat.py``).

The heartbeat coroutine stamps how late the serving event loop runs, on that loop; the supervisor's liveness
watchdog, on its own thread, journals one ``asgi_loop_late`` row with the loop thread's stack per episode in
which a beat falls a whole interval behind while browser sockets are open: never for a loop that beats on
time, never without an open socket, never twice for one episode. Every case runs the real heartbeat on a real
event loop thread, or the real watchdog thread on a hand-moved clock.
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import PurePath

from ouroboros import asgi_beat, server_liveness, server_runtime

BEAT_SEC = 0.05


def _until(predicate, budget: float = 5.0) -> None:
    end = time.monotonic() + budget
    while not predicate():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


class _ServingLoop:
    """A real event loop on its own thread running the real heartbeat, as the server's lifespan does."""

    def __init__(self, on_beat=None) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="test-asgi-loop", daemon=True)
        self.thread.start()
        heartbeat = server_runtime.ws_heartbeat_loop(lambda: False, None, BEAT_SEC, on_beat)
        self.task = asyncio.run_coroutine_threadsafe(self._start(heartbeat), self.loop).result(5)

    async def _start(self, heartbeat):
        return asyncio.ensure_future(heartbeat)

    def hold(self, release: threading.Event) -> threading.Event:
        """Block the loop thread inside ``_hold_the_loop`` until ``release``; returns once it is held."""
        held = threading.Event()

        def _hold_the_loop() -> None:
            held.set()
            release.wait(10)

        self.loop.call_soon_threadsafe(_hold_the_loop)
        assert held.wait(5)
        return held

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.task.cancel)
        _until(self.task.done)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


def test_the_heartbeat_stamps_each_beat_with_its_lateness_and_unstamps_when_it_ends(monkeypatch):
    beat = asgi_beat.AsgiBeat()
    monkeypatch.setattr(server_runtime, "ASGI_BEAT", beat)
    late: list[float] = []
    serving = _ServingLoop(on_beat=late.append)
    release = threading.Event()
    try:
        _until(lambda: len(late) >= 2)
        assert beat.ident == serving.thread.ident and beat.interval == BEAT_SEC
        serving.hold(release)
        time.sleep(0.3)
        release.set()
        _until(lambda: any(value >= 0.2 for value in late))  # the held beat came 0.3 s late, measured on the loop
    finally:
        release.set()
        serving.stop()
    assert beat.stamp is None, "the stamp ends with its heartbeat: nothing beats, nothing is overdue"
    assert beat.overdue(time.monotonic()) == 0.0


def test_a_blocked_loop_gives_one_row_with_its_stack_and_a_healthy_or_unwatched_loop_none(monkeypatch):
    beat = asgi_beat.AsgiBeat()
    monkeypatch.setattr(server_runtime, "ASGI_BEAT", beat)
    sockets_open = [True]
    watch = asgi_beat.AsgiLateWatch(server_liveness._loop_thread_stack, lambda: sockets_open[0], beat)
    serving = _ServingLoop()
    release = threading.Event()
    try:
        _until(lambda: beat.stamp is not None)
        assert watch.tick(time.monotonic()) is None, "a loop that beats on time has no row"
        serving.hold(release)
        time.sleep(3 * BEAT_SEC)  # past due by more than one whole beat
        row = watch.tick(time.monotonic())
        assert row is not None and row["type"] == "asgi_loop_late" and row["resumed"] is False, row
        assert row["overdue_sec"] >= 0.0 and set(row) <= {"type", "overdue_sec", "resumed", "stack", "stack_truncated"}
        assert any(frame.endswith(" in _hold_the_loop") for frame in row["stack"]), row["stack"]
        assert watch.tick(time.monotonic()) is None, "one episode, one row"
        stamp = beat.stamp
        release.set()
        _until(lambda: beat.stamp != stamp)
        assert watch.tick(time.monotonic()) is None, "the loop beats on time again"
        sockets_open[0] = False
        release.clear()
        serving.hold(release)
        time.sleep(3 * BEAT_SEC)
        assert watch.tick(time.monotonic()) is None, "no open socket: nobody waits on this loop's beat"
        sockets_open[0] = True
        assert watch.tick(time.monotonic())["type"] == "asgi_loop_late", "the same episode, now with a socket"
    finally:
        release.set()
        serving.stop()


class _HandClock:
    """The ``time`` the watchdog reads: a monotonic clock the test sets, and a sleep that only yields."""

    def __init__(self, mono: float) -> None:
        self.mono = mono

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, _seconds: float) -> None:
        time.sleep(0.01)


def test_the_liveness_watchdog_journals_a_late_asgi_beat_once(monkeypatch):
    from ouroboros.gateway import ws
    from supervisor.active_activity import get_direct_activity_registry

    rows: list = []
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda path, obj: rows.append((str(path), dict(obj))) or True)
    monkeypatch.setattr("supervisor.state.load_state", lambda: {})
    get_direct_activity_registry().clear()
    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    clock = _HandClock(time.monotonic() + 1000.0)
    monkeypatch.setattr(server_liveness, "time", clock)
    monkeypatch.setattr(ws, "has_ws_clients", lambda: True)
    for name, value in (("stamp", clock.mono - 100.0), ("interval", 15.0), ("ident", threading.get_ident())):
        monkeypatch.setattr(asgi_beat.ASGI_BEAT, name, value)
    liveness = [clock.mono, {"phase": "events"}, time.thread_time(), None]  # the supervisor loop itself is fresh
    stop = threading.Event()
    try:
        server_liveness._start_supervisor_liveness_watchdog(liveness, stop)
        _until(lambda: rows)
        time.sleep(0.1)  # several more watchdog ticks over the same overdue beat
    finally:
        stop.set()
        for thread in threading.enumerate():
            if thread.name == "supervisor-liveness-watchdog":
                thread.join(5)
    [(path, row)] = rows
    assert PurePath(path).as_posix().endswith("logs/supervisor.jsonl")
    assert row["type"] == "asgi_loop_late" and row["overdue_sec"] == 85.0 and row["resumed"] is False
    assert row["stack"] and "ts" in row
