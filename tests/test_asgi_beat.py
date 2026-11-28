"""The ASGI loop's own beat (``ouroboros/asgi_beat.py``).

The heartbeat coroutine stamps how late the serving event loop runs, on that loop; the supervisor's liveness
watchdog, on its own thread, journals one ``asgi_loop_late`` row with the loop thread's stack per episode in
which a beat falls a whole interval behind while browser sockets are open: never for a loop that beats on
time, never without an open socket, never twice for one episode, never while the server stops or for a
generation that ended while its watchdog slept. The watchdog looks at the beat before it writes any row of its
pass, and the row carries the watchdog's own lateness (a blocked loop or a process that stood still). Every
case runs the real heartbeat on a real event loop thread, or the real watchdog thread on a hand-moved clock.
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import PurePath

import pytest

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
        assert watch.tick(time.monotonic(), 0.0) is None, "a loop that beats on time has no row"
        serving.hold(release)
        time.sleep(3 * BEAT_SEC)  # past due by more than one whole beat
        row = watch.tick(time.monotonic(), 0.0)
        assert row is not None and row["type"] == "asgi_loop_late" and row["resumed"] is False, row
        assert row["overdue_sec"] >= 0.0 and row["watchdog_late_sec"] == 0.0
        assert set(row) <= {"type", "overdue_sec", "watchdog_late_sec", "resumed", "stack", "stack_truncated"}
        assert any(frame.endswith(" in _hold_the_loop") for frame in row["stack"]), row["stack"]
        assert watch.tick(time.monotonic(), 0.0) is None, "one episode, one row"
        stamp = beat.stamp
        release.set()
        _until(lambda: beat.stamp != stamp)
        assert watch.tick(time.monotonic(), 0.0) is None, "the loop beats on time again"
        sockets_open[0] = False
        release.clear()
        serving.hold(release)
        time.sleep(3 * BEAT_SEC)
        assert watch.tick(time.monotonic(), 0.0) is None, "no open socket: nobody waits on this loop's beat"
        sockets_open[0] = True
        assert watch.tick(time.monotonic(), 0.0)["type"] == "asgi_loop_late", "the same episode, now with a socket"
    finally:
        release.set()
        serving.stop()


@pytest.mark.parametrize("flag", ["_supervisor_stop", "_exit_signalled", "_restart_requested"])
def test_a_stopping_server_has_no_row_and_the_same_late_beat_rows_once_it_serves_on(monkeypatch, flag):
    """A lifespan teardown, a process exit or a restart holds the loop with its own work: while one is latched an
    overdue beat is no evidence; the very same beat is reported once nothing stops the server."""
    from ouroboros import server_process

    latched = threading.Event()
    monkeypatch.setattr(server_process, flag, latched)
    beat, now = asgi_beat.AsgiBeat(), time.monotonic()
    beat.stamp, beat.interval, beat.ident = now - 100.0, 15.0, threading.get_ident()
    watch = asgi_beat.AsgiLateWatch(server_liveness._loop_thread_stack, lambda: True, beat)
    latched.set()
    assert watch.tick(now, 0.0) is None, f"{flag} is latched: the server is stopping"
    latched.clear()
    row = watch.tick(now, 0.0)
    assert row is not None and row["type"] == "asgi_loop_late" and row["overdue_sec"] == 85.0, row


class _HandClock:
    """The ``time`` the watchdog reads: a monotonic clock the test sets, and a sleep that only yields (or, with
    ``oversleep``, wakes that much later than it asked, as a process that stood still does)."""

    def __init__(self, mono: float, oversleep: float = 0.0) -> None:
        self.mono, self.oversleep, self.on_sleep = mono, oversleep, None

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, seconds: float) -> None:
        if self.on_sleep is not None:
            self.on_sleep()
        time.sleep(0.01)
        if self.oversleep:
            self.mono += seconds + self.oversleep


def _watch_a_late_beat(monkeypatch, clock: _HandClock, stop: threading.Event, *, supervisor_age: float = 0.0,
                       on_row=None) -> list:
    """Start the real liveness watchdog of the generation ``stop`` ends on ``clock``, over an ASGI beat 100 s old
    while a socket is open; returns its journal rows as they arrive (``on_row`` sees each one as it is written)."""
    from ouroboros.gateway import ws
    from supervisor.active_activity import get_direct_activity_registry

    rows: list = []

    def journal(path, obj):
        if on_row is not None:
            on_row(dict(obj))
        rows.append((str(path), dict(obj)))
        return True

    monkeypatch.setattr("supervisor.state.append_jsonl", journal)
    monkeypatch.setattr("supervisor.state.load_state", lambda: {})
    get_direct_activity_registry().clear()
    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")  # a one-second watchdog interval
    monkeypatch.setattr(server_liveness, "time", clock)
    monkeypatch.setattr(ws, "has_ws_clients", lambda: True)
    for name, value in (("stamp", clock.mono - 100.0), ("interval", 15.0), ("ident", threading.get_ident())):
        monkeypatch.setattr(asgi_beat.ASGI_BEAT, name, value)
    liveness = [clock.mono - supervisor_age, {"phase": "events"}, time.thread_time(), None]
    server_liveness._start_supervisor_liveness_watchdog(liveness, stop)
    return rows


def _end_watchdog(stop: threading.Event) -> None:
    stop.set()
    for thread in threading.enumerate():
        if thread.name == "supervisor-liveness-watchdog":
            thread.join(5)


def test_the_liveness_watchdog_journals_a_late_asgi_beat_once(monkeypatch):
    clock, stop = _HandClock(time.monotonic() + 1000.0), threading.Event()
    rows = _watch_a_late_beat(monkeypatch, clock, stop)  # the supervisor loop itself is fresh
    try:
        _until(lambda: rows)
        time.sleep(0.1)  # several more watchdog ticks over the same overdue beat
    finally:
        _end_watchdog(stop)
    [(path, row)] = rows
    assert PurePath(path).as_posix().endswith("logs/supervisor.jsonl")
    assert row["type"] == "asgi_loop_late" and row["overdue_sec"] == 85.0 and row["resumed"] is False
    assert row["watchdog_late_sec"] == 0.0, "the watchdog woke on time: the loop stood while the process ran"
    assert row["stack"] and "ts" in row


def test_a_process_that_stood_still_shows_in_the_watchdogs_own_lateness(monkeypatch):
    """The watchdog's own sleep ends 40 s late, as when the whole process (or the machine) stood still: the row
    says so, where a loop blocked while other threads ran says 0 (above)."""
    clock, stop = _HandClock(time.monotonic() + 1000.0, oversleep=40.0), threading.Event()
    rows = _watch_a_late_beat(monkeypatch, clock, stop)
    try:
        _until(lambda: any(row["type"] == "asgi_loop_late" for _path, row in rows))
    finally:
        _end_watchdog(stop)
    [late] = [row for _path, row in rows if row["type"] == "asgi_loop_late"]
    assert late["watchdog_late_sec"] == 40.0 and late["overdue_sec"] == 126.0, late


def test_a_watchdog_whose_generation_ended_during_its_sleep_writes_nothing(monkeypatch):
    """The generation's stop token is set while its watchdog sleeps (a failed init, an exit, a revival on the way):
    the watchdog asks again on waking and ends without a look, however late the beat it would have seen."""
    clock, stop = _HandClock(time.monotonic() + 1000.0), threading.Event()
    clock.on_sleep = stop.set  # the generation ends during the watchdog's first sleep
    rows = _watch_a_late_beat(monkeypatch, clock, stop)
    _end_watchdog(stop)  # joins the watchdog, which ends by itself on waking
    assert rows == [], rows


def test_a_beat_that_resumes_while_the_supervisor_stall_is_written_still_rows(monkeypatch):
    """Both loops are late when the watchdog wakes, as on 27.11: the serving loop beats again while the supervisor
    stall row is being written (a sink waiting on the disk). The ASGI episode was observed before that write, so
    its row is journaled with the stack taken while the loop still stood."""
    clock = _HandClock(time.monotonic() + 1000.0)

    def resume_during_the_stall_row(row: dict) -> None:
        if row["type"] == "supervisor_loop_stall":
            asgi_beat.ASGI_BEAT.stamp = clock.mono  # the serving loop beats again meanwhile

    stop = threading.Event()
    rows = _watch_a_late_beat(monkeypatch, clock, stop, supervisor_age=100.0, on_row=resume_during_the_stall_row)
    try:
        _until(lambda: len(rows) >= 2)
        time.sleep(0.1)  # more ticks: the stall stays open, the beat is on time now
    finally:
        _end_watchdog(stop)
    assert [row["type"] for _path, row in rows] == ["supervisor_loop_stall", "asgi_loop_late"], rows
    late = rows[1][1]
    assert late["overdue_sec"] == 85.0 and late["resumed"] is False and late["stack"], late
