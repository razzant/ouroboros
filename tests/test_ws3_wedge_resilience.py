"""WS3 — chat-lane wedge resilience (v6.34.0).

A dedicated watchdog thread (outside the supervisor loop) observes TWO silent-wedge
classes instead of silent hours, and the two are reported DIFFERENTLY. A
heartbeat-silent in-process direct-chat turn still alerts the owner, because
/restart is a recovery they can actually perform. A supervisor loop stall is
JOURNAL ONLY (owner decision 4C): the durable `supervisor_loop_stall` row with the
phase facts the loop published, its `supervisor_loop_stall_end` closer and the
`log.error` — no owner chat row and no toast, because a stall the owner cannot act
on is an alarm rather than information. New-message intake is reordered EARLY in
the loop so a blocking step can't starve it. The watchdog cannot kill a hung thread
or free the chat-agent lock (a wedged turn holds it for its whole duration;
out-of-process kill was deferred per owner), so it detects + reports rather than
force-recovering in-process; WS10 ephemeral decision turns keep the chat responsive
meanwhile.
"""

from __future__ import annotations

import threading
import time


def test_supervisor_loop_stalled_detection():
    import server

    now = 1000.0
    assert server._supervisor_loop_stalled(now - 100, now, 90) is True   # past deadline
    assert server._supervisor_loop_stalled(now - 30, now, 90) is False   # healthy tick
    assert server._supervisor_loop_stalled(now - 100, now, 0) is False   # 0 = disabled


def test_supervisor_liveness_deadline_getter(monkeypatch):
    from ouroboros.config import (
        SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC,
        get_supervisor_liveness_deadline_sec,
    )

    monkeypatch.delenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", raising=False)
    assert get_supervisor_liveness_deadline_sec() == SUPERVISOR_LIVENESS_DEADLINE_DEFAULT_SEC
    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "30")
    assert get_supervisor_liveness_deadline_sec() == 30
    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "0")
    assert get_supervisor_liveness_deadline_sec() == 0  # disabled


def test_watchdog_noop_when_disabled(monkeypatch):
    import server

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "0")
    before = threading.active_count()
    server._start_supervisor_liveness_watchdog([time.monotonic()])
    assert threading.active_count() == before  # no watchdog thread spawned


def test_watchdog_journals_a_stall_without_an_owner_message_or_toast(monkeypatch):
    """Owner decision 4C, both directions: the episode is durable — one
    `supervisor_loop_stall` row carrying the facts the loop published with its last
    stamp, and one `supervisor_loop_stall_end` once the loop ticks again — and the
    owner's chat stays silent: no message, no `task_incident`, no toast."""
    import server

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().clear()  # isolate the stall half
    alerts = _collect_alerts(monkeypatch, 5)
    rows: list = []
    monkeypatch.setattr("supervisor.state.append_jsonl",
                        lambda path, row: rows.append(row) or True)
    # The liveness tick is a MONOTONIC stamp (OB-03) — seed it on the same clock.
    liveness = [time.monotonic() - 100, {"phase": "maintenance"}, 0.0, None]
    stop = threading.Event()  # local per-test token; do NOT touch the global restart flag
    try:
        server._start_supervisor_liveness_watchdog(liveness, stop)  # already stale
        _wait_until(lambda: any(r["type"] == "supervisor_loop_stall" for r in rows))

        def recovered() -> bool:
            liveness[0] = time.monotonic()  # the loop ticks again, and keeps ticking
            return any(r["type"] == "supervisor_loop_stall_end" for r in rows)

        _wait_until(recovered)
        episode = list(rows)  # a later iteration re-stalls the frozen stamp
    finally:
        _stop_watchdog(stop)  # join it too: a leaked in-flight iteration outlives the monkeypatch
    assert [row["type"] for row in episode] == [
        "supervisor_loop_stall", "supervisor_loop_stall_end"], episode
    assert episode[0]["phase"] == "maintenance" and episode[0]["stalled_sec"] >= 100
    assert episode[1]["phase"] == "maintenance" and episode[1]["stalled_sec"] >= 100
    assert alerts == [], "a loop stall is journal only — no owner row, no toast"


def test_chat_turn_wedged_detection():
    import server

    now = 1000.0
    assert server._chat_turn_wedged(True, now - 100, now, 90) is True    # busy + silent past deadline
    assert server._chat_turn_wedged(True, now - 30, now, 90) is False    # busy + recent heartbeat
    assert server._chat_turn_wedged(False, now - 100, now, 90) is False  # not busy
    assert server._chat_turn_wedged(True, None, now, 90) is False        # liveness loop not started yet
    assert server._chat_turn_wedged(True, now - 100, now, 0) is False    # 0 = disabled


def test_chat_turn_liveness_reads_all_actors_without_taking_admission_lock(monkeypatch):
    import types
    import supervisor.workers as w
    from supervisor.active_activity import get_direct_activity_registry

    registry = get_direct_activity_registry()
    assert w.chat_turn_liveness() == []
    for tid, stamp in (("t1", 1234.0), ("t2", 2345.0)):
        registry.register(tid, 1, actor=types.SimpleNamespace(
            _busy=True, _current_task_id=tid, _last_activity_ts=stamp))
    assert w._repo_writer_gate_lock.acquire(blocking=False)
    try:
        assert w.chat_turn_liveness() == [("t1", 1234.0), ("t2", 2345.0)]
    finally:
        w._repo_writer_gate_lock.release()


def test_watchdog_alerts_on_chat_turn_wedge(monkeypatch):
    import types

    import server

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    alerts = []

    class _Bridge:
        def send_message(self, chat_id, text, *a, **k):
            alerts.append((chat_id, text, k))
            return (True, "")

    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: _Bridge())
    monkeypatch.setattr("supervisor.state.load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda *a, **k: None)
    # The heartbeat stamp is MONOTONIC (OB-03) — seed it on the same clock.
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register("wedged1", 1, actor=types.SimpleNamespace(
        _busy=True, _current_task_id="wedged1", _last_activity_ts=time.monotonic() - 100))
    stop = threading.Event()  # local per-test token
    try:
        server._start_supervisor_liveness_watchdog([time.monotonic()], stop)
        end = time.time() + 6
        while not any("wedged" in a[1] for a in alerts) and time.time() < end:
            time.sleep(0.1)
    finally:
        _stop_watchdog(stop)  # join it too: a leaked in-flight iteration outlives the monkeypatch
    assert any("wedged" in a[1] for a in alerts)  # the chat-turn wedge was surfaced
    assert any(a[0] == 7 for a in alerts)
    wedge = next(a for a in alerts if "wedged" in a[1])
    assert wedge[2]["is_progress"] is True
    assert wedge[2]["task_id"] == "wedged1"
    assert wedge[2]["progress_meta"] == {
        "task_incident": "chat_turn_wedge",
        "toast_once": "wedged1:chat_turn_wedge",
        "narration": False,
    }


# --- OB-03: both watchdog halves share the one monotonic clock ---------------


class _FakeServerClock:
    """A controllable stand-in for the ``time`` module ``server`` reads.

    Only the three calls the watchdog makes are driven (``sleep``/``time``/
    ``monotonic``); everything else falls through to the real module. The TEST
    HARNESS keeps the real clock — this module's own ``time`` is never patched —
    so a simulated wall-clock jump cannot make the harness's own timeouts lie.
    """

    def __init__(self, *, wall: float, mono: float) -> None:
        self.wall = wall
        self.mono = mono
        self.ticks = 0

    def __getattr__(self, name):  # anything the watchdog does not drive stays real
        return getattr(time, name)

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, _seconds: float) -> None:
        self.ticks += 1
        time.sleep(0.01)  # a REAL yield; the fake clock advances only when a test says so


def _wait_until(predicate, budget: float = 6.0) -> None:
    end = time.time() + budget  # real clock: the harness never rides the fake one
    while not predicate() and time.time() < end:
        time.sleep(0.01)


def _stop_watchdog(stop: threading.Event) -> None:
    """Set the token AND wait for the thread to leave the loop.

    The watchdog re-reads the token only at the TOP of the loop, so an iteration
    already in flight still completes its clock reads and checks. Returning before
    that finishes would let it run against a torn-down monkeypatch — reading the
    real ``time`` module against a fake liveness stamp — and emit a phantom alert
    into whatever the next test has patched.
    """
    stop.set()
    for thread in threading.enumerate():
        if thread.name == "supervisor-liveness-watchdog":
            thread.join(timeout=5)


def _collect_alerts(monkeypatch, chat_id: int) -> list:
    alerts: list = []

    class _Bridge:
        def send_message(self, cid, text, *a, **k):
            alerts.append((cid, text, k))
            return (True, "")

    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: _Bridge())
    monkeypatch.setattr("supervisor.state.load_state", lambda: {"owner_chat_id": chat_id})
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda *a, **k: None)
    return alerts


def test_wall_clock_jump_neither_fabricates_nor_masks_a_supervisor_stall(monkeypatch):
    """OB-03: the stall half is measured on ``time.monotonic()``.

    The liveness tick and its comparison used to both be ``time.time()``, so an
    ordinary wall-clock step — NTP correction, DST/timezone change, manual set, a
    resumed VM — was indistinguishable from an unresponsive supervisor loop. Both
    directions are pinned on the surface a stall still has (4C: the journal, never
    the owner's chat): a forward jump must not INVENT a stall row, and a backward
    jump must not MASK a real one.
    """
    import server

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().clear()  # isolate the stall half
    alerts = _collect_alerts(monkeypatch, 11)
    rows: list = []
    monkeypatch.setattr("supervisor.state.append_jsonl",
                        lambda path, row: rows.append(row) or True)

    boot_mono = 500.0
    wall = 1_700_000_000.0
    clock = _FakeServerClock(wall=wall, mono=boot_mono)
    # The watchdog reads its clock from its owner module (v7 server split).
    from ouroboros import server_liveness
    monkeypatch.setattr(server_liveness, "time", clock)
    stop = threading.Event()  # local per-test token
    try:
        # The loop ticked "just now" on the monotonic clock — it is healthy.
        server._start_supervisor_liveness_watchdog([boot_mono], stop)
        # Now the wall clock steps an hour forward while the loop keeps ticking.
        clock.wall = wall + 3600.0
        _wait_until(lambda: clock.ticks >= 3)
        assert rows == [], "a wall-clock jump must not fabricate a supervisor stall"

        # A genuine stall (100s of MONOTONIC silence) is still caught, and a
        # backward wall step cannot hide it.
        clock.wall = wall - 86_400.0
        clock.mono = boot_mono + 100.0
        _wait_until(lambda: rows)
    finally:
        _stop_watchdog(stop)
    assert [row["type"] for row in rows] == ["supervisor_loop_stall"]
    assert rows[0]["stalled_sec"] == 100.0
    assert alerts == [], "the stall never speaks in the owner's chat"


def test_wall_clock_jump_neither_fabricates_nor_masks_a_chat_turn_wedge(monkeypatch):
    """OB-03, the wedge half — the CONTRACT, not the implementation: a silent chat
    turn is detected, and a wall-clock jump neither fabricates a wedge nor masks one.
    ``agent._last_activity_ts`` is a monotonic stamp (agent.py), compared against the
    same monotonic ``now`` as the stall half — one clock, one class."""
    import types

    import server

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    alerts = _collect_alerts(monkeypatch, 13)

    boot_mono = 500.0
    wall = 1_700_000_000.0
    clock = _FakeServerClock(wall=wall, mono=boot_mono)
    # The watchdog reads its clock from its owner module (v7 server split).
    from ouroboros import server_liveness
    monkeypatch.setattr(server_liveness, "time", clock)
    agent_stub = types.SimpleNamespace(
        _busy=True, _current_task_id="wedged-mono", _last_activity_ts=boot_mono)
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register("wedged-mono", 1, actor=agent_stub)
    stop = threading.Event()  # local per-test token
    try:
        server._start_supervisor_liveness_watchdog([boot_mono], stop)
        # The turn heartbeat is fresh; an hour-forward WALL step must not invent a
        # wedge (pre-fix, a wall-fed comparison fabricated exactly this alert).
        clock.wall = wall + 3600.0
        _wait_until(lambda: clock.ticks >= 3)
        assert alerts == [], "a wall-clock jump must not fabricate a chat-turn wedge"

        # A genuine wedge: 100s of MONOTONIC heartbeat silence while the loop's own
        # tick stays healthy; a backward wall step cannot hide it.
        clock.wall = wall - 86_400.0
        clock.mono = boot_mono + 100.0
        stop_liveness = [clock.mono]  # loop keeps ticking; only the TURN is silent
        _stop_watchdog(stop)
        stop = threading.Event()
        alerts.clear()
        server._start_supervisor_liveness_watchdog(stop_liveness, stop)
        _wait_until(lambda: alerts)
    finally:
        _stop_watchdog(stop)
    assert alerts, "a heartbeat-silent chat turn must be detected under any wall clock"
    assert "wedged" in alerts[0][1]
    assert alerts[0][2]["task_id"] == "wedged-mono"
    assert not any("stalled" in a[1] for a in alerts)  # the healthy tick raised nothing


def test_watchdog_and_heartbeat_writers_share_the_monotonic_clock():
    """Source-level pin of the PRODUCTION writers (the behavioural tests above feed
    hand-built stamps, so a writer regressed to ``time.time()`` would leave them
    green): the supervisor loop's two liveness stamps and the direct-chat turn's two
    heartbeat stamps must all be taken on ``time.monotonic()``. The idiom follows
    test_server_shutdown.py's existing ``inspect.getsource`` pin of _run_supervisor.
    """
    import inspect
    import re

    import server
    from ouroboros.agent import OuroborosAgent

    sup_src = inspect.getsource(server._run_supervisor)
    liveness_writes = re.findall(r"_loop_liveness(?:\[0\])?\s*=\s*\[?time\.(\w+)\(\)", sup_src)
    assert liveness_writes and set(liveness_writes) == {"monotonic"}, liveness_writes

    agent_src = inspect.getsource(OuroborosAgent)
    heartbeat_writes = re.findall(r"_last_activity_ts\s*=\s*time\.(\w+)\(\)", agent_src)
    assert heartbeat_writes and set(heartbeat_writes) == {"monotonic"}, heartbeat_writes
