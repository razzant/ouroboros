"""Honest measurements of a stalled supervisor loop (C1).

The watchdog used to record only the ONSET of a stall and nothing about it: no
phase, no end, no CPU-vs-wall split. A 64-stall night could therefore not tell a
loop thread BURNING its wall gap from one blocked on a lock or starved of the
GIL, nor say which coarse phase it went silent in. These pins cover the
measurement contract in both directions: the facts are published BY THE LOOP
THREAD with its liveness stamp, the watchdog thread only reads them, and nothing
is invented for an event a worker never stamped.
"""

from __future__ import annotations

import inspect
from pathlib import PurePath
import logging
import re
import threading
import time

import pytest


class _Clock:
    """Controllable stand-in for the ``time`` module the watchdog reads.

    Only ``monotonic``/``sleep`` are driven; the harness keeps the real clock, so
    a simulated stall cannot make the test's own timeouts lie.
    """

    def __init__(self, *, mono: float) -> None:
        self.mono = mono
        self.ticks = 0

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, _seconds: float) -> None:
        self.ticks += 1
        time.sleep(0.01)  # a REAL yield; the fake clock moves only when a test says so


def _wait_until(predicate, budget: float = 6.0) -> None:
    end = time.time() + budget  # real clock: the harness never rides the fake one
    while not predicate() and time.time() < end:
        time.sleep(0.01)


def _stop_watchdog(stop: threading.Event) -> None:
    stop.set()
    for thread in threading.enumerate():
        if thread.name == "supervisor-liveness-watchdog":
            thread.join(timeout=5)


@pytest.fixture()
def journal(monkeypatch):
    """Collect the durable supervisor rows without touching a live data root."""
    rows: list = []

    def _append(path, obj):
        rows.append((str(path), dict(obj)))
        return True

    monkeypatch.setattr("supervisor.state.append_jsonl", _append)
    monkeypatch.setattr("supervisor.state.load_state", lambda: {})  # no owner chat: no alert send
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().clear()  # isolate the loop-stall half
    return rows


def _live_liveness(phase: str = "maintenance") -> list:
    """A liveness list exactly as ``server.py::_run_supervisor`` publishes it."""
    from ouroboros import server_liveness

    liveness = [time.monotonic(), {}, time.thread_time(), None]
    server_liveness.observe_worker_event_lag(liveness, {
        "type": "task_heartbeat", "task_id": "t-1", "ts": _iso_ago(4.0),
    })
    liveness[1] = server_liveness.loop_phase_facts(liveness, phase)
    return liveness


def _iso_ago(seconds: float) -> str:
    import datetime as _dt

    return (_dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(seconds=seconds)).isoformat()


def test_stall_row_carries_phase_cpu_and_event_lag(monkeypatch, journal):
    """The onset row names WHERE the loop went silent and what the thread was doing."""
    import server
    from ouroboros import server_liveness

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    liveness = _live_liveness("maintenance")
    clock = _Clock(mono=liveness[0] + 100.0)  # 100s of MONOTONIC silence
    monkeypatch.setattr(server_liveness, "time", clock)
    stop = threading.Event()
    try:
        server._start_supervisor_liveness_watchdog(liveness, stop)
        _wait_until(lambda: journal)
    finally:
        _stop_watchdog(stop)
    stalls = [row for _path, row in journal if row["type"] == "supervisor_loop_stall"]
    assert len(stalls) == 1, journal
    assert PurePath(journal[0][0]).as_posix().endswith("logs/supervisor.jsonl")
    row = stalls[0]
    assert row["stalled_sec"] == pytest.approx(100.0, abs=1.0)
    assert row["phase"] == "maintenance"
    assert isinstance(row["loop_thread_cpu_sec"], float)
    assert row["max_event_lag_sec"] == pytest.approx(4.0, abs=2.0)
    assert "daemon_pin_matched" in row and row["daemon_pin_matched"] in (True, False, None)


def test_stall_end_is_written_once_per_alerted_stall(monkeypatch, journal):
    """Closing row: exactly one per alerted stall, carrying the STALLED phase."""
    import server
    from ouroboros import server_liveness

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    liveness = _live_liveness("maintenance")
    clock = _Clock(mono=liveness[0] + 100.0)
    monkeypatch.setattr(server_liveness, "time", clock)
    stop = threading.Event()
    try:
        server._start_supervisor_liveness_watchdog(liveness, stop)
        _wait_until(lambda: journal)
        # The loop ticks again, in its NEXT phase: the episode closes exactly once.
        liveness[1] = server_liveness.loop_phase_facts(liveness, "assign")
        liveness[0] = clock.mono
        _wait_until(lambda: any(row["type"] == "supervisor_loop_stall_end" for _p, row in journal))
        ticks = clock.ticks
        _wait_until(lambda: clock.ticks > ticks + 2)  # keep polling a healthy loop
    finally:
        _stop_watchdog(stop)
    kinds = [row["type"] for _path, row in journal]
    assert kinds.count("supervisor_loop_stall") == 1, journal
    assert kinds.count("supervisor_loop_stall_end") == 1, journal
    end = next(row for _p, row in journal if row["type"] == "supervisor_loop_stall_end")
    assert end["stalled_sec"] == pytest.approx(100.0, abs=1.0)
    assert end["phase"] == "maintenance"  # where it was stuck, not where it resumed
    # The recovery stamp publishes the loop thread's CPU over the stalled interval itself.
    assert end["loop_thread_cpu_sec"] == liveness[1]["loop_thread_cpu_sec"]
    assert isinstance(end["loop_thread_cpu_sec"], float)


def test_a_healthy_loop_journals_neither_row(monkeypatch, journal):
    """The quiet direction: a ticking loop writes no stall and no end row."""
    import server
    from ouroboros import server_liveness

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    liveness = _live_liveness("events")
    clock = _Clock(mono=liveness[0])
    monkeypatch.setattr(server_liveness, "time", clock)
    stop = threading.Event()
    try:
        server._start_supervisor_liveness_watchdog(liveness, stop)
        _wait_until(lambda: clock.ticks >= 3)
    finally:
        _stop_watchdog(stop)
    assert journal == []


def test_an_event_without_a_worker_stamp_never_produces_a_lag(monkeypatch):
    """Events carry the worker's own ``ts``; an unstamped one is skipped, not invented."""
    from ouroboros import server_liveness

    liveness = [time.monotonic(), {}, time.thread_time(), None]
    for unstamped in ({"type": "task_message_injected"}, {"type": "x", "ts": ""},
                      {"type": "x", "ts": "not-a-timestamp"}, {"type": "x", "ts": None}):
        server_liveness.observe_worker_event_lag(liveness, unstamped)
    assert "max_event_lag_sec" not in server_liveness.loop_phase_facts(liveness, "maintenance")
    # The other direction: a worker-stamped event IS measured.
    server_liveness.observe_worker_event_lag(liveness, {"type": "task_heartbeat", "ts": _iso_ago(7.0)})
    facts = server_liveness.loop_phase_facts(liveness, "maintenance")
    assert facts["max_event_lag_sec"] == pytest.approx(7.0, abs=2.0)


def test_the_drain_maximum_is_scoped_to_one_tick(monkeypatch):
    """``new_tick`` opens a fresh drain maximum, so a stale lag cannot ride forever."""
    from ouroboros import server_liveness

    liveness = [time.monotonic(), {}, time.thread_time(), None]
    server_liveness.observe_worker_event_lag(liveness, {"type": "task_heartbeat", "ts": _iso_ago(5.0)})
    opening = server_liveness.loop_phase_facts(liveness, "events", new_tick=True)
    assert opening["max_event_lag_sec"] == pytest.approx(5.0, abs=2.0)
    assert "max_event_lag_sec" not in server_liveness.loop_phase_facts(liveness, "maintenance")


def test_loop_thread_cpu_separates_a_burning_thread_from_a_blocked_one(monkeypatch):
    """The CPU-vs-wall split: the delta is the LOOP THREAD's own processor time."""
    from ouroboros import server_liveness

    liveness = [time.monotonic(), {}, time.thread_time(), None]
    time.sleep(0.05)  # wall time passes, this thread burns nothing
    idle = server_liveness.loop_phase_facts(liveness, "events")["loop_thread_cpu_sec"]
    burn_until = time.thread_time() + 0.05
    while time.thread_time() < burn_until:
        pass
    busy = server_liveness.loop_phase_facts(liveness, "maintenance")["loop_thread_cpu_sec"]
    assert idle < 0.03, idle
    assert busy >= 0.04, busy


def test_the_loop_publishes_one_monotonic_stamp_per_tick_phase():
    """Source pin of the PRODUCER (the behavioural tests feed hand-built stamps).

    Three coarse phases, one stamp each, every stamp taken on ``time.monotonic()``
    (OB-03: a wall-clock jump must never fabricate or mask a stall), and the drain
    itself observes the worker-event lag it reports.
    """
    import server

    source = inspect.getsource(server._run_supervisor)
    stamps = re.findall(
        r'_loop_liveness\[1\], _loop_liveness\[0\] = loop_phase_facts\(\s*_loop_liveness, "(\w+)"'
        r'[^)]*\), time\.(\w+)\(\)',
        source,
    )
    assert [phase for phase, _clock in stamps] == ["events", "maintenance", "assign"], stamps
    assert {clock for _phase, clock in stamps} == {"monotonic"}, stamps
    # The bounded drain receives the loop's own liveness list and observes the lag itself.
    from ouroboros import server_liveness

    assert re.search(r"drain_worker_events\(\s*get_event_q\(\), _event_ctx, _loop_liveness,", source), source
    assert "observe_worker_event_lag(liveness, evt)" in inspect.getsource(server_liveness.drain_worker_events)


def _run_custody_tick(monkeypatch, *, failing_step=None):
    """Drive ONE 600s custody pass with every step stubbed; ``failing_step`` raises.

    The pass runs on its own daemon thread (INV-B), so the tick is joined before the
    caller reads what it logged."""
    import threading
    from types import SimpleNamespace

    import ouroboros.process_custody as pc
    import ouroboros.server_maintenance as sm

    threads: list = []

    def tracked(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread

    def _step(name):
        def _run(*_a, **_k):
            if name == failing_step:
                raise RuntimeError(f"{name} exploded")
            return []
        return _run

    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])  # 20s cadence stays quiet
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(pc, "reap_orphaned_processes", _step("reap_orphaned_processes"))
    monkeypatch.setattr(sm, "_reconcile_delegated_runs", _step("reconcile_delegated_runs"))
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals", _step("cursor_refresh_settled_terminals"))
    monkeypatch.setattr(
        "ouroboros.claudexor_daemon.get_owned_daemon",
        lambda: type("_D", (), {"clear_start_failure_latch": lambda self, **_k: False})(),
    )
    monkeypatch.setattr(sm, "threading", SimpleNamespace(Thread=tracked))
    sm._periodic_supervisor_maintenance([0.0], [time.time()])
    for thread in threads:
        thread.join(5)
    assert all(not thread.is_alive() for thread in threads)


@pytest.mark.parametrize("failing_step", ["reap_orphaned_processes", "reconcile_delegated_runs",
                                          "cursor_refresh_settled_terminals"])
def test_a_failed_custody_step_is_loud_and_names_itself(monkeypatch, caplog, failing_step):
    """A 600s block that dies silently at DEBUG is the reason the night is unproven."""
    with caplog.at_level(logging.DEBUG):
        _run_custody_tick(monkeypatch, failing_step=failing_step)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and failing_step in r.getMessage()]
    assert warnings, [(r.levelname, r.getMessage()) for r in caplog.records]


def test_a_healthy_custody_pass_stays_quiet(monkeypatch, caplog):
    """The other direction: nothing failed, so nothing is reported."""
    with caplog.at_level(logging.DEBUG):
        _run_custody_tick(monkeypatch)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_failed_init_is_not_ready_on_the_state_api_while_boot_waiters_still_settle(monkeypatch, tmp_path):
    """TZ-1: the failure rail used to SET readiness so the boot finalizer would
    not hang; ``/api/state`` then answered ``supervisor_ready: true`` beside
    ``supervisor_error`` and the chat header (readiness is the typed boolean
    alone) painted Online over a supervisor that never reached its loop.
    Readiness and the init outcome are separate latches: the real endpoint,
    wired to the real failure rail, says not-ready with the error, and the
    finalizer returns at once with a failed verdict."""
    import asyncio
    import json
    import types

    from starlette.requests import Request

    import ouroboros.config as config
    import server
    from ouroboros import usage_accounting as ua
    from ouroboros.gateway.state import api_state
    from supervisor import queue as queue_mod, state as state_mod, workers

    # Wiring pin: the endpoint reads the SAME latch and error the rail writes.
    assert server.app.app.state.supervisor_ready_event is server._supervisor_ready
    assert server.app.app.state.get_supervisor_error() is server._supervisor_error

    ready = threading.Event()
    init_done = threading.Event()
    monkeypatch.setattr(server, "_supervisor_ready", ready)
    monkeypatch.setattr(server, "_supervisor_init_done", init_done)
    monkeypatch.setattr(server, "_supervisor_thread", threading.current_thread())
    monkeypatch.setattr(server, "_supervisor_error", None)
    monkeypatch.setattr(server, "_consciousness", None)
    monkeypatch.setattr(server, "_apply_settings_to_env", lambda _settings: None)
    monkeypatch.setattr(server, "_start_supervisor_liveness_watchdog", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_startup_worker_pids", lambda _root: set())
    monkeypatch.setattr(server, "_run_startup_task_recovery", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "ensure_legacy_imported",
                        lambda _root: (_ for _ in ()).throw(RuntimeError("boot dependency refused")))
    server._run_supervisor({})
    assert init_done.is_set() and not ready.is_set()
    assert server._supervisor_error == "Supervisor init failed: boot dependency refused"
    assert server._wait_for_supervisor_update_finalize() is False, "a known failed outcome never blocks the boot"

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    repo = tmp_path / "repo"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/ouroboros\n", encoding="utf-8")
    (repo / ".git" / "refs" / "heads" / "ouroboros").write_text("1234567890abcdef1234567890abcdef12345678\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    ua.ensure_legacy_imported(root)
    monkeypatch.setattr(config, "REPO_DIR", repo)
    monkeypatch.setattr(state_mod, "TOTAL_BUDGET_LIMIT", 0.0)
    monkeypatch.setattr(state_mod, "load_state", lambda: {"current_branch": None, "current_sha": None})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(queue_mod, "get_evolution_status_snapshot", lambda **_kwargs: {})
    request = Request({
        "type": "http", "method": "GET", "path": "/api/state", "headers": [],
        "query_string": b"", "scheme": "http", "server": ("test", 80), "client": ("test", 1),
        "app": types.SimpleNamespace(state=types.SimpleNamespace(
            drive_root=root, app_start=0.0,
            supervisor_ready_event=server._supervisor_ready,
            get_supervisor_error=lambda: server._supervisor_error,
        )),
    })
    payload = json.loads(asyncio.run(api_state(request)).body)
    assert payload["supervisor_ready"] is False
    assert payload["supervisor_error"] == "Supervisor init failed: boot dependency refused"
