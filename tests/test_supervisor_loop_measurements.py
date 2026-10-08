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
from pathlib import Path, PurePath
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
    # One recovery stamp: the CPU charged to the stall is that stamp's delta, over
    # the same wall interval the stall lasted.
    assert end["loop_thread_cpu_sec"] == pytest.approx(liveness[1]["loop_thread_cpu_sec"], abs=1e-3)
    assert isinstance(end["loop_thread_cpu_sec"], float)
    assert end["cpu_interval_sec"] == end["stalled_sec"]


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
    itself observes the worker-event lag it reports. Startup names the usage
    store's one-time import as its own sub-phase, then returns to "startup".
    """
    import server

    source = inspect.getsource(server._run_supervisor)
    stamps = re.findall(
        r'_loop_liveness\[1\], _loop_liveness\[0\] = loop_phase_facts\(\s*_loop_liveness, "([\w:]+)"'
        r'[^)]*\), time\.(\w+)\(\)',
        source,
    )
    assert [phase for phase, _clock in stamps] == ["startup", "events", "maintenance", "assign"], stamps
    assert {clock for _phase, clock in stamps} == {"monotonic"}, stamps
    # The usage store's one-time import runs as its own sub-phase, stamped the same way.
    from ouroboros import server_liveness

    assert 'run_startup_phase(_loop_liveness, "startup:usage_store",' in source
    helper = inspect.getsource(server_liveness.run_startup_phase)
    assert re.findall(r'loop_phase_facts\(liveness, (\w+|"\w+")\), time\.(\w+)\(\)', helper) == [
        ("phase", "monotonic"), ('"startup"', "monotonic")], helper
    # The bounded drain receives the loop's own liveness list and observes the lag itself.

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
    monkeypatch.setattr("ouroboros.terminal_projection.reconcile_terminal_projections", _step("terminal_projection"))
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
    from ouroboros import usage_store
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
    migrate = usage_store.migrate_from_journal
    monkeypatch.setattr("ouroboros.usage_store.migrate_from_journal",
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
    migrate(root)  # the store the refused boot never created, so the state API reads money
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


def test_stall_end_charges_the_whole_stall_not_the_last_phase(monkeypatch, journal):
    """Several phases can pass between the loop's recovery and the watchdog's next
    look. The end row charges the thread's CPU from the stamp it went silent on to
    its LATEST stamp (the cumulative totals), never only the last phase's delta."""
    import server
    from ouroboros import server_liveness

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    liveness = _live_liveness("maintenance")
    onset_total = liveness[1]["loop_thread_cpu_total_sec"]
    clock = _Clock(mono=liveness[0] + 100.0)
    monkeypatch.setattr(server_liveness, "time", clock)
    stop = threading.Event()
    try:
        server._start_supervisor_liveness_watchdog(liveness, stop)
        _wait_until(lambda: journal)
        # The loop recovers and publishes THREE stamps, burning CPU before each,
        # before it moves the trigger stamp the watchdog reads.
        for phase in ("assign", "events", "maintenance"):
            burn_until = time.thread_time() + 0.02
            while time.thread_time() < burn_until:
                pass
            liveness[1] = server_liveness.loop_phase_facts(liveness, phase)
        liveness[0] = clock.mono
        _wait_until(lambda: any(row["type"] == "supervisor_loop_stall_end" for _p, row in journal))
    finally:
        _stop_watchdog(stop)
    end = next(row for _p, row in journal if row["type"] == "supervisor_loop_stall_end")
    whole = liveness[1]["loop_thread_cpu_total_sec"] - onset_total
    assert end["loop_thread_cpu_sec"] == pytest.approx(whole, abs=2e-3)
    assert whole >= 0.05 and end["loop_thread_cpu_sec"] > liveness[1]["loop_thread_cpu_sec"]
    assert end["cpu_interval_sec"] == end["stalled_sec"] == pytest.approx(100.0, abs=1.0)


def test_stall_row_carries_the_loop_threads_stack(monkeypatch, journal):
    """The onset row names the LINE the loop thread is on, bounded to a few frames."""
    import server
    from ouroboros import server_liveness

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    # The watchdog must not fetch source through linecache: disk may be stalled.
    import linecache
    monkeypatch.setattr(linecache, "getline", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("disk read")))
    liveness = _live_liveness("maintenance")
    clock = _Clock(mono=liveness[0] + 100.0)
    monkeypatch.setattr(server_liveness, "time", clock)
    release = threading.Event()

    def _pretend_stalled_maintenance_step():
        release.wait(10)

    stalled = threading.Thread(target=_pretend_stalled_maintenance_step, name="fake-supervisor-loop")
    stalled.start()
    stop = threading.Event()
    try:
        server._start_supervisor_liveness_watchdog(liveness, stop, loop_thread_ident=stalled.ident)
        _wait_until(lambda: journal)
    finally:
        release.set()
        stalled.join(5)
        _stop_watchdog(stop)
    row = next(row for _p, row in journal if row["type"] == "supervisor_loop_stall")
    stack = row["stack"]
    assert isinstance(stack, list) and 1 <= len(stack) <= server_liveness._STALL_STACK_FRAMES, stack
    assert any("_pretend_stalled_maintenance_step" in frame for frame in stack), stack
    assert all(isinstance(frame, str) and " in " in frame for frame in stack), stack
    assert row["phase"] == "maintenance"  # the coarse phase still rides beside the exact line


def test_the_watched_thread_defaults_to_the_one_that_started_the_watchdog(monkeypatch, journal):
    """The loop thread starts its own watchdog (startup phase included), so the
    default identity is the CALLER: this test thread, caught in its own wait."""
    import server
    from ouroboros import server_liveness

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    liveness = _live_liveness("startup")
    clock = _Clock(mono=liveness[0] + 100.0)
    monkeypatch.setattr(server_liveness, "time", clock)
    stop = threading.Event()
    try:
        server._start_supervisor_liveness_watchdog(liveness, stop)
        _wait_until(lambda: journal)
    finally:
        _stop_watchdog(stop)
    row = next(row for _p, row in journal if row["type"] == "supervisor_loop_stall")
    assert row["phase"] == "startup"
    assert any("_wait_until" in frame for frame in row["stack"]), row["stack"]


def test_a_gone_thread_yields_no_stack_rather_than_an_invented_one():
    from ouroboros import server_liveness

    assert server_liveness._loop_thread_stack(None) == []
    assert server_liveness._loop_thread_stack(-1) == []


def test_a_stamp_names_the_interval_its_cpu_covers():
    """``loop_thread_cpu_sec`` is honest only beside ``cpu_interval_sec``, and the
    cumulative total lets the watchdog charge a whole stall it did not watch stamp by stamp."""
    from ouroboros import server_liveness

    liveness = [time.monotonic(), {}, time.thread_time(), None]
    time.sleep(0.05)
    first = server_liveness.loop_phase_facts(liveness, "events")
    assert 0.045 <= first["cpu_interval_sec"] < 5.0, first
    assert first["loop_thread_cpu_total_sec"] == pytest.approx(time.thread_time(), abs=0.05)
    liveness[0] = time.monotonic()
    second = server_liveness.loop_phase_facts(liveness, "maintenance")
    assert second["cpu_interval_sec"] < 0.05, second
    assert second["loop_thread_cpu_total_sec"] >= first["loop_thread_cpu_total_sec"]
    assert server_liveness._stall_cpu_over(first, second) == pytest.approx(
        second["loop_thread_cpu_total_sec"] - first["loop_thread_cpu_total_sec"], abs=1e-3)
    # A foreign stamp without a total falls back to the latest delta; nothing is invented.
    assert server_liveness._stall_cpu_over({}, second) == second["loop_thread_cpu_sec"]
    assert server_liveness._stall_cpu_over({}, {}) is None


def test_the_watchdog_watches_startup_and_every_generation_exit_stops_it():
    """Source pin: the watchdog starts BEFORE the init block, so a hung recovery,
    worktree prune or worker spawn is a journaled "startup" stall with its stack
    instead of a silent wedge behind ``_supervisor_ready``; and BOTH generation
    exits — the init-failure return and the loop exit — set the per-generation
    stop token, so no watchdog outlives the liveness list it reads."""
    import server

    source = inspect.getsource(server._run_supervisor)
    start = source.index("_start_supervisor_liveness_watchdog(_loop_liveness, _watchdog_stop)")
    assert start < source.index("migrate_from_journal("), "the watchdog must start before init"
    assert source.index("\n    try:\n", source.index("prior_worker_pids:")) < start, "watchdog setup must use the init failure rail"
    assert source.index('loop_phase_facts(_loop_liveness, "startup", new_tick=True)') < start
    assert source.count("_watchdog_stop.set()") == 2, "init-failure exit and loop exit"
    failure = source.index('_supervisor_error = f"Supervisor init failed: {exc}"')
    assert source.index("_watchdog_stop.set()", failure) < source.index("\n        return\n", failure)


def test_watchdog_start_failure_publishes_init_failure_without_running_recovery_on_live_data(monkeypatch):
    """The earlier watchdog start must not bypass the supervisor init-outcome rail."""
    import server

    ready = threading.Event()
    init_done = threading.Event()
    recovery = []
    monkeypatch.setattr(server, "_supervisor_ready", ready)
    monkeypatch.setattr(server, "_supervisor_init_done", init_done)
    monkeypatch.setattr(server, "_supervisor_thread", threading.current_thread())
    monkeypatch.setattr(server, "_supervisor_error", None)
    monkeypatch.setattr(server, "_consciousness", None)
    monkeypatch.setattr(server, "_apply_settings_to_env", lambda _settings: None)
    monkeypatch.setattr(server, "_startup_worker_pids", lambda _root: set())
    monkeypatch.setattr(server, "_run_startup_task_recovery", lambda *_a, **_k: recovery.append(True))
    monkeypatch.setattr(server, "_start_supervisor_liveness_watchdog", lambda *_a: (_ for _ in ()).throw(RuntimeError("watchdog start refused")))
    server._run_supervisor({})
    assert init_done.is_set(), "the outcome is published for boot waiters"
    assert not ready.is_set(), "a failed init is an outcome, never readiness"
    assert "watchdog start refused" in server._supervisor_error
    assert server._supervisor_thread is None
    assert recovery == [True]


def test_a_cut_stack_is_flagged_and_a_startup_stall_does_not_claim_a_chat_that_answers(monkeypatch, journal, caplog):
    """The onset row says when its bounded stack is not the whole one (``loop_stack_truncated``),
    and a stall in the startup phase is worded as an unfinished start: no native chat answers
    for a generation that has not initialized."""
    import logging

    import server
    from ouroboros import server_liveness

    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    liveness = _live_liveness("startup")
    clock = _Clock(mono=liveness[0] + 100.0)
    monkeypatch.setattr(server_liveness, "time", clock)
    release = threading.Event()

    def _deep(depth):
        if depth:
            return _deep(depth - 1)
        release.wait(10)

    stalled = threading.Thread(target=lambda: _deep(server_liveness._STALL_STACK_FRAMES + 5), name="fake-loop")
    stalled.start()
    stop = threading.Event()
    try:
        with caplog.at_level(logging.ERROR):
            server._start_supervisor_liveness_watchdog(liveness, stop, loop_thread_ident=stalled.ident)
            _wait_until(lambda: journal)
    finally:
        release.set()
        stalled.join(5)
        _stop_watchdog(stop)
    row = next(row for _p, row in journal if row["type"] == "supervisor_loop_stall")
    assert row["loop_stack_truncated"] is True and len(row["stack"]) == server_liveness._STALL_STACK_FRAMES
    assert all(" in " in frame and "locals" not in frame for frame in row["stack"])
    message = next(r.getMessage() for r in caplog.records if "STALLED" in r.getMessage())
    assert "startup has not finished" in message and "native chat still answers" not in message

    shallow: dict = {}
    assert server_liveness._loop_thread_stack(threading.get_ident(), limit=500, facts=shallow)
    assert "loop_stack_truncated" not in shallow  # a whole stack carries no flag


def test_stall_end_carries_samples_top_frames_and_the_last_stack(monkeypatch, journal):
    """While a stall is open the watchdog samples the thread's stack once per interval; the
    closing row says how many samples it took, which repository frames they fold into (at
    most five) and the last stack - where the thread SPENT the stall, not only where it began."""
    import server
    from ouroboros import config, server_liveness

    # Shared isolation uses an empty disposable repo. This read-only stack test
    # needs its actual source root to exercise repository-frame attribution.
    monkeypatch.setattr(config, "REPO_DIR", Path(__file__).resolve().parents[1])
    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "1")
    liveness = _live_liveness("maintenance")
    clock = _Clock(mono=liveness[0] + 100.0)
    monkeypatch.setattr(server_liveness, "time", clock)
    release = threading.Event()

    def _pretend_stalled_maintenance_step():
        release.wait(10)

    stalled = threading.Thread(target=_pretend_stalled_maintenance_step, name="fake-supervisor-loop")
    stalled.start()
    stop = threading.Event()
    try:
        server._start_supervisor_liveness_watchdog(liveness, stop, loop_thread_ident=stalled.ident)
        _wait_until(lambda: journal)
        ticks = clock.ticks
        _wait_until(lambda: clock.ticks > ticks + 3)  # several intervals inside the open stall
        liveness[1] = server_liveness.loop_phase_facts(liveness, "assign")
        liveness[0] = clock.mono
        _wait_until(lambda: any(row["type"] == "supervisor_loop_stall_end" for _p, row in journal))
    finally:
        release.set()
        stalled.join(5)
        _stop_watchdog(stop)
    end = next(row for _p, row in journal if row["type"] == "supervisor_loop_stall_end")
    assert end["samples"] >= 3 and 1 <= len(end["top_frames"]) <= 5
    assert sum(item["samples"] for item in end["top_frames"]) == end["samples"]
    hot = end["top_frames"][0]["frame"]
    assert hot == "tests/test_supervisor_loop_measurements.py:_pretend_stalled_maintenance_step", hot
    assert any("_pretend_stalled_maintenance_step" in frame for frame in end["last_stack"])
    import sys
    runtime_only = [f"{sys.prefix}/lib/python3/threading.py:1 in wait".replace("\\", "/")]
    assert server_liveness._innermost_repo_frame(runtime_only) == ""  # a stall entirely inside the runtime names nothing
    assert server_liveness._innermost_repo_frame(["/srv/app/x.py:3 in f", *runtime_only]) == "/srv/app/x.py:f"
    from types import SimpleNamespace
    # A relocatable interpreter reports its prefix through ``bin/..`` yet loads the normalized path.
    monkeypatch.setattr(server_liveness, "sys", SimpleNamespace(prefix="/opt/py/bin/..", base_prefix="/opt/py/bin/.."))
    for row in ("/opt/py/lib/python3.10/threading.py:324 in wait", "/opt/py/bin/../lib/python3.10/threading.py:1 in f"):
        assert server_liveness._innermost_repo_frame([row]) == "", row


def test_stack_rows_parse_windows_drives_and_colons_inside_paths():
    """A row is ``path:line in func`` with the path before the LAST ``:<line> in ``: a Windows
    drive letter (``C:``) is part of an absolute path, never a relative path named ``C``."""
    from ouroboros import server_liveness

    windows = ["C:/Users/owner/app/ouroboros/tool.py:7 in helper", "ouroboros/loop.py:12 in run_step"]
    assert server_liveness._innermost_repo_frame(windows[::-1]) == "ouroboros/loop.py:run_step"
    assert server_liveness._innermost_repo_frame(windows[:1]) == "C:/Users/owner/app/ouroboros/tool.py:helper"
    assert server_liveness._innermost_repo_frame(["C:\\Tools\\x.py:3 in f"]) == "C:\\Tools\\x.py:f"
    assert server_liveness._innermost_repo_frame(["pkg/a:b.py:9 in <lambda>"]) == "pkg/a:b.py:<lambda>"
    assert server_liveness._innermost_repo_frame(["not a stack row"]) == ""


def test_a_recurring_step_failure_backs_off_to_powers_of_two_and_reports_recovery(monkeypatch, caplog):
    """Failures 1, 2, 4, 8 of one periodic step are WARNING with the traceback; the ones between
    are DEBUG; the first success afterwards is one INFO naming the streak."""
    import logging

    import ouroboros.server_maintenance as sm

    monkeypatch.setattr(sm, "_STEP_FAILURES", {})
    with caplog.at_level(logging.DEBUG, logger=sm.log.name):
        for _ in range(5):
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                sm._step_failed("demo_step")
        sm._step_recovered("demo_step")
        sm._step_recovered("demo_step")  # no streak: nothing said
    rows = [(r.levelname, r.getMessage()) for r in caplog.records if "demo_step" in r.getMessage()]
    assert [level for level, _ in rows] == ["WARNING", "WARNING", "DEBUG", "WARNING", "DEBUG", "INFO"], rows
    assert rows[0][1].endswith("(failure 1 in a row)") and rows[-1][1] == "demo_step recovered after 5 failure(s)"
    assert all(r.exc_info for r in caplog.records if "failure" in r.getMessage() and "recovered" not in r.getMessage())


def test_a_due_cadence_finding_its_latch_held_journals_the_duty_stall_with_its_stack(monkeypatch, journal):
    """An off-loop pass that outlives its own cadence is a host duty stall: the next due tick
    journals it once with the pass thread's stack (its threshold is the pass's cadence, never
    the loop deadline), and the pass's end journals the closing row."""
    import ouroboros.server_maintenance as sm

    monkeypatch.setattr(sm, "_DUTIES", {})
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    release = threading.Event()

    def _slow_custody_pass(stop_event, latch):
        try:
            release.wait(10)
        finally:
            latch.release()

    monkeypatch.setattr(sm, "_run_periodic_custody_sweep", _slow_custody_pass)
    last_custody, last_reconcile = [0.0], [time.time()]
    try:
        sm._periodic_supervisor_maintenance(last_custody, last_reconcile)
        assert sm._CUSTODY_SWEEP_LOCK.locked() and not journal
        last_custody[0] = 0.0  # the next cadence is due while the pass still runs...
        sm._periodic_supervisor_maintenance(last_custody, last_reconcile)
        assert not journal, "...but the pass has not run a cadence yet: a due tick alone is no stall"
        sm._DUTIES[id(sm._CUSTODY_SWEEP_LOCK)]["since"] -= 601  # now it has outlived its cadence
        sm._periodic_supervisor_maintenance(last_custody, last_reconcile)
        sm._periodic_supervisor_maintenance(last_custody, last_reconcile)  # journaled once, not per tick
    finally:
        release.set()
        _wait_until(lambda: not sm._CUSTODY_SWEEP_LOCK.locked())
        _wait_until(lambda: any(row["type"] == "host_duty_stall_end" for _p, row in journal))
    kinds = [row["type"] for _p, row in journal]
    assert kinds == ["host_duty_stall", "host_duty_stall_end"], kinds
    stall = journal[0][1]
    assert stall["duty"] == "custody-maintenance" and stall["cadence_sec"] == 600
    assert any("_slow_custody_pass" in frame for frame in stall["stack"]), stall
    assert journal[1][1]["duty"] == "custody-maintenance" and journal[1][1]["running_sec"] >= 0


def test_a_reconcile_pass_just_started_after_an_old_end_stamp_is_not_a_stall(monkeypatch, journal):
    """The reconcile marker is stamped when a pass ENDS, so it is still old while the next pass
    starts: every following tick is "due" at once. The duty threshold is the pass's own running
    time, so a pass that has run a moment is never journaled, and one past its cadence is."""
    import ouroboros.server_maintenance as sm

    monkeypatch.setattr(sm, "_DUTIES", {})
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    release = threading.Event()

    def _slow_reconcile_pass(marker, stop_event=None, latch=None, on_orphans_healed=None):
        try:
            release.wait(10)
        finally:
            marker[0] = time.time()
            latch.release()

    monkeypatch.setattr(sm, "_run_periodic_reconcile_sweep", _slow_reconcile_pass)
    last_custody, last_reconcile = [time.time()], [time.time() - 3600]  # the previous pass ended long ago
    try:
        sm._periodic_supervisor_maintenance(last_custody, last_reconcile)  # starts the pass
        assert sm._RECONCILE_SWEEP_LOCK.locked()
        for _ in range(3):
            sm._periodic_supervisor_maintenance(last_custody, last_reconcile)  # due every tick, latch held
        assert not journal, journal
        sm._DUTIES[id(sm._RECONCILE_SWEEP_LOCK)]["since"] -= 301
        sm._periodic_supervisor_maintenance(last_custody, last_reconcile)
        assert [row["type"] for _p, row in journal] == ["host_duty_stall"]
        assert journal[0][1]["duty"] == "reconcile-maintenance" and journal[0][1]["running_sec"] >= 300
    finally:
        release.set()
        _wait_until(lambda: not sm._RECONCILE_SWEEP_LOCK.locked())
        _wait_until(lambda: any(row["type"] == "host_duty_stall_end" for _p, row in journal))


def test_a_finished_pass_never_drops_the_duty_its_successor_registered(monkeypatch, journal):
    """A pass releases its latch inside its target and its wrapper closes the duty afterwards;
    a successor started in between registers its own duty, which the old wrapper must keep, so
    the successor's own overrun is still journaled."""
    import ouroboros.server_maintenance as sm

    monkeypatch.setattr(sm, "_DUTIES", {})
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    released, first_may_return, second_release = threading.Event(), threading.Event(), threading.Event()
    passes = []

    def _custody_pass(stop_event, latch):
        passes.append(threading.current_thread())
        if len(passes) == 1:
            latch.release()  # the first pass frees its latch ...
            released.set()
            first_may_return.wait(10)  # ... and its wrapper closes the duty only later
            return
        try:
            second_release.wait(10)
        finally:
            latch.release()

    monkeypatch.setattr(sm, "_run_periodic_custody_sweep", _custody_pass)
    last_custody, last_reconcile = [0.0], [time.time()]
    try:
        sm._periodic_supervisor_maintenance(last_custody, last_reconcile)
        _wait_until(released.is_set)
        last_custody[0] = 0.0
        sm._periodic_supervisor_maintenance(last_custody, last_reconcile)  # the successor starts
        successor = sm._DUTIES[id(sm._CUSTODY_SWEEP_LOCK)]
        first_may_return.set()
        _wait_until(lambda: not passes[0].is_alive())
        assert sm._DUTIES.get(id(sm._CUSTODY_SWEEP_LOCK)) is successor, "the old wrapper dropped its successor"
        successor["since"] -= 601
        last_custody[0] = 0.0
        sm._periodic_supervisor_maintenance(last_custody, last_reconcile)
        assert [row["type"] for _p, row in journal] == ["host_duty_stall"]
    finally:
        first_may_return.set()
        second_release.set()
        _wait_until(lambda: not sm._CUSTODY_SWEEP_LOCK.locked())
        _wait_until(lambda: any(row["type"] == "host_duty_stall_end" for _p, row in journal))
    assert not sm._DUTIES, "the successor closed its own duty"
