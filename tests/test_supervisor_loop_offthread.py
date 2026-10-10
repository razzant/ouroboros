"""INV-B: the thread that answers workers runs only queue-bounded work.

The ~600 s custody block — skill-payload hashing, the orphaned-process reaper,
delegated-run reconciliation (gateway handshake, custody replays, registration
retirement) and the settled-terminal cursor — was measured at 11.6-17.6 s on a
healthy install and far longer on a sick one, all of it INLINE on the supervisor
loop thread, ahead of assignment and behind every worker ack.

It now runs on a daemon thread under a non-blocking module lock (busy => skip,
never queue), reads its CANDIDATES before it reads LIVENESS through one shared
live-owner source, and stops mutating the moment its loop generation ends —
reaching the daemon attach-only while a stop, restart or panic is in flight.

The ~300 s reconcile block (zombie heal over every stored task result, artifact
materialization of a healed row, the child-ref promotion walk over every child
drive) was the residual the invariant tolerated INLINE on the tick, and the
child-ref walk also sat inside the 20 s cancel sweep, holding that latch for its
whole length. Both ride the same off-loop shape now: own latch, own daemon
thread, marker stamped when the pass ENDS (issue #1230), generation token
re-read before every step. Every guard below is pinned in both directions.
"""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import pytest

@pytest.fixture(autouse=True)
def _fresh_custody_sweep_latch(monkeypatch):
    """The custody and reconcile latches are process-global and a pass may outlive the test
    that started it (the first tick of any real loop starts one): every test here gets its
    own latches, so a busy one left behind by another test can neither skip this sweep nor
    be released by it."""
    import threading

    from ouroboros import server_maintenance

    monkeypatch.setattr(server_maintenance, "_CUSTODY_SWEEP_LOCK", threading.Lock())
    monkeypatch.setattr(server_maintenance, "_RECONCILE_SWEEP_LOCK", threading.Lock())


def _track_threads(monkeypatch) -> list:
    """Capture the threads the tick starts so a test can join them."""
    from ouroboros import server_maintenance as sm

    threads: list = []

    def tracked(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(sm, "threading", SimpleNamespace(Thread=tracked))
    return threads


@pytest.fixture
def quiet_tick(tmp_path, monkeypatch):
    """The 600 s cadence alone: no 20 s sweep, no daemon latch, no live owners."""
    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros import post_task_checkpoint, server_maintenance as sm
    from supervisor import active_activity, queue, workers

    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(post_task_checkpoint, "POST_TASK_SYNTHESIS_INFLIGHT", {})
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY",
                        active_activity.DirectActivityRegistry())
    monkeypatch.setattr(daemon_mod, "get_owned_daemon", lambda: SimpleNamespace(
        clear_start_failure_latch=lambda *, cleared_by: False))
    return sm


def test_a_slow_custody_block_never_holds_the_loop_tick(quiet_tick, monkeypatch):
    """The tick that STARTS the block returns at once, so the drain, the fence
    acks and assignment keep running while the block is still inside its reap;
    a second tick while the first is busy SKIPS — no second pass, no queue."""
    from ouroboros import process_custody as pc

    sm = quiet_tick
    threads = _track_threads(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def slow_reap(root, **kwargs):
        entered.set()
        assert release.wait(5), "the test must release the custody block"
        return []

    monkeypatch.setattr(pc, "reap_orphaned_processes", slow_reap)
    monkeypatch.setattr(sm, "_reconcile_delegated_runs", lambda live, **kwargs: None)
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals", lambda *a, **kwargs: None)
    try:
        started = time.monotonic()
        sm._periodic_supervisor_maintenance([0.0], [time.time()])
        tick = time.monotonic() - started
        assert entered.wait(5), "the custody block really ran"
        assert tick < 2.0, f"the loop tick waited {tick:.1f}s for the custody block"
        assert not release.is_set(), "the block is still in flight"
        sm._periodic_supervisor_maintenance([0.0], [time.time()])
        assert len(threads) == 1, [thread.name for thread in threads]
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert not sm._CUSTODY_SWEEP_LOCK.locked(), "the block released its latch in finally"


def test_the_reaper_reads_its_candidates_before_it_reads_liveness(tmp_path, monkeypatch):
    """A task assigned WHILE the reaper reads its ledger is not reaped.

    Off the loop thread nothing serializes assignment against the sweep any more,
    so a snapshot taken BEFORE the ledger read would reap a process whose owner was
    admitted in that window. Candidates first, liveness second: a candidate exists
    => its owner was registered earlier, so absence from the LATER snapshot is real.
    """
    from ouroboros import platform_layer, process_custody as pc

    live: set[str] = set()
    entry = {"pid": 424242, "pgid": 0, "scope": "task", "owner_task": "late-task",
             "purpose": "task:late-task", "session_id": pc._SESSION_ID}
    killed: list = []

    def reading_the_ledger(root, strict=False):
        live.add("late-task")  # admitted under _queue_lock while the ledger is read
        return True, [dict(entry)], []

    monkeypatch.setattr(pc, "_fingerprint_matches", lambda row: True)
    monkeypatch.setattr(pc, "_service_group_survives_leader", lambda row: False)
    monkeypatch.setattr(pc, "_rewrite_ledger", lambda *a, **kwargs: None)
    monkeypatch.setattr(platform_layer, "kill_pid_tree",
                        lambda pid, **kwargs: killed.append(pid))
    monkeypatch.setattr(pc, "_read_ledger_records", reading_the_ledger)

    assert pc.reap_orphaned_processes(tmp_path, running_task_ids=lambda: set(live)) == []
    assert killed == [], "the owner was admitted before the decision was taken"

    live.clear()
    monkeypatch.setattr(pc, "_read_ledger_records",
                        lambda root, strict=False: (True, [dict(entry)], []))
    assert pc.reap_orphaned_processes(tmp_path, running_task_ids=lambda: set(live)) == [424242]
    assert killed == [424242], "an owner that really is gone is still reaped"


def test_the_delegated_reconciler_reads_its_candidates_before_it_reads_liveness(
    tmp_path, monkeypatch,
):
    """The same rule on the delegated surface: `open_runs` replays the custody log
    for seconds, and a run whose owner was admitted during that replay is not an
    orphan. A set still means the same thing; None still means touch nothing."""
    from ouroboros import delegate_custody as dc
    from ouroboros import delegate_custody_reconcile as dcr

    live: set[str] = set()
    run = SimpleNamespace(task_id="late-task", run_id="run-1", settled=False)
    seen: list = []

    def replaying_the_log(root, state=None):
        live.add("late-task")  # admitted while the custody log is replayed
        return [run]

    monkeypatch.setattr(dc, "pending_invocations", lambda root, rows=None: [])
    monkeypatch.setattr(dcr, "_reconcile_each", lambda root, runs, factory, **kwargs:
                        seen.append([c.run_id for c in runs]) or [])

    monkeypatch.setattr(dc, "open_runs", replaying_the_log)
    dc.reconcile_orphaned_runs(tmp_path, running_task_ids=lambda: set(live))
    assert seen == [[]], "a run whose owner was admitted during the replay is spared"

    live.clear()
    monkeypatch.setattr(dc, "open_runs", lambda root, state=None: [run])
    dc.reconcile_orphaned_runs(tmp_path, running_task_ids=lambda: set(live))
    assert seen[-1] == ["run-1"], "a genuinely orphaned run is still reconciled"

    assert dc.reconcile_orphaned_runs(tmp_path, running_task_ids=None) == []
    assert len(seen) == 2, "unknown liveness still touches nothing"


def test_both_custody_surfaces_and_the_cursor_refresh_share_one_live_source(
    quiet_tick, monkeypatch,
):
    """One live source, handed to all three consumers — two copies of "is the
    owner still running" is exactly how one surface reaps while its twin does not."""
    from ouroboros import process_custody as pc

    sm = quiet_tick
    threads = _track_threads(monkeypatch)
    seen: dict = {}
    monkeypatch.setattr(pc, "reap_orphaned_processes",
                        lambda root, **kw: seen.__setitem__("processes", kw.get("running_task_ids")) or [])
    monkeypatch.setattr(sm, "_reconcile_delegated_runs",
                        lambda live, **kwargs: seen.__setitem__("delegated", live))
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals",
                        lambda live=None: seen.__setitem__("cursor", live))
    try:
        sm._periodic_supervisor_maintenance([0.0], [time.time()])
    finally:
        for thread in threads:
            thread.join(5)
    assert seen["processes"] is seen["delegated"] is seen["cursor"] is sm._live_task_ids


def test_the_one_live_source_names_every_owner_without_touching_disk(quiet_tick, tmp_path,
                                                                     monkeypatch):
    """RUNNING, busy worker slots, the direct-activity registry and in-flight
    post-task synthesis — the owners `queue.task_has_live_ownership` names, read
    from MEMORY only: the hot path may not pay a durable result load."""
    from ouroboros import post_task_checkpoint, task_results
    from supervisor import active_activity, queue, workers

    sm = quiet_tick
    monkeypatch.setattr(task_results, "load_task_result", lambda *a, **kw: pytest.fail(
        "the live-owner snapshot must not read a durable result"))
    queue.RUNNING["queue-live"] = {"task": {"id": "queue-live"}}
    workers.WORKERS[3] = SimpleNamespace(busy_task_id="worker-live")
    active_activity.get_direct_activity_registry().register("native-live", 1)
    post_task_checkpoint.POST_TASK_SYNTHESIS_INFLIGHT[(str(tmp_path.resolve()), "post-live")] = None

    assert sm._live_task_ids() == {"queue-live", "worker-live", "native-live", "post-live"}


def test_a_closed_generation_stops_the_sweep_before_its_next_mutation(quiet_tick, monkeypatch):
    """The block outlives the loop that started it, so it re-reads the
    per-generation token before EVERY mutation and stops; an OPEN generation runs
    the whole pass. The latch is released either way."""
    from ouroboros import process_custody as pc

    sm = quiet_tick
    done: list = []
    stop = threading.Event()
    monkeypatch.setattr(pc, "reap_orphaned_processes",
                        lambda root, **kw: done.append("reap") or stop.set() or [])
    monkeypatch.setattr(sm, "_reconcile_delegated_runs", lambda live, **kw: done.append("reconcile"))
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals", lambda live=None: done.append("cursor"))

    assert sm._CUSTODY_SWEEP_LOCK.acquire(blocking=False)
    sm._run_periodic_custody_sweep(stop)
    assert done == ["reap"], "the generation ended mid-pass: nothing further mutates"
    assert not sm._CUSTODY_SWEEP_LOCK.locked()

    done.clear()
    monkeypatch.setattr(pc, "reap_orphaned_processes", lambda root, **kw: done.append("reap") or [])
    assert sm._CUSTODY_SWEEP_LOCK.acquire(blocking=False)
    sm._run_periodic_custody_sweep(threading.Event())
    assert done == ["reap", "reconcile", "cursor"], "an open generation runs every step"

    done.clear()
    closed = threading.Event()
    closed.set()
    assert sm._CUSTODY_SWEEP_LOCK.acquire(blocking=False)
    sm._run_periodic_custody_sweep(closed)
    assert done == [], "a generation already closed at thread start mutates nothing"
    assert not sm._CUSTODY_SWEEP_LOCK.locked()


def test_a_stop_in_flight_makes_the_sweep_gateway_attach_only(tmp_path, monkeypatch):
    """ARCHITECTURE §9 / Process Custody Rule: an `ensure` between a stop request
    and the daemon stop starts the engine the teardown is about to end. A healthy
    generation may still start it; a stop, restart or panic makes the sweep attach-only."""
    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros import delegate_custody as dc
    from ouroboros import delegate_recovery, server_maintenance as sm
    from supervisor import queue

    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(delegate_recovery, "recoverable_task_ids", lambda root: set())
    used: list = []
    monkeypatch.setattr(daemon_mod, "ensure_owned_gateway",
                        lambda **kwargs: used.append(("ensure", kwargs)) or SimpleNamespace())
    monkeypatch.setattr(daemon_mod, "read_owned_gateway",
                        lambda: used.append(("attach", {})) or SimpleNamespace())
    monkeypatch.setattr(dc, "reconcile_orphaned_runs",
                        lambda root, **kwargs: kwargs["gateway_factory"]() and [])

    sm._reconcile_delegated_runs(lambda: set())
    assert used == [("ensure", {"admission_wait_sec": 0})], "a healthy sweep may start the engine"

    stop = threading.Event()
    stop.set()
    sm._reconcile_delegated_runs(lambda: set(), stop_event=stop)
    assert used[-1] == ("attach", {}), "a closed generation never ensures"

    sm._restart_requested.set()
    try:
        sm._reconcile_delegated_runs(lambda: set())
    finally:
        sm._restart_requested.clear()
    assert used[-1] == ("attach", {}), "a restart in flight never ensures either"


def test_reconcile_cadence_is_stamped_when_the_pass_ends(quiet_tick, monkeypatch, caplog):
    """Off the loop thread the rule of issue #1230 still holds: the marker is stamped
    when the pass ENDS (before its latch opens), so a pass slower than its cadence never
    re-arms on the very next tick; a pass that raises still stamps and still releases,
    and its failure is a WARNING on the maintenance thread, never a crash of the loop."""
    import logging

    sm = quiet_tick
    clock = [1_000_000.0]
    monkeypatch.setattr(sm.time, "time", lambda: clock[0])
    monkeypatch.setattr("ouroboros.observability.retry_pending_child_ref_promotions", lambda root, **kwargs: {})
    monkeypatch.setattr(sm, "_STEP_FAILURES", {})
    threads = _track_threads(monkeypatch)
    calls = []

    def slow_pass(**kwargs):
        calls.append(kwargs)
        clock[0] += 400.0
        if len(calls) == 2:
            raise RuntimeError("the pass itself failed")

    monkeypatch.setattr(sm, "_periodic_zombie_reconcile", slow_pass)
    last_custody_reap = [clock[0] + 10_000]  # the 600 s sweep is not due

    def tick(marker):
        sm._periodic_supervisor_maintenance(last_custody_reap, marker)
        for thread in threads:
            thread.join(5)

    marker = [clock[0] - 301]
    tick(marker)
    assert len(calls) == 1 and marker[0] == clock[0]  # stamped at the END of the 400 s pass
    tick(marker)
    assert len(calls) == 1, "a pass slower than its cadence must not re-arm on the next tick"

    clock[0] += 301.0
    with caplog.at_level(logging.WARNING):
        tick(marker)
    assert len(calls) == 2 and marker[0] == clock[0], "a failing pass still stamps when it ends"
    assert not sm._RECONCILE_SWEEP_LOCK.locked(), "and still releases its latch"
    assert any("Periodic reconcile_sweep failed (failure 1 in a row)" in r.getMessage() for r in caplog.records)
    tick(marker)
    assert len(calls) == 2
    clock[0] += 301.0
    tick(marker)
    assert len(calls) == 3, "the next eligible run still happens after the cadence"
    assert all(not thread.is_alive() for thread in threads)


def _quiet_reconcile_steps(monkeypatch, sm, done: list, *, heal=None):
    """Stub every step of the reconcile block; each records its name (and the thread it
    ran on) so a test can pin order, placement and the generation cut."""
    def step(name, value=0):
        def run(*_a, **_k):
            done.append((name, threading.current_thread().name))
            return value
        return run

    monkeypatch.setattr("ouroboros.skill_review_runner.reconcile_stale_review_jobs", step("review_jobs"))
    monkeypatch.setattr("ouroboros.task_status.reconcile_orphaned_running_tasks", heal or step("orphans"))
    monkeypatch.setattr("ouroboros.projects_registry.reconcile_projects", step("projects"))
    monkeypatch.setattr(sm, "_resume_interrupted_project_deletions", step("deletions"))
    monkeypatch.setattr("ouroboros.observability.retry_pending_child_ref_promotions", step("child_refs", {}))


def test_a_slow_reconcile_block_never_holds_the_loop_tick(quiet_tick, monkeypatch):
    """The 300 s zombie/artifact reconcile ran INLINE on the tick: a history-sized heal
    held the drain, the fence acks and assignment for its whole walk. The tick that
    STARTS it now returns at once; a tick while it is busy SKIPS (one thread, no queue);
    the marker is stamped only when the pass ENDS; the latch opens in ``finally``; and
    the orphan-heal notification reaches the alarm clock from the maintenance thread."""
    sm = quiet_tick
    threads = _track_threads(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    done, healed = [], []

    def slow_heal(root, **kwargs):
        entered.set()
        assert release.wait(5), "the test must release the reconcile block"
        return 2

    _quiet_reconcile_steps(monkeypatch, sm, done, heal=slow_heal)
    marker = [0.0]
    try:
        started = time.monotonic()
        sm._periodic_supervisor_maintenance(
            [time.time()], marker,
            on_orphans_healed=lambda count: healed.append((count, threading.current_thread().name)))
        tick = time.monotonic() - started
        assert entered.wait(5), "the reconcile block really ran"
        assert tick < 2.0, f"the loop tick waited {tick:.1f}s for the reconcile block"
        assert marker[0] == 0.0, "not stamped until the pass ENDS (issue #1230)"
        sm._periodic_supervisor_maintenance([time.time()], marker)
        assert [thread.name for thread in threads] == ["reconcile-maintenance"], "busy => skipped"
        assert threads[0].daemon and threads[0].is_alive()
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert all(not thread.is_alive() for thread in threads)
    assert healed == [(2, "reconcile-maintenance")]
    assert [name for name, _ in done] == ["review_jobs", "projects", "deletions", "child_refs"]
    assert {thread for _, thread in done} == {"reconcile-maintenance"}
    assert marker[0] > 0.0 and not sm._RECONCILE_SWEEP_LOCK.locked(), "stamped, then released"


def test_child_ref_promotion_retries_left_the_cancel_sweep(quiet_tick, monkeypatch):
    """The child-ref promotion walk (every child drive, a result load each) rode the
    20 s cancel sweep under ITS latch, so the cancel-intent watchdog could not run again
    until the walk ended. It now runs AFTER the heal in the reconcile block; a 20 s
    sweep that is due runs its three steps and nothing else."""
    sm = quiet_tick
    done: list = []
    _quiet_reconcile_steps(monkeypatch, sm, done)
    monkeypatch.setattr("supervisor.task_lifecycle.sweep_cancel_intents", lambda: done.append(("cancel", "")) or {})
    monkeypatch.setattr("supervisor.terminal_delivery.replay_pending_deliveries", lambda root: done.append(("delivery", "")))
    monkeypatch.setattr(sm, "_reconcile_abandoned_usage", lambda root: done.append(("usage", "")))
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [0.0])
    monkeypatch.setattr(sm, "_CANCEL_INTENT_SWEEP_LOCK", threading.Lock())
    threads = _track_threads(monkeypatch)

    sm._periodic_supervisor_maintenance([time.time()], [time.time()])  # the 20 s sweep alone
    for thread in threads:
        thread.join(5)
    assert [name for name, _ in done] == ["cancel", "delivery", "usage"], "no history-sized step"
    assert not sm._CANCEL_INTENT_SWEEP_LOCK.locked()

    done.clear()
    sm._periodic_supervisor_maintenance([time.time()], [0.0])  # the 300 s block alone
    for thread in threads:
        thread.join(5)
    assert [name for name, _ in done] == ["review_jobs", "orphans", "projects", "deletions", "child_refs"]
    assert [thread.name for thread in threads] == ["terminal-maintenance", "reconcile-maintenance"]


def test_a_closed_generation_stops_the_reconcile_block_before_its_next_mutation(quiet_tick, monkeypatch):
    """Like the custody block, the reconcile block outlives the loop that started it,
    so it re-reads the per-generation token before EVERY step and stops; an OPEN
    generation runs every step, heal before promote; a restart in flight closes it the
    same way. The marker is stamped and the latch released whichever way it ends."""
    sm = quiet_tick
    done: list = []
    stop = threading.Event()

    def heal_then_close(root, **kwargs):
        done.append(("orphans", threading.current_thread().name))
        stop.set()
        return 0

    _quiet_reconcile_steps(monkeypatch, sm, done, heal=heal_then_close)
    marker = [0.0]
    assert sm._RECONCILE_SWEEP_LOCK.acquire(blocking=False)
    sm._run_periodic_reconcile_sweep(marker, stop)
    assert [n for n, _ in done] == ["review_jobs", "orphans"], "generation ended mid-pass: nothing further"
    assert marker[0] > 0.0 and not sm._RECONCILE_SWEEP_LOCK.locked()

    done.clear()
    _quiet_reconcile_steps(monkeypatch, sm, done)
    assert sm._RECONCILE_SWEEP_LOCK.acquire(blocking=False)
    sm._run_periodic_reconcile_sweep(marker, threading.Event())
    assert [n for n, _ in done] == ["review_jobs", "orphans", "projects", "deletions", "child_refs"]

    done.clear()
    closed = threading.Event()
    closed.set()
    assert sm._RECONCILE_SWEEP_LOCK.acquire(blocking=False)
    sm._run_periodic_reconcile_sweep(marker, closed)
    assert done == [] and not sm._RECONCILE_SWEEP_LOCK.locked(), "closed at thread start: mutates nothing"

    sm._restart_requested.set()
    try:
        assert sm._RECONCILE_SWEEP_LOCK.acquire(blocking=False)
        sm._run_periodic_reconcile_sweep(marker, threading.Event())
    finally:
        sm._restart_requested.clear()
    assert done == [] and not sm._RECONCILE_SWEEP_LOCK.locked(), "a restart in flight closes it too"


def test_a_reconcile_thread_that_cannot_start_releases_its_latch_and_waits_a_cadence(
    quiet_tick, monkeypatch, caplog,
):
    """A start refusal is the one failure the tick sees itself: the latch it took opens
    again, the marker is stamped (one warning per cadence, not one per 0.5 s tick)."""
    import logging

    sm = quiet_tick
    clock = [5_000.0]
    monkeypatch.setattr(sm, "time", SimpleNamespace(time=lambda: clock[0]))

    def refuse(**kwargs):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(sm, "threading", SimpleNamespace(Thread=refuse))
    marker = [0.0]
    with caplog.at_level(logging.WARNING):
        sm._periodic_supervisor_maintenance([clock[0]], marker)
    assert marker[0] == clock[0]
    assert sm._RECONCILE_SWEEP_LOCK.acquire(blocking=False)
    sm._RECONCILE_SWEEP_LOCK.release()
    assert any("reconcile-maintenance could not start" in r.getMessage() for r in caplog.records)
    clock[0] += 10.0
    sm._periodic_supervisor_maintenance([clock[0]], marker)
    assert marker[0] == clock[0] - 10.0, "not due again until a full cadence has passed"


def test_startup_custody_still_runs_inline_and_starts_no_maintenance_thread(tmp_path, monkeypatch):
    """Startup reconciles addressed custody inline; historical refresh and bulk
    state housekeeping belong to the later maintenance pass."""
    from ouroboros import process_custody as pc
    from ouroboros import server_maintenance as sm

    order: list = []
    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sm, "_installed_skill_names", lambda: None)
    monkeypatch.setattr(pc, "reap_orphaned_processes",
                        lambda root, **kw: order.append(("reap", threading.current_thread().name)) or [])
    monkeypatch.setattr(sm, "_reconcile_delegated_runs",
                        lambda live, **kw: order.append(("reconcile", threading.current_thread().name)))
    monkeypatch.setattr("ouroboros.delegate_terminal.backfill_terminal_reconciliations",
                        lambda root: order.append(("backfill", "")) or [])
    monkeypatch.setattr(sm, "_cursor_refresh_settled_terminals", lambda live=None: order.append(("cursor", "")))
    monkeypatch.setattr("supervisor.terminal_delivery.replay_pending_deliveries",
                        lambda root: order.append(("replay", "")))
    monkeypatch.setattr("ouroboros.delegate_state_sweep.sweep_settled_delegate_state",
                        lambda root: order.append(("delegate_state", "")) or {})
    threads = _track_threads(monkeypatch)

    sm._startup_custody_sweep()
    assert [name for name, _ in order] == ["reap", "reconcile", "backfill", "replay"]
    assert {thread for _, thread in order[:2]} == {threading.current_thread().name}
    assert threads == [], "startup custody never hands its work to a maintenance thread"


def test_drive_custody_rides_the_reconcile_pass_bounded_and_never_startup(tmp_path, monkeypatch):
    """Child and direct drives are settled by the off-loop reconcile pass through the one
    settlement owner, with the supervisor's probe and ownership interlock, at most
    DRIVE_SETTLEMENTS_PER_PASS attempts per layout and pass from a memory-only cursor;
    the startup sweep copies and hashes no child store (readiness waits on nothing)."""
    from ouroboros import headless, server_maintenance as sm
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.startup_migrations import prepare_startup_state
    prepare_startup_state(tmp_path)

    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sm, "_DRIVE_PRUNE_CURSOR", {"headless": "", "direct": ""})
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", "1")
    monkeypatch.setattr("ouroboros.retention.age_cutoff", lambda *a, **k: 4_000_000_000)
    monkeypatch.setattr(headless, "DRIVE_SETTLEMENTS_PER_PASS", 2)
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    interlocks = []

    import supervisor.queue as queue_mod
    real = queue_mod.task_settlement_interlock

    def counted(stop=None):
        interlocks.append(threading.current_thread().name)
        return real(stop=stop)

    monkeypatch.setattr(queue_mod, "task_settlement_interlock", counted)
    for name in ("d1", "d2", "d3"):
        drive = headless.prepare_task_drive(tmp_path, name, "empty")
        write_task_result(tmp_path, name, "cancelled", result="x", delegation_role="subagent", child_drive_root=str(drive))
    base = tmp_path / "state" / "headless_tasks"

    sm._startup_prune_sweeps()
    assert sorted(p.name for p in base.iterdir()) == ["d1", "d2", "d3"], "startup settles no drive"

    sm._run_drive_custody_pass()
    assert sorted(p.name for p in base.iterdir()) == ["d3"] and sm._DRIVE_PRUNE_CURSOR["headless"] == "d2"
    assert len(interlocks) == 2
    sm._run_drive_custody_pass()
    assert list(base.iterdir()) == [] and sm._DRIVE_PRUNE_CURSOR["headless"] == "d3"
    assert all(load_task_result(tmp_path, name)["status"] == "cancelled" for name in ("d1", "d2", "d3"))

    closed = threading.Event()
    closed.set()
    drive = headless.prepare_task_drive(tmp_path, "d4", "empty")
    write_task_result(tmp_path, "d4", "cancelled", result="x", delegation_role="subagent", child_drive_root=str(drive))
    sm._run_drive_custody_pass(closed)
    assert drive.is_dir(), "a closed generation settles nothing"


def test_startup_sweeps_only_the_script_fallback_and_owes_the_tree_walk_to_the_first_pass(tmp_path, monkeypatch):
    """The whole-tree walk for orphaned atomic temp files left the startup path (it delayed
    readiness by hundreds of thousands of stat calls): startup sweeps only the top-level
    tmp_scripts fallback, and the first off-loop reconcile pass of the generation sweeps the
    tree once; a deferred startup owes nothing."""
    from ouroboros import server_maintenance as sm

    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sm, "_STARTUP_TEMP_SWEEP_OWED", [False])
    monkeypatch.setattr(sm, "_periodic_zombie_reconcile", lambda **kwargs: None)
    monkeypatch.setattr(sm, "_run_drive_custody_pass", lambda stop_event=None: None)
    aged = time.time() - 7200
    script = tmp_path / "tmp_scripts" / "script_dead.py"
    orphan = tmp_path / "state" / "deep" / ".state.json.tmp.1.2.abc"
    for path in (script, orphan):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        os.utime(path, (aged, aged))

    sm._startup_prune_sweeps()
    assert not script.exists() and orphan.exists() and sm._STARTUP_TEMP_SWEEP_OWED == [True]

    assert sm._RECONCILE_SWEEP_LOCK.acquire(blocking=False)
    sm._run_periodic_reconcile_sweep([0.0], threading.Event())
    assert not orphan.exists() and sm._STARTUP_TEMP_SWEEP_OWED == [False]

    orphan.write_text("x", encoding="utf-8")
    os.utime(orphan, (aged, aged))
    assert sm._RECONCILE_SWEEP_LOCK.acquire(blocking=False)
    sm._run_periodic_reconcile_sweep([0.0], threading.Event())
    assert orphan.exists(), "the walk runs once per generation, not every pass"

    monkeypatch.setattr(sm, "_STARTUP_TEMP_SWEEP_OWED", [False])
    sm._startup_prune_sweeps(preserve_task_sources=True)
    assert sm._STARTUP_TEMP_SWEEP_OWED == [False]
