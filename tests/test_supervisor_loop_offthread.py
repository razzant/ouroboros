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
Every guard below is pinned in both directions.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

@pytest.fixture(autouse=True)
def _fresh_custody_sweep_latch(monkeypatch):
    """The custody latch is process-global and a pass may outlive the test that started it
    (the first tick of any real loop starts one): every test here gets its own latch, so a
    busy one left behind by another test can neither skip this sweep nor be released by it."""
    import threading

    from ouroboros import server_maintenance

    monkeypatch.setattr(server_maintenance, "_CUSTODY_SWEEP_LOCK", threading.Lock())


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


def test_reconcile_cadence_is_stamped_when_the_pass_ends(quiet_tick, monkeypatch):
    """The 300-s zombie reconcile stamps its marker when the pass ENDS, so a pass
    slower than its cadence never re-arms on the very next tick (issue #1230); a
    pass that raises still stamps, and the next eligible run still happens."""
    sm = quiet_tick
    clock = [1_000_000.0]
    monkeypatch.setattr(sm.time, "time", lambda: clock[0])
    calls = []

    def slow_pass(**kwargs):
        calls.append(kwargs)
        clock[0] += 400.0
        if len(calls) == 2:
            raise RuntimeError("the pass itself failed")

    monkeypatch.setattr(sm, "_periodic_zombie_reconcile", slow_pass)
    busy = threading.Lock()
    busy.acquire()
    monkeypatch.setattr(sm, "_CANCEL_INTENT_SWEEP_LOCK", busy)  # the 20 s sweep is skipped
    last_custody_reap = [clock[0] + 10_000]  # the 600 s sweep is not due
    marker = [clock[0] - 301]

    sm._periodic_supervisor_maintenance(last_custody_reap, marker)
    assert len(calls) == 1 and marker[0] == clock[0]  # stamped at the END of the 400 s pass
    sm._periodic_supervisor_maintenance(last_custody_reap, marker)
    assert len(calls) == 1, "a pass slower than its cadence must not re-arm on the next tick"

    clock[0] += 301.0
    with pytest.raises(RuntimeError):
        sm._periodic_supervisor_maintenance(last_custody_reap, marker)
    assert len(calls) == 2 and marker[0] == clock[0], "a failing pass still stamps when it ends"
    sm._periodic_supervisor_maintenance(last_custody_reap, marker)
    assert len(calls) == 2
    clock[0] += 301.0
    sm._periodic_supervisor_maintenance(last_custody_reap, marker)
    assert len(calls) == 3, "the next eligible run still happens after the cadence"
