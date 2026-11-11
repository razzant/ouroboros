"""Elasticity through actual assignment, readiness, snapshot and reaper consumers."""

import threading
import time
from types import SimpleNamespace

import pytest

from supervisor import queue, worker_pool_lifecycle as lifecycle, workers
from tests.test_owner_wait_pool import pool as pool, park, Process, InputQueue
from tests.worker_pool_support import affirm_ready


# Bind real threads/ledger before the pool fixture stubs them for contention/ledger tests.
_REAL_THREAD = threading.Thread
_REAL_RECORD = lifecycle._record_worker_processes


@pytest.fixture
def assigning(pool, monkeypatch):
    from supervisor import evolution_lifecycle, state
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    return pool


def task(tid, **fields):
    return {"id": tid, "type": "task", "chat_id": 0, "text": "work", **fields}


def ready(slot):
    slot.reaping = False
    return affirm_ready(slot)


def handed(slot):
    """The task rows put to ``slot`` so far (its input queue drained); a quiescence probe hands it no work."""
    rows = []
    while not slot.in_q.empty():
        item = slot.in_q.get_nowait()
        if item.get("type") != "worker_quiescence":
            rows.append(item.get("id") or item.get("phase"))
    return rows


@pytest.mark.parametrize("cap", [1, 3, 10])
def test_burst_materializes_only_residual_demand_and_never_demand_plus_spare(assigning, monkeypatch, cap):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(cap))
    workers.PENDING.extend(task(str(i)) for i in range(cap + 2))
    workers.assign_tasks()  # sole current worker is the busy parent
    assert len(workers.WORKERS) == cap
    assert len(assigning.created) == cap - 1
    assert workers.RUNNING == {"owner": assigning.meta}  # forecasting claims nothing
    assert all("admitted_dispatch" not in row for row in workers.PENDING)
    for slot in workers.WORKERS.values():
        ready(slot)
    workers.assign_tasks()
    assert len(workers.RUNNING) == cap and len(workers.PENDING) == 3
    workers.assign_tasks()
    assert len(assigning.created) == cap - 1


def test_sole_busy_parent_can_grow_child_but_same_project_roots_do_not(assigning, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    assigning.meta["task"].update(project_id="project", root_task_id="owner")
    workers.PENDING.append(task("held-root", project_id="project"))
    workers.assign_tasks()
    assert not assigning.created
    workers.PENDING.append(task("child", project_id="project", root_task_id="owner",
                                parent_task_id="owner", delegation_role="subagent", depth=1))
    workers.assign_tasks()
    assert len(assigning.created) == 1
    ready(workers.WORKERS[1])
    workers.assign_tasks()
    assert "child" in workers.RUNNING and "held-root" not in workers.RUNNING
    assert len(workers.WORKERS) == 2


def test_forecast_projects_project_leases_and_root_child_limit(assigning, monkeypatch):
    from ouroboros import config
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(10))
    monkeypatch.setattr(config, "get_max_active_subagents_per_root", lambda: 1)
    workers.PENDING.extend([task("a", project_id="p"), task("b", project_id="p"),
        task("c", delegation_role="subagent", root_task_id="owner", parent_task_id="owner", depth=1),
        task("d", delegation_role="subagent", root_task_id="owner", parent_task_id="owner", depth=1)])
    workers.assign_tasks()
    assert len(assigning.created) == 2  # one project root and one eligible child
    assert set(workers.RUNNING) == {"owner"}


@pytest.mark.parametrize("hold", ["_owner_hold", "_project_admission_restore_hold", "_budget_pause"])
def test_held_rows_do_not_grow_the_pool(assigning, monkeypatch, hold):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(10))
    workers.PENDING.append(task("held", **{hold: {"reason": "held"}}))
    workers.assign_tasks()
    assert not assigning.created


def test_zero_budget_resumes_owner_wait_and_keeps_parked_evolution_across_forecasts(assigning, monkeypatch):
    """At zero dispatch only owner-wait; never clean paused work, with or without a free slot."""
    from supervisor import state
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 0)
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    resume = {**assigning.meta["task"], "_owner_wait_resume": {**assigning.wait, "restart_transaction_id": "tx"}}
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    workers.PENDING.extend([task("evo", type="evolution", admitted_dispatch="none"), resume])
    for _tick in range(2):  # the second tick has no free worker: forecast only
        workers.assign_tasks()
        assert [row["id"] for row in workers.PENDING] == ["evo"] and workers.PENDING[0]["_budget_pause"]
    assert assigning.original.in_q.get_nowait()["_owner_wait_resume"] == resume["_owner_wait_resume"]
    assert set(workers.RUNNING) == {"owner"} and not assigning.created


def test_parked_authors_lend_capacity_but_buy_no_replacement_without_demand(assigning, monkeypatch):
    """4A: the pool's quiescent spare is the conversation lane's; growth buys only demand."""
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(10))
    park(assigning)
    workers.assign_tasks()
    assert not assigning.created and not assigning.original.active_capacity
    workers.PENDING.append(task("demanded"))
    workers.assign_tasks()
    assert len(assigning.created) == 1  # the lent capacity is spent on demanded work only
    workers.assign_tasks()
    assert len(assigning.created) == 1  # booting reservation is already charged
    assert workers.RUNNING["owner"] is assigning.meta


def test_a_readiness_park_allows_growth_beside_a_healthy_slot_and_no_healthy_slot_is_exhaustion(assigning, monkeypatch):
    """One park beside a healthy process permits growth; a successful worker_ready clears the park count.
    With no healthy slot and a new readiness park, the pool is exhausted."""
    from supervisor import pool_admission

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "")  # no number: memory, the floor and finiteness decide
    broken = workers.Worker(1, Process(43), InputQueue(), reaping=True, booting=True)
    workers.WORKERS[1] = broken
    broken.proc.alive = False
    lifecycle._replace_or_park_unready_slot(1, broken, 0, 0.0, lifecycle.WORKER_READY_MAX_ATTEMPTS)
    assert broken.readiness_exhausted and workers.worker_pool_admission_state()["available"]  # the parent is healthy
    workers.PENDING.extend([task("a"), task("b")])
    with queue._queue_lock:
        assert lifecycle.admissible_starts(pool_admission.read()) > 0, "a blip does not freeze healthy capacity"
    workers.assign_tasks()
    assert len(assigning.created) == 2 and "a" not in workers.RUNNING
    fresh = workers.WORKERS[2]
    assert lifecycle._open_ready_slot(2, fresh, {"pid": fresh.proc.pid}, "", 0, 0.0, 1)
    assert pool_admission.parked(lifecycle.project_slots()) == 0
    with queue._queue_lock:
        assert lifecycle.admissible_starts(pool_admission.read()) > 0, "growth resumes after a ready"
    workers.assign_tasks()  # the opened slot takes one row (an idle process: the machine is not read)
    workers.assign_tasks()  # the other row already has its booting reservation
    assert workers.RUNNING["a"]["worker_id"] == 2 and len(assigning.created) == 2
    for wid in (0, 2, 3):  # only the parked slot is left
        workers.WORKERS.pop(wid, None)
    assert workers._worker_pool_execution_state()["disabled_reason"] == "", "a ready since the park lifts it"
    pool_admission.note_park(broken)  # parked again after the last ready, and no healthy slot: exhausted
    assert workers._worker_pool_execution_state()["disabled_reason"] == "worker_readiness_exhausted"


@pytest.mark.parametrize("state", ["alive_unready", "dead_after_ready", "foreign_birth"])
def test_assignment_and_physical_free_count_require_current_live_ready(assigning, state):
    slot = assigning.original
    workers.RUNNING.clear()
    slot.busy_task_id = None
    if state == "alive_unready":
        slot.ready_identity = None
    elif state == "dead_after_ready":
        slot.proc.alive = False
    else:
        slot.process_birth = "another-birth"
    workers.PENDING.append(task("queued"))
    workers.assign_tasks()
    assert not workers.RUNNING and len(workers.PENDING) == 1
    queue.persist_queue_snapshot()
    import json
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert snap["assignable_idle_workers"] == 0


def acknowledge(slot):
    with queue._queue_lock:
        lifecycle.request_worker_quiescence(slot)
    probe = slot.in_q.get_nowait()
    lifecycle.handle_worker_quiescent({**probe, "worker_id": slot.wid, "pid": slot.proc.pid,
                                      "process_birth": slot.process_birth}, workers)
    return probe


def test_assignment_revokes_late_probe_and_surplus_returns_to_stable_spare(assigning, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    extra = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    workers.WORKERS[1] = extra
    probes = [acknowledge(slot) for slot in workers.WORKERS.values()]
    workers.PENDING.extend([task("first"), task("second")])
    workers.assign_tasks()
    assert len(workers.RUNNING) == 2 and not extra.quiescence_token
    lifecycle.handle_worker_quiescent({**probes[1], "worker_id": 1, "pid": 43,
                                      "process_birth": extra.process_birth}, workers)
    assert not extra.quiescence_confirmed
    workers.RUNNING.clear()
    for slot in workers.WORKERS.values():
        slot.busy_task_id = None
        slot.in_q.get_nowait()  # consume the real assignment
        acknowledge(slot)
    workers.assign_tasks()
    assert extra.retiring and extra.active_capacity and extra.proc.is_alive()
    lifecycle.reap_idle_worker(queue._reap_queue.get_nowait())
    assert workers.WORKERS == {0: assigning.original}


def test_unconfirmed_retirement_keeps_reservation_and_never_becomes_crash(assigning, monkeypatch):
    from supervisor.task_reaper import TerminalFileRecoveryPending
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    acknowledge(assigning.original)
    lifecycle.retire_quiescent_worker(assigning.original)
    monkeypatch.setattr(lifecycle, "kill_worker_tree", lambda *a, **kw: None)
    job = queue._reap_queue.get_nowait()
    with pytest.raises(TerminalFileRecoveryPending):
        lifecycle.reap_idle_worker(job)
    workers.PENDING.append(task("queued"))
    workers.assign_tasks()
    # The unconfirmed process stays charged as retiring, never a crash; a new one is the machine's to admit
    # (nothing acts and nothing starts: the floor), not a slot number's.
    assert assigning.original.retiring and len(assigning.created) == 1
    assert not workers._ensure_workers_healthy_locked(queue)[1]
    assert queue._reap_queue.empty()


@pytest.mark.parametrize("reason", ["owner_stop", "managed_update", "worker_readiness_exhausted"])
def test_elasticity_does_not_reopen_disabled_pool(assigning, monkeypatch, reason):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", reason)
    workers.PENDING.append(task("queued"))
    workers.assign_tasks()
    assert not assigning.created


@pytest.mark.parametrize("prior", ["", "worker_readiness_exhausted"])
def test_failed_disable_fence_restores_the_prior_admission_state(assigning, monkeypatch, prior):
    """An enabled pool reopens after an aborted shutdown; an exhaustion latch stays closed."""
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", prior)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kw: False)
    assert workers.kill_workers(disable_reason=prior or "managed_update") is False
    assert workers._WORKER_POOL_DISABLED_REASON == prior
    assert workers.WORKERS == {0: assigning.original} and assigning.original.proc.is_alive()


def test_active_work_buys_no_spare_and_retires_an_acknowledged_idle_resident(assigning, monkeypatch):
    """4A: the project pool keeps no spare of its own beside work; the lane holds the one spare."""
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    spare = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    workers.WORKERS[1] = spare
    acknowledge(spare)
    workers.assign_tasks()
    assert spare.retiring and not assigning.created and not queue._reap_queue.empty()


def _resume_with_fresh_work(assigning):
    from supervisor import worker_owner_wait
    park(assigning)
    lifecycle._spawn_worker_slot(1)  # a second project worker demand bought while the author parked (4A)
    replacement = ready(workers.WORKERS[1])
    worker_owner_wait.handle_owner_wait({**assigning.event, "phase": "resume"}, workers)
    workers.PENDING.append(task("fresh"))
    return replacement


def test_a_warm_return_goes_first_without_holding_or_retiring_anything_then_the_queue_runs(assigning):
    """The one return rule at cap 1: the author's warm return is granted before fresh work is chosen, on its own
    process; the idle replacement is neither held nor probed nor retired, and fresh work waits only for the owner's
    cap of acting tasks (the author acts again), not for a process. When the author completes, the queue runs."""
    import json
    replacement = _resume_with_fresh_work(assigning)
    workers.assign_tasks()
    assert assigning.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert assigning.original.active_capacity and handed(replacement) == [] and not replacement.retiring
    assert "fresh" not in workers.RUNNING, "the cap counts the returned author"
    queue.persist_queue_snapshot()
    document = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert (document["acting_worker_count"], document["assignable_idle_workers"]) == (1, 1)
    assert len(assigning.created) == 1, "growth never spends a process on a return"
    workers.RUNNING.pop("owner")  # the resumed author completes; the cap frees
    assigning.original.busy_task_id = None
    workers.assign_tasks()
    assert workers.RUNNING["fresh"]["worker_id"] in {0, 1} and len(assigning.created) == 1


@pytest.mark.parametrize("revocation", ["owner_stop", "task_ended"])
def test_a_return_without_its_authority_takes_nothing_from_fresh_work(assigning, revocation):
    """A wake whose authority is gone (an owner Stop, or the task left) is not granted; the parked author does not
    act, so fresh work runs at once on the idle replacement."""
    from ouroboros.cancel_intents import request_cancel
    replacement = _resume_with_fresh_work(assigning)
    if revocation == "owner_stop":
        request_cancel(assigning.root, "owner", reason="owner stop", source="test")
    else:
        workers.RUNNING.pop("owner")
    workers.assign_tasks()
    assert assigning.original.in_q.empty() and workers.RUNNING["fresh"]["worker_id"] == 1
    assert not replacement.retiring


@pytest.mark.parametrize("revocation", ["owner_stop", "newer_pause"])
def test_unauthorized_pending_resume_never_withholds_growth_from_admissible_work(assigning, revocation):
    """A stale request flag is no reservation: only the owner's authorized Resume takes capacity back."""
    from ouroboros import owner_pause
    from ouroboros.cancel_intents import request_cancel
    from supervisor import worker_owner_wait
    event = {**assigning.event, "checkpoint": {**assigning.wait, "reason": "owner_pause",
             "budget_paused_sec": 0.0, "sleep_started_at": time.time()}}
    worker_owner_wait.handle_owner_wait(event, workers)
    assert assigning.original.in_q.get_nowait()["phase"] == "parked"
    worker_owner_wait.handle_owner_wait({**event, "phase": "resume", "resume_reason": "control:owner_resume"}, workers)
    if revocation == "owner_stop":
        request_cancel(assigning.root, "owner", reason="owner stop", source="test")
    else:
        owner_pause.install_fence(assigning.root, "owner", request_id="newer-pause")
    workers.PENDING.append(task("independent"))
    workers.assign_tasks()
    assert assigning.meta["owner_wait_resume_requested"] and not assigning.original.active_capacity
    assert len(assigning.created) == 1, "the waiting author acts no more while its Resume is not authorized"
    ready(workers.WORKERS[1])
    workers.assign_tasks()
    assert workers.RUNNING["independent"]["worker_id"] == 1


@pytest.mark.parametrize("allowed", [False, True])
def test_writer_gate_grows_only_for_its_authorized_assisted_task(assigning, monkeypatch, allowed):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "managed_update")
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda row: allowed and row["id"] == "resolver")
    workers.PENDING.extend([task("ordinary"), task("resolver")])
    workers.assign_tasks()
    assert len(assigning.created) == int(allowed)
    assert set(workers.RUNNING) == {"owner"}


def test_unknown_timeout_reaper_is_charged_but_is_not_future_ready_capacity(assigning, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor import task_reaper
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(4))
    slot = assigning.original
    slot.busy_task_id, slot.reaping = None, True
    monkeypatch.setattr(lifecycle, "kill_worker_tree", lambda *a, **kw: None)
    monkeypatch.setattr(task_reaper, "send_with_budget", lambda *a, **kw: None)
    task_reaper.reap_timed_out_task({"worker_id": slot.wid, "worker": slot, "proc": slot.proc,
        "task_id": "owner", "task": assigning.meta["task"], "meta": assigning.meta,
        "attempt": 3, "drive_root": str(assigning.root)})
    assert slot.reaping and slot.proc.is_alive() and not slot.booting
    assert load_task_result(assigning.root, "owner")["reason_code"] == "reaper_wedged_worker_alive"
    workers.PENDING.append(task("eligible"))
    workers.assign_tasks()
    assert len(assigning.created) == 1 and len(workers.WORKERS) == 2
    assert workers.WORKERS[1].booting and workers.WORKERS[0] is slot


def test_late_current_ready_row_cannot_unreap_retiring_slot(assigning):
    slot = assigning.original
    workers.RUNNING.clear()
    slot.busy_task_id = None
    acknowledge(slot)
    lifecycle.retire_quiescent_worker(slot)
    assert not lifecycle._open_ready_slot(slot.wid, slot,
        {"pid": slot.proc.pid, "process_birth": slot.process_birth}, "", 0, 0, 1)
    assert slot.retiring and slot.reaping and not lifecycle.worker_ready(slot)


def _timed_out_dead_slot(assigning):
    slot = assigning.original
    workers.RUNNING.clear()  # a timeout moves custody from RUNNING to its captured reaper job
    slot.busy_task_id, slot.reaping, slot.proc.alive = None, True, False
    return slot


def test_enabled_pool_still_replaces_a_captured_dead_slot(assigning):
    from supervisor import task_reaper
    timed_out = _timed_out_dead_slot(assigning)
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=timed_out)
    fresh = workers.WORKERS[0]
    assert fresh is not timed_out and [fresh.proc] == assigning.created
    assert fresh.booting and fresh.reaping and fresh.active_capacity


@pytest.mark.parametrize("reason", ["worker_readiness_exhausted", "managed_update"])
def test_a_fenced_pool_still_runs_queue_hygiene_but_assigns_and_grows_nothing(assigning, monkeypatch, reason):
    """Under a fence only assignment and growth wait for Restart: a pending task the owner
    cancels is still settled, terminalization retries and Project holds still run."""
    from supervisor import task_admission
    seen = []
    monkeypatch.setattr(workers, "_drop_cancelled_pending", lambda: seen.append("cancelled") or True)
    monkeypatch.setattr(workers, "_retry_terminalization_pending_for_assignment",
                        lambda _queue: seen.append("terminalization"))
    monkeypatch.setattr(task_admission, "revalidate_project_holds", lambda: seen.append("holds"))
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", reason)
    workers.PENDING.append(task("queued"))
    workers.assign_tasks()
    assert seen == ["cancelled", "terminalization", "holds"]
    assert [row["id"] for row in workers.PENDING] == ["queued"] and not assigning.created


@pytest.mark.parametrize("reason", ["managed_update", "worker_readiness_exhausted"])
def test_a_fence_leaves_a_dead_lent_slot_to_its_stop_owner(assigning, monkeypatch, reason):
    park(assigning)
    assigning.original.proc.alive = False
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", reason)
    assert not workers.respawn_worker(0)
    assert workers.WORKERS == {0: assigning.original} and not assigning.created


def test_growth_ledgers_each_fresh_child_once_off_the_loop_and_retirement_ledgers_nothing(assigning, monkeypatch):
    """Reviewer measurements: each spawn re-ledgered every worker, each retirement every survivor (A1);
    the loop thread paid the three ``ps`` per demanded child (Opus R1 D7.3). Each child is now ledgered
    once, by its own readiness thread, before that thread watches for its row."""
    from ouroboros import process_custody
    monkeypatch.setattr(lifecycle, "_record_worker_processes", _REAL_RECORD)
    recorded, readiness = [], []

    def record(root, *, pid, **_kw):
        recorded.append(pid)
        return {"fingerprint": {"start_time": f"birth-{pid}"}}

    monkeypatch.setattr(process_custody, "record_process", record)
    monkeypatch.setattr(lifecycle.threading, "Thread", lambda target, args, **_kw: SimpleNamespace(
        start=lambda: readiness.append((target, args))))
    monkeypatch.setattr(lifecycle, "_watch_booting_slots", lambda *_args: None)  # the ledger half only
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(10))
    workers.PENDING.extend(task(str(i)) for i in range(12))
    workers.assign_tasks()  # the busy parent grows the pool to its cap
    assert recorded == [] and len(readiness) == 9, "the assignment (loop) thread ledgers nothing"
    for target, args in readiness:
        target(*args)
    assert sorted(recorded) == sorted(proc.pid for proc in assigning.created) and len(recorded) == 9
    assert all(slot.process_birth == f"birth-{slot.proc.pid}" for wid, slot in workers.WORKERS.items() if wid)
    workers.RUNNING.clear()
    workers.PENDING.clear()
    for slot in workers.WORKERS.values():
        slot.busy_task_id = None
        acknowledge(ready(slot))
    workers.assign_tasks()  # nine surplus residents retire through the reaper
    while not queue._reap_queue.empty():
        lifecycle.reap_idle_worker(queue._reap_queue.get_nowait())
    assert workers.WORKERS == {0: assigning.original} and len(recorded) == 9


def _run_in_thread(target):
    done = threading.Event()

    def run():
        try:
            target()
        finally:
            done.set()

    thread = _REAL_THREAD(target=run, daemon=True)
    thread.start()
    return thread, done


def test_a_retirement_mid_kill_never_costs_a_pending_task_its_tick(assigning, monkeypatch):
    """Parent witness: every tick's assignment entry blocked while the reaper held the serializer;
    then (owner 14 Oct 21:09 #4) the try-lock skipped the whole turn. The serializer now guards
    membership only: the reserved slot's native kill runs outside it, and the tick assigns."""
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    surplus = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    workers.WORKERS[1] = surplus
    acknowledge(surplus)
    assert lifecycle.retire_quiescent_worker(surplus)
    job = queue._reap_queue.get_nowait()
    entered, release = threading.Event(), threading.Event()

    def native_teardown(pid, **_kw):
        entered.set()
        assert release.wait(10)
        surplus.proc.alive = False

    monkeypatch.setattr(lifecycle, "kill_worker_tree", native_teardown)
    (reaper, _), caller = _run_in_thread(lambda: lifecycle.reap_idle_worker(job)), None
    try:
        assert entered.wait(5)
        workers.PENDING.append(task("fresh"))
        caller, returned = _run_in_thread(workers.assign_tasks)
        assert returned.wait(2), "the assignment entry waited behind native teardown"
        assert workers.RUNNING["fresh"]["worker_id"] == 0, "the tick assigned beside the native kill"
        assert workers.WORKERS[1] is surplus and surplus.retiring, "membership waits for confirmed death"
        checked = []
        monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0.0)
        monkeypatch.setattr(workers, "_ensure_workers_healthy_locked", lambda q: checked.append(q) or ([], False))
        workers.ensure_workers_healthy()
        assert checked == [queue], "the health check keeps its tick too"
    finally:
        release.set()
        for thread in (reaper, caller):
            if thread is not None:
                thread.join(10)
    assert workers.WORKERS == {0: assigning.original} and workers.RUNNING["fresh"]["worker_id"] == 0


def test_a_deferred_turn_never_hands_a_requested_resume_slot_to_fresh_work(assigning):
    replacement = _resume_with_fresh_work(assigning)  # requested after the last maintenance ran
    held, release = threading.Event(), threading.Event()

    def off_loop_lifecycle_operation():
        with lifecycle._WORKER_LIFECYCLE_LOCK:
            held.set()
            assert release.wait(10)

    (holder, _), caller = _run_in_thread(off_loop_lifecycle_operation), None
    try:
        assert held.wait(5)
        caller, returned = _run_in_thread(workers.assign_tasks)
        assert returned.wait(2), "the assignment entry waited behind the lifecycle serializer"
        assert "fresh" not in workers.RUNNING and replacement.in_q.empty()
    finally:
        release.set()
        for thread in (holder, caller):
            if thread is not None:
                thread.join(10)
    workers.assign_tasks()  # the Resume still goes first, on its own process
    assert assigning.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert handed(replacement) == [] and "fresh" not in workers.RUNNING  # the cap counts the returned author


def _retiring_last_spare(assigning):
    """Cap 1: the parked author's replacement, idle and acknowledged, is retiring; its reaper has not run."""
    park(assigning)
    lifecycle._spawn_worker_slot(1)  # a second project worker demand bought while the author parked (4A)
    replacement = ready(workers.WORKERS[1])
    workers.PENDING.append(task("next-independent"))  # accepted while the pool is available
    acknowledge(replacement)
    assert lifecycle.retire_quiescent_worker(replacement)
    assert replacement.retiring and not assigning.original.active_capacity
    return replacement, queue._reap_queue.get_nowait()


def _stop_parked_author(assigning):
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import load_task_result
    from supervisor import task_lifecycle
    request_cancel(assigning.root, "owner", reason="owner stop", source="http")
    assert task_lifecycle.cancel_task_custody("owner", deliver=False) == "cancelled"
    assert load_task_result(assigning.root, "owner")["status"] == "cancelled"


def _ticks_dispatch_next_independent_task():
    for _tick in range(2):
        workers.assign_tasks()
        for slot in workers.WORKERS.values():
            if slot.booting:
                ready(slot)  # the child's own worker_ready row
    assert workers.worker_pool_admission_state()["available"]
    assert list(workers.RUNNING) == ["next-independent"] and len(workers.WORKERS) == 1


@pytest.mark.parametrize("order", ["stop_first", "unknown_release", "exited_before_reap"])
def test_stopping_a_parked_author_beside_a_retiring_spare_keeps_the_enabled_pool_dispatching(assigning, monkeypatch,
                                                                                              order):
    """Reviewer counterexample (cap 1): Stop landed between a spare's reserved retirement and its reap, the Stop's
    respawn counted the retiring slot and forgot the author, and the reap then emptied an enabled pool that growth
    can never refill."""
    from supervisor.task_reaper import TerminalFileRecoveryPending
    replacement, job = _retiring_last_spare(assigning)
    if order == "reap_first":
        lifecycle.reap_idle_worker(job)
    elif order == "unknown_release":
        kill = lifecycle.kill_worker_tree
        monkeypatch.setattr(lifecycle, "kill_worker_tree", lambda *a, **kw: None)
        with pytest.raises(TerminalFileRecoveryPending):
            lifecycle.reap_idle_worker(job)  # charged until its death is observed
        monkeypatch.setattr(lifecycle, "kill_worker_tree", kill)
    elif order == "exited_before_reap":
        replacement.proc.alive = False
    _stop_parked_author(assigning)
    if order != "reap_first":
        lifecycle.reap_idle_worker(job)  # first reap, or the health tick's retry
    assert workers._WORKER_POOL_DISABLED_REASON == "" and replacement not in workers.WORKERS.values()
    assert len(assigning.created) == 2, "one replacement process, started only after the death was seen"
    _ticks_dispatch_next_independent_task()


@pytest.mark.parametrize("control", ["resume", "stop"])
def test_owner_pause_resume_or_stop_at_cap_keeps_the_pool_dispatching(assigning, control):
    """Positive controls through the real Pause park, Resume grant, Stop custody and reaper (MAX 1)."""
    import json

    from ouroboros import owner_pause
    from supervisor import worker_owner_wait
    owner_pause.install_fence(assigning.root, "owner", request_id="pause-1")
    event = {**assigning.event, "checkpoint": {**assigning.wait, "reason": "owner_pause",
             "budget_paused_sec": 0.0, "sleep_started_at": time.time()}}
    worker_owner_wait.handle_owner_wait(event, workers)
    assert assigning.original.in_q.get_nowait()["phase"] == "parked"
    lifecycle._spawn_worker_slot(1)  # a second project worker demand bought while the author parked (4A)
    replacement = ready(workers.WORKERS[1])
    if control == "stop":
        _stop_parked_author(assigning)
        assert workers.WORKERS == {1: replacement}, "an enabled cap-full pool forgets the dead lent slot"
    else:
        owner_pause.release_fence(assigning.root, "owner", reason="owner_resume")
        worker_owner_wait.handle_owner_wait(
            {**event, "phase": "resume", "resume_reason": "control:owner_resume"}, workers)
        worker_owner_wait.grant_warm_returns()  # the return continues on its own process at once
        assert assigning.original.in_q.get_nowait()["phase"] == "resume_granted"
        assert workers.WORKERS == {0: assigning.original, 1: replacement} and assigning.original.active_capacity
        rows = [json.loads(line) for line in (assigning.root / "logs" / "supervisor.jsonl").read_text().splitlines()]
        assert not [row for row in rows if row["type"] == "worker_retired"], "no process retires for a return"
        workers.RUNNING.pop("owner")  # the resumed author completes
        assigning.original.busy_task_id = None
    workers.PENDING.append(task("next-independent"))
    workers.assign_tasks()
    assert list(workers.RUNNING) == ["next-independent"] and len(assigning.created) == 1


@pytest.mark.parametrize("reason", ["worker_readiness_exhausted", "managed_update"])
def test_a_fence_never_refills_the_last_retiring_member(assigning, monkeypatch, reason):
    replacement, job = _retiring_last_spare(assigning)
    _stop_parked_author(assigning)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", reason)
    lifecycle.reap_idle_worker(job)
    assert workers.WORKERS == {1: replacement} and len(assigning.created) == 1


def test_maintenance_never_forgets_an_enabled_pools_last_member(assigning):
    """Same class: maintenance forgot a parked author the timeout reaper had killed before that
    reaper's own respawn could fill the vacancy, emptying the pool."""
    from supervisor import task_reaper, worker_owner_wait
    park(assigning)  # no replacement exists yet
    timed_out = _timed_out_dead_slot(assigning)
    worker_owner_wait.grant_warm_returns()
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=timed_out)
    assert timed_out not in workers.WORKERS.values() and len(assigning.created) == 1
    workers.PENDING.append(task("next-independent"))
    _ticks_dispatch_next_independent_task()


class _ReapedOnJoin(Process):
    """A native child whose SIGKILL death is observable only after it is joined."""

    def __init__(self, pid):
        super().__init__(pid)
        self.killed = False

    def join(self, timeout=None):
        if self.killed:
            self.alive, self.exitcode = False, -9


def test_an_ordinary_idle_exit_retires_in_one_reaper_pass_and_is_journaled(assigning, monkeypatch):
    import json
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    surplus = affirm_ready(workers.Worker(1, _ReapedOnJoin(43), InputQueue()))
    workers.WORKERS[1] = surplus
    probe = acknowledge(surplus)
    assert lifecycle.retire_quiescent_worker(surplus)
    monkeypatch.setattr(lifecycle, "kill_worker_tree", lambda pid, **kw: setattr(surplus.proc, "killed", True))
    lifecycle.reap_idle_worker(queue._reap_queue.get_nowait())  # no TerminalFileRecoveryPending
    assert workers.WORKERS == {0: assigning.original}
    rows = [json.loads(line) for line in (assigning.root / "logs" / "supervisor.jsonl").read_text().splitlines()]
    retired, = [row for row in rows if row["type"] == "worker_retired"]
    assert {k: retired[k] for k in ("worker_id", "pid", "process_birth", "token", "replaced")} == {
        "worker_id": 1, "pid": 43, "process_birth": "test-birth", "token": probe["token"], "replaced": False}


@pytest.mark.parametrize("retained", [0, 2])
def test_a_retired_idle_worker_journals_the_dialogue_rows_it_still_held(assigning, monkeypatch, retained):
    import json
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    surplus = affirm_ready(workers.Worker(1, _ReapedOnJoin(43), InputQueue()))
    workers.WORKERS[1] = surplus
    with queue._queue_lock:
        lifecycle.request_worker_quiescence(surplus)
    probe = surplus.in_q.get_nowait()
    lifecycle.handle_worker_quiescent({**probe, "worker_id": 1, "pid": 43, "process_birth": surplus.process_birth,
                                       "retained_rows": retained}, workers)
    assert lifecycle.retire_quiescent_worker(surplus)
    monkeypatch.setattr(lifecycle, "kill_worker_tree", lambda pid, **kw: setattr(surplus.proc, "killed", True))
    lifecycle.reap_idle_worker(queue._reap_queue.get_nowait())
    rows = [json.loads(line) for line in (assigning.root / "logs" / "supervisor.jsonl").read_text().splitlines()]
    retired, = [row for row in rows if row["type"] == "worker_retired"]
    assert retired.get("dialogue_rows_lost") == (retained or None)


def test_an_unready_child_reaped_on_join_continues_its_bounded_step_in_one_pass(assigning, monkeypatch):
    slot = workers.Worker(1, _ReapedOnJoin(43), InputQueue(), reaping=True)
    workers.WORKERS[1] = slot
    respawned = []
    monkeypatch.setattr(lifecycle, "respawn_worker", lambda wid, **kw: respawned.append((wid, kw)))
    monkeypatch.setattr(lifecycle, "kill_worker_tree", lambda pid, **kw: setattr(slot.proc, "killed", True))
    lifecycle.finish_unready_teardown({"worker_id": 1, "worker": slot, "owner_chat_id": 0, "started": 0.0,
                                       "attempt": 1, "drive_root": str(assigning.root), "first_stop": True})
    assert respawned == [(1, {"ready_attempt": 2})]


def test_the_health_module_claims_the_readiness_bound_only_for_live_but_unready_replacements():
    """The test above is what readiness exhaustion bounds: a live-but-unready child's replacement steps to its next
    attempt. A child that dies before readiness respawns at attempt 1, so the module claims no boot-death fence."""
    from supervisor import worker_health

    doc = " ".join(worker_health.__doc__.split())
    assert "readiness exhaustion bounds only live-but-unready replacements" in doc and "boot-death" not in doc


def test_a_failed_birth_ledger_is_a_visible_warning(assigning, monkeypatch, caplog):
    from ouroboros import process_custody
    monkeypatch.setattr(lifecycle, "_record_worker_processes", _REAL_RECORD)
    monkeypatch.setattr(process_custody, "record_process",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("ledger unavailable")))
    workers.WORKERS[1] = workers.Worker(1, Process(43), InputQueue(), reaping=True, booting=True)
    with caplog.at_level("WARNING", logger=lifecycle.log.name):
        lifecycle._record_worker_processes()
    assert not workers.WORKERS[1].process_birth
    assert any(r.levelname == "WARNING" and "ledger" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("max_workers,saturated", [(3, False), (2, True), ("", False)])
def test_saturation_notice_needs_the_owners_cap_of_acting_tasks_reached(tmp_path, monkeypatch, max_workers, saturated):
    """The notice counts acting tasks against the owner's cap: a parked author waits and is not counted; with no cap
    the machine's memory is the limit and no notice claims saturation."""
    from types import SimpleNamespace

    from supervisor import events as ev_module

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(max_workers))
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {"a": {"task": {"id": "a"}}, "b": {"task": {"id": "b"}},
                   "parked": {"task": {"id": "parked"}, "owner_wait": {"state": "waiting"}}}
        WORKERS = {0: SimpleNamespace(busy_task_id="a"), 1: SimpleNamespace(busy_task_id="b"),
                   2: SimpleNamespace(busy_task_id="parked", active_capacity=False)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((text, kwargs))

        def enqueue_task(self, task):
            return dict(task)

        def persist_queue_snapshot(self, reason=""):
            pass

    child_root = str(tmp_path / "state" / "headless_tasks" / "child9" / "data")
    ev_module._handle_schedule_task({
        "type": "schedule_subagent", "task_id": "child9", "objective": "Inspect", "expected_output": "Notes",
        "context": "Parent facts", "depth": 1, "parent_task_id": "parent9", "root_task_id": "root9",
        "delegation_role": "subagent", "memory_mode": "forked", "drive_root": child_root,
        "child_drive_root": child_root, "budget_drive_root": str(tmp_path),
    }, FakeCtx())

    (text, kwargs), = sent
    assert bool(kwargs["progress_meta"].get("worker_saturation_warning")) is saturated
    assert ("limit of acting tasks is reached" in text) is saturated


def test_contended_launch_commits_stay_demand_and_retire_no_ready_worker(assigning, monkeypatch):
    """Wave-1 M2: with every launch commit contended, the pool retired 2 of 3 ready workers
    while 3 tasks pended. A real ``BEGIN IMMEDIATE`` holder makes each dispatch-mark commit
    fail with a confirmed rollback; that contention is deferred demand, never a refusal."""
    import sqlite3

    from ouroboros import store
    from ouroboros.task_results import load_task_result, write_task_result

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    for wid in (1, 2):
        workers.WORKERS[wid] = affirm_ready(workers.Worker(wid, Process(42 + wid), InputQueue()))
    for slot in workers.WORKERS.values():
        acknowledge(slot)  # every resident is quiet: a zero forecast retires two of them
    for tid in ("a", "b", "c"):  # admitted work whose launch carries the Project dispatch mark
        write_task_result(assigning.root, tid, "scheduled")
        workers.PENDING.append(task(tid, _project_scope_none=True))
    forecasts = []
    real_reconcile = lifecycle.reconcile_worker_capacity
    monkeypatch.setattr(lifecycle, "reconcile_worker_capacity",
                        lambda residual, reading=None: forecasts.append(residual()) or real_reconcile(residual, reading))
    holder = sqlite3.connect(store.store_path(assigning.root), isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    try:
        workers.assign_tasks()
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert forecasts == [3], "contended commits stay residual demand"
    assert [row["id"] for row in workers.PENDING] == ["a", "b", "c"] and not workers.RUNNING
    assert all(row.get("admitted_dispatch", "none") == "none" for row in workers.PENDING), "nothing claimed"
    assert all(load_task_result(assigning.root, tid).get("admitted_dispatch") is None for tid in "abc")
    assert queue._reap_queue.empty() and not any(slot.retiring for slot in workers.WORKERS.values())
    workers.assign_tasks()  # released: the same first dispatches, one per ready worker
    assert sorted(workers.RUNNING) == ["a", "b", "c"] and len(workers.WORKERS) == 3


@pytest.mark.parametrize("outcome", [None, False], ids=["unknown", "refused"])
def test_an_unresolved_launch_commit_stays_demand_but_a_refusal_does_not(assigning, monkeypatch, outcome):
    """An unknown commit outcome is retried by this same process next tick (its witness keeps the
    row), so it is residual demand and retires no ready worker; a refused commit is not demand."""
    from ouroboros.task_results import write_task_result
    from supervisor import task_admission
    from supervisor.worker_assignment import _COMMITTED_UNSENT

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    for wid in (1, 2):
        workers.WORKERS[wid] = affirm_ready(workers.Worker(wid, Process(42 + wid), InputQueue()))
    for slot in workers.WORKERS.values():
        acknowledge(slot)
    for tid in ("a", "b", "c"):
        write_task_result(assigning.root, tid, "scheduled")
        workers.PENDING.append(task(tid, _project_scope_none=True))
    monkeypatch.setattr(task_admission, "commit_dispatch_marks", lambda *a, **k: outcome)
    forecasts = []
    real_reconcile = lifecycle.reconcile_worker_capacity
    monkeypatch.setattr(lifecycle, "reconcile_worker_capacity",
                        lambda residual, reading=None: forecasts.append(residual()) or real_reconcile(residual, reading))
    workers.assign_tasks()
    retiring = [slot.wid for slot in workers.WORKERS.values() if slot.retiring]
    if outcome is None:
        assert sorted(_COMMITTED_UNSENT) == ["a", "b", "c"] and forecasts == [3] and retiring == []
    else:
        assert not _COMMITTED_UNSENT and forecasts == [0] and len(retiring) == 2


# ---- the published pool counters: measured from the pool, read without side effects --------------

def _published(root):
    import json
    return json.loads((root / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))


def test_a_forty_ceiling_with_fewer_slots_and_a_held_queue_publishes_what_each_count_is(
        assigning, monkeypatch, tmp_path):
    """The owner's case: a 40 ceiling, a pool that grew only to demand, pending rows that are held. No forecast is
    published: the counters say what acts, what boots, what is idle and what is held, and no reader chooses."""
    from ouroboros.context_runtime_facts import _queue_context_fact
    from ouroboros.engine_facts import engine_pool_line
    from supervisor import dispatch_choice
    from supervisor.queue_publication import _pool_counts

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(40))
    workers.PENDING.extend([task("a"), task("b"), task("c"),
                            # Use the canonical stop projection shape.  Legacy marker-only
                            # rows are intentionally not classified as held by a read.
                            task("paused", _budget_pause={"reason": "budget", "exact_continuation": True,
                                                         "status": "paused_exact_continuation"}),
                            task("held", _budget_pause={"reason": "owner", "exact_continuation": True,
                                                       "status": "paused_before_dispatch"})])
    workers.assign_tasks()  # the sole slot is the busy parent: growth buys the three runnable rows only
    assert len(assigning.created) == 3
    queue.persist_queue_snapshot("counters")
    document = _published(tmp_path)
    assert (document["max_workers"], document["worker_total"],
            document["busy_worker_count"], document["acting_worker_count"], document["parked_worker_count"],
            document["booting_count"], document["assignable_idle_workers"],
            document["pending_count"]) == (40, 4, 1, 1, 0, 3, 0, 5)
    memory = document["memory_admission"]  # the one reading that admitted the three, dated
    assert memory["read_at"] and (memory["starts_admitted"], memory["reason"]) == (39, "memory_room")
    assert "project_demand" not in document and "dispatch_refusals" not in document
    pending = [dict(row) for row in workers.PENDING]
    monkeypatch.setattr(dispatch_choice, "_collect", lambda *a, **kw: pytest.fail("a reader chose work"))
    # Every reader takes the recorded counts: none chooses, reads the store for a row or touches one.
    fact = _queue_context_fact({"budget_drive_root": str(tmp_path)})
    line = engine_pool_line(tmp_path)
    _pool_counts()
    assert [dict(row) for row in workers.PENDING] == pending
    assert all("admitted_dispatch" not in row for row in workers.PENDING)
    assert (fact["recorded_max_workers"], fact["worker_total"], fact["busy_worker_count"], fact["acting_worker_count"],
            fact["pending_count"], fact["pending_paused_or_held_count"], fact["pending_sleeping_count"],
            fact["free_worker_slots"]) == (40, 4, 1, 1, 5, 2, 0, 0)
    assert "project_demand" not in fact
    assert ("owner cap on acting tasks 40, total 4, with a task 1, acting 1, waiting on their process 0, booting 3, "
            "reaping (incl. booting) 3, assignable idle 0, pending rows 5; "
            "conversation carriers 0, outside the cap and memory admission; memory admission read ") in line


def test_a_steady_pool_keeps_its_census_and_a_real_change_moves_it(assigning, monkeypatch):
    """Nothing in the published pool facts moves on an unchanged pool, so no tick rewrites the document."""
    from supervisor.queue_publication import _pool_counts

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    workers.PENDING.append(task("held", _owner_hold={"reason": "owner"}))
    workers.assign_tasks()
    census = _pool_counts()
    for _tick in range(3):
        workers.assign_tasks()
    assert _pool_counts() == census and not assigning.created
    workers.PENDING.append(task("runnable"))
    workers.assign_tasks()
    assert _pool_counts() != census and len(assigning.created) == 1


@pytest.mark.parametrize("ending", ["lifecycle_busy", "authority_unavailable", "depth_unwritable", "disabled"])
def test_a_pass_that_ends_before_growth_grows_nothing(assigning, monkeypatch, ending):
    """Growth reads only what this pass chose: a pass that ended first starts no process (no older answer stands)."""
    from supervisor import state

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    workers.PENDING.append(task("runnable", depth="not-a-depth" if ending == "depth_unwritable" else 0))
    if ending == "lifecycle_busy":
        held, release = threading.Event(), threading.Event()

        def hold():
            with lifecycle._WORKER_LIFECYCLE_LOCK:
                held.set()
                release.wait(5)

        holder = _REAL_THREAD(target=hold)
        holder.start()
        held.wait(5)
        try:
            workers.assign_tasks()
        finally:
            release.set()
            holder.join(5)
    elif ending == "authority_unavailable":
        def unavailable(*_a, **_kw):
            raise RuntimeError("ledger unreadable")
        monkeypatch.setattr(state, "budget_remaining", unavailable)
        workers.assign_tasks()
    elif ending == "depth_unwritable":
        monkeypatch.setattr(workers, "_quarantine_invalid_pending_depths", lambda: ([], []))
        monkeypatch.setattr(workers, "_terminalize_invalid_pending_depth", lambda *_a: False)
        workers.assign_tasks()
    else:
        monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "crash_storm")
        workers.assign_tasks()
    assert not assigning.created and [row["id"] for row in workers.PENDING] == ["runnable"]
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)  # the other direction: an ordinary pass
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    workers.PENDING[0]["depth"] = 0
    workers.assign_tasks()
    assert len(assigning.created) == 1, "the next ordinary pass grows for the same row"
