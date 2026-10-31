"""Elasticity through actual assignment, readiness, snapshot and reaper consumers."""

import threading
import time
from types import SimpleNamespace

import pytest

from supervisor import queue, worker_pool_lifecycle as lifecycle, workers
from tests.test_owner_wait_pool import pool as pool, park, Process, InputQueue
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial

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


@pytest.mark.parametrize("cap", [1, 3, 10])
def test_burst_materializes_only_residual_demand_and_never_demand_plus_spare(assigning, monkeypatch, cap):
    monkeypatch.setattr(workers, "MAX_WORKERS", cap)
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
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
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
    monkeypatch.setattr(workers, "MAX_WORKERS", 10)
    monkeypatch.setattr(config, "get_max_active_subagents_per_root", lambda: 1)
    workers.PENDING.extend([task("a", project_id="p"), task("b", project_id="p"),
        task("c", delegation_role="subagent", root_task_id="owner", parent_task_id="owner", depth=1),
        task("d", delegation_role="subagent", root_task_id="owner", parent_task_id="owner", depth=1)])
    workers.assign_tasks()
    assert len(assigning.created) == 2  # one project root and one eligible child
    assert set(workers.RUNNING) == {"owner"}


@pytest.mark.parametrize("hold", ["_owner_hold", "_project_admission_restore_hold", "_budget_pause"])
def test_held_rows_do_not_grow_the_pool(assigning, monkeypatch, hold):
    monkeypatch.setattr(workers, "MAX_WORKERS", 10)
    workers.PENDING.append(task("held", **{hold: {"reason": "held"}}))
    workers.assign_tasks()
    assert not assigning.created


def test_zero_budget_resumes_owner_wait_and_keeps_parked_evolution_across_forecasts(assigning, monkeypatch):
    """At zero dispatch only owner-wait; never clean paused work, with or without a free slot."""
    from supervisor import state
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 0)
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    resume = {**assigning.meta["task"], "_owner_wait_resume": {**assigning.wait, "restart_transaction_id": "tx"}}
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    workers.PENDING.extend([task("evo", type="evolution", admitted_dispatch="none"), resume])
    for _tick in range(2):  # the second tick has no free worker: forecast only
        workers.assign_tasks()
        assert [row["id"] for row in workers.PENDING] == ["evo"] and workers.PENDING[0]["_budget_pause"]
    assert assigning.original.in_q.get_nowait()["_owner_wait_resume"] == resume["_owner_wait_resume"]
    assert set(workers.RUNNING) == {"owner"} and not assigning.created


def test_parked_authors_with_empty_queue_get_one_replacement(assigning, monkeypatch):
    monkeypatch.setattr(workers, "MAX_WORKERS", 10)
    park(assigning)
    workers.assign_tasks()
    assert len(assigning.created) == 1 and not assigning.original.active_capacity
    workers.assign_tasks()
    assert len(assigning.created) == 1  # booting reservation is already charged
    assert workers.RUNNING["owner"] is assigning.meta


def test_one_exhausted_slot_keeps_budget_but_does_not_exhaust_unused_headroom(assigning, monkeypatch):
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    workers.RUNNING.clear()
    slot = assigning.original
    slot.busy_task_id, slot.readiness_exhausted, slot.reaping = None, True, True
    slot.proc.alive = False
    assert workers.worker_pool_admission_state()["available"]
    workers.assign_tasks()
    assert workers.WORKERS[0] is slot and slot.readiness_exhausted
    assert len(assigning.created) == 1
    workers.assign_tasks()
    assert len(assigning.created) == 1


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
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
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
    assert not assigning.created and assigning.original.retiring
    assert not workers._ensure_workers_healthy_locked(queue)[1]
    assert queue._reap_queue.empty()


@pytest.mark.parametrize("reason", ["owner_stop", "managed_update", "worker_crash_storm"])
def test_elasticity_does_not_reopen_disabled_pool(assigning, monkeypatch, reason):
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", reason)
    workers.PENDING.append(task("queued"))
    workers.assign_tasks()
    assert not assigning.created


def test_pending_storm_stop_keeps_membership_and_still_runs(assigning, monkeypatch):
    from supervisor.task_reaper import _stop_crashed_worker_pool
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    assigning.original.proc.alive = False
    monkeypatch.setattr(workers, "_worker_crash_storm_detected", lambda **kw: True)
    workers._ensure_workers_healthy_locked(queue)
    queue._reap_queue.get_nowait()  # empty dead slot, no terminal custody
    storm = queue._reap_queue.get_nowait()
    workers.PENDING.append(task("queued"))
    workers.assign_tasks()
    assert not assigning.created
    stopped = []
    monkeypatch.setattr(workers, "kill_workers", lambda **kw: stopped.append(kw) or True)
    _stop_crashed_worker_pool(storm)
    assert stopped == [{"disable_reason": "worker_crash_storm"}]


@pytest.mark.parametrize("prior", ["", "worker_crash_storm"])
def test_failed_disable_fence_restores_the_prior_admission_state(assigning, monkeypatch, prior):
    """An enabled pool reopens after an aborted shutdown; a storm latch stays closed."""
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", prior)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kw: False)
    assert workers.kill_workers(disable_reason=prior or "managed_update") is False
    assert workers._WORKER_POOL_DISABLED_REASON == prior
    assert workers.WORKERS == {0: assigning.original} and assigning.original.proc.is_alive()


def test_failed_storm_snapshot_keeps_the_latch_so_growth_cannot_void_the_stop(assigning, monkeypatch):
    """Reviewer counterexample: real health -> failed snapshot -> growth -> storm no-op."""
    from supervisor.task_reaper import TerminalFileRecoveryPending, _stop_crashed_worker_pool
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    assigning.original.proc.alive = False
    monkeypatch.setattr(workers, "_worker_crash_storm_detected", lambda **kw: True)
    durable = [False]
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kw: durable[0])
    workers._ensure_workers_healthy_locked(queue)
    queue._reap_queue.get_nowait()  # the dead idle slot owes no terminal
    storm = queue._reap_queue.get_nowait()
    with pytest.raises(TerminalFileRecoveryPending):
        _stop_crashed_worker_pool(storm)
    assert workers._WORKER_POOL_DISABLED_REASON == "worker_crash_storm"
    workers.PENDING.append(task("queued"))
    workers.assign_tasks()
    assert not assigning.created and set(workers.WORKERS) == {0}
    durable[0] = True
    stopped = []
    monkeypatch.setattr(workers, "kill_workers", lambda **kw: stopped.append(kw) or True)
    _stop_crashed_worker_pool(storm)
    assert stopped == [{"disable_reason": "worker_crash_storm"}]


def test_active_work_keeps_existing_spare_without_buying_one(assigning, monkeypatch):
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    spare = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    workers.WORKERS[1] = spare
    acknowledge(spare)
    workers.assign_tasks()
    assert not spare.retiring and not assigning.created and queue._reap_queue.empty()


def test_full_cap_resume_probes_all_idle_residents_and_uses_one_completed_owner(assigning, monkeypatch):
    from supervisor import worker_owner_wait
    monkeypatch.setattr(workers, "MAX_WORKERS", 2)
    park(assigning)
    late = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    quiet = affirm_ready(workers.Worker(2, Process(44), InputQueue()))
    workers.WORKERS.update({1: late, 2: quiet})
    worker_owner_wait.handle_owner_wait({**assigning.event, "phase": "resume"}, workers)
    worker_owner_wait.maintain_owner_wait_capacity()
    assert late.quiescence_token and quiet.quiescence_token
    assert not late.retiring and not quiet.retiring and not assigning.original.active_capacity
    acknowledge(quiet)  # the first resident still owns late review work
    worker_owner_wait.maintain_owner_wait_capacity()
    assert quiet.retiring and late.proc.is_alive() and not late.retiring
    lifecycle.reap_idle_worker(queue._reap_queue.get_nowait())
    worker_owner_wait.maintain_owner_wait_capacity()
    assert assigning.original.active_capacity and late.proc.is_alive()


def _resume_with_fresh_work(assigning):
    from supervisor import worker_owner_wait
    park(assigning)
    workers.assign_tasks()  # one replacement for the parked author
    replacement = ready(workers.WORKERS[1])
    worker_owner_wait.handle_owner_wait({**assigning.event, "phase": "resume"}, workers)
    workers.PENDING.append(task("fresh"))
    return replacement


def test_cap_full_warm_resume_outranks_fresh_work_then_the_queue_runs(assigning):
    """Reviewer counterexample (MAX 1): fresh work took every idle opportunity, forever."""
    import json
    replacement = _resume_with_fresh_work(assigning)
    workers.assign_tasks()
    probe = replacement.in_q.get_nowait()
    assert probe["type"] == "worker_quiescence" and replacement.in_q.empty()
    assert replacement.resume_hold == "owner" and "fresh" not in workers.RUNNING
    queue.persist_queue_snapshot()
    assert json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["assignable_idle_workers"] == 0
    lifecycle.handle_worker_quiescent({**probe, "worker_id": 1, "pid": replacement.proc.pid,
                                      "process_birth": replacement.process_birth}, workers)
    workers.assign_tasks()
    assert replacement.retiring and "fresh" not in workers.RUNNING
    lifecycle.reap_idle_worker(queue._reap_queue.get_nowait())
    workers.assign_tasks()
    assert assigning.original.active_capacity and workers.WORKERS == {0: assigning.original}
    assert assigning.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert len(assigning.created) == 1, "growth never spends the capacity the Resume took back"
    workers.RUNNING.pop("owner")  # the resumed author completes; its slot is free again
    assigning.original.busy_task_id = None
    workers.assign_tasks()
    assert workers.RUNNING["fresh"]["worker_id"] == 0


@pytest.mark.parametrize("revocation", ["owner_stop", "task_ended"])
def test_resume_hold_lapses_with_its_authorization(assigning, revocation):
    from ouroboros.cancel_intents import request_cancel
    replacement = _resume_with_fresh_work(assigning)
    workers.assign_tasks()
    assert replacement.resume_hold == "owner" and "fresh" not in workers.RUNNING
    if revocation == "owner_stop":
        request_cancel(assigning.root, "owner", reason="owner stop", source="test")
    else:
        workers.RUNNING.pop("owner")
    workers.assign_tasks()
    assert not replacement.resume_hold and workers.RUNNING["fresh"]["worker_id"] == 1


def test_one_resume_retires_one_slot_while_its_reaper_runs(assigning, monkeypatch):
    from supervisor import worker_owner_wait
    monkeypatch.setattr(workers, "MAX_WORKERS", 2)
    park(assigning)
    first = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    second = affirm_ready(workers.Worker(2, Process(44), InputQueue()))
    workers.WORKERS.update({1: first, 2: second})
    worker_owner_wait.handle_owner_wait({**assigning.event, "phase": "resume"}, workers)
    worker_owner_wait.maintain_owner_wait_capacity()
    acknowledge(first)
    acknowledge(second)
    worker_owner_wait.maintain_owner_wait_capacity()
    assert first.retiring and not second.retiring
    worker_owner_wait.maintain_owner_wait_capacity()  # the reaper has not confirmed death yet
    assert not second.retiring, "a retiring slot already pays for this Resume"
    lifecycle.reap_idle_worker(queue._reap_queue.get_nowait())
    worker_owner_wait.maintain_owner_wait_capacity()
    assert assigning.original.active_capacity and second.proc.is_alive() and not second.retiring
    assert queue._reap_queue.empty()


def test_growth_never_spends_capacity_a_requested_resume_takes_back(assigning):
    from supervisor import worker_owner_wait
    replacement = _resume_with_fresh_work(assigning)
    worker_owner_wait.maintain_owner_wait_capacity()  # this tick's owner authorizes the Resume
    # The reaper confirms the retired slot dead after this tick's maintenance ran.
    replacement.proc.alive = False
    assert lifecycle.retire_worker(1, replacement)
    lifecycle.reconcile_worker_capacity(lambda: 1)
    assert len(assigning.created) == 1 and set(workers.WORKERS) == {0}
    workers.assign_tasks()
    assert assigning.original.active_capacity and "fresh" not in workers.RUNNING


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
    assert len(assigning.created) == 1, "the lent slot stays lent while its Resume is not authorized"
    ready(workers.WORKERS[1])
    workers.assign_tasks()
    assert workers.RUNNING["independent"]["worker_id"] == 1


@pytest.mark.parametrize("allowed", [False, True])
def test_writer_gate_grows_only_for_its_authorized_assisted_task(assigning, monkeypatch, allowed):
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "managed_update")
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda row: allowed and row["id"] == "resolver")
    workers.PENDING.extend([task("ordinary"), task("resolver")])
    workers.assign_tasks()
    assert len(assigning.created) == int(allowed)
    assert set(workers.RUNNING) == {"owner"}


def test_unknown_timeout_reaper_is_charged_but_is_not_future_ready_capacity(assigning, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor import task_reaper
    monkeypatch.setattr(workers, "MAX_WORKERS", 4)
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


def test_exhaustion_spends_each_materialized_reservation_only_once(assigning, monkeypatch):
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    workers.RUNNING.clear()
    for wid in range(3):
        slot = workers.WORKERS[wid]
        slot.busy_task_id, slot.reaping, slot.readiness_exhausted = None, True, True
        slot.booting, slot.proc.alive = False, False
        if wid < 2:
            workers.assign_tasks()
            assert len(assigning.created) == wid + 1
    assert not workers.worker_pool_admission_state()["available"]
    stopped = []
    monkeypatch.setattr(workers, "kill_workers", lambda **kw: stopped.append(kw) or True)
    workers.assign_tasks()
    assert len(assigning.created) == 2 and stopped[0]["disable_reason"] == "worker_readiness_exhausted"


def _timed_out_dead_slot(assigning):
    slot = assigning.original
    workers.RUNNING.clear()  # a timeout moves custody from RUNNING to its captured reaper job
    slot.busy_task_id, slot.reaping, slot.proc.alive = None, True, False
    return slot


def test_captured_timeout_after_a_storm_latch_starts_no_child_and_the_stop_keeps_its_membership(
        assigning, monkeypatch):
    """Reviewer counterexample (F1): the replacement started after the latch, then its rejected child lost custody."""
    from supervisor import task_reaper
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    timed_out = _timed_out_dead_slot(assigning)
    crashed = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    crashed.proc.alive = False
    workers.WORKERS[1] = crashed
    monkeypatch.setattr(workers, "_worker_crash_storm_detected", lambda **kw: True)
    workers._ensure_workers_healthy_locked(queue)  # the real owner latches while the timeout is off-loop
    queue._reap_queue.get_nowait()  # the crashed idle slot owes no terminal
    storm = queue._reap_queue.get_nowait()
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=timed_out)
    assert assigning.created == [] and workers.WORKERS == {0: timed_out, 1: crashed}
    stopped = []
    monkeypatch.setattr(workers, "kill_workers", lambda **kw: stopped.append(kw) or True)
    task_reaper._stop_crashed_worker_pool(storm)
    assert stopped == [{"disable_reason": "worker_crash_storm"}]


def test_enabled_pool_still_replaces_a_captured_dead_slot(assigning):
    from supervisor import task_reaper
    timed_out = _timed_out_dead_slot(assigning)
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=timed_out)
    fresh = workers.WORKERS[0]
    assert fresh is not timed_out and [fresh.proc] == assigning.created
    assert fresh.booting and fresh.reaping and fresh.active_capacity


@pytest.mark.parametrize("reason", ["worker_crash_storm", "worker_readiness_exhausted"])
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


@pytest.mark.parametrize("reason", ["worker_crash_storm", "managed_update", "worker_readiness_exhausted"])
def test_a_fence_leaves_a_dead_lent_slot_to_its_stop_owner(assigning, monkeypatch, reason):
    park(assigning)
    assigning.original.proc.alive = False
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", reason)
    assert not workers.respawn_worker(0)
    assert workers.WORKERS == {0: assigning.original} and not assigning.created


def test_a_lent_timeout_reaped_ahead_of_the_storm_stop_cannot_void_it(assigning, monkeypatch):
    """Reviewer counterexample: the timeout job ahead of the storm job forgot its dead lent slot,
    so the captured membership never matched and the latch outlived a stop that never ran."""
    import json

    from supervisor import task_reaper
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    timed_out = _timed_out_dead_slot(assigning)
    timed_out.active_capacity = False  # an owner-wait park lent this slot
    crashed = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    crashed.proc.alive = False
    workers.WORKERS[1] = crashed
    monkeypatch.setattr(workers, "_worker_crash_storm_detected", lambda **kw: True)
    workers._ensure_workers_healthy_locked(queue)
    queue._reap_queue.get_nowait()  # the crashed idle slot owes no terminal
    storm = queue._reap_queue.get_nowait()
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=timed_out)
    assert workers.WORKERS == {0: timed_out, 1: crashed} and not assigning.created
    task_reaper._stop_crashed_worker_pool(storm)  # the real stop, not a stub
    assert workers.WORKERS == {} and workers._WORKER_POOL_DISABLED_REASON == "worker_crash_storm"
    assert json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["worker_pool_disabled_reason"] == "worker_crash_storm"


@pytest.mark.parametrize("failed_put", ["worker_crash_storm", "confirmed_dead_worker"])
def test_an_unsent_storm_stop_waits_with_the_reaper_retries_so_its_latch_keeps_an_owner(
        assigning, monkeypatch, failed_put):
    from supervisor import task_reaper
    monkeypatch.setattr(task_reaper, "_deferred_reap_jobs", [])
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    second = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    workers.WORKERS[1] = second
    for slot in workers.WORKERS.values():
        slot.proc.alive = False
    monkeypatch.setattr(workers, "_worker_crash_storm_detected", lambda **kw: True)
    put = queue._reap_queue.put

    def refuse(job, *args, **kwargs):
        if job["kind"] == failed_put:
            raise OSError("reaper queue unavailable")
        put(job, *args, **kwargs)

    monkeypatch.setattr(queue._reap_queue, "put", refuse)
    with pytest.raises(OSError):
        workers._ensure_workers_healthy_locked(queue)
    assert workers._WORKER_POOL_DISABLED_REASON == "worker_crash_storm"
    assert [job["kind"] for job in task_reaper._deferred_reap_jobs] == ["worker_crash_storm"]
    handed = {job["worker_id"] for job in list(queue._reap_queue.queue)}
    assert all(slot.reaping == (wid in handed) for wid, slot in workers.WORKERS.items()), \
        "a slot whose job was never handed off stays visible to the next health tick"
    task_reaper._retry_deferred_reap_jobs()  # the next tick's hand-off may fail again
    assert len(task_reaper._deferred_reap_jobs) == int(failed_put == "worker_crash_storm")
    monkeypatch.setattr(queue._reap_queue, "put", put)
    task_reaper._retry_deferred_reap_jobs()
    storm, = [job for job in queue._reap_queue.queue if job["kind"] == "worker_crash_storm"]
    assert task_reaper._deferred_reap_jobs == []
    task_reaper._stop_crashed_worker_pool(storm)  # dead idle slots owe no terminal
    assert workers.WORKERS == {} and workers._WORKER_POOL_DISABLED_REASON == "worker_crash_storm"



def test_a_storm_latch_keeps_its_captured_membership_through_every_own_door(assigning, monkeypatch):
    """Under the latch only its stop changes membership: no own caller forgets or starts a slot."""
    from supervisor import task_reaper, worker_owner_wait
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    workers.RUNNING.clear()
    lent = assigning.original
    lent.busy_task_id, lent.active_capacity = None, False  # a lent slot whose author has ended
    retiring = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    workers.WORKERS[1] = retiring
    acknowledge(retiring)
    assert lifecycle.retire_quiescent_worker(retiring)
    job = queue._reap_queue.get_nowait()
    lent.proc.alive = retiring.proc.alive = False
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "worker_crash_storm")
    captured = dict(workers.WORKERS)
    lifecycle.reap_idle_worker(job)
    worker_owner_wait.maintain_owner_wait_capacity()
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=lent)
    assert not lifecycle.retire_worker(0, lent) and not lifecycle.retire_or_replace_worker(1, retiring)
    workers.assign_tasks()
    assert workers.WORKERS.keys() == captured.keys() and not assigning.created
    assert all(workers.WORKERS[wid] is slot for wid, slot in captured.items())
    stopped = []
    monkeypatch.setattr(workers, "kill_workers", lambda **kw: stopped.append(kw) or True)
    task_reaper._stop_crashed_worker_pool({"kind": "worker_crash_storm", "workers": tuple(captured.items()),
                                           "drive_root": str(assigning.root)})
    assert stopped == [{"disable_reason": "worker_crash_storm"}]

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
    monkeypatch.setattr(workers, "MAX_WORKERS", 10)
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
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
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
    workers.assign_tasks()  # the Resume still goes first
    assert replacement.in_q.get_nowait()["type"] == "worker_quiescence"
    assert replacement.resume_hold == "owner" and "fresh" not in workers.RUNNING


def _resume_reserving_the_last_spare(assigning):
    """MAX 1: the warm Resume reserved its replacement's retirement; its reaper has not run."""
    from supervisor import worker_owner_wait
    park(assigning)
    workers.assign_tasks()  # one replacement for the parked author
    replacement = ready(workers.WORKERS[1])
    workers.PENDING.append(task("next-independent"))  # accepted while the pool is available
    worker_owner_wait.handle_owner_wait({**assigning.event, "phase": "resume"}, workers)
    acknowledge(replacement)
    worker_owner_wait.maintain_owner_wait_capacity()
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


@pytest.mark.parametrize("order", ["stop_first", "reap_first", "unknown_release", "exited_before_reap"])
def test_stopping_a_resuming_author_keeps_the_enabled_pool_dispatching(assigning, monkeypatch, order):
    """Reviewer counterexample (MAX 1): Stop landed between the Resume's reserved retirement and
    its reap, the Stop's respawn counted the retiring slot and forgot the author, and the reap
    then emptied an enabled pool that growth can never refill."""
    from supervisor.task_reaper import TerminalFileRecoveryPending
    replacement, job = _resume_reserving_the_last_spare(assigning)
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
    workers.assign_tasks()  # the lent slot's replacement
    replacement = ready(workers.WORKERS[1])
    if control == "stop":
        _stop_parked_author(assigning)
        assert workers.WORKERS == {1: replacement}, "an enabled cap-full pool forgets the dead lent slot"
    else:
        owner_pause.release_fence(assigning.root, "owner", reason="owner_resume")
        worker_owner_wait.handle_owner_wait(
            {**event, "phase": "resume", "resume_reason": "control:owner_resume"}, workers)
        acknowledge(replacement)
        worker_owner_wait.maintain_owner_wait_capacity()
        lifecycle.reap_idle_worker(queue._reap_queue.get_nowait())  # not the last member: retired
        worker_owner_wait.maintain_owner_wait_capacity()
        assert assigning.original.in_q.get_nowait()["phase"] == "resume_granted"
        assert workers.WORKERS == {0: assigning.original} and assigning.original.active_capacity
        rows = [json.loads(line) for line in (assigning.root / "logs" / "supervisor.jsonl").read_text().splitlines()]
        assert [row["replaced"] for row in rows if row["type"] == "worker_retired"] == [False]
        workers.RUNNING.pop("owner")  # the resumed author completes
        assigning.original.busy_task_id = None
    workers.PENDING.append(task("next-independent"))
    workers.assign_tasks()
    assert list(workers.RUNNING) == ["next-independent"] and len(assigning.created) == 1


@pytest.mark.parametrize("reason", ["worker_crash_storm", "managed_update"])
def test_a_fence_never_refills_the_last_retiring_member(assigning, monkeypatch, reason):
    replacement, job = _resume_reserving_the_last_spare(assigning)
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
    worker_owner_wait.maintain_owner_wait_capacity()
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
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
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
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
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


@pytest.mark.parametrize("max_workers,saturated", [(3, False), (2, True)])
def test_saturation_notice_needs_every_worker_busy_at_the_configured_cap(tmp_path, monkeypatch, max_workers, saturated):
    """Below the cap an all-busy elastic pool grows for the queued child; a parked author lent its slot."""
    from types import SimpleNamespace

    from supervisor import events as ev_module

    monkeypatch.setattr(workers, "MAX_WORKERS", max_workers)
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
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
    assert ("all workers are currently busy" in text) is saturated


def test_contended_launch_commits_stay_demand_and_retire_no_ready_worker(assigning, monkeypatch):
    """Wave-1 M2: with every launch commit contended, the pool retired 2 of 3 ready workers
    while 3 tasks pended. A real ``BEGIN IMMEDIATE`` holder makes each dispatch-mark commit
    fail with a confirmed rollback; that contention is deferred demand, never a refusal."""
    import sqlite3

    from ouroboros import store
    from ouroboros.task_results import load_task_result, write_task_result

    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
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
                        lambda residual: forecasts.append(residual()) or real_reconcile(residual))
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

    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
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
                        lambda residual: forecasts.append(residual()) or real_reconcile(residual))
    workers.assign_tasks()
    retiring = [slot.wid for slot in workers.WORKERS.values() if slot.retiring]
    if outcome is None:
        assert sorted(_COMMITTED_UNSENT) == ["a", "b", "c"] and forecasts == [3] and retiring == []
    else:
        assert not _COMMITTED_UNSENT and forecasts == [0] and len(retiring) == 2
