"""Required waiting lends capacity without replaying or detaching its task."""

import datetime as dt
import json
import queue as stdqueue
import time
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import store_actor_source_bytes
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, worker_owner_wait, worker_pool_lifecycle, workers
from tests.worker_pool_support import affirm_ready


class Process:
    def __init__(self, pid):
        self.pid, self.alive, self.exitcode = pid, True, None

    def is_alive(self):
        return self.alive

    def start(self):
        assert not queue._queue_lock._is_owned(), "process.start inherited the queue lock"

    def join(self, timeout=None):
        pass


class InputQueue(stdqueue.Queue):
    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        pass


@pytest.fixture
def pool(tmp_path, monkeypatch):
    running, pending, slots = {}, [], {}
    for module in (workers, queue):
        monkeypatch.setattr(module, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(module, "RUNNING", running)
        monkeypatch.setattr(module, "PENDING", pending)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(queue, "_reap_queue", stdqueue.Queue())
    monkeypatch.setattr(workers, "WORKERS", slots)
    monkeypatch.setattr(workers, "MAX_WORKERS", 1)
    monkeypatch.setattr(workers, "CRASH_TS", [])
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda task: True)
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(worker_pool_lifecycle, "_record_worker_processes", lambda: None)
    monkeypatch.setattr(worker_pool_lifecycle.threading, "Thread",
                        lambda **kw: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(workers, "get_event_q", InputQueue)
    created = []

    def make_process(**kwargs):
        process = Process(1000 + len(created))
        created.append(process)
        return process

    monkeypatch.setattr(workers, "_get_ctx", lambda: SimpleNamespace(Queue=InputQueue, Process=make_process))

    def kill(pid, **kwargs):
        for worker in slots.values():
            if worker.proc.pid == pid:
                worker.proc.alive = False
                worker.proc.exitcode = 0

    monkeypatch.setattr(worker_pool_lifecycle, "kill_worker_tree", kill)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "")
    root_task = {"id": "owner", "type": "task", "chat_id": 1, "_attempt": 3, "text": "work"}
    original = workers.Worker(0, Process(42), InputQueue(), busy_task_id="owner")
    affirm_ready(original)
    slots[0] = original
    started = time.time() - 4000
    running["owner"] = {"task": root_task, "worker_id": 0, "attempt": 3,
                        "started_at": started, "last_heartbeat_at": time.time()}
    write_task_result(tmp_path, "owner", "running", result="prepared", total_rounds=7)
    source = store_actor_source_bytes(tmp_path, "owner", category="context_checkpoints",
                                     source_id="wait", data=b'{"effects":"done"}', extension="json")
    wait = {"wait_id": "wait", "task_attempt": 3, "reason": "owner", "quiz_id": "quiz", "source_ref": source,
            "execution_drive_root": str(tmp_path), "started_at": started}
    event = {"type": "owner_wait", "task_id": "owner", "worker_id": 0, "pid": 42,
             "task_attempt": 3, "wait_id": "wait", "phase": "park", "checkpoint": wait}
    return SimpleNamespace(root=tmp_path, original=original, meta=running["owner"],
                           event=event, wait=wait, created=created)


def maintain_capacity():
    worker_owner_wait.maintain_owner_wait_capacity()
    worker_pool_lifecycle.reconcile_worker_capacity(lambda: 0)


def quiesce_and_reap(worker):
    command = worker.in_q.get_nowait()
    assert command["type"] == "worker_quiescence"
    worker_pool_lifecycle.handle_worker_quiescent({
        "worker_id": worker.wid, "pid": worker.proc.pid,
        "process_birth": worker.process_birth, "token": command["token"]}, workers)
    worker_owner_wait.maintain_owner_wait_capacity()
    assert worker.retiring and worker.active_capacity
    worker_pool_lifecycle.reap_idle_worker(queue._reap_queue.get_nowait())
    worker_owner_wait.maintain_owner_wait_capacity()


def park(pool):
    worker_owner_wait.handle_owner_wait(pool.event, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"


def test_park_is_durable_before_ack_and_preserves_running_ownership(pool):
    park(pool)
    saved = load_task_result(pool.root, "owner")
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert saved["status"] == "running" and saved["total_rounds"] == 7
    assert saved["owner_wait"] == {**pool.wait, "state": "waiting"}
    assert workers.RUNNING["owner"] is pool.meta
    assert workers.WORKERS[0] is pool.original and pool.original.busy_task_id == "owner"
    assert snap["running"][0]["owner_wait"] == saved["owner_wait"]
    assert snap["active_worker_count"] == 0 and snap["parked_worker_count"] == 1
    assert snap["running_count"] == 1 and not workers.PENDING


def test_snapshot_failure_never_lends_capacity_or_confirms_park(pool, monkeypatch):
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kw: False)
    worker_owner_wait.handle_owner_wait(pool.event, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "refused"
    assert pool.original.active_capacity and not pool.created


def test_replacement_readiness_and_same_stack_resume(pool):
    park(pool)
    maintain_capacity()
    replacement = workers.WORKERS[1]
    assert replacement.reaping and replacement.active_capacity
    assert pool.original is workers.WORKERS[0] and len(pool.created) == 1
    replacement.reaping = False
    affirm_ready(replacement)
    replacement.busy_task_id = "independent"
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    maintain_capacity()
    assert pool.original.in_q.empty() and not pool.original.active_capacity
    replacement.busy_task_id = None
    maintain_capacity()
    assert replacement.proc.is_alive() and not pool.original.active_capacity
    quiesce_and_reap(replacement)
    assert set(workers.WORKERS) == {0} and not replacement.proc.is_alive()
    grant = pool.original.in_q.get_nowait()
    assert grant == {"type": "owner_wait", "task_id": "owner", "task_attempt": 3,
                     "wait_id": "wait", "phase": "resume_granted"}
    assert pool.original.active_capacity and workers.RUNNING["owner"] is pool.meta
    assert pool.meta["started_at"] == pool.wait["started_at"] and pool.meta["attempt"] == 3
    assert load_task_result(pool.root, "owner")["owner_wait"]["state"] == "resumed"
    assert not workers.PENDING
    # Late duplicate transport cannot re-park an already consumed wait.
    worker_owner_wait.handle_owner_wait(pool.event, workers)
    assert pool.original.active_capacity and pool.original.in_q.empty()


@pytest.mark.parametrize("field,value", [("pid", 43), ("task_attempt", 2), ("worker_id", 9)])
def test_foreign_or_stale_event_does_not_park(pool, field, value):
    worker_owner_wait.handle_owner_wait({**pool.event, field: value}, workers)
    assert pool.original.active_capacity and pool.original.in_q.empty()
    assert not load_task_result(pool.root, "owner").get("owner_wait")


def test_dead_parked_worker_is_removed_without_respawn(pool):
    park(pool)
    maintain_capacity()
    replacement = workers.WORKERS[1]
    pool.original.proc.alive = False
    assert workers.respawn_worker(0)
    assert workers.WORKERS == {1: replacement} and len(pool.created) == 1


@pytest.mark.parametrize("hard_axis", [None, "deadline", "absolute_ceiling"])
def test_owner_wait_spares_only_idle_timeout(pool, monkeypatch, hard_axis):
    park(pool)
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 10)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 1000 if hard_axis == "absolute_ceiling" else 10000)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    reaps = stdqueue.Queue()
    monkeypatch.setattr(queue, "_reap_queue", reaps)
    if hard_axis == "deadline":
        pool.meta["task"]["deadline_at"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()
    queue._enforce_task_timeouts_locked(workers, time.time(), 0, {})
    if hard_axis:
        job = reaps.get_nowait()
        assert job["terminal_reason"] == hard_axis and not {"will_retry", "retry_task_id"} & set(job)
    else:
        assert reaps.empty() and "owner" in workers.RUNNING


def test_resume_snapshot_failure_keeps_original_parked(pool, monkeypatch):
    park(pool)
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kw: False)
    maintain_capacity()
    assert not pool.original.active_capacity and pool.original.in_q.empty()
    assert load_task_result(pool.root, "owner")["owner_wait"]["state"] == "waiting"


def test_ordinary_pool_does_not_grow_to_configured_max(pool, monkeypatch):
    monkeypatch.setattr(workers, "MAX_WORKERS", 10)
    maintain_capacity()
    assert set(workers.WORKERS) == {0} and not pool.created


def exhausted_replacement(pool):
    park(pool)
    maintain_capacity()
    replacement = workers.WORKERS[1]
    replacement.proc.alive = False
    replacement.readiness_exhausted = True
    return replacement


def test_same_stack_reclaims_exhausted_replacement_without_new_spawn(pool):
    replacement = exhausted_replacement(pool)
    maintain_capacity()
    assert workers.worker_pool_admission_state()["available"]
    assert len(pool.created) == 1 and replacement.active_capacity
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    maintain_capacity()
    assert workers.WORKERS == {0: pool.original}
    assert pool.original.active_capacity and len(pool.created) == 1
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert workers.RUNNING["owner"] is pool.meta and pool.meta["attempt"] == 3
    assert replacement.in_q.closed


@pytest.mark.parametrize("failure", ["state", "snapshot", "command"])
def test_exhausted_reservation_and_original_rollback_together(pool, monkeypatch, failure):
    replacement = exhausted_replacement(pool)
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    if failure == "state":
        original = worker_owner_wait.set_owner_wait
        calls = []

        def fail_first(*a, **k):
            calls.append(True)
            if len(calls) == 1:
                raise OSError("state unavailable")
            return original(*a, **k)

        monkeypatch.setattr(worker_owner_wait, "set_owner_wait", fail_first)
    elif failure == "snapshot":
        monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **k: False)
    else:
        def fail_command(*a, **k):
            raise OSError("command unavailable")
        monkeypatch.setattr(worker_owner_wait, "_command", fail_command)
    maintain_capacity()
    assert not pool.original.active_capacity and replacement.active_capacity
    assert len(pool.created) == 1 and workers.WORKERS[1] is replacement
    assert pool.original.in_q.empty() and pool.meta["owner_wait"]["state"] == "waiting"
    assert load_task_result(pool.root, "owner")["owner_wait"]["state"] == "waiting"


def test_wait_lends_capacity_to_real_assignment(pool, monkeypatch):
    from supervisor import evolution_lifecycle, state

    monkeypatch.setattr(state, "budget_remaining", lambda *args, **kwargs: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    workers.PENDING.append({"id": "independent", "type": "task", "chat_id": 1, "text": "other work"})
    park(pool)
    workers.assign_tasks()
    assert "independent" not in workers.RUNNING  # New slot still awaits readiness.
    replacement = workers.WORKERS[1]
    replacement.reaping = False
    affirm_ready(replacement)
    workers.assign_tasks()
    assert replacement.in_q.get_nowait()["id"] == "independent"
    assert workers.RUNNING["independent"]["worker_id"] == 1
    assert workers.RUNNING["owner"] is pool.meta
    assert pool.original.in_q.empty() and not workers.PENDING


@pytest.mark.parametrize("policy,granted", [("immediate", False), ("finalize_then_cancel", True)])
def test_stop_policy_keeps_its_authority_at_resume(pool, monkeypatch, policy, granted):
    from ouroboros import cancel_intents

    park(pool)
    maintain_capacity()
    replacement = workers.WORKERS[1]
    replacement.reaping = False
    affirm_ready(replacement)
    monkeypatch.setattr(cancel_intents, "active_intents",
                        lambda *a, **k: {"owner": {"stop_policy": policy}})
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    maintain_capacity()
    if granted:
        quiesce_and_reap(replacement)
    assert pool.original.active_capacity is granted
    assert pool.original.in_q.empty() is not granted
    if not granted:
        assert replacement.proc.is_alive()  # A hard Stop cannot churn idle replacements.


def test_parked_crash_terminalizes_without_replaying_completed_effects(pool, monkeypatch):
    from ouroboros import delegate_recovery

    park(pool)
    pool.original.proc.alive, pool.original.proc.exitcode = False, 1
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda pid: None)
    monkeypatch.setattr(delegate_recovery, "reconcile_unrecoverable_task", lambda *a: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *a, **k: {
        "cost_accounting_status": "available", "cost_final": True, "total_rounds": 7,
        "accounted_upper_bound_usd": 2.0,
    })
    respawn_ids, disabled = workers._ensure_workers_healthy_locked(queue)
    assert not disabled and respawn_ids == []
    from supervisor.worker_health import recover_confirmed_dead_worker

    recover_confirmed_dead_worker(queue._reap_queue.get_nowait())
    saved = load_task_result(pool.root, "owner")
    assert saved["status"] == "failed" and saved["reason_code"] == "worker_crash_owner_wait"
    assert saved["owner_wait"]["source_ref"] == pool.wait["source_ref"]
    assert saved["total_rounds"] == 7 and not workers.PENDING
    assert len(workers.WORKERS) == 1 and workers.WORKERS[0].active_capacity
    assert workers.WORKERS[0] is not pool.original


def test_stale_snapshot_keeps_only_verified_planned_owner_handoff(pool, monkeypatch):
    from ouroboros import owner_wait

    workers.RUNNING.clear()
    selected = {"id": "resume", "type": "task", "chat_id": 1, "text": "resume",
                "_attempt": 3, "_owner_wait_resume": {**pool.wait, "restart_transaction_id": "tx"}}
    workers.PENDING.extend([selected, {"id": "old", "chat_id": 1, "type": "task", "text": "old"}])
    assert queue.persist_queue_snapshot()
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert snap["pending"][0]["task"]["_owner_wait_resume"] == selected["_owner_wait_resume"]
    stamp_path = queue.QUEUE_SNAPSHOT_PATH.with_name("queue.alive")
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    stamp["ts"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
    workers.PENDING.clear()
    monkeypatch.setattr(owner_wait, "restore_owner_wait_allowed", lambda root, task: task["id"] == "resume")
    monkeypatch.setattr(queue, "enqueue_task", lambda task, **kwargs: workers.PENDING.append(task))
    assert queue.restore_pending_from_snapshot(max_age_sec=900) == 1
    assert [task["id"] for task in workers.PENDING] == ["resume"]
    assert workers.PENDING[0]["_attempt"] == 3


def test_cold_assignment_restores_old_clock_without_budget_replay_classification(pool, monkeypatch):
    from supervisor import evolution_lifecycle, state

    monkeypatch.setattr(state, "budget_remaining", lambda *args, **kwargs: 0)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    task = {**pool.meta["task"], "_owner_wait_resume": {**pool.wait, "restart_transaction_id": "tx"}}
    workers.RUNNING.clear()
    pool.original.busy_task_id = None
    workers.PENDING.append(task)
    workers.assign_tasks()
    sent = pool.original.in_q.get_nowait()
    assert sent["_attempt"] == 3 and sent["_owner_wait_resume"] == task["_owner_wait_resume"]
    assert workers.RUNNING["owner"]["started_at"] == pool.wait["started_at"]
    assert load_task_result(pool.root, "owner")["status"] == "running"
    assert load_task_result(pool.root, "owner")["total_rounds"] == 7


@pytest.mark.parametrize("replacement_exists", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_stopped_last_sleeper_preserves_enabled_capacity_only(pool, monkeypatch, replacement_exists, disabled):
    park(pool)
    if replacement_exists:
        maintain_capacity()
    if disabled:
        monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "worker_crash_storm")
    workers.RUNNING.pop("owner")
    pool.original.busy_task_id = None
    pool.original.proc.alive = False  # The Stop/reaper owner already proved death.
    assert workers.respawn_worker(0) is not disabled
    active = [w for w in workers.WORKERS.values() if w.active_capacity]
    assert len(active) == int(replacement_exists or not disabled)
    # A fence's stop owner tears down the membership it captured, this dead slot included.
    assert (pool.original in workers.WORKERS.values()) is disabled
    assert all(w.reaping for w in active)  # A replacement still owes readiness.


def test_live_sleeper_never_loses_physical_ownership_to_respawn(pool):
    park(pool)
    assert not workers.respawn_worker(0)
    assert workers.WORKERS == {0: pool.original} and not pool.created
    assert workers.RUNNING["owner"] is pool.meta


def test_vacancy_spawn_failure_keeps_the_dead_slot_for_existing_recovery(pool, monkeypatch):
    park(pool)
    pool.original.proc.alive = False
    original_start = Process.start

    def fail_start(self):
        raise OSError("spawn unavailable")

    monkeypatch.setattr(Process, "start", fail_start)
    with pytest.raises(OSError, match="spawn unavailable"):
        workers.respawn_worker(0)
    assert workers.WORKERS[0] is pool.original
    monkeypatch.setattr(Process, "start", original_start)
    assert workers.respawn_worker(0)
    assert workers.WORKERS[0] is not pool.original and workers.WORKERS[0].reaping


@pytest.mark.parametrize("hard_axis", [None, "deadline", "absolute_ceiling"])
def test_cold_loading_gets_new_idle_observation_without_resetting_hard_clock(pool, monkeypatch, hard_axis):
    from supervisor import evolution_lifecycle, state

    monkeypatch.setattr(state, "budget_remaining", lambda *args, **kwargs: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    task = {**pool.meta["task"], "_owner_wait_resume": {**pool.wait, "restart_transaction_id": "tx"}}
    workers.RUNNING.clear()
    pool.original.busy_task_id = None
    workers.PENDING.append(task)
    workers.assign_tasks()
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 10)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 1000 if hard_axis == "absolute_ceiling" else 10000)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    reaps = stdqueue.Queue()
    monkeypatch.setattr(queue, "_reap_queue", reaps)
    if hard_axis == "deadline":
        workers.RUNNING["owner"]["task"]["deadline_at"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()
    assert workers.RUNNING["owner"]["started_at"] == pool.wait["started_at"]
    queue._enforce_task_timeouts_locked(workers, time.time(), 0, {})
    if hard_axis:
        assert reaps.get_nowait()["terminal_reason"] == hard_axis
    else:
        assert reaps.empty(), "Cold source loading inherited the task's old idle clock"
        assert "owner" in workers.RUNNING


@pytest.mark.parametrize("stage", ["grant_pending", "grant_consumed", "cold_loading"])
def test_native_continuation_crash_never_replays_original_work(pool, monkeypatch, stage):
    from ouroboros import delegate_recovery
    from supervisor import evolution_lifecycle, state

    pool.meta["attempt"] = pool.meta["task"]["_attempt"] = 1
    pool.event["task_attempt"] = pool.wait["task_attempt"] = 1
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda pid: None)
    monkeypatch.setattr(delegate_recovery, "reconcile_unrecoverable_task", lambda *a: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *a, **k: {
        "cost_accounting_status": "available", "cost_final": True, "total_rounds": 7,
        "accounted_upper_bound_usd": 2.0,
    })
    if stage == "cold_loading":
        monkeypatch.setattr(state, "budget_remaining", lambda *args, **kwargs: 100)
        monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
        task = {**pool.meta["task"], "_owner_wait_resume": {**pool.wait, "restart_transaction_id": "tx"}}
        workers.RUNNING.clear()
        pool.original.busy_task_id = None
        workers.PENDING.append(task)
        workers.assign_tasks()
        assert pool.original.in_q.get_nowait()["id"] == "owner"
    else:
        park(pool)
        worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
        maintain_capacity()
        assert pool.original.active_capacity
        if stage == "grant_consumed":
            assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    pool.original.proc.alive, pool.original.proc.exitcode = False, 1
    respawn_ids, disabled = workers._ensure_workers_healthy_locked(queue)
    assert not disabled and respawn_ids == []
    from supervisor.worker_health import recover_confirmed_dead_worker

    recover_confirmed_dead_worker(queue._reap_queue.get_nowait())
    assert not workers.PENDING, "Original input was requeued after completed effects were checkpointed"
    saved = load_task_result(pool.root, "owner")
    assert saved["status"] == "failed" and saved["reason_code"] == "worker_crash_owner_wait"
    assert saved["total_rounds"] == 7


def test_grant_consumes_old_cold_marker_before_publishing_snapshot(pool):
    pool.meta["task"]["_owner_wait_resume"] = {**pool.wait, "restart_transaction_id": "tx"}
    park(pool)
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    maintain_capacity()
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert "_owner_wait_resume" not in pool.meta["task"]
    row = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["running"][0]
    assert "_owner_wait_resume" not in row["task"]
    assert row["owner_wait"]["state"] == "resumed"


@pytest.mark.parametrize("failure", ["snapshot", "command"])
def test_failed_grant_rolls_back_marker_and_wait_together(pool, monkeypatch, failure):
    marker = {**pool.wait, "restart_transaction_id": "tx"}
    pool.meta["task"]["_owner_wait_resume"] = marker
    park(pool)
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    if failure == "snapshot":
        monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kw: False)
    else:
        def fail_command(*args, **kwargs):
            raise OSError("input queue unavailable")
        monkeypatch.setattr(worker_owner_wait, "_command", fail_command)
    maintain_capacity()
    assert not pool.original.active_capacity and pool.original.in_q.empty()
    assert pool.meta["task"]["_owner_wait_resume"] == marker
    assert pool.meta["owner_wait"]["state"] == "waiting"
    assert load_task_result(pool.root, "owner")["owner_wait"]["state"] == "waiting"


@pytest.mark.parametrize("stage", ["resumed", "cold_loading", "ordinary", "other_attempt"])
def test_idle_timeout_never_clones_a_checkpointed_attempt(pool, monkeypatch, stage):
    from supervisor import evolution_lifecycle, state

    if stage == "cold_loading":
        monkeypatch.setattr(state, "budget_remaining", lambda *args, **kwargs: 100)
        monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
        task = {**pool.meta["task"], "_owner_wait_resume": {**pool.wait, "restart_transaction_id": "tx"}}
        workers.RUNNING.clear()
        pool.original.busy_task_id = None
        workers.PENDING.append(task)
        workers.assign_tasks()
        assert pool.original.in_q.get_nowait()["id"] == "owner"
    elif stage != "ordinary":
        park(pool)
        worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
        maintain_capacity()
        assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
        if stage == "other_attempt":
            pool.meta["owner_wait"]["task_attempt"] = 2
    monkeypatch.setattr(queue, "QUEUE_MAX_RETRIES", 3)
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 10)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 10000)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    reaps = stdqueue.Queue()
    monkeypatch.setattr(queue, "_reap_queue", reaps)
    queue._enforce_task_timeouts_locked(workers, time.time() + 1000, 0, {})
    assert reaps.empty() and not workers.PENDING
    assert workers.RUNNING["owner"]["silence_attention"]["state"] == "pending_attention"
    assert not workers.RUNNING["owner"].get("finalization_requested_at")


def test_the_wait_stamp_rides_park_grant_and_restart_handoff_beside_the_task_start(pool):
    """``parked_at`` dates THIS wait; the supervisor still owns ``started_at``
    (the lifetime clock) and every later write carries the stamp unchanged."""
    from ouroboros.owner_wait import prepare_owner_wait_handoffs

    stamped = {**pool.wait, "parked_at": "2026-09-26T11:30:00+00:00"}
    worker_owner_wait.handle_owner_wait({**pool.event, "checkpoint": stamped}, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    saved = load_task_result(pool.root, "owner")["owner_wait"]
    assert saved["parked_at"] == "2026-09-26T11:30:00+00:00"
    assert saved["started_at"] == pool.meta["started_at"] != saved["parked_at"]
    selected = prepare_owner_wait_handoffs(pool.root, workers.RUNNING, "tx1")
    assert selected == {"owner"}
    assert pool.meta["task"]["_owner_wait_resume"]["parked_at"] == "2026-09-26T11:30:00+00:00"
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    maintain_capacity()
    resumed = load_task_result(pool.root, "owner")["owner_wait"]
    assert resumed["state"] == "resumed" and resumed["parked_at"] == "2026-09-26T11:30:00+00:00"
