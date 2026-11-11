"""Readiness exhaustion closes admission without destroying an unsettled task."""

import queue as stdqueue
import threading
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result
from supervisor import queue, task_admission, worker_pool_lifecycle, workers


class Process:
    def __init__(self, pid, alive=True):
        self.pid = pid
        self.alive = alive

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass


@pytest.fixture
def pool(tmp_path, monkeypatch):
    pending, running, slots = [], {}, {}
    for module in (queue, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", running)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    for name in ("ADMISSION_RESERVATIONS", "ACCEPTANCE_FENCES", "BUDGET_ROOT_FENCES"):
        monkeypatch.setattr(queue, name, {})
    monkeypatch.setattr(workers, "WORKERS", slots)
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(1))
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "")
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda _pid: None)
    monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", lambda *a, **k: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *a, **k: {
        "cost_accounting_status": "unavailable", "cost_final": False,
    })
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(queue, "load_state", lambda: {"owner_chat_id": 1})
    events = stdqueue.Queue()
    monkeypatch.setattr(workers, "get_event_q", lambda: events)

    def kill(pid, **kwargs):
        for worker in slots.values():
            if worker.proc.pid == pid:
                worker.proc.alive = False

    monkeypatch.setattr(worker_pool_lifecycle, "kill_worker_tree", kill)
    monkeypatch.setattr(workers, "kill_worker_tree", kill)
    slot = workers.Worker(0, Process(50001), stdqueue.Queue(), reaping=True, booting=True)
    slots[0] = slot
    return SimpleNamespace(root=tmp_path, slot=slot, events=events)


def exhaust(pool):
    worker_pool_lifecycle._replace_unready_slot(
        0, pool.slot, 0, 0, worker_pool_lifecycle.WORKER_READY_MAX_ATTEMPTS,
    )


def test_final_readiness_attempt_fails_pending_and_closes_every_admission(pool, monkeypatch):
    workers.PENDING.append({"id": "pending", "type": "task", "chat_id": 1, "text": "work"})
    exhaust(pool)
    assert pool.slot.readiness_exhausted and pool.slot.reaping
    assert not workers.WORKERS and not workers.PENDING
    result = load_task_result(pool.root, "pending")
    assert result["status"] == "failed"
    assert "startup attempts were exhausted" in result["result"]
    assert workers.worker_pool_admission_state()["disabled_reason"] == "worker_readiness_exhausted"
    assert task_admission.reserve_task_admission("new", "token", require_worker_pool=True)["status"] == "blocked"
    assert queue.enqueue_task({"id": "new", "type": "task", "chat_id": 1,
                               "_require_worker_pool": True})["_admission_blocked"] == "worker_pool_unavailable"
    monkeypatch.setattr(workers, "spawn_workers", lambda *a: pytest.fail("automatic restart"))
    assert not workers.ensure_worker_pool_started()
    assert queue.queue_deep_self_review_task("owner:/review", force=True, chat_id=1) is None
    assert not workers.PENDING


@pytest.mark.parametrize("healthy_state", ["booting", "working", "temporary_reaping"])
def test_unexhausted_reservation_remains_a_queue_target(pool, healthy_state):
    workers.WORKERS[1] = workers.Worker(
        1, Process(50002, alive=healthy_state != "temporary_reaping"), stdqueue.Queue(),
        reaping=healthy_state != "working", busy_task_id="busy" if healthy_state == "working" else None,
        booting=healthy_state == "booting",
    )
    exhaust(pool)
    assert workers.WORKERS[0] is pool.slot and pool.slot.readiness_exhausted
    assert workers.worker_pool_admission_state()["available"]
    assert workers.WORKERS[1].proc.alive is (healthy_state != "temporary_reaping")


def test_failed_disable_snapshot_still_refuses_work_and_can_retry(pool, monkeypatch):
    original = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kwargs: False)
    exhaust(pool)
    assert not workers._WORKER_POOL_DISABLED_REASON
    assert workers.WORKERS[0] is pool.slot
    assert workers.worker_pool_admission_state()["disabled_reason"] == "worker_readiness_exhausted"
    assert task_admission.reserve_task_admission("new", "t", require_worker_pool=True)["status"] == "blocked"
    monkeypatch.setattr(queue, "persist_queue_snapshot", original)
    assert workers.disable_exhausted_worker_pool()
    assert not workers.WORKERS


def test_failed_pending_write_retains_existing_terminalization_custody(pool, monkeypatch):
    workers.PENDING.append({"id": "pending", "type": "task", "chat_id": 1, "text": "work"})
    original = workers._write_failure_result

    def fail(*a, **k):
        raise OSError("temporarily unavailable")

    monkeypatch.setattr(workers, "_write_failure_result", fail)
    exhaust(pool)
    assert len(workers.PENDING) == 1 and workers.PENDING[0]["id"] == "pending"
    assert workers._terminalization_retry_spec(workers.PENDING[0])
    assert not workers.WORKERS
    monkeypatch.setattr(workers, "_write_failure_result", original)
    workers._retry_terminalization_pending()
    assert load_task_result(pool.root, "pending")["status"] == "failed"
    assert not workers.PENDING


def test_late_readiness_cannot_reopen_exhausted_slot(pool, monkeypatch):
    monkeypatch.setattr(workers, "disable_exhausted_worker_pool", lambda: False)
    exhaust(pool)
    worker_pool_lifecycle._open_ready_slot(
        0, pool.slot, {"pid": pool.slot.proc.pid, "git_sha": "sha"}, "sha", 0, 0, 3,
    )
    worker_pool_lifecycle._release_booting_slot(0, pool.slot, 0, 3, "watcher_error")
    assert pool.slot.reaping and not workers.worker_pool_admission_state()["available"]


def test_unsettled_dead_task_is_left_to_existing_completion_owner(pool):
    pool.slot.readiness_exhausted = True
    workers.RUNNING["saved"] = {"task": {"id": "saved"}, "worker_id": 3}
    assert not workers.disable_exhausted_worker_pool()
    assert "saved" in workers.RUNNING and workers.WORKERS[0] is pool.slot


def test_exhaustion_check_does_not_wait_for_another_lifecycle_operation(pool, monkeypatch):
    lock = threading.RLock()
    monkeypatch.setattr(worker_pool_lifecycle, "_WORKER_LIFECYCLE_LOCK", lock)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    outcomes = []
    pool.slot.readiness_exhausted = True

    def hold_lifecycle():
        with lock:
            entered.set()
            assert release.wait(5)

    def check_exhaustion():
        try:
            outcomes.append(workers.disable_exhausted_worker_pool())
        finally:
            completed.set()

    holder = threading.Thread(target=hold_lifecycle)
    check = threading.Thread(target=check_exhaustion)
    holder.start()
    try:
        assert entered.wait(2)
        check.start()
        assert completed.wait(2), "health intake waited for lifecycle cleanup"
        assert outcomes == [False]
        assert workers.WORKERS[0] is pool.slot
    finally:
        release.set()
        holder.join(5)
        if check.ident is not None:
            check.join(5)
    assert workers.disable_exhausted_worker_pool()
    assert not workers.WORKERS


def test_repository_writer_policy_stays_separate_from_execution_capacity(pool, monkeypatch):
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "managed update")
    assert not workers.worker_pool_admission_state()["available"]
    assert workers._worker_pool_execution_state()["available"]
    assert task_admission.reserve_task_admission("resolver", "t", require_worker_pool=True)["status"] == "reserved"
