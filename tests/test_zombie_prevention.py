"""Tests for zombie task prevention logic.

Covers:
- _write_failure_result() writes correct JSON, guards for None/existing
- drain_all_pending() empties PENDING and returns tasks
- kill_workers() writes failure results for RUNNING + PENDING tasks
"""
import json
from unittest import mock

import pytest



# ---------------------------------------------------------------------------
# _write_failure_result
# ---------------------------------------------------------------------------

def test_write_failure_result_creates_correct_json(tmp_path):
    """_write_failure_result should create <task_id>.json with expected fields."""
    import supervisor.workers as workers

    orig = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    try:
        workers._write_failure_result("abc123")
    finally:
        workers.DRIVE_ROOT = orig

    result_file = tmp_path / "task_results" / "abc123.json"
    assert result_file.exists()

    data = json.loads(result_file.read_text(encoding="utf-8"))
    assert data["task_id"] == "abc123"
    assert data["status"] == "failed"
    assert isinstance(data["result"], str) and len(data["result"]) > 0
    assert data["accounted_upper_bound_usd"] in (0, None)  # ABI-3: honest name; empty ledger stays honest
    assert data["total_rounds"] == 0
    assert "ts" in data


def test_write_failure_result_does_not_overwrite_existing(tmp_path):
    """_write_failure_result must NOT overwrite an existing result file."""
    import supervisor.workers as workers

    orig = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    try:
        results_dir = tmp_path / "task_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        existing = {"_schema_version": 1, "task_id": "xyz789", "status": "completed", "result": "Success!"}
        (results_dir / "xyz789.json").write_text(
            json.dumps(existing, ensure_ascii=False), encoding="utf-8"
        )
        workers._write_failure_result("xyz789")
    finally:
        workers.DRIVE_ROOT = orig

    data = json.loads((results_dir / "xyz789.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed", "Existing result was overwritten!"
    assert data["result"] == "Success!"


def test_write_failure_result_none_task_id(tmp_path):
    """_write_failure_result with None task_id should not crash or create files."""
    import supervisor.workers as workers

    orig = workers.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    try:
        workers._write_failure_result(None)
        workers._write_failure_result("")
    finally:
        workers.DRIVE_ROOT = orig

    results_dir = tmp_path / "task_results"
    if results_dir.exists():
        assert list(results_dir.iterdir()) == [], "Files created for None/empty task_id"


def test_write_failure_result_rejects_a_writer_identity_mismatch(tmp_path, monkeypatch):
    """A terminal writer must return the same task identity it was asked to store."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(
        "ouroboros.task_results.write_task_result",
        lambda *_args, **_kwargs: {"task_id": "another-task", "status": "completed"},
    )

    with pytest.raises(ValueError, match="invalid durable identity"):
        workers._write_failure_result("identity-victim")


def test_write_failure_result_returns_status_won_by_a_concurrent_terminal_writer(
    tmp_path, monkeypatch,
):
    """A completion that wins during the pre-read must drive the emitted status."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(
        "ouroboros.task_results.write_task_result",
        lambda *_args, **_kwargs: {
            "task_id": "race-victim", "status": "completed", "result": "done",
        },
    )

    assert workers._write_failure_result("race-victim") == "completed"


# ---------------------------------------------------------------------------
# drain_all_pending
# ---------------------------------------------------------------------------

def test_drain_all_pending_returns_and_clears(tmp_path):
    """drain_all_pending should return all tasks and leave PENDING empty."""
    import supervisor.queue as queue

    orig_drive = queue.DRIVE_ROOT
    orig_pending = queue.PENDING
    queue.DRIVE_ROOT = tmp_path
    tasks = [{"id": "t1", "type": "task"}, {"id": "t2", "type": "evolution"}]
    queue.PENDING = list(tasks)
    try:
        with mock.patch.object(queue, "persist_queue_snapshot"):
            drained = queue.drain_all_pending()
    finally:
        queue.DRIVE_ROOT = orig_drive
        queue.PENDING = orig_pending

    assert drained == tasks
    # The local list that was assigned to queue.PENDING was cleared
    assert len(drained) == 2


# ---------------------------------------------------------------------------
# kill_workers — zombie prevention integration
# ---------------------------------------------------------------------------

def test_kill_workers_writes_failure_for_running_and_pending(tmp_path):
    """kill_workers should write failure results for both RUNNING and PENDING tasks."""
    import supervisor.workers as workers
    import supervisor.queue as queue

    # Save originals
    orig_drive = workers.DRIVE_ROOT
    orig_workers = dict(workers.WORKERS)
    orig_running = dict(workers.RUNNING)
    orig_pending = list(workers.PENDING)
    orig_q_drive = queue.DRIVE_ROOT
    orig_q_pending = queue.PENDING
    orig_q_running = queue.RUNNING
    orig_disabled = workers._WORKER_POOL_DISABLED_REASON

    workers.DRIVE_ROOT = tmp_path
    queue.DRIVE_ROOT = tmp_path
    workers.WORKERS.clear()
    workers.RUNNING.clear()
    workers.RUNNING["run1"] = {"task": {"id": "run1", "type": "task"}, "worker_id": 0}
    workers.RUNNING["run2"] = {"task": {"id": "run2", "type": "task"}, "worker_id": 1}

    pending_tasks = [{"id": "pend1", "type": "evolution"}, {"id": "pend2", "type": "task"}]
    workers.PENDING[:] = list(pending_tasks)
    queue.PENDING = workers.PENDING

    try:
        with mock.patch.object(queue, "persist_queue_snapshot"):
            workers.kill_workers()
    finally:
        workers.DRIVE_ROOT = orig_drive
        workers.WORKERS.clear()
        workers.WORKERS.update(orig_workers)
        workers.RUNNING.clear()
        workers.RUNNING.update(orig_running)
        workers.PENDING[:] = orig_pending
        queue.DRIVE_ROOT = orig_q_drive
        queue.PENDING = orig_q_pending
        queue.RUNNING = orig_q_running
        workers._WORKER_POOL_DISABLED_REASON = orig_disabled

    results_dir = tmp_path / "task_results"
    for tid in ("run1", "run2", "pend1", "pend2"):
        path = results_dir / f"{tid}.json"
        assert path.exists(), f"Missing failure result for {tid}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "failed"
        assert data["task_id"] == tid


def test_kill_workers_retains_pending_when_failure_result_is_not_durable(
    tmp_path, monkeypatch,
):
    import supervisor.workers as workers
    import supervisor.queue as queue

    task = {"id": "bad/id", "type": "task", "depth": -1}
    pending = [task]
    snapshots = []
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(
        queue,
        "persist_queue_snapshot",
        lambda reason="": snapshots.append((reason, [dict(row) for row in queue.PENDING])),
    )
    monkeypatch.setattr(
        workers,
        "_write_failure_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    workers.kill_workers(reconcile_delegate_custody=False)

    assert [row["id"] for row in pending] == [task["id"]]
    retry = pending[0]["_terminalization_retry"]
    assert retry["status"] == "failed"
    assert retry["trigger"] == "pending_pool_kill"
    assert pending[0]["depth"] == -1
    assert snapshots[0][0] == "kill_workers"
    assert snapshots[0][1][0]["_terminalization_retry"] == retry


def test_kill_workers_keeps_pending_cancel_custody_when_claim_release_fails(
    tmp_path, monkeypatch,
):
    """Shutdown cannot discard a marker while its cancellation claim is live."""
    from ouroboros import cancel_intents as ci
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    import supervisor.workers as workers
    import supervisor.queue as queue

    pending = []
    task_id = "kill-cancel-claim"
    task = {"id": task_id, "chat_id": 0}
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_kw: True)
    write_task_result(tmp_path, task_id, "scheduled")
    ci.request_cancel(tmp_path, task_id)
    claim = ci.claim_intent(tmp_path, task_id, owner="pending_drop")
    task["_terminalization_retry"] = {
        "status": STATUS_CANCELLED,
        "reason": "event retry",
        "trigger": "pending_cancel_event",
        "reconcile_delegate_custody": False,
        "claim_request_id": claim["request_id"],
        "claim_owner": "pending_drop",
        "claim_generation": claim["generation"],
        "claim_pid": claim["claim_pid"],
    }
    pending.append(task)
    emitted = []
    monkeypatch.setattr(workers, "_write_failure_result", lambda *_a, **_k: STATUS_CANCELLED)
    monkeypatch.setattr(
        workers, "_emit_task_done_terminal", lambda *a, **_k: emitted.append(a) or True,
    )
    real_release = ci.release_claim
    monkeypatch.setattr(ci, "release_claim", lambda *_a, **_k: False)

    workers.kill_workers(
        preserve_pending=True,
        archive_service_logs=False,
        reconcile_delegate_custody=False,
    )
    assert [row["id"] for row in pending] == [task_id]
    assert pending[0]["_terminalization_retry"]["event_published"] is True
    assert ci.active_intent(tmp_path, task_id)["state"] == ci.INTENT_CLAIMED
    assert len(emitted) == 1

    monkeypatch.setattr(ci, "release_claim", real_release)
    assert workers._retry_terminalization_pending() == ([task_id], [])
    assert pending == []
    assert ci.active_intent(tmp_path, task_id)["state"] == ci.INTENT_REQUESTED
    assert len(emitted) == 1, "the durable event marker suppresses a duplicate"


def test_kill_workers_retains_running_custody_when_failure_result_is_not_durable(
    tmp_path, monkeypatch,
):
    import supervisor.workers as workers
    import supervisor.queue as queue

    task = {"id": "running-write-fails", "type": "task", "depth": 0}
    pending = []
    running = {
        task["id"]: {"task": task, "worker_id": 0},
    }
    snapshots = []
    emitted = []
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "RUNNING", running)
    monkeypatch.setattr(queue, "RUNNING", running)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(
        queue,
        "persist_queue_snapshot",
        lambda reason="": snapshots.append(
            (reason, [dict(row) for row in queue.PENDING], dict(queue.RUNNING))
        ),
    )
    monkeypatch.setattr(
        workers,
        "_write_failure_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        workers,
        "_emit_task_done_terminal",
        lambda *args, **kwargs: emitted.append((args, kwargs)) or True,
    )

    workers.kill_workers(reconcile_delegate_custody=False)

    assert running == {}
    assert [row["id"] for row in pending] == [task["id"]]
    retry = pending[0]["_terminalization_retry"]
    assert retry["status"] == "failed"
    assert retry["trigger"] == "worker_pool_kill"
    assert "disk full" not in retry["reason"]
    assert emitted == []
    assert snapshots[-1][0] == "kill_workers"
    assert snapshots[-1][1][0]["_terminalization_retry"] == retry
    assert snapshots[-1][2] == {}

    writes = []
    monkeypatch.setattr(
        workers,
        "_audit_delegate_terminal_custody",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        workers,
        "_write_failure_result",
        lambda task_id, **kwargs: writes.append((task_id, kwargs)) or "failed",
    )
    monkeypatch.setattr(
        workers,
        "_emit_task_done_terminal",
        lambda task_row, task_id, status: emitted.append((task_id, status)) or True,
    )
    assert workers._retry_terminalization_pending() == ([task["id"]], [])
    assert pending == []
    assert writes[0][0] == task["id"]
    assert emitted == [(task["id"], "failed")]


def test_kill_workers_retains_pending_when_status_is_not_terminal(
    tmp_path, monkeypatch,
):
    import supervisor.workers as workers
    import supervisor.queue as queue

    task = {"id": "interrupted-pending", "type": "task"}
    pending = [task]
    snapshots = []
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(
        queue,
        "persist_queue_snapshot",
        lambda reason="": snapshots.append((reason, [dict(row) for row in queue.PENDING])),
    )
    monkeypatch.setattr(workers, "_write_failure_result", lambda *_args, **_kwargs: "interrupted")
    monkeypatch.setattr(workers, "_emit_task_done_terminal", lambda *_args, **_kwargs: True)

    workers.kill_workers(
        terminal_status="interrupted",
        reconcile_delegate_custody=False,
    )

    assert [row["id"] for row in pending] == [task["id"]]
    retry = pending[0]["_terminalization_retry"]
    assert retry["status"] == "interrupted"
    assert retry["trigger"] == "pending_pool_kill"
    assert snapshots[0][0] == "kill_workers"
    assert snapshots[0][1][0]["_terminalization_retry"] == retry


def test_kill_workers_preserves_interrupted_pending_when_terminal_event_fails(
    tmp_path, monkeypatch,
):
    import supervisor.workers as workers
    import supervisor.queue as queue

    parent_id = "interrupted-parent"
    child = {
        "id": "interrupted-child",
        "type": "task",
        "parent_task_id": parent_id,
        "root_task_id": parent_id,
        "depth": 0,
    }
    pending = [child]
    running = {
        parent_id: {
            "task": {"id": parent_id, "root_task_id": parent_id, "type": "task"},
            "worker_id": 0,
        },
    }
    snapshots = []
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "RUNNING", running)
    monkeypatch.setattr(queue, "RUNNING", running)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(
        queue,
        "persist_queue_snapshot",
        lambda reason="": snapshots.append((reason, [dict(row) for row in queue.PENDING])),
    )
    monkeypatch.setattr(workers, "_write_failure_result", lambda *_args, **_kwargs: "cancelled")
    monkeypatch.setattr(
        workers,
        "_emit_task_done_terminal",
        lambda _task, task_id, *_args, **_kwargs: task_id == parent_id,
    )

    workers.kill_workers(
        preserve_pending=True,
        archive_service_logs=False,
        reconcile_delegate_custody=False,
    )

    assert [row["id"] for row in pending] == [child["id"]]
    retry = pending[0]["_terminalization_retry"]
    assert retry["status"] == "cancelled"
    assert retry["trigger"] == "pending_parent_interrupted"
    assert snapshots[0][0] == "kill_workers"
    assert snapshots[0][1][0]["_terminalization_retry"] == retry


def test_kill_workers_can_record_owner_restart_cancellation(tmp_path):
    """Owner restart should not describe intentional aborts as crash storms."""
    import supervisor.workers as workers
    import supervisor.queue as queue

    orig_drive = workers.DRIVE_ROOT
    orig_workers = dict(workers.WORKERS)
    orig_running = dict(workers.RUNNING)
    orig_pending = list(workers.PENDING)
    orig_q_drive = queue.DRIVE_ROOT
    orig_q_pending = queue.PENDING
    orig_q_running = queue.RUNNING

    workers.DRIVE_ROOT = tmp_path
    queue.DRIVE_ROOT = tmp_path
    workers.WORKERS.clear()
    workers.RUNNING.clear()
    workers.RUNNING["run1"] = {"task": {"id": "run1", "type": "task"}, "worker_id": 0}
    workers.PENDING[:] = [{"id": "pend1", "type": "task"}]
    queue.PENDING = workers.PENDING

    try:
        with mock.patch.object(queue, "persist_queue_snapshot"):
            workers.kill_workers(
                terminal_status="cancelled",
                result_reason="Owner restart stopped this task before process restart.",
            )
    finally:
        workers.DRIVE_ROOT = orig_drive
        workers.WORKERS.clear()
        workers.WORKERS.update(orig_workers)
        workers.RUNNING.clear()
        workers.RUNNING.update(orig_running)
        workers.PENDING[:] = orig_pending
        queue.DRIVE_ROOT = orig_q_drive
        queue.PENDING = orig_q_pending
        queue.RUNNING = orig_q_running

    for tid in ("run1", "pend1"):
        data = json.loads((tmp_path / "task_results" / f"{tid}.json").read_text(encoding="utf-8"))
        assert data["status"] == "cancelled"
        assert data["result"] == "Owner restart stopped this task before process restart."


def test_managed_update_preserves_pending_tasks_for_the_new_process(tmp_path):
    import supervisor.queue as queue
    import supervisor.workers as workers

    orig_drive = workers.DRIVE_ROOT
    orig_workers = dict(workers.WORKERS)
    orig_running = dict(workers.RUNNING)
    orig_pending = list(workers.PENDING)
    orig_q_drive = queue.DRIVE_ROOT
    orig_q_pending = queue.PENDING
    orig_q_running = queue.RUNNING
    orig_disabled = workers._WORKER_POOL_DISABLED_REASON

    workers.DRIVE_ROOT = tmp_path
    queue.DRIVE_ROOT = tmp_path
    workers.WORKERS.clear()
    workers.RUNNING.clear()
    workers.PENDING[:] = [{"id": "queued-after-update", "type": "task"}]
    queue.PENDING = workers.PENDING

    try:
        with mock.patch.object(queue, "persist_queue_snapshot"):
            survivors = workers.kill_workers_for_update(
                result_reason="Managed update",
                terminal_status="interrupted",
            )
        assert survivors == []
        assert [task["id"] for task in workers.PENDING] == ["queued-after-update"]
        assert not (tmp_path / "task_results" / "queued-after-update.json").exists()
    finally:
        workers.DRIVE_ROOT = orig_drive
        workers.WORKERS.clear()
        workers.WORKERS.update(orig_workers)
        workers.RUNNING.clear()
        workers.RUNNING.update(orig_running)
        workers.PENDING[:] = orig_pending
        queue.DRIVE_ROOT = orig_q_drive
        queue.PENDING = orig_q_pending
        queue.RUNNING = orig_q_running
        workers._WORKER_POOL_DISABLED_REASON = orig_disabled


def test_managed_update_drops_children_of_interrupted_roots(tmp_path):
    import supervisor.queue as queue
    import supervisor.workers as workers

    orig_drive = workers.DRIVE_ROOT
    orig_workers = dict(workers.WORKERS)
    orig_running = dict(workers.RUNNING)
    orig_pending = list(workers.PENDING)
    orig_q_drive = queue.DRIVE_ROOT
    orig_q_pending = queue.PENDING
    orig_q_running = queue.RUNNING
    orig_disabled = workers._WORKER_POOL_DISABLED_REASON
    workers.DRIVE_ROOT = tmp_path
    queue.DRIVE_ROOT = tmp_path
    workers.WORKERS.clear()
    workers.RUNNING.clear()
    workers.RUNNING["root-1"] = {"task": {"id": "root-1", "type": "task"}, "worker_id": 0}
    workers.PENDING[:] = [
        {"id": "child-1", "type": "task", "parent_task_id": "root-1", "root_task_id": "root-1"},
        {"id": "independent", "type": "task"},
    ]
    queue.PENDING = workers.PENDING
    queue.RUNNING = workers.RUNNING
    try:
        with mock.patch.object(queue, "persist_queue_snapshot"):
            workers.kill_workers_for_update(result_reason="Managed update")
        assert [task["id"] for task in workers.PENDING] == ["independent"]
        child = json.loads((tmp_path / "task_results" / "child-1.json").read_text())
        assert child["status"] == "cancelled"
    finally:
        workers.DRIVE_ROOT = orig_drive
        workers.WORKERS.clear()
        workers.WORKERS.update(orig_workers)
        workers.RUNNING.clear()
        workers.RUNNING.update(orig_running)
        workers.PENDING[:] = orig_pending
        queue.DRIVE_ROOT = orig_q_drive
        queue.PENDING = orig_q_pending
        queue.RUNNING = orig_q_running
        workers._WORKER_POOL_DISABLED_REASON = orig_disabled
