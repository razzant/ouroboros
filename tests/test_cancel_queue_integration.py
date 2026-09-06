"""The queue side of a cancel intent: pending drop, snapshot restore, fail, steer.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: the readers that consult
the intent projection before acting, the decisions they stamp, the write failures that
leave an intent open, the deference to a live claim owner, and the steering that is
refused while a cancel is active.

The v7next tip adds the upstream dispatch-authority family: assignment and the timeout
reaper block (retain, never clone) when the cancel authority cannot be read, and the
cancel-authority hold never releases a terminal row to dispatch.
"""

from __future__ import annotations
import json
import types
import pytest
from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCEL_REQUESTED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)

from tests._cancel_intents_shared import qenv as _qenv

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
qenv = _qenv


def test_drop_cancelled_pending_consults_the_intent_projection(qenv, monkeypatch):
    from supervisor import workers

    emitted: list = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal",
                        lambda task, tid, status, **kw: emitted.append((tid, status, kw)) or True)
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)

    qenv.q.PENDING[:] = [
        {"id": "keepme", "chat_id": 1},
        {"id": "dropme", "chat_id": 1},
    ]
    write_task_result(qenv.drive, "dropme", "scheduled")
    ci.request_cancel(qenv.drive, "dropme", reason="parent stopped the plan")

    workers._drop_cancelled_pending()

    assert [t["id"] for t in qenv.q.PENDING] == ["keepme"]
    stored = load_task_result(qenv.drive, "dropme")
    assert stored["status"] == STATUS_CANCELLED
    assert "cost_accounting_status" in stored  # reconstructed, not omitted
    assert ci.active_intent(qenv.drive, "dropme") is None
    assert emitted and emitted[0][0] == "dropme" and emitted[0][1] == "cancelled"

def test_assignment_blocks_when_cancel_intent_projection_is_unreadable(
    tmp_path, monkeypatch,
):
    """A corrupt intent projection cannot be treated as an empty projection."""
    from supervisor import queue, state, workers

    delivered: list[dict] = []
    worker = types.SimpleNamespace(
        wid=1,
        busy_task_id=None,
        reaping=False,
        in_q=types.SimpleNamespace(put=lambda task: delivered.append(dict(task))),
    )
    task = {
        "id": "blocked-by-intent-corruption",
        "type": "task",
        "chat_id": 1,
        "depth": 0,
        "budget_drive_root": str(tmp_path),
    }
    pending = [task]
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {1: worker})
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(
        state,
        "budget_remaining",
        lambda *_args, **_kwargs: pytest.fail("budget must not run after authority failure"),
    )
    snapshots: list[str] = []
    monkeypatch.setattr(
        queue,
        "persist_queue_snapshot",
        lambda reason="": snapshots.append(reason) or True,
    )
    queue.BUDGET_ROOT_FENCES.clear()

    write_task_result(tmp_path, task["id"], "scheduled")
    projection = tmp_path / "state" / "cancel_intents.json"
    projection.parent.mkdir(parents=True, exist_ok=True)
    corrupt_bytes = b'{"intents": [broken'
    projection.write_bytes(corrupt_bytes)

    workers.assign_tasks()

    assert pending == [task]
    assert delivered == []
    assert worker.busy_task_id is None
    assert projection.read_bytes() == corrupt_bytes
    assert snapshots == ["cancellation_authority_indeterminate"]

def test_assignment_retains_pending_when_claim_authority_raises(qenv, monkeypatch):
    """Only an explicit claim refusal may yield custody to another owner."""
    from supervisor import state, workers

    task_id = "claim-authority-failure"
    pending = [{"id": task_id, "type": "task", "chat_id": 1, "depth": 0}]
    delivered: list[dict] = []
    worker = types.SimpleNamespace(
        wid=1,
        busy_task_id=None,
        reaping=False,
        in_q=types.SimpleNamespace(put=lambda task: delivered.append(dict(task))),
    )
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {1: worker})
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(
        state,
        "budget_remaining",
        lambda *_args, **_kwargs: pytest.fail("budget must not run after claim failure"),
    )
    monkeypatch.setattr(
        ci,
        "claim_intent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ci.CancelIntentProjectionCorrupt("projection changed during claim")
        ),
    )
    write_task_result(qenv.drive, task_id, "scheduled")
    ci.request_cancel(qenv.drive, task_id, reason="stop")

    workers.assign_tasks()

    assert [row["id"] for row in pending] == [task_id]
    assert delivered == []
    assert worker.busy_task_id is None
    assert load_task_result(qenv.drive, task_id)["status"] == "scheduled"
    assert ci.active_intent(qenv.drive, task_id)["state"] == ci.INTENT_REQUESTED

def test_timeout_reaper_does_not_clone_over_unreadable_cancel_authority(
    tmp_path, monkeypatch,
):
    """Every physical retry must prove that the old id has no cancel intent."""
    from supervisor import queue, workers

    now = 10_000.0
    task_id = "timeout-over-corrupt-intent-store"
    task = {"id": task_id, "type": "task", "chat_id": 0}
    meta = {
        "task": task,
        "started_at": now - 1000.0,
        "last_heartbeat_at": now - 1000.0,
        "last_progress_at": now - 1000.0,
        "attempt": 1,
        "worker_id": -1,
    }
    running = {task_id: meta}
    queue.init_queue_refs([], running, {"value": 0})
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0.0)
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 60.0)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 0.0)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 10_000_000.0)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    jobs: list[dict] = []
    monkeypatch.setattr(queue, "_reap_queue", types.SimpleNamespace(put=jobs.append))
    monkeypatch.setattr(workers, "WORKERS", {})

    projection = tmp_path / "state" / "cancel_intents.json"
    projection.parent.mkdir(parents=True)
    projection.write_bytes(b'{"intents": [broken')

    queue._enforce_task_timeouts_locked(workers, now, 0, {})

    assert len(jobs) == 1
    assert jobs[0]["task_id"] == task_id
    assert jobs[0]["will_retry"] is False
    assert jobs[0]["retry_task_id"] == ""
    assert task_id not in running

def test_snapshot_restore_refuses_a_task_with_active_intent(qenv, monkeypatch):
    from ouroboros.utils import utc_now_iso

    ci.request_cancel(qenv.drive, "restoreme")
    snapshot = {
        "ts": utc_now_iso(),
        "pending": [{"task": {"id": "restoreme", "chat_id": 1, "type": "chat"}}],
        "running": [],
        "acceptance_fences": [],
        "budget_root_fences": [],
    }
    state_dir = qenv.drive / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(qenv.q, "QUEUE_SNAPSHOT_PATH", state_dir / "queue_snapshot.json",
                        raising=False)

    restored = qenv.q.restore_pending_from_snapshot()

    assert restored == 0
    assert qenv.q.PENDING == []

def test_snapshot_restore_blocks_when_cancel_intent_projection_is_unreadable(
    tmp_path, monkeypatch,
):
    """Restart recovery must not resurrect a row over a corrupt intent store."""
    from ouroboros.utils import utc_now_iso
    from supervisor import queue

    pending: list[dict] = []
    running: dict = {}
    counter = {"value": 0}
    queue.init_queue_refs(pending, running, counter)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    snapshot_path = tmp_path / "state" / "queue_snapshot.json"
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", snapshot_path)
    queue.ACCEPTANCE_FENCES.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    queue.ADMISSION_RESERVATIONS.clear()

    task_id = "restore-over-corrupt-intent-store"
    write_task_result(tmp_path, task_id, "scheduled")
    projection = tmp_path / "state" / "cancel_intents.json"
    projection.parent.mkdir(parents=True, exist_ok=True)
    corrupt_bytes = b'{"intents": [broken'
    projection.write_bytes(corrupt_bytes)
    task = {
        "id": task_id,
        "type": "task",
        "chat_id": 1,
        "depth": 0,
        "budget_drive_root": str(tmp_path),
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps({
            "ts": utc_now_iso(),
            "pending": [{"id": task_id, "queue_seq": 1, "task": task}],
            "running": [],
            "acceptance_fences": [],
            "budget_root_fences": [],
        }),
        encoding="utf-8",
    )

    assert queue.restore_pending_from_snapshot() == 1
    assert [row["id"] for row in pending] == [task_id]
    assert "_terminalization_retry" not in pending[0]
    assert isinstance(pending[0].get("_cancel_intent_authority_hold"), dict)
    assert load_task_result(tmp_path, task_id)["status"] == "scheduled"
    assert projection.read_bytes() == corrupt_bytes
    persisted = json.loads(snapshot_path.read_text(encoding="utf-8"))
    persisted_task = persisted["pending"][0]["task"]
    assert isinstance(persisted_task.get("_cancel_intent_authority_hold"), dict)
    assert not isinstance(persisted_task.get("_terminalization_retry"), dict)

    # Repairing the projection releases this authority hold; it must not turn
    # the ordinary scheduled row into a synthetic terminal failure.
    from supervisor import state, workers

    delivered: list[dict] = []
    worker = types.SimpleNamespace(
        wid=1,
        busy_task_id=None,
        reaping=False,
        in_q=types.SimpleNamespace(put=lambda row: delivered.append(dict(row))),
    )
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", running)
    monkeypatch.setattr(workers, "WORKERS", {1: worker})
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda _task: True)
    monkeypatch.setattr(state, "budget_remaining", lambda *_args, **_kwargs: 100.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    queue.BUDGET_ROOT_FENCES.clear()
    projection.unlink()

    workers.assign_tasks()

    assert pending == []
    assert [row["id"] for row in delivered] == [task_id]
    assert "_terminalization_retry" not in delivered[0]
    assert "_cancel_intent_authority_hold" not in delivered[0]
    assert load_task_result(tmp_path, task_id)["status"] == "scheduled"

def test_cancel_authority_hold_never_releases_a_terminal_row_to_dispatch(
    tmp_path, monkeypatch,
):
    """A repaired hold distinguishes ordinary resume from terminal cleanup."""
    from supervisor import queue, state, workers

    task_id = "terminal-under-authority-hold"
    task = {
        "id": task_id,
        "type": "task",
        "chat_id": 1,
        "depth": 0,
        "_cancel_intent_authority_hold": {
            "reason": "Cancel-intent authority is unreadable; dispatch is blocked.",
            "held_at": "2026-08-28T00:00:00+00:00",
        },
    }
    pending = [task]
    running: dict = {}
    delivered: list[dict] = []
    worker = types.SimpleNamespace(
        wid=1,
        busy_task_id=None,
        reaping=False,
        in_q=types.SimpleNamespace(put=lambda row: delivered.append(dict(row))),
    )
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", running)
    monkeypatch.setattr(workers, "WORKERS", {1: worker})
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda _task: True)
    monkeypatch.setattr(state, "budget_remaining", lambda *_args, **_kwargs: 100.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr(
        workers,
        "_emit_task_done_terminal",
        lambda *_args, **_kwargs: pytest.fail("authority hold does not own terminal events"),
    )
    queue.BUDGET_ROOT_FENCES.clear()
    write_task_result(tmp_path, task_id, STATUS_CANCELLED, result="cancelled")

    workers.assign_tasks()

    assert pending == []
    assert delivered == []
    assert worker.busy_task_id is None
    assert load_task_result(tmp_path, task_id)["status"] == STATUS_CANCELLED

def test_preserve_pending_shutdown_keeps_cancel_authority_hold_nonterminal(
    qenv, monkeypatch,
):
    """A restart hold is queue custody, never a synthetic failed outcome."""
    from supervisor import queue, workers

    task_id = "held-through-planned-restart"
    held = {
        "id": task_id,
        "type": "task",
        "chat_id": 0,
        "depth": 0,
        "_cancel_intent_authority_hold": {
            "reason": "Cancel-intent authority is unreadable; dispatch is blocked.",
            "held_at": "2026-08-28T00:00:00+00:00",
        },
    }
    qenv.q.PENDING[:] = [held]
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    write_task_result(qenv.drive, task_id, "scheduled")

    assert workers.kill_workers(
        preserve_pending=True,
        reconcile_delegate_custody=False,
    ) is True

    assert qenv.q.PENDING == [held]
    assert "_terminalization_retry" not in qenv.q.PENDING[0]
    assert load_task_result(qenv.drive, task_id)["status"] == "scheduled"

def test_steering_is_refused_while_a_cancel_intent_is_active(tmp_path, monkeypatch):
    """A1.8: no NEW steering writes into a task whose cancellation is pending."""
    import supervisor.events as events_mod
    from ouroboros.owner_mailbox import drain_owner_messages
    from supervisor.events import _handle_steer_task

    ci.request_cancel(tmp_path, "steerme", reason="tearing down")
    receipts: list = []
    monkeypatch.setattr(
        events_mod, "_emit_routing_receipt",
        lambda ctx, evt, **kw: receipts.append(kw) or {},
    )
    sent: list = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"steerme": {"task": {"id": "steerme", "chat_id": 1}}},
        PENDING=[],
        get_chat_agent=lambda: None,
        send_with_budget=lambda *a, **k: sent.append(a),
        persist_queue_snapshot=lambda **_kw: True,
    )
    _handle_steer_task(
        {"target_task_id": "steerme", "message": "new orders", "chat_id": 1}, ctx,
    )
    assert receipts and receipts[0]["status"] == "rejected"
    assert receipts[0]["reason"] == "cancel_pending"
    assert drain_owner_messages(tmp_path, "steerme") == []
    assert sent and "cancellation is pending" in sent[0][1]

def test_drop_cancelled_pending_stamps_the_decision_and_honors_the_stored_status(
    qenv, monkeypatch,
):
    """A-F4: the pre-assignment drop follows custody's rules."""
    from supervisor import workers

    emitted: list = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal",
                        lambda task, tid, status, **kw: emitted.append((tid, status)) or True)
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)

    qenv.q.PENDING[:] = [
        {"id": "drop-decided", "chat_id": 1},
        {"id": "drop-completed", "chat_id": 1},
    ]
    write_task_result(qenv.drive, "drop-decided", "scheduled")
    ci.request_cancel(qenv.drive, "drop-decided", reason="parent stopped the plan",
                      requested_by="parent7")
    # This one finished on its own between the intent and the drop.
    write_task_result(qenv.drive, "drop-completed", "scheduled")
    ci.request_cancel(qenv.drive, "drop-completed")
    write_task_result(qenv.drive, "drop-completed", STATUS_COMPLETED, result="won the race")

    workers._drop_cancelled_pending()

    decided = load_task_result(qenv.drive, "drop-decided")
    assert decided["status"] == STATUS_CANCELLED
    assert decided["parent_decision"] == "cancelled"
    assert decided["parent_decision_reason"] == "parent stopped the plan"
    # Completion wins: the stored status is what the card resolves to.
    assert load_task_result(qenv.drive, "drop-completed")["status"] == STATUS_COMPLETED
    assert ("drop-completed", STATUS_COMPLETED) in emitted
    assert ("drop-decided", STATUS_CANCELLED) in emitted

def test_drop_cancelled_pending_retains_custody_until_task_done_is_published(
    qenv, monkeypatch,
):
    """A durable cancellation without task_done remains non-dispatchable."""
    from supervisor import workers

    emitted: list = []
    publish_results = iter([False, True])
    snapshots: list[str] = []

    def publish(_task, task_id, status, **_kwargs):
        emitted.append((task_id, status))
        return next(publish_results)

    monkeypatch.setattr(workers, "_emit_task_done_terminal", publish)
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)
    monkeypatch.setattr(
        qenv.q,
        "persist_queue_snapshot",
        lambda reason="": snapshots.append(reason) or True,
    )
    qenv.q.PENDING[:] = [{"id": "drop-event-gap", "chat_id": 1, "depth": 0}]
    write_task_result(qenv.drive, "drop-event-gap", "scheduled")
    ci.request_cancel(
        qenv.drive,
        "drop-event-gap",
        reason="parent stopped the plan",
        requested_by="parent7",
    )

    workers._drop_cancelled_pending()

    assert [row["id"] for row in qenv.q.PENDING] == ["drop-event-gap"]
    retry = qenv.q.PENDING[0]["_terminalization_retry"]
    assert retry["status"] == STATUS_CANCELLED
    assert retry["trigger"] == "pending_cancel_event"
    assert retry["reconcile_delegate_custody"] is False
    assert snapshots == ["pending_terminal_event_retry"]
    stored = load_task_result(qenv.drive, "drop-event-gap")
    assert stored["status"] == STATUS_CANCELLED
    assert stored["parent_decision"] == "cancelled"
    assert ci.active_intent(qenv.drive, "drop-event-gap") is None

    assert workers._retry_terminalization_pending() == (["drop-event-gap"], [])
    assert qenv.q.PENDING == []
    assert emitted == [
        ("drop-event-gap", STATUS_CANCELLED),
        ("drop-event-gap", STATUS_CANCELLED),
    ]

@pytest.mark.parametrize("settle_failure", ["returns_none", "raises"])
def test_drop_cancelled_pending_releases_a_failed_intent_claim(
    qenv, monkeypatch, settle_failure,
):
    """A failed intent settle cannot strand a claim while event custody retries."""
    from supervisor import workers

    emitted: list[tuple[str, str]] = []
    publish_results = iter([False, True])

    def publish(_task, task_id, status, **_kwargs):
        emitted.append((task_id, status))
        return next(publish_results)

    monkeypatch.setattr(workers, "_emit_task_done_terminal", publish)
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)
    monkeypatch.setattr(
        qenv.q,
        "persist_queue_snapshot",
        lambda reason="": True,
    )

    def failed_settle(*_args, **_kwargs):
        if settle_failure == "raises":
            raise OSError("intent projection temporarily unavailable")
        return None

    released: list[dict] = []
    real_release = ci.release_claim

    def release_claim(root, task_id, **kwargs):
        released.append({"task_id": task_id, **kwargs})
        return real_release(root, task_id, **kwargs)

    monkeypatch.setattr(ci, "settle_intent", failed_settle)
    monkeypatch.setattr(ci, "release_claim", release_claim)

    task_id = "drop-settle-gap"
    qenv.q.PENDING[:] = [{"id": task_id, "chat_id": 1, "depth": 0}]
    write_task_result(qenv.drive, task_id, "scheduled")
    ci.request_cancel(qenv.drive, task_id, reason="parent stopped the plan")

    workers._drop_cancelled_pending()

    assert released and released[0]["task_id"] == task_id
    intent = ci.active_intent(qenv.drive, task_id)
    assert intent is not None
    assert intent["state"] == ci.INTENT_REQUESTED
    assert "claim_owner" not in intent
    assert intent["last_error"] == "pending-drop intent settlement failed"
    retry = qenv.q.PENDING[0]["_terminalization_retry"]
    assert retry["status"] == STATUS_CANCELLED
    assert retry["trigger"] == "pending_cancel_event"
    assert retry["reconcile_delegate_custody"] is False

    # The durable result/event retry is independent of the re-opened intent;
    # the watchdog can settle that intent on its next custody pass.
    assert workers._retry_terminalization_pending() == ([task_id], [])
    assert qenv.q.PENDING == []
    assert emitted == [(task_id, STATUS_CANCELLED), (task_id, STATUS_CANCELLED)]
    intent = ci.active_intent(qenv.drive, task_id)
    assert intent is not None and intent["state"] == ci.INTENT_REQUESTED

def test_drop_cancelled_pending_does_not_assume_settled_when_settle_helper_missing(
    qenv, monkeypatch,
):
    """A missing settle helper must retain the active claim for a later retry."""
    from supervisor import workers

    task_id = "drop-missing-settle"
    emitted: list[tuple[str, str]] = []
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)
    monkeypatch.setattr(
        workers,
        "_emit_task_done_terminal",
        lambda _task, tid, status, **_kwargs: emitted.append((tid, status)) or True,
    )
    monkeypatch.setattr(qenv.q, "persist_queue_snapshot", lambda reason="": True)

    qenv.q.PENDING[:] = [{"id": task_id, "chat_id": 1, "depth": 0}]
    write_task_result(qenv.drive, task_id, "scheduled")
    ci.request_cancel(qenv.drive, task_id, reason="parent stopped the plan")

    real_settle = ci.settle_intent
    real_release = ci.release_claim
    monkeypatch.setattr(ci, "settle_intent", None)
    monkeypatch.setattr(ci, "release_claim", lambda *_args, **_kwargs: False)

    workers._drop_cancelled_pending()

    assert [row["id"] for row in qenv.q.PENDING] == [task_id]
    retry = qenv.q.PENDING[0]["_terminalization_retry"]
    assert retry["trigger"] == "pending_cancel_intent"
    assert retry["event_published"] is True
    intent = ci.active_intent(qenv.drive, task_id)
    assert intent is not None and intent["state"] == ci.INTENT_CLAIMED
    assert emitted == [(task_id, STATUS_CANCELLED)]

    # Restore the real helpers: the retained claim and marker can now finish
    # without emitting a second terminal event.
    monkeypatch.setattr(ci, "settle_intent", real_settle)
    monkeypatch.setattr(ci, "release_claim", real_release)
    assert workers._retry_terminalization_pending() == ([task_id], [])
    assert qenv.q.PENDING == []
    intent = ci.active_intent(qenv.drive, task_id)
    assert intent is not None and intent["state"] == ci.INTENT_REQUESTED
    assert emitted == [(task_id, STATUS_CANCELLED)]

def test_drop_cancelled_pending_leaves_the_intent_open_when_the_write_fails(
    qenv, monkeypatch,
):
    """A-F4: never publish a cancellation that is not on disk."""
    from supervisor import workers

    emitted: list = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal",
                        lambda task, tid, status, **kw: emitted.append((tid, status)) or True)
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)
    monkeypatch.setattr(
        "ouroboros.task_results.write_task_result",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")),
    )
    qenv.q.PENDING[:] = [{"id": "drop-nowrite", "chat_id": 1}]
    write_task_result(qenv.drive, "drop-nowrite", "scheduled")
    ci.request_cancel(qenv.drive, "drop-nowrite")

    workers._drop_cancelled_pending()

    assert qenv.q.PENDING == [], "it must not be assigned to a worker"
    assert emitted == [], "no task_done for a cancellation that never persisted"
    assert ci.active_intent(qenv.drive, "drop-nowrite") is not None

def test_steering_refusal_covers_the_legacy_latch_too(tmp_path, monkeypatch):
    """A-F19: a pre-migration wedged task must not accept new owner messages."""
    import supervisor.events as events_mod
    from ouroboros.owner_mailbox import drain_owner_messages
    from supervisor.events import _handle_steer_task

    write_task_result(tmp_path, "legacy-steer", STATUS_CANCEL_REQUESTED, result="wedged")
    receipts: list = []
    monkeypatch.setattr(events_mod, "_emit_routing_receipt",
                        lambda ctx, evt, **kw: receipts.append(kw) or {})
    sent: list = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"legacy-steer": {"task": {"id": "legacy-steer", "chat_id": 1}}},
        PENDING=[],
        get_chat_agent=lambda: None,
        send_with_budget=lambda *a, **k: sent.append(a),
        persist_queue_snapshot=lambda **_kw: True,
    )
    _handle_steer_task(
        {"target_task_id": "legacy-steer", "message": "new orders", "chat_id": 1}, ctx,
    )
    assert receipts and receipts[0]["reason"] == "cancel_pending"
    assert drain_owner_messages(tmp_path, "legacy-steer") == []

def test_drop_cancelled_pending_yields_to_a_live_claim_owner(qenv, monkeypatch):
    """AR2-2: the pre-assignment drop CLAIMS before it settles. A live custody's
    claim wins — assignment retains the authoritative pending row and aborts,
    then the claim owner captures/removes that same row."""
    from supervisor import workers

    emitted: list = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal",
                        lambda task, tid, status, **kw: emitted.append((tid, status)) or True)
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)
    qenv.q.PENDING[:] = [{"id": "drop-owned", "chat_id": 1}]
    write_task_result(qenv.drive, "drop-owned", "scheduled")
    ci.request_cancel(qenv.drive, "drop-owned")
    ci.claim_intent(qenv.drive, "drop-owned", owner="cancel_task_custody")  # live owner

    assert workers._drop_cancelled_pending() is False

    assert [row["id"] for row in qenv.q.PENDING] == ["drop-owned"]
    assert emitted == [], "the claim owner emits, not the drop"
    assert load_task_result(qenv.drive, "drop-owned")["status"] == "scheduled"
    intent = ci.active_intent(qenv.drive, "drop-owned")
    assert intent["state"] == ci.INTENT_CLAIMED
    assert intent["claim_owner"] == "cancel_task_custody"

    assert ci.release_claim(qenv.drive, "drop-owned", error="test owner resumes")
    assert qenv.tl.cancel_task_custody("drop-owned") == qenv.tl.CANCEL_CANCELLED
    assert qenv.q.PENDING == []

def test_drop_cancelled_pending_defers_when_intent_vanishes_before_settle(
    qenv, monkeypatch,
):
    """A changed claim aborts the whole assignment pass, not just the drop."""
    from supervisor import queue, state, workers

    task_id = "drop-intent-race"
    pending = [{"id": task_id, "type": "task", "chat_id": 1, "depth": 0}]
    delivered: list[dict] = []
    worker = types.SimpleNamespace(
        wid=1,
        busy_task_id=None,
        reaping=False,
        in_q=types.SimpleNamespace(put=lambda task: delivered.append(dict(task))),
    )
    monkeypatch.setattr(workers, "PENDING", pending, raising=False)
    monkeypatch.setattr(workers, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {1: worker}, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(
        state,
        "budget_remaining",
        lambda *_args, **_kwargs: pytest.fail(
            "budget must not run after cancellation custody changes"
        ),
    )
    snapshots: list[str] = []
    monkeypatch.setattr(
        queue,
        "persist_queue_snapshot",
        lambda reason="": snapshots.append(reason) or True,
    )
    queue.BUDGET_ROOT_FENCES.clear()
    write_task_result(qenv.drive, task_id, "scheduled")
    first = ci.request_cancel(qenv.drive, task_id, reason="old request")
    real_settle = ci.settle_intent

    def vanished_claim(root, tid, *, owner):
        assert owner == "pending_drop"
        real_settle(
            root, tid, outcome="cancelled",
            expected_generation=first["generation"], request_id=first["request_id"],
        )
        ci.request_cancel(root, tid, reason="new request", source="race")
        return None

    monkeypatch.setattr(ci, "claim_intent", vanished_claim)
    workers.assign_tasks()

    assert [row["id"] for row in pending] == [task_id]
    assert delivered == []
    assert worker.busy_task_id is None
    assert load_task_result(qenv.drive, task_id)["status"] == "scheduled"
    replacement = ci.active_intent(qenv.drive, task_id)
    assert replacement is not None
    assert replacement["state"] == ci.INTENT_REQUESTED
    assert replacement["reason"] == "new request"
    assert snapshots == ["cancellation_authority_indeterminate"]

def test_snapshot_restore_consults_the_intent_projection_under_the_queue_lock(
    qenv, monkeypatch,
):
    """AR2-10 (§8-A1): the projection read at restore holds the queue lock, so
    the "no active intent" view and the enqueue are one serialized step."""
    from ouroboros.utils import utc_now_iso

    consults: list = []

    def _spy(root, tid, *, strict=False):
        consults.append(qenv.q._queue_lock._is_owned())
        return True  # refusal path: no enqueue side effects in this harness

    monkeypatch.setattr("ouroboros.cancel_intents.has_active_intent", _spy)
    snapshot = {
        "ts": utc_now_iso(),
        "pending": [{"task": {"id": "locked-restore", "chat_id": 1, "type": "chat"}}],
        "running": [],
        "acceptance_fences": [],
        "budget_root_fences": [],
    }
    state_dir = qenv.drive / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(qenv.q, "QUEUE_SNAPSHOT_PATH", state_dir / "queue_snapshot.json",
                        raising=False)

    assert qenv.q.restore_pending_from_snapshot() == 0
    assert qenv.q.PENDING == []
    assert consults == [True], "the intent consult must hold the queue lock"

def test_steer_refusal_removes_the_just_staged_attachments(tmp_path, monkeypatch):
    """GR2-9: a steering message refused by the transactional cancel re-check
    must not leave its just-staged input files in the dying task's store."""
    import supervisor.events as events_mod
    import supervisor.queue as queue_mod
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    write_task_result(tmp_path, "steer-stage", STATUS_RUNNING, result="working")
    source = tmp_path / "owner-input.txt"
    source.write_text("owner attachment", encoding="utf-8")

    # The up-front check passes (no cancel yet); the cancel ingress lands in the
    # window before the transactional re-check — exactly the staged-then-refused
    # shape the fix removes.
    checks = {"n": 0}
    real_pending = ci.cancel_pending

    def _racing_cancel_pending(root, tid):
        checks["n"] += 1
        if checks["n"] == 2 and tid == "steer-stage":
            ci.request_cancel(tmp_path, "steer-stage", reason="race")
        return real_pending(root, tid)

    monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", _racing_cancel_pending)
    receipts: list = []
    monkeypatch.setattr(events_mod, "_emit_routing_receipt",
                        lambda ctx, evt, **kw: receipts.append(kw) or {})
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"steer-stage": {"task": {"id": "steer-stage", "chat_id": 1}}},
        PENDING=[],
        get_chat_agent=lambda: None,
        send_with_budget=lambda *a, **k: None,
        persist_queue_snapshot=lambda **_kw: True,
    )
    _handle_steer_task(
        {"target_task_id": "steer-stage", "message": "new orders", "chat_id": 1,
         "attachment_uploads": [{"path": str(source), "label": "input"}]},
        ctx,
    )

    assert receipts and receipts[-1]["reason"] == "cancel_pending"
    from ouroboros.artifacts import task_artifact_dir_path

    attach_dir = task_artifact_dir_path(tmp_path, "steer-stage") / "attachments"
    staged = list(attach_dir.glob("*")) if attach_dir.exists() else []
    assert staged == [], f"staged inputs must be removed on refusal: {staged}"
    from ouroboros.owner_mailbox import drain_owner_messages

    assert drain_owner_messages(tmp_path, "steer-stage") == []
