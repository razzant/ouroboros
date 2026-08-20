"""The queue side of a cancel intent: pending drop, snapshot restore, fail, steer.

Split out of ``tests/test_cancel_intents_phase_a.py`` by theme: the readers that consult
the intent projection before acting, the decisions they stamp, the write failures that
leave an intent open, the deference to a live claim owner, and the steering that is
refused while a cancel is active.
"""

from __future__ import annotations

import json
import types

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


def test_fail_tasks_honors_active_intent(tmp_path):
    from ouroboros.task_results import fail_tasks

    write_task_result(tmp_path, "b1", "scheduled")
    ci.request_cancel(tmp_path, "b1", reason="owner cancel")
    written = fail_tasks(
        tmp_path, [{"id": "b1"}], reason_code="budget_exhausted", result="drained",
    )
    assert written == 1
    assert load_task_result(tmp_path, "b1")["status"] == STATUS_CANCELLED
    assert ci.active_intent(tmp_path, "b1") is None  # settled by the drain


def test_drop_cancelled_pending_consults_the_intent_projection(qenv, monkeypatch):
    from supervisor import workers

    emitted: list = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal",
                        lambda task, tid, status, **kw: emitted.append((tid, status, kw)))
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


def test_snapshot_restore_refuses_a_task_with_active_intent(qenv, monkeypatch):
    from supervisor import state as state_mod
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
    monkeypatch.setattr(state_mod, "QUEUE_SNAPSHOT_PATH", state_dir / "queue_snapshot.json",
                        raising=False)

    restored = qenv.q.restore_pending_from_snapshot()

    assert restored == 0
    assert qenv.q.PENDING == []


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
                        lambda task, tid, status, **kw: emitted.append((tid, status)))
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


def test_drop_cancelled_pending_leaves_the_intent_open_when_the_write_fails(
    qenv, monkeypatch,
):
    """A-F4: never publish a cancellation that is not on disk."""
    from supervisor import workers

    emitted: list = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal",
                        lambda task, tid, status, **kw: emitted.append((tid, status)))
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
    claim wins — the task still leaves the queue (it must not be assigned) but
    nothing is written, settled, or emitted here; the claim owner does all three."""
    from supervisor import workers

    emitted: list = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal",
                        lambda task, tid, status, **kw: emitted.append((tid, status)))
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)
    qenv.q.PENDING[:] = [{"id": "drop-owned", "chat_id": 1}]
    write_task_result(qenv.drive, "drop-owned", "scheduled")
    ci.request_cancel(qenv.drive, "drop-owned")
    ci.claim_intent(qenv.drive, "drop-owned", owner="cancel_task_custody")  # live owner

    workers._drop_cancelled_pending()

    assert qenv.q.PENDING == [], "it must not be assigned to a worker"
    assert emitted == [], "the claim owner emits, not the drop"
    assert load_task_result(qenv.drive, "drop-owned")["status"] == "scheduled"
    intent = ci.active_intent(qenv.drive, "drop-owned")
    assert intent["state"] == ci.INTENT_CLAIMED
    assert intent["claim_owner"] == "cancel_task_custody"


def test_fail_tasks_yields_to_a_live_claim_owner(tmp_path):
    """AR2-2: the budget drain claims before settling; a live custody's claim
    wins and the drain leaves the task entirely to that owner."""
    from ouroboros.task_results import fail_tasks

    write_task_result(tmp_path, "b2", "scheduled")
    ci.request_cancel(tmp_path, "b2")
    ci.claim_intent(tmp_path, "b2", owner="cancel_task_custody")

    written = fail_tasks(
        tmp_path, [{"id": "b2"}], reason_code="budget_exhausted", result="drained",
    )

    assert written == 0
    assert load_task_result(tmp_path, "b2")["status"] == "scheduled"
    assert ci.active_intent(tmp_path, "b2")["claim_owner"] == "cancel_task_custody"


def test_snapshot_restore_consults_the_intent_projection_under_the_queue_lock(
    qenv, monkeypatch,
):
    """AR2-10 (§8-A1): the projection read at restore holds the queue lock, so
    the "no active intent" view and the enqueue are one serialized step."""
    from supervisor import state as state_mod
    from ouroboros.utils import utc_now_iso

    consults: list = []

    def _spy(root, tid):
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
    monkeypatch.setattr(state_mod, "QUEUE_SNAPSHOT_PATH", state_dir / "queue_snapshot.json",
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
