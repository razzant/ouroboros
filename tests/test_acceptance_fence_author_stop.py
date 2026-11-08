"""A saved unfinished stop does not turn its admission seal into subtree cancellation."""

from __future__ import annotations

import json

import pytest

from ouroboros import cancel_intents
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import utc_now_iso
from tests._cancel_intents_shared import qenv as _qenv
from tests._store_shared import result_signature

qenv = _qenv


def _stopped_parent_snapshot(qenv, monkeypatch, *, saved=True, status="sealed",
                             outcome="author_stop", budget=False):
    root_id, child_id = "stopping-root", "queued-child"
    write_task_result(
        qenv.drive, root_id, "completed" if saved else "running", chat_id=1,
        result="Stopping with unfinished work.",
        outcome_axes={"objective": {"status": "fail", "reason": "author_stop"}},
    )
    child = {
        "id": child_id, "type": "task", "text": "finish assigned work", "chat_id": 1,
        "root_task_id": root_id, "parent_task_id": root_id, "delegation_role": "subagent",
    }
    write_task_result(qenv.drive, child_id, "scheduled", **{
        key: value for key, value in child.items() if key != "id"
    })
    path = qenv.drive / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ts": utc_now_iso(),
        "pending": [{"task": child}],
        # The terminal result was saved, but task_done has not retired this row/seal.
        "running": [{"id": root_id, "task": {"id": root_id, "chat_id": 1}}],
        "acceptance_fences": [{
            "token": "a" * 32, "root_task_id": root_id, "task_id": root_id,
            "status": status, "outcome": outcome,
        }],
        "budget_root_fences": [{
            "fence_id": "budget-stop", "root_task_id": root_id, "status": "paused",
        }] if budget else [],
    }), encoding="utf-8")
    monkeypatch.setattr(qenv.q, "QUEUE_SNAPSHOT_PATH", path)
    monkeypatch.setattr(qenv.q, "PRIOR_DIRECT_ROOTS", {})
    return root_id, child_id


def test_saved_author_stop_restores_queued_child_without_cancelling_it(qenv, monkeypatch):
    root_id, child_id = _stopped_parent_snapshot(qenv, monkeypatch)
    parent_before = result_signature(qenv.drive, root_id)

    fenced = []
    assert qenv.q.restore_pending_from_snapshot(terminalized=fenced) == 1
    assert fenced == []
    assert [row["id"] for row in qenv.q.PENDING] == [child_id]
    assert "_terminalization_retry" not in qenv.q.PENDING[0]
    assert load_task_result(qenv.drive, child_id)["status"] == "scheduled"
    assert cancel_intents.active_intent(qenv.drive, root_id) is None
    assert cancel_intents.active_intent(qenv.drive, child_id) is None
    assert result_signature(qenv.drive, root_id) == parent_before


@pytest.mark.parametrize("status,outcome", [
    ("active", "author_stop"), ("active", "terminal"),
    ("sealed", "terminal"),
])
def test_only_the_sealed_author_stop_changes_acceptance_restore(
    qenv, monkeypatch, status, outcome,
):
    _root_id, child_id = _stopped_parent_snapshot(
        qenv, monkeypatch, status=status, outcome=outcome,
    )

    assert qenv.q.restore_pending_from_snapshot() == 0
    assert qenv.q.PENDING == []
    child = load_task_result(qenv.drive, child_id)
    assert child["status"] == "cancelled"
    assert "root had entered acceptance review" in child["result"]


def test_stop_seal_before_terminal_save_preserves_shutdown_custody(qenv, monkeypatch):
    root_id, child_id = _stopped_parent_snapshot(qenv, monkeypatch, saved=False)

    fenced = []
    assert qenv.q.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == [root_id]
    intent = cancel_intents.active_intent(qenv.drive, root_id)
    assert (intent["reason"], intent["source"]) == ("server_shutdown", "snapshot_restore")
    assert [row["id"] for row in qenv.q.PENDING] == [child_id]
    assert qenv.q.PENDING[0]["_terminalization_retry"]["trigger"] == "pending_parent_interrupted"
    assert load_task_result(qenv.drive, child_id)["status"] == "scheduled"


def test_saved_stop_does_not_clear_a_child_cancel_intent(qenv, monkeypatch):
    _root_id, child_id = _stopped_parent_snapshot(qenv, monkeypatch)
    intent = cancel_intents.request_cancel(qenv.drive, child_id, reason="owner pressed Stop")

    assert qenv.q.restore_pending_from_snapshot() == 0
    assert qenv.q.PENDING == []
    assert cancel_intents.active_intent(qenv.drive, child_id)["request_id"] == intent["request_id"]


def test_saved_stop_does_not_resume_a_budget_fenced_child(qenv, monkeypatch):
    from supervisor.queue_transitions import budget_pause_fact

    root_id, child_id = _stopped_parent_snapshot(qenv, monkeypatch, budget=True)

    assert qenv.q.restore_pending_from_snapshot() == 1
    assert [row["id"] for row in qenv.q.PENDING] == [child_id]
    assert budget_pause_fact(qenv.q.PENDING[0])["root_task_id"] == root_id
    assert load_task_result(qenv.drive, child_id)["status"] == "scheduled"
    assert qenv.q.BUDGET_ROOT_FENCES[root_id]["status"] == "paused"
    assert qenv.q.BUDGET_ROOT_FENCES[root_id]["auto_resume"] is False
