"""Cancellation observations describe sources without claiming caller prose is fact."""

import json
import queue
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import cancel_intents, delegate_custody
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import observe_cancellation_target
from ouroboros.tools.join_ledger import _cancel_task
from ouroboros.utils import atomic_write_json, utc_now_iso


def _queue(root, task_id, *, stale=False):
    atomic_write_json(root / "state" / "queue_snapshot.json", {
        "ts": "2000-01-01T00:00:00Z" if stale else utc_now_iso(),
        "running": [{"id": task_id}], "pending": [],
    })


def _caller(root):
    return SimpleNamespace(task_id="parent", drive_root=root, pending_events=[], event_queue=queue.Queue(),
                           task_metadata={"root_task_id": "parent", "delegation_role": "root"})


def test_agent_cancel_preserves_recorded_start_and_caller_reason_separately(tmp_path):
    write_task_result(tmp_path, "child", "running", parent_task_id="parent", root_task_id="parent", delegation_role="subagent",
                      started_at="2026-09-06T01:00:00+00:00", accounted_upper_bound_usd=3.25, cost_final=False)
    _queue(tmp_path, "child")
    delegate_custody.emit(tmp_path, delegate_custody.STARTED, {"run_id": "r1", "task_id": "child"})
    reason = "I think it never started and cost nothing"
    response = _cancel_task(_caller(tmp_path), "child", reason)
    assert response.startswith("Cancel requested:") and "cancel_state=pending" in response
    intent = cancel_intents.active_intent(tmp_path, "child")
    observation = intent["observation"]
    assert intent["reason"] == reason
    assert observation["observed_task_id"] == "child"
    assert observation["matches_cancel_target"] is True
    assert observation["queue_snapshot"]["status"] == "running"
    assert observation["queue_snapshot"]["fresh"] is True
    assert observation["task_result"]["started_at"] == "2026-09-06T01:00:00+00:00"
    assert observation["task_result"]["cost"]["accounted_upper_bound_usd"] == 3.25
    assert observation["delegated_execution"]["delegated_runs_started"] == 1
    assert observation["request_origin"] == {"kind": "agent_task", "task_id": "parent"}
    assert "separate from the caller's reason" in response
    assert '"delegated_runs_started": 1' in response
    assert load_task_result(tmp_path, "child")["status"] == "running"
    events = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()]
    assert next(row for row in events if row.get("event") == "requested")["observation"] == observation
    from supervisor.cancel_publication import _intent_outcome_fields
    fields = _intent_outcome_fields(intent)
    assert fields["parent_decision_reason"] == reason and fields["cancel_observation"] == observation


def test_stale_snapshot_is_unknown_not_the_fail_open_live_hint(tmp_path):
    write_task_result(tmp_path, "child", "running")
    _queue(tmp_path, "child", stale=True)
    from ouroboros.task_status import task_has_live_queue_ownership
    assert task_has_live_queue_ownership(tmp_path, "child") is True
    observation = observe_cancellation_target(tmp_path, "child")
    assert observation["queue_snapshot"] == {"status": "unknown", "ts": "2000-01-01T00:00:00Z", "fresh": False}
    assert observation["task_result"]["started_at"] is None


def _retry(root):
    write_task_result(root, "original", "interrupted", root_task_id="original", delegation_role="root",
                      retry_task_id="retry", superseded_by="retry")
    write_task_result(root, "retry", "running", root_task_id="original", delegation_role="root",
                      original_task_id="original", timeout_retry_from="original", supersedes_task_id="original")
    _queue(root, "retry")


def test_observation_names_resolved_retry_and_discloses_a_later_target_change(tmp_path):
    _retry(tmp_path)
    observation = observe_cancellation_target(tmp_path, "original")
    assert observation["requested_task_id"] == "original" and observation["observed_task_id"] == "retry"
    intent = cancel_intents.request_cancel(tmp_path, "original", observation=observation)
    assert intent["task_id"] == "retry" and intent["observation"]["matches_cancel_target"] is True
    # Widening an existing leaf intent rekeys it to the logical cascade root;
    # it keeps the observed physical identity instead of rewriting that fact.
    widened = cancel_intents.request_cancel(tmp_path, "original", scope="cascade", observation=observation)
    assert widened["task_id"] == "original"
    assert widened["observation"]["observed_task_id"] == "retry"
    assert widened["observation"]["matches_cancel_target"] is False


def test_completed_child_keeps_its_result(tmp_path):
    write_task_result(tmp_path, "child", "completed", parent_task_id="parent", root_task_id="parent", delegation_role="subagent", result="finished work")
    atomic_write_json(tmp_path / "state" / "queue_snapshot.json", {"ts": utc_now_iso(), "running": [], "pending": []})
    response = _cancel_task(_caller(tmp_path), "child", "unneeded")
    assert response.startswith("Nothing to cancel:")
    assert cancel_intents.active_intent(tmp_path, "child") is None
    assert load_task_result(tmp_path, "child")["result"] == "finished work"


@pytest.mark.parametrize("cascade", [False, True])
def test_http_observation_records_transport_without_inventing_owner_identity(tmp_path, monkeypatch, cascade):
    from ouroboros.gateway import tasks
    from supervisor import queue as task_queue

    write_task_result(tmp_path, "task", "running")
    _queue(tmp_path, "task")
    monkeypatch.setattr(task_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(task_queue, "task_has_live_ownership", lambda task_id: True)
    monkeypatch.setattr(task_queue, "task_subtree_is_live", lambda task_id: True)
    monkeypatch.setattr(task_queue, "cancel_task_custody", lambda task_id: task_queue.CANCEL_CANCELLED)
    monkeypatch.setattr(tasks, "_run_cascade_cancel", lambda task_id: True)
    monkeypatch.setattr("ouroboros.delegate_evidence.task_execution_evidence", lambda *_a: pytest.fail("HTTP must not scan delegated descendants"))
    app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", tasks.api_task_cancel, methods=["POST"])])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        response = client.post("/api/tasks/task/cancel", json={"cascade": cascade})
    assert response.status_code == 200
    intent = cancel_intents.active_intent(tmp_path, "task")
    assert intent["requested_by"] == ""
    assert intent["observation"]["request_origin"] == {
        "kind": "http_client", "source": "http_cascade" if cascade else "http_single"}
    assert "delegated_execution" not in intent["observation"]
