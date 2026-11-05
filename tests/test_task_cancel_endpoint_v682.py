"""HTTP contract for /api/tasks/{id}/cancel with the optional cascade body (v6.82 P5).

Both the plain (no-body / empty-body) request and {"cascade": true} record the
durable intent FIRST — no liveness pre-probe gates it — and drive it through
the canonical ``drive_cancel_intent_scope`` before answering: 200 with the
pre-v6.82 single-task envelope (plus "cascade": true) once the teardown
settled, the legacy 404 when there is nothing to cancel, and 202
``cancel_state: "pending"`` while the recorded intent stays open for the
cancel-intent watchdog. 503 means nothing was recorded.
"""

from __future__ import annotations

import pytest

import json

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import api_task_cancel


def _isolate_queue(monkeypatch, tmp_path, tasks):
    from supervisor import queue
    from supervisor import workers

    pending = [dict(task) for task in tasks]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    return queue, pending


def _client(tmp_path):
    app = Starlette(routes=[
        Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"]),
    ])
    app.state.drive_root = tmp_path
    return TestClient(app)


def test_cancel_404_contract_both_modes(tmp_path, monkeypatch):
    _isolate_queue(monkeypatch, tmp_path, [])
    with _client(tmp_path) as client:
        plain = client.post("/api/tasks/ghost/cancel")
        cascade = client.post("/api/tasks/ghost/cancel", json={"cascade": True})
    assert plain.status_code == 404
    assert cascade.status_code == 404
    assert cascade.json()["error"] == "task not found or not active"


def test_a_failing_teardown_is_reported_and_recorded_never_swallowed(tmp_path, monkeypatch):
    queue, _pending = _isolate_queue(
        monkeypatch,
        tmp_path,
        [{"id": "root", "chat_id": 0, "root_task_id": "root"}],
    )

    def _boom(task_id):
        raise RuntimeError("teardown exploded")

    monkeypatch.setattr(queue, "cancel_task_by_id", _boom)
    with _client(tmp_path) as client:
        response = client.post("/api/tasks/root/cancel", json={"cascade": True})
    # The cascade is awaited, so a teardown that explodes is REPORTED (never an
    # ok:true cancellation that did not happen): the intent was recorded first,
    # so the answer is the typed pending Stop the watchdog finishes, and the
    # failure is also a durable owner-visible incident. The server does not crash.
    assert response.status_code == 202
    assert response.json() == {
        "ok": True, "task_id": "root", "cancel_state": "pending", "stop_policy": "immediate",
        "request_id": response.json()["request_id"], "reason_code": "teardown_unconfirmed",
        "cascade": True,
    }
    from ouroboros.cancel_intents import active_intent

    assert (active_intent(tmp_path, "root") or {}).get("request_id") == response.json()["request_id"]
    log_path = tmp_path / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = [row for row in rows if row.get("type") == "task_cancel_cascade_error"]
    assert errors and errors[0]["task_id"] == "root"
    assert "teardown exploded" in errors[0]["error"]


def test_non_boolean_cascade_is_a_client_error_not_a_subtree_cancel(tmp_path, monkeypatch):
    """`{"cascade": "false"}` must never select the destructive path — a non-empty
    string is truthy in Python, so the flag is parsed STRICTLY and a non-boolean
    value is a 400 rather than a silent single-task (or worse, subtree) cancel."""
    queue, pending = _isolate_queue(
        monkeypatch,
        tmp_path,
        [
            {"id": "root", "chat_id": 0, "root_task_id": "root"},
            {"id": "child", "chat_id": 0, "root_task_id": "root", "parent_task_id": "root"},
        ],
    )
    with _client(tmp_path) as client:
        for bad in ("false", "true", 1, 0):
            response = client.post("/api/tasks/root/cancel", json={"cascade": bad})
            assert response.status_code == 400, (bad, response.text)
            assert "cascade must be a boolean" in response.text
    # Nothing was cancelled by any of the rejected requests.
    assert [task["id"] for task in pending] == ["root", "child"]


@pytest.fixture(autouse=True)
def _isolate_cancellation_globals_endpoint(monkeypatch, tmp_path):
    """Same module-global rule as tests/test_cancel_cascade_v664.py: these tests
    drive the real cascade path, so the process-global fence registries must be
    reset around each one or a later test enqueuing a task with a common id is
    refused with root_cancelled."""
    import supervisor.queue as queue
    import supervisor.task_lifecycle as tl

    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)


# ---------------------------------------------------------------------------
# Intent first (no liveness pre-probe); a recorded, unconfirmed Stop is pending
# ---------------------------------------------------------------------------


def test_stop_action_replay_joins_the_pending_intent_and_other_arguments_are_409(tmp_path, monkeypatch):
    """The exact retry of one Stop action joins the recorded intent (same request
    id, still pending); the same stop_action_id with different arguments is
    refused 409 and changes nothing."""
    from ouroboros.task_results import write_task_result

    queue, pending = _isolate_queue(monkeypatch, tmp_path, [{"id": "act-1", "chat_id": 0, "root_task_id": "act-1"}])
    write_task_result(tmp_path, "act-1", "scheduled")
    monkeypatch.setattr(queue, "cancel_task_custody", lambda task_id, **_kw: queue.CANCEL_FAILED)
    with _client(tmp_path) as client:
        first = client.post("/api/tasks/act-1/cancel", json={"stop_action_id": "stop-a"})
        replay = client.post("/api/tasks/act-1/cancel", json={"stop_action_id": "stop-a"})
        conflict = client.post("/api/tasks/act-1/cancel", json={"stop_action_id": "stop-a", "cascade": True})
    assert (first.status_code, replay.status_code) == (202, 202)
    assert replay.json()["request_id"] == first.json()["request_id"]
    assert conflict.status_code == 409
    assert conflict.json()["reason_code"] == "stop_action_conflict"
    assert [task["id"] for task in pending] == ["act-1"]
