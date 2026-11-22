"""The real Resume endpoint keeps technical refusal identity and usable owner text."""
import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import api_task_resume


@pytest.mark.parametrize("kind,action,phrase", [
    ("unreadable", "fix_billing_authority_then_resume", "still could not be read. You can try Resume again later."),
    ("refused", "cancel_or_new_run", "does not authorize"),
])
def test_resume_authority_refusal_is_conflict_not_missing_task(monkeypatch, kind, action, phrase):
    monkeypatch.setattr("supervisor.queue.resume_budget_paused_task", lambda _id: {
        "ok": False, "error": "billing_authority_unavailable", "billing_authority": kind,
        "cause": "synthetic_authority_cause", "action": action,
    })
    with TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])) as client:
        response = client.post("/api/tasks/task-one/resume")
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "billing_authority_unavailable"
    assert body["reason_code"] == "synthetic_authority_cause" and body["action"] == action
    assert phrase in body["error"] and "budget exhaustion was not established" in body["error"]
    assert "increase" not in body["error"].lower()
    assert "fix that cause" not in body["error"].lower()  # the owner cannot diagnose an unreadable record (#1911)


@pytest.mark.parametrize("refusal,status", [
    ({"ok": False, "error": "pause_source_unreadable"}, 409),
    ({"ok": False, "error": "external_runs_unsettled", "runs": [{"run_id": "r1", "state": "running"}]}, 409),
    # A typed refusal of a task that exists is a conflict, never Not Found; only an id the queue does not hold is.
    ({"ok": False, "error": "panic_requires_owner_resume"}, 409),
    ({"ok": False, "error": "stop_not_resumable"}, 409),
    ({"ok": False, "error": "selection_authority_unavailable"}, 409),
    ({"ok": False, "error": "selection_observation_required"}, 409),
    ({"ok": False, "error": "task_not_pending"}, 404),
])
def test_resume_preserves_the_same_ordinary_refusal_as_the_mind(monkeypatch, refusal, status):
    from ouroboros.work_stop import resume_outcome

    outcome = resume_outcome(refusal)
    monkeypatch.setattr("supervisor.queue.resume_budget_paused_task", lambda _id: outcome)
    with TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])) as client:
        response = client.post("/api/tasks/saved/resume")
    assert response.status_code == status
    assert response.json()["error"] == outcome["error"]
    assert response.json()["message"] == outcome["message"]
    if "blockers" in outcome:
        assert response.json()["blockers"] == outcome["blockers"]


def test_the_http_resume_refusal_keeps_the_launch_evidence(monkeypatch):
    from ouroboros.work_stop import resume_outcome

    outcome = resume_outcome({"ok": False, "error": "dispatch_outcome_unknown", "refusal_basis": "dispatch_unconfirmed"})
    monkeypatch.setattr("supervisor.queue.resume_budget_paused_task", lambda _id: outcome)
    with TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])) as client:
        response = client.post("/api/tasks/saved/resume")
    assert 400 <= response.status_code < 500
    assert response.json()["error"] == "dispatch_outcome_unknown"
    assert response.json()["refusal_basis"] == "dispatch_unconfirmed"
    assert response.json()["message"] == "Resume refused: An earlier assignment may already have reached a worker"


def test_actual_authority_resume_success_matches_python_and_browser_contract(tmp_path, monkeypatch):
    import pathlib
    from typing import get_type_hints
    from ouroboros import budget_pause as bp
    from ouroboros.gateway.client_contracts import TaskResumeResponse
    from ouroboros.gateway.schema import validate_ingress
    from tests.test_gateway_parity import _js_typedef_fields
    from tests.test_authority_pause_composition import _park_with_body

    env, _row = _park_with_body(tmp_path, monkeypatch, False)
    try:
        with TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])) as client:
            response = client.post("/api/tasks/sleeper/resume")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] and body["exact_continuation"] and body["task_id"] == "sleeper"
        assert bp.budget_pause_row(tmp_path, "sleeper")["state"] == "resume_granted"
        fields = set(get_type_hints(TaskResumeResponse, include_extras=True))
        assert not validate_ingress(body, TaskResumeResponse) and set(body) <= fields
        js = pathlib.Path(__file__).resolve().parents[1] / "web/modules/client_contract_types.js"
        assert _js_typedef_fields(js.read_text(), "TaskResumeResponse") == fields
    finally:
        env.runtime.close(wait=True)


def test_resume_runs_off_the_web_loop_with_the_owner_door_and_named_unknowns_bound(monkeypatch):
    """C01 PR II: the Resume's queue writes run on a worker thread (``asyncio.to_thread``), so a write waiting on Q
    or the disk never stops the web loop answering others; the owner door and the named held calls stay bound
    where the call runs (a moved Presence turn's allowance reads that door, #2566), and the answer waits for it."""
    import asyncio

    from ouroboros.control_actor import accepted_unknown, owner_door_bound

    seen = {}

    def resume(task_id):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        seen.update(task_id=task_id, door=owner_door_bound().get("door"), unknown=accepted_unknown())
        return {"ok": True, "task_id": task_id, "grant_id": "g-1"}

    monkeypatch.setattr("supervisor.queue.resume_budget_paused_task", resume)
    with TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])) as client:
        response = client.post("/api/tasks/saved/resume", json={"accept_unknown": ["op-1"], "reason": "seen it"})
    assert response.status_code == 200 and response.json()["grant_id"] == "g-1"
    assert seen["on_loop"] is False, "the Resume ran inside the web loop"
    assert seen["task_id"] == "saved" and seen["door"] == "api_task_resume"
    assert seen["unknown"] == {"ids": ["op-1"], "text": "seen it"}
    assert owner_door_bound() == {} and accepted_unknown() == {}  # nothing leaks past the call
