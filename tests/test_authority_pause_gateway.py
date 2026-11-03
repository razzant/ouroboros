"""The real Resume endpoint keeps technical refusal identity and usable owner text."""
import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import api_task_resume


@pytest.mark.parametrize("kind,action,phrase", [
    ("unreadable", "fix_billing_authority_then_resume", "could not be read"),
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


@pytest.mark.parametrize("refusal", [
    {"ok": False, "error": "pause_source_unreadable"},
    {"ok": False, "error": "external_runs_unsettled", "runs": [{"run_id": "r1", "state": "running"}]},
])
def test_resume_preserves_the_same_ordinary_refusal_as_the_mind(monkeypatch, refusal):
    from ouroboros.work_stop import resume_outcome

    outcome = resume_outcome(refusal)
    monkeypatch.setattr("supervisor.queue.resume_budget_paused_task", lambda _id: outcome)
    with TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])) as client:
        response = client.post("/api/tasks/saved/resume")
    assert response.status_code == 409
    assert response.json()["error"] == outcome["error"]
    assert response.json()["message"] == outcome["message"]
    if "blockers" in outcome:
        assert response.json()["blockers"] == outcome["blockers"]


@pytest.mark.serial
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
