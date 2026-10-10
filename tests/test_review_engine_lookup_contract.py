"""Real engine problem envelopes retain diagnostic and existing-operation custody."""

import json
from pathlib import Path

import httpx
import pytest

from ouroboros import delegate_custody as custody
from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable
from ouroboros.tools.review_response import parse_model_response
from ouroboros.triad_review import parse_model_review_results
from tests._review_session_route_shared import (
    _owned_gateway_uses_each_test_transport as _transport_fixture,
    _run_session_directly,
    fake_route as _route_fixture,
)

_owned_gateway_uses_each_test_transport = _transport_fixture
fake_route = _route_fixture
pytestmark = pytest.mark.serial  # the session fixture mutates its shared transport registry
RESPONSES = json.loads((Path(__file__).parent / "fixtures/review_engine_lookup_problems.json")
                       .read_text(encoding="utf-8"))["responses"]


def _problem(row):
    # _problem is a pure HTTP translator; constructing a gateway would resolve the live endpoint.
    gateway = object.__new__(ClaudexorGateway)
    return gateway._problem(httpx.Response(row["status"], json=row["problem"]))


@pytest.mark.parametrize("row", RESPONSES, ids=lambda r: f'{r["route"]}-{r["scenario"]}')
def test_engine_wire_problem_reaches_the_review_actor_without_reclassification(row):
    error = _problem(row)
    assert error.code == row["problem"]["code"] and error.status_code == row["status"]
    assert error.required_actions == tuple(row["problem"]["requiredActions"])
    envelope = parse_model_response("example/model", {
        "error": str(error), "failure_code": error.code, "http_status": error.status_code,
        "reported_cause": error.reported_cause, "usage": {},
    }, {})
    record = parse_model_review_results({"results": [envelope]}).actor_records[0].to_dict()
    assert record["failure_code"] == error.code and record["http_status"] == error.status_code
    context = row["problem"]["context"]
    if isinstance(context.get("cause"), dict):
        assert record["reported_cause"] == error.reported_cause
        assert context["stage"] in record["reported_cause"]
        if context["cause"].get("code"):
            assert context["cause"]["code"] in record["reported_cause"]
    else:
        assert record.get("reported_cause", "") == ""


@pytest.mark.parametrize("route", ["create", "retry"])
def test_conflict_recovery_keeps_original_key_request_and_registration(tmp_path, fake_route, route):
    row = next(r for r in RESPONSES if r["route"] == route and r["scenario"] == "conflict")
    request = {
        "prompt": "review this", "instructions": "original assignment",
        "authPreference": "subscription", "mode": "ask", "access": "readonly",
        "scope": {"kind": "project", "root": "/tmp/fake-repo"},
        "harnesses": ["fake-review"], "primaryHarness": "fake-review",
        "model": "fake-small", "effort": "low", "maxSeconds": 30,
    }
    assert custody.record_start_requested(
        tmp_path, run_id="", task_id="t-b", idempotency_key="logical-key",
        invocation_id="original-key", operation_id="op", max_seconds=30, request=request,
        project_id="prior-project", project_owned=True, route="fake-review",
        surface="scope_review", slot_id="scope_slot_1",
    )
    state = {"pending_invocation_id": "original-key"}
    fake_route.start_error = _problem(row)
    with pytest.raises(ClaudexorUnavailable) as raised:
        _run_session_directly(tmp_path, retry_state=state, operation_id="op")
    assert raised.value.code == "idempotency_conflict" and raised.value.status_code == 409
    gateway = fake_route.instances[-1]
    assert gateway.start_requests == [request] and gateway.start_keys == ["original-key"]
    assert gateway.removals == [] and state == {"pending_invocation_id": "original-key"}
    retained = custody.invocation_record(tmp_path, "original-key")
    assert retained["state"] == "pending" and retained["request"] == request


@pytest.mark.parametrize("existing_project", [False, True])
def test_fresh_conflict_retires_only_the_registration_this_start_created(tmp_path, fake_route, existing_project):
    row = next(r for r in RESPONSES if r["route"] == "create" and r["scenario"] == "conflict")
    fake_route.project_unregistered = not existing_project
    fake_route.start_error = _problem(row)
    state = {}
    with pytest.raises(ClaudexorUnavailable, match="different request"):
        _run_session_directly(tmp_path, retry_state=state, operation_id="fresh")
    gateway = fake_route.instances[-1]
    assert len(gateway.start_requests) == 1 and state == {}
    assert gateway.removals == ([] if existing_project else ["proj-new"])
    retained = custody.invocation_record(tmp_path, gateway.start_keys[0])
    assert retained["state"] == "failed_definite"


def test_problem_context_is_only_diagnostic_and_old_engines_keep_empty_cause():
    row = {"status": 503, "problem": {
        "code": "idempotency_status_unavailable", "message": "Cannot observe the old request.",
        "requiredActions": [], "context": {"stage": "lookup_before_preflight", "cause": {
            "code": "subscription_window_exhausted", "message": "nested diagnostic",
            "resetsAt": "2030-01-01T00:00:00Z",
        }},
    }}
    error = _problem(row)
    assert type(error) is ClaudexorUnavailable and error.code == "idempotency_status_unavailable"
    row["problem"]["context"] = {}
    assert _problem(row).reported_cause == ""


@pytest.mark.parametrize("existing_project", [False, True])
@pytest.mark.parametrize("retained_thread", [False, True])
def test_unwritable_request_keeps_existing_or_thread_owned_registration(
    tmp_path, fake_route, monkeypatch, existing_project, retained_thread,
):
    from ouroboros import observability, review_execution

    fake_route.project_unregistered = not existing_project
    threads = []

    def create_thread(self, request, **_kwargs):
        threads.append(request)
        return {"id": "retained-thread"}

    def fail_blob(*_args, **_kwargs):
        raise OSError("request blob cannot be persisted")

    monkeypatch.setattr(fake_route, "create_thread", create_thread, raising=False)
    monkeypatch.setattr(observability, "write_blob", fail_blob)
    state = {}
    invocation = review_execution.SessionInvocation(
        task_id="task", surface="plan_review" if retained_thread else "scope_review",
        slot_id="scope_slot_1", timeout_sec=30, use_thread=retained_thread,
        retry_state=state,
    )
    with pytest.raises(review_execution.ReviewRouteUnavailable) as raised:
        review_execution.run_delegated_review_session(
            prompt="review", root="/tmp/fake-repo", custody_drive=tmp_path, invocation=invocation,
        )
    gateway = fake_route.instances[-1]
    assert raised.value.code == "start_request_row_unwritable"
    assert gateway.start_requests == [] and state == {}
    assert gateway.removals == ([] if existing_project or retained_thread else ["proj-new"])
    assert bool(threads) is retained_thread
    if retained_thread:
        assert threads[0]["scope"] == {"kind": "project", "root": "/tmp/fake-repo"}
