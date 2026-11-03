"""Live session terminals have the same typed interruption truth as cold collection."""
from dataclasses import asdict

import pytest

from ouroboros.observability import read_call_payload
from ouroboros.review_substrate import run_review_request, compact_review_projection
from ouroboros.triad_review import parse_seat_answers, emit_review_model_error_events
from tests._review_session_route_shared import (
    _owned_gateway_uses_each_test_transport as _transport, fake_route as _fake_route,
    _agent_request, _agent_slot, _terminal_detail, FakeLLM,
)
from tests.test_review_seat_outcome import _errors

fake_route = _fake_route
_owned_gateway_uses_each_test_transport = _transport


@pytest.mark.parametrize("state,code", [("cancelled", None), ("interrupted", "host_restart"),
                                        ("failed", "harness_failed"), ("succeeded", None)])
def test_live_terminal_interruption_failure_and_success(tmp_path, fake_route, state, code):
    detail = _terminal_detail("[]\nNO_FINDINGS" if state == "succeeded" else "", state=state, conformance="passed")
    if state != "succeeded":
        detail["summary"]["failure"] = {"code": code, "safeMessage": "engine's terminal cause"}
    fake_route.detail = detail
    llm = FakeLLM()
    request = _agent_request(surface="result_review", policy={"parts": ["result"]})
    run = run_review_request(request, slots=[_agent_slot()], drive_root=tmp_path, llm=llm)
    [actor] = run.actors
    interrupted = state in {"cancelled", "interrupted"}
    outcome = actor["outcome"]
    assert outcome["status"] == ("interrupted" if interrupted else "error" if state == "failed" else "responded")
    assert outcome["transport"] == ("interrupted" if interrupted else "provider_transport_error" if state == "failed" else "success")
    if state != "succeeded":
        assert actor["reported_cause"] == "engine's terminal cause"
        assert actor["failure_code"] == ("review_run_cancelled" if interrupted else code)
        assert actor["usage"]["review_run_state"] == state
        assert actor["usage"]["review_run_failure_code"] == (code or "run_cancelled")
    assert outcome["verdict"] == ("PASS" if state == "succeeded" else "")
    manifest, payload, _ref = read_call_payload(tmp_path, **outcome["source_ref"])
    assert manifest["producer_complete"] and payload["producer_outcome"]["outcome"] == outcome
    parsed = parse_seat_answers({"results": [actor]}, {actor["slot_id"]: ["result"]})
    emit_review_model_error_events(None, parsed, source="result_review")
    assert len(_errors(tmp_path)) == int(state == "failed")
    card = compact_review_projection([asdict(run)])["panels"][0]["actors"][0]
    assert card["transport_status"] == outcome["transport"] and card["semantic_verdict"] == outcome["verdict"]
    assert not llm.calls, "terminal collection never buys a second extraction"
