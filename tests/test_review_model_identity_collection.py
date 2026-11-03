"""Both session collectors retain the final attempt through real response persistence."""

import json

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.observability import read_blob_ref
from ouroboros.review_substrate import run_review_request
from tests._review_session_route_shared import (
    _agent_request,
    _agent_slot,
    _owned_gateway_uses_each_test_transport,  # noqa: F401
    _terminal_detail,
)
from tests._review_session_route_shared import (
    fake_route as fake_route,
)
from tests.test_contributor_model_identity import LABEL, MODEL, PROFILE, _proved_attempt, _telemetry
from tests.test_review_operation_collection import (
    PASS,
    SURFACE,
    TASK,
    _collect,
    _ctx,
    _observe,
    _session_run,
)
from tests.test_review_operation_collection import (
    forbid_effects as forbid_effects,
)


def _assert_persisted(root, actor):
    usage = actor["usage"]
    observed = usage["observed_attempt"]
    assert usage["resolved_model"] == LABEL
    assert usage["applied_profile"] == PROFILE and usage["delegated_route"] == "cursor"
    assert observed["model"] == LABEL and observed["profile_id"] == PROFILE
    assert observed["observed_model_catalog_match"] == _proved_attempt()["observed_model_catalog_match"]
    payload = read_blob_ref(root, actor["response_ref"]["redacted_projection_ref"])
    assert payload["usage"]["observed_attempt"] == observed
    assert payload["usage"]["claudexor"]["observed_attempt"] == observed
    assert payload["message"]["session_transcript"]
    return observed


def test_live_session_retains_identity_in_response(tmp_path, fake_route):
    fake_route.catalog_entry["id"] = "cursor"
    fake_route.detail = _terminal_detail("[]", conformance="passed", model=MODEL)
    fake_route.detail["summary"]["effectiveAccess"] = "readonly"
    fake_route.telemetry = {"run_id": "run-1", "final_attempt_id": "a02", "attempts": [_proved_attempt()]}

    class NoGeneration:
        def chat(self, **kwargs):
            pytest.fail("identity collection requested a new model generation")

    target = f"cursor={MODEL}"
    result = run_review_request(
        _agent_request(session_root=str(tmp_path)),
        slots=[_agent_slot(model=target, session_target=target, session_profile=PROFILE)],
        drive_root=tmp_path, llm=NoGeneration())
    actor = result.actors[0]
    assert actor["status"] == "ok"
    _assert_persisted(tmp_path, actor)
    assert sum(len(gateway.start_requests) for gateway in fake_route.instances) == 1


def test_late_collection_retains_identity_once_without_effects(tmp_path, monkeypatch, forbid_effects):
    detail = _telemetry(tmp_path, _proved_attempt(), run_id="run-late")
    detail["summary"].update(state="succeeded", runId="run-late", outputConformance="passed")
    detail["primaryOutput"] = {"text": PASS, "truncated": False}
    calls = _observe(monkeypatch, detail)
    custody.emit(tmp_path, custody.START_REQUESTED, {
        "invocation_id": "inv-op-session", "task_id": TASK, "operation_id": "op-session",
        "slot_id": "s", "surface": SURFACE, "root_task_id": TASK,
        "request": {"prompt": "review", "harnesses": ["cursor"], "model": MODEL}})
    custody.emit(tmp_path, custody.STARTED, {
        "run_id": "run-late", "invocation_id": "inv-op-session", "task_id": TASK, "surface": SURFACE})
    run = _session_run()
    run["slot_roster"][0].update(model=f"cursor={MODEL}", session_target=f"cursor={MODEL}", session_profile=PROFILE)
    actor, fact = _collect(run, tmp_path, _ctx(tmp_path))
    assert fact == "collected" and actor["status"] == "ok"
    _assert_persisted(tmp_path, actor)

    # A later read uses the retained producer response, not the now-changed engine file.
    (tmp_path / "engine-run" / "final" / "telemetry.yaml").write_text(json.dumps({"run_id": "other"}))
    replay, replay_fact = _collect(run, tmp_path, _ctx(tmp_path))
    assert replay_fact == fact
    assert replay["response_ref"]["redacted_projection_ref"] == actor["response_ref"]["redacted_projection_ref"]
    assert replay["usage"]["observed_attempt"] == actor["usage"]["observed_attempt"]
    assert calls == [("handshake",), ("get_run", "run-late"), ("close",)]
