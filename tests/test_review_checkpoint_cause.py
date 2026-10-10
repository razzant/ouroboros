"""A failed pre-send checkpoint keeps its primary cause and authority attribution."""

import pytest

from ouroboros.review_execution import ReviewRouteUnavailable
from ouroboros.review_projection import _panel_transport, _review_actor_projection, _transport_error_status
from ouroboros.review_session_custody import checkpoint_pending_invocation
from tests._review_session_route_shared import (
    FakeLLM, _agent_request, _agent_slot,
    _owned_gateway_uses_each_test_transport as _transport_fixture,
    fake_route as _route_fixture,
)

_owned_gateway_uses_each_test_transport = _transport_fixture
fake_route = _route_fixture
pytestmark = pytest.mark.serial  # the session fixture mutates its shared transport registry


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_checkpoint_cause_survives_cleanup_and_clears_unsent_token(cleanup_fails):
    state = {"pending_invocation_id": "invocation"}
    primary = TimeoutError("review state unavailable (reason=permission, errno=13)")
    primary.reported_cause = '{"reason":"permission","errno":13}'
    cleanup_calls = []

    def checkpoint(_invocation):
        raise primary

    def cleanup():
        cleanup_calls.append(True)
        if cleanup_fails:
            raise RuntimeError("cleanup transport unavailable")

    with pytest.raises(ReviewRouteUnavailable) as raised:
        checkpoint_pending_invocation(checkpoint=checkpoint, invocation_id="invocation",
                                      state=state, on_failure=cleanup)
    assert raised.value.code == "review_custody_checkpoint_unwritable"
    assert raised.value.__cause__ is primary
    assert '"reason":"permission"' in raised.value.reported_cause
    assert ("cleanup transport unavailable" in raised.value.reported_cause) is cleanup_fails
    assert state == {} and cleanup_calls == [True]


def test_successful_checkpoint_does_not_cleanup_or_drop_binding():
    state = {"pending_invocation_id": "invocation"}
    recorded = []
    checkpoint_pending_invocation(checkpoint=recorded.append, invocation_id="invocation", state=state,
                                  on_failure=lambda: pytest.fail("cleanup on a successful binding"))
    assert recorded == ["invocation"] and state["pending_invocation_id"] == "invocation"


def test_local_failure_phase_controls_actor_and_panel_word_not_error_prose():
    error = RuntimeError("a timeout mentioned in local checkpoint prose")
    assert _transport_error_status(error, failure_phase="authority") == "authority_error"
    assert _transport_error_status(error, failure_phase="delivery") == "timeout"
    assert _transport_error_status(RuntimeError("provider unavailable"), failure_phase="delivery") == "provider_transport_error"
    actor = _review_actor_projection({
        "status": "error", "operation_state": "settled", "error": str(error),
        "usage": {"review_failure_phase": "authority"},
        "failure_code": "review_custody_checkpoint_unwritable",
    }, "multi_model_review")
    assert actor["transport_status"] == "authority_error"
    assert _panel_transport([actor["transport_status"]]) == "authority_error"


@pytest.mark.parametrize("refused", [False, True])
def test_real_session_checkpoint_preserves_lock_cause_before_any_post(tmp_path, fake_route, refused):
    from dataclasses import asdict
    from ouroboros.review_state import ReviewStateLockError
    from ouroboros.review_substrate import ReviewCoordinator
    from ouroboros.tools.review_response import parse_model_response
    from ouroboros.triad_review import parse_model_review_results

    def checkpoint(_invocation_id):
        if refused:
            raise ReviewStateLockError(tmp_path / "locks/advisory_review.lock", {
                "reason": "contention", "errno": None, "elapsed_sec": 4.0, "timeout_sec": 4.0,
            })

    actor = ReviewCoordinator(llm=FakeLLM(), drive_root=tmp_path)._run_slot(
        _agent_request(), _agent_slot(), pending_invocation_checkpoint=checkpoint,
    )
    gateway = fake_route.instances[-1]
    if refused:
        assert gateway.start_requests == [] and actor.status == "error"
        assert actor.failure_code == "review_custody_checkpoint_unwritable"
        assert actor.usage["review_failure_phase"] == "authority"
        assert actor.transport_status == "authority_error"
        assert '"reason":"contention"' in actor.reported_cause
        envelope = parse_model_response(actor.model, asdict(actor), {})
        durable = parse_model_review_results({"results": [envelope]}).actor_records[0]
        assert durable.reported_cause == actor.reported_cause
        assert durable.transport_status == "authority_error"
        assert actor.response_ref, "the physical failure keeps its durable source"
    else:
        assert len(gateway.start_requests) == 1 and actor.status == "ok"
        assert actor.raw_text == "[]" and not actor.reported_cause


@pytest.mark.parametrize("state", ["unresolved", "settled", "future_state"])
def test_session_checkpoint_excludes_incoming_caller_capture(tmp_path, fake_route, state):
    from ouroboros import usage_accounting as ua
    from ouroboros.review_state import ReviewStateLockError
    from ouroboros.review_substrate import ReviewCoordinator

    incoming = ua.PhysicalAttemptCapture(
        attempt_id="unrelated-previous-attempt", model="unrelated/model",
        provider="synthetic", state=state, candidate_measurement_kind="opaque",
    )

    def checkpoint(_invocation_id):
        raise ReviewStateLockError(tmp_path / "lock", {"reason": "contention", "errno": None})

    token = ua._LAST_PHYSICAL_ATTEMPT.set(incoming)
    try:
        actor = ReviewCoordinator(llm=FakeLLM(), drive_root=tmp_path)._run_slot(
            _agent_request(), _agent_slot(), pending_invocation_checkpoint=checkpoint,
        )
        assert fake_route.instances[-1].start_requests == []
        assert actor.status == "error"
        assert actor.failure_code == "review_custody_checkpoint_unwritable"
        assert actor.operation_state == "settled"
        assert actor.usage["review_failure_phase"] == "authority"
        assert "physical_attempt_state" not in actor.usage
        assert actor.transport_status == "authority_error"
        assert '"reason":"contention"' in actor.reported_cause
        assert actor.response_ref
        assert ua.last_physical_attempt_capture() is incoming
        assert incoming.state == state
    finally:
        ua._LAST_PHYSICAL_ATTEMPT.reset(token)


@pytest.mark.parametrize("later_capture", ["unchanged", "cleared", "released"])
def test_session_checkpoint_retains_same_actor_retry_capture(
    tmp_path, fake_route, monkeypatch, later_capture,
):
    from ouroboros import usage_accounting as ua
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAttemptResult
    from ouroboros.review_state import ReviewStateLockError
    from ouroboros.review_substrate import ReviewCoordinator

    incoming = ua.PhysicalAttemptCapture(
        attempt_id="unrelated-previous-attempt", model="unrelated/model",
        provider="synthetic", state="settled", provider_status_code=503,
        candidate_measurement_kind="opaque",
    )
    own = ua.PhysicalAttemptCapture(
        attempt_id="this-actor-attempt", model="api/model-a", provider="synthetic",
        state="unresolved", candidate_measurement_kind="opaque",
    )
    execute = AgentSessionReviewExecutor.execute
    attempts = []

    def empty_then_checkpoint(executor):
        attempts.append(True)
        if len(attempts) == 1:
            # A legacy empty result exposes its physical fact only in the
            # execution context. The second attempt refuses before any POST.
            ua.adopt_physical_attempt_capture(own)
            return ReviewAttemptResult(message={"content": ""}, usage={}, raw_text="")
        if later_capture == "cleared":
            ua.adopt_physical_attempt_capture(None)
        elif later_capture == "released":
            ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
                attempt_id="this-actor-unsent-retry", model="api/model-a",
                provider="synthetic", state="released", candidate_measurement_kind="opaque",
            ))
        return execute(executor)

    def checkpoint(_invocation_id):
        raise ReviewStateLockError(tmp_path / "lock", {"reason": "contention", "errno": None})

    monkeypatch.setattr(AgentSessionReviewExecutor, "execute", empty_then_checkpoint)
    token = ua._LAST_PHYSICAL_ATTEMPT.set(incoming)
    try:
        actor = ReviewCoordinator(llm=FakeLLM(), drive_root=tmp_path)._run_slot(
            _agent_request(), _agent_slot(), pending_invocation_checkpoint=checkpoint,
        )
    finally:
        ua._LAST_PHYSICAL_ATTEMPT.reset(token)

    assert len(attempts) == 2
    assert fake_route.instances[-1].start_requests == []
    assert actor.status == "error"
    assert actor.failure_code == "provider_outcome_unknown"
    assert actor.operation_state == "custody_lost"
    assert actor.late_result_pending is True
    assert actor.usage["physical_attempt_state"] == "unresolved"
    assert actor.http_status is None
    assert "provider_status_code" not in actor.usage
    assert actor.usage["review_failure_phase"] == "authority"
    assert actor.transport_status == "authority_error"
    assert '"reason":"contention"' in actor.reported_cause
    assert actor.response_ref


@pytest.mark.parametrize("scenario,expected", [
    ("authority", "authority_error"), ("authority_unsent", "authority_error"),
    ("provider", "provider_transport_error"), ("unsent", "not_dispatched"),
    ("success", "success"),
])
def test_failure_facts_reach_persisted_task_history_and_actual_card(tmp_path, fake_route, monkeypatch, scenario, expected):
    from dataclasses import asdict, replace
    from ouroboros import review_ledger
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_state import ReviewStateLockError
    from ouroboros.review_substrate import ReviewCoordinator
    from ouroboros.tools.review_response import parse_model_response
    from ouroboros.triad_review import parse_model_review_results
    from tests.test_review_ledger import _facts
    from tests.test_review_record_card_projection import _render, _terminal

    def checkpoint(_):
        raise ReviewStateLockError(tmp_path / "lock", {"reason": "contention", "errno": None})

    coordinator = ReviewCoordinator(llm=FakeLLM(), drive_root=tmp_path)
    request, slot = _agent_request(task_id="task-1"), _agent_slot()
    if scenario == "provider":
        error = ClaudexorUnavailable("idempotency_status_unavailable", "Cannot observe the request", status_code=503)
        error.reported_cause = "daemon lookup transport unavailable"

        def unavailable(self, request, *, idempotency_key=""):
            self.start_requests.append(dict(request))
            self.start_keys.append(idempotency_key)
            raise error

        monkeypatch.setattr(fake_route, "start_run", unavailable)
    if scenario == "unsent":
        actors = [coordinator._error_actor(request, slot, "No dispatch admitted", operation_state="not_dispatched")]
    else:
        actors = [coordinator._run_slot(request, slot,
                  pending_invocation_checkpoint=checkpoint if scenario.startswith("authority") else None)]
    if scenario == "authority_unsent":
        actors.append(coordinator._error_actor(request, replace(slot, slot_id="withheld"),
                      "No dispatch admitted", operation_state="not_dispatched"))
    envelopes = [parse_model_response(actor.model, {
        **asdict(actor), "choices": [{"message": {"content": actor.raw_text}}],
    }, {}) for actor in actors]
    raws = [actor.to_dict() for actor in parse_model_review_results({"results": envelopes}).actor_records]
    record = review_ledger.write_record(tmp_path, review_ledger.build_commit_gate_record(_facts(raws)))
    readback = review_ledger.load_record(tmp_path, record["record_id"])
    stored, event, history = _terminal(tmp_path, "task-1")
    projection = stored["review_projection"]
    assert projection == event == history["review_projection"]
    panel = projection["panels"][0]
    assert panel["transport_status"] == expected
    shown = _render("task-1", projection)
    assert f"transport={expected}" in shown["card"]
    if scenario.startswith("authority") or scenario == "provider":
        cause = actors[0].reported_cause
        assert readback["rows"][0]["reported_cause"] == cause
        assert panel["actors"][0]["reported_cause"] == cause
        assert cause in shown["card"]
        assert any(cause in attempt["detailText"] for group in shown["groups"] for attempt in group["attempts"])
    if scenario.startswith("authority") or scenario == "unsent":
        assert all(not gateway.start_requests for gateway in fake_route.instances)
        assert "provider_transport_error" not in shown["card"]
    elif scenario == "provider":
        keys = [key for gateway in fake_route.instances for key in gateway.start_keys]
        assert keys and len(set(keys)) == 1  # observation retries retain the one original operation
    else:
        assert sum(len(gateway.start_requests) for gateway in fake_route.instances) == 1


def test_mixed_provider_failure_is_not_hidden_and_legacy_causes_are_not_invented():
    from ouroboros.review_projection import ledger_record_panel
    from ouroboros import review_ledger
    from tests.test_review_ledger import _facts

    assert _panel_transport(["authority_error", "provider_transport_error"]) == "provider_transport_error"
    assert _panel_transport(["authority_error", "timeout"]) == "provider_transport_error"
    record = review_ledger.build_commit_gate_record(_facts([{
        "slot_id": "old", "status": "error", "model_id": "old/model",
    }])).to_dict()
    actor = ledger_record_panel(record)["actors"][0]
    assert actor["reason"] == "" and "reported_cause" not in actor
