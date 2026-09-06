"""Fail-closed regressions for malformed physical review-custody facts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("actor_status", "operation_state", "provider_status", "expected_state"),
    [
        ("error", "not_dispatched", 503, "settled"),
        ("not_dispatched", "settled", 503, "settled"),
        ("error", "not_dispatched", None, "custody_lost"),
    ],
)
def test_positive_capture_outranks_synthetic_not_dispatched_facts(
    tmp_path, actor_status, operation_state, provider_status, expected_state,
):
    """Contradictory synthetic labels cannot authorize another physical send."""
    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    suffix = f"{actor_status}-{operation_state}-{provider_status}"
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id=f"positive-wins-{suffix}",
        retry_key=f"plan_review:positive-wins:{suffix}",
    )
    # The slot returns at once; the timeout is not the property under test. A
    # tight 0.2 s let a loaded macOS runner time the worker out before it was
    # scheduled and report in_flight (CI run 33577803504).
    slot = ReviewSlot(
        slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
        timeout_sec=10.0,
    )

    def run_slot(slot, operation_id, _retry_state, _deadline, _checkpoint):
        calls.append(operation_id)
        usage = {"physical_attempt_state": "unresolved"}
        if provider_status is not None:
            usage["provider_status_code"] = provider_status
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status=actor_status,
            error="contradictory custody facts", operation_id=operation_id,
            operation_state=operation_state, http_status=provider_status, usage=usage,
        )

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
            late_result_pending=operation_state in {"in_flight", "custody_lost"},
        )

    args = dict(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(
            drive_root=tmp_path, task_id=request.task_id,
        ), run_slot=run_slot, error_actor=error_actor,
    )
    first = run_custodied_review_slots(**args)
    second = run_custodied_review_slots(**args)

    assert len(calls) == 1
    assert first[0].operation_state == expected_state
    assert second[0].operation_state == expected_state
    assert second[0].operation_id == first[0].operation_id
    if provider_status is not None:
        assert first[0].status == "error"
        assert second[0].http_status == provider_status
    else:
        assert first[0].failure_code == "provider_outcome_unknown"


def test_api_retry_token_cannot_impersonate_delegated_recovery(tmp_path):
    """Only an agent-session invocation token can survive process-local loss."""
    from ouroboros.review_custody import (
        prepare_frozen_review_reconciliation, run_custodied_review_slots,
    )
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    ctx = SimpleNamespace()
    prepare_frozen_review_reconciliation(ctx, SimpleNamespace(
        triad_raw_results=[{
            "slot_id": "only", "model_id": "test/model", "status": "error",
            "error": "logical wait expired", "operation_id": "op-old",
            "operation_state": "in_flight", "late_result_pending": True,
            "pending_invocation_id": "fake-api-token",
        }],
        scope_raw_result={},
    ))
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="fake-api-token",
        retry_key="commit_review:fake-api-token", reconcile_only=True,
        deadline_at="2000-01-01T00:00:00Z",
    )
    slot = ReviewSlot(
        slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
    )

    def run_slot(*args):
        calls.append(args)
        return ReviewActorRecord(slot_id="only", model="test/model", status="ok")

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
            late_result_pending=operation_state in {"in_flight", "custody_lost"},
        )

    [actor] = run_custodied_review_slots(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={"deadline_at": request.deadline_at},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id=request.task_id),
        run_slot=run_slot, error_actor=error_actor,
    )

    assert calls == []
    assert actor.operation_state == "custody_lost"
    assert actor.late_result_pending is True


def test_delegated_retry_token_is_bound_to_its_physical_operation():
    """A valid token from another operation cannot settle this review actor."""
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.review_session_custody import review_recovery_facts

    route_id = "claude"
    record = {
        "task_id": "task-1", "surface": "scope_review", "slot_id": "scope_slot_1",
        "operation_id": "op-original", "route": route_id,
        "project_id": "project-1", "project_owned": False,
        "idempotency_key": "invocation-1",
    }
    request = {
        "scope": {"kind": "project", "root": "/tmp/repo"},
        "primaryHarness": route_id, "harnesses": [route_id],
        "authPreference": "subscription", "mode": "ask", "access": "readonly",
        "prompt": "same review prompt", "model": "claude-opus-5",
    }

    with pytest.raises(ReviewRouteUnavailable) as raised:
        review_recovery_facts(
            record, request, None,
            prompt="same review prompt", root="/tmp/repo", claimant_task_id="task-1",
            claimant_surface="scope_review", claimant_slot_id="scope_slot_1",
            claimant_operation_id="op-replacement",
        )

    assert raised.value.code == "review_recovery_request_mismatch"
