"""Focused terminal replay and physical-attempt history regressions."""

from __future__ import annotations

import pytest


def test_keyed_terminal_api_error_is_replayed_while_sibling_is_late(tmp_path):
    """A settled API slot must not be bought twice while its cycle sibling settles."""
    import threading
    from types import SimpleNamespace

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    late_started = threading.Event()
    late_finished = threading.Event()
    release_late = threading.Event()
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="mixed-cycle",
        retry_key="plan_review:mixed-cycle:1",
    )
    slots = [
        ReviewSlot(slot_id="fast", model="test/model", route=ReviewRouteKind.API_CHAT,
                   timeout_sec=0.2),
        ReviewSlot(slot_id="late", model="test/model", route=ReviewRouteKind.API_CHAT,
                   timeout_sec=0.01),
    ]

    def run_slot(slot, operation_id, _retry_state, _deadline, _checkpoint):
        calls.append(slot.slot_id)
        if slot.slot_id == "late":
            late_started.set()
            release_late.wait(1.0)
            late_finished.set()
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error",
            error=f"{slot.slot_id} terminal API error", operation_id=operation_id,
            http_status=500 if slot.slot_id == "fast" else 503,
            # A terminal actor is sufficient for same-cycle custody.  The
            # optional PhysicalAttemptCapture is intentionally absent here,
            # matching adapters that return only the typed actor record.
            usage={},
        )

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
        )

    args = dict(
        request=request, slots=slots, usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(drive_root=tmp_path, task_id=request.task_id),
        run_slot=run_slot, error_actor=error_actor,
    )
    first = run_custodied_review_slots(**args)
    assert {actor.slot_id: actor.operation_state for actor in first} == {
        "fast": "settled", "late": "in_flight",
    }
    assert late_started.wait(1.0)
    assert getattr(ctx, "_review_settled_attempts", {})
    second = run_custodied_review_slots(**args)
    assert calls == ["fast", "late"]
    assert {actor.slot_id: actor.operation_state for actor in second} == {
        "fast": "settled", "late": "in_flight",
    }

    release_late.set()
    assert late_finished.wait(1.0)


def test_unresolved_terminal_http_capture_is_replayed_without_second_dispatch(tmp_path):
    """A typed provider response makes an unresolved ledger row known-terminal."""
    from types import SimpleNamespace

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="terminal-capture",
        retry_key="plan_review:terminal-capture:1",
    )
    slot = ReviewSlot(
        slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
        timeout_sec=0.2,
    )

    def run_slot(slot, operation_id, _retry_state, _deadline, _checkpoint):
        calls.append(slot.slot_id)
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error",
            error="provider returned 503", operation_id=operation_id,
            http_status=503,
            usage={"physical_attempt_state": "unresolved", "provider_status_code": 503},
        )

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
        )

    args = dict(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(drive_root=tmp_path, task_id=request.task_id),
        run_slot=run_slot, error_actor=error_actor,
    )
    first = run_custodied_review_slots(**args)
    second = run_custodied_review_slots(**args)

    assert calls == ["only"]
    assert first[0].operation_state == "settled"
    assert second[0].operation_id == first[0].operation_id
    assert second[0].usage["physical_attempt_state"] == "unresolved"
    assert second[0].http_status == 503


def test_unresolved_capture_without_terminal_status_is_no_resend_without_typed_code(tmp_path):
    """A legacy actor cannot turn an unknown physical outcome into a resend."""
    from types import SimpleNamespace

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="unknown-capture",
        retry_key="plan_review:unknown-capture:1",
    )
    slot = ReviewSlot(
        slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
        timeout_sec=0.2,
    )

    def run_slot(slot, operation_id, _retry_state, _deadline, _checkpoint):
        calls.append(slot.slot_id)
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error",
            error="socket outcome unknown", operation_id=operation_id,
            usage={"physical_attempt_state": "unresolved"},
        )

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
        )

    args = dict(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(drive_root=tmp_path, task_id=request.task_id),
        run_slot=run_slot, error_actor=error_actor,
    )
    first = run_custodied_review_slots(**args)
    second = run_custodied_review_slots(**args)

    assert calls == ["only"]
    assert first[0].operation_state == "custody_lost"
    assert second[0].operation_state == "custody_lost"
    assert second[0].failure_code == "provider_outcome_unknown"
    assert second[0].late_result_pending is True


def test_unknown_physical_capture_state_is_no_resend_even_with_terminal_status(tmp_path):
    """Malformed physical provenance must win over an otherwise typed status."""
    from types import SimpleNamespace

    from ouroboros.review_custody import (
        _NO_RESEND, _attempt_key, run_custodied_review_slots,
    )
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="unknown-physical-state",
        retry_key="plan_review:unknown-physical-state:1",
    )
    slot = ReviewSlot(
        slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
        timeout_sec=0.2,
    )

    def run_slot(slot, operation_id, _retry_state, _deadline, _checkpoint):
        calls.append(operation_id)
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error",
            error="malformed physical state", operation_id=operation_id,
            http_status=503,
            usage={"physical_attempt_state": "future_state", "provider_status_code": 503},
        )

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
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
    assert first[0].operation_state == "custody_lost"
    assert first[0].failure_code == "provider_outcome_unknown"
    assert first[0].late_result_pending is True
    assert first[0].usage["physical_attempt_state"] == "unresolved"
    assert first[0].usage.get("provider_status_code") is None
    assert first[0].http_status is None
    assert second[0].operation_state == "custody_lost"
    assert second[0].operation_id == first[0].operation_id
    assert _attempt_key(request, slot) in _NO_RESEND


def test_unknown_exception_capture_projects_custody_loss_over_http_status():
    """Exception capture validation must not let status text launder custody."""
    from types import SimpleNamespace

    from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection

    exc = RuntimeError("provider returned 503")
    exc.physical_attempt_capture = SimpleNamespace(
        state="future_state", provider_status_code=503,
    )
    history = _ReviewAttemptHistory()
    history.observe(exc)

    custody, state, status, operation_state, failure_code = _review_exception_projection(
        exc, {"provider_status_code": 503}, history, {},
    )

    assert state == "unresolved"
    assert status is None
    assert operation_state == "custody_lost"
    assert failure_code == "provider_outcome_unknown"
    assert custody == {"physical_attempt_state": "unresolved", "review_failure_phase": "delivery"}


def test_coordinator_does_not_retry_malformed_physical_capture(tmp_path, monkeypatch):
    """A typed 503 cannot launder an unknown capture into retry authority."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    class MalformedCaptureExecutor:
        def __init__(self):
            self.execute_calls = 0

        class ProviderFailure(RuntimeError):
            @property
            def code(self):
                return ""

        def restore_custody(self, _state):
            return None

        def set_pending_invocation_checkpoint(self, _checkpoint):
            return None

        def prompt_payload(self):
            return {"messages": []}

        def prompt_chars(self):
            return 0

        def execute(self):
            self.execute_calls += 1
            error = self.ProviderFailure("provider returned 503")
            error.status_code = 503
            error.physical_attempt_capture = SimpleNamespace(
                state="future_state", provider_status_code=503,
            )
            raise error

        def failure_custody(self):
            return {}

    executor = MalformedCaptureExecutor()
    monkeypatch.setattr(
        "ouroboros.review_substrate._review_route_executor",
        lambda *_args, **_kwargs: executor,
    )
    result = run_review_request(
        ReviewRequest(
            surface="multi_model_review", goal="review",
            task_id="malformed-capture-retry",
            retry_key="commit_review:malformed-capture-retry:1",
        ),
        slots=[ReviewSlot(
            slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
        )],
        drive_root=tmp_path, usage_ctx=SimpleNamespace(),
    )

    assert executor.execute_calls == 1
    assert result.actors[0]["operation_state"] == "custody_lost"
    assert result.actors[0]["failure_code"] == "provider_outcome_unknown"
    assert result.actors[0]["late_result_pending"] is True
    assert "provider returned 503" in result.actors[0]["error"]


def test_coordinator_propagates_terminal_capture_status_for_same_cycle_replay(
    tmp_path, monkeypatch,
):
    """The coordinator must carry capture status into custody, not just its state."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    class ProviderFailure(RuntimeError):
        pass

    class TerminalExecutor:
        def __init__(self):
            self.execute_calls = 0

        def restore_custody(self, _state):
            return None

        def set_pending_invocation_checkpoint(self, _checkpoint):
            return None

        def prompt_payload(self):
            return {"messages": []}

        def prompt_chars(self):
            return 0

        def execute(self):
            self.execute_calls += 1
            error = ProviderFailure("provider HTTP 503")
            error.physical_attempt_capture = SimpleNamespace(
                state="unresolved", provider_status_code=503,
            )
            raise error

        def failure_custody(self):
            return {}

    executor = TerminalExecutor()
    monkeypatch.setattr(
        "ouroboros.review_substrate._review_route_executor",
        lambda *_args, **_kwargs: executor,
    )
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="capture-propagation",
        retry_key="plan_review:capture-propagation:1",
    )
    slot = ReviewSlot(slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT)
    ctx = SimpleNamespace()
    first = run_review_request(request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx)
    second = run_review_request(request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx)

    assert executor.execute_calls == 1
    assert first.actors[0]["http_status"] == 503
    assert first.actors[0]["usage"]["provider_status_code"] == 503
    assert second.actors[0]["operation_id"] == first.actors[0]["operation_id"]


def test_budget_exceeded_review_failure_remains_retryable(tmp_path, monkeypatch):
    """A budget refusal before reserve is a $0 row, never sticky custody."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
    from ouroboros.usage_accounting import BudgetExceeded

    class BudgetExecutor:
        execute_calls = 0

        def restore_custody(self, _state):
            return None

        def set_pending_invocation_checkpoint(self, _checkpoint):
            return None

        def prompt_payload(self):
            return {"messages": []}

        def prompt_chars(self):
            return 0

        def execute(self):
            self.execute_calls += 1
            raise BudgetExceeded("review budget exhausted")

        def failure_custody(self):
            return {}

    executor = BudgetExecutor()
    monkeypatch.setattr(
        "ouroboros.review_substrate._review_route_executor",
        lambda *_args, **_kwargs: executor,
    )
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="budget-refusal",
        retry_key="plan_review:budget-refusal:1",
    )
    slot = ReviewSlot(slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT)
    ctx = SimpleNamespace()
    first = run_review_request(request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx)
    second = run_review_request(request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx)

    assert executor.execute_calls == 2
    assert first.actors[0]["operation_state"] == "not_dispatched"
    assert second.actors[0]["operation_state"] == "not_dispatched"
    assert not getattr(ctx, "_review_settled_attempts", {})


def test_not_dispatched_error_remains_retryable_for_same_cycle(tmp_path):
    """A $0 admission refusal must not become a sticky replay actor."""
    from types import SimpleNamespace

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="retryable-refusal",
        retry_key="plan_review:retryable-refusal:1",
    )
    slot = ReviewSlot(
        slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
        timeout_sec=0.2,
    )

    def run_slot(slot, operation_id, _retry_state, _deadline, _checkpoint):
        calls.append(slot.slot_id)
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error",
            error="deadline exhausted before dispatch", operation_id=operation_id,
            operation_state="not_dispatched",
        )

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="not_dispatched",
            error=error, operation_id=operation_id, operation_state=operation_state,
        )

    args = dict(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(drive_root=tmp_path, task_id=request.task_id),
        run_slot=run_slot, error_actor=error_actor,
    )
    first = run_custodied_review_slots(**args)
    second = run_custodied_review_slots(**args)

    assert [actor.operation_state for actor in first] == ["not_dispatched"]
    assert [actor.operation_state for actor in second] == ["not_dispatched"]
    assert calls == ["only", "only"]
    assert not getattr(ctx, "_review_settled_attempts", {})


@pytest.mark.parametrize("physical_state", ["reserved", "released"])
def test_pre_dispatch_capture_state_remains_retryable_for_same_cycle(
    tmp_path, physical_state,
):
    """Explicit pre-dispatch capture facts must not become sticky replay rows."""
    from types import SimpleNamespace

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id=f"retryable-{physical_state}",
        retry_key=f"plan_review:retryable-{physical_state}:1",
    )
    slot = ReviewSlot(
        slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
        timeout_sec=0.2,
    )

    def run_slot(slot, operation_id, _retry_state, _deadline, _checkpoint):
        calls.append(slot.slot_id)
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error",
            error="pre-dispatch failure", operation_id=operation_id,
            usage={"physical_attempt_state": physical_state},
        )

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
        )

    args = dict(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id=request.task_id),
        run_slot=run_slot, error_actor=error_actor,
    )
    first = run_custodied_review_slots(**args)
    second = run_custodied_review_slots(**args)

    assert [actor.operation_state for actor in first] == ["settled"]
    assert [actor.operation_state for actor in second] == ["settled"]
    assert calls == ["only", "only"]
    assert not getattr(ctx, "_review_settled_attempts", {})


def test_late_pre_dispatch_capture_state_remains_retryable_for_same_cycle(tmp_path):
    """A late worker release must not become a sticky replay row either."""
    import threading
    import time
    from types import SimpleNamespace

    from ouroboros.review_custody import (
        _ACTIVE,
        _ACTIVE_LOCK,
        _attempt_key,
        run_custodied_review_slots,
    )
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    late_started = threading.Event()
    release_late = threading.Event()
    late_finished = threading.Event()
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="late-retryable-release",
        retry_key="plan_review:late-retryable-release:1",
    )
    slot = ReviewSlot(
        slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
        timeout_sec=0.01,
    )
    key = _attempt_key(request, slot)

    def run_slot(slot, operation_id, _retry_state, _deadline, _checkpoint):
        calls.append(slot.slot_id)
        if len(calls) == 1:
            late_started.set()
            assert release_late.wait(1.0)
            late_finished.set()
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error",
            error="pre-dispatch failure", operation_id=operation_id,
            usage={"physical_attempt_state": "released"},
        )

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
        )

    args = dict(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id=request.task_id),
        run_slot=run_slot, error_actor=error_actor,
    )
    first = run_custodied_review_slots(**args)
    assert [actor.operation_state for actor in first] == ["in_flight"]
    assert late_started.wait(1.0)
    release_late.set()
    assert late_finished.wait(1.0)
    # The worker signals just before returning, while settlement and removal
    # from process-local custody happen in the coordinator thread afterwards.
    # Wait for that handoff so the second call cannot race the first settlement.
    deadline = time.monotonic() + 1.0
    while True:
        with _ACTIVE_LOCK:
            settled = key not in _ACTIVE
        if settled:
            break
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert not getattr(ctx, "_review_settled_attempts", {})

    second = run_custodied_review_slots(**args)
    assert [actor.operation_state for actor in second] == ["settled"]
    assert calls == ["only", "only"]
    assert not getattr(ctx, "_review_settled_attempts", {})


def test_frozen_reconciliation_preserves_typed_custody_facts(tmp_path):
    """Durable rows must round-trip the typed failure and capture evidence."""
    from types import SimpleNamespace

    from ouroboros.review_custody import (
        prepare_frozen_review_reconciliation, run_custodied_review_slots,
    )
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    ctx = SimpleNamespace()
    prepare_frozen_review_reconciliation(ctx, SimpleNamespace(
        triad_raw_results=[{
            "slot_id": "only", "model_id": "test/model", "status": "error",
            "text": "", "error": "provider HTTP 503", "operation_id": "op-1",
            "operation_state": "settled", "late_result_pending": False,
            "failure_code": "provider_http_error", "reset_at": "2030-01-01T00:00:00Z",
            "http_status": 503, "transport_status": "http_error",
            "physical_attempt_state": "unresolved", "provider_status_code": 503,
        }],
        scope_raw_result={},
    ))
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="frozen-facts",
        retry_key="commit_review:frozen-facts", reconcile_only=True,
    )
    slot = ReviewSlot(
        slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT,
    )
    actors = run_custodied_review_slots(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(
            drive_root=tmp_path, task_id=request.task_id,
        ),
        run_slot=lambda *_args: pytest.fail("frozen row must not dispatch"),
        error_actor=lambda *_args, **_kwargs: pytest.fail("frozen row must not error"),
    )

    actor = actors[0]
    assert actor.failure_code == "provider_http_error"
    assert actor.reset_at == "2030-01-01T00:00:00Z"
    assert actor.http_status == 503
    assert actor.transport_status == "http_error"
    assert actor.late_result_pending is False
    assert actor.usage["physical_attempt_state"] == "unresolved"
    assert actor.usage["provider_status_code"] == 503


def test_pre_dispatch_route_refusal_is_retryable_not_sticky(tmp_path, monkeypatch):
    """A typed route admission refusal is $0 and can be retried on recovery."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind, ReviewRouteUnavailable
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    class NoTransport:
        def __init__(self):
            self.calls = 0

        def restore_custody(self, _state):
            return None

        def set_pending_invocation_checkpoint(self, _checkpoint):
            return None

        def prompt_payload(self):
            return {"messages": []}

        def prompt_chars(self):
            return 0

        def execute(self):
            self.calls += 1
            if self.calls == 1:
                raise ReviewRouteUnavailable("transport unavailable", code="api_chat_unavailable")
            from ouroboros.review_execution import ReviewAttemptResult
            return ReviewAttemptResult(message={"content": "[]"}, usage={}, raw_text="[]")

        def failure_custody(self):
            return {}

    executor = NoTransport()
    monkeypatch.setattr(
        "ouroboros.review_substrate._review_route_executor",
        lambda *_args, **_kwargs: executor,
    )
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="route-refusal",
        retry_key="plan_review:route-refusal:1",
    )
    slot = ReviewSlot(slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT)
    ctx = SimpleNamespace()
    first = run_review_request(request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx)
    second = run_review_request(request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx)

    assert first.actors[0]["operation_state"] == "not_dispatched"
    assert second.actors[0]["status"] == "ok"
    assert executor.calls == 2
    cached = getattr(ctx, "_review_settled_attempts", {})
    assert all(getattr(actor, "status", "") == "ok" for actor in cached.values())


def test_explicit_custody_lost_worker_actor_cannot_be_cached(tmp_path):
    """A missing exact recovery remains no-resend even without a failure code."""
    from types import SimpleNamespace

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="custody-lost-worker",
        retry_key="plan_review:custody-lost-worker:1",
    )
    slot = ReviewSlot(slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT)

    def run_slot(*_args):
        calls.append(1)
        from ouroboros.review_execution import ReviewRouteUnavailable
        raise ReviewRouteUnavailable("custody missing", code="review_custody_lost")

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
        )

    args = dict(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(
            drive_root=tmp_path, task_id=request.task_id,
        ), run_slot=run_slot, error_actor=error_actor,
    )
    first = run_custodied_review_slots(**args)
    second = run_custodied_review_slots(**args)

    assert calls == [1]
    assert first[0].operation_state == "custody_lost"
    assert second[0].operation_state == "custody_lost"
    assert not getattr(ctx, "_review_settled_attempts", {})


def test_terminal_status_overrides_legacy_unknown_code(tmp_path):
    """A typed terminal HTTP response stays replayable despite stale legacy code."""
    from types import SimpleNamespace

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls = []
    ctx = SimpleNamespace()
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="terminal-precedence",
        retry_key="plan_review:terminal-precedence:1",
    )
    slot = ReviewSlot(slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT)

    def run_slot(slot, operation_id, *_args):
        calls.append(1)
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error",
            error="HTTP 503", operation_id=operation_id,
            failure_code="provider_outcome_unknown", http_status=503,
            usage={"physical_attempt_state": "unresolved", "provider_status_code": 503},
        )

    args = dict(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(
            drive_root=tmp_path, task_id=request.task_id,
        ), run_slot=run_slot,
        error_actor=lambda slot, error, operation_id="", operation_state="settled":
            ReviewActorRecord(
                slot_id=slot.slot_id, model=slot.model, status="error", error=error,
                operation_id=operation_id, operation_state=operation_state,
            ),
    )
    first = run_custodied_review_slots(**args)
    second = run_custodied_review_slots(**args)

    assert calls == [1]
    assert first[0].operation_state == "settled"
    assert second[0].http_status == 503
    assert getattr(ctx, "_review_settled_attempts", {})


def test_rail_exhaustion_after_paid_attempts_is_not_zero_dispatch(tmp_path, monkeypatch):
    """A released rail reservation cannot erase earlier paid attempts."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
    from ouroboros.usage_accounting import AttemptRequest, execute_physical_attempt

    class Provider503(RuntimeError):
        status_code = 503

    class RailExecutor:
        def __init__(self):
            self.calls = 0
            self.physical_sends = 0

        def restore_custody(self, _state):
            return None

        def set_pending_invocation_checkpoint(self, _checkpoint):
            return None

        def prompt_payload(self):
            return {"messages": []}

        def prompt_chars(self):
            return 0

        def execute(self):
            self.calls += 1
            request = AttemptRequest(
                model="test/model", provider="openrouter", reservation_usd=0.01,
                drive_root=tmp_path, task_id="rail-exhaustion",
                root_task_id="rail-exhaustion",
            )

            def send():
                self.physical_sends += 1
                raise Provider503("provider HTTP 503")

            for _ in range(3):
                try:
                    execute_physical_attempt(request, send)
                except Provider503:
                    continue

        def failure_custody(self):
            return {}

    executor = RailExecutor()
    monkeypatch.setattr(
        "ouroboros.review_substrate._review_route_executor",
        lambda *_args, **_kwargs: executor,
    )
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="rail-exhaustion",
        retry_key="commit_review:rail-exhaustion:1",
    )
    slot = ReviewSlot(slot_id="only", model="test/model", route=ReviewRouteKind.API_CHAT)
    ctx = SimpleNamespace()
    first = run_review_request(request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx)
    second = run_review_request(request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx)

    assert executor.calls == 1
    assert executor.physical_sends == 2
    assert first.actors[0]["status"] == "error"
    assert first.actors[0]["operation_state"] == "custody_lost"
    assert first.actors[0]["failure_code"] == "provider_outcome_unknown"
    assert first.actors[0]["usage"]["physical_attempt_state"] == "unresolved"
    assert second.actors[0]["operation_state"] == "custody_lost"


def test_unknown_rail_capture_cannot_be_overwritten_by_later_terminal_status():
    """Unknown paid custody is monotonic across one actor's retry rail."""
    from types import SimpleNamespace

    from ouroboros.review_custody import _ReviewAttemptHistory

    history = _ReviewAttemptHistory()
    first = RuntimeError("unknown socket outcome")
    first.physical_attempt_capture = SimpleNamespace(
        state="unresolved", provider_status_code=None,
    )
    later = RuntimeError("later terminal response")
    later.physical_attempt_capture = SimpleNamespace(
        state="unresolved", provider_status_code=503,
    )
    history.observe(first)
    history.observe(later)
    custody = {"physical_attempt_state": "released", "provider_status_code": 503}
    history.preserve(custody, "released")

    assert custody == {"physical_attempt_state": "unresolved"}


def test_agent_session_config_refusal_is_zero_send_not_dispatched(tmp_path):
    """A typed route-construction failure is visible but spends no review cycle."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import (
        ReviewRequest, ReviewSlot, run_review_request,
    )

    paid = []
    ctx = SimpleNamespace(
        drive_root=tmp_path, task_id="zero-send-config",
        _review_paid_stamp=lambda: paid.append("paid"),
    )
    result = run_review_request(
        ReviewRequest(
            surface="plan_review", goal="review", task_id="zero-send-config",
            retry_key="plan_review:zero-send-config:1",
            session_root=str(tmp_path), session_task="",
        ),
        slots=[ReviewSlot(
            slot_id="slot-1", model="test/model",
            route=ReviewRouteKind.AGENT_SESSION,
        )],
        drive_root=tmp_path, usage_ctx=ctx,
    )

    actor = result.actors[0]
    assert paid == []
    assert actor["status"] == "not_dispatched"
    assert actor["operation_state"] == "not_dispatched"
    assert actor["failure_code"] == "session_task_missing"
    assert not actor.get("late_result_pending")


def test_native_pre_send_refusals_are_zero_send_not_dispatched():
    """A native tool-round refusal raised BEFORE the first provider send (no
    inspection registry; a bound below the first send) is a $0 retryable row,
    never a settled paid attempt."""
    from ouroboros.review_custody import _worker_exception_operation_state
    from ouroboros.review_execution import ReviewRouteUnavailable

    for code in ("native_inspection_unavailable", "native_bound_below_first_send"):
        error = ReviewRouteUnavailable("refused before any send", code=code)
        assert _worker_exception_operation_state(error, {}) == "not_dispatched", code
    # Ends AFTER a paid round stay settled.
    for code in ("native_transcript_cap_exceeded", "native_round_without_progress"):
        error = ReviewRouteUnavailable("refused after a paid round", code=code)
        assert _worker_exception_operation_state(error, {}) == "settled", code


def test_native_episode_with_paid_rounds_is_dispatched_whatever_ended_it():
    """A mid-episode owner deadline after one paid native round must not
    project as a $0 retryable row: the executor's own `native_rounds` fact
    makes the actor dispatched (settled), whatever the exception's capture."""
    from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection
    from ouroboros.review_execution import ReviewRouteUnavailable

    error = ReviewRouteUnavailable("owner deadline exhausted mid native review episode", code="deadline_exhausted")
    custody = {"delivery": "native_tool_rounds", "native_rounds": 1, "native_end_reason": "deadline_exhausted",
               "ledger_attempt_ids": ["attempt-1"], "resolved_model": "m"}
    _custody, _capture, _http, operation_state, failure_code = _review_exception_projection(
        error, custody, _ReviewAttemptHistory(), {},
    )
    assert operation_state == "settled" and failure_code == "deadline_exhausted"
    # The same exception with NO paid round stays a $0 row.
    _custody, _capture, _http, operation_state, _code = _review_exception_projection(
        error, {"delivery": "native_tool_rounds", "native_rounds": 0}, _ReviewAttemptHistory(), {},
    )
    assert operation_state == "not_dispatched"


def test_post_stamp_checkpoint_refusal_cannot_be_relabelled_zero_send():
    """Missing capture metadata must not erase a failure after write-ahead."""
    from ouroboros.review_custody import _worker_exception_operation_state
    from ouroboros.review_execution import ReviewRouteUnavailable

    error = ReviewRouteUnavailable(
        "pending invocation could not be checkpointed",
        code="review_custody_checkpoint_unwritable",
    )

    assert _worker_exception_operation_state(error, {}) == "settled"


def test_post_stamp_start_row_refusal_cannot_be_relabelled_zero_send():
    """The paid write-ahead precedes the durable START_REQUESTED write."""
    from ouroboros.review_custody import _worker_exception_operation_state
    from ouroboros.review_execution import ReviewRouteUnavailable

    error = ReviewRouteUnavailable(
        "start-request row could not be written",
        code="start_request_row_unwritable",
    )

    assert _worker_exception_operation_state(error, {}) == "settled"
