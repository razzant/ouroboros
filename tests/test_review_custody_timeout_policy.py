"""Focused timeout and retry regressions for review custody."""

from __future__ import annotations

import pytest


def test_deadline_crossing_before_queue_wait_preserves_late_result(tmp_path, monkeypatch):
    """A scheduling gap after expiry checks must not break physical custody."""
    import threading
    from types import SimpleNamespace

    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    ticks = iter((100.0, 100.04))
    # The slot starts with 50ms, passes expiry at 40ms, then resumes at 100ms.
    # Patch only the coordinator's clock; Queue.get retains its real contract.
    monkeypatch.setattr(custody, "monotonic_now", lambda _slot: next(ticks, 100.1))
    release = threading.Event()
    entered = threading.Event()
    workers, calls, paid = [], [], []
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="queue-deadline-gap",
        retry_key="commit_review:queue-deadline-gap",
    )
    slot = ReviewSlot(slot_id="slot-1", model="test/model", timeout_sec=0.05)
    ctx = SimpleNamespace(_review_paid_stamp=lambda: paid.append("paid"))

    def run_slot(_slot, operation_id, _retry_state, deadline, _checkpoint):
        workers.append(threading.current_thread())
        calls.append((operation_id, deadline))
        entered.set()
        assert release.wait(10), "test did not release the review worker"
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="ok", raw_text="[]",
        )

    def error_actor(_slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
            late_result_pending=operation_state == "in_flight",
        )

    kwargs = dict(
        request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id,
        usage_meta={}, review_usage_scope=UsageScope(drive_root=tmp_path),
        run_slot=run_slot, error_actor=error_actor,
    )
    try:
        [first] = custody.run_custodied_review_slots(**kwargs)
        assert first.operation_state == "in_flight"
        assert first.late_result_pending is True
        assert entered.wait(10), "review worker never started"
        assert calls == [(first.operation_id, 100.05)]
    finally:
        release.set()
        assert entered.wait(10), "review worker never started"
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive(), "review worker did not settle"

    [replayed] = custody.run_custodied_review_slots(**kwargs)
    assert calls == [(first.operation_id, 100.05)]
    assert paid == ["paid"]
    assert replayed.operation_id == first.operation_id
    assert replayed.operation_state == "late_settled"
    assert replayed.late_result_pending is False
    assert replayed.status == "ok"
    assert replayed.raw_text == "[]"


def test_spent_deadline_restart_reconciliation_gets_only_a_settlement_window(
    tmp_path, monkeypatch,
):
    import time
    from types import SimpleNamespace

    import ouroboros.config as config
    from ouroboros.review_custody import (
        prepare_frozen_review_reconciliation, run_custodied_review_slots,
    )
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    monkeypatch.setattr(config, "NESTED_SETTLEMENT_MARGIN_SEC", 0.2)
    attempt = SimpleNamespace(triad_raw_results=[{
        "slot_id": "slot-1", "model_id": "cursor/test", "status": "error",
        "operation_id": "op-existing", "operation_state": "in_flight",
        "late_result_pending": True, "pending_invocation_id": "inv-existing",
    }], scope_raw_result={})
    paid = []
    ctx = SimpleNamespace(_review_paid_stamp=lambda: paid.append("paid"))
    prepare_frozen_review_reconciliation(ctx, attempt)
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="task-spent-restart",
        retry_key="commit_review:spent-restart", reconcile_only=True,
        deadline_at="2000-01-01T00:00:00Z",
    )
    slot = ReviewSlot(
        slot_id="slot-1", model="cursor/test", route=ReviewRouteKind.AGENT_SESSION,
    )
    calls = []

    def recover(_slot, operation_id, retry_state, deadline, _checkpoint):
        calls.append((operation_id, dict(retry_state), deadline - time.monotonic()))
        return ReviewActorRecord(slot_id="slot-1", model="cursor/test", status="ok")

    [actor] = run_custodied_review_slots(
        request=request,
        slots=[slot],
        usage_ctx=ctx,
        task_id="task-spent-restart",
        usage_meta={"deadline_at": "2000-01-01T00:00:00Z"},
        review_usage_scope=UsageScope(
            drive_root=tmp_path, task_id="task-spent-restart",
        ),
        run_slot=recover,
        error_actor=lambda *_args, **_kwargs: None,
    )

    assert len(calls) == 1
    operation_id, retry_state, remaining = calls[0]
    assert operation_id == "op-existing"
    assert retry_state == {"pending_invocation_id": "inv-existing"}
    # The deadline is monotonic_now + 0.2 and ``remaining`` re-reads the same
    # clock, so it can only shrink — but on Windows the coarse monotonic clock
    # returns the identical reading twice, leaving pure float rounding noise
    # above 0.2 (observed +4.5e-13 on the CI shard). Tolerate one epsilon.
    assert 0 < remaining <= 0.2 + 1e-9
    assert paid == []
    assert actor.operation_id == "op-existing"
    assert actor.operation_state == "settled"


def test_reserved_operation_keeps_not_dispatched_identity_inside_reserve(
    tmp_path, monkeypatch,
):
    from types import SimpleNamespace

    from ouroboros.review_custody import (
        reconcile_reserved_review_roster, run_custodied_review_slots,
    )
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    reserved = {
        "multi_model_review": [{
            "slot_id": "slot-1", "model_id": "cursor/test", "route": "api_chat",
            "status": "in_flight", "operation_id": "op-reserved",
            "operation_state": "in_flight", "late_result_pending": True,
        }],
        "scope_review": [],
    }
    ctx = SimpleNamespace(
        _review_reserved_operations={"multi_model_review": {"slot-1": "op-reserved"}},
        _review_reserved_roster=reserved,
    )

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="not_dispatched",
            error=error, operation_id=operation_id, operation_state=operation_state,
        )

    [actor] = run_custodied_review_slots(
        request=ReviewRequest(
            surface="multi_model_review", goal="review", task_id="reserve-only",
            retry_key="commit_review:reserve-only", deadline_at="2000-01-01T00:00:00Z",
        ),
        slots=[ReviewSlot(
            slot_id="slot-1", model="cursor/test", route=ReviewRouteKind.API_CHAT,
        )],
        usage_ctx=ctx, task_id="reserve-only",
        usage_meta={"deadline_at": "2000-01-01T00:00:00Z"},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id="reserve-only"),
        run_slot=lambda *_args: pytest.fail("reserve-only row dispatched"),
        error_actor=error_actor,
    )

    assert actor.operation_id == "op-reserved"
    assert actor.operation_state == "not_dispatched"
    ctx._last_triad_raw_results = [{
        "slot_id": actor.slot_id, "status": actor.status,
        "operation_id": actor.operation_id, "operation_state": actor.operation_state,
        "late_result_pending": False,
    }]
    ctx._last_scope_raw_results = []
    reconcile_reserved_review_roster(ctx, reserved)
    assert ctx._last_triad_raw_results[0]["operation_state"] == "not_dispatched"
    assert getattr(ctx, "_review_custody_lost", False) is False


def test_custody_lost_error_actor_remains_pending_for_plan_review(tmp_path):
    from ouroboros.review_substrate import (
        ReviewCoordinator, ReviewRequest, ReviewSlot,
    )

    coordinator = ReviewCoordinator(drive_root=tmp_path, usage_ctx=None)
    actor = coordinator._error_actor(
        ReviewRequest(surface="plan_review", goal="review", task_id="pending"),
        ReviewSlot(slot_id="slot-1", model="test/model"),
        "provider outcome is unknown",
        operation_id="op-1",
        operation_state="custody_lost",
    )

    assert actor.operation_state == "custody_lost"
    assert actor.late_result_pending is True


def test_failed_commit_roster_stamp_drops_unsent_process_local_reservation(
    monkeypatch,
):
    from types import SimpleNamespace

    from ouroboros.tools.parallel_review import _reserve_parallel_review_roster

    ctx = SimpleNamespace(_triad_withheld_seat_records=[])

    def fail_stamp(_ctx):
        raise RuntimeError("write-ahead unavailable")

    monkeypatch.setattr(
        "ouroboros.review_dispatch.stamp_review_paid_on_dispatch", fail_stamp,
    )
    with pytest.raises(RuntimeError, match="write-ahead unavailable"):
        _reserve_parallel_review_roster(
            ctx,
            {"row_plan": {
                "models": ["test/model"], "routes": ["api_chat"],
                "efforts": ["high"], "slot_ids": ["slot-1"],
            }},
        )

    assert getattr(ctx, "_review_reserved_roster", None) is None
    assert getattr(ctx, "_review_reserved_operations", {}) == {}


def test_spent_owner_window_does_not_stamp_zero_dispatch_commit_roster(
    monkeypatch,
):
    from types import SimpleNamespace

    from ouroboros.tools.parallel_review import _reserve_parallel_review_roster

    stamp_calls = []
    ctx = SimpleNamespace(
        _triad_withheld_seat_records=[],
        task_metadata={"deadline_at": "2000-01-01T00:00:00Z"},
    )
    monkeypatch.setattr(
        "ouroboros.review_dispatch.stamp_review_paid_on_dispatch",
        lambda _ctx: stamp_calls.append("paid"),
    )

    _reserve_parallel_review_roster(
        ctx,
        {"row_plan": {
            "models": ["test/model"], "routes": ["api_chat"],
            "efforts": ["high"], "slot_ids": ["slot-1"],
        }},
    )

    assert stamp_calls == []
    assert ctx._review_reserved_roster["multi_model_review"][0]["operation_id"]
    assert ctx._review_reserved_operations["multi_model_review"]["slot-1"]


def test_binding_refusal_clears_commit_reconcile_mode_before_return(
    tmp_path, monkeypatch,
):
    from types import SimpleNamespace

    from ouroboros.tools import git as git_tools

    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        task_id="binding-refusal",
        task_metadata={},
        _review_resume_pending=True,
        _pending_review_attempt=SimpleNamespace(review_retry_key="exact"),
    )
    monkeypatch.setattr(
        git_tools, "_stage_candidate_for_review",
        lambda *_args, **_kwargs: ([], [], None),
    )
    monkeypatch.setattr(
        git_tools, "_fingerprint_staged_diff",
        lambda *_args, **_kwargs: {"ok": True, "fingerprint": "fp"},
    )
    monkeypatch.setattr(
        git_tools, "_free_cycle_gate",
        lambda run_ctx, *_args, **_kwargs: (
            setattr(run_ctx, "_review_reconcile_only", True) or None
        ),
    )
    monkeypatch.setattr(
        git_tools, "_review_binding_precondition_error",
        lambda *_args, **_kwargs: "binding is invalid",
    )
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *_a, **_k: None)

    outcome = git_tools._run_reviewed_stage_cycle(ctx, "message", 0.0)

    assert outcome["block_reason"] == "review_binding_invalid"
    assert ctx._review_reconcile_only is False


@pytest.mark.parametrize(
    "pending_invocation_id,operation_id",
    [("inv-existing", ""), ("", "op-existing")],
)
def test_restart_reconciliation_requires_complete_exact_identity(
    tmp_path, pending_invocation_id, operation_id,
):
    from types import SimpleNamespace

    from ouroboros.review_custody import (
        prepare_frozen_review_reconciliation, run_custodied_review_slots,
    )
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    attempt = SimpleNamespace(triad_raw_results=[{
        "slot_id": "slot-1", "model_id": "cursor/test", "status": "error",
        "operation_id": operation_id, "operation_state": "in_flight",
        "late_result_pending": True,
        "pending_invocation_id": pending_invocation_id,
    }], scope_raw_result={})
    ctx = SimpleNamespace()
    prepare_frozen_review_reconciliation(ctx, attempt)
    calls = []

    def error_actor(slot, error, actor_operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=actor_operation_id, operation_state=operation_state,
            late_result_pending=operation_state in {"in_flight", "custody_lost"},
        )

    [actor] = run_custodied_review_slots(
        request=ReviewRequest(
            surface="multi_model_review", goal="review", task_id="task-partial",
            retry_key="commit_review:partial", reconcile_only=True,
        ),
        slots=[ReviewSlot(
            slot_id="slot-1", model="cursor/test", route=ReviewRouteKind.AGENT_SESSION,
        )],
        usage_ctx=ctx,
        task_id="task-partial",
        usage_meta={},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id="task-partial"),
        run_slot=lambda *_args: calls.append("dispatched"),
        error_actor=error_actor,
    )

    assert calls == []
    assert actor.operation_state == "custody_lost"
    assert ctx._review_custody_lost is True


def test_logical_timeout_actor_carries_live_delegated_restart_token(tmp_path):
    import threading
    import time
    from types import SimpleNamespace

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    release = threading.Event()
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="task-timeout-token",
        retry_key="commit_review:token",
    )
    slot = ReviewSlot(
        slot_id="slot-1", model="cursor/test", route=ReviewRouteKind.AGENT_SESSION,
        timeout_sec=0.1,
    )
    calls = []

    def still_running(_slot, _operation_id, retry_state, _deadline, _checkpoint):
        calls.append(_operation_id)
        retry_state["pending_invocation_id"] = "inv-live"
        retry_state["delegated_run_id"] = "run-live"
        release.wait(0.4)
        return ReviewActorRecord(slot_id="slot-1", model="cursor/test", status="ok")

    def error_actor(_slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id="slot-1", model="cursor/test", status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
            late_result_pending=operation_state == "in_flight",
        )

    started = time.monotonic()
    ctx = SimpleNamespace()
    [actor] = run_custodied_review_slots(
        request=request,
        slots=[slot],
        usage_ctx=ctx,
        task_id="task-timeout-token",
        usage_meta={},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id="task-timeout-token"),
        run_slot=still_running,
        error_actor=error_actor,
    )
    [joined] = run_custodied_review_slots(
        request=ReviewRequest(
            surface="multi_model_review", goal="review",
            task_id="task-timeout-token", retry_key="commit_review:token",
            reconcile_only=True, deadline_at="2000-01-01T00:00:00Z",
        ),
        slots=[slot],
        usage_ctx=ctx,
        task_id="task-timeout-token",
        usage_meta={"deadline_at": "2000-01-01T00:00:00Z"},
        review_usage_scope=UsageScope(
            drive_root=tmp_path, task_id="task-timeout-token",
        ),
        run_slot=still_running,
        error_actor=error_actor,
    )
    release.set()

    assert time.monotonic() - started < 0.3
    assert len(calls) == 1
    assert actor.operation_state == "in_flight"
    assert actor.usage["pending_invocation_id"] == "inv-live"
    assert actor.usage["delegated_run_id"] == "run-live"
    assert joined.operation_id == actor.operation_id
    assert joined.operation_state == "in_flight"
    assert joined.usage["pending_invocation_id"] == "inv-live"
    assert joined.usage["delegated_run_id"] == "run-live"


def test_coordinator_keeps_fresh_empty_session_custody_cell_shared(
    tmp_path, monkeypatch,
):
    """The public coordinator adapter must not replace an empty custody dict."""
    import threading
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewAttemptResult, ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    release = threading.Event()
    settled = threading.Event()
    checkpoints = []

    class BlockingSessionExecutor:
        def __init__(self):
            self.state = None
            self.checkpoint = None
            self.execute_calls = 0

        def restore_custody(self, state):
            self.state = state

        def set_pending_invocation_checkpoint(self, checkpoint):
            self.checkpoint = checkpoint

        def prompt_payload(self):
            return {"session_prompt": "review"}

        def prompt_chars(self):
            return 6

        def execute(self):
            self.execute_calls += 1
            self.state["pending_invocation_id"] = "inv-shared"
            self.state["delegated_run_id"] = "run-shared"
            if self.checkpoint is not None:
                self.checkpoint("inv-shared")
            # Hang-escape only, NOT a deadline the test body must beat: with a
            # 1.0s cap a slow CI runner (Windows shard) let the wait expire
            # between the two run_review_request calls, so the run settled
            # naturally with empty usage and the join replayed "late_settled"
            # instead of sharing the in-flight custody cell.
            release.wait(30.0)
            settled.set()
            return ReviewAttemptResult(
                message={"content": "[]"}, usage={}, raw_text="[]",
            )

        def failure_custody(self):
            return dict(self.state or {})

    executor = BlockingSessionExecutor()
    monkeypatch.setattr(
        "ouroboros.review_substrate._review_route_executor",
        lambda *_args, **_kwargs: executor,
    )
    ctx = SimpleNamespace(
        _review_pending_invocation_checkpoint=lambda **facts: checkpoints.append(facts),
    )
    request = ReviewRequest(
        surface="multi_model_review",
        goal="review",
        task_id="shared-empty-cell",
        retry_key="commit_review:shared-empty-cell",
        session_root=str(tmp_path),
        session_task="review this tree",
    )
    slot = ReviewSlot(
        slot_id="session-slot",
        model="cursor/test",
        route=ReviewRouteKind.AGENT_SESSION,
        timeout_sec=0.2,
    )

    try:
        actor = run_review_request(
            request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx,
        ).actors[0]
        assert actor["operation_state"] == "in_flight"
        assert actor["usage"]["pending_invocation_id"] == "inv-shared"
        assert actor["usage"]["delegated_run_id"] == "run-shared"
        assert checkpoints == [{
            "surface": "multi_model_review",
            "slot_id": "session-slot",
            "operation_id": actor["operation_id"],
            "invocation_id": "inv-shared",
        }]
        joined = run_review_request(
            request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx,
        ).actors[0]
        assert executor.execute_calls == 1
        assert joined["operation_id"] == actor["operation_id"]
        assert joined["usage"]["pending_invocation_id"] == "inv-shared"
        assert joined["usage"]["delegated_run_id"] == "run-shared"
    finally:
        release.set()
        assert settled.wait(1.0)


def test_review_owner_deadline_rechecked_after_prompt_persistence(
    tmp_path, monkeypatch,
):
    """A slow prompt/custody phase must not create a late paid API send."""
    import datetime as dt
    from types import SimpleNamespace

    import ouroboros.deadline_utils as deadlines
    import ouroboros.review_substrate as substrate
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    clock = [dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)]
    monkeypatch.setattr(deadlines, "utc_now", lambda: clock[0])
    original_persist = substrate.persist_call

    def slow_prompt_persist(drive_root, *, call_type="", **kwargs):
        if str(call_type).endswith("_prompt"):
            clock[0] += dt.timedelta(seconds=400)
        return original_persist(drive_root, call_type=call_type, **kwargs)

    monkeypatch.setattr(substrate, "persist_call", slow_prompt_persist)

    class NeverCalled:
        calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            raise AssertionError("expired review owner deadline must not dispatch")

    llm = NeverCalled()
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="late-review",
        call_type="multi_model_review",
        deadline_at=(clock[0] + dt.timedelta(seconds=300)).isoformat(),
    )
    slot = ReviewSlot(
        slot_id="slot-1", model="openai/test", route=ReviewRouteKind.API_CHAT,
        timeout_sec=30,
    )
    result = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=llm,
        usage_ctx=SimpleNamespace(),
    )

    actor = result.actors[0]
    assert llm.calls == 0
    assert actor["operation_state"] == "not_dispatched"
    assert actor["status"] == "not_dispatched"


def test_coordinator_rejoins_exact_recovery_after_spent_owner_deadline(
    tmp_path, monkeypatch,
):
    """The public coordinator must not launder a paid recovery as $0 dispatch."""
    from types import SimpleNamespace

    from ouroboros.review_custody import (
        merge_frozen_review_reconciliation, prepare_frozen_review_reconciliation,
    )
    from ouroboros.review_execution import ReviewAttemptResult
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    class RecoveringExecutor:
        def __init__(self):
            self.state = None
            self.execute_calls = 0

        def restore_custody(self, state):
            self.state = state

        def set_pending_invocation_checkpoint(self, _checkpoint):
            return None

        def prompt_payload(self):
            return {"session_prompt": "review"}

        def prompt_chars(self):
            return 6

        def execute(self):
            self.execute_calls += 1
            return ReviewAttemptResult(
                message={"content": "[]"}, usage={}, raw_text="[]",
            )

        def failure_custody(self):
            return dict(self.state or {})

    executor = RecoveringExecutor()
    monkeypatch.setattr(
        "ouroboros.review_substrate._review_route_executor",
        lambda *_args, **_kwargs: executor,
    )
    ctx = SimpleNamespace(
        task_id="spent-recovery",
        drive_root=tmp_path,
        task_metadata={"deadline_at": "2000-01-01T00:00:00Z"},
    )
    prepare_frozen_review_reconciliation(ctx, SimpleNamespace(
        triad_raw_results=[{
            "slot_id": "slot-1", "model_id": "test/model", "status": "error",
            "operation_id": "op-existing", "operation_state": "in_flight",
            "late_result_pending": True, "pending_invocation_id": "inv-existing",
        }],
        scope_raw_result={},
    ))
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="spent-recovery",
        retry_key="commit_review:spent-recovery", reconcile_only=True,
        deadline_at="2000-01-01T00:00:00Z",
    )
    result = run_review_request(
        request,
        slots=[ReviewSlot(
            slot_id="slot-1", model="test/model",
            route="agent_session",
        )],
        drive_root=tmp_path,
        usage_ctx=ctx,
    )

    actor = result.actors[0]
    assert executor.execute_calls == 1
    assert actor["operation_id"] == "op-existing"
    assert actor["operation_state"] == "settled"
    assert actor["status"] == "ok"
    ctx._last_triad_raw_results = result.actors
    merge_frozen_review_reconciliation(ctx)
    row = ctx._last_triad_raw_results[0]
    assert row["operation_state"] == "settled"
    assert row.get("pending_invocation_id", "") == ""


def test_durable_rows_carry_delegated_restart_identity():
    """One wave, one row shape: every seat's actor record — a packet seat's and a
    retrieving seat's alike — carries the delegated restart identity."""
    from ouroboros.tools.review import _parse_model_response
    from ouroboros.triad_review import parse_seat_answers

    def _envelope(slot_id, op, inv, run):
        return _parse_model_response("cursor/test", {
            "choices": [{"message": {"content": "[]"}}], "slot_id": slot_id,
            "operation_id": op, "operation_state": "in_flight",
            "late_result_pending": True,
            "usage": {"pending_invocation_id": inv, "delegated_run_id": run},
        }, None)

    parsed = parse_seat_answers(
        {"results": [_envelope("slot_1", "op-1", "inv-1", "run-1"), _envelope("slot_2", "op-2", "inv-2", "run-2")]},
        {"slot_1": ("change",), "slot_2": ("change", "coupling")})
    rows = {record.slot_id: record.to_dict() for record in parsed.actor_records}
    assert rows["slot_1"]["pending_invocation_id"] == "inv-1"
    assert rows["slot_1"]["delegated_run_id"] == "run-1"
    assert rows["slot_2"]["pending_invocation_id"] == "inv-2"
    assert rows["slot_2"]["delegated_run_id"] == "run-2"


def test_review_does_not_retry_an_unknown_dispatched_api_attempt(tmp_path):
    from types import SimpleNamespace

    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _NO_RESEND, _attempt_key
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    class _AmbiguousReviewLLM:
        calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            exc = TimeoutError("review provider outcome unknown")
            exc.physical_attempt_capture = SimpleNamespace(
                state="unresolved", provider_status_code=None,
                provider_code="", provider_error_type="TimeoutError",
            )
            raise exc

    llm = _AmbiguousReviewLLM()
    paid = []
    ctx = SimpleNamespace(
        task_id="task-review", event_queue=None, pending_events=[],
        _review_paid_stamp=lambda: paid.append("paid"),
    )
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="task-review",
        call_type="multi_model_review", retry_key="commit_review:unknown",
    )
    slot = ReviewSlot(slot_id="slot-1", model="openai/test")
    key = _attempt_key(request, slot)
    try:
        first = run_review_request(
            request, slots=[slot], drive_root=tmp_path, llm=llm, usage_ctx=ctx,
        )
        second = run_review_request(
            request, slots=[slot], drive_root=tmp_path, llm=llm, usage_ctx=ctx,
        )

        assert llm.calls == 1
        assert paid == ["paid"]
        assert first.actors[0]["status"] == "error"
        assert first.actors[0]["operation_state"] == "custody_lost"
        assert first.actors[0]["late_result_pending"] is True
        assert second.actors[0]["operation_id"] == first.actors[0]["operation_id"]
        assert second.actors[0]["operation_state"] == "custody_lost"
        assert second.actors[0]["failure_code"] == "provider_outcome_unknown"
        assert second.actors[0]["late_result_pending"] is True
        assert ctx._review_custody_lost is True
        with _ACTIVE_LOCK:
            assert key not in _ACTIVE
            assert _NO_RESEND[key] == first.actors[0]["operation_id"]
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.pop(key, None)
            _NO_RESEND.pop(key, None)


@pytest.mark.parametrize(
    ("surface", "raw_text"),
    [("task_acceptance", "malformed reviewer output"), ("multi_model_review", "")],
)
def test_review_retry_rail_honors_durable_cancel_for_format_and_empty_paths(
    tmp_path, monkeypatch, surface, raw_text,
):
    """A cancel that lands after attempt one forbids every retry shape."""
    from types import SimpleNamespace

    from ouroboros.cancel_intents import request_cancel
    from ouroboros.review_execution import ReviewAttemptResult, ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    calls = []
    task_id = f"cancel-retry-{surface.replace('_', '-')}"

    def attempt(_assignment, **_kwargs):
        calls.append(1)
        request_cancel(tmp_path, task_id, reason="owner stopped retry rail", source="test")
        return ReviewAttemptResult(
            message={"content": raw_text}, usage={}, raw_text=raw_text,
        )

    monkeypatch.setattr("ouroboros.review_substrate._execute_slot_attempt", attempt)
    ctx = SimpleNamespace(task_id=task_id, drive_root=tmp_path)
    result = run_review_request(
        ReviewRequest(
            surface=surface, goal="review", task_id=task_id,
            evidence={}, call_type=f"{surface}_review",
        ),
        slots=[ReviewSlot(
            slot_id="slot-1", model="test/model", route=ReviewRouteKind.API_CHAT,
        )],
        drive_root=tmp_path, usage_ctx=ctx,
    )

    assert calls == [1]
    assert result.actors[0]["raw_text"] == raw_text


@pytest.mark.parametrize(
    ("surface", "raw_text"),
    [("task_acceptance", "malformed reviewer output"), ("multi_model_review", "")],
)
def test_review_retry_rail_honors_logical_root_cancel_from_physical_retry_leaf(
    tmp_path, monkeypatch, surface, raw_text,
):
    """A proven root-retry leaf cannot escape its logical cascade stop."""
    from types import SimpleNamespace

    from ouroboros.cancel_intents import (
        SCOPE_CASCADE,
        STOP_POLICY_FINALIZE,
        request_cancel,
    )
    from ouroboros.review_execution import ReviewAttemptResult, ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
    from ouroboros.task_results import write_task_result

    root_id = f"logical-cancel-{surface.replace('_', '-')}"
    leaf_id = f"physical-retry-{surface.replace('_', '-')}"
    write_task_result(
        tmp_path, root_id, "interrupted", root_task_id=root_id,
        delegation_role="root", superseded_by=leaf_id, retry_task_id=leaf_id,
    )
    write_task_result(
        tmp_path, leaf_id, "running", root_task_id=root_id, parent_task_id="",
        delegation_role="root", supersedes_task_id=root_id,
        original_task_id=root_id, timeout_retry_from=root_id,
    )
    calls = []

    def attempt(_assignment, **_kwargs):
        calls.append(1)
        request_cancel(
            tmp_path, root_id, reason="owner stopped logical retry tree",
            source="test", scope=SCOPE_CASCADE,
            requested_stop_policy=STOP_POLICY_FINALIZE,
        )
        return ReviewAttemptResult(
            message={"content": raw_text}, usage={}, raw_text=raw_text,
        )

    monkeypatch.setattr("ouroboros.review_substrate._execute_slot_attempt", attempt)
    ctx = SimpleNamespace(
        task_id=leaf_id, drive_root=tmp_path,
        task_metadata={"root_task_id": root_id},
    )
    result = run_review_request(
        ReviewRequest(
            surface=surface, goal="review", task_id=leaf_id,
            evidence={}, call_type=f"{surface}_review",
        ),
        slots=[ReviewSlot(
            slot_id="slot-1", model="test/model", route=ReviewRouteKind.API_CHAT,
        )],
        drive_root=tmp_path, usage_ctx=ctx,
    )

    assert calls == [1]
    assert result.actors[0]["raw_text"] == raw_text


def test_a_collection_while_a_released_slot_runs_keeps_the_settled_roster(tmp_path):
    """Fix cycle 4, J2: a $0 collection (drain window 0, reconcile_only) while released
    slots are still running re-registers the released wave. The outcomes already recorded
    for that wave must survive the re-registration: otherwise the roster shrinks to the
    still-pending slots and the final settled-wave frame under-counts, while the frame
    promises fewer than the roster size only when a slot timed out before the release.
    One frame is still written for the wave, once every roster slot has an outcome."""
    import threading
    import time
    from types import SimpleNamespace

    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewSlot
    from tests.test_plan_review_event_route import (
        _custody_kwargs,
        _held_worker,
        _mailbox_entries,
    )

    slots = [ReviewSlot(slot_id=f"s{index}", model=f"m/{index}", timeout_sec=30.0)
             for index in (1, 2, 3)]
    release = {slot.slot_id: threading.Event() for slot in slots}
    entered = {slot.slot_id: threading.Event() for slot in slots}
    _calls, run_slot = _held_worker(slots, {}, release, entered)
    progress = []
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=progress.append)
    request, kwargs = _custody_kwargs(
        tmp_path, surface="plan_review", retry_key="plan_review:" + "e" * 64 + ":1",
        slots=slots, run_slot=run_slot, ctx=ctx)
    request.drain_deadline = time.monotonic()
    deadline = time.time() + 10
    try:
        first = custody.run_custodied_review_slots(**kwargs)
        assert {actor.operation_state for actor in first} == {"pending_dispatch"}
        assert all(entered[slot.slot_id].wait(10) for slot in slots)
        release["s1"].set()
        release["s2"].set()
        def first_two_settled():
            return all(any(f"reviewer m/{slot_id[1:]} answered" in line
                           for line in progress) for slot_id in ("s1", "s2"))
        while not first_two_settled() and time.time() < deadline:
            time.sleep(0.01)
        assert first_two_settled(), progress
        assert _mailbox_entries(tmp_path, request.task_id) == []  # s3 is still running
        request.reconcile_only, request.drain_deadline = True, time.monotonic()
        collected = custody.run_custodied_review_slots(**kwargs)
        assert sorted((actor.slot_id, actor.status) for actor in collected) == [
            ("s1", "ok"), ("s2", "ok"), ("s3", "error")]
        assert _mailbox_entries(tmp_path, request.task_id) == []  # the collection writes no frame
        release["s3"].set()
        while not _mailbox_entries(tmp_path, request.task_id) and time.time() < deadline:
            time.sleep(0.01)
    finally:
        for event in release.values():
            event.set()
    frames = _mailbox_entries(tmp_path, request.task_id)
    assert len(frames) == 1, [frame["text"] for frame in frames]
    assert "3 of 3 reviewer slot(s) settled (3 ok, 0 failed)" in frames[0]["text"]
    assert not custody._RELEASED_WAVES


def test_poll_blind_spot_on_a_started_run_projects_in_flight_never_settled_or_retried():
    """#1547 link 2, the classifier table: a typed observation blind spot on a
    STARTED delegated run is ``in_flight`` — not the settled failure a started
    run's other exceptions become, and never a $0 retry."""
    from ouroboros.review_custody import _worker_exception_operation_state as state_of
    from ouroboros.review_execution import ReviewPollUnavailable

    blind = ReviewPollUnavailable("blind on both reads", run_id="run-9")
    assert state_of(blind, {}) == "in_flight"
    assert state_of(blind, {"pending_invocation_id": "inv-9"}) == "in_flight"
    assert state_of(TimeoutError("slot budget exceeded"), {"pending_invocation_id": "inv-9"}) == "settled"


def test_an_unobservable_live_session_stays_pending_for_an_attach_only_rejoin(tmp_path):
    """#1547 link 2 at the custody seam: the worker that could not observe its
    live run ends ``in_flight`` with its pending invocation re-registered, the
    row is a typed delivery failure (not a settled terminal, no replay cache),
    and the next collect rejoins the SAME operation with that token instead of
    dispatching a second paid run."""
    from types import SimpleNamespace

    from ouroboros.review_custody import _attempt_key, run_custodied_review_slots
    from ouroboros.review_execution import ReviewPollUnavailable, ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    calls, ctx = [], SimpleNamespace(drive_root=tmp_path, task_id="blind-1547")
    request = ReviewRequest(surface="plan_review", goal="review", task_id="blind-1547",
                            retry_key="plan_review:blind-1547")
    slot = ReviewSlot(slot_id="seat", model="cursor/test", route=ReviewRouteKind.AGENT_SESSION, timeout_sec=10.0)

    def run_slot(slot, operation_id, retry_state, _deadline, _checkpoint):
        calls.append((operation_id, dict(retry_state)))
        if len(calls) == 1:
            retry_state["pending_invocation_id"], retry_state["delegated_run_id"] = "inv-b", "run-b"
            raise ReviewPollUnavailable("blind on both reads", run_id="run-b")
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="ok", raw_text="[]")

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="error", error=error,
                                 operation_id=operation_id, operation_state=operation_state,
                                 late_result_pending=operation_state == "in_flight")

    args = dict(request=request, slots=[slot], usage_ctx=ctx, task_id=request.task_id, usage_meta={},
                review_usage_scope=UsageScope(drive_root=tmp_path, task_id=request.task_id),
                run_slot=run_slot, error_actor=error_actor)
    [first] = run_custodied_review_slots(**args)
    assert first.operation_state == "in_flight" and first.late_result_pending is True
    assert first.failure_code == "review_poll_unavailable" and first.usage["pending_invocation_id"] == "inv-b"
    assert ctx._review_pending_invocations[_attempt_key(request, slot)]["pending_invocation_id"] == "inv-b"
    assert not getattr(ctx, "_review_settled_attempts", {})
    [second] = run_custodied_review_slots(**args)
    assert second.status == "ok" and second.operation_id == first.operation_id
    assert calls[1] == (first.operation_id, {"pending_invocation_id": "inv-b"})
