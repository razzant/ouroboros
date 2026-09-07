from __future__ import annotations

import datetime as dt

import pytest


def test_logical_review_window_is_narrowed_by_owner_deadline():
    from ouroboros.deadline_utils import logical_operation_timeout_sec

    deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=5)).isoformat()
    value = logical_operation_timeout_sec(300, deadline_at=deadline, fallback=2700)
    assert 0 < value <= 5


def test_logical_review_window_uses_transport_only_as_settlement_fallback():
    from ouroboros.deadline_utils import logical_operation_timeout_sec

    assert logical_operation_timeout_sec(None, fallback=17) == 17


def test_logical_review_window_does_not_widen_explicit_zero():
    from ouroboros.deadline_utils import logical_operation_timeout_sec

    assert logical_operation_timeout_sec(0, fallback=2700) == 1
    assert logical_operation_timeout_sec(-1, fallback=2700) == 1


def test_transport_timeout_is_narrowed_by_numeric_owner_deadline(monkeypatch):
    import ouroboros.deadline_utils as deadlines

    monkeypatch.setattr(deadlines.time, "time", lambda: 1000.0)
    assert deadlines.transport_timeout_with_deadline(90) == 90
    assert deadlines.transport_timeout_with_deadline(
        90, deadline_ts=1010.0, reserve_sec=3,
    ) == 7.0
    # A spent deadline stays a tiny bounded transport operation, never an
    # unbounded/default 90-second child process.
    assert deadlines.transport_timeout_with_deadline(90, deadline_ts=999.0) == 0.001


def test_update_letter_timeout_is_a_clamped_config_getter(monkeypatch):
    from ouroboros.config import SETTINGS_DEFAULTS, get_update_letter_timeout_sec

    monkeypatch.delenv("OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC", raising=False)
    assert get_update_letter_timeout_sec() == float(SETTINGS_DEFAULTS["OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC"])
    monkeypatch.setenv("OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC", "not-a-number")
    assert get_update_letter_timeout_sec() == 120.0
    monkeypatch.setenv("OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC", "0")
    assert get_update_letter_timeout_sec() == 10.0
    monkeypatch.setenv("OUROBOROS_UPDATE_LETTER_TIMEOUT_SEC", "99999")
    assert get_update_letter_timeout_sec() == 600.0


def test_dispatch_window_distinguishes_no_deadline_from_spent_deadline(monkeypatch):
    import ouroboros.deadline_utils as deadlines

    monkeypatch.setattr(deadlines.time, "time", lambda: 1000.0)
    assert deadlines.dispatch_window_remaining_sec() is None
    assert deadlines.dispatch_window_remaining_sec(deadline_ts=1010.0, reserve_sec=3) == 7.0
    assert deadlines.dispatch_window_remaining_sec(deadline_ts=999.0) == 0.0


def test_owner_deadline_admission_can_reserve_settlement_window(monkeypatch):
    import ouroboros.deadline_utils as deadlines

    monkeypatch.setattr(deadlines.time, "time", lambda: 1000.0)
    assert deadlines.owner_deadline_exhausted(deadline_ts=1005.0) is False
    assert deadlines.owner_deadline_exhausted(
        deadline_ts=1005.0, reserve_sec=5,
    ) is True


def test_spent_web_search_deadline_never_reverts_to_the_provider_default(monkeypatch):
    from types import SimpleNamespace
    from ouroboros.tools.search import _web_search_transport_timeout

    monkeypatch.setenv("OUROBOROS_WEBSEARCH_TIMEOUT_SEC", "480")
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    ctx = SimpleNamespace(task_metadata={"deadline_at": "2000-01-01T00:00:00Z"})
    assert _web_search_transport_timeout(ctx) == 0.001


def test_bounded_engine_seconds_never_widens_explicit_zero():
    from ouroboros.deadline_utils import bounded_seconds

    assert bounded_seconds(0, default=300, maximum=3600) == 1
    assert bounded_seconds(0.001, default=300, maximum=3600) == 1
    assert bounded_seconds(1.2, default=300, maximum=3600) == 2
    assert bounded_seconds(None, default=300, maximum=3600) == 300


def test_main_llm_transport_preserves_anthropic_default_but_narrows_deadline(monkeypatch):
    import ouroboros.deadline_utils as deadlines
    from ouroboros.loop_llm_call import _main_transport_timeout

    monkeypatch.setattr(deadlines.time, "time", lambda: 1000.0)
    monkeypatch.setattr("ouroboros.loop_llm_call.get_finalization_grace_sec", lambda: 3)
    assert _main_transport_timeout("anthropic::claude-fable-5", None) == 120
    assert _main_transport_timeout("anthropic::claude-fable-5", 1010.0) == 10.0
    assert _main_transport_timeout(
        "anthropic::claude-fable-5", 1010.0, reserve_sec=3,
    ) == 7.0


def test_low_level_main_transport_admission_and_timeout_share_raw_default(monkeypatch):
    import ouroboros.deadline_utils as deadlines
    from ouroboros.loop_llm_call import _main_transport_timeout

    monkeypatch.setattr(deadlines.time, "time", lambda: 1000.0)
    monkeypatch.setattr("ouroboros.loop_llm_call.get_finalization_grace_sec", lambda: 120)
    assert _main_transport_timeout("openai/gpt-5.5", 1005.0) == 5.0


def test_spent_main_deadline_does_not_dispatch_or_fallback(tmp_path):
    from ouroboros.loop_llm_call import call_llm_with_retry

    class NeverCalled:
        def chat(self, **_kwargs):
            raise AssertionError("spent owner deadline must not call the provider")

    usage = {}
    message, cost = call_llm_with_retry(
        NeverCalled(), [{"role": "user", "content": "x"}], "openai/gpt-5.5",
        None, "high", 1, tmp_path / "logs", "deadline-task", 1, None, usage,
        deadline_ts=1,
    )

    assert message is None and cost is None
    assert usage["_last_llm_error_kind"] == "deadline_exhausted"
    assert usage["reason_code"] == "deadline_exhausted"


def test_main_call_inside_finalization_reserve_does_not_dispatch(tmp_path, monkeypatch):
    import ouroboros.deadline_utils as deadlines
    from ouroboros.loop_llm_call import call_llm_with_retry

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    monkeypatch.setattr(deadlines.time, "time", lambda: 1000.0)

    class NeverCalled:
        def chat(self, **_kwargs):
            raise AssertionError("reserve-only owner window must not call provider")

    usage = {}
    message, cost = call_llm_with_retry(
        NeverCalled(), [{"role": "user", "content": "x"}], "openai/gpt-5.5",
        None, "high", 1, tmp_path / "logs", "reserve-task", 1, None, usage,
        deadline_ts=1005, transport_reserve_sec=120,
    )

    assert message is None and cost is None
    assert usage["reason_code"] == "deadline_exhausted"


def test_main_deadline_is_rechecked_after_slow_message_preparation(tmp_path, monkeypatch):
    import ouroboros.deadline_utils as deadlines
    import ouroboros.loop_llm_call as loop_call

    clock = [100.0]
    monkeypatch.setattr(deadlines.time, "time", lambda: clock[0])

    def slow_prepare(messages, **_kwargs):
        clock[0] = 101.0
        return messages

    monkeypatch.setattr(loop_call, "_prepare_main_messages", slow_prepare)

    class NeverCalled:
        def chat(self, **_kwargs):
            raise AssertionError("preparation-expired owner deadline must not dispatch")

    usage = {}
    message, cost = loop_call.call_llm_with_retry(
        NeverCalled(), [{"role": "user", "content": "x"}], "openai/gpt-5.5",
        None, "high", 1, tmp_path / "logs", "deadline-prep", 1, None, usage,
        deadline_ts=100.5,
    )

    assert message is None and cost is None
    assert usage["reason_code"] == "deadline_exhausted"


def test_nested_logical_window_reserves_finalization_grace():
    from ouroboros.deadline_utils import logical_operation_timeout_sec

    deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=10)).isoformat()
    value = logical_operation_timeout_sec(None, deadline_at=deadline, fallback=2700, reserve_sec=3)
    assert 0 < value <= 7


def test_spent_owner_deadline_has_no_logical_review_window():
    from ouroboros.deadline_utils import logical_operation_timeout_sec

    assert logical_operation_timeout_sec(
        300, deadline_at="2000-01-01T00:00:00Z", fallback=2700, reserve_sec=3,
    ) == 0.0


def test_review_timeout_override_rejects_non_finite_and_invalid_values(monkeypatch, caplog):
    import ouroboros.deadline_utils as deadlines
    from ouroboros.tools import git as git_tools

    for raw in ("inf", "nan", "-1", "garbage"):
        monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", raw)
        assert deadlines.review_logical_fallback_timeout_sec() is None
        assert any(raw in record.message for record in caplog.records)
        # Tool registration must remain available under a malformed operator override.
        assert {entry.name for entry in git_tools.get_tools()} >= {
            "commit_reviewed", "vcs_commit_reviewed",
        }


def test_review_route_owns_unset_logical_fallback(monkeypatch):
    import ouroboros.config as config
    import ouroboros.deadline_utils as deadlines

    monkeypatch.delenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 21_600)
    monkeypatch.setattr(deadlines, "llm_transport_timeout_sec", lambda *_args: 2_700.0)

    assert deadlines.review_operation_timeout_sec(route="api_chat") == 2_700.0
    assert deadlines.review_operation_timeout_sec(route="agent_session") == 21_600.0
    assert deadlines.review_operation_timeout_sec(600, route="agent_session") == 600.0
    monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "900")
    assert deadlines.review_operation_timeout_sec(route="api_chat") == 900.0
    assert deadlines.review_operation_timeout_sec(route="agent_session") == 900.0


def test_plan_task_outer_envelope_covers_agent_session_lifetime(monkeypatch):
    from ouroboros.tools import plan_review

    monkeypatch.setattr(plan_review, "get_llm_transport_read_timeout_sec", lambda: 2700.0)
    monkeypatch.setattr(plan_review, "get_finalization_grace_sec", lambda: 120.0)
    monkeypatch.setattr(plan_review, "get_task_abs_ceiling_sec", lambda: 21_600.0)

    # max(transport + 2*grace, task ceiling + grace), not a short API-only
    # wrapper that can return while an agent-session worker is still paid/live.
    assert plan_review._plan_task_tool_timeout_sec() == 21_720.0
    entry = next(item for item in plan_review.get_tools() if item.name == "plan_task")
    assert entry.timeout_sec == 21_720.0


def test_reconcile_only_missing_custody_never_dispatches(tmp_path):
    from types import SimpleNamespace

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    request = ReviewRequest(
        surface="multi_model_review",
        goal="review",
        task_id="task-1",
        call_type="multi_model_review",
        retry_key="commit_review:exact",
        reconcile_only=True,
    )
    slot = ReviewSlot(slot_id="slot-1", model="openai/test", route=ReviewRouteKind.API_CHAT)
    ctx = SimpleNamespace()
    calls = []

    def error_actor(_slot, error, operation_id="", operation_state="settled"):
        return SimpleNamespace(
            slot_id="slot-1", status="error", error=error, usage={},
            response_ref={}, operation_id=operation_id,
            operation_state=operation_state,
            late_result_pending=operation_state == "in_flight",
        )

    actors = run_custodied_review_slots(
        request=request,
        slots=[slot],
        usage_ctx=ctx,
        task_id="task-1",
        usage_meta={},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id="task-1"),
        run_slot=lambda *_args: calls.append("dispatched"),
        error_actor=error_actor,
    )

    assert calls == []
    assert actors[0].operation_state == "custody_lost"
    assert ctx._review_custody_lost is True


def test_strict_paid_stamp_failure_starts_no_worker_and_leaks_no_active_row(tmp_path):
    from types import SimpleNamespace

    from ouroboros.review_custody import _ACTIVE, _attempt_key, run_custodied_review_slots
    from ouroboros.review_dispatch import ReviewPaidStamp
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="task-stamp",
        retry_key="commit_review:stamp",
    )
    slot = ReviewSlot(slot_id="slot-1", model="openai/test")
    calls = []

    def fail_write():
        raise OSError("state write failed")

    stamp = ReviewPaidStamp(fail_write, fail_closed=True)
    ctx = SimpleNamespace(_review_paid_stamp=stamp)
    with pytest.raises(OSError, match="state write failed"):
        run_custodied_review_slots(
            request=request,
            slots=[slot],
            usage_ctx=ctx,
            task_id="task-stamp",
            usage_meta={},
            review_usage_scope=UsageScope(drive_root=tmp_path, task_id="task-stamp"),
            run_slot=lambda *_args: calls.append("sent"),
            error_actor=lambda *_args, **_kwargs: None,
        )

    assert calls == []
    # Strict task-acceptance stamps latch the failed write so every parallel
    # dispatcher replays the same refusal without attempting another payment.
    assert stamp.fired is True
    assert _attempt_key(request, slot) not in _ACTIVE


def test_default_paid_stamp_remains_fail_open_for_skill_marker():
    from types import SimpleNamespace

    from ouroboros.review_dispatch import ReviewPaidStamp, stamp_review_paid_on_dispatch

    def fail_write():
        raise OSError("marker unavailable")

    stamp = ReviewPaidStamp(fail_write)
    stamp_review_paid_on_dispatch(SimpleNamespace(_review_paid_stamp=stamp))
    assert stamp.fired is True


def test_paid_reviewing_row_without_late_flag_is_exact_resume_candidate(tmp_path):
    from types import SimpleNamespace

    from ouroboros.review_state import (
        AdvisoryReviewState, CommitAttemptRecord, make_repo_key, save_state,
    )
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt

    repo = tmp_path / "repo"
    repo.mkdir()
    state_root = tmp_path / "data"
    row = CommitAttemptRecord(
        ts="2026-01-01T00:00:00+00:00",
        commit_message="pending paid wave",
        status="reviewing",
        repo_key=make_repo_key(repo),
        tool_name="commit_reviewed",
        task_id="task-paid",
        attempt=4,
        paid=True,
        late_result_pending=False,
        review_retry_key="commit_review:exact",
    )
    save_state(state_root, AdvisoryReviewState(attempts=[row]))
    ctx = SimpleNamespace(
        repo_dir=repo, drive_root=state_root, task_id="task-paid",
        _current_review_tool_name="commit_reviewed",
    )

    assert _check_overlapping_review_attempt(ctx) is None
    assert ctx._review_resume_pending is True
    assert ctx._pending_review_attempt.review_retry_key == "commit_review:exact"
    assert ctx._current_review_attempt_number == 4


def test_unreadable_review_state_blocks_before_paid_dispatch(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import ouroboros.review_state as review_state
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt

    monkeypatch.setattr(
        review_state, "update_state",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("ledger unavailable")),
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="task-state-failure",
        _current_review_tool_name="commit_reviewed",
    )

    message = _check_overlapping_review_attempt(ctx)
    assert message and "REVIEW_STATE_UNAVAILABLE" in message
    assert ctx._review_resume_pending is False


def test_frozen_roster_survives_reassembly_exit_and_stays_pending():
    from types import SimpleNamespace

    from ouroboros.review_custody import (
        merge_frozen_review_reconciliation, prepare_frozen_review_reconciliation,
    )
    from ouroboros.tools.git import _review_custody_pending

    terminal = {
        "slot_id": "slot_1", "model_id": "m1", "status": "responded",
        "raw_text": "[]", "operation_id": "op-1", "operation_state": "settled",
        "late_result_pending": False,
    }
    pending = {
        "slot_id": "slot_2", "model_id": "m2", "status": "error",
        "raw_text": "", "operation_id": "op-2", "operation_state": "in_flight",
        "late_result_pending": True, "pending_invocation_id": "inv-2",
    }
    ctx = SimpleNamespace(_last_triad_raw_results=[], _last_scope_raw_result={})
    attempt = SimpleNamespace(
        triad_raw_results=[terminal, pending], scope_raw_result={},
    )
    prepare_frozen_review_reconciliation(ctx, attempt)

    # Simulate a fresh assembly/admission exit that produced no actor rows.
    merge_frozen_review_reconciliation(ctx)

    assert ctx._last_triad_raw_results[0] == terminal
    assert ctx._last_triad_raw_results[1]["operation_id"] == "op-2"
    assert ctx._last_triad_raw_results[1]["operation_state"] == "custody_lost"
    assert _review_custody_pending(ctx) is True


def test_reconcile_current_roster_cannot_dispatch_unmatched_slot(tmp_path):
    from types import SimpleNamespace

    from ouroboros.review_custody import (
        prepare_frozen_review_reconciliation, run_custodied_review_slots,
    )
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    attempt = SimpleNamespace(triad_raw_results=[{
        "slot_id": "slot-1", "model_id": "m1", "status": "responded",
        "raw_text": "[]", "operation_id": "op-1", "operation_state": "settled",
    }], scope_raw_result={})
    ctx = SimpleNamespace()
    prepare_frozen_review_reconciliation(ctx, attempt)
    calls = []

    def error_actor(slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error", error=error,
            operation_id=operation_id, operation_state=operation_state,
            late_result_pending=operation_state in {"in_flight", "custody_lost"},
        )

    actors = run_custodied_review_slots(
        request=ReviewRequest(
            surface="multi_model_review", goal="review", task_id="task-roster",
            retry_key="commit_review:roster", reconcile_only=True,
        ),
        slots=[
            ReviewSlot(slot_id="slot-1", model="m1"),
            ReviewSlot(slot_id="slot-new", model="m-new"),
        ],
        usage_ctx=ctx,
        task_id="task-roster",
        usage_meta={},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id="task-roster"),
        run_slot=lambda *_args: calls.append("sent"),
        error_actor=error_actor,
    )

    assert calls == []
    assert {actor.slot_id: actor.operation_state for actor in actors} == {
        "slot-1": "settled", "slot-new": "custody_lost",
    }


def test_restart_reconciliation_hydrates_delegated_invocation_without_paid_stamp(tmp_path):
    from types import SimpleNamespace

    from ouroboros.review_custody import (
        prepare_frozen_review_reconciliation, run_custodied_review_slots,
    )
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.usage_accounting import UsageScope

    attempt = SimpleNamespace(triad_raw_results=[{
        "slot_id": "slot-1", "model_id": "cursor/test", "status": "error",
        "operation_id": "op-existing", "operation_state": "in_flight",
        "late_result_pending": True, "pending_invocation_id": "inv-existing",
    }], scope_raw_result={})
    paid = []
    ctx = SimpleNamespace(_review_paid_stamp=lambda: paid.append("paid"))
    prepare_frozen_review_reconciliation(ctx, attempt)
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="task-restart",
        retry_key="commit_review:exact", reconcile_only=True,
    )
    slot = ReviewSlot(
        slot_id="slot-1", model="cursor/test", route=ReviewRouteKind.AGENT_SESSION,
    )
    calls = []

    def recover(_slot, operation_id, retry_state, _deadline, _checkpoint):
        calls.append((operation_id, dict(retry_state)))
        return ReviewActorRecord(slot_id="slot-1", model="cursor/test", status="ok")

    [actor] = run_custodied_review_slots(
        request=request,
        slots=[slot],
        usage_ctx=ctx,
        task_id="task-restart",
        usage_meta={},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id="task-restart"),
        run_slot=recover,
        error_actor=lambda *_args, **_kwargs: None,
    )

    assert calls == [("op-existing", {"pending_invocation_id": "inv-existing"})]
    assert paid == []
    assert actor.operation_id == "op-existing"
    assert actor.operation_state == "settled"


def test_paid_or_pending_review_attempt_never_ttl_authorizes_a_resend():
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord

    old = "2020-01-01T00:00:00+00:00"
    state = AdvisoryReviewState(attempts=[
        CommitAttemptRecord(ts=old, commit_message="unpaid", status="reviewing", attempt=1),
        CommitAttemptRecord(
            ts=old, commit_message="paid", status="reviewing", attempt=2,
            paid=True, review_retry_key="commit_review:paid",
        ),
        CommitAttemptRecord(
            ts=old, commit_message="late", status="reviewing", attempt=3,
            late_result_pending=True, review_retry_key="commit_review:late",
        ),
    ])

    expired = state.expire_stale_attempts(now_ts="2026-01-01T00:00:00+00:00")

    assert [item.commit_message for item in expired] == ["unpaid"]
    assert state.attempts[1].status == "reviewing"
    assert state.attempts[2].late_result_pending is True


def test_commit_retry_key_round_trips_in_review_state(tmp_path):
    from ouroboros.review_state import (
        AdvisoryReviewState, CommitAttemptRecord, load_state, save_state,
    )

    save_state(tmp_path, AdvisoryReviewState(attempts=[CommitAttemptRecord(
        ts="2026-01-01T00:00:00+00:00",
        commit_message="pending",
        status="reviewing",
        review_retry_key="commit_review:roundtrip",
    )]))

    assert load_state(tmp_path).attempts[0].review_retry_key == "commit_review:roundtrip"


def test_exact_pending_commit_retry_reconciles_before_cycle_cap(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros.tools import git as git_tools

    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        task_id="task-1",
        task_metadata={},
        _review_resume_pending=True,
        _pending_review_attempt=SimpleNamespace(review_retry_key="commit_review:exact"),
    )
    monkeypatch.setattr(git_tools, "_commit_review_retry_key", lambda *_a, **_k: "commit_review:exact")
    monkeypatch.setattr(git_tools, "commit_review_contract_fingerprint", lambda: "contract")
    monkeypatch.setattr(
        git_tools, "check_review_cycles_ceiling",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cap must not gate reconciliation")),
    )
    monkeypatch.setattr(
        git_tools, "check_identical_verdict_refusal",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("replay gate must not run")),
    )

    outcome = git_tools._free_cycle_gate(
        ctx,
        "message",
        0.0,
        pre_fingerprint={"fingerprint": "binding"},
        review_rebuttal="",
        goal="goal",
        scope="scope",
    )

    assert outcome is None
    assert ctx._review_reconcile_only is True
    assert ctx._current_review_retry_key == "commit_review:exact"


def test_commit_pending_retry_reconciles_same_paid_attempt(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    import ouroboros.review_custody as review_custody
    from ouroboros.review_custody import prepare_frozen_review_reconciliation
    from ouroboros.review_state import load_state, make_repo_key
    from ouroboros.tools import git as git_tools
    from ouroboros.tools.parallel_review import _reserve_parallel_review_roster

    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Test"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    path = repo / "value.txt"
    path.write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "value.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    path.write_text("two\n", encoding="utf-8")

    drive = tmp_path / "data"
    (drive / "logs").mkdir(parents=True)
    ctx = SimpleNamespace(
        repo_dir=repo, drive_root=drive, drive_path=lambda name: drive / name,
        task_id="task-commit-custody",
        task_metadata={}, event_queue=None, current_task_type="", parent_task_id="",
        emit_progress_fn=lambda *_args: None, drive_logs=lambda: drive / "logs",
    )
    monkeypatch.setattr(git_tools, "_advisory_and_tests_gate", lambda *_a, **_k: None)
    monkeypatch.setattr(git_tools, "_review_binding_precondition_error", lambda *_a, **_k: "")
    monkeypatch.setattr(git_tools, "commit_review_contract_fingerprint", lambda: "contract")
    monkeypatch.setattr(
        git_tools, "_aggregate_review_verdict",
        lambda *_a, **_k: (False, "", "", [], []),
    )
    reserved_reconciliations = []
    real_reserved_reconciliation = review_custody.reconcile_reserved_review_roster

    def reconcile_reserved(run_ctx, reserved):
        reserved_reconciliations.append(reserved)
        return real_reserved_reconciliation(run_ctx, reserved)

    monkeypatch.setattr(
        review_custody, "reconcile_reserved_review_roster", reconcile_reserved,
    )
    waves = 0
    operation_id = ""

    def wave(run_ctx, *_args, **_kwargs):
        nonlocal operation_id, waves
        waves += 1
        if waves == 1:
            _reserve_parallel_review_roster(
                run_ctx,
                {"row_plan": {
                    "models": ["m1"],
                    "routes": ["api_chat"],
                    "efforts": ["high"],
                    "slot_ids": ["slot_1"],
                }},
                [],
            )
            operation_id = run_ctx._review_reserved_operations[
                "multi_model_review"
            ]["slot_1"]
            run_ctx._last_triad_raw_results = [{
                "slot_id": "slot_1", "model_id": "m1", "status": "error",
                "operation_id": operation_id, "operation_state": "in_flight",
                "late_result_pending": True,
            }]
        else:
            assert run_ctx._review_reconcile_only is True
            prepare_frozen_review_reconciliation(run_ctx, run_ctx._pending_review_attempt)
            run_ctx._last_triad_raw_results = [{
                "slot_id": "slot_1", "model_id": "m1", "status": "responded",
                "raw_text": "[]", "operation_id": operation_id,
                "operation_state": "settled", "late_result_pending": False,
            }]
        run_ctx._last_scope_raw_result = {}
        return None, None, "critical_findings", []

    monkeypatch.setattr(git_tools, "_run_parallel_review", wave)
    first = git_tools._run_non_committing_review_cycle(ctx, "same message")
    assert first["status"] == "blocked", first
    assert first["block_reason"] == "review_late_result_pending"

    monkeypatch.setattr(
        git_tools, "check_review_cycles_ceiling",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cap must not gate exact custody")),
    )
    second = git_tools._run_non_committing_review_cycle(ctx, "same message")
    assert second["status"] == "passed"
    assert waves == 2
    assert len(reserved_reconciliations) == 1

    rows = load_state(drive).filter_attempts(
        repo_key=make_repo_key(repo), tool_name="commit_reviewed",
        task_id="task-commit-custody",
    )
    assert len(rows) == 1
    assert rows[0].attempt == 1 and rows[0].paid is True
    assert rows[0].late_result_pending is False
    assert rows[0].triad_raw_results[0]["operation_id"] == operation_id
    assert rows[0].triad_raw_results[0]["operation_state"] == "settled"


def test_preflight_timeout_override_rejects_infinity(monkeypatch):
    from ouroboros.preflight_runner import _resolve_preflight_timeout

    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "inf")
    assert _resolve_preflight_timeout(900) == 900


def test_commit_retry_key_uses_canonical_staged_binding_despite_external_diff(
    tmp_path, monkeypatch,
):
    import subprocess
    from types import SimpleNamespace

    from ouroboros.tools.git import _fingerprint_staged_diff
    from ouroboros.tools.parallel_review import _commit_review_retry_key

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    path = repo / "value.txt"
    path.write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "value.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "/usr/bin/true")
    ctx = SimpleNamespace(repo_dir=repo, _current_review_contract_fingerprint="contract")

    path.write_text("two\n", encoding="utf-8")
    subprocess.run(["git", "add", "value.txt"], cwd=repo, check=True)
    first = _fingerprint_staged_diff(repo)
    first_key = _commit_review_retry_key(
        ctx, "msg", goal="goal", scope="scope", review_rebuttal="",
        binding_fingerprint=first["fingerprint"],
    )
    path.write_text("three\n", encoding="utf-8")
    subprocess.run(["git", "add", "value.txt"], cwd=repo, check=True)
    second = _fingerprint_staged_diff(repo)
    second_key = _commit_review_retry_key(
        ctx, "msg", goal="goal", scope="scope", review_rebuttal="",
        binding_fingerprint=second["fingerprint"],
    )

    assert first["fingerprint"] != second["fingerprint"]
    assert first_key != second_key


def test_bounded_delegate_poll_passes_remaining_transport_window():
    from ouroboros.delegate_progress import bounded_poll

    class Gateway:
        def __init__(self):
            self.calls = []

        def get_run(self, run_id, *, timeout_sec=None):
            self.calls.append((run_id, timeout_sec))
            return {"summary": {"state": "succeeded"}}

    gateway = Gateway()
    detail = bounded_poll(gateway, "run-1", 10)
    assert detail["summary"]["state"] == "succeeded"
    assert gateway.calls[0][0] == "run-1"
    assert 0 < gateway.calls[0][1] <= 10


def test_strict_review_poll_does_not_raise_the_remaining_window_to_five_seconds():
    from ouroboros.delegate_progress import bounded_poll

    class Gateway:
        def __init__(self):
            self.timeout = None

        def get_run(self, _run_id, *, timeout_sec=None):
            self.timeout = timeout_sec
            return {"summary": {"state": "succeeded"}}

    gateway = Gateway()
    bounded_poll(gateway, "run-1", 0.001, strict=True)
    assert 0 < gateway.timeout <= 0.001


def test_expiring_strict_review_poll_keeps_the_subsecond_bound():
    from ouroboros.delegate_progress import expiring_poll

    class Gateway:
        def __init__(self):
            self.timeout = None

        def get_run(self, _run_id, *, timeout_sec=None):
            self.timeout = timeout_sec
            return {"summary": {"state": "succeeded"}}

    gateway = Gateway()
    expiring_poll(gateway, "run-1", strict=True)
    assert 0 < gateway.timeout <= 0.001


def test_strict_poll_splits_http_phase_budget_and_recomputes_retry():
    import time
    from ouroboros.delegate_progress import bounded_poll

    class AtomicRace(Exception):
        code = "ENOENT"

    class Gateway:
        def __init__(self):
            self.timeouts = []

        def get_run(self, _run_id, *, timeout_sec=None):
            self.timeouts.append(timeout_sec)
            if len(self.timeouts) == 1:
                # Guarantees monotonic advances between the two poll_bound
                # computations, so the retry ask is measurably below the first.
                time.sleep(0.01)
                raise AtomicRace("/.git/objects/ab/tmp_obj_123")
            return {"summary": {"state": "succeeded"}}

    gateway = Gateway()
    # A coarse budget below the 60s read default: the contract under test is
    # the SPLIT (first ask bounded by the whole budget, retry ask recomputed
    # from the remainder), not stopwatch accuracy.  The previous 0.08s budget
    # required the first phase (thread spawn + 0.01s sleep + raise) to finish
    # with window remaining inside 80ms of wall clock; on a loaded CI host the
    # budget was already spent by the catch, so the injected AtomicRace
    # escaped instead of earning its one re-read (same margin redesign as
    # test_strict_poll_phase_budget_is_bounded_in_wall_time).
    detail = bounded_poll(gateway, "run-1", 30.0, strict=True)
    assert detail == {"summary": {"state": "succeeded"}}
    assert 0 < gateway.timeouts[0] <= 30.0
    assert 0 < gateway.timeouts[1] < gateway.timeouts[0]


def test_strict_poll_phase_budget_is_bounded_in_wall_time():
    import pytest
    import threading
    import time
    from ouroboros.delegate_progress import bounded_poll
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    # Margins are deliberately coarse: the previous 0.06s-sleep-inside-a-0.08s
    # budget with a <0.1s wall assertion measured OS scheduler precision, not
    # the poll contract, and flaked on loaded CI hosts (observed 0.207s). The
    # contract under test is discrimination, not stopwatch accuracy: a phase
    # answering within the budget returns as soon as it answers (seconds, not
    # the 30s budget), and a stalled phase is cut at the small budget (seconds,
    # not the 30s stall).

    class OneSlowPhase:
        def get_run(self, _run_id, *, timeout_sec=None):
            time.sleep(0.05)
            return {"summary": {"state": "succeeded"}}

    started = time.monotonic()
    detail = bounded_poll(OneSlowPhase(), "run-1", 30.0, strict=True)
    assert detail == {"summary": {"state": "succeeded"}}
    assert time.monotonic() - started < 5.0

    stall_release = threading.Event()

    class StalledPhase:
        def get_run(self, _run_id, *, timeout_sec=None):
            stall_release.wait(30.0)
            return {"summary": {"state": "succeeded"}}

    started = time.monotonic()
    with pytest.raises(ClaudexorUnavailable, match="wall-clock bound"):
        bounded_poll(StalledPhase(), "run-2", 0.25, strict=True)
    assert time.monotonic() - started < 5.0
    # Let the abandoned daemon poll thread exit now instead of in 30s.
    stall_release.set()


def test_claudexor_bound_applies_to_connect_phase_too():
    import httpx
    from contextlib import contextmanager
    from ouroboros.gateways import claudexor as cx

    calls = []

    class Recorder:
        @contextmanager
        def stream(self, method, path, **kwargs):
            calls.append(kwargs)
            yield httpx.Response(200, json={"id": "run-1", "summary": {}})

    gateway = cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", 1, "token"))
    gateway.close()
    gateway._client = Recorder()
    gateway.get_run("run-1", timeout_sec=0.001)
    assert calls[-1]["timeout"].read == 0.001
    assert calls[-1]["timeout"].connect == 0.001


def test_main_round_call_propagates_task_attempt(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import ouroboros.loop as loop_mod

    captured = {}
    monkeypatch.setattr(loop_mod.task_pacing, "get_finalization_grace_sec", lambda: 7)

    def fake_call(*args, **kwargs):
        captured.update(kwargs)
        captured["task_attempt"] = args[10].get("_task_attempt")
        return {"content": "ok"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    tools = SimpleNamespace(
        _ctx=SimpleNamespace(task_attempt=7, task_metadata={}),
    )
    ctx = loop_mod._RoundModelCallContext(
        llm=object(), messages=[], tools=tools, context_fit_plan=None,
        active_model="model", tool_schemas=[], active_effort="high",
        max_retries=1, drive_logs=tmp_path / "logs", task_id="task",
        round_idx=1, event_queue=None, accumulated_usage={"_task_attempt": 7}, task_type="task",
        active_use_local=False, active_context_mode="max", drive_root=tmp_path,
    )
    loop_mod._dispatch_round_model(ctx, None, attempt_cap=1)
    assert captured["task_attempt"] == 7
    assert captured["transport_reserve_sec"] == 7


def test_forced_finalization_transport_uses_full_grace_deadline(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import ouroboros.loop as loop_mod

    captured = {}

    def fake_call(*args, **kwargs):
        captured.update(kwargs)
        captured["task_attempt"] = args[10].get("_task_attempt")
        return {"content": "final"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    ctx = loop_mod._RoundLimitContext(
        messages=[], llm=object(), active_model="model", active_effort="high",
        max_retries=1, drive_logs=tmp_path / "logs", task_id="task", round_idx=1,
        event_queue=None, accumulated_usage={"_task_attempt": 4}, task_type="task",
        active_use_local=False, max_rounds=1,
        tools=SimpleNamespace(_ctx=SimpleNamespace(task_attempt=4)),
    )
    ctx.deadline_ts = __import__("time").time() + 3
    assert loop_mod._call_forced_model_once(ctx) == "final"
    assert captured["task_attempt"] == 4
    assert captured["transport_reserve_sec"] == 0.0
    assert 0 < captured["deadline_ts"] - __import__("time").time() <= 3.1


def test_finalize_control_carries_original_grace_deadline(monkeypatch, tmp_path):
    import queue
    import time
    from types import SimpleNamespace
    import ouroboros.loop as loop_mod
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

    monkeypatch.setattr(loop_mod.task_pacing, "effective_finalization_reserve_sec", lambda _ctx: 3)
    before = time.time()
    assert write_owner_message(tmp_path, "deadline", "task", kind=KIND_FINALIZE_NOW)
    controls = loop_mod._drain_incoming_messages(
        [], queue.Queue(), tmp_path, "task", None, set(),
        owner_ctx=SimpleNamespace(task_attempt=1),
    )
    assert controls["finalize_now"] == "deadline"
    # The deadline derives from the mailbox entry's ``ts`` — stamped via
    # datetime.now().isoformat(), i.e. truncated to WHOLE microseconds — while
    # ``before`` keeps time.time()'s full float precision. On Windows the
    # coarse system clock hands both reads the same instant, so the stamp can
    # sit up to 1us BELOW ``before`` (observed −3.4e-7s on the CI shard).
    # Compare against the microsecond-truncated lower bound.
    assert before - 1e-6 + 3 <= controls["finalize_deadline_ts"] <= time.time() + 3


def test_forced_finalization_does_not_rebase_existing_grace(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import ouroboros.loop as loop_mod

    deadline = __import__("time").time() + 1
    ctx = loop_mod._RoundLimitContext(
        messages=[], llm=object(), active_model="model", active_effort="high",
        max_retries=1, drive_logs=tmp_path / "logs", task_id="task", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, max_rounds=1, deadline_ts=deadline,
        tools=SimpleNamespace(_ctx=SimpleNamespace()),
    )
    monkeypatch.setattr(loop_mod, "_finalize_forced_services", lambda *_args: None)
    monkeypatch.setattr(
        loop_mod, "_forced_swarm_router_result",
        lambda *_args: ("routed", {}, {}),
    )
    loop_mod._forced_final_answer(
        ctx, prompt="finish", fallback_text="fallback",
        reason_code="finalization_grace",
    )
    assert ctx.deadline_ts == deadline


def test_expired_supervisor_grace_does_not_dispatch_a_paid_final_call(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import time
    import ouroboros.loop as loop_mod

    observed = {}
    ctx = loop_mod._RoundLimitContext(
        messages=[], llm=object(), active_model="model", active_effort="high",
        max_retries=1, drive_logs=tmp_path / "logs", task_id="task", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, max_rounds=1, deadline_ts=time.time() - 1,
        tools=SimpleNamespace(_ctx=SimpleNamespace()), llm_trace={},
    )
    monkeypatch.setattr(loop_mod, "_finalize_forced_services", lambda *_args: None)
    monkeypatch.setattr(
        loop_mod, "_call_forced_model_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("paid final call")),
    )

    def fake_fallback(_ctx, _trace, text, reason, **kwargs):
        observed.update(text=text, reason=reason, **kwargs)
        return text, _ctx.accumulated_usage, _trace

    monkeypatch.setattr(loop_mod, "_forced_fallback_result", fake_fallback)
    result = loop_mod._handle_forced_finalization(ctx, "idle_timeout")
    assert result[0].startswith("⚠️ Task reached idle_timeout")
    assert observed["source"] == "finalization_grace_window_elapsed"
    assert ctx.accumulated_usage == {
        "execution_status": "failed",
        "reason_code": "finalization_grace",
    }


def test_expired_local_deadline_does_not_dispatch_a_paid_final_call(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import ouroboros.loop as loop_mod

    ctx = loop_mod._RoundLimitContext(
        messages=[], llm=object(), active_model="model", active_effort="high",
        max_retries=1, drive_logs=tmp_path / "logs", task_id="task", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, max_rounds=1, llm_trace={},
    )
    tools = SimpleNamespace(_ctx=SimpleNamespace(
        task_metadata={"deadline_at": "2000-01-01T00:00:00Z"},
    ))
    ctx.tools = tools
    monkeypatch.setattr(loop_mod, "_finalize_forced_services", lambda *_args: None)
    monkeypatch.setattr(
        loop_mod, "_call_forced_model_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("paid final call")),
    )

    result = loop_mod._maybe_deadline_local_finalize(ctx, tools)

    assert result is not None
    assert result[0].startswith("⚠️ Task reached its deadline")
    usage = dict(ctx.accumulated_usage)
    assert usage.pop("terminal_origin") == "host_notice"
    assert usage == {
        "execution_status": "failed",
        "reason_code": "deadline_local",
    }
