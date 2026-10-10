"""Focused frozen-roster reconciliation regressions."""

import json
from types import SimpleNamespace

import pytest

from ouroboros.review_custody import (
    merge_frozen_review_reconciliation,
    prepare_frozen_review_reconciliation,
    reconcile_reserved_review_roster,
)


def test_parallel_commit_roster_is_atomic_and_token_updates_are_exact(tmp_path):
    import concurrent.futures
    import os
    import time

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_state import load_state
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.git import _install_paid_dispatch_stamp
    from ouroboros.tools.parallel_review import _reserve_parallel_review_roster

    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "data"
    ctx = SimpleNamespace(
        repo_dir=repo,
        drive_root=drive,
        task_id="task-roster",
        task_metadata={},
        _current_review_tool_name="commit_reviewed",
        _current_review_retry_key="commit_review:atomic-roster",
        _current_review_contract_fingerprint="contract-1",
        _current_review_rebuttal_sha256="",
        _review_reconcile_only=False,
        _review_advisory=[],
    )
    _install_paid_dispatch_stamp(
        ctx,
        "reserve roster",
        time.time(),
        {"fingerprint": "binding-1"},
    )
    # One wave: the packet seat and the coupling-only seat are rows of the SAME
    # roster surface, each with its own reserved operation.
    plan = {
        "models": ["model-a", "model-b"],
        "routes": [ReviewRouteKind.AGENT_SESSION, ReviewRouteKind.AGENT_SESSION],
        "efforts": ["high", "xhigh"],
        "slot_ids": ["slot-a", "scope-a"],
        "parts": [("change",), ("coupling",)],
    }

    _reserve_parallel_review_roster(ctx, {"row_plan": plan})

    state = load_state(drive)
    attempt = state.attempts[-1]
    assert attempt.paid is True
    assert attempt.late_result_pending is True
    from ouroboros.process_custody import current_custody_session_id

    assert attempt.review_owner_session_id == current_custody_session_id()
    assert attempt.review_owner_pid == os.getpid()
    triad, scope = attempt.triad_raw_results
    assert triad["operation_id"] == ctx._review_reserved_operations[
        "multi_model_review"
    ]["slot-a"]
    assert scope["operation_id"] == ctx._review_reserved_operations[
        "multi_model_review"
    ]["scope-a"]

    from ouroboros.review_custody import run_custodied_review_slots
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest
    from ouroboros.usage_accounting import UsageScope

    seen_operations = []

    def _finish(slot, operation_id, _retry_state, _deadline, _checkpoint):
        seen_operations.append(operation_id)
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="ok")

    [actor] = run_custodied_review_slots(
        request=ReviewRequest(
            surface="multi_model_review",
            goal="review",
            task_id="task-roster",
            retry_key="commit_review:atomic-roster",
        ),
        slots=[ReviewSlot(
            slot_id="slot-a",
            model="model-a",
            route=ReviewRouteKind.AGENT_SESSION,
        )],
        usage_ctx=ctx,
        task_id="task-roster",
        usage_meta={},
        review_usage_scope=UsageScope(drive_root=drive, task_id="task-roster"),
        run_slot=_finish,
        error_actor=lambda *_args, **_kwargs: None,
    )
    assert seen_operations == [triad["operation_id"]]
    assert actor.operation_id == triad["operation_id"]

    checkpoint = ctx._review_pending_invocation_checkpoint
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                checkpoint,
                surface="multi_model_review",
                slot_id="slot-a",
                operation_id=triad["operation_id"],
                invocation_id="inv-triad",
            ),
            pool.submit(
                checkpoint,
                surface="multi_model_review",
                slot_id="scope-a",
                operation_id=scope["operation_id"],
                invocation_id="inv-scope",
            ),
        ]
        for future in futures:
            future.result()

    updated = load_state(drive).attempts[-1]
    assert updated.triad_raw_results[0]["pending_invocation_id"] == "inv-triad"
    assert updated.triad_raw_results[1]["pending_invocation_id"] == "inv-scope"


def test_first_paid_wave_keeps_empty_and_partial_reserved_rosters_in_custody():
    reserved = {
        "multi_model_review": [{
            "slot_id": "triad-a", "operation_id": "op-triad-a",
            "operation_state": "in_flight", "late_result_pending": True,
        }],
        "scope_review": [
            {
                "slot_id": "scope-a", "operation_id": "op-scope-a",
                "operation_state": "in_flight", "late_result_pending": True,
            },
            {
                "slot_id": "scope-b", "operation_id": "op-scope-b",
                "operation_state": "in_flight", "late_result_pending": True,
            },
        ],
    }
    ctx = SimpleNamespace(
        _last_triad_raw_results=[],
        _last_scope_raw_result={
            "raw_results": [
                {
                    "slot_id": "scope-a", "operation_id": "op-scope-a",
                    "operation_state": "settled", "late_result_pending": False,
                },
                {
                    "slot_id": "scope_slot_error", "status": "error",
                    "operation_state": "settled", "late_result_pending": False,
                },
            ],
        },
        _review_custody_lost=False,
    )

    reconcile_reserved_review_roster(ctx, reserved)

    triad = ctx._last_triad_raw_results
    scope = ctx._last_scope_raw_result["raw_results"]
    assert ctx._review_custody_lost is True
    assert triad[0]["operation_id"] == "op-triad-a"
    assert triad[0]["operation_state"] == "custody_lost"
    assert triad[0]["late_result_pending"] is True
    assert scope[0]["operation_id"] == "op-scope-a"
    assert scope[0]["operation_state"] == "settled"
    assert scope[1]["operation_id"] == "op-scope-b"
    assert scope[1]["operation_state"] == "custody_lost"
    assert scope[1]["late_result_pending"] is True
    assert scope[2]["slot_id"] == "scope_slot_error"
    assert scope[2]["operation_state"] == "custody_lost"


def test_row_level_active_custody_reuses_terminal_attempt_instead_of_forking(tmp_path):
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        load_state,
        make_repo_key,
        save_state,
    )
    from ouroboros.tools.commit_gate import _record_commit_attempt

    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "data"
    repo_key = make_repo_key(repo)
    state = AdvisoryReviewState()
    state.record_attempt(CommitAttemptRecord(
        ts="2026-08-24T00:00:00+00:00",
        commit_message="same",
        status="blocked",
        repo_key=repo_key,
        tool_name="commit_reviewed",
        task_id="task-row-custody",
        attempt=1,
        paid=True,
        review_retry_key="commit_review:row-custody",
        triad_raw_results=[{
            "slot_id": "slot-a", "operation_id": "op-a",
            "operation_state": "in_flight", "late_result_pending": True,
        }],
    ))
    save_state(drive, state)
    ctx = SimpleNamespace(
        repo_dir=repo,
        drive_root=drive,
        task_id="task-row-custody",
        task_metadata={},
        current_task_type="",
        parent_task_id="",
        _current_review_tool_name="commit_reviewed",
        _current_review_attempt_number=1,
        _review_advisory=[],
    )

    _record_commit_attempt(
        ctx,
        "same",
        "reviewing",
        late_result_pending=True,
        review_retry_key="commit_review:row-custody",
    )

    attempts = load_state(drive).filter_attempts(
        repo_key=repo_key,
        tool_name="commit_reviewed",
        task_id="task-row-custody",
    )
    assert len(attempts) == 1
    assert attempts[0].attempt == 1
    assert attempts[0].status == "reviewing"
    assert attempts[0].late_result_pending is True
    assert attempts[0].triad_raw_results[0]["operation_id"] == "op-a"


def test_terminal_authority_failure_reconciles_reserved_roster(tmp_path, monkeypatch):
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        load_state,
        make_repo_key,
        save_state,
    )
    from ouroboros.tools import git as git_tools

    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "data"
    repo_key = make_repo_key(repo)
    state = AdvisoryReviewState()
    state.record_attempt(CommitAttemptRecord(
        ts="2026-08-24T00:00:00+00:00",
        commit_message="same",
        status="reviewing",
        repo_key=repo_key,
        tool_name="commit_reviewed",
        task_id="task-authority",
        attempt=1,
        paid=True,
        triad_raw_results=[{
            "slot_id": "slot-a", "operation_id": "op-a",
            "operation_state": "in_flight", "late_result_pending": True,
        }],
    ))
    save_state(drive, state)
    monkeypatch.setattr(
        "supervisor.evolution_lifecycle.check_evolution_authority",
        lambda **_kwargs: {"ok": False, "reason": "claim_gone"},
    )
    ctx = SimpleNamespace(
        repo_dir=repo,
        drive_root=drive,
        task_id="task-authority",
        task_metadata={
            "evolution_transaction": {
                "campaign_id": "campaign",
                "transaction_id": "transaction",
                "task_id": "task-authority",
            },
        },
        _current_review_attempt_number=1,
        _last_triad_models=[],
        _last_scope_model="",
        _last_triad_raw_results=[{
            "slot_id": "slot-a", "operation_id": "op-a",
            "operation_state": "settled", "late_result_pending": False,
        }],
        _last_scope_raw_result={},
    )

    _claim, message = git_tools._check_evolution_commit_stage(
        ctx, "same", 0.0, phase="pre_commit_authority",
    )

    assert "claim_gone" in message
    attempt = load_state(drive).attempts[-1]
    assert attempt.status == "blocked"
    assert attempt.triad_raw_results[0]["operation_state"] == "settled"
    assert attempt.late_result_pending is False
    assert load_state(drive).get_active_attempts(repo_key=repo_key) == []


def test_duplicate_current_slot_is_rejected_order_independently():
    original = {
        "slot_id": "slot-1", "model_id": "m1", "status": "error",
        "operation_id": "op-1", "operation_state": "in_flight",
        "late_result_pending": True,
    }
    duplicate_rows = [
        {
            "slot_id": "slot-1", "model_id": "m1", "status": "responded",
            "raw_text": "[]", "operation_id": "op-1", "operation_state": "settled",
        },
        {
            "slot_id": "slot-1", "model_id": "m1", "status": "responded",
            "raw_text": "different", "operation_id": "op-other",
            "operation_state": "settled",
        },
    ]
    outcomes = []
    for current in (duplicate_rows, list(reversed(duplicate_rows))):
        ctx = SimpleNamespace(
            _last_triad_raw_results=current, _last_scope_raw_result={},
        )
        prepare_frozen_review_reconciliation(
            ctx, SimpleNamespace(triad_raw_results=[original], scope_raw_result={}),
        )
        merge_frozen_review_reconciliation(ctx)
        outcomes.append(ctx._last_triad_raw_results)
        assert ctx._review_custody_lost is True

    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0]["operation_state"] == "custody_lost"
    assert outcomes[0][0]["late_result_pending"] is True


def test_duplicate_against_terminal_original_stays_lost_on_next_retry():
    original = {
        "slot_id": "slot-1", "model_id": "m1", "status": "responded",
        "raw_text": "[]", "operation_id": "op-1", "operation_state": "settled",
        "late_result_pending": False,
    }
    duplicates = [
        dict(original),
        {**original, "raw_text": "other", "operation_id": "op-other"},
    ]
    first = SimpleNamespace(
        _last_triad_raw_results=duplicates, _last_scope_raw_result={},
    )
    prepare_frozen_review_reconciliation(
        first, SimpleNamespace(triad_raw_results=[original], scope_raw_result={}),
    )
    merge_frozen_review_reconciliation(first)

    assert first._review_custody_lost is True
    assert first._last_triad_raw_results[0]["operation_state"] == "custody_lost"
    assert first._last_triad_raw_results[0]["late_result_pending"] is True

    same_operation_settled = [{
        **original, "operation_state": "settled", "late_result_pending": False,
    }]
    second = SimpleNamespace(
        _last_triad_raw_results=same_operation_settled, _last_scope_raw_result={},
    )
    prepare_frozen_review_reconciliation(
        second,
        SimpleNamespace(
            triad_raw_results=first._last_triad_raw_results,
            scope_raw_result=first._last_scope_raw_result,
        ),
    )
    merge_frozen_review_reconciliation(second)

    assert second._review_custody_lost is True
    assert second._last_triad_raw_results[0]["operation_state"] == "custody_lost"
    assert second._last_triad_raw_results[0]["late_result_pending"] is True

    third = SimpleNamespace(
        _last_triad_raw_results=same_operation_settled, _last_scope_raw_result={},
    )
    prepare_frozen_review_reconciliation(
        third,
        SimpleNamespace(
            triad_raw_results=second._last_triad_raw_results,
            scope_raw_result=second._last_scope_raw_result,
        ),
    )
    merge_frozen_review_reconciliation(third)

    assert third._review_custody_lost is True
    assert third._last_triad_raw_results[0]["operation_state"] == "custody_lost"
    assert third._last_triad_raw_results[0]["late_result_pending"] is True


def test_mixed_non_object_triad_row_becomes_durable_custody_loss():
    terminal = {
        "slot_id": "slot-1", "status": "responded", "raw_text": "[]",
        "operation_id": "op-1", "operation_state": "settled",
    }
    ctx = SimpleNamespace(_last_triad_raw_results=[], _last_scope_raw_result={})
    prepare_frozen_review_reconciliation(
        ctx,
        SimpleNamespace(triad_raw_results=[terminal, "malformed"], scope_raw_result={}),
    )
    merge_frozen_review_reconciliation(ctx)

    assert ctx._review_custody_lost is True
    assert ctx._last_triad_raw_results[0] == terminal
    assert ctx._last_triad_raw_results[1]["operation_state"] == "custody_lost"
    assert ctx._last_triad_raw_results[1]["late_result_pending"] is True


def test_malformed_scope_row_and_container_become_durable_custody_loss():
    for malformed_scope in ({"raw_results": ["malformed"]}, "malformed"):
        ctx = SimpleNamespace(_last_triad_raw_results=[], _last_scope_raw_result={})
        prepare_frozen_review_reconciliation(
            ctx,
            SimpleNamespace(
                triad_raw_results=[{
                    "slot_id": "slot-1", "status": "responded", "raw_text": "[]",
                    "operation_id": "op-1", "operation_state": "settled",
                }],
                scope_raw_result=malformed_scope,
            ),
        )
        merge_frozen_review_reconciliation(ctx)

        scope_rows = ctx._last_scope_raw_result["raw_results"]
        assert ctx._review_custody_lost is True
        assert scope_rows[0]["operation_state"] == "custody_lost"
        assert scope_rows[0]["late_result_pending"] is True


def test_malformed_durable_roster_containers_survive_real_state_loading(tmp_path):
    from ouroboros.review_state import load_state, make_repo_key
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt

    state_path = tmp_path / "state" / "advisory_review.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({
        "attempts": [
            {
                "ts": "2026-08-24T00:00:00Z",
                "commit_message": "test",
                "status": "reviewing",
                "task_id": "task-1",
                "repo_key": make_repo_key(tmp_path),
                "tool_name": "commit_reviewed",
                "attempt": 1,
                "paid": True,
                "review_retry_key": "commit_review:exact",
                "triad_raw_results": "malformed",
                "scope_raw_result": ["malformed"],
            },
            {
                "ts": "2026-08-24T00:01:00Z",
                "commit_message": "historical",
                "status": "succeeded",
                "task_id": "task-old",
                "repo_key": make_repo_key(tmp_path),
                "tool_name": "commit_reviewed",
                "attempt": 2,
                "paid": True,
                "triad_raw_results": "malformed",
                "scope_raw_result": ["malformed"],
            },
        ],
    }), encoding="utf-8")

    state = load_state(tmp_path)

    assert len(state.attempts) == 2
    attempt = next(row for row in state.attempts if row.status == "reviewing")
    historical = next(row for row in state.attempts if row.status == "succeeded")
    assert attempt.paid is True
    assert attempt.late_result_pending is False
    assert attempt.triad_raw_results[0]["operation_state"] == "custody_lost"
    assert attempt.scope_raw_result["raw_results"][0]["operation_state"] == "custody_lost"
    assert historical.late_result_pending is False
    assert historical.triad_raw_results[0]["operation_state"] == "custody_lost"
    assert historical not in state.get_active_attempts(repo_key=make_repo_key(tmp_path))

    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        task_id="task-1",
        _current_review_tool_name="commit_reviewed",
    )
    assert _check_overlapping_review_attempt(ctx) is None
    assert ctx._review_resume_pending is True
    assert ctx._pending_review_attempt.triad_raw_results[0]["operation_state"] == "custody_lost"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"attempts": {}},
        {"attempts": ["not-an-object"]},
        {"attempts": [{"status": []}]},
        {"attempts": [{"status": "reviewing", "task_id": 7}]},
        {"attempts": [{"status": "reviewing", "attempt": "1"}]},
        {"attempts": [{"status": "reviewing", "paid": "yes"}]},
        {"attempts": [{"status": "reviewing", "blocked": 1}]},
        {"attempts": [{"status": "reviewing", "finished_ts": []}]},
        {"attempts": [{"status": "reviewing", "review_owner_pid": "123"}]},
        {"attempts": [{"status": "reviewing", "review_owner_session_id": 7}]},
    ],
)
def test_locked_update_refuses_malformed_attempt_authority_without_rewrite(
    tmp_path, payload,
):
    from ouroboros.review_state import load_state, update_state

    path = tmp_path / "state" / "advisory_review.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    original = path.read_bytes()
    called = []

    # The ordinary diagnostic read remains fail-soft for the same malformed
    # bytes; only a locked authority mutation is strict.
    assert load_state(tmp_path) is not None
    with pytest.raises(ValueError):
        update_state(tmp_path, lambda state: called.append(state))

    assert called == []
    assert path.read_bytes() == original


def test_startup_settles_only_tokenless_process_local_rows(tmp_path, monkeypatch):
    from ouroboros.agent_startup_checks import _reconcile_review_attempts_on_startup
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        load_state,
        save_state,
    )

    tokenless = {
        "slot_id": "slot_api", "status": "error", "operation_id": "op-api",
        "operation_state": "in_flight", "late_result_pending": True,
    }
    delegated = {
        "slot_id": "slot_session", "status": "error", "operation_id": "op-session",
        "operation_state": "in_flight", "late_result_pending": True,
        "pending_invocation_id": "invocation-1",
    }
    scope_tokenless = {
        "slot_id": "scope_api", "status": "error", "operation_id": "op-scope",
        "operation_state": "in_flight", "late_result_pending": True,
    }
    save_state(tmp_path, AdvisoryReviewState(attempts=[CommitAttemptRecord(
        ts="2020-01-01T00:00:00+00:00",
        commit_message="pending",
        status="reviewing",
        attempt=1,
        paid=True,
        review_retry_key="commit_review:startup",
        review_owner_session_id="prior-generation",
        review_owner_pid=999_999_999,
        triad_raw_results=[tokenless, delegated],
        scope_raw_result={"raw_results": [scope_tokenless]},
    )]))

    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda _pid: False)
    outcome = _reconcile_review_attempts_on_startup(
        SimpleNamespace(drive_root=tmp_path),
    )

    assert len(outcome["reconciled"]) == 1
    assert outcome["expired"] == []
    attempt = load_state(tmp_path).attempts[0]
    assert attempt.status == "reviewing"
    assert attempt.phase == "late_wait"
    assert attempt.block_reason == "review_late_result_pending"
    assert attempt.late_result_pending is True
    assert attempt.triad_raw_results[0]["operation_state"] == "settled"
    assert attempt.triad_raw_results[0]["late_result_pending"] is False
    assert attempt.triad_raw_results[0]["failure_code"] == "process_local_review_worker_lost"
    assert attempt.triad_raw_results[1] == delegated
    assert attempt.scope_raw_result["raw_results"][0]["operation_state"] == "settled"


def test_confirmed_owner_death_terminalizes_custody_loss_and_is_idempotent(tmp_path):
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord

    row = {
        "slot_id": "slot_api", "operation_id": "op-api", "status": "error",
        "operation_state": "custody_lost", "late_result_pending": True,
    }
    state = AdvisoryReviewState(attempts=[CommitAttemptRecord(
        ts="2020-01-01T00:00:00+00:00",
        commit_message="pending",
        status="reviewing",
        attempt=1,
        paid=True,
        review_owner_pid=42,
        triad_raw_results=[row],
    )])

    assert state.reconcile_process_local_review_custody_after_owner_loss(
        confirmed_dead_owner_pids={42},
    )
    attempt = state.attempts[0]
    assert attempt.status == "failed"
    assert attempt.late_result_pending is False
    assert attempt.finished_ts
    assert attempt.triad_raw_results[0]["operation_state"] == "settled"
    assert attempt.triad_raw_results[0]["failure_code"] == "process_local_review_worker_lost"
    assert state.reconcile_process_local_review_custody_after_owner_loss(
        confirmed_dead_owner_pids={42},
    ) == []


def test_confirmed_owner_death_keeps_identical_verdict_streak_fail_closed(tmp_path):
    """A lost paid worker is infra noise, not a new terminal review result.

    The identical-diff refusal must survive owner-death reconciliation so a
    later retry cannot buy a duplicate paid wave for unchanged bytes.
    """
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        load_state,
        make_repo_key,
        save_state,
    )
    from ouroboros.tools.commit_gate import check_identical_verdict_refusal

    fingerprint = "fp-owner-loss"
    contract = "contract-owner-loss"
    repo_key = make_repo_key(tmp_path)
    save_state(tmp_path, AdvisoryReviewState(attempts=[CommitAttemptRecord(
        ts="2020-01-01T00:00:00+00:00",
        commit_message="owner lost",
        status="reviewing",
        attempt=1,
        paid=True,
        repo_key=repo_key,
        task_id="task-owner-loss",
        review_owner_pid=42,
        pre_review_fingerprint=fingerprint,
        review_contract_fingerprint=contract,
        triad_raw_results=[{
            "slot_id": "slot-api",
            "operation_id": "op-api",
            "operation_state": "custody_lost",
            "late_result_pending": True,
        }],
    )]))

    state = load_state(tmp_path)
    changed = state.reconcile_process_local_review_custody_after_owner_loss(
        confirmed_dead_owner_pids={42},
    )
    assert len(changed) == 1
    assert state.attempts[0].status == "failed"
    assert state.attempts[0].phase == "infra"
    save_state(tmp_path, state)

    refusal = check_identical_verdict_refusal(
        SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="task-owner-loss"),
        fingerprint,
        contract_fingerprint=contract,
    )
    assert refusal == ""


def test_confirmed_owner_death_unblocks_a_new_commit(tmp_path):
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord, make_repo_key

    repo_key = make_repo_key(tmp_path)
    state = AdvisoryReviewState(attempts=[CommitAttemptRecord(
        ts="2020-01-01T00:00:00+00:00",
        commit_message="pending",
        status="reviewing",
        attempt=1,
        paid=True,
        repo_key=repo_key,
        task_id="task-1",
        review_owner_pid=43,
        triad_raw_results=[{
            "slot_id": "slot_api",
            "operation_id": "op-api",
            "operation_state": "in_flight",
            "late_result_pending": True,
        }],
    )])

    state.reconcile_process_local_review_custody_after_owner_loss(
        confirmed_dead_owner_pids={43},
    )

    assert state.get_active_attempts(repo_key=repo_key) == []
    assert state.attempts[0].status == "failed"
    assert state.attempts[0].paid is True


def test_unowned_legacy_paid_attempt_remains_fail_closed():
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord

    state = AdvisoryReviewState(attempts=[CommitAttemptRecord(
        ts="2020-01-01T00:00:00+00:00",
        commit_message="legacy paid stamp",
        status="reviewing",
        attempt=1,
        paid=True,
        review_retry_key="commit_review:legacy-empty",
    )])

    changed = state.reconcile_process_local_review_custody_after_owner_loss(
        confirmed_dead_owner_pids={44},
    )

    assert changed == []
    assert state.attempts[0].status == "reviewing"
    assert state.attempts[0].paid is True


def test_same_generation_worker_boot_does_not_settle_live_peer(tmp_path):
    import os

    from ouroboros.agent_startup_checks import _reconcile_review_attempts_on_startup
    from ouroboros.process_custody import current_custody_session_id
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        load_state,
        save_state,
    )

    save_state(tmp_path, AdvisoryReviewState(attempts=[CommitAttemptRecord(
        ts="2020-01-01T00:00:00+00:00",
        commit_message="live sibling worker",
        status="reviewing",
        attempt=1,
        paid=True,
        late_result_pending=True,
        review_owner_session_id=current_custody_session_id(),
        review_owner_pid=os.getpid(),
        triad_raw_results=[{
            "slot_id": "slot-live",
            "operation_id": "op-live",
            "operation_state": "in_flight",
            "late_result_pending": True,
        }],
    )]))

    outcome = _reconcile_review_attempts_on_startup(
        SimpleNamespace(drive_root=tmp_path),
    )

    attempt = load_state(tmp_path).attempts[0]
    assert outcome["reconciled"] == []
    assert attempt.status == "reviewing"
    assert attempt.late_result_pending is True
    assert attempt.triad_raw_results[0]["operation_state"] == "in_flight"


def test_prior_generation_pid_must_be_proven_dead(tmp_path, monkeypatch):
    from ouroboros.agent_startup_checks import _reconcile_review_attempts_on_startup
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        load_state,
        save_state,
    )

    save_state(tmp_path, AdvisoryReviewState(attempts=[CommitAttemptRecord(
        ts="2020-01-01T00:00:00+00:00",
        commit_message="prior process still alive",
        status="reviewing",
        attempt=1,
        paid=True,
        late_result_pending=True,
        review_owner_session_id="prior-generation",
        review_owner_pid=707,
        triad_raw_results=[{
            "slot_id": "slot-prior",
            "operation_id": "op-prior",
            "operation_state": "in_flight",
            "late_result_pending": True,
        }],
    )]))
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda _pid: True)

    outcome = _reconcile_review_attempts_on_startup(
        SimpleNamespace(drive_root=tmp_path),
    )

    attempt = load_state(tmp_path).attempts[0]
    assert outcome["reconciled"] == []
    assert attempt.status == "reviewing"
    assert attempt.triad_raw_results[0]["operation_state"] == "in_flight"


def test_confirmed_death_targets_only_the_exact_owner_pid(tmp_path):
    from ouroboros.review_owner_custody import (
        reconcile_review_custody_after_confirmed_process_death,
    )
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        load_state,
        save_state,
    )

    def attempt(pid):
        return CommitAttemptRecord(
            ts="2020-01-01T00:00:00+00:00",
            commit_message=f"owner {pid}",
            status="reviewing",
            attempt=pid,
            paid=True,
            late_result_pending=True,
            review_owner_session_id="same-generation",
            review_owner_pid=pid,
            triad_raw_results=[{
                "slot_id": f"slot-{pid}",
                "operation_id": f"op-{pid}",
                "operation_state": "in_flight",
                "late_result_pending": True,
            }],
        )

    save_state(tmp_path, AdvisoryReviewState(attempts=[attempt(101), attempt(202)]))
    outcome = reconcile_review_custody_after_confirmed_process_death(
        tmp_path,
        101,
    )
    state = load_state(tmp_path)
    by_pid = {item.review_owner_pid: item for item in state.attempts}

    assert len(outcome["reconciled"]) == 1
    assert outcome["reconciled"][0].review_owner_pid == 101
    assert by_pid[101].status == "failed"
    assert by_pid[202].status == "reviewing"
    assert by_pid[202].triad_raw_results[0]["operation_state"] == "in_flight"


def test_startup_recovers_start_requested_token_by_reserved_operation(tmp_path, monkeypatch):
    from ouroboros import delegate_custody
    from ouroboros.agent_startup_checks import _reconcile_review_attempts_on_startup
    from ouroboros.review_state import (
        AdvisoryReviewState,
        CommitAttemptRecord,
        load_state,
        save_state,
    )

    save_state(tmp_path, AdvisoryReviewState(attempts=[CommitAttemptRecord(
        ts="2020-01-01T00:00:00+00:00",
        commit_message="pending",
        status="reviewing",
        attempt=1,
        paid=True,
        review_retry_key="commit_review:start-window",
        review_owner_session_id="prior-generation",
        review_owner_pid=999_999_999,
        triad_raw_results=[{
            "slot_id": "slot-a",
            "operation_id": "op-start-window",
            "operation_state": "in_flight",
            "late_result_pending": True,
        }],
    )]))
    assert delegate_custody.record_start_requested(
        tmp_path,
        run_id="",
        task_id="task-a",
        invocation_id="inv-start-window",
        operation_id="op-start-window",
        idempotency_key="logical",
        max_seconds=300,
        request={"prompt": "review"},
        project_id="project-a",
        project_owned=False,
        route="claude",
        surface="multi_model_review",
        slot_id="slot-a",
    )

    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda _pid: False)
    outcome = _reconcile_review_attempts_on_startup(
        SimpleNamespace(drive_root=tmp_path),
    )

    attempt = load_state(tmp_path).attempts[0]
    assert outcome["reconciled"] == [attempt]
    assert attempt.status == "reviewing"
    assert attempt.late_result_pending is True
    assert (
        attempt.triad_raw_results[0]["pending_invocation_id"]
        == "inv-start-window"
    )


def test_commit_reconciliation_does_not_leak_reconcile_only_to_next_surface():
    from ouroboros.tools.git import _reconcile_and_clear_review_roster

    ctx = SimpleNamespace(
        _review_reconcile_only=True,
        _review_reserved_roster=None,
        _review_paid_stamp=object(),
        _review_pending_invocation_checkpoint=object(),
        _review_reserved_operations={"commit": "op-1"},
    )

    _reconcile_and_clear_review_roster(ctx)

    assert ctx._review_reconcile_only is False
    assert ctx._review_paid_stamp is None
    assert ctx._review_pending_invocation_checkpoint is None
    assert ctx._review_reserved_operations == {}
