"""A released commit still retains its exact pre-reserved reviewer invocation."""
from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ouroboros import review_state as state_store
from ouroboros.review_execution import ReviewRouteKind
from ouroboros.tools.commit_gate import _record_commit_attempt
from ouroboros.tools.git import _install_paid_dispatch_stamp
from ouroboros.tools.parallel_review import _reserve_parallel_review_roster


@pytest.fixture
def reserved(tmp_path):
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    ctx = SimpleNamespace(
        repo_dir=repo, drive_root=drive, task_id="late-commit-task", task_metadata={},
        current_task_type="", parent_task_id="", _review_advisory=[],
        _current_review_tool_name="commit_reviewed",
        _current_review_retry_key="commit_review:reserved-cycle",
        _current_review_contract_fingerprint="original-contract",
        _current_review_rebuttal_sha256="", _review_reconcile_only=False,
    )
    _install_paid_dispatch_stamp(ctx, "fixture commit", time.time(), {"fingerprint": "original-subject"})
    # ONE wave: a packet seat and a retrieving seat (both parts) ride the same
    # surface roster; there is no second surface to reserve.
    _reserve_parallel_review_roster(
        ctx, {"row_plan": {"models": ["fake/packet", "fake/retrieving"],
                           "routes": [ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION],
                           "efforts": ["high", "high"], "slot_ids": ["seat-a", "seat-b"],
                           "parts": [("change",), ("change", "coupling")]}},
    )
    attempt = load(ctx)
    assert attempt.status == "reviewing" and attempt.paid
    return ctx


def load(ctx, *, number=None):
    return state_store.load_state(ctx.drive_root).latest_attempt_for(
        repo_key=state_store.make_repo_key(ctx.repo_dir), tool_name="commit_reviewed",
        task_id=ctx.task_id, attempt=number or ctx._current_review_attempt_number,
    )


def args_for(ctx, seat=0):
    item = load(ctx)
    row = item.triad_raw_results[seat]
    return dict(repo_key=item.repo_key, tool_name=item.tool_name, task_id=item.task_id,
                attempt=item.attempt, review_retry_key=item.review_retry_key, surface="multi_model_review",
                slot_id=row["slot_id"], operation_id=row["operation_id"], invocation_id="fixture-invocation-" + row["slot_id"])


def immutable_evidence(item):
    value = asdict(item)
    value.pop("updated_ts")
    value.pop("late_result_pending")
    for row in value["triad_raw_results"]:
        row.pop("pending_invocation_id", None)
        row.pop("late_result_pending", None)
    return value


@pytest.mark.parametrize("status", ["reviewing", "reviewed", "succeeded", "failed", "blocked"])
@pytest.mark.parametrize("seat", [0, 1])
def test_exact_paid_checkpoint_survives_author_continuation(reserved, status, seat):
    ctx = reserved
    arguments = args_for(ctx, seat)
    _record_commit_attempt(ctx, "fixture commit", status, _strict=True)
    before = load(ctx)
    state_store.checkpoint_pending_review_invocation(ctx.drive_root, **arguments)
    after = load(ctx)
    assert after.status == status
    assert immutable_evidence(after) == immutable_evidence(before)
    assert after.paid and after.late_result_pending
    assert after.triad_raw_results[seat]["pending_invocation_id"] == arguments["invocation_id"]
    assert "pending_invocation_id" not in after.triad_raw_results[1 - seat]
    assert len(state_store.load_state(ctx.drive_root).attempts) == 1


@pytest.mark.parametrize("axis", ["repo_key", "tool_name", "task_id", "attempt", "review_retry_key", "surface", "slot_id", "operation_id"])
def test_wrong_identity_cannot_checkpoint_another_attempt(reserved, axis):
    ctx = reserved
    _record_commit_attempt(ctx, "fixture commit", "succeeded", _strict=True)
    arguments = args_for(ctx)
    arguments[axis] = 99 if axis == "attempt" else "other-" + arguments[axis]
    before = asdict(state_store.load_state(ctx.drive_root))
    with pytest.raises(ValueError):
        state_store.checkpoint_pending_review_invocation(ctx.drive_root, **arguments)
    assert asdict(state_store.load_state(ctx.drive_root)) == before


def test_replacement_cycle_does_not_receive_old_invocation(reserved):
    ctx = reserved
    arguments = args_for(ctx)
    def replace(state):
        current = state.latest_attempt_for(repo_key=arguments["repo_key"], task_id=ctx.task_id,
                                           tool_name="commit_reviewed", attempt=arguments["attempt"])
        current.review_retry_key = "commit_review:new-cycle"
        current.triad_raw_results[0]["operation_id"] = "new-operation"
        current.status = "succeeded"
    state_store.update_state(ctx.drive_root, replace)
    before = asdict(state_store.load_state(ctx.drive_root))
    with pytest.raises(ValueError, match="unavailable"):
        state_store.checkpoint_pending_review_invocation(ctx.drive_root, **arguments)
    assert asdict(state_store.load_state(ctx.drive_root)) == before


@pytest.mark.parametrize("defect", ["unpaid", "settled_row", "other_invocation", "duplicate_slot"])
def test_exact_reservation_requirements_remain(reserved, defect):
    ctx = reserved
    arguments = args_for(ctx)
    def change(state):
        current = state.latest_attempt_for(repo_key=arguments["repo_key"], task_id=ctx.task_id,
                                           tool_name="commit_reviewed", attempt=arguments["attempt"])
        current.status = "succeeded"
        if defect == "unpaid":
            current.paid = False
        elif defect == "settled_row":
            current.triad_raw_results[0]["operation_state"] = "settled"
        elif defect == "other_invocation":
            current.triad_raw_results[0]["pending_invocation_id"] = "different-invocation"
        else:
            current.triad_raw_results.append(copy.deepcopy(current.triad_raw_results[0]))
    state_store.update_state(ctx.drive_root, change)
    before = asdict(state_store.load_state(ctx.drive_root))
    with pytest.raises(ValueError):
        state_store.checkpoint_pending_review_invocation(ctx.drive_root, **arguments)
    assert asdict(state_store.load_state(ctx.drive_root)) == before


def test_delayed_checkpoint_closure_and_parallel_slot_keep_terminal_status(reserved):
    ctx = reserved
    checkpoint = ctx._review_pending_invocation_checkpoint
    arguments = [args_for(ctx, seat) for seat in (0, 1)]
    entered, release = threading.Barrier(3), threading.Event()
    def late(arguments):
        entered.wait(timeout=5)
        assert release.wait(timeout=5)
        checkpoint(**{key: arguments[key] for key in ("surface", "slot_id", "operation_id", "invocation_id")})
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(late, item) for item in arguments]
        try:
            entered.wait(timeout=5)
            _record_commit_attempt(ctx, "fixture commit", "succeeded", _strict=True)
            before = load(ctx)
        finally:
            release.set()
        for item in pending:
            item.result(timeout=5)
    after = load(ctx)
    assert after.status == "succeeded" and immutable_evidence(after) == immutable_evidence(before)
    assert after.triad_raw_results[0]["pending_invocation_id"] == arguments[0]["invocation_id"]
    assert after.triad_raw_results[1]["pending_invocation_id"] == arguments[1]["invocation_id"]
    state_store.checkpoint_pending_review_invocation(ctx.drive_root, **arguments[0])
    assert immutable_evidence(load(ctx)) == immutable_evidence(after)
    assert len(state_store.load_state(ctx.drive_root).attempts) == 1


def test_newer_other_attempt_is_untouched_by_exact_older_checkpoint(reserved):
    ctx = reserved
    arguments = args_for(ctx)
    _record_commit_attempt(ctx, "fixture commit", "succeeded", _strict=True)
    older = load(ctx)
    newer = copy.deepcopy(older)
    newer.attempt += 1
    newer.review_retry_key = "commit_review:newer"
    newer.triad_raw_results[0]["operation_id"] = "newer-operation"
    state_store.update_state(ctx.drive_root, lambda state: state.record_attempt(newer))
    before = asdict(load(ctx, number=newer.attempt))
    state_store.checkpoint_pending_review_invocation(ctx.drive_root, **arguments)
    assert load(ctx).status == "succeeded"
    assert asdict(load(ctx, number=newer.attempt)) == before
    assert load(ctx, number=arguments["attempt"]).triad_raw_results[0]["pending_invocation_id"] == arguments["invocation_id"]
