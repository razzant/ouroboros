"""#2710: a review attempt whose roster rows settled stops reading as a reviewer still working.

A start token left on a row whose own named operation settled is a lease nobody released (a collector projects
``late_settled`` only when no invocation is pending; the start checkpoint never binds a settled row), so it holds
nothing: the attempt compacts out of the newest-50 window, an unpaid one is evictable, and its round's checkout
goes. A reviewer genuinely in flight, a running additional adviser, a lost seat and an outcome merged without its
operation keep their custody. Every case asserts both directions.
"""
from __future__ import annotations

import pytest

from ouroboros import review_state as rs
from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord

TOKEN = "inv-left-behind"


def _row(state, *, token=TOKEN, operation="op-1", slot="s1", **extra):
    return {"slot_id": slot, "model_id": "m", "status": "ok" if state.endswith("settled") else "error",
            "operation_id": operation, "operation_state": state, "pending_invocation_id": token,
            "late_result_pending": state in {"in_flight", "custody_lost"}, "raw_text": "x" * 4000, **extra}


def _attempt(rows, *, attempt=1, paid=True, status="reviewed", task="t-old", **extra):
    return CommitAttemptRecord(ts=f"2026-10-0{1 + attempt % 9}T00:00:00+00:00", commit_message="review",
                               status=status, phase="review", task_id=task, tool_name="review_change", attempt=attempt,
                               paid=paid, review_retry_key=f"round-{attempt}", triad_raw_results=rows, **extra)


ADVISER = {"review_plan": {"additional": True}}


@pytest.mark.parametrize("rows, held", [
    ([_row("late_settled")], False),  # the live shape: 354 rows here, every run terminal
    ([_row("settled")], False),
    ([_row("in_flight")], True),  # a reviewer genuinely in flight
    ([_row("late_settled"), _row("in_flight", slot="a1", **ADVISER)], True),  # a running additional adviser
    ([_row("custody_lost")], True),  # a lost seat keeps the token it may still be rejoined by
    ([_row("late_settled", operation="")], True),  # an outcome merged without its operation proves nothing
], ids=["late_settled", "settled", "in_flight", "adviser_running", "custody_lost", "unidentified"])
def test_a_rows_own_settled_operation_outranks_its_leftover_start_token(rows, held):
    attempt = _attempt(rows)
    assert rs._attempt_has_active_review_custody(attempt) is held
    assert rs._review_roster_row_is_pending(rows[-1]) is held
    assert rs.attempt_counted_seats_closed(attempt) is (not held or rows[-1].get("review_plan") is not None)


def _history(target):
    """``target`` as the oldest attempt, then fifty newer paid rows: one more review pushes it out of the window."""
    newer = [_attempt([], attempt=number, task=f"t-{number}") for number in range(2, 52)]
    state = AdvisoryReviewState()
    state.attempts = [target, *newer]
    state.record_attempt(_attempt([], attempt=99, task="t-new"))
    return next((row for row in state.attempts if row.task_id == "t-old"), None)


@pytest.mark.parametrize("paid", [True, False])
def test_a_settled_row_with_a_leftover_token_compacts_and_an_unpaid_one_is_evicted(paid):
    """A paid row (the money ledger) is compacted but never evicted; an unpaid one goes first."""
    kept = _history(_attempt([_row("late_settled")], paid=paid))
    if paid:
        assert kept is not None and kept.raw_stripped and kept.triad_raw_results == [], kept
    else:
        assert kept is None, "an unpaid settled row is evictable"


@pytest.mark.parametrize("rows", [[_row("in_flight")], [_row("late_settled"), _row("in_flight", slot="a1", **ADVISER)]],
                         ids=["in_flight", "adviser_running"])
@pytest.mark.parametrize("paid", [True, False])
def test_a_reviewer_still_working_keeps_its_token_and_payload_out_of_the_window(rows, paid):
    kept = _history(_attempt([dict(row) for row in rows], paid=paid))
    assert kept is not None and not kept.raw_stripped, "never evicted or compacted while a seat runs"
    assert kept.triad_raw_results[-1]["pending_invocation_id"] == TOKEN
    assert len(kept.triad_raw_results[-1]["raw_text"]) == 4000


def test_a_late_snapshot_cannot_reopen_a_settled_row_or_raise_its_attempts_late_flag():
    """A-6: the merge reads a settled row's leftover token as closed, so a late pending snapshot of the SAME operation
    neither replaces its outcome nor keeps the attempt's late flag raised; a seat still running keeps both."""
    snapshot = _row("in_flight", token=TOKEN)
    settled = rs._merge_attempt(_attempt([_row("late_settled")]),
                                _attempt([dict(snapshot)], status="reviewing", late_result_pending=True))
    assert settled.triad_raw_results[0]["operation_state"] == "late_settled" and not settled.late_result_pending
    running = rs._merge_attempt(_attempt([_row("in_flight")]),
                                _attempt([dict(snapshot)], status="reviewing", late_result_pending=True))
    assert running.triad_raw_results[0]["operation_state"] == "in_flight" and running.late_result_pending


@pytest.mark.parametrize("state", ["late_settled", "in_flight"])
def test_a_dead_owners_settled_rows_close_its_attempt_and_a_delegated_seat_stays_recoverable(state):
    """A-6: confirmed owner death closes an attempt whose rows all settled (a leftover token held it ``late_wait``
    forever); a seat whose delegated run may still go keeps the attempt reviewing for its exact rejoin."""
    attempt = _attempt([_row(state)], status="reviewing", late_result_pending=True, review_owner_pid=4242)
    state_doc = AdvisoryReviewState()
    state_doc.attempts = [attempt]
    state_doc.reconcile_process_local_review_custody_after_owner_loss(confirmed_dead_owner_pids={4242})
    if state == "late_settled":
        assert (attempt.status, attempt.phase, attempt.late_result_pending) == ("failed", "infra", False)
        assert not rs._attempt_has_active_review_custody(attempt)
    else:
        assert (attempt.status, attempt.phase, attempt.late_result_pending) == ("reviewing", "review", True)
        assert attempt.triad_raw_results[0]["pending_invocation_id"] == TOKEN
