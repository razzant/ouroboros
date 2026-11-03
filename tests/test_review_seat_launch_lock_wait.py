"""A seat launch waits out a busy review-state lock; only a slot that cannot take its token refuses.

Live incident (17.10, task 88a45e6e, seat subagent_sa5g1l): the delegated seat's start token could
not be bound to its reserved commit-review row because another writer held the shared
``state/advisory_review.json`` lock past the 4.0 s budget (reported cause
``{"reason":"contention","elapsed_sec":4.004,"timeout_sec":4.0}``), so the seat was refused with
``review_custody_checkpoint_unwritable`` and no run started. Binding the token is bookkeeping before
a paid start: contention is now waited out (``SEAT_LAUNCH_LOCK_WAIT_SEC``), and a reserved row that
cannot take the token still refuses, under its own reason and code. The lock, the review state, the
delegated custody and the session executor are real; only the Claudexor gateway is scripted.
"""
from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import asdict

import pytest

from ouroboros import review_state as state_store
from ouroboros import review_state_custody
from ouroboros.review_execution import ReviewRouteUnavailable
from ouroboros.review_session_custody import checkpoint_pending_invocation
from tests._review_session_route_shared import (
    FakeLLM, _agent_request, _agent_slot,
    _owned_gateway_uses_each_test_transport as _transport_fixture,
    fake_route as _route_fixture,
)
from tests.test_commit_late_invocation_checkpoint import args_for, reserved  # noqa: F401 - pytest fixture

_owned_gateway_uses_each_test_transport = _transport_fixture
fake_route = _route_fixture

HOLD_SEC = 4.5  # past the 4.0 s budget that refused the live seat
OPERATION = "commit_review_scope_review_scope_slot_1_lockwait"


def _hold(drive_root, held, seconds):
    fd = state_store.acquire_review_state_lock(drive_root)
    assert fd is not None, "the test could not take the review-state lock"
    held.set()
    try:
        time.sleep(seconds)
    finally:
        state_store.release_review_state_lock(drive_root, fd)


@contextlib.contextmanager
def _busy_lock(drive_root, seconds):
    """Another writer holds the review-state lock for ``seconds``."""
    held = threading.Event()
    holder = threading.Thread(target=_hold, args=(drive_root, held, seconds), daemon=True)
    holder.start()
    assert held.wait(10), "the holder never took the lock"
    try:
        yield
    finally:
        holder.join(seconds + 10)


def _bound(ctx, arguments):
    """The commit gate's own checkpoint callable, bound to one reserved seat."""
    return lambda invocation_id: ctx._review_pending_invocation_checkpoint(
        surface=arguments["surface"], slot_id=arguments["slot_id"],
        operation_id=arguments["operation_id"], invocation_id=invocation_id)


@pytest.mark.serial  # the session fixture mutates its shared transport registry
def test_a_delegated_seat_waits_out_a_busy_review_state_lock_and_starts_once(tmp_path, fake_route):
    from ouroboros.review_substrate import ReviewCoordinator

    request, slot = _agent_request(), _agent_slot()
    row = {"slot_id": slot.slot_id, "operation_id": OPERATION, "operation_state": "in_flight",
           "late_result_pending": True}
    state_store.update_state(tmp_path, lambda state: state.record_attempt(state_store.CommitAttemptRecord(
        ts="2026-10-17T10:27:00+00:00", commit_message="c", status="reviewing", task_id=request.task_id,
        root_task_id=request.task_id, tool_name="review_change", attempt=1, repo_key="repo", paid=True,
        review_retry_key="key", scope_raw_result={"raw_results": [row]})))

    def checkpoint(invocation_id):
        state_store.checkpoint_pending_review_invocation(
            tmp_path, repo_key="repo", tool_name="review_change", task_id=request.task_id, attempt=1,
            review_retry_key="key", surface=request.surface, slot_id=slot.slot_id, operation_id=OPERATION,
            invocation_id=invocation_id)

    with _busy_lock(tmp_path, HOLD_SEC):
        started = time.monotonic()
        actor = ReviewCoordinator(llm=FakeLLM(), drive_root=tmp_path)._run_slot(
            request, slot, operation_id=OPERATION, pending_invocation_checkpoint=checkpoint)
        waited = time.monotonic() - started
    gateway = fake_route.instances[-1]
    assert actor.status == "ok", (actor.failure_code, actor.error, actor.reported_cause)
    assert len(gateway.start_requests) == 1, "the seat started once, after the lock was free"
    assert waited > 4.0, "the start waited for the busy writer past the old 4 s budget"
    [saved] = state_store.load_state(tmp_path).attempts[-1].scope_raw_result["raw_results"]
    assert saved["operation_state"] == "in_flight"
    assert saved["pending_invocation_id"] == gateway.start_keys[0], "the token that started is the one bound"


def test_a_lock_held_past_the_launch_wait_refuses_with_its_lock_facts(reserved, monkeypatch):  # noqa: F811 - imported fixture
    """The wait is a transport bound (shortened here): a writer still holding the lock when it ends
    leaves the seat unbound and refused as unwritable, carrying the lock's own facts."""
    ctx = reserved
    arguments = args_for(ctx, 1)
    monkeypatch.setattr(review_state_custody, "SEAT_LAUNCH_LOCK_WAIT_SEC", 0.3)
    state = {"pending_invocation_id": arguments["invocation_id"]}
    before = asdict(state_store.load_state(ctx.drive_root))
    with _busy_lock(ctx.drive_root, 2.0):
        with pytest.raises(ReviewRouteUnavailable) as raised:
            checkpoint_pending_invocation(checkpoint=_bound(ctx, arguments), invocation_id=arguments["invocation_id"],
                                          state=state, on_failure=lambda: None)
    assert raised.value.code == "review_custody_checkpoint_unwritable"
    assert isinstance(raised.value.__cause__, state_store.ReviewStateLockError)
    assert '"reason":"contention"' in raised.value.reported_cause
    assert '"timeout_sec":0.3' in raised.value.reported_cause
    assert state == {} and asdict(state_store.load_state(ctx.drive_root)) == before


@pytest.mark.parametrize("defect,reason", [
    ("settled_row", "operation is not in flight"),
    ("other_invocation", "invocation identity changed"),
    ("replaced_attempt", "attempt is unavailable"),
])
def test_a_reserved_slot_that_cannot_take_the_token_refuses_at_once_with_its_own_reason(reserved, defect, reason):  # noqa: F811
    """A genuinely invalid slot still refuses, immediately, under its own code and words instead of
    the storage code; nothing is bound and the review state is unchanged."""
    ctx = reserved
    arguments = args_for(ctx, 1)

    def change(state):
        current = state.latest_attempt_for(repo_key=arguments["repo_key"], task_id=ctx.task_id,
                                           tool_name="commit_reviewed", attempt=arguments["attempt"])
        row = current.triad_raw_results[1]
        if defect == "settled_row":
            row["operation_state"] = "settled"
        elif defect == "other_invocation":
            row["pending_invocation_id"] = "different-invocation"
        else:
            current.review_retry_key = "commit_review:new-cycle"
    state_store.update_state(ctx.drive_root, change)
    before = asdict(state_store.load_state(ctx.drive_root))
    state = {"pending_invocation_id": arguments["invocation_id"]}
    started = time.monotonic()
    with pytest.raises(ReviewRouteUnavailable) as raised:
        checkpoint_pending_invocation(checkpoint=_bound(ctx, arguments), invocation_id=arguments["invocation_id"],
                                      state=state, on_failure=lambda: None)
    assert time.monotonic() - started < 2.0, "an invalid slot refuses on its own reason, not after a lock wait"
    assert raised.value.code == "review_custody_slot_refused"
    assert reason in str(raised.value) and reason in raised.value.reported_cause
    assert isinstance(raised.value.__cause__, state_store.ReservedReviewSlotRefused)
    assert state == {} and asdict(state_store.load_state(ctx.drive_root)) == before
