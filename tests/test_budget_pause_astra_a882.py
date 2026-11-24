"""Regression tests for the four Astra run-a882315dbcd7 findings closed in #1196
(commit 73d9c9a2): an orphaned durable Resume grant is revoked instead of
answering ``resume_already_granted`` forever; an unreadable custody chain is
``custody_read=failed``, never "no open runs"; a pending (START_REQUESTED,
unbound) invocation of THIS task is unknown custody; and a parallel tool batch
that leaves on ``UsageAccountingError`` waits for its already-started calls.

Fixtures come from the exact budget-pause suite (``tests/_budget_pause_exact_helpers``).
"""

from __future__ import annotations

import pytest

from tests._budget_pause_exact_helpers import (  # noqa: F401 -- shared fixtures of the exact-pause suite
    _charge_elsewhere,
    _daily_wake_pause,
    _install_queue,
    _loop_ctx,
    _parked,
)


def _recording_writer(monkeypatch, budget_pause):
    """Wrap the real durable writer so a test can see every row it was asked to write."""
    real_writer = budget_pause.set_budget_pause
    written = []

    def _record(root, task_id, row, expected_pause_id=None, **kwargs):
        written.append((task_id, dict(row), dict(kwargs)))
        return real_writer(root, task_id, row, expected_pause_id, **kwargs)

    monkeypatch.setattr(budget_pause, "set_budget_pause", _record)
    return written


def _crash_restored_carrier(task, handoff):
    """The queue row a restore sees after a crash BETWEEN the durable grant and the
    snapshot: only the prior ``_budget_pause`` marker, no handoff, no hold."""
    stale = {key: value for key, value in task.items()
             if key not in {"_budget_pause_resume", "budget_resumed_at", "_budget_pause_hold"}}
    stale["_budget_pause"] = dict(handoff["pause"])
    return stale


# --- #1: orphaned grant without a queue carrier ------------------------------------


def test_a_grant_still_carried_by_a_queue_row_is_not_orphaned(tmp_path, monkeypatch):
    """Negative control for #1: while a PENDING or RUNNING row carries the matching
    ``_budget_pause_resume``, a second Resume is still ``resume_already_granted``
    and nothing is revoked or re-minted (single use holds)."""
    from ouroboros import budget_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="a882-carried")
    first = queue.resume_budget_paused_task("a882-carried")
    assert first["ok"] is True
    first_grant = first["grant_id"]
    handoff = task["_budget_pause_resume"]
    written = _recording_writer(monkeypatch, budget_pause)

    # A stale ``_budget_pause`` copy is located first, but the real carrier is still PENDING.
    stale = _crash_restored_carrier(task, handoff)
    workers.PENDING.insert(0, stale)
    refused = queue.resume_budget_paused_task("a882-carried")
    assert refused["error"] == "resume_already_granted" and refused["grant_id"] == first_grant

    # The carrier dispatched to a worker (RUNNING, grant not yet consumed): still carried.
    workers.PENDING.remove(task)
    workers.RUNNING["a882-carried"] = {"task": task, "wid": 0}
    refused_running = queue.resume_budget_paused_task("a882-carried")
    assert refused_running["error"] == "resume_already_granted" and refused_running["grant_id"] == first_grant

    assert written == []  # no revocation, no new grant was written
    row = budget_pause.budget_pause_row(tmp_path, "a882-carried")
    assert row["state"] == budget_pause.STATE_RESUME_GRANTED
    assert row["grant"]["grant_id"] == first_grant and not row["grant"].get("revoked_at")
    assert row["resume_generation"] == 1
    assert task["_budget_pause_resume"]["grant_id"] == first_grant and "_budget_pause" in stale


@pytest.mark.parametrize("lost", ["crash", "withdrawn", "withdrawal_unwritten"])
def test_a_wakes_lost_grant_never_costs_it_its_renewal_the_next_resume_reads_the_same_pause_fresh(
        tmp_path, monkeypatch, lost):
    """The grant the queue lost (a crash before its snapshot, a confirmed withdrawal, a withdrawal the restart could not
    write) leaves the wake's pause as it was. The next owner Resume withdraws what is left of it and renews from the room
    it reads NOW for that very pause: never a renewal-less grant that pauses again at the same ceiling, never the first
    grant's number."""
    from ouroboros import budget_pause
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import continuation_refusal
    from supervisor import budget_resume
    from tests.test_continue_blocker_diagnostics import _http

    queue, task, _metadata = _daily_wake_pause(tmp_path, monkeypatch)
    pause_id = budget_pause.budget_pause_row(tmp_path, "own-wake")["pause_id"]
    status, first = _http(api_task_resume, "own-wake")
    assert status == 200 and first["consciousness_resume"]["additional_usd"] == 7.0
    if lost == "crash":
        queue.PENDING[:] = [_crash_restored_carrier(task, task["_budget_pause_resume"])]
        assert budget_pause.budget_pause_row(tmp_path, "own-wake")["state"] == budget_pause.STATE_RESUME_GRANTED
    else:
        real_writer = budget_pause.set_budget_pause
        if lost == "withdrawal_unwritten":
            def unwritable(*_args, **_kwargs):
                raise OSError("task store busy")

            monkeypatch.setattr(budget_pause, "set_budget_pause", unwritable)
        assert budget_resume.revoke_exact_budget_resume(task, "restart_before_dispatch") is (lost == "withdrawn")
        monkeypatch.setattr(budget_pause, "set_budget_pause", real_writer)
        assert bool(continuation_refusal(task)) is (lost == "withdrawal_unwritten"), "the unwritten one keeps its typed hold"
    _charge_elsewhere(tmp_path, 2.0)  # the room is observed again, not carried over: $7 narrows to $5

    status, second = _http(api_task_resume, "own-wake")
    assert status == 200 and second["grant_id"] != first["grant_id"]
    assert second["consciousness_resume"]["additional_usd"] == 5.0
    live = queue.PENDING[0]  # the row the grant now stands on (the restored copy after a crash)
    assert not continuation_refusal(live) and live["_budget_pause_resume"]["grant_id"] == second["grant_id"]
    row = budget_pause.budget_pause_row(tmp_path, "own-wake")
    assert row["pause_id"] == pause_id and row["grant"]["grant_id"] == second["grant_id"]


# --- Astra run-6fe5bf761449 follow-ups: one snapshot, and malformed custody lines ----
