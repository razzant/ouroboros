"""Regression tests for the four Astra run-a882315dbcd7 findings closed in #1196
(commit 73d9c9a2): an orphaned durable Resume grant is revoked instead of
answering ``resume_already_granted`` forever; an unreadable custody chain is
``custody_read=failed``, never "no open runs"; a pending (START_REQUESTED,
unbound) invocation of THIS task is unknown custody; and a parallel tool batch
that leaves on ``UsageAccountingError`` waits for its already-started calls.

Fixtures come from the exact budget-pause suite (``tests/_budget_pause_exact_helpers``).
"""

from __future__ import annotations



from tests._budget_pause_exact_helpers import (  # noqa: F401 -- shared fixtures of the exact-pause suite
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


# --- Astra run-6fe5bf761449 follow-ups: one snapshot, and malformed custody lines ----
