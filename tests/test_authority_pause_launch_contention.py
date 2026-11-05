"""Positive pre-seed launch-lock contention preserves the same pause attempt."""
from contextlib import contextmanager

import pytest

from ouroboros import budget_pause as bp, owner_pause
from ouroboros.model_wait import ModelWaitInterrupted
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_store import document, publish


@pytest.mark.parametrize(("stop", "rail"), [
    pytest.param(False, bp.RAIL_BILLING_AUTHORITY, id='False-billing_authority_unreadable'),
    pytest.param(True, bp.RAIL_CONTROL_AUTHORITY, id='True-control_authority_unreadable'),
])
def test_launch_lock_release_keeps_one_seed_and_capture(tmp_path, monkeypatch, rail, stop):
    publish(tmp_path, document("member", status="running"))
    ctx, limit = _loop_ctx(tmp_path, "member")
    real_lock, real_save, real_prepare = owner_pause.launch_lock, bp.set_budget_pause, bp._exact_continuation_row
    held = real_lock(tmp_path, "member")
    held.__enter__()
    released, attempts, seeds, captures = [], [], [], []
    control = []

    @contextmanager
    def acquire(root, task_id):
        attempts.append(task_id)
        with real_lock(root, task_id, wait=False):
            yield

    def save(*args, **kwargs):
        if args[2].get("source_ref") is None:
            seeds.append(args[2])
        return real_save(*args, **kwargs)

    def prepare(*args, **kwargs):
        result = real_prepare(*args, **kwargs)
        captures.append(result)
        return result

    def release(_ctx, row):
        if row.get("state") == bp.STATE_PAUSING and not released:
            assert bp.dispatch_fenced("member") and not seeds and not captures
            held.__exit__(None, None, None)
            released.append(True)
            if stop:
                control.append("cancelled")

    monkeypatch.setattr(owner_pause, "launch_lock", acquire)
    monkeypatch.setattr(bp, "set_budget_pause", save)
    monkeypatch.setattr(bp, "_exact_continuation_row", prepare)
    monkeypatch.setattr(bp, "_publish_hold", release)
    monkeypatch.setattr(bp, "_hold_control_reason", lambda _ctx: control[0] if control else "")
    monkeypatch.setattr(bp, "_HOLD_POLL_SEC", 0.001)
    try:
        with pytest.raises(ModelWaitInterrupted if stop else bp.BudgetPauseRequested) as outcome:
            bp.request_pause(limit, rail=rail, scope="task", reason_text="unreadable authority")
        assert released == [True]
        if stop:
            assert outcome.value.control_reason == "cancelled"
            assert len(attempts) == 1 and not seeds and not captures
            assert bp.dispatch_fenced("member")
        else:
            assert len(attempts) == 2 and len(seeds) == len(captures) == 1
            row = outcome.value.pause
            assert row["pause_id"] == seeds[0]["pause_id"] == captures[0]["pause_id"]
            assert row["source_ref"] == captures[0]["source_ref"]
            assert bp.budget_pause_row(tmp_path, "member") == row
    finally:
        if not released:
            held.__exit__(None, None, None)
        bp.end_dispatch_fence("member")


@pytest.mark.parametrize("reason", ["permission", "unknown"])
def test_noncontention_launch_failure_is_not_retried(tmp_path, monkeypatch, reason):
    publish(tmp_path, document("member", status="running"))
    _ctx, limit = _loop_ctx(tmp_path, "member")
    calls = []

    @contextmanager
    def unavailable(*args, **kwargs):
        calls.append(True)
        raise owner_pause.LaunchBusy(reason)
        yield  # contextmanager protocol; the protected body is never entered

    monkeypatch.setattr(owner_pause, "launch_lock", unavailable)
    assert bp.request_pause(limit, rail=bp.RAIL_CONTROL_AUTHORITY, scope="task", reason_text="unreadable") is None
    assert calls == [True] and not bp.dispatch_fenced("member")
    assert limit.accumulated_usage["exact_pause_unavailable"] == bp.HOLD_PAUSE_RECORD_UNWRITABLE
