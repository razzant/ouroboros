"""#1547 link 1: a rejoin inherits what remains of the ORIGINAL window.

A paid delegated review whose worker is gone (process restart, a vanished
thread) is rejoined by the next collect. That rejoin must not shrink the run's
remaining time to the settlement margin: the window is computed from SAVED
facts — the durable STARTED / START_REQUESTED custody rows of the run (start
instant + ``max_seconds``), else the frozen row's own ``awaiting_since`` plus
the slot's logical window — as ``max(remaining, NESTED_SETTLEMENT_MARGIN_SEC)``.
Saved facts make the formula deterministic: a repeated collect never extends
the window, an expired run gets exactly the margin, never a fresh full window.

The "first process" here is a real custody run whose worker leaves ``_ACTIVE``
on a slot-budget failure; the "second process" is a fresh usage context that
knows the row only through the plan-review freeze of the persisted row
(``_plan_row_from_actor`` → ``_freeze_roster_rows``), exactly as
``tools/plan_review.py`` rebuilds it after a restart.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

TASK_ID = "task-1547-rejoin"
RETRY_KEY = "plan_review:1547"
INVOCATION = "inv-1547"
RUN_ID = "run-1547"
MAX_SECONDS = 600


def _past_iso(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def _slot(timeout_sec=None):
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    return ReviewSlot(slot_id="seat-1", model="cursor/test",
                      route=ReviewRouteKind.AGENT_SESSION, timeout_sec=timeout_sec)


def _request(**overrides):
    from ouroboros.review_substrate import ReviewRequest

    values = dict(surface="plan_review", goal="review the plan", task_id=TASK_ID,
                  retry_key=RETRY_KEY)
    values.update(overrides)
    return ReviewRequest(**values)


def _error_actor(slot, error, operation_id="", operation_state="settled"):
    from ouroboros.review_substrate import ReviewActorRecord

    return ReviewActorRecord(
        slot_id=slot.slot_id, model=slot.model, status="error", error=error,
        operation_id=operation_id, operation_state=operation_state,
        late_result_pending=operation_state == "in_flight",
    )


def _wait_worker_gone(key: str) -> None:
    import ouroboros.review_custody as custody

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with custody._ACTIVE_LOCK:
            if key not in custody._ACTIVE:
                return
        time.sleep(0.01)
    raise AssertionError("the first process's review worker never left _ACTIVE")


def _first_process(tmp_path, monkeypatch, *, sent_at_iso: str) -> dict:
    """Dispatch once, lose the worker on a slot-budget failure, return the persisted row.

    The worker records the live delegated tokens into the shared retry cell the
    way ``run_delegated_review_session`` does, so the persisted plan row carries
    ``pending_invocation_id`` / ``delegated_run_id`` beside ``awaiting_since``.
    """
    import ouroboros.review_custody as custody
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor
    from ouroboros.usage_accounting import UsageScope

    release = threading.Event()
    published = threading.Event()
    slot = _slot(timeout_sec=0.05)
    request = _request()
    key = custody._attempt_key(request, slot)

    def run_slot(_slot, _operation_id, retry_state, _deadline, _checkpoint):
        retry_state["pending_invocation_id"] = INVOCATION
        retry_state["delegated_run_id"] = RUN_ID
        published.set()
        assert release.wait(10)
        raise TimeoutError(f"delegated review session {RUN_ID} exceeded the slot budget")

    # This fixture represents an ALREADY dispatched remote invocation. A busy
    # worker can spend the entire 50 ms window in its started-event publication,
    # before run_slot has submitted anything. Let that publication happen before
    # collecting the timeout actor; keep the real timeout and worker settlement.
    # No token is manufactured by the collector, and the real actor must copy it.
    original_timeout_actor = custody._late_or_timeout_actor

    def dispatched_timeout_actor(*args, **kwargs):
        assert published.wait(10), "the delegated worker never published its invocation"
        return original_timeout_actor(*args, **kwargs)

    ctx = SimpleNamespace(drive_root=tmp_path, task_id=TASK_ID)
    try:
        with monkeypatch.context() as process:
            process.setattr(custody, "utc_now_iso", lambda: sent_at_iso)
            process.setattr(custody, "_late_or_timeout_actor", dispatched_timeout_actor)
            [actor] = custody.run_custodied_review_slots(
                request=request, slots=[slot], usage_ctx=ctx, task_id=TASK_ID, usage_meta={},
                review_usage_scope=UsageScope(drive_root=tmp_path, task_id=TASK_ID),
                run_slot=run_slot, error_actor=_error_actor,
            )
    finally:
        release.set()
        _wait_worker_gone(key)
    assert actor.operation_state == "in_flight"
    assert actor.usage["pending_invocation_id"] == INVOCATION
    row = _plan_row_from_actor(asdict(actor), slot)
    assert row["pending_invocation_id"] == INVOCATION
    assert row["operation_id"] == actor.operation_id
    assert row["awaiting_since"] == sent_at_iso
    return row


def _write_durable_custody(tmp_path, monkeypatch, *, started_iso: str, operation_id: str,
                           max_seconds_on_started: bool = True) -> None:
    """The rows the delegated route writes before polling, stamped at ``started_iso``."""
    from ouroboros import delegate_custody as custody

    monkeypatch.setattr(custody, "utc_now_iso", lambda: started_iso)
    assert custody.emit(tmp_path, custody.START_REQUESTED, {
        "run_id": "", "task_id": TASK_ID, "invocation_id": INVOCATION,
        "operation_id": operation_id, "surface": "plan_review", "slot_id": "seat-1",
        "max_seconds": MAX_SECONDS, "idempotency_key": "key-1547",
    })
    shape = {"effort": "high", "access": "readonly", "mode": "ask", "isolation": "none",
             "delegated": True, "root": str(tmp_path), "surface": "plan_review",
             "slot_id": "seat-1"}
    if max_seconds_on_started:
        shape["max_seconds"] = MAX_SECONDS
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id=RUN_ID, task_id=TASK_ID, invocation_id=INVOCATION,
        route_id="claude", model="cursor/test", ledger_root=str(tmp_path),
    ), shape=shape)
    monkeypatch.undo()
    started_ts, max_seconds = custody.run_timing(tmp_path, RUN_ID)
    assert started_ts == started_iso
    assert max_seconds == (MAX_SECONDS if max_seconds_on_started else 0)


def _second_process(tmp_path, row: dict, *, deadline_at: str, timeout_sec=None) -> tuple:
    """A fresh context rejoins the pending seat exactly as plan review's resume does."""
    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewActorRecord
    from ouroboros.usage_accounting import UsageScope

    ctx = SimpleNamespace(drive_root=tmp_path, task_id=TASK_ID)
    ctx._review_frozen_rows = {"plan_review": custody._freeze_roster_rows(ctx, "plan_review", [row])}
    request = _request(reconcile_only=True, deadline_at=deadline_at)
    slot = _slot(timeout_sec=timeout_sec)
    calls = []

    def rejoin(_slot, operation_id, retry_state, deadline, _checkpoint):
        calls.append((operation_id, dict(retry_state), deadline - time.monotonic()))
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="ok", raw_text="[]")

    [actor] = custody.run_custodied_review_slots(
        request=request, slots=[slot], usage_ctx=ctx, task_id=TASK_ID,
        usage_meta={"deadline_at": deadline_at} if deadline_at else {},
        review_usage_scope=UsageScope(drive_root=tmp_path, task_id=TASK_ID),
        run_slot=rejoin, error_actor=_error_actor,
    )
    assert len(calls) == 1, calls
    operation_id, retry_state, remaining = calls[0]
    assert operation_id == row["operation_id"]
    assert retry_state == {"pending_invocation_id": INVOCATION}
    return actor, remaining


SPENT_OWNER_DEADLINE = "2000-01-01T00:00:00Z"


def test_rejoin_after_worker_startup_outlasts_the_collection_window(tmp_path, monkeypatch):
    """A busy worker still supplies real invocation custody before this fixture rejoins it."""
    import ouroboros.review_custody as custody

    expired = threading.Event()
    original_clock = custody.monotonic_now
    original_emit = custody._emit_operation
    deadline = None

    def observe_clock(slot_id=None):
        nonlocal deadline
        now = original_clock(slot_id)
        if deadline is None:
            deadline = now + 0.05
        elif now >= deadline:
            expired.set()
        return now

    def delayed_start(*args, **kwargs):
        if kwargs.get("phase") == "started":
            assert expired.wait(10), "the collector never reached its real timeout"
        return original_emit(*args, **kwargs)

    with monkeypatch.context() as startup:
        startup.setattr(custody, "monotonic_now", observe_clock)
        startup.setattr(custody, "_emit_operation", delayed_start)
        row = _first_process(tmp_path, monkeypatch, sent_at_iso=_past_iso(200))
    assert expired.is_set()
    actor, remaining = _second_process(tmp_path, row, deadline_at="", timeout_sec=MAX_SECONDS)
    assert actor.status == "ok"
    assert remaining == pytest.approx(MAX_SECONDS - 200, abs=5)


def test_restart_rejoin_inherits_the_remaining_original_window(tmp_path, monkeypatch):
    """STARTED ``ts = now − 200 s``, ``max_seconds = 600`` → the rejoin worker gets
    ≈400 s, not the 30 s margin and not a new full window — even though the
    owner's own deadline is long spent."""
    from ouroboros.config import NESTED_SETTLEMENT_MARGIN_SEC

    row = _first_process(tmp_path, monkeypatch, sent_at_iso=_past_iso(200))
    _write_durable_custody(tmp_path, monkeypatch, started_iso=_past_iso(200),
                           operation_id=row["operation_id"])
    actor, remaining = _second_process(tmp_path, row, deadline_at=SPENT_OWNER_DEADLINE)
    assert remaining == pytest.approx(MAX_SECONDS - 200, abs=5)
    assert remaining > float(NESTED_SETTLEMENT_MARGIN_SEC)
    assert actor.status == "ok"
    assert actor.operation_id == row["operation_id"]


def test_rejoin_from_legacy_started_row_reads_max_seconds_from_the_start_request(tmp_path, monkeypatch):
    """A STARTED row that predates ``max_seconds`` still has the cap on its own
    START_REQUESTED row: the same ≈400 s, no invented number."""
    row = _first_process(tmp_path, monkeypatch, sent_at_iso=_past_iso(200))
    _write_durable_custody(tmp_path, monkeypatch, started_iso=_past_iso(200),
                           operation_id=row["operation_id"], max_seconds_on_started=False)
    _actor, remaining = _second_process(tmp_path, row, deadline_at=SPENT_OWNER_DEADLINE)
    assert remaining == pytest.approx(MAX_SECONDS - 200, abs=5)


def test_a_later_collect_does_not_extend_the_window(tmp_path, monkeypatch):
    """Sixty seconds later the same saved facts give ≈340 s: the window ends at
    the same wall-clock instant whichever collect reads it."""
    row = _first_process(tmp_path, monkeypatch, sent_at_iso=_past_iso(260))
    _write_durable_custody(tmp_path, monkeypatch, started_iso=_past_iso(260),
                           operation_id=row["operation_id"])
    _actor, remaining = _second_process(tmp_path, row, deadline_at=SPENT_OWNER_DEADLINE)
    assert remaining == pytest.approx(MAX_SECONDS - 260, abs=5)


@pytest.mark.parametrize("seconds_ago", [590, 900])
def test_an_almost_or_fully_expired_run_gets_exactly_the_margin(tmp_path, monkeypatch, seconds_ago):
    """``ts = now − 590`` leaves 10 s → the margin; an expired run → the margin;
    never a fresh full window."""
    from ouroboros.config import NESTED_SETTLEMENT_MARGIN_SEC

    row = _first_process(tmp_path, monkeypatch, sent_at_iso=_past_iso(seconds_ago))
    _write_durable_custody(tmp_path, monkeypatch, started_iso=_past_iso(seconds_ago),
                           operation_id=row["operation_id"])
    _actor, remaining = _second_process(tmp_path, row, deadline_at=SPENT_OWNER_DEADLINE)
    assert remaining == pytest.approx(float(NESTED_SETTLEMENT_MARGIN_SEC), abs=1.0)


def test_legacy_rejoin_without_durable_rows_uses_awaiting_since_plus_the_logical_window(tmp_path, monkeypatch):
    """No STARTED / START_REQUESTED row at all (legacy custody): the frozen row's
    ``awaiting_since`` (``now − 200 s``) plus this slot's logical window (600 s)
    → ≈400 s."""
    row = _first_process(tmp_path, monkeypatch, sent_at_iso=_past_iso(200))
    _actor, remaining = _second_process(tmp_path, row, deadline_at="", timeout_sec=MAX_SECONDS)
    assert remaining == pytest.approx(MAX_SECONDS - 200, abs=5)


def test_rejoin_with_no_saved_source_gets_the_margin(tmp_path, monkeypatch):
    """Neither durable rows nor a usable window behind ``awaiting_since`` (the
    owner deadline is spent) → exactly the settlement margin."""
    from ouroboros.config import NESTED_SETTLEMENT_MARGIN_SEC

    row = _first_process(tmp_path, monkeypatch, sent_at_iso=_past_iso(200))
    _actor, remaining = _second_process(tmp_path, row, deadline_at=SPENT_OWNER_DEADLINE)
    assert remaining == pytest.approx(float(NESTED_SETTLEMENT_MARGIN_SEC), abs=1.0)
