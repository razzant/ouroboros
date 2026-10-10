"""#1547 incident form (05.10) on plan review, end to end through the engine.

A panel of three; one seat is a delegated session whose run outlives the first
process. After the restart the next $0 collection rejoins that run for what
remains of ITS window (≈9 minutes here, from the durable STARTED facts), so:

* GREEN is never declared while the seat's paid answer is still owed;
* the collection itself never waits for it (drain window 0);
* the late blocking finding is not lost — it reaches the wave as a typed
  finding and the aggregate holds the plan open (``REVIEW_REQUIRED``).

The engine, substrate, custody and the durable delegated-run rows are real;
only the route executor is a scripted seat (the same seam
``tests/test_plan_review_event_route.py`` uses), which writes the write-ahead
rows the real session route writes before it POSTs.
"""

from __future__ import annotations

import json
import threading
import time

from tests.test_plan_review_engine import CLEAN, _call, _control, _finding, _slots, _state
from tests.test_plan_review_engine import harness as _engine_harness
from tests.test_plan_review_event_route import _mailbox_entries, _wait_until

harness = _engine_harness  # re-bound: a directly imported fixture name is an F811 under the ruff gate

INVOCATION, RUN_ID, MAX_SECONDS = "inv-1547-s3", "run-1547-s3", 540  # the run's own cap: 9 minutes


class _ScriptedSeats:
    """Route executors for the panel: s1/s2 answer clean at once; s3 is the
    delegated seat whose three executions tell the incident story."""

    def __init__(self):
        self.s3_calls = 0
        self.first_release, self.rejoin_release = threading.Event(), threading.Event()
        self.rejoin_remaining = None
        self.rejoin_cell = None

    def __call__(self, assignment, **_kw):
        return _Seat(self, assignment)


class _Seat:
    def __init__(self, seats, assignment):
        self.seats, self.assignment, self.cell = seats, assignment, {}
        self.slot_id = str(assignment.slot.slot_id)

    def restore_custody(self, cell):
        self.cell = cell

    def set_pending_invocation_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint

    def prompt_payload(self):
        return {"messages": []}

    def prompt_chars(self):
        return 0

    def failure_custody(self):  # the real session executor's shape
        run_id = str(self.cell.get("delegated_run_id") or "")
        return {"delegated_run_started": bool(run_id), "delegated_run_id": run_id,
                "pending_invocation_id": str(self.cell.get("pending_invocation_id") or "")}

    def _start_run_durably(self):
        """What the real route does before and after its POST: the START_REQUESTED
        row (invocation ↔ operation, ``max_seconds``) and the STARTED row of the run."""
        from ouroboros import delegate_custody as custody

        root = self.assignment.custody_root
        assert custody.record_start_requested(
            root, run_id="", task_id="task-1", root_task_id="task-1", idempotency_key="key-s3",
            invocation_id=INVOCATION, operation_id=self.assignment.call_id, max_seconds=MAX_SECONDS,
            surface="plan_review", slot_id="s3", route="cursor")
        assert custody.record_started(root, custody.RunCustody(
            run_id=RUN_ID, task_id="task-1", invocation_id=INVOCATION, route_id="cursor",
            model="m/c", ledger_root=str(root),
        ), shape={"effort": "high", "access": "readonly", "mode": "ask", "isolation": "none", "delegated": True,
                  "root": str(root), "surface": "plan_review", "slot_id": "s3", "max_seconds": MAX_SECONDS})
        self.cell["pending_invocation_id"], self.cell["delegated_run_id"] = INVOCATION, RUN_ID
        self.checkpoint(INVOCATION)

    def execute(self):
        from ouroboros.review_execution import ReviewAttemptResult

        usage = {"prompt_tokens": 10, "completion_tokens": 5, "physical_attempt_state": "settled"}
        if self.slot_id != "s3":
            return ReviewAttemptResult(message={"content": CLEAN}, usage=usage, raw_text=CLEAN)
        seats = self.seats
        seats.s3_calls += 1
        if seats.s3_calls == 1:
            # First process: the run starts, then the slot budget passes with it live.
            self._start_run_durably()
            assert seats.first_release.wait(20)
            raise TimeoutError(f"delegated review session {RUN_ID} exceeded the slot budget")
        if seats.s3_calls == 2:
            # Still the first process: the identical envelope's rejoin also runs out
            # with the run live — the in_flight row WITH its token is what persists.
            raise TimeoutError(f"delegated review session {RUN_ID} exceeded the slot budget")
        # Second process: the rejoin after the restart. Record the window custody
        # granted, then answer late with a blocking finding.
        seats.rejoin_remaining = self._logical_deadline_monotonic - time.monotonic()
        seats.rejoin_cell = dict(self.cell)
        assert seats.rejoin_release.wait(30)
        text = json.dumps([_finding("b1", "blocking", breaks="claim_1", summary="the deck has six slides")])
        return ReviewAttemptResult(message={"content": text}, usage=usage, raw_text=text)


def _collect(ctx, fingerprint):
    """The agent's $0 collection of the current wave (drain window 0)."""
    from ouroboros.tools import plan_review as pr

    return pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fingerprint, "items": []})


def _s3_row(h):
    return next(a for a in _state(h)["waves"][-1]["actors"] if a["slot_id"] == "s3")


def test_a_seat_surviving_a_restart_holds_green_and_delivers_its_late_blocking_finding(harness, monkeypatch):
    import ouroboros.review_custody as custody
    from ouroboros import delegate_custody
    from ouroboros.config import NESTED_SETTLEMENT_MARGIN_SEC

    harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"), ("s3", "m/c", "session"))
    monkeypatch.setattr("ouroboros.tools.plan_review_runtime.plan_panel_health_snapshot", lambda _slots: {})
    seats = _ScriptedSeats()
    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor", seats)

    # ---- first process ------------------------------------------------------
    ctx = harness.make_ctx()
    try:
        first = _call(ctx)
        assert not _control(first)["closed"]
        assert _wait_until(lambda: seats.s3_calls == 1)
        seats.first_release.set()
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, "task-1")) == 1)
    finally:
        seats.first_release.set()
    # The identical envelope resumes the in-flight wave: two clean seats and the
    # delegated seat still owed — persisted as in_flight WITH its token. Not GREEN.
    resumed = _control(_call(ctx))
    assert resumed["outcome"] != "GREEN" and resumed["closed"] is False, resumed
    row = _s3_row(harness)
    assert row["operation_state"] == "in_flight" and row["pending_invocation_id"] == INVOCATION, row
    assert _state(harness)["waves"][-1]["custody_pending"] is True
    assert seats.s3_calls == 2
    fingerprint = _state(harness)["waves"][-1]["request_fingerprint"]
    started_ts, cap = delegate_custody.run_timing(harness.drive, RUN_ID)
    assert started_ts and cap == MAX_SECONDS

    # ---- restart ------------------------------------------------------------
    with custody._ACTIVE_LOCK:
        assert not custody._ACTIVE, "no live worker survives a restart"
    fresh = harness.make_ctx()  # a new process knows the seat only through the persisted row

    # ---- second process: the $0 collection rejoins the run, never waits ------
    started = time.monotonic()
    try:
        collected = _control(_collect(fresh, fingerprint))
        assert time.monotonic() - started < 10, "a collection must not wait for the owed seat"
        assert collected["outcome"] != "GREEN" and collected["closed"] is False, collected
        assert _state(harness)["waves"][-1]["custody_pending"] is True
        assert _wait_until(lambda: seats.rejoin_remaining is not None)
        # The rejoin inherits what remains of the run's ORIGINAL window (its 540 s cap
        # minus the seconds already run), not the settlement margin, not a fresh window.
        assert MAX_SECONDS - 20 < seats.rejoin_remaining <= MAX_SECONDS, seats.rejoin_remaining
        assert seats.rejoin_remaining > float(NESTED_SETTLEMENT_MARGIN_SEC)
        assert seats.rejoin_cell == {"pending_invocation_id": INVOCATION}
        seats.rejoin_release.set()
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, "task-1")) == 2)
    finally:
        seats.rejoin_release.set()
    final = _control(_collect(fresh, fingerprint))
    assert final == {"outcome": "REVIEW_REQUIRED", "closed": False}, final
    wave = _state(harness)["waves"][-1]
    assert wave["custody_pending"] is False and wave["paid"] is True
    assert "s3:b1" in {f.get("finding_id") for f in wave.get("findings") or []}, wave.get("findings")
    assert seats.s3_calls == 3, "one rejoin, never a second paid send"
