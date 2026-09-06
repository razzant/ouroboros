"""The session poller: early termination and honest cancel outcomes.

Split by theme out of ``tests/test_review_agent_session_route.py``. This module
owns the poller: a waiting-on-user session ends early and typed, the cancel
read is honest about what it proved on both branches, a discovered success is
never lost to a re-read blip, and confirmed natural terminals are attributed to
the run rather than the host.
"""

from types import SimpleNamespace

import pytest


from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport

# The autouse transport fixture is requested by pytest, not by name, so it is re-bound
# through a module attribute exactly as in the sibling suites: leaving it behind would
# have silently let this suite reach the real owned gateway.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport

# ---------------------------------------------------------------------------
# F18: the poller terminates a slot EARLY on a session waiting on user
# ---------------------------------------------------------------------------


def test_poller_terminates_a_waiting_on_user_session_early_and_typed(tmp_path):
    """F18 (sol #6 minimal form + grok): a delegated review session that parks
    on an interactive question cannot be answered host-side (review slots are
    non-interactive; answering support is a future issue) — waiting out the
    engine timeout burns the whole slot budget in silence. The poller cancels
    the run through the verified-cancel path under its OWN typed reason and
    raises a typed failure naming the pending question."""
    import time as _time

    from ouroboros.review_execution import (
        ReviewSessionWaitingOnUser,
        _poll_session_terminal,
    )

    cancelled = {}

    class _WaitingGateway:
        def get_run(self, run_id, *, timeout_sec=None):
            return {
                "lastSeq": 3,
                "pendingInteractions": [{
                    "interactionId": "int-9",
                    "questions": [{"id": "q1",
                                   "question": "Which schema version applies?",
                                   "options": [], "multi_select": False}],
                }],
                "summary": {"state": "running", "waitingOnUser": True},
            }

    class _CustodyStub:
        @staticmethod
        def is_terminal(detail):
            return False

        @staticmethod
        def summary_of(detail):
            return detail.get("summary") or {}

        @staticmethod
        def cancel_and_verify(drive, gateway, entry, reason):
            cancelled["reason"] = reason
            return {"outcome": "confirmed"}

    started = _time.monotonic()
    with pytest.raises(ReviewSessionWaitingOnUser) as excinfo:
        _poll_session_terminal(_WaitingGateway(), _CustodyStub(), tmp_path,
                               SimpleNamespace(run_id="run-w"), "run-w", 600.0)
    # EARLY: no slot-long burn — the first poll already decided.
    assert _time.monotonic() - started < 30.0
    # The run was cancelled under the typed host-side reason (no
    # cancel-vs-decline ambiguity), and the failure names the question.
    assert cancelled["reason"] == "review_session_waiting_on_user"
    text = str(excinfo.value)
    assert "int-9" in text
    assert "Which schema version applies?" in text
    assert "host-cancelled" in text


def _parked_detail(timeout_at):
    row = {
        "interactionId": "int-9",
        "questions": [{"id": "q1", "question": "Which schema version applies?",
                       "options": [], "multi_select": False}],
    }
    if timeout_at is not None:
        row["timeoutAt"] = timeout_at
    return {
        "lastSeq": 3,
        "pendingInteractions": [row],
        "summary": {"state": "running", "waitingOnUser": True},
    }


class _PollCustodyStub:
    def __init__(self):
        self.cancelled = {}

    @staticmethod
    def is_terminal(detail):
        return str((detail.get("summary") or {}).get("state") or "") == "succeeded"

    @staticmethod
    def summary_of(detail):
        return detail.get("summary") or {}

    def cancel_and_verify(self, drive, gateway, entry, reason):
        self.cancelled["reason"] = reason
        return {"outcome": "confirmed"}


def test_poller_keeps_polling_when_the_engine_timeout_lands_inside_the_slot(
        tmp_path, monkeypatch):
    """R2-2 (regressions HIGH — the genuine F18 regression): a parked question
    whose OWN timeout_at provably lands inside the slot's remaining budget is
    the recoverable pre-F18 case — the engine benign-declines it and the
    session resumes. The poller must KEEP POLLING on the slot's own clock (not
    an owner host-wait), never cancel a session the engine is about to
    resume."""
    from datetime import datetime, timedelta, timezone

    from ouroboros import review_execution as rx

    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.01)
    soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    calls = {"n": 0}

    class _RecoveringGateway:
        def get_run(self, run_id, *, timeout_sec=None):
            calls["n"] += 1
            if calls["n"] < 3:
                return _parked_detail(soon)
            return {"lastSeq": 4, "summary": {"state": "succeeded"}}

    custody = _PollCustodyStub()
    detail = rx._poll_session_terminal(
        _RecoveringGateway(), custody, tmp_path,
        SimpleNamespace(run_id="run-w"), "run-w", 600.0)
    assert detail["summary"]["state"] == "succeeded"
    assert calls["n"] >= 3
    assert custody.cancelled == {}, "a recoverable park must not be cancelled"


@pytest.mark.parametrize("timeout_at", [
    None,                     # absent: no engine expiry exists
    "not-a-timestamp",        # unparseable: no PROVEN expiry
    "at_deadline",            # lands at/after the slot deadline
    "far_future",             # far beyond the slot
])
def test_poller_still_terminates_when_no_expiry_lands_inside_the_slot(
        tmp_path, timeout_at):
    """R2-2, the four negative shapes: without a PROVEN engine expiry inside
    the slot budget, the park is still the F18 slot-long silent burn — cancel
    plus the typed raise, exactly as before."""
    from datetime import datetime, timedelta, timezone

    from ouroboros.review_execution import (
        ReviewSessionWaitingOnUser,
        _poll_session_terminal,
    )

    slot_seconds = 600.0
    if timeout_at == "at_deadline":
        timeout_at = (datetime.now(timezone.utc)
                      + timedelta(seconds=slot_seconds + 1)).isoformat()
    elif timeout_at == "far_future":
        timeout_at = (datetime.now(timezone.utc)
                      + timedelta(days=1)).isoformat()

    class _ParkedGateway:
        def get_run(self, run_id, *, timeout_sec=None):
            return _parked_detail(timeout_at)

    custody = _PollCustodyStub()
    with pytest.raises(ReviewSessionWaitingOnUser):
        _poll_session_terminal(_ParkedGateway(), custody, tmp_path,
                               SimpleNamespace(run_id="run-w"), "run-w", slot_seconds)
    assert custody.cancelled["reason"] == "review_session_waiting_on_user"


# ---------------------------------------------------------------------------
# BR1-1: the poller is HONEST about what the cancel proved, on both branches
# ---------------------------------------------------------------------------


class _OutcomeCustodyStub:
    """cancel_and_verify scripted to one typed outcome (or an exception)."""

    def __init__(self, outcome, state="running", raises=False):
        self.outcome, self.state, self.raises = outcome, state, raises
        self.cancels = []

    @staticmethod
    def is_terminal(detail):
        return str((detail.get("summary") or {}).get("state") or "") in (
            "succeeded", "failed", "cancelled", "interrupted")

    @staticmethod
    def summary_of(detail):
        return detail.get("summary") or {}

    def cancel_and_verify(self, drive, gateway, entry, reason):
        self.cancels.append(reason)
        if self.raises:
            raise RuntimeError("daemon died mid-cancel")
        return {"outcome": self.outcome, "state": self.state,
                "accepted": self.outcome in ("confirmed", "requested"),
                "control_status": "", "fault_reason": "", "detail": ""}


class _RunningGateway:
    """Non-terminal, no pending questions — drives the timeout branch."""

    def __init__(self, waiting=False):
        self.waiting = waiting

    def get_run(self, run_id, *, timeout_sec=None):
        if self.waiting:
            return _parked_detail(None)
        return {"lastSeq": 1, "summary": {"state": "running"}}


class _SucceededAfterCancelGateway(_RunningGateway):
    """The natural-success race: the verify read finds the run SUCCEEDED, so
    the re-read after cancel returns the natural terminal."""

    def __init__(self, waiting=False):
        super().__init__(waiting)
        self.cancel_seen = False

    def get_run(self, run_id, *, timeout_sec=None):
        if self.cancel_seen:
            return {"lastSeq": 9, "summary": {"state": "succeeded"},
                    "primaryOutput": {"text": "[]"}}
        return super().get_run(run_id, timeout_sec=timeout_sec)


_CANCEL_OUTCOME_CASES = [
    ("confirmed", "host-cancelled"),
    ("requested", "may still be live"),
    ("failed", "may still be live"),
    ("containment_fault_run_may_still_be_live", "may still be live"),
]


@pytest.mark.parametrize("outcome,expected", _CANCEL_OUTCOME_CASES)
def test_waiting_on_user_raise_carries_the_honest_cancel_outcome(
        tmp_path, outcome, expected):
    """BR1-1(b): "host-cancelled" is claimed ONLY for a `confirmed` verified
    receipt; requested/failed/containment-fault raises say the cancel was
    requested-but-unverified and the run MAY STILL BE LIVE — same typed
    exception class, distinct reason text."""
    from ouroboros.review_execution import (
        ReviewSessionWaitingOnUser,
        _poll_session_terminal,
    )

    stub = _OutcomeCustodyStub(outcome)
    with pytest.raises(ReviewSessionWaitingOnUser) as excinfo:
        _poll_session_terminal(_RunningGateway(waiting=True), stub, tmp_path,
                               SimpleNamespace(run_id="run-h"), "run-h", 600.0)
    text = str(excinfo.value)
    assert expected in text
    if outcome != "confirmed":
        assert "host-cancelled" not in text
        assert outcome in text
    assert stub.cancels == ["review_session_waiting_on_user"]


@pytest.mark.parametrize("outcome,expected", _CANCEL_OUTCOME_CASES)
def test_slot_timeout_raise_carries_the_honest_cancel_outcome(
        tmp_path, monkeypatch, outcome, expected):
    """BR1-1(c): the same outcome-honesty on the slot-timeout cancel — a
    TimeoutError whose text claims "host-cancelled" only when the receipt is
    verified."""
    from ouroboros import review_execution as rx

    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.01)
    stub = _OutcomeCustodyStub(outcome)
    with pytest.raises(TimeoutError) as excinfo:
        rx._poll_session_terminal(_RunningGateway(), stub, tmp_path,
                                  SimpleNamespace(run_id="run-t"), "run-t", 1.0)
    text = str(excinfo.value)
    assert "exceeded the slot budget" in text
    assert expected in text
    if outcome != "confirmed":
        assert "host-cancelled" not in text
        assert outcome in text
    assert stub.cancels == ["review_slot_timeout"]


@pytest.mark.parametrize("waiting,slot_seconds", [
    (True, 600.0),   # waiting-on-user branch
    (False, 1.0),    # slot-timeout branch
])
def test_natural_success_discovered_by_the_cancel_read_wins_on_both_branches(
        tmp_path, monkeypatch, waiting, slot_seconds):
    """BR1-1(a) COMPLETION WINS: when cancel/verify reports the run reached a
    natural SUCCESS terminal (settled state=succeeded), the poller consumes
    that terminal as the slot's ordinary result instead of raising."""
    from ouroboros import review_execution as rx

    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.01)
    gateway = _SucceededAfterCancelGateway(waiting=waiting)
    stub = _OutcomeCustodyStub("confirmed", state="succeeded")
    orig = stub.cancel_and_verify

    def _cancel(drive, gw, entry, reason):
        gateway.cancel_seen = True
        return orig(drive, gw, entry, reason)

    stub.cancel_and_verify = _cancel
    detail = rx._poll_session_terminal(
        gateway, stub, tmp_path, SimpleNamespace(run_id="run-s"), "run-s",
        slot_seconds)
    assert detail["summary"]["state"] == "succeeded"
    assert detail["primaryOutput"]["text"] == "[]"


@pytest.mark.parametrize("waiting", [True, False])
def test_a_raising_cancel_is_reported_unverified_not_host_cancelled(
        tmp_path, monkeypatch, waiting):
    """BR1-1 exception shape: a cancel/verify that RAISES is an unverified
    attempt — the typed slot failure still fires (same exception class per
    branch) and its text says the run may still be live, never
    "host-cancelled"."""
    from ouroboros import review_execution as rx

    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.01)
    stub = _OutcomeCustodyStub("confirmed", raises=True)
    exc_type = rx.ReviewSessionWaitingOnUser if waiting else TimeoutError
    with pytest.raises(exc_type) as excinfo:
        rx._poll_session_terminal(
            _RunningGateway(waiting=waiting), stub, tmp_path,
            SimpleNamespace(run_id="run-x"), "run-x",
            600.0 if waiting else 1.0)
    text = str(excinfo.value)
    assert "may still be live" in text
    assert "host-cancelled" not in text


# ---------------------------------------------------------------------------
# BR2-1: a discovered success is NEVER lost to a re-read failure
# BR2-2: a confirmed natural terminal is attributed to the run, not the host
# ---------------------------------------------------------------------------


class _CarryingCustodyStub(_OutcomeCustodyStub):
    """cancel_and_verify that also carries the verify read's own detail
    (the additive `terminal_detail` key of `_cancel_result`, BR2-1)."""

    def __init__(self, outcome, state="running", carried=None):
        super().__init__(outcome, state)
        self.carried = carried

    def cancel_and_verify(self, drive, gateway, entry, reason):
        result = super().cancel_and_verify(drive, gateway, entry, reason)
        if self.carried is not None:
            result["terminal_detail"] = self.carried
        return result


class _BlippingGateway(_RunningGateway):
    """After the cancel, `fail_reads` re-reads RAISE (the transport blip),
    then the settled success detail is served."""

    def __init__(self, waiting=False, fail_reads=99):
        super().__init__(waiting)
        self.cancel_seen = False
        self.fail_reads = fail_reads
        self.reads_after_cancel = 0

    def get_run(self, run_id, *, timeout_sec=None):
        if self.cancel_seen:
            self.reads_after_cancel += 1
            if self.reads_after_cancel <= self.fail_reads:
                raise RuntimeError("transport blip")
            return {"lastSeq": 9, "summary": {"state": "succeeded"},
                    "primaryOutput": {"text": "[]"}}
        return super().get_run(run_id, timeout_sec=timeout_sec)


def _arm_cancel(gateway, stub):
    """Flip the gateway into its post-cancel behaviour when the stub cancels."""
    orig = stub.cancel_and_verify

    def _cancel(drive, gw, entry, reason):
        gateway.cancel_seen = True
        return orig(drive, gw, entry, reason)

    stub.cancel_and_verify = _cancel


@pytest.mark.parametrize("waiting", [True, False])
def test_the_carried_terminal_detail_wins_with_no_second_fetch(
        tmp_path, monkeypatch, waiting):
    """BR2-1: when cancel_and_verify carried the verify read's own succeeded
    detail, the poller consumes it AS the slot result — even though every
    re-read would raise — and issues no post-cancel fetch at all (the extra
    get_run round-trip of the success race is gone)."""
    from ouroboros import review_execution as rx

    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.01)
    carried = {"lastSeq": 9, "summary": {"state": "succeeded"},
               "primaryOutput": {"text": "[]"}}
    gateway = _BlippingGateway(waiting=waiting, fail_reads=99)
    stub = _CarryingCustodyStub("confirmed", state="succeeded", carried=carried)
    _arm_cancel(gateway, stub)
    detail = rx._poll_session_terminal(
        gateway, stub, tmp_path, SimpleNamespace(run_id="run-c"), "run-c",
        600.0 if waiting else 1.0)
    assert detail is carried
    assert gateway.reads_after_cancel == 0, "no second fetch after a carried detail"


def test_an_uncarried_success_survives_one_re_read_blip(tmp_path, monkeypatch):
    """BR2-1 bounded retry: without a carried detail (older custody shape),
    a single re-read failure does not lose the success — the one retry
    fetches the settled terminal."""
    from ouroboros import review_execution as rx

    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.01)
    gateway = _BlippingGateway(fail_reads=1)
    stub = _OutcomeCustodyStub("confirmed", state="succeeded")
    _arm_cancel(gateway, stub)
    detail = rx._poll_session_terminal(
        gateway, stub, tmp_path, SimpleNamespace(run_id="run-r"), "run-r", 1.0)
    assert detail["summary"]["state"] == "succeeded"
    assert gateway.reads_after_cancel == 2


@pytest.mark.parametrize("waiting", [True, False])
def test_an_unreadable_settled_success_raises_typed_never_may_still_be_live(
        tmp_path, monkeypatch, waiting):
    """BR2-1 last resort: the state is KNOWN (succeeded, settled) — when the
    detail stays unreadable after the bounded retry, the typed failure says
    exactly that and names the recovery surfaces; it never falls through to
    the "may still be live" honesty clause, and the read attempts stay
    bounded (one retry, never a loop)."""
    from ouroboros import review_execution as rx

    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.01)
    gateway = _BlippingGateway(waiting=waiting, fail_reads=99)
    stub = _OutcomeCustodyStub("confirmed", state="succeeded")
    _arm_cancel(gateway, stub)
    with pytest.raises(rx.ReviewSessionSucceededResultUnavailable) as excinfo:
        rx._poll_session_terminal(
            gateway, stub, tmp_path, SimpleNamespace(run_id="run-u"), "run-u",
            600.0 if waiting else 1.0)
    text = str(excinfo.value)
    assert "SUCCEEDED" in text and "settled" in text
    assert "delegate_wait" in text, "the recovery surface is named"
    assert "may still be live" not in text
    assert "host-cancelled" not in text
    assert gateway.reads_after_cancel == 2, "one bounded retry, never a loop"
    assert excinfo.value.delegated_run_started is True
    assert excinfo.value.delegated_run_id == "run-u"


@pytest.mark.parametrize("state,must,must_not", [
    ("failed", "its own terminal state 'failed'", "host-cancelled"),
    ("interrupted", "its own terminal state 'interrupted'", "host-cancelled"),
    ("cancelled", "host-cancelled with a verified terminal receipt", "its own terminal"),
    ("settled", "host-cancelled with a verified terminal receipt", "its own terminal"),
    ("absent", "host-cancelled with a verified terminal receipt", "its own terminal"),
    ("", "host-cancelled with a verified terminal receipt", "its own terminal"),
])
def test_confirmed_attribution_follows_the_verified_state(state, must, must_not):
    """BR2-2, every wording branch: a `confirmed` whose verified state is the
    run's OWN non-success terminal (failed/interrupted) is attributed to the
    run; 'cancelled' — and the receipt-is-the-cancel states ''/settled/absent —
    keep the host-cancelled verified-receipt wording; nothing here ever says
    "may still be live"."""
    from ouroboros.review_execution import _cancel_honesty_clause

    text = _cancel_honesty_clause("confirmed", state)
    assert must in text, text
    assert must_not not in text, text
    assert "may still be live" not in text
    # The unverified wording is unchanged by the state parameter.
    assert "may still be live" in _cancel_honesty_clause("requested", state)


@pytest.mark.parametrize("waiting", [True, False])
def test_a_confirmed_natural_terminal_is_attributed_to_the_run_not_the_host(
        tmp_path, monkeypatch, waiting):
    """BR2-2 through the poller: a run that had ALREADY failed on its own when
    the verify read arrived raises the branch's typed failure wording the
    natural terminal — never claiming the host's cancel stopped it."""
    from ouroboros import review_execution as rx

    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.01)
    stub = _OutcomeCustodyStub("confirmed", state="failed")
    exc_type = rx.ReviewSessionWaitingOnUser if waiting else TimeoutError
    with pytest.raises(exc_type) as excinfo:
        rx._poll_session_terminal(
            _RunningGateway(waiting=waiting), stub, tmp_path,
            SimpleNamespace(run_id="run-f"), "run-f",
            600.0 if waiting else 1.0)
    text = str(excinfo.value)
    assert "its own terminal state 'failed'" in text
    assert "host-cancelled" not in text
    assert "may still be live" not in text


def test_clock_crossing_between_timeout_checks_still_cancels_the_running_session(
        tmp_path, monkeypatch):
    """The second clock read may cross the deadline before sleep is entered."""
    from ouroboros import review_execution as rx

    ticks = iter([0.0, 0.0, 0.5, 1.1, 1.2])
    # Replace the module reference, not attributes on the process-global
    # ``time`` module: pytest itself calls monotonic on Windows and would
    # exhaust the finite tick iterator (same scoping precedent as
    # tests/test_external_unmetered_dispatch.py).
    monkeypatch.setattr(
        rx,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _seconds: None),
    )
    monkeypatch.setattr(
        rx,
        "_poll_detail",
        lambda *_args, **_kwargs: {"lastSeq": 1, "summary": {"state": "running"}},
    )
    stub = _OutcomeCustodyStub("confirmed")
    with pytest.raises(TimeoutError, match="exceeded the slot budget"):
        rx._poll_session_terminal(
            _RunningGateway(), stub, tmp_path,
            SimpleNamespace(run_id="run-cross"), "run-cross", 1.0,
        )
    assert stub.cancels == ["review_slot_timeout"]
