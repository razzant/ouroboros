"""A result review's binding follows a genuinely new outcome of the same record, never a cost-only revision.

The binding keys on each settled seat's per-operation ``outcome_id`` plus that outcome's terminal
status, advisers included (``review_seat_outcome.outcome_identity``). An explicit continuation (a new
operation) or a late answer under the same operation refreshes the bound aggregate and withdraws the
author's earlier informed finish, which read the old outcome; money, tests or wait cursors that raise
the record's revision change nothing. The outcome axes (``outcomes._review_axis``) are the consumer.
"""
from __future__ import annotations

import copy

from ouroboros.outcomes import _review_axis
from ouroboros.review_seat_outcome import outcome_identity
from ouroboros.tools.review_result import _bind_review_run


def _row(seat, operation, status, *, additional=False, open_=False):
    row = {"seat_id": seat, "operation_id": operation, "status": "pending" if open_ else status,
           "operation_state": "in_flight" if open_ else "settled", "additional": additional}
    if not open_:
        row["outcome"] = {"outcome_id": f"{operation}:{seat}", "status": status}
    return row


def _record(rows, aggregate, critical=1, revision=3, usd=0.5):
    return {"record_id": "rl-result", "revision": revision, "rows": rows, "cost": {"usd": usd, "unknown": False},
            "verdict": {"aggregate": aggregate, "reason": aggregate.lower(),
                        "critical_findings": [{"item": "x"}] * critical}}


STOPPED = [_row("a", "op-a", "responded"), _row("b", "op-b", "error"), _row("adviser", "op-c", "responded", additional=True)]


def _finished(trace, record):
    run = _bind_review_run(trace, record, "subject-1", "advisory")
    run["informed_finish"] = True  # the mind read these findings and delivered (result_review_at_delivery)
    return run


def test_a_cost_or_wait_revision_keeps_the_binding_and_the_informed_finish():
    trace = {}
    _finished(trace, _record(STOPPED, "FAIL"))
    run = _bind_review_run(trace, _record(copy.deepcopy(STOPPED), "FAIL", revision=9, usd=0.75), "subject-1", "advisory")
    assert run["informed_finish"] is True and run["aggregate"] == "FAIL"
    assert _review_axis(trace)["author_finish"] == "author_finish"  # a legacy ``True`` finish stays readable


def test_a_continued_seat_refreshes_the_binding_and_withdraws_the_old_informed_finish():
    trace = {}
    _finished(trace, _record(STOPPED, "FAIL"))
    continued = [STOPPED[0], _row("b", "review_continue_x", "responded"), STOPPED[2]]
    run = _bind_review_run(trace, _record(continued, "FAIL", critical=3), "subject-1", "advisory")
    assert "informed_finish" not in run and run["critical_count"] == 3
    assert run["outcomes"] == outcome_identity(_record(continued, "FAIL"))
    assert "author_finish" not in _review_axis(trace)  # the new outcome has not been read by the author yet
    trace = {}
    _finished(trace, _record(STOPPED, "FAIL"))
    passed = _bind_review_run(trace, _record(continued, "PASS", critical=0), "subject-1", "advisory")
    assert passed["aggregate_signal"] == "PASS" and _review_axis(trace)["status"] == "pass"


def test_a_late_answer_under_the_same_operation_is_a_new_outcome():
    lost = [STOPPED[0], {**_row("b", "op-b", "error"), "outcome": {"outcome_id": "op-b:b", "status": "custody_lost"}}, STOPPED[2]]
    late = [STOPPED[0], _row("b", "op-b", "responded"), STOPPED[2]]
    assert outcome_identity(_record(lost, "FAIL")) != outcome_identity(_record(late, "FAIL"))
    trace = {}
    _finished(trace, _record(lost, "FAIL"))
    assert "informed_finish" not in _bind_review_run(trace, _record(late, "FAIL"), "subject-1", "advisory")


def test_an_adviser_finding_counts_and_an_open_seat_does_not_yet():
    with_adviser = outcome_identity(_record(STOPPED, "FAIL"))
    adviser_changed = outcome_identity(_record([*STOPPED[:2], _row("adviser", "op-d", "responded", additional=True)], "FAIL"))
    assert with_adviser != adviser_changed  # advisers vote nowhere, but their findings are new feedback
    reopened = outcome_identity(_record([STOPPED[0], _row("b", "review_continue_x", "", open_=True), STOPPED[2]], "FAIL"))
    assert reopened == sorted(["op-a:a:responded", "op-c:adviser:responded"])


def test_a_binding_from_before_the_identity_existed_refreshes_only_on_a_changed_verdict():
    trace = {}
    run = _finished(trace, _record(STOPPED, "FAIL"))
    run.pop("outcomes")  # a checkpointed binding written before this field
    _bind_review_run(trace, _record(STOPPED, "FAIL"), "subject-1", "advisory")
    assert run["informed_finish"] is True and run["outcomes"]
    run.pop("outcomes")
    _bind_review_run(trace, _record(STOPPED, "PASS", critical=0), "subject-1", "advisory")
    assert "informed_finish" not in run and run["aggregate"] == "PASS"
