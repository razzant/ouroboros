"""After a Restart, the exact reference collection reads a seat's saved complete answer; owner death is no verdict.

The shape of a real sample (#2590 class): the server restarted while a commit wave was settling. Every
seat had already written its complete, bound producer response (``observability/calls/<op>_response``),
but the attempt rows were still in flight and its ledger rows still open, so the confirmed death of the
owning process rewrote each tokenless row to ``process_local_review_worker_lost`` and failed the attempt.
``commit_reviewed`` with the returned ``review_reference`` (no disposition: the collection of that paid
gate, the same path an explicit ``continue_seats`` takes) then recovers those saved answers at $0 before
trusting the rewritten rows: no reviewer is asked again, the ledger settles with their verdict, and nothing
is offered as a continuation (a lost process left no engine stop to continue). Only the API seats' model is
scripted; the wave, retained sources, attempt rows, owner reconciliation and collection are real.
"""
from __future__ import annotations

import json


from ouroboros import review_ledger
from ouroboros.review_continuation import continuable_seats
from ouroboros.review_ledger import ledger_root, load_record
from ouroboros.review_state import ReviewStateMutation, load_state, update_state
from tests.test_receptive_commit_review import _git, _join, _commit
from tests.test_receptive_commit_review import author as _author

author = _author
DEAD_OWNER = 4_194_301


def _commit_text(a, **kwargs):
    outcome = {}
    _join(_commit(a, outcome, **kwargs))
    return outcome["result"]


def _restart_left_the_wave_open(a, record_id):
    """What the Restart left: the attempt and its ledger rows open under the dead owner, answers saved."""
    def reopen(state):
        attempt = state.attempts[-1]
        attempt.status, attempt.phase, attempt.late_result_pending = "reviewing", "late_wait", True
        attempt.review_owner_pid, attempt.finished_ts = DEAD_OWNER, ""
        for row in attempt.triad_raw_results:
            row.update(operation_state="in_flight", status="pending", late_result_pending=True)
            for key in ("error", "failure_code", "raw_text", "text", "verdict"):
                row.pop(key, None)
        return ReviewStateMutation(None, changed=True)
    update_state(a.data, reopen)

    def open_rows(record):
        for seat in record["rows"]:
            seat.update(status="pending", operation_state="in_flight")
        return {**record, "state": review_ledger.STATE_PENDING}
    review_ledger.revise_record(ledger_root(a.ctx), record_id, open_rows)


def _forget_this_process(a):
    """A new server process: nothing in memory replays the earlier wave (only the disk speaks)."""
    from ouroboros import delegate_custody, review_custody

    for name in ("_review_settled_attempts", "_review_pending_invocations"):
        setattr(a.ctx, name, {})
    with review_custody._ACTIVE_LOCK:
        review_custody._ACTIVE.clear()
        review_custody._NO_RESEND.clear()
    delegate_custody._CUSTODY.clear()


def test_a_reference_collection_after_a_restart_reads_the_saved_answers_and_offers_no_continuation(author):
    from ouroboros.review_owner_custody import reconcile_review_custody_after_confirmed_process_deaths

    a = author
    a.reviewer.release()
    head = _git(a.repo, "rev-parse", "HEAD")
    first = _commit_text(a)
    reference = json.JSONDecoder().raw_decode(first, first.index("{\"review_reference\""))[0]["review_reference"]
    record_id = reference["review_record_id"]
    asked = list(a.reviewer.rounds)
    assert sorted(asked) == ["m1", "m2"] and load_record(ledger_root(a.ctx), record_id)["verdict"]["aggregate"] == "PASS"

    _restart_left_the_wave_open(a, record_id)
    _forget_this_process(a)
    reconciled = reconcile_review_custody_after_confirmed_process_deaths(a.data, {DEAD_OWNER})
    attempt = load_state(a.data).attempts[-1]
    assert reconciled["reconciled"] and attempt.status == "failed", attempt
    assert {row["failure_code"] for row in attempt.triad_raw_results} == {"process_local_review_worker_lost"}
    lost = load_record(ledger_root(a.ctx), record_id)
    assert {seat["operation_state"] for seat in lost["rows"]} == {"in_flight"}  # the ledger never heard of the death
    assert continuable_seats(lost) == []  # owner death leaves nothing an engine could continue

    _git(a.repo, "add", "change.py")  # the handback unstaged the candidate; the reviewed bytes again
    collected = _commit_text(a, review_reference=reference)
    record = load_record(ledger_root(a.ctx), record_id)
    assert a.reviewer.rounds == asked, "the saved answers were read; no reviewer was asked again"
    assert record["state"] == "settled" and record["verdict"]["aggregate"] == "PASS", record["verdict"]
    assert {seat["status"] for seat in record["rows"]} == {"responded"}
    assert "continuable_seats" not in collected and _git(a.repo, "rev-parse", "HEAD") == head, collected
