"""A review that was merely AWAITED when its task ended is a gap, never a degradation.

A plan ask whose participants had simply not answered yet, or a review run whose slots
were still working, leaves the task's own outcome clean: the loud final-message
disclosure stays and a typed fact is recorded (``execution.plan_review`` /
``review.status`` = ``awaiting``). Every real outcome — an open objection, no quorum,
unresolved custody, a refused or failed slot, a critic verdict beneath the placeholder,
FAIL, a rail — keeps exactly the status, reason code and sentence it had. The review runs
here are the stored host panels (``review_runs`` rows with actors) the outcome axes read,
old task-acceptance panels included.
"""
from __future__ import annotations

import queue

import pytest

from ouroboros import _outcome_receipts, loop
from ouroboros.outcomes import _objective_axis, _review_axis, derive_loop_outcome, public_task_result
from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict, completion_status_label
from tests.test_delivery_forced_finalization import _forced_test_context

ANSWER = "The complete report includes the requested budget."
AWAITED_NOTICE = ("\n\n⚠️ Plan review: the answers of ask(s) a1 had not arrived when the task ended; "
                  "silence holds nothing and is not agreement.")
OBJECTIONS_NOTICE = ("\n\n⚠️ Plan review: 1 open objection remains open — triad_a/o1 (ask a1, revision 1): the bar is low; "
                     "work proceeded under the owner-selected advisory enforcement.")
AWAITED_SENTENCE = "Not every plan reviewer had answered when the task ended."
OBJECTIONS_SENTENCE = TASK_CAUSE_PHRASES["plan_review_objections_open"]


# ------------------------------------------------------------------ plan: finalization and the record

def _decision(**facts):
    return {"required": True, "self_opened": True, "enforcement": "advisory", "status": "clear", "allow": True,
            "open_objections": [], "holding": [], "pending_asks": [], "asks_dispatched": 1, "force_plan_met": True,
            "revision": 1, **facts}


def _awaiting():
    return _decision(status="awaiting", pending_asks=["a1"], review_only_awaited=True)


def _objections():
    return _decision(status="objections_open", plan_review_class="objections_open", holding=["triad_a:o1"],
                     open_objections=[{"seat_id": "triad_a", "id": "o1", "text": "the bar is low", "holds": True,
                                       "not_adjudicated": False, "ask_id": "a1", "revision": 1}])


def _finalize(tmp_path, monkeypatch, decision):
    _loop, tools, ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_kw: False)
    monkeypatch.setattr(loop, "_plan_gate_decision", lambda *_a, **_kw: dict(decision))
    loop._replace_delivery_candidate(tools, ctx, trace, ANSWER, control="candidate")
    text, usage, trace = loop._no_tool_final_answer(
        ANSWER, ctx, trace, tools, queue.Queue(), set(), lambda _t: None)
    assert text == ANSWER
    outcome = derive_loop_outcome(text, usage, trace)
    record = public_task_result({"status": "completed", "reason_code": outcome["reason_code"],
                                 "outcome_axes": outcome["outcome_axes"]})
    return usage, trace, outcome, record


def test_a_task_that_finished_over_an_awaited_plan_ask_is_not_degraded(tmp_path, monkeypatch):
    usage, trace, outcome, record = _finalize(tmp_path, monkeypatch, _awaiting())
    # BIBLE P3: the loud, durable, owner-visible sentence is exactly the one it was.
    assert usage["terminal_host_notice"] == AWAITED_NOTICE.strip()
    assert trace["delivery_candidate"]["degraded"] is False
    assert trace["delivery_candidate"]["degraded_reason"] == ""
    assert outcome["degraded"] is False and outcome["reason_code"] == "final_message"
    execution = outcome["outcome_axes"]["execution"]
    assert execution["status"] == "ok" and execution["failure"] is None
    assert execution["plan_review"] == "awaiting"  # the durable typed fact
    assert record["outcome_axes"]["execution"]["plan_review"] == "awaiting"  # survives normalization
    assert record["outcome_axes"]["objective"]["status"] == "not_evaluated"
    assert completion_status_label(record, {}) == "Done"
    assert _completion_verdict(record, {}) == AWAITED_SENTENCE == TASK_CAUSE_PHRASES["plan_review_awaiting"]
    assert _completion_verdict({**record, "reason_code": ""}, {}) == AWAITED_SENTENCE  # a row with no reason code
    # The fact is the cause sentence of a CLEAN card only: an amber or red card owes its colour to something else.
    warned = {**record, "outcome_axes": {**record["outcome_axes"], "objective": {
        "status": "not_evaluated", "warning": "residual_tool_errors_without_review"}}}
    assert completion_status_label(warned, {}) == "Done with warnings"
    assert _completion_verdict(warned, {}) == TASK_CAUSE_PHRASES["residual_tool_errors_without_review"]
    failed = {**record, "outcome_axes": {**record["outcome_axes"], "artifacts": {"status": "missing"}}}
    assert completion_status_label(failed, {}) == "Failed"
    assert _completion_verdict(failed, {}) == TASK_CAUSE_PHRASES["artifacts_unavailable"]


def test_an_open_objection_at_delivery_keeps_the_advisory_degradation_and_its_sentence(tmp_path, monkeypatch):
    usage, trace, outcome, record = _finalize(tmp_path, monkeypatch, _objections())
    assert usage["terminal_host_notice"] == OBJECTIONS_NOTICE.strip()
    assert trace["delivery_candidate"]["degraded"] is True
    assert trace["delivery_candidate"]["degraded_reason"] == "plan_review_advisory"
    assert outcome["degraded"] is True and outcome["reason_code"] == "plan_review_advisory"
    execution = outcome["outcome_axes"]["execution"]
    assert execution["status"] == "degraded" and execution["plan_review"] == "objections_open"
    assert execution["failure"] == {"kind": "finalization_control", "reason_code": "plan_review_advisory"}
    assert outcome["outcome_axes"]["objective"] == {
        "status": "degraded", "source": "delivery_finalization_control", "review_status": "skipped"}
    assert usage["terminal_plan_review_open"] is True
    assert completion_status_label(record, {}) == "Done with warnings"
    assert _completion_verdict(record, {}) == OBJECTIONS_SENTENCE


def test_a_rail_over_an_open_objection_keeps_its_own_outcome(tmp_path, monkeypatch):
    decision = _decision(status="rail_degraded", reason="round_limit", holding=["triad_a:o1"],
                         open_objections=_objections()["open_objections"], plan_review_class="objections_open")
    usage, trace, outcome, _record = _finalize(tmp_path, monkeypatch, decision)
    assert usage["terminal_host_notice"].startswith("⚠️ Plan review is open (1 open objection")
    assert "`round_limit`" in usage["terminal_host_notice"]
    assert trace["delivery_candidate"]["degraded_reason"] == "plan_review_advisory"
    assert outcome["outcome_axes"]["execution"]["plan_review"] == "objections_open"


def test_a_clean_plan_leaves_no_notice_and_no_fact(tmp_path, monkeypatch):
    usage, trace, outcome, _record = _finalize(tmp_path, monkeypatch, _decision())
    assert "terminal_host_notice" not in usage and "terminal_plan_review_open" not in usage
    assert trace["delivery_candidate"]["degraded"] is False
    assert "plan_review" not in outcome["outcome_axes"]["execution"]


def test_the_typed_fact_never_rides_a_non_clean_execution():
    """A rail owns its own reason: a stale projected decision cannot mark a forced terminal."""
    trace = {"plan_gate": _awaiting(), "tool_calls": [],
             "delivery_candidate": {"degraded": True, "degraded_reason": "round_limit"}}
    forced = derive_loop_outcome(ANSWER, {"execution_status": "failed", "reason_code": "round_limit",
                                          "_best_effort_extracted": True}, trace)
    assert forced["outcome_axes"]["execution"]["status"] == "best_effort"
    assert "plan_review" not in forced["outcome_axes"]["execution"]
    clean = derive_loop_outcome(ANSWER, {}, {"plan_gate": _decision(), "tool_calls": []})
    assert "plan_review" not in clean["outcome_axes"]["execution"]


# ------------------------------------------------------------------ review runs: the typed predicate

def _actor(slot, state="settled", *, status="ok", signal="PASS", parsed=True, raw="{}"):
    return {"slot_id": slot, "status": status, "signal": signal, "operation_state": state,
            "parsed": {"verdict": signal, "outcome_tier": "solved"} if parsed else None,
            "raw_text": raw if parsed else "",
            "late_result_pending": state != "settled"}


def _pending(slot):
    return _actor(slot, "pending_dispatch", status="error", signal="DEGRADED", parsed=False)


def _run(actors, signal="DEGRADED", **extra):
    return {"authority": "host_root", "aggregate_signal": signal, "degraded": signal == "DEGRADED",
            "actors": list(actors), **extra}


AWAITED_RUNS = {
    "nobody answered yet": _run([_pending("a"), _pending("b"), _pending("c")]),
    "one PASS below quorum beside two waits": _run([_actor("a"), _pending("b"), _pending("c")]),
}
REAL_RUNS = {
    "a wait beside a transport failure": _run(
        [_actor("a", status="error", signal="DEGRADED", parsed=False), _pending("b")]),
    "a wait beside a typed refusal": _run(
        [_actor("a", "not_dispatched", status="not_dispatched", signal="", parsed=False), _pending("b")]),
    "a wait beside a parse-degraded answer": _run([_actor("a", signal="DEGRADED"), _pending("b")]),
    "the logical window expired (in_flight)": _run(
        [_actor("a", "in_flight", status="error", signal="DEGRADED", parsed=False), _pending("b")]),
    "custody lost": _run([_actor("a", "custody_lost", status="error", signal="DEGRADED", parsed=False)]),
    "a settled no-quorum panel": _run(
        [_actor(s, status="error", signal="DEGRADED", parsed=False) for s in "abc"]),
    "an infrastructure failure with no actors": _run([]),
    "a pending row that already carries an answer": _run(
        [{**_pending("a"), "parsed": {"verdict": "DEGRADED"}, "raw_text": "{}"}]),
    "a pending row marked ok": _run([{**_pending("a"), "status": "ok"}, _pending("b")]),
    "a roster with a row that is not a record": _run([_pending("a"), "garbage"]),
}


@pytest.mark.parametrize("name", sorted(AWAITED_RUNS))
def test_a_review_run_that_was_only_awaited_is_awaiting_not_degraded(name):
    axis = _review_axis({"review_runs": [AWAITED_RUNS[name]]})
    assert axis["status"] == "awaiting" and axis["aggregate_signals"] == ["DEGRADED"]
    assert "outcome_tier" not in axis  # a minority answer is not the panel's tier
    assert _objective_axis(axis) == {"status": "not_evaluated", "source": "none", "review_status": "awaiting"}
    record = {"status": "completed", "reason_code": "final_message",
              "outcome_axes": {"execution": {"status": "ok"}, "review": axis, "objective": _objective_axis(axis)}}
    assert completion_status_label(record, {}) == "Done"


@pytest.mark.parametrize("name", sorted(REAL_RUNS))
def test_every_real_review_degradation_keeps_the_degraded_axis(name):
    axis = _review_axis({"review_runs": [REAL_RUNS[name]]})
    assert axis["status"] == "degraded"
    assert _objective_axis(axis)["status"] == "degraded"
    record = {"status": "completed", "reason_code": "final_message",
              "outcome_axes": {"execution": {"status": "ok"}, "review": axis, "objective": _objective_axis(axis)}}
    assert completion_status_label(record, {}) == "Done with warnings"


def test_a_fail_and_a_pass_keep_their_words_beside_a_wait():
    failed = _run([_actor("a", signal="FAIL"), _pending("b")], signal="FAIL")
    assert _review_axis({"review_runs": [failed]})["status"] == "fail"
    passed = _run([_actor("a"), _actor("b"), _pending("c")], signal="PASS")
    assert _review_axis({"review_runs": [passed]})["status"] == "pass"
    assert _outcome_receipts.review_runs_only_awaited([failed]) is False
    assert _outcome_receipts.review_runs_only_awaited([passed]) is False
    # A real degradation beside an awaited panel keeps the louder word.
    mixed = [AWAITED_RUNS["nobody answered yet"], REAL_RUNS["a settled no-quorum panel"]]
    assert _review_axis({"review_runs": mixed})["status"] == "degraded"


@pytest.mark.parametrize("tier, objective, label", [
    ("blocked_with_evidence", "fail", "Failed"), ("best_effort", "best_effort", "Done with warnings")])
def test_a_collected_pass_that_judged_the_work_unsolved_keeps_its_word_beside_a_wait(tier, objective, label):
    judged = {**_actor("a"), "parsed": {"verdict": "PASS", "outcome_tier": tier}}
    axis = _review_axis({"review_runs": [_run([judged, _pending("b"), _pending("c")])]})
    assert axis["status"] == "degraded" and axis["outcome_tier"] == tier  # exactly today's axis
    assert _objective_axis(axis)["status"] == objective
    record = {"status": "completed", "reason_code": "final_message",
              "outcome_axes": {"execution": {"status": "ok"}, "review": axis, "objective": _objective_axis(axis)}}
    assert completion_status_label(record, {}) == label


def test_the_predicate_itself_refuses_a_fail_aggregate_over_awaited_rows():
    # `_review_axis` answers `fail` before it asks; the ledger asks the predicate directly.
    assert _outcome_receipts.review_runs_only_awaited([_run([_pending("a"), _pending("b")], signal="FAIL")]) is False
    assert _outcome_receipts.review_runs_only_awaited([]) is False


def _selection(runs, *, replaced=False):
    return _outcome_receipts.ReviewRunSelection(
        all_runs=list(runs), current_runs=[run for run in runs if not run.get("superseded_by_revision")],
        superseded_only_acceptance_gap=False, superseded_aggregate_signals=[],
        current_candidate_unaccepted=False, has_replacement=replaced)


def test_the_verification_ledger_does_not_call_an_awaited_panel_a_failed_verification():
    for name, run in AWAITED_RUNS.items():
        status, superseded = _outcome_receipts.review_run_ledger_status(run, _selection([run]))
        assert (status, superseded) == ("not_evaluated", False), name
    for name, run in REAL_RUNS.items():
        assert _outcome_receipts.review_run_ledger_status(run, _selection([run]))[0] == "failed", name
    passed = _run([_actor("a"), _actor("b"), _pending("c")], signal="PASS")
    assert _outcome_receipts.review_run_ledger_status(passed, _selection([passed]))[0] == "ok"
    old = _run([_pending("a")], superseded_by_revision=True)
    assert _outcome_receipts.review_run_ledger_status(old, _selection([old, passed], replaced=True))[0] == "superseded"


def test_a_blocking_unaccepted_terminal_is_unchanged_by_an_awaited_panel():
    # An acceptance decision stored before the result review replaced that machine still reads.
    trace = {"review_runs": [AWAITED_RUNS["nobody answered yet"]], "acceptance_decision": {
        "status": "finalized_unaccepted", "reason": "review_degraded", "enforcement": "blocking"}}
    objective = _objective_axis(_review_axis(trace))
    assert (objective["status"], objective["outcome_tier"]) == ("fail", "blocked_with_evidence")
