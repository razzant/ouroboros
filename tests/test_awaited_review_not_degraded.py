"""A review that was merely AWAITED when its task ended is a gap, never a degradation.

The stored wave/panel keeps its fail-closed placeholder (``DEGRADED`` + open custody);
what changes is the task's own outcome: a finish over reviewers that had simply not
answered yet is not stamped ``degraded``. The loud final-message disclosure stays and
a typed fact is recorded (``execution.plan_review`` / ``review.status`` = ``awaiting``).
Every real outcome — no quorum, unresolved custody, a refused or failed slot, a critic
verdict beneath the placeholder, FAIL, a rail, a spent cap, a blocking exit — keeps
exactly the status, reason code and sentence it had.
"""
from __future__ import annotations

import json
import queue

import pytest

from ouroboros import _outcome_receipts, config, loop
from ouroboros.outcomes import _objective_axis, _review_axis, derive_loop_outcome, public_task_result
from ouroboros.owner_hurry import force_plan_decision, plan_review_disclosure, plan_wave_only_awaited
from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict, completion_status_label
from tests.test_acceptance_async_loop import ANSWER, call, full_loop, keep  # noqa: F401
from tests.test_delivery_forced_finalization import _forced_test_context
from tests.test_plan_finalization_collection import _AnswerExecutor, _sent
from tests.test_plan_review_engine import CLEAN, _call, _finding, _state
from tests.test_plan_review_engine import harness as _engine_harness
from tests.test_plan_review_event_route import _wait_until

harness = _engine_harness

AWAITED_NOTICE = (
    "\n\n⚠️ Plan review is still open (reviewer work is running or awaiting collection); work "
    "proceeded under the owner-selected advisory enforcement. A reviewer slot can still settle, "
    "so a late result is still owed."
)
NO_QUORUM_NOTICE = (
    "\n\n⚠️ Plan review is still open (DEGRADED; no parseable reviewer quorum); work proceeded "
    "under the owner-selected advisory enforcement."
)
ADVISORY_SENTENCE = "The plan review was never closed; the work went on with what the reviewers said."
AWAITED_SENTENCE = "Not every plan reviewer had answered when the task ended."


# ------------------------------------------------------------------ plan: the typed predicate

def _row(slot, state="settled", *, ok=False, failure_code="", error="boom", late=False, op=""):
    """One durable plan actor row, shaped like the recorded ones."""
    return {"slot_id": slot, "model": "m/x", "ok": ok, "error": None if ok else error,
            "operation_state": state, "late_result_pending": late, "failure_code": failure_code,
            "operation_id": op or f"op-{slot}"}


def _awaiting(slot):
    return _row(slot, "pending_dispatch", late=True,
                error="Pending dispatch; the physical review operation is in flight (window 21600s)")


def _wave(actors, *, answered=0, findings=(), custody=True, quorum=2, supplements=()):
    return {"aggregate": "DEGRADED", "closed": False, "custody_pending": custody, "cycle_index": 1,
            "actors": list(actors), "findings": list(findings),
            "counts": {"configured": len(actors), "parseable": answered, "quorum": quorum},
            "historical_supplements": list(supplements)}


_NOTE = {"finding_id": "s1:n1", "class": "note", "slot": "s1"}

ONLY_AWAITED = {
    "nobody answered yet": _wave([_awaiting("s1"), _awaiting("s2"), _awaiting("s3")]),
    "one answer below quorum, with findings": _wave(
        [_row("s1", ok=True), _awaiting("s2"), _awaiting("s3")], answered=1, findings=[_NOTE]),
    "a clean quorum held open by the last slot": _wave(
        [_row("s1", ok=True), _row("s2", ok=True), _awaiting("s3")], answered=2),
    "a quorum whose only findings are notes: notes never move the verdict": _wave(
        [_row("s1", ok=True), _row("s2", ok=True), _awaiting("s3")], answered=2, findings=[_NOTE]),
}
NOT_ONLY_AWAITED = {
    "a wait beside a real failure": _wave(
        [_row("s1", failure_code="run_failed"), _awaiting("s2"), _awaiting("s3")]),
    "a wait beside a typed $0 refusal": _wave(
        [_row("s1", "not_dispatched", failure_code="budget_refused"), _awaiting("s2"), _awaiting("s3")]),
    "a wait beside a parse failure": _wave(
        [_row("s1", error="no parseable findings"), _awaiting("s2"), _awaiting("s3")]),
    "the logical window expired (in_flight)": _wave(
        [_row("s1", "in_flight", late=True), _awaiting("s2"), _awaiting("s3")]),
    "custody lost on every slot": _wave(
        [_row(s, "custody_lost", late=True, failure_code="review_recovery_request_mismatch")
         for s in ("s1", "s2", "s3")]),
    "a settled answer nobody collected yet": _wave(
        [_awaiting("s1"), _awaiting("s2"), _awaiting("s3")],
        supplements=[{"operation_id": "op-s1", "cycle_index": 1, "operation_state": "late_settled"}]),
    "a settled real no-quorum wave": _wave(
        [_row(s, failure_code="run_failed") for s in ("s1", "s2", "s3")], custody=False),
    "a settled wave with nothing pending": _wave(
        [_row("s1", ok=True), _row("s2", ok=True), _row("s3", ok=True)], answered=3, custody=False),
    # A collected blocking finding keeps the wave open whatever the awaited slots answer.
    "one answer below quorum raised a blocking finding": _wave(
        [_row("s1", ok=True), _awaiting("s2"), _awaiting("s3")], answered=1,
        findings=[{"finding_id": "s1:b1", "class": "blocking", "slot": "s1"}]),
    "one answer below quorum asked for evidence": _wave(
        [_row("s1", ok=True), _awaiting("s2"), _awaiting("s3")], answered=1,
        findings=[{"finding_id": "s1:e1", "class": "need_evidence", "slot": "s1"}]),
    # Facts the typed record cannot vouch for are never a mere wait.
    "a roster with a row that is not a record": _wave([_awaiting("s1"), "garbage", _awaiting("s3")]),
    "a wave with no usable quorum": _wave([_awaiting("s1"), _awaiting("s2")], quorum=None),
    "a pending row that already carries an answer": _wave(
        [{**_awaiting("s1"), "parsed": {"findings": []}, "raw_text": "{}"}, _awaiting("s2")]),
    "a pending row marked ok": _wave([{**_awaiting("s1"), "ok": True}, _awaiting("s2")]),
}


@pytest.mark.parametrize("name", sorted(ONLY_AWAITED))
def test_a_wave_open_only_for_unanswered_slots_is_only_awaited(name):
    assert plan_wave_only_awaited(ONLY_AWAITED[name]) is True


@pytest.mark.parametrize("name", sorted(NOT_ONLY_AWAITED))
def test_every_other_open_wave_keeps_its_real_outcome(name):
    assert plan_wave_only_awaited(NOT_ONLY_AWAITED[name]) is False


def test_a_malformed_wave_is_never_only_awaited():
    for wave in (None, [], {}, {"custody_pending": True}, {"custody_pending": True, "actors": "s1"}):
        assert plan_wave_only_awaited(wave) is False


# ------------------------------------------------------------------ plan: the real gate

@pytest.fixture
def panel(monkeypatch):
    from ouroboros import review_custody

    executors = {f"s{i}": _AnswerExecutor(CLEAN) for i in range(1, 4)}
    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor",
                        lambda assignment, **_kw: executors[assignment.slot.slot_id])
    yield executors
    for executor in executors.values():
        executor.release.set()
    assert _wait_until(lambda: not review_custody._RELEASED_WAVES)


def _settled(harness_, count):
    return _wait_until(lambda: sum(
        " answered — " in line for line in harness_.progress) == count)  # the reviewer row family, never the wave line


@pytest.fixture
def cyber():
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline("cyber_pro")
    yield
    config.reset_runtime_mode_baseline_for_tests()


@pytest.mark.parametrize("answered", [0, 2])
@pytest.mark.parametrize("mode", ["advisory", "hurry"])
def test_an_advisory_release_over_an_awaited_wave_carries_the_typed_fact(harness, panel, mode, answered):
    harness.state["enforcement"] = "advisory" if mode == "advisory" else "blocking"
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: _sent(panel) == 3)
    for slot in ("s1", "s2")[:answered]:
        panel[slot].release.set()
    assert _settled(harness, answered)
    if mode == "hurry":
        ctx._owner_hurry_latch = {"reason": "owner_hurry"}
    decision = force_plan_decision(ctx, {}, enforcement=harness.state["enforcement"])
    assert decision["status"] == "advisory_open" and decision["allow"] is True
    assert decision["custody_pending"] is True and decision["review_only_awaited"] is True
    # The loud owner sentence is unchanged, and the stored wave is the same fail-closed floor.
    assert plan_review_disclosure(decision) == AWAITED_NOTICE
    wave = _state(harness)["waves"][-1]
    assert (wave["aggregate"], wave["closed"], wave["custody_pending"]) == ("DEGRADED", False, True)
    assert _sent(panel) == 3  # the gate collected at $0 and sent nothing


def test_cyber_pro_reads_an_awaited_wave_the_same_way(harness, panel, cyber):
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: _sent(panel) == 3)
    decision = force_plan_decision(ctx, {}, enforcement="blocking")
    assert decision["decision_authority"] == "cyber_pro" and decision["status"] == "advisory_open"
    assert decision["review_only_awaited"] is True
    assert "review_only_awaited" not in force_plan_decision(
        ctx, {}, enforcement="blocking", hard_rail="round_limit")


def test_blocking_holds_an_awaited_wave_and_a_rail_keeps_its_own_outcome(harness, panel):
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: _sent(panel) == 3)
    held = force_plan_decision(ctx, {}, enforcement="blocking")
    assert held["allow"] is False and held["status"] == "open" and "review_only_awaited" not in held
    for enforcement in ("blocking", "advisory"):
        railed = force_plan_decision(ctx, {}, enforcement=enforcement, hard_rail="round_limit")
        assert railed["status"] == "rail_degraded" and "review_only_awaited" not in railed


def test_a_critic_verdict_beneath_the_placeholder_is_not_a_mere_wait(harness, panel):
    harness.state["enforcement"] = "advisory"
    for slot in ("s1", "s2"):
        panel[slot].answer = json.dumps([_finding("b1", "blocking", breaks="claim_1")])
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: _sent(panel) == 3)
    panel["s1"].release.set()
    panel["s2"].release.set()
    assert _settled(harness, 2)
    decision = force_plan_decision(ctx, {}, enforcement="advisory")
    assert decision["custody_pending"] is True and decision["allow"] is True
    assert "review_only_awaited" not in decision


def test_a_wait_beside_a_real_failure_is_not_a_mere_wait(harness, panel):
    harness.state["enforcement"] = "advisory"
    panel["s1"].answer = "this is not a findings document"
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: _sent(panel) == 3)
    panel["s1"].release.set()
    assert _settled(harness, 1)
    decision = force_plan_decision(ctx, {}, enforcement="advisory")
    assert decision["custody_pending"] is True and decision["allow"] is True
    assert "review_only_awaited" not in decision


def _class_outcome(tmp_path, monkeypatch, decision):
    """Finalize over a projected decision and return (outcome, the record's execution axis)."""
    _usage, _trace, outcome, record = _finalize(tmp_path, monkeypatch, decision)
    return outcome, record["outcome_axes"]["execution"]


def test_the_gate_decision_names_the_plan_review_class(harness, panel, tmp_path, monkeypatch):
    """From REAL waves through ``force_plan_decision``: an only-awaited wave carries no
    class; one answer beside two failures is ``unanswered``; three failures are
    ``none_answered``; three answers left open by REVISE_PLAN are ``answered_open``;
    a closed wave carries nothing. The class then rides ``execution.plan_review``
    while the execution status, reason code, failure and ``degraded_reason`` stay
    byte-identical to a decision without it."""
    harness.state["enforcement"] = "advisory"
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: _sent(panel) == 3)
    awaited = force_plan_decision(ctx, {}, enforcement="advisory")
    assert awaited["review_only_awaited"] is True and "plan_review_class" not in awaited
    assert (awaited["reviewers_answered"], awaited["reviewers_configured"]) == (0, 3)

    panel["s2"].answer = panel["s3"].answer = "this is not a findings document"
    for slot in ("s1", "s2", "s3"):
        panel[slot].release.set()
    assert _settled(harness, 3)
    unanswered = force_plan_decision(ctx, {}, enforcement="advisory")
    assert unanswered["plan_review_class"] == "unanswered" and "review_only_awaited" not in unanswered
    assert (unanswered["reviewers_answered"], unanswered["reviewers_configured"]) == (1, 3)

    # The record: the class rides the execution axis and nothing else moves.
    outcome, execution = _class_outcome(tmp_path / "unanswered", monkeypatch, unanswered)
    plain, plain_execution = _class_outcome(tmp_path / "plain", monkeypatch, {
        key: value for key, value in unanswered.items()
        if key not in {"plan_review_class", "reviewers_answered", "reviewers_configured"}})
    assert execution["plan_review"] == "unanswered" and "plan_review" not in plain_execution
    for key in ("status", "reason_code", "failure"):
        assert execution[key] == plain_execution[key], key
    assert (outcome["degraded"], outcome["degraded_reason"], outcome["reason_code"]) == (
        plain["degraded"], plain["degraded_reason"], plain["reason_code"]) == (True, "plan_review_advisory", "plan_review_advisory")
    assert _completion_verdict({"status": "completed", "reason_code": outcome["reason_code"],
                                "outcome_axes": outcome["outcome_axes"]}, {}) == (
        "Only some of the plan reviewers answered; the work went on with their notes.")


def test_every_other_real_wave_names_its_class(harness, panel):
    harness.state["enforcement"] = "advisory"
    for slot in ("s1", "s2", "s3"):
        panel[slot].answer = "this is not a findings document"
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: _sent(panel) == 3)
    for slot in ("s1", "s2", "s3"):
        panel[slot].release.set()
    assert _settled(harness, 3)
    none = force_plan_decision(ctx, {}, enforcement="advisory")
    assert none["plan_review_class"] == "none_answered" and none["reviewers_answered"] == 0


def test_an_open_wave_every_reviewer_answered_is_answered_open(harness, panel):
    harness.state["enforcement"] = "advisory"
    for slot in ("s1", "s2", "s3"):
        panel[slot].answer = json.dumps([_finding("b1", "blocking", breaks="claim_1")])
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: _sent(panel) == 3)
    for slot in ("s1", "s2", "s3"):
        panel[slot].release.set()
    assert _settled(harness, 3)
    decision = force_plan_decision(ctx, {}, enforcement="advisory")
    assert decision["outcome"] == "REVISE_PLAN" and decision["closed"] is False
    assert decision["plan_review_class"] == "answered_open"
    assert (decision["reviewers_answered"], decision["reviewers_configured"]) == (3, 3)


def test_a_closed_wave_carries_no_class(harness, panel):
    harness.state["enforcement"] = "advisory"
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: _sent(panel) == 3)
    for slot in ("s1", "s2", "s3"):
        panel[slot].release.set()
    assert _settled(harness, 3)
    decision = force_plan_decision(ctx, {}, enforcement="advisory")
    assert decision["closed"] is True and decision["status"] == "closed"
    assert not {"plan_review_class", "reviewers_answered"} & decision.keys()


def test_a_spent_cap_keeps_its_own_outcome_under_advisory(monkeypatch):
    """The gate's typed capacity fact outranks the wait: nothing about a spent cap changes."""
    from ouroboros import owner_hurry, task_results

    base = {"enforcement": "advisory", "status": "advisory_open", "allow": True, "closed": False,
            "outcome": "DEGRADED", "custody_pending": True, "reviewer_slots_degraded": True}
    wave = ONLY_AWAITED["nobody answered yet"]
    state = {"schema_version": 2, "waves": [wave], "current_attempt": {}}
    monkeypatch.setattr(task_results, "load_plan_review_state", lambda *_a: state)
    monkeypatch.setattr("ouroboros.tools.plan_review_collect.collect_before_gate", lambda _c, s: s)
    ctx = type("Ctx", (), {"task_id": "t", "drive_root": "/nonexistent", "task_metadata": {"force_plan": True}})()
    monkeypatch.setattr(owner_hurry, "_canonical_root", lambda _ctx: "/nonexistent")
    for extra, expected in (({}, True), ({"review_capacity_reason": "review_cycles_exhausted"}, False)):
        monkeypatch.setattr(task_results, "plan_review_gate_projection",
                            lambda *_a, _e=extra, **_k: {**base, **_e})
        decision = force_plan_decision(ctx, {}, enforcement="advisory")
        assert decision.get("review_only_awaited", False) is expected


# ------------------------------------------------------------------ plan: finalization and the record

def _decision(**facts):
    return {"required": True, "self_opened": True, "enforcement": "advisory", "status": "advisory_open",
            "allow": True, "closed": False, "outcome": "DEGRADED", "reviewer_slots_degraded": True, **facts}


def _finalize(tmp_path, monkeypatch, decision):
    _loop, tools, ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_kw: False)
    monkeypatch.setattr(loop, "_force_plan_decision", lambda *_a, **_kw: dict(decision))
    loop._replace_delivery_candidate(tools, ctx, trace, ANSWER, control="candidate")
    text, usage, trace = loop._no_tool_final_answer(
        ANSWER, ctx, trace, tools, queue.Queue(), set(), lambda _t: None)
    assert text == ANSWER
    outcome = derive_loop_outcome(text, usage, trace)
    record = public_task_result({"status": "completed", "reason_code": outcome["reason_code"],
                                 "outcome_axes": outcome["outcome_axes"]})
    return usage, trace, outcome, record


def test_a_task_that_finished_over_an_awaited_plan_wave_is_not_degraded(tmp_path, monkeypatch):
    usage, trace, outcome, record = _finalize(tmp_path, monkeypatch, _decision(
        custody_pending=True, review_late_result_pending=True, review_only_awaited=True))
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
    assert completion_status_label(warned, {}) == "Done with warnings" and _completion_verdict(warned, {}) == ""
    failed = {**record, "outcome_axes": {**record["outcome_axes"], "artifacts": {"status": "missing"}}}
    assert completion_status_label(failed, {}) == "Failed" and _completion_verdict(failed, {}) == ""


@pytest.mark.parametrize("facts, notice", [
    ({}, NO_QUORUM_NOTICE),  # a settled real no-quorum wave
    ({"custody_pending": True, "review_late_result_pending": True}, AWAITED_NOTICE),  # unresolved or mixed custody
    ({"outcome": "REVISE_PLAN", "reviewer_slots_degraded": False},
     "\n\n⚠️ Plan review is still open (REVISE_PLAN); work proceeded under the owner-selected "
     "advisory enforcement."),
    ({"status": "rail_degraded", "reason": "round_limit", "review_only_awaited": True},
     "\n\n⚠️ Plan review is open (DEGRADED; no parseable reviewer quorum); the task-wide rail "
     "`round_limit` required best-effort finalization."),
], ids=["real_no_quorum", "unresolved_or_mixed_custody", "revise_plan", "rail_degraded"])
def test_every_other_open_plan_review_keeps_todays_degraded_outcome(tmp_path, monkeypatch, facts, notice):
    decision = _decision(**facts)
    if decision["status"] == "rail_degraded":
        decision.pop("review_only_awaited")  # the gate never sets it beside a rail (asserted above)
    usage, trace, outcome, record = _finalize(tmp_path, monkeypatch, decision)
    assert usage["terminal_host_notice"] == notice.strip()
    assert trace["delivery_candidate"]["degraded"] is True
    assert trace["delivery_candidate"]["degraded_reason"] == "plan_review_advisory"
    assert outcome["degraded"] is True and outcome["reason_code"] == "plan_review_advisory"
    execution = outcome["outcome_axes"]["execution"]
    # The AWAITING fact never leaks onto a degraded task; a projected decision
    # with no wave class stamps nothing else either (the class rides a real wave).
    assert execution["status"] == "degraded" and execution.get("plan_review") != "awaiting"
    assert "plan_review" not in execution
    assert execution["failure"] == {"kind": "finalization_control", "reason_code": "plan_review_advisory"}
    assert outcome["outcome_axes"]["objective"] == {
        "status": "degraded", "source": "delivery_finalization_control", "review_status": "skipped"}
    assert completion_status_label(record, {}) == "Done with warnings"
    assert _completion_verdict(record, {}) == ADVISORY_SENTENCE


def test_blocking_terminals_keep_their_blocked_objective(tmp_path, monkeypatch):
    """The released unreachable-quorum exit and the spent cap are real degradations: the
    awaited rule never reaches a blocking decision, so both keep their blocked terminal."""
    for status, extra, source, reason in (
        ("open", {"quorum_unreachable": True}, "plan_review_quorum_unreachable", "plan_review_quorum_unreachable"),
        ("cycles_exhausted", {"cycles_paid": 2}, "plan_review_cycles_exhausted", "review_cycles_exhausted"),
    ):
        _usage, trace, outcome, record = _finalize(tmp_path / status, monkeypatch, _decision(
            enforcement="blocking", status=status, custody_pending=True, **extra))
        assert trace["delivery_candidate"]["degraded_reason"] == "plan_review_advisory"
        assert outcome["outcome_axes"]["execution"]["status"] == "degraded"
        objective = outcome["outcome_axes"]["objective"]
        assert (objective["status"], objective["source"], objective["reason"]) == ("fail", source, reason)
        assert objective["outcome_tier"] == "blocked_with_evidence"
        assert completion_status_label(record, {}) == "Failed"


def test_the_typed_fact_never_rides_a_non_clean_execution():
    """A rail owns its own reason: a stale projected decision cannot mark a forced terminal."""
    trace = {"force_plan_decision": _decision(review_only_awaited=True), "tool_calls": [],
             "delivery_candidate": {"degraded": True, "degraded_reason": "round_limit"}}
    forced = derive_loop_outcome(ANSWER, {"execution_status": "failed", "reason_code": "round_limit",
                                          "_best_effort_extracted": True}, trace)
    assert forced["outcome_axes"]["execution"]["status"] == "best_effort"
    assert "plan_review" not in forced["outcome_axes"]["execution"]
    clean = derive_loop_outcome(ANSWER, {}, {"force_plan_decision": _decision(), "tool_calls": []})
    assert "plan_review" not in clean["outcome_axes"]["execution"]


# ------------------------------------------------------------------ acceptance: the typed predicate

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
def test_an_acceptance_panel_that_was_only_awaited_is_awaiting_not_degraded(name):
    axis = _review_axis({"review_runs": [AWAITED_RUNS[name]]})
    assert axis["status"] == "awaiting" and axis["aggregate_signals"] == ["DEGRADED"]
    assert "outcome_tier" not in axis  # a minority answer is not the panel's tier
    assert _objective_axis(axis) == {"status": "not_evaluated", "source": "none", "review_status": "awaiting"}
    record = {"status": "completed", "reason_code": "final_message",
              "outcome_axes": {"execution": {"status": "ok"}, "review": axis, "objective": _objective_axis(axis)}}
    assert completion_status_label(record, {}) == "Done"


@pytest.mark.parametrize("name", sorted(REAL_RUNS))
def test_every_real_acceptance_degradation_keeps_the_degraded_axis(name):
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
    trace = {"review_runs": [AWAITED_RUNS["nobody answered yet"]], "acceptance_decision": {
        "status": "finalized_unaccepted", "reason": "review_degraded", "enforcement": "blocking"}}
    objective = _objective_axis(_review_axis(trace))
    assert (objective["status"], objective["outcome_tier"]) == ("fail", "blocked_with_evidence")


# ------------------------------------------------------------------ acceptance: the real loop

@pytest.mark.parametrize("mode", ["advisory_finish", "cyber_pro"])
def test_a_real_answer_released_over_a_running_panel_is_not_a_degraded_review(full_loop, monkeypatch, mode):  # noqa: F811
    f = full_loop
    if mode == "cyber_pro":
        monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    else:
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    f.ctx.owner_wait_callback = lambda *_a, **_kw: pytest.fail("a released answer must not park")

    def main(_llm, messages, *_a, **_kw):
        f.model_inputs.append(list(messages))
        f.model_step += 1
        if mode == "cyber_pro":
            return {"content": ANSWER}, 0.0
        if f.model_step == 1:
            return {"content": "", "tool_calls": [call("task_acceptance_review", {"claim": ANSWER}, "nominate")]}, 0.0
        assert f.entered.wait(5) and not f.release.is_set()
        control = json.loads(keep(f)["content"])
        return {"content": json.dumps({**control, "pending_review": "finish"})}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, usage, trace = f.run()
    assert result == ANSWER and not f.release.is_set()
    run = trace["review_runs"][-1]
    # The stored panel is the same fail-closed placeholder it always was.
    assert run["aggregate_signal"] == "DEGRADED" and run["degraded"] is True
    assert run["actors"][0]["operation_state"] == "pending_dispatch" and run["actors"][0]["status"] != "ok"
    assert trace["acceptance_decision"]["reason"] == "author_finish"
    axes = derive_loop_outcome(result, usage, trace)["outcome_axes"]
    assert axes["review"]["status"] == "awaiting" and axes["review"]["eligibility"] == "review_in_flight"
    assert axes["objective"]["status"] == "pass" and axes["objective"]["source"] == "author_acceptance"
    assert axes["objective"]["review_status"] == "awaiting"
    # What the owner reads: the host's decision sentence still says no reviewer signed the answer off.
    card = {"status": "completed", "reason_code": "final_message", "outcome_axes": axes}
    assert completion_status_label(card, {}) == "Done"
    assert _completion_verdict(card, {}) == TASK_CAUSE_PHRASES["author_finish"]
    record = {"status": "completed", "reason_code": "final_message", "outcome_axes": axes}
    assert completion_status_label(record, {}) == "Done"
    assert _completion_verdict(record, {}) == TASK_CAUSE_PHRASES["author_finish"]


def test_nobody_answered_beside_a_failed_reviewer_is_none_answered_even_with_an_awaited_sibling(monkeypatch):
    """A wave where one reviewer failed and another is still awaited, with no answer at all,
    is ``none_answered`` — not the legacy "the work went on with what the reviewers said";
    a wave that is ONLY awaited still carries no class (the awaited fact speaks instead)."""
    from ouroboros import owner_hurry
    from ouroboros.tools import plan_review_runtime as runtime

    def census(counts):
        base = {"answered": [], "failed": [], "skipped": [], "unresolved": [], "uncollected": [], "awaiting": [], "configured": 0}
        base.update({k: [object()] * v for k, v in counts.items() if k != "configured"})
        base["configured"] = counts["configured"]
        return base

    monkeypatch.setattr(runtime, "plan_wave_slot_census", lambda wave: census(wave))
    mixed = owner_hurry.plan_review_class_facts({"answered": 0, "failed": 1, "awaiting": 1, "configured": 2}, awaited=False)
    assert mixed["plan_review_class"] == "none_answered" and mixed["reviewers_answered"] == 0
    only_awaited = owner_hurry.plan_review_class_facts({"answered": 0, "awaiting": 2, "configured": 2}, awaited=False)
    assert "plan_review_class" not in only_awaited
    partial = owner_hurry.plan_review_class_facts({"answered": 1, "failed": 1, "awaiting": 1, "configured": 3}, awaited=False)
    assert partial["plan_review_class"] == "unanswered"


# ------------------------------------------------------------------ plan: the open set at the consumer boundary

def test_an_advisory_wave_closed_by_a_reasoned_reject_is_not_stamped_degraded(tmp_path, monkeypatch, harness):
    """The real gate over a real recorded wave: one below-quorum blocking finding, rejected
    with a rationale under advisory, closes the wave (recorded GREEN), so the finished task
    is not stamped ``degraded`` and carries no ``terminal_plan_review_open``. Quiet side:
    the same wave with an unanswered question to the author stays open and keeps today's
    ``plan_review_advisory`` stamp."""
    from ouroboros.tools import plan_review as pr

    harness.state["enforcement"] = "advisory"
    blocking = json.dumps([_finding("b1", "blocking", breaks="claim_1")])
    harness.install({"s1": blocking, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    _call(ctx)
    fp = _state(harness)["waves"][-1]["request_fingerprint"]
    open_decision = force_plan_decision(ctx, {}, enforcement="advisory")
    assert open_decision["status"] == "advisory_open" and open_decision["closed"] is False
    pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": [
        {"finding_id": "s1:b1", "decision": "reject", "rationale": "the claim is checked by the demo"}]})
    closed_decision = force_plan_decision(ctx, {}, enforcement="advisory")
    assert closed_decision["status"] == "closed" and closed_decision["outcome"] == "GREEN"
    usage, trace, outcome, _record = _finalize(tmp_path / "closed", monkeypatch, closed_decision)
    assert trace["delivery_candidate"]["degraded"] is False and outcome["degraded"] is False
    assert "terminal_plan_review_open" not in usage and "terminal_host_notice" not in usage
    # Quiet side: an undispositioned question keeps the wave open and the stamp.
    question = json.dumps([_finding("q1", "need_evidence", breaks="claim_1", summary="Why five?")])
    harness.install({"s1": question, "s2": CLEAN, "s3": CLEAN})
    asked = harness.make_ctx(task_id="task-asked")
    _call(asked)
    asked_decision = force_plan_decision(asked, {}, enforcement="advisory")
    assert asked_decision["status"] == "advisory_open" and asked_decision["outcome"] == "REVIEW_REQUIRED"
    usage, trace, outcome, _record = _finalize(tmp_path / "open", monkeypatch, asked_decision)
    assert trace["delivery_candidate"]["degraded_reason"] == "plan_review_advisory"
    assert outcome["reason_code"] == "plan_review_advisory" and usage["terminal_plan_review_open"] is True
