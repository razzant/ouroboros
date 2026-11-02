"""The cause line names the cause the record actually holds (owner item, spam B).

Rows written before #1583 carry ``delegated_custody_unreconciled`` as their
reason_code (and objective warning) whenever a delegated run was unreconciled at
the terminal. The debt then heals from the WRITE side, and ``docs/ARCHITECTURE.md``
(the stored ``delegated_runs_unreconciled`` projection is healed only from the
write side) forbids that refresh rewriting reason_code - so the stored code outlives
the fact. Nine of fourteen terminal rows in the audited window named a debt the same
record showed as empty, and one repeated it twenty-one minutes later through the
project summary, which shares this renderer. New rows keep their own cause; these
renderers read every row by its current list.

New module rather than tests/test_terminal_truth_projection.py: that file is 986
lines and would cross the 1000-line target here.
"""

from __future__ import annotations

import pytest

from ouroboros.outcomes import WARN_DELEGATED_CUSTODY_UNRECONCILED
from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict

# The one owner sentence for the debt; the raw code stays typed on the row.
CUSTODY_SENTENCE = TASK_CAUSE_PHRASES[WARN_DELEGATED_CUSTODY_UNRECONCILED]


def _row(**fields) -> dict:
    row = {"status": "completed", "reason_code": WARN_DELEGATED_CUSTODY_UNRECONCILED}
    row.update(fields)
    return row


def test_a_healed_debt_yields_the_execution_reason_instead() -> None:
    healed = _row(
        delegated_runs_unreconciled=[],
        outcome_axes={"execution": {"status": "degraded", "reason_code": "tool_failure"}},
    )
    assert _completion_verdict(healed, {}) == "A tool this task used failed and nothing recovered it."


def test_an_open_debt_is_still_named() -> None:
    open_debt = _row(
        delegated_runs_unreconciled=["run-a1"],
        outcome_axes={"execution": {"status": "ok"}},
    )
    assert _completion_verdict(open_debt, {}) == CUSTODY_SENTENCE


def test_both_real_are_stated_in_one_line_separated_by_a_middle_dot() -> None:
    both = _row(
        delegated_runs_unreconciled=["run-a1", "run-b2"],
        outcome_axes={"execution": {"status": "failed", "reason_code": "provider_unavailable"}},
    )
    assert _completion_verdict(both, {}) == f"The task could not get a usable model answer, so it could not finish · {CUSTODY_SENTENCE}"


def test_a_healed_debt_with_no_execution_cause_states_nothing() -> None:
    """The row earned no cause of its own and no longer owes anything: inventing
    a Reason line here is exactly the false statement this closes."""
    silent = _row(delegated_runs_unreconciled=[], outcome_axes={"execution": {"status": "ok"}})
    assert _completion_verdict(silent, {}) == ""


def test_the_debt_may_arrive_on_the_event_instead_of_the_result() -> None:
    """Both lifecycle writers share this renderer; the live event carries the
    same fields the stored result does."""
    event = {
        "reason_code": WARN_DELEGATED_CUSTODY_UNRECONCILED,
        "delegated_runs_unreconciled": ["run-a1"],
        "outcome_axes": {"execution": {"status": "ok"}},
    }
    assert _completion_verdict({}, event) == CUSTODY_SENTENCE


def test_every_other_reason_code_passes_through_untouched() -> None:
    plain = {"status": "failed", "reason_code": "mystery_rail_code"}
    assert _completion_verdict(plain, {}) == "mystery_rail_code."
    assert _completion_verdict({"status": "completed"}, {}) == ""


def _legacy_axes(execution: dict, *other_warnings: str) -> dict:
    """The exact objective bytes a pre-#1583 row kept: the custody stamp as its warning."""
    warnings = [WARN_DELEGATED_CUSTODY_UNRECONCILED, *other_warnings]
    return {"lifecycle": {"status": "completed"}, "execution": execution,
            "objective": {"status": "not_evaluated", "source": "none",
                          "warning": WARN_DELEGATED_CUSTODY_UNRECONCILED, "warnings": warnings}}


def test_a_healed_debt_is_never_resurrected_by_its_own_frozen_warning() -> None:
    """Once the debt list is empty the debt is gone, whatever the old axes still say (#1583).

    An older row keeps BOTH its stamped reason code and its custody warning: the
    write-side refresh may rewrite neither. Neither is licence to name the debt
    again: the headline reads the row's current list, so a known-empty list stops
    the frozen custody warning from counting and the row reads Done; a Reason line
    naming a debt the same record shows as empty is the false statement this rule
    removes. The current execution reason speaks when there is one. Every other
    warning, and the custody warning of a row carrying no list, keep their word.
    """
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE, outcome_phase

    axes = _legacy_axes({"status": "ok", "reason_code": ""})
    healed = _row(outcome_axes=axes, delegated_runs_unreconciled=[])
    assert OUTCOME_PHASE_HEADLINE[outcome_phase(healed, {})] == "Done"
    assert OUTCOME_PHASE_HEADLINE[outcome_phase({}, healed)] == "Done", "the event carries the same list"
    assert _completion_verdict(healed, {}) == ""
    assert healed["outcome_axes"] == _legacy_axes({"status": "ok", "reason_code": ""}), "history keeps its bytes"

    # While the debt is real the row names it, headline and cause agreeing.
    owed = _row(outcome_axes=axes, delegated_runs_unreconciled=["run-a1"])
    assert OUTCOME_PHASE_HEADLINE[outcome_phase(owed, {})] == "Done with warnings"
    assert _completion_verdict(owed, {}) == CUSTODY_SENTENCE

    # No list is no proof of reconciliation: the stored warning keeps its word.
    unknown = _row(outcome_axes=axes)
    assert OUTCOME_PHASE_HEADLINE[outcome_phase(unknown, {})] == "Done with warnings"

    # Only the custody warning heals; another warning the row earned still heads it.
    other = _row(outcome_axes=_legacy_axes({"status": "ok"}, "receipt_absent"), delegated_runs_unreconciled=[])
    assert OUTCOME_PHASE_HEADLINE[outcome_phase(other, {})] == "Done with warnings"
    assert _completion_verdict(other, {}) == TASK_CAUSE_PHRASES["receipt_absent"]

    # A new row keeps its own cause; its open debt alone warns, from the list.
    fresh = {"status": "completed", "reason_code": "final_message", "delegated_runs_unreconciled": ["run-a1"],
             "outcome_axes": {"execution": {"status": "ok", "reason_code": "final_message"}}}
    assert OUTCOME_PHASE_HEADLINE[outcome_phase(fresh, {})] == "Done with warnings"
    assert OUTCOME_PHASE_HEADLINE[outcome_phase({**fresh, "delegated_runs_unreconciled": []}, {})] == "Done"

    # A current execution cause is what a healed row renders when it has one.
    railed = _row(outcome_axes=_legacy_axes({"status": "failed", "reason_code": "provider_unavailable"}),
                  delegated_runs_unreconciled=[])
    assert _completion_verdict(railed, {}) == "The task could not get a usable model answer, so it could not finish."


def test_a_legacy_stamp_is_never_read_back_as_its_own_cause() -> None:
    """R1 QA: a stored older row (stamp, no typed execution cause, known-empty debt) used to read
    back through ``normalize_outcome_axes`` with the stamp as its execution reason, so the card
    named a debt the row shows as empty. The cause stays unrecorded; nothing is guessed."""
    from ouroboros.outcomes import normalize_outcome_axes, recorded_cause

    stamped = {"status": "failed", "reason_code": WARN_DELEGATED_CUSTODY_UNRECONCILED, "delegated_runs_unreconciled": [],
               "outcome_axes": {"execution": {"status": "failed"}, "lifecycle": {"status": "failed"},
                                "objective": {"warning": WARN_DELEGATED_CUSTODY_UNRECONCILED,
                                              "warnings": [WARN_DELEGATED_CUSTODY_UNRECONCILED]}}}
    axes = normalize_outcome_axes(stamped)
    assert axes["execution"] == {"status": "failed", "reason_code": ""}
    assert axes["objective"]["warning"] == WARN_DELEGATED_CUSTODY_UNRECONCILED, "raw history keeps its bytes"
    assert _completion_verdict(stamped, {}) == ""
    # A copy an earlier read already normalized carries the stamp as the execution reason: still no cause.
    sent = {**stamped, "outcome_axes": {**stamped["outcome_axes"], "execution": {
        "status": "failed", "reason_code": WARN_DELEGATED_CUSTODY_UNRECONCILED}}}
    assert recorded_cause(sent) == "" and _completion_verdict({}, sent) == ""
    owed = {**sent, "delegated_runs_unreconciled": ["run-a1"]}
    assert _completion_verdict(owed, {}) == CUSTODY_SENTENCE, "a current debt still speaks"
    # Other rows read as before: a recorded reason, or the status of a failed row without one.
    assert normalize_outcome_axes({"status": "failed", "reason_code": "provider_failure"})["execution"][
        "reason_code"] == "provider_failure"
    assert normalize_outcome_axes({"status": "failed"})["execution"]["reason_code"] == "failed"
    typed = {**stamped, "outcome_axes": {**stamped["outcome_axes"], "execution": {
        "status": "failed", "reason_code": "provider_unavailable"}}}
    assert normalize_outcome_axes(typed)["execution"]["reason_code"] == "provider_unavailable"


_DEFERRED_BESIDE_PLAN = {
    "status": "completed", "reason_code": "child_results_deferred", "terminal_plan_review_open": True,
    "outcome_axes": {"execution": {"status": "degraded", "reason_code": "child_results_deferred"},
                     "objective": {"status": "best_effort", "source": "child_result_disposition",
                                   "deferred_count": 2}},
}
_CHILD_CLAUSE = TASK_CAUSE_PHRASES["child_results_deferred"]


def test_an_open_plan_review_is_stated_beside_the_child_fact() -> None:
    """The plan fact used to survive only as chat prose once a deferred child took
    the single degraded slot; the typed flag now states it beside the child."""
    expected = f"{_CHILD_CLAUSE} · {TASK_CAUSE_PHRASES['terminal_plan_review_open']}."
    assert _completion_verdict(_DEFERRED_BESIDE_PLAN, {}) == expected
    assert _completion_verdict({}, _DEFERRED_BESIDE_PLAN) == expected
    # Both directions: without the flag the child fact stands alone, without the
    # child the plan fact stands alone, and a class outranks the bare flag.
    no_flag = {key: value for key, value in _DEFERRED_BESIDE_PLAN.items() if key != "terminal_plan_review_open"}
    assert _completion_verdict(no_flag, {}) == f"{_CHILD_CLAUSE}."
    classed = {**_DEFERRED_BESIDE_PLAN, "outcome_axes": {
        **_DEFERRED_BESIDE_PLAN["outcome_axes"],
        "execution": {"status": "degraded", "reason_code": "child_results_deferred", "plan_review": "unanswered"}}}
    assert _completion_verdict(classed, {}) == f"{_CHILD_CLAUSE} · {TASK_CAUSE_PHRASES['plan_review_unanswered']}"
    # An acceptance limitation never hides the deferred children (astra F2).
    reviewed = {**_DEFERRED_BESIDE_PLAN, "outcome_axes": {
        **_DEFERRED_BESIDE_PLAN["outcome_axes"],
        "review": {"status": "degraded", "acceptance_decision": {
            "status": "finalized_unaccepted", "reason": "review_skipped_deadline_reserve"}}}}
    assert _completion_verdict(reviewed, {}) == (
        f"{_CHILD_CLAUSE} · {TASK_CAUSE_PHRASES['review_skipped_deadline_reserve'][:-1]} · "
        f"{TASK_CAUSE_PHRASES['terminal_plan_review_open']}.")


@pytest.mark.parametrize("plan_class, sentence", [
    ("unanswered", "Only some of the plan reviewers answered; the work went on with their notes."),
    ("none_answered", "None of the plan reviewers answered; the work went on without their notes."),
    ("answered_open", "The plan reviewers answered, but the review was never closed; the work went on with their notes."),
])
def test_the_plan_review_class_picks_the_owner_sentence(plan_class, sentence) -> None:
    """One sentence per class on the record; a legacy row without a class keeps the
    general sentence, and the recorded advisory reason states its class once."""
    record = {"status": "completed", "reason_code": "plan_review_advisory",
              "outcome_axes": {"execution": {"status": "degraded", "reason_code": "plan_review_advisory",
                                             "plan_review": plan_class}}}
    assert _completion_verdict(record, {}) == sentence
    assert _completion_verdict({}, record) == sentence
    assert sentence.count("plan reviewers") == 1 and "DEGRADED" not in sentence
    legacy = {"status": "completed", "reason_code": "plan_review_advisory",
              "outcome_axes": {"execution": {"status": "degraded"}}}
    assert _completion_verdict(legacy, {}) == TASK_CAUSE_PHRASES["plan_review_advisory"]
    # The flag beside the recorded advisory reason is the same fact: stated once.
    flagged = {**legacy, "terminal_plan_review_open": True}
    assert _completion_verdict(flagged, {}) == TASK_CAUSE_PHRASES["plan_review_advisory"]
    # Beside a different primary cause the class states the standing limitation and words
    # it; the class rides the live event and the replayed row where the result-only flag
    # does not, so the flag is redundant beside it and only a record with NEITHER stays silent.
    beside = {"status": "completed", "reason_code": "budget_exhausted", "terminal_plan_review_open": True,
              "outcome_axes": {"execution": {"status": "degraded", "plan_review": plan_class}}}
    assert _completion_verdict(beside, {}) == f"{TASK_CAUSE_PHRASES['budget_exhausted']} · {sentence}"
    unflagged = {key: value for key, value in beside.items() if key != "terminal_plan_review_open"}
    assert _completion_verdict(unflagged, {}) == f"{TASK_CAUSE_PHRASES['budget_exhausted']} · {sentence}"
    neither = {"status": "completed", "reason_code": "budget_exhausted",
               "outcome_axes": {"execution": {"status": "degraded"}}}
    assert _completion_verdict(neither, {}) == f"{TASK_CAUSE_PHRASES['budget_exhausted']}."


@pytest.mark.parametrize("source, reason, sentence", [
    ("plan_review_quorum_unreachable", "plan_review_quorum_unreachable",
     "Too few plan reviewers could answer, so the work was held."),
    ("plan_review_cycles_exhausted", "review_cycles_exhausted",
     TASK_CAUSE_PHRASES["review_cycles_exhausted"]),
    ("plan_review_author_stop", "author_stop", TASK_CAUSE_PHRASES["author_stop"]),
])
def test_a_held_task_never_says_the_work_went_on(source, reason, sentence) -> None:
    """A blocking plan exit ends Failed with its objective's own reason: the card
    states the hold as its primary cause, never the recorded advisory reason's
    sentence about work that went on (the fixture's own shape, both twins)."""
    for reason_code in ("plan_review_advisory", "final_message"):
        held = {"status": "completed", "reason_code": reason_code, "terminal_plan_review_open": True,
                "outcome_axes": {"execution": {"status": "degraded", "plan_review": "unanswered"},
                                 "objective": {"status": "fail", "source": source, "reason": reason,
                                               "outcome_tier": "blocked_with_evidence"}}}
        assert _completion_verdict(held, {}) == sentence, reason_code
        assert "went on" not in _completion_verdict({}, held)
    # A Failed card whose objective is not a plan hold keeps its own execution cause.
    other = {"status": "failed", "reason_code": "provider_unavailable",
             "outcome_axes": {"execution": {"status": "failed", "reason_code": "provider_unavailable"},
                              "objective": {"status": "fail", "source": "task_acceptance_review"}}}
    assert _completion_verdict(other, {}) == "The task could not get a usable model answer, so it could not finish."


def test_the_event_and_the_row_render_one_reason_line_over_the_shared_fixture() -> None:
    """S1: one debt rule on every surface, from one source.

    ``web/tests/fixtures/outcome_phase_parity.json`` is the twin fixture the
    browser reads in ``web/tests/reason_detail.test.js``: every case that
    declares a Reason line is asserted there against ``taskReasonDetail`` and
    here against ``_completion_verdict``, so a rule that lives on only one
    surface fails on both sides of the boundary. The same record is asserted in
    both lifecycle positions, because a live ``task_done`` event and the durable
    row reach this renderer through different arguments and must never disagree:
    the custody warning is named while the record's own debt list is non-empty,
    the current execution cause stands once it is empty, and a record carrying
    no list states nothing about the debt at all.
    """
    import json
    import pathlib

    fixture = (pathlib.Path(__file__).resolve().parents[1]
               / "web" / "tests" / "fixtures" / "outcome_phase_parity.json")
    cases = json.loads(fixture.read_text(encoding="utf-8"))["cases"]
    asserted = [case for case in cases if "acceptance_clause" in case]
    assert len(asserted) >= 5, "the fixture lost its cause-line cases"
    for case in asserted:
        record, clause = case["record"], case["acceptance_clause"]
        assert _completion_verdict(record, {}) == clause, case["name"]
        assert _completion_verdict({}, record) == clause, case["name"]


@pytest.mark.parametrize("event_only", [False, True])
@pytest.mark.parametrize("prior", [
    {"status": "completed"},
    {"status": "failed"},
    {"status": "cancelled", "cancel_origin": {"actor": "owner", "reason": "old stop"}},
    {"status": "failed", "outcome_axes": {"objective": {
        "status": "fail", "source": "plan_review_gate", "reason": "plan_review_advisory"}}},
])
def test_foreign_install_reason_preserves_the_path_and_explicit_move_instruction(prior, event_only):
    detail = ("/installs/<install_A>/task_drives/t belongs to another Ouroboros install at "
              "/installs/<install_A>; move it explicitly to work on it here")
    record = {**prior, "reason_code": "foreign_install_root", "reason_detail": detail}
    assert _completion_verdict({} if event_only else record, record if event_only else {}) == detail + "."


@pytest.mark.parametrize("reason, expected", [
    ("final_message", ""), ("some_future_code", "some_future_code."),
])
def test_other_reason_codes_do_not_render_foreign_install_detail(reason, expected):
    record = {"status": "completed", "reason_code": reason, "reason_detail": "Not this task's reason."}
    assert _completion_verdict(record, {}) == expected
    assert _completion_verdict({"reason_code": "foreign_install_root"}, {}) == "foreign_install_root."
