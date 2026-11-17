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
    other = _row(outcome_axes=_legacy_axes({"status": "ok"}, "expected_output_ungrounded"),
                 delegated_runs_unreconciled=[])
    assert OUTCOME_PHASE_HEADLINE[outcome_phase(other, {})] == "Done with warnings"
    assert _completion_verdict(other, {}) == TASK_CAUSE_PHRASES["expected_output_ungrounded"]

    # A new row keeps its own cause; its open debt alone warns, from the list.
    fresh = {"status": "completed", "reason_code": "final_message", "delegated_runs_unreconciled": ["run-a1"],
             "outcome_axes": {"execution": {"status": "ok", "reason_code": "final_message"}}}
    assert OUTCOME_PHASE_HEADLINE[outcome_phase(fresh, {})] == "Done with warnings"
    assert OUTCOME_PHASE_HEADLINE[outcome_phase({**fresh, "delegated_runs_unreconciled": []}, {})] == "Done"

    # A current execution cause is what a healed row renders when it has one.
    railed = _row(outcome_axes=_legacy_axes({"status": "failed", "reason_code": "provider_unavailable"}),
                  delegated_runs_unreconciled=[])
    assert _completion_verdict(railed, {}) == "The task could not get a usable model answer, so it could not finish."


@pytest.mark.parametrize("detail", ["receipt_absent", "verification_red_unreconciled", "children_undecided"])
def test_a_detail_only_warning_keeps_done_and_a_warning_beside_it_still_warns(detail) -> None:
    """Owner, 9 Nov 2026 (audit 11.11, item 18): nothing examined the result, a failed check no later run of it
    passed and finished children without a recorded decision ask nothing of the owner. Alone such a fact keeps the
    headline Done, read from the row or its event, and still says itself in the cause line; a warning that does
    ask something, beside it, still makes the row Done with warnings."""
    from ouroboros.outcomes import DETAIL_ONLY_WARNINGS
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE, outcome_phase

    def row(*warnings):
        return {"status": "completed", "reason_code": "final_message",
                "outcome_axes": {"execution": {"status": "ok", "reason_code": "final_message"},
                                 "objective": {"status": "not_evaluated", "source": "none",
                                               "warning": warnings[0], "warnings": list(warnings)}}}

    assert detail in DETAIL_ONLY_WARNINGS
    alone = row(detail)
    assert OUTCOME_PHASE_HEADLINE[outcome_phase(alone, {})] == "Done"
    assert OUTCOME_PHASE_HEADLINE[outcome_phase({}, alone)] == "Done", "the event reads it alike"
    assert _completion_verdict(alone, {}) == TASK_CAUSE_PHRASES[detail], "the detail still says itself"
    beside = row(detail, "expected_output_ungrounded")
    assert OUTCOME_PHASE_HEADLINE[outcome_phase(beside, {})] == "Done with warnings"


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
