"""TZ-2 C4: the host never writes an author's disposition for it.

An author finish is the author's act. The host records that act; it invents no
stance ("accepted") and derives no grade ("solved") from it. Blocking review
stays mandatory, a local preparation failure stays a failure, the plan/commit
stance envelopes keep their contract, and reviewer tiers keep their meaning.
"""
from __future__ import annotations

import pytest

from ouroboros.outcomes import _objective_axis, normalize_outcome_axes
from ouroboros.review_records import (
    build_author_disposition,
    build_author_disposition_from_mapping,
    validate_author_disposition,
)


def _record(**overrides):
    fields = {"disposition": "", "action": "finish", "rationale": "Main submitted this complete response.",
              "subject_hash": "subject-1", "reviewer_signal": "FAIL", "enforcement": "advisory",
              "source": "author_final_response"}
    fields.update(overrides)
    return build_author_disposition(**fields)


def _author_finish(record, *, review_status="fail", **decision):
    return {"status": review_status, "acceptance_decision": {
        "status": "finalized_unaccepted", "reason": "author_finish", "author_disposition": record, **decision}}


def test_an_act_alone_is_a_valid_author_record_with_no_stance():
    record = _record()
    assert record["disposition"] == "" and record["action"] == "finish"
    assert validate_author_disposition(record, subject_hash="subject-1") == record
    assert validate_author_disposition(record, subject_hash="another-subject") is None  # still subject-bound


@pytest.mark.parametrize("fields", [
    {"action": ""},  # neither a stance nor an act
    {"action": "ship"},  # an unknown act
    {"disposition": "maybe"},  # an unknown stance beside a known act
    {"subject_hash": ""},  # an unbound act
    {"rationale": " "},  # an unexplained act
])
def test_a_record_still_needs_a_known_stance_or_act_bound_and_explained(fields):
    with pytest.raises(ValueError):
        _record(**fields)
    stored = {"disposition": "", "action": "finish", "rationale": "r", "subject_hash": "h", **fields}
    assert validate_author_disposition(stored) is None


def test_stance_envelopes_of_plan_and_commit_still_require_a_disposition():
    with pytest.raises(ValueError):
        build_author_disposition_from_mapping({"disposition": "", "rationale": "r"}, subject_hash="h")
    assert build_author_disposition_from_mapping(
        {"disposition": "deferred", "rationale": "r"}, subject_hash="h")["disposition"] == "deferred"


@pytest.mark.parametrize("record", [
    _record(),  # Cyber's submitted final: the act only
    _record(disposition="partial", source="author"),  # an explicit author stance
    {k: v for k, v in _record(disposition="accepted").items() if k != "action"},  # a historical record
])
def test_author_finish_is_an_author_pass_with_no_host_tier(record):
    objective = _objective_axis(_author_finish(record))
    assert objective == {"status": "pass", "source": "author_acceptance", "review_status": "fail",
                         "reason": "author_finish"}


@pytest.mark.parametrize("review", [
    _author_finish(_record(enforcement="blocking")),  # Blocking needs a fresh critic approval
    _author_finish(_record(action="stop")),  # an honest stop is not a finish
    _author_finish(None),  # the host needs the author's own record
    _author_finish(_record(), acceptance_incident={"status": "failed", "stage": "preparation"}),
])
def test_everything_that_is_not_an_advisory_author_finish_keeps_its_own_outcome(review):
    objective = _objective_axis(review)
    assert objective["source"] != "author_acceptance"
    assert objective["status"] == "fail"


def test_reviewer_tiers_keep_their_meaning():
    clean = _objective_axis({"status": "pass", "outcome_tier": "solved",
                             "acceptance_decision": {"status": "accepted", "reason": "clean_pass"}})
    assert (clean["status"], clean["source"], clean["outcome_tier"]) == ("pass", "task_acceptance_review", "solved")


def test_normalized_history_keeps_both_new_and_old_author_finish_rows():
    review = _author_finish(_record())
    new = normalize_outcome_axes({"status": "completed", "outcome_axes": {
        "objective": _objective_axis(review), "review": review}})
    assert new["objective"]["source"] == "author_acceptance" and "outcome_tier" not in new["objective"]
    # A row written before this change keeps its stored words: history is not rewritten.
    old_objective = {**_objective_axis(review), "outcome_tier": "solved"}
    old = normalize_outcome_axes({"status": "completed", "outcome_axes": {"objective": old_objective, "review": review}})
    assert old["objective"]["status"] == "pass" and old["objective"]["outcome_tier"] == "solved"
