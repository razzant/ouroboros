"""A stored author completion reads as the author's decision, never as a rewritten review.

The task-acceptance decisions recorded before the result review replaced that machine
still render: an advisory author finish over a failed critic is the author's own pass
on the objective axis while the review axis keeps its FAIL; blocking enforcement or an
author stop never turns green.
"""


def test_current_author_completion_does_not_rewrite_critic_or_independent_failure(monkeypatch):
    from ouroboros.outcomes import _objective_axis
    from ouroboros.project_dialogue import outcome_phase
    from ouroboros.review_records import build_author_disposition
    author = build_author_disposition(disposition="partial", rationale="Corrected and verified.", subject_hash="current", reviewer_signal="FAIL", enforcement="advisory")
    review = {"status": "fail", "acceptance_decision": {"reason": "author_finish", "author_disposition": author}}
    objective = _objective_axis(review)
    assert objective["status"] == "pass" and review["status"] == "fail"
    row = {"status": "completed", "outcome_axes": {"objective": objective, "review": review, "execution": {"status": "ok"}}}
    assert outcome_phase(row, {}) == "done"
    row["outcome_axes"]["execution"]["status"] = "failed"
    assert outcome_phase(row, {}) == "error"
    author["enforcement"] = "blocking"
    assert _objective_axis(review)["status"] != "pass"
    review["acceptance_decision"].update(status="finalized_unaccepted", reason="author_stop")
    assert _objective_axis(review)["outcome_tier"] == "blocked_with_evidence"
