"""#1223/#1224 — old records of a LOCAL acceptance-preparation failure still read truthfully.

The task-acceptance machine that assembled a packet before any reviewer existed is
gone, and nothing writes a preparation incident any more. Records stored before
the fold still carry one: the acceptance decision's ``acceptance_incident`` and the
review projection's panel-less incident snapshot. These pin how the readers treat
those stored facts: the owner-facing cause states the local failure AND the rail of
that attempt, a genuine reviewer FAIL is never replaced by it, a resolved incident
says nothing, the decision projection carries it, and a delayed snapshot or replica
can neither erase a fresher warning nor resurrect a resolved one.
"""

from __future__ import annotations


def test_the_final_cause_states_the_local_failure_and_the_rail_together(tmp_path):
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict

    incident = {"incident_id": "acceptance-preparation:abc", "status": "failed",
                "stage": "preparation", "attempts": 1}
    record = {
        "status": "completed", "reason_code": "budget_exhausted",
        "outcome_axes": {
            "execution": {"status": "degraded"},
            "review": {"status": "degraded", "acceptance_decision": {
                "status": "finalized_unaccepted", "reason": "acceptance_preparation_failed",
                "acceptance_incident": incident}},
        },
    }
    verdict = _completion_verdict(record, {})
    assert TASK_CAUSE_PHRASES["acceptance_preparation_failed"][:-1] in verdict
    assert TASK_CAUSE_PHRASES["budget_exhausted"] in verdict
    assert "this preparation attempt dispatched no new reviewers" in verdict
    assert "rework" not in verdict and "no reviewer ever saw" not in verdict


def test_a_genuine_reviewer_fail_is_not_replaced_by_the_incident(tmp_path):
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict

    record = {
        "status": "completed",
        "outcome_axes": {"review": {"acceptance_decision": {
            "status": "finalized_unaccepted", "reason": "reviewer_fail_no_capsule",
            "acceptance_incident": {"incident_id": "acceptance-preparation:abc",
                                    "status": "failed", "attempts": 1}}}},
    }
    verdict = _completion_verdict(record, {})
    assert TASK_CAUSE_PHRASES["reviewer_fail_no_capsule"][:-1] in verdict
    assert TASK_CAUSE_PHRASES["acceptance_preparation_failed"] in verdict


def test_a_resolved_incident_states_nothing_beside_the_answer(tmp_path):
    from ouroboros.project_dialogue import _completion_verdict

    record = {
        "status": "completed",
        "outcome_axes": {"review": {"acceptance_decision": {
            "status": "finalized_unaccepted", "reason": "author_finish",
            "acceptance_incident": {"incident_id": "acceptance-preparation:abc",
                                    "status": "resolved", "attempts": 1}}}},
    }
    assert "could not assemble" not in _completion_verdict(record, {})


def test_the_incident_rides_the_existing_projections():
    from ouroboros.review_projection import acceptance_decision_projection

    # The incident exactly as a pre-fold decision stored it.
    incident = {"incident_id": "acceptance-preparation:abc", "status": "failed", "stage": "preparation",
                "attempts": 1, "source_known": True, "feedback_delivered": False,
                "failure_kind": "RuntimeError", "failure_detail": "builder exploded"}
    projection = acceptance_decision_projection({
        "status": "finalized_unaccepted", "reason": "acceptance_preparation_failed",
        "acceptance_incident": incident,
    })
    assert projection["acceptance_incident"]["incident_id"] == incident["incident_id"]
    assert projection["acceptance_incident"]["attempts"] == 1
    assert projection["acceptance_incident"]["stage"] == "preparation"


def test_the_projection_merge_keeps_the_incident_of_the_current_publication_only():
    from ouroboros.task_results import merge_review_projection

    panel = {"surface": "task_acceptance", "panel_id": "p1", "publication_revision": 1}
    stale = {"panels": [panel], "acceptance_incident": {"incident_id": "acceptance-preparation:old",
                                                        "status": "failed", "attempts": 1}}
    fresh = {"panels": [{**panel, "publication_revision": 2}]}
    assert "acceptance_incident" not in merge_review_projection(stale, fresh)
    carried = {**fresh, "acceptance_incident": {"incident_id": "acceptance-preparation:new",
                                                "status": "resolved", "attempts": 1}}
    assert merge_review_projection(stale, carried)["acceptance_incident"]["incident_id"] == "acceptance-preparation:new"


def test_the_projection_merge_orders_the_incident_by_publication_in_both_directions():
    """A delayed snapshot or replica can neither erase a fresher warning nor
    resurrect an incident a newer publication resolved — with or without panels."""
    from ouroboros.task_results import merge_review_projection

    failed = {"incident_id": "acceptance-preparation:one", "status": "failed", "attempts": 1}
    resolved = {**failed, "status": "resolved"}
    fresh_warning = {"panels": [], "publication_revision": 3, "acceptance_incident": failed}
    delayed_clean = {"panels": [], "publication_revision": 2}
    merged = merge_review_projection(fresh_warning, delayed_clean)
    assert merged["acceptance_incident"] == failed and merged["publication_revision"] == 3
    assert merge_review_projection(delayed_clean, fresh_warning)["acceptance_incident"] == failed
    later_resolution = {"panels": [], "publication_revision": 4, "acceptance_incident": resolved}
    replayed_failure = {"panels": [], "publication_revision": 3, "acceptance_incident": failed}
    assert merge_review_projection(later_resolution, replayed_failure)["acceptance_incident"] == resolved
    assert merge_review_projection(replayed_failure, later_resolution)["acceptance_incident"] == resolved
    cleared = {"panels": [], "publication_revision": 5}
    assert "acceptance_incident" not in merge_review_projection(later_resolution, cleared)
    assert "acceptance_incident" not in merge_review_projection(cleared, later_resolution)
    assert merge_review_projection(cleared, later_resolution)["publication_revision"] == 5
    # An older snapshot without its own stamp orders by its newest panel.
    panel_only = {"panels": [{"surface": "task_acceptance", "panel_id": "p1", "publication_revision": 2}]}
    assert merge_review_projection(panel_only, fresh_warning)["acceptance_incident"] == failed
    older_incident_only = {"panels": [], "publication_revision": 1, "acceptance_incident": failed}
    merged = merge_review_projection(panel_only, older_incident_only)
    assert "acceptance_incident" not in merged and "publication_revision" not in merged
    assert merged["panels"] == panel_only["panels"]
    # A new task attempt starts its own revisions: its first publication is the newest.
    attempt_one = {"panels": [], "task_attempt": 1, "publication_revision": 7, "acceptance_incident": failed}
    attempt_two = {"panels": [], "task_attempt": 2, "publication_revision": 1}
    assert "acceptance_incident" not in merge_review_projection(attempt_one, attempt_two)
    assert "acceptance_incident" not in merge_review_projection(attempt_two, attempt_one)
    assert merge_review_projection(attempt_two, attempt_one)["task_attempt"] == 2
    # Legacy snapshots without any ordering keep their historical merge: the incoming one wins.
    assert "acceptance_incident" not in merge_review_projection({"panels": [], "acceptance_incident": failed}, {"panels": []})


def test_a_delayed_incident_snapshot_cannot_erase_the_fresher_warning_in_the_task_result(tmp_path):
    from ouroboros.task_results import load_task_result, write_task_result

    failed = {"incident_id": "acceptance-preparation:one", "status": "failed", "attempts": 1}
    write_task_result(tmp_path, "t", "running", review_projection={"panels": [], "publication_revision": 2,
                                                                  "acceptance_incident": failed})
    write_task_result(tmp_path, "t", "running", review_projection={"panels": [], "publication_revision": 1})
    stored = load_task_result(tmp_path, "t")["review_projection"]
    assert stored["acceptance_incident"] == failed and stored["publication_revision"] == 2
    write_task_result(tmp_path, "t", "running", review_projection={"panels": [], "publication_revision": 3})
    stored = load_task_result(tmp_path, "t")["review_projection"]
    assert "acceptance_incident" not in stored and stored["publication_revision"] == 3

