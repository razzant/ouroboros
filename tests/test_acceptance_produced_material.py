"""The FINAL ANSWER latch, and the per-round owner-source observation stays cheap.

The task-acceptance material capture this module pinned (``acceptance_material``:
retained producer sources, GUI receipts, the eligibility predicate) is gone with
the acceptance machine; what a task produced is now read by the result review
(``tools/review_result.produced_material``). Two behaviors it exercised remain:
the anytime FINAL ANSWER latch (``loop_messages._latch_final_answer_marker``) that
``outcomes.selected_final_answer`` falls back to, and the observation captured
every round, which must not read artifacts or git.
"""
from __future__ import annotations

from types import SimpleNamespace

from ouroboros.artifacts import store_task_artifact_bytes
from ouroboros.loop_messages import _latch_final_answer_marker
from ouroboros.outcomes import selected_final_answer


def context(tmp_path):
    return SimpleNamespace(drive_root=tmp_path, task_id="produced", repo_dir=None,
                           task_metadata={"answer_protocol": "final_answer_line"}, task_contract={})


def test_answer_replacement_and_stale_latch_select_the_latest_marker(tmp_path):
    trace = {"tool_calls": []}
    _latch_final_answer_marker(trace, "Reasoning.\nFINAL ANSWER: éclair")
    assert selected_final_answer("", trace) == "éclair"
    _latch_final_answer_marker(trace, "Revised.\nFINAL ANSWER: π")
    assert selected_final_answer("No new grounding; retaining the selected answer.", trace) == "π"
    trace["tool_calls"].append({"tool": "read_file", "status": "ok"})
    assert selected_final_answer("New grounding without a selected answer.", trace) == ""


def test_observation_does_not_capture_artifacts_or_git(tmp_path, monkeypatch):
    from ouroboros.loop_messages import prepare_acceptance_observation
    from ouroboros import artifacts
    from unittest.mock import Mock
    from ouroboros import mutation_attribution

    ctx, trace = context(tmp_path), {"tool_calls": []}
    ctx.repo_dir = tmp_path
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    store_task_artifact_bytes(tmp_path, ctx.task_id, "report.txt", b"produced report")
    collect = Mock(wraps=artifacts.collect_task_artifact_records)
    monkeypatch.setattr(artifacts, "collect_task_artifact_records", collect)
    git_capture = Mock(wraps=mutation_attribution.attributed_git_candidates)
    monkeypatch.setattr(mutation_attribution, "attributed_git_candidates", git_capture)
    for _ in range(10):
        prepare_acceptance_observation(ctx, trace, None, [])
    assert collect.call_count == 0
    assert git_capture.call_count == 0


def test_explicit_empty_answer_withdraws_previous_selection(tmp_path):
    trace = {"tool_calls": []}
    _latch_final_answer_marker(trace, "FINAL ANSWER: previous")
    assert selected_final_answer("", trace) == "previous"
    _latch_final_answer_marker(trace, "Reconsidered.\nFINAL ANSWER: ")
    assert "best_valid_final_answer" not in trace
    assert selected_final_answer("No replacement", trace) == ""
