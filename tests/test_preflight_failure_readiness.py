"""Legacy advisory rows are history: projected (secrets redacted), never commit readiness.

Decision 3A retired the advisory gate, so no advisory status, open obligation or
commit-readiness debt holds a commit; the rows a former install recorded stay visible
to ``review_status`` and the review evidence, while no surface projects a
``repo_commit_ready`` verdict from them.
"""

import json
from types import SimpleNamespace

import pytest

from ouroboros.agent_task_pipeline import build_review_context
from ouroboros.review_evidence import collect_review_evidence
from ouroboros.review_state import (
    AdvisoryRunRecord, CommitReadinessDebtItem, ObligationItem, compute_snapshot_hash, load_state, make_repo_key,
    update_state,
)
from ouroboros.tools.preflight_review import _handle_review_status
from tests.test_git_review_preflight_gate import candidate  # noqa: F401


def _seed(ctx, *, phase="format", operation_state="settled"):
    # A row a former install wrote: no writer remains, so the test files it directly.
    update_state(ctx.drive_root, lambda state: state.advisory_runs.append(AdvisoryRunRecord(
        snapshot_hash=compute_snapshot_hash(ctx.repo_dir), repo_key=make_repo_key(ctx.repo_dir),
        commit_message="candidate", status="error", ts="2026-09-06T00:00:00Z",
        raw_result="the complete failed source", execution={"failure_phase": phase, "operation_state": operation_state},
    )))


def test_status_rejoin_projection_redacts_secrets_without_mutating_the_record(candidate):  # noqa: F811
    fake_token = "ghp_" + "a" * 36
    intent = {"commit_message": "candidate", "goal": f"Verify token {fake_token}",
              "scope": "exact scope", "review_rebuttal": "evidence\n" * 500}
    update_state(candidate.drive_root, lambda state: state.advisory_runs.append(AdvisoryRunRecord(
        snapshot_hash=compute_snapshot_hash(candidate.repo_dir), repo_key=make_repo_key(candidate.repo_dir),
        commit_message="candidate", status="pending", ts="2026-09-06T00:00:00Z",
        execution={"pending_invocation_id": "existing-invocation", "intent": intent},
    )))
    result = json.loads(_handle_review_status(candidate))
    projected = result["advisory_runs"][0]["execution"]
    assert projected["pending_invocation_id"] == "existing-invocation"
    assert projected["intent"]["review_rebuttal"] == intent["review_rebuttal"]
    assert fake_token not in json.dumps(result) and "REDACTED" in projected["intent"]["goal"]
    assert load_state(candidate.drive_root).advisory_runs[-1].execution["intent"] == intent


@pytest.mark.parametrize("owed", [False, True])
@pytest.mark.parametrize("enforcement", ["advisory", "blocking"])
def test_a_failed_legacy_row_and_owed_work_are_history_never_readiness(candidate, monkeypatch, enforcement, owed):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    _seed(candidate)
    if owed:
        repo_key = make_repo_key(candidate.repo_dir)

        def findings(state):
            state.open_obligations.append(ObligationItem(
                "obl-0001", "existing_contract", "critical", "owed work", "earlier", "earlier change", repo_key=repo_key))
            state.commit_readiness_debts.append(CommitReadinessDebtItem(
                "crd-0001", "verification", "owed proof", repo_key=repo_key))

        update_state(candidate.drive_root, findings)
    status = json.loads(_handle_review_status(candidate))
    evidence = collect_review_evidence(candidate.drive_root, repo_dir=candidate.repo_dir)
    context = build_review_context(SimpleNamespace(drive_root=candidate.drive_root, repo_dir=candidate.repo_dir))
    assert status["latest_advisory_status"] == "error"
    assert status["advisory_runs"][0]["failure_phase"] == "format"
    assert evidence["current_repo"]["advisory_status"] == "error"
    assert "repo_commit_ready" not in status and "repo_commit_ready" not in evidence["current_repo"]
    assert "repo_commit_ready" not in context and "Advisory readiness" not in context
    assert ("open_obligations=1" in context) is owed and ("[crd-0001]" in context) is owed
    assert bool(status["open_obligations"]) is owed
