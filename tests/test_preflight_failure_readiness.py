"""Readiness permits a matched advisory failure without calling it fresh."""

import json
from types import SimpleNamespace

import pytest

from ouroboros.agent_task_pipeline import build_review_context
from ouroboros.review_evidence import collect_review_evidence
from ouroboros.review_state import AdvisoryRunRecord, compute_snapshot_hash, make_repo_key, update_state
from ouroboros.review_status_projection import build_review_projection
from ouroboros.tools import claude_advisory_review as advisory
from tests.test_advisory_inline_freshness import candidate  # noqa: F401


def _seed(ctx, *, phase="format", operation_state="settled"):
    update_state(ctx.drive_root, lambda state: state.add_run(AdvisoryRunRecord(
        snapshot_hash=compute_snapshot_hash(ctx.repo_dir), repo_key=make_repo_key(ctx.repo_dir),
        commit_message="candidate", status="error", ts="2026-09-06T00:00:00Z",
        raw_result="the complete failed source", execution={"failure_phase": phase, "operation_state": operation_state},
    )))


def test_status_rejoin_projection_redacts_secrets_without_mutating_the_record(candidate):  # noqa: F811
    from ouroboros.review_state import load_state

    fake_token = "ghp_" + "a" * 36
    intent = {"commit_message": "candidate", "goal": f"Verify token {fake_token}",
              "scope": "exact scope", "review_rebuttal": "evidence\n" * 500}
    update_state(candidate.drive_root, lambda state: state.add_run(AdvisoryRunRecord(
        snapshot_hash=compute_snapshot_hash(candidate.repo_dir), repo_key=make_repo_key(candidate.repo_dir),
        commit_message="candidate", status="pending", ts="2026-09-06T00:00:00Z",
        execution={"pending_invocation_id": "existing-invocation", "intent": intent},
    )))
    result = json.loads(advisory._handle_review_status(candidate))
    projected = result["advisory_runs"][0]["execution"]
    assert projected["pending_invocation_id"] == "existing-invocation"
    assert projected["intent"]["review_rebuttal"] == intent["review_rebuttal"]
    assert fake_token not in json.dumps(result) and "REDACTED" in projected["intent"]["goal"]
    assert load_state(candidate.drive_root).advisory_runs[-1].execution["intent"] == intent


@pytest.mark.parametrize("enforcement", ["advisory", "blocking"])
def test_three_readiness_callers_keep_failure_status(candidate, monkeypatch, enforcement):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    _seed(candidate)
    expected = enforcement == "advisory"
    status = json.loads(advisory._handle_review_status(candidate))
    evidence = collect_review_evidence(candidate.drive_root, repo_dir=candidate.repo_dir)
    context = build_review_context(SimpleNamespace(drive_root=candidate.drive_root, repo_dir=candidate.repo_dir))
    assert status["repo_commit_ready"] is expected
    assert status["latest_advisory_status"] == "error"
    assert status["advisory_runs"][0]["failure_phase"] == "format"
    assert evidence["current_repo"]["repo_commit_ready"] is expected
    assert evidence["current_repo"]["advisory_status"] == "error"
    assert f"repo_commit_ready={'yes' if expected else 'no'}" in context
    assert "advisory_status=error" in context
    if expected:
        assert "not PASS" in status["next_step"]


@pytest.mark.parametrize("control", ["stale", "hash_unavailable", "foreign", "pending", "authority", "admission"])
def test_unmatched_or_independent_failure_never_claims_permission(candidate, monkeypatch, control):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    _seed(candidate, phase=control if control in {"authority", "admission"} else "format",
          operation_state="in_flight" if control == "pending" else "settled")
    if control == "stale":
        (candidate.repo_dir / "change.py").write_text("value = 3\n")
    elif control == "hash_unavailable":
        monkeypatch.setattr(advisory, "compute_snapshot_hash", lambda *a, **kw: (_ for _ in ()).throw(OSError("unreadable")))
    elif control == "foreign":
        foreign = candidate.repo_dir.parent / "foreign"
        foreign.mkdir()
        (foreign / "change.py").write_text("value = 2\n")
        result = build_review_projection(candidate.drive_root, repo_dir=foreign,
                                         repo_key=make_repo_key(candidate.repo_dir),
                                         snapshot_hash_fn=lambda *a, **kw: compute_snapshot_hash(candidate.repo_dir))
        assert result["repo_commit_ready"] is False
        return
    status = json.loads(advisory._handle_review_status(candidate))
    assert status["repo_commit_ready"] is False
    assert "Advisory enforcement permits" not in status["next_step"]
