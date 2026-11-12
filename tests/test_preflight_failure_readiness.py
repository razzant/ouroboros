"""Legacy advisory rows are an archive: kept as written, never projected or commit readiness.

Decision 3A retired the advisory gate, so no advisory status, open obligation or
commit-readiness debt holds a commit; the rows a former install recorded stay in the
state file, which ``review_status`` names (``advisory_run_archive``,
``obligation_archive``), while no surface projects a ``repo_commit_ready`` verdict from
them; earlier open obligations show only as history until a commit lands.
"""

import json
from types import SimpleNamespace

import pytest

from ouroboros.agent_task_pipeline import build_review_context
from ouroboros.review_evidence import collect_review_evidence
from ouroboros.review_state import _STATE_RELPATH, compute_snapshot_hash, load_state, make_repo_key, update_state
from ouroboros.tools.preflight_review import _handle_review_status
from tests.test_git_review_preflight_gate import candidate  # noqa: F401


def _seed(ctx, *, phase="format", operation_state="settled"):
    # A row a former install wrote: no writer remains, so the test files it directly.
    update_state(ctx.drive_root, lambda state: state.advisory_runs.append({
        "snapshot_hash": compute_snapshot_hash(ctx.repo_dir), "repo_key": make_repo_key(ctx.repo_dir),
        "commit_message": "candidate", "status": "error", "ts": "2026-09-06T00:00:00Z",
        "raw_result": "the complete failed source",
        "execution": {"failure_phase": phase, "operation_state": operation_state},
    }))


def test_status_names_the_archive_without_projecting_a_rows_secret_bearing_content(candidate):  # noqa: F811
    fake_token = "ghp_" + "a" * 36
    intent = {"commit_message": "candidate", "goal": f"Verify token {fake_token}",
              "scope": "exact scope", "review_rebuttal": "evidence\n" * 500}
    row = {"snapshot_hash": compute_snapshot_hash(candidate.repo_dir), "repo_key": make_repo_key(candidate.repo_dir),
           "commit_message": "candidate", "status": "pending", "ts": "2026-09-06T00:00:00Z",
           "execution": {"pending_invocation_id": "existing-invocation", "intent": intent}}
    update_state(candidate.drive_root, lambda state: state.advisory_runs.append(row))
    result = json.loads(_handle_review_status(candidate))
    assert result["advisory_run_archive"]["advisory_runs"] == 1
    text = json.dumps(result)
    assert fake_token not in text and "existing-invocation" not in text and "exact scope" not in text
    assert load_state(candidate.drive_root).advisory_runs == [row]


@pytest.mark.parametrize(("enforcement", "owed"), [("advisory", False), ("blocking", True)])
def test_a_failed_legacy_row_and_owed_work_are_history_never_readiness(candidate, monkeypatch, enforcement, owed):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    _seed(candidate)
    if owed:  # rows an install that had Blocking left in its state file
        repo_key = make_repo_key(candidate.repo_dir)
        path = candidate.drive_root / _STATE_RELPATH
        data = json.loads(path.read_text(encoding="utf-8"))
        data["open_obligations"] = [{"obligation_id": "obl-0001", "item": "existing_contract", "severity": "critical",
                                     "reason": "owed work", "source_attempt_ts": "earlier",
                                     "source_attempt_msg": "earlier change", "status": "still_open", "repo_key": repo_key}]
        data["commit_readiness_debts"] = [{"debt_id": "crd-0001", "category": "verification", "summary": "owed proof",
                                           "repo_key": repo_key}]
        path.write_text(json.dumps(data), encoding="utf-8")
    status = json.loads(_handle_review_status(candidate))
    evidence = collect_review_evidence(candidate.drive_root, repo_dir=candidate.repo_dir)
    context = build_review_context(SimpleNamespace(drive_root=candidate.drive_root, repo_dir=candidate.repo_dir))
    assert status["latest_advisory_status"] == "none" and "advisory_runs" not in status
    assert status["advisory_run_archive"]["advisory_runs"] == 1
    assert "advisory_status" not in evidence["current_repo"] and "failure_phase" not in json.dumps(evidence)
    assert "repo_commit_ready" not in status and "repo_commit_ready" not in evidence["current_repo"]
    assert "repo_commit_ready" not in context and "Advisory readiness" not in context
    # Owed work shows as history until a commit lands; a debt is projected nowhere.
    assert ("earlier_open_obligations=1" in context) is owed and "crd-0001" not in context
    for retired in ("open_obligations", "open_obligations_count", "commit_readiness_debts", "retry_anchor"):
        assert retired not in status and retired not in evidence, retired
    assert ("obligation_archive" in status) is owed
    if owed:
        assert (status["obligation_archive"]["open_obligations"], status["obligation_archive"]["commit_readiness_debts"]) == (1, 1)
