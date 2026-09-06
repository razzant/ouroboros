"""Prepared-candidate preflight uses ledger facts, never rendered error text."""

import subprocess
import json

import pytest

from ouroboros.review_state import AdvisoryRunRecord, compute_snapshot_hash, load_state, make_repo_key, update_state
from ouroboros.tools import claude_advisory_review as advisory
from ouroboros.tools import git
from ouroboros.tools.registry import ToolContext


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    for args in (("init",), ("config", "user.name", "Test"), ("config", "user.email", "test@example.invalid")):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "change.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "change.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    (repo / "change.py").write_text("value = 2\n")
    subprocess.run(["git", "add", "change.py"], cwd=repo, check=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="inline-task", emit_progress_fn=lambda *a: None)
    monkeypatch.setattr(advisory, "advisory_review_route", lambda: "agent_session")
    monkeypatch.setattr(advisory, "advisory_slot_enabled", lambda: True)
    monkeypatch.setattr(advisory, "check_worktree_readiness", lambda *a, **kw: [])
    monkeypatch.setattr(advisory, "_release_metadata_preflight", lambda *a: None)
    monkeypatch.setattr(advisory, "_check_worktree_version_sync_shared", lambda *a: "")
    monkeypatch.setattr(git, "advisory_gate_unavailable", lambda: False)
    monkeypatch.setattr(git, "_managed_candidate_needs_proof", lambda ctx: False)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    return ctx


def _gate(ctx, **kwargs):
    return git._advisory_and_tests_gate(
        ctx, "candidate", 0,
        classification_paths=["change.py"], advisory_paths=["change.py"],
        skip_advisory_pre_review=kwargs.pop("skip_advisory_pre_review", False), skip_tests=False, **kwargs,
    )


def _record(ctx, status, **kwargs):
    record = AdvisoryRunRecord(
        snapshot_hash=compute_snapshot_hash(ctx.repo_dir, paths=["change.py"]),
        commit_message="candidate", status=status, ts="2026-09-06T00:00:00Z",
        repo_key=make_repo_key(ctx.repo_dir), task_id=ctx.task_id, **kwargs,
    )
    update_state(ctx.drive_root, lambda state: state.add_run(record))
    return record


@pytest.mark.parametrize("status", ["fresh", "error", "parse_failure"])
def test_advisory_permission_keeps_existing_obligations_and_their_acknowledgment(candidate, monkeypatch, status):
    from ouroboros.review_state import CommitReadinessDebtItem, ObligationItem

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    execution = {"operation_state": "settled", "failure_phase": "format"} if status != "fresh" else {}
    row = _record(candidate, status, raw_result="complete received review", execution=execution)
    repo_key = make_repo_key(candidate.repo_dir)

    def prior_findings(state):
        state.open_obligations.append(ObligationItem(
            "obl-0001", "existing_contract", "critical", "prior obligation marker", "earlier", "earlier change", repo_key=repo_key))
        state.commit_readiness_debts.append(CommitReadinessDebtItem(
            "crd-0001", "verification", "prior debt marker", repo_key=repo_key))

    update_state(candidate.drive_root, prior_findings)
    assert git._check_advisory_freshness(candidate, "candidate", paths=["change.py"]) is None
    events = [json.loads(line) for line in (candidate.drive_logs() / "events.jsonl").read_text().splitlines()]
    acknowledgment = [event for event in events if event.get("type") == "advisory_obligations_acknowledged"]
    assert len(acknowledgment) == 1
    assert acknowledgment[0]["open_obligations_count"] == 1 and acknowledgment[0]["open_debts_count"] == 1
    assert "prior obligation marker" in str(acknowledgment[0]["open_obligations"])
    assert "prior debt marker" in str(acknowledgment[0]["open_debts"])
    state = load_state(candidate.drive_root)
    assert state.advisory_runs[-1].status == status and state.advisory_runs[-1].raw_result == row.raw_result
    assert state.get_open_obligations(repo_key=repo_key) and state.get_open_commit_readiness_debts(repo_key=repo_key)
    if status != "fresh":
        assert "prior obligation marker" in str(candidate._review_advisory)
        assert "prior debt marker" in str(candidate._review_advisory)


def test_failed_advisory_acknowledgment_is_not_silently_permitted(candidate, monkeypatch):
    from ouroboros.review_state import ObligationItem

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    _record(candidate, "error", raw_result="complete failure", execution={"operation_state": "settled", "failure_phase": "delivery"})
    update_state(candidate.drive_root, lambda state: state.open_obligations.append(ObligationItem(
        "obl-0001", "existing_contract", "critical", "owed work", "earlier", "earlier change", repo_key=make_repo_key(candidate.repo_dir))))
    monkeypatch.setattr("ouroboros.utils.append_jsonl", lambda *a, **kw: False)
    result = git._check_advisory_freshness(candidate, "candidate", paths=["change.py"])
    assert "Failed advisory disclosure could not be recorded" in result
    assert "owed work" in result and "Advisory is current" not in result
    assert load_state(candidate.drive_root).advisory_runs[-1].status == "error"


def test_inline_preflight_runs_after_tests_and_preserves_rebuttal(candidate, monkeypatch):
    calls = []
    rebuttal = "The branch is unreachable because the caller validates the input."
    monkeypatch.setattr(advisory, "_run_advisory_tests", lambda ctx: calls.append("tests"))
    monkeypatch.setattr(advisory, "_auto_sync_release_metadata_if_needed", lambda *a: pytest.fail("prepared candidate must not mutate"))

    def critic(repo, message, ctx, **kwargs):
        assert calls == ["tests"]
        calls.append("critic")
        assert kwargs["options"]["review_rebuttal"] == rebuttal
        return [], "[]", "reviewer", 100

    monkeypatch.setattr(advisory, "_run_claude_advisory", critic)
    assert _gate(candidate, review_rebuttal=rebuttal) is None
    assert calls == ["tests", "critic"]
    row = load_state(candidate.drive_root).advisory_runs[-1]
    assert row.status == "fresh" and row.review_rebuttal == rebuttal
    assert _gate(candidate, review_rebuttal=rebuttal) is None
    assert calls == ["tests", "critic"]


def test_new_rebuttal_invalidates_fresh_shortcut(candidate, monkeypatch):
    _record(candidate, "fresh", raw_result="[]", review_rebuttal="previous evidence")
    calls = []
    monkeypatch.setattr(advisory, "_run_advisory_tests", lambda ctx: None)
    monkeypatch.setattr(advisory, "_run_claude_advisory", lambda *a, **kw: (calls.append(kw["options"]["review_rebuttal"]) or [], "[]", "reviewer", 1))
    assert _gate(candidate, review_rebuttal="new evidence") is None
    assert calls == ["new evidence"]


def test_deterministic_block_is_not_a_refresh_request(candidate, monkeypatch):
    _record(candidate, "preflight_blocked", reason_kind="syntax", raw_result="No fresh advisory run found for this snapshot")
    monkeypatch.setattr(git, "_handle_advisory_pre_review", lambda *a, **kw: pytest.fail("must not dispatch"))
    result = _gate(candidate)
    assert result["block_reason"] == "no_advisory"
    assert "SyntaxError" in result["message"]


@pytest.mark.parametrize("test_error", [None, "failed targeted suite"])
def test_free_replay_compensates_tests_even_when_backend_available(candidate, monkeypatch, test_error):
    old = _record(candidate, "parse_failure", raw_result="unparsed original source")
    monkeypatch.setattr(git, "_handle_advisory_pre_review", lambda *a, **kw: pytest.fail("free replay must not buy preflight"))
    calls = []
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda ctx: calls.append("tests") or test_error)
    result = _gate(candidate, free_replay=True, skip_advisory_pre_review=True)
    assert calls == ["tests"]
    assert (result is None) == (test_error is None)
    rows = load_state(candidate.drive_root).advisory_runs
    assert rows[0].raw_result == old.raw_result
    assert not any(row.status == "fresh" for row in rows)


def test_free_replay_reads_freshness_without_buying_preflight_or_implying_skip(candidate, monkeypatch):
    _record(candidate, "parse_failure", raw_result="retained failed review")
    real_check = git._check_advisory_freshness
    checks = []

    def freshness(*args, **kwargs):
        checks.append(True)
        return real_check(*args, **kwargs)

    monkeypatch.setattr(git, "_check_advisory_freshness", freshness)
    monkeypatch.setattr(git, "_handle_advisory_pre_review", lambda *a, **kw: pytest.fail("free replay cannot purchase preflight"))
    result = _gate(candidate, free_replay=True)
    assert checks == [True] and result["block_reason"] == "no_advisory"
    assert load_state(candidate.drive_root).advisory_runs[-1].status == "parse_failure"


def test_fresh_free_replay_keeps_compensating_tests_without_a_new_review(candidate, monkeypatch):
    _record(candidate, "fresh", raw_result="[]")
    calls = []
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda ctx: calls.append("tests"))
    monkeypatch.setattr(git, "_handle_advisory_pre_review", lambda *a, **kw: pytest.fail("fresh replay cannot buy preflight"))
    assert _gate(candidate, free_replay=True) is None
    assert calls == ["tests"]
    assert [row.status for row in load_state(candidate.drive_root).advisory_runs] == ["fresh"]


def test_rebuttal_reaches_real_prompt_builder(candidate, monkeypatch):
    monkeypatch.setattr(advisory, "_get_staged_diff", lambda *a, **kw: "diff")
    monkeypatch.setattr(advisory, "_get_changed_file_list", lambda *a, **kw: "M change.py")
    from ouroboros.tools.preflight_review_prompt import _build_advisory_prompt

    rebuttal = "New evidence: both callers preserve a zero chat identifier."
    prompt = _build_advisory_prompt(candidate.repo_dir, "candidate", prompt_context={"review_rebuttal": rebuttal}, governance_by_retrieval=True)
    assert rebuttal in prompt
    assert "Developer's rebuttal" in prompt
    assert "offset/limit" not in prompt
    assert "start_line/max_lines" in prompt
