"""Real Git: paid Advisory criticism returns before the author's free continuation."""
import json

import pytest

from ouroboros.mutation_attribution import capture_mutation_baseline
from ouroboros.review_state import load_state
from ouroboros.task_results import write_task_result
from ouroboros.tools import git
from ouroboros.review_ledger import CouplingOutcome
from tests.test_git_review_preflight_gate import candidate  # noqa: F401


@pytest.mark.parametrize(("cap", "corrected", "failure"), [
    ("1", False, "critical"),
    ("1", True, "pending"),
    ("2", True, "scope_critical"),
    ("unlimited", False, "infra"),
])
def test_explicit_continuation_commits_current_bytes_without_repaying(candidate, monkeypatch, corrected, cap, failure):  # noqa: F811
    ctx = candidate
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", cap)
    ctx.branch_dev = git.run_cmd(["git", "branch", "--show-current"], cwd=ctx.repo_dir).strip()
    git.run_cmd(["git", "reset", "--hard", "HEAD"], cwd=ctx.repo_dir)
    (ctx.repo_dir / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    git.run_cmd(["git", "add", "VERSION"], cwd=ctx.repo_dir)
    git.run_cmd(["git", "commit", "-m", "fixture version"], cwd=ctx.repo_dir)
    write_task_result(ctx.drive_root, ctx.task_id, "running")
    capture_mutation_baseline(ctx.drive_root, ctx.task_id, [{"surface_type": "system_repo", "host_root": str(ctx.repo_dir)}])
    (ctx.repo_dir / "change.py").write_text("value = 2\n", encoding="utf-8")
    checks, reviews, pushes = [], [], []
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_kw: checks.append((ctx.repo_dir / "change.py").read_text(encoding="utf-8")))
    monkeypatch.setattr(git, "_post_commit_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_auto_push", lambda *_a, **_kw: pushes.append("mock publication") or "")
    def reviewer(_ctx, message, **kw):
        from ouroboros.review_dispatch import invoke_review_paid_stamp
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        reviews.append(kw["review_binding_fingerprint"])
        finding = {"item": "budget", "severity": "critical", "verdict": "FAIL", "reason": "wrong amount"}
        ctx._last_review_critical_findings = [finding] if failure == "critical" else []
        ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "parsed": [finding], "raw_text": "wrong amount"}]
        scope = CouplingOutcome(verdict="FAIL" if failure == "scope_critical" else "PASS", blocked=failure == "scope_critical", status="responded", critical_findings=[finding] if failure == "scope_critical" else [])
        if failure in {"infra", "pending"}:
            ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "error", "error": "unavailable",
                "operation_id": "paid-original", "operation_state": "in_flight" if failure == "pending" else "settled"}]
        if failure == "pending":
            # Received feedback beside the live seat: the seat asked the coupling
            # question answered while the critic is still running.
            ctx._last_triad_raw_results.append({
                "slot_id": "coupling", "status": "responded", "parts": ["coupling"], "raw_text": "[]",
                "answers": {"coupling": {"status": "responded", "verdict": "PASS", "findings": [],
                                         "critical": 0, "coverage": "full"}}})
        return ("review unavailable" if failure in {"infra", "pending"} else None), scope, "infra_failure" if failure in {"infra", "pending"} else "", []
    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    before = git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
    first = git._repo_commit_push(ctx, "Fix amount", skip_advisory_review=True)
    assert "Review outcome returned before commit" in first, first
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == before
    assert not pushes
    reference = json.loads(first.split("\n", 1)[1])["review_reference"]
    critic = load_state(ctx.drive_root).attempts[-1]
    assert critic.paid
    if failure in {"critical", "scope_critical"}:
        assert critic.critical_findings[0]["severity"] == "critical"
    assert critic.status == ("reviewing" if failure == "pending" else "reviewed")
    assert bool(critic.finished_ts) is (failure != "pending")
    if corrected:
        (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")
    second = git._repo_commit_push(ctx, "Fix amount", review_reference=reference,
        author_disposition={"disposition": "partial" if corrected else "rejected", "rationale": "I inspected the findings and accept this current candidate."})
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) != before, second
    assert len(reviews) == 1 and len(checks) == 2 and len(pushes) == 1
    records = load_state(ctx.drive_root).attempts
    assert records[-1].status == "succeeded", second
    assert sum(row.paid for row in records) == 1
    assert records[-1].author_disposition.get("review_reference") == reference, (ctx._author_commit_record, [r.__dict__ for r in records])
    assert records[-1].author_disposition["subject_hash"] == records[-1].pre_review_fingerprint
    assert (records[-1].pre_review_fingerprint != critic.pre_review_fingerprint) == corrected
    assert load_state(ctx.drive_root).attempts[-2].triad_raw_results == critic.triad_raw_results
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    refused = git._repo_commit_push(ctx, "Finish", review_reference=reference,
        author_disposition={"disposition": "accepted", "rationale": "Cannot override Blocking."})
    assert "requires Advisory" in refused


@pytest.fixture
def runtime_mode(request, monkeypatch):
    from ouroboros import config

    mode, enforcement = request.param
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    yield mode, enforcement
    config.reset_runtime_mode_baseline_for_tests()


@pytest.mark.parametrize("runtime_mode", [("pro", "advisory"), ("cyber_pro", "advisory"), ("cyber_pro", "blocking"),
                                          ("pro", "blocking")], indirect=True)
@pytest.mark.parametrize("outcome", ["clean", "pending"])
def test_every_nonblocking_outcome_returns_before_git(candidate, monkeypatch, runtime_mode, outcome):  # noqa: F811
    """#1741: under Advisory or Cyber Pro every panel outcome — a clean PASS and a still-running
    wave included — reaches the author before commit/tag/push; the commit is the author's
    explicit continuation over the current bytes. Ordinary Blocking keeps its one-call PASS."""
    from ouroboros import review_ledger
    from ouroboros.tools import commit_gate

    mode, enforcement = runtime_mode
    nonblocking = mode == "cyber_pro" or enforcement == "advisory"
    ctx = candidate
    ctx.branch_dev = git.run_cmd(["git", "branch", "--show-current"], cwd=ctx.repo_dir).strip()
    git.run_cmd(["git", "reset", "--hard", "HEAD"], cwd=ctx.repo_dir)
    (ctx.repo_dir / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    git.run_cmd(["git", "add", "VERSION"], cwd=ctx.repo_dir)
    git.run_cmd(["git", "commit", "-m", "fixture version"], cwd=ctx.repo_dir)
    write_task_result(ctx.drive_root, ctx.task_id, "running")
    capture_mutation_baseline(ctx.drive_root, ctx.task_id, [{"surface_type": "system_repo", "host_root": str(ctx.repo_dir)}])
    (ctx.repo_dir / "change.py").write_text("value = 2\n", encoding="utf-8")
    reviews, pushes = [], []
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_post_commit_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_auto_push", lambda *_a, **_kw: pushes.append("push") or "")

    def reviewer(_ctx, message, **kw):
        from ouroboros.review_dispatch import invoke_review_paid_stamp
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        reviews.append(kw["review_binding_fingerprint"])
        ctx._last_review_critical_findings = []
        if outcome == "pending":
            ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "error", "error": "still running",
                                            "operation_id": "paid-op", "operation_state": "in_flight",
                                            "late_result_pending": True}]
            return "review pending", None, "review_late_result_pending", []
        ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "parsed": [],
                                        "raw_text": "[] — checked the change and its callers; PASS"}]
        return None, CouplingOutcome(verdict="PASS", status="responded"), "", []
    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    head = git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
    first = git._repo_commit_push(ctx, "Change value", skip_advisory_review=True)
    if not nonblocking:
        if outcome == "clean":
            assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) != head, first
        else:
            assert "REVIEW_PENDING" in first and git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == head
        return
    assert first.startswith("Review outcome returned before commit"), first
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == head and not pushes
    reference = json.loads(first.split("\n", 1)[1])["review_reference"]
    critic = load_state(ctx.drive_root).attempts[-1]
    assert critic.paid and critic.phase == ("late_wait" if outcome == "pending" else "review_only")
    # The author corrects after reading: the old PASS must not approve the changed bytes.
    (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")
    monkeypatch.setattr(git, "_run_parallel_review", lambda *_a, **_kw: pytest.fail("continuation buys no reviewer"))
    second = git._repo_commit_push(ctx, "Change value", review_reference=reference, author_disposition={
        "disposition": "partial", "rationale": "I read the returned outcome and corrected the value."})
    if outcome == "pending" and mode != "cyber_pro":
        assert "needs received feedback" in second and git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == head
        return
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) != head, second
    assert git.run_cmd(["git", "show", "HEAD:change.py"], cwd=ctx.repo_dir).strip() == "value = 3"
    assert len(reviews) == 1 and len(pushes) == 1
    assert ctx._commit_review_status == "author_continued", "an author continuation is never the critic's PASS"
    final = load_state(ctx.drive_root).attempts[-1]
    author = final.author_disposition
    assert final.status == "succeeded" and author["review_reference"] == reference
    assert author["subject_hash"] == final.pre_review_fingerprint != critic.pre_review_fingerprint
    record = review_ledger.load_record(review_ledger.ledger_root(ctx), reference["review_record_id"]) or {}
    assert author["reviewer_signal"] == ((record.get("verdict") or {}).get("aggregate") or critic.block_reason)
    assert commit_gate.resolve_commit_review_reference(ctx, reference).pre_review_fingerprint == critic.pre_review_fingerprint


def test_no_current_test_proof_means_no_author_commit(candidate, monkeypatch):  # noqa: F811
    from ouroboros.tools.commit_gate import _record_commit_attempt
    ctx = candidate
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    git._reset_commit_review_state(ctx)
    before = git._fingerprint_staged_diff(ctx.repo_dir)
    _record_commit_attempt(ctx, "candidate", "reviewing", pre_review_fingerprint=before["fingerprint"], paid=True)
    _record_commit_attempt(ctx, "candidate", "reviewed", phase="review_only", pre_review_fingerprint=before["fingerprint"])
    source = load_state(ctx.drive_root).attempts[-1]
    reference = {"surface": "commit", **{key: getattr(source, key) for key in ("repo_key", "task_id", "tool_name", "attempt", "pre_review_fingerprint")}}
    git._reset_commit_review_state(ctx)
    ctx._author_commit_source = source
    ctx._author_commit_reference = reference
    ctx._author_commit_decision = {"disposition": "partial", "rationale": "I checked the prior review."}
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_kw: "actual current test failed")
    monkeypatch.setattr(git, "_run_parallel_review", lambda *_a, **_kw: pytest.fail("free author continuation cannot dispatch"))
    head = git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
    result = git._run_reviewed_stage_cycle(ctx, "candidate", 0, paths=["change.py"], require_release_tag=False)
    assert result["status"] == "blocked" and result["block_reason"] == "tests_preflight_blocked"
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == head
    assert sum(row.paid for row in load_state(ctx.drive_root).attempts) == 1


@pytest.mark.parametrize("signal", ["passed", "author_continued", "unknown"])
def test_evolution_receipt_preserves_actual_review_authority(tmp_path, signal):
    from types import SimpleNamespace
    from ouroboros.tools.git_evolution import _record_evolution_commit_receipt
    from supervisor import evolution_lifecycle
    from tests._evolution_state_shared import _active_transaction

    campaign, tx = _active_transaction(tmp_path)
    author = {"subject_hash": "current", "reviewer_signal": "critical_findings", "enforcement": "advisory"} if signal == "author_continued" else None
    ctx = SimpleNamespace(_commit_review_status=signal, _author_commit_record=author)
    claim = {"campaign_id": campaign["id"], "transaction_id": tx["transaction_id"], "task_id": tx["task_id"]}
    assert _record_evolution_commit_receipt(ctx, "fixture commit", 0, claim, "a" * 40) == ""
    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["triad_scope_status"] == signal
    assert stored.get("author_disposition") == author
