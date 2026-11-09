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


def _unrelated_paid_rows(ctx, count):
    """Other tasks' paid commit waves, through the real state writer: authority rows the bounded attempt history
    never evicts, so they crowd the free rows out (``ReviewState._trim_attempt_history``)."""
    from ouroboros.review_state import CommitAttemptRecord, _utc_now, make_repo_key, update_state

    key = make_repo_key(ctx.repo_dir)
    for index in range(count):
        row = CommitAttemptRecord(ts=_utc_now(), commit_message="other", status="succeeded", phase="commit", repo_key=key,
                                  tool_name="commit_reviewed", task_id=f"other-{index}", root_task_id=f"other-{index}",
                                  attempt=1, paid=True, pre_review_fingerprint="other-bytes")
        update_state(ctx.drive_root, lambda state, row=row: state.record_attempt(row))


def _ceiling_author(ctx, monkeypatch, failure="critical"):
    """The last allowed paid wave (cap 1) returns its outcome; returns the reviewer's sends and the first reference."""
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    ctx.branch_dev = git.run_cmd(["git", "branch", "--show-current"], cwd=ctx.repo_dir).strip()
    git.run_cmd(["git", "reset", "--hard", "HEAD"], cwd=ctx.repo_dir)
    (ctx.repo_dir / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    git.run_cmd(["git", "add", "VERSION"], cwd=ctx.repo_dir)
    git.run_cmd(["git", "commit", "-m", "fixture version"], cwd=ctx.repo_dir)
    write_task_result(ctx.drive_root, ctx.task_id, "running")
    capture_mutation_baseline(ctx.drive_root, ctx.task_id, [{"surface_type": "system_repo", "host_root": str(ctx.repo_dir)}])
    (ctx.repo_dir / "change.py").write_text("value = 2\n", encoding="utf-8")
    reviews = []
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_post_commit_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_auto_push", lambda *_a, **_kw: "")

    def reviewer(_ctx, message, **kw):
        from ouroboros.review_dispatch import invoke_review_paid_stamp
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        reviews.append(kw["review_binding_fingerprint"])
        finding = {"item": "budget", "severity": "critical", "verdict": "FAIL", "reason": "wrong amount"}
        ctx._last_review_critical_findings = [finding] if failure == "critical" else []
        ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "parsed": [finding], "raw_text": "wrong amount"}]
        if failure == "pending":  # the seat is still running: no terminal outcome exists yet
            ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "error", "error": "unavailable",
                                            "operation_id": "paid-original", "operation_state": "in_flight"}]
        if failure == "infra":  # every seat ended in a technical failure: no verdict was formed
            ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "error", "error": "provider unavailable",
                                            "operation_id": "paid-original", "operation_state": "settled"}]
            return "review unavailable", CouplingOutcome(), "infra_failure", []
        scope = CouplingOutcome(verdict="PASS", blocked=False, status="responded", critical_findings=[])
        return ("review unavailable" if failure == "pending" else None), scope, "infra_failure" if failure == "pending" else "", []
    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    first = git._repo_commit_push(ctx, "Fix amount", skip_advisory_review=True)
    assert "Review outcome returned before commit" in first, first
    return reviews, json.loads(first.split("\n", 1)[1])["review_reference"]


def _named_row(ctx, reference):
    return next((row for row in load_state(ctx.drive_root).attempts
                 if row.task_id == reference["task_id"] and row.attempt == reference["attempt"]), None)


@pytest.mark.parametrize("actor", ["own", "successor"])
@pytest.mark.parametrize("history", ["small", "evicted_after", "evicted_by_author"])
def test_a_ceiling_outcome_the_bounded_history_evicted_still_carries_the_authors_own_decision(candidate, monkeypatch, history, actor):  # noqa: F811
    """#2830, owner 9 Nov: after the last allowed review returned FAIL and the author fixed it, the next commit buys no
    panel (the paid-cycle ceiling) and returns a free outcome and reference. Under Advisory the author explicitly decides
    the CURRENT bytes with that reference. Its free hot row can be evicted by other tasks' paid waves before the decision
    (``evicted_after``) or by the author's own continuation row once it validated it (``evicted_by_author``); the settled
    ledger record it names still carries the decision: no new send, the paid outcome and the ceiling's NOT_DISPATCHED
    stay as they were, and the acting task (a recorded Continue successor too) is the one recorded as deciding, on its
    own landing row and on the record."""
    from ouroboros.owner_continue import binding_sha
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.task_store import patch_document

    ctx = candidate
    reviews, paid_reference = _ceiling_author(ctx, monkeypatch)
    paid_verdict = load_record(ledger_root(ctx), paid_reference["review_record_id"])["verdict"]
    (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")  # the author's fix of the FAIL
    if history == "evicted_by_author":
        _unrelated_paid_rows(ctx, 55)  # the next free row appended evicts the oldest free one
    ceiling = git._repo_commit_push(ctx, "Fix amount", skip_advisory_review=True)
    assert "Review outcome returned before commit" in ceiling and len(reviews) == 1, ceiling
    reference = json.loads(ceiling.split("\n", 1)[1])["review_reference"]
    record_id = reference["review_record_id"]
    assert load_record(ledger_root(ctx), record_id)["verdict"]["aggregate"] == "NOT_DISPATCHED"
    if history == "evicted_after":
        _unrelated_paid_rows(ctx, 55)
    assert (_named_row(ctx, reference) is None) is (history == "evicted_after")
    if actor == "successor":
        write_task_result(ctx.drive_root, ctx.task_id, "completed")
        write_task_result(ctx.drive_root, "successor-task", "running")
        binding = {"successor_task_id": "successor-task"}
        patch_document(ctx.drive_root, ctx.task_id, lambda current: {**current, "continued_by": {
            "successor_task_id": "successor-task", "binding": binding, "binding_sha256": binding_sha(binding),
            "state": "admitted"}})
        ctx.task_id = "successor-task"
    before = git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
    decided = {"disposition": "accepted", "rationale": "I read the FAIL, verified my fix and accept the current bytes."}
    message = ""
    if history == "evicted_after":  # the record keeps no commit message: it is named, never invented
        unnamed = git._repo_commit_push(ctx, "", review_reference=reference, author_disposition=decided)
        assert "keeps no commit message" in unnamed and git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == before
        message = "Fix amount"
    landed = git._repo_commit_push(ctx, message, review_reference=reference, author_disposition=decided)
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) != before, landed
    assert git.run_cmd(["git", "show", "HEAD:change.py"], cwd=ctx.repo_dir) == "value = 3"
    assert len(reviews) == 1, "the decision bought no panel"
    record = load_record(ledger_root(ctx), record_id)
    assert record["verdict"]["aggregate"] == "NOT_DISPATCHED" and record["task_id"] == reference["task_id"]
    decision = record["author_decision"]
    assert decision["disposition"] == "accepted" and decision["decided_by_task"] == ctx.task_id
    assert decision["review_reference"] == reference
    assert load_record(ledger_root(ctx), paid_reference["review_record_id"])["verdict"] == paid_verdict
    assert (_named_row(ctx, reference) is None) is (history != "small")
    rows = load_state(ctx.drive_root).attempts
    [own] = [row for row in rows if row.task_id == ctx.task_id and row.status == "succeeded"]
    assert own.author_disposition["review_reference"] == reference, own.author_disposition
    assert sum(row.paid for row in rows if row.task_id in {"inline-task", "successor-task"}) == 1, "no new paid wave"


def test_blocking_ignores_an_evicted_ceiling_reference_and_keeps_its_ceiling(candidate, monkeypatch):  # noqa: F811
    """The unchanged Blocking path: neither the ceiling nor the ledger grants an author override."""
    ctx = candidate
    reviews, _paid = _ceiling_author(ctx, monkeypatch)
    (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")
    ceiling = git._repo_commit_push(ctx, "Fix amount", skip_advisory_review=True)
    reference = json.loads(ceiling.split("\n", 1)[1])["review_reference"]
    _unrelated_paid_rows(ctx, 55)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    before = git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
    refused = git._repo_commit_push(ctx, "", review_reference=reference, author_disposition={
        "disposition": "accepted", "rationale": "Blocking keeps the ceiling."})
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused and "requires Advisory" in refused, refused
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == before and len(reviews) == 1


def test_a_running_review_whose_hot_row_is_gone_is_never_taken_for_a_terminal_outcome(candidate, monkeypatch):  # noqa: F811
    """Running or UNKNOWN is not a terminal technical failure: a pending record whose hot row is missing is refused, never
    recovered as a finished outcome, and nothing lands or is sent."""
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.review_state import ReviewStateMutation, update_state

    ctx = candidate
    reviews, reference = _ceiling_author(ctx, monkeypatch, failure="pending")
    assert load_record(ledger_root(ctx), reference["review_record_id"])["state"] == "pending"

    def lose(state):
        state.attempts = [row for row in state.attempts if not (row.task_id == reference["task_id"]
                                                                and row.attempt == reference["attempt"])]
        return ReviewStateMutation(None, changed=True)
    update_state(ctx.drive_root, lose)
    before = git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
    refused = git._repo_commit_push(ctx, "", review_reference=reference, author_disposition={
        "disposition": "accepted", "rationale": "Nothing terminal was returned."})
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused, refused
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == before and len(reviews) == 1


def _continue_to(ctx, successor):
    """The owner's recorded Continue from ``ctx``'s task to ``successor``, which then acts in ``ctx``."""
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_store import patch_document

    write_task_result(ctx.drive_root, ctx.task_id, "completed")
    write_task_result(ctx.drive_root, successor, "running")
    binding = {"successor_task_id": successor}
    patch_document(ctx.drive_root, ctx.task_id, lambda current: {**current, "continued_by": {
        "successor_task_id": successor, "binding": binding, "binding_sha256": binding_sha(binding), "state": "admitted"}})
    ctx.task_id = successor


@pytest.mark.parametrize("actor", ["own", "successor"])
def test_a_technical_failure_at_the_ceiling_is_decided_by_the_current_author_with_no_verdict(candidate, monkeypatch, actor):  # noqa: F811
    """Owner 9 Nov, the second agreed case: the last allowed paid wave (cap 1) ended in a technical failure and formed no
    verdict, and the ceiling refuses another wave. Under Advisory the current author (or the successor its recorded
    Continue leads to), having received the failure facts, explicitly decides the current bytes with that wave's
    reference: no panel is bought, the record keeps its own no-verdict aggregate and is never read as PASS, the decision
    names the deciding task, and Blocking still refuses the same decision."""
    from ouroboros.review_ledger import ledger_root, load_record

    ctx = candidate
    reviews, reference = _ceiling_author(ctx, monkeypatch, failure="infra")
    record = load_record(ledger_root(ctx), reference["review_record_id"])
    aggregate = record["verdict"]["aggregate"]
    assert record["state"] == "settled" and aggregate not in {"PASS", "FAIL"}, record["verdict"]
    (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")  # the author's own check and fix
    ceiling = git._repo_commit_push(ctx, "Fix amount", skip_advisory_review=True)
    assert "Review outcome returned before commit" in ceiling and len(reviews) == 1, "the cap bought nothing"
    refusal = json.loads(ceiling.split("\n", 1)[1])["review_reference"]["review_record_id"]
    assert load_record(ledger_root(ctx), refusal)["verdict"]["aggregate"] == "NOT_DISPATCHED"
    if actor == "successor":
        _continue_to(ctx, "successor-task")
    decided = {"disposition": "accepted",
               "rationale": "The review failed technically and gave no verdict; I checked the fix myself."}
    before = git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    refused = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=decided)
    assert "requires Advisory" in refused and git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == before
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=decided)
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) != before, landed
    assert git.run_cmd(["git", "show", "HEAD:change.py"], cwd=ctx.repo_dir) == "value = 3"
    assert len(reviews) == 1 and ctx._commit_review_status == "author_continued", "no panel, and no reviewer PASS"
    after = load_record(ledger_root(ctx), reference["review_record_id"])
    assert after["verdict"]["aggregate"] == aggregate and after["author_decision"]["decided_by_task"] == ctx.task_id
    rows = load_state(ctx.drive_root).attempts
    [own] = [row for row in rows if row.task_id == ctx.task_id and row.status == "succeeded"]
    assert own.author_disposition["review_reference"] == reference
    assert own.author_disposition["reviewer_signal"] == aggregate and own.author_disposition["enforcement"] == "advisory"
    assert sum(row.paid for row in rows if row.task_id in {"inline-task", "successor-task"}) == 1, "no new paid wave"
