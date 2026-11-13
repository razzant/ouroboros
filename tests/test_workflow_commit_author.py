"""Real Git: paid criticism returns before the author's free continuation."""
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


@pytest.fixture
def runtime_mode(request, monkeypatch):
    from ouroboros import config

    mode = request.param
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    yield mode
    config.reset_runtime_mode_baseline_for_tests()


@pytest.mark.parametrize("runtime_mode", ["pro", "cyber_pro"], indirect=True)
@pytest.mark.parametrize("outcome", ["clean", "pending"])
def test_every_nonblocking_outcome_returns_before_git(candidate, monkeypatch, runtime_mode, outcome):  # noqa: F811
    """#1741: every panel outcome — a clean PASS and a still-running wave included — reaches
    the author before commit/tag/push; the commit is the author's explicit continuation over
    the current bytes. A leftover Blocking value of the retired setting changes nothing."""
    from ouroboros import review_ledger
    from ouroboros.tools import commit_gate

    mode = runtime_mode
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")  # retired: read by nothing
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
    panel (the paid-cycle ceiling) and returns a free outcome and reference. The author explicitly decides
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
    verdict, and the ceiling buys no other wave. The current author (or the successor its recorded
    Continue leads to), having received the failure facts, explicitly decides the current bytes with that wave's
    reference: no panel is bought, the record keeps its own no-verdict aggregate and is never read as PASS, and the
    decision names the deciding task."""
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


LATE_CRITICAL = json.dumps([{"item": "late bug", "severity": "critical", "verdict": "FAIL", "reason": "found after the handback"}])
DECISION = {"disposition": "accepted", "rationale": "I read the returned outcome and accept the current bytes."}


def _returned_partial(ctx, monkeypatch, *, cap="3"):
    """A paid wave returned while its critic still runs and its coupling seat has answered: the returned reference, the
    reviewer's sends. Real Git and the real handback; only the reviewers' transport is supplied."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", cap)
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
        ctx._last_review_critical_findings = []
        ctx._last_triad_raw_results = [
            {"slot_id": "critic", "status": "error", "error": "unavailable", "operation_id": "paid-original",
             "operation_state": "in_flight", "late_result_pending": True},
            {"slot_id": "coupling", "status": "responded", "parts": ["coupling"], "raw_text": "[]", "operation_id": "paid-coupling",
             "operation_state": "settled", "answers": {"coupling": {"status": "responded", "verdict": "PASS", "findings": [],
                                                                    "critical": 0, "coverage": "full"}}}]
        return "review unavailable", CouplingOutcome(verdict="PASS", status="responded"), "infra_failure", []
    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    first = git._repo_commit_push(ctx, "Fix amount", skip_advisory_review=True)
    assert first.startswith("Review outcome returned before commit"), first
    return json.loads(first.split("\n", 1)[1])["review_reference"], reviews


def _settle_late(ctx, reference, *, raw_text=LATE_CRITICAL, status="responded"):
    """The late answer through the real writers a left wave settles with: the ledger record first, then its attempt row."""
    from ouroboros import review_pause
    from ouroboros.review_ledger import ledger_root

    left = {"surface": "commit", "task_id": reference["task_id"], "record_id": reference["review_record_id"],
            "retry_key": "", "attempt": reference["attempt"], "drive_root": str(ctx.drive_root), "canonical_root": str(ledger_root(ctx))}
    seat = {"surface": "multi_model_review", "slot_id": "critic", "parts": ["change"], "model": "late-model", "status": status,
            "operation_id": "paid-original", "operation_state": "settled", "late_result_pending": False, "raw_text": raw_text}
    settled = review_pause._settle_left_record(left, [seat], final=False)
    assert settled is not None
    review_pause._settle_left_attempt(left, _named_row(ctx, reference), [seat], settled)
    return settled


def _head(ctx):
    return git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)


def _staged(ctx):
    return git.run_cmd(["git", "diff", "--cached", "--name-only"], cwd=ctx.repo_dir).strip()


def _read_back(ctx, reference, reviews, *, expect="critical"):
    """The free collection the refusal names: the index no longer holds the reviewed bytes, so the outcome reads back."""
    git.run_cmd(["git", "reset"], cwd=ctx.repo_dir)
    (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")
    count = len(reviews)
    collected = git._repo_commit_push(ctx, "", review_reference=reference)
    assert len(reviews) == count, "collection buys no reviewer"
    assert "committed" not in collected.split("\n", 1)[0] or "nothing was staged" in collected, collected
    assert expect in collected, collected
    return json.loads(collected[collected.index("\n{"):])["review_reference"]


def test_feedback_settled_after_the_reference_returned_needs_a_fresh_decision_and_no_new_review(candidate, monkeypatch):  # noqa: F811
    """#2902: a decision answers the outcome that reached its author. A reviewer's later FAIL, settled by the real
    writers, refuses that decision before anything is staged; the free collection returns it, and a new decision on the
    new reference lands the current bytes. Nothing is bought and the FAIL is never read as PASS."""
    from ouroboros.review_ledger import ledger_root, load_record

    ctx = candidate
    reference, reviews = _returned_partial(ctx, monkeypatch)
    settled = _settle_late(ctx, reference)
    assert settled["verdict"]["critical_findings"], settled["verdict"]
    head = _head(ctx)
    git.run_cmd(["git", "reset"], cwd=ctx.repo_dir)
    refused = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused and "review_reference" in refused, refused
    assert "author_disposition" in refused and "$0" in refused, refused
    assert _head(ctx) == head and not _staged(ctx), "refused before staging or Git"
    assert not load_record(ledger_root(ctx), reference["review_record_id"]).get("author_decision")
    fresh = _read_back(ctx, reference, reviews)
    assert fresh != reference and fresh["review_record_id"] == reference["review_record_id"]
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=fresh, author_disposition=DECISION)
    assert _head(ctx) != head, landed
    assert git.run_cmd(["git", "show", "HEAD:change.py"], cwd=ctx.repo_dir).strip() == "value = 3"
    assert len(reviews) == 1 and ctx._commit_review_status == "author_continued"
    record = load_record(ledger_root(ctx), reference["review_record_id"])
    assert record["verdict"]["aggregate"] == settled["verdict"]["aggregate"] != "PASS"
    assert record["author_decision"]["review_reference"] == fresh


def test_feedback_settled_after_validation_is_caught_at_the_commit_boundary(candidate, monkeypatch):  # noqa: F811
    """The race: the decision validates, staging and the tests run, and only then the late FAIL settles. The check before
    Git refuses it with the bytes preserved, unstaged and uncommitted; collecting and deciding again lands at no cost."""
    from ouroboros.review_ledger import ledger_root, load_record

    ctx = candidate
    reference, reviews = _returned_partial(ctx, monkeypatch)
    head = _head(ctx)
    tests_ran = []

    def late_during_tests(*_a, **_kw):
        tests_ran.append(_settle_late(ctx, reference))
    monkeypatch.setattr(git, "_run_review_preflight_tests", late_during_tests)
    refused = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert tests_ran, refused
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused and _head(ctx) == head, refused
    assert not _staged(ctx) and (ctx.repo_dir / "change.py").read_text(encoding="utf-8") == "value = 2\n"
    row = load_state(ctx.drive_root).attempts[-1]
    assert row.status == "blocked" and row.block_reason == "review_outcome_unread"
    assert "landing" not in (load_record(ledger_root(ctx), reference["review_record_id"]).get("author_decision") or {})
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_kw: None)
    fresh = _read_back(ctx, reference, reviews)
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=fresh, author_disposition=DECISION)
    assert _head(ctx) != head and len(reviews) == 1, landed


def test_a_decision_stays_usable_while_no_outcome_was_added(candidate, monkeypatch):  # noqa: F811
    """Accounting and custody revisions of the record are no new feedback. The returned outcome here is partial: one seat
    answered while another runs. Deciding on received feedback beside a running seat is the existing contract
    (``resolve_commit_review_reference``: ``review_outcome_received``; ``test_explicit_continuation...[pending]``), which
    this repair neither widens nor treats as terminal; a wave with no received feedback still waits."""
    from ouroboros.review_ledger import ledger_root, revise_record
    from ouroboros.review_waiting import save_wait

    ctx = candidate
    reference, reviews = _returned_partial(ctx, monkeypatch)

    def account(record):
        record["rows"] = [{**row, "usd": 0.25, "tokens_in": 7} for row in record["rows"]]
        record["cost"] = {"usd": 0.5, "unknown": False}
        return record
    revised = revise_record(ledger_root(ctx), reference["review_record_id"], account)
    saved = save_wait(ctx, reference["review_record_id"], {"seen_outcomes": ["a"], "seen_outcome_states": ["a#responded"]})
    assert saved["revision"] > revised["revision"] > 1
    head = _head(ctx)
    git.run_cmd(["git", "reset"], cwd=ctx.repo_dir)
    (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert _head(ctx) != head and len(reviews) == 1, landed
    assert load_state(ctx.drive_root).attempts[-1].author_disposition["review_reference"] == reference


def test_collecting_an_unchanged_outcome_returns_the_same_identity(candidate, monkeypatch):  # noqa: F811
    """Same-state replay: reading a settled review twice names one identity, and its reference still collects (the
    exact-identity check of a collection is not disturbed by the delivery identity it carries)."""
    ctx = candidate
    reference, reviews = _returned_partial(ctx, monkeypatch)
    _settle_late(ctx, reference)
    first = _read_back(ctx, reference, reviews)
    second = _read_back(ctx, first, reviews)
    assert first == second


def test_a_reference_without_a_delivery_identity_recovers_at_no_cost(candidate, monkeypatch):  # noqa: F811
    """A reference minted before deliveries were identified proves nothing about what its author read: it is neither
    trusted nor dead. Its outcome reads back free, and the decision proceeds on the reference that read returns."""
    ctx = candidate
    reference, reviews = _returned_partial(ctx, monkeypatch)
    legacy = {key: value for key, value in reference.items() if key != "shown_outcomes"}
    assert legacy != reference
    head = _head(ctx)
    git.run_cmd(["git", "reset"], cwd=ctx.repo_dir)
    refused = git._repo_commit_push(ctx, "Fix amount", review_reference=legacy, author_disposition=DECISION)
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused and _head(ctx) == head and not _staged(ctx), refused
    fresh = _read_back(ctx, legacy, reviews, expect="coupling")
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=fresh, author_disposition=DECISION)
    assert _head(ctx) != head and len(reviews) == 1, landed


def test_a_legacy_not_dispatched_reference_has_nothing_unread(candidate, monkeypatch):  # noqa: F811
    """A record that dispatched nothing can have no late outcome and offers nothing to collect, so an older reference to
    it keeps the author's decision the owner chose on 9 November."""
    ctx = candidate
    reviews, _paid = _ceiling_author(ctx, monkeypatch)
    (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")
    ceiling = git._repo_commit_push(ctx, "Fix amount", skip_advisory_review=True)
    reference = json.loads(ceiling.split("\n", 1)[1])["review_reference"]
    reference.pop("shown_outcomes", None)
    head = _head(ctx)
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert _head(ctx) != head and len(reviews) == 1, landed


def test_a_continue_successor_collects_late_feedback_and_decides_itself(candidate, monkeypatch):  # noqa: F811
    from ouroboros.review_ledger import ledger_root, load_record

    ctx = candidate
    reference, reviews = _returned_partial(ctx, monkeypatch)
    _settle_late(ctx, reference)
    _continue_to(ctx, "successor-task")
    head = _head(ctx)
    git.run_cmd(["git", "reset"], cwd=ctx.repo_dir)
    refused = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused and _head(ctx) == head, refused
    fresh = _read_back(ctx, reference, reviews)
    assert fresh["task_id"] == reference["task_id"] != ctx.task_id, "the original reference stays canonical"
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=fresh, author_disposition=DECISION)
    assert _head(ctx) != head, landed
    decision = load_record(ledger_root(ctx), reference["review_record_id"])["author_decision"]
    assert decision["decided_by_task"] == "successor-task" and decision["review_reference"] == fresh


def test_a_late_technical_failure_is_new_feedback_and_stays_decidable(candidate, monkeypatch):  # noqa: F811
    """A seat that closes without a verdict is still an outcome the author has not read. Once read, the author's explicit
    decision (owner 9 Nov) proceeds with no verdict and no PASS."""
    ctx = candidate
    reference, reviews = _returned_partial(ctx, monkeypatch)
    settled = _settle_late(ctx, reference, raw_text="", status="error")
    assert settled["verdict"]["aggregate"] != "PASS"
    head = _head(ctx)
    git.run_cmd(["git", "reset"], cwd=ctx.repo_dir)
    refused = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused and _head(ctx) == head, refused
    fresh = _read_back(ctx, reference, reviews, expect="rows")
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=fresh, author_disposition=DECISION)
    assert _head(ctx) != head and len(reviews) == 1 and ctx._commit_review_status == "author_continued", landed


@pytest.mark.parametrize("runtime_mode", ["cyber_pro"], indirect=True)
def test_cyber_pro_keeps_its_judgment_and_states_the_unread_outcome(candidate, monkeypatch, runtime_mode):  # noqa: F811
    """No internal veto for Cyber Pro: the decision lands, the result and row say it did not read the later outcome."""
    ctx = candidate
    reference, reviews = _returned_partial(ctx, monkeypatch)
    _settle_late(ctx, reference)
    head = _head(ctx)
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert _head(ctx) != head and len(reviews) == 1, landed
    assert "did not read it" in landed, landed
    final = load_state(ctx.drive_root).attempts[-1]
    assert any("did not read it" in str(item) for item in final.advisory_findings)
    assert final.author_disposition["evidence_at_decision"]["unread_outcomes"] is True, final.author_disposition
    from ouroboros.review_ledger import ledger_root, load_record
    assert load_record(ledger_root(ctx), reference["review_record_id"])["author_decision"]["evidence_at_decision"]["unread_outcomes"] is True


@pytest.mark.parametrize("runtime_mode", ["cyber_pro"], indirect=True)
def test_cyber_pro_evidence_names_only_a_real_gap_including_one_opened_by_the_tests(candidate, monkeypatch, runtime_mode):  # noqa: F811
    from ouroboros.review_ledger import ledger_root, load_record

    ctx = candidate
    reference, _reviews = _returned_partial(ctx, monkeypatch)
    late = []
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_kw: late.append(_settle_late(ctx, reference)))
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert late and "did not read" in landed, landed
    decided = load_record(ledger_root(ctx), reference["review_record_id"])["author_decision"]
    assert decided["evidence_at_decision"]["unread_outcomes"] is True, decided
    assert load_state(ctx.drive_root).attempts[-1].author_disposition["evidence_at_decision"]["unread_outcomes"] is True


@pytest.mark.parametrize("runtime_mode", ["cyber_pro"], indirect=True)
def test_cyber_pro_evidence_claims_no_gap_when_the_outcome_was_returned(candidate, monkeypatch, runtime_mode):  # noqa: F811
    ctx = candidate
    reference, _reviews = _returned_partial(ctx, monkeypatch)
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert "did not read" not in landed, landed
    assert "unread_outcomes" not in load_state(ctx.drive_root).attempts[-1].author_disposition["evidence_at_decision"]


def test_a_status_reference_has_shown_nothing_and_asks_for_the_free_read(candidate, monkeypatch):  # noqa: F811
    from ouroboros.tools.preflight_review import _handle_review_status

    ctx = candidate
    reference, reviews = _returned_partial(ctx, monkeypatch)
    projected = json.loads(_handle_review_status(ctx))
    assert "shown_outcomes" not in projected["review_reference"] and "no author_disposition" in projected["next_step"]
    head = _head(ctx)
    git.run_cmd(["git", "reset"], cwd=ctx.repo_dir)
    refused = git._repo_commit_push(ctx, "Fix amount", review_reference=projected["review_reference"], author_disposition=DECISION)
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused and _head(ctx) == head, refused
    assert json.loads(_handle_review_status(ctx))["review_reference"]["attempt"] == reference["attempt"] and len(reviews) == 1


def test_a_seat_settled_on_the_record_ahead_of_its_row_is_not_named_as_shown(candidate, monkeypatch):  # noqa: F811
    """Writers settle the record first: a handback in that gap renders the seat as running, so its reference must not
    name the seat's answer."""
    from ouroboros import review_pause
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.review_waiting import shown_outcomes
    from ouroboros.tools.review_collection import shown_field

    ctx = candidate
    reference, _reviews = _returned_partial(ctx, monkeypatch)
    left = {"task_id": reference["task_id"], "record_id": reference["review_record_id"], "retry_key": "",
            "drive_root": str(ctx.drive_root), "canonical_root": str(ledger_root(ctx))}
    seat = {"slot_id": "critic", "parts": ["change"], "model": "late-model", "status": "responded", "operation_id": "paid-original",
            "operation_state": "settled", "late_result_pending": False, "raw_text": LATE_CRITICAL}
    assert review_pause._settle_left_record(left, [seat], final=False)
    row = _named_row(ctx, reference)
    assert row.late_result_pending, "the row still shows the seat running"
    shown = shown_field(ctx, row)
    assert shown == {"shown_outcomes": reference["shown_outcomes"]}
    assert shown["shown_outcomes"] != shown_outcomes(load_record(ledger_root(ctx), reference["review_record_id"]))


def _handback_again(ctx, reference):
    """The normal pending handback, which renders the attempt row as it stands."""
    from ouroboros.tools import commit_gate

    bound = {"fingerprint": reference["pre_review_fingerprint"]}
    ctx._current_review_attempt_number, ctx._current_review_record_id = reference["attempt"], reference["review_record_id"]
    return commit_gate._return_commit_feedback(ctx, "Fix amount", 0.0, bound, bound, pending=True)["review_reference"]


@pytest.mark.parametrize("late", ["answered_after_failure", "continued_seat"])
def test_a_row_showing_the_old_outcome_never_certifies_the_new_one_on_its_record(candidate, monkeypatch, late):  # noqa: F811
    """Writers settle the record before the row. In that gap the row can show a seat closed with its OLD outcome while the
    record already holds a NEW one (an answer after a technical failure, or a continued seat's successor): nothing about
    the seat is pending, and only the rendered operation and outcome tell that the handback did not carry the new one."""
    from ouroboros import review_ledger as ledger
    from ouroboros.review_waiting import shown_outcomes
    from ouroboros.tools import commit_gate

    ctx = candidate
    root = ledger.ledger_root(ctx)
    if late == "answered_after_failure":
        _reviews, reference = _ceiling_author(ctx, monkeypatch, failure="infra")
        row = _named_row(ctx, reference)
        ctx._pending_review_attempt, ctx._current_review_retry_key = row, row.review_retry_key
        ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "raw_text": LATE_CRITICAL, "operation_id": "paid-original",
                                        "operation_state": "settled", "parsed": json.loads(LATE_CRITICAL)}]
        commit_gate.settle_commit_review_ledger(ctx, "Fix amount", pre_fingerprint={"fingerprint": reference["pre_review_fingerprint"]},
                                                blocked=True, block_reason="critical_findings", combined_findings=json.loads(LATE_CRITICAL))
        seat_id, shown_operation = "critic", "paid-original"
    else:
        reference, _reviews = _returned_partial(ctx, monkeypatch)

        def reopen(record):  # the ledger write of an explicit continuation (``review_continuation``): a successor on the seat
            record["rows"] = [{**seat, "status": "pending", "operation_state": "in_flight", "operation_id": "paid-coupling-2",
                               "superseded_operation_ids": ["paid-coupling"]} if seat["seat_id"] == "coupling" else seat
                              for seat in record["rows"]]
            return record
        ledger.revise_record(root, reference["review_record_id"], reopen)
        ledger.settle_from_collected(root, reference["review_record_id"], [{
            "slot_id": "coupling", "status": "responded", "operation_id": "paid-coupling-2", "parts": ["coupling"], "raw_text": LATE_CRITICAL,
            "answers": {"coupling": {"status": "responded", "verdict": "FAIL", "findings": json.loads(LATE_CRITICAL), "critical": 1, "coverage": "full"}}}])
        seat_id, shown_operation = "coupling", "paid-coupling"
    current = ledger.load_record(root, reference["review_record_id"])
    seat = next(item for item in current["rows"] if item["seat_id"] == seat_id)
    assert seat["status"] == "responded" and not ledger.seat_open(seat), seat
    shown = next(item for item in _named_row(ctx, reference).triad_raw_results if item["slot_id"] == seat_id)
    assert shown["operation_id"] == shown_operation and shown.get("operation_state") == "settled", "the row shows it closed, with the old outcome"
    again = _handback_again(ctx, reference)
    assert again["shown_outcomes"] != shown_outcomes(current), "the handback rendered only the old outcome"
    head = _head(ctx)
    git.run_cmd(["git", "reset"], cwd=ctx.repo_dir)
    refused = git._repo_commit_push(ctx, "Fix amount", review_reference=again, author_disposition=DECISION)
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused and _head(ctx) == head, refused


def test_a_refused_landing_keeps_a_managed_merge_and_its_index(candidate, monkeypatch):  # noqa: F811
    """At the commit boundary a managed merge stays exactly resolved and retryable: its index, MERGE_HEAD and files are
    untouched, and only its transaction phase returns to resolution. Real Git merge state; the transaction reader is
    ``restore_assisted_resolution_after_commit_error``'s own, supplied here."""
    from ouroboros.tools import commit_gate

    ctx = candidate
    reference, _reviews = _returned_partial(ctx, monkeypatch)
    _settle_late(ctx, reference)
    repo = ctx.repo_dir
    git.run_cmd(["git", "reset"], cwd=repo)
    base = git.run_cmd(["git", "branch", "--show-current"], cwd=repo).strip()
    git.run_cmd(["git", "checkout", "-q", "-b", "update-target"], cwd=repo)
    (repo / "other.txt").write_text("merged\n", encoding="utf-8")
    git.run_cmd(["git", "add", "other.txt"], cwd=repo)
    git.run_cmd(["git", "commit", "-q", "-m", "target"], cwd=repo)
    git.run_cmd(["git", "checkout", "-q", base], cwd=repo)
    git.run_cmd(["git", "merge", "--no-commit", "--no-ff", "update-target"], cwd=repo)
    merge_head, index, head = (git.run_cmd(["git", "rev-parse", "MERGE_HEAD"], cwd=repo), git.run_cmd(["git", "write-tree"], cwd=repo), _head(ctx))
    ctx._author_commit_source = commit_gate.resolve_commit_review_reference(ctx, reference)
    ctx._author_commit_reference = reference
    restored = []
    monkeypatch.setattr("supervisor.update_merge.restore_assisted_resolution_after_commit_error", lambda tx: restored.append(tx) or True)
    tx = {"task_id": ctx.task_id, "phase": "committing_assisted"}
    sha, refusal = git._commit_reviewed_candidate(ctx, "Merge update", 0.0, tx)
    assert sha == "" and "REVIEW_AUTHOR_CONTINUATION_INVALID" in refusal and restored == [tx], refusal
    assert (git.run_cmd(["git", "rev-parse", "MERGE_HEAD"], cwd=repo), git.run_cmd(["git", "write-tree"], cwd=repo), _head(ctx)) == (merge_head, index, head)
    assert (repo / "other.txt").read_text(encoding="utf-8") == "merged\n" and (repo / "change.py").read_text(encoding="utf-8") == "value = 2\n"
    assert load_state(ctx.drive_root).attempts[-1].block_reason == "review_outcome_unread"


def test_a_refusal_whose_cleanup_fails_is_still_recorded_and_returned(candidate, monkeypatch):  # noqa: F811
    """The landing refusal's index cleanup can fail; the refusal still returns typed, names that failure and records its
    blocked attempt, and no commit lands."""
    ctx = candidate
    reference, _reviews = _returned_partial(ctx, monkeypatch)
    head, calls, original = _head(ctx), [], git._unstage_candidate
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_kw: _settle_late(ctx, reference) and None)

    def failing_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("reset failed")
        return original(*args, **kwargs)
    monkeypatch.setattr(git, "_unstage_candidate", failing_once)
    refused = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition=DECISION)
    assert "REVIEW_AUTHOR_CONTINUATION_INVALID" in refused and "OSError: reset failed" in refused, refused
    assert _head(ctx) == head and calls, refused
    row = load_state(ctx.drive_root).attempts[-1]
    assert row.status == "blocked" and row.block_reason == "review_outcome_unread" and "reset failed" in row.block_details
