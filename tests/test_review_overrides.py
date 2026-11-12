"""The owner-visible trace of a review that returned FAIL or no verdict (``ouroboros/review_overrides.py``).

One writer keeps the event, the install counter and the work's card fact (count, latest reason) on its root task; the
author's explicit continuation over such an outcome adds ``continued`` and a commit Git records over it adds
``landed``, each only from the call that met the outcome or the reviewed attempt its author continuation names. The
card shows each part only when recorded, in Python and in the app alike; the agent's restart notice, a settled
adoption and an evolution cycle's notice name the landed commits with their outcome, "running" only on the adoption
record's word, and never PASS.
"""
from __future__ import annotations

import copy
import json
import subprocess
import time
import uuid
from types import SimpleNamespace

import pytest

from ouroboros.review_overrides import (
    COUNTER_REL, card_line, landed_line, landed_outcome, note_author_continuation, note_override_landing,
    record_override,
)
from ouroboros.utils import read_json_dict
from tests._git_review_pipeline_shared import _make_ctx
from tests._store_shared import read_result, seed_result

GATE = "over a FAIL or no verdict"
WHY = "I read the finding and choose to continue."


def _events(ctx):
    path = ctx.drive_root / "logs" / "events.jsonl"
    return [row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if row.get("type") == "review_advisory_override"] if path.exists() else []


def _decide(ctx, reason, message="finding"):
    ctx._last_review_block_reason = reason
    record_override(ctx, message)


def _commit(repo, name):
    (repo / name).write_text(name, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", name], cwd=repo, check=True, capture_output=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _landed(ctx, *rows):
    """Counter rows ``(sha, reason[, disposition, rationale])``, as ``note_override_landing`` writes them."""
    (ctx.drive_root / COUNTER_REL).parent.mkdir(parents=True, exist_ok=True)
    (ctx.drive_root / COUNTER_REL).write_text(json.dumps({"count": len(rows), "landed": [
        {"sha": sha, "reason": reason, "task_id": "t", "at": "2026-11-11T10:00:00+00:00",
         **dict(zip(("disposition", "rationale"), author))} for sha, reason, *author in rows]}), encoding="utf-8")


def _fact(root, task_id):
    return (read_result(root, task_id) or {}).get("review_overrides") or {}


def _evolution_notice(monkeypatch, data_dir, sha):
    """The owner's notice of an absorbed evolution cycle, as ``notify_owner_cycle_outcome`` sends it."""
    from supervisor import evolution_lifecycle

    sent: list = []
    monkeypatch.setattr("supervisor.queue.DRIVE_ROOT", data_dir)
    monkeypatch.setattr("supervisor.state.load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda chat, text, **kw: sent.append(text))
    evolution_lifecycle.notify_owner_cycle_outcome({"objective": "Sharpen"}, {"cycle_outcome": "absorbed",
                                                                              "commit_sha": sha})
    assert len(sent) == 1 and "reviewed" not in sent[0] and "PASS" not in sent[0], sent
    return sent[0]


def test_each_decision_counts_on_its_work_card_beside_the_event_and_the_counter(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    ctx = _make_ctx(tmp_path)
    ctx.task_id = "root-1"
    seed_result(ctx.drive_root, "root-1", {"status": "running", "chat_id": 1})
    child = copy.copy(ctx)
    child.task_id, child.task_metadata = "child-1", {"root_task_id": "root-1"}
    _decide(ctx, "critical_findings", "[CRITICAL] tests_affected: broken")
    _decide(child, "review_quorum", "Too few reviewers answered")
    fact = _fact(ctx.drive_root, "root-1")
    assert fact["count"] == 2 and fact["latest"]["reason"] == "review_quorum" and fact["latest"]["task_id"] == "child-1"
    assert fact["latest"]["text"] == "Too few reviewers answered"
    first, second = _events(ctx)
    assert second["task_id"] == "child-1" and second["work_task_id"] == "root-1" and second["review_overrides"] == fact
    assert second["decision_authority"] == "advisory" and second["block_reason"] == "review_quorum"
    assert first["review_overrides"]["count"] == 1, "the live copy is the fact this decision left"
    counter = read_json_dict(ctx.drive_root / COUNTER_REL)
    assert counter["count"] == 2 and [row["block_reason"] for row in counter["recent"]] == ["critical_findings", "review_quorum"]
    assert card_line(fact) == "Review: FAIL or no verdict 2× (latest: review quorum)"
    assert card_line({"count": 0}) == card_line(None) == "", "no decision, no field"
    # A work with no record to hold it: no partial row is minted; the event and the counter keep the decision.
    ghost = copy.copy(ctx)
    ghost.task_id = "ghost"
    _decide(ghost, "infra_failure")
    assert read_result(ctx.drive_root, "ghost") is None and "review_overrides" not in _events(ctx)[-1]
    assert read_json_dict(ctx.drive_root / COUNTER_REL)["count"] == 3
    _decide(ctx, "critical_findings", "word " * 100)  # a long reason keeps a head that says it is cut
    latest = _fact(ctx.drive_root, "root-1")["latest"]["text"]
    assert len(latest) <= 300 and latest.endswith("word…") and latest.startswith("word word")


def test_the_author_continuation_and_the_landing_add_their_parts_and_keep_the_others(tmp_path):
    """The card's three parts live on one field: a later outcome keeps the author's continuation and the landing, an
    old ``{count, latest}`` field shows its one part with no zeros, and an author continuation counts only over an
    attempt that met a FAIL or no verdict."""
    ctx = _make_ctx(tmp_path)
    ctx.task_id = "root-1"
    seed_result(ctx.drive_root, "root-1", {"status": "running", "review_overrides": {"count": 3, "latest": {
        "at": "2026-11-11T10:00:00+00:00", "reason": "critical_findings", "task_id": "root-1", "text": "finding"}}})
    old = _fact(ctx.drive_root, "root-1")
    assert card_line(old) == "Review: FAIL or no verdict 3× (latest: critical findings)", "an old field: one part"
    ctx._author_commit_source = SimpleNamespace(review_override_reason="", review_record_id="rec-0")
    note_author_continuation(ctx, {"disposition": "accepted", "rationale": "clean"})
    assert _fact(ctx.drive_root, "root-1") == old, "a continuation over a clean review is no part of the trace"
    ctx._author_commit_source = SimpleNamespace(review_override_reason="critical_findings", review_record_id="rec-1")
    note_author_continuation(ctx, {"disposition": "rejected", "rationale": "x" * 400})
    continued = _fact(ctx.drive_root, "root-1")["continued"]
    assert continued["count"] == 1 and continued["latest"]["record_id"] == "rec-1"
    assert continued["latest"]["reason"] == "critical_findings" and continued["latest"]["disposition"] == "rejected"
    assert len(continued["latest"]["rationale"]) == 300 and continued["latest"]["rationale"].endswith("x…")
    ctx._author_commit_record = {"disposition": "rejected", "rationale": WHY}
    note_override_landing(ctx, "a" * 40)
    _decide(ctx, "review_quorum")  # a later outcome keeps both parts
    fact = _fact(ctx.drive_root, "root-1")
    assert fact["count"] == 4 and fact["continued"] == continued and fact["landed"]["count"] == 1
    assert fact["landed"]["latest"]["sha"] == "a" * 40
    assert _events(ctx)[-1]["review_overrides"] == fact, "the live copy carries every part"
    assert card_line(fact) == ('Review: FAIL or no verdict 4× (latest: review quorum) · author continued 1× (latest: '
                               f'rejected — "{"x" * 79}…") · landed over it: 1 commit')
    [row] = read_json_dict(ctx.drive_root / COUNTER_REL)["landed"]
    assert (row["disposition"], row["rationale"], row["reason"]) == ("rejected", WHY, "critical_findings")


@pytest.mark.parametrize("split", [False, True])
def test_a_commit_lands_over_a_decision_only_from_its_call_or_the_attempt_it_continues(tmp_path, split):
    from ouroboros.review_state import load_state, make_repo_key
    from ouroboros.tools.commit_gate import _record_commit_attempt
    from ouroboros.tools.git_review_cycle import _reset_commit_review_state

    ctx = _make_ctx(tmp_path)
    ctx.task_id, install = "t1", ctx.drive_root
    if split:  # a task drive apart from the install's data root, which restart and adoption read
        install = tmp_path / "install"
        ctx.budget_drive_root = str(install)

    def reviewed(decision=""):  # one call whose review hands its outcome back before Git (Advisory, Cyber Pro)
        _reset_commit_review_state(ctx)
        _record_commit_attempt(ctx, "change", "reviewing", phase="review")
        if decision:
            _decide(ctx, decision)
        _record_commit_attempt(ctx, "change", "reviewed", phase="review_only", block_reason=decision or "author_decision_required")
        return ctx._current_review_attempt_number

    def continues(attempt=None):  # a later call; its validated reference resolved to that durable attempt
        _reset_commit_review_state(ctx)
        if attempt is not None:
            ctx._author_commit_source = load_state(ctx.drive_root).latest_attempt_for(
                repo_key=make_repo_key(ctx.repo_dir), task_id="t1", tool_name="commit_reviewed", attempt=attempt)

    def landed():
        return [row["sha"][:1] for row in (read_json_dict(install / COUNTER_REL) or {}).get("landed") or []]

    over, clean = reviewed("critical_findings"), reviewed()
    assert load_state(ctx.drive_root).latest_attempt_for(attempt=over).review_override_reason == "critical_findings"
    continues(clean)
    note_override_landing(ctx, "a" * 40)
    assert landed() == [], "a commit over a review that met no decision is not over the earlier one"
    continues(over)
    note_override_landing(ctx, "b" * 40)
    assert landed() == ["b"]
    continues()  # a later call that continues nothing keeps no stale reference alive
    note_override_landing(ctx, "c" * 40)
    assert landed() == ["b"]
    continues(clean)
    _decide(ctx, "preflight")  # the continuation's own preflight, waved through in this very call
    note_override_landing(ctx, "d" * 40)
    assert landed() == ["b", "d"]
    for index in range(70):
        note_override_landing(ctx, f"{index:040d}")
    assert len(landed()) == 64, "the newest 64 landings are kept"
    if split:
        base, sha = _commit(ctx.repo_dir, "base"), _commit(ctx.repo_dir, "over")
        continues(over)
        note_override_landing(ctx, sha)
        assert sha[:12] in landed_line(ctx.repo_dir, install, base=base, target=sha, running=False)
        assert "landed" not in read_json_dict(ctx.drive_root / COUNTER_REL), "the task drive keeps no landing"


def test_an_open_wave_counts_nothing_yet_and_a_collected_outcome_counts_once(tmp_path, monkeypatch):
    """One outcome, one count (REV6 C2): the gate hands back a wave still open twice in one call and counts neither,
    yet a commit this call lands still goes over it; the call that collects that wave counts how it ended, and a call
    collecting an attempt whose outcome was already counted adds nothing."""
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    ctx = _make_ctx(tmp_path)
    ctx.task_id = "root-1"
    seed_result(ctx.drive_root, "root-1", {"status": "running"})
    _decide(ctx, "review_late_result_pending", "REVIEW_PENDING")
    _decide(ctx, "review_late_result_pending", "REVIEW_BLOCKED: review NOT_PERFORMED")
    assert _fact(ctx.drive_root, "root-1") == {} and _events(ctx) == [] and read_json_dict(ctx.drive_root / COUNTER_REL) is None
    note_override_landing(ctx, "a" * 40)  # Cyber Pro: the same call commits over the open wave
    assert [row["reason"] for row in read_json_dict(ctx.drive_root / COUNTER_REL)["landed"]] == ["review_late_result_pending"]
    assert landed_outcome(ctx.drive_root, "a" * 40) == "no review verdict when it landed: review late result pending"
    ctx._review_override_this_call, ctx._review_reconcile_only = "", True  # later calls collect that wave
    ctx._pending_review_attempt = SimpleNamespace(review_override_reason="review_late_result_pending")
    _decide(ctx, "critical_findings")  # it ended in FAIL
    ctx._pending_review_attempt = SimpleNamespace(review_override_reason="critical_findings")
    _decide(ctx, "critical_findings")  # the same call made again collects the counted FAIL
    fact = _fact(ctx.drive_root, "root-1")
    assert fact["count"] == 1 and fact["latest"]["reason"] == "critical_findings" and len(_events(ctx)) == 1
    assert read_json_dict(ctx.drive_root / COUNTER_REL)["count"] == 1
    _decide(ctx, "review_quorum")  # a collected attempt's new outcome is another one
    assert _fact(ctx.drive_root, "root-1")["count"] == 2 and len(_events(ctx)) == 2
    ctx._review_reconcile_only = False  # a new wave of the task; the collected attempt is only left on the context
    _decide(ctx, "critical_findings")  # the same reason the left-over attempt keeps
    assert _fact(ctx.drive_root, "root-1")["count"] == 3 and len(_events(ctx)) == 3, "a left-over attempt silences nothing"


def test_a_continuation_over_a_wave_handed_back_open_goes_over_what_its_record_says_now(tmp_path, monkeypatch):
    """B and C go over the reason the gate kept on the continued attempt, but a wave it handed back still open stands
    at what its review record holds now (REV6 C1): passed since, it is gone over by nothing; failed since, by its FAIL;
    still open, by no verdict; with no readable record, as open as the gate saw it. A landing names the continued
    attempt's outcome before this call's own check (REV6 A9)."""
    from ouroboros.review_overrides import _went_on_over

    ctx = _make_ctx(tmp_path)
    ctx.task_id = "root-1"
    seed_result(ctx.drive_root, "root-1", {"status": "running"})
    records = {name: {"verdict": {"aggregate": aggregate, "reason": reason}} for name, aggregate, reason in (
        ("pass", "PASS", "pass"), ("fail", "FAIL", "critical_findings"),
        ("open", "NOT_PERFORMED", "review_late_result_pending"),
        ("gate", "NOT_PERFORMED", "gate_block:tests_preflight_blocked"))} | {"bare": {}}
    records["open_fail"] = {"rows": [{"seat_id": "s1"}, {"seat_id": "s2"}, {"seat_id": "adv", "additional": True}],
                            "verdict": {"aggregate": "NOT_PERFORMED", "reason": "review_late_result_pending",
                                        "per_row": {"s1": "FAIL", "s2": "pending", "adv": "PASS"}}}
    records["open_adviser_fail"] = {**records["open_fail"], "verdict": {**records["open_fail"]["verdict"],
                                                                        "per_row": {"s1": "PASS", "adv": "FAIL"}}}

    def load(_root, record_id):
        if record_id == "corrupt":
            raise ValueError("review ledger record corrupt exists but is not readable as a record")
        return records.get(record_id)

    monkeypatch.setattr("ouroboros.review_ledger.load_record", load)

    def source(kept, record_id):
        return SimpleNamespace(review_override_reason=kept, review_record_id=record_id)

    open_wave = "review_late_result_pending"
    for kept, record_id, over in (
            (open_wave, "pass", ""), (open_wave, "fail", "critical_findings"), (open_wave, "open", open_wave),
            (open_wave, "gate", "tests_preflight_blocked"), (open_wave, "missing", open_wave),
            (open_wave, "corrupt", open_wave), (open_wave, "bare", open_wave), (open_wave, "", open_wave),
            ("critical_findings", "pass", "critical_findings"), ("preflight", "pass", "preflight"),
            ("review_quorum", "", "review_quorum"), ("", "fail", ""), ("", "", ""),
            (open_wave, "open_fail", "critical_findings"), (open_wave, "open_adviser_fail", open_wave),
            ("preflight", "fail", "critical_findings"), ("preflight", "open_fail", "critical_findings"),
            ("tests_preflight_blocked", "open", "tests_preflight_blocked"), ("preflight", "corrupt", "preflight")):
        assert _went_on_over(ctx, source(kept, record_id)) == over, (kept, record_id)
    assert _went_on_over(ctx, None) == ""
    ctx._author_commit_source = source("review_late_result_pending", "pass")
    note_author_continuation(ctx, {"disposition": "accepted", "rationale": "Read the settled PASS."})
    note_override_landing(ctx, "a" * 40)
    assert _fact(ctx.drive_root, "root-1") == {} and read_json_dict(ctx.drive_root / COUNTER_REL) is None
    ctx._author_commit_source = source("review_late_result_pending", "fail")
    ctx._review_override_this_call = "preflight"  # this continuation's own check failed too (Cyber Pro)
    note_author_continuation(ctx, {"disposition": "accepted", "rationale": WHY})
    note_override_landing(ctx, "b" * 40)
    fact = _fact(ctx.drive_root, "root-1")
    assert fact["continued"]["latest"]["reason"] == "critical_findings" and fact["landed"]["count"] == 1
    assert [row["reason"] for row in read_json_dict(ctx.drive_root / COUNTER_REL)["landed"]] == ["critical_findings"]
    # Handed back open, that outcome was never counted: the author's decision counts it, once per review record.
    assert (fact["count"], fact["latest"]["reason"], fact["latest"]["record_id"]) == (1, "critical_findings", "fail")
    assert len(_events(ctx)) == 1 and read_json_dict(ctx.drive_root / COUNTER_REL)["count"] == 1
    note_author_continuation(ctx, {"disposition": "accepted", "rationale": WHY})  # the same attempt continued again
    fact = _fact(ctx.drive_root, "root-1")
    assert fact["count"] == 1 and fact["continued"]["count"] == 2 and len(_events(ctx)) == 1
    ctx._author_commit_source = source("critical_findings", "fail")  # counted when it was handed back
    note_author_continuation(ctx, {"disposition": "accepted", "rationale": WHY})
    assert _fact(ctx.drive_root, "root-1")["count"] == 1 and len(_events(ctx)) == 1
    records["fail2"] = records["fail"]
    ctx._author_commit_source = source("preflight", "fail2")  # a panel FAIL behind a failed check, never counted
    note_author_continuation(ctx, {"disposition": "accepted", "rationale": WHY})
    fact = _fact(ctx.drive_root, "root-1")
    assert (fact["count"], fact["latest"]["reason"], fact["latest"]["record_id"]) == (2, "critical_findings", "fail2")
    ctx._author_commit_source = source("preflight", "pass")  # the panel passed: the counted check stands alone
    note_author_continuation(ctx, {"disposition": "accepted", "rationale": WHY})
    fact = _fact(ctx.drive_root, "root-1")
    assert fact["count"] == 2 and fact["continued"]["latest"]["reason"] == "preflight"


def test_an_open_wave_never_overwrites_a_reason_its_attempt_keeps(tmp_path, monkeypatch):
    """A failed free check stays named on its attempt when the panel's wave then waits open; a settled outcome
    replaces it (REV6C A2)."""
    from ouroboros.review_state import load_state
    from ouroboros.tools.commit_gate import _record_commit_attempt
    from ouroboros.tools.git_review_cycle import _reset_commit_review_state

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    ctx = _make_ctx(tmp_path)
    ctx.task_id = "t1"
    _reset_commit_review_state(ctx)
    _record_commit_attempt(ctx, "change", "reviewing", phase="review")

    def kept():
        return load_state(ctx.drive_root).latest_attempt_for(attempt=ctx._current_review_attempt_number).review_override_reason

    _decide(ctx, "review_late_result_pending")
    assert kept() == "review_late_result_pending", "an attempt with no reason keeps the open wave's"
    _decide(ctx, "preflight")
    _decide(ctx, "review_late_result_pending")
    assert kept() == "preflight"
    _decide(ctx, "critical_findings")
    assert kept() == "critical_findings", "a settled outcome replaces it"


@pytest.mark.parametrize("case", ["warm", "restarted", "successor", "clean", "fixed", "refused"])
def test_the_author_continuation_lands_over_its_attempts_decision_in_any_context(tmp_path, monkeypatch, case):
    """Through the real gate: Advisory hands the reviewed outcome back before Git and the author's continuation commits
    it. Its validated reference names the durable attempt that keeps the decision, so the same context, a fresh one (a
    restarted worker) and a Continue successor land the commit alike, and the continuing work's card counts the
    author's continuation and the landing beside the outcome. A review that met no decision, a FAIL the author fixed
    before a passing wave, lands nothing over it; a continuation Git refused counts the decision, never a landing."""
    from ouroboros import commit_admission, owner_continue
    from ouroboros.mutation_attribution import capture_mutation_baseline
    from ouroboros.review_dispatch import invoke_review_paid_stamp
    from ouroboros.review_ledger import CouplingOutcome
    from ouroboros.review_state import load_state
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import git, review, review_helpers
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(review_helpers, "check_worktree_readiness", lambda *a, **k: [])
    monkeypatch.setattr(commit_admission, "release_metadata_diagnostics", lambda *a, **k: {
        "status": "not_applicable", "findings": [], "unavailable": []})
    for name, stub in (("_managed_candidate_needs_proof", lambda *a: False), ("_run_review_preflight_tests", lambda *a, **k: None),
                       ("_post_commit_result", lambda *a, **k: None), ("_auto_push", lambda *a, **k: "")):
        monkeypatch.setattr(git, name, stub)
    ctx = _make_ctx(tmp_path)
    ctx.task_id, ctx.branch_dev = "author", "ouroboros"
    (ctx.repo_dir / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    subprocess.run(["git", "add", "VERSION"], cwd=ctx.repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "version"], cwd=ctx.repo_dir, check=True, capture_output=True)
    for task_id in ("author", "successor"):
        write_task_result(ctx.drive_root, task_id, "running")
        capture_mutation_baseline(ctx.drive_root, task_id, [{"surface_type": "system_repo", "host_root": str(ctx.repo_dir)}])
    (ctx.repo_dir / "change.py").write_text("value = 2\n", encoding="utf-8")
    waves: list = []

    def reviewer(actual, message, **kw):  # the paid model alone is scripted; the gate's decision seam is real
        invoke_review_paid_stamp(actual._review_paid_stamp)
        fails = case != "clean" and not (case == "fixed" and waves)  # the fixed code's second wave passes
        waves.append(fails)
        finding = {"item": "code_quality", "severity": "critical", "verdict": "FAIL" if fails else "PASS",
                   "reason": "wrong value"}
        actual._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "parsed": [finding], "raw_text": "x"}]
        if fails:
            actual._last_review_critical_findings, actual._last_review_block_reason = [finding], "critical_findings"
            assert review._handle_review_block_or_warning(actual, "critical feedback", "") is None
        return None, CouplingOutcome(verdict="PASS", status="responded"), "", []

    def head():
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir, check=True, capture_output=True,
                              encoding="utf-8").stdout.strip()

    base = head()
    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    first = git._repo_commit_push(ctx, "Change value", skip_advisory_review=True)
    assert first.startswith("Review outcome returned before commit"), first
    if case == "fixed":  # the author fixes the code and a new paid wave passes it
        (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")
        first = git._repo_commit_push(ctx, "Change value", skip_advisory_review=True)
        assert first.startswith("Review outcome returned before commit") and waves == [True, False], first
    reference = json.loads(first.split("\n", 1)[1])["review_reference"]
    author = ctx
    if case in {"restarted", "successor"}:  # a context built afresh, as a restarted worker or the successor builds it
        author = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root,
                             task_id="successor" if case == "successor" else "author")
        author.branch_dev = "ouroboros"
    monkeypatch.setattr(owner_continue, "recorded_continuation", lambda _root, before, after: (before, after) == (
        "author", "successor"))  # the owner's recorded Continue from the author to its successor
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **k: pytest.fail("the continuation buys no review"))
    if case == "refused":  # Git refuses the commit after the gate settled the author's continuation
        monkeypatch.setattr(git, "_commit_reviewed_candidate", lambda *a: ("", "⚠️ GIT_ERROR (commit): hook refused"))
    said = git._repo_commit_push(author, "Change value", review_reference=reference, author_disposition={
        "disposition": "rejected", "rationale": WHY})
    sha = head()
    landed = (read_json_dict(ctx.drive_root / COUNTER_REL) or {}).get("landed") or []
    outcome = (read_json_dict(ctx.drive_root / COUNTER_REL) or {}).get("count")
    work = _fact(ctx.drive_root, "successor" if case == "successor" else "author")
    assert card_line(_fact(ctx.drive_root, "author")).startswith(
        "" if case == "clean" else "Review: FAIL or no verdict 1× (latest: critical findings)"), case
    if case in {"clean", "fixed"}:
        assert load_state(ctx.drive_root).attempts[-1].status == "succeeded", said
        assert landed == [] and outcome == (None if case == "clean" else 1), (case, said)
        assert "continued" not in work and "landed" not in work and card_line(work) == (
            "" if case == "clean" else "Review: FAIL or no verdict 1× (latest: critical findings)")
        assert landed_line(ctx.repo_dir, ctx.drive_root, base=base, target=sha, running=False) == ""
        assert "no landing over a FAIL or missing verdict is recorded for it." in _evolution_notice(monkeypatch, ctx.drive_root, sha)
        return
    said_by = f'rejected — "{WHY}"'
    assert work["continued"]["count"] == 1 and work["continued"]["latest"]["reason"] == "critical_findings", said
    assert work["continued"]["latest"]["rationale"] == WHY and work["continued"]["latest"]["record_id"]
    if case == "refused":
        assert "GIT_ERROR" in said and sha == base and landed == [] and "landed" not in work, said
        assert card_line(work) == f"Review: FAIL or no verdict 1× (latest: critical findings) · author continued 1× (latest: {said_by})"
        return
    assert load_state(ctx.drive_root).attempts[-1].status == "succeeded", said
    assert [row["sha"] for row in landed] == [sha] and landed[0]["disposition"] == "rejected", (case, said)
    assert work["landed"]["count"] == 1 and work["landed"]["latest"]["sha"] == sha
    parts = f"(latest: {said_by}) · landed over it: 1 commit"
    assert card_line(work) == (f"Review: author continued 1× over critical findings {parts}" if case == "successor" else
                               f"Review: FAIL or no verdict 1× (latest: critical findings) · author continued 1× {parts}"), case
    named = f"{sha[:12]} (review FAIL: critical findings; author continued: {said_by})"
    assert landed_line(ctx.repo_dir, ctx.drive_root, base=base, target=sha, running=False) == (
        f"Committed {GATE}: {named}; deployment unconfirmed, as no record yet binds a running process to them.")
    assert (f"now live (restart verified); it landed over a FAIL or no verdict (review FAIL: critical findings; "
            f"author continued: {said_by}).") in _evolution_notice(monkeypatch, ctx.drive_root, sha)


def test_the_restart_line_names_only_landed_commits_of_its_range_and_says_running_only_when_adopted(tmp_path):
    ctx = _make_ctx(tmp_path)
    repo = ctx.repo_dir
    old = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    base = _commit(repo, "base")
    first, plain, middle, last = _commit(repo, "one"), _commit(repo, "two"), _commit(repo, "check"), _commit(repo, "three")
    _landed(ctx, (old, "critical_findings"), (first, "critical_findings", "accepted", "x" * 100),
            (middle, "tests_preflight_blocked"), (last, "review_quorum"))
    line = landed_line(repo, ctx.drive_root, base=base, target="HEAD", running=False)
    assert line == (f"Committed {GATE}: {last[:12]} (no review verdict: review quorum), {middle[:12]} (check FAIL: "
                    f'tests preflight blocked), {first[:12]} (review FAIL: critical findings; author continued: accepted '
                    f'— "{"x" * 79}…"); deployment unconfirmed, as no record yet binds a running process to them.')
    assert old[:12] not in line and plain[:12] not in line
    running = landed_line(repo, ctx.drive_root, base=base, target=last, running=True)
    assert running.startswith(f"Now running {GATE}: {last[:12]} (no review verdict: review quorum), ") and running.endswith(").")
    for said in (line, running):
        assert "merged" not in said and "Blocking" not in said and "PASS" not in said
    assert landed_line(repo, ctx.drive_root, base=last, target="HEAD", running=False) == ""
    assert landed_line(repo, ctx.drive_root, base="", target="HEAD", running=False) == "", "an unknown base names nothing"
    assert landed_outcome(ctx.drive_root, last) == "no review verdict: review quorum"
    assert landed_outcome(ctx.drive_root, plain) == landed_outcome(ctx.drive_root, "") == "", "no row, no outcome"


def test_the_agent_restart_notice_carries_the_line_from_the_running_baseline(tmp_path, monkeypatch):
    from ouroboros import server_process, server_restart

    ctx = _make_ctx(tmp_path)
    base = _commit(ctx.repo_dir, "base")
    landed = _commit(ctx.repo_dir, "change")
    _landed(ctx, (landed, "critical_findings"))
    monkeypatch.setattr(server_process, "server_source_baseline", lambda: base)
    monkeypatch.setattr(server_restart, "_AGENT_RESTART", {})
    monkeypatch.setattr(server_restart, "_started_restart_work", lambda: {})
    sent: list = []
    host = SimpleNamespace(REPO_DIR=ctx.repo_dir, DRIVE_ROOT=ctx.drive_root, load_state=lambda: {"owner_chat_id": 7},
                           send_with_budget=lambda chat, text, **kw: sent.append((text, kw.get("system_type"))))
    server = SimpleNamespace(uuid=uuid, time=time, _pending_restart={}, _live_running_task_ids=lambda _ctx: ["busy"],
                             _perform_supervisor_restart=lambda *a, **k: pytest.fail("the drain waits for live work"))
    server_restart._handle_restart_in_supervisor(server, {"type": "restart_request", "reason": "body change"}, host)
    assert sent[0] == (f"♻️ Restart requested by agent: body change\nCommitted {GATE}: {landed[:12]} (review FAIL: "
                       "critical findings); deployment unconfirmed, as no record yet binds a running process to them.",
                       "restart_notice")
    sent.clear()
    _landed(ctx)  # nothing landed over a decision: the notice says only what it always said
    server_restart._AGENT_RESTART.clear()
    server_restart._handle_restart_in_supervisor(server, {"type": "restart_request", "reason": "body change"}, host)
    assert sent[0] == ("♻️ Restart requested by agent: body change", "restart_notice")


@pytest.mark.parametrize("outcome,said", [("adopted", "Now running"), ("unconfirmed", "Committed"), ("not_applied", None)])
def test_a_settled_adoption_names_its_landed_commits_running_only_when_adopted(tmp_path, monkeypatch, outcome, said):
    from ouroboros import body_adoption

    ctx = _make_ctx(tmp_path)
    previous = _commit(ctx.repo_dir, "previous")
    candidate = _commit(ctx.repo_dir, "candidate")
    _landed(ctx, (candidate, "critical_findings", "rejected", WHY))
    monkeypatch.setattr(body_adoption, "finalize_on_boot", lambda *a, **k: {
        "outcome": outcome, "note": f"Adoption note ({outcome}).", "adoption_id": "a1", "task_id": "",
        "candidate": candidate, "previous": previous, "branch": "ouroboros", "tests": {"tier": "smoke"}})
    monkeypatch.setattr("supervisor.git_ops_reset._record_checkout_facts", lambda facts: None)
    sent: list = []
    monkeypatch.setattr("supervisor.state.load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda chat, text, **kw: sent.append(text))
    result = body_adoption.settle_on_boot(ctx.drive_root, ctx.repo_dir, supervisor_ready=True)
    named = f'{candidate[:12]} (review FAIL: critical findings; author continued: rejected — "{WHY}")'
    if said is None:
        assert named not in result["note"] and named not in "".join(sent)
    else:
        assert f"{said} {GATE}: {named}" in result["note"] and f"{said} {GATE}: {named}" in sent[0]


def test_a_failing_join_or_store_never_fails_the_gate_the_commit_or_the_notice(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    ctx.task_id, ctx._current_review_attempt_number = "t1", 2
    seed_result(ctx.drive_root, "t1", {"status": "running"})

    def unreadable(*_args):
        raise OSError("repository identity unreadable")

    monkeypatch.setattr("ouroboros.review_state.make_repo_key", unreadable)
    _decide(ctx, "critical_findings")  # the decision still reaches its event and the counter
    assert _events(ctx)[-1]["block_reason"] == "critical_findings"
    assert read_json_dict(ctx.drive_root / COUNTER_REL)["count"] == 1
    ctx._review_override_this_call = ""
    ctx._author_commit_source = SimpleNamespace(review_override_reason="critical_findings", review_record_id="r")

    def unwritable(*_args):
        raise OSError("counter unwritable")

    monkeypatch.setattr("ouroboros.review_overrides.update_json_locked", unwritable)
    note_override_landing(ctx, "a" * 40)  # Git already succeeded: the landing's failure is logged, never raised
    assert "landed" not in read_json_dict(ctx.drive_root / COUNTER_REL)
    assert _fact(ctx.drive_root, "t1")["landed"]["count"] == 1, "the card counts the landing the counter lost"

    def no_store(*_args, **_kwargs):
        raise OSError("task store unwritable")

    monkeypatch.setattr("ouroboros.task_store.patch_document", no_store)
    note_author_continuation(ctx, {"disposition": "accepted", "rationale": WHY})  # never raised into the gate
    note_override_landing(ctx, "b" * 40)
    assert "continued" not in _fact(ctx.drive_root, "t1")
    assert landed_line(ctx.repo_dir, None, base="a", target="b", running=False) == "", "an unreadable counter names nothing"
    (ctx.drive_root / COUNTER_REL).write_text("{not json", encoding="utf-8")
    assert landed_outcome(ctx.drive_root, "a" * 40) is landed_outcome(None, "a" * 40) is None, "unread is not absent"
    assert "whether it landed over a FAIL or no verdict could not be read." in _evolution_notice(monkeypatch, ctx.drive_root, "a" * 40)
    (ctx.drive_root / COUNTER_REL).unlink()
    assert landed_outcome(ctx.drive_root, "a" * 40) == "", "no counter: nothing recorded, never PASS"
    assert "no landing over a FAIL or missing verdict is recorded for it." in _evolution_notice(monkeypatch, ctx.drive_root, "a" * 40)


_TWIN = """
import { reviewGateLine } from '%(card)s';
let input = ''; for await (const chunk of process.stdin) input += chunk;
process.stdout.write(JSON.stringify(JSON.parse(input).map(reviewGateLine)));
"""


def _app_lines(facts):
    import shutil
    from pathlib import Path

    node = shutil.which("node")
    assert node, "Node is required for the Python-to-web twin witness"
    card = Path(__file__).resolve().parents[1] / "web" / "modules" / "review_gate_card.js"
    probe = subprocess.run([node, "--input-type=module", "-e", _TWIN % {"card": card.as_uri()}], input=json.dumps(facts),
                           encoding="utf-8", capture_output=True, timeout=60, check=True)
    return json.loads(probe.stdout)


def test_the_app_line_is_the_python_line_for_every_recorded_part_and_for_none():
    """``card_line`` and ``reviewGateLine`` say the same line for one fact: each part only when recorded (an old field
    with no ``continued`` or ``landed`` shows no zero), the author's reason cut at the same character, a non-ASCII
    one too, and nothing for an absent or malformed fact."""
    latest = {"at": "2026-11-12T10:00:00+00:00", "reason": "critical_findings", "task_id": "w", "text": "finding"}
    author = {"at": "2026-11-12T10:01:00+00:00", "disposition": "accepted", "rationale": "the lock order is intended",
              "reason": "review_quorum", "record_id": "rec-1"}
    landed = {"count": 2, "latest": {"sha": "f" * 40, "at": "2026-11-12T10:02:00+00:00"}}
    facts = [
        {"count": 3, "latest": {**latest, "reason": "review_quorum"}}, {"count": 12}, {"count": 2, "continued": {}},
        {"count": 3, "latest": latest, "continued": {"count": 1, "latest": author}, "landed": landed},
        {"continued": {"count": 2, "latest": {**author, "rationale": "y" * 300}}, "landed": {"count": 1}},
        {"continued": {"count": 1, "latest": {**author, "rationale": "\U0001f642" * 100}}},
        {"continued": {"count": 1, "latest": {"disposition": "deferred"}}}, {"continued": {"count": 1, "latest": {
            "rationale": "why"}}}, {"continued": {"count": 1}}, {"landed": landed},
        None, {}, {"count": 0}, {"count": "3"}, {"count": True}, {"continued": {"count": 0}, "landed": {}}, "fact",
    ]
    edges = "\x1c\x85﻿\xa0 \t"  # the cut drops only ASCII space and control space, in both twins (REV6 A3)
    facts += [{"continued": {"count": 1, "latest": {"rationale": "a" * 78 + edge + "b" * 10}}} for edge in edges]
    expected = [
        "Review: FAIL or no verdict 3× (latest: review quorum)", "Review: FAIL or no verdict 12× (latest: unknown)",
        "Review: FAIL or no verdict 2× (latest: unknown)",
        'Review: FAIL or no verdict 3× (latest: critical findings) · author continued 1× (latest: accepted — "the lock '
        'order is intended") · landed over it: 2 commits',
        f'Review: author continued 2× over review quorum (latest: accepted — "{"y" * 79}…") · landed over it: 1 commit',
        f'Review: author continued 1× over review quorum (latest: accepted — "{chr(0x1f642) * 79}…")',
        "Review: author continued 1× (latest: deferred)", 'Review: author continued 1× (latest: "why")',
        "Review: author continued 1×", "Review: landed over a FAIL or no verdict: 2 commits", *[""] * 7,
        *[f'Review: author continued 1× (latest: "{"a" * 78}{edge}…")' for edge in edges[:4]],
        *['Review: author continued 1× (latest: "' + "a" * 78 + '…")'] * 2,
    ]
    lines = [card_line(fact) for fact in facts]
    assert lines == expected
    assert _app_lines(facts) == lines, "the app's line is the Python line"
    assert all("Blocking" not in line for line in lines)


_CARD_ON_LOAD = """
import { installDom, restoreDom, walkCard } from '%(fixture)s';
import { createChatInstance } from '%(chat)s';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const messages = JSON.parse(input);
const reads = [];
const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () => {
    if (String(url).startsWith('/api/tasks/')) reads.push(String(url));
    return String(url).startsWith('/api/chat/history') ? { messages } : { active_direct_turns: [] };
} }));
const instance = createChatInstance({ ws: { on() { return () => {}; }, isConnected: () => true, send() {} },
    state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
    stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
        gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
    chatId: 1, idPrefix: 'chat', mountEl: mount });
await instance.refreshHistory({ revision: 1 });
const card = (id) => walkCard(globalThis.document.byId.get('chat-messages'), id);
const meta = (id) => card(id)?.querySelector('[data-live-meta]')?.innerHTML ?? null;
const painted = { gate: meta('gate-work'), plain: meta('plain-work'), reads,
    finished: ['gate-work', 'plain-work'].map((id) => card(id)?.dataset.finished === '1') };
instance.destroy(); restoreDom(prior);
process.stdout.write(JSON.stringify(painted));
"""


@pytest.mark.parametrize("finished", [True, False], ids=["finished", "running"])
def test_a_page_load_shows_a_works_field_from_the_history_endpoint(tmp_path, finished):
    """The real history endpoint carries the work's fact on its row and the real card paints the Python line before
    Details, every recorded part of it, with no detail read, for a finished and a running work; a work without
    decisions shows nothing. Without it the line is absent."""
    import asyncio
    import shutil
    from pathlib import Path

    from ouroboros import dialogue_store
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.post_task_checkpoint import set_root_post_task_checkpoint
    from ouroboros.post_task_synthesis import _record_task_facts
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from ouroboros.utils import utc_now_iso

    ctx = _make_ctx(tmp_path)
    root = ctx.drive_root
    env = SimpleNamespace(drive_root=root, repo_dir=ctx.repo_dir)
    for task_id in ("gate-work", "plain-work"):
        task = {"id": task_id, "type": "task", "chat_id": 1}
        write_task_result(root, task_id, STATUS_SCHEDULED, chat_id=1, root_task_id=task_id,
                          result="Task accepted and scheduled.")
        # The work's progress row, as the message bus stores it: the card the page load draws.
        dialogue_store.append(root, "progress", {"ts": utc_now_iso(), "type": "send_message", "task_id": task_id,
                                                 "is_progress": True, "direction": "out", "chat_id": 1,
                                                 "text": "Changing my body", "content": "Changing my body",
                                                 "role": "assistant", "system_type": ""})
        if task_id == "gate-work":
            ctx.task_id = task_id
            _decide(ctx, "critical_findings", "[CRITICAL] tests_affected: the new refusal path has no test")
            ctx._author_commit_source = SimpleNamespace(review_override_reason="critical_findings", review_record_id="r")
            ctx._author_commit_record = {"disposition": "accepted", "rationale": "The lock order is intended."}
            note_author_continuation(ctx, ctx._author_commit_record)
            note_override_landing(ctx, "f" * 40)
        if not finished:
            continue
        _store_task_result(env, task, "Done.", {"rounds": 1, "cost": 0}, {"tool_calls": []})
        _record_task_facts(env, task, {"rounds": 1}, {"tool_calls": []}, root / "logs")
        set_root_post_task_checkpoint(env, task, "completed")  # post-task synthesis settled: the work is finished
    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"chat_id": "1"})))
    messages = json.loads(response.body)["messages"]
    node = shutil.which("node")
    assert node, "Node is required for the Python-to-web page-load witness"
    web = Path(__file__).resolve().parents[1] / "web"
    script = _CARD_ON_LOAD % {"fixture": (web / "tests" / "chat_dom_fixture.js").as_uri(),
                              "chat": (web / "modules" / "chat.js").as_uri()}
    probe = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(messages), encoding="utf-8",
                           capture_output=True, timeout=60, check=True)
    painted = json.loads(probe.stdout)
    line = card_line(_fact(root, "gate-work"))
    assert line == ('Review: FAIL or no verdict 1× (latest: critical findings) · author continued 1× (latest: accepted '
                    '— "The lock order is intended.") · landed over it: 1 commit')
    assert line in painted["gate"], painted
    assert painted["plain"] is not None and "Review:" not in painted["plain"], painted
    assert painted["reads"] == [], "the page load needs no per-card detail read"
    assert painted["finished"] == [finished, finished], "the works' cards are in their real phase"
    carried = {(m["task_id"], m["review_overrides"]["count"]) for m in messages if "review_overrides" in m}
    assert carried == {("gate-work", 1)}, "only the work with a decision carries its fact, on its history row"
