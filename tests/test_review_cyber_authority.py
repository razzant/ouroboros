"""Cyber action authority is separate from reviewer and physical-operation facts."""

import copy
import json

import pytest

from ouroboros import config
from ouroboros.tools import git
from ouroboros.tools.parallel_review import aggregate_review_verdict
from ouroboros.tools.review_helpers import review_enforcement_blocks
from ouroboros.review_ledger import CouplingOutcome
from tests.test_git_review_preflight_gate import candidate  # noqa: F401
from tests.test_plan_task_tool import harness  # noqa: F401 — the planning organ's fixture


@pytest.fixture(params=["pro", "cyber_pro"])
def access(request, monkeypatch):
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(request.param)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    yield request.param
    config.reset_runtime_mode_baseline_for_tests()


def test_effective_authority_keeps_configured_enforcement(access):
    assert config.get_review_enforcement() == "blocking"
    assert review_enforcement_blocks() == (access == "pro")
    assert not review_enforcement_blocks("advisory")


@pytest.mark.parametrize("status", [
    "responded", "unanswered", "error", "not_performed", "not_dispatched", "pending",
])
def test_coupling_facts_survive_action_authority(candidate, access, status):  # noqa: F811
    """The gate's one verdict (``review_err``) is projected under the owner's
    action authority; the coupling question's facts — its status and its
    findings — are carried through unchanged whichever way the authority goes."""
    finding = {"item": "contract", "severity": "critical", "verdict": "FAIL", "reason": "Original criticism"}
    result = CouplingOutcome(verdict="FAIL", blocked=True, status=status, critical_findings=[finding])
    before = copy.deepcopy(result.__dict__)
    blocked, _, reason, _findings, coupling_items = aggregate_review_verdict(
        "⚠️ REVIEW_BLOCKED: critical findings", result, "critical_findings", [],
        candidate, "candidate", 0, candidate.repo_dir,
    )
    assert blocked == (access == "pro") and reason == "critical_findings"
    assert result.__dict__ == before
    assert coupling_items[0]["verdict"] == "FAIL" and coupling_items[0]["tag"] == "coupling"
    if access == "cyber_pro":
        event = json.loads((candidate.drive_logs() / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert event["review_enforcement"] == "blocking"
        assert event["decision_authority"] == "cyber_pro"


def test_missing_preflight_is_a_stated_fact_never_a_review_or_a_block(candidate, access, monkeypatch):  # noqa: F811
    from ouroboros.review_state import load_state
    from ouroboros.tools import commit_gate

    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *a, **kw: None)
    monkeypatch.setattr(commit_gate, "run_commit_preflight", lambda *a, **kw: pytest.fail("no row was named"))
    outcome = git._preflight_and_tests_gate(candidate, "candidate", 0, classification_paths=["change.py"])
    assert outcome is None, "a commit without a named preflight row blocks under neither authority"
    assert commit_gate._review_preflight_facts(candidate) == {"status": "not_performed", "record_id": ""}
    assert load_state(candidate.drive_root).advisory_runs == []


def test_review_status_readiness_matches_the_actual_gate(candidate, access):  # noqa: F811
    from ouroboros.tools.preflight_review import _handle_review_status
    from ouroboros.review_state import load_state

    projection = json.loads(_handle_review_status(candidate))
    assert "repo_commit_ready" not in projection, "no readiness axis is projected under either authority"
    assert not projection["open_obligations"] and not projection["commit_readiness_debts"]
    assert "advisory_runs" not in projection and "advisory_run_archive" not in projection
    assert not load_state(candidate.drive_root).advisory_runs


def test_failed_review_returns_before_git_and_cyber_continues_explicitly(candidate, access, monkeypatch):  # noqa: F811
    """#1741: a Cyber outcome, even a failed review, reaches the author before Git; only the
    author's explicit continuation commits the independently bound current bytes."""
    import json as _json
    from ouroboros.tools import commit_gate, git_review_cycle

    # The wave's one verdict: nobody answered the coupling question (the seat
    # asked it errored) — NOT_PERFORMED, carried as ``review_err`` with the
    # coupling outcome's facts beside it.
    result = CouplingOutcome(blocked=True, status="error")
    not_performed = ("⚠️ REVIEW_BLOCKED: review NOT_PERFORMED — the coupling question (Part 2) "
                     "was answered by no seat — asked of: slot_2.")
    monkeypatch.setattr(git, "_preflight_and_tests_gate", lambda *a, **k: None)
    monkeypatch.setattr(git, "_install_paid_dispatch_stamp", lambda *a, **k: None)
    monkeypatch.setattr(git, "_reconcile_and_clear_review_roster", lambda *a, **k: None)
    monkeypatch.setattr(git, "_run_parallel_review",
                        lambda *a, **k: (not_performed, result, "coupling_not_performed", []))
    git._reset_commit_review_state(candidate)
    outcome = git_review_cycle._run_reviewed_stage_cycle(
        candidate, "candidate", 0, paths=["change.py"], require_release_tag=False,
    )
    assert outcome["status"] == ("reviewed" if access == "cyber_pro" else "blocked")
    assert result.status == "error" and result.blocked
    if access != "cyber_pro":
        return
    assert "Review outcome returned before commit" in outcome["message"]
    reference = _json.loads(outcome["message"].split("\n", 1)[1])["review_reference"]
    assert reference == outcome["review_reference"] and reference["surface"] == "commit"
    assert "value = 2" not in git.run_cmd(["git", "diff", "--cached"], cwd=candidate.repo_dir), "nothing stays staged"
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **k: pytest.fail("continuation buys no reviewer"))
    git._reset_commit_review_state(candidate)
    assert commit_gate.prepare_author_commit_request(candidate, reference, {
        "disposition": "rejected", "rationale": "I read the unperformed coupling outcome and accept these bytes."}, "") is None
    assert git._check_overlapping_review_attempt(candidate) is None
    continued = git_review_cycle._run_reviewed_stage_cycle(
        candidate, "candidate", 0, paths=["change.py"], require_release_tag=False)
    assert continued["status"] == "passed", continued
    assert candidate._commit_review_status == "author_continued"
    author = candidate._author_commit_record
    assert author["review_reference"] == reference and author["subject_hash"] == continued["pre_fingerprint"]["fingerprint"]
    assert author["reviewer_signal"] and author["reviewer_signal"] != "passed"
    assert "value = 2" in git.run_cmd(["git", "show", ":change.py"], cwd=candidate.repo_dir)


def test_pending_review_retains_custody_when_author_continues(candidate, access):  # noqa: F811
    from ouroboros.review_state import load_state

    candidate._last_triad_raw_results = [{"slot_id": "s1", "status": "error", "operation_state": "in_flight", "operation_id": "op-1"}]
    candidate._current_review_retry_key = "same-paid-work"
    result = git._finalize_pending_review(candidate, "candidate", 0,
        pre_fingerprint={"fingerprint": "fp"}, post_fingerprint={"fingerprint": "fp"})
    assert "REVIEW_PENDING" in result, "a pending wave never licenses Git in either mode"
    saved = load_state(candidate.drive_root).attempts[-1]
    assert saved.status == "reviewing" and saved.late_result_pending
    assert saved.triad_raw_results[0]["operation_id"] == "op-1"
    if access == "cyber_pro":
        git._record_commit_attempt(candidate, "candidate", "succeeded")
        saved = load_state(candidate.drive_root).attempts[-1]
        assert saved.late_result_pending
        assert saved.triad_raw_results[0]["operation_state"] == "in_flight"


def test_pending_cyber_commit_uses_no_second_dispatch(candidate, access, monkeypatch):  # noqa: F811
    from ouroboros.review_state import load_state

    candidate._current_review_retry_key = "old-review"
    candidate._last_triad_raw_results = [{"slot_id": "critic", "operation_id": "original-op", "operation_state": "in_flight"}]
    git._finalize_pending_review(candidate, "old candidate", 0,
        pre_fingerprint={"fingerprint": "old"}, post_fingerprint={"fingerprint": "old"})
    before = load_state(candidate.drive_root).attempts[-1].triad_raw_results
    git._reset_commit_review_state(candidate)
    outcome = git._check_overlapping_review_attempt(candidate)
    # The caller's own exact open attempt is rejoined in every mode (never a Cyber skip).
    assert outcome is None and candidate._review_resume_pending and not candidate._review_cyber_pending
    if access == "cyber_pro":
        monkeypatch.setattr(git, "check_review_cycles_ceiling", lambda *a, **k: pytest.fail("new review admission"))
        free = git._free_cycle_gate(candidate, "new candidate", 0,
            pre_fingerprint={"fingerprint": "new"}, review_rebuttal="")
        # Other intent than the open paid wave: its pending outcome returns, nothing is bought or vetoed.
        assert free["status"] == "reviewed" and "this other intent waits; nothing was dispatched" in free["message"]
        assert free["review_reference"]["attempt"] == load_state(candidate.drive_root).attempts[-1].attempt
        assert load_state(candidate.drive_root).attempts[-1].triad_raw_results == before
        monkeypatch.setattr(git, "_preflight_and_tests_gate", lambda *a, **k: None)
        monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **k: pytest.fail("duplicate paid panel"))
        cycle = git._run_reviewed_stage_cycle(candidate, "new candidate", 0,
            paths=["change.py"], require_release_tag=False)
        assert cycle["status"] == "reviewed", "no Git effect while the paid review is open"
        reference = cycle["review_reference"]
        git._reset_commit_review_state(candidate)
        from ouroboros.tools import commit_gate
        assert commit_gate.prepare_author_commit_request(candidate, reference, {
            "disposition": "deferred", "rationale": "Continue now; the open review stays collectible."}, "") is None
        assert git._check_overlapping_review_attempt(candidate) is None and not candidate._review_resume_pending
        continued = git._run_reviewed_stage_cycle(candidate, "new candidate", 0,
            paths=["change.py"], require_release_tag=False)
        assert continued["status"] == "passed", continued
        git._record_commit_attempt(candidate, "new candidate", "succeeded")
        attempts = load_state(candidate.drive_root).attempts
        assert attempts[-2].late_result_pending and attempts[-2].triad_raw_results == before
        assert not attempts[-1].late_result_pending and not attempts[-1].triad_raw_results


@pytest.mark.parametrize("status,stale", [("blockers", False), ("pending", False), ("clean", True)])
def test_skill_gate_does_not_relabel_the_verdict(access, status, stale):
    from ouroboros.skill_review_status import skill_review_gate

    result = skill_review_gate(status, stale=stale, findings=[{"item": "skill_preflight", "verdict": "FAIL"}])
    assert result["executable_review"] == (access == "cyber_pro")
    assert result["status"] == status and result["stale"] == stale
    assert result["review_enforcement"] == "blocking"
    assert result["preflight_failed"] == (not stale)


def test_skill_author_can_finish_without_fabricating_first_feedback(tmp_path, access, monkeypatch):
    from ouroboros.skill_loader import load_review_state, save_enabled
    from ouroboros.tool_access_types import ResolvedResourceBinding
    from ouroboros.tools import skill_exec
    from tests.test_skill_exec import _build_skill, _make_ctx

    ctx = _make_ctx(tmp_path)
    directory = _build_skill(ctx.drive_root / "skills" / "external", "demo")
    binding = ResolvedResourceBinding(profile="self_modification", root="skill_payload", operation="review",
        base_path=directory, target_path=directory, source="test", skill_name="demo", state_drive_root=ctx.drive_root)
    monkeypatch.setattr(skill_exec, "run_skill_review_lifecycle_blocking", lambda *a, **k: pytest.fail("Unexpected panel"), raising=False)
    result = skill_exec._author_finish_existing_skill_review(ctx, binding, "demo",
        disposition="accepted", rationale="Run this local greeter with the available evidence.")
    if access == "pro":
        assert "requires advisory" in result["error"]
        return
    assert "error" not in result, result
    saved = load_review_state(ctx.drive_root, "demo")
    assert saved.status == "pending" and not saved.raw_actor_records and not saved.raw_result
    assert saved.author_disposition["reviewer_signal"] == "pending"
    assert saved.author_disposition["enforcement"] == "blocking"
    save_enabled(ctx.drive_root, "demo", True)
    actual = json.loads(skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py"))
    assert actual["exit_code"] == 0 and "hello from skill" in actual["stdout"]


def test_skill_review_texts_state_the_gate_skill_exec_obeys(tmp_path, access, monkeypatch):
    """#1735: no skill_review text promises a lock Cyber Pro does not apply."""
    from ouroboros.skill_loader import save_enabled
    from ouroboros.tools import review_change, skill_exec
    from tests.test_skill_exec import _build_skill, _make_ctx, _valid_script_manifest

    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    # A declared script that does not exist fails the deterministic preflight: $0, no panel.
    manifest = _valid_script_manifest("demo").replace(
        "---\n# body", "  - name: absent.py\n    description: Missing.\n---\n# body")
    _build_skill(skills_root, "demo", manifest=manifest)
    monkeypatch.setattr(review_change, "run_review_change", lambda *a, **k: pytest.fail("no panel"))
    reviewed = skill_exec._handle_review_skill(ctx, skill="demo")
    assert "status=pending" in reviewed and "skill_preflight failed" in reviewed
    assert "not executable" not in reviewed
    for schema in (skill_exec._REVIEW_SCHEMA, skill_exec._EXEC_SCHEMA):
        description = schema["description"]
        assert "never executable" not in description and "Cyber Pro" in description
        assert "must be enabled and carry a fresh executable review verdict" not in description
    save_enabled(ctx.drive_root, "demo", True)
    ran = skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py")
    if access == "cyber_pro":
        assert json.loads(ran)["exit_code"] == 0, ran
    else:
        assert ran.startswith("⚠️ SKILL_EXEC_BLOCKED") and "review_pending" in ran


def test_real_plan_gate_ignores_legacy_force_plan_in_every_access_mode(harness, access):  # noqa: F811
    from ouroboros.owner_hurry import plan_gate_decision

    result = plan_gate_decision(harness.make_ctx(force_plan=True), {}, enforcement="blocking")
    assert result["allow"] is True and result["status"] == "not_required"
