"""Tests for git safety tools, commit gate hardening, and operational polish.

Verifies (Phase 4):
- New tools registered: pull_from_remote, restore_to_head, revert_commit
- SAFETY_CRITICAL_PATHS blocks dangerous operations
- Confirm gates prevent accidental destructive actions
- Auto-tagging on version bump
- Credential helper in git_ops (no token in remote URL)
- New tools in CORE_TOOL_NAMES

Verifies (Phase 5):
- Auto-push wired into commit functions
- legacy token-in-URL credential migration is retired
- ARCHITECTURE.md version sync in startup checks
"""
import importlib
import inspect
import json
import os
import sys
import types

import pytest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_git_module():
    return importlib.import_module("ouroboros.tools.git")


def _get_registry_module():
    return importlib.import_module("ouroboros.tools.registry")


def _get_git_ops_module():
    return importlib.import_module("supervisor.git_ops")


# --- Tool registration tests ---

@pytest.mark.parametrize("tool_name", ["vcs_pull_ff", "vcs_restore", "vcs_revert"])
def test_tool_registered(tool_name):
    git_mod = _get_git_module()
    names = [t.name for t in git_mod.get_tools()]
    assert tool_name in names


CONTRACT_FP = "contract-fp-1"


def _identical_ctx(tmp_path, task_id="t-cap"):
    return types.SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id=task_id)


def _add_attempt(tmp_path, status, fingerprint, *, block_reason="critical_findings",
                 attempt=1, phase="blocking_review", task_id="t-cap",
                 block_class="", rebuttal_sha256="",
                 review_contract_fingerprint=CONTRACT_FP,
                 critical_findings=None, paid=True):
    import pathlib

    from ouroboros.review_state import (
        CommitAttemptRecord,
        make_repo_key,
        update_state,
        _utc_now,
    )

    repo_key = make_repo_key(pathlib.Path(tmp_path))

    def _mutate(state):
        state.attempts.append(CommitAttemptRecord(
            ts=_utc_now(), commit_message="msg", status=status,
            block_reason=block_reason if status == "blocked" else "",
            repo_key=repo_key, tool_name="commit_reviewed", task_id=task_id,
            attempt=attempt, phase=phase,
            pre_review_fingerprint=fingerprint,
            block_class=block_class,
            rebuttal_sha256=rebuttal_sha256,
            review_contract_fingerprint=review_contract_fingerprint,
            critical_findings=list(critical_findings or []),
            paid=paid,
        ))
    update_state(pathlib.Path(tmp_path), _mutate)


def test_identical_diff_refused_free_from_first_verdict_block(tmp_path, monkeypatch):
    """Q12/Q16 contract: identical bytes are never re-reviewed for pay. ONE
    review-verdict block of a staged-diff fingerprint refuses a byte-identical
    resubmission (quoting the recorded verdict), regardless of the cycles knob;
    a changed diff starts fresh; a cross-task identical resubmit stays refused
    (anti-laundering); a success ends the streak."""
    from ouroboros.tools.commit_gate import check_identical_verdict_refusal

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")  # refusal is knob-independent
    ctx = _identical_ctx(tmp_path)

    # FIRST verdict-block already refuses — no paid streak of N is required.
    _add_attempt(tmp_path, "blocked", "fp-same", block_class="verdict",
                 critical_findings=[{"item": "bug_x", "reason": "boom", "severity": "critical"}])
    msg = check_identical_verdict_refusal(ctx, "fp-same", contract_fingerprint=CONTRACT_FP)
    assert "IDENTICAL_DIFF_REFUSED" in msg
    assert "bug_x" in msg  # quotes the recorded verdict
    # Cross-task: the byte-identical diff is the identity.
    other = _identical_ctx(tmp_path, task_id="t-other")
    assert "IDENTICAL_DIFF_REFUSED" in check_identical_verdict_refusal(
        other, "fp-same", contract_fingerprint=CONTRACT_FP)
    # A different staged diff is a fresh paid case.
    assert check_identical_verdict_refusal(ctx, "fp-other", contract_fingerprint=CONTRACT_FP) == ""
    # A refusal record must not reset the streak.
    _add_attempt(tmp_path, "blocked", "fp-same", block_reason="identical_diff_refused",
                 phase="preflight", attempt=2)
    assert "IDENTICAL_DIFF_REFUSED" in check_identical_verdict_refusal(
        ctx, "fp-same", contract_fingerprint=CONTRACT_FP)
    # A successful commit ends the streak.
    _add_attempt(tmp_path, "succeeded", "fp-same", attempt=3, phase="commit")
    assert check_identical_verdict_refusal(ctx, "fp-same", contract_fingerprint=CONTRACT_FP) == ""


def test_identical_refusal_rebuttal_by_content_and_contract_lapse(tmp_path, monkeypatch):
    """Q16/Q22 contract: a rebuttal hash NEW to the streak buys exactly one
    paid re-review; the SAME hash is refused free; a changed (or unknown)
    review-contract fingerprint lapses the streak entirely."""
    from ouroboros.tools.commit_gate import (
        check_identical_verdict_refusal,
        compute_rebuttal_sha256,
    )

    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    ctx = _identical_ctx(tmp_path)
    _add_attempt(tmp_path, "blocked", "fp-r", block_class="verdict")

    new_sha = compute_rebuttal_sha256("the finding is a false positive because ...")
    assert new_sha and compute_rebuttal_sha256("") == ""
    # NEW rebuttal content: exempt (buys one paid re-review).
    assert check_identical_verdict_refusal(
        ctx, "fp-r", rebuttal_sha256=new_sha, contract_fingerprint=CONTRACT_FP) == ""
    # That rebuttal is spent on the streak (recorded on the next verdict-block):
    _add_attempt(tmp_path, "blocked", "fp-r", attempt=2, block_class="verdict",
                 rebuttal_sha256=new_sha)
    repeated = check_identical_verdict_refusal(
        ctx, "fp-r", rebuttal_sha256=new_sha, contract_fingerprint=CONTRACT_FP)
    assert "IDENTICAL_DIFF_REFUSED" in repeated
    assert "repeated rebuttal" in repeated
    # A genuinely different rebuttal buys again.
    assert check_identical_verdict_refusal(
        ctx, "fp-r", rebuttal_sha256=compute_rebuttal_sha256("different evidence"),
        contract_fingerprint=CONTRACT_FP) == ""
    # A rebuttal is "spent" only when it BOUGHT a dispatch (machine-4/wording-2):
    # one recorded on an UNDISPATCHED refusal row (e.g. a ceiling refusal) stays
    # fresh — after the owner raises the cap it still buys its paid re-review.
    undispatched = compute_rebuttal_sha256("never dispatched")
    _add_attempt(tmp_path, "blocked", "fp-r", attempt=3,
                 block_reason="review_cycles_exhausted", phase="preflight",
                 rebuttal_sha256=undispatched, paid=False)
    assert check_identical_verdict_refusal(
        ctx, "fp-r", rebuttal_sha256=undispatched, contract_fingerprint=CONTRACT_FP) == ""
    # Q22: a changed contract fingerprint invalidates the streak — a paid
    # review is allowed and the refusal never quotes across the change.
    assert check_identical_verdict_refusal(
        ctx, "fp-r", contract_fingerprint="another-contract") == ""
    # An unknown current contract (fail-open "") never refuses.
    assert check_identical_verdict_refusal(ctx, "fp-r", contract_fingerprint="") == ""
    # The lapse applies to the streak HEAD only: an OLDER row from a previous
    # contract ends the streak but a NEWER verdict under the current contract
    # keeps its refusal authority.
    _add_attempt(tmp_path, "blocked", "fp-mixed", attempt=1, block_class="verdict",
                 review_contract_fingerprint="old-contract")
    _add_attempt(tmp_path, "blocked", "fp-mixed", attempt=2, block_class="verdict",
                 review_contract_fingerprint=CONTRACT_FP)
    assert "IDENTICAL_DIFF_REFUSED" in check_identical_verdict_refusal(
        ctx, "fp-mixed", contract_fingerprint=CONTRACT_FP)


def test_identical_refusal_skips_infra_and_preflight_rows(tmp_path, monkeypatch):
    """Δ5 contract: infra-blocks (fit/quorum/transport/revalidation) and
    preflight facts neither build the refusal streak nor reset it — the
    recorded verdict stays authoritative through infra noise, and infra-only
    history never refuses anything."""
    from ouroboros.tools.commit_gate import check_identical_verdict_refusal

    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    ctx = _identical_ctx(tmp_path)

    # Infra-only history: retry freely, never a refusal.
    _add_attempt(tmp_path, "blocked", "fp-i", block_reason="review_quorum",
                 block_class="infra")
    _add_attempt(tmp_path, "blocked", "fp-i", block_reason="fixed_overflow",
                 block_class="infra", attempt=2)
    assert check_identical_verdict_refusal(ctx, "fp-i", contract_fingerprint=CONTRACT_FP) == ""

    # A verdict-block, then infra + preflight noise: still refused.
    _add_attempt(tmp_path, "blocked", "fp-i", attempt=3, block_class="verdict")
    _add_attempt(tmp_path, "blocked", "fp-i", block_reason="review_quorum",
                 block_class="infra", attempt=4)
    _add_attempt(tmp_path, "blocked", "fp-i", block_reason="tests_preflight_blocked",
                 phase="preflight", attempt=5)
    _add_attempt(tmp_path, "blocked", "", block_reason="tests_preflight_blocked",
                 phase="preflight", task_id="t-new", attempt=1)
    # machine-2: a FAILED infra/expired row (lock timeout, path error, expired
    # reviewing attempt) is a transient too — it must not reset the streak.
    _add_attempt(tmp_path, "failed", "fp-i", phase="infra", attempt=6, paid=False)
    _add_attempt(tmp_path, "failed", "fp-i", phase="expired", attempt=7)
    assert "IDENTICAL_DIFF_REFUSED" in check_identical_verdict_refusal(
        ctx, "fp-i", contract_fingerprint=CONTRACT_FP)
    # A POST-REVIEW failure (the paid review completed, usually with a PASS)
    # supersedes the old verdict and ends the streak.
    _add_attempt(tmp_path, "failed", "fp-i", phase="post_commit_tests", attempt=8)
    assert check_identical_verdict_refusal(ctx, "fp-i", contract_fingerprint=CONTRACT_FP) == ""


def test_tests_preflight_block_recorded_with_preflight_phase():
    """The tests-preflight `_record_commit_attempt` call site must stamp
    phase="preflight": without it `infer_review_phase` defaults a blocked
    record to "blocking_review" and legacy-row classification could read a
    flaky test failure as a review verdict for the identical-diff refusal."""
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod)
    idx = source.find('block_reason="tests_preflight_blocked"')
    assert idx != -1
    # The phase stamp must live in the same _record_commit_attempt call.
    window = source[idx:idx + 400]
    assert 'phase="preflight"' in window


def test_legacy_rows_classify_by_block_reason(tmp_path, monkeypatch):
    """Pre-upgrade ledger rows carry no block_class: critical_findings rows
    keep building the refusal streak (verdict), while quorum/fit/transport
    rows classify infra and never refuse; preflight/refusal rows stay
    unclassified."""
    import types as _types

    from ouroboros.tools.commit_gate import (
        BLOCK_CLASS_INFRA,
        BLOCK_CLASS_VERDICT,
        attempt_block_class,
        check_identical_verdict_refusal,
    )

    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)

    def _legacy(status, block_reason, phase="blocking_review", scope_raw=None):
        return _types.SimpleNamespace(
            status=status, block_reason=block_reason, phase=phase,
            block_class="", scope_raw_result=scope_raw or {},
        )

    assert attempt_block_class(_legacy("blocked", "critical_findings")) == BLOCK_CLASS_VERDICT
    assert attempt_block_class(_legacy("blocked", "review_quorum")) == BLOCK_CLASS_INFRA
    assert attempt_block_class(_legacy("blocked", "fixed_overflow")) == BLOCK_CLASS_INFRA
    assert attempt_block_class(_legacy("blocked", "no_advisory", phase="advisory_gate")) == ""
    assert attempt_block_class(_legacy("blocked", "attempt_cap_reached", phase="preflight")) == ""
    # Legacy scope_blocked rows: verdict only when a RESPONDED actor row
    # carried critical findings; sub-floor/overflow scope blocks are infra.
    responded = {"raw_results": [{"status": "responded", "critical_findings": [{"item": "x"}]}]}
    sub_floor = {"raw_results": [{"status": "sub_floor", "critical_findings": []}]}
    assert attempt_block_class(_legacy("blocked", "scope_blocked", scope_raw=responded)) == BLOCK_CLASS_VERDICT
    assert attempt_block_class(_legacy("blocked", "scope_blocked", scope_raw=sub_floor)) == BLOCK_CLASS_INFRA

    # End-to-end on the ledger: a legacy critical_findings row (no block_class)
    # still refuses the identical resubmission.
    ctx = _identical_ctx(tmp_path)
    _add_attempt(tmp_path, "blocked", "fp-legacy", block_class="")
    assert "IDENTICAL_DIFF_REFUSED" in check_identical_verdict_refusal(
        ctx, "fp-legacy", contract_fingerprint=CONTRACT_FP)


def test_non_committing_review_cycle_exists_and_reuses_shared_stage_cycle():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._run_non_committing_review_cycle)
    assert "_run_reviewed_stage_cycle" in source
    assert '"reviewed"' in source
    assert '"review_only"' in source
    assert '["git", "reset", "HEAD"]' in source
    assert '["git", "commit"' not in source


def test_non_committing_review_cycle_runtime_unstages_on_success(monkeypatch, tmp_path):
    git_mod = _get_git_module()
    reset_calls = []
    recorded = []
    released = []

    monkeypatch.setattr(git_mod, "_check_overlapping_review_attempt", lambda ctx: None)
    monkeypatch.setattr(git_mod, "_reconcile_advisory_before_preparation", lambda *a, **kw: "")
    monkeypatch.setattr(git_mod, "_acquire_git_lock", lambda ctx: "lock-token")
    monkeypatch.setattr(git_mod, "_release_git_lock", lambda lock: released.append(lock))
    monkeypatch.setattr(
        git_mod,
        "_run_reviewed_stage_cycle",
        lambda *args, **kwargs: {
            "status": "passed",
            "message": "stage cycle passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post"},
        },
    )
    monkeypatch.setattr(
        git_mod,
        "_record_commit_attempt",
        lambda *args, **kwargs: recorded.append(
            {"status": args[2], "phase": kwargs.get("phase")}
        ),
    )
    monkeypatch.setattr(
        git_mod,
        "run_cmd",
        lambda cmd, cwd=None: reset_calls.append((tuple(cmd), cwd)) or "",
    )

    ctx = types.SimpleNamespace(repo_dir="/tmp/repo", drive_root=tmp_path)
    outcome = git_mod._run_non_committing_review_cycle(ctx, "test commit")

    assert outcome["status"] == "passed"
    assert "Commit was not created" in outcome["message"]
    assert ctx._scope_review_history == {}
    assert recorded == [{"status": "reviewed", "phase": "review_only"}]
    assert released == ["lock-token"]
    assert reset_calls == [(("git", "reset", "HEAD"), "/tmp/repo")]


def test_non_committing_review_cycle_runtime_unstages_on_block(monkeypatch, tmp_path):
    git_mod = _get_git_module()
    reset_calls = []
    released = []

    monkeypatch.setattr(git_mod, "_check_overlapping_review_attempt", lambda ctx: None)
    monkeypatch.setattr(git_mod, "_reconcile_advisory_before_preparation", lambda *a, **kw: "")
    monkeypatch.setattr(git_mod, "_acquire_git_lock", lambda ctx: "lock-token")
    monkeypatch.setattr(git_mod, "_release_git_lock", lambda lock: released.append(lock))
    monkeypatch.setattr(
        git_mod,
        "_run_reviewed_stage_cycle",
        lambda *args, **kwargs: {
            "status": "blocked",
            "message": "review blocked",
            "block_reason": "critical_findings",
        },
    )
    monkeypatch.setattr(
        git_mod,
        "run_cmd",
        lambda cmd, cwd=None: reset_calls.append((tuple(cmd), cwd)) or "",
    )

    ctx = types.SimpleNamespace(repo_dir="/tmp/repo", drive_root=tmp_path)
    outcome = git_mod._run_non_committing_review_cycle(ctx, "test commit")

    assert outcome["status"] == "blocked"
    assert outcome["block_reason"] == "critical_findings"
    assert released == ["lock-token"]
    assert reset_calls == [(("git", "reset", "HEAD"), "/tmp/repo")]


def test_repo_commit_push_uses_shared_reviewed_stage_cycle():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._repo_commit_push)
    assert "_run_reviewed_stage_cycle" in source


# --- Protected-path checks ---

def test_restore_to_head_blocks_protected_paths():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._restore_to_head)
    assert "is_protected_runtime_path" in source or "protected_paths_in" in source
    assert "RESTORE_BLOCKED" in source


def test_revert_commit_blocks_protected_paths():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._revert_commit)
    assert "protected_paths_in" in source
    assert "REVERT_BLOCKED" in source


# --- Confirm gates ---

def test_revert_commit_has_confirm_gate():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._revert_commit)
    assert "confirm" in source
    assert "Call again with confirm=true" in source


def test_restore_to_head_has_confirm_gate():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._restore_to_head)
    assert "confirm" in source
    assert "Call again with confirm=true" in source


# --- Auto-tagging ---
# Removed in v5.15.x:
#   test_auto_tag_function_exists (callable-existence check, no logic)
#   test_auto_tag_called_in_commit_functions (inspect.getsource substring pin)
# The actual auto-tag behavior is exercised end-to-end by the git pipeline
# integration tests in test_git_review_pipeline.py.


def test_auto_tag_not_gated_by_test_warnings():
    """Auto-tagging must run unconditionally — not skipped when tests fail."""
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._repo_commit_push)
    # Find the line(s) that call _auto_tag_on_version_bump
    for line in source.splitlines():
        if "_auto_tag_on_version_bump" in line:
            assert "if not test_warning" not in line, (
                "_repo_commit_push: _auto_tag_on_version_bump must not be gated "
                "by test_warning_ref — tags must always be created on VERSION bump"
            )


# --- Credential helper ---
# test_credential_helper_exists removed in v5.15.x — pure callable-existence
# check; the helper's behavior is exercised by
# test_configure_remote_uses_clean_url below which calls the public
# configure_remote() wrapper.


def test_configure_remote_uses_clean_url():
    """configure_remote must not embed token in the remote URL."""
    git_ops = _get_git_ops_module()
    source = inspect.getsource(git_ops.configure_remote)
    assert "x-access-token" not in source, (
        "configure_remote must use credential helper, not embed token in URL"
    )
    assert "_configure_credential_helper" in source


# --- CORE_TOOL_NAMES ---

def test_new_tools_in_core_tool_names():
    registry = _get_registry_module()
    for name in ("vcs_pull_ff", "vcs_restore", "vcs_revert"):
        assert name in registry.CORE_TOOL_NAMES, (
            f"{name} must be in CORE_TOOL_NAMES"
        )


# --- Pull tool specifics ---

def test_pull_uses_ff_only():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._ff_pull)
    assert "--ff-only" in source, "Pull must use --ff-only for safety"


def test_pull_fetches_before_merge():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._ff_pull)
    fetch_pos = source.find("git fetch")
    merge_pos = source.find("git merge")
    assert fetch_pos != -1, "Must call git fetch"
    assert merge_pos != -1, "Must call git merge"
    assert fetch_pos < merge_pos, "Fetch must come before merge"


# --- Revert tool specifics ---

def test_revert_uses_git_lock():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._revert_commit)
    assert "_acquire_git_lock" in source
    assert "_release_git_lock" in source


def test_revert_aborts_on_failure():
    """On revert failure, git revert --abort must be called."""
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._revert_commit)
    assert '"--abort"' in source and '"revert"' in source


def test_revert_commit_blocks_merge_commits():
    """revert_commit must reject merge commits upfront."""
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._revert_commit)
    assert "merge commit" in source.lower()
    assert "rev-list" in source or "parents" in source


def test_restore_to_head_blocks_safety_critical_full_restore():
    """Full restore (no paths) must check dirty files against protected paths."""
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._restore_to_head)
    assert "affected_critical" in source or "dirty_files" in source, (
        "Full restore must parse dirty files and check against protected paths"
    )


# --- Auto-push ---
# test_auto_push_function_exists removed in v5.15.x — callable-existence
# check superseded by the behavioral tests below that exercise _auto_push
# wiring inside the commit functions.


def test_auto_push_called_in_commit_functions():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._repo_commit_push)
    assert "_auto_push" in source, "_repo_commit_push must call _auto_push after successful commit"


def test_auto_push_not_in_rollback_tools():
    """Auto-push must NOT be wired into restore_to_head or revert_commit."""
    git_mod = _get_git_module()
    for fn_name in ("_restore_to_head", "_revert_commit", "_ff_pull"):
        source = inspect.getsource(getattr(git_mod, fn_name))
        assert "_auto_push" not in source, (
            f"{fn_name} must NOT call _auto_push"
        )


def test_auto_push_is_best_effort():
    """_auto_push must catch all exceptions and return a string (never raise)."""
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._auto_push)
    assert "except Exception" in source
    assert "non-fatal" in source.lower() or "non_fatal" in source.lower()


def test_only_evolution_authority_recheck_and_auto_push_hold_git_lock():
    """Evolution push stays inside the lock; ordinary push returns outside it."""
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._repo_commit_push)
    authority_pos = source.find("_evolution_publication_stopped_result")
    evolution_push_pos = source.find("_auto_push", authority_pos)
    lock_release_pos = source.find("_release_git_lock", evolution_push_pos)
    ordinary_push_pos = source.find("_auto_push", lock_release_pos)
    assert authority_pos < evolution_push_pos < lock_release_pos < ordinary_push_pos


# --- Credential configuration (legacy token-in-URL migration retired) ---

def test_migrate_remote_credentials_is_retired():
    git_ops = _get_git_ops_module()
    assert not hasattr(git_ops, "migrate_remote_credentials")


def test_configure_remote_remains_credential_helper_surface():
    git_ops = _get_git_ops_module()
    configure_source = inspect.getsource(git_ops.configure_remote)
    helper_source = inspect.getsource(git_ops._configure_credential_helper)
    assert "_configure_credential_helper" in configure_source
    assert ".git/credentials" in helper_source


# --- ARCHITECTURE version sync (Phase 5) ---

def test_version_sync_checks_architecture_md():
    """check_version_sync must compare VERSION with ARCHITECTURE.md header."""
    sys.path.insert(0, REPO)
    startup_mod = importlib.import_module("ouroboros.agent_startup_checks")
    source = inspect.getsource(startup_mod.check_version_sync)
    assert "ARCHITECTURE" in source
    assert "architecture_version" in source


# ---------------------------------------------------------------------------
# Advisory pre-review gate (new)
# ---------------------------------------------------------------------------

def _get_advisory_module():
    sys.path.insert(0, REPO)
    return importlib.import_module("ouroboros.tools.claude_advisory_review")


def _get_review_state_module():
    sys.path.insert(0, REPO)
    return importlib.import_module("ouroboros.review_state")


def test_advisory_pre_review_registered():
    """advisory_pre_review must be registered as a tool."""
    adv_mod = _get_advisory_module()
    names = [t.name for t in adv_mod.get_tools()]
    assert "advisory_review" in names


def test_review_status_registered():
    """review_status must be registered as a tool."""
    adv_mod = _get_advisory_module()
    names = [t.name for t in adv_mod.get_tools()]
    assert "review_status" in names


def test_advisory_gate_in_repo_commit_push():
    """The shared reviewed stage must gate review on advisory freshness (the
    check lives in the extracted _advisory_and_tests_gate helper, called before
    any paid dispatch and after the free Max-Review-Cycles gate)."""
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._run_reviewed_stage_cycle)
    gate_pos = source.find("_advisory_and_tests_gate")
    review_pos = source.find("_run_parallel_review")
    assert gate_pos != -1, "_advisory_and_tests_gate not found in _run_reviewed_stage_cycle"
    assert review_pos != -1, "_run_parallel_review not found in _run_reviewed_stage_cycle"
    assert gate_pos < review_pos, "Advisory gate must precede parallel review"
    gate_source = inspect.getsource(git_mod._advisory_and_tests_gate)
    assert "_check_advisory_freshness" in gate_source
    # Verify _run_parallel_review contains the triad phases (Q25-A: assembly
    # before dispatch superseded the single _run_unified_review call).
    parallel_source = inspect.getsource(git_mod._run_parallel_review)
    assert "_prepare_unified_review" in parallel_source
    assert "_dispatch_unified_review" in parallel_source


def test_advisory_freshness_blocks_without_fresh_run(tmp_path):
    """_check_advisory_freshness must return ADVISORY_PRE_REVIEW_REQUIRED if no fresh run."""
    git_mod = _get_git_module()

    class FakeCtx:
        repo_dir = tmp_path
        drive_root = tmp_path
        task_id = "test-task"
        def drive_logs(self):
            logs = tmp_path / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            return logs

    # Initialize a bare git repo so compute_snapshot_hash works
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    result = git_mod._check_advisory_freshness(FakeCtx(), "test commit message")
    assert result is not None
    assert "ADVISORY_PRE_REVIEW_REQUIRED" in result


def test_advisory_freshness_passes_with_fresh_run(tmp_path):
    """_check_advisory_freshness must return None when a fresh run exists."""
    import subprocess
    git_mod = _get_git_module()
    rs_mod = _get_review_state_module()

    # Separate repo_dir and drive_root so drive data doesn't pollute git status
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    (drive_root / "state").mkdir()
    (drive_root / "logs").mkdir()

    # Init git repo in repo_dir
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

    commit_message = "test commit"

    class FakeCtx:
        pass
    ctx = FakeCtx()
    ctx.repo_dir = repo_dir
    ctx.drive_root = drive_root
    ctx.task_id = "test-task"
    ctx.drive_logs = lambda: drive_root / "logs"

    # advisory_review.json is excluded from snapshot hash (see _SNAPSHOT_EXCLUDE_PATHS)
    # drive_root is outside repo_dir so no git pollution
    snapshot_hash = rs_mod.compute_snapshot_hash(repo_dir, commit_message)

    # Inject a fresh run with that exact hash
    state = rs_mod.AdvisoryReviewState()
    state.add_run(rs_mod.AdvisoryRunRecord(
        snapshot_hash=snapshot_hash,
        commit_message=commit_message,
        status="fresh",
        ts="2026-01-01T00:00:00",
    ))
    rs_mod.save_state(drive_root, state)

    # Hash is stable — drive_root is outside repo_dir, no git status pollution
    result = git_mod._check_advisory_freshness(ctx, commit_message)
    assert result is None, f"Expected gate to pass but got: {result}"


def test_advisory_freshness_blocks_on_open_commit_readiness_debt(tmp_path, monkeypatch):
    """Fresh advisory is not enough when commit-readiness debt remains open."""
    import subprocess

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    git_mod = _get_git_module()
    rs_mod = _get_review_state_module()

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    (drive_root / "state").mkdir()
    (drive_root / "logs").mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

    commit_message = "test commit"
    snapshot_hash = rs_mod.compute_snapshot_hash(repo_dir, commit_message)
    repo_key = rs_mod.make_repo_key(repo_dir)

    state = rs_mod.AdvisoryReviewState()
    state.add_run(rs_mod.AdvisoryRunRecord(
        snapshot_hash=snapshot_hash,
        commit_message=commit_message,
        status="fresh",
        ts="2026-01-01T00:00:00",
        repo_key=repo_key,
        readiness_warnings=["Manual verification still required before commit."],
    ))
    state._sync_commit_readiness_debts(repo_key=repo_key)
    assert len(state.get_open_commit_readiness_debts(repo_key=repo_key)) == 1
    rs_mod.save_state(drive_root, state)

    class FakeCtx:
        pass

    ctx = FakeCtx()
    ctx.repo_dir = repo_dir
    ctx.drive_root = drive_root
    ctx.task_id = "test-task"
    ctx.drive_logs = lambda: drive_root / "logs"

    result = git_mod._check_advisory_freshness(ctx, commit_message)
    assert result is not None
    assert "ADVISORY_PRE_REVIEW_REQUIRED" in result
    assert "Commit-readiness debt" in result


def test_advisory_obligations_acknowledged_under_advisory_enforcement(tmp_path, monkeypatch):
    """Fresh advisory downgrades obligations/debt under advisory enforcement and audits it."""
    import subprocess

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    git_mod = _get_git_module()
    rs_mod = _get_review_state_module()

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    (drive_root / "state").mkdir()
    (drive_root / "logs").mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

    commit_message = "test commit"
    snapshot_hash = rs_mod.compute_snapshot_hash(repo_dir, commit_message)
    repo_key = rs_mod.make_repo_key(repo_dir)

    state = rs_mod.AdvisoryReviewState()
    state.add_run(rs_mod.AdvisoryRunRecord(
        snapshot_hash=snapshot_hash,
        commit_message=commit_message,
        status="fresh",
        ts="2026-01-01T00:00:00",
        repo_key=repo_key,
        readiness_warnings=["Manual verification still required before commit."],
    ))
    state.add_blocking_attempt(rs_mod.CommitAttemptRecord(
        ts="2026-01-01T00:05:00",
        commit_message="blocked commit",
        status="blocked",
        repo_key=repo_key,
        block_reason="critical_findings",
        critical_findings=[{
            "item": "tests_affected",
            "verdict": "FAIL",
            "severity": "critical",
            "reason": "missing tests",
        }],
    ))
    state.open_obligations = [
        rs_mod.ObligationItem(
            obligation_id=f"obl-{idx:04d}",
            item=f"item_{idx}",
            severity="critical",
            reason=f"missing tests {idx}",
            source_attempt_ts="2026-01-01T00:05:00",
            source_attempt_msg="blocked commit",
            repo_key=repo_key,
        )
        for idx in range(1, 7)
    ]
    state.commit_readiness_debts = [
        rs_mod.CommitReadinessDebtItem(
            debt_id=f"crd-{idx:04d}",
            category=f"category_{idx}",
            summary=f"readiness debt {idx}",
            repo_key=repo_key,
        )
        for idx in range(1, 7)
    ]
    assert state.get_open_obligations(repo_key=repo_key)
    assert state.get_open_commit_readiness_debts(repo_key=repo_key)
    rs_mod.save_state(drive_root, state)

    class FakeCtx:
        pass

    ctx = FakeCtx()
    ctx.repo_dir = repo_dir
    ctx.drive_root = drive_root
    ctx.task_id = "test-task"
    ctx.drive_logs = lambda: drive_root / "logs"

    result = git_mod._check_advisory_freshness(ctx, commit_message)

    assert result is None
    events = [
        json.loads(line)
        for line in (drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event = [item for item in events if item.get("type") == "advisory_obligations_acknowledged"][0]
    assert event["snapshot_hash"] == snapshot_hash
    assert event["repo_key"] == repo_key
    assert event["open_obligations_count"] == 6
    assert event["open_debts_count"] >= 6
    assert len(event["open_obligations"]) == event["open_obligations_count"]
    assert len(event["open_debts"]) == event["open_debts_count"]
    assert any("obl-0006" in item for item in event["open_obligations"])
    assert any("crd-0006" in item for item in event["open_debts"])


def test_advisory_freshness_is_repo_scoped(tmp_path):
    """A fresh run for repo A must not satisfy repo B when hashes coincide."""
    import subprocess
    git_mod = _get_git_module()
    rs_mod = _get_review_state_module()

    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    (drive_root / "state").mkdir()
    (drive_root / "logs").mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_a), capture_output=True)
    subprocess.run(["git", "init"], cwd=str(repo_b), capture_output=True)

    commit_message = "same commit message"
    snapshot_hash = rs_mod.compute_snapshot_hash(repo_a, commit_message)
    state = rs_mod.AdvisoryReviewState()
    state.add_run(rs_mod.AdvisoryRunRecord(
        snapshot_hash=snapshot_hash,
        commit_message=commit_message,
        status="fresh",
        ts="2026-01-01T00:00:00",
        repo_key=rs_mod.make_repo_key(repo_a),
    ))
    rs_mod.save_state(drive_root, state)

    class FakeCtx:
        pass

    ctx = FakeCtx()
    ctx.repo_dir = repo_b
    ctx.drive_root = drive_root
    ctx.task_id = "repo-b-task"
    ctx.drive_logs = lambda: drive_root / "logs"

    result = git_mod._check_advisory_freshness(ctx, commit_message)
    assert result is not None
    assert "ADVISORY_PRE_REVIEW_REQUIRED" in result


def test_open_obligations_are_repo_scoped(tmp_path):
    """Open obligations in repo A must not block a fresh advisory in repo B."""
    import subprocess
    git_mod = _get_git_module()
    rs_mod = _get_review_state_module()

    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    (drive_root / "state").mkdir()
    (drive_root / "logs").mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_a), capture_output=True)
    subprocess.run(["git", "init"], cwd=str(repo_b), capture_output=True)

    commit_message = "shared message"
    state = rs_mod.AdvisoryReviewState()
    state.add_run(rs_mod.AdvisoryRunRecord(
        snapshot_hash=rs_mod.compute_snapshot_hash(repo_b, commit_message),
        commit_message=commit_message,
        status="fresh",
        ts="2026-01-01T00:00:00",
        repo_key=rs_mod.make_repo_key(repo_b),
    ))
    state.add_blocking_attempt(rs_mod.CommitAttemptRecord(
        ts="2026-01-01T00:05:00",
        commit_message="repo a blocked",
        status="blocked",
        repo_key=rs_mod.make_repo_key(repo_a),
        block_reason="critical_findings",
        critical_findings=[{
            "item": "tests_affected",
            "verdict": "FAIL",
            "severity": "critical",
            "reason": "missing tests in repo a",
        }],
    ))
    rs_mod.save_state(drive_root, state)

    class FakeCtx:
        pass

    ctx = FakeCtx()
    ctx.repo_dir = repo_b
    ctx.drive_root = drive_root
    ctx.task_id = "repo-b-task"
    ctx.drive_logs = lambda: drive_root / "logs"

    result = git_mod._check_advisory_freshness(ctx, commit_message)
    assert result is None, f"Repo-scoped obligations should not block repo B: {result}"


def test_snapshot_hash_stable_on_message_change(tmp_path):
    """Snapshot hash must NOT differ when only commit_message changes.

    Hash is now based on code content only (decoupled from commit_message
    to make freshness less brittle when the message is slightly rephrased).
    """
    import subprocess
    rs_mod = _get_review_state_module()
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)

    h1 = rs_mod.compute_snapshot_hash(tmp_path, "message A")
    h2 = rs_mod.compute_snapshot_hash(tmp_path, "message B")
    assert h1 == h2


def test_bypass_is_audited(tmp_path):
    """Bypassing advisory gate must write advisory_review_bypassed to events.jsonl."""
    import json
    import subprocess
    git_mod = _get_git_module()
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    class FakeCtx:
        repo_dir = tmp_path
        drive_root = tmp_path
        task_id = "bypass-task"
        def drive_logs(self):
            return tmp_path / "logs"

    result = git_mod._check_advisory_freshness(
        FakeCtx(), "bypassed commit", skip_advisory_pre_review=True
    )
    assert result is None  # bypass passes

    events_path = tmp_path / "logs" / "events.jsonl"
    assert events_path.exists(), "events.jsonl must exist after bypass"
    events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
    bypass_events = [e for e in events if e.get("type") == "advisory_review_bypassed"]
    assert len(bypass_events) == 1, "Exactly one bypass event must be logged"
    assert bypass_events[0]["task_id"] == "bypass-task"


def test_advisory_pre_review_tool_schema_has_skip_param():
    """advisory_review schema must expose skip_advisory_review param."""
    adv_mod = _get_advisory_module()
    tools = adv_mod.get_tools()
    adv_tool = next(t for t in tools if t.name == "advisory_review")
    props = adv_tool.schema["parameters"]["properties"]
    assert "skip_advisory_review" in props
    assert props["skip_advisory_review"].get("default") is False


def test_repo_commit_schema_has_skip_advisory_param():
    """commit_reviewed schema must expose skip_advisory_review param."""
    git_mod = _get_git_module()
    tools = git_mod.get_tools()
    commit_tool = next(t for t in tools if t.name == "commit_reviewed")
    props = commit_tool.schema["parameters"]["properties"]
    assert "skip_advisory_review" in props


def test_advisory_choice_guidance_is_shared_across_model_facing_schemas():
    adv_mod = _get_advisory_module()
    git_mod = _get_git_module()
    advisory_tools = {tool.name: tool for tool in adv_mod.get_tools()}
    git_tools = {tool.name: tool for tool in git_mod.get_tools()}

    advisory_tool = advisory_tools["preflight_review"]
    status_tool = advisory_tools["review_status"]
    commit_tool = git_tools["commit_reviewed"]
    alias_tool = git_tools["vcs_commit_reviewed"]
    advisory_skip = advisory_tool.schema["parameters"]["properties"]["skip_advisory_review"]
    commit_skip = commit_tool.schema["parameters"]["properties"]["skip_advisory_review"]
    alias_skip = alias_tool.schema["parameters"]["properties"]["skip_advisory_review"]

    guidance = adv_mod.ADVISORY_REVIEW_CHOICE_GUIDANCE
    surfaces = [
        advisory_tool.schema["description"],
        advisory_skip["description"],
        status_tool.schema["description"],
        commit_tool.schema["description"],
        commit_skip["description"],
        alias_tool.schema["description"],
        alias_skip["description"],
    ]
    assert all(guidance in surface for surface in surfaces)
    assert all("skip_advisory_review=True" in surface for surface in surfaces)
    assert "bypasses only the requirements for advisory freshness" in guidance
    assert "records remain visible" in guidance
    assert "removes only advisory" not in guidance
    assert commit_tool.schema["description"] == alias_tool.schema["description"]
    assert commit_skip["description"] == alias_skip["description"]
    assert "advisory-readiness projection" in status_tool.schema["description"]
    assert "not the full commit gate" in status_tool.schema["description"]
    assert "bypass the entire commit gate" not in " ".join(surfaces).lower()


def test_advisory_auto_bypass_on_missing_key(tmp_path, monkeypatch):
    """advisory_pre_review auto-bypasses with audit when the advisory model's
    provider credentials are absent (the retired ANTHROPIC_API_KEY probe's
    successor: availability follows the routed model)."""
    import json
    import subprocess
    adv_mod = _get_advisory_module()
    rs_mod = _get_review_state_module()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    (drive_root / "state").mkdir()
    (drive_root / "logs").mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    for _key in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                 "MINIMAX_API_KEY", "GIGACHAT_AUTH_KEY", "CLOUD_RU_API_KEY"):
        monkeypatch.delenv(_key, raising=False)
    progress_calls = []

    class FakeCtx:
        pass
    ctx = FakeCtx()
    ctx.repo_dir = str(repo_dir)
    ctx.drive_root = str(drive_root)
    ctx.task_id = "autobypass-task"
    ctx.drive_logs = lambda: drive_root / "logs"
    ctx.emit_progress_fn = lambda msg: progress_calls.append(msg)

    result_raw = adv_mod._handle_advisory_pre_review(ctx, commit_message="test commit")
    result = json.loads(result_raw)

    # Must be bypassed, not errored
    assert result["status"] == "bypassed"
    assert "no provider credentials" in result["bypass_reason"]

    # Must create a fresh advisory state (bypassed counts as fresh for gate)
    state = rs_mod.load_state(drive_root)
    assert state.latest() is not None
    assert state.latest().status == "bypassed"

    # Must audit bypass to events.jsonl
    events_path = drive_root / "logs" / "events.jsonl"
    assert events_path.exists(), "events.jsonl must exist after auto-bypass"
    events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
    bypass_events = [e for e in events if e.get("type") == "advisory_review_bypassed"]
    assert len(bypass_events) == 1
    assert "no provider credentials" in bypass_events[0]["bypass_reason"]


def test_advisory_prompt_contains_blocking_history_when_blocked(tmp_path):
    """Advisory prompt must include blocking history section when last commit was blocked."""
    import subprocess
    adv_mod = _get_advisory_module()
    rs_mod = _get_review_state_module()

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    (drive_root / "state").mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

    # Create a blocked commit attempt with structured critical findings
    state = rs_mod.AdvisoryReviewState()
    attempt = rs_mod.CommitAttemptRecord(
        ts="2026-04-02T22:00:00",
        commit_message="test blocked commit",
        status="blocked",
        block_reason="critical_findings",
        block_details=(
            "⚠️ REVIEW_BLOCKED: Critical issues found.\n"
            "  CRITICAL: [gpt-5.5] bible_compliance: Missing BIBLE.md update\n"
            "  CRITICAL: [gpt-5.5] tests_affected: No tests for new function\n"
            "  WARN: [opus] self_consistency: Minor doc drift"
        ),
        critical_findings=[
            {"verdict": "FAIL", "severity": "critical",
             "item": "bible_compliance", "reason": "Missing BIBLE.md update", "model": "m"},
            {"verdict": "FAIL", "severity": "critical",
             "item": "tests_affected", "reason": "No tests for new function", "model": "m"},
        ],
    )
    state.add_blocking_attempt(attempt)
    rs_mod.save_state(drive_root, state)

    # Build the advisory prompt with drive_root
    prompt = adv_mod._build_advisory_prompt(
        repo_dir, "test commit", drive_root=drive_root
    )

    # Must contain obligations section (new format)
    assert "Unresolved obligations" in prompt
    assert "bible_compliance" in prompt
    assert "tests_affected" in prompt
    assert "should explicitly address" in prompt


def test_advisory_prompt_no_blocking_history_when_succeeded(tmp_path):
    """Advisory prompt must NOT include blocking history when last commit succeeded."""
    import subprocess
    adv_mod = _get_advisory_module()
    rs_mod = _get_review_state_module()

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    (drive_root / "state").mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

    state = rs_mod.AdvisoryReviewState()
    state.attempts = [rs_mod.CommitAttemptRecord(
        ts="2026-04-02T22:00:00",
        commit_message="test commit",
        status="succeeded",
    )]
    rs_mod.save_state(drive_root, state)

    prompt = adv_mod._build_advisory_prompt(
        repo_dir, "test commit", drive_root=drive_root
    )

    assert "## Unresolved obligations from previous blocking rounds" not in prompt


def test_advisory_prompt_no_blocking_history_without_drive_root(tmp_path):
    """Advisory prompt must gracefully skip blocking history when no drive_root."""
    import subprocess
    adv_mod = _get_advisory_module()

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

    prompt = adv_mod._build_advisory_prompt(repo_dir, "test commit")
    assert "## Unresolved obligations from previous blocking rounds" not in prompt


def test_advisory_prompt_strictness_formulations():
    """Advisory prompt must contain the same strictness language as blocking reviewers."""
    import subprocess
    adv_mod = _get_advisory_module()

    import pathlib as _pl
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        repo_dir = _pl.Path(d)
        (repo_dir / "BIBLE.md").write_text("test bible", encoding="utf-8")
        subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

        prompt = adv_mod._build_advisory_prompt(repo_dir, "test commit")

        # Key strictness formulations that must be present
        assert "same rigor" in prompt.lower() or "same severity threshold" in prompt.lower()
        assert "do not stop after finding the first issue" in prompt.lower()
        assert "distinct problem" in prompt.lower()
        assert "read the full content of every changed file" in prompt.lower()
        assert "all bugs, logic errors" in prompt.lower()
        # Must NOT contain the old relaxing language
        assert "findings do not directly block" not in prompt.lower()


def test_advisory_prompt_references_architecture_doc_via_read_tool():
    """Advisory prompt must inline ARCHITECTURE.md content when available.

    The v4.15.1 prompt restores ARCHITECTURE.md directly into the advisory context so
    the reviewer always sees version-sync and module-structure facts without an extra
    read step. The touched-file pack must avoid duplicating it separately.
    """
    import subprocess
    adv_mod = _get_advisory_module()

    import pathlib as _pl
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        repo_dir = _pl.Path(d)
        (repo_dir / "BIBLE.md").write_text("test bible", encoding="utf-8")
        (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v99.0.0 — Architecture", encoding="utf-8"
        )
        subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

        prompt = adv_mod._build_advisory_prompt(repo_dir, "test commit")

        assert "ARCHITECTURE.md" in prompt, "Prompt must include an ARCHITECTURE.md section"
        assert "## ARCHITECTURE.md" in prompt, "Prompt should expose ARCHITECTURE.md as a first-class section"
        assert "Ouroboros v99.0.0" in prompt, (
            "ARCHITECTURE.md content should now be inlined for advisory review"
        )


def test_advisory_prompt_strictness_concrete_fix_requirement():
    """Advisory prompt must require concrete fix suggestions for FAIL findings."""
    import subprocess
    adv_mod = _get_advisory_module()

    import pathlib as _pl
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        repo_dir = _pl.Path(d)
        subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

        prompt = adv_mod._build_advisory_prompt(repo_dir, "test commit")

        # Must require actionable fix suggestions
        assert "concrete" in prompt.lower()
        assert "fix" in prompt.lower()
        assert "how to fix" in prompt.lower() or "how to change" in prompt.lower() or "what to change" in prompt.lower()


def test_blocking_history_section_with_scope_blocked(tmp_path):
    """Blocking history should also work for scope_blocked commits."""
    adv_mod = _get_advisory_module()
    rs_mod = _get_review_state_module()

    drive_root = tmp_path
    (drive_root / "state").mkdir(parents=True)

    state = rs_mod.AdvisoryReviewState()
    attempt = rs_mod.CommitAttemptRecord(
        ts="2026-04-02T22:00:00",
        commit_message="scope blocked commit",
        status="blocked",
        block_reason="scope_blocked",
        block_details=(
            "⚠️ SCOPE_REVIEW_BLOCKED: Missing touchpoint.\n"
            "CRITICAL: [opus] forgotten_touchpoints: ARCHITECTURE.md not updated"
        ),
        critical_findings=[
            {"verdict": "FAIL", "severity": "critical",
             "item": "forgotten_touchpoints", "reason": "ARCHITECTURE.md not updated", "model": "opus"},
        ],
    )
    state.add_blocking_attempt(attempt)
    rs_mod.save_state(drive_root, state)

    section = adv_mod._build_blocking_history_section(drive_root)
    assert "Unresolved obligations" in section
    assert "scope_blocked" in section
    assert "ARCHITECTURE.md" in section


def test_review_blocked_message_prefers_fix_over_rebuttal():
    """REVIEW_BLOCKED coaching (issue #447, В8=A): fix first; rebuttal is legitimate
    for factual errors, unsupported severity, or disproportionate remedies — but it
    never overrides owner-chosen enforcement, and a repeated finding means fix."""
    from ouroboros.tools.review import _build_critical_block_message

    class FakeCtx:
        _review_iteration_count = 1
        _review_history = []

    msg = _build_critical_block_message(
        FakeCtx(), "test commit", ["bible_compliance: violation"], [], ""
    )
    # Whitespace-normalized: the message wraps lines mid-phrase.
    lowered = " ".join(msg.lower().split())
    assert "factually incorrect" in lowered
    # Proportionality channel is open: disproportionate remedies are arguable.
    assert "disproportionate" in lowered
    # Non-override clause: rebuttal is argument, not authority.
    assert "never overrides owner-chosen enforcement" in lowered
    # Fix-on-repeat coaching survives the replacement.
    assert "implement the fix" in lowered


def test_review_blocked_5plus_hint_suggests_split():
    """v4.9.2: After 5+ attempts, hint suggests implementing the fix or splitting."""
    from ouroboros.tools.review import _build_critical_block_message

    class FakeCtx:
        # v4.33.0 lowered the threshold from 5 to 3 — 5 still triggers but
        # the phrasing changed from "report the blockage" to "send_user_message
        # to escalate" which carries the same semantic weight.
        _review_iteration_count = 5
        _review_history = []

    msg = _build_critical_block_message(
        FakeCtx(), "test commit", ["tests_affected: missing tests"], [], ""
    )
    lowered = msg.lower()
    assert "split" in lowered, f"missing split-the-diff guidance: {msg!r}"
    assert ("send_user_message" in lowered or "escalate" in lowered
            or "report" in lowered), (
        f"missing escalation guidance: {msg!r}"
    )


def test_review_blocked_message_requires_reaudit_after_first_block():
    """Blocked-review guidance should explicitly require a full-diff re-audit after the first block."""
    from ouroboros.tools.review import _build_critical_block_message

    class FakeCtx:
        _review_iteration_count = 2
        _review_history = []
        _last_review_critical_findings = [{"item": "code_quality"}]
        _last_review_advisory_findings = []

    msg = _build_critical_block_message(
        FakeCtx(), "test commit", ["code_quality: review mismatch"], [], ""
    )
    lowered = msg.lower()
    assert "re-read the full diff" in lowered
    assert "group obligations by root cause" in lowered
    assert "rewrite the plan" in lowered


def test_self_consistency_listed_as_critical_in_severity_rules():
    """self_consistency (item 13) must be treated as conditionally critical, not always advisory."""
    import pathlib
    checklists_path = pathlib.Path(__file__).parent.parent / "docs" / "CHECKLISTS.md"
    content = checklists_path.read_text(encoding="utf-8")

    # The severity rules section must describe self_consistency as conditionally critical
    assert "self_consistency" in content
    # Must NOT say items 11-13 are ALL advisory
    lines = content.split("\n")
    for line in lines:
        if "items 11-13 are advisory" in line.lower():
            raise AssertionError(
                f"Found old 'items 11-13 are advisory' rule — self_consistency "
                f"must now be conditionally critical:\n  {line}"
            )
    # Must say item 13 is conditionally critical
    assert "item 13" in content.lower() and "critical" in content.lower()
    # v4.33.0: the old "README test counts" example was folded into the
    # broader Critical surface whitelist. Narrative / prose / commentary
    # mismatches outside the whitelist must be explicitly advisory.
    assert "Critical surface whitelist" in content
    assert "advisory" in content.lower()
    # And the "narrative" framing of commit-message / doc wording remains.
    assert "narrative" in content.lower()


def test_development_compliance_checklist_expanded():
    """development_compliance description must include specific concrete checks."""
    import pathlib
    checklists_path = pathlib.Path(__file__).parent.parent / "docs" / "CHECKLISTS.md"
    content = checklists_path.read_text(encoding="utf-8")

    # All these concrete checks must appear in the checklist
    required_terms = [
        "snake_case",
        "PascalCase",
        "Gateway",
        "LLMClient",
        "[:N]",
        "ToolEntry",
    ]
    for term in required_terms:
        assert term in content, (
            f"development_compliance checklist must mention '{term}' for concrete checks, "
            f"but it's missing from CHECKLISTS.md"
        )


# test_triad_review_prompt_has_thoroughness_instructions and
# test_triad_review_reasoning_effort_is_medium_not_low removed in v5.15.x —
# both pinned exact prompt-template / inspect.getsource() substrings.
# Prompt quality and effort level evolve over time; the behavioral
# contract (review produces correct verdicts at adequate depth) is
# exercised by the actual triad-review integration tests in
# test_review_fidelity.py, test_review_observability.py, and the
# git+review pipeline suite.


def test_advisory_prompt_contains_obligation_targeting_instructions(tmp_path):
    """_build_advisory_prompt must instruct the reviewer how to target a specific
    obligation when multiple open obligations share the same checklist item.
    Without this, a generic item-name PASS cannot disambiguate which obligation
    was resolved, and the resolution logic leaves all same-item obligations open.
    """
    import tempfile
    import pathlib as _pl
    import subprocess as _sp
    adv_mod = _get_advisory_module()

    with tempfile.TemporaryDirectory() as d:
        repo_dir = _pl.Path(d)
        _sp.run(["git", "init"], cwd=str(repo_dir), capture_output=True)

        prompt = adv_mod._build_advisory_prompt(repo_dir, "test commit")

        # Must explain the (obligation <id>) suffix mechanism
        assert "obligation" in prompt.lower(), (
            "Prompt must mention 'obligation' targeting to allow per-finding resolution"
        )
        assert "(obligation" in prompt, (
            "Prompt must show the '(obligation <id>)' suffix syntax for targeting specific obligations"
        )
        # Must warn that a generic PASS won't resolve all same-item obligations
        assert "will NOT resolve" in prompt or "will not resolve" in prompt.lower(), (
            "Prompt must warn that generic item-name PASS won't resolve all same-item obligations"
        )
