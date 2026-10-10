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
    source = inspect.getsource(_get_git_module()._preflight_and_tests_gate)
    assert '"tests_preflight_blocked"' in source
    idx = source.find("_record_commit_attempt(")
    assert idx != -1
    # The phase stamp must live in the same _record_commit_attempt call.
    window = source[idx:idx + 400]
    assert "block_reason=reason" in window and 'phase="preflight"' in window


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

    ctx = types.SimpleNamespace(
        repo_dir="/tmp/repo", drive_root=tmp_path,
        _coupling_review_history={"snap": [{"round": 1, "status": "FAIL"}]},
    )
    outcome = git_mod._run_non_committing_review_cycle(ctx, "test commit")

    assert outcome["status"] == "passed"
    assert "Commit was not created" in outcome["message"]
    # A completed cycle closes the subject's coupling rounds; the next starts fresh.
    assert ctx._coupling_review_history == {}
    assert recorded == [{"status": "reviewed", "phase": "review_only"}]
    assert released == ["lock-token"]
    assert reset_calls == [(("git", "reset", "HEAD"), "/tmp/repo")]


def test_non_committing_review_cycle_runtime_unstages_on_block(monkeypatch, tmp_path):
    git_mod = _get_git_module()
    reset_calls = []
    released = []

    monkeypatch.setattr(git_mod, "_check_overlapping_review_attempt", lambda ctx: None)
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
# The author's preflight (decision 3A): the wrapper names and the commit gate
# ---------------------------------------------------------------------------

def _get_preflight_module():
    sys.path.insert(0, REPO)
    return importlib.import_module("ouroboros.tools.preflight_review")


def _get_review_state_module():
    sys.path.insert(0, REPO)
    return importlib.import_module("ouroboros.review_state")


def test_advisory_pre_review_registered():
    """The old advisory_review name stays callable as an alias of preflight_review."""
    names = [t.name for t in _get_preflight_module().get_tools()]
    assert "advisory_review" in names and "preflight_review" in names


def test_review_status_registered():
    """review_status must be registered as a tool."""
    names = [t.name for t in _get_preflight_module().get_tools()]
    assert "review_status" in names


def test_preflight_gate_in_repo_commit_push():
    """The shared reviewed stage runs the free checks, the tests and the optional named
    preflight (the extracted _preflight_and_tests_gate helper) before any paid dispatch;
    no advisory freshness is read any more."""
    from ouroboros.tools import git_review_cycle

    git_mod = _get_git_module()
    # `_run_reviewed_stage_cycle` runs the cycle body under the commit's composed panel.
    assert "_reviewed_stage_cycle(" in inspect.getsource(git_mod._run_reviewed_stage_cycle)
    source = inspect.getsource(git_review_cycle._reviewed_stage_cycle)
    gate_pos = source.find("_preflight_and_tests_gate")
    review_pos = source.find("_run_parallel_review")
    assert gate_pos != -1, "_preflight_and_tests_gate not found in _run_reviewed_stage_cycle"
    assert review_pos != -1, "_run_parallel_review not found in _run_reviewed_stage_cycle"
    assert gate_pos < review_pos, "the preflight gate must precede parallel review"
    gate_source = inspect.getsource(git_mod._preflight_and_tests_gate)
    assert "deterministic_preflight" in gate_source and "run_commit_preflight" in gate_source
    assert "_check_advisory_freshness" not in gate_source
    # Verify _run_parallel_review contains the triad phases (Q25-A: assembly
    # before dispatch superseded the single _run_unified_review call).
    parallel_source = inspect.getsource(git_mod._run_parallel_review)
    assert "_prepare_unified_review" in parallel_source
    assert "_dispatch_unified_review" in parallel_source


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


def test_repo_commit_schema_has_skip_advisory_param():
    """commit_reviewed schema must expose skip_advisory_review param."""
    git_mod = _get_git_module()
    tools = git_mod.get_tools()
    commit_tool = next(t for t in tools if t.name == "commit_reviewed")
    props = commit_tool.schema["parameters"]["properties"]
    assert "skip_advisory_review" in props


def test_commit_schema_states_the_preflight_as_a_fact_not_a_bypass():
    """Without a named row the commit records preflight not_performed, and skipping records
    skipped: both are facts, never an audited bypass of advisory freshness."""
    git_tools = {tool.name: tool for tool in _get_git_module().get_tools()}
    commit, alias = git_tools["commit_reviewed"], git_tools["vcs_commit_reviewed"]
    assert commit.schema == {**alias.schema, "name": "commit_reviewed"}
    props = commit.schema["parameters"]["properties"]
    assert "not_performed" in commit.schema["description"]
    assert "preflight: skipped" in props["skip_advisory_review"]["description"]
    assert "TOOL_ARG_ERROR" in props["preflight_reviewer"]["description"]
    surfaces = " ".join((commit.schema["description"], props["skip_advisory_review"]["description"],
                         props["preflight_reviewer"]["description"])).lower()
    assert "audited bypass" not in surfaces and "advisory freshness" not in surfaces
    wrapper = next(t for t in _get_preflight_module().get_tools() if t.name == "preflight_review")
    assert "skip_advisory_review" not in wrapper.schema["parameters"]["properties"]


def test_review_blocked_message_keeps_evidence_based_rebuttal_on_repeat():
    """REVIEW_BLOCKED coaching (issue #447, В8=A): fix first; rebuttal is legitimate
    for factual errors, unsupported severity, or disproportionate remedies — but it
    never overrides owner-chosen enforcement. Repetition is not evidence."""
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
    assert "repetition alone does not validate a finding" in lowered
    assert "retaining a justified rebuttal" in lowered


def test_review_blocked_later_attempts_keep_judgment_without_numeric_stop():
    """Later attempts keep reassessment options as examples; no attempt number
    or fix count turns them into a mandatory STOP or message format."""
    from ouroboros.tools.review import _build_critical_block_message

    class FakeCtx:
        _review_iteration_count = 5
        _review_history = []

    msg = _build_critical_block_message(
        FakeCtx(), "test commit", ["tests_affected: missing tests"], [], ""
    )
    lowered = " ".join(msg.lower().split())
    assert "if attempts stop converging, reconsider the approach" in lowered
    assert "split the diff" in lowered and "escalate" in lowered
    for retired in ("circuit-breaker", "two concrete fixes", "stop retrying",
                    "one subject line", "attempt 5+"):
        assert retired not in lowered, retired


def test_review_blocked_retry_note_owes_outcome_not_procedure():
    """From attempt 2 the note keeps every open finding and the unchanged review
    state, defers paid/free retry to the gate's recorded eligibility, and leaves
    inspection/grouping/order to the author."""
    from ouroboros.tools.review import _build_critical_block_message
    from ouroboros.tools.review_prompt_text import REVIEW_REPAIR_JUDGMENT

    class FakeCtx:
        _review_iteration_count = 2
        _review_history = []
        _last_review_critical_findings = [
            {"item": "code_quality"}, {"item": "tests_affected"}]
        _last_review_advisory_findings = []

    msg = _build_critical_block_message(
        FakeCtx(), "test commit", ["code_quality: review mismatch"], [], ""
    )
    assert "REVIEW_BLOCKED (attempt 2)" in msg
    note = msg.split("Before the next commit_reviewed:", 1)[1]
    assert "  - Finding: code_quality" in note and "  - Finding: tests_affected" in note
    assert REVIEW_REPAIR_JUDGMENT in note
    lowered = " ".join(note.lower().split())
    assert "recorded review state and the individual findings below stand" in lowered
    assert "not rewrite them or the configured enforcement" in lowered
    assert ("follows the gate's recorded replay eligibility, custody, budget and "
            "cycle limit, not this note") in lowered
    assert ("an eligible recorded verdict on an unchanged diff under the same review "
            "contract is not re-reviewed without a genuinely new review_rebuttal") in lowered
    for retired in ("do not call commit_reviewed", "addressed / rebutted / pending",
                    "re-read the full diff", "group obligations by root cause",
                    "rewrite the plan", "another paid review needs"):
        assert retired not in lowered, retired


def test_self_consistency_listed_as_critical_in_severity_rules():
    """self_consistency (Ouroboros Body Layer item 15) must be treated as conditionally critical, not always advisory."""
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
    # Must say item 15 (self_consistency) is conditionally critical
    assert "item 15 (self_consistency) is conditionally critical" in content.lower()
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
