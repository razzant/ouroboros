"""Tests for git safety tools, commit gate hardening, and operational polish.

Verifies (Phase 4):
- New tools registered: pull_from_remote, restore_to_head, revert_commit
- SAFETY_CRITICAL_PATHS blocks dangerous operations
- Confirm gates prevent accidental destructive actions
- Auto-tagging on version bump
- Credential helper in git_ops (no token in remote URL)
- New tools in CORE_TOOL_NAMES

Verifies (Phase 5):
- Auto-push stays out of rollback tools and is best-effort
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
    classify as verdicts, quorum/fit/transport rows as infra; preflight/refusal
    rows stay unclassified."""
    import types as _types

    from ouroboros.tools.commit_gate import (
        BLOCK_CLASS_INFRA,
        BLOCK_CLASS_VERDICT,
        attempt_block_class,
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


def test_non_committing_review_cycle_exists_and_reuses_shared_stage_cycle():
    git_mod = _get_git_module()
    source = inspect.getsource(git_mod._run_non_committing_review_cycle)
    assert "_run_reviewed_stage_cycle" in source
    assert '"reviewed"' in source
    assert '"review_only"' in source
    assert '["git", "commit"' not in source


def _staged_whole_tree(ctx, outcome):
    """What the real stage cycle records after ``git add -A`` (paths omitted): the finally undoes it."""
    from ouroboros.tools.git_plumbing import WHOLE_INDEX

    ctx._staged_selection = WHOLE_INDEX
    return outcome


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
        lambda ctx, *args, **kwargs: _staged_whole_tree(ctx, {
            "status": "passed",
            "message": "stage cycle passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post"},
        }),
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
    assert reset_calls == [(("git", "reset", "-q", "HEAD", "--", ":/"), "/tmp/repo")]  # a path reset keeps a staged merge


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
        lambda ctx, *args, **kwargs: _staged_whole_tree(ctx, {
            "status": "blocked",
            "message": "review blocked",
            "block_reason": "critical_findings",
        }),
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
    assert reset_calls == [(("git", "reset", "-q", "HEAD", "--", ":/"), "/tmp/repo")]  # a path reset keeps a staged merge


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


# --- Confirm gates ---

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
# check superseded by the behavioral wiring test
# tests/test_evolution_publication.py::test_only_evolution_push_stays_under_git_lock.


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


# --- Credential configuration (legacy token-in-URL migration retired) ---

def test_configure_remote_remains_credential_helper_surface():
    git_ops = _get_git_ops_module()
    configure_source = inspect.getsource(git_ops.configure_remote)
    helper_source = inspect.getsource(git_ops._configure_credential_helper)
    assert "_configure_credential_helper" in configure_source
    assert ".git/credentials" in helper_source


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
    # Same arguments and the same complete contract; only the older name's lead says what it is.
    assert commit.schema["parameters"] == alias.schema["parameters"]
    assert alias.schema["description"].endswith(commit.schema["description"].split("workflow: ", 1)[1])
    props = commit.schema["parameters"]["properties"]
    assert "not_performed" in commit.schema["description"] and "not_performed" in alias.schema["description"]
    assert "preflight: skipped" in props["skip_advisory_review"]["description"]
    assert "TOOL_ARG_ERROR" in props["preflight_reviewer"]["description"]
    surfaces = " ".join((commit.schema["description"], props["skip_advisory_review"]["description"],
                         props["preflight_reviewer"]["description"])).lower()
    assert "audited bypass" not in surfaces and "advisory freshness" not in surfaces
    wrapper = next(t for t in _get_preflight_module().get_tools() if t.name == "preflight_review")
    assert "skip_advisory_review" not in wrapper.schema["parameters"]["properties"]


def test_review_blocked_message_keeps_evidence_based_rebuttal_on_repeat():
    """The retry note (issue #447, В8=A) after the commit block message left with Blocking: from
    the second round it keeps rebuttal an evidence-based answer to a finding (reasons, the record
    preserved), and repetition never stands in for evidence: an unchanged subject under the same
    contract is not re-reviewed without a genuinely new rebuttal. Round one adds nothing."""
    from ouroboros.tools.review_prompt_text import build_self_verification_template

    assert build_self_verification_template([{"item": "bible_compliance"}], attempt_idx=1) == ""
    note = build_self_verification_template([{"item": "bible_compliance"}], attempt_idx=2,
                                            tool_name="skill_review", context_noun="skill pack")
    lowered = " ".join(note.lower().split())
    assert "rebut unsupported findings or their current applicability with reasons" in lowered
    assert "preserving the record and anything unresolved" in lowered
    assert ("is not re-reviewed without a genuinely new review_rebuttal") in lowered
    assert "before the next skill_review" in lowered and "unchanged skill pack" in lowered


def test_review_blocked_later_attempts_keep_judgment_without_numeric_stop():
    """Later attempts keep reassessment options as examples; no attempt number
    or fix count turns them into a mandatory STOP or message format."""
    from ouroboros.tools.review_prompt_text import build_self_verification_template

    note = build_self_verification_template([{"item": "tests_affected"}], attempt_idx=5)
    lowered = " ".join(note.lower().split())
    assert "if attempts stop converging, reconsider the approach" in lowered
    assert "split the diff" in lowered and "escalate" in lowered
    for retired in ("circuit-breaker", "two concrete fixes", "stop retrying",
                    "one subject line", "attempt 5+"):
        assert retired not in lowered, retired


def test_review_blocked_retry_note_owes_outcome_not_procedure():
    """From attempt 2 the note keeps every open finding and the unchanged review
    state, defers paid/free retry to the gate's recorded eligibility, and leaves
    inspection/grouping/order to the author."""
    from ouroboros.tools.review_prompt_text import REVIEW_REPAIR_JUDGMENT, build_self_verification_template

    note = build_self_verification_template([{"item": "code_quality"}, {"item": "tests_affected"}], attempt_idx=2)
    note = note.split("Before the next commit_reviewed:", 1)[1]
    assert "  - Finding: code_quality" in note and "  - Finding: tests_affected" in note
    assert REVIEW_REPAIR_JUDGMENT in note
    lowered = " ".join(note.lower().split())
    assert "recorded review state and the individual findings below stand" in lowered
    assert "your response does not rewrite them." in lowered and "enforcement" not in lowered
    assert ("follows the gate's recorded replay eligibility, custody, budget and "
            "cycle limit, not this note") in lowered
    assert ("an eligible recorded verdict on an unchanged diff under the same review "
            "contract is not re-reviewed without a genuinely new review_rebuttal") in lowered
    for retired in ("do not call commit_reviewed", "addressed / rebutted / pending",
                    "re-read the full diff", "group obligations by root cause",
                    "rewrite the plan", "another paid review needs"):
        assert retired not in lowered, retired


# test_triad_review_prompt_has_thoroughness_instructions and
# test_triad_review_reasoning_effort_is_medium_not_low removed in v5.15.x —
# both pinned exact prompt-template / inspect.getsource() substrings.
# Prompt quality and effort level evolve over time; the behavioral
# contract (review produces correct verdicts at adequate depth) is
# exercised by the actual triad-review integration tests in
# test_review_fidelity.py, test_review_observability.py, and the
# git+review pipeline suite.
