"""Post-commit phase of a managed-update assisted merge: the blocking tests gate,
the smoke finish, owner-Pause retention of the committed merge, and Resume of that
exact retained commit.

The update transaction (``supervisor.update_merge``) stores the ``postcommit_resume``
receipt; Pause owes the remaining gates, never another commit or a rollback.
``tools.git`` re-exports these functions; ``_git()`` preserves its patchable facade
bindings."""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, Optional, Tuple

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result


def _git():
    """Read the public facade at call time so monkeypatches keep one binding."""
    from ouroboros.tools import git

    return git


def _managed_post_commit_tests_gate(
    ctx, commit_message: str, commit_start: float, skip_tests: bool,
    test_warning_ref, managed_tx: Dict[str, Any],
    fingerprints: Tuple[Dict[str, Any], Dict[str, Any]] = ({}, {}),
) -> Optional[str]:
    """BLOCKING post-commit test gate for managed-update merges only: a failed
    suite rolls the assisted merge back instead of shipping a warning (ordinary
    commits keep the warning-only contract later in the flow). The gate is
    MANDATORY: neither the caller's skip_tests nor OUROBOROS_PRE_PUSH_TESTS=0
    can wave a managed merge through untested. The shared runner reuses a
    PROCESS-HELD proof only when candidate files, source index, HEAD and the
    effective test/environment contract match, after the distinct post-commit
    baseline checks. A commit changes HEAD and requires a fresh run; repeated
    checks of the same subject may reuse it. The authority is the ctx record;
    durable ``tests_evidence`` tx copy is resolver-writable forensics and a
    forged tree there never suppresses this run; a restart loses the proof
    and requires a fresh run. The terminal record carries the
    same review metadata/fingerprints as every sibling failure record, so an
    operator can reconstruct WHICH reviewed revision the gate rejected."""
    if not managed_tx:
        return None
    del skip_tests  # deliberately ignored for managed merges
    # The shared runner rechecks the post-commit baseline before comparing the
    # complete workload. A tree-only fast path here would skip both checks.
    from ouroboros.owner_pause import OwnerPauseRefused, run_operation
    try:
        post_test_error = run_operation(
            ctx, _git()._post_commit_result, ctx, commit_message, False, test_warning_ref, force=True,
        )
    except OwnerPauseRefused as exc:
        return _managed_commit_paused(ctx, managed_tx, commit_message, commit_start, exc)

    if not post_test_error:
        return None
    failure = test_warning_ref[0].strip() or post_test_error
    failure = _git()._managed_commit_gate_failure("assisted_post_commit_tests_failed", failure)
    pre_fingerprint, post_fingerprint = fingerprints
    _git()._record_commit_attempt(
        ctx, commit_message, "failed",
        block_reason="post_commit_tests_failed", block_details=failure,
        duration_sec=time.time() - commit_start, phase="post_commit_tests",
        pre_review_fingerprint=(pre_fingerprint or {}).get("fingerprint", ""),
        post_review_fingerprint=(post_fingerprint or {}).get("fingerprint", ""),
        fingerprint_status="matched",
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
        degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []),
    )
    return failure


def _managed_commit_paused(ctx, tx, message, started_at, refusal):
    """Pause owes the remaining gates, never another commit or a rollback."""
    from supervisor.update_merge import UpdateTxCorrupt, update_tx_phase

    sha = str(tx.get("merge_commit") or "")
    detail = (f"⚠️ OWNER_PAUSE: Managed commit {sha} is retained locally. "
              f"Post-commit verification is unfinished ({refusal}); after Resume call "
              "commit_reviewed again to finish this transaction without another commit.")
    try:
        update_tx_phase(tx, {"phase": "committing_assisted", "merge_commit": sha,
                             "postcommit_resume": tx.get("postcommit_resume") or {}})
    except UpdateTxCorrupt:
        detail += " The transaction marker is unreadable; recovery is required."
    ctx.last_reviewed_commit_sha = sha
    ctx.last_push_succeeded = False
    _git()._record_commit_attempt(ctx, message, "blocked", block_reason="owner_pause",
                                  block_details=detail, duration_sec=time.time() - started_at,
                                  phase="post_commit")
    return _publish_tool_result(ctx, ToolResult(status="blocked", code="LEGACY_BLOCKED", text=detail))


def _finish_managed_commit(ctx, tx, commit_sha, message, started_at, warning, fingerprints):
    from ouroboros.owner_pause import OwnerPauseRefused, run_operation
    from supervisor.update_merge import managed_assisted_postcommit

    try:
        ok, detail = run_operation(ctx, managed_assisted_postcommit, tx, commit_sha)
    except OwnerPauseRefused as exc:
        return _managed_commit_paused(ctx, tx, message, started_at, exc)
    ctx.last_push_succeeded = False
    if not ok:
        _git()._record_commit_attempt(ctx, message, "failed", block_reason="managed_update_smoke_failed",
                                      block_details=detail, duration_sec=time.time() - started_at)
        return detail
    _git().record_bound_commit_success(ctx, message, started_at, *fingerprints)
    ctx._coupling_review_history = {}  # the subject's coupling rounds end with its commit
    return _git()._publish_post_commit_test_fact(
        ctx, _git()._format_commit_result(ctx, message, "", warning[0]) + "\n\n" + detail, warning[0])


def _resume_managed_commit(ctx, tx, started_at):
    """Rejoin only the retained commit, after rechecking its exact immutable binding."""
    saved = tx["postcommit_resume"]
    if not isinstance(saved, dict):
        return "⚠️ MANAGED_UPDATE_ERROR: retained post-commit receipt is invalid; recovery is required."
    for key, value in (saved.get("review_context") or {}).items():
        if key in {"_last_triad_models", "_last_scope_model", "_last_triad_raw_results",
                   "_last_scope_raw_result", "_review_degraded_reasons", "_author_commit_record"}:
            setattr(ctx, key, value)
    sha, message = str(tx.get("merge_commit") or ""), str(saved.get("commit_message") or "")
    fingerprints = (saved.get("pre_fingerprint") or {}, saved.get("post_fingerprint") or {})
    lock = _git()._acquire_git_lock(ctx)
    try:
        ok, detail = _git()._verify_reviewed_commit_binding(
            pathlib.Path(ctx.repo_dir), sha, fingerprints[1], verify_expected_tag=False)
        dirty = _git().run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir).strip()
        if not ok or dirty:
            return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR", text=(
                f"⚠️ MANAGED_UPDATE_POSTCOMMIT_CHANGED: Retained commit {sha} cannot resume "
                f"verification ({detail or 'worktree/index changed'}). Recovery is required.")))
        warning = [""]
        failure = _git()._managed_post_commit_tests_gate(ctx, message, started_at, False, warning, tx,
                                                          fingerprints=fingerprints)
        if failure:
            return failure
        ctx.last_reviewed_commit_sha = sha
    finally:
        _git()._release_git_lock(lock)
    return _finish_managed_commit(ctx, tx, sha, message, started_at, warning, fingerprints)
