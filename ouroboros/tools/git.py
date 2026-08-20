"""Git/write tools with advisory, triad, and scope review commit gates."""

from __future__ import annotations

import json
import hashlib  # noqa: F401
import logging
import os
import pathlib
import re  # noqa: F401
import subprocess  # noqa: F401
import time
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.config import get_runtime_mode  # noqa: F401
from ouroboros.runtime_mode_policy import (
    core_patch_notice,  # noqa: F401
    format_protected_paths,  # noqa: F401
    is_protected_runtime_path,  # noqa: F401
    mode_allows_protected_write,  # noqa: F401
    normalize_repo_path,  # noqa: F401
    protected_paths_in,  # noqa: F401
    protected_write_block_message,  # noqa: F401
)
from ouroboros.platform_layer import acquire_exclusive_file_lock, unlink_lockfile  # noqa: F401
from ouroboros.tools.registry import (
    ToolContext,
    ToolEntry,
    _authorized_managed_update_resolver,  # noqa: F401
    system_repo_dir_for,  # noqa: F401
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,  # noqa: F401
    binding_targets_system_repo,  # noqa: F401
    build_resolved_resource_binding,  # noqa: F401
)
from ouroboros.tools.claude_advisory_review import (
    ADVISORY_REVIEW_CHOICE_GUIDANCE,
    advisory_gate_unavailable,  # noqa: F401
)
from ouroboros.tools.commit_gate import (
    _check_advisory_freshness,  # noqa: F401
    _check_overlapping_review_attempt,
    _invalidate_advisory,  # noqa: F401
    _record_commit_attempt,
    check_blocked_attempt_cap,  # noqa: F401
)
from ouroboros.tools.review_revalidation import handle_revalidation_failure  # noqa: F401
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result  # noqa: F401
from ouroboros.utils import utc_now_iso, write_text, safe_relpath, run_cmd  # noqa: F401
from ouroboros.tools.parallel_review import run_parallel_review as _run_parallel_review, aggregate_review_verdict as _aggregate_review_verdict  # noqa: F401
from ouroboros.tools.review_helpers import (
    _run_review_preflight_tests,  # noqa: F401
    format_review_history_entry,
    paths_from_name_status,  # noqa: F401
    paths_from_porcelain_line as _review_paths_from_porcelain_line,  # noqa: F401
)
from ouroboros.tools.core import _data_skill_path, _str_match_replace, is_skill_control_plane_path  # noqa: F401
from ouroboros.contracts.task_constraint import normalize_task_constraint, resolve_payload_path  # noqa: F401
from ouroboros.contracts.skill_payload_policy import (
    cross_skill_redirect_error,  # noqa: F401
    decide_payload_short_form,  # noqa: F401
)
from ouroboros.tools.git_plumbing import (  # noqa: F401
    _BINARY_EXTENSIONS,
    _acquire_git_lock,
    _binding_repo_rel,
    _binding_targets_system_repo,
    _current_runtime_mode,
    _ensure_gitignore,
    _protected_paths_block_message,
    _publish_git_error,
    _publish_review_blocked,
    _release_git_lock,
    _sanitize_git_error,
    _unstage_binaries,
)
from ouroboros.tools.git_review_cycle import (  # noqa: F401
    _DOC_ONLY_EXTENSIONS,
    _diff_is_doc_only,
    _finalize_blocked_review,
    _fingerprint_staged_diff,
    _handle_revalidation_failure,
    _mark_failed_bypass_advisory_stale,
    _refuse_capped_attempt,
    _review_binding_precondition_error,
    _review_cycle_infra_failure,
    _run_non_committing_review_cycle,
    _run_reviewed_stage_cycle,
    _stage_candidate_for_review,
    _verify_reviewed_commit_binding,
)
from ouroboros.tools.git_evolution import (  # noqa: F401
    _check_evolution_commit_stage,
    _evolution_commit_authority,
    _evolution_publication_stopped_result,
    _preserve_evolution_orphan,
    _record_evolution_commit_receipt,
)
from ouroboros.tools.git_repo_edit import (  # noqa: F401
    _CONTENT_OMITTED_PREFIX,
    _check_shrink_guard,
    _repo_write,
    _str_replace_editor,
)
from ouroboros.tools.git_vcs_ops import (  # noqa: F401
    _binding_relative_path,
    _ff_pull,
    _git_diff,
    _git_status,
    _limit_git_output,
    _pull_from_remote,
    _restore_to_head,
    _revert_commit,
    _vcs_binding,
    _vcs_result,
)

log = logging.getLogger(__name__)


def _auto_tag_on_version_bump(
    repo_dir: pathlib.Path,
    commit_message: str,
    *,
    expected_commit_sha: str = "",
    expected_tag: Optional[str] = None,
) -> str:
    try:
        commit_sha = expected_commit_sha or run_cmd(
            ["git", "rev-parse", "HEAD^{commit}"], cwd=repo_dir
        ).strip()
        if expected_tag is not None:
            tag_name = str(expected_tag or "")
            if not tag_name:
                return ""
        else:
            tag_name = ""
            changed = run_cmd(
                ["git", "diff-tree", "-m", "--no-commit-id", "--name-only", "-r", commit_sha],
                cwd=repo_dir,
            ).strip().splitlines()
            if "VERSION" not in changed:
                return ""
            version = run_cmd(["git", "show", f"{commit_sha}:VERSION"], cwd=repo_dir).strip()
            if not version:
                return ""
            tag_name = f"v{version}"
        tag_msg = f"{tag_name}: {commit_message}"
        try:
            run_cmd(
                ["git", "tag", "-a", tag_name, commit_sha, "-m", tag_msg], cwd=repo_dir
            )
            return f" [tagged: {tag_name}]"
        except Exception as e:
            if "already exists" in str(e):
                try:
                    target = run_cmd(
                        ["git", "rev-parse", f"refs/tags/{tag_name}^{{commit}}"], cwd=repo_dir
                    ).strip()
                except Exception as resolve_exc:
                    return f" [tag verification failed: {_sanitize_git_error(str(resolve_exc))}]"
                if target != commit_sha:
                    return (
                        f" [tag target mismatch: {tag_name} -> {target}, expected {commit_sha}]"
                    )
                return f" [tag {tag_name} already exists at reviewed commit]"
            log.warning("Auto-tag failed: %s", e)
            return f" [tag failed: {e}]"
    except Exception as e:
        log.warning("Auto-tag check failed: %s", e)
        return ""

def _auto_push(repo_dir: pathlib.Path) -> str:
    try:
        from supervisor.git_ops import push_to_remote
        ok, msg = push_to_remote()
        if ok:
            return f" [pushed: {msg}]"
        return f" [push skipped: {msg}]"
    except Exception as e:
        log.debug("Auto-push failed (non-fatal): %s", e)
        return " [push failed — will retry later]"


MAX_TEST_OUTPUT = 8000
_consecutive_test_failures: int = 0


def _log_test_failure(ctx: ToolContext, commit_message: str, test_output: str) -> None:
    from ouroboros.utils import append_jsonl, utc_now_iso  # noqa: F811
    try:
        append_jsonl(ctx.drive_path("logs") / "events.jsonl", {
            "ts": utc_now_iso(), "type": "commit_test_failure",
            "commit_message": commit_message,  # full — no [:200] truncation
            "test_output": test_output[:2000],
            "consecutive_failures": _consecutive_test_failures,
        })
    except Exception:
        pass


def _run_pre_push_tests(ctx: ToolContext, force: bool = False) -> Optional[str]:
    if ctx is None:
        log.warning("_run_pre_push_tests called with ctx=None, skipping tests")
        return None
    if not force and os.environ.get("OUROBOROS_PRE_PUSH_TESTS", "1") != "1":
        return None
    # NO `tests/` existence check here: whether the repository is in scope is
    # run_hermetic_pytest's decision. This entry point runs POST-commit, so it
    # compares HEAD and HEAD~1 (the default phase) — a candidate that deleted the
    # suite is a hard block, while a repo that never had one is out of scope.
    try:
        from ouroboros.preflight_runner import run_hermetic_pytest

        # Timeout owned by run_hermetic_pytest (default + OUROBOROS_PREFLIGHT_TIMEOUT_SEC).
        return run_hermetic_pytest(
            pathlib.Path(ctx.repo_dir),
            max_output=MAX_TEST_OUTPUT,
        )
    except Exception as e:
        log.warning(f"Pre-push tests failed with exception: {e}", exc_info=True)
        return f"⚠️ PRE_PUSH_TEST_ERROR: Unexpected error running tests: {e}"


def _git_commit_with_tests(ctx: ToolContext, force: bool = False) -> Optional[str]:
    test_error = _run_pre_push_tests(ctx, force=force)
    if test_error:
        log.error("Post-commit verification failed")
        ctx.last_push_succeeded = False
        return (
            "⚠️ TESTS_FAILED: Post-commit verification failed.\n"
            f"{test_error}\n"
            "The commit was already created and preserved. Inspect the failures before relying on this revision."
        )
    return None


def _post_commit_result(ctx, commit_message, skip_tests, tw_ref, force: bool = False) -> Optional[str]:
    global _consecutive_test_failures
    if skip_tests and not force:
        return None
    push_error = _git_commit_with_tests(ctx, force=force)
    if push_error:
        _consecutive_test_failures += 1
        _log_test_failure(ctx, commit_message, push_error)
        tw_ref[0] = (f"\n\n⚠️ TESTS_FAILED (commit preserved, "
                     f"consecutive failures: {_consecutive_test_failures}):\n{push_error}")
        return push_error
    else:
        _consecutive_test_failures = 0
    return None


def _managed_commit_gate_failure(reason: str, message: str) -> str:
    """Rollback a landed assisted commit, or keep the failed gate durable.

    A rollback that RAISES is no different from one that returns False: it runs
    several git commands before clearing the marker, so a raise halfway leaves
    the same pre-gate phase on disk — the re-phase to gate_blocked must run
    independently of the rollback's own error handling. When even that re-phase
    cannot be written, the message must stop claiming the tx is pinned."""
    from supervisor.update_merge import (
        mark_update_tx_gate_blocked,
        rollback_managed_update,
    )

    try:
        ok, detail = rollback_managed_update(reason)
    except Exception as exc:
        log.warning("managed update rollback after a red gate raised", exc_info=True)
        ok, detail = False, f"rollback raised {type(exc).__name__}: {exc}"
    if ok:
        return f"{message}\n\nThe managed update was rolled back: {detail}"
    try:
        pinned = bool(mark_update_tx_gate_blocked(reason, detail))
    except Exception as exc:
        log.warning("pinning the update tx gate_blocked failed", exc_info=True)
        pinned = False
        detail = f"{detail}; re-phase raised {type(exc).__name__}: {exc}"
    if not pinned:
        # The message must not claim the tx is pinned when nothing was written
        # (absent/corrupt marker, or the write itself failed).
        return (
            f"{message}\n\n⚠️ MANAGED_UPDATE_ROLLBACK_FAILED ({detail}); the update tx "
            "marker could NOT be re-phased to gate_blocked — if a tx marker remains, "
            "clear or roll it back before the next boot."
        )
    return (
        f"{message}\n\n⚠️ MANAGED_UPDATE_GATE_BLOCKED: rollback could not be verified "
        f"({detail}). The tx is marked gate_blocked; restart/recovery is required."
    )


def _managed_post_commit_tests_gate(
    ctx, commit_message: str, commit_start: float, skip_tests: bool,
    test_warning_ref, managed_tx: Dict[str, Any],
    fingerprints: Tuple[Dict[str, Any], Dict[str, Any]] = ({}, {}),
) -> Optional[str]:
    """BLOCKING post-commit test gate for managed-update merges only: a failed
    suite rolls the assisted merge back instead of shipping a warning (ordinary
    commits keep the warning-only contract later in the flow). The gate is
    MANDATORY: neither the caller's skip_tests nor OUROBOROS_PRE_PUSH_TESTS=0
    can wave a managed merge through untested. The terminal record carries the
    same review metadata/fingerprints as every sibling failure record, so an
    operator can reconstruct WHICH reviewed revision the gate rejected."""
    if not managed_tx:
        return None
    del skip_tests  # deliberately ignored for managed merges
    post_test_error = _post_commit_result(
        ctx, commit_message, False, test_warning_ref, force=True,
    )
    if not post_test_error:
        return None
    failure = test_warning_ref[0].strip() or post_test_error
    failure = _managed_commit_gate_failure("assisted_post_commit_tests_failed", failure)
    pre_fingerprint, post_fingerprint = fingerprints
    _record_commit_attempt(
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


def _review_binding_failure(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    message: str,
    *,
    binding_kind: str,
    fingerprints: Tuple[Dict[str, Any], Dict[str, Any]],
    managed_tx: Dict[str, Any],
) -> str:
    """Record either exact-tree binding failure through one shared path."""
    block_reason, phase, managed_reason = {
        "commit": ("review_binding_mismatch", "commit_binding", "assisted_commit_binding_mismatch"),
        "tag": ("review_tag_binding_mismatch", "tag_binding", "assisted_tag_binding_mismatch"),
    }[binding_kind]
    pre_fingerprint, post_fingerprint = fingerprints
    _record_commit_attempt(
        ctx,
        commit_message,
        "failed",
        block_reason=block_reason,
        block_details=message,
        duration_sec=time.time() - commit_start,
        phase=phase,
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        post_review_fingerprint=post_fingerprint.get("fingerprint", ""),
        fingerprint_status="mismatch",
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
        degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []),
    )
    return _managed_commit_gate_failure(managed_reason, message) if managed_tx else message


def _check_ci_status_after_push(repo_dir: pathlib.Path) -> str:
    """Return CI status for the just-pushed commit SHA, or empty on error."""
    try:
        import urllib.request
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        repo = os.environ.get("GITHUB_REPO", "").strip()
        if not token or not repo:
            return ""
        branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir).strip()
        if not branch or branch == "HEAD":
            return ""
        local_sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip()
        if not local_sha:
            return ""
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ouroboros-ci-check",
        }
        import urllib.parse
        runs_url = (
            f"https://api.github.com/repos/{repo}/actions/runs"
            f"?per_page=10&branch={urllib.parse.quote(branch, safe='')}"
            f"&event=push&head_sha={urllib.parse.quote(local_sha, safe='')}"
        )
        with urllib.request.urlopen(urllib.request.Request(runs_url, headers=headers), timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        runs = [r for r in (data.get("workflow_runs") or []) if r.get("head_sha") == local_sha]
        if not runs:
            return "\n\n⏳ CI: Run not yet registered — check GitHub Actions in ~30s."
        if runs[0].get("status") in ("in_progress", "queued"):
            return "\n\n⏳ CI: Run in progress — check GitHub Actions for results."
        completed = next((r for r in runs if r.get("status") == "completed"), None)
        if completed is None:
            return "\n\n⏳ CI: Run queued — check GitHub Actions for results."
        conclusion = completed.get("conclusion", "")
        if conclusion == "success":
            return "\n\n✅ CI: Run passed for this commit."
        run_number = completed.get("run_number", "?")
        html_url = completed.get("html_url", "")
        jobs_url = completed.get("jobs_url", "")
        failed_summary = "unknown job"
        if jobs_url:
            try:
                with urllib.request.urlopen(urllib.request.Request(jobs_url, headers=headers), timeout=8) as jresp:
                    jdata = json.loads(jresp.read().decode("utf-8"))
                failed_parts = []
                for job in jdata.get("jobs") or []:
                    if job.get("conclusion") == "failure":
                        failed_step = next((s.get("name", "?") for s in job.get("steps") or []
                                            if s.get("conclusion") == "failure"), "?")
                        failed_parts.append(f"{job.get('name', '?')} → {failed_step}")
                if failed_parts:
                    failed_summary = "; ".join(failed_parts)
            except Exception:
                pass  # Fall back to generic summary — run_number/html_url still surfaced below
        if conclusion == "failure":
            return (
                f"\n\n⚠️ CI STATUS: Run FAILED for this commit (run #{run_number})\n"
                f"  Failed: {failed_summary}\n"
                f"  Fix: investigate failing tests, then push a fix commit.\n"
                f"  URL: {html_url}"
            )
        return (
            f"\n\n⚠️ CI STATUS: Run {conclusion.upper()} for this commit (run #{run_number})\n"
            f"  URL: {html_url}"
        )
    except Exception:
        return ""


def _format_commit_result(ctx, commit_message, push_status, test_warning):
    result = f"OK: committed to {ctx.branch_dev}: {commit_message}{push_status}"
    if test_warning:
        result += test_warning
    if ctx._review_advisory:
        result += "\n\n⚠️ Advisory warnings:\n" + "\n".join(
            f"  - {format_review_history_entry(w)}" for w in ctx._review_advisory
        )
    return result


def _prepare_review_commit_worktree(
    ctx: ToolContext,
    managed_tx: Optional[Dict[str, Any]],
) -> tuple[bool, str]:
    """Select the commit branch without orphaning detached work.

    Returns ``(came_from_detached_checkout, error_message)``. Managed assisted
    merges keep their live MERGE_HEAD and only run the existing transaction
    precommit verification.
    """
    is_detached = False
    came_from_detached_checkout = False
    if not managed_tx:
        try:
            current_branch = run_cmd(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ctx.repo_dir
            ).strip()
            is_detached = current_branch == "HEAD"
        except Exception:
            pass
    try:
        if not managed_tx:
            if is_detached:
                run_cmd(
                    ["git", "checkout", "-B", ctx.branch_dev, "HEAD"],
                    cwd=ctx.repo_dir,
                )
                came_from_detached_checkout = True
            else:
                run_cmd(["git", "checkout", ctx.branch_dev], cwd=ctx.repo_dir)
    except Exception as exc:
        error_message = _sanitize_git_error(str(exc))
        already_on_target = False
        try:
            current_branch_after = run_cmd(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ctx.repo_dir
            ).strip()
            already_on_target = current_branch_after == ctx.branch_dev
        except Exception:
            pass
        if not already_on_target:
            return came_from_detached_checkout, _publish_git_error(
                ctx,
                f"⚠️ GIT_ERROR (checkout): {error_message}",
            )
        try:
            unmerged = run_cmd(
                ["git", "diff", "--name-only", "--diff-filter=U"], cwd=ctx.repo_dir
            ).strip()
        except Exception as status_exc:
            return came_from_detached_checkout, _publish_git_error(
                ctx,
                (
                    "⚠️ GIT_ERROR (checkout): "
                    f"{error_message}\n\nCould not verify index state after checkout failure: "
                    f"{_sanitize_git_error(str(status_exc))}"
                ),
            )
        if unmerged:
            return came_from_detached_checkout, _publish_git_error(
                ctx,
                (
                    "⚠️ GIT_ERROR (checkout): "
                    f"{error_message}\n\nRepository has unmerged paths; refusing to treat "
                    "the checkout failure as an incidental dirty-tree no-op.\n"
                    f"{unmerged}"
                ),
            )
    if managed_tx:
        from supervisor.update_merge import managed_assisted_precommit_verify

        verified, error_message = managed_assisted_precommit_verify(managed_tx)
        if not verified:
            return came_from_detached_checkout, error_message
    return came_from_detached_checkout, ""


def _task_attributed_commit_paths(
    ctx: ToolContext,
    paths: Optional[List[str]],
) -> tuple[Optional[List[str]], Optional[Dict[str, Any]], str, Optional[tuple]]:
    """Resolve the exact task-owned commit candidates through the attribution SSOT.

    Returns (paths, attribution, error, binding) where binding is the
    (results_root, evidence_task_id) pair a successful commit uses to advance
    the baseline epoch. NOTE: the results root must resolve to the same
    location the host wrote the baseline to (agent.py uses the task's
    budget_drive_root falling back to the env drive root; for queued tasks the
    task/metadata field carries that value into this context).
    """
    from ouroboros.mutation_attribution import (
        attribution_task_id,
        resolve_attributed_git_paths,
    )

    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        return paths, None, "", None
    metadata = getattr(ctx, "task_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    results_root = pathlib.Path(
        str(
            metadata.get("budget_drive_root")
            or getattr(ctx, "budget_drive_root", "")
            or ctx.drive_root
        )
    )
    root_task_id = str(metadata.get("root_task_id") or "").strip() or task_id
    evidence_task_id = attribution_task_id(results_root, (root_task_id, task_id))
    if not evidence_task_id:
        # No host-captured baseline (manual ToolContext, external dry-run
        # review): the legacy explicit/whole-tree staging contract applies.
        return paths, None, "", None
    selected, attribution, error = resolve_attributed_git_paths(
        results_root,
        evidence_task_id,
        pathlib.Path(ctx.repo_dir),
        paths,
    )
    return selected, attribution, error, (results_root, evidence_task_id)


def _publish_reviewed_commit(
    ctx: ToolContext,
    commit_message: str,
    commit_sha: str,
    tag_info: str,
    test_warning: str,
    paths: Optional[List[str]],
    push_status: str,
) -> str:
    """Record push state and format the successful reviewed commit result."""
    is_evolution = str(ctx.current_task_type or "") == "evolution"
    ctx.last_push_succeeded = "[pushed:" in push_status
    if is_evolution:
        try:
            from supervisor.evolution_lifecycle import update_evolution_transaction

            update_evolution_transaction(
                str(ctx.task_id or ""),
                push_status="pushed" if ctx.last_push_succeeded else "skipped_or_failed",
            )
        except Exception:
            log.debug("Failed to record evolution transaction push status", exc_info=True)
    ci_note = _check_ci_status_after_push(ctx.repo_dir) if ctx.last_push_succeeded else ""
    result = _format_commit_result(ctx, commit_message, push_status + tag_info, test_warning)
    if is_evolution:
        result += (
            "\n\nEvolution transaction open: this cycle should contain at most one reviewed commit. "
            "If this commit is the intended change, call request_restart once now and then stop."
        )
    if paths is not None:
        try:
            untracked = run_cmd(["git", "ls-files", "--others", "--exclude-standard"], cwd=ctx.repo_dir)
            if untracked.strip():
                files = ", ".join(untracked.strip().split("\n"))
                result += f"\n⚠️ WARNING: untracked files remain: {files}"
        except Exception:
            pass
    return result + ci_note


def _repo_commit_push(ctx: ToolContext, commit_message: str,
                       paths: Optional[List[str]] = None,
                       skip_tests: bool = False,
                       review_rebuttal: str = "",
                       skip_advisory_review: bool = False,
                       skip_advisory_pre_review: bool = False,
                       goal: str = "",
                       scope: str = "") -> str:
    """Stage, review, and commit files with unified pre-commit review."""
    skip_advisory_pre_review = bool(skip_advisory_review or skip_advisory_pre_review)
    ctx.last_push_succeeded = False
    ctx._review_advisory = []
    ctx._last_triad_models = []
    ctx._last_scope_model = ""
    ctx._last_triad_raw_results = []
    ctx._last_scope_raw_result = {}
    ctx._review_degraded_reasons = []
    ctx._current_review_tool_name = "commit_reviewed"
    _commit_start = time.time()
    if not commit_message.strip():
        return "⚠️ ERROR: commit_message must be non-empty."
    ctx._current_review_commit_message = commit_message
    # Managed-update merge (P2/SC2): the tx marker authorizes exactly ONE resolution task to
    # commit while a managed merge is staged in the live tree. The resolved tree commits as a
    # reviewed 2-parent merge (native MERGE_HEAD), with push/tag suppressed + an inline
    # pre-restart smoke. Any OTHER task is blocked from committing while the tx is active.
    from supervisor.update_merge import managed_assisted_tx_for
    _managed_tx, _managed_block = managed_assisted_tx_for(getattr(ctx, "task_id", ""), getattr(ctx, "task_metadata", None))
    if _managed_block:
        _record_commit_attempt(ctx, commit_message, "blocked",
                               block_reason="managed_update_in_progress", block_details=_managed_block,
                               duration_sec=0.0, phase="preflight")
        return _managed_block
    attribution_binding = None
    if _managed_tx:
        paths = None  # a managed merge always stages the WHOLE resolved tree (ignore paths)
    else:
        paths, _attribution, attribution_error, attribution_binding = _task_attributed_commit_paths(ctx, paths)
        if attribution_error:
            _record_commit_attempt(
                ctx,
                commit_message,
                "blocked",
                block_reason="mutation_attribution",
                block_details=attribution_error,
                duration_sec=0.0,
                phase="preflight",
            )
            return attribution_error
    overlap_err = _check_overlapping_review_attempt(ctx)
    if overlap_err:
        _record_commit_attempt(
            ctx,
            commit_message,
            "blocked",
            block_reason="overlap_guard",
            block_details=overlap_err,
            duration_sec=0.0,
            phase="preflight",
        )
        return overlap_err
    _record_commit_attempt(ctx, commit_message, "reviewing")
    try:
        lock = _acquire_git_lock(ctx)
    except (TimeoutError, Exception) as e:
        _record_commit_attempt(ctx, commit_message, "failed",
                               block_reason="infra_failure",
                               block_details=f"Git lock: {e}",
                               duration_sec=time.time() - _commit_start)
        return _publish_git_error(ctx, f"⚠️ GIT_ERROR (lock): {e}")
    test_warning_ref = [""]
    _fail = lambda msg: (_record_commit_attempt(ctx, commit_message, "failed",
        block_reason="infra_failure", block_details=msg,
        duration_sec=time.time() - _commit_start), msg)[1]
    try:
        came_from_detached_checkout, preparation_error = _prepare_review_commit_worktree(
            ctx, _managed_tx
        )
        if preparation_error:
            return _fail(preparation_error)
        evolution_claim: Dict[str, str] = {}
        if str(ctx.current_task_type or "") == "evolution":
            evolution_claim, authority_error = _check_evolution_commit_stage(ctx, commit_message, _commit_start, phase="pre_review_authority")
            if authority_error:
                return authority_error
        outcome = _run_reviewed_stage_cycle(
            ctx,
            commit_message,
            _commit_start,
            paths=paths,
            skip_advisory_pre_review=skip_advisory_pre_review,
            skip_tests=skip_tests,
            goal=goal,
            scope=scope,
            review_rebuttal=review_rebuttal,
            came_from_detached_checkout=came_from_detached_checkout,
            require_release_tag=not bool(_managed_tx),
        )
        if outcome.get("status") != "passed":
            if _managed_tx:
                # A blocked review's index reset can clear the live MERGE_HEAD; re-establish it so
                # the agent can fix the flagged issues and re-commit (precommit_verify needs it),
                # rather than stranding the resolution until the watchdog / boot rollback.
                try:
                    from supervisor.update_merge import reestablish_merge_head

                    reestablish_merge_head(str(_managed_tx.get("target_sha") or ""))
                except Exception:
                    log.debug("reestablish_merge_head after blocked managed review failed", exc_info=True)
            return str(outcome.get("message", "") or "")
        pre_fingerprint = outcome.get("pre_fingerprint", {}) or {}
        post_fingerprint = outcome.get("post_fingerprint", {}) or {}

        if evolution_claim:
            _, authority_error = _check_evolution_commit_stage(ctx, commit_message, _commit_start, phase="pre_commit_authority")
            if authority_error:
                return authority_error

        if _managed_tx:
            # PRIMARY conflict-marker leakage gate (a `git add`-ed marker file is a resolved
            # entry that --diff-filter=U misses), then mark the crash-window phase before the
            # native 2-parent commit (MERGE_HEAD is still set, so `git commit` records both
            # parents — reviewed pre_update_sha + target).
            from supervisor.update_merge import managed_assisted_marker_check, write_update_tx

            _mok, _merr = managed_assisted_marker_check()
            if not _mok:
                return _fail(_merr)
            _committing = dict(_managed_tx)
            _committing["phase"] = "committing_assisted"
            write_update_tx(_committing)

        try:
            run_cmd(["git", "commit", "-m", commit_message], cwd=ctx.repo_dir)
            commit_sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
        except Exception as e:
            err_msg = f"⚠️ GIT_ERROR (commit): {_sanitize_git_error(str(e))}"
            if _managed_tx:
                from supervisor.update_merge import restore_assisted_resolution_after_commit_error
                restore_assisted_resolution_after_commit_error(_managed_tx)
            _record_commit_attempt(ctx, commit_message, "failed",
                                   block_reason="infra_failure", block_details=err_msg,
                                   duration_sec=time.time() - _commit_start,
                                   triad_models=getattr(ctx, "_last_triad_models", []),
                                   scope_model=getattr(ctx, "_last_scope_model", ""),
                                   triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
                                   scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
                                   degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []))
            return _publish_git_error(ctx, err_msg)
        binding_ok, binding_detail = _verify_reviewed_commit_binding(
            pathlib.Path(ctx.repo_dir),
            commit_sha,
            post_fingerprint,
            verify_expected_tag=False,
        )
        if not binding_ok:
            binding_msg = (
                "⚠️ REVIEW_BINDING_FAILED: Git created a local commit, but its exact tree, "
                "parents, or staged VERSION do not match the reviewed binding. The commit "
                "was NOT tagged or pushed; inspect the repository and re-run review on a "
                f"new exact tree. Detail: {binding_detail}"
            )
            if evolution_claim:
                binding_msg += " " + _preserve_evolution_orphan(ctx, commit_sha)
            return _review_binding_failure(
                ctx, commit_message, _commit_start, binding_msg,
                binding_kind="commit",
                fingerprints=(pre_fingerprint, post_fingerprint),
                managed_tx=_managed_tx,
            )
        reviewed_binding = post_fingerprint.get("binding", {}) or {}
        gate_failure = _managed_post_commit_tests_gate(
            ctx, commit_message, _commit_start, skip_tests, test_warning_ref, _managed_tx,
            fingerprints=(pre_fingerprint, post_fingerprint),
        )
        if gate_failure:
            return gate_failure
        # A managed-update merge commit must NOT auto-tag/auto-push pre-restart (an un-smoked
        # update would otherwise reach origin / create a version tag, and a later rollback would
        # diverge from origin). The official version tag is handled on the owner's terms.
        tag_info = ""
        created_tag = ""
        if not _managed_tx:
            if evolution_claim:
                _, authority_error = _check_evolution_commit_stage(ctx, commit_message, _commit_start, phase="pre_tag_authority", commit_sha=commit_sha)
                if authority_error:
                    containment = _preserve_evolution_orphan(ctx, commit_sha)
                    return f"{authority_error}\n\n{containment}"
            reviewed_tag = str(reviewed_binding.get("expected_tag") or "")
            tag_info = _auto_tag_on_version_bump(
                pathlib.Path(ctx.repo_dir),
                commit_message,
                expected_commit_sha=commit_sha,
                expected_tag=reviewed_tag,
            )
            created_tag = reviewed_tag if tag_info == f" [tagged: {reviewed_tag}]" else ""
        binding_ok, binding_detail = _verify_reviewed_commit_binding(
            pathlib.Path(ctx.repo_dir),
            commit_sha,
            post_fingerprint,
            verify_expected_tag=not bool(_managed_tx),
        )
        if not binding_ok:
            binding_msg = (
                "⚠️ REVIEW_BINDING_FAILED: the reviewed commit was created locally, but "
                "release-tag verification failed. Nothing was pushed; immutable tags are "
                f"never retargeted. Detail: {binding_detail}"
            )
            if evolution_claim:
                binding_msg += " " + _preserve_evolution_orphan(
                    ctx, commit_sha, created_tag=created_tag,
                )
            return _review_binding_failure(
                ctx, commit_message, _commit_start, binding_msg,
                binding_kind="tag",
                fingerprints=(pre_fingerprint, post_fingerprint),
                managed_tx=_managed_tx,
            )
        if evolution_claim:
            receipt_error = _record_evolution_commit_receipt(
                ctx, commit_message, _commit_start, evolution_claim, commit_sha,
                created_tag=created_tag,
            )
            if receipt_error:
                return receipt_error
        if not _managed_tx:
            # Ordinary self-modification contract: post-commit tests are reported
            # as a warning. Managed-update merges already ran them as the BLOCKING
            # assisted_post_commit_tests gate right after the commit above.
            _post_commit_result(ctx, commit_message, skip_tests, test_warning_ref)
        push_status = ""
        if not _managed_tx and evolution_claim:
            publication_error = _evolution_publication_stopped_result(
                ctx, commit_message, commit_sha, test_warning_ref[0],
                created_tag=created_tag, started_at=_commit_start, fingerprints=(pre_fingerprint, post_fingerprint),
            )
            if publication_error:
                return publication_error
            push_status = _auto_push(ctx.repo_dir)
        ctx.last_reviewed_commit_sha = commit_sha
        if attribution_binding is not None:
            # The task's own commit moved HEAD: open the next attributed-staging
            # epoch so a follow-up commit does not read as ``baseline_stale``.
            try:
                from ouroboros.mutation_attribution import advance_mutation_baseline

                advance_mutation_baseline(
                    attribution_binding[0],
                    attribution_binding[1],
                    pathlib.Path(ctx.repo_dir),
                )
            except Exception:
                log.warning("mutation baseline advance failed after commit", exc_info=True)
        _record_commit_attempt(ctx, commit_message, "succeeded",
                               duration_sec=time.time() - _commit_start,
                               phase="commit",
                               pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
                               post_review_fingerprint=post_fingerprint.get("fingerprint", ""),
                               fingerprint_status="matched",
                               triad_models=getattr(ctx, "_last_triad_models", []),
                               scope_model=getattr(ctx, "_last_scope_model", ""),
                               triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
                               scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
                               degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []))
        ctx._scope_review_history = {}  # Clear on success — next commit starts fresh
    finally:
        _release_git_lock(lock)
    if _managed_tx:
        # Inline pre-restart smoke + tx transition (auto_merge parity); on failure it rolls back
        # and the agent is told. No push (the merge lands locally; restart + boot finalize seal it).
        from supervisor.update_merge import managed_assisted_postcommit

        _ok_pc, _msg_pc = managed_assisted_postcommit(_managed_tx, commit_sha)
        ctx.last_push_succeeded = False
        if not _ok_pc:
            # Smoke failed and the merge was rolled back — do NOT advertise an 'OK: committed'
            # result (callers key on the leading text). Return the failure first + record it.
            _record_commit_attempt(ctx, commit_message, "failed",
                                   block_reason="managed_update_smoke_failed", block_details=_msg_pc,
                                   duration_sec=time.time() - _commit_start)
            return _msg_pc
        return _format_commit_result(ctx, commit_message, "", test_warning_ref[0]) + "\n\n" + _msg_pc
    if not evolution_claim:
        push_status = _auto_push(ctx.repo_dir)
    return _publish_reviewed_commit(
        ctx, commit_message, commit_sha, tag_info, test_warning_ref[0], paths, push_status,
    )


def get_tools() -> List[ToolEntry]:
    reviewed_commit_description = (
        "Commit already-changed files through the unified reviewed commit workflow. "
        f"{ADVISORY_REVIEW_CHOICE_GUIDANCE}"
    )
    skip_advisory_description = (
        "Choose the audited advisory-only skip for this call. "
        f"{ADVISORY_REVIEW_CHOICE_GUIDANCE}"
    )
    return [
        ToolEntry("commit_reviewed", {
            "name": "commit_reviewed",
            "description": reviewed_commit_description,
            "parameters": {"type": "object", "properties": {
                "commit_message": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Optional subset of task-attributed clean-at-baseline paths. Omitted computes the full attributed candidate set; an empty set never stages the whole tree."},
                "skip_tests": {"type": "boolean", "default": False, "description": "Skip pre-commit tests."},
                "review_rebuttal": {"type": "string", "default": "",
                    "description": "If previous commit was blocked by reviewers and you disagree, include counter-argument."},
                "skip_advisory_review": {"type": "boolean", "default": False,
                    "description": skip_advisory_description},
                "goal": {"type": "string", "default": "",
                    "description": "High-level goal of this change. Used by scope reviewer to judge completeness."},
                "scope": {"type": "string", "default": "",
                    "description": "Declared scope boundary. Issues outside scope are advisory-only for scope reviewer."},
            }, "required": ["commit_message"]},
        }, _repo_commit_push, is_code_tool=True),
        ToolEntry("vcs_commit_reviewed", {
            "name": "vcs_commit_reviewed",
            "description": reviewed_commit_description,
            "parameters": {"type": "object", "properties": {
                "commit_message": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Optional subset of task-attributed clean-at-baseline paths. Omitted computes candidates; empty never means git add -A."},
                "skip_tests": {"type": "boolean", "default": False, "description": "Skip pre-commit tests."},
                "review_rebuttal": {"type": "string", "default": ""},
                "skip_advisory_review": {"type": "boolean", "default": False,
                    "description": skip_advisory_description},
                "goal": {"type": "string", "default": ""},
                "scope": {"type": "string", "default": ""},
            }, "required": ["commit_message"]},
        }, _repo_commit_push, is_code_tool=True),
        ToolEntry("vcs_status", {
            "name": "vcs_status",
            "description": "git status --porcelain for the selected repository.",
            "parameters": {"type": "object", "properties": {
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace", "description": "Omit for the active project workspace; use system_repo for Ouroboros source."},
                "path": {"type": "string", "default": "", "description": "Optional path filter relative to the selected repository"},
                "max_chars": {"type": "integer", "default": 0, "description": "Optional output character limit; 0 means no explicit limit"},
            }, "required": []},
        }, _git_status, is_code_tool=True),
        ToolEntry("vcs_diff", {
            "name": "vcs_diff",
            "description": "git diff for the selected repository (use staged=true to see staged changes after git add).",
            "parameters": {"type": "object", "properties": {
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace", "description": "Omit for the active project workspace; use system_repo for Ouroboros source."},
                "staged": {"type": "boolean", "default": False, "description": "If true, show staged changes (--staged)"},
                "path": {"type": "string", "default": "", "description": "Optional path filter relative to the selected repository"},
                "stat": {"type": "boolean", "default": False, "description": "If true, show --stat output"},
                "name_only": {"type": "boolean", "default": False, "description": "If true, show --name-only output"},
                "max_chars": {"type": "integer", "default": 0, "description": "Optional output character limit; 0 means no explicit limit"},
            }, "required": []},
        }, _git_diff, is_code_tool=True),
        ToolEntry("vcs_pull_ff", {
            "name": "vcs_pull_ff",
            "description": "Fetch from origin and fast-forward merge in the selected repository. Safe: never rewrites history.",
            "parameters": {"type": "object", "properties": {
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace", "description": "Omit for the active project workspace; use system_repo for Ouroboros source."},
            }, "required": []},
        }, _pull_from_remote, is_code_tool=True, mutates_worktree=True),
        ToolEntry("vcs_restore", {
            "name": "vcs_restore",
            "description": "Discard uncommitted changes in the selected repository, restoring to HEAD.",
            "parameters": {"type": "object", "properties": {
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace", "description": "Omit for the active project workspace; use system_repo for Ouroboros source."},
                "confirm": {"type": "boolean", "description": "Must be true to execute."},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Specific files to restore"},
            }, "required": ["confirm"]},
        }, _restore_to_head, is_code_tool=True, mutates_worktree=True),
        ToolEntry("vcs_revert", {
            "name": "vcs_revert",
            "description": "Revert a commit in the selected repository by creating a new undo commit. Safe: no history rewrite.",
            "parameters": {"type": "object", "properties": {
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace", "description": "Omit for the active project workspace; use system_repo for Ouroboros source."},
                "sha": {"type": "string", "description": "Commit SHA to revert"},
                "confirm": {"type": "boolean", "description": "Must be true to execute."},
            }, "required": ["sha", "confirm"]},
        }, _revert_commit, is_code_tool=True, mutates_worktree=True),
    ]
