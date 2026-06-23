"""Git/write tools with advisory, triad, and scope review commit gates."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import pathlib
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

from ouroboros.config import get_runtime_mode
from ouroboros.runtime_mode_policy import (
    core_patch_notice,
    format_protected_paths,
    is_protected_runtime_path,
    mode_allows_protected_write,
    normalize_repo_path,
    protected_paths_in,
    protected_write_block_message,
)
from ouroboros.platform_layer import acquire_exclusive_file_lock, unlink_lockfile
from ouroboros.tools.registry import ToolContext, ToolEntry, active_repo_dir_for
from ouroboros.tools.commit_gate import (
    _check_advisory_freshness,
    _check_overlapping_review_attempt,
    _invalidate_advisory,
    _record_commit_attempt,
    check_blocked_attempt_cap,
)
from ouroboros.tools.review_revalidation import handle_revalidation_failure
from ouroboros.utils import utc_now_iso, write_text, safe_relpath, run_cmd
from ouroboros.tools.parallel_review import run_parallel_review as _run_parallel_review, aggregate_review_verdict as _aggregate_review_verdict
from ouroboros.tools.review_helpers import (
    _run_review_preflight_tests,
    format_review_history_entry,
    paths_from_name_status,
    paths_from_porcelain_line as _review_paths_from_porcelain_line,
)
from ouroboros.tools.core import _data_skill_path, _str_match_replace, is_skill_control_plane_path
from ouroboros.contracts.task_constraint import normalize_task_constraint, resolve_payload_path
from ouroboros.contracts.skill_payload_policy import (
    cross_skill_redirect_error,
    decide_payload_short_form,
)
_CONTENT_OMITTED_PREFIX = "<<CONTENT_OMITTED"
log = logging.getLogger(__name__)


def _current_runtime_mode() -> str:
    try:
        return get_runtime_mode()
    except Exception:
        return "advanced"


def _protected_paths_block_message(paths, *, runtime_mode: str, action: str) -> str:
    rendered = format_protected_paths(paths)
    return (
        f"⚠️ CORE_PROTECTION_BLOCKED: runtime_mode={runtime_mode!r} refuses "
        f"to {action} protected Ouroboros core/contract/release path(s): {rendered}. "
        "Use runtime_mode='pro' and pass the normal triad + scope review before "
        "committing protected surfaces."
    )


def _sanitize_git_error(msg: str) -> str:
    return re.sub(r"(https?://)([^@\s]+@)", r"\1<redacted>@", msg)


def _fingerprint_staged_diff(repo_dir: pathlib.Path) -> Dict[str, Any]:
    """Return a deterministic fingerprint of the staged diff being reviewed."""
    try:
        diff_text = run_cmd(
            ["git", "diff", "--cached", "--binary", "--no-ext-diff"],
            cwd=repo_dir,
        )
    except Exception as exc:
        return {
            "ok": False,
            "fingerprint": "",
            "status": "unavailable",
            "reason": f"git diff --cached failed: {_sanitize_git_error(str(exc))}",
        }

    digest = hashlib.sha256(diff_text.encode("utf-8", errors="replace")).hexdigest()[:32]
    return {
        "ok": True,
        "fingerprint": digest,
        "status": "ok",
        "reason": "",
        "chars": len(diff_text),
    }


def _handle_revalidation_failure(*args, **kwargs):
    return handle_revalidation_failure(
        *args,
        **kwargs,
        record_commit_attempt=_record_commit_attempt,
    )


def _finalize_blocked_review(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    combined_msg: str,
    block_reason: str,
    combined_findings: List[Dict[str, Any]],
    pre_fingerprint: Dict[str, Any],
    post_fingerprint: Dict[str, Any],
) -> str:
    """Persist a genuine blocked review result, then unstage the reviewed diff."""
    _record_commit_attempt(
        ctx,
        commit_message,
        "blocked",
        block_reason=block_reason,
        block_details=combined_msg,
        duration_sec=time.time() - commit_start,
        critical_findings=combined_findings,
        phase="blocking_review",
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        post_review_fingerprint=post_fingerprint.get("fingerprint", ""),
        fingerprint_status="matched",
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
        degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []),
    )
    try:
        run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
    except Exception as e:
        warning = f"⚠️ GIT_WARNING (reset): {_sanitize_git_error(str(e))}"
        return f"{combined_msg}\n\n---\n{warning}"
    return combined_msg


_DOC_ONLY_EXTENSIONS = (".md", ".txt", ".rst")


def _diff_is_doc_only(staged_paths: List[str]) -> bool:
    """Return True only for docs outside tests; JSON/config keep preflight."""
    if not staged_paths:
        return False
    saw_any = False
    for raw in staged_paths:
        p = str(raw).strip()
        if not p:
            continue
        saw_any = True
        if p.startswith("tests/") or "/tests/" in p:
            return False
        if not p.lower().endswith(_DOC_ONLY_EXTENSIONS):
            return False
    return saw_any


def _mark_failed_bypass_advisory_stale(
    ctx: ToolContext,
    commit_message: str,
    advisory_paths: Optional[List[str]],
) -> None:
    """Prevent a failed bypass preflight from satisfying later freshness checks."""
    try:
        from ouroboros.review_state import (
            compute_snapshot_hash,
            make_repo_key,
            update_state,
            _utc_now,
        )

        snapshot_hash = compute_snapshot_hash(
            pathlib.Path(ctx.repo_dir),
            commit_message,
            paths=advisory_paths,
        )
        repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))

        def _mutate(state):
            state.mark_stale(snapshot_hash)
            state.last_stale_from_edit_ts = _utc_now()
            state.last_stale_reason = "tests_preflight_blocked"
            state.last_stale_repo_key = repo_key

        update_state(pathlib.Path(ctx.drive_root), _mutate)
    except Exception:
        log.debug("Failed to stale bypass advisory after preflight block", exc_info=True)


def _refuse_capped_attempt(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    pre_fingerprint: Dict[str, Any],
    review_rebuttal: str,
) -> Optional[Dict[str, Any]]:
    """Identical-diff blocked-attempt cap preflight; None allows the attempt."""
    cap_msg = check_blocked_attempt_cap(
        ctx,
        pre_fingerprint.get("fingerprint", ""),
        has_rebuttal=bool(str(review_rebuttal or "").strip()),
    )
    if not cap_msg:
        return None
    try:
        run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
    except Exception:
        pass
    _record_commit_attempt(
        ctx,
        commit_message,
        "blocked",
        block_reason="attempt_cap_reached",
        block_details=cap_msg,
        duration_sec=time.time() - commit_start,
        phase="preflight",
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
    )
    return {
        "status": "blocked",
        "message": cap_msg,
        "block_reason": "attempt_cap_reached",
    }


def _run_reviewed_stage_cycle(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    paths: Optional[List[str]] = None,
    skip_advisory_review: bool = False,
    skip_advisory_pre_review: bool = False,
    skip_tests: bool = False,
    goal: str = "",
    scope: str = "",
    review_rebuttal: str = "",
    came_from_detached_checkout: bool = False,
) -> Dict[str, Any]:
    skip_advisory_pre_review = bool(skip_advisory_review or skip_advisory_pre_review)
    def _failed(message: str) -> Dict[str, Any]:
        _record_commit_attempt(
            ctx,
            commit_message,
            "failed",
            block_reason="infra_failure",
            block_details=message,
            duration_sec=time.time() - commit_start,
        )
        return {"status": "failed", "message": message}

    if paths:
        try:
            safe_paths = [safe_relpath(path) for path in paths if str(path).strip()]
        except ValueError as exc:
            return _failed(f"⚠️ PATH_ERROR: {exc}")
        add_cmd = ["git", "add"] + safe_paths
    else:
        _ensure_gitignore(ctx.repo_dir)
        add_cmd = ["git", "add", "-A"]
    try:
        run_cmd(add_cmd, cwd=ctx.repo_dir)
    except Exception as exc:
        return _failed(f"⚠️ GIT_ERROR (add): {_sanitize_git_error(str(exc))}")
    if not paths:
        removed = _unstage_binaries(ctx.repo_dir)
        if removed:
            log.warning("Unstaged %d binary files: %s", len(removed), removed)
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir)
    except Exception as exc:
        return _failed(f"⚠️ GIT_ERROR (status): {_sanitize_git_error(str(exc))}")
    if not status.strip():
        if came_from_detached_checkout:
            # BUG 2 fix: a clean tree right after a detached-HEAD reconciliation is NOT a
            # legitimate no-op — it usually means commits/edits were orphaned. Emit a
            # distinct, actionable diagnostic instead of the generic GIT_NO_CHANGES that
            # previously misled the agent (it misdiagnosed it as a missing API key).
            # The token carries the _FAILED marker (loop_tool_execution._FAILURE_MARKERS) and
            # deliberately does NOT use the "⚠️ GIT_ERROR" prefix, which is carved out to
            # is_error=False — so this DOES classify as an infra failure (execution=infra_failed)
            # and counts toward evolution_consecutive_failures instead of resetting it.
            return _failed(
                "⚠️ GIT_LOST_WORKTREE_ON_DETACHED_CHECKOUT_FAILED: working tree is clean after "
                "detached HEAD reconciliation. The detached commits may have been orphaned. "
                "Inspect `git reflog` and restore if needed."
            )
        return _failed("⚠️ GIT_NO_CHANGES: nothing to commit.")

    try:
        staged_status_raw = run_cmd(["git", "diff", "--cached", "--name-status", "-M"], cwd=ctx.repo_dir)
        classification_paths = paths_from_name_status(staged_status_raw)
    except Exception as exc:
        try:
            staged_names_raw = run_cmd(
                ["git", "diff", "--cached", "--name-only"],
                cwd=ctx.repo_dir,
            )
        except Exception:
            return _failed(f"⚠️ GIT_ERROR (staged-status): {_sanitize_git_error(str(exc))}")
        classification_paths = [
            line.strip() for line in staged_names_raw.splitlines() if line.strip()
        ]
    advisory_paths = classification_paths or None
    if advisory_paths is None:
        try:
            staged_names_raw = run_cmd(
                ["git", "diff", "--cached", "--name-only"],
                cwd=ctx.repo_dir,
            )
        except Exception as exc:
            return _failed(f"⚠️ GIT_ERROR (staged-names): {_sanitize_git_error(str(exc))}")
        advisory_paths = [
            line.strip() for line in staged_names_raw.splitlines() if line.strip()
        ] or None
        classification_paths = advisory_paths or []
    protected_staged_paths = protected_paths_in(classification_paths)
    runtime_mode = _current_runtime_mode()
    if protected_staged_paths and not mode_allows_protected_write(runtime_mode):
        msg = _protected_paths_block_message(
            protected_staged_paths,
            runtime_mode=runtime_mode,
            action="commit",
        )
        try:
            run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
        except Exception:
            pass
        _record_commit_attempt(
            ctx,
            commit_message,
            "blocked",
            block_reason="core_protection_blocked",
            block_details=msg,
            duration_sec=time.time() - commit_start,
            critical_findings=[],
            phase="preflight",
        )
        return {
            "status": "blocked",
            "message": msg,
            "block_reason": "core_protection_blocked",
        }
    advisory_err = _check_advisory_freshness(
        ctx,
        commit_message,
        skip_advisory_pre_review,
        paths=advisory_paths,
    )
    if advisory_err:
        run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
        _record_commit_attempt(
            ctx,
            commit_message,
            "blocked",
            block_reason="no_advisory",
            block_details=advisory_err,
            duration_sec=time.time() - commit_start,
        )
        return {
            "status": "blocked",
            "message": advisory_err,
            "block_reason": "no_advisory",
        }

    _advisory_bypassed = skip_advisory_pre_review or not os.environ.get("ANTHROPIC_API_KEY", "")
    _diff_aware = (os.environ.get("OUROBOROS_PREFLIGHT_DIFF_AWARE", "true") or "true").strip().lower() in ("true", "1", "yes")
    _doc_only = _diff_aware and _diff_is_doc_only(classification_paths)
    if _advisory_bypassed and not skip_tests and not _doc_only:
        try:
            ctx.emit_progress_fn(
                "Advisory bypassed — running test preflight before triad + scope review..."
            )
        except Exception:
            pass
        test_err = _run_review_preflight_tests(ctx)
        if test_err:
            msg = (
                "⚠️ TESTS_PREFLIGHT_BLOCKED: Tests must pass before triad + scope review "
                "when advisory is bypassed.\n"
                "Fix the failures below, then re-run commit_reviewed (or drop "
                "skip_advisory_review=True to run the full advisory flow).\n"
                "Set OUROBOROS_PRE_PUSH_TESTS=0 to skip tests entirely.\n\n"
                f"{test_err}"
            )
            try:
                run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
            except Exception:
                pass
            _record_commit_attempt(
                ctx,
                commit_message,
                "blocked",
                block_reason="tests_preflight_blocked",
                block_details=msg,
                duration_sec=time.time() - commit_start,
                # Preflight, not a review verdict: must neither inflate nor
                # reset the identical-diff blocked-attempt cap streak.
                phase="preflight",
            )
            _mark_failed_bypass_advisory_stale(ctx, commit_message, advisory_paths)
            return {
                "status": "blocked",
                "message": msg,
                "block_reason": "tests_preflight_blocked",
            }
    elif _advisory_bypassed:
        if skip_tests and _doc_only:
            _skip_reason = "skip_tests + doc_only"
        elif skip_tests:
            _skip_reason = "skip_tests"
        else:
            _skip_reason = "doc_only"
        try:
            ctx.emit_progress_fn(
                f"Advisory bypassed — preflight tests skipped ({_skip_reason})."
            )
        except Exception:
            pass

    pre_fingerprint = _fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
    if not pre_fingerprint.get("ok"):
        return {
            "status": "blocked",
            "message": _handle_revalidation_failure(
                ctx,
                commit_message,
                commit_start,
                pre_fingerprint=pre_fingerprint,
                kind="fingerprint_unavailable",
            ),
            "block_reason": "fingerprint_unavailable",
            "pre_fingerprint": pre_fingerprint,
            "post_fingerprint": {},
        }
    cap_refusal = _refuse_capped_attempt(
        ctx, commit_message, commit_start,
        pre_fingerprint=pre_fingerprint, review_rebuttal=review_rebuttal,
    )
    if cap_refusal is not None:
        return cap_refusal
    _record_commit_attempt(
        ctx,
        commit_message,
        "reviewing",
        duration_sec=time.time() - commit_start,
        phase="review",
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        fingerprint_status="pending",
    )

    review_err, scope_result, triad_block_reason, triad_advisory = _run_parallel_review(
        ctx,
        commit_message,
        goal=goal,
        scope=scope,
        review_rebuttal=review_rebuttal,
    )
    blocked, combined_msg, block_reason, combined_findings, scope_advisory = _aggregate_review_verdict(
        review_err,
        scope_result,
        triad_block_reason,
        triad_advisory,
        ctx,
        commit_message,
        commit_start,
        ctx.repo_dir,
    )
    if scope_advisory:
        advisory_list = getattr(ctx, "_review_advisory", None)
        if isinstance(advisory_list, list):
            advisory_list.extend(scope_advisory)
    post_fingerprint = _fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
    if not post_fingerprint.get("ok"):
        return {
            "status": "blocked",
            "message": _handle_revalidation_failure(
                ctx,
                commit_message,
                commit_start,
                pre_fingerprint=pre_fingerprint,
                post_fingerprint=post_fingerprint,
                kind="fingerprint_unavailable",
            ),
            "block_reason": "fingerprint_unavailable",
            "pre_fingerprint": pre_fingerprint,
            "post_fingerprint": post_fingerprint,
        }
    if post_fingerprint.get("fingerprint") != pre_fingerprint.get("fingerprint"):
        return {
            "status": "blocked",
            "message": _handle_revalidation_failure(
                ctx,
                commit_message,
                commit_start,
                pre_fingerprint=pre_fingerprint,
                post_fingerprint=post_fingerprint,
                kind="revalidation_failed",
            ),
            "block_reason": "revalidation_failed",
            "pre_fingerprint": pre_fingerprint,
            "post_fingerprint": post_fingerprint,
        }
    if blocked:
        return {
            "status": "blocked",
            "message": _finalize_blocked_review(
                ctx,
                commit_message,
                commit_start,
                combined_msg=combined_msg,
                block_reason=block_reason,
                combined_findings=combined_findings,
                pre_fingerprint=pre_fingerprint,
                post_fingerprint=post_fingerprint,
            ),
            "block_reason": block_reason,
            "pre_fingerprint": pre_fingerprint,
            "post_fingerprint": post_fingerprint,
            "combined_findings": combined_findings,
        }
    return {
        "status": "passed",
        "message": "",
        "pre_fingerprint": pre_fingerprint,
        "post_fingerprint": post_fingerprint,
    }


def _run_non_committing_review_cycle(
    ctx: ToolContext,
    commit_message: str,
    *,
    paths: Optional[List[str]] = None,
    skip_advisory_review: bool = False,
    skip_advisory_pre_review: bool = False,
    goal: str = "",
    scope: str = "",
    review_rebuttal: str = "",
) -> Dict[str, Any]:
    skip_advisory_pre_review = bool(skip_advisory_review or skip_advisory_pre_review)
    ctx.last_push_succeeded = False
    ctx.last_reviewed_commit_sha = ""
    ctx._review_advisory = []
    ctx._last_triad_models = []
    ctx._last_scope_model = ""
    ctx._last_triad_raw_results = []
    ctx._last_scope_raw_result = {}
    ctx._review_degraded_reasons = []
    ctx._current_review_tool_name = "commit_reviewed"
    commit_start = time.time()
    if not commit_message.strip():
        return {"status": "failed", "message": "⚠️ ERROR: commit_message must be non-empty."}
    ctx._current_review_commit_message = commit_message
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
        return {
            "status": "blocked",
            "message": overlap_err,
            "block_reason": "overlap_guard",
        }
    try:
        lock = _acquire_git_lock(ctx)
    except (TimeoutError, Exception) as exc:
        _record_commit_attempt(
            ctx,
            commit_message,
            "failed",
            block_reason="infra_failure",
            block_details=f"Git lock: {exc}",
            duration_sec=time.time() - commit_start,
        )
        return {"status": "failed", "message": f"⚠️ GIT_ERROR (lock): {exc}"}

    unstage_warning = ""
    try:
        outcome = _run_reviewed_stage_cycle(
            ctx,
            commit_message,
            commit_start,
            paths=paths,
            skip_advisory_pre_review=skip_advisory_pre_review,
            goal=goal,
            scope=scope,
            review_rebuttal=review_rebuttal,
        )
        if outcome.get("status") == "passed":
            pre_fingerprint = outcome.get("pre_fingerprint", {}) or {}
            post_fingerprint = outcome.get("post_fingerprint", {}) or {}
            _record_commit_attempt(
                ctx,
                commit_message,
                "reviewed",
                duration_sec=time.time() - commit_start,
                phase="review_only",
                pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
                post_review_fingerprint=post_fingerprint.get("fingerprint", ""),
                fingerprint_status="matched",
                triad_models=getattr(ctx, "_last_triad_models", []),
                scope_model=getattr(ctx, "_last_scope_model", ""),
                triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
                scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
                degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []),
            )
            ctx._scope_review_history = {}
            outcome["message"] = "Review-only cycle passed. Commit was not created and the index was unstaged."
        return outcome
    finally:
        try:
            run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
        except Exception as exc:
            unstage_warning = f"⚠️ GIT_WARNING (reset): {_sanitize_git_error(str(exc))}"
        _release_git_lock(lock)
        if unstage_warning:
            if 'outcome' in locals():
                message = str(outcome.get("message", "") or "")
                outcome["message"] = f"{message}\n\n---\n{unstage_warning}" if message else unstage_warning


def _auto_tag_on_version_bump(repo_dir: pathlib.Path, commit_message: str) -> str:
    try:
        changed = run_cmd(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            cwd=repo_dir,
        ).strip().splitlines()
        if "VERSION" not in changed:
            return ""
        version = (repo_dir / "VERSION").read_text(encoding="utf-8").strip()
        if not version:
            return ""
        tag_name = f"v{version}"
        tag_msg = f"v{version}: {commit_message}"
        try:
            run_cmd(["git", "tag", "-a", tag_name, "-m", tag_msg], cwd=repo_dir)
            return f" [tagged: {tag_name}]"
        except Exception as e:
            if "already exists" in str(e):
                return f" [tag {tag_name} already exists]"
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

_BINARY_EXTENSIONS = frozenset({
    ".so", ".dylib", ".dll", ".a", ".lib", ".o", ".obj",
    ".pyc", ".pyo", ".whl", ".egg",
})

def _ensure_gitignore(repo_dir) -> None:
    gi = pathlib.Path(repo_dir) / ".gitignore"
    if not gi.exists():
        write_text(gi, "__pycache__/\n*.pyc\n*.pyo\n*.so\n*.dylib\n*.dll\n"
                       "*.dist-info/\nbase_library.zip\n.DS_Store\n")  # atomic (G)
def _unstage_binaries(repo_dir) -> List[str]:
    try:
        staged = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=repo_dir)
    except Exception:
        return []
    removed = []
    for f in staged.strip().splitlines():
        f = f.strip()
        if f and pathlib.Path(f).suffix.lower() in _BINARY_EXTENSIONS:
            try:
                run_cmd(["git", "reset", "HEAD", "--", f], cwd=repo_dir)
                removed.append(f)
            except Exception:
                pass
    return removed


def _acquire_git_lock(ctx: ToolContext, timeout_sec: int = 120) -> pathlib.Path:
    lock_dir = ctx.drive_path("locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "git.lock"
    fd = acquire_exclusive_file_lock(
        lock_path,
        timeout_sec=float(timeout_sec),
        stale_sec=600.0,
        metadata=f"locked_at={utc_now_iso()}\n",
        poll_sec=0.5,
    )
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        return lock_path
    raise TimeoutError(f"Git lock not acquired within {timeout_sec}s: {lock_path}")


def _release_git_lock(lock_path: pathlib.Path) -> None:
    unlink_lockfile(lock_path)

MAX_TEST_OUTPUT = 8000
_consecutive_test_failures: int = 0


def _log_test_failure(ctx: ToolContext, commit_message: str, test_output: str) -> None:
    from ouroboros.utils import append_jsonl, utc_now_iso
    try:
        append_jsonl(ctx.drive_path("logs") / "events.jsonl", {
            "ts": utc_now_iso(), "type": "commit_test_failure",
            "commit_message": commit_message,  # full — no [:200] truncation
            "test_output": test_output[:2000],
            "consecutive_failures": _consecutive_test_failures,
        })
    except Exception:
        pass


def _run_pre_push_tests(ctx: ToolContext) -> Optional[str]:
    if ctx is None:
        log.warning("_run_pre_push_tests called with ctx=None, skipping tests")
        return None
    if os.environ.get("OUROBOROS_PRE_PUSH_TESTS", "1") != "1":
        return None
    tests_dir = pathlib.Path(ctx.repo_dir) / "tests"
    if not tests_dir.exists():
        return None
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


def _git_commit_with_tests(ctx: ToolContext) -> Optional[str]:
    test_error = _run_pre_push_tests(ctx)
    if test_error:
        log.error("Post-commit verification failed")
        ctx.last_push_succeeded = False
        return (
            "⚠️ TESTS_FAILED: Post-commit verification failed.\n"
            f"{test_error}\n"
            "The commit was already created and preserved. Inspect the failures before relying on this revision."
        )
    return None


def _post_commit_result(ctx, commit_message, skip_tests, tw_ref):
    global _consecutive_test_failures
    if skip_tests:
        return
    push_error = _git_commit_with_tests(ctx)
    if push_error:
        _consecutive_test_failures += 1
        _log_test_failure(ctx, commit_message, push_error)
        tw_ref[0] = (f"\n\n⚠️ TESTS_FAILED (commit preserved, "
                     f"consecutive failures: {_consecutive_test_failures}):\n{push_error}")
    else:
        _consecutive_test_failures = 0


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


def _check_shrink_guard(ctx: ToolContext, file_path: str, new_content: str, force: bool = False) -> Optional[str]:
    """Block likely accidental tracked-file truncation unless force=True."""
    if force:
        return None
    try:
        target = ctx.repo_path(file_path)
        if not target.exists():
            return None
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", safe_relpath(file_path)],
            cwd=str(active_repo_dir_for(ctx)), capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        old_content = target.read_text(encoding="utf-8")
        old_len = len(old_content)
        new_len = len(new_content)
        if old_len > 0 and new_len < old_len * 0.7:
            pct = round(new_len / old_len * 100)
            return (
                f"⚠️ WRITE_BLOCKED: new content for '{file_path}' is {pct}% of original "
                f"({old_len} -> {new_len} chars). This looks like accidental truncation. "
                f"Use edit_text for surgical edits, or pass force=true to confirm "
                f"intentional rewrite."
            )
    except Exception:
        pass
    return None


def _repo_write(ctx: ToolContext, path: str = "", content: str = "",
                files: Optional[List[Dict[str, str]]] = None,
                force: bool = False,
                display_root: str = "active_workspace") -> str:
    """Write file(s) to the repo working directory without committing."""
    write_list: List[Dict[str, str]] = []
    if files:
        for entry in files:
            if not isinstance(entry, dict):
                return "⚠️ WRITE_ERROR: each item in files must be {path, content}."
            p = entry.get("path", "").strip()
            c = entry.get("content", "")
            if not p:
                return "⚠️ WRITE_ERROR: every file entry must have a non-empty 'path'."
            write_list.append({"path": p, "content": c})
    elif path and content is not None:
        write_list.append({"path": path.strip(), "content": content})
    else:
        return "⚠️ WRITE_ERROR: provide either (path + content) or files array."

    if not write_list:
        return "⚠️ WRITE_ERROR: nothing to write."

    for e in write_list:
        norm = normalize_repo_path(e["path"])
        if not ctx.is_workspace_mode() and is_protected_runtime_path(norm) and not mode_allows_protected_write(_current_runtime_mode()):
            return protected_write_block_message(
                path=norm,
                runtime_mode=_current_runtime_mode(),
                action="write",
            )
        if isinstance(e["content"], str) and e["content"].strip().startswith(_CONTENT_OMITTED_PREFIX):
            return (
                f"⚠️ WRITE_ERROR: content for '{e['path']}' looks like a compaction marker. "
                "Re-read the file and provide the actual content."
            )

    written = []
    written_paths: List[str] = []
    for e in write_list:
        shrink_warning = _check_shrink_guard(ctx, e["path"], e["content"], force=force)
        if shrink_warning:
            if written:
                _invalidate_advisory(
                    ctx,
                    changed_paths=written_paths,
                    mutation_root=active_repo_dir_for(ctx),
                    source_tool="write_file",
                )
            return shrink_warning
        try:
            target = ctx.repo_path(e["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            write_text(target, e["content"])
            written.append(f"{display_root}:{safe_relpath(e['path'])} ({len(e['content'])} chars)")
            written_paths.append(e["path"])
        except Exception as exc:
            if written:
                _invalidate_advisory(
                    ctx,
                    changed_paths=written_paths,
                    mutation_root=active_repo_dir_for(ctx),
                    source_tool="write_file",
                )
            already = ", ".join(written) if written else "(none)"
            return (
                f"⚠️ FILE_WRITE_ERROR on '{e['path']}': {exc}\n"
                f"Successfully written before error: {already}"
            )

    _invalidate_advisory(
        ctx,
        changed_paths=written_paths,
        mutation_root=active_repo_dir_for(ctx),
        source_tool="write_file",
    )
    summary = ", ".join(written)
    if ctx.is_workspace_mode():
        result = (
            f"✅ Written {len(written)} file(s): {summary}\n"
            "Files are on disk in the active workspace. Do not commit; the headless runner will emit a patch artifact."
        )
    else:
        result = (
            f"✅ Written {len(written)} file(s): {summary}\n"
            "Files are on disk but NOT committed. Run commit_reviewed when ready.\n"
            "⚠️ Advisory pre-review is now stale — run advisory_review before commit_reviewed."
        )
    protected_written = [] if ctx.is_workspace_mode() else protected_paths_in(written_paths)
    if protected_written and mode_allows_protected_write(_current_runtime_mode()):
        result += "\n\n" + core_patch_notice(protected_written)
    return result


def _str_replace_editor(
    ctx: ToolContext,
    path: str,
    old_str: str,
    new_str: str,
    bucket: str = "",
    skill_name: str = "",
    display_root: str = "active_workspace",
) -> str:
    """Replace exactly one occurrence of old_str with new_str in a file."""
    if not path or not path.strip():
        return "⚠️ STR_REPLACE_ERROR: path is required."
    if not old_str:
        return "⚠️ STR_REPLACE_ERROR: old_str is required (cannot be empty)."

    norm = normalize_repo_path(path)
    if not ctx.is_workspace_mode() and is_protected_runtime_path(norm) and not mode_allows_protected_write(_current_runtime_mode()):
        return protected_write_block_message(
            path=norm,
            runtime_mode=_current_runtime_mode(),
            action="edit",
        )

    existing_tc = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    data_skill_target = None
    task_constraint = existing_tc
    short_form = None
    if ctx.is_workspace_mode():
        try:
            target = ctx.repo_path(path)
        except ValueError as e:
            return f"⚠️ PATH_ERROR: {e}"
        invalidation_root = active_repo_dir_for(ctx)
    else:
        short_form = decide_payload_short_form(
            bucket=bucket,
            skill_name=skill_name,
            path_text=path,
            repo_dir=pathlib.Path(ctx.repo_dir),
            drive_root=pathlib.Path(ctx.drive_root),
        )
        if short_form.error:
            return f"⚠️ STR_REPLACE_ERROR: {short_form.error}"
        synth = short_form.constraint
        redirect_err = cross_skill_redirect_error(existing_tc, synth)
        if redirect_err:
            return f"⚠️ SKILL_REDIRECT_BLOCKED: {redirect_err}"
        task_constraint = existing_tc if existing_tc and existing_tc.mode == "skill_repair" else synth or existing_tc

    if not ctx.is_workspace_mode() and task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
        try:
            target = resolve_payload_path(pathlib.Path(ctx.drive_root), task_constraint, path)
            data_skill_target = target
        except ValueError as e:
            return f"⚠️ STR_REPLACE_ERROR: {e}"
        if is_skill_control_plane_path(target, pathlib.Path(ctx.drive_root).resolve(strict=False)):
            return (
                "⚠️ STR_REPLACE_BLOCKED: skill provenance, launcher seed, "
                "marketplace, dependency, and self-authored markers are "
                "control-plane state. Edit user-authored payload files instead."
            )
        invalidation_root = pathlib.Path(ctx.drive_root)
    elif not ctx.is_workspace_mode():
        data_skill_target = _data_skill_path(path, pathlib.Path(ctx.drive_root))
        if data_skill_target is not None:
            if is_skill_control_plane_path(data_skill_target, pathlib.Path(ctx.drive_root).resolve(strict=False)):
                return (
                    "⚠️ STR_REPLACE_BLOCKED: skill provenance, launcher seed, "
                    "marketplace, dependency, and self-authored markers are "
                    "control-plane state. Edit user-authored payload files instead."
                )
            target = data_skill_target
            invalidation_root = pathlib.Path(ctx.drive_root)
        else:
            try:
                target = ctx.repo_path(path)
            except ValueError as e:
                return f"⚠️ PATH_ERROR: {e}"
            invalidation_root = active_repo_dir_for(ctx)

    if not target.exists():
        return f"⚠️ STR_REPLACE_ERROR: file not found: {path}"

    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        return f"⚠️ STR_REPLACE_ERROR: cannot read {path}: {e}"

    # Shared exact-match single-replacement (deferral 4): identical count==0/count>1
    # feedback for the repo and data-plane editors.
    new_content, _match_err = _str_match_replace(content, old_str, new_str, path, "STR_REPLACE_ERROR")
    if _match_err:
        return _match_err
    if data_skill_target is not None:
        # Deferral 5: a data-plane skill payload edited via the active_workspace route gets
        # the SAME shrink guard as the root=skill_payload editor — no silent >30% truncation
        # of a payload file. (Intentional large rewrites go through root=skill_payload, which
        # carries the force escape hatch.)
        from ouroboros.tools.core import _check_data_shrink_guard

        _shrink_block = _check_data_shrink_guard(target, new_content)
        if _shrink_block:
            return _shrink_block
    try:
        write_text(target, new_content)
    except Exception as e:
        return f"⚠️ STR_REPLACE_ERROR: write failed for {path}: {e}"

    replacement_line = new_content[:new_content.index(new_str)].count('\n') + 1
    context_start = max(0, replacement_line - 3)
    context_lines = new_content.splitlines()[context_start:replacement_line + len(new_str.splitlines()) + 2]
    context_preview = "\n".join(
        f"{context_start + i + 1:>4}| {line}" for i, line in enumerate(context_lines)
    )

    _invalidate_advisory(
        ctx,
        changed_paths=[path],
        mutation_root=invalidation_root,
        source_tool="edit_text",
    )
    result = (
        f"✅ Replaced in {display_root}:{safe_relpath(path)} (line {replacement_line}).\n"
        f"Context:\n{context_preview}\n\n"
        "File is on disk but NOT committed."
    )
    if short_form is not None and short_form.ignored_reason:
        result += f"\n⚠️ SKILL_SHORT_FORM_IGNORED: {short_form.ignored_reason}."
    if data_skill_target is None and ctx.is_workspace_mode():
        result += "\nDo not commit; the headless runner will emit a patch artifact."
    elif data_skill_target is None:
        result += "\nRun commit_reviewed when ready.\n⚠️ Advisory pre-review is now stale — run advisory_review before commit_reviewed."
    else:
        result += "\nRun skill_review for this skill before enabling or declaring it ready."
    if not ctx.is_workspace_mode() and is_protected_runtime_path(norm) and mode_allows_protected_write(_current_runtime_mode()):
        result += "\n\n" + core_patch_notice([norm])
    return result


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

    _managed_tx, _managed_block = managed_assisted_tx_for(getattr(ctx, "task_id", ""))
    if _managed_block:
        _record_commit_attempt(ctx, commit_message, "blocked",
                               block_reason="managed_update_in_progress", block_details=_managed_block,
                               duration_sec=0.0, phase="preflight")
        return _managed_block
    if _managed_tx:
        paths = None  # a managed merge always stages the WHOLE resolved tree (ignore paths)
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
        return f"⚠️ GIT_ERROR (lock): {e}"
    test_warning_ref = [""]
    _fail = lambda msg: (_record_commit_attempt(ctx, commit_message, "failed",
        block_reason="infra_failure", block_details=msg,
        duration_sec=time.time() - _commit_start), msg)[1]
    try:
        is_detached = False
        came_from_detached_checkout = False
        if not _managed_tx:
            # BUG 1 detection: a detached HEAD ahead of the branch must NOT be reconciled with
            # a plain `git checkout ctx.branch_dev` (it would orphan the in-flight commits). The
            # managed-merge path keeps a live MERGE_HEAD and does no checkout below, so it never
            # needs this — only probe HEAD on the normal path.
            try:
                current_branch = run_cmd(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=ctx.repo_dir,
                ).strip()
                is_detached = (current_branch == "HEAD")
            except Exception:
                pass
        try:
            # A managed assisted merge has a LIVE MERGE_HEAD + unmerged index (the agent
            # resolved files but `git add` only happens in the stage cycle). A `git checkout`
            # of the current branch can refuse on a conflicted index, so skip it — materialize
            # already pinned HEAD to the branch and managed_assisted_precommit_verify re-checks.
            if not _managed_tx:
                if is_detached:
                    # BUG 1 fix: HEAD detached (e.g. seeded from a tag checkout). A plain
                    # `git checkout ctx.branch_dev` would jump to the branch's (older) tip and
                    # silently discard in-flight commits/work-tree edits. Force-move the branch
                    # to the current HEAD instead, preserving the detached commits and work tree.
                    run_cmd(["git", "checkout", "-B", ctx.branch_dev, "HEAD"], cwd=ctx.repo_dir)
                    came_from_detached_checkout = True
                else:
                    run_cmd(["git", "checkout", ctx.branch_dev], cwd=ctx.repo_dir)
        except Exception as e:
            err_msg = _sanitize_git_error(str(e))
            already_on_target = False
            try:
                current_branch_after = run_cmd(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=ctx.repo_dir,
                ).strip()
                already_on_target = (current_branch_after == ctx.branch_dev)
            except Exception:
                pass
            if not already_on_target:
                return _fail(f"⚠️ GIT_ERROR (checkout): {err_msg}")
            try:
                unmerged = run_cmd(
                    ["git", "diff", "--name-only", "--diff-filter=U"],
                    cwd=ctx.repo_dir,
                ).strip()
            except Exception as status_err:
                return _fail(
                    "⚠️ GIT_ERROR (checkout): "
                    f"{err_msg}\n\nCould not verify index state after checkout failure: "
                    f"{_sanitize_git_error(str(status_err))}"
                )
            if unmerged:
                return _fail(
                    "⚠️ GIT_ERROR (checkout): "
                    f"{err_msg}\n\nRepository has unmerged paths; refusing to treat "
                    "the checkout failure as an incidental dirty-tree no-op.\n"
                    f"{unmerged}"
                )
        if _managed_tx:
            from supervisor.update_merge import managed_assisted_precommit_verify

            _mok, _merr = managed_assisted_precommit_verify(_managed_tx)
            if not _mok:
                return _fail(_merr)
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

        if _managed_tx:
            # PRIMARY conflict-marker leakage gate (a `git add`-ed marker file is a resolved
            # entry that --diff-filter=U misses), then mark the crash-window phase before the
            # native 2-parent commit (MERGE_HEAD is still set, so `git commit` records both
            # parents — local_snapshot + target).
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
            _record_commit_attempt(ctx, commit_message, "failed",
                                   block_reason="infra_failure", block_details=err_msg,
                                   duration_sec=time.time() - _commit_start,
                                   triad_models=getattr(ctx, "_last_triad_models", []),
                                   scope_model=getattr(ctx, "_last_scope_model", ""),
                                   triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
                                   scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
                                   degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []))
            return err_msg
        ctx.last_reviewed_commit_sha = commit_sha
        if str(ctx.current_task_type or "") == "evolution":
            try:
                from supervisor.evolution_lifecycle import update_evolution_transaction

                update_evolution_transaction(
                    str(ctx.task_id or ""),
                    preflight_status="passed",
                    advisory_status="fresh_or_bypassed",
                    triad_scope_status="passed",
                    commit_sha=commit_sha,
                    restart_required=True,
                    restart_verified=False,
                )
            except Exception:
                log.debug("Failed to record evolution transaction commit", exc_info=True)
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
        _post_commit_result(ctx, commit_message, skip_tests, test_warning_ref)
        # A managed-update merge commit must NOT auto-tag/auto-push pre-restart (an un-smoked
        # update would otherwise reach origin / create a version tag, and a later rollback would
        # diverge from origin). The official version tag is handled on the owner's terms.
        tag_info = "" if _managed_tx else _auto_tag_on_version_bump(ctx.repo_dir, commit_message)
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
    push_status = _auto_push(ctx.repo_dir)
    ctx.last_push_succeeded = "[pushed:" in push_status
    if str(ctx.current_task_type or "") == "evolution":
        try:
            from supervisor.evolution_lifecycle import update_evolution_transaction

            update_evolution_transaction(
                str(ctx.task_id or ""),
                push_status="pushed" if ctx.last_push_succeeded else "skipped_or_failed",
            )
        except Exception:
            log.debug("Failed to record evolution transaction push status", exc_info=True)
    ci_note = ""
    if ctx.last_push_succeeded:
        ci_note = _check_ci_status_after_push(ctx.repo_dir)
    result = _format_commit_result(ctx, commit_message, push_status + tag_info, test_warning_ref[0])
    if str(ctx.current_task_type or "") == "evolution":
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


def _limit_git_output(text: str, max_chars: int = 0) -> str:
    limit = int(max_chars or 0)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n⚠️ OUTPUT_TRUNCATED: git output limited to {limit} characters by max_chars."


def _git_status(ctx: ToolContext, path: str = "", max_chars: int = 0) -> str:
    try:
        cmd = ["git", "status", "--porcelain"]
        if str(path or "").strip():
            cmd.extend(["--", safe_relpath(str(path))])
        return _limit_git_output(run_cmd(cmd, cwd=active_repo_dir_for(ctx)), max_chars)
    except Exception as e:
        return f"⚠️ GIT_ERROR: {_sanitize_git_error(str(e))}"


def _git_diff(
    ctx: ToolContext,
    staged: bool = False,
    path: str = "",
    stat: bool = False,
    name_only: bool = False,
    max_chars: int = 0,
) -> str:
    try:
        repo_dir = active_repo_dir_for(ctx)
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if name_only:
            cmd.append("--name-only")
        elif stat:
            cmd.append("--stat")
        if str(path or "").strip():
            cmd.extend(["--", safe_relpath(str(path))])
        from ouroboros.protected_artifacts import shell_block_reason as protected_artifact_shell_block_reason

        protected_block = protected_artifact_shell_block_reason(ctx, cmd, cwd=str(repo_dir), default_cwd=repo_dir)
        if protected_block:
            return protected_block
        return _limit_git_output(run_cmd(cmd, cwd=repo_dir), max_chars)
    except Exception as e:
        return f"⚠️ GIT_ERROR: {_sanitize_git_error(str(e))}"


def _ff_pull(repo_dir: pathlib.Path) -> str:
    try:
        branch = run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir,
        ).strip()
    except Exception as e:
        return f"⚠️ PULL_ERROR: Could not determine current branch: {e}"
    if not branch or branch == "HEAD":
        return "⚠️ PULL_ERROR: Not on a named branch (detached HEAD). Cannot pull."
    try:
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
    except Exception as e:
        return f"⚠️ PULL_ERROR: git fetch failed: {_sanitize_git_error(str(e))}"
    try:
        before_sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip()
        remote_sha = run_cmd(
            ["git", "rev-parse", f"origin/{branch}"], cwd=repo_dir,
        ).strip()
    except Exception as e:
        return f"⚠️ PULL_ERROR: Could not resolve SHAs: {e}"
    if before_sha == remote_sha:
        return f"Already up to date. HEAD={before_sha[:8]} matches origin/{branch}."
    try:
        new_commits = run_cmd(
            ["git", "log", "--oneline", f"HEAD..origin/{branch}"], cwd=repo_dir,
        ).strip()
    except Exception:
        new_commits = "(could not list commits)"
    try:
        run_cmd(["git", "merge", "--ff-only", f"origin/{branch}"], cwd=repo_dir)
    except Exception as e:
        err = str(e).strip()
        if "Not possible to fast-forward" in err or "diverged" in err.lower():
            return (
                f"⚠️ PULL_ERROR: Branches have diverged — cannot fast-forward.\n"
                f"Local HEAD: {before_sha[:8]}, origin/{branch}: {remote_sha[:8]}\n"
                "Manual resolution needed."
            )
        return f"⚠️ PULL_ERROR: git merge --ff-only failed: {_sanitize_git_error(err)}"
    try:
        after_sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip()
    except Exception:
        after_sha = remote_sha
    lines = [
        f"Pulled origin/{branch}: {before_sha[:8]} → {after_sha[:8]}",
        "", "New commits:",
    ]
    for line in (new_commits or "(none)").splitlines():
        lines.append(f"  {line}")
    return "\n".join(lines)


def _pull_from_remote(ctx: ToolContext) -> str:
    return _ff_pull(pathlib.Path(ctx.repo_dir))


def _restore_to_head(ctx: ToolContext, confirm: bool = False,
                     paths: Optional[List[str]] = None) -> str:
    repo_dir = pathlib.Path(ctx.repo_dir)
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir).strip()
    except Exception as e:
        return f"⚠️ RESTORE_ERROR: git status failed: {e}"
    if not status:
        return "Nothing to restore — working directory is already clean."
    dirty_files = [
        path
        for line in status.splitlines()
        for path in _review_paths_from_porcelain_line(line)
    ]
    affected_protected = protected_paths_in(dirty_files)
    if paths:
        for p in paths:
            norm = normalize_repo_path(p)
            if is_protected_runtime_path(norm):
                return (
                    f"⚠️ RESTORE_BLOCKED: Cannot restore protected file: {norm}. "
                    "Protected core/contract/release paths must be changed through reviewed commits."
                )
    elif affected_protected:
        return (
            f"⚠️ RESTORE_BLOCKED: Uncommitted changes touch protected file(s): "
            f"{format_protected_paths(affected_protected)}. "
            f"Use paths= to restore specific non-critical files, or resolve manually."
        )
    if not confirm:
        try:
            diff_stat = run_cmd(["git", "diff", "--stat"], cwd=repo_dir).strip()
        except Exception:
            diff_stat = "(could not generate diff)"
        try:
            untracked = run_cmd(
                ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo_dir,
            ).strip()
        except Exception:
            untracked = ""
        preview = ["Uncommitted changes that will be lost:", "", diff_stat]
        if untracked:
            preview.append("")
            preview.append("Untracked files that will be removed:")
            for f in untracked.splitlines()[:15]:
                preview.append(f"  {f}")
        preview.append("")
        preview.append("Call again with confirm=true to proceed.")
        return "\n".join(preview)
    if paths:
        safe_paths = [os.path.normpath(p.strip().lstrip("./")) for p in paths if p.strip()]
        if not safe_paths:
            return "⚠️ RESTORE_ERROR: No valid paths provided."
        try:
            run_cmd(["git", "checkout", "HEAD", "--"] + safe_paths, cwd=repo_dir)
        except Exception as e:
            return f"⚠️ RESTORE_ERROR: git checkout failed: {e}"
        try:
            run_cmd(["git", "clean", "-fd", "--"] + safe_paths, cwd=repo_dir)
        except Exception:
            pass
        return f"Restored {len(safe_paths)} path(s) to HEAD."
    else:
        try:
            run_cmd(["git", "checkout", "HEAD", "--", "."], cwd=repo_dir)
        except Exception as e:
            return f"⚠️ RESTORE_ERROR: git checkout failed: {e}"
        try:
            run_cmd(["git", "clean", "-fd"], cwd=repo_dir)
        except Exception:
            pass
        return "All uncommitted changes discarded. Working directory matches HEAD."


def _revert_commit(ctx: ToolContext, sha: str, confirm: bool = False) -> str:
    repo_dir = pathlib.Path(ctx.repo_dir)
    sha = sha.strip()
    if not sha:
        return "⚠️ REVERT_ERROR: sha parameter is required."
    try:
        full_sha = run_cmd(
            ["git", "rev-parse", "--verify", sha], cwd=repo_dir,
        ).strip()
    except Exception:
        return f"⚠️ REVERT_ERROR: Commit '{sha}' not found."
    try:
        parents = run_cmd(
            ["git", "rev-list", "--parents", "-1", full_sha], cwd=repo_dir,
        ).strip().split()
    except Exception:
        parents = [full_sha]
    if len(parents) > 2:
        return (
            f"⚠️ REVERT_ERROR: Commit {sha[:8]} is a merge commit ({len(parents)-1} parents). "
            "git revert on merge commits requires specifying a parent."
        )
    try:
        changed_files = run_cmd(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", full_sha],
            cwd=repo_dir,
        ).strip().splitlines()
    except Exception:
        changed_files = []
    protected_changes = protected_paths_in(changed_files)
    if protected_changes:
        return (
            f"⚠️ REVERT_BLOCKED: Commit {sha[:8]} touches protected file(s): "
            f"{format_protected_paths(protected_changes)}. "
            "Direct vcs_revert cannot create protected-path commits; stage the intended "
            "revert manually and use commit_reviewed so the normal triad + scope review covers it."
        )
    try:
        commit_msg = run_cmd(
            ["git", "log", "-1", "--format=%s", full_sha], cwd=repo_dir,
        ).strip()
    except Exception:
        commit_msg = "(unknown)"
    if not confirm:
        try:
            diff_stat = run_cmd(
                ["git", "diff", f"{full_sha}^..{full_sha}", "--stat"], cwd=repo_dir,
            ).strip()
        except Exception:
            diff_stat = "(could not generate diff)"
        return (
            f"This will revert commit {full_sha[:8]}:\n"
            f"  Message: {commit_msg}\n"
            f"  Files changed:\n{diff_stat}\n\n"
            "A new commit will be created that undoes these changes.\n"
            "Call again with confirm=true to proceed."
        )
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir).strip()
    except Exception:
        status = ""
    if status:
        return (
            "⚠️ REVERT_ERROR: Working directory is not clean.\n"
            "Commit or discard changes first (use vcs_restore), then retry."
        )
    lock = _acquire_git_lock(ctx)
    try:
        try:
            run_cmd(["git", "revert", "--no-edit", full_sha], cwd=repo_dir)
        except Exception as e:
            try:
                run_cmd(["git", "revert", "--abort"], cwd=repo_dir)
            except Exception:
                pass
            return f"⚠️ REVERT_ERROR: git revert failed: {e}"
    finally:
        _release_git_lock(lock)
    return f"Reverted commit {full_sha[:8]}: {commit_msg}\nNew revert commit created."


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("commit_reviewed", {
            "name": "commit_reviewed",
            "description": (
                "Commit already-changed files. Requires a fresh advisory_review run first. "
                "Includes unified pre-commit multi-model review before commit, "
                "with configurable Advisory/Blocking enforcement, plus blocking scope review."
            ),
            "parameters": {"type": "object", "properties": {
                "commit_message": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Files to add (empty = git add -A)"},
                "skip_tests": {"type": "boolean", "default": False, "description": "Skip pre-commit tests."},
                "review_rebuttal": {"type": "string", "default": "",
                    "description": "If previous commit was blocked by reviewers and you disagree, include counter-argument."},
                "skip_advisory_review": {"type": "boolean", "default": False,
                    "description": "Bypass advisory pre-review gate (durably audited). Use only when necessary."},
                "goal": {"type": "string", "default": "",
                    "description": "High-level goal of this change. Used by scope reviewer to judge completeness."},
                "scope": {"type": "string", "default": "",
                    "description": "Declared scope boundary. Issues outside scope are advisory-only for scope reviewer."},
            }, "required": ["commit_message"]},
        }, _repo_commit_push, is_code_tool=True),
        ToolEntry("vcs_commit_reviewed", {
            "name": "vcs_commit_reviewed",
            "description": "Alias of commit_reviewed for version-control workflows.",
            "parameters": {"type": "object", "properties": {
                "commit_message": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Files to add (empty = git add -A)"},
                "skip_tests": {"type": "boolean", "default": False, "description": "Skip pre-commit tests."},
                "review_rebuttal": {"type": "string", "default": ""},
                "skip_advisory_review": {"type": "boolean", "default": False},
                "goal": {"type": "string", "default": ""},
                "scope": {"type": "string", "default": ""},
            }, "required": ["commit_message"]},
        }, _repo_commit_push, is_code_tool=True),
        ToolEntry("vcs_status", {
            "name": "vcs_status",
            "description": "git status --porcelain for the active repo/workspace",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "default": "", "description": "Optional path filter relative to the active repo/workspace"},
                "max_chars": {"type": "integer", "default": 0, "description": "Optional output character limit; 0 means no explicit limit"},
            }, "required": []},
        }, _git_status, is_code_tool=True),
        ToolEntry("vcs_diff", {
            "name": "vcs_diff",
            "description": "git diff for the active repo/workspace (use staged=true to see staged changes after git add)",
            "parameters": {"type": "object", "properties": {
                "staged": {"type": "boolean", "default": False, "description": "If true, show staged changes (--staged)"},
                "path": {"type": "string", "default": "", "description": "Optional path filter relative to the active repo/workspace"},
                "stat": {"type": "boolean", "default": False, "description": "If true, show --stat output"},
                "name_only": {"type": "boolean", "default": False, "description": "If true, show --name-only output"},
                "max_chars": {"type": "integer", "default": 0, "description": "Optional output character limit; 0 means no explicit limit"},
            }, "required": []},
        }, _git_diff, is_code_tool=True),
        ToolEntry("vcs_pull_ff", {
            "name": "vcs_pull_ff",
            "description": "Fetch from origin and fast-forward merge. Safe: never rewrites history.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, _pull_from_remote, is_code_tool=True, mutates_worktree=True),
        ToolEntry("vcs_restore", {
            "name": "vcs_restore",
            "description": "Discard uncommitted changes, restoring to last committed state (HEAD).",
            "parameters": {"type": "object", "properties": {
                "confirm": {"type": "boolean", "description": "Must be true to execute."},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Specific files to restore"},
            }, "required": ["confirm"]},
        }, _restore_to_head, is_code_tool=True, mutates_worktree=True),
        ToolEntry("vcs_revert", {
            "name": "vcs_revert",
            "description": "Revert a specific commit by creating a new undo commit. Safe: no history rewrite.",
            "parameters": {"type": "object", "properties": {
                "sha": {"type": "string", "description": "Commit SHA to revert"},
                "confirm": {"type": "boolean", "description": "Must be true to execute."},
            }, "required": ["sha", "confirm"]},
        }, _revert_commit, is_code_tool=True, mutates_worktree=True),
    ]
