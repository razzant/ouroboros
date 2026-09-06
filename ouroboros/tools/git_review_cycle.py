"""Staging, advisory/triad/scope review and reviewed-material binding for the
commit gate, split out of ``ouroboros/tools/git.py`` (v7 module-size
discipline). Every span is extracted VERBATIM from the parent's tip bytes by
scripts/v7next_transplant.py; the parent re-exports every moved name.
Parent-scope helpers the monolith read as module globals — including the
post-cutoff paid-cycle gate family — are read through the call-time handle
``_git()`` — never a from-import — so the facade binding stays the one tests
monkeypatch. ``_sanitize_git_error`` is the one f-string-read exception (the
byte gate cannot rewrite f-string internals): it binds the plumbing owner at
import time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import subprocess
import time
from typing import Any, Dict, List, Optional

from ouroboros.tools.registry import ToolContext
from ouroboros.tools.git_plumbing import _sanitize_git_error
from ouroboros.tools.git_plumbing import _publish_git_error, _publish_review_blocked

# The parent's logger name is pinned so moved log records keep their %(name)s
# in server.log/stdout — the same logger object the parent binds.
log = logging.getLogger("ouroboros.tools.git")


def _git():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros.tools import git

    return git


def _fingerprint_staged_diff(repo_dir: pathlib.Path) -> Dict[str, Any]:
    """Bind review to the exact commit material, not only a textual diff.

    ``git write-tree`` is the staged snapshot Git will commit. HEAD plus every
    MERGE_HEAD row is the exact parent vector. VERSION is read from the index,
    and a staged VERSION bump binds the expected release tag and any pre-existing
    tag target. The existing durable fingerprint fields remain the review-state
    mechanism; only their input becomes complete.
    """
    try:
        diff_text = _git().run_cmd(
            ["git", "diff", "--cached", "--binary", "--no-ext-diff"],
            cwd=repo_dir,
        )
        tree_sha = _git().run_cmd(["git", "write-tree"], cwd=repo_dir).strip()
        head_sha = _git().run_cmd(["git", "rev-parse", "HEAD^{commit}"], cwd=repo_dir).strip()
        merge_heads: list[str] = []
        git_path = _git().run_cmd(["git", "rev-parse", "--git-path", "MERGE_HEAD"], cwd=repo_dir).strip()
        merge_head_path = pathlib.Path(git_path)
        if not merge_head_path.is_absolute():
            merge_head_path = repo_dir / merge_head_path
        if merge_head_path.exists():
            for raw_sha in merge_head_path.read_text(encoding="utf-8").splitlines():
                raw_sha = raw_sha.strip()
                if not raw_sha:
                    continue
                resolved = _git().run_cmd(
                    ["git", "rev-parse", f"{raw_sha}^{{commit}}"], cwd=repo_dir
                ).strip()
                if resolved and resolved not in merge_heads and resolved != head_sha:
                    merge_heads.append(resolved)
        version_staged = bool(
            _git().run_cmd(
                ["git", "diff", "--cached", "--name-only", "--", "VERSION"],
                cwd=repo_dir,
            ).strip()
        )
        try:
            staged_version = _git().run_cmd(["git", "show", ":VERSION"], cwd=repo_dir).strip()
        except Exception:
            staged_version = ""
        if version_staged and not staged_version:
            raise RuntimeError("staged VERSION is missing or empty")
        expected_tag = f"v{staged_version}" if version_staged else ""
        existing_tag_target = ""
        if expected_tag:
            tag_probe = subprocess.run(
                ["git", "rev-parse", "-q", "--verify", f"refs/tags/{expected_tag}^{{commit}}"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if tag_probe.returncode == 0:
                existing_tag_target = tag_probe.stdout.strip()
            elif tag_probe.returncode not in (1, 128):
                raise RuntimeError(
                    "could not verify expected tag target: "
                    + _sanitize_git_error(tag_probe.stderr.strip() or f"exit {tag_probe.returncode}")
                )
    except Exception as exc:
        return {
            "ok": False,
            "fingerprint": "",
            "status": "unavailable",
            "reason": f"git diff --cached failed: {_sanitize_git_error(str(exc))}",
        }

    binding = {
        "tree_sha": tree_sha,
        "parents": [head_sha, *merge_heads],
        "staged_version": staged_version,
        "version_staged": version_staged,
        "expected_tag": expected_tag,
        "existing_tag_target": existing_tag_target,
        "diff_sha256": hashlib.sha256(
            diff_text.encode("utf-8", errors="replace")
        ).hexdigest(),
    }
    encoded_binding = json.dumps(
        binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256(encoded_binding).hexdigest()[:32]
    return {
        "ok": True,
        "fingerprint": digest,
        "status": "ok",
        "reason": "",
        "chars": len(diff_text),
        "binding": binding,
    }


def _review_binding_precondition_error(
    fingerprint: Dict[str, Any], *, require_release_tag: bool = True
) -> str:
    """Reject a staged release that would reuse an existing immutable tag."""
    binding = fingerprint.get("binding") if isinstance(fingerprint, dict) else None
    if not isinstance(binding, dict):
        return "⚠️ REVIEW_BINDING_BLOCKED: staged review binding is missing."
    expected_tag = str(binding.get("expected_tag") or "")
    existing_target = str(binding.get("existing_tag_target") or "")
    if require_release_tag and expected_tag and existing_target:
        return (
            f"⚠️ REVIEW_BINDING_BLOCKED: expected release tag {expected_tag} already "
            f"targets {existing_target}. Release tags are immutable; bump VERSION or "
            "verify/release a new patch version instead of retargeting the tag."
        )
    return ""


def _verify_reviewed_commit_binding(
    repo_dir: pathlib.Path,
    commit_sha: str,
    fingerprint: Dict[str, Any],
    *,
    verify_expected_tag: bool,
) -> tuple[bool, str]:
    """Verify the created commit/tag are exactly the material reviewed above."""
    binding = fingerprint.get("binding") if isinstance(fingerprint, dict) else None
    if not isinstance(binding, dict):
        return False, "review binding is missing"
    try:
        resolved_commit = _git().run_cmd(
            ["git", "rev-parse", f"{commit_sha}^{{commit}}"], cwd=repo_dir
        ).strip()
        current_head = _git().run_cmd(["git", "rev-parse", "HEAD^{commit}"], cwd=repo_dir).strip()
        actual_tree = _git().run_cmd(
            ["git", "rev-parse", f"{resolved_commit}^{{tree}}"], cwd=repo_dir
        ).strip()
        parent_line = _git().run_cmd(
            ["git", "rev-list", "--parents", "-n", "1", resolved_commit], cwd=repo_dir
        ).strip().split()
        actual_parents = parent_line[1:] if parent_line else []
        actual_version = _git().run_cmd(
            ["git", "show", f"{resolved_commit}:VERSION"], cwd=repo_dir
        ).strip()
    except Exception as exc:
        return False, _sanitize_git_error(str(exc))
    expected_tree = str(binding.get("tree_sha") or "")
    expected_parents = [str(value) for value in (binding.get("parents") or [])]
    expected_version = str(binding.get("staged_version") or "")
    if current_head != resolved_commit:
        return False, f"HEAD moved to {current_head}; created commit was {resolved_commit}"
    if actual_tree != expected_tree:
        return False, f"tree mismatch: reviewed={expected_tree}, committed={actual_tree}"
    if actual_parents != expected_parents:
        return False, f"parent mismatch: reviewed={expected_parents}, committed={actual_parents}"
    if actual_version != expected_version:
        return False, f"VERSION mismatch: reviewed={expected_version!r}, committed={actual_version!r}"
    expected_tag = str(binding.get("expected_tag") or "")
    if verify_expected_tag and expected_tag:
        try:
            tag_target = _git().run_cmd(
                ["git", "rev-parse", f"refs/tags/{expected_tag}^{{commit}}"], cwd=repo_dir
            ).strip()
        except Exception as exc:
            return False, f"expected tag {expected_tag} is unavailable: {_sanitize_git_error(str(exc))}"
        if tag_target != resolved_commit:
            return False, (
                f"tag mismatch: {expected_tag} targets {tag_target}, expected {resolved_commit}"
            )
    return True, ""


def _handle_revalidation_failure(*args, **kwargs):
    return _git().handle_revalidation_failure(
        *args,
        **kwargs,
        record_commit_attempt=_git()._record_commit_attempt,
    )


def _revalidation_outcome(ctx, commit_message, commit_start, before, after, *, worktree_changed=False):
    """Keep prepared and reviewed material bound through the same transition."""
    if not after.get("ok"):
        kind = "fingerprint_unavailable"
    elif worktree_changed or after.get("fingerprint") != before.get("fingerprint"):
        kind = "revalidation_failed"
    else:
        return None
    return {
        "status": "blocked", "block_reason": kind,
        "message": _git()._handle_revalidation_failure(
            ctx, commit_message, commit_start,
            pre_fingerprint=before, post_fingerprint=after, kind=kind,
        ),
        "pre_fingerprint": before, "post_fingerprint": after,
    }


def _finalize_pending_review(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    pre_fingerprint: Dict[str, Any],
    post_fingerprint: Dict[str, Any],
) -> str:
    """Persist the non-terminal wave and leave its exact retry fail-closed."""
    custody_lost = bool(getattr(ctx, "_review_custody_lost", False))
    message = (
        "⚠️ REVIEW_CUSTODY_LOST: the paid review wave is still unresolved, but "
        "its exact process-local custody is unavailable. A second dispatch was "
        "not started; operator reconciliation is required."
        if custody_lost else
        "⚠️ REVIEW_PENDING: physical reviewer work remains in flight. Retry the "
        "same commit to reconcile that exact paid wave; no second dispatch is allowed."
    )
    post_value = str(post_fingerprint.get("fingerprint") or "")
    pre_value = str(pre_fingerprint.get("fingerprint") or "")
    fingerprint_status = (
        "matched" if post_value and post_value == pre_value
        else "mismatch" if post_value else "unavailable"
    )
    _git()._record_commit_attempt(
        ctx,
        commit_message,
        "reviewing",
        block_reason="review_custody_lost" if custody_lost else "review_late_result_pending",
        block_details=message,
        duration_sec=time.time() - commit_start,
        phase="late_wait",
        late_result_pending=True,
        pre_review_fingerprint=pre_value,
        post_review_fingerprint=post_value,
        fingerprint_status=fingerprint_status,
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
        degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []),
        review_retry_key=str(getattr(ctx, "_current_review_retry_key", "") or ""),
    )
    # The index is part of this live wave's identity. Retain it for exact
    # reconciliation; rebuilding it from the worktree could review new bytes.
    return message


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
    block_class: str = "",
) -> str:
    """Persist a genuine blocked review result, then unstage the reviewed diff."""
    _git()._record_commit_attempt(
        ctx,
        commit_message,
        "blocked",
        block_reason=block_reason,
        block_details=combined_msg,
        duration_sec=time.time() - commit_start,
        critical_findings=combined_findings,
        phase="blocking_review",
        block_class=block_class,
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
        _git().run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
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
        if not p.lower().endswith(_git()._DOC_ONLY_EXTENSIONS):
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


def _review_cycle_infra_failure(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    message: str,
) -> Dict[str, Any]:
    """Record and return one fail-closed stage-cycle infrastructure result."""
    if not bool(getattr(ctx, "_review_resume_pending", False)):
        _git()._record_commit_attempt(
            ctx,
            commit_message,
            "failed",
            block_reason="infra_failure",
            block_details=message,
            duration_sec=time.time() - commit_start,
        )
    return {"status": "failed", "message": message}


def _stage_candidate_for_review(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    paths: Optional[List[str]],
    came_from_detached_checkout: bool,
) -> tuple[List[str], Optional[List[str]], Optional[Dict[str, Any]]]:
    """Stage the candidate and return its paths without invoking any reviewer."""
    if not bool(getattr(ctx, "_review_resume_pending", False)):
        from ouroboros.commit_admission import auto_sync_release_metadata_if_needed

        synced = auto_sync_release_metadata_if_needed(
            ctx, pathlib.Path(ctx.repo_dir), pathlib.Path(ctx.drive_root), paths,
        )
        if paths is not None and synced:
            paths = sorted(set(paths) | set(synced))
        if paths:
            try:
                safe_paths = [_git().safe_relpath(path) for path in paths if str(path).strip()]
            except ValueError as exc:
                error = _git()._review_cycle_infra_failure(
                    ctx, commit_message, commit_start, f"⚠️ PATH_ERROR: {exc}"
                )
                return [], None, error
            add_cmd = ["git", "add"] + safe_paths
        else:
            _git()._ensure_gitignore(ctx.repo_dir)
            add_cmd = ["git", "add", "-A"]
        try:
            _git().run_cmd(add_cmd, cwd=ctx.repo_dir)
        except Exception as exc:
            error = _git()._review_cycle_infra_failure(
                ctx,
                commit_message,
                commit_start,
                _publish_git_error(
                    ctx,
                    f"⚠️ GIT_ERROR (add): {_sanitize_git_error(str(exc))}",
                ),
            )
            return [], None, error
        if not paths and not _git()._authorized_managed_update_resolver(ctx):
            removed = _git()._unstage_binaries(ctx.repo_dir)
            if removed:
                log.warning("Unstaged %d binary files: %s", len(removed), removed)
    try:
        status = _git().run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir)
    except Exception as exc:
        error = _git()._review_cycle_infra_failure(
            ctx,
            commit_message,
            commit_start,
            _publish_git_error(
                ctx,
                f"⚠️ GIT_ERROR (status): {_sanitize_git_error(str(exc))}",
            ),
        )
        return [], None, error
    if not status.strip():
        if came_from_detached_checkout:
            message = (
                "⚠️ GIT_LOST_WORKTREE_ON_DETACHED_CHECKOUT_FAILED: working tree is clean "
                "after detached HEAD reconciliation. The detached commits may have been "
                "orphaned. Inspect `git reflog` and restore if needed."
            )
        else:
            message = "⚠️ GIT_NO_CHANGES: nothing to commit."
        return [], None, _git()._review_cycle_infra_failure(
            ctx, commit_message, commit_start, message
        )

    try:
        staged_status_raw = _git().run_cmd(
            ["git", "diff", "--cached", "--name-status", "-M"], cwd=ctx.repo_dir
        )
        classification_paths = _git().paths_from_name_status(staged_status_raw)
    except Exception as exc:
        try:
            staged_names_raw = _git().run_cmd(
                ["git", "diff", "--cached", "--name-only"], cwd=ctx.repo_dir
            )
        except Exception:
            error = _git()._review_cycle_infra_failure(
                ctx,
                commit_message,
                commit_start,
                _publish_git_error(
                ctx,
                f"⚠️ GIT_ERROR (staged-status): {_sanitize_git_error(str(exc))}",
            ),
            )
            return [], None, error
        classification_paths = [
            line.strip() for line in staged_names_raw.splitlines() if line.strip()
        ]
    advisory_paths = classification_paths or None
    if advisory_paths is None:
        try:
            staged_names_raw = _git().run_cmd(
                ["git", "diff", "--cached", "--name-only"], cwd=ctx.repo_dir
            )
        except Exception as exc:
            error = _git()._review_cycle_infra_failure(
                ctx,
                commit_message,
                commit_start,
                _publish_git_error(
                ctx,
                f"⚠️ GIT_ERROR (staged-names): {_sanitize_git_error(str(exc))}",
            ),
            )
            return [], None, error
        advisory_paths = [
            line.strip() for line in staged_names_raw.splitlines() if line.strip()
        ] or None
        classification_paths = advisory_paths or []
    return classification_paths, advisory_paths, None


def _reset_commit_review_state(ctx):
    """One per-call reset for committing and review-only entry points."""
    ctx.last_push_succeeded = False
    ctx._review_advisory = []
    ctx._last_triad_models = []
    ctx._last_scope_model = ""
    ctx._last_triad_raw_results = []
    ctx._last_scope_raw_result = {}
    ctx._review_degraded_reasons = []
    ctx._current_review_tool_name = "commit_reviewed"
    ctx._current_review_retry_key = ""
    ctx._review_reconcile_only = False
    ctx._review_frozen_rows = {}
    ctx._review_custody_lost = False
    ctx._current_review_attempt_number = None


def _reconcile_advisory_before_preparation(ctx, commit_message, *, goal, scope, paths, review_rebuttal):
    """Resolve the same delegated preflight before touching its candidate."""
    from ouroboros.tools.preflight_review_run import pending_advisory_execution

    ctx._advisory_reconciled = False
    try:
        execution, _ = pending_advisory_execution(
            ctx, commit_message, goal=goal, scope=scope, paths=paths, review_rebuttal=review_rebuttal,
        )
        if execution.get("pending_invocation_id"):
            _git()._handle_advisory_pre_review(
                ctx, commit_message, goal=goal, scope=scope, paths=paths,
                review_rebuttal=review_rebuttal, prepared=True,
            )
            remaining, _ = pending_advisory_execution(
                ctx, commit_message, goal=goal, scope=scope, paths=paths, review_rebuttal=review_rebuttal,
            )
            if remaining.get("pending_invocation_id"):
                return "⚠️ REVIEW_PENDING: the exact preflight invocation remains unresolved; no candidate preparation was performed."
            ctx._advisory_reconciled = True
    except Exception as exc:
        return f"⚠️ REVIEW_PENDING: preflight custody could not be reconciled: {exc}"
    return ""


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
    require_release_tag: bool = True,
) -> Dict[str, Any]:
    skip_advisory_pre_review = bool(skip_advisory_review or skip_advisory_pre_review)
    # Subject evidence and memo are scoped to this exact attempt.
    ctx._last_review_subject_trees = set()
    ctx._managed_review_subject_memo = {}
    classification_paths, advisory_paths, stage_error = _git()._stage_candidate_for_review(
        ctx,
        commit_message,
        commit_start,
        paths=paths,
        came_from_detached_checkout=came_from_detached_checkout,
    )
    if stage_error is not None:
        return stage_error
    protected_staged_paths = _git().protected_paths_in(classification_paths)
    runtime_mode = _git()._current_runtime_mode()
    if (
        protected_staged_paths
        and not _git().mode_allows_protected_write(runtime_mode)
        and not _git()._authorized_managed_update_resolver(ctx)
    ):
        msg = _git()._protected_paths_block_message(
            protected_staged_paths,
            runtime_mode=runtime_mode,
            action="commit",
        )
        try:
            if not bool(getattr(ctx, "_review_resume_pending", False)):
                _git().run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
        except Exception:
            pass
        if not bool(getattr(ctx, "_review_resume_pending", False)):
            _git()._record_commit_attempt(
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
    pre_fingerprint = _git()._fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
    if not pre_fingerprint.get("ok"):
        if bool(getattr(ctx, "_review_resume_pending", False)):
            return {
                "status": "blocked",
                "message": "⚠️ REVIEW_BINDING_UNAVAILABLE: cannot verify the exact pending review identity; no new dispatch was started.",
                "block_reason": "fingerprint_unavailable",
                "pre_fingerprint": pre_fingerprint,
                "post_fingerprint": {},
            }
        return {
            "status": "blocked",
            "message": _git()._handle_revalidation_failure(
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
    # Free-cycle identity runs before advisory freshness and any paid dispatch.
    gate_outcome = _git()._free_cycle_gate(
        ctx, commit_message, commit_start,
        pre_fingerprint=pre_fingerprint, review_rebuttal=review_rebuttal,
        goal=goal, scope=scope,
    )
    advisory_replay: Optional[Dict[str, Any]] = None
    if gate_outcome is not None:
        if "advisory_replay" in gate_outcome:
            advisory_replay = gate_outcome
        else:
            return gate_outcome
    binding_error = _git()._review_binding_precondition_error(
        pre_fingerprint, require_release_tag=require_release_tag
    )
    if binding_error:
        if not bool(getattr(ctx, "_review_reconcile_only", False)):
            _git()._record_commit_attempt(
                ctx,
                commit_message,
                "blocked",
                block_reason="review_binding_invalid",
                block_details=binding_error,
                duration_sec=time.time() - commit_start,
                phase="preflight",
                pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
                fingerprint_status="invalid",
            )
        # Reconciliation is owned only by the exact commit-review dispatch
        # below.  This pre-dispatch refusal has no finally block to clear it.
        ctx._review_reconcile_only = False
        return {
            "status": "blocked",
            "message": binding_error,
            "block_reason": "review_binding_invalid",
            "pre_fingerprint": pre_fingerprint,
            "post_fingerprint": {},
        }
    from ouroboros.review_state import compute_snapshot_hash

    prepared_snapshot = compute_snapshot_hash(pathlib.Path(ctx.repo_dir), commit_message, paths=advisory_paths)
    advisory_gate_outcome = None
    if not bool(getattr(ctx, "_review_reconcile_only", False)):
        advisory_gate_outcome = _git()._advisory_and_tests_gate(
            ctx, commit_message, commit_start,
            classification_paths=classification_paths,
            advisory_paths=advisory_paths,
            skip_advisory_pre_review=skip_advisory_pre_review,
            skip_tests=skip_tests,
            review_rebuttal=review_rebuttal,
            free_replay=advisory_replay is not None,
            goal=goal, scope=scope,
        )
    if advisory_gate_outcome is not None:
        return advisory_gate_outcome
    if not bool(getattr(ctx, "_review_reconcile_only", False)):
        after_preflight = _git()._fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
        changed = compute_snapshot_hash(pathlib.Path(ctx.repo_dir), commit_message, paths=advisory_paths) != prepared_snapshot
        revalidation = _revalidation_outcome(
            ctx, commit_message, commit_start, pre_fingerprint, after_preflight, worktree_changed=changed,
        )
        if revalidation is not None:
            return revalidation
    _git()._record_commit_attempt(
        ctx,
        commit_message,
        "reviewing",
        duration_sec=time.time() - commit_start,
        phase="review",
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        fingerprint_status="pending",
        # PAID lands write-ahead only at the first physical dispatch; assembly
        # refusals and advisory free replays remain unpaid.
        rebuttal_sha256=str(getattr(ctx, "_current_review_rebuttal_sha256", "") or ""),
        review_contract_fingerprint=str(
            getattr(ctx, "_current_review_contract_fingerprint", "") or ""
        ),
        review_retry_key=str(getattr(ctx, "_current_review_retry_key", "") or ""),
        late_result_pending=bool(getattr(ctx, "_review_reconcile_only", False)),
    )

    if advisory_replay is not None:
        # ADVISORY free outcome: disclose loudly and let the commit proceed
        # without buying another triad+scope run. Honest wording per cause
        # (wording-3): an identical-diff replay REUSES a recorded verdict; a
        # ceiling exhaustion on NEW bytes has no verdict to reuse — the diff
        # ships without a fresh review and the disclosure must say so.
        review_err, scope_result, triad_block_reason, triad_advisory = None, None, "", []
        replay_reason = str(advisory_replay.get("replay_reason") or "")
        if replay_reason == _git().IDENTICAL_DIFF_BLOCK_REASON:
            progress_note = (
                "Max Review Cycles: identical staged diff — reusing the recorded "
                "review verdict, no paid triad+scope dispatch."
            )
        else:
            progress_note = (
                "Max Review Cycles: paid-cycle ceiling exhausted — no review outcome "
                "exists for this diff; the commit proceeds without a fresh triad+scope "
                "review under advisory enforcement."
            )
        disclosure = (
            "Review enforcement=Advisory: no new triad+scope review was bought for "
            f"this commit ({replay_reason}); no fresh automatic preflight was bought. "
            + str(advisory_replay.get("advisory_replay") or "")
        )
        advisory_list = getattr(ctx, "_review_advisory", None)
        if isinstance(advisory_list, list):
            advisory_list.append(disclosure)
        try:
            ctx.emit_progress_fn(progress_note)
        except Exception:
            pass
    else:
        _git()._install_paid_dispatch_stamp(ctx, commit_message, commit_start, pre_fingerprint)
        try:
            review_err, scope_result, triad_block_reason, triad_advisory = _git()._run_parallel_review(
                ctx,
                commit_message,
                goal=goal,
                scope=scope,
                review_rebuttal=review_rebuttal,
                review_binding_fingerprint=str(pre_fingerprint.get("fingerprint") or ""),
            )
        finally:
            _git()._reconcile_and_clear_review_roster(ctx)
    blocked, combined_msg, block_reason, combined_findings, scope_advisory = _git()._aggregate_review_verdict(
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
    post_fingerprint = _git()._fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
    if _git()._review_custody_pending(ctx):
        return {
            "status": "blocked",
            "message": _git()._finalize_pending_review(
                ctx,
                commit_message,
                commit_start,
                pre_fingerprint=pre_fingerprint,
                post_fingerprint=post_fingerprint,
            ),
            "block_reason": (
                "review_custody_lost"
                if bool(getattr(ctx, "_review_custody_lost", False))
                else "review_late_result_pending"
            ),
            "pre_fingerprint": pre_fingerprint,
            "post_fingerprint": post_fingerprint,
        }
    revalidation = _revalidation_outcome(ctx, commit_message, commit_start, pre_fingerprint, post_fingerprint)
    if revalidation is not None:
        return revalidation
    _subject_mismatch = _git()._subject_binding_mismatch_outcome(
        ctx, commit_message, commit_start, pre_fingerprint, post_fingerprint
    )
    if _subject_mismatch is not None:
        return _subject_mismatch
    from ouroboros.review_custody import review_retry_cancelled
    from ouroboros.deadline_utils import owner_deadline_exhausted_for_context

    if review_retry_cancelled(ctx) or owner_deadline_exhausted_for_context(ctx):
        blocked, combined_msg, block_reason = True, "⚠️ REVIEW_STOPPED: owner cancellation or deadline prevents this commit.", "owner_stopped"
    if blocked:
        # Typed block-row classification (Q16/Δ5): a reviewer VERDICT builds
        # the identical-diff refusal streak; an INFRA fact (fit/quorum/
        # transport/sub-floor) never does and retries freely. Money is a
        # separate axis: paid was stamped at PHYSICAL dispatch, so a
        # dispatched-then-infra-blocked wave still counts toward the ceiling,
        # while an assembly-refused (undispatched) infra block stays free.
        block_class = _git().classify_review_block(
            triad_blocked=bool(review_err),
            triad_block_reason=str(triad_block_reason or ""),
            scope_blocked=bool(scope_result is not None and getattr(scope_result, "blocked", False)),
            scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}) or {},
        )
        blocked_message = _git()._finalize_blocked_review(
            ctx, commit_message, commit_start, combined_msg=combined_msg,
            block_reason=block_reason, combined_findings=combined_findings,
            pre_fingerprint=pre_fingerprint, post_fingerprint=post_fingerprint,
            block_class=block_class,
        )
        if block_reason == "critical_findings":
            blocked_message = _publish_review_blocked(ctx, blocked_message)
        return {
            "status": "blocked",
            "message": blocked_message,
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
    ctx.last_reviewed_commit_sha = ""
    _git()._reset_commit_review_state(ctx)
    commit_start = time.time()
    if not commit_message.strip():
        return {"status": "failed", "message": "⚠️ ERROR: commit_message must be non-empty."}
    ctx._current_review_commit_message = commit_message
    overlap_err = _git()._check_overlapping_review_attempt(ctx)
    if overlap_err:
        _git()._record_commit_attempt(
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
    preflight_pending = _git()._reconcile_advisory_before_preparation(
        ctx, commit_message, goal=goal, scope=scope, paths=paths, review_rebuttal=review_rebuttal,
    )
    if preflight_pending:
        return {"status": "blocked", "message": preflight_pending, "block_reason": "advisory_pending"}
    try:
        lock = _git()._acquire_git_lock(ctx)
    except (TimeoutError, Exception) as exc:
        if not bool(getattr(ctx, "_review_resume_pending", False)):
            _git()._record_commit_attempt(
                ctx,
                commit_message,
                "failed",
                block_reason="infra_failure",
                block_details=f"Git lock: {exc}",
                duration_sec=time.time() - commit_start,
            )
        return {"status": "failed", "message": _publish_git_error(ctx, f"⚠️ GIT_ERROR (lock): {exc}")}

    unstage_warning = ""
    try:
        outcome = _git()._run_reviewed_stage_cycle(
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
            _git()._record_commit_attempt(
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
            outcome["message"] = (
                "Review-only cycle completed under advisory enforcement; failed or missing review remains recorded. "
                if "review_technical_failure_advisory" in (getattr(ctx, "_review_degraded_reasons", []) or [])
                else "Review-only cycle passed. "
            ) + "Commit was not created and the index was unstaged."
        return outcome
    finally:
        try:
            if not (_git()._review_custody_pending(ctx)
                    or (bool(getattr(ctx, "_review_resume_pending", False))
                        and (locals().get("outcome") or {}).get("status") != "passed")):
                _git().run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
        except Exception as exc:
            unstage_warning = f"⚠️ GIT_WARNING (reset): {_sanitize_git_error(str(exc))}"
        _git()._release_git_lock(lock)
        if unstage_warning:
            if 'outcome' in locals():
                message = str(outcome.get("message", "") or "")
                outcome["message"] = f"{message}\n\n---\n{unstage_warning}" if message else unstage_warning
