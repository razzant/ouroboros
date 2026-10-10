"""Commit staging, review, continuation and binding. ``tools.git`` re-exports
these functions; ``_git()`` preserves its patchable facade bindings, while
neutral plumbing imports bind their own owner."""

from __future__ import annotations

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

import contextlib
import hashlib
import json
import logging
import pathlib
import subprocess
import time
from typing import Any, Dict, List, Optional

from ouroboros.tools.registry import ToolContext
from ouroboros.tools.git_plumbing import _sanitize_git_error, _publish_git_error, _publish_review_blocked
from ouroboros.tools.review_helpers import review_enforcement_blocks

# Keep the public facade's logger name in server/stdout records.
log = logging.getLogger("ouroboros.tools.git")


def _git():
    """Read the public facade at call time so monkeypatches keep one binding."""
    from ouroboros.tools import git

    return git


def _review_custody_pending(ctx: ToolContext) -> bool:
    """Keep the prepared candidate while the panel's physical review custody is unresolved.
    A preflight never holds it: its wave froze its own tree and keeps its custody on its own
    ``review_change`` rows, so cleaning the commit's index cannot strand it."""
    if bool(getattr(ctx, "_review_custody_lost", False)):
        return True
    triad = list(getattr(ctx, "_last_triad_raw_results", []) or [])
    scope_raw = getattr(ctx, "_last_scope_raw_result", {}) or {}
    scope_rows = list(scope_raw.get("raw_results") or [scope_raw]) if isinstance(scope_raw, dict) else []
    return any(
        bool(row.get("late_result_pending"))
        or str(row.get("operation_state") or "") in {"in_flight", "custody_lost"}
        for row in [*triad, *scope_rows]
        if isinstance(row, dict)
    )


def _release_review_evidence_if_settled(ctx: ToolContext) -> None:
    """The existing review-custody boundary owns temporary session-file lifetime."""
    from ouroboros.review_evidence import release_commit_review_session_view

    if not _review_custody_pending(ctx):
        release_commit_review_session_view(getattr(ctx, "_commit_review_evidence", None) or {})


def _fingerprint_staged_diff(repo_dir: pathlib.Path) -> Dict[str, Any]:
    """Bind review to the exact commit material, not only a textual diff.

    ``git write-tree`` is the staged snapshot Git will commit. HEAD plus every
    MERGE_HEAD row is the exact parent vector. VERSION is read from the index,
    and a staged VERSION bump binds the expected release tag and any pre-existing
    tag target. The existing durable fingerprint fields remain the review-state
    mechanism; only their input becomes complete. ``diff_sha256`` is the frozen
    subject's own patch identity (``review_subject.staged_patch``).
    """
    from ouroboros.tools.review_subject import staged_patch

    try:
        patch, diff_sha256 = staged_patch(repo_dir)
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
        "diff_sha256": diff_sha256,
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
        "chars": len(patch.decode("utf-8", "replace").strip()),
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
) -> Optional[str]:
    """Retain non-terminal custody; Cyber may continue without closing the wave."""
    if getattr(ctx, "_review_cyber_pending", "") and not review_enforcement_blocks("blocking"):
        # The pending row belongs to an earlier invocation, not this free continuation.
        return None
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
    if not review_enforcement_blocks("blocking"):
        from ouroboros.tools.review import _handle_review_block_or_warning

        ctx._last_review_block_reason = "review_late_result_pending"
        _handle_review_block_or_warning(ctx, True,
            "Physical review remains pending; its source and invocation are retained for collection.", "")
        return None
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
    snapshot_paths = classification_paths or None
    if snapshot_paths is None:
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
        snapshot_paths = [
            line.strip() for line in staged_names_raw.splitlines() if line.strip()
        ] or None
        classification_paths = snapshot_paths or []
    return classification_paths, snapshot_paths, None


def _reset_commit_review_state(ctx):
    """One per-call reset for committing and review-only entry points."""
    ctx.last_push_succeeded = False
    ctx._review_advisory = []
    ctx._last_triad_models = []
    ctx._last_scope_model = ""
    ctx._last_triad_raw_results = []
    ctx._last_review_critical_findings = []
    ctx._last_review_block_reason = ""
    ctx._last_review_advisory_findings = []
    ctx._last_scope_raw_result, ctx._last_review_structured = {}, {}
    ctx._review_degraded_reasons = []
    ctx._current_review_tool_name = "commit_reviewed"
    ctx._current_review_retry_key = ctx._current_review_record_id = ""
    ctx._review_reconcile_only = False
    ctx._review_frozen_rows = {}
    ctx._last_review_slot_executions = {}
    ctx._review_custody_lost = False
    ctx._current_review_attempt_number = None
    ctx._author_commit_source = None
    ctx._author_commit_decision = None
    ctx._author_commit_record = None
    ctx._commit_review_status = "unknown"
    ctx._commit_preflight = None
    ctx._commit_review_panel = None


from ouroboros.tools.commit_gate import _return_commit_feedback, settle_commit_review_ledger  # noqa: E402


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
    preflight_reviewer: str = "",
    panel: Any = None,
) -> Dict[str, Any]:
    """The reviewed stage cycle under this commit's panel (``commit_gate.compose_commit_panel``):
    every pool reader of the cycle — the contract fingerprint, the paid roster, the wave's
    seat vectors, the record — sees the composed seats, exactly as a ``review_change`` wave
    runs under its own; ``None`` (and a panel that is the configured pool) reads the pool."""
    from ouroboros.tools.review_change import _panel_in_force

    ctx._commit_review_panel = dict(panel.facts) if panel is not None else None
    with (_panel_in_force(panel) if panel is not None else contextlib.nullcontext()):
        return _reviewed_stage_cycle(
            ctx, commit_message, commit_start, paths=paths, skip_advisory_review=skip_advisory_review,
            skip_advisory_pre_review=skip_advisory_pre_review, skip_tests=skip_tests, goal=goal, scope=scope,
            review_rebuttal=review_rebuttal, came_from_detached_checkout=came_from_detached_checkout,
            require_release_tag=require_release_tag, preflight_reviewer=preflight_reviewer)


def _reviewed_stage_cycle(
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
    preflight_reviewer: str = "",
) -> Dict[str, Any]:
    skip_advisory_pre_review = bool(skip_advisory_review or skip_advisory_pre_review)
    # Subject evidence and memo are scoped to this exact attempt.
    ctx._last_review_subject_trees = set()
    ctx._managed_review_subject_memo = {}
    classification_paths, snapshot_paths, stage_error = _git()._stage_candidate_for_review(
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
    # Free-cycle identity runs before the preflight and any paid dispatch.
    author_source = getattr(ctx, "_author_commit_source", None)
    gate_outcome = None if author_source is not None else _git()._free_cycle_gate(
        ctx, commit_message, commit_start, pre_fingerprint=pre_fingerprint,
        review_rebuttal=review_rebuttal, goal=goal, scope=scope,
    )
    advisory_replay: Optional[Dict[str, Any]] = None
    if author_source is not None:
        from ouroboros.tools.commit_gate import bind_author_commit_candidate
        preflight = bind_author_commit_candidate(ctx, commit_message, pre_fingerprint)
        if preflight:
            from ouroboros.commit_admission import preflight_evidence_unavailable
            return {"status": "blocked", "message": preflight, "block_reason": "infra_failure" if preflight_evidence_unavailable(preflight) else "preflight"}
        advisory_replay = {"advisory_replay": "Explicit current-author continuation; original reviewer facts retained.", "replay_reason": "author_finish"}
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

    prepared_snapshot = compute_snapshot_hash(pathlib.Path(ctx.repo_dir), commit_message, paths=snapshot_paths)
    from ouroboros.review_evidence import capture_commit_review_evidence, pending_commit_review_evidence

    ctx._commit_review_evidence = (
        pending_commit_review_evidence(ctx) if getattr(ctx, "_review_reconcile_only", False)
        else capture_commit_review_evidence(ctx) if advisory_replay is None else {})
    preflight_gate_outcome = None
    if not bool(getattr(ctx, "_review_reconcile_only", False)):
        preflight_gate_outcome = _git()._preflight_and_tests_gate(
            ctx, commit_message, commit_start,
            classification_paths=classification_paths,
            preflight_reviewer=preflight_reviewer,
            skip_advisory_pre_review=skip_advisory_pre_review,
            skip_tests=skip_tests,
            review_rebuttal=review_rebuttal,
            goal=goal, scope=scope,
        )
    if preflight_gate_outcome is not None:
        _release_review_evidence_if_settled(ctx)
        return preflight_gate_outcome
    if not bool(getattr(ctx, "_review_reconcile_only", False)):
        after_preflight = _git()._fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
        changed = compute_snapshot_hash(pathlib.Path(ctx.repo_dir), commit_message, paths=snapshot_paths) != prepared_snapshot
        revalidation = _revalidation_outcome(
            ctx, commit_message, commit_start, pre_fingerprint, after_preflight, worktree_changed=changed,
        )
        if revalidation is not None:
            _release_review_evidence_if_settled(ctx)
            return revalidation
    _git()._record_commit_attempt(
        ctx,
        commit_message,
        "reviewing",
        duration_sec=time.time() - commit_start,
        phase="review",
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        fingerprint_status="pending",
        rebuttal_sha256=str(getattr(ctx, "_current_review_rebuttal_sha256", "") or ""),
        review_contract_fingerprint=str(
            getattr(ctx, "_current_review_contract_fingerprint", "") or ""
        ),
        review_retry_key=str(getattr(ctx, "_current_review_retry_key", "") or ""),
        late_result_pending=bool(getattr(ctx, "_review_reconcile_only", False)),
    )

    if author_source is not None:
        review_err, scope_result, triad_block_reason, triad_advisory = None, None, "", []
        ctx._review_advisory.append("Explicit Advisory author continuation; no fresh reviewer approval was created.")
    elif advisory_replay is not None:
        review_err, scope_result, triad_block_reason, triad_advisory = None, None, "", []
        from ouroboros.tools.commit_gate import disclose_commit_review_replay
        disclose_commit_review_replay(ctx, advisory_replay)
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
            _release_review_evidence_if_settled(ctx)
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
    settle_commit_review_ledger(ctx, commit_message, goal=goal, scope=scope, pre_fingerprint=pre_fingerprint,
                                blocked=blocked, block_reason=block_reason,
                                combined_findings=combined_findings, author_source=author_source, advisory_replay=advisory_replay)
    post_fingerprint = _git()._fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
    if author_source is None and _git()._review_custody_pending(ctx) and (pending_message := _git()._finalize_pending_review(
            ctx, commit_message, commit_start,
            pre_fingerprint=pre_fingerprint, post_fingerprint=post_fingerprint)):
        from ouroboros.config import get_review_enforcement
        if get_review_enforcement() == "advisory" and review_enforcement_blocks("blocking"):
            return _return_commit_feedback(ctx, commit_message, commit_start, pre_fingerprint, post_fingerprint, pending=True)
        return {
            "status": "blocked",
            "message": pending_message,
            "block_reason": (
                "review_custody_lost"
                if bool(getattr(ctx, "_review_custody_lost", False))
                else "review_late_result_pending"
            ),
            "review_record_id": str(getattr(ctx, "_current_review_record_id", "") or ""),
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
    from ouroboros.config import get_review_enforcement
    material = (blocked or combined_findings or getattr(ctx, "_last_review_critical_findings", [])
                or getattr(ctx, "_last_review_advisory_findings", []) or scope_advisory
                or getattr(ctx, "_last_review_block_reason", "") or triad_block_reason
                or getattr(ctx, "_review_degraded_reasons", []) or advisory_replay is not None)
    if (author_source is None and get_review_enforcement() == "advisory" and review_enforcement_blocks("blocking")
            and material and block_reason != "owner_stopped"):
        return _return_commit_feedback(ctx, commit_message, commit_start, pre_fingerprint, post_fingerprint,
                                       reason=block_reason or triad_block_reason or getattr(ctx, "_last_review_block_reason", "") or "author_decision_required", findings=combined_findings or [
                *getattr(ctx, "_last_review_critical_findings", []),
                *(scope_result.critical_findings if scope_result is not None else [])])
    if blocked:
        # Verdicts extend the identical-diff refusal streak; infrastructure does
        # not. Either outcome retains the physical dispatch's paid-cycle fact.
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
            "status": "blocked", "message": blocked_message, "block_reason": block_reason,
            "review_record_id": str(getattr(ctx, "_current_review_record_id", "") or ""),
            "pre_fingerprint": pre_fingerprint,
            "post_fingerprint": post_fingerprint,
            "combined_findings": combined_findings,
        }
    ctx._commit_review_status = ("author_continued" if author_source is not None else
        "passed" if advisory_replay is None and not material and scope_result is not None
        and getattr(scope_result, "status", "") == "responded" and getattr(ctx, "_last_triad_raw_results", []) else "not_confirmed")
    return {
        "status": "passed", "message": "", "review_record_id": str(getattr(ctx, "_current_review_record_id", "") or ""),
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
    preflight_reviewer: str = "",
    reviewers: Optional[List[str]] = None,
    reason: str = "",
) -> Dict[str, Any]:
    from ouroboros.tools.commit_gate import compose_commit_panel
    from ouroboros.tools.review_change import ReviewChangeArgumentError

    skip_advisory_pre_review = bool(skip_advisory_review or skip_advisory_pre_review)
    ctx.last_reviewed_commit_sha = ""
    _git()._reset_commit_review_state(ctx)
    commit_start = time.time()
    if not commit_message.strip():
        return {"status": "failed", "message": _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text="⚠️ ERROR: commit_message must be non-empty."))}
    try:
        panel = compose_commit_panel(ctx, list(reviewers or []), str(reason or ""))
    except ReviewChangeArgumentError as exc:
        return {"status": "failed", "message": _publish_tool_result(ctx, ToolResult(
            status="error", code="TOOL_ARG_ERROR", text=f"⚠️ TOOL_ARG_ERROR: {exc} Nothing was staged, reviewed or recorded."))}
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
            preflight_reviewer=preflight_reviewer,
            panel=panel,
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
            ctx._coupling_review_history = {}
            outcome["message"] = (
                "Cyber Pro: review-only operation completed; independent failures and pending work remain recorded. "
                if not review_enforcement_blocks("blocking") else
                "Review-only cycle completed under advisory enforcement; failed or missing review remains recorded. "
                if "review_technical_failure_advisory" in (getattr(ctx, "_review_degraded_reasons", []) or [])
                else "Review-only cycle passed. "
            ) + ("Commit was not created; the index is retained while review custody is pending."
                 if _git()._review_custody_pending(ctx) else "Commit was not created and the index was unstaged.")
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
        if unstage_warning and 'outcome' in locals():
            message = str(outcome.get("message", "") or "")
            outcome["message"] = f"{message}\n\n---\n{unstage_warning}" if message else unstage_warning
