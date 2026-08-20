"""Staging, advisory/triad/scope review, and reviewed-material binding.

Owns the non-committing half of the reviewed commit workflow: it stages the
candidate, binds the review to the exact staged tree/parents/VERSION, runs the
parallel review, and terminalises blocked or revalidation-failed cycles.
Creating, tagging, and pushing the commit remain with ``tools/git.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import subprocess
import time
from typing import Any, Dict, List, Optional

from ouroboros.runtime_mode_policy import (
    mode_allows_protected_write,
    protected_paths_in,
)
from ouroboros.tools.registry import (
    ToolContext,
    _authorized_managed_update_resolver,
)
from ouroboros.tools.claude_advisory_review import advisory_gate_unavailable
from ouroboros.tools.commit_gate import (
    _check_advisory_freshness,
    _check_overlapping_review_attempt,
    _record_commit_attempt,
    check_blocked_attempt_cap,
)
from ouroboros.tools.review_revalidation import handle_revalidation_failure
from ouroboros.utils import safe_relpath, run_cmd
from ouroboros.tools.parallel_review import run_parallel_review as _run_parallel_review, aggregate_review_verdict as _aggregate_review_verdict
from ouroboros.tools.review_helpers import (
    _run_review_preflight_tests,
    paths_from_name_status,
)
from ouroboros.tools.git_plumbing import (
    _acquire_git_lock,
    _current_runtime_mode,
    _ensure_gitignore,
    _protected_paths_block_message,
    _publish_git_error,
    _publish_review_blocked,
    _release_git_lock,
    _sanitize_git_error,
    _unstage_binaries,
)

log = logging.getLogger(__name__)


def _fingerprint_staged_diff(repo_dir: pathlib.Path) -> Dict[str, Any]:
    """Bind review to the exact commit material, not only a textual diff.

    ``git write-tree`` is the staged snapshot Git will commit. HEAD plus every
    MERGE_HEAD row is the exact parent vector. VERSION is read from the index,
    and a staged VERSION bump binds the expected release tag and any pre-existing
    tag target. The existing durable fingerprint fields remain the review-state
    mechanism; only their input becomes complete.
    """
    try:
        diff_text = run_cmd(
            ["git", "diff", "--cached", "--binary", "--no-ext-diff"],
            cwd=repo_dir,
        )
        tree_sha = run_cmd(["git", "write-tree"], cwd=repo_dir).strip()
        head_sha = run_cmd(["git", "rev-parse", "HEAD^{commit}"], cwd=repo_dir).strip()
        merge_heads: list[str] = []
        git_path = run_cmd(["git", "rev-parse", "--git-path", "MERGE_HEAD"], cwd=repo_dir).strip()
        merge_head_path = pathlib.Path(git_path)
        if not merge_head_path.is_absolute():
            merge_head_path = repo_dir / merge_head_path
        if merge_head_path.exists():
            for raw_sha in merge_head_path.read_text(encoding="utf-8").splitlines():
                raw_sha = raw_sha.strip()
                if not raw_sha:
                    continue
                resolved = run_cmd(
                    ["git", "rev-parse", f"{raw_sha}^{{commit}}"], cwd=repo_dir
                ).strip()
                if resolved and resolved not in merge_heads and resolved != head_sha:
                    merge_heads.append(resolved)
        version_staged = bool(
            run_cmd(
                ["git", "diff", "--cached", "--name-only", "--", "VERSION"],
                cwd=repo_dir,
            ).strip()
        )
        try:
            staged_version = run_cmd(["git", "show", ":VERSION"], cwd=repo_dir).strip()
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
        resolved_commit = run_cmd(
            ["git", "rev-parse", f"{commit_sha}^{{commit}}"], cwd=repo_dir
        ).strip()
        current_head = run_cmd(["git", "rev-parse", "HEAD^{commit}"], cwd=repo_dir).strip()
        actual_tree = run_cmd(
            ["git", "rev-parse", f"{resolved_commit}^{{tree}}"], cwd=repo_dir
        ).strip()
        parent_line = run_cmd(
            ["git", "rev-list", "--parents", "-n", "1", resolved_commit], cwd=repo_dir
        ).strip().split()
        actual_parents = parent_line[1:] if parent_line else []
        actual_version = run_cmd(
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
            tag_target = run_cmd(
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


def _review_cycle_infra_failure(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    message: str,
) -> Dict[str, Any]:
    """Record and return one fail-closed stage-cycle infrastructure result."""
    _record_commit_attempt(
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
    if paths:
        try:
            safe_paths = [safe_relpath(path) for path in paths if str(path).strip()]
        except ValueError as exc:
            error = _review_cycle_infra_failure(
                ctx, commit_message, commit_start, f"⚠️ PATH_ERROR: {exc}"
            )
            return [], None, error
        add_cmd = ["git", "add"] + safe_paths
    else:
        _ensure_gitignore(ctx.repo_dir)
        add_cmd = ["git", "add", "-A"]
    try:
        run_cmd(add_cmd, cwd=ctx.repo_dir)
    except Exception as exc:
        error = _review_cycle_infra_failure(
            ctx,
            commit_message,
            commit_start,
            _publish_git_error(
                ctx,
                f"⚠️ GIT_ERROR (add): {_sanitize_git_error(str(exc))}",
            ),
        )
        return [], None, error
    if not paths and not _authorized_managed_update_resolver(ctx):
        removed = _unstage_binaries(ctx.repo_dir)
        if removed:
            log.warning("Unstaged %d binary files: %s", len(removed), removed)
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir)
    except Exception as exc:
        error = _review_cycle_infra_failure(
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
        return [], None, _review_cycle_infra_failure(
            ctx, commit_message, commit_start, message
        )

    try:
        staged_status_raw = run_cmd(
            ["git", "diff", "--cached", "--name-status", "-M"], cwd=ctx.repo_dir
        )
        classification_paths = paths_from_name_status(staged_status_raw)
    except Exception as exc:
        try:
            staged_names_raw = run_cmd(
                ["git", "diff", "--cached", "--name-only"], cwd=ctx.repo_dir
            )
        except Exception:
            error = _review_cycle_infra_failure(
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
            staged_names_raw = run_cmd(
                ["git", "diff", "--cached", "--name-only"], cwd=ctx.repo_dir
            )
        except Exception as exc:
            error = _review_cycle_infra_failure(
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
    classification_paths, advisory_paths, stage_error = _stage_candidate_for_review(
        ctx,
        commit_message,
        commit_start,
        paths=paths,
        came_from_detached_checkout=came_from_detached_checkout,
    )
    if stage_error is not None:
        return stage_error
    protected_staged_paths = protected_paths_in(classification_paths)
    runtime_mode = _current_runtime_mode()
    if (
        protected_staged_paths
        and not mode_allows_protected_write(runtime_mode)
        and not _authorized_managed_update_resolver(ctx)
    ):
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

    # Route/slot-aware bypass detection (#123): the bare ANTHROPIC_API_KEY probe
    # missed a disabled advisory slot (audited bypass with NO compensating test
    # preflight) and falsely bypassed the keyless delegated route (duplicate
    # hermetic pytest + a false "Advisory bypassed" progress line).
    if skip_advisory_pre_review:
        _advisory_bypassed = True
    else:
        try:
            _advisory_bypassed = advisory_gate_unavailable()
        except ValueError:
            # Malformed slots/route config: fail closed INTO the compensating
            # preflight — an unreadable advisory gate must cost a hermetic
            # pytest run, never silently skip it.
            _advisory_bypassed = True
    # DISCLOSED RESIDUAL (owner decision, this release): this reads the CURRENT
    # advisory availability, not the status of the advisory record that
    # satisfied freshness above. Settings live outside the Git snapshot, so a
    # run that recorded `bypassed` (slot off, or api route with no key) and was
    # then followed by enabling the slot / adding the key reaches the commit
    # with no compensating preflight. That is UNCHANGED from the key-only
    # predicate this replaced — the same transition skipped it before — and the
    # evidence-based alternative (deriving compensation from the matching
    # AdvisoryRunRecord's recorded status) was weighed and deliberately not
    # taken here. What DID change is the same-configuration case, which is
    # where the silent gap actually lived: a disabled slot now costs the
    # preflight instead of skipping both advisory and tests.
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
    binding_error = _review_binding_precondition_error(
        pre_fingerprint, require_release_tag=require_release_tag
    )
    if binding_error:
        _record_commit_attempt(
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
        return {
            "status": "blocked",
            "message": binding_error,
            "block_reason": "review_binding_invalid",
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
        blocked_message = _finalize_blocked_review(
            ctx,
            commit_message,
            commit_start,
            combined_msg=combined_msg,
            block_reason=block_reason,
            combined_findings=combined_findings,
            pre_fingerprint=pre_fingerprint,
            post_fingerprint=post_fingerprint,
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
        return {"status": "failed", "message": _publish_git_error(ctx, f"⚠️ GIT_ERROR (lock): {exc}")}

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
