"""Commit staging, review, continuation and binding. ``tools.git`` re-exports
these functions; ``_git()`` preserves its patchable facade bindings, while
neutral plumbing imports bind their own owner."""

from __future__ import annotations

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, append_published_text

import contextlib
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import time
from typing import Any, Dict, List, Optional

from ouroboros.tools.registry import ToolContext
from ouroboros.tools.git_plumbing import WHOLE_INDEX, _sanitize_git_error, _publish_git_error
from ouroboros.tools.review_helpers import cyber_review_authority

# Keep the public facade's logger name in server/stdout records.
log = logging.getLogger("ouroboros.tools.git")


def _git():
    """Read the public facade at call time so monkeypatches keep one binding."""
    from ouroboros.tools import git

    return git


def _review_custody_pending(ctx: ToolContext, *, counted_only: bool = False) -> bool:
    """Retain physical custody, or judge only counted seats when deciding a hold."""
    triad = list(getattr(ctx, "_last_triad_raw_results", []) or [])
    scope_raw = getattr(ctx, "_last_scope_raw_result", {}) or {}
    scope_rows = list(scope_raw.get("raw_results") or [scope_raw]) if isinstance(scope_raw, dict) else []
    if counted_only:
        from ouroboros.review_ledger import build_rows, counted_pending

        structured = getattr(ctx, "_last_review_structured", {}) or {}
        rows = build_rows({"triad_raw": triad, "structured": structured})
        return counted_pending(rows if structured.get("rows") else [*rows, *scope_rows],
                               pending=bool(getattr(ctx, "_review_custody_lost", False)))
    if bool(getattr(ctx, "_review_custody_lost", False)):
        return True
    return any(row.get("late_result_pending") or row.get("operation_state") in {"in_flight", "custody_lost"}
               for row in [*triad, *scope_rows] if isinstance(row, dict))


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
    tag target; a development marker (BIBLE P9) binds none. The existing durable
    fingerprint fields remain the review-state mechanism; only their input becomes
    complete. ``diff_sha256`` is the frozen
    subject's own patch identity (``review_subject.staged_patch``).
    """
    from ouroboros.tools.release_sync import is_development_marker
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
        expected_tag = f"v{staged_version}" if version_staged and not is_development_marker(staged_version) else ""
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


def _publish_pending_review(ctx: ToolContext, message: str, custody_lost: bool) -> str:
    """Both pending handbacks preserve custody facts beside their returned text."""
    return _publish_tool_result(ctx, ToolResult(
        status="unavailable" if custody_lost else "ok",
        code="LEGACY_UNAVAILABLE" if custody_lost else "OK", text=message,
        meta={"review_pending": True, "review_custody_lost": custody_lost}))


def _finalize_pending_review(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    pre_fingerprint: Dict[str, Any],
    post_fingerprint: Dict[str, Any],
) -> Optional[str]:
    """Retain non-terminal custody and its exact index; no Git effect follows a pending wave."""
    if getattr(ctx, "_review_cyber_pending", "") and cyber_review_authority():
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
    from ouroboros.tools.commit_gate import watch_left_commit_review
    from ouroboros.tools.review_change import left_wave_terms

    if watch_left_commit_review(ctx):
        event = ("A selected outcome or wake time" if any(row.get("reason") in {"helper_result", "review_result", "wake_at"} for row in
                 getattr(ctx, "_review_detached_operations", []) or []) else "An addressed message")
        message += f" {event} ended your wait; {left_wave_terms(commit=True)}"
    # The index is part of this live wave's identity. Retain it for exact
    # reconciliation; rebuilding it from the worktree could review new bytes.
    from ouroboros.tools.review_collection import reference_for
    from ouroboros.review_ledger import ledger_root, load_record
    record = load_record(ledger_root(ctx), str(getattr(ctx, "_current_review_record_id", "") or ""))
    if record:
        message += "\n" + json.dumps({"review_reference": {**reference_for(record), "surface": "commit"}})
    return _publish_pending_review(ctx, message, custody_lost)


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
    phase: str = "blocking_review",
) -> str:
    """Persist a genuine blocked review result (``phase=owner_control`` for the owner's stop), then unstage the diff."""
    _git()._record_commit_attempt(
        ctx,
        commit_message,
        "blocked",
        block_reason=block_reason,
        block_details=combined_msg,
        duration_sec=time.time() - commit_start,
        critical_findings=combined_findings,
        phase=phase,
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
        _git()._unstage_candidate(ctx)
    except Exception as e:
        warning = f"⚠️ GIT_WARNING (reset): {_sanitize_git_error(str(e))}"
        return f"{combined_msg}\n\n---\n{warning}"
    return combined_msg

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


def _index_outside_selection(ctx: ToolContext, commit_message: str, commit_start: float, outside: List[str],
                             attribution: Optional[Dict[str, Any]], rejoin: bool) -> Dict[str, Any]:
    """Refuse an index that already stages paths outside the selection; nothing is synced or staged
    and the index keeps every entry. Task attribution labels each path; without it the paths are
    only outside the selected set."""
    labels = (("excluded_preexisting_dirty", "already changed when this task's baseline was captured"),
              ("uncertain", "attribution uncertain, not in paths"), ("candidates", "task-attributed, not in paths"))
    facts = ", ".join(outside) if attribution is None else "; ".join(f"{path} (" + next(
        (label for key, label in labels if path in (attribution.get(key) or [])), "not attributed to this task") + ")"
        for path in outside)
    message = ("⚠️ GIT_INDEX_OUTSIDE_SELECTION_BLOCKED: the index already stages path(s) outside the selected paths"
               + (" and this task's open-review candidates" if rejoin and attribution is not None else "")
               + f": {facts}. The review and the commit take the whole index, so they would land with it. Nothing "
               "was synced, staged or reviewed; the index and files are unchanged. Add the ones that belong to this "
               "commit to paths, or retry once the others are no longer staged.")
    if not getattr(ctx, "_review_resume_pending", False):
        _git()._record_commit_attempt(ctx, commit_message, "blocked", block_reason="index_outside_selection",
                                      block_details=message, duration_sec=time.time() - commit_start, phase="preflight")
    return {"status": "blocked", "message": message, "block_reason": "index_outside_selection"}


def _stage_candidate_for_review(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    paths: Optional[List[str]],
    came_from_detached_checkout: bool,
    attribution: Optional[Dict[str, Any]] = None,
) -> tuple[List[str], Optional[List[str]], Optional[Dict[str, Any]]]:
    """Stage the candidate and return its paths without invoking any reviewer.

    ``paths=None`` stages the whole tree. A list stages exactly its literal names, and an empty one
    is refused; a name absent from disk that the index already deletes is not added again (Git
    refuses it, #2417), a recreated one is. Before any metadata sync or staging, an index already
    staging a path outside the list (plus VERSION's release carriers when VERSION is listed) is
    refused; a call collecting this task's open review exactly (resumed, or ``collected_open_peer``) keeps its
    attributed and uncertain ones. ``ctx._staged_selection`` records, after the sync and before the add, the paths
    and the staged ones taken in; the exits undo that (``git_plumbing._unstage_candidate``), a refusal nothing.
    """
    resume, repo = bool(getattr(ctx, "_review_resume_pending", False)), pathlib.Path(ctx.repo_dir)
    rejoin = resume or getattr(ctx, "_review_collected_peer", None) is not None  # never a merely possible peer

    def fail(message: str) -> tuple:
        return [], None, _git()._review_cycle_infra_failure(ctx, commit_message, commit_start, message)

    def git_failure(label: str, exc: Exception) -> tuple:
        return fail(_publish_git_error(ctx, f"⚠️ GIT_ERROR ({label}): {_sanitize_git_error(str(exc))}"))

    staged: Dict[str, str] = {}
    gone: List[str] = []  # reviewed entries Git now stages as none (``replaced_by_directories``): only their removal
    if paths is not None:
        try:
            paths = [_git().safe_relpath(path) for path in paths if str(path).strip()]
        except ValueError as exc:
            return fail(f"⚠️ PATH_ERROR: {exc}")
        if not paths:
            return fail("⚠️ PATH_ERROR: paths names no file, so nothing was staged; name the files, or omit paths.")
        from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS
        from ouroboros.tools.review_collection import replaced_by_directories

        try:
            staged, gone = _git()._staged_index(ctx.repo_dir), replaced_by_directories(ctx, paths)
        except Exception as exc:
            return git_failure("staged-status", exc)

        allowed = set(paths) | (set(CARRIER_SPAN_PATHS) if "VERSION" in paths else set())
        if rejoin and attribution is not None:
            allowed |= set(attribution.get("candidates") or []) | set(attribution.get("uncertain") or [])
        if attribution is not None:
            # Carrier expansion never adopts foreign work or freshly unselected uncertainty.
            allowed -= set(attribution.get("excluded_preexisting_dirty") or [])
            if not rejoin:
                allowed -= set(attribution.get("uncertain") or []) - set(paths)
        if outside := sorted(set(staged) - allowed):
            return [], None, _index_outside_selection(ctx, commit_message, commit_start, outside, attribution, rejoin)
    if resume:
        ctx._staged_selection = WHOLE_INDEX if paths is None else sorted(staged)
    else:
        from ouroboros.commit_admission import auto_sync_release_metadata_if_needed

        synced = auto_sync_release_metadata_if_needed(ctx, repo, pathlib.Path(ctx.drive_root), paths)
        if paths is not None and synced:
            paths = sorted(set(paths) | set(synced))
        if paths is None:
            _git()._ensure_gitignore(ctx.repo_dir)
            add_cmd = ["git", "add", "-A"]
        else:
            names = [path for path in paths if path not in gone and (staged.get(path) != "D" or os.path.lexists(repo / path))]
            add_cmd = ["git", "--literal-pathspecs", "add", "--", *names] if names else []
        ctx._staged_selection = WHOLE_INDEX if paths is None else sorted(set(paths) | set(staged))
        try:
            if add_cmd:
                _git().run_cmd(add_cmd, cwd=ctx.repo_dir)
            if gone:  # exact entries, never pathspecs: nothing below them is added
                _git().run_cmd(["git", "update-index", "--force-remove", "--", *gone], cwd=ctx.repo_dir)
        except Exception as exc:
            return git_failure("add", exc)
        if paths is None and not _git()._authorized_managed_update_resolver(ctx):
            removed = _git()._unstage_binaries(ctx.repo_dir)
            if removed:
                log.warning("Unstaged %d binary files: %s", len(removed), removed)
    try:
        status = _git().run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir)
    except Exception as exc:
        return git_failure("status", exc)
    if not status.strip():
        return fail(
            "⚠️ GIT_LOST_WORKTREE_ON_DETACHED_CHECKOUT_FAILED: working tree is clean "
            "after detached HEAD reconciliation. The detached commits may have been "
            "orphaned. Inspect `git reflog` and restore if needed."
            if came_from_detached_checkout else "⚠️ GIT_NO_CHANGES: nothing to commit.")
    try:
        # Re-read the resulting index through the same NUL-delimited owner, including both rename sides.
        classification_paths = list(_git()._staged_index(ctx.repo_dir))
    except Exception as exc:
        return git_failure("staged-status", exc)
    snapshot_paths = classification_paths or None
    if attribution is not None and isinstance(attribution.get("selection"), dict):
        # A rejoin keeps the earlier whole index; describe that subject, not the narrower automatic proposal.
        uncertain, selected = set(attribution.get("uncertain") or []), set(classification_paths)
        attribution["selection"] = {**attribution["selection"],
                                    "uncertain": sorted(uncertain & selected),
                                    "left_out": sorted(uncertain - selected)}
        ctx._commit_attribution = attribution["selection"]
    return classification_paths, snapshot_paths, None


def _receptive_wave(ctx: ToolContext, wave: contextlib.ExitStack, *, review_only: bool) -> tuple:
    """Bind receptive custody over frozen bytes; managed/read-only calls stay synchronous."""
    from ouroboros import review_pause
    from ouroboros.tools.review_subject import gate_reading_checkout

    binding = (None if review_only or _git()._authorized_managed_update_resolver(ctx)
               else review_pause.receptive_binding(ctx, state=getattr(ctx, "_review_wait_selection", None)))
    if binding is not None and getattr(ctx, "_review_reconcile_only", False):
        with review_pause.receptive_author(binding):
            binding = review_pause.rejoin_binding(ctx, str(getattr(ctx, "_current_review_record_id", "") or
                                                           getattr(ctx._pending_review_attempt, "review_record_id", "")))
    subject, unfrozen = None, ""
    try:
        subject = wave.enter_context(gate_reading_checkout(
            ctx, retry_key=str(getattr(ctx, "_current_review_retry_key", "") or ""), receptive=binding is not None,
            retain=lambda: {"review_custody_pending": True} if _git()._review_custody_pending(ctx) else {}))
    except Exception as exc:
        unfrozen = (f"receptive_wait_unavailable: the reviewed subject could not be frozen ({type(exc).__name__}: "
                    f"{exc}); this wave kept the synchronous wait on the live index")
    wave.enter_context(review_pause.receptive_author(binding if subject is not None else None))
    return subject, unfrozen


def _reset_commit_review_state(ctx):
    """One per-call reset for committing and review-only entry points."""
    ctx.last_push_succeeded = False
    ctx._review_advisory = []
    ctx._last_triad_models = []
    ctx._last_scope_model = ""
    ctx._last_triad_raw_results = []
    ctx._last_review_critical_findings = []
    ctx._last_review_block_reason = ctx._review_override_this_call = ""  # a decision this call met (review_overrides)
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
    ctx._collect_commit_source = ctx._review_collected_peer = None  # the reference or exact peer this call collects
    ctx._author_commit_source = ctx._author_commit_scope = None  # the reference's reviewed subject (``author_scope``)
    ctx._author_commit_decision = None
    ctx._author_commit_record = None
    ctx._commit_review_status = "unknown"
    ctx._commit_preflight = None
    ctx._commit_review_panel = None
    ctx._preflight_execution, ctx._review_since, ctx._commit_attribution = None, "", None
    ctx._review_detached_operations = []  # what THIS call's author left (``review_pause.watch_left_review``)
    ctx._staged_selection = None  # what this call staged (``_stage_candidate_for_review``)


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
    review_only: bool = False,
    attribution: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The reviewed stage cycle under this commit's panel (``commit_gate.compose_commit_panel``):
    every pool reader of the cycle — the contract fingerprint, the paid roster, the wave's
    seat vectors, the record — sees the composed seats, exactly as a ``review_change`` wave
    runs under its own; ``None`` (and a panel that is the configured pool) reads the pool.
    ``review_only`` is the non-committing operator cycle, which never reaches Git."""
    from ouroboros.tools.review_change import _panel_in_force

    ctx._commit_review_panel = dict(panel.facts) if panel is not None else None
    with (_panel_in_force(panel) if panel is not None else contextlib.nullcontext()):
        return _reviewed_stage_cycle(
            ctx, commit_message, commit_start, paths=paths, skip_advisory_review=skip_advisory_review,
            skip_advisory_pre_review=skip_advisory_pre_review, skip_tests=skip_tests, goal=goal, scope=scope,
            review_rebuttal=review_rebuttal, came_from_detached_checkout=came_from_detached_checkout,
            require_release_tag=require_release_tag, preflight_reviewer=preflight_reviewer,
            review_only=review_only, attribution=attribution)


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
    review_only: bool = False,
    attribution: Optional[Dict[str, Any]] = None,
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
        attribution=attribution,
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
        if not bool(getattr(ctx, "_review_resume_pending", False)):
            try:
                _git()._unstage_candidate(ctx)
            except Exception:
                pass
            _git()._record_commit_attempt(
                ctx, commit_message, "blocked", block_reason="core_protection_blocked", block_details=msg,
                duration_sec=time.time() - commit_start, critical_findings=[], phase="preflight")
        return {"status": "blocked", "message": msg, "block_reason": "core_protection_blocked"}
    pre_fingerprint = _git()._fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
    if not pre_fingerprint.get("ok"):
        if bool(getattr(ctx, "_review_resume_pending", False)):
            message = "⚠️ REVIEW_BINDING_UNAVAILABLE: cannot verify the exact pending review identity; no new dispatch was started."
        else:
            message = _git()._handle_revalidation_failure(
                ctx,
                commit_message,
                commit_start,
                pre_fingerprint=pre_fingerprint,
                kind="fingerprint_unavailable",
            )
        return {
            "status": "blocked",
            "message": message,
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
            from ouroboros.tools.review import _handle_review_block_or_warning

            ctx._last_review_block_reason = "infra_failure" if preflight_evidence_unavailable(preflight) else "preflight"
            if not cyber_review_authority():  # outside Cyber Pro the continued candidate passes its deterministic checks
                return {"status": "blocked", "message": preflight, "block_reason": ctx._last_review_block_reason}
            _handle_review_block_or_warning(ctx, preflight, "")
        advisory_replay = {"advisory_replay": "Explicit current-author continuation; original reviewer facts retained.", "replay_reason": "author_finish"}
    if gate_outcome is not None:
        if "advisory_replay" in gate_outcome:
            advisory_replay = gate_outcome
        else:
            gate_outcome["message"] = _publish_tool_result(ctx, ToolResult(
                status="blocked", code="LEGACY_BLOCKED", text=gate_outcome["message"]))
            return gate_outcome
    # Only a call resuming its own open attempt continues that call, whose checks ran before its wave was sent;
    # a new call collecting the completed round's usable verdict at $0 (``_free_cycle_gate``) owes its own (#2410).
    resumes_open = bool(getattr(ctx, "_review_reconcile_only", False) and getattr(ctx, "_review_resume_pending", False))
    binding_error = _git()._review_binding_precondition_error(
        pre_fingerprint, require_release_tag=require_release_tag
    )
    if binding_error:
        if not resumes_open:
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
    if resumes_open:  # the seats' frozen exhibit, never this call's tests
        ctx._preflight_execution = pending_commit_review_evidence(ctx, "preflight_execution") or {"status": "unobserved"}
    preflight_gate_outcome = None
    if not resumes_open:
        preflight_gate_outcome = _git()._preflight_and_tests_gate(
            ctx, commit_message, commit_start,
            classification_paths=classification_paths,
            preflight_reviewer=preflight_reviewer,
            skip_advisory_pre_review=skip_advisory_pre_review,
            skip_tests=skip_tests,
            review_rebuttal=review_rebuttal,
            goal=goal, scope=scope, review_only=review_only,
        )
    if preflight_gate_outcome is not None:
        ctx._review_reconcile_only = False  # refused before collecting: leak no no-dispatch flag (as above)
        _release_review_evidence_if_settled(ctx)
        return preflight_gate_outcome
    if not resumes_open:
        after_preflight = _git()._fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
        changed = compute_snapshot_hash(pathlib.Path(ctx.repo_dir), commit_message, paths=snapshot_paths) != prepared_snapshot
        revalidation = _revalidation_outcome(
            ctx, commit_message, commit_start, pre_fingerprint, after_preflight, worktree_changed=changed,
        )
        if revalidation is not None:
            ctx._review_reconcile_only = False
            _release_review_evidence_if_settled(ctx)
            return revalidation
    _git()._record_commit_attempt(
        ctx, commit_message, "reviewing", duration_sec=time.time() - commit_start, phase="review",
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""), fingerprint_status="pending",
        rebuttal_sha256=str(getattr(ctx, "_current_review_rebuttal_sha256", "") or ""),
        review_contract_fingerprint=str(getattr(ctx, "_current_review_contract_fingerprint", "") or ""),
        review_retry_key=str(getattr(ctx, "_current_review_retry_key", "") or ""),
        late_result_pending=bool(getattr(ctx, "_review_reconcile_only", False)),
    )

    if author_source is not None:
        review_err, scope_result, triad_block_reason, triad_advisory = None, None, "", []
        ctx._review_advisory.append("Explicit author continuation; no fresh reviewer approval was created.")
    elif advisory_replay is not None:
        review_err, scope_result, triad_block_reason, triad_advisory = None, None, "", []
        from ouroboros.tools.commit_gate import disclose_commit_review_replay
        disclose_commit_review_replay(ctx, advisory_replay)
    else:
        _git()._install_paid_dispatch_stamp(ctx, commit_message, commit_start, pre_fingerprint)
        try:
            with contextlib.ExitStack() as wave:
                subject, unfrozen = _receptive_wave(ctx, wave, review_only=review_only)
                review_err, scope_result, triad_block_reason, triad_advisory = _git()._run_parallel_review(
                    ctx, commit_message, goal=goal, scope=scope, review_rebuttal=review_rebuttal,
                    review_binding_fingerprint=str(pre_fingerprint.get("fingerprint") or ""),
                    **({"subject": subject} if subject is not None else {}))
            if unfrozen:
                ctx._review_degraded_reasons = [*(getattr(ctx, "_review_degraded_reasons", None) or []), unfrozen]
        finally:
            _git()._reconcile_and_clear_review_roster(ctx)
            _release_review_evidence_if_settled(ctx)
    blocked, combined_msg, block_reason, combined_findings, scope_advisory = _git()._aggregate_review_verdict(
        review_err, scope_result, triad_block_reason, triad_advisory, ctx, commit_message, commit_start, ctx.repo_dir)
    if scope_advisory:
        advisory_list = getattr(ctx, "_review_advisory", None)
        if isinstance(advisory_list, list):
            advisory_list.extend(scope_advisory)
    settle_commit_review_ledger(ctx, commit_message, goal=goal, scope=scope, pre_fingerprint=pre_fingerprint,
                                blocked=blocked, block_reason=block_reason,
                                combined_findings=combined_findings, author_source=author_source, advisory_replay=advisory_replay)
    post_fingerprint = _git()._fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
    # Every committing outcome, PASS and pending too, returns before Git; a review-only cycle outside Cyber Pro
    # returns its material outcome.
    ordinary_advisory = not cyber_review_authority()
    hand_back_all = not review_only
    if author_source is None and _git()._review_custody_pending(ctx, counted_only=True) and (pending_message := _git()._finalize_pending_review(
            ctx, commit_message, commit_start,
            pre_fingerprint=pre_fingerprint, post_fingerprint=post_fingerprint)):
        custody_lost = bool(getattr(ctx, "_review_custody_lost", False))
        if ordinary_advisory or hand_back_all:
            feedback = _return_commit_feedback(ctx, commit_message, commit_start, pre_fingerprint, post_fingerprint, pending=True)
            feedback["message"] = _publish_pending_review(ctx, feedback["message"], custody_lost)
            return feedback
        return {
            "status": "blocked",
            "message": pending_message,
            "block_reason": "review_custody_lost" if custody_lost else "review_late_result_pending",
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
    from ouroboros.review_custody import REVIEW_STOP_WORDS, review_stop_cause
    from ouroboros.deadline_utils import owner_deadline_exhausted_for_context

    material = (blocked or combined_findings or getattr(ctx, "_last_review_critical_findings", [])
                or getattr(ctx, "_last_review_advisory_findings", []) or scope_advisory
                or getattr(ctx, "_last_review_block_reason", "") or triad_block_reason
                or getattr(ctx, "_review_degraded_reasons", []) or advisory_replay is not None)
    stop = review_stop_cause(ctx) or ("owner_deadline" if owner_deadline_exhausted_for_context(ctx) else "")
    if stop:  # a stop of the work, not a review refusal: say what stopped it and keep what the review returned
        # No record id in the words: ``commit_gate.name_review_record`` appends it, and the stopped seats' offer, after.
        record = str(getattr(ctx, "_current_review_record_id", "") or "")
        blocked, block_reason, combined_msg = True, "owner_stopped", (
            f"⚠️ REVIEW_STOPPED: {REVIEW_STOP_WORDS[stop]}; this commit was not made."
            + (" The review's outcome is kept in its review record." if record else "")
            + (f"\n\nWhat the review returned:\n{combined_msg}" if combined_msg else ""))
    if (author_source is None and (hand_back_all or (ordinary_advisory and material))
            and block_reason != "owner_stopped"):
        return _return_commit_feedback(ctx, commit_message, commit_start, pre_fingerprint, post_fingerprint,
                                       reason=block_reason or triad_block_reason or getattr(ctx, "_last_review_block_reason", "") or "author_decision_required", findings=combined_findings or [
                *getattr(ctx, "_last_review_critical_findings", []),
                *(scope_result.critical_findings if scope_result is not None else [])])
    if blocked:  # only the owner's stop reaches here; the attempt keeps the dispatch's paid-cycle fact
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
            block_class=block_class, phase="owner_control",
        )
        if block_reason == "owner_stopped":
            blocked_message = _publish_tool_result(ctx, ToolResult(
                status="blocked", code="LEGACY_BLOCKED", text=blocked_message))
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
            "message": _publish_tool_result(ctx, ToolResult(
                status="blocked", code="LEGACY_BLOCKED", text=overlap_err)),
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
            ctx, commit_message, commit_start, paths=paths,
            skip_advisory_pre_review=skip_advisory_pre_review, goal=goal, scope=scope,
            review_rebuttal=review_rebuttal, preflight_reviewer=preflight_reviewer, panel=panel, review_only=True)
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
                if cyber_review_authority() else
                "Review-only cycle completed; failed or missing review remains recorded. "
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
                _git()._unstage_candidate(ctx)  # nothing when the call staged nothing
        except Exception as exc:
            unstage_warning = f"⚠️ GIT_WARNING (reset): {_sanitize_git_error(str(exc))}"
        _git()._release_git_lock(lock)
        if unstage_warning and 'outcome' in locals():
            message = str(outcome.get("message", "") or "")
            outcome["message"] = append_published_text(ctx, message, f"\n\n---\n{unstage_warning}") if message else unstage_warning
