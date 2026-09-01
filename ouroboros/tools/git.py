"""Git/write tools with advisory, triad, and scope review commit gates."""

from __future__ import annotations

import copy
import json
import hashlib
import logging
import os
import pathlib
import posixpath
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

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
from ouroboros.tools.registry import (
    ToolContext,
    ToolEntry,
    _authorized_managed_update_resolver,
    system_repo_dir_for,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    binding_targets_system_repo,
    build_resolved_resource_binding,
)
from ouroboros.tools.claude_advisory_review import (
    ADVISORY_REVIEW_CHOICE_GUIDANCE,
    advisory_gate_unavailable,
)
from ouroboros.tools.commit_gate import (
    BLOCK_CLASS_INFRA,
    IDENTICAL_DIFF_BLOCK_REASON,
    _check_advisory_freshness,
    _check_overlapping_review_attempt,
    _current_review_tool_name,
    _invalidate_advisory,
    _record_commit_attempt,
    check_identical_verdict_refusal,
    check_review_cycles_ceiling,
    classify_review_block,
    commit_review_contract_fingerprint,
    compute_rebuttal_sha256,
    resolve_root_task_id,
)
from ouroboros.review_cycles import (
    REASON_REVIEW_CYCLES_EXHAUSTED,
    emit_review_cycles_exhausted,
)
from ouroboros.tools.review_revalidation import handle_revalidation_failure
from ouroboros.utils import utc_now_iso, write_text, safe_relpath, run_cmd
from ouroboros.tools.parallel_review import (
    _commit_review_retry_key,
    aggregate_review_verdict as _aggregate_review_verdict,
    run_parallel_review as _run_parallel_review,
)
from ouroboros.tools.review_helpers import (
    _run_review_preflight_tests,
    format_review_history_entry,
    paths_from_name_status,
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
    block_class: str = "",
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
        run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
    except Exception as e:
        warning = f"⚠️ GIT_WARNING (reset): {_sanitize_git_error(str(e))}"
        return f"{combined_msg}\n\n---\n{warning}"
    return combined_msg


def _review_custody_pending(ctx: ToolContext) -> bool:
    """Whether this gate still owns paid work that is not terminal."""
    if bool(getattr(ctx, "_review_custody_lost", False)):
        return True
    triad = list(getattr(ctx, "_last_triad_raw_results", []) or [])
    scope_raw = getattr(ctx, "_last_scope_raw_result", {}) or {}
    scope_rows = list(scope_raw.get("raw_results") or []) if isinstance(scope_raw, dict) else []
    if not scope_rows and isinstance(scope_raw, dict) and scope_raw:
        scope_rows = [scope_raw]
    return any(
        bool(row.get("late_result_pending"))
        or str(row.get("operation_state") or "") in {"in_flight", "custody_lost"}
        for row in [*triad, *scope_rows]
        if isinstance(row, dict)
    )


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
    _record_commit_attempt(
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
    try:
        run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
    except Exception:
        pass
    return message


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


def _repair_managed_merge_head(ctx: ToolContext) -> None:
    """Re-establish MERGE_HEAD for an authorized managed resolver after a
    refusal's index reset, so the resolution is not stranded (same repair the
    blocked-review path performs in ``_repo_commit_push``)."""
    try:
        from supervisor.update_merge import managed_assisted_tx_for, reestablish_merge_head

        tx, _block = managed_assisted_tx_for(
            getattr(ctx, "task_id", ""), getattr(ctx, "task_metadata", None)
        )
        if tx:
            reestablish_merge_head(str(tx.get("target_sha") or ""))
    except Exception:
        log.debug("reestablish_merge_head after free refusal failed", exc_info=True)


def _free_cycle_gate(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    pre_fingerprint: Dict[str, Any],
    review_rebuttal: str,
    goal: str = "",
    scope: str = "",
) -> Optional[Dict[str, Any]]:
    """Max-Review-Cycles gate, run BEFORE advisory freshness and any paid
    dispatch. ``None`` allows a paid attempt. Two free typed outcomes:
    a byte-identical resubmission of a verdict-blocked diff without a NEW
    rebuttal, and an exhausted per-root-task paid-cycle ceiling. Under
    blocking enforcement each is a free refusal dict; under advisory each is
    an ``{"advisory_replay": …}`` marker — the commit proceeds with a loud
    disclosure and WITHOUT buying another review."""
    from ouroboros.config import get_review_enforcement

    fp = pre_fingerprint.get("fingerprint", "")
    rebuttal_sha = compute_rebuttal_sha256(review_rebuttal)
    contract_fp = commit_review_contract_fingerprint()
    ctx._current_review_rebuttal_sha256 = rebuttal_sha
    ctx._current_review_contract_fingerprint = contract_fp
    root_task_id = resolve_root_task_id(ctx)
    retry_key = _commit_review_retry_key(
        ctx,
        commit_message,
        goal=goal,
        scope=scope,
        review_rebuttal=review_rebuttal,
        binding_fingerprint=fp,
    )
    ctx._current_review_retry_key = retry_key
    pending = getattr(ctx, "_pending_review_attempt", None)
    if bool(getattr(ctx, "_review_resume_pending", False)):
        if (
            pending is not None
            and contract_fp
            and str(getattr(pending, "review_retry_key", "") or "") == retry_key
        ):
            ctx._review_reconcile_only = True
            return None
        try:
            run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
        except Exception:
            pass
        return {
            "status": "blocked",
            "message": (
                "⚠️ REVIEWED_ATTEMPT_IN_PROGRESS: paid review work is still "
                "unresolved for different bytes or intent. The new candidate was "
                "not dispatched; reconcile the original exact attempt first."
            ),
            "block_reason": "overlap_guard",
        }
    ctx._review_reconcile_only = False
    identical_msg = check_identical_verdict_refusal(
        ctx, fp, rebuttal_sha256=rebuttal_sha, contract_fingerprint=contract_fp
    )
    ceiling = None if identical_msg else check_review_cycles_ceiling(
        ctx, root_task_id=root_task_id
    )
    if not identical_msg and ceiling is None:
        return None
    enforcement = str(get_review_enforcement() or "")
    reason = IDENTICAL_DIFF_BLOCK_REASON if identical_msg else REASON_REVIEW_CYCLES_EXHAUSTED
    message = identical_msg or str(ceiling["message"])
    if ceiling is not None:
        emit_review_cycles_exhausted(
            getattr(ctx, "event_queue", None), ctx.drive_root,
            surface="commit_gate", task_id=str(getattr(ctx, "task_id", "") or ""),
            cycles_paid=int(ceiling["cycles_paid"]), cap=int(ceiling["cap"]),
            enforcement=enforcement, root_task_id=root_task_id, fingerprint=str(fp),
        )
    if enforcement != "blocking":
        # ADVISORY: neither state hard-blocks a commit — disclose loudly (typed
        # event + result message) and reuse the recorded outcome for free.
        # The identical-replay half of this branch is structurally near-dead
        # TODAY (fable P3-2, deliberate): verdict rows are only minted under
        # blocking enforcement, and the review-contract fingerprint includes
        # enforcement, so blocking-era verdicts never match an advisory-era
        # contract — the streak lapses and the commit degrades to an ordinary
        # PAID review. Kept as defense-in-depth for any future path that
        # records advisory-era verdict rows.
        try:
            from ouroboros.utils import append_jsonl

            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": utc_now_iso(), "type": "commit_review_free_replay",
                "reason": reason, "enforcement": enforcement,
                "task_id": str(getattr(ctx, "task_id", "") or ""),
                "root_task_id": root_task_id,
                "pre_review_fingerprint": str(fp),
            })
        except Exception:
            log.debug("commit_review_free_replay event emission failed", exc_info=True)
        return {"advisory_replay": message, "replay_reason": reason}
    try:
        run_cmd(["git", "reset", "HEAD"], cwd=ctx.repo_dir)
    except Exception:
        pass
    if _authorized_managed_update_resolver(ctx):
        _repair_managed_merge_head(ctx)
    _record_commit_attempt(
        ctx,
        commit_message,
        "blocked",
        block_reason=reason,
        block_details=message,
        duration_sec=time.time() - commit_start,
        phase="preflight",
        pre_review_fingerprint=str(fp),
        rebuttal_sha256=rebuttal_sha,
        review_contract_fingerprint=contract_fp,
    )
    return {
        "status": "blocked",
        "message": message,
        "block_reason": reason,
    }


def _install_paid_dispatch_stamp(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    pre_fingerprint: Dict[str, Any],
) -> None:
    """WRITE-AHEAD paid stamp (Q16 fix round, F2): durably merge ``paid=True``
    onto the current attempt row immediately BEFORE the first physical
    reviewer transport call, on EITHER side — triad and scope dispatch in
    parallel, so scope spend can coexist with a triad assembly overflow, and
    any side dispatching makes the cycle paid. Assembly-only exits (triad fit
    ladder, scope pack signals) never invoke the stamp, so a $0 attempt stays
    outside the ceiling; a crash after dispatch keeps the paid fact
    (write-ahead). The coordinator captures ``ctx._review_paid_stamp`` and
    each route executor invokes that exact object at its point of no return —
    the seam where the L-review lane's two-phase admission slots in."""
    from ouroboros.review_dispatch import ReviewPaidStamp

    ctx._review_reserved_roster = None
    ctx._review_reserved_operations = {}
    reserved_attempt_number = 0
    reserved_retry_key = ""
    reserved_repo_key = ""
    reserved_tool_name = _current_review_tool_name(ctx)
    reserved_task_id = str(getattr(ctx, "task_id", "") or "")

    def _write() -> None:
        nonlocal reserved_attempt_number, reserved_repo_key, reserved_retry_key
        retry_key = str(getattr(ctx, "_current_review_retry_key", "") or "")
        if not retry_key:
            raise RuntimeError("commit review paid write-ahead has no retry identity")
        reserved = getattr(ctx, "_review_reserved_roster", None)
        triad_rows = (
            copy.deepcopy(reserved.get("multi_model_review") or [])
            if isinstance(reserved, dict)
            else None
        )
        scope_rows = (
            copy.deepcopy(reserved.get("scope_review") or [])
            if isinstance(reserved, dict)
            else None
        )
        roster_pending = bool(
            (triad_rows or scope_rows)
            and any(
                bool(row.get("late_result_pending"))
                or str(row.get("operation_state") or "") in {"in_flight", "custody_lost"}
                for row in list(triad_rows or []) + list(scope_rows or [])
                if isinstance(row, dict)
            )
        )
        _record_commit_attempt(
            ctx,
            commit_message,
            "reviewing",
            duration_sec=time.time() - commit_start,
            phase="review",
            pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
            fingerprint_status="pending",
            paid=True,
            rebuttal_sha256=str(getattr(ctx, "_current_review_rebuttal_sha256", "") or ""),
            review_contract_fingerprint=str(
                getattr(ctx, "_current_review_contract_fingerprint", "") or ""
            ),
            review_retry_key=retry_key,
            late_result_pending=(
                roster_pending or bool(getattr(ctx, "_review_reconcile_only", False))
            ),
            triad_raw_results=triad_rows,
            scope_raw_result=(
                {"raw_results": scope_rows} if scope_rows is not None else None
            ),
            _strict=True,
        )
        from ouroboros.review_state import load_state, make_repo_key

        recorded = load_state(pathlib.Path(ctx.drive_root)).latest_attempt_for(
            repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)),
            tool_name=_current_review_tool_name(ctx),
            task_id=str(getattr(ctx, "task_id", "") or ""),
            attempt=int(getattr(ctx, "_current_review_attempt_number", 0) or 0),
        )
        if recorded is None or not recorded.paid or recorded.review_retry_key != retry_key:
            raise RuntimeError("commit review paid write-ahead could not be verified")
        if isinstance(reserved, dict):
            expected = {
                str(row.get("operation_id") or "")
                for row in list(triad_rows or []) + list(scope_rows or [])
                if isinstance(row, dict) and row.get("operation_id")
            }
            scope_raw = (
                recorded.scope_raw_result.get("raw_results")
                if isinstance(recorded.scope_raw_result, dict)
                else []
            )
            actual = {
                str(row.get("operation_id") or "")
                for row in list(recorded.triad_raw_results or []) + list(scope_raw or [])
                if isinstance(row, dict) and row.get("operation_id")
            }
            if not expected or actual != expected:
                raise RuntimeError("commit review custody roster could not be verified")
        reserved_attempt_number = int(recorded.attempt or 0)
        reserved_retry_key = retry_key
        reserved_repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))

    ctx._review_paid_stamp = ReviewPaidStamp(_write, fail_closed=True)

    def _checkpoint_pending_invocation(
        *, surface: str, slot_id: str, operation_id: str, invocation_id: str,
    ) -> None:
        from ouroboros.review_state import (
            checkpoint_pending_review_invocation,
        )

        checkpoint_pending_review_invocation(
            pathlib.Path(ctx.drive_root),
            repo_key=reserved_repo_key,
            tool_name=reserved_tool_name,
            task_id=reserved_task_id,
            attempt=reserved_attempt_number,
            review_retry_key=reserved_retry_key,
            surface=surface,
            slot_id=slot_id,
            operation_id=operation_id,
            invocation_id=invocation_id,
        )

    ctx._review_pending_invocation_checkpoint = _checkpoint_pending_invocation


def _reconcile_and_clear_review_roster(ctx: ToolContext) -> None:
    """Merge the paid wave into its exact roster, then release process state."""
    reserved_roster = getattr(ctx, "_review_reserved_roster", None)
    try:
        if bool(getattr(ctx, "_review_reconcile_only", False)):
            from ouroboros.review_custody import merge_frozen_review_reconciliation

            merge_frozen_review_reconciliation(ctx)
        elif isinstance(reserved_roster, dict):
            from ouroboros.review_custody import reconcile_reserved_review_roster

            reconcile_reserved_review_roster(ctx, reserved_roster)
    finally:
        ctx._review_paid_stamp = None
        ctx._review_pending_invocation_checkpoint = None
        ctx._review_reserved_roster = None
        ctx._review_reserved_operations = {}
        # This mode belongs only to the exact commit-review reconciliation.
        # A reused ToolContext may subsequently dispatch another review surface
        # (notably Skill Review); do not leak the commit-only no-dispatch flag.
        ctx._review_reconcile_only = False


def _review_cycle_infra_failure(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    message: str,
) -> Dict[str, Any]:
    """Record and return one fail-closed stage-cycle infrastructure result."""
    if not bool(getattr(ctx, "_review_resume_pending", False)):
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
            f"⚠️ GIT_ERROR (add): {_sanitize_git_error(str(exc))}",
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
            f"⚠️ GIT_ERROR (status): {_sanitize_git_error(str(exc))}",
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
                f"⚠️ GIT_ERROR (staged-status): {_sanitize_git_error(str(exc))}",
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
                f"⚠️ GIT_ERROR (staged-names): {_sanitize_git_error(str(exc))}",
            )
            return [], None, error
        advisory_paths = [
            line.strip() for line in staged_names_raw.splitlines() if line.strip()
        ] or None
        classification_paths = advisory_paths or []
    return classification_paths, advisory_paths, None


def _tests_preflight_block_message(managed_needs_proof: bool, test_err: str) -> str:
    """The red-preflight block text. Managed merges get the honest mandate
    wording: the env/skip escapes the ordinary text offers do NOT lift the
    managed requirement — they only defer the same mandatory suite to the
    post-commit gate, which would commit and then roll the merge back on red."""
    if managed_needs_proof:
        return (
            "⚠️ TESTS_PREFLIGHT_BLOCKED: The full hermetic suite is MANDATORY for a "
            "managed update resolution and must be green BEFORE review and commit.\n"
            "skip_tests and OUROBOROS_PRE_PUSH_TESTS are not honored for managed "
            "merges: an unproven candidate pays the same suite at the commit gate "
            "and a red result rolls the merge back. Fix the failures below, then "
            "re-run commit_reviewed.\n\n"
            f"{test_err}"
        )
    return (
        "⚠️ TESTS_PREFLIGHT_BLOCKED: Tests must pass before triad + scope review "
        "when advisory is bypassed.\n"
        "Fix the failures below, then re-run commit_reviewed (or drop "
        "skip_advisory_review=True to run the full advisory flow).\n"
        "Set OUROBOROS_PRE_PUSH_TESTS=0 to skip tests entirely.\n\n"
        f"{test_err}"
    )


def _managed_candidate_needs_proof(ctx: ToolContext) -> bool:
    """Managed single-run mandate (Q10): True when the authorized resolver's
    CURRENT candidate tree carries no recorded green-suite proof (advisory ran
    with skip_tests, or the tree changed since) — the compensating preflight
    must then run PRE-commit, before paid review and before any commit exists,
    regardless of skip_tests/doc-only, so a red candidate is fixed in place
    instead of committed and rolled back.

    AUTHORITY (synthesis F2): the proof consulted here is the PROCESS-HELD ctx
    record pinned by ``record_managed_tests_proof`` when the host itself ran
    the suite — never the durable ``tests_evidence`` tx copy, which is a plain
    resolver-writable file (forensic only; a forged tree there must not
    suppress the mandatory run). A restart between the proof run and the
    commit therefore re-runs the suite once."""
    if not _authorized_managed_update_resolver(ctx):
        return False
    try:
        from supervisor.update_merge import worktree_snapshot_tree

        cand_tree, _cand_err = worktree_snapshot_tree("HEAD")
        proofs = getattr(ctx, "_managed_tests_proof_trees", None) or ()
        return not (cand_tree and cand_tree in proofs)
    except Exception:
        log.debug("managed proof check failed; running the preflight", exc_info=True)
        return True


def _subject_binding_mismatch_outcome(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    pre_fingerprint: Dict[str, Any],
    post_fingerprint: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Defense-in-depth for the managed review subject: every gate subject
    built during this attempt recorded its S tree (review_subject,
    surface="gate"). Assert the reviewed subject is EXACTLY the tree the
    binding fingerprints — i.e. the tree this commit writes. Empty set =
    non-managed attempt (or a managed one that never built a subject): nothing
    to assert, returns None. A mismatch returns the typed blocked outcome."""
    subject_trees = {
        str(t) for t in (getattr(ctx, "_last_review_subject_trees", None) or ())
        if str(t or "").strip()
    }
    if not subject_trees:
        return None
    binding_tree = str((pre_fingerprint.get("binding") or {}).get("tree_sha") or "")
    if subject_trees == {binding_tree}:
        return None
    mismatch_msg = (
        "⚠️ REVIEW_BLOCKED: the managed review subject is not bound to "
        "the staged candidate — reviewed subject tree(s) "
        f"{', '.join(sorted(subject_trees))} do not equal the "
        f"review-binding index tree {binding_tree or '(unavailable)'}. "
        "The reviewers judged material that is not the tree this commit "
        "would write; nothing was committed. Re-stage the intended "
        "candidate and retry."
    )
    _record_commit_attempt(
        ctx,
        commit_message,
        "blocked",
        block_reason="review_subject_binding_mismatch",
        block_details=mismatch_msg,
        duration_sec=time.time() - commit_start,
        phase="revalidation",
        # Infra by construction (A1 composition): a binding mismatch is a gate
        # fact, not a reviewer verdict — it must never build a refusal streak
        # nor anchor an identical-diff refusal quote.
        block_class=BLOCK_CLASS_INFRA,
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        fingerprint_status="invalid",
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
    )
    return {
        "status": "blocked",
        "message": mismatch_msg,
        "block_reason": "review_subject_binding_mismatch",
        "pre_fingerprint": pre_fingerprint,
        "post_fingerprint": post_fingerprint,
    }


def _advisory_and_tests_gate(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    classification_paths: List[str],
    advisory_paths: Optional[List[str]],
    skip_advisory_pre_review: bool,
    skip_tests: bool,
) -> Optional[Dict[str, Any]]:
    """Advisory-freshness gate plus the compensating tests preflight (moved
    whole out of ``_run_reviewed_stage_cycle`` at the function-size gate).
    ``None`` = proceed to review."""
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
    _managed_needs_proof = _managed_candidate_needs_proof(ctx)
    if (_advisory_bypassed and not skip_tests and not _doc_only) or _managed_needs_proof:
        try:
            ctx.emit_progress_fn(
                "Managed candidate lacks a pre-commit test proof — running the mandatory "
                "hermetic suite before review..."
                if _managed_needs_proof
                else "Advisory bypassed — running test preflight before triad + scope review..."
            )
        except Exception:
            pass
        from ouroboros.commit_admission import run_tests_preflight_with_proof

        test_err = run_tests_preflight_with_proof(
            ctx, runner=lambda c: _run_review_preflight_tests(c))
        if test_err:
            msg = _tests_preflight_block_message(_managed_needs_proof, test_err)
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
                # reset the identical-diff refusal streak.
                phase="preflight",
            )
            _mark_failed_bypass_advisory_stale(ctx, commit_message, advisory_paths)
            return {
                "status": "blocked",
                "message": msg,
                "block_reason": "tests_preflight_blocked",
            }
        # Q10 single-run: the green preflight IS the managed pre-commit proof —
        # recorded by the shared admission helper (commit_admission SSOT).
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
    return None


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
        if not bool(getattr(ctx, "_review_resume_pending", False)):
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
    pre_fingerprint = _fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
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
    # Free-cycle identity runs before advisory freshness and any paid dispatch.
    gate_outcome = _free_cycle_gate(
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
    advisory_gate_outcome = None
    if not bool(getattr(ctx, "_review_reconcile_only", False)):
        advisory_gate_outcome = _advisory_and_tests_gate(
            ctx, commit_message, commit_start,
            classification_paths=classification_paths,
            advisory_paths=advisory_paths,
            skip_advisory_pre_review=skip_advisory_pre_review,
            skip_tests=skip_tests,
        )
    if advisory_gate_outcome is not None:
        return advisory_gate_outcome
    binding_error = _review_binding_precondition_error(
        pre_fingerprint, require_release_tag=require_release_tag
    )
    if binding_error:
        if not bool(getattr(ctx, "_review_reconcile_only", False)):
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
    _record_commit_attempt(
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
        if replay_reason == IDENTICAL_DIFF_BLOCK_REASON:
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
            f"this commit ({replay_reason}). "
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
        _install_paid_dispatch_stamp(ctx, commit_message, commit_start, pre_fingerprint)
        try:
            review_err, scope_result, triad_block_reason, triad_advisory = _run_parallel_review(
                ctx,
                commit_message,
                goal=goal,
                scope=scope,
                review_rebuttal=review_rebuttal,
                review_binding_fingerprint=str(pre_fingerprint.get("fingerprint") or ""),
            )
        finally:
            _reconcile_and_clear_review_roster(ctx)
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
    if _review_custody_pending(ctx):
        return {
            "status": "blocked",
            "message": _finalize_pending_review(
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
    _subject_mismatch = _subject_binding_mismatch_outcome(
        ctx, commit_message, commit_start, pre_fingerprint, post_fingerprint
    )
    if _subject_mismatch is not None:
        return _subject_mismatch
    if blocked:
        # Typed block-row classification (Q16/Δ5): a reviewer VERDICT builds
        # the identical-diff refusal streak; an INFRA fact (fit/quorum/
        # transport/sub-floor) never does and retries freely. Money is a
        # separate axis: paid was stamped at PHYSICAL dispatch, so a
        # dispatched-then-infra-blocked wave still counts toward the ceiling,
        # while an assembly-refused (undispatched) infra block stays free.
        block_class = classify_review_block(
            triad_blocked=bool(review_err),
            triad_block_reason=str(triad_block_reason or ""),
            scope_blocked=bool(scope_result is not None and getattr(scope_result, "blocked", False)),
            scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}) or {},
        )
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
                block_class=block_class,
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
    ctx._current_review_retry_key = ""
    ctx._review_reconcile_only = False
    ctx._review_frozen_rows = {}
    ctx._review_custody_lost = False
    ctx._current_review_attempt_number = None
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
        if not bool(getattr(ctx, "_review_resume_pending", False)):
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


def _managed_committing_phase_error(managed_tx: Dict[str, Any]) -> Optional[str]:
    """Enter ``committing_assisted`` via a merge-write on the FRESH durable tx.

    Returns ``None`` on success. A CORRUPT marker returns the typed failure
    message WITHOUT writing (synthesis F1): silently replacing an unreadable
    marker with the caller's stale snapshot would destroy the corruption
    evidence ``read_update_tx_strict`` fails closed to preserve — the refusal
    here is consistent with the corrupt-marker block ``managed_assisted_tx_for``
    already applies at the commit preflight."""
    from supervisor.update_merge import UpdateTxCorrupt, update_tx_phase

    try:
        update_tx_phase(managed_tx, {"phase": "committing_assisted"})
        return None
    except UpdateTxCorrupt as exc:
        return (
            "⚠️ MANAGED_UPDATE_ERROR: the durable update tx marker is corrupt "
            f"({exc}). Refusing the commit-phase transition: the marker bytes are "
            "preserved for recovery, nothing was committed, and restart/recovery "
            "is required."
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
    can wave a managed merge through untested — but the mandate is "the full
    suite provably ran green on the exact committed tree", not "run it twice":
    when the resolver's pre-commit run (advisory preflight or the compensating
    bypass preflight) pinned a PROCESS-HELD proof for a tree byte-identical to
    the committed one, that proof is reused and the duplicate run is skipped
    (Q10). The authority is the host-written ctx record (synthesis F2) — the
    durable ``tests_evidence`` tx copy is resolver-writable forensics and a
    forged tree there never suppresses this run; a restart between the proof
    and the commit re-runs the suite once. The terminal record carries the
    same review metadata/fingerprints as every sibling failure record, so an
    operator can reconstruct WHICH reviewed revision the gate rejected."""
    if not managed_tx:
        return None
    del skip_tests  # deliberately ignored for managed merges
    try:
        committed_tree = run_cmd(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=ctx.repo_dir
        ).strip()
    except Exception:
        committed_tree = ""
    proofs = getattr(ctx, "_managed_tests_proof_trees", None) or ()
    if committed_tree and committed_tree in proofs:
        try:
            ctx.emit_progress_fn(
                "Managed post-commit tests: reusing the green pre-commit hermetic "
                "run (exact tree match) — no duplicate suite run."
            )
        except Exception:
            pass
        return None
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


def _binding_repo_rel(binding: ResolvedResourceBinding) -> str:
    return binding.target_path.relative_to(binding.base_path).as_posix()


def _binding_targets_system_repo(ctx: ToolContext, binding: ResolvedResourceBinding) -> bool:
    return binding.base_path.resolve(strict=False) == system_repo_dir_for(ctx).resolve(strict=False)


def _check_shrink_guard(
    binding: ResolvedResourceBinding,
    new_content: str,
    force: bool = False,
) -> Optional[str]:
    """Block likely accidental tracked-file truncation unless force=True."""
    if force:
        return None
    try:
        target = binding.target_path
        file_path = _binding_repo_rel(binding)
        if not target.exists():
            return None
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", safe_relpath(file_path)],
            cwd=str(binding.base_path), capture_output=True, text=True,
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
                mode: str = "overwrite",
                force: bool = False,
                display_root: str = "active_workspace",
                _resolved_binding: (
                    ResolvedResourceBinding | tuple[ResolvedResourceBinding, ...] | None
                ) = None) -> str:
    """Write file(s) to the repo working directory without committing.

    ``mode="append"`` appends instead of overwriting (#447 D2: write_file declares
    the parameter for every root — dropping it here turned a chunked large-file
    write into an overwrite that destroyed every prior chunk while reporting
    success). An append chunk is not a full file, so the full-file syntax guard,
    the shrink guard, and the overwrite diff do not apply to it."""
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

    try:
        if _resolved_binding is None:
            binding_items = tuple(
                build_resolved_resource_binding(
                    ctx, root=display_root, operation="write", path=e["path"],
                )
                for e in write_list
            )
        elif isinstance(_resolved_binding, tuple):
            binding_items = _resolved_binding
        else:
            binding_items = (_resolved_binding,)
        if len(binding_items) != len(write_list):
            return "⚠️ WRITE_ERROR: resolved target count does not match files."
    except Exception as exc:
        return f"⚠️ WRITE_ERROR: could not resolve target: {type(exc).__name__}: {exc}"

    for e, binding in zip(write_list, binding_items):
        norm = normalize_repo_path(_binding_repo_rel(binding))
        if (
            _binding_targets_system_repo(ctx, binding)
            and is_protected_runtime_path(norm)
            and not mode_allows_protected_write(_current_runtime_mode())
            and not _authorized_managed_update_resolver(ctx)
        ):
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

    # Pre-write syntax guard for known formats (from edit_sketch's verification
    # rails, editbench v2): a full-file overwrite that doesn't even parse is
    # never intentional — block BEFORE any write, force bypasses (deliberately
    # invalid fixtures). Runs before the write loop so the batch stays atomic.
    # P3: the force bypass is never silent — a forced write of invalid content
    # still discloses what the guard found in the success message.
    syntax_bypass_notes: List[str] = []
    from ouroboros.tools.edit_ops import _syntax_check

    if mode != "append":  # an append chunk is not a full file — the guard would block every chunk
        for e, binding in zip(write_list, binding_items):
            rel_path = _binding_repo_rel(binding)
            syntax_err = _syntax_check(rel_path, e["content"])
            if not syntax_err:
                continue
            if force:
                syntax_bypass_notes.append(f"{rel_path}: {syntax_err}")
                continue
            return (
                f"⚠️ WRITE_BLOCKED_SYNTAX: {syntax_err} for '{e['path']}'. "
                "Nothing was written. Fix the content, or pass force=true for an "
                "intentionally invalid file."
            )

    written = []
    written_paths: List[str] = []
    overwrite_diffs: List[str] = []
    for e, binding in zip(write_list, binding_items):
        rel_path = _binding_repo_rel(binding)
        # Append can only grow a file, so the truncation shrink-guard does not apply.
        shrink_warning = None if mode == "append" else _check_shrink_guard(binding, e["content"], force=force)
        if shrink_warning:
            if written:
                _invalidate_advisory(
                    ctx,
                    changed_paths=written_paths,
                    mutation_root=binding_items[0].base_path,
                    source_tool="write_file",
                )
            return shrink_warning
        try:
            target = binding.target_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(e["content"])  # append is intentionally NOT atomized
                written.append(f"{display_root}:{rel_path} (+{len(e['content'])} chars appended)")
                written_paths.append(rel_path)
                continue
            old_content: Optional[str] = None
            if target.exists():
                try:
                    old_content = target.read_text(encoding="utf-8")
                except Exception:
                    old_content = None
            write_text(target, e["content"])
            written.append(f"{display_root}:{rel_path} ({len(e['content'])} chars)")
            written_paths.append(rel_path)
            if old_content is not None and old_content != e["content"]:
                from ouroboros.tools.edit_ops import _unified_diff

                overwrite_diffs.append(_unified_diff(rel_path, old_content, e["content"], cap=120))
        except Exception as exc:
            if written:
                _invalidate_advisory(
                    ctx,
                    changed_paths=written_paths,
                    mutation_root=binding_items[0].base_path,
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
        mutation_root=binding_items[0].base_path,
        source_tool="write_file",
    )
    summary = ", ".join(written)
    system_target = _binding_targets_system_repo(ctx, binding_items[0])
    if ctx.is_workspace_mode() and not system_target:
        result = (
            f"✅ Written {len(written)} file(s): {summary}\n"
            "Files are on disk in the active workspace. Do not commit; the headless runner will emit a patch artifact."
        )
    else:
        result = (
            f"✅ Written {len(written)} file(s): {summary}\n"
            "Files are on disk but NOT committed. Run commit_reviewed when ready.\n"
            "⚠️ Advisory pre-review is now stale — run preflight_review before commit_reviewed."
        )
    result += f"\nResolved root: {binding_items[0].base_path}"
    if syntax_bypass_notes:
        result += (
            "\n⚠️ SYNTAX_GUARD_BYPASSED (force=true): "
            + "; ".join(syntax_bypass_notes)
        )
    if overwrite_diffs:
        result += (
            "\nDiff vs the previous version (verify it matches your intent):\n"
            + "\n".join(overwrite_diffs)
        )
    if system_target and any(pathlib.PurePosixPath(item).parts[:1] == ("skills",) for item in written_paths):
        result += (
            "\nℹ️ Native seed boundary: system_repo/skills changed; the installed "
            "data/skills/native copy remains unchanged until launcher reseed."
        )
    protected_written = protected_paths_in(written_paths) if system_target else []
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
    force: bool = False,
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> str:
    """Replace exactly one occurrence of old_str with new_str in a file."""
    if not path or not path.strip():
        return "⚠️ STR_REPLACE_ERROR: path is required."
    if not old_str:
        return "⚠️ STR_REPLACE_ERROR: old_str is required (cannot be empty)."

    existing_tc = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    data_skill_target = None
    task_constraint = existing_tc
    short_form = None
    binding = _resolved_binding
    if binding is not None:
        target = binding.target_path
        invalidation_root = binding.base_path
    elif not ctx.is_workspace_mode():
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

    if binding is None and not ctx.is_workspace_mode() and task_constraint and task_constraint.mode == "skill_repair" and task_constraint.payload_root:
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
    elif binding is None and not ctx.is_workspace_mode():
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
    if binding is None and data_skill_target is None:
        try:
            binding = build_resolved_resource_binding(
                ctx, root=display_root, operation="edit", path=path,
            )
        except Exception as exc:
            return f"⚠️ PATH_ERROR: {exc}"
        target = binding.target_path
        invalidation_root = binding.base_path

    rel_path = _binding_repo_rel(binding) if binding is not None else safe_relpath(path)
    system_target = bool(binding and _binding_targets_system_repo(ctx, binding))
    norm = normalize_repo_path(rel_path)
    if (
        system_target
        and is_protected_runtime_path(norm)
        and not mode_allows_protected_write(_current_runtime_mode())
        and not _authorized_managed_update_resolver(ctx)
    ):
        return protected_write_block_message(
            path=norm,
            runtime_mode=_current_runtime_mode(),
            action="edit",
        )

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

        _shrink_block = _check_data_shrink_guard(target, new_content, force)
        if _shrink_block:
            return _shrink_block
    elif binding is not None:
        _shrink_block = _check_shrink_guard(binding, new_content, force)
        if _shrink_block:
            return _shrink_block
    # X3 hash-bind: the ADMITTED repair task's payload edits CAS-check the
    # repair's own hash chain; drift outside the repair is a typed stale
    # terminalization, never a silent write over foreign changes.
    _repair_cas_constraint = (
        task_constraint
        if task_constraint and task_constraint.mode == "skill_repair"
        and str(getattr(task_constraint, "skill_name", "") or "")
        else None
    )
    if _repair_cas_constraint is not None:
        from ouroboros.skill_repair_admission import repair_write_cas_error

        _cas = repair_write_cas_error(
            pathlib.Path(ctx.drive_root), _repair_cas_constraint,
            task_id=str(getattr(ctx, "task_id", "") or ""),
            # Mandatory only for a real repair TASK; a synthesized short-form
            # selector on an ordinary edit lane is not an admitted repair.
            repair_task=bool(existing_tc and existing_tc.mode == "skill_repair"))
        if _cas:
            return _cas
    try:
        write_text(target, new_content)
    except Exception as e:
        return f"⚠️ STR_REPLACE_ERROR: write failed for {path}: {e}"
    if _repair_cas_constraint is not None:
        from ouroboros.skill_repair_admission import advance_repair_expected_hash

        advance_repair_expected_hash(
            pathlib.Path(ctx.drive_root), _repair_cas_constraint,
            task_id=str(getattr(ctx, "task_id", "") or ""))

    replacement_line = new_content[:new_content.index(new_str)].count('\n') + 1
    context_start = max(0, replacement_line - 3)
    context_lines = new_content.splitlines()[context_start:replacement_line + len(new_str.splitlines()) + 2]
    context_preview = "\n".join(
        f"{context_start + i + 1:>4}| {line}" for i, line in enumerate(context_lines)
    )

    _invalidate_advisory(
        ctx,
        changed_paths=[rel_path],
        mutation_root=invalidation_root,
        source_tool="edit_text",
    )
    result = (
        f"✅ Replaced in {display_root}:{rel_path} (line {replacement_line}).\n"
        f"Context:\n{context_preview}\n\n"
        "File is on disk but NOT committed."
    )
    if binding is not None:
        result += f"\nResolved root: {binding.base_path}"
    if short_form is not None and short_form.ignored_reason:
        result += f"\n⚠️ SKILL_SHORT_FORM_IGNORED: {short_form.ignored_reason}."
    if data_skill_target is None and ctx.is_workspace_mode() and not system_target:
        result += "\nDo not commit; the headless runner will emit a patch artifact."
    elif data_skill_target is None:
        result += "\nRun commit_reviewed when ready.\n⚠️ Advisory pre-review is now stale — run preflight_review before commit_reviewed."
    else:
        result += "\nRun skill_review for this skill before enabling or declaring it ready."
    if system_target and pathlib.PurePosixPath(rel_path).parts[:1] == ("skills",):
        result += (
            "\nℹ️ Native seed boundary: system_repo/skills changed; the installed "
            "data/skills/native copy remains unchanged until launcher reseed."
        )
    if system_target and is_protected_runtime_path(norm) and mode_allows_protected_write(_current_runtime_mode()):
        result += "\n\n" + core_patch_notice([norm])
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
            return came_from_detached_checkout, f"⚠️ GIT_ERROR (checkout): {error_message}"
        try:
            unmerged = run_cmd(
                ["git", "diff", "--name-only", "--diff-filter=U"], cwd=ctx.repo_dir
            ).strip()
        except Exception as status_exc:
            return came_from_detached_checkout, (
                "⚠️ GIT_ERROR (checkout): "
                f"{error_message}\n\nCould not verify index state after checkout failure: "
                f"{_sanitize_git_error(str(status_exc))}"
            )
        if unmerged:
            return came_from_detached_checkout, (
                "⚠️ GIT_ERROR (checkout): "
                f"{error_message}\n\nRepository has unmerged paths; refusing to treat "
                "the checkout failure as an incidental dirty-tree no-op.\n"
                f"{unmerged}"
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


def _evolution_commit_authority(
    ctx: ToolContext, *, commit_sha: str = "", require_receipt: bool = True,
    require_uncommitted: bool = False,
) -> tuple[Dict[str, str], Dict[str, Any]]:
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    tx = metadata.get("evolution_transaction")
    tx = tx if isinstance(tx, dict) else {}
    claim = {
        "campaign_id": str(tx.get("campaign_id") or ""),
        "transaction_id": str(tx.get("transaction_id") or ""),
        "task_id": str(getattr(ctx, "task_id", "") or tx.get("task_id") or ""),
    }
    from supervisor.evolution_lifecycle import check_evolution_authority

    authority = check_evolution_authority(
        **claim,
        commit_sha=str(commit_sha or "") if require_receipt else "",
        require_uncommitted=bool(require_uncommitted),
    )
    expected_sha = str(commit_sha or "").strip()
    if authority.get("ok") and expected_sha:
        try:
            head = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
        except Exception as exc:
            authority = {**authority, "ok": False, "reason": f"git_state_unavailable:{exc}"}
        else:
            if head != expected_sha:
                authority = {**authority, "ok": False, "reason": "head_mismatch"}
    return claim, authority


def _check_evolution_commit_stage(
    ctx: ToolContext,
    commit_message: str,
    started_at: float,
    *,
    phase: str,
    commit_sha: str = "",
) -> tuple[Dict[str, str], str]:
    """Recheck the exact evolution claim at a commit/publication boundary."""
    claim, authority = _evolution_commit_authority(
        ctx,
        commit_sha=commit_sha,
        require_receipt=phase != "pre_tag_authority",
        require_uncommitted=phase in {"pre_review_authority", "pre_commit_authority"},
    )
    if authority.get("ok"):
        return claim, ""
    reason = authority.get("reason") or "unknown"
    if phase == "pre_review_authority":
        message = (
            "⚠️ EVOLUTION_AUTHORITY_REVOKED: the exact campaign/transaction/task "
            f"claim is no longer active ({reason}). No reviewer was called and no "
            "commit was created."
        )
    elif phase == "pre_commit_authority":
        message = (
            "⚠️ EVOLUTION_AUTHORITY_REVOKED: review completed, but the exact "
            f"campaign claim disappeared before commit ({reason}). Nothing was committed."
        )
    else:
        message = (
            "⚠️ EVOLUTION_PUBLICATION_STOPPED: Git created reviewed local commit "
            f"{commit_sha}, but campaign authority changed before local tag creation "
            f"({reason}). Nothing was tagged, pushed, or scheduled for restart."
        )
    _record_commit_attempt(
        ctx,
        commit_message,
        "failed" if phase == "pre_tag_authority" else "blocked",
        block_reason="evolution_authority",
        block_details=message,
        duration_sec=time.time() - started_at,
        phase=phase,
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
    )
    return claim, message


def _preserve_evolution_orphan(
    ctx: ToolContext, commit_sha: str, *, created_tag: str = "",
) -> str:
    """Keep an unauthorized local commit inspectable but outside normal push refs.

    Ref containment deliberately never touches the index or worktree: another task may
    have edited tracked bytes after the commit was created. Leaving those bytes visibly
    dirty is safer than aligning them to the rewound branch and losing concurrent work.
    """
    sha = str(commit_sha or "").strip()
    ref_name = f"refs/ouroboros/evolution-orphans/{sha}"
    try:
        resolved = run_cmd(
            ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"], cwd=ctx.repo_dir,
        ).strip()
        head = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
        parent = run_cmd(["git", "rev-parse", f"{sha}^"], cwd=ctx.repo_dir).strip()
        if resolved != sha or head != sha or not parent:
            raise RuntimeError("the unauthorized commit is no longer the exact HEAD")
        branch_ref = run_cmd(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=ctx.repo_dir,
        ).strip()
        if not branch_ref.startswith("refs/heads/"):
            raise RuntimeError("HEAD is not attached to a local branch")
        commands = [
            "start",
            f"update {ref_name} {sha}",
            f"update {branch_ref} {parent} {sha}",
        ]
        tag_note = ""
        tag_name = str(created_tag or "").strip()
        target_oid = ""
        if tag_name:
            try:
                target_commit = run_cmd(
                    ["git", "rev-parse", f"refs/tags/{tag_name}^{{commit}}"], cwd=ctx.repo_dir,
                ).strip()
                target_oid = run_cmd(
                    ["git", "rev-parse", f"refs/tags/{tag_name}"], cwd=ctx.repo_dir,
                ).strip()
            except Exception:
                target_commit = target_oid = ""
            if target_commit == sha and target_oid:
                commands.append(f"delete refs/tags/{tag_name} {target_oid}")
                tag_note = f"; deleted local tag {tag_name}"
        commands.extend(("prepare", "commit"))
        transaction_error = ""
        for _attempt in range(2):
            # BYTES stdin, deliberately not text mode: Python's text pipes translate
            # \n to os.linesep, and on Windows git's --stdin parser rejects the
            # resulting "start\r" as an unknown command — every transaction then
            # silently degraded to the decomposed CAS fallback.
            proc = subprocess.run(
                ["git", "update-ref", "--stdin"],
                cwd=ctx.repo_dir,
                input=("\n".join(commands) + "\n").encode("utf-8"),
                capture_output=True,
                check=False,
            )
            if proc.returncode == 0:
                transaction_error = ""
                break
            transaction_error = (
                proc.stderr.decode("utf-8", "replace").strip()
                or "git update-ref transaction failed"
            )

        # A ref transaction is atomic, so a failed transaction can be decomposed into
        # individually verified CAS operations without risking a partial worktree reset.
        if transaction_error:
            zero_oid = "0" * 40
            try:
                current_orphan = run_cmd(
                    ["git", "rev-parse", "--verify", ref_name], cwd=ctx.repo_dir,
                ).strip()
            except Exception:
                current_orphan = ""
            if current_orphan != sha:
                fallback = subprocess.run(
                    ["git", "update-ref", ref_name, sha, zero_oid],
                    cwd=ctx.repo_dir, text=True, capture_output=True, check=False,
                )
                if fallback.returncode != 0:
                    raise RuntimeError(
                        fallback.stderr.strip() or f"could not create {ref_name}"
                    )

            current_branch = run_cmd(
                ["git", "rev-parse", branch_ref], cwd=ctx.repo_dir,
            ).strip()
            if current_branch == sha:
                fallback = subprocess.run(
                    ["git", "update-ref", branch_ref, parent, sha],
                    cwd=ctx.repo_dir, text=True, capture_output=True, check=False,
                )
                if fallback.returncode != 0:
                    raise RuntimeError(
                        fallback.stderr.strip() or f"could not reset {branch_ref}"
                    )

            if tag_name and target_oid:
                try:
                    current_tag_oid = run_cmd(
                        ["git", "rev-parse", f"refs/tags/{tag_name}"], cwd=ctx.repo_dir,
                    ).strip()
                except Exception:
                    current_tag_oid = ""
                if current_tag_oid == target_oid:
                    fallback = subprocess.run(
                        ["git", "update-ref", "-d", f"refs/tags/{tag_name}", target_oid],
                        cwd=ctx.repo_dir, text=True, capture_output=True, check=False,
                    )
                    if fallback.returncode != 0:
                        raise RuntimeError(
                            fallback.stderr.strip() or f"could not delete tag {tag_name}"
                        )

        final_orphan = run_cmd(
            ["git", "rev-parse", "--verify", ref_name], cwd=ctx.repo_dir,
        ).strip()
        if final_orphan != sha:
            raise RuntimeError("private orphan ref does not resolve to the unauthorized commit")
        final_head = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
        reachable = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, branch_ref],
            cwd=ctx.repo_dir, text=True, capture_output=True, check=False,
        ).returncode == 0
        if reachable:
            raise RuntimeError("the unauthorized commit remains reachable from the active branch")
        if tag_name:
            try:
                final_tag_target = run_cmd(
                    ["git", "rev-parse", f"refs/tags/{tag_name}^{{commit}}"], cwd=ctx.repo_dir,
                ).strip()
            except Exception:
                final_tag_target = ""
            if final_tag_target == sha:
                raise RuntimeError(f"tag {tag_name} still reaches the unauthorized commit")
        branch_note = (
            f"the active branch was reset to {parent[:12]}"
            if final_head == parent
            else f"a concurrent branch update to {final_head[:12]} was preserved"
        )
        return (
            f"The commit remains at private local ref {ref_name}; {branch_note}{tag_note}. "
            "The index and worktree were left untouched for lossless recovery."
        )
    except Exception as exc:
        return (
            "⚠️ EVOLUTION_ORPHAN_CONTAINMENT_FAILED: normal publication remains blocked, "
            f"but the active ref could not be reset safely ({_sanitize_git_error(str(exc))})."
        )


def _record_evolution_commit_receipt(
    ctx: ToolContext,
    commit_message: str,
    started_at: float,
    claim: Dict[str, str],
    commit_sha: str,
    created_tag: str = "",
) -> str:
    """Record the exact reviewed SHA or leave it as an inspectable local orphan."""
    from supervisor.evolution_lifecycle import record_evolution_commit

    receipt = record_evolution_commit(**claim, commit_sha=commit_sha)
    if receipt.get("ok"):
        return ""
    containment = _preserve_evolution_orphan(
        ctx, commit_sha, created_tag=created_tag,
    )
    message = (
        "⚠️ EVOLUTION_COMMIT_ORPHANED: Git created reviewed local commit "
        f"{commit_sha}, but its exact campaign authority disappeared before the "
        f"SHA receipt was recorded ({receipt.get('reason') or 'unknown'}). "
        f"Nothing was pushed or scheduled for restart. {containment}"
    )
    _record_commit_attempt(
        ctx,
        commit_message,
        "failed",
        block_reason="evolution_authority",
        block_details=message,
        duration_sec=time.time() - started_at,
        phase="post_commit_authority",
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
    )
    return message


def _evolution_publication_stopped_result(
    ctx: ToolContext, commit_message: str, commit_sha: str, test_warning: str,
    created_tag: str = "", started_at: float = 0.0,
    fingerprints: Optional[tuple[Dict[str, Any], Dict[str, Any]]] = None,
) -> str:
    """Format a local-only result when the SHA receipt loses authority."""
    if str(ctx.current_task_type or "") != "evolution":
        return ""
    _, authority = _evolution_commit_authority(ctx, commit_sha=commit_sha)
    if authority.get("ok"):
        return ""
    ctx.last_push_succeeded = False
    containment = _preserve_evolution_orphan(
        ctx, commit_sha, created_tag=created_tag,
    )
    pre_fingerprint, post_fingerprint = fingerprints or ({}, {})
    message = (
        "⚠️ EVOLUTION_PUBLICATION_STOPPED: campaign authority changed after "
        f"the local SHA receipt ({authority.get('reason') or 'unknown'}). Nothing "
        f"was pushed and restart remains blocked. {containment}{test_warning}"
    )
    _record_commit_attempt(
        ctx, commit_message, "failed",
        block_reason="evolution_authority", block_details=message,
        duration_sec=time.time() - started_at if started_at else 0.0,
        phase="publication_authority", fingerprint_status="matched",
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        post_review_fingerprint=post_fingerprint.get("fingerprint", ""),
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
        degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []),
    )
    return message


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
    ctx._current_review_retry_key = ""
    ctx._review_reconcile_only = False
    ctx._review_frozen_rows = {}
    ctx._review_custody_lost = False
    ctx._current_review_attempt_number = None
    _commit_start = time.time()
    if not commit_message.strip():
        return "⚠️ ERROR: commit_message must be non-empty."
    ctx._current_review_commit_message = commit_message
    # A managed marker authorizes exactly one reviewed two-parent resolution.
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
    if not bool(getattr(ctx, "_review_resume_pending", False)):
        _record_commit_attempt(ctx, commit_message, "reviewing")
    try:
        lock = _acquire_git_lock(ctx)
    except (TimeoutError, Exception) as e:
        if not bool(getattr(ctx, "_review_resume_pending", False)):
            _record_commit_attempt(ctx, commit_message, "failed",
                                   block_reason="infra_failure",
                                   block_details=f"Git lock: {e}",
                                   duration_sec=time.time() - _commit_start)
        return f"⚠️ GIT_ERROR (lock): {e}"
    test_warning_ref = [""]
    _fail = lambda msg: msg if bool(getattr(ctx, "_review_resume_pending", False)) else (
        _record_commit_attempt(ctx, commit_message, "failed",
            block_reason="infra_failure", block_details=msg,
            duration_sec=time.time() - _commit_start,
            triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
            scope_raw_result=getattr(ctx, "_last_scope_raw_result", {})), msg)[1]
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
            from supervisor.update_merge import managed_assisted_marker_check

            _mok, _merr = managed_assisted_marker_check()
            if not _mok:
                return _fail(_merr)
            # Merge-write onto the FRESH durable tx: ``_managed_tx`` was
            # snapshotted BEFORE the stage cycle, and the compensating tests
            # preflight records ``tests_evidence`` into the durable marker
            # mid-attempt — a wholesale snapshot write here would drop that
            # record. A CORRUPT marker refuses the transition (F1): nothing
            # has been committed yet, so fail closed like the preflight does.
            _phase_error = _managed_committing_phase_error(_managed_tx)
            if _phase_error:
                return _fail(_phase_error)

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
            return err_msg
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


def _limit_git_output(text: str, max_chars: int = 0) -> str:
    limit = int(max_chars or 0)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n⚠️ OUTPUT_TRUNCATED: git output limited to {limit} characters by max_chars."


def _vcs_binding(
    ctx: ToolContext,
    binding: Optional[ResolvedResourceBinding],
    *,
    root: str = "system_repo",
    path: str = ".",
) -> ResolvedResourceBinding:
    """Return the dispatch binding, with a system-repo fallback for direct callers.

    Public Tool API calls always receive a registry-built binding. The fallback
    preserves the historical system-repo target for internal/direct helper calls
    without making handler-local target selection part of the public contract.
    """

    resolved = binding or build_resolved_resource_binding(
        ctx,
        root=root,
        operation="vcs",
        path=path or ".",
    )
    if resolved.operation != "vcs" or resolved.root not in {"active_workspace", "system_repo"}:
        raise ValueError(
            "generic VCS tools require a vcs binding on active_workspace or system_repo"
        )
    return resolved


def _vcs_result(text: str, binding: ResolvedResourceBinding) -> str:
    receipt = f"VCS target: root={binding.root}; repo={binding.base_path}"
    rendered = str(text or "").rstrip()
    return f"{rendered}\n\n{receipt}" if rendered else receipt


def _binding_relative_path(binding: ResolvedResourceBinding, requested: str) -> str:
    if not str(requested or "").strip():
        return ""
    try:
        relative = binding.target_path.relative_to(binding.base_path)
    except ValueError as exc:
        raise ValueError("VCS path escapes the selected repository") from exc
    return str(relative) if str(relative) != "." else ""


def _git_status(
    ctx: ToolContext,
    path: str = "",
    max_chars: int = 0,
    root: str = "system_repo",
    _resolved_binding: Optional[ResolvedResourceBinding] = None,
) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root, path=path or ".")
        cmd = ["git", "status", "--porcelain"]
        if relative := _binding_relative_path(binding, path):
            cmd.extend(["--", safe_relpath(relative)])
        return _vcs_result(
            _limit_git_output(run_cmd(cmd, cwd=binding.base_path), max_chars),
            binding,
        )
    except Exception as e:
        return f"⚠️ GIT_ERROR: {_sanitize_git_error(str(e))}"


def _git_diff(
    ctx: ToolContext,
    staged: bool = False,
    path: str = "",
    stat: bool = False,
    name_only: bool = False,
    max_chars: int = 0,
    root: str = "system_repo",
    _resolved_binding: Optional[ResolvedResourceBinding] = None,
) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root, path=path or ".")
        repo_dir = binding.base_path
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if name_only:
            cmd.append("--name-only")
        elif stat:
            cmd.append("--stat")
        if relative := _binding_relative_path(binding, path):
            cmd.extend(["--", safe_relpath(relative)])
        from ouroboros.protected_artifacts import shell_block_reason as protected_artifact_shell_block_reason

        protected_block = protected_artifact_shell_block_reason(
            ctx, cmd, cwd=str(repo_dir), default_cwd=repo_dir, binding=binding,
        )
        if protected_block:
            return _vcs_result(protected_block, binding)
        return _vcs_result(_limit_git_output(run_cmd(cmd, cwd=repo_dir), max_chars), binding)
    except Exception as e:
        return f"⚠️ GIT_ERROR: {_sanitize_git_error(str(e))}"


def _run_git_network_cmd(cmd: List[str], cwd: pathlib.Path) -> str:
    """``run_cmd``-shaped adapter for network git commands (fetch/push).

    Routes through the shared bounded git runner (wall-clock ceiling, process
    tree kill, ``GIT_TERMINAL_PROMPT=0``, low-speed abort) with the caller's
    repository as the explicit cwd, so a dead network cannot hang the tool
    forever. Keeps ``run_cmd``'s contract: stdout on success, ``RuntimeError``
    with the same message shape on failure.
    """
    from supervisor.update_source import _git_network_bounded

    rc, out, err = _git_network_bounded(list(cmd[1:]), cwd=cwd)
    if rc != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}"
        )
    return out


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
        _run_git_network_cmd(["git", "fetch", "origin"], cwd=repo_dir)
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


def _pull_from_remote(
    ctx: ToolContext,
    root: str = "system_repo",
    _resolved_binding: Optional[ResolvedResourceBinding] = None,
) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root)
        return _vcs_result(_ff_pull(binding.base_path), binding)
    except Exception as e:
        return f"⚠️ PULL_ERROR: {_sanitize_git_error(str(e))}"


def _restore_to_head(ctx: ToolContext, confirm: bool = False,
                     paths: Optional[List[str]] = None,
                     root: str = "system_repo",
                     _resolved_binding: Optional[ResolvedResourceBinding] = None) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root)
    except Exception as e:
        return f"⚠️ RESTORE_ERROR: {_sanitize_git_error(str(e))}"
    repo_dir = binding.base_path
    try:
        # NUL-delimited porcelain via the shared review helper: run_cmd() strips
        # its stdout, and a worktree-only modification renders as " M path", so
        # the stripped first line lost its leading space and the column-based
        # line parser dropped the first CHARACTER of that path — the protected
        # gate then judged "IBLE.md" while the restore acted on BIBLE.md.
        from ouroboros.tools.review_helpers import list_changed_paths_from_git_status

        dirty_files = list_changed_paths_from_git_status(
            pathlib.Path(repo_dir), include_sources_for_renames=True,
        )
    except Exception as e:
        return _vcs_result(f"⚠️ RESTORE_ERROR: git status failed: {e}", binding)
    if not dirty_files:
        return _vcs_result("Nothing to restore — working directory is already clean.", binding)
    targets_system = binding_targets_system_repo(ctx, binding)
    affected_protected = protected_paths_in(dirty_files) if targets_system else []
    # Resolve each requested path to the SETTLED form git itself will act on, and
    # let both the protected-path gate below and the checkout/clean action consume
    # that identical value. A divergent normalizer was a protected-path bypass:
    # the gate used normalize_repo_path() while the action used lstrip("./"),
    # which strips leading '.' AND '/' characters, so `../ouroboros/safety.py`
    # was judged as a different (unprotected) string and then checked out against
    # the real protected file. Collapsing is the load-bearing half: git resolves
    # `..` inside a pathspec, while normalize_repo_path() does not, so a gate fed
    # the uncollapsed string reads `tests/../ouroboros/safety.py` as unprotected
    # and the action still lands on `ouroboros/safety.py`. Judge and act on the
    # collapsed value, and refuse anything that leaves the repository root.
    normalized_paths: List[str] = []
    for _p in (paths or []):
        _raw = str(_p or "").strip()
        if not _raw:
            continue
        if _raw.startswith(":"):
            # Pathspec magic (":/", ":(glob)", ":!", ...) re-scopes or negates a
            # pathspec behind the gate's back; a plain repo-relative path never
            # needs it, so refuse rather than judge a string git reads differently.
            return _vcs_result(
                f"⚠️ RESTORE_ERROR: pathspec magic is not supported: {_raw}", binding,
            )
        _collapsed = posixpath.normpath(normalize_repo_path(_raw))
        if posixpath.isabs(_collapsed) or _collapsed == ".." or _collapsed.startswith("../"):
            return _vcs_result(
                f"⚠️ RESTORE_ERROR: path escapes the repository root: {_raw}", binding,
            )
        normalized_paths.append(_collapsed)
    if paths and not normalized_paths:
        return _vcs_result("⚠️ RESTORE_ERROR: No valid paths provided.", binding)
    if normalized_paths and targets_system:
        # A pathspec is not a path: git expands directories, fnmatch wildcards,
        # and "." to a FILE SET, so judging the requested string alone let
        # `paths=["."]`, `["ouroboros"]`, or `["ouroboros/safety.*"]` reach dirty
        # protected files past a gate that only knew exact names. Judge the
        # damage set instead: the files this restore would actually mutate —
        # dirty tracked matches (checkout reverts them) and untracked matches
        # (`git clean -fd` deletes them) — resolved by git itself from the SAME
        # pathspecs the action below consumes.
        for norm in normalized_paths:
            if is_protected_runtime_path(norm):
                return _vcs_result(
                    f"⚠️ RESTORE_BLOCKED: Cannot restore protected file: {norm}. "
                    "Protected core/contract/release paths must be changed through reviewed commits.",
                    binding,
                )
        try:
            # The SAME pathspec-scoped porcelain the dirty set came from — not
            # `git ls-files`, which resolves against the index and so cannot see
            # a STAGED deletion or rename-away of a protected file (the path is
            # gone from the index but `checkout HEAD -- .` would resurrect it,
            # discarding the protected staged change). Porcelain lists staged
            # deletes, rename sources, and untracked files alike.
            scoped_dirty = list_changed_paths_from_git_status(
                pathlib.Path(repo_dir), normalized_paths,
                include_sources_for_renames=True,
            )
        except Exception as e:
            return _vcs_result(
                f"⚠️ RESTORE_ERROR: could not resolve pathspec matches: {e}", binding,
            )
        damage_protected = sorted({
            f for f in scoped_dirty if f and is_protected_runtime_path(f)
        })
        if damage_protected:
            return _vcs_result(
                f"⚠️ RESTORE_BLOCKED: pathspec matches protected file(s) with uncommitted "
                f"changes: {format_protected_paths(damage_protected)}. "
                "Protected core/contract/release paths must be changed through reviewed commits.",
                binding,
            )
    elif affected_protected:
        return _vcs_result(
            f"⚠️ RESTORE_BLOCKED: Uncommitted changes touch protected file(s): "
            f"{format_protected_paths(affected_protected)}. "
            f"Use paths= to restore specific non-critical files, or resolve manually.",
            binding,
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
        return _vcs_result("\n".join(preview), binding)
    if normalized_paths:
        # Reuse the exact contained, normalized paths the gate above judged.
        try:
            run_cmd(["git", "checkout", "HEAD", "--"] + normalized_paths, cwd=repo_dir)
        except Exception as e:
            return _vcs_result(f"⚠️ RESTORE_ERROR: git checkout failed: {e}", binding)
        try:
            run_cmd(["git", "clean", "-fd", "--"] + normalized_paths, cwd=repo_dir)
        except Exception as e:
            # Do not swallow: the checkout landed but untracked cleanup did not,
            # so the working tree does NOT fully match HEAD. Report it.
            return _vcs_result(
                f"⚠️ RESTORE_PARTIAL: checked out {len(normalized_paths)} path(s), "
                f"but `git clean` failed and untracked files may remain: {e}",
                binding,
            )
        return _vcs_result(f"Restored {len(normalized_paths)} path(s) to HEAD.", binding)
    else:
        try:
            run_cmd(["git", "checkout", "HEAD", "--", "."], cwd=repo_dir)
        except Exception as e:
            return _vcs_result(f"⚠️ RESTORE_ERROR: git checkout failed: {e}", binding)
        try:
            run_cmd(["git", "clean", "-fd"], cwd=repo_dir)
        except Exception as e:
            # A swallowed failure previously let this claim "matches HEAD" while
            # untracked files survived. Surface it instead of asserting a state
            # that was not verified.
            return _vcs_result(
                f"⚠️ RESTORE_PARTIAL: tracked changes discarded, but `git clean` "
                f"failed and untracked files may remain: {e}",
                binding,
            )
        return _vcs_result(
            "All uncommitted changes discarded. Working directory matches HEAD.",
            binding,
        )


def _revert_commit(
    ctx: ToolContext,
    sha: str,
    confirm: bool = False,
    root: str = "system_repo",
    _resolved_binding: Optional[ResolvedResourceBinding] = None,
) -> str:
    try:
        binding = _vcs_binding(ctx, _resolved_binding, root=root)
    except Exception as e:
        return f"⚠️ REVERT_ERROR: {_sanitize_git_error(str(e))}"
    repo_dir = binding.base_path
    sha = sha.strip()
    if not sha:
        return _vcs_result("⚠️ REVERT_ERROR: sha parameter is required.", binding)
    try:
        full_sha = run_cmd(
            ["git", "rev-parse", "--verify", sha], cwd=repo_dir,
        ).strip()
    except Exception:
        return _vcs_result(f"⚠️ REVERT_ERROR: Commit '{sha}' not found.", binding)
    try:
        parents = run_cmd(
            ["git", "rev-list", "--parents", "-1", full_sha], cwd=repo_dir,
        ).strip().split()
    except Exception:
        parents = [full_sha]
    if len(parents) > 2:
        return _vcs_result(
            f"⚠️ REVERT_ERROR: Commit {sha[:8]} is a merge commit ({len(parents)-1} parents). "
            "git revert on merge commits requires specifying a parent.",
            binding,
        )
    try:
        changed_files = run_cmd(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", full_sha],
            cwd=repo_dir,
        ).strip().splitlines()
    except Exception:
        changed_files = []
    protected_changes = (
        protected_paths_in(changed_files)
        if binding_targets_system_repo(ctx, binding)
        else []
    )
    if protected_changes:
        return _vcs_result(
            f"⚠️ REVERT_BLOCKED: Commit {sha[:8]} touches protected file(s): "
            f"{format_protected_paths(protected_changes)}. "
            "Direct vcs_revert cannot create protected-path commits; stage the intended "
            "revert manually and use commit_reviewed so the normal triad + scope review covers it.",
            binding,
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
        return _vcs_result(
            f"This will revert commit {full_sha[:8]}:\n"
            f"  Message: {commit_msg}\n"
            f"  Files changed:\n{diff_stat}\n\n"
            "A new commit will be created that undoes these changes.\n"
            "Call again with confirm=true to proceed.",
            binding,
        )
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir).strip()
    except Exception:
        status = ""
    if status:
        return _vcs_result(
            "⚠️ REVERT_ERROR: Working directory is not clean.\n"
            "Commit or discard changes first (use vcs_restore), then retry.",
            binding,
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
            return _vcs_result(f"⚠️ REVERT_ERROR: git revert failed: {e}", binding)
    finally:
        _release_git_lock(lock)
    return _vcs_result(
        f"Reverted commit {full_sha[:8]}: {commit_msg}\nNew revert commit created.",
        binding,
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
                    "description": "If the previous commit was blocked by reviewers and you disagree, include a counter-argument. The rebuttal is identified by CONTENT: a rebuttal new to the current identical-diff streak buys exactly ONE paid re-review of the unchanged diff; resubmitting the same rebuttal (or none) is refused for free, quoting the recorded verdict."},
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
                "review_rebuttal": {"type": "string", "default": "",
                    "description": "Content-hashed counter-argument to a prior review block: a NEW rebuttal buys exactly one paid re-review of an unchanged diff; a repeated one is refused free."},
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
