"""Git/write tools with the optional preflight, triad, and scope review commit gates."""

from __future__ import annotations

from ouroboros.tools.tool_result import (
    ToolResult, _publish_tool_result, _published_tool_result, _replace_tool_result,
)

import copy
import json
import logging
import os
import pathlib
import subprocess  # noqa: F401
import time
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.config import get_runtime_mode  # noqa: F401
from ouroboros.tools.review_helpers import cyber_review_authority
from ouroboros.test_selection import bind_commit_test_request, commit_test_options, widening_properties
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
    _authorized_managed_update_resolver,
    system_repo_dir_for,  # noqa: F401
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,  # noqa: F401
    binding_targets_system_repo,  # noqa: F401
    build_resolved_resource_binding,  # noqa: F401
)
from ouroboros.tools.commit_gate import (
    BLOCK_CLASS_INFRA,
    _check_overlapping_review_attempt,
    _current_review_tool_name,
    _invalidate_advisory,  # noqa: F401
    _record_commit_attempt,
    record_bound_commit_success, prepare_author_commit_request, exact_open_peer,
    name_review_record,
    check_review_cycles_ceiling,
    classify_review_block,  # noqa: F401
    commit_review_contract_fingerprint,
    compute_rebuttal_sha256,
    commit_preflight_choice_error as _commit_preflight_choice_error,
    resolve_root_task_id,
)
from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED, emit_review_cycles_exhausted
from ouroboros.review_overrides import note_override_landing
from ouroboros.tools.release_sync import is_development_marker
from ouroboros.tools.review_revalidation import handle_revalidation_failure  # noqa: F401
from ouroboros.utils import utc_now_iso, write_text, safe_relpath, run_cmd  # noqa: F401
from ouroboros.tools.parallel_review import (
    _commit_review_retry_key,
    aggregate_review_verdict as _aggregate_review_verdict,  # noqa: F401
    run_parallel_review as _run_parallel_review,  # noqa: F401
)
from ouroboros.tools.review_helpers import (
    _run_review_preflight_tests,
    format_review_history_entry,
    paths_from_name_status,  # noqa: F401
)
from ouroboros.tools.core import _data_skill_path, _str_match_replace, is_skill_control_plane_path  # noqa: F401
from ouroboros.contracts.task_constraint import normalize_task_constraint, resolve_payload_path  # noqa: F401
from ouroboros.contracts.skill_payload_policy import (
    cross_skill_redirect_error,  # noqa: F401
    decide_payload_short_form,  # noqa: F401
)
log = logging.getLogger(__name__)


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
    """Collect paid identity before preflight (an open attempt, a reference, or the usable verdict of the round's
    completed wave, whose replays end when it formed none); only new work reads today's pool or cycle limits."""
    from ouroboros.review_records import RECORDED_REVIEW_AUTHORITY

    if getattr(ctx, "_review_cyber_pending", "") and cyber_review_authority():
        return {"advisory_replay": ctx._review_cyber_pending, "replay_reason": "review_pending"}
    fp = pre_fingerprint.get("fingerprint", "")
    rebuttal_sha = compute_rebuttal_sha256(review_rebuttal)
    pending = getattr(ctx, "_pending_review_attempt", None)
    contract_fp = ((str(getattr(pending, "review_contract_fingerprint", "") or "")
                    if bool(getattr(ctx, "_review_resume_pending", False)) else "")
                   or commit_review_contract_fingerprint())
    ctx._current_review_rebuttal_sha256 = rebuttal_sha
    ctx._current_review_contract_fingerprint = contract_fp
    root_task_id = resolve_root_task_id(ctx)
    retry_key = _commit_review_retry_key(ctx, commit_message, goal=goal, scope=scope, review_rebuttal=review_rebuttal,
                                         binding_fingerprint=fp)
    ctx._current_review_retry_key = retry_key
    from ouroboros.tools.review_collection import bind_commit_collection, recorded_round_choice
    peer = recorded_round_choice(ctx, retry_key) or exact_open_peer(
        ctx, commit_message, goal=goal, scope=scope, review_rebuttal=review_rebuttal, fingerprint=fp)
    if peer is not None:  # this call repeats its exact open attempt (or its round's recorded choice): rejoin it
        ctx._review_resume_pending, ctx._pending_review_attempt, pending = True, peer, peer
        ctx._current_review_attempt_number = int(peer.attempt or 0)
        ctx._current_review_contract_fingerprint = contract_fp = str(peer.review_contract_fingerprint or contract_fp)
        ctx._current_review_retry_key = retry_key = str(peer.review_retry_key)
    error, retry_key = bind_commit_collection(ctx, fp, retry_key)
    if error:
        return error
    if bool(getattr(ctx, "_review_resume_pending", False)):
        if pending is not None and str(getattr(pending, "review_retry_key", "") or "") == retry_key:
            ctx._review_reconcile_only = True
            return None
        if cyber_review_authority():  # Cyber: no second wave and no veto
            from ouroboros.tools.commit_gate import _return_commit_feedback
            return _return_commit_feedback(ctx, commit_message, commit_start, pre_fingerprint, pre_fingerprint, pending=True,
                note="Your still-open paid review reads the live index, so this other intent waits; nothing was dispatched. ")
        return {"status": "blocked", "message": (
            "⚠️ REVIEWED_ATTEMPT_IN_PROGRESS: paid review work is still unresolved for different bytes or "
            "intent. The new candidate was not dispatched; reconcile the original exact attempt first."),
            "block_reason": "overlap_guard"}
    ctx._review_reconcile_only = False
    from ouroboros.review_state import make_repo_key, settle_completed_round
    reusable = settle_completed_round(  # collected with its own roster at $0 after this call's checks
        ctx, pathlib.Path(ctx.drive_root), repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)),
        tool_name=_current_review_tool_name(ctx), task_id=str(getattr(ctx, "task_id", "") or ""),
        review_retry_key=retry_key, exclude_attempt=int(getattr(ctx, "_current_review_attempt_number", 0) or 0))
    if reusable is not None:
        ctx._pending_review_attempt, ctx._review_reconcile_only = reusable, True
        return None
    ceiling = check_review_cycles_ceiling(ctx, root_task_id=root_task_id)
    if ceiling is None:
        return None
    reason, message = REASON_REVIEW_CYCLES_EXHAUSTED, str(ceiling["message"])
    emit_review_cycles_exhausted(
        getattr(ctx, "event_queue", None), ctx.drive_root,
        surface="commit_gate", task_id=str(getattr(ctx, "task_id", "") or ""),
        cycles_paid=int(ceiling["cycles_paid"]), cap=int(ceiling["cap"]),
        enforcement=RECORDED_REVIEW_AUTHORITY, root_task_id=root_task_id, fingerprint=str(fp),
    )
    # The spent allowance is a free replay: the outcome returns for the author's explicit decision.
    try:
        from ouroboros.utils import append_jsonl

        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": utc_now_iso(), "type": "commit_review_free_replay",
            "reason": reason, "enforcement": RECORDED_REVIEW_AUTHORITY,
            "task_id": str(getattr(ctx, "task_id", "") or ""),
            "root_task_id": root_task_id,
            "pre_review_fingerprint": str(fp),
        })
    except Exception:
        log.debug("commit_review_free_replay event emission failed", exc_info=True)
    return {"advisory_replay": message, "replay_reason": reason}


def _install_paid_dispatch_stamp(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    pre_fingerprint: Dict[str, Any],
) -> None:
    """WRITE-AHEAD paid stamp (Q16 fix round, F2): the reservation merges the
    roster onto the current attempt row, owned by this process, before any
    worker starts; ``paid=True`` joins that exact row immediately BEFORE the
    first physical reviewer transport call (``mark_review_attempt_paid``), on
    EITHER side — triad and scope dispatch in parallel, and any side
    dispatching makes the cycle paid. Assembly-only exits, replays and refusals
    before a send never invoke the stamp, and a rejoin's own send stamps the
    attempt it collects, so a $0 attempt stays outside the ceiling; a crash
    after dispatch keeps the paid fact (write-ahead). The coordinator captures
    ``ctx._review_paid_stamp`` and each route executor invokes that exact object
    at its point of no return."""
    from ouroboros.review_dispatch import ReviewPaidStamp
    from ouroboros.review_state import (
        checkpoint_pending_review_invocation, load_state, make_repo_key, mark_review_attempt_paid)

    ctx._review_reserved_roster = None
    ctx._review_reserved_operations = {}
    reserved_attempt_number = 0
    reserved_retry_key = ""
    reserved_repo_key = ""
    reserved_tool_name = _current_review_tool_name(ctx)
    reserved_task_id = str(getattr(ctx, "task_id", "") or "")
    drive = pathlib.Path(ctx.drive_root)

    def _record(*, paid: bool) -> None:
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
            paid=paid,
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
            _bind_owner=True,
        )
        recorded = load_state(drive).latest_attempt_for(
            repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)),
            tool_name=_current_review_tool_name(ctx),
            task_id=str(getattr(ctx, "task_id", "") or ""),
            attempt=int(getattr(ctx, "_current_review_attempt_number", 0) or 0),
        )
        if recorded is None or (paid and not recorded.paid) or recorded.review_retry_key != retry_key:
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

    def _write() -> None:
        if reserved_attempt_number:  # the reserved row: only the paid fact, whenever the first send happens
            mark_review_attempt_paid(drive, repo_key=reserved_repo_key, tool_name=reserved_tool_name,
                                     task_id=reserved_task_id, attempt=reserved_attempt_number,
                                     review_retry_key=reserved_retry_key)
        else:  # a rejoin's own send: the attempt it collects
            _record(paid=True)

    ctx._review_paid_stamp = ReviewPaidStamp(_write, fail_closed=True, reserve=lambda: _record(paid=False))

    def _checkpoint_pending_invocation(
        *, surface: str, slot_id: str, operation_id: str, invocation_id: str,
    ) -> None:
        checkpoint_pending_review_invocation(
            drive,
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
        "⚠️ TESTS_PREFLIGHT_BLOCKED: Tests must pass before the review panel runs and before Git.\n"
        "Fix the failures below, then re-run commit_reviewed "
        "(skip_tests=True for intentionally incomplete work).\n"
        "Set OUROBOROS_PRE_PUSH_TESTS=0 to skip tests entirely.\n\n"
        f"{test_err}"
    )


def _managed_candidate_needs_proof(ctx: ToolContext) -> bool:
    """Managed single-run mandate (Q10): True when the authorized resolver's
    CURRENT candidate workload carries no recorded green-suite proof (advisory ran
    with skip_tests, or the tree changed since) — the compensating preflight
    must then run PRE-commit, before paid review and before any commit exists,
    regardless of skip_tests (the only exemption the ordinary rule knows), so a
    red candidate is fixed in place instead of committed and rolled back.

    AUTHORITY: the proof consulted here is the PROCESS-HELD ctx record the
    hermetic runner pins — never the durable ``tests_evidence`` tx copy, which
    stays forensic. A restart loses that record, so the runner is entered again
    and may reuse a matching unsigned receipt of a complete run — but only a
    process running the code on disk mints or accepts one: the resolver booted
    on the base before its merge landed runs the suite and records nothing
    durable, so reuse starts once a process that loaded the merge ran it (owner
    decision 3A retired synthesis F2's resolver-writable-file premise: the
    receipt is writable too and authenticates no one)."""
    if not _authorized_managed_update_resolver(ctx):
        return False
    try:
        from ouroboros.commit_admission import preflight_test_proof_matches

        return not preflight_test_proof_matches(ctx, ctx.repo_dir)
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
        # fact, not a reviewer verdict.
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


def _preflight_and_tests_gate(
    ctx: ToolContext,
    commit_message: str,
    commit_start: float,
    *,
    classification_paths: List[str],
    preflight_reviewer: str = "",
    skip_advisory_pre_review: bool = False,
    skip_tests: bool = False,
    review_rebuttal: str = "",
    goal: str = "",
    scope: str = "",
    review_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """Ahead of the panel: the free deterministic checks, the tests preflight, then the author's
    preflight when one row was named (decision 3A). The checks and the tests block as before
    (Cyber Pro continues with a warning); the preflight only informs. ``None`` = proceed."""
    from ouroboros.commit_admission import preflight_evidence_unavailable, run_tests_preflight_with_proof
    from ouroboros.tools.commit_gate import deterministic_preflight, run_commit_preflight

    ctx._commit_preflight = {"status": "skipped" if skip_advisory_pre_review else "not_performed", "record_id": ""}
    named = bool(preflight_reviewer) and not skip_advisory_pre_review
    message = deterministic_preflight(ctx, commit_message, classification_paths or None, review_only=review_only)
    reason = "infra_failure" if preflight_evidence_unavailable(message) else "preflight"
    _managed_needs_proof = _managed_candidate_needs_proof(ctx)
    # The retired advisory's own tests rule, for every commit: the suite runs unless the author
    # says skip_tests (documentation has tests too), and a managed resolution always pays it.
    if not message and (not skip_tests or _managed_needs_proof):
        ctx.emit_progress_fn(
            "Managed candidate lacks a pre-commit test proof — running the mandatory hermetic suite before review..."
            if _managed_needs_proof else "Running the tests preflight before the panel...")
        test_err = run_tests_preflight_with_proof(
            ctx, runner=lambda c, **kw: _run_review_preflight_tests(c, **kw))
        if test_err:
            message, reason = _tests_preflight_block_message(_managed_needs_proof, test_err), "tests_preflight_blocked"
        # Q10 single-run: the green preflight IS the managed pre-commit proof —
        # recorded by the shared admission helper (commit_admission SSOT).
    elif not message:
        ctx.emit_progress_fn("Tests preflight skipped (skip_tests).")
    from ouroboros.preflight_receipts import execution_exhibit  # frozen for the author's look and the panel
    if message and reason in {"preflight", "infra_failure"}:
        # This invocation never reached the tests, even if ctx holds an earlier proof.
        ctx._preflight_execution = {"status": "not_run", "reason": (
            f"tests not started: deterministic preflight refused ({reason}): {message}")}
    else:
        ctx._preflight_execution = execution_exhibit(ctx, otherwise="tests preflight did not pass" if reason == (
            "tests_preflight_blocked") else "skipped by the author (skip_tests)" if skip_tests and not _managed_needs_proof else "")
    if message and cyber_review_authority():
        from ouroboros.tools.review import _handle_review_block_or_warning

        ctx._last_review_block_reason = reason
        _handle_review_block_or_warning(ctx, message, "")
    elif message:
        try:
            _unstage_candidate(ctx)  # the tests ran unguarded for minutes: keep paths staged meanwhile
        except Exception:
            pass
        # Preflight, not a review verdict.
        _record_commit_attempt(ctx, commit_message, "blocked", block_reason=reason, block_details=message,
                               duration_sec=time.time() - commit_start, phase="preflight")
        return {"status": "blocked", "message": message, "block_reason": reason}
    if named:
        ctx._commit_preflight = run_commit_preflight(ctx, preflight_reviewer, commit_message=commit_message,
                                                     goal=goal, scope=scope, review_rebuttal=review_rebuttal)
    return None


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
            changed = run_cmd(
                ["git", "diff-tree", "-m", "--no-commit-id", "--name-only", "-r", commit_sha],
                cwd=repo_dir,
            ).strip().splitlines()
            if "VERSION" not in changed:
                return ""
            version = run_cmd(["git", "show", f"{commit_sha}:VERSION"], cwd=repo_dir).strip()
            if not version or is_development_marker(version):
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

def _auto_push(repo_dir: pathlib.Path, *, created_tag: str = "", expected_commit_sha: str = "") -> str:
    try:
        from supervisor.git_ops import push_to_remote
        ok, msg = push_to_remote(created_tag=created_tag, expected_commit_sha=expected_commit_sha)
        if ok:
            return f" [pushed: {msg}]"
        return f" [push skipped: {msg}]"
    except Exception as e:
        log.debug("Auto-push failed (non-fatal): %s", e)
        return " [push failed — no automatic retry]"


def _auto_push_after_review(ctx: ToolContext, *, publication_paused: bool = False, created_tag: str = "", expected_commit_sha: str = "") -> str:
    """A completed commit remains completed when Pause refuses its later push."""
    from ouroboros.owner_pause import OwnerPauseRefused, run_operation

    try:
        if publication_paused:
            raise OwnerPauseRefused("owner_pause")
        return run_operation(ctx, _auto_push, ctx.repo_dir, created_tag=created_tag, expected_commit_sha=expected_commit_sha)
    except OwnerPauseRefused as exc:
        return f" [push not started: owner pause ({exc})]"


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
    # The runner checks HEAD and HEAD~1: deleting the suite blocks;
    # a repository that never had tests stays out of scope.
    try:
        from ouroboros.preflight_runner import run_hermetic_pytest

        # Budget: run_hermetic_pytest's; none for the full suite (5A), else OUROBOROS_PREFLIGHT_TIMEOUT_SEC or default.
        return run_hermetic_pytest(
            pathlib.Path(ctx.repo_dir),
            max_output=MAX_TEST_OUTPUT,
            ctx=ctx, **commit_test_options(ctx, force=force, post_commit=True),
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
    """Return each push workflow's state for the just-pushed SHA, or empty on error."""
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
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                   "User-Agent": "ouroboros-ci-check"}
        import urllib.parse
        runs_url = (
            f"https://api.github.com/repos/{repo}/actions/runs"
            f"?per_page=50&branch={urllib.parse.quote(branch, safe='')}"
            f"&event=push&head_sha={urllib.parse.quote(local_sha, safe='')}"
        )
        with urllib.request.urlopen(urllib.request.Request(runs_url, headers=headers), timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        runs = [r for r in (data.get("workflow_runs") or []) if r.get("head_sha") == local_sha]
        if not runs:
            return "\n\nCI: No push workflow run is registered for this commit."
        # A push may start zero or several workflows, depending on the repository: each is
        # reported under its own name and none stands for the others. The API lists newest
        # first, so the first row per workflow belongs to the latest push of this commit.
        latest: Dict[str, dict] = {}
        for run in runs:
            latest.setdefault(str(run.get("name") or run.get("path") or "workflow"), run)
        states = {
            name: str(run.get("conclusion") or "unknown") if run.get("status") == "completed"
            else str(run.get("status") or "queued").replace("_", " ")
            for name, run in latest.items()
        }
        summary = "; ".join(f"{name}: {state}" for name, state in states.items())
        settled = ("success", "skipped", "neutral")  # A workflow whose jobs all skip is not a failure.
        red = [name for name, run in latest.items()
               if run.get("status") == "completed" and states[name] not in settled]
        if not red:
            if all(state in settled for state in states.values()):
                return f"\n\n✅ CI: registered push runs passed for this commit — {summary}."
            return f"\n\n⏳ CI: push runs in progress — {summary}. Read the results later: github_read(method='checks', target='{local_sha}', repo='{repo}')."
        broken = [name for name in red if states[name] == "failure"]
        first = latest[(broken or red)[0]]  # The run whose number, jobs and URL the note carries.
        headline = f"{', '.join(broken)} FAILED" if broken else f"{red[0]} {states[red[0]].upper()}"
        lines = [f"\n\n⚠️ CI STATUS: {headline} for this commit (run #{first.get('run_number', '?')})",
                 f"  Workflows: {summary}"]
        if broken:
            failed_summary = "unknown job"
            try:
                request = urllib.request.Request(first.get("jobs_url", ""), headers=headers)
                with urllib.request.urlopen(request, timeout=8) as jresp:
                    jdata = json.loads(jresp.read().decode("utf-8"))
                failed_parts = [
                    f"{job.get('name', '?')} → " + next((s.get("name", "?") for s in job.get("steps") or []
                                                         if s.get("conclusion") == "failure"), "?")
                    for job in jdata.get("jobs") or [] if job.get("conclusion") == "failure"]
                failed_summary = "; ".join(failed_parts) or failed_summary
            except Exception:
                pass  # The run number and URL below still locate the failure.
            lines += [f"  Failed: {failed_summary}",
                      "  Fix: investigate the failed jobs; push a fix commit when this commit caused them."]
        return "\n".join(lines + [f"  URL: {first.get('html_url', '')}"])
    except Exception:
        return ""


def _publish_post_commit_test_fact(ctx, result: str, test_warning: str) -> str:
    """Carry "the post-commit tests failed" as a TYPED fact beside the text.

    A commit whose post-commit verification failed is PRESERVED and reported as
    a success with a warning appended, so nothing about the call is an error and
    nothing may make it one. The failing tests are still the most reflection
    worthy thing the task did, and a reader that had to find the word in the
    result body was a keyword gate standing in for a fact the producer holds
    here. The text is returned byte-identical, which the registry's publication
    rule requires.
    """
    if test_warning:
        _publish_tool_result(ctx, ToolResult(
            status="ok", code="OK", text=result, meta={"post_commit_tests": "failed"},
        ))
    return result


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
    """Select the commit branch without orphaning detached work or a staged merge.

    Returns ``(came_from_detached_checkout, error_message)``. A successful checkout clears
    MERGE_HEAD, MERGE_MODE and MERGE_MSG, so a staged merge would commit with one parent: the
    current branch is never checked out again, and no switch runs while a merge is in progress.
    Managed assisted merges keep their live MERGE_HEAD and only run the existing transaction
    precommit verification.
    """
    if managed_tx:
        from supervisor.update_merge import managed_assisted_precommit_verify

        verified, error_message = managed_assisted_precommit_verify(managed_tx)
        return False, "" if verified else error_message
    try:
        current_branch = run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ctx.repo_dir
        ).strip()
        if current_branch != "HEAD":
            # A missing branch must never resolve as a package path or remote guess.
            run_cmd(["git", "show-ref", "--verify", f"refs/heads/{ctx.branch_dev}"], cwd=ctx.repo_dir)
    except Exception as exc:
        return False, (
            f"⚠️ GIT_BRANCH_UNAVAILABLE: cannot verify local branch {ctx.branch_dev!r}. "
            f"Current branch, index and files were preserved. {_sanitize_git_error(str(exc))}"
        )
    if current_branch != ctx.branch_dev:
        try:
            merge_path = ctx.repo_dir / run_cmd(["git", "rev-parse", "--git-path", "MERGE_HEAD"], cwd=ctx.repo_dir)
            try:
                merge_path.stat()
                merging = True
            except FileNotFoundError:
                merging = False
        except Exception as exc:
            return False, f"⚠️ GIT_ERROR (checkout): Could not verify merge state: {_sanitize_git_error(str(exc))}"
        if merging:
            return False, (
                f"⚠️ GIT_MERGE_IN_PROGRESS_BLOCKED: a merge is in progress on {current_branch!r}, and switching "
                f"to {ctx.branch_dev!r} would drop it. Nothing was switched, staged or reviewed: commit or abort "
                "that merge where it is, or redo it on the commit branch.")
        try:
            run_cmd(["git", "checkout", "-B", ctx.branch_dev, "HEAD"] if current_branch == "HEAD"
                    else ["git", "checkout", "--no-guess", ctx.branch_dev, "--"], cwd=ctx.repo_dir)
        except Exception as exc:
            return False, f"⚠️ GIT_ERROR (checkout): {_sanitize_git_error(str(exc))}"
        return current_branch == "HEAD", ""
    # Git's checkout refused an unmerged index; without one, the conflicts are refused here, never staged as resolved.
    try:
        unmerged = run_cmd(["git", "diff", "--name-only", "--diff-filter=U"], cwd=ctx.repo_dir).strip()
    except Exception as exc:
        return False, f"⚠️ GIT_ERROR (checkout): Could not verify index state: {_sanitize_git_error(str(exc))}"
    if unmerged:
        return False, ("⚠️ GIT_ERROR (checkout): Repository has unmerged paths; resolve them or abort the "
                       f"merge before committing.\n{unmerged}")
    return False, ""


def _task_attributed_commit_paths(
    ctx: ToolContext,
    paths: Optional[List[str]],
) -> tuple[Optional[List[str]], Optional[Dict[str, Any]], str, Optional[tuple]]:
    """Resolve candidates and their (results_root, evidence_task_id) binding for baseline advancement.

    The results root must match the producer in agent.py: the task/metadata budget_drive_root,
    else the execution drive. Use the candidate owner's same resolver, not a second copy.
    """
    from ouroboros import body_candidate
    from ouroboros.commit_admission import version_sync_refusal
    from ouroboros.mutation_attribution import attribution_task_id, resolve_attributed_git_paths

    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    # A bound body candidate is attributed like the serving checkout: ``body_candidate.bind`` records it in
    # the lineage's baseline (re-established here when that failed or came late), resolving on ``ctx.repo_dir``.
    if not task_id:
        return paths, None, "", None
    bound = body_candidate.is_bound(ctx)
    attached = body_candidate.attach_attribution(ctx) if bound else {}
    results_root = body_candidate._results_root(ctx)
    root_task_id = str(body_candidate._metadata(ctx).get("root_task_id") or "").strip() or task_id
    evidence_task_id = attribution_task_id(results_root, (root_task_id, task_id), **({"surface_type": "system_repo"} if bound else {"repo_root": ctx.repo_dir}))
    if not evidence_task_id:
        # No host baseline of this repo, or for a bound candidate of the body (a manual ToolContext, a dry-run review,
        # a chat turn's folder-only baseline): the legacy explicit/whole-tree staging contract applies.
        return paths, None, "", None
    selected, attribution, error = resolve_attributed_git_paths(
        results_root,
        evidence_task_id,
        pathlib.Path(ctx.repo_dir),
        paths,
        candidate_root=bound,
        retained=bool(getattr(ctx, "_review_resume_pending", False) or getattr(ctx, "_review_collected_peer", None)),
    )
    # Staging's VERSION sync adds the carriers it rewrites, whole: only work this selection takes may ride along.
    if not (error or getattr(ctx, "_review_resume_pending", False)):
        error = version_sync_refusal(pathlib.Path(ctx.repo_dir), selected, attribution)
        selected = [] if error else selected
    if attached.get("state") == "failed":  # distinct from known foreign work: the record itself is missing
        attribution["attach_failure"] = attached.get("error")
        error += f" Recording this body candidate in the task baseline failed: {attached.get('error')}." if error else ""
    # The selected composition's facts reach the wave's brief, the author's handback and the next epoch.
    ctx._commit_attribution = None if error else attribution.get("selection")
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
    return _publish_post_commit_test_fact(ctx, result + ci_note, test_warning)


def _commit_reviewed(ctx: ToolContext, commit_message: str = "", *args: Any, **kwargs: Any) -> str:
    """The public commit handler: an outcome of a call that wrote a review record names it."""
    if error := bind_commit_test_request(ctx, kwargs):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=error))
    from ouroboros.tools.review_collection import default_commit_continuation, publish_commit_delivery
    if refusal := default_commit_continuation(ctx, kwargs):  # the review comes first, its outcome shown (owner 11.10)
        return publish_commit_delivery(ctx, _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=refusal)))
    return publish_commit_delivery(ctx, name_review_record(ctx, _repo_commit_push(ctx, commit_message, *args, **kwargs)))


_COMMIT_ROOT_REFUSAL = (
    "⚠️ TOOL_ARG_ERROR: commit_reviewed lands in the system repository; review a project copy "
    "with review_change and commit it with ordinary git."
)


def _commit_request_refusals(ctx: ToolContext, *, root: str, preflight_reviewer: str, skipped: bool,
                             continuation: bool, reviewers: Optional[List[str]], reason: str,
                             goal: str = "", scope: str = "", review_rebuttal: str = "", collecting: bool = False,
                             wake_on: Any = None) -> tuple:
    """Validate before staging; a paid wave keeps its recorded panel."""
    from ouroboros.tools.commit_gate import compose_commit_panel, bind_commit_wake
    from ouroboros.tools.review_collection import set_aside_wrong_step_fields

    if str(root or "system_repo") != "system_repo":  # no id to name
        return None, _COMMIT_ROOT_REFUSAL
    wake_on = set_aside_wrong_step_fields(ctx, continuation, wake_on)  # never a refusal (06.11)
    selector = "" if collecting and not (skipped or continuation) else preflight_reviewer
    error = _commit_preflight_choice_error(selector, skipped=skipped, continuation=continuation,
                                         ctx=ctx, goal=goal, scope=scope, review_rebuttal=review_rebuttal)
    if collecting and not error:
        bind_commit_wake(ctx, None, None)
        return None, ""
    if not error:
        try:
            panel = compose_commit_panel(ctx, list(reviewers or []), str(reason or ""))
            bind_commit_wake(ctx, wake_on, panel)
            return panel, ""
        except ValueError as exc:
            error = str(exc)
    return None, f"⚠️ TOOL_ARG_ERROR: {error} Nothing was staged, reviewed or recorded."


def _commit_reviewed_candidate(ctx, commit_message, started_at, managed_tx):
    """Admit the reviewed commit; join outside the launch lock so Pause stays prompt."""
    from ouroboros.owner_pause import OwnerPauseRefused, run_operation
    from ouroboros.tools.review_collection import commit_landing_refusal

    if refusal := commit_landing_refusal(ctx, commit_message, started_at, managed_tx):
        return "", refusal
    try:
        run_operation(ctx, run_cmd, ["git", "commit", "-m", commit_message], cwd=ctx.repo_dir)
        return run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip(), ""
    except OwnerPauseRefused as exc:
        message = (f"⚠️ OWNER_PAUSE_NOT_STARTED: Commit NOT STARTED ({exc}). "
                   "The reviewed candidate and review results are retained; no commit, tag or push was started.")
        _record_commit_attempt(ctx, commit_message, "blocked", block_reason="owner_pause",
                               block_details=message, duration_sec=time.time() - started_at,
                               phase="commit", triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
                               scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}))
        return "", _publish_tool_result(ctx, ToolResult(status="blocked", code="OWNER_PAUSE_NOT_STARTED", text=message))
    except Exception as exc:
        error = f"⚠️ GIT_ERROR (commit): {_sanitize_git_error(str(exc))}"
        if managed_tx:
            from supervisor.update_merge import restore_assisted_resolution_after_commit_error
            restore_assisted_resolution_after_commit_error(managed_tx)
        _record_commit_attempt(ctx, commit_message, "failed",
                               block_reason="infra_failure", block_details=error,
                               duration_sec=time.time() - started_at,
                               triad_models=getattr(ctx, "_last_triad_models", []),
                               scope_model=getattr(ctx, "_last_scope_model", ""),
                               triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
                               scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
                               degraded_reasons=list(getattr(ctx, "_review_degraded_reasons", []) or []))
        return "", error


def _repo_commit_push(ctx: ToolContext, commit_message: str,
                       paths: Optional[List[str]] = None,
                       skip_tests: bool = False,
                       review_rebuttal: str = "",
                       skip_advisory_review: bool = False,
                       skip_advisory_pre_review: bool = False,
                       goal: str = "",
                       scope: str = "", review_reference: Optional[dict] = None,
                       author_disposition: Optional[dict] = None, root: str = "",
                       preflight_reviewer: str = "", reviewers: Optional[List[str]] = None, reason: str = "",
                       wake_on: Any = None, since_review: str = "", continue_seats: Optional[list] = None) -> str:
    """Stage, review, and commit files with unified pre-commit review."""
    from ouroboros import body_candidate  # lazy: the Git tools reach the candidate owner only when committing
    skip_advisory_pre_review = bool(skip_advisory_review or skip_advisory_pre_review)
    preflight_reviewer = str(preflight_reviewer or "").strip()
    _reset_commit_review_state(ctx)
    ctx._review_since = str(since_review or "").strip()  # read when the wave prepares its history
    ctx._left_review_wait = None  # a new commit act is the author's explicit next choice
    continuation, collecting_reference = review_reference is not None or author_disposition is not None, review_reference is not None and author_disposition is None
    from ouroboros.tools.review_collection import collect_commit_reference, collected_open_peer
    if collecting_reference or continue_seats is not None:
        source = collect_commit_reference(ctx, review_reference, commit_message=commit_message, continue_seats=continue_seats,
            mutations=(review_rebuttal, goal, scope, reviewers, reason, preflight_reviewer, ctx._review_since, author_disposition))
        if isinstance(source, str):  # refused, or a settled outcome read back: no staging, no Git effect
            return source
        commit_message = source.commit_message
    # A managed marker authorizes exactly one reviewed two-parent resolution.
    from supervisor.update_merge import managed_assisted_tx_for
    _managed_tx, _managed_block = managed_assisted_tx_for(getattr(ctx, "task_id", ""), getattr(ctx, "task_metadata", None))
    managed_resume = bool(_managed_tx and _managed_tx.get("postcommit_resume"))
    overlap_err, collecting = None, collecting_reference
    # An empty message is refused below as an argument error before any state read; a continuation takes its review's.
    if ((commit_message.strip() or (continuation and not collecting_reference)) and str(root or "system_repo") == "system_repo"
            and not (preflight_reviewer and (skip_advisory_pre_review or continuation))):
        error = None if collecting_reference else prepare_author_commit_request(  # a managed merge keeps its whole index
            ctx, review_reference, author_disposition, review_rebuttal, paths_omitted=paths is None and not _managed_tx)
        if error:
            return error
        commit_message = commit_message if commit_message.strip() else str(getattr(ctx._author_commit_source, "commit_message", "") or "")
        if not managed_resume and not collecting_reference:
            overlap_err = _check_overlapping_review_attempt(ctx)
            collecting = bool(getattr(ctx, "_review_resume_pending", False))
            ctx._review_collected_peer = collected_open_peer(ctx, commit_message, goal, scope, review_rebuttal, paths)
    panel, refusal = _commit_request_refusals(  # before any staging, review or record
        ctx, root=root, preflight_reviewer=preflight_reviewer, skipped=skip_advisory_pre_review,
        continuation=continuation, reviewers=reviewers, reason=reason,
        goal=goal or commit_message, scope=scope, review_rebuttal=review_rebuttal, collecting=collecting, wake_on=wake_on)
    if refusal:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=refusal))
    _commit_start = time.time()
    if not commit_message.strip():
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("⚠️ ERROR: commit_message must be non-empty." + (" Its review's attempt left the recent review history and the ledger record it was read back from keeps no commit message, so name one." if continuation and getattr(ctx, "_author_commit_source", None) is not None and not ctx._author_commit_source.commit_message else ""))))
    ctx._current_review_commit_message = commit_message
    if _managed_block:
        _record_commit_attempt(ctx, commit_message, "blocked",
                               block_reason="managed_update_in_progress", block_details=_managed_block,
                               duration_sec=0.0, phase="preflight")
        return _managed_block
    if managed_resume:
        return _resume_managed_commit(ctx, _managed_tx, _commit_start)
    attribution_binding, attribution_error = None, ""
    if _managed_tx:
        paths = None  # a managed merge always stages the WHOLE resolved tree (ignore paths)
    else:  # an author continuation's omitted paths are the paths its reference reviewed (``review_collection.author_scope``)
        paths, _attribution, attribution_error, attribution_binding = _task_attributed_commit_paths(
            ctx, (getattr(ctx, "_author_commit_scope", None) or {}).get("selected", paths))
    for block_reason, error in (("mutation_attribution", attribution_error), ("overlap_guard", overlap_err)):
        if error:
            _record_commit_attempt(
                ctx,
                commit_message,
                "blocked",
                block_reason=block_reason,
                block_details=error,
                duration_sec=0.0,
                phase="preflight",
            )
            if block_reason == "overlap_guard":
                return _publish_tool_result(ctx, ToolResult(
                    status="blocked", code="LEGACY_BLOCKED", text=error))
            return error
    if not (getattr(ctx, "_review_resume_pending", False) or getattr(ctx, "_review_open_peers", None)):
        _record_commit_attempt(ctx, commit_message, "reviewing")  # open peers: identity decides after staging
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
        came_from_detached_checkout = False
        if not bool(getattr(ctx, "_review_resume_pending", False)):
            came_from_detached_checkout, preparation_error = _prepare_review_commit_worktree(ctx, _managed_tx)
            if preparation_error:
                return _fail(preparation_error)
        evolution_claim: Dict[str, str] = {}
        if str(ctx.current_task_type or "") == "evolution":
            evolution_claim, authority_error = _check_evolution_commit_stage(ctx, commit_message, _commit_start, phase="pre_review_authority")
            if authority_error:
                return authority_error
        outcome = _run_reviewed_stage_cycle(
            ctx, commit_message, _commit_start, paths=paths, attribution=None if _managed_tx else _attribution,
            skip_advisory_pre_review=skip_advisory_pre_review, skip_tests=skip_tests,
            goal=goal, scope=scope, review_rebuttal=review_rebuttal,
            came_from_detached_checkout=came_from_detached_checkout,
            require_release_tag=not bool(_managed_tx), preflight_reviewer=preflight_reviewer, panel=panel)
        if outcome.get("status") != "passed":
            if _managed_tx:
                # Retain managed recovery for interrupted older attempts; current unstaging
                # preserves MERGE_HEAD. Precommit verification still needs the managed parent.
                try:
                    from supervisor.update_merge import reestablish_merge_head

                    reestablish_merge_head(str(_managed_tx.get("target_sha") or ""))
                except Exception:
                    log.debug("reestablish_merge_head after blocked managed review failed", exc_info=True)
            return str(outcome.get("message", "") or "")
        pre_fingerprint = outcome.get("pre_fingerprint", {}) or {}
        post_fingerprint = outcome.get("post_fingerprint", {}) or {}

        if evolution_claim:
            _, authority_error = _check_evolution_commit_stage(ctx, commit_message, _commit_start, phase="pre_commit_authority", fingerprint=post_fingerprint)
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

        from ouroboros.owner_pause import OwnerPauseRefused, run_operation

        commit_sha, commit_error = _commit_reviewed_candidate(ctx, commit_message, _commit_start, _managed_tx)
        if commit_error:
            return commit_error
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
        if _managed_tx:
            # Persist this continuation only when a later phase refuses Pause.
            # It is a commit/binding receipt, never proof that tests or smoke passed.
            _managed_tx.update(merge_commit=commit_sha, postcommit_resume={
                "commit_message": commit_message, "pre_fingerprint": pre_fingerprint,
                "post_fingerprint": post_fingerprint,
                "review_context": {key: getattr(ctx, key, None) for key in (
                    "_last_triad_models", "_last_scope_model", "_last_triad_raw_results",
                    "_last_scope_raw_result", "_review_degraded_reasons", "_author_commit_record")},
            })
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
        publication_paused = False
        if not _managed_tx:
            if evolution_claim:
                _, authority_error = _check_evolution_commit_stage(ctx, commit_message, _commit_start, phase="pre_tag_authority", commit_sha=commit_sha)
                if authority_error:
                    containment = _preserve_evolution_orphan(ctx, commit_sha)
                    message = f"{authority_error}\n\n{containment}"
                    containment_result = _published_tool_result(ctx, None)
                    if isinstance(containment_result, ToolResult) and containment_result.text == containment:
                        return _publish_tool_result(ctx, _replace_tool_result(containment_result, text=message))
                    return message
            reviewed_tag = str(reviewed_binding.get("expected_tag") or "")
            try:
                tag_info = run_operation(
                    ctx, _auto_tag_on_version_bump, pathlib.Path(ctx.repo_dir), commit_message,
                    expected_commit_sha=commit_sha, expected_tag=reviewed_tag)
            except OwnerPauseRefused as exc:
                publication_paused = True
                tag_info = f" [tag not started: owner pause ({exc})]"
            created_tag = reviewed_tag if tag_info == f" [tagged: {reviewed_tag}]" else ""
        binding_ok, binding_detail = _verify_reviewed_commit_binding(
            pathlib.Path(ctx.repo_dir),
            commit_sha,
            post_fingerprint,
            verify_expected_tag=not bool(_managed_tx) and not publication_paused,
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
        if not _managed_tx and not publication_paused:
            # Evolution always runs this gate, full outside a body candidate, even after skip_tests.
            try:
                run_operation(ctx, _post_commit_result, ctx, commit_message, skip_tests, test_warning_ref, force=bool(evolution_claim))
            except OwnerPauseRefused as exc:
                publication_paused = True
                tag_info += f" [post-commit tests not started: owner pause ({exc})]"
        push_status = ""
        if not _managed_tx and evolution_claim:
            publication_error = _evolution_publication_stopped_result(
                ctx, commit_message, commit_sha, test_warning_ref[0],
                created_tag=created_tag, started_at=_commit_start, fingerprints=(pre_fingerprint, post_fingerprint),
            )
            if publication_error:
                return publication_error
            push_status = (_evolution_test_publication_stop(ctx, commit_sha, test_warning_ref[0])
                           or (_auto_push_after_review(ctx, publication_paused=publication_paused, created_tag=created_tag, expected_commit_sha=commit_sha) if not body_candidate.is_bound(ctx) else body_candidate.publication_note(ctx)))
        ctx.last_reviewed_commit_sha = commit_sha
        body_candidate.record_reviewed_commit(ctx, commit_sha, created_tag=created_tag)
        if attribution_binding is not None:
            # The task's own commit moved HEAD: open the next attributed-staging
            # epoch so a follow-up commit does not read as ``baseline_stale``.
            try:
                from ouroboros.mutation_attribution import advance_mutation_baseline

                advance_mutation_baseline(
                    attribution_binding[0],
                    attribution_binding[1],
                    pathlib.Path(ctx.repo_dir),
                    commit=commit_sha, attribution=getattr(ctx, "_commit_attribution", None),
                )
            except Exception:
                log.warning("mutation baseline advance failed after commit", exc_info=True)
        if not _managed_tx:  # a managed merge records its success after its post-commit phase
            record_bound_commit_success(ctx, commit_message, _commit_start, pre_fingerprint, post_fingerprint, commit_sha)
            note_override_landing(ctx, commit_sha)  # landed over a review FAIL or no verdict?
            ctx._coupling_review_history = {}  # the subject's coupling rounds end with its commit (the attempt rows keep them)
    finally:
        _release_git_lock(lock)
    if _managed_tx:
        return _finish_managed_commit(
            ctx, _managed_tx, commit_sha, commit_message, _commit_start, test_warning_ref,
            (pre_fingerprint, post_fingerprint))
    if not evolution_claim:
        push_status = _auto_push_after_review(ctx, publication_paused=publication_paused, created_tag=created_tag, expected_commit_sha=commit_sha) if not body_candidate.is_bound(ctx) else body_candidate.publication_note(ctx)
    return _publish_reviewed_commit(
        ctx, commit_message, commit_sha, tag_info, test_warning_ref[0], paths, push_status,
    )


def _run_git_network_cmd(cmd: List[str], cwd: pathlib.Path) -> str:
    """Bounded network git, returning stdout or raising with stdout/stderr (a work repository's in the work domain)."""
    from ouroboros.tools.git_plumbing import network_git_bounded

    rc, out, err = network_git_bounded(list(cmd[1:]), cwd=cwd)
    if rc != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}"
        )
    return out


def get_tools() -> List[ToolEntry]:
    from ouroboros.review_continuation import CONTINUE_SEATS_SCHEMA
    from ouroboros.review_doctrine import AUTHOR_DISPOSITION_SCHEMA
    from ouroboros.review_waiting import wake_on_schema
    reviewed_commit_contract = (  # one contract, two leads: the older name keeps it whole for callers that see only that name
        "free deterministic checks, the tests preflight, then the review panel (both questions). Every outcome, PASS and "
        "pending included, returns before commit/tag/push and the commit is your explicit "
        "continuation (review_reference + author_disposition). preflight_reviewer optionally buys ONE named row's early look at the worktree first (review_change surface=preflight): it informs, never "
        "gates, and its record never answers the panel. Without it the commit records preflight not_performed, "
        "a fact rather than a bypass. Omitted reviewers mean the whole pool; outside Cyber Pro name at least one ticked reviewer "
        "and give a reason when you name fewer than all; Cyber Pro composes freely. Unticked advisers never count."
    )
    reviewed_commit_description = "Commit already-changed files of Ouroboros's own body (system_repo) through the unified reviewed commit workflow: " + reviewed_commit_contract
    legacy_commit_description = "Older name of commit_reviewed: commit changes to Ouroboros's own body through the same reviewed workflow. It runs " + reviewed_commit_contract
    commit_properties = {
        "wake_on": wake_on_schema(),
        **widening_properties(),
        "root": {"type": "string", "enum": ["system_repo"], "default": "system_repo", "description": "The reviewed commit lands in Ouroboros's own body only; review any other root with review_change and commit it with ordinary git."},
        "commit_message": {"type": "string"},
        "paths": {"type": "array", "items": {"type": "string"}, "description": "Optional subset of task-attributed paths. Omitted computes candidates (with author_disposition: the reviewed paths); empty never stages the whole tree."},
        "skip_tests": {"type": "boolean", "default": False, "description": "Skip pre-commit tests."},
        "review_rebuttal": {"type": "string", "default": "", "description": "A NEW content-hashed counter-argument buys one paid re-review within capacity."},
        "preflight_reviewer": {"type": "string", "default": "", "description": "One ENABLED catalog row (id or handle; a review-pool member or not) for an early informational look at the worktree before the panel; unknown or disabled, or named beside skip_advisory_review=True or an author continuation, is TOOL_ARG_ERROR before anything is staged."},
        "reviewers": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Enabled catalog ids/handles. Omitted means the whole pool; outside Cyber Pro name at least one ticked reviewer. Cyber Pro composes freely. Unticked advisers are heard, not counted; unknown or disabled is TOOL_ARG_ERROR."},
        "reason": {"type": "string", "default": "", "description": "Why these reviewers; required outside Cyber Pro when you name fewer than all. Missing reason refuses at $0 before staging."},
        "skip_advisory_review": {"type": "boolean", "default": False, "description": "Record that this commit deliberately goes without a preflight (preflight: skipped). The deterministic checks, the tests (see skip_tests) and the panel still run; the opposite of preflight_reviewer."},
        "goal": {"type": "string", "default": "", "description": "High-level goal of this change. Used by the panel's coupling questions to judge completeness."}, "scope": {"type": "string", "default": "", "description": "Declared scope boundary. Without a goal it is the intended transformation the panel's coupling questions judge the change against (completeness, forgotten adjacent surfaces)."},
        "review_reference": {"type": "object", "description": "Exact reference a prior commit call of this task, or of the predecessor its recorded Continue continues, returned. With author_disposition: the free explicit author continuation. Without it: collect that paid gate under its ordinary checks before Git when the index holds exactly its reviewed bytes, else read its outcome back without staging; it sends nothing but a continuation recorded for it and not yet sent; continue_seats records and sends a new one."},
        "continue_seats": {**CONTINUE_SEATS_SCHEMA, "description": "With review_reference and no author_disposition: " + CONTINUE_SEATS_SCHEMA["description"]},
        "author_disposition": AUTHOR_DISPOSITION_SCHEMA,
        "since_review": {"type": "string", "default": "", "description": "A review_record_id of an earlier change review of this repository (another task, or a copy such as a reviewed candidate) to compare with: the panel is shown what changed since its subject, its verdict only as a label of those earlier bytes; no PASS, finding or obligation carries over. Unknown, foreign or unreadable is a stated comparison gap. Omitted: this task's earlier reviews of this checkout."},
    }
    return [
        ToolEntry("commit_reviewed", {"name": "commit_reviewed", "description": reviewed_commit_description,
            "parameters": {"type": "object", "properties": commit_properties, "required": []}}, _commit_reviewed, is_code_tool=True),
        ToolEntry("vcs_commit_reviewed", {"name": "vcs_commit_reviewed", "description": legacy_commit_description,
            "parameters": {"type": "object", "properties": commit_properties, "required": []}}, _commit_reviewed, is_code_tool=True, alias_for="commit_reviewed"),
        ToolEntry("vcs_pull_ff", {
            "name": "vcs_pull_ff",
            "description": (
                "Fetch origin and fast-forward the selected repository's current named branch. Changes the checkout; "
                'divergence is refused with both SHAs, without merging or rewriting history.'
            ),
            "parameters": {"type": "object", "properties": {
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace", "description": "Omit for the active project workspace; use system_repo for Ouroboros source."},
            }, "required": []},
        }, _pull_from_remote, is_code_tool=True, mutates_worktree=True),
        ToolEntry("vcs_restore", {
            "name": "vcs_restore",
            "description": (
                'Restore all uncommitted changes, or paths, to HEAD in index and files. Deletes new files in scope, '
                "staged or not; ignored files stay. confirm=false previews; true executes. Dirty protected paths in "
                "Ouroboros's own repository are refused."
            ),
            "parameters": {"type": "object", "properties": {
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace", "description": "Omit for the active project workspace; use system_repo for Ouroboros source."},
                "confirm": {"type": "boolean", "description": "false previews; true executes."},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Specific files to restore"},
            }, "required": ["confirm"]},
        }, _restore_to_head, is_code_tool=True, mutates_worktree=True),
        ToolEntry("vcs_revert", {
            "name": "vcs_revert",
            "description": (
                'Undo one commit by creating a revert commit. confirm=false previews; true requires a clean worktree. '
                "Refuses merge commits. In Ouroboros's own repository, protected-path reverts follow the caller's "
                "effective runtime mode (task limits included) or existing managed-update authority. "
                "Mechanical revert requires no new review."
            ),
            "parameters": {"type": "object", "properties": {
                "root": {"type": "string", "enum": ["active_workspace", "system_repo"], "default": "active_workspace", "description": "Omit for the active project workspace; use system_repo for Ouroboros source."},
                "sha": {"type": "string", "description": "Commit SHA to revert"},
                "confirm": {"type": "boolean", "description": "Must be true to execute."},
            }, "required": ["sha", "confirm"]},
        }, _revert_commit, is_code_tool=True, mutates_worktree=True),
    ]
# --- v7 git split re-exports (module-size discipline) -----------------------
# Every moved name keeps a re-export under its historical spelling, so
# existing importers and monkeypatch targets keep working unchanged; the
# facade binding IS the leaf's object. Cycle-free: the leaves reach parent-
# owned helpers only at call time through their _git() handle.
from ouroboros.tools.git_plumbing import (  # noqa: E402,F401
    _BINARY_EXTENSIONS,
    _acquire_git_lock,
    _binding_repo_rel,
    _binding_targets_system_repo,
    _current_runtime_mode,
    _ensure_gitignore,
    _protected_paths_block_message,
    _publish_git_error,
    _release_git_lock,
    _sanitize_git_error,
    _staged_index,
    _unstage_binaries,
    _unstage_candidate,
)


from ouroboros.tools.git_review_cycle import (  # noqa: E402,F401
    _finalize_blocked_review,
    _finalize_pending_review,
    _fingerprint_staged_diff,
    _handle_revalidation_failure,
    _review_binding_precondition_error,
    _review_custody_pending,
    _review_cycle_infra_failure,
    _run_non_committing_review_cycle,
    _reset_commit_review_state,
    _run_reviewed_stage_cycle,
    _stage_candidate_for_review,
    _verify_reviewed_commit_binding,
)


from ouroboros.tools.git_managed_postcommit import (  # noqa: E402,F401
    _finish_managed_commit,
    _managed_commit_paused,
    _managed_post_commit_tests_gate,
    _resume_managed_commit,
)


from ouroboros.tools.git_evolution import (  # noqa: E402,F401
    _check_evolution_commit_stage,
    _evolution_commit_authority,
    _evolution_publication_stopped_result,
    _evolution_test_publication_stop,
    _preserve_evolution_orphan,
    _record_evolution_commit_intent,
    _record_evolution_commit_receipt,
)


from ouroboros.tools.git_repo_edit import (  # noqa: E402,F401
    _CONTENT_OMITTED_PREFIX,
    _check_shrink_guard,
    _repo_write,
    _str_replace_editor,
)


from ouroboros.tools.git_vcs_ops import (  # noqa: E402,F401
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
