"""Advisory pre-review gate.

Normally runs a cheap read-only advisory review through the configured route
before multi-model commit review. The LLM may instead choose the audited
advisory-only skip; tests, triad/scope review, exact-snapshot revalidation, and
final commit binding remain authoritative. Any edit after advisory makes it
stale.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re  # noqa: F401 -- historical import surface kept for monkeypatching tests
import subprocess  # noqa: F401 -- historical import surface kept for monkeypatching tests
from typing import List, Optional

from ouroboros.triad_review import (
    REVIEW_JSON_ARRAY_CONTRACT,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    REVIEW_JSON_MATRIX_CONTRACT,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    empty_array_is_verified_clean,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    extract_json_array,  # noqa: F401 -- historical import surface kept for monkeypatching tests
)
from ouroboros.skill_review_status import SEVERITY_DRIVEN_ITEMS  # noqa: F401 -- historical import surface kept for monkeypatching tests
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.review_state import (
    AdvisoryRunRecord,
    AdvisoryReviewState,
    compute_snapshot_hash,
    load_state,
    make_repo_key,
    update_state,
    _utc_now,
)
from ouroboros.tools.review_helpers import (
    build_advisory_changed_context,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    build_skill_host_context,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    build_blocking_findings_json_section,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    load_checklist_section,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    build_goal_section,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    build_scope_section,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    check_worktree_readiness,
    check_worktree_version_sync as _check_worktree_version_sync_shared,
    parse_changed_paths_from_porcelain,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    CRITICAL_FINDING_CALIBRATION,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    REVIEW_SEVERITY_THRESHOLDS,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    REVIEW_THOROUGHNESS_BLOCK,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    get_advisory_runtime_diagnostics as _get_runtime_diagnostics,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    format_advisory_sdk_error as _format_advisory_error,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    load_governance_doc,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    normalize_reviewer_obligation_id,
    strip_obligation_suffix,
    _ANTI_THRASHING_RULE_VERDICT,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    _ANTI_THRASHING_RULE_ITEM_NAME,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    _HISTORY_VERIFICATION_ONLY_RULE,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    _run_review_preflight_tests,
    emit_review_event,  # noqa: F401 -- historical import surface kept for monkeypatching tests
    emit_review_usage,  # noqa: F401 -- historical import surface kept for monkeypatching tests
)
from ouroboros.utils import (
    append_jsonl,
    utc_now_iso,
    truncate_review_artifact as _truncate_review_artifact,
)
from ouroboros.review_evidence import build_review_projection, build_review_status_payload
from ouroboros.tools.review_advisory_prompt import (  # noqa: F401 -- intentional public re-exports
    _MAX_DIFF_CHARS_ERROR,
    _auto_sync_release_metadata_if_needed,
    _build_advisory_prompt,
    _build_blocking_history_section,
    _changed_paths,
    _get_changed_file_list,
    _get_staged_diff,
    _release_metadata_preflight,
    _syntax_preflight_staged_py_files,
)
from ouroboros.tools.review_advisory_run import (  # noqa: F401 -- intentional public re-exports
    ADVISORY_REVIEW_ROUTE_ENV,
    _ADVISORY_EXTRACT_CONTRACT,
    _ADVISORY_PROMPT_MAX_CHARS,
    _ADVISORY_SESSION_MAX_SECONDS,
    _advisory_sdk_budget,
    _advisory_session_deltas,
    _check_expected_items,
    _is_checklist_array,
    _is_clean_verdict,
    _llm_extract_advisory_items,
    _needs_fallback_extraction,
    _note_meta_error,
    _parse_advisory_output,
    _resolve_fallback_model,
    _run_advisory_delegated,
    _run_claude_advisory,
    advisory_gate_unavailability_reason,
    advisory_gate_unavailable,
    advisory_review_route,
    advisory_route_requires_api_key,
    advisory_slot_enabled,
)

log = logging.getLogger(__name__)


ADVISORY_REVIEW_CHOICE_GUIDANCE = (
    "Normally the LLM runs the cheap advisory_review immediately before "
    "commit_reviewed. When advisory review is slow, unhealthy, unavailable, or "
    "low-value, the LLM may deliberately choose skip_advisory_review=True; the "
    "choice is durably audited. This skip bypasses only the requirements for "
    "advisory freshness, advisory obligations, and advisory debt; unresolved "
    "obligation and debt records remain visible, while tests, triad review, "
    "applicable scope review, snapshot/fingerprint revalidation, and final "
    "commit/tag/SHA binding still apply."
)


def _json_response(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


# -- Audit logging --

def _audit_bypass(ctx: ToolContext, snapshot_hash: str, commit_message: str,
                  bypass_reason: str, task_id: str) -> None:
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "advisory_review_bypassed",
            "snapshot_hash": snapshot_hash,
            "commit_message": commit_message,  # full — no [:200] truncation
            "bypass_reason": bypass_reason,
            "task_id": task_id,
        })
    except Exception:
        pass


def _identical_diff_cap_note() -> str:
    """Schema-build-time NOTE about the identical-diff attempt cap, derived from the
    shared OUROBOROS_REVIEW_MAX_CYCLES (never a hardcoded number)."""
    from ouroboros.review_cycles import review_max_cycles

    cap = review_max_cycles()
    if cap is None:
        return (
            "NOTE: no identical-diff cap is configured (OUROBOROS_REVIEW_MAX_CYCLES=unlimited): "
            "commit_reviewed never refuses a resubmission on cap grounds."
        )
    return (
        f"NOTE: after {cap} genuine review-verdict block(s) of a byte-identical staged diff "
        "(the shared OUROBOROS_REVIEW_MAX_CYCLES cap), commit_reviewed refuses further "
        "attempts (attempt_cap_reached) until the diff changes or a review_rebuttal is provided."
    )


def _advisory_run_record(
    snapshot_hash: str,
    commit_message: str,
    status: str,
    *,
    repo_key: str,
    task_id: str,
    **fields,
) -> AdvisoryRunRecord:
    return AdvisoryRunRecord(
        snapshot_hash=snapshot_hash,
        commit_message=commit_message,
        status=status,
        ts=_utc_now(),
        repo_key=repo_key,
        tool_name="advisory_review",
        task_id=task_id,
        items=list(fields.get("items") or []),
        snapshot_summary=str(fields.get("snapshot_summary") or ""),
        raw_result=str(fields.get("raw_result") or ""),
        bypass_reason=str(fields.get("bypass_reason") or ""),
        bypassed_by_task=str(fields.get("bypassed_by_task") or ""),
        snapshot_paths=fields.get("snapshot_paths"),
        readiness_warnings=list(fields.get("readiness_warnings") or []),
        prompt_chars=int(fields.get("prompt_chars") or 0),
        model_used=str(fields.get("model_used") or ""),
        session_id=str(fields.get("session_id") or ""),
        duration_sec=float(fields.get("duration_sec") or 0.0),
    )


def _record_bypass(ctx: ToolContext, state: "AdvisoryReviewState", snapshot_hash: str,
                   commit_message: str, reason: str, task_id: str,
                   drive_root: pathlib.Path,
                   snapshot_paths: Optional[List[str]] = None) -> str:
    """Audit, record, and save a bypassed advisory run. Returns JSON response."""
    _audit_bypass(ctx, snapshot_hash, commit_message, reason, task_id)
    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))

    def _mutate(bypass_state: "AdvisoryReviewState") -> None:
        bypass_state.add_run(_advisory_run_record(
            snapshot_hash, commit_message, "bypassed",
            repo_key=repo_key, task_id=task_id,
            bypass_reason=reason, bypassed_by_task=task_id,
            snapshot_paths=snapshot_paths,
        ))

    update_state(drive_root, _mutate)
    # Persistent visibility (same mechanism as advisory-enforcement overrides):
    # review_status surfaces how often the advisory layer was bypassed/absent.
    try:
        from ouroboros.utils import update_json_locked, utc_now_iso as _now_iso

        def _bump(current: dict) -> dict:
            recent = list(current.get("recent") or [])
            recent.append({"ts": _now_iso(), "block_reason": f"advisory_bypass: {reason}"[:200], "message_head": str(commit_message or "")[:200]})
            return {"count": int(current.get("count") or 0) + 1, "recent": recent[-10:]}

        update_json_locked(pathlib.Path(drive_root) / "state" / "advisory_overrides.json", _bump)
    except Exception:
        log.debug("Failed to persist advisory bypass visibility", exc_info=True)
    if "ANTHROPIC_API_KEY" in reason:
        # Route-dependent honesty (plan 5.8 site 4): the key is only the API
        # route's requirement — the owner also has the keyless delegated route.
        msg = (
            "⚠️ ANTHROPIC_API_KEY is not set — advisory review skipped automatically "
            "because the configured advisory route (api) requires it. "
            "Bypass has been durably audited in events.jsonl. "
            "Set ANTHROPIC_API_KEY in Settings, or switch the advisory to the "
            "delegated subscription route (OUROBOROS_ADVISORY_REVIEW_ROUTE="
            "agent_session), which needs no API key."
        )
    else:
        msg = "Advisory review bypassed. Bypass has been durably audited."
    return _json_response({
        "status": "bypassed",
        "snapshot_hash": snapshot_hash,
        "bypass_reason": reason,
        "message": msg,
    })


def _resolve_matching_obligations(
    state: "AdvisoryReviewState",
    items: list,
    snapshot_hash: str,
    *,
    repo_key: str | None = None,
) -> None:
    """Resolve obligations only on unambiguous PASS without same-item FAIL."""
    if not items:
        return
    # Build per-item verdict sets to detect contradictions.
    item_verdicts: dict[str, set[str]] = {}
    obligation_verdicts: dict[str, set[str]] = {}
    for i in items:
        if not isinstance(i, dict):
            continue
        verdict = str(i.get("verdict", "")).upper().strip()
        item_name = str(i.get("item", "")).strip()
        if not item_name or not verdict:
            continue
        explicit_obligation_id = normalize_reviewer_obligation_id(i.get("obligation_id", ""))
        normalized_item_name, suffix_obligation_id = strip_obligation_suffix(item_name)
        normalized_item_name = normalized_item_name.strip().lower()
        if normalized_item_name:
            item_verdicts.setdefault(normalized_item_name, set()).add(verdict)
        # Explicit id and suffix id must agree; mismatches are ambiguous and
        # must not clear unrelated obligations/debt.
        if explicit_obligation_id and suffix_obligation_id:
            if explicit_obligation_id.lower() == suffix_obligation_id.lower():
                obligation_verdicts.setdefault(explicit_obligation_id, set()).add(verdict)
            # Mismatch: skip both ids for this entry.
            continue
        if explicit_obligation_id:
            obligation_verdicts.setdefault(explicit_obligation_id, set()).add(verdict)
        elif suffix_obligation_id:
            obligation_verdicts.setdefault(suffix_obligation_id, set()).add(verdict)

    # Only PASS items with no FAIL entry for the same item.
    unambiguous_pass = {
        item_name
        for item_name, verdicts in item_verdicts.items()
        if "PASS" in verdicts and "FAIL" not in verdicts
    }
    unambiguous_pass_ids = {
        obligation_id
        for obligation_id, verdicts in obligation_verdicts.items()
        if "PASS" in verdicts and "FAIL" not in verdicts
    }

    open_obs = state.get_open_obligations(repo_key=repo_key)

    # Item-name fallback is safe only with exactly one open obligation per item.
    from collections import Counter as _Counter
    item_open_count = _Counter(o.item.lower() for o in open_obs)

    resolved = [
        o.obligation_id for o in open_obs
        if o.obligation_id.lower() in unambiguous_pass_ids
        or (
            o.item.lower() in unambiguous_pass
            and item_open_count[o.item.lower()] == 1
        )
    ]
    if resolved:
        state.resolve_obligations(
            resolved,
            resolved_by=f"advisory run {snapshot_hash[:12]}",
            repo_key=repo_key,
        )
        state._sync_commit_readiness_debts(repo_key=repo_key)


def _next_step_guidance(latest: Optional["AdvisoryRunRecord"], state: "AdvisoryReviewState",
                        stale_from_edit: bool, stale_from_edit_ts: Optional[str],
                        open_obs: list, open_debts: list, effective_is_fresh: bool = False) -> str:
    """Return a concrete next-step string based on current advisory state.

    Snapshot binding of record-derived claims (the v6.74.5 "SyntaxError" stale
    template that cost a release ~25 min) is enforced UPSTREAM by the
    projection: a blocked record whose hash differs from the current tree sets
    ``stale_from_edit`` (review_evidence hash_mismatch), which routes to the
    generic "invalidated" message below instead of asserting the problem class
    — that assertion only ever fires for a record of the CURRENT snapshot. The
    one unbindable case stays as before: an uncomputable current hash cannot
    establish a mismatch either way.
    """
    def _debt_hint() -> str:
        parts = []
        if open_obs:
            parts.append(f"{len(open_obs)} open obligation(s) from previous blocking rounds")
        if open_debts:
            parts.append(f"{len(open_debts)} commit-readiness debt item(s) surfaced by review_status")
        return (" ".join(parts) + ". ") if parts else ""

    regroup = "After the first blocked review, stop patching one finding at a time: re-read the full diff, group obligations by root cause, rewrite the plan, finish all remaining edits, then run advisory_review(commit_message='...')."

    def _with_choices(message: str) -> str:
        return f"{message.rstrip()} {ADVISORY_REVIEW_CHOICE_GUIDANCE}"

    if not effective_is_fresh:
        status = str(getattr(latest, "status", "") or "")
        if latest and status in {"tests_preflight_blocked", "preflight_blocked"} and not stale_from_edit:
            if status == "tests_preflight_blocked":
                problem = "test preflight: pytest failed before the Claude SDK call"
                fix = "Fix the failing tests and re-run advisory_review. Use advisory_review(skip_tests=True) only for intentional WIP code."
            else:
                problem = "syntax preflight: a staged .py file has a SyntaxError"
                fix = "See raw_result for file:line:msg, fix it, and re-run advisory_review."
            return _with_choices(
                f"Last advisory run was blocked by {problem}. {fix} {_debt_hint()}".strip()
            )
        if latest and status == "parse_failure" and not stale_from_edit:
            suffix = (
                regroup + " Or bypass: commit_reviewed(skip_advisory_review=True) (audited)."
                if (open_obs or open_debts)
                else "Re-run: advisory_review(commit_message='...'), or bypass: commit_reviewed(skip_advisory_review=True) (audited)."
            )
            return _with_choices(
                f"Last advisory run produced unparseable output (parse_failure). {_debt_hint()}{suffix}"
            )
        if open_obs or open_debts:
            prefix = f"Advisory was invalidated by a worktree edit at {stale_from_edit_ts}. " if stale_from_edit else "Advisory is stale or missing for the current snapshot. "
            return _with_choices(prefix + _debt_hint() + regroup)
        if stale_from_edit:
            return _with_choices(
                f"Advisory was invalidated by a worktree edit at {stale_from_edit_ts}. Complete ALL remaining edits, then run: advisory_review(commit_message='...')"
            )
        if not state.advisory_runs:
            return _with_choices("No advisory run yet. Run: advisory_review(commit_message='...')")
        return _with_choices("Advisory is stale (snapshot changed). Run: advisory_review(commit_message='...')")

    # Advisory is effectively fresh — check obligations and findings
    if open_obs or open_debts:
        return _with_choices(
            f"Advisory is current but unresolved review debt remains. {_debt_hint()}commit_reviewed will be blocked until that debt is cleared. Re-read the full diff, group obligations by root cause, and rewrite the plan. Fix the issues, re-run advisory_review so it marks them PASS, or bypass: commit_reviewed(skip_advisory_review=True) (audited)."
        )

    if latest and latest.status == "skipped":
        return "Advisory was skipped — prompt exceeded the budget gate (prompt too large for advisory). commit_reviewed may proceed. Consider splitting the commit into smaller chunks so advisory can run on the next change."

    if latest and latest.status == "bypassed":
        return "Advisory was bypassed (audited). No open obligations — commit_reviewed should proceed. Consider running advisory_review for a proper review."

    fresh_critical = [
        i for i in (latest.items if latest else []) or []
        if isinstance(i, dict) and str(i.get("verdict", "")).upper() == "FAIL"
        and str(i.get("severity", "")).lower() == "critical"
    ]
    if fresh_critical:
        return _with_choices(
            f"Advisory found {len(fresh_critical)} critical issue(s). Fix ALL critical findings, then re-run advisory_review, or deliberately choose the audited advisory skip."
        )
    return "Advisory is fresh with no critical findings. Proceed with: commit_reviewed(commit_message='...'). ⚠️ Do NOT make any further edits — any edit will make advisory stale."


def _persist_preflight_record(
    ctx: ToolContext,
    snapshot_hash: str,
    commit_message: str,
    record: dict,
) -> None:
    """Persist a durable preflight-blocked advisory record; never raises."""
    try:
        record = dict(record or {})
        drive_root = pathlib.Path(ctx.drive_root)
        repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
        task_id = str(getattr(ctx, "task_id", "") or "")

        def _mutate(pre_state: AdvisoryReviewState) -> None:
            pre_state.add_run(_advisory_run_record(
                snapshot_hash, commit_message, str(record.get("status") or "error"),
                repo_key=repo_key, task_id=task_id,
                snapshot_summary=("advisory SDK error" if record.get("session_id") else "preflight block — SDK not called"),
                raw_result=record.get("raw_result"),
                snapshot_paths=record.get("paths"),
                readiness_warnings=record.get("readiness_warnings"),
                prompt_chars=record.get("prompt_chars"),
                model_used=record.get("model_used"),
                session_id=record.get("session_id"),
                duration_sec=record.get("duration_sec"),
            ))
        update_state(drive_root, _mutate)
    except Exception:
        log.debug("_persist_preflight_record failed (non-critical)", exc_info=True)


def _advisory_pre_sdk_gate(
    ctx: ToolContext,
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    snapshot_hash: str,
    commit_message: str,
    paths: Optional[List[str]],
    skip_tests: bool,
):
    """Run cheap pre-SDK gates and return warnings/status/early JSON exit."""
    repo_key = make_repo_key(repo_dir)
    task_id = str(getattr(ctx, "task_id", "") or "")
    state = load_state(drive_root)

    # Readiness gate first: reject clean worktree before fresh-run shortcut.
    readiness_warnings = check_worktree_readiness(repo_dir, paths=paths)
    if readiness_warnings and any("no uncommitted changes" in w.lower() for w in readiness_warnings):
        ctx.emit_progress_fn(f"⚠️ Advisory readiness gate: {'; '.join(readiness_warnings)}")
        return readiness_warnings, "", _json_response({
            "status": "error",
            "snapshot_hash": snapshot_hash,
            "message": "No uncommitted changes detected — nothing to review.",
            "readiness_warnings": readiness_warnings,
        })

    if readiness_warnings:
        try:
            append_jsonl(drive_root / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "advisory_readiness_gate",
                "warnings": readiness_warnings,
                "task_id": task_id,
            })
        except Exception:
            pass

    # Fresh-run shortcut only when no obligations/debt remain.
    existing = state.find_by_hash(snapshot_hash, repo_key=repo_key)
    open_obligations = state.get_open_obligations(repo_key=repo_key)
    open_debts = state.get_open_commit_readiness_debts(repo_key=repo_key)
    already_fresh_ok = (
        existing and existing.status in ("fresh", "bypassed", "skipped")
        and not open_obligations and not open_debts
    )
    if already_fresh_ok:
        return readiness_warnings, "", _json_response({
            "status": "already_fresh",
            "snapshot_hash": snapshot_hash,
            "ts": existing.ts,
            "items": existing.items,
            "readiness_warnings": readiness_warnings,
            "message": "A fresh advisory run already exists for this snapshot. Proceed with commit_reviewed.",
        })

    ctx.emit_progress_fn("Running advisory pre-review (Claude Code, read-only)...")
    changed_files = _get_changed_file_list(repo_dir, paths=paths)

    if changed_files.startswith("⚠️ ADVISORY_ERROR"):
        return readiness_warnings, changed_files, _json_response({
            "status": "error",
            "snapshot_hash": snapshot_hash,
            "error": changed_files,
            "message": (
                "Advisory review aborted: could not retrieve changed file list. "
                "Fix the error and retry, or use skip_advisory_review=True to bypass (will be audited)."
            ),
        })

    release_preflight_err = _release_metadata_preflight(repo_dir, commit_message, paths)
    if release_preflight_err:
        ctx.emit_progress_fn(release_preflight_err)
        _persist_preflight_record(
            ctx=ctx,
            snapshot_hash=snapshot_hash,
            commit_message=commit_message,
            record={
                "status": "preflight_blocked",
                "raw_result": release_preflight_err,
                "paths": paths,
                "duration_sec": 0.0,
                "readiness_warnings": readiness_warnings,
            },
        )
        return readiness_warnings, changed_files, _json_response({
            "status": "preflight_blocked",
            "snapshot_hash": snapshot_hash,
            "error": release_preflight_err,
            "readiness_warnings": readiness_warnings,
            "message": (
                "Advisory SDK was skipped: deterministic release metadata preflight "
                "failed before provider budget was spent."
            ),
        })

    # Version-sync check is a non-fatal warning.
    version_sync_warning = _check_worktree_version_sync_shared(repo_dir)
    if version_sync_warning:
        ctx.emit_progress_fn(f"⚠️ Advisory preflight: {version_sync_warning}")

    # Test preflight before the expensive SDK call.
    if not skip_tests:
        ctx.emit_progress_fn("Running tests before advisory SDK call...")
        test_err = _run_advisory_tests(ctx)
        if test_err:
            msg = (
                "⚠️ TESTS_PREFLIGHT_BLOCKED: Tests must pass before advisory review.\n"
                "Fix the failures below, then re-run advisory_review.\n"
                "Use skip_tests=True if this is intentionally incomplete WIP code.\n\n"
                f"{test_err}"
            )
            ctx.emit_progress_fn(msg)
            # Persist non-fresh blocker so review_status can surface it after restart.
            _persist_preflight_record(
                ctx=ctx,
                snapshot_hash=snapshot_hash,
                commit_message=commit_message,
                record={
                    "status": "tests_preflight_blocked",
                    "raw_result": msg,
                    "paths": paths,
                    "duration_sec": 0.0,
                    "readiness_warnings": readiness_warnings,
                },
            )
            return readiness_warnings, changed_files, _json_response({
                "status": "tests_preflight_blocked",
                "snapshot_hash": snapshot_hash,
                "message": msg,
                "readiness_warnings": readiness_warnings,
            })
        ctx.emit_progress_fn("Tests passed ✓ — proceeding with advisory SDK call.")

    return readiness_warnings, changed_files, None


def _run_advisory_tests(ctx: ToolContext) -> Optional[str]:
    """Run shared pytest preflight while preserving this monkeypatch seam."""
    return _run_review_preflight_tests(ctx)


def _handle_advisory_pre_review(
    ctx: ToolContext,
    commit_message: str = "",
    skip_advisory_review: bool = False,
    skip_advisory_pre_review: bool = False,
    goal: str = "",
    scope: str = "",
    paths: Optional[List[str]] = None,
    skip_tests: bool = False,
) -> str:
    """Run an advisory pre-commit review through the configured read-only route."""
    skip_advisory_pre_review = bool(skip_advisory_review or skip_advisory_pre_review)
    repo_dir = pathlib.Path(ctx.repo_dir)
    drive_root = pathlib.Path(ctx.drive_root)

    # KNOWN ORDERING DEBT (v6.82 backlog, deliberately NOT restructured here): this self-repair
    # runs ~87 lines AFTER `_release_metadata_preflight`, the gate it exists to satisfy, so with
    # respect to that gate it is dead code — a desynced version carrier still blocks. Left in
    # place because reordering runtime review machinery is out of scope for a provenance commit.
    auto_synced_paths = _auto_sync_release_metadata_if_needed(ctx, repo_dir, drive_root, paths)
    if paths is not None and auto_synced_paths:
        paths = sorted({str(p) for p in list(paths) + auto_synced_paths if str(p).strip()})

    snapshot_hash = compute_snapshot_hash(repo_dir, commit_message, paths=paths)

    # Bypass recording state; the pre-SDK gate derives its own under 8 params.
    repo_key = make_repo_key(repo_dir)
    task_id = str(getattr(ctx, "task_id", "") or "")
    state = load_state(drive_root)

    # Auto-bypass a missing Anthropic key ONLY when the configured advisory
    # route actually needs it (plan 5.8 site 3 — the dangerous one): on the
    # delegated route the constitutional gate RUNS instead of recording a
    # routine-looking "auto-bypassed" over a commit the free route could have
    # reviewed. A misconfigured route token is a loud error, not a bypass.
    try:
        _requires_key = advisory_route_requires_api_key()
        _advisory_enabled = advisory_slot_enabled()
    except ValueError as exc:
        return _json_response({
            "status": "error",
            "snapshot_hash": snapshot_hash,
            "error": f"⚠️ ADVISORY_ERROR: {exc}",
            "message": "Fix the advisory reviewer configuration "
                       "(OUROBOROS_REVIEWER_SLOTS / OUROBOROS_ADVISORY_REVIEW_ROUTE) and retry.",
        })
    if not _advisory_enabled:
        # The owner switched the advisory slot off (6.2). The constitutional
        # gate still runs — as an AUDITED BYPASS on this exact snapshot, the
        # same durable record an explicit per-call skip produces.
        return _record_bypass(ctx, state, snapshot_hash, commit_message,
                               "advisory reviewer disabled in settings — audited bypass",
                               task_id, drive_root,
                               snapshot_paths=paths)
    if _requires_key and not os.environ.get("ANTHROPIC_API_KEY", ""):
        return _record_bypass(ctx, state, snapshot_hash, commit_message,
                               "ANTHROPIC_API_KEY not set — auto-bypassed (advisory route=api)",
                               task_id, drive_root,
                               snapshot_paths=paths)

    # Explicit audited bypass.
    if skip_advisory_pre_review:
        return _record_bypass(ctx, state, snapshot_hash, commit_message,
                               "explicit skip_advisory_review=True", task_id, drive_root,
                               snapshot_paths=paths)

    readiness_warnings, changed_files, early_exit = _advisory_pre_sdk_gate(
        ctx=ctx,
        repo_dir=repo_dir,
        drive_root=drive_root,
        snapshot_hash=snapshot_hash,
        commit_message=commit_message,
        paths=paths,
        skip_tests=skip_tests,
    )
    if early_exit is not None:
        return early_exit

    import time as _time
    _advisory_start = _time.monotonic()
    items, raw_result, model_used, prompt_chars = _run_claude_advisory(
        repo_dir,
        commit_message,
        ctx,
        goal=goal,
        scope=scope,
        paths=paths,
        options={"drive_root": drive_root},
    )
    _advisory_duration = _time.monotonic() - _advisory_start
    advisory_meta = dict(getattr(ctx, "_last_claude_advisory_meta", {}) or {})
    advisory_session_id = str(advisory_meta.get("session_id") or "")

    # SDK/CLI errors.
    if raw_result.startswith("⚠️ ADVISORY_ERROR"):
        _persist_preflight_record(
            ctx=ctx,
            snapshot_hash=snapshot_hash,
            commit_message=commit_message,
            record={
                "status": "error",
                "raw_result": raw_result,
                "paths": paths,
                "duration_sec": _advisory_duration,
                "readiness_warnings": readiness_warnings,
                "prompt_chars": prompt_chars,
                "model_used": model_used,
                "session_id": advisory_session_id,
            },
        )
        return _json_response({
            "status": "error",
            "snapshot_hash": snapshot_hash,
            "error": raw_result,
            "session_id": advisory_session_id,
            "readiness_warnings": readiness_warnings,
            "message": (
                "Advisory review failed to run. Fix the error and retry, "
                "or use skip_advisory_review=True to bypass (will be audited)."
            ),
        })

    # Syntax preflight skipped SDK; persist explicit blocker, not parse_failure.
    if raw_result.startswith("⚠️ PREFLIGHT_BLOCKED"):
        _persist_preflight_record(
            ctx=ctx,
            snapshot_hash=snapshot_hash,
            commit_message=commit_message,
            record={
                "status": "preflight_blocked",
                "raw_result": raw_result,
                "paths": paths,
                "duration_sec": _advisory_duration,
                "readiness_warnings": readiness_warnings,
            },
        )
        return _json_response({
            "status": "preflight_blocked",
            "snapshot_hash": snapshot_hash,
            "error": raw_result,
            "readiness_warnings": readiness_warnings,
            "message": (
                "Advisory SDK was skipped: a staged .py file has a SyntaxError. "
                "Fix the syntax error listed above and re-run advisory_review."
            ),
        })

    # Prompt too large: persist non-blocking skipped run as fresh for this snapshot.
    if raw_result.startswith("⚠️ ADVISORY_SKIPPED:"):
        snapshot_summary = f"{changed_files.count(chr(10)) + 1} file(s) changed"
        def _mutate_skip(skip_state: AdvisoryReviewState) -> None:
            skip_state.add_run(_advisory_run_record(
                snapshot_hash, commit_message, "skipped",
                repo_key=repo_key, task_id=task_id,
                snapshot_summary=snapshot_summary, raw_result=raw_result,
                snapshot_paths=paths, readiness_warnings=readiness_warnings,
                prompt_chars=prompt_chars, model_used=model_used,
                session_id=advisory_session_id, duration_sec=_advisory_duration,
            ))

        update_state(drive_root, _mutate_skip)
        return _json_response({
            "status": "skipped",
            "snapshot_hash": snapshot_hash,
            "message": raw_result,
            "session_id": advisory_session_id,
            "readiness_warnings": readiness_warnings,
        })

    # Classify findings.
    critical_fails = [i for i in items if isinstance(i, dict)
                      and str(i.get("verdict", "")).upper() == "FAIL"
                      and str(i.get("severity", "")).lower() == "critical"]
    advisory_fails = [i for i in items if isinstance(i, dict)
                      and str(i.get("verdict", "")).upper() == "FAIL"
                      and str(i.get("severity", "")).lower() != "critical"]

    snapshot_summary = f"{changed_files.count(chr(10)) + 1} file(s) changed"

    # An empty array counts as a real "no findings" verdict only when the model
    # emitted the NO_FINDINGS sentinel the prompt asks for (REVIEW_JSON_ARRAY_CONTRACT),
    # or a bare `[]`-only body. A `[]` buried in refusal prose stays parse_failure.
    # Same predicate as triad, so one contract cannot mean two things.
    verified_clean = not items and _is_clean_verdict(raw_result)
    run_status = "fresh" if (items or verified_clean) else "parse_failure"
    run = _advisory_run_record(
        snapshot_hash, commit_message, run_status,
        repo_key=repo_key, task_id=task_id,
        items=items, snapshot_summary=snapshot_summary, raw_result=raw_result,
        snapshot_paths=paths, readiness_warnings=readiness_warnings,
        prompt_chars=prompt_chars, model_used=model_used,
        session_id=advisory_session_id, duration_sec=_advisory_duration,
    )

    # Locked read-modify-write against the LIVE ledger: the SDK call above runs
    # for minutes, and a state object loaded before it would clobber stale-marks
    # and concurrent runs recorded meanwhile (the pre-SDK `state` snapshot is
    # only used for gating decisions, never persisted from here on).
    def _record_run(live_state: "AdvisoryReviewState") -> None:
        live_state.add_run(run)
        if run_status != "parse_failure" and items:
            _resolve_matching_obligations(live_state, items, snapshot_hash, repo_key=repo_key)

    update_state(drive_root, _record_run)

    # Surface parse failures explicitly.
    if run_status == "parse_failure":
        return _json_response({
            "status": "parse_failure",
            "snapshot_hash": snapshot_hash,
            "error": "Advisory ran but returned no parseable checklist items.",
            "raw_result": _truncate_review_artifact(raw_result),
            "session_id": advisory_session_id,
            "readiness_warnings": readiness_warnings,
            "message": (
                "Advisory output could not be parsed. Re-run advisory_review, "
                "or use skip_advisory_review=True to bypass (will be audited)."
            ),
        })

    # Build human-readable summary.
    findings_summary: List[str] = []
    for item in critical_fails:
        findings_summary.append(f"  CRITICAL [{item.get('item','?')}]: {item.get('reason','')}")
    for item in advisory_fails:
        findings_summary.append(f"  ADVISORY [{item.get('item','?')}]: {item.get('reason','')}")

    result = {
        "status": "fresh",
        "snapshot_hash": snapshot_hash,
        "ts": run.ts,
        "items": items,
        "critical_count": len(critical_fails),
        "advisory_count": len(advisory_fails),
        "snapshot_summary": snapshot_summary,
        "session_id": advisory_session_id,
        "readiness_warnings": readiness_warnings,
        "message": (
            "Advisory review complete. No findings. Run commit_reviewed when ready."
            if verified_clean else
            f"Advisory review complete. {len(critical_fails)} critical, "
            f"{len(advisory_fails)} advisory findings. "
            "Fix issues and run commit_reviewed when ready."
        ),
    }
    if findings_summary:
        result["findings"] = findings_summary

    return _json_response(result)


def _handle_review_status(
    ctx: ToolContext,
    repo_key: str = "",
    tool_name: str = "",
    task_id: str = "",
    attempt: Optional[int] = None,
    include_raw: bool = False,
) -> str:
    """Show advisory freshness, review debt, guidance, and optional raw evidence."""
    projection = build_review_projection(
        ctx.drive_root,
        repo_dir=getattr(ctx, "repo_dir", ""),
        repo_key=repo_key,
        tool_name=tool_name,
        task_id=task_id,
        attempt=attempt,
        snapshot_hash_fn=compute_snapshot_hash,
    )
    next_step = _next_step_guidance(
        projection["guidance_run"],
        projection["state"],
        projection["stale_from_edit"],
        projection["stale_from_edit_ts"],
        projection["open_obligations"],
        projection["open_debts"],
        effective_is_fresh=projection["effective_is_fresh"],
    )
    return json.dumps(
        build_review_status_payload(projection, next_step=next_step, include_raw=include_raw),
        ensure_ascii=False,
        indent=2,
    )


_schema_param = lambda param_type, description, **extra: {"type": param_type, "description": description, **extra}


def get_tools() -> list:
    return [
        ToolEntry(
            name="advisory_review",
            timeout_sec=1200,
            schema={
                "name": "advisory_review",
                "description": (
                    "Run an advisory pre-commit review through the configured read-only route. "
                    "Returns structured JSON findings; any edit afterward makes the result stale. "
                    f"{ADVISORY_REVIEW_CHOICE_GUIDANCE} "
                    f"{_identical_diff_cap_note()}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "commit_message": _schema_param("string", "Intended commit message. Used to bind the advisory run to this specific commit."),
                        "skip_advisory_review": _schema_param(
                            "boolean",
                            "Choose the audited advisory-only skip for this call. "
                            f"{ADVISORY_REVIEW_CHOICE_GUIDANCE} Default: False.",
                            default=False,
                        ),
                        "goal": _schema_param("string", "High-level goal of this change. Used to judge completeness."),
                        "scope": _schema_param("string", "Declared scope boundary. Issues outside scope are advisory-only."),
                        "paths": _schema_param("array", "Explicit list of changed file paths. Auto-detected from git status if omitted.", items={"type": "string"}),
                        "skip_tests": _schema_param("boolean", "Skip the pre-advisory pytest run. Default: False (tests run by default). Use True only for intentionally incomplete WIP code where test failures are expected. Tests are run before the SDK call — in a hermetic worktree, as the same two passes CI runs (parallel 'not serial' then serial) — to catch broken code early and avoid wasting advisory budget.", default=False),
                    },
                    "required": ["commit_message"],
                },
            },
            handler=_handle_advisory_pre_review,
        ),
        ToolEntry(
            name="review_status",
            schema={
                "name": "review_status",
                "description": (
                    "Show recent advisory pre-review run history. Read-only diagnostic — use to check advisory freshness before commit_reviewed. Also shows: last commit attempt state (reviewing/blocked/succeeded/failed) with block reason and actionable guidance; whether advisory is stale because of a worktree edit; open obligations from previous blocking rounds; open commit-readiness debt (durable repo-scoped anti-thrashing signal with fields `commit_readiness_debts`, `commit_readiness_debts_count`); `repo_commit_ready` (an advisory-readiness projection only: a fresh/bypassed/skipped advisory and no open advisory obligations or debt, not the full commit gate); `retry_anchor` (non-null, currently `commit_readiness_debt`, when debt is open — start the next retry from that record instead of patching one obligation at a time); and a concrete next_step recommendation. "
                    f"{ADVISORY_REVIEW_CHOICE_GUIDANCE} "
                    "Pass include_raw=true to surface the full per-actor evidence (triad_raw_results, scope_raw_result) for the targeted attempt."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_key": _schema_param("string", "Optional repo identity filter for attempt/advisory history."),
                        "tool_name": _schema_param("string", "Optional tool-name filter (for example commit_reviewed)."),
                        "task_id": _schema_param("string", "Optional task-id filter for attempt/advisory history."),
                        "attempt": _schema_param("integer", "Optional attempt number filter within the selected repo/tool/task scope."),
                        "include_raw": _schema_param("boolean", "If true, append full per-actor evidence (triad_raw_results, scope_raw_result) for the targeted commit attempt to the output. Without this flag the output contains only structured summaries. Defaults to false."),
                    },
                    "required": [],
                },
            },
            handler=_handle_review_status,
        ),
    ]
