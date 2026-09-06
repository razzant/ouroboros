"""Reviewed-commit fingerprint revalidation helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ouroboros.tools.commit_gate import _record_commit_attempt
from ouroboros.utils import append_jsonl, utc_now_iso


def _emit_review_state_event(ctx, event_type: str, **payload: Any) -> None:
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": event_type,
            "task_id": str(getattr(ctx, "task_id", "") or ""),
            **payload,
        })
    except Exception:
        pass


def _build_revalidation_failure(
    *,
    kind: str,
    before: Dict[str, Any],
    after: Optional[Dict[str, Any]] = None,
) -> str:
    if kind == "revalidation_failed":
        return (
            "⚠️ REVIEW_REVALIDATION_FAILED: the prepared review candidate changed. "
            f"before={before.get('fingerprint', '')[:12]}, "
            f"after={str((after or {}).get('fingerprint', ''))[:12]}. "
            "The reviewed findings were invalidated and were NOT carried forward. "
            "Re-run preflight_review and commit_reviewed on the final staged diff."
        )
    detail = before.get("reason") or (after or {}).get("reason") or "fingerprint unavailable"
    return (
        "⚠️ REVIEW_REVALIDATION_FAILED: could not fingerprint the staged diff "
        f"for reviewed-commit revalidation ({detail}). "
        "The attempt was recorded as degraded and reviewed findings were NOT carried forward. "
        "Fix the git diff issue, then re-run preflight_review and commit_reviewed."
    )


def handle_revalidation_failure(
    ctx,
    commit_message: str,
    commit_start: float,
    *,
    pre_fingerprint: Dict[str, Any],
    post_fingerprint: Optional[Dict[str, Any]] = None,
    kind: str,
    record_commit_attempt=_record_commit_attempt,
) -> str:
    msg = _build_revalidation_failure(kind=kind, before=pre_fingerprint, after=post_fingerprint)
    fingerprint_status = "mismatch" if kind == "revalidation_failed" else "unavailable"
    degraded_reason = (
        pre_fingerprint.get("reason")
        or (post_fingerprint or {}).get("reason")
        or msg
    )
    _emit_review_state_event(
        ctx,
        "reviewed_attempt_revalidation_failed",
        kind=kind,
        tool=str(getattr(ctx, "_current_review_tool_name", "") or ""),
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        post_review_fingerprint=(post_fingerprint or {}).get("fingerprint", ""),
        detail=degraded_reason,
    )
    ctx._review_advisory = []
    record_commit_attempt(
        ctx,
        commit_message,
        "blocked",
        block_reason="revalidation_failed" if kind == "revalidation_failed" else "fingerprint_unavailable",
        block_details=msg,
        duration_sec=time.time() - commit_start,
        critical_findings=[],
        advisory_findings=[],
        readiness_warnings=["Reviewed findings invalidated; commit must be re-reviewed."],
        phase="revalidation",
        pre_review_fingerprint=pre_fingerprint.get("fingerprint", ""),
        post_review_fingerprint=(post_fingerprint or {}).get("fingerprint", ""),
        fingerprint_status=fingerprint_status,
        degraded_reasons=[degraded_reason] + list(getattr(ctx, "_review_degraded_reasons", []) or []),
        triad_models=getattr(ctx, "_last_triad_models", []),
        scope_model=getattr(ctx, "_last_scope_model", ""),
        triad_raw_results=getattr(ctx, "_last_triad_raw_results", []),
        scope_raw_result=getattr(ctx, "_last_scope_raw_result", {}),
    )
    return msg
