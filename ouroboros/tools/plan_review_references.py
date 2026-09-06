"""Small owner-visible invalidations for canonical task review state writes."""

from __future__ import annotations

import json
import logging
import pathlib
from hashlib import sha256
from typing import Any, Optional

from ouroboros.task_results import (
    load_plan_review_state,
    mark_plan_review_cycles_exhausted,
    record_plan_review_attempt,
)
from ouroboros.tools.plan_review_runtime import record_raw_plan_request_attempt
from ouroboros.utils import append_jsonl, emit_log_event, utc_now_iso

log = logging.getLogger(__name__)


def _emit_plan_review_reference(
    ctx: Any,
    task_id: str,
    state: Optional[dict] = None,
    *,
    state_root: Optional[pathlib.Path] = None,
) -> None:
    """Best-effort persist and publish an invalidation; task result remains authority.

    The progress row is only a reconnect/history hint.  A filesystem failure on
    that presentation rail must never interrupt the canonical Plan Review write
    that just completed or the live invalidation event below.
    """
    if state is None:
        try:
            state = load_plan_review_state(state_root, task_id)
        except (OSError, TimeoutError, ValueError):
            log.debug("Failed to reload plan-review state reference", exc_info=True)
            return
    attempt = state.get("current_attempt") if isinstance(state, dict) else {}
    _emit_review_reference(ctx, task_id, state, surface="plan_review", state_root=state_root,
                           fingerprint=str((attempt or {}).get("fingerprint") or ""))


def _emit_review_reference(
    ctx: Any, task_id: str, state: Any, *, surface: str,
    state_root: Optional[pathlib.Path] = None, fingerprint: str = "", chat_id: Any = None,
) -> None:
    """Invalidate the existing task-detail read model after its durable write."""
    event_queue = getattr(ctx, "event_queue", None)
    serialized = json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)
    revision = sha256(serialized.encode("utf-8")).hexdigest()
    try:
        from supervisor.message_bus import notification_chat_route

        chat_id = notification_chat_route(
            chat_id if chat_id is not None else getattr(ctx, "current_chat_id", None), 1,
        )
        if chat_id is None:
            chat_id = 1
    except (TypeError, ValueError):
        chat_id = 1
    ts = utc_now_iso()
    payload = {
        "type": "review_reference", "surface": surface,
        "task_id": str(task_id or ""), "chat_id": chat_id,
        "presentation_owner_task_id": str(task_id or ""),
        "review_fingerprint": fingerprint,
        "state_revision": revision, "ts": ts,
    }
    raw_root = (
        state_root
        or getattr(ctx, "budget_drive_root", "")
        or getattr(ctx, "drive_root", "")
    )
    if raw_root:
        # Existing progress JSONL is the reconnect/history rail. The row is an
        # opaque invalidation only; the task result remains the read-side owner.
        try:
            written = append_jsonl(pathlib.Path(raw_root) / "logs" / "progress.jsonl", {
                **payload,
                "direction": "out", "is_progress": True,
                "user_id": 0, "text": "", "content": "", "format": "",
            })
        except Exception:
            log.warning("Failed to append review progress reference", exc_info=True)
        else:
            if not written:
                log.warning("Failed to append review progress reference")
    emit_log_event(
        event_queue, payload, log_label=f"{surface.replace('_', '-')} state reference",
    )


def _record_plan_review_attempt_with_reference(
    ctx: Any,
    state_root: pathlib.Path,
    task_id: str,
    **attempt: Any,
) -> dict:
    """Persist an attempt and publish its durable revision immediately."""
    state = record_plan_review_attempt(state_root, task_id, **attempt)
    _emit_plan_review_reference(ctx, task_id, state, state_root=state_root)
    return state


def _record_raw_plan_request_with_reference(
    ctx: Any,
    state_root: pathlib.Path,
    task_id: str,
    envelope: dict,
    *,
    reason: str,
) -> str:
    """Persist an undecodable request and immediately invalidate Plan detail."""
    fingerprint = record_raw_plan_request_attempt(
        envelope, state_root, task_id, reason=reason,
    )
    _emit_plan_review_reference(ctx, task_id, state_root=state_root)
    return fingerprint


def _record_cycles_exhausted_with_references(
    ctx: Any,
    state_root: pathlib.Path,
    task_id: str,
    *,
    wave_fingerprint: str,
    attempt_fingerprint: str,
    cycles_paid: int,
    cap: int,
) -> dict:
    """Persist both cap writes and publish each durable revision immediately."""
    marked_wave = mark_plan_review_cycles_exhausted(
        state_root, task_id, fingerprint=wave_fingerprint,
    )
    _emit_plan_review_reference(ctx, task_id, state_root=state_root)
    attempt_state = record_plan_review_attempt(
        state_root, task_id, fingerprint=attempt_fingerprint,
        status="cycles_exhausted",
        reason=f"{cycles_paid}/{cap} paid plan-review cycles spent",
    )
    _emit_plan_review_reference(ctx, task_id, attempt_state, state_root=state_root)
    return marked_wave
