"""Small owner-visible invalidations for canonical task review state writes."""

from __future__ import annotations

import json
import logging
import pathlib
from hashlib import sha256
from typing import Any, Optional

from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID
from ouroboros.utils import emit_log_event, utc_now_iso

log = logging.getLogger(__name__)


def _emit_review_reference(
    ctx: Any, task_id: str, state: Any, *, surface: str,
    state_root: Optional[pathlib.Path] = None, fingerprint: str = "", chat_id: Any = None,
) -> None:
    """Invalidate the existing task-detail read model after its durable write.

    The row is addressed like every other task-scoped notice: the task's DURABLE
    project binding first, then the chat the caller named, and the hidden
    partition when neither answers. Main is never a default — a review row for a
    run that has no owner-visible room belongs to the hidden partition, not to
    the owner's conversation (`chat_id_policy`, owner decision 6a=A).
    """
    event_queue = getattr(ctx, "event_queue", None)
    serialized = json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)
    revision = sha256(serialized.encode("utf-8")).hexdigest()
    try:
        from supervisor.log_addressing import resolve_project_chat
        from supervisor.message_bus import notification_chat_route

        metadata = getattr(ctx, "task_metadata", None) or {}
        bound = resolve_project_chat(
            getattr(ctx, "budget_drive_root", "") or getattr(ctx, "drive_root", ""),
            task_id,
            metadata.get("parent_task_id"),
            metadata.get("root_task_id"),
        )
        chat_id = notification_chat_route(
            bound or None,
            chat_id if chat_id is not None else getattr(ctx, "current_chat_id", None),
        )
        if chat_id is None:
            chat_id = HIDDEN_CHAT_ID
    except (TypeError, ValueError):
        chat_id = HIDDEN_CHAT_ID
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
        # The progress stream of the dialogue store is the reconnect/history rail. The row is
        # an opaque invalidation only; the task result remains the read-side owner.
        from ouroboros import dialogue_store

        written = dialogue_store.append_quietly(pathlib.Path(raw_root), "progress", {
            **payload,
            "direction": "out", "is_progress": True,
            "user_id": 0, "text": "", "content": "", "format": "",
        })
        if written is None:
            log.warning("Failed to append review progress reference")
    emit_log_event(
        event_queue, payload, log_label=f"{surface.replace('_', '-')} state reference",
    )
