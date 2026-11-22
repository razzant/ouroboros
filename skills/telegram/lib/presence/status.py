"""Circles-compatible ``GET status`` snapshot and payload builders for PresenceRuntime."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional

from ..telegram_mirror import mirror_mode, owner_chat_mode


def build_status_snapshot(runtime: Any, custody: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    custody = runtime.store.status_snapshot() if custody is None else custody
    return {
        "runtime_state": runtime._runtime_state,
        "bot_label": runtime._bot_label,
        **(
            {"identity": runtime.submitter.connector_identity}
            if getattr(runtime.submitter, "presence_identity", 0) == 1
            and getattr(runtime.submitter, "connector_identity", {})
            else {}
        ),
        "last_error": runtime._last_error,
        "delivery_reporting": getattr(runtime.submitter, "delivery_reporting_status", "unsupported"),
        "presence_continuation_version": getattr(runtime.submitter, "continuation_version", 0),
        "reaction_context": (
            "available" if getattr(runtime.submitter, "presence_reaction_context", 0) == 1 else "update_required"
        ),
        "reaction_history": (
            "available" if getattr(runtime.submitter, "presence_explicit_completion", 0) == 1 else "update_required"
        ),
        "last_event_at": custody.get("last_event_at", ""),
        "last_delivery_at": custody.get("last_delivery_at", ""),
    }


def build_status_payload(
    runtime: Any,
    bridge_status: Dict[str, Any],
    mini_app_status: Dict[str, Any],
    settings: Dict[str, Any],
    *,
    requested_account_id: str = "",
) -> Dict[str, Any]:
    """Full Circles-compatible ``GET status`` response combining Presence custody and bridge state."""
    counts = runtime.store.status_snapshot()
    snap = build_status_snapshot(runtime, counts)
    binding_id = str(settings.get("binding_id") or "").strip()
    account, unavailable = runtime.confirmed_bot()
    requested = str(requested_account_id or "").strip()
    mismatched = bool(requested and requested != account)
    groups = None
    if mismatched:
        unavailable = (
            f"Active bot is {account}, not {requested}" if account else (unavailable or "Telegram bot unavailable")
        )
    elif account:
        try:
            groups = runtime.store.offered_groups(account)
        except (sqlite3.Error, OSError):
            unavailable = "Observed chats could not be read"
    payload: Dict[str, Any] = {
        "runtime_state": snap.get("runtime_state", "not_started"),
        "bot_label": snap.get("bot_label", "not connected"),
        **({"account_id": account} if account else {}),
        **({"identity": snap["identity"]} if "identity" in snap and not mismatched else {}),
        "last_event_at": snap.get("last_event_at", ""),
        "last_delivery_at": snap.get("last_delivery_at", ""),
        "last_error": snap.get("last_error", ""),
        "delivery_reporting": snap.get("delivery_reporting", "unknown"),
        "presence_continuation_version": snap.get("presence_continuation_version", 0),
        "reaction_context": snap.get("reaction_context", "update_required"),
        "reaction_history": snap.get("reaction_history", "update_required"),
        "has_presence_binding": bool(binding_id),
        "management_group_id": str(settings.get("management_group_id") or ""),
        "binding_state": "configured" if binding_id else "missing",
        "owner_chat_mode": owner_chat_mode(settings),
        "mirror_mode": mirror_mode(settings),
        "bridge": bridge_status,
        "mini_app": mini_app_status,
        "connect": groups,
        **({"connect_unavailable": unavailable} if groups is None else {}),
        **counts,
    }
    held = payload.get("outbox_held", 0)
    text, tone = (
        (unavailable, "warn")
        if mismatched
        else (f"{held} held", "warn")
        if held
        else (payload["connect_unavailable"], "warn")
        if payload["connect"] is None
        else (payload["runtime_state"], "muted")
    )
    payload.update(text=text, tone=tone, title="Telegram")
    return payload
