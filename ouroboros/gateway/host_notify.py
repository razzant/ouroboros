"""``POST /notify``: one owner notification from a reviewed skill.

Immediate, or — with ``at``/``cron`` — a ``kind: "notify"`` row of the ONE
schedule table that the supervisor tick fires without a model turn. Its own
contract (grant, the events-row fact, keyed deferred rows, the owner's
suppression answers) and its own reason to change; it lives beside
``host_service.py`` on the same loopback trust boundary, under the same token
and grant checks, and ``create_host_service_app`` mounts it.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.event_bus import (OWNER_NOTIFICATION_KEY_CHARS, OWNER_NOTIFICATION_TEXT_CHARS, emit_owner_notification,
                                 owner_notification_chat_id)
from ouroboros.gateway._helpers import run_sync_to_completion
from ouroboros.gateway.host_service import HostServiceAuthError, HostServiceContext, _json_error
from ouroboros.utils import utc_now_iso


async def _api_notify(request: Request) -> JSONResponse:
    """One owner notification from a skill: never a chat row, never a model turn.

    The host stamps the source; the fact is one durable ``owner_notification``
    events row (its live browser frame is the append's log-sink copy) plus the
    ``owner.notification`` topic. ``key`` is the producer's identity for the
    notice (a redelivery collapses on the client); a failed write is 503."""
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, token_payload = ctx.authenticate_token_payload(
            request.headers.get("x-skill-token", "")
        )
        ctx.require_permission(skill_name, token_payload, "notify_owner")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:notify"):
        return _json_error("rate limit exceeded", 429)
    try:
        payload = await request.json()
    except Exception:
        return _json_error("invalid json", 400)
    if not isinstance(payload, dict):
        return _json_error("request body must be a JSON object", 400)
    cancel = payload.get("cancel", False)
    if not isinstance(cancel, bool):
        return _json_error("cancel must be a boolean", 400)
    text = payload.get("text", "")
    if not isinstance(text, str):
        return _json_error("text must be a string", 400)
    text = text.strip()
    if not text and not cancel:  # a cancel names a key, not a text
        return _json_error("text is required", 400)
    if len(text) > OWNER_NOTIFICATION_TEXT_CHARS:
        return _json_error(f"text must be at most {OWNER_NOTIFICATION_TEXT_CHARS} characters", 400)
    key = payload.get("key", "")
    if key is not None and not isinstance(key, str):
        return _json_error("key must be a string", 400)
    key = (key or "").strip()
    if len(key) > OWNER_NOTIFICATION_KEY_CHARS:
        return _json_error(f"key must be at most {OWNER_NOTIFICATION_KEY_CHARS} characters", 400)
    at_raw = payload.get("at")
    cron_raw = payload.get("cron")
    if at_raw is not None or cron_raw is not None or cancel:
        return await run_sync_to_completion(
            _schedule_owner_notification, ctx, skill_name, text, key,
            at_raw, cron_raw, payload.get("timezone"), cancel,
        )
    def _emit_now() -> Optional[Dict[str, Any]]:
        # The state read and the append both touch files under a lock: off the
        # event loop, like every other durable write this service performs.
        return emit_owner_notification(
            ctx.data_dir, chat_id=owner_notification_chat_id(ctx.data_dir),
            category="notice", text=text, source=f"skill:{skill_name}", key=key,
        )

    try:
        row = await run_sync_to_completion(_emit_now)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if row is None:
        return _json_error("notification log write failed; retry the same notice", 503)
    return JSONResponse({"ok": True, "ts": row["ts"], "chat_id": row["chat_id"]})


def _notify_schedule_id(skill_name: str, key: str) -> str:
    """One row per (skill, key) inside the schedule-id contract (≤81 URL-safe
    characters): the slug for humans, truncated to leave room for a hash of
    the raw (skill, key) pair, so names or keys that slug alike — or long
    names the truncation would fold together — never collide, and a long key
    never yields an id the owner's lifecycle endpoints refuse."""
    from hashlib import sha256

    from ouroboros.schedule_contract import schedule_slug

    digest = sha256(f"{skill_name}\n{key}".encode("utf-8")).hexdigest()[:8]
    slug = schedule_slug("notify", skill_name, key)[:72]
    # The slug keeps single dots; the id contract refuses ".." (a path guard), so
    # a key like "a..b" collapses its dot runs — the digest keeps it distinct.
    slug = re.sub(r"\.{2,}", ".", slug).rstrip("-._")
    return f"{slug}-{digest}"


def _notify_fresh_schedule_id(skill_name: str) -> str:
    """A keyless post is fire-and-forget: a fresh, contract-sized id each time."""
    from ouroboros.schedule_contract import schedule_slug

    return schedule_slug("notify", skill_name, utc_now_iso()[:19].replace(":", ""), os.urandom(3).hex())


def _schedule_owner_notification(ctx: "HostServiceContext", skill_name: str, text: str, key: str,
                                 at_raw: Any, cron_raw: Any, timezone_raw: Any, cancel: bool) -> JSONResponse:
    """A deferred notice is a ``kind: "notify"`` row of the ONE schedule table:
    the tick fires it without a model turn; ``key`` is the row's identity (a
    repeat moves it, ``cancel`` removes it through the audited delete — unless
    the owner suppressed the row, which stays and answers ``suppressed``);
    without a key a row is fire-and-forget."""
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.schedule_contract import cron_error, timezone_error
    from supervisor.queue import (
        ScheduleRefused, ScheduleStoreUnreadable, load_schedule_store, mutate_scheduled_task,
        schedule_transaction, upsert_scheduled_task,
    )

    source = f"skill:{skill_name}"
    if cancel:
        if at_raw is not None or cron_raw is not None:
            return _json_error("cancel cannot be combined with at or cron", 400)
        if not key:
            return _json_error("cancel requires the key of the scheduled notification", 400)
        schedule_id = _notify_schedule_id(skill_name, key)
        try:
            with schedule_transaction(ctx.data_dir):
                rows = load_schedule_store(ctx.data_dir).get("tasks") or []
                current = next((row for row in rows if str(row.get("id") or "") == schedule_id), None)
                if current is None or str(current.get("source") or "") != source:
                    return _json_error("no scheduled notification with that key", 404)
                outcome = mutate_scheduled_task(
                    "delete", schedule_id, drive_root=ctx.data_dir, actor=source,
                    reason="notification cancelled by its skill")
        except ScheduleStoreUnreadable as exc:
            return _json_error(str(exc), 503)
        if not outcome.get("ok") and not outcome.get("changed"):
            # The lifecycle refused (its audit could not be written): nothing
            # changed, and the skill must not read that as a cancellation.
            return _json_error(str(outcome.get("detail") or outcome.get("status") or "cancel refused"), 503)
        if outcome.get("status") == "suppressed" and not outcome.get("changed"):
            # The owner's off switch outlives the skill's cancel (see lifecycle).
            return JSONResponse({"ok": True, "cancelled": False, "id": schedule_id, "status": "suppressed"})
        if not outcome.get("ok"):
            # Removed, but the outcome audit was lost: both facts, no retry.
            return JSONResponse({"ok": False, "cancelled": True, "id": schedule_id,
                                 "status": str(outcome.get("status") or "changed_audit_incomplete")})
        return JSONResponse({"ok": True, "cancelled": bool(outcome.get("changed")), "id": schedule_id})
    if (at_raw is None) == (cron_raw is None):
        return _json_error("supply exactly one of at (ISO 8601 instant) or cron (5-field expression)", 400)
    timezone = str(timezone_raw or "").strip()
    if at_raw is not None:
        at_instant = parse_deadline_ts(at_raw.strip()) if isinstance(at_raw, str) else None
        if at_instant is None:
            return _json_error("at must be a parseable ISO 8601 instant", 400)
        trigger = {"type": "once", "run_at": at_instant.isoformat()}
    else:
        if not isinstance(cron_raw, str):
            return _json_error("cron must be a 5-field expression", 400)
        if err := cron_error(cron_raw.strip()):
            return _json_error(err, 400)
        trigger = {"type": "cron", "expr": cron_raw.strip()}
    if err := timezone_error(timezone):
        return _json_error(err, 400)
    schedule_id = _notify_schedule_id(skill_name, key) if key else _notify_fresh_schedule_id(skill_name)
    # ``name`` is audit material; the sentence is not — the UI reads it from
    # ``notification`` itself.
    record = {
        "id": schedule_id, "name": f"Reminder from {skill_name}", "kind": "notify", "source": source,
        "enabled": True, "timezone": timezone, "trigger": trigger,
        "notification": {"text": text, "key": key},
    }
    try:
        with schedule_transaction(ctx.data_dir):
            rows = load_schedule_store(ctx.data_dir).get("tasks") or []
            current = next((row for row in rows if str(row.get("id") or "") == schedule_id), None)
            if current is not None:
                if str(current.get("source") or "") != source:
                    return _json_error("that schedule id belongs to another source", 409)
                if str(current.get("manual_override") or "") in ("disabled", "deleted"):
                    # The owner switched this reminder off; only their restore lifts it.
                    return JSONResponse({"ok": True, "scheduled": False, "id": schedule_id,
                                         "status": "suppressed"})
            stored = upsert_scheduled_task(
                record, drive_root=ctx.data_dir, actor=source,
                reason="notification scheduled by its skill")
    except ScheduleRefused as refusal:
        # An audit that could not be written is the host failing, not the skill: retry.
        return _json_error(refusal.message, 503 if refusal.status == "audit_unavailable" else 400)
    except ScheduleStoreUnreadable as exc:
        return _json_error(str(exc), 503)
    return JSONResponse({
        "ok": stored.get("audit") == "recorded", "scheduled": True, "id": schedule_id,
        "next_run_at": stored.get("next_run_at") or trigger.get("run_at") or "",
    })
