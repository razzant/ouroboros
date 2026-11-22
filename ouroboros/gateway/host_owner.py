"""The external command owner a reviewed transport registers when it has bound its owner.

A transport skill that proved who its owner is (Telegram: the owner chat it bound by a one-time pairing code, or
the first private chat) registers that owner's identity as the core's external command owner,
``owner_external_id``/``owner_external_chat_id``, and the emergency door observes it at once
(``server_control.PanicIngress``), so the owner's first ``/panic`` from there stops. ``POST /owner/external``
binds only into a slot known to be empty, exactly as the core's first forwarded slash command does (server.py);
a bound pair is never replaced (it stays until owner Reset), and a slot that cannot be read binds nothing. The
same skill token and ``inject_chat`` grant admit it as ``/chat/inject``, which already binds that slot with the
first slash command it carries: the route grants no new power, it only stops the binding from needing a command
that Strict mode never forwards and Panic never is.

Answer: ``{"ok": true, "owner": "bound" | "same" | "pending" | "other" | "unknown"}``. ``bound`` (written now) and
``same`` (already this pair) are answered only once the live emergency door, read after the write, holds the pair:
a door the core built from an earlier read while it started is reached too. ``pending``: the slot holds this pair
(written now or before) but no live door holds it yet; the transport binds that same owner and asks again, so the
slot and the transport's owner never name different chats. ``other``: another pair holds the slot. ``unknown``: the
state cannot be read, and nothing was written.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.gateway.host_service import HostServiceAuthError, _authenticated, _json_error
from ouroboros.utils import utc_now_iso


def _positive(value: Any) -> bool:
    return type(value) is int and value > 0


def _door_holds(bridge_getter: Callable[[], Any], saved: dict, pair: tuple) -> bool:
    """Whether the live emergency door, observing ``saved`` now, accepts ``pair``; no published bridge: not yet."""
    try:
        panic = getattr(bridge_getter(), "panic", None)
    except Exception:  # noqa: BLE001 — no bridge published yet
        return False
    if panic is None:
        return False
    panic.observe_owner(saved)
    return panic.observed_owner() == pair


def bind_external_owner(bridge_getter: Callable[[], Any], user_id: int, chat_id: int) -> str:
    """Bind ``(user_id, chat_id)`` into the empty external owner slot; the live door observes the saved pair."""
    from ouroboros.server_control import external_owner_binding
    from supervisor.state import StateUnavailable, control_value, load_state, update_state

    current = load_state()
    known, owner, pair = external_owner_binding(current)
    if not known:
        return "unknown"
    if owner is not None:
        if pair != (user_id, chat_id):
            return "other"
        return "same" if _door_holds(bridge_getter, current, pair) else "pending"
    outcome = {"value": "unknown"}

    def bind(live: dict) -> None:
        if control_value(live, "owner_external_id") == (True, None):
            live["owner_external_id"] = user_id
            live["owner_external_chat_id"] = chat_id
            live["owner_external_bound_at"] = utc_now_iso()
            outcome["value"] = "bound"
            return
        live_known, _owner, live_pair = external_owner_binding(live)  # bound meanwhile, by another writer
        outcome["value"] = ("same" if live_pair == (user_id, chat_id) else "other") if live_known else "unknown"

    try:
        saved = update_state(bind)
    except StateUnavailable:
        return "unknown"
    if outcome["value"] in ("bound", "same") and not _door_holds(bridge_getter, saved, (user_id, chat_id)):
        return "pending"  # saved for this pair; the door cannot accept it yet: a later ask answers "same"
    return outcome["value"]


async def api_owner_external(request: Request) -> JSONResponse:
    ctx = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""), "inject_chat")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:owner"):
        return _json_error("rate limit exceeded", 429)
    try:
        body = await request.json()
    except ValueError:
        return _json_error("invalid json", 400)
    user_id = body.get("user_id") if isinstance(body, dict) else None
    chat_id = body.get("chat_id") if isinstance(body, dict) else None
    if not (_positive(user_id) and _positive(chat_id)):
        return _json_error("user_id and chat_id must be positive integers", 400)
    try:
        outcome = await asyncio.to_thread(bind_external_owner, ctx.bridge_getter, user_id, chat_id)
    except Exception as exc:  # noqa: BLE001 — the state writer's failure, named by its type only
        return _json_error(f"the owner could not be registered ({type(exc).__name__})", 503)
    return JSONResponse({"ok": True, "owner": outcome})


def host_owner_routes() -> list:
    return [Route("/owner/external", api_owner_external, methods=["POST"])]
