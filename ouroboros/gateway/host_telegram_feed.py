"""Host Service doors of the owner's own Telegram bot (A) for the Official ``telegram-bot`` connector.

The bundled ``telegram`` bridge reads A and keeps its token. While its poller runs in this process it registers
its feed with core (``telegram_owner_bot.register_bridge``); these doors authenticate the connector (its own skill
token and granted ``presence``, skill name ``telegram-bot`` only) and pass each request to that handler, which
owns the feed, the exact-place checks and the provider call (``skills/telegram/lib/telegram_group_feed.py``).
The identity, feed and relay doors serve only an Official release's exact bytes (``official_connector``, the
answer the ``owner_bot`` projection reads too); ``poller`` serves any admitted ``telegram-bot``, since it keeps
A from that connector. Core itself answers only who A is for arbitration: ``same_as_owner_bridge`` reads Settings
and the bridge's enablement, so a connector's own bot (B) learns it may not poll A in either start order, also
while the bridge's poller is not running. No door here carries a token or a provider URL; refusals are
Telegram-shaped 403 answers (never 5xx, which the connector reads as an unknown outcome) naming ``relay_refusal``.
"""
from __future__ import annotations

import asyncio
import functools
import json
import re
from typing import Any, Awaitable, Callable, Mapping

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ouroboros.telegram_owner_bot import FEED_CONNECTOR, current_bridge, official_connector, owner_bridge_bot_id

TELEGRAM_GROUP_FEED_VERSION = 1
_INFLIGHT = 8  # one long poll, the outbox, typing and a download at once, with room to spare
_MAX_BODY = 1024 * 1024  # JSON only; files travel by path (upload) or in the answer (file)


def _handler() -> Callable[[str, Mapping[str, Any]], Awaitable[tuple[int, Any, str]]] | None:
    bridge = current_bridge()
    return getattr(bridge, "handle", None)


def refusal(code: str, description: str, *, retry_after: int = 0) -> JSONResponse:
    """A relay refusal: nothing was sent, in the shape a Bot API refusal has."""
    body: dict[str, Any] = {"ok": False, "error_code": 403, "description": description, "relay_refusal": code}
    if retry_after:
        body["parameters"] = {"retry_after": retry_after}
    return JSONResponse(body, status_code=403)


async def _caller(request: Request, *, official: bool = True) -> tuple[Any, JSONResponse | None]:
    from ouroboros.gateway import host_service as host
    ctx = request.app.state.host_service_context
    try:
        skill, payload = await host._authenticated(ctx, request.headers.get("x-skill-token", ""), "presence")
    except host.HostServiceAuthError as exc:
        return ctx, refusal("auth_blocked", str(exc))
    except OSError:
        return ctx, refusal("auth_unavailable", "Ouroboros could not check this skill now.", retry_after=5)
    if skill != FEED_CONNECTOR:
        return ctx, refusal("connector_mismatch", "Only the telegram-bot connector uses these doors.")
    # The package whose token just authenticated: the Official check reads the same bytes, not a later landing.
    issued_for = str((payload or {}).get("content_hash") or "")
    if official and not await host._bounded_host_read(
            ctx, functools.partial(official_connector, ctx.data_dir, content_hash=issued_for)):
        return ctx, refusal("connector_mismatch", "Only an Official telegram-bot release, unchanged, reads your own "
                                                  "bot's groups.")
    return ctx, None


def _same_as_owner_bridge(data_dir: Any, bot_id: str) -> bool | None:
    """Whether ``bot_id`` is the owner's own bot A, the bridge's to read: the one Settings decision every process
    shares (``owner_bridge_bot_id``), never an in-memory holder a carrier worker does not have. ``None`` when Settings
    cannot be read: unknown, never a "no"."""
    if not bot_id.isdigit():
        return None
    try:
        from ouroboros.config import load_settings
        owner = owner_bridge_bot_id(data_dir, settings=load_settings(strict=True), strict=True)
    except Exception:
        return None  # unreadable: unknown, never a "no"
    return bool(owner) and owner == bot_id


def _answer(status: int, body: Any, media_type: str) -> Response:
    if isinstance(body, (bytes, bytearray)):
        return Response(bytes(body), status_code=status, media_type=media_type or "application/octet-stream")
    return JSONResponse(body, status_code=status)


async def _identity(request: Request) -> Response:
    """Who A is and its consent; with ``groups=1`` only, also ``group_ids``: the groups the bridge's index shows A in
    this consent period (``FeedBridge.group_ids``), ``null`` while unknown, the bridge not running here included."""
    ctx, refused = await _caller(request)
    if refused is not None:
        return refused
    same = await asyncio.to_thread(_same_as_owner_bridge, ctx.data_dir, str(request.query_params.get("bot_id") or ""))
    groups = request.query_params.get("groups") == "1"
    handler = _handler()
    facts: dict[str, Any] = {"account_id": "", "owner_bound": False, "invite_url": "", "enabled": False,
                             "reason": "bridge_not_ready", **({"group_ids": None} if groups else {})}
    if handler is not None:
        status, body, _media = await handler("identity", {"groups": "1"} if groups else {})
        if status == 200 and isinstance(body, dict):
            facts = {**({"group_ids": None} if groups else {}), **body}
    from ouroboros.telegram_owner_bot import connector_token_held
    held = await asyncio.to_thread(connector_token_held, ctx.data_dir)
    return JSONResponse({"v": TELEGRAM_GROUP_FEED_VERSION, **facts, "same_as_owner_bridge": same,
                         "connector_token_held": held})


_OPERATIONS = {"GET": frozenset({"identity", "updates", "file"}), "POST": frozenset({"call", "upload", "poller"})}
TELEGRAM_GROUP_POLL_CLAIM_VERSION = 1  # /identity: the connector asks before it polls its own bot (``poller``)


async def _poller(request: Request) -> Response:
    """``POST /telegram/feed/poller {bot_id}`` right before the connector's deleteWebhook and getUpdates; any admitted
    telegram-bot asks here, an Official release or not: the answer keeps A from it."""
    ctx, refused = await _caller(request, official=False)
    if refused is not None:
        return refused
    try:
        payload = json.loads((await request.body()).decode("utf-8") or "null")
    except (UnicodeDecodeError, ValueError, RecursionError):
        payload = None
    if not isinstance(payload, dict) or set(payload) != {"bot_id"} or not isinstance(payload["bot_id"], str):
        return JSONResponse({"ok": False, "code": "bot_id_invalid"}, status_code=400)
    from ouroboros.telegram_owner_bot import claim_connector_poll
    status, body = await asyncio.to_thread(claim_connector_poll, ctx.data_dir, payload["bot_id"])
    return JSONResponse(body, status_code=status)


async def _door(request: Request) -> Response:
    """``GET /telegram/feed/{identity|updates|file}`` and ``POST /telegram/feed/{call|upload}``."""
    operation = str(request.path_params.get("operation") or "")
    if operation not in _OPERATIONS.get(request.method, ()):
        return refusal("method_not_allowed", "No such feed door.")
    if operation == "identity":
        return await _identity(request)
    if operation == "poller":
        return await _poller(request)
    ctx, refused = await _caller(request)
    if refused is not None:
        return refused
    handler = _handler()
    if handler is None:
        return refusal("bridge_not_ready", "Your own Telegram bot is not running here.", retry_after=5)
    if request.method == "POST":
        raw = await request.body()
        if len(raw) > _MAX_BODY:
            return refusal("argument_not_allowed", "The request is too large; files travel by path.")
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except (UnicodeDecodeError, ValueError, RecursionError):
            payload = None
        if not isinstance(payload, dict):
            return refusal("argument_not_allowed", "The request must be one JSON object.")
    else:
        payload = dict(request.query_params)
        if "connected_revision" in payload:  # a query carries text: only a plain nonnegative integer reads as one
            payload["connected_revision"] = _revision(payload["connected_revision"])
    if not ctx._enter_inflight("telegram_feed", _INFLIGHT):
        return refusal("busy", "Too many requests for your own bot at once.", retry_after=1)
    try:
        status, body, media = await handler(operation, payload)
    finally:
        ctx._leave_inflight("telegram_feed")
    return _answer(status, body, media)


_REVISION = re.compile(r"0|[1-9][0-9]{0,17}")


def _revision(value: str) -> Any:
    """A GET door's ``connected_revision``: the integer its plain decimal text names, else the text as sent, which
    no place binds (a JSON body sends the integer itself)."""
    return int(value) if isinstance(value, str) and _REVISION.fullmatch(value) else value


class _Doors:
    """One door for every method: ``_door`` answers the GET and POST operations and refuses the rest itself."""

    async def __call__(self, scope, receive, send) -> None:
        response = await _door(Request(scope, receive))
        await response(scope, receive, send)


def host_telegram_feed_routes() -> list[Route]:
    return [Route("/telegram/feed/{operation}", _Doors())]


__all__ = ["TELEGRAM_GROUP_FEED_VERSION", "TELEGRAM_GROUP_POLL_CLAIM_VERSION", "host_telegram_feed_routes",
           "refusal"]
