"""How the bridge's poller and routes use the group feed (``telegram_group_feed.py``).

One feed per loaded bridge. The verified bot is the one getMe confirmed for the token generation in force now
(``note_verified`` after validation); a replaced or cleared token reads as unconfirmed until getMe answers for it.
The feed is registered with core only while the poller runs here, so the Host doors and the exact place writer
refuse while it does not. Nothing here keeps a bridge's API alive: its feed and verified bot are held by a weak
key (a stand-in without weak references is pinned instead) and forgotten at its unload (``forget``), the feed's own
callbacks reach the API through a weak reference, and a reload builds a new feed.
"""
from __future__ import annotations

import asyncio
import pathlib
import sqlite3
import weakref
from typing import Any, Optional

from starlette.responses import JSONResponse

from .telegram_api import TelegramRequestRejected
from .telegram_group_feed import ALLOWED_UPDATES, FeedBridge, FeedRefusal, group_chat
from .telegram_poll_binding import bot_id
from .telegram_state import _data_dir, _load_settings
from .telegram_token import binding_for

_VERIFIED: "weakref.WeakKeyDictionary[object, tuple[int, str, str]]" = weakref.WeakKeyDictionary()
_PINNED: dict[int, tuple[object, tuple[int, str, str]]] = {}  # an API object without weak references (stand-ins)
_FEEDS: "weakref.WeakKeyDictionary[object, FeedBridge]" = weakref.WeakKeyDictionary()
_PINNED_FEEDS: dict[int, tuple[object, FeedBridge]] = {}


def _api_ref(api):
    """A way back to ``api`` that does not keep it alive (a stand-in without weak references is pinned anyway)."""
    try:
        return weakref.ref(api)
    except TypeError:
        return lambda: api


def forget(api) -> None:
    """At unload: this API's feed leaves core (if registered) and every record of it here goes."""
    bridge = _held_feed(api)
    if bridge is not None:
        withdraw = getattr(bridge, "withdraw", None)
        if callable(withdraw):
            withdraw()
    for weak in (_FEEDS, _VERIFIED):
        try:
            weak.pop(api, None)
        except TypeError:
            pass
    for pinned in (_PINNED_FEEDS, _PINNED):
        held = pinned.get(id(api))
        if held is not None and held[0] is api:
            pinned.pop(id(api), None)


def _held_feed(api) -> Optional[FeedBridge]:
    try:
        return _FEEDS.get(api)
    except TypeError:
        held = _PINNED_FEEDS.get(id(api))
        return held[1] if held and held[0] is api else None


def note_verified(api, generation: int, bot: str, me: Any) -> None:
    """getMe answered ``me`` for the token ``generation``: its bot and @name are the verified identity."""
    me = me.get("result") if isinstance(me, dict) and isinstance(me.get("result"), dict) else me
    username = str(me.get("username") or "") if isinstance(me, dict) and str(me.get("id") or "") == str(bot) else ""
    record = (int(generation), str(bot), username)
    try:
        _VERIFIED[api] = record
    except TypeError:
        _PINNED[id(api)] = (api, record)


def _noted(api) -> Optional[tuple[int, str, str]]:
    try:
        return _VERIFIED.get(api)
    except TypeError:
        held = _PINNED.get(id(api))
        return held[1] if held and held[0] is api else None


def verified(api) -> Optional[tuple[str, str]]:
    """The bot getMe confirmed for the token in force, while this bridge is its reader here; else None."""
    binding = binding_for(api)
    token, generation = binding.snapshot()
    noted = _noted(api)
    if not noted or not binding.is_validated() or noted[0] != generation or noted[1] != bot_id(token):
        return None
    try:
        from ouroboros.telegram_owner_bot import NATIVE_SKILL, bot_holder
    except ImportError:
        return noted[1], noted[2]
    if bot_holder(noted[1]) != NATIVE_SKILL:
        return None
    # A permanently rejected token parks the poller; its earlier getMe identity is not a running reader. Readiness
    # and consent follow the poller's own recorded state, never a stale identity (the last name stays for display).
    if binding.noted()[0] == "error":
        return None
    return noted[1], noted[2]


class TelegramBotInUse(TelegramRequestRejected):
    """This process gave the bot to the Telegram Bot connector's own reader: the bridge does not contact Telegram
    with it (no getMe, no getUpdates) and checks again after its back-off; only a restart or another bot ends it."""

    @property
    def transient(self) -> bool:
        return True


def claim_before_contact(client) -> None:
    """Before the bridge's first request with a token (start, or a rotated or replaced token): this bot's
    one local reader is the bridge, unless the connector already holds it (``TelegramBotInUse``)."""
    try:
        from ouroboros.telegram_owner_bot import NATIVE_SKILL, claim_bot
    except ImportError:  # an older core: no local arbitration, as before
        return
    if claim_bot(NATIVE_SKILL, client.bot_id) != NATIVE_SKILL:
        raise TelegramBotInUse("The Telegram Bot connector reads this bot here.", status_code=409)


def _owner_bound(api) -> bool:
    try:
        return int(str(_load_settings(api).get("TELEGRAM_CHAT_ID") or "").strip()) > 0
    except Exception:
        return False


def _no_client():
    raise RuntimeError("this feed has no provider client")


def install(api, client_factory) -> None:
    """At registration: this bridge's feed with its client factory (``client_factory(api)`` returns the bridge's
    ``(client, generation)``), its owner door ``group_feed`` and its unload."""
    ref = _api_ref(api)

    def client():
        current = ref()
        if current is None:
            raise RuntimeError("this bridge was unloaded")
        return client_factory(current)[0]
    feed(api).client_factory = client
    api.register_route("group_feed", handler=make_route(api), methods=("GET", "POST"))
    if callable(getattr(api, "on_unload", None)):
        api.on_unload(lambda: (lambda current: current is not None and forget(current))(ref()))


def feed(api) -> FeedBridge:
    """The one feed of this bridge (``api``); without ``install`` it keeps and reads, but calls no provider."""
    held = _held_feed(api)
    if held is not None:
        return held
    ref = _api_ref(api)

    def through(call, default):
        current = ref()
        return default if current is None else call(current)
    bridge = FeedBridge(pathlib.Path(api.get_state_dir()), _data_dir(api), client_factory=_no_client,
                        verified=lambda: through(verified, None), owner_bound=lambda: through(_owner_bound, False),
                        log=lambda *args, **fields: through(lambda current: current.log(*args, **fields), None))
    try:
        _FEEDS[api] = bridge
    except TypeError:
        _PINNED_FEEDS[id(api)] = (api, bridge)
    return bridge


async def while_polling(api, polling) -> None:
    """Run the poller coroutine ``polling`` with the feed registered with core, withdrawn when it ends."""
    bridge = feed(api)
    bridge.accepted_poll = None
    bridge.withdraw = register_running(api, bridge)
    try:
        await polling
    finally:
        bridge.withdraw()


def allowed_updates(period: Optional[dict]) -> Optional[list[str]]:
    """The subscription a getUpdates received under ``period`` names: groups' kinds beside the bridge's own while
    groups are on for its bot and the owner is bound (``FeedBridge.intake``); None keeps Telegram's remembered one."""
    return list(ALLOWED_UPDATES) if period is not None else None


async def get_updates(bridge: FeedBridge, client, offset: int) -> tuple[list, Optional[dict]]:
    """getUpdates and the consent record its batch is received under, read once before the request: as before while
    groups are off or no owner is bound (no selection named, nothing of the batch kept)."""
    period = await asyncio.to_thread(bridge.intake, client.bot_id)
    selection = allowed_updates(period)
    if selection is None:
        return await client.get_updates(offset), period
    if bridge.accepted_poll != period:
        # Establish the new subscription without a second full idle poll. The offset and the single reader stay put.
        updates = await client.get_updates(offset, allowed_updates=selection, poll_timeout=0)
        bridge.accepted_poll = period
        return updates, period
    return await client.get_updates(offset, allowed_updates=selection), period


def on_group(api, bridge: FeedBridge, client, period: Optional[dict] = None, *, cursor=None):
    """The poll batch's group intake under its ``period``: keep the update in the feed (never raises; a failure is the
    feed's gap). Without a period the update is confirmed untouched: nothing kept, no disk read."""
    async def take(update) -> None:
        observed = cursor.group_period(update, period) if cursor is not None else period
        if observed is None:
            return
        try:
            await asyncio.to_thread(bridge.publish, update, client.bot_id, observed)
        except Exception as exc:  # the feed counts its own gaps; this keeps the owner's batch moving regardless
            api.log("warning", f"Telegram group update could not be kept ({type(exc).__name__}).")
    return take


def register_running(api, bridge: FeedBridge):
    """Register the feed with core while the poller runs; returns the callable that withdraws it."""
    try:
        from ouroboros.telegram_owner_bot import register_bridge
    except ImportError:  # an older core without the feed: groups stay as before
        return lambda: None
    return register_bridge(bridge)


def make_route(api):
    """``GET /api/extensions/telegram/group_feed``: status; ``POST {"enabled", "expected_account_id"}``: consent for the
    bot the owner saw (Off: the ``consent_account_id`` they saw). After validation the POST carries local status;
    a malformed body is 400 with no status or write. Read status to reconcile, never write again blindly."""
    async def route(request):
        bridge = feed(api)
        if request.method == "POST":
            try:
                body = await request.json()
            except ValueError:
                body = None
            if (not isinstance(body, dict) or set(body) - {"enabled", "expected_account_id"}
                    or type(body.get("enabled")) is not bool
                    or not isinstance(body.get("expected_account_id", ""), str)):
                return JSONResponse({"ok": False, "code": "invalid", "error": "Send enabled and the bot you saw."},
                                    status_code=400)
            try:
                await asyncio.to_thread(bridge.set_consent, body["enabled"], body.get("expected_account_id", ""))
            except FeedRefusal as refusal:
                status = 503 if refusal.code == "consent_unknown" else 409
                return JSONResponse({"ok": False, "code": refusal.code, "error": refusal.description,
                                     **await asyncio.to_thread(_status, api, bridge)}, status_code=status)
            except (OSError, sqlite3.Error):
                return JSONResponse({"ok": False, "code": "unwritable", "error": "The choice could not be saved.",
                                     **await asyncio.to_thread(_status, api, bridge)}, status_code=503)
            return JSONResponse({"ok": True, **await asyncio.to_thread(_status, api, bridge)})
        return JSONResponse(await asyncio.to_thread(_status, api, bridge))
    return route


def _status(api, bridge: FeedBridge) -> dict[str, Any]:
    """The feed's status with the bridge's ``state``: ready only for the verified bot of a poller running here."""
    try:
        from ouroboros.telegram_owner_bot import connector_token_held, current_bridge
        running, held = current_bridge() is bridge, connector_token_held(bridge.data_dir)
    except ImportError:
        running, held = False, None
    token = binding_for(api).snapshot()[0]
    state = ("no_token" if not token else "bridge_not_ready" if not running
             else "ready" if verified(api) else "identity_unconfirmed")
    # The Telegram Bot connector's own token names this bridge's bot, which it is not given: an older connector's
    # Widget then says only that it has no token, so this is where the owner sees why.
    status = bridge.status()
    return {**status, "state": state, "intake_ready": state == "ready" and status["intake_ready"],
            "connector_token_held": held}


__all__ = ["TelegramBotInUse", "allowed_updates", "claim_before_contact", "feed", "forget", "get_updates", "group_chat",
           "install", "make_route", "note_verified", "on_group",
           "register_running", "verified", "while_polling"]
