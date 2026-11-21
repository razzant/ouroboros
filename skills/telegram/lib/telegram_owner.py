"""Who the bridge's owner is: the chat that opened the pending pairing link, or else the first private writer.

``owner_gate`` decides every polled update before anything else reads it. A private chat's ``/start <code>``
matching the pending code (``TELEGRAM_PAIRING_CODE``, lib/telegram_pairing.py) binds that chat as the owner,
replacing an earlier one (saving the code was the owner's own act), consumes the code, registers the owner with the
core's emergency stop and greets once; it never reaches the Main chat. While a code is pending nothing else binds;
with none, or one this skill may not read (not granted), the first positive private chat binds, as before (owner
rule of 02.09). While the saved settings cannot be read nothing binds; the next update asks again. ``/start`` is the
bridge's own in every command mode: the owner is greeted again, and a private chat whose code binds nothing is told
once that the link is no longer valid. Other updates from a non-owner are refused as before; the owner's own go on.

The emergency stop accepts an external ``/panic`` only from the core's external command owner
(``owner_external_id``/``owner_external_chat_id``, ``ouroboros/server_control.PanicIngress``). The core's ordinary
intake binds that pair only from a first forwarded slash command, which Strict mode never forwards and Panic itself
never is (``/chat/inject`` takes it to the emergency door alone). So the gate registers the owner itself through the
Host Service (``POST /owner/external``): at the pairing, and for an owner bound earlier while the core's slot is
still empty, on that owner's next update, before a ``/panic`` in it is sent. The first ``/panic`` after any owner
binding then stops, in every command mode.
"""
from __future__ import annotations

import asyncio
import contextlib
import pathlib
import weakref
from typing import Any, Optional

from starlette.responses import JSONResponse

from ..scripts.telegram_settings import merge_settings
from . import telegram_pairing as pairing
from .miniapp_registration import _read_status
from .telegram_api import _LOCALIZED_TEXTS, TelegramRequestRejected
from .telegram_i18n import Index
from .telegram_poll_binding import bot_id
from .telegram_state import _data_dir, _owner_message_seen
from .telegram_token import accepted_token

# The wait of the /panic inject itself (plugin.py): registering the owner never holds an emergency longer.
_HOST_WAIT_SEC = 2.0
_UNREAD = object()
_TEXTS = Index("pairing", {
    "greeting": "✅ Connected: this chat now talks with Ouroboros. /menu opens the controls; /panic is the emergency stop.",
    "link_invalid": "This link is no longer valid. Open the Telegram card in Ouroboros for a fresh QR code.",
}, "the Telegram bridge's pairing lines: the greeting of a connected owner, a link that no longer binds")
_IDENTITY: "weakref.WeakKeyDictionary[Any, tuple[str, str]]" = weakref.WeakKeyDictionary()
_SETTLED: "weakref.WeakKeyDictionary[Any, set]" = weakref.WeakKeyDictionary()


def saved_code(api) -> Optional[str]:
    """The saved ``TELEGRAM_PAIRING_CODE`` as this skill may read it now: "" when none is saved, when this skill
    holds no grant for it (no code: the first-writer rule) or when it cannot be a code; None while the saved
    settings cannot be read."""
    try:
        reader = getattr(api, "get_settings_state", None)
        reply = (reader([pairing.CODE_KEY]) if callable(reader)
                 else {"status": "ok", "values": dict(api.get_settings([pairing.CODE_KEY]) or {})})
    except Exception:  # noqa: BLE001 — a read that failed says nothing about the code
        return None
    status = str((reply or {}).get("status") or "")
    if status == "unknown":
        return None
    value = ((reply.get("values") or {}).get(pairing.CODE_KEY) if status == "ok" else "") or ""
    return pairing.normalize_code(str(value).strip())


async def owner_gate(api, client, update: dict, pinned_chat: str, lang: str, post) -> tuple[Optional[str], bool]:
    """``(owner chat, answered)`` for one update: the owner's chat id when it comes from the owner (bound before or
    by this update), and whether the gate answered it itself (a ``/start``), so nothing else handles it.

    ``post(api, path, payload)`` is the bridge's Host Service call. A chat or sender id that is not a number raises
    ``ValueError`` (the poller isolates that update), as before."""
    callback = update.get("callback_query") or {}
    message = update.get("message") or {}
    chat = ((callback.get("message") or {}).get("chat") or (message.get("chat") or {}))
    sender = callback.get("from") or message.get("from") or {}
    chat_id = int(chat.get("id") or 0)
    user_id = int(sender.get("id") or 0)
    if chat_id <= 0 or user_id <= 0:
        return None, False
    state_dir = pathlib.Path(api.get_state_dir())
    private = str(chat.get("type") or "") == "private" and chat_id == user_id
    start = None if callback else pairing.start_command(message.get("text"))
    code: Any = _UNREAD
    if start and private:
        code = saved_code(api)
        digest = pairing.matching_digest(state_dir, code, start)
        if digest:
            await _pair(api, client, chat_id, digest, lang, post, message)
            return str(chat_id), True
    if not pinned_chat and private:
        code = saved_code(api) if code is _UNREAD else code
        if code is not None and pairing.first_writer_may_bind(pairing.pending(state_dir, code)):
            _bind(api, client, chat_id)
            pinned_chat = str(chat_id)
    if pinned_chat and str(chat_id) == pinned_chat and str(user_id) == pinned_chat:
        await ensure_core_owner(api, chat_id, post)
        if start is None:
            return pinned_chat, False
        _owner_message_seen(api, client.bot_id, chat_id, int(message.get("message_id") or 0))
        await greet(api, client, chat_id, lang)
        return pinned_chat, True
    if callback:
        with contextlib.suppress(Exception):  # a refused press is told once, never retried
            await client.answer_callback_query(str(callback.get("id") or ""),
                                               text=_LOCALIZED_TEXTS[lang]["not_authorized"])
    elif start and private:
        await client.send_message(chat_id, _TEXTS[lang]["link_invalid"])
    return None, False


def _bind(api, client, chat_id: int) -> None:
    """Save the owner chat where the bridge always kept it (its own settings), only while the client's token is
    still the one in force: an update polled for a replaced token binds nothing and stays unconfirmed."""
    check = getattr(client, "check_binding", None)
    if callable(check):
        check()
    merge_settings(pathlib.Path(api.get_state_dir()), {"TELEGRAM_CHAT_ID": str(chat_id)})


async def _pair(api, client, chat_id: int, digest: str, lang: str, post, message: dict) -> None:
    """The pending code's ``/start`` from a private chat: bind it first (a code consumed before the bind would leave
    no owner and no code), then consume the code, register the owner with the emergency stop and greet."""
    _bind(api, client, chat_id)
    pairing.record_consumed(pathlib.Path(api.get_state_dir()), digest)
    api.log("info", "Telegram owner bound by the pairing link.")
    await ensure_core_owner(api, chat_id, post, force=True)
    _owner_message_seen(api, client.bot_id, chat_id, int(message.get("message_id") or 0))
    await greet(api, client, chat_id, lang)


async def greet(api, client, chat_id: int, lang: str) -> None:
    """One greeting, with the Mini App's Open button while the Mini App is ready (it starts after the binding)."""
    text = _TEXTS[lang]["greeting"]
    url = (await asyncio.to_thread(_read_status, api)).get("public_url")
    if url:
        try:
            await client.send_message_with_inline_keyboard(
                chat_id, text, [[{"text": _LOCALIZED_TEXTS[lang]["btn_open_app"], "web_app": {"url": url}}]])
            return
        except TelegramRequestRejected as exc:
            if not exc.plain_retry_safe:
                raise
    await client.send_message(chat_id, text)


def _settled(api) -> set:
    """The owner chats whose registration this load already settled (bound, the same, or another pair)."""
    try:
        return _SETTLED.setdefault(api, set())
    except TypeError:  # an API stand-in without weak references keeps nothing: each update asks again
        return set()


def _core_slot(api) -> Any:
    """The core's external owner pair as its state file holds it now, read without a lock: "empty", the
    ``(user, chat)`` pair, or "unknown" when the file or its initialization cannot be read."""
    try:
        from supervisor.state import control_in_copy

        path = _data_dir(api) / "state" / "state.json"
        (user_known, user), (chat_known, chat) = (
            control_in_copy(path, key) for key in ("owner_external_id", "owner_external_chat_id"))
    except Exception:  # noqa: BLE001 — an unreadable binding proves nothing
        return "unknown"
    if not (user_known and chat_known):
        return "unknown"
    if user is None:
        return "empty"
    return (user, chat) if type(user) is int and type(chat) is int else "unknown"


async def ensure_core_owner(api, chat_id: int, post, *, force: bool = False) -> str:
    """Register the owner chat as the core's external command owner while the core's slot is empty, once per load.

    Without ``force`` the slot is read first: an unknown slot or another pair asks the host nothing (it could bind
    neither); an empty slot, or this pair (so the emergency door observes it), asks once. ``force`` (a pairing) asks
    at once. A known answer (bound, same, other) settles the chat for this load; a failure asks again at the owner's
    next update. Returns that answer, "settled", "unknown" or "failed"."""
    settled = _settled(api)
    if chat_id in settled:
        return "settled"
    slot = "ask" if force else _core_slot(api)
    if slot == "unknown":
        return slot
    if isinstance(slot, tuple) and slot != (chat_id, chat_id):
        settled.add(chat_id)
        _note_other_owner(api)
        return "other"
    try:
        status, body = await asyncio.wait_for(
            post(api, "/owner/external", {"user_id": chat_id, "chat_id": chat_id}), _HOST_WAIT_SEC)
    except Exception as exc:  # noqa: BLE001 — one line; the next update asks again
        api.log("warning", f"Telegram could not register its owner with the emergency stop yet "
                           f"({type(exc).__name__}); the owner's next message asks again.")
        return "failed"
    outcome = str(body.get("owner") or "") if status < 400 else ""
    if outcome in ("bound", "same", "other"):
        settled.add(chat_id)
    if outcome == "bound":
        api.log("info", "Telegram owner registered with the emergency stop: the first /panic from this chat stops.")
    elif outcome == "other":
        _note_other_owner(api)
    elif outcome != "same":
        api.log("warning", f"Telegram could not register its owner with the emergency stop yet (host answered "
                           f"{status} {outcome or 'nothing'}); the owner's next message asks again.")
    return outcome or "failed"


def _note_other_owner(api) -> None:
    api.log("warning", "The core's emergency stop is bound to another chat than this Telegram owner: /panic from "
                       "here is refused until Ouroboros is reset.")


def note_identity(api, bot: str, reply: Any) -> None:
    """Keep the username the validating ``getMe`` answered for ``bot``; the status route names the bot by it."""
    result = reply.get("result") if isinstance(reply, dict) else None
    username = result.get("username") if isinstance(result, dict) else None
    if bot and pairing.username_ok(username):
        with contextlib.suppress(TypeError):  # an API stand-in without weak references: no name to show
            _IDENTITY[api] = (bot, username)


def bot_username(api) -> Optional[str]:
    """The bot's username, only for the token in force (another bot's name is never shown for it)."""
    try:
        bot, username = _IDENTITY.get(api) or ("", "")
    except TypeError:
        return None
    return username if bot and bot == bot_id(accepted_token(api)) else None


def register_pairing(api) -> None:
    """``pairing/status`` (GET), served as ``miniapp/status`` is: the bot's username (null until the bridge has
    validated its token), whether an owner is bound and whether the bridge sees a pending pairing code it may read.
    Never the code or the token."""
    def facts() -> dict:
        state_dir = pathlib.Path(api.get_state_dir())
        return {"bot_username": bot_username(api), **pairing.local_status(state_dir, saved_code(api) or "")}

    async def status(_request):
        return JSONResponse(await asyncio.to_thread(facts))

    api.register_route("pairing/status", handler=status, methods=("GET",))


__all__ = ["bot_username", "ensure_core_owner", "greet", "note_identity", "owner_gate", "register_pairing", "saved_code"]
