"""Who the bridge's owner is: the chat that opened the pending pairing link, or else the first private writer.

``owner_gate`` decides every polled update before anything else reads it. The pairing code is this skill's own
write-only setting (lib/telegram_pairing.py). A private chat's ``/start <code>`` (as text or as a caption) matching
the code that waits first asks the core whether its emergency stop can be this chat's (``POST /owner/external``).
When it can (the slot was empty, or is this chat already), one write of the skill's settings binds the chat, clears
the code and keeps its digest, and the chat is greeted. When another Telegram account holds the emergency stop
nothing changes: the bridge keeps its owner, the code keeps waiting, the scanner is told why and ``pairing/status``
says ``blocked``. When the core cannot answer, nothing changes and the scanner is asked to open the link again.
While a code waits nothing else binds; with none, the first positive private chat binds, as before (owner rule of
02.09). While the settings cannot be read nothing binds. ``/start`` never reaches the Main chat in any command mode:
the owner is greeted again (or told that a stale link no longer binds), and while a code waits anyone else's
``/start`` is told the link does not bind. Every reply of the gate is sent beside the poll loop with its own time
limit, so no reply holds the next update, an owner's ``/panic`` among them.

The emergency stop accepts an external ``/panic`` only from the core's external command owner
(``owner_external_id``/``owner_external_chat_id``, ``ouroboros/server_control.PanicIngress``), a pair written once
into an empty slot and kept until owner Reset. The bridge registers its owner there itself: at a pairing, before it
binds; from its load on for an owner bound earlier or by a first writer (``keep_owner_registered``, asked again
while the core cannot settle it); and when a ``/panic`` was refused before that registration settled
(``dispatch_panic`` registers and sends it once more). A settled owner's ``/panic`` goes out at once, alone.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import pathlib
import weakref
from typing import Any, Optional

from starlette.responses import JSONResponse

from ..scripts.telegram_settings import TelegramSettingsError, load_settings, merge_settings
from . import telegram_pairing as pairing
from .miniapp_registration import _read_status
from .telegram_api import _LOCALIZED_TEXTS, TelegramRequestRejected
from .telegram_i18n import Index
from .telegram_poll_binding import HostRefused, bot_id
from .telegram_state import _owner_message_seen
from .telegram_token import accepted_token, binding_for

# Longer than the core's state lock wait (4 s, supervisor/state.update_state): a busy core delays the
# registration, never loses it. The /panic send itself keeps its own 2 s (plugin.py).
_HOST_WAIT_SEC = 6.0
_RETRY_SEC = 2.0  # the pause before an owner registration the core did not settle is asked again (a transport retry)
_REPLY_WAIT_SEC = 20.0  # one reply of the gate, beside the poll loop: no longer than one Telegram send may take
_SETTLED = ("bound", "same", "other")
_TEXTS = Index("pairing", {
    "greeting": "✅ Connected: this chat now talks with Ouroboros. /menu opens the controls; /panic is the emergency stop.",
    "link_invalid": "This link is no longer valid. Open the Telegram card in Ouroboros for a fresh QR code.",
    "link_blocked": "This bot's emergency stop belongs to another Telegram account. Its owner can move it to this "
                    "chat only by resetting Ouroboros.",
    "link_retry": "Ouroboros could not confirm this connection just now. Open the link again in a moment.",
}, "the Telegram bridge's pairing lines: the greeting of a connected owner, a link that no longer binds, a link "
   "refused because another account holds the emergency stop, a link to open again")


class _Load:
    """What one load of the bridge knows about its owner beyond its settings (nothing of it is saved)."""

    def __init__(self) -> None:
        self.identity = ("", "")  # (digest of the token its getMe answered for, the bot's username)
        self.settled: set = set()  # owner chats whose registration the core settled: bound, same or other
        self.failing: set = set()  # owner chats whose failed registration was logged, until it settles
        self.blocked = ""  # the digest of the waiting code whose pairing another account's emergency stop refused
        self.replies: set = set()  # the gate's replies in flight beside the poll loop


_LOADS: "weakref.WeakKeyDictionary[Any, _Load]" = weakref.WeakKeyDictionary()


def _load(api) -> _Load:
    try:
        load = _LOADS.get(api)
        if load is None:
            load = _LOADS[api] = _Load()
        return load
    except TypeError:  # an API stand-in without weak references keeps nothing: each update asks again
        return _Load()


def _settings(api) -> Optional[dict]:
    """The skill's own settings, or None while they cannot be read (nothing binds then)."""
    try:
        return load_settings(pathlib.Path(api.get_state_dir()))
    except TelegramSettingsError:
        return None


def waiting_code(api) -> bool:
    """Whether a pairing code waits for its link (then only that link binds an owner)."""
    return bool(pairing.pending_code(_settings(api)))


def _check(client) -> None:
    """Only while the client's token is still the one in force: an update polled for a replaced token binds nothing
    and stays unconfirmed (TelegramBindingChanged)."""
    check = getattr(client, "check_binding", None)
    if callable(check):
        check()


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
    owner = bool(pinned_chat) and str(chat_id) == pinned_chat == str(user_id)
    private = str(chat.get("type") or "") == "private" and chat_id == user_id
    start = None if callback else pairing.start_command(message.get("text") or message.get("caption"))
    if start is not None and private:
        return await _start(api, client, chat_id, start, pinned_chat if owner else "", lang, post, message)
    if owner:
        return pinned_chat, False
    if not pinned_chat and private and _first_writer(api, client, chat_id, None):
        return str(chat_id), False
    if callback:
        with contextlib.suppress(Exception):  # a refused press is told once, never retried
            await client.answer_callback_query(str(callback.get("id") or ""),
                                               text=_LOCALIZED_TEXTS[lang]["not_authorized"])
    return None, False


async def _start(api, client, chat_id: int, payload: str, owner: str, lang: str, post,
                 message: dict) -> tuple[Optional[str], bool]:
    """A private chat's ``/start`` (``owner``: the owner chat when it is the owner's, else ""): always answered here."""
    settings = _settings(api)
    if settings is None:
        return owner or None, True  # nothing binds while the settings cannot be read; the next /start asks again
    digest = pairing.matching_digest(settings, payload)
    if digest:
        paired = await _pair(api, client, chat_id, digest, lang, post)
        if paired:
            _owner_message_seen(api, client.bot_id, chat_id, int(message.get("message_id") or 0))
            return paired, True
        return owner or None, True
    waiting, bound_now = bool(pairing.pending_code(settings)), False
    if not owner and not pairing.owner_chat(settings) and not waiting:
        if not _first_writer(api, client, chat_id, payload):
            return None, True
        owner, bound_now = str(chat_id), True
    if owner:  # greeted; an owner's payload that bound nothing is told so, never "Connected"
        _owner_message_seen(api, client.bot_id, chat_id, int(message.get("message_id") or 0))
        _reply(api, client, chat_id, "link_invalid" if payload and not bound_now else "greeting", lang)
        return owner, True
    if waiting:  # someone else, while a code waits: told once that this link does not bind
        _reply(api, client, chat_id, "link_invalid", lang)
    return None, True  # with an owner bound and no code waiting, someone else's /start gets no answer


def _first_writer(api, client, chat_id: int, payload: Optional[str]) -> bool:
    """The first positive private chat binds while no code waits, decided inside the settings write (a code saved
    meanwhile binds it nothing); a code-shaped ``/start`` payload that bound it consumes that code, should it arrive
    later. Unreadable or busy settings bind nothing; the next update asks again."""
    _check(client)
    try:
        saved = merge_settings(pathlib.Path(api.get_state_dir()),
                               functools.partial(pairing.first_written, chat_id=chat_id, payload=payload))
    except TelegramSettingsError:
        return False
    return pairing.owner_chat(saved) == chat_id


async def _pair(api, client, chat_id: int, digest: str, lang: str, post) -> Optional[str]:
    """The waiting code's ``/start``: the core first, then one settings write. The new owner's chat, or None when
    nothing changed (the scanner is told why)."""
    _check(client)
    outcome = await ensure_core_owner(api, chat_id, post, force=True)
    if outcome == "other":
        _load(api).blocked = digest
        _reply(api, client, chat_id, "link_blocked", lang)
        return None
    if outcome not in ("bound", "same"):
        _reply(api, client, chat_id, "link_retry", lang)
        return None
    _check(client)
    try:
        merge_settings(pathlib.Path(api.get_state_dir()),
                       functools.partial(pairing.paired, chat_id=chat_id, digest=digest))
    except TelegramSettingsError as exc:
        api.log("warning", f"Telegram could not save the owner its pairing link bound ({exc}); opening the link "
                           "again finishes it.")
        _reply(api, client, chat_id, "link_retry", lang)
        return None
    _load(api).blocked = ""
    api.log("info", "Telegram owner bound by the pairing link.")
    _reply(api, client, chat_id, "greeting", lang)
    return str(chat_id)


def _reply(api, client, chat_id: int, key: str, lang: str) -> None:
    """One line of the gate, sent beside the poll loop with its own time limit; a failed send is logged, never
    retried, and changes nothing the gate decided."""
    async def send() -> None:
        try:
            if key == "greeting":
                await asyncio.wait_for(greet(api, client, chat_id, lang), _REPLY_WAIT_SEC)
            else:
                await asyncio.wait_for(client.send_message(chat_id, _TEXTS[lang][key]), _REPLY_WAIT_SEC)
        except Exception as exc:  # noqa: BLE001 — a reply is never worth more than one line
            api.log("warning", f"Telegram could not send its pairing reply ({type(exc).__name__}).")

    replies = _load(api).replies
    task = asyncio.get_running_loop().create_task(send())
    replies.add(task)
    task.add_done_callback(replies.discard)


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


async def ensure_core_owner(api, chat_id: int, post, *, force: bool = False) -> str:
    """Ask the core to make ``chat_id`` its external command owner (``POST /owner/external``), once per load unless
    ``force`` (a pairing asks each time). "bound", "same" or "other" (another pair holds the slot until owner Reset)
    settle the chat for this load; "unknown" or "failed" while the core cannot answer, logged once until it can.
    "settled" when this load had already settled it."""
    load = _load(api)
    if not force and chat_id in load.settled:
        return "settled"
    try:
        status, body = await asyncio.wait_for(
            post(api, "/owner/external", {"user_id": chat_id, "chat_id": chat_id}), _HOST_WAIT_SEC)
    except Exception as exc:  # noqa: BLE001 — one line per episode; the caller asks again
        outcome, cause = "failed", type(exc).__name__
    else:
        outcome = str(body.get("owner") or "") if status < 400 else ""
        cause = f"host answered {status} {outcome or 'nothing'}"
    if outcome in _SETTLED:
        load.failing.discard(chat_id)
        if outcome == "bound":
            api.log("info", "Telegram owner registered with the emergency stop: the first /panic from this chat stops.")
        elif outcome == "other" and chat_id not in load.settled:
            api.log("warning", "The core's emergency stop is bound to another chat than this Telegram owner: /panic "
                               "from here is refused until Ouroboros is reset.")
        load.settled.add(chat_id)
        return outcome
    if chat_id not in load.failing:
        load.failing.add(chat_id)
        api.log("warning", f"Telegram could not register its owner with the emergency stop yet ({cause}); it "
                           "asks again.")
    return outcome or "failed"


async def keep_owner_registered(api, post) -> None:
    """From the bridge's load on: an owner the core does not know yet (bound before 1.5.0, or by a first writer) is
    registered, and asked again after a pause while the core cannot settle it, so the first /panic finds the
    emergency stop ready. A settled owner costs one read of the skill's settings per pause. It ends with the load
    (unload, Off or a revoked grant close the token binding), also when a cancellation was lost in ``wait_for``."""
    binding = binding_for(api)
    while binding.state != "closed":
        chat = pairing.owner_chat(_settings(api))
        if chat:
            await ensure_core_owner(api, chat, post)
        await asyncio.sleep(_RETRY_SEC)


async def dispatch_panic(api, chat_id: int, send, post) -> None:
    """Send the owner's /panic at once (``send()``). When the emergency door refused it (HTTP 503) before this load
    settled the owner's registration, the door had not seen the owner yet and nothing stopped: register the owner
    and send it once more. An owner the core has settled is never asked again, so its /panic is one request."""
    try:
        await send()
        return
    except HostRefused as exc:
        if exc.status != 503 or chat_id in _load(api).settled:
            raise
    if await ensure_core_owner(api, chat_id, post) not in ("bound", "same"):
        raise HostRefused(503)
    await send()


def note_identity(api, token: str, reply: Any) -> None:
    """Keep the username of the validating ``getMe`` only when its ``result.id`` is the bot ``token`` names; any
    other answer forgets the name. The status route shows it only while that same token is in force."""
    result = reply.get("result") if isinstance(reply, dict) else None
    bot = result.get("id") if isinstance(result, dict) else None
    username = result.get("username") if isinstance(result, dict) else None
    named = bot_id(token)
    load = _load(api)
    if type(bot) is int and named and str(bot) == named and pairing.username_ok(username):
        load.identity = (pairing.code_digest(token), username)
    else:
        load.identity = ("", "")


def bot_username(api) -> Optional[str]:
    """The bot's username, only for the token in force (a replaced token, even of the same bot, shows none until its
    own getMe answered)."""
    digest, username = _load(api).identity
    token = accepted_token(api)
    return username if digest and token and pairing.same_digest(digest, pairing.code_digest(token)) else None


def register_pairing(api, post) -> None:
    """``pairing/status`` (GET), served as ``miniapp/status`` is: the bot's username (null until the bridge has
    validated its token), whether an owner is bound, whether a pairing code waits, and why its link cannot bind
    (``blocked``: ``"emergency_stop_owner"`` once its pairing was refused because another account holds the
    emergency stop; else null). Never the code or the token; 409 while the skill's settings cannot be read. Also
    starts the owner's registration with the emergency stop (``keep_owner_registered``)."""
    def facts() -> Optional[dict]:
        settings = _settings(api)
        if settings is None:
            return None
        code, blocked = pairing.pending_code(settings), _load(api).blocked
        return {"bot_username": bot_username(api), "owner_bound": bool(pairing.owner_chat(settings)),
                "pending": bool(code), "blocked": "emergency_stop_owner" if code and pairing.same_digest(
                    pairing.code_digest(code), blocked) else None}

    async def status(_request):
        answer = await asyncio.to_thread(facts)
        if answer is None:
            return JSONResponse({"ok": False, "message": "Telegram settings cannot be read."}, status_code=409)
        return JSONResponse(answer)

    api.register_route("pairing/status", handler=status, methods=("GET",))
    api.register_supervised_task("owner_registration", functools.partial(keep_owner_registered, api, post),
                                 restart_policy="on_failure", max_restarts=10)


__all__ = [
    "bot_username", "dispatch_panic", "ensure_core_owner", "greet", "keep_owner_registered", "note_identity",
    "owner_gate", "register_pairing", "waiting_code",
]
