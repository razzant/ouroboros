"""The poller's binding to a Bot API server: which server, and where it is in its queue.

Every server keeps its own update queue and numbering, so an offset is meaningful only
on the server it came from: carried to another server it could confirm (delete) that
server's unseen updates or skip past them. A cursor saved for another base therefore
starts the current server at offset 0. The old server is never contacted: the owner may
have called logOut or close there, and any request would launch the bot on it again.

A server answering offset 0 may still return the last batch this bridge handled there
but had not yet confirmed: after a return to it, or when two spellings of the base reach
one server (127.0.0.1 and localhost). The cursor keeps the identities of the last handled
batch of each recent server (chat and message ID, callback query ID: the same on every
server, never reused), so such a replay is confirmed without being handled again.

A server also keeps one queue per bot, so the cursor names the bot it belongs to: the
numeric ID that opens every token (``<bot id>:<secret>``), never the secret part. Another
bot starts from its oldest update with no replay history, since its message IDs may
repeat the old bot's; a rotated token of the same bot keeps its position. A file saved
before the cursor named its bot keeps its offset: which bot it came from is unknown.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from ..scripts.telegram_settings import (
    TelegramSettingsError,
    telegram_api_base,
    telegram_files_dir,
    validate_telegram_api_base,
)

CLOUD_BASE = validate_telegram_api_base("")
_RECENT_SERVERS = 4
_PARK_CHECK_SEC = 5.0


class InjectNotAccepted(RuntimeError):
    """The host proves this attempt did not enter the dialogue store."""


def saved_server(api) -> tuple[str, str | None]:
    """The saved Bot API server origin and its file store, read fresh for every client."""
    state_dir = Path(api.get_state_dir())
    return telegram_api_base(state_dir), telegram_files_dir(state_dir)


def server_fingerprint(api, token: str | None = None):
    """The token and saved server a client is built from: local reads, no request.

    ``token`` is the bridge's accepted token (lib/telegram_token.py); without it the saved one is
    read once. None without a token or a readable saved server: neither says the settings changed.
    """
    try:
        if token is None:
            from .telegram_token import TOKEN_KEY, read_token_state

            token = str((read_token_state(api).get("values") or {}).get(TOKEN_KEY) or "").strip()
        return (token, saved_server(api)) if token else None
    except Exception:
        return None


def bot_id(token: str) -> str:
    """The bot's numeric ID from its token, or "" when the token has none."""
    head = str(token or "").split(":", 1)[0].strip()
    return head if head.isdigit() else ""


_NO_READING = {  # why a parked poller has nothing to compare; logged when the reason changes
    "missing": ("info", "Telegram bot token was cleared while polling waits; still waiting for a saved one."),
    "unreadable": ("warning", "Telegram settings could not be read while polling waits; still waiting."),
}
_TOKENLESS = {"missing": "missing", "unknown": "unreadable"}  # a binding holding no token, by its state


async def park_until_settings_change(api, fingerprint, binding=None) -> None:
    """After a permanent rejection ask that origin nothing more until a Save changes it.

    Parking instead of raising keeps the supervised poller alive: its restarts would run
    out, and a corrected Save would then find no poller. ``binding`` is the bridge's token
    binding (lib/telegram_token.py): its accepted token is compared, and its state tells a
    cleared token from settings that cannot be read. Neither says the settings changed, so
    only a successful read that differs from ``fingerprint`` ends the wait. A line is logged
    when the reason changes; the same reason again, even after an unchanged reading, stays quiet.
    """
    said = ""
    while True:
        token = None if binding is None else binding.snapshot()[0]
        current = server_fingerprint(api, token)
        if current is not None and current != fingerprint:
            break
        # No token in force says why by the binding's state (a closing one says nothing); a token
        # without a reading means the saved settings could not be read.
        reason = "" if current is not None else _TOKENLESS.get(binding.state, "") if token == "" else "unreadable"
        if reason and reason != said:
            said = reason
            api.log(*_NO_READING[reason])
        await asyncio.sleep(_PARK_CHECK_SEC)
    api.log("info", "Telegram settings changed after a rejection; polling resumes.")


def update_key(update: dict) -> str:
    """One Telegram event's identity on any server; empty when it carries none."""
    callback = update.get("callback_query") or {}
    if callback.get("id"):
        return f"cb:{callback['id']}"
    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    if chat_id and message.get("message_id"):
        return f"m:{chat_id}:{message['message_id']}"
    for field, prefix in (("edited_message", "em"), ("channel_post", "cp"), ("edited_channel_post", "ecp")):
        item = update.get(field) or {}
        cid, mid = (item.get("chat") or {}).get("id"), item.get("message_id")
        if cid and mid:
            if field.startswith("edited_"):
                digest = hashlib.sha256(
                    json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8", "surrogatepass")
                ).hexdigest()[:16]
                return f"{prefix}:{cid}:{mid}:{digest}"
            return f"{prefix}:{cid}:{mid}"
    reaction = update.get("message_reaction") or {}
    if (reaction.get("chat") or {}).get("id") and reaction.get("message_id"):
        actor = (reaction.get("user") or reaction.get("actor_chat") or {}).get("id") or ""
        digest = hashlib.sha256(
            json.dumps(reaction, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8", "surrogatepass")
        ).hexdigest()[:16]
        return f"mr:{reaction['chat']['id']}:{reaction['message_id']}:{reaction.get('date') or 0}:{actor}:{digest}"
    rcount = update.get("message_reaction_count") or {}
    if (rcount.get("chat") or {}).get("id") and rcount.get("message_id"):
        digest = hashlib.sha256(
            json.dumps(rcount, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8", "surrogatepass")
        ).hexdigest()[:16]
        return f"mrc:{rcount['chat']['id']}:{rcount['message_id']}:{rcount.get('date') or 0}:{digest}"
    member = update.get("my_chat_member") or {}
    if (member.get("chat") or {}).get("id"):
        status = (member.get("new_chat_member") or {}).get("status") or ""
        digest = hashlib.sha256(
            json.dumps(member, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8", "surrogatepass")
        ).hexdigest()[:16]
        return f"mcm:{member['chat']['id']}:{member.get('date') or 0}:{status}:{digest}"
    return ""


class PollCursor:
    def __init__(self, path: Path, base: str, bot: str = ""):
        self.path, self.base, self.bot, self.offset = Path(path), CLOUD_BASE, "", 0
        self._handled: dict[str, list[str]] = {}
        self._early: dict[str, dict[str, int]] = {}
        self._held: dict[str, dict] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.offset = int(data.get("offset") or 0)
            # A file without a base predates server binding: it was the cloud's.
            self.base = str(data.get("base") or CLOUD_BASE)
            self.bot = str(data.get("bot") or "")
            self._handled = {str(key): [str(item) for item in value]
                             for key, value in dict(data.get("handled") or {}).items()}
            self._early = {base: {key: int(uid) for key, uid in items.items()}
                           for base, items in dict(data.get("early") or {}).items()}
            self._held = {base: item for base, item in dict(data.get("held") or {}).items()
                          if isinstance(item.get("target"), dict)}
        except Exception:
            pass
        self.follow(base, bot)

    def follow(self, base: str, bot: str = "") -> None:
        """Bind to ``base`` and ``bot``; a different server or bot starts from its own oldest update."""
        if bot and self.bot and bot != self.bot:
            self.base, self.offset, self._handled = base, 0, {}
            self._early, self._held = {}, {}
        elif base != self.base:
            self.base, self.offset = base, 0
        self.bot = bot or self.bot

    def is_replay(self, update: dict) -> bool:
        key = update_key(update)
        return bool(key) and (any(key in keys for keys in self._handled.values())
                              or any(key in keys for keys in self._early.values()))

    def remember_early(self, update: dict) -> None:
        if (key := update_key(update)) and (uid := int(update.get("update_id") or 0)) > 0:
            self._early.setdefault(self.base, {})[key] = uid

    def held_target(self, update: dict):
        key = update_key(update)
        return next((held["target"] for held in self._held.values() if key and held.get("key") == key), None)

    def hold_target(self, update: dict, target: dict) -> None:
        self._held[self.base] = {"key": update_key(update), "update_id": update["update_id"], "target": target}

    def retire_target(self, key: str) -> None:
        if key:
            self._held = {base: held for base, held in self._held.items() if held.get("key") != key}

    def save(self, offset: int, handled: list[str]) -> None:
        self.offset = offset
        self._handled.pop(self.base, None)
        self._handled[self.base] = [key for key in handled if key]
        self._handled = dict(list(self._handled.items())[-_RECENT_SERVERS:])
        if self.base in self._early:
            self._early[self.base] = {key: uid for key, uid in self._early[self.base].items() if uid >= offset}
        self._early = {base: items for base, items in self._early.items() if items and base in self._handled}
        if self._held.get(self.base, {}).get("update_id", offset) < offset:
            self._held.pop(self.base, None)
        self._held = {base: item for base, item in self._held.items() if base in self._handled}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"offset": offset, "base": self.base, "bot": self.bot, "handled": self._handled,
                                   **({"early": self._early} if self._early else {}),
                                   **({"held": self._held} if self._held else {})}), encoding="utf-8")
        tmp.replace(self.path)


async def handle_poll_batch(api, client, updates, cursor, binding, generation, pinned_chat, max_updates, command_mode,
                            lang, *, authorize, is_panic, on_message, on_callback, on_non_owner=None) -> tuple[bool, str]:
    """Confirm updates in order. An already received owner Panic runs without waiting behind earlier slow updates."""
    from .telegram_api import TelegramBindingChanged, _LOCALIZED_TEXTS

    offset, handled, deferred = cursor.offset, [], False
    panic_slots = (
        [i for i, u in enumerate(updates) if i > 0 and is_panic(u) and not cursor.is_replay(u)]
        if pinned_chat else []
    )
    panic_results: dict[int, tuple[str, Any]] = {}

    async def _run_early_panics() -> None:
        for idx in panic_slots:
            if not binding.current(generation):
                panic_results[idx] = ("binding_changed", None)
                break
            upd = updates[idx]
            uid = int(upd.get("update_id") or 0)
            try:
                auth_chat = await authorize(api, client, upd, pinned_chat, lang)
                if not auth_chat:
                    panic_results[idx] = ("unauthorized", None)
                    continue
                await on_message(api, client, upd, uid, command_mode, lang, reply_target=lambda read: read())
                panic_results[idx] = ("ok", auth_chat)
            except TelegramBindingChanged as exc:
                panic_results[idx] = ("binding_changed", exc)
                break
            except InjectNotAccepted as exc:
                panic_results[idx] = ("not_accepted", exc)
            except TelegramSettingsError as exc:
                panic_results[idx] = ("settings_error", exc)
                break
            except Exception as exc:
                panic_results[idx] = ("error", exc)

    panic_task = asyncio.create_task(_run_early_panics()) if panic_slots else None
    panic_consumed: set[int] = set()
    owner_deferred = False
    owner_handled = 0
    unwinding_cancel = False
    try:
        for index, update in enumerate(updates):
            if not binding.current(generation):
                break
            early = deferred or index >= max_updates
            replay = cursor.is_replay(update)
            key = update_key(update)
            presence_only_defer = bool(pinned_chat) and deferred and not owner_deferred
            if early and not (bool(pinned_chat) and (is_panic(update) or (presence_only_defer and owner_handled < max_updates))):
                if replay:
                    handled.append(key)
                continue
            if panic_task is not None and index in panic_slots:
                await panic_task
                status, detail = panic_results.get(index, ("unauthorized", None))
                if status != "unauthorized":
                    panic_consumed.add(index)
                    update_id = int(update.get("update_id") or 0)
                    if status == "binding_changed":
                        api.log("info", "Telegram bot token changed while an update was handled; it stays unconfirmed.")
                        break
                    if status == "not_accepted":
                        deferred = owner_deferred = True
                        api.log("info", "The host did not record a Telegram update; it will be delivered again.")
                        continue
                    if status == "settings_error":
                        raise detail
                    if update_id > 0 and not early:
                        offset = update_id + 1
                    handled.append(key)
                    if status == "error":
                        api.log("warning", f"Telegram update processing failed ({type(detail).__name__}).")
                    cursor.retire_target(key)
                    if early:
                        cursor.remember_early(update)
                    continue
            authorized_chat, confirmed, target = None, (offset, len(handled)), cursor.held_target(update)

            def confirm():
                nonlocal confirmed
                confirmed = None  # a local or host effect already happened
                if early:
                    cursor.remember_early(update)

            def resolve_target(read):
                nonlocal target
                if target is None:
                    target = read()
                return target

            try:
                update_id = int(update.get("update_id") or 0)
                if update_id > 0 and not early:
                    offset = update_id + 1  # retain the existing recovery for a restarted server's numbering
                handled.append(key)
                if replay:
                    cursor.retire_target(key)
                    continue
                authorized_chat = await authorize(api, client, update, pinned_chat, lang)
                if not authorized_chat:
                    if early:
                        handled = handled[:confirmed[1]]
                        continue
                    if on_non_owner is not None:
                        await on_non_owner(api, client, update, pinned_chat)
                    cursor.retire_target(key)
                    continue
                pinned_chat = authorized_chat
                if update.get("callback_query"):
                    lang = await on_callback(api, client, update, pinned_chat, command_mode, lang, confirm)
                else:
                    await on_message(api, client, update, update_id, command_mode, lang, reply_target=resolve_target)
            except TelegramBindingChanged:
                if confirmed is not None:
                    offset, handled = confirmed[0], handled[:confirmed[1]]
                    api.log("info", "Telegram bot token changed while an update was handled; it stays unconfirmed.")
                else:
                    cursor.retire_target(key)
                    api.log("info", "Telegram bot token changed after an update was applied; its reply was not sent.")
                break
            except InjectNotAccepted:
                if confirmed is not None:
                    offset, handled = confirmed[0], handled[:confirmed[1]]
                    deferred = True
                    if authorized_chat:
                        owner_deferred = True
                    if (not early or presence_only_defer) and target is not None and update_id > 0:
                        cursor.hold_target(update, target)
                    api.log("info", "The host did not record a Telegram update; it will be delivered again.")
                    continue
                api.log("warning", "Telegram update was not accepted after its local effect; it is not retried.")
            except TelegramSettingsError:
                if confirmed is not None:
                    offset, handled = confirmed[0], handled[:confirmed[1]]
                else:
                    cursor.retire_target(key)
                raise
            except Exception as exc:
                api.log("warning", f"Telegram update processing failed ({type(exc).__name__}).")
                if authorized_chat:
                    try:
                        await client.send_message(int(authorized_chat), _LOCALIZED_TEXTS[lang]["update_failed"])
                    except Exception:
                        api.log("warning", "Telegram update failure notice could not be delivered.")
            except BaseException:
                if confirmed is not None:
                    offset, handled = confirmed[0], handled[:confirmed[1]]
                else:
                    cursor.retire_target(key)
                raise
            if authorized_chat and not is_panic(update):
                owner_handled += 1
            cursor.retire_target(key)
            if early:
                cursor.remember_early(update)
    except asyncio.CancelledError:
        unwinding_cancel = True
        raise
    finally:
        cancelled_exc: BaseException | None = None
        if panic_task is not None:
            if not panic_task.done():
                if unwinding_cancel:
                    panic_task.cancel()
                    try:
                        await asyncio.gather(panic_task, return_exceptions=True)
                    except BaseException:
                        pass
                else:
                    try:
                        await asyncio.gather(panic_task, return_exceptions=True)
                    except asyncio.CancelledError as exc:
                        cancelled_exc = exc
                        if not panic_task.done():
                            panic_task.cancel()
                            try:
                                await asyncio.gather(panic_task, return_exceptions=True)
                            except BaseException:
                                pass
            for idx, (status, _detail) in list(panic_results.items()):
                if idx not in panic_consumed and status in {"ok", "error"}:
                    panic_consumed.add(idx)
                    upd = updates[idx]
                    k = update_key(upd)
                    if k:
                        handled.append(k)
                    cursor.remember_early(upd)
        if updates and (handled or deferred or offset != cursor.offset):
            if unwinding_cancel or cancelled_exc is not None:
                try:
                    cursor.save(offset, handled)
                except Exception as exc:
                    try:
                        api.log("warning", f"Telegram cursor save failed during cancellation ({type(exc).__name__}).")
                    except Exception:
                        pass
            else:
                cursor.save(offset, handled)
        if cancelled_exc is not None:
            raise cancelled_exc
    return deferred, lang
