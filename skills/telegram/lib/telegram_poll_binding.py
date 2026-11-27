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
import json
from pathlib import Path

from ..scripts.telegram_settings import telegram_api_base, telegram_files_dir, validate_telegram_api_base

CLOUD_BASE = validate_telegram_api_base("")
_RECENT_SERVERS = 4
_PARK_CHECK_SEC = 5.0
_MAX_GROUP_WRITES = 4  # bound group disk waits before the next Bot API poll


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
    return f"m:{chat_id}:{message['message_id']}" if chat_id and message.get("message_id") else ""


class PollCursor:
    def __init__(self, path: Path, base: str, bot: str = ""):
        self.path, self.base, self.bot, self.offset = Path(path), CLOUD_BASE, "", 0
        self._handled: dict[str, list[str]] = {}
        self._early: dict[str, dict[str, int]] = {}
        self._held: dict[str, dict] = {}
        self._group_periods: dict[str, dict[str, dict | None]] = {}
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
            self._group_periods = {base: {uid: period for uid, period in items.items()
                                          if uid.isdecimal() and (period is None or isinstance(period, dict))}
                                   for base, items in dict(data.get("group_periods") or {}).items()
                                   if isinstance(items, dict)}
        except Exception:
            pass
        self.follow(base, bot)

    def follow(self, base: str, bot: str = "") -> None:
        """Bind to ``base`` and ``bot``; a different server or bot starts from its own oldest update."""
        if bot and self.bot and bot != self.bot:
            self.base, self.offset, self._handled = base, 0, {}
            self._early, self._held, self._group_periods = {}, {}, {}
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

    def defer_group(self, update: dict, period: dict | None) -> bool:
        """Keep the original observation with an unconfirmed update, including None (no consent)."""
        uid = update.get("update_id")
        if type(uid) is int and uid > 0:
            pending = self._group_periods.setdefault(self.base, {})
            if str(uid) not in pending:
                pending[str(uid)] = period
                return True
        return False

    def group_period(self, update: dict, current: dict | None) -> dict | None:
        return self._group_periods.get(self.base, {}).get(str(update.get("update_id")), current)

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
        if self.base in self._group_periods:
            self._group_periods[self.base] = {uid: period for uid, period in self._group_periods[self.base].items()
                                              if int(uid) >= offset}
        self._group_periods = {base: items for base, items in self._group_periods.items()
                               if items and base in self._handled}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"offset": offset, "base": self.base, "bot": self.bot, "handled": self._handled,
                                   **({"early": self._early} if self._early else {}),
                                   **({"held": self._held} if self._held else {}),
                                   **({"group_periods": self._group_periods} if self._group_periods else {})}),
                       encoding="utf-8")
        tmp.replace(self.path)


async def _owner_panic_behind_groups(api, client, updates, cursor, pinned_chat, lang, *, authorize, is_panic,
                                     on_group) -> bool:
    """Whether a private Panic of the bound owner, not handled before, waits in this batch behind a group update.
    The owner authorizer decides who sent it; with a bound owner it only compares ids (no binding, no reply), and
    without one no group consent can exist, so group work delays nothing."""
    from .telegram_group_feed import group_chat

    if on_group is None or not pinned_chat:
        return False
    behind_group = False
    for update in updates:
        if group_chat(update) is not None:
            behind_group = True
        elif behind_group and is_panic(update) and not cursor.is_replay(update) and await authorize(
                api, client, update, pinned_chat, lang):
            return True
    return False


async def handle_poll_batch(api, client, updates, cursor, binding, generation, pinned_chat, max_updates, command_mode,
                            lang, *, authorize, is_panic, on_message, on_callback, on_group=None,
                            group_period=None) -> tuple[bool, str]:
    """Confirm updates in order. An already received Panic alone may pass a limit or an acceptance gap.

    ``on_group`` takes a group or supergroup update (lib/telegram_group_feed.py) before the owner authorizer, which
    never sees it: it is never an owner control, Panic included, and in the early phase it waits like any other.

    An owner's private Panic not handled before never waits behind group disk work in its batch: every group update
    of that batch is left for its next delivery and the offset stops at the first one, so no later offset confirms
    it; private updates keep their order and, once past that point, are remembered as early replays
    (``remember_early``). The next delivery then gives the groups their turn and repeats no control: the handled
    Panic is a replay there, which never holds a batch again. Groups held by the write limit, the ordinary limit or
    that priority keep their first observed period in the normal cursor save, with no write ahead of Panic."""
    from .telegram_api import TelegramBindingChanged, _LOCALIZED_TEXTS
    from .telegram_group_feed import group_chat

    offset, handled, deferred = cursor.offset, [], False
    groups_wait = await _owner_panic_behind_groups(api, client, updates, cursor, pinned_chat, lang,
                                                   authorize=authorize, is_panic=is_panic, on_group=on_group)
    frozen = False  # a group update of this batch was left: no later item may confirm past it
    group_writes, group_deferred = 0, False
    for index, update in enumerate(updates):
        if not binding.current(generation):
            break
        group = on_group is not None and group_chat(update) is not None
        replay = cursor.is_replay(update)
        key = update_key(update)
        if group and (groups_wait or group_writes >= _MAX_GROUP_WRITES):
            group_deferred |= cursor.defer_group(update, group_period)
            frozen = True
            continue
        held = deferred or index >= max_updates  # the existing early phase: only a received Panic passes it
        early = held or frozen
        if held and (group or not is_panic(update)):
            if group:
                group_deferred |= cursor.defer_group(update, group_period)
            if replay:
                handled.append(key)
            continue
        authorized_chat, confirmed, target = None, (offset, len(handled)), cursor.held_target(update)

        def confirm():
            nonlocal confirmed
            confirmed = None  # a local or host effect already happened

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
            if group:
                group_writes += cursor.group_period(update, group_period) is not None
                await on_group(update)  # kept or a counted gap: confirmed either way, the owner never waits
                cursor.retire_target(key)
                continue
            authorized_chat = await authorize(api, client, update, pinned_chat, lang)
            if not authorized_chat:
                cursor.retire_target(key)
                if early:
                    cursor.remember_early(update)  # its refusal is not answered again at the next delivery
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
                if not held and target is not None and update_id > 0:  # a reply's found card, past a frozen group too
                    cursor.hold_target(update, target)
                api.log("info", "The host did not record a Telegram update; it will be delivered again.")
                continue
            api.log("warning", "Telegram update was not accepted after its local effect; it is not retried.")
        except Exception as exc:
            api.log("warning", f"Telegram update processing failed ({type(exc).__name__}).")
            if authorized_chat:
                try:
                    await client.send_message(int(authorized_chat), _LOCALIZED_TEXTS[lang]["update_failed"])
                except Exception:
                    api.log("warning", "Telegram update failure notice could not be delivered.")
        cursor.retire_target(key)
        if early:
            cursor.remember_early(update)
    if updates and (handled or deferred or offset != cursor.offset or group_deferred):
        cursor.save(offset, handled)
    return deferred, lang
