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
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.offset = int(data.get("offset") or 0)
            # A file without a base predates server binding: it was the cloud's.
            self.base = str(data.get("base") or CLOUD_BASE)
            self.bot = str(data.get("bot") or "")
            self._handled = {str(key): [str(item) for item in value]
                             for key, value in dict(data.get("handled") or {}).items()}
        except Exception:
            pass
        self.follow(base, bot)

    def follow(self, base: str, bot: str = "") -> None:
        """Bind to ``base`` and ``bot``; a different server or bot starts from its own oldest update."""
        if bot and self.bot and bot != self.bot:
            self.base, self.offset, self._handled = base, 0, {}
        elif base != self.base:
            self.base, self.offset = base, 0
        self.bot = bot or self.bot

    def is_replay(self, update: dict) -> bool:
        key = update_key(update)
        return bool(key) and any(key in keys for keys in self._handled.values())

    def save(self, offset: int, handled: list[str]) -> None:
        self.offset = offset
        self._handled.pop(self.base, None)
        self._handled[self.base] = [key for key in handled if key]
        self._handled = dict(list(self._handled.items())[-_RECENT_SERVERS:])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"offset": offset, "base": self.base, "bot": self.bot, "handled": self._handled}),
                       encoding="utf-8")
        tmp.replace(self.path)
