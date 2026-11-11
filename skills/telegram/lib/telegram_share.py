"""Share from the Mini App (docs/DESIGN.md "Replies"): the bot prepares the exact words the owner pressed Share… for
as one Telegram message (Bot API savePreparedInlineMessage), and the Mini App's shareMessage opens Telegram's own
chat picker for it. Only the owner's Mini App session reaches this route: the sidecar authenticates it, checks its
origin and marks what it forwards (scripts/sidecar.py). The body is the text alone; the user is the bound owner and
the client is the bridge's own (its token generation, proxy and server). Nothing is sent from here: the owner sends
the prepared message in Telegram, or not. The words are never logged."""
from __future__ import annotations

import json
import secrets
from pathlib import Path

from starlette.responses import JSONResponse

from .telegram_api import (
    _TELEGRAM_TEXT_LIMIT,
    TelegramBindingChanged,
    TelegramRequestRejected,
    TelegramTokenMissing,
    TelegramTransportError,
    _u16len,
)
from ..scripts.telegram_settings import MINIAPP_MARKER_HEADER, TelegramSettingsError, owner_chat_id

_BODY_LIMIT = 64 * 1024  # the text's own bound in UTF-8, with room for JSON escapes
_TITLE_LIMIT = 64
_LOOPBACK = {"127.0.0.1", "::1"}


def _refuse(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": message, "reason_code": code}, status_code=status)


def _from_miniapp(request) -> bool:
    """Forwarded by the sidecar for the owner's Mini App session: its mark, from this computer."""
    headers = getattr(request, "headers", None) or {}
    host = str(getattr(getattr(request, "client", None), "host", "") or "").strip()
    return str(headers.get(MINIAPP_MARKER_HEADER, "")).strip() == "1" and host in _LOOPBACK


def _title(text: str) -> str:
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return line if len(line) <= _TITLE_LIMIT else line[: _TITLE_LIMIT - 1].rstrip() + "…"


def prepared_article(text: str) -> dict:
    """The one inline result: the words as plain text, as the preview showed them, without a link preview."""
    return {
        "type": "article",
        "id": secrets.token_hex(16),
        "title": _title(text),
        "input_message_content": {"message_text": text, "link_preview_options": {"is_disabled": True}},
    }


def make_prepare_share(api, client_factory):
    """POST {"text"} → {"id", "expiration_date"}: Telegram's PreparedInlineMessage for the owner, or a refusal with
    its `reason_code`. `client_factory(api)` returns the bridge's `(TelegramClient, generation)`."""
    async def prepare_share(request):
        if not _from_miniapp(request):
            return _refuse(403, "miniapp_only", "Sharing to Telegram is prepared only for the owner's Mini App.")
        raw = await request.body()
        if len(raw) > _BODY_LIMIT:
            return _refuse(413, "text_too_long", f"Telegram takes up to {_TELEGRAM_TEXT_LIMIT} characters.")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return _refuse(400, "invalid_request", "The request is not JSON.")
        if not isinstance(body, dict) or set(body) != {"text"} or not isinstance(body["text"], str):
            return _refuse(400, "invalid_request", "Send only the text to share.")
        text = body["text"]
        if not text.strip():
            return _refuse(400, "empty_text", "There is nothing to share.")
        if _u16len(text) > _TELEGRAM_TEXT_LIMIT:
            return _refuse(413, "text_too_long", f"Telegram takes up to {_TELEGRAM_TEXT_LIMIT} characters.")
        try:
            owner = owner_chat_id(Path(api.get_state_dir()))
        except TelegramSettingsError as exc:
            return _refuse(503, "settings_invalid", str(exc))
        if not owner:
            return _refuse(409, "owner_not_bound", "No Telegram owner is bound to this bot yet.")
        try:
            client, _generation = client_factory(api)
            payload = await client.call("savePreparedInlineMessage", data={
                "user_id": owner,
                "result": json.dumps(prepared_article(text), ensure_ascii=False),
                "allow_user_chats": True, "allow_group_chats": True, "allow_channel_chats": True,
                "allow_bot_chats": False,
            })
        except TelegramTokenMissing:
            return _refuse(503, "token_missing", "The Telegram bot token is not set.")
        except TelegramBindingChanged:
            return _refuse(409, "bot_changed", "The Telegram bot changed. Press Share… again.")
        except TelegramRequestRejected as exc:
            return _refuse(502, "telegram_rejected", str(exc))
        except TelegramTransportError as exc:
            return _refuse(504, "telegram_unreachable", str(exc))
        result = payload.get("result") if isinstance(payload, dict) else None
        message_id = result.get("id") if isinstance(result, dict) else None
        expires = result.get("expiration_date") if isinstance(result, dict) else None
        if not (isinstance(message_id, str) and message_id) or not isinstance(expires, int) or isinstance(expires, bool):
            return _refuse(502, "invalid_answer", "Telegram returned no prepared message.")
        return JSONResponse({"id": message_id, "expiration_date": expires})
    return prepare_share
