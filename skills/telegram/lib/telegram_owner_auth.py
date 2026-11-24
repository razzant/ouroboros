"""Owner confirmation, live binding guards, and entity-aware /panic parsing for Telegram."""

from __future__ import annotations

import hmac
import json
import os
import pathlib
import re
import secrets
import time
from typing import Any, Dict, Optional

from ..scripts.telegram_settings import TelegramSettingsError, load_settings, merge_settings, telegram_api_base

_PAIRING_FILE = "owner_pairing.json"
_PAIRING_TTL_SEC = 600.0
_PAIRING_MAX_ATTEMPTS = 5
_MAIN_CMD_RE = re.compile(r"^/(?:main|m)(?:@[A-Za-z0-9_]+)?(?:\s+(.*))?$", re.IGNORECASE | re.DOTALL)
_PANIC_CMD_RE = re.compile(r"^/panic(?:@([A-Za-z0-9_]+))?(?:\s|$)", re.IGNORECASE)
_PAIR_CMD_RE = re.compile(r"^/(?:start|pair)(?:@[A-Za-z0-9_]+)?\s+(\S+)\s*$", re.IGNORECASE)
_LITERAL_ENTITY_TYPES = frozenset({
    "code",
    "pre",
    "blockquote",
    "expandable_blockquote",
    "url",
    "text_link",
    "email",
    "phone_number",
})


def _resolve_state_dir(state_dir: Any) -> pathlib.Path:
    getter = getattr(state_dir, "get_state_dir", None)
    return pathlib.Path(getter() if callable(getter) else state_dir)


def _pairing_path(state_dir: Any) -> pathlib.Path:
    return _resolve_state_dir(state_dir) / _PAIRING_FILE


def _write_private_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def clear_owner_pairing(state_dir: Any) -> None:
    try:
        _pairing_path(state_dir).unlink(missing_ok=True)
    except OSError:
        pass


def issue_owner_pairing_code(state_dir: Any, *, ttl_sec: float = _PAIRING_TTL_SEC, now: Optional[float] = None) -> Dict[str, Any]:
    """Issue a single-use, time-bounded owner confirmation code from Ouroboros (loopback-only)."""
    ts = time.time() if now is None else float(now)
    ttl = max(30.0, min(float(ttl_sec), 3600.0))
    code = secrets.token_urlsafe(24)
    record = {
        "code": code,
        "issued_at": ts,
        "expires_at": ts + ttl,
        "failed_attempts": 0,
    }
    _write_private_json(_pairing_path(state_dir), record)
    return {"code": code, "expires_at": record["expires_at"], "ttl_sec": ttl}


def redeem_owner_pairing_code(
    state_dir: Any, chat_id: int, text: str, *, now: Optional[float] = None
) -> bool:
    """Validate and atomically consume an Ouroboros-issued one-time pairing code from a private DM."""
    if int(chat_id or 0) <= 0:
        return False
    match = _PAIR_CMD_RE.match(str(text or "").strip())
    if not match:
        return False
    candidate = match.group(1)
    path = _pairing_path(state_dir)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        clear_owner_pairing(state_dir)
        return False
    if not isinstance(record, dict):
        clear_owner_pairing(state_dir)
        return False
    ts = time.time() if now is None else float(now)
    expected = str(record.get("code") or "")
    expires_at = float(record.get("expires_at") or 0.0)
    failed = int(record.get("failed_attempts") or 0)
    if not expected or ts > expires_at or failed >= _PAIRING_MAX_ATTEMPTS:
        clear_owner_pairing(state_dir)
        return False
    if not hmac.compare_digest(expected, candidate):
        failed += 1
        if failed >= _PAIRING_MAX_ATTEMPTS:
            clear_owner_pairing(state_dir)
        else:
            record["failed_attempts"] = failed
            try:
                _write_private_json(path, record)
            except OSError:
                clear_owner_pairing(state_dir)
        return False
    # Consume single-use token BEFORE persisting TELEGRAM_CHAT_ID so replay is impossible.
    clear_owner_pairing(state_dir)
    merge_settings(_resolve_state_dir(state_dir), {"TELEGRAM_CHAT_ID": str(int(chat_id))})
    return True


def live_owner_chat_id(state_dir: Any) -> int:
    """Current explicitly bound owner chat ID from settings, or 0 if unbound/invalid."""
    try:
        raw = str(load_settings(_resolve_state_dir(state_dir)).get("TELEGRAM_CHAT_ID") or "").strip()
        cid = int(raw)
        return cid if cid > 0 else 0
    except (TelegramSettingsError, TypeError, ValueError, OSError):
        return 0


def owner_matches_live(state_dir: Any, chat_id: int) -> bool:
    """True only while ``chat_id`` still matches the live pinned owner in settings."""
    cid = int(chat_id or 0)
    if cid <= 0:
        return False
    return live_owner_chat_id(state_dir) == cid


def api_base_matches_live(state_dir: Any, expected_origin: str) -> bool:
    """True only while saved ``TELEGRAM_API_BASE`` still matches ``expected_origin``."""
    try:
        return telegram_api_base(_resolve_state_dir(state_dir)).rstrip("/") == str(expected_origin or "").rstrip("/")
    except (TelegramSettingsError, ValueError, OSError):
        return False


def _leading_utf16_offset(raw: str) -> int:
    stripped = raw.lstrip()
    lead = raw[: len(raw) - len(stripped)]
    return len(lead.encode("utf-16-le")) // 2


def _entity_marks_literal(raw: str, entities: Any) -> bool:
    """True when Telegram formatting entities mark the leading token as literal text (e.g. code/pre)."""
    if not isinstance(entities, list) or not entities or not raw.strip():
        return False
    start = _leading_utf16_offset(raw)
    covering = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        offset = entity.get("offset")
        length = entity.get("length")
        etype = str(entity.get("type") or "").strip().lower()
        if type(offset) is int and type(length) is int and length > 0 and offset <= start < offset + length:
            covering.append(etype)
    if not covering:
        return False
    if any(etype in _LITERAL_ENTITY_TYPES for etype in covering):
        return True
    # If entities cover offset 0 and none is bot_command (e.g. bold/italic/spoiler/etc. on a non-command),
    # Telegram never parsed a slash command at offset 0.
    return "bot_command" not in covering


def _message_text_and_entities(message: Dict[str, Any]) -> tuple[str, Any]:
    if isinstance(message.get("text"), str) and message.get("text"):
        return message["text"], message.get("entities")
    if isinstance(message.get("caption"), str) and message.get("caption"):
        return message["caption"], message.get("caption_entities")
    return "", None


def is_literal_slash_message(message: Any) -> bool:
    """True when a message starting with '/' is explicitly formatted as literal text by Telegram entities."""
    if not isinstance(message, dict):
        return False
    raw, entities = _message_text_and_entities(message)
    if not raw.strip().startswith("/"):
        return False
    return _entity_marks_literal(raw, entities)


def is_panic_command(message: Any, *, bot_username: str = "") -> bool:
    """True ONLY for a genuine /panic command (optionally wrapped in /main or /m), never code/literal or /panicking."""
    if not isinstance(message, dict):
        return False
    raw, entities = _message_text_and_entities(message)
    if not raw.strip():
        return False
    if _entity_marks_literal(raw, entities):
        return False
    cleaned = raw.strip()
    wrapped = _MAIN_CMD_RE.match(cleaned)
    if wrapped:
        inner = (wrapped.group(1) or "").strip()
        # Check if an entity marks the inner /panic token as literal code/pre too!
        if inner and isinstance(entities, list) and entities:
            inner_pos = raw.find(inner)
            if inner_pos >= 0:
                inner_u16 = len(raw[:inner_pos].encode("utf-16-le")) // 2
                for entity in entities:
                    if not isinstance(entity, dict):
                        continue
                    off, length = entity.get("offset"), entity.get("length")
                    etype = str(entity.get("type") or "").strip().lower()
                    if type(off) is int and type(length) is int and length > 0 and off <= inner_u16 < off + length:
                        if etype in _LITERAL_ENTITY_TYPES:
                            return False
        cleaned = inner
    match = _PANIC_CMD_RE.match(cleaned)
    if not match:
        return False
    target_bot = str(match.group(1) or "").strip()
    own_bot = str(bot_username or "").lstrip("@").strip()
    if target_bot and own_bot and target_bot.casefold() != own_bot.casefold():
        return False
    return True


def is_panic_text(text: Any, *, bot_username: str = "") -> bool:
    """Pure-text /panic command check (strict command boundary, rejects /panicking)."""
    return is_panic_command({"text": str(text or "")}, bot_username=bot_username)


__all__ = [
    "api_base_matches_live",
    "clear_owner_pairing",
    "is_literal_slash_message",
    "is_panic_command",
    "is_panic_text",
    "issue_owner_pairing_code",
    "live_owner_chat_id",
    "owner_matches_live",
    "redeem_owner_pairing_code",
]
