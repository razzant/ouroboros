"""Owner confirmation, live binding guards, and entity-aware /panic parsing for Telegram."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import re
import secrets
import time
from typing import Any, Dict, Optional

from ..scripts.telegram_settings import (
    TelegramSettingsError,
    load_settings,
    merge_settings_unlocked,
    settings_lock,
    telegram_api_base,
)

_PAIRING_FILE = "owner_pairing.json"
_PAIRING_TTL_SEC = 600.0
_PAIRING_MAX_ATTEMPTS = 5
_REDEEMED_MEMORY_SEC = 86400.0
_MIN_START_PAIR_LEN = 16
_MAIN_CMD_RE = re.compile(r"^/(?:main|m)(?:@[A-Za-z0-9_]+)?(?:\s+(.*))?$", re.IGNORECASE | re.DOTALL)
_PANIC_CMD_RE = re.compile(r"^/panic(?:@([A-Za-z0-9_]+))?(?:\s|$)", re.IGNORECASE)
_PAIR_CMD_RE = re.compile(r"^/(start|pair)(?:@[A-Za-z0-9_]+)?\s+(\S+)\s*$", re.IGNORECASE)
_PAIR_ANY_RE = re.compile(r"^/pair(?:@[A-Za-z0-9_]+)?(?:\s|$)", re.IGNORECASE)
_START_ARG_RE = re.compile(r"^/start(?:@[A-Za-z0-9_]+)?\s+(\S+)", re.IGNORECASE)
_LITERAL_ENTITY_TYPES = frozenset({"code", "pre"})


def _resolve_state_dir(state_dir: Any) -> pathlib.Path:
    getter = getattr(state_dir, "get_state_dir", None)
    return pathlib.Path(getter() if callable(getter) else state_dir)


def _pairing_path(state_dir: Any) -> pathlib.Path:
    return _resolve_state_dir(state_dir) / _PAIRING_FILE


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _read_pairing_record(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _clear_pairing_unlocked(path: pathlib.Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def clear_owner_pairing(state_dir: Any) -> None:
    root = _resolve_state_dir(state_dir)
    try:
        with settings_lock(root):
            _clear_pairing_unlocked(_pairing_path(root))
    except (TelegramSettingsError, OSError):
        _clear_pairing_unlocked(_pairing_path(root))


def issue_owner_pairing_code(state_dir: Any, *, ttl_sec: float = _PAIRING_TTL_SEC, now: Optional[float] = None) -> Dict[str, Any]:
    """Issue a single-use, time-bounded owner confirmation code from Ouroboros (loopback-only)."""
    root = _resolve_state_dir(state_dir)
    ts = time.time() if now is None else float(now)
    ttl = max(30.0, min(float(ttl_sec), 3600.0))
    code = secrets.token_urlsafe(24)
    record = {
        "code": code,
        "issued_at": ts,
        "expires_at": ts + ttl,
        "failed_attempts": 0,
    }
    with settings_lock(root):
        _write_private_json(_pairing_path(root), record)
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
    cmd, candidate = match.group(1).lower(), match.group(2)
    root = _resolve_state_dir(state_dir)
    path = _pairing_path(root)
    try:
        with settings_lock(root):
            # Refuse redemption if an owner is already bound under the same lock.
            try:
                existing_owner = int(str(load_settings(root).get("TELEGRAM_CHAT_ID") or "").strip() or "0")
            except (TelegramSettingsError, TypeError, ValueError):
                existing_owner = 0
            if existing_owner > 0:
                return False
            if not path.is_file():
                return False
            record = _read_pairing_record(path)
            if record is None:
                _clear_pairing_unlocked(path)
                return False
            expected = str(record.get("code") or "")
            if not expected:
                return False
            try:
                expires_at = float(record.get("expires_at") or 0.0)
                failed = int(record.get("failed_attempts") or 0)
            except (TypeError, ValueError):
                _clear_pairing_unlocked(path)
                return False
            ts = time.time() if now is None else float(now)
            if ts > expires_at or failed >= _PAIRING_MAX_ATTEMPTS:
                _clear_pairing_unlocked(path)
                return False
            if not hmac.compare_digest(expected, candidate):
                # Always count /pair attempts; count /start only for plausible token lengths so stray short /start args don't burn a code.
                if cmd == "pair" or len(candidate) >= _MIN_START_PAIR_LEN:
                    failed += 1
                    if failed >= _PAIRING_MAX_ATTEMPTS:
                        _clear_pairing_unlocked(path)
                    else:
                        record["failed_attempts"] = failed
                        try:
                            _write_private_json(path, record)
                        except OSError:
                            _clear_pairing_unlocked(path)
                return False
            # Atomically replace active challenge with redeemed-hash sentinel (no plaintext code) BEFORE persisting owner.
            sentinel = {"redeemed_sha256": _sha256_text(expected), "redeemed_at": ts}
            try:
                _write_private_json(path, sentinel)
            except OSError:
                return False
            merge_settings_unlocked(root, {"TELEGRAM_CHAT_ID": str(int(chat_id))})
            return True
    except (TelegramSettingsError, OSError):
        return False


def is_pairing_command(state_dir: Any, message: Any) -> bool:
    """True when ``message`` carries a pairing command/code that must be consumed locally, never forwarded."""
    if not isinstance(message, dict):
        return False
    raw, _entities = _message_text_and_entities(message)
    cleaned = raw.strip()
    if not cleaned:
        return False
    if _PAIR_ANY_RE.match(cleaned):
        return True
    start_match = _START_ARG_RE.match(cleaned)
    if not start_match:
        return False
    arg = start_match.group(1)
    root = _resolve_state_dir(state_dir)
    chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
    is_private = str(chat.get("type") or "") == "private"
    owner = live_owner_chat_id(root)
    if owner <= 0 and is_private:
        return True
    record = _read_pairing_record(_pairing_path(root))
    if not record:
        return False
    active_code = str(record.get("code") or "")
    if active_code and hmac.compare_digest(active_code, arg):
        # Owner is already bound and sent an unused active pairing code: clear it locally and never forward.
        clear_owner_pairing(root)
        return True
    redeemed_hash = str(record.get("redeemed_sha256") or "")
    if redeemed_hash:
        try:
            redeemed_at = float(record.get("redeemed_at") or 0.0)
        except (TypeError, ValueError):
            redeemed_at = 0.0
        if (time.time() - redeemed_at) <= _REDEEMED_MEMORY_SEC and hmac.compare_digest(redeemed_hash, _sha256_text(arg)):
            return True
    return False


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


def _utf16_offset_for_pos(raw: str, pos: int) -> int:
    return len(raw[:pos].encode("utf-16-le")) // 2


def _entity_marks_literal_at(raw: str, entities: Any, char_pos: int) -> bool:
    """True ONLY when a Telegram ``code`` or ``pre`` entity covers ``char_pos``."""
    if not isinstance(entities, list) or not entities or char_pos < 0 or char_pos >= len(raw):
        return False
    start = _utf16_offset_for_pos(raw, char_pos)
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        offset = entity.get("offset")
        length = entity.get("length")
        etype = str(entity.get("type") or "").strip().lower()
        if type(offset) is int and type(length) is int and length > 0 and offset <= start < offset + length:
            if etype in _LITERAL_ENTITY_TYPES:
                return True
    return False


def _entity_marks_literal(raw: str, entities: Any) -> bool:
    """True when Telegram ``code`` or ``pre`` entities mark the leading token as literal text."""
    stripped = raw.lstrip()
    if not stripped:
        return False
    return _entity_marks_literal_at(raw, entities, len(raw) - len(stripped))


def _message_text_and_entities(message: Dict[str, Any]) -> tuple[str, Any]:
    if isinstance(message.get("text"), str) and message.get("text"):
        return message["text"], message.get("entities")
    if isinstance(message.get("caption"), str) and message.get("caption"):
        return message["caption"], message.get("caption_entities")
    return "", None


def is_literal_span(message: Any, substring: str) -> bool:
    """True when ``substring`` inside ``message`` starts with '/' and is covered by a ``code``/``pre`` entity."""
    if not isinstance(message, dict) or not substring or not substring.strip().startswith("/"):
        return False
    raw, entities = _message_text_and_entities(message)
    pos = raw.find(substring.strip())
    if pos < 0:
        return False
    return _entity_marks_literal_at(raw, entities, pos)


def is_literal_slash_message(message: Any) -> bool:
    """True when a message starting with '/' is formatted as ``code`` or ``pre`` by Telegram entities."""
    if not isinstance(message, dict):
        return False
    raw, entities = _message_text_and_entities(message)
    if not raw.strip().startswith("/"):
        return False
    return _entity_marks_literal(raw, entities)


def is_panic_command(message: Any, *, bot_username: str = "") -> bool:
    """True ONLY for a genuine /panic command (optionally wrapped in /main or /m), never code/pre or /panicking."""
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
        if inner and is_literal_span(message, inner):
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
    "is_literal_span",
    "is_pairing_command",
    "is_panic_command",
    "is_panic_text",
    "issue_owner_pairing_code",
    "live_owner_chat_id",
    "owner_matches_live",
    "redeem_owner_pairing_code",
]
