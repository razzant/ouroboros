"""Durable, non-evicting secret-message guard for Telegram owner and Presence lanes.

Tracks every sent secret-request card message ID, every refused secret-reply message ID,
and any uncertain/in-flight secret-card delivery in an owner-only SQLite ledger
(``secret_guard.sqlite3``). Never evicts known secret IDs, fails closed when storage is
unreadable, and blocks subsequent replies, quotes, edits, external_replies, and forwards
from carrying secret values into chat history, Presence custody, or model context.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, Optional, Sequence

_DB_NAME = "secret_guard.sqlite3"
_LOCK = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS secret_messages(
    bot_id TEXT NOT NULL,
    api_base TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'card',
    task_id TEXT NOT NULL DEFAULT '',
    quiz_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    PRIMARY KEY(bot_id, api_base, chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS secret_messages_chat_msg
    ON secret_messages(chat_id, message_id);
CREATE TABLE IF NOT EXISTS secret_inflight(
    op_id TEXT PRIMARY KEY,
    bot_id TEXT NOT NULL,
    api_base TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    min_epoch INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS secret_uncertain(
    bot_id TEXT NOT NULL,
    api_base TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    min_epoch INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    PRIMARY KEY(bot_id, api_base, chat_id)
);
"""


class SecretGuardUnreadable(RuntimeError):
    """Secret guard storage could not be read; replies/quotes must fail closed."""


def _state_dir(api: Any) -> pathlib.Path:
    if isinstance(api, (str, pathlib.Path)):
        return pathlib.Path(api)
    return pathlib.Path(api.get_state_dir())


def _db_path(api: Any) -> pathlib.Path:
    return _state_dir(api) / _DB_NAME


def _norm_base(api_base: str) -> str:
    return str(api_base or "https://api.telegram.org").strip().rstrip("/")


def _norm_bot(bot: str) -> str:
    return str(bot or "").strip()


def _open(path: pathlib.Path) -> sqlite3.Connection:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0), 0o600)
        os.close(fd)
    conn = sqlite3.connect(path, timeout=5.0, isolation_level=None, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.executescript(_SCHEMA)
    except BaseException:
        conn.close()
        raise
    return conn


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def record_secret_message(
    api: Any,
    bot_id: str,
    chat_id: int,
    message_id: int,
    *,
    api_base: str = "",
    kind: str = "card",
    task_id: str = "",
    quiz_id: str = "",
) -> None:
    """Permanently record a secret card or refused secret reply message ID (never evicted)."""
    cid, mid = _int_or_zero(chat_id), _int_or_zero(message_id)
    if cid == 0 or mid <= 0:
        return
    with _LOCK:
        conn = _open(_db_path(api))
        try:
            conn.execute(
                "INSERT OR IGNORE INTO secret_messages"
                "(bot_id, api_base, chat_id, message_id, kind, task_id, quiz_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_norm_bot(bot_id), _norm_base(api_base), cid, mid, str(kind or "card"),
                 str(task_id or ""), str(quiz_id or ""), time.time()),
            )
        finally:
            conn.close()


def begin_secret_send(
    api: Any, bot_id: str, chat_id: int, *, api_base: str = "", op_id: str = ""
) -> str:
    """Persist an in-flight secret-card send marker BEFORE bytes leave the machine."""
    cid = _int_or_zero(chat_id)
    token = str(op_id or uuid.uuid4().hex)
    if cid == 0:
        return token
    now = time.time()
    min_epoch = max(0, int(now) - 60)
    with _LOCK:
        conn = _open(_db_path(api))
        try:
            conn.execute(
                "INSERT OR REPLACE INTO secret_inflight(op_id, bot_id, api_base, chat_id, min_epoch, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (token, _norm_bot(bot_id), _norm_base(api_base), cid, min_epoch, now),
            )
        finally:
            conn.close()
    return token


def finish_secret_send(
    api: Any,
    op_id: str,
    *,
    bot_id: str,
    chat_id: int,
    api_base: str = "",
    confirmed_ids: Sequence[int] = (),
    uncertain: bool = False,
    reason: str = "",
    task_id: str = "",
    quiz_id: str = "",
) -> None:
    """Complete a secret-card send: clear in-flight when confirmed, or permanently mark uncertain."""
    cid = _int_or_zero(chat_id)
    now = time.time()
    norm_bot, norm_base = _norm_bot(bot_id), _norm_base(api_base)
    with _LOCK:
        conn = _open(_db_path(api))
        try:
            conn.execute("BEGIN IMMEDIATE")
            for raw_mid in confirmed_ids:
                mid = _int_or_zero(raw_mid)
                if cid != 0 and mid > 0:
                    conn.execute(
                        "INSERT OR IGNORE INTO secret_messages"
                        "(bot_id, api_base, chat_id, message_id, kind, task_id, quiz_id, created_at) "
                        "VALUES (?, ?, ?, ?, 'card', ?, ?, ?)",
                        (norm_bot, norm_base, cid, mid, str(task_id or ""), str(quiz_id or ""), now),
                    )
            row = conn.execute("SELECT min_epoch FROM secret_inflight WHERE op_id=?", (str(op_id),)).fetchone()
            min_epoch = int(row["min_epoch"]) if row is not None else max(0, int(now) - 60)
            if uncertain and cid != 0:
                conn.execute(
                    "INSERT INTO secret_uncertain(bot_id, api_base, chat_id, min_epoch, reason, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(bot_id, api_base, chat_id) DO UPDATE SET "
                    "min_epoch=MIN(min_epoch, excluded.min_epoch), reason=excluded.reason, created_at=excluded.created_at",
                    (norm_bot, norm_base, cid, min_epoch, str(reason or "unknown_delivery")[:200], now),
                )
            conn.execute("DELETE FROM secret_inflight WHERE op_id=?", (str(op_id),))
            conn.commit()
        finally:
            conn.close()


def mark_secret_uncertain(
    api: Any, bot_id: str, chat_id: int, *, api_base: str = "", reason: str = "unknown_delivery", min_epoch: Optional[int] = None
) -> None:
    """Explicitly mark ``(bot_id, api_base, chat_id)`` as having an unconfirmed/unknown secret card delivery."""
    cid = _int_or_zero(chat_id)
    if cid == 0:
        return
    now = time.time()
    epoch = max(0, int(now) - 60) if min_epoch is None else max(0, int(min_epoch))
    with _LOCK:
        conn = _open(_db_path(api))
        try:
            conn.execute(
                "INSERT INTO secret_uncertain(bot_id, api_base, chat_id, min_epoch, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(bot_id, api_base, chat_id) DO UPDATE SET "
                "min_epoch=MIN(min_epoch, excluded.min_epoch), reason=excluded.reason, created_at=excluded.created_at",
                (_norm_bot(bot_id), _norm_base(api_base), cid, epoch, str(reason or "unknown_delivery")[:200], now),
            )
        finally:
            conn.close()


def lookup_secret_message(
    api: Any, bot_id: str, chat_id: int, message_id: int, *, api_base: str = ""
) -> Optional[Dict[str, Any]]:
    """Look up a secret card or refused secret reply in the durable ledger (raises on corrupt DB)."""
    cid, mid = _int_or_zero(chat_id), _int_or_zero(message_id)
    if cid == 0 or mid <= 0:
        return None
    db = _db_path(api)
    if not db.exists():
        return None
    try:
        with _LOCK:
            conn = _open(db)
            try:
                norm_bot, norm_base = _norm_bot(bot_id), _norm_base(api_base)
                row = conn.execute(
                    "SELECT * FROM secret_messages WHERE chat_id=? AND message_id=? "
                    "AND (bot_id=? OR bot_id='') AND (api_base=? OR ?='') LIMIT 1",
                    (cid, mid, norm_bot, norm_base, str(api_base or "")),
                ).fetchone()
                if row is None and not norm_bot:
                    row = conn.execute(
                        "SELECT * FROM secret_messages WHERE chat_id=? AND message_id=? LIMIT 1",
                        (cid, mid),
                    ).fetchone()
                if row is None:
                    return None
                return {
                    "task_id": str(row["task_id"] or ""),
                    "quiz_id": str(row["quiz_id"] or ""),
                    "chat_id": int(row["chat_id"]),
                    "message_id": int(row["message_id"]),
                    "kind": str(row["kind"] or "card"),
                    "options": [],
                    "secret": True,
                }
            finally:
                conn.close()
    except (sqlite3.Error, OSError) as exc:
        raise SecretGuardUnreadable(f"secret guard unreadable ({type(exc).__name__})") from exc


def has_uncertain_secret(api: Any, bot_id: str, chat_id: int, *, api_base: str = "") -> Optional[int]:
    """Return ``min_epoch`` if ``(bot_id, api_base, chat_id)`` has an uncertain or in-flight secret card."""
    cid = _int_or_zero(chat_id)
    if cid == 0:
        return None
    db = _db_path(api)
    if not db.exists():
        return None
    try:
        with _LOCK:
            conn = _open(db)
            try:
                norm_bot, norm_base = _norm_bot(bot_id), _norm_base(api_base)
                vals = []
                for table in ("secret_uncertain", "secret_inflight"):
                    row = conn.execute(
                        f"SELECT MIN(min_epoch) AS ep FROM {table} WHERE chat_id=? "
                        "AND (bot_id=? OR bot_id='') AND (api_base=? OR ?='')",
                        (cid, norm_bot, norm_base, str(api_base or "")),
                    ).fetchone()
                    if row is not None and row["ep"] is not None:
                        vals.append(int(row["ep"]))
                return min(vals) if vals else None
            finally:
                conn.close()
    except (sqlite3.Error, OSError) as exc:
        raise SecretGuardUnreadable(f"secret guard unreadable ({type(exc).__name__})") from exc


def _quiz_json_readable_or_raise(api: Any) -> Dict[str, Any]:
    path = _state_dir(api) / "quiz_state.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SecretGuardUnreadable(f"quiz_state.json unreadable ({type(exc).__name__})") from exc
    if not isinstance(raw, dict):
        raise SecretGuardUnreadable("quiz_state.json root is not an object")
    return raw


def _message_has_reference(message: Dict[str, Any]) -> bool:
    return bool(
        isinstance(message.get("reply_to_message"), dict)
        or isinstance(message.get("external_reply"), dict)
        or isinstance(message.get("quote"), dict)
        or isinstance(message.get("forward_origin"), dict)
    )


def _referenced_targets(message: Dict[str, Any], chat_id: int) -> list[tuple[int, int, Optional[Dict[str, Any]]]]:
    """Collect ``(target_chat_id, target_message_id, ref_obj)`` from reply, external_reply, and forward_origin."""
    targets: list[tuple[int, int, Optional[Dict[str, Any]]]] = []
    reply = message.get("reply_to_message")
    if isinstance(reply, dict):
        rchat = _int_or_zero((reply.get("chat") or {}).get("id")) or chat_id
        rmid = _int_or_zero(reply.get("message_id"))
        if rmid > 0:
            targets.append((rchat, rmid, reply))
    ext = message.get("external_reply")
    if isinstance(ext, dict):
        echat = _int_or_zero((ext.get("chat") or {}).get("id")) or chat_id
        emid = _int_or_zero(ext.get("message_id"))
        if emid > 0:
            targets.append((echat, emid, ext))
    fwd = message.get("forward_origin")
    if isinstance(fwd, dict):
        fchat = _int_or_zero((fwd.get("chat") or {}).get("id")) or chat_id
        fmid = _int_or_zero(fwd.get("message_id"))
        if fmid > 0:
            targets.append((fchat, fmid, fwd))
    return targets


def _ref_looks_like_bot(ref: Optional[Dict[str, Any]], bot_id: str) -> bool:
    if not isinstance(ref, dict):
        return False
    sender = ref.get("from")
    if isinstance(sender, dict):
        if sender.get("is_bot") is True:
            return True
        if bot_id and str(sender.get("id") or "") == str(bot_id):
            return True
    origin = ref.get("origin")
    if isinstance(origin, dict):
        su = origin.get("sender_user")
        if isinstance(su, dict) and (su.get("is_bot") is True or (bot_id and str(su.get("id") or "") == str(bot_id))):
            return True
    return False


def classify_secret_exposure(
    api: Any,
    bot_id: str,
    message: Dict[str, Any],
    *,
    api_base: str = "",
    is_edit: bool = False,
    known_non_secret_fn: Any = None,
) -> Optional[Dict[str, Any]]:
    """Fail-closed check whether ``message`` (or edit/quote/reply) is exposed to a secret request.

    Returns a secret-ref dict when the message MUST be refused locally and never enter chat history,
    Presence custody, or model context. Also persists this inbound message's own ``message_id`` as a
    ``refused_reply`` so later replies/quotes/edits targeting it stay blocked permanently.
    """
    if not isinstance(message, dict):
        return None
    chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
    chat_id = _int_or_zero(chat.get("id"))
    own_mid = _int_or_zero(message.get("message_id"))
    if chat_id == 0:
        return None
    norm_bot = _norm_bot(bot_id)
    has_ref = _message_has_reference(message)
    edited = bool(is_edit or message.get("edit_date"))

    def _quarantine(record: Dict[str, Any]) -> Dict[str, Any]:
        if own_mid > 0:
            with contextlib.suppress(Exception):
                record_secret_message(
                    api, norm_bot, chat_id, own_mid, api_base=api_base, kind="refused_reply",
                    task_id=str(record.get("task_id") or ""), quiz_id=str(record.get("quiz_id") or ""),
                )
        return record

    try:
        # 1) Check quiz_state.json readability FIRST (fail closed if corrupt on any reply/quote/edit!)
        quiz_data = _quiz_json_readable_or_raise(api)

        # 2) Edit of an already-refused secret reply or secret card stays blocked forever.
        if edited and own_mid > 0:
            own_hit = lookup_secret_message(api, norm_bot, chat_id, own_mid, api_base=api_base)
            if own_hit is not None:
                return _quarantine(own_hit)

        # 3) Direct reply / external_reply / quote / forward targeting ANY secret card or refused secret reply.
        targets = _referenced_targets(message, chat_id)
        for target_chat, target_mid, _ref in targets:
            hit = lookup_secret_message(api, norm_bot, target_chat, target_mid, api_base=api_base)
            if hit is not None:
                return _quarantine(hit)
            # Legacy quiz_state.json fallback (including pre-SQLite entries).
            is_known_safe = bool(callable(known_non_secret_fn) and known_non_secret_fn(target_chat, target_mid))
            secrets_map = quiz_data.get("secret_messages") if isinstance(quiz_data.get("secret_messages"), dict) else {}
            if norm_bot and isinstance(secrets_map.get(f"{norm_bot}:{target_chat}:{target_mid}"), dict):
                return _quarantine({**secrets_map[f"{norm_bot}:{target_chat}:{target_mid}"], "secret": True})
            if not is_known_safe and isinstance(secrets_map.get(f"{target_chat}:{target_mid}"), dict):
                return _quarantine({**secrets_map[f"{target_chat}:{target_mid}"], "secret": True})
            for bucket_name in ("bots", "quizzes"):
                bucket = quiz_data.get(bucket_name) if isinstance(quiz_data.get(bucket_name), dict) else {}
                for rec in bucket.values():
                    if not isinstance(rec, dict) or not rec.get("secret"):
                        continue
                    if _int_or_zero(rec.get("chat_id")) != target_chat:
                        continue
                    rec_bot = str(rec.get("bot") or "").strip()
                    if rec_bot and norm_bot and rec_bot != norm_bot:
                        continue
                    if not rec_bot and is_known_safe:
                        continue
                    rec_ids = {_int_or_zero(rec.get("message_id")), *(_int_or_zero(x) for x in (rec.get("message_ids") or []))}
                    if target_mid in rec_ids:
                        return _quarantine({**rec, "secret": True})

        # 4) Uncertain / in-flight secret card send in this chat: fail closed on replies/quotes to
        #    bot or unproven messages (not positively recorded as a non-secret message in message_refs).
        uncertain_epoch = has_uncertain_secret(api, norm_bot, chat_id, api_base=api_base)
        if uncertain_epoch is not None and has_ref:
            for target_chat, target_mid, ref in targets:
                if target_chat != chat_id:
                    continue
                ref_date = _int_or_zero((ref or {}).get("date"))
                is_known_safe = bool(callable(known_non_secret_fn) and known_non_secret_fn(target_chat, target_mid))
                if not is_known_safe and (
                    _ref_looks_like_bot(ref, norm_bot)
                    or ref_date == 0
                    or ref_date >= uncertain_epoch
                ):
                    return _quarantine({
                        "chat_id": chat_id, "message_id": target_mid, "options": [],
                        "secret": True, "uncertain": True,
                    })
            # Quote without identifiable message_id when chat has uncertain secret card -> fail closed!
            if isinstance(message.get("quote"), dict) and not targets:
                return _quarantine({"chat_id": chat_id, "message_id": 0, "options": [], "secret": True, "uncertain": True})
    except SecretGuardUnreadable:
        if has_ref or edited:
            # Fail closed on ANY reply/quote/forward/edit when secret state cannot be read; never treat as ordinary chat.
            return {"chat_id": chat_id, "message_id": own_mid, "options": [], "secret": True, "unreadable": True}
    return None


__all__ = [
    "SecretGuardUnreadable",
    "begin_secret_send",
    "classify_secret_exposure",
    "finish_secret_send",
    "has_uncertain_secret",
    "lookup_secret_message",
    "mark_secret_uncertain",
    "record_secret_message",
]
