"""Chats the bot has been seen in, offered as the owner's Connect choices in Circles.

Each private chat, group or supergroup update leaves its id, type, label and the bot's
newest membership evidence beside custody, in the transaction that advances the
provider offset. Nothing here creates an inbox event, a turn or a place: the
owner connects each chat. A listed chat is an observation, not a live
membership check.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Sequence, Tuple

from .events import _integer, utf8_safe


_ACCOUNT = re.compile(r"[1-9][0-9]*")
_GROUP_TYPES = ("group", "supergroup")
_CHAT_TYPES = ("private", *_GROUP_TYPES)
_BACKFILL_KEY = "chat_backfill"
_BACKFILL_BATCH = 200
_SIGHTING_SEQ_KEY = "sighting_seq"
_ORDER_SCALE = 1_000_000_000

SEEN_SIGHTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS seen_sightings(
    sighting_key TEXT PRIMARY KEY
)"""

GROUPS_TABLE = """
CREATE TABLE IF NOT EXISTS telegram_groups(
    account_id TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    chat_type TEXT NOT NULL,
    title TEXT NOT NULL,
    title_update_id INTEGER NOT NULL,
    membership TEXT NOT NULL CHECK(membership IN ('present','left','kicked','migrated')),
    membership_update_id INTEGER NOT NULL,
    PRIMARY KEY(account_id, chat_id)
)"""

_RECORD = """
INSERT INTO telegram_groups(
    account_id, chat_id, chat_type, title, title_update_id, membership, membership_update_id
) VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(account_id, chat_id) DO UPDATE SET
    chat_type=CASE WHEN excluded.title_update_id>title_update_id
                   THEN excluded.chat_type ELSE chat_type END,
    title=CASE WHEN excluded.title_update_id>title_update_id THEN excluded.title ELSE title END,
    title_update_id=MAX(title_update_id, excluded.title_update_id),
    membership=CASE WHEN excluded.membership_update_id>membership_update_id
                    THEN excluded.membership ELSE membership END,
    membership_update_id=MAX(membership_update_id, excluded.membership_update_id)
"""


@dataclass(frozen=True)
class GroupSighting:
    account_id: str
    chat_id: str
    membership: str  # present | left | kicked | migrated
    chat_type: str = "group"
    title: Optional[str] = None
    event_date: Optional[int] = None


def group_sightings(update: Any, account_id: str) -> Tuple[GroupSighting, ...]:
    """What one raw update shows about the chats of the bot ``account_id``."""
    if not isinstance(update, dict) or not _ACCOUNT.fullmatch(str(account_id)):
        return ()
    change = update.get("my_chat_member")
    if isinstance(change, dict):
        member = _object(change.get("new_chat_member"))
        if _user_id(member.get("user")) != account_id:
            return ()
        status = member.get("status")
        when = _integer(change.get("date"))
        if status in ("creator", "administrator", "member") or (
            status == "restricted" and member.get("is_member") is True
        ):
            return _sighting(account_id, change.get("chat"), "present", event_date=when)
        if status in ("left", "kicked", "restricted"):
            return _sighting(account_id, change.get("chat"), "kicked" if status == "kicked" else "left", event_date=when)
        return ()
    message = next((update[kind] for kind in ("message", "edited_message")
                    if isinstance(update.get(kind), dict)), None)
    if message is None:
        reaction = update.get("message_reaction") or update.get("message_reaction_count")
        r_obj = _object(reaction)
        return _sighting(account_id, r_obj.get("chat"), "present", event_date=_integer(r_obj.get("date")))
    chat = message.get("chat")
    when = _integer(message.get("edit_date")) or _integer(message.get("date"))
    if _user_id(message.get("left_chat_member")) == account_id:
        return _sighting(account_id, chat, "left", event_date=when)
    if _integer(message.get("migrate_to_chat_id")) is not None:
        return _sighting(account_id, chat, "migrated", event_date=when)
    seen = _sighting(account_id, chat, "present", event_date=when)
    old = _integer(message.get("migrate_from_chat_id"))
    if seen and old is not None and str(old) != seen[0].chat_id:
        seen += (GroupSighting(account_id, str(old), "migrated", event_date=when),)
    return seen


def _next_order_key(conn: sqlite3.Connection, event_date: Optional[int]) -> int:
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (_SIGHTING_SEQ_KEY,)).fetchone()
    seq = (int(row[0]) if row is not None else 0) + 1
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (_SIGHTING_SEQ_KEY, str(seq)),
    )
    raw_epoch = int(event_date) if event_date is not None else int(time.time())
    epoch = max(0, min(9_000_000_000, raw_epoch))
    return epoch * _ORDER_SCALE + seq


def record_sightings(
    conn: sqlite3.Connection,
    update_id: int,
    sightings: Sequence[GroupSighting],
    *,
    sighting_key: str = "",
) -> None:
    """Keep the newest evidence of each sighting across server migrations, inside the caller's transaction."""
    if not sightings:
        return
    if sighting_key:
        cur = conn.execute("INSERT OR IGNORE INTO seen_sightings(sighting_key) VALUES (?)", (utf8_safe(sighting_key),))
        if cur.rowcount == 0:
            return
    for item in sightings:
        order_key = _next_order_key(conn, item.event_date)
        conn.execute(_RECORD, (
            item.account_id, item.chat_id, item.chat_type, utf8_safe(item.title or ""),
            order_key if item.title is not None else -1, item.membership, order_key,
        ))


def offered_groups(conn: sqlite3.Connection, account_id: str) -> List[Dict[str, str]]:
    """Connect rows for chats whose newest evidence shows this bot present, by label."""
    rows = conn.execute(
        "SELECT chat_id, title, chat_type FROM telegram_groups WHERE account_id=? AND membership='present'",
        (account_id,),
    ).fetchall()
    return [
        {
            "account_id": account_id,
            "conversation_id": chat_id,
            "label": f"{' '.join(title.split()) or ('Private chat' if kind == 'private' else 'Group')} ({chat_id})",
        }
        for chat_id, title, kind in sorted(
            ((str(row[0]), str(row[1]), str(row[2])) for row in rows),
            key=lambda row: (row[1].casefold(), row[0]),
        )
    ]


def backfill_legacy(connect: Callable[[], ContextManager[sqlite3.Connection]]) -> None:
    """Observe the chats of saved inbox rows once, including pre-upgrade private chats."""
    with connect() as conn:
        if _progress(conn)["done"]:
            return
    while True:
        with connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            progress = _progress(conn)
            if progress["done"]:
                conn.rollback()
                return
            rows = conn.execute(
                "SELECT rowid, update_id, payload_json FROM inbox WHERE rowid>? ORDER BY rowid LIMIT ?",
                (progress["after"], _BACKFILL_BATCH),
            ).fetchall()
            for _rowid, update_id, payload in rows:
                sightings = _captured_group(payload)
                if sightings is None:
                    progress["unreadable"] += 1
                else:
                    record_sightings(conn, update_id, sightings)
            if rows:
                progress["after"] = int(rows[-1][0])
            progress["done"] = len(rows) < _BACKFILL_BATCH
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (_BACKFILL_KEY, json.dumps(progress, sort_keys=True)),
            )
            conn.commit()
        if progress["done"]:
            return


def backfill_unreadable(conn: sqlite3.Connection) -> int:
    """Inbox rows the legacy backfill could not attribute to a bot and a chat."""
    return _progress(conn)["unreadable"]


def _captured_group(payload: Any) -> Optional[Tuple[GroupSighting, ...]]:
    try:
        event = json.loads(payload)
    except (TypeError, ValueError, RecursionError):
        return None
    conversation = _object(event.get("conversation") if isinstance(event, dict) else None)
    account = str(event.get("account_id") or "") if conversation else ""
    chat_id = _integer(event.get("conversation_id")) if conversation else None
    chat_type = conversation.get("chat_type")
    if (
        not _ACCOUNT.fullmatch(account)
        or chat_id is None
        or str(conversation.get("chat_id")) != str(chat_id)
        or chat_type not in ("private", "channel", *_GROUP_TYPES)
    ):
        return None
    if chat_type not in _CHAT_TYPES:
        return ()
    if chat_type == "private":
        actor = _object(event.get("actor"))
        person = actor if str(actor.get("platform_actor_id")) == str(chat_id) else {}
        title = _private_label({**person, **{k: v for k, v in conversation.items() if v}})
    else:
        title = conversation.get("title")
    msg = _object(event.get("message"))
    when = _integer(msg.get("sent_at_epoch")) or 0
    return (GroupSighting(account, str(chat_id), "present", chat_type, title if isinstance(title, str) else "", when),)


def _progress(conn: sqlite3.Connection) -> Dict[str, Any]:
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (_BACKFILL_KEY,)).fetchone()
    try:
        value = json.loads(row[0]) if row is not None else {}
        return {
            "after": int(value["after"]),
            "unreadable": int(value["unreadable"]),
            "done": value["done"] is True,
        }
    except (KeyError, TypeError, ValueError):
        return {"after": -1, "unreadable": 0, "done": False}


def _sighting(
    account_id: str, chat: Any, membership: str, *, event_date: Optional[int] = None
) -> Tuple[GroupSighting, ...]:
    chat = _object(chat)
    chat_id, title = _integer(chat.get("id")), chat.get("title")
    if chat.get("type") not in _CHAT_TYPES or chat_id is None:
        return ()
    if chat["type"] == "private":
        title = _private_label(chat)
    return (
        GroupSighting(account_id, str(chat_id), membership, chat["type"], title if isinstance(title, str) else "", event_date),
    )


def _private_label(chat: Dict[str, Any]) -> str:
    parts = [
        chat[key].strip()
        for key in ("first_name", "last_name")
        if isinstance(chat.get(key), str) and chat[key].strip()
    ]
    username = chat.get("username")
    if isinstance(username, str) and username.strip():
        parts.append("@" + username.strip().lstrip("@"))
    return " ".join(parts)


def _object(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _user_id(value: Any) -> str:
    return str(value.get("id")) if isinstance(value, dict) else ""
