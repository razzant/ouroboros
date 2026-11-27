"""Identity and membership evidence for the bundled Telegram group's feed."""
from __future__ import annotations

import sqlite3
from typing import Any, Mapping

REACTION_KINDS = ("message_reaction", "message_reaction_count")
GROUP_KINDS = ("message", "edited_message", "my_chat_member", *REACTION_KINDS)
GROUP_TYPES = {"group", "supergroup"}
_GROUP_RECORD = """
INSERT INTO groups VALUES(?,?,?,?,?) ON CONFLICT(account_id, chat_id) DO UPDATE SET
    chat_type=excluded.chat_type, membership=excluded.membership, update_id=excluded.update_id
WHERE groups.membership<>'migrated' AND (? OR groups.membership='present')"""


def group_chat(update: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    """The kind and chat of a group update that the feed carries."""
    for kind in GROUP_KINDS:
        body = update.get(kind)
        if isinstance(body, Mapping):
            chat = body.get("chat")
            if isinstance(chat, Mapping) and chat.get("type") in GROUP_TYPES:
                return kind, chat
            return None
    return None


def topic_of(body: Mapping[str, Any]) -> str:
    """A forum topic's id as Telegram names it; empty for General or no forum."""
    thread = body.get("message_thread_id")
    return str(thread) if body.get("is_topic_message") and type(thread) is int and thread > 0 else ""


def event_key(update: Mapping[str, Any], kind: str, body: Mapping[str, Any], chat_id: str) -> str:
    """The same key for a sighting and a full event. Distinct edits can share the same edit_date."""
    message = body.get("message_id")
    if kind == "message" and type(message) is int:
        return f"{chat_id}:m:{message}"
    return f"{chat_id}:u:{int(update.get('update_id') or 0)}"


def coalesce_sightings(db: sqlite3.Connection, account: str, chat: str, key: str, now: float) -> None:
    """Keep the latest discovery hint without replacing any full event or forgetting a superseded identity."""
    old = ("FROM feed WHERE account_id=? AND dedupe>=? AND dedupe<? AND dedupe<>? "
           "AND json_type(body, '$.sighting') IS NOT NULL")
    params = (account, f"{chat}:", f"{chat};", key)
    db.execute("INSERT OR IGNORE INTO seen SELECT account_id, dedupe, ? " + old, (now, *params))
    db.execute("DELETE " + old, params)


def member_status(member: Mapping[str, Any]) -> str:
    """Telegram ChatMember: restricted says separately whether the bot is still a member."""
    status = member.get("status")
    if status == "restricted":
        is_member = member.get("is_member")
        return "present" if is_member is True else "left" if is_member is False else ""
    if status in ("creator", "administrator", "member"):
        return "present"
    return str(status) if status in ("left", "kicked") else ""


def sighting(update: Mapping[str, Any], kind: str, chat: Mapping[str, Any], bot_id: str) -> dict[str, Any]:
    """What a Connect chooser needs; no words, sender, files or other members."""
    body = update[kind]
    seen: dict[str, Any] = {"chat": {"id": str(chat.get("id")), "type": chat.get("type"),
                                     **{key: chat[key] for key in ("title", "username", "is_forum") if key in chat}},
                            "membership": "present", "update_id": int(update.get("update_id") or 0),
                            "date": int(body.get("date") or 0)}
    if kind == "my_chat_member":
        seen["membership"] = member_status(body.get("new_chat_member") or {}) or "unknown"
    elif str((body.get("left_chat_member") or {}).get("id") or "") == bot_id:
        seen["membership"] = "left"
    if body.get("migrate_to_chat_id"):
        seen.update(membership="migrated", migrate_to_chat_id=str(body["migrate_to_chat_id"]))
    elif body.get("migrate_from_chat_id"):
        seen["migrate_from_chat_id"] = str(body["migrate_from_chat_id"])
    return seen


def record_membership(db: sqlite3.Connection, account: str, seen: Mapping[str, Any], kind: str) -> None:
    """Evidence is in arrival order; update IDs can restart after a quiet week. A migration retires its old ID."""
    chat, status = seen["chat"], seen["membership"]
    update_id = seen["update_id"]
    if status != "unknown":
        db.execute(_GROUP_RECORD, (account, chat["id"], chat["type"], status, update_id,
                                   kind == "my_chat_member" or status != "present" or "migrate_from_chat_id" in seen))
    if seen.get("migrate_to_chat_id"):
        db.execute(_GROUP_RECORD, (account, seen["migrate_to_chat_id"], "supergroup", "present", update_id, True))
    if seen.get("migrate_from_chat_id"):
        db.execute(_GROUP_RECORD, (account, seen["migrate_from_chat_id"], "group", "migrated", update_id, True))
