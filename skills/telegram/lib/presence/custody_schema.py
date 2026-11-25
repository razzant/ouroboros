"""SQLite schema, lease records and pure message/reaction helpers for Presence custody."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sqlite3
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from . import discovery
from .events import ALBUM_POSITION_KEY, TelegramEvent
from .formatting import prepare_text


class InboxFullError(RuntimeError):
    """The Presence inbox reached its pending-row capacity; update must be retried without advancing cursors."""


@dataclass(frozen=True)
class InboxLease:
    event_id: str
    payload: Dict[str, Any]
    attempts: int
    lease_token: str = ""


@dataclass(frozen=True)
class OutboxLease:
    delivery_id: str
    payload: Dict[str, Any]
    attempts: int
    previous_unanswered: bool = False
    unrecorded_attempt: bool = False
    lease_token: str = ""


@dataclass(frozen=True)
class WorkLease:
    event_id: str
    binding_id: str
    work_ref: str
    event: Dict[str, Any]
    attempts: int
    role: str = "deferred"
    lease_token: str = ""


PRESENCE_WORK_TABLE = """
CREATE TABLE IF NOT EXISTS {name}(
    work_ref TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    binding_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'deferred' CHECK(role IN ('deferred','turn','child')),
    event_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK(
        state IN ('pending','leased','completed','failed','cancelled')
    ),
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at REAL NOT NULL,
    lease_until REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    completed_at REAL,
    last_error TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(event_id) REFERENCES inbox(event_id)
)"""

MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS telegram_messages(
    account_id TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    topic_id TEXT NOT NULL DEFAULT '',
    chat_forum INTEGER,
    author_id TEXT NOT NULL DEFAULT '',
    author_name TEXT NOT NULL DEFAULT '',
    is_self INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL DEFAULT '',
    media_group_id TEXT NOT NULL DEFAULT '',
    album_origin_known INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(account_id, chat_id, message_id)
)"""

MESSAGES_FORUM_INDEX = """
CREATE INDEX IF NOT EXISTS telegram_messages_forum
    ON telegram_messages(account_id, chat_id, message_id) WHERE chat_forum IS NOT NULL"""

MESSAGES_ALBUM_INDEX = """
CREATE INDEX IF NOT EXISTS telegram_messages_album
    ON telegram_messages(account_id, chat_id, media_group_id, message_id) WHERE media_group_id<>''"""

ALBUM_EARLIER = (
    "SELECT COUNT(*) FROM telegram_messages WHERE account_id=? AND chat_id=? AND media_group_id=? "
    "AND media_group_id<>'' AND message_id<?"
)

REMEMBER_MESSAGE = """
INSERT OR REPLACE INTO telegram_messages(
    account_id, chat_id, message_id, topic_id, chat_forum, author_id, author_name, is_self, text, media_group_id,
    album_origin_known
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

DOWNLOAD_FAILURES_TABLE = """
CREATE TABLE IF NOT EXISTS download_failures(
    event_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    own_failures INTEGER NOT NULL DEFAULT 0,
    first_failed_at REAL NOT NULL DEFAULT 0,
    last_failed_at REAL NOT NULL,
    last_error TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(event_id, ordinal)
)"""

TYPING_TABLE = """
CREATE TABLE IF NOT EXISTS typing_requests(
    chat_id TEXT NOT NULL,
    topic_id TEXT NOT NULL DEFAULT '',
    turn_ref TEXT NOT NULL,
    source_event_id TEXT NOT NULL DEFAULT '',
    account_id TEXT NOT NULL DEFAULT '',
    place_binding_json TEXT NOT NULL DEFAULT '',
    requested_at REAL NOT NULL,
    PRIMARY KEY(chat_id, topic_id, turn_ref)
)"""

HELD_REPORTS_SQL = (
    "SELECT COUNT(*) FROM outbox, json_each(outbox.reports_json) AS entry WHERE outbox.report_due_at>0 "
    "AND json_extract(entry.value,'$.acked')=0 AND json_extract(entry.value,'$.error') LIKE 'Held:%'"
)

CLAIMABLE_INBOX = """
SELECT event_id, payload_json, attempts
FROM inbox AS candidate
WHERE (
    (candidate.state='pending' AND candidate.available_at<=?)
    OR (candidate.state='leased' AND candidate.lease_until<=?)
)
AND NOT EXISTS (
    SELECT 1
    FROM inbox AS earlier
    WHERE earlier.conversation_key=candidate.conversation_key
      AND earlier.rowid<candidate.rowid
      AND earlier.state IN ('pending','leased')
)
ORDER BY candidate.rowid ASC
LIMIT 1
"""

CLAIMABLE_WORK = """
SELECT event_id, binding_id, work_ref, role, event_json, attempts
FROM presence_work
WHERE (state='pending' AND available_at<=?)
   OR (state='leased' AND lease_until<=?)
ORDER BY available_at ASC, created_at ASC
LIMIT 1
"""

CLAIMABLE_OUTBOX = """
SELECT candidate.delivery_id, candidate.payload_json, candidate.attempts,
       candidate.state, candidate.last_error
FROM outbox AS candidate
WHERE ((candidate.state='pending' AND candidate.available_at<=?)
   OR (candidate.state='leased' AND candidate.lease_until<=?))
AND NOT EXISTS (
    SELECT 1 FROM outbox AS earlier
    WHERE earlier.rowid < candidate.rowid
      AND earlier.state IN ('pending','leased')
      AND json_extract(earlier.payload_json, '$.chat_id') = json_extract(candidate.payload_json, '$.chat_id')
      AND CAST(COALESCE(json_extract(earlier.payload_json, '$.topic_id'), '') AS TEXT) = CAST(COALESCE(json_extract(candidate.payload_json, '$.topic_id'), '') AS TEXT)
      AND COALESCE(NULLIF(CAST(json_extract(earlier.payload_json, '$.account_id') AS TEXT), ''), CAST(json_extract(earlier.payload_json, '$._reporting.account_id') AS TEXT), '')
        = COALESCE(NULLIF(CAST(json_extract(candidate.payload_json, '$.account_id') AS TEXT), ''), CAST(json_extract(candidate.payload_json, '$._reporting.account_id') AS TEXT), '')
)
ORDER BY candidate.created_at ASC, candidate.rowid ASC
LIMIT 1
"""

REACTION_EXCERPT_CHARS = 500
UNANSWERED_ATTEMPT = frozenset({"released_at_runtime_start", "worker_cancelled"})
REACTION_ATTEMPTS_MARKED = "_attempts_marked"
STAMP_PATH = "$." + REACTION_ATTEMPTS_MARKED
CALL_STARTED = "_call_started"
REACTION_WITHHELD = "reaction_host_context_unavailable"
CONFIRMED_BOT_KEY = "confirmed_bot"

UNMARKED_NO_EFFECT = re.compile(
    r"TelegramApiError: Telegram (?:setMessageReaction failed \((?:400|401|403|404|429)\)"
    r"|(?:getMe|deleteWebhook) failed \(\w+\)): |RuntimeError: telegram token is unavailable\Z"
)
UNRECORDED_SEND_NO_EFFECT = re.compile(
    r"TelegramApiError: Telegram (?:(?:sendMessage|sendPhoto|sendDocument|sendVideo|sendAudio|sendVoice"
    r"|sendAnimation|sendMediaGroup) failed \((?:400|401|403|404|429)\): "
    r"|(?:getMe|deleteWebhook) failed \(\w+\): |upload failed \(transport\): )"
    r"|RuntimeError: telegram token is unavailable\Z|(?:FileNotFoundError|IsADirectoryError|NotADirectoryError"
    r"|PermissionError): "
)
SEND_KINDS = frozenset({"message", "photo", "document", "video", "audio", "voice", "animation", "album"})


def schema_script() -> str:
    return (
        """
        CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS inbox(
            event_id TEXT PRIMARY KEY,
            update_id INTEGER NOT NULL,
            conversation_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending','leased','submitted','failed')),
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL,
            lease_until REAL NOT NULL DEFAULT 0,
            received_at REAL NOT NULL,
            submitted_at REAL,
            failed_at REAL,
            submission_ref TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS inbox_ready_idx
            ON inbox(state, available_at, lease_until);
        """
        + PRESENCE_WORK_TABLE.format(name="presence_work")
        + """;
        CREATE INDEX IF NOT EXISTS presence_work_ready_idx
            ON presence_work(state, available_at, lease_until, created_at);
        CREATE TABLE IF NOT EXISTS outbox(
            delivery_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending','leased','delivered','failed')),
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL,
            lease_until REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            delivered_at REAL,
            failed_at REAL,
            provider_receipt_json TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS outbox_ready_idx
            ON outbox(state, available_at, lease_until, created_at);
        """
        + discovery.GROUPS_TABLE
        + ";"
        + discovery.SEEN_SIGHTINGS_TABLE
        + ";"
        + MESSAGES_TABLE
        + ";"
        + MESSAGES_FORUM_INDEX
        + ";"
        + MESSAGES_ALBUM_INDEX
        + ";"
        + DOWNLOAD_FAILURES_TABLE
        + ";"
        + TYPING_TABLE
    )


def token_binding_prefix(token: str, api_base: str = "") -> str:
    base = str(api_base or "https://api.telegram.org").strip().rstrip("/")
    return hashlib.sha256(f"{base}\0{token}".encode("utf-8")).hexdigest() + ":"


def migrate_custody_schema(conn: sqlite3.Connection) -> None:
    """Idempotently add backward-compatible security & lease-fencing columns to custody tables."""
    outbox_cols = {r[1] for r in conn.execute("PRAGMA table_info(outbox)")}
    if "reports_json" not in outbox_cols:
        conn.execute("ALTER TABLE outbox ADD COLUMN reports_json TEXT NOT NULL DEFAULT '[]'")
        conn.execute("ALTER TABLE outbox ADD COLUMN report_due_at REAL NOT NULL DEFAULT 0")
    if "lease_token" not in outbox_cols:
        conn.execute("ALTER TABLE outbox ADD COLUMN lease_token TEXT NOT NULL DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS outbox_reports_due ON outbox(report_due_at)")
    inbox_cols = {r[1] for r in conn.execute("PRAGMA table_info(inbox)")}
    if "host_receipt_json" not in inbox_cols:
        conn.execute("ALTER TABLE inbox ADD COLUMN host_receipt_json TEXT NOT NULL DEFAULT ''")
    if "lease_token" not in inbox_cols:
        conn.execute("ALTER TABLE inbox ADD COLUMN lease_token TEXT NOT NULL DEFAULT ''")
    work_cols = {r[1] for r in conn.execute("PRAGMA table_info(presence_work)")}
    if "lease_token" not in work_cols:
        conn.execute("ALTER TABLE presence_work ADD COLUMN lease_token TEXT NOT NULL DEFAULT ''")
    typing_cols = {r[1] for r in conn.execute("PRAGMA table_info(typing_requests)")}
    if "account_id" not in typing_cols:
        conn.execute("ALTER TABLE typing_requests ADD COLUMN account_id TEXT NOT NULL DEFAULT ''")
    if "place_binding_json" not in typing_cols:
        conn.execute("ALTER TABLE typing_requests ADD COLUMN place_binding_json TEXT NOT NULL DEFAULT ''")
    msg_cols = {r[1] for r in conn.execute("PRAGMA table_info(telegram_messages)")}
    if "album_origin_known" not in msg_cols:
        conn.execute("ALTER TABLE telegram_messages ADD COLUMN album_origin_known INTEGER NOT NULL DEFAULT 0")


def build_delivery_receipt(row: sqlite3.Row, *, stored_answers_fn: Any, outbox_files_mod: Any) -> Dict[str, Any]:
    payload = json.loads(row["payload_json"])
    reaction = payload.get("kind") == "operation" and payload.get("method") == "setMessageReaction"
    reporting = payload.get("_reporting") or {}
    answers = stored_answers_fn(payload)
    files = [
        {
            "file_name": pathlib.PurePath(str(item["path"])).name,
            "source_path": str(item.get("source") or ""),
            "copy_path": str(item["path"]),
            "sha256": str(item.get("sha256") or ""),
            "size_bytes": item.get("size"),
        }
        for item in outbox_files_mod.records(payload)
    ]
    state_val = (
        "held"
        if row["state"] == "pending" and str(row["last_error"]).startswith("Held:")
        else "uncertain"
        if payload.get("_terminal_outcome") == "uncertain"
        else str(row["state"])
    )
    return {
        **(
            {"history_limitation": "Host history for reactions is unavailable: this receipt stays local."}
            if reaction
            and (reporting.get("version") != -1 or row["state"] != "pending")
            and reporting.get("explicit_completion") != 1
            and not any(entry["acked"] for entry in json.loads(row["reports_json"]))
            else {}
        ),
        "state": state_val,
        "attempts": int(row["attempts"]),
        "delivered_at": iso_timestamp(row["delivered_at"]),
        "provider_receipt": json.loads(row["provider_receipt_json"] or "{}"),
        **(
            {
                "confirmed_messages": answers[0],
                **({"answers_without_message_id": answers[1]} if answers[1] else {}),
            }
            if answers is not None and row["state"] != "delivered"
            else {}
        ),
        **({"files": files} if files else {}),
        "error": str(row["last_error"]),
        "delivery_reporting": payload.get("_reporting", {"version": 0, "status": "legacy_unconfirmed"}),
        "history_reports": json.loads(row["reports_json"]),
    }


def build_status_snapshot(conn: sqlite3.Connection, *, telegram_offset: int, backfill_unreadable: int) -> Dict[str, Any]:
    inbox = {
        str(row["state"]): int(row["count"])
        for row in conn.execute("SELECT state, COUNT(*) AS count FROM inbox GROUP BY state")
    }
    outbox = {
        str(row["state"]): int(row["count"])
        for row in conn.execute("SELECT state, COUNT(*) AS count FROM outbox GROUP BY state")
    }
    work = {
        str(row["state"]): int(row["count"])
        for row in conn.execute("SELECT state, COUNT(*) AS count FROM presence_work GROUP BY state")
    }
    last_event = conn.execute("SELECT MAX(received_at) AS ts FROM inbox").fetchone()["ts"]
    last_delivery = conn.execute("SELECT MAX(delivered_at) AS ts FROM outbox").fetchone()["ts"]
    withheld = conn.execute(
        "SELECT COUNT(*) FROM inbox WHERE state='failed' AND last_error=?", (REACTION_WITHHELD,)
    ).fetchone()[0]
    uncertain = conn.execute(
        "SELECT COUNT(*) FROM outbox WHERE state='failed' "
        "AND json_extract(payload_json,'$._terminal_outcome')='uncertain'"
    ).fetchone()[0]
    return {
        "telegram_offset": telegram_offset,
        "inbox_waiting": inbox.get("pending", 0),
        "inbox_leased": inbox.get("leased", 0),
        "inbox_submitted": inbox.get("submitted", 0),
        "inbox_retained": conn.execute(
            "SELECT COUNT(DISTINCT event_id) FROM presence_work "
            "WHERE role IN ('turn','child') AND state IN ('pending','leased')"
        ).fetchone()[0],
        "inbox_failed": inbox.get("failed", 0) - withheld,
        "reactions_withheld": withheld,
        "outbox_waiting": outbox.get("pending", 0),
        "outbox_held": conn.execute(
            "SELECT COUNT(*) FROM outbox WHERE state='pending' AND last_error LIKE 'Held:%'"
        ).fetchone()[0],
        "outbox_leased": outbox.get("leased", 0),
        "outbox_delivered": outbox.get("delivered", 0),
        "outbox_failed": outbox.get("failed", 0) - uncertain,
        "outbox_uncertain": uncertain,
        "work_waiting": work.get("pending", 0),
        "work_leased": work.get("leased", 0),
        "work_terminal": sum(work.get(state, 0) for state in ("completed", "failed", "cancelled")),
        "delivery_reports_pending": conn.execute("SELECT COUNT(*) FROM outbox WHERE report_due_at>0").fetchone()[0],
        "delivery_reports_held": conn.execute(HELD_REPORTS_SQL).fetchone()[0],
        "group_backfill_unreadable": backfill_unreadable,
        "last_event_at": iso_timestamp(last_event),
        "last_delivery_at": iso_timestamp(last_delivery),
    }


def chat_forum(chat_type: Any, is_forum: Any, topic_id: Any = "") -> Optional[int]:
    if is_forum is True or topic_id not in ("", None):
        return 1
    return 0 if chat_type in {"group", "supergroup", "channel"} else None


def unmarked_reaction(payload: Dict[str, Any]) -> bool:
    return (
        payload.get("kind") == "operation"
        and payload.get("method") == "setMessageReaction"
        and not {"_provider_call_started", "_operation_result"} & set(payload)
    )


def recorded_calls(payload: Dict[str, Any]) -> bool:
    kind = str(payload.get("kind") or "message")
    if kind in SEND_KINDS or kind == "moderation":
        return True
    return kind == "operation" and payload.get("method") != "setMessageReaction"


_ACTIVE_OWNERS: set[str] = set()


def register_runtime_owner(owner_id: str) -> str:
    _ACTIVE_OWNERS.add(str(owner_id))
    return str(owner_id)


def unregister_runtime_owner(owner_id: str) -> None:
    _ACTIVE_OWNERS.discard(str(owner_id))


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        from ...scripts.platform_support import process_alive

        return bool(process_alive(pid))
    except Exception:
        if os.name == "nt":
            return True  # Indeterminate on Windows: retain lease until expiry rather than signaling.
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return True


def lease_releasable(lease_until: float, lease_token: str, now: float) -> bool:
    if float(lease_until or 0.0) <= now or not lease_token:
        return True
    parts = str(lease_token).split(":", 2)
    if len(parts) != 3 or not parts[0].isdigit():
        return False
    pid, owner = int(parts[0]), parts[1]
    return (owner not in _ACTIVE_OWNERS) if pid == os.getpid() else (not _pid_alive(pid))


def held_by_clause(attempts: Optional[int] = None, lease_token: Optional[str] = None) -> tuple:
    clauses, args = [], []
    if attempts is not None:
        clauses.append(" AND attempts=?")
        args.append(int(attempts))
    if lease_token:
        clauses.append(" AND lease_token=?")
        args.append(str(lease_token))
    return ("".join(clauses), tuple(args))


def unproven_earlier_attempt(payload: Dict[str, Any], attempts: int, last_error: str, no_effect: re.Pattern) -> bool:
    stamp = payload.get(REACTION_ATTEMPTS_MARKED)
    other = attempts - (stamp if type(stamp) is int else 0)
    return not 0 <= other <= 1 or (other == 1 and not no_effect.match(last_error))


def remember_event(conn: sqlite3.Connection, event: TelegramEvent) -> None:
    actor = event.actor
    name = " ".join(part for part in (str(actor.get("first_name") or ""), str(actor.get("last_name") or "")) if part)
    mid = int(event.message["message_id"])
    mg = str(event.message.get("media_group_id") or "")
    origin_known = 0
    if mg:
        earlier = int(conn.execute(ALBUM_EARLIER, (event.account_id, event.conversation_id, mg, mid)).fetchone()[0])
        if earlier == 0 and mid > 1:
            prev = conn.execute(
                "SELECT media_group_id FROM telegram_messages WHERE account_id=? AND chat_id=? AND message_id=?",
                (event.account_id, event.conversation_id, mid - 1),
            ).fetchone()
            if prev is not None and str(prev["media_group_id"] or "") != mg:
                origin_known = 1
    conn.execute(
        REMEMBER_MESSAGE,
        (
            event.account_id,
            event.conversation_id,
            mid,
            event.thread_id,
            chat_forum(event.conversation.get("chat_type"), event.conversation.get("is_forum"), event.thread_id),
            str(actor.get("platform_actor_id") or ""),
            name or str((actor.get("sender_chat") or {}).get("title") or ""),
            0,
            event.text,
            mg,
            origin_known,
        ),
    )


def with_album_position(conn: sqlite3.Connection, event: TelegramEvent) -> TelegramEvent:
    group = str(event.message.get("media_group_id") or "")
    if not group:
        return event
    earlier = conn.execute(
        ALBUM_EARLIER, (event.account_id, event.conversation_id, group, int(event.message["message_id"]))
    ).fetchone()[0]
    return replace(event, message={**event.message, ALBUM_POSITION_KEY: int(earlier) + 1})


def retained_message(conn: sqlite3.Connection, account_id: str, chat_id: str, message_id: Any) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM telegram_messages WHERE account_id=? AND chat_id=? AND message_id=?",
        (str(account_id), str(chat_id), int(message_id)),
    ).fetchone()


def newest_chat_forum(conn: sqlite3.Connection, account_id: str, conversation_id: str) -> Optional[int]:
    newest = conn.execute(
        "SELECT chat_forum FROM telegram_messages WHERE account_id=? AND chat_id=? "
        "AND chat_forum IS NOT NULL ORDER BY message_id DESC LIMIT 1",
        (str(account_id), str(conversation_id)),
    ).fetchone()
    return newest["chat_forum"] if newest else None


def with_reaction_target(conn: sqlite3.Connection, event: TelegramEvent) -> TelegramEvent:
    fact = event.message["reaction"]
    row = retained_message(conn, event.account_id, event.conversation_id, event.message["message_id"])
    if row is None:
        if (
            fact["target"]["resolution"] == "resolved"
            and newest_chat_forum(conn, event.account_id, event.conversation_id) == 1
        ):
            return replace(
                event,
                message={
                    **event.message,
                    "reaction": {**fact, "target": {**fact["target"], "resolution": "unresolved"}},
                },
            )
        return event
    author = {
        **({"platform_actor_id": row["author_id"]} if row["author_id"] else {}),
        **({"display_name": row["author_name"]} if row["author_name"] else {}),
        "is_self": bool(row["is_self"]),
    }
    words = str(row["text"])
    target = {
        **fact["target"],
        "resolution": "resolved",
        "author": author,
        **(
            {"excerpt": {"text": words[:REACTION_EXCERPT_CHARS], "truncated": len(words) > REACTION_EXCERPT_CHARS}}
            if words.strip()
            else {}
        ),
    }
    topic = str(row["topic_id"])
    return replace(
        event,
        thread_id=topic,
        conversation_key=f"telegram:{event.account_id}:{event.conversation_id}:{topic or '0'}",
        conversation={**event.conversation, "topic_id": int(topic) if topic else None},
        message={
            **event.message,
            "reaction": {**fact, "target": target},
            "target_lookup": {
                "source": "telegram custody",
                "text": words,
                "topic_id": topic,
                "author_id": row["author_id"],
                "author_name": row["author_name"],
                "is_self": bool(row["is_self"]),
            },
        },
    )


def anchor_message_id(conn: sqlite3.Connection, event: Dict[str, Any], anchor: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(anchor, dict) or not isinstance(anchor.get("message"), dict):
        return None
    message_id = anchor["message"].get("message_id")
    if isinstance(message_id, bool) or message_id in (None, ""):
        return None
    row = conn.execute(
        "SELECT conversation_key, payload_json FROM inbox WHERE event_id=?",
        (str(anchor.get("source_event_id") or ""),),
    ).fetchone()
    if row is None or row["conversation_key"] != event.get("conversation_key"):
        return None
    captured = (json.loads(row["payload_json"]).get("message") or {}).get("message_id")
    return captured if str(captured) == str(message_id) else None


def reply_payload(event: Dict[str, Any], text: str, reply_to: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "kind": "message",
        "chat_id": str(event.get("conversation_id") or ""),
        "text": str(text),
        "markdown": True,
        "_rendered_chunks": prepare_text(str(text)),
        "_reporting": event.get("_reporting", {"version": 0}),
    }
    account_id = str(event.get("account_id") or "").strip()
    if account_id:
        payload["account_id"] = account_id
    thread_id = str(event.get("thread_id") or "").strip()
    if thread_id:
        payload["topic_id"] = thread_id
    if isinstance(event.get("_place_binding"), dict):
        payload["_place_binding"] = dict(event["_place_binding"])
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    reply_id = message.get("message_id") if reply_to in (None, "") else reply_to
    if reply_id not in (None, ""):
        payload["reply_to_message_id"] = reply_id
    return payload


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def iso_timestamp(value: Any) -> str:
    if not value:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(value)))
