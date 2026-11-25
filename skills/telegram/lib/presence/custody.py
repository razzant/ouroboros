"""SQLite custody store for Telegram Presence updates, work tracking and deliveries."""

from __future__ import annotations

import json, os, pathlib, sqlite3, threading, time, uuid, weakref
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

from . import discovery, outbox_files
from .custody_schema import (
    ALBUM_EARLIER, CALL_STARTED, CLAIMABLE_INBOX, CLAIMABLE_OUTBOX, CLAIMABLE_WORK, CONFIRMED_BOT_KEY,
    PRESENCE_WORK_TABLE, REACTION_ATTEMPTS_MARKED, REACTION_WITHHELD, REMEMBER_MESSAGE,
    STAMP_PATH, UNANSWERED_ATTEMPT, UNMARKED_NO_EFFECT, UNRECORDED_SEND_NO_EFFECT, InboxFullError, InboxLease,
    OutboxLease, WorkLease, anchor_message_id, build_delivery_receipt, build_status_snapshot, chat_forum,
    compact_json, held_by_clause, lease_releasable, migrate_custody_schema, newest_chat_forum, recorded_calls,
    register_runtime_owner, remember_event, reply_payload, retained_message, schema_script, token_binding_prefix,
    unmarked_reaction, unregister_runtime_owner, unproven_earlier_attempt, with_album_position, with_reaction_target,
)
from .delivery import stored_answers
from .events import TelegramEvent, utf8_safe
from .file_access import idempotency_fingerprint

_SHARED: "weakref.WeakValueDictionary[str, CustodyStore]" = weakref.WeakValueDictionary()
_SHARED_LOCK = threading.Lock()


class CustodyStore:
    """SQLite custody over one kept connection; SQLite owns queue truth across restarts."""

    def __init__(self, path: pathlib.Path):
        self.path, self._lock, self._conn = pathlib.Path(path), threading.Lock(), None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owner_id = uuid.uuid4().hex[:12]
        self._initialize()

    def register_owner(self, owner_id: Optional[str] = None) -> str:
        with _SHARED_LOCK:
            return register_runtime_owner(str(owner_id or self._owner_id))

    def unregister_owner(self, owner_id: Optional[str] = None) -> None:
        with _SHARED_LOCK:
            unregister_runtime_owner(str(owner_id or self._owner_id))

    def _mint_lease_token(self, owner_id: Optional[str] = None) -> str:
        return f"{os.getpid()}:{owner_id or self._owner_id}:{uuid.uuid4().hex}"

    @classmethod
    def shared(cls, path: pathlib.Path) -> "CustodyStore":
        key = os.path.realpath(path)
        with _SHARED_LOCK:
            store = _SHARED.get(key)
            if store is None:
                store = _SHARED[key] = cls(path)
            return store

    def __del__(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
            if conn is not None:
                conn.close()

    def commit_update(
        self, update_id: int, event: Optional[TelegramEvent], groups: Sequence[discovery.GroupSighting] = (),
        *, sighting_key: str = "", max_pending_inbox: int = 2000,
    ) -> bool:
        """Atomically deduplicate an event, note chat sightings and advance offset."""
        now, inserted = time.time(), False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if event is not None:
                if "reaction" in event.message:
                    event = with_reaction_target(conn, event)
                else:
                    event = with_album_position(conn, event)
                exists = conn.execute("SELECT 1 FROM inbox WHERE event_id=?", (event.source_event_id,)).fetchone() is not None
                if not exists and max_pending_inbox > 0:
                    pending = conn.execute("SELECT COUNT(*) FROM inbox WHERE state='pending'").fetchone()[0]
                    if pending >= max_pending_inbox:
                        conn.rollback()
                        raise InboxFullError(f"Presence inbox is full ({pending}/{max_pending_inbox} pending)")
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO inbox(event_id, update_id, conversation_key, payload_json, "
                    "state, attempts, available_at, received_at) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?)",
                    (event.source_event_id, int(update_id), event.conversation_key, compact_json(event.to_dict()), now, now),
                )
                inserted = cursor.rowcount == 1
                if inserted and "reaction" not in event.message:
                    remember_event(conn, event)
            if event is None or inserted:
                discovery.record_sightings(conn, int(update_id), groups, sighting_key=sighting_key)
            next_offset = max(self._metadata_int(conn, "telegram_offset"), int(update_id) + 1)
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES('telegram_offset', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(next_offset),),
            )
            conn.commit()
        return inserted

    def record_inbox_overflow(
        self, update_id: int, conversation_key: str, groups: Sequence[discovery.GroupSighting] = (), *, sighting_key: str = ""
    ) -> None:
        """Record an inbox overflow gap and advance offset without losing sightings or stalling /panic."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            discovery.record_sightings(conn, int(update_id), groups, sighting_key=sighting_key)
            count = self._metadata_int(conn, "inbox_overflow_count") + 1
            next_offset = max(self._metadata_int(conn, "telegram_offset"), int(update_id) + 1)
            for key, val in (
                ("inbox_overflow_count", str(count)),
                ("inbox_overflow_last_conversation", str(conversation_key or "")[:200]),
                ("telegram_offset", str(next_offset)),
            ):
                conn.execute(
                    "INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, val),
                )
            conn.commit()

    def telegram_offset(self) -> int:
        with self._connect() as conn:
            return self._metadata_int(conn, "telegram_offset")

    def record_confirmed_bot(self, token: str, account_id: str, *, api_base: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (CONFIRMED_BOT_KEY, token_binding_prefix(token, api_base) + account_id),
            )

    def forget_confirmed_bot(self, token: str, *, api_base: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM metadata WHERE key=? AND value LIKE ?",
                (CONFIRMED_BOT_KEY, token_binding_prefix(token, api_base) + "%"),
            )

    def confirmed_account(self, token: str, *, api_base: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key=?", (CONFIRMED_BOT_KEY,)).fetchone()
        binding = token_binding_prefix(token, api_base)
        return row["value"][len(binding):] if row is not None and row["value"].startswith(binding) else ""

    @staticmethod
    def _claim_candidate(conn: sqlite3.Connection, query: str, now: float) -> Optional[sqlite3.Row]:
        if conn.execute(query, (now, now)).fetchone() is None:
            return None
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(query, (now, now)).fetchone()
        if row is None:
            conn.commit()
        return row

    def claim_inbox(self, *, lease_sec: float = 90.0) -> Optional[InboxLease]:
        now, token = time.time(), self._mint_lease_token()
        with self._connect() as conn:
            row = self._claim_candidate(conn, CLAIMABLE_INBOX, now)
            if row is None:
                return None
            updated = conn.execute(
                "UPDATE inbox SET state='leased', lease_until=?, attempts=attempts+1, lease_token=?, last_error='' "
                "WHERE event_id=? AND ((state='pending' AND available_at<=?) OR (state='leased' AND lease_until<=?))",
                (now + max(1.0, lease_sec), token, row["event_id"], now, now),
            )
            conn.commit()
            if updated.rowcount != 1:
                return None
            return InboxLease(
                event_id=str(row["event_id"]),
                payload=json.loads(row["payload_json"]),
                attempts=int(row["attempts"]) + 1,
                lease_token=token,
            )

    def inbox_lease_valid(self, event_id: str, lease_token: str) -> bool:
        if not lease_token:
            return True
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM inbox WHERE event_id=? AND state='leased' AND lease_token=?",
                (event_id, lease_token),
            ).fetchone() is not None

    def release_inbox(
        self, event_id: str, *, reason: str, retry_after_sec: float, lease_token: Optional[str] = None
    ) -> bool:
        held, args = held_by_clause(None, lease_token)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE inbox SET state='pending', available_at=?, lease_until=0, lease_token='', last_error=? "
                "WHERE event_id=? AND state='leased'" + held,
                (time.time() + max(0.0, retry_after_sec), utf8_safe(str(reason)[:500]), event_id, *args),
            )
            return cur.rowcount == 1

    def defer_inbox(
        self, event_id: str, *, reason: str, retry_after_sec: float, lease_token: Optional[str] = None
    ) -> bool:
        held, args = held_by_clause(None, lease_token)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE inbox SET state='pending', available_at=?, lease_until=0, lease_token='', last_error=?, "
                "attempts=MAX(0, attempts-1) WHERE event_id=? AND state='leased'" + held,
                (time.time() + max(0.0, retry_after_sec), utf8_safe(str(reason)[:500]), event_id, *args),
            )
            return cur.rowcount == 1

    def release_own_leases(self) -> Dict[str, int]:
        now, released = time.time(), {}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for table, pk in (("inbox", "event_id"), ("outbox", "delivery_id"), ("presence_work", "work_ref")):
                rows = conn.execute(f"SELECT {pk}, lease_until, lease_token FROM {table} WHERE state='leased'").fetchall()
                stale = [r[pk] for r in rows if lease_releasable(r["lease_until"], str(r["lease_token"] or ""), now)]
                for row_id in stale:
                    conn.execute(
                        f"UPDATE {table} SET state='pending', available_at=?, lease_until=0, lease_token='', "
                        f"last_error='released_at_runtime_start' WHERE {pk}=? AND state='leased'", (now, row_id),
                    )
                released[table] = len(stale)
            conn.commit()
        return released

    def checkpoint_inbox(self, event_id: str, payload: Dict[str, Any], *, lease_token: Optional[str] = None) -> bool:
        held, args = held_by_clause(None, lease_token)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE inbox SET payload_json=? WHERE event_id=? AND state='leased'" + held,
                (compact_json(payload), event_id, *args),
            )
            return cur.rowcount == 1

    def download_failure(self, event_id: str, ordinal: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM download_failures WHERE event_id=? AND ordinal=?",
                (event_id, int(ordinal)),
            ).fetchone()
        return dict(row) if row is not None else None

    def note_download_failure(
        self, event_id: str, ordinal: int, *, error: str, counted: bool, now: float
    ) -> Dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO download_failures(event_id, ordinal, own_failures, first_failed_at, last_failed_at,
                                              last_error)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id, ordinal) DO UPDATE SET
                    own_failures=own_failures + excluded.own_failures,
                    first_failed_at=CASE WHEN first_failed_at=0 THEN excluded.first_failed_at
                                         ELSE first_failed_at END,
                    last_failed_at=excluded.last_failed_at, last_error=excluded.last_error
                """,
                (event_id, int(ordinal), int(counted), now if counted else 0.0, now, utf8_safe(str(error)[:500])),
            )
            return dict(
                conn.execute(
                    "SELECT * FROM download_failures WHERE event_id=? AND ordinal=?",
                    (event_id, int(ordinal)),
                ).fetchone()
            )

    def forget_download_failures(self, event_id: str) -> None:
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM download_failures WHERE event_id=? LIMIT 1", (event_id,)).fetchone():
                conn.execute("DELETE FROM download_failures WHERE event_id=?", (event_id,))

    def request_typing(
        self, *, chat_id: str, topic_id: str, turn_ref: str, source_event_id: str, account_id: str = "",
        place_binding: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO typing_requests"
                "(chat_id, topic_id, turn_ref, source_event_id, account_id, place_binding_json, requested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(chat_id), str(topic_id or ""), str(turn_ref), str(source_event_id or ""), str(account_id or ""),
                 compact_json(place_binding) if isinstance(place_binding, dict) else "", time.time()),
            )

    def typing_requests(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM typing_requests ORDER BY requested_at")]

    def forget_typing(self, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM typing_requests WHERE chat_id=? AND topic_id=? AND turn_ref=? AND requested_at=?",
                [(row["chat_id"], row["topic_id"], row["turn_ref"], row["requested_at"]) for row in rows],
            )

    def clear_typing(self) -> None:
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM typing_requests LIMIT 1").fetchone():
                conn.execute("DELETE FROM typing_requests")

    def turn_live(self, source_event_id: str, turn_ref: str) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM inbox WHERE event_id=? AND state='leased' UNION ALL "
                    "SELECT 1 FROM presence_work WHERE work_ref=? AND state IN ('pending','leased') LIMIT 1",
                    (str(source_event_id or ""), str(turn_ref or "")),
                ).fetchone()
                is not None
            )

    def mark_inbox_failed(
        self, event_id: str, *, reason: str, receipt: Optional[Dict[str, Any]] = None, lease_token: Optional[str] = None
    ) -> bool:
        held, args = held_by_clause(None, lease_token)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE inbox SET state='failed', lease_until=0, lease_token='', failed_at=?, last_error=?, "
                "host_receipt_json=COALESCE(?, host_receipt_json) WHERE event_id=? AND state='leased'" + held,
                (time.time(), utf8_safe(str(reason)[:500]), compact_json(receipt) if receipt else None, event_id, *args),
            )
            return cur.rowcount == 1

    def record_retention(
        self, event_id: str, *, binding_id: str, receipt: Dict[str, Any], lease_token: Optional[str] = None
    ) -> bool:
        response = receipt.get("response") if isinstance(receipt.get("response"), dict) else {}
        turn_ref = str(response.get("turn_ref") or "").strip()
        if not turn_ref:
            raise ValueError("retained receipt must name its original turn")
        child_ref = str(response.get("work_ref") or "").strip()
        mode = int(response.get("delivery_reporting_version") == 1)
        now, (held, args) = time.time(), held_by_clause(None, lease_token)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT payload_json FROM inbox WHERE event_id=? AND state='leased'" + held, (event_id, *args)
            ).fetchone()
            if row is None:
                conn.rollback()
                if lease_token:
                    return False
                raise ValueError("inbox event is not leased")
            event = json.loads(row["payload_json"])
            self._add_work(conn, event_id, binding_id, turn_ref, "turn", event, mode, now)
            if child_ref and child_ref != turn_ref:
                self._add_work(conn, event_id, binding_id, child_ref, "child", event, mode, now)
            updated = conn.execute(
                "UPDATE inbox SET state='submitted', lease_until=0, lease_token='', submitted_at=?, submission_ref=?, "
                "host_receipt_json=?, last_error='' WHERE event_id=? AND state='leased'" + held,
                (now, turn_ref[:500], compact_json(receipt), event_id, *args),
            )
            if updated.rowcount != 1:
                conn.rollback()
                if lease_token:
                    return False
                raise ValueError("inbox event lease changed")
            conn.commit()
            return True

    @staticmethod
    def _add_work(
        conn: sqlite3.Connection,
        event_id: str,
        binding_id: str,
        work_ref: str,
        role: str,
        event: Dict[str, Any],
        mode: int,
        now: float,
    ) -> None:
        source = {key: value for key, value in event.items() if key != "_reporting"}
        source["_reporting"] = {
            "version": mode,
            "account_id": event.get("account_id", ""),
            "origin": {
                "kind": "automatic",
                "task_id": work_ref,
                "source_event_id": event.get("source_event_id", ""),
            },
        }
        conn.execute(
            """
            INSERT OR IGNORE INTO presence_work(
                work_ref, event_id, binding_id, role, event_json, state,
                attempts, available_at, created_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?)
            """,
            (work_ref, event_id, binding_id, role, compact_json(source), now, now),
        )

    def record_submission(
        self,
        event_id: str,
        *,
        binding_id: str,
        turn_ref: str,
        outcome: str,
        text: str,
        work_ref: str,
        delivery_reporting_version: int = 0,
        continuation_ref: str = "",
        output_ref: str = "",
        reply_anchor: Optional[Dict[str, Any]] = None,
        lease_token: Optional[str] = None,
    ) -> bool:
        now, (held, args) = time.time(), held_by_clause(None, lease_token)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT payload_json FROM inbox WHERE event_id=? AND state='leased'" + held, (event_id, *args)
            ).fetchone()
            if row is None:
                conn.rollback()
                if lease_token:
                    return False
                raise ValueError("inbox event is not leased")
            event = json.loads(row["payload_json"])
            event["_reporting"] = {
                "version": delivery_reporting_version,
                "account_id": event.get("account_id", ""),
                "origin": {
                    "kind": "automatic",
                    "task_id": turn_ref,
                    "source_event_id": event.get("source_event_id", ""),
                },
            }
            if text and outcome in {"message", "deferred"}:
                if output_ref:
                    self._enqueue_output(
                        conn, event_id, turn_ref, event,
                        {"output_ref": output_ref, "text": text, "reply_anchor": reply_anchor}, now,
                    )
                else:
                    self._enqueue_outbox(
                        conn, f"presence:{event_id}:turn",
                        reply_payload(event, text, anchor_message_id(conn, event, reply_anchor)), now,
                    )
            observed = continuation_ref or (work_ref if outcome == "deferred" else "")
            if outcome == "deferred" and not observed:
                conn.rollback()
                raise ValueError("deferred submission requires work_ref")
            if observed:
                conn.execute(
                    "INSERT OR IGNORE INTO presence_work(event_id, binding_id, work_ref, event_json, state, "
                    "attempts, available_at, created_at) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?)",
                    (event_id, binding_id, observed, compact_json(event), now, now),
                )
            if continuation_ref and work_ref and work_ref != continuation_ref:
                self._add_work(conn, event_id, binding_id, work_ref, "child", event, delivery_reporting_version, now)
            updated = conn.execute(
                "UPDATE inbox SET state='submitted', lease_until=0, lease_token='', submitted_at=?, submission_ref=?, "
                "last_error='' WHERE event_id=? AND state='leased'" + held,
                (now, str(turn_ref or work_ref)[:500], event_id, *args),
            )
            if updated.rowcount != 1:
                conn.rollback()
                if lease_token:
                    return False
                raise ValueError("inbox event lease changed")
            conn.commit()
            return True

    def enqueue_outbox(self, delivery_id: str, payload: Dict[str, Any]) -> bool:
        now = time.time()
        with self._connect() as conn:
            cursor = self._enqueue_outbox(conn, delivery_id, payload, now)
            return cursor.rowcount == 1

    def enqueue_outbox_checked(self, delivery_id: str, payload: Dict[str, Any], *, fingerprint: str = "") -> str:
        """Atomically enqueue or verify idempotent retry matches prior caller, target, and content."""
        now = time.time()
        stored_payload = {**payload, "_idempotency_fingerprint": fingerprint} if fingerprint else payload
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT payload_json FROM outbox WHERE delivery_id=?", (delivery_id,)).fetchone()
            if existing is None:
                self._enqueue_outbox(conn, delivery_id, stored_payload, now)
                conn.commit()
                return "queued"
            conn.commit()
            prev = json.loads(existing["payload_json"])
            prev_fp = str(prev.get("_idempotency_fingerprint") or "")
            if fingerprint:
                if prev_fp:
                    return "already_queued" if prev_fp == fingerprint else "conflict"
                if prev.get("kind") not in {"message", "moderation", "operation"} and not prev.get("_frozen"):
                    return "conflict"
                return "already_queued" if idempotency_fingerprint(prev) == fingerprint else "conflict"
            return "conflict"

    @staticmethod
    def _enqueue_outbox(conn: sqlite3.Connection, delivery_id: str, payload: Dict[str, Any], now: float) -> sqlite3.Cursor:
        return conn.execute(
            "INSERT OR IGNORE INTO outbox(delivery_id, payload_json, state, attempts, available_at, created_at) "
            "VALUES (?, ?, 'pending', 0, ?, ?)",
            (delivery_id, compact_json(payload), now, now),
        )

    def claim_work(self, *, lease_sec: float = 90.0) -> Optional[WorkLease]:
        now, token = time.time(), self._mint_lease_token()
        with self._connect() as conn:
            row = self._claim_candidate(conn, CLAIMABLE_WORK, now)
            if row is None:
                return None
            updated = conn.execute(
                "UPDATE presence_work SET state='leased', lease_until=?, attempts=attempts+1, lease_token=?, last_error='' "
                "WHERE work_ref=? AND ((state='pending' AND available_at<=?) OR (state='leased' AND lease_until<=?))",
                (now + max(1.0, lease_sec), token, row["work_ref"], now, now),
            )
            conn.commit()
            if updated.rowcount != 1:
                return None
            return WorkLease(
                event_id=str(row["event_id"]), binding_id=str(row["binding_id"]),
                work_ref=str(row["work_ref"]), event=json.loads(row["event_json"]),
                attempts=int(row["attempts"]) + 1, role=str(row["role"]), lease_token=token,
            )

    def release_work(
        self, work_ref: str, *, reason: str, retry_after_sec: float, child_ref: str = "",
        lease_token: Optional[str] = None,
    ) -> bool:
        now, (held, args) = time.time(), held_by_clause(None, lease_token)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT event_id, binding_id, event_json FROM presence_work WHERE work_ref=? AND state='leased'" + held,
                (work_ref, *args),
            ).fetchone()
            if row is None:
                conn.rollback()
                return False
            if child_ref and child_ref != work_ref:
                event = json.loads(row["event_json"])
                mode = int((event.get("_reporting") or {}).get("version") == 1)
                self._add_work(conn, row["event_id"], row["binding_id"], child_ref, "child", event, mode, now)
            cur = conn.execute(
                "UPDATE presence_work SET state='pending', available_at=?, lease_until=0, lease_token='', last_error=? "
                "WHERE work_ref=? AND state='leased'" + held,
                (now + max(0.0, retry_after_sec), utf8_safe(str(reason)[:500]), work_ref, *args),
            )
            conn.commit()
            return cur.rowcount == 1

    def record_work_outputs(
        self, work_ref: str, outputs: Sequence[Dict[str, Any]], *, lease_token: Optional[str] = None
    ) -> bool:
        held, args = held_by_clause(None, lease_token)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            query = ("SELECT event_id, event_json FROM presence_work WHERE work_ref=? AND state='leased'" + held
                     if lease_token else "SELECT event_id, event_json FROM presence_work WHERE work_ref=?")
            row = conn.execute(query, (work_ref, *args) if lease_token else (work_ref,)).fetchone()
            if row is None:
                conn.rollback()
                if lease_token:
                    return False
                raise ValueError("presence work reference is missing")
            event = json.loads(row["event_json"])
            for output in outputs:
                self._enqueue_output(conn, str(row["event_id"]), work_ref, event, output, time.time())
            conn.commit()
            return True

    def _enqueue_output(
        self,
        conn: sqlite3.Connection,
        event_id: str,
        work_ref: str,
        event: Dict[str, Any],
        output: Dict[str, Any],
        now: float,
    ) -> None:
        output_ref = str(output["output_ref"])
        prefix = f"presence:{event_id}:output:"
        queued = conn.execute(
            "SELECT delivery_id FROM outbox WHERE delivery_id >= ? AND delivery_id < ?",
            (prefix, prefix[:-1] + ";"),
        ).fetchall()
        if any(str(item[0]).endswith(":" + output_ref) for item in queued):
            return
        reply_to = anchor_message_id(conn, event, output.get("reply_anchor"))
        self._enqueue_outbox(
            conn,
            f"{prefix}{work_ref}:{output_ref}",
            reply_payload(event, str(output["text"]), reply_to),
            now,
        )

    def complete_work(
        self,
        work_ref: str,
        *,
        status: str,
        text: str,
        child_ref: str = "",
        reply_anchor: Optional[Dict[str, Any]] = None,
        lease_token: Optional[str] = None,
    ) -> bool:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("work status is not terminal")
        now, (held, args) = time.time(), held_by_clause(None, lease_token)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT event_id, binding_id, role, event_json FROM presence_work WHERE work_ref=? AND state='leased'" + held,
                (work_ref, *args),
            ).fetchone()
            if row is None:
                conn.rollback()
                if lease_token:
                    return False
                raise ValueError("presence work is not leased")
            if child_ref and child_ref != work_ref:
                event = json.loads(row["event_json"])
                mode = int((event.get("_reporting") or {}).get("version") == 1)
                self._add_work(conn, row["event_id"], row["binding_id"], child_ref, "child", event, mode, now)
            updated = conn.execute(
                "UPDATE presence_work SET state=?, lease_until=0, lease_token='', completed_at=?, last_error='' "
                "WHERE work_ref=? AND state='leased'" + held,
                (status, now, work_ref, *args),
            )
            if updated.rowcount != 1:
                conn.rollback()
                if lease_token:
                    return False
                raise ValueError("presence work lease changed")
            if text:
                event_id, role = str(row["event_id"]), str(row["role"])
                delivery_id = (
                    f"presence:{event_id}:work" if role == "deferred" else f"presence:{event_id}:{role}:{work_ref}"
                )
                event = json.loads(row["event_json"])
                self._enqueue_outbox(
                    conn, delivery_id, reply_payload(event, text, anchor_message_id(conn, event, reply_anchor)), now,
                )
            conn.commit()
            return True

    def claim_outbox(self, *, lease_sec: float = 90.0) -> Optional[OutboxLease]:
        now, token = time.time(), self._mint_lease_token()
        with self._connect() as conn:
            row = self._claim_candidate(conn, CLAIMABLE_OUTBOX, now)
            if row is None:
                return None
            payload = json.loads(row["payload_json"])
            unanswered = row["state"] == "leased" or row["last_error"] in UNANSWERED_ATTEMPT
            unrecorded, marked = False, None
            if unmarked_reaction(payload):
                unanswered = unanswered or unproven_earlier_attempt(
                    payload, row["attempts"], row["last_error"], UNMARKED_NO_EFFECT
                )
                if not unanswered:
                    payload = {**payload, REACTION_ATTEMPTS_MARKED: row["attempts"] + 1}
                    marked = compact_json(payload)
            elif recorded_calls(payload) and CALL_STARTED not in payload:
                unrecorded = unproven_earlier_attempt(
                    payload, row["attempts"], row["last_error"], UNRECORDED_SEND_NO_EFFECT
                )
                if not unrecorded:
                    payload = {**payload, REACTION_ATTEMPTS_MARKED: row["attempts"] + 1}
                    marked = compact_json(payload)
            updated = conn.execute(
                "UPDATE outbox SET state='leased', lease_until=?, attempts=attempts+1, lease_token=?, last_error='', "
                "payload_json=COALESCE(?, payload_json) WHERE delivery_id=? AND "
                "((state='pending' AND available_at<=?) OR (state='leased' AND lease_until<=?))",
                (now + max(1.0, lease_sec), token, marked, row["delivery_id"], now, now),
            )
            conn.commit()
            if updated.rowcount != 1:
                return None
            return OutboxLease(
                delivery_id=str(row["delivery_id"]), payload=payload, attempts=int(row["attempts"]) + 1,
                previous_unanswered=unanswered, unrecorded_attempt=unrecorded, lease_token=token,
            )

    def outbox_lease_valid(
        self, delivery_id: str, attempts: Optional[int] = None, lease_token: Optional[str] = None
    ) -> bool:
        held, args = held_by_clause(attempts, lease_token)
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM outbox WHERE delivery_id=? AND state='leased'" + held, (delivery_id, *args)
            ).fetchone() is not None

    def release_outbox(
        self, delivery_id: str, *, reason: str, retry_after_sec: float,
        attempts: Optional[int] = None, lease_token: Optional[str] = None,
    ) -> bool:
        held, args = held_by_clause(attempts, lease_token)
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE outbox SET state='pending', available_at=?, lease_until=0, lease_token='', last_error=? "
                "WHERE delivery_id=? AND state='leased'" + held,
                (time.time() + max(0.0, retry_after_sec), utf8_safe(str(reason)[:500]), delivery_id, *args),
            )
            return updated.rowcount == 1

    def hold_outbox(
        self, delivery_id: str, reason: str, *, retry_after_sec: float = 5.0,
        attempts: Optional[int] = None, lease_token: Optional[str] = None,
    ) -> bool:
        held, args = held_by_clause(attempts, lease_token)
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE outbox SET state='pending',available_at=?,lease_until=0,lease_token='',"
                "last_error=?,attempts=MAX(0,attempts-1),payload_json=CASE WHEN "
                "json_extract(payload_json,?)=attempts THEN json_set(payload_json,?,MAX(0,attempts-1)) "
                "ELSE payload_json END WHERE delivery_id=? AND state='leased'" + held,
                (time.time() + max(0.0, retry_after_sec), utf8_safe(str(reason)[:500]), STAMP_PATH, STAMP_PATH, delivery_id, *args),
            )
            return updated.rowcount == 1

    def mark_outbox_failed(
        self,
        delivery_id: str,
        *,
        reason: str,
        report=None,
        state: str = "failed",
        attempts: Optional[int] = None,
        lease_token: Optional[str] = None,
    ) -> bool:
        if state not in {"failed", "uncertain"}:
            raise ValueError("terminal outbox state must be failed or uncertain")
        held, args = held_by_clause(attempts, lease_token)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = (
                conn.execute(
                    "SELECT payload_json FROM outbox WHERE delivery_id=? AND state='leased'" + held,
                    (delivery_id, *args),
                ).fetchone()
                if state == "uncertain"
                else None
            )
            payload = json.loads(row[0]) if row is not None else None
            if payload is not None:
                payload["_terminal_outcome"] = "uncertain"
            updated = conn.execute(
                "UPDATE outbox SET state='failed', lease_until=0, lease_token='', failed_at=?, last_error=?, "
                "payload_json=COALESCE(?,payload_json) WHERE delivery_id=? AND state='leased'" + held,
                (time.time(), utf8_safe(str(reason)[:500]), compact_json(payload) if payload is not None else None, delivery_id, *args),
            )
            if (attempts is not None or lease_token) and updated.rowcount != 1:
                conn.rollback()
                return False
            self._add_reports(conn, delivery_id, [report] if report else [])
            conn.commit()
            return updated.rowcount == 1

    def start_outbox_call(
        self, delivery_id: str, attempts: int, payload: Dict[str, Any], *, lease_token: Optional[str] = None
    ) -> bool:
        held, args = held_by_clause(attempts, lease_token)
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE outbox SET payload_json=? WHERE delivery_id=? AND state='leased'" + held,
                (compact_json(payload), delivery_id, *args),
            )
            return updated.rowcount == 1

    def checkpoint_outbox(
        self,
        delivery_id: str,
        payload: Dict[str, Any],
        *,
        report=None,
        sent: Optional[Sequence[Any]] = None,
        attempts: Optional[int] = None,
        lease_token: Optional[str] = None,
    ) -> bool:
        reports = report if isinstance(report, list) else [report] if report else []
        rows = sent if isinstance(sent, list) else [sent] if sent else []
        held, args = held_by_clause(attempts, lease_token)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                "UPDATE outbox SET payload_json=? WHERE delivery_id=? AND state='leased'" + held,
                (compact_json(payload), delivery_id, *args),
            )
            if (attempts is not None or lease_token) and updated.rowcount != 1:
                conn.rollback()
                return False
            for row in rows:
                if row:
                    conn.execute(REMEMBER_MESSAGE, row)
            self._add_reports(conn, delivery_id, [entry for entry in reports if entry])
            conn.commit()
            return updated.rowcount == 1

    def mark_delivered(
        self,
        delivery_id: str,
        *,
        provider_receipt: Dict[str, Any],
        report=None,
        attempts: Optional[int] = None,
        lease_token: Optional[str] = None,
    ) -> bool:
        held, args = held_by_clause(attempts, lease_token)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                "UPDATE outbox SET state='delivered', lease_until=0, lease_token='', delivered_at=?, "
                "provider_receipt_json=?, last_error='' WHERE delivery_id=? AND state='leased'" + held,
                (time.time(), compact_json(provider_receipt), delivery_id, *args),
            )
            if (attempts is not None or lease_token) and updated.rowcount != 1:
                conn.rollback()
                return False
            self._add_reports(conn, delivery_id, [report] if report else [])
            conn.commit()
            return updated.rowcount == 1

    def _add_reports(self, conn: sqlite3.Connection, delivery_id: str, reports: list) -> None:
        if not reports:
            return
        row = conn.execute("SELECT reports_json FROM outbox WHERE delivery_id=?", (delivery_id,)).fetchone()
        entries = json.loads(row[0])
        keys = {(x["payload"]["part_id"], x["payload"]["state"]) for x in entries}
        for report in reports:
            key = (report["part_id"], report["state"])
            if key not in keys:
                entries.append(
                    {"payload": report, "acked": False, "attempts": 0, "available_at": time.time(), "error": ""}
                )
                keys.add(key)
        self._write_reports(conn, delivery_id, entries)

    @staticmethod
    def _write_reports(conn: sqlite3.Connection, delivery_id: str, entries: list) -> None:
        due = min((x["available_at"] for x in entries if not x["acked"]), default=0)
        conn.execute(
            "UPDATE outbox SET reports_json=?, report_due_at=? WHERE delivery_id=?",
            (compact_json(entries), due, delivery_id),
        )

    def next_delivery_report(self):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT delivery_id,reports_json FROM outbox WHERE report_due_at>0 AND report_due_at<=? "
                "ORDER BY report_due_at LIMIT 1",
                (time.time(),),
            ).fetchone()
        if row:
            for index, entry in enumerate(json.loads(row["reports_json"])):
                if not entry["acked"] and entry["available_at"] <= time.time():
                    return row["delivery_id"], index, entry["payload"]
        return None

    def finish_delivery_report(self, delivery_id: str, index: int, *, error: str = "") -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            entries = json.loads(
                conn.execute("SELECT reports_json FROM outbox WHERE delivery_id=?", (delivery_id,)).fetchone()[0]
            )
            entry = entries[index]
            entry["attempts"] += 1
            entry.update(
                acked=not error,
                error=error,
                available_at=time.time() + min(300, 2 ** min(entry["attempts"], 8)),
            )
            self._write_reports(conn, delivery_id, entries)
            conn.commit()

    def reaction_target(self, account_id: str, chat_id: str, message_id: int) -> Dict[str, Any]:
        with self._connect() as conn:
            row = retained_message(conn, account_id, chat_id, message_id)
            if row is not None:
                mg = str(row["media_group_id"])
                earlier = (
                    int(conn.execute(ALBUM_EARLIER, (str(account_id), str(chat_id), mg, int(message_id))).fetchone()[0])
                    if mg else 0
                )
                return {
                    "retained": True, "topic_id": str(row["topic_id"]),
                    "media_group_id": mg, "is_self": bool(row["is_self"]), "album_earlier": earlier,
                    "album_origin_known": bool(row["album_origin_known"]),
                }
            return {"retained": False, "forum": newest_chat_forum(conn, account_id, chat_id)}

    def outbox_payload(self, delivery_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM outbox WHERE delivery_id=?", (delivery_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def delivery_receipt(self, delivery_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state, attempts, delivered_at, provider_receipt_json, last_error, payload_json, reports_json "
                "FROM outbox WHERE delivery_id=?",
                (delivery_id,),
            ).fetchone()
        if row is None:
            return {"state": "not_found"}
        return build_delivery_receipt(row, stored_answers_fn=stored_answers, outbox_files_mod=outbox_files)

    def offered_groups(self, account_id: str) -> List[Dict[str, str]]:
        with self._connect() as conn:
            return discovery.offered_groups(conn, account_id)

    def status_snapshot(self) -> Dict[str, Any]:
        with self._connect() as conn:
            return build_status_snapshot(
                conn,
                telegram_offset=self._metadata_int(conn, "telegram_offset"),
                backfill_unreadable=discovery.backfill_unreadable(conn),
            )

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(schema_script())
            conn.execute("BEGIN IMMEDIATE")
            if "role" not in {r[1] for r in conn.execute("PRAGMA table_info(presence_work)")}:
                self._key_work_by_reference(conn)
            migrate_custody_schema(conn)
            conn.commit()
        discovery.backfill_legacy(self._connect)

    @staticmethod
    def _key_work_by_reference(conn: sqlite3.Connection) -> None:
        conn.execute(PRESENCE_WORK_TABLE.format(name="presence_work_by_ref"))
        conn.execute(
            """
            INSERT INTO presence_work_by_ref(
                work_ref, event_id, binding_id, role, event_json, state, attempts,
                available_at, lease_until, created_at, completed_at, last_error
            )
            SELECT work_ref, event_id, binding_id, 'deferred', event_json, state, attempts,
                   available_at, lease_until, created_at, completed_at, last_error
            FROM presence_work
            """
        )
        conn.execute("DROP TABLE presence_work")
        conn.execute("ALTER TABLE presence_work_by_ref RENAME TO presence_work")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS presence_work_ready_idx "
            "ON presence_work(state, available_at, lease_until, created_at)"
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._conn is None:
                self._conn = self._open()
            conn = self._conn
            try:
                yield conn
            finally:
                if conn.in_transaction:
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        self._conn = None
                        conn.close()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None, check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
        except BaseException:
            conn.close()
            raise
        return conn

    @staticmethod
    def _metadata_int(conn: sqlite3.Connection, key: str) -> int:
        row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if row is None:
            return 0
        try:
            return max(0, int(row["value"]))
        except (TypeError, ValueError):
            return 0


__all__ = [
    "CALL_STARTED",
    "CustodyStore",
    "InboxLease",
    "OutboxLease",
    "REACTION_WITHHELD",
    "WorkLease",
    "chat_forum",
    "recorded_calls",
]
