"""The dialogue in ``state/dialog.sqlite``: exact chat/progress JSON and one writer seam.

Stream/chat/task/lineage/direction/kind/client/origin facts index JSON. ``seq``/``length`` retain original
byte spans (newline/CRLF included), even after revision: history ids/cursors/wake boundaries stay ordered.
``ordinal`` counts objects; ``pos`` counts non-A2A chat for task-events/logs/chronicle, relative to import
segments. Newest-first ``import_ledger`` settles absolute bases; earlier answers say importing. Skips and
legacy (gen, line) aliases stay indexed; room revisions inherit the project's visible_revision at cutover.
``append`` atomically writes rows, room revision/last row/activity and named inbound dispatch (UNIQUE
chat_id/client_message_id). Dispatch: NULL for unnamed/imported, none for named acceptance without handoff,
else pending/dispatched/completed/abandoned. ``message_ingress`` claims once. Required writers use WAIT_SEC;
best-effort uses zero wait and content-addressed dialogue_retry, transactionally deduped without live replay.
Reads union bounded room/task/owner probes; Main excludes Project rooms; below-import coverage is unknown.
Separate from current.sqlite waits: store errors, thread handles and header lock tier (PERSIST+FULL/gated WAL).
``dialogue_import`` owns cutover, frozen archives and rollback export; text revisions retain all other facts.
"""

from __future__ import annotations

import contextlib
import heapq
import json
import logging
import os
import pathlib
import sqlite3
import uuid
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple

from ouroboros import store, store_empty
from ouroboros.store import StoreBusy, StoreError, StoreLost, StoreMigrating, TIER_ENFORCED, TIER_NAME
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

STORE_REL = pathlib.Path("state/dialog.sqlite")
LOCK_NAME = "dialog.lock"  # the name tier's one lock around every access
IMPORT_LOCK_NAME = "dialog_import.lock"  # creation, cutover, tier changes
SCHEMA_VERSION = 2  # the file's ``user_version``; a file of another version is refused, never replaced
STREAMS = ("chat", "progress")
LIVE_REL = {"chat": pathlib.Path("logs/chat.jsonl"), "progress": pathlib.Path("logs/progress.jsonl")}  # the frozen chains
WAIT_SEC = store.STORE_WAIT_SEC
# The header's application id names the lock protocol before the file is opened ('OUDE'/'OUDN').
_APPLICATION_ID = {TIER_ENFORCED: 0x4F554445, TIER_NAME: 0x4F55444E}
DISPATCH_STATES = ("none", "pending", "dispatched", "completed", "abandoned")
# Every probe names its index (``INDEXED BY``): the plan is a fact of the schema, not the planner's
# estimate -- without statistics SQLite walks the primary key (every newer row of every room) instead.
_PROBE_INDEX = {"chat_id": "dialogue_room", "task_id": "dialogue_task", "parent_task_id": "dialogue_parent",
                "root_task_id": "dialogue_root", "subagent_task_id": "dialogue_subagent", "origin": "dialogue_origin",
                "client_message_id": "dialogue_named", "kind": "dialogue_kind"}
_TASK_KEYS = ("task_id", "parent_task_id", "root_task_id", "subagent_task_id")  # every identifier a task's row may carry
SKIPPED_KINDS = ("invalid", "non_object")  # a line that is not JSON; valid JSON that is not an object
_SCHEMA = """
CREATE TABLE dialogue (
  stream TEXT NOT NULL, seq INTEGER NOT NULL, length INTEGER NOT NULL, segment INTEGER NOT NULL,
  ordinal INTEGER NOT NULL, pos INTEGER, ts TEXT NOT NULL DEFAULT '', chat_id INTEGER,
  task_id TEXT, parent_task_id TEXT, root_task_id TEXT, subagent_task_id TEXT, direction TEXT NOT NULL DEFAULT '',
  kind TEXT, client_message_id TEXT, origin TEXT, dispatch TEXT, dispatch_input TEXT, body TEXT NOT NULL,
  PRIMARY KEY (stream, seq)) WITHOUT ROWID;
CREATE INDEX dialogue_room ON dialogue(stream, chat_id, seq);
CREATE INDEX dialogue_room_ts ON dialogue(stream, chat_id, ts);
CREATE INDEX dialogue_task ON dialogue(stream, task_id, seq) WHERE task_id IS NOT NULL;
CREATE INDEX dialogue_parent ON dialogue(stream, parent_task_id, seq) WHERE parent_task_id IS NOT NULL;
CREATE INDEX dialogue_root ON dialogue(stream, root_task_id, seq) WHERE root_task_id IS NOT NULL;
CREATE INDEX dialogue_subagent ON dialogue(stream, subagent_task_id, seq) WHERE subagent_task_id IS NOT NULL;
CREATE INDEX dialogue_ordinal ON dialogue(segment, ordinal);
CREATE INDEX dialogue_pos ON dialogue(segment, pos) WHERE pos IS NOT NULL;
CREATE INDEX dialogue_kind ON dialogue(stream, kind, seq) WHERE kind IS NOT NULL;
CREATE INDEX dialogue_named ON dialogue(stream, client_message_id, chat_id, seq) WHERE client_message_id IS NOT NULL;
CREATE INDEX dialogue_origin ON dialogue(stream, origin, seq) WHERE origin IS NOT NULL;
CREATE INDEX dialogue_inbound ON dialogue(stream, chat_id, seq) WHERE direction='in';
CREATE UNIQUE INDEX dialogue_accepted ON dialogue(chat_id, client_message_id)
  WHERE direction='in' AND client_message_id IS NOT NULL AND dispatch IS NOT NULL;
CREATE INDEX dialogue_pending ON dialogue(stream, seq) WHERE dispatch='pending';
CREATE TABLE streams (stream TEXT PRIMARY KEY, next_seq INTEGER NOT NULL, cutover_seq INTEGER, files_seq INTEGER NOT NULL DEFAULT 0,
  live INTEGER NOT NULL, next_ordinal INTEGER NOT NULL DEFAULT 0, next_pos INTEGER NOT NULL DEFAULT 0);
CREATE TABLE rooms (chat_id INTEGER PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0, last_seq INTEGER,
  read_revision INTEGER NOT NULL DEFAULT 0, activity_at TEXT NOT NULL DEFAULT '');
CREATE TABLE legacy_addresses (gen TEXT NOT NULL, line INTEGER NOT NULL, seq INTEGER NOT NULL, PRIMARY KEY (gen, line, seq)) WITHOUT ROWID;
CREATE INDEX legacy_seq ON legacy_addresses(seq);
CREATE TABLE skipped (stream TEXT NOT NULL, seq INTEGER NOT NULL, length INTEGER NOT NULL, kind TEXT NOT NULL,
  PRIMARY KEY (stream, seq)) WITHOUT ROWID;
CREATE TABLE import_ledger (id INTEGER PRIMARY KEY, stream TEXT NOT NULL, name TEXT NOT NULL, base INTEGER NOT NULL,
  size INTEGER NOT NULL, mtime_ns INTEGER, state TEXT NOT NULL, rows INTEGER, stream_rows INTEGER, bad INTEGER, gen TEXT,
  ordinal_base INTEGER, pos_base INTEGER, imported_at TEXT, UNIQUE (stream, name));
CREATE INDEX ledger_state ON import_ledger(stream, state, base);
CREATE INDEX ledger_gen ON import_ledger(gen) WHERE gen IS NOT NULL;
CREATE INDEX ledger_ordinal ON import_ledger(stream, ordinal_base) WHERE ordinal_base IS NOT NULL;
CREATE INDEX ledger_pos ON import_ledger(stream, pos_base) WHERE pos_base IS NOT NULL;
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Row(NamedTuple):
    """Coordinate, decoded body, dispatch and allocated span; ``end`` is the next boundary."""

    stream: str
    seq: int
    row: Dict[str, Any]
    dispatch: Optional[str]
    length: int

    @property
    def history_id(self) -> str:
        return f"{self.stream}:{self.seq}"

    @property
    def end(self) -> int:
        return self.seq + self.length


class Receipt(NamedTuple):
    """Append's coordinate, stored row, duplicate flag, dispatch and allocated span."""

    stream: str
    seq: int
    row: Dict[str, Any]
    duplicate: bool
    dispatch: Optional[str]
    length: int

    @property
    def end(self) -> int:
        return self.seq + self.length


# --- the file, its tier and its transactions -------------------------------------------------

def path(root: Any) -> pathlib.Path:
    return pathlib.Path(root) / STORE_REL


def present(root: Any) -> bool:
    return store.file_identity(path(root)) is not None


def encode(row: Dict[str, Any]) -> str:
    """The exact text a chain line held: ``append_jsonl``'s own encoding, without the newline."""
    return json.dumps(row, ensure_ascii=False)


def decode(body: str) -> Dict[str, Any]:
    value = json.loads(body)
    return value if isinstance(value, dict) else {}


def _typed(exc: sqlite3.Error) -> StoreError:
    if isinstance(exc, sqlite3.OperationalError) and store.busy(exc):
        error = StoreBusy(f"dialogue store busy: {exc}", reason="contention")
        if hasattr(exc, "rollback_confirmed"):
            error.rollback_confirmed = exc.rollback_confirmed
        return error
    if isinstance(exc, sqlite3.DatabaseError) and not isinstance(
            exc, (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.ProgrammingError)):
        return StoreLost(f"dialogue store unreadable: {exc}")
    return StoreError(f"dialogue store unavailable: {exc}")


def probed_tier(root: Any) -> str:
    """The tier today's kernel enforces for this root (the boot re-probe, as ``current.sqlite``)."""
    from ouroboros.platform_layer import kernel_file_locks_enforced

    return TIER_ENFORCED if kernel_file_locks_enforced(pathlib.Path(root) / "state" / LOCK_NAME) else TIER_NAME


def header_tier(root: Any) -> Optional[str]:
    """The tier the file's header names; ``None`` for a missing or foreign file."""
    return store.header_tier(path(root), _APPLICATION_ID)


def _tier(root: pathlib.Path) -> str:
    tier = header_tier(root)
    if tier is None:
        if present(root):
            raise StoreLost(f"dialogue store unreadable or foreign, never replaced: {path(root)}")
        raise StoreMigrating(f"dialogue store not created yet: {path(root)}")
    return tier


def _open(path: pathlib.Path, tier: str, **kwargs: Any) -> sqlite3.Connection:
    """``store.connect`` plus the schema gate: the header's ``user_version`` is this body's
    ``SCHEMA_VERSION`` or the file is refused typed (``StoreLost``) -- there is no migration of a
    dialogue store; it is never replaced either."""
    conn = store.connect(path, tier, **kwargs)
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.Error:
        conn.close()
        raise
    if version != SCHEMA_VERSION:
        conn.close()
        raise StoreLost(f"dialogue store schema {version}, this body needs {SCHEMA_VERSION}, never replaced: {path}")
    return conn


def hand_off(root: Any) -> bool:
    """``store.hand_off`` for the dialogue store: a WAL file is checkpointed into DELETE mode before
    another body takes the root (a rollback-journal file is left untouched)."""
    target = path(root)
    if not store.wal_file(target):
        return False
    store.prepare_handoff(target, _tier(pathlib.Path(root)))  # WAL is never chosen on the name tier
    return True


def holds_dialogue_files(root: Any) -> bool:
    """Whether the root holds chain files the cutover would import (live rows or archive segments)."""
    root = pathlib.Path(root)
    for stream in STREAMS:
        try:
            if (root / LIVE_REL[stream]).stat().st_size:
                return True
        except OSError:
            pass
    try:
        with os.scandir(root / "archive") as entries:
            return any(entry.name.startswith(("chat_", "progress_")) and entry.name.endswith(".jsonl") for entry in entries)
    except OSError:
        return False


def create(root: Any, *, tier: Optional[str] = None) -> pathlib.Path:
    """The empty store, built as a sibling and published by rename (idempotent; under the import lock)."""
    root = pathlib.Path(root)
    target = path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with store.process_lock(target), store.name_lock(root, IMPORT_LOCK_NAME, timeout_sec=WAIT_SEC, stale_sec=600.0):
        if present(root):
            return target
        tier = tier or probed_tier(root)
        tmp = target.with_name(f"{target.name}.import-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        try:
            with contextlib.closing(store.connect(tmp, tier, create=True)) as conn:
                # Keep DDL, header and seed rows in the same durable commit.
                conn.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA)
                conn.execute(f"PRAGMA application_id = {_APPLICATION_ID[tier]}")
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                for stream in STREAMS:
                    # The stream's live unit: every row stored after the cutover counts in it.
                    live = conn.execute("INSERT INTO import_ledger (stream, name, base, size, state) VALUES (?, ?, 0, 0, 'live')",
                                        (stream, LIVE_REL[stream].name)).lastrowid
                    conn.execute("INSERT INTO streams (stream, next_seq, live) VALUES (?, 0, ?)", (stream, live))
                for key, value in (("schema_version", SCHEMA_VERSION), ("lock_tier", tier), ("created_at", utc_now_iso())):
                    conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value)))
                store.commit(conn)
            os.replace(tmp, target)
        except sqlite3.Error as exc:
            raise _typed(exc) from exc
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        store.fsync_directory(target.parent)
    return target


def retier(root: Any) -> Optional[str]:
    """Bring the header to today's probed tier (the boot re-probe); returns the former tier when it changed.
    The file leaves WAL under the former tier first: the new one is published only on a DELETE file."""
    root = pathlib.Path(root)
    wanted, current = probed_tier(root), header_tier(root)
    if current is None or current == wanted:
        return None
    with store.process_lock(path(root)), store.name_lock(root, IMPORT_LOCK_NAME, timeout_sec=WAIT_SEC, stale_sec=600.0):
        try:
            store.prepare_handoff(path(root), current, wait_sec=WAIT_SEC, seam=False)  # closes this process's handles
            with contextlib.closing(store.connect(path(root), current, wait_sec=WAIT_SEC)) as conn:
                conn.execute(f"PRAGMA application_id = {_APPLICATION_ID[wanted]}")
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('lock_tier', ?)", (json.dumps(wanted),))
                store.commit(conn)
        except sqlite3.Error as exc:
            raise _typed(exc) from exc
    return current


def ensure(root: Any) -> pathlib.Path:
    """The store ready for use: present, or created and cut over from this root's own chain
    files (the first row or read of a child root pays for its own small files, never a boot;
    the main root is cut over by the boot coordinator before the dialogue opens)."""
    if not present(root):
        from ouroboros.dialogue_import import cutover

        cutover(root)
    return path(root)


@contextlib.contextmanager
def transaction(root: Any, *, write: bool, wait_sec: float = WAIT_SEC) -> Iterator[sqlite3.Connection]:
    """One transaction on the root's store (``store.connection`` + ``store.transaction``): the name
    lock around a name-tier access, ``StoreBusy`` on contention, the cutover for a root whose
    chain files were never cut over (its own files; unknown is never answered empty), an empty
    in-memory schema for a read of a root that holds no dialogue at all."""
    wait_sec = store.bounded_wait(wait_sec)
    root = pathlib.Path(root)
    if not present(root):
        if write or holds_dialogue_files(root):
            ensure(root)
        else:
            with store_empty.empty_store(_SCHEMA) as conn, store.transaction(conn, write=False):
                yield conn
            return
    tier = _tier(root)
    with contextlib.ExitStack() as stack:
        if tier == TIER_NAME:
            stack.enter_context(store.name_lock(root, LOCK_NAME, timeout_sec=wait_sec))
        try:
            conn = stack.enter_context(store.connection(path(root), tier, wait_sec=wait_sec, opener=_open))
            with store.transaction(conn, write=write, commit_wait_sec=wait_sec if tier == TIER_ENFORCED else None):
                yield conn
        except sqlite3.Error as exc:
            raise _typed(exc) from exc


def meta(root: Any, key: str) -> Any:
    with transaction(root, write=False) as conn:
        return _meta(conn, key)


def _meta(conn: sqlite3.Connection, key: str) -> Any:
    record = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if record is None else json.loads(record["value"])


def _set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))


_PLANNED: set = set()  # stores (by path) whose cutover fact this process has seen: it is never removed


def _plan_first(root: pathlib.Path, wait_sec: float = WAIT_SEC) -> None:
    """Finish a cutover that failed after ``create`` published the store: before a row takes a
    coordinate (the plan places the chain files first) and before an accepted id is looked up (the
    live file it rotated holds recent ids, found only once the plan names it pending)."""
    if str(path(root)) in _PLANNED or not present(root):
        return
    with transaction(root, write=False, wait_sec=wait_sec) as conn:
        planned = _planned(conn, root)
    if not planned:
        from ouroboros.dialogue_import import cutover

        cutover(root)


def _planned(conn: sqlite3.Connection, root: Any) -> bool:
    """Whether the store holds its cutover fact: the chain files were planned and the live tail
    stored. ``create`` publishes the store before that, so a cutover that failed after it leaves a
    store whose coordinates are not planned yet (``append`` finishes it, readers say importing)."""
    key = str(path(root))
    if key in _PLANNED or not present(root):  # absent: a root without chain files, read in memory
        return True
    if _meta(conn, "cutover") is None:
        return False
    _PLANNED.add(key)
    return True


# --- the row's typed facts -------------------------------------------------------------------

def facts(row: Dict[str, Any]) -> Dict[str, Any]:
    """The columns a row is addressed by, read from the body as every chain reader read them
    (an absent or empty key is ``None``: the partial indexes hold only rows that carry the fact)."""
    chat_id: Optional[int]
    try:
        chat_id = int(row["chat_id"]) if row.get("chat_id") not in (None, "") else None
    except (TypeError, ValueError):
        chat_id = None

    def text(key: str) -> str:
        value = row.get(key)
        return "" if value is None or isinstance(value, (dict, list)) else str(value)

    origin = None
    ref = row.get("origin_message_ref")
    if isinstance(ref, dict) and ref.get("client_message_id"):
        try:
            origin = f"{int(ref.get('chat_id') or 0)}:{ref['client_message_id']}"
        except (TypeError, ValueError):
            origin = None
    return {"ts": text("ts"), "chat_id": chat_id, "task_id": text("task_id") or None,
            "parent_task_id": text("parent_task_id") or None, "root_task_id": text("root_task_id") or None,
            "subagent_task_id": text("subagent_task_id") or None, "direction": text("direction"),
            "kind": text("type") or None, "client_message_id": text("client_message_id") or None, "origin": origin}


def stream_row(stream: str, columns: Dict[str, Any]) -> bool:
    """Whether the row takes a chat stream position (``pos``): a chat row outside A2A traffic, as
    ``chat_chain._stream_rows`` and the consolidation cursor counted."""
    return stream == "chat" and (columns["chat_id"] is None or columns["chat_id"] >= 0)


def _row(record: sqlite3.Row) -> Row:
    from ouroboros.place_arrivals import project_source
    return Row(record["stream"], record["seq"], project_source(record), record["dispatch"], record["length"])


_SELECT = "SELECT stream, seq, body, dispatch, dispatch_input, length FROM dialogue"


# --- the one writer seam ---------------------------------------------------------------------

_INSERT = ("INSERT{} INTO dialogue (stream, seq, length, segment, ordinal, pos, ts, chat_id, task_id, parent_task_id, "
           "root_task_id, subagent_task_id, direction, kind, client_message_id, origin, dispatch, dispatch_input, body) "
           "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")


def row_values(stream: str, seq: int, length: int, segment: int, ordinal: int, pos: Optional[int], columns: Dict[str, Any],
               body: str, dispatch: Optional[str] = None, dispatch_input: Optional[Dict[str, Any]] = None) -> Tuple[Any, ...]:
    """The ``_INSERT`` parameters of one row (``append`` and the importer share the column order)."""
    return (stream, seq, length, segment, ordinal, pos, columns["ts"], columns["chat_id"], columns["task_id"],
            columns["parent_task_id"], columns["root_task_id"], columns["subagent_task_id"], columns["direction"],
            columns["kind"], columns["client_message_id"], columns["origin"], dispatch,
            None if dispatch_input is None else json.dumps(dispatch_input, ensure_ascii=False), body)


def insert_row(conn: sqlite3.Connection, stream: str, seq: int, row: Dict[str, Any], body: str, *, length: int,
               segment: int, ordinal: int, pos: Optional[int], dispatch: Optional[str] = None,
               dispatch_input: Optional[Dict[str, Any]] = None, revision: int = 0) -> None:
    """One row and its room facts inside the caller's transaction (``append`` and the importer)."""
    columns = facts(row)
    conn.execute(_INSERT.format(""), row_values(stream, seq, length, segment, ordinal, pos, columns, body, dispatch, dispatch_input))
    if columns["chat_id"] is not None:
        touch_room(conn, columns["chat_id"], seq if stream == "chat" else None, columns["ts"], revision)


def touch_room(conn: sqlite3.Connection, chat_id: int, last_seq: Optional[int], activity_at: str, revision: int) -> None:
    conn.execute(
        "INSERT INTO rooms (chat_id, revision, last_seq, read_revision, activity_at) VALUES (?, ?, ?, 0, ?) "
        "ON CONFLICT(chat_id) DO UPDATE SET revision = rooms.revision + excluded.revision, "
        "last_seq = CASE WHEN excluded.last_seq IS NULL THEN rooms.last_seq "
        "ELSE max(coalesce(rooms.last_seq, -1), excluded.last_seq) END, "
        "activity_at = max(rooms.activity_at, excluded.activity_at)",
        (chat_id, revision, last_seq, activity_at))


def _existing_named(conn: sqlite3.Connection, chat_id: Optional[int], client_message_id: str) -> Optional[sqlite3.Row]:
    """The newest inbound row the id names in this chat: the answer a full chain scan gave."""
    return conn.execute(
        f"{_SELECT} INDEXED BY dialogue_named WHERE stream='chat' AND client_message_id=? AND chat_id IS ? "
        "AND direction='in' ORDER BY seq DESC LIMIT 1", (client_message_id, chat_id)).fetchone()


def append(root: Any, stream: str, row: Dict[str, Any], *, dispatch: Optional[Dict[str, Any]] = None,
           feed: bool = False, wait_sec: float = WAIT_SEC, idempotent: bool = False,
           emit_live: bool = True) -> Receipt:
    """Store exact JSON, room facts and named dispatch in one transaction.

    Named inbound duplicates return the existing receipt (the UNIQUE index also
    settles concurrent writers). Dispatch input is JSON without bytes; presence
    means pending, absence means none. Feed advances the unread revision. Optional
    idempotence compares content identity under the existing room/time index.
    Store failures propagate; append_quietly owns best-effort retry and live fallback.
    """
    if stream not in STREAMS:
        raise ValueError(f"unknown dialogue stream: {stream!r}")
    body = encode(row)
    columns = facts(row)
    named = columns["direction"] == "in" and bool(columns["client_message_id"])
    state = None if not named else ("pending" if dispatch is not None else "none")
    root = pathlib.Path(root)
    length = len(body.encode("utf-8")) + 1
    counted = stream_row(stream, columns)
    _plan_first(root, wait_sec)
    with transaction(root, write=True, wait_sec=wait_sec) as conn:
        if idempotent:
            from ouroboros.chat_chain import source_row_id

            digest = source_row_id(row)
            candidates = conn.execute(
                f"{_SELECT} INDEXED BY dialogue_room_ts WHERE stream=? AND chat_id IS ? AND ts=?",
                (stream, columns["chat_id"], columns["ts"]))
            for existing in candidates:
                if source_row_id(decode(existing["body"])) == digest:
                    return _receipt(existing)
        if named:
            existing = _existing_named(conn, columns["chat_id"], columns["client_message_id"])
            if existing is not None:
                return _receipt(existing)
        marks = conn.execute("SELECT next_seq, live, next_ordinal, next_pos FROM streams WHERE stream=?", (stream,)).fetchone()
        seq = int(marks["next_seq"])
        try:
            insert_row(conn, stream, seq, row, body, length=length, segment=int(marks["live"]), ordinal=int(marks["next_ordinal"]),
                       pos=int(marks["next_pos"]) if counted else None, dispatch=state, dispatch_input=dispatch,
                       revision=int(bool(feed)))
        except sqlite3.IntegrityError:
            existing = _existing_named(conn, columns["chat_id"], columns["client_message_id"]) if named else None
            if existing is None:
                raise
            return _receipt(existing)
        conn.execute("UPDATE streams SET next_seq=?, next_ordinal=next_ordinal+1, next_pos=next_pos+? WHERE stream=?",
                     (seq + length, int(counted), stream))
    if stream == "progress" and emit_live:
        _stream_live(row)
    return Receipt(stream, seq, row, False, state, length)


def revise_text(root: Any, stream: str, seq: int, text: str) -> Dict[str, Any]:
    """Revise a row's text atomically; retain its allocated span (mapped on export) and all other facts."""
    with transaction(root, write=True) as conn:
        existing = conn.execute(f"{_SELECT} WHERE stream=? AND seq=?", (stream, seq)).fetchone()
        if existing is None:
            raise ValueError("dialogue row missing for text revision")
        row = decode(existing["body"])
        if row.get("text") != text:
            row["text"] = text
            conn.execute("UPDATE dialogue SET body=? WHERE stream=? AND seq=?", (encode(row), stream, seq))
            columns = facts(row)
            if columns["chat_id"] is not None:
                touch_room(conn, columns["chat_id"], None, columns["ts"], 1)
        return row


def _receipt(existing: sqlite3.Row) -> Receipt:
    return Receipt(existing["stream"], existing["seq"], decode(existing["body"]), True, existing["dispatch"], existing["length"])


def append_quietly(root: Any, stream: str, row: Dict[str, Any], **kwargs: Any) -> Optional[Receipt]:
    """Try once without waiting; retain failures for content-addressed retry.

    Progress reaches the live sink even under contention. Retry never emits it again.
    Required inbound/terminal writers use ``append`` and never enter this buffer.
    """
    from ouroboros import dialogue_retry

    dialogue_retry.before_append()
    try:
        with store.acquisition_budget(0):
            return append(root, stream, row, **kwargs)
    except (StoreError, OSError, ValueError):
        if stream == "progress":
            _stream_live(row)
        dialogue_retry.retain(root, stream, row, feed=kwargs.get("feed", False))
        return None


def _stream_live(row: Dict[str, Any]) -> None:
    """The live frame a progress row's writer owed through ``append_jsonl``'s sink (``utils.set_log_sink``):
    the file is gone, the one live frame is not. Chat rows have their own live channel."""
    from ouroboros import utils

    sink = getattr(utils, "_log_sink", None)
    if sink is not None:
        try:
            sink(row)
        except Exception:
            pass


# --- the dispatch obligation -----------------------------------------------------------------

def dispatch_facts(root: Any, chat_id: Optional[int], client_message_id: str) -> Optional[Dict[str, Any]]:
    """The accepted row's dispatch facts: ``state`` and, once handed over, the ``process`` (the
    custody session that took it) and ``dispatcher``; ``None`` for no row, an imported row or
    an unnamed one."""
    if not client_message_id or not present(root):
        return None
    with transaction(root, write=False) as conn:
        record = conn.execute(
            "SELECT dispatch, dispatch_input FROM dialogue INDEXED BY dialogue_named WHERE stream='chat' AND client_message_id=? "
            "AND chat_id IS ? AND direction='in' ORDER BY seq DESC LIMIT 1", (client_message_id, chat_id)).fetchone()
    if record is None or record["dispatch"] is None:
        return None
    recorded = json.loads(record["dispatch_input"] or "{}")
    return {"state": record["dispatch"], "process": str(recorded.get("process") or ""),
            "dispatcher": str(recorded.get("dispatcher") or "")}


def dispatch_state(root: Any, chat_id: Optional[int], client_message_id: str) -> Optional[str]:
    facts_ = dispatch_facts(root, chat_id, client_message_id)
    return None if facts_ is None else facts_["state"]


def _move_dispatch(root: Any, chat_id: Optional[int], client_message_id: str, *, states: Tuple[str, ...],
                   process: str) -> bool:
    if not client_message_id or not present(root):
        return False
    with transaction(root, write=True) as conn:
        cursor = conn.execute(
            "UPDATE dialogue INDEXED BY dialogue_named SET dispatch='dispatched', "
            "dispatch_input=json_set(coalesce(dispatch_input, '{}'), '$.process', ?) WHERE stream='chat' AND client_message_id=? "
            f"AND chat_id IS ? AND direction='in' AND dispatch IN ({','.join('?' * len(states))})",
            (process, client_message_id, chat_id, *states))
        return cursor.rowcount == 1


def claim_dispatch(root: Any, chat_id: Optional[int], client_message_id: str, *, process: str = "") -> bool:
    """Take the owed dispatch of an accepted row (pending -> dispatched): the durable "entered"
    mark, with the ``process`` (custody session) that takes it. Exactly one caller wins -- the
    acceptor, a same-id retry or the recovery."""
    return _move_dispatch(root, chat_id, client_message_id, states=("pending",), process=process)


def note_dispatched(root: Any, chat_id: Optional[int], client_message_id: str, *, process: str = "") -> bool:
    """The entry mark of a named acceptance that owed no bus dispatch (``none`` -> dispatched):
    its caller hands it to its own dispatcher now."""
    return _move_dispatch(root, chat_id, client_message_id, states=("none", "pending"), process=process)


def complete_dispatch(root: Any, chat_id: Optional[int], client_message_id: str) -> bool:
    """The handoff returned (dispatched -> completed; the recorded input shrinks to the dispatcher
    and the process); a callback that raised after entry leaves ``dispatched``, which is
    ambiguous and never replayed."""
    if not client_message_id or not present(root):
        return False
    with transaction(root, write=True) as conn:
        cursor = conn.execute(
            "UPDATE dialogue INDEXED BY dialogue_named SET dispatch='completed', dispatch_input=json_object("
            "'dispatcher', json_extract(dispatch_input, '$.dispatcher'), 'process', json_extract(dispatch_input, '$.process')) "
            "WHERE stream='chat' AND client_message_id=? AND chat_id IS ? AND direction='in' AND dispatch='dispatched'",
            (client_message_id, chat_id))
        return cursor.rowcount == 1


def abandon_dispatch(root: Any, chat_id: Optional[int], client_message_id: str, reason: str) -> bool:
    """A pending obligation nobody can dispatch (its inputs are gone): the typed outcome."""
    if not present(root):
        return False
    with transaction(root, write=True) as conn:
        cursor = conn.execute(
            "UPDATE dialogue INDEXED BY dialogue_named SET dispatch='abandoned', "
            "dispatch_input=json_set(coalesce(dispatch_input, '{}'), '$.abandoned', ?) "
            "WHERE stream='chat' AND client_message_id=? AND chat_id IS ? AND direction='in' AND dispatch='pending'",
            (reason, client_message_id, chat_id))
        return cursor.rowcount == 1


def pending_dispatches(root: Any) -> List[Tuple[Row, Dict[str, Any]]]:
    """Every accepted row whose dispatch is still owed, oldest first, with the input it recorded
    (the partial index ``dialogue_pending``; no chain is read)."""
    if not present(root):
        return []
    with transaction(root, write=False) as conn:
        records = conn.execute("SELECT stream, seq, body, dispatch, length, dispatch_input FROM dialogue INDEXED BY dialogue_pending "
                               "WHERE stream='chat' AND dispatch='pending' ORDER BY seq").fetchall()
    return [(_row(record), json.loads(record["dispatch_input"] or "{}")) for record in records]


# --- addressed reads -------------------------------------------------------------------------

def row_at(root: Any, stream: str, seq: int) -> Optional[Row]:
    with transaction(root, write=False) as conn:
        record = conn.execute(f"{_SELECT} WHERE stream=? AND seq=?", (stream, seq)).fetchone()
    return None if record is None else _row(record)


def rows_between(root: Any, stream: str, lower: int, upper: int) -> List[Row]:
    """Every row with ``lower <= seq < upper`` in order (a frozen page replayed by its coordinates)."""
    with transaction(root, write=False) as conn:
        records = conn.execute(f"{_SELECT} WHERE stream=? AND seq>=? AND seq<? ORDER BY seq", (stream, lower, upper)).fetchall()
    return [_row(record) for record in records]


def rows_after(root: Any, stream: str, seq: int, *, limit: int = 1000) -> List[Row]:
    """The rows appended after ``seq`` (a live follower's next batch, a wake's window), in order."""
    with transaction(root, write=False) as conn:
        records = conn.execute(f"{_SELECT} WHERE stream=? AND seq>? ORDER BY seq LIMIT ?", (stream, seq, limit)).fetchall()
    return [_row(record) for record in records]


def iter_rows(root: Any, stream: str, *, lower: int = 0, upper: Optional[int] = None, batch: int = 500) -> Iterator[Row]:
    """Every row with ``lower <= seq < upper`` in order, one short read transaction per
    ``batch`` rows (a whole-stream reader -- consolidation, the chronicle import, an audit --
    never holds the store's read lock across its own work)."""
    bound = _bound(upper)
    while True:
        with transaction(root, write=False) as conn:
            records = conn.execute(f"{_SELECT} WHERE stream=? AND seq>=? AND seq<? ORDER BY seq LIMIT ?",
                                   (stream, lower, bound, batch)).fetchall()
        for record in records:
            yield _row(record)
        if len(records) < batch:
            return
        lower = records[-1]["seq"] + 1


def last_inbound(root: Any, chat_id: Optional[int], *, before: Optional[int] = None) -> Optional[Row]:
    """The newest inbound (owner) row of one chat below ``before`` (the partial index
    ``dialogue_inbound``: the chat's own inbound rows, not its outbound history)."""
    with transaction(root, write=False) as conn:
        record = conn.execute(f"{_SELECT} INDEXED BY dialogue_inbound WHERE stream='chat' AND chat_id IS ? AND direction='in' "
                              "AND seq<? ORDER BY seq DESC LIMIT 1", (chat_id, _bound(before))).fetchone()
    return None if record is None else _row(record)


def newest(root: Any, stream: str, *, limit: int, before: Optional[int] = None) -> List[Row]:
    """The stream's newest rows below ``before`` (its end when None), oldest first."""
    with transaction(root, write=False) as conn:
        records = conn.execute(f"{_SELECT} WHERE stream=? AND seq<? ORDER BY seq DESC LIMIT ?",
                               (stream, _bound(before), limit)).fetchall()
    return [_row(record) for record in reversed(records)]


def _bound(before: Optional[int]) -> int:
    return (1 << 62) if before is None else int(before)


def probe_rows(root: Any, stream: str, keys: Iterable[Tuple[str, Any]], *, before: Optional[int] = None,
               lower: int = 0, limit: int = 200, newest_first: bool = True) -> List[Row]:
    """The union of the ``(column, value)`` keys' rows in ``[lower, before)``, newest first, at
    most ``limit`` rows: the candidate rows of a room or a task tree. One statement per column reads
    at most ``limit`` coordinates through its index; only the union's winning rows fetch bodies in
    that snapshot; JSON decoding follows release. ``("chat_id", None)`` probes NULL chats."""
    with transaction(root, write=False) as conn:
        records = _probe(conn, stream, keys, before=before, lower=lower, limit=limit, newest_first=newest_first)
    return [_row(record) for record in records]


def _probe(conn: sqlite3.Connection, stream: str, keys: Iterable[Tuple[str, Any]], *, before: Optional[int],
           lower: int, limit: int, newest_first: bool) -> List[sqlite3.Row]:
    """Capture the winning stored records; the transaction owner decodes after release.

    One statement per column selects, through that column's index, the first ``limit`` coordinates
    matching any of its values (one JSON list), so Python receives at most columns x ``limit``
    integers however many keys a room has: a row among the union's first ``limit`` is among its own
    column's first ``limit``, so the first ``limit`` of these is the union's."""
    values: Dict[str, list] = {}
    for column, value in dict.fromkeys(tuple(keys)):
        if column not in _PROBE_INDEX:
            raise ValueError(f"not a dialogue probe column: {column!r}")
        if value is not None or column == "chat_id":  # the partial indexes hold only rows that carry the fact
            values.setdefault(column, []).append(value)
    found = set()
    order, bounds = ("DESC" if newest_first else "ASC"), (_bound(before), lower, limit)
    for column, listed in values.items() if limit > 0 else ():
        select = f"SELECT seq FROM dialogue INDEXED BY {_PROBE_INDEX[column]} WHERE stream=? AND {column} "
        tail = f" AND seq<? AND seq>=? ORDER BY seq {order} LIMIT ?"
        if None in listed:  # the chat-less rows, through the full room index
            found.update(record["seq"] for record in conn.execute(select + "IS NULL" + tail, (stream, *bounds)))
        if any(value is not None for value in listed):
            found.update(record["seq"] for record in conn.execute(
                select + "IN (SELECT +value FROM json_each(?))" + tail,  # unary + keeps the column's affinity
                (stream, json.dumps([value for value in listed if value is not None]), *bounds)))
    chosen = heapq.nlargest(limit, found) if newest_first else heapq.nsmallest(limit, found)
    if not chosen:
        return []
    # Python 3.10 has no getlimit; 999 is the default variable limit at our SQLite 3.24 floor.
    variables = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) if hasattr(conn, "getlimit") else 999
    records = {}
    for start in range(0, len(chosen), variables - 1):  # one binding is the stream, including for 10,000-row readers
        batch = chosen[start:start + variables - 1]
        marks = ",".join("?" for _ in batch)
        for record in conn.execute(f"{_SELECT} WHERE stream=? AND seq IN ({marks})", (stream, *batch)):
            records[record["seq"]] = record
    return [records[seq] for seq in chosen]


def room_candidates(root: Any, stream: str, chat_id: int, *, tasks: Iterable[str] = (), origins: Iterable[str] = (),
                    before: Optional[int] = None, lower: int = 0, limit: int = 200) -> List[Row]:
    """A room's possible rows, newest first (``project_dialogue.room_membership`` decides each):
    the rows written to its chat, every row of a task the room rule can place in it (by task,
    parent or root id) and the owner rows its bindings name (``origin`` = ``<chat>:<client id>``)."""
    keys: List[Tuple[str, Any]] = [("chat_id", int(chat_id))]
    for task in tasks:
        keys.extend((("task_id", task), ("parent_task_id", task), ("root_task_id", task)))
    keys.extend(("origin", origin) for origin in origins)
    return probe_rows(root, stream, keys, before=before, lower=lower, limit=limit)


def main_candidates(root: Any, stream: str, *, exclude_chats: Iterable[int], pinned_kinds: Iterable[str] = (),
                    before: Optional[int] = None, lower: int = 0, limit: int = 200) -> List[Row]:
    """Main's possible rows, newest first: the rows of every chat no Project owns (the rooms the
    store knows minus ``exclude_chats``, plus rows without a chat id) and the Project rows of the
    kinds Main mirrors (``pinned_kinds``: its questions)."""
    excluded = {int(chat) for chat in exclude_chats}
    with transaction(root, write=False) as conn:
        chats = [record["chat_id"] for record in conn.execute("SELECT chat_id FROM rooms ORDER BY chat_id")
                 if record["chat_id"] not in excluded]
        keys: List[Tuple[str, Any]] = [("chat_id", chat) for chat in chats] + [("chat_id", None)]
        keys.extend(("kind", kind) for kind in pinned_kinds)
        records = _probe(conn, stream, keys, before=before, lower=lower, limit=limit, newest_first=True)
    return [_row(record) for record in records]


def task_rows(root: Any, task_ids: Iterable[str], *, stream: str, after: int = -1,
              before: Optional[int] = None, limit: int = 1000) -> List[Row]:
    """A task tree's rows in one stream, oldest first: every row whose task, parent, root or
    ``subagent_task_id`` (a child's rows written under its parent's task) is one of ``task_ids``
    and lies after ``after`` and before ``before`` (a captured boundary, or the live end)."""
    keys: List[Tuple[str, Any]] = [(column, task) for task in task_ids for column in _TASK_KEYS]
    return probe_rows(root, stream, keys, lower=after + 1, before=before, limit=limit, newest_first=False)


def accepted_row(root: Any, chat_id: Optional[int], client_message_id: str) -> Optional[Row]:
    """The inbound row ``(chat_id, client_message_id)`` names (the newest), as the full chain scan
    answered. While the chat archive is still importing, a miss is checked against the segments
    not yet imported (``dialogue_import.scan_pending``): unknown is never answered "absent"."""
    if not client_message_id:
        return None
    _plan_first(pathlib.Path(root))
    with transaction(root, write=False) as conn:
        record = _existing_named(conn, chat_id, client_message_id)
        importing = record is None and conn.execute(
            "SELECT 1 FROM import_ledger WHERE stream='chat' AND state='pending' LIMIT 1").fetchone() is not None
    if record is not None:
        return _row(record)
    if importing:
        from ouroboros.dialogue_import import scan_pending

        return scan_pending(root, chat_id, client_message_id)
    return None


def rows_by_client_id(root: Any, client_message_id: str, *, stream: str = "chat", chat_id: Any = -1,
                      limit: int = 50) -> List[Row]:
    """The rows carrying a client message id (any direction), newest first; ``chat_id`` narrows
    when given (``None`` for rows without one)."""
    if not client_message_id:
        return []
    where, params = "", ()
    if chat_id != -1:
        where, params = " AND chat_id IS ?", (chat_id,)
    with transaction(root, write=False) as conn:
        records = conn.execute(f"{_SELECT} INDEXED BY dialogue_named WHERE stream=? AND client_message_id=?{where} "
                               "ORDER BY seq DESC LIMIT ?", (stream, client_message_id, *params, limit)).fetchall()
    return [_row(record) for record in records]


def rows_by_identity(root: Any, chat_id: Optional[int], ts: str, *, stream: str = "chat") -> List[Row]:
    """The rows of one chat stamped ``ts`` (a ``chat_chain`` row address's chat and time), in order."""
    with transaction(root, write=False) as conn:
        records = conn.execute(f"{_SELECT} INDEXED BY dialogue_room_ts WHERE stream=? AND chat_id IS ? AND ts=? ORDER BY seq",
                               (stream, chat_id, ts)).fetchall()
    return [_row(record) for record in records]


def legacy_candidates(root: Any, gen: str, line: int) -> List[Row]:
    """Every chat row a legacy ``(gen, line)`` address may name, in chain order: ``gen`` the
    generation's first-line sha256 (``utils.jsonl_generation_signature``), ``line`` its physical
    line number. Two generations that start with the same line share a signature; the old
    reader checked each against the address's chat, time and hash -- the caller still does."""
    if not gen or not line:
        return []
    with transaction(root, write=False) as conn:
        records = conn.execute(f"{_SELECT} WHERE stream='chat' AND seq IN (SELECT seq FROM legacy_addresses WHERE gen=? AND line=?) "
                               "ORDER BY seq", (gen, int(line))).fetchall()
    return [_row(record) for record in records]


def resolve_legacy(root: Any, gen: str, line: int) -> Optional[Row]:
    """The first chat row (chain order) a legacy ``(gen, line)`` address names; ``legacy_candidates``
    lists every alias when the generation signature repeats."""
    candidates = legacy_candidates(root, gen, line)
    return candidates[0] if candidates else None


def legacy_address(root: Any, seq: int) -> Optional[Tuple[str, int]]:
    """The ``(gen, line)`` an imported chat row had (``None`` for a row appended here); one indexed
    probe (``legacy_seq``), whatever the other rooms imported."""
    with transaction(root, write=False) as conn:
        record = conn.execute("SELECT gen, line FROM legacy_addresses INDEXED BY legacy_seq WHERE seq=? LIMIT 1", (seq,)).fetchone()
    return None if record is None else (record["gen"], record["line"])


def legacy_generation(root: Any, gen: str) -> Optional[Dict[str, Any]]:
    """The imported chat segment whose first-line signature is ``gen`` (the first in chain order
    when the signature repeats, as the chain reader chose): its ledger row -- ``id`` (the unit),
    ``name``, ``base`` (the coordinate of its first byte; a legacy ``(gen, byte offset)`` cursor
    is the coordinate ``base + offset``), ``size``, ``rows`` (its stored objects), ``stream_rows``
    (its stream rows outside A2A), ``ordinal_base``/``pos_base`` (the absolute counts before its
    first row, None until the archive is complete). ``None`` while the segment is not imported
    yet (its signature is read at import) or never existed: a caller then reports the range as
    still importing when ``dialogue_import.coverage`` says so, never as gone."""
    if not gen:
        return None
    with transaction(root, write=False) as conn:
        record = conn.execute("SELECT * FROM import_ledger WHERE stream='chat' AND gen=? AND state='done' ORDER BY base LIMIT 1",
                              (gen,)).fetchone()
    return None if record is None else dict(record)


def _gaps(conn: sqlite3.Connection, stream: str, lower: int, upper: int) -> int:
    return int(conn.execute("SELECT count(*) AS n FROM skipped WHERE stream=? AND seq>=? AND seq<?",
                            (stream, lower, upper)).fetchone()["n"])



def stream_end(root: Any, stream: str) -> int:
    """The stream's next coordinate: a page's frozen upper bound, and the one change counter a
    reader compares instead of a file's stat."""
    with transaction(root, write=False) as conn:
        record = conn.execute("SELECT next_seq FROM streams WHERE stream=?", (stream,)).fetchone()
    return 0 if record is None else int(record["next_seq"])


def room(root: Any, chat_id: int) -> Optional[Dict[str, Any]]:
    with transaction(root, write=False) as conn:
        record = conn.execute("SELECT * FROM rooms WHERE chat_id=?", (int(chat_id),)).fetchone()
    return None if record is None else dict(record)


def rooms(root: Any) -> List[Dict[str, Any]]:
    """Every room the store knows (its revision, last chat row and activity), by chat id."""
    with transaction(root, write=False) as conn:
        return [dict(record) for record in conn.execute("SELECT * FROM rooms ORDER BY chat_id")]


def mark_room_read(root: Any, chat_id: int, revision: int) -> Dict[str, Any]:
    """The owner read the room through ``revision`` (never backwards); returns the room."""
    with transaction(root, write=True) as conn:
        conn.execute("INSERT INTO rooms (chat_id, read_revision) VALUES (?, ?) ON CONFLICT(chat_id) DO UPDATE SET "
                     "read_revision = max(rooms.read_revision, excluded.read_revision)", (int(chat_id), int(revision)))
        return dict(conn.execute("SELECT * FROM rooms WHERE chat_id=?", (int(chat_id),)).fetchone())


def counts(root: Any) -> Dict[str, int]:
    """Rows per stream (tests and the import report; a full count, not a hot-path read)."""
    with transaction(root, write=False) as conn:
        return {record["stream"]: record["n"] for record in conn.execute("SELECT stream, count(*) AS n FROM dialogue GROUP BY stream")}


def iter_probes(root: Any, stream: str, keys: Iterable[Tuple[str, Any]], *, lower: int = 0,
                upper: Optional[int] = None, batch: int = 200, newest_first: bool = False) -> Iterator[Row]:
    """Added by Q1b: iterate an addressed union in a frozen range, in short transactions.

    Each page reads only the indexed keys, deduplicates overlapping probes and advances
    by coordinate. A caller may stop early without holding a connection or a read lock.
    """
    if batch < 1:
        raise ValueError("a dialogue probe batch must be positive")
    keys = tuple(keys)
    if not keys:
        return
    upper = stream_end(root, stream) if upper is None else int(upper)
    while lower < upper:
        rows = probe_rows(root, stream, keys, lower=lower, before=upper,
                          limit=batch, newest_first=newest_first)
        yield from rows
        if len(rows) < batch:
            return
        if newest_first:
            upper = rows[-1].seq
        else:
            lower = rows[-1].seq + 1


def range_coverage(root: Any, stream: str, *, lower: int = 0,
                   upper: Optional[int] = None) -> Dict[str, Any]:
    """Added by Q1b: import coverage of exactly ``[lower, upper)``.

    Same answer as ``dialogue_import.coverage``, restricted to intersecting segments:
    an older pending archive does not make a fully imported recent page incomplete.
    Reading starts the root's normal cutover if needed, never the archive import job.
    ``gaps`` (Q1c) counts the skipped lines starting in the range: imported, but holes a
    reader names (``skipped_between``), never rows.
    """
    if stream not in STREAMS:
        raise ValueError(f"not a dialogue stream: {stream!r}")
    with transaction(root, write=False) as conn:
        counts = {record["state"]: record["n"] for record in conn.execute(
            "SELECT state, count(*) AS n FROM import_ledger INDEXED BY ledger_state "
            "WHERE stream=? AND state IN ('pending', 'mismatch') AND base<? AND base+size>? "
            "GROUP BY state", (stream, _bound(upper), lower))} if lower < _bound(upper) else {}
        gaps = _gaps(conn, stream, lower, _bound(upper))
        planned = _planned(conn, root)
    pending, mismatch = int(counts.get("pending", 0)), int(counts.get("mismatch", 0))
    if not planned:  # a cutover that failed after ``create``: the chain files are still to plan and import
        pending = max(pending, 1)
    return {"imported": not pending and not mismatch, "pending": pending, "mismatch": mismatch, "gaps": gaps,
            **({} if planned else {"unplanned": True})}


def require_imported(root: Any, stream: str, *, lower: int = 0, upper: Optional[int] = None) -> None:
    """Added by Q1b: an absence/complete-fold consumer cannot turn pending import into truth.

    Readers with a coverage field use ``range_coverage`` directly; readers whose public
    answer cannot express an unknown interval propagate the store's typed failure.
    """
    if stream not in STREAMS:
        raise ValueError(f"not a dialogue stream: {stream!r}")
    bound = _bound(upper)
    if lower >= bound:
        return
    with transaction(root, write=False) as conn:  # existence, not counts: one indexed probe per state
        def holds(state: str) -> bool:
            return conn.execute(
                "SELECT 1 FROM import_ledger INDEXED BY ledger_state WHERE stream=? AND state=? AND base<? "
                "AND base+size>? LIMIT 1", (stream, state, bound, lower)).fetchone() is not None

        mismatch, pending = holds("mismatch"), holds("pending") or not _planned(conn, root)
    if mismatch:
        raise StoreError("Dialogue history: a segment could not be imported")
    if pending:
        raise StoreMigrating("Dialogue history is still importing")


def generation_metadata(root: Any, stream: str) -> List[Dict[str, Any]]:
    """Appended read function (Q1b2 M): the imported generations and live counting unit.

    Only whole-stream recall and legacy cursor translation need this catalogue;
    addressed room/row readers never enumerate it.
    """
    with transaction(root, write=False) as conn:
        return [dict(record) for record in conn.execute(
            "SELECT * FROM import_ledger WHERE stream=? ORDER BY base, id", (stream,))]


def last_legacy_generation(root: Any, *, gen: str = "", name: str = "") -> Optional[Dict[str, Any]]:
    """Appended read function (Q1b2 M): the last matching legacy segment for a Presence byte cursor.

    Presence selected the last repeated first-line signature, unlike the memory
    ordinal cursor's first match. A named empty-live boundary uses its archive name.
    """
    column, value = ("name", name) if name else ("gen", gen)
    with transaction(root, write=False) as conn:
        record = conn.execute(f"SELECT * FROM import_ledger WHERE stream='chat' AND {column}=? "
                              "AND state='done' ORDER BY base DESC LIMIT 1", (value,)).fetchone()
    return None if record is None else dict(record)


