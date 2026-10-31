"""Append-only derived memory (the chronicle) with a disposable indexed projection.

``memory/chronicle/records.jsonl`` is the sole authority. One line is one
transaction ``{"kind": "transaction", "schema": 1, "records": [...],
"scan_state": {...}}``, appended and fsynced under the short
``.publication.lock``. ``index.sqlite3`` is only a projection: deleting it
replays the journal into the same rows, sequences, page coverage and folds; a
torn or invalid transaction becomes an addressed ``gap`` record and its
neighbours stay intact. A reader takes the lock only to bring a stale index up
to date. The index also keeps the facts of the chat rows after the legacy
frontier (``tail_rows``), a projection of ``state/dialog.sqlite`` that
``memory_inventory`` fills and extends. ``scan_state`` merges by top-level key,
so one writer's frontier never erases another's.

A record is never rewritten. A ``page`` seals a SET of row refs of its room
(``page_rows``: chat ``row_sha256`` values and ``note:<id>``), once per level:
a page whose set intersects an already sealed set is refused
``already_sealed``; ``covers.stream_span[0]`` only orders pages. A ``part``
folds adjacent effective records of one lower level of its room (pages, legacy
sections or parts), each at most once (``folded``). A helper's page or part (the
Light writer's, or a delegated child's or nanny's signed with its focus) is
a draft that acts at once; the mind's ``decision`` accepts or rejects it, and a
rejected draft stops acting, so its rows are open and its members unfolded
again; a draft already folded into a part is not rejected while that part acts
(``already_folded``). Only the mind corrects, and a correction stands under the
original wherever the record is shown, signed and dated (``_interpret``); it never
replaces the record's words.

An ``account`` is the mind's own text connecting experience of several rooms. It names
the exact versions of the records it is based on (``sources``: id, the revision the mind
read, the body's sha256, and a draft's status then); that relation is provenance, never
ownership: it seals no row, folds nothing, moves no room head, and the same source may
inform several accounts and still be folded locally. Whether an account stands in the
common view, and which named records it tells in their place, is a separate ``selection``
by the mind (the latest per account acts; ``shown`` false withdraws it). Both are
corrected like any record; neither is rewritten. What changed in a source since the
account was written is read at interpretation time (``_source_state``), never written
back into the account.

Every precondition is checked inside the same publication lock, and a refusal
is a typed ``PublishResult`` (not an exception) carrying the current revision
or room head and the conflicting ids, never their text. The room head is the
newest sequence of the records that change what a room's story says; marks and
the activation receipt do not move it. The store reads no chat rows: quotes
are verified by an injected resolver before the lock is taken.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sqlite3
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

from ouroboros import store, store_timing
from ouroboros.chronicle_quotes import QuoteResolver, quote_refusal
from ouroboros.platform_layer import (acquire_exclusive_file_lock, kernel_file_locks_enforced,
                                      release_exclusive_file_lock)
from ouroboros.utils import append_jsonl, assert_test_data_path, utc_now_iso

SCHEMA_VERSION = 3
KINDS = frozenset({"page", "part", "note", "correction", "decision", "mark", "mark_view", "mark_release",
                   "legacy", "gap", "activation", "account", "selection"})
AUTHOR_KINDS = frozenset({"mind", "helper", "host", "legacy_helper"})
STORY_KINDS = ("page", "part", "note", "legacy", "gap", "account")
# An account or its selection changes the common view, not what one room's story says: neither moves a room head.
HEAD_KINDS = ("page", "part", "note", "correction", "decision", "legacy", "gap")
_CORRECTABLE = frozenset({"page", "part", "note", "legacy", "gap", "account"})
_FOLDABLE = frozenset({"page", "part", "legacy"})
_TEXT_KINDS = frozenset({"page", "part", "note", "correction", "mark", "account"})
_SOURCE_FIELDS = ("id", "kind", "room_id", "revision", "sha256")  # what an account freezes of each source
_NO_BLOCK = float("inf")
_CHUNK = 500

# A schema change bumps SCHEMA_VERSION: a reader on the lock-free path (``_current``) checks only the version.
_SCHEMA = """
    CREATE TABLE IF NOT EXISTS records (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        id TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
        room TEXT NOT NULL, target TEXT NOT NULL, body TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS room_records ON records(room, kind, sequence);
    CREATE INDEX IF NOT EXISTS target_records ON records(target, sequence);
    CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS active_marks (
        id TEXT PRIMARY KEY, room TEXT NOT NULL, scope TEXT NOT NULL,
        sequence INTEGER NOT NULL, body TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS room_marks ON active_marks(room, sequence);
    CREATE INDEX IF NOT EXISTS global_marks ON active_marks(scope, sequence);
    CREATE TABLE IF NOT EXISTS page_rows (
        room TEXT NOT NULL, page_id TEXT NOT NULL, row_ref TEXT NOT NULL, UNIQUE(room, row_ref));
    CREATE INDEX IF NOT EXISTS page_rows_page ON page_rows(page_id);
    CREATE TABLE IF NOT EXISTS folded (member_id TEXT PRIMARY KEY, part_id TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS folded_part ON folded(part_id);
    CREATE TABLE IF NOT EXISTS tail_rows (pos INTEGER PRIMARY KEY, seq INTEGER NOT NULL, row_sha256 TEXT NOT NULL,
                                          legacy TEXT, meta TEXT NOT NULL, terminal TEXT);
    CREATE TABLE IF NOT EXISTS told (record TEXT NOT NULL, account TEXT NOT NULL, sequence INTEGER NOT NULL,
                                     PRIMARY KEY (record, account));
    CREATE INDEX IF NOT EXISTS told_account ON told(account);
"""
_RESET = ("DELETE FROM records; DELETE FROM state; DELETE FROM active_marks; DELETE FROM page_rows; "
          "DELETE FROM folded; DELETE FROM tail_rows; DELETE FROM told; DELETE FROM sqlite_sequence WHERE name='records'; "
          f"PRAGMA user_version={SCHEMA_VERSION};")


class PublishResult(NamedTuple):
    """Outcome of one publication; ``reason`` is ``saved``, ``unchanged`` or a typed refusal.

    Refusals: ``revision_required``, ``revision_conflict``, ``already_sealed``,
    ``already_folded``, ``quote_mismatch``, ``target_missing``, ``not_integrator``, ``invalid``. A
    refusal names ids (``conflict_ids``), the current revision or room head, and
    a short ``detail``; it never carries record text.
    """
    ok: bool
    reason: str
    record: Optional[Dict[str, Any]] = None
    current_revision: Optional[str] = None
    current_head: Optional[int] = None
    conflict_ids: Tuple[str, ...] = ()
    detail: str = ""


def _refuse(reason: str, detail: str = "", **fields: Any) -> PublishResult:
    return PublishResult(False, reason, detail=detail, **fields)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def body_sha256(record: Mapping[str, Any]) -> str:
    """The sha256 of a record's canonical body (``_json`` of the journal record, without its sequence)."""
    body = {key: value for key, value in record.items() if key != "sequence"}
    return hashlib.sha256(_json(body).encode("utf-8")).hexdigest()


def source_time_span(timestamps: Iterable[Any], *, incomplete: bool = False) -> Dict[str, Any]:
    """Known source bounds, never publication time or proof of continuous coverage."""
    from ouroboros.deadline_utils import parse_deadline_ts

    parsed = [parse_deadline_ts(value) for value in timestamps]
    known = [value for value in parsed if value is not None]
    return {"start": min(known).isoformat() if known else None,
            "end": max(known).isoformat() if known else None,
            "incomplete": bool(incomplete or not known or len(known) != len(parsed))}


# A host-signed mark that keeps late reviewer feedback in view in its room (``late_review_marks``). The
# feedback waits for the integrating mind: a delegated child or nanny is a helper, so only another
# focus releases it, with a reason (``_rule_mark_release``); a first show, a page seal or a read never does.
LATE_REVIEW_OPERATION = "late_review"
DELEGATED_FOCUS_ROLES = frozenset({"child", "nanny"})


def is_late_review_mark(mark: Any) -> bool:
    author = mark.get("author") if isinstance(mark, Mapping) else None
    return isinstance(author, Mapping) and author.get("kind") == "host" and author.get("operation") == LATE_REVIEW_OPERATION


def _author_problem(author: Any) -> str:
    if not isinstance(author, dict) or author.get("kind") not in AUTHOR_KINDS:
        return "author.kind is one of " + ", ".join(sorted(AUTHOR_KINDS))
    if author["kind"] == "mind":
        focus = author.get("focus")
        if not isinstance(focus, dict) or not _nonblank(focus.get("role")) or not _nonblank(author.get("task_id")):
            return "a mind author carries focus.role and task_id"
    return ""


# The one sentence a delegated child's or nanny's role line adds about the chronicle: a helper is not
# the integrating mind, so tools/chronicle.py signs its page or part as a draft and the mind decides.
CHILD_DRAFT_RIGHT = ("You may also publish chronicle pages and parts as drafts in your own name; the integrating "
                     "mind accepts or rejects them.")


def draft_signer(author: Any) -> str:
    """Who signed a helper's draft page or part, as the view and ``memory_read`` name it.

    A delegated child or nanny drafts under the focus the host signed (``child, task
    <id>``); a helper author without a focus is the Light writer (``Light``).
    """
    author = author if isinstance(author, dict) else {}
    focus = author.get("focus") if isinstance(author.get("focus"), dict) else {}
    role = str(focus.get("role") or "").strip()
    if not role:
        return "Light"
    task = str(author.get("task_id") or focus.get("task_id") or "").strip()
    return f"{role}, task {task}" if task else role


def correction_line(fix: Mapping[str, Any]) -> str:
    """The signature a correction stands under, beside its original: ``[correction <id> by mind (<focus>), <date>]``.

    Only the mind corrects, so the focus (``draft_signer``'s words) and the publication date
    say who and when; the same line wherever the record is shown.
    """
    return f"[correction {fix.get('id')} by mind ({draft_signer(fix.get('author'))}), {str(fix.get('ts') or '')[:10] or 'date not recorded'}]"


def verify_quotes(quotes: Any, resolver: Optional[QuoteResolver]) -> Optional[PublishResult]:
    """Each quote is the words of the row at its address, spoken by the named class (``chronicle_quotes``): a
    typed refusal, or None. ``resolver`` reads chat rows, so it runs before the publication lock is taken."""
    refusal = quote_refusal(quotes, resolver)
    return None if refusal is None else _refuse(*refusal)


class _IndexLookups:
    """Every read the readers and the interpretation make, answered by the index itself (a
    publisher's open transaction included): one statement per answer, the SQL in one place."""
    __slots__ = ("db",)

    def __init__(self, db):
        self.db = db

    def get(self, record_id):
        row = self.db.execute("SELECT sequence,body FROM records WHERE id=?", (str(record_id),)).fetchone()
        return {**json.loads(row[1]), "sequence": row[0]} if row else None

    def targeting(self, record_id, kinds, after=0, limit=-1):
        """``(sequence, body)`` of the records of ``kinds`` aimed at a record after ``after`` (at most ``limit``), in order."""
        return [(seq, json.loads(body)) for seq, body in self.db.execute(
            "SELECT sequence,body FROM records WHERE target=? AND kind IN (%s) AND sequence>? ORDER BY sequence LIMIT ?"
            % ",".join("?" * len(kinds)), (str(record_id), *kinds, int(after), int(limit)))]

    def part_of(self, record_id):
        row = self.db.execute("SELECT part_id FROM folded WHERE member_id=?", (str(record_id),)).fetchone()
        return row[0] if row else None

    def rows(self, room=None, kinds=(), after=0, limit=-1, author_kind=None, unfolded=False):
        """``(sequence, stored body)`` of at most ``limit`` records after ``after`` (one room's, of ``kinds``, of one
        author kind, none a part folds), in order."""
        where, args = ["sequence > ?"], [int(after)]
        for clause, value in (("room=?", room), ("json_extract(body,'$.author.kind')=?", author_kind)):
            if value is not None:
                where.append(clause)
                args.append(str(value))
        if kinds:
            where.append("kind IN (%s)" % ",".join("?" * len(kinds)))
            args.extend(kinds)
        if unfolded:
            where.append("id NOT IN (SELECT member_id FROM folded)")
        return self.db.execute("SELECT sequence,body FROM records WHERE " + " AND ".join(where)
                               + " ORDER BY sequence LIMIT ?", [*args, int(limit)]).fetchall()

    def sealed(self, room):
        return {ref for (ref,) in self.db.execute("SELECT row_ref FROM page_rows WHERE room=?", (str(room),))}

    def marks(self, room=None, include_global=True):
        """Stored bodies of the acting marks (one room's and, with ``include_global``, the global ones), in order."""
        if room is None:
            return [body for (body,) in self.db.execute("SELECT body FROM active_marks ORDER BY sequence")]
        query = "SELECT body FROM active_marks WHERE room=?" + (" OR scope='global'" if include_global else "")
        return [body for (body,) in self.db.execute(query + " ORDER BY sequence", (str(room),))]

    def state(self, key, default=None):
        return ChronicleStore._state(self.db, key, default)

    def head(self, room):
        return self.db.execute("SELECT COALESCE(MAX(sequence),0) FROM records WHERE room=? AND kind IN (%s)"
                               % ",".join("?" * len(HEAD_KINDS)), (str(room), *HEAD_KINDS)).fetchone()[0]

    def tail(self, terminal_only=False):
        sql = _TAIL_SQL + (_TERMINAL if terminal_only else "")
        return sorted(json.loads(self.db.execute(sql).fetchone()[0] or "[]"))

    def acting_pages(self, room=None):
        """``[sequence, room, id, author kind, ts, covered end]`` of the acting pages (one room's), in order."""
        sql, args = (_ACTING_PAGES, ()) if room is None else (_ACTING_PAGES + " AND room=?", (str(room),))
        return sorted(json.loads(self.db.execute(sql, args).fetchone()[0] or "[]"))

    def acting_parts(self):
        """``[id, author kind, metadata.candidate]`` of the parts that fold their members."""
        return json.loads(self.db.execute(_ACTING_PARTS).fetchone()[0] or "[]")

    def members_of(self, record_id):
        """A part's member ids (its ``covers.member_ids``), read without its body; none for any other record."""
        row = self.db.execute("SELECT json_extract(body,'$.covers.member_ids') FROM records WHERE id=? AND kind='part'",
                              (str(record_id),)).fetchone()
        return [str(member) for member in json.loads(row[0] or "[]")] if row else []


_TAIL_SQL = "SELECT json_group_array(json_array(pos,seq,row_sha256,json(legacy),json(meta),terminal)) FROM tail_rows"
_TERMINAL = " WHERE terminal IS NOT NULL"
# The acting pages are those whose rows a seal holds (``page_rows``: a final page, a draft not rejected, an adopted
# candidate); the parts that fold (``folded``) act unless ``_status`` names them a candidate. No body is read whole.
_ACTING_PAGES = ("SELECT json_group_array(json_array(sequence,room,id,json_extract(body,'$.author.kind'),"
                 "json_extract(body,'$.ts'),json_extract(body,'$.covers.ts_span.end'))) FROM records "
                 "WHERE kind='page' AND EXISTS (SELECT 1 FROM page_rows WHERE page_id=records.id)")
_ACTING_PARTS = ("SELECT json_group_array(json_array(id,json_extract(body,'$.author.kind'),"
                 "json_extract(body,'$.metadata.candidate'))) FROM records WHERE id IN (SELECT part_id FROM folded)")
# A chunk of tail rows in ONE statement (one step), as a page is read: [pos, seq, sha, legacy text, meta text, terminal].
_TAIL_INSERT = ("INSERT OR IGNORE INTO tail_rows SELECT json_extract(value,'$[0]'), json_extract(value,'$[1]'), "
                "json_extract(value,'$[2]'), json_extract(value,'$[3]'), json_extract(value,'$[4]'), "
                "json_extract(value,'$[5]') FROM json_each(?)")
# One state of the index: each table in ONE statement, one ``sqlite3_step`` (one GIL release) per table, not per row.
_SNAPSHOT_SQL = ("SELECT json_group_array(json_array(sequence,id,kind,room,target,body)) FROM records",
                 "SELECT json_group_array(json_array(member_id,part_id)) FROM folded",
                 "SELECT json_group_array(json_array(room,row_ref)) FROM page_rows",
                 "SELECT json_group_array(json_array(sequence,room,scope,body)) FROM active_marks",
                 "SELECT json_group_array(json_array(key,value)) FROM state", _ACTING_PAGES, _ACTING_PARTS)
_TAIL_SCOPES = {"all": (_TAIL_SQL,), "terminal": (_TAIL_SQL + _TERMINAL,), "none": ()}


class _Snapshot:
    """``_IndexLookups`` answered from one state of the index (``_SNAPSHOT_SQL``, parsed after its read
    transaction closed): the same filters and order and no statement. Each record handed out is parsed
    anew from its stored text, as the index returns it; ``memo`` holds facts one capture derives once.
    Of the stored tail facts it holds what its view reads (``tail_scope``)."""

    def __init__(self, texts, tail_scope):
        records, folded, sealed, marks, state, acting, self.parts, *tail = (json.loads(text or "[]") for text in texts)
        self.acting, self.tail_scope, self.acting_by_room = sorted(acting), tail_scope, {}
        for row in self.acting:
            self.acting_by_room.setdefault(row[1], []).append(row)
        self.records = sorted(records)  # [sequence, id, kind, room, target, body] by sequence
        self.by_id = {row[1]: row for row in self.records}
        self.by_room: Dict[str, List[Any]] = {}
        self.by_target: Dict[str, List[Any]] = {}
        for row in self.records:
            self.by_room.setdefault(row[3], []).append(row)
            self.by_target.setdefault(row[4], []).append(row)
        self.folded = dict(folded)
        self.page_rows: Dict[str, set] = {}
        for room, ref in sealed:
            self.page_rows.setdefault(room, set()).add(ref)
        self.active = sorted(marks)
        self.values = dict(state)
        self.tail_rows = sorted(tail[0]) if tail else []  # [pos, seq, row_sha256, legacy, meta, terminal] by position
        self.memo: Dict[str, Any] = {"tail_scope": tail_scope}

    def get(self, record_id):
        row = self.by_id.get(str(record_id))
        return {**json.loads(row[5]), "sequence": row[0]} if row else None

    def targeting(self, record_id, kinds):
        return [(row[0], json.loads(row[5])) for row in self.by_target.get(str(record_id), ()) if row[2] in kinds]

    def part_of(self, record_id):
        return self.folded.get(str(record_id))

    def rows(self, room=None, kinds=(), after=0, limit=-1, author_kind=None, unfolded=False):
        rows = self.records if room is None else self.by_room.get(str(room), ())
        return [(row[0], row[5]) for row in rows if row[0] > int(after) and (not kinds or row[2] in kinds) and not (
            unfolded and row[1] in self.folded) and (author_kind is None or _mapping(json.loads(row[5]).get("author")).get(
                "kind") == author_kind)][:None if limit < 0 else limit]

    def sealed(self, room):
        return set(self.page_rows.get(str(room), ()))

    def marks(self, room=None, include_global=True):
        return [body for _seq, held, scope, body in self.active
                if room is None or held == str(room) or include_global and scope == "global"]

    def state(self, key, default=None):
        return json.loads(self.values[key]) if key in self.values else default

    def head(self, room):
        return max((row[0] for row in self.by_room.get(str(room), ()) if row[2] in HEAD_KINDS), default=0)

    def holds_tail(self, terminal_only):
        return self.tail_scope == "all" or terminal_only and self.tail_scope == "terminal"

    def tail(self, terminal_only=False):
        return [list(row) for row in self.tail_rows if not terminal_only or row[5] is not None]

    def acting_pages(self, room=None):
        return [list(row) for row in (self.acting if room is None else self.acting_by_room.get(str(room), ()))]

    def acting_parts(self):
        return [list(row) for row in self.parts]

    def members_of(self, record_id):
        row = self.by_id.get(str(record_id))
        covers = json.loads(row[5]).get("covers") if row and row[2] == "part" else None
        return [str(member) for member in covers.get("member_ids") or ()] if isinstance(covers, dict) else []


def _lookups(db):
    return db if isinstance(db, (_IndexLookups, _Snapshot)) else _IndexLookups(db)


_CAPTURE: ContextVar[Optional[Tuple[str, _Snapshot]]] = ContextVar("chronicle_capture", default=None)


@contextmanager
def capture(data_root: Any, *, tail: str = "all"):
    """One state of this installation's chronicle for every reader inside the block (one capture).

    The index is read once (``_SNAPSHOT_SQL`` in one read transaction, closed before any reader runs, so a
    publisher never waits on a capture) and pinned for the block; inside a capture of the same chronicle
    this is that capture. Of the stored tail facts it reads what its view reads (``tail``: every row,
    ``"terminal"`` rows or ``"none"``); a reader asking for more reads the index. It yields the facts its
    readers derive once (``capture_memo``); without a journal nothing is pinned and it yields None. A
    publication inside the block is not seen by its readers.
    """
    store = ChronicleStore(data_root)
    if (pinned := store._pinned()) is not None or not store.log_path.exists():
        yield None if pinned is None else pinned.memo
        return
    with store._index() as db:
        texts = [db.execute(sql).fetchone()[0] for sql in (*_SNAPSHOT_SQL, *_TAIL_SCOPES[tail])]
    snapshot = _Snapshot(texts, tail)
    token = _CAPTURE.set((str(store.index_path), snapshot))
    try:
        yield snapshot.memo
    finally:
        _CAPTURE.reset(token)


def capture_memo(data_root: Any, key: str, compute: Callable[[], Any]) -> Any:
    """``compute()`` once per capture of this chronicle, so its readers share one answer; outside a capture,
    each call computes."""
    snapshot = ChronicleStore(data_root)._pinned()
    if snapshot is None:
        return compute()
    if key not in snapshot.memo:
        snapshot.memo[key] = compute()
    return snapshot.memo[key]


class ChronicleStore:
    """One installation's derived pages, parts, notes, corrections, decisions and marks."""

    def __init__(self, data_root: Any):
        self.data_root = pathlib.Path(data_root)
        self.directory = self.data_root / "memory" / "chronicle"
        self.log_path = self.data_root / "memory" / "chronicle" / "records.jsonl"
        self.index_path = self.data_root / "memory" / "chronicle" / "index.sqlite3"
        self.lock_path = self.data_root / "memory" / "chronicle" / ".publication.lock"

    def _pinned(self) -> Optional[_Snapshot]:
        held = _CAPTURE.get()
        return held[1] if held is not None and held[0] == str(self.index_path) else None

    def journal_size(self) -> int:
        """The size of ``records.jsonl`` the reads reflect: the file's, or the size a pinned capture indexed."""
        snapshot = self._pinned()
        return snapshot.state("offset", 0) if snapshot is not None else self.log_path.stat().st_size

    # --- projection --------------------------------------------------------------------------------

    @contextmanager
    def _index(self):
        """A reader's projection. While the index holds every authority byte (its ``offset`` equals the size
        of ``records.jsonl`` within one read snapshot) it is read on a read-only connection with no lock file,
        no schema script and no catch-up: SQLite's own locks order the read against a publisher's commit.
        Otherwise, and where kernel locks are not enforced (the name tier), the publisher's locked catch-up."""
        db = self._current()
        if db is None:
            with self._publishing() as db:
                yield db
            return
        store_timing.count("chronicle.read")
        try:
            yield db
        finally:
            db.close()

    def _current(self) -> Optional[sqlite3.Connection]:
        # The platform layer keeps each directory's decided lock tier; an undecided probe is asked again.
        if not self.index_path.exists() or not kernel_file_locks_enforced(self.lock_path):
            return None
        try:
            # Counted like every store connection: a raw read of this file never closes a descriptor under the
            # snapshot (POSIX drops this process's locks on any close; ``store._borrow``).
            db = store.counted_connect(f"{self.index_path.absolute().as_uri()}?mode=ro", self.index_path, uri=True)
        except (sqlite3.Error, OSError):  # an index removed meanwhile is the locked path's to rebuild
            return None
        try:
            db.execute("BEGIN")  # the version, the offset and every read after them: one snapshot
            if (db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION and self._state(db, "offset", 0)
                    == (self.log_path.stat().st_size if self.log_path.exists() else 0)):
                return db
        except (sqlite3.Error, OSError, ValueError):
            pass
        db.close()
        return None

    @contextmanager
    def _publishing(self, wait_sec: float = 10):
        assert_test_data_path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        fd = acquire_exclusive_file_lock(self.lock_path, timeout_sec=wait_sec, stale_sec=90, owner_aware_stale=True)
        if fd is None:
            raise TimeoutError("chronicle publication lock unavailable")
        store_timing.count("chronicle.locked")  # every locked open; a lock-free read counts "chronicle.read"
        db = None
        try:
            try:
                db = store.counted_connect(str(self.index_path), self.index_path, timeout=wait_sec)
                db.executescript(_SCHEMA)
            except sqlite3.DatabaseError as exc:
                if db is not None:
                    db.close()
                if store.busy(exc):  # a reader's lock is contention, never a corrupt index to delete
                    raise TimeoutError("chronicle index busy") from exc
                # This is only an index. Preserve the authority and reconstruct it.
                self.index_path.unlink(missing_ok=True)
                db = store.counted_connect(str(self.index_path), self.index_path, timeout=wait_sec)
                db.executescript(_SCHEMA)
            if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                # Only the disposable projection changes schema; replay the authority once.
                db.executescript(_RESET)
            self._catch_up(db)
            yield db
        except sqlite3.OperationalError as exc:  # lock-free readers hold SQLite's read locks: past ``wait_sec``, busy
            if not store.busy(exc):
                raise
            raise TimeoutError("chronicle index busy") from exc
        finally:
            if db is not None:
                db.close()
            release_exclusive_file_lock(self.lock_path, fd)

    @staticmethod
    def _state(db, key, default=None):
        row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    @staticmethod
    def _set_state(db, key, value):
        db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, _json(value)))

    def _catch_up(self, db):
        offset = self._state(db, "offset", 0)
        if not self.log_path.exists():
            if offset:
                raise ValueError("chronicle authority missing; index is not memory")
            return
        if self.log_path.stat().st_size < offset:
            raise ValueError("chronicle authority shortened")
        with self.log_path.open("rb") as stream:
            stream.seek(offset)
            for raw in stream:
                if not raw.strip():
                    offset = stream.tell()
                    self._set_state(db, "offset", offset)
                    continue
                # Completed transactions remain usable after an interrupted append.
                # The bad payload stays in authority, with a stable addressed gap.
                db.execute("SAVEPOINT journal_transaction")
                try:
                    tx = json.loads(raw)
                    if not isinstance(tx, dict) or tx.get("kind") != "transaction" or not isinstance(tx.get("records"), list):
                        raise ValueError("invalid transaction")
                    self._project(db, tx)
                except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                    db.execute("ROLLBACK TO journal_transaction")
                    payload = raw.rstrip(b"\r\n")
                    end = offset + len(payload)
                    digest = hashlib.sha256(payload).hexdigest()
                    gap = {"id": f"journal-gap-{offset}-{digest}", "kind": "gap", "room_id": "legacy",
                           "ts": "", "author": {"kind": "host", "operation": "journal_recovery"},
                           "text": f"Derived-memory transaction unreadable at bytes {offset}:{end}; "
                                   "its records and source frontiers were not published. Original bytes are retained.",
                           "source_refs": [{"kind": "chronicle_journal", "path": "memory/chronicle/records.jsonl",
                                            "start_byte": offset, "end_byte": end, "sha256": digest}],
                           "metadata": {"source_gap": "unreadable_derived_transaction"}}
                    self._project(db, {"records": [gap]})
                finally:
                    db.execute("RELEASE journal_transaction")
                offset = stream.tell()
                self._set_state(db, "offset", offset)
            db.commit()

    def _project(self, db, tx):
        for record in tx["records"]:
            self._apply(db, record)
        scan = tx.get("scan_state")
        if scan is not None:
            if not isinstance(scan, dict):
                raise ValueError("scan_state merges by top-level key")
            self._set_state(db, "scan", {**self._state(db, "scan", {}), **scan})

    def _apply(self, db, record):
        encoded = _json(record)
        old = db.execute("SELECT body FROM records WHERE id=?", (record["id"],)).fetchone()
        if old:
            if old[0] != encoded:
                raise ValueError("chronicle record identity collision")
            return
        kind, room, record_id = record["kind"], str(record.get("room_id", "")), record["id"]
        try:
            seq = db.execute("INSERT INTO records(id,kind,room,target,body) VALUES (?,?,?,?,?)",
                             (record_id, kind, room, record.get("target_id", ""), encoded)).lastrowid
            if kind == "activation":
                self._set_state(db, "activation", record)
            elif kind == "mark":
                db.execute("INSERT INTO active_marks VALUES (?,?,?,?,?)",
                           (record_id, room, record.get("scope", "room"), seq, encoded))
            elif kind == "mark_view":
                active = db.execute("SELECT body FROM active_marks WHERE id=?", (record["target_id"],)).fetchone()
                if active:
                    mark = {**json.loads(active[0]), "visibility": record["visibility"], "view_decision": record}
                    db.execute("UPDATE active_marks SET body=? WHERE id=?", (_json(mark), record["target_id"]))
            elif kind == "mark_release":
                db.execute("DELETE FROM active_marks WHERE id=?", (record["target_id"],))
            elif kind == "page" and not record.get("metadata", {}).get("candidate"):
                db.executemany("INSERT INTO page_rows VALUES (?,?,?)",
                               [(room, record_id, str(ref)) for ref in record["covers"]["rows"]])
            elif kind == "part":
                db.executemany("INSERT INTO folded VALUES (?,?)",
                               [(str(member), record_id) for member in record["covers"]["member_ids"]])
            elif kind == "decision" and not record.get("accepted"):
                # A rejected draft stops acting: its rows are open and its members unfolded again.
                db.execute("DELETE FROM page_rows WHERE page_id=?", (record["target_id"],))
                db.execute("DELETE FROM folded WHERE part_id=?", (record["target_id"],))
            elif kind == "selection":  # an account's latest selection is its only acting claim on what it tells
                db.execute("DELETE FROM told WHERE account=?", (record["target_id"],))
                db.executemany("INSERT OR REPLACE INTO told VALUES (?,?,?)",
                               [(str(ref), record["target_id"], seq) for ref in record["replaces"] if record["shown"]])
            elif kind == "decision" and record.get("accepted"):
                target = self._get(db, record["target_id"])
                if target.get("metadata", {}).get("candidate"):
                    db.executemany("INSERT INTO page_rows VALUES (?,?,?)",
                                   [(target["room_id"], target["id"], ref) for ref in target["covers"]["rows"]])
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"chronicle projection conflict: {exc}") from exc

    # --- publication -------------------------------------------------------------------------------

    def publish(self, records: List[Dict[str, Any]], *, scan_state: Optional[Dict[str, Any]] = None) -> PublishResult:
        """Append records and a scan-state delta in ONE durable transaction.

        An existing id with the same body is ``unchanged`` (its original
        timestamp replays); a different body under that id is ``invalid``. A
        repeated activation returns the existing receipt. ``record`` is the
        batch's last record with its ``sequence``.
        """
        with self._publishing() as db:
            return self._commit(db, records, scan_state)

    def room_head(self, room_id: Any) -> int:
        with self._reading() as look:
            return look.head(room_id)

    def publish_page(self, *, room_id: Any, text: str, covers: Dict[str, Any], author: Dict[str, Any],
                     quotes: Iterable[Dict[str, Any]] = (), host_stamp: Optional[Dict[str, Any]] = None,
                     record_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                     quote_resolver: Optional[QuoteResolver] = None,
                     expected_sequence: Optional[int] = None) -> PublishResult:
        """Seal a host-expanded row set; a helper author makes it a draft that acts at once."""
        room = str(room_id)
        record = {"kind": "page", "room_id": room, "text": text, "author": author, "metadata": metadata or {},
                  "covers": {"room_id": room, **covers} if isinstance(covers, dict) else covers}
        quotes = list(quotes or ())
        if quotes:
            record["quotes"] = quotes
        if host_stamp is not None:
            record["host_stamp"] = host_stamp
        if record_id:
            record["id"] = str(record_id)
        refusal = verify_quotes(quotes, quote_resolver)
        if refusal is not None:
            return refusal
        with self._publishing() as db:
            return self._commit(db, [record], quotes_verified=True,
                                pre=lambda: self._head_refusal(db, room, expected_sequence))

    def publish_part(self, *, room_id: Any, text: str, member_ids: List[str], author: Dict[str, Any],
                     expected_sequence: Optional[int], quotes: Iterable[Dict[str, Any]] = (),
                     metadata: Optional[Dict[str, Any]] = None,
                     quote_resolver: Optional[QuoteResolver] = None,
                     host_stamp: Optional[Dict[str, Any]] = None) -> PublishResult:
        """Fold adjacent effective records of one lower level; the room head it read is required.

        ``host_stamp`` (a helper's part: its members' stamps folded by ``part_stamp``) lies
        where a page keeps its own, so a later fold, the view and ``memory_read`` read it.
        """
        room = str(room_id)
        quotes = list(quotes or ())
        refusal = verify_quotes(quotes, quote_resolver)
        if refusal is not None:
            return refusal
        with self._publishing() as db:
            if expected_sequence is None:
                return _refuse("revision_required", "a part folds revisable records: pass expected_sequence, "
                               "the room head its text is based on", current_head=self._head(db, room))
            record = {"kind": "part", "room_id": room, "text": text, "author": author,
                      "covers": self._part_covers(db, member_ids), "metadata": metadata or {}}
            if quotes:
                record["quotes"] = quotes
            if host_stamp is not None:
                record["host_stamp"] = host_stamp
            return self._commit(db, [record], quotes_verified=True,
                                pre=lambda: self._head_refusal(db, room, expected_sequence))

    def write_note(self, *, room_id: Any, task_id: str, text: str, author: Dict[str, Any]) -> PublishResult:
        """The mind's separate record for its future self; a page later covers it by ``note:<id>``."""
        return self.publish([{"kind": "note", "room_id": str(room_id), "task_id": task_id, "text": text,
                              "author": author}])

    def publish_account(self, *, room_id: Any, text: str, sources: Iterable[Any], author: Dict[str, Any],
                        metadata: Optional[Dict[str, Any]] = None, record_id: Optional[str] = None) -> PublishResult:
        """The mind's account over exact source versions; each source is ``{id, revision?}`` or an id.

        The revision named is the one the mind read (the record itself or one of its corrections);
        unnamed, the current one is frozen. A source that is not in this chronicle is
        ``target_missing``; a revision that is not that record's is ``revision_conflict``.
        """
        with self._publishing() as db:
            refs = []
            for given in sources if isinstance(sources, (list, tuple)) else ():
                frozen, refusal = self._source_version(db, given)
                if refusal is not None:
                    return refusal
                refs.append(frozen)
            record = {"kind": "account", "room_id": str(room_id), "text": text, "author": author, "sources": refs,
                      "metadata": metadata or {}}
            if record_id:
                record["id"] = str(record_id)
            return self._commit(db, [record])

    def select_account(self, account_id: str, *, replaces: Iterable[Any], author: Dict[str, Any], reason: str,
                       shown: bool = True) -> PublishResult:
        """The mind's choice of the common view: show the account (or withdraw it) in place of the named records."""
        with self._publishing() as db:
            target = self._get(db, account_id)
            if target is None:
                return _refuse("target_missing", conflict_ids=(str(account_id),))
            return self._commit(db, [{"kind": "selection", "room_id": target["room_id"], "target_id": target["id"],
                                      "replaces": [str(r) for r in (replaces or ())], "shown": shown, "reason": reason,
                                      "author": author}])

    def correct(self, target_id: str, text: str, author: Dict[str, Any], *,
                expected_revision: Optional[str] = None, expected_sequence: Optional[int] = None) -> PublishResult:
        """The mind's correction beside the original; it acts at once and moves the target room's head."""
        if not isinstance(author, dict) or author.get("kind") != "mind":
            return _refuse("invalid", "only the mind corrects a record")
        with self._publishing() as db:
            target = self._get(db, target_id)
            if target is None:
                return _refuse("target_missing", conflict_ids=(str(target_id),))
            room = target["room_id"]

            def pre():
                refusal = self._head_refusal(db, room, expected_sequence)
                if refusal is not None or target["kind"] not in _CORRECTABLE:
                    return refusal
                current = self._current_revision(db, target["id"])
                if expected_revision is None and current != target["id"]:
                    return _refuse("revision_required", "the target already has a correction: pass "
                                   "expected_revision", current_revision=current)
                if expected_revision is not None and expected_revision != current:
                    return _refuse("revision_conflict", current_revision=current)
                return None

            return self._commit(db, [{"kind": "correction", "room_id": room, "target_id": target["id"],
                                      "text": text, "author": author}], pre=pre)

    def decide(self, draft_id: str, accepted: bool, author: Dict[str, Any], reason: str) -> PublishResult:
        """The mind accepts or rejects a helper's draft page or part, once, with a reason."""
        with self._publishing() as db:
            target = self._get(db, draft_id)
            if target is None:
                return _refuse("target_missing", conflict_ids=(str(draft_id),))
            return self._commit(db, [{"kind": "decision", "room_id": target["room_id"], "target_id": target["id"],
                                      "accepted": bool(accepted), "reason": reason, "author": author}])

    def mark(self, target_ref: Dict[str, Any], text: str, author: Dict[str, Any], *, room_id: Any,
             scope: str = "room", quote: Optional[str] = None, record_id: Optional[str] = None,
             anchor: Optional[Dict[str, Any]] = None) -> PublishResult:
        record = {"kind": "mark", "room_id": str(room_id), "target_ref": target_ref, "text": text,
                  "author": author, "scope": scope, "quote": quote, "visibility": "full"}
        if anchor is not None:
            record["anchor"] = anchor  # an action the mark meets me at (``mark_anchors``)
        if record_id:
            record["id"] = str(record_id)
        return self.publish([record])

    def set_mark_view(self, mark_id: str, visibility: str, author: Dict[str, Any], reason: str) -> PublishResult:
        return self._mark_decision("mark_view", mark_id, author, reason, visibility=visibility)

    def release_mark(self, mark_id: str, author: Dict[str, Any], reason: str) -> PublishResult:
        return self._mark_decision("mark_release", mark_id, author, reason)

    def _mark_decision(self, kind, mark_id, author, reason, **fields):
        with self._publishing() as db:
            active = db.execute("SELECT room FROM active_marks WHERE id=?", (str(mark_id),)).fetchone()
            room = active[0] if active else ""
            return self._commit(db, [{"kind": kind, "room_id": room, "target_id": str(mark_id),
                                      "author": author, "reason": reason, **fields}])

    def _commit(self, db, records, scan_state=None, *, quotes_verified=False, pre=None) -> PublishResult:
        if scan_state is not None and not isinstance(scan_state, dict):
            return _refuse("invalid", "scan_state is an object merged by top-level key")
        if not isinstance(records, (list, tuple)) or not all(isinstance(r, dict) for r in records):
            return _refuse("invalid", "records is a list of objects")
        if any(r.get("kind") == "activation" for r in records):
            active = self._state(db, "activation")
            if active:
                return PublishResult(True, "unchanged", self._get(db, active["id"]) or active)
        prepared, identities = [], {}
        for supplied in records:
            record = json.loads(_json(supplied))
            record["id"] = str(record.get("id") or uuid.uuid4().hex)
            record["room_id"] = str(record.get("room_id", ""))
            old = db.execute("SELECT body FROM records WHERE id=?", (record["id"],)).fetchone()
            old_record = json.loads(old[0]) if old else identities.get(record["id"])
            record.setdefault("ts", old_record["ts"] if old_record else utc_now_iso())
            if old_record is not None and old_record != record:
                return _refuse("invalid", "identity collision: this id already holds a different record",
                               conflict_ids=(record["id"],))
            if record["id"] not in identities:
                prepared.append(record)
                identities[record["id"]] = record
        new = [r for r in prepared if not db.execute("SELECT 1 FROM records WHERE id=?", (r["id"],)).fetchone()]
        scan = self._state(db, "scan", {})
        changed = scan_state is not None and {**scan, **scan_state} != scan
        last = prepared[-1] if prepared else None
        if not new and not changed:
            return PublishResult(True, "unchanged", self._get(db, last["id"]) if last else None)
        if not db.in_transaction:  # lock-free readers drain BEFORE the authority append: busy refuses here, not after it
            db.execute("BEGIN EXCLUSIVE")
        refusal = pre() if pre is not None and new else None
        if refusal is None:
            refusal = self._trial(db, new, quotes_verified)
        if refusal is not None:
            return refusal
        tx = {"kind": "transaction", "schema": 1, "records": new, "scan_state": scan_state}
        if not append_jsonl(self.log_path, tx, ensure_record_boundary=True, require_lock=True):
            raise OSError("chronicle transaction append failed")
        with self.log_path.open("ab") as stream:
            os.fsync(stream.fileno())
        self._project(db, tx)
        self._set_state(db, "offset", self.log_path.stat().st_size)
        db.commit()
        saved = self._get(db, last["id"]) if last else None
        head = self._head(db, saved["room_id"]) if saved and saved["kind"] in HEAD_KINDS else None
        revision = saved["id"] if saved and saved["kind"] in _CORRECTABLE | {"correction"} else None
        return PublishResult(True, "saved", saved, current_revision=revision, current_head=head)

    def _trial(self, db, records, quotes_verified) -> Optional[PublishResult]:
        """Check every record against the index plus the batch's earlier records, then roll back."""
        db.execute("SAVEPOINT chronicle_trial")
        try:
            for record in records:
                refusal = self._rules(db, record, quotes_verified)
                if refusal is not None:
                    return refusal
                try:
                    self._apply(db, record)
                except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                    return _refuse("invalid", f"{record.get('kind')} {record['id']}: {exc}")
            return None
        finally:
            db.execute("ROLLBACK TO chronicle_trial")
            db.execute("RELEASE chronicle_trial")

    # --- preconditions -----------------------------------------------------------------------------

    def _rules(self, db, record, quotes_verified) -> Optional[PublishResult]:
        kind = record.get("kind")
        if kind not in KINDS:
            return _refuse("invalid", f"unknown record kind {kind!r}")
        problem = _author_problem(record.get("author"))
        if problem:
            return _refuse("invalid", problem)
        if kind in _TEXT_KINDS and not _nonblank(record.get("text")):
            return _refuse("invalid", f"a {kind} needs text")
        if kind in ("page", "part"):
            if record["author"]["kind"] not in ("mind", "helper"):
                return _refuse("invalid", f"a {kind} is written by the mind, or drafted by a helper")
            if record.get("quotes") and not quotes_verified:
                return _refuse("quote_mismatch", "quotes are verified by publish_page/publish_part")
        rule = getattr(self, "_rule_" + kind, None)
        return rule(db, record) if rule else None

    def _rule_page(self, db, record):
        if record.get("metadata", {}).get("candidate") and record.get("author", {}).get("kind") != "helper":
            return _refuse("invalid", "a candidate is a signed helper draft awaiting the mind's decision")
        covers, room = record.get("covers"), record["room_id"]
        rows = covers.get("rows") if isinstance(covers, dict) else None
        if (not isinstance(rows, list) or not rows or not all(_nonblank(r) for r in rows)
                or len(set(rows)) != len(rows)):
            return _refuse("invalid", "covers.rows is the page's non-empty set of distinct row refs")
        if str(covers.get("room_id", room)) != room:
            return _refuse("invalid", "covers.room_id is the page's room")
        span = covers.get("stream_span")
        if not (isinstance(span, list) and len(span) == 2 and all(type(v) is int for v in span)):
            return _refuse("invalid", "covers.stream_span is [first, last] stream position; it orders pages")
        sealed = set()
        for start in range(0, len(rows), _CHUNK):
            chunk = rows[start:start + _CHUNK]
            sealed.update(page for (page,) in db.execute(
                "SELECT DISTINCT page_id FROM page_rows WHERE room=? AND row_ref IN (%s)" % ",".join("?" * len(chunk)),
                (room, *chunk)))
        if sealed:
            return _refuse("already_sealed", "the page's set intersects rows another page of this room seals",
                           conflict_ids=self._by_sequence(db, sealed))
        return None

    def _rule_part(self, db, record):
        covers, room = record.get("covers"), record["room_id"]
        ids = covers.get("member_ids") if isinstance(covers, dict) else None
        if not isinstance(ids, list) or not ids or not all(_nonblank(i) for i in ids) or len(set(ids)) != len(ids):
            return _refuse("invalid", "member_ids is a non-empty list of distinct record ids")
        members = [self._get(db, i) for i in ids]
        missing = tuple(i for i, m in zip(ids, members) if m is None)
        if missing:
            return _refuse("target_missing", conflict_ids=missing)
        kinds = {m["kind"] for m in members}
        if len(kinds) != 1 or not kinds <= _FOLDABLE or any(m["room_id"] != room for m in members):
            return _refuse("invalid", "members are all pages, all legacy sections or all parts of the part's room")
        if any(self._status(db, m) in ("rejected", "candidate") for m in members):
            return _refuse("invalid", "a rejected or pending candidate draft does not act and cannot be folded")
        holders = {part for (part,) in db.execute(
            "SELECT part_id FROM folded WHERE member_id IN (%s)" % ",".join("?" * len(ids)), ids)}
        if holders:
            return _refuse("already_folded", "a member is already folded into a part",
                           conflict_ids=self._by_sequence(db, holders))
        run = self._unfolded(db, room, kinds.pop())
        places = sorted(run.index(i) for i in ids)
        if places != list(range(places[0], places[0] + len(places))):
            return _refuse("invalid", "members are adjacent in their room's order of unfolded records")
        return None

    def _rule_note(self, db, record):
        if record["author"]["kind"] != "mind" or not _nonblank(record.get("task_id")):
            return _refuse("invalid", "a note is the mind's, with its task_id")
        return None

    def _rule_correction(self, db, record):
        if record["author"]["kind"] != "mind":
            return _refuse("invalid", "only the mind corrects a record")
        target = self._get(db, record.get("target_id", ""))
        if target is None:
            return _refuse("target_missing", conflict_ids=(str(record.get("target_id", "")),))
        if target["kind"] not in _CORRECTABLE or target["room_id"] != record["room_id"]:
            return _refuse("invalid", "a correction targets a page, part, note, legacy or gap record of its room")
        return None

    def _rule_decision(self, db, record):
        if record["author"]["kind"] != "mind" or not _nonblank(record.get("reason")):
            return _refuse("invalid", "a decision is the mind's, with a reason")
        if not isinstance(record.get("accepted"), bool):
            return _refuse("invalid", "accepted is true or false")
        target = self._get(db, record.get("target_id", ""))
        if target is None:
            return _refuse("target_missing", conflict_ids=(str(record.get("target_id", "")),))
        if (target["kind"] not in ("page", "part") or target.get("author", {}).get("kind") != "helper"
                or target["room_id"] != record["room_id"]):
            return _refuse("invalid", "only a helper's draft page or part takes a decision")
        prior = db.execute("SELECT id FROM records WHERE target=? AND kind='decision'", (target["id"],)).fetchone()
        if prior:
            return _refuse("invalid", "the draft is already decided", conflict_ids=(prior[0],))
        if record["accepted"] and target.get("metadata", {}).get("candidate"):
            return self._rule_page(db, target)  # adoption must still have exclusive coverage
        holder = db.execute("SELECT part_id FROM folded WHERE member_id=?", (target["id"],)).fetchone()
        if holder and not record["accepted"]:
            # A rejected draft stops acting, so it cannot stay a member of an acting part.
            return _refuse("already_folded", "the draft is folded into a part: reject that part first when it "
                           "is a helper's draft, or correct this draft beside its original", conflict_ids=(holder[0],))
        return None

    def _source_version(self, db, given) -> Tuple[Optional[Dict[str, Any]], Optional[PublishResult]]:
        """The frozen form of one source reference (``{id, revision?}`` or an id), or a typed refusal.

        The revision is the one named (the record's own id or one of its corrections'; unnamed, the
        current one); the body's sha256 and a draft's status are the record's as they are now.
        """
        ref = _mapping(given) if isinstance(given, Mapping) else {"id": given}
        found = self._get(db, ref.get("id", ""))
        if found is None:
            return None, _refuse("target_missing", "a source is a record of this chronicle", conflict_ids=(str(ref.get("id")),))
        revisions = self._revisions(db, found["id"])
        cited = ref.get("revision") or revisions[-1]
        if cited not in revisions:
            return None, _refuse("revision_conflict", f"{cited} is not a revision of {found['id']}",
                                 current_revision=revisions[-1], conflict_ids=(found["id"],))
        frozen = {"id": found["id"], "kind": found["kind"], "room_id": found["room_id"], "revision": cited,
                  "sha256": body_sha256(found)}
        status = self._status(db, found)
        return ({**frozen, "status": status} if status else frozen), None

    def _rule_account(self, db, record):
        if record["author"]["kind"] != "mind":
            return _refuse("invalid", "an account is the mind's own text")
        refs = record.get("sources")
        if not isinstance(refs, list) or not refs or len({str(_mapping(r).get("id")) for r in refs}) != len(refs):
            return _refuse("invalid", "sources is a non-empty list of distinct source versions (publish_account fills them)")
        for ref in refs:
            frozen, refusal = self._source_version(db, ref)
            if refusal is not None:
                return refusal
            if {key: _mapping(ref).get(key) for key in frozen} != frozen:  # a hand-made reference is checked, never trusted
                return _refuse("invalid", f"source {frozen['id']}: kind, room, sha256 and status are the record's own")
        return None

    def _rule_selection(self, db, record):
        if record["author"]["kind"] != "mind" or not _nonblank(record.get("reason")):
            return _refuse("invalid", "a selection is the mind's, with a reason")
        if not isinstance(record.get("shown"), bool):
            return _refuse("invalid", "shown is true or false")
        target = self._get(db, record.get("target_id", ""))
        if target is None:
            return _refuse("target_missing", conflict_ids=(str(record.get("target_id", "")),))
        if target["kind"] != "account":
            return _refuse("invalid", "a selection chooses the common view of an account")
        ids = record.get("replaces")
        if not isinstance(ids, list) or len(set(map(str, ids))) != len(ids) or target["id"] in map(str, ids):
            return _refuse("invalid", "replaces is a list of distinct record ids other than the account itself")
        missing = tuple(str(i) for i in ids if self._get(db, i) is None)
        if missing:
            return _refuse("target_missing", "replaces names records of this chronicle", conflict_ids=missing)
        return None

    def _rule_mark(self, db, record):
        if record.get("scope", "room") not in ("room", "global"):
            return _refuse("invalid", "scope is room or global")
        if "anchor" in record:
            from ouroboros.mark_anchors import anchor_problem

            if problem := anchor_problem(record["anchor"]):
                return _refuse("invalid", problem)
        target = record.get("target_ref")
        if not isinstance(target, dict) or not target:
            return _refuse("invalid", "target_ref names a chronicle record, chat row, task or retained source")
        if target.get("kind") == "chronicle":
            found = self._get(db, target.get("id", ""))
            if found is None:
                return _refuse("target_missing", conflict_ids=(str(target.get("id", "")),))
            quote = record.get("quote")
            texts = (found.get("text", ""), self._interpret(db, found)["current_text"])
            if quote and not any(quote in str(text) for text in texts):
                return _refuse("quote_mismatch", "the quote is not an exact substring of the marked record")
        return None

    def _rule_mark_view(self, db, record):
        if db.execute("SELECT 1 FROM active_marks WHERE id=?", (record.get("target_id", ""),)).fetchone() is None:
            return _refuse("target_missing", "no active mark has this id", conflict_ids=(str(record.get("target_id")),))
        if record.get("visibility") not in ("full", "meaning") or not _nonblank(record.get("reason")):
            return _refuse("invalid", "a mark view is full or meaning, with a reason")
        return None

    def _rule_mark_release(self, db, record):
        active = db.execute("SELECT body FROM active_marks WHERE id=?", (record.get("target_id", ""),)).fetchone()
        if active is None:
            return _refuse("target_missing", "no active mark has this id", conflict_ids=(str(record.get("target_id")),))
        if not _nonblank(record.get("reason")):
            return _refuse("invalid", "a mark release needs an explicit reason")
        author = record["author"]
        if is_late_review_mark(json.loads(active[0])) and (
                author["kind"] != "mind" or author["focus"]["role"] in DELEGATED_FOCUS_ROLES):
            return _refuse("not_integrator", "late review feedback the host keeps in view is released by the "
                           "integrating mind with its reason; a delegated helper reports it instead",
                           conflict_ids=(str(record["target_id"]),))
        return None

    def _head(self, db, room) -> int:
        return _lookups(db).head(room)

    def _head_refusal(self, db, room, expected) -> Optional[PublishResult]:
        if expected is None:
            return None
        head = self._head(db, room)
        if type(expected) is not int or expected < 0:
            return _refuse("invalid", "expected_sequence is the room head (an integer) the text is based on",
                           current_head=head)
        if expected == head:
            return None
        newer = tuple(rid for (rid,) in db.execute(
            "SELECT id FROM records WHERE room=? AND sequence>? AND kind IN (%s) ORDER BY sequence"
            % ",".join("?" * len(HEAD_KINDS)), (room, expected, *HEAD_KINDS)))
        return _refuse("revision_conflict", "the room changed since that head", current_head=head, conflict_ids=newer)

    def _current_revision(self, db, record_id) -> str:
        fixes = _lookups(db).targeting(record_id, ("correction",))
        return fixes[-1][1]["id"] if fixes else record_id

    def _revisions(self, db, record_id) -> List[str]:
        """The record's own id, then each correction's, in publication order: every revision it ever had."""
        return [str(record_id)] + [fix["id"] for _seq, fix in _lookups(db).targeting(record_id, ("correction",))]

    # --- order, interpretation ---------------------------------------------------------------------

    def _get(self, db, record_id) -> Optional[Dict[str, Any]]:
        return _lookups(db).get(record_id)

    @staticmethod
    def _by_sequence(db, ids) -> Tuple[str, ...]:
        ids = list(ids)
        return tuple(rid for (rid,) in db.execute(
            "SELECT id FROM records WHERE id IN (%s) ORDER BY sequence" % ",".join("?" * len(ids)), ids))

    def _status(self, db, record) -> Optional[str]:
        if record.get("kind") not in ("page", "part"):
            return None
        if record.get("author", {}).get("kind") != "helper":
            return "final"
        decisions = _lookups(db).targeting(record["id"], ("decision",))
        if not decisions:
            return "candidate" if record.get("metadata", {}).get("candidate") else "draft"
        return "accepted" if decisions[-1][1].get("accepted") else "rejected"

    def _order_key(self, db, record, cache) -> Tuple[Any, ...]:
        """Legacy sections by block, pages by ``stream_span[0]``, a part by its first member."""
        if record["id"] not in cache:
            covers, kind = record.get("covers") or {}, record["kind"]
            if kind == "part":
                first = self._get(db, (covers.get("member_ids") or [""])[0])
                key = self._order_key(db, first, cache) if first else (2, 0, record["sequence"])
            elif kind == "legacy":
                block = (record.get("metadata") or {}).get("legacy_block")
                key = (0, block if type(block) is int else _NO_BLOCK, record["sequence"])
            else:
                key = (1, (covers.get("stream_span") or [0])[0], record["sequence"])
            cache[record["id"]] = key
        return cache[record["id"]]

    def _unfolded(self, db, room, kind) -> List[str]:
        rows = [{**json.loads(body), "sequence": seq} for seq, body in db.execute(
            "SELECT sequence,body FROM records WHERE room=? AND kind=? AND id NOT IN (SELECT member_id FROM folded) "
            "ORDER BY sequence", (room, kind))]
        cache: Dict[str, Tuple[Any, ...]] = {}
        effective = [r for r in rows if self._status(db, r) not in ("rejected", "candidate")]
        return [r["id"] for r in sorted(effective, key=lambda r: self._order_key(db, r, cache))]

    @staticmethod
    def _stream_span(record) -> Optional[List[int]]:
        covers = record.get("covers") or {}
        span = covers.get("stream_span") or (covers.get("raw_range") or {}).get("pos")
        return span if isinstance(span, list) and len(span) == 2 and all(type(v) is int for v in span) else None

    def _part_covers(self, db, member_ids) -> Dict[str, Any]:
        if not isinstance(member_ids, (list, tuple)):
            return {"member_ids": member_ids}
        members = [self._get(db, i) for i in member_ids]
        if any(m is None for m in members) or len({m["kind"] for m in members}) != 1:
            return {"member_ids": [str(i) for i in member_ids]}
        cache: Dict[str, Tuple[Any, ...]] = {}
        members.sort(key=lambda m: self._order_key(db, m, cache))
        covers: Dict[str, Any] = {"member_ids": [m["id"] for m in members]}
        spans = [s for s in (self._stream_span(m) for m in members) if s]
        if spans:
            covers["stream_span"] = [min(s[0] for s in spans), max(s[1] for s in spans)]
        # A legacy section keeps its period inside its raw_range; a page or part, in covers.
        stamps = [(m.get("covers") or {}).get("ts_span") or ((m.get("covers") or {}).get("raw_range") or {}).get(
            "ts_span") or {} for m in members]
        covers["ts_span"] = source_time_span([t.get(k) for t in stamps for k in ("start", "end") if t.get(k)],
                                             incomplete=any(not t or t.get("incomplete") for t in stamps))
        return covers

    def _interpret(self, db, record, *, revision: Optional[str] = None, sources: bool = True) -> Dict[str, Any]:
        """The record as it acts: its own words, then every correction of the mind under them, each signed.

        ``current_text`` keeps the original and adds each correction in publication order under
        ``correction_line``; the author stays the record's. ``revision`` is the last correction's id
        (what ``expected_revision`` names) and ``corrections`` lists them all. A correction never
        replaces a record's words: to say a record anew the mind folds it into a part over it.
        As of a ``revision`` (its own id or a correction's; else ValueError) only corrections up to it, the rest named in
        ``later_corrections``; without ``sources`` an account keeps frozen references; a mark gets ``view``/``released``.
        """
        record, look = dict(record), _lookups(db)
        fixes = [fix for _seq, fix in look.targeting(record["id"], ("correction",))]
        if revision is not None:
            revisions = [record["id"], *(fix["id"] for fix in fixes)]
            if revision not in revisions:
                raise ValueError(f"{revision} is not a revision of {record['id']}; its revisions: " + ", ".join(revisions))
            fixes, record["later_corrections"] = fixes[:revisions.index(revision)], revisions[revisions.index(revision) + 1:]
        record["current_text"] = "\n\n".join([str(record.get("text", "")),
                                              *(f"{correction_line(fix)}\n{fix.get('text', '')}" for fix in fixes)])
        record["revision"] = fixes[-1]["id"] if fixes else record["id"]
        record["corrections"] = [{"id": fix["id"], "author": fix.get("author"), "ts": fix.get("ts")} for fix in fixes]
        status = self._status(look, record)
        if status:
            record["status"] = status
        record["folded_into"] = look.part_of(record["id"])
        if record.get("kind") == "account":
            if sources:
                record["sources"] = [self._source_state(look, ref) for ref in record.get("sources") or ()]
            record["selection"] = self._selection_of(look, record["id"])
        elif record.get("kind") == "mark":
            decided = look.targeting(record["id"], ("mark_view", "mark_release"))[::-1]
            record["view"], record["released"] = (next((e for _s, e in decided if e["kind"] == k), None) for k in ("mark_view", "mark_release"))
        return record

    def _source_state(self, db, ref) -> Dict[str, Any]:
        """A frozen source reference plus what changed since: the account's basis stays what it was.

        ``current_revision``, ``later_corrections`` (after the cited revision), ``status_now`` (a
        draft's), ``folded_into`` now, and ``missing`` when the record is not in this chronicle;
        ``revision_known`` is false when the cited revision is not one this journal holds.
        """
        state, db = {key: _mapping(ref).get(key) for key in (*_SOURCE_FIELDS, "status")}, _lookups(db)
        found = self._get(db, state["id"] or "")
        if found is None:
            return {**state, "missing": True, "revision_known": False, "later_corrections": [], "current_revision": None,
                    "status_now": None, "folded_into": None}
        revisions = self._revisions(db, found["id"])
        known = state["revision"] in revisions
        later = revisions[revisions.index(state["revision"]) + 1:] if known else []
        return {**state, "missing": False, "revision_known": known, "current_revision": revisions[-1],
                "later_corrections": later, "status_now": self._status(db, found), "folded_into": db.part_of(found["id"]),
                "nested_changes": self._nested_source_changes(db, found)}

    def _nested_source_changes(self, db, source) -> List[Dict[str, Any]]:
        """Changes inside retained account/part sources, independent of current display or fold ownership.

        Account edges retain exact cited revisions/statuses. Parts retain member IDs only, so their
        members' corrections and decisions are reported without inventing a revision read by the part.
        Visit only this source graph; a repeated node's immutable edges need expanding just once.
        """
        changes, todo, expanded, db = [], [(source, [])], set(), _lookups(db)
        while todo:
            parent, path = todo.pop()
            if parent["id"] in expanded:
                continue
            expanded.add(parent["id"])
            if parent["kind"] == "account":
                refs = parent.get("sources") or ()
            elif parent["kind"] == "part":
                refs = [{"id": ident} for ident in (parent.get("covers") or {}).get("member_ids") or ()]
            else:
                continue
            via = [*path, parent["id"]]
            for ref in refs:
                member = self._get(db, ref["id"])
                if member is None:
                    continue
                revision = ref.get("revision")
                revisions = self._revisions(db, member["id"])
                later = (revisions[revisions.index(revision) + 1:] if revision in revisions else
                         revisions[1:] if revision is None else [])
                for _seq, event in db.targeting(member["id"], ("correction", "decision")):
                    if (event["kind"] == "correction" and event["id"] not in later
                            or event["kind"] == "decision" and revision is not None
                            and ref.get("status") == self._status(db, member)):
                        continue
                    changes.append({"source_id": member["id"], "kind": member["kind"], "room_id": member["room_id"],
                                    "via": via, "revision": revision, "event": event})
                todo.append((member, via))
        return changes

    def _selection_of(self, db, account_id) -> Optional[Dict[str, Any]]:
        """The latest selection of an account (the one that acts), or None while it was never selected."""
        selections = _lookups(db).targeting(account_id, ("selection",))
        if not selections:
            return None
        sequence, chosen = selections[-1]
        return {"id": chosen["id"], "shown": bool(chosen.get("shown")), "replaces": [str(r) for r in chosen.get("replaces") or ()],
                "reason": str(chosen.get("reason") or ""), "ts": chosen.get("ts"), "sequence": sequence}

    # --- the dialogue tail's facts: a projection of state/dialog.sqlite kept beside the journal's ------

    def tail_facts(self, source: Mapping[str, Any], *,
                   terminal_only: bool = False) -> Optional[Tuple[Dict[str, Any], List[List[Any]]]]:
        """``(state, rows)`` of the stored facts of the chat rows after the legacy frontier (only those with a
        terminal status: ``terminal_only``), when they were projected from ``source`` (``memory_inventory`` names
        the dialogue store and frontier, projects the rows and checks the state's ``anchor``); rows
        ``[pos, seq, row_sha256, legacy, meta, terminal]`` by position."""
        snapshot = self._pinned()  # a capture that holds less of the tail than asked leaves it to the index
        with self._reading(pinned=snapshot is not None and snapshot.holds_tail(terminal_only)) as look:
            state = look.state("tail")
            return ((state, look.tail(terminal_only)) if isinstance(state, dict) and state.get("source") == source
                    else None)

    def tail_state(self) -> Any:
        with self._reading() as look:
            return look.state("tail")

    def extend_tail(self, expected: Any, rows: Iterable[List[Any]], state: Dict[str, Any], *, replace: bool = False,
                    wait_sec: float = 0) -> bool:
        """Store projected tail rows and the ``state`` naming them in ONE index transaction, only while the stored
        state is still the ``expected`` one its writer read (compare-and-set: concurrent writers in several
        processes store each row once and never leave a hole). ``replace`` drops the stored rows first. A lock
        held past ``wait_sec`` or a refused write stores nothing: the rows wait for the next writer."""
        try:
            with self._publishing(wait_sec) as db:
                if self._state(db, "tail") != expected:
                    return False
                if replace:
                    db.execute("DELETE FROM tail_rows")
                chunk = [[pos, seq, sha, None if legacy is None else _json(legacy), _json(meta), terminal]
                         for pos, seq, sha, legacy, meta, terminal in rows]
                if chunk:
                    db.execute(_TAIL_INSERT, (_json(chunk),))
                self._set_state(db, "tail", state)
                db.commit()
                return True
        except (TimeoutError, sqlite3.Error):
            return False

    # --- reads -------------------------------------------------------------------------------------

    @contextmanager
    def _reading(self, *, pinned: bool = True):
        """A reader's lookups: the pinned capture's snapshot, else the index on its read path (``_index``)."""
        snapshot = self._pinned() if pinned else None
        if snapshot is not None:
            yield snapshot
            return
        with self._index() as db:
            yield _IndexLookups(db)

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._reading() as look:
            return look.get(record_id)

    def rejected_input(self, room_id: Any, unit_key: str, input_sha256: str,
                       light_binding: Mapping[str, Any]) -> bool:
        """A rejected helper attempt at this exact input/route, including old published drafts.

        Read the addressed room's indexed draft metadata and latest decision;
        never recreate a lost receipt or append published drafts to scan state.
        """
        with self._index() as db:
            rows = db.execute(
                "SELECT body FROM records WHERE room=? AND kind IN ('page','part') "
                "AND json_extract(body, '$.metadata.unit.key')=? "
                "AND json_extract(body, '$.metadata.input_sha256')=?",
                (str(room_id), unit_key, input_sha256))
            for (body,) in rows:
                record = json.loads(body)
                if ((record.get("metadata") or {}).get("light_binding") == dict(light_binding)
                        and self._status(db, record) == "rejected"):
                    return True
        return False

    def records(self, room_id: Any = None, *, kinds: Optional[Iterable[str]] = None, after_seq: int = 0,
                limit: Optional[int] = None, author_kind: Optional[str] = None, unfolded: bool = False) -> List[Dict[str, Any]]:
        """Stored records in order, filtered by the index (room, ``kinds``, author kind, unfolded); ``limit`` bounds the statement."""
        with self._reading() as look:
            return [{**json.loads(body), "sequence": seq} for seq, body in look.rows(
                room_id, tuple(kinds or ()), after_seq, -1 if limit is None else int(limit), author_kind, unfolded)]

    def interpret(self, records: Iterable[Mapping[str, Any]], *, revision: Optional[str] = None,
                  sources: bool = True) -> List[Dict[str, Any]]:
        """Records as read interpreted in one read (``_interpret``, as of ``revision``): ``memory_read`` and
        ``memory_reader`` share it, so one rule composes a record's words at every revision."""
        with self._reading() as look:
            return [self._interpret(look, record, revision=revision, sources=sources) for record in records]

    def interpreted(self, room_id: Any = None, *, kinds: Iterable[str] = (), after_seq: int = 0,
                    limit: Optional[int] = None, author_kind: Optional[str] = None, unfolded: bool = False,
                    sources: bool = True) -> List[Dict[str, Any]]:
        """``records`` interpreted in the same one read: a listing costs one index open, never one per record."""
        with self._reading() as look:
            return [self._interpret(look, {**json.loads(body), "sequence": seq}, sources=sources) for seq, body in look.rows(
                room_id, tuple(kinds), after_seq, -1 if limit is None else int(limit), author_kind, unfolded)]

    def targeting(self, record_id: Any, kinds: Iterable[str], *, after_seq: int = 0,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """At most ``limit`` records of ``kinds`` aimed at one record after ``after_seq``: addressed, never its room's."""
        with self._index() as db:  # an extension's read, outside any capture
            return [{**body, "sequence": seq} for seq, body in
                    _IndexLookups(db).targeting(record_id, tuple(kinds), after_seq, -1 if limit is None else int(limit))]

    def rooms(self, *, after_room: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rooms holding records in room-id order (the host's room ``""`` first), after ``after_room`` when named."""
        with self._index() as db:
            return [{"room_id": room, "records": count, "newest_sequence": newest, "head": _IndexLookups(db).head(room)}
                    for room, count, newest in db.execute("SELECT room,COUNT(*),MAX(sequence) FROM records WHERE ? IS NULL "
                                                          "OR room>? GROUP BY room ORDER BY room LIMIT ?",
                                                          (after_room, after_room, -1 if limit is None else int(limit)))]

    def telling_selections(self, record_ids: Iterable[Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Per record, the ``told_map`` entries that decide whether an account tells it in the common view: the acting
        shown selections naming it (the ``told`` projection) and, for an account, its own; indexed, never every one."""
        out: Dict[str, List[Dict[str, Any]]] = {str(ident): [] for ident in record_ids}
        with self._index() as db:
            for record, account, seq in db.execute("SELECT record,account,sequence FROM told WHERE record IN "
                                                   "(SELECT value FROM json_each(?))", (_json(sorted(out)),)):
                out[record].append({"id": account, "selection_sequence": seq, "replaces": [record]})
            for ident, entries in out.items():
                own = self._selection_of(db, ident)
                entries += [{"id": ident, "selection_sequence": own["sequence"], "replaces": []}] if own and own["shown"] else []
        return out

    def room_records(self, room_id: Any, *, after_seq: int = 0,
                     kinds: Iterable[str] = STORY_KINDS) -> List[Dict[str, Any]]:
        """A room's (``None``: every room's) pages, parts, accounts, notes, legacy sections and gaps (of ``kinds``),
        without rejected drafts and candidates, in publication order.

        Each is interpreted (``_interpret``): ``current_text`` with the mind's corrections under
        the original, ``revision``, ``corrections``, ``status`` for pages and parts and ``folded_into``.
        """
        return [r for r in self.interpreted(room_id, kinds=kinds, after_seq=after_seq)
                if r.get("status") not in ("rejected", "candidate")]

    def pages_of_room(self, room_id: Any) -> List[Dict[str, Any]]:
        """Effective pages and parts of a room, interpreted, in story order."""
        with self._reading() as look:
            rows = [{**json.loads(body), "sequence": seq} for seq, body in look.rows(room_id, ("page", "part"))]
            cache: Dict[str, Tuple[Any, ...]] = {}
            effective = [r for r in (self._interpret(look, row) for row in rows) if r.get("status") not in ("rejected", "candidate")]
            return sorted(effective, key=lambda r: self._order_key(look, r, cache))

    def accounts(self) -> List[Dict[str, Any]]:
        """Every account, interpreted (sources with their later changes, the acting selection), in publication order."""
        return self.interpreted(kinds=("account",))

    def folded_members(self, part_id: Any) -> List[str]:
        """Every record under a part through nested parts: its own members first, then theirs (no member read whole)."""
        with self._reading() as look:
            out, todo = [], [str(part_id)]
            while todo:
                members = look.members_of(todo.pop(0))
                out += members
                todo += members
            return out

    def acting_pages(self, room_id: Any = None) -> List[Dict[str, Any]]:
        """The acting pages (one room's), in publication order: those whose rows a seal holds, folded ones too,
        with their author's kind, publication time and covered period's end, and no body read whole."""
        with self._reading() as look:
            return [{"room_id": room, "id": ident, "author_kind": kind, "ts": ts, "covered_end": end}
                    for _seq, room, ident, kind, ts, end in look.acting_pages(room_id)]

    def acting_parts(self) -> int:
        """How many parts act (fold their members, and no pending candidate): every room's, folded ones too."""
        with self._reading() as look:
            return sum(self._status(look, {"id": ident, "kind": "part", "author": {"kind": kind},
                                           "metadata": {"candidate": candidate}}) not in ("rejected", "candidate")
                       for ident, kind, candidate in look.acting_parts())

    def sealed_row_refs(self, room_id: Any) -> set:
        """Row refs (``row_sha256`` and ``note:<id>``) the room's acting pages seal; the rest is open."""
        with self._reading() as look:
            return look.sealed(room_id)

    def legacy_pointer_rows(self) -> List[Dict[str, Any]]:
        """One compact row per legacy section or gap: room, label, period and how to open it."""
        pointers = []
        for row in self.interpreted(kinds=("legacy", "gap")):
            meta = row.get("metadata") or {}
            pointers.append({"node_id": row["id"], "kind": row["kind"], "room_id": row["room_id"],
                             "label": meta.get("label"), "legacy_type": meta.get("legacy_type"),
                             "legacy_block": meta.get("legacy_block"), "range_text": meta.get("legacy_range_text"),
                             "messages": meta.get("room_message_count"),  # the old writer's own count for the room
                             "covers": row.get("covers"), "revision": row["revision"],
                             "folded_into": row["folded_into"], "sequence": row["sequence"]})
        return sorted(pointers, key=lambda p: (p["legacy_block"] if type(p["legacy_block"]) is int else _NO_BLOCK,
                                               p["room_id"], p["sequence"]))

    def active_marks(self, room_id: Any = None, include_global: bool = True) -> List[Dict[str, Any]]:
        """A room's acting marks (and the global ones); ``room_id=None`` is every room's."""
        with self._reading() as look:
            return [json.loads(body) for body in look.marks(room_id, include_global)]

    def observation_snapshot(self, boundary: Optional[Dict[str, Any]] = None, *, limit: Optional[int] = None):
        """New immutable records (the first ``limit`` of them) plus the newest position, under one read lock.

        Without a previous position, establish a baseline using only the index;
        the caller discloses that earlier changes were not inventoried.
        """
        with self._index() as db:
            latest = db.execute("SELECT sequence,id FROM records ORDER BY sequence DESC LIMIT 1").fetchone()
            current = {"sequence": latest[0] if latest else 0, "record_id": latest[1] if latest else None}
            if boundary is None:
                return [], current
            sequence = boundary.get("sequence")
            if type(sequence) is not int or sequence < 0:
                raise ValueError("invalid accepted memory sequence")
            anchor = db.execute("SELECT id FROM records WHERE sequence=?", (sequence,)).fetchone()
            if sequence and (not anchor or anchor[0] != boundary.get("record_id")):
                raise ValueError("accepted memory boundary no longer matches source")
            rows = db.execute("SELECT sequence,body FROM records WHERE sequence>? ORDER BY sequence LIMIT ?", (sequence, -1 if limit is None else int(limit)))
            return [{**json.loads(body), "sequence": seq} for seq, body in rows], current

    def scan_state(self) -> Dict[str, Any]:
        with self._reading() as look:
            return look.state("scan", {})

    def activation(self) -> Optional[Dict[str, Any]]:
        with self._reading() as look:
            return look.state("activation")

    def ensure_activated(self, *, wait: bool = False) -> Dict[str, Any]:
        """The activation receipt, importing the legacy dialogue memory once (``chronicle_import``)."""
        from ouroboros.chronicle_import import ensure_activated

        return ensure_activated(self, wait=wait)
