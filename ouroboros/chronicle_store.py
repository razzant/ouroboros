"""Append-only derived memory (the chronicle) with a disposable indexed projection.

``memory/chronicle/records.jsonl`` is the sole authority. One line is one
transaction ``{"kind": "transaction", "schema": 1, "records": [...],
"scan_state": {...}}``, appended and fsynced under the short
``.publication.lock``. ``index.sqlite3`` is only a projection: deleting it
replays the journal into the same rows, sequences, page coverage and folds; a
torn or invalid transaction becomes an addressed ``gap`` record and its
neighbours stay intact. ``scan_state`` merges by top-level key, so one writer's
frontier never erases another's.

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
import re
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.utils import append_jsonl, assert_test_data_path, utc_now_iso

SCHEMA_VERSION = 1
KINDS = frozenset({"page", "part", "note", "correction", "decision", "mark", "mark_view", "mark_release",
                   "legacy", "gap", "activation"})
AUTHOR_KINDS = frozenset({"mind", "helper", "host", "legacy_helper"})
SPEAKERS = frozenset({"human", "ouroboros", "child", "host", "helper", "unattributed"})
STORY_KINDS = ("page", "part", "note", "legacy", "gap")
HEAD_KINDS = ("page", "part", "note", "correction", "decision", "legacy", "gap")
_CORRECTABLE = frozenset({"page", "part", "note", "legacy", "gap"})
_FOLDABLE = frozenset({"page", "part", "legacy"})
_TEXT_KINDS = frozenset({"page", "part", "note", "correction", "mark"})
_NO_BLOCK = float("inf")
_CHUNK = 500

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
"""
_RESET = ("DELETE FROM records; DELETE FROM state; DELETE FROM active_marks; DELETE FROM page_rows; "
          "DELETE FROM folded; DELETE FROM sqlite_sequence WHERE name='records'; "
          f"PRAGMA user_version={SCHEMA_VERSION};")

QuoteResolver = Callable[[Any], Optional[Tuple[str, str]]]


class PublishResult(NamedTuple):
    """Outcome of one publication; ``reason`` is ``saved``, ``unchanged`` or a typed refusal.

    Refusals: ``revision_required``, ``revision_conflict``, ``already_sealed``,
    ``already_folded``, ``quote_mismatch``, ``target_missing``, ``invalid``. A
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


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def source_time_span(timestamps: Iterable[Any], *, incomplete: bool = False) -> Dict[str, Any]:
    """Known source bounds, never publication time or proof of continuous coverage."""
    from ouroboros.deadline_utils import parse_deadline_ts

    parsed = [parse_deadline_ts(value) for value in timestamps]
    known = [value for value in parsed if value is not None]
    return {"start": min(known).isoformat() if known else None,
            "end": max(known).isoformat() if known else None,
            "incomplete": bool(incomplete or not known or len(known) != len(parsed))}


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


# A markdown emphasis marker: a run of up to three ``*``/``_`` that opens before a word (nothing
# word-like or ``*`` before it, a non-space after it) or closes after one. ``memory_read``,
# ``2*3`` and a ``* `` bullet carry no marker; ``**but not**`` and ``_this_`` do.
_EMPHASIS = re.compile(r"(?<![\w*])[*_]{1,3}(?![*_\s])|(?<=[^*_\s])[*_]{1,3}(?![\w*])")


def quote_in(quote: str, source: str) -> bool:
    """Whether ``quote`` is ``source``'s words, exact up to markdown emphasis markers on either side.

    Words, punctuation, spacing and order stay exact: only ``**``, ``__`` and a single
    ``*``/``_`` around words are set aside, so a helper that copied a row's words without
    its bold is not refused, while a changed word still is.
    """
    if quote in source:
        return True
    plain = _EMPHASIS.sub("", quote)
    return bool(plain.strip()) and plain in _EMPHASIS.sub("", source)


def verify_quotes(quotes: Any, resolver: Optional[QuoteResolver]) -> Optional[PublishResult]:
    """Each quote is the words of the row at its address (``quote_in``), spoken by the named class.

    ``resolver(address)`` returns ``(rendered row text, author kind)`` or ``None``;
    it reads chat rows, so it runs before the publication lock is taken.
    """
    if not quotes:
        return None
    if not isinstance(quotes, (list, tuple)):
        return _refuse("invalid", "quotes is a list of {address, text, speaker}")
    for index, quote in enumerate(quotes):
        where = f"quotes[{index}]"
        if (not isinstance(quote, dict) or not _nonblank(quote.get("text")) or not quote.get("address")
                or quote.get("speaker") not in SPEAKERS):
            return _refuse("quote_mismatch", f"{where} needs an address, non-empty text and a speaker in "
                           + ", ".join(sorted(SPEAKERS)))
        if resolver is None:
            return _refuse("quote_mismatch", f"{where}: no row resolver was supplied to verify it")
        resolved = resolver(quote["address"])
        if resolved is None:
            return _refuse("quote_mismatch", f"{where}: the address resolves to no chat row")
        rendered, speaker = resolved
        if not quote_in(str(quote["text"]), str(rendered)):
            return _refuse("quote_mismatch", f"{where}: the text is not an exact substring of that row")
        if speaker != quote["speaker"]:
            return _refuse("quote_mismatch", f"{where}: that row is spoken by {speaker}, not {quote['speaker']}")
    return None


class ChronicleStore:
    """One installation's derived pages, parts, notes, corrections, decisions and marks."""

    def __init__(self, data_root: Any):
        self.data_root = pathlib.Path(data_root)
        self.directory = self.data_root / "memory" / "chronicle"
        self.log_path = self.data_root / "memory" / "chronicle" / "records.jsonl"
        self.index_path = self.data_root / "memory" / "chronicle" / "index.sqlite3"
        self.lock_path = self.data_root / "memory" / "chronicle" / ".publication.lock"

    # --- projection --------------------------------------------------------------------------------

    @contextmanager
    def _index(self):
        assert_test_data_path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        fd = acquire_exclusive_file_lock(self.lock_path, timeout_sec=10, stale_sec=90, owner_aware_stale=True)
        if fd is None:
            raise TimeoutError("chronicle publication lock unavailable")
        db = None
        try:
            try:
                db = sqlite3.connect(self.index_path)
                db.executescript(_SCHEMA)
            except sqlite3.DatabaseError:
                if db is not None:
                    db.close()
                # This is only an index. Preserve the authority and reconstruct it.
                self.index_path.unlink(missing_ok=True)
                db = sqlite3.connect(self.index_path)
                db.executescript(_SCHEMA)
            if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                # Only the disposable projection changes schema; replay the authority once.
                db.executescript(_RESET)
            self._catch_up(db)
            yield db
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
            elif kind == "page":
                db.executemany("INSERT INTO page_rows VALUES (?,?,?)",
                               [(room, record_id, str(ref)) for ref in record["covers"]["rows"]])
            elif kind == "part":
                db.executemany("INSERT INTO folded VALUES (?,?)",
                               [(str(member), record_id) for member in record["covers"]["member_ids"]])
            elif kind == "decision" and not record.get("accepted"):
                # A rejected draft stops acting: its rows are open and its members unfolded again.
                db.execute("DELETE FROM page_rows WHERE page_id=?", (record["target_id"],))
                db.execute("DELETE FROM folded WHERE part_id=?", (record["target_id"],))
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
        with self._index() as db:
            return self._commit(db, records, scan_state)

    def room_head(self, room_id: Any) -> int:
        with self._index() as db:
            return self._head(db, str(room_id))

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
        with self._index() as db:
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
        with self._index() as db:
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

    def correct(self, target_id: str, text: str, author: Dict[str, Any], *,
                expected_revision: Optional[str] = None, expected_sequence: Optional[int] = None) -> PublishResult:
        """The mind's correction beside the original; it acts at once and moves the target room's head."""
        if not isinstance(author, dict) or author.get("kind") != "mind":
            return _refuse("invalid", "only the mind corrects a record")
        with self._index() as db:
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
        with self._index() as db:
            target = self._get(db, draft_id)
            if target is None:
                return _refuse("target_missing", conflict_ids=(str(draft_id),))
            return self._commit(db, [{"kind": "decision", "room_id": target["room_id"], "target_id": target["id"],
                                      "accepted": bool(accepted), "reason": reason, "author": author}])

    def mark(self, target_ref: Dict[str, Any], text: str, author: Dict[str, Any], *, room_id: Any,
             scope: str = "room", quote: Optional[str] = None, record_id: Optional[str] = None) -> PublishResult:
        record = {"kind": "mark", "room_id": str(room_id), "target_ref": target_ref, "text": text,
                  "author": author, "scope": scope, "quote": quote, "visibility": "full"}
        if record_id:
            record["id"] = str(record_id)
        return self.publish([record])

    def set_mark_view(self, mark_id: str, visibility: str, author: Dict[str, Any], reason: str) -> PublishResult:
        return self._mark_decision("mark_view", mark_id, author, reason, visibility=visibility)

    def release_mark(self, mark_id: str, author: Dict[str, Any], reason: str) -> PublishResult:
        return self._mark_decision("mark_release", mark_id, author, reason)

    def _mark_decision(self, kind, mark_id, author, reason, **fields):
        with self._index() as db:
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
        if any(self._status(db, m) == "rejected" for m in members):
            return _refuse("invalid", "a rejected draft does not act and cannot be folded")
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
        holder = db.execute("SELECT part_id FROM folded WHERE member_id=?", (target["id"],)).fetchone()
        if holder and not record["accepted"]:
            # A rejected draft stops acting, so it cannot stay a member of an acting part.
            return _refuse("already_folded", "the draft is folded into a part: reject that part first when it "
                           "is a helper's draft, or correct this draft beside its original", conflict_ids=(holder[0],))
        return None

    def _rule_mark(self, db, record):
        if record.get("scope", "room") not in ("room", "global"):
            return _refuse("invalid", "scope is room or global")
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
        if db.execute("SELECT 1 FROM active_marks WHERE id=?", (record.get("target_id", ""),)).fetchone() is None:
            return _refuse("target_missing", "no active mark has this id", conflict_ids=(str(record.get("target_id")),))
        if not _nonblank(record.get("reason")):
            return _refuse("invalid", "a mark release needs an explicit reason")
        return None

    def _head(self, db, room) -> int:
        return db.execute("SELECT COALESCE(MAX(sequence),0) FROM records WHERE room=? AND kind IN (%s)"
                          % ",".join("?" * len(HEAD_KINDS)), (room, *HEAD_KINDS)).fetchone()[0]

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
        row = db.execute("SELECT id FROM records WHERE target=? AND kind='correction' ORDER BY sequence DESC LIMIT 1",
                         (record_id,)).fetchone()
        return row[0] if row else record_id

    # --- order, interpretation ---------------------------------------------------------------------

    def _get(self, db, record_id) -> Optional[Dict[str, Any]]:
        row = db.execute("SELECT sequence,body FROM records WHERE id=?", (str(record_id),)).fetchone()
        return {**json.loads(row[1]), "sequence": row[0]} if row else None

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
        decision = db.execute("SELECT body FROM records WHERE target=? AND kind='decision' ORDER BY sequence DESC "
                              "LIMIT 1", (record["id"],)).fetchone()
        if not decision:
            return "draft"
        return "accepted" if json.loads(decision[0]).get("accepted") else "rejected"

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
        effective = [r for r in rows if self._status(db, r) != "rejected"]
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

    def _interpret(self, db, record) -> Dict[str, Any]:
        """The record as it acts: its own words, then every correction of the mind under them, each signed.

        ``current_text`` keeps the original and adds each correction in publication order under
        ``correction_line``; the author stays the record's. ``revision`` is the last correction's id
        (what ``expected_revision`` names) and ``corrections`` lists them all. A correction never
        replaces a record's words: to say a record anew the mind folds it into a part over it.
        """
        record = dict(record)
        fixes = [json.loads(body) for (body,) in db.execute(
            "SELECT body FROM records WHERE target=? AND kind='correction' ORDER BY sequence", (record["id"],))]
        record["current_text"] = "\n\n".join([str(record.get("text", "")),
                                              *(f"{correction_line(fix)}\n{fix.get('text', '')}" for fix in fixes)])
        record["revision"] = fixes[-1]["id"] if fixes else record["id"]
        record["corrections"] = [{"id": fix["id"], "author": fix.get("author"), "ts": fix.get("ts")} for fix in fixes]
        status = self._status(db, record)
        if status:
            record["status"] = status
        folded = db.execute("SELECT part_id FROM folded WHERE member_id=?", (record["id"],)).fetchone()
        record["folded_into"] = folded[0] if folded else None
        return record

    # --- reads -------------------------------------------------------------------------------------

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._index() as db:
            return self._get(db, record_id)

    def records(self, room_id: Any = None, *, kinds: Optional[Iterable[str]] = None,
                after_seq: int = 0) -> List[Dict[str, Any]]:
        """Records in publication order; the caller bounds a page."""
        where, args = ["sequence > ?"], [int(after_seq)]
        if room_id is not None:
            where.append("room=?")
            args.append(str(room_id))
        kinds = list(kinds or ())
        if kinds:
            where.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            args.extend(kinds)
        with self._index() as db:
            return [{**json.loads(body), "sequence": seq} for seq, body in db.execute(
                "SELECT sequence,body FROM records WHERE " + " AND ".join(where) + " ORDER BY sequence", args)]

    def room_records(self, room_id: Any, *, after_seq: int = 0) -> List[Dict[str, Any]]:
        """A room's pages, parts, notes, legacy sections and gaps, without rejected drafts.

        Each is interpreted (``_interpret``): ``current_text`` with the mind's corrections under
        the original, ``revision``, ``corrections``, ``status`` for pages and parts and ``folded_into``.
        """
        with self._index() as db:
            rows = [{**json.loads(body), "sequence": seq} for seq, body in db.execute(
                "SELECT sequence,body FROM records WHERE room=? AND sequence>? AND kind IN (%s) ORDER BY sequence"
                % ",".join("?" * len(STORY_KINDS)), (str(room_id), int(after_seq), *STORY_KINDS))]
            return [r for r in (self._interpret(db, row) for row in rows) if r.get("status") != "rejected"]

    def pages_of_room(self, room_id: Any) -> List[Dict[str, Any]]:
        """Effective pages and parts of a room, interpreted, in story order."""
        with self._index() as db:
            rows = [{**json.loads(body), "sequence": seq} for seq, body in db.execute(
                "SELECT sequence,body FROM records WHERE room=? AND kind IN ('page','part') ORDER BY sequence",
                (str(room_id),))]
            cache: Dict[str, Tuple[Any, ...]] = {}
            effective = [r for r in (self._interpret(db, row) for row in rows) if r.get("status") != "rejected"]
            return sorted(effective, key=lambda r: self._order_key(db, r, cache))

    def folded_members(self, part_id: Any) -> List[str]:
        """Every record under a part through nested parts: its own members first, then theirs."""
        with self._index() as db:
            out, todo = [], [str(part_id)]
            while todo:
                part = self._get(db, todo.pop(0)) or {}
                covers = part.get("covers") if part.get("kind") == "part" and isinstance(part.get("covers"), dict) else {}
                members = [str(member) for member in covers.get("member_ids") or ()]
                out += members
                todo += members
            return out

    def sealed_row_refs(self, room_id: Any) -> set:
        """Row refs (``row_sha256`` and ``note:<id>``) the room's acting pages seal; the rest is open."""
        with self._index() as db:
            return {ref for (ref,) in db.execute("SELECT row_ref FROM page_rows WHERE room=?", (str(room_id),))}

    def legacy_pointer_rows(self) -> List[Dict[str, Any]]:
        """One compact row per legacy section or gap: room, label, period and how to open it."""
        with self._index() as db:
            rows = [self._interpret(db, {**json.loads(body), "sequence": seq}) for seq, body in db.execute(
                "SELECT sequence,body FROM records WHERE kind IN ('legacy','gap') ORDER BY sequence")]
        pointers = []
        for row in rows:
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
        with self._index() as db:
            if room_id is None:
                return [json.loads(body) for (body,) in db.execute("SELECT body FROM active_marks ORDER BY sequence")]
            query = "SELECT body FROM active_marks WHERE room=?"
            if include_global:
                query += " OR scope='global'"
            return [json.loads(body) for (body,) in db.execute(query + " ORDER BY sequence", (str(room_id),))]

    def observation_snapshot(self, boundary: Optional[Dict[str, Any]] = None):
        """New immutable records plus their exact accepted position, under one read lock.

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
            rows = db.execute("SELECT sequence,body FROM records WHERE sequence>? ORDER BY sequence", (sequence,))
            return [{**json.loads(body), "sequence": seq} for seq, body in rows], current

    def scan_state(self) -> Dict[str, Any]:
        with self._index() as db:
            return self._state(db, "scan", {})

    def activation(self) -> Optional[Dict[str, Any]]:
        with self._index() as db:
            return self._state(db, "activation")

    def ensure_activated(self, *, wait: bool = False) -> Dict[str, Any]:
        """The activation receipt, importing the legacy dialogue memory once (``chronicle_import``)."""
        from ouroboros.chronicle_import import ensure_activated

        return ensure_activated(self, wait=wait)
