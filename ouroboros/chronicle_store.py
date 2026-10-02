"""Append-only interpretations of memory, with a disposable indexed projection.

The JSONL transaction is the sole authority for derived records and their scan
frontiers. Raw sources keep their existing custody. SQLite stores an indexed
copy only: deletion rebuilds it, interrupted publication replays just the unseen
suffix. A process never rewrites a memory record or guesses project membership.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any

from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.utils import append_jsonl, assert_test_data_path, utc_now_iso


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
    CREATE TABLE IF NOT EXISTS memory_nodes (
        id TEXT PRIMARY KEY, room TEXT NOT NULL, sequence INTEGER NOT NULL,
        current_id TEXT NOT NULL, kind TEXT NOT NULL, rendered_chars INTEGER NOT NULL,
        covers TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS node_rooms ON memory_nodes(room, sequence);
    CREATE TABLE IF NOT EXISTS node_tasks (task_id TEXT NOT NULL, node_id TEXT NOT NULL,
        PRIMARY KEY(task_id, node_id));
    CREATE INDEX IF NOT EXISTS tasks_by_node ON node_tasks(node_id);
    CREATE TABLE IF NOT EXISTS room_cover (room TEXT NOT NULL, id TEXT NOT NULL,
        sequence INTEGER NOT NULL, PRIMARY KEY(room, id));

"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_row_id(row: dict) -> str:
    """Identity of an original raw row, before any room/view metadata is added."""
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_time_span(timestamps, *, incomplete=False) -> dict:
    """Known source bounds, never publication time or proof of continuous coverage."""
    from ouroboros.deadline_utils import parse_deadline_ts

    parsed = [parse_deadline_ts(value) for value in timestamps]
    known = [value for value in parsed if value is not None]
    return {"start": min(known).isoformat() if known else None,
            "end": max(known).isoformat() if known else None,
            "incomplete": bool(incomplete or not known or len(known) != len(parsed))}


class ChronicleStore:
    """One installation's derived episodes, revisions and significance marks."""

    def __init__(self, data_root: Any):
        self.data_root = pathlib.Path(data_root)
        self.directory = self.data_root / "memory" / "chronicle"
        self.log_path = self.directory / "records.jsonl"
        self.index_path = self.data_root / "memory" / "chronicle" / "index.sqlite3"

    @contextmanager
    def _index(self):
        assert_test_data_path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = self.directory / ".publication.lock"
        fd = acquire_exclusive_file_lock(lock, timeout_sec=10, stale_sec=90, owner_aware_stale=True)
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
            if db.execute("PRAGMA user_version").fetchone()[0] != 3:
                # Only disposable projections change schema. Replay authority once.
                db.executescript("DELETE FROM records; DELETE FROM state; DELETE FROM active_marks; "
                                 "DELETE FROM memory_nodes; DELETE FROM room_cover; DELETE FROM node_tasks; "
                                 "DELETE FROM sqlite_sequence WHERE name='records'; PRAGMA user_version=3;")
            self._catch_up(db)
            yield db
        finally:
            if db is not None:
                db.close()
            release_exclusive_file_lock(lock, fd)

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
                except (ValueError, KeyError, TypeError, AttributeError):
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
                           "metadata": {"view_root": True, "source_gap": "unreadable_derived_transaction"}}
                    self._project(db, {"records": [gap]})
                finally:
                    db.execute("RELEASE journal_transaction")
                offset = stream.tell()
                self._set_state(db, "offset", offset)
            db.commit()

    def _project(self, db, tx):
        affected = set()
        for record in tx["records"]:
            encoded = _json(record)
            old = db.execute("SELECT body FROM records WHERE id=?", (record["id"],)).fetchone()
            if old:
                if old[0] != encoded:
                    raise ValueError("chronicle record identity collision")
                continue
            db.execute("INSERT INTO records(id,kind,room,target,body) VALUES (?,?,?,?,?)",
                       (record["id"], record["kind"], record.get("room_id", ""),
                        record.get("target_id", ""), encoded))
            seq = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            self._set_state(db, "room_sequence:" + record.get("room_id", ""), seq)
            if record["kind"] in {"episode", "digest", "legacy", "gap"}:
                affected.add(record["id"])
            elif record["kind"] == "revision":
                affected.add(record["target_id"])
            elif record["kind"] == "revision_decision":
                target = db.execute("SELECT target FROM records WHERE id=?", (record["target_id"],)).fetchone()
                if target:
                    affected.add(target[0])
            if record["kind"] == "activation":
                self._set_state(db, "activation", record)
            elif record["kind"] == "mark":
                db.execute("INSERT INTO active_marks VALUES (?,?,?,?,?)",
                           (record["id"], record.get("room_id", ""), record.get("scope", "room"), seq, encoded))
            elif record["kind"] == "mark_view":
                active = db.execute("SELECT body FROM active_marks WHERE id=?", (record["target_id"],)).fetchone()
                if active:
                    mark = {**json.loads(active[0]), "visibility": record["visibility"], "view_decision": record}
                    db.execute("UPDATE active_marks SET body=? WHERE id=?", (_json(mark), record["target_id"]))
            elif record["kind"] == "mark_release":
                db.execute("DELETE FROM active_marks WHERE id=?", (record["target_id"],))
        for record_id in affected:
            self._update_node(db, record_id)
        for room, frontier in tx.get("frontiers", {}).items():
            self._set_state(db, "frontier:" + room, frontier)
        if tx.get("scan_state") is not None:
            self._set_state(db, "scan", tx["scan_state"])

    def publish(self, records: list[dict], *, expected_frontiers: dict | None = None,
                scan_state: dict | None = None, frontiers: dict | None = None) -> list[dict]:
        """Append records and source progress in ONE durable transaction.

        An existing id replays its original timestamp; a different payload under
        the same id is an error. A losing frontier writer advances nothing.
        """
        with self._index() as db:
            if any(r.get("kind") == "activation" for r in records):
                active = self._state(db, "activation")
                if active:
                    return [active]
            for room, expected in (expected_frontiers or {}).items():
                if self._state(db, "frontier:" + str(room), {}) != expected:
                    raise ValueError("chronicle frontier changed")
            prepared, identities = [], {}
            for supplied in records:
                record = json.loads(_json(supplied))
                record.setdefault("id", uuid.uuid4().hex)
                if not record.get("kind"):
                    raise ValueError("chronicle record needs a kind")
                record["room_id"] = str(record.get("room_id", ""))
                old = db.execute("SELECT body FROM records WHERE id=?", (record["id"],)).fetchone()
                old_record = json.loads(old[0]) if old else identities.get(record["id"])
                record.setdefault("ts", old_record["ts"] if old_record else utc_now_iso())
                if old_record and old_record != record:
                    raise ValueError("chronicle record identity collision")
                if record["id"] in identities and identities[record["id"]] != record:
                    raise ValueError("chronicle record identity collision inside publication")
                if record["id"] not in identities:
                    prepared.append(record)
                    identities[record["id"]] = record
            new_records = [r for r in prepared if not db.execute(
                "SELECT 1 FROM records WHERE id=?", (r["id"],)).fetchone()]
            changed = any(self._state(db, "frontier:" + str(k), {}) != v for k, v in (frontiers or {}).items())
            changed |= scan_state is not None and self._state(db, "scan", {}) != scan_state
            if not new_records and not changed:
                return prepared
            tx = {"kind": "transaction", "schema": 1, "records": new_records,
                  "frontiers": {str(k): v for k, v in (frontiers or {}).items()}, "scan_state": scan_state}
            if not append_jsonl(self.log_path, tx, ensure_record_boundary=True, require_lock=True):
                raise OSError("chronicle transaction append failed")
            with self.log_path.open("ab") as stream:
                os.fsync(stream.fileno())
            self._project(db, tx)
            self._set_state(db, "offset", self.log_path.stat().st_size)
            db.commit()
            return prepared

    def append_episode(self, room_id, text, source_refs, author, *, record_id=None,
                       metadata=None, expected_frontier=None, frontier=None, kind="episode"):
        record = {"kind": kind, "room_id": str(room_id), "text": text,
                  "source_refs": source_refs, "author": author, "metadata": metadata or {}}
        if record_id:
            record["id"] = record_id
        return self.publish([record], expected_frontiers=(
            {str(room_id): expected_frontier} if expected_frontier is not None else None),
            frontiers={str(room_id): frontier} if frontier is not None else None)[0]

    def get(self, record_id):
        with self._index() as db:
            row = db.execute("SELECT sequence,body FROM records WHERE id=?", (record_id,)).fetchone()
            return {**json.loads(row[1]), "sequence": row[0]} if row else None

    def records(self, room_id=None, *, kinds=None, after_seq=0, limit=None):
        where, args = ["sequence > ?"], [after_seq]
        if room_id is not None:
            where.append("room=?")
            args.append(str(room_id))
        if kinds:
            where.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            args.extend(kinds)
        query = "SELECT sequence,body FROM records WHERE " + " AND ".join(where) + " ORDER BY sequence"
        if limit is not None:
            query += " LIMIT ?"
            args.append(max(0, int(limit)))
        with self._index() as db:
            return [{**json.loads(body), "sequence": seq} for seq, body in db.execute(query, args)]

    def observation_snapshot(self, boundary=None):
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

    def pending_episodes(self, *, limit=None):
        """Source-bound originals not yet given their automatic helper correction."""
        query = ("SELECT e.sequence,e.body FROM records e WHERE e.kind='episode' "
                 "AND json_array_length(json_extract(e.body,'$.source_refs')) > 0 "
                 "AND NOT EXISTS (SELECT 1 FROM records r WHERE r.target=e.id AND r.kind='revision' "
                 "AND json_extract(r.body,'$.metadata.auto_correction')=1) ORDER BY e.sequence")
        args = []
        if limit is not None:
            query += " LIMIT ?"
            args.append(max(0, int(limit)))
        with self._index() as db:
            return [{**json.loads(body), "sequence": seq} for seq, body in db.execute(query, args)]

    def room_state(self, room_id):
        with self._index() as db:
            return {"frontier": self._state(db, "frontier:" + str(room_id), {}),
                    "sequence": self._state(db, "room_sequence:" + str(room_id), 0)}

    def scan_state(self):
        with self._index() as db:
            return self._state(db, "scan", {})

    def activation(self):
        with self._index() as db:
            return self._state(db, "activation")

    def revise(self, record_id, text, author, *, status="published", record_id_override=None, metadata=None):
        original = self.get(record_id)
        if not original or original["kind"] not in {"episode", "digest", "legacy", "gap"}:
            raise ValueError("revision needs an existing episode")
        record = {"kind": "revision", "room_id": original["room_id"], "target_id": record_id,
                  "text": text, "author": author, "status": status, "metadata": metadata or {}}
        if record_id_override:
            record["id"] = record_id_override
        return self.publish([record])[0]

    def decide_revision(self, revision_id, accepted, author, reason):
        revision = self.get(revision_id)
        if not revision or revision["kind"] != "revision" or not str(reason).strip():
            raise ValueError("revision decision needs a revision and reason")
        return self.publish([{"kind": "revision_decision", "room_id": revision["room_id"],
                              "target_id": revision_id, "accepted": bool(accepted),
                              "author": author, "reason": reason}])[0]

    def _interpret_record(self, db, row):
        row["current_text"], row["current_author"] = row.get("text", ""), row.get("author", {})
        revisions = db.execute("SELECT body FROM records WHERE target=? AND kind='revision' ORDER BY sequence",
                               (row["id"],)).fetchall()
        row["revisions"] = []
        for (body,) in revisions:
            revision = json.loads(body)
            decision = db.execute("SELECT body FROM records WHERE target=? AND kind='revision_decision' "
                                  "ORDER BY sequence DESC LIMIT 1", (revision["id"],)).fetchone()
            decided = json.loads(decision[0]) if decision else None
            row["revisions"].append({**revision, "decision": decided})
            if revision.get("status") == "published" and (not decided or decided["accepted"]):
                row.update(correction=revision, decision=decided, current_text=revision["text"],
                           current_author=revision["author"])
        return row

    def _record_at(self, db, record_id):
        row = db.execute("SELECT sequence,body FROM records WHERE id=?", (record_id,)).fetchone()
        return self._interpret_record(db, {**json.loads(row[1]), "sequence": row[0]}) if row else None

    def _update_node(self, db, record_id):
        row = self._record_at(db, record_id)
        if not row or row["kind"] not in {"episode", "digest", "legacy", "gap"}:
            return
        metadata = row.get("metadata") or {}
        if not metadata.get("view_root", True):
            return
        from ouroboros.chronicle_view import _current_id, _record_text
        db.execute("INSERT OR REPLACE INTO memory_nodes VALUES (?,?,?,?,?,?,?)",
                   (row["id"], row["room_id"], row["sequence"], _current_id(row), row["kind"],
                    len(_record_text(row)), _json(metadata.get("covers_record_ids") or [])))
        task_ids = metadata.get("task_ids", [(row.get("author") or {}).get("task_id")]) or []
        db.execute("DELETE FROM node_tasks WHERE node_id=?", (row["id"],))
        db.executemany("INSERT OR IGNORE INTO node_tasks VALUES (?,?)",
                       [(str(task_id), row["id"]) for task_id in task_ids if task_id])
        self._set_state(db, "cover_dirty:" + row["room_id"], True)

    def records_for_tasks(self, task_ids):
        """Resolve source task membership, independent of current room bindings."""
        selected = set(str(task_id) for task_id in task_ids if task_id)
        if not selected:
            return []
        with self._index() as db:
            ids = set()
            for task_id in selected:
                ids.update(row[0] for row in db.execute("SELECT node_id FROM node_tasks WHERE task_id=?", (task_id,)))
            rows = [self._record_at(db, record_id) for record_id in ids]
            return sorted(rows, key=lambda row: row["sequence"])

    def room_ids(self):
        """Enumerate stored rooms from indexed descriptors, without memory bodies."""
        with self._index() as db:
            return [row[0] for row in db.execute("SELECT DISTINCT room FROM memory_nodes ORDER BY room")]

    def room_cover(self, room_id):
        """Coarsest valid published cover; old covered bodies are not loaded.

        Digest dependencies name exact current revisions. A corrected child
        invalidates its old digest and any ancestors, without rereading their
        payloads. The coarsest cover is cached until this room changes.
        """
        room = str(room_id)
        with self._index() as db:
            if self._state(db, "cover_dirty:" + room, True):
                descriptors = [dict(zip(("id", "sequence", "current", "kind", "chars", "covers"), row))
                    for row in db.execute("SELECT id,sequence,current_id,kind,rendered_chars,covers FROM memory_nodes "
                                          "WHERE room=? ORDER BY sequence", (room,))]
                active = {r["current"]: r for r in descriptors if r["kind"] != "digest"}
                remaining = [r for r in descriptors if r["kind"] == "digest"]
                for row in remaining:
                    row["covers"] = json.loads(row["covers"])
                while remaining:
                    choices = [(sum(active[c]["chars"] for c in r["covers"]) - r["chars"], r)
                               for r in remaining if r["covers"] and set(r["covers"]).issubset(active)]
                    choices = [(saved, r) for saved, r in choices if saved > 0]
                    if not choices:
                        break
                    _, chosen = max(choices, key=lambda item: (item[0], item[1]["sequence"]))
                    for child in chosen["covers"]:
                        active.pop(child)
                    active[chosen["current"]] = chosen
                    remaining.remove(chosen)
                db.execute("DELETE FROM room_cover WHERE room=?", (room,))
                db.executemany("INSERT INTO room_cover VALUES (?,?,?)",
                               [(room, r["id"], r["sequence"]) for r in active.values()])
                self._set_state(db, "cover_dirty:" + room, False)
                db.commit()
            ids = db.execute("SELECT id FROM room_cover WHERE room=? ORDER BY sequence", (room,)).fetchall()
            return [{**self._record_at(db, record_id), "_cover_root": True} for (record_id,) in ids]

    def room_records(self, room_id, *, after_seq=0, limit=None, include_revisions=False):
        """Explicit detailed read; ordinary background rooms use room_cover."""
        rows = self.records(room_id, after_seq=after_seq, limit=limit,
                            kinds=None if include_revisions else ["episode", "digest", "legacy", "gap"])
        if not include_revisions:
            rows = [r for r in rows if r.get("metadata", {}).get("view_root", True)]
        with self._index() as db:
            return [self._interpret_record(db, row) for row in rows]

    def mark(self, target_ref, text, author, *, room_id="", scope="room", quote=None):
        return self.publish([{"kind": "mark", "room_id": str(room_id), "target_ref": target_ref,
                              "text": text, "author": author, "scope": scope, "quote": quote, "visibility": "full"}])[0]

    def set_mark_view(self, mark_id, visibility, author, reason):
        mark = self.get(mark_id)
        if (not mark or mark["kind"] != "mark" or visibility not in {"full", "meaning"}
                or not str(reason).strip() or not str(mark.get("text", "")).strip()):
            raise ValueError("mark view needs an existing meaning, full/meaning visibility and a reason")
        return self.publish([{"kind": "mark_view", "room_id": mark["room_id"], "target_id": mark_id,
                              "visibility": visibility, "author": author, "reason": reason}])[0]

    def release_mark(self, mark_id, author, reason):
        mark = self.get(mark_id)
        if not mark or mark["kind"] != "mark" or not str(reason).strip():
            raise ValueError("mark release needs a mark and explicit reason")
        return self.publish([{"kind": "mark_release", "room_id": mark["room_id"],
                              "target_id": mark_id, "author": author, "reason": reason}])[0]

    def active_marks(self, room_id, include_global=True):
        with self._index() as db:
            query = "SELECT body FROM active_marks WHERE room=?"
            if include_global:
                query += " OR scope='global'"
            return [json.loads(body) for (body,) in db.execute(query + " ORDER BY sequence", (str(room_id),))]

    def import_legacy(self, *, already_locked=False):
        from ouroboros.chronicle_import import import_legacy
        return import_legacy(self, already_locked=already_locked)
