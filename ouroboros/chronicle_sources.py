"""Disposable row locators over the existing rotation-aware chat source.

No raw text is stored here. Existing JsonlChainSnapshot owns physical discovery
and reads; this cache remembers where rows and their immutable identities are,
so a quiet room is not lost behind unrelated traffic. Append indexing reads only
the unseen suffix, while project membership is resolved afresh at query time.
"""
from __future__ import annotations

import hashlib
import json

from ouroboros.chronicle_store import ChronicleStore, source_row_id
from ouroboros.context_budget import canonical_context_json as _encoded
from ouroboros.jsonl_tail import JsonlChainSnapshot
from ouroboros.utils import JsonlChainUnreadable

_READ_BYTES = 64 * 1024  # I/O chunk, never a memory horizon or selection quota.
_SCHEMA = """
    CREATE TABLE IF NOT EXISTS raw_generations (
        generation TEXT PRIMARY KEY, indexed INTEGER NOT NULL,
        size INTEGER NOT NULL, mtime INTEGER NOT NULL, lines INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS raw_rows (
        generation TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
        line INTEGER NOT NULL, chat INTEGER NOT NULL, type TEXT NOT NULL,
        task TEXT NOT NULL, parent TEXT NOT NULL, root TEXT NOT NULL,
        raw_sha TEXT NOT NULL, source_id TEXT NOT NULL, rendered_chars INTEGER NOT NULL,
        PRIMARY KEY(generation,start));
    CREATE INDEX IF NOT EXISTS raw_chat ON raw_rows(chat);
    CREATE INDEX IF NOT EXISTS raw_task ON raw_rows(task);
    CREATE INDEX IF NOT EXISTS raw_parent ON raw_rows(parent);
    CREATE INDEX IF NOT EXISTS raw_root ON raw_rows(root);
    CREATE TABLE IF NOT EXISTS raw_origins (
        identity TEXT NOT NULL, generation TEXT NOT NULL, start INTEGER NOT NULL,
        PRIMARY KEY(identity,generation,start));
    CREATE INDEX IF NOT EXISTS origins_generation ON raw_origins(generation);
    CREATE TABLE IF NOT EXISTS raw_gaps (
        generation TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
        reason TEXT NOT NULL, PRIMARY KEY(generation,start));
"""
_COLUMNS = "generation,start,end,line,chat,type,task,parent,root,raw_sha,source_id,rendered_chars"
_KEYS = _COLUMNS.split(",")


def format_source_row(row):
    """Full words and semantic metadata, with transport bookkeeping by reference."""
    from ouroboros.dialogue_provenance import dialogue_author, dialogue_text

    author = "Ouroboros" if row.get("direction") in {"out", "outgoing", "system"} else dialogue_author(row)
    omitted = {"text", "ts", "session_id", "sender_session_id", "client_message_id", "client_surface"}
    if row.get("type") == "acceptance_late_settlement" and isinstance(row.get("late_evidence"), dict):
        omitted.add("late_evidence")  # dialogue_text already renders this exact evidence, with its role.
    metadata = {key: value for key, value in row.items()
                if key not in omitted and value not in (None, "", {}, [])}
    return (f"[{row.get('ts', '')}; {author}; source_row_id={source_row_id(row)}]\n"
            + _encoded(metadata) + "\n" + dialogue_text(row))


def retain_room_source(memory, task_id, rows, locators, coverage):
    """Reuse existing immutable blobs per physical generation, not full-room task copies."""
    from types import SimpleNamespace
    from ouroboros.consolidator import retain_memory_source
    from ouroboros.observability import write_blob

    payload = {"kind": "chronicle_room_source", "room_id": coverage.get("room_id"), "coverage": coverage}
    if coverage.get("full_room_in_view") and len(rows) == len(locators):
        chunks = {}
        for row, locator in zip(rows, locators):
            chunks.setdefault(locator["generation"], []).append(row)
        payload.update(source_chunks=[write_blob(memory.drive_root, chunk) for chunk in chunks.values()],
                       row_count=len(rows))
    else:
        payload["row_locators"] = locators
    ref = retain_memory_source(SimpleNamespace(drive_root=memory.drive_root, task_id=task_id),
                               "focused_room", _encoded(payload).encode("utf-8"), "json")
    return ref


def _source_bounds(start, end, total):
    lower = 0 if start is None else start
    if type(lower) is not int or (end is not None and type(end) is not int):
        raise ValueError("source range must contain integer row positions")
    upper = total if end is None else min(end, total)
    if not 0 <= lower <= upper:
        raise ValueError("source range must satisfy 0 <= start <= end")
    return lower, upper


def read_locator_rows(data_root, locators):
    """Resolve captured identities through the canonical chain, never locator paths.

    Rotation may change a path, but not the captured generation/bytes/hash. Read
    only requested rows; missing or rewritten bytes remain explicit gaps.
    """
    from pathlib import Path

    rows, gaps = [], []
    try:
        snapshot = JsonlChainSnapshot(Path(data_root) / "logs/chat.jsonl")
        generations = {_generation(stat): snapshot.segment(n)
                       for n, (_path, stat, _live) in enumerate(snapshot.entries)}
    except OSError as exc:
        return [], [{"kind": "chat_chain_unreadable", "detail": str(exc)}]
    for locator in locators:
        try:
            base, boundary = generations[locator["generation"]]
            start, end = locator["start_byte"], locator["end_byte"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= boundary - base:
                raise ValueError("captured row is outside its source generation")
            raw = snapshot._read(base + start, base + end)
            if hashlib.sha256(raw).hexdigest() != locator["sha256"]:
                raise ValueError("captured row bytes changed")
            row = json.loads(raw)
            if not isinstance(row, dict) or source_row_id(row) != locator["source_row_id"]:
                raise ValueError("captured row identity changed")
            rows.append(row)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            gaps.append({"kind": "captured_row_unavailable", "source_row_id": locator.get("source_row_id"),
                         "generation": locator.get("generation"), "detail": str(exc)})
    return rows, gaps


def read_chronicle_source(data_root, ref, task_id="", *, start=None, end=None, require_complete=False):
    """One source reader for exact text, row lists, immutable chunks and locators."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.observability import read_blob_ref

    raw = read_actor_source_bytes(data_root, str(ref.get("task_id") or task_id), ref).decode("utf-8")
    try:
        payload = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        payload = {"rows": payload}
    if not isinstance(payload, dict):
        return raw
    missing, locator_page = list(payload.get("missing") or []), False
    if payload.get("kind") == "chronicle_room_source" and "rows" not in payload:
        if "source_chunks" in payload:
            rows = [row for chunk in payload.pop("source_chunks") for row in read_blob_ref(data_root, chunk)]
            if len(rows) != payload.get("row_count") or any(not isinstance(row, dict) for row in rows):
                raise ValueError("retained room source chunks failed row verification")
            payload["rows"] = rows
        elif "row_locators" in payload:
            locators = payload["row_locators"]
            lower, upper = _source_bounds(start, end, len(locators))
            total, locator_page = len(locators), True
            payload["row_locators"] = locators[lower:upper]
            payload["rows"], missing = read_locator_rows(data_root, payload["row_locators"])
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return _encoded(payload)
    ids = [source_row_id(row) for row in rows]
    if payload.get("source_row_ids", ids) != ids:
        raise ValueError("retained source row identities do not match its exact rows")
    if not locator_page:
        total = len(rows)
        lower, upper = _source_bounds(start, end, total)
        payload["rows"] = rows[lower:upper]
    payload["source_row_ids"] = [source_row_id(row) for row in payload["rows"]]
    if "row_count" in payload:
        payload["row_count"] = len(payload["rows"])
    if locator_page or start is not None or end is not None:
        payload["range"] = {"start": lower, "end": upper, "total": total, "unit": "matching_rows"}
        payload["page_complete"] = (payload.get("page_complete", True) and lower == 0 and upper == total
                                    and not missing and payload.get("coverage", {}).get("complete", True))
        payload["range_complete"] = not missing
    if missing:
        coverage = payload.setdefault("coverage", {})
        coverage.update(complete=False, snapshot_stable=False, gaps=[*coverage.get("gaps", []), *missing])
        payload["missing"] = missing
    if require_complete and (missing or payload.get("range_complete") is False):
        raise ValueError("the selected retained source has unavailable rows")
    return _encoded(payload)


def _generation(stat):
    return f"{stat.st_dev}:{stat.st_ino}:{getattr(stat, 'st_birthtime', '')}"


def _index_generation(db, snapshot, number, generation, stat):
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
    from ouroboros.project_dialogue import _entry_source_identities, _row_chat_id

    prior = db.execute("SELECT indexed,size,mtime,lines FROM raw_generations WHERE generation=?", (generation,)).fetchone()
    offset, line = (prior[0], prior[3]) if prior else (0, 0)
    if prior and (stat.st_size < prior[1] or (stat.st_size == prior[1] and stat.st_mtime_ns != prior[2])):
        # Same-sized replacement/truncation invalidates only this disposable generation.
        for table in ("raw_rows", "raw_origins", "raw_gaps"):
            db.execute(f"DELETE FROM {table} WHERE generation=?", (generation,))
        offset, line = 0, 0
    base, end = snapshot.segment(number)
    position, remainder = offset, b""
    cursor = base + offset
    def remember(raw):
        nonlocal position, line
        next_offset, line = position + len(raw), line + 1
        try:
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("not a chat record")
            if not is_a2a_chat_id(row.get("chat_id")):
                source_id = source_row_id(row)
                rendered = len(format_source_row(row)) + 2
                db.execute("INSERT OR REPLACE INTO raw_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                    generation, position, next_offset, line, _row_chat_id(row), str(row.get("type") or ""),
                    str(row.get("task_id") or ""), str(row.get("parent_task_id") or ""),
                    str(row.get("root_task_id") or ""), hashlib.sha256(raw).hexdigest(), source_id, rendered))
                db.executemany("INSERT OR IGNORE INTO raw_origins VALUES (?,?,?)", [
                    (_encoded(identity), generation, position) for identity in _entry_source_identities(row)])
        except (ValueError, UnicodeError) as exc:
            if raw.strip():
                db.execute("INSERT OR REPLACE INTO raw_gaps VALUES (?,?,?,?)",
                           (generation, position, next_offset, f"invalid_chat_row:{type(exc).__name__}"))
        position = next_offset
    while cursor < end:
        upper = min(end, cursor + _READ_BYTES)
        data = remainder + snapshot._read(cursor, upper)
        cursor = upper
        chunks = data.split(b"\n")
        remainder = chunks.pop()
        for fragment in chunks:
            remember(fragment + b"\n")
    if remainder and not snapshot.entries[number][2]:
        # A frozen archive can have a complete final object without a newline.
        remember(remainder)
        remainder = b""
    # An incomplete final line belongs to the writer; retry it from its start.
    db.execute("INSERT OR REPLACE INTO raw_generations VALUES (?,?,?,?,?)",
               (generation, position, stat.st_size, stat.st_mtime_ns, line))
    return {"generation": generation, "indexed_bytes": position, "physical_bytes": stat.st_size,
            "incomplete_tail": bool(remainder), "new_index_bytes": max(0, position - offset)}


def _candidates(db, chat, project_ids, bindings, refs):
    from ouroboros.project_dialogue import _source_ref_identity

    selected = {}
    def add(query, args):
        for row in db.execute(query, args):
            descriptor = dict(zip(_KEYS, row))
            selected[(descriptor["generation"], descriptor["start"])] = descriptor
    if chat == 1:
        # Main is the canonical non-project lens, including unregistered rooms.
        add("SELECT " + _COLUMNS + " FROM raw_rows", ())
    else:
        add("SELECT " + _COLUMNS + " FROM raw_rows WHERE chat=?", (chat,))
    if chat in project_ids:
        for task_id, destination in bindings.items():
            if destination == chat:
                for column in ("task", "parent", "root"):
                    add("SELECT " + _COLUMNS + f" FROM raw_rows WHERE {column}=?", (task_id,))
        for ref in refs:
            key = _source_ref_identity(ref)
            if key is not None:
                for generation, start in db.execute("SELECT generation,start FROM raw_origins WHERE identity=?", (_encoded(key),)):
                    add("SELECT " + _COLUMNS + " FROM raw_rows WHERE generation=? AND start=?", (generation, start))
                    selected[(generation, start)]["origin_match"] = True
    return selected.values()


def capture_room(memory, room_id, byte_budget=None, *, rendered_chars_budget=None):
    """Return an entire matched room when its actual selected representation fits.

    Budgets are supplied physical/rendering facts, never global-log quotas. When
    the complete matched room exceeds one, return exact locators and named
    omission instead of an arbitrary prefix. The open arc has its own reader.
    """
    from ouroboros.project_dialogue import room_membership, source_refs_for_project
    from ouroboros.projects_registry import all_task_bindings, reserved_project_chat_ids

    chat = int(room_id)
    for value in (byte_budget, rendered_chars_budget):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("room budgets must be nonnegative integers or unknown (None)")
    store = ChronicleStore(memory.drive_root)
    try:
        snapshot = JsonlChainSnapshot(memory.logs_path("chat.jsonl"))
    except OSError as exc:
        return [], {"snapshot_stable": False, "complete": False,
                    "gaps": [{"kind": "chat_chain_unreadable", "detail": str(exc)}], "row_locators": []}
    project_ids = reserved_project_chat_ids(memory.drive_root)
    bindings = all_task_bindings(memory.drive_root)
    refs = source_refs_for_project(memory.drive_root, chat)
    matches = room_membership(chat, project_ids, refs, bindings)
    generations, summaries, gaps = {}, [], []
    with store._index() as db:
        db.executescript(_SCHEMA)
        for n, (path, stat, _was_live) in enumerate(snapshot.entries):
            key = _generation(stat)
            base, _end = snapshot.segment(n)
            generations[key] = {"base": base, "path": str(path), "order": n}
            try:
                summary = _index_generation(db, snapshot, n, key, stat)
            except OSError as exc:
                gaps.append({"kind": "chat_generation_unreadable", "generation": key, "detail": str(exc)})
                continue
            summaries.append(summary)
            if summary["incomplete_tail"]:
                gaps.append({"kind": "incomplete_chat_line", "generation": key,
                             "start_byte": summary["indexed_bytes"], "end_byte": stat.st_size})
        db.commit()
        present = set(generations)
        for key, start, end, reason in db.execute("SELECT generation,start,end,reason FROM raw_gaps"):
            if key in present:
                gaps.append({"kind": reason, "generation": key, "start_byte": start, "end_byte": end})
        prior_generations = {row[0] for row in db.execute("SELECT generation FROM raw_generations")}
        if prior_generations - present:
            gaps.append({"kind": "previously_indexed_generation_missing", "generations": sorted(prior_generations - present)})
        selected = []
        for row in _candidates(db, chat, project_ids, bindings, refs):
            metadata = {"chat_id": row["chat"], "type": row["type"], "task_id": row["task"],
                        "parent_task_id": row["parent"], "root_task_id": row["root"]}
            if row["generation"] in present and (row.get("origin_match") or matches(row["chat"], metadata)):
                selected.append(row)
    selected.sort(key=lambda row: (generations[row["generation"]]["order"], row["start"]))
    physical_bytes = sum(row["end"] - row["start"] for row in selected)
    rendered_chars = sum(row["rendered_chars"] for row in selected)
    locators = [{"kind": "chat_row", "generation": row["generation"],
                 "path": generations[row["generation"]]["path"], "start_byte": row["start"],
                 "end_byte": row["end"], "line": row["line"], "sha256": row["raw_sha"],
                 "source_row_id": row["source_id"]} for row in selected]
    coverage = {"snapshot_stable": True, "complete": not gaps, "gaps": gaps, "room_id": str(chat),
                "matched_rows": len(selected), "matched_physical_bytes": physical_bytes,
                "matched_rendered_chars": rendered_chars, "generations": summaries, "row_locators": locators,
                "excluded_a2a": True, "reader": "memory_read(raw_room=true, room_id=...)"}
    oversized = ((byte_budget is not None and physical_bytes > byte_budget)
                 or (rendered_chars_budget is not None and rendered_chars > rendered_chars_budget))
    if oversized:
        coverage.update(full_room_in_view=False, omission="matched_room_exceeds_supplied_budget")
        return [], coverage
    rows = []
    for descriptor in selected:
        base = generations[descriptor["generation"]]["base"]
        try:
            raw = snapshot._read(base + descriptor["start"], base + descriptor["end"])
            if hashlib.sha256(raw).hexdigest() != descriptor["raw_sha"]:
                raise JsonlChainUnreadable("indexed row bytes changed")
            row = json.loads(raw)
            if not matches(descriptor["chat"], row):
                raise JsonlChainUnreadable("row no longer matches its captured room lens")
            rows.append(row)
        except (OSError, ValueError) as exc:
            gaps.append({"kind": "indexed_row_unavailable", "source_row_id": descriptor["source_id"], "detail": str(exc)})
    coverage.update(complete=not gaps, snapshot_stable=not gaps, full_room_in_view=not gaps)
    return rows, coverage



def capture_covered_focus(memory, store, locators, covered):
    """Read represented focus sources plus typed closure facts, not old bodies.

    The existing row index supplies type only for locating candidate facts;
    canonical lifecycle logic interprets the actual returned rows. Ordering and
    exact source identity are inherited from the captured focused locators.
    """
    selected = []
    with store._index() as db:
        for locator in locators:
            row_type = db.execute("SELECT type FROM raw_rows WHERE generation=? AND start=?",
                (locator["generation"], locator["start_byte"])).fetchone()
            if locator["source_row_id"] in covered or (row_type and row_type[0]):
                selected.append(locator)
    return read_locator_rows(memory.drive_root, selected)


def capture_pending_rows(memory, store, represented):
    """Read only unrepresented rows after the generation-aware legacy cursor.

    SQL OFFSET counts valid non-A2A rows, exactly like the consolidation stream;
    physical line numbers also count blank/invalid/A2A lines and cannot be used.
    The existing row index and canonical chain own all physical source reads.
    """
    from ouroboros.consolidator import _chat_log_signature, _resolve_generation_segments

    scan = store.scan_state()
    if scan.get("raw_scan_unavailable"):
        return None, [{"kind": "chronicle_open_range_unavailable", "cause": scan["raw_scan_unavailable"]}]
    source_path = memory.logs_path("chat.jsonl")
    segments, offset, gap = _resolve_generation_segments(scan, source_path)
    if gap:
        return None, [{"kind": "chronicle_open_range_unavailable", "cause": "cursor_generation_missing"}]
    try:
        snapshot = JsonlChainSnapshot(source_path)
        descriptors = {path: (n, stat, _generation(stat)) for n, (path, stat, _live) in enumerate(snapshot.entries)}
        first = (scan.get("chat_log_signature") or {}).get("first_line_sha256")
        if first and _chat_log_signature(segments[0]).get("first_line_sha256") != first:
            raise JsonlChainUnreadable("cursor generation changed during capture")
        locators, gaps, skipped = [], [], offset
        with store._index() as db:
            db.executescript(_SCHEMA)
            for path in segments:
                if path not in descriptors:
                    if path == source_path and not path.exists():
                        continue  # A rotated archive can be the complete current chain.
                    raise JsonlChainUnreadable("cursor segment vanished during capture")
                number, stat, key = descriptors[path]
                summary = _index_generation(db, snapshot, number, key, stat)
                if summary["incomplete_tail"]:
                    gaps.append({"kind": "incomplete_chat_line", "generation": key})
                count = db.execute("SELECT COUNT(*) FROM raw_rows WHERE generation=? AND end<=?",
                                   (key, summary["indexed_bytes"])).fetchone()[0]
                rows = db.execute("SELECT start,end,raw_sha,source_id FROM raw_rows WHERE generation=? AND end<=? "
                                  "ORDER BY start LIMIT -1 OFFSET ?", (key, summary["indexed_bytes"], skipped))
                locators.extend({"generation": key, "start_byte": start, "end_byte": end,
                                 "sha256": digest, "source_row_id": source_id}
                                for start, end, digest, source_id in rows if source_id not in represented)
                skipped = max(0, skipped - count)
                gaps.extend({"kind": reason, "generation": key, "start_byte": start, "end_byte": end}
                            for start, end, reason in db.execute(
                                "SELECT start,end,reason FROM raw_gaps WHERE generation=?", (key,)))
            db.commit()
        if skipped:
            raise JsonlChainUnreadable("cursor is beyond the indexed complete source rows")
        rows, missing = read_locator_rows(memory.drive_root, locators)
        return rows, [*gaps, *missing]
    except (OSError, ValueError) as exc:
        return None, [{"kind": "chronicle_open_range_unavailable", "detail": str(exc)}]
