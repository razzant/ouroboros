"""The dialogue's cutover from the JSONL chains, the archive import job and the rollback export.

``cutover(root)`` is the one durable transition of a data root from files to
``state/dialog.sqlite``: the live ``logs/chat.jsonl`` and ``logs/progress.jsonl`` are rotated
into ``archive/`` one last time under today's names, every archive segment is planned into the
``import_ledger`` with its chain coordinate (``base`` = the sizes of the segments before it, by
name), and the final live segments are imported at once -- small, so the recent client ids
dedup before the first accepted message. It records the cutover coordinate per stream and is
cheap afterwards: a stat of two files per boot. A non-empty live file at a later boot means a
store-unaware body appended (a downgrade ran): it is rotated and imported the same way, so
nothing a body wrote anywhere is lost; a row the export already placed in the file keeps its
coordinate and is ignored on re-import. Child roots are cut over at their first row, never at
boot.

``run_dialogue_import(root)`` imports the frozen archive as a resumable, idempotent job in the
shape of ``store_import.run_legacy_reconcile`` (the maintenance child runs it): newest segment
first, chat before progress, one segment per transaction, the ledger as the cursor. Coverage is
a fact: ``coverage`` answers whether a coordinate range is imported, and a reader returns "no
older rows" only inside the imported range. The archive stays frozen and is never deleted
(BIBLE P1).

Each imported segment is a counting unit: its rows keep the counts the chain readers kept
(``ordinal`` among its objects, ``pos`` among its chat rows outside A2A) relative to the
segment, its ledger row the totals (``rows``, ``stream_rows``) and the generation signature the
old addresses carry (``gen``); the lines it skipped are recorded as ``skipped`` intervals with
their exact bytes, and every ``(gen, line)`` alias of its chat rows is kept. The transaction that
imports the last pending segment below the cutover settles the absolute bases of every unit
(``_settle``): the archive's units in chain order, then the live unit the rows appended after
the cutover count in -- so those rows continue the same counting. A segment that joins after the
cutover (an older body ran) is imported synchronously by that cutover, before any new row, and
its rows count in the live unit at their chain place. At every cutover a room's ``revision``
rises to the project's ``projects.json`` ``visible_revision`` when that is higher, so a read
marker a client holds keeps seeing new arrivals.

``export_for_older_body(root)`` appends every row the files do not hold yet (``seq >= files_seq``)
back to ``logs/<stream>.jsonl`` in the exact line format. Its coordinate map preserves store
identity when revised bodies change physical spans; ``store_switch`` calls it before a root
is handed to such a body.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import pathlib
import time
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

from ouroboros import dialogue_store as ds
from ouroboros import store, store_journal
from ouroboros.store import StoreBusy
from ouroboros.utils import jsonl_append_lock_path, utc_now_iso

log = logging.getLogger(__name__)

JOB = "dialogue_import"
JOB_LOCK_NAME = "dialogue_import_job.lock"  # one runner of the archive import per root


# --- the chain's segments --------------------------------------------------------------------

def segment_plan(root: pathlib.Path, stream: str) -> List[Dict[str, Any]]:
    """Every archive segment of ``stream`` by name (the chain's order), with its size and base."""
    prefix = f"{stream}_"
    try:
        with os.scandir(root / "archive") as entries:
            names = sorted(entry.name for entry in entries
                           if entry.name.startswith(prefix) and entry.name.endswith(".jsonl") and entry.is_file())
    except FileNotFoundError:
        return []
    plan, base = [], 0
    for name in names:
        info = os.stat(root / "archive" / name)
        plan.append({"name": name, "size": info.st_size, "mtime_ns": info.st_mtime_ns, "base": base})
        base += info.st_size
    return plan


def rotate_final(root: pathlib.Path, stream: str) -> Optional[pathlib.Path]:
    """Rename a non-empty live ``logs/<stream>.jsonl`` into ``archive/<stream>_<UTC>.jsonl`` under
    the chain's append lock (today's naming, the ``_<n>`` collision suffix included); ``None``
    when there is nothing to rotate. A held append lock is ``StoreBusy``: the next boot retries."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    live = root / ds.LIVE_REL[stream]
    try:
        if live.stat().st_size == 0:
            return None
    except FileNotFoundError:
        return None
    lock = jsonl_append_lock_path(live)
    fd = acquire_exclusive_file_lock(lock, timeout_sec=ds.WAIT_SEC, stale_sec=10.0, owner_aware_stale=True)
    if fd is None:
        raise StoreBusy(f"dialogue cutover: the append lock of {live} is held", reason="contention")
    try:
        if not live.exists() or live.stat().st_size == 0:
            return None
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        target = root / "archive" / f"{stream}_{stamp}.jsonl"
        suffix = 0
        while target.exists():
            suffix += 1
            target = root / "archive" / f"{stream}_{stamp}_{suffix}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(live, target)
        live.touch()
        store.fsync_directory(target.parent)
        store.fsync_directory(live.parent)
        return target
    finally:
        release_exclusive_file_lock(lock, fd)


def _gen(data: bytes) -> str:
    """The generation signature a legacy address carries (``utils.jsonl_generation_signature``): the
    first non-blank line as that text-mode reader ends it, at ``\\n``/``\\r`` only. A row ``append_jsonl``
    wrote keeps U+2028, U+2029 and U+0085 raw, and ``str.splitlines`` would cut it there."""
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                return hashlib.sha256(line.strip().encode("utf-8", errors="replace")).hexdigest()
    return ""


class Line(NamedTuple):
    """One physical line of a segment: its coordinate, physical line number, exact length (the
    line terminator included), the body without the terminator, the row (an object) or None,
    and its ``kind``: ``row``, ``blank``, or a skipped ``invalid`` / ``non_object`` line."""

    seq: int
    number: int
    length: int
    body: str
    row: Optional[Dict[str, Any]]
    kind: str


def segment_rows(data: bytes, base: int) -> Iterator[Line]:
    """Every line of a segment as a ``Line``; a blank or skipped line holds no row but its bytes
    still count in the coordinates (``seq + length`` is the next line's coordinate)."""
    position = base
    for number, raw in enumerate(data.splitlines(keepends=True), 1):
        line = raw[:-1] if raw.endswith(b"\n") else raw
        if line.endswith(b"\r"):
            line = line[:-1]
        row, kind = None, "blank"
        if line.strip():
            try:
                value = json.loads(line.decode("utf-8", errors="replace").strip())
                row, kind = (value, "row") if isinstance(value, dict) else (None, "non_object")
            except ValueError:
                kind = "invalid"
        yield Line(position, number, len(raw), line.decode("utf-8", errors="replace"), row, kind)
        position += len(raw)


# --- the cutover ------------------------------------------------------------------------------

def cutover(root: Any, *, initialize_journal: bool = False) -> Dict[str, Any]:
    """Files -> store, once per root; afterwards the cheap re-check described above. Returns a report.
    ``initialize_journal`` is the boot's (``server._dialogue_cutover``): after any retier it brings an
    existing file to this runtime's journal (``store_journal``); a child root's lazy cutover keeps it."""
    root = pathlib.Path(root)
    report: Dict[str, Any] = {"created": False}
    if ds.present(root):
        former = ds.retier(root)
        if former:
            report["retiered_from"] = former
        if initialize_journal:
            tier = ds._tier(root)
            with store.name_lock(root, ds.LOCK_NAME, timeout_sec=ds.WAIT_SEC) if tier == store.TIER_NAME \
                    else contextlib.nullcontext():
                report["journal_selected"] = store_journal.configure_journal(ds.path(root), tier, wait_sec=ds.WAIT_SEC)
        if ds.meta(root, "cutover") is not None and not _replan_owed(root):
            report.update(first=False, replanned=False)  # the steady boot: three stats and two meta rows, no listing
            return report
    else:
        ds.create(root)
        report["created"] = True
    with store.process_lock(ds.path(root)), store.name_lock(root, ds.IMPORT_LOCK_NAME, timeout_sec=ds.WAIT_SEC, stale_sec=600.0):
        first = ds.meta(root, "cutover") is None
        for stream in ds.STREAMS:
            rotated = rotate_final(root, stream)
            plan = segment_plan(root, stream)
            report[stream] = _plan(root, stream, plan)
            report[stream]["rotated"] = rotated.name if rotated else None
            # The final live segment now, and every segment that joined the chain at or beyond the
            # cutover coordinate (an older body rotated during a downgrade): their rows count in the
            # live unit, so they are stored before any new row is appended.
            for segment in _live_tail(root, stream, rotated):
                outcome = import_segment(root, stream, segment)
                if rotated is not None and segment == rotated.name:
                    report[stream]["imported"] = outcome
                else:
                    report[stream].setdefault("joined", []).append({"name": segment, **outcome})
        with ds.transaction(root, write=True) as conn:
            report["rooms"] = _inherit_revisions(root, conn)
            marks = {record["stream"]: dict(record) for record in conn.execute("SELECT * FROM streams")}
            facts = {"at": utc_now_iso(), **{stream: marks[stream] for stream in ds.STREAMS}}
            ds._set_meta(conn, "cutover" if first else "reconciled", facts)
            stamp, tails = _archive_seen(root)
            ds._set_meta(conn, "archive_tails", tails)  # what this plan saw (``_replan_owed``)
            ds._set_meta(conn, "archive_stamp", stamp)
            for stream in ds.STREAMS:
                # A live file rotated into the archive (by this run, or by one a stop interrupted after the
                # rename: the live file is now shorter than the batch's offset) holds no export batch any more.
                batch = ds._meta(conn, f"export_batch:{stream}")
                if batch and (report[stream]["added"] or report[stream].get("rotated")
                              or _live_size(root, stream) < int(batch.get("offset") or 0)):
                    conn.execute("DELETE FROM meta WHERE key=?", (f"export_batch:{stream}",))
    report["first"] = first
    return report


def _live_size(root: pathlib.Path, stream: str) -> int:
    try:
        return (root / ds.LIVE_REL[stream]).stat().st_size
    except FileNotFoundError:
        return 0


def _archive_stamp(root: pathlib.Path) -> List[int]:
    """The archive directory's identity and mtime (``[st_ino, st_mtime_ns]``; ``[]`` without the
    directory): a segment added to or removed from it moves the mtime, whatever its stream, as
    ``delegate_custody_current._archive_stamp`` relies on for the custody chain."""
    try:
        info = os.stat(root / "archive")
    except FileNotFoundError:
        return []
    return [info.st_ino, info.st_mtime_ns]


def _archive_seen(root: pathlib.Path) -> Tuple[List[int], Dict[str, List[Any]]]:
    """The stamp, then the tails: a segment landing between the two moves the stamp the next
    boot compares, so it lists again; the other order would let it hide behind a current stamp."""
    stamp = _archive_stamp(root)
    return stamp, _archive_tails(root)


def _archive_tails(root: pathlib.Path) -> Dict[str, List[Any]]:
    """Each stream's archive segment count and newest name: one directory listing, no stat."""
    tails: Dict[str, List[Any]] = {stream: [0, ""] for stream in ds.STREAMS}
    try:
        with os.scandir(root / "archive") as entries:
            for entry in entries:
                for stream in ds.STREAMS:
                    if entry.name.startswith(f"{stream}_") and entry.name.endswith(".jsonl"):
                        tails[stream][0] += 1
                        tails[stream][1] = max(tails[stream][1], entry.name)
    except FileNotFoundError:
        pass
    return tails


def _replan_owed(root: pathlib.Path) -> bool:
    """After the first cutover a boot plans the chains again only when another body wrote them:
    a non-empty live file (a store-unaware body appended) or archive segments the last plan did
    not see (one rotated them). The archive is listed only after its directory's stamp moved since
    the plan or the last check (one stat against one meta row, whatever the archive holds); a move
    that brought no segment records the new stamp and costs the next boot nothing. A frozen segment
    that changed size is found by the import."""
    for stream in ds.STREAMS:
        try:
            if (root / ds.LIVE_REL[stream]).stat().st_size:
                return True
        except FileNotFoundError:
            continue
    with ds.transaction(root, write=False) as conn:
        recorded_stamp, recorded_tails = ds._meta(conn, "archive_stamp"), ds._meta(conn, "archive_tails")
    if recorded_stamp is not None and _archive_stamp(root) == recorded_stamp:
        return False
    stamp, tails = _archive_seen(root)  # the stamp again, right before the listing it vouches for
    if tails != recorded_tails:
        return True
    with ds.transaction(root, write=True) as conn:
        ds._set_meta(conn, "archive_stamp", stamp)
    return False


def _live_tail(root: pathlib.Path, stream: str, rotated: Optional[pathlib.Path]) -> List[str]:
    """The pending segment names this cutover imports itself, in chain order."""
    with ds.transaction(root, write=False) as conn:
        names = [record["name"] for record in conn.execute(
            "SELECT name FROM import_ledger WHERE stream=? AND state='pending' AND (base>=(SELECT cutover_seq FROM streams "
            "WHERE stream=?) OR name=?) ORDER BY base", (stream, stream, rotated.name if rotated else ""))]
    return names


def _inherit_revisions(root: pathlib.Path, conn: Any) -> Dict[int, int]:
    """Each project room's ``revision`` continues from the registry's ``visible_revision`` (never
    below it): the unread counter a client's read marker compares against keeps its meaning."""
    from ouroboros.projects_registry import list_reserved_projects

    try:
        projects = list_reserved_projects(root)
    except Exception:
        log.warning("dialogue cutover: the project registry of %s could not be read; room revisions start at 0", root,
                    exc_info=True)
        return {}
    inherited: Dict[int, int] = {}
    for project in projects:
        try:
            chat_id, revision = int(project.get("chat_id") or 0), int(project.get("visible_revision") or 0)
        except (TypeError, ValueError):
            continue
        if chat_id > 0 and revision > 0:
            conn.execute("INSERT INTO rooms (chat_id, revision) VALUES (?, ?) ON CONFLICT(chat_id) DO UPDATE SET "
                         "revision = max(rooms.revision, excluded.revision)", (chat_id, revision))
            inherited[chat_id] = revision
    return inherited


def _plan(root: pathlib.Path, stream: str, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Record the chain's segments in the ledger (new ones pending) and move the stream's
    coordinates: ``next_seq`` and ``files_seq`` never below the chain's length. A new segment whose
    chain place lies under coordinates the store already gave to other rows -- one a store-unaware
    body appended WITHOUT the rollback export, or one whose name sorts before imported segments (a
    clock set back, a copy from a backup) -- is placed after ``next_seq`` instead: its rows are
    never lost, at the price of their chain offsets. A segment that starts with the very row the
    store holds at its place (the rollback export's lines, continued by an older body) keeps it.
    An export map overrides that heuristic when any physical span differs: reserve a fresh
    range for new rows, but import the mapped lines at their original canonical coordinates."""
    total = sum(segment["size"] for segment in plan)
    added = mismatched = shifted = 0
    with ds.transaction(root, write=True) as conn:
        marks = conn.execute("SELECT next_seq, files_seq FROM streams WHERE stream=?", (stream,)).fetchone()
        next_seq, files_seq = int(marks["next_seq"]), int(marks["files_seq"])
        known = {record["name"]: dict(record) for record in conn.execute(
            "SELECT * FROM import_ledger WHERE stream=?", (stream,))}
        export = ds._meta(conn, f"export:{stream}") or {}
        coordinates = export.get("coordinates", [])
        end, planned = total, []
        for segment in plan:
            kept = known.get(segment["name"])
            if kept is None:
                base = segment["base"]
                # Rotation may sort this file before older archives (a clock set back).
                # Its generation and file-relative offsets survive that rename.
                generation = ""
                if coordinates:
                    with open(root / "archive" / segment["name"], "rb") as handle:
                        generation = _gen(handle.readline())
                mapped = ([[base + offset - export["file_base"], *entry] for offset, *entry in coordinates
                           if offset - export["file_base"] < segment["size"]]
                          if coordinates and (generation == export.get("generation")
                                              or base == export.get("file_base")) else [])
                changed = any(offset != seq or length != physical for offset, seq, length, physical, _digest in mapped)
                if base < next_seq and (changed or not _continues_store(root, conn, stream, segment)):
                    base, shifted = next_seq, shifted + 1  # its place is taken: there its rows would be lost
                next_seq = max(next_seq, base + segment["size"])
                end = max(end, base + segment["size"])
                cursor = conn.execute("INSERT INTO import_ledger (stream, name, base, size, mtime_ns, state) VALUES (?,?,?,?,?,'pending')",
                                      (stream, segment["name"], base, segment["size"], segment["mtime_ns"]))
                planned.append(cursor.lastrowid)
                if mapped:
                    # Keep physical offsets separate from canonical coordinates, including when
                    # a changed span made us reserve a fresh range for the older body's new rows.
                    ds._set_meta(conn, f"import_coordinates:{cursor.lastrowid}", [
                        [offset - segment["base"], *entry] for offset, *entry in mapped])
                    # A torn line is skipped just as before; later exports start after its
                    # allocated span, not after a now-unrelated physical byte offset.
                    files_seq = max(files_seq, max(entry[1] + entry[2] for entry in mapped))
                    coordinates = []
                added += 1
            elif kept["size"] != segment["size"] and kept["state"] != "mismatch":
                # A frozen segment whose bytes changed: its rows may not be the ledger's. A segment another
                # one moved in the chain (an earlier one grew or joined) is byte for byte what the ledger
                # recorded: its rows keep their recorded coordinates and an import stays done.
                conn.execute("UPDATE import_ledger SET state='mismatch' WHERE stream=? AND name=?", (stream, segment["name"]))
                mismatched += 1
        if added and export.get("coordinates"):
            # All live files were rotated before planning. Entries past their actual end
            # belong to an unwritten/torn batch, not to a future file at that offset.
            ds._set_meta(conn, f"export:{stream}", {**export, "coordinates": []})
        if planned and stream == "chat":  # this body derives them as it imports; an older body would not
            ds._set_meta(conn, ds.APPEARANCES_AWAITING, sorted({*(ds._meta(conn, ds.APPEARANCES_AWAITING) or []), *planned}))
            # Planned segments are tracked by awaiting, including on first cutover. Cross only a
            # previously derived prefix, never older-body live rows still below the frontier.
            conn.execute("UPDATE meta SET value=? WHERE key=? AND value=?",
                         (json.dumps(end), ds.APPEARANCES_MARK, json.dumps(int(marks["next_seq"]))))
        conn.execute("UPDATE streams SET next_seq=max(next_seq, ?), files_seq=max(files_seq, ?), "
                     "cutover_seq=coalesce(cutover_seq, ?) WHERE stream=?",
                     (end, total if not shifted else files_seq, total, stream))
        _settle(conn, stream)  # a chain without archive segments settles its counts here
    return {"segments": len(plan), "added": added, "mismatched": mismatched, "shifted": shifted, "chain_bytes": total}


def _continues_store(root: pathlib.Path, conn: Any, stream: str, segment: Dict[str, Any]) -> bool:
    """Whether ``segment`` starts with the row the store already holds at its chain place: the
    exported lines of a rollback, which a re-cutover takes back at their own coordinates."""
    try:
        with open(root / "archive" / segment["name"], "rb") as handle:
            first = handle.readline()
    except OSError:
        return False
    stored = conn.execute("SELECT body, length FROM dialogue WHERE stream=? AND seq=?",
                          (stream, segment["base"])).fetchone()
    line = next(iter(segment_rows(first, segment["base"])), None)
    return (stored is not None and line is not None and line.row is not None
            and line.length == stored["length"] and line.row == ds.decode(stored["body"]))


# --- importing segments ----------------------------------------------------------------------

def pending_segments(root: Any) -> List[Dict[str, Any]]:
    """The ledger's pending segments in import order: chat before progress, newest first."""
    with ds.transaction(root, write=False) as conn:
        records = conn.execute("SELECT * FROM import_ledger WHERE state='pending' "
                               "ORDER BY CASE stream WHEN 'chat' THEN 0 ELSE 1 END, base DESC").fetchall()
    return [dict(record) for record in records]


def import_keys(root: Any, lines: List[Any]) -> Optional[List[Tuple[str, int]]]:
    """``(key, seq)`` of the imported chat rows a Project shows outside their written chat; ``None``
    when the room rule cannot be read now (the job derives them; the mark stays)."""
    from ouroboros.room_appearances import write_facts

    keys: List[Tuple[str, int]] = []
    for line in lines:
        found, _shown = write_facts(root, line.row)
        if found is None:
            return None
        keys.extend((key, line.seq) for key in found)
    return keys


def record_imported(conn: Any, keys: List[Tuple[str, int]], lower: int, upper: int, segment: int) -> None:
    """Record the segment's keys and retire its awaiting entry in its import transaction. Advance the
    frontier only if it reaches this segment: an older body's unindexed appends may precede a newly
    restored archive, and importing that archive must not certify those earlier rows."""
    conn.execute(ds.APPEARANCES_DDL)
    conn.executemany("INSERT OR IGNORE INTO room_appearances (key, stream, seq) VALUES (?, 'chat', ?)", keys)
    conn.execute("UPDATE meta SET value=? WHERE key=? AND CAST(value AS INTEGER) >= ? AND CAST(value AS INTEGER) < ?",
                 (json.dumps(upper), ds.APPEARANCES_MARK, lower, upper))
    awaiting = ds._meta(conn, ds.APPEARANCES_AWAITING) or []
    if segment in awaiting:
        ds._set_meta(conn, ds.APPEARANCES_AWAITING, [entry for entry in awaiting if entry != segment])


def import_segment(root: Any, stream: str, name: str) -> Dict[str, Any]:
    """One frozen segment into the store in ONE transaction: its rows at their chain coordinates
    with their counts, the lines it skipped, the chat rows' legacy addresses, the rooms they
    touched, the ledger row ``done`` with the segment's totals, and the units' absolute bases
    when this segment completes the archive. A segment already imported (by this process or
    another) is left alone; one whose size is not the ledger's is recorded ``mismatch`` and
    skipped."""
    root = pathlib.Path(root)
    with ds.transaction(root, write=False) as conn:
        ledger = conn.execute("SELECT * FROM import_ledger WHERE stream=? AND name=?", (stream, name)).fetchone()
        marks = conn.execute("SELECT cutover_seq, live FROM streams WHERE stream=?", (stream,)).fetchone()
        coordinates = ds._meta(conn, f"import_coordinates:{ledger['id']}") if ledger is not None else None
    if ledger is None or ledger["state"] != "pending":
        return {"status": "skipped", "state": None if ledger is None else ledger["state"]}
    data = (root / "archive" / name).read_bytes()
    if len(data) != ledger["size"]:
        with ds.transaction(root, write=True) as conn:
            conn.execute("UPDATE import_ledger SET state='mismatch' WHERE stream=? AND name=? AND state='pending'", (stream, name))
        return {"status": "mismatch", "size": len(data), "expected": ledger["size"]}
    # An archive segment (below the cutover coordinate) is its own counting unit; one that joined
    # later counts in the live unit, after every row stored so far.
    archive = marks["cutover_seq"] is not None and int(ledger["base"]) < int(marks["cutover_seq"])
    gen = _gen(data)
    parsed, addresses, skipped = [], [], []
    touched: Dict[int, Tuple[Optional[int], str]] = {}
    objects = stream_rows = 0
    mapped = {entry[0]: entry for entry in coordinates or []}
    exported = set()
    verified = []
    for line in segment_rows(data, int(ledger["base"])):
        offset = line.seq - int(ledger["base"])
        entry = mapped.get(offset)
        if entry is not None:
            _offset, seq, length, physical, digest = entry
            raw = data[offset:offset + line.length]
            # A short write can leave a complete final object without only its LF.
            # Restore that byte for verification, never for an interior or malformed line.
            if (line.row is not None and line.length + 1 == physical
                    and offset + line.length == len(data) and not raw.endswith(b"\n")):
                raw += b"\n"
            if line.row is not None and len(raw) == physical and hashlib.sha256(raw).hexdigest() == digest:
                line = line._replace(seq=seq, length=length)
                exported.add(seq)
                verified.append(entry)
        if line.row is None:
            if line.kind != "blank":
                skipped.append((stream, line.seq, line.length, line.kind))
            continue
        columns = ds.facts(line.row)
        counted = ds.stream_row(stream, columns)
        parsed.append((line, columns, objects, stream_rows if counted else None, counted))
        objects, stream_rows = objects + 1, stream_rows + int(counted)
        if gen and stream == "chat":
            addresses.append((gen, line.number, line.seq))
        if columns["chat_id"] is not None:
            last_seq, last_ts = touched.get(columns["chat_id"], (None, ""))
            touched[columns["chat_id"]] = (max(last_seq or -1, line.seq) if stream == "chat" else last_seq,
                                           max(last_ts, columns["ts"]))
    # Imported rows are derived as they land (``room_appearances.import_keys``): the mark may then pass them.
    shown = import_keys(root, [line for line, *_rest in parsed]) if stream == "chat" else None
    with ds.transaction(root, write=True) as conn:
        current = conn.execute("SELECT state FROM import_ledger WHERE stream=? AND name=?", (stream, name)).fetchone()
        if current is None or current["state"] != "pending":
            return {"status": "skipped", "state": None if current is None else current["state"]}
        if archive:
            rows = [ds.row_values(stream, line.seq, line.length, int(ledger["id"]), ordinal, pos, columns, line.body)
                    for line, columns, ordinal, pos, _counted in parsed]
        else:
            rows = _live_rows(conn, stream, int(marks["live"]), int(ledger["base"]), int(ledger["size"]), parsed, exported)
        conn.executemany(ds._INSERT.format(" OR IGNORE"), rows)
        if addresses:
            conn.executemany("INSERT OR IGNORE INTO legacy_addresses (gen, line, seq) VALUES (?,?,?)", addresses)
        if skipped:
            conn.executemany("INSERT OR IGNORE INTO skipped (stream, seq, length, kind) VALUES (?,?,?,?)", skipped)
        if coordinates is not None:
            ds._set_meta(conn, f"import_coordinates:{ledger['id']}", verified)
        if shown is not None:
            record_imported(conn, shown, int(ledger["base"]), int(ledger["base"]) + int(ledger["size"]), int(ledger["id"]))
        for chat_id, (last_seq, last_ts) in touched.items():
            ds.touch_room(conn, chat_id, last_seq, last_ts, 0)
        conn.execute("UPDATE import_ledger SET state='done', rows=?, stream_rows=?, bad=?, gen=?, imported_at=? "
                     "WHERE stream=? AND name=?",
                     (objects, stream_rows if stream == "chat" else None, len(skipped), gen or None, utc_now_iso(), stream, name))
        if archive:
            _settle(conn, stream)
        else:
            _segment_bases(conn, stream, int(ledger["id"]), int(ledger["base"]), int(ledger["size"]))
    return {"status": "done", "rows": objects, "bad": len(skipped)}


def _live_rows(conn: Any, stream: str, live: int, base: int, size: int, parsed: List[Tuple[Any, ...]],
               exported: set[int]) -> List[Tuple[Any, ...]]:
    """The rows of a segment that joined after the cutover, counted in the live unit: a row the
    rollback export placed (already stored at its coordinate) keeps its counts; every other
    takes the next ones, in chain order."""
    present = {record["seq"] for record in conn.execute("SELECT seq FROM dialogue WHERE stream=? AND seq>=? AND seq<?",
                                                        (stream, base, base + size))}
    present.update(exported)
    marks = conn.execute("SELECT next_ordinal, next_pos FROM streams WHERE stream=?", (stream,)).fetchone()
    ordinal, pos = int(marks["next_ordinal"]), int(marks["next_pos"])
    rows = []
    for line, columns, _ordinal, _pos, counted in parsed:
        if line.seq in present:
            continue
        rows.append(ds.row_values(stream, line.seq, line.length, live, ordinal, pos if counted else None, columns, line.body))
        ordinal, pos = ordinal + 1, pos + int(counted)
    conn.execute("UPDATE streams SET next_ordinal=?, next_pos=? WHERE stream=?", (ordinal, pos, stream))
    return rows


def _settle(conn: Any, stream: str) -> None:
    """Once every segment below the cutover is imported: the absolute bases of the archive's
    units in chain order, then of the live unit (the archive's totals), then of the segments
    that joined after the cutover (``_segment_bases``). Settled once; cheap to ask again."""
    marks = conn.execute("SELECT cutover_seq, live FROM streams WHERE stream=?", (stream,)).fetchone()
    if marks["cutover_seq"] is None or conn.execute(
            "SELECT ordinal_base FROM import_ledger WHERE id=?", (marks["live"],)).fetchone()["ordinal_base"] is not None:
        return
    if conn.execute("SELECT 1 FROM import_ledger WHERE stream=? AND base<? AND state IN ('pending', 'mismatch') LIMIT 1",
                    (stream, marks["cutover_seq"])).fetchone():
        return
    ordinal = pos = 0
    for unit in conn.execute("SELECT id, rows, stream_rows FROM import_ledger WHERE stream=? AND state='done' AND base<? "
                             "ORDER BY base, id", (stream, marks["cutover_seq"])).fetchall():
        conn.execute("UPDATE import_ledger SET ordinal_base=?, pos_base=? WHERE id=?", (ordinal, pos, unit["id"]))
        ordinal, pos = ordinal + int(unit["rows"] or 0), pos + int(unit["stream_rows"] or 0)
    conn.execute("UPDATE import_ledger SET ordinal_base=?, pos_base=? WHERE id=?", (ordinal, pos, marks["live"]))
    for unit in conn.execute("SELECT id, base, size FROM import_ledger WHERE stream=? AND state='done' AND base>=? ORDER BY base",
                             (stream, marks["cutover_seq"])).fetchall():
        _segment_bases(conn, stream, int(unit["id"]), int(unit["base"]), int(unit["size"]))


def _segment_bases(conn: Any, stream: str, unit: int, base: int, size: int) -> None:
    """The absolute bases of a segment that joined after the cutover: the live unit's bases plus
    the counts of its first stored rows (None while the live unit is unsettled, or without rows)."""
    live = conn.execute("SELECT ordinal_base, pos_base FROM import_ledger WHERE id=(SELECT live FROM streams WHERE stream=?)",
                        (stream,)).fetchone()
    if live["ordinal_base"] is None:
        return
    first = conn.execute("SELECT ordinal FROM dialogue WHERE stream=? AND seq>=? AND seq<? ORDER BY seq LIMIT 1",
                         (stream, base, base + size)).fetchone()
    counted = conn.execute("SELECT pos FROM dialogue WHERE stream=? AND seq>=? AND seq<? AND pos IS NOT NULL ORDER BY seq LIMIT 1",
                           (stream, base, base + size)).fetchone()
    coordinates = ds._meta(conn, f"import_coordinates:{unit}") or []
    if coordinates:
        mapped = conn.execute("SELECT min(ordinal) AS ordinal, min(pos) AS pos FROM dialogue WHERE stream=? "
                              "AND seq IN (SELECT value FROM json_each(?))",
                              (stream, json.dumps([entry[1] for entry in coordinates]))).fetchone()
        if mapped["ordinal"] is not None and (first is None or mapped["ordinal"] < first["ordinal"]):
            first = mapped
        if mapped["pos"] is not None and (counted is None or mapped["pos"] < counted["pos"]):
            counted = mapped
    conn.execute("UPDATE import_ledger SET ordinal_base=?, pos_base=? WHERE id=?",
                 (None if first is None else int(live["ordinal_base"]) + int(first["ordinal"]),
                  None if counted is None else int(live["pos_base"]) + int(counted["pos"]), unit))


def run_dialogue_import(root: Any, *, progress: Optional[Callable[[dict], None]] = None,
                        should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """The archive import as a named job (``maintenance_jobs`` row ``dialogue_import``): resumable
    through the ledger, idempotent (a done segment is never read again), stoppable between
    segments (the row stays ``pending`` with its cursor), failures recorded with the segment."""
    from ouroboros import maintenance_jobs

    root = pathlib.Path(root)
    with store.name_lock(root, JOB_LOCK_NAME, timeout_sec=ds.WAIT_SEC, stale_sec=600.0):
        cutover(root)  # the store and its plan; a stat of two files when nothing changed

        def record(**facts: Any) -> dict:
            updated = maintenance_jobs.record(root, JOB, **facts)
            if progress is not None:
                progress(updated)
            return updated

        pending = pending_segments(root)
        row = maintenance_jobs.job(root, JOB) or {}
        if not pending:
            return row if row.get("state") == "done" else record(state="done", done=int(row.get("done") or 0), total=int(row.get("done") or 0))
        done = int(row.get("done") or 0)
        if should_stop is not None and should_stop():
            return record(state="pending", done=done, total=done + len(pending))
        segment: Dict[str, Any] = {}
        try:
            record(state="running", done=done, total=done + len(pending), error=None, waiting_on=None)
            for segment in pending:
                if should_stop is not None and should_stop():
                    return record(state="pending")
                outcome = import_segment(root, segment["stream"], segment["name"])
                if outcome["status"] == "done":
                    done += 1
                record(cursor=f"{segment['stream']}:{segment['name']}", done=done)
            return record(state="done", total=done)
        except BaseException as exc:
            maintenance_jobs.record(root, JOB, state="failed" if isinstance(exc, Exception) else "pending",
                                    error=f"{type(exc).__name__}: {exc}",
                                    waiting_on={"segment": segment.get("name"), "stream": segment.get("stream")})
            raise


# --- coverage ---------------------------------------------------------------------------------

def coverage(root: Any, stream: str, seq: Optional[int] = None) -> Dict[str, Any]:
    """Whether every segment below ``seq`` (the whole stream when None) is imported: ``imported``
    False means a reader that found no rows there must say "still importing", never "no rows";
    ``pending`` counts the segments not yet imported below ``seq``, ``mismatch`` the ones it never will."""
    if not ds.present(root):
        return {"imported": not ds.holds_dialogue_files(root), "pending": 0, "mismatch": 0, "gaps": 0}
    bound = (1 << 62) if seq is None else int(seq)
    with ds.transaction(root, write=False) as conn:
        counts = {record["state"]: record["n"] for record in conn.execute(
            "SELECT state, count(*) AS n FROM import_ledger WHERE stream=? AND state IN ('pending', 'mismatch') AND base<? "
            "GROUP BY state", (stream, bound))}
        gaps = ds._gaps(conn, stream, 0, bound)
        planned = ds._planned(conn, root)
    pending, mismatch = int(counts.get("pending", 0)), int(counts.get("mismatch", 0))
    if not planned:  # a cutover that failed after ``create``: the chain files are still to plan and import
        pending = max(pending, 1)
    return {"imported": not pending and not mismatch, "pending": pending, "mismatch": mismatch, "gaps": gaps,
            **({} if planned else {"unplanned": True})}


def import_owed(root: Any) -> bool:
    """Whether any planned segment still waits for the import (the maintenance dispatcher's look):
    one indexed probe per stream, never a listing or a count over the ledger."""
    with ds.transaction(root, write=False) as conn:
        return any(conn.execute("SELECT 1 FROM import_ledger INDEXED BY ledger_state WHERE stream=? AND state='pending' "
                                "LIMIT 1", (stream,)).fetchone() is not None for stream in ds.STREAMS)


def import_status(root: Any) -> Dict[str, Any]:
    """Per stream: segments by state, rows imported, bytes still pending (the health document's job view)."""
    status: Dict[str, Any] = {}
    if not ds.present(root):
        return status
    with ds.transaction(root, write=False) as conn:
        for stream in ds.STREAMS:
            summary = {"segments": 0, "done": 0, "pending": 0, "mismatch": 0, "rows": 0, "pending_bytes": 0}
            for record in conn.execute("SELECT state, count(*) AS n, coalesce(sum(rows), 0) AS rows, sum(size) AS bytes "
                                       "FROM import_ledger WHERE stream=? AND state<>'live' GROUP BY state", (stream,)):
                summary["segments"] += record["n"]
                summary[record["state"]] = record["n"]
                summary["rows"] += record["rows"]
                if record["state"] == "pending":
                    summary["pending_bytes"] = record["bytes"]
            status[stream] = summary
    return status


def scan_pending(root: Any, chat_id: Optional[int], client_message_id: str) -> Optional[ds.Row]:
    """While the chat archive is still importing, the inbound row an id names in the segments not
    yet imported (newest first; only lines holding the encoded id are parsed). ``None`` when
    the pending segments hold none -- then the store's answer was complete."""
    root = pathlib.Path(root)
    with ds.transaction(root, write=False) as conn:
        pending = conn.execute("SELECT name, base FROM import_ledger WHERE stream='chat' AND state='pending' ORDER BY base DESC").fetchall()
    if not pending:
        return None
    # A writer may have escaped the id (``ensure_ascii``) or not: both spellings are candidates.
    needles = {json.dumps(client_message_id, ensure_ascii=False).encode("utf-8"),
               json.dumps(client_message_id, ensure_ascii=True).encode("utf-8")}
    for record in pending:
        try:
            data = (root / "archive" / record["name"]).read_bytes()
        except OSError:
            continue
        if b'"in"' not in data or not any(needle in data for needle in needles):
            continue
        found = None
        for line in segment_rows(data, int(record["base"])):
            row = line.row
            if row is None or not any(needle in line.body.encode("utf-8") for needle in needles):
                continue
            if row.get("direction") == "in" and row.get("client_message_id") == client_message_id \
                    and ds.facts(row)["chat_id"] == chat_id:
                found = ds.Row("chat", line.seq, row, None, line.length)  # the newest row naming the id, as the store answers
        if found is not None:
            return found
    return None


# --- the rollback export ----------------------------------------------------------------------

def export_for_older_body(root: Any) -> Dict[str, Any]:
    """Append every row the chain does not hold yet (``seq >= files_seq``) to ``logs/<stream>.jsonl``
    in the exact line format, root and every descendant drive with a store; then the files hold
    everything (``files_seq = next_seq``). Run with the writers stopped (the switch seam)."""
    from ouroboros.store_import import descendant_drives

    root = pathlib.Path(root)
    report: Dict[str, Any] = {}
    for drive in (root, *descendant_drives(root)):
        if ds.present(drive):
            report[str(drive)] = {stream: _export_stream(drive, stream) for stream in ds.STREAMS}
    return report


def _export_stream(root: pathlib.Path, stream: str) -> Dict[str, Any]:
    """One stream's rows the chain files do not hold, appended as one batch. The batch is recorded
    (where it starts in the file, its bytes' hash, the coordinate it reaches) before its first byte
    is written, so an export interrupted after the append and before ``files_seq`` moved finishes
    that batch instead of appending it twice: the bytes already in the file must be the batch's
    beginning, and only the rest is appended. The same transaction records each exported line's
    [physical chain offset, canonical seq, allocated length, physical length, SHA-256] in
    ``export:<stream>.coordinates``, with the file's base and generation to survive rotation.
    Planning transfers that map to the segment's ledger meta;
    revised JSON can change byte lengths without changing any existing row's identity."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    key = f"export_batch:{stream}"
    with ds.transaction(root, write=False) as conn:
        marks = conn.execute("SELECT next_seq, files_seq FROM streams WHERE stream=?", (stream,)).fetchone()
        begun = ds._meta(conn, key)  # a batch an interrupted export recorded, wholly or partly written
        if begun and int(begun.get("from", -1)) != int(marks["files_seq"]):
            # A cutover took the chain files since (the written part is in an imported segment, a cut
            # line its skipped interval): the record is stale and this export starts afresh.
            begun = None
        upper = int(begun["to"]) if begun else int(marks["next_seq"])
        # The rows the files do not hold: appended after ``files_seq`` and not inside an imported segment.
        records = conn.execute(
            "SELECT seq, length, body FROM dialogue WHERE stream=? AND seq>=? AND seq<? AND NOT EXISTS (SELECT 1 FROM import_ledger "
            "WHERE import_ledger.stream=dialogue.stream AND state='done' AND base<=seq AND seq<base+size) ORDER BY seq",
            (stream, marks["files_seq"], upper)).fetchall()
    data = b"".join(record["body"].encode("utf-8") + b"\n" for record in records)
    live = root / ds.LIVE_REL[stream]
    if records:
        live.parent.mkdir(parents=True, exist_ok=True)
        lock = jsonl_append_lock_path(live)
        fd = acquire_exclusive_file_lock(lock, timeout_sec=ds.WAIT_SEC, stale_sec=10.0, owner_aware_stale=True)
        if fd is None:
            raise StoreBusy(f"dialogue export: the append lock of {live} is held", reason="contention")
        try:
            size = live.stat().st_size if live.exists() else 0
            digest = hashlib.sha256(data).hexdigest()
            if begun:
                offset, written = int(begun["offset"]), b""
                if size > offset:  # an absent file at offset 0 holds none of the batch yet
                    with open(live, "rb") as handle:
                        handle.seek(offset)
                        written = handle.read()
                if size < offset or digest != begun["sha256"] or not data.startswith(written):
                    raise ds.StoreError(f"dialogue export: {live} changed under an interrupted export; nothing appended")
                remainder = data[len(written):]
            else:
                file_base = sum(segment["size"] for segment in segment_plan(root, stream))
                offset = file_base + size
                if size:
                    with open(live, "rb") as handle:
                        generation = _gen(handle.readline())
                else:
                    generation = _gen(data)
                coordinates = []
                for record in records:
                    raw = record["body"].encode("utf-8") + b"\n"
                    coordinates.append([offset, record["seq"], record["length"], len(raw), hashlib.sha256(raw).hexdigest()])
                    offset += len(raw)
                with ds.transaction(root, write=True) as conn:
                    ds._set_meta(conn, key, {"offset": size, "length": len(data), "sha256": digest, "to": upper,
                                             "from": int(marks["files_seq"])})
                    export = ds._meta(conn, f"export:{stream}") or {}
                    ds._set_meta(conn, f"export:{stream}", {**export, "file_base": file_base, "generation": generation,
                        "coordinates": [*export.get("coordinates", []), *coordinates]})
                remainder = data
            if remainder:
                # Byte-exact: a Windows descriptor is text mode without O_BINARY and would write each
                # LF as CRLF, so the exported lines would no longer be the stored rows at their coordinates.
                handle = os.open(str(live), os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o644)
                try:
                    view = memoryview(remainder)
                    while view:
                        view = view[os.write(handle, view):]
                    os.fsync(handle)
                finally:
                    os.close(handle)
                store.fsync_directory(live.parent)
        finally:
            release_exclusive_file_lock(lock, fd)
    elif not begun:
        return {"rows": 0, "from": marks["files_seq"], "to": marks["next_seq"]}
    with ds.transaction(root, write=True) as conn:
        conn.execute("UPDATE streams SET files_seq=? WHERE stream=?", (upper, stream))
        conn.execute("DELETE FROM meta WHERE key=?", (key,))
        export = ds._meta(conn, f"export:{stream}") or {}
        ds._set_meta(conn, f"export:{stream}", {**export, "at": utc_now_iso(), "from": marks["files_seq"], "to": upper,
                                                "rows": len(records)})
    report = {"rows": len(records), "from": marks["files_seq"], "to": upper, "bytes": len(data)}
    if begun and upper < int(marks["next_seq"]):  # the finished batch first, then the rows after it
        later = _export_stream(root, stream)
        report = {**later, "from": report["from"], "rows": report["rows"] + later["rows"],
                  "bytes": report["bytes"] + later.get("bytes", 0)}
    return report
