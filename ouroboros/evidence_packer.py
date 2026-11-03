"""Evidence packing passes the maintenance child runs (``ouroboros/maintenance.py``, job ``evidence_packing``);
storage protocol: docs/PERSISTENCE.md, evidence pack row.

A task is packed once its row is terminal and no review of it is open (``evidence_packs.task_packable``:
the row's ``review_collection_pending`` fact names an open review operation or an uncollected acceptance,
whose agent-session reader was handed a retained handle's absolute path): no duty, money or delivery
fact is consulted, because packing moves verified bytes, never content, and anything written for the
task later becomes its next generation. Loose blobs stay for the tasks whose row is live or under an
open review (the keep-set, read through the status and review indexes) and for the manifests published
during a pass (the publication journal). A pass's census is the tasks whose rows were written terminal
since the position its last pass recorded (``tasks.terminal_seq``) plus the candidates that pass left
unfinished; a position holds only under the store's body fact it was recorded with (``store_body``: an
older body's writes stamp no ``terminal_seq``), else the pass lists the evidence folders whole (``census_since``).
"""
from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import pathlib
import sqlite3
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ouroboros import evidence_packs as packs, store

log = logging.getLogger(__name__)
# ``packed`` = new generations written; ``examined`` = candidates processed; ``verified_bytes`` = raw bytes re-read;
# ``missing`` = referenced records neither loose nor packed; ``live``/``reading``/``storeless`` = folders still
# holding loose evidence because their task runs / is terminal under an open review / has no store row;
# ``deferred`` = candidates whose unlink waited for a held custody lock or a moved stamp (retried next pass).
COUNTERS = ("packed", "examined", "swept", "candidates", "raw", "compressed", "verified_bytes",
            "unlinked", "kept", "busy", "faults", "archived", "missing", "live", "reading", "storeless", "deferred")


_ROW_COLUMNS = ("status", "write_seq", "updated_at", "review_collection_pending")


def _terminal(row: Optional[tuple]) -> bool:
    from ouroboros.task_status import SETTLED_STATUSES

    return row is not None and str(row[0] or "") in SETTLED_STATUSES


def _packable(row: Optional[tuple]) -> bool:
    return row is not None and packs.row_packable(row[0], row[3])


def task_rows(root: pathlib.Path, ids, *, txn=None) -> Dict[str, Optional[tuple]]:
    """``{task_id: (status, write_seq, updated_at, review_collection_pending)}`` for IDS from one indexed
    select (``None``: no row); the only store facts a pass reads, in one short read closed before any
    directory probe."""
    ids = sorted(set(ids))
    with (store.open_store(root, write=False) if txn is None else contextlib.nullcontext(txn)) as txn:
        found = {record["task_id"]: tuple(record[name] for name in _ROW_COLUMNS) for record in
                 txn.conn.execute(f"SELECT task_id, {', '.join(_ROW_COLUMNS)} FROM tasks WHERE task_id IN "
                                  "(SELECT value FROM json_each(?))", (json.dumps(ids),))}
    return {task: found.get(task) for task in ids}


def task_stamps(root: pathlib.Path, task_id: str, *, txn=None) -> str:
    """One task's stamp: its row from one short read (or the caller's ``txn``), then ``_stamp``."""
    with (store.open_store(root, write=False) if txn is None else contextlib.nullcontext(txn)) as txn:
        row = txn.row(str(task_id), *_ROW_COLUMNS)
    return _stamp(root, task_id, row)


def _stamp(root: pathlib.Path, task_id: str, row: Optional[tuple]) -> str:
    """The facts the rule reads: the row's status and review fact (``None``: no row) and the two evidence
    directories' identity, built from facts already read: no store access, so no transaction spans its
    ``stat`` calls."""
    from ouroboros.artifacts import task_artifact_dir_path

    stamps: list = [("status", None if row is None else str(row[0] or "")), ("review", None if row is None else row[3])]
    paths = [root / "observability" / "calls" / task_id,
             task_artifact_dir_path(root, task_id, create=False).joinpath(*packs.CHECKPOINTS_REL.parts)]
    for path in paths:
        try:
            stat = path.stat()
            stamps.append((str(path.relative_to(root)), stat.st_ino, stat.st_size, stat.st_mtime_ns))
        except OSError:
            stamps.append((str(path.relative_to(root)), 0))
    return json.dumps(stamps, sort_keys=True, separators=(",", ":"))


def _loose_task_ids(root: pathlib.Path, stop: Optional[Callable[[], bool]] = None) -> set:
    ids = set()
    for directory, checkpoints in ((root / "observability" / "calls", False),
                                   (root / "task_results" / "artifacts", True)):
        if stop is not None and stop():
            break
        with contextlib.suppress(OSError), os.scandir(directory) as entries:
            for entry in entries:
                if stop is not None and stop():
                    return ids
                if entry.is_dir(follow_symlinks=False) and (not checkpoints or
                        (pathlib.Path(entry.path) / "source_handles" / "context_checkpoints").is_dir()):
                    ids.add(entry.name)
    return ids


def swept_stamps(root: pathlib.Path, *, discovery: bool = False, ids=None) -> Optional[Dict[str, str]]:
    """Per task, the stamps the last sweep recorded (``ids``: those tasks' rows only, by key); ``None``
    when the cache is unavailable."""
    if not packs.packs_dir(root).is_dir():
        return {}
    query, params = "SELECT task_id, stamps FROM swept", ()
    if ids is not None:
        query, params = query + " WHERE task_id IN (SELECT value FROM json_each(?))", (json.dumps(sorted(ids)),)
    try:
        with packs.open_index(root, **({"wait_sec": 0, "rebuild": False} if discovery else {})) as conn:
            return {row["task_id"]: row["stamps"] for row in conn.execute(query, params)}
    except store.StoreBusy:
        if discovery:
            raise  # advisory discovery skips this pass instead of waiting or spawning on contention
        return None
    except (store.StoreError, sqlite3.Error, OSError) as exc:
        if discovery and isinstance(exc, sqlite3.OperationalError) and store.busy(exc):
            raise store.StoreBusy("evidence index read busy", reason="contention") from exc
        return None


def record_swept(root: pathlib.Path, task_id: str, stamps: str) -> None:
    with contextlib.suppress(store.StoreError, sqlite3.Error, OSError):
        with packs.open_index(root, write=True) as conn:
            conn.execute("INSERT OR REPLACE INTO swept VALUES (?,?)", (task_id, stamps))


def _task_loose(root: pathlib.Path, task_id: str) -> bool:
    """``_loose_task_ids``' rule for one task: a calls folder, or a checkpoint folder under its artifacts."""
    from ouroboros.artifacts import task_artifact_dir_path

    return ((root / "observability" / "calls" / task_id).is_dir() or
            task_artifact_dir_path(root, task_id, create=False).joinpath(*packs.CHECKPOINTS_REL.parts).is_dir())


def loose_task_ids_since(root: pathlib.Path, since: int, owed=()) -> set:
    """The census of a pass with a position: the tasks whose rows were written terminal after the
    counter stood at SINCE (``task_store.terminal_writes_since``, one indexed read) and the candidates
    an earlier pass left unfinished (OWED), those with loose evidence (two stats each); never a listing."""
    from ouroboros.task_results import validate_task_id
    from ouroboros.task_store import terminal_writes_since

    ids = set()
    for task_id in (*owed, *terminal_writes_since(root, since)):
        try:
            task_id = validate_task_id(task_id)
        except ValueError:
            continue
        if _task_loose(root, task_id):
            ids.add(task_id)
    return ids


def store_body(root: pathlib.Path) -> Optional[Dict[str, Any]]:
    """The store's body fact, ``meta.last_store_body`` (``store_import._ensure``: the HEAD that last booted
    on the store, rewritten only when it differs): one meta read, the fact a position is recorded under."""
    with store.open_store(root, write=False) as txn:
        body = txn.meta("last_store_body")
    return dict(body) if isinstance(body, dict) else None


def census_since(row: Dict[str, Any], counter: int, body: Optional[Dict[str, Any]]) -> Optional[int]:
    """The counter position the job row's last pass covered, when the rows written since can be
    addressed: this body's pass records ``census`` beside ``terminal`` (the same value), ``pending``
    as the ids it left and ``body`` as the store's body fact (BODY: the fact now), so a pass of a body
    that stamps no ``terminal_seq`` (it moves ``terminal`` alone and counts ``pending``), a store restored
    from a copy (the counter below the position) and another HEAD booted on the store since (its writes
    moved the counter without a stamp, whether or not its packer ran) all read as ``None``: a whole
    census is owed, as at the first run."""
    census, pending = row.get("census"), row.get("pending")
    if (isinstance(census, int) and census == row.get("terminal") and 0 <= census <= int(counter)
            and (not pending or isinstance(pending, list)) and row.get("body") == body):
        return census
    return None


def owed_ids(row: Dict[str, Any]) -> Tuple[str, ...]:
    """The candidates the job row's last pass left unfinished (``pending``: deferred, busy or faulted)."""
    pending = row.get("pending")
    return tuple(str(task) for task in pending) if isinstance(pending, list) else ()


def _changed_stamps(root: pathlib.Path, *, rows: Optional[dict] = None, ids: Optional[set] = None,
                    stop: Optional[Callable[[], bool]] = None, since: Optional[int] = None,
                    owed=()) -> Iterator[Tuple[str, str]]:
    """Tasks with loose evidence whose status or directories changed since a sweep recorded them, with
    their stamps; an owed task is yielded whatever its stamp (another attempt is what it is owed).
    The rows come from ONE short read that is closed before the directory probes."""
    root = pathlib.Path(root)
    if ids is None and since is not None:
        ids = loose_task_ids_since(root, since, owed)
    elif ids is None:
        ids = _loose_task_ids(root) if stop is None else _loose_task_ids(root, stop)
    if not ids or (stop is not None and stop()):
        return
    if rows is None:
        rows = task_rows(root, ids)
    if stop is not None and stop():
        return
    swept = swept_stamps(root, discovery=stop is not None, ids=ids)
    for task in sorted(ids):
        if stop is not None and stop():
            return
        stamp = _stamp(root, task, rows.get(task))
        if swept is None or task in owed or swept.get(task) != stamp:
            yield task, stamp


def changed_stamps(root: pathlib.Path, **census) -> Dict[str, str]:
    """Full child census; never reuse the parent's advisory discovery as unlink authority."""
    return dict(_changed_stamps(root, **census))


def packing_pending(root: pathlib.Path, stop: Callable[[], bool], *, since: Optional[int] = None, owed=()) -> bool:
    """Advisory, stop-aware discovery: archives first, then only the first changed stamp of the
    census (``since``/``owed`` as ``run_pass``: the rows written terminal since the position and
    the owed candidates; without a position, every loose folder).

    Contended stores defer the pass. Index reconstruction belongs to the pass (``run_pass``),
    not this reader. A quiet nonempty root without a position still needs a full stamp walk;
    the maintenance child runs it, never the server (``ouroboros/maintenance.py``).
    """
    if stop():
        return False
    if archive_pending(root):
        return not stop()
    with store.acquisition_budget(0):
        changed = next(_changed_stamps(root, stop=stop, since=since, owed=owed), None)
    return changed is not None and not stop()


def changed_tasks(root: pathlib.Path, **census) -> List[str]:
    return sorted(changed_stamps(root, **census))


def archive_pending(root: pathlib.Path) -> bool:
    directory = pathlib.Path(root) / packs.USAGE_ARCHIVE_REL
    return directory.is_dir() and any(directory.glob("segment_ep*.jsonl"))


def terminal_revision(root: pathlib.Path) -> int:
    """The dispatcher's fact for the packing rule: how many times a terminal row was written, the
    counter the store keeps where it writes rows (``store.TERMINAL_WRITES``): one meta read, whatever
    the history holds. A task closing, or a closed task written again (late evidence rides a row
    write), moves it; a live task's writes do not, so the child is never spawned for them."""
    with store.open_store(root, write=False) as txn:
        return int(txn.meta(store.TERMINAL_WRITES) or 0)


def ordered_candidates(root: pathlib.Path, packable: List[str], *, rows: Optional[dict] = None) -> List[Tuple[str, int]]:
    """Packable tasks with their loose raw size: the ones whose result changed after the newest
    published pack first (newest result first), then the backfill by descending raw size."""
    from ouroboros.task_store import signature

    def timestamp(value):
        try:
            return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            return 0

    if not packable:  # nothing to order: no sidecar is read
        return []
    newest_pack = max((timestamp(sidecar.get("created_at")) for _, sidecar in packs.list_sidecars(root)), default=0)
    recent, backfill = [], []
    for task in packable:
        raw = sum(int(m["size"]) for m in packs.loose_members(root, task))
        row = rows.get(task) if rows is not None else None
        revision = row[1:] if row is not None else signature(root, task)
        touched = timestamp(revision[1]) if revision else 0
        (recent if newest_pack and touched > newest_pack else backfill).append((task, raw, touched))
    recent.sort(key=lambda item: (-item[2], item[0]))
    backfill.sort(key=lambda item: (-item[1], item[0]))
    return [(task, raw) for task, raw, _ in recent + backfill]


def keep_rows(root: pathlib.Path) -> Dict[str, tuple]:
    """The rows whose loose blobs a pass keeps (``PackSweep.collect_keep``): the tasks not settled
    (``task_status.NONTERMINAL_STATUSES``) and the tasks under an open review, from the status and
    review indexes in one short read, whatever the census covers."""
    from ouroboros.task_status import NONTERMINAL_STATUSES

    live = sorted(NONTERMINAL_STATUSES)
    # Each term on its own index, joined as a union: under the statistics the maintenance child's ANALYZE writes,
    # SQLite plans the plain disjunction as a scan of every task (tests/test_store_capture_readers.py).
    with store.open_store(root, write=False) as txn:
        return {record["task_id"]: tuple(record[name] for name in _ROW_COLUMNS) for record in txn.conn.execute(
            f"SELECT task_id, {', '.join(_ROW_COLUMNS)} FROM tasks WHERE task_id IN (SELECT task_id FROM tasks "
            f"WHERE unlikely(status IN ({','.join('?' * len(live))})) UNION SELECT task_id FROM tasks "
            "WHERE unlikely(review_collection_pending=1))", live)}


class PackSweep:
    """The keep-set of one pass: the blobs referenced by tasks whose row is live or under an open review
    (``keep_rows``), read once outside the deletion locks (``collect_keep``), plus the refs of manifests
    published during the pass, read from the journal suffix under the publication lock (``refresh_keep``).
    Store-less folders are neither packed nor kept. A blob such a task shares with a packed one stays loose
    until the sharer's own pass; a terminal task resumed after the census reads its packed blobs from the pack."""
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.keep: set = set()
        self.journal = None

    def collect_keep(self, rows: Dict[str, Optional[tuple]]) -> None:
        for task, row in rows.items():
            if row is not None and not _packable(row):
                self.keep.update(sha for _, sha, _ in packs.manifest_blob_refs(self.root, task))

    def refresh_keep(self) -> set:
        # The caller holds the publication lock: only the appended suffix is read.
        for line in self.journal:
            sha = line.decode("ascii").strip()
            if len(sha) != 64 or sha.strip("0123456789abcdef"):
                raise OSError("incomplete evidence publication journal")
            self.keep.add(sha)
        return self.keep


@contextlib.contextmanager
def pack_sweep(root: pathlib.Path):
    """Serialize packer passes, tracking publications from before the census to the
    last unlink. A crashed pass's scratch journal is replaced by the next pass."""
    with store.name_lock(root, "packs_sweep.lock", timeout_sec=packs.INDEX_WAIT_SEC):
        sweep = PackSweep(root)
        path = packs.packs_dir(root) / packs.PUBLICATIONS_NAME
        with store.name_lock(root, packs.PUBLICATION_LOCK_NAME, timeout_sec=packs.INDEX_WAIT_SEC):
            path.parent.mkdir(parents=True, exist_ok=True)
            sweep.journal = path.open("w+b")
        try:
            yield sweep
        finally:
            with store.name_lock(root, packs.PUBLICATION_LOCK_NAME, timeout_sec=packs.INDEX_WAIT_SEC):
                sweep.journal.close()
                path.unlink()


def run_pass(root: pathlib.Path, *, sweep=None, since: Optional[int] = None, owed=()) -> Dict[str, Any]:
    """Pack every terminal candidate of the census one task at a time, then the money archive and the
    unpacker. The census is the rows written terminal after the counter stood at ``since`` and the
    ``owed`` candidates (``loose_task_ids_since``); without a position (``None``: the first run, a
    repair, a recorded position that cannot be read) every loose folder. ``unfinished`` names the
    candidates this pass leaves owed (deferred, busy, faulted)."""
    root = pathlib.Path(root)
    if sweep is None:
        with pack_sweep(root) as sweep:
            return run_pass(root, sweep=sweep, since=since, owed=owed)
    report: Dict[str, Any] = {"status": "completed", **dict.fromkeys(COUNTERS, 0), "unfinished": []}
    ids = _loose_task_ids(root) if since is None else loose_task_ids_since(root, since, owed)
    rows = task_rows(root, ids)  # the census baseline: older than every fact read below
    report["live"] = sum(1 for row in rows.values() if row is not None and not _terminal(row))
    report["reading"] = sum(1 for row in rows.values() if _terminal(row) and not _packable(row))
    report["storeless"] = sum(1 for row in rows.values() if row is None)
    # A live, reviewed or store-less folder is recorded at its census stamp, which moves with its status
    # and review fact; a packed task is recorded as its own unlinks left its directories (below), so a
    # later census over unchanged history finds nothing, and a row written terminal again brings the
    # task back whatever its stamp moved (``loose_task_ids_since``).
    revisions = changed_stamps(root, rows=rows, ids=ids, owed=owed)
    packable = [task for task in revisions if _packable(rows.get(task))]
    for task in revisions:
        if task not in packable:
            record_swept(root, task, revisions[task])
            report["swept"] += 1
    candidates = ordered_candidates(root, packable, rows=rows)
    report["candidates"] = len(candidates)
    # A pass from a position reads each candidate's own sidecars by name (``published_generations``); a whole
    # census (no position: the first run, a repair) shares one listing of them all.
    listing = packs.list_sidecars(root) if since is None and candidates else None
    if candidates:
        sweep.collect_keep(keep_rows(root))
    for task, raw in candidates:
        try:
            outcome = packs.pack_task(root, task, keep_shas=set(), listing=listing, revision=revisions[task],
                                      sweep=sweep)
        except Exception as exc:
            report["faults"] += 1
            report["unfinished"].append(task)
            packs.record_storage_pack(root, phase="fault", task_id=task, exception_class=type(exc).__name__)
            log.warning("Evidence pack of %s failed", task, exc_info=True)
            continue
        report["examined"] += 1
        report["packed"] += int(outcome.get("generation") is not None)
        report["raw"] += int(outcome.get("raw") or 0)
        report["compressed"] += int(outcome.get("packed") or 0)
        for key in ("unlinked", "kept", "busy", "verified_bytes", "missing", "deferred"):
            report[key] += int(outcome.get(key) or 0)
        # A blob kept for a live sharer does not re-queue this task: the sharer's own pass unlinks it
        # once it closes (its stamp moves then); busy files and deferred candidates retry. A finished
        # candidate is recorded as its unlinks left its directories, outside every lock.
        if not any(outcome.get(key) for key in ("busy", "deferred")):
            record_swept(root, task, _stamp(root, task, rows.get(task)))
            report["swept"] += 1
        else:
            report["unfinished"].append(task)
    report["unfinished"].sort()
    try:
        report["archived"] = packs.archive_usage_segments(root)["archived"]
    except Exception as exc:
        report["faults"] += 1
        packs.record_storage_pack(root, phase="fault", task_id="", exception_class=type(exc).__name__)
    return report
