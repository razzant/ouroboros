"""How much WAL log a store keeps, when a writer copies it back inline, and the job that copies it off that path.

SQLite checkpoints INLINE: whichever COMMIT leaves the log at ``wal_autocheckpoint`` frames or more copies every
frame no checkpoint copied yet into the database file and syncs it, on that writer's thread, inside its COMMIT
(the default, 1,000 frames, is about 4 MiB of 4 KiB pages). ``store.connect`` sets the threshold of every WAL
connection it opens, in every process, to ``ceiling``: the frames that fit in ``JOURNAL_SIZE_LIMIT``, the log size
each store already keeps after a reset (16,288 frames of 4 KiB pages). Nothing is persisted in the file: an older
body opens with SQLite's default again. A rollback-journal store (PERSIST, the name tier) has no log; its threshold
is left alone and the job passes it by.

Below the ceiling the maintenance child's job ``wal_checkpoint`` copies the log, in a lane of its own
(``maintenance.OWN_LANES``: packing or an import never holds it), with ``PRAGMA wal_checkpoint(PASSIVE)``: it waits
for no reader and no writer and stops at the oldest snapshot a reader holds (never RESTART or TRUNCATE, which wait
for both). The job is due when a store's log holds ``inline_frames`` committed frames (SQLite's own default: the
work it would have done inline anyway) that its last pass did not see. That is read from the log's own header and
frame headers (``store.log_identity``; the log carries no lock), never by opening a store to count, never from its index
(closing any descriptor of the ``-shm`` drops this process's locks on it: ``store.is_wal_index``) and never by a
clock alone; the server's 20 s duty ``wal_checkpoint_look`` (``server_maintenance``) only asks the lane to look.

A pass records, per store kind (``store_timing.KINDS``), SQLite's three numbers (``busy``; ``log``: frames in the
log; ``checkpointed``: frames now in the database) and the checkpoint's wall ``seconds``, on the job's row and as
one ``wal_checkpoint`` row of ``logs/supervisor.jsonl``. Beside the ``supervisor_load`` rows' commit times they show
whether long COMMITs go on while the log stays short (an interactive COMMIT checkpoints only at the ceiling),
whether a reader holds the log back (``checkpointed`` below ``log``), and whether the log is ever reset (``log``
growing pass after pass though each pass copied it all: a reader was on the log whenever a writer began).

The worse worst case, disclosed: when the job is down, late, starved or held back by a reader, the log grows to the
ceiling and the next interactive COMMIT copies up to ~64 MiB of frames inline, more than any inline checkpoint
before (about 16 times the old 4 MiB); a last close (no other connection in any process) copies as much inside its
``close``; crash recovery replays up to the ceiling on the next open. ``store.hand_off`` and the export seam still
truncate the log.
"""
from __future__ import annotations

import contextlib
import functools
import pathlib
import sqlite3
import time
from typing import Any, Callable, Dict, Iterator, Tuple

JOURNAL_SIZE_LIMIT = 64 << 20  # bytes a store keeps of an idle rollback journal, or of its WAL log after a reset
_LOG_HEADER, _FRAME_HEADER = 32, 24  # the WAL log's header and each frame's header (sqlite.org/fileformat2.html §4)
NAME = "wal_checkpoint"  # the maintenance job and its row (``maintenance_jobs``)


def ceiling(page_size: int) -> int:
    """The frames of PAGE_SIZE bytes that fit in ``JOURNAL_SIZE_LIMIT`` after the log's header."""
    return (JOURNAL_SIZE_LIMIT - _LOG_HEADER) // (page_size + _FRAME_HEADER)


def limit(conn: sqlite3.Connection) -> None:
    """``store.connect``'s journal limits: the bytes kept after a reset, and a WAL connection's inline ceiling."""
    conn.execute(f"PRAGMA journal_size_limit = {JOURNAL_SIZE_LIMIT}")
    if conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
        conn.execute(f"PRAGMA wal_autocheckpoint = {ceiling(conn.execute('PRAGMA page_size').fetchone()[0])}")


@functools.lru_cache(maxsize=None)
def inline_frames() -> int:
    """SQLite's own default ``wal_autocheckpoint`` as this build has it (1,000), read once from a memory database;
    at least one frame, so an unchanged log is never due."""
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        return max(1, int(conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]))


def _logs(root: pathlib.Path) -> Iterator[Tuple[str, pathlib.Path, Callable[[], Any], list]]:
    """Each store of ROOT whose WAL log holds a commit: its kind, path, header-tier reader and the log's
    ``[generation, last committed frame]``. A store without a log costs one ``stat``; the database header is read
    only beside a log (a rollback store's leftover log is not a WAL store's)."""
    from ouroboros import store, store_journal, store_timing

    for path, tier, _lock in store_journal.root_stores(root):
        log = store.log_identity(path)
        if log is not None and store.wal_file(path):
            yield store_timing.kind_of(path), path, tier, log[:2]


def unseen(log: list, seen: Any) -> int:
    """The committed frames of LOG a pass that saw SEEN did not: past its end in the same generation, else all."""
    generation, end = log
    if isinstance(seen, list) and len(seen) == 2 and seen[0] == generation and end >= int(seen[1]):
        return end - int(seen[1])
    return end


def due(root: Any, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """A store's log holds ``inline_frames`` committed frames the job's last pass did not see (module docstring)."""
    stores = row.get("stores") or {}
    return any(unseen(log, (stores.get(kind) or {}).get("wal")) >= inline_frames()
               for kind, _path, _tier, log in _logs(pathlib.Path(root)))


def run(root: Any, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """One PASSIVE checkpoint of each WAL store whose log holds a commit the last pass did not see, through the
    store's own connection (its tier, ``synchronous=FULL``); the row keeps per store the numbers and the log it saw.
    A store that cannot be opened or checkpointed keeps its error class there instead, at the log it met, so it is
    tried again when its log moves on, never on a clock."""
    from ouroboros import maintenance_jobs, store
    from ouroboros.utils import utc_now_iso

    root = pathlib.Path(root)
    stores = dict((maintenance_jobs.job(root, NAME) or {}).get("stores") or {})  # the store's foreign-root check first
    passed: Dict[str, Dict[str, Any]] = {}
    for kind, path, tier, log in _logs(root):
        if should_stop():
            break
        if not unseen(log, (stores.get(kind) or {}).get("wal")):
            continue
        try:
            with contextlib.closing(store.connect(path, tier(), wait_sec=store.STORE_WAIT_SEC)) as conn:
                began = time.monotonic()
                busy, frames, copied = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                facts = {"busy": busy, "log": frames, "checkpointed": copied,
                         "seconds": round(time.monotonic() - began, 3)}
        except (sqlite3.Error, store.StoreError, OSError) as exc:
            facts = {"error": type(exc).__name__}
        passed[kind] = facts
        stores[kind] = {**facts, "wal": log, "at": utc_now_iso()}
    if passed:
        maintenance_jobs.record(root, NAME, state="done", stores=stores)
        store.supervisor_row(root, NAME, {"stores": passed})
    return {"state": "done", "items": len(passed)}
