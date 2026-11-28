"""How much WAL log a store keeps, when a writer copies it back inline, and the job that copies it off that path.

SQLite checkpoints INLINE: whichever COMMIT leaves the log at ``wal_autocheckpoint`` frames or more copies every
frame no checkpoint copied yet into the database file and syncs it, on that writer's thread, inside its COMMIT
(the default, 1,000 frames, is about 4 MiB of 4 KiB pages). ``store.connect`` sets that threshold, in every process,
to ``ceiling`` on a WAL connection to a store the job below passes (``serviced``: the three stores of the install the
process acts as): the frames that fit in ``JOURNAL_SIZE_LIMIT``, the log size each store already keeps after a reset
(16,288 frames of 4 KiB pages). Every other WAL store (a child drive's, headless child data's, a private build's, the
evidence index) keeps SQLite's own threshold, since no pass copies its log. Nothing is persisted in the file: an
older body opens with SQLite's default again. A rollback-journal store (PERSIST, the name tier) has no log; its
threshold is left alone and the job passes it by.

The ceiling is a trigger, not a bound. SQLite counts the log's frames after a transaction has appended its pages,
so one large transaction crosses it by its own size; a reader holding an old snapshot stops every checkpoint at that
snapshot and keeps the log from resetting while writers append past the ceiling, and the first commit after that
reader lets go copies the whole backlog inline. A last close (no other connection in any process) copies every
frame not yet copied inside its ``close``, and crash recovery reads the whole log on the next open, however long it
grew: ``JOURNAL_SIZE_LIMIT`` bounds only the file a reset leaves.

Below the ceiling the maintenance child's job ``wal_checkpoint`` copies the log, in a lane of its own
(``maintenance.OWN_LANES``: packing or an import never holds it), with ``PRAGMA wal_checkpoint(PASSIVE)``: it waits
for no reader and no writer and stops at the oldest snapshot a reader holds (never RESTART or TRUNCATE, which wait
for both). The job is due when a store's log holds ``inline_frames`` committed frames (SQLite's own default: the
work it would have done inline anyway) that its last pass did not see. That is read from the log's own header and
frame headers (``store.log_identity``; the log carries no lock), never by opening a store to count, never from its index
(closing any descriptor of the ``-shm`` drops this process's locks on it: ``store.is_wal_index``) and never by a
clock alone. The look reads them before the job's row (``backlog``, ``maintenance.FACT_FIRST``): while no log holds
that many frames past the baseline the row held for its store when a look last read it (kept in memory: a pass copies
a log without emptying it, so a quiet log keeps frames a pass already saw), it opens no store, so a quiet install,
or one whose stores keep a rollback journal, takes no store lock for it. The server's 20 s duty
``wal_checkpoint_look`` (``server_maintenance``) only asks the lane to look, and the lane looks again after each pass
that left the job done (``maintenance._look_again``), spawning while it is due. A store a pass finds held
(contention, or SQLite's ``busy``: another checkpointer held the log) keeps its last entry and so stays due; the pass
says ``pending``.

A pass records, per store kind (``store_timing.KINDS``), SQLite's three numbers (``busy``; ``log``: frames in the
log; ``checkpointed``: frames now in the database), the checkpoint's wall ``seconds`` and the ``wal`` it met
(``[generation, last committed frame]``), on the job's row and as one ``wal_checkpoint`` row of the store-timing
journal (``store_timing.JOURNAL``), not of ``logs/supervisor.jsonl``, whose tail every context reads (on a busy
install the lane passes one to three times a minute), and the lane's clean child runs write no ``maintenance`` rows
there either (``maintenance._unclean_only``). Each busy minute's ``supervisor_load`` row carries, beside its commit times, the
most committed frames the minute's looks saw in each store kind's log (``drain_frames``). Together they show whether
long COMMITs go on while the log stays short (an interactive COMMIT checkpoints inline only at the ceiling), whether
a reader holds the log back (``checkpointed`` below ``log``), and whether the log is ever reset (``log`` growing pass
after pass though each pass copied it all: a reader was on the log whenever a writer began). A log reset between two
looks, or a slow sync with no checkpoint, leaves a long COMMIT's cause unknown.

The worse worst case, disclosed: when the job is down, late, starved or held back by a reader, the log grows to the
ceiling and the next interactive COMMIT copies its frames inline, about 64 MiB (more past a large transaction or a
pinned reader, as above), more than any inline checkpoint before (about 16 times the old 4 MiB); a last close and
crash recovery meet the same log. The look rides the supervisor loop's tick: while that loop stalls, the lane looks
only after a pass of its own and stops at the first look that finds the job at rest, and at the night rates measured
on a busy install (3,000 to 5,000 committed frames a minute into the usage store) the ceiling is then 3 to 5 minutes
away. ``store.hand_off`` and the export seam still truncate the log.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import os
import pathlib
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Iterator, Tuple

from ouroboros import store_timing

JOURNAL_SIZE_LIMIT = 64 << 20  # bytes a store keeps of an idle rollback journal, or of its WAL log after a reset
_LOG_HEADER, _FRAME_HEADER = 32, 24  # the WAL log's header and each frame's header (sqlite.org/fileformat2.html §4)
NAME = "wal_checkpoint"  # the maintenance job and its row (``maintenance_jobs``)
_LOOKS: Dict[str, Any] = {"looks": 0, "max": {}}  # the looks since the last ``drain_frames`` and what they saw
_LOOKS_LOCK = threading.Lock()
_SEEN: Dict[str, Any] = {}  # store path -> its job-row entry's ``wal`` at the last look that read it (three per root)


def ceiling(page_size: int) -> int:
    """The frames of PAGE_SIZE bytes that fit in ``JOURNAL_SIZE_LIMIT`` after the log's header."""
    return (JOURNAL_SIZE_LIMIT - _LOG_HEADER) // (page_size + _FRAME_HEADER)


def limit(conn: sqlite3.Connection, path: Any) -> None:
    """``store.connect``'s journal limits on CONN to PATH: the bytes kept after a reset, and the inline ceiling of a
    WAL connection to a store the job passes (``serviced``); any other keeps SQLite's own threshold."""
    conn.execute(f"PRAGMA journal_size_limit = {JOURNAL_SIZE_LIMIT}")
    if conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal" and serviced(path):
        conn.execute(f"PRAGMA wal_autocheckpoint = {ceiling(conn.execute('PRAGMA page_size').fetchone()[0])}")


def serviced(path: Any) -> bool:
    """PATH is a store the job passes: a store kind (``store_timing.KINDS``) right in the ``state`` folder of the
    install this process acts as (``install_identity.canonical_root``), i.e. one of its ``store_journal.root_stores``.
    A child drive's store, headless child data's (``canonical_root`` is the install around that layout), a private
    build's temporary file or the evidence index is not; an install root that cannot be read leaves SQLite's own."""
    from ouroboros.install_identity import canonical_root
    from ouroboros.launch_roots import same_location

    path = pathlib.Path(path)
    if store_timing.kind_of(path) == "other" or path.parent.name != "state":
        return False
    try:
        return same_location(path.parent.parent, canonical_root())
    except (OSError, RuntimeError, ValueError):
        return False


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
    from ouroboros import store, store_journal

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


def backlog(root: Any) -> bool:
    """A log of ROOT holds ``inline_frames`` committed frames past the baseline its store's entry on the job's row held
    when a look in this process last read it (``_SEEN``; none known: all its committed frames): the look's first step
    (``maintenance.FACT_FIRST``), from the logs' headers alone, before the job's row is read. A pass only moves a
    baseline on, so a remembered one reads the row as often or more, never less; a log a writer reset, or a store file
    replaced, begins a generation no baseline matches. Each look notes the committed frames of every log it read, in
    memory (``drain_frames``)."""
    logs = [(kind, os.fspath(path), log) for kind, path, _tier, log in _logs(pathlib.Path(root))]
    with _LOOKS_LOCK:
        _LOOKS["looks"] += 1
        for kind, _path, log in logs:
            _LOOKS["max"][kind] = max(log[1], _LOOKS["max"].get(kind, 0))
    return any(unseen(log, _SEEN.get(path)) >= inline_frames() for _kind, path, log in logs)


def drain_frames() -> Dict[str, Any]:
    """The looks since the last drain and the most committed frames one of them saw in each store kind's log, then an
    empty window: the ``supervisor_load`` row's ``wal_frames`` (``server_liveness._load_sections``), so a busy minute's
    commit times stand beside how long the logs were. Memory only: no file is read on the caller's thread, the loop's.
    A kind whose log held no commit at any look is absent; ``looks`` 0 is a minute nobody looked (a stalled loop, a
    long pass): unknown, never an empty log."""
    with _LOOKS_LOCK:
        window = {"looks": _LOOKS["looks"], "max": dict(_LOOKS["max"])}
        _LOOKS["looks"], _LOOKS["max"] = 0, {}
    return window


def due(root: Any, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """A store's log holds ``inline_frames`` committed frames the job's last pass did not see (module docstring); the
    baseline the row holds for each store with a log is kept for the next look's ``backlog``."""
    stores, counts = row.get("stores") or {}, []
    for kind, path, _tier, log in _logs(pathlib.Path(root)):
        _SEEN[os.fspath(path)] = seen = (stores.get(kind) or {}).get("wal")
        counts.append(unseen(log, seen))
    return any(count >= inline_frames() for count in counts)


def run(root: Any, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """One PASSIVE checkpoint of each WAL store whose log holds a commit the last pass did not see, through the
    store's own connection (its tier, ``synchronous=FULL``); the row keeps per store the numbers and the log it saw.
    A store that cannot be opened or checkpointed keeps its error class there instead, at the log it met, so it is
    tried again when its log moves on, never on a clock. A store found held (contention, as ``maintenance._contention``
    reads it, or SQLite's ``busy``: another checkpointer held the log) is no fault: it keeps the entry its last pass
    left, so it stays due, and the pass says ``pending``. The pass's row, SQLite's numbers included, is written before
    the job's row, so a current store held past the wait (the job row's write refused) still leaves it."""
    from ouroboros import maintenance_jobs, store
    from ouroboros.utils import append_jsonl, utc_now_iso

    root = pathlib.Path(root)
    stores = dict((maintenance_jobs.job(root, NAME) or {}).get("stores") or {})  # the store's foreign-root check first
    passed: Dict[str, Dict[str, Any]] = {}
    held = set()
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
                if busy:  # another checkpointer (an inline one, a handoff) held the log: this pass copied nothing
                    held.add(kind)
        except (sqlite3.Error, store.StoreError, OSError) as exc:
            facts = {"error": type(exc).__name__}
            if isinstance(exc, store.StoreBusy) or (isinstance(exc, sqlite3.Error) and store.busy(exc)):
                held.add(kind)
        passed[kind] = {**facts, "wal": log}
        if kind not in held:
            stores[kind] = {**facts, "wal": log, "at": utc_now_iso()}
    state = "pending" if held else "done"
    if passed:  # the pass's own row (module docstring), before the job's row a held current store can refuse
        try:
            append_jsonl(root / "logs" / store_timing.JOURNAL, {"ts": utc_now_iso(), "type": NAME, "stores": passed})
        except OSError:
            logging.getLogger(__name__).warning("%s row not recorded", NAME, exc_info=True)
    if set(passed) - held:  # a held store alone changes no entry: nothing to write
        maintenance_jobs.record(root, NAME, state=state, stores=stores)
    return {"state": state, "items": len(passed)}
