"""Who chooses a store file's SQLite journal, and when (``state/current.sqlite``, ``state/usage.sqlite``,
``state/dialog.sqlite``, the evidence index).

``store._journal_mode`` is this runtime's eligibility: WAL only with a SQLite that carries the WAL-reset
fix (3.51.3, or the 3.50.7 and 3.44.6 backports; the bug is a race between two connections, in threads
or processes, writing or checkpointing the same WAL file at once: sqlite.org/wal.html §11), on the
enforced lock tier and a proven local filesystem; otherwise the rollback journal. The gate is a
compatibility fact about this runtime, not evidence that any file was damaged.

Three owners act on that choice, each at a lifecycle boundary:

* a private build (``store.connect(create=True)``) is created in it and published by rename with its
  header (WAL survives the rename; its log and index do not, ``store._StoreConnection.close``);
* the boot brings an existing file to it before its producers start (``configure_journal``): the
  main root's ``store_import`` boot preparation, ``usage_store.migrate_from_journal`` and the
  dialogue cutover, each with explicit ``initialize_journal``; the boot's child-header pass leaves
  incompatible WAL before lazy consumers open. Body stamps and supervisor revivals grant no authority;
* an export/switch seam, a lock-tier change or a boot leaving WAL makes a standalone DELETE file
  (``store.prepare_handoff``), under the tier the file is still used with before a new one is published.

Every other open (``open_store``, ``dialogue_store.transaction``, ``usage_store.hold``, the evidence
index; ``write=False`` too, which is still ``mode=rw`` and may write SQLite's own journal files) keeps
the file's WAL/rollback boundary (``store._kept_journal``): connection-local PERSIST on a rollback file,
WAL left as it is, and a WAL file this runtime must not share refused typed, never used or converted.
A helper using this core cannot switch the file into WAL; an incompatible app open refuses rather
than attempting a competing journal conversion.
"""

from __future__ import annotations

import logging
import pathlib
import sqlite3
import time

from ouroboros import store

log = logging.getLogger(__name__)


def configure_journal(path: pathlib.Path, tier: str, *, wait_sec: float = store.STORE_WAIT_SEC) -> bool:
    """The boot's choice for an existing file, before its producers; True when the file now has it.

    Leaving WAL is required compatibility: the file is handed off (``prepare_handoff``, ``seam=False``),
    retrying only contention within ``wait_sec``; exhaustion raises retryable ``StoreBusy`` and
    ordinary opens still refuse the file. No failed conversion publishes a new tier. Entering WAL is
    optional tuning: an early reader's lease, a writer that outlasts ``wait_sec`` or SQLite declining
    WAL defers it with a warning and False while the rollback journal keeps serving, until a later boot.
    A name-tier caller holds its name lock.
    """
    wanted = store._journal_mode(path, tier) == "WAL"
    if store.wal_file(path) == wanted:
        return True
    if not wanted:
        deadline = time.monotonic() + max(0.0, wait_sec)
        while True:
            try:
                store.prepare_handoff(path, tier, wait_sec=max(0.0, deadline - time.monotonic()), seam=False)
                return True
            except (store.StoreBusy, sqlite3.OperationalError) as exc:
                if isinstance(exc, store.StoreBusy):
                    if exc.reason != "contention":
                        raise
                elif not store.busy(exc):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise store.StoreBusy(f"Journal handoff unavailable: {path}", reason="contention") from exc
                time.sleep(min(0.05, remaining))
    try:
        store.close_connections(path)
        store.connect(path, tier, wait_sec=wait_sec, journal_mode="WAL").close()
    except (store.StoreBusy, sqlite3.OperationalError) as exc:
        log.warning("Journal selection for %s deferred; the rollback journal keeps serving: %s", path, exc)
        return False
    return True


def prepare_children(root: pathlib.Path) -> None:
    """Boot compatibility for existing child stores; no SQL or import for compatible/absent files.

    The custody walk uses drive directories only, without frozen result files. Only incompatible WAL
    is handed off; a lazy first build owns its journal, and ordinary opens gain no lifecycle authority.
    """
    import contextlib
    from ouroboros import dialogue_store, store_import, usage_store

    for drive in store_import.descendant_drives(root, scan_results=False):
        stores = ((store.store_path(drive), lambda: store.probe(drive)[1], store.LOCK_NAME),
                  (dialogue_store.path(drive), lambda: dialogue_store.header_tier(drive), dialogue_store.LOCK_NAME),
                  (drive / usage_store.STORE_REL, lambda: usage_store._header_tier(drive / usage_store.STORE_REL),
                   usage_store.LOCK_REL.name))
        for path, read_tier, lock in stores:
            if not store.wal_file(path):
                continue
            tier = read_tier()
            if tier and store._journal_mode(path, tier) != "WAL":
                with store.process_lock(path), (store.name_lock(drive, lock, timeout_sec=store.STORE_WAIT_SEC)
                        if tier == store.TIER_NAME else contextlib.nullcontext()):
                    configure_journal(path, tier)
