"""What each SQLite journal mode makes busy, and what a reader leaves behind, for store tests.

A store is built in WAL where the engine, the lock tier and the filesystem allow it, else in the
rollback journal (PERSIST; ``store._journal_mode``), and an ordinary ``store.connect`` keeps whichever
the file has (``ouroboros/store_journal.py``). The two modes block different things:

* a held write lock (``BEGIN IMMEDIATE``) makes another writer's acquisition busy in both;
* a held read makes a rollback-journal COMMIT wait for it to drain; a WAL COMMIT waits for no reader;
* ``BEGIN EXCLUSIVE`` makes every rollback-journal reader busy too, while under WAL it is only the
  write lock: a WAL reader is busy only while another connection holds the database file itself
  exclusively, as one in exclusive locking mode, the last close's checkpoint or a recovery do
  (sqlite.org/wal.html, "Sometimes Queries Return SQLITE_BUSY In WAL Mode").

A WAL reader also writes SQLite's own bookkeeping, none of it a commit: the ``-shm`` index on every
read, the ``-wal``/``-shm`` pair its first open creates and the last close removes (that close's
checkpoint also moves the log's frames into the file); the header's journal-format fields change only
when a build or lifecycle owner opens a file into WAL (sqlite.org/fileformat2.html). A store held open
across a window (``held_open``) keeps all of that outside it.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import pathlib
import sqlite3
import time
from typing import Any, Dict, Iterator, Optional, Tuple

from ouroboros import store

# A database's own journal files: what they hold is the database's committed content.
_JOURNAL_SUFFIXES = ("-wal", "-shm", "-journal")
# The header fields a rollback-journal file's first WAL open rewrites (sqlite.org/fileformat2.html
# §1.3): the format versions (18-19), the change counter (24-27), the version-valid-for number
# (92-95) and the version of the SQLite that wrote the file last (96-99).
_WAL_OPEN_FIELDS = (slice(18, 20), slice(24, 28), slice(92, 100))


def use_journal(monkeypatch, mode: str) -> None:
    """Every store this test builds from now on is created in MODE ("PERSIST" or "WAL"), whatever
    this interpreter's SQLite would choose: for a subject that exists in one mode only (a reader holding
    a COMMIT is the rollback journal's), or a test that compares the two. An ordinary open never moves a
    file's WAL/rollback boundary, so the stores this process already holds idle handles to are brought
    to MODE the way a boot owner does (``store_journal.configure_journal``), once those handles close; a
    lease an earlier test's late thread still holds on its own store stays with that thread."""
    from ouroboros.store_journal import configure_journal

    with store._CONNECTION_GUARD:
        idle = {(held.path, held.tier) for pool in list(store._CONNECTION_POOLS)
                for held in pool.entries.values() if not held.active}
    monkeypatch.setattr(store, "_journal_mode", lambda *_args: mode)
    store._close_cached(lambda _name: True, idle_only=True)
    for path, tier in idle:
        if path.is_file():
            configure_journal(path, tier)
            _CONVERTED.append((path, tier))


# The stores ``use_journal`` brought to a forced mode, given this interpreter's own journal back once the test's
# monkeypatch is undone (tests/conftest.py): a WAL file never outlives its test on a SQLite that must not share WAL.
_CONVERTED: list = []


def give_back_converted_journals() -> None:
    """A store an earlier test's late thread still leases keeps its forced mode until a later teardown:
    the lease is that thread's (``use_journal``), and the test ending now broke nothing."""
    from ouroboros.store_journal import configure_journal

    waiting = []
    while _CONVERTED:
        path, tier = _CONVERTED.pop()
        if not path.is_file():
            continue
        key = str(pathlib.Path(os.path.abspath(path)))
        with store._CONNECTION_GUARD:
            leased = any(entry.active for pool in list(store._CONNECTION_POOLS)
                         for name, entry in pool.entries.items() if name == key)
        try:
            if not leased:
                store.close_connections(path)
                configure_journal(path, tier, wait_sec=0)  # a busy file waits for a later teardown, not here
                continue
        except store.StoreBusy:  # a lease taken since, or another process holding the file at the handoff
            pass
        waiting.append((path, tier))
    _CONVERTED.extend(waiting)


def exclusive_hold(path: Any, *, wait_sec: float = 5.0) -> sqlite3.Connection:
    """A connection inside a transaction under which every other connection's read and write of
    PATH is busy until it rolls back and closes (the caller does both): ``BEGIN EXCLUSIVE`` on a
    rollback-journal file; on a WAL file the same in exclusive locking mode, which takes the file's
    exclusive lock, once this process's idle handles in the folder closed (each WAL handle keeps a
    shared lock on the file while it is open). The file's journal mode is left as it is.
    ``wait_sec`` waits out other holders (5 s, ``sqlite3.connect``'s own default); any thread may
    end the hold."""
    path = pathlib.Path(path)
    if not store.wal_file(path):
        conn = store.connect(path, store.TIER_ENFORCED, wait_sec=wait_sec, shared=True, journal_mode="")
        return _begin(conn, "BEGIN EXCLUSIVE")
    deadline = time.monotonic() + wait_sec
    while True:
        store.close_connections_under(path.parent, idle_only=True)  # a leased handle stays: retried below
        conn = store.connect(path, store.TIER_ENFORCED, shared=True, journal_mode="")
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")  # before this connection's first read
        try:
            return _begin(conn, "BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            if not store.busy(exc) or time.monotonic() >= deadline:
                raise
        time.sleep(0.01)


def read_hold(path: Any, *, wait_sec: float = 5.0) -> sqlite3.Connection:
    """A connection inside a read transaction on PATH: on a rollback journal it holds the next
    COMMIT (which waits for readers to drain); under WAL it blocks nothing."""
    return _begin(store.connect(pathlib.Path(path), store.TIER_ENFORCED, wait_sec=wait_sec), "BEGIN")


def write_hold(path: Any, *, wait_sec: float = 5.0) -> sqlite3.Connection:
    """A connection inside a transaction that makes the next write of PATH busy, the way the file's
    journal mode contends a write, while PATH stays readable: a held read on a rollback journal (the
    writer's COMMIT waits for it to drain), the held write lock under WAL (whose COMMIT waits for no
    reader, so only the next writer's acquisition can be contended)."""
    conn = store.connect(pathlib.Path(path), store.TIER_ENFORCED, wait_sec=wait_sec)
    wal = conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    return _begin(conn, "BEGIN IMMEDIATE" if wal else "BEGIN")


def _begin(conn: sqlite3.Connection, statement: str) -> sqlite3.Connection:
    try:
        conn.execute(statement)
        conn.execute("SELECT 1 FROM meta LIMIT 1").fetchone()  # the transaction's lock, taken now
    except BaseException:
        conn.close()
        raise
    return conn


def committed_content(path: Any) -> Tuple[Any, ...]:
    """A SQLite file's committed content whatever its journal mode: its application id, user
    version, schema and rows, read through an independent connection (the WAL log's frames
    included). Equal content says the rows are as they were, not that nothing wrote: a VACUUM
    keeps it (``StoreState`` adds the write's own trace). ``iterdump`` lists the tables by name,
    each with its rows in rowid order, the order obligation sets and handoffs read back in, so a
    rewrite of the same rows in another order still differs; only the index, trigger and view
    statements are compared sorted: they follow in schema-row order, which a VACUUM rebuilds."""
    with contextlib.closing(sqlite3.connect(pathlib.Path(path), isolation_level=None)) as conn:
        header = tuple(conn.execute(f"PRAGMA {name}").fetchone()[0] for name in ("application_id", "user_version"))
        schema = {f"{sql};" for (sql,) in conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql NOT NULL AND type IN ('index', 'trigger', 'view')")}
        dump = tuple(conn.iterdump())
        ordered = tuple(line for line in dump if line not in schema)
        return header + ordered + tuple(sorted(line for line in dump if line in schema))


@dataclasses.dataclass(frozen=True, eq=False)
class StoreState:
    """A SQLite file as a write changes it. ``content`` (``committed_content``) says what the rows
    are, never that nothing was written: a VACUUM keeps it. The write's own trace after the
    journal's set-up is the rest: the commits the WAL log holds (``store.log_identity``), the file's
    bytes but for the header fields a first WAL open rewrites (``masked``), and those fields too
    while the file keeps a rollback journal, whose every commit moves the change counter."""

    content: Tuple[Any, ...]
    log: Any
    masked: bytes
    rollback_fields: Optional[bytes]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StoreState):
            return NotImplemented
        # The header fields count while both sides keep a rollback journal: a first WAL open rewrites them.
        held = (self.rollback_fields, other.rollback_fields)
        fields = None in held or held[0] == held[1]
        return fields and (self.content, self.log, self.masked) == (other.content, other.log, other.masked)

    def __repr__(self) -> str:
        digest = hashlib.sha256(self.masked).hexdigest()[:12]
        return (f"StoreState(content: {len(self.content)} lines, log={self.log}, masked sha256 {digest}, "
                f"rollback fields {self.rollback_fields.hex() if self.rollback_fields is not None else None})")


def store_state(path: Any) -> StoreState:
    """``StoreState`` of the SQLite file at PATH, its bytes read through the store's borrowed
    descriptor (closing a descriptor of our own would drop this process's locks on the file)."""
    raw = store.read_file_bytes(path)
    masked = bytearray(raw)
    for part in _WAL_OPEN_FIELDS:
        masked[part] = bytes(part.stop - part.start)
    rollback = raw[18:20] != b"\x02\x02"  # versions 2 mean WAL
    return StoreState(committed_content(path), store.log_identity(path), bytes(masked),
                      b"".join(raw[part] for part in _WAL_OPEN_FIELDS) if rollback else None)


@contextlib.contextmanager
def held_open(path: Any) -> Iterator[None]:
    """The store at PATH open for the block, in the journal mode the store chooses and read once:
    its journal's set-up falls before the block (a first WAL open rewrites the header, a first read
    creates the ``-wal``/``-shm`` pair), and within it no last close removes the pair or checkpoints
    the log into the file. The file's and its folder's mtimes then move only with a write, in
    either journal mode (``tree_state(mtimes=True)``)."""
    with contextlib.closing(store.connect(pathlib.Path(path), store.TIER_ENFORCED)) as conn:
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()  # the first read; no transaction stays
        yield


def tree_state(root: Any, *, mtimes: bool = False, directories: bool = True) -> Dict[str, Any]:
    """Every path under ROOT with what a write changes, keyed by its relative POSIX name: a file's
    bytes, a SQLite database's ``store_state``, a directory as ``None``; with ``mtimes`` each one's
    mtime too, which under WAL holds across a window only while the store stays open (``held_open``).
    A database's own journal files are left out by name, never opened: they come and go with
    readers, their frames are in its state, and closing a descriptor of a live ``-shm`` would drop
    this process's locks on it (``tests/test_store_wal_index.py``)."""
    root = pathlib.Path(root)
    state: Dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.name.endswith(_JOURNAL_SUFFIXES) and store.is_sqlite_file(path.with_name(path.name.rsplit("-", 1)[0])):
            continue
        stamp = path.stat().st_mtime_ns if mtimes else None
        if path.is_dir():
            if directories:
                state[name] = stamp
        elif store.is_sqlite_file(path):
            state[name] = (store_state(path), stamp)
        else:
            state[name] = (path.read_bytes(), stamp)
    return state
