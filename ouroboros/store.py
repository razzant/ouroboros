"""Current facts in ``state/current.sqlite`` and the SQLite protocol shared with ``usage_store``.
PERSIST (WAL only on a fixed SQLite and proven local filesystem; only a private build or a lifecycle
owner chooses, an ordinary open keeps the file's choice: ``store_journal``), ``synchronous=FULL``; a lazy
connection per file and thread, short ``BEGIN IMMEDIATE`` writes, every name-tier access under its name
lock. SQLite 3.24 floor. ``probe``/``open_store``: boot imports within the import window; first writes
import absent stores; reads never import (facts present: ``StoreMigrating`` until publication; no facts:
empty). The write gate requires a closed import window and PUBLISHED phase; ``service`` serves the
importer/exporter/seam."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import errno
import json
import os
import pathlib
import re
import shutil
import sqlite3
import stat
import sys
import threading
import time
import weakref
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ouroboros.utils import append_jsonl, machine_json, utc_now_iso  # one compact JSON text: key order kept, facts round-trip
from ouroboros import store_empty, store_hot_facts, store_task_cards, store_timing
from ouroboros.swarm_rollup import retain_swarm_basis

STORE_REL = pathlib.Path("state/current.sqlite")
LOCK_NAME = "current.lock"  # the name-tier lock around every access
IMPORT_LOCK_NAME = "current_import.lock"
GENERATION = 8  # the lifecycle generation that owns the store
GENERATION_KEY = "current_store_generation"  # in state/migrations.json
SCHEMA_VERSION = 9  # 6: hot facts; 7: task cards; 8: root windows, terminal_seq; 9: filtered-list selection
STORE_WAIT_SEC = 10.0
JOURNAL_SIZE_LIMIT = 64 << 20  # retain at most 64 MiB of an idle rollback journal (PERSISTENCE.md: what it bounds)
TIER_ENFORCED, TIER_NAME = "enforced", "name"
PUBLISHED, EXPORTING, RETIRED = "published", "exporting", "retired"
# The header's application id names the lock protocol before the file is opened ('OUCE'/'OUCN').
_APPLICATION_ID = {TIER_ENFORCED: 0x4F554345, TIER_NAME: 0x4F55434E}
TYPED_COLUMNS = ("status", "parent_task_id", "root_task_id", "delegation_role", "superseded_by",
                 "retry_task_id", "project_id", "ts", "updated_at", "child_drive_root",
                 "headless_child_drive_root", "drive_root", "presence_binding_id")
# set name -> flag column; the writer decides ``pending_drive`` (nonterminal, or terminal files not ready).
# A column added after schema 1 is nullable: a whole-row write by an older body (its own column list)
# leaves NULL, "not derived", which the next boot derives (``store_import._derive_flag_columns``).
FLAG_COLUMNS = {"synthesis": "owes_synthesis", "terminal_projection": "owes_terminal_projection",
                "promotions": "has_pending_promotion", "pause_notices": "has_pause_notices",
                "delegated_runs": "delegated_runs_unreconciled", "pending_drives": "pending_drive",
                "owner_questions": "has_owner_questions"}
# The per-operation projections of a document that live in ``task_handoffs`` rows (``kind`` = the
# field): a claim, a wait or a retirement writes its one row, never the body. The body keeps a
# present key as ``{}`` (its place and presence); ``load`` puts the rows back there, so a document
# round-trips byte for byte. A key the body never had appears while it has rows.
ROW_BACKED_FIELDS: Tuple[str, ...] = ("launch_handoffs", "model_waits", "retired_model_consumers", "swarm_fanouts")
TERMINAL_WRITES = "terminal_writes"  # meta: terminal row writes (``obligations.task_written``, which stamps ``terminal_seq``)
TERMINAL_SEQ_INDEX = "CREATE INDEX IF NOT EXISTS tasks_terminal_seq ON tasks(terminal_seq) WHERE terminal_seq IS NOT NULL;"
REVIEW_COLLECTION_INDEX = ("CREATE INDEX IF NOT EXISTS tasks_review_collection ON tasks(review_collection_pending, task_id) "
                           "WHERE review_collection_pending IS NULL OR review_collection_pending=1;")
PLAN_COLLECTION_INDEX = ("CREATE INDEX IF NOT EXISTS tasks_plan_collection ON tasks(plan_collection_pending, task_id) "
                         "WHERE plan_collection_pending IS NULL OR plan_collection_pending=1;")
ROOT_RESULT_WHERE = "parent_task_id IS NULL AND (delegation_role IS NULL OR delegation_role != 'subagent')"
ROOT_WINDOWS = tuple(f"CREATE INDEX IF NOT EXISTS {name} ON tasks({key}) WHERE {ROOT_RESULT_WHERE};\n" for name, key in
                     (("tasks_root_ts", "ts, task_id"), ("tasks_project_root_ts", "project_id, ts, task_id")))
_SCHEMA = """
CREATE TABLE tasks (
  task_id TEXT PRIMARY KEY, status TEXT, parent_task_id TEXT, root_task_id TEXT, delegation_role TEXT,
  superseded_by TEXT, retry_task_id TEXT, project_id TEXT, ts TEXT, updated_at TEXT,
  child_drive_root TEXT, headless_child_drive_root TEXT, drive_root TEXT, presence_binding_id TEXT,
  presence_authority_recorded INTEGER NOT NULL DEFAULT 0, write_seq INTEGER NOT NULL DEFAULT 0,
  owes_synthesis INTEGER NOT NULL DEFAULT 0, owes_terminal_projection INTEGER NOT NULL DEFAULT 0,
  has_pending_promotion INTEGER NOT NULL DEFAULT 0, has_pause_notices INTEGER NOT NULL DEFAULT 0,
  delegated_runs_unreconciled INTEGER NOT NULL DEFAULT 0, pending_drive INTEGER NOT NULL DEFAULT 0,
  has_owner_questions INTEGER,
  review_collection_pending INTEGER, plan_collection_pending INTEGER, drive_retention_at TEXT, terminal_seq INTEGER);
CREATE INDEX tasks_parent ON tasks(parent_task_id); CREATE INDEX tasks_root ON tasks(root_task_id);
CREATE INDEX tasks_status ON tasks(status); CREATE INDEX tasks_project ON tasks(project_id);
CREATE INDEX tasks_ts ON tasks(ts, task_id); CREATE INDEX tasks_updated ON tasks(updated_at, task_id);
CREATE INDEX tasks_superseded ON tasks(superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX tasks_retry ON tasks(retry_task_id) WHERE retry_task_id IS NOT NULL;
CREATE TABLE task_bodies (task_id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE obligations (set_name TEXT NOT NULL, id TEXT NOT NULL, facts TEXT NOT NULL, PRIMARY KEY (set_name, id));
CREATE TABLE task_handoffs (task_id TEXT NOT NULL, op_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT,
  claimed_at TEXT, returned_at TEXT, owner TEXT, facts TEXT NOT NULL, PRIMARY KEY (task_id, op_id));
CREATE TABLE custody_runs (run_id TEXT PRIMARY KEY, task_id TEXT, project_id TEXT, invocation_id TEXT, state TEXT,
  facts TEXT NOT NULL, updated_at TEXT);
CREATE INDEX custody_task ON custody_runs(task_id); CREATE INDEX custody_state ON custody_runs(state);
CREATE INDEX custody_project ON custody_runs(project_id) WHERE project_id<>'';
CREATE INDEX custody_invocation ON custody_runs(invocation_id) WHERE invocation_id<>'';
CREATE TABLE custody_outbox (event_id TEXT PRIMARY KEY, run_id TEXT, task_id TEXT, event TEXT NOT NULL, created_at TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
""" + REVIEW_COLLECTION_INDEX + PLAN_COLLECTION_INDEX + TERMINAL_SEQ_INDEX + "".join(f"CREATE INDEX tasks_{column} ON tasks({column}) WHERE {column}=1;\n"
    for column in FLAG_COLUMNS.values()) + "".join(ROOT_WINDOWS) + store_task_cards.SCHEMA + store_hot_facts.SCHEMA


class StoreError(RuntimeError):
    """The current-facts store cannot serve; the subclass says why."""


class StoreMigrating(StoreError):
    """Not published yet: the importer runs, or never ran on this root."""


class StoreRetired(StoreError):
    """The export began or completed: the files are the authority now."""


class StoreLost(StoreError):
    """The store is missing beside its completed stamp, or unreadable; never replaced silently."""


class StoreBusy(StoreError):
    """Contention or exhausted process resources: retry; never a new store."""

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason


_WAIT_BUDGET = contextvars.ContextVar("current_store_wait_budget", default=None)
_ADMISSION_NO_WAIT = contextvars.ContextVar("current_store_admission_no_wait", default=False)


@contextlib.contextmanager
def acquisition_budget(seconds: float):
    """Narrow nested store acquisitions; zero makes an interlocked probe nonblocking."""
    prior = _WAIT_BUDGET.get()
    token = _WAIT_BUDGET.set(min(prior, seconds) if prior is not None else seconds)
    try:
        yield
    finally:
        _WAIT_BUDGET.reset(token)


def bounded_wait(wait_sec: float) -> float:
    """The wait one acquisition may spend: none under ``admission_acquisition``, else narrowed by ``acquisition_budget``."""
    if _ADMISSION_NO_WAIT.get():
        return 0.0
    budget = _WAIT_BUDGET.get()
    return wait_sec if budget is None else min(wait_sec, budget)


@contextlib.contextmanager
def admission_acquisition():
    """Probe/claim without waiting; this temporary guard is not the body's IO policy."""
    token = _ADMISSION_NO_WAIT.set(True)
    try:
        yield
    finally:
        _ADMISSION_NO_WAIT.reset(token)


@contextlib.contextmanager
def physical_acquisition():
    """At accepted handoff, copy only the caller's physical budget into its body.

    Other ContextVars and explicit acquisition_budget scopes remain unchanged;
    a nested physical start must acquire its own fresh admission guard."""
    token = _ADMISSION_NO_WAIT.set(False)
    try:
        yield
    finally:
        _ADMISSION_NO_WAIT.reset(token)


_PROCESS_LOCKS: Dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def process_lock(path: pathlib.Path) -> threading.RLock:
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(str(path), threading.RLock())


def file_identity(path: pathlib.Path) -> Optional[tuple]:
    """``(inode, device)`` of a present file; ``None`` only for a missing one."""
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return None
    return info.st_ino, info.st_dev


class _Inode:
    """One file's open uses in this process: SQLite connections from ``connect`` and raw readers,
    with the unbuffered handles the readers opened (``files[0]`` serves every read)."""

    __slots__ = ("conns", "readers", "files")

    def __init__(self) -> None:
        self.conns, self.readers, self.files = 0, 0, []


_INODES: Dict[Tuple[int, int], _Inode] = {}
_TIERS: Dict[tuple, Optional[str]] = {}  # (path, identity) -> its header's tier; a retier in place sets None: read each time
# Reentrant: a reader generator finalized by the cyclic GC returns its borrow on whatever thread
# triggered the collection, possibly inside this thread's own guarded section.
_INODE_GUARD = threading.RLock()


def _settle(key: Tuple[int, int], entry: _Inode) -> None:
    """Close the readers' descriptors once nothing in this process uses the inode (guard held):
    with no connection open, no lock of this process on the file can be released by a close."""
    if entry.conns or entry.readers:
        return
    for handle in entry.files:
        with contextlib.suppress(OSError):
            handle.close()
    _INODES.pop(key, None)


def _borrow(path: Any, dir_fd: Optional[int] = None) -> Tuple[Tuple[int, int], int]:
    """A read descriptor of PATH's file (POSIX): on POSIX closing ANY descriptor of a file releases
    every advisory lock this process holds on it, an open transaction's on another thread included,
    and another process then writes the same pages (sqlite.org howtocorrupt §2.2). So a raw read
    shares one descriptor per inode, and ``_return`` closes it only when no store connection of this
    process has the file open (SQLite's unix VFS defers its own closes the same way). With ``dir_fd``
    PATH is one entry of that held directory (``confined_files``), stated and opened relative to it,
    never through a link."""
    key = _identity_key(os.stat(path, dir_fd=dir_fd, follow_symlinks=dir_fd is None))
    with _INODE_GUARD:
        entry = _INODES.get(key)
        if entry is not None and entry.files:
            entry.readers += 1
            return key, entry.files[0].fileno()
    if dir_fd is None:
        handle = pathlib.Path(path).open("rb", buffering=0)  # outside the guard: a slow open stalls only this caller
    else:
        from ouroboros.confined_files import FILE_FLAGS  # no-follow; a FIFO swapped in never blocks
        handle = open(os.open(path, FILE_FLAGS, dir_fd=dir_fd), "rb", buffering=0)
    key = _identity_key(os.fstat(handle.fileno()))  # the path may name a newer file than the stat did
    with _INODE_GUARD:
        entry = _INODES.setdefault(key, _Inode())
        entry.files.append(handle)  # never closed early: a second handle waits for the settle too
        entry.readers += 1
        return key, entry.files[0].fileno()


def _return(key: Tuple[int, int]) -> None:
    with _INODE_GUARD:
        entry = _INODES.get(key)
        if entry is not None:
            entry.readers -= 1
            _settle(key, entry)


def _identity_key(info: os.stat_result) -> Tuple[int, int]:
    return info.st_dev, info.st_ino


class _StoreConnection(sqlite3.Connection):
    """A ``connect`` connection, counted on its file's inode until it closes (see ``_borrow``)."""

    _store_key: Optional[Tuple[int, int]] = None
    _store_build = False

    def close(self) -> None:
        if self._store_build:
            # A private build is published by renaming ONLY the database, keeping its chosen journal (a WAL header):
            # complete or not, no journal, log or index stays under its temporary name (this sole close removes them).
            if self.in_transaction:
                self.execute("ROLLBACK")
            _delete_journal(self, keep_wal=True)
            self._store_build = False
        super().close()  # a close that raises leaves the connection counted: its lock may still be held
        key, self._store_key = self._store_key, None
        if key is not None:
            with _INODE_GUARD:
                entry = _INODES.get(key)
                if entry is not None:
                    entry.conns -= 1
                    _settle(key, entry)


def counted_connect(target: str, path: Any, **kwargs: Any) -> sqlite3.Connection:
    """``sqlite3.connect`` counted on PATH's inode before any statement can take a lock (``_borrow``)."""
    conn = sqlite3.connect(target, factory=_StoreConnection, **kwargs)
    if os.name != "nt":
        try:
            key = _identity_key(os.stat(path))
        except OSError:
            conn.close()
            raise
        with _INODE_GUARD:
            _INODES.setdefault(key, _Inode()).conns += 1
            conn._store_key = key
    return conn


def _read_header(path: pathlib.Path, dir_fd: Optional[int] = None) -> bytes:
    """The first 100 bytes, read through the inode registry (``_borrow``); Windows locks belong
    to their handle, and a held handle there would block the export rename."""
    if os.name == "nt":
        with open(path, "rb") as handle:
            return handle.read(100)
    key, fd = _borrow(path, dir_fd)
    try:
        return os.pread(fd, 100, 0)
    finally:
        _return(key)


def held_descriptor(path: Any) -> Optional[int]:
    """The descriptor this process's open uses of PATH's file share right now, else ``None``."""
    if os.name == "nt":
        return None
    try:
        key = _identity_key(os.stat(path))
    except OSError:
        return None
    with _INODE_GUARD:
        entry = _INODES.get(key)
        return entry.files[0].fileno() if entry is not None and entry.files else None


def is_sqlite_file(path: Any, *, dir_fd: Optional[int] = None) -> bool:
    """Whether PATH holds a SQLite database (its header magic), read without closing a descriptor
    a store connection may depend on; a WAL index is none, and is never opened to look. ``dir_fd``:
    PATH is one entry of that held directory, and a link there is none (``_borrow``)."""
    try:
        if (not stat.S_ISREG(os.stat(path, dir_fd=dir_fd, follow_symlinks=dir_fd is None).st_mode)
                or is_wal_index(path, dir_fd=dir_fd)):
            return False  # never open a FIFO or a device to look, nor a WAL index (``is_wal_index``)
        return _read_header(pathlib.Path(path), dir_fd).startswith(b"SQLite format 3\x00")
    except OSError:
        return False


class _BorrowedReader:
    """A binary reader over a borrowed descriptor (``_borrow``); closing returns the borrow."""

    def __init__(self, path: Any, dir_fd: Optional[int] = None) -> None:
        self._key, self._fd = _borrow(path, dir_fd)
        self._offset, self._open = 0, True

    def __enter__(self) -> "_BorrowedReader":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._open:
            self._open = False
            _return(self._key)

    def _check(self) -> None:
        if not self._open:  # the descriptor may be closed or reused by another file by now
            raise ValueError("I/O operation on closed file")

    def fileno(self) -> int:
        self._check()
        return self._fd

    def tell(self) -> int:
        self._check()
        return self._offset

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._check()
        if whence not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END):
            raise ValueError(f"invalid whence ({whence})")
        base = os.fstat(self._fd).st_size if whence == os.SEEK_END else (0 if whence == os.SEEK_SET else self._offset)
        if base + offset < 0:
            raise ValueError("negative seek position")
        self._offset = base + offset
        return self._offset

    def read(self, size: Optional[int] = -1) -> bytes:
        """Up to ``size`` bytes (all that remain when negative); short ``pread``s are continued."""
        self._check()
        wanted, parts = (-1 if size is None else size), []
        while wanted:
            data = os.pread(self._fd, wanted if wanted > 0 else 1 << 20, self._offset)
            if not data:
                break
            parts.append(data)
            self._offset += len(data)
            wanted = wanted - len(data) if wanted > 0 else wanted
        return b"".join(parts)


class WalIndexRefused(PermissionError):
    """A raw read of a SQLite WAL index (``is_wal_index``), refused before any descriptor opens."""


def is_wal_index(path: Any, *, dir_fd: Optional[int] = None) -> bool:
    """Whether PATH is a SQLite WAL index: a regular file (nothing else ever is, and nothing is read
    before that is known) whose real name (a link stands for the file it names) is its database's name
    plus ``-shm``, folded as a case-insensitive volume folds it, where the database, a sibling in that
    real directory, is no link and is SQLite by its header (read only when it is no index itself: the
    same rule on a strictly shorter name, so the question always ends), and the volume resolves
    ``<database>-shm`` to PATH's own file, so on a case-sensitive volume ``x.db-SHM`` stays the user's.
    Every WAL connection holds locks on the index (POSIX record locks, which closing any descriptor of
    the file drops; Windows byte-range locks, which fail a read across them), and its bytes are shared
    memory a copy rebuilds from the log: the store's raw readers (``open_for_read`` and what reads
    through it, ``copy_file_bytes``) refuse it, and the copies and captures that apply this rule leave
    it out, tracked or not. The rule assumes this process's own databases never end in ``-shm`` (they
    are ``current.sqlite``, ``usage.sqlite``, ``dialog.sqlite``, ``index.sqlite3``, ``index.sqlite`` and
    temporaries ending in hex): beside a database ``x``, a database ``x-shm`` reads as ``x``'s index and
    its own index ``x-shm-shm`` as a file, whose raw read drops only this process's locks, and this
    process holds none on ``x-shm``. The ``-wal`` log carries no lock and reads as any file does.
    ``dir_fd`` (POSIX) asks it of one entry of that held directory (``confined_files``): every name is
    stated relative to it and none is followed, so a link there is no index and names no database."""
    path = (pathlib.PurePath(path) if dir_fd is not None
            else pathlib.Path(os.path.realpath(path) if os.path.islink(path) else path))
    # No database is named ``.`` or ``..`` (``with_name(".")`` would raise).
    if len(path.name) <= 4 or path.name[-4:].casefold() != "-shm" or path.name[:-4] in {".", ".."}:
        return False
    database = path.with_name(path.name[:-4])
    try:
        if dir_fd is not None:  # the header is read last, only for the entry ``<database>-shm`` names
            own = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
            named = os.stat(database.with_name(database.name + "-shm"), dir_fd=dir_fd, follow_symlinks=False)
            return (stat.S_ISREG(own.st_mode) and os.path.samestat(own, named)
                    and is_sqlite_file(database, dir_fd=dir_fd))
        return (stat.S_ISREG(os.stat(path).st_mode) and not os.path.islink(database) and is_sqlite_file(database)
                and os.path.samefile(path, database.with_name(database.name + "-shm")))
    except OSError:
        return False


def refuse_wal_index(path: Any, *, dir_fd: Optional[int] = None) -> None:
    """Raise ``WalIndexRefused`` for a WAL index, before anything opens it."""
    if is_wal_index(path, dir_fd=dir_fd):
        raise WalIndexRefused(errno.EACCES, "a SQLite WAL index is not read as a file (its bytes are shared "
                              "memory; the database's content is in the database and its log)", str(path))


def open_for_read(path: Any, *, dir_fd: Optional[int] = None) -> Any:
    """``open(path, "rb")`` for any reader in a process that may hold a store: on POSIX the read
    borrows the inode's shared descriptor, whose close waits until no store connection of this
    process has the file open; Windows opens and closes as usual. A WAL index is refused. ``dir_fd``
    (POSIX) reads one entry of that held directory, never through a link."""
    refuse_wal_index(path, dir_fd=dir_fd)
    return pathlib.Path(path).open("rb") if os.name == "nt" else _BorrowedReader(path, dir_fd)


def iter_file_bytes(path: Any, *, limit: Optional[int] = None, chunk: int = 1 << 20) -> Iterator[bytes]:
    """A file's bytes (the first ``limit``, or all), read through ``open_for_read``."""
    with open_for_read(path) as handle:
        while limit is None or limit > 0:
            data = handle.read(chunk if limit is None else min(chunk, limit))
            if not data:
                return
            if limit is not None:
                limit -= len(data)
            yield data


def read_file_bytes(path: Any, *, limit: Optional[int] = None) -> bytes:
    return b"".join(iter_file_bytes(path, limit=limit))


def copy_file_bytes(source: Any, destination: Any) -> Any:
    """``shutil.copy2`` that reads a SQLite file through ``open_for_read`` (a ``copytree``
    ``copy_function``) and refuses a WAL index; the copy is a byte copy, not a consistent snapshot
    of a live store."""
    refuse_wal_index(source)
    if os.name == "nt" or not is_sqlite_file(source):
        return shutil.copy2(source, destination)
    target = pathlib.Path(destination)
    if target.is_dir():
        target = target / pathlib.Path(source).name
    if target.exists() and os.path.samefile(source, target):
        raise shutil.SameFileError(f"{source!r} and {str(target)!r} are the same file")
    with open(target, "wb") as handle:
        for data in iter_file_bytes(source):
            handle.write(data)
    shutil.copystat(source, target)
    return str(target)


def header_tier(path: pathlib.Path, application_ids: Dict[str, int], identity: Optional[tuple] = None) -> Optional[str]:
    """The lock tier the file's SQLite header names; ``None`` for a missing or foreign file; read once per ``identity``."""
    if tier := _TIERS.get((str(path), identity)):
        return tier
    try:
        header = _read_header(path)
    except FileNotFoundError:
        return None
    if len(header) < 100 or not header.startswith(b"SQLite format 3\x00"):
        return None
    application_id = int.from_bytes(header[68:72], "big")
    tier = next((tier for tier, value in application_ids.items() if value == application_id), None)
    return (_TIERS.setdefault((str(path), identity), tier) or tier) if tier and identity is not None else tier


_LOCAL_FILESYSTEMS = frozenset({"apfs", "hfs", "ufs", "ext2", "ext3", "ext4", "xfs", "btrfs",
                               "f2fs", "jfs", "reiserfs", "zfs", "bcachefs", "tmpfs", "ramfs",
                               "overlay-local", "fixed"})


def _linux_filesystem(path: pathlib.Path, mountinfo: str) -> Optional[str]:
    """The containing mount, including bind mounts. Overlay is local only when ALL
    backing directories are visible and themselves on known local mounts."""
    def unescape(text: str) -> str:
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), text)

    mounts = []
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            mounts.append((pathlib.Path(unescape(fields[4])), fields[separator + 1], fields[separator + 3]))
        except (ValueError, IndexError):
            return None

    def containing(target: pathlib.Path, seen: frozenset) -> Optional[str]:
        candidates = [row for row in mounts if target.is_relative_to(row[0])]
        if not candidates:
            return None
        mount, kind, options = max(candidates, key=lambda row: len(row[0].parts))
        if kind != "overlay":
            return kind
        if mount in seen:
            return None  # host paths hidden by this namespace cannot prove local backing
        backing = []
        for option in options.split(","):
            key, _, value = option.partition("=")
            if key in {"upperdir", "lowerdir", "workdir"}:
                backing.extend(pathlib.Path(unescape(part)) for part in value.split(":"))
        if not backing or any(not part.is_absolute() or not part.is_dir() for part in backing):
            return None
        if all(containing(part.resolve(), seen | {mount}) in _LOCAL_FILESYSTEMS for part in backing):
            return "overlay-local"
        return None

    return containing(path.resolve(), frozenset())


def _bsd_filesystem(path: pathlib.Path) -> Optional[str]:
    """Read f_fstypename using the native Darwin/FreeBSD statfs ABI; other BSDs are unknown."""
    import ctypes

    u32, u64 = ctypes.c_uint32, ctypes.c_uint64
    if sys.platform == "darwin":
        fields = [("bsize", u32), ("iosize", ctypes.c_int32)] + [
            (name, u64) for name in ("blocks", "bfree", "bavail", "files", "ffree")]
        fields += [("fsid", ctypes.c_int32 * 2), ("owner", u32), ("type", u32), ("flags", u32), ("subtype", u32)]
        symbol = "statfs64"
    elif sys.platform.startswith("freebsd"):
        fields = [("version", u32), ("type", u32)] + [(name, u64) for name in (
            "flags", "bsize", "iosize", "blocks", "bfree", "bavail", "files", "ffree",
            "syncwrites", "asyncwrites", "syncreads", "asyncreads")]
        fields += [("spare", u64 * 10), ("namemax", u32), ("owner", u32),
                   ("fsid", ctypes.c_int32 * 2), ("charspare", ctypes.c_char * 80)]
        symbol = "statfs"
    else:
        return None

    class StatFS(ctypes.Structure):
        # Only the prefix through f_fstypename is read. Reserve space for both mount
        # names and the ABI's remaining fields (Darwin 2168 / FreeBSD 2344 bytes).
        _fields_ = fields + [("fstypename", ctypes.c_char * 16), ("tail", ctypes.c_char * 4096)]

    statfs = getattr(ctypes.CDLL(None, use_errno=True), symbol)
    statfs.argtypes, statfs.restype = [ctypes.c_char_p, ctypes.POINTER(StatFS)], ctypes.c_int
    result = StatFS()
    if statfs(os.fsencode(path), ctypes.byref(result)) != 0:
        return None
    return result.fstypename.decode("ascii")


def _windows_filesystem(path: pathlib.Path) -> Optional[str]:
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    volume = kernel.GetVolumePathNameW
    volume.argtypes, volume.restype = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32], ctypes.c_int
    drive_type = kernel.GetDriveTypeW
    drive_type.argtypes, drive_type.restype = [ctypes.c_wchar_p], ctypes.c_uint32
    root = ctypes.create_unicode_buffer(32768)  # Windows' extended path limit
    if not volume(str(path.resolve()), root, len(root)):
        return None
    return "fixed" if drive_type(root.value) == 3 else None  # DRIVE_FIXED


def _filesystem_type(path: pathlib.Path) -> Optional[str]:
    try:
        if sys.platform == "win32":
            return _windows_filesystem(path)
        if sys.platform.startswith("linux"):
            return _linux_filesystem(path, pathlib.Path("/proc/self/mountinfo").read_text(encoding="utf-8"))
        kind = _bsd_filesystem(path)
        return kind if kind in {"apfs", "hfs", "ufs"} else None
    except (OSError, ValueError, AttributeError):
        return None


def _journal_mode(path: pathlib.Path, tier: str) -> str:
    version = sqlite3.sqlite_version_info
    fixed = (version >= (3, 51, 3) or (version[:2] == (3, 50) and version[2] >= 7)
             or (version[:2] == (3, 44) and version[2] >= 6))
    return "WAL" if (fixed and tier == TIER_ENFORCED and _filesystem_type(path) in _LOCAL_FILESYSTEMS) else "PERSIST"


def _set_journal_mode(conn: sqlite3.Connection, mode: str, wait_sec: float) -> str:
    """``PRAGMA journal_mode``; a switch INTO WAL (a boot owner's choice: ``store_journal``) waits out
    contention within ``wait_sec`` in all. Turning a rollback-journal file into WAL takes the write lock
    from inside a read transaction, where SQLite skips the busy handler (two converters would deadlock),
    so a connection writing then would fail it at once. A retry's busy handler gets only the time left,
    and the open's own busy timeout (``wait_sec``) comes back afterwards. Leaving WAL is never retried:
    every other open WAL connection keeps its shared lock for its life, so waiting would only delay the
    same refusal."""
    budget = max(0.0, wait_sec) if mode.upper() == "WAL" else 0.0
    deadline, bounded = time.monotonic() + budget, False
    while True:
        try:
            actual = conn.execute(f"PRAGMA journal_mode = {mode}").fetchone()[0]
            break
        except sqlite3.OperationalError as exc:
            if not busy(exc) or (left := deadline - time.monotonic()) <= 0:
                raise
            time.sleep(min(0.01, left))
            if (left := deadline - time.monotonic()) <= 0:
                raise
            conn.execute(f"PRAGMA busy_timeout = {max(1, int(left * 1000))}")
            bounded = True
    if bounded:
        conn.execute(f"PRAGMA busy_timeout = {int(budget * 1000)}")
    return actual


def _kept_journal(conn: sqlite3.Connection, path: pathlib.Path, tier: str) -> str:
    """An ordinary open's journal (``store_journal``): none for a file handed off at a seam, connection-local
    PERSIST for a rollback file (every writer's), WAL kept as it is; never a conversion either way. A WAL
    file this SQLite, lock tier or filesystem must not share (``_journal_mode``) is refused instead."""
    if os.path.abspath(path) in _HANDED_OFF:
        return ""
    if conn.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
        return "PERSIST"
    if _journal_mode(path, tier) == "WAL":
        return ""
    raise sqlite3.OperationalError(f"store journal is WAL, which SQLite {sqlite3.sqlite_version}, this lock tier "
                                   "or filesystem must not share; a boot or handoff of this runtime leaves WAL")


def connect(path: pathlib.Path, tier: str, *, create: bool = False, wait_sec: float = 0.0,
            shared: bool = False, journal_mode: Optional[str] = None) -> sqlite3.Connection:
    """A connection with the tier's lock protocol (``sqlite3.Error`` is the caller's to type);
    ``wait_sec`` is SQLite's busy wait on the enforced tier (the name tier's one lock is the
    name protocol the caller holds). Pragmas are set once here. ``shared`` allows the store
    owner to close an idle cached connection from the export/fork thread. Only ``create`` (a private
    build) or a lifecycle owner's ``journal_mode`` chooses WAL versus rollback, else ``_kept_journal``;
    an empty one leaves the journal alone for the checkpoint/handoff owner."""
    query = "mode=rwc" if create else "mode=rw"
    if tier == TIER_NAME:
        query += "&vfs=" + ("win32-none" if os.name == "nt" else "unix-none")
    conn = counted_connect(f"{path.absolute().as_uri()}?{query}", path, uri=True, isolation_level=None,
                           timeout=max(0.0, wait_sec) if tier == TIER_ENFORCED else 0.0, check_same_thread=not shared)
    conn.row_factory = sqlite3.Row
    conn._store_kind = store_timing.kind_of(path)
    try:
        conn.execute("PRAGMA synchronous = FULL")
        if journal_mode is None:
            journal_mode = _journal_mode(path, tier) if create else _kept_journal(conn, path, tier)
        if journal_mode:
            actual = _set_journal_mode(conn, journal_mode, wait_sec if tier == TIER_ENFORCED else 0.0)
            if actual != journal_mode.lower():
                raise sqlite3.OperationalError(f"store journal mode {actual}, expected {journal_mode}")
        conn.execute(f"PRAGMA journal_size_limit = {JOURNAL_SIZE_LIMIT}")
    except BaseException as exc:
        exc.rollback_confirmed = not conn.in_transaction
        conn.close()
        raise
    conn._store_build = create
    return conn


class _CachedConnection:
    def __init__(self, path: pathlib.Path, tier: str, identity: Optional[tuple]) -> None:
        self.path, self.tier, self.conn, self.identity, self.active = path, tier, None, identity, True


# Cached connections one thread keeps open (an LRU of idle handles): the root's few hot stores stay
# open across requests, while a thread that touches many stores over its life (child drives, test
# roots) closes the least recently used idle one instead of holding a descriptor per store forever.
_THREAD_POOL_LIMIT = 8


class _ThreadConnections:
    def __init__(self) -> None:
        self.entries: Dict[str, _CachedConnection] = {}  # insertion order = least recently leased first

    def __del__(self) -> None:
        # Thread-local values die with their thread, including short-lived request threads.
        if not self.entries:
            return
        with _CONNECTION_GUARD:
            for entry in self.entries.values():
                if entry.conn is not None:
                    entry.conn.close()


_LOCAL_CONNECTIONS = threading.local()
_CONNECTION_POOLS: weakref.WeakSet = weakref.WeakSet()
_CONNECTION_GUARD = threading.Condition(threading.RLock())


@contextlib.contextmanager
def connection(path: pathlib.Path, tier: str, *, wait_sec: float = 0.0, identity: Optional[tuple] = None,
               opener: Optional[Callable[..., sqlite3.Connection]] = None) -> Iterator[sqlite3.Connection]:
    """Lease this thread's connection; no transaction survives a returned lease.

    A nested access keeps its historical independent transaction (a temporary connection),
    never commits or rolls back the caller's transaction. A new file ``identity`` (the caller's stat, never one
    across a wait) or tier invalidates the cached connection. Name-tier callers hold their name lock throughout."""
    path = pathlib.Path(os.path.abspath(path))
    key, opener = str(path), opener or connect
    identity = file_identity(path) if identity is None else identity  # a stat waits for the GIL: never under the guard
    with _CONNECTION_GUARD:
        pool = getattr(_LOCAL_CONNECTIONS, "pool", None)
        if pool is None:
            pool = _LOCAL_CONNECTIONS.pool = _ThreadConnections()
            _CONNECTION_POOLS.add(pool)
        entry = pool.entries.get(key)
        if entry is not None and not entry.active and (entry.identity != identity or entry.tier != tier):
            entry.conn.close()
            del pool.entries[key]
            entry = None
        nested = entry is not None and entry.active
        if not nested:
            entry = entry or _CachedConnection(path, tier, identity)
            pool.entries.pop(key, None)
            pool.entries[key] = entry  # now the most recently leased
            entry.active = True
            idle = [name for name, held in pool.entries.items() if not held.active]
            for name in idle[:max(0, len(pool.entries) - _THREAD_POOL_LIMIT)]:
                evicted = pool.entries.pop(name)
                if evicted.conn is not None:
                    evicted.conn.close()
    if nested:
        with contextlib.closing(opener(path, tier, wait_sec=wait_sec)) as conn:
            yield conn
        return
    try:
        if entry.conn is None:
            # SQLite may wait on a different process. Never hold the pool guard during
            # that wait: unrelated stores and active transactions must keep progressing.
            entry.conn = opener(path, tier, wait_sec=wait_sec, shared=True)
        # Acquisition budgets vary by request, unlike durability/journal pragmas.
        entry.conn.execute(f"PRAGMA busy_timeout = {max(0, int(wait_sec * 1000)) if tier == TIER_ENFORCED else 0}")
        yield entry.conn
    finally:
        try:
            if entry.conn is not None and entry.conn.in_transaction:
                entry.conn.execute("ROLLBACK")
        except sqlite3.Error:
            entry.conn.close()
            entry.conn = None
        finally:
            with _CONNECTION_GUARD:
                if entry.conn is None:
                    pool.entries.pop(key, None)
                entry.active = False
                _CONNECTION_GUARD.notify_all()


def _close_cached(selected: Callable[[str], bool], *, idle_only: bool = False) -> None:
    """Close every thread's idle cached handle whose file ``selected`` names: all or none.

    The lifecycle owner must quiesce users first. An active lease refuses the handoff
    with StoreBusy; it is never interrupted or silently rolled back by another thread.
    ``idle_only`` closes the idle ones and leaves a leased handle to its thread instead."""
    with _CONNECTION_GUARD:
        entries = [(pool, name, entry) for pool in list(_CONNECTION_POOLS)
                   for name, entry in pool.entries.items() if selected(name)]
        if idle_only:
            entries = [item for item in entries if not item[2].active]
        elif any(entry.active for _, _, entry in entries):
            raise StoreBusy("store connection still in use at handoff", reason="contention")
        for pool, name, entry in entries:
            entry.conn.close()
            del pool.entries[name]


def close_connections(path: Optional[pathlib.Path] = None) -> None:
    """Close cached handles before a rename/retire, or all handles at shutdown (``_close_cached``)."""
    key = str(pathlib.Path(os.path.abspath(path))) if path is not None else None
    _close_cached(lambda name: key is None or name == key)


def close_connections_under(root: pathlib.Path, *, idle_only: bool = False) -> None:
    """Close this process's cached handles on stores inside ROOT before ROOT moves or goes.

    Windows refuses to rename a folder while a file below it is open, and on any system a
    handle kept past the move holds the moved files' space. Another process (a pool worker
    that ran the task) releases its own, ``idle_only``: a late thread's lease stays with
    that thread, which closes it on exit. Contract as ``_close_cached``; paths resolve
    outside the pool guard, so a slow volume never stalls other stores' leases."""
    def canonical(name: Any) -> str:
        return os.path.normcase(os.path.realpath(name))  # a short or linked spelling names the same file

    prefix = os.path.join(canonical(root), "")
    with _CONNECTION_GUARD:
        names = {name for pool in list(_CONNECTION_POOLS) for name in pool.entries}
    inside = {name for name in names if canonical(name).startswith(prefix)}
    _close_cached(inside.__contains__, idle_only=idle_only)


atexit.register(close_connections)  # before module globals (including the inode guard) are cleared


# Files handed off at an export/switch seam (``prepare_handoff``): a later open in this process (the seam's
# receipt, a late writer) sets no journal mode on them, not even PERSIST, until a refused switch hands them
# back (``forget_handoffs``). A lock-tier re-record converts the file the same way but never enters this set.
_HANDED_OFF: set = set()


def forget_handoffs() -> None:
    """Nothing switches after all (a refused door): this process keeps its stores, in rollback until a boot chooses."""
    _HANDED_OFF.clear()


_WAL_MAGIC = (0x377F0682, 0x377F0683)


def log_identity(path: Any) -> Optional[list]:
    """The commits the WAL log holds (``[checkpoint sequence and salts, last commit frame, its
    checksum, the log's size, its mtime_ns]``), else ``None``. A commit lives there until a checkpoint
    moves it into the database file; an empty or absent log (opened empty by a reader, deleted at the
    last close) holds none, so its coming and going is no change. Read from the log, which carries no
    lock (sqlite.org/fileformat2.html §4): the committed frames are a prefix carrying the generation's
    salts, the last one ending a commit (a database size); frames after them may carry any salts but
    no size. What moves: a commit, its commit frame (with the checksum, size and time); a checkpointed
    log reused in place at the same inode and size, the sequence and salts; a rolled-back transaction's
    frames, the size and time. A writer killed between writing its commit frame and publishing it in
    the index leaves a commit frame no reader sees: the next commit of the same length overwrites it,
    moving its checksum; a shorter one leaves it the last, so only the log's time moves, and where
    that time stays (a held file on Windows, a coarse clock) the identity stays until the next commit."""
    name = f"{os.fspath(path)}-wal"
    try:
        if not stat.S_ISREG(os.stat(name).st_mode):
            return None  # never open a FIFO or a device to look
        with open(name, "rb", buffering=0) as log:
            header = log.read(32)
            if len(header) < 32 or int.from_bytes(header[:4], "big") not in _WAL_MAGIC:
                return None  # no frame is valid before the header
            frame = 24 + int.from_bytes(header[8:12], "big")

            def frame_header(number: int) -> bytes:  # frame NUMBER, from 1
                log.seek(32 + (number - 1) * frame)
                return log.read(24)

            held, above = 0, (os.fstat(log.fileno()).st_size - 32) // frame + 1
            while above - held > 1:  # ends at or past the last committed frame: all before carry the salts
                middle = (held + above) // 2
                held, above = (middle, above) if frame_header(middle)[8:16] == header[16:24] else (held, middle)
            while held and frame_header(held)[4:8] == b"\0\0\0\0":
                held -= 1
            checksum, info = frame_header(held)[16:24] if held else b"", os.fstat(log.fileno())
    except OSError:
        return None
    return [header[12:24].hex(), held, checksum.hex(), info.st_size, info.st_mtime_ns] if held else None


def prepare_handoff(path: pathlib.Path, tier: str, *, wait_sec: float = STORE_WAIT_SEC, seam: bool = True) -> None:
    """Quiescent export/switch seam: release cached handles and leave a standalone DELETE file.
    Name-tier callers hold their name lock. An older SQLite may open the resulting file. A lock-tier
    re-record or a boot owner leaving WAL (``seam=False``) converts it the same way while this process
    keeps running; its ordinary opens then keep the rollback journal (``store_journal``)."""
    close_connections(path)
    with contextlib.closing(connect(path, tier, wait_sec=wait_sec, journal_mode="")) as conn:
        _delete_journal(conn)
    if seam:
        _HANDED_OFF.add(os.path.abspath(path))


def wal_file(path: pathlib.Path) -> bool:
    """The database header says WAL (bytes 18-19 equal 2; sqlite.org/fileformat2.html), read
    through the inode registry like every raw read (``_read_header``): a plain close would drop
    this process's locks on the file, a transaction's on another thread included."""
    try:
        header = _read_header(path)
    except FileNotFoundError:
        return False
    return len(header) >= 20 and header[18] == 2 and header[19] == 2


def hand_off(root: Any, *, wait_sec: float = STORE_WAIT_SEC) -> bool:
    """Before another body takes a quiescent root: a present WAL store is checkpointed and left in
    DELETE mode, because a body that never sets the journal mode (a store-aware older release, an
    older SQLite) could otherwise rename the database away from its log. A rollback-journal store
    is left untouched. True when a file was converted."""
    state, tier = probe(root)
    if state != "present" or not wal_file(store_path(root)):
        return False
    prepare_handoff(store_path(root), tier, wait_sec=wait_sec)  # WAL is never chosen on the name tier
    return True


def _delete_journal(conn: sqlite3.Connection, *, keep_wal: bool = False) -> None:
    wal = conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    if wal and conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
        raise StoreBusy("store checkpoint busy at handoff", reason="contention")
    if not (wal and keep_wal) and conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0] != "delete":
        raise StoreBusy("store journal still in use at handoff", reason="contention")


def _before_fork() -> None:
    # Fork callers are outside store transactions. Drain other threads before copying
    # SQLite's native mutexes; the child must never inherit a reusable connection.
    _CONNECTION_GUARD.acquire()
    while any(entry.active for pool in list(_CONNECTION_POOLS) for entry in pool.entries.values()):
        _CONNECTION_GUARD.wait()
    for pool in list(_CONNECTION_POOLS):
        for entry in pool.entries.values():
            entry.conn.close()
        pool.entries.clear()
    _INODE_GUARD.acquire()


def _after_fork_parent() -> None:
    _INODE_GUARD.release()
    _CONNECTION_GUARD.release()


def _after_fork_child() -> None:
    global _INODE_GUARD, _CONNECTION_GUARD, _PROCESS_LOCKS_GUARD, _IMPORTING_GUARD
    for entry in _INODES.values():
        for handle in entry.files:
            handle.close()
    _INODES.clear()
    _PROCESS_LOCKS.clear()
    store_empty.forget()
    _INODE_GUARD = threading.RLock()
    _CONNECTION_GUARD = threading.Condition(threading.RLock())
    _PROCESS_LOCKS_GUARD = threading.Lock()
    _IMPORTING_GUARD = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent, after_in_child=_after_fork_child)


def busy(exc: BaseException) -> bool:
    code = getattr(exc, "sqlite_errorcode", None)
    return code in {5, 6} or "locked" in str(exc).lower()


def commit(conn: sqlite3.Connection) -> None:
    """COMMIT, rolling back on failure; the ``sqlite3.OperationalError`` is the caller's to type."""
    try:
        conn.execute("COMMIT")
    except sqlite3.OperationalError as exc:
        exc.rollback_confirmed = False
        try:
            conn.execute("ROLLBACK")
            exc.rollback_confirmed = not conn.in_transaction
        except sqlite3.Error:
            pass
        raise


def fsync_directory(directory: pathlib.Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection, *, write: bool, commit_wait_sec: Optional[float] = None,
                finish: Optional[Callable[[], None]] = None) -> Iterator[sqlite3.Connection]:
    """One transaction: a write locks at ``BEGIN IMMEDIATE``, a read at ``SELECT 1`` (its snapshot, its only wait);
    an error rolls back and propagates; ``finish`` (``commit``) ends it, a write's COMMIT waiting ``commit_wait_sec``
    for readers to drain (a read's never waits). The hold goes to ``store_timing``, a failed COMMIT's as failed."""
    began, held = False, None
    try:
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        began = True
        if not write:  # take the read lock now
            conn.execute("SELECT 1 FROM meta LIMIT 1").fetchone()
        held = store_timing.started()
        if write and commit_wait_sec is not None:
            conn.execute(f"PRAGMA busy_timeout = {int(commit_wait_sec * 1000)}")
        yield conn
        (finish or (lambda: commit(conn)))()
    except BaseException as exc:
        if not began:
            exc.rollback_confirmed = not conn.in_transaction
        elif not (getattr(exc, "rollback_confirmed", False) and not conn.in_transaction):
            try:
                conn.execute("ROLLBACK")
                exc.rollback_confirmed = not conn.in_transaction
            except sqlite3.Error:
                exc.rollback_confirmed = False
        if held:
            store_timing.record(conn, write, held, failed=True)
        raise
    store_timing.record(conn, write, held)


def store_path(root: Any) -> pathlib.Path:
    return pathlib.Path(root) / STORE_REL


def generation_stamped(root: Any) -> bool:
    """Whether ``state/migrations.json`` records a completed store generation (a torn
    file proves nothing and reads as unstamped)."""
    from ouroboros.startup_migrations import watermarks

    try:
        return watermarks(root).get(GENERATION_KEY) == GENERATION
    except ValueError:
        return False


def probe(root: Any) -> Tuple[str, Optional[str]]:
    """``("absent", None)`` or ``("present", tier)``; the third answer is typed: ``StoreBusy`` when
    the file cannot be inspected (an exhausted descriptor table is not absence), ``StoreLost`` for a foreign file."""
    path = store_path(root)
    try:
        identity = file_identity(path)  # one stat: the presence, and the file its header's tier is kept for
        tier = header_tier(path, _APPLICATION_ID, identity) if identity is not None else None
    except OSError as exc:
        name = errno.errorcode.get(exc.errno, str(exc.errno))
        raise StoreBusy(f"current store not inspectable ({name}): {path}", reason="uninspectable") from exc
    if identity is None:
        close_connections(path)
        return "absent", None
    if tier is None:
        raise StoreLost(f"current store unreadable or foreign, never replaced: {path}")
    return "present", tier


def retired_sibling(root: Any) -> Optional[pathlib.Path]:
    """The newest ``current.sqlite.exported-<UTC>`` beside the store path, or ``None``."""
    path = store_path(root)
    try:
        names = sorted(name for name in os.listdir(path.parent) if name.startswith(path.name + ".exported-"))
    except OSError:
        return None
    return path.with_name(names[-1]) if names else None


def _typed(exc: sqlite3.Error) -> StoreError:
    if isinstance(exc, sqlite3.OperationalError) and busy(exc):
        error = StoreBusy(f"current store busy: {exc}", reason="contention")
        if hasattr(exc, "rollback_confirmed"):
            error.rollback_confirmed = exc.rollback_confirmed
        return error
    if isinstance(exc, sqlite3.DatabaseError) and not isinstance(
            exc, (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.ProgrammingError)):
        return StoreLost(f"current store unreadable: {exc}")
    return StoreError(f"current store unavailable: {exc}")


@contextlib.contextmanager
def name_lock(root: Any, filename: str, *, timeout_sec: float, stale_sec: float = 90.0) -> Iterator[None]:
    """The owner-aware name lock under ``state/`` (the name tier's one lock, the import lock)."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    path = pathlib.Path(root) / "state" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    outcome = {}
    fd = acquire_exclusive_file_lock(path, timeout_sec=timeout_sec, stale_sec=stale_sec,
                                     owner_aware_stale=True, outcome=outcome)
    if fd is None:
        error = StoreBusy(f"current store lock unavailable: {path}", reason=outcome.get("reason", "unknown"))
        error.rollback_confirmed = True  # No connection/transaction opened behind this name lock.
        raise error
    try:
        yield
    finally:
        release_exclusive_file_lock(path, fd)


_IMPORTING: set = set()  # roots this process's lifespan is importing: their readers see ``StoreMigrating``
_IMPORTING_GUARD = threading.Lock()
_FIRST_TOUCH = threading.local()  # roots THIS thread is importing on first touch: a nested open answers typed
_SERVICE = threading.local()  # the import owner's thread: its own post-publication writes pass the window


def begin_import(root: Any) -> None:
    """Open the lifespan's import window for ROOT: every public write of that root answers
    ``StoreMigrating`` and ``migrating`` is true until ``end_import`` (the boot door closes it
    only after a successful publication; a failed import leaves it open for the process's life)."""
    with _IMPORTING_GUARD:
        _IMPORTING.add(str(pathlib.Path(root)))


def end_import(root: Any) -> None:
    with _IMPORTING_GUARD:
        _IMPORTING.discard(str(pathlib.Path(root)))


@contextlib.contextmanager
def importing(root: Any) -> Iterator[None]:
    """The import window for the extent of a block (tests and the first-touch import)."""
    begin_import(root)
    try:
        yield
    finally:
        end_import(root)


@contextlib.contextmanager
def service_thread() -> Iterator[None]:
    """Service mode for every transaction THIS thread opens: the import owner's steps after
    publication write through the public APIs (latch migration, obligations) past its own window."""
    previous = getattr(_SERVICE, "active", False)
    _SERVICE.active = True
    try:
        yield
    finally:
        _SERVICE.active = previous


def _in_window(root: Any) -> bool:
    with _IMPORTING_GUARD:
        return str(pathlib.Path(root)) in _IMPORTING


def published(root: Any) -> bool:
    """Whether a published store answers for ROOT: present and in phase PUBLISHED (readiness is
    publication, never the file's existence)."""
    try:
        if probe(root)[0] != "present":
            return False
        with open_store(root, write=False) as txn:
            return txn.phase() == PUBLISHED
    except StoreError:
        return False


def store_ready(root: Any) -> bool:
    """Whether a write door may proceed: no import of ROOT runs here and its store is published."""
    return not _in_window(root) and published(root)


def migrating(root: Any) -> bool:
    """The doors' predicate (P1): True while the import window is open for ROOT, while its
    present store is not published (an export runs), or while ROOT holds facts no import has
    published yet (the boot door imports them); a root without facts is not migrating — its first
    write publishes an empty store. A pure read: nothing is opened for write, nothing imported."""
    if _in_window(root):
        return True
    root = pathlib.Path(root)
    try:
        state, _tier = probe(root)
    except StoreError:
        return False  # the door proceeds; its write gets the typed answer
    if state == "present":
        return not published(root)
    return holds_facts(root) and not (generation_stamped(root) or retired_sibling(root))


def holds_facts(root: pathlib.Path) -> bool:
    """Whether ROOT holds facts the importer would read: result files or obligation set files."""
    from ouroboros.store_import import holds_facts as _holds

    return _holds(root)


def _absent(root: pathlib.Path, *, write: bool, service: bool) -> bool:
    """The typed answer for an absent store; a first-touch import when a public WRITE asks for
    one and nothing forbids it (the window, a nested open); ``True`` when a READ may answer
    empty because ROOT holds no facts to import (a read never imports: a root with facts is
    the boot door's, ``StoreMigrating`` until it publishes)."""
    if generation_stamped(root):
        raise StoreLost(f"current store missing beside its completed stamp: {store_path(root)}")
    retired = retired_sibling(root)
    if retired is not None:
        raise StoreRetired(f"current store retired by an export as {retired.name}: the files are the authority")
    entered = getattr(_FIRST_TOUCH, "roots", None) or set()
    if _in_window(root) or str(root) in entered or (write and service):
        raise StoreMigrating(f"current store not published yet: {store_path(root)}")
    if not write:
        if holds_facts(root):
            raise StoreMigrating(f"current store not published yet: {store_path(root)} (a read never imports)")
        return True
    from ouroboros.store_import import ensure_current_store

    # Another thread importing the same root is waited for inside ``ensure_current_store`` (its
    # name lock); only THIS thread's nested open of the root answers typed instead of re-entering.
    _FIRST_TOUCH.roots = entered | {str(root)}
    try:
        ensure_current_store(root, with_children=False)
    except StoreError:
        raise
    except Exception as exc:  # the files the import reads are unavailable: the store cannot answer
        raise StoreMigrating(f"current store not published: the first-touch import failed: {exc}") from exc
    finally:
        _FIRST_TOUCH.roots = entered
    return False


@contextlib.contextmanager
def open_store(root: Any, *, write: bool, wait_sec: float = STORE_WAIT_SEC, service: bool = False,
               barrier: bool = False) -> Iterator["Txn"]:
    """One transaction on the published store: ``StoreMigrating`` while the lifespan imports or
    for a read of a never-imported root that holds facts (a read never imports; a root without
    facts, or one that does not exist, reads empty and creates nothing), a first-touch import
    when a public write reaches a never-imported root (a nested drive's first row), ``StoreLost``
    for a missing store beside its completed stamp, ``StoreRetired`` for one an export set aside
    (the files are the authority until the next import), ``StoreBusy`` on contention;
    ``mode=rw``, never creating here.

    The one write gate: a public write (``write`` without ``service``) refuses ``StoreMigrating``
    while ROOT is in the import window and ``StoreRetired`` inside the transaction when the phase
    is not PUBLISHED. ``service`` is the importer's, the exporter's and the seam's mode (also the
    import owner thread's, ``service_thread``): their transactions pass both checks. A ``barrier``
    holds the write lock and the gate without writing (the evidence packer's phase gate) and ends
    in ROLLBACK, so its release never waits for readers to drain as a COMMIT would."""
    from ouroboros.install_identity import check_store_root
    check_store_root(root, write=write)
    wait_sec = bounded_wait(wait_sec)
    root = pathlib.Path(root)
    service = service or bool(getattr(_SERVICE, "active", False))
    if write and not service and _in_window(root):
        raise StoreMigrating(f"current store not published yet: the import of {root} runs")
    state, tier = probe(root)
    if state == "absent" and ((not write and not root.is_dir()) or _absent(root, write=write, service=service)):
        with store_empty.empty_store(_SCHEMA) as conn, transaction(conn, write=False):
            yield Txn(conn, root)
        return
    if state == "absent":
        state, tier = probe(root)
        if state == "absent":
            raise StoreMigrating(f"current store not published yet: {store_path(root)}")
    with contextlib.ExitStack() as stack:
        if tier == TIER_NAME:
            stack.enter_context(name_lock(root, LOCK_NAME, timeout_sec=wait_sec))
        try:
            conn = stack.enter_context(connection(store_path(root), tier, wait_sec=wait_sec))
        except sqlite3.Error as exc:
            raise _typed(exc) from exc
        handoff = seam = False
        try:
            changes = conn.total_changes
            with transaction(conn, write=write, finish=(lambda: conn.execute("ROLLBACK")) if barrier else None):
                txn = Txn(conn, root)
                phase = txn.phase()
                handoff = seam = phase != PUBLISHED
                if write and not service and handoff:
                    raise StoreRetired(f"current store phase is {phase}: the files are the authority")
                yield txn
                if barrier and conn.total_changes != changes:
                    raise StoreError("a barrier transaction wrote rows; its ROLLBACK would discard them")
                if write and conn.total_changes != changes:
                    txn.set_meta("commit_seq", txn.revision() + 1)
            handoff, seam = handoff or txn._handoff, seam or txn._seam
        except sqlite3.Error as exc:
            raise _typed(exc) from exc
        finally:
            if handoff:
                stack.close()  # return the lease before closing every thread's idle handle
                with name_lock(root, LOCK_NAME, timeout_sec=wait_sec) if tier == TIER_NAME else contextlib.nullcontext():
                    try:
                        prepare_handoff(store_path(root), tier, wait_sec=wait_sec, seam=seam)
                    except sqlite3.Error as exc:
                        raise _typed(exc) from exc


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def supervisor_row(root: pathlib.Path, kind: str, payload: Dict[str, Any], *, best_effort: bool = True) -> None:
    """One supervisor row; pack callers keep append failures, import/switch log them."""
    try:
        append_jsonl(root / "logs" / "supervisor.jsonl", {"ts": utc_now_iso(), "type": kind, **payload})
    except Exception:
        if not best_effort:
            raise
        import logging

        logging.getLogger(__name__).warning("%s row not recorded", kind, exc_info=True)


def document_text(document: Dict[str, Any]) -> str:
    """The whole document serialized exactly as the result writer serializes a file: what an
    export writes (the exact inverse of ``load``), what the content baseline hashes."""
    return machine_json(document)


def encode_document(document: Dict[str, Any]) -> str:
    """The body text stored in ``task_bodies``: the document with each present row-backed field
    emptied (the rows hold its content) and without the card values its ``task_cards`` row holds."""
    return document_text({key: ({} if key in ROW_BACKED_FIELDS else value)
                          for key, value in store_task_cards.strip(document).items()})


def typed_columns(document: Dict[str, Any], *, pending_drive: Optional[bool] = None) -> Dict[str, Any]:
    """Top-level facts and derived Presence, obligation, collection and retention projections."""
    from ouroboros.dialogue_provenance import presence_metadata_binding, presence_record_binding
    from ouroboros.obligations import result_memberships, review_collection_pending
    from ouroboros.plan_record import PLAN_RECORD_KEY, collection_pending
    from ouroboros.retention import task_result_retention_at

    names = result_memberships(document)
    columns: Dict[str, Any] = {}
    for column in TYPED_COLUMNS:
        value = document.get(column)
        text = "" if value is None else str(value).strip()
        columns[column] = text or None  # NULL is "no fact": the lineage prefilter and its index test IS NOT NULL
    # One derived scalar selects a Presence binding's own work without a body read; an empty
    # scalar is not proof of absent provenance (a malformed carrier, or a ceiling whose
    # carrier was lost, must never inherit a queue claim), hence the second fact.
    columns["presence_binding_id"] = presence_record_binding(document) or None
    contract = document.get("task_contract")
    columns["presence_authority_recorded"] = int(
        presence_metadata_binding(document.get("metadata")) is not None
        or (isinstance(contract, dict) and "capability_ceiling" in contract))
    for name, column in FLAG_COLUMNS.items():
        columns[column] = int(name in names)
    columns["pending_drive"] = int("nonterminal" in names if pending_drive is None else pending_drive)
    columns["review_collection_pending"] = int(review_collection_pending(document))
    columns["plan_collection_pending"] = int(collection_pending(document.get(PLAN_RECORD_KEY)))
    columns["drive_retention_at"] = task_result_retention_at(document)
    columns.update(store_hot_facts.task_columns(document))
    return columns


def decode_handoffs(task_ids: List[str], records: list) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{task_id: {kind: {op_id: facts}}}`` from stored handoff rows, every kind present (no store held)."""
    rows: Dict[str, Dict[str, Dict[str, Any]]] = {task_id: {kind: {} for kind in ROW_BACKED_FIELDS} for task_id in task_ids}
    for record in records:
        rows[record["task_id"]].setdefault(record["kind"], {})[record["op_id"]] = json.loads(record["facts"])
    return rows


def project_handoffs(document: Dict[str, Any], rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The body's emptied keys take their rows back; a key the body lacks appears only with rows."""
    for kind, held in rows.items():
        if held or kind in document:
            document[kind] = held
    return document


def materialize(task_id: str, body: Optional[str], records: list, card: Any) -> Optional[Dict[str, Any]]:
    """The document from its stored parts (``Txn.raw``), decoded now: the body, its handoff rows put
    back (``project_handoffs``), its card row over it (``store_task_cards.project``); ``None`` without a body."""
    if body is None:
        return None
    return store_task_cards.project(project_handoffs(json.loads(body), decode_handoffs([task_id], records)[task_id]), card)


class Txn:
    """One store transaction: documents, obligation rows, meta; riders of the cutover
    (handoffs, custody, outbox) share the same connection."""

    def __init__(self, conn: sqlite3.Connection, root: Any = None) -> None:
        self.conn = conn
        self.root = pathlib.Path(root) if root is not None else None
        self._handoff = self._seam = False  # hand the file off; ``_seam``: an export/switch seam

    def meta(self, key: str) -> Any:
        record = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if record is None else json.loads(record["value"])

    def set_meta(self, key: str, value: Any) -> None:
        # The import owner publishes schema versions here, in the same transaction as its DDL: each
        # schema step's addition runs once, with the version, never on a passive read.
        if key == "schema_version":
            held = int(self.meta(key) or 1)
            if (int(value) >= store_hot_facts.VERSION > held
                    or int(value) >= store_hot_facts.LIST_VERSION > held):
                store_hot_facts.upgrade(self)
            if int(value) >= store_task_cards.VERSION > held:
                store_task_cards.upgrade(self)
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, machine_json(value)))
        if (key == "phase" and value in (EXPORTING, RETIRED)) or key == "export_manifest":
            self._handoff = self._seam = True
        elif key == "lock_tier":
            self._handoff = True  # re-recorded at boot: the file is converted, this process carries on

    def phase(self) -> str:
        return str(self.meta("phase") or PUBLISHED)

    def revision(self) -> int:
        """Committed current-fact writes; rollback and empty write barriers do not advance it."""
        return int(self.meta("commit_seq") or 0)

    def task_ids(self) -> List[str]:
        return [record["task_id"] for record in self.conn.execute("SELECT task_id FROM tasks ORDER BY task_id")]

    def write_seqs(self) -> Dict[str, Tuple[int, Any]]:
        """Every task's ``(write_seq, updated_at)`` (``task_store.signature``): one narrow scan of which rows changed."""
        return {record["task_id"]: (record["write_seq"], record["updated_at"])
                for record in self.conn.execute("SELECT task_id, write_seq, updated_at FROM tasks")}

    def body_text(self, task_id: str) -> Optional[str]:
        record = self.conn.execute("SELECT body FROM task_bodies WHERE task_id=?", (task_id,)).fetchone()
        return None if record is None else record["body"]

    def load(self, task_id: str) -> Optional[Dict[str, Any]]:
        """The document: its body with the handoff rows and the card row projected in (``materialize``)."""
        return materialize(task_id, *self.raw(task_id))

    def raw(self, task_id: str) -> tuple:
        """The document's stored parts, nothing decoded: ``(body text, handoff records, card row)``."""
        body = self.body_text(task_id)
        if body is None:
            return None, [], None
        return body, self.handoff_records([task_id]), store_task_cards.read(self, task_id)

    def store_task(self, document: Dict[str, Any], *, pending_drive: Optional[bool] = None,
                   current: Optional[Dict[str, Any]] = None) -> None:
        """Upsert body/columns/handoffs/card; advance write_seq. Reuse the pristine current document
        held by this write transaction, or read for a direct upsert. Update changed bytes in place;
        tasks REPLACE NULLs columns omitted by older writers ("not derived", the import's cue)."""
        task_id = str(document["task_id"])
        retain_swarm_basis(self, document, current=current)
        columns = typed_columns(document, pending_drive=pending_drive)
        columns["write_seq"] = (self.row(task_id, "write_seq") or 0) + 1
        self.conn.execute("INSERT INTO task_bodies (task_id, body) VALUES (?, ?) ON CONFLICT(task_id) DO UPDATE "
                          "SET body=excluded.body WHERE task_bodies.body<>excluded.body", (task_id, encode_document(document)))
        names = ("task_id", *columns)
        self.conn.execute(f"INSERT OR REPLACE INTO tasks ({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                          (task_id, *columns.values()))
        self.sync_handoffs(task_id, document)
        store_task_cards.merge(self, task_id, document)  # the same monotonic rule: a stale document regresses no card
        from ouroboros.obligations import task_written
        task_written(self, task_id, columns)

    def touch(self, task_id: str) -> None:
        """A row-only write still moves the task's signature (the memos re-read)."""
        self.conn.execute("UPDATE tasks SET write_seq = write_seq + 1 WHERE task_id=?", (task_id,))
        from ouroboros.obligations import task_written
        task_written(self, task_id)

    def row(self, task_id: str, *columns: str) -> Any:
        """One task's typed columns (one value for one column, a tuple for several; ``None`` when absent)."""
        record = self.conn.execute(f"SELECT {','.join(columns)} FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if record is None:
            return None
        return record[0] if len(columns) == 1 else tuple(record)

    def delete_task(self, task_id: str) -> None:
        for table in ("task_bodies", "tasks", "task_handoffs"):
            self.conn.execute(f"DELETE FROM {table} WHERE task_id=?", (task_id,))
        store_task_cards.delete(self, task_id)

    def set_names(self) -> List[str]:
        return [record["set_name"] for record in
                self.conn.execute("SELECT DISTINCT set_name FROM obligations ORDER BY set_name")]

    def set_rows(self, set_name: str) -> Dict[str, Dict[str, Any]]:
        """One obligation set, in insertion order (the order its file had)."""
        return {record["id"]: json.loads(record["facts"]) for record in self.conn.execute(
            "SELECT id, facts FROM obligations WHERE set_name=? ORDER BY rowid", (set_name,))}

    def replace_set(self, set_name: str, rows: Dict[str, Dict[str, Any]]) -> None:
        self.conn.execute("DELETE FROM obligations WHERE set_name=?", (set_name,))
        if set_name in store_hot_facts.REQUEST_SETS:
            for identity, facts in rows.items():
                self.put_row(set_name, str(identity), facts)
            return
        self.conn.executemany("INSERT INTO obligations (set_name, id, facts) VALUES (?, ?, ?)",
                              [(set_name, str(identity), machine_json(facts)) for identity, facts in rows.items()])

    def put_row(self, set_name: str, identity: str, facts: Dict[str, Any]) -> None:
        """One owner-held obligation row (an existing row keeps its place; only changed facts write)."""
        self.conn.execute(
            "INSERT INTO obligations (set_name,id,facts) VALUES (?,?,?) ON CONFLICT(set_name,id) "
            "DO UPDATE SET facts=excluded.facts WHERE obligations.facts<>excluded.facts",
            (set_name, identity, machine_json(facts)))
        if set_name in store_hot_facts.REQUEST_SETS:
            self.conn.execute("UPDATE obligations SET request_root_task_id=? WHERE set_name=? AND id=? "
                              "AND request_root_task_id IS NULL", (store_hot_facts.request_root(facts), set_name, identity))

    def delete_row(self, set_name: str, identity: str) -> None:
        self.conn.execute("DELETE FROM obligations WHERE set_name=? AND id=?", (set_name, identity))

    def get_row(self, set_name: str, identity: str) -> Optional[Dict[str, Any]]:
        record = self.conn.execute("SELECT facts FROM obligations WHERE set_name=? AND id=?",
                                   (set_name, identity)).fetchone()
        return None if record is None else json.loads(record["facts"])

    # -- handoff rows: one row per operation of a task's ``ROW_BACKED_FIELDS`` projection, keyed
    #    (task_id, op_id); ``kind`` names the field. Insertion order is the dict order the document had.
    def handoff_records(self, task_ids: List[str]) -> list:
        """The handoff rows of the given tasks as stored (``facts`` still text), in row order."""
        records: list = []
        for start in range(0, len(task_ids), 500):
            chunk = task_ids[start:start + 500]
            records.extend(self.conn.execute(
                f"SELECT task_id, kind, op_id, facts FROM task_handoffs WHERE task_id IN "
                f"({','.join('?' * len(chunk))}) ORDER BY rowid", chunk).fetchall())
        return records

    def handoffs(self, task_id: str) -> Dict[str, Dict[str, Any]]:
        """``{kind: {op_id: facts}}`` of one task (one query), every kind present."""
        return decode_handoffs([task_id], self.handoff_records([task_id]))[task_id]

    def handoff(self, task_id: str, op_id: str) -> Optional[Any]:
        record = self.conn.execute("SELECT facts FROM task_handoffs WHERE task_id=? AND op_id=?",
                                   (task_id, op_id)).fetchone()
        return None if record is None else json.loads(record["facts"])

    def put_handoff(self, task_id: str, kind: str, op_id: str, facts: Any) -> None:
        """Upsert one operation row (an existing row keeps its place); the typed columns are read
        from a dict's ``state`` / ``claimed_at`` / ``returned_at`` / ``local_owner`` when present."""
        typed = facts if isinstance(facts, dict) else {}
        owner = typed.get("local_owner")
        self.conn.execute(
            "INSERT INTO task_handoffs (task_id, op_id, kind, state, claimed_at, returned_at, owner, facts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(task_id, op_id) DO UPDATE SET kind=excluded.kind, "
            "state=excluded.state, claimed_at=excluded.claimed_at, returned_at=excluded.returned_at, "
            "owner=excluded.owner, facts=excluded.facts",
            (task_id, op_id, kind, *(None if typed.get(key) is None else str(typed[key])
                                     for key in ("state", "claimed_at", "returned_at")),
             machine_json(owner) if isinstance(owner, dict) else None, machine_json(facts)))

    def delete_handoff(self, task_id: str, op_id: str) -> None:
        self.conn.execute("DELETE FROM task_handoffs WHERE task_id=? AND op_id=?", (task_id, op_id))

    def sync_handoffs(self, task_id: str, document: Dict[str, Any]) -> None:
        """The rows follow a whole-document write (the importer, a mutator over the projected
        fields): rows the document no longer names go, changed or new ones are written."""
        held = self.handoffs(task_id)
        for kind in ROW_BACKED_FIELDS:
            wanted = document.get(kind) if isinstance(document.get(kind), dict) else {}
            current = held.get(kind, {})
            if kind == "swarm_fanouts":  # immutable producer facts survive stale replicas/older writers
                wanted = {**wanted, **current}
            for op_id in current.keys() - wanted.keys():
                self.delete_handoff(task_id, str(op_id))
            for op_id, facts in wanted.items():
                if str(op_id) not in current or current[str(op_id)] != facts:
                    self.put_handoff(task_id, kind, str(op_id), facts)

    # -- custody rows: one row per delegated run, the reducer's output, indexed by task / project /
    #    invocation; ``state`` is the custody hold (``held`` while the run owes work, ``closed`` after).
    def custody_rows(self, where: str = "", params: Tuple[Any, ...] = ()) -> Dict[str, Dict[str, Any]]:
        return {key: json.loads(raw) for key, raw in self.custody_raw_rows(where, params).items()}

    def custody_raw_rows(self, where: str = "", params: Tuple[Any, ...] = ()) -> Dict[str, str]:
        """Fetch one consistent selection; callers can decode after releasing the read transaction."""
        clause = f" WHERE {where}" if where else ""
        return {record["run_id"]: record["facts"] for record in
                self.conn.execute(f"SELECT run_id, facts FROM custody_runs{clause} ORDER BY run_id", params)}

    def custody_row(self, run_id: str) -> Optional[Dict[str, Any]]:
        record = self.conn.execute("SELECT facts FROM custody_runs WHERE run_id=?", (run_id,)).fetchone()
        return None if record is None else json.loads(record["facts"])

    def put_custody(self, run_id: str, facts: Dict[str, Any], *, state: str) -> None:
        columns = store_hot_facts.custody_columns(facts)
        self.conn.execute(
            "INSERT INTO custody_runs (run_id, task_id, project_id, invocation_id, state, facts, updated_at, "
            f"{','.join(columns)}, write_seq) VALUES ({','.join('?' * (7 + len(columns)))}, 1) "
            "ON CONFLICT(run_id) DO UPDATE SET task_id=excluded.task_id, "
            "project_id=excluded.project_id, invocation_id=excluded.invocation_id, state=excluded.state, "
            "facts=excluded.facts, updated_at=excluded.updated_at, write_seq=custody_runs.write_seq+1, "
            + ",".join(f"{name}=excluded.{name}" for name in columns),
            (run_id, str(facts.get("task_id") or ""), str(facts.get("project_id") or ""),
             str(facts.get("invocation_id") or ""), state, machine_json(facts), utc_now_iso(), *columns.values()))

    # -- the custody outbox: an emitted event waits here between its COMMIT and its chain append.
    def outbox_put(self, event: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO custody_outbox (event_id, run_id, task_id, event, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(event.get("event_id") or ""), str(event.get("run_id") or ""), str(event.get("task_id") or ""),
             machine_json(event), utc_now_iso()))

    def outbox_rows(self) -> List[Dict[str, Any]]:
        return [json.loads(record["event"]) for record in
                self.conn.execute("SELECT event FROM custody_outbox ORDER BY created_at, event_id")]

    def outbox_delete(self, event_id: str) -> None:
        self.conn.execute("DELETE FROM custody_outbox WHERE event_id=?", (event_id,))


def update_task(root: Any, task_id: str, mutator: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]], *,
                pending_drive: Any = None) -> Optional[Dict[str, Any]]:
    """Merge CURRENT under the published write gate; ``None`` commits without a write.
    ``pending_drive`` is a value or ``(txn, current, updated) -> Optional[bool]``, ``current`` the
    pristine document (the mutator works on its own copy, decoded from the same stored bytes, not
    an encoded deep copy). Schema preparation precedes the callback, including on a first-touched
    old child store."""
    while True:
        with open_store(root, write=True) as txn:
            if int(txn.meta("schema_version") or 1) >= SCHEMA_VERSION:
                raw = txn.raw(task_id)
                current, updated = materialize(task_id, *raw) or {}, mutator(materialize(task_id, *raw) or {})
                if updated is None:
                    return None
                flag = pending_drive(txn, current, updated) if callable(pending_drive) else pending_drive
                updated = {**updated, "task_id": task_id}
                txn.store_task(updated, pending_drive=flag, current=current)
                return updated
        # No callback ran: the import owner upgrades with the same acquisition budget.
        from ouroboros.store_import import upgrade_current_schema
        upgrade_current_schema(root)
