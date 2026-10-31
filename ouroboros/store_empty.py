"""The empty answer of an absent SQLite store (``store.open_store``, ``dialogue_store.transaction``).

A read of a root that holds no store reads an empty in-memory copy of the store's schema. Building it
is one DDL statement per table, index and trigger (43 for ``state/current.sqlite``), and in CPython every
statement releases and re-takes the GIL: under a busy server each hand-off waits, so a history page
whose settled child drives no longer exist spent seconds building schemas nobody reads. This module
builds one read-only copy per schema, thread and nesting depth, and hands it back to the next read.
It caches the schema, never an answer: whether a root holds a store is asked again on every open.
A copy holds about 0.35 MiB (current store) or 0.25 MiB (dialogue) per thread that read an absent
store, until that thread ends.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from typing import Iterator

_IDLE = threading.local()  # this thread's idle copies by schema text; a nested read takes another


@contextlib.contextmanager
def empty_store(schema: str) -> Iterator[sqlite3.Connection]:
    """An empty, read-only (``query_only``) store of ``schema`` for one read; the caller runs the
    transaction. The copy returns to this thread's idle list unless the read left it mid-transaction."""
    idle = _IDLE.__dict__.setdefault("copies", {}).setdefault(schema, [])
    conn = idle.pop() if idle else None
    if conn is None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.executescript(schema)
        conn.execute("PRAGMA query_only = ON")
    try:
        yield conn
    finally:
        if conn.in_transaction:
            conn.close()
        else:
            idle.append(conn)


def forget() -> None:
    """A forked child keeps none of the forking thread's copies (``store._after_fork_child``)."""
    _IDLE.__dict__.pop("copies", None)
