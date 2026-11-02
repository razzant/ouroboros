"""Durable facts of named background jobs, one ``meta`` row per job in ``state/current.sqlite``.

Work that grows with accumulated history (a legacy reconcile, an archive backfill, a
promotion retry) is a named job, not a boot or request step. Its facts live here so that
"once" means once per installation, never once per process: a restart reads the job's
row instead of redoing the work, and a negative result ("scanned, nothing found") is a
fact like any other. A job row is a small JSON document under the ``meta`` key
``job:<name>``; every update is one store transaction, so concurrent writers (the server
and the maintenance child) never interleave half-updates. The health document, the
maintenance child and the tests read the same rows.

Row fields are the job's own; the ones every job keeps are ``state`` (one of ``STATES``),
``updated_at`` and, while it advances, ``cursor``/``done``/``total`` (``total`` is
``None`` when the job cannot know its denominator), plus ``error`` for a failed run and ``last_error``
``{class, at}`` while its last run threw (``maintenance.run``: the state is left as it was). A job
that cannot proceed records ``failed`` with its reason and the inputs it waits on; it is
retried when those inputs change, never by an elapsed-time or attempt-count rule.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable, Optional

from ouroboros import store
from ouroboros.utils import utc_now_iso

PREFIX = "job:"
STATES = ("pending", "running", "done", "failed")


def _key(name: str) -> str:
    name = str(name or "").strip()
    if not name or ":" in name:
        raise ValueError(f"invalid job name: {name!r}")
    return PREFIX + name


def job(root: Any, name: str) -> Optional[Dict[str, Any]]:
    """The job's current row, or ``None`` when it never ran on this root."""
    with store.open_store(pathlib.Path(root), write=False) as txn:
        value = txn.meta(_key(name))
    return dict(value) if isinstance(value, dict) else None


def jobs(root: Any, names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Job rows keyed by job name: every row of the root (one indexed range read of ``meta``), or
    only the named jobs' rows (the dispatcher's look: a row another owner keeps proportional to
    its own inventory, the drive settlement's verdicts, is not decoded for it)."""
    import json

    keys = None if names is None else [_key(name) for name in names]
    if keys is not None and not keys:
        return {}
    with store.open_store(pathlib.Path(root), write=False) as txn:
        if keys is None:
            rows = txn.conn.execute(
                "SELECT key, value FROM meta WHERE key >= ? AND key < ?", (PREFIX, PREFIX[:-1] + ";"),
            ).fetchall()
        else:
            rows = txn.conn.execute(
                f"SELECT key, value FROM meta WHERE key IN ({','.join('?' * len(keys))})", keys).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        value = json.loads(row["value"])
        if isinstance(value, dict):
            out[str(row["key"])[len(PREFIX):]] = value
    return out


def record(root: Any, name: str, **facts: Any) -> Dict[str, Any]:
    """Merge ``facts`` into the job's row in one write transaction; returns the new row."""
    with store.open_store(pathlib.Path(root), write=True) as txn:
        return record_in(txn, name, **facts)


def record_in(txn: store.Txn, name: str, **facts: Any) -> Dict[str, Any]:
    """``record`` inside the caller's write transaction (a job that records the revision its own write leaves)."""
    state = facts.get("state")
    if state is not None and state not in STATES:
        raise ValueError(f"invalid job state: {state!r}")
    key = _key(name)
    current = txn.meta(key)
    row = dict(current) if isinstance(current, dict) else {}
    row.update(facts)
    row["updated_at"] = utc_now_iso()
    txn.set_meta(key, row)
    return row


def forget_in(txn: store.Txn, name: str, *fields: str) -> Dict[str, Any]:
    """Drop ``fields`` a job no longer keeps on its row, inside the caller's write transaction."""
    key = _key(name)
    current = txn.meta(key)
    row = {field: value for field, value in (current.items() if isinstance(current, dict) else ()) if field not in fields}
    txn.set_meta(key, row)
    return row


def is_done(root: Any, name: str) -> bool:
    row = job(root, name)
    return bool(row) and row.get("state") == "done"
