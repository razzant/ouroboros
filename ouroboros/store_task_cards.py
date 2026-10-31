"""The task card's display facts (``tool_calls``, ``last_line``, ``last_line_at``) in one small
row per task (``task_cards``, schema 7), never in the task body.

Every delivered tool result and every model narration changes one of these three facts; a body
write for that would re-journal a document of up to tens of MB to move one counter. So the card
has its own row: a row-only writer (``merge``) merges monotonically in SQL (``tool_calls`` never
goes down, a line is replaced only by one that is not older), creates nothing for a task without a
lifecycle row, reads no body, writes no body and leaves ``tasks.write_seq`` (the document signature
the memos key on) alone. Whole-document writes merge the card values a document carries through the
same rule, so a stale writer and an import cannot regress a card, and ``encode_document`` stores no
admissible card value in the body; readers project the row back (``project``: the row wins over a
legacy body copy key by key, keys the body no longer keeps are appended in ``FIELDS`` order), so
``Txn.load``, the implicit document reads, the listings, the display views and the export all show
the card. An inadmissible value (a string count, a negative one, a non-string line) is not a card
fact: it stays in the body as stored and ``task_activity_facts.card_facts`` reads it as 0 or "".

A store an upgrade has not reached (a child drive at schema 6) has no table: readers see its legacy
body copies and writers reach it only through the schema gate (``Txn.set_meta`` runs ``upgrade``).
The row goes with the task (``Txn.delete_task``) and travels with the document through every seam
that moves documents (import, export, reconcile), because they all go through ``store_task``/``load``.
Only plain SQL of the store's SQLite floor here (no JSON functions): the export runs it bare.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, Mapping, Optional

TABLE = "task_cards"
VERSION = 7  # the store schema version that holds the table
FIELDS = ("tool_calls", "last_line", "last_line_at")
SCHEMA = f"CREATE TABLE IF NOT EXISTS {TABLE} (task_id TEXT PRIMARY KEY, tool_calls INTEGER, last_line TEXT, last_line_at TEXT);\n"
# The monotonic merge: a count never goes down, a line is replaced only by one not older than the
# held one (an absent new line keeps the held one). The INSERT selects its values so that a task
# without a lifecycle row inserts nothing (a display observation creates no lifecycle owner).
_MERGE = f"""INSERT INTO {TABLE} (task_id, tool_calls, last_line, last_line_at)
SELECT ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM tasks WHERE task_id=?)
ON CONFLICT(task_id) DO UPDATE SET
 tool_calls = CASE WHEN excluded.tool_calls IS NULL THEN {TABLE}.tool_calls
  WHEN {TABLE}.tool_calls IS NULL THEN excluded.tool_calls ELSE max({TABLE}.tool_calls, excluded.tool_calls) END,
 last_line = CASE WHEN excluded.last_line_at IS NULL OR excluded.last_line_at < coalesce({TABLE}.last_line_at, '')
  THEN {TABLE}.last_line ELSE excluded.last_line END,
 last_line_at = CASE WHEN excluded.last_line_at IS NULL OR excluded.last_line_at < coalesce({TABLE}.last_line_at, '')
  THEN {TABLE}.last_line_at ELSE excluded.last_line_at END"""


def admissible(document: Mapping[str, Any]) -> Dict[str, Any]:
    """The card values a document (or a writer's facts) carries that the row takes."""
    values: Dict[str, Any] = {}
    calls = document.get("tool_calls")
    if type(calls) is int and calls >= 0:
        values["tool_calls"] = calls
    if isinstance(document.get("last_line"), str):  # a stamp without its line is no card fact
        values["last_line"] = document["last_line"]
        if isinstance(document.get("last_line_at"), str):
            values["last_line_at"] = document["last_line_at"]
    return values


def strip(document: Mapping[str, Any]) -> Dict[str, Any]:
    """The document without the card values the row holds (what the body stores)."""
    held = admissible(document)
    return {key: value for key, value in document.items() if key not in held}


def row_values(row: Any) -> Dict[str, Any]:
    """The card facts a row (``sqlite3.Row`` or mapping) holds: the non-NULL ones only."""
    if row is None:
        return {}
    keys = row.keys()
    return {name: row[name] for name in FIELDS if name in keys and row[name] is not None}


def project(document: Dict[str, Any], row: Any) -> Dict[str, Any]:
    """The row's facts into the document: the row wins over a legacy body copy, key by key."""
    for name, value in row_values(row).items():
        document[name] = value
    return document


def canonical(document: Mapping[str, Any]) -> Dict[str, Any]:
    """The shape ``load`` returns for a stored document: what the content baseline hashes."""
    return project(strip(document), admissible(document))


def present(txn) -> bool:
    """Whether the store holds the table (memoized on the transaction)."""
    held = getattr(txn, "_cards", None)
    if held is None:
        held = txn._cards = txn.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone() is not None
    return held


def upgrade(txn) -> None:
    """Schema-only addition inside the schema-version publication gate: no body is decoded."""
    txn.conn.execute(SCHEMA.rstrip(";\n"))
    txn._cards = True


def read(txn, task_id: str) -> Optional[sqlite3.Row]:
    """The task's card row, ``None`` without one (or before the table exists)."""
    if not present(txn):
        return None
    return txn.conn.execute(f"SELECT tool_calls, last_line, last_line_at FROM {TABLE} WHERE task_id=?",
                            (task_id,)).fetchone()


def merge(txn, task_id: str, values: Mapping[str, Any]) -> int:
    """Merge ``values`` (``admissible`` keys only) into the task's row: rows changed (0 for an
    absent task, nothing admissible, or a store without the table)."""
    held = admissible(values)
    if not held or not present(txn):
        return 0
    return txn.conn.execute(_MERGE, (task_id, held.get("tool_calls"), held.get("last_line"),
                                     held.get("last_line_at"), task_id)).rowcount


def delete(txn, task_id: str) -> None:
    if present(txn):
        txn.conn.execute(f"DELETE FROM {TABLE} WHERE task_id=?", (task_id,))


def select(txn) -> tuple:
    """``(columns, join)`` to add to a ``tasks AS t`` query for the card columns (empty without the table)."""
    if not present(txn):
        return "", ""
    return (", " + ", ".join(f"c.{name}" for name in FIELDS), f" LEFT JOIN {TABLE} AS c ON c.task_id = t.task_id")
