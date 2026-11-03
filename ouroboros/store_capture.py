"""A read's answer carried out of its transaction in ONE SQLite step, decoded after the transaction closed.

CPython's ``sqlite3`` releases and retakes the GIL around every ``sqlite3_step``: a SELECT stepped row by
row (iteration, ``fetchall``) inside a read transaction keeps the store's lock across one GIL re-queue per
row, and beside busy threads on a loaded host each re-queue can wait for another thread or for a core.
Under a rollback journal that lock is SHARED, which every writer's COMMIT waits out (and, through
PENDING, every new reader behind that writer). ``capture`` runs the caller's own SELECT once as
``SELECT json_group_array(json_array(<columns>)) FROM (<select>)``: SQLite builds the whole answer with
the GIL released and Python receives one string. ``Captured.rows`` decodes it after the caller's
transaction ended: one crossing per statement inside a read transaction, decoded after release
(DEVELOPMENT «Projection over replay»). Several captures inside one transaction are one snapshot.

``columns`` are the SELECT's own output names, given by the caller; no SQL is parsed or inferred.
TEXT (a JSON string: JSON text stays text; NUL and every code point travel), INTEGER (to 2**63-1) and
NULL travel exactly. A REAL value is refused when decoded (``CaptureTypeError``): no SQL text rendering
of a double round-trips every value, so a REAL consumer reads that column outside the aggregate. A BLOB
fails inside SQLite. A value with SQLite's JSON subtype (a JSON function's object or array) nests as
JSON: ``'' || <expr>`` keeps it text. SQLite promises no aggregate order below 3.44 (the store's floor
is 3.24), so a caller that needs the SELECT's order restores it with ``key`` over columns that fix it.
One answer is one SQLite string, below SQLite's length limit (1e9 bytes by default): a selection
that can approach it (every body of a store) is captured in pages of its key (``task_store._fetch``).
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")  # an unquoted SQL name: a typo fails in SQLite, never reads a literal


class CaptureTypeError(ValueError):
    """A captured value the capture cannot carry exactly: a REAL (its column is read outside the aggregate)."""


def _refuse_real(literal: str) -> Any:
    raise CaptureTypeError(f"a REAL value ({literal}) cannot travel in a capture: read that column outside it")


class Captured:
    """One captured answer: the aggregate's text and its column names; decoded only by ``rows``/``tuples``."""

    __slots__ = ("text", "columns")

    def __init__(self, text: str, columns: Tuple[str, ...]) -> None:
        self.text, self.columns = text, columns

    def tuples(self) -> List[tuple]:
        """The rows as tuples in ``columns`` order, aggregate order (``rows`` restores a SELECT's order)."""
        return [tuple(row) for row in json.loads(self.text, parse_float=_refuse_real)]

    def rows(self, key: Optional[Callable[[Dict[str, Any]], Any]] = None, *,
             reverse: bool = False) -> List[Dict[str, Any]]:
        """The rows as dicts by column name, sorted by ``key`` (``reverse``) when the caller needs an order."""
        rows = [dict(zip(self.columns, row)) for row in self.tuples()]
        return rows if key is None else sorted(rows, key=key, reverse=reverse)


def capture(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = (), columns: Sequence[str] = ()) -> Captured:
    """SQL's whole answer as one string, in one step on ``conn``'s open transaction (``Captured`` decodes it)."""
    names = tuple(str(name) for name in columns)
    if not names or not all(_NAME.match(name) for name in names):
        raise ValueError(f"a capture names its SELECT's output columns: {names!r}")
    record = conn.execute(f"SELECT json_group_array(json_array({', '.join(names)})) FROM ({sql})",
                          tuple(params)).fetchone()
    return Captured(record[0] or "[]", names)
