"""The counts and exact intervals the chain readers kept, answered from ``state/dialog.sqlite``.

The JSONL readers addressed rows by counting: ``chat_chain._stream_rows`` numbered the chat
rows outside A2A traffic (the consolidation cursor's offset inside one generation, chronicle
``raw_range.pos`` and ``stream_span``), task-events v1 ``line`` and the logs tail's ``_line``
numbered every parsed object of a source. ``dialogue_store`` keeps both counts with every row,
relative to the row's unit (an archive segment, or the live unit of everything stored after the
cutover) and the import settles the units' absolute bases in one transaction once the archive
below the cutover is complete (``dialogue_import._settle``). This module turns a coordinate
into its absolute counts and back, resolves a legacy generation cursor, and lists the exact
byte intervals the import skipped. Every answer is one indexed probe of the row's own unit,
never a count over other rooms' history; an absolute count that is not settled yet is a typed
``StoreMigrating`` (or ``StoreError`` when a segment never will be imported), never a number.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, List, NamedTuple, Optional

from ouroboros import dialogue_store as ds
from ouroboros.dialogue_store import Row
from ouroboros.store import StoreError, StoreMigrating


class Ordinals(NamedTuple):
    """A row's absolute counts: ``ordinal`` among every stored object of its stream (0-based; the
    chain readers' ``line`` is ``ordinal + 1`` from a window's first object), ``pos`` its chat
    stream position outside A2A rows (None for an A2A or a progress row)."""

    ordinal: int
    pos: Optional[int]


class Skipped(NamedTuple):
    """A line the import did not store: its coordinate, exact length and kind (``SKIPPED_KINDS``)."""

    stream: str
    seq: int
    length: int
    kind: str

    @property
    def end(self) -> int:
        return self.seq + self.length


# --- ordinals, exact ends, skipped intervals --------------------------------------------------

def _uncovered(conn: sqlite3.Connection, stream: str) -> StoreError:
    """Why an absolute count is unknown: the archive below the cutover is still importing
    (``StoreMigrating``) or holds a segment that never will be (``StoreError``)."""
    states = {record["state"] for record in conn.execute(
        "SELECT state FROM import_ledger WHERE stream=? AND state IN ('pending', 'mismatch')", (stream,))}
    if "mismatch" in states:
        return StoreError("Dialogue history: a segment could not be imported")
    return StoreMigrating("Dialogue history is still importing")


_ORDINALS = ("SELECT seq, ordinal + (SELECT ordinal_base FROM import_ledger WHERE id=segment) AS ordinal, "
             "pos + (SELECT pos_base FROM import_ledger WHERE id=segment) AS pos, "
             "(SELECT ordinal_base FROM import_ledger WHERE id=segment) IS NULL AS unknown FROM dialogue")


def ordinals_of(root: Any, stream: str, seqs: Iterable[int]) -> Dict[int, Ordinals]:
    """The absolute ``Ordinals`` of the rows at ``seqs`` (a page's ``line`` numbers in one
    statement and ONE step per 500 rows, as ``dialogue_store._page`` reads a page; an absent
    coordinate is absent). Each row's unit answers through its own base: never another room's
    history. Raises ``StoreMigrating``/``StoreError`` while the counts below the cutover are not
    settled: an unknown ordinal is never a number."""
    wanted = sorted({int(seq) for seq in seqs})
    found: Dict[int, Ordinals] = {}
    with ds.transaction(root, write=False) as conn:
        for start in range(0, len(wanted), 500):
            chunk = wanted[start:start + 500]
            listed = conn.execute(f"SELECT json_group_array(json_array(seq, ordinal, pos, unknown)) FROM ({_ORDINALS} "
                                  f"WHERE stream=? AND seq IN ({','.join('?' * len(chunk))}))",
                                  (stream, *chunk)).fetchone()[0]
            for seq, ordinal, pos, unknown in json.loads(listed or "[]"):
                if unknown:
                    raise _uncovered(conn, stream)
                found[seq] = Ordinals(int(ordinal), pos)
    return found


def ordinal_at(root: Any, stream: str, seq: int) -> Optional[Ordinals]:
    """The absolute counts of one row (``None`` for no row at ``seq``); typed while unsettled."""
    return ordinals_of(root, stream, [seq]).get(int(seq))


def _unit_row(conn: sqlite3.Connection, stream: str, column: str, index: str, value: int) -> Optional[sqlite3.Row]:
    """The row at absolute ``value`` of ``column`` (``ordinal`` or ``pos``): its unit by base
    (the ledger's index), then the row by its relative count (``dialogue_ordinal``/``dialogue_pos``).
    A unit is an archive segment below the cutover coordinate or the live unit; a segment that
    joined after the cutover counts in the live unit and is no unit itself."""
    start, span = f"{column}_base", "rows" if column == "ordinal" else "stream_rows"
    unit = conn.execute(
        f"SELECT id, {start} AS start FROM import_ledger INDEXED BY {index} WHERE stream=? AND {start}<=? AND (state='live' OR "
        f"(base<(SELECT cutover_seq FROM streams WHERE streams.stream=import_ledger.stream) AND {start}+coalesce({span}, 0)>?)) "
        f"ORDER BY {start} DESC LIMIT 1", (stream, value, value)).fetchone()
    if unit is None:
        return None
    return conn.execute(f"{ds._SELECT} INDEXED BY dialogue_{column} WHERE segment=? AND {column}=?",
                        (unit["id"], value - int(unit["start"]))).fetchone()


def _settled(conn: sqlite3.Connection, stream: str) -> bool:
    return conn.execute("SELECT ordinal_base FROM import_ledger WHERE id=(SELECT live FROM streams WHERE stream=?)",
                        (stream,)).fetchone()["ordinal_base"] is not None


def row_at_ordinal(root: Any, stream: str, ordinal: int) -> Optional[Row]:
    """The row at absolute object ``ordinal`` (0-based; ``None`` beyond the stream's end); typed
    while the counts are unsettled."""
    with ds.transaction(root, write=False) as conn:
        record = _unit_row(conn, stream, "ordinal", "ledger_ordinal", int(ordinal))
        if record is None and not _settled(conn, stream):
            raise _uncovered(conn, stream)
    return None if record is None else ds._row(record)


def row_at_position(root: Any, pos: int) -> Optional[Row]:
    """The chat row at absolute stream position ``pos`` (``chat_chain`` ``pos``, a chronicle
    ``raw_range``/``stream_span`` bound; ``None`` beyond the end); typed while unsettled."""
    with ds.transaction(root, write=False) as conn:
        record = _unit_row(conn, "chat", "pos", "ledger_pos", int(pos))
        if record is None and not _settled(conn, "chat"):
            raise _uncovered(conn, "chat")
    return None if record is None else ds._row(record)


def last_stream_position(root: Any) -> int:
    """The chat projection's current head, excluding A2A, or -1 for an empty stream; no history scan."""
    ds.require_imported(root, "chat")
    with ds.transaction(root, write=False) as conn:
        row = conn.execute("SELECT next_pos, pos_base FROM streams JOIN import_ledger ON id=live "
                           "WHERE streams.stream='chat'").fetchone()
        if row is None:
            return -1
        if row["pos_base"] is None:
            raise _uncovered(conn, "chat")
        return int(row["pos_base"]) + int(row["next_pos"]) - 1


def generation_row(root: Any, gen: str, offset: int) -> Optional[Row]:
    """The row a legacy consolidation cursor names: the ``offset``-th stream row counted from
    the first line of generation ``gen`` on, through the following generations
    (``chat_chain._resolve_generation_segments``; ``None`` past the chain's end). An unknown
    ``gen`` while chat segments are pending is ``StoreMigrating``; absolute counts are needed
    only past the generation's own rows."""
    if not gen or int(offset) < 0:
        return None
    with ds.transaction(root, write=False) as conn:
        unit = conn.execute("SELECT * FROM import_ledger WHERE stream='chat' AND gen=? AND state='done' ORDER BY base LIMIT 1",
                            (gen,)).fetchone()
        if unit is None:
            if conn.execute("SELECT 1 FROM import_ledger WHERE stream='chat' AND state='pending' LIMIT 1").fetchone():
                raise StoreMigrating("Dialogue history is still importing")
            return None
        cutover = conn.execute("SELECT cutover_seq FROM streams WHERE stream='chat'").fetchone()["cutover_seq"]
        if unit["pos_base"] is not None:
            record = _unit_row(conn, "chat", "pos", "ledger_pos", int(unit["pos_base"]) + int(offset))
        elif cutover is not None and int(unit["base"]) < int(cutover) and int(offset) < int(unit["stream_rows"] or 0):
            record = conn.execute(f"{ds._SELECT} INDEXED BY dialogue_pos WHERE segment=? AND pos=?", (unit["id"], int(offset))).fetchone()
        else:
            raise _uncovered(conn, "chat")
    return None if record is None else ds._row(record)


def skipped_between(root: Any, stream: str, lower: int = 0, upper: Optional[int] = None) -> List[Skipped]:
    """The lines the import did not store whose coordinates start in ``[lower, upper)``, in
    order: the exact gaps a reader reports (``history_gap``, an unreadable-row reset). Blank
    lines are not among them: they hold nothing and are no gap."""
    with ds.transaction(root, write=False) as conn:
        records = conn.execute("SELECT stream, seq, length, kind FROM skipped WHERE stream=? AND seq>=? AND seq<? ORDER BY seq",
                               (stream, int(lower), ds._bound(upper))).fetchall()
    return [Skipped(record["stream"], record["seq"], record["length"], record["kind"]) for record in records]
