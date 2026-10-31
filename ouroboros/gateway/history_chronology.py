"""Time-ordered pages of a room that holds imported earlier messages.

``presence_history`` stores a connector's earlier messages after the live rows they
precede, so stored order is not reading order there. ``ChronologicalSource`` pages such
a room's chat rows by their reading key ``[time, order, position]``
(``place_arrivals.READING_TIME``/``READING_ORDER``): the provider send time and order a new
arrival keeps in its ``source_order`` block, else its ``ts`` and no order (null, before every
order), then its stored position. Every read seeks the room's ``dialogue_room_reading`` index
from a row key in at most three bounded ranges (the key's own time and order, its time at
lower orders, then other times), never a count, an offset or a walk of the rows sharing one
time or order. Rows a later import stored between a frozen chain's keys are stepped over in
that index, never read or delivered, so a frozen page costs those too
(``test_a_frozen_older_page_steps_over_rows_imported_after_it_froze`` measures it).

The set stays frozen at the cursor's physical ``upper``, and rows keep their physical
``history_id`` and ``history_position``. A page continues from the key of a row it
delivered (the cursor's ``chrono`` keys, ``history_paging.history_page_tokens``). Coverage
and the deferral consumers (``history_paging.deferred_before``, the lineage deferral in
``history.py``) compare order-preserving integers, so a row's coordinate is its place in
reading order, counted down from ``_HORIZON``: the newest frozen row is ``_HORIZON - 1``,
each older one a step below, and 0 names the room's beginning. A page learns where it
starts from its cursor's integer boundary, never by counting what lies above it, so rows
sharing one time keep distinct coordinates and a span holds exactly the rows its page
delivered. A page around one row (``pin``) puts that row at ``_PIVOT`` and begins a chain
of its own (``-at-<its stored position>``): its coordinates are never the newest page's.
Every cut a page reads below or reports is also named by the key of the first frozen row
above it (``None``: the present), so the coverage ``keys`` of any read of these rows
compare whatever their chain (``history_paging.history_page_coverage``). A line the import
could not store has no known time, so it is a gap of every page.
"""
from __future__ import annotations

from ouroboros import dialogue_store as ds
from ouroboros.gateway.history_paging import (
    _CHAIN_WITNESS,
    _PROBE_ROWS,
    _READ_CEILING_ROWS,
    _RECENT_WINDOW_BYTES,
    _RECENT_WINDOW_ROWS,
    HistoryCursorError,
    HistorySource,
)
from ouroboros.place_arrivals import READING_ORDER, READING_TIME, reading_key
from ouroboros.store import StoreMigrating

_KEYS = (f"SELECT seq, {READING_TIME} AS time, {READING_ORDER} AS ord FROM dialogue INDEXED BY dialogue_room_reading "
         "WHERE stream='chat' AND chat_id=?")
_DOWN = f"ORDER BY {READING_TIME} DESC, {READING_ORDER} DESC, seq DESC"
_UP = f"ORDER BY {READING_TIME}, {READING_ORDER}, seq"
_HORIZON = 1 << 52  # the coordinate just above a frozen room's newest row (a safe integer for every client)
_PIVOT = 1 << 51  # the coordinate of the row a page around it begins its chain at


def _internal(key):
    """A public row key ``[time, order or null, position]`` as the index compares it (no order: -1)."""
    return (key[0], -1 if key[1] is None else key[1], key[2])


def _public(key):
    return [key[0], None if key[1] == -1 else key[1], key[2]]


class ChronologicalSource(HistorySource):
    """The chat stream of one room in reading order; ``keys`` are the cursor's ``chrono`` row keys."""

    top = _HORIZON

    def __init__(self, root, lens, upper=None):
        super().__init__(root, "chat", lens, upper)
        if self.floor:
            raise StoreMigrating("Dialogue history is still importing")  # no time order over unread rows
        self.chat_id = int(dict(lens.keys)["chat_id"])
        self.keys, self.lower_key, self.top_key, self._after = {}, None, None, None

    @property
    def chain(self) -> str:
        """The coverage witness: the frozen end, and the pivot's stored position for a chain begun around it."""
        origin = self.keys.get("origin")
        return f"{_CHAIN_WITNESS}-chrono-{self.upper}" + ("" if origin is None else f"-at-{origin}")

    @property
    def horizon(self) -> int:
        return self.top

    def coordinate(self, entry) -> int:
        return entry["_history_end"] - 1

    def pin(self, entry):
        """Bound the reads around a pivot by its key, older ones below it and newer ones after it, in the
        pivot's own chain."""
        time, order, seq = _internal(entry["_chrono_key"])
        self.keys, self._after = {"before": entry["_chrono_key"], "origin": seq}, (time, order, seq + 1)

    def _seek(self, *, below=None, start=None, lower=None, limit):
        """Up to ``limit`` frozen rows of the room as ``(row, key)``: newest first below the exclusive key
        ``below`` (the newest when None) and not below the inclusive key ``lower``, or oldest first from the
        inclusive key ``start`` (keys as the index compares them). The key's own time and order, then its
        time at other orders, then other times: each an index range, so rows sharing a time or an order are
        read, never stepped through; keys and bodies share one snapshot."""
        frozen = self.upper
        with ds.transaction(self.root, write=False) as conn:
            if start is not None:
                time, order, seq = start
                found = conn.execute(f"{_KEYS} AND {READING_TIME}=? AND {READING_ORDER}=? AND seq>=? AND seq<? "
                                     "ORDER BY seq LIMIT ?", (self.chat_id, time, order, seq, frozen, limit)).fetchall()
                if len(found) < limit:
                    found += conn.execute(f"{_KEYS} AND {READING_TIME}=? AND {READING_ORDER}>? AND seq<? "
                                          f"ORDER BY {READING_ORDER}, seq LIMIT ?",
                                          (self.chat_id, time, order, frozen, limit - len(found))).fetchall()
                if len(found) < limit:
                    found += conn.execute(f"{_KEYS} AND {READING_TIME}>? AND seq<? {_UP} LIMIT ?",
                                          (self.chat_id, time, frozen, limit - len(found))).fetchall()
            elif below is None:
                found = conn.execute(f"{_KEYS} AND seq<? {_DOWN} LIMIT ?", (self.chat_id, frozen, limit)).fetchall()
            else:
                time, order, seq = below
                found = conn.execute(f"{_KEYS} AND {READING_TIME}=? AND {READING_ORDER}=? AND seq<? "
                                     "ORDER BY seq DESC LIMIT ?", (self.chat_id, time, order, min(seq, frozen), limit)).fetchall()
                if len(found) < limit:
                    found += conn.execute(f"{_KEYS} AND {READING_TIME}=? AND {READING_ORDER}<? AND seq<? "
                                          f"ORDER BY {READING_ORDER} DESC, seq DESC LIMIT ?",
                                          (self.chat_id, time, order, frozen, limit - len(found))).fetchall()
                if len(found) < limit:
                    found += conn.execute(f"{_KEYS} AND {READING_TIME}<? AND seq<? {_DOWN} LIMIT ?",
                                          (self.chat_id, time, frozen, limit - len(found))).fetchall()
            keyed = [(record["seq"], (record["time"], record["ord"], record["seq"])) for record in found]
            if lower is not None:
                keyed = [(seq, key) for seq, key in keyed if key >= lower]
            records = {}
            for index in range(0, len(keyed), 900):  # within SQLite's parameter floor
                batch = [seq for seq, _key in keyed[index:index + 900]]
                records.update((record["seq"], record) for record in conn.execute(
                    f"{ds._SELECT} WHERE stream='chat' AND seq IN ({','.join('?' * len(batch))})", batch))
        return [(ds._row(records[seq]), key) for seq, key in keyed]

    def _entry(self, row, key, coordinate):
        from ouroboros.chat_references import projection_fields
        return {**row.row, **projection_fields(row.row), "history_id": row.history_id,
                "history_position": {"source": self.source, "offset": row.seq},
                "_history_end": coordinate + 1, "_chrono_key": _public(key)}

    def _entries(self, rows):
        """An around target: its row at the pivot of the chain it begins (``pin``)."""
        return self._warmed([self._entry(row, reading_key(row.row, row.seq), _PIVOT) for row in rows])

    def gaps(self, lower: int, upper: int) -> set:
        return super().gaps(0, self.upper) if lower < upper else set()

    def recent(self, want, counts):
        return self._page(self.top, None, want, counts, _READ_CEILING_ROWS, window=True)

    def older(self, before, want, counts, ceiling=None):
        """Rows below the cursor's cut (``None``: the frozen present, where a deferral left the newest row)."""
        if not 0 <= before <= self.top:
            raise HistoryCursorError("history_cursor_invalid", 400)
        below = self.keys.get("before")
        if want <= 0 or not before:
            self.top_key = self.lower_key = below
            return [], before, set()
        return self._page(before, below, want, counts, _READ_CEILING_ROWS if ceiling is None else ceiling)

    def _page(self, top, below, want, counts, ceiling, window=False):
        """Rows below the public key ``below``, newest first, each a step below the one after it from ``top``."""
        selected, counted, read, seen, coordinate = [], 0, 0, 0, top
        self.top_key, self.lower_key = below, below  # an empty page replays nothing
        key = None if below is None else _internal(below)
        while True:
            batch = self._seek(below=key, limit=_PROBE_ROWS)
            labelled = [self._entry(row, found, coordinate - step) for step, (row, found) in enumerate(batch, 1)]
            coordinate -= len(batch)
            for entry, (row, _key) in zip(self._warmed(labelled), batch):  # newest first
                selected.append(entry)
                counted += bool(counts(entry))
                read += 1
                seen += row.length
                windowed = not window or read >= _RECENT_WINDOW_ROWS or seen >= _RECENT_WINDOW_BYTES
                if (counted >= want and windowed) or read >= ceiling:
                    self.lower_key = entry["_chrono_key"]
                    if len(batch) < _PROBE_ROWS and entry is labelled[-1]:  # nothing older: the beginning
                        return list(reversed(selected)), 0, self.gaps(0, top)
                    return list(reversed(selected)), self.coordinate(entry), self.gaps(self.coordinate(entry), top)
            if selected:
                self.lower_key = selected[-1]["_chrono_key"]
            if len(batch) < _PROBE_ROWS:  # the room's frozen rows below are exhausted: its beginning
                return list(reversed(selected)), 0, self.gaps(0, top)
            key = _internal(selected[-1]["_chrono_key"])

    def task_pivot(self, task_id, member, in_room=None):
        """The task's earliest row of the room in reading order; its rows are one indexed probe."""
        from ouroboros.chat_references import projection_fields
        self._warmed([{"task_id": task_id}])
        seen, lower, best = False, 0, None
        while True:
            rows = ds.probe_rows(self.root, self.source, [("task_id", task_id)], before=self.upper, lower=lower,
                                 limit=_PROBE_ROWS, newest_first=False)
            for row in rows:
                entry = {**row.row, **projection_fields(row.row), "history_id": row.history_id,
                         "history_position": {"source": self.source, "offset": row.seq}}
                if member(entry):
                    key = reading_key(row.row, row.seq)
                    best = (key, row) if best is None or key < best[0] else best
                else:
                    seen = seen or in_room is None or not in_room(entry)
            if len(rows) < _PROBE_ROWS:
                break
            lower = rows[-1].seq + 1
        if best is not None:
            return "found", self._entries([best[1]])[0]
        return ("moved" if seen else "none"), None

    def _first_from(self, key):
        """The cut at ``key`` by its name, the public key of the first frozen row from it on (``None``: the present)."""
        found = self._seek(start=key, limit=1)
        return _public(found[0][1]) if found else None

    def ahead(self, start, want, counts, ceiling=None):
        """Rows after the pinned key, oldest first, each a step above the one before it from ``start``. The
        page's top cut is named by the first row above its newest (``top_key``; ``None``: the present)."""
        if want <= 0:
            self.top_key = self._first_from(self._after)
            return [], start, set()
        ceiling = _READ_CEILING_ROWS if ceiling is None else ceiling
        key, coordinate, selected, counted = self._after, start, [], 0
        while True:
            batch = self._seek(start=key, limit=_PROBE_ROWS)
            labelled = [self._entry(row, found, coordinate + step) for step, (row, found) in enumerate(batch)]
            coordinate += len(batch)
            for step, entry in enumerate(self._warmed(labelled), 1):
                selected.append(entry)
                counted += bool(counts(entry))
                if counted >= want or len(selected) >= ceiling:
                    time, order, seq = _internal(entry["_chrono_key"])
                    self.top_key = (labelled[step]["_chrono_key"] if step < len(batch) else None
                                    if len(batch) < _PROBE_ROWS else self._first_from((time, order, seq + 1)))
                    return selected, entry["_history_end"], self.gaps(start, entry["_history_end"])
            if len(batch) < _PROBE_ROWS:
                break
            time, order, seq = _internal(selected[-1]["_chrono_key"])
            key = (time, order, seq + 1)
        self.top_key = None  # the frozen present was reached
        return selected, coordinate, self.gaps(start, coordinate)

    def replay(self, lower, before):
        """A frozen page again: the rows between its keys, counted down from ``before`` as first read."""
        if not 0 <= lower <= before:
            raise HistoryCursorError("history_cursor_invalid", 400)
        start, below, found = self.keys.get("lower"), self.keys.get("before"), []
        key, floor = (None if below is None else _internal(below)), (None if start is None else _internal(start))
        while floor is not None:
            batch = self._seek(below=key, lower=floor, limit=_PROBE_ROWS)
            found.extend(batch)
            if len(batch) < _PROBE_ROWS:  # its oldest key, or the room's beginning, was reached
                break
            key = batch[-1][1]
        entries = [self._entry(row, found_key, before - step) for step, (row, found_key) in enumerate(found, 1)]
        self.lower_key, self.top_key = start, self.keys.get("before")
        return list(reversed(self._warmed(entries))), lower, self.gaps(lower, before)
