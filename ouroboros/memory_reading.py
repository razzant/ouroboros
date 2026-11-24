"""A mark of reading follows delivery: what an actor was shown whole or read, and what its acts count as read.

One predicate for every reader (BIBLE P1, "the unread is named"):

- **Recording.** At each accepted model request (``loop._record_transcript_prefix``, after a usable answer) the
  host notes the rows of the actor's own room its captured view held whole: the lane-1 rows the selected forms of
  "This room" did not shorten (``memory_floor.kept_level`` of that request's own projection), as runs over the
  room's open rows, gaps kept. A place arrival drained into the work and covered by an answered send
  (``presence_continuation.covered_send``) and the waking event count by their event id, and the messages of the
  waking event's thread snapshot (``presence_context._thread_snapshot``, whole in every accepted request) by their
  provider message id. An exact read
  (``memory_read(rows=true)``) notes the rows it returned whole, in the room it read. Nothing is read for this:
  the facts are those the request and the read already hold.
- **The stamp.** A work's own room is stamped on its finish (``read_spans``: ``{"room", "spans", "events",
  "messages"}``, disjoint position spans with gaps preserved, never one bound) through the terminal fields; a reply of mine counts what its work's finish
  names. No stamp on a finish written before this rule: such a finish keeps the positional rule (everything at
  or before it), and nothing is migrated.
- **Consideration.** ``Acts`` collects a room's considering acts (an outward reply, a quiet finish, a ``fold_read``
  finish) with their stamps and answers whether a row is considered and by which act first. A row no act names
  stays unread: named by count, owed nothing.
- **Delegated coverage.** ``bound_delegated`` intersects a delegated writer's covers with what it was shown or read
  (the same record), as the fallback writer's ``_trimmed`` keeps only rows it gave whole.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

STAMP = "read_spans"
NOTHING: Dict[str, Any] = {"room": None, "spans": [], "events": [], "messages": []}  # a running work's reply: its finish will say
_RECORD = "_memory_reading"


# --- spans ----------------------------------------------------------------------------------------

def merged(spans: Iterable[Sequence[int]]) -> List[List[int]]:
    """Integer position intervals, overlapping or touching ones joined: no stream position lies between ``b`` and
    ``b + 1``, so a join never names a row neither interval held."""
    out: List[List[int]] = []
    for first, last in sorted((int(a), int(b)) for a, b in spans):
        if out and first <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], last)
        else:
            out.append([first, last])
    return out


def runs(order: Sequence[int], whole: Iterable[int]) -> List[List[int]]:
    """Spans of consecutive ``order`` positions that are all in ``whole``; a shortened row between two ends a run."""
    held, out, start, last = set(whole), [], None, None
    for pos in order:
        if pos in held:
            start, last = (pos if start is None else start), pos
        elif start is not None:
            out.append([start, last])
            start = None
    if start is not None:
        out.append([start, last])
    return out


def inside(spans: Iterable[Sequence[int]], pos: int) -> bool:
    return any(first <= pos <= last for first, last in spans)


# --- recording ------------------------------------------------------------------------------------

def whole_positions(room: Mapping[str, Any], gone: Mapping[str, Iterable[str]]) -> List[int]:
    """Positions of the lane-1 rows ``memory_view._room_text`` draws whole under the selected forms ``gone``:
    a row it turns into an opening, a pointer, a work's gist or an aggregate count is not whole; an immune row is."""
    from ouroboros.memory_view_work import folded_work_rows

    short = {ident.removeprefix("bare:") for step in ("F2", "F7", "F7p") for ident in gone.get(step, ())}
    work, _lines = folded_work_rows(room, gone)
    return [item["pos"] for item in room.get("lane1") or ()
            if item.get("immune") or (item["address"] not in short and item["address"] not in work)]


def _record(ctx: Any) -> Dict[str, Dict[str, Any]]:
    record = getattr(ctx, _RECORD, None)
    if not isinstance(record, dict):
        record = {}
        setattr(ctx, _RECORD, record)
    return record


def _add(ctx: Any, room: str, spans: Iterable[Sequence[int]] = (), events: Iterable[str] = (),
         messages: Iterable[str] = ()) -> None:
    entry = _record(ctx).setdefault(str(room), {"spans": [], "events": [], "messages": []})
    entry["spans"] = merged([*entry["spans"], *spans])
    for key, values in (("events", events), ("messages", messages)):
        entry[key] = sorted({*entry.get(key, ()), *(str(value) for value in values if value)})


def note_read(ctx: Any, room: Any, entries: Sequence[Tuple[Any, Any, int]], *, contiguous: bool) -> None:
    """An exact read returned ``entries`` (``(address, row, pos)``, stored order) whole: one span when they are
    the room's consecutive rows, else one span each (a task or period filter skips rows between them)."""
    if ctx is None or not entries:
        return
    positions = [int(entry[2]) for entry in entries]
    _add(ctx, str(room), [[positions[0], positions[-1]]] if contiguous else [[p, p] for p in positions])


def _arrivals(ctx: Any) -> Tuple[List[str], List[str]]:
    """``(event ids, message ids)``: the waking event and the place arrivals an answered send covered
    (``reply_anchor``'s covered set), and the provider ids of the waking event's thread snapshot messages."""
    metadata = getattr(ctx, "task_metadata", None) or {}
    event = ((metadata.get("presence") or {}).get("event") or {}) if isinstance(metadata, Mapping) else {}
    delivered = getattr(ctx, "_presence_delivered", None) or []
    covered = delivered[:int(getattr(ctx, "_presence_covered", 0) or 0)]
    history = ((event.get("conversation") or {}).get("thread_history") or {}) if isinstance(event, Mapping) else {}
    thread = (history.get("data") or {}).get("messages") if isinstance(history, Mapping) else None
    messages = [str(row.get("message_id") or row.get("id") or "") for row in thread or () if isinstance(row, Mapping)]
    return [str(row["source_event_id"]) for row in [event, *covered] if row.get("source_event_id")], messages


def note_view(ctx: Any, usage: Optional[Dict[str, Any]] = None) -> None:
    """After an accepted request: its own room's rows the request's view held whole, its covered arrivals, then the
    work's stamp in ``usage`` (the terminal fields carry it to the finish row). A work restored without its record
    starts from the stamp its usage kept. Never raises into the loop."""
    try:
        from ouroboros.memory_floor import kept_level
        from ouroboros.memory_view import snapshot_from_json

        plan = getattr(ctx, "context_fit_plan", None)
        core = getattr(plan, "core", None)
        if core is None or not getattr(core, "memory_view_json", ""):
            return
        snapshot = snapshot_from_json(core.memory_view_json)
        room = snapshot.room if isinstance(snapshot.room, dict) else {}
        if not snapshot.spec.room_lanes or not room.get("room_id") or not snapshot.active:
            return
        here = str(room["room_id"])
        kept = (usage or {}).get(STAMP)
        if getattr(ctx, _RECORD, None) is None and isinstance(kept, dict) and str(kept.get("room")) == here:
            _add(ctx, here, kept.get("spans") or (), kept.get("events") or (), kept.get("messages") or ())
        mode = getattr(ctx, "active_context_mode", "") or "max"
        gone = dict(kept_level(snapshot, plan.projection(mode).memory_facts).addressed)
        lane1 = [item["pos"] for item in room.get("lane1") or ()]
        _add(ctx, here, runs(lane1, whole_positions(room, gone)), *_arrivals(ctx))
        if usage is not None:
            usage[STAMP] = stamp(ctx, here)
    except Exception:  # an unrecorded view counts as unread, never as a failed turn
        import logging

        logging.getLogger(__name__).debug("reading record unavailable", exc_info=True)


def stamp(ctx: Any, room: Any) -> Dict[str, Any]:
    entry = _record(ctx).get(str(room)) or {}
    return {"room": str(room), "spans": [list(span) for span in entry.get("spans", ())],
            "events": list(entry.get("events", ())), "messages": list(entry.get("messages", ()))}


def shown(ctx: Any, room: Any) -> Dict[str, Any]:
    """What this actor was shown whole or read of ``room`` so far (``stamp``'s shape)."""
    return stamp(ctx, room)


# --- consideration --------------------------------------------------------------------------------

def event_of(meta: Mapping[str, Any]) -> str:
    from ouroboros.dialogue_provenance import place_row_facts

    return str(place_row_facts(meta).get("source_event_id") or "")


def names(mark: Mapping[str, Any], pos: int, meta: Mapping[str, Any]) -> bool:
    """Whether a stamp names a row: inside one of its spans, or its arrival event or provider message is one the work
    was shown whole (the events and thread snapshot messages are matched only when the stamp lists any)."""
    from ouroboros.dialogue_provenance import place_row_facts

    if inside(mark.get("spans") or (), pos):
        return True
    if mark.get("events") and event_of(meta) in mark["events"]:
        return True
    message = place_row_facts(meta).get("message") if mark.get("messages") else None
    return isinstance(message, Mapping) and str(message.get("message_id") or "") in mark["messages"]


class Acts:
    """A room's considering acts, each ``(position, stamp)``: ``None`` is a finish without a stamp (it considers
    every row at or before it), a stamp names exactly what its work was shown or read here."""

    def __init__(self, room: Any) -> None:
        self.room, self.acts = str(room), []

    def add(self, pos: int, mark: Optional[Mapping[str, Any]]) -> None:
        self.acts.append((int(pos), mark))
        self.acts.sort(key=lambda act: act[0])

    def __bool__(self) -> bool:
        return bool(self.acts)

    def _names(self, mark: Optional[Mapping[str, Any]], pos: int, at: int, meta: Mapping[str, Any]) -> bool:
        if mark is None:
            return pos <= at
        if str(mark.get("room")) != self.room:
            return False
        return names(mark, pos, meta)

    def first(self, entry: Tuple[Any, Mapping[str, Any], int]) -> Optional[int]:
        """The index of the earliest act that considers ``entry``, or None."""
        return next((n for n, (at, mark) in enumerate(self.acts) if self._names(mark, entry[2], at, entry[1])), None)

    def considers(self, entry: Tuple[Any, Mapping[str, Any], int]) -> bool:
        return self.first(entry) is not None


def reply_mark(meta: Mapping[str, Any], finishes: Mapping[str, Mapping[str, Any]],
               running: Iterable[str]) -> Optional[Mapping[str, Any]]:
    """What my outward reply counts as read: its work's finish stamp; nothing while that work runs (its finish will
    say); the positional rule (None) for a reply whose work finished without one."""
    task = str(meta.get("task_id") or "")
    if task and task in finishes:
        return finishes[task]
    return NOTHING if task and task in set(running) else None


# --- delegated coverage ---------------------------------------------------------------------------

class NotShown(ValueError):
    """A delegated writer's covers hold no row it was shown whole or read: ``left`` names what stayed open and why."""

    def __init__(self, left: Mapping[str, Any]) -> None:
        super().__init__("covers name no row this writer was shown whole or read")
        self.left = dict(left)


def bound_delegated(root: Any, shown_of: Callable[[str], Mapping[str, Any]], room: Any, covers: Dict[str, Any],
                    rows: List[Tuple[Dict[str, Any], Dict[str, Any], int]]
                    ) -> Tuple[Dict[str, Any], List[Tuple[Any, Any, int]], Dict[str, Any]]:
    """``(covers, rows, left)``: a delegated writer's covers keep only the rows it was shown whole or read
    (``shown_of(room)``, ``stamp``'s shape; a conversation's rows each in their own room) and none of running work;
    ``left`` names how many stayed open, why, and their first and last address. Notes stay as they were named.
    The fallback writer's covers are the rows it gave whole (``restrict``): one predicate. Nothing left raises
    ``NotShown``."""
    from ouroboros import chat_chain
    from ouroboros.task_store import task_facts

    rooms = {sha: str(each) for each, refs in (covers.get("rooms") or {}).items() for sha in refs}
    busy, kept, unread, live = set(task_facts(root, where="status=?", params=("running",))), [], [], []
    for entry in rows:
        address, meta, pos = entry
        mark = shown_of(rooms.get(address["row_sha256"], str(room)))
        if {str(meta.get(key) or "") for key in ("task_id", "root_task_id")} & busy:
            live.append(entry)
        elif names(mark, pos, meta):
            kept.append(entry)
        else:
            unread.append(entry)
    left = {reason: {"count": len(items), "from": chat_chain.format_address(items[0][0]),
                     "to": chat_chain.format_address(items[-1][0])}
            for reason, items in (("not shown to this writer whole or read", unread), ("running work", live)) if items}
    if not kept:
        raise NotShown(left)
    notes = [ref for ref in covers.get("rows", ()) if str(ref).startswith("note:")]
    return restrict(covers, kept, notes=notes), kept, left


def restrict(covers: Dict[str, Any], kept: List[Tuple[Dict[str, Any], Dict[str, Any], int]], *,
             notes: Sequence[str] = ()) -> Dict[str, Any]:
    """Covers over exactly ``kept`` (``(address, row, pos)``, stored order), and the named notes: the rows, their
    bounds, tasks, times and stream span; a conversation's room map keeps only its kept rows."""
    from ouroboros.chronicle_store import source_time_span

    shas = [entry[0]["row_sha256"] for entry in kept]
    out = {**covers, "rows": shas + list(notes), "note_ids": [str(n)[5:] for n in notes] if notes else [],
           "first": kept[0][0], "last": kept[-1][0], "count": len(kept),
           "task_ids": list(dict.fromkeys(str(e[1].get("task_id")) for e in kept if e[1].get("task_id"))),
           "ts_span": source_time_span(e[1].get("ts") for e in kept), "stream_span": [kept[0][2], kept[-1][2]]}
    if isinstance(covers.get("rooms"), dict):
        held = set(shas)
        out["rooms"] = {room: [ref for ref in refs if ref in held or str(ref).startswith("note:")]
                        for room, refs in covers["rooms"].items()}
        out["rooms"] = {room: refs for room, refs in out["rooms"].items() if refs}
    return out
