"""A mark of reading follows delivery: what an actor was shown whole or read, and what its acts count as read.

One predicate for every reader (BIBLE P1, "the unread is named"):

- **Recording.** At each accepted model request (``loop._record_transcript_prefix``, after a usable answer) the
  host notes, from facts the request already holds:
  - the rows of the work's own room the request's view drew whole: the lane-1 rows the selected forms of "This
    room" did not shorten (``memory_floor.kept_level`` of that request's projection), as runs over the room's rows
    in stored order (a shortened row or a task line between two whole rows ends a run);
  - the waking event, the place arrivals an answered send covered (``presence_continuation.covered_send``) and
    the messages of the waking event's thread snapshot (``presence_context._thread_snapshot``);
  - my human's rows the view drew whole (``owner_words_sources``) and records it drew whole (notes, retold sections,
    story records); words another conversation's block drew whole count nothing: only an act in their room does;
  - exact reads (``memory_read(rows=true)``, ``memory_read(node_id=…)``) whose result this request carried whole:
    a read is staged when its tool runs and credited only once an accepted request delivered it, row by row as the
    delivered text holds each row whole; a row or record too long for one page counts once its windows reach its end.
  The record lives in the work's usage, so it survives a park, a pause and a resume with the work's saved state.
- **The stamp.** A work's finish carries ``read_spans`` (``stamp``: its own room's spans, its exact reads of other
  rooms, the events and thread messages, my human's rows by digest); a work under this rule starts with an empty
  one (``task_results``), so a work that never wrote its own finish counts nothing. A finish written before this
  rule has none and keeps the positional rule.
- **Consideration.** ``Acts`` indexes a room's considering acts (an outward reply, a quiet finish, a ``fold_read``
  finish) once; a question is logarithmic and ``claims`` gives each row its earliest fold in one walk of the folds.
  An act never names a row after its own position. A reply counts its work's finish stamp wherever that finish
  lives (``reply_marks``), nothing while its work has not ended.
- **Delegated coverage** reads the same record (``memory_coverage``).
"""
from __future__ import annotations

import bisect
import hashlib
import logging
import threading
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

STAMP = "read_spans"
RECORD = "memory_reading"  # usage key: kept with the work's saved state (checkpoints, parks, pauses)
NOTHING: Dict[str, Any] = {"room": None, "spans": [], "events": [], "messages": []}  # counts nothing
_LOCK = threading.Lock()
_KEY = 80  # the opening of a staged text, found in the delivered result before its digest is compared
log = logging.getLogger(__name__)


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


def runs(order: Iterable[int], whole: Iterable[int]) -> List[List[int]]:
    """Spans of consecutive positions, in sorted order, that are all in ``whole``; any other position between two
    whole ones ends a run (a display order of imported rows never widens or empties one)."""
    held, out, start, last = set(whole), [], None, None
    for pos in sorted(set(order) | held):
        if pos in held:
            start, last = (pos if start is None else start), pos
        elif start is not None:
            out.append([start, last])
            start = None
    if start is not None:
        out.append([start, last])
    return out


def inside(spans: Sequence[Sequence[int]], pos: int, starts: Optional[Sequence[int]] = None) -> bool:
    """Whether ``pos`` lies in a sorted, merged list of spans (bisect over ``starts``, their first positions)."""
    at = bisect.bisect_right([span[0] for span in spans] if starts is None else starts, pos) - 1
    return at >= 0 and spans[at][0] <= pos <= spans[at][1]


def message_keys(message: Any, conversation: Any = "") -> List[str]:
    """A provider message's identities, as the thread snapshot marks its waking message (``message_id``, ``id``,
    ``ts``), each scoped by its conversation so that two chats' numbers never meet."""
    if not isinstance(message, Mapping):
        return []
    return [f"{conversation}:{value}" for value in (message.get("message_id"), message.get("id"), message.get("ts"))
            if value not in (None, "")]


def sender_of(meta: Mapping[str, Any]) -> str:
    """The work a delivery of mine records as its sender: the row's task, else the delivery's own origin."""
    origin = (meta.get("transport") or {}).get("origin") if isinstance(meta.get("transport"), Mapping) else None
    return str(meta.get("task_id") or (origin.get("task_id") if isinstance(origin, Mapping) else "") or "")


def row_keys(meta: Mapping[str, Any]) -> Tuple[str, List[str]]:
    """``(event id, message keys)`` of a row from its stored facts."""
    from ouroboros.dialogue_provenance import place_row_facts

    facts = place_row_facts(meta)
    return str(facts.get("source_event_id") or ""), message_keys(facts.get("message"), facts.get("conversation_id") or "")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


# --- recording ------------------------------------------------------------------------------------

def whole_positions(room: Mapping[str, Any], gone: Mapping[str, Iterable[str]]) -> List[int]:
    """Positions of the lane-1 rows ``memory_view._room_text`` draws whole under the selected forms ``gone``:
    a row it turns into an opening, a pointer, a work's gist or an aggregate count is not whole; an immune row is."""
    from ouroboros.memory_view_work import folded_work_rows

    short = {ident.removeprefix("bare:") for step in ("F2", "F7", "F7p") for ident in gone.get(step, ())}
    work, _lines = folded_work_rows(room, gone)
    return [item["pos"] for item in room.get("lane1") or ()
            if item.get("immune") or (item["address"] not in short and item["address"] not in work)]


def record_of(ctx: Any, usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The work's reading record, in its usage when it has one (saved with the work), else on the context."""
    holder = usage if isinstance(usage, dict) else getattr(ctx, "_accumulated_usage", None)
    if not isinstance(holder, dict):
        holder = getattr(ctx, "_reading_holder", None)
        if not isinstance(holder, dict):
            holder = {}
            setattr(ctx, "_reading_holder", holder)
    record = holder.get(RECORD)
    if not isinstance(record, dict):
        record = holder[RECORD] = {}
    for key, empty in (("rooms", {}), ("events", []), ("messages", []), ("nodes", []), ("windows", {}),
                       ("pending", [])):
        record.setdefault(key, empty)
    return record


def _add(record: Dict[str, Any], room: Any = None, *, spans=(), read=(), shas=(), events=(), messages=(),
         nodes=()) -> None:
    if room is not None:
        entry = record["rooms"].setdefault(str(room), {})
        entry["spans"] = merged([*entry.get("spans", ()), *spans])
        entry["read"] = merged([*entry.get("read", ()), *read])
        entry["shas"] = sorted({*entry.get("shas", ()), *(str(s) for s in shas if s)})
    for key, values in (("events", events), ("messages", messages), ("nodes", nodes)):
        record[key] = sorted({*record[key], *(str(value) for value in values if value)})


def _staged(line: str) -> Dict[str, Any]:
    return {"key": line[:_KEY], "length": len(line), "sha": _digest(line)}


def note_read(ctx: Any, room: Any, entries: Sequence[Tuple[Any, Any, int, str]], *, contiguous: bool,
              window: Optional[Tuple[Tuple[Any, Any, int], int, int, int, str]] = None) -> None:
    """An exact read produced ``entries`` (``(address, row, pos, rendered line)``, stored order) whole, and perhaps one
    row's character ``window`` (``(entry, begin, end, total, rendered text)``): staged under the running tool call
    until an accepted request delivers it (``note_view``). ``contiguous``: the room's consecutive rows."""
    from ouroboros.tool_call_log import current_tool_call_id

    if ctx is None or not (entries or window):
        return
    staged = {"call": current_tool_call_id(), "room": str(room), "contiguous": contiguous,
              "rows": [[int(entry[2]), str(entry[0]["row_sha256"]), _staged(entry[3])] for entry in entries]}
    if window is not None:
        entry, begin, end, total, text = window
        staged["window"] = [int(entry[2]), str(entry[0]["row_sha256"]), begin, end, total, _staged(text)]
    with _LOCK:
        record_of(ctx)["pending"].append(staged)


def note_node(ctx: Any, node_id: Any, text: str, start: int) -> None:
    """An exact read of one record returned ``text`` from character ``start``: staged until an accepted request
    delivers it whole. Its continuation is the result's window line (``result_window``: the first line ``chars a–b of
    n; next_start=N`` or ``…; complete``), never a word of the body."""
    import re
    from ouroboros.tool_call_log import current_tool_call_id

    if ctx is None or not node_id or not text:
        return
    found = re.search(r"^chars \d+–\d+ of \d+; (?:next_start=(\d+)|complete)", text, re.M)
    found = found if found and found.group(1) else None
    with _LOCK:
        record_of(ctx)["pending"].append({"call": current_tool_call_id(), "node": str(node_id), "text": _staged(text),
                                          "begin": int(start or 0), "next": int(found.group(1)) if found else None})


def _delivered(content: str, staged: Mapping[str, Any]) -> bool:
    """Whether ``content`` holds the staged text whole: its opening found, the slice from there matching its digest."""
    start = content.find(staged["key"])
    while start >= 0:
        if _digest(content[start:start + staged["length"]]) == staged["sha"]:
            return True
        start = content.find(staged["key"], start + 1)
    return False


def _tool_texts(messages: Sequence[Mapping[str, Any]], calls: Iterable[str]) -> Dict[str, str]:
    """The text of the tool results of ``calls`` this request carried, and of no other."""
    from ouroboros.loop_messages import _extract_plain_text_from_content

    wanted = set(calls)
    return {str(m.get("tool_call_id")): _extract_plain_text_from_content(m.get("content"))
            for m in messages or () if isinstance(m, Mapping) and m.get("role") == "tool"
            and str(m.get("tool_call_id") or "") in wanted}


def _promote(record: Dict[str, Any], messages: Sequence[Mapping[str, Any]]) -> None:
    """Credit each staged read this request carried: only the rows and records its delivered text held whole."""
    if not record["pending"]:
        return
    texts, left = _tool_texts(messages, {str(staged.get("call") or "") for staged in record["pending"]}), []
    for staged in record["pending"]:
        content = texts.get(staged.get("call") or "")
        if content is None:
            left.append(staged)  # not delivered yet: a later accepted request may carry it
            continue
        if staged.get("node") and _delivered(content, staged["text"]):
            got = record["windows"].setdefault("node:" + staged["node"], {"got": [], "end": None})
            if staged.get("next") is None:
                got["end"] = staged.get("begin", 0)  # the window that reached the record's end
            else:
                got["got"] = merged([*got["got"], [staged.get("begin", 0), max(staged.get("begin", 0), staged["next"] - 1)]])
            end = got["end"]
            if end is not None and (end == 0 or got["got"] and got["got"][0][0] == 0 and got["got"][0][1] >= end - 1):
                _add(record, nodes=[staged["node"]])
                record["windows"].pop("node:" + staged["node"], None)
        if staged.get("node"):
            continue
        whole = [pos for pos, _sha, text in staged["rows"] if _delivered(content, text)]
        if staged.get("window"):
            pos, sha, begin, end, total, text = staged["window"]
            if _delivered(content, text):
                got = record["windows"].setdefault(sha, {"pos": pos, "room": staged["room"], "total": total, "got": []})
                got["got"] = merged([*got["got"], [begin, max(begin, end - 1)]])
                if total == 0 or got["got"][0][0] == 0 and got["got"][0][1] >= total - 1:
                    whole.append(pos)
                    record["windows"].pop(sha, None)
        order = [pos for pos, _sha, _text in staged["rows"]]
        spans = runs(order, whole) if staged["contiguous"] else [[pos, pos] for pos in whole]
        _add(record, staged["room"], read=spans)
    record["pending"] = left


def _arrivals(ctx: Any) -> Tuple[List[str], List[str]]:
    """``(event ids, message keys)``: the waking event and the place arrivals an answered send covered
    (``reply_anchor``'s covered set), and the waking event's thread snapshot messages."""
    metadata = getattr(ctx, "task_metadata", None) or {}
    event = ((metadata.get("presence") or {}).get("event") or {}) if isinstance(metadata, Mapping) else {}
    delivered = getattr(ctx, "_presence_delivered", None) or []
    covered = delivered[:int(getattr(ctx, "_presence_covered", 0) or 0)]
    history = ((event.get("conversation") or {}).get("thread_history") or {}) if isinstance(event, Mapping) else {}
    thread = (history.get("data") or {}).get("messages") if isinstance(history, Mapping) else None
    keys = [key for row in thread or () for key in message_keys(row, event.get("conversation_id") or "")]
    return [str(row["source_event_id"]) for row in [event, *covered] if row.get("source_event_id")], keys


def _drawn_records(snapshot: Any, gone: Mapping[str, Sequence[str]]) -> List[str]:
    """Record ids the view drew whole: unsealed notes, retold sections and pages of the room, story records."""
    room = snapshot.room if isinstance(snapshot.room, dict) else {}
    notes = [] if gone.get("F8a") else [n["id"] for n in room.get("notes") or () if n["id"] not in gone.get("F8", ())]
    retold = [r["id"] for r in [*room.get("legacy", ()), *room.get("under_parts", ())]
              if r["id"] not in gone.get("F4", ()) and not r.get("standing")]
    shortened, counted_rooms = {*gone.get("F5", ()), *gone.get("F5l", ())}, set(gone.get("F3", ()))
    story = [] if gone.get("F5a") else [r["id"] for r in snapshot.story if r.get("id") and r["id"] not in shortened
                                        and not r.get("told_by")  # told through an account: its words are absent
                                        and not (r.get("kind") == "legacy" and str(r.get("room_id")) in counted_rooms)]
    return [*notes, *retold, *story]


def note_view(ctx: Any, usage: Optional[Dict[str, Any]] = None, messages: Optional[Sequence[Any]] = None) -> None:
    """After an accepted request: credit the reads it delivered and what its view drew whole, then keep the work's
    stamp in ``usage`` (the terminal fields carry it to the finish row). A failure leaves an empty stamp (it counts
    nothing) and a warning, never the positional rule."""
    from ouroboros.memory_floor import kept_level
    from ouroboros.memory_view import snapshot_from_json

    holder = usage if isinstance(usage, dict) else getattr(ctx, "_accumulated_usage", None)
    with _LOCK:
        record, here = record_of(ctx, usage), None
        try:
            _promote(record, messages or ())
            plan = getattr(ctx, "context_fit_plan", None)
            core = getattr(plan, "core", None)
            json_text = getattr(core, "memory_view_json", "") if core is not None else ""
            snapshot = snapshot_from_json(json_text) if json_text else None
            if snapshot is not None and snapshot.active:
                gone = dict(kept_level(snapshot, plan.projection(getattr(ctx, "active_context_mode", "") or "max")
                                       .memory_facts).addressed)
                room = snapshot.room if isinstance(snapshot.room, dict) else {}
                for source in (snapshot.owner_words_sources or {}).get("rows") or ():
                    if source.get("row_sha256") and source.get("room_id") is not None:
                        _add(record, source["room_id"], shas=[source["row_sha256"]])
                events, keys = _arrivals(ctx)  # the waking thread is in the request whether or not lanes are
                _add(record, nodes=_drawn_records(snapshot, gone), events=events, messages=keys)
                if snapshot.spec.room_lanes and room.get("room_id"):
                    here = str(room["room_id"])
                    order = [item["pos"] for item in room.get("lane1") or ()] + [
                        pos for item in room.get("lane2") or () for pos in (item.get("pos"), item.get("last_pos"))
                        if isinstance(pos, int)]
                    _add(record, here, spans=runs(order, whole_positions(room, gone)))
            record["here"] = here or record.get("here")
            if isinstance(holder, dict):
                holder[STAMP] = stamp(record)
        except Exception:  # an unrecorded view counts nothing; it never falls back to the positional rule
            log.warning("reading record unavailable; this request's view counts nothing", exc_info=True)
            if isinstance(holder, dict):
                holder.setdefault(STAMP, dict(NOTHING))


def stamp(record: Mapping[str, Any]) -> Dict[str, Any]:
    """The finish's stamp: its own room's spans (shown or read), its exact reads of other rooms (``reads``), my
    human's rows by digest, the events and thread messages."""
    here = record.get("here")
    entry = (record.get("rooms") or {}).get(str(here)) or {} if here else {}
    out = {"room": here, "spans": merged([*entry.get("spans", ()), *entry.get("read", ())]),
           "events": list(record.get("events") or ()), "messages": list(record.get("messages") or ())}
    if entry.get("shas"):
        out["shas"] = list(entry["shas"])
    reads = {room: list(other["read"]) for room, other in (record.get("rooms") or {}).items()
             if room != str(here) and other.get("read")}
    if reads:
        out["reads"] = reads
    return out


def finish_stamp(usage: Mapping[str, Any]) -> Dict[str, Any]:
    """The stamp a finish carries: the work's own, else an explicit empty one (a work under this rule that recorded
    nothing counts nothing; only a finish written before the rule has no stamp)."""
    kept = usage.get(STAMP)
    return dict(kept) if isinstance(kept, Mapping) else dict(NOTHING)


# --- consideration --------------------------------------------------------------------------------

class _Index:
    """One room's acts, indexed once: the last unstamped position (the positional rule), merged spans bounded by
    each act's own position (its own room's, and its exact reads of this room), and the newest act naming each event,
    message and digest."""

    def __init__(self, room: str, acts: Sequence[Tuple[int, Optional[Mapping[str, Any]]]]) -> None:
        self.legacy = max((at for at, mark in acts if mark is None), default=-1)
        spans, self.events, self.messages, self.shas = [], {}, {}, {}
        for at, mark in acts:
            if mark is None:
                continue
            for key, values in ((self.events, mark.get("events")), (self.messages, mark.get("messages"))):
                for value in values or ():
                    key[value] = max(key.get(value, -1), at)
            for a, b in act_spans(room, mark):
                if a <= at:
                    spans.append([a, min(b, at)])
            for sha in (mark.get("shas") or ()) if str(mark.get("room")) == room else ():
                self.shas[sha] = max(self.shas.get(sha, -1), at)
        self.spans = merged(spans)
        self.starts = [span[0] for span in self.spans]
        self.keyed = bool(self.events or self.messages)

    def names(self, entry: Tuple[Any, Mapping[str, Any], int], keys: Callable[[], Tuple[str, List[str]]]) -> bool:
        pos = entry[2]
        if pos <= self.legacy or inside(self.spans, pos, self.starts) or (
                self.shas.get(str(entry[0].get("row_sha256")), -1) >= pos):
            return True
        if not self.keyed:
            return False
        event, messages = keys()
        return (bool(event) and self.events.get(event, -1) >= pos) or any(
            self.messages.get(key, -1) >= pos for key in messages)


def act_spans(room: str, mark: Mapping[str, Any]) -> List[Sequence[int]]:
    """A stamp's spans in ``room``: its own room's, else the exact reads it made of that room."""
    if str(mark.get("room")) == room:
        return list(mark.get("spans") or ())
    return list(((mark.get("reads") or {}).get(room)) or ())


class Acts:
    """A room's considering acts, each ``(position, stamp)``: ``None`` is a finish written before stamps (it counts
    every row at or before it); a stamp names exactly what its work was shown or read, never a row after the act.
    The index is built once at the first question; an event or a thread message counts in any room of its act."""

    def __init__(self, room: Any) -> None:
        self.room, self.acts, self._all = str(room), [], None
        self._keys: Dict[int, Tuple[str, List[str]]] = {}
        self.ops = 0  # rows a fold walk examined (``claims``): flat in the number of folds a row misses

    def add(self, pos: int, mark: Optional[Mapping[str, Any]]) -> None:
        self.acts.append((int(pos), mark))
        self._all = None

    def __bool__(self) -> bool:
        return bool(self.acts)

    def _row_keys(self, entry: Tuple[Any, Mapping[str, Any], int]) -> Callable[[], Tuple[str, List[str]]]:
        def keys() -> Tuple[str, List[str]]:
            if id(entry[1]) not in self._keys:
                self._keys[id(entry[1])] = row_keys(entry[1])
            return self._keys[id(entry[1])]
        return keys

    def considers(self, entry: Tuple[Any, Mapping[str, Any], int]) -> bool:
        if self._all is None:
            self._all = _Index(self.room, self.acts)
        return self._all.names(entry, self._row_keys(entry))

    def claims(self, entries: Sequence[Tuple[Any, Mapping[str, Any], int]]) -> Dict[int, int]:
        """``{entry index: act index}``: for each entry some act names, the earliest such act by position, in one walk
        of the acts over the entries' sorted positions and key maps (an act claims only rows at or before it)."""
        named = [n for n, entry in enumerate(entries) if self.considers(entry)]  # the room's index rules out misses
        if not named:
            return {}
        order = sorted(named, key=lambda n: entries[n][2])
        positions = [entries[n][2] for n in order]
        keyed = any(mark and (mark.get("events") or mark.get("messages")) for _at, mark in self.acts)
        by_key: Dict[str, List[int]] = {}
        for n in named if keyed else ():
            event, messages = self._row_keys(entries[n])()
            for key in ([("e", event)] if event else []) + [("m", m) for m in messages]:
                by_key.setdefault(key[0] + ":" + key[1], []).append(n)
        claimed: Dict[int, int] = {}
        skip = list(range(len(order) + 1))  # the next unclaimed sorted index: each row is claimed by a span once

        def free(k: int) -> int:
            while skip[k] != k:
                skip[k] = skip[skip[k]]
                k = skip[k]
            return k

        for act in sorted(range(len(self.acts)), key=lambda k: self.acts[k][0]):
            at, mark = self.acts[act]
            if mark is None:
                spans: List[Sequence[int]] = [[-1, at]]
            else:
                spans = [[a, min(b, at)] for a, b in act_spans(self.room, mark) if a <= at]
            for a, b in spans:
                k, end = free(bisect.bisect_left(positions, a)), bisect.bisect_right(positions, b)
                while k < end:
                    self.ops += 1
                    claimed.setdefault(order[k], act)
                    skip[k] = k + 1
                    k = free(k + 1)
            for key in [*("e:" + e for e in (mark or {}).get("events") or ()),
                        *("m:" + m for m in (mark or {}).get("messages") or ())]:
                for n in by_key.get(key, ()):
                    self.ops += 1
                    if entries[n][2] <= at:
                        claimed.setdefault(n, act)
        return claimed


def reply_marks(root: Any, tasks: Iterable[str]) -> Dict[str, Optional[Mapping[str, Any]]]:
    """What my replies count as read, for works whose finish row is not in hand: the stamp their stored result keeps;
    ``NOTHING`` while a work has not ended; the positional rule (None) for a work that ended before stamps or is
    not in the store. One indexed read of those works' status and stamp."""
    import json

    from ouroboros.store import open_store
    from ouroboros.task_status import FINAL_STATUSES

    ids = sorted({str(task) for task in tasks if task})
    if not ids:
        return {}
    found = {}
    with open_store(root, write=False) as txn:
        for task_id, status, mark in txn.conn.execute(
                "SELECT t.task_id, t.status, CASE WHEN json_valid(b.body) THEN json_extract(b.body, '$.read_spans') END "
                "FROM tasks AS t LEFT JOIN task_bodies AS b ON b.task_id = t.task_id "
                "WHERE t.task_id IN (SELECT value FROM json_each(?))", (json.dumps(ids),)):
            stored = json.loads(mark) if isinstance(mark, str) and mark else None
            found[str(task_id)] = (NOTHING if str(status or "") not in FINAL_STATUSES  # not ended: whatever is stored
                                   else stored if isinstance(stored, dict) else None)
    return {task: found.get(task) for task in ids}
