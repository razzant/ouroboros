"""What of my memory is still open and what is folded: one computation for every reader.

The memory view and the memory writers read these facts here and never compute
them a second time:

- An **open row** of a room is a chat row of that room after the legacy frontier
  (``chronicle_import.legacy_frontier``) that no acting page seals
  (``ChronicleStore.sealed_row_refs``). Under an ``unknown`` frontier the boundary
  is the chain end observed at activation. One row can be open in two rooms (the
  owner message that started a Project stays in Main too).
- A **legacy unit** is one ``legacy`` record (one block's section for one room, an
  import gap or the flat file) or a journal ``gap``. Its rows are its positional
  ``raw_range`` within its room. It is **folded** when it is a member of an acting
  part, or when it has rows and every one of them is sealed. A unit without rows (an
  unknown range, a gap, the mixed pseudo-room ``legacy``, a room with no row in its
  block) folds only through a part: an empty set is never "all covered". A journal
  ``gap`` (a torn line of the chronicle itself) is not a member the store folds, so it
  stays a pointer: readable and correctable, never folded.
- A **period** (a block of the old memory) is folded when every unit of it is.
- A **story record's period and place** (``record_period``) are read from its room's own rows:
  a page's covers, a retold record's rows (without rows, its block's recorded period, labelled),
  a part's union over its members, an account's union over its sources (several rooms: their
  rows' span, which is not every event between and seals no row). The aggregate a part
  records is historical and not shown.
- The **standing facts** every reader prints are counted here once: the story's pages and parts
  (``story_counts``) and one room's open rows by lane, their size and times, its unsealed notes,
  its pages and the last row they cover (``room_facts``).
- An **open segment** is the oldest run of a room's open rows (after the frontier, or within
  one legacy unit's range) with no sealed row inside: what one page may seal without being
  refused ``already_sealed``. The sealed set is taken away before the choice.
- A **shortage** is the view's floor fact in a task's trace (``VIEW_TRACE_KEY``) read as a
  writer's cue: open rows of a room shown only by address, or old narrative shown only by
  pointer because the window could not hold it.
- The **journal changes** since an accepted wake are the records written after its accepted
  sequence, one line of text each with the exact read of the record and of its target.

Rows are addressed in ``state/dialog.sqlite`` by the legacy frontier's durable
stream position. A room query selects only that room's indexed candidates and
then applies current membership, so a later binding still moves earlier rows.
Whole-mind inventory reads the facts of every row after the frontier stored
beside the chronicle (``_tail``) plus the rows appended since; membership and
seals are applied at read. No reader walks JSONL or keeps a process-local copy.

Nothing here publishes a record or calls a model.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Tuple

from ouroboros import chat_chain
from ouroboros.chronicle_import import legacy_frontier
from ouroboros.chronicle_store import ChronicleStore, capture, capture_memo, source_time_span
from ouroboros.dialogue_provenance import in_reading_order, rooms_read_by_time, row_class, row_provenance

# The key of the memory view's last-capture fact in a task's ``llm_trace``.
VIEW_TRACE_KEY = "memory_view"

# The row fields the room and attribution readers consult (``project_dialogue.room_membership``,
# ``dialogue_provenance.row_author``) plus the host facts of a task row, its recorded finality and
# authority included. Texts and bulky payloads (review projections, outcome axes, quiz cards) stay
# in the chain, read by address. Facts stored before a field joined this list keep their old form
# (the stored prefix is never rewritten for it): a reader shows such a field as unknown.
_META_FIELDS = ("chat_id", "ts", "type", "direction", "task_id", "parent_task_id", "root_task_id",
                "client_message_id", "subagent_task_id", "delegation_role", "subagent_role", "initiator",
                "summary_kind", "sender_label", "username", "author", "source", "transport",
                "status", "outcome", "outcome_phase", "reason_code", "outcome_final", "outcome_authority",
                "result_ref", "presence", "presence_provenance", "host_fact", "historical_import", "source_order", "terminal_origin")
_MEMBERSHIP_FIELDS = ("chat_id", "ts", "type", "task_id", "parent_task_id", "root_task_id", "source_keys")
_REPLY_FIELDS = ("address", "chat_id", "task_id", "role", "target_home", "unresolved")
# Bumped when ``row_meta`` gains a fact membership reads: a stored prefix of another version is
# another source, read live until the maintenance job ``memory_tail`` refills it.
_META_VERSION = 4

Entry = Tuple[Dict[str, Any], Dict[str, Any], int]


# --- rows and rooms ----------------------------------------------------------------------------

def row_meta(row: Mapping[str, Any]) -> Dict[str, Any]:
    """A chat row without its text: the fields room membership and attribution read.

    ``text_chars`` is the text's length; an inbound row also carries ``text_sha256``
    and ``source_keys`` (its owner-message identities, the keys a Project binding's
    ``source_ref`` matches), so membership needs no text.
    """
    meta = {key: row[key] for key in _META_FIELDS if key in row}
    meta["text_chars"] = len(str(row.get("text") or ""))
    if row.get("wake_silent") is True:
        from ouroboros.task_finalization import quiet_finish

        # Canonical terminal projections already carry the verdict before they drop delivery fields.
        meta["quiet_finish"] = (row["quiet_effective"] is True if "quiet_effective" in row else quiet_finish(dict(row)))
    quiz = row.get("quiz") if isinstance(row.get("quiz"), dict) else {}
    if isinstance(contact := row.get("presence_contact") or quiz.get("presence_contact"), dict):
        meta["presence_contact"] = dict(contact)  # ``presence_owner_contact.private_main_row`` reads it
    if isinstance(ref := row.get("reply_to"), dict):  # ``room_membership.reply_target_room`` reads these, a
        meta["reply_to"] = {key: ref[key] for key in _REPLY_FIELDS if key in ref}  # saved target by its address
    if str(row.get("direction") or "") == "in":
        from ouroboros.project_dialogue import _entry_source_identities, _text_sha256  # D15->D17 is lazy-only

        meta["text_sha256"] = _text_sha256(row.get("text"))
        meta["source_keys"] = tuple(sorted(_entry_source_identities(row)))
    return meta


class MembershipFacts(NamedTuple):
    """The registry facts room membership reads, taken once per call; ``rooms`` is
    the room rule over the ``bindings`` snapshot the fingerprint names."""
    project_chat_ids: FrozenSet[int]
    bindings: Dict[str, int]
    projects_by_source_key: Dict[Tuple[Any, ...], FrozenSet[int]]
    rooms: Any

    def fingerprint(self) -> str:
        payload = json.dumps([sorted(self.project_chat_ids), sorted(self.bindings.items()),
                              sorted((list(key), sorted(chats)) for key, chats in self.projects_by_source_key.items())],
                             ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def membership_facts(root: Any) -> MembershipFacts:
    """Reserved Project rooms, task bindings and the bound owner messages, as ``iter_room_rows`` reads them;
    every reader of one capture (``chronicle_store.capture``) takes the same facts."""
    return capture_memo(root, "membership", lambda: _membership_facts(root))


def _membership_facts(root: Any) -> MembershipFacts:
    from ouroboros.project_dialogue import _source_ref_identity  # D15->D17 is lazy-only
    from ouroboros.projects_registry import all_task_bindings, list_reserved_projects, project_task_bindings, RoomResolver

    projects = frozenset(int(project["chat_id"]) for project in list_reserved_projects(root))
    by_key: Dict[Tuple[Any, ...], set] = {}
    for binding in project_task_bindings(root).values():
        try:
            chat = int(binding.get("project_chat_id") or 0)
        except (TypeError, ValueError):
            continue
        ref = binding.get("source_ref")
        key = _source_ref_identity(ref) if chat in projects and isinstance(ref, dict) and ref else None
        if key is not None:
            by_key.setdefault(key, set()).add(chat)
    return MembershipFacts(projects, dict(all_task_bindings(root)),
                           {key: frozenset(chats) for key, chats in by_key.items()}, RoomResolver(root))


def rooms_of_row(meta: Mapping[str, Any], facts: MembershipFacts) -> FrozenSet[str]:
    """Every room ``room_membership.row_rooms`` puts this row in, as room ids, plus the Projects
    whose binding names it as their origin owner message (it also stays where it was written)."""
    from ouroboros.room_membership import binding_origin_allowed, row_rooms  # D15->D17 is lazy-only

    chat = chat_chain._row_chat_id(meta)
    rooms = set(row_rooms(chat, meta, facts.project_chat_ids, facts.rooms))
    if binding_origin_allowed(chat, meta):
        for key in meta.get("source_keys") or ():
            rooms.update(facts.projects_by_source_key.get(tuple(key), ()))
    return frozenset(str(room) for room in rooms)


# --- the open-row projection --------------------------------------------------------------------

class Tail(NamedTuple):
    """Every chat row after the legacy frontier, ``(pos, address, row_meta, terminal_status)`` in append order;
    ``complete`` is False while the stored facts are not complete for this dialogue store (rows read live)."""
    rows: List[Tuple[int, Dict[str, Any], Dict[str, Any], Optional[str]]]
    complete: bool


def terminal_status(row: Mapping[str, Any]) -> Optional[str]:
    """A task's typed terminal row (a host summary or its final report): its status, ``""`` unrecorded; else None."""
    host = row.get("type") in ("task_summary", "project_completion_summary")
    final = row.get("direction") == "out" and not row.get("is_progress") and (
        row.get("terminal_origin") or row.get("task_terminal_status") or row.get("task_phase") == "finalizing")
    return str(row.get("status") or row.get("task_terminal_status") or "") if host or final else None


def _tail_source(root: Any, frontier: Mapping[str, Any]) -> Dict[str, Any]:
    """The dialogue file after the frontier, with its text-revision generation.

    Appends preserve the prefix; a corrected earlier row changes its content address
    without moving the tail anchor, so the stored facts must be projected again.
    """
    from ouroboros import dialogue_store as ds
    from ouroboros.store import file_identity

    return {"store": list(file_identity(ds.path(root)) or ()), "frontier": int(frontier["pos"]),
            "text_revision": int(ds.meta(root, "chat_text_revision") or 0), "meta": _META_VERSION}


def _projected(entries: Iterable[Any]) -> List[List[Any]]:
    """Rows read now (``chat_chain.iter_stream``), stored form: ``[pos, seq, row_sha256, legacy, meta, terminal]``."""
    return [[pos, entry.seq, chat_chain.source_row_id(entry.row), None if legacy is None else list(legacy),
             row_meta(entry.row), terminal_status(entry.row)] for entry, legacy, pos in entries]


def _entry(fact: List[Any]) -> Tuple[int, Dict[str, Any], Dict[str, Any], Optional[str]]:
    """``(pos, address, meta, terminal)`` of a stored or just projected row: the address its live read gives."""
    pos, seq, sha, legacy, meta, terminal = fact
    meta = dict(meta)
    if "source_keys" in meta:  # stored as JSON lists; ``row_meta`` gives tuples
        meta["source_keys"] = tuple(tuple(key) for key in meta["source_keys"])
    return pos, chat_chain.address_at(meta, seq, legacy, sha=sha), meta, terminal


def _anchored(root: Any, state: Mapping[str, Any]) -> bool:
    """Whether the stored prefix's last row is still the dialogue store's row at its coordinate."""
    from ouroboros import dialogue_store as ds

    anchor = state.get("anchor")
    row = ds.row_at(root, "chat", anchor[0]) if anchor else None
    return anchor is None or (row is not None and chat_chain.source_row_id(row.row) == anchor[1])


def _tail(root: Any, frontier: Mapping[str, Any]) -> Tail:
    """Every chat row after the frontier, read once per capture: the facts stored beside the chronicle
    (``ChronicleStore.tail_facts``), then the rows appended since, read live from the stored prefix's last
    row, which must still answer (its anchor), projected, and stored by this reader. Until a complete prefix
    stands for this dialogue store and frontier (an update, a rebuilt index, a replaced store) the rows are
    read live, today's cost, and the maintenance job fills it (``fill_tail``)."""
    return capture_memo(root, "tail", lambda: _read_tail(root, frontier))


def _read_tail(root: Any, frontier: Mapping[str, Any], *, terminal_only: bool = False) -> Tail:
    """``_tail``'s read; ``terminal_only`` keeps the rows with a terminal status (the stored ones read alone)."""
    from ouroboros import dialogue_store as ds

    store, source = ChronicleStore(root), _tail_source(root, frontier)
    end = ds.stream_end(root, "chat")
    held = store.tail_facts(source, terminal_only=terminal_only)
    if held is not None:
        state, stored = held
        anchor = state.get("anchor")
        live = list(chat_chain.iter_stream(root, lower=anchor[0] if anchor else state["upper"], upper=end))
        first = live[0][0] if live else None
        if anchor is None or (first is not None and [first.seq, chat_chain.source_row_id(first.row)] == anchor):
            gap = _projected(live[1:] if anchor else live)
            if state.get("complete") and end > state["upper"]:  # one zero-wait attempt; a busy lock leaves it
                store.extend_tail(state, gap, {**state, "upper": end, "anchor": gap[-1][1:3] if gap else anchor})
            return Tail([_entry(fact) for fact in stored + gap if not terminal_only or fact[5] is not None],
                        bool(state.get("complete")))
    lower = chat_chain.position_boundary(root, source["frontier"])
    return Tail([_entry(fact) for fact in _projected(chat_chain.iter_stream(root, lower=lower, upper=end))
                 if not terminal_only or fact[5] is not None], False)


_FILL_ROWS = 2_000  # rows per index transaction: bounds one publication-lock hold, so publishers interleave


def tail_owed(root: Any) -> bool:
    """Whether the stored tail facts wait for the fill: the chronicle is active and no complete prefix stands
    for this dialogue store and frontier (none, unfinished, another source, or an anchor that no longer answers)."""
    store = ChronicleStore(root)
    frontier = _active_frontier(store)
    if frontier is None:
        return False
    state = store.tail_state()
    return not (isinstance(state, dict) and state.get("complete")
                and state.get("source") == _tail_source(root, frontier) and _anchored(root, state))


def fill_tail(root: Any, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """The maintenance job ``memory_tail``: store the facts of every chat row after the legacy frontier,
    ``_FILL_ROWS`` an index transaction, resuming an unfinished prefix of this source; a turn never pays it."""
    from ouroboros import dialogue_store as ds
    from ouroboros.store import StoreMigrating

    store = ChronicleStore(root)
    frontier = _active_frontier(store)
    if frontier is None:
        return {"state": "done", "items": 0}
    try:
        source, end, state = _tail_source(root, frontier), ds.stream_end(root, "chat"), store.tail_state()
        resume = isinstance(state, dict) and state.get("source") == source and _anchored(root, state)
        if resume and state.get("complete"):
            return {"state": "done", "items": 0}
        anchor = state.get("anchor") if resume else None
        start = ((anchor[0] if anchor else state["upper"]) if resume
                 else chat_chain.position_boundary(root, source["frontier"]))
        skip = start if anchor else None  # the stored prefix's last row, read again, stored once
        expected, replace, chunk, items = state, not resume, [], 0
        for read in chat_chain.iter_stream(root, lower=start, upper=end):
            if read[0].seq == skip:
                continue
            chunk.append(read)
            if len(chunk) == _FILL_ROWS:
                rows = _projected(chunk)
                anchor, new = rows[-1][1:3], {"source": source, "upper": chunk[-1][0].end, "complete": False}
                if not store.extend_tail(expected, rows, {**new, "anchor": anchor}, replace=replace, wait_sec=10):
                    return {"state": "pending", "items": items}
                expected, replace, chunk, items = {**new, "anchor": anchor}, False, [], items + len(rows)
                if should_stop():
                    return {"state": "pending", "items": items}
        rows = _projected(chunk)
        done = {"source": source, "upper": end, "anchor": rows[-1][1:3] if rows else anchor, "complete": True}
        if not store.extend_tail(expected, rows, done, replace=replace, wait_sec=10):
            return {"state": "pending", "items": items}
        return {"state": "done", "items": items + len(rows)}
    except StoreMigrating:  # positions settle with the archive import: the job before this one
        return {"state": "pending", "items": 0}


def _projection_rows(root: Any, frontier: Mapping[str, Any]) -> List[Entry]:
    return [(address, meta, pos) for pos, address, meta, _terminal in _tail(root, frontier).rows]


def terminal_rows(root: Any) -> List[Tuple[Dict[str, Any], Dict[str, Any], int, str]]:
    """``(address, row_meta, pos, status)`` of every typed terminal row after the legacy frontier, in append order."""
    frontier = _active_frontier(ChronicleStore(root))
    if frontier is None:
        return []
    # A capture whose view reads every row reads the whole tail once for both readers; else these rows alone.
    tail = (_tail(root, frontier) if capture_memo(root, "tail_scope", lambda: None) == "all" else
            capture_memo(root, "terminal_tail", lambda: _read_tail(root, frontier, terminal_only=True)))
    return [(address, meta, pos, terminal) for pos, address, meta, terminal in tail.rows if terminal is not None]


def _authority(store: ChronicleStore) -> int:
    """The chronicle authority the facts read now reflect: the append-only journal's size projects one index."""
    return store.journal_size()


def _active_frontier(store: ChronicleStore) -> Optional[Dict[str, Any]]:
    """The legacy frontier, or ``None`` before activation (a reader never creates the chronicle)."""
    if not store.log_path.exists():
        return None
    frontier = legacy_frontier(store)
    return frontier if type(frontier.get("pos")) is int else None


def rows_after_frontier(root: Any) -> List[Entry]:
    """``(address, row_meta, pos)`` of every chat row after the legacy frontier, in append order.

    Empty before activation: without the frontier nothing is known to be open.
    """
    frontier = _active_frontier(ChronicleStore(root))
    return [] if frontier is None else _projection_rows(root, frontier)


def open_rows_by_room(root: Any, gaps: Optional[Dict[str, set]] = None,
                      quiet: Optional[Dict[str, List[Entry]]] = None) -> Dict[str, List[Entry]]:
    """The open rows of every room that has one: after the frontier, outside the room's sealed set, in
    reading order (``open_room_rows``). ``gaps`` collects, by room, what its rows may lack: a Reply whose
    saved target cannot be read is Main's here and a gap of each Project it could join
    (``room_membership.unknown_target_rooms``). ``quiet`` collects typed quiet finishes, including sealed ones,
    from the same tail; they can consider earlier human words for liveness, never seal them."""
    store = ChronicleStore(root)
    with capture(root):  # one state of the chronicle: its frontier and every room's sealed set
        frontier = _active_frontier(store)
        if frontier is None:
            return {}
        facts = membership_facts(root)
        grouped: Dict[str, List[Entry]] = {}
        projected = _projection_rows(root, frontier)
        # The rows' task lineages in one read a generation (``RoomResolver.warm``), not one read per task.
        facts.rooms.warm({str(entry[1].get(key) or "") for entry in projected
                          for key in ("task_id", "parent_task_id", "root_task_id")} - {""})
        if gaps is not None:
            from ouroboros.room_appearances import REPLY_TARGET_UNREADABLE  # D15->D17 is lazy-only
            from ouroboros.room_membership import unknown_target_rooms
        for entry in projected:
            for room in rooms_of_row(entry[1], facts):
                grouped.setdefault(room, []).append(entry)
                if quiet is not None and entry[1].get("quiet_finish") is True:
                    quiet.setdefault(room, []).append(entry)
            if gaps is not None and isinstance(entry[1].get("reply_to"), dict):
                for room in unknown_target_rooms(entry[1], facts.project_chat_ids, facts.rooms):
                    gaps.setdefault(str(room), set()).add(REPLY_TARGET_UNREADABLE)
        rooms: Dict[str, List[Entry]] = {}
        timed = rooms_read_by_time(root, (entry[1] for entry in projected))  # every room at once, sealed rows or not
        for room, entries in grouped.items():
            sealed = store.sealed_row_refs(room)
            kept = [entry for entry in entries if entry[0]["row_sha256"] not in sealed]
            if kept:
                rooms[room] = in_reading_order(kept, timed, lambda entry: entry[1])
    return rooms


def _room_entries(root: Any, store: ChronicleStore, room: str,
                  pos_range: Optional[Tuple[int, int]] = None, gaps: Optional[set] = None) -> List[Entry]:
    """One room's rows in append order, sealed ones included: after the frontier, or in a half-open range.

    ``pos_range`` is a legacy unit's ``raw_range.pos``; its rows carry the membership fields
    (``_MEMBERSHIP_FIELDS``) as their metadata. Empty before activation, and for the room-less
    retelling (``chronicle_import.LEGACY_ROOM_ID``): it names no chat, so no row is its own.
    """
    if not str(room).lstrip("-").isdigit():
        return []
    if pos_range is None:
        frontier = _active_frontier(store)
        if frontier is None:
            return []
        start, end = int(frontier["pos"]), None
    else:
        if not store.log_path.exists():
            return []
        start, end = pos_range
    lower = chat_chain.position_boundary(root, start)
    upper = chat_chain.position_boundary(root, end) if end is not None else None
    return [(address, row_meta(row), pos) for address, row, pos in
            chat_chain.iter_room_rows(root, room, lower=lower, upper=upper, gaps=gaps)]


def open_room_rows(root: Any, room_id: Any, gaps: Optional[set] = None) -> List[Entry]:
    """``(address, row_meta, pos)`` of one room's open rows in reading order: append order, or in a room
    holding imported earlier messages its time order, though pages sealed them all
    (``dialogue_provenance.in_reading_order``), so lanes and the floor's oldest-first walk follow time;
    ``room_facts`` bounds their stored times. Addresses and positions stay physical."""
    store, room = ChronicleStore(root), str(room_id)
    entries = _room_entries(root, store, room, gaps=gaps)  # ``gaps``: what this Project's read may lack (``chat_chain.room_rows``)
    if not entries:
        return []
    sealed = store.sealed_row_refs(room)
    kept = [entry for entry in entries if entry[0]["row_sha256"] not in sealed]
    return in_reading_order(kept, rooms_read_by_time(root, (entry[1] for entry in kept)), lambda entry: entry[1])


# --- legacy units -------------------------------------------------------------------------------

@dataclass(frozen=True)
class LegacyUnit:
    """One record of the old memory and whether it is folded (the one rule of this module)."""
    record_id: str
    block: Optional[int]  # the legacy block (period); None for the flat file and gaps
    room_id: str
    raw: str  # "exact" | "unknown": whether the rows of the retelling are established
    rows: int  # rows of raw_range within the room
    uncovered: int  # of them, outside the room's sealed set
    retelling_chars: int  # the imported retelling's length (a later correction is not counted)
    folded: bool
    refusal: Optional[Dict[str, Any]]  # the fallback writer's refusal receipt for this unit, if any
    ts_span: Optional[Dict[str, Any]] = None  # ``source_time_span`` of the room's own rows; None without rows
    first_pos: int = -1  # the stream position of the room's first row in the range; -1 without rows


_RowSet = Tuple[FrozenSet[str], Optional[Dict[str, Any]], int]  # row_sha256 set, time span, first position
_RETELLING_CHARS: Dict[str, Dict[str, int]] = {}
_Triple = Tuple[str, str, str]  # a row's (task, parent, root) ids, as the room rule takes them
_UNITS: Dict[str, Tuple[Tuple[Any, ...], Tuple[_Triple, ...], Optional[Tuple[int, ...]], List[LegacyUnit]]] = {}
_SPAN_ROWS: Dict[str, Dict[Tuple[int, int], List[Entry]]] = {}


FALLBACK_REFUSAL_PREFIX = "fallback_refusal:"


def fallback_refusals(scan: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Read per-unit receipts over the old whole map; a null key clears its receipt.

    The journal's top-level scan deltas already provide per-unit replacement;
    old authority stays readable without rewriting its accumulated maps.
    """
    old = scan.get("fallback_refusals")
    found = dict(old) if isinstance(old, dict) else {}
    found.update((name[len(FALLBACK_REFUSAL_PREFIX):], value) for name, value in scan.items()
                 if name.startswith(FALLBACK_REFUSAL_PREFIX))
    return {key: receipt for key, receipt in found.items() if isinstance(receipt, dict)}


def _exact_range(pointer: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    raw = (pointer.get("covers") or {}).get("raw_range") or {}
    span = raw.get("pos")
    if raw.get("status") != "exact" or not isinstance(span, list) or len(span) != 2 \
            or not all(type(value) is int for value in span):
        return None
    return span[0], span[1]


def _span_rows(root: pathlib.Path, spans: Iterable[Tuple[int, int]]) -> Dict[Tuple[int, int], List[Entry]]:
    """Every stream row of each retained span, read once a process: a unit's exact range holds rows the
    old memory had already read, and a new row lands after every range, so a span's rows never change
    (on the owner's install 23 spans over the stream's first 12,400 rows, about 10 MB held until exit)."""
    known = _SPAN_ROWS.setdefault(str(root.resolve()), {})
    for start, end in sorted(set(spans) - set(known)):
        lower, upper = chat_chain.position_boundary(root, start), chat_chain.position_boundary(root, end)
        known[(start, end)] = [(address, row_meta(row), pos) for address, row, pos in
                               chat_chain.iter_rows(root, lower=lower, upper=upper)]
    return known


def _legacy_row_sets(root: pathlib.Path, ranges: Dict[str, Tuple[str, Tuple[int, int]]],
                     facts: MembershipFacts) -> Dict[str, _RowSet]:
    """Each retained unit's own room and ordinal interval, without reading the preceding chain: its span's
    rows (``_span_rows``) that the room rule puts in its room under the current membership facts, the
    rows' task lineages read together first, so a recovered lineage or a new binding counts at once."""
    spans = _span_rows(root, {bounds for _room, bounds in ranges.values()})
    facts.rooms.warm({str(entry[1].get(key) or "") for bounds in {bounds for _room, bounds in ranges.values()}
                      for entry in spans[bounds] for key in ("task_id", "parent_task_id", "root_task_id")} - {""})
    rooms: Dict[Tuple[int, int], List[FrozenSet[str]]] = {}
    sets = {}
    for record_id, (room, bounds) in ranges.items():
        if not str(room).lstrip("-").isdigit():  # the room-less retelling names no chat (``_room_entries``)
            sets[record_id] = (frozenset(), None, -1)
            continue
        if bounds not in rooms:
            rooms[bounds] = [rooms_of_row(entry[1], facts) for entry in spans[bounds]]
        entries = [entry for entry, held in zip(spans[bounds], rooms[bounds]) if room in held]
        stamps = [row.get("ts") for _addr, row, _pos in entries]
        sets[record_id] = (frozenset(address["row_sha256"] for address, _row, _pos in entries),
                           source_time_span(stamps) if stamps else None, entries[0][2] if entries else -1)
    return sets


def _task_triples(entries: Iterable[Entry]) -> Tuple[_Triple, ...]:
    """The distinct (task, parent, root) ids of the rows that name a task: what the room rule asks the lineage."""
    return tuple(sorted({(str(meta.get("task_id") or ""), str(meta.get("parent_task_id") or ""),
                          str(meta.get("root_task_id") or "")) for _address, meta, _pos in entries if meta.get("task_id")}))


def _lineage_answers(facts: MembershipFacts, triples: Tuple[_Triple, ...]) -> Optional[Tuple[int, ...]]:
    """The room rule's answer for each triple, the one membership input ``facts.fingerprint`` leaves out: a
    task result admitted later, or a lineage the store could not read under contention, changes it with
    no row written. ``None`` while the store cannot answer every task: a count made then is not reused."""
    if not facts.rooms.warm({task_id for triple in triples for task_id in triple} - {""}):
        return None
    return tuple(int(facts.rooms(*triple) or 0) for triple in triples)


def _retelling_chars(store: ChronicleStore, ids: set) -> Dict[str, int]:
    key = str(store.data_root.resolve())
    known = _RETELLING_CHARS.get(key, {})
    if not ids <= set(known):
        known = {record["id"]: len(str(record.get("text") or ""))
                 for record in store.records(kinds=("legacy", "gap"))}
        _RETELLING_CHARS[key] = known
    return known


def legacy_pointers(store: ChronicleStore) -> List[Dict[str, Any]]:
    """``ChronicleStore.legacy_pointer_rows``, read once per capture for its readers (shared: never changed)."""
    return capture_memo(store.data_root, "legacy_pointers", store.legacy_pointer_rows)


def legacy_units(store: ChronicleStore, root: Any) -> List[LegacyUnit]:
    """Every legacy record and journal gap, in the store's pointer order, with its folded verdict.

    The verdicts depend only on the chronicle's authority, the registry facts and the room rule's
    answers for the tasks of the units' rows (the rows themselves never change, ``_span_rows``), so the
    last answer is reused while all three are unchanged: a row written since changes none of them.
    """
    if not store.log_path.exists():
        return []
    root = pathlib.Path(root)
    authority, facts = _authority(store), membership_facts(root)
    from ouroboros import dialogue_store as ds
    ds.require_imported(root, "chat")
    key = (authority, facts.fingerprint())
    cached = _UNITS.get(str(root.resolve()))
    if cached is not None and cached[0] == key and cached[2] is not None \
            and _lineage_answers(facts, cached[1]) == cached[2]:
        return list(cached[3])
    # Every verdict reads one chronicle state: ``authority``'s, or a later one recounted next call (no tail fact).
    with capture(root, tail="none"):
        pointers = legacy_pointers(store)
        refusals = fallback_refusals(store.scan_state())
        chars = _retelling_chars(store, {pointer["node_id"] for pointer in pointers}) if pointers else {}
        ranges = {pointer["node_id"]: (str(pointer["room_id"]), span) for pointer in pointers
                  if (span := _exact_range(pointer)) is not None}
        bounds = {span for _room, span in ranges.values()}
        spans = _span_rows(root, bounds)
        triples = _task_triples(entry for span in bounds for entry in spans[span])
        answers = _lineage_answers(facts, triples)  # before the rows are counted: a lineage moving meanwhile recounts next call
        row_sets = _legacy_row_sets(root, ranges, facts)
        units = []
        for pointer in pointers:
            record_id, room = pointer["node_id"], str(pointer["room_id"])
            rows, span, first = row_sets.get(record_id, (frozenset(), None, -1))
            uncovered = len(rows - store.sealed_row_refs(room)) if rows else 0
            refusal = refusals.get(record_id)
            block = pointer.get("legacy_block")
            units.append(LegacyUnit(
                record_id=record_id, block=block if type(block) is int else None, room_id=room,
                raw="exact" if record_id in ranges else "unknown", rows=len(rows), uncovered=uncovered,
                retelling_chars=chars.get(record_id, 0),
                folded=bool(pointer.get("folded_into")) or (bool(rows) and uncovered == 0),
                refusal=dict(refusal) if isinstance(refusal, dict) else None, ts_span=span, first_pos=first))
    _UNITS[str(root.resolve())] = (key, triples, answers, units)
    return list(units)


def legacy_progress(units: List[LegacyUnit]) -> Dict[str, Any]:
    """``{"periods", "folded", "pending"}``: a period (block) counts as folded only when all its units are."""
    blocks: Dict[int, bool] = {}
    for unit in units:
        if unit.block is not None:
            blocks[unit.block] = blocks.get(unit.block, True) and unit.folded
    pending = sorted(block for block, folded in blocks.items() if not folded)
    return {"periods": len(blocks), "folded": len(blocks) - len(pending), "pending": pending}


# --- the period and place of a story record ----------------------------------------------------

class Period(NamedTuple):
    """A story record's period as read from its room's rows: ``span`` (``source_time_span`` form, None when no
    row and no block period is known), ``first`` (the stream position its story order starts from; -1 unknown)
    and ``source``: ``rows`` (the room's own rows), ``block`` (a retold record without rows: its block's recorded
    period and position) or ``mixed`` (a part over both)."""
    span: Optional[Dict[str, Any]]
    first: int
    source: str

    def note(self) -> str:
        """What a reader prints after the period: nothing for the room's own rows, else whose period it is."""
        return {"block": " (block period)", "mixed": " (partly block period)"}.get(self.source, "") if self.span else ""


def record_period(store: ChronicleStore, record: Mapping[str, Any], units: Mapping[str, LegacyUnit], *,
                  records: Optional[Mapping[str, Any]] = None) -> Period:
    """The period and place of a page, part or retold record, from its room's own rows, never the journal's aggregate.

    A page: its own covers. A retold record: its unit's rows (``units``: the ready legacy map of one reader);
    without rows, its block's recorded period and position, labelled ``block``. A part: the union over its
    members, recursively, so a room that takes the tail of a block is dated and ordered by its own rows; the
    ``covers.ts_span`` the part recorded (its members' block aggregate) is not read. An account: the union
    over its sources the same way, whatever their rooms (a missing source adds nothing and leaves it incomplete).
    A reader may supply its complete source-record snapshot to avoid reopening the index for every member.
    """
    covers = record.get("covers") if isinstance(record.get("covers"), Mapping) else {}
    if record.get("kind") in ("part", "account"):
        ids = covers.get("member_ids") if record.get("kind") == "part" else [
            (ref.get("id") if isinstance(ref, Mapping) else ref) for ref in record.get("sources") or ()]
        get = records.get if records is not None else store.get
        members = [record_period(store, member, units, records=records) for member_id in ids or ()
                   if (member := _member(get, units, str(member_id))) is not None]
        if len(members) != len(ids or ()):
            members.append(Period(None, -1, "rows"))  # a source not in this chronicle: an incomplete period
        spans = [period.span for period in members if period.span]
        sources = {period.source for period in members if period.span}
        span = source_time_span([span[key] for span in spans for key in ("start", "end") if span.get(key)],
                                incomplete=any(not period.span or period.span.get("incomplete") for period in members)
                                ) if spans else None
        firsts = [period.first for period in members if period.first >= 0]
        return Period(span, min(firsts) if firsts else -1, "mixed" if len(sources) > 1 else next(iter(sources), "rows"))
    if record.get("kind") in ("legacy", "gap"):
        unit = units.get(str(record.get("id")))
        if unit is not None and unit.ts_span:
            return Period(dict(unit.ts_span), unit.first_pos, "rows")
        raw = covers.get("raw_range") if isinstance(covers.get("raw_range"), Mapping) else {}
        span, pos = (raw.get("ts_span"), raw.get("pos")) if raw.get("status") == "exact" else (None, None)
        return Period(dict(span) if isinstance(span, Mapping) and (span.get("start") or span.get("end")) else None,
                      pos[0] if isinstance(pos, list) and pos and type(pos[0]) is int else -1, "block")
    span, stream = covers.get("ts_span"), covers.get("stream_span")
    return Period(dict(span) if isinstance(span, Mapping) and (span.get("start") or span.get("end")) else None,
                  stream[0] if isinstance(stream, list) and stream and type(stream[0]) is int else -1, "rows")


def _member(get: Callable[[str], Any], units: Mapping[str, LegacyUnit], member_id: str) -> Any:
    """A part's or account's member as ``record_period`` reads it: a retold record whose unit holds its rows
    answers from the unit, its retelling not read; any other record is read."""
    unit = units.get(member_id)
    return {"id": member_id, "kind": "legacy"} if unit is not None and unit.ts_span else get(member_id)


# --- the standing facts --------------------------------------------------------------------------

def story_counts(store: ChronicleStore) -> Dict[str, Any]:
    """``{"pages_by_me", "latest_by_me", "helper_pages", "parts"}`` over every room's acting pages and parts.

    Folded records count (they still seal their rows), rejected drafts do not; ``latest_by_me`` is the
    date (``YYYY-MM-DD``) the mind last sealed a page, ``""`` before the first. A helper's pages are
    counted apart: on an installation without consciousness the helper writes them, and "none by me"
    alone would mislead.
    """
    counts: Dict[str, Any] = {"pages_by_me": 0, "latest_by_me": "", "helper_pages": 0, "parts": store.acting_parts()}
    for page in store.acting_pages():  # counted in the index, no page read whole
        if page["author_kind"] == "mind":
            counts["pages_by_me"] += 1
            counts["latest_by_me"] = max(counts["latest_by_me"], str(page["ts"] or "")[:10])
        else:
            counts["helper_pages"] += 1
    return counts


def room_facts(store: ChronicleStore, room: str, entries: List[Entry], lineage: Mapping[str, Any],
               notes: int) -> Dict[str, Any]:
    """One room's standing facts, counted once for every reader that prints them.

    ``entries`` are the room's open rows (``open_room_rows``): ``people`` and ``mine`` are lane 1 by author
    (``dialogue_provenance.row_class``, with the chronicle's ``lineage``), ``task_facts`` lane 2, ``chars``
    their text; ``earliest`` and ``latest`` bound the open rows' stored times (``source_time_span``): in a room
    read by time a row received now can read first and an import sent long ago last. ``notes`` are the room's
    notes no page sealed. ``pages`` are the room's acting pages (folded ones too) and ``last_covered`` the latest
    row time among their covers — not a boundary: a page covers a set of rows and an earlier row can stay open, so
    the gap shows from ``earliest`` beside it and no sealing frontier is computed.
    """
    spoken = [cls["author"].get("kind") for cls in (row_class(meta, pos=pos, **lineage) for _a, meta, pos in entries)
              if cls["lane"] == 1]
    pages = store.acting_pages(room)
    ends = [page["covered_end"] for page in pages if page["covered_end"]]
    span = source_time_span(meta.get("ts") for _address, meta, _pos in entries)
    return {"rows": len(entries), "people": spoken.count("human"), "mine": spoken.count("ouroboros"),
            "task_facts": len(entries) - len(spoken),
            "chars": sum(int(meta.get("text_chars") or 0) for _address, meta, _pos in entries),
            "earliest": str(span["start"] or ""), "latest": str(span["end"] or ""), "notes": notes, "pages": len(pages),
            "last_covered": str(source_time_span(ends).get("end") or "") if ends else "",
            **({"place": place} if (place := place_activity(entries)) else {})}


def place_activity(entries: List[Entry]) -> Dict[str, Any]:
    """Index facts from open place rows; delivered event ids are facts, not an inferred obligation.

    Counts cover the open rows only. Multipart receipts count their source event once;
    failed receipts do not mark an event answered; imported earlier messages are history,
    not arrivals. ``latest`` bounds the rows' stored times (as ``room_facts``), ``last_address``
    names the last row as the room reads. No transcript text enters this index.
    """
    from ouroboros.dialogue_provenance import place_row_facts

    rows = [entry for entry in entries if row_provenance(entry[1]) == "place"]
    if not rows:
        return {}
    facts = [place_row_facts(meta) for _address, meta, _pos in rows]
    arrivals = {fact["source_event_id"] for (_address, meta, _pos), fact in zip(rows, facts)
                if fact.get("source_event_id") and meta.get("direction") == "in"
                and not isinstance(meta.get("historical_import"), dict)}
    answered = {fact["source_event_id"] for (_address, meta, _pos), fact in zip(rows, facts)
                if fact.get("source_event_id") and meta.get("type") == "presence_delivery"
                and (meta.get("transport") or {}).get("delivery", {}).get("state") in {"delivered", "accepted"}}
    return {"place_id": next((fact["place_id"] for fact in reversed(facts) if fact.get("place_id")), ""),
            "conversation_key": next((fact["conversation_key"] for fact in reversed(facts)
                                      if fact.get("conversation_key")), ""),
            "answered_events": len(answered), "unanswered_events": len(arrivals - answered),
            "arrivals": len(arrivals), "rows": len(rows),
            "latest": str(source_time_span(meta.get("ts") for _address, meta, _pos in rows)["end"] or ""),
            "last_address": chat_chain.format_address(rows[-1][0])}


def place_room_index(root: Any) -> List[Dict[str, Any]]:
    """Open conversation facts for the own-wake places line; zero rows means no open activity.

    The caller joins these rooms to its place registry for labels and current state.
    Unanswered counts mean arrivals without a recorded accepted/delivered receipt.
    """
    return [{"room_id": room, **facts} for room, entries in open_rows_by_room(root).items()
            if (facts := place_activity(entries))]


# --- open segments --------------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenSegment:
    """The oldest run of one room's open rows with no sealed row inside: what one page may seal."""
    room_id: str
    from_addr: Dict[str, Any]  # with ``to_addr``, the inclusive bounds ``tools/chronicle.page_covers`` takes
    to_addr: Dict[str, Any]
    # ``(address, metadata, pos)`` in append order, one per distinct row: ``row_meta`` after the
    # frontier, the membership fields within a legacy range (texts are read by address).
    rows: Tuple[Entry, ...]
    head_sequence: int  # the room head when read (a part's ``expected_sequence``)


def _is_row(address: Mapping[str, Any], bound: Mapping[str, Any]) -> bool:
    """Whether a full row address is the row a (possibly 12-hex) address names."""
    return (address["chat_id"] == bound["chat_id"] and address["ts"] == bound["ts"]
            and str(address["row_sha256"]).startswith(bound["row_sha256"]))


def oldest_open_segment(root: Any, store: ChronicleStore, room_id: Any, *,
                        pos_range: Optional[Tuple[int, int]] = None,
                        until: Any = None) -> Optional[OpenSegment]:
    """The room's oldest run of open rows; ``None`` when nothing is open there.

    The rows are the room's rows after the legacy frontier, or, with ``pos_range`` (a
    legacy unit's half-open ``raw_range.pos``), that unit's rows of the room. The room's
    sealed set is taken away first, so a stretch the mind partly sealed yields only its
    oldest unsealed run: from the first open row to the last open row before the next
    sealed one. ``until`` (a row address, e.g. ``newest_addressed_row``) ends the run at
    that row inclusive; when it names no row of these, or one before the first open row,
    there is no run. People's words and short replies inside the run stay in it.
    """
    return next(open_segments(root, store, room_id, pos_range=pos_range, until=until), None)


def open_segments(root: Any, store: ChronicleStore, room_id: Any, *,
                  pos_range: Optional[Tuple[int, int]] = None, until: Any = None) -> Iterator[OpenSegment]:
    """The room's runs of open rows, oldest first (``oldest_open_segment`` is the first).

    A writer that cannot read the oldest run (one row larger than its window) takes the
    next one instead of stopping the room's open rows for good.
    """
    room = str(room_id)
    bound = None
    if until is not None:
        try:
            bound = chat_chain._normalized(until)
        except ValueError:
            return
    entries = _room_entries(root, store, room, pos_range)
    if bound is not None:
        hits = [index for index, entry in enumerate(entries) if _is_row(entry[0], bound)]
        entries = entries[:hits[-1] + 1] if hits else []
    if not entries:
        return
    sealed = store.sealed_row_refs(room)
    head = store.room_head(room)
    run: List[Entry] = []
    seen = set()
    for entry in [*entries, None]:
        sha = entry[0]["row_sha256"] if entry is not None else None
        if entry is None or sha in sealed:
            if run:
                yield OpenSegment(room_id=room, from_addr=dict(run[0][0]), to_addr=dict(run[-1][0]),
                                  rows=tuple(run), head_sequence=head)
                run, seen = [], set()
            continue
        if sha not in seen:  # a byte-identical redelivered row is one member of a page's set
            seen.add(sha)
            run.append(entry)


# --- the view's floor fact as a writer's shortage ---------------------------------------------------

@dataclass(frozen=True)
class ShortageFact:
    """What a task's view showed only by address, when that asks a writer for a page or a part."""
    kind: str  # "open_rows" | "narrative"
    room_id: str
    newest_addressed_row: Optional[Dict[str, Any]] = None  # open_rows
    pointer_records: Tuple[str, ...] = ()  # narrative
    window_tokens: Optional[int] = None
    mode: str = ""


_ROW_STEPS = frozenset({"F2", "F7"})  # my replies, this room's people: the steps that ask the mind
_NARRATIVE_STEPS = frozenset({"F4", "F5l", "F5"})  # this room's retold page, story labels and pointers


def shortage_from_trace(trace: Any) -> Optional[ShortageFact]:
    """The floor fact of a task's last view (``trace[VIEW_TRACE_KEY]``, written by the view) as a shortage.

    ``open_rows``: the floor showed my replies or this room's people only by address (F2,
    F7) and names the newest addressed row of this room. ``narrative``: it showed old
    narrative by label or pointer (F4, F5l, F5), names those records, and the window alone took
    them (``by_budget == 0``; an owner-selected mode target is no shortage of the window).
    Fact lines (F1), quiet rooms (F1b), retold pointers (F3) and other rooms' people (F6)
    alone, a foreign shape or no fact: ``None``.
    """
    fact = trace.get(VIEW_TRACE_KEY) if isinstance(trace, Mapping) else None
    floor = fact.get("floor") if isinstance(fact, Mapping) else None
    if not isinstance(floor, Mapping) or not isinstance(floor.get("steps"), Mapping):
        return None
    room = fact.get("room_id")
    if room is None or not str(room).strip():
        return None
    steps = {str(step) for step, count in floor["steps"].items() if type(count) is int and count > 0}
    window = floor.get("window_tokens")
    common = {"room_id": str(room), "window_tokens": window if type(window) is int else None,
              "mode": str(floor.get("mode") or "")}
    newest = floor.get("newest_addressed_row")
    if steps & _ROW_STEPS and newest is not None:
        try:
            return ShortageFact("open_rows", newest_addressed_row=chat_chain._normalized(newest), **common)
        except ValueError:
            pass
    pointers = floor.get("pointer_records")
    by_budget = floor.get("by_budget")
    if steps & _NARRATIVE_STEPS and type(by_budget) is int and by_budget == 0 and isinstance(pointers, (list, tuple)):
        records = tuple(dict.fromkeys(str(ident) for ident in pointers if isinstance(ident, str) and ident.strip()))
        if records:
            return ShortageFact("narrative", pointer_records=records, **common)
    return None


# --- journal changes since an accepted wake -----------------------------------------------------------

# What a wake observes of the journal: the records that change what my memory says or keeps in
# view. A mark's release is one, with who released it: any focus, a helper included, may release
# any mark, a global one too. The import's legacy sections, gaps and activation receipt
# are standing inventory (the view's pointers), and a mark's view only changes how much of an
# observed mark is shown.
CHANGE_KINDS = ("page", "part", "note", "correction", "decision", "mark", "mark_release", "account", "selection")


def _read_call(record_id: Any) -> str:
    return f"memory_read(node_id='{record_id}')"


def _author_words(author: Any) -> str:
    author = author if isinstance(author, Mapping) else {}
    kind = str(author.get("kind") or "unknown author")
    if kind == "mind":
        focus = author.get("focus") if isinstance(author.get("focus"), Mapping) else {}
        return f"mind ({focus.get('role') or 'focus not recorded'}, task {author.get('task_id') or 'not recorded'})"
    detail = author.get("writer") or author.get("operation") or author.get("attribution") or ""
    return f"{kind} ({detail})" if detail else kind


def _change_line(record: Mapping[str, Any], released: Optional[Mapping[str, Any]] = None) -> str:
    """One journal record as a line of text: what it is, whose, and the exact reads of it and its target.

    ``released`` is the mark a ``mark_release`` takes away; its scope says where the release acts.
    """
    kind, author = str(record.get("kind")), record.get("author")
    covers = record.get("covers") if isinstance(record.get("covers"), Mapping) else {}
    draft = " draft" if kind in ("page", "part") and isinstance(author, Mapping) and author.get("kind") == "helper" else ""
    scoped = record if kind == "mark" else released if isinstance(released, Mapping) else {}
    where = "global" if scoped.get("scope") == "global" else f"in room {record.get('room_id')}"
    rows, members = covers.get("count", len(covers.get("rows") or ())), len(covers.get("member_ids") or ())
    sources, replaced = len(record.get("sources") or ()), len(record.get("replaces") or ())
    what = {"page": f"seals {rows} row{'' if rows == 1 else 's'}",
            "part": f"folds {members} record{'' if members == 1 else 's'}",
            "account": f"is based on {sources} source{'' if sources == 1 else 's'} (nothing sealed or folded)",
            "selection": (f"shows the account in the common view in place of {replaced} record{'' if replaced == 1 else 's'}"
                          if record.get("shown") else "withdraws the account from the common view"),
            "decision": "accepts the draft" if record.get("accepted") else "rejects the draft",
            "correction": "corrects its target", "mark_release": "releases its target mark"}.get(kind, "")
    target = record.get("target_id")
    if kind == "mark":
        ref = record.get("target_ref") if isinstance(record.get("target_ref"), Mapping) else {}
        target = ref.get("id") if ref.get("kind") == "chronicle" else None
    return (f"- {kind}{draft} {record.get('id')} {where}, by {_author_words(author)}" + (f": {what}" if what else "")
            + f"; {_read_call(record.get('id'))}" + (f"; target {_read_call(target)}" if target else ""))


def memory_changes(root: Any, boundary: Any, gaps: set) -> Tuple[List[Tuple[str, None, str]], Any, Optional[Dict[str, Any]]]:
    """``(events, boundary to accept, window)``: the journal records written since an accepted wake.

    Each event is ``("memory_change", None, line)`` for a record of ``CHANGE_KINDS`` after the
    accepted ``{sequence, record_id}``, in publication order; source dates do not order
    publication. Without an accepted position the current one is the baseline and nothing
    earlier is inventoried (the window says so). Without a journal there are no events and
    nothing is created: the boundary is sequence 0. A read failure, or an accepted position
    whose record no longer matches, is a disclosed gap and the accepted boundary stays.
    """
    empty = {"sequence": 0, "record_id": None}
    store = ChronicleStore(root)
    try:
        if not store.log_path.exists():
            if isinstance(boundary, Mapping) and boundary.get("sequence"):
                raise ValueError("accepted memory source is missing")
            return [], empty, None  # an empty install gets no memory directory or index
        records, current = store.observation_snapshot(boundary)
        window = {"lower": boundary.get("sequence") if isinstance(boundary, Mapping) else current["sequence"],
                  "upper": current["sequence"], "last_record_id": current["record_id"],
                  "basis": "accepted_sequence" if boundary is not None else "initial_baseline"}
        events = [("memory_change", None, _change_line(record, store.get(str(record.get("target_id") or ""))
                                                       if record.get("kind") == "mark_release" else None))
                  for record in records if record.get("kind") in CHANGE_KINDS]
        return events, current, window
    except Exception as exc:  # a disclosed gap beats a missing wake
        gaps.add(f"memory changes unreadable: {type(exc).__name__}; accepted sequence retained")
        lower = boundary.get("sequence") if isinstance(boundary, Mapping) else None
        return [], boundary, {"basis": "unreadable", "lower": lower}
