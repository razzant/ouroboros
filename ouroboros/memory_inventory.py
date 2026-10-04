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
- An **open segment** is the oldest run of a room's open rows (after the frontier, or within
  one legacy unit's range) with no sealed row inside: what one page may seal without being
  refused ``already_sealed``. The sealed set is taken away before the choice.
- A **shortage** is the view's floor fact in a task's trace (``VIEW_TRACE_KEY``) read as a
  writer's cue: open rows of a room shown only by address, or old narrative shown only by
  pointer because the window could not hold it.
- The **journal changes** since an accepted wake are the records written after its accepted
  sequence, one line of text each with the exact read of the record and of its target.

Rows come from a process-local projection over the chat chain with
``chat_chain``'s own row rule and stream positions. It starts at the generation of
the legacy frontier (the last covered row's address and position anchor it), keeps
row metadata without text (``row_meta``; texts are read by address), and on later
calls reads only the bytes appended since. The live file is read up to its last
complete line; a rotation is recognized by the generation signatures. The row sets
of legacy units need the chain before the frontier: it is read once per process and
cached by generation signature. Room membership is recomputed on every call
(``rooms_of_row``, the rule of ``project_dialogue.room_membership``): a later Project
binding moves earlier rows.

Nothing here publishes a record or calls a model.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, NamedTuple, Optional, Tuple

from ouroboros import chat_chain
from ouroboros.chronicle_import import legacy_frontier
from ouroboros.chronicle_store import ChronicleStore, source_time_span
from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, is_a2a_chat_id

# The key of the memory view's last-capture fact in a task's ``llm_trace``.
VIEW_TRACE_KEY = "memory_view"

MAIN_ROOM = 1

# The row fields the room and attribution readers consult (``project_dialogue.room_membership``,
# ``dialogue_provenance.row_author``) plus the host facts of a task row. Texts and bulky
# payloads (review projections, outcome axes, quiz cards) stay in the chain, read by address.
_META_FIELDS = ("chat_id", "ts", "type", "direction", "task_id", "parent_task_id", "root_task_id",
                "client_message_id", "subagent_task_id", "delegation_role", "subagent_role", "initiator",
                "summary_kind", "sender_label", "username", "author", "source", "transport",
                "status", "outcome", "outcome_phase", "reason_code", "result_ref")
_MEMBERSHIP_FIELDS = ("chat_id", "ts", "type", "task_id", "parent_task_id", "root_task_id", "source_keys")

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
    if str(row.get("direction") or "") == "in":
        from ouroboros.project_dialogue import _entry_source_identities, _text_sha256  # D15->D17 is lazy-only

        meta["text_sha256"] = _text_sha256(row.get("text"))
        meta["source_keys"] = tuple(sorted(_entry_source_identities(row)))
    return meta


class MembershipFacts(NamedTuple):
    """The registry facts room membership reads, taken once per call."""
    project_chat_ids: FrozenSet[int]
    bindings: Dict[str, int]
    projects_by_source_key: Dict[Tuple[Any, ...], FrozenSet[int]]

    def fingerprint(self) -> str:
        payload = json.dumps([sorted(self.project_chat_ids), sorted(self.bindings.items()),
                              sorted((list(key), sorted(chats)) for key, chats in self.projects_by_source_key.items())],
                             ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def membership_facts(root: Any) -> MembershipFacts:
    """Reserved Project rooms, task bindings and the bound owner messages, as ``iter_room_rows`` reads them."""
    from ouroboros.project_dialogue import _source_ref_identity  # D15->D17 is lazy-only
    from ouroboros.projects_registry import all_task_bindings, list_reserved_projects, project_task_bindings

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
                           {key: frozenset(chats) for key, chats in by_key.items()})


def rooms_of_row(meta: Mapping[str, Any], facts: MembershipFacts) -> FrozenSet[str]:
    """Every room ``project_dialogue.room_membership`` puts this row in, as room ids.

    A Project room holds its own chat, the rows of tasks bound to it and the bound
    owner message (also kept by the room it was written in); lifecycle rows pinned to
    Main never enter a Project. Main holds every other unbound row outside Projects
    except the hidden partition; any other chat holds its own unbound rows.
    """
    from ouroboros.project_dialogue import (MAIN_PINNED_ROW_TYPES, ORIGIN_ADDRESSED_NOTICE_TYPES,  # D15->D17 is lazy-only
                                            bound_room_chat)

    chat = chat_chain._row_chat_id(meta)
    if is_a2a_chat_id(chat):
        return frozenset()
    kind = meta.get("type")
    bound = 0 if kind in ORIGIN_ADDRESSED_NOTICE_TYPES else bound_room_chat(facts.bindings, meta)
    pinned = kind in MAIN_PINNED_ROW_TYPES
    projects = facts.project_chat_ids
    rooms = set()
    if not pinned:
        rooms.update(room for room in (bound, chat) if room in projects)
        for key in meta.get("source_keys") or ():
            rooms.update(facts.projects_by_source_key.get(tuple(key), ()))
    if MAIN_ROOM not in projects and chat != HIDDEN_CHAT_ID and chat not in projects and (pinned or not bound):
        rooms.add(MAIN_ROOM)
    if chat not in projects and chat != MAIN_ROOM and not bound:
        rooms.add(chat)
    return frozenset(str(room) for room in rooms)


# --- the open-row projection --------------------------------------------------------------------

class _Stale(Exception):
    """The chain no longer continues the projection (a rewrite, a missing generation)."""


@dataclass
class _Generation:
    sig: str
    consumed: int = 0  # bytes read
    lines: int = 0  # physical lines read


@dataclass
class _Projection:
    frontier_key: Tuple[Any, ...]
    start_sig: Optional[str]  # None: the chain's first generation
    next_pos: int
    min_pos: int
    generations: List[_Generation] = field(default_factory=list)
    rows: List[Entry] = field(default_factory=list)


_LOCK = threading.Lock()
_PROJECTIONS: Dict[str, _Projection] = {}


def _read_lines(path: pathlib.Path, start: int, *, live: bool) -> Tuple[List[bytes], int]:
    """Physical lines from byte ``start`` and the bytes they span; the live file stops at its last newline."""
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read()
    if live:
        data = data[:data.rfind(b"\n") + 1]
    if not data:
        return [], 0
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines.pop()
    return lines, len(data)


def _frontier_key(frontier: Mapping[str, Any]) -> Tuple[Any, ...]:
    last = frontier.get("last_covered") if isinstance(frontier.get("last_covered"), dict) else {}
    return (frontier.get("status"), frontier.get("pos"), last.get("row_sha256"))


def _ingest(projection: _Projection, sig: str, lines: List[bytes], first_line: int) -> None:
    for offset, raw in enumerate(lines):
        row = chat_chain._decoded(raw)
        if row is None or is_a2a_chat_id(row.get("chat_id", 1)):
            continue  # Not a stream row: it takes no position (chat_chain's rule).
        pos = projection.next_pos
        projection.next_pos += 1
        if pos >= projection.min_pos:
            projection.rows.append((chat_chain.row_address(row, gen=sig, line=first_line + offset),
                                    row_meta(row), pos))


def _advance(root: pathlib.Path, projection: _Projection) -> None:
    """Read what the chain gained since the last call: new bytes of known generations, then new ones."""
    sigs = chat_chain.generation_signatures(root)
    names = [sig for _path, sig in sigs]
    if projection.start_sig is None:
        chain = sigs
    elif projection.start_sig in names:
        chain = sigs[names.index(projection.start_sig):]
    else:
        raise _Stale
    if len(chain) < len(projection.generations):
        raise _Stale
    live = root / "logs" / "chat.jsonl"
    for index, (path, sig) in enumerate(chain):
        if index < len(projection.generations):
            generation = projection.generations[index]
            if generation.sig != sig:
                # Only an empty generation may gain its first line; anything else was rewritten.
                if index != len(projection.generations) - 1 or generation.consumed:
                    raise _Stale
                generation.sig = sig
        else:
            generation = _Generation(sig)
            projection.generations.append(generation)
        size = path.stat().st_size
        if size < generation.consumed:
            raise _Stale
        if size == generation.consumed:
            continue
        lines, used = _read_lines(path, generation.consumed, live=path == live)
        _ingest(projection, sig, lines, generation.lines + 1)
        generation.consumed += used
        generation.lines += len(lines)


def _build(root: pathlib.Path, frontier: Mapping[str, Any]) -> _Projection:
    """A new projection, anchored at the last covered row when its address still resolves."""
    pos = int(frontier["pos"])
    last = frontier.get("last_covered")
    if pos > 0 and isinstance(last, dict):
        row, found = chat_chain.resolve_row(root, last)
        if row is not None:
            # The last covered row holds position pos - 1: what follows it in its generation is open.
            path, line = root / found["path"], int(found["line"])
            projection = _Projection(_frontier_key(frontier), found["gen"], pos, pos)
            lines, used = _read_lines(path, 0, live=path == root / "logs" / "chat.jsonl")
            _ingest(projection, found["gen"], lines[line:], line + 1)
            projection.generations.append(_Generation(found["gen"], used, len(lines)))
            return projection
    # Nothing covered, or the anchor no longer resolves: count positions from the chain's start.
    return _Projection(_frontier_key(frontier), None, 0, pos)


def _projection_rows(root: Any, frontier: Mapping[str, Any]) -> List[Entry]:
    root = pathlib.Path(root)
    key = str(root.resolve())
    with _LOCK:
        projection = _PROJECTIONS.get(key)
        if projection is None or projection.frontier_key != _frontier_key(frontier):
            projection = _build(root, frontier)
        try:
            _advance(root, projection)
        except (_Stale, OSError):
            projection = _build(root, frontier)
            _advance(root, projection)
        _PROJECTIONS[key] = projection
        return list(projection.rows)


_SEALED: Dict[str, Tuple[Tuple[int, int], Dict[str, FrozenSet[str]]]] = {}


def _authority(store: ChronicleStore) -> Tuple[int, int]:
    """Identity of the chronicle's append-only authority: the same bytes project the same index."""
    stat = store.log_path.stat()
    return stat.st_size, stat.st_mtime_ns


def _sealed(store: ChronicleStore, room: str, authority: Tuple[int, int]) -> FrozenSet[str]:
    """``store.sealed_row_refs(room)``, read once per room for one state of the authority."""
    key = str(store.data_root.resolve())
    cached = _SEALED.get(key)
    if cached is None or cached[0] != authority:
        cached = (authority, {})
        _SEALED[key] = cached
    if room not in cached[1]:
        cached[1][room] = frozenset(store.sealed_row_refs(room))
    return cached[1][room]


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


def open_rows_by_room(root: Any) -> Dict[str, List[Entry]]:
    """The open rows of every room that has one: after the frontier, outside the room's sealed set."""
    store = ChronicleStore(root)
    frontier = _active_frontier(store)
    if frontier is None:
        return {}
    facts, authority = membership_facts(root), _authority(store)
    grouped: Dict[str, List[Entry]] = {}
    for entry in _projection_rows(root, frontier):
        for room in rooms_of_row(entry[1], facts):
            grouped.setdefault(room, []).append(entry)
    rooms: Dict[str, List[Entry]] = {}
    for room, entries in grouped.items():
        sealed = _sealed(store, room, authority)
        kept = [entry for entry in entries if entry[0]["row_sha256"] not in sealed]
        if kept:
            rooms[room] = kept
    return rooms


def _room_entries(root: Any, store: ChronicleStore, room: str,
                  pos_range: Optional[Tuple[int, int]] = None) -> List[Entry]:
    """One room's rows in append order, sealed ones included: after the frontier, or in a half-open range.

    ``pos_range`` is a legacy unit's ``raw_range.pos``; its rows carry the membership fields
    (``_MEMBERSHIP_FIELDS``) as their metadata. Empty before activation.
    """
    if pos_range is None:
        frontier = _active_frontier(store)
        if frontier is None:
            return []
        facts = membership_facts(root)
        return [entry for entry in _projection_rows(root, frontier) if room in rooms_of_row(entry[1], facts)]
    if not store.log_path.exists():
        return []
    start, end = pos_range
    _read, stream = _legacy_stream(pathlib.Path(root), end)
    facts = membership_facts(root)
    return [(stream[pos][0], stream[pos][1], pos) for pos in range(max(start, 0), min(end, len(stream)))
            if room in rooms_of_row(stream[pos][1], facts)]


def open_room_rows(root: Any, room_id: Any) -> List[Entry]:
    """``(address, row_meta, pos)`` of one room's open rows, in append order."""
    store, room = ChronicleStore(root), str(room_id)
    entries = _room_entries(root, store, room)
    if not entries:
        return []
    sealed = _sealed(store, room, _authority(store))
    return [entry for entry in entries if entry[0]["row_sha256"] not in sealed]


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


_LEGACY_ROWS: Dict[str, Tuple[Tuple[str, ...], int, List[Tuple[Dict[str, Any], Dict[str, Any]]]]] = {}
_LEGACY_SETS: Dict[str, Tuple[Tuple[Any, ...], Dict[str, Tuple[FrozenSet[str], Optional[Dict[str, Any]]]]]] = {}
_RETELLING_CHARS: Dict[str, Dict[str, int]] = {}
_UNITS: Dict[str, Tuple[Tuple[Any, ...], List[LegacyUnit]]] = {}


def _exact_range(pointer: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    raw = (pointer.get("covers") or {}).get("raw_range") or {}
    span = raw.get("pos")
    if raw.get("status") != "exact" or not isinstance(span, list) or len(span) != 2 \
            or not all(type(value) is int for value in span):
        return None
    return span[0], span[1]


def _legacy_stream(root: pathlib.Path, stop: int) -> Tuple[Tuple[str, ...], List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """``(generations read, [(address, membership fields)] by stream position below stop)``.

    One pass per process, cached by the signatures of the generations it read: archives
    never change and a live generation only grows, so its earlier rows stay put.
    """
    key = str(root.resolve())
    names = tuple(sig for _path, sig in chat_chain.generation_signatures(root))
    cached = _LEGACY_ROWS.get(key)
    if cached is not None and names[:len(cached[0])] == cached[0] and cached[1] >= stop:
        return cached[0], cached[2]
    rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    read: Tuple[str, ...] = ()
    for address, row, pos in chat_chain.iter_rows(root):
        if pos >= stop:
            break
        meta = row_meta(row)
        rows.append((address, {k: meta[k] for k in _MEMBERSHIP_FIELDS if k in meta}))
        gen = address["hint"].get("gen")
        if gen in names and (not read or read[-1] != gen):
            read = names[:names.index(gen) + 1]
    _LEGACY_ROWS[key] = (read, stop, rows)
    return read, rows


def _legacy_row_sets(root: pathlib.Path, ranges: Dict[str, Tuple[str, Tuple[int, int]]],
                     facts: MembershipFacts) -> Dict[str, Tuple[FrozenSet[str], Optional[Dict[str, Any]]]]:
    """``record id -> (row_sha256 set, time span of those rows)`` of each exact unit: its range's rows of its room."""
    if not ranges:
        return {}
    stop = max(end for _room, (_start, end) in ranges.values())
    read, stream = _legacy_stream(root, stop)
    key = (read, len(stream), facts.fingerprint(),
           hashlib.sha256(json.dumps(sorted(ranges.items())).encode("utf-8")).hexdigest())
    cached = _LEGACY_SETS.get(str(root.resolve()))
    if cached is not None and cached[0] == key:
        return cached[1]
    rooms_at: Dict[int, FrozenSet[str]] = {}
    sets: Dict[str, Tuple[FrozenSet[str], Optional[Dict[str, Any]]]] = {}
    for record_id, (room, (start, end)) in ranges.items():
        members, stamps = set(), []
        for pos in range(max(start, 0), min(end, len(stream))):
            if pos not in rooms_at:
                rooms_at[pos] = rooms_of_row(stream[pos][1], facts)
            if room in rooms_at[pos]:
                members.add(stream[pos][0]["row_sha256"])
                stamps.append(stream[pos][1].get("ts"))
        sets[record_id] = (frozenset(members), source_time_span(stamps) if stamps else None)
    _LEGACY_SETS[str(root.resolve())] = (key, sets)
    return sets


def _retelling_chars(store: ChronicleStore, ids: set) -> Dict[str, int]:
    key = str(store.data_root.resolve())
    known = _RETELLING_CHARS.get(key, {})
    if not ids <= set(known):
        known = {record["id"]: len(str(record.get("text") or ""))
                 for record in store.records(kinds=("legacy", "gap"))}
        _RETELLING_CHARS[key] = known
    return known


def legacy_units(store: ChronicleStore, root: Any) -> List[LegacyUnit]:
    """Every legacy record and journal gap, in the store's pointer order, with its folded verdict.

    The verdicts depend only on the chronicle's authority, the registry facts and the chat
    generations, so the last answer is reused while all three are unchanged.
    """
    if not store.log_path.exists():
        return []
    root = pathlib.Path(root)
    authority, facts = _authority(store), membership_facts(root)
    key = (authority, facts.fingerprint(), tuple(sig for _path, sig in chat_chain.generation_signatures(root)))
    cached = _UNITS.get(str(root.resolve()))
    if cached is not None and cached[0] == key:
        return list(cached[1])
    pointers = store.legacy_pointer_rows()
    refusals = store.scan_state().get("fallback_refusals")
    refusals = refusals if isinstance(refusals, dict) else {}
    chars = _retelling_chars(store, {pointer["node_id"] for pointer in pointers}) if pointers else {}
    ranges = {pointer["node_id"]: (str(pointer["room_id"]), span) for pointer in pointers
              if (span := _exact_range(pointer)) is not None}
    row_sets = _legacy_row_sets(root, ranges, facts)
    units = []
    for pointer in pointers:
        record_id, room = pointer["node_id"], str(pointer["room_id"])
        rows, span = row_sets.get(record_id, (frozenset(), None))
        uncovered = len(rows - _sealed(store, room, authority)) if rows else 0
        refusal = refusals.get(record_id)
        block = pointer.get("legacy_block")
        units.append(LegacyUnit(
            record_id=record_id, block=block if type(block) is int else None, room_id=room,
            raw="exact" if record_id in ranges else "unknown", rows=len(rows), uncovered=uncovered,
            retelling_chars=chars.get(record_id, 0),
            folded=bool(pointer.get("folded_into")) or (bool(rows) and uncovered == 0),
            refusal=dict(refusal) if isinstance(refusal, dict) else None, ts_span=span))
    _UNITS[str(root.resolve())] = (key, units)
    return list(units)


def legacy_progress(units: List[LegacyUnit]) -> Dict[str, Any]:
    """``{"periods", "folded", "pending"}``: a period (block) counts as folded only when all its units are."""
    blocks: Dict[int, bool] = {}
    for unit in units:
        if unit.block is not None:
            blocks[unit.block] = blocks.get(unit.block, True) and unit.folded
    pending = sorted(block for block, folded in blocks.items() if not folded)
    return {"periods": len(blocks), "folded": len(blocks) - len(pending), "pending": pending}


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
    sealed = _sealed(store, room, _authority(store))
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
_NARRATIVE_STEPS = frozenset({"F4", "F5"})  # this room's retold page, my old pages and parts


def shortage_from_trace(trace: Any) -> Optional[ShortageFact]:
    """The floor fact of a task's last view (``trace[VIEW_TRACE_KEY]``, written by the view) as a shortage.

    ``open_rows``: the floor showed my replies or this room's people only by address (F2,
    F7) and names the newest addressed row of this room. ``narrative``: it showed old
    narrative only by pointer (F4, F5), names those records, and the window alone took
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
CHANGE_KINDS = ("page", "part", "note", "correction", "decision", "mark", "mark_release")


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
    what = {"page": f"seals {rows} row{'' if rows == 1 else 's'}",
            "part": f"folds {members} record{'' if members == 1 else 's'}",
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
