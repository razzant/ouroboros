"""The resident memory view of the acting mind: one render for every role.

The view replaces the old dialogue history and chat tail in a request. Block B carries my
story (``## My story``); block C the live part (``## Marks I keep in view``,
``## Live rooms`` and ``## This room (<label>) — head <n>``). Which parts a role sees is
one ``ViewSpec``: ``ROLE_DEFAULTS`` by role (Main, a root and a Project task are the
integrator; consciousness; Presence; a delegated child; a nanny), or an explicit spec a
task carries under ``memory_view``. The current room is the task's own room
(``dialogue_evidence.own_room_chat``); outside a Project and Main it is Main, except a
Presence turn, which sees its own conversation.

``capture_memory_view`` reads the facts once per request into a ``MemoryViewSnapshot``
(texts included), so rendering for another window or mode reads nothing again. It
activates the chronicle exactly once (the one legacy import, no model call). Until that
import has completed the view says so in one visible line, reads no chain and never
fails the task. What of memory is open and what is folded comes from
``memory_inventory`` only; what a window shows only by an address line is
``memory_floor``'s verdict, a ``FloorLevel``.

The story block depends only on the chronicle, the room labels, the helper route and the
floor, with no capture time, task id, JSON, relative time or ordinal: in Max, or while no
owner Low/Nano budget step reaches it (that budget weighs the whole view, the room too), the
same bytes for Main, any room's root, consciousness and Presence (``INTEGRATING``: they read the
first block of the old retelling whole, so the room page does not repeat it; a child names it by
pointer). Texts of records keep their words and are indented by two spaces, so their own ``## ``
lines never read as sections.

Nothing here publishes a record or calls a model.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import pathlib
from collections import Counter
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros import chat_chain, memory_inventory
from ouroboros.chronicle_import import LEGACY_ROOM_ID, LEGACY_ROOM_LABEL, legacy_frontier, row_lineage
from ouroboros.chronicle_store import CHILD_DRAFT_RIGHT, ChronicleStore, draft_signer
from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
from ouroboros.dialogue_provenance import (RoomLabelResolver, is_presence_task, render_memory_row, render_row_text,
                                           row_class)
from ouroboros.memory_view_legacy import INDENT, indented as _indented, retold_lines
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

MAIN_ROOM = str(WEB_UI_CHAT_ID)
ROLES = ("integrator", "consciousness", "presence", "child", "nanny")
# The focuses that integrate my life read the first block of the old retelling (the time before rooms had memory
# of their own, which the fallback writer never folds) whole in their story; a child and a nanny do not.
INTEGRATING = ("integrator", "consciousness", "presence")
LIVE_ROOMS = ("none", "lines", "lines_with_words")
MARKS = ("all", "room_and_global", "none")
LEGACY_FILE = "memory/dialogue_blocks.json"
# A task's typed host facts its lane-2 line names by type and address, never by text or JSON:
# the late review the owner did not see, a failed or uncertain send, a cancel receipt (it may
# carry an unreviewed model draft), a custody notice and a terminal incident (the old passive
# view showed each of them).
TYPED_HOST_FACTS: Mapping[str, str] = MappingProxyType({
    "acceptance_late_settlement": "late review evidence", "presence_delivery": "delivery",
    "cancel_receipt": "cancel receipt", "custody_notice": "custody notice", "terminal_incident": "terminal incident"})
_TERMINAL = frozenset({"terminal_root_projection", "terminal_result_projection"})
# The host's facts row; a ``task_summary`` without a kind is the old writer's model prose, never a status.
_HOST_FACTS, _KINDLESS = frozenset({"host_task_facts"}), frozenset({"", None})
# The delegated child's role text, carried verbatim, then its draft right: it publishes only drafts in its own name.
CHILD_ROLE_TEXT = ("Work from this assignment first — it is written to be enough; read memory or sources only to "
                   "fill a gap it leaves, and name what you read in your report. " + CHILD_DRAFT_RIGHT)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _minute(value: Any) -> str:
    try:
        moment = datetime.datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if moment.tzinfo is not None:
        moment = moment.astimezone(datetime.timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M")


def _period(span: Any) -> str:
    """``start → end`` of a source time span, minutes in UTC; a span with unknown bounds says so."""
    span = _mapping(span)
    if not span.get("start") and not span.get("end"):
        return "period unknown"
    text = f"{_minute(span.get('start') or '?')} → {_minute(span.get('end') or '?')}"
    return text + " (incomplete)" if span.get("incomplete") else text


def _labeler(root: pathlib.Path) -> Callable[..., str]:
    """``label(room, sample=None)``: one registry snapshot per capture (``RoomLabelResolver``)."""
    resolver = RoomLabelResolver(root)

    def label(room: Any, sample: Optional[Mapping[str, Any]] = None) -> str:
        if str(room) == LEGACY_ROOM_ID:
            return LEGACY_ROOM_LABEL
        try:
            chat = int(str(room))
        except ValueError:
            return f"Unresolved room [chat_id={room}]"
        facts = {key: value for key, value in _mapping(sample).items() if key in ("transport", "presence_provenance")}
        return resolver.label({"chat_id": chat, **facts})

    return label


# --- what a role sees ------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ViewSpec:
    """What one request's memory view holds; a helper's composition is set by its role defaults."""
    role: str
    story: bool = True  # block B: the top level of my story
    room_id: Optional[str] = None  # the current room; None: no ``## This room``
    room_page: bool = True  # the room's retold records not yet folded, pages under my parts, my open notes
    room_lanes: bool = True  # the open conversation (lane 1) and one line per task (lane 2)
    origin_words: bool = True  # the owner's words that started a Project, when its rows are not open
    live_rooms: str = "lines"  # none | lines | lines_with_words
    marks: str = "all"  # all | room_and_global | none
    knowledge: bool = True  # overview, knowledge index, patterns, project journal (rendered by the context)
    owner_words: bool = False  # the owner's words that caused a helper's work (owner_words.py)


ROLE_DEFAULTS: Mapping[str, ViewSpec] = MappingProxyType({
    "integrator": ViewSpec("integrator"),
    "consciousness": ViewSpec("consciousness", room_page=False, room_lanes=False, origin_words=False),
    "presence": ViewSpec("presence", origin_words=False, live_rooms="none", marks="room_and_global"),
    "child": ViewSpec("child", room_lanes=False, live_rooms="none", marks="room_and_global", knowledge=False,
                      owner_words=True),
    "nanny": ViewSpec("nanny", story=False, room_lanes=False, live_rooms="none", marks="room_and_global",
                      knowledge=False, owner_words=True),
})
_ROOMLESS = frozenset({"consciousness"})
_FIELDS = {field.name: field.type for field in dataclasses.fields(ViewSpec)}


def view_role(task: Mapping[str, Any]) -> str:
    """The view role, first match wins, in ``knowledge.focus_signature``'s order.

    A nanny is the one dispatch fact (``_nanny_route_dispatched_for``), then a delegated
    child, a wake's own ledger category, a Presence turn; everything else integrates.
    The facts are read at the task's top level or under ``metadata``.
    """
    from ouroboros.consciousness_authority import CONSCIOUSNESS_CATEGORY
    from ouroboros.subagent_dispatch_notes import _nanny_route_dispatched_for  # D03->D07 is lazy-only

    meta = dict(_mapping(task.get("metadata")))
    if _nanny_route_dispatched_for(dict(task), None) or _nanny_route_dispatched_for(meta, None):
        return "nanny"
    if str(task.get("delegation_role") or meta.get("delegation_role") or "").strip().lower() == "subagent":
        return "child"
    if (task.get("usage_category") or meta.get("usage_category")) == CONSCIOUSNESS_CATEGORY:
        return "consciousness"
    if is_presence_task(task):
        return "presence"
    return "integrator"


def _explicit_problem(explicit: Any) -> str:
    if not isinstance(explicit, Mapping):
        return "memory_view is an object of ViewSpec fields"
    unknown = sorted(set(explicit) - set(_FIELDS))
    if unknown:
        return "unknown ViewSpec field(s): " + ", ".join(map(str, unknown))
    for name, value in explicit.items():
        if name == "role" and value not in ROLES:
            return f"role is one of {', '.join(ROLES)}"
        if name == "live_rooms" and value not in LIVE_ROOMS:
            return f"live_rooms is one of {', '.join(LIVE_ROOMS)}"
        if name == "marks" and value not in MARKS:
            return f"marks is one of {', '.join(MARKS)}"
        if name == "room_id" and not (value is None or isinstance(value, (str, int)) and not isinstance(value, bool)):
            return "room_id is a room id or null"
        if _FIELDS[name] == "bool" and not isinstance(value, bool):
            return f"{name} is true or false"
    return ""


def _task_ctx(task: Mapping[str, Any]) -> SimpleNamespace:
    """The namespace ``own_room_chat`` reads, built from a queue task record."""
    meta = dict(_mapping(task.get("metadata")))
    for key in ("parent_task_id", "root_task_id", "delegation_role"):
        if task.get(key) not in (None, ""):
            meta.setdefault(key, task[key])
    return SimpleNamespace(task_id=str(task.get("id") or task.get("task_id") or ""), task_metadata=meta,
                           current_chat_id=task.get("chat_id"))


def view_room_id(task: Mapping[str, Any], ctx: Any, drive_root: Any, *, project_chat_ids: Any,
                 presence: bool = False) -> str:
    """The current room of a view: ``own_room_chat`` (binding, then chat, then ancestors).

    A result outside the Projects and Main, or none (the hidden partition, another
    transport chat, a task without a chat), is Main, as the old focused view was; a
    Presence turn keeps its own conversation.
    """
    from ouroboros.dialogue_evidence import own_room_chat  # D03->D06 is lazy-only

    try:
        own = own_room_chat(ctx if ctx is not None else _task_ctx(task), drive_root)
    except (TypeError, ValueError, OSError):
        own = None
    if own is not None and (presence or str(own) == MAIN_ROOM or int(own) in set(project_chat_ids or ())):
        return str(own)
    return MAIN_ROOM


def view_spec_for_task(task: Mapping[str, Any], drive_root: Any, *, ctx: Any = None) -> ViewSpec:
    """The role's default spec, or the task's explicit ``memory_view`` after its fields check.

    An explicit spec with an unknown field or value is not used: the role default is, and the
    refusal is one event in ``logs/events.jsonl``. Every role but consciousness gets its current room.
    """
    role = view_role(task)
    spec = ROLE_DEFAULTS[role]
    explicit = task.get("memory_view")
    if explicit is not None:
        problem = _explicit_problem(explicit)
        if problem:
            log.warning("memory_view of task %s refused, role default used: %s", task.get("id"), problem)
            append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {"ts": utc_now_iso(), "role": role, "reason": problem,
                         "type": "context_memory_view_spec_refused", "task_id": str(task.get("id") or "")})
        else:
            spec = dataclasses.replace(spec, **dict(explicit))
    if spec.room_id is not None:
        return dataclasses.replace(spec, room_id=str(spec.room_id))
    if spec.role in _ROOMLESS:
        return spec
    from ouroboros.memory_inventory import membership_facts

    projects = membership_facts(drive_root).project_chat_ids
    return dataclasses.replace(spec, room_id=view_room_id(task, ctx, drive_root, project_chat_ids=projects,
                                                          presence=spec.role == "presence"))


# --- the captured facts ----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MemoryViewSnapshot:
    """Everything a render needs, read once per request; ``snapshot_json`` is its canonical form."""
    spec: ViewSpec
    store_status: Dict[str, Any]  # {"state": "active" | an import kind, "reason"?}
    frontier: Dict[str, Any]  # {"status", "pos"} of the legacy frontier
    story: Tuple[Dict[str, Any], ...] = ()  # legacy pointers, then pages and parts, in story order
    room: Optional[Dict[str, Any]] = None  # the current room's facts and texts
    live_rooms: Tuple[Dict[str, Any], ...] = ()
    marks: Tuple[Dict[str, Any], ...] = ()
    legacy_blocks: Dict[str, Any] = dataclasses.field(default_factory=dict)  # the story status facts
    fallback_refusals: Tuple[Dict[str, Any], ...] = ()
    owner_words: str = ""  # the rendered block of the owner's words that caused a helper's work

    @property
    def active(self) -> bool:
        return self.store_status.get("state") == "active"


_TUPLES = ("story", "live_rooms", "marks", "fallback_refusals")


def snapshot_json(snapshot: MemoryViewSnapshot) -> str:
    """The canonical JSON of a snapshot (sorted keys): the context core keeps it and hashes it."""
    return json.dumps(dataclasses.asdict(snapshot), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def snapshot_from_json(text: str) -> MemoryViewSnapshot:
    data = json.loads(text)
    data["spec"] = ViewSpec(**data["spec"])
    for name in _TUPLES:
        data[name] = tuple(data.get(name) or ())
    return MemoryViewSnapshot(**data)


def _owner_words_block(task: Mapping[str, Any]) -> str:
    from ouroboros.owner_words import render_owner_words, task_governing_words  # D03->D17 is lazy-only

    meta = _mapping(task.get("metadata"))
    root = str(task.get("root_task_id") or meta.get("root_task_id") or "")
    rows, absent = task_governing_words(task)
    return render_owner_words(rows, absent, audience="child", root_task_id=root, indent=INDENT)


def _activation(store: ChronicleStore) -> Dict[str, Any]:
    """``{"state": "active"}`` after the one legacy import, else the import's kind and reason."""
    try:
        receipt = store.ensure_activated()
    except Exception as exc:  # the store's own refusal to open its journal; the view must still render
        log.warning("memory view: the chronicle cannot be activated", exc_info=True)
        return {"state": "journal_unreadable", "reason": f"{type(exc).__name__}: {exc}"}
    if isinstance(receipt, dict) and receipt.get("kind") == "activation":
        return {"state": "active"}
    receipt = receipt if isinstance(receipt, dict) else {}
    return {"state": str(receipt.get("kind") or "import_failed"),
            "reason": str(receipt.get("reason") or receipt.get("kind") or "no activation receipt")}


# --- the story: legacy pointers, pages and parts ---------------------------------------------------

def _legacy_period(pointer: Mapping[str, Any], rows: Any = None) -> str:
    """The pointer's period: its room's own rows (``LegacyUnit.ts_span``), else its block's, labelled as the block's."""
    if _mapping(rows).get("start") or _mapping(rows).get("end"):
        return _period(rows)
    raw = _mapping(_mapping(pointer.get("covers")).get("raw_range"))
    span = _mapping(raw.get("ts_span"))
    if raw.get("status") == "exact" and (span.get("start") or span.get("end")):
        return _period(span) + " (block period)"
    if pointer.get("range_text"):
        return f"{pointer['range_text']} (block period)"
    return "period known from the retelling text only"


def _legacy_span(rows: Any) -> List[str]:
    """``[start, end]`` minutes of the room's own rows with both bounds, else ``[]`` (the floor merges them)."""
    span = _mapping(rows)
    return [_minute(span["start"]), _minute(span["end"])] if span.get("start") and span.get("end") else []


def _gap_detail(store: ChronicleStore, pointer: Mapping[str, Any]) -> str:
    record = store.get(pointer["node_id"]) or {}
    return str(_mapping(record.get("metadata")).get("source_gap") or "the old writer marked this period as a gap")


def _stamp_summary(stamp: Any) -> str:
    """``host stamp: 3 tasks — completed 2, failed 1``: statuses only, never the time it was computed."""
    tasks = [entry for entry in _mapping(stamp).get("tasks") or [] if isinstance(entry, Mapping)]
    if not tasks:
        return ""
    counts = Counter(str(entry.get("status") or "unknown") for entry in tasks)
    return f"host stamp: {len(tasks)} tasks — " + ", ".join(f"{status} {n}" for status, n in sorted(counts.items()))


def _fixes(store: ChronicleStore, record: Mapping[str, Any], fixes: Mapping[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """The mind's corrections and rejections aimed at a part's members, nested parts' too, one per record."""
    members = store.folded_members(record["id"]) if record.get("kind") == "part" else []
    return [{"kind": fix["kind"], "target": str(member), "text": str(fix.get("text") or fix.get("reason") or "")}
            for member in members for fix in fixes.get(str(member), ())]


def _story_pages(store: ChronicleStore, label: Callable[..., str]) -> Tuple[List[Dict[str, Any]], int]:
    """Every acting page and part not folded into a part, all rooms, by ``stream_span[0]`` then sequence."""
    fixes: Dict[str, List[Dict[str, Any]]] = {}
    for record in store.records(kinds=("correction", "decision")):
        if record["kind"] == "correction" or record.get("accepted") is False:
            fixes.setdefault(str(record.get("target_id") or ""), []).append(
                {"kind": "correction" if record["kind"] == "correction" else "rejection",
                 "text": record.get("text"), "reason": record.get("reason")})
    rooms = sorted({str(record["room_id"]) for record in store.records(kinds=("page", "part"))})
    keyed, mine = [], 0
    for room in rooms:
        for record in store.room_records(room):
            if record["kind"] not in ("page", "part"):
                continue
            mine += record["kind"] == "page" and _mapping(record.get("author")).get("kind") == "mind"
            if record.get("folded_into"):
                continue
            covers = _mapping(record.get("covers"))
            span = covers.get("stream_span")
            first = span[0] if isinstance(span, list) and span and type(span[0]) is int else -1
            keyed.append(((first, record["sequence"]), {
                "kind": record["kind"], "id": record["id"], "room_id": room,
                "label": str(_mapping(record.get("metadata")).get("room_label") or label(room)),
                "period": _period(covers.get("ts_span")), "text": str(record.get("current_text") or ""),
                "status": str(record.get("status") or ""), "signer": draft_signer(record.get("author")),
                "revision": record["revision"] if record.get("revision") != record["id"] else "",
                "stamp": _stamp_summary(record.get("host_stamp")), "fixes": _fixes(store, record, fixes), "quotes": record.get("quotes") or []}))
    return [entry for _key, entry in sorted(keyed, key=lambda pair: pair[0])], mine


def _capture_story(store: ChronicleStore, root: pathlib.Path, label: Callable[..., str], *,
                   whole_first: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """``(story entries, story status, helper refusals)``; folded or not is ``memory_inventory``'s verdict.

    ``whole_first`` (an integrating focus): a record of the first block keeps its current words, so it renders whole.
    """
    pointers = {pointer["node_id"]: pointer for pointer in store.legacy_pointer_rows()}
    units = memory_inventory.legacy_units(store, root)
    first = sorted({unit.room_id for unit in units if whole_first and unit.block == 0 and not unit.folded})
    words = {record["id"]: str(record.get("current_text") or "") for room in first
             for record in store.room_records(room) if record["kind"] == "legacy"}
    story, refusals = [], []
    for unit in units:
        if unit.folded:
            continue
        pointer = pointers.get(unit.record_id) or {"node_id": unit.record_id}
        gap = pointer.get("kind") == "gap" or pointer.get("legacy_type") in ("gap", "cursor_gap")
        entry = {"kind": "legacy", "id": unit.record_id, "room_id": unit.room_id, "block": unit.block,
                 "label": str(pointer.get("label") or label(unit.room_id)), "period": _legacy_period(pointer, unit.ts_span),
                 "span": _legacy_span(unit.ts_span), "rows": unit.rows if unit.raw == "exact" else None,
                 "chars": unit.retelling_chars, "messages": pointer.get("messages"),
                 "gap": _gap_detail(store, pointer) if gap else ""}
        if unit.block == 0 and not gap and unit.record_id in words:
            entry["text"] = words[unit.record_id]
        story.append(entry)
        if unit.refusal:
            read = _mapping(_mapping(_mapping(unit.refusal.get("response_ref")).get("read")).get("arguments"))
            refusals.append({"id": unit.record_id, "label": entry["label"], "period": entry["period"],
                             "kind": str(unit.refusal.get("kind") or "refused"), "path": str(read.get("path") or "")})
    pages, mine = _story_pages(store, label)
    progress = memory_inventory.legacy_progress(units)
    open_units = [unit for unit in units if not unit.folded]
    status = {"folded": progress["folded"], "total": progress["periods"], "pages_by_me": mine,
              "open_records": len(open_units), "open_rows": sum(unit.rows for unit in open_units),
              "open_chars": sum(unit.retelling_chars for unit in open_units), "helper_route": ""}
    if status["folded"] < status["total"]:
        from ouroboros.model_slots import get_light_model

        status["helper_route"] = get_light_model()
    return story + pages, status, refusals


# --- the live part: marks, live rooms, this room -----------------------------------------------------

Entry = Tuple[Dict[str, Any], Dict[str, Any], int]


def _row_texts(root: pathlib.Path, entries: List[Entry]) -> Dict[str, Dict[str, Any]]:
    """``row_sha256 -> row`` read by address: each hinted generation once, a missed hint by search."""
    wanted: Dict[str, Dict[int, str]] = {}
    for address, _meta, _pos in entries:
        hint = _mapping(address.get("hint"))
        if type(hint.get("line")) is int:
            wanted.setdefault(str(hint.get("gen") or ""), {})[hint["line"]] = address["row_sha256"]
    paths = {sig: path for path, sig in chat_chain.generation_signatures(root)}
    rows: Dict[str, Dict[str, Any]] = {}
    for gen, lines, last in ((gen, lines, max(lines)) for gen, lines in wanted.items()):  # max once, not per line
        if gen not in paths:
            continue
        with paths[gen].open("rb") as handle:
            for number, raw in enumerate(handle, 1):
                sha = lines.get(number)
                row = chat_chain._decoded(raw) if sha else None
                if row is not None and chat_chain.source_row_id(row) == sha:
                    rows[sha] = row
                if number >= last:
                    break
    for address, _meta, _pos in entries:
        if address["row_sha256"] not in rows:
            found, _status = chat_chain.resolve_row(root, address)
            if found is not None:
                rows[address["row_sha256"]] = found
    return rows


def _row_head(entry: Entry, author: Mapping[str, Any]) -> str:
    """``[<ts>; <author>; row:…]`` from a row's facts alone: the header of its line and of its address line."""
    address, meta, _pos = entry
    return f"[{meta.get('ts') or 'time not recorded'}; {author.get('label')}; {chat_chain.format_address(address)}]"


def _row_line(entry: Entry, author: Mapping[str, Any], texts: Mapping[str, Dict[str, Any]]) -> str:
    row = texts.get(entry[0]["row_sha256"])
    if row is None:
        return _row_head(entry, author) + " (row unreadable by its address)"
    return render_memory_row(entry[0], row, author=author, indent=INDENT)


def _spoken(entry: Entry, author: Mapping[str, Any], texts: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A verbatim lane-1 line with the facts its address line needs (head, size, position)."""
    return {"line": _row_line(entry, author, texts), "head": _row_head(entry, author), "kind": author.get("kind"),
            "address": chat_chain.format_address(entry[0]), "ts": entry[1].get("ts"),
            "chars": entry[1].get("text_chars", 0), "pos": entry[2]}


def _typed_fact(meta: Mapping[str, Any]) -> str:
    name = TYPED_HOST_FACTS.get(str(meta.get("type") or ""), "")
    if name == "delivery":
        state = _mapping(_mapping(meta.get("transport")).get("delivery")).get("state")
        name = f"delivery {state or 'state not recorded'}"
    return name


def _task_line(root_id: str, group: List[Tuple[Entry, Dict[str, Any]]], texts: Mapping[str, Dict[str, Any]]) -> str:
    """One line for a task's lane-2 rows: the host's status, never a child's report or a retelling."""
    def summary(kinds: frozenset, task: str, kind: str = "task_summary") -> Optional[Entry]:
        return next((entry for entry, _cls in reversed(group) if entry[1].get("type") == kind
                     and entry[1].get("summary_kind") in kinds and str(entry[1].get("task_id") or "") == task), None)

    # The terminal projection, else a Project's completion row pinned to Main (the host writes it
    # without a kind), else the host's facts row; a helper's retelling is never a status.
    chosen = (summary(_TERMINAL, root_id) or summary(_KINDLESS, root_id, "project_completion_summary")
              or summary(_HOST_FACTS, root_id))
    if chosen is not None:
        status = _indented(render_row_text(texts.get(chosen[0]["row_sha256"]) or chosen[1])).lstrip()
    else:
        (address, meta, _pos), cls = group[-1]
        status = (f"running or unreported; last row {meta.get('type') or meta.get('direction') or 'row'} by "
                  f"{cls['author'].get('label')}, {meta.get('text_chars', 0)} chars, {chat_chain.format_address(address)}")
    children = list(dict.fromkeys(str(entry[1].get("task_id")) for entry, _cls in group
                                  if entry[1].get("task_id") and str(entry[1].get("task_id")) != root_id))
    tally = Counter(str(entry[1].get("status") or "unknown") for entry, _cls in group
                    if entry[1].get("summary_kind") == "terminal_result_projection"
                    and str(entry[1].get("task_id") or "") in children)
    parts = [status]
    if children:
        parts.append(f"children {len(children)}" + (" (" + ", ".join(f"{name} {n}" for name, n in sorted(tally.items()))
                                                    + ")" if tally else ""))
    first, last = group[0][0], group[-1][0]
    parts += [f"rows {len(group)}", f"get_task_result(task_id='{root_id}')",
              f"{chat_chain.format_address(first[0])}..{chat_chain.format_address(last[0])}"]
    parts += [f"{name}: {chat_chain.format_address(entry[0])}" for entry, _cls in group
              if (name := _typed_fact(entry[1]))]
    return f"[{first[1].get('ts') or 'time not recorded'}; host; task {root_id}] " + "; ".join(parts)


def _loose_line(entry: Entry, cls: Mapping[str, Any], texts: Mapping[str, Dict[str, Any]]) -> str:
    """A lane-2 row without a task: a host notice by its words, anything else by type and size only."""
    address, meta, _pos = entry
    head = f"[{meta.get('ts') or 'time not recorded'}; {cls['author'].get('label')}; {chat_chain.format_address(address)}]"
    fact = _typed_fact(meta)
    if fact:
        return f"{head} {fact}"
    if cls["author"].get("kind") == "host":
        return f"{head} " + _indented(render_row_text(texts.get(address["row_sha256"]) or meta)).lstrip()
    return f"{head} {meta.get('type') or meta.get('direction') or 'row'}, {meta.get('text_chars', 0)} chars, read by address"


def _lanes(root: pathlib.Path, entries: List[Entry], lineage: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]],
                                                                                            List[Dict[str, Any]]]:
    """Lane 1 verbatim (people and my words to them) and lane 2 as one line per root task."""
    classed = [(entry, row_class(entry[1], pos=entry[2], **lineage)) for entry in entries]
    groups: Dict[str, List[Tuple[Entry, Dict[str, Any]]]] = {}
    order: List[Tuple[int, str, Any]] = []
    needed: List[Entry] = []
    for entry, cls in classed:
        meta = entry[1]
        if cls["lane"] == 1:
            needed.append(entry)
            continue
        task = str(meta.get("root_task_id") or meta.get("task_id") or "")
        if task and task not in groups:
            order.append((entry[2], task, None))
        if task:
            groups.setdefault(task, []).append((entry, cls))
        else:
            order.append((entry[2], "", (entry, cls)))
        if meta.get("type") in ("task_summary", "project_completion_summary") or (
                not task and cls["author"].get("kind") == "host"):
            needed.append(entry)
    texts = _row_texts(root, needed)
    lane1 = [_spoken(entry, cls["author"], texts) for entry, cls in classed if cls["lane"] == 1]
    lane2 = []
    for pos, task, loose in order:
        rows = groups[task] if task else [loose]
        line = _task_line(task, rows, texts) if task else _loose_line(loose[0], loose[1], texts)
        lane2.append({"line": line, "task": task, "ts": rows[0][0][1].get("ts"), "pos": pos,
                      "first": chat_chain.format_address(rows[0][0][0]), "last_ts": rows[-1][0][1].get("ts"),
                      "last": chat_chain.format_address(rows[-1][0][0]), "last_pos": rows[-1][0][2]})
    return lane1, lane2


def _notes_by_room(store: ChronicleStore) -> Dict[str, List[Dict[str, Any]]]:
    """My notes no page has sealed yet (``note:<id>`` outside the room's sealed set), by room."""
    rooms = sorted({str(record["room_id"]) for record in store.records(kinds=("note",))})
    notes: Dict[str, List[Dict[str, Any]]] = {}
    for room in rooms:
        sealed = store.sealed_row_refs(room)
        kept = [{"id": record["id"], "date": _minute(record.get("ts")).split(" ")[0],
                 "role": str(_mapping(_mapping(record.get("author")).get("focus")).get("role") or "mind"),
                 "text": str(record.get("current_text") or "")}
                for record in store.room_records(room)
                if record["kind"] == "note" and f"note:{record['id']}" not in sealed]
        if kept:
            notes[room] = kept
    return notes


def _origins(root: pathlib.Path, room: str, lane_rows: List[Entry]) -> List[Dict[str, Any]]:
    """The owner's words that started this Project, when no open row of the room carries them."""
    from ouroboros.project_dialogue import _source_ref_identity, project_origin_rows  # D03->D17 is lazy-only

    present = {tuple(key) for _address, meta, _pos in lane_rows for key in meta.get("source_keys") or ()}
    return [{"ts": str(origin["ref"].get("ts") or ""), "text": origin["text"],
             "ref": f"chat {origin['ref'].get('chat_id')} / {origin['ref'].get('client_message_id') or 'no client id'}"}
            for origin in project_origin_rows(root, int(room)) if _source_ref_identity(origin["ref"]) not in present]


def _capture_room(store: ChronicleStore, root: pathlib.Path, spec: ViewSpec, label: Callable[..., str],
                  entries: List[Entry], notes: Mapping[str, List[Dict[str, Any]]],
                  lineage: Mapping[str, Any], whole: Any = frozenset()) -> Dict[str, Any]:
    """The current room: its head, page (retold records, pages under parts, notes), origin words and lanes.

    A retold record my story already shows whole (``whole``: the first block, to an integrating focus) is not repeated.
    """
    room = str(spec.room_id)
    records = store.room_records(room)
    sample = next((meta for _address, meta, _pos in reversed(entries) if meta.get("transport")), None)
    facts: Dict[str, Any] = {"room_id": room, "label": label(room, sample), "head": store.room_head(room),
                             "legacy": [], "under_parts": [], "notes": [], "origins": [], "since": "",
                             "lane1": [], "lane2": []}
    if spec.room_page:
        units = {unit.record_id: unit for unit in memory_inventory.legacy_units(store, root)}
        retold = [record for record in records if record["kind"] in ("legacy", "gap")]
        if room == MAIN_ROOM and spec.role != "nanny":  # the room-less retellings (the flat summary, a mixed
            # era) predate rooms and were Main's memory: whole and first for every focus that starts with the top
            # level of the life account (a child included); a nanny carries no life account and no story, so it
            # gets neither the text nor a pointer
            retold[:0] = [record for record in store.room_records(LEGACY_ROOM_ID) if record["kind"] == "legacy"
                          and _mapping(record.get("metadata")).get("legacy_type") not in ("gap", "cursor_gap")]
        facts["legacy"] = [{"id": record["id"], "text": str(record.get("current_text") or ""), "period": _legacy_period(
                                {"covers": record.get("covers"), "range_text": _mapping(record.get("metadata")).get(
                                    "legacy_range_text")}, getattr(unit, "ts_span", None)),
                            "of": LEGACY_ROOM_LABEL if str(record.get("room_id")) == LEGACY_ROOM_ID else ""}
                           for record in retold if not getattr(unit := units.get(record["id"]), "folded", False)
                           and record["id"] not in whole]
        own = {} if spec.story else {e["id"]: {**e, "part": None} for e in _story_pages(store, label)[0] if e["room_id"] == room}  # story order: the floor takes the oldest first
        facts["under_parts"] = [own.get(record["id"]) or {"id": record["id"], "kind": record["kind"], "part": record["folded_into"],
                                 "period": _period(_mapping(record.get("covers")).get("ts_span")), "text": str(record.get("current_text") or "")}
                                for record in (records if spec.story else store.pages_of_room(room)) if record["kind"] in ("page", "part") and (record.get("folded_into") or record["id"] in own)]
        facts["notes"] = list(notes.get(room, ()))
    if spec.origin_words and room.lstrip("-").isdigit() and int(room) in memory_inventory.membership_facts(
            root).project_chat_ids:
        facts["origins"] = _origins(root, room, entries if spec.room_lanes else [])
    if spec.room_lanes and entries:
        facts["since"] = _minute(entries[0][1].get("ts"))
        facts["lane1"], facts["lane2"] = _lanes(root, entries, lineage)
    return facts


def _live_rooms(root: pathlib.Path, spec: ViewSpec, label: Callable[..., str], by_room: Mapping[str, List[Entry]],
                notes: Mapping[str, List[Dict[str, Any]]], lineage: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every other room with open rows or notes not yet sealed, oldest last activity first."""
    rooms = []
    for room in sorted((set(by_room) | set(notes)) - {spec.room_id}):
        entries = by_room.get(room, [])
        classed = [(entry, row_class(entry[1], pos=entry[2], **lineage)) for entry in entries]
        people = [(entry, cls) for entry, cls in classed if cls["lane"] == 1 and cls["author"].get("kind") == "human"]
        words = []
        if spec.live_rooms == "lines_with_words" and people:
            texts = _row_texts(root, [entry for entry, _cls in people])
            words = [_spoken(entry, cls["author"], texts) for entry, cls in people]
        sample = next((meta for _address, meta, _pos in reversed(entries) if meta.get("transport")), None)
        rooms.append({"room_id": room, "label": label(room, sample), "key": entries[-1][2] if entries else -1,
                      "first": _minute(entries[0][1].get("ts")) if entries else "",
                      "last": _minute(entries[-1][1].get("ts")) if entries else "",
                      "people": len(people), "rows": len(entries),
                      "mine": sum(cls["lane"] == 1 and cls["author"].get("kind") == "ouroboros" for _e, cls in classed),
                      "facts": sum(cls["lane"] == 2 for _e, cls in classed),
                      "notes": list(notes.get(room, ())), "words": words})
    return sorted(rooms, key=lambda item: (item["key"], item["room_id"]))


def _mark_target(target: Any) -> str:
    target = _mapping(target)
    if target.get("kind") == "chronicle":
        return f"node {target.get('id')}"
    if target.get("kind") == "chat_row" and target.get("row_sha256"):
        return chat_chain.format_address(dict(target))
    if target.get("kind") == "task":
        return f"task {target.get('task_id')}"
    location = f" at {target['location']}" if target.get("location") else ""
    return f"source {target.get('path') or target.get('kind') or 'not recorded'}{location}"


def _capture_marks(store: ChronicleStore, spec: ViewSpec, label: Callable[..., str]) -> List[Dict[str, Any]]:
    """The acting marks a role keeps in view: every room's, or this room's and the global ones."""
    if spec.marks == "none":
        return []
    marks = store.active_marks(None)
    if spec.marks == "room_and_global":
        marks = [mark for mark in marks if mark.get("scope") == "global" or str(mark.get("room_id")) == spec.room_id]
    entries = []
    for mark in marks:
        author = _mapping(mark.get("author"))
        by = _mapping(author.get("focus")).get("role") or (
            "the old dialogue writer" if author.get("kind") == "legacy_helper" else author.get("kind") or "not recorded")
        entries.append({"id": mark["id"], "scope": str(mark.get("scope") or "room"),
                        "room": label(mark.get("room_id")), "text": str(mark.get("text") or ""), "by": str(by),
                        "date": _minute(mark.get("ts")).split(" ")[0], "target": _mark_target(mark.get("target_ref")),
                        "quote": str(mark.get("quote") or "") if mark.get("visibility", "full") == "full" else ""})
    return sorted(entries, key=lambda entry: (entry["scope"] == "global", "" if entry["scope"] == "global" else entry["room"]))


def _capture_live(store: ChronicleStore, root: pathlib.Path, spec: ViewSpec, label: Callable[..., str],
                  whole: Any = frozenset()) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(this room, live rooms, marks)`` of one capture, from one read of the open rows."""
    lineage = row_lineage(root)
    notes = _notes_by_room(store) if spec.room_page or spec.live_rooms != "none" else {}
    if spec.live_rooms != "none":
        by_room = memory_inventory.open_rows_by_room(root)
    elif spec.room_id is not None and (spec.room_lanes or spec.origin_words):
        by_room = {spec.room_id: memory_inventory.open_room_rows(root, spec.room_id)}
    else:
        by_room = {}
    room = (_capture_room(store, root, spec, label, by_room.get(spec.room_id, []), notes, lineage, whole)
            if spec.room_id is not None else None)
    live = _live_rooms(root, spec, label, by_room, notes, lineage) if spec.live_rooms != "none" else []
    return room, live, _capture_marks(store, spec, label)


def capture_memory_view(drive_root: Any, task: Mapping[str, Any], spec: ViewSpec) -> MemoryViewSnapshot:
    """The facts of one request's view, read once; the chronicle is activated exactly once here.

    Before the import has completed (another importer holds the legacy lock, the
    import was refused, the journal cannot be opened) the snapshot carries only the
    reason and the room id: no story, lanes or live rooms, and no chain read.
    """
    root = pathlib.Path(drive_root)
    owner_words = _owner_words_block(task) if spec.owner_words else ""
    status = _activation(ChronicleStore(root))
    label = _labeler(root)
    bare = {"room_id": spec.room_id, "label": label(spec.room_id)} if spec.room_id is not None else None
    if status["state"] != "active":
        return MemoryViewSnapshot(spec=spec, store_status=status, frontier={}, room=bare, owner_words=owner_words)
    try:
        store = ChronicleStore(root)
        frontier = legacy_frontier(store)
        story, story_status, refusals = (_capture_story(store, root, label, whole_first=spec.role in INTEGRATING)
                                         if spec.story else ([], {}, []))
        room, live, marks = _capture_live(store, root, spec, label, {entry["id"] for entry in story if entry.get("text")
                                                                     and entry.get("kind") == "legacy"})
    except Exception as exc:  # a journal that turns unreadable mid-capture still leaves a view with its reason
        log.warning("memory view: the chronicle could not be read", exc_info=True)
        return MemoryViewSnapshot(spec=spec, store_status={"state": "journal_unreadable",
                                                           "reason": f"{type(exc).__name__}: {exc}"},
                                  frontier={}, room=bare, owner_words=owner_words)
    return MemoryViewSnapshot(spec=spec, store_status=status,
                              frontier={"status": frontier.get("status"), "pos": frontier.get("pos")},
                              story=tuple(story), room=room, live_rooms=tuple(live), marks=tuple(marks),
                              legacy_blocks=story_status, fallback_refusals=tuple(refusals), owner_words=owner_words)


# --- rendering ------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FloorLevel:
    """The physical floor's verdict: per step, the ids of the elements shown only by address.

    The empty level is the full view; ``memory_floor.fit_memory_view`` fills it in ladder
    order. ``by_budget`` counts the elements only an owner-selected mode target took (the
    window alone would have kept them).
    """
    addressed: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    by_budget: int = 0

    @property
    def steps(self) -> Tuple[Tuple[str, int], ...]:
        """``(("F1", 7), ("F2", 1))``: how many elements each step turned into addresses."""
        return tuple((step, len(ids)) for step, ids in self.addressed)


FULL_VIEW = FloorLevel()
_STORY_INTRO = ("Sealed pages and parts, oldest first; each is mine unless marked. Anything named here is one "
                "memory_read away by its id.")


def _page_pointer(entry: Mapping[str, Any]) -> str:
    return f"- {entry['label']}; {entry['period']}; {entry['kind']} {entry['id']}; memory_read(node_id='{entry['id']}')"


def _page_lines(entry: Mapping[str, Any]) -> List[str]:
    lines = ["", f"### {entry['label']} · {entry['period']} · {entry['kind']} {entry['id']}", _indented(entry["text"]), *(f"- quote ({q.get('speaker')}, {q.get('address')}): {q.get('text')}" for q in entry.get("quotes") or ())]
    if entry.get("status") == "draft":
        lines.append(f"(draft by a helper ({entry.get('signer') or 'Light'}), not yet accepted or rejected by me)")
    elif entry.get("status") == "accepted":
        lines.append(f"(drafted by a helper ({entry.get('signer') or 'Light'}), accepted by me)")
    facts = [entry.get("stamp") or "", f"corrected by me: {entry['revision']}" if entry.get("revision") else ""]
    if any(facts):
        lines.append("(" + "; ".join(fact for fact in facts if fact) + ")")
    for fix in entry.get("fixes") or ():
        verb = "correction by me of" if fix["kind"] == "correction" else "my rejection of the draft"
        lines.append(f"- {verb} {fix['target']}:\n{_indented(fix['text'])}")
    return lines


def _retold(item: Mapping[str, Any], short: bool = False) -> str:
    """A record of this room's page (retold, under a part, or a storyless view's own page with its evidence), whole or by address with its length (F4)."""
    head = (f"#### {item['kind']} {item['id']} — {item['period']} — under part {item['part']}" if item.get("part")
            else f"#### {item['id']} — {item['period']}" + (f" — {item['of']}" if item.get("of") else ""))
    if short:
        return f"{head} — {len(item['text'])} chars — memory_read(node_id='{item['id']}')"
    return "\n".join([f"{head}\n{_indented(item['text'])}", *(_page_lines(item)[3:] if "stamp" in item else ())])


def _row_pointer(item: Mapping[str, Any], what: str) -> str:
    return f"{item['head']} ({what}, {item['chars']} chars — read by address)"


def _facts_line(room: Mapping[str, Any], items: List[Dict[str, Any]]) -> str:
    """The earliest task lines of this room as one address line with their period (F1)."""
    last = max(items, key=lambda item: item["last_pos"])
    return (f"{len(items)} earlier task{'' if len(items) == 1 else 's'}, {_minute(items[0]['ts'] or '?')} → "
            f"{_minute(last['last_ts'] or '?')}: memory_read(room_id='{room['room_id']}', rows=true, "
            f"from='{items[0]['first']}', to='{last['last']}')")


def _rooms_line(rooms: List[Dict[str, Any]]) -> str:
    """Other live rooms without notes as one line: their period and every id (F1b)."""
    return (f"{len(rooms)} more open room{'' if len(rooms) == 1 else 's'} without notes; "
            f"{min(room['first'] for room in rooms)} → {max(room['last'] for room in rooms)}; "
            "memory_read(room_id=<id>, rows=true) reads each: " + ", ".join(room["room_id"] for room in rooms))


def _live_room(room: Mapping[str, Any], spoken: Any = frozenset()) -> List[str]:
    if room["rows"]:
        lines = [f"### {room['label']} — open {room['first']} → {room['last']}; people {room['people']}, "
                 f"mine {room['mine']}, task facts {room['facts']}", f"memory_read(room_id='{room['room_id']}', rows=true)"]
    else:
        lines = [f"### {room['label']} — no open rows; my notes not yet sealed: {len(room['notes'])}",
                 f"memory_read(room_id='{room['room_id']}')"]
    return lines + _note_lines(room["notes"]) + [
        _row_pointer(word, "words") if word["address"] in spoken else word["line"] for word in room["words"]]


def _status_lines(status: Mapping[str, Any], refusals: Tuple[Dict[str, Any], ...]) -> List[str]:
    """The story status while the old memory is not all folded; one line per helper refusal."""
    if not status or status.get("folded", 0) >= status.get("total", 0):
        return []
    lines = ["", f"Story status: the helper retelling is folded {status['folded']} of {status['total']} blocks; "
                 f"{status['open_records']} retold records are still open ({status['open_rows']} rows, "
                 f"{status['open_chars']} chars of retelling; helper route {status['helper_route']}); "
                 f"pages sealed by me: {status['pages_by_me']}."]
    for refusal in refusals:  # the retelling's own id is not the helper's answer
        read = (f"its answer: read_file(root='runtime_data', path='{refusal['path']}')" if refusal.get("path")
                else "its answer was not retained")
        lines.append(f"A helper could not fold: {refusal['label']}; {refusal['period']}; {refusal['kind']}; {read}")
    return lines


def render_story(snapshot: MemoryViewSnapshot, level: FloorLevel = FULL_VIEW) -> str:
    """Block B's tail, ``## My story``: the retold old memory, then my pages and parts, then the status.

    A retold record of the first block is whole to an integrating focus, every other one a
    pointer. Its bytes depend only on the chronicle, the room labels, the helper route,
    whether the focus integrates and ``level`` (the physical floor's; the empty level renders
    the full view): F3 turns a room's retold records into one line, F5 my oldest pages into
    address lines.
    """
    spec = snapshot.spec
    if not spec.story:
        return ""
    if not snapshot.active:
        return (f"## My story — unavailable now ({snapshot.store_status.get('reason')})\n\n"
                f"The old memory files are untouched: read_file(root='runtime_data', path='{LEGACY_FILE}').")
    gone = dict(level.addressed)
    pointers = [entry for entry in snapshot.story if entry.get("kind") == "legacy"]
    pages = [entry for entry in snapshot.story if entry.get("kind") != "legacy"]
    lines = ["## My story", "", _STORY_INTRO]
    if pointers:
        taken = set(gone.get("F3", ()))
        whole = any(entry.get("text") and str(entry["room_id"]) not in taken for entry in pointers)
        lines += ["", "### Old memory retold by a helper before the update (not lived; "
                      + ("its first block whole, the later ones read by id)" if whole else "read by id)")]
        lines += retold_lines(pointers, taken)
    shown = set(gone.get("F5", ()))
    if shown:  # the oldest pages: a prefix of the story order
        lines += ["", "### My older pages and parts, by address"]
        lines += [_page_pointer(entry) for entry in pages if entry["id"] in shown]
    for entry in pages:
        if entry["id"] not in shown:
            lines += _page_lines(entry)
    if not snapshot.story:
        lines += ["", "No page or part is sealed yet."]
    lines += _status_lines(snapshot.legacy_blocks, snapshot.fallback_refusals)
    return "\n".join(lines)


def _note_lines(notes: List[Dict[str, Any]]) -> List[str]:
    return [f"note {note['id']} by {note['role']} on {note['date']}\n{_indented(note['text'])}" for note in notes]


def _marks_text(marks: Tuple[Dict[str, Any], ...]) -> str:
    lines = ["## Marks I keep in view", ""]
    for mark in marks:
        lines.append(f"- [{mark['scope']}; {mark['room']}] " + _indented(mark["text"]).lstrip()
                     + f" — marked by {mark['by']} on {mark['date']}; target {mark['target']}; "
                     f"memory_mark(release_id='{mark['id']}', reason=…)")
        if mark.get("quote"):
            lines.append(f"{INDENT}quote: " + _indented(mark["quote"]).lstrip())
    return "\n".join(lines)


def _live_text(rooms: Tuple[Dict[str, Any], ...], gone: Mapping[str, List[str]]) -> str:
    quiet, spoken = set(gone.get("F1b", ())), set(gone.get("F6", ()))
    lines = ["## Live rooms", "", "Other rooms with open rows or notes not yet sealed, oldest last activity first."]
    if quiet:
        lines += ["", _rooms_line([room for room in rooms if room["room_id"] in quiet])]
    for room in rooms:
        if room["room_id"] not in quiet:
            lines += ["", *_live_room(room, spoken)]
    return "\n".join(lines)


def _room_text(room: Mapping[str, Any], gone: Mapping[str, List[str]]) -> str:
    """``## This room (<label>) — head <n>``: the room page, the words that started it, my notes, two lanes."""
    retold, mine, people, facts = (set(gone.get(step, ())) for step in ("F4", "F2", "F7", "F1"))
    lines = [f"## This room ({room['label']}) — head {room['head']}"]
    if room["legacy"]:
        lines += ["", "### Retold before the update (helper retelling, not lived)"]
        lines += [_retold(item, item["id"] in retold) for item in room["legacy"]]
    lines += (["", "### Pages under my parts" if all(i["part"] for i in room["under_parts"]) else "### Pages of this room",
               *(_retold(item, item["id"] in retold) for item in room["under_parts"])] if room["under_parts"] else [])
    if room["origins"]:
        lines += ["", "### Words that started this work (retention-proof)"]
        lines += [f"[{item['ts'] or 'time not recorded'}; owner; {item['ref']}] " + _indented(item["text"]).lstrip()
                  for item in room["origins"]]
    if room["notes"]:
        lines += ["", "### My notes not yet sealed"] + _note_lines(room["notes"])
    if room["lane1"]:
        lines += ["", f"### Open conversation since {room['since']} (verbatim: people and my replies)"]
        lines += [_row_pointer(item, "my reply") if item["address"] in mine else _row_pointer(item, "words")
                  if item["address"] in people else item["line"] for item in room["lane1"]]
    if room["lane2"]:
        early = [item for item in room["lane2"] if item["first"] in facts]
        lines += ["", "### Task facts of this conversation (host; one line per task; a row:… address reads with "
                      f"memory_read(room_id='{room['room_id']}', rows=true, from=<address>, to=<address>))"]
        lines += ([_facts_line(room, early)] if early else []) + [
            item["line"] for item in room["lane2"] if item["first"] not in facts]
    if len(lines) == 1:
        lines += ["", "Nothing open, retold or noted in this room."]
    return "\n".join(lines)


def render_room(snapshot: MemoryViewSnapshot, level: FloorLevel = FULL_VIEW, *, floor_note: str = "") -> str:
    """Block C's memory middle: a helper's owner words, marks, live rooms and this room.

    ``level`` is the physical floor's (the empty level renders the full view);
    ``floor_note`` (``### Physical floor``) closes the room, or the live rooms when the
    view has no room. Before the import has completed the room is one line naming the
    reason and a reader that works without the chronicle.
    """
    parts = [snapshot.owner_words]
    room = snapshot.room
    if not snapshot.active:
        if room:
            parts.append(f"## This room ({room.get('label')})\n\nOpen conversation unavailable until my memory is "
                         f"activated ({snapshot.store_status.get('reason')}); read it: chat_history(count=100) — every "
                         f"room, newest first; this room is chat_id {room['room_id']}.")
        return "\n\n".join(part for part in parts if part)
    gone = dict(level.addressed)
    if snapshot.marks:
        parts.append(_marks_text(snapshot.marks))
    if snapshot.live_rooms:
        parts.append(_live_text(snapshot.live_rooms, gone))
    if room:
        parts.append(_room_text(room, gone))
    parts.append(floor_note)
    return "\n\n".join(part for part in parts if part)


# --- the role line ----------------------------------------------------------------------------------

def working_sources_line(spec: ViewSpec, snapshot: Optional[MemoryViewSnapshot] = None) -> str:
    """A helper's ``## Working sources`` section, listed from the same spec that drew its view.

    The list cannot disagree with the view: every loaded item is a field of ``spec``,
    and what a field leaves out is named as not loaded, with the readers that reach it.
    Before my memory is activated the view holds none of the memory items: they are named
    as not loaded, with the reason.
    """
    room = snapshot.room if snapshot is not None and isinstance(snapshot.room, dict) else {}
    where = str(room.get("label") or (f"chat {spec.room_id}" if spec.room_id is not None else ""))
    loaded = ["the Constitution and system prompt", "the navigation of both books", "your identity"]
    missing, memory = [], []
    (memory if spec.story else missing).append("the top level of your story")
    if spec.room_id is not None and spec.room_page:
        memory.append(f"the page of your parent's room {where}")
    if spec.room_id is not None and spec.room_lanes:
        memory.append(f"the open conversation of {where}")
    else:
        missing.append("raw conversations")
    if spec.room_id is not None and spec.origin_words and (snapshot is None or room.get("origins") != []):
        memory.append("the words that started that work")
    if spec.marks != "none":
        memory.append("the memory marks of that room and global ones" if spec.marks == "room_and_global"
                      else "all memory marks")
    if spec.live_rooms != "none":
        memory.append("one line per other live room")
    else:
        missing.append("other rooms' pages")
    if snapshot is not None and not snapshot.active and memory:
        missing.append(f"{', '.join(memory)} (my memory is not activated yet: {snapshot.store_status.get('reason')})")
    else:
        loaded += memory
    (loaded if spec.knowledge else missing).append("knowledge (overview, index, patterns)")
    if spec.owner_words and (snapshot is None or snapshot.owner_words):  # named only when the block was drawn
        loaded.append("the words of my human that caused this work")
    missing += ["the global scratchpad", "earlier task reports"]
    return ("## Working sources\n\n"
            f"Loaded above: {', '.join(loaded)}; your own recent process is loaded below. "
            f"Not loaded: {', '.join(missing)}; memory_read, chat_history, knowledge_read and get_task_result "
            "reach them, as they reach your parent.\n" + CHILD_ROLE_TEXT)
