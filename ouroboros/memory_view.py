"""The resident memory view of the acting mind: one render for every role.

The view replaces the old dialogue history and chat tail in a request. Block B carries my
story (``## My story``); block C the live part (this room's ``## Late review feedback (host)``,
``## Marks I keep in view``, ``## Live rooms`` and ``## This room (<label>) — head <n>``).
Which parts a role sees is one ``ViewSpec``: ``ROLE_DEFAULTS`` by role (Main, a root and a Project task are the
integrator; consciousness; Presence; a delegated child; a nanny), or an explicit spec a
task carries under ``memory_view``. The current room is the task's own room
(``dialogue_evidence.own_room_chat``); outside a Project and Main it is Main (a wake's too:
it reads Main as the Main turn does, plus a directory of other live rooms), except a
Presence turn, which sees its own conversation.

``capture_memory_view`` reads the facts once per request into a ``MemoryViewSnapshot``
(texts included), so rendering for another window or mode reads nothing again. It
activates the chronicle exactly once (the one legacy import, no model call). Until that
import has completed the view says so in one visible line, reads no chain and never
fails the task. What of memory is open and what is folded comes from
``memory_inventory`` only; what a window shows only by an address line is
``memory_floor``'s verdict, a ``FloorLevel``.

The story block depends only on the chronicle, the room labels, the helper route and the
selected form, with no capture time, task id, JSON, relative time or ordinal: while the
one start allowance keeps that form (the conversation answers to it too), the
same bytes for every focus that carries a story (Main, any room's root, consciousness,
Presence and a delegated child alike). Every unfolded record of it is whole — the old
retelling too, no block privileged — until the physical floor or my own selection says
otherwise: an account I wrote across rooms, once selected, stands in the story and tells the
records it names in their place (``memory_view_account``), their count, period and exact composition still visible; the
source reader keeps their detail. Texts of records keep their words and are indented by two
spaces, so their own ``## `` lines never read as sections.

Nothing here publishes a record or calls a model.
"""
from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import logging
import pathlib
import re
from collections import Counter
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros import chat_chain, memory_inventory
from ouroboros.chronicle_import import LEGACY_ROOM_ID, LEGACY_ROOM_LABEL, legacy_frontier, row_lineage
from ouroboros.chronicle_store import (CHILD_DRAFT_RIGHT, DELEGATED_FOCUS_ROLES, ChronicleStore, capture, draft_signer,
                                       is_late_review_mark, source_time_span)
from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
from ouroboros.dialogue_provenance import (RoomLabelResolver, is_presence_task, render_memory_row, render_row_text,
                                           row_class, warm_lineage)
from ouroboros.mark_anchors import describe as describe_anchor
from ouroboros.memory_view_account import account_entries, account_lines, accounts_line, told_map
from ouroboros.memory_view_legacy import INDENT, indented as _indented, retold_lines
from ouroboros.utils import append_jsonl, sha256_text, utc_now_iso

log = logging.getLogger(__name__)

MAIN_ROOM = str(WEB_UI_CHAT_ID)
ROLES = ("integrator", "consciousness", "presence", "child", "nanny")
LIVE_ROOMS = ("none", "lines", "lines_with_words", "directory")
MARKS = ("all", "room_and_global", "none")
LEGACY_FILE = "memory/dialogue_blocks.json"
# A task's typed host facts its lane-2 line names by type and address, never by text or JSON:
# the late review the owner did not see (old task-acceptance rows only; nothing writes them
# now), a failed or uncertain send, a cancel receipt (it may carry an unreviewed model draft),
# a custody notice and a terminal incident (the old passive view showed each of them).
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


def _period(span: Any, note: str = "") -> str:
    """``start → end`` of a source time span, minutes in UTC, then ``note`` (a ``Period``'s: a block's period is
    labelled as the block's); a span with unknown bounds says so."""
    span = _mapping(span)
    if not span.get("start") and not span.get("end"):
        return "period unknown"
    text = f"{_minute(span.get('start') or '?')} → {_minute(span.get('end') or '?')}"
    return (text + " (incomplete)" if span.get("incomplete") else text) + note


def room_labeler(root: pathlib.Path) -> Callable[..., str]:
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
    live_rooms: str = "lines"  # none | lines | lines_with_words | directory
    marks: str = "all"  # all | room_and_global | none
    knowledge: bool = True  # overview, knowledge index, project journal (rendered by the context)
    work_room_levels: int = 1
    owner_words: bool = False  # the owner's words that caused a helper's work (owner_words.py)


ROLE_DEFAULTS: Mapping[str, ViewSpec] = MappingProxyType({
    "integrator": ViewSpec("integrator"),
    # A wake sees ordinary Main and one directory line per other room.
    "consciousness": ViewSpec("consciousness", live_rooms="directory"),
    "presence": ViewSpec("presence", origin_words=False, live_rooms="none", marks="room_and_global"),
    "child": ViewSpec("child", room_lanes=False, live_rooms="none", marks="room_and_global", knowledge=False,
                      owner_words=True),
    "nanny": ViewSpec("nanny", story=False, room_lanes=False, live_rooms="none", marks="room_and_global",
                      knowledge=False, owner_words=True),
})
_FIELDS = {field.name: field.type for field in dataclasses.fields(ViewSpec)}


def view_role(task: Mapping[str, Any]) -> str:
    """The view role, first match wins, in ``knowledge.focus_signature``'s order.

    A nanny is the one dispatch fact (``_nanny_route_dispatched_for``), then a delegated
    child, an explicit focus cause, a wake's ledger category, and legacy Presence.
    Everything else integrates.
    The facts are read at the task's top level or under ``metadata``.
    """
    from ouroboros.consciousness_authority import CONSCIOUSNESS_CATEGORY, WAKE_CAUSE_FOCUS, wake_cause
    from ouroboros.subagent_dispatch_notes import _nanny_route_dispatched_for  # D03->D07 is lazy-only

    meta = dict(_mapping(task.get("metadata")))
    if _nanny_route_dispatched_for(dict(task), None) or _nanny_route_dispatched_for(meta, None):
        return "nanny"
    if str(task.get("delegation_role") or meta.get("delegation_role") or "").strip().lower() == "subagent":
        return "child"
    if wake_cause({**task, **meta}) == WAKE_CAUSE_FOCUS:
        return "presence"
    if (task.get("usage_category") or meta.get("usage_category")) == CONSCIOUSNESS_CATEGORY:
        return "consciousness"
    if is_presence_task(task):
        return "presence"
    return "integrator"


def start_room(task: Mapping[str, Any], role: str) -> int:
    """The role's start work room from the task's own facts, read at its top level or under ``metadata``."""
    from ouroboros.context_budget import start_room_levels

    meta = _mapping(task.get("metadata"))
    return start_room_levels(role, direct_chat=bool(task.get("_is_direct_chat") or meta.get("_is_direct_chat")))


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
    """Use the focus room when supplied, then ``own_room_chat`` for other tasks.

    A result outside the Projects and Main, or none (the hidden partition, another
    transport chat, a task without a chat), is Main, as the old focused view was; a
    Presence turn keeps its own conversation.
    """
    from ouroboros.dialogue_evidence import own_room_chat  # D03->D06 is lazy-only

    focus = _mapping(_mapping(task.get("metadata")).get("focus") or task.get("focus"))
    if presence and focus.get("room_chat_id") is not None:
        return str(focus["room_chat_id"])
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
    refusal is one event in ``logs/events.jsonl``. Every role gets its current room (``view_room_id``);
    a wake, which has no chat of its own, gets Main, and reads it as the Main turn does.
    """
    role = view_role(task)
    spec = dataclasses.replace(ROLE_DEFAULTS[role], work_room_levels=start_room(task, role))
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
    from ouroboros.memory_inventory import membership_facts

    projects = membership_facts(drive_root).project_chat_ids
    return dataclasses.replace(spec, room_id=view_room_id(task, ctx, drive_root, project_chat_ids=projects,
                                                          presence=spec.role == "presence"))


# --- the captured facts ----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MemoryViewSnapshot:
    """Everything a render needs, read once per request; ``snapshot_json`` is its canonical form."""
    spec: ViewSpec
    store_status: Dict[str, Any]  # {"state": "active" | an import kind, "reason"?, "tail_facts": "filling"?}
    frontier: Dict[str, Any]  # {"status", "pos"} of the legacy frontier
    story: Tuple[Dict[str, Any], ...] = ()  # legacy records, pages, parts and selected accounts; told_by keeps source addresses, in story order
    room: Optional[Dict[str, Any]] = None  # the current room's facts and texts
    live_rooms: Tuple[Dict[str, Any], ...] = ()
    other_rooms: Tuple[Dict[str, Any], ...] = ()  # open rows alone keep an address, not a live-room line
    marks: Tuple[Dict[str, Any], ...] = ()
    legacy_blocks: Dict[str, Any] = dataclasses.field(default_factory=dict)  # the story status facts
    fallback_refusals: Tuple[Dict[str, Any], ...] = ()
    owner_words: str = ""  # the rendered block of the owner's words that caused a helper's work
    waiting: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.store_status.get("state") == "active"


_TUPLES = ("story", "live_rooms", "other_rooms", "marks", "fallback_refusals")


def snapshot_json(snapshot: MemoryViewSnapshot) -> str:
    """The canonical JSON of a snapshot (sorted keys): the context core keeps it and hashes it."""
    return json.dumps(dataclasses.asdict(snapshot), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def snapshot_from_json(text: str) -> MemoryViewSnapshot:
    """A captured view's snapshot, parsed once per text: the fit plan, the floor and the wake
    facts of one build share it, so no consumer mutates it."""
    return _parsed_view(text)


@functools.lru_cache(maxsize=4)  # a memory bound, not a policy: a miss parses again
def _parsed_view(text: str) -> MemoryViewSnapshot:
    data = json.loads(text)
    data.pop("candidates", None)  # a saved view from before candidates stopped being produced
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


def _story_pages(store: ChronicleStore, label: Callable[..., str], units: Mapping[str, memory_inventory.LegacyUnit],
                 room_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every acting page and part not folded into a part (``room_id``'s alone, when named), and every account my
    selection shows, all rooms; the records a shown account tells in its place are marked ``told_by``.

    The settled retellings come first — parts and accounts, by the first row of their own room then sequence —
    and then the pages no part or account holds yet, in publication order (sequence). A newly sealed page so
    joins the end and preserves the prefix on routes with prefix caching. With one cache marker for the whole
    block B, a new page misses that block (``context_fit._system_blocks``). A fold rewrites the story from its
    first changed retelling, as before."""
    fixes: Dict[str, List[Dict[str, Any]]] = {}
    for record in store.records(kinds=("correction", "decision")):
        if record["kind"] == "correction" or record.get("accepted") is False:
            fixes.setdefault(str(record.get("target_id") or ""), []).append(
                {"kind": "correction" if record["kind"] == "correction" else "rejection", "id": record["id"],
                 "text": record.get("text"), "reason": record.get("reason")})
    records = [record for record in store.room_records(room_id, kinds=("page", "part"))
               if not record.get("folded_into")]
    from ouroboros.memory_sealing import capture_terminal_rows, terminal_links
    terminal_cache = capture_terminal_rows(store.data_root, records)
    keyed = [((entry.pop("first"), entry["sequence"]), entry) for entry in account_entries(
        store, label, units, fixes, lambda period: _period(period.span, period.note()))]
    for record in records:
        room = str(record["room_id"])
        period = memory_inventory.record_period(store, record, units)
        keyed.append(((period.first, record["sequence"]), {
            "kind": record["kind"], "id": record["id"], "room_id": room,
            "label": str(_mapping(record.get("metadata")).get("room_label") or label(room)),
            "source_period": period.span, "period_basis": period.source,
            "period": _period(period.span, period.note()), "text": str(record.get("current_text") or ""),
            "status": str(record.get("status") or ""), "signer": draft_signer(record.get("author")),
            "stamp": _stamp_summary(record.get("host_stamp")), "fixes": _fixes(store, record, fixes),
            "terminal_links": terminal_links(store.data_root, record, terminal_cache), "quotes": record.get("quotes") or []}))
    told = told_map([entry for _key, entry in keyed if entry["kind"] == "account"])
    ordered = sorted(keyed, key=lambda pair: (1, pair[0][1]) if pair[1]["kind"] == "page" else (0, *pair[0]))
    # The floor spends the oldest covered period first, independently of publication order in B.
    return [{**entry, "floor_order": list(key), **({"told_by": told[entry["id"]]} if entry["id"] in told else {})}
            for key, entry in ordered]


def capture_story(store: ChronicleStore, root: pathlib.Path, label: Callable[..., str],
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """``(story entries, story status, helper refusals)``; folded or not is ``memory_inventory``'s verdict.

    Every unfolded retold record keeps its current words, whatever its block: whole in the story until the
    floor turns its room into one line (F3) or a selected account tells it in its place (``told_by``).
    """
    pointers = {pointer["node_id"]: pointer for pointer in memory_inventory.legacy_pointers(store)}
    units = memory_inventory.legacy_units(store, root)
    open_rooms = {unit.room_id for unit in units if not unit.folded}
    words = {record["id"]: str(record.get("current_text") or "") for room in sorted(open_rooms)
             for record in store.room_records(room, kinds=("legacy",))}
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
        period = memory_inventory.record_period(store, {**pointer, "id": unit.record_id, "kind": "legacy"},
                                                {unit.record_id: unit})
        entry.update(source_period=period.span, period_basis=period.source)
        if not gap and unit.record_id in words:
            entry["text"] = words[unit.record_id]
        story.append(entry)
        if unit.refusal:
            read = _mapping(_mapping(_mapping(unit.refusal.get("response_ref")).get("read")).get("arguments"))
            refusals.append({"id": unit.record_id, "label": entry["label"], "period": entry["period"],
                             "kind": str(unit.refusal.get("kind") or "refused"), "path": str(read.get("path") or "")})
    pages = _story_pages(store, label, {unit.record_id: unit for unit in units})
    told = told_map([entry for entry in pages if entry["kind"] == "account"])  # a retold record an account tells
    story = [{**entry, "told_by": told[entry["id"]]} if entry["id"] in told else entry for entry in story]
    progress = memory_inventory.legacy_progress(units)
    open_units = [unit for unit in units if not unit.folded]
    accounts = store.accounts()
    status = {"folded": progress["folded"], "total": progress["periods"], **memory_inventory.story_counts(store),
              "open_records": len(open_units), "open_rows": sum(unit.rows for unit in open_units),
              "open_chars": sum(unit.retelling_chars for unit in open_units), "helper_route": "",
              "accounts": len(accounts),  # in the common view: whole there, not told by a later account
              "accounts_shown": sum(1 for entry in pages if entry["kind"] == "account" and not entry.get("told_by"))}
    if status["folded"] < status["total"]:
        from ouroboros.model_slots import get_light_model

        status["helper_route"] = get_light_model()
    return story + pages, status, refusals


# --- the live part: marks, live rooms, this room -----------------------------------------------------

Entry = Tuple[Dict[str, Any], Dict[str, Any], int]


def _row_texts(root: pathlib.Path, entries: List[Entry]) -> Dict[str, Dict[str, Any]]:
    """Read exactly the selected row identities from the dialogue store, together (``rows_by_address``)."""
    return chat_chain.rows_by_address(root, [address for address, _meta, _pos in entries])


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
    from ouroboros.dialogue_provenance import row_provenance

    row = texts.get(entry[0]["row_sha256"])
    return {"owner_side": author.get("kind") == "human" and row_provenance(entry[1]) != "place",
            "ref": f"chat {entry[0]['chat_id']} / {entry[1].get('client_message_id') or ''}",
            "text_sha256": sha256_text(str(row.get("text") or "")) if row else "",
            "line": _row_line(entry, author, texts), "head": _row_head(entry, author), "kind": author.get("kind"),
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
    for entry, _cls in group:
        row = texts.get(entry[0]["row_sha256"]) or {}
        if entry != chosen and isinstance(row.get("presence_recovery"), Mapping):
            # A later canonical terminal owns status, but does not erase the earlier
            # source-bound observation. This history is not a new current verdict.
            parts.append(f"observation at {row.get('ts') or 'unknown time'}: " + _indented(render_row_text(row)).lstrip())
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
    by_room: Dict[str, List[Dict[str, Any]]] = {}
    for record in store.room_records(None, kinds=("note",)):
        by_room.setdefault(str(record["room_id"]), []).append(record)
    notes: Dict[str, List[Dict[str, Any]]] = {}
    for room in sorted(by_room):
        sealed = store.sealed_row_refs(room)
        kept = [{"id": record["id"], "date": _minute(record.get("ts")).split(" ")[0],
                 "author_kind": str(_mapping(record.get("author")).get("kind") or ""),
                 "task_id": str(record.get("author", {}).get("task_id") or ""),
                 "role": str(_mapping(_mapping(record.get("author")).get("focus")).get("role") or "mind"),
                 "origin": dict(_mapping(_mapping(record.get("author")).get("focus"))),
                 "text": str(record.get("current_text") or "")}
                for record in by_room[room] if f"note:{record['id']}" not in sealed]
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
                  lineage: Mapping[str, Any], story: List[Dict[str, Any]] = (),
                  gaps: Optional[Mapping[str, set]] = None) -> Dict[str, Any]:
    """The current room: its head and standing facts, page (retold records, pages under parts, notes), origin words, lanes.

    Retold records already whole in my story are not repeated. A selected account carries its records, a part
    it tells its members too; their detail stays at their source readers rather than reappearing in this room.
    A standing part (the mind's, or a helper's draft not pending as a candidate) stands for its members here
    as in the story: they keep their period and address (``standing``), a viewer with a story only.
    """
    room = str(spec.room_id)
    records = store.room_records(room)
    whole = {e["id"] for e in story if e.get("kind") == "legacy" and e.get("text") and not e.get("told_by")}
    told = set(told_map([e for e in story if e.get("kind") == "account"]))  # the latest shown selections
    told |= {member for record in records if record["kind"] == "part" and record["id"] in told
             for member in store.folded_members(record["id"])}
    standing = {record["id"] for record in records if record["kind"] == "part"}  # room_records: no candidate or rejected
    sample = next((meta for _address, meta, _pos in reversed(entries)
                   if meta.get("transport") or meta.get("presence_provenance")), None)
    facts: Dict[str, Any] = {"room_id": room, "label": label(room, sample), "head": store.room_head(room),
                             "facts": _with_gaps(memory_inventory.room_facts(store, room, entries, lineage,
                                                                             len(notes.get(room, ()))), gaps, room),
                             "legacy": [], "under_parts": [], "notes": [], "origins": [], "since": "",
                             "lane1": [], "lane2": []}
    if spec.room_page:
        units = {unit.record_id: unit for unit in memory_inventory.legacy_units(store, root)}
        retold = [record for record in records if record["kind"] in ("legacy", "gap")]
        if room == MAIN_ROOM and spec.role != "nanny":  # the room-less retellings (the flat summary, a mixed
            # era) predate rooms and were Main's memory: whole and first for every focus that starts with the top
            # level of the life account (a child included); a nanny carries no life account and no story, so it
            # gets neither the text nor a pointer
            retold[:0] = [record for record in store.room_records(LEGACY_ROOM_ID, kinds=("legacy",))
                          if _mapping(record.get("metadata")).get("legacy_type") not in ("gap", "cursor_gap")]
        facts["legacy"] = [{"id": record["id"], "text": str(record.get("current_text") or ""), "period": _legacy_period(
                                {"covers": record.get("covers"), "range_text": _mapping(record.get("metadata")).get(
                                    "legacy_range_text")}, getattr(unit, "ts_span", None)),
                            "of": LEGACY_ROOM_LABEL if str(record.get("room_id")) == LEGACY_ROOM_ID else ""}
                           for record in retold if not getattr(unit := units.get(record["id"]), "folded", False)
                           and record["id"] not in whole and record["id"] not in told]
        own = {} if spec.story else {e["id"]: {**e, "part": None} for e in _story_pages(store, label, units, room)
                                      if e["room_id"] == room and e["kind"] != "account"}  # floor_order retains period age
        facts["under_parts"] = [own.get(record["id"]) or {"id": record["id"], "kind": record["kind"], "part": record["folded_into"],
                                 "period": _period((p := memory_inventory.record_period(store, record, units)).span, p.note()), "text": str(record.get("current_text") or ""),
                                 **({"standing": True} if spec.story and record["folded_into"] in standing else {})}
                                for record in (records if spec.story else store.pages_of_room(room)) if record["kind"] in ("page", "part") and record["id"] not in told and (record.get("folded_into") or record["id"] in own)]
        facts["notes"] = list(notes.get(room, ()))
    if spec.origin_words and room.lstrip("-").isdigit() and int(room) in memory_inventory.membership_facts(
            root).project_chat_ids:
        facts["origins"] = _origins(root, room, entries if spec.room_lanes else [])
    if spec.room_lanes and entries:
        facts["since"] = _minute(facts["facts"]["earliest"])  # the rows' first stored time, not the first row read
        facts["lane1"], facts["lane2"] = _lanes(root, entries, lineage)
    return facts


def _room_focus(root, by_room, notes, shown_rooms, running):
    """Latest authored focus only for rooms actually shown in the directory."""
    from ouroboros.focus import compact_focus
    from ouroboros.task_store import display_documents

    ids = {room: {str(row.get("task_id") or "") for _a, row, _p in entries}
           for room, entries in by_room.items() if room in shown_rooms}
    for room, entries in notes.items():
        if room in shown_rooms:
            ids.setdefault(room, set()).update(note.get("task_id", "") for note in entries)
    for room in shown_rooms:
        ids.setdefault(room, set()).update(running.get(room, ()))
    bindings = memory_inventory.membership_facts(root).bindings
    for task_id, room in bindings.items():
        if str(room) in ids:
            ids[str(room)].add(task_id)
    documents = display_documents(root, set().union(*ids.values()) if ids else ())
    documents = {task_id: json.loads(text) for task_id, text in documents.items()}
    focus = {task_id: compact_focus(doc.get("focus")) for task_id, doc in documents.items()}
    dated = {}
    for task_id, line in focus.items():
        try:
            dated[task_id] = datetime.datetime.fromisoformat(line["authored_at"]).timestamp()
        except (TypeError, ValueError, OverflowError):
            continue
    latest = {}
    for room, tasks in ids.items():
        candidates = [focus[t] for t in tasks if t in dated and focus[t].get("author_task_id") == t]
        if candidates:
            line = max(candidates, key=lambda item: (dated[item["author_task_id"]], item["author_task_id"]))
            latest[room] = f"{line['text']} (task {line['author_task_id']}, {_minute(line['authored_at'])})"
    return latest


def _live_rooms(store: ChronicleStore, root: pathlib.Path, spec: ViewSpec, label: Callable[..., str],
                by_room: Mapping[str, List[Entry]], notes: Mapping[str, List[Dict[str, Any]]],
                lineage: Mapping[str, Any], gaps: Optional[Mapping[str, set]] = None,
                quiet: Optional[Mapping[str, List[Entry]]] = None, projects=None) -> List[Dict[str, Any]]:
    """All other open rooms, with a live verdict; an inactive room keeps only its directory address."""
    from ouroboros.dialogue_provenance import row_provenance
    from ouroboros.memory_room_activity import considered_through, running_by_room, unconsidered_people

    rooms = []
    if projects is None:  # the explicit room-directory reader has no waiting census to reuse
        from ouroboros.projects_registry import list_projects
        projects = list_projects(root)
    running = running_by_room(root, by_room, notes, memory_inventory.membership_facts(root).bindings, projects)
    shown = {entry[0]["row_sha256"] for entry in by_room.get(spec.room_id, [])} if spec.room_lanes else set()
    for room in sorted((set(by_room) | set(notes) | {room for room, kinds in (gaps or {}).items() if kinds}
                        | {room for room, tasks in running.items() if tasks})
                       - {spec.room_id}):
        entries = by_room.get(room, [])
        words = []
        if spec.live_rooms == "lines_with_words":
            people = [(entry, cls) for entry in entries if entry[0]["row_sha256"] not in shown
                      and row_provenance(entry[1]) != "place"
                      and (cls := row_class(entry[1], pos=entry[2], **lineage))["lane"] == 1
                      and cls["author"].get("kind") == "human"]
            texts = _row_texts(root, [entry for entry, _cls in people]) if people else {}
            words = [_spoken(entry, cls["author"], texts) for entry, cls in people]
            shown.update(entry[0]["row_sha256"] for entry, _cls in people)
        sample = next((meta for _address, meta, _pos in reversed(entries)
                       if meta.get("transport") or meta.get("presence_provenance")), None)
        facts = _with_gaps(memory_inventory.room_facts(store, room, entries, lineage, len(notes.get(room, ()))), gaps, room)
        facts["unconsidered_people"] = unconsidered_people(entries, lineage, considered_through((quiet or {}).get(room, ()), lineage))
        facts["running_tasks"] = running.get(room, [])
        # A room can have only notes left after its rows are sealed. Its writer's
        # typed origin still keeps those place notes out of the integrating view.
        if not facts.get("place"):
            origin = next((note["origin"] for note in notes.get(room, ())
                           if note.get("origin", {}).get("room_id") == room
                           and (note["origin"].get("place_id") or note["origin"].get("conversation_key"))), None)
            if origin:
                facts["place"] = {"place_id": origin.get("place_id", ""),
                                  "conversation_key": origin.get("conversation_key", ""),
                                  "answered_events": 0, "last_address": "no open rows"}
        rooms.append({"room_id": room, "label": label(room, sample), "key": entries[-1][2] if entries else -1,
                      "live": bool((spec.live_rooms == "lines_with_words" and facts["people"])
                                   or facts["unconsidered_people"] or facts["notes"] or facts["running_tasks"] or facts.get("gaps")),
                      "facts": facts, "directory": spec.live_rooms == "directory", "focus": "",
                      "notes": list(notes.get(room, ())), "words": words})
    if spec.live_rooms == "directory":
        focus = _room_focus(root, by_room, notes, {room["room_id"] for room in rooms if room["live"]}, running)
        for room in rooms:
            room["focus"] = focus.get(room["room_id"], "")
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


def source_reader(target: Mapping[str, Any], source_read: Any = None) -> str:
    """The reader a retained-source target names to every focus: the writer's explicit reader, else
    ``memory_read`` over the compact ref, verified on read. An author's own ``read`` recipe resolves in
    the author's drive only, so it never travels (BIBLE P1: a source this actor can actually resolve)."""
    if isinstance(source_read, Mapping):
        return f"{source_read.get('tool')}({json.dumps(source_read.get('arguments') or {}, ensure_ascii=False, sort_keys=True)})"
    ref = {key: target[key] for key in ("kind", "root", "path", "size", "sha256", "task_id") if key in target}
    return f"memory_read(source_ref={json.dumps(ref, ensure_ascii=False, sort_keys=True)})"


def _late_review_entry(mark: Mapping[str, Any], label: Callable[..., str], *, own_room: bool = True) -> Dict[str, Any]:
    """A host-kept late review (``late_review_marks``): its words, when it settled, its full source and its release."""
    review, target = _mapping(mark.get("review")), _mapping(mark.get("target_ref"))
    reader = (source_reader(target, review.get("source_read")) if own_room
              else f"memory_read(node_id={json.dumps(mark['id'])})")  # the full mark names the same reader
    return {"id": mark["id"], "scope": "room", "room": label(mark.get("room_id")),
            "text": str(mark.get("text") or "") if own_room else "",
            "by": "host", "date": _minute(mark.get("ts")).split(" ")[0], "target": _mark_target(target), "quote": "",
            "late_review": {"task_id": str(review.get("task_id") or ""), "source": reader,
                            "sha256": str(target.get("sha256") or ""), "own_room": own_room}}


def _capture_marks(store: ChronicleStore, spec: ViewSpec, label: Callable[..., str], marks=None) -> List[Dict[str, Any]]:
    """The acting marks a role keeps in view: every room's, or this room's and the global ones.

    Late review feedback the host keeps in view belongs to the room of its work whatever the role's
    mark scope: that room holds its words; consciousness gets pointers for foreign rooms.
    Ordinary foreign-room turns keep omitting it.
    """
    if spec.marks == "none":
        return []
    marks = store.active_marks(None) if marks is None else marks
    if spec.marks == "room_and_global":
        marks = [mark for mark in marks if mark.get("scope") == "global" or str(mark.get("room_id")) == spec.room_id]
    entries = []
    for mark in marks:
        if is_late_review_mark(mark):
            if spec.role == "consciousness" or str(mark.get("room_id")) == spec.room_id:
                entries.append(_late_review_entry(mark, label, own_room=str(mark.get("room_id")) == spec.room_id))
            continue
        author = _mapping(mark.get("author"))
        by = _mapping(author.get("focus")).get("role") or (
            "the old dialogue writer" if author.get("kind") == "legacy_helper" else author.get("kind") or "not recorded")
        entries.append({"id": mark["id"], "scope": str(mark.get("scope") or "room"),
                        "room": label(mark.get("room_id")), "text": str(mark.get("text") or ""), "by": str(by),
                        "date": _minute(mark.get("ts")).split(" ")[0], "target": _mark_target(mark.get("target_ref")),
                        "quote": str(mark.get("quote") or "") if mark.get("visibility", "full") == "full" else "",
                        "anchor": describe_anchor(mark["anchor"]) if isinstance(mark.get("anchor"), dict) else ""})
    return sorted(entries, key=lambda entry: (entry["scope"] == "global", "" if entry["scope"] == "global" else entry["room"]))


def _capture_live(store: ChronicleStore, root: pathlib.Path, spec: ViewSpec, label: Callable[..., str],
                  story: List[Dict[str, Any]] = ()) -> tuple:
    """``(this room, live rooms, marks, waiting)`` from the same captured rows, notes and active marks."""
    lineage = row_lineage(root)
    integrating = spec.role not in DELEGATED_FOCUS_ROLES
    notes = _notes_by_room(store) if integrating or spec.room_id is not None or spec.live_rooms != "none" else {}
    gaps: Dict[str, set] = {}  # by room, what its open rows may lack (``room_appearances.GAP_DETAILS``)
    quiet: Dict[str, List[Entry]] = {}
    if spec.live_rooms != "none":
        by_room = memory_inventory.open_rows_by_room(root, gaps=gaps, quiet=quiet)
    elif spec.room_id is not None:  # the room's facts are counted from its open rows whatever the view prints of them
        by_room = {spec.room_id: memory_inventory.open_room_rows(root, spec.room_id, gaps=gaps.setdefault(spec.room_id, set()))}
    else:
        by_room = {}
    # The authors of rows written before the lineage epoch, read at once for every room (not one task at a time).
    warm_lineage(lineage, ((meta, pos) for entries in by_room.values() for _address, meta, pos in entries))
    warm_lineage(lineage, ((meta, pos) for entries in quiet.values() for _address, meta, pos in entries))
    room = (_capture_room(store, root, spec, label, by_room.get(spec.room_id, []), notes, lineage, story, gaps)
            if spec.room_id is not None else None)
    marks = store.active_marks(None) if spec.marks != "none" or integrating else []
    waiting, projects = {}, None
    if integrating or spec.live_rooms != "none":
        from ouroboros.memory_waiting import measure
        from ouroboros.projects_registry import list_projects
        from ouroboros.store import StoreError

        try:
            projects = list_projects(root, strict=True, oldest_first=True)
        except (OSError, ValueError, StoreError) as exc:
            # An unavailable project census does not make readable room history unavailable.
            log.warning("memory view: project waiting facts unavailable (%s)", type(exc).__name__)
            projects = None
        if integrating:
            waiting = measure(marks, notes, [p for p in projects if "closed" not in p] if projects is not None else None)
    live = _live_rooms(store, root, spec, label, by_room, notes, lineage, gaps, quiet, projects or []) if spec.live_rooms != "none" else []
    return room, live, _capture_marks(store, spec, label, marks), waiting


def capture_memory_view(drive_root: Any, task: Mapping[str, Any], spec: ViewSpec) -> MemoryViewSnapshot:
    """The facts of one request's view, read once; the chronicle is activated exactly once here.

    Before the import has completed (another importer holds the legacy lock, the
    import was refused, the journal cannot be opened) the snapshot carries only the
    reason and the room id: no story, lanes or live rooms, and no chain read.
    """
    root = pathlib.Path(drive_root)
    owner_words = _owner_words_block(task) if spec.owner_words else ""
    status = _activation(ChronicleStore(root))
    label = room_labeler(root)
    bare = {"room_id": spec.room_id, "label": label(spec.room_id)} if spec.room_id is not None else None
    if status["state"] != "active":
        return MemoryViewSnapshot(spec=spec, store_status=status, frontier={}, room=bare, owner_words=owner_words)
    try:
        store = ChronicleStore(root)
        # One state of the chronicle for every reader (activation may publish: before), with the stored tail facts
        # the view reads: every row for the live rooms' open rows, else the story's terminal rows.
        tail = ("all" if spec.live_rooms != "none" else
                "terminal" if spec.story or spec.room_id is not None and spec.room_page else "none")
        with capture(root, tail=tail) as derived:
            frontier = legacy_frontier(store)
            story, story_status, refusals = capture_story(store, root, label) if spec.story else ([], {}, [])
            room, live, marks, waiting = _capture_live(store, root, spec, label, story)
        if room:
            from ouroboros.owner_words import task_governing_words

            # Governing words bind by identity: their ref, or (without one) the row's own exact text.
            governing, _absence = task_governing_words(task)
            ids = {g.get("ref") or sha256_text(g.get("text") or "") for g in governing if g.get("ref") or g.get("text")}
            last_reply = max((item["pos"] for item in room["lane1"] if item["kind"] == "ouroboros"), default=-1)
            for item in room["lane1"]:
                item["immune"] = item.get("owner_side", False) and (
                    item["pos"] > last_reply or bool({item.get("ref"), item.get("text_sha256")} & ids))
                # Open rows have no page coverage. A reply alone is not proof of carried commitments.
                item["carried"] = False
    except Exception as exc:  # a journal that turns unreadable mid-capture still leaves a view with its reason
        log.warning("memory view: the chronicle could not be read", exc_info=True)
        return MemoryViewSnapshot(spec=spec, store_status={"state": "journal_unreadable",
                                                           "reason": f"{type(exc).__name__}: {exc}"},
                                  frontier={}, room=bare, owner_words=owner_words)
    if any(not read.complete for read in ((derived or {}).get(key) for key in ("tail", "terminal_tail")) if read):
        status = {**status, "tail_facts": "filling"}  # its rows were read live: the stored tail is not complete
    return MemoryViewSnapshot(spec=spec, store_status=status,
                              frontier={"status": frontier.get("status"), "pos": frontier.get("pos")},
                              story=tuple(story), room=room, live_rooms=tuple(r for r in live if r["live"]),
                              other_rooms=tuple(r for r in live if not r["live"]), marks=tuple(marks), waiting=waiting,
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
_STORY_INTRO = ("My parts and my accounts across rooms, oldest first, then the sealed pages no part or account holds "
                "yet, in the order they were sealed; each is mine unless marked. "
                "Anything named here is one memory_read away by its id.")


def _page_pointer(entry: Mapping[str, Any]) -> str:
    selection = f"; selection {entry['selection']}" if entry.get("selection") else ""
    return f"- {entry['label']}; {entry['period']}; {entry['kind']}; memory_read(node_id='{entry['id']}'){selection}"


def _page_label(entry: Mapping[str, Any]) -> str:
    """The page's first sentence, without a length cut or an invented summary, and its reader."""
    sentence = re.split(r'(?<=[.!?。！？])\s+', entry['text'].strip(), maxsplit=1)[0]
    attribution = (f" (draft by {entry.get('signer') or 'Light'}, not yet accepted)"
                   if entry.get('status') == 'draft' else '')
    return _page_pointer(entry) + attribution + "\n" + _indented(sentence)


def _told_lines(account_id: str, told: List[Dict[str, Any]]) -> List[str]:
    """One count and period over the selected records, nested selections included; exact IDs stay in the reader."""
    children: Dict[str, List[Dict[str, Any]]] = {}
    for item in told:
        children.setdefault(item["told_by"], []).append(item)
    records, pending = [], [account_id]
    while pending:
        for item in children.get(pending.pop(), ()):
            records.append(item)
            pending.append(item["id"])
    if not records:
        return []
    spans = [_mapping(item.get("source_period")) for item in records]
    span = source_time_span([s[key] for s in spans for key in ("start", "end") if s.get(key)],
                            incomplete=any(not s or s.get("incomplete") for s in spans))
    basis = " (includes block periods)" if any(item.get("period_basis") in ("block", "mixed") for item in records) else ""
    return [f"- {len(records)} story records told through this account (including nested selections); "
            f"{_period(span)}{basis}; exact records and selections in the composition reader"]


def _page_lines(entry: Mapping[str, Any]) -> List[str]:
    head = f"### {entry['label']} · {entry['period']} · {entry['kind']} {entry['id']}"
    if entry.get("kind") == "account":  # its words, compact composition, and new source changes still in full
        return ["", head, *account_lines(entry)]
    lines = ["", head, _indented(entry["text"]), *(f"- quote ({q.get('speaker')}, {q.get('address')}): {q.get('text')}" for q in entry.get("quotes") or ())]
    if entry.get("status") == "draft":
        lines.append(f"(draft by a helper ({entry.get('signer') or 'Light'}), not yet accepted or rejected by me)")
    elif entry.get("status") == "accepted":
        lines.append(f"(drafted by a helper ({entry.get('signer') or 'Light'}), accepted by me)")
    if entry.get("stamp"):
        lines.append(f"({entry['stamp']})")
    lines.extend(entry.get("terminal_links") or ())
    for fix in entry.get("fixes") or ():  # a record's own corrections stand inside its text (chronicle_store._interpret)
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


def _rows_line(room: Mapping[str, Any], items: List[Dict[str, Any]]) -> str:
    first, last = min(items, key=lambda i: i['pos']), max(items, key=lambda i: i['pos'])
    mine = sum(item['kind'] == 'ouroboros' for item in items)
    return (f"{len(items)} earlier rows, oldest first, by address ({mine} my replies, "
            f"{len(items) - mine} people's words), {_minute(first['ts'])} → {_minute(last['ts'])}; "
            f"memory_read(room_id='{room['room_id']}', rows=true, from='{first['address']}', to='{last['address']}'). "
            "Verbatim rows within this range remain below in place.")


def _story_line(items: List[Dict[str, Any]]) -> str:
    return (f"{len(items)} older pages and parts by address, {items[0]['period']} → {items[-1]['period']}; "
            "memory_read(story=true, start=0) lists every id and its exact reader.")


def _facts_line(room: Mapping[str, Any], items: List[Dict[str, Any]]) -> str:
    """The earliest task lines of this room as one address line with their period (F1)."""
    last = max(items, key=lambda item: item["last_pos"])
    return (f"{len(items)} earlier task{'' if len(items) == 1 else 's'}, {_minute(items[0]['ts'] or '?')} → "
            f"{_minute(last['last_ts'] or '?')}: memory_read(room_id='{room['room_id']}', rows=true, "
            f"from='{items[0]['first']}', to='{last['last']}')")


def _with_gaps(facts: Dict[str, Any], gaps: Optional[Mapping[str, set]], room: str) -> Dict[str, Any]:
    """A room's standing facts with what its open rows may lack, when anything (``_facts_text`` names it)."""
    kinds = sorted((gaps or {}).get(room) or ())
    return {**facts, "gaps": kinds} if kinds else facts


# What a room's open rows may lack, as its standing facts say it (``room_appearances.read_gaps`` kinds).
_GAP_WORDS = {"appearances_deriving": "rows it shows from other rooms are still being indexed",
              "reply_target_unreadable": "a Reply whose target cannot be read may belong here"}


def _facts_text(facts: Mapping[str, Any], of: str = "this room") -> str:
    """A room's standing facts (``memory_inventory.room_facts``) as one phrase; the same words wherever a room is named.

    The earliest open row and the last row a page covers stand side by side: a page covers a set of rows,
    an earlier row can stay open, and the gap shows from the two facts without a frontier.
    """
    rows = (f"open {_minute(facts['earliest'])} → {_minute(facts['latest'])}; people {facts['people']}, "
            f"mine {facts['mine']}, task facts {facts['task_facts']}, ~{facts['chars']} chars" if facts["rows"]
            else "no open rows")
    rows += "".join(f"; incomplete: {_GAP_WORDS.get(kind, kind)}" for kind in facts.get("gaps") or ())
    if facts.get("running_tasks"):
        rows += "; running work: " + ", ".join(f"get_task_result(task_id='{task}')" for task in facts["running_tasks"])
    notes = f"; my notes not yet sealed: {facts['notes']}" if of == "this room" else ""
    pages = (f"pages of {of}: {facts['pages']}, last covered row {_minute(facts['last_covered'])}" if facts["pages"]
             else f"no page of {of} yet")
    return f"{rows}{notes}; {pages}"


def _rooms_line(rooms: List[Dict[str, Any]]) -> str:
    """A bounded aggregate with a directory reader, regardless of the room count."""
    facts = [room["facts"] for room in rooms]
    summed = {key: sum(item.get(key, 0) for item in facts) for key in ("rows", "people", "mine", "task_facts", "chars", "notes", "pages")}
    summed.update(earliest=min((item["earliest"] for item in facts if item["earliest"]), default=""),
                  latest=max((item["latest"] for item in facts), default=""),
                  last_covered=max((item["last_covered"] for item in facts), default=""))
    return f"{len(rooms)} open rooms; {_facts_text(summed, 'these rooms')}; memory_read(room_id='*', start=0)"


def _place_pointer(place: Mapping[str, Any], notes: List[Dict[str, Any]]) -> List[str]:
    """A place room stays a pointer: its standing facts and one exact reader per note, never their words."""
    return [f"Place {place.get('place_id') or '?'}; conversation {place.get('conversation_key') or '?'}; "
            f"answered events {place.get('answered_events', 0)}; last row {place.get('last_address') or '?'}",
            *(f"note {note['id']}: memory_read(node_id='{note['id']}')" for note in notes)]


def _directory_room(room: Mapping[str, Any]) -> str:
    focus = str(room.get("focus") or "")
    place = room["facts"].get("place")
    extra = "; " + "; ".join(_place_pointer(place, room["notes"])) if place else ""
    return " ".join((f"- {room['label']}; {_facts_text(room['facts'])}; last activity {_minute(room['facts'].get('latest'))}"
            f"{'; authored focus: ' + focus if focus else ''}{extra}; memory_read(room_id='{room['room_id']}', rows=true)").split())


def _range_room(room: Mapping[str, Any]) -> str:
    words = room["words"]
    if not words:
        return "\n".join(_live_room(room))
    return "\n".join([*_live_room({**room, "words": []}),
        f"{len(words)} spoken rows, {_minute(words[0]['ts'])} → {_minute(words[-1]['ts'])}; "
        f"memory_read(room_id='{room['room_id']}', rows=true, from='{words[0]['address']}', to='{words[-1]['address']}')"])


def _live_room(room: Mapping[str, Any], spoken: Any = frozenset()) -> List[str]:
    lines = [f"### {room['label']} — {_facts_text(room['facts'])}",
             f"memory_read(room_id='{room['room_id']}'{', rows=true' if room['facts']['rows'] else ''})"]
    if place := room["facts"].get("place"):
        return lines + _place_pointer(place, room["notes"])
    return lines + _note_lines(room["notes"]) + [
        _row_pointer(word, "words") if word["address"] in spoken else word["line"] for word in room["words"]]


def _pages_line(status: Mapping[str, Any]) -> str:
    """The story's standing last line: my pages (and when I last sealed one), a helper's drafts, parts; zeros say zeros."""
    latest = f" (latest {status['latest_by_me']})" if status.get("latest_by_me") else ""
    return (f"Pages sealed by me: {status.get('pages_by_me', 0)}{latest}; drafted by a helper: "
            f"{status.get('helper_pages', 0)}; parts: {status.get('parts', 0)}.")


def _status_lines(status: Mapping[str, Any], refusals: Tuple[Dict[str, Any], ...]) -> List[str]:
    """The story status while the old memory is not all folded; one line per helper refusal."""
    if not status or status.get("folded", 0) >= status.get("total", 0):
        return []
    lines = [f"Story status: the helper retelling is folded {status['folded']} of {status['total']} blocks; "
             f"{status['open_records']} retold records are still open ({status['open_rows']} rows, "
             f"{status['open_chars']} chars of retelling; helper route {status['helper_route']})."]
    for refusal in refusals:  # the retelling's own id is not the helper's answer
        read = (f"its answer: read_file(root='runtime_data', path='{refusal['path']}')" if refusal.get("path")
                else "its answer was not retained")
        lines.append(f"A helper could not fold: {refusal['label']}; {refusal['period']}; {refusal['kind']}; {read}")
    return lines


def render_story(snapshot: MemoryViewSnapshot, level: FloorLevel = FULL_VIEW) -> str:
    """Block B's tail, ``## My story``: the settled retellings, then the pages not yet folded, then the status.

    Every unfolded retold record is whole, then my parts and selected accounts, then the pages no
    part or account holds yet in publication order (``_story_pages``), so a new page joins the end;
    the records an account tells retain their combined period and count with its composition reader. Its bytes depend only on the
    chronicle, the room labels, the helper route and ``level`` (the physical floor's; the
    empty level renders the full view): F3 turns a room's retold records into one line; F5l keeps
    page first sentences, then F5 addresses pages and parts by covered-period age, independently
    of display order. Selected accounts stay whole.
    """
    spec = snapshot.spec
    if not spec.story:
        return ""
    if not snapshot.active:
        return (f"## My story — unavailable now ({snapshot.store_status.get('reason')})\n\n"
                f"The old memory files are untouched: read_file(root='runtime_data', path='{LEGACY_FILE}').")
    gone = dict(level.addressed)
    told = [entry for entry in snapshot.story if entry.get("told_by")]
    pointers = [entry for entry in snapshot.story if entry.get("kind") == "legacy" and not entry.get("told_by")]
    accounts = [entry for entry in snapshot.story if entry.get("kind") == "account" and not entry.get("told_by")]
    pages = [entry for entry in snapshot.story if entry.get("kind") not in ("legacy", "account") and not entry.get("told_by")]
    lines = ["## My story", "", _STORY_INTRO]
    if accounts:
        lines += ["", "### My account across rooms"]
        for entry in accounts:
            lines += _page_lines(entry) + _told_lines(entry['id'], told)
    if pointers:
        taken = set(gone.get("F3", ()))
        lines += ["", "### Old memory retold by a helper before the update (not lived; exact sources readable by id)"]
        lines += retold_lines(pointers, taken)
    shown = set(gone.get("F5", ()))
    aggregate = [entry for entry in pages if entry['id'] in shown] if gone.get('F5a') else []
    # A selected form changes this entry alone, without regrouping unrelated entries or changing their headers.
    for entry in pages:
        if aggregate and entry is aggregate[0]:
            lines += ['', '### My older pages and parts, by address', _story_line(aggregate)]
        if not aggregate or entry['id'] not in shown:
            lines += (["", _page_pointer(entry)] if entry["id"] in shown else
                      ["", _page_label(entry)] if entry["id"] in gone.get('F5l', ()) else _page_lines(entry))
        told_lines = _told_lines(entry["id"], told)
        if aggregate and entry['id'] in shown and told_lines:
            lines += [f"Account {entry['id']}; memory_read(node_id='{entry['id']}')"]
        lines += told_lines
    lines += ["", _pages_line(snapshot.legacy_blocks)]
    if snapshot.legacy_blocks.get("accounts"):
        lines.append(accounts_line(snapshot.legacy_blocks))
    lines += _status_lines(snapshot.legacy_blocks, snapshot.fallback_refusals)
    return "\n".join(lines)


def _note_lines(notes: List[Dict[str, Any]]) -> List[str]:
    """Authored source bytes and their stamp only; the start allowance never changes a note's rendering."""
    return [f"note {note['id']} by {note['role']} on {note['date']}\n{_indented(note['text'])}" for note in notes]


def _note_pointer(note: Mapping[str, Any]) -> str:
    """A source-derived label, never a summary or a function of the start allowance."""
    sentence = re.split(r'(?<=[.!?。！？])\s+', ' '.join(note['text'].split()), maxsplit=1)[0]
    label = sentence if len(sentence) <= 160 else sentence[:159].rstrip() + '…'
    return (f"note {note['id']} by {note['role']} on {note['date']}: {label}; "
            f"memory_read(node_id='{note['id']}')")


def _notes_line(notes: List[Dict[str, Any]]) -> str:
    dates = sorted(note['date'] for note in notes)
    roles = ', '.join(sorted({note['role'] for note in notes}))
    return (f"{len(notes)} notes not yet sealed, {dates[0]} → {dates[-1]}, by {roles}; "
            f"ids: {', '.join(note['id'] for note in notes)} (memory_read(node_id=…))")


def _room_note_lines(notes: List[Dict[str, Any]], gone: Mapping[str, Any]) -> List[str]:
    addressed = set(gone.get('F8', ()))
    aggregate = [note for note in notes if note['id'] in addressed] if gone.get('F8a') else []
    lines = []
    for note in notes:
        if aggregate and note is aggregate[0]:
            lines.append(_notes_line(aggregate))
        if note['id'] not in addressed:
            lines.extend(_note_lines([note]))
        elif not aggregate:
            lines.append(_note_pointer(note))
    return lines


_LATE_REVIEW_INTRO = ("Reviewer feedback that settled after its work ended, kept here by the host for this room "
                      "until {who}. It is advice, not a decision or a veto; releasing it says it was dealt with, "
                      "not that it was agreed with.")


def _late_reviews_text(marks: List[Dict[str, Any]], role: str) -> str:
    """This room's late reviews; a delegated helper reads who releases them (``chronicle_store``), not a call it cannot make."""
    helper = role in DELEGATED_FOCUS_ROLES
    who = ("the integrating mind releases it; as a delegated helper I report what matters in it" if helper
           else "I release it with my reason")
    lines = ["## Late review feedback (host)", "", _LATE_REVIEW_INTRO.format(who=who)]
    for mark in marks:
        late = mark["late_review"]
        if late.get("own_room") is False:
            lines.append(f"- [{mark['room']}] Late review of task {late['task_id']}: {late['source']}")
            continue
        release = "" if helper else f"; memory_mark(release_ids=['{mark['id']}'], reason=…)"
        lines += [f"- [{mark['room']}; kept since {mark['date']}] " + _indented(mark["text"]).lstrip(),
                  f"{INDENT}full source (sha256 {late['sha256'][:12]}): {late['source']}{release}"]
    return "\n".join(lines)


def _marks_text(marks: List[Dict[str, Any]]) -> str:
    lines = ["## Marks I keep in view", ""]
    for mark in marks:
        lines.append(f"- [{mark['scope']}; {mark['room']}] " + _indented(mark["text"]).lstrip()
                     + f" — marked by {mark['by']} on {mark['date']}; target {mark['target']}; "
                     + (f"meets me at {mark['anchor']}; " if mark.get("anchor") else "")
                     + f"memory_mark(release_ids=['{mark['id']}'], reason=…)")
        if mark.get("quote"):
            lines.append(f"{INDENT}quote: " + _indented(mark["quote"]).lstrip())
    return "\n".join(lines)


def _live_text(rooms: Tuple[Dict[str, Any], ...], gone: Mapping[str, List[str]], other: Tuple[Dict[str, Any], ...] = ()) -> str:
    aggregate = set(gone.get("F1b", ())) | set(gone.get("F6a", ()))
    lines = ["## Live rooms", "", "Other rooms with human words still unconsidered, my unsealed notes, running work or gaps."]
    if aggregate or other:
        lines += ["", _rooms_line([*other, *(room for room in rooms if room["room_id"] in aggregate)])]
    for room in rooms:
        ident = room["room_id"]
        if ident not in aggregate:
            lines += ["", _directory_room(room) if ident in gone.get("F6", ()) or room.get("directory") else
                      _range_room(room) if ident in gone.get("F6r", ()) else "\n".join(_live_room(room))]
    return "\n".join(lines)


def _room_text(room: Mapping[str, Any], gone: Mapping[str, List[str]]) -> str:
    """``## This room (<label>) — head <n>; <standing facts>``: the room page, the words that started it, notes, two lanes."""
    retold, mine, people, facts = (set(gone.get(step, ())) for step in ("F4", "F2", "F7", "F1"))
    people |= set(gone.get("F7p", ()))
    lines = [f"## This room ({room['label']}) — head {room['head']}; {_facts_text(room['facts'])}"]
    if room["legacy"]:
        lines += ["", "### Retold before the update (helper retelling, not lived)"]
        lines += [_retold(item, item["id"] in retold) for item in room["legacy"]]
    lines += (["", "### Pages under my parts" if all(i["part"] for i in room["under_parts"]) else "### Pages of this room",
               *(_retold(item, item["id"] in retold or item.get("standing", False)) for item in room["under_parts"])]
              if room["under_parts"] else [])
    if room["origins"]:
        lines += ["", "### Words that started this work (retention-proof)"]
        lines += [f"[{item['ts'] or 'time not recorded'}; owner; {item['ref']}] " + _indented(item["text"]).lstrip()
                  for item in room["origins"]]
    if room["notes"]:
        lines += ["", "### My notes not yet sealed"] + _room_note_lines(room["notes"], gone)
    if room["lane1"]:
        lines += ["", f"### Open conversation since {room['since']} (verbatim: people and my replies)"]
        addressed = mine | people
        aggregate = [item for item in room['lane1'] if item['address'] in addressed
                     and not item.get('immune')] if gone.get('F7a') else []
        grouped = {item['address'] for item in aggregate}
        for item in room['lane1']:
            if aggregate and item is aggregate[0]:
                lines.append(_rows_line(room, aggregate))
            if item['address'] not in grouped:
                lines.append(item['line'] if item.get('immune') else _row_pointer(item, 'my reply')
                             if item['address'] in mine else _row_pointer(item, 'words')
                             if item['address'] in people else item['line'])
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
    late = [mark for mark in snapshot.marks if mark.get("late_review")]
    if late:
        parts.append(_late_reviews_text(late, snapshot.spec.role))
    if len(late) < len(snapshot.marks):
        parts.append(_marks_text([mark for mark in snapshot.marks if not mark.get("late_review")]))
    if snapshot.spec.live_rooms != "none" and (snapshot.live_rooms or snapshot.other_rooms):
        parts.append(_live_text(snapshot.live_rooms, gone, snapshot.other_rooms))
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
    (loaded if spec.knowledge else missing).append("knowledge (overview, index, project journal and workpad" + (
        ")" if spec.knowledge else "; the index arrives with your first delegation)"))  # delegation_knowledge_index
    if spec.owner_words and (snapshot is None or snapshot.owner_words):  # named only when the block was drawn
        loaded.append("the words of my human that caused this work")
    missing += ["the global scratchpad", "earlier task reports"]
    return ("## Working sources\n\n"
            f"Loaded above: {', '.join(loaded)}; your own recent process is loaded below. "
            f"Not loaded: {', '.join(missing)}; memory_read, chat_history, knowledge_list, knowledge_read and "
            "get_task_result reach them, as they reach your parent.\n" + CHILD_ROLE_TEXT)
