"""The resident memory view of the acting mind: one render for every role.

The view replaces the old dialogue history and chat tail in a request. Block B carries my
story (``## My story``); block C the live part (this room's ``## Late review feedback (host)``,
``## Marks I keep in view``, ``## This room (<label>) — head <n>`` and ``## Other conversations``,
``memory_view_conversations``). Which parts a role sees is one ``ViewSpec``: ``ROLE_DEFAULTS`` by role (Main,
a root and a Project task are the integrator; consciousness; Presence; a delegated child; a nanny), or an
explicit spec a task carries under ``memory_view``; every focus that integrates holds the same parts, only its
room differs, and a delegated child or nanny carries its assignment instead of my other conversations. The
current room is the task's own room (``dialogue_evidence.own_room_chat``); outside a Project and Main it is Main
(a wake's too: it reads Main as the Main turn does), except a Presence turn, which sees its own conversation.

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

from ouroboros.text_sentences import first_sentence

import dataclasses
import datetime
import functools
import json
import logging
import pathlib
from collections import Counter, OrderedDict
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros import chat_chain, memory_inventory
from ouroboros.chronicle_import import LEGACY_ROOM_ID, LEGACY_ROOM_LABEL, legacy_frontier, row_lineage
from ouroboros.chronicle_store import (CHILD_DRAFT_RIGHT, DELEGATED_FOCUS_ROLES, ChronicleStore, capture, draft_signer,
                                       is_late_review_mark, source_time_span)
from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
from ouroboros.dialogue_provenance import (RoomLabelResolver, is_presence_task, render_memory_row, render_row_text, stamp_summary,
                                           row_class, warm_lineage)
from ouroboros.mark_anchors import describe as describe_anchor
from ouroboros.memory_view_account import (account_entries, account_lines, accounts_line, fix_map, quote_line as quote_line,
                                         page_quote_lines, story_quote_rows as story_quote_rows, told_here, told_map)
from ouroboros.memory_view_legacy import INDENT, indented as _indented, retold_lines
from ouroboros.context_timing import counted, timed_section
from ouroboros.utils import append_jsonl, sha256_text, utc_now_iso

log = logging.getLogger(__name__)

MAIN_ROOM = str(WEB_UI_CHAT_ID)
ROLES = ("integrator", "consciousness", "presence", "child", "nanny")
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
CHILD_ROLE_TEXT = "Read memory or sources whenever they could change what you do, and name what you read in your answer. " + CHILD_DRAFT_RIGHT


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
    """``label(room, sample=None)``: one registry snapshot per capture (``RoomLabelResolver``); ``label.warm(rooms)``
    reads at once what names the rooms a reader is about to label."""
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

    label.warm = resolver.warm  # type: ignore[attr-defined]
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
    origin_words: bool = True  # the owner's words that started a Project, when its rows are not open (a helper's set carries them)
    other_conversations: bool = True  # ``## Other conversations``: every other conversation of mine
    marks: str = "all"  # all | room_and_global | none
    knowledge: bool = True  # overview, knowledge index, project journal (rendered by the context)
    work_room_levels: int = 1
    owner_words: bool = False  # the owner's words that caused a helper's work (owner_words.py)


ROLE_DEFAULTS: Mapping[str, ViewSpec] = MappingProxyType({
    # Every focus that integrates holds the whole (BIBLE P1 "The subject of continuity"): one row, its room aside.
    **{role: ViewSpec(role) for role in ("integrator", "consciousness", "presence")},
    "child": ViewSpec("child", room_lanes=False, origin_words=False, other_conversations=False, marks="room_and_global",
                      knowledge=False, owner_words=True),
    "nanny": ViewSpec("nanny", story=False, room_lanes=False, origin_words=False, other_conversations=False,
                      marks="room_and_global", knowledge=False, owner_words=True),
})
_FIELDS = {field.name: field.type for field in dataclasses.fields(ViewSpec)}


def view_role(task: Mapping[str, Any]) -> str:
    """The view role, first match wins, in ``knowledge.focus_signature``'s order: a nanny is the one
    dispatch fact (``_nanny_route_dispatched_for``), then a delegated child, an explicit focus cause,
    a wake's ledger category, and legacy Presence; everything else integrates. The facts are read at
    the task's top level or under ``metadata``; a child's role also under ``task_contract.lineage``,
    as the declared validator reads it (``context_input_selection._validate_declared_input_task``).
    """
    from ouroboros.consciousness_authority import CONSCIOUSNESS_CATEGORY, WAKE_CAUSE_FOCUS, wake_cause
    from ouroboros.subagent_dispatch_notes import _nanny_route_dispatched_for  # D03->D07 is lazy-only

    meta = dict(_mapping(task.get("metadata")))
    if _nanny_route_dispatched_for(dict(task), None) or _nanny_route_dispatched_for(meta, None):
        return "nanny"
    lineage = _mapping(_mapping(task.get("task_contract") or meta.get("task_contract")).get("lineage"))
    if str(task.get("delegation_role") or meta.get("delegation_role") or lineage.get("delegation_role") or "").strip().lower() == "subagent":
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
    """The task's own room (``own_room_chat``): the internal Project it lives in, then a Presence
    focus's own conversation, then any other own room.

    A result outside the Projects and Main, or none (the hidden partition, another
    transport chat, a task without a chat), is Main, as the old focused view was; a
    Presence turn keeps its own conversation. Work a focus started carries that focus by
    value, and its Project is still its room: the stored focus never outranks it.
    """
    from ouroboros.dialogue_evidence import own_room_chat  # D03->D06 is lazy-only

    try:
        own = own_room_chat(ctx if ctx is not None else _task_ctx(task), drive_root)
    except (TypeError, ValueError, OSError):
        own = None
    projects = set(project_chat_ids or ())
    if own is not None and int(own) in projects:
        return str(own)
    focus = _mapping(_mapping(task.get("metadata")).get("focus") or task.get("focus"))
    if presence and focus.get("room_chat_id") is not None:
        return str(focus["room_chat_id"])
    if own is not None and (presence or str(own) == MAIN_ROOM):
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
    conversations: Tuple[Dict[str, Any], ...] = ()  # one group per other conversation (``memory_view_conversations``)
    marks: Tuple[Dict[str, Any], ...] = ()
    legacy_blocks: Dict[str, Any] = dataclasses.field(default_factory=dict)  # the story status facts
    fallback_refusals: Tuple[Dict[str, Any], ...] = ()
    owner_words: str = ""  # the rendered block of the owner's words that caused a helper's work
    waiting: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # The newest chronicle record this capture read (its marks among them); later mark changes reach a running
    # focus's tail (``mark_anchors.append_mark_changes``). None: the view read no chronicle. It rides beside the view
    # (``context_fit.ContextCore.memory_view_sequence``), never in ``snapshot_json``: a record no section shows moves
    # no hash.
    chronicle_sequence: Optional[int] = dataclasses.field(default=None, compare=False)

    @property
    def active(self) -> bool:
        return self.store_status.get("state") == "active"

    def piece(self, key: Any, make: Callable[[], str]) -> int:
        """The characters one local form of this view adds to a request's JSON, made once for this snapshot by building
        its text (``view_pieces``); a whole text's price is their sum (``room_chars``). Only the count stays, as the
        floor prices far more forms than it delivers: delivery builds its texts again (``render_story`` and
        ``render_room``). A non-field attribute like the floor's element table: ``snapshot_json``, equality and
        ``dataclasses.replace`` never see it, and it lives as long as the snapshot object."""
        pieces = self.__dict__.setdefault("_pieces", {})
        if key not in pieces:
            counted("view_pieces", key=(id(self), key))
            pieces[key] = json_chars(make())
        return pieces[key]


_TUPLES = ("story", "conversations", "marks", "fallback_refusals")


def json_chars(text: str) -> int:
    """What a text adds to a request's JSON: its escaped characters, quotes excluded."""
    return len(json.encoder.encode_basestring(text)) - 2


def snapshot_json(snapshot: MemoryViewSnapshot) -> str:
    """Canonical JSON of captured plain data, excluding render caches and the capture's journal boundary."""
    def fields(value: Any) -> Dict[str, Any]:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)
                    if not (isinstance(value, MemoryViewSnapshot) and field.name == "chronicle_sequence")}
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return json.dumps(snapshot, default=fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _owner_words_block(task: Mapping[str, Any], drive_root: Any = None, marks: Any = None) -> str:
    """A helper's birth set (``owner_relay.task_words_text``): its field, then its room's rows read once here; the
    files the rows carry are staged on the helper's own drive and named by the path it opens them with. ``marks``:
    the acting marks the view's capture holds, so the marked rows are cut from them and the chronicle is not
    entered again; None (a declared start outside a capture) lets the set read them itself."""
    from ouroboros.owner_relay import task_words_text  # D03->D01 is lazy-only

    return task_words_text(drive_root, task, indent=INDENT, stage=True, marks=marks)


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


def _fixes(store: ChronicleStore, record: Mapping[str, Any], fixes: Mapping[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """The mind's corrections and rejections aimed at a part's members, nested parts' too, one per record."""
    members = store.folded_members(record["id"]) if record.get("kind") == "part" else []
    return [{"kind": fix["kind"], "target": str(member), "text": str(fix.get("text") or fix.get("reason") or "")}
            for member in members for fix in fixes.get(str(member), ())]


def _record_words(record: Mapping[str, Any]) -> Dict[str, Any]:
    """A page's, part's or note's ``text`` as it acts: its own words, then each signed correction under them
    (``chronicle_store._interpret``). A corrected record also keeps ``own_chars``, the length of its own words in it,
    so a shorter form of those words keeps the corrections beside them (``_own_words``)."""
    text = str(record.get("current_text") or "")
    return {"text": text, **({"own_chars": len(str(record.get("text") or ""))} if record.get("corrections") else {})}


def _story_pages(store: ChronicleStore, label: Callable[..., str], units: Mapping[str, memory_inventory.LegacyUnit],
                 room_id: Optional[str] = None, terminal: Optional[Callable[..., Any]] = None) -> List[Dict[str, Any]]:
    """Every acting page and part not folded into a part (``room_id``'s alone, when named), and every account my
    selection shows, all rooms; the records a shown account tells in its place are marked ``told_by``.

    The settled retellings come first — parts and accounts, by the first row of their own room then sequence —
    and then the pages no part or account holds yet, in publication order (sequence). A newly sealed page so
    joins the end and preserves the prefix on routes with prefix caching. With one cache marker for the whole
    block B, a new page misses that block (``context_fit._system_blocks``). A fold rewrites the story from its
    first changed retelling, as before. ``terminal`` groups the writers' later terminal rows (``capture_story``'s
    records what it read)."""
    fixes = fix_map(store)
    records = [record for record in store.room_records(room_id, kinds=("page", "part"))
               if not record.get("folded_into")]
    getattr(label, "warm", lambda _rooms: None)({str(record["room_id"]) for record in records
                                                if not _mapping(record.get("metadata")).get("room_label")})
    from ouroboros.memory_sealing import capture_terminal_rows, terminal_links
    terminal_cache = terminal(records) if terminal else capture_terminal_rows(store.data_root, records)
    keyed = [((entry.pop("first"), entry["sequence"]), entry) for entry in account_entries(
        store, label, units, fixes, lambda period: _period(period.span, period.note()))]
    for record in records:
        room = str(record["room_id"])
        period = memory_inventory.record_period(store, record, units)
        keyed.append(((period.first, record["sequence"]), {
            "kind": record["kind"], "id": record["id"], "room_id": room,
            "label": str(_mapping(record.get("metadata")).get("room_label") or label(room)),
            **({"lead_rule": record["metadata"]["lead_rule"]} if "lead_rule" in _mapping(record.get("metadata")) else {}),
            "source_period": period.span, "period_basis": period.source,
            "period": _period(period.span, period.note()), **_record_words(record),
            "status": str(record.get("status") or ""), "signer": draft_signer(record.get("author")),
            "stamp": stamp_summary(record.get("host_stamp")), "fixes": _fixes(store, record, fixes),
            "terminal_links": terminal_links(store.data_root, record, terminal_cache), "quotes": record.get("quotes") or []}))
    told = told_map([entry for _key, entry in keyed if entry["kind"] == "account"])
    ordered = sorted(keyed, key=lambda pair: (1, pair[0][1]) if pair[1]["kind"] == "page" else (0, *pair[0]))
    # The floor spends the oldest covered period first, independently of publication order in B.
    return [{**entry, "floor_order": list(key), **({"told_by": told[entry["id"]]} if entry["id"] in told else {})}
            for key, entry in ordered]


class _StorySources:
    """What one story capture read beside the chronicle and its legacy units: the room labels it named, its page
    writers' later terminal rows and, while the old memory is not all folded, the helper route."""

    def __init__(self, root: pathlib.Path, label: Callable[..., str]) -> None:
        # ``route``: None until asked, then its answer in a 1-tuple (an answer may itself be None)
        self.root, self.labels, self.starts, self.rows, self.route = root, {}, {}, {}, None

        def named(room: Any) -> str:
            self.labels[room] = label(room)
            return self.labels[room]
        named.warm = getattr(label, "warm", lambda _rooms: None)  # type: ignore[attr-defined]
        self.label = named

    def terminal(self, records: List[Dict[str, Any]]) -> Dict[Any, Any]:
        from ouroboros.memory_sealing import terminal_starts, writer_terminal_rows

        self.starts = terminal_starts(records)
        self.rows = writer_terminal_rows(self.root, self.starts)
        return self.rows

    def helper_route(self) -> str:
        from ouroboros.model_slots import get_light_model

        self.route = (get_light_model(),)
        return self.route[0]

    def unchanged(self, label: Callable[..., str]) -> bool:
        """Whether each source, asked again (of the caller's labeler), answers as it did."""
        from ouroboros.memory_sealing import writer_terminal_rows
        from ouroboros.model_slots import get_light_model

        getattr(label, "warm", lambda _rooms: None)(self.labels)
        return (all(label(room) == text for room, text in self.labels.items())
                and writer_terminal_rows(self.root, self.starts) == self.rows
                and (self.route is None or self.route == (get_light_model(),)))


# The last story of each chronicle index with what it read: (journal size, legacy units, sources, story); a few
# chronicles at most, a memory bound (a process reads one data root; a test process many).
_STORIES: "OrderedDict[str, Tuple[int, List[memory_inventory.LegacyUnit], _StorySources, Tuple[Any, ...]]]" = OrderedDict()
_STORIES_KEPT = 4


@timed_section("memory_view.story")
def capture_story(store: ChronicleStore, root: pathlib.Path, label: Callable[..., str],
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """``(story entries, story status, helper refusals)``; folded or not is ``memory_inventory``'s verdict.

    Every unfolded retold record keeps its current words, whatever its block: whole in the story until the
    floor turns its room into one line (F3) or a selected account tells it in its place (``told_by``).

    The story is a function of its sources alone (this module's docstring): a capture of the stored chronicle at
    the journal size of the last one, with equal legacy units, and whose room labels, page writers' later terminal
    rows and helper route answer as they did, is that capture byte for byte, so it is reused in this process; any
    other answer captures the story whole. A proposal's capture reads an unsaved selection: it neither reuses nor
    keeps one. Its entries are shared, and no reader mutates them (as a parsed view's).
    """
    with capture(root, tail="terminal"):  # one state for the identity and every read; inside a capture, that one
        size, units = store.stored_state(), memory_inventory.legacy_units(store, root)
        held = _STORIES.get(str(store.index_path)) if size is not None else None
        if held is None or held[:2] != (size, units) or not held[2].unchanged(label):
            sources = _StorySources(root, label)
            held = (size, units, sources, _capture_story(store, sources, units))
            if size is not None:
                _STORIES.pop(str(store.index_path), None)  # the newest last
                _STORIES[str(store.index_path)] = held
                if len(_STORIES) > _STORIES_KEPT:
                    _STORIES.popitem(last=False)
    story, status, refusals = held[3]
    return list(story), dict(status), list(refusals)


def _capture_story(store: ChronicleStore, sources: _StorySources, units: List[memory_inventory.LegacyUnit],
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    label = sources.label
    pointers = {pointer["node_id"]: pointer for pointer in memory_inventory.legacy_pointers(store)}
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
    pages = _story_pages(store, label, {unit.record_id: unit for unit in units}, terminal=sources.terminal)
    told = told_map([entry for entry in pages if entry["kind"] == "account"])  # a retold record an account tells
    story = [{**entry, "told_by": told[entry["id"]]} if entry["id"] in told else entry for entry in story]
    progress = memory_inventory.legacy_progress(units)
    open_units = [unit for unit in units if not unit.folded]
    status = {"folded": progress["folded"], "total": progress["periods"], **memory_inventory.story_counts(store),
              "open_records": len(open_units), "open_rows": sum(unit.rows for unit in open_units),
              "open_chars": sum(unit.retelling_chars for unit in open_units), "helper_route": "",
              "accounts": len(store.records(kinds=("account",))),  # in the common view: whole there, not told by a later account
              "accounts_shown": sum(1 for entry in pages if entry["kind"] == "account" and not entry.get("told_by"))}
    if status["folded"] < status["total"]:
        status["helper_route"] = sources.helper_route()
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
    """A verbatim lane-1 line with the facts its address line needs (head, size, position); my human's side is
    ``owner_speech``'s."""
    from ouroboros.dialogue_provenance import owner_speech
    from ouroboros.memory_view_work import speech_facts

    row = texts.get(entry[0]["row_sha256"])
    return {**speech_facts(row or entry[1], author), "owner_side": owner_speech(entry[1], author),
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
    elif not name and (event := _mapping(_mapping(meta.get("quiz")).get("secret_outcome")).get("event")):
        name = f"secret request {event}"  # the form's host outcome (``memory_inventory.row_meta``), never a value
    return name


@timed_section("memory_view.working_notes")
def _working_notes(root: pathlib.Path, tasks: List[str]) -> Dict[str, str]:
    """Per root task whose result names working notes (``main_context_authority.WORKING_NOTES_FIELD``: what its latest
    applied fold kept), the fact and exact reader from that field alone. One batch reads the results; a result
    without a notes address makes no entry. The notes' words and availability are read only by choice."""
    from ouroboros.main_context_authority import WORKING_NOTES_FIELD, retained_reader
    from ouroboros.task_results import load_task_results

    facts: Dict[str, str] = {}
    for task, saved in load_task_results(root, tasks).items():
        ref = saved.get(WORKING_NOTES_FIELD) if isinstance(saved, Mapping) else None
        if not isinstance(ref, Mapping) or not ref.get("path"):
            continue
        facts[task] = f"working notes kept: {source_reader({}, retained_reader(root, task, ref))}"
    return facts


def _task_line(root_id: str, group: List[Tuple[Entry, Dict[str, Any]]], texts: Mapping[str, Dict[str, Any]],
               summaries: Mapping[str, Any], notes: str = "") -> Dict[str, str]:
    """One line for a task's lane-2 rows: the host's status, never a child's report or a retelling; ``notes``: the
    working notes its result names (``_working_notes``)."""
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
              if (name := _typed_fact(entry[1]))] + ([notes] if notes else [])
    head = f"[{first[1].get('ts') or 'time not recorded'}; host; task {root_id}] "
    without = head + "; ".join(parts)
    if gist := summaries.get(root_id):
        parts[0] += f" — summary: «{gist['summary']}»"
    return {"line": head + "; ".join(parts), "line_without_gist": without}


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


def _lanes(root: pathlib.Path, entries: List[Entry], lineage: Mapping[str, Any],
           summaries: Optional[Mapping[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Lane 1 verbatim (people and my words to them) and lane 2 as one line per root task; ``summaries``: the
    capture's gists by work (``memory_view_work.gists``), none when a caller gives none."""
    summaries = summaries or {}
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
    notes = _working_notes(root, list(groups))
    lane1 = [_spoken(entry, cls["author"], texts) for entry, cls in classed if cls["lane"] == 1]
    lane2 = []
    for pos, task, loose in order:
        rows = groups[task] if task else [loose]
        lines = _task_line(task, rows, texts, summaries, notes.get(task, "")) if task else {"line": _loose_line(loose[0], loose[1], texts)}
        lane2.append({**lines, "task": task, "ts": rows[0][0][1].get("ts"), "pos": pos,
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
        kept = [{**({"lead_rule": record["metadata"]["lead_rule"]} if "lead_rule" in _mapping(record.get("metadata")) else {}),
                 "id": record["id"], "date": _minute(record.get("ts")).split(" ")[0], "ts": str(record.get("ts") or ""),
                 "author_kind": str(_mapping(record.get("author")).get("kind") or ""),
                 "task_id": str(record.get("author", {}).get("task_id") or ""),
                 "role": str(_mapping(_mapping(record.get("author")).get("focus")).get("role") or "mind"),
                 "origin": dict(_mapping(_mapping(record.get("author")).get("focus"))), **_record_words(record)}
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
                  gaps: Optional[Mapping[str, set]] = None, *, summaries: Mapping[str, Any]) -> Dict[str, Any]:
    """The current room: its head and standing facts, page (retold records, pages under parts, notes), origin words, lanes.

    Retold records already whole in my story are not repeated. A selected account carries its records, a part
    it tells its members too; their detail stays at their source readers rather than reappearing in this room,
    where ``told`` names how many of the room's records accounts in my story tell, and which ones (``told_here``),
    records folded under a part among them.
    A standing part (the mind's, or a helper's draft not pending as a candidate) stands for its members here
    as in the story: they keep their period and address (``standing``), a viewer with a story only.
    """
    room = str(spec.room_id)
    records = store.room_records(room)
    whole = {e["id"] for e in story if e.get("kind") == "legacy" and e.get("text") and not e.get("told_by")}
    teller = told_map([e for e in story if e.get("kind") == "account"])  # the latest shown selections
    told = set(teller)
    told |= {member for record in records if record["kind"] == "part" and record["id"] in told
             for member in store.folded_members(record["id"])}
    standing = {record["id"] for record in records if record["kind"] == "part"}  # room_records: no candidate or rejected
    sample = next((meta for _address, meta, _pos in reversed(entries)
                   if meta.get("transport") or meta.get("presence_provenance")), None)
    facts: Dict[str, Any] = {"room_id": room, "label": label(room, sample), "head": store.room_head(room),
                             "facts": _with_gaps(memory_inventory.room_facts(store, room, entries, lineage,
                                                                             len(notes.get(room, ()))), gaps, room),
                             "legacy": [], "under_parts": [], "notes": [], "origins": [], "since": "",
                             "lane1": [], "lane2": [], "told": [], "project_words": None, "newest": "", "sealed_by": []}
    if entries and spec.role not in DELEGATED_FOCUS_ROLES:  # S1: what a page in my finishing response covers here
        facts["newest"] = chat_chain.format_address(max(entries, key=lambda entry: entry[2])[0])
    facts["sealed_by"] = [  # S3: a conversation page another room holds seals rows here too
        {"id": record["id"], "label": str(_mapping(record.get("metadata")).get("room_label") or label(record["room_id"]))}
        for record in store.room_records(None, kinds=("page",))
        if str(record["room_id"]) != room and room in _mapping(_mapping(record.get("covers")).get("rooms"))]
    if spec.room_page:
        units = {unit.record_id: unit for unit in memory_inventory.legacy_units(store, root)}
        retold = [record for record in records if record["kind"] in ("legacy", "gap")]
        pointer_records = records
        if room == MAIN_ROOM and spec.role != "nanny":  # the room-less retellings (the flat summary, a mixed
            # era) predate rooms and were Main's memory: whole and first for every focus that starts with the top
            # level of the life account (a child included); a nanny carries no life account and no story, so it
            # gets neither the text nor a pointer
            roomless = store.room_records(LEGACY_ROOM_ID)
            pointer_records = [*records, *roomless]
            retold[:0] = [record for record in roomless if record["kind"] == "legacy"
                          and _mapping(record.get("metadata")).get("legacy_type") not in ("gap", "cursor_gap")]
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
        main = room == MAIN_ROOM and spec.role != "nanny"  # Main's page carries the room-less retellings too
        # A folded record is no story entry, so it joins ``told_here`` from the room's own records.
        folded = [{"id": record["id"], "room_id": str(record["room_id"]), "told_by": teller[record["id"]],
                   "source_period": memory_inventory.record_period(store, record, units).span}
                  for record in pointer_records if record.get("folded_into")
                  and record["id"] in teller]
        facts["told"] = told_here([*story, *folded], {room, *((LEGACY_ROOM_ID,) if main else ())})
    sealed = spec.room_lanes and spec.role == "integrator"  # the root's own view of its project's sealed rows
    project = (spec.origin_words or sealed) and room.lstrip("-").isdigit() and int(room) in (
        memory_inventory.membership_facts(root).project_chat_ids)
    if spec.origin_words and project:
        facts["origins"] = _origins(root, room, entries if spec.room_lanes else [])
    if spec.room_lanes and entries:
        facts["since"] = _minute(facts["facts"]["earliest"])  # the rows' first stored time, not the first row read
        facts["lane1"], facts["lane2"] = _lanes(root, entries, lineage, summaries)
    if sealed and project:  # never Main: my human's project rows that left the open conversation, whole and dated
        from ouroboros.owner_words import project_words

        facts["project_words"] = project_words(root, int(room), open_rows=[*facts["lane1"], *facts["origins"]], indent=INDENT)
    return facts


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


def _late_review_entry(mark: Mapping[str, Any], label: Callable[..., str]) -> Dict[str, Any]:
    """A host-kept late review of this room (``late_review_marks``): its words, when it settled, its full source and
    its release. Other rooms' late reviews are no entries: the waiting facts count them (``memory_waiting.measure``)
    and ``memory_read(room_id='late_reviews')`` lists them by room and work."""
    review, target = _mapping(mark.get("review")), _mapping(mark.get("target_ref"))
    return {"id": mark["id"], "scope": "room", "room": label(mark.get("room_id")),
            "text": str(mark.get("text") or ""),
            "by": "host", "date": _minute(mark.get("ts")).split(" ")[0], "target": _mark_target(target), "quote": "",
            "late_review": {"task_id": str(review.get("task_id") or ""),
                            "source": source_reader(target, review.get("source_read")),
                            "sha256": str(target.get("sha256") or "")}}


def mark_in_view(mark: Mapping[str, Any], spec: ViewSpec, *, running: bool = False) -> bool:
    """Whether a role keeps a mark in view by its scope: none, every room's, or this room's and the global ones.
    A focus already running meets later changes of its room's (``spec.room_id``: the room it is in now) and the
    global ones only, whatever its start view holds (``mark_anchors.append_mark_changes``)."""
    if spec.marks == "none":
        return False
    return (spec.marks == "all" and not running) or mark.get("scope") == "global" or str(mark.get("room_id")) == spec.room_id


def mark_entry(mark: Mapping[str, Any], label: Callable[..., str]) -> Dict[str, Any]:
    """A mark as every view names it: its words, who marked it and when, its target, its quote while visibility
    is full, and the action it is tied to."""
    author = _mapping(mark.get("author"))
    by = _mapping(author.get("focus")).get("role") or (
        "the old dialogue writer" if author.get("kind") == "legacy_helper" else author.get("kind") or "not recorded")
    return {"id": mark["id"], "scope": str(mark.get("scope") or "room"),
            "room": label(mark.get("room_id")), "text": str(mark.get("text") or ""), "by": str(by),
            "date": _minute(mark.get("ts")).split(" ")[0], "target": _mark_target(mark.get("target_ref")),
            "quote": str(mark.get("quote") or "") if mark.get("visibility", "full") == "full" else "",
            "anchor": describe_anchor(mark["anchor"]) if isinstance(mark.get("anchor"), dict) else ""}


def _capture_marks(store: ChronicleStore, spec: ViewSpec, label: Callable[..., str], marks=None) -> List[Dict[str, Any]]:
    """The acting marks a role keeps in view (``mark_in_view``).

    Late review feedback the host keeps in view belongs to the room of its work whatever the role's
    mark scope: that room holds its words, every role alike. Other rooms' late reviews are counted in the
    waiting facts and listed by room and work on request, never enumerated here: a wake once drew one
    pointer per foreign mark, hundreds of addresses on every start, beside the waiting line that already
    counted them (one source delivered twice; BIBLE P1).
    """
    if spec.marks == "none":
        return []
    marks = store.active_marks(None) if marks is None else marks
    entries = []
    for mark in (mark for mark in marks if mark_in_view(mark, spec)):
        if is_late_review_mark(mark):
            if str(mark.get("room_id")) == spec.room_id:
                entries.append(_late_review_entry(mark, label))
            continue
        entries.append(mark_entry(mark, label))
    return sorted(entries, key=lambda entry: (entry["scope"] == "global", "" if entry["scope"] == "global" else entry["room"]))


def _capture_live(store: ChronicleStore, root: pathlib.Path, spec: ViewSpec, label: Callable[..., str],
                  story: List[Dict[str, Any]] = ()) -> tuple:
    """``(this room, my other conversations, marks, waiting, acting)`` from the same captured rows, notes and active
    marks; ``acting`` is every room's acting marks as read here (None when this view reads none)."""
    from ouroboros import memory_view_conversations as conversations

    lineage = row_lineage(root)
    integrating = spec.role not in DELEGATED_FOCUS_ROLES
    notes = _notes_by_room(store) if integrating or spec.room_id is not None or spec.other_conversations else {}
    gaps: Dict[str, set] = {}  # by room, what its open rows may lack (``room_appearances.GAP_DETAILS``)
    quiet: Dict[str, List[Entry]] = {}
    sealed: Dict[str, List[Entry]] = {}
    if spec.other_conversations:
        by_room = memory_inventory.open_rows_by_room(root, gaps=gaps, quiet=quiet, sealed=sealed)
    elif spec.room_id is not None:  # the room's facts are counted from its open rows whatever the view prints of them
        by_room = {spec.room_id: memory_inventory.open_room_rows(root, spec.room_id, gaps=gaps.setdefault(spec.room_id, set()))}
    else:
        by_room = {}
    # The authors of rows written before the lineage epoch, read at once for every room (not one task at a time).
    warm_lineage(lineage, ((meta, pos) for entries in by_room.values() for _address, meta, pos in entries))
    warm_lineage(lineage, ((meta, pos) for entries in quiet.values() for _address, meta, pos in entries))
    from ouroboros.memory_view_work import bind, gists
    summaries = gists(by_room, chat_chain.format_address, quiet, sealed)
    room = (_capture_room(store, root, spec, label, by_room.get(spec.room_id, []), notes, lineage, story, gaps, summaries=summaries)
            if spec.room_id is not None else None)
    marks = store.active_marks(None) if spec.marks != "none" or integrating else None
    waiting, projects = {}, None
    if integrating or spec.other_conversations:
        from ouroboros.projects_registry import list_projects
        from ouroboros.store import StoreError

        try:
            projects = list_projects(root, strict=True, oldest_first=True)
        except (OSError, ValueError, StoreError) as exc:
            # An unavailable project census does not make readable room history unavailable.
            log.warning("memory view: project waiting facts unavailable (%s)", type(exc).__name__)
            projects = None
    groups = (conversations.capture(store, root, spec, label, by_room, notes, lineage, quiet=quiet, sealed=sealed,
                                    gaps=gaps, projects=projects or [], summaries=summaries) if spec.other_conversations else [])
    bind(room, groups, summaries)
    if integrating:
        from ouroboros.memory_waiting import measure

        waiting = measure(marks, notes, [p for p in projects if "closed" not in p] if projects is not None else None,
                          groups)
    return room, groups, _capture_marks(store, spec, label, marks), waiting, marks


def capture_memory_view(drive_root: Any, task: Mapping[str, Any], spec: ViewSpec) -> MemoryViewSnapshot:
    """The facts of one request's view, read once; the chronicle is activated exactly once here.

    Before the import has completed (another importer holds the legacy lock, the
    import was refused, the journal cannot be opened) the snapshot carries only the
    reason and the room id: no story, lanes or other conversations, and no chain read; the owner-words
    block still reads the bound room (``owner_relay.task_words_text``). On an active chronicle that
    block is drawn after the capture, over the acting marks the capture read: a helper's birth enters
    the chronicle no more often than its view does.
    """
    root = pathlib.Path(drive_root)

    def words(marks: Any = None) -> str:
        return _owner_words_block(task, root, marks=marks) if spec.owner_words else ""

    status = _activation(ChronicleStore(root))
    label = room_labeler(root)
    bare = {"room_id": spec.room_id, "label": label(spec.room_id)} if spec.room_id is not None else None
    if status["state"] != "active":
        return MemoryViewSnapshot(spec=spec, store_status=status, frontier={}, room=bare, owner_words=words())
    owner_words: Optional[str] = None  # drawn once the capture holds its marks; a journal lost before reads the store
    try:
        store = ChronicleStore(root)
        # One state of the chronicle for every reader (activation may publish: before), with the stored tail facts
        # the view reads: every row for my other conversations' open rows, else the story's terminal rows.
        tail = ("all" if spec.other_conversations else
                "terminal" if spec.story or spec.room_id is not None and spec.room_page else "none")
        with capture(root, tail=tail) as derived:
            frontier, sequence = legacy_frontier(store), store.newest_sequence()  # the marks below read this state
            story, story_status, refusals = capture_story(store, root, label) if spec.story else ([], {}, [])
            room, groups, marks, waiting, acting = _capture_live(store, root, spec, label, story)
        owner_words = words(acting)
        if room:
            from ouroboros.owner_words import task_governing_words

            # Governing words bind by identity: their ref, or (without one) the row's own exact text.
            governing, _absence = task_governing_words(task)
            ids = {g.get("ref") or sha256_text(g.get("text") or "") for g in governing if g.get("ref") or g.get("text")}
            last_reply = max((item["pos"] for item in room["lane1"] if item["kind"] == "ouroboros" and item.get("via") != "own_account"), default=-1)
            targets = {item.get("reply_to") for item in room["lane1"]
                       if item.get("owner_side") and item["pos"] > last_reply}
            for item in room["lane1"]:
                item["protected_reply"] = item["kind"] == "ouroboros" and (
                    item["pos"] == last_reply or item["address"] in targets)
                item["immune"] = item.get("owner_side", False) and (
                    item["pos"] > last_reply or bool({item.get("ref"), item.get("text_sha256")} & ids))
                # Open rows have no page coverage. A reply alone is not proof of carried commitments.
                item["carried"] = False
    except Exception as exc:  # a journal that turns unreadable mid-capture still leaves a view with its reason
        log.warning("memory view: the chronicle could not be read", exc_info=True)
        return MemoryViewSnapshot(spec=spec, store_status={"state": "journal_unreadable",
                                                           "reason": f"{type(exc).__name__}: {exc}"},
                                  frontier={}, room=bare, owner_words=words() if owner_words is None else owner_words)
    if any(not read.complete for read in ((derived or {}).get(key) for key in ("tail", "terminal_tail")) if read):
        status = {**status, "tail_facts": "filling"}  # its rows were read live: the stored tail is not complete
    return MemoryViewSnapshot(spec=spec, store_status=status,
                              frontier={"status": frontier.get("status"), "pos": frontier.get("pos")},
                              story=tuple(story), room=room, conversations=tuple(groups), marks=tuple(marks), waiting=waiting,
                              legacy_blocks=story_status, fallback_refusals=tuple(refusals), owner_words=owner_words,
                              chronicle_sequence=sequence)


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


def _own_words(entry: Mapping[str, Any]) -> Tuple[str, str]:
    """``(own words, signed corrections)`` of a captured page, part or note (``_record_words``). Every form that prints
    its own words, whole or shorter, keeps the corrections beside them; a view saved before the split keeps none apart."""
    text, own = str(entry.get("text") or ""), entry.get("own_chars")
    return (text, "") if own is None else (text[:own], text[own:].removeprefix("\n\n"))


def _fix_lines(entry: Mapping[str, Any]) -> List[str]:
    """A part's signed corrections and rejections of its members, beside every form of the part's words."""
    return [f"- {'correction by me of' if fix['kind'] == 'correction' else 'my rejection of the draft'} {fix['target']}:\n"
            f"{_indented(fix['text'])}" for fix in entry.get("fixes") or ()]


def _page_label(entry: Mapping[str, Any]) -> str:
    """The page's first sentence, without a length cut or an invented summary, its signed corrections beside it
    (``_own_words``, ``_fix_lines``), and its reader."""
    words, corrections = _own_words(entry)
    said = "\n\n".join(text for text in (first_sentence(words), corrections) if text)
    attribution = (f" (draft by {entry.get('signer') or 'Light'}, not yet accepted)"
                   if entry.get('status') == 'draft' else '')
    missing = " (record opening, no lead)" if "lead_rule" not in entry else ""
    return "\n".join([_page_pointer(entry) + attribution + missing, _indented(said), *_fix_lines(entry)])


def _page_short(entry, gone=(), *, lead=False):
    return "\n".join([_page_label(entry) if lead else _page_pointer(entry), *page_quote_lines(entry, gone, short=True)])


def _told_lines(account_id: str, children: Mapping[str, List[Dict[str, Any]]]) -> List[str]:
    """One count and period over the selected records, nested selections included; exact IDs stay in the reader.

    ``children``: the told records by their teller, indexed once per render (``render_story``)."""
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


def _page_lines(entry: Mapping[str, Any], gone=()) -> List[str]:
    head = f"### {entry['label']} · {entry['period']} · {entry['kind']} {entry['id']}"
    if entry.get("kind") == "account":  # its words, compact composition, and new source changes still in full
        return ["", head, *account_lines(entry)]
    lines = ["", head, _indented(entry["text"]), *page_quote_lines(entry, gone)]
    if entry.get("status") == "draft":
        lines.append(f"(draft by a helper ({entry.get('signer') or 'Light'}), not yet accepted or rejected by me)")
    elif entry.get("status") == "accepted":
        lines.append(f"(drafted by a helper ({entry.get('signer') or 'Light'}), accepted by me)")
    if entry.get("stamp"):
        lines.append(f"({entry['stamp']})")
    lines.extend(entry.get("terminal_links") or ())
    return lines + _fix_lines(entry)  # a record's own corrections stand inside its text (chronicle_store._interpret)


def _retold(item: Mapping[str, Any], short: bool = False) -> str:
    """A record of this room's page (retold, under a part, or a storyless view's own page with its evidence), whole or by address with its length (F4)."""
    head = (f"#### {item['kind']} {item['id']} — {item['period']} — under part {item['part']}" if item.get("part")
            else f"#### {item['id']} — {item['period']}" + (f" — {item['of']}" if item.get("of") else ""))
    if short:
        return f"{head} — {len(item['text'])} chars — memory_read(node_id='{item['id']}')"
    return "\n".join([f"{head}\n{_indented(item['text'])}", *(_page_lines(item)[3:] if "stamp" in item else ())])


def _told_here_line(item: Mapping[str, Any]) -> str:
    """Where this room's records my accounts tell went: their count and span, the accounts and the room's reader."""
    many = len(item["accounts"]) > 1
    return (f"- {item['count']} record{'' if item['count'] == 1 else 's'}, {_period(item['span'])}: told in my story by "
            f"my account{'s' if many else ''} {', '.join(item['accounts'])}; memory_read(room_id='{item['room_id']}') "
            "lists each with its words and teller")


def _row_pointer(item: Mapping[str, Any], what: str) -> str:
    if item.get("kind") == "place_fact":  # no words to size
        return f"{item['head']} (reaction — read by address)"
    return f"{item['head']} ({what}, {item['chars']} chars — read by address)"


def _row_opening(item: Mapping[str, Any], what: str) -> str:
    if item.get("kind") == "place_fact":
        return _row_pointer(item, what)
    opening = str(item.get("text") or item["line"].removeprefix(item["head"])).strip().splitlines()[0:1]
    return f"{item['head']} ({what}, {item['chars']} chars; opening, no retelling): " + " ".join(opening)[:160]


def _rows_line(room: Mapping[str, Any], items: List[Dict[str, Any]]) -> str:
    first, last = min(items, key=lambda i: i['pos']), max(items, key=lambda i: i['pos'])
    mine = sum(item['kind'] == 'ouroboros' for item in items)
    reactions = sum(item['kind'] == 'place_fact' for item in items)
    return (f"{len(items)} earlier rows, oldest first, by address ({mine} my replies, "
            f"{len(items) - mine - reactions} people's words{f', {reactions} reactions' if reactions else ''}), "
            f"{_minute(first['ts'])} → {_minute(last['ts'])}; "
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
    reactions = f"reactions {facts['reactions']}, " if facts.get("reactions") else ""
    reactions += f"author not recorded {facts['unattributed']}, " if facts.get("unattributed") else ""
    rows = (f"open {_minute(facts['earliest'])} → {_minute(facts['latest'])}; people {facts['people']}, "
            f"mine {facts['mine']}, {reactions}task facts {facts['task_facts']}, ~{facts['chars']} chars" if facts["rows"]
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
    summed = {key: sum(item.get(key, 0) for item in facts) for key in ("rows", "people", "mine", "reactions", "task_facts", "chars", "notes", "pages")}
    summed.update(earliest=min((item["earliest"] for item in facts if item["earliest"]), default=""),
                  latest=max((item["latest"] for item in facts), default=""),
                  last_covered=max((item["last_covered"] for item in facts), default=""))
    return f"{len(rooms)} open rooms; {_facts_text(summed, 'these rooms')}; memory_read(room_id='*', start=0)"


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


STORY_STEPS = ("F3", "F5l", "F5", "F5a", "F7")  # F7 affects only the story's human quotes


def render_story(snapshot: MemoryViewSnapshot, level: FloorLevel = FULL_VIEW) -> str:
    """Block B's tail, ``## My story``: the settled retellings, then the pages not yet folded, then the status.

    Every unfolded retold record is whole, then my parts and selected accounts, then the pages no
    part or account holds yet in publication order (``_story_pages``), so a new page joins the end;
    the records an account tells retain their combined period and count with its composition reader. Its bytes depend only on the
    chronicle, the room labels, the helper route and ``level`` (the physical floor's; the
    empty level renders the full view): F3 turns a room's retold records into one line; F5l keeps
    page first sentences, then F5 addresses pages and parts by covered-period age, independently
    of display order. Their human quotes stay until F7. Selected accounts stay whole.
    """
    key = _story_key(level)
    counted("view_renders", key=("story", id(snapshot), key))
    return _story_text(snapshot, dict(key))


def _story_key(level: FloorLevel) -> Tuple[Any, ...]:
    return tuple((step, shown) for step, ids in level.addressed if step in STORY_STEPS
                 if (shown := tuple(i for i in ids if i.startswith("quote:")) if step == "F7" else ids))


def story_chars(snapshot: MemoryViewSnapshot, level: FloorLevel = FULL_VIEW) -> int:
    """``json_chars(render_story(snapshot, level))``, read from the snapshot's pieces: what pricing a level costs."""
    key = _story_key(level)
    return snapshot.piece(("story", key), lambda: _story_text(snapshot, dict(key)))


def whole_story_at(snapshot: MemoryViewSnapshot, level: FloorLevel) -> bool:
    """Whether this level's story is the whole story's text. Prices decide unless a level naming story steps prices the
    same as the whole, and only then are the two texts drawn (two renders of the build's work)."""
    key, same = _story_key(level), story_chars(snapshot, level) == story_chars(snapshot)
    if not same or not key:
        return same
    counted("view_renders", 2)
    return _story_text(snapshot, dict(key)) == _story_text(snapshot, {})


def _story_text(snapshot: MemoryViewSnapshot, gone: Mapping[str, Tuple[str, ...]]) -> str:
    spec = snapshot.spec
    if not spec.story:
        return ""
    if not snapshot.active:
        return (f"## My story — unavailable now ({snapshot.store_status.get('reason')})\n\n"
                f"The old memory files are untouched: read_file(root='runtime_data', path='{LEGACY_FILE}').")
    told: Dict[str, List[Dict[str, Any]]] = {}
    for entry in snapshot.story:
        if entry.get("told_by"):
            told.setdefault(entry["told_by"], []).append(entry)
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
    quotes_gone = set(gone.get("F7", ()))
    aggregate = [entry for entry in pages if entry['id'] in shown] if gone.get('F5a') else []
    # A selected form changes this entry alone, without regrouping unrelated entries or changing their headers.
    for entry in pages:
        if aggregate and entry is aggregate[0]:
            lines += ['', '### My older pages and parts, by address', _story_line(aggregate)]
        if not aggregate or entry['id'] not in shown:
            lines += (["", _page_short(entry, quotes_gone)] if entry["id"] in shown else
                      ["", _page_short(entry, quotes_gone, lead=True)] if entry["id"] in gone.get('F5l', ())
                      else _page_lines(entry, quotes_gone))
        else:
            lines += page_quote_lines(entry, quotes_gone, short=True)
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
    """A source-derived label, never a summary or a function of the start allowance; its signed corrections beside it."""
    words, corrections = _own_words(note)
    sentence = first_sentence(' '.join(words.split()))
    label = sentence + (" (record opening, no lead)" if "lead_rule" not in note else "")
    return (f"note {note['id']} by {note['role']} on {note['date']}: {label}; "
            f"memory_read(node_id='{note['id']}')") + ("\n" + _indented(corrections) if corrections else "")


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
        release = "" if helper else f"; memory_mark(release_ids=['{mark['id']}'], reason=…)"
        lines += [f"- [{mark['room']}; kept since {mark['date']}] " + _indented(mark["text"]).lstrip(),
                  f"{INDENT}full source (sha256 {late['sha256'][:12]}): {late['source']}{release}"]
    return "\n".join(lines)


def _marks_text(marks: List[Dict[str, Any]]) -> str:
    lines = ["## Marks I keep in view", ""]
    for mark in marks:
        lines.append(f"- [{mark['scope']}; {mark['room']}] " + _indented(mark["text"]).lstrip()
                     + f" — marked by {mark['by']} on {mark['date']}; target {mark['target']}; "
                     + (f"also shown after a task's first call matching {mark['anchor']}; " if mark.get("anchor") else "")
                     + f"memory_mark(release_ids=['{mark['id']}'], reason=…)")
        if mark.get("quote"):
            lines.append(f"{INDENT}quote: " + _indented(mark["quote"]).lstrip())
    return "\n".join(lines)


def _room_text(room: Mapping[str, Any], gone: Mapping[str, List[str]]) -> str:
    """``## This room (<label>) — head <n>``: the room page, the words that started it, a project root's sealed
    rows of its human (``project_words``; folded by F7 into dated addresses), notes, two lanes, and last the
    standing facts that change with every row (so the room's earlier bytes stay a reusable prefix)."""
    from ouroboros.owner_words import _FRAMES, _HEADINGS
    from ouroboros.memory_view_work import folded_work_rows, task_line, works, task_gist_line, task_gists_before_work_lines, row_reader

    retold, mine, people, facts = (set(gone.get(step, ())) for step in ("F4", "F2", "F7", "F1"))
    people |= set(gone.get("F7p", ()))
    bare = {ident[5:] for ident in mine | people if ident.startswith("bare:")}
    mine |= {ident[5:] for ident in mine if ident.startswith("bare:")}
    people |= {ident[5:] for ident in people if ident.startswith("bare:")}
    lines = [f"## This room ({room['label']}) — head {room['head']}"]
    if room["legacy"]:
        lines += ["", "### Retold before the update (helper retelling, not lived)"]
        lines += [_retold(item, item["id"] in retold) for item in room["legacy"]]
    lines += (["", "### Pages under my parts" if all(i["part"] for i in room["under_parts"]) else "### Pages of this room",
               *(_retold(item, item["id"] in retold or item.get("standing", False)) for item in room["under_parts"])]
              if room["under_parts"] else [])
    if room.get("told"):
        lines += ["", "### Records of this room my accounts tell"] + [_told_here_line(item) for item in room["told"]]
    if room["origins"]:
        lines += ["", "### Words that started this work (retention-proof)"]
        lines += [f"[{item['ts'] or 'time not recorded'}; owner; {item['ref']}] " + _indented(item["text"]).lstrip()
                  for item in room["origins"]]
    if room.get("project_words"):  # a project root: its human's rows that left the open conversation, whole and dated
        lines += ["", _HEADINGS["root"], _FRAMES["root"], "\n\n".join(  # under the actor frame; a gap line has no address
            item["short"] if item.get("address") in people else item["line"] for item in room["project_words"])]
    if room["notes"]:
        lines += ["", "### My notes not yet sealed"] + _room_note_lines(room["notes"], gone)
    if room["lane1"]:
        work_rows, work_lines = folded_work_rows(room, gone)
        addressed = mine | people | work_rows
        aggregate = [item for item in room['lane1'] if item['address'] in addressed
                     and not item.get('immune')] if gone.get('F7a') else []
        grouped = {item['address'] for item in aggregate}
        omitted = grouped | work_rows
        description = "people and my replies" if work_lines and not aggregate else "verbatim: people and my replies"
        lines += ["", f"### Open conversation since {room['since']} ({description})"]
        lines += [_FRAMES["root"]] if room.get("project_words") is not None else []  # the root's frame over the open rows
        if work_lines and not aggregate:
            lines.append("Completed work replies may stand as their gist. " + row_reader(room['room_id']))
        for item in room['lane1']:
            if aggregate and item is aggregate[0]:
                lines.append(_rows_line(room, aggregate))
            if item['address'] in work_lines and not aggregate:
                lines.append(work_lines[item['address']])
            if item['address'] not in omitted:
                render = _row_pointer if item['address'] in bare or item['address'] in gone.get('F7', ()) else _row_opening
                lines.append(item['line'] if item.get('immune') else render(item, 'my reply')
                             if item['address'] in mine else render(item, 'words') if item['address'] in people else item['line'])
    if room["lane2"]:
        early = [item for item in room["lane2"] if item["first"] in facts]
        gists = task_gists_before_work_lines(room, gone)
        task_room = ("<room from address>" if any(str(t['gist']['room_id']) != str(room['room_id']) for t in gists)
                     else f"'{room['room_id']}'")
        folded = {key for key, _rows in works(room) if any("work:" + key + suffix in mine for suffix in (":progress", ":final"))}
        lines += ["", "### Task facts of this conversation (host; one line per task; a row:… address reads with "
                      f"memory_read(room_id={task_room}, rows=true, from=<address>, to=<address>))"]
        lines += ([_facts_line(room, early)] if early else []) + [
            task_gist_line(t) for t in gists] + [
            task_line(item, folded=not gone.get("F7a") and item.get("task") in folded)
            for item in room["lane2"] if item["first"] not in facts]
    if len(lines) == 1:
        lines += ["", "Nothing open, retold or noted in this room."]
    lines += ["", "### This room now", _facts_text(room["facts"])]
    lines += [f"- sealed with the page {item['id']} of {item['label']}: memory_read(node_id='{item['id']}')"
              for item in room.get("sealed_by") or ()]
    return "\n".join(lines)


def render_room(snapshot: MemoryViewSnapshot, level: FloorLevel = FULL_VIEW, *, floor_note: str = "") -> str:
    """Block C's memory middle: a helper's owner words, late reviews, marks, this room, then my other conversations.

    ``level`` is the physical floor's (the empty level renders the full view);
    ``floor_note`` (``### Physical floor``) closes the block. Before the import has completed
    the room is one line naming the reason and a reader that works without the chronicle.
    """
    counted("view_renders", key=("room", id(snapshot), level.addressed, floor_note))
    texts = _room_parts(snapshot, dict(level.addressed), lambda _key, make: make())
    return "\n\n".join(part for part in (*texts, floor_note if snapshot.active else "") if part)


def room_chars(snapshot: MemoryViewSnapshot, level: FloorLevel = FULL_VIEW, *, floor_note: str = "") -> int:
    """``json_chars(render_room(snapshot, level, floor_note=...))`` from the snapshot's pieces and the blank lines joining them."""
    sizes = snapshot.__dict__.setdefault("_room_sizes", {})
    key = tuple((step, ids) for step, ids in level.addressed if step in (*_ROOM_STEPS, 'F6w', 'F6g', 'F6a'))
    if key not in sizes:
        chars = [count for count in _room_parts(snapshot, dict(level.addressed), snapshot.piece) if count]
        sizes[key] = sum(chars) + 4 * max(0, len(chars) - 1), bool(chars)
    total, present = sizes[key]
    if floor_note and snapshot.active:
        total += json_chars(floor_note) + (4 if present else 0)
    return total  # Only numeric prices are cached, never candidate texts.


_ROOM_STEPS = ("F1", "F2", "F4", "F7", "F7p", "F7a", "F8", "F8a")  # the steps ``_room_text`` reads


def _room_parts(snapshot: MemoryViewSnapshot, gone: Mapping[str, Tuple[str, ...]],
                piece: Callable[[Any, Callable[[], str]], Any]) -> List[Any]:
    """The block's parts before the floor note, in order and joined by a blank line, each through ``piece``: its
    price (``snapshot.piece``) or its text."""
    from ouroboros import memory_view_conversations as conversations

    parts = [piece(("owner_words",), lambda: snapshot.owner_words)]
    spec = snapshot.spec  # a view drawing nothing of its room (a declared request's) prints no ``## This room``
    room = snapshot.room if spec.room_page or spec.room_lanes or spec.origin_words else None
    if not snapshot.active:
        if room:
            parts.append(piece(("inactive_room",), lambda: (
                f"## This room ({room.get('label')})\n\nOpen conversation unavailable until my memory is "
                f"activated ({snapshot.store_status.get('reason')}); read it: chat_history(count=100) — every "
                f"room, newest first; this room is chat_id {room['room_id']}.")))
        return parts
    # A snapshot persisted before other rooms' late reviews left this view (a pointer entry: ``own_room`` False, no
    # words) is re-rendered after an update (``context_fit``); such an entry draws nothing.
    late = [mark for mark in snapshot.marks if mark.get("late_review") and mark["late_review"].get("own_room") is not False]
    ordinary = [mark for mark in snapshot.marks if not mark.get("late_review")]
    if late:
        parts.append(piece(("late_reviews",), lambda: _late_reviews_text(late, snapshot.spec.role)))
    if ordinary:
        parts.append(piece(("marks",), lambda: _marks_text(ordinary)))
    if room:
        parts.append(piece(("room", tuple((step, gone.get(step, ())) for step in _ROOM_STEPS)),
                           lambda: _room_text(room, gone)))
    if snapshot.spec.other_conversations and snapshot.conversations:
        parts.extend(conversations.pieces(list(snapshot.conversations), gone, piece, index=conversations.source_index(snapshot)))
    return parts


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
    if spec.other_conversations:
        memory.append("my other conversations")
    else:
        missing.append("other conversations")
    if snapshot is not None and not snapshot.active and memory:
        missing.append(f"{', '.join(memory)} (my memory is not activated yet: {snapshot.store_status.get('reason')})")
    else:
        loaded += memory
    loaded.append("knowledge (overview, index, project journal and workpad)" if spec.knowledge
                  else "the knowledge outline (every note by address and title)")  # context.helper_navigation_sections
    if not spec.knowledge:  # the whole index rides the first accepted delegation's result (delegation_knowledge_index)
        missing.append("the overview, the whole knowledge index (it arrives with your first delegation), the project journal and workpad")
    if spec.owner_words and (snapshot is None or snapshot.owner_words):  # named only when the block was drawn
        loaded.append("my human's words for this work; my human's later words reach you at your next round")
    missing += ["the global scratchpad", "earlier task reports"]
    return ("## Working sources\n\n"
            f"Loaded above: {', '.join(loaded)}; your own recent process is loaded below. "
            f"Not loaded: {', '.join(missing)}; memory_read, chat_history, knowledge_list, knowledge_read, "
            "journal_read, workpad_read and get_task_result reach them, as they reach your parent.\n" + CHILD_ROLE_TEXT)
