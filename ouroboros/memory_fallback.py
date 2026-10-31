"""One signed Light draft only while consciousness is off (or unknown).

With consciousness off (or unknown), a root whose own start memory lowered to Low while everything
else fit the window (``crowded_start``: its view fact's Max room, ``memory_floor.crowded_by_memory``)
comes first: one room's oldest run of unfolded pages, whoever wrote them, folded into a part
(``max_room_units``); then the existing acting-draft selection:

1. The task's view showed this room's open rows only by address (``shortage_from_trace``,
   ``open_rows``): the room's oldest open segment, up to the newest addressed row (a
   later one when an older one is unreadable to the helper or already refused).
2. Else the view showed old narrative only by pointer (``narrative``): adjacent unfolded
   runs of one kind of one room among those pointers, oldest first, folded into a part.
3. Else, only after a root that was not a direct turn (the late phase of a direct turn
   blocks the owner's next message), the oldest unfolded period of the old memory,
   blocks 1-22 (block 0 is already a retelling of two months the mind folds itself):
   the oldest unsealed segment of that unit's rows, its retelling given as a hint.

The draft acts at once under the helper's name (``author.kind == "helper"``) until the
mind accepts, rejects or corrects it. A page carries no ``expected_sequence`` (an
unrelated record of a live room never voids it; a double cover is ``already_sealed``);
a part carries the room head read at selection. Rows the helper received only by
address, and the mind's notes it never reads, are not covered: they stay open, since a
page seals only what its writer actually read.

One call, no retry: a refusal or keep-original on the same input and Light route is
recorded in its own ``fallback_refusal:<unit>`` scan-state key; an available response
body is retained. An exact rejected draft is skipped until input or route changes. The helper never nominates
knowledge, writes ``overview`` or touches the old dialogue files. Nothing here activates
the chronicle: before activation the writer makes no call and creates no file.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import pathlib
from dataclasses import dataclass, field, replace
from math import ceil, inf
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from ouroboros import chat_chain, memory_inventory
from ouroboros.chronicle_import import row_lineage
from ouroboros.chronicle_store import ChronicleStore, PublishResult, draft_signer, source_time_span
from ouroboros.consolidator import LIGHT_ANSWER_CEILING_TOKENS
from ouroboros.dialogue_provenance import (memory_row_header, recorded_finality, render_memory_row, render_row_text,
                                           row_class)
from ouroboros.memory_floor import crowded_by_memory
from ouroboros.memory_inventory import VIEW_TRACE_KEY, OpenSegment, ShortageFact, shortage_from_trace
from ouroboros.utils import extract_trailing_json_object, utc_now_iso

log = logging.getLogger(__name__)

LABEL = "memory_fallback_page"
ANSWER_RESERVE_TOKENS = LIGHT_ANSWER_CEILING_TOKENS  # the answer ceiling the Light transport sends
INDENT = "  "
# A refusal of these kinds is a fact about this input on this route: paying again changes nothing.
RECEIPT_KINDS = frozenset({"context_overflow", "output_truncated", "empty_summary", "invalid", "quote_mismatch",
                           "target_missing", "revision_required"})
CONFLICT_REASONS = frozenset({"revision_conflict", "already_sealed", "already_folded"})  # someone sealed first

_PAGE = ("You are a helper drafting one memory page for Ouroboros from the rows below. You are not Ouroboros "
         "and did not live these events: write in the third person and attribute every statement to whoever "
         "made it — the owner or another person, Ouroboros, a child task, the host.\n"
         "Cover only the rows supplied; say nothing of what is not here. A row shown only as an address was "
         "not read: name it if it matters, never guess its words.\n")
_PART = ("You are a helper drafting one memory part for Ouroboros: one account that folds the records below, "
         "oldest first. You are not Ouroboros and did not live these events: write in the third person and "
         "use only these records and keep each statement's speaker and confidence. An account's 'I' names "
         "its author, not the owner. A prior account is evidence of what its author reported, not fresh "
         "verification or current permission.\n"
         "For each thread preserve who chose what, when and under which limits; what was done, what remains "
         "open, and what a signed correction changes. A later scoped choice does not lift unrelated limits. "
         "Keep completed work distinct from missing work, possibilities from accomplished effects, and "
         "unknown from disproved. Preserve both new evidence and the limits it does not remove.\n"
         "Preserve each author's conclusions—lessons learned, errors admitted and reasons for choices—alongside "
         "decisions, outcomes and their limits. Combine genuinely repeated explanations across records; omit "
         "incidental particulars, not distinctions that carry meaning. Keep substantive history rather than a "
         "disclaimer. Produce a shorter faithful account when useful, not merely a rewording; otherwise keep "
         "the originals.\n")
_COMMON = ("A task's outcome is its host stamp: never call a task done, accepted or verified unless its stamp "
           "says so.\n"
           "Keep what a later reader needs: what was asked and by whom, what was decided or refused, what was "
           "done and how it ended, what stays open.\n"
           "Quote the words of people and of Ouroboros only through the quotes field; in the text, say it in "
           "your own words.\n")
_QUOTES = ("Quote the decisive words — what was asked, decided, promised or refused — exactly in quotes.\n"
           "Each quote is an exact substring of one supplied row, with that row's address (row:…) and its "
           "speaker: the author class written first in the row's brackets — human, ouroboros, child, host, "
           "helper or unattributed.\n"
           "A quote is copied character for character, markdown included (**, _, `).\n")
_NO_QUOTES = "No chat rows are supplied here, so leave quotes empty.\n"
_MEMBER_QUOTES = ("No chat rows are supplied here: quote only by copying a quote shown under a record below exactly, "
                  "with its address and speaker, into quotes.\n")
_ANSWER = ('Answer with one JSON object and nothing else: {"text": "<the account>", "quotes": [{"address": '
           '"row:…", "text": "<exact words>", "speaker": "<class>"}]}')
_KEEP_ORIGINAL = ('\nFor this part only, when no useful shorter faithful account is possible, answer instead '
                  'with {"keep_original": true, "reason": "<why retaining these records is better>"}.')


# --- when and on which route -----------------------------------------------------------------------

def fallback_active() -> bool:
    """Whether the fallback writer works now: consciousness is not KNOWN to be on (#1307).

    The alarm clock's own read (``supervisor.state.load_state`` + ``control_is``); an
    unreadable or unconfirmed switch is off, so the helper drafts.
    """
    from supervisor.state import control_is, load_state  # D15->D08 is lazy-only

    try:
        state = dict(load_state() or {})
    except Exception:
        state = {}
    return not control_is(state, "bg_consciousness_enabled", True)


def light_binding() -> Dict[str, Any]:
    """The Light binding a call dispatches on now (``consolidator._light_dispatch_binding``), JSON-plain."""
    from ouroboros.consolidator import _light_dispatch_binding

    return json.loads(json.dumps(_light_dispatch_binding(), ensure_ascii=False, sort_keys=True, default=str))


def _chars_measure(text: str) -> int:
    return ceil(len(text) / 4)


def _light_fit(binding: Mapping[str, Any]) -> Tuple[Optional[int], Callable[[str], int]]:
    """``(input limit in tokens or None, measure)``: the Light route's known window less the answer reserve.

    The same route evidence and estimator the transport's own preflight uses; an
    unknown window is no fitting at all (a provider refusal by size is a receipt).
    """
    try:
        from ouroboros.context_window import resolve_window
        from ouroboros.context_fit import (_route_calibration_ratio, estimate_context_prompt_tokens,
                                           resolve_context_fit_route)
        from ouroboros.provider_models import provider_for_model

        task = {"model": binding["model"], "use_local_model": binding["use_local"], "model_role": "light",
                "credential_profile_id": binding.get("model_account_override"), "model_route": None}
        route, evidence = resolve_context_fit_route(
            task, allow_fetch=bool(binding["use_local"]) or provider_for_model(binding["model"]) == "claudexor")
        density = _route_calibration_ratio(None, evidence.route_fp, route["model"])
        from ouroboros.response_limits import ResponseLimit
        reserve = ResponseLimit(**(getattr(evidence, "response_limit", {}) or {})).ceiling(ANSWER_RESERVE_TOKENS)
        if binding["use_local"]:
            from ouroboros.llm_local import local_context_limits

            _local_window, reserve = local_context_limits(reserve)
        scope = route.get("window_observation_fp") or evidence.route_fp
        window = resolve_window(evidence, reserve, observation_fp=scope, bound=bool(scope))
    except Exception:
        log.debug("fallback writer: Light window unknown", exc_info=True)
        return None, _chars_measure

    def measure(text: str) -> int:
        return ceil(estimate_context_prompt_tokens([{"role": "user", "content": text}], None,
                                                  provider=route["provider"], reasoning_effort="low") * density)

    return window.usable, measure


# --- which unit ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FallbackUnit:
    """One unit the helper may draft: a page over rows of one room, or a part over adjacent records."""
    kind: str  # "page" (open rows after the frontier), "legacy" (one old period's rows of a room), "part"
    room_id: str
    key: str  # its receipt's suffix after memory_inventory.FALLBACK_REFUSAL_PREFIX
    segment: Optional[OpenSegment] = None
    record_id: str = ""  # a legacy unit's record: its retelling is a hint, never a source
    member_ids: Tuple[str, ...] = ()
    head_sequence: int = 0  # the room head at selection (a part's ``expected_sequence``)

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "room_id": self.room_id, "key": self.key}
        if self.record_id:
            out["record_id"] = self.record_id
        if self.member_ids:
            out["member_ids"] = list(self.member_ids)
        if self.segment is not None:
            out.update({"from": chat_chain.format_address(self.segment.from_addr),
                        "to": chat_chain.format_address(self.segment.to_addr), "rows": len(self.segment.rows)})
        return out


def _stream_start(record: Mapping[str, Any]) -> float:
    covers = record.get("covers") if isinstance(record.get("covers"), Mapping) else {}
    span = covers.get("stream_span") or (covers.get("raw_range") or {}).get("pos")
    return span[0] if isinstance(span, list) and span and type(span[0]) is int else inf


def _unfolded_run(store: ChronicleStore, room: str, kind: str) -> List[str]:
    """The room's acting, unfolded records of one kind in story order (the order a part's members keep)."""
    if kind == "legacy":
        return [p["node_id"] for p in store.legacy_pointer_rows()
                if p["kind"] == "legacy" and str(p["room_id"]) == room and not p["folded_into"]]
    return [r["id"] for r in store.pages_of_room(room) if r["kind"] == kind and not r["folded_into"]]


def _narrative_units(store: ChronicleStore, pointer_records: Tuple[str, ...]) -> Iterator[FallbackUnit]:
    """Disjoint adjacent pointer runs, oldest first; one refused run cannot hide the rest."""
    runs: Dict[Tuple[str, str], List[str]] = {}
    found = []
    for ident in pointer_records:
        record = store.get(ident)
        if not record or record.get("kind") not in ("page", "part", "legacy"):
            continue
        place = (str(record["room_id"]), str(record["kind"]))
        if place not in runs:
            runs[place] = _unfolded_run(store, *place)
        if ident in runs[place]:
            found.append(((_stream_start(record), record["sequence"]), place, ident))
    remaining = set(pointer_records)
    for _order, place, first in sorted(found):
        if first not in remaining:
            continue
        run, members = runs[place], []
        for ident in run[run.index(first):]:
            if ident not in remaining:
                break
            members.append(ident)
        remaining.difference_update(members)
        room = place[0]
        yield FallbackUnit("part", room, f"part:{room}:{first}", member_ids=tuple(members),
                           head_sequence=store.room_head(room))


def _legacy_units(root: pathlib.Path, store: ChronicleStore) -> Iterator[FallbackUnit]:
    """Unfolded legacy units of blocks 1-22 with established rows: block, then their first unsealed row."""
    spans = {p["node_id"]: memory_inventory._exact_range(p) for p in store.legacy_pointer_rows()}
    units = [unit for unit in memory_inventory.legacy_units(store, root)
             if not unit.folded and unit.raw == "exact" and type(unit.block) is int and unit.block >= 1
             and unit.uncovered and spans.get(unit.record_id)]
    for block in sorted({unit.block for unit in units}):
        found = []
        for unit in (unit for unit in units if unit.block == block):
            for index, segment in enumerate(memory_inventory.open_segments(
                    root, store, unit.room_id, pos_range=spans[unit.record_id])):
                first = segment.rows[0][0]["row_sha256"][:12]
                key = unit.record_id if index == 0 else f"{unit.record_id}:from:{first}"
                found.append((segment.rows[0][2], unit.record_id, key, segment))
        for _pos, record_id, key, segment in sorted(found, key=lambda item: item[:2]):
            yield FallbackUnit("legacy", segment.room_id, key, segment=segment, record_id=record_id,
                               head_sequence=segment.head_sequence)


def crowded_start(trace: Any) -> bool:
    """This root's own start was lowered to Low by memory: the view fact its trace keeps (``VIEW_TRACE_KEY``) names a
    Max room ``crowded`` while everything but memory fit the window (``memory_floor.crowded_by_memory``). No fact,
    a held start, a route, or tool schemas that tipped it alone: ``False``."""
    fact = trace.get(VIEW_TRACE_KEY) if isinstance(trace, Mapping) else None
    return crowded_by_memory(fact.get("max_room") if isinstance(fact, Mapping) else None)


def max_room_units(store: ChronicleStore) -> Iterator[FallbackUnit]:
    """After a root's start that memory crowded (``crowded_start``): a room's runs of unfolded pages no shown
    account tells (the view's ``told_map``), oldest first, whoever wrote them, the room holding most such text
    first; one page is a run too, Light judges whether folding it helps, and a run it kept as it is leaves the
    next one in turn (``select_unit``). Never an account across rooms: that is the mind's."""
    from ouroboros.memory_view_account import told_map

    told = set(told_map([{"id": r["id"], "replaces": r["selection"].get("replaces") or (),
                          "selection_sequence": r["selection"]["sequence"]}
                         for r in store.accounts() if (r.get("selection") or {}).get("shown")]))
    sizes: Dict[str, int] = {}
    for record in store.room_records(None, kinds=("page",)):
        if not record.get("folded_into") and record["id"] not in told:
            sizes[str(record["room_id"])] = sizes.get(str(record["room_id"]), 0) + len(str(record.get("current_text") or ""))
    for room in sorted(sizes, key=lambda room: (-sizes[room], room)):
        head, run = store.room_head(room), []
        # A told page ends a run: a part's members are adjacent in the room's unfolded order.
        for ident in [*_unfolded_run(store, room, "page"), None]:
            if ident is not None and ident not in told:
                run.append(ident)
            elif run:
                yield FallbackUnit("part", room, f"max:{room}:{run[0]}", member_ids=tuple(run), head_sequence=head)
                run = []


def candidate_units(root: Any, store: ChronicleStore, shortage: Optional[ShortageFact], *,
                    allow_legacy: bool) -> Iterator[FallbackUnit]:
    """Units in consideration order: the shortage's disjoint runs, then old periods."""
    root = pathlib.Path(root)
    if shortage is not None and shortage.kind == "open_rows":
        # Oldest run first; a later run only when an older one has no readable input or a receipt.
        for segment in memory_inventory.open_segments(root, store, shortage.room_id,
                                                      until=shortage.newest_addressed_row):
            first = segment.rows[0][0]["row_sha256"][:12]
            yield FallbackUnit("page", segment.room_id, f"open:{segment.room_id}:{first}", segment=segment,
                               head_sequence=segment.head_sequence)
    elif shortage is not None and shortage.kind == "narrative":
        yield from _narrative_units(store, shortage.pointer_records)
    if allow_legacy:
        yield from _legacy_units(root, store)


def _refusals(store: ChronicleStore) -> Dict[str, Any]:
    return memory_inventory.fallback_refusals(store.scan_state())


def select_unit(root: Any, store: ChronicleStore, shortage: Optional[ShortageFact], *, allow_legacy: bool,
                binding: Mapping[str, Any], fit: Callable[[], Tuple[Optional[int], Callable[[str], int]]],
                units: Optional[Iterable[FallbackUnit]] = None,
                ) -> Optional[Tuple[FallbackUnit, "WriterInput"]]:
    """First readable input/route not already refused, kept or rejected, including untouched suffixes."""
    refusals = _refusals(store)
    part_rooms = {}
    part_units = None
    for unit in units if units is not None else candidate_units(root, store, shortage, allow_legacy=allow_legacy):
        budget, measure = fit()
        records, part_cache = None, None
        if unit.kind == "part":
            if part_units is None:
                part_units = {u.record_id: u for u in memory_inventory.legacy_units(store, root)}
            # Reuse this selection pass's room view, never a later head's view.
            # Publication still checks the selected head after the model call.
            key = (unit.room_id, unit.head_sequence)
            if key not in part_rooms:
                part_rooms[key] = ({r["id"]: r for r in store.room_records(unit.room_id)}, {})
            records, part_cache = part_rooms[key]
        while True:
            draft = build_writer_input(root, store, unit, budget_tokens=budget, measure=measure,
                                       part_records=records, part_units=part_units, part_cache=part_cache)
            if draft is None:
                break
            keys = _suppression_keys(unit)
            suppressed = any(isinstance(receipt := refusals.get(key), dict)
                             and receipt.get("input_sha256") == draft.input_sha256
                             and receipt.get("light_binding") == dict(binding) for key in keys)
            if not suppressed and not any(store.rejected_input(unit.room_id, key, draft.input_sha256, binding)
                                          for key in keys):
                return unit, draft
            # Only the fitted prefix was attempted. Never retry its overlapping
            # subsets; the untouched suffix can still be useful in this pass.
            unit = _untouched_suffix(unit, draft)
            if unit is None:
                break
    return None


def _suppression_keys(unit: FallbackUnit) -> Tuple[str, ...]:
    """Existing keys for these rows, not new attempt identities.

    Legacy enumeration uses the record id; an untouched suffix uses its first row.
    Sealing only the prefix makes that suffix the next head. Check both stored keys
    in either position, still requiring the exact input and route to match.
    """
    if unit.kind != "legacy" or unit.segment is None or not unit.segment.rows:
        return (unit.key,)
    first = unit.segment.rows[0][0]["row_sha256"][:12]
    return tuple(dict.fromkeys((unit.key, unit.record_id, f"{unit.record_id}:from:{first}")))


def _untouched_suffix(unit: FallbackUnit, draft: "WriterInput") -> Optional[FallbackUnit]:
    """Advance past the fitted input, not merely its sealed rows (some words were only addressed)."""
    if unit.kind == "part":
        suffix = unit.member_ids[len(draft.member_ids):]
        return replace(unit, key=f"part:{unit.room_id}:{suffix[0]}", member_ids=suffix) if suffix else None
    if unit.segment is None or draft.fitted_to is None:
        return None
    rows = unit.segment.rows
    end = next((i for i, (address, _row, _pos) in enumerate(rows)
                if memory_inventory._is_row(address, draft.fitted_to)), len(rows))
    suffix = rows[end + 1:]
    if not suffix:
        return None
    first = suffix[0][0]["row_sha256"][:12]
    key = f"{unit.record_id}:from:{first}" if unit.kind == "legacy" else f"open:{unit.room_id}:{first}"
    return replace(unit, key=key, segment=replace(unit.segment, rows=suffix, from_addr=suffix[0][0]))


# --- what the helper reads -------------------------------------------------------------------------

@dataclass(frozen=True)
class WriterInput:
    """The helper's whole input and what its draft would seal."""
    prompt: str
    input_sha256: str
    input_tokens: Optional[int]
    covers: Optional[Dict[str, Any]] = None  # a page's covers: rows read verbatim or as their task's stamp line
    positions: Dict[str, int] = field(default_factory=dict)  # row_sha256 -> stream pos (quotes of pre-epoch rows)
    addressed_rows: Tuple[str, ...] = ()  # shown only by address: not read, so not covered; they stay open
    stamp: Optional[Dict[str, Any]] = None  # a page's host stamp, a part's ``part_stamp``
    member_ids: Tuple[str, ...] = ()  # a part's members (a prefix of the unit's when the window is short)
    fitted_to: Optional[Dict[str, Any]] = None  # last row in a page input, including address-only rows


def _remembering(root: pathlib.Path) -> str:
    """The mind's own guidance for remembering (knowledge note ``remembering``, global), if it wrote one."""
    from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address

    try:
        return read_knowledge_note(resolve_knowledge_address(root, "remembering", "global")).text.strip()
    except (OSError, ValueError):
        return ""


def _indented(text: Any) -> str:
    return "\n".join(INDENT + line if line else line for line in str(text or "").split("\n"))


def _head(root: pathlib.Path, opening: str, quotes: str, *, answer: str = _ANSWER) -> str:
    guidance = _remembering(root)
    head = opening + _COMMON + quotes + answer
    if guidance:
        head += ("\n\n## Authored guidance from the mind (its knowledge note `remembering`)\n" + _indented(guidance))
    return head


def _room_label(root: pathlib.Path, room: str) -> str:
    try:
        from ouroboros.dialogue_provenance import RoomLabelResolver

        return f"{RoomLabelResolver(root).label({'chat_id': int(room)})} (room {room})"
    except Exception:
        return f"room {room}"


def _stamp_line(entry: Mapping[str, Any], lane_rows: int) -> str:
    """One task as the host recorded it: status, outcome, phase, the finality and authority its source
    recorded (a host facts stamp without one says unknown) and the source of the stamp."""
    facts = [f"status {entry.get('status') or 'not recorded'}"]
    for label, key in (("outcome", "outcome"), ("phase", "outcome_phase"), ("review", "review_verdict"),
                       ("reason", "reason_code"), ("objective", "objective_status")):
        if entry.get(key):
            facts.append(f"{label} {entry[key]}")
    finality = recorded_finality(entry.get("outcome_final"), entry.get("outcome_authority"),
                                 host_facts=entry.get("source") == "host_task_facts")
    if finality:
        facts.append(finality)
    facts.append(f"stamp from {entry.get('source') or 'nothing recorded'}")
    if lane_rows:
        facts.append(f"{lane_rows} host/child row{'' if lane_rows == 1 else 's'} here")
    return f"- task {entry.get('task_id')}: " + "; ".join(facts) + f"; get_task_result(task_id='{entry.get('task_id')}')"


@dataclass(frozen=True)
class _Row:
    entry: Tuple[Dict[str, Any], Dict[str, Any], int]
    task: str
    line: str  # its verbatim text line; "" for a task's host/child row (its task line stands for it)
    pointer: str  # its address line: only my own spoken words may become one (people's words never do)


def _rows(root: pathlib.Path, read_rows: List[Tuple[Dict[str, Any], Dict[str, Any]]],
          positions: Dict[str, int]) -> List[_Row]:
    lineage = row_lineage(root)  # one lineage lookup for the pass
    items = []
    for address, row in read_rows:
        pos = positions.get(address["row_sha256"])
        if pos is None:
            pos = chat_chain.stream_position_of(root, address)
        cls = row_class(row, pos=pos, **lineage)
        author = {"label": f"{cls['author'].get('kind')}: {cls['author'].get('label')}"}
        task = str(row.get("task_id") or "")
        spoken = cls["lane"] == 1
        line = render_memory_row(address, row, author=author, indent=INDENT) if spoken or not task else ""
        pointer = (f"{memory_row_header(address, row, author=author)} (words, {len(render_row_text(row))} chars — "
                   "read by address)") if spoken and cls["author"].get("kind") != "human" else ""
        items.append(_Row((address, row, pos), task, line, pointer))
    return items


def _owner_words(root: pathlib.Path, items: List[_Row]) -> Dict[str, Tuple[str, str, str]]:
    """``task -> (group, verbatim block, address block)``: the owner's words that caused each task."""
    from ouroboros.owner_words import render_owner_words, task_owner_words  # D15->D17 is lazy-only

    out: Dict[str, Tuple[str, str, str]] = {}
    for item in items:
        if not item.task or item.task in out:
            continue
        try:
            rows, absent = task_owner_words(root, item.task)
        except Exception:
            log.debug("fallback writer: owner words unreadable for %s", item.task, exc_info=True)
            continue
        group = json.dumps([rows, absent], ensure_ascii=False, sort_keys=True)
        root_task = str(item.entry[1].get("root_task_id") or item.task)
        block = render_owner_words(rows, absent, audience="writer", root_task_id=root_task)
        if not block:
            continue
        refs = [f"[{row.get('ts') or 'time not recorded'} · owner · {row.get('source')} of task {row.get('task_id')}"
                f" · {row.get('ref')}] ({len(row.get('text') or '')} chars — not shown: the input could not hold "
                "them)" for row in rows]
        pointer = "## Words of my human that caused this work (by reference)\n" + "\n".join(refs) if refs else block
        out[item.task] = (group, block, pointer)
    return out


def _compose(head: str, label: str, items: List[_Row], n: int, addressed: set, owners: Dict[str, Tuple[str, str, str]],
             owner_refs: bool, stamps: Dict[str, Dict[str, Any]], tail: str) -> str:
    kept = items[:n]
    lines = [item.pointer if index in addressed else item.line for index, item in enumerate(kept) if item.line]
    first, last = kept[0].entry, kept[-1].entry
    sections = [head, f"## The rows: {label}, {n} rows, {first[1].get('ts')} → {last[1].get('ts')} (oldest first)\n"
                + ("\n".join(lines) or "(no spoken row: every row here belongs to a task line below)")]
    tasks = list(dict.fromkeys(item.task for index, item in enumerate(kept) if item.task and index not in addressed))
    if tasks:
        lane = {task: sum(1 for item in kept if item.task == task and not item.line) for task in tasks}
        sections.append("## Tasks of these rows (the host's stamp is each task's outcome)\n" + "\n".join(
            _stamp_line(stamps.get(task) or {"task_id": task}, lane[task]) for task in tasks))
    groups = {}
    for task in tasks:
        if task in owners:
            groups.setdefault(owners[task][0], owners[task][2] if owner_refs else owners[task][1])
    sections.extend(groups.values())
    if tail:
        sections.append(tail)
    return "\n\n".join(sections)


def _floor(items: List[_Row], budget: Optional[int], measure: Callable[[str], int],
           compose: Callable[[int, set, bool], str]) -> Tuple[int, set, bool]:
    """``(prefix length, addressed indexes, owner words by reference)`` that fit the Light window.

    (1) My own spoken rows become address lines, longest first; (2) the unit shortens to
    the longest prefix by position that fits (the rest stays open); (3) last, the owner's
    words that caused the tasks go by reference. People's words are never addressed.
    """
    n, addressed, refs = len(items), set(), False
    prompt = compose(n, addressed, refs)
    if budget is None:
        return n, addressed, refs
    tokens = measure(prompt)
    if tokens <= budget:
        return n, addressed, refs
    ratio = tokens / max(len(prompt), 1)  # one measurement; the choices below go by characters

    def fits(size: int) -> bool:
        return size * ratio <= budget

    own = sorted((i for i, item in enumerate(items) if item.pointer), key=lambda i: -len(items[i].line))

    def addressing(k: int) -> set:  # (1) within a prefix of k rows
        chosen: set = set()
        for index in (i for i in own if i < k):
            if fits(len(compose(k, chosen, False))):
                break
            chosen.add(index)
        return chosen

    addressed = addressing(n)
    if not fits(len(compose(n, addressed, False))):
        low, high = 1, n  # (2) the longest prefix that can fit; more rows never compose shorter: bisect
        while low < high:
            mid = (low + high + 1) // 2
            low, high = (mid, high) if fits(len(compose(mid, set(own), False))) else (low, mid - 1)
        n = low
        addressed = addressing(n)
    if not fits(len(compose(n, addressed, False))):
        refs = True  # (3)
    while n > 1 and measure(compose(n, addressed, refs)) > budget:
        n -= 1  # the character ratio underestimated; the transport's preflight refuses what still does not fit
    return n, {index for index in addressed if index < n}, refs


def _trimmed(covers: Dict[str, Any], kept: List[_Row]) -> Dict[str, Any]:
    """The covers of the rows read: never a row given only by address, never a mind note.

    The helper's input holds no note of the mind, so its draft seals none; a note stays
    open and in the view until a page of the mind covers it.
    """
    shas = [item.entry[0]["row_sha256"] for item in kept]
    return {**covers, "rows": shas, "note_ids": [], "first": kept[0].entry[0], "last": kept[-1].entry[0],
            "count": len(kept), "task_ids": list(dict.fromkeys(item.task for item in kept if item.task)),
            "ts_span": source_time_span(item.entry[1].get("ts") for item in kept),
            "stream_span": [kept[0].entry[2], kept[-1].entry[2]]}


def _page_input(root: pathlib.Path, store: ChronicleStore, unit: FallbackUnit, budget: Optional[int],
                measure: Callable[[str], int]) -> Optional[WriterInput]:
    from ouroboros.tools.chronicle import host_stamp, page_covers

    segment = unit.segment
    try:
        read = page_covers(root, unit.room_id, from_addr=segment.from_addr, to_addr=segment.to_addr)
    except (LookupError, ValueError):
        return None
    positions = {entry[0]["row_sha256"]: entry[2] for entry in segment.rows}
    items = _rows(root, read["rows"], positions)
    owners = _owner_words(root, items)
    tail = ""
    if unit.record_id:
        retold = next((record for record in store.room_records(unit.room_id) if record["id"] == unit.record_id), {})
        if str(retold.get("current_text") or "").strip():
            tail = ("## Earlier helper retelling of this period, not a source, may be wrong\n"
                    + _indented(retold["current_text"]))
    head, label = _head(root, _PAGE, _QUOTES), _room_label(root, unit.room_id)
    full = host_stamp(root, read["covers"]["task_ids"], rows=read["rows"])
    stamps = {entry["task_id"]: entry for entry in full["tasks"]}
    n, addressed, refs = _floor(items, budget, measure,
                                lambda k, a, r: _compose(head, label, items, k, a, owners, r, stamps, tail))
    kept = [item for index, item in enumerate(items[:n]) if index not in addressed]
    if not kept:
        return None
    covers = read["covers"]
    if n < len(items):
        covers = page_covers(root, unit.room_id, from_addr=items[0].entry[0], to_addr=items[n - 1].entry[0])["covers"]
    covers = _trimmed(covers, kept)
    stamp = host_stamp(root, covers["task_ids"], rows=[(item.entry[0], item.entry[1]) for item in kept])
    prompt = _compose(head, label, items, n, addressed, owners, refs,
                      {entry["task_id"]: entry for entry in stamp["tasks"]}, tail)
    return WriterInput(prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                       measure(prompt) if budget is not None else None, covers=covers,
                       positions={item.entry[0]["row_sha256"]: item.entry[2] for item in items},
                       addressed_rows=tuple(chat_chain.format_address(items[i].entry[0]) for i in sorted(addressed)),
                       stamp=stamp, fitted_to=items[n - 1].entry[0])


def _part_input(root: pathlib.Path, store: ChronicleStore, unit: FallbackUnit, budget: Optional[int],
                measure: Callable[[str], int], records: Optional[Mapping[str, Any]] = None,
                units: Optional[Mapping[str, Any]] = None,
                cache: Optional[Dict[str, Any]] = None) -> Optional[WriterInput]:
    from ouroboros.terminal_projection import part_stamp  # D15->D17 is lazy-only

    if records is None:
        records = {record["id"]: record for record in store.room_records(unit.room_id)}
    members = [records.get(ident) for ident in unit.member_ids]
    if not members or any(member is None for member in members):
        return None
    head = _head(root, _PART, _MEMBER_QUOTES if any(member.get("quotes") for member in members) else _NO_QUOTES,
                 answer=_ANSWER + _KEEP_ORIGINAL)
    if units is None:
        units = {u.record_id: u for u in memory_inventory.legacy_units(store, root)}
    if cache is None:
        cache = {}
    # Scope is one room/head in one selection pass. Include rejected nested
    # records in period sources; the acting room view intentionally omits them.
    if "period_records" not in cache:
        cache["period_records"] = {r["id"]: r for r in store.records(unit.room_id)}
    rendered = cache.setdefault("blocks", {})
    label, blocks = _room_label(root, unit.room_id), []
    for member in members:
        if member["id"] in rendered:
            blocks.append(rendered[member["id"]])
            continue
        period = memory_inventory.record_period(store, member, units, records=cache["period_records"])
        span = period.span or {}
        dated = (f"{span.get('start') or '?'} → {span.get('end') or '?'}{period.note()}"
                 + ("; incomplete" if span.get("incomplete") else "")) if span else "unknown"
        author = member.get("author") or {}
        signer = f" ({draft_signer(author)})" if author.get("kind") == "helper" else ""
        blocks.append(f"### {member['kind']} {member['id']} — by {memory_inventory._author_words(author)}{signer}"
                      + f" — status {member.get('status') or 'not recorded'} — period {dated}"
                      + (" — prior helper account, retain its attributed meaning" if member["kind"] == "legacy" else "")
                      + "\n" + _indented(member.get("current_text"))
                      + "".join(f"\nquote ({q.get('speaker')}, {q.get('address')}): {q.get('text')}"
                                for q in member.get("quotes") or ()))
        rendered[member["id"]] = blocks[-1]

    def compose(k: int) -> Tuple[str, Dict[str, Any]]:
        stamp = part_stamp(member.get("host_stamp") for member in members[:k])
        counts = "; ".join(f"{c['tasks']} from {c['source'] or 'nothing recorded'}, phase {c['outcome_phase'] or 'not recorded'}"
                           for c in stamp["counts"])
        lines = [_stamp_line(entry, 0) for entry in stamp["tasks"]]
        tasks = ("## Host stamp of the members' tasks\n" + (counts or "no task stamped")
                 + ("\n" + "\n".join(lines) if lines else ""))
        return "\n\n".join([head, f"## Records to fold: {label}, oldest first", *blocks[:k], tasks]), stamp

    k = len(members)
    prompt, stamp = compose(k)
    # Later terminal stamps can remove earlier details: prefix sizes are not
    # monotonic. Keep the longest fitting prefix, reusing rendered source blocks.
    while budget is not None and k > 1 and measure(prompt) > budget:
        k -= 1
        prompt, stamp = compose(k)
    return WriterInput(prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                       measure(prompt) if budget is not None else None, stamp=stamp,
                       member_ids=tuple(unit.member_ids[:k]))


def build_writer_input(root: Any, store: ChronicleStore, unit: FallbackUnit, *, budget_tokens: Optional[int],
                       measure: Callable[[str], int] = _chars_measure,
                       part_records: Optional[Mapping[str, Any]] = None,
                       part_units: Optional[Mapping[str, Any]] = None,
                       part_cache: Optional[Dict[str, Any]] = None) -> Optional[WriterInput]:
    """The helper's input for one unit within the Light window, or ``None`` when nothing of it is readable."""
    root = pathlib.Path(root)
    if unit.kind == "part":
        return _part_input(root, store, unit, budget_tokens, measure, part_records, part_units, part_cache)
    return _page_input(root, store, unit, budget_tokens, measure)


# --- one call, one publication or one receipt ------------------------------------------------------

@dataclass
class FallbackRun:
    """What one stage run did: ``consciousness_on``, ``not_activated``, ``nothing``, ``busy``,
    ``published``, ``kept_original``, ``conflict``, ``refused`` (a receipt) or ``failed`` (no receipt; ``errors``)."""
    outcome: str
    unit: Optional[FallbackUnit] = None
    record_id: str = ""
    kind: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: Optional[int] = None
    effect: str = ""  # a published draft, measured: what it folded, its characters before and after


def _parse(text: str, *, part: bool = False) -> Optional[Dict[str, Any]]:
    _prose, found, _duplicate = extract_trailing_json_object(text)
    if part and isinstance(found, dict) and found.get("keep_original") is True:
        reason = found.get("reason")
        if isinstance(reason, str) and reason.strip() and not found.get("text") and not found.get("quotes"):
            return {"keep_original": True, "reason": reason.strip()}
        return None
    if not isinstance(found, dict) or not isinstance(found.get("text"), str) or not found["text"].strip():
        return None
    quotes = found.get("quotes") if found.get("quotes") is not None else []
    return {"text": found["text"].strip(), "quotes": quotes} if isinstance(quotes, list) else None


def _publish(root: pathlib.Path, store: ChronicleStore, unit: FallbackUnit, draft: WriterInput,
             answer: Dict[str, Any], usage: Dict[str, Any], binding: Dict[str, Any]) -> PublishResult:
    from ouroboros.knowledge import observed_route_stamp
    from ouroboros.tools.chronicle import _quote_resolver

    author = {"kind": "helper", "writer": "fallback_part" if unit.kind == "part" else "fallback_page",
              "route": observed_route_stamp(usage), "attribution": "helper draft, not lived"}
    metadata = {"unit": unit.describe(), "input_sha256": draft.input_sha256, "light_binding": binding,
                "addressed_rows": list(draft.addressed_rows)}
    resolver = _quote_resolver(root, row_lineage(root), draft.positions)
    if unit.kind == "part":
        return store.publish_part(room_id=unit.room_id, text=answer["text"], member_ids=list(draft.member_ids),
                                  author=author, expected_sequence=unit.head_sequence, quotes=answer["quotes"],
                                  metadata=metadata, host_stamp=draft.stamp, quote_resolver=resolver)
    return store.publish_page(room_id=unit.room_id, text=answer["text"], covers=draft.covers, author=author,
                              quotes=answer["quotes"], host_stamp=draft.stamp, metadata=metadata,
                              quote_resolver=resolver)


def _set_refusal(store: ChronicleStore, key: str, receipt: Optional[Dict[str, Any]]) -> None:
    """Append only this unit's scan delta under the writer lock; null clears an old receipt."""
    if receipt is None and key not in _refusals(store):
        return
    store.publish([], scan_state={memory_inventory.FALLBACK_REFUSAL_PREFIX + key: receipt})


def _refuse(root: pathlib.Path, store: ChronicleStore, run: FallbackRun, draft: WriterInput, kind: str,
            binding: Dict[str, Any], task_id: str, answer: str, *, outcome: str = "refused") -> FallbackRun:
    ref = None
    if answer:
        ref = chat_chain.retain_memory_source(SimpleNamespace(drive_root=root, task_id=task_id),
                                              "memory_fallback_response", answer.encode("utf-8"))
    _set_refusal(store, run.unit.key, {"input_sha256": draft.input_sha256, "light_binding": binding, "kind": kind,
                                       "at": utc_now_iso(), "task_id": task_id, "response_ref": ref})
    run.outcome, run.kind = outcome, kind
    return run


def run_fallback_draft(env: Any, task: Mapping[str, Any], llm: Any, drive_logs: Any, trace: Any) -> FallbackRun:
    """At most one Light call: draft the chosen unit and publish it, or record why it was refused."""
    if not fallback_active():
        return FallbackRun("consciousness_on")
    root = pathlib.Path(task.get("budget_drive_root") or pathlib.Path(drive_logs).parent)
    store = ChronicleStore(root)
    if not store.log_path.exists() or not store.activation():
        return FallbackRun("not_activated")  # the writer reads the chronicle, it never creates it
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    # The unit is chosen under the lock: a choice made outside it could pay again for an input
    # another root refused, or sealed, while this one was choosing.
    lock_path = root / "memory" / "chronicle" / ".fallback.lock"
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=0, owner_aware_stale=True)
    if fd is None:
        return FallbackRun("busy")
    try:
        binding, fitted = light_binding(), []

        def fit() -> Tuple[Optional[int], Callable[[str], int]]:
            if not fitted:
                fitted.append(_light_fit(binding))
            return fitted[0]

        shortage, allow_legacy = shortage_from_trace(trace), not task.get("_is_direct_chat")
        units = (itertools.chain(
            max_room_units(store), candidate_units(root, store, shortage, allow_legacy=allow_legacy))
            if crowded_start(trace) else None)
        chosen = select_unit(root, store, shortage, allow_legacy=allow_legacy, binding=binding, fit=fit, units=units)
        if chosen is None:
            return FallbackRun("nothing")
        unit, draft = chosen
        from ouroboros import consolidator

        text, usage = consolidator._call_consolidation_llm(llm, draft.prompt, LABEL, reasoning_effort="low")
        errors = [row for row in usage.get("_consolidation_errors") or [] if isinstance(row, dict)]
        run = FallbackRun("failed", unit, usage=usage, errors=errors, input_tokens=draft.input_tokens)
        phase = "binding_failed"
        try:
            bound, task_id = light_binding(), str(task.get("id") or task.get("task_id") or "")
            phase = "receipt_failed"
            if errors:
                run.kind = str(errors[-1].get("kind") or "")
                return _refuse(root, store, run, draft, run.kind, bound, task_id, "") if run.kind in RECEIPT_KINDS else run
            answer = _parse(text, part=unit.kind == "part")
            if answer is None:
                return _refuse(root, store, run, draft, "invalid", bound, task_id, text)
            if answer.get("keep_original"):
                return _refuse(root, store, run, draft, "keep_original", bound, task_id, text, outcome="kept_original")
            phase = "publish_failed"
            result = _publish(root, store, unit, draft, answer, usage, bound)
            if result.ok:
                run.outcome, run.record_id = "published", str((result.record or {}).get("id") or "")
                if unit.kind == "part":
                    from ouroboros.tools.chronicle import _fold_effect

                    run.effect = _fold_effect(store, "a helper draft folded", draft.member_ids, answer["text"])
                phase = "receipt_clear_failed"
                _set_refusal(store, unit.key, None)
            elif result.reason in CONFLICT_REASONS:
                run.outcome, run.kind = "conflict", result.reason
            else:
                phase = "receipt_failed"
                return _refuse(root, store, run, draft, result.reason, bound, task_id, text)
        except Exception as error:
            # The call already returned: keep usage and any published record ID.
            # Receipt trouble is not setup failure, nor proof publication failed.
            log.debug("fallback writer: %s", phase, exc_info=True)
            run.kind = phase
            run.errors = [*run.errors, {"kind": phase, "label": LABEL, "message": f"{type(error).__name__}: {error}"}]
        return run
    finally:
        release_exclusive_file_lock(lock_path, fd)
