"""One start allowance and an ordered walk over an immutable memory snapshot.

After the start, only a physical deficit narrows the selected forms. Owner rows
without carried commitments may spend the work room, but never silently disappear.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from functools import lru_cache
from typing import Callable, Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros import context_budget
from ouroboros.chat_chain import parse_address
from ouroboros.context_layout import INDEX_STEPS as INDEX_STEPS
from ouroboros import memory_view as mv
from ouroboros import memory_view_legacy as legacy
from ouroboros.utils import estimate_tokens
from ouroboros.section_floor import PROTECTED, retain_section, relocated_text, section_views

log = logging.getLogger(__name__)

LADDER = ("F1", "F1b", "F6r", "F6", "F6a", "F3", "F2", "F4", "F7p", "F7", "F7a", "F5", "F5a", "F8", "F8a")
MODES = ("max", "low", "nano")
_COLLAPSING = ("F1", "F1b", "F6a", "F7a", "F5a", "F8a")
WINDOW_ONLY = frozenset({'F8', 'F8a'})
_LIST_FORMS = {'F7a': ('F2', 'F7p', 'F7'), 'F5a': ('F5',), 'F8a': ('F8',)}
_SHOWN = (("F2", "{} of my replies"), ("F7", "{} owner rows"), ("F7p", "{} other spoken rows"),
          ("F6r", "{} other rooms as ranges"), ("F6", "{} other rooms as directory lines"),
          ("F6a", "{} other rooms in an aggregate"), ("F4", "{} room records"),
          ("F5", "{} pages, parts or accounts of my story"), ("F3", "{} retold rooms"),
          ("F1", "{} task fact lines"), ("F1b", "{} quiet rooms"))
_SHOWN += (("F7a", "{} aggregate row ranges"), ("F5a", "{} aggregate story lists"))
_SHOWN += (("F8", "{} unsealed notes by label"), ("F8a", "{} aggregate note lists"))
_PATH_CASES = ((), ("enable_tools",), ("list_available_tools",), ("chronicle_write", "list_available_tools"))
_SEALING_TOOLS = frozenset({"chronicle_write", "enable_tools"})  # sent, or loadable through enable_tools


def view_tokens(text: str) -> int:
    """The host's estimate of a text inside a request: its JSON-escaped characters / 4."""
    return estimate_tokens(json.dumps(str(text), ensure_ascii=False)) if text else 0


def floor_elements(snapshot: mv.MemoryViewSnapshot) -> List[Tuple[str, str, str, str]]:
    """Ordered addressable forms, oldest first. Core and governing words have no rung."""
    if not snapshot.active:
        return []
    room = snapshot.room or {}
    lane1, lane2 = room.get("lane1") or [], room.get("lane2") or []
    quiet = [r for r in snapshot.live_rooms if not r["notes"] and not r["words"] and (
        not r.get("directory") or not (r["facts"].get("people") or r["facts"].get("mine") or r.get("focus")))]
    others = [r for r in snapshot.live_rooms if r not in quiet]
    mine = sorted((i for i in lane1 if i["kind"] == "ouroboros"), key=lambda i: i["pos"])
    steps = {
        "F1": [(i["first"], i["line"], "" if n else mv._facts_line(room, [i])) for n, i in enumerate(lane2)],
        "F1b": [(r["room_id"], mv._directory_room(r) if r.get("directory") else "\n".join(mv._live_room(r)),
                  "" if n else mv._rooms_line([r])) for n, r in enumerate(quiet)],
        "F6r": [(r["room_id"], "\n".join(mv._live_room(r)), mv._range_room(r)) for r in others if not r.get("directory")],
        "F6": [(r["room_id"], mv._range_room(r), mv._directory_room(r)) for r in others if not r.get("directory")],
        "F6a": [(r["room_id"], mv._directory_room(r), "" if n or quiet else mv._rooms_line([r])) for n, r in enumerate(others)],
        "F3": [(rid, "\n".join(map(legacy.retold_record, group)), legacy.room_pointer(group))
               for rid, group in legacy.pointer_rooms(snapshot.story).items()],
        "F2": [(i["address"], i["line"], mv._row_pointer(i, "my reply")) for i in mine],
        "F4": [(i["id"], mv._retold(i), mv._retold(i, True)) for i in [*room.get("legacy", ()), *room.get("under_parts", ())]],
        "F7p": [(i["address"], i["line"], mv._row_pointer(i, "words")) for i in lane1
                if i["kind"] != "ouroboros" and not i.get("owner_side", i["kind"] == "human")],
        "F7": [(i["address"], i["line"], mv._row_pointer(i, "words")) for i in lane1
               if i.get("owner_side", i["kind"] == "human") and not i.get("immune")],
        "F5": [(e["id"], "\n".join(mv._page_lines(e)), mv._page_pointer(e))
               for e in snapshot.story if e.get("kind") != "legacy" and not e.get("told_by")],
        # These are forms of an already addressed set, never permission to cut another row/page.
        "F7a": [(str(room['room_id']), '', '')] if lane1 else [],
        "F5a": [('story', '', '')] if snapshot.story else [],
        "F8": [(n['id'], '\n'.join(mv._note_lines([n])), mv._note_pointer(n)) for n in room.get('notes', ())],
        "F8a": [(str(room['room_id']), '', '')] if room.get('notes') else [],
    }
    return [(step, ident, whole, short) for step in LADDER for ident, whole, short in steps[step]
            if step in _COLLAPSING or step == 'F8' or len(short) < len(whole)]


def degradable_elements(snapshot: mv.MemoryViewSnapshot) -> List[Tuple[str, str, int, int]]:
    """``(step, element id, tokens whole, tokens by address)`` in ladder order: each element's known saving."""
    return [(step, ident, view_tokens(whole), view_tokens(short))
            for step, ident, whole, short in floor_elements(snapshot)]


def floor_allowances(*, window_tokens: Optional[int], output_reserve_tokens: Optional[int], non_memory_tokens: int,
                     target_tokens: Optional[int] = None, calibration_ratio: float = 1.0,
                     work_room_levels: int = 1, boundary_tokens: Optional[int] = None) -> Dict[str, Optional[int]]:
    """One start bound for all unsolicited material; a physical bound for uncovered owner rows."""
    frame = dict(output_reserve_tokens=output_reserve_tokens, non_memory_tokens=non_memory_tokens,
                 calibration_ratio=calibration_ratio, margin_count=work_room_levels, boundary_tokens=boundary_tokens)
    physical = context_budget.request_context_budget(window_tokens=window_tokens, **frame)
    start = context_budget.request_context_budget(window_tokens=window_tokens, target_tokens=target_tokens, **frame)
    return {"margin": start["with_margin_tokens"], "physical": physical["without_margin_tokens"],
            "budget": start["with_margin_tokens"] if target_tokens else None}


def _memoized_measure(measure):
    """One walk's immutable forms; never cache across captures or request envelopes."""
    prices = {}
    def measured(level):
        if level.addressed not in prices:
            prices[level.addressed] = measure(level)
        return prices[level.addressed]
    return measured


def fit_memory_view(snapshot: mv.MemoryViewSnapshot, allowances: Mapping[str, Optional[int]], *,
                    kept: mv.FloorLevel = mv.FULL_VIEW, saving: int = 0,
                    measure: Optional[Callable[[mv.FloorLevel], int]] = None) -> mv.FloorLevel:
    """One ordered walk, then give back new cuts that fit. Previously chosen forms stay selected."""
    elements = degradable_elements(snapshot)
    old = {(step, ident) for step, ids in kept.addressed for ident in ids}
    taken = set(old)
    measure = _memoized_measure(measure or (
        lambda level: view_tokens(mv.render_story(snapshot, level)) + view_tokens(mv.render_room(snapshot, level))))
    current = measure(kept)
    bounds = [b for b in (allowances.get("margin"), allowances.get("budget")) if b is not None]
    bound = min(bounds) if bounds else None
    target = min(bound if bound is not None else current, current - saving) if saving else bound
    budget = allowances.get("budget") is not None and allowances.get("budget") == bound

    def level(selected):
        selected = {(s, i) for s, i in selected if s not in _LIST_FORMS or any(
            prior in _LIST_FORMS[s] for prior, _ in selected)}
        return mv.FloorLevel(tuple((s, tuple(i for step, i, *_ in elements if step == s and (s, i) in selected))
                                  for s in LADDER if any(step == s for step, _ in selected)),
                             by_budget=kept.by_budget + (sum(s not in WINDOW_ONLY for s, _ in selected - old) if budget else 0))

    uncovered = uncovered_owner_rows(snapshot, elements)
    last = {(step, ident) for step, ident, *_ in uncovered}
    added = []
    for step, ident, whole, short in elements:
        if target is not None and current <= target:
            current = measure(level(taken))  # aggregate boundaries and the receipt are part of the budget
            if current <= target:
                break
        if (step, ident) in taken or target is None or current <= target or (step, ident) in last or step in WINDOW_ONLY:
            continue
        if step in _LIST_FORMS:
            if not any(s in _LIST_FORMS[step] for s, _ in taken):
                continue
            before, after = measure(level(taken)), measure(level(taken | {(step, ident)}))
            if after >= before:
                continue
            current, delta = after, before - after
            taken.add((step, ident))
            added.append((step, ident, delta))
            continue
        delta = max(0, whole - short)
        taken.add((step, ident))
        added.append((step, ident, delta))
        current -= delta
    current = measure(level(taken))
    # A coarse final cut can leave space for an earlier element; give it back without unfolding kept choices.
    for step, ident, delta in reversed(added):
        if target is not None and current + delta <= target:
            trial = taken - {(step, ident)}
            actual = measure(level(trial))
            if actual <= target:
                taken, current = trial, actual
    # Uncovered owner rows follow the carried/story rungs, oldest first, only under physical pressure.
    physical = allowances.get("physical")
    for step, ident, whole, short in uncovered:
        if physical is None or current <= physical:
            break
        if (step, ident) not in taken:
            taken.add((step, ident))
            current -= max(0, whole - short)
            if current <= physical:
                current = measure(level(taken))
            # Uncovered words still go oldest first after every carried/story rung.
            # Their new addresses can use the same range form, without taking another row.
            if current > physical:
                form = ('F7a', str((snapshot.room or {})['room_id']))
                current = measure(level(taken))
                if form not in taken:
                    after = measure(level(taken | {form}))
                    if after < current:
                        taken.add(form)
                        current = after
    # Held notes yield only after all earlier forms, never to a start target or requested saving.
    if physical is not None:
        current = measure(level(taken))
    for step, ident, _whole, _short in elements:
        if physical is None or current <= physical:
            break
        if step not in WINDOW_ONLY or (step, ident) in taken:
            continue
        trial = taken | {(step, ident)}
        if step == 'F8a':
            # A tiny note's label plus reader can cost more than its body; the combined
            # list can still save space after all individual label forms were tried.
            trial.update((s, i) for s, i, *_ in elements if s == 'F8')
        after = measure(level(trial))
        if after < current:
            taken, current = trial, after
    return level(taken)


def uncovered_owner_rows(snapshot: mv.MemoryViewSnapshot, elements=None) -> List[Tuple[str, str, int, int]]:
    """Owner rows no page or note carries yet, oldest first; governing and unanswered rows have no address."""
    owner = {i["address"]: i for i in (snapshot.room or {}).get("lane1", ())}
    return [e for e in (degradable_elements(snapshot) if elements is None else elements)
            if e[0] == "F7" and not owner[e[1]].get("carried")]


def kept_level(snapshot: mv.MemoryViewSnapshot, facts: Mapping[str, Any]) -> mv.FloorLevel:
    """The level a projection's floor fact names: each step's first ``count`` elements, in ladder order."""
    floor = (facts or {}).get("floor") or {}
    if "addressed" in floor:
        return mv.FloorLevel(tuple((step, tuple(ids)) for step, ids in floor["addressed"]),
                             by_budget=int(floor.get("by_budget") or 0))
    counts = dict(floor.get("steps") or {})
    ids: Dict[str, List[str]] = {}
    for step, ident, _whole, _short in floor_elements(snapshot):
        if len(ids.setdefault(step, [])) < int(counts.get(step) or 0):
            ids[step].append(ident)
    return mv.FloorLevel(tuple((step, tuple(ids[step])) for step in LADDER if ids.get(step)),
                         by_budget=int(floor.get("by_budget") or 0))


def level_with(snapshot: mv.MemoryViewSnapshot, kept: mv.FloorLevel, addresses: Iterable[str], *,
               owner_note: bool = False) -> mv.FloorLevel:
    """Select exact inspected forms; a note's :whole address restores its captured source."""
    eligible = {f"{step}:{ident}": (step, ident) for step, ident, *_ in floor_elements(snapshot)}
    taken = {(step, ident) for step, ids in kept.addressed for ident in ids}
    for address in addresses:
        if address.endswith(':whole') and address[:-6] in eligible and eligible[address[:-6]][0] == 'F8':
            taken.discard(eligible[address[:-6]])
            if not any(s == 'F8' for s, _ in taken):
                taken = {(s, i) for s, i in taken if s != 'F8a'}
            continue
        if address not in eligible:
            raise ValueError(f"Unknown or protected view_address: {address}")
        step, ident = eligible[address]
        if step in _LIST_FORMS:
            if not any(s in _LIST_FORMS[step] for s, _ in taken):
                raise ValueError('Aggregate forms require already addressed rows, pages or notes')
            if step == 'F7a' and not owner_note and any(
                    ('F7', row['address']) in taken and not row.get('carried') for row in snapshot.room['lane1']):
                raise ValueError('Uncovered owner commitments require an authored working_note before changing their form')
        if step == "F7" and not owner_note and not next(
                row.get("carried") for row in snapshot.room["lane1"] if row["address"] == ident):
            raise ValueError("Uncovered owner commitments require an authored working_note before changing their form")
        taken.add((step, ident))
        if step == 'F8':
            taken = {(s, i) for s, i in taken if s != 'F8a'}  # an explicit label selection unfolds the note list
        if step in ("F6", "F6a"):
            taken.update((prior, ident) for prior in ("F6r", "F6") if f"{prior}:{ident}" in eligible)
    return mv.FloorLevel(tuple((step, tuple(ident for s, ident, *_ in floor_elements(snapshot)
                                           if s == step and (s, ident) in taken)) for step in LADDER
                               if any(s == step for s, _ in taken)), by_budget=kept.by_budget)


def select_view(plan, mode: str, addresses: Iterable[str], *, owner_note: bool = False):
    """Render selected forms, preserving other projections and the tail.

    The boundary's fit callback decides publication. Ordinary Main accepts an
    authored restore with an advisory capacity deficit; dispatch fits it later.
    """
    from ouroboros.context_fit import _governance_blocks, _system_blocks, _request_tokens

    if plan is None or getattr(plan, "core", None) is None or not plan.core.memory_view_json:
        raise ValueError("This actor has no captured memory view")
    snapshot = mv.snapshot_from_json(plan.core.memory_view_json)
    shown = plan.projection(mode)
    level = level_with(snapshot, kept_level(snapshot, shown.memory_facts), addresses, owner_note=owner_note)
    old = shown.memory_facts.get("floor") or {}
    switched = old.get("mode_switch") or {}
    note = floor_note(level, window_tokens=old.get("window_tokens"), mode=mode,
                      target_tokens=old.get("target_tokens"), budget_mode=plan.preferred_mode,
                      lowered_from=switched.get("from"))
    story, room = mv.render_story(snapshot, level), mv.render_room(snapshot, level, floor_note=note)
    facts = view_facts(snapshot, level, window_tokens=old.get("window_tokens"), mode=mode,
                       allowances={"margin": old.get("allowance_tokens"), "physical": old.get("physical_allowance_tokens"),
                                   "budget": old.get("budget_allowance_tokens")},
                       target_tokens=old.get("target_tokens"), lowered_from=switched.get("from"))
    governance = _governance_blocks(None, plan.core, mode="low" if mode == "nano" else mode)
    sections = (*plan.core.semi_stable_sections, *plan.core.dynamic_head_sections, *plan.core.dynamic_sections,
                *(governance[0] if not isinstance(governance[0], str) else ()))
    receipts = shown.memory_facts.get("host_sections_addressed", ())
    if receipts:
        facts["host_sections_addressed"] = receipts
    system = _system_blocks(plan.core, governance, story, room, mode, section_views(sections, receipts))
    tokens = _request_tokens(system, shown.user_content_json or plan.user_content_json,
                            current_messages=plan.messages_for(mode))
    projection = dataclasses.replace(shown, system_content_json=json.dumps(system, ensure_ascii=False, sort_keys=True),
        memory_facts=view_receipt(snapshot, story, room, facts), estimated_tokens=tokens,
        calibrated_tokens=int(tokens * shown.calibration_ratio))
    return dataclasses.replace(plan, **{f"{mode}_projection": projection})


def owner_note_capsule(plan, addresses, note, request, *, drive_root, task_id, mode="max"):
    """Keep the mind's account of selected owner commitments with their exact captured source."""
    from ouroboros import context_compaction as cc
    from ouroboros.context_budget import _Selection

    wanted = {address[3:] for address in addresses if address.startswith("F7:")}
    if any(address.startswith("F7a:") for address in addresses):
        wanted.update(dict(kept_level(mv.snapshot_from_json(plan.core.memory_view_json),
                                     plan.projection(mode).memory_facts).addressed).get("F7", ()))
    if not wanted:
        return None
    snap = mv.snapshot_from_json(plan.core.memory_view_json)
    rows = [{"role": "user", "content": row["line"]} for row in snap.room["lane1"]
            if row["address"] in wanted and not row.get("carried")]
    if not rows:
        return None
    if not note.strip():
        raise ValueError("Uncovered owner commitments require an authored working_note")
    request = dataclasses.replace(request, transcript_sha256=cc.context_reclaim_transcript_sha256(rows))
    unit = cc._unit_from_slice(rows, 0, len(rows) - 1, trace_refs_by_tool_call_id={}, measurement_density=1.0)
    selection = _Selection((unit,), request.transcript_sha256, unit.predicted_reclaim_tokens)
    source = cc._persist_reclaim_checkpoint(rows, request, selection, drive_root=drive_root, task_id=task_id)
    if not source:
        raise ValueError("The exact owner-word source could not be retained")
    return cc._capsule_message(unit, note, [cc._part(unit.unit_id, unit.source_text)], source, request)[0]


def view_inventory(plan, mode: str, *, memory_facts=None) -> List[Dict[str, Any]]:
    if plan is None or getattr(plan, "core", None) is None or not plan.core.memory_view_json:
        return []
    snapshot = mv.snapshot_from_json(plan.core.memory_view_json)
    level = kept_level(snapshot, plan.projection(mode).memory_facts if memory_facts is None else memory_facts)
    chosen = {(step, ident) for step, ids in level.addressed for ident in ids}
    return [{"view_address": f"{step}:{ident}", "kind": "memory", "tokens": whole,
             "saving_tokens": max(0, whole - short), "eligible": True, "selected": (step, ident) in chosen,
             **({"restore_view_address": f"F8:{ident}:whole", "form":
                 'whole' if (step, ident) not in chosen else 'aggregate' if dict(level.addressed).get('F8a') else 'line'}
                if step == 'F8' else {})}
            for step, ident, whole, short in degradable_elements(snapshot)]


def memory_read_path(tool_names: Optional[Iterable[str]]) -> str:
    """How this request reaches ``memory_read``, read from the names of the schemas it sends.

    Nothing when it sends ``memory_read`` or its schemas are not known (``None``: no claim);
    ``enable_tools`` when it sends that; otherwise the parent, who can read an address for me,
    and ``list_available_tools`` when it sends that. A fact of the request, never of a role.
    """
    if tool_names is None:
        return ""
    names = set(tool_names)
    if "memory_read" in names:
        return ""
    if "enable_tools" in names:
        return "(memory_read is reachable through enable_tools)"
    listing = "list_available_tools shows what this task can call, and " if "list_available_tools" in names else ""
    return f"(memory_read is not among this request's tools; {listing}my parent task can read any address I name to it)"


def floor_note(level: mv.FloorLevel, *, window_tokens: Optional[int], mode: str, target_tokens: Optional[int] = None,
               lowered_from: Optional[str] = None, tool_names: Optional[Iterable[str]] = None,
               budget_mode: Optional[str] = None) -> str:
    """``### Physical floor``: a fact and a possibility for the mind, exactly when a step past F1/F1b ran
    or the window lowered the task's mode; never an instruction or a threshold. In Nano it
    closes with how this request reaches ``memory_read`` (``tool_names``: the schemas it sends). The
    sealing possibility is named only when the request can call ``chronicle_write``: it sends it or
    ``enable_tools`` (unknown schemas keep it); like the path line, a fact of the request, not a role.
    Cuts are retained history, not a claim that the current mode cannot hold the full memory. The
    window names the cut's frame; a fitting route keeps this note's bytes, and the round's facts
    name the acting window. The mode names this view and the mode the task runs in, lowered at its
    start or on a later route; ``budget_mode`` (default ``mode``) names the owner's target that took
    the budget cuts."""
    names = None if tool_names is None else tuple(tool_names)
    counts = dict(level.steps)
    asked = set(counts) - set(_COLLAPSING)
    if not asked and not lowered_from:
        return ""
    lines, name = ["### Physical floor"], mode.capitalize()
    if asked:
        bounds = ([f"the window ({window_tokens} tokens)"]
                  if window_tokens is not None else []) + (
            [f"the {(budget_mode or mode).capitalize()} mode budget ({target_tokens} tokens)"] if level.by_budget else [])
        subject = " and ".join(bounds)
        shown = ", ".join(text.format(counts[step]) for step, text in _SHOWN if counts.get(step))
        sealing = (" Sealing a closed part of the open conversation as a page (chronicle_write kind=page) or folding "
                   "old pages (kind=part) can carry the commitments in my own words."
                   if names is None or _SEALING_TOOLS & set(names) else "")
        lines.append(f"This {name} view retains memory cuts for {subject}. "
                     f"Shown above only by address: {shown}. The retained sources remain readable: memory_read reads "
                     f"each.{sealing}")
    if lowered_from:
        lines.append("Low: the books did not fit beside the story and the work room." if mode == "low" else
            f"The window cannot hold {lowered_from.capitalize()} with even the shortest view of my memory; this task runs in {name}.")
    path = memory_read_path(names) if mode == "nano" else ""
    return "\n".join(lines + [path] if path else lines)


def view_facts(snapshot: mv.MemoryViewSnapshot, level: mv.FloorLevel, *, window_tokens: Optional[int], mode: str,
               allowances: Mapping[str, Optional[int]], target_tokens: Optional[int] = None,
               lowered_from: Optional[str] = None) -> Dict[str, Any]:
    """The floor fact of one projection, kept in the task trace: steps, boundaries, what is shown by address.

    ``newest_addressed_row`` is the newest row of this room that F1, F2 or F7 shows only by
    address; ``pointer_records`` the records F4 and F5 show by a pointer. No list of rows.
    """
    gone = {step: set(ids) for step, ids in level.addressed}
    room = snapshot.room or {}
    rows = [(item["last_pos"], item["last"]) for item in room.get("lane2") or () if item["first"] in gone.get("F1", ())]
    addressed = gone.get("F2", set()) | gone.get("F7", set()) | gone.get("F7p", set())  # once, not per lane-1 row
    rows += [(item["pos"], item["address"]) for item in room.get("lane1") or () if item["address"] in addressed]
    status = snapshot.legacy_blocks or {}
    return {"role": snapshot.spec.role, "room_id": snapshot.spec.room_id,
            "floor": {"steps": dict(level.steps), "addressed": level.addressed, "window_tokens": window_tokens, "mode": mode,
                      "allowance_tokens": allowances.get("margin"), "physical_allowance_tokens": allowances.get("physical"),
                      "target_tokens": target_tokens, "budget_allowance_tokens": allowances.get("budget"),
                      "by_budget": level.by_budget, "mode_switch": {"from": lowered_from, "to": mode} if lowered_from else None,
                      "newest_addressed_row": parse_address(max(rows)[1]) if rows else None,
                      "pointer_records": [ident for step, ids in level.addressed if step in ("F4", "F5") for ident in ids]},
            "story_status": {"folded": status.get("folded", 0), "total": status.get("total", 0)}}


def render_view_for_mode(snapshot: mv.MemoryViewSnapshot, *, mode: str, owner_mode: str, window_tokens: Optional[int],
                         known_window: bool, output_reserve: Optional[int], ratio: float, non_memory_tokens: int,
                         lowered_from: Optional[str] = None, tool_names: Optional[Iterable[str]] = None,
                         kept: Optional[mv.FloorLevel] = None, saving: int = 0, boundary_tokens: Optional[int] = None,
                         working_reserve_tokens: Optional[int] = None,
                         measure_text: Optional[Callable[[str, str], int]] = None) -> Tuple[str, str, Dict[str, Any]]:
    """``(story text, room text, facts)`` of one mode's projection, the floor decided once.

    The window counts only when known and fresh (``known_window``). An owner target binds
    only the mode the owner selected: task-local Low and a mode the window lowered have
    none. The reply reserve is the mode's (Nano keeps its own headroom). ``tool_names`` are
    the schemas this mode's request sends, which the floor note reads (``memory_read_path``).
    ``kept`` narrows a view already shown: from its level, against the window minus the reply
    reserve alone -- no working margin, no target -- so nothing it addressed comes back.
    What ``kept.by_budget`` counts the owner's mode target took (only it carries one), so the note
    and the fact name that budget in whatever mode the view is narrowed, hop after hop.
    """
    target, reserve = context_budget.context_mode_limits(mode, owner_mode, output_reserve, working_reserve_tokens=working_reserve_tokens)
    if kept is not None:
        target = context_budget.context_mode_limits(owner_mode, owner_mode, output_reserve)[0] if kept.by_budget else None
    window = int(window_tokens) if known_window and window_tokens else None
    allowances = floor_allowances(window_tokens=window, output_reserve_tokens=reserve, non_memory_tokens=non_memory_tokens,
                                  target_tokens=None if kept is not None else target, calibration_ratio=ratio,
                                  work_room_levels=snapshot.spec.work_room_levels, boundary_tokens=boundary_tokens)
    if kept is not None:
        allowances = {**allowances, "margin": allowances["physical"]}
    size = lambda shown: view_tokens(mv.render_story(snapshot, shown)) + view_tokens(mv.render_room(snapshot, shown))
    uncovered = uncovered_owner_rows(snapshot)
    @lru_cache(maxsize=128)
    def receipt(level):
        note = floor_note(level, window_tokens=window, mode=mode, target_tokens=target, lowered_from=lowered_from,
                          tool_names=tool_names, budget_mode=owner_mode)
        if allowances["margin"] is not None and size(level) > allowances["margin"]:
            gone = dict(level.addressed)
            held = [f"F7:{ident}" for step, ident, *_ in uncovered if ident not in gone.get(step, ())]
            owners = held and size(level_with(snapshot, level, held, owner_note=True)) <= allowances["margin"]
            notes = (snapshot.room or {}).get('notes', ())
            held_notes = notes and (not gone.get('F8a') or any(n['id'] not in gone.get('F8', ()) for n in notes))
            note += "\nStart room is smaller than requested: " + (
                "uncovered owner words remain in view; only physical pressure can turn their oldest rows into named addresses."
                if owners else "unsealed notes remain in their selected forms; only physical pressure can narrow them."
                if held_notes else "what my memory shows even in its shortest forms exceeds it.")
        return note
    # The request owner can measure JSON separators and rounding in the actual envelope.
    measure_text = measure_text or (lambda story, room: view_tokens(story) + view_tokens(room))
    measure = lambda level: measure_text(mv.render_story(snapshot, level),
                                       mv.render_room(snapshot, level, floor_note=receipt(level)))
    level = fit_memory_view(snapshot, allowances, kept=kept or mv.FULL_VIEW, saving=saving, measure=measure)
    note = receipt(level)
    return (mv.render_story(snapshot, level), mv.render_room(snapshot, level, floor_note=note),
            view_facts(snapshot, level, window_tokens=window, mode=mode, allowances=allowances, target_tokens=target,
                       lowered_from=lowered_from))


TRACE_FIELDS = ("role", "room_id", "floor", "story_status", "host_sections_addressed")  # what a task trace keeps of a view receipt


def trace_facts(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """The part of a view receipt the task context and the task trace keep (``VIEW_TRACE_KEY``)."""
    return {key: receipt[key] for key in TRACE_FIELDS if key in receipt}


def view_receipt(snapshot: mv.MemoryViewSnapshot, story: str, room: str, facts: Mapping[str, Any]) -> Dict[str, Any]:
    """A projection's view fact: the floor fact plus the spec, the chronicle's state and block sizes.

    ``role``, ``room_id``, ``floor`` and ``story_status`` are what a task trace keeps
    (``memory_inventory.VIEW_TRACE_KEY``); the rest is the receipt's (estimator tokens).
    """
    return {**facts, "spec": dataclasses.asdict(snapshot.spec), "store_status": dict(snapshot.store_status),
            "legacy_frontier_status": snapshot.frontier.get("status"),
            "blocks": {"story_tokens": view_tokens(story), "room_tokens": view_tokens(room),
                       "live_rooms": len(snapshot.live_rooms), "marks": len(snapshot.marks)}}


def minimal_view_tokens(snapshot: mv.MemoryViewSnapshot, *, window_tokens: Optional[int] = None) -> int:
    """The shortest view of my memory, with the longest floor note for its selected forms."""
    taken: Dict[str, List[str]] = {}
    for step, ident, _whole, _short in floor_elements(snapshot):  # grouped in ladder order
        if step in _LIST_FORMS and not any(taken.get(prior) for prior in _LIST_FORMS[step]):
            continue
        taken.setdefault(step, []).append(ident)
    candidates = [taken]
    if taken.get('F8'):
        whole_notes = {s: ids for s, ids in taken.items() if s not in WINDOW_ONLY}
        lines = [i for s, i, whole, short in degradable_elements(snapshot) if s == 'F8' and short < whole]
        candidates += [whole_notes, {**whole_notes, 'F8': lines}]
    def price(selected):
        level = mv.FloorLevel(tuple((step, tuple(ids)) for step, ids in selected.items() if ids))
        note = max((floor_note(level, window_tokens=window_tokens, mode="nano", lowered_from="max", tool_names=names)
                    for names in _PATH_CASES), key=view_tokens)  # an upper bound, whatever schemas it sends
        return view_tokens(mv.render_story(snapshot, level)) + view_tokens(mv.render_room(snapshot, level, floor_note=note))
    # Optional list forms must earn their space, just as in the walk. Price
    # every combination (at most eight) with its actual floor receipt; their
    # interactions must not make an available pointer-only view impossible.
    alternatives = []
    for selected in candidates:
        forms = [s for s in _LIST_FORMS if selected.get(s)]
        for mask in range(1 << len(forms)):
            omitted = {s for index, s in enumerate(forms) if mask & (1 << index)}
            alternatives.append({s: ids for s, ids in selected.items() if s not in omitted})
    return min(map(price, alternatives))


# Host producers before indexes, then the memory walk, all under the same start budget.
# Running views narrow only under physical pressure or a refusal.
HOST_STEPS = ("runtime", "registry", "installed_skills",
              "drive_state", "pattern_register", "world", "subagents",
              "execution_reflections", "architecture_map", "development_map")


def _host_steps(sections, previous, *, measure, window, reserve, ratio, saving, root, task_id, allowance=None):
    """First steps of mode_views, using the same physical allowance as memory.

    Producers own meaning; this walk selects only explicit smaller views. Retained
    source handles precede pointers. A failed capture leaves the full bytes inline.
    Previously shown addresses stay addressed; their savings never pay a new deficit.
    """
    if measure is None or not window or not root or not task_id:
        return list(previous), saving
    receipts = list(previous)
    addressed = section_views(sections, receipts)
    candidates = [part for part in sections if part.section_id not in addressed
                  and part.section_id not in PROTECTED and part.operative_view is not None
                  and len(part.operative_view) < len(part.full_text)]
    if not candidates:
        return receipts, saving
    initial = current = measure(addressed)
    physical = floor_allowances(window_tokens=window, output_reserve_tokens=reserve,
                                non_memory_tokens=0, calibration_ratio=ratio)["physical"]
    bound = allowance if allowance is not None else physical
    if (bound is None or current <= bound) and saving == 0:
        return receipts, 0
    order = {name: i for i, name in enumerate((*HOST_STEPS, *INDEX_STEPS))}
    for part in sorted(candidates, key=lambda p: (p.section_id in INDEX_STEPS,
                                               order.get(p.section_id, len(HOST_STEPS)))):
        if part.section_id in addressed:
            continue
        if (bound is None or current <= bound) and initial - current >= saving:
            break
        if part.section_id in INDEX_STEPS and current <= physical and initial - current >= saving:
            continue  # the knowledge outline remains a physical step, not a start-target choice
        try:
            row = retain_section(part, root=root, task_id=task_id)
        except (OSError, ValueError):
            log.warning("Host section capture unavailable; keeping %s whole", part.section_id, exc_info=True)
            continue
        short = relocated_text(part, row["reader"])
        saved = view_tokens(part.full_text) - view_tokens(short)
        if saved <= 0:
            continue
        addressed[part.section_id] = short
        receipts.append({**row, "tokens_saved": saved})
        current = measure(addressed)
    return receipts, max(0, saving - (initial - current))
