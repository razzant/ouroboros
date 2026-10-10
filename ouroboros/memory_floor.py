"""The physical floor of the memory view: what of my memory a request shows only by address.

The view's structure decides what it holds; the floor only takes away, and only when that
structure does not fit. ``fit_memory_view`` walks the ladder once, element by element,
while the view is larger than the boundary of the element's step, and every element it
takes is drawn as an address line with its period: the horizon stays, the granularity
changes. No search, no second render to choose a level, nothing added to fill room.

Three boundaries, each in estimator tokens of memory (``context_budget.request_context_budget``):
host facts, room headers, retold pointers and pages answer to the window minus the reply
reserve and the working margins (``MEMORY_VIEW_WORKING_MARGINS``); my own replies and
people's words only to the window minus the reply reserve, so a margin never costs the
conversation itself; an owner-selected Low or Nano target, with its own margins
(``MODE_TARGET_WORKING_MARGINS``), bounds the steps of ``MODE_TARGET_STEPS`` alone: a chosen
target trims facts, headers and retold memory, never the conversation. An unknown window
takes no window step.
``render_view_for_mode`` is the whole floor decision of one mode's projection (story text,
room text and the floor fact the task trace keeps).

``physical_mode`` lowers a task's starting mode (Max, Low, Nano) only when the fixed part
of the preferred mode with the shortest view of my memory cannot fit the window minus the
reply reserve on the route's calibrated estimate (estimator tokens scaled by the route's measured
ratio): all of my memory is already addresses and the request still cannot be sent.
``mode_views`` renders every mode of one plan and picks that starting mode (a new route
re-renders it from the same snapshot). Rendering stays ``memory_view``'s; nothing here reads
the chronicle, publishes a record or calls a model.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros import context_budget
from ouroboros.chat_chain import parse_address
from ouroboros import memory_view as mv
from ouroboros import memory_view_legacy as legacy
from ouroboros.utils import estimate_tokens

# The ladder, old before new and people last: host fact lines of this room's tasks
# (F1), other live rooms without notes (F1b), retold records, whole or not, as one line per room
# (F3), my oldest pages, parts and accounts (F5; a record an account tells stays its address line), this
# room's retold page (F4), my longest replies (F2), then people's words: other rooms' (F6, only when the
# view shows them) and this room's (F7).
LADDER = ("F1", "F1b", "F3", "F5", "F4", "F2", "F6", "F7")
# The only steps an owner-selected Low or Nano target takes: facts, headers,
# pointers and old retold memory; my replies and people's words answer to the window alone.
MODE_TARGET_STEPS = ("F1", "F1b", "F3", "F5", "F4")
# Both sides of the conversation, my replies and people's words, become addresses only when the
# window minus the reply reserve cannot hold them: never for the working margins.
WINDOW_ONLY_STEPS = ("F2", "F6", "F7")
MODES = ("max", "low", "nano")  # the starting-mode order the window may lower through
_COLLAPSING = ("F1", "F1b")  # many elements, one line: the first element carries it
_SHOWN = (("F2", "{} of my replies"), ("F7", "{} lines of people in this room"),
          ("F6", "{} lines of people in other rooms"), ("F4", "{} records of this room's page"),
          ("F5", "{} pages, parts or accounts of my story"), ("F3", "the retold records of {} rooms (one line per room)"),
          ("F1", "{} task fact lines"), ("F1b", "{} other open rooms"))
# One name set per closing ``floor_note`` can print (``memory_read_path``, with or without the
# chronicle_write sentence): the shortest view takes the longest.
_PATH_CASES = ((), ("enable_tools",), ("list_available_tools",), ("chronicle_write", "list_available_tools"))
_SEALING_TOOLS = frozenset({"chronicle_write", "enable_tools"})  # sent, or loadable through enable_tools


def view_tokens(text: str) -> int:
    """The host's estimate of a text inside a request: its JSON-escaped characters / 4."""
    return estimate_tokens(json.dumps(str(text), ensure_ascii=False)) if text else 0


def floor_elements(snapshot: mv.MemoryViewSnapshot) -> List[Tuple[str, str, str, str]]:
    """``(step, element id, whole text, address line)`` of what the floor may address, in ladder order.

    Inside a step oldest first; my replies longest first, then oldest. Where a step folds
    many elements into one line (F1, F1b) the first carries that line. An element whose
    address line is not shorter than its text is not degradable. Never here: identity,
    knowledge, marks, notes, the words that started the work, a helper's owner words, this
    room's header, a retold room's one line, the F1b summary.
    """
    if not snapshot.active:
        return []
    room = snapshot.room or {}
    lane1, lane2 = room.get("lane1") or [], room.get("lane2") or []
    quiet = [item for item in snapshot.live_rooms if not item["notes"] and not item["words"]]
    mine = sorted((item for item in lane1 if item["kind"] == "ouroboros"), key=lambda item: (-item["chars"], item["pos"]))
    words = sorted((word for item in snapshot.live_rooms for word in item["words"]), key=lambda word: word["pos"])
    steps = {
        "F1": [(item["first"], item["line"], "" if i else mv._facts_line(room, [item])) for i, item in enumerate(lane2)],
        "F1b": [(item["room_id"], "\n".join(mv._live_room(item)), "" if i else mv._rooms_line([item]))
                for i, item in enumerate(quiet)],
        "F3": [(room_id, "\n".join(map(legacy.retold_record, group)), legacy.room_pointer(group))
               for room_id, group in legacy.pointer_rooms(snapshot.story).items()],
        "F5": [(entry["id"], "\n".join(mv._page_lines(entry)), mv._page_pointer(entry))
               for entry in snapshot.story if entry.get("kind") != "legacy" and not entry.get("told_by")],
        "F4": [(item["id"], mv._retold(item), mv._retold(item, True))
               for item in [*room.get("legacy", ()), *room.get("under_parts", ())]],
        "F2": [(item["address"], item["line"], mv._row_pointer(item, "my reply")) for item in mine],
        "F6": [(word["address"], word["line"], mv._row_pointer(word, "words")) for word in words],
        "F7": [(item["address"], item["line"], mv._row_pointer(item, "words"))
               for item in lane1 if item["kind"] != "ouroboros"],
    }
    return [(step, ident, whole, short) for step in LADDER for ident, whole, short in steps[step]
            if step in _COLLAPSING or len(short) < len(whole)]


def degradable_elements(snapshot: mv.MemoryViewSnapshot) -> List[Tuple[str, str, int, int]]:
    """``(step, element id, tokens whole, tokens by address)`` in ladder order: each element's known saving."""
    return [(step, ident, view_tokens(whole), view_tokens(short))
            for step, ident, whole, short in floor_elements(snapshot)]


def floor_allowances(*, window_tokens: Optional[int], output_reserve_tokens: Optional[int], non_memory_tokens: int,
                     target_tokens: Optional[int] = None, calibration_ratio: float = 1.0) -> Dict[str, Optional[int]]:
    """The memory allowances of the three boundaries (``None``: not known, so no step answers to it).

    ``margin``: the window minus the reply reserve and the working margins; ``physical``: the
    window minus the reply reserve; ``budget``: an owner-selected target with its own margins.
    """
    frame = dict(output_reserve_tokens=output_reserve_tokens, non_memory_tokens=non_memory_tokens,
                 calibration_ratio=calibration_ratio)
    window = context_budget.request_context_budget(
        window_tokens=window_tokens, margin_count=context_budget.MEMORY_VIEW_WORKING_MARGINS, **frame)
    budget = context_budget.request_context_budget(
        window_tokens=window_tokens, target_tokens=target_tokens,
        margin_count=context_budget.MODE_TARGET_WORKING_MARGINS, **frame)["with_margin_tokens"] if target_tokens else None
    return {"margin": window["with_margin_tokens"], "physical": window["without_margin_tokens"], "budget": budget}


def fit_memory_view(snapshot: mv.MemoryViewSnapshot, allowances: Mapping[str, Optional[int]]) -> mv.FloorLevel:
    """One pass down the ladder: an element becomes an address while the view exceeds its step's boundary.

    My replies and people's words answer to ``physical``, every other step to ``margin`` and, in
    ``MODE_TARGET_STEPS``, also to ``budget``. Each element saves its known difference;
    the view is measured once, in full, and never rendered again to choose.
    """
    current = view_tokens(mv.render_story(snapshot)) + view_tokens(mv.render_room(snapshot))
    taken: Dict[str, List[str]] = {}
    by_budget = 0
    for step, ident, whole, short in degradable_elements(snapshot):
        window = allowances.get("physical" if step in WINDOW_ONLY_STEPS else "margin")
        budget = allowances.get("budget") if step in MODE_TARGET_STEPS else None
        over_window = window is not None and current > window
        if not over_window and not (budget is not None and current > budget):
            continue
        current -= whole - short
        taken.setdefault(step, []).append(ident)
        by_budget += not over_window
    return mv.FloorLevel(tuple((step, tuple(taken[step])) for step in LADDER if step in taken), by_budget=by_budget)


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
               lowered_from: Optional[str] = None, tool_names: Optional[Iterable[str]] = None) -> str:
    """``### Physical floor``: a fact and a possibility for the mind, exactly when a step past F1/F1b ran
    or the window lowered the task's starting mode; never an instruction or a threshold. In Nano it
    closes with how this request reaches ``memory_read`` (``tool_names``: the schemas it sends). The
    sealing possibility is named only when the request can call ``chronicle_write``: it sends it or
    ``enable_tools`` (unknown schemas keep it); like the path line, a fact of the request, not a role."""
    names = None if tool_names is None else tuple(tool_names)
    counts = dict(level.steps)
    asked = set(counts) - set(_COLLAPSING)
    if not asked and not lowered_from:
        return ""
    lines, name = ["### Physical floor"], mode.capitalize()
    if asked:
        bounds = ([f"this window ({window_tokens} tokens, {name})"] if sum(counts.values()) > level.by_budget else []) + (
            [f"the {name} mode budget ({target_tokens} tokens)"] if level.by_budget else [])
        subject = " and ".join(bounds)
        shown = ", ".join(text.format(counts[step]) for step, text in _SHOWN if counts.get(step))
        sealing = (" Sealing a closed part of the open conversation as a page (chronicle_write kind=page) or folding "
                   "old pages (kind=part) brings it back in my own words."
                   if names is None or _SEALING_TOOLS & set(names) else "")
        lines.append(f"{subject[0].upper()}{subject[1:]} {'do' if len(bounds) > 1 else 'does'} not hold all of my "
                     f"memory verbatim. Shown above only by address: {shown}. Nothing is lost: memory_read reads "
                     f"each.{sealing}")
    if lowered_from:
        lines.append(f"This window ({window_tokens} tokens) cannot hold {lowered_from.capitalize()} with even the "
                     f"shortest view of my memory; this task started in {name}.")
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
    addressed = gone.get("F2", set()) | gone.get("F7", set())  # once, not per lane-1 row
    rows += [(item["pos"], item["address"]) for item in room.get("lane1") or () if item["address"] in addressed]
    status = snapshot.legacy_blocks or {}
    return {"role": snapshot.spec.role, "room_id": snapshot.spec.room_id,
            "floor": {"steps": dict(level.steps), "window_tokens": window_tokens, "mode": mode,
                      "allowance_tokens": allowances.get("margin"), "physical_allowance_tokens": allowances.get("physical"),
                      "target_tokens": target_tokens, "budget_allowance_tokens": allowances.get("budget"),
                      "by_budget": level.by_budget, "mode_switch": {"from": lowered_from, "to": mode} if lowered_from else None,
                      "newest_addressed_row": parse_address(max(rows)[1]) if rows else None,
                      "pointer_records": [ident for step, ids in level.addressed if step in ("F4", "F5") for ident in ids]},
            "story_status": {"folded": status.get("folded", 0), "total": status.get("total", 0)}}


def render_view_for_mode(snapshot: mv.MemoryViewSnapshot, *, mode: str, owner_mode: str, window_tokens: Optional[int],
                         known_window: bool, output_reserve: Optional[int], ratio: float, non_memory_tokens: int,
                         lowered_from: Optional[str] = None,
                         tool_names: Optional[Iterable[str]] = None) -> Tuple[str, str, Dict[str, Any]]:
    """``(story text, room text, facts)`` of one mode's projection, the floor decided once.

    The window counts only when known and fresh (``known_window``). An owner target binds
    only the mode the owner selected: task-local Low and a mode the window lowered have
    none. The reply reserve is the mode's (Nano keeps its own headroom). ``tool_names`` are
    the schemas this mode's request sends, which the floor note reads (``memory_read_path``).
    """
    target, reserve = context_budget.context_mode_limits(mode, owner_mode, output_reserve)
    window = int(window_tokens) if known_window and window_tokens else None
    allowances = floor_allowances(window_tokens=window, output_reserve_tokens=reserve, non_memory_tokens=non_memory_tokens,
                                  target_tokens=target, calibration_ratio=ratio)
    level = fit_memory_view(snapshot, allowances)
    note = floor_note(level, window_tokens=window, mode=mode, target_tokens=target, lowered_from=lowered_from,
                      tool_names=tool_names)
    return (mv.render_story(snapshot, level), mv.render_room(snapshot, level, floor_note=note),
            view_facts(snapshot, level, window_tokens=window, mode=mode, allowances=allowances, target_tokens=target,
                       lowered_from=lowered_from))


TRACE_FIELDS = ("role", "room_id", "floor", "story_status")  # what a task trace keeps of a view receipt


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
    """The shortest view of my memory: every degradable element by address, with the longest floor note."""
    taken: Dict[str, List[str]] = {}
    for step, ident, _whole, _short in floor_elements(snapshot):  # grouped in ladder order
        taken.setdefault(step, []).append(ident)
    level = mv.FloorLevel(tuple((step, tuple(ids)) for step, ids in taken.items()))
    note = max((floor_note(level, window_tokens=window_tokens, mode="nano", lowered_from="max", tool_names=names)
                for names in _PATH_CASES), key=view_tokens)  # an upper bound, whatever schemas it sends
    return view_tokens(mv.render_story(snapshot, level)) + view_tokens(mv.render_room(snapshot, level, floor_note=note))


def physical_mode(preferred: str, fixed_tokens_by_mode: Mapping[str, int], minimal_view_tokens: int, *,
                  window_tokens: Optional[int], known_window: bool, reserve_by_mode: Mapping[str, Optional[int]],
                  calibration_ratio: float = 1.0) -> str:
    """The first of the preferred mode and the smaller ones whose fixed part and shortest view fit.

    Measured by the one capacity frame on the calibrated estimate: the window minus the
    mode's reply reserve, no working margin. An unknown window keeps the preferred mode;
    when even Nano cannot fit, Nano is sent as it is.
    """
    if not known_window or not window_tokens or preferred not in MODES:
        return preferred
    candidates = MODES[MODES.index(preferred):]
    for mode in candidates:
        frame = context_budget.request_context_budget(
            window_tokens=window_tokens, output_reserve_tokens=reserve_by_mode.get(mode),
            non_memory_tokens=fixed_tokens_by_mode[mode] + minimal_view_tokens, calibration_ratio=calibration_ratio)
        if frame["free_tokens"] is not None and frame["free_tokens"] >= 0:  # free space: no working margin in it
            return mode
    return candidates[-1]


def mode_views(snapshot: mv.MemoryViewSnapshot, *, preferred: str, fixed_tokens_by_mode: Mapping[str, int],
               window_tokens: Optional[int], known_window: bool, output_reserve: Optional[int], ratio: float,
               start: Optional[str] = None, tool_names: Optional[Mapping[str, Iterable[str]]] = None,
               allow_mode_lowering: bool = True,
               ) -> Tuple[Dict[str, Tuple[str, str, Dict[str, Any]]], str]:
    """Every mode's ``(story text, room text, view receipt)`` and the mode the task starts in.

    The starting mode is ``physical_mode`` of the owner's ``preferred`` one, or ``start`` (the
    mode a task already runs in, on a new route) when that is lower: a route switch never raises
    a mode, and only the owner's mode carries a target. A caller that keeps current Max books
    until actual refusal sets ``allow_mode_lowering=False``: memory views still fit their
    ordinary allowances, but estimated pressure cannot select another mode or report a switch.
    The projection of a mode this window
    chose names the change in its ``### Physical floor`` (and its fact, ``mode_switch``), never
    in the runtime facts, which are captured before any mode is chosen. ``tool_names`` maps a
    mode to the schemas its request sends (``None``: not known, so its floor claims no path).
    """
    window = int(window_tokens) if known_window and window_tokens else None
    physical = (physical_mode(preferred, fixed_tokens_by_mode, minimal_view_tokens(snapshot, window_tokens=window),
                              window_tokens=window, known_window=known_window, calibration_ratio=ratio,
                              reserve_by_mode={mode: context_budget.context_mode_limits(mode, preferred, output_reserve)[1]
                                               for mode in MODES})
                if allow_mode_lowering else preferred)
    begin = start if start in MODES and MODES.index(start) > MODES.index(physical) else physical
    views = {}
    for mode in MODES:
        story, room, facts = render_view_for_mode(
            snapshot, mode=mode, owner_mode=preferred, window_tokens=window, known_window=known_window,
            output_reserve=output_reserve, ratio=ratio, non_memory_tokens=fixed_tokens_by_mode[mode],
            lowered_from=preferred if mode == begin == physical != preferred else None,
            tool_names=(tool_names or {}).get(mode))
        views[mode] = (story, room, view_receipt(snapshot, story, room, facts))
    return views, begin
