"""One start allowance and a distance-ordered walk over an immutable memory snapshot.

A replacement is my telling of that source in the same block beside its address,
a host fact, or a recorded window/mode emergency. ``memory_forms.STAGES`` starts
with remote ordinary forms, then this room's task facts and intermediate replies,
remote untold words, this room's final replies, remote last-gist counts and story/notes;
physical forms follow, with owner words last. The clean-start guard excludes emergencies
when ordinary forms fit the same complete frame with the work room.
After the start, automatic fitting narrows selected forms only for a physical deficit.
This room's uncovered owner words may spend the work room.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
from functools import cache, cached_property
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros import context_budget
from ouroboros.context_timing import timed_section
from ouroboros import memory_view as mv
from ouroboros import memory_view_conversations as conversations
from ouroboros import memory_view_legacy as legacy
from ouroboros.memory_forms import (LADDER, list_elements as _list_elements, ordinary_pairs, normalized, swallowed,
                                    form_counts, holding_room, Selection, supersedes, WorkPrices, restore,
                                    ORDINARY_STAGES, TELLING_STAGES, ROOM_STEPS)
from ouroboros.memory_form_facts import holding_lines, telling_emergency, physical_floor
from ouroboros.chat_chain import parse_address
from ouroboros.context_layout import WHOLE_OR_OUTLINE_STEPS
from ouroboros.section_floor import PROTECTED, relocated_text, retain_section, section_views

log = logging.getLogger(__name__)

MODES = ("max", "low", "nano")
_COLLAPSING = ("F1", "F7a", "F5a", "F8a")
_ORDINARY = ("F1",)  # task lines collapsed: the form a view takes before any cut
_LIST_FORMS = {'F7a': ('F2', 'F7p', 'F7'), 'F5a': ('F5',), 'F8a': ('F8',)}
_SHOWN = (("F2", "{} of my replies"), ("F7", "{} owner rows"), ("F7p", "{} other spoken rows"),
          ("F6w", "{} words of my other conversations"), ("F6g", "{} other conversations as their lines"),
          ("F6a", "{} count line of my other conversations (the window's last rung)"), ("F4", "{} room records"),
          ("F5l", "{} story pages by first sentence"), ("F5", "{} pages or parts of my story"), ("F3", "{} retold rooms"),
          ("F1", "{} task fact lines"))
_SHOWN += (("F7a", "{} aggregate row ranges"), ("F5a", "{} aggregate story lists"))
_SHOWN += (("F8", "{} unsealed notes by label"), ("F8a", "{} aggregate note lists"))
# The forms a floor fact names (``view_facts``): a change to what a step's elements are or how they are listed raises
# it, so a fact of other forms is read anew (``kept_level``). A fact without one names the forms before it, version 1.
FORMS_VERSION = 2


def tokens_of(chars: int) -> int:
    """The host's estimate of a text of ``chars`` JSON-escaped characters inside a request (``view_tokens``)."""
    return max(1, (chars + 5) // 4) if chars else 0


def view_tokens(text: str) -> int:
    """The host's estimate of a text inside a request: its JSON-escaped characters / 4."""
    return tokens_of(mv.json_chars(str(text))) if text else 0


def memory_tokens(view: Any) -> int:
    """A priced view's two memory texts, story and room, in tokens, as a start's Max room counts them."""
    return tokens_of(view.story_chars) + tokens_of(view.room_chars)


class ViewFit:
    """One frame's memory view: ``level``, its receipt ``note`` and ``facts``, priced from the snapshot's pieces. Its
    whole texts are drawn only when read, so a frame the fit walks away from is only priced, piece by piece."""

    def __init__(self, snapshot: mv.MemoryViewSnapshot, level: mv.FloorLevel, note: str, facts: Dict[str, Any]):
        self.snapshot, self.level, self.note, self.facts = snapshot, level, note, facts

    @cached_property
    def story(self) -> str:
        return mv.render_story(self.snapshot, self.level)

    @cached_property
    def room(self) -> str:
        return mv.render_room(self.snapshot, self.level, floor_note=self.note)

    @cached_property
    def story_chars(self) -> int:
        return mv.story_chars(self.snapshot, self.level)

    @cached_property
    def room_chars(self) -> int:
        return mv.room_chars(self.snapshot, self.level, floor_note=self.note)


NO_VIEW = SimpleNamespace(story="", room="", facts={}, level=None, story_chars=0, room_chars=0)  # a frame with no memory


def sealed_owner_rows(room: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """A project root's rows of its human that left the open conversation (``owner_words.project_words``), as F7
    holds them before lane 1's: by their chat address, oldest first, uncovered (only physical pressure or the root's
    own authored fold changes their form, into dated addresses); a store gap line is no row."""
    return [item for item in room.get("project_words") or () if item.get("address")]


class _Elements:
    """One snapshot's floor elements, listed once (``_elements``): each step's ids in ladder order and the
    ``step:id`` addresses, so a level is one pass over its own steps; their token prices on first need."""

    def __init__(self, listed: List[Tuple[str, str, str, str]], stages):
        self.listed, self.ids, self.stages = tuple(listed), {}, stages
        for step, ident, _whole, _short in self.listed:
            self.ids.setdefault(step, []).append(ident)
        self.ids = {s: tuple(ids) for s, ids in self.ids.items()}

    @cached_property
    def eligible(self):
        return {f"{step}:{ident}": (step, ident) for step, ident, *_ in self.listed}

    @cached_property
    def positions(self):
        return {s: {i: n for n, i in enumerate(ids)} for s, ids in self.ids.items()}

    @cached_property
    def replacements(self):
        upper = {(s, i): below for s, ids in self.ids.items() for i in ids if (below := supersedes(s, i))}
        lower = {}
        for p, below in upper.items():
            for b in below:
                lower.setdefault(b, []).append(p)
        return upper, lower

    @cached_property
    def priced(self) -> Tuple[Tuple[str, str, int, int], ...]:
        return tuple((step, ident, view_tokens(whole), view_tokens(short)) for step, ident, whole, short in self.listed)

    @cached_property
    def order(self) -> str:
        """A fingerprint of each step's ids in their order: what index runs count against."""
        return hashlib.sha256(json.dumps(self.ids, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

    def runs(self, level: mv.FloorLevel) -> Dict[str, List[List[int]]]:
        """Each step's addressed elements as ``[first, last]`` positions in that step's order."""
        runs: Dict[str, List[List[int]]] = {}
        for step, chosen in level.addressed:
            chosen, spans = set(chosen), runs.setdefault(step, [])
            for n, ident in enumerate(self.ids.get(step, ())):
                if ident in chosen and spans and spans[-1][1] == n - 1:
                    spans[-1][1] = n
                elif ident in chosen:
                    spans.append([n, n])
        return runs

    def level(self, taken, by_budget: int = 0) -> mv.FloorLevel:
        """The level of ``(step, id)`` pairs, each step's ids in ladder order."""
        chosen = ((step, tuple(i for i in self.ids.get(step, ()) if (step, i) in taken)) for step in LADDER)
        return mv.FloorLevel(tuple((step, ids) for step, ids in chosen if ids), by_budget=by_budget)


def _elements(snapshot: mv.MemoryViewSnapshot) -> _Elements:
    """The snapshot's element table: a non-field attribute (``snapshot_json`` and equality never see it) that
    lives as long as the snapshot object (``memory_view._parsed_view``); a replaced snapshot lists its own, and a
    different ``LADDER`` (the steps a table lists) lists anew rather than reading a table it never built."""
    ladder, table = snapshot.__dict__.get("_floor_elements") or ((), None)
    if ladder != LADDER:
        listed = [element for element in _list_elements(snapshot) if element[0] in LADDER]
        table = _Elements(listed, snapshot.__dict__.get("_form_stages", {}))
        object.__setattr__(snapshot, "_floor_elements", (LADDER, table))
    return table


def floor_elements(snapshot: mv.MemoryViewSnapshot) -> Tuple[Tuple[str, str, str, str], ...]:
    """Ordered addressable forms, oldest first; F5 by covered-period start then sequence, across kinds.
    Display order is independent. Core and governing words have no rung. F7 holds a project root's
    sealed rows of its human before the open ones: dated addresses, never a retelling."""
    return _elements(snapshot).listed


def degradable_elements(snapshot: mv.MemoryViewSnapshot) -> Tuple[Tuple[str, str, int, int], ...]:
    """``(step, element id, tokens whole, tokens by address)`` in ladder order: each element's known saving."""
    return _elements(snapshot).priced


def floor_allowances(*, window_tokens: Optional[int], output_reserve_tokens: Optional[int], non_memory_tokens: int,
                     target_tokens: Optional[int] = None, calibration_ratio: float = 1.0,
                     work_room_levels: int = 1, boundary_tokens: Optional[int] = None) -> Dict[str, Optional[int]]:
    """One start bound for all unsolicited material; a physical bound for uncovered owner rows."""
    frame = dict(output_reserve_tokens=output_reserve_tokens, non_memory_tokens=non_memory_tokens,
                 calibration_ratio=calibration_ratio, margin_count=work_room_levels, boundary_tokens=boundary_tokens)
    physical = context_budget.request_context_budget(window_tokens=window_tokens, **frame)
    start = context_budget.request_context_budget(window_tokens=window_tokens, target_tokens=target_tokens, **frame)
    # A target names its cuts only where it is tighter than the window's own start bound, never the window's cut
    window = physical["with_margin_tokens"]  # None: an unknown window, the target alone bounds the start
    budget = start["with_margin_tokens"] if target_tokens and (window is None or start["with_margin_tokens"] < window) else None
    return {"margin": start["with_margin_tokens"], "physical": physical["without_margin_tokens"], "window": window,
            "budget": budget}


def max_room_fact(snapshot: Optional[mv.MemoryViewSnapshot], core: Any, books: Iterable[str], *, story: int, fixed: int,
                  schemas: int, window: int, reserve: int, work: int, ratio: float, allowance: Optional[int],
                  addressed: Optional[Mapping[str, str]] = None) -> Optional[Dict[str, Any]]:
    """The Max room of one start, the host's measure in the owner's Max frame with whole books.

    ``story`` is its tokens. ``allowance`` is the memory room that frame leaves (``None``: an unknown window, no fact).
    ``floor_need`` is the books, core, my knowledge (its index, shared understanding, a project's journal), source
    registry, scratchpad and held note heads, runtime facts (Health, recent activity, drive state too), the skills (their
    lines and always-in-context bodies) and subagents, the connections map Max shows, assignment, reply reserve and work
    room; ``zero_need`` adds the tool schemas, ``need`` my memory in its ordinary forms (task lines collapsed, nothing
    cut): ``story`` (``unfolded``: its pages no part or account holds yet) and ``room``. ``parts`` read the host's
    sections by id as the request carries them (``addressed``: those sent in a shorter form), each part by whose it is
    (``WHOSE``), in measured tokens before ``density``. Measures only: the start alone chooses the mode, and the mind
    judges what to fold.
    """
    if allowance is None:
        return None
    addressed = addressed or {}

    def sent(sections, *ids):  # these host sections as the request carries them; ``held_note:`` names every held note
        return sum(view_tokens(addressed.get(part.section_id, part.full_text)) for part in sections if part.section_id in ids
                   or ("held_note:" in ids and part.section_id.startswith("held_note:")))
    base = view_tokens(core.base_prompt) + view_tokens(core.bible_md)
    stable = sent(core.semi_stable_sections, "subagents", "always_on_skills")  # the stable block's skills and subagents
    skills = sent(core.dynamic_sections, "installed_skills")
    mine = {"knowledge": view_tokens(core.dynamic_head_text), "source registry": sent(core.dynamic_sections, "registry"),
            "scratchpad": sent(core.dynamic_sections, "scratchpad"), "held note heads": sent(core.dynamic_sections, "held_note:")}
    parts = {"books": max(0, sum(map(view_tokens, books)) - base),
             "core": max(0, base + view_tokens(core.semi_stable_text) - stable),
             "runtime facts": max(0, view_tokens(core.dynamic_text) - skills - sum(  # the changing block's other sections
                 mine[name] for name in ("source registry", "scratchpad", "held note heads"))),
             **{name: tokens for name, tokens in {**mine, "skills and subagents": stable + skills}.items() if tokens},
             **({"connections": view_tokens(core.connections[0])} if core.connections[0] else {})}  # Max's map (``_system_blocks``)
    parts["assignment"] = max(0, fixed - schemas - sum(parts.values()))  # the task's messages and their framing
    ordinary = ordinary_level(snapshot) if snapshot else mv.FULL_VIEW
    memory = {"story": tokens_of(mv.story_chars(snapshot, ordinary)) if snapshot else story, "room": tokens_of(mv.room_chars(snapshot, ordinary)) if snapshot else 0,
              "unfolded": unfolded_pages(snapshot)["tokens"]}
    zero = math.ceil(fixed * ratio) + reserve + work
    return {"window": window, "allowance": allowance, "floor_need": math.ceil((fixed - schemas) * ratio) + reserve + work,
            "schemas": schemas, "zero_need": zero, "need": zero + math.ceil((memory["story"] + memory["room"]) * ratio),
            **memory, "reserve": reserve, "work": work, "density": ratio,
            "levels": snapshot.spec.work_room_levels if snapshot else 1,
            "room_id": snapshot.spec.room_id if snapshot else None,
            "folds": sum(int((snapshot.legacy_blocks or {}).get(key) or 0) for key in ("parts", "accounts")) if snapshot else 0,
            "parts": sorted(([name, tokens] for name, tokens in parts.items()), key=lambda part: -part[1])}


def max_room_lost(fact: Mapping[str, Any]) -> bool:
    """Max cannot hold whole books here: my memory, even in its shortest telling of the whole horizon
    (``telling_level``), does not fit beside them and the work's room (``telling_need``; ``need`` before it existed)."""
    return fact.get("telling_need", fact["need"]) > fact["window"]


def with_max_room(facts: Mapping[str, Any], fact: Optional[Mapping[str, Any]], mode: str,
                  required_book: int = 0) -> Dict[str, Any]:
    """The view facts with the start's settled Max room: ``held`` when Max kept whole books (``cuts``: the steps past
    F1 it took); ``route`` when the books, core, assignment, reply reserve and work room alone exceed the window
    (a disclosed fact of the route); else ``crowded`` (memory or tool schemas took the room). ``headroom`` or
    ``short`` compare the whole need with the window. A held Max also counts the ``required_book`` it carries beyond
    the measured view (calibrated tokens); the choices above were made without it."""
    if not fact:
        return dict(facts)
    steps = dict((facts.get("floor") or {}).get("steps") or {})
    cause = "held" if mode == "max" else "route" if fact["floor_need"] > fact["window"] else "crowded"
    if mode == "max" and required_book:
        fact = {**fact, **{key: fact[key] + required_book for key in ("floor_need", "zero_need", "need", "telling_need")
                           if key in fact},
                "parts": [*fact["parts"], ["directly required handbook", required_book]]}
    gap = fact["window"] - fact["need"]
    return {**facts, "max_room": {**fact, "cause": cause, **({"headroom": gap} if gap >= 0 else {"short": -gap}),
                                  "cuts": [step for step in LADDER if cause == "held" and step not in _ORDINARY and steps.get(step)]}}


def crowded_by_memory(fact: Optional[Mapping[str, Any]]) -> bool:
    """A settled Max room (``with_max_room``) that memory lowered to Low: ``crowded`` while everything but memory
    (books, core, assignment, tool schemas, reply reserve and work room: ``zero_need``) fits the window, so folding
    memory can give a later start its books back. A route, or tool schemas that tip a start alone, are not memory's."""
    return (isinstance(fact, Mapping) and fact.get("cause") == "crowded" and isinstance(fact.get("zero_need"), int)
            and isinstance(fact.get("window"), int) and fact["zero_need"] <= fact["window"])


REQUIRED_BOOK_WHOLE = " (this task's directly required handbook stays whole)"
# An owner's Max whose whole books did not fit runs in Low, temporarily, for the window, and is named Low wherever its
# mode is shown, while my human's choice stays Max (owner 21.11.2026). Its rendered key is ``low`` and its profile
# ``task_local_low``; the owner's own Low has ``preferred_mode`` low.
TEMPORARY_LOW = "Low, temporarily, for the window"
# What the owner's Low target may take (``render_view_for_mode``, owner 15.11.2026, 3A): other conversations and my
# story down to my telling; this room and my human's words not at all. The rest only the window's own start bound takes.
TARGET_FLOOR, TARGET_KEEPS = TELLING_STAGES, ROOM_STEPS
LOWERED_CAUSES = ("memory", "request", "route")  # ``lowered_cause``
# Whose each part of a start's measure is (``measure_text``): core, the assignment and any other part stand alone.
WHOSE = (("mine", ("knowledge", "source registry", "scratchpad", "held note heads")),
         ("what the request itself carries", ("connections", "runtime facts", "skills and subagents")),
         ("my human's", ("books",)))


def gate_need(fact: Mapping[str, Any]) -> int:
    """What the books gate weighed (``max_room_lost``): whole books, my memory at the gate's level (my shortest telling
    of the whole horizon, ``telling_level``), the work's room and the reply reserve; the ordinary need where no telling
    was priced, the gate then weighing that."""
    return fact.get("telling_need", fact["need"])


def lowered_cause(fact: Mapping[str, Any]) -> str:
    """Why a Max room lost its whole books, by its own numbers: ``route`` when the books, core, assignment, reply reserve
    and work room exceed the window with no memory and no tool schemas (``floor_need``), ``request`` when what the request
    itself carries tips that over (``zero_need``), else ``memory``. A measure, not blame."""
    return ("route" if fact["floor_need"] > fact["window"] else "request" if fact["zero_need"] > fact["window"]
            else "memory")


def lowered_need(fact: Mapping[str, Any]) -> Tuple[str, int]:
    """``(cause, need)``: the need a lowered start's cause names against the window (``lowered_cause``)."""
    cause = lowered_cause(fact)
    return cause, gate_need(fact) if cause == "memory" else fact["zero_need" if cause == "request" else "floor_need"]


def start_cause(facts: Optional[Mapping[str, Any]]) -> str:
    """Why a start fit's own projection (its ``memory_facts``) is below the owner's mode: ``""`` without a mode
    switch (``floor.mode_switch``); for an owner's Max whose whole books did not fit, its Max room's ``memory``,
    ``request`` or ``route`` (``lowered_cause``); any other narrowing by the window (Low to Nano) is ``window``."""
    facts = facts or {}
    if not (facts.get("floor") or {}).get("mode_switch"):
        return ""
    room = facts.get("max_room")
    return lowered_cause(room) if isinstance(room, Mapping) and room.get("cause") in ("crowded", "route") else "window"


def drawn_tokens(entry: Mapping[str, Any]) -> int:
    """One story entry as the full view draws it: a retold record whole (``F3``'s whole form), else its own page
    lines (``memory_view._page_lines``; an account's told lines aside). Told by an account, it is not drawn: 0."""
    if entry.get("told_by"):
        return 0
    return view_tokens(legacy.retold_record(entry) if entry.get("kind") == "legacy" else "\n".join(mv._page_lines(entry)))


def story_measure(story: Iterable[Mapping[str, Any]], status: Optional[Mapping[str, Any]] = None,
                  refusals: Iterable[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    """My story from its captured entries alone (``memory_view.capture_story``): ``tokens`` as the full view draws
    the whole block (``render_story``), ``drawn`` each entry's own tokens by id (``drawn_tokens``) and ``unfolded``
    (``unfolded_pages``). What ``memory_read(story=true)`` reports; a measure, never a mode."""
    story = tuple(story)
    snapshot = mv.MemoryViewSnapshot(spec=mv.ViewSpec("integrator"), store_status={"state": "active"}, frontier={},
                                     story=story, legacy_blocks=dict(status or {}), fallback_refusals=tuple(refusals))
    return {"tokens": view_tokens(mv.render_story(snapshot)), "unfolded": unfolded_pages(snapshot),
            "drawn": {str(entry["id"]): drawn_tokens(entry) for entry in story if not entry.get("told_by")}}


def unfolded_pages(snapshot: Optional[mv.MemoryViewSnapshot]) -> Dict[str, Any]:
    """My story's pages no part or account holds yet, from the captured snapshot alone: how many, their tokens and
    ``rooms`` as ``[label, tokens, pages]``, the room holding most of them first (where they pile up)."""
    rooms: Dict[str, List[Any]] = {}
    for entry in snapshot.story if snapshot else ():
        if entry.get("kind") == "page" and not entry.get("told_by"):
            room = rooms.setdefault(str(entry.get("room_id")), [str(entry.get("label") or entry.get("room_id")), 0, 0])
            room[1] += drawn_tokens(entry)
            room[2] += 1
    ranked = sorted(rooms.values(), key=lambda room: (-room[1], room[0]))
    return {"pages": sum(room[2] for room in ranked), "tokens": sum(room[1] for room in ranked), "rooms": ranked}


def pages_text(pages: Mapping[str, Any], *, rooms: int = 3) -> str:
    """``N pages no part or account holds yet, T tokens, most in <room> t, …``: the rooms holding most of them."""
    most = ", ".join(f"{label} {tokens:,}" for label, tokens, _count in pages["rooms"][:rooms])
    return (f"{pages['pages']} page{'' if pages['pages'] == 1 else 's'} no part or account holds yet, "
            f"{pages['tokens']:,} tokens" + (f", most in {most}" if most else ""))


def addressed_text(steps: Mapping[str, Any]) -> str:
    """``3 task fact lines, 2 owner rows``: what a view shows only by address, in its floor note's words."""
    return ", ".join(text.format(steps[step]) for step, text in _SHOWN if steps.get(step))


def measure_text(fact: Mapping[str, Any], *, parts: bool = False, pages: Optional[Mapping[str, Any]] = None) -> str:
    """The host's measure of one start in the provable parts; the mind judges what they mean: the gate's need
    (``gate_need``: whole books, my memory at the gate's level, the work's room, the reply reserve) against the window,
    the ordinary need (my memory with no shortened form) under that name and the need with no memory; with ``parts``
    the measure by whose part it is (``WHOSE``), else the tool schemas and the need without them too. ``pages``
    (``unfolded_pages`` of the same snapshot) names how many unfolded pages there are and where they pile up."""
    window, gate = fact["window"], gate_need(fact)
    unfolded = pages_text(pages) if pages is not None else f"pages no part or account holds yet {fact['unfolded']:,}"
    text = (f"window {window:,}; the gate's need {gate:,}, {'headroom' if gate <= window else 'short by'} "
            f"{abs(window - gate):,}; the ordinary need {fact['need']:,}; with no memory {fact['zero_need']:,}")
    if not parts:  # the parts name the schemas whose; here they and the need without them stand alone
        text += f"; tool schemas {fact['schemas']:,}; without them {fact['floor_need']:,}"
    if parts:
        named, grouped = dict(fact["parts"]), {name for _whose, names in WHOSE for name in names}
        listed = {"mine": [f"my memory at the gate's level {fact.get('telling', fact['story'] + fact['room']):,}"],
                  "what the request itself carries": [f"tool schemas {fact['schemas']:,}"], "my human's": []}
        for whose, names in WHOSE:
            listed[whose] += [f"{name} {named[name]:,}" for name in names if name in named]
        listed["my human's"] += [f"reply reserve {fact['reserve']:,}", f"work room {fact['work']:,}"]
        alone = sorted((part for part in fact["parts"] if part[0] not in grouped),  # core, assignment, then the rest
                       key=lambda part: ("core", "assignment").index(part[0]) if part[0] in ("core", "assignment") else 2)
        density = float(fact.get("density") or 1.0)  # the needs apply it; reserve and work room are window tokens
        text += "; " + "; ".join([f"{whose}: {', '.join(items)}" for whose, items in listed.items()]
                                 + [f"{name} {tokens:,}" for name, tokens in alone]
                                 + ([f"parts before density {density:.2f} except reply reserve and work room"]
                                    if abs(density - 1.0) > 0.005 else []))
    return (f"{text}; my story {fact['story']:,} ({unfolded}) + rooms {fact['room']:,} in ordinary forms, beside a "
            f"Max room of {fact['allowance']:,}")


_LOWERED_WORDS = {"memory": "Whole books, my memory at the gate's level and the work's room need",
                  "request": "With no memory, whole books, what the request itself carries and the work's room need",
                  "route": ("With no memory and no tool schemas, whole books beside the rest of this start (my core, the "
                            "host's sections and the assignment) and the work's room need")}


def lowered_line(mode: str, lowered_from: str, fact: Optional[Mapping[str, Any]] = None) -> str:
    """Why a start left the owner's mode, in measured parts: facts only, never an instruction or a possibility."""
    if mode != "low":
        return (f"The window cannot hold {lowered_from.capitalize()} with even the shortest view of my memory; "
                f"this task runs in {mode.capitalize()}.")
    head = f"{TEMPORARY_LOW}; my human's mode is {lowered_from.capitalize()}."
    if not fact:
        return f"{head} The whole books did not fit beside my memory and the work's room."
    cause, need = lowered_need(fact)
    short = need - fact["window"]
    route = (", a disclosed fact of this route: folding my memory or dropping tool schemas cannot make room here"
             if cause == "route" else "")
    return (f"{head} {_LOWERED_WORDS[cause]} {need:,}; the window holds {fact['window']:,}"
            + (f"; short by {short:,}" if short > 0 else "") + f"{route}. Only the window cuts my memory here; Low's own "
            f"target does not bind. The host's measure of this start: {measure_text(fact, parts=True)}.")


def required_book_note(*, lowered: bool, book_tokens: int, free: Optional[int], work: int) -> str:
    """What a start says of this task's directly required handbook, appended after every ordinary choice (owner
    1.11.2026): a Low its room lowered keeps that book whole, and a work room it leaves short is a fact, never a veto.
    ``book_tokens`` and ``free`` are calibrated; ``free`` is None when nothing measured the start's room."""
    lines = ["This task's directly required handbook stays whole beside these chapter maps."] if lowered else []
    if free is not None and free < work:
        lines.append(f"Start room is smaller than requested: this task's directly required handbook takes "
                     f"{book_tokens:,} tokens; about {max(0, free):,} of the {work:,} requested remain.")
    return "\n".join(lines)


@timed_section("fit.walk")
def fit_memory_view(snapshot: mv.MemoryViewSnapshot, allowances: Mapping[str, Optional[int]], *,
                    kept: mv.FloorLevel = mv.FULL_VIEW, saving: int = 0,
                    measure: Optional[Callable[[mv.FloorLevel], int]] = None,
                    floor: Optional[frozenset] = None, floor_bound: Optional[int] = None,
                    keeps: frozenset = frozenset()) -> mv.FloorLevel:
    """One ordered walk, then give back new cuts that fit. Previously chosen forms stay selected. A level is priced
    from the snapshot's pieces (``render_view_for_mode``'s ``measure``, by default its story and room characters).
    ``floor``: the stages this walk may take for its bound (``keeps``: steps it may not take at all); any other form
    is taken only while the view exceeds ``floor_bound`` (the window's own start bound), never with no such bound,
    and a target's count (``by_budget``) never names it."""
    table = _elements(snapshot)
    elements = table.priced
    work_prices = WorkPrices(snapshot.room or {}, elements, view_tokens)
    previous = {(step, ident): short for step, ident, _whole, short in elements if step in ('F5l', 'F6g')}
    old = {(step, ident) for step, ids in kept.addressed for ident in ids}
    taken = Selection(table, old, old=old, aggregates=_LIST_FORMS)
    measure = measure or (lambda level: tokens_of(mv.story_chars(snapshot, level)) + tokens_of(mv.room_chars(snapshot, level)))
    measure = cache(measure)  # One immutable frame; later frames have their own prices.
    current = measure(kept)
    bounds = [b for b in (allowances.get("margin"), allowances.get("budget"), allowances.get("physical")) if b is not None]
    bound = min(bounds) if bounds else None
    target = min(bound if bound is not None else current, current - saving) if saving else bound
    budget = allowances.get("budget") is not None and allowances.get("budget") == bound

    forced = set()  # what only the window's bound took past the floor

    def level(selected):
        return selected.level(LADDER, kept.by_budget, budget, forced)

    ordinary = ordinary_pairs(snapshot, table.listed)
    # The same minimum ordinary view prices the clean-start guard and both Max decisions.
    clean = target is None or measure(level(Selection(table, old | ordinary, old=old, aggregates=_LIST_FORMS))
                                      if old or budget or kept.by_budget else ordinary_level(snapshot)) <= target
    uncovered = uncovered_owner_rows(snapshot, elements)
    last = {(step, ident) for step, ident, *_ in uncovered}
    added = []
    counted_groups = {i for s, i in old if s == "F6a" and i != "unread"}
    for step, ident, whole, short in elements:
        if target is not None and current <= target:
            current = measure(level(taken))
        if target is None or current <= target:
            break
        pair = step, ident
        if pair in taken or pair in last or (clean and pair not in ordinary) or swallowed(snapshot, step, ident, taken):
            continue
        held = floor is not None and (step in keeps or table.stages[pair] not in floor)
        if held and (floor_bound is None or (current := measure(level(taken))) <= floor_bound):
            continue  # the window's bound is judged on the priced view
        if step in (*_LIST_FORMS, "F8") or step == "F6a" and (ident == "unread" or not counted_groups):
            extra = [pair] + ([("F8", i) for i in table.ids.get("F8", ()) if ("F8", i) not in taken] if step == "F8a" else [])
            before = measure(level(taken))
            for added_pair in extra:
                taken.add(added_pair)
            actual = measure(level(taken))
            # The first counted group pays the shared header; later groups amortize it.
            if actual >= before and not (step == "F6a" and ident != "unread" and not counted_groups):
                for added_pair in reversed(extra):
                    taken.discard(added_pair)
                continue
            delta, current = before - actual, actual
            forced.update(extra if held else ())
        else:
            if step == "F6g" or step == "F6a":
                key = ident.split(":", 1)[1] if ident.startswith(("all:", "last:", "events:")) else ident
                whole = next((previous["F6g", name] for name in ("last:" + key, "all:" + key, key, "events:" + key)
                              if ("F6g", name) in taken), whole)
            if step == "F5l" and ("F5", ident) in taken:
                continue
            if step == "F5" and ("F5l", ident) in taken:
                whole = previous['F5l', ident]
            delta = work_prices.saving(ident, taken) if step == "F2" and ident.startswith("work:") else whole - short
            if delta <= 0:
                continue
            current -= delta
            taken.add(pair)
            forced.update([pair] if held else ())
        added.append((pair, delta))
        if step == "F6a" and ident != "unread":
            counted_groups.add(ident)
    current = measure(level(taken))
    current = restore(snapshot, table, taken, added, target, current, level, measure, view_tokens)
    # A Max view floored at my tellings keeps my human's words while its books are whole: when the window cannot
    # hold it, the books go to chapter maps first (the next frame), and only there may the window take them (4A).
    physical = None if floor is not None and floor_bound is None else allowances.get("physical")
    for step, ident, _whole, _short in uncovered:
        if physical is None or current <= physical:
            break
        if (step, ident) not in taken:
            taken.add((step, ident))
            forced.update([(step, ident)] if floor is not None else ())  # the window's cut, never a floored target's
            current = measure(level(taken))
            if current > physical:
                form = ("F7a", str((snapshot.room or {})['room_id']))
                if form[1] not in table.ids.get("F7a", ()):
                    continue
                had_form = form in taken
                taken.add(form)
                marked = floor is not None and not had_form and form not in forced
                forced.update([form] if marked else ())  # priced as the window's cut, its receipt included
                after = measure(level(taken))
                if after < current:
                    current = after
                elif not had_form:
                    taken.discard(form)
                    if marked:
                        forced.discard(form)
    return level(taken)


def ordinary_level(snapshot, stages_of=ORDINARY_STAGES):
    """All ordinary replacements, retaining every source that has no in-view telling; with ``TELLING_STAGES``
    (``telling_level``) every form down to my telling or a conversation's line, nothing below it."""
    table, key = _elements(snapshot), "_ordinary_level" if stages_of is ORDINARY_STAGES else "_telling_level"
    previous, level = snapshot.__dict__.get(key, (None, None))
    if previous is not table:
        level = table.level(normalized(ordinary_pairs(snapshot, table.listed, stages_of), table.replacements[0]))
        object.__setattr__(snapshot, key, (table, level))
    return level


def telling_level(snapshot):
    """My shortest telling of the whole horizon: what a Max start prices beside whole books (``max_room_lost``)."""
    return ordinary_level(snapshot, TELLING_STAGES)


def owner_rows(snapshot: mv.MemoryViewSnapshot) -> Dict[str, Mapping[str, Any]]:
    """My human's rows by their F7 id: this room's by address (a project root's sealed rows, ``sealed_owner_rows``,
    among them), my other conversations' by ``room:address``, and carried human quotes by page and index."""
    room = snapshot.room or {}
    return {**conversations.owner_items(snapshot.conversations),
            **mv.story_quote_rows(snapshot),
            **{i["address"]: i for i in [*sealed_owner_rows(room), *room.get("lane1", ())]}}


def uncovered_owner_rows(snapshot: mv.MemoryViewSnapshot, elements=None) -> List[Tuple[str, str, int, int]]:
    """This room's owner rows no page or note carries yet, oldest first; governing and unanswered rows have no
    address. A project root's sealed rows of its human count as uncovered: only physical pressure folds them. My
    human's words of my other conversations are ordinary F7 elements: they keep the work room."""
    room = snapshot.room or {}
    owner = {i["address"]: i for i in [*sealed_owner_rows(room), *room.get("lane1", ())]}
    return [e for e in (degradable_elements(snapshot) if elements is None else elements)
            if e[0] == "F7" and e[1] in owner and not owner[e[1]].get("carried")]


def kept_level(snapshot: mv.MemoryViewSnapshot, facts: Mapping[str, Any]) -> mv.FloorLevel:
    """The level a projection's floor fact names, exactly or not at all: its index runs while the fact names these
    forms (``FORMS_VERSION``) and its runs count against this snapshot's element order, which fingerprints the exact
    source ids (an older fact's ``addressed`` ids likewise). Any other fact is read anew: no element is kept, so the
    next fit walks from the whole view; only its count of what the owner's mode budget took stays, which names that
    budget to the walk and the note (``render_view_for_mode``). A step's count never stands for its first elements."""
    floor, table = (facts or {}).get("floor") or {}, _elements(snapshot)
    forms, budget = floor.get("forms_version", 1) == FORMS_VERSION, int(floor.get("by_budget") or 0)
    if forms and "runs" in floor and floor.get("element_order") == table.order:
        return table.level({(step, table.ids[step][n]) for step, spans in floor["runs"].items()
                            for first, last in spans for n in range(first, last + 1)}, budget)
    if forms and "addressed" in floor:
        return mv.FloorLevel(tuple((step, tuple(ids)) for step, ids in floor["addressed"]), by_budget=budget)
    return mv.FloorLevel(by_budget=budget)


def level_with(snapshot: mv.MemoryViewSnapshot, kept: mv.FloorLevel, addresses: Iterable[str], *,
               owner_note: bool = False) -> mv.FloorLevel:
    """Select exact inspected forms; a note's :whole address restores its captured source."""
    table = _elements(snapshot)  # listed once per snapshot: a receipt prices many levels
    eligible = table.eligible
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
        if step == "F7" and not owner_note and not owner_rows(snapshot)[ident].get("carried"):
            raise ValueError("Uncovered owner commitments require an authored working_note before changing their form")
        taken.add((step, ident))
        if step in ('F5l', 'F5'):
            taken.discard(('F5' if step == 'F5l' else 'F5l', ident))
        if step == 'F8':
            taken = {(s, i) for s, i in taken if s != 'F8a'}  # an explicit label selection unfolds the note list
    return table.level(normalized(taken), kept.by_budget)


def select_view(plan, mode: str, addresses: Iterable[str], *, owner_note: bool = False, tool_names=None):
    """Render selected forms, preserving other projections and the tail.

    The boundary's fit callback decides publication. Ordinary Main accepts an
    authored restore with an advisory capacity deficit; dispatch fits it later.
    ``tool_names``: the schemas the next request sends, which its floor note reads; else those its start priced.
    """
    from ouroboros.context_fit import _governance_blocks, _system_blocks, _request_tokens

    if plan is None or getattr(plan, "core", None) is None or not plan.core.memory_view_json:
        raise ValueError("This actor has no captured memory view")
    snapshot = mv.snapshot_from_json(plan.core.memory_view_json)
    shown = plan.projection(mode)
    level = level_with(snapshot, kept_level(snapshot, shown.memory_facts), addresses, owner_note=owner_note)
    old = shown.memory_facts.get("floor") or {}
    switched = old.get("mode_switch") or {}
    max_room = shown.memory_facts.get("max_room")  # the start's fact: a selection keeps its frame
    names = tool_names if tool_names is not None else plan.start_schema_names(mode)  # None: no claim
    note = floor_note(level, window_tokens=old.get("window_tokens"), mode=mode, tool_names=names,
                      target_tokens=old.get("target_tokens"), budget_mode=plan.preferred_mode,
                      lowered_from=switched.get("from"), max_room=max_room, snapshot=snapshot)
    story, room = mv.render_story(snapshot, level), mv.render_room(snapshot, level, floor_note=note)
    facts = view_facts(snapshot, level, window_tokens=old.get("window_tokens"), mode=mode,
                       allowances={"margin": old.get("allowance_tokens"), "physical": old.get("physical_allowance_tokens"),
                                   "budget": old.get("budget_allowance_tokens")},
                       target_tokens=old.get("target_tokens"), lowered_from=switched.get("from"))
    facts.update({"max_room": max_room} if max_room else {})
    # Low's missed target was the start's: what its request took with no memory stays its fact (3A)
    facts["floor"].update({"no_memory_tokens": old["no_memory_tokens"]} if "no_memory_tokens" in old else {})
    governance = _governance_blocks(None, plan.core, mode="low" if mode == "nano" else mode)
    sections = (*plan.core.semi_stable_sections, *plan.core.dynamic_head_sections, *plan.core.dynamic_sections,
                *(governance[0] if not isinstance(governance[0], str) else ()))
    receipts = shown.memory_facts.get("host_sections_addressed", ())
    if receipts:
        facts["host_sections_addressed"] = receipts
    system = _system_blocks(plan.core, governance, story, room, mode, section_views(sections, receipts), mv.story_is_whole(level))
    tokens = _request_tokens(system, shown.user_content_json or plan.user_content_json,
                            current_messages=plan.messages_for(mode))
    projection = dataclasses.replace(shown, system_content_json=json.dumps(system, ensure_ascii=False, sort_keys=True),
        memory_facts=view_receipt(snapshot, story, room, facts), estimated_tokens=tokens,
        calibrated_tokens=int(tokens * shown.calibration_ratio))
    return dataclasses.replace(plan, **{f"{mode}_projection": projection})


def owner_note_capsule(plan, addresses, note, request, *, drive_root, task_id, mode="max", written_at=None):
    """Keep the mind's account of selected owner commitments with their exact captured source, dated like the call's
    other notes by ``written_at`` (when the mind wrote it; else now, as for a request queued before it carried one)."""
    from ouroboros import context_compaction as cc
    from ouroboros.context_budget import _Selection

    wanted = {address[3:] for address in addresses if address.startswith("F7:")}
    if any(address.startswith("F7a:") for address in addresses):
        wanted.update(dict(kept_level(mv.snapshot_from_json(plan.core.memory_view_json),
                                     plan.projection(mode).memory_facts).addressed).get("F7", ()))
    if not wanted:
        return None
    snap = mv.snapshot_from_json(plan.core.memory_view_json)
    rows = [{"role": "user", "content": row["line"]} for ident, row in owner_rows(snap).items()
            if ident in wanted and not row.get("carried")]
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
    return cc._capsule_message(unit, note, [cc._part(unit.unit_id, unit.source_text)], source, request,
                               written_at=written_at)[0]


def view_inventory(plan, mode: str, *, memory_facts=None) -> List[Dict[str, Any]]:
    if plan is None or getattr(plan, "core", None) is None:
        return []
    from ouroboros.section_floor import section_inventory
    facts = plan.projection(mode).memory_facts if memory_facts is None else memory_facts
    host_rows = section_inventory(plan, mode, facts)
    if not plan.core.memory_view_json:
        return host_rows
    snapshot = mv.snapshot_from_json(plan.core.memory_view_json)
    level = kept_level(snapshot, plan.projection(mode).memory_facts if memory_facts is None else memory_facts)
    chosen = {(step, ident) for step, ids in level.addressed for ident in ids}
    return [{"view_address": f"{step}:{ident}", "kind": "memory", "tokens": whole,
             "saving_tokens": max(0, whole - short), "eligible": True, "selected": (step, ident) in chosen,
             **({"restore_view_address": f"F8:{ident}:whole", "form":
                 'whole' if (step, ident) not in chosen else 'aggregate' if dict(level.addressed).get('F8a') else 'line'}
                if step == 'F8' else {})}
            for step, ident, whole, short in degradable_elements(snapshot)] + host_rows


def memory_read_path(tool_names: Optional[Iterable[str]]) -> str:
    """How ``memory_read`` is reached, from the schema names sent when this view is rendered: nothing when they hold
    it or are unknown (``None``: no claim); with ``enable_tools``, that it loads the reader where this task's policy
    permits (``disabled_tools`` can refuse it); else its absence then, which a later schema-only change does not
    redraw, and ``list_available_tools`` when sent. A fact of that rendering's tools, never of a role or a grant."""
    if tool_names is None:
        return ""
    names = set(tool_names)
    if "memory_read" in names:
        return ""
    if "enable_tools" in names:
        return "(enable_tools loads memory_read if permitted)"
    listing = "; list_available_tools shows what this task can call" if "list_available_tools" in names else ""
    return f"(memory_read was not among my tools when this view was rendered{listing})"


def floor_note(level: mv.FloorLevel, *, window_tokens: Optional[int], mode: str, target_tokens: Optional[int] = None,
               lowered_from: Optional[str] = None, tool_names: Optional[Iterable[str]] = None,
               budget_mode: Optional[str] = None, max_room: Optional[Mapping[str, Any]] = None,
               snapshot: Optional[mv.MemoryViewSnapshot] = None) -> str:
    """Retained cuts, their final visible forms and the sources holding this room.

    The three bins distinguish my telling, host facts and emergencies; the global unread count is a
    separate window fact. Nano names the reader path whenever the tools it is rendered with lack memory_read, cut
    or not (``### Memory check``). The receipt is priced with every candidate and preserves a held view's frame.
    """
    names = None if tool_names is None else tuple(tool_names)
    counts = dict(level.steps)
    asked = set(counts) - set(_COLLAPSING)
    forms = form_counts(snapshot, level) if snapshot else {}
    holding = snapshot and (telling_emergency(snapshot, level) or lowered_from and crowded_by_memory(max_room))
    path = memory_read_path(names) if mode == "nano" else ""
    if not asked and not lowered_from and not holding:
        return f"### Memory check\nAddresses this view shows read with memory_read.\n{path}" if path else ""
    lines, name = ["### Physical floor"], mode.capitalize()  # a Low the window took from Max is Low (21.11)
    if asked:
        bounds = ([f"the window ({window_tokens} tokens)"]
                  if window_tokens is not None else []) + (
            [f"the {(budget_mode or mode).capitalize()} mode budget ({target_tokens} tokens)"] if level.by_budget else [])
        subject = " and ".join(bounds)
        def describe(kind):
            return ", ".join(f"{label}: {fact['lines']} lines/{fact['sources']} sources" for label, fact in forms.get(kind, {}).items()) or "none"
        shown = (f"my telling: {describe('retold')}; host facts: {describe('host')}; "
                 f"emergency without my telling: {describe('emergency')}" if forms else addressed_text(counts))
        lines.append(f"{name} cuts for {subject}. Reduced forms — {shown}. Read sources with memory_read.")
        if forms.get("window"):
            lines.append(f"Window emergency over unread words: {describe('window')}; not a failure of my telling.")
    if holding:
        lines.extend(holding_lines(holding_room(snapshot, level), short=physical_floor(snapshot, level, mode)))
    if lowered_from:
        lines.append(lowered_line(mode, lowered_from, max_room))
    return "\n".join(lines + [path] if path else lines)


def view_facts(snapshot: mv.MemoryViewSnapshot, level: mv.FloorLevel, *, window_tokens: Optional[int], mode: str,
               allowances: Mapping[str, Optional[int]], target_tokens: Optional[int] = None,
               lowered_from: Optional[str] = None) -> Dict[str, Any]:
    """The floor fact of one projection, kept in the task trace: steps, boundaries, what is shown by address.

    ``runs`` are each step's addressed elements as index runs in its element order, which ``element_order``
    fingerprints, of the forms ``forms_version`` names (``kept_level``); ``newest_addressed_row`` is the newest row of
    this room that F1, F2 or F7 shows only by address; ``pointer_records`` the records F4 and F5 show by a pointer.
    No list of rows.
    """
    gone = {step: set(ids) for step, ids in level.addressed}
    room = snapshot.room or {}
    rows = [(item["last_pos"], item["last"]) for item in room.get("lane2") or () if item["first"] in gone.get("F1", ())]
    addressed = gone.get("F2", set()) | gone.get("F7", set()) | gone.get("F7p", set())  # once, not per lane-1 row
    addressed |= {i[5:] for i in addressed if i.startswith("bare:")}
    from ouroboros.memory_view_work import folded_work_rows
    addressed |= folded_work_rows(room, gone)[0]
    rows += [(item["pos"], item["address"]) for item in room.get("lane1") or () if item["address"] in addressed]
    status = snapshot.legacy_blocks or {}
    return {"role": snapshot.spec.role, "room_id": snapshot.spec.room_id,
            "floor": {"steps": dict(level.steps), "runs": _elements(snapshot).runs(level), "forms_version": FORMS_VERSION,
                      "element_order": _elements(snapshot).order, "window_tokens": window_tokens, "mode": mode,
                      "allowance_tokens": allowances.get("margin"), "physical_allowance_tokens": allowances.get("physical"),
                      "target_tokens": target_tokens, "budget_allowance_tokens": allowances.get("budget"),
                      "by_budget": level.by_budget, "mode_switch": {"from": lowered_from, "to": mode} if lowered_from else None,
                      "forms": form_counts(snapshot, level), "holding": holding_room(snapshot, level),
                      "newest_addressed_row": parse_address(max(rows)[1]) if rows else None,
                      "folded_rooms": conversations.folded_rooms(snapshot.conversations, gone),
                      "pointer_records": [ident for step, ids in level.addressed if step in ("F4", "F5l", "F5") for ident in ids]},
            "story_status": {"folded": status.get("folded", 0), "total": status.get("total", 0)}}


def render_view_for_mode(snapshot: mv.MemoryViewSnapshot, *, mode: str, owner_mode: str, window_tokens: Optional[int],
                         known_window: bool, output_reserve: Optional[int], ratio: float, non_memory_tokens: int,
                         lowered_from: Optional[str] = None, tool_names: Optional[Iterable[str]] = None,
                         kept: Optional[mv.FloorLevel] = None, saving: int = 0, boundary_tokens: Optional[int] = None,
                         working_reserve_tokens: Optional[int] = None,
                         measure_text: Optional[Callable[[str, str], int]] = None,
                         start_work: bool = False,
                         max_room: Optional[Mapping[str, Any]] = None, stated_window_tokens: Optional[int] = None,
                         shortest: bool = False, level: Optional[mv.FloorLevel] = None,
                         report_nano_miss: bool = False) -> ViewFit:
    """One mode's projection, the floor decided once.

    The window counts only when known and fresh (``known_window``). An owner target binds
    only the mode the owner selected: task-local Low and a mode the window lowered have
    none. The reply reserve is the mode's (Nano keeps its own headroom). ``tool_names`` are
    the schemas this mode's request sends, which the floor note reads (``memory_read_path``).
    ``kept`` narrows a view already shown: from its level, against the window minus the reply
    reserve alone -- no working margin, no target -- so nothing it addressed comes back.
    What ``kept.by_budget`` counts the owner's mode target took (only it carries one), so the note
    and the fact name that budget in whatever mode the view is narrowed, hop after hop.
    ``shortest``: where this walk ends when only uncovered owner rows are left to address, with its
    completed receipt -- what a start's host walk judges beside the window (``_host_steps``).
    ``level``: such a walk's level again, without walking, its receipt priced beside this request's host forms.
    A level is priced from the snapshot's pieces, never by drawing it; ``measure_text(story_chars, room_chars)`` prices
    the two memory texts of a whole request (by default their own tokens).
    """
    target, reserve = context_budget.context_mode_limits(mode, owner_mode, output_reserve, working_reserve_tokens=working_reserve_tokens)
    if kept is not None:
        target = context_budget.context_mode_limits(owner_mode, owner_mode, output_reserve)[0] if kept.by_budget else None
    window = int(window_tokens) if known_window and window_tokens else None
    named = int(stated_window_tokens) if window is not None and stated_window_tokens else window  # as the note names it
    allowances = floor_allowances(window_tokens=window, output_reserve_tokens=reserve, non_memory_tokens=non_memory_tokens,
                                  target_tokens=None if kept is not None else target, calibration_ratio=ratio,
                                  work_room_levels=snapshot.spec.work_room_levels, boundary_tokens=boundary_tokens)
    if kept is not None and not start_work:
        allowances = {**allowances, "margin": allowances["physical"]}
    # Owner 15.11.2026 (3A, 4A): Low's target never takes memory below my shortest telling of the whole horizon and
    # never takes this room, and a Max start keeping whole books never goes below it either; only the window's own
    # start bound may. A miss of the target is its own fact. Nano's number still governs all memory (owner 11.10, 13.10).
    binds = kept is None and target is not None and owner_mode == "low" and (window is None or target < window)
    reports_miss = binds or (report_nano_miss and kept is None and mode == owner_mode == "nano"
                             and target is not None and (window is None or target < window))
    floor, keeps = ((TARGET_FLOOR, TARGET_KEEPS) if binds else (TELLING_STAGES, frozenset())  # the one order (4A)
                    if mode == "max" and (kept is None or start_work) else (None, frozenset()))
    size = lambda shown: tokens_of(mv.story_chars(snapshot, shown)) + tokens_of(mv.room_chars(snapshot, shown))
    room_bound = allowances["window"] if binds else allowances["margin"]  # a target's miss is its own fact, below
    uncovered = uncovered_owner_rows(snapshot)
    def total(level, note):  # the whole request with this view and its receipt, as the target counts it
        return math.ceil((non_memory_tokens + measure_text(mv.story_chars(snapshot, level),
                                                           mv.room_chars(snapshot, level, floor_note=note))) * ratio) + reserve
    def receipt(level):
        note = floor_note(level, window_tokens=named, mode=mode, target_tokens=target, lowered_from=lowered_from,
                          tool_names=tool_names, budget_mode=owner_mode, max_room=max_room, snapshot=snapshot)
        if reports_miss and (need := total(level, note)) > target:
            telling = " with my memory in its shortest tellings" if binds else ""
            note += (f"\n{mode.capitalize()}'s target ({target:,} tokens) is not met at this start: this request needs "
                     f"about {need:,}{telling}, "
                     f"{math.ceil(non_memory_tokens * ratio) + reserve:,} with no memory at all.")
        if room_bound is not None and size(level) > room_bound:
            gone = dict(level.addressed)
            held = [f"F7:{ident}" for step, ident, *_ in uncovered if ident not in gone.get(step, ())]
            owners = held and size(level_with(snapshot, level, held, owner_note=True)) <= room_bound
            note += "\nStart room is smaller than requested: " + (
                "uncovered owner words remain in view; only physical pressure can turn their oldest rows into named addresses."
                if owners else "what my memory shows even in its shortest forms exceeds it.")
        return note
    # The request owner measures JSON separators and rounding in the actual envelope, beside these two texts' characters.
    measure_text = measure_text or (lambda story_chars, room_chars: tokens_of(story_chars) + tokens_of(room_chars))
    def measure(level):  # the two texts this level adds to a request, its receipt included
        return measure_text(mv.story_chars(snapshot, level), mv.room_chars(snapshot, level, floor_note=receipt(level)))
    walk = dict(kept=kept or mv.FULL_VIEW, measure=measure, floor=floor, keeps=keeps,
                floor_bound=allowances["window"] if binds else None)
    if shortest and level is None:  # A saving no form can pay takes every form in walk order; no physical bound
        # stops it before owner rows. The physical judgment of host forms never stops at a Max start's telling floor:
        # what the window cannot hold beside the books is the books' own step (``context_fit``), not an outline's.
        level = fit_memory_view(snapshot, {**allowances, "physical": None}, saving=math.inf,
                                **{**walk, **({} if binds else {"floor": None, "keeps": frozenset()})})
    elif level is None:
        level = fit_memory_view(snapshot, allowances, saving=saving, **walk)
    facts = view_facts(snapshot, level, window_tokens=named, mode=mode, allowances=allowances,
                       target_tokens=target, lowered_from=lowered_from)
    bound, note = allowances["margin"], receipt(level)
    facts["floor"]["ordinary_unfit"] = bound is not None and measure(ordinary_level(snapshot)) > bound
    facts["floor"]["target_missed"] = reports_miss and total(level, note) > target
    if facts["floor"]["target_missed"]:  # where the mode is chosen it says what the rest of the request took (3A)
        facts["floor"]["no_memory_tokens"] = math.ceil(non_memory_tokens * ratio) + reserve
    return ViewFit(snapshot, level, note, facts)


TRACE_FIELDS = ("role", "room_id", "floor", "story_status", "max_room", "host_sections_addressed")  # what a task trace keeps of a view receipt


def trace_facts(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """The part of a view receipt the task context and the task trace keep (``VIEW_TRACE_KEY``)."""
    return {key: receipt[key] for key in TRACE_FIELDS if key in receipt}


def view_receipt(snapshot: mv.MemoryViewSnapshot, story: str, room: str, facts: Mapping[str, Any]) -> Dict[str, Any]:
    """A projection's view fact: the floor fact plus the spec, the chronicle's state and block sizes.

    ``role``, ``room_id``, ``floor``, ``story_status`` and the start's ``max_room`` are what a task trace
    keeps (``memory_inventory.VIEW_TRACE_KEY``); the rest is the receipt's (estimator tokens).
    """
    return {**facts, "spec": dataclasses.asdict(snapshot.spec), "store_status": dict(snapshot.store_status),
            "legacy_frontier_status": snapshot.frontier.get("status"),
            "blocks": {"story_tokens": view_tokens(story), "room_tokens": view_tokens(room),
                       "conversations": len(snapshot.conversations), "marks": len(snapshot.marks)}}


# Host producers before the whole-or-outline sections, then the memory walk, all under the same
# start budget. Running views narrow only under physical pressure or a refusal.
HOST_STEPS = ("runtime", "review_pool", "review_continuity", "installed_skills",
              "drive_state", "world", "subagents", "architecture_map", "development_map", "return_facts_words")
# Host forms only the window's own start bound takes: my WORLD map stays whole while that bound holds the request with
# memory at its shortest forms, whatever soft frame the start has (memory audit 15.11.2026, F7); when the bound cannot
# hold it, it still steps before memory.
WINDOW_ONLY = ("world",)
# The facts beside my human's word (``return_facts``) and the people's newest words beside them (``return_facts_words``,
# owner 18.11.2026, 1A): they stand while the window's start bound holds the request with memory at my shortest telling
# and the books as the frame holds them, whatever a target or a selected Low asks; past it the words step first to their
# readers, then the facts to their totals, before the books or a telling would (owner 15.11.2026, 4A: steps 3 and 4).
TELLING_SPARED = ("return_facts", "return_facts_words")


def _held_note_steps(sections, receipts, fits, *, form="summary"):
    """The last host forms to yield: the captured heads in address order, one at a time until ``fits``, a head only a
    link holds (``carried_by``) before every claimed one.

    In a map frame each steps to its authored summary beside its exact reader, once the physical window has left no
    index or registry it could still outline and the request with memory at its shortest forms still exceeds the
    heads' bound (``_host_steps``); there a claimed head without a summary stays whole, while a carried one whose
    summary form saves no room (or that has none) keeps only its exact reader where that is shorter, so it still
    yields first; a head shorter than its reader form stays whole. At the emergency floor only the exact reader
    remains. A head whose reader the actor cannot open has no shorter form at either step: it stays whole.
    """
    from ouroboros.held_notes import HeldNoteSection

    addressed = section_views(sections, receipts)
    positions = {row["section_id"]: index for index, row in enumerate(receipts)}
    entries = [part for part in sections if isinstance(part, HeldNoteSection)]
    # At the emergency floor a resident summary already kept its meaning: spend the label's small saving last.
    entries.sort(key=lambda part: (form == "reader" and part.section_id in positions
                                   and part.summary_is_resident(addressed), not part.carried_by))
    saved = 0
    for part in entries:
        position = positions.get(part.section_id)
        if position is not None and (form != "reader" or receipts[position].get("held_note_form") != "summary"):
            continue
        row = part.window_receipt(form)
        short = part.view_for_receipt(row, addressed)
        if form == "summary" and part.carried_by and not part.summary_saves(addressed):  # its reader yields first
            row = part.window_receipt("reader")
            short = part.view_for_receipt(row, addressed)
        current = addressed.get(part.section_id, part.full_text)
        if short is None or (delta := view_tokens(current) - view_tokens(short)) <= 0:
            continue
        if fits(addressed):
            break
        addressed[part.section_id] = short
        if position is None:
            receipts.append({**row, "tokens_saved": delta})
        else:
            receipts[position] = {**row, "tokens_saved": receipts[position]["tokens_saved"] + delta}
        saved += delta
    return saved


def _host_steps(sections, previous, *, measure, window, reserve, ratio, saving, root, task_id, allowance=None,
                shortest=None, budget_mode=None, slack=0, bare=None, retain=None, window_bound=None, told=None,
                held_least=None, held_bound=None):
    """First steps of mode_views, using the same physical allowance as memory.

    Producers own meaning; this walk selects only explicit smaller views. Retained
    source handles precede pointers. A failed capture leaves the full bytes inline.
    Previously shown addresses stay addressed; their savings never pay a new deficit.
    A start's allowance below the physical one is a soft frame: there the knowledge
    indexes and the registry (``WHOLE_OR_OUTLINE_STEPS``) step only when the request
    exceeds the physical window, never for a requested saving; a receipt records
    ``start_room`` when its step paid that soft frame while the window held the request:
    ``True``, or ``budget_mode`` when the owner's budget for that mode framed it.
    Elsewhere they step while the request exceeds the window or ``saving`` still holds
    a measured or refused excess; ``slack``, the part of ``saving`` a caller asks
    beyond it, only ordinary host forms and memory pay.
    A start's frame judges what it would send: ``shortest(addressed, physical)`` measures
    the request beside these host forms once that frame's own memory walk has taken every
    form before uncovered owner rows, its floor receipt included
    (``render_view_for_mode(shortest=True)``). After each new address that receipt can
    grow, so the frame re-prices the walk's level beside the new forms and walks again
    only when the level no longer fits (``context_fit._view_projections``).
    ``retain`` captures a section before its pointer (``retain_section``; a write-free fit passes
    ``section_floor.predicted_section``). ``bare(addressed)`` measures a request without memory: when even every candidate at
    its shorter form exceeds the window that way, no price here can fit, so each candidate
    steps for the window unpriced and the saving stays unpaid.
    Every frame begins from what was shown (``previous``) and pays what that frame
    still owes (``context_fit._view_projections``): steps an abandoned Max trial took
    never reach Low, and books that paid a deficit leave no host step to pay it again.
    A ``WINDOW_ONLY`` section steps only while the request with memory at its shortest forms (``shortest``) exceeds
    ``window_bound``, the window's own start bound: neither an owner's target nor the start's own room shortens it.
    A ``TELLING_SPARED`` section steps only while the window cannot hold the request with memory at my shortest telling
    and the books as the frame holds them (``told``: ``max_room_lost``'s own test), so before the books or a telling.
    Held heads yield last (``_held_note_steps``). A map frame that holds them passes ``held_least(addressed,
    physical)``, the request beside these host forms with memory at its shortest forms (``shortest``'s measure, for a
    view already sent too), and ``held_bound``, the window's own bound for a head: a start's keeps its work room, a
    view already sent has the window less its reply reserve. The indexes and the registry answer to the physical
    window alone in every frame, with or without heads (owner 2.11.2026, ``context_layout.WHOLE_OR_OUTLINE_STEPS``).
    A head steps only when both hold: this walk left no outline untaken that the physical window still holds whole,
    and the request still exceeds ``held_bound``; then one at a time. So where a start misses its own bound while the
    physical window holds the whole index, neither yields: that start has less work room, as a start without heads
    has there. A target, a soft frame or a requested saving shortens no head.
    """
    from ouroboros.held_notes import HeldNoteSection

    receipts = list(previous)
    physical = floor_allowances(window_tokens=window or None, output_reserve_tokens=reserve,
                                non_memory_tokens=0, calibration_ratio=ratio)["physical"]

    def finish(remaining, outline_left=False):
        if held_least is not None and physical is not None and not outline_left:
            def fits(forms):  # as the request stands first: memory is priced at its shortest forms only past the bound
                return (measure is not None and measure(forms) <= held_bound) or held_least(forms, physical) <= held_bound
            remaining = max(0, remaining - _held_note_steps(sections, receipts, fits))
        return receipts, remaining

    if measure is None or not window or not root or not task_id:
        return finish(saving)
    addressed = section_views(sections, receipts)
    candidates = [part for part in sections if part.section_id not in addressed
                  and part.section_id not in PROTECTED and not isinstance(part, HeldNoteSection)
                  and part.operative_view is not None
                  and len(part.operative_view) < len(part.full_text)]
    if not candidates:
        return finish(saving)
    # No step reaches a shorter form (a pointer follows the operative view) and memory only adds to a request.
    least = bare({**addressed, **{part.section_id: min((part.full_text, part.operative_view), key=view_tokens)
                                  for part in candidates}}) if bare else None
    if least is not None and least > physical:  # every price this walk could take exceeds the window
        measure = shortest = lambda _addressed, *_physical: least
    initial = current = measure(addressed)
    bound = allowance if allowance is not None else physical
    if (bound is None or current <= bound) and saving == 0:
        return receipts, 0  # this frame's allowance holds the request as it stands: nothing yields, a head least
    soft = bound is not None and bound < physical
    excess = max(0, saving - slack)  # what a window or a refusal measured; slack never outlines
    fits = None  # what this start can send, uncovered owner words whole: priced when a step needs it
    outline_left = False  # an index or the registry the physical window still holds whole: a head waits for it
    order = {name: i for i, name in enumerate((*HOST_STEPS, *WHOLE_OR_OUTLINE_STEPS))}
    for part in sorted(candidates, key=lambda p: (p.section_id in WHOLE_OR_OUTLINE_STEPS,
                                               order.get(p.section_id, len(HOST_STEPS)))):
        if part.section_id in addressed:
            continue
        if (bound is None or current <= bound) and initial - current >= saving:
            break
        if fits is None:
            sent_least = shortest(addressed, physical) if shortest else current  # memory at its shortest forms
            fits = sent_least <= physical
        if part.section_id in WHOLE_OR_OUTLINE_STEPS and fits and (soft or initial - current >= excess):
            outline_left = True
            continue  # an outline is a physical step, never a start-room or slack choice
        spared = part.section_id in WINDOW_ONLY or part.section_id in TELLING_SPARED
        if part.section_id in WINDOW_ONLY and window_bound is not None and sent_least <= window_bound:
            continue  # the window's start bound holds it beside memory's shortest forms: no soft frame shortens it
        if part.section_id in TELLING_SPARED and (told(addressed) if told else window_bound is None or current <= window_bound):
            continue  # the window holds it beside my shortest telling and the books: neither yields to it
        try:
            row = (retain or retain_section)(part, root=root, task_id=task_id)
        except (OSError, ValueError):
            log.warning("Host section capture unavailable; keeping %s whole", part.section_id, exc_info=True)
            continue
        start_room = ((not spared and budget_mode or True) if soft and fits else False) \
            if part.section_id not in TELLING_SPARED else False  # spared: the window's own room; told: the window's fact
        short = relocated_text(part, row["reader"], start_room=start_room)
        saved = view_tokens(part.full_text) - view_tokens(short)
        if saved <= 0:
            continue
        addressed[part.section_id] = short
        receipts.append({**row, "tokens_saved": saved, **({"start_room": start_room} if start_room else {})})
        current, fits = measure(addressed), None
    return finish(max(0, saving - (initial - current)), outline_left)


# --- a pairwise start fit of other memories in one captured frame ------------------------------------------------

STORY_FIELDS = ("legacy", "under_parts", "told")  # this room's records whose form my story decides


def reprojected(captured: mv.MemoryViewSnapshot, fresh: mv.MemoryViewSnapshot) -> mv.MemoryViewSnapshot:
    """The captured view with only what my story decides read again (``fresh``, a capture of the same spec with no
    lanes, conversations or marks): the story, its status and refusals, this room's retold records, pages under parts
    and told records. Rows, conversations, marks, notes and every other fact stay as this task captured them."""
    room = captured.room
    if room is not None and fresh.room is not None:
        room = {**room, **{key: fresh.room[key] for key in STORY_FIELDS if key in fresh.room}}
    return dataclasses.replace(captured, story=fresh.story, legacy_blocks=fresh.legacy_blocks,
                               store_status=fresh.store_status, fallback_refusals=fresh.fallback_refusals, room=room)


def counterfactual_starts(plan: Any, snapshots: Mapping[str, mv.MemoryViewSnapshot]) -> Dict[str, Any]:
    """Each named memory fitted as a fresh start of this plan's one captured frame by the start's own fitter
    (``context_fit._view_projections``): the same core, books, assignment, tool schemas in the shape the start priced, window
    record, reply and working reserves, density and owner mode; nothing else is read. A host step's section is
    priced at the address its capture would publish (``section_floor.predicted_section``) and nothing is written: no
    view receipt, calibration, checkpoint or plan. When no view holds the whole input, the start's own rescue applies
    as production's (``context_input_selection.rescued_start``) to the deliveries this start captured, in its order
    (``rescued_inputs``); an arm on one says ``"input": "rescued"``. ``{"measured": False, "missing": [...]}`` names
    inputs this plan lacks; each arm is ``{"mode", "max_room", "steps", "host"}``, ``{"refused": reason}`` when no
    start view fits, or ``{"unmeasured": reason}`` (a further rescue this start never captured included)."""
    from ouroboros import context_fit as cf
    from ouroboros.section_floor import predicted_section

    core, record = getattr(plan, "core", None), getattr(plan, "window_record", None)
    missing = [name for name, held in (
        ("the captured context core and memory view", core is not None and bool(getattr(core, "memory_view_json", ""))),
        ("the tool schemas its start priced", getattr(plan, "start_tool_schemas_json", None) is not None),
        ("its window record", record is not None)) if not held]
    if missing:
        return {"measured": False, "missing": missing}
    schemas, ratio = json.loads(plan.start_tool_schemas_json), float(plan.max_projection.calibration_ratio)
    governance = {form: cf._governance_blocks(None, core, mode=form) for form in ("max", "low")}
    rescues, last = rescued_inputs(plan)
    arms: Dict[str, Any] = {}
    for name, snapshot in snapshots.items():
        if not snapshot.active:
            arms[name] = {"unmeasured": "memory capture unavailable: " + str(snapshot.store_status.get("state"))}
            continue  # an unreadable journal is not an empty history that fits Max
        def fit(user_content_json: str, snapshot=snapshot) -> Tuple[Dict[str, Any], str]:
            return cf._view_projections(
                dataclasses.replace(core, memory_view_json=mv.snapshot_json(snapshot)), governance,
                user_content_json, preferred=plan.preferred_mode, tool_schemas=schemas,
                window_tokens=record.total_equivalent, known_window=record.usable is not None,
                output_reserve=plan.output_reserve_tokens, window_record=record, ratio=ratio,
                working_reserve_tokens=plan.working_reserve_tokens, work_room_levels=snapshot.spec.work_room_levels,
                retain=predicted_section, provider=plan.provider, nano_base=getattr(plan, "start_nano_base", None),
                reasoning_effort=getattr(plan, "start_reasoning_effort", ""))
        rescued = False
        try:
            try:
                contents, mode = fit(core.user_content_json)
            except cf.MinimumViewUnfit as unfit:
                if not unfit.physical or unfit.required_book_tokens or not rescues:
                    raise  # a refusal production names too; or, with no captured rescue, unknown below
                for rescue in rescues:  # production's order; the last refusal stands as its own
                    try:
                        contents, mode = fit(rescue)
                        break
                    except cf.MinimumViewUnfit as refused:
                        unfit = refused
                else:
                    if not last:
                        raise _UncapturedRescue("every rescue this start captured is refused; its further forms "
                                                "were never captured") from unfit
                    raise unfit
                rescued = True
        except cf.MinimumViewUnfit as unfit:
            arms[name] = ({"unmeasured": "the whole input does not fit and this start captured no rescue of it"}
                          if unfit.physical and not unfit.required_book_tokens and not rescues else {"refused": str(unfit)})
            continue
        except Exception as exc:  # the read keeps its other answers; this arm is unknown, never a mode
            if not isinstance(exc, _UncapturedRescue):
                log.warning("pairwise start fit of %s not measured", name, exc_info=True)
            arms[name] = {"unmeasured": str(exc) if isinstance(exc, _UncapturedRescue) else type(exc).__name__}
            continue
        facts = contents[mode][1]
        arms[name] = {"mode": mode, "max_room": facts.get("max_room"), "steps": dict((facts.get("floor") or {}).get(
            "steps") or {}), "host": [(row["section_id"], row["tokens_saved"]) for row in facts.get(
                "host_sections_addressed", ())], **({"input": "rescued"} if rescued else {})}
    frame = {"core": plan.core_sha256[:12], "preferred": plan.preferred_mode, "window": record.total_equivalent,
             "known": record.usable is not None, "reserve": plan.output_reserve_tokens, "ratio": ratio,
             "schemas": len(schemas or ()), "started": plan.start[0] or plan.initial_mode,  # the start's own mode
             "unmeasured": tuple(plan.start) == ("max", "unmeasured"),  # a Max start whose route named no window
             # an owner's Max the window took to Low, temporarily, carries its cause (``start_cause``), whatever is chosen now
             "maps": (plan.start[0] or plan.initial_mode) == "low" and bool(
                 plan.start[1] if plan.start[0] else start_cause(plan.projection(plan.initial_mode).memory_facts))}
    return {"measured": True, "frame": frame, "arms": arms}


class _UncapturedRescue(Exception):
    """A pairwise start fit reached past the rescues its start captured: unknown, never a refusal."""


def rescued_inputs(plan: Any) -> Tuple[List[str], bool]:
    """The deliveries this plan's start used in place of its whole input (``context_input_selection.rescued_start``,
    held on its projections), in production's rescue order: a wake's bound form (every mode's) first, owner-Nano's
    exact input pointer (Nano's alone) last; and whether that last is production's last rescue, so its refusal is
    one. Nothing is read or written: what the start did not capture is not here."""
    held = {mode: plan.projection(mode).user_content_json for mode in MODES}
    wake = held["max"] or held["low"]
    pointer = held["nano"] if held["nano"] and held["nano"] != wake else None
    rescues = [text for text in (wake, pointer) if text]
    return rescues, bool(pointer) and plan.preferred_mode == "nano"


def counterfactual_lines(fit: Mapping[str, Any]) -> List[str]:
    """``counterfactual_starts`` as the reader's lines: the frame, then each arm's start mode and Max room in the
    host's measured parts (``measure_text``), its floor steps and predicted host steps; a missing input is named."""
    head = "## The next start in this task's captured frame (a pairwise fit; nothing written)"
    if not fit.get("measured"):
        return ["", head, "not measured: this task's plan does not hold " + "; ".join(fit.get("missing") or ()) + "."]
    f = fit["frame"]
    started = (TEMPORARY_LOW if f.get("maps") else "Max, window not measured at the start" if f.get("unmeasured")
               else f["started"])
    lines = ["", head, (
        f"frame: core {f['core']}, owner mode {f['preferred']}, window {f['window']:,}"
        f"{'' if f['known'] else ' (not known: no window bound applies)'}, reply reserve {f['reserve']:,}, density "
        f"{f['ratio']:.3f}, {f['schemas']} tool schemas as this task's start priced them; my story, its status and this "
        "room's retold records read now, everything else as this task captured it. "
        + ("This task's original start is not known: its continuation did not carry it."
           if f["started"] == "unknown" else f"This task started in {started} (its start projection, a separate "
           "fact: my chronicle may have changed since)."))]
    for name, arm in fit["arms"].items():
        if "refused" in arm or "unmeasured" in arm:
            lines.append(f"- {name}: no start view fits ({arm['refused']})" if "refused" in arm
                         else f"- {name}: not measured ({arm['unmeasured']}); its mode is unknown")
            continue
        room = arm["max_room"]
        steps = ", ".join(f"{step} {count}" for step, count in arm["steps"].items()) or "none"
        host = ", ".join(f"{section} ~{saved:,}" for section, saved in arm["host"])
        maps = arm["mode"] == "low" and f["preferred"] == "max"  # Low, temporarily, for the window: my human's Max
        lines.append(f"- {name}: starts {TEMPORARY_LOW if maps else arm['mode'].capitalize()}"
                     + (" on the input this start rescued" if arm.get("input") == "rescued" else "")
                     + (f"; Max room {'held' if room['cause'] == 'held' else 'lost, cause ' + lowered_cause(room)}: "
                        f"{measure_text(room)}" if room else "; no Max room measured")
                     + f"; floor steps {steps}"
                     + (f"; host sections by address, if their capture succeeds: {host}" if host else ""))
    return lines
