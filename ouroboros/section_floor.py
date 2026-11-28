"""Host section views and exact source retention; memory_floor alone selects the steps.

Captures use existing task source handles and their retention/promotion lifecycle,
not a second context-view store or cleanup loop. Unknown producers stay whole.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from ouroboros.context_layout import TIER0_ALWAYS_FULL

log = logging.getLogger(__name__)
PROTECTED = TIER0_ALWAYS_FULL


@dataclass(frozen=True, eq=False, repr=False)
class HostSection(str):
    """A producer's text plus its explicit window view, still usable by string callers.

    The string value is exactly ``full_text``: legacy joins, cache blocks and
    public builders keep their bytes. The frozen fields survive in ContextCore
    for fitting; plain strings enter the capture with no operative view.
    ``reader`` names the original source; relocation always reads the retained
    capture with read_file, so a later change at that source cannot change it.
    """

    section_id: str
    full_text: str
    operative_view: Optional[str] = None
    reader: str = ""

    def __new__(cls, section_id: str, full_text: str, operative_view: Optional[str] = None, reader: str = ""):
        return str.__new__(cls, full_text)

    def __getnewargs__(self):
        return self.section_id, self.full_text, self.operative_view, self.reader

    def view_for_receipt(self, row: Mapping, addressed: Mapping) -> Optional[str]:
        if self.operative_view is None:
            return None
        return relocated_text(self, row["reader"], start_room=row.get("start_room", False))


@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class _ReportSection(HostSection):
    """A report label on otherwise anonymous text; excluded from the captured core identity."""

    report_family: str

    def __new__(cls, section_id, full_text, operative_view=None, reader="", *, report_family=""):
        return super().__new__(cls, section_id, full_text, operative_view, reader)


def report_text(text: str, family: str) -> HostSection:
    return _ReportSection("", text, report_family=family)


def capture_sections(parts: Iterable[str], *, prefix: str) -> tuple[HostSection, ...]:
    """Keep structural section boundaries without guessing an unknown producer's meaning."""
    return tuple(replace(part, section_id=f"{prefix}_{i}") if isinstance(part, _ReportSection) and not part.section_id
                 else part if isinstance(part, HostSection) else HostSection(f"{prefix}_{i}", part)
                 for i, part in enumerate(parts))


def labelled_section(section: str, label: Callable[[str], str]) -> str:
    """Apply the same snapshot label to both forms without losing the captured metadata."""
    if not isinstance(section, HostSection):
        return label(section)
    return replace(section, full_text=label(section.full_text),
                   operative_view=None if section.operative_view is None else label(section.operative_view))


def relocated_text(section: HostSection, reader: str, *, start_room: Union[bool, str] = False) -> str:
    """``start_room``: the step paid this start's allowance while the physical window held the whole request;
    a mode name says the owner's budget for that mode framed it, ``True`` the start's own room. An empty ``reader``:
    this actor can open no pointer (``memory_floor.OPTIONAL_STEPS`` alone step so), and the section keeps only its
    heading and that it is unavailable to this actor: no path or reading instruction it could not follow."""
    soft = f"the owner's {start_room.capitalize()} mode budget" if isinstance(start_room, str) else "this start's room"
    bound = f"{soft}, not for the physical window" if start_room else "the physical window"
    if not reader:
        heading = next((line for line in section.full_text.splitlines() if line.startswith("## ")), section.section_id)
        return f"{heading}\n\n[Omitted for {bound}; unavailable to this actor.]"
    return section.operative_view + "\n\n" + f"[Host section details omitted for {bound}; retained full text: {reader}.]"


def retain_section(section: HostSection, *, root: str, task_id: str) -> dict:
    """Capture before publishing its pointer; existing source custody owns lifetime."""
    from ouroboros.artifacts import store_actor_source_bytes

    ref = store_actor_source_bytes(root, task_id, category="context_checkpoints",
                                   source_id="host-" + section.section_id,
                                   data=section.full_text.encode("utf-8"), extension="md")
    return {"section_id": section.section_id, "source_ref": ref, "reader": _reader(ref)}


def predicted_section(section: HostSection, *, root: str = "", task_id: str = "") -> dict:
    """``retain_section``'s row for a fit that writes nothing: the digest-named address its capture would publish
    for these bytes (``artifacts.actor_source_reference``), so the pointer costs what it would. It holds only if
    that capture succeeds (a failed one keeps the section whole); ``predicted`` keeps it out of every sent view."""
    from ouroboros.artifacts import actor_source_reference

    ref = actor_source_reference(category="context_checkpoints", source_id="host-" + section.section_id,
                                 data=section.full_text.encode("utf-8"), extension="md")
    return {"section_id": section.section_id, "source_ref": ref, "reader": _reader(ref), "predicted": True}


def _reader(ref: Mapping[str, Any]) -> str:
    return f"read_file(root='artifact_store', path={ref['path']!r})"


def section_views(sections: Iterable[HostSection], receipts: Iterable[Mapping]) -> dict[str, str]:
    rows = {row["section_id"]: row for row in receipts}  # a receipt without the flag renders as recorded before it
    return {part.section_id: text for part in sections if part.section_id in rows
            if (text := part.view_for_receipt(rows[part.section_id], rows)) is not None}


def render_sections(sections: Iterable[HostSection], fallback: str, addressed: Mapping[str, str]) -> str:
    parts = tuple(sections)
    return "\n\n".join(addressed.get(part.section_id, part.full_text) for part in parts) if parts else fallback


def rendered_texts(core: Any, addressed: Mapping[str, str]) -> Any:
    """The core with its captured texts as sent under ``addressed``: a measure of parts reads what the request carries."""
    return replace(core, semi_stable_text=render_sections(core.semi_stable_sections, core.semi_stable_text, addressed),
                   dynamic_head_text=render_sections(core.dynamic_head_sections, core.dynamic_head_text, addressed),
                   dynamic_text=render_sections(core.dynamic_sections, core.dynamic_text, addressed))


def section_inventory(plan, mode, facts):
    """The captured host forms and their resident prices, for the ordinary view inventory.

    These are producer-owned forms selected by the shared floor, not dialogue
    units the mind can replace with a note. The reader retains the exact full source.
    """
    from ouroboros.context_fit import _governance_blocks
    from ouroboros.utils import estimate_tokens

    core = plan.core
    governance = _governance_blocks(None, core, mode='low' if mode == 'nano' else mode)
    sections = (*core.semi_stable_sections, *core.dynamic_head_sections, *core.dynamic_sections,
                *(governance[0] if not isinstance(governance[0], str) else ()))
    receipts = {row['section_id']: row for row in facts.get('host_sections_addressed', ())}
    rendered = section_views(sections, receipts.values())
    return [{'view_address': 'host:' + section.section_id, 'section_id': section.section_id,
             'kind': 'host', 'tokens': estimate_tokens(section.full_text),
             'resident_tokens': estimate_tokens(rendered.get(section.section_id, section.full_text)),
             'form': 'operative' if section.section_id in rendered else 'whole',
             'operative_tokens': estimate_tokens(section.operative_view) if section.operative_view is not None else None,
             'selected': section.section_id in rendered, 'eligible': False,
             'protected': section.section_id in PROTECTED,
             **({'source_ref': receipts[section.section_id]['source_ref']} if section.section_id in receipts else {})}
            for section in sections]


def _governance_blocks(env: Any, core: Any, *, mode: str, reference_docs: Callable) -> Tuple[Any, ...]:
    # One resident Body Map in every mode and focus (owner, 25.11.2026); no mode or binding preloads a manual
    # behind it. ``mode`` only says whether the Map may step down to its parts here: never in the owner's Max, in a
    # lowered frame for the physical window alone. The whole handbook rides only as a task's own requirement.
    del env  # the capture holds every source this block renders
    static_parts = [report_text(core.base_prompt, "system"), report_text("## BIBLE.md\n\n" + core.bible_md, "bible")]
    sections, development = reference_docs(
        context_mode=mode,
        body_map=core.body_map,
        body_map_parts=core.body_map_parts,
        required_book=core.development_required,
        development_text=core.development_md,
    )
    static_parts.extend(sections)
    static_parts.extend(core.reference_book_errors)
    return (capture_sections(static_parts, prefix="governance"), development)


def render_governance(governance: Tuple[Any, ...], addressed: Optional[Mapping[str, str]] = None) -> Tuple[str, ...]:
    """The governance blocks as sent: captured sections in their selected views, plain parts whole."""
    return tuple(render_sections(part, "", addressed or {}) if not isinstance(part, str) else part
                 for part in governance)


def with_program_history(rows, data):
    """Bind the projected data after the assignment, never as the first (owner) input or a system block."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, PROGRAM_HISTORY_KIND, bound_host_row

    tail = [row for row in rows if row.get(HOST_CONTEXT_KIND_KEY) != PROGRAM_HISTORY_KIND]
    if data:
        after = next((i + 1 for i, row in enumerate(tail) if row.get("role") == "user"), len(tail))
        tail.insert(after, bound_host_row(PROGRAM_HISTORY_KIND, data))
    return tail


def _system_blocks(core: Any, governance: Tuple[Any, ...], story: str = "", room: str = "", mode: str = "",
                   addressed: Optional[Mapping[str, str]] = None, story_whole: bool = True,
                   trace: Any = None) -> List[Dict[str, Any]]:
    # All stable items precede C; D exists only for a task's directly required handbook, in any mode, so every
    # other task spends no breakpoint on it. The mode picks the connections map's form in B. Never create an empty
    # cache boundary.
    # B declares what of it other works share (``SHARED_HEAD_KEY``): its head, and its story when that is whole.
    addressed = addressed or {}
    common, *handbook = render_governance(governance, addressed)
    semi = render_sections(core.semi_stable_sections, core.semi_stable_text, addressed)
    dynamic = render_sections(core.dynamic_sections, core.dynamic_text, addressed)
    connections = core.connections[mode == "nano"]
    shared = "\n\n".join(part for part in (semi, connections) if part)
    stable = (common, *handbook, "\n\n".join(part for part in (shared, story) if part))
    head = render_sections(core.dynamic_head_sections, core.dynamic_head_text, addressed)
    blocks = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}} for text in stable if text.strip()]
    if shared.strip() and story.strip():
        from ouroboros.llm_messages import SHARED_HEAD_KEY  # here, as its sibling key is: no transport at import time

        blocks[-1][SHARED_HEAD_KEY] = {"chars": len(shared) + 2, "story_whole": bool(story_whole)}
    changing = "\n\n".join(part for part in (head, room, dynamic) if part)
    if trace is not None:
        trace((common, handbook, (semi, connections), story, head, room, dynamic, stable, changing))
    return blocks + [{"type": "text", "text": changing}]


def capture_system_projection(core, governance, story, room, mode, addressed, receipts, whole, *, shown, facts):
    """Render normally and keep its pieces for the selected projection, without opening sources again."""
    from ouroboros.context_assembly_trace import observe_system_layout

    live = []
    system = _system_blocks(core, governance, story, room, mode, addressed, whole, trace=live.append)
    report = observe_system_layout(core, governance, addressed, receipts, live, system, story_whole=whole, shown=shown)
    return system, {**facts, '_assembly_layout': report}


def projection_fields(system, facts):
    """Keep the private observer off the projection; witness each rendered view, small and bound to its blocks.

    A later rebind may select a mode without rendering it again (a cold restore keeping Low): its witness is the
    one taken when that view's bytes were rendered, never a reread of the sources or a held renderer string."""
    memory_facts = dict(facts)
    report = memory_facts.pop('_assembly_layout', None)
    witnesses = report(system, memory_facts.get('first_contact_preference')) if report is not None else ()
    return {'memory_facts': memory_facts, 'assembly_witnesses': witnesses or ()}
