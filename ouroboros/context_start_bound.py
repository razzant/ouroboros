"""The lower bound of a start: what every request a start could send holds whole, priced before its memory is read.

A start of the mind's own focus (not a declared start, a child, a nanny or a held-note resume) always carries, whole,
its system prompt and constitution, its tier-0 sections (identity, scratchpad, the authored overview where its view
holds knowledge: ``context_layout.TIER0_ALWAYS_FULL``), the marks it keeps in view (counted only when the chronicle is
already active; this module never activates it), the connections text (the smaller of its two forms: it has no shorter
one) and the Nano frame's resident tool schemas, beside Nano's reply reserve. Nano is the last physical rescue of the
fit (``context_fit._view_projections``), so when that frame already exceeds the route's known window no request of
this start can fit: it is refused before the memory capture (``context.build_context_fit_plan``), with no memory read.

Everything else a request holds is left out, and each omission keeps this a lower bound: the assignment (priced as
empty, so ``context_input_selection.rescued_start`` keeps its smaller inputs), book maps, the knowledge outline,
skills, helpers, runtime facts, memory beyond the marks, a call's session wrapper and call tools, separators and
envelopes. The texts are priced as the fitter prices them (``memory_view.json_chars`` summed and floor-divided by
four, plus ``tool_schema_tokens`` of the Nano selection ``context_fit._frame_choices`` makes), against the Nano reserve
and window of the same route frame (``context_fit.route_window``). The density is 1.0 on an estimator-basis window and
otherwise the lowest fresh witness of the route (``capability_evidence.lowest_main_token_density``), never the newest
one the fit applies: the bound never refuses a request the full fit admits.

The route is looked at without consuming or fetching: an arrived ``RouteQuestion`` answer (``peek``), or any other
resolver asked with ``allow_fetch=False``; no answer, an unknown window or a failed read means no bound. An answer
taken beside the build never refuses by itself (owner, 12.11): the same question is asked again at the usual point and
only a second refusal stands. Any failure inside the bound other than its refusal means no bound; the start goes on to
the full fit unchanged. The refusal is ``context_fit.MinimumViewUnfit`` with ``memory_measured`` False and the frame's
named ``parts``; ``refusal_parts`` names the parts of the fitter's own refusals the same way. Those name core,
assignment, control-tool and work-room nonfit, protected owner words beside the minimum view, or a directly required
handbook; a cold start's Nano frame whose fixed part alone overflows is refused there without rendering memory, and
says so (``memory_measured`` False) rather than reporting an unmeasured memory as 0.

Residual: a chronicle readable here and unreadable seconds later at the capture would have had its marks counted
though the degraded view would not show them; the refusal is still true for the state it measured.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
from ouroboros.context_fit import DENSITY_VIEW_KINDS, MinimumViewUnfit, RouteQuestion, _frame_choices, route_window
from ouroboros.context_measurement import tool_schema_tokens
from ouroboros.contracts.task_contract import task_input_sources
from ouroboros.memory_view import json_chars

log = logging.getLogger(__name__)

_SECTION_NAMES = {"shared_understanding": "overview"}
_GOVERNANCE_NAMES = {"system": "system prompt", "bible": "constitution"}
_ANONYMOUS_PREFIXES = {"stable", "dynamic", "head", "governance"}  # ``section_floor.capture_sections``' positional ids


def refuse_before_capture(task: Dict[str, Any], ctx: Any, *, spec: Callable[[], Any],
                          frame_texts: Callable[[Any], List[Tuple[str, str]]], route_resolver: Callable[..., Any],
                          tool_schemas: Optional[List[Dict[str, Any]]], nano_base: Optional[Tuple[str, ...]],
                          reasoning_effort: str, preferred: str) -> None:
    """Raise ``MinimumViewUnfit`` when this start's frame cannot fit its route's known window; else return.

    ``spec`` reads the start's ``ViewSpec``; ``frame_texts(spec)`` reads its named whole texts (``context`` owns both
    readers). Read only after the route names a window, so a start without one reads nothing here.
    """
    try:
        refusal = _refusal(task, ctx, spec, frame_texts, route_resolver, tool_schemas=tool_schemas, nano_base=nano_base,
                           effort=reasoning_effort, preferred=preferred)
    except Exception:
        log.debug("Start bound unavailable; the full fit decides", exc_info=True)
        return
    if refusal is not None:
        raise refusal


def _refusal(task, ctx, spec, frame_texts, resolver, *, tool_schemas, nano_base, effort, preferred):
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    if (task_input_sources(task) == "declared" or getattr(ctx, "_held_note_capture", None) is not None
            or str(task.get("delegation_role") or metadata.get("delegation_role") or "") == "subagent"):
        return None
    view = spec()
    if view.role in DELEGATED_FOCUS_ROLES:
        return None
    preferred = str(preferred or "max").strip().lower()
    preferred = preferred if preferred in {"low", "max", "nano"} else "max"
    peek = getattr(resolver, "peek", None)
    first = _route_frame(peek() if callable(peek) else resolver(task, allow_fetch=False), task, tool_schemas=tool_schemas,
                         nano_base=nano_base, effort=effort, preferred=preferred)
    if first is None:
        return None
    texts = frame_texts(view)
    chars = sum(json_chars(text) for _name, text in texts)
    if not _refuses(chars, first):
        return None
    # An answer taken beside the build never refuses by itself: the same question at the usual point decides.
    again = resolver.again(task) if isinstance(resolver, RouteQuestion) else resolver(task, allow_fetch=True)
    second = _route_frame(again, task, tool_schemas=tool_schemas, nano_base=nano_base, effort=effort, preferred=preferred)
    if second is None or not _refuses(chars, second):
        return None
    reserve, window, ratio, schemas = second
    frame = chars // 4 + schemas
    parts = [(name, json_chars(text) // 4) for name, text in texts] + [("tool schemas", schemas)]
    return MinimumViewUnfit("Core, resident tools and reserve cannot fit before memory", fixed_tokens=frame,
                            memory_tokens=0, reserve_tokens=reserve, window_tokens=window, ratio=ratio,
                            work_room_tokens=0, memory_measured=False, parts=_largest_first(parts))


def _route_frame(answer: Any, task: Mapping[str, Any], *, tool_schemas, nano_base, effort: str,
                 preferred: str) -> Optional[Tuple[int, int, float, int]]:
    """``(Nano reserve, Nano window, density, Nano tool-schema tokens)`` of one route answer, or None without a
    known window or a readable density."""
    if answer is None:
        return None
    route, evidence = answer
    output_reserve, working_reserve, basis, _bound, record = route_window(route, evidence, preferred)
    if record.usable is None:
        return None
    tools_for, limits = _frame_choices(tool_schemas, resident=False, nano_base=nano_base, low_tools=None,
                                       nano_tools=None, tool_frames=None, window_record=record,
                                       window_tokens=record.total_equivalent, admitted=False, preferred=preferred,
                                       output_reserve=output_reserve, working_reserve_tokens=working_reserve)
    _target, reserve, window, _record = limits("nano")
    ratio = 1.0 if basis == "estimator" else _lowest_density(task, route, evidence)
    if ratio is None:
        return None
    schemas = tool_schema_tokens(list(tools_for("nano").schemas), provider=str(route.get("provider") or ""),
                                 reasoning_effort=effort)
    return reserve, window, ratio, schemas


def _lowest_density(task: Mapping[str, Any], route: Mapping[str, Any], evidence: Any) -> Optional[float]:
    from ouroboros.capability_evidence import canonical_evidence_root, lowest_main_token_density
    from ouroboros.memory_view import view_role

    # The witness view the fit reads (``context_fit.build_context_fit_plan``).
    meta = task.get("task_metadata") if isinstance(task.get("task_metadata"), dict) else {}
    role = view_role({**task, "metadata": task.get("metadata") or meta})
    return lowest_main_token_density(canonical_evidence_root(), str(evidence.route_fp or ""), str(route.get("model") or ""),
                                     view_kind=DENSITY_VIEW_KINDS.get(role, role))


def _refuses(chars: int, frame: Tuple[int, int, float, int]) -> bool:
    reserve, window, ratio, schemas = frame
    return math.ceil((chars // 4 + schemas) * ratio) + reserve > window


def _largest_first(parts: Iterable[Tuple[str, int]]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted(((name, size) for name, size in parts if size), key=lambda part: (-part[1], part[0])))


def refusal_parts(core: Any, sections: Iterable[Any], receipts: Iterable[Mapping[str, Any]], schemas: Iterable[Any],
                  shown: Any, user_content_json: str, *, mode: str, provider: str,
                  reasoning_effort: str) -> Tuple[Tuple[str, int], ...]:
    """Named sizes, in estimator tokens and largest first, of what a request the fitter refused holds.

    Host sections by their producer's name (governance as ``system prompt`` and ``constitution``, a report by its
    family, positional sections summed as ``other host sections``) in the forms the walk chose; the connections text,
    the assignment and the tool schemas; the story and the room only when memory was rendered (``shown``).
    """
    from ouroboros import memory_floor as floor
    from ouroboros.section_floor import section_views

    sections = tuple(sections)
    views = section_views(sections, receipts)
    sizes: Dict[str, int] = {}
    for part in sections:
        family, (prefix, _sep, index) = getattr(part, "report_family", ""), part.section_id.rpartition("_")
        name = (_GOVERNANCE_NAMES.get(family, family) if family else "other host sections"
                if prefix in _ANONYMOUS_PREFIXES and index.isdigit() else _SECTION_NAMES.get(part.section_id, part.section_id))
        sizes[name] = sizes.get(name, 0) + json_chars(views.get(part.section_id, part.full_text)) // 4
    for parts, text in ((core.semi_stable_sections, core.semi_stable_text), (core.dynamic_sections, core.dynamic_text),
                        (core.dynamic_head_sections, core.dynamic_head_text)):
        if not parts and text:  # a core captured as plain text: its sections are drawn from it whole
            sizes["other host sections"] = sizes.get("other host sections", 0) + json_chars(text) // 4
    sizes["connections"] = json_chars(core.connections[mode == "nano"]) // 4
    sizes["assignment"] = len(user_content_json) // 4
    sizes["tool schemas"] = tool_schema_tokens(list(schemas), provider=provider, reasoning_effort=reasoning_effort)
    if shown is not floor.NO_VIEW:
        sizes["story"], sizes["room"] = floor.tokens_of(shown.story_chars), floor.tokens_of(shown.room_chars)
    return _largest_first(sizes.items())
