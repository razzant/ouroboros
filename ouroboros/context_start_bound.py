"""The lower bound of a start: what every request a start could send holds whole, priced before its memory is read.

A start of the mind's own focus (not a declared start, a child, a nanny or a held-note resume) always carries, whole,
its system prompt and constitution, its tier-0 sections (identity, scratchpad, the authored overview where its view
holds knowledge: ``context_layout.TIER0_ALWAYS_FULL``), the marks it keeps in view (counted only when the chronicle is
already active; this module never activates it), the connections text (of its two forms the one with fewer escaped
characters, the measure the fit prices it by: no mode carries less; Nano's leaves a server's instructions behind its
reader, ``tool_connections.nano_connections_type``) and the Nano frame's resident tool schemas,
beside Nano's reply reserve. Nano is the last physical rescue of the fit (``context_fit._view_projections``), so when
that frame already exceeds the route's known window no request of this start can fit: it is refused before the memory
capture (``context.build_context_fit_plan``), with no memory read.

Everything else a request holds is left out, and each omission keeps this a lower bound: the assignment (priced as
empty, so ``context_input_selection.rescued_start`` keeps its smaller inputs), book maps, the knowledge outline,
skills, helpers, runtime facts, memory beyond the marks, a call's session wrapper and call tools, separators and
envelopes. The texts are priced as the fitter prices them (``memory_view.json_chars`` summed and floor-divided by
four, plus ``tool_schema_tokens`` of the Nano selection ``context_fit._frame_choices`` makes), against the Nano reserve
and window of the same route frame (``context_fit.route_window``). The density is 1.0 on an estimator-basis window and
otherwise the lowest fresh witness of the route (``capability_evidence.lowest_main_token_density``), never the newest
one the fit applies: the bound never refuses a start that a completed capture of the same state would fit.

Order of reads: the cheap exclusions (a declared input, a held-note resume, a child), then the route, looked at
without consuming or fetching: an arrived ``RouteQuestion`` answer (``peek``), or any other resolver asked with
``allow_fetch=False``. No answer, an unknown window or a failed read means no bound, and nothing of the start is read
here; only a known window reads the view spec (a delegated role then gets no bound), the frame texts and the density.
An answer taken beside the build never refuses by itself (owner, 12.11): the same question is asked again at the usual
point and only a second refusal stands. Any failure inside the bound other than its refusal means no bound; the start
goes on to the full fit unchanged.

Where it applies: a resolver asked without fetching (a call's route, ``turn.resolve_route``), a ``RouteQuestion``
whose answer has arrived (a Claudexor text start), and a question with no key (every other text route), whose resolver
is read without fetching (cached facts, never the network). For that one a refusal stands only on its second answer,
the fit's own read of the route (``RouteQuestion.__call__``, a network read only where the fit makes it), so a cached
or declared window only triggers the question: the number the full fit would use decides.

The refusal is ``context_fit.MinimumViewUnfit`` with ``memory_measured`` False and the frame's named ``parts``;
``context_view_facts.refusal_parts`` names the parts of the fitter's own refusals the same way. Those name core, assignment, control-tool
and work-room nonfit, protected owner words beside the minimum view, or a directly required handbook; a cold start's
Nano frame whose fixed part alone overflows is refused there without rendering memory, and says so
(``memory_measured`` False) rather than reporting an unmeasured memory as 0.

Residual: the marks are counted from a chronicle readable here. A capture that fails anywhere and degrades (any
unreadable memory source) shows no marks, so where the marks decide the refusal, that memory-less start would have
been admitted on this route and is refused. The refusal is true for the healthy state, and its Details name the marks.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
from ouroboros.context_fit import DENSITY_VIEW_KINDS, MinimumViewUnfit, RouteQuestion, _frame_choices, route_window
from ouroboros.context_measurement import tool_schema_tokens
from ouroboros.context_view_facts import _largest_first
from ouroboros.contracts.task_contract import task_input_sources
from ouroboros.memory_view import json_chars

log = logging.getLogger(__name__)



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
    preferred = str(preferred or "max").strip().lower()
    preferred = preferred if preferred in {"low", "max", "nano"} else "max"
    peek = getattr(resolver, "peek", None)
    keyless = isinstance(resolver, RouteQuestion) and resolver.key is None  # a route no engine is asked about
    answer = (resolver.resolver({**task, **resolver.route_fields}, allow_fetch=False) if keyless
              else peek() if callable(peek) else resolver(task, allow_fetch=False))
    first = _route_frame(answer, task, tool_schemas=tool_schemas, nano_base=nano_base, effort=effort, preferred=preferred)
    if first is None:  # no known window: nothing of the start is read here
        return None
    view = spec()
    if view.role in DELEGATED_FOCUS_ROLES:
        return None
    texts = frame_texts(view)
    chars = sum(json_chars(text) for _name, text in texts)
    if not _refuses(chars, first):
        return None
    # An answer taken beside the build never refuses by itself: the same question at the usual point decides.
    # A key-less question's second answer is the fit's own read (``RouteQuestion.__call__``): that number decides.
    again = (resolver.again(task) if isinstance(resolver, RouteQuestion) and not keyless
             else resolver(task, allow_fetch=True))
    second = _route_frame(again, task, tool_schemas=tool_schemas, nano_base=nano_base, effort=effort, preferred=preferred)
    if second is None or not _refuses(chars, second):
        return None
    reserve, window, ratio, schemas = second
    frame = chars // 4 + schemas
    parts = [(name, json_chars(text) // 4) for name, text in texts] + [("tool schemas", schemas)]
    refusal = MinimumViewUnfit("Core, resident tools and reserve cannot fit before memory", fixed_tokens=frame,
                               memory_tokens=0, reserve_tokens=reserve, window_tokens=window, ratio=ratio,
                               work_room_tokens=0, memory_measured=False, parts=_largest_first(parts))
    # The serving program the fit's refusal names (``context_fit``), for its guidance (``agent_dispatch``).
    refusal.server_program = ((again[0].get("_resolved_target") or {}).get("ollama_record") or {}).get("protocol")
    return refusal


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
