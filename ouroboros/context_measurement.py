"""Measurement of a sealed Main request in its route's units and working room.

Captured-context projection remains in context_fit; this leaf owns the resulting
round facts, deficits and reply allowance. The public context_fit names stay available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from ouroboros.context_anchor import InputAnchor
    from ouroboros.context_fit import ContextFitPlan

ContextProfile = Literal["owner_max", "owner_low", "owner_nano", "task_local_low", "task_local_nano"]
MeasurementBasis = Literal["fresh_route_usage", "fresh_model_usage", "refused_input", "cold_estimate"]
#: A view receipt's estimator tokens of the start note a directly required handbook brings (``context_fit``).
BOOK_NOTE_KEY = "required_book_note_tokens"


def required_book_addition(plan: Any, mode: str) -> int:
    """What this task's directly required handbook adds to ``mode``'s view of ``plan``, its start note included
    (estimator tokens; 0 without a captured core): the window charges it, and no automatic choice is sized by it."""
    core = getattr(plan, "core", None)
    return 0 if core is None else core.required_book_tokens(mode) + int(
        (plan.projection(mode).memory_facts or {}).get(BOOK_NOTE_KEY) or 0)


@dataclass(frozen=True)
class MainFitMeasurement:
    route_fp: str
    round_id: str
    profile: ContextProfile
    rendered_mode: Literal["max", "low", "nano"]
    # Anchored: the accepted input carried to this transcript, U + (P' - P) * U / B (``InputAnchor.measure``);
    # otherwise raw_input_tokens at measurement_density (a refusal's, a fresh witness's or cold; 1 on an estimator window).
    estimated_input_tokens: int
    response_reserve_tokens: int  # the mode's reply floor, raised by any declared working reserve
    target_total_tokens: Optional[int]
    capacity_total_tokens: Optional[int]
    measurement_basis: MeasurementBasis
    measurement_density: float
    target_deficit_tokens: Optional[int]
    capacity_deficit_tokens: Optional[int]
    reclaim_goal_tokens: int
    # Low-water margin the goal carries ABOVE the deficit (0 without a deficit):
    # the pass is deficit-triggered but sized to land below the boundary.
    low_water_margin_tokens: int = 0
    raw_input_tokens: int = 0  # P': the fit estimator's count of this transcript (images at the proxy), no density
    # The reply allowance Main PLANS for this candidate (``context_budget.reply_allowance_tokens``);
    # the send finalizer computes the sent value on the sealed candidate with the same rule.
    reply_allowance_tokens: int = 0
    # Provenance of the estimate (``InputAnchor``): the provider-reported input of the last request this
    # model accepted in this conversation and its round, or None before this task has an anchor.
    anchor_input_tokens: Optional[int] = None
    anchor_round: Optional[int] = None
    anchor_rewritten: bool = False
    # Accepted input is a lower bound; a measured retention limit remains independent.
    accepted_input_tokens: Optional[int] = None
    window_reading_tokens: Optional[int] = None
    view_kind: str = ""
    window_witness_bound: bool = False
    window_observation_fp: str = ""
    window: Dict[str, Any] = field(default_factory=dict)
    # What a directly required handbook adds to this candidate, its start note included (density applied): it
    # counts against the window, while nothing automatic narrows or shrinks for it and no owner target sizes it.
    required_book_tokens: int = 0
    preferred_mode: str = ""  # the plan's owner mode at this measurement
    mode_cause: str = ""  # why ``rendered_mode`` is lower than it (``mode_cause``); "" when it is not

    @property
    def reprojection_input_tokens(self) -> Optional[int]:
        """Carry the anchored input into the floor's estimator units, rounding conservatively.

        Unanchored, refused-density and estimator-window measurements already price the
        transcript directly; they need no preparation offset at reprojection.
        """
        if self.anchor_input_tokens is None or self.measurement_basis != "fresh_route_usage":
            return None
        return math.ceil(self.estimated_input_tokens / self.measurement_density)


@dataclass(frozen=True)
class MainFitDisposition:
    measurement: MainFitMeasurement
    action: Literal["send", "send_target_miss"]
    automatic_pass_used: bool
    predicted_capacity_miss: bool


def reclaim_low_water_margin(
    target_total_tokens: Optional[int], capacity_total_tokens: Optional[int],
) -> int:
    """Tokens a reclaim pass lands BELOW the binding boundary: ceil(boundary / divisor).

    The boundary is the smaller known positive one of owner target T and route
    capacity W; 0 when neither is known. The divisor is read at call time so
    the SSOT constant stays the one place to change it.
    """
    from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR

    known = [int(value) for value in (target_total_tokens, capacity_total_tokens)
             if value is not None and int(value) > 0]
    return int(math.ceil(min(known) / RECLAIM_LOW_WATER_DIVISOR)) if known else 0


def room_fact(measurement, components, *, source="", memory=""):
    """Passive measured room, before this facts row; usable by either actor driver."""
    from ouroboros.context_budget import bound_host_row
    window = measurement['capacity_total_tokens']
    reserve, used = measurement['response_reserve_tokens'], measurement['estimated_input_tokens']
    free = window - reserve - used if window is not None else 'unknown'
    memory = f" | {memory} |" if memory else ""
    largest = ', '.join(f'{name} ~{size:,}' for name, size in components)
    return bound_host_row('ouroboros_context_facts', (
        f"[CONTEXT_FACTS] Fresh view before this facts row: window {window if window is not None else 'unknown'}; "
        f"reply/working reserve {reserve}; input ~{used}; free ~{free}. Largest residents: {largest}.{memory}{source}"))


def measure_main_fit(
    plan: ContextFitPlan,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    *,
    profile: ContextProfile,
    rendered_mode: Literal["max", "low", "nano"],
    round_id: str,
    automatic_pass_used: bool = False,
    reasoning_effort: str = "",
    anchor: Optional[InputAnchor] = None,
    accepted_input_tokens: Optional[int] = None,
    refused_density: Optional[float] = None,
) -> MainFitDisposition:
    """Measure one sealed Main candidate against owner target T and route W.

    The estimate is this conversation's own: ``anchor`` (``context_anchor.main_input_anchor``) carries the last
    accepted request's input forward (``InputAnchor.measure``); while a size refusal's passes, and a renewal after
    them, size that refusal, its own ``refused_density`` (``context_anchor.refused_request_density``, never
    recorded) does instead. Without either, a fresh exact-route witness
    of comparable input size calibrates it; absent that, it stays cold. Estimator windows
    always use the raw estimate. Acceptance raises a disproved stated limit,
    never a measured retention limit. The deficits,
    the reclaim goal (deficit + ``reclaim_low_water_margin``) and the reply allowance are facts for
    the round and recovery. A known window with no tool-result room starts the same refusal
    passes before sending (owner revision, 20 October 2026); an unknown window never does.
    Both deficits count the mode's reply floor or the route's larger working reserve; the reply the
    candidate will actually get is ``reply_allowance_tokens``. The recorded target facts compare the
    whole candidate; the reclaim goal, the reply and tool delivery are sized without a directly required
    handbook (``required_book_addition``), which only the capacity deficit charges, beside that reply.
    """
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.capability_evidence import canonical_evidence_root, is_known, resolve_main_token_density
    from ouroboros.context_budget import (
        OWNER_LOW_TARGET_TOKENS, OWNER_NANO_TARGET_TOKENS, context_mode_limits, reply_allowance_tokens,
    )

    raw_input = int(estimate_context_prompt_tokens(messages, tools, provider=plan.provider, reasoning_effort=reasoning_effort))
    anchored = anchor is not None and plan.window_basis != "estimator" and not refused_density
    density, basis = ((1.0, "cold_estimate") if plan.window_basis == "estimator" else
                      (float(refused_density), "refused_input") if refused_density else
                     (float(anchor.density), "fresh_route_usage") if anchor is not None else
                      resolve_main_token_density(canonical_evidence_root(), plan.route_fp, plan.model,
                                                 estimate_tokens=raw_input, view_kind=plan.view_kind))
    # The anchor carries its accepted U: a send's image preparation stays counted, never priced twice.
    estimated_input = anchor.measure(raw_input) if anchored else int(math.ceil(raw_input * density))
    ceiling = int(plan.output_reserve_tokens or 0)
    nano = profile.endswith("_nano")
    reserve_args = ({"working_reserve_tokens": plan.working_reserve_tokens}
                    if plan.working_reserve_tokens is not None else {})
    reserve = context_mode_limits("nano" if nano else "max", plan.preferred_mode, ceiling, **reserve_args)[1]
    total = estimated_input + reserve
    target = (OWNER_NANO_TARGET_TOKENS if profile == "owner_nano"
              else OWNER_LOW_TARGET_TOKENS if profile == "owner_low" else None)
    from ouroboros.context_window import resolve_window

    reading = int(plan.window_tokens or 0) if is_known(plan, require_fresh=True) else None
    current = resolve_window(plan, reserve, bound=plan.window_witness_bound,
                             accepted_input_tokens=accepted_input_tokens)
    capacity = current.total_equivalent if current.usable is not None else None
    if capacity is None:
        reading = None
    raw_book = required_book_addition(plan, rendered_mode)
    book = math.ceil(raw_book * density)
    # A directly required handbook keeps the reply the same candidate without it gets (Nano's is the only elastic one):
    # a known window holds both or its deficit says so; nothing here shrinks for the book (owner 1.11.2026).
    reply = reply_allowance_tokens(caller_max_tokens=ceiling, nano=nano, owner_nano=profile == "owner_nano",
                                   input_tokens=estimated_input - book, raw_input_tokens=raw_input - raw_book, window_tokens=capacity)
    target_deficit = max(0, total - target) if target is not None else None
    capacity_deficit = max(0, total + (max(0, reply - reserve) if book else 0) - capacity) if capacity is not None else None
    economic_deficit = None if target_deficit is None else max(0, target_deficit - book)
    deficit = max(
        [value for value in (economic_deficit, capacity_deficit) if value is not None]
        or [0]
    )
    margin = reclaim_low_water_margin(target, capacity) if deficit > 0 else 0
    goal = deficit + margin
    from ouroboros.context_view_facts import window_projection
    measurement = MainFitMeasurement(
        window=window_projection(current),
        view_kind=plan.view_kind,
        route_fp=str(plan.route_fp or ""),
        round_id=str(round_id or ""),
        profile=profile,
        rendered_mode=rendered_mode,
        estimated_input_tokens=estimated_input,
        response_reserve_tokens=reserve,
        target_total_tokens=target,
        capacity_total_tokens=capacity,
        measurement_basis=basis,
        measurement_density=density,
        target_deficit_tokens=target_deficit,
        capacity_deficit_tokens=capacity_deficit,
        reclaim_goal_tokens=goal,
        low_water_margin_tokens=margin,
        raw_input_tokens=raw_input,
        reply_allowance_tokens=reply,
        anchor_input_tokens=anchor.input_tokens if anchor is not None else None,
        anchor_round=anchor.round if anchor is not None else None,
        anchor_rewritten=bool(anchor is not None and anchor.rewritten),
        accepted_input_tokens=accepted_input_tokens,
        window_reading_tokens=reading,
        window_witness_bound=plan.window_witness_bound,
        window_observation_fp=current.route_fp,
        required_book_tokens=book, preferred_mode=str(plan.preferred_mode or ""), mode_cause=mode_cause(plan, rendered_mode),
    )
    action: Literal["send", "send_target_miss"] = (
        "send_target_miss" if target_deficit and target_deficit > 0 else "send")
    return MainFitDisposition(
        measurement=measurement,
        action=action,
        automatic_pass_used=bool(automatic_pass_used),
        predicted_capacity_miss=bool(capacity_deficit and capacity_deficit > 0),
    )


def mode_cause(plan: Any, rendered_mode: str) -> str:
    """Why a candidate renders below the plan's owner mode, from the plan's own facts: ``""`` when it does not; the
    start's own mode carries the start's cause (``ContextFitPlan.start``, ``memory_floor.start_cause``: ``memory``,
    ``route`` or ``window``), which no reprojection or route rebind replaces; any other lower mode, or a start
    lowered by no mode switch, is ``after_start`` (taken by the task after its start). A restored task whose start
    was not carried (``context_fit.UNKNOWN_START``) names no cause: ``""`` beside a lower mode is unknown."""
    from ouroboros.memory_floor import start_cause

    if rendered_mode == str(getattr(plan, "preferred_mode", "") or "") or not hasattr(plan, "projection"):
        return ""
    started, cause = getattr(plan, "start", None) or ("", "")
    if started == "unknown":  # a restore whose continuation did not carry the start: why it is lower is unknown
        return ""
    if not started:  # a plan no start fit built: its current selection and that projection's facts
        started = str(getattr(plan, "initial_mode", "") or "")
        cause = start_cause(plan.projection(started).memory_facts) if started else ""
    return cause if rendered_mode == started and cause else "after_start"
