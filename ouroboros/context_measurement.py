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
MeasurementBasis = Literal["fresh_route_usage", "fresh_model_usage", "cold_estimate"]


@dataclass(frozen=True)
class MainFitMeasurement:
    route_fp: str
    round_id: str
    profile: ContextProfile
    rendered_mode: Literal["max", "low", "nano"]
    estimated_input_tokens: int  # provider density applied, or raw for an estimator window
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
    raw_input_tokens: int = 0  # the same estimate before the density
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
) -> MainFitDisposition:
    """Measure one sealed Main candidate against owner target T and route W.

    The estimate is this conversation's own: ``anchor`` (``context_anchor.main_input_anchor``) scales the raw
    estimate by the last accepted request on this route. Without one, a fresh exact-route witness
    of comparable input size calibrates it; absent that, it stays cold. Estimator windows
    always use the raw estimate. Acceptance raises a disproved stated limit,
    never a measured retention limit. The deficits,
    the reclaim goal (deficit + ``reclaim_low_water_margin``) and the reply allowance are facts for
    the round and recovery. A known window with no tool-result room starts the same refusal
    passes before sending (owner revision, 20 October 2026); an unknown window never does.
    Both deficits count the mode's reply floor or the route's larger working reserve; the reply the
    candidate will actually get is ``reply_allowance_tokens``.
    """
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.capability_evidence import canonical_evidence_root, is_known, resolve_main_token_density
    from ouroboros.context_budget import (
        OWNER_LOW_TARGET_TOKENS, OWNER_NANO_TARGET_TOKENS, context_mode_limits, reply_allowance_tokens,
    )

    raw_input = int(estimate_context_prompt_tokens(messages, tools, provider=plan.provider, reasoning_effort=reasoning_effort))
    density, basis = ((1.0, "cold_estimate") if plan.window_basis == "estimator" else
                     (float(anchor.density), "fresh_route_usage") if anchor is not None else
                      resolve_main_token_density(canonical_evidence_root(), plan.route_fp, plan.model,
                                                 estimate_tokens=raw_input, view_kind=plan.view_kind))
    estimated_input = int(math.ceil(raw_input * density))
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
    target_deficit = max(0, total - target) if target is not None else None
    capacity_deficit = max(0, total - capacity) if capacity is not None else None
    deficit = max(
        [value for value in (target_deficit, capacity_deficit) if value is not None]
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
        reply_allowance_tokens=reply_allowance_tokens(
            caller_max_tokens=ceiling, nano=nano, owner_nano=profile == "owner_nano",
            input_tokens=estimated_input, raw_input_tokens=raw_input, window_tokens=capacity),
        anchor_input_tokens=anchor.input_tokens if anchor is not None else None,
        anchor_round=anchor.round if anchor is not None else None,
        anchor_rewritten=bool(anchor is not None and anchor.rewritten),
        accepted_input_tokens=accepted_input_tokens,
        window_reading_tokens=reading,
        window_witness_bound=plan.window_witness_bound,
        window_observation_fp=current.route_fp,
    )
    action: Literal["send", "send_target_miss"] = (
        "send_target_miss" if target_deficit and target_deficit > 0 else "send")
    return MainFitDisposition(
        measurement=measurement,
        action=action,
        automatic_pass_used=bool(automatic_pass_used),
        predicted_capacity_miss=bool(capacity_deficit and capacity_deficit > 0),
    )
