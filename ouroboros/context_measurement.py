"""Measurement of a sealed Main request in its route's units and working room.

Captured-context projection stays in context_fit. This leaf owns the shared wire
estimate and the resulting round facts, deficits and reply allowance. The public
context_fit names stay available.
"""
from __future__ import annotations

import json
import math
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from ouroboros.context_timing import counted
from ouroboros.utils import estimate_tokens

if TYPE_CHECKING:
    from ouroboros.context_anchor import InputAnchor
    from ouroboros.context_fit import ContextFitPlan
    from ouroboros.context_window import WindowRecord
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

ContextProfile = Literal["owner_max", "owner_low", "owner_nano", "task_local_low", "task_local_nano"]
MeasurementBasis = Literal["fresh_route_usage", "fresh_model_usage", "refused_input", "cold_estimate"]
#: A view receipt's estimator tokens of the start note a directly required handbook brings (``context_fit``).
BOOK_NOTE_KEY = "required_book_note_tokens"
FIRST_CONTACT_START = {True: "first_contact", "maps": "first_contact_maps"}  # not a physical mode cause


def _compact_json(value: Any) -> str:
    """The request estimator's one serialization (``estimate_context_prompt_tokens``): compact, keys sorted."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _custom_ladder(provider: str, reasoning_effort: str) -> bool:
    """Whether a send on ``provider`` at ``reasoning_effort`` may take direct OpenAI reasoning's custom ladder."""
    return (str(provider or "").strip().lower() == "openai"
            and str(reasoning_effort or "").strip().lower() not in {"", "none"})


def effort_lengthens_request(provider: str, before: str, after: str) -> bool:
    """Whether a send's move from effort ``before`` to ``after`` can lengthen the request a fit priced: only opening
    the custom ladder adds a shape beside the canonical one (``_request_shapes``); any other move keeps or drops it."""
    return _custom_ladder(provider, after) and not _custom_ladder(provider, before)


def _request_shapes(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], provider: str,
                    reasoning_effort: str) -> Tuple[Tuple[Any, Any], ...]:
    """Every ``(messages, tools)`` shape this route may send: direct OpenAI reasoning adds its custom ladder.
    Other routes read canonical data without borrowing the public helper's owned deep copies."""
    if _custom_ladder(provider, reasoning_effort):
        from ouroboros.openai_chat_dispatch import direct_openai_context_projections

        return direct_openai_context_projections(messages, tools, provider=provider, reasoning_effort=reasoning_effort)
    return ((messages, [dict(item) for item in tools or []]),)


def tool_schema_tokens(tools: Optional[List[Dict[str, Any]]], *, provider: str = "", reasoning_effort: str = "") -> int:
    """chars/4 estimate of the TOOL-SCHEMA segment of a prompt: the ``tools`` member as the request estimator
    serializes it, in its longest shape for this provider and effort (``_request_shapes``).

    One seam for every consumer that has to account for the schemas: they are sent
    on the wire beside ``messages``, so any measure built from the transcript alone
    silently omits them (~148K chars / ~37K tokens on the submarine traces — enough
    to make an emergency-compaction trigger fire a whole tool envelope late). The
    density witness calibrates the compact count (``MAIN_DENSITY_BASIS``); separator
    spaces priced the same schemas about 3% above it (15.11.2026).
    """
    if not tools:
        return 0
    return max(estimate_tokens(_compact_json(shape)) for _rows, shape in _request_shapes([], tools, provider, reasoning_effort))


def payload_prompt_rows(prompt_payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """A physical request's input as estimator rows: Chat/Messages ``messages`` or Responses ``input`` items
    (a bare string input is one user row), a top-level ``system``/``instructions`` leading; None without either."""
    rows = prompt_payload.get("messages")
    if not isinstance(rows, list):
        rows = prompt_payload.get("input")
        rows = [{"role": "user", "content": rows}] if isinstance(rows, str) else rows
    if not isinstance(rows, list):
        return None
    system = prompt_payload.get("system") or prompt_payload.get("instructions")
    return [{"role": "system", "content": system}, *rows] if isinstance(system, (str, list)) and system else list(rows)


def bounded_prompt_tokens_for_payload(prompt_payload: Dict[str, Any], fallback_chars: int) -> int:
    """The density witness's basis (``capability_evidence.MAIN_DENSITY_BASIS``): the
    fit estimator's own token count for a request payload (``payload_prompt_rows``
    + tools, images at the proxy, a top-level ``system`` — the Messages API's separate
    field — counted as a leading system message), or ``fallback_chars // 4``
    when there is no input list. Kept beside the estimator so the two can
    never diverge; ``estimate_message_chars`` dropped tool_call objects and made
    density ~1.4x high on the tool-heavy shape, and an uncounted ``system``
    over-sized the direct-Anthropic reply by its whole length (measure_main_fit
    and the send finalizer multiply THIS quantity)."""
    try:
        rows = payload_prompt_rows(prompt_payload)
        if rows is not None:
            return int(estimate_context_prompt_tokens(
                rows, prompt_payload.get("tools") or prompt_payload.get("functions")))
    except Exception:
        pass
    return max(0, int(fallback_chars) // 4)


def messages_carry_native_images(messages: Any) -> bool:
    """Whether a message carries an image part the proxy bounds instead of pricing."""
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and str(part.get("type") or "") in {"image", "image_url"}
            for part in content
        ):
            return True
    return False


def estimate_context_prompt_tokens(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    provider: str = "",
    reasoning_effort: str = "",
    chars: bool = False,
) -> int:
    """Estimate the complete inspectable context shape with bounded images (``chars``: its JSON characters)."""
    from ouroboros.anthropic_native_custody import (
        context_custody_proxy,
        custody_private_key,
    )
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY, REPLY_SOURCE_SHA_KEY, IMAGE_BLOCK_CHAR_EQUIVALENT

    def project(value: Any, *, part: bool = False, envelope: bool = False) -> Any:
        if isinstance(value, list):
            return [project(item, part=part, envelope=envelope) for item in value]
        if not isinstance(value, dict):
            return value
        private = {key: item for key, item in value.items() if custody_private_key(key)}
        if private:
            value = {**{key: item for key, item in value.items() if key not in private},
                     **context_custody_proxy(private)}
        media = value.get("inlineData")
        if part and (value.get("type") in ("image", "image_url", "input_image")
                     or isinstance(media, dict) and str(media.get("mimeType") or "").startswith("image/")):
            return {"type": value.get("type") or "image", "image_token_proxy": "#" * IMAGE_BLOCK_CHAR_EQUIVALENT}
        # Descend through chat content and the two realtime event envelopes. Function arguments,
        # tool schemas and other JSON are opaque even when their keys resemble an image block.
        row = dict(value)
        row.pop("_context_capsule", None)
        # A provider's sealed setup may carry a Chat-shaped messages envelope
        # rather than events. Only recognize that envelope at the input boundary.
        if envelope and "role" not in row and isinstance(row.get("messages"), list):
            row["messages"] = project(row["messages"])
        if row.get("type") == "function_call_output" and isinstance(row.get("output"), list):
            row["output"] = project(row["output"], part=True)
        for key in ("content", "parts", "events", "item", "clientContent", "turns"):
            child = row.get(key)
            if isinstance(child, list) or key in ("item", "clientContent") and isinstance(child, dict):
                row[key] = project(child, part=key in ("content", "parts"))
        return row

    from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY, REVIEW_CONTEXT_INDEX_KEY
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    from ouroboros.llm_messages import DISPLAY_KEY, HOST_ONLY_MESSAGE_KEYS, STABLE_PREFIX_BLOCKS_KEY

    # These top-level host receipts remain canonical but are absent from sends (the leading system message's stable
    # block count included: every send copy pops it, so the density witness never counted it).
    # Do not strip same-named keys inside native payloads or user/tool arguments.
    hidden = {HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY, REPLY_SOURCE_SHA_KEY, REVIEW_HISTORY_MESSAGE_KEY, REVIEW_CONTEXT_INDEX_KEY,
              TOOL_RESULT_RECORD_KEY, STABLE_PREFIX_BLOCKS_KEY, DISPLAY_KEY}
    messages = [{key: value for key, value in message.items() if key not in hidden} for message in messages]
    messages = [{**message, "content": [
        {key: value for key, value in block.items() if key not in HOST_ONLY_MESSAGE_KEYS}
        if isinstance(block, dict) else block for block in message["content"]]}
        if isinstance(message.get("content"), list) else message for message in messages]

    def serialized(projected_messages: Any, projected_tools: Any) -> str:
        text = _compact_json({"messages": project(projected_messages, envelope=True), "tools": projected_tools})
        counted("request_serializations", key=hash(text))
        return text

    longest = max(len(serialized(*shape)) for shape in _request_shapes(messages, tools, provider, reasoning_effort))
    return longest if chars else max(1, (longest + 3) // 4)


def required_book_addition(plan: Any, mode: str) -> int:
    """What this task's directly required handbook adds to ``mode``'s view of ``plan``, its start note included
    (estimator tokens; 0 without a captured core). The window charges it; no target, memory or transcript narrowing
    is sized by it. A first request that cannot go out with it yields only maps for it (``ContextCore.required_book_tokens``)."""
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
    # Low's missed target: the start's request with no memory at all, reply reserve included (``floor.no_memory_tokens``).
    no_memory_tokens: Optional[int] = None
    window_boundary_tokens: Optional[int] = None
    requested_load_window: Optional[int] = None
    window_record: InitVar[Optional[WindowRecord]] = None  # carried only to the actual admission predicate

    def __post_init__(self, window_record):
        if window_record is not None:
            object.__setattr__(self, "window_record", window_record)

    @property
    def probes_reply_reservation(self) -> bool:
        """This request goes out inside the reply reservation of a stated total nothing has proved, measured on a
        request this route answered: only the favourable reading (``WindowRecord.admitted``) admits it, so the
        provider's size refusal of it is that reading's answer, whatever its words (owner 30.09: the host sends once
        and records the provider's answer). A proved total and an input limit have no such band."""
        usable, admitted = self.window.get("usable"), self.window.get("admitted")
        return bool(self.anchor_input_tokens is not None and usable is not None and admitted is not None
                    and usable < self.estimated_input_tokens <= admitted)

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


def room_fact(measurement, largest, *, source="", memory=""):
    """Passive measured room, before this facts row; usable by either actor driver. ``largest``: the view's largest
    parts (``context_view_facts.largest_parts``)."""
    from ouroboros.context_budget import bound_host_row
    window = measurement['capacity_total_tokens']
    reserve, used = measurement['response_reserve_tokens'], measurement['estimated_input_tokens']
    free = window - reserve - used if window is not None else 'unknown'
    memory = f" | {memory} |" if memory else ""
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
    # A round measures a request that exists: its capacity is the admitted line (``WindowRecord.admitted``) in the
    # frame of consumers which subtract R, so the check, tool delivery, the passes and the facts line read one number.
    capacity = current.admitted_equivalent if current.admitted is not None else None
    if capacity is None:
        reading = None
    raw_book = required_book_addition(plan, rendered_mode)
    book = math.ceil(raw_book * density)
    # A directly required handbook keeps the reply the same candidate without it gets (Nano's is the only elastic one):
    # a known window holds both or its deficit says so; nothing here shrinks for the book (owner 1.11.2026).
    reply = reply_allowance_tokens(caller_max_tokens=ceiling, nano=nano, owner_nano=profile == "owner_nano",
        input_tokens=estimated_input - book, raw_input_tokens=raw_input - raw_book, window_tokens=capacity, window_record=current)
    target_deficit = max(0, total - target) if target is not None else None
    physical_input = current.input_room(reply if book else 0, under_way=True)
    capacity_deficit = max(0, estimated_input - physical_input) if physical_input is not None else None
    economic_deficit = None if target_deficit is None else max(0, target_deficit - book)
    ordinary_capacity_deficit = None if capacity is None else max(0, total - book - capacity)
    deficit = max(
        [value for value in (economic_deficit, ordinary_capacity_deficit) if value is not None]
        or [0]
    )
    margin = reclaim_low_water_margin(target, current.boundary) if deficit > 0 else 0
    goal = deficit + margin
    from ouroboros.context_view_facts import window_projection
    measurement = MainFitMeasurement(
        window=window_projection(current, under_way=True),
        view_kind=plan.view_kind,
        requested_load_window=getattr(plan, "requested_load_window", None),
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
        no_memory_tokens=_no_memory_tokens(plan, rendered_mode), window_boundary_tokens=current.boundary, window_record=current,
    )
    action: Literal["send", "send_target_miss"] = (
        "send_target_miss" if target_deficit and target_deficit > 0 else "send")
    return MainFitDisposition(
        measurement=measurement,
        action=action,
        automatic_pass_used=bool(automatic_pass_used),
        predicted_capacity_miss=bool(capacity_deficit and capacity_deficit > 0),
    )


def _no_memory_tokens(plan: Any, rendered_mode: str) -> Optional[int]:
    """The start's own receipt of a missed Low target: what the request took with no memory (``memory_floor``)."""
    projection = getattr(plan, "projection", None)
    if not callable(projection):
        return None
    value = ((getattr(projection(rendered_mode), "memory_facts", None) or {}).get("floor") or {}).get("no_memory_tokens")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def mode_cause(plan: Any, rendered_mode: str) -> str:
    """Why a candidate renders below the plan's owner mode, from the plan's own facts: ``""`` when it does not; the
    start's own mode carries the start's cause (``ContextFitPlan.start``, ``memory_floor.start_cause``: ``memory``,
    ``request``, ``route`` or ``window``), which no reprojection or route rebind replaces. A first-send preference
    or an unmeasured Max is not a physical reason. Any other lower mode, or a start lowered by no mode switch,
    is ``after_start`` (taken by the task after its start). A restored task whose start
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
    if rendered_mode == started and cause in FIRST_CONTACT_START.values():
        return ""
    cause = "" if cause == "unmeasured" else cause
    return cause if rendered_mode == started and cause else "after_start"


def _physical_context_for_fit(disposition: MainFitDisposition, plan: Any = None) -> PhysicalAttemptContext:
    """Carry the measured candidate's facts to the physical attempt; dispatch still binds the actual scope."""
    from ouroboros.usage_accounting import PhysicalAttemptContext

    measurement = disposition.measurement
    projection_for = getattr(plan, "projection", None)
    projection = projection_for(measurement.rendered_mode) if callable(projection_for) else None
    return PhysicalAttemptContext(
        view_kind=measurement.view_kind,
        profile=measurement.profile,
        rendered_mode=measurement.rendered_mode,
        measurement_basis=measurement.measurement_basis,
        route_fp=measurement.route_fp,
        round_id=measurement.round_id,
        target_total_tokens=measurement.target_total_tokens,
        capacity_total_tokens=measurement.capacity_total_tokens,
        context_target_miss=disposition.action == "send_target_miss",
        automatic_pass_used=disposition.automatic_pass_used,
        measurement_density=measurement.measurement_density,
        response_reserve_tokens=measurement.response_reserve_tokens,
        window_witness_bound=getattr(measurement, "window_witness_bound", False),
        window_observation_fp=measurement.window_observation_fp,
        requested_load_window=getattr(measurement, "requested_load_window", None),
        required_book_tokens=measurement.required_book_tokens, preferred_mode=measurement.preferred_mode,
        mode_cause=measurement.mode_cause, no_memory_tokens=measurement.no_memory_tokens,
        window_boundary_tokens=measurement.window_boundary_tokens,
        window_record=getattr(measurement, "window_record", None),
        base_prompt_source=getattr(getattr(plan, "core", None), "base_prompt_source", None),
        bible_source=getattr(getattr(plan, "core", None), "bible_source", None),
        assembly_witnesses=getattr(projection, "assembly_witnesses", ()),
        first_contact_plan=((plan.first_contact, plan.first_contact_normal is not None)
                            if getattr(plan, "first_contact", "") else None),
    )
