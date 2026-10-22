"""Task-local context-fit projections for the ordinary Main model path.

This module is deliberately data-only around the existing context builder and
capability-evidence SSOT.  It does not own routing, provider retries, or global
context-mode state; callers supply the captured context core and exact-route
resolver so ``ouroboros.context`` remains the public compatibility surface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pathlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from ouroboros.context_anchor import InputAnchor
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

from ouroboros.context_layout import reference_doc_sections
from ouroboros.reference_books import ReferenceBook
# Tool-result delivery (the whole-batch projection under the measured frame)
# lives in ``tool_result_delivery``; the consolidator imports its contract here.
from ouroboros.tool_result_delivery import project_tool_result_batch  # noqa: F401
from ouroboros.utils import estimate_tokens
from ouroboros.context_timing import timed_section

log = logging.getLogger(__name__)


def extract_plain_text_from_content(content: Any) -> str:
    """Text of a string or multipart message content, for transcript sealing.

    The one extractor lives in ``loop_messages`` (the retired ``delivery_protocol``
    leaf carried a copy). Read at CALL time: ``loop_messages`` imports
    ``ouroboros.llm`` at module top and the LLM lanes import this module, so a
    top-level import here would be an import cycle.
    """
    from ouroboros.loop_messages import _extract_plain_text_from_content

    return _extract_plain_text_from_content(content)

ContextProfile = Literal["owner_max", "owner_low", "owner_nano", "task_local_low", "task_local_nano"]
MeasurementBasis = Literal["fresh_route_usage", "fresh_model_usage", "cold_estimate"]


@dataclass(frozen=True)
class ContextFitProjection:
    """One deterministic Low/Max rendering of a shared immutable context core."""

    mode: str
    system_content_json: str
    estimated_tokens: int
    calibrated_tokens: int
    calibration_ratio: float
    user_content_json: Optional[str] = None
    # The memory view's fact of this projection (``memory_floor.view_receipt``): role,
    # room, floor steps and boundaries, block sizes; empty without a view (declared input).
    memory_facts: Mapping[str, Any] = field(default_factory=dict)
    # A route rebind may measure the actual shortened transcript, not the initial tail.
    continuation_messages_json: Optional[str] = None

    def system_message(self) -> Dict[str, Any]:
        from ouroboros.llm_messages import STABLE_PREFIX_BLOCKS_KEY

        # Keep every nonempty stable block before the changing tail in the leading
        # system group: common governance A', optional handbook D, identity/story B.
        # Codex reuses another conversation's prefix at whole-item boundaries; B in
        # a notice loses that reuse. Count independently of cache markers, which
        # Codex strips. Only C (knowledge, rooms, runtime facts) becomes the notice.
        content = json.loads(self.system_content_json)
        if isinstance(content, list):
            content = [block for block in content[:-1] if str(block.get("text") or "").strip()] + content[-1:]
        stable = max(1, len(content) - 1) if isinstance(content, list) else 1
        return {"role": "system", "content": content, STABLE_PREFIX_BLOCKS_KEY: stable}


@dataclass(frozen=True)
class MainFitMeasurement:
    route_fp: str
    round_id: str
    profile: ContextProfile
    rendered_mode: Literal["max", "low", "nano"]
    estimated_input_tokens: int  # calibrated: the estimate times the fresh density
    response_reserve_tokens: int  # the reply floor the fit leaves (Nano: min(8,192, ceiling); else the ceiling)
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
    # The largest input this model accepted in this task; a stated window below it is no window
    # (``capacity_total_tokens`` is then None and ``window_reading_tokens`` keeps the stated number).
    accepted_input_tokens: Optional[int] = None
    window_reading_tokens: Optional[int] = None
    view_kind: str = ""


@dataclass(frozen=True)
class MainFitDisposition:
    measurement: MainFitMeasurement
    action: Literal["send", "send_target_miss"]
    automatic_pass_used: bool
    predicted_capacity_miss: bool


@dataclass(frozen=True)
class ContextFitPlan:
    """Task-local context fit decision for the ordinary Main agent path.

    The two projections are rendered from the same captured core.  The plan is
    deliberately data-only: it neither owns provider routing nor changes the
    owner-selected global context mode.  P3 review calls do not use this path.
    """

    core_sha256: str
    preferred_mode: str
    initial_mode: str
    model: str
    provider: str
    route_fp: str
    # Named for ``capability_evidence.CapabilityEvidence``, not paraphrased: the plan
    # is an evidence-shaped record, so ``is_known`` applies to it DIRECTLY instead of
    # each reader restating the three-clause freshness boolean (the wire keys stay
    # ``evidence_status`` / ``evidence_stale``).
    status: str
    stale: bool
    window_tokens: int
    output_reserve_tokens: int
    user_content_json: str
    max_projection: ContextFitProjection
    low_projection: ContextFitProjection
    model_role: str = "main"
    model_route: Dict[str, Any] = field(default_factory=dict)
    evidence_source: str = ""
    nano_projection: Optional[ContextFitProjection] = None
    core: Optional["ContextCore"] = None  # the capture a new route re-renders the memory view from
    view_kind: str = ""

    def reproject_for_route(self, *, window_tokens: int, known_window: bool, ratio: float, output_reserve: int,
                            tool_schemas: Optional[List[Dict[str, Any]]], start_mode: Optional[str] = None,
                            current_messages: Optional[List[Dict[str, Any]]] = None,
                            refused_tokens: int = 0, select_mode: Optional[str] = None) -> "ContextFitPlan":
        """This plan on another route's window: fit and starting mode measured anew, shown views kept while they fit.

        Measured on the actual running tail and resident schemas from ``start_mode`` (the task's mode; default
        ``preferred_mode``). While that mode's request with its reply reserve fits a known fresh window -- a
        smaller one included -- or the window is unknown, every view keeps its bytes: a larger window or a
        shorter tail unfolds nothing (owner decision 2026-10-11, 2A). A request the window cannot hold, or
        ``refused_tokens`` (size recovery's calibrated deficit, at least 1), narrows each view from the
        level it shows against the window alone (``memory_floor.render_view_for_mode``), from
        ``core.memory_view_json`` (chronicle and chat not read again). Current Max books stay full until
        size recovery selects another mode (``select_mode``: rendered from the level ``start_mode`` shows, never
        its own unsent one); lower starts keep their mode. Only the owner's mode carries a target.
        """
        from dataclasses import replace

        contents, start, books = {}, start_mode or self.preferred_mode, ("max", "low")
        if current_messages is None:
            retained = self.projection(start).continuation_messages_json
            if retained is not None:
                current_messages = [self.projection(start).system_message(), *json.loads(retained)]
        tail_json = (json.dumps(_request_tail(current_messages), ensure_ascii=False, sort_keys=True)
                     if current_messages is not None else None)
        if self.core is not None and self.core.memory_view_json:
            from ouroboros import memory_floor
            from ouroboros.context_budget import context_mode_limits
            from ouroboros.memory_view import snapshot_from_json
            from ouroboros.tool_policy import select_tool_schemas

            shown = self.projection(start)
            names = [schema["function"]["name"] for schema in tool_schemas or []]
            sent = select_tool_schemas(tool_schemas or [], context_mode=start, schema_names=names).schemas
            request = math.ceil((_request_tokens(json.loads(shown.system_content_json),
                                                 shown.user_content_json or self.user_content_json,
                                                 supplementary_messages_json=self.core.supplementary_messages_json,
                                                 current_messages=current_messages)
                                 + tool_schema_tokens(list(sent))) * ratio)
            request += context_mode_limits(start, self.preferred_mode, output_reserve)[1]
            deficit = max(request - window_tokens if known_window and window_tokens else 0, int(refused_tokens or 0))
            if deficit > 0 or select_mode:
                snapshot = snapshot_from_json(self.core.memory_view_json)
                contents, start = _view_projections(
                    self.core, {form: _governance_blocks(None, self.core, mode=form) for form in books},
                    self.user_content_json, preferred=self.preferred_mode, start=start_mode, select=select_mode,
                    tool_schemas=tool_schemas,
                    window_tokens=window_tokens, known_window=known_window, output_reserve=output_reserve, ratio=ratio,
                    resident=True, current_messages=current_messages,  # actual running tail, enable_tools included
                    kept={mode: memory_floor.kept_level(snapshot, self.projection(mode).memory_facts)
                          for mode in memory_floor.MODES},
                    saving={start: math.ceil(deficit / max(ratio, 0.01))})

        def narrowed(projection: ContextFitProjection) -> bool:
            # A view whose level, budget and mode line did not move keeps its bytes (its floor note names its
            # cut window); equal counts from another level never keep a view's budget provenance.
            old, new = (dict(facts.get("floor") or {}) for facts in (projection.memory_facts or {},
                                                                     contents[projection.mode][1] or {}))
            return any(old.get(key) != new.get(key) for key in ("steps", "by_budget", "target_tokens", "mode_switch"))

        def project(projection: Optional[ContextFitProjection]) -> Optional[ContextFitProjection]:
            if projection is not None and projection.mode in contents and narrowed(projection):
                system, facts = contents[projection.mode]
                projection = replace(projection, system_content_json=json.dumps(system, ensure_ascii=False, sort_keys=True),
                                     memory_facts=facts, continuation_messages_json=tail_json,
                                     estimated_tokens=_request_tokens(system, projection.user_content_json or self.user_content_json,
                                         supplementary_messages_json=self.core.supplementary_messages_json,
                                         current_messages=current_messages))
            elif projection is not None and current_messages is not None:
                projection = replace(projection, continuation_messages_json=tail_json,
                    estimated_tokens=_request_tokens(json.loads(projection.system_content_json),
                        projection.user_content_json or self.user_content_json, current_messages=current_messages))
            calibrated = int(int(projection.estimated_tokens or 0) * ratio) if projection is not None else 0
            return projection and replace(projection, calibrated_tokens=calibrated, calibration_ratio=ratio)

        return replace(self, initial_mode=select_mode or start, window_tokens=window_tokens,
                       output_reserve_tokens=output_reserve,
                       max_projection=project(self.max_projection), low_projection=project(self.low_projection),
                       nano_projection=project(self.nano_projection))

    def projection(self, mode: str) -> ContextFitProjection:
        if str(mode or "").lower() == "nano" and self.nano_projection is not None:
            return self.nano_projection
        return self.low_projection if str(mode or "").lower() == "low" else self.max_projection

    def messages_for(self, mode: str) -> List[Dict[str, Any]]:
        projection = self.projection(mode)
        if projection.continuation_messages_json is not None:
            return [projection.system_message(), *json.loads(projection.continuation_messages_json)]
        return [
            projection.system_message(),
            {"role": "user", "content": json.loads(projection.user_content_json or self.user_content_json)},
            *json.loads(self.core.supplementary_messages_json if self.core is not None else "[]"),
        ]

    def reproject_transcript(
        self,
        messages: List[Dict[str, Any]],
        mode: str,
    ) -> List[Dict[str, Any]]:
        """Replace only the captured system view; preserve every dialogue/tool turn."""
        if not messages:
            return self.messages_for(mode)
        rebuilt = list(messages)
        if str(rebuilt[0].get("role") or "") == "system":
            rebuilt[0] = self.projection(mode).system_message()
        else:
            rebuilt.insert(0, self.projection(mode).system_message())
        return rebuilt

    def projected_tokens_with_tools(
        self,
        mode: str,
        tools: Optional[List[Dict[str, Any]]],
        *,
        provider: str = "",
        reasoning_effort: str = "",
    ) -> int:
        """Calibrated physical prompt projection after schemas are available."""
        projection = self.projection(mode)
        if not (
            str(provider or "").strip().lower() == "openai"
            and str(reasoning_effort or "").strip().lower() not in {"", "none"}
            and tools
        ):
            return int(
                (projection.estimated_tokens + tool_schema_tokens(tools))
                * projection.calibration_ratio
            )
        return int(
            estimate_context_prompt_tokens(
                self.messages_for(mode),
                tools,
                provider=provider,
                reasoning_effort=reasoning_effort,
            )
            * projection.calibration_ratio
        )

@dataclass(frozen=True)
class ContextCore:
    """Single captured context source rendered into deterministic projections."""

    base_prompt: str
    bible_md: str
    architecture_md: str
    development_md: str
    semi_stable_text: str
    dynamic_text: str
    user_content_json: str
    docs_need_development: bool
    reference_books: Tuple[ReferenceBook, ...] = ()
    compact_reference_docs: bool = False
    reference_book_errors: Tuple[str, ...] = ()
    # Knowledge (overview, index, patterns, project journal) leading the changing block.
    dynamic_head_text: str = ""
    # The captured memory view (``memory_view.snapshot_json``); each projection renders
    # it for its own mode and window. Empty: no memory view (a declared-input child).
    memory_view_json: str = ""
    # Captured source rows follow the assignment, outside the immutable system view.
    supplementary_messages_json: str = "[]"


def _render_context_system_content(
    env: Any,
    core: ContextCore,
    *,
    mode: str,
    story: str = "",
    room: str = "",
) -> List[Dict[str, Any]]:
    """``[A'▸, D?▸, B▸, C]``: common governance, optional handbook, identity/story, changing facts.

    ``story`` and ``room`` are the memory view rendered for this projection's mode and
    window (``memory_floor.mode_views``); empty without a view.
    """
    return _system_blocks(core, _governance_blocks(env, core, mode=mode), story, room)


def _governance_blocks(env: Any, core: ContextCore, *, mode: str) -> Tuple[str, ...]:
    # D-ARCH (owner, 2026-08-08): the reference-doc form follows the RENDERED
    # mode directly — ARCHITECTURE is full in max for every task class and the
    # nav map in low; DEVELOPMENT inclusion is the caller's mode-independent
    # decision carried on the core (the former per-task force_low_docs lever is
    # gone: workspace binding no longer shapes the docs).
    static_parts = [core.base_prompt, "## BIBLE.md\n\n" + core.bible_md]
    sections, development = reference_doc_sections(
        env,
        context_mode="low" if core.compact_reference_docs else mode,
        include_development=core.docs_need_development,
        architecture_text=core.architecture_md,
        development_text=core.development_md,
        books=core.reference_books,
    )
    static_parts.extend(sections)
    static_parts.extend(core.reference_book_errors)
    return ("\n\n".join(static_parts), development)


def _system_blocks(core: ContextCore, governance: Tuple[str, ...], story: str = "", room: str = "") -> List[Dict[str, Any]]:
    # All stable items precede C, including D only when the captured core calls
    # for the full handbook. Never create an empty cache boundary.
    stable = (*governance, "\n\n".join(part for part in (core.semi_stable_text, story) if part))
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}
            for text in stable if text.strip()] + [
        {"type": "text", "text": "\n\n".join(part for part in (core.dynamic_head_text, room, core.dynamic_text) if part)},
    ]


def seal_task_transcript(messages: List[Dict[str, Any]], min_prefix_tokens: int = 2048) -> None:
    """Mark ONE stable message-side boundary for provider prompt caching.

    The boundary is the newest user or tool row with text, so no tool result
    waits outside it (#596) and a tool-free stretch advances it too (#1186).
    Assistant turns, replayed native content included, never carry it; an
    empty or opaque tool result is skipped (``logical_tool_result_text``). The
    loop normally seals before fresh facts, clock and answer; a recovery
    reseal can include existing facts. Until that prefix is worth a breakpoint the
    task message is the anchor: without it everything after the SYSTEM markers
    -- the mutable context block and the task contract -- is re-sent uncached
    every round, the short-lived nanny and leaf shape. A message without text
    is never marked: Anthropic rejects a marker on an empty text block.

    The mark is host-only, a message-level ``cache_control``: the send copies
    of marker-carrying routes render it on the row's last non-empty text block
    and every other copy drops it, so canonical content, host-row bindings and
    a marker-free route's bytes never change. Anthropic permits four
    breakpoints: A', optional D, B and this one; the finalizer marks schemas
    only if a slot remains. The old in-place seal is undone without another
    shape change: a text-only tool list flattens to its exact text, as every
    seal has done, and a block-list task message only loses its markers.
    """
    from ouroboros.tool_result_record import logical_tool_result_text

    for msg in messages:
        msg.pop("cache_control", None)  # this function's previous boundary
        content = msg.get("content")
        if msg.get("role") == "tool" and isinstance(content, list) and (
                plain := logical_tool_result_text(content)) is not None:
            msg["content"] = plain
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if isinstance(first_user, dict) and isinstance(first_user.get("content"), list):
        for block in first_user["content"]:
            if isinstance(block, dict):
                block.pop("cache_control", None)

    def has_text(message: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(message, dict) or message.get("role") not in {"user", "tool"}:
            return False
        content = message.get("content")
        if message["role"] == "tool":
            return bool((logical_tool_result_text(content) or "").strip())
        if isinstance(content, str):
            return bool(content.strip())
        return isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "text" and str(block.get("text") or "").strip()
            for block in content)

    index = next((i for i in range(len(messages) - 1, -1, -1) if has_text(messages[i])), None)
    boundary = messages[index] if index is not None else None
    if boundary is not None and boundary is not first_user and sum(
            len(extract_plain_text_from_content(m.get("content", ""))) for m in messages[:index + 1]
    ) // 4 < min_prefix_tokens:  # rough 4-chars-per-token estimate
        boundary = first_user  # not yet a worthwhile rolling boundary: the task message anchors
    if has_text(boundary):
        boundary["cache_control"] = {"type": "ephemeral"}


def tool_schema_tokens(tools: Optional[List[Dict[str, Any]]]) -> int:
    """chars/4 estimate of the TOOL-SCHEMA segment of a prompt.

    One seam for every consumer that has to account for the schemas: they are sent
    on the wire beside ``messages``, so any measure built from the transcript alone
    silently omits them (~148K chars / ~37K tokens on the submarine traces — enough
    to make an emergency-compaction trigger fire a whole tool envelope late).
    """
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools, ensure_ascii=False, sort_keys=True, default=str))


def bounded_prompt_tokens_for_payload(prompt_payload: Dict[str, Any], fallback_chars: int) -> int:
    """The density witness's basis (``capability_evidence.MAIN_DENSITY_BASIS``): the
    fit estimator's own token count for a request payload (messages + tools,
    images at the proxy, a top-level ``system`` — the Messages API's separate
    field — counted as a leading system message), or ``fallback_chars // 4``
    when there is no message list. Kept beside the estimator so the two can
    never diverge; ``estimate_message_chars`` dropped tool_call objects and made
    density ~1.4x high on the tool-heavy shape, and an uncounted ``system``
    over-sized the direct-Anthropic reply by its whole length (measure_main_fit
    and the send finalizer multiply THIS quantity)."""
    try:
        messages = prompt_payload.get("messages")
        if isinstance(messages, list):
            system = prompt_payload.get("system")
            if isinstance(system, (str, list)) and system:
                messages = [{"role": "system", "content": system}, *messages]
            return int(estimate_context_prompt_tokens(
                messages, prompt_payload.get("tools") or prompt_payload.get("functions")))
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
) -> int:
    """Estimate the complete inspectable context shape with bounded images."""
    from ouroboros.anthropic_native_custody import (
        context_custody_proxy,
        custody_private_key,
    )
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY, IMAGE_BLOCK_CHAR_EQUIVALENT
    from ouroboros.openai_chat_dispatch import direct_openai_context_projections

    def project(value: Any) -> Any:
        if isinstance(value, dict):
            if any(custody_private_key(key) for key in value):
                value = context_custody_proxy(value)
            if str(value.get("type") or "") in {"image", "image_url"}:
                return {
                    "type": str(value.get("type") or "image"),
                    "image_token_proxy": "#" * IMAGE_BLOCK_CHAR_EQUIVALENT,
                }
            return {
                str(key): project(item)
                for key, item in value.items()
                if str(key) != "_context_capsule"
            }
        if isinstance(value, list):
            return [project(item) for item in value]
        return value

    from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY, REVIEW_CONTEXT_INDEX_KEY
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    # These top-level host receipts remain canonical but are absent from sends.
    # Do not strip same-named keys inside native payloads or user/tool arguments.
    hidden = {HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY, REVIEW_HISTORY_MESSAGE_KEY, REVIEW_CONTEXT_INDEX_KEY,
              TOOL_RESULT_RECORD_KEY}
    messages = [{key: value for key, value in message.items() if key not in hidden} for message in messages]
    projections = direct_openai_context_projections(
        messages, tools, provider=provider, reasoning_effort=reasoning_effort,
    )
    return max(
        int(estimate_tokens(json.dumps(
            {"messages": project(projected_messages), "tools": projected_tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )))
        for projected_messages, projected_tools in projections
    )


def _route_calibration_ratio(
    drive_root: Optional[pathlib.Path],
    route_fp: str,
    model: str,
    estimate_tokens: Optional[int] = None,
    *, view_kind: str = "",
) -> float:
    """Fresh comparable-input exact-route/model witness, never an orphan maximum.

    ``None`` reads the canonical host evidence root (one observation store).
    """
    try:
        from ouroboros.capability_evidence import (
            canonical_evidence_root, resolve_main_token_density,
        )

        root = drive_root if drive_root is not None else canonical_evidence_root()
        return float(resolve_main_token_density(root, route_fp, model, estimate_tokens=estimate_tokens,
                                               view_kind=view_kind)[0])
    except Exception:
        log.debug("Fresh route token calibration unavailable", exc_info=True)
        return 1.0


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
    estimate by the last accepted request of this model. Without one, a fresh same-model witness
    of comparable input size calibrates it; absent that, it stays cold. A stated window below
    ``accepted_input_tokens`` is no window (unknown). The deficits,
    the reclaim goal (deficit + ``reclaim_low_water_margin``) and the reply allowance are facts for
    the round and recovery. A known window with no tool-result room starts the same refusal
    passes before sending (owner revision, 20 October 2026); an unknown window never does.
    Both deficits count the reply floor the fit leaves (Nano: ``min(8192, C)``); the reply the
    candidate will actually get is ``reply_allowance_tokens``.
    """
    from ouroboros.capability_evidence import canonical_evidence_root, is_known, resolve_main_token_density
    from ouroboros.context_budget import (
        OWNER_LOW_TARGET_TOKENS, OWNER_NANO_TARGET_TOKENS, NANO_MIN_HEADROOM_TOKENS, reply_allowance_tokens,
    )

    raw_input = int(estimate_context_prompt_tokens(messages, tools, provider=plan.provider, reasoning_effort=reasoning_effort))
    density, basis = ((float(anchor.density), "fresh_route_usage") if anchor is not None else
                      resolve_main_token_density(canonical_evidence_root(), plan.route_fp, plan.model,
                                                 estimate_tokens=raw_input, view_kind=plan.view_kind))
    estimated_input = int(math.ceil(raw_input * density))
    ceiling = int(plan.output_reserve_tokens or 0)
    nano = profile.endswith("_nano")
    reserve = min(NANO_MIN_HEADROOM_TOKENS, ceiling) if nano else ceiling
    total = estimated_input + reserve
    target = (OWNER_NANO_TARGET_TOKENS if profile == "owner_nano"
              else OWNER_LOW_TARGET_TOKENS if profile == "owner_low" else None)
    from ouroboros.context_anchor import floored_window

    reading = int(plan.window_tokens or 0) if is_known(plan, require_fresh=True) else None
    floored, known = floored_window(reading, reading is not None, accepted_input_tokens)
    capacity = floored if known else None
    target_deficit = max(0, total - target) if target is not None else None
    capacity_deficit = max(0, total - capacity) if capacity is not None else None
    deficit = max(
        [value for value in (target_deficit, capacity_deficit) if value is not None]
        or [0]
    )
    margin = reclaim_low_water_margin(target, capacity) if deficit > 0 else 0
    goal = deficit + margin
    measurement = MainFitMeasurement(
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
    )
    action: Literal["send", "send_target_miss"] = (
        "send_target_miss" if target_deficit and target_deficit > 0 else "send")
    return MainFitDisposition(
        measurement=measurement,
        action=action,
        automatic_pass_used=bool(automatic_pass_used),
        predicted_capacity_miss=bool(capacity_deficit and capacity_deficit > 0),
    )


def _context_route(task: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve the same effective settings and account identity on success or failure."""
    from ouroboros.capability_evidence import model_account_options
    from ouroboros.config import runtime_settings
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    settings, _changed, _keys = apply_runtime_provider_defaults(runtime_settings())
    local_override = task.get("use_local_model")
    route = _active_main_route(
        settings, model_override=str(task.get("model") or "").strip(),
        use_local_override=(
            bool(local_override) if local_override is not None else None
        ),
    )
    from ouroboros.model_slots import task_model_binding
    role, account = task_model_binding(task)
    route["model_role"] = role
    route["options"] = {}
    if route["provider"] == "claudexor":
        route["options"] = model_account_options(
            route["model"], role=role, settings=settings,
            credential_profile_id=account,
            model_route=task.get("model_route"),
        )
    return route, settings


def resolve_context_fit_route(
    task: Dict[str, Any],
    *,
    allow_fetch: bool,
) -> Tuple[Dict[str, Any], Any]:
    """Resolve capacity from effective settings and exact account evidence.

    Auto discovery is advertised preparation evidence, not proof of the account
    that will serve the next operation. The caller replaces ``model_route`` from
    actual operation receipts when it rebinds. A manual role window is an owner
    sizing assertion only; it neither changes the provider nor writes a scope ack.
    """
    from dataclasses import replace
    from ouroboros.capability_evidence import SOURCE_USER_SETTING, STATUS_ASSERTED, probe
    from ouroboros.config import DATA_DIR
    from ouroboros.model_slots import MODEL_CONTEXT_WINDOWS_KEY, model_role_option

    route, settings = _context_route(task)
    evidence = probe(
        DATA_DIR,
        provider=route["provider"],
        model=route["model"],
        base_url=route["base_url"],
        use_local=route["use_local"],
        allow_fetch=allow_fetch,
        options=route["options"] or None,
    )
    window = int(model_role_option(MODEL_CONTEXT_WINDOWS_KEY, route["model_role"],
                                   settings=settings))
    if window:
        evidence = replace(evidence, window_tokens=window, status=STATUS_ASSERTED,
                           source=SOURCE_USER_SETTING, stale=False,
                           detail=f"Context for {route['model_role']} asserted by user; provider limit unchanged")
    return route, evidence


def _failed_route_evidence(task: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    from ouroboros.capability_evidence import route_fingerprint

    route, _settings = _context_route(task)
    evidence = SimpleNamespace(
        route_fp=route_fingerprint(
            provider=route["provider"],
            base_url=route["base_url"],
            model=route["model"],
            options=route["options"] or None,
        ),
        status="failed",
        stale=True,
        window_tokens=0,
    )
    return route, evidence


def main_output_reserve_tokens(*, use_local: bool, evidence=None) -> int:
    """Predict the existing send cap without changing its physical allowance."""
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS

    requested = MAIN_LOOP_MAX_TOKENS
    if use_local:
        from ouroboros.llm_local import local_context_limits
        requested = local_context_limits(requested)[1]
    from ouroboros.response_limits import ResponseLimit
    return ResponseLimit(**(getattr(evidence, "response_limit", {}) or {})).ceiling(requested)


def _request_tail(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return messages[1:] if messages and messages[0].get("role") == "system" else messages


def _request_tokens(system_content: List[Dict[str, Any]], user_content_json: str, *,
                    supplementary_messages_json: str = "[]", current_messages: Optional[List[Dict[str, Any]]] = None) -> int:
    tail = (_request_tail(current_messages) if current_messages is not None else
            [{"role": "user", "content": json.loads(user_content_json)}, *json.loads(supplementary_messages_json)])
    return estimate_context_prompt_tokens([{"role": "system", "content": system_content}, *tail])


def _view_projections(core: ContextCore, governance: Mapping[str, Tuple[str, ...]], user_content_json: str, *, preferred: str,
                      tool_schemas: Optional[List[Dict[str, Any]]], window_tokens: int, known_window: bool, output_reserve: int,
                      ratio: float, start: Optional[str] = None, resident: bool = False,
                      current_messages: Optional[List[Dict[str, Any]]] = None,
                      kept: Optional[Mapping[str, Any]] = None, saving: Optional[Mapping[str, int]] = None,
                      select: Optional[str] = None,
                      ) -> Tuple[Dict[str, Tuple[List[Dict[str, Any]], Dict]], str]:
    """Each mode's ``(system content, view receipt)`` and the mode the task starts in.

    ``governance`` is A' and optional D by book form. A mode's fixed part is its request without my memory
    plus the schemas it sends: the owner-mode selection of ``tool_schemas`` on the first request,
    or, ``resident``, the list a running task already sends (an ``enable_tools`` addition counts,
    whatever the mode). View and starting mode are ``memory_floor.mode_views`` of the core's
    snapshot, so a new route re-renders both from the same capture; ``kept``/``saving`` narrow
    the shown views from their own levels and ``select`` is a mode an actual refusal chose
    (``ContextFitPlan.reproject_for_route``).
    """
    form = {mode: "low" if core.compact_reference_docs or mode == "nano" else mode for mode in ("max", "low", "nano")}
    if not core.memory_view_json:
        return {mode: (_system_blocks(core, governance[form[mode]]), {}) for mode in form}, start or preferred
    from ouroboros import memory_floor
    from ouroboros.memory_view import snapshot_from_json
    from ouroboros.tool_policy import select_tool_schemas
    names = [schema["function"]["name"] for schema in tool_schemas or []] if resident else None
    sent = {mode: select_tool_schemas(tool_schemas or [], context_mode=mode, schema_names=names) for mode in form}
    fixed = {mode: _request_tokens(_system_blocks(core, governance[form[mode]]), user_content_json,
                supplementary_messages_json=core.supplementary_messages_json, current_messages=current_messages)
             + tool_schema_tokens(list(sent[mode].schemas)) for mode in form}
    views, start = memory_floor.mode_views(snapshot_from_json(core.memory_view_json), preferred=preferred, start=start,
        fixed_tokens_by_mode=fixed, tool_names=None if tool_schemas is None else {m: sent[m].chosen for m in form},
        window_tokens=window_tokens, known_window=known_window, output_reserve=output_reserve, ratio=ratio,
        allow_mode_lowering=preferred != "max" or start in {"low", "nano"}, kept=kept, saving=saving, select=select)
    return {mode: (_system_blocks(core, governance[form[mode]], story, room), receipt)
            for mode, (story, room, receipt) in views.items()}, start


@timed_section("fit")
def build_context_fit_plan(
    env: Any,
    core: ContextCore,
    task: Dict[str, Any],
    *,
    preferred_mode: str,
    route_resolver: Callable[..., Tuple[Dict[str, Any], Any]],
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> ContextFitPlan:
    """Deterministically project one captured core into ordinary-task Max, Low and Nano.

    Each mode renders the captured memory view against its own fixed part: the books
    of its view, the tool schemas it would send (``tool_schemas``, Nano's selection of
    them) and its reply reserve; the physical floor decides what of my memory that
    mode shows only by address (``memory_floor.render_view_for_mode``).
    """
    preferred = str(preferred_mode or "max").strip().lower()
    if preferred not in {"low", "max", "nano"}:
        preferred = "max"

    meta = task.get("task_metadata") if isinstance(task.get("task_metadata"), dict) else {}
    is_subagent = str(
        task.get("delegation_role") or meta.get("delegation_role") or ""
    ).strip().lower() == "subagent"
    try:
        route, evidence = route_resolver(task, allow_fetch=not is_subagent)
    except Exception:
        log.debug("Context-fit route evidence unavailable; preserving Max", exc_info=True)
        route, evidence = _failed_route_evidence(task)

    user_content = json.loads(core.user_content_json)
    # Keep the fit projection tied to the physical dispatch contract instead of
    # duplicating its output reservation.  The lazy import avoids coupling the
    # data-only fit representation to the high-level model loop.
    from ouroboros.capability_evidence import is_known
    from ouroboros.tool_policy import select_tool_schemas

    output_reserve = main_output_reserve_tokens(use_local=bool(route.get("use_local")), evidence=evidence)
    known_window = is_known(evidence, require_fresh=True)
    view_args = dict(
        preferred=preferred, tool_schemas=tool_schemas, window_tokens=int(evidence.window_tokens or 0),
        known_window=known_window, output_reserve=output_reserve)
    governance = {form: _governance_blocks(env, core, mode=form) for form in ("max", "low")}
    from ouroboros.memory_view import view_role

    role = view_role({**task, "metadata": task.get("metadata") or meta})
    view_kind = {"consciousness": "own_wake", "presence": "focus", "integrator": "owner_turn"}.get(role, role)
    # Size the cold starting view before selecting a witness, just as round one does.
    contents, initial_mode = _view_projections(core, governance, core.user_content_json, **view_args, ratio=1.0)
    ratio = _route_calibration_ratio(
        None,  # canonical host evidence, never a child task's own drive
        str(evidence.route_fp or ""),
        str(route["model"] or ""),
        estimate_context_prompt_tokens(
            [{"role": "system", "content": contents[initial_mode][0]},
             {"role": "user", "content": user_content}, *json.loads(core.supplementary_messages_json)],
            list(select_tool_schemas(tool_schemas or [], context_mode=initial_mode).schemas),
            provider=str(route.get("provider") or "")),
        view_kind=view_kind,
    )
    if ratio != 1.0:
        contents, initial_mode = _view_projections(core, governance, core.user_content_json, **view_args, ratio=ratio)
    input_source = None

    from ouroboros.context_budget import context_mode_limits

    # The task-input pointer is decided on the calibrated Nano request: the schemas it sends and its reply floor.
    nano_sent_tokens = tool_schema_tokens(list(select_tool_schemas(tool_schemas or [], context_mode="nano").schemas))
    nano_reserve = context_mode_limits("nano", preferred, output_reserve)[1]
    window = int(evidence.window_tokens or 0) if known_window else 0

    def nano_total(tokens: int) -> int:
        return math.ceil((tokens + nano_sent_tokens) * ratio) + nano_reserve

    def _projection(mode: str) -> ContextFitProjection:
        nonlocal input_source

        system_content, memory_facts = contents[mode]
        system_content_json = json.dumps(system_content, ensure_ascii=False, sort_keys=True)
        estimated = _request_tokens(system_content, core.user_content_json,
                                    supplementary_messages_json=core.supplementary_messages_json)
        user_projection = None
        # Physical rescue only: the owner's words are delivered whole unless a known fresh window cannot hold
        # this request and the pointer request fits it. The Nano target never hides them, and neither does
        # an unknown window (owner decision 2026-10-13, #1527).
        if preferred == "nano" and mode == "nano" and window and nano_total(estimated) > window:
            # The original owner input stays exact in the captured core and
            # existing source store. Only its initial Nano delivery changes;
            # this is neither the external-assignment compiler nor a summary.
            try:
                from ouroboros.artifacts import store_actor_source_bytes
                if input_source is None:
                    input_source = store_actor_source_bytes(env.drive_root, str(task["id"]),
                        category="context_checkpoints", source_id="task_input",
                        data=core.user_content_json.encode("utf-8"), extension="json")
                source_content = (
                    "[Exact task input source]\nThe complete original user input is stored as JSON at this source. "
                    "Read its full content through the given reader in ranges before substantive decisions. "
                    "This pointer is not a summary or a change to the task.\n"
                    + json.dumps(input_source, ensure_ascii=False, sort_keys=True)
                )
                source_estimate = estimate_context_prompt_tokens([
                    {"role": "system", "content": json.loads(system_content_json)},
                    {"role": "user", "content": source_content}, *json.loads(core.supplementary_messages_json)])
                if nano_total(source_estimate) <= window:
                    user_projection = json.dumps(source_content, ensure_ascii=False)
                    estimated = source_estimate
            except (OSError, ValueError, KeyError):
                log.warning("Exact task input source could not be retained; preserving complete input", exc_info=True)
        return ContextFitProjection(
            mode=mode,
            system_content_json=system_content_json,
            estimated_tokens=estimated,
            calibrated_tokens=int(estimated * ratio),
            calibration_ratio=ratio,
            user_content_json=user_projection,
            memory_facts=memory_facts,
        )

    max_projection = _projection("max")
    low_projection = _projection("low")
    nano_projection = _projection("nano")
    # Prediction fits memory granularity but never removes current Max books, even when
    # the shortest estimated view misses the window. Pre-send or provider size recovery selects Low;
    # owner Low/Nano and already-lowered starts retain their existing sizing behavior.

    core_payload = json.dumps(
        {
            "base_prompt": core.base_prompt,
            "bible_md": core.bible_md,
            "architecture_md": core.architecture_md,
            "development_md": core.development_md,
            "semi_stable_text": core.semi_stable_text,
            "dynamic_head_text": core.dynamic_head_text,
            "dynamic_text": core.dynamic_text,
            "memory_view_json": core.memory_view_json,
            **({"supplementary_messages": json.loads(core.supplementary_messages_json)}
               if json.loads(core.supplementary_messages_json) else {}),
            "user_content": user_content,
            "docs_need_development": core.docs_need_development,
            "compact_reference_docs": core.compact_reference_docs,
            "reference_book_errors": core.reference_book_errors,
            "reference_sources": [(book.book_id, source.source_path, source.sha256)
                                  for book in core.reference_books for source in (book.entrypoint, *book.chapters)],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return ContextFitPlan(
        view_kind=view_kind,
        core_sha256=hashlib.sha256(core_payload.encode("utf-8")).hexdigest(),
        preferred_mode=preferred,
        initial_mode=initial_mode,
        model=str(route["model"] or ""),
        provider=str(route["provider"] or ""),
        route_fp=str(evidence.route_fp or ""),
        status=str(evidence.status or ""),
        stale=bool(evidence.stale),
        window_tokens=int(evidence.window_tokens or 0),
        output_reserve_tokens=output_reserve,
        user_content_json=core.user_content_json,
        max_projection=max_projection,
        low_projection=low_projection,
        nano_projection=nano_projection,
        model_role=str(route.get("model_role") or "main"),
        model_route={
            "source": str(getattr(evidence, "source_id", "") or ""),
            "model": str(route["model"]).partition("=")[2],
            "credentialProfileId": str(getattr(evidence, "credential_profile_id", "") or ""),
            "accountFingerprint": str(getattr(evidence, "account_fingerprint", "") or ""),
        } if route["provider"] == "claudexor" else {},
        evidence_source=str(getattr(evidence, "source", "") or ""),
        core=core,
    )


def _physical_context_for_fit(disposition: Any) -> PhysicalAttemptContext:
    from ouroboros.usage_accounting import PhysicalAttemptContext

    measurement = disposition.measurement
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
    )
