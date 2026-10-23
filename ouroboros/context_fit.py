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
from typing import TYPE_CHECKING, Any, Iterable, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.context_measurement import (
    ContextProfile as ContextProfile, MeasurementBasis as MeasurementBasis,
    MainFitMeasurement as MainFitMeasurement, MainFitDisposition as MainFitDisposition,
    measure_main_fit as measure_main_fit, reclaim_low_water_margin as reclaim_low_water_margin,
)
if TYPE_CHECKING:
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

from ouroboros.context_layout import reference_doc_sections
from ouroboros.section_floor import HostSection, section_views, _governance_blocks as _section_governance, _system_blocks
from ouroboros.section_floor import PATTERN_REGISTER_POINTER as PATTERN_REGISTER_POINTER
from ouroboros.reference_books import ReferenceBook
from ouroboros import context_budget
from ouroboros.context_window import WindowRecord, resolve_window, route_observation_scope
from ouroboros.tool_result_delivery import project_tool_result_batch  # noqa: F401
from ouroboros.utils import estimate_tokens

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
class ContextFitPlan:
    """Task-local Main fit from one immutable capture; no global mode or review authority."""

    core_sha256: str
    preferred_mode: str
    initial_mode: str
    model: str
    provider: str
    route_fp: str
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
    working_reserve_tokens: Optional[int] = None  # route working room, independent of the reply allowance
    window_basis: str = ""  # provider/estimator for a declared conversation window; empty for ordinary text
    view_kind: str = ""
    window_record: Optional[WindowRecord] = None
    window_witness_bound: bool = True
    route_base_url: str = ""

    def reproject_for_route(self, *, window_tokens: int, known_window: bool, ratio: float, output_reserve: int,
                            tool_schemas: Optional[List[Dict[str, Any]]], start_mode: Optional[str] = None,
                            current_messages: Optional[List[Dict[str, Any]]] = None,
                            refused_tokens: int = 0, select_mode: Optional[str] = None) -> "ContextFitPlan":
        """Keep fitting shown bytes; narrow on physical deficit from the same snapshot and selected forms."""
        from dataclasses import replace

        ratio = 1.0 if self.window_basis == "estimator" else ratio
        reserve_args = ({"working_reserve_tokens": self.working_reserve_tokens}
                        if self.working_reserve_tokens is not None else {})
        contents, start, books = {}, start_mode or self.preferred_mode, ("max", "low")
        reserve = context_budget.context_mode_limits(start, self.preferred_mode, output_reserve, **reserve_args)[1]
        record = resolve_window(self, reserve, bound=self.window_witness_bound) if self.window_record else None
        if record is not None:
            if not known_window or window_tokens not in (self.window_tokens, record.total_equivalent):
                record = None  # the caller supplied a different physical route frame
            else:
                window_tokens = record.total_equivalent  # input limits keep their bound when the reserve changes
        if current_messages is None:
            retained = self.projection(start).continuation_messages_json
            if retained is not None:
                current_messages = [self.projection(start).system_message(), *json.loads(retained)]
        tail_json = (json.dumps(_request_tail(current_messages), ensure_ascii=False, sort_keys=True)
                     if current_messages is not None else None)
        if self.core is not None:
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
            request += context_mode_limits(start, self.preferred_mode, output_reserve, **reserve_args)[1]
            deficit = max(request - window_tokens if known_window and window_tokens else 0, int(refused_tokens or 0))
            if deficit > 0 or select_mode:
                snapshot = snapshot_from_json(self.core.memory_view_json) if self.core.memory_view_json else None
                contents, start = _view_projections(
                    self.core, {form: _governance_blocks(None, self.core, mode=form) for form in books},
                    self.user_content_json, preferred=self.preferred_mode, start=start_mode, select=select_mode,
                    tool_schemas=tool_schemas, working_reserve_tokens=self.working_reserve_tokens,
                    window_tokens=window_tokens, known_window=known_window, output_reserve=output_reserve, ratio=ratio,
                    window_record=record,
                    resident=True, current_messages=current_messages,  # actual running tail, enable_tools included
                    kept={mode: memory_floor.kept_level(snapshot, self.projection(mode).memory_facts) if snapshot else None
                          for mode in memory_floor.MODES},
                    kept_sections={mode: self.projection(mode).memory_facts.get("host_sections_addressed", ())
                                   for mode in memory_floor.MODES},
                    saving={start: math.ceil(deficit / max(ratio, 0.01))})

        def narrowed(projection: ContextFitProjection) -> bool:
            # A view whose level, budget and mode line did not move keeps its bytes (its floor note names its
            # cut window); equal counts from another level never keep a view's budget provenance.
            old, new = (dict(facts.get("floor") or {}) for facts in (projection.memory_facts or {},
                                                                     contents[projection.mode][1] or {}))
            return (any(old.get(key) != new.get(key) for key in ("steps", "by_budget", "target_tokens", "mode_switch"))
                    or projection.memory_facts.get("host_sections_addressed")
                    != contents[projection.mode][1].get("host_sections_addressed"))

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

        return replace(self, initial_mode=start if contents else select_mode or start, window_tokens=window_tokens,
                       window_record=record,
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
    # The Pattern Register section as captured in ``dynamic_head_text``: Max, Low and every
    # later route render it; Nano renders PATTERN_REGISTER_POINTER in its place (owner, 2026-10-20).
    pattern_register_text: str = ""
    # The captured memory view (``memory_view.snapshot_json``); each projection renders
    # it for its own mode and window. Empty: no memory view (a declared-input child).
    memory_view_json: str = ""
    # Captured source rows follow the assignment, outside the immutable system view.
    supplementary_messages_json: str = "[]"
    semi_stable_sections: Tuple[HostSection, ...] = ()
    dynamic_head_sections: Tuple[HostSection, ...] = ()
    dynamic_sections: Tuple[HostSection, ...] = ()
    section_root: str = ""
    section_task_id: str = ""


def _governance_blocks(env: Any, core: ContextCore, *, mode: str) -> Tuple[Any, ...]:
    return _section_governance(env, core, mode=mode, reference_docs=reference_doc_sections)


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
    selected forms; empty without a view.
    """
    return _system_blocks(core, _governance_blocks(env, core, mode=mode), story, room, mode)


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
    route["window_observation_fp"] = route_observation_scope(route, settings)
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
    from ouroboros.capability_evidence import probe, with_role_window
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
    if route["provider"] == "claudexor" and getattr(evidence, "account_fingerprint", ""):  # the account receipts name
        route["options"] = {**route["options"], **{key: getattr(evidence, key) for key in (
            "source_id", "credential_profile_id", "account_fingerprint")}}
        route["window_observation_fp"] = route_observation_scope(route, settings)
    window = int(model_role_option(MODEL_CONTEXT_WINDOWS_KEY, route["model_role"],
                                   settings=settings))
    return route, with_role_window(evidence, window)


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


class MinimumViewUnfit(ValueError):
    """The core, assignment, resident tools and work room cannot fit this route."""
    physical = True  # False: the request itself fits the window, only the start work room beside it does not

    def __init__(self, message: str, *, fixed_tokens: int, memory_tokens: int,
                 reserve_tokens: int, window_tokens: int, ratio: float, work_room_tokens: int = 0):
        # Retain the fitter's exact rounding: fixed + memory is calibrated once.
        self.raw_input_tokens = fixed_tokens + memory_tokens
        self.measurement_density = ratio
        self.fixed_input_tokens = math.ceil(fixed_tokens * ratio)
        self.estimated_input_tokens = math.ceil(self.raw_input_tokens * ratio)
        self.memory_view_tokens = self.estimated_input_tokens - self.fixed_input_tokens
        self.response_reserve_tokens = reserve_tokens
        self.window_tokens = window_tokens
        self.deficit_tokens = max(0, self.estimated_input_tokens + reserve_tokens - window_tokens)
        self.work_room_tokens = work_room_tokens
        self.work_room_deficit_tokens = max(0, self.fixed_input_tokens + reserve_tokens + work_room_tokens - window_tokens)
        super().__init__(f"{message}: fixed input={self.fixed_input_tokens}, memory view={self.memory_view_tokens}, "
                         f"reply/working reserve={reserve_tokens}, window={window_tokens}, "
                         f"deficit={self.deficit_tokens}, work room={work_room_tokens}, "
                         f"fixed/work-room deficit={self.work_room_deficit_tokens} tokens")


def _view_projections(core: ContextCore, governance: Mapping[str, Tuple[Any, ...]], user_content_json: str, *, preferred: str,
                      tool_schemas: Optional[List[Dict[str, Any]]], window_tokens: int, known_window: bool, output_reserve: int,
                      ratio: float, start: Optional[str] = None, resident: bool = False,
                      current_messages: Optional[List[Dict[str, Any]]] = None,
                      kept: Optional[Mapping[str, Any]] = None, saving: Optional[Mapping[str, int]] = None,
                      select: Optional[str] = None, kept_sections: Optional[Mapping[str, Iterable[Mapping]]] = None,
                      working_reserve_tokens: Optional[int] = None, window_record: Optional[WindowRecord] = None, work_room_levels: int = 1,
                      ) -> Tuple[Dict[str, Tuple[List[Dict[str, Any]], Dict]], str]:
    """Select books once, walk memory once. Unsent book forms share the selected memory, not another budget."""
    from dataclasses import replace
    from ouroboros import memory_floor as floor
    from ouroboros.memory_view import snapshot_from_json, render_story, render_room, FULL_VIEW
    from ouroboros.tool_policy import select_tool_schemas

    mode = select or start or preferred
    form = {m: "low" if core.compact_reference_docs or m == "nano" else m for m in floor.MODES}
    snapshot = snapshot_from_json(core.memory_view_json) if core.memory_view_json else None
    names = [schema["function"]["name"] for schema in tool_schemas or []] if resident else None

    sections = {m: (*core.semi_stable_sections, *core.dynamic_sections,
                    *(part for part in core.dynamic_head_sections
                      if not (m == "nano" and part.section_id == "pattern_register")),
                    *(governance[form[m]][0] if not isinstance(governance[form[m]][0], str) else ()))
                for m in form}
    previous = tuple((kept_sections or {}).get(start or preferred, ()))
    receipts = list(previous)
    held = (kept or {}).get(start or preferred) or FULL_VIEW
    story_before, room_before = (render_story(snapshot, held), render_room(snapshot, held)) if snapshot else ("", "")

    def frame(selected):
        nonlocal receipts
        sent = select_tool_schemas(tool_schemas or [], context_mode=selected, schema_names=names)
        def measure(addressed, story="", room=""):
            return _request_tokens(_system_blocks(core, governance[form[selected]], story, room,
                mode=selected, addressed=addressed), user_content_json,
                supplementary_messages_json=core.supplementary_messages_json,
                current_messages=current_messages) + tool_schema_tokens(list(sent.schemas))
        target, reserve = context_budget.context_mode_limits(selected, preferred, output_reserve,
                                                            working_reserve_tokens=working_reserve_tokens)
        record = replace(window_record, reserve=reserve) if window_record else None
        window = record.total_equivalent if record and record.usable is not None else window_tokens
        boundary = record.boundary if record else None
        levels = snapshot.spec.work_room_levels if snapshot else work_room_levels
        host_allowances = floor.floor_allowances(window_tokens=window if known_window else None,
            output_reserve_tokens=reserve, non_memory_tokens=0, target_tokens=target if kept is None else None,
            calibration_ratio=ratio, work_room_levels=levels, boundary_tokens=boundary)
        receipts, _ = floor._host_steps(sections[selected], receipts,
            measure=lambda addressed: measure(addressed, story_before, room_before),
            window=window if known_window else None, reserve=reserve, ratio=ratio,
            saving=(saving or {}).get(start or preferred, 0), root=core.section_root, task_id=core.section_task_id,
            allowance=host_allowances["margin" if kept is None else "physical"])
        fixed = measure(section_views(sections[selected], receipts))
        allowances = floor.floor_allowances(window_tokens=window if known_window else None,
            output_reserve_tokens=reserve, non_memory_tokens=fixed, target_tokens=target if kept is None else None,
            calibration_ratio=ratio, work_room_levels=levels, boundary_tokens=boundary)
        return sent, fixed, reserve, window, boundary, allowances

    sent, fixed, reserve, window, boundary, allowances = frame(mode)
    original_fixed = _request_tokens(_system_blocks(core, governance[form[start or preferred]],
        mode=start or preferred, addressed=section_views(sections[start or preferred], previous)), user_content_json,
        supplementary_messages_json=core.supplementary_messages_json, current_messages=current_messages) + tool_schema_tokens(list(sent.schemas))
    original_reserve = reserve
    if kept is not None and known_window:
        minimum = floor.minimal_view_tokens(snapshot, window_tokens=window) if snapshot else 0
        while mode != "nano" and math.ceil((fixed + minimum) * ratio) + reserve > window:
            mode = "low" if mode == "max" else "nano"
            sent, fixed, reserve, window, boundary, allowances = frame(mode)
    def work():
        target = context_budget.context_mode_limits(mode, preferred, output_reserve)[0]
        return math.ceil(min(boundary or window, target or window) / context_budget.RECLAIM_LOW_WATER_DIVISOR) * (
            snapshot.spec.work_room_levels if snapshot else work_room_levels)

    # Max names whole books. Story is the last rung: change books before taking its pages.
    whole_story = render_story(snapshot) if snapshot else ""
    if kept is None and mode == "max" and allowances["margin"] is not None:
        if floor.view_tokens(whole_story) > allowances["margin"] or math.ceil(fixed * ratio) + reserve + work() > window:
            mode = "low"
            sent, fixed, reserve, window, boundary, allowances = frame(mode)
    # The same ordinary Nano is the last physical rescue for every Main route.
    if (kept is None and known_window and mode == "low"
            and math.ceil((fixed + (floor.minimal_view_tokens(snapshot, window_tokens=window) if snapshot else 0)) * ratio) + reserve > window):
        mode = "nano"
        sent, fixed, reserve, window, boundary, allowances = frame(mode)
    if kept is None and known_window and math.ceil(fixed * ratio) + reserve + work() > window:
        minimum = floor.minimal_view_tokens(snapshot, window_tokens=window) if snapshot else 0
        unfit = MinimumViewUnfit("Core, assignment, resident control tools and work room cannot fit",
                                 fixed_tokens=fixed, memory_tokens=minimum, reserve_tokens=reserve,
                                 window_tokens=window, ratio=ratio, work_room_tokens=work())
        unfit.physical = math.ceil((fixed + minimum) * ratio) + reserve > window  # with my shortest memory view
        raise unfit

    def render():
        saved = max(0, (saving or {}).get(start or preferred, 0) - (original_fixed - fixed)
                    - math.floor((original_reserve - reserve) / max(ratio, 0.01)))
        return floor.render_view_for_mode(snapshot, mode=mode, owner_mode=preferred,
            window_tokens=window, known_window=known_window, output_reserve=output_reserve, ratio=ratio,
            non_memory_tokens=fixed, lowered_from=preferred if mode != preferred else None,
            tool_names=None if tool_schemas is None else sent.chosen,
            kept=None if kept is None else kept.get(start or preferred), saving=saved, boundary_tokens=boundary,
            working_reserve_tokens=working_reserve_tokens,
            measure_text=lambda story, room: _request_tokens(_system_blocks(core, governance[form[mode]], story, room,
                mode=mode, addressed=section_views(sections[mode], receipts)), user_content_json,
                supplementary_messages_json=core.supplementary_messages_json,
                current_messages=current_messages) + tool_schema_tokens(list(sent.schemas)) - fixed)
    if snapshot:
        story, room, facts = render()
        # Judged on the completed start view: Max keeps whole books only beside the whole story and the work room.
        if kept is None and mode == "max" and allowances["margin"] is not None and (
                story != whole_story or floor.view_tokens(story) + floor.view_tokens(room) > allowances["margin"]):
            mode = "low"
            sent, fixed, reserve, window, boundary, allowances = frame(mode)
            story, room, facts = render()
        if kept is None and known_window and math.ceil((fixed + floor.view_tokens(story) + floor.view_tokens(room)) * ratio) + reserve > window:
            raise MinimumViewUnfit("Protected owner words and the minimum view cannot fit the usable input",
                                   fixed_tokens=fixed, memory_tokens=floor.view_tokens(story) + floor.view_tokens(room),
                                   reserve_tokens=reserve, window_tokens=window, ratio=ratio)
        receipt = floor.view_receipt(snapshot, story, room, facts)
    else:
        story, room, receipt = "", "", {}
    if receipts:
        receipt["host_sections_addressed"] = receipts
    return {m: (_system_blocks(core, governance[form[m]], story, room, m,
        section_views(sections[m], receipts)), receipt) for m in floor.MODES}, mode


def _whole_request(core, governance, mode, tool_schemas):
    """Uncut selected text for a route's comparable density witness, before any floor step."""
    from ouroboros.memory_view import snapshot_from_json, render_story, render_room
    from ouroboros.tool_policy import select_tool_schemas

    snapshot = snapshot_from_json(core.memory_view_json) if core.memory_view_json else None
    story, room = (render_story(snapshot), render_room(snapshot)) if snapshot else ("", "")
    form = "low" if core.compact_reference_docs or mode == "nano" else mode
    messages = [{"role": "system", "content": _system_blocks(core, governance[form], story, room, mode)},
                {"role": "user", "content": json.loads(core.user_content_json)},
                *json.loads(core.supplementary_messages_json)]
    return messages, list(select_tool_schemas(tool_schemas or [], context_mode=mode).schemas)


def build_context_fit_plan(
    env: Any,
    core: ContextCore,
    task: Dict[str, Any],
    *,
    preferred_mode: str,
    route_resolver: Callable[..., Tuple[Dict[str, Any], Any]],
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> ContextFitPlan:
    """Capture route facts, choose books and fit one start view."""
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

    output_reserve = main_output_reserve_tokens(use_local=bool(route.get("use_local")), evidence=evidence)
    route_facts = getattr(evidence, "route_facts", {}) or {}
    working_reserve = route_facts.get("working_reserve_tokens")
    window_basis = str(route_facts.get("basis") or "")
    reserve = context_budget.context_mode_limits(preferred, preferred, output_reserve,
                                                working_reserve_tokens=working_reserve)[1]
    scope = route.get("window_observation_fp", evidence.route_fp)
    bound = bool(scope) and (route.get("provider") != "claudexor" or bool((route.get("options") or {}).get("account_fingerprint")))
    window_record = resolve_window(evidence, reserve, bound=bound, observation_fp=scope)
    known_window = window_record.usable is not None
    view_args = dict(
        preferred=preferred, tool_schemas=tool_schemas, window_tokens=window_record.total_equivalent,
        known_window=known_window, output_reserve=output_reserve, window_record=window_record,
        working_reserve_tokens=working_reserve)
    governance = {form: _governance_blocks(env, core, mode=form) for form in ("max", "low")}
    from ouroboros.memory_view import start_room, view_role

    view_task = {**task, "metadata": task.get("metadata") or meta}
    role = view_role(view_task)
    view_kind = {"consciousness": "own_wake", "presence": "focus", "integrator": "owner_turn"}.get(role, role)
    # Select density on the whole captured request before the single start walk.
    rows, schemas = _whole_request(core, governance, preferred, tool_schemas)
    ratio = 1.0 if window_basis == "estimator" else _route_calibration_ratio(
        None, str(evidence.route_fp or ""), str(route["model"] or ""),
        estimate_context_prompt_tokens(rows, schemas, provider=str(route.get("provider") or "")), view_kind=view_kind)
    levels, pointer = start_room(view_task, role), None
    try:
        contents, initial_mode = _view_projections(core, governance, core.user_content_json, **view_args, ratio=ratio,
                                                   work_room_levels=levels)
    except MinimumViewUnfit as unfit:
        # Physical rescue only (owner 2026-10-13, #1527): the exact source pointer replaces the whole input only when the
        # owner-Nano request itself overflows; a room-only shortfall or an unfit pointer request names the start (7A).
        from ouroboros.context_input_selection import task_input_pointer
        pointer = task_input_pointer(env, task, core.user_content_json) if preferred == "nano" and unfit.physical else None
        if pointer is None:
            raise
        contents, initial_mode = _view_projections(core, governance, pointer, **view_args, ratio=ratio,
                                                   work_room_levels=levels)
    def _projection(mode: str) -> ContextFitProjection:
        system_content, memory_facts = contents[mode]
        system_content_json = json.dumps(system_content, ensure_ascii=False, sort_keys=True)
        user_projection = pointer if mode == "nano" else None
        estimated = _request_tokens(system_content, user_projection or core.user_content_json,
                                    supplementary_messages_json=core.supplementary_messages_json)
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

    core_payload = json.dumps(
        {
            "base_prompt": core.base_prompt,
            "bible_md": core.bible_md,
            "architecture_md": core.architecture_md,
            "development_md": core.development_md,
            "semi_stable_text": core.semi_stable_text,
            "host_sections": [(part.section_id, part.full_text, part.operative_view, part.reader)
                              for part in (*core.semi_stable_sections, *core.dynamic_head_sections, *core.dynamic_sections)],
            "dynamic_head_text": core.dynamic_head_text,
            **({"pattern_register_text": core.pattern_register_text} if core.pattern_register_text else {}),
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
        status="confirmed" if known_window else str(evidence.status or ""),
        stale=False if known_window else bool(evidence.stale),
        window_tokens=window_record.total_equivalent,
        output_reserve_tokens=output_reserve,
        working_reserve_tokens=working_reserve,
        window_basis=window_basis,
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
        evidence_source=window_record.source, window_record=window_record, window_witness_bound=bound,
        route_base_url=str(route.get("base_url") or ""),
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
        response_reserve_tokens=measurement.response_reserve_tokens,
        window_witness_bound=getattr(measurement, "window_witness_bound", False),
        window_observation_fp=measurement.window_observation_fp,
    )
