"""Task-local context-fit projections for the ordinary Main model path.

This module is deliberately data-only around the existing context builder and
capability-evidence SSOT.  It does not own routing, provider retries, or global
context-mode state; callers supply the captured context core and exact-route
resolver so ``ouroboros.context`` remains the public compatibility surface. It owns
only the handle of the fit's own route question (``RouteQuestion``), asked beside the build.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import math
import pathlib
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.context_measurement import (
    ContextProfile as ContextProfile, MeasurementBasis as MeasurementBasis,
    MainFitMeasurement as MainFitMeasurement, MainFitDisposition as MainFitDisposition,
    measure_main_fit as measure_main_fit, reclaim_low_water_margin as reclaim_low_water_margin,
    BOOK_NOTE_KEY, required_book_addition,
)
if TYPE_CHECKING:
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

from ouroboros.context_layout import reference_doc_sections, whole_development_book
from ouroboros.section_floor import (HostSection, render_governance, rendered_texts, section_views,
                                     _governance_blocks as _section_governance, _system_blocks)
from ouroboros.reference_books import ReferenceBook
from ouroboros import context_budget
from ouroboros.context_window import WindowRecord, resolve_window, route_observation_scope
from ouroboros.tool_result_delivery import project_tool_result_batch  # noqa: F401
from ouroboros.utils import estimate_tokens
from ouroboros.context_timing import adopt, another_pass, apart, counted, timed, timed_section

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
    # room, floor steps and boundaries, block sizes; empty without a view.
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
    account_pin: str = ""  # the profile the role pins ("": Auto, where an answer naming no route names no account)
    window_evidence: Tuple[str, str] = ("", "")  # the start's ``capability_evidence.window_evidence``, for its receipt
    start_tool_schemas_json: Optional[str] = None  # what the start fit priced (``null``: none); None: not captured
    # The start fit's own mode and why it is below the owner's (``memory_floor.start_cause``): every reprojection
    # and route rebind keeps it, while ``initial_mode`` follows the current selection. ("", ""): not a built start;
    # ``UNKNOWN_START``: a restored task whose continuation did not carry its start (``restored_start``).
    start: Tuple[str, str] = ("", "")

    def reproject_for_route(self, *, window_tokens: int, known_window: bool, ratio: float, output_reserve: int,
                            tool_schemas: Optional[List[Dict[str, Any]]], start_mode: Optional[str] = None,
                            current_messages: Optional[List[Dict[str, Any]]] = None,
                            refused_tokens: int = 0, select_mode: Optional[str] = None,
                            measured_input_tokens: Optional[int] = None, start_work: bool = False,
                            slack_tokens: int = 0) -> "ContextFitPlan":
        """Fit shown bytes; the prepared envelope's raw estimate includes its signed size difference.

        The measurement belongs to this candidate, includes tools and excludes the reply; withholding an image can shrink it.
        A requested saving alone does not establish pressure on held owner words or notes.
        ``slack_tokens`` asks room beyond the measured or refused excess: memory and ordinary host forms may pay it,
        a whole-or-outline section never (``memory_floor._host_steps``).
        """
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
            raw = (_request_tokens(json.loads(shown.system_content_json),
                                   shown.user_content_json or self.user_content_json,
                                   supplementary_messages_json=self.core.supplementary_messages_json,
                                   current_messages=current_messages) + tool_schema_tokens(list(sent)))
            overhead = int(measured_input_tokens) - raw if measured_input_tokens is not None else 0
            request = math.ceil((raw + overhead) * ratio)
            request += context_mode_limits(start, self.preferred_mode, output_reserve, **reserve_args)[1]
            if start_work and known_window:
                request += math.ceil((record.boundary if record else window_tokens) / context_budget.RECLAIM_LOW_WATER_DIVISOR)
            request -= math.ceil(required_book_addition(self, start) * ratio)  # nothing narrows to pay for the book
            measured, units = (request - window_tokens if known_window and window_tokens else 0), max(ratio, 0.01)
            excess, deficit = max(measured, int(refused_tokens or 0)), max(measured, int(refused_tokens or 0) + slack_tokens)
            if deficit > 0 or select_mode:
                snapshot = snapshot_from_json(self.core.memory_view_json) if self.core.memory_view_json else None
                contents, start = _view_projections(
                    self.core, {form: _governance_blocks(None, self.core, mode=form) for form in books},
                    self.user_content_json, preferred=self.preferred_mode, start=start_mode, select=select_mode,
                    tool_schemas=tool_schemas, working_reserve_tokens=self.working_reserve_tokens,
                    window_tokens=window_tokens, known_window=known_window, output_reserve=output_reserve, ratio=ratio,
                    window_record=record,
                    envelope_overhead_tokens=overhead,
                    start_work=start_work,
                    resident=True, current_messages=current_messages,  # actual running tail, enable_tools included
                    kept={mode: memory_floor.kept_level(snapshot, self.projection(mode).memory_facts) if snapshot else None
                          for mode in memory_floor.MODES},
                    kept_sections={mode: self.projection(mode).memory_facts.get("host_sections_addressed", ())
                                   for mode in memory_floor.MODES},
                    saving={start: math.ceil(deficit / units)}, slack=math.ceil(deficit / units) - math.ceil(excess / units))

        def narrowed(projection: ContextFitProjection) -> bool:
            # A view whose level, budget and mode line did not move keeps its bytes (its floor note names its
            # cut window); equal counts from another level never keep a view's budget provenance.
            old, new = (dict(facts.get("floor") or {}) for facts in (projection.memory_facts or {},
                                                                     contents[projection.mode][1] or {}))
            return (any(old.get(key) != new.get(key) for key in ("forms_version", "steps", "runs", "by_budget", "target_tokens", "mode_switch"))
                    or projection.memory_facts.get("host_sections_addressed")
                    != contents[projection.mode][1].get("host_sections_addressed"))

        def project(projection: Optional[ContextFitProjection]) -> Optional[ContextFitProjection]:
            if projection is not None and projection.mode in contents and narrowed(projection):
                system, facts = contents[projection.mode]  # the start's Max room outlives a narrowing, as a selection's
                projection = replace(projection, system_content_json=json.dumps(system, ensure_ascii=False, sort_keys=True),
                                     memory_facts={**facts, **{key: value for key, value in projection.memory_facts.items()
                                                               if key == "max_room"}}, continuation_messages_json=tail_json,
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
        return int(estimate_context_prompt_tokens(self.messages_for(mode), tools, provider=provider,
                                                  reasoning_effort=reasoning_effort) * projection.calibration_ratio)

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
    # Knowledge (overview, index, project journal) leading the changing block.
    dynamic_head_text: str = ""
    # The captured memory view (``memory_view.snapshot_json``); each projection renders
    # it for its own mode and window. Empty: no memory view was captured.
    memory_view_json: str = ""
    # That capture's journal boundary (``MemoryViewSnapshot.chronicle_sequence``), beside the view and not hashed.
    memory_view_sequence: Optional[int] = None
    # Captured source rows follow the assignment, outside the immutable system view.
    supplementary_messages_json: str = "[]"
    semi_stable_sections: Tuple[HostSection, ...] = ()
    dynamic_head_sections: Tuple[HostSection, ...] = ()
    dynamic_sections: Tuple[HostSection, ...] = ()
    section_root: str = ""
    section_task_id: str = ""
    # The connections map (``tool_connections.connection_views``): the Low/Max text, then the Nano text.
    connections: Tuple[str, str] = ("", "")
    # This task's own direct requirement of the whole handbook (``development_required``), beside
    # ``docs_need_development``, which then holds what the subject alone says for Max.
    development_required: bool = False

    def required_book_tokens(self, mode: str) -> int:
        """What the directly required handbook adds to the ordinary view of ``mode``, 0 when that view holds it.

        Compact views keep their chapter map beside it; Max for another subject carries the map in the book's
        place. It counts against the window only: every owner target, automatic narrowing and rescue is sized
        without it. A missing source adds nothing: the capture discloses the gap instead of a partial book.
        """
        full = mode == "max" and not self.compact_reference_docs
        if not self.development_required or not self.development_md.strip() or (full and self.docs_need_development):
            return 0
        chapter_map = reference_doc_sections(None, context_mode="max", include_development=False, architecture_text="",
                                             development_text=self.development_md, books=self.reference_books)[1] if full else ""
        # Characters as the request estimate counts them: its own system item (comma included), less the map and its
        # separator inside B; the estimate is chars/4 over the whole request.
        item = len(json.dumps({"text": whole_development_book(self.development_md), "type": "text"}, ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"))) + 1
        replaced = len(json.dumps(chapter_map, ensure_ascii=False)) + 2 if chapter_map else 0
        return max(0, (item - replaced + 3) // 4)


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
    return _system_blocks(core, _governance_blocks(env, core, mode=mode), story, room, mode=mode)


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
    a marker-free route's bytes never change. Anthropic builders may add a
    system-tail endpoint in a spare slot (llm_messages); the payload finalizer
    owns the breakpoint cap and TTL. The old in-place seal is undone without another
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
    chars: bool = False,
) -> int:
    """Estimate the complete inspectable context shape with bounded images (``chars``: its JSON characters)."""
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
            # A realtime engine's own wire shapes are images too: OpenAI's ``input_image`` and Gemini's
            # ``inlineData`` (28.10: four screenshots in a Gemini renewal counted as 309k tokens of base64).
            media = value.get("inlineData")
            if (str(value.get("type") or "") in {"image", "image_url", "input_image"}
                    or (isinstance(media, dict) and str(media.get("mimeType") or "").startswith("image/"))):
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
    from ouroboros.llm_messages import HOST_ONLY_MESSAGE_KEYS
    messages = [{key: value for key, value in message.items() if key not in hidden} for message in messages]
    messages = [{**message, "content": [
        {key: value for key, value in block.items() if key not in HOST_ONLY_MESSAGE_KEYS}
        if isinstance(block, dict) else block for block in message["content"]]}
        if isinstance(message.get("content"), list) else message for message in messages]
    # Only this route/effort can use the custom/function ladder. Other estimates
    # read canonical data without borrowing the public helper's owned deep copies.
    if (str(provider or "").strip().lower() == "openai"
            and str(reasoning_effort or "").strip().lower() not in {"", "none"}):
        projections = direct_openai_context_projections(
            messages, tools, provider=provider, reasoning_effort=reasoning_effort,
        )
    else:
        projections = ((messages, [dict(item) for item in tools or []]),)

    def serialized(projected_messages: Any, projected_tools: Any) -> str:
        text = json.dumps({"messages": project(projected_messages), "tools": projected_tools}, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"), default=str)
        counted("request_serializations", key=hash(text))
        return text

    longest = max(len(serialized(*projection)) for projection in projections)
    return longest if chars else max(1, (longest + 3) // 4)


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
    from ouroboros.route_choice import task_lead
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    settings, _changed, _keys = apply_runtime_provider_defaults(runtime_settings())
    carried = None if task.get("model") else task_lead(task.get("task_metadata", task.get("metadata")))  # sized from its first plan
    local_override = carried["use_local"] if carried else task.get("use_local_model")
    route = _active_main_route(
        settings, model_override=str((carried or task).get("model") or "").strip(),
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
        route["account_pin"] = str(model_account_options(route["model"], role=role, settings=settings,
                                                         credential_profile_id=account)["credential_profile_id"])
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
    # The account a restore's send will prefer (``model_route_preferred``) is looked up, never asked as a pin: without
    # its fresh record the question stays the route's own (Auto, or the role's pin). An account observed serving (a
    # receipt) is read as that account, its fingerprint checked.
    carried = (route["provider"] == "claudexor" and bool(route["options"].get("account_fingerprint"))
               and bool(task.get("model_route_preferred")))
    evidence = probe(
        DATA_DIR,
        provider=route["provider"],
        model=route["model"],
        base_url=route["base_url"],
        use_local=route["use_local"],
        allow_fetch=allow_fetch,
        force=bool(task.get("route_read_again")),  # the horizon guard's repeat of its own failed early read
        options=route["options"] or None,
        **({"ask": {"source_id": route["options"]["source_id"], "credential_profile_id": route["account_pin"]}}
           if carried else {}),
    )
    if route["provider"] == "claudexor" and getattr(evidence, "account_fingerprint", ""):  # the account receipts name
        route["options"] = {**route["options"], **{key: getattr(evidence, key) for key in (
            "source_id", "credential_profile_id", "account_fingerprint")}}
        route["window_observation_fp"] = route_observation_scope(route, settings)
    window = int(model_role_option(MODEL_CONTEXT_WINDOWS_KEY, route["model_role"],
                                   settings=settings))
    return route, with_role_window(evidence, window)


def _route_key(task: Dict[str, Any]) -> Optional[str]:
    """The Claudexor route a fit's question asks the engine about (``_context_route``: settings, role, pin and carried
    account), or None: another route's evidence needs no engine question beside the build."""
    try:
        route = _context_route(task)[0]
        return json.dumps(route, sort_keys=True, default=str) if route.get("provider") == "claudexor" else None
    except Exception:
        return None


class RouteQuestion:
    """The fit's route question, asked when the build starts and answered beside it (W2 H4, owner 12.11.2026).

    The same question through the same resolver (``context._context_fit_route``) and the same evidence, in a thread
    running in a copy of the starting context, where the account that just failed is only read. It counts its work in
    a window of its own (``context_timing.apart``), adopted by the build's receipt when the fit takes its answer; a
    voided question's work is not. One handle per build: the fit takes its answer only for the route it asked about
    (``_route_key``, compared before and after the wait), so a changed route asks its own question, and it waits on it
    as on its own read, with no timer.
    ``early`` names an answer taken from it, which ``build_context_fit_plan`` asks ``again`` at the usual point when
    using it would show less memory than the selected view, or none (``narrowed_by_window``, a refused start).
    """

    def __init__(self, task: Dict[str, Any], resolver: Optional[Callable[..., Tuple[Dict[str, Any], Any]]] = None):
        from ouroboros import context

        self.resolver, self.early, self.work = resolver or context._context_fit_route, False, {}
        self.key, self.asked_key, self.answer, self.error = _route_key(task), None, None, None
        self.asked = threading.Event()
        if self.key is None:
            self.asked.set()
            return
        copied = contextvars.copy_context()
        threading.Thread(target=copied.run, args=(self._ask, dict(task)), name="fit-route-question", daemon=True).start()

    def _ask(self, task: Dict[str, Any]) -> None:
        try:
            with apart() as self.work:  # its counts join the build's receipt only when the fit takes this answer
                self.asked_key = _route_key(task)
                self.answer = self.resolver(task, allow_fetch=True)
        except Exception as error:  # the fit meets it where it would have met its own failed read
            self.error = error
        finally:
            self.asked.set()

    def __call__(self, task: Dict[str, Any], *, allow_fetch: bool) -> Tuple[Dict[str, Any], Any]:
        key, self.key, self.early = self.key, None, False  # used once: any later call asks its own question
        if allow_fetch and key is not None and _route_key(task) == key:
            timed("fit.route_wait", self.asked.wait)
            if self.asked_key == key and _route_key(task) == key:  # a route changed during the wait asks its own
                self.early = True
                adopt(self.work)
                if self.error is not None:
                    raise self.error
                return self.answer
        return timed("fit.route", self.resolver, task, allow_fetch=allow_fetch)

    def again(self, task: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        """The same question at the fit's usual point (the horizon guard). When the early answer came from a read that
        failed, the question is read anew: the failure that read wrote in this build answers nothing here."""
        early, self.early = self.answer, False
        failed = isinstance(early, tuple) and len(early) == 2 and getattr(early[1], "read", "") == "failed"
        return timed("fit.route", self.resolver, {**task, "route_read_again": True} if failed else task, allow_fetch=True)


def narrowed_by_window(contents: Mapping[str, Tuple[Any, Mapping[str, Any]]], mode: str, preferred: str,
                       state: Any, delivered: Mapping[str, Any]) -> bool:
    """Whether a start shows less memory than the selected view because of its window: a mode below the owner's, a
    rescued input, host forms or any step of the walk the window took, collapsed task lines (F1) included. Cuts the
    owner's own mode budget took are no narrowing of the window's, but only while that budget is below the window: a
    window at or below the mode's target sets the start bound itself. Under the budget, only the window's physical
    fold of this room's uncovered owner rows is the window's; it needs ordinary memory above the budget and takes F7
    (``memory_floor.fit_memory_view``). An unknown window narrows nothing."""
    if not state.known_window:
        return False
    if delivered or mode != preferred:
        return True
    facts = contents[mode][1] or {}
    floor = facts.get("floor") or {}
    target = context_budget.context_mode_limits(preferred, preferred, state.output_reserve)[0]
    budget = bool(target) and target < state.window_record.total_equivalent
    steps = floor.get("steps") or {}
    cut = any(steps.values())
    owners = budget and not (floor.get("ordinary_unfit") and steps.get("F7"))  # no physical fold: the budget cut alone
    return bool(cut and not (floor.get("by_budget") and owners)) or bool(facts.get("host_sections_addressed") and not budget)


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
                    supplementary_messages_json: str = "[]", current_messages: Optional[List[Dict[str, Any]]] = None,
                    chars: bool = False) -> int:
    tail = (_request_tail(current_messages) if current_messages is not None else
            [{"role": "user", "content": json.loads(user_content_json)}, *json.loads(supplementary_messages_json)])
    return estimate_context_prompt_tokens([{"role": "system", "content": system_content}, *tail], chars=chars)


class MinimumViewUnfit(ValueError):
    """The core, assignment, resident tools and work room cannot fit this route."""
    physical = True  # False: the request itself fits the window, only the start work room beside it does not

    def __init__(self, message: str, *, fixed_tokens: int, memory_tokens: int,
                 reserve_tokens: int, window_tokens: int, ratio: float, work_room_tokens: int = 0,
                 required_book_tokens: int = 0):
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
        # Set only when the ordinary view fits and this task's required handbook alone does not: never a rescue's.
        self.required_book_tokens = math.ceil(required_book_tokens * ratio)
        super().__init__(f"{message}: fixed input={self.fixed_input_tokens}, memory view={self.memory_view_tokens}, "
                         f"reply/working reserve={reserve_tokens}, window={window_tokens}, "
                         f"deficit={self.deficit_tokens}, work room={work_room_tokens}, "
                         f"fixed/work-room deficit={self.work_room_deficit_tokens} tokens"
                         + (f", required handbook={self.required_book_tokens}" if required_book_tokens else ""))


def _view_projections(core: ContextCore, governance: Mapping[str, Tuple[Any, ...]], user_content_json: str, *, preferred: str,
                      tool_schemas: Optional[List[Dict[str, Any]]], window_tokens: int, known_window: bool, output_reserve: int,
                      ratio: float, start: Optional[str] = None, resident: bool = False,
                      current_messages: Optional[List[Dict[str, Any]]] = None,
                      kept: Optional[Mapping[str, Any]] = None, saving: Optional[Mapping[str, int]] = None,
                      select: Optional[str] = None, kept_sections: Optional[Mapping[str, Iterable[Mapping]]] = None,
                      working_reserve_tokens: Optional[int] = None, window_record: Optional[WindowRecord] = None, work_room_levels: int = 1,
                      envelope_overhead_tokens: int = 0, start_work: bool = False, slack: int = 0, retain: Any = None,
                      ) -> Tuple[Dict[str, Tuple[List[Dict[str, Any]], Dict]], str]:
    """Select books once, choose memory in one walk. Unsent book forms share the selected memory, not another budget.

    Every automatic choice reads the ordinary view of this capture; a directly required handbook joins it last with
    its own note, so neither costs a form, mode or schema (owner 1.11.2026). At a fresh start on a known window the
    ordinary estimate, the book, its note and the reply the ordinary view keeps must fit, else the named refusal.
    ``retain``: how a host step captures its section (default ``section_floor.retain_section``).
    """
    from dataclasses import replace
    from ouroboros import memory_floor as floor
    from ouroboros.memory_view import FULL_VIEW, json_chars, room_chars, snapshot_from_json, story_chars
    from ouroboros.tool_policy import select_tool_schemas

    whole_core, whole_governance = core, governance
    if core.development_required:
        core = replace(core, development_required=False)
        governance = {name: _governance_blocks(None, core, mode=name) for name in governance}
    mode = select or start or preferred
    form = {m: "low" if core.compact_reference_docs or m == "nano" else m for m in floor.MODES}
    snapshot = snapshot_from_json(core.memory_view_json) if core.memory_view_json else None
    names = [schema["function"]["name"] for schema in tool_schemas or []] if resident else None

    sections = {m: (*core.semi_stable_sections, *core.dynamic_sections,
                    *core.dynamic_head_sections,
                    *(governance[form[m]][0] if not isinstance(governance[form[m]][0], str) else ()))
                for m in form}
    previous = tuple((kept_sections or {}).get(start or preferred, ()))
    receipts, room_fact = list(previous), None  # room_fact: the start's Max room, named by a view lowered from Max
    held = (kept or {}).get(start or preferred) or FULL_VIEW
    story_before, room_before = (story_chars(snapshot, held), room_chars(snapshot, held)) if snapshot else (0, 0)

    # One serialization per request shape; a host section's shorter form and a memory text then add or take only
    # their own characters (JSON escapes per character, and these texts are embedded verbatim).
    shapes, text_chars = {}, {}
    def escaped(text):
        if text not in text_chars:
            text_chars[text] = json_chars(text)
        return text_chars[text]
    def measure(selected, sent, addressed, story=0, room=0):  # ``story``, ``room``: the memory texts' characters
        key = (selected, sent.chosen, bool(story), bool(room))
        if key not in shapes:  # one-letter memory texts hold the shape; only presence changes separators and blocks
            counted("request_forms")
            whole = _request_tokens(_system_blocks(core, governance[form[selected]], "s" if story else "", "r" if room else "",
                mode=selected, addressed={}), user_content_json, supplementary_messages_json=core.supplementary_messages_json,
                current_messages=current_messages, chars=True)
            shapes[key] = whole - bool(story) - bool(room), tool_schema_tokens(list(sent.schemas))
        base, schemas = shapes[key]
        full = {part.section_id: part.full_text for part in sections[selected]}
        forms = sum(escaped(short) - escaped(full[name]) for name, short in addressed.items())
        return max(1, (base + forms + story + room + 3) // 4) + schemas + envelope_overhead_tokens
    asked = (saving or {}).get(start or preferred, 0)  # the shown request's saving, in estimator tokens
    original_fixed = measure(start or preferred, select_tool_schemas(tool_schemas or [], context_mode=mode,
        schema_names=names), section_views(sections[start or preferred], previous))
    original_reserve = context_budget.context_mode_limits(mode, preferred, output_reserve, working_reserve_tokens=working_reserve_tokens)[1]
    def owed(fixed, reserve):  # what a frame still owes of it once its books, host forms and reserve paid their part
        return max(0, asked - (original_fixed - fixed) - math.floor((original_reserve - reserve) / max(ratio, 0.01)))
    def view(selected, sent, addressed, window, boundary, fixed=None, **walk):  # one frame's memory beside host forms
        fixed = measure(selected, sent, addressed) if fixed is None else fixed
        return floor.render_view_for_mode(snapshot, mode=selected, owner_mode=preferred, window_tokens=window, **walk,
            known_window=known_window, output_reserve=output_reserve, ratio=ratio, non_memory_tokens=fixed,
            # The Max room as the receipt settles it (``with_max_room`` decides its cause), so the floor note of a
            # start memory lowered names what holds the room.
            lowered_from=preferred if selected != preferred else None, start_work=start_work,
            max_room=floor.with_max_room({}, room_fact, selected).get("max_room"),
            tool_names=None if tool_schemas is None else sent.chosen, working_reserve_tokens=working_reserve_tokens,
            boundary_tokens=boundary, measure_text=lambda *memory: measure(selected, sent, addressed, *memory) - fixed)

    def frame(selected):
        nonlocal receipts
        sent = select_tool_schemas(tool_schemas or [], context_mode=selected, schema_names=names)
        target, reserve = context_budget.context_mode_limits(selected, preferred, output_reserve,
                                                            working_reserve_tokens=working_reserve_tokens)
        record = replace(window_record, reserve=reserve) if window_record else None
        window = record.total_equivalent if record and record.usable is not None else window_tokens
        boundary = record.boundary if record else None
        levels = snapshot.spec.work_room_levels if snapshot else work_room_levels
        host_allowances = floor.floor_allowances(window_tokens=window if known_window else None,
            output_reserve_tokens=reserve, non_memory_tokens=0, target_tokens=target if kept is None else None,
            calibration_ratio=ratio, work_room_levels=levels, boundary_tokens=boundary)
        held = [None]  # the frame's one shortest level: a host step's receipt re-prices it, a miss walks again
        def shortest(forms, physical):  # what a start sends once its own walk took each form before uncovered owner rows
            shown = held[0] and view(selected, sent, forms, window, boundary, level=held[0])
            price = shown and measure(selected, sent, forms, shown.story_chars, shown.room_chars)
            if price is None or price > physical:  # no walk yet, or its level no longer fits beside these host forms
                shown = view(selected, sent, forms, window, boundary, shortest=True)
                held[0], price = floor.kept_level(snapshot, shown.facts), measure(selected, sent, forms, shown.story_chars, shown.room_chars)
            return price
        receipts, _ = floor._host_steps(sections[selected], previous,  # every frame starts from what was shown
            measure=lambda addressed: measure(selected, sent, addressed, story_before, room_before),
            shortest=shortest if snapshot and kept is None else None,  # a held view was sent: it keeps its steps
            window=window if known_window else None, reserve=reserve, ratio=ratio, slack=slack,
            saving=owed(measure(selected, sent, section_views(sections[selected], previous)), reserve) if asked else 0,
            root=core.section_root, task_id=core.section_task_id, bare=lambda forms: measure(selected, sent, forms),
            allowance=host_allowances["margin" if kept is None or start_work else "physical"],
            budget_mode=selected if kept is None and target and target < window else None,
            **({"retain": retain} if retain is not None else {}))
        fixed = measure(selected, sent, section_views(sections[selected], receipts))
        allowances = floor.floor_allowances(window_tokens=window if known_window else None,
            output_reserve_tokens=reserve, non_memory_tokens=fixed, target_tokens=target if kept is None else None,
            calibration_ratio=ratio, work_room_levels=levels, boundary_tokens=boundary)
        return sent, fixed, reserve, window, boundary, allowances

    sent, fixed, reserve, window, boundary, allowances = timed(f"fit.frame_{mode}", frame, mode)
    def work():
        target = context_budget.context_mode_limits(mode, preferred, output_reserve)[0]
        return math.ceil(min(boundary or window, target or window) / context_budget.RECLAIM_LOW_WATER_DIVISOR) * (
            snapshot.spec.work_room_levels if snapshot else work_room_levels)

    # Both Max decisions use the clean-start guard's minimum ordinary memory,
    # priced in this request with its receipt, schemas, reserve and work room.
    ordinary_measures = {}
    def ordinary_need():
        if mode in ordinary_measures:
            return ordinary_measures[mode]
        forms = section_views(sections[mode], receipts)
        minimum = view(mode, sent, forms, window, boundary, level=floor.ordinary_level(snapshot)) if snapshot else floor.NO_VIEW
        price = measure(mode, sent, forms, minimum.story_chars, minimum.room_chars)
        ordinary_measures[mode] = minimum, math.ceil(price * ratio) + reserve + work()
        return ordinary_measures[mode]
    minimum, need = ordinary_need() if (kept is None or start_work) and mode == "max" else (floor.NO_VIEW, 0)
    room_fact = floor.max_room_fact(snapshot, rendered_texts(core, section_views(sections[mode], receipts)),
        render_governance(governance[form[mode]], section_views(sections[mode], receipts)), story=floor.tokens_of(minimum.story_chars),
        fixed=fixed, ratio=ratio, schemas=tool_schema_tokens(list(sent.schemas)), window=window, reserve=reserve,
        work=work(), allowance=allowances["margin"]) if (kept is None or start_work) and mode == "max" else None
    if room_fact:
        room_fact.update(need=need, story=floor.tokens_of(minimum.story_chars), room=floor.tokens_of(minimum.room_chars))
    if room_fact and floor.max_room_lost(room_fact):
        mode = "low"
        sent, fixed, reserve, window, boundary, allowances = timed(f"fit.frame_{mode}", frame, mode)

    def render():  # this frame's own completed memory view, its floor receipt included
        return view(mode, sent, section_views(sections[mode], receipts), window, boundary, saving=owed(fixed, reserve),
                    fixed=fixed, kept=None if kept is None else kept.get(start or preferred)) if snapshot else floor.NO_VIEW

    def overflows(fit=floor.NO_VIEW):  # the known window cannot hold this frame's request beside this memory view
        memory = floor.tokens_of(fit.story_chars) + floor.tokens_of(fit.room_chars)
        return known_window and math.ceil((fixed + memory) * ratio) + reserve > window

    # Books step down on the frame's own completed view. Max keeps whole books beside
    # ordinary memory and the work room; the same ordinary Nano is the last physical rescue for every
    # Main route, a view already shown included. A frame whose core alone cannot fit steps down unrendered. A frame
    # is judged by its pieces' prices: a frame the fit walks away from is never drawn whole.
    while True:
        shown = floor.NO_VIEW if mode != "nano" and overflows() else render()
        books_lost = ((kept is None or start_work) and mode == "max" and allowances["margin"] is not None
                      and ordinary_need()[1] > window)
        if mode == "nano" or not (books_lost or overflows(shown)):
            break
        mode = "low" if mode == "max" else "nano"
        sent, fixed, reserve, window, boundary, allowances = timed(f"fit.frame_{mode}", frame, mode)
    if kept is None and overflows(shown):  # a start's refusal names its last view, never a work room's shortfall
        core_alone = overflows()
        raise MinimumViewUnfit("Core, assignment, resident control tools and work room cannot fit" if core_alone
                               else "Protected owner words and the minimum view cannot fit the usable input",
                               fixed_tokens=fixed, memory_tokens=floor.tokens_of(shown.story_chars) + floor.tokens_of(shown.room_chars),
                               reserve_tokens=reserve, window_tokens=window, ratio=ratio,
                               work_room_tokens=work() if core_alone else 0)
    story, room, facts = shown.story, shown.room, shown.facts  # the one view this start delivers: drawn here
    book = whole_core.required_book_tokens(mode)
    ordinary = book and fixed + floor.view_tokens(story) + floor.view_tokens(room)  # the ordinary admission's estimate
    note = floor.required_book_note(lowered=bool(room_fact) and mode == "low", book_tokens=math.ceil(book * ratio), work=work(),
        free=window - reserve - math.ceil((ordinary + book) * ratio) if known_window and (kept is None or start_work) else None
    ) if book and snapshot else ""
    if note:  # its own cost, like the book's, stays outside every later automatic choice (``required_book_tokens``)
        room, facts = "\n\n".join(filter(None, (room, note))), {**facts, BOOK_NOTE_KEY: floor.view_tokens("\n\n" + note)}
    receipt = floor.view_receipt(snapshot, story, room, floor.with_max_room(facts, room_fact and (
        {**room_fact, "book_required": True} if book else room_fact), mode, required_book=math.ceil(book * ratio))) if snapshot else {}
    if receipts:
        receipt["host_sections_addressed"] = receipts
    if book and kept is None and known_window:
        reply = context_budget.reply_allowance_tokens(caller_max_tokens=output_reserve, nano=mode == "nano", owner_nano=False,
                                                      input_tokens=math.ceil(ordinary * ratio), raw_input_tokens=ordinary, window_tokens=window)
        memory = floor.view_tokens(story) + floor.view_tokens(room)
        if math.ceil((fixed + book + memory) * ratio) + max(reserve, reply) > window:
            raise MinimumViewUnfit("This task's directly required handbook cannot fit beside its ordinary view",
                                   fixed_tokens=fixed + book, memory_tokens=memory, reserve_tokens=max(reserve, reply),
                                   window_tokens=window, ratio=ratio, required_book_tokens=book)
    counted("view_texts_kept", 2 if snapshot else 0)  # the story and the room text this fit delivers
    return {m: (_system_blocks(whole_core, whole_governance[form[m]], story, room, m,
        section_views(sections[m], receipts)), receipt) for m in floor.MODES}, mode


def _whole_request(core, governance, mode, tool_schemas):
    """Uncut selected request for a route's comparable density witness, before any floor step: the request with
    one-letter memory texts, its tools, and the characters the whole story and room add beyond those letters."""
    from ouroboros.memory_view import room_chars, snapshot_from_json, story_chars
    from ouroboros.tool_policy import select_tool_schemas

    snapshot = snapshot_from_json(core.memory_view_json) if core.memory_view_json else None
    story, room = (story_chars(snapshot), room_chars(snapshot)) if snapshot else (0, 0)
    form = "low" if core.compact_reference_docs or mode == "nano" else mode
    messages = [{"role": "system", "content": _system_blocks(core, governance[form], "s" if story else "", "r" if room else "", mode)},
                {"role": "user", "content": json.loads(core.user_content_json)},
                *json.loads(core.supplementary_messages_json)]
    return messages, list(select_tool_schemas(tool_schemas or [], context_mode=mode).schemas), story + room - bool(story) - bool(room)


START_CAUSES = ("", "memory", "route", "window")  # ``memory_floor.start_cause``
UNKNOWN_START = ("unknown", "")  # a restored task's start its continuation did not carry: never the restore's fit


def carried_start(carried: Any) -> Tuple[str, str]:
    """A continuation's stored start (``owner_wait.continuation_state``) as ``ContextFitPlan.start``: the task's own
    (mode, cause), or ``UNKNOWN_START`` when it is missing, malformed or already unknown (a source written before
    the start was carried); nothing is read to recover it."""
    if (isinstance(carried, (list, tuple)) and len(carried) == 2 and carried[0] in ("max", "low", "nano")
            and carried[1] in START_CAUSES):
        return str(carried[0]), str(carried[1])
    return UNKNOWN_START


def restored_start(plan: Any, carried: Any) -> Any:
    """``plan`` holding a continuation's carried start (``carried_start``): a cold restore's new fit is this process's
    capture, never the task's start, so an uncarried one stays unknown through every later checkpoint and rebind."""
    from dataclasses import replace

    return replace(plan, start=carried_start(carried)) if isinstance(plan, ContextFitPlan) else plan


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
    """Capture route facts, choose books and fit one start view."""
    preferred = str(preferred_mode or "max").strip().lower()
    if preferred not in {"low", "max", "nano"}:
        preferred = "max"

    meta = task.get("task_metadata") if isinstance(task.get("task_metadata"), dict) else {}
    user_content = json.loads(core.user_content_json)
    governance = {form: _governance_blocks(env, core, mode=form) for form in ("max", "low")}
    from ouroboros.capability_evidence import evidence_model_route, window_evidence
    from ouroboros.memory_view import start_room, view_role

    view_task = {**task, "metadata": task.get("metadata") or meta}
    role = view_role(view_task)
    view_kind = {"consciousness": "own_wake", "presence": "focus", "integrator": "owner_turn"}.get(role, role)
    # Select density before the single start walk, comparing the whole captured request at the size that can be sent.
    rows, schemas, memory_chars = _whole_request(core, governance, preferred, tool_schemas)
    counted("request_forms")
    levels, sized = start_room(view_task, role), {}

    def answer(ask: Callable[[], Tuple[Dict[str, Any], Any]]) -> Tuple[Dict[str, Any], Any]:
        try:  # a child probes its own route too: its parent's warm evidence is no fact about it
            return ask()
        except Exception:
            log.debug("Context-fit route evidence unavailable; preserving Max", exc_info=True)
            return _failed_route_evidence(task)

    def frame(route: Dict[str, Any], evidence: Any) -> SimpleNamespace:
        """What the start walk reads of one route answer: reserves, the window record and the density."""
        # Keep the fit projection tied to the physical dispatch contract instead of duplicating its output reservation.
        output_reserve = main_output_reserve_tokens(use_local=bool(route.get("use_local")), evidence=evidence)
        route_facts = getattr(evidence, "route_facts", {}) or {}
        working_reserve = route_facts.get("working_reserve_tokens")
        window_basis = str(route_facts.get("basis") or "")
        reserve = context_budget.context_mode_limits(preferred, preferred, output_reserve,
                                                    working_reserve_tokens=working_reserve)[1]
        scope = route.get("window_observation_fp", evidence.route_fp)
        bound = bool(scope) and (route.get("provider") != "claudexor" or bool((route.get("options") or {}).get("account_fingerprint")))
        window_record = resolve_window(evidence, reserve, bound=bound, observation_fp=scope)
        provider = str(route.get("provider") or "")
        if provider not in sized:  # the whole request measured once per provider, however many answers are framed
            sized[provider] = estimate_context_prompt_tokens(rows, schemas, provider=provider, chars=True)
        sendable = min(max(1, (sized[provider] + memory_chars + 3) // 4), window_record.usable or math.inf)
        ratio = 1.0 if window_basis == "estimator" else _route_calibration_ratio(
            None, str(evidence.route_fp or ""), str(route["model"] or ""), sendable, view_kind=view_kind)
        limits = tuple((limit.tokens, limit.kind) if limit else None for limit in (window_record.input, window_record.total))
        return SimpleNamespace(route=route, evidence=evidence, output_reserve=output_reserve, working_reserve=working_reserve,
                               window_basis=window_basis, bound=bound, window_record=window_record, ratio=ratio,
                               known_window=window_record.usable is not None,
                               walked=(limits, window_record.usable is not None, output_reserve, working_reserve, ratio))

    def walk(state: SimpleNamespace) -> Tuple[Dict[str, Tuple[List[Dict[str, Any]], Dict]], str, Dict[str, Any]]:
        view_args = dict(
            preferred=preferred, tool_schemas=tool_schemas, window_tokens=state.window_record.total_equivalent,
            known_window=state.known_window, output_reserve=state.output_reserve, window_record=state.window_record,
            working_reserve_tokens=state.working_reserve)

        def fit(user_content_json: str) -> Tuple[Dict[str, Tuple[List[Dict[str, Any]], Dict]], str]:
            return _view_projections(core, governance, user_content_json, **view_args, ratio=state.ratio,
                                     work_room_levels=levels)
        try:
            return (*fit(core.user_content_json), {})
        except MinimumViewUnfit as unfit:
            # Physical rescue only (owner 2026-10-13, #1527): an exact source replaces the whole input only when no view
            # of the known window holds it, a wake's bound projection first (``context_input_selection.rescued_start``).
            from ouroboros.context_input_selection import rescued_start
            return rescued_start(env, task, core.user_content_json, unfit, preferred, fit)

    early = isinstance(route_resolver, RouteQuestion)  # it times its own wait (``fit.route_wait``) and read
    state = frame(*answer(lambda: route_resolver(task, allow_fetch=True) if early else timed(
        "fit.route", route_resolver, task, allow_fetch=True)))
    refused = None
    try:
        contents, initial_mode, delivered = walk(state)
    except MinimumViewUnfit as unfit:  # the early answer's window cannot hold the core: the usual point decides
        if not (early and route_resolver.early):
            raise
        refused, contents, initial_mode, delivered = unfit, {}, "", {}
    if early and route_resolver.early and (refused or narrowed_by_window(contents, initial_mode, preferred, state, delivered)):
        # Horizon guard (W2 H4): an answer asked beside the build that would show less than the selected view, or none,
        # is asked again here, at the usual point, and that answer is used (its refusal is final); a walk that reads
        # the same frame is not repeated unless it was refused.
        with another_pass():  # a repeat by design, not work done again (``context_health`` names only the latter)
            late = frame(*answer(lambda: route_resolver.again(task)))
            if refused is not None or late.walked != state.walked:
                contents, initial_mode, delivered = walk(late)
        state = late
    route, evidence, window_record, known_window = state.route, state.evidence, state.window_record, state.known_window
    output_reserve, working_reserve, window_basis, ratio = (state.output_reserve, state.working_reserve,
                                                            state.window_basis, state.ratio)
    bound = state.bound
    estimates = {}  # projections that read alike are one form of the request

    @timed_section("fit.projections")
    def _projection(mode: str) -> ContextFitProjection:
        system_content, memory_facts = contents[mode]
        system_content_json = json.dumps(system_content, ensure_ascii=False, sort_keys=True)
        user_projection = delivered.get(mode)
        form = (system_content_json, user_projection)
        if form not in estimates:
            counted("request_forms")
            estimates[form] = _request_tokens(system_content, user_projection or core.user_content_json,
                                              supplementary_messages_json=core.supplementary_messages_json)
        estimated = estimates[form]
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
            "dynamic_text": core.dynamic_text,
            "memory_view_json": core.memory_view_json,
            **({"connections": list(core.connections)} if any(core.connections) else {}),
            **({"supplementary_messages": json.loads(core.supplementary_messages_json)}
               if json.loads(core.supplementary_messages_json) else {}),
            "user_content": user_content,
            "docs_need_development": core.docs_need_development,
            **({"development_required": True} if core.development_required else {}),
            "compact_reference_docs": core.compact_reference_docs,
            "reference_book_errors": core.reference_book_errors,
            "reference_sources": [(book.book_id, source.source_path, source.sha256)
                                  for book in core.reference_books for source in (book.entrypoint, *book.chapters)],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    from ouroboros.memory_floor import start_cause

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
        model_route=evidence_model_route(evidence, route["model"]) if route["provider"] == "claudexor" else {},
        evidence_source=window_record.source, window_record=window_record, window_witness_bound=bound,
        route_base_url=str(route.get("base_url") or ""), account_pin=str(route.get("account_pin") or ""),
        window_evidence=window_evidence(evidence, window_record),
        core=core, start_tool_schemas_json=json.dumps(tool_schemas, ensure_ascii=False, sort_keys=True),
        start=(initial_mode, "" if initial_mode == preferred else start_cause(contents[initial_mode][1])),
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
        required_book_tokens=measurement.required_book_tokens, preferred_mode=measurement.preferred_mode,
        mode_cause=measurement.mode_cause,
    )
