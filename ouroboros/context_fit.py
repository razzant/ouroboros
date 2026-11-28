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
from dataclasses import dataclass, field, replace
from functools import partial
from types import SimpleNamespace
from typing import Any, Iterable, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.context_measurement import (
    ContextProfile as ContextProfile, MeasurementBasis as MeasurementBasis,
    MainFitMeasurement as MainFitMeasurement, MainFitDisposition as MainFitDisposition,
    measure_main_fit as measure_main_fit, reclaim_low_water_margin as reclaim_low_water_margin,
    bounded_prompt_tokens_for_payload as bounded_prompt_tokens_for_payload,
    effort_lengthens_request as effort_lengthens_request, estimate_context_prompt_tokens as estimate_context_prompt_tokens,
    messages_carry_native_images as messages_carry_native_images, payload_prompt_rows as payload_prompt_rows,
    tool_schema_tokens as tool_schema_tokens, FIRST_CONTACT_START as FIRST_CONTACT_START, _custom_ladder,
    BOOK_NOTE_KEY, required_book_addition, _physical_context_for_fit as _physical_context_for_fit,
)

from ouroboros.context_layout import reference_doc_sections, whole_development_book
from ouroboros.section_floor import (HostSection, _governance_blocks as _section_governance, _system_blocks,
    capture_system_projection, predicted_section, projection_fields, render_governance, rendered_texts, section_views,
    with_program_history)
from ouroboros.reference_books import ReferenceBook
from ouroboros import context_budget
from ouroboros.chronicle_store import DELEGATED_FOCUS_ROLES
from ouroboros.context_window import WindowRecord, resolve_window, route_observation_scope
from ouroboros.tool_result_delivery import project_tool_result_batch  # noqa: F401
from ouroboros.context_timing import adopt, another_pass, apart, counted, timed, timed_section

log = logging.getLogger(__name__)

_Schemas = Optional[List[Dict[str, Any]] | Callable[[], List[Dict[str, Any]]]]
_ToolFrames = Optional[Callable[[str], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]]
DENSITY_VIEW_KINDS = {"consciousness": "own_wake", "presence": "focus", "integrator": "owner_turn"}  # role -> witness view


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
    assembly_witnesses: tuple = ()  # host side only; never a message field
    # A route rebind may measure the actual shortened transcript, not the initial tail.
    continuation_messages_json: Optional[str] = None
    room_data: str = ""

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
    start_nano_base: Optional[Tuple[str, ...]] = None  # a live call's own Nano tools, priced by its start
    start_reasoning_effort: str = ""  # effort paired with this frame's provider for next-start pricing; ``start`` keeps history
    # The start fit's own mode and cause (physical, or a first-contact preference): every reprojection and
    # route rebind keeps it while ``initial_mode`` follows the current selection. ("", ""): not a built start;
    # ``UNKNOWN_START``: a restored task whose continuation did not carry its start (``restored_start``).
    start: Tuple[str, str] = ("", "")
    # A fresh direct chat's start-hint scope as its fit read it ("": not such a plan, e.g. a root, wake or restore),
    # and the ordinary projections a hint replaced (three modes, selection and start), until a dispatch of the hinted
    # view (``loop_routes._FirstContactHint``) or a rebind to another scope (``loop_routes.first_contact_route``).
    first_contact: str = ""
    first_contact_normal: Optional[Tuple[ContextFitProjection, ContextFitProjection, ContextFitProjection,
                                         str, Tuple[str, str]]] = field(default=None, compare=False)

    def first_contact_spent(self) -> "ContextFitPlan":
        """After its first physical send a hint answers to that send's own outcome; no ordinary form stays to restore."""
        return replace(self, first_contact_normal=None) if self.first_contact_normal is not None else self

    def ordinary_first_contact(self) -> "ContextFitPlan":
        """The view this plan would have sent without the start hint (another scope dispatches it)."""
        if self.first_contact_normal is None:
            return self
        max_projection, low_projection, nano_projection, initial_mode, start = self.first_contact_normal
        return replace(self, max_projection=max_projection, low_projection=low_projection, nano_projection=nano_projection,
                       initial_mode=initial_mode, start=start, first_contact_normal=None)

    requested_load_window: Optional[int] = None  # the selected physical load, including Auto (0)

    def start_schema_names(self, mode: str) -> Optional[Tuple[str, ...]]:
        """The schema names the start priced, selected for ``mode`` as the start selected them (a live call's own
        Nano base, else the standing work set); None when the start priced none or was not captured."""
        from ouroboros.tool_policy import select_tool_schemas

        start, base = json.loads(self.start_tool_schemas_json or "null"), self.start_nano_base
        return None if start is None else select_tool_schemas(
            start, context_mode=mode, schema_names=base, standing=base is None).chosen

    def reproject_for_route(self, *, window_tokens: int, known_window: bool, ratio: float, output_reserve: int,
                            tool_schemas: Optional[List[Dict[str, Any]]], start_mode: Optional[str] = None,
                            current_messages: Optional[List[Dict[str, Any]]] = None,
                            refused_tokens: int = 0, select_mode: Optional[str] = None,
                            measured_input_tokens: Optional[int] = None, start_work: bool = False,
                            slack_tokens: int = 0, room_if_narrowed: bool = False,
                            admitted: bool = False, reasoning_effort: str = "",
                            restoring_held: bool = False, telling_floor: bool = False,
                            books_must_save: bool = False, books_rung: bool = False,
                            nano_tools: _Schemas = None, low_tools: _Schemas = None,
                            tool_frames: _ToolFrames = None, unsent_start: bool = False) -> "ContextFitPlan":
        """Fit shown bytes; the prepared envelope's raw estimate includes its signed size difference.

        The measurement belongs to this candidate, includes tools and excludes the reply; withholding an image can shrink it.
        It is the estimator's one count of messages and tools in the shape this plan's provider and the next send's
        ``reasoning_effort`` give them, the count a round measures (``context_measurement.measure_main_fit``).
        A requested saving alone does not establish pressure on held owner words or notes.
        ``slack_tokens`` asks room beyond the measured or refused excess: memory and ordinary host forms may pay it,
        a whole-or-outline section never (``memory_floor._host_steps``).
        ``admitted``: the plan's own window is read at ``WindowRecord.admitted``, the line this work's own requests go
        out at (Main's rebind, restore, retry in a lower mode and memory pass); without it at ``usable``, as a start
        is sized (voice; a work's first start is ``_build_context_fit_plan``).
        ``room_if_narrowed`` (Main's rebind): a view that must narrow for the window is written again anyway, so with
        its excess it asks a wall's slack (``REFUSAL_PASSES``), one eighth, which memory detail and ordinary host
        forms pay; a whole-or-outline section pays the measured excess alone. ``telling_floor`` tries my tellings in
        Max before the Low frame or deeper forms, against the measured saving including slack. A view that fits is never
        touched, at a restore and on a place that holds no copy of it too: nothing is taken on the guess that the
        work will reach its wall (owner, 30 Sep 2026). ``start_work`` is a first send's own room and Max decision (voice).
        ``books_rung``: a refusal's books rung, whose Low frame carries the Body Map at its parts and addresses (all
        that rung has to give); any other selected Low leaves the Map to the window's own test. ``books_must_save``:
        the measured rung, which also rejects a frame that does not shrink the same held view.
        ``low_tools`` and ``nano_tools`` are the caller's alternative schema sets, read lazily. A ``tool_frames``
        caller supplies each mode's fitted schemas and their notice rows together; only it adopts the selected pair.
        ``unsent_start``: no request of this work was read yet (``loop_round_limits.start_unread``): a start's Map sparing.
        """
        ratio = 1.0 if self.window_basis == "estimator" else ratio
        reserve_args = ({"working_reserve_tokens": self.working_reserve_tokens}
                        if self.working_reserve_tokens is not None else {})
        contents, start, held_cost = {}, start_mode or self.preferred_mode, None
        refit = False
        reserve = context_budget.context_mode_limits(start, self.preferred_mode, output_reserve, **reserve_args)[1]
        # The plan's own window, named in either of its frames (a start's, or the admitted one a round measures in);
        # any other number is a different physical route frame the caller supplied, and stands as given.
        own = resolve_window(self, reserve, bound=self.window_witness_bound) if known_window else None
        if own is not None and (own.usable is None or window_tokens not in (
                self.window_tokens, own.total_equivalent, own.admitted_equivalent)):
            own = None
        record, frame = own if self.window_record else None, window_tokens
        if own is not None:  # the plan keeps the start's frame: input limits keep their bound when the reserve changes
            frame, window_tokens = own.admitted_equivalent if admitted else own.total_equivalent, own.total_equivalent
        if current_messages is None:
            retained = self.projection(start).continuation_messages_json
            if retained is not None:
                current_messages = [self.projection(start).system_message(), *json.loads(retained)]
        tail_json = (json.dumps(_request_tail(current_messages), ensure_ascii=False, sort_keys=True)
                     if current_messages is not None else None)
        if self.core is not None:
            from ouroboros import memory_floor
            from ouroboros.held_notes import HeldNoteSection
            from ouroboros.memory_view import snapshot_from_json
            from ouroboros.tool_policy import select_tool_schemas

            shown = self.projection(start)
            schemas, extra = tool_frames(start) if tool_frames else (tool_schemas or [], [])
            names = [schema["function"]["name"] for schema in schemas]
            sent = select_tool_schemas(schemas, context_mode=start, schema_names=names).schemas
            rows = _request_rows(json.loads(shown.system_content_json), shown.user_content_json or self.user_content_json,
                                 self.core.supplementary_messages_json, current_messages, room_data=shown.room_data)
            raw = estimate_context_prompt_tokens([*rows, *extra] if extra else rows, list(sent),
                                                 provider=self.provider, reasoning_effort=reasoning_effort)
            overhead = int(measured_input_tokens) - raw if measured_input_tokens is not None else 0
            book = required_book_addition(self, start)  # nothing narrows to pay for this book
            request = math.ceil((raw + overhead) * ratio) + reserve - math.ceil(book * ratio)
            room = math.ceil((own.boundary if own else window_tokens) / context_budget.RECLAIM_LOW_WATER_DIVISOR
                             ) if known_window and window_tokens else 0
            request += room if start_work else 0
            measured, units = (request - frame if known_window and frame else 0), max(ratio, 0.01)
            if room_if_narrowed and measured > 0:  # narrowing writes the start anyway: a wall's slack with it
                refused_tokens, slack_tokens = max(int(refused_tokens or 0), measured), max(slack_tokens, room)
            excess, deficit = max(measured, int(refused_tokens or 0)), max(measured, int(refused_tokens or 0) + slack_tokens)
            walk = deficit > 0 or bool(select_mode)
            if (restoring_held or not walk) and any(isinstance(part, HeldNoteSection) for part in self.core.dynamic_sections):
                held_cost = _held_request_cost(self.core, start, list(sent), current_messages,
                                               provider=self.provider, reasoning_effort=reasoning_effort)
                # The cold build priced the saved request. Restore can add rows or adopt current schemas before sending.
                refit = restoring_held and (held_cost != shown.memory_facts.get("held_note_request_tokens")
                                            or ratio != shown.calibration_ratio)
            if walk or refit:
                snapshot = snapshot_from_json(self.core.memory_view_json) if self.core.memory_view_json and not refit else None
                contents, start = _view_projections(
                    self.core, {form: _governance_blocks(None, self.core, mode=form) for form in ("max", "low")},
                    self.user_content_json, preferred=self.preferred_mode, start=start_mode, select=select_mode,
                    tool_schemas=tool_schemas, working_reserve_tokens=self.working_reserve_tokens,
                    window_tokens=frame, known_window=known_window, output_reserve=output_reserve, ratio=ratio,
                    window_record=own, admitted=admitted,  # each mode's frame is read off the window (its reserve differs)
                    envelope_overhead_tokens=overhead, provider=self.provider, reasoning_effort=reasoning_effort,
                    start_work=start_work,
                    resident=True, current_messages=current_messages,  # actual running tail, enable_tools included
                    kept=None if refit else {mode: memory_floor.kept_level(snapshot, self.projection(mode).memory_facts) if snapshot else None
                          for mode in memory_floor.MODES},
                    kept_sections=None if refit else {mode: self.projection(mode).memory_facts.get("host_sections_addressed", ())
                                   for mode in memory_floor.MODES},
                    saving=None if refit else {start: math.ceil(deficit / units)},
                    slack=0 if refit else math.ceil(deficit / units) - math.ceil(excess / units),
                    books_must_save=books_must_save, books_rung=books_rung or books_must_save,
                    nano_tools=nano_tools, low_tools=low_tools, tool_frames=tool_frames,
                    running_limit=(raw + overhead - book - math.ceil(deficit / units)
                                   if telling_floor and deficit > 0 and not refit else None), unsent_start=unsent_start)
                held_cost = contents[start][1].get("held_note_request_tokens") if contents else None
                if books_must_save and not contents:
                    select_mode = None  # a rejected trial does not select a mode or publish a view
        elif books_must_save and select_mode:
            # Without a captured core, the only comparable change is between the two stored views: system rows and data.
            def stored_price(mode):
                return _request_tokens(json.loads(self.projection(mode).system_content_json), self.user_content_json,
                    current_messages=current_messages, tools=tool_schemas, provider=self.provider, reasoning_effort=reasoning_effort,
                    room_data=self.projection(mode).room_data)
            if stored_price(select_mode) >= stored_price(start):
                select_mode = None

        def narrowed(projection: ContextFitProjection) -> bool:
            # Unchanged levels keep their bytes unless pressure requires a different receipt; equal counts
            # from another level never keep a view's budget provenance.
            old, new = (dict(facts.get("floor") or {}) for facts in (projection.memory_facts or {},
                                                                     contents[projection.mode][1] or {}))
            return (any(old.get(key) != new.get(key) for key in ("forms_version", "steps", "runs", "by_budget", "target_tokens", "mode_switch"))
                    or projection.memory_facts.get("host_sections_addressed")
                    != contents[projection.mode][1].get("host_sections_addressed")
                    # Under measured pressure, a shorter receipt can pay the excess without another floor step.
                    or ((deficit > 0 or refit) and projection.mode == start
                        and (json.loads(projection.system_content_json) != contents[projection.mode][0]
                             or projection.room_data != contents[projection.mode][2])))

        def project(projection: Optional[ContextFitProjection]) -> Optional[ContextFitProjection]:
            if projection is not None and projection.mode in contents and narrowed(projection):
                system, facts, data = contents[projection.mode]
                number = (projection.memory_facts.get("floor") or {}).get("no_memory_tokens")  # so does Low's miss (3A)
                if number is not None and isinstance(facts.get("floor"), dict):
                    facts = {**facts, "floor": {**facts["floor"], "no_memory_tokens": number}}
                projection = replace(projection, system_content_json=json.dumps(system, ensure_ascii=False, sort_keys=True),
                                     **projection_fields(system, facts | {key: projection.memory_facts[key]
                                         for key in ("max_room", "first_contact_preference") if key in projection.memory_facts}),
                                     continuation_messages_json=tail_json, room_data=data,
                                     estimated_tokens=_request_tokens(system, projection.user_content_json or self.user_content_json,
                                         supplementary_messages_json=self.core.supplementary_messages_json,
                                         current_messages=current_messages, room_data=data))
            elif projection is not None and current_messages is not None:
                projection = replace(projection, continuation_messages_json=tail_json,
                    estimated_tokens=_request_tokens(json.loads(projection.system_content_json),
                        projection.user_content_json or self.user_content_json, current_messages=current_messages,
                        room_data=projection.room_data))
            if projection is not None and projection.mode == start and held_cost is not None:
                projection = replace(projection, memory_facts={**projection.memory_facts,
                                                               "held_note_request_tokens": held_cost})
            calibrated = int(int(projection.estimated_tokens or 0) * ratio) if projection is not None else 0
            return projection and replace(projection, calibrated_tokens=calibrated, calibration_ratio=ratio)

        # An unsent hint's ordinary form stays beside it until a send spends it or a route rebind moves the scope.
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
        tail = json.loads(projection.continuation_messages_json) if projection.continuation_messages_json is not None else [
            {"role": "user", "content": json.loads(projection.user_content_json or self.user_content_json)},
            *json.loads(self.core.supplementary_messages_json if self.core is not None else "[]"),
        ]
        return [projection.system_message(), *with_program_history(tail, projection.room_data)]

    def reproject_transcript(
        self,
        messages: List[Dict[str, Any]],
        mode: str,
    ) -> List[Dict[str, Any]]:
        """Replace the captured memory frame, not its dialogue/tool tail."""
        if not messages:
            return self.messages_for(mode)
        projection = self.projection(mode)
        return [projection.system_message(), *with_program_history(_request_tail(messages), projection.room_data)]

    def projected_tokens_with_tools(
        self,
        mode: str,
        tools: Optional[List[Dict[str, Any]]],
        *,
        provider: str = "",
        reasoning_effort: str = "",
    ) -> int:
        """Calibrated physical prompt projection after schemas are available: the estimator's one count of this
        mode's messages and ``tools`` in their longest shape for ``provider`` and ``reasoning_effort``."""
        return int(estimate_context_prompt_tokens(self.messages_for(mode), tools, provider=provider,
                                                  reasoning_effort=reasoning_effort) * self.projection(mode).calibration_ratio)

@dataclass(frozen=True)
class ContextCore:
    """Single captured context source rendered into deterministic projections."""

    base_prompt: str
    bible_md: str
    # The resident Body Map as every mode and focus carries it (``reference_books.body_map``), and below
    # ``body_map_parts``, what stands in its place when a physical window cannot hold it ("" when it has no
    # shorter form). ``development_md`` is the composed handbook: sent only as this task's own requirement.
    body_map: str
    development_md: str
    semi_stable_text: str
    dynamic_text: str
    user_content_json: str
    body_map_parts: str = ""
    reference_books: Tuple[ReferenceBook, ...] = ()  # the sources the Map was rendered from, for its receipts
    # A delegated child or a declared-input focus (``context._capture_context_core``): never an integrating view.
    delegated_focus: bool = False
    reference_book_errors: Tuple[str, ...] = ()
    # Knowledge (overview, index, project journal) leading the changing block.
    dynamic_head_text: str = ""
    # The captured memory view (``memory_view.snapshot_json``); each projection renders
    # it for its own mode and window. Empty: no memory view was captured.
    memory_view_json: str = ""
    # That capture's journal boundary (``MemoryViewSnapshot.chronicle_sequence``), beside the view and not hashed.
    memory_view_sequence: Optional[int] = None
    supplementary_messages_json: str = "[]"
    opening_program_address: str = ""  # the same canonical note already has the assignment's data role
    semi_stable_sections: Tuple[HostSection, ...] = ()
    dynamic_head_sections: Tuple[HostSection, ...] = ()
    dynamic_sections: Tuple[HostSection, ...] = ()
    section_root: str = ""
    section_task_id: str = ""
    # The connections map (``tool_connections.connection_views``): the Low/Max text, then the Nano text.
    connections: Tuple[str, str] = ("", "")
    # This task's own direct requirement of the whole handbook (``development_required``).
    development_required: bool = False
    # Repo-relative path, raw digest, normalized character count and digest.
    base_prompt_source: Optional[Tuple[str, str, int, str]] = None
    bible_source: Optional[Tuple[str, str, int, str]] = None
    initial_attachment_facts: Mapping[str, Any] = field(default_factory=dict, compare=False)
    held_note_resume: Optional[Mapping[str, Any]] = None

    def required_book_tokens(self) -> int:
        """What the directly required handbook adds to the ordinary view, the same in every mode.

        The window charges it; owner targets, memory or transcript narrowing and rescues are sized without it.
        Only a first request that cannot go out with it yields for it (``memory_floor.yield_for_book``). A missing
        source adds nothing: the capture discloses the gap instead of a partial book.
        """
        if not self.development_required or not self.development_md.strip():
            return 0
        # Characters as the request estimate counts them: its own system item (comma included); the estimate is
        # chars/4 over the whole request.
        item = len(json.dumps({"text": whole_development_book(self.development_md), "type": "text"}, ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"))) + 1
        return max(0, (item + 3) // 4)


def _governance_blocks(env: Any, core: ContextCore, *, mode: str) -> Tuple[Any, ...]:
    return _section_governance(env, core, mode=mode, reference_docs=reference_doc_sections)


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
    from ouroboros.response_limits import bind_context_route
    bind_context_route(route, settings)
    from ouroboros.server_window import bind_native_load
    bind_native_load(route, task.get("load_window_binding") or {}, settings, role)
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
    from ouroboros.capability_evidence import with_role_window
    from ouroboros.response_limits import probe_context_route
    from ouroboros.config import DATA_DIR
    from ouroboros.model_slots import MODEL_CONTEXT_WINDOWS_KEY, model_role_option

    route, settings = _context_route(task)
    # The account a restore's send will prefer (``model_route_preferred``) is looked up, never asked as a pin: without
    # its fresh record the question stays the route's own (Auto, or the role's pin). An account observed serving (a
    # receipt) is read as that account, its fingerprint checked.
    carried = (route["provider"] == "claudexor" and bool(route["options"].get("account_fingerprint"))
               and bool(task.get("model_route_preferred")))
    evidence = probe_context_route(
        DATA_DIR, route, task=task, allow_fetch=allow_fetch,
        **({"ask": {"source_id": route["options"]["source_id"], "credential_profile_id": route["account_pin"]}}
           if carried else {}),
    )
    if route["provider"] == "claudexor" and getattr(evidence, "account_fingerprint", ""):  # the account receipts name
        route["options"] = {**route["options"], **{key: getattr(evidence, key) for key in (
            "source_id", "credential_profile_id", "account_fingerprint")}}
        route["window_observation_fp"] = route_observation_scope(route, settings, target=route.get("_resolved_target"))
    target = route.get("_resolved_target") or {}
    window = route.get("bound_role_window") if target.get("requested_load_window") is not None else None
    if window is None:
        window = int(model_role_option(MODEL_CONTEXT_WINDOWS_KEY, route["model_role"], settings=settings))
    return route, with_role_window(evidence, window)


def _route_key(task: Dict[str, Any]) -> Optional[str]:
    """The Claudexor route a fit's question asks the engine about (``_context_route``: settings, role, pin and carried
    account), or None: another route's evidence needs no engine question beside the build."""
    try:
        route = _context_route(task)[0]
        # The retained dispatch snapshot is not another part of the engine's question.
        route.pop("_resolved_target", None)
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

    def __init__(self, task: Dict[str, Any], resolver: Optional[Callable[..., Tuple[Dict[str, Any], Any]]] = None,
                 *, route_fields: Optional[Dict[str, Any]] = None):
        from ouroboros import context

        # A cold start previews its next send without rewriting the original task. Keep
        # that projection on every read, including the horizon guard's repeat.
        self.route_fields = dict(route_fields or {})
        task = {**task, **self.route_fields}
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
        task = {**task, **self.route_fields}
        key, self.key, self.early = self.key, None, False  # used once: any later call asks its own question
        if allow_fetch and key is not None and _route_key(task) == key:
            timed("fit.route_wait", self.asked.wait)
            if self.asked_key == key and _route_key(task) == key:  # a route changed during the wait asks its own
                self.early = True
                adopt(self.work)
                if self.error is not None:
                    raise self.error
                return self.answer
        return timed("fit.route", self.resolver, {**task, "route_start_read": True}, allow_fetch=allow_fetch)

    def peek(self) -> Optional[Tuple[Dict[str, Any], Any]]:
        """The answer already arrived for the asked route, else None: never waits, consumes the key or adopts work."""
        return self.answer if self.key is not None and self.asked.is_set() and self.error is None and self.asked_key == self.key else None

    def again(self, task: Dict[str, Any], *, failed: bool = False) -> Tuple[Dict[str, Any], Any]:
        """The same question at the fit's usual point (the horizon guard, or a start whose route named no window). When
        the early answer came from a read that failed, or the fit names its first answer failed (``failed``), the
        question is read anew: the failure that read wrote in this build answers nothing here."""
        task = {**task, **self.route_fields}
        early, self.early = self.answer, False
        failed = failed or isinstance(early, tuple) and len(early) == 2 and getattr(early[1], "read", "") == "failed"
        return timed("fit.route", self.resolver, {**task, "route_read_again": True} if failed else task, allow_fetch=True)


def narrowed_by_window(contents: Mapping[str, Tuple[Any, Mapping[str, Any], str]], mode: str, preferred: str,
                       state: Any, delivered: Mapping[str, Any]) -> bool:
    """Whether a start shows less memory than the selected view because of its window: a mode below the owner's, a
    rescued input, host forms or any step of the walk the window took, collapsed task lines (F1) included. Cuts the
    owner's own mode budget took are no narrowing of the window's, but only while that budget is below the window: a
    window at or below the mode's target sets the start bound itself. Under the budget only the window's own steps count:
    the physical fold of this room's uncovered owner rows (F7, needing ordinary memory above the budget) and a host form
    without ``start_room``, which the budget's steps carry (``memory_floor._host_steps``). An unknown window narrows nothing."""
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
    host = facts.get("host_sections_addressed") or ()
    return (bool(cut and not (floor.get("by_budget") and owners))
            or any(not budget or not row.get("start_room") or row.get("held_note_form") for row in host))


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


def route_window(route: Dict[str, Any], evidence: Any, preferred: str) -> Tuple[int, Any, str, bool, WindowRecord]:
    """One route answer's ``(output reserve, working reserve, window basis, bound, window record)`` for every frame."""
    # Keep the fit projection tied to the physical dispatch contract instead of duplicating its output reservation.
    output_reserve = main_output_reserve_tokens(use_local=bool(route.get("use_local")), evidence=evidence)
    route_facts = getattr(evidence, "route_facts", {}) or {}
    working_reserve = route_facts.get("working_reserve_tokens")
    reserve = context_budget.context_mode_limits(preferred, preferred, output_reserve,
                                                working_reserve_tokens=working_reserve)[1]
    scope = route.get("window_observation_fp", evidence.route_fp)
    bound = bool(scope) and (route.get("provider") != "claudexor" or bool((route.get("options") or {}).get("account_fingerprint")))
    return output_reserve, working_reserve, str(route_facts.get("basis") or ""), bound, resolve_window(
        evidence, reserve, bound=bound, observation_fp=scope, load=(route.get("_resolved_target") or {}).get("requested_load_window"))


def _request_tail(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return messages[1:] if messages and messages[0].get("role") == "system" else messages


def _request_rows(system_content: List[Dict[str, Any]], user_content_json: str, supplementary_messages_json: str = "[]",
                  current_messages: Optional[List[Dict[str, Any]]] = None, *, room_data: str = "") -> List[Dict[str, Any]]:
    tail = (_request_tail(current_messages) if current_messages is not None else
            [{"role": "user", "content": json.loads(user_content_json)}, *json.loads(supplementary_messages_json)])
    return [{"role": "system", "content": system_content}, *with_program_history(tail, room_data)]


def _request_tokens(system_content: List[Dict[str, Any]], user_content_json: str, *,
                    supplementary_messages_json: str = "[]", current_messages: Optional[List[Dict[str, Any]]] = None,
                    chars: bool = False, tools: Optional[List[Dict[str, Any]]] = None, provider: str = "",
                    reasoning_effort: str = "", room_data: str = "") -> int:
    """The request estimator's one count of this view: its messages and, given ``tools``, their schemas."""
    return estimate_context_prompt_tokens(_request_rows(system_content, user_content_json, supplementary_messages_json,
                                                        current_messages, room_data=room_data), tools, provider=provider,
                                          reasoning_effort=reasoning_effort, chars=chars)


class MinimumViewUnfit(ValueError):
    """The core, assignment, resident tools and work room cannot fit this route."""
    physical = True  # False: the request itself fits the window, only the start work room beside it does not

    def __init__(self, message: str, *, fixed_tokens: int, memory_tokens: int,
                 reserve_tokens: int, window_tokens: int, ratio: float, work_room_tokens: int = 0,
                 required_book_tokens: int = 0, memory_measured: bool = True, parts: Iterable[Tuple[str, int]] = ()):
        self.memory_measured, self.parts = memory_measured, tuple(parts)  # False: no memory priced (never a measured 0)
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
        super().__init__(f"{message}: fixed input={self.fixed_input_tokens}, memory view={self.memory_view_tokens if memory_measured else 'not measured'}, "
                         f"reply/working reserve={reserve_tokens}, window={window_tokens}, "
                         f"deficit={self.deficit_tokens}, work room={work_room_tokens}, "
                         f"fixed/work-room deficit={self.work_room_deficit_tokens} tokens"
                         + (f", required handbook={self.required_book_tokens}" if required_book_tokens else ""))


def _wire_measure(core, governance, sections, form, user, current, *, provider, effort, sized, overhead, required, tool_frames,
                  snapshot):
    """Price a walk by deltas; a mode decision with a required handbook compares the complete rendered requests.

    A walk prices each selected program line with the room beside the one data envelope it reserves; a complete
    request draws that room as the system part and program data a send carries, once per fit (``ViewFit.model_parts``)."""
    from functools import cache
    from ouroboros import memory_floor as floor
    from ouroboros.memory_view import json_chars

    full = {m: {part.section_id: part.full_text for part in parts} for m, parts in sections.items()}
    shapes, escaped = {}, cache(json_chars)
    head = floor.program_history_head(snapshot)  # the envelope alone: its lines are already the room's

    def rows_for(mode, system, data):
        rows = _request_rows(system, user, core.supplementary_messages_json, current, room_data=data)
        return [*rows, *tool_frames(mode)[1]] if tool_frames else rows

    def measure(mode, sent, addressed, story=0, room=0):
        # Max and Low carry the same Body Map and connections: one shape unless a caller's frames differ by mode.
        key = (mode if tool_frames or mode == "nano" else "max", sent.chosen, bool(story), bool(room))
        if key not in shapes:  # one letter per nonempty text holds the wire separators and blocks
            rows = rows_for(mode, _system_blocks(core, governance[form[mode]], "s" if story else "",
                                                "r" if room else "", mode=mode, addressed={}), head)
            schemas = list(sent.schemas)
            whole = sized[2] if sized is not None and sized[:2] == (rows, schemas) else None
            if whole is None:
                counted("request_forms")
                whole = estimate_context_prompt_tokens(rows, schemas, provider=provider, reasoning_effort=effort, chars=True)
            shapes[key] = whole - bool(story) - bool(room)
        forms = sum(escaped(short) - escaped(full[mode][name]) for name, short in addressed.items())
        return max(1, (shapes[key] + forms + story + room + 3) // 4) + overhead

    def complete_price(mode, sent, forms, fit):
        if required is None:
            return measure(mode, sent, forms, fit.story_chars, fit.room_chars)
        whole_core, whole_governance = required
        room, data = fit.model_parts(whole_core.opening_program_address) if fit.level is not None else ("", "")
        system = _system_blocks(whole_core, whole_governance[form[mode]], fit.story, room, mode, forms)
        return overhead + estimate_context_prompt_tokens(rows_for(mode, system, data), list(sent.schemas),
                                                         provider=provider, reasoning_effort=effort)

    return measure, complete_price


def _max_start_room(snapshot, texts, governance, *, fixed, ratio, schemas, forms, window, reserve, work, allowances,
                    memory_price):
    """Price Max's ordinary start and, if it misses, the telling that decides whether Max holds."""
    from ouroboros import memory_floor as floor

    def need_at(level_of):
        view, price = memory_price(level_of(snapshot) if snapshot else None)
        return view, math.ceil(price * ratio) + reserve + work

    ordinary, need = need_at(floor.ordinary_level)
    fact = floor.max_room_fact(snapshot, texts, governance, story=floor.tokens_of(ordinary.story_chars), addressed=forms,
        fixed=fixed, ratio=ratio, schemas=schemas, window=window, reserve=reserve, work=work, allowance=allowances['margin'])
    if fact:
        fact.update(need=need, story=floor.tokens_of(ordinary.story_chars), room=floor.tokens_of(ordinary.room_chars))
        if need > window:
            told, telling_need = need_at(floor.telling_level)
            fact.update(telling_need=telling_need, telling=floor.memory_tokens(told))
    return fact


def _final_view(shown, snapshot, core, mode, room_fact, *, fixed, ratio, window, reserve, output_reserve, window_record,
                start_room, require_admission, work, optional_map, cold_return=False):
    """Draw the selected room as system text and program data; then add the required handbook.

    A start yields for it (``optional_map``) or names its refusal. ``require_admission`` is off on a local
    route: its sender counts the sealed request exactly, an estimate here cannot.
    """
    from ouroboros import memory_floor as floor

    story, facts = shown.story, shown.facts
    book = core.required_book_tokens()
    room, room_data, note, unfit = floor.fit_required_book(shown, book=book, fixed=fixed, ratio=ratio, reserve=reserve,
        output_reserve=output_reserve, window=window, window_record=window_record, mode=mode, work=work() if book else 0,
        lowered=bool(room_fact) and mode == "low", note_window=start_room, admit=require_admission,
        optional_map=optional_map, opening_address=core.opening_program_address, cold_return=cold_return)
    if unfit is not None:
        raise MinimumViewUnfit("This task's directly required handbook cannot fit beside its ordinary view",
            window_tokens=window, ratio=ratio, required_book_tokens=book, **unfit)
    if note:
        facts = {**facts, BOOK_NOTE_KEY: note}
    if book and room_fact:
        room_fact = {**room_fact, "book_required": True}
    receipt = floor.view_receipt(snapshot, story, room, floor.with_max_room(
        facts, room_fact, mode, required_book=math.ceil(book * ratio)), room_data=room_data) if snapshot else {}
    return story, room, room_data, receipt


def _frame_choices(tool_schemas, *, resident, nano_base, low_tools, nano_tools, tool_frames,
                   window_record, window_tokens, admitted, preferred, output_reserve, working_reserve_tokens):
    """Resolve per-frame tools and admissible window before pricing any memory; alternative tools stay lazy."""
    from functools import cache
    from ouroboros.tool_policy import select_tool_schemas

    names = [schema["function"]["name"] for schema in tool_schemas or []] if resident else nano_base

    @cache
    def tools_for(mode):
        if tool_frames:
            return select_tool_schemas(tool_frames(mode)[0], context_mode="low")
        alternative = nano_tools if mode == "nano" else low_tools if mode == "low" else None
        if alternative is not None:
            return select_tool_schemas(alternative() if callable(alternative) else alternative, context_mode="low")
        return select_tool_schemas(tool_schemas or [], context_mode=mode, schema_names=names,
                                   standing=resident or nano_base is None)

    @cache
    def limits(mode):
        target, reserve = context_budget.context_mode_limits(mode, preferred, output_reserve,
                                                            working_reserve_tokens=working_reserve_tokens)
        record = replace(window_record, reserve=reserve) if window_record else None
        window = ((record.admitted_equivalent if admitted else record.total_equivalent)
                  if record and record.usable is not None else window_tokens)
        return target, reserve, window, record

    return tools_for, limits


def _wire_room(raw, complete, *, mode, preferred, reserve, window, record, admitted, output_reserve, ratio):
    """Compare wire frames after the reply is priced; an input-only limit does not pay for that reply."""
    reply = context_budget.reply_allowance_tokens(caller_max_tokens=output_reserve, nano=mode == "nano",
        owner_nano=preferred == "nano", input_tokens=math.ceil(raw * ratio), raw_input_tokens=raw,
        window_tokens=window, window_record=record)
    room = record.input_room(reply, under_way=admitted) if record else None
    return (window - max(reserve, reply) if room is None else room) - math.ceil(complete * ratio)


def _integrating(core: ContextCore, role: Any) -> bool:
    """A view of a focus that integrates (BIBLE P1): no delegated capture, no delegated child's or nanny's ``role``."""
    return not core.delegated_focus and role not in DELEGATED_FOCUS_ROLES


def _mode_host_receipts(selected: str, form_mode: str, rows: Iterable[Mapping[str, Any]],
                        low_maps: Iterable[Mapping[str, Any]], book_maps: Iterable[str]) -> list:
    if form_mode != "low":
        return [row for row in rows if not row.get("held_note_form") and row.get("section_id") not in book_maps]
    if selected == "max":  # Max carries the Body Map whole, a delegated focus's Max too
        return [row for row in rows if row.get("section_id") not in book_maps]
    seen = {row.get("section_id") for row in rows}  # below Max a Map already at its parts stays there
    return [*rows, *(row for row in low_maps if row.get("section_id") not in seen)]


def _view_projections(core: ContextCore, governance: Mapping[str, Tuple[Any, ...]], user_content_json: str, *, preferred: str,
                      tool_schemas: Optional[List[Dict[str, Any]]], window_tokens: int, known_window: bool, output_reserve: int,
                      ratio: float, start: Optional[str] = None, resident: bool = False,
                      current_messages: Optional[List[Dict[str, Any]]] = None,
                      kept: Optional[Mapping[str, Any]] = None, saving: Optional[Mapping[str, int]] = None,
                      select: Optional[str] = None, kept_sections: Optional[Mapping[str, Iterable[Mapping]]] = None,
                      working_reserve_tokens: Optional[int] = None, window_record: Optional[WindowRecord] = None, work_room_levels: int = 1,
                      envelope_overhead_tokens: int = 0, start_work: bool = False, slack: int = 0, retain: Any = None,
                      admitted: bool = False, provider: str = "", reasoning_effort: str = "",
                      sized: Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]] = None,
                      nano_base: Optional[Tuple[str, ...]] = None, start_saving: int = 0,
                      running_limit: Optional[int] = None, books_must_save: bool = False, books_rung: bool = False,
                      nano_tools: _Schemas = None, low_tools: _Schemas = None, tool_frames: _ToolFrames = None,
                      unsent_start: bool = False) -> Tuple[Dict[str, Tuple[List[Dict[str, Any]], Dict, str]], str]:
    """Select the frame once, choose memory in one walk. Unsent Map forms share the selected memory, not another budget.

    Required DEVELOPMENT joins the ordinary fit last; wire admission still charges it. ``retain`` overrides the
    host section capture. ``nano_base`` supplies a call's own standing tools; ``resident`` prices the tools given.
    ``admitted`` selects the running rather than start line of ``WindowRecord``. ``sized`` carries a start's priced
    ``(rows, schemas, characters)`` for the same provider/effort shape. ``unsent_start``: a start's Body Map policy.
    """
    from ouroboros import memory_floor as floor
    from ouroboros.held_notes import HeldNoteSection, can_replay_frame, frame_key
    from ouroboros.memory_view import FULL_VIEW, room_chars, snapshot_from_json, story_chars

    shown_mode = start or preferred
    mode = select or shown_mode
    running = (running_limit is not None and mode == preferred == "max" and not core.delegated_focus
               and kept is not None and not start_work)
    whole_core, whole_governance = core, governance
    if core.development_required:
        core = replace(core, development_required=False)
        governance = {name: _governance_blocks(None, core, mode=name) for name in governance}
    form = {m: "low" if core.delegated_focus or m == "nano" else m for m in floor.MODES}  # held notes and receipts
    shape = {m: "low" if m == "nano" else m for m in floor.MODES}  # governance: the Body Map steps only below Max
    snapshot = snapshot_from_json(core.memory_view_json) if core.memory_view_json else None
    levels = snapshot.spec.work_room_levels if snapshot else work_room_levels
    held_heads = any(isinstance(part, HeldNoteSection) for part in core.dynamic_sections)
    cold_resume, saved = core.held_note_resume is not None, core.held_note_resume or {}
    saved_rows = {row["section_id"]: row for row in saved.get("receipts", ())}
    tools_for, limits = _frame_choices(tool_schemas, resident=resident, nano_base=nano_base, low_tools=low_tools,
        nano_tools=nano_tools, tool_frames=tool_frames, window_record=window_record, window_tokens=window_tokens,
        admitted=admitted, preferred=preferred, output_reserve=output_reserve, working_reserve_tokens=working_reserve_tokens)
    sections = {m: (*core.semi_stable_sections, *core.dynamic_sections, *core.dynamic_head_sections,
                    *(governance[shape[m]][0] if not isinstance(governance[shape[m]][0], str) else ())) for m in form}
    can_yield_maps = ((kept is None and not cold_resume) or start_work or unsent_start) and _integrating(core, snapshot.spec.role if snapshot else None)
    book_maps = floor.replay_book_maps(sections["low"], saved if kept is None else {}, core, retain=retain)
    low_maps = book_maps if kept is None else [r for r in (kept_sections or {}).get("low", ()) if r.get("section_id") in floor.BOOK_MAPS]
    saved_frame = saved.get("frame") if len(book_maps) == len(saved.get("book_maps", ())) else None  # a list now detailed: refit
    previous = tuple((kept_sections or {}).get(shown_mode, ()))
    receipts, room_fact = list(previous), None  # room_fact: the start's Max room, named by a view lowered from Max

    held = (kept or {}).get(shown_mode) or FULL_VIEW
    whole_memory = (story_chars(snapshot), room_chars(snapshot)) if snapshot and held_heads else (0, 0)
    story_before, room_before = (story_chars(snapshot, held), room_chars(snapshot, held)) if snapshot else (0, 0)

    measure, complete_price = _wire_measure(core, governance, sections, shape, user_content_json, current_messages,
        provider=provider, effort=reasoning_effort, sized=sized, overhead=envelope_overhead_tokens, tool_frames=tool_frames,
        required=(whole_core, whole_governance) if whole_core.development_required else None, snapshot=snapshot)

    def request_cost(selected, sent):
        return measure(selected, sent, {}, *whole_memory) - envelope_overhead_tokens

    def restores_frame(selected, sent, reserve):
        if kept is not None or not (held_heads and saved_frame and selected == saved_frame[0]):
            return False
        current = frame_key(selected, preferred, window_tokens if known_window else 0, window_record,
                            reserve, ratio, request_cost(selected, sent))
        return can_replay_frame(core, saved_frame, current, saved_rows)

    if not (select or start) and saved_frame and saved_frame[0] in floor.MODES:
        restored_mode = saved_frame[0]
        _, reserve, _, _ = limits(restored_mode)
        if restores_frame(restored_mode, tools_for(restored_mode), reserve):
            mode = restored_mode
    asked = (saving or {}).get(shown_mode, 0)  # the shown request's saving, in estimator tokens
    original_fixed = measure(shown_mode, tools_for(shown_mode), section_views(sections[shown_mode], previous))
    _, original_reserve, original_window, _ = limits(shown_mode)
    original_input = original_window - original_reserve

    def credit(window, reserve):
        # Compare admissible input, not the reply reserves: an input-only cap gives no credit.
        return math.floor((window - reserve - original_input) / max(ratio, 0.01)) if known_window else 0

    def owed(fixed, window, reserve):
        return max(0, asked - (original_fixed - fixed) - credit(window, reserve))

    def running_goal(window, reserve):
        return running_limit + credit(window, reserve) if running else None

    def view(selected, sent, addressed, window, boundary, fixed=None, **walk):  # one frame's memory beside host forms
        fixed = measure(selected, sent, addressed) if fixed is None else fixed
        return floor.render_view_for_mode(snapshot, mode=selected, owner_mode=preferred, window_tokens=window, **walk,
            known_window=known_window, output_reserve=output_reserve, ratio=ratio, non_memory_tokens=fixed,
            # The Max room as the receipt settles it (``with_max_room`` decides its cause), so the floor note of a
            # start memory lowered names what holds the room.
            lowered_from=preferred if selected != preferred else None, start_work=start_work,
            max_room=floor.with_max_room({}, room_fact, selected).get("max_room") or first_contact,
            tool_names=None if tool_schemas is None else sent.chosen, working_reserve_tokens=working_reserve_tokens,
            stated_window_tokens=stated.get(selected),  # the note names the window as stated, never the admitted frame
            boundary_tokens=boundary, measure_text=lambda *memory: measure(selected, sent, addressed, *memory) - fixed,
            report_nano_miss=held_heads)

    stated: Dict[str, Optional[int]] = {}  # each walked mode's window in the start's frame (``frame`` fills it)

    map_parts = partial(floor.lowered_maps, sections["low"], core, retain=retain)  # the Map at its parts and addresses
    # A start walked in Low with a refusal's rung is a refused first send's frame (``_prefer_first_contact``): its note says so.
    first_contact = floor.FIRST_CONTACT_ROOM if books_rung and kept is None and mode != preferred else None
    def books_save(level, window, boundary):
        # Only the frame changes: the Map at its parts. A new memory/host concession may not disguise a switch that costs tokens.
        def price(selected, maps):
            sent = tools_for(selected)
            forms = section_views(sections[selected], _mode_host_receipts(selected, form[selected], previous, maps, floor.BOOK_MAPS))
            same = view(selected, sent, forms, window, boundary, kept=held, level=level) if snapshot else floor.NO_VIEW
            return complete_price(selected, sent, forms, same)
        return price("low", map_parts(low_maps, retain=predicted_section)) < price("max", low_maps)

    if books_must_save and not books_save(held, window_tokens, window_record.boundary if window_record else None):
        return {}, shown_mode
    low_maps = map_parts(low_maps) if books_rung else low_maps  # all that rung has to give; first contact prepares it too

    def telling_fits(selected, sent, window, boundary, reserve, forms, start_room=kept is None, budget=None):
        shown = view(selected, sent, forms, window, boundary, level=floor.telling_level(snapshot)) if snapshot else floor.NO_VIEW
        work = math.ceil(min(boundary or window, window) / context_budget.RECLAIM_LOW_WATER_DIVISOR) * levels if start_room else 0
        price = measure(selected, sent, forms, shown.story_chars, shown.room_chars)
        return math.ceil(price * ratio) + reserve + work <= window and (budget is None or price <= budget)

    def frame(selected):
        nonlocal receipts
        prior = _mode_host_receipts(selected, form[selected], previous, low_maps, floor.BOOK_MAPS)
        available = sections[selected]
        sent = tools_for(selected)
        target, reserve, window, record = limits(selected)
        replay = restores_frame(selected, sent, reserve)
        if replay:
            prior = _mode_host_receipts(selected, form[selected], saved_rows.values(), low_maps, floor.BOOK_MAPS)
            if "knowledge_index" not in saved_rows:
                available = tuple(part for part in available if part.section_id != "knowledge_index")
        stated[selected] = record.total_equivalent if record and record.usable is not None else None
        boundary = record.boundary if record else None
        host_allowances = floor.floor_allowances(window_tokens=window if known_window else None,
            output_reserve_tokens=reserve, non_memory_tokens=0, target_tokens=target if kept is None else None,
            calibration_ratio=ratio, work_room_levels=levels, boundary_tokens=boundary)
        least_level = [None]  # reprice this frame's shortest level after each host step, walk again on a miss
        sent_view = {} if kept is None else {"kept": held}  # a view already sent walks on from its level
        def shortest(forms, physical):  # what a start sends once its own walk took each form before uncovered owner rows
            shown = least_level[0] and view(selected, sent, forms, window, boundary, level=least_level[0], **sent_view)
            price = shown and measure(selected, sent, forms, shown.story_chars, shown.room_chars)
            if price is None or price > physical:  # no walk yet, or its level no longer fits beside these host forms
                shown = view(selected, sent, forms, window, boundary, shortest=True, **sent_view)
                least_level[0] = shown.level
                price = measure(selected, sent, forms, shown.story_chars, shown.room_chars)
            return price
        told = partial(telling_fits, selected, sent, window, boundary, reserve)  # ``max_room_lost``'s test: my shortest telling
        goal = running_goal(window, reserve)
        host_saving = owed(measure(selected, sent, section_views(sections[selected], prior)), window, reserve) if asked else 0
        receipts, _ = floor._host_steps(available, prior,  # every frame starts from what was shown
            optional_fits=lambda forms, level=None: floor.optional_excerpt_fits(
                goal if running else host_allowances['window' if kept is None or start_work else 'physical'],
                0 if running else owed(measure(selected, sent, forms), window, reserve) - slack, (story_before, room_before),
                lambda *memory: measure(selected, sent, forms, *memory),
                lambda: view(selected, sent, forms, window, boundary, telling_only=not running, telling_floor=running,
                             request_limit=goal, level=level, **sent_view) if snapshot else floor.NO_VIEW),
            measure=lambda addressed: measure(selected, sent, addressed, story_before, room_before),
            shortest=shortest if snapshot and kept is None else None,  # a held view was sent: it keeps its steps
            told=told if snapshot and known_window else None,
            # Held heads yield last, to the window alone (voice setup is a start even with a prepared view).
            held_least=shortest if held_heads and snapshot and known_window and form[selected] == "low" and not replay else None,
            held_bound=host_allowances["window" if kept is None or start_work else "physical"],
            window=window if known_window else None, reserve=reserve, ratio=ratio, saving=host_saving,
            slack=max(slack, slack + host_saving - asked) if running else slack,  # a trial's growth buys no host outline
            root=core.section_root, task_id=core.section_task_id, bare=lambda forms: measure(selected, sent, forms),
            allowance=host_allowances["margin" if kept is None or start_work else "physical"],
            budget_mode=selected if kept is None and target and target < window else None,
            window_bound=host_allowances["window"], retain=retain,  # the window's own start bound, whatever target frames the start
            optional_walk=(lambda forms: view(selected, sent, forms, window, boundary)) if snapshot else None,
            # The Map yields to the window, and before Nano's own target takes my memory below a telling (BIBLE P1).
            maps_told=(lambda forms: told(forms, (kept is None or start_work or unsent_start) and selected != preferred,
                                          host_allowances["budget"] if selected == "nano" else None)) if known_window else None)
        fixed = measure(selected, sent, section_views(sections[selected], receipts))
        allowances = floor.floor_allowances(window_tokens=window if known_window else None,
            output_reserve_tokens=reserve, non_memory_tokens=fixed, target_tokens=target if kept is None else None,
            calibration_ratio=ratio, work_room_levels=levels, boundary_tokens=boundary)
        return sent, fixed, reserve, window, boundary, allowances

    sent, fixed, reserve, window, boundary, allowances = timed(f"fit.frame_{mode}", frame, mode)
    def work():
        return math.ceil(min(boundary or window, limits(mode)[0] or window) / context_budget.RECLAIM_LOW_WATER_DIVISOR) * levels

    if (kept is None or start_work) and mode == "max":
        forms = section_views(sections[mode], receipts)
        def memory_price(level):
            shown = view(mode, sent, forms, window, boundary, level=level) if snapshot else floor.NO_VIEW
            return shown, measure(mode, sent, forms, shown.story_chars, shown.room_chars)
        room_fact = _max_start_room(snapshot, rendered_texts(core, forms), render_governance(governance[shape[mode]], forms),
            fixed=fixed, ratio=ratio, forms=forms, window=window, reserve=reserve, work=work(), allowances=allowances,
            schemas=tool_schema_tokens(list(sent.schemas), provider=provider, reasoning_effort=reasoning_effort),
            memory_price=memory_price)
    running_max_lost = (not running and mode == "max" and form[mode] == "max" and known_window and snapshot
                        and kept is not None and not start_work and held_heads and not telling_fits(
                            mode, sent, window, boundary, reserve, section_views(sections[mode], receipts), False))
    if (room_fact and floor.max_room_lost(room_fact)) or running_max_lost:
        mode, low_maps = "low", map_parts(low_maps)  # Low for the window: the Map steps down before memory (BIBLE P1)
        sent, fixed, reserve, window, boundary, allowances = timed(f"fit.frame_{mode}", frame, mode)

    running_shown = None
    def render(forms=None, *, level=None, telling_floor=False):
        if not snapshot:
            return floor.NO_VIEW
        forms = section_views(sections[mode], receipts) if forms is None else forms
        price = measure(mode, sent, forms)
        return view(mode, sent, forms, window, boundary, saving=owed(price, window, reserve), fixed=price,
                    kept=None if kept is None else held, level=level,
                    start_saving=start_saving, telling_floor=telling_floor, request_limit=running_goal(window, reserve))

    def spare(fit, forms):
        return window - reserve - math.ceil(measure(mode, sent, forms, fit.story_chars, fit.room_chars) * ratio)

    def wire_room(fit):
        forms = section_views(sections[mode], receipts)
        return _wire_room(measure(mode, sent, forms, fit.story_chars, fit.room_chars), complete_price(mode, sent, forms, fit),
            mode=mode, preferred=preferred, reserve=reserve, window=window, record=limits(mode)[3], admitted=admitted,
            output_reserve=output_reserve, ratio=ratio)

    def overflows(fit=floor.NO_VIEW):  # the known window cannot hold this frame's request beside this memory view
        if running or held_heads:
            return known_window and spare(fit, section_views(sections[mode], receipts)) < 0
        memory = floor.tokens_of(fit.story_chars) + floor.tokens_of(fit.room_chars)
        return known_window and math.ceil((fixed + memory) * ratio) + reserve > window

    def misses_running_goal(fit):
        forms = section_views(sections[mode], receipts)
        return overflows(fit) or measure(mode, sent, forms, fit.story_chars, fit.room_chars) > running_goal(window, reserve)

    if running:
        running_shown = render(telling_floor=True)
        if misses_running_goal(running_shown):
            if books_save(running_shown.level, window, boundary):
                mode, low_maps = "low", map_parts(low_maps)
                sent, fixed, reserve, window, boundary, allowances = timed("fit.frame_low", frame, mode)
                # Walk Low from the held level, not from the discarded Max trial's tellings.
                running_shown = render(telling_floor=True)
            if misses_running_goal(running_shown):
                # An unsent trial cannot pin cuts: the unrestricted walk may give some back after a later saving.
                running_shown = render()

    # Step down on the frame's own price, without drawing it whole. Nano is the last physical rescue, even mid-run.
    last = stays = None
    while True:
        # A held view stays where Nano can buy no room. Without heads this can be decided before drawing Nano (a cold Nano too).
        stays = (last is not None and not (held_heads or whole_core.development_required)
                 and window - reserve <= last[3] - last[2] and fixed >= last[1])
        shown = floor.NO_VIEW if stays or (mode != "nano" or kept is None and not cold_resume and current_messages is None and not held_heads) and overflows() else running_shown or render()
        if mode == "nano" or not overflows(shown):
            break
        # A live trial still has to reselect tools and measure its chosen wire reserve.
        if (running or mode == "low") and kept is not None and admitted:
            prior_view = (running_shown or render()) if shown is floor.NO_VIEW else shown
            last = (sent, fixed, reserve, window, boundary, allowances, receipts, prior_view, wire_room(prior_view), mode)
        mode, running_shown = ("low" if mode == "max" and not running else "nano"), None
        sent, fixed, reserve, window, boundary, allowances = timed(f"fit.frame_{mode}", frame, mode)
    if held_heads and mode == "nano" and overflows(shown):
        emergency_level = shown.level
        if floor._held_note_steps(sections[mode], receipts, form="reader",
                                  fits=lambda forms: spare(render(forms, level=emergency_level), forms) >= 0):
            fixed = measure(mode, sent, section_views(sections[mode], receipts))
            shown = render()
            if overflows(shown):
                shown = render(level=emergency_level)
    compare_wire = held_heads or running or whole_core.development_required or nano_tools is not None or tool_frames is not None
    if last is not None and (stays or compare_wire and wire_room(shown) <= last[-2]):
        sent, fixed, reserve, window, boundary, allowances, receipts, shown, _, mode = last
    if kept is None and not cold_resume and current_messages is None and overflows(shown):  # a cold restore still reconciles its saved tools
        core_alone = overflows()  # then a Nano frame without held heads rendered no memory (the loop's cold skip)
        from ouroboros.context_view_facts import refusal_parts  # names what the refused request holds
        raise MinimumViewUnfit("Core, assignment, resident control tools and work room cannot fit" if core_alone
                               else "Protected owner words and the minimum view cannot fit the usable input",
                               fixed_tokens=fixed, memory_tokens=floor.tokens_of(shown.story_chars) + floor.tokens_of(shown.room_chars),
                               reserve_tokens=reserve, window_tokens=window, ratio=ratio, memory_measured=held_heads or not core_alone,
                               work_room_tokens=work() if core_alone else 0, parts=refusal_parts(core, sections[mode], receipts, sent.schemas, shown, user_content_json, mode=mode, provider=provider, reasoning_effort=reasoning_effort))
    def optional_map():  # the required handbook's next yield: optional maps, then an unread Low start's Body Map
        nonlocal receipts
        receipts = floor.yield_for_book(sections[mode], receipts, core, retain=retain, maps=can_yield_maps and mode == "low") or receipts
        return measure(mode, sent, section_views(sections[mode], receipts))
    story, room, room_data, receipt = _final_view(shown, snapshot, whole_core, mode, room_fact, fixed=fixed, ratio=ratio,
        window=window, reserve=reserve, output_reserve=output_reserve, work=work, window_record=limits(mode)[3], optional_map=optional_map,
        start_room=known_window and (kept is None or start_work), cold_return=cold_resume,
        require_admission=kept is None and known_window and current_messages is None and provider != "local")
    counted("view_texts_kept", 2 if snapshot else 0)  # the story and the room text this fit delivers
    whole = shown.level is None or floor.mv.story_is_whole(shown.level)  # the delivered level's own fact: no render
    contents = {}
    for m in floor.MODES:
        rows = _mode_host_receipts(m, form[m], receipts, low_maps, floor.BOOK_MAPS)
        facts = {**receipt, "held_note_request_tokens": request_cost(mode, sent)} if held_heads and m == mode else receipt
        system, observation = capture_system_projection(whole_core, whole_governance[shape[m]], story, room, m,
            section_views(sections[m], rows), rows, whole, shown=shown,
            facts={**facts, "host_sections_addressed": rows} if rows else facts)
        contents[m] = (system, observation, room_data)
    return contents, mode


def _whole_request(core, governance, mode, current_messages=None):
    """Uncut selected request for a route's comparable density witness, before any floor step: the request with
    one-letter memory texts and the characters the whole story and room add beyond those letters."""
    from ouroboros.memory_view import room_chars, snapshot_from_json, story_chars
    from ouroboros.memory_floor import program_history_head

    snapshot = snapshot_from_json(core.memory_view_json) if core.memory_view_json else None
    story, room = (story_chars(snapshot), room_chars(snapshot)) if snapshot else (0, 0)
    form = "low" if mode == "nano" else mode  # the governance a frame carries: the Body Map steps only below Max
    blocks = governance[form] if governance is not None else _governance_blocks(None, core, mode=form)
    messages = _request_rows(_system_blocks(core, blocks, "s" if story else "", "r" if room else "", mode),
                             core.user_content_json, core.supplementary_messages_json, current_messages,
                             room_data=program_history_head(snapshot))
    return messages, story + room - bool(story) - bool(room)


def _held_request_cost(core, mode, schemas, current_messages, *, provider, reasoning_effort):
    """Price the uncut ordinary view even when a fitting rebind keeps the shown system row."""
    if core.development_required:
        core = replace(core, development_required=False)
    rows, memory = _whole_request(core, None, mode, current_messages)
    size = estimate_context_prompt_tokens(rows, schemas, provider=provider, reasoning_effort=reasoning_effort, chars=True)
    return (size + memory + 3) // 4


# ``memory_floor.start_cause``; ``unmeasured``: a Max start whose route named no window, never a held Max (F1 #10)
START_CAUSES = ("", "memory", "request", "route", "window", "unmeasured", *FIRST_CONTACT_START.values())
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
    return replace(plan, start=carried_start(carried)) if isinstance(plan, ContextFitPlan) else plan


def _prefer_first_contact(core, preferred, normal, forms, target, tool_schemas, limits):
    """Choose among legal first-send frames, never put a soft hint into WindowRecord or a floor allowance.

    The start keeps its own mode (Max, or the Low a known window or the owner chose): ordinary memory folding
    pays what the ordinary view exceeds below the refused size of that wire (``first_contact_limits``); a directly
    required handbook and its note are no memory's debt, though every physical comparison counts them. Where my
    human's Max at its folded tellings still stands above a refused Max size that a Body-Map-at-its-parts send
    followed, the start is Low, temporarily, for the window: the same tellings, the Map at its parts and addresses,
    every tool kept (BIBLE P1: the Map steps down before my memory goes below a telling). That frame is walked as
    Low, so its facts, its floor note and a later selection in it are Low's and name the refusal, never a window that
    did not fit. It is taken only below every size this wire refused. A fold must also save the ordinary view's
    reclaim margin. The note names the planned connection, even if a later route keeps the shown view. Unchanged
    returns ``normal``."""
    from ouroboros.context_measurement import BOOK_NOTE_KEY
    from ouroboros.context_window import text_chat_rows
    from ouroboros.llm_messages import STABLE_PREFIX_BLOCKS_KEY
    from ouroboros.llm_openai_compatible import _OpenAICompatibleLaneMixin, _project_openai_family_system
    from ouroboros.memory_floor import TEMPORARY_LOW
    from ouroboros.tool_policy import select_tool_schemas

    def raw(contents, mode):
        system, _facts, data = contents[mode]
        rows = _request_rows(system, core.user_content_json, core.supplementary_messages_json, room_data=data)
        if not text_chat_rows(rows):
            return 0
        rows[0][STABLE_PREFIX_BLOCKS_KEY] = max(1, len(system) - 1)
        schemas = list(select_tool_schemas(tool_schemas or [], context_mode=mode).schemas)
        return bounded_prompt_tokens_for_payload({"messages": _project_openai_family_system(dict(target), rows),
                                                  "tools": schemas}, 0) if schemas else 0

    def noted(contents, mode, maps=False):
        system, facts, data = contents[mode]
        tail = {**system[-1], "text": system[-1]["text"] + (
            "\n\n### First-send composition\nAn earlier request was refused on the connection this first send was "
            "composed for, so this view is smaller than its ordinary form" + (
                f": {TEMPORARY_LOW}, the body map at its parts and addresses" if maps else "")
            + ". The refusal stated no window.")}
        facts = {**facts, "first_contact_preference": "maps" if maps else True}
        if room := normal[0][mode][1].get("max_room"):
            facts["max_room"] = {**room, "cause": "first_contact", "cuts": []} if maps else room
        return {**contents, mode: ([*system[:-1], tail], facts, data)}

    def safe(size):
        return size - max(1, reclaim_low_water_margin(None, size))

    def ordinary(size, contents, mode):
        return size - core.required_book_tokens() - int(contents[mode][1].get(BOOK_NOTE_KEY) or 0)

    contents, initial = normal
    max_wire, before_maps, any_form = limits
    refused = max_wire if initial == "max" else any_form
    if (initial not in {"max", "low"} or not refused
            or target["provider"] == "openrouter" and _OpenAICompatibleLaneMixin._openrouter_main_web_search_tool()):
        return normal
    goal = safe(min(refused, before_maps or refused)) if initial == "max" else safe(refused)
    if (whole := raw(contents, initial)) <= goal:
        return normal
    ordinary_whole = ordinary(whole, contents, initial)
    reclaim = safe(ordinary_whole)  # the ordinary request pays its own margin, not the required book's
    selected, size, folds = contents, whole, False
    lowers = initial == preferred == "max" and bool(before_maps) and not core.delegated_focus  # the Map's rung is proven
    if (saving := ordinary_whole - goal) > 0:
        try:
            folded, mode = forms(start_saving=saving)
        except MinimumViewUnfit:  # the optional fold and its own receipt must not refuse an otherwise legal start
            folded, mode = normal
        if mode == initial and folded[mode][1].get("floor", {}).get("runs") != contents[mode][1].get("floor", {}).get("runs"):
            candidate = noted(folded, mode)
            if 0 < (smaller := raw(candidate, mode)) and ordinary(smaller, candidate, mode) <= reclaim:
                selected, size, folds = candidate, smaller, True
    if lowers and ordinary(size, selected, initial) > safe(before_maps):
        try:  # the Low frame's own walk: the same fold, the Map at its parts
            lowered = noted(forms(select="low", books_rung=True, **({"start_saving": saving} if folds else {}))[0], "low", maps=True)
        except MinimumViewUnfit:
            lowered = None
        if lowered and 0 < raw(lowered, "low") < min(size, any_form):
            return lowered, "low"
    return (selected, initial) if folds else normal


@timed_section("fit")
def build_context_fit_plan(
    env: Any,
    core: ContextCore,
    task: Dict[str, Any],
    *,
    preferred_mode: str,
    route_resolver: Callable[..., Tuple[Dict[str, Any], Any]],
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    reasoning_effort: str = "",
    nano_base: Optional[Tuple[str, ...]] = None,
) -> ContextFitPlan:
    """Capture route facts, choose the frame and fit one start view, measured in the shape the route's provider and the
    first send's ``reasoning_effort`` give the request (``context_measurement._request_shapes``); ``nano_base``: a live call's own Nano
    tools (``_view_projections``)."""
    preferred = str(preferred_mode or "max").strip().lower()
    if preferred not in {"low", "max", "nano"}:
        preferred = "max"

    meta = task.get("task_metadata") if isinstance(task.get("task_metadata"), dict) else {}
    user_content = json.loads(core.user_content_json)
    governance = {form: _governance_blocks(env, core, mode=form) for form in ("max", "low")}
    from ouroboros.capability_evidence import evidence_model_route, window_evidence
    from ouroboros.memory_view import start_room, view_role
    from ouroboros.tool_policy import select_tool_schemas

    view_task = {**task, "metadata": task.get("metadata") or meta}
    role = view_role(view_task)
    view_kind = DENSITY_VIEW_KINDS.get(role, role)
    # Select density before the single start walk, comparing the whole captured request at the size that can be sent.
    restored = core.held_note_resume or {}
    current_messages = restored.get("messages")
    fresh_chat = (bool(task.get("_is_direct_chat")) and view_kind == "owner_turn" and preferred in {"max", "low"}
                  and not restored and not getattr(route_resolver, "route_fields", None) and nano_base is None)
    if (saved_schemas := restored.get("tool_schemas")) is not None:
        tool_schemas = saved_schemas
        nano_base = tuple(schema["function"]["name"] for schema in tool_schemas)
    rows, memory_chars = _whole_request(core, governance, preferred, current_messages)
    schemas = list(select_tool_schemas(tool_schemas or [], context_mode=preferred, schema_names=nano_base,
                                       standing=nano_base is None).schemas)
    counted("request_forms")
    levels, sized = start_room(view_task, role), {}

    from ouroboros.llm_ollama import OllamaUpgradeRequired

    def answer(ask: Callable[[], Tuple[Dict[str, Any], Any]]) -> Tuple[Dict[str, Any], Any]:
        try:  # a child probes its own route too: its parent's warm evidence is no fact about it
            return ask()
        except OllamaUpgradeRequired:
            raise
        except Exception:
            log.debug("Context-fit route evidence unavailable; preserving Max", exc_info=True)
            # A cold start asks about its next send; the original task may name another route.
            return _failed_route_evidence({**task, **getattr(route_resolver, "route_fields", {})})

    def frame(route: Dict[str, Any], evidence: Any) -> SimpleNamespace:
        """What the start walk reads of one route answer: reserves, the window record, the density and the request's
        shape (``context_measurement._request_shapes``: its provider matters only where it opens the custom ladder)."""
        output_reserve, working_reserve, window_basis, bound, window_record = route_window(route, evidence, preferred)
        provider = str(route.get("provider") or "")
        if provider not in sized:  # the whole request measured once per provider, however many answers are framed
            sized[provider] = estimate_context_prompt_tokens(rows, schemas, provider=provider,
                                                             reasoning_effort=reasoning_effort, chars=True)
        sendable = min(max(1, (sized[provider] + memory_chars + 3) // 4), window_record.usable or math.inf)
        ratio = 1.0 if window_basis == "estimator" else _route_calibration_ratio(
            None, str(evidence.route_fp or ""), str(route["model"] or ""), sendable, view_kind=view_kind)
        limits = tuple((limit.tokens, limit.kind) if limit else None for limit in (window_record.input, window_record.total))
        return SimpleNamespace(route=route, evidence=evidence, output_reserve=output_reserve, working_reserve=working_reserve,
                               window_basis=window_basis, bound=bound, window_record=window_record, ratio=ratio,
                               known_window=window_record.usable is not None,
                               walked=(limits, window_record.usable is not None, output_reserve, working_reserve, ratio,
                                       _custom_ladder(provider, reasoning_effort)))

    def project(state, user_content_json, **options):
        provider = str(state.route.get("provider") or "")
        try:
            return _view_projections(core, governance, user_content_json,
                preferred=preferred, tool_schemas=tool_schemas, window_tokens=state.window_record.total_equivalent,
                known_window=state.known_window, output_reserve=state.output_reserve, window_record=state.window_record,
                working_reserve_tokens=state.working_reserve, provider=provider, reasoning_effort=reasoning_effort,
                sized=(rows, schemas, sized[provider]), nano_base=nano_base, current_messages=current_messages,
                ratio=state.ratio, work_room_levels=levels, **options)
        except MinimumViewUnfit as unfit:  # the refusal names its serving program and window in effect
            from ouroboros.context_view_facts import refusal_route
            raise refusal_route(unfit, state.route, replace(state.window_record, reserve=unfit.response_reserve_tokens))

    def walk(state):
        try:
            return (*project(state, core.user_content_json), {})
        except MinimumViewUnfit as unfit:
            # Physical rescue only (owner 2026-10-13, #1527): an exact source replaces the whole input only when no view
            # of the known window holds it, a wake's bound projection first (``context_input_selection.rescued_start``).
            from ouroboros.context_input_selection import rescued_start
            return rescued_start(env, task, core.user_content_json, unfit, preferred, lambda text: project(state, text))

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
    guard = early and route_resolver.early and (refused or narrowed_by_window(contents, initial_mode, preferred, state, delivered))
    if guard or not state.known_window:
        # Horizon guard (W2 H4): an answer asked beside the build that would show less than the selected view, or none,
        # is asked again here, at the usual point, and that answer is used (its refusal is final); a walk that reads
        # the same frame is not repeated unless it was refused. A start whose route named no window asks it once more
        # before its first send (F1 #10): that answer is used, a window then known walked like any answer, one that read
        # the route without a window (an account, a reply limit) too; only a failed one (``status`` failed, as a raised
        # ask is) keeps the first; no bound is guessed. A first answer whose read failed, live or as the probe's
        # ten-minute failure record, forces a new probe. Without a failed read, the resolver's cache policy still applies.
        failed = not state.known_window and "failed" in (getattr(state.evidence, "status", ""),
                                                         getattr(state.evidence, "read", ""))
        with another_pass():  # a repeat by design, not work done again (``context_health`` names only the latter)
            late = frame(*answer(lambda: route_resolver.again(task, failed=failed) if early else timed(
                "fit.route", route_resolver, {**task, "route_read_again": True} if failed else task, allow_fetch=True)))
            if guard or late.known_window or getattr(late.evidence, "status", "") != "failed":
                if refused is not None or late.walked != state.walked:
                    contents, initial_mode, delivered = walk(late)
                state = late
    route, evidence, window_record, known_window = state.route, state.evidence, state.window_record, state.known_window
    # Only the ordinary walk can trigger the guard; the soft hint reads the final binding, even when no rewalk was needed.
    normal, first_contact = None, ""
    if fresh_chat and not delivered and not _custom_ladder(str(route.get("provider") or ""), reasoning_effort):
        from ouroboros.context_input_selection import opening_owner_turn
        from ouroboros.context_window import first_contact_limits, first_contact_route_scope

        first_contact, cap = first_contact_route_scope(route)
        target = route.get("_resolved_target")
        if (first_contact and schemas and core.memory_view_json and cap == state.output_reserve
                and (limits := first_contact_limits(target, cap, scope=first_contact))[2] and opening_owner_turn(env, core, task)):
            ordinary = contents, initial_mode
            chosen = _prefer_first_contact(core, preferred, ordinary, lambda **options: project(state, core.user_content_json, **options),
                                           target, tool_schemas, limits)
            if chosen is not ordinary:
                normal = ordinary
                contents, initial_mode = chosen
    output_reserve, working_reserve, window_basis, ratio = (state.output_reserve, state.working_reserve,
                                                            state.window_basis, state.ratio)
    bound = state.bound
    estimates = {}  # projections that read alike are one form of the request
    tail_json = (json.dumps(_request_tail(current_messages), ensure_ascii=False, sort_keys=True)
                 if current_messages is not None else None)

    @timed_section("fit.projections")
    def _projection(mode: str, source=None) -> ContextFitProjection:
        system_content, memory_facts, data = (contents if source is None else source)[mode]
        system_content_json = json.dumps(system_content, ensure_ascii=False, sort_keys=True)
        user_projection = delivered.get(mode)
        form = (system_content_json, user_projection, data)
        if form not in estimates:
            counted("request_forms")
            estimates[form] = _request_tokens(system_content, user_projection or core.user_content_json,
                                              supplementary_messages_json=core.supplementary_messages_json,
                                              current_messages=current_messages, room_data=data)
        estimated = estimates[form]
        return ContextFitProjection(
            mode=mode,
            system_content_json=system_content_json,
            estimated_tokens=estimated,
            calibrated_tokens=int(estimated * ratio),
            calibration_ratio=ratio,
            user_content_json=user_projection,
            **projection_fields(system_content, memory_facts),
            continuation_messages_json=tail_json, room_data=data,
        )

    max_projection, low_projection, nano_projection = (_projection(mode) for mode in ("max", "low", "nano"))
    from ouroboros.memory_floor import start_cause

    def started(source, mode: str) -> Tuple[str, str]:
        facts = source[mode][1]
        # A further advisory fold in a physically lowered view did not lower it; a refused first send's Low is its own cause.
        physical = start_cause(facts) if mode != preferred and facts.get("first_contact_preference") != "maps" else ""
        return mode, physical or FIRST_CONTACT_START.get(facts.get("first_contact_preference"), "") or (
            "unmeasured" if mode == "max" and not known_window else "")

    normal_start = (*(_projection(mode, normal[0]) for mode in ("max", "low", "nano")), normal[1], started(*normal)
                    ) if normal is not None else None

    core_payload = json.dumps(
        {
            "base_prompt": core.base_prompt,
            "bible_md": core.bible_md,
            "body_map": core.body_map,
            "body_map_parts": core.body_map_parts,
            **({"development_md": core.development_md} if core.development_required else {}),
            "semi_stable_text": core.semi_stable_text,
            "host_sections": [part.__getnewargs__()
                              for part in (*core.semi_stable_sections, *core.dynamic_head_sections, *core.dynamic_sections)],
            "dynamic_head_text": core.dynamic_head_text,
            "dynamic_text": core.dynamic_text,
            "memory_view_json": core.memory_view_json,
            **({"connections": list(core.connections)} if any(core.connections) else {}),
            **({"supplementary_messages": json.loads(core.supplementary_messages_json)}
               if json.loads(core.supplementary_messages_json) else {}),
            "user_content": user_content,
            **({"development_required": True} if core.development_required else {}),
            "delegated_focus": core.delegated_focus,
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
        model_route=evidence_model_route(evidence, route["model"]) if route["provider"] == "claudexor" else {},
        evidence_source=window_record.source, window_record=window_record, window_witness_bound=bound,
        route_base_url=str(route.get("base_url") or ""), account_pin=str(route.get("account_pin") or ""),
        window_evidence=window_evidence(evidence, window_record),
        requested_load_window=window_record.load,  # the load its record was resolved for, never another scope's
        core=core, start_tool_schemas_json=json.dumps(tool_schemas, ensure_ascii=False, sort_keys=True),
        start_nano_base=nano_base,
        start_reasoning_effort=reasoning_effort,
        start=started(contents, initial_mode), first_contact=first_contact,
        first_contact_normal=normal_start,
    )
