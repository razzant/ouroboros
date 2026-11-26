"""Host entry for initial realtime setup, using the ordinary context and plugin owners.

The caller is an admitted host turn, with its env/memory/task/ToolContext. This
module opens no stores of its own and provides no public HTTP or model-tool door.
"""
from __future__ import annotations

import asyncio

from ouroboros.contracts.realtime import (
    RealtimeInput, RealtimeSetupError, canonical_json, sha256,
)
from ouroboros.extension_registry_state import (
    _extensions, _lock, _realtime_providers, _unloading,
)


def rows_for_engine(rows: list[dict], *, vision: bool, fresh=False, owner_mode='max', schemas=()) -> list[dict]:
    """What the speech engine receives from the ordinary history.

    An engine whose catalog says it does not see images gets each image part as
    the caption that names the saved file, so a screenshot taken in a Grok call
    reaches the model as text instead of closing the call (owner's stand, 27.10).
    The ordinary history keeps the image for the text model.
    """
    from ouroboros.llm_messages import project_wire_messages
    if fresh:
        from ouroboros.context_images import fresh_call_images
        rows = fresh_call_images(rows, owner_mode=owner_mode)
    from ouroboros.context_images import render_image_placeholders
    reader = any(schema.get('function', {}).get('name') == 'view_image' for schema in schemas)
    rows = render_image_placeholders(rows, reader_available=reader)
    if vision:
        return project_wire_messages(rows, delivery_marks=True)
    out = []
    for row in rows:
        content = row.get("content")
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
            texts = {str(p.get("text") or "").strip() for p in content if isinstance(p, dict) and p.get("type") == "text"}
            row = {**row, "content": [_image_caption(p, texts) if isinstance(p, dict) and p.get("type") == "image_url"
                                      else p for p in content]}
        out.append(row)
    return project_wire_messages(out, delivery_marks=True)


def _image_caption(part: dict, texts: set) -> dict:
    caption = str(part.get("_caption") or "").strip()
    where = ("the file named above" if caption and caption in texts  # the attach caption already names it
             else caption or f"the file saved at {part.get('_source_path') or 'the attachment'}")
    return {"type": "text", "text": (f"[Image not shown: this voice engine does not see images; vlm_query with "
                                     f"file_path describes {where} in text.]")}


def input_from_context_plan(plan, *, model: str, selected_tools: list[dict],
                            output_allowance_tokens: int, messages: list[dict] | None = None,
                            tool_reader: bool = True) -> RealtimeInput:
    """Seal a complete ordinary projection; do not select a new mode, tools or window."""
    if plan.core is None or not plan.core.memory_view_json:
        raise RealtimeSetupError("missing_memory_view", "the ordinary host capture must include MemoryViewSnapshot")
    from ouroboros.memory_view import snapshot_from_json

    snapshot_from_json(plan.core.memory_view_json)  # malformed captures are not whole-view evidence
    if plan.initial_mode not in {"max", "low", "nano"}:
        raise RealtimeSetupError("context_mode", "the host must select an ordinary context mode")
    if plan.initial_mode == "nano" and plan.nano_projection is None:
        raise RealtimeSetupError("context_mode", "the selected Nano projection is missing")
    if plan.projection(plan.initial_mode).user_content_json is not None and not tool_reader:
        # This is the ordinary fitter's mandatory original-input reader, not an
        # optional memory link. Setup has no tool loop to fulfill that obligation.
        raise RealtimeSetupError("task_input_reader_required", "setup cannot read the required original task input")
    if not isinstance(model, str) or not model.strip():
        raise RealtimeSetupError("model", "the host must select the provider model")
    if type(output_allowance_tokens) is not int or output_allowance_tokens < 1:
        raise RealtimeSetupError("output_allowance", "the host must supply its positive output allowance")
    from ouroboros.llm_messages import project_wire_messages
    return RealtimeInput(
        model=model, messages_json=canonical_json(
            project_wire_messages(plan.messages_for(plan.initial_mode) if messages is None else messages)),
        tools_json=canonical_json(selected_tools), output_allowance_tokens=output_allowance_tokens,
        mode=plan.initial_mode, core_sha256=plan.core_sha256,
        memory_view_sha256=sha256(plan.core.memory_view_json),
    )


def loaded_provider(provider: str) -> dict:
    """A snapshot of a live provider's publication (handler, route, generation)."""
    with _lock:
        spec = _realtime_providers.get(provider)
        if spec is None or spec["skill"] in _unloading or spec["skill"] not in _extensions:
            raise RealtimeSetupError("provider_unavailable", provider)
        return dict(spec)


class _LoopWork:
    """Thread-safe Stop request; completion is the surrounding work scope's exit."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.task = None
        self.cancel_requested = False

    def cancel(self):
        self.loop.call_soon_threadsafe(self._cancel_once)

    def _cancel_once(self):
        if not self.cancel_requested:
            self.cancel_requested = True
            self.task.cancel()


def text_input_estimate(messages, tools):
    """The ordinary chars/4 basis, with media blocks excluded from a text witness."""
    from ouroboros.context_fit import estimate_context_prompt_tokens
    rows = []
    for message in messages:
        row = dict(message)
        if isinstance(row.get("content"), list):
            row["content"] = [part for part in row["content"]
                              if isinstance(part, dict) and part.get("type") == "text"]
        rows.append(row)
    return estimate_context_prompt_tokens(rows, tools)


# The refusal's internal message; the person and the mind read ``capacity_notice``.
CONTEXT_CAPACITY_DETAIL = "This room's context and working reserve do not fit this engine's window."
CALL_ENDED = "Call ended. Its transcript is in this room."
CALL_ENDED_UNSTARTED = "Call ended before it started."  # the owner's End during preparation: there is no transcript
_NEEDS = {True: "The call could not start on {engine}: this room needs {need} tokens of input, and {engine} takes {takes} "
                "on this route today.",
          False: "The call could not start on {engine}: this room needs at least {need} tokens of input, and {engine} "
                 "takes {takes} on this route today."}
_STAGES = {"task_preparation": "Stage: this room's ordinary context was built for the call, before the engine was reached.",
           "session_candidate": "Stage: the call's own instructions and tools were added to the room's context, before the "
                                "session opened."}


def capacity_notice(engine, facts, *, stage, need, measured, reserve, window, fixed=None, parts=(), extra=""):
    """The one System line a call that could not start for lack of room leaves, with its step and Details.

    ``need`` is the input this room needs (a lower bound when ``measured`` is false: memory was not priced); what the
    engine takes is the window minus the reserve, in the same units. ``facts`` are the route's ``RouteFacts`` (or
    their dict). Returns the ``host_notice_row`` the terminal seam turns into the room's row."""
    from ouroboros.host_notice import STEPS, phrase

    def fact(name):
        return facts.get(name) if isinstance(facts, dict) else getattr(facts, name, None)

    takes = max(0, window - reserve)
    unit = ("estimated tokens (characters / 4)" if fact("basis") == "estimator"
            else "tokens, the provider's count through this route's calibration")
    lines = [_STAGES.get(stage, f"Stage: {stage}."),
             f"Window: {window:,} {unit}; source: {fact('source') or 'not stated'}; observed: "
             f"{fact('observed') or 'not stated'}.",
             f"Reserve for the reply and the call's working room: {reserve:,} tokens.",
             f"{engine} takes {takes:,} tokens of input on this route today: the window minus the reserve.",
             (f"Input this room needs: {need:,} tokens" + (f" (fixed part {fixed:,}, memory {need - fixed:,})."
              if fixed is not None else ", measured whole with the call's instructions and tools.")) if measured else
             f"Input this room needs: at least {need:,} tokens, its fixed part alone; memory was not measured."]
    if parts:
        lines.append("Parts, largest first (estimated tokens): " + "; ".join(f"{name} {size:,}" for name, size in parts) + ".")
    if extra:
        lines.append(extra)
    text, said = phrase("context_capacity", _NEEDS[bool(measured)], step="open_calls", details="\n".join(lines),
                        engine=engine, need=f"{need:,}", takes=f"{takes:,}")
    return {"text": text, "record": f"{text} {STEPS['open_calls']}", **said}


def preparation_refusal_notice(turn, unfit):
    """Path 1: the ordinary build refused (``MinimumViewUnfit``) before any call host existed."""
    measured = getattr(unfit, "memory_measured", True)
    book = getattr(unfit, "required_book_tokens", 0)
    return capacity_notice(getattr(turn, "native_model", "") or turn.spec.model, turn.catalog_model.facts,
        stage="task_preparation", need=unfit.estimated_input_tokens, measured=measured,
        reserve=unfit.response_reserve_tokens, window=unfit.window_tokens,
        fixed=unfit.fixed_input_tokens if measured else None, parts=tuple(getattr(unfit, "parts", ()) or ()),
        extra=f"This task directly requires the development handbook: {book:,} tokens of the input." if book else "")


def session_refusal_notice(host):
    """Path 2: the session candidate refused before the first connection opened; ``None`` once a session opened."""
    refusal = host.trace.get("fit_refusal") or {}
    measurement = (refusal.get("fit") or {}).get("measurement") or {}
    if host._session_opened or measurement.get("capacity_total_tokens") is None:
        return None
    return capacity_notice(host.turn.native_model, refusal.get("route_facts") or {}, stage="session_candidate",
        need=measurement["estimated_input_tokens"], measured=True, reserve=measurement["response_reserve_tokens"],
        window=measurement["capacity_total_tokens"], extra=str(refusal.get("detail") or ""))


def call_ending(reason, status, detail="", notice=None, *, opened=True):
    """A call's terminal words and usage: one home for the ending every refusal and End share.

    ``opened`` is false when no session ever opened (the End came during preparation)."""
    if notice is not None:
        words = notice["record"]
    elif reason == "not_connected":  # nobody joined: one short notice, no transcript to point at
        words = f"The call did not start: {detail.rstrip('.')}."
    elif reason == "finished" and not opened:
        words = CALL_ENDED_UNSTARTED
    else:
        words = (CALL_ENDED if status in ("ok", "cancelled") or not detail
                 else f"Call ended: {detail.rstrip('.')}. Its transcript is in this room.")
    usage = {"terminal_origin": "host_notice", "execution_status": status, "reason_code": reason}
    return words, ({**usage, "host_notice_row": notice} if notice is not None else usage)


def _renewal_window(record, facts):
    """Cache session growth room separately from physical response admission.

    Retention and estimator declarations measure the session, with their own
    working reserve. A large reply allowance does not consume that reserve.
    Raiseable routes keep the resolved accepted-input frame, including its reserve.
    """
    from dataclasses import replace

    if facts.acceptance_can_raise:
        return record
    declared = next((limit for limit in record.declared if limit.kind == "total"), None)
    return replace(record, input=None, total=declared, reserve=facts.working_reserve_tokens or 0)


def observe_live_input(host, raw_usage):
    """Teach the ordinary window owner at a completed response boundary.

    The session's captured plan owns the account scope even if credentials have
    changed since open. WindowRecord decides whether acceptance can raise a
    declared limit; retention and estimator windows keep their own units.
    """
    from dataclasses import replace
    from ouroboros.context_window import observation_scope, record_observation, resolve_window
    from ouroboros.live_usage import input_token_witness

    plan = host.plan
    accepted = input_token_witness(raw_usage)
    if (not host._connection_accounting or host.model.facts.basis != "provider"
            or not plan.window_witness_bound or not accepted):
        return
    reserve = host._physical_context.response_reserve_tokens
    record_observation(host.ctx.drive_root, observation_scope(plan), accepted=accepted, reserve=reserve)
    record = resolve_window(plan, plan.window_record.reserve, root=host.ctx.drive_root)
    host.plan = host.ctx.context_fit_plan = replace(plan, window_record=record, window_tokens=record.total_equivalent)
    host._renewal_window = _renewal_window(
        resolve_window(host.plan, reserve, root=host.ctx.drive_root), host.model.facts)


def _session_density(host, plan, estimate):
    from ouroboros.capability_evidence import resolve_main_token_density
    return (resolve_main_token_density(host.ctx.drive_root, plan.route_fp, plan.model,
                                      estimate_tokens=estimate, view_kind=plan.view_kind)
            if host.model.facts.basis == "provider" else (1.0, "cold_estimate"))


def measure_session_wire(host, plan, messages, schemas, factory, options, *, _room_fact=True, fresh=None,
                         canonical_rows=None, opening_allowance_tokens=None):
    """Seal and price one complete candidate, without selecting a view or opening a socket.

    Admission and authored-view receipts use this same envelope measurement.
    A provider session's constructor builds the wire. Only its host-owned open starts I/O or claims resume state.
    Catch-up prices the allowance already sent in setup, not what a new setup could negotiate.
    """
    import math
    from dataclasses import asdict, replace
    from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS, reply_allowance_tokens
    from ouroboros.context_fit import estimate_context_prompt_tokens, measure_main_fit
    from ouroboros.context_measurement import reclaim_low_water_margin, required_book_addition
    from ouroboros.context_window import resolve_window
    from ouroboros.context_view_facts import window_projection
    from ouroboros.llm_attempt import main_context_response_reserve
    from ouroboros.loop_model_call import _main_context_profile, _physical_context_for_fit
    from ouroboros.usage_accounting import AttemptRequest

    if fresh is None:
        fresh = not options.resume_handle
    if fresh:
        import copy
        from ouroboros.tools.tool_discovery import update_resident_notice
        messages = copy.deepcopy(messages)
        update_resident_notice(host.tools, schemas, messages)
        if canonical_rows is not None:
            canonical_rows[:] = messages
    if _room_fact and fresh:
        from ouroboros.context_source_view import fresh_room_rows
        messages, source = fresh_room_rows(messages, drive_root=host.ctx.drive_root, task_id=host.ctx.task_id)
        from ouroboros.context_measurement import room_fact
        from ouroboros.context_view_facts import memory_facts
        from ouroboros.loop_messages import _context_components
        from types import SimpleNamespace
        _, _, measured, _ = measure_session_wire(host, plan, messages, schemas, factory, options, _room_fact=False,
                                                 opening_allowance_tokens=opening_allowance_tokens)
        parts = _context_components(SimpleNamespace(messages=messages, tool_schemas=schemas,
            accumulated_usage={'_context_measurement_density': measured['fit']['measurement']['measurement_density']},
            context_fit_plan=plan), measured=True)
        rows = [*messages, room_fact(measured['fit']['measurement'], parts, source=source,
            memory=memory_facts(plan, plan.initial_mode, messages))]
        session, request, evidence, window = measure_session_wire(host, plan, rows, schemas, factory, options,
            _room_fact=False, opening_allowance_tokens=opening_allowance_tokens)
        if canonical_rows is not None:
            canonical_rows[:] = rows
        return session, request, evidence, window

    facts = host.model.facts
    # Every candidate an engine receives passes here (setup, renewal, an authored view's measurement): a blind
    # engine gets image parts as captions, the canonical history keeps the images (R15-1).
    messages = rows_for_engine(messages, vision=host.model.vision, fresh=fresh, owner_mode=plan.preferred_mode, schemas=schemas)
    rows = [*messages, *([{"role": "system", "content": options.instructions}] if options.instructions else [])]
    fit = measure_main_fit(plan, rows, schemas, profile=_main_context_profile(plan, plan.initial_mode),
        rendered_mode=plan.initial_mode, round_id=host.ctx.task_id + ":call", reasoning_effort=host.effort)
    ceiling = min(plan.output_reserve_tokens, options.model_output_bound_tokens or plan.output_reserve_tokens)
    selected = input_from_context_plan(plan, model=host.turn.native_model, selected_tools=schemas,
        output_allowance_tokens=opening_allowance_tokens or min(ceiling, fit.measurement.reply_allowance_tokens), messages=messages)
    canonical = estimate_context_prompt_tokens(rows, schemas)
    raw_book = required_book_addition(plan, plan.initial_mode)
    def measure(allowance):
        item = replace(selected, output_allowance_tokens=allowance)
        session = factory(item, options, observe=host.observe)
        bound = session.output_bound_tokens
        if bound is not None and (type(bound) is not int or bound < 1):
            raise RealtimeSetupError("output_bound", "provider output bound must be positive or unknown")
        raw = max(canonical, estimate_context_prompt_tokens([session.wire]))
        ratio, basis = _session_density(host, plan, raw)
        measured, book = math.ceil(raw * ratio), math.ceil(raw_book * ratio)
        working = max(facts.working_reserve_tokens or plan.output_reserve_tokens, bound or 0)
        request = AttemptRequest(model=host.turn.model, provider=host.turn.route.accounting_provider,
            drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, source="agent.live_conversation",
            prompt_tokens_estimate=measured, prompt_tokens_bounded_estimate=raw,
            max_completion_tokens=bound or item.output_allowance_tokens, working_reserve_tokens=working,
            allow_live_fetch=False, candidate_measurement_kind="canonical_json_v1",
            physical_context=replace(_physical_context_for_fit(fit), response_reserve_tokens=working))
        reserve = main_context_response_reserve(request)
        window = resolve_window(plan, reserve, root=host.ctx.drive_root, bound=plan.window_witness_bound)
        reply = (reply_allowance_tokens(caller_max_tokens=ceiling, nano=True,
            owner_nano=plan.preferred_mode == 'nano', input_tokens=measured - book,
            raw_input_tokens=raw - raw_book, window_record=window)
            if window.usable is not None else allowance)
        return item, session, request, window, raw, ratio, basis, measured, book, reserve, reply

    price_reply = (plan.initial_mode == 'nano' and opening_allowance_tokens is None
                   and resolve_window(plan, plan.output_reserve_tokens, root=host.ctx.drive_root,
                                      bound=plan.window_witness_bound).usable is not None)
    if price_reply:
        floor = min(NANO_MIN_HEADROOM_TOKENS, ceiling)
        # Try representable decimal widths in [floor, ceiling] from largest to smallest: serialized
        # allowance length depends at most on digit width on supported wires, while token density can
        # drop as well as rise on +1 raw token. Always verify the actual candidate's final wire.
        result = None
        for width in range(len(str(ceiling)), len(str(floor)) - 1, -1):
            band_lo, band_hi = max(floor, 10 ** (width - 1)), min(ceiling, 10 ** width - 1)
            probe = selected.output_allowance_tokens if band_lo <= selected.output_allowance_tokens <= band_hi else band_hi
            trial = measure(probe)
            reply = trial[-1]
            if reply < band_lo:
                continue
            candidate = min(reply, band_hi)
            final = trial if candidate == probe else measure(candidate)
            if candidate <= final[-1]:
                result = final
                break
        if result is None:
            result = measure(floor)
    else:
        result = measure(selected.output_allowance_tokens)
    selected, session, request, window, raw, ratio, basis, measured, book, reserve, _ = result
    wire = canonical_json(session.wire)
    capacity = window.total_equivalent if window.usable is not None else None
    deficit = max(0, measured + reserve - capacity) if capacity is not None else None
    target = fit.measurement.target_total_tokens
    target_deficit = max(0, measured + reserve - target) if target is not None else None
    largest_deficit = max(target_deficit or 0, deficit or 0)
    margin = reclaim_low_water_margin(target, window.boundary) if largest_deficit else 0
    measurement = replace(fit.measurement, estimated_input_tokens=measured, raw_input_tokens=raw,
        measurement_density=ratio, measurement_basis=basis, response_reserve_tokens=reserve,
        window=window_projection(window), window_boundary_tokens=window.boundary, window_record=replace(window, floor=None),
        capacity_total_tokens=capacity, capacity_deficit_tokens=deficit,
        required_book_tokens=book, reply_allowance_tokens=selected.output_allowance_tokens,
        target_deficit_tokens=target_deficit, low_water_margin_tokens=margin, reclaim_goal_tokens=largest_deficit + margin)
    fit = replace(fit, measurement=measurement, predicted_capacity_miss=bool(deficit),
                  action='send_target_miss' if target_deficit else 'send')
    request = replace(request, physical_context=_physical_context_for_fit(fit),
        candidate_raw_sha256=sha256(wire), candidate_raw_size_bytes=len(wire.encode()),
        candidate_context_sha256=selected.sha256, candidate_context_size_bytes=len(selected.messages_json.encode()))
    evidence = {"fit": asdict(fit), "route_facts": asdict(facts), "tool_count": len(schemas),
                "input_sha256": request.candidate_context_sha256, "wire_sha256": request.candidate_raw_sha256,
                "units": facts.basis}
    return session, request, evidence, window


def _publish_room_reader(host, plan, prior, rows, schemas, factory, options, evidence):
    """Publish the exact reader only for the selected send."""
    from ouroboros.context_source_view import fresh_room_rows
    from ouroboros.context_budget import bound_host_row
    import copy
    from ouroboros.context_images import evict_stale_image_blocks
    current = copy.deepcopy(prior)
    evict_stale_image_blocks(current, owner_mode=plan.preferred_mode, readable_only=True, reader_available=False)
    _, placeholder = fresh_room_rows(current, drive_root=host.ctx.drive_root, task_id=host.ctx.task_id)
    _, reader = fresh_room_rows(current, drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, persist=True)
    rows[-1] = bound_host_row('ouroboros_context_facts', rows[-1]['content'].replace(placeholder, reader))
    session, request, sealed, window = measure_session_wire(host, plan, rows, schemas, factory, options, _room_fact=False)
    room = evidence['work_room']
    remaining = (window.usable - sealed['fit']['measurement']['estimated_input_tokens']
                 if window.usable is not None else None)
    sealed.update(recovery=evidence['recovery'], work_room={**room, 'remaining_tokens': remaining,
        'deficit_tokens': max(0, room['requested_tokens'] - remaining) if remaining is not None else 0})
    sealed['fit']['automatic_pass_used'] = evidence['fit']['automatic_pass_used']
    return session, request, sealed, window


def _accept_session_candidate(host, plan, rows, options, physical_window, raw):
    """Publish this privately fitted view; the serving canonical prefix changes at adoption."""
    facts = host.model.facts
    complete = [*rows, *([{"role": "system", "content": options.instructions}] if options.instructions else [])]
    host._renewal_window, host._renewal_facts = _renewal_window(physical_window, facts), facts
    host._session_instructions = options.instructions
    host._selected_text_estimate = text_input_estimate(complete, host.schemas)
    has_media = any(isinstance(row.get('content'), list) and any(
        not isinstance(part, dict) or part.get('type') != 'text' for part in row['content']) for row in complete)
    # Only text-only native envelopes can teach density from the complete sealed denominator.
    host._density_input = host._selected_text_estimate if has_media else raw
    host._selected_raw_estimate = raw
    host.plan = host.ctx.context_fit_plan = plan
    host.messages[:] = rows
    host._context_changed()
    host.ctx.active_context_mode = plan.initial_mode


def _reply_room(plan, measurement, floor):
    """The part of a candidate's reserve that is its own reply above its mode's least reserve.

    Nano's reply takes what its input leaves beside the estimate slack (``context_budget.reply_allowance_tokens``) and
    shrinks before the window refuses, so a running call can grow into that part; folding memory only lengthens the
    reply. A native bound and a reply at its floor are fixed: none of their reserve counts.
    """
    from ouroboros.context_budget import context_mode_limits
    reserve, reply = measurement['response_reserve_tokens'], measurement['reply_allowance_tokens']
    least = context_mode_limits(plan.initial_mode, plan.preferred_mode, plan.output_reserve_tokens,
                                working_reserve_tokens=floor)[1]
    return max(0, reserve - least) if reserve == reply else 0


def session_candidate(host, factory, options, *, observation_fp=None):
    """Refit current canonical rows through the ordinary window ladder, then seal.

    Additional feature instructions and current schemas participate in the same
    fit. The physical guard checks the provider's final envelope. A failed fit
    asks the existing ladder for that measured deficit; unchanged bytes refuse.
    The sealed raw input (``reproject_for_route(measured_input_tokens=...)``) is
    supplied on a wire refusal or a first send's room shortage and discarded when
    the mode or the tools change.
    """
    import copy
    import math
    from dataclasses import replace
    from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR, bound_host_row
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.context_measurement import reclaim_low_water_margin
    from ouroboros.context_window import WindowLimit, WindowRecord, resolve_window
    from ouroboros.llm_attempt import main_context_fit_refusal

    plan, facts = host.plan, host.model.facts
    known = facts.conversation_window_tokens is not None
    window = facts.conversation_window_tokens or 0
    # The plan keeps its reply allowance; the route's working room travels beside it
    # and the ordinary ladder honours both. The physical guard below checks the larger.
    reserve = facts.working_reserve_tokens or plan.output_reserve_tokens
    limit = WindowLimit(window, "total", facts.source, facts.observed, facts.acceptance_can_raise) if known else None
    scope = host.turn.observation_fingerprint() if observation_fp is None else observation_fp
    record = WindowRecord(scope, total=limit, reserve=reserve,
                          declared=(limit,) if limit else ())
    plan = replace(plan, route_fp=scope, window_witness_bound=bool(scope), window_record=record, window_tokens=window, status="confirmed" if known else "unprobeable", stale=False,
                   working_reserve_tokens=facts.working_reserve_tokens, window_basis=facts.basis)
    record = resolve_window(plan, reserve, root=host.ctx.drive_root, bound=plan.window_witness_bound)
    window, known = record.total_equivalent, record.usable is not None
    plan = replace(plan, window_record=record, window_tokens=window)
    start_work = not getattr(host, "_session_opened", False)
    extra = [{"role": "system", "content": options.instructions}] if options.instructions else []
    from ouroboros.context_source_view import emergency_address_view
    from ouroboros.context_budget import ContextReclaimRequest, REFUSAL_PASSES
    from ouroboros.context_compaction import context_reclaim_transcript_sha256
    from ouroboros.context_renewal import renewal_view, physical_window_cut
    from ouroboros.capability_breakage import _TOLD, write_start_notice
    from ouroboros.tools.tool_discovery import update_resident_notice

    current = list(host.messages)
    schema_mode = host._schemas_mode
    loaded = host.ctx._actor_loaded_tool_names - host.call_tools.keys()
    schemas = [current_schema for schema in host.schemas if (current_schema := (
        host.tools.get_schema_by_name(schema['function']['name']) if schema['function']['name'] in loaded else schema)) is not None]
    if schemas != host.schemas:
        update_resident_notice(host.tools, schemas, current)
    told = getattr(host.ctx, _TOLD, None)

    selected_by_mode = {schema_mode: schemas}

    def schemas_for(mode):
        if mode not in selected_by_mode:
            selected_by_mode[mode] = host._tool_schemas_for_mode(mode)
        return selected_by_mode[mode]

    trial_catalogs = dict(low_tools=lambda: schemas_for('low'), nano_tools=lambda: schemas_for('nano'))

    def project(rebound):
        rows = rebound.reproject_transcript(current, rebound.initial_mode)
        if rebound.initial_mode == schema_mode:
            return rows, schemas, told
        selected = schemas_for(rebound.initial_mode)
        trial = copy.copy(host.ctx)
        trial.is_direct_chat = False  # this trial cannot publish Main's capability facts
        setattr(trial, _TOLD, copy.deepcopy(told))
        write_start_notice(host.tools.copy_for(trial), rows)
        update_resident_notice(host.tools, selected, rows)
        return rows, selected, getattr(trial, _TOLD, None)

    running = known and not start_work
    ordered = (running and plan.initial_mode == plan.preferred_mode == "max"
               and plan.core and not plan.core.compact_reference_docs)
    if options.resume_handle and not running:
        raw = estimate_context_prompt_tokens([*current, *extra], schemas)
        ratio, _ = _session_density(host, plan, raw)
        plan = plan.reproject_for_route(window_tokens=window, known_window=known, ratio=ratio,
            output_reserve=plan.output_reserve_tokens, tool_schemas=schemas, start_mode=plan.initial_mode,
            current_messages=[*current, *extra], start_work=start_work, **trial_catalogs)
        current, schemas, told = project(plan)
        schema_mode = plan.initial_mode
    steps = iter(("floor",) if options.resume_handle else (*REFUSAL_PASSES[0], *(step for step in REFUSAL_PASSES[1] if step != "books"), "seed", "physical_cut"))
    receipts = []
    room_attempted = False
    room_facts = not options.resume_handle
    return_room, floor_attempted = 0, False
    while True:
        rows = [*current, *extra]
        sent_rows = list(current)
        session, request, evidence, physical_window = measure_session_wire(
            host, plan, current, schemas, factory, options, canonical_rows=sent_rows, _room_fact=room_facts)
        measurement = evidence['fit']['measurement']
        measured, raw = measurement['estimated_input_tokens'], measurement['raw_input_tokens']
        ratio = measurement['measurement_density']
        usable = physical_window.usable
        available = usable - measured if usable is not None else None
        wanted = math.ceil(physical_window.boundary / RECLAIM_LOW_WATER_DIVISOR) if start_work and usable is not None else 0
        shortage = max(0, wanted - available) if available is not None else 0
        evidence['work_room'] = dict(first_send=start_work, requested_tokens=wanted,
            remaining_tokens=available, deficit_tokens=shortage)
        evidence['recovery'] = list(receipts)
        evidence['fit']['automatic_pass_used'] = bool(receipts)
        refusal = main_context_fit_refusal(request)
        if refusal is None and shortage and not room_attempted:
            room_attempted = True
            # Soft start room uses the ordinary memory walk; it never removes delivered rows.
            rebound = plan.reproject_for_route(window_tokens=window, known_window=known, ratio=ratio,
                output_reserve=plan.output_reserve_tokens, tool_schemas=schemas, start_mode=plan.initial_mode,
                current_messages=rows, refused_tokens=shortage, measured_input_tokens=raw, start_work=start_work,
                **trial_catalogs)
            current, schemas, told = project(rebound)
            plan, schema_mode = rebound, rebound.initial_mode
            continue
        if refusal is None and shortage:
            detail = ('Start room is smaller than requested: the complete input and reply/working reserve '
                      'leave less than one eighth of the window. The physical request fits; this work starts with less room.')
            evidence['work_room'].update(reason='start_room_shortfall', detail=detail)
            if not any(row.get('_host_context_kind') == 'context_start_room' for row in current):
                notice = bound_host_row('context_start_room', detail)
                _, with_notice, _, _ = measure_session_wire(host, plan, [*current, notice], schemas, factory, options, _room_fact=room_facts)
                if main_context_fit_refusal(with_notice) is None:
                    current.append(notice)
                    continue
        book = measurement['required_book_tokens']
        book_only = bool(refusal and book and 0 < -refusal['free_tokens'] <= book)
        if refusal and not book_only and not return_room:
            return_room = reclaim_low_water_margin(None, physical_window.boundary)
        ordinary_free = available + book + _reply_room(plan, measurement, reserve) if available is not None else None
        top_up = (refusal is None and not start_work and not floor_attempted
                  and ordinary_free is not None and ordinary_free < return_room)
        if refusal is None and not top_up:
            if room_facts:
                published = _publish_room_reader(host, plan, current, sent_rows, schemas, factory, options, evidence)
                session, request, evidence, physical_window = published
            if schema_mode != host._schemas_mode or schemas != host.schemas:
                host._reselect_tools(schema_mode, selected=schemas)
                if told is not None:
                    setattr(host.ctx, _TOLD, told)
            _accept_session_candidate(host, plan, sent_rows, options, physical_window, evidence['fit']['measurement']['raw_input_tokens'])
            return session, request, evidence
        rung = None if book_only else "floor" if top_up else next(steps, None)
        if rung is not None:
            def measure(candidate):
                return measure_session_wire(host, plan, candidate, schemas, factory, options, _room_fact=room_facts)[2]['fit']['measurement']['estimated_input_tokens']
            usable += book  # transcript and memory recovery cannot pay for a directly required handbook
            goal = max(0, return_room - ordinary_free)
            if rung == "bodies" and not start_work:  # visible bodies pay only the physical excess
                goal = max(0, -ordinary_free)
                if not goal:
                    continue
            before = measured
            if rung == "floor":
                floor_attempted = True
                # The same memory + book walk as text, after exposed tool bodies.
                # Each iteration must shrink the complete provider envelope; the slack never outlines.
                while True:
                    # The wire's reserve is this mode's reply (or native bound), a ceiling for this fit only; the
                    # route's working floor stays the floor, so a Nano trial is priced at its own reserve. The
                    # plan's allowance then goes into the next complete candidate.
                    basis = replace(plan, window_record=physical_window, working_reserve_tokens=reserve) if running else plan
                    room = min(return_room, goal)
                    rebound = basis.reproject_for_route(window_tokens=physical_window.total_equivalent if running else window,
                        known_window=known, ratio=ratio, output_reserve=physical_window.reserve if running else plan.output_reserve_tokens,
                        tool_schemas=schemas, start_mode=plan.initial_mode,
                        current_messages=[*(sent_rows if running else current), *extra], refused_tokens=goal - room,
                        slack_tokens=room, measured_input_tokens=raw, start_work=False, telling_floor=bool(ordered),
                        unsent_start=start_work,  # an unsent call start still spares its detailed book maps (#3649)
                        select_mode="low" if not ordered and plan.initial_mode == "max" else None, **trial_catalogs)
                    if running:
                        rebound = replace(rebound, output_reserve_tokens=plan.output_reserve_tokens,
                                          working_reserve_tokens=plan.working_reserve_tokens)
                    candidate, selected, trial_told = project(rebound)
                    trial_rows = list(candidate)
                    _, trial, next_fit, next_window = measure_session_wire(host, rebound, candidate, selected,
                        factory, options, canonical_rows=trial_rows, _room_fact=room_facts)
                    next_measurement = next_fit['fit']['measurement']
                    after = next_measurement['estimated_input_tokens']
                    next_free = (next_window.usable - after + next_measurement['required_book_tokens']
                                 + _reply_room(rebound, next_measurement, reserve))
                    if next_free <= ordinary_free or top_up and main_context_fit_refusal(trial) is not None:
                        break
                    receipts.append({'rung': rung, 'before_tokens': before, 'after_tokens': after})
                    plan, current, schemas, told = rebound, candidate, selected, trial_told
                    schema_mode = rebound.initial_mode
                    physical_window, sent_rows = next_window, trial_rows
                    ordinary_free, before = next_free, after
                    raw, ratio = next_measurement['raw_input_tokens'], next_measurement['measurement_density']
                    goal = max(0, return_room - ordinary_free)
                    if not goal or ordinary_free >= 0 and (not running or after > physical_window.usable):
                        break
                continue
            elif rung == "seed":
                current, receipt = renewal_view(current,
                    observation=getattr(host.ctx, '_last_context_observation', None),
                    drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, measure=measure)
            elif rung == "physical_cut":
                current, receipt = physical_window_cut(current, drive_root=host.ctx.drive_root,
                    task_id=host.ctx.task_id, measure=measure, usable=usable)
            else:
                reclaim = ContextReclaimRequest(route_fp=plan.route_fp,
                    round_id=f"{host.ctx.task_id}:call:{host.connection_number}",
                    transcript_sha256=context_reclaim_transcript_sha256(current),
                    measurement_basis=measurement['measurement_basis'], measurement_density=ratio,
                    reclaim_goal_tokens=goal, allow_partial_shrink=True)
                current, receipt = emergency_address_view(current, reclaim, rung=rung,
                    observation=getattr(host.ctx, '_last_context_observation', None), measure_candidate=measure,
                    drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, usable_input=usable,
                    replacing_host_kinds=('ouroboros_context_facts',))
            after = measure(current)
            if after < measured:
                receipts.append({'rung': rung, 'before_tokens': measured, 'after_tokens': after})
            continue
        if room_facts:
            room_facts = False  # refusal evidence names no unpublished prior-view reader
            continue
        from ouroboros.observability import persist_physical_candidate
        detail = CONTEXT_CAPACITY_DETAIL
        if book_only:
            from ouroboros.context_view_facts import required_book_fact
            detail = required_book_fact({}, {"book": book, "capacity": measurement['capacity_total_tokens'],
                "reserve": measurement['response_reserve_tokens'], "tokens": measured}).strip()
            evidence['detail'] = detail
        evidence.update(reason="context_capacity", refusal=refusal,
                        deficit_tokens=max(1, -refusal["free_tokens"] + 1))
        receipt = persist_physical_candidate(host.ctx.drive_root, task_id=host.ctx.task_id,
            attempt_id=f"call-{host.turn.spec.call_id}-{host.connection_number}-refused",
            candidate=session.wire, candidate_facts=evidence)
        event = {"type": "call.fit_refused", **evidence, "manifest_ref": receipt["manifest_ref"]}
        host.trace["fit_refusal"] = event
        host.observe("host", canonical_json(event))
        raise RealtimeSetupError("context_capacity", detail)
