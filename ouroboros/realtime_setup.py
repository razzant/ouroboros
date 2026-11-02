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


CONTEXT_CAPACITY_DETAIL = "This room's context and working reserve do not fit this engine's window."


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


def measure_session_wire(host, plan, messages, schemas, factory, options, *, _room_fact=True, fresh=None, canonical_rows=None):
    """Seal and price one complete candidate, without selecting a view or opening a socket.

    Admission and authored-view receipts use this same envelope measurement.
    Instantiating a provider session is inert; only its host-owned open starts I/O.
    """
    import math
    from dataclasses import asdict, replace
    from ouroboros.context_fit import estimate_context_prompt_tokens, measure_main_fit
    from ouroboros.context_window import resolve_window
    from ouroboros.context_view_facts import window_projection
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
        _, _, measured, _ = measure_session_wire(host, plan, messages, schemas, factory, options, _room_fact=False)
        parts = _context_components(SimpleNamespace(messages=messages, tool_schemas=schemas,
            accumulated_usage={'_context_measurement_density': measured['fit']['measurement']['measurement_density']},
            context_fit_plan=plan), measured=True)
        rows = [*messages, room_fact(measured['fit']['measurement'], parts, source=source,
            memory=memory_facts(plan, plan.initial_mode, messages))]
        session, request, evidence, window = measure_session_wire(host, plan, rows, schemas, factory, options, _room_fact=False)
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
    selected = input_from_context_plan(plan, model=host.turn.native_model, selected_tools=schemas,
        output_allowance_tokens=fit.measurement.reply_allowance_tokens, messages=messages)
    session = factory(selected, options, observe=host.observe)
    bound = session.output_bound_tokens
    if bound is not None and (type(bound) is not int or bound < 1):
        raise RealtimeSetupError("output_bound", "provider output bound must be positive or unknown")
    wire = canonical_json(session.wire)
    raw = max(estimate_context_prompt_tokens(rows, schemas), estimate_context_prompt_tokens([session.wire]))
    ratio, basis = _session_density(host, plan, raw)
    measured = math.ceil(raw * ratio)
    working = max(facts.working_reserve_tokens or plan.output_reserve_tokens, bound or 0)
    window = resolve_window(plan, working, root=host.ctx.drive_root, bound=plan.window_witness_bound)
    capacity = window.total_equivalent if window.usable is not None else None
    deficit = max(0, measured + working - capacity) if capacity is not None else None
    from ouroboros.context_measurement import reclaim_low_water_margin
    target = fit.measurement.target_total_tokens
    target_deficit = max(0, measured + working - target) if target is not None else None
    largest_deficit = max(target_deficit or 0, deficit or 0)
    margin = reclaim_low_water_margin(target, capacity) if largest_deficit else 0
    measurement = replace(fit.measurement, estimated_input_tokens=measured, raw_input_tokens=raw,
        measurement_density=ratio, measurement_basis=basis, response_reserve_tokens=working,
        window=window_projection(window), capacity_total_tokens=capacity, capacity_deficit_tokens=deficit,
        target_deficit_tokens=target_deficit, low_water_margin_tokens=margin, reclaim_goal_tokens=largest_deficit + margin)
    fit = replace(fit, measurement=measurement, predicted_capacity_miss=bool(deficit),
                  action='send_target_miss' if target_deficit else 'send')
    request = AttemptRequest(model=host.turn.model, provider=host.turn.route.accounting_provider,
        drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, source="agent.live_conversation",
        prompt_tokens_estimate=measured, prompt_tokens_bounded_estimate=raw,
        max_completion_tokens=bound or selected.output_allowance_tokens, working_reserve_tokens=working,
        allow_live_fetch=False, candidate_raw_sha256=sha256(wire), candidate_raw_size_bytes=len(wire.encode()),
        candidate_context_sha256=selected.sha256, candidate_context_size_bytes=len(selected.messages_json.encode()),
        candidate_measurement_kind="canonical_json_v1", physical_context=_physical_context_for_fit(fit))
    evidence = {"fit": asdict(fit), "route_facts": asdict(facts), "tool_count": len(schemas),
                "input_sha256": selected.sha256, "wire_sha256": sha256(wire), "units": facts.basis}
    return session, request, evidence, window


def _publish_room_reader(host, plan, prior, rows, factory, options, evidence):
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
    session, request, sealed, window = measure_session_wire(host, plan, rows, host.schemas, factory, options, _room_fact=False)
    sealed.update({key: evidence[key] for key in ('work_room', 'recovery')})
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


def session_candidate(host, factory, options, *, observation_fp=None):
    """Refit current canonical rows through the ordinary window ladder, then seal.

    Additional feature instructions and current schemas participate in the same
    fit. The physical guard checks the provider's final envelope. A failed fit
    asks the existing ladder for that measured deficit; unchanged bytes refuse.
    """
    import math
    from dataclasses import replace
    from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR, bound_host_row
    from ouroboros.context_fit import estimate_context_prompt_tokens
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
    current = list(host.messages)
    if options.resume_handle:
        raw = estimate_context_prompt_tokens([*current, *extra], host.schemas)
        ratio, _ = _session_density(host, plan, raw)
        plan = plan.reproject_for_route(window_tokens=window, known_window=known, ratio=ratio,
            output_reserve=plan.output_reserve_tokens, tool_schemas=host.schemas, start_mode=plan.initial_mode,
            current_messages=[*current, *extra], start_work=start_work)
        current = plan.reproject_transcript(current, plan.initial_mode)
    steps = iter(("floor",) if options.resume_handle else (*REFUSAL_PASSES[0], *(step for step in REFUSAL_PASSES[1] if step != "books"), "seed", "physical_cut"))
    receipts = []
    room_attempted = False
    room_facts = not options.resume_handle
    while True:
        rows = [*current, *extra]
        sent_rows = list(current)
        session, request, evidence, physical_window = measure_session_wire(
            host, plan, current, host.schemas, factory, options, canonical_rows=sent_rows, _room_fact=room_facts)
        measurement = evidence['fit']['measurement']
        measured, raw = measurement['estimated_input_tokens'], measurement['raw_input_tokens']
        ratio = measurement['measurement_density']
        available = physical_window.total_equivalent - measured - physical_window.reserve if known else None
        wanted = math.ceil(physical_window.boundary / RECLAIM_LOW_WATER_DIVISOR) if start_work and known else 0
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
                output_reserve=plan.output_reserve_tokens, tool_schemas=host.schemas, start_mode=plan.initial_mode,
                current_messages=rows, refused_tokens=shortage, measured_input_tokens=raw, start_work=start_work)
            changed = rebound.reproject_transcript(current, rebound.initial_mode)
            plan, current = rebound, changed
            if plan.initial_mode != host._schemas_mode:
                host.messages[:] = current
                host._reselect_tools(plan.initial_mode)
                current = list(host.messages)
            continue
        if refusal is None and shortage:
            detail = ('Start room is smaller than requested: the complete input and reply/working reserve '
                      'leave less than one eighth of the window. The physical request fits; this work starts with less room.')
            evidence['work_room'].update(reason='start_room_shortfall', detail=detail)
            if not any(row.get('_host_context_kind') == 'context_start_room' for row in current):
                notice = bound_host_row('context_start_room', detail)
                _, with_notice, _, _ = measure_session_wire(host, plan, [*current, notice], host.schemas, factory, options, _room_fact=room_facts)
                if main_context_fit_refusal(with_notice) is None:
                    current.append(notice)
                    continue
        if refusal is None:
            if room_facts:
                published = _publish_room_reader(host, plan, current, sent_rows, factory, options, evidence)
                session, request, evidence, physical_window = published
            _accept_session_candidate(host, plan, sent_rows, options, physical_window, evidence['fit']['measurement']['raw_input_tokens'])
            return session, request, evidence
        rung = next(steps, None)
        if rung is not None:
            def measure(candidate):
                return measure_session_wire(host, plan, candidate, host.schemas, factory, options, _room_fact=room_facts)[2]['fit']['measurement']['estimated_input_tokens']
            usable = physical_window.total_equivalent - physical_window.reserve
            goal = max(0, measured - usable) + math.ceil(physical_window.boundary / RECLAIM_LOW_WATER_DIVISOR)
            before = measured
            if rung == "floor":
                # The same memory + book walk as text, after exposed tool bodies.
                # Each iteration must shrink the complete provider envelope.
                while True:
                    rebound = plan.reproject_for_route(window_tokens=window, known_window=known, ratio=ratio,
                        output_reserve=plan.output_reserve_tokens, tool_schemas=host.schemas, start_mode=plan.initial_mode,
                        current_messages=[*current, *extra], refused_tokens=goal, measured_input_tokens=raw,
                        start_work=False, select_mode="low" if plan.initial_mode == "max" else None)
                    candidate = rebound.reproject_transcript(current, rebound.initial_mode)
                    old_plan = plan
                    plan = rebound
                    after = measure(candidate)
                    if after >= before:
                        plan = old_plan
                        break
                    current, before = candidate, after
                    if plan.initial_mode != host._schemas_mode:
                        host.messages[:] = current
                        host._reselect_tools(plan.initial_mode)
                        current = list(host.messages)
                    if after <= usable:
                        break
                    raw = estimate_context_prompt_tokens([*current, *extra], host.schemas)
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
        evidence.update(reason="context_capacity", refusal=refusal,
                        deficit_tokens=max(1, -refusal["free_tokens"] + 1))
        receipt = persist_physical_candidate(host.ctx.drive_root, task_id=host.ctx.task_id,
            attempt_id=f"call-{host.turn.spec.call_id}-{host.connection_number}-refused",
            candidate=session.wire, candidate_facts=evidence)
        event = {"type": "call.fit_refused", **evidence, "manifest_ref": receipt["manifest_ref"]}
        host.trace["fit_refusal"] = event
        host.observe("host", canonical_json(event))
        raise RealtimeSetupError("context_capacity", CONTEXT_CAPACITY_DETAIL)
