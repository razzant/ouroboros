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
    return RealtimeInput(
        model=model, messages_json=canonical_json(
            plan.messages_for(plan.initial_mode) if messages is None else messages),
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


def session_candidate(host, factory, options, *, observation_fp=None):
    """Refit current canonical rows through the ordinary window ladder, then seal.

    Additional feature instructions and current schemas participate in the same
    fit. The physical guard checks the provider's final envelope. A failed fit
    asks the existing ladder for that measured deficit; unchanged bytes refuse.
    """
    import math
    from dataclasses import asdict, replace
    from ouroboros.capability_evidence import is_known, resolve_main_token_density
    from ouroboros.context_fit import estimate_context_prompt_tokens, measure_main_fit
    from ouroboros.context_window import WindowLimit, WindowRecord, resolve_window
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.loop_model_call import _main_context_profile, _physical_context_for_fit
    from ouroboros.usage_accounting import AttemptRequest

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
    extra = [{"role": "system", "content": options.instructions}] if options.instructions else []
    rows = [*host.messages, *extra]
    raw = estimate_context_prompt_tokens(rows, host.schemas)
    def density(estimate):
        return (resolve_main_token_density(host.ctx.drive_root, plan.route_fp, plan.model,
                                           estimate_tokens=estimate, view_kind=plan.view_kind)
                if facts.basis == "provider" else (1.0, "cold_estimate"))
    ratio, basis = density(raw)
    deficit, previous = 0, None
    while True:
        plan = plan.reproject_for_route(window_tokens=window, known_window=known, ratio=ratio,
            output_reserve=plan.output_reserve_tokens, tool_schemas=host.schemas, start_mode=plan.initial_mode,
            current_messages=rows, refused_tokens=deficit)
        if plan.initial_mode != host._schemas_mode:
            # The ordinary ladder chose this mode. Rebind its starting tools,
            # keeping explicit discovery, then fit the changed envelope once.
            host._reselect_tools(plan.initial_mode)
            rows = [*host.messages, *extra]
            previous, deficit = None, 0
            continue
        current = plan.reproject_transcript(host.messages, plan.initial_mode)
        rows = [*current, *extra]
        fit = measure_main_fit(plan, rows, host.schemas, profile=_main_context_profile(plan, plan.initial_mode),
            rendered_mode=plan.initial_mode, round_id=host.ctx.task_id + ":call", reasoning_effort=host.effort)
        selected = input_from_context_plan(plan, model=host.turn.native_model, selected_tools=host.schemas,
            output_allowance_tokens=fit.measurement.reply_allowance_tokens, messages=current)
        session = factory(selected, options, observe=host.observe)
        bound = session.output_bound_tokens
        if bound is not None and (type(bound) is not int or bound < 1):
            raise RealtimeSetupError("output_bound", "provider output bound must be positive or unknown")
        wire = canonical_json(session.wire)
        raw = estimate_context_prompt_tokens(rows, host.schemas)
        # Budget the sealed envelope as well as the ordinary selection. This
        # includes provider-added instructions/language sentences without core
        # knowing provider field names. The ordinary ladder reclaims any deficit.
        raw = max(raw, estimate_context_prompt_tokens([session.wire]))
        ratio, basis = density(raw)
        measured = math.ceil(raw * ratio)
        working = max(reserve, bound or 0)
        physical_window = resolve_window(plan, working, root=host.ctx.drive_root, bound=plan.window_witness_bound)
        measurement = replace(fit.measurement, estimated_input_tokens=measured, raw_input_tokens=raw,
            measurement_density=ratio, measurement_basis=basis,
            response_reserve_tokens=working,
            capacity_total_tokens=physical_window.total_equivalent if is_known(plan) else None)
        fit = replace(fit, measurement=measurement)
        request = AttemptRequest(model=host.turn.model, provider=host.turn.route.accounting_provider,
            drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, source="agent.live_conversation",
            prompt_tokens_estimate=measured, prompt_tokens_bounded_estimate=raw,
            max_completion_tokens=bound or selected.output_allowance_tokens, working_reserve_tokens=working,
            allow_live_fetch=False, candidate_raw_sha256=sha256(wire), candidate_raw_size_bytes=len(wire.encode()),
            candidate_context_sha256=selected.sha256, candidate_context_size_bytes=len(selected.messages_json.encode()),
            candidate_measurement_kind="canonical_json_v1", physical_context=_physical_context_for_fit(fit))
        refusal = main_context_fit_refusal(request)
        evidence = {"fit": asdict(fit), "route_facts": asdict(facts), "tool_count": len(host.schemas),
                    "input_sha256": selected.sha256, "wire_sha256": sha256(wire), "units": facts.basis}
        if refusal is None:
            host._renewal_window, host._renewal_facts = _renewal_window(physical_window, facts), facts
            host._session_instructions = options.instructions
            host._selected_text_estimate = text_input_estimate(rows, host.schemas)
            host._density_input = host._selected_text_estimate
            host._selected_raw_estimate = raw
            host.plan = host.ctx.context_fit_plan = plan
            host.messages[:] = current
            host._context_changed()
            host.ctx.active_context_mode = plan.initial_mode
            return session, request, evidence
        if known and plan.initial_mode == "max":
            # The ordinary pre-send rung after an actual Max overflow. Once
            # Low is selected the same physical floor can continue to Nano.
            plan = plan.reproject_for_route(window_tokens=window, known_window=known, ratio=ratio,
                output_reserve=plan.output_reserve_tokens, tool_schemas=host.schemas, start_mode="max",
                current_messages=rows, select_mode="low")
            previous, deficit = None, 0
            continue
        if previous is not None and measured >= previous:
            from ouroboros.observability import persist_physical_candidate

            # This final candidate never opens a connection, but its measured
            # refusal must remain as inspectable as an admitted preparation.
            evidence.update(reason="context_capacity", refusal=refusal,
                            deficit_tokens=max(1, -refusal["free_tokens"] + 1))
            receipt = persist_physical_candidate(host.ctx.drive_root, task_id=host.ctx.task_id,
                attempt_id=f"call-{host.turn.spec.call_id}-{host.connection_number}-refused",
                candidate=session.wire, candidate_facts=evidence)
            event = {"type": "call.fit_refused", **evidence, "manifest_ref": receipt["manifest_ref"]}
            host.trace["fit_refusal"] = event
            host.observe("host", canonical_json(event))
            raise RealtimeSetupError("context_capacity", CONTEXT_CAPACITY_DETAIL)
        previous, deficit = measured, max(1, -refusal["free_tokens"] + 1)
