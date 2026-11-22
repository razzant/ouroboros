"""The round boundary's route handover: live roles and explicit primary choices.

Owns the shared sanitize/fit rebind for a normal round and a post-mail owner act.
The forced call and its budget lookahead bind the acting route here without consuming the next round’s choice. A tool-budget
tail binds the retained route without consuming overrides or publishing a fresh lead;
a drained owner act reapplies the durable primary before this round's call.
Call dispatch and fallback attempts remain in loop_model_call.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List

from ouroboros.llm import LLMClient, normalize_reasoning_effort
from ouroboros.tools.registry import ToolRegistry


def _loop():
    """Resolve the shared route hooks through their existing loop bindings."""
    from ouroboros import loop

    return loop


@contextmanager
def fallback_round_view(owner, messages, schemas, usage):
    """An interrupted walk is saved as the round that entered it, including its resident tools.

    Physical receipts and the wait controller's owner choices belong to the task, even when
    its candidate never answered. Snapshot only the provisional rendering and route fields.
    """
    from ouroboros.context_anchor import _restore_context_fit_usage, _snapshot_context_fit_usage
    from ouroboros.loop_llm_call import REBOUND_PHYSICAL_CONTEXT_KEY
    from ouroboros.memory_inventory import VIEW_TRACE_KEY
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.transcript_prefix import ATTEMPT_PREFIX_KEY, OBSERVER_STATE_ATTRS
    from ouroboros.usage_accounting import BudgetExceeded

    transcript, resident = deepcopy(messages), list(schemas) if schemas is not None else None
    missing = object()
    fields = {key: getattr(owner, key, missing) for key in (
        "active_model", "active_use_local", "active_context_mode", "context_fit_plan",
        "_route_left_out_tool_names", "_pending_handover_reason", "_route_facts_pending",
        "_authored_memory_plan", "route_choice", "memory_view_facts", *OBSERVER_STATE_ATTRS)}
    route_usage = {key: usage.get(key, missing) for key in ("_model_route", REBOUND_PHYSICAL_CONTEXT_KEY, ATTEMPT_PREFIX_KEY)}
    primary_fit = _snapshot_context_fit_usage(usage)
    trace = getattr(owner, "_execution_trace", None)
    view_fact = trace.get(VIEW_TRACE_KEY, missing) if isinstance(trace, dict) else missing

    def restore():
        messages[:] = deepcopy(transcript)
        owner.messages = messages
        if schemas is not None:
            schemas[:] = resident
        for key, value in fields.items():
            if value is not missing:
                setattr(owner, key, value)
            elif hasattr(owner, key):
                delattr(owner, key)
        for mapping, values in ((usage, route_usage), (trace, {VIEW_TRACE_KEY: view_fact})):
            if isinstance(mapping, dict):
                for key, value in values.items():
                    if value is missing:
                        mapping.pop(key, None)
                    else:
                        mapping[key] = value
        _restore_context_fit_usage(usage, primary_fit)

    try:
        yield restore
    except (ModelWaitInterrupted, BudgetExceeded):
        restore()
        raise


def reprepare_source(ctx, kwargs):
    """Rebuild the acting call from canonical messages, including a native reset only on its receipt."""
    model = kwargs["model"]
    observed = kwargs.pop("_model_observed_route", None)
    native_reset = bool(observed and observed.pop("_native_reset", False))
    # Waiting kwargs can contain a caption/off projection. Keep the canonical
    # images, then build the newly selected route's physical view below.
    prepared = LLMClient.sanitize_reasoning_on_model_switch(ctx.messages, ctx.active_model, model)
    if native_reset:
        from ouroboros.llm_messages import drop_source_native_messages, reset_native_messages

        prepared, changed = reset_native_messages(
            prepared, observed, source=observed.get("source"), model=observed.get("model"))
        if not changed:
            prepared, _ = drop_source_native_messages(
                prepared, source=str(observed.get("source") or ""))
    ctx.messages[:] = prepared
    return observed


def reprepare_wire(ctx, kwargs, *, deadline_ts):
    """Build the selected model's image view and wire options outside the rejected Main attempt."""
    from pathlib import Path
    from ouroboros.usage_accounting import bind_physical_attempt_context

    model, role, use_local = kwargs["model"], kwargs["model_role"], kwargs.get("use_local", False)
    from ouroboros.send_clock import MainSendClock
    from ouroboros.loop_llm_call import _prepare_main_messages
    from ouroboros.llm_attempt import single_physical_send, track_physical_attempts

    # Any vision work belongs to this actual reprepare, outside both the Main
    # fit probe and the failed attempt's physical precondition/measurement —
    # and outside Main's clock and send claim: a helper is not a Main request.
    with bind_physical_attempt_context(None), MainSendClock(None).bound(), single_physical_send(None), track_physical_attempts(None):
        kwargs["messages"] = _prepare_main_messages(
            ctx.messages, model=model, model_role=role,
            model_account_override=kwargs.get("model_account_override"),
            llm=ctx.llm, accumulated_usage=ctx.accumulated_usage,
            drive_root=ctx.drive_root or Path(ctx.drive_logs).parent, task_id=ctx.task_id,
            event_queue=ctx.event_queue, use_local=use_local,
            task_attempt=ctx.accumulated_usage.get("_task_attempt"),
            deadline_ts=deadline_ts,
        )
    from ouroboros.llm_claudexor import cache_key_for_model
    from ouroboros.provider_models import provider_for_model
    kwargs["cache_affinity"] = "" if use_local else cache_key_for_model(model)
    kwargs["allow_server_web_search"] = (_loop()._server_web_allowed_by_task(getattr(ctx.tools, "_ctx", None))
                                         and not use_local and provider_for_model(model) != "claudexor")
    if provider_for_model(model) == "claudexor":
        kwargs["bypass_response_cache"] = False


def _plan_for_acting_route(plan: Any, ctx: Any) -> bool:
    """Whether Main's plan names the forced call's acting role, model and locality.
    A local route's plan names provider ``local``, as ``gateway.settings._active_main_route`` resolves it."""
    from ouroboros.model_slots import acting_model_role
    from ouroboros.provider_models import provider_for_model

    model = str(getattr(ctx, "active_model", "") or "")
    local = bool(getattr(ctx, "active_use_local", False)) or provider_for_model(model) == "local"
    owner = getattr(getattr(ctx, "tools", None), "_ctx", None)
    role = acting_model_role(owner, model_role=getattr(ctx, "model_role", ""))
    return (plan is not None and str(getattr(plan, "model", "") or "") == model
            and (str(getattr(plan, "provider", "") or "") == "local") == local
            and (not role or getattr(plan, "model_role", "") == role))


def _bind_forced_route(ctx: Any) -> bool:
    """Fit the acting route before a forced candidate is priced or sent. A cold restore may have fitted a pending
    route for the next ordinary round, with another model, role, locality or effort; forced preparation does not
    consume that choice. On rebind failure, another route's plan stays unmeasured; a matching route still measures
    the acting effort (``_forced_physical_context``). True when fitted here; ordinary work that follows a priced
    candidate never sent refits its own round's route from the loop's plan (``loop_routes.bind_round_route``)."""
    from ouroboros.context_fit import effort_lengthens_request

    tools = getattr(ctx, "tools", None)
    tools_ctx = getattr(tools, "_ctx", None)
    plan = getattr(tools_ctx, "context_fit_plan", None)
    if plan is None or (_plan_for_acting_route(plan, ctx) and not effort_lengthens_request(
            getattr(plan, "provider", ""), getattr(plan, "start_reasoning_effort", ""), ctx.active_effort)):
        return False
    try:
        from ouroboros.model_slots import acting_model_role, task_model_binding
        from ouroboros.owner_wait import _conversation_account

        waiter = getattr(tools_ctx, "model_wait_context", None)
        role, account = task_model_binding({"model_role": acting_model_role(tools_ctx, model_role=getattr(ctx, "model_role", "")),
            "task_metadata": getattr(tools_ctx, "task_metadata", {})},
            context_fit_plan=plan, overrides=getattr(waiter, "overrides", None))
        _loop()._rebind_context_fit_plan(
            plan, tools, ctx.messages, model=ctx.active_model, use_local=ctx.active_use_local,
            tool_schemas=list(getattr(ctx, "tool_schemas", None) or []), model_role=role, credential_profile_id=account,
            model_route=_conversation_account(tools_ctx, ctx.messages, role, account), route_is_preference=True,
            start_mode=getattr(tools_ctx, "active_context_mode", None), reasoning_effort=ctx.active_effort)
    except Exception:
        logging.getLogger("ouroboros.loop").debug("Forced rail kept the restored plan of another route", exc_info=True)
        return False
    return True


def pending_runtime_route(ctx: Any, model: str, use_local: bool, effort: str) -> tuple:
    """Read the next ordinary call's one-shot choices without consuming them."""
    local = getattr(ctx, "active_use_local_override", None)
    return (getattr(ctx, "active_model_override", None) or model,
            use_local if local is None else local,
            normalize_reasoning_effort(getattr(ctx, "active_effort_override", None), default=effort))


def bind_round_route(tools: ToolRegistry, messages: List[Dict[str, Any]], route: tuple,
                     plan: Any, mode: str, preferred: str, tool_schemas: List[Dict[str, Any]], *,
                     llm: Any = None, pending_tool_budget: bool = False) -> tuple:
    """Bind the round's model/locality/effort and context mode; publish only a fresh route.

    Both round entry and owner mail use this boundary. Tool tails keep their route
    and pending tool overrides until a new call, unless owner mail explicitly moves it.
    """
    from ouroboros.lead_fact import publish_lead

    ctx = tools._ctx
    if not pending_tool_budget:
        model, local, effort, plan, mode = _apply_round_route_overrides(
            ctx, tools, messages, route, plan, mode, preferred, tool_schemas, llm=llm)
    else:
        model, local, effort = route
    ctx.active_model, ctx.active_use_local, ctx.active_effort = model, local, effort
    ctx.active_context_mode = mode
    if not pending_tool_budget:
        ctx.context_fit_plan = plan
        ctx._restored_context_model_role = None  # ordinary handover ends the saved actor
        publish_lead(ctx)
    return model, local, effort, plan, mode


def after_mail(limit: Any, plan: Any, mode: str, preferred: str, controls: dict) -> tuple:
    """Reapply a drained owner act before this very call, using the normal fit handover."""
    route = limit.active_model, limit.active_use_local, limit.active_effort
    if not controls.get("owner_model"):
        return *route, plan, mode
    model, local, effort, plan, mode = bind_round_route(
        limit.tools, limit.messages, route, plan, mode, preferred, limit.tool_schemas, llm=limit.llm)
    limit.active_model, limit.active_use_local, limit.active_effort = model, local, effort
    return model, local, effort, plan, mode


def _apply_round_route_overrides(ctx: Any, tools: ToolRegistry, messages: List[Dict[str, Any]], route: tuple,
                                 context_fit_plan: Any, active_context_mode: str, preferred_mode: str,
                                 tool_schemas: List[Dict[str, Any]], llm: Any = None) -> tuple:
    """Apply the round boundary's route: a following route's current role binding, then one-shot
    ``switch_model`` choices.

    ``route`` is ``(model, use_local, effort)``. A symbolic primary return also names
    the primary's role, so the rebound fit plan, account and processing binding are the
    primary's own (Auto stays Auto: no observed account becomes a pin); effort intent is
    untouched. The rebind fits the view in the effort this round sends, a switch's new one
    with its new model; an effort alone rebinds only where it can lengthen the request.
    Returns ``(model, use_local, effort, plan, mode)``.
    """
    from ouroboros import route_choice
    from ouroboros.context_fit import effort_lengthens_request

    selected = getattr(tools._ctx, "_authored_memory_plan", None)
    if selected is not None and selected.core_sha256 == getattr(context_fit_plan, "core_sha256", None):
        context_fit_plan, tools._ctx._authored_memory_plan = selected, None
    previous = route[:2]
    current = route_choice.round_route(ctx, llm, previous)  # a Settings change reaches a following route here
    if current is not None:
        ctx.active_model_override, ctx.active_use_local_override = current
        if not getattr(ctx, "active_role_override", None):  # the mind's own return says why; this is the host's
            route_choice.note_settings_change(ctx, previous[0], current[0], route_choice.follows(ctx.route_choice))
            ctx._pending_handover_reason = "settings_change"
    model, use_local, effort = _loop()._apply_runtime_overrides(ctx, *route)
    role, ctx.active_role_override = getattr(ctx, "active_role_override", None), None
    if role and getattr(ctx, "primary_route", None) and not getattr(ctx, "_owner_model_primary_changed", False):
        from ouroboros.primary_route_observation import record_primary_return_request
        record_primary_return_request(ctx, model, use_local, role)
    if ((model, use_local) != previous or role  # a view that still fits is never rewritten for an effort alone
            or effort_lengthens_request(str(getattr(context_fit_plan, "provider", "") or ""),
                getattr(context_fit_plan, "start_reasoning_effort", route[2]), effort)):
        context_fit_plan, active_context_mode = _loop()._rebind_context_fit_plan(
            context_fit_plan, tools, messages, model=model, use_local=use_local, preferred_mode=preferred_mode,
            start_mode=active_context_mode, tool_schemas=tool_schemas, reasoning_effort=effort,
            **({"model_role": role, "model_route": {}} if role else {}))
    if model != previous[0]:
        # Cross-FAMILY switch: discard provider-private reasoning
        # signatures before the new route sees them (same family is a no-op).
        sanitized = LLMClient.sanitize_reasoning_on_model_switch(messages, previous[0], model)
        if sanitized is not messages:
            messages[:] = sanitized
    if request_id := getattr(ctx, "_owner_model_request_id", None):
        ctx._owner_model_applied_id, ctx._owner_model_request_id = request_id, None
        ctx._owner_model_primary_changed = False
    return model, use_local, effort, context_fit_plan, active_context_mode
