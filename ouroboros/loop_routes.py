"""The round boundary's route handover: live roles and explicit primary choices.

Owns the shared sanitize/fit rebind for a normal round and a post-mail owner act.
Call dispatch and fallback attempts remain in loop_model_call.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ouroboros.llm import LLMClient
from ouroboros.tools.registry import ToolRegistry


def _loop():
    """Resolve the shared route hooks through their existing loop bindings."""
    from ouroboros import loop

    return loop


def _apply_round_route_overrides(ctx: Any, tools: ToolRegistry, messages: List[Dict[str, Any]], route: tuple,
                                 context_fit_plan: Any, active_context_mode: str, preferred_mode: str,
                                 tool_schemas: List[Dict[str, Any]], llm: Any = None) -> tuple:
    """Apply the round boundary's route: a following route's current role binding, then one-shot
    ``switch_model`` choices.

    ``route`` is ``(model, use_local, effort)``. A symbolic primary return also names
    the primary's role, so the rebound fit plan, account and processing binding are the
    primary's own (Auto stays Auto: no observed account becomes a pin); effort intent is
    untouched. Returns ``(model, use_local, effort, plan, mode)``.
    """
    from ouroboros import route_choice
    from ouroboros.loop_model_call import _record_authoring_handover

    selected = getattr(tools._ctx, "_authored_memory_plan", None)
    if selected is not None and selected.core_sha256 == getattr(context_fit_plan, "core_sha256", None):
        context_fit_plan, tools._ctx._authored_memory_plan = selected, None
    previous = route[:2]
    current = route_choice.round_route(ctx, llm, previous)  # a Settings change reaches a following route here
    if current is not None:
        ctx.active_model_override, ctx.active_use_local_override = current
        if not getattr(ctx, "active_role_override", None):  # the mind's own return says why; this is the host's
            route_choice.note_settings_change(ctx, previous[0], current[0], route_choice.follows(ctx.route_choice))
            trace = getattr(ctx, "_execution_trace", None)
            _record_authoring_handover(ctx, from_model=previous[0], to_model=current[0], reason="settings_change",
                                       tool_calls_at_handover=len((trace or {}).get("tool_calls") or []) if isinstance(trace, dict) else 0)
    model, use_local, effort = _loop()._apply_runtime_overrides(ctx, *route)
    role, ctx.active_role_override = getattr(ctx, "active_role_override", None), None
    if role and getattr(ctx, "primary_route", None) and not getattr(ctx, "_owner_model_primary_changed", False):
        from ouroboros.primary_route_observation import record_primary_return_request
        record_primary_return_request(ctx, model, use_local, role)
    if (model, use_local) != previous or role:
        context_fit_plan, active_context_mode = _loop()._rebind_context_fit_plan(
            context_fit_plan, tools, messages, model=model, use_local=use_local, preferred_mode=preferred_mode,
            start_mode=active_context_mode, tool_schemas=tool_schemas,
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
