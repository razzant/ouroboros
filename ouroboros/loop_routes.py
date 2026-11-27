"""The round boundary's route handover: live roles and explicit primary choices.

Owns the shared sanitize/fit rebind for a normal round and a post-mail owner act.
Binding and lead publication use the same boundary at both points. A tool-budget
tail binds the retained route without consuming overrides or publishing a fresh lead;
a drained owner act reapplies the durable primary before this round's call.
Call dispatch and fallback attempts remain in loop_model_call; a fresh chat's start
hint is armed, restored and spent at the seams below, which every route rebind and
round call reaches.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ouroboros.llm import LLMClient, normalize_reasoning_effort
from ouroboros.tools.registry import ToolRegistry


def _loop():
    """Resolve the shared route hooks through their existing loop bindings."""
    from ouroboros import loop

    return loop


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
        ctx._restored_native_load_window = None
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
    if model != previous[0]:
        # Price the destination's sanitized wire, not reasoning it cannot replay.
        messages[:] = LLMClient.sanitize_reasoning_on_model_switch(messages, previous[0], model)
    if ((model, use_local) != previous or role  # a view that still fits is never rewritten for an effort alone
            or effort_lengthens_request(str(getattr(context_fit_plan, "provider", "") or ""),
                getattr(context_fit_plan, "start_reasoning_effort", route[2]), effort)):
        context_fit_plan, active_context_mode = _loop()._rebind_context_fit_plan(
            context_fit_plan, tools, messages, model=model, use_local=use_local, preferred_mode=preferred_mode,
            start_mode=active_context_mode, tool_schemas=tool_schemas, reasoning_effort=effort,
            **({"model_role": role, "model_route": {}} if role else {}))
    if request_id := getattr(ctx, "_owner_model_request_id", None):
        ctx._owner_model_applied_id, ctx._owner_model_request_id = request_id, None
        ctx._owner_model_primary_changed = False
    return model, use_local, effort, context_fit_plan, active_context_mode


def _unsent_ordinary(plan: Any, tool_ctx: Any, messages: List[Dict[str, Any]]) -> Any:
    """``plan``'s ordinary form in place of its unsent start hint. An armed hint means no request of this work has
    gone, so the facts row its preparation minted for the hinted view (``loop_messages.append_context_facts``: its
    kind, text-bound hash and the line it records) never left either: it goes, and the next preparation mints its own.
    No sent history or owner/control row is touched."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, host_row_intact
    from ouroboros.context_view_facts import CONTEXT_FACTS_NAME

    usage = getattr(tool_ctx, "_accumulated_usage", None) or {}
    if line := usage.pop("_context_facts_line", None):
        messages[:] = [row for row in messages if not (row.get(HOST_CONTEXT_KIND_KEY) == CONTEXT_FACTS_NAME
                                                      and host_row_intact(row) and row.get("content") == line)]
        usage.pop("_context_facts_key", None)
    return plan.ordinary_first_contact()


def ordinary_view(plan: Any, tool_ctx: Any, messages: List[Dict[str, Any]]) -> Any:
    """``_unsent_ordinary`` applied where no route rebind follows (a send seam's rebound, a wrap-up): the view row of
    ``messages`` replaced, the task context holding the ordinary plan and its view facts."""
    from ouroboros.loop_model_call import _adopt_view_facts
    from ouroboros.transcript_prefix import rewrite_view

    plan = _unsent_ordinary(plan, tool_ctx, messages)
    messages[:] = plan.reproject_transcript(messages, plan.initial_mode)
    rewrite_view(tool_ctx, getattr(tool_ctx, "task_id", ""), "first_contact_rebound")
    tool_ctx.context_fit_plan, tool_ctx.active_context_mode, tool_ctx.messages = plan, plan.initial_mode, messages
    _adopt_view_facts(tool_ctx, plan, plan.initial_mode)
    return plan


def first_contact_route(plan: Any, start_mode: Optional[str], route: Dict[str, Any], tool_ctx: Any,
                        messages: List[Dict[str, Any]]) -> Tuple[Any, Optional[str]]:
    """The shared route rebind's start-hint seam (``loop_model_call._rebind_context_fit_plan``: a round's handover, a
    walk candidate, a wait's reprepare, the resend after an owner wait, a forced rail's route). An unsent hint stays
    armed while ``route`` still resolves to the scope its fit read, the send seam deciding a key, cap or dialect that
    moves later (``first_contact_guard``); a moved or unresolved route starts from the ordinary form
    (``_unsent_ordinary``), which the rebind then fits and shows. A sent view stays; either way the next observation
    must now bind the new plan's own route, not the original hint's."""
    if not getattr(plan, "first_contact", ""):
        return plan, start_mode
    from dataclasses import replace
    from ouroboros.context_window import first_contact_route_scope

    scope, _cap = first_contact_route_scope(route)
    if scope == plan.first_contact:
        return plan, start_mode
    if plan.first_contact_normal is not None:
        plan = _unsent_ordinary(plan, tool_ctx, messages)
        start_mode = plan.initial_mode
    return replace(plan, first_contact=scope), start_mode


class _FirstContactHint:
    """One round call's start-hint seam over every physical attempt of that call (``first_contact_guard``).

    It hears the call's dispatches (``physical_attempt_context.observe_dispatch``): the plan keeps its ordinary form
    until one of this call's attempts with the hinted view is dispatched (a reservation, a send refused before its
    bytes or another call's capture is no send), then keeps none (``ContextFitPlan.first_contact_spent``); an
    identity-bound release of the last such attempt arms it again. Until a send, ``hint(admit)`` refuses a hinted
    candidate the send seam signed for another scope (``context_window.first_contact_rebound``) and ``rebound`` names
    it for the caller's ordinary resend; after one, a later attempt keeps the view it showed."""

    def __init__(self, ctx: Any) -> None:
        self.ctx, self.rebound, self.sent, self.spent = ctx, False, set(), None

    def __call__(self, admit: Any) -> Any:
        def guarded(request: Any) -> Any:
            from ouroboros.context_window import first_contact_rebound

            self.rebound = not self.sent and first_contact_rebound(request.physical_context)
            return False if self.rebound else admit(request)
        return guarded

    def heard(self, capture: Any) -> None:
        from dataclasses import replace
        from ouroboros.physical_attempt_context import OBSERVED

        if self.ctx is None or not (
                getattr(getattr(capture, "physical_context", None), "first_contact_plan", None) or ("", False))[1]:
            return  # a call already returned, or no candidate a hint shaped: a helper's, an ordinary view's
        ctx, plan = self.ctx, self.ctx.context_fit_plan
        if capture.state in OBSERVED:
            self.sent.add(capture.attempt_id)
            if getattr(plan, "first_contact_normal", None) is not None:
                self.spent = (plan.first_contact_spent(), plan.first_contact_normal)
                ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = self.spent[0]
        elif capture.state == "released" and capture.attempt_id in self.sent:
            self.sent.discard(capture.attempt_id)
            if not self.sent and self.spent is not None and plan is self.spent[0]:  # proven unsent: armed again
                ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = replace(plan, first_contact_normal=self.spent[1])


@contextlib.contextmanager
def first_contact_guard(ctx: Any) -> Iterator[_FirstContactHint]:
    """One round call's start-hint seam (``_FirstContactHint``): a rebound send is swallowed for its ordinary resend."""
    from ouroboros.physical_attempt_context import observe_dispatch
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    hint = _FirstContactHint(ctx)
    with observe_dispatch(hint.heard):
        try:
            yield hint
        except PhysicalAttemptPreconditionFailed:
            if not hint.rebound:
                raise
        finally:
            hint.ctx = None  # a sender's late outcome, heard in its own thread, changes no later plan


def adopt_ordinary_first_contact(ctx: Any, disposition: Any) -> Any:
    """The rebound send's replacement: the same captured core's ordinary view (``ordinary_view``), measured again for
    its own send."""
    ctx.context_fit_plan = ordinary_view(ctx.context_fit_plan, ctx.tools._ctx, ctx.messages)
    ctx.active_context_mode = ctx.context_fit_plan.initial_mode
    return _loop()._measure_round_main_fit(ctx, automatic_pass_used=bool(getattr(disposition, "automatic_pass_used", False)))
