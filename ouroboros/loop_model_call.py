"""The per-round model call: context-fit identification, measurement and memory,
dispatch, the main-context reclaim, overflow-retry predicates, the cross-model
fallback chain and context-fit plan rebinding. Extracted from loop.py (v7 L-B
split); loop.py re-exports every name."""

from __future__ import annotations

from ouroboros.context_fit import _physical_context_for_fit

from ouroboros.config import runtime_setting

import logging
import contextlib
import copy
import json
import pathlib
import queue
import time

from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from ouroboros import task_pacing
from ouroboros import loop_output_exhaustion as output_recovery
from ouroboros.context_budget import WHOLE_CATEGORY
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import REBOUND_PHYSICAL_CONTEXT_KEY, REFUSED_CANDIDATE_KEY, TIMEOUT_REPEAT_KEY, TRANSPORT_DEATHS_KEY
from ouroboros.loop_round_limits import _coarsen_memory_view, _run_emergency_address_pass
from ouroboros.observability import new_execution_id
from ouroboros.tools.registry import ToolRegistry
from ouroboros.usage_accounting import PhysicalAttemptContext, PhysicalAttemptPreconditionFailed, invalidate_task_cache_splits


log = logging.getLogger("ouroboros.loop")
RESOURCE_REFUSAL_KEY = "resource_refusal"
_ROUND_WAIT_KINDS = ("transport_unavailable", "provider_outcome_unknown")


def _loop():
    """The parent loop module, read at call time.

    The loop's members stay monkeypatch-addressable at their historical
    ``ouroboros.loop`` bindings (tests rebind them there), so this leaf
    resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import loop

    return loop


def _record_authoring_handover(
    tool_ctx: Any,
    *,
    from_model: str,
    to_model: str,
    reason: str,
    tool_calls_at_handover: int,
) -> None:
    """Record a host-driven authoring handover for the current loop.

    A route change is not itself an error: the successor may continue normally.
    It is, however, a typed fact that the next tool-less response must see as a
    continuation when the predecessor already used tools.  Keep the fact on the
    existing ToolContext/usage/trace seams; no second custody store is needed.
    """
    source = str(from_model or "").strip()
    target = str(to_model or "").strip()
    if not source or not target or source == target or tool_calls_at_handover < 1:
        return
    row = {
        "from_model": source,
        "to_model": target,
        "reason": str(reason or "host_route_change"),
        "tool_calls_at_handover": int(tool_calls_at_handover),
        "recovery_prompted": False,
        "status": "pending",
    }
    tool_ctx._authoring_handover = row
    usage = getattr(tool_ctx, "_accumulated_usage", None)
    if isinstance(usage, dict):
        handovers = usage.setdefault("authoring_handovers", [])
        if isinstance(handovers, list):
            handovers.append(row)
    trace = getattr(tool_ctx, "_execution_trace", None)
    if isinstance(trace, dict):
        trace.setdefault("route_handovers", []).append(row)


def _pending_model_wait_handover(
    tool_ctx: Any, *, from_model: str, to_model: str, tool_calls: int,
) -> None:
    """Hold one distinct wait-switch until its re-prepared send succeeds."""
    if str(from_model or "") == str(to_model or ""):
        return
    pending = getattr(tool_ctx, "_pending_model_wait_handover", None)
    tool_ctx._pending_model_wait_handover = (
        pending[0] if pending else str(from_model or ""), str(to_model or ""),
        pending[2] if pending else int(tool_calls),
    )


def _adopt_fallback_route(
    ctx: Any,
    tools: ToolRegistry,
    fallback_model: str,
    fallback_use_local: bool,
    messages: List[Dict[str, Any]],
    fallback_messages: List[Dict[str, Any]],
    context_fit_plan: Any,
    active_context_mode: str,
    tool_schemas: List[Dict[str, Any]],
    accumulated_usage: Dict[str, Any],
    *,
    handover_from_model: str = "",
    handover_reason: str = "fallback",
    tool_calls_at_handover: Optional[int] = None,
) -> tuple:
    """Round-4 C1.1: adopt a SUCCESSFUL cross-family fallback as the active
    route for the rest of the loop. Otherwise a later round (esp. a tool
    loop) replays THIS fallback's reasoning/thinking back to the original
    primary family with no model-switch sanitizer firing (active_model never
    changed) — the cross-family signature replay, in reverse. Adopting the
    sanitized transcript keeps the old family's provider-private blocks off
    the switched route (a later switch_model/override re-triggers the
    round-start sanitizer); the caller already rebound the context-fit plan
    to this exact route, so adoption makes that tested projection canonical.
    Returns ``(active_model, active_use_local, context_fit_plan, context_mode)``."""
    ctx.active_model = fallback_model
    ctx.active_use_local = fallback_use_local
    messages[:] = fallback_messages
    trace = getattr(ctx, "_execution_trace", None)
    trace_calls = trace.get("tool_calls") if isinstance(trace, dict) else []
    _record_authoring_handover(
        ctx,
        from_model=handover_from_model,
        to_model=fallback_model,
        reason=handover_reason,
        tool_calls_at_handover=(
            len(trace_calls or [])
            if tool_calls_at_handover is None else int(tool_calls_at_handover)
        ),
    )
    if context_fit_plan is not None:
        tools._ctx.context_fit_plan = context_fit_plan
        tools._ctx.messages = messages
        tools._ctx.active_context_mode = active_context_mode
        # _call_round_model already recorded the accepted candidate's complete
        # same-basis fit facts. Do not replace them with a raw char estimate.
    return fallback_model, fallback_use_local, context_fit_plan, active_context_mode


def _route_fit_key(key: str) -> bool:
    """Route-fit state a failed candidate's walk must not leave behind. The conversation's per-model
    facts (anchors, accepted floors, refused sizes) and this round's own size refusal are not route
    state: a candidate's refusal still earns the round its renewal (review r2, 18.10)."""
    from ouroboros.context_anchor import ACCEPTED_INPUTS_KEY, INPUT_ANCHOR_KEY, REFUSED_INPUTS_KEY

    return key.startswith("_context_") and key not in {
        INPUT_ANCHOR_KEY, ACCEPTED_INPUTS_KEY, REFUSED_INPUTS_KEY, CONTEXT_REFUSAL_ROUND_KEY}


def _snapshot_context_fit_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in usage.items() if _route_fit_key(key)}


def _restore_context_fit_usage(
    usage: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> None:
    for key in tuple(usage):
        if _route_fit_key(key):
            usage.pop(key, None)
    usage.update(snapshot)


def _route_candidates(tool_ctx: Any, active_model: str, active_use_local: bool, acting_role: str,
                      waiter: Any) -> List[Tuple[str, str, bool, bool]]:
    """Configured routes other than the acting binding: ``(model, role, use_local, is_primary)``.

    The task's primary binding comes first whenever another route is acting (a failing
    adopted fallback may return to it; the owner never duplicates Main in the chain), then
    the ordered chain. Identity is the complete binding (``model_slots.route_binding``), so
    a chain entry with Main's model on another account is a real alternative.
    """
    from ouroboros.config import fallback_candidate_targets
    from ouroboros.model_slots import route_binding

    overrides = waiter.overrides if waiter is not None else None
    chain_local = runtime_setting("USE_LOCAL_FALLBACK", "").lower() in ("true", "1")
    primary = getattr(tool_ctx, "primary_route", None)
    rows = ([(primary["model"], primary["role"], bool(primary["use_local"]), True)]
            if isinstance(primary, dict) and primary.get("model") else [])
    rows += [(target.model_id, f"fallback:{index}", chain_local, False)
             for index, target in enumerate(fallback_candidate_targets(preserve_slots=True))]
    role = acting_role or (primary["role"] if rows and rows[0][3] and rows[0][0] == active_model else "main")
    seen, candidates = {route_binding(active_model, active_use_local, role, overrides=overrides)}, []
    for model, row_role, use_local, is_primary in rows:
        binding = route_binding(model, use_local, row_role, overrides=overrides)
        if binding not in seen:
            seen.add(binding)
            candidates.append((model, row_role, use_local, is_primary))
    return candidates


def _run_cross_model_fallback_chain(
    *, llm, ctx, tools, messages, active_model, active_use_local, tool_schemas,
    active_effort, max_retries, drive_logs, task_id, round_idx, event_queue,
    accumulated_usage, task_type, emit_progress, context_fit_plan,
    active_context_mode,
) -> tuple:
    """Try the configured routes other than the acting binding before any wait begins.

    An eligible unknown outcome moves on to the next route with facts-only recovery
    input (a real, possibly charged generation); an ineligible one (an accepted operation still readable)
    or a spent deadline stops the walk. A round whose own failure is an
    outage or an unknown outcome keeps that failure as its wait when none answers.
    """
    from ouroboros import fallback_cooldown as _fcd
    from ouroboros.loop_transport import append_unknown_recovery_input
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option, task_model_binding, route_binding
    from ouroboros.model_wait import current_model_wait
    from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS as _cooldown_kinds
    from ouroboros.primary_route_observation import failure_facts, route_facts_text
    from ouroboros.provider_models import provider_for_model

    def _cooled(model: str, use_local: bool, role: str) -> None:
        if str(accumulated_usage.get("_last_llm_error_kind") or "") in _cooldown_kinds:
            _fcd.mark_cooldown(*route_binding(model, use_local, role, overrides=waiter.overrides if waiter else None))

    def _disclose_unknown(target: List[Dict[str, Any]], route: str) -> None:
        # A NEW generation after an eligible unknown outcome names the unknown attempt it follows.
        if accumulated_usage.get("_pending_transport_outcome") or str(accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown":
            append_unknown_recovery_input(
                target, accumulated_usage, dict(accumulated_usage.get("_pending_transport_outcome") or {}),
                lead=f"The previous attempt ended without a usable answer; the configured route {route} continues.",
                continuation="configured_route")

    waiter = current_model_wait()
    _cooled(active_model, active_use_local, str(getattr(context_fit_plan, "model_role", "") or "main"))
    primary_context_usage = _snapshot_context_fit_usage(accumulated_usage)
    attempt_cap = _fcd.attempts_per_model()
    # The round's own outage or unknown outcome remains its wait if no route answers.
    entry_model, entry_kind = active_model, str(accumulated_usage.get("_last_llm_error_kind") or "")
    own_wait = entry_kind in _ROUND_WAIT_KINDS
    entry_failure = failure_facts(accumulated_usage)  # the failing attempt's own row, before any candidate adds one
    # A primary resource refusal returned here, not waited: the configured routes come
    # first, and the owner question is that refusal's own wait, asked after them.
    deferred, tools._ctx._deferred_resource_refusal = getattr(tools._ctx, "_deferred_resource_refusal", None), None
    owner_question = (task_type != "presence" and deferred is not None and not own_wait
                      and waiter is not None and waiter.waits_allowed)
    deferred_candidate = None  # The route that actually refused; never replay an unrelated primary.
    rows = _route_candidates(tools._ctx, active_model, active_use_local,
                             str(getattr(context_fit_plan, "model_role", "") or ""), waiter)
    tried = []
    recovering_output = output_recovery.ready_for_fallback(accumulated_usage)
    if recovering_output:
        rows = [row for row in rows if list(route_binding(row[0], row[2], row[1],
                overrides=waiter.overrides if waiter else None)) not in accumulated_usage[output_recovery.KEY]["tried"]]
    # The notice names the model that was actually just tried. `active_model`
    # stays the acting route until a candidate succeeds, so a second switch would
    # otherwise read "primary -> B" beside B's predecessor's failure reason.
    previous_model, previous_tag = active_model, " (local)" if active_use_local else ""
    msg = None
    # ABI-4: the chain ladder arrives as typed ResolvedModelTarget values; `.model_id`
    # is read once in `_route_candidates` and crosses to strings only at the LLM
    # transport boundary. Every chain row uses the single global USE_LOCAL_FALLBACK
    # flag (the pre-existing chain contract); the primary keeps its own locality.
    for index, (fallback_model, fallback_role, fallback_use_local, is_primary) in enumerate(rows):
        if _fcd.is_cooling_down(*route_binding(fallback_model, fallback_use_local, fallback_role,
                                             overrides=waiter.overrides if waiter else None)):
            continue
        deadline = _loop()._task_deadline_epoch(tools)
        if deadline and time.time() >= deadline:
            break
        ftag = " (local)" if fallback_use_local else ""
        # Name the account the dispatch will actually use: a task-local wait
        # override replaces the configured one for this role, or selects Auto,
        # so the same binding must speak here as at the send.
        _bound_role, bound_account = task_model_binding(
            {"model_role": fallback_role, "task_metadata": getattr(tools._ctx, "task_metadata", {})},
            overrides=waiter.overrides if waiter else None)
        fallback_account = (bound_account.strip() if bound_account is not None
                            else str(model_role_option(MODEL_ACCOUNTS_KEY, fallback_role) or ""))
        account_route = provider_for_model(fallback_model) == "claudexor"
        account_note = f"; account: {fallback_account or 'Auto'}" if account_route else ""
        reason = str(accumulated_usage.get("_last_llm_error_kind") or "")
        emit_progress(f"⚡ Fallback: {previous_model}{previous_tag} → {fallback_model}{ftag}"
                      f"{' (primary)' if is_primary else ''}{account_note}"
                      f"{f'; reason: {reason}' if reason else ''}"
                      f"{'; sign-in is still needed there' if reason == 'auth_error' else ''}"
                      f"{'; the earlier attempt’s outcome and cost stay unknown' if reason == 'provider_outcome_unknown' else ''}"
                      f"{'; pinned account: siblings were not tried' if account_route and fallback_account else ''}",
                      incident={"task_incident": "model_lane_switch", "toast_once": f"{task_id}:model_lane_switch:{round_idx}:{fallback_model}"})
        _disclose_unknown(messages, fallback_model)
        # Cross-FAMILY fallback must not replay the primary's
        # provider-private reasoning to a different family (the GLM->Claude
        # 400 "Invalid signature" death); the SSOT sanitizer no-ops same-family.
        fallback_messages = LLMClient.sanitize_reasoning_on_model_switch(messages, active_model, fallback_model)
        # Bind exact route evidence and choose its deterministic projection
        # BEFORE physical dispatch: the fallback's first request must not
        # inherit the failed primary route's Max projection/fingerprint. It
        # then uses the ordinary single confirmed-overflow Low retry path.
        candidate_plan, candidate_mode = _loop()._rebind_context_fit_plan(
            context_fit_plan,
            tools,
            fallback_messages,
            model=fallback_model,
            use_local=fallback_use_local,
            preferred_mode=str(
                getattr(context_fit_plan, "preferred_mode", "") or active_context_mode
            ),
            tool_schemas=tool_schemas,
            model_role=fallback_role,
            model_route={},
            start_mode=active_context_mode,  # the shown mode: a fallback or the primary may lower it, never raise it
        )
        candidate_call = _loop()._RoundModelCallContext(
                llm=llm,
                messages=fallback_messages,
                tools=tools,
                context_fit_plan=candidate_plan,
                active_model=fallback_model,
                tool_schemas=tool_schemas,
                active_effort=active_effort,
                max_retries=max_retries,
                drive_logs=drive_logs,
                task_id=task_id,
                round_idx=round_idx,
                event_queue=event_queue,
                accumulated_usage=accumulated_usage,
                task_type=task_type,
                active_use_local=fallback_use_local,
                active_context_mode=candidate_mode,
                drive_root=pathlib.Path(drive_logs).parent,
                attempt_cap=1 if recovering_output else attempt_cap,
                model_role=fallback_role,
                emit_progress=emit_progress,
                defer_resource_wait=(recovering_output or task_type == "presence" or owner_question
                                     or _route_follows(rows[index + 1:])),
                output_recovery_fallback=recovering_output,
            )
        if recovering_output and not output_recovery.fallback_untried(candidate_call):
            continue
        tried.append(fallback_model)
        resident = (None if tool_schemas is None else list(tool_schemas),
                    getattr(tools._ctx, "_route_left_out_tool_names", None))
        msg, _cost, candidate_mode = _loop()._call_round_model(candidate_call)
        if deferred is None and msg is None and not recovering_output:
            # Each fallback clears the transient context slot before its own send. Keep the
            # first actual resource refusal for this chain: a later bad request cannot
            # erase evidence that an allowed route refused before generation.
            deferred = getattr(tools._ctx, "_deferred_resource_refusal", None)
            if deferred is not None:
                deferred_candidate = candidate_call
            owner_question = (task_type != "presence" and deferred is not None and not own_wait
                              and waiter is not None and waiter.waits_allowed)
        if msg is not None or (not recovering_output and accumulated_usage.get("_last_llm_error_kind") == output_recovery.KIND
                               and not accumulated_usage.get("_pending_transport_outcome")):
            (
                active_model,
                active_use_local,
                context_fit_plan,
                active_context_mode,
            ) = _adopt_fallback_route(
                ctx,
                tools,
                candidate_call.active_model,
                candidate_call.active_use_local,
                messages,
                fallback_messages,
                candidate_call.context_fit_plan,
                candidate_mode,
                tool_schemas,
                accumulated_usage,
                # The predecessor author is the route whose round entered this
                # fallback chain.  ``previous_model`` may name a candidate that
                # failed before the successful candidate ever authored a turn.
                handover_from_model=active_model,
                handover_reason=reason or "fallback",
                tool_calls_at_handover=len(
                    ((getattr(tools._ctx, "_execution_trace", {})
                      if isinstance(getattr(tools._ctx, "_execution_trace", {}), dict) else {})
                     .get("tool_calls") or [])
                ),
            )
            tools._ctx._route_facts_pending = "" if is_primary else route_facts_text(
                tools._ctx, model=fallback_model, use_local=fallback_use_local, role=fallback_role,
                failed_model=entry_model, failure=entry_failure, waiter=waiter, usage=accumulated_usage,
                round_idx=round_idx)
            if not is_primary:
                tools._ctx.route_wait_on_primary = False  # a declared wait belonged to the primary route
            break
        tools._ctx.context_fit_plan = context_fit_plan
        tools._ctx.messages = messages
        tools._ctx.active_context_mode = active_context_mode
        _restore_context_fit_usage(accumulated_usage, primary_context_usage)
        if (resident[0] is not None and resident[0] != tool_schemas and fallback_messages is not messages
                and deferred_candidate is not candidate_call):
            # Its ceiling fit left with its transcript copy, notice included; the next route gets the list it had.
            # (A same-family candidate wrote its notice into the shared transcript, so its fit stays with it.)
            tool_schemas[:], tools._ctx._route_left_out_tool_names = resident
            invalidate_task_cache_splits(task_id)
        if _walk_fenced(tools._ctx, accumulated_usage) or (not recovering_output and str(
                accumulated_usage.get("_last_llm_error_kind") or "") == "llm_output_exhausted"):
            break  # an exhausted candidate answered: its outcome is the round's, no further route is dialed
        _cooled(fallback_model, fallback_use_local, fallback_role)
        previous_model, previous_tag = fallback_model, ftag
    fenced = msg is None and _walk_fenced(tools._ctx, accumulated_usage)
    if msg is None and tried and not fenced and str(accumulated_usage.get("_last_llm_error_kind") or "") != "llm_output_exhausted":
        accumulated_usage["_acting_route_failure"] = entry_failure  # its own failure decides its salvage; the last row stays the walk's
    if deferred is None:
        deferred = getattr(tools._ctx, "_deferred_resource_refusal", None)
    owner_question = (not recovering_output and task_type != "presence" and deferred is not None and not own_wait
                      and waiter is not None and waiter.waits_allowed)
    accumulated_usage.pop(RESOURCE_REFUSAL_KEY, None)
    if msg is None and owner_question and not fenced:
        # Only the refused route is eligible to re-send after the owner wait;
        # the primary might have failed permanently before a fallback's quota refusal.
        deferred.ask_owner(waiter)
        retry_call = deferred_candidate or _loop()._RoundModelCallContext(
            llm=llm, messages=messages, tools=tools, context_fit_plan=context_fit_plan,
            active_model=active_model, tool_schemas=tool_schemas, active_effort=active_effort,
            max_retries=max_retries, drive_logs=drive_logs, task_id=task_id, round_idx=round_idx,
            event_queue=event_queue, accumulated_usage=accumulated_usage, task_type=task_type,
            active_use_local=active_use_local, active_context_mode=active_context_mode,
            drive_root=pathlib.Path(drive_logs).parent, emit_progress=emit_progress, defer_resource_wait=False)
        retry_call.defer_resource_wait = False
        _disclose_unknown(retry_call.messages, retry_call.active_model)
        # An owner may select a DIFFERENT account on either the primary or a
        # fallback model. Rebind the retained call before measurement and physical
        # send; the pre-wait account's fingerprint/capacity is not evidence for B.
        role, account = task_model_binding(
            {"model_role": retry_call.model_role,
             "task_metadata": getattr(tools._ctx, "task_metadata", {})},
            context_fit_plan=retry_call.context_fit_plan, overrides=waiter.overrides)
        retry_call.context_fit_plan, retry_call.active_context_mode = _loop()._rebind_context_fit_plan(
            retry_call.context_fit_plan, tools, retry_call.messages,
            model=retry_call.active_model, use_local=retry_call.active_use_local,
            start_mode=retry_call.active_context_mode, tool_schemas=tool_schemas,
            model_role=role, model_route={}, credential_profile_id=account)
        msg, _cost, active_context_mode = _loop()._call_round_model(retry_call)
        if msg is not None and deferred_candidate is not None:
            active_model, active_use_local, context_fit_plan, active_context_mode = _adopt_fallback_route(
                ctx, tools, retry_call.active_model, retry_call.active_use_local,
                messages, retry_call.messages, retry_call.context_fit_plan, active_context_mode,
                tool_schemas, accumulated_usage, handover_from_model=active_model,
                handover_reason="owner_wait")
            tools._ctx._route_facts_pending = route_facts_text(
                tools._ctx, model=active_model, use_local=active_use_local, role=role, failed_model=entry_model,
                failure=entry_failure, waiter=waiter, usage=accumulated_usage, round_idx=round_idx)
        else:
            active_model, active_use_local = retry_call.active_model, retry_call.active_use_local
            context_fit_plan = retry_call.context_fit_plan
    elif msg is None and deferred is not None and not own_wait:  # the refused primary buys no forced final either
        accumulated_usage[RESOURCE_REFUSAL_KEY] = deferred.terminal(
            fallbacks_tried=tried, owner_wait="not_asked" if owner_question else "not_allowed")
    return (
        msg,
        active_model,
        active_use_local,
        context_fit_plan,
        active_context_mode,
    )


def _recover_failed_round(limit_ctx: Any, tools: ToolRegistry, msg: Any, episode: Any, *,
                          context_fit_plan: Any, active_context_mode: str, emit_progress: Callable[..., None]) -> tuple:
    """The recovery order of one round's outcome.

    Each observed failure can use configured alternatives, including during an existing
    outage. That episode keeps its elapsed clock, backoff and prior custody; a failed
    alternative never resets those bounds. Returns ``(msg, model, use_local, plan, mode, episode)``.
    """
    from ouroboros.model_wait import current_model_wait

    usage, ctx = limit_ctx.accumulated_usage, tools._ctx
    model, use_local = limit_ctx.active_model, limit_ctx.active_use_local

    def reconcile(current: Any) -> Any:
        return _loop()._reconcile_transport_wait(
            current, ctx, msg_present=msg is not None, error_kind=str(usage.get("_last_llm_error_kind") or ""),
            drive_logs=limit_ctx.drive_logs, task_id=limit_ctx.task_id, model=model, emit_progress=emit_progress)

    if episode is not None:
        episode = reconcile(episode)
        if msg is not None:
            return msg, model, use_local, context_fit_plan, active_context_mode, episode
    kind, pending = str(usage.get("_last_llm_error_kind") or ""), usage.get("_pending_transport_outcome")
    if (msg is None and _loop()._fallback_chain_allowed(ctx, kind, None, usage) and _route_candidates(
            ctx, model, use_local, str(getattr(context_fit_plan, "model_role", "") or ""), current_model_wait())):
        msg, model, use_local, context_fit_plan, active_context_mode = _loop()._run_cross_model_fallback_chain(
            llm=limit_ctx.llm, ctx=ctx, tools=tools, messages=limit_ctx.messages, active_model=model,
            active_use_local=use_local, tool_schemas=limit_ctx.tool_schemas, active_effort=limit_ctx.active_effort,
            max_retries=limit_ctx.max_retries, drive_logs=limit_ctx.drive_logs, task_id=limit_ctx.task_id,
            round_idx=limit_ctx.round_idx, event_queue=limit_ctx.event_queue, accumulated_usage=usage,
            task_type=limit_ctx.task_type, emit_progress=emit_progress, context_fit_plan=context_fit_plan,
            active_context_mode=active_context_mode)
        outstanding = usage.get("_pending_transport_outcome") or pending
        exhausted = usage.get("_last_llm_error_kind") == "llm_output_exhausted"
        if (msg is None and not _walk_fenced(ctx, usage) and (outstanding or not exhausted)
                and (kind in _ROUND_WAIT_KINDS or usage.get("_pending_transport_outcome"))):
            # The round's own outage or unknown outcome owns its wait: a later candidate's
            # failure never re-aims it (nor the probe's expected route). A candidate that answered
            # but spent its reply allowance keeps that kind when no attempt is outstanding (the
            # loop's next round reads it); otherwise its host fact waits for the continuation.
            if exhausted:
                _loop()._append_or_merge_user_message(
                    limit_ctx.messages, _loop()._output_exhausted_notice(usage.get("_last_llm_output_exhausted")))
            usage["_last_llm_error_kind"] = "provider_outcome_unknown" if outstanding else kind
            if outstanding:
                usage["_pending_transport_outcome"] = outstanding
    return msg, model, use_local, context_fit_plan, active_context_mode, reconcile(episode)


def _walk_fenced(tool_ctx: Any, usage: Dict[str, Any]) -> bool:
    """A spent deadline, or an unknown outcome that permits no new generation, ends the walk."""
    from ouroboros.loop_transport import new_generation_after_unknown

    kind = str(usage.get("_last_llm_error_kind") or "")
    return kind == "deadline_exhausted" or (
        kind == "provider_outcome_unknown" and not new_generation_after_unknown(tool_ctx, usage))


def _apply_round_route_overrides(ctx: Any, tools: ToolRegistry, messages: List[Dict[str, Any]], route: tuple,
                                 context_fit_plan: Any, active_context_mode: str, preferred_mode: str,
                                 tool_schemas: List[Dict[str, Any]]) -> tuple:
    """Apply one-shot ``switch_model`` choices at the round boundary.

    ``route`` is ``(model, use_local, effort)``. A symbolic primary return also names
    the primary's role, so the rebound fit plan, account and processing binding are the
    primary's own (Auto stays Auto: no observed account becomes a pin); effort intent is
    untouched. Returns ``(model, use_local, effort, plan, mode)``.
    """
    selected = getattr(tools._ctx, "_authored_memory_plan", None)
    if selected is not None and selected.core_sha256 == getattr(context_fit_plan, "core_sha256", None):
        context_fit_plan, tools._ctx._authored_memory_plan = selected, None
    previous = route[:2]
    model, use_local, effort = _loop()._apply_runtime_overrides(ctx, *route)
    role, ctx.active_role_override = getattr(ctx, "active_role_override", None), None
    if role and getattr(ctx, "primary_route", None):
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
    return model, use_local, effort, context_fit_plan, active_context_mode


def _rebind_context_fit_plan(
    plan: Any,
    tools: ToolRegistry,
    messages: List[Dict[str, Any]],
    *,
    model: str,
    use_local: bool,
    preferred_mode: Optional[str] = None,  # the owner's mode; None keeps the plan's
    tool_schemas: List[Dict[str, Any]],
    model_role: str = "",
    model_route: Optional[Dict[str, Any]] = None,
    credential_profile_id: Optional[str] = None,
    start_mode: Optional[str] = None,  # the mode the task runs in: a new route may lower it, never raise it
) -> Tuple[Any, str]:
    if plan is None or not all(
        hasattr(plan, name) for name in ("max_projection", "low_projection", "core_sha256")
    ):
        raise RuntimeError(
            "CONTEXT_FIT_REBUILD_FAILED: immutable context core is unavailable for route switch"
        )
    from ouroboros.context import _context_fit_route
    from ouroboros.context_anchor import main_input_anchor
    from ouroboros.context_fit import (_failed_route_evidence, _route_calibration_ratio,
                                       estimate_context_prompt_tokens, main_output_reserve_tokens)
    from ouroboros.provider_models import parse_claudexor_model

    metadata = getattr(tools._ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    task = {
        "model": model,
        "use_local_model": use_local,
        "task_metadata": metadata,
        "delegation_role": metadata.get("delegation_role"),
        "model_role": model_role or getattr(plan, "model_role", "main"),
        "model_route": model_route if model_route is not None else getattr(plan, "model_route", {}),
        "credential_profile_id": credential_profile_id,
    }
    is_subagent = str(metadata.get("delegation_role") or "").lower() == "subagent"
    try:
        route, evidence = _context_fit_route(task, allow_fetch=not is_subagent)
    except Exception:
        log.debug("Route-switch capability probe failed; preserving unknown Max", exc_info=True)
        route, evidence = _failed_route_evidence(task)
    # Account/server changes have their own anchors and durable window observations.
    anchor, _accepted = main_input_anchor(getattr(tools._ctx, "_accumulated_usage", None), SimpleNamespace(
        model=str(route.get("model") or model), provider=str(route.get("provider") or ""),
        route_fp=route.get("window_observation_fp", evidence.route_fp)))
    ratio = anchor.density if anchor is not None else _route_calibration_ratio(
        None,  # canonical evidence root (one observation store)
        str(getattr(evidence, "route_fp", "") or ""),
        str(route.get("model") or model),
        estimate_context_prompt_tokens(messages, tool_schemas, provider=str(route.get("provider") or "")),
        view_kind=plan.view_kind,
    )
    from ouroboros.context_window import estimator_ratio, resolve_window, route_window_kwargs
    output_reserve = main_output_reserve_tokens(use_local=bool(route.get("use_local", use_local)), evidence=evidence)
    scope = route.get("window_observation_fp", evidence.route_fp)
    bound = bool(scope) and (route.get("provider") != "claudexor" or bool((route.get("options") or {}).get("account_fingerprint")))
    window = resolve_window(evidence, output_reserve, bound=bound, observation_fp=scope)
    window_tokens, known_window = window.total_equivalent, window.usable is not None
    preferred = preferred_mode or getattr(plan, "preferred_mode", "")
    preferred = preferred if preferred in {"low", "max", "nano"} else "max"
    rebound = replace(
        plan,
        preferred_mode=preferred,
        model=str(route.get("model") or model),
        provider=str(route.get("provider") or ""),
        route_fp=str(getattr(evidence, "route_fp", "") or ""),
        status=str(getattr(evidence, "status", "") or ""),
        stale=bool(getattr(evidence, "stale", False)),
        model_role=task["model_role"],
        model_route={
            "source": str(getattr(evidence, "source_id", "") or ""),
            "model": parse_claudexor_model(str(route.get("model") or model))[1],
            "credentialProfileId": str(getattr(evidence, "credential_profile_id", "") or ""),
            "accountFingerprint": str(getattr(evidence, "account_fingerprint", "") or ""),
        } if route.get("provider") == "claudexor" else {},
        evidence_source=str(getattr(evidence, "source", "") or ""),
        **route_window_kwargs(evidence),
        window_record=window, window_witness_bound=bound, route_base_url=str(route.get("base_url") or ""),
    ).reproject_for_route(  # the shown view kept while it fits this route's window, else narrowed
        window_tokens=window_tokens, known_window=known_window, ratio=estimator_ratio(evidence, ratio),
        output_reserve=output_reserve, tool_schemas=tool_schemas, start_mode=start_mode, current_messages=messages)
    mode = rebound.initial_mode
    projected_prompt_tokens = rebound.projected_tokens_with_tools(mode, tool_schemas)
    shown = messages[0] if messages else None
    messages[:] = rebound.reproject_transcript(messages, mode)
    invalidate_task_cache_splits(getattr(tools._ctx, "task_id", ""))
    tools._ctx.context_fit_plan = rebound
    tools._ctx.messages = messages
    tools._ctx.active_context_mode = mode
    _adopt_view_facts(tools._ctx, rebound, mode)
    try:
        if mode != start_mode:  # a task that already ran in this mode was told so before
            _emit_physical_mode(getattr(tools._ctx, "event_queue", None), str(getattr(tools._ctx, "task_id", "") or ""),
                                tools._ctx.drive_logs(), rebound, mode)
        _loop()._emit_checkpoint_event(
            getattr(tools._ctx, "event_queue", None),
            str(getattr(tools._ctx, "task_id", "") or ""),
            tools._ctx.drive_logs(),
            {
                "checkpoint_kind": "context_fit_route_rebound",
                "model": rebound.model,
                "route_fp": rebound.route_fp,
                "core_sha256": rebound.core_sha256,
                "preferred_mode": preferred,
                "effective_mode": mode,
                "evidence_status": rebound.status,
                "window_tokens": rebound.window_tokens,
                "known_window": known_window,  # a floored window keeps its number, not its authority
                "projected_prompt_tokens": projected_prompt_tokens,
                "view_kept": messages[0] == shown,  # the system view sent before this rebind, byte for byte
            },
        )
    except Exception:
        log.debug("Failed to emit route-switch context-fit checkpoint", exc_info=True)
    return rebound, mode


def _adopt_view_facts(tool_ctx: Any, plan: Any, mode: str) -> None:
    """The view fact of the projection now sent: on the task context and in the task trace."""
    receipt = dict(getattr(plan.projection(mode), "memory_facts", None) or {})
    if not receipt:
        return
    from ouroboros.memory_floor import trace_facts
    from ouroboros.memory_inventory import VIEW_TRACE_KEY

    tool_ctx.memory_view_facts = trace_facts(receipt)
    trace = getattr(tool_ctx, "_execution_trace", None)
    if isinstance(trace, dict):
        trace[VIEW_TRACE_KEY] = dict(tool_ctx.memory_view_facts)


def _emit_physical_mode(event_queue: Any, task_id: str, drive_logs: Any, plan: Any, mode: str) -> None:
    """The known window lowered this plan's starting mode (no shorter view fit): one owner-visible checkpoint.

    The sent projection's own fact says so (``mode_switch``); a lower mode a task kept from before
    (task-local Low after an overflow, an earlier route's choice) is not this window's doing.
    """
    projection = plan.projection(mode) if hasattr(plan, "projection") else None
    switch = ((getattr(projection, "memory_facts", None) or {}).get("floor") or {}).get("mode_switch")
    if not switch or mode != str(getattr(plan, "initial_mode", "") or ""):
        return
    _loop()._emit_checkpoint_event(event_queue, task_id, drive_logs, {
        "checkpoint_kind": "context_fit_physical_mode",
        "route_fp": str(getattr(plan, "route_fp", "") or ""),
        "preferred_mode": str(switch.get("from") or ""),
        "effective_mode": mode,
        "window_tokens": int(getattr(plan, "window_tokens", 0) or 0),
        "owner_visible": True,
    })


@dataclass
class _RoundModelCallContext:
    llm: LLMClient
    messages: List[Dict[str, Any]]
    tools: ToolRegistry
    context_fit_plan: Any
    active_model: str
    tool_schemas: List[Dict[str, Any]]
    active_effort: str
    max_retries: int
    drive_logs: pathlib.Path
    task_id: str
    round_idx: int
    event_queue: Optional[queue.Queue]
    accumulated_usage: Dict[str, Any]
    task_type: str
    active_use_local: bool
    active_context_mode: str
    drive_root: Optional[pathlib.Path]
    attempt_cap: Optional[int] = None
    model_role: str = ""
    # The loop-level owner notifier (run_llm_loop's own parameter), the one
    # callable documented to accept incident=; the ToolContext ABI's
    # emit_progress_fn takes a single argument and must not carry the pair.
    emit_progress: Optional[Callable[..., None]] = None
    # None: a primary round, which defers a resource refusal while a configured route
    # follows or the turn may not wait. The chain sets it for each candidate.
    defer_resource_wait: Optional[bool] = None
    output_recovery_fallback: bool = False
    physical_send_claim: Optional[Callable[[Any], None]] = None


def _route_follows(candidates: List[Tuple[str, str, bool, bool]]) -> bool:
    from ouroboros import fallback_cooldown

    from ouroboros.model_slots import route_binding
    from ouroboros.model_wait import current_model_wait

    waiter = current_model_wait()
    return any(not fallback_cooldown.is_cooling_down(*route_binding(
        model, use_local, role, overrides=waiter.overrides if waiter else None))
        for model, role, use_local, _primary in candidates)


def _fallback_route_follows(ctx: _RoundModelCallContext) -> bool:
    """Whether this round's configured-route walk would still dial a route.

    A declared wait on the primary keeps its resource refusal on the primary's own
    visible wait instead of deferring it to paid alternatives.
    """
    from ouroboros.model_wait import current_model_wait

    tool_ctx = ctx.tools._ctx
    if (bool(getattr(tool_ctx, "exact_model_route", False)) or getattr(tool_ctx, "route_wait_on_primary", False)
            or isinstance(ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)):
        return False
    return _route_follows(_route_candidates(
        tool_ctx, ctx.active_model, ctx.active_use_local,
        str(getattr(ctx.context_fit_plan, "model_role", "") or ctx.model_role or ""), current_model_wait()))


def _context_fit_round_id(ctx: _RoundModelCallContext) -> str:
    execution_id = str(ctx.accumulated_usage.setdefault("execution_id", new_execution_id()))
    return f"{execution_id}:round:{ctx.round_idx}"


def _main_context_profile(plan: Any, rendered_mode: str) -> str:
    if rendered_mode == "nano":  # the owner's Nano carries its target; a Nano the window chose, the window alone
        return "owner_nano" if str(getattr(plan, "preferred_mode", "")) == "nano" else "task_local_nano"
    if rendered_mode != "low":
        return "owner_max"
    # Effective Low is the sizing authority even when a bare env override
    # keeps owner intent Max for P3. A Low entered after a real Max overflow
    # is task-local and does not inherit the economy target T.
    return "owner_low" if str(getattr(plan, "preferred_mode", "")) == "low" else "task_local_low"


def _remember_main_fit(ctx: _RoundModelCallContext, disposition: Any) -> None:
    measurement = disposition.measurement
    usage = ctx.accumulated_usage
    usage["_context_route_fp"] = measurement.route_fp
    usage["_context_prompt_estimate"] = measurement.estimated_input_tokens
    usage["_context_fit_mode"] = measurement.rendered_mode
    usage["_context_profile"] = measurement.profile
    usage["_context_measurement_basis"] = measurement.measurement_basis
    usage["_context_measurement_density"] = measurement.measurement_density
    usage["_context_raw_input_tokens"] = measurement.raw_input_tokens
    usage["_context_reply_allowance_tokens"] = measurement.reply_allowance_tokens
    usage["_context_response_reserve_tokens"] = measurement.response_reserve_tokens  # the reply floor fit counts
    usage["_context_target_total_tokens"] = measurement.target_total_tokens
    usage["_context_capacity_total_tokens"] = measurement.capacity_total_tokens
    usage["_context_target_deficit_tokens"] = measurement.target_deficit_tokens
    usage["_context_capacity_deficit_tokens"] = measurement.capacity_deficit_tokens
    usage["_context_reclaim_goal_tokens"] = measurement.reclaim_goal_tokens
    usage["_context_target_miss"] = disposition.action == "send_target_miss"
    usage["_context_automatic_pass_used"] = disposition.automatic_pass_used
    usage["_context_predicted_capacity_miss"] = disposition.predicted_capacity_miss
    for name in ("anchor_input_tokens", "anchor_round", "anchor_rewritten", "accepted_input_tokens", "window_reading_tokens", "window"):
        usage[f"_context_{name}"] = getattr(measurement, name)  # the estimate's provenance and the window floor


def _measure_round_main_fit(
    ctx: _RoundModelCallContext,
    *,
    automatic_pass_used: bool,
) -> Any:
    plan = ctx.context_fit_plan
    if plan is None or str(ctx.active_model or "") != str(getattr(plan, "model", "") or ""):
        return None
    from ouroboros.context_anchor import main_input_anchor
    from ouroboros.context_fit import measure_main_fit
    from ouroboros.transcript_prefix import rewrite_epoch

    rendered_mode = str(ctx.active_context_mode) if str(ctx.active_context_mode) in {"max", "low", "nano"} else "max"
    anchor, accepted = main_input_anchor(ctx.accumulated_usage, plan, rewrite_epoch=rewrite_epoch(ctx.tools._ctx))
    disposition = measure_main_fit(
        plan,
        ctx.messages,
        ctx.tool_schemas,
        profile=_main_context_profile(plan, rendered_mode),
        rendered_mode=rendered_mode,
        round_id=_context_fit_round_id(ctx),
        automatic_pass_used=automatic_pass_used,
        reasoning_effort=ctx.active_effort,
        anchor=anchor,
        accepted_input_tokens=accepted,
    )
    _remember_main_fit(ctx, disposition)
    return disposition



def _measure_main_context_view(plan, messages, schemas, mode, effort, round_id, usage=None) -> dict:
    """Measure an authored candidate without changing Main's dispatch policy.

    Predictions inform the next ordinary reclaim/send; neither capacity nor
    economy estimates veto a useful authored view. No route probe or model runs.
    ``usage`` (the task's) supplies this conversation's own anchor; a candidate
    view is a rewrite, so its estimate is labelled as one.
    """
    from ouroboros.context_anchor import main_input_anchor
    from ouroboros.context_fit import estimate_context_prompt_tokens, measure_main_fit
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS

    if plan is None:
        return {"accepted": True, "strict_bound_proven": False,
                "estimated_input_tokens": estimate_context_prompt_tokens(messages, schemas, reasoning_effort=effort),
                "response_reserve_tokens": MAIN_LOOP_MAX_TOKENS, "capacity_total_tokens": None,
                "measurement_basis": "cold_estimate", "reason": "main_route_capacity_unknown"}
    anchor, accepted = main_input_anchor(usage, plan)
    disposition = measure_main_fit(
        plan, messages, schemas, profile=_main_context_profile(plan, mode),
        rendered_mode=mode, round_id=round_id, reasoning_effort=effort,
        anchor=anchor, accepted_input_tokens=accepted,
    )
    return {"accepted": True, "strict_bound_proven": False,
            "predicted_capacity_miss": disposition.predicted_capacity_miss,
            **asdict(disposition.measurement)}


@contextlib.contextmanager
def _presend_context_fit(ctx: _RoundModelCallContext | None = None):
    """Use the final physical candidate for the existing strict-shrink recovery, without a paid send."""
    from ouroboros.llm_attempt import main_context_fit_refusal

    refused = None

    def admit(request):
        nonlocal refused
        refused = main_context_fit_refusal(request)
        return refused is None

    try:
        yield admit
    except PhysicalAttemptPreconditionFailed:
        if refused is None or ctx is None:
            raise  # a failed strict-shrink or custody check is not a size prediction
        ctx.accumulated_usage.update({
            REFUSED_CANDIDATE_KEY: refused, "_last_llm_error_kind": "context_overflow", "reason_code": "context_overflow",
            "_last_llm_error": "No result room in the final candidate; pre-send size recovery.",
            "_last_llm_provider_message": "", "_last_llm_status_code": None, "_last_llm_provider_code": "",
            "_last_llm_provider_fields": {}, "_last_llm_provider_message_cut": False,
        })
        ctx.accumulated_usage["_last_llm_failure"] = {"model": ctx.active_model, "failure_code": "context_overflow",
                                                    "error": ctx.accumulated_usage["_last_llm_error"]}
        _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
            "checkpoint_kind": "context_fit_refused", "round": ctx.round_idx, **refused,
        })


def _dispatch_round_model(
    ctx: _RoundModelCallContext,
    disposition: Any,
    *,
    attempt_cap: Optional[int],
    candidate_predicate: Optional[Callable[[Any], Any]] = None,
    max_tokens: Optional[int] = None,  # the strict-shrink retry's ceiling: the failed attempt's sent allowance
) -> Tuple[Any, float]:
    from ouroboros.model_wait import current_model_wait
    from ouroboros.loop_transport import emit_model_substitution, transport_repeat_stop_requested
    from ouroboros.owner_mailbox import OwnerMailboxPeek
    from ouroboros.send_clock import main_clock_policy
    from ouroboros.subagent_bootstrap import is_configured_api_child
    from ouroboros.loop_messages import append_context_facts

    _loop()._retire_main_reasoning(ctx)
    if append_context_facts(ctx):
        disposition = _loop()._measure_round_main_fit(
            ctx, automatic_pass_used=bool(getattr(disposition, "automatic_pass_used", False)))
    mailbox_peek = OwnerMailboxPeek()
    ctx.tools._ctx._transport_repeat_control_reason = ""

    waiter = current_model_wait()
    plan = getattr(ctx, "context_fit_plan", None) or getattr(ctx.tools._ctx, "context_fit_plan", None)
    from ouroboros.model_slots import task_model_binding, task_processing_preference
    role, account = task_model_binding({
        "model_role": getattr(ctx, "model_role", ""),
        "task_metadata": getattr(ctx.tools._ctx, "task_metadata", {})},
        context_fit_plan=plan, overrides=waiter.overrides if waiter else None)
    primary = getattr(ctx, "defer_resource_wait", None) is None
    if primary:
        ctx.tools._ctx._deferred_resource_refusal = None
        ctx.accumulated_usage.pop(RESOURCE_REFUSAL_KEY, None)
    elif ctx.task_type == "presence":
        ctx.tools._ctx._deferred_resource_refusal = None
    previous_call = ctx.accumulated_usage.get("_last_llm_call_meta")
    ctx.tools._ctx._usable_main_capture = None
    from ouroboros.loop_delivery import completion_observation
    trace = getattr(ctx.tools._ctx, "_execution_trace", None) or {}
    ctx.tools._ctx._completion_observation = completion_observation(ctx.tools._ctx, trace)

    result = (None, 0.0)
    with contextlib.ExitStack() as binding, _presend_context_fit(ctx) as admit:
        deferral = None
        if waiter is not None:
            from ouroboros.loop_llm_call import classify_llm_exception
            resource_reason = (lambda error: {"auth_error": "auth", "quota_exhausted": "quota"}.get(
                classify_llm_exception(error).kind, "")) if getattr(ctx.tools._ctx, "route_wait_on_primary", False) else None
            binding.enter_context(waiter.register_reprepare(
                role, lambda kwargs: _reprepare_waiting_main(ctx, kwargs), resource_reason=resource_reason))
            if ((not waiter.waits_allowed or _fallback_route_follows(ctx)) if primary else ctx.defer_resource_wait):
                deferral = binding.enter_context(waiter.defer_resource_wait(role))
        result = _loop().call_llm_with_retry(
            ctx.llm, ctx.messages, ctx.active_model, ctx.tool_schemas,
            ctx.active_effort, ctx.max_retries, ctx.drive_logs, ctx.task_id,
            ctx.round_idx, ctx.event_queue, ctx.accumulated_usage, ctx.task_type,
            use_local=ctx.active_use_local,
            deadline_ts=_loop()._task_deadline_epoch(ctx.tools),
            transport_reserve_sec=task_pacing.get_finalization_grace_sec(),
            attempt_cap=attempt_cap,
            stop_retry_check=(lambda: transport_repeat_stop_requested(ctx.tools._ctx, mailbox_peek=mailbox_peek)) if attempt_cap is None else None,
            allow_server_web_search=_loop()._server_web_allowed_by_task(ctx.tools._ctx),
            physical_context=(_physical_context_for_fit(disposition) if disposition is not None else None),
            candidate_predicate=candidate_predicate, physical_fit_predicate=admit, model_role=role, model_account_override=account,
            processing_preference=task_processing_preference(
                {"task_metadata": getattr(ctx.tools._ctx, "task_metadata", {})}, model_role=role),
            # The loop's own active-turn slot: a reprepared send keeps this exact
            # owner because the slot survives kwargs deep-copying by identity.
            model_turn_state=getattr(ctx.tools._ctx, "model_turn_state", None),
            send_clock_policy=main_clock_policy(
                getattr(ctx.tools._ctx, "task_metadata", {}), task_type=ctx.task_type),
            configured_api_child=is_configured_api_child(ctx.tools._ctx),
            physical_send_claim=ctx.physical_send_claim,
            **({"max_tokens": int(max_tokens)} if max_tokens else {}),
        )
    ctx.accumulated_usage.pop(REBOUND_PHYSICAL_CONTEXT_KEY, None)  # consumed by the call's later attempts, if any
    if result[0] is not None:  # an accepted request is this conversation's next size anchor
        from ouroboros.context_anchor import record_input_anchor
        from ouroboros.transcript_prefix import rewrite_epoch
        ctx.accumulated_usage.pop(CONTEXT_REFUSAL_ROUND_KEY, None)  # an answer ends the round's refusal (review r2)
        record_input_anchor(ctx.accumulated_usage, ctx.context_fit_plan, round_idx=ctx.round_idx,
                            rewrite_epoch=rewrite_epoch(ctx.tools._ctx))
    capture = _loop().last_physical_attempt_capture()
    if primary and deferral is not None and deferral.fact and result[0] is None:
        ctx.tools._ctx._deferred_resource_refusal = deferral
        if not waiter.waits_allowed:  # typed at once: the terminal may come before any chain
            ctx.accumulated_usage[RESOURCE_REFUSAL_KEY] = deferral.terminal(fallbacks_tried=[], owner_wait="not_allowed")
    elif deferral is not None and deferral.fact and result[0] is None:
        ctx.tools._ctx._deferred_resource_refusal = deferral
    pending_wait_handover = getattr(ctx.tools._ctx, "_pending_model_wait_handover", None)
    if pending_wait_handover is not None:
        if result[0] is not None:
            from_model, to_model, tool_count = pending_wait_handover
            _record_authoring_handover(
                ctx.tools._ctx,
                from_model=from_model,
                to_model=to_model,
                reason="model_wait",
                tool_calls_at_handover=tool_count,
            )
        ctx.tools._ctx._pending_model_wait_handover = None
    from ouroboros.primary_route_observation import record_round_route_result
    record_round_route_result(ctx, role, accepted=result[0] is not None)
    observed = ctx.accumulated_usage.get("_model_route")
    if (plan is not None and isinstance(observed, dict)
            and observed != getattr(plan, "model_route", {})):
        # A response/error may expose an Auto rotation after preparation. Withdraw
        # the prior account's capacity before another physical call is prepared.
        ctx.context_fit_plan, ctx.active_context_mode = _loop()._rebind_context_fit_plan(
            ctx.context_fit_plan, ctx.tools, ctx.messages, model=ctx.active_model,
            use_local=ctx.active_use_local, start_mode=ctx.active_context_mode,
            tool_schemas=ctx.tool_schemas, model_role=role, model_route=observed,
            credential_profile_id=(waiter.overrides.get(role, {}).get("model_account_override") if waiter else None))
    emit_model_substitution(ctx.accumulated_usage, task_id=ctx.task_id,
                            emit_progress=getattr(ctx, "emit_progress", None))
    call = ctx.accumulated_usage.get("_last_llm_call_meta")
    execution_id = ctx.accumulated_usage.get("execution_id")
    if (result[0] is not None and isinstance(call, dict) and call is not previous_call
            and execution_id and call.get("execution_id") == execution_id
            and call.get("round_id") == f"{execution_id}:round:{ctx.round_idx}"
            and call.get("llm_call_id")):
        call["usable_solve_response"] = True
        if capture is not None and capture.physical_context is not None and capture.physical_context.round_id == call["round_id"]:
            ctx.tools._ctx._usable_main_capture = capture
    return result


def _reprepare_waiting_main(ctx: _RoundModelCallContext, kwargs: dict):
    """Rebuild from the same immutable core; never replay completed tools/review."""
    from ouroboros.model_wait import PreparedModelCall
    from ouroboros.usage_accounting import current_physical_attempt_predicate, bind_physical_attempt_context

    model, use_local = kwargs["model"], kwargs.get("use_local", False)
    previous_model = str(ctx.active_model or "")
    role = kwargs["model_role"]
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
    ctx.context_fit_plan, ctx.active_context_mode = _loop()._rebind_context_fit_plan(
        ctx.context_fit_plan, ctx.tools, ctx.messages, model=model, use_local=use_local,
        start_mode=ctx.active_context_mode, tool_schemas=ctx.tool_schemas,
        model_role=role, model_route=observed or {},
        credential_profile_id=kwargs.get("model_account_override"))
    ctx.active_model, ctx.active_use_local = model, use_local
    ctx.tools._ctx.active_model = model
    ctx.tools._ctx.active_use_local = use_local
    if _fit_route_tool_ceiling(ctx):  # an owner's switch may land on a route with a schema ceiling
        kwargs["tools"] = ctx.tool_schemas
    trace = getattr(ctx.tools._ctx, "_execution_trace", {})
    _pending_model_wait_handover(
        ctx.tools._ctx,
        from_model=previous_model,
        to_model=str(model or ""),
        tool_calls=len(trace.get("tool_calls") or []) if isinstance(trace, dict) else 0,
    )
    _project_wake_input(ctx)
    # The physical pre-send predicate rechecks this newly adopted route before dispatch.
    from ouroboros.loop_messages import append_context_facts

    _loop()._retire_main_reasoning(ctx)
    append_context_facts(ctx)
    disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    from ouroboros.send_clock import MainSendClock
    from ouroboros.loop_llm_call import _prepare_main_messages
    from ouroboros.llm_attempt import single_physical_send

    # Any vision work belongs to this actual reprepare, outside both the Main
    # fit probe and the failed attempt's physical precondition/measurement —
    # and outside Main's clock and send claim: a helper is not a Main request.
    with bind_physical_attempt_context(None), MainSendClock(None).bound(), single_physical_send(None):
        kwargs["messages"] = _prepare_main_messages(
            ctx.messages, model=model, model_role=role,
            model_account_override=kwargs.get("model_account_override"),
            llm=ctx.llm, accumulated_usage=ctx.accumulated_usage,
            drive_root=ctx.drive_root or ctx.drive_logs.parent, task_id=ctx.task_id,
            event_queue=ctx.event_queue, use_local=use_local,
            task_attempt=ctx.accumulated_usage.get("_task_attempt"),
            deadline_ts=_loop()._task_deadline_epoch(ctx.tools),
        )
    from ouroboros.llm_claudexor import cache_key_for_model
    from ouroboros.provider_models import provider_for_model
    kwargs["cache_affinity"] = "" if use_local else cache_key_for_model(model)
    kwargs["allow_server_web_search"] = (_loop()._server_web_allowed_by_task(ctx.tools._ctx)
                                         and not use_local and provider_for_model(model) != "claudexor")
    if provider_for_model(model) == "claudexor":
        kwargs["bypass_response_cache"] = False
    physical = _physical_context_for_fit(disposition) if disposition else None
    if physical is not None:  # the remaining attempts of this call send under the new route's measurement
        ctx.accumulated_usage[REBOUND_PHYSICAL_CONTEXT_KEY] = physical
    return PreparedModelCall(kwargs, physical, current_physical_attempt_predicate())


def _reproject_actual_overflow_low(ctx: _RoundModelCallContext) -> None:
    """Size recovery lowers Max's books; a smaller physical frame may require Nano.

    Only the books change: Low shows my memory from the level the refused Max view showed, never the
    Low view prepared at startup and not sent (``ContextFitPlan.reproject_for_route``, ``select_mode``)."""
    if ctx.active_context_mode != "max" or ctx.context_fit_plan is None:
        return
    plan = ctx.context_fit_plan
    fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True)
    window = fit.measurement.capacity_total_tokens
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan.reproject_for_route(
        window_tokens=window or 0, known_window=window is not None,
        ratio=fit.measurement.measurement_density, output_reserve=int(plan.output_reserve_tokens),
        tool_schemas=ctx.tool_schemas, start_mode="max", current_messages=ctx.messages, select_mode="low")
    mode = ctx.context_fit_plan.initial_mode
    ctx.messages[:] = ctx.context_fit_plan.reproject_transcript(ctx.messages, mode)
    invalidate_task_cache_splits(ctx.task_id)
    ctx.active_context_mode = mode
    ctx.tools._ctx.messages = ctx.messages
    ctx.tools._ctx.active_context_mode = mode
    _adopt_view_facts(ctx.tools._ctx, ctx.context_fit_plan, mode)
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_fit_low_retry",
        "round": ctx.round_idx,
        "route_fp": str(getattr(ctx.context_fit_plan, "route_fp", "") or ""),
        "preferred_mode": str(getattr(ctx.context_fit_plan, "preferred_mode", "") or ""),
        "effective_mode": mode,
        "owner_visible": True,
    })


def _refused_candidate(facts: Dict[str, Any]) -> Any:
    """A pre-dispatch size refusal/prediction as this round's own comparison candidate."""
    physical = facts.get("physical_context")
    return SimpleNamespace(**{**facts, "physical_context": PhysicalAttemptContext(**physical) if physical else None,
                              "refused_before_dispatch": True})


def _failed_capture_is_comparable(capture: Any) -> bool:
    return bool(
        capture is not None
        and (getattr(capture, "state", None) in {"dispatched", "settled", "unresolved"}
             or getattr(capture, "refused_before_dispatch", False))
        and capture.candidate_measurement_kind == "canonical_json_v1"
        and capture.candidate_raw_sha256
        and capture.candidate_context_size_bytes is not None
        and capture.physical_context is not None
    )


def _strict_context_shrink_predicate(failed: Any, *, same_round: bool = True) -> Callable[[Any], bool]:
    """Admit only a strictly smaller candidate of the same provider/model.

    Size recovery also binds the round. Output recovery may carry its exact
    comparison with an unsent prepared view in the saved task across rounds.

    The account is not part of the comparison: a same-model Auto rotation observed on
    the refusal cannot veto the retry semantically. Its fresh capacity still binds the
    retry physically, because the dispatcher rebinds the plan to the observed route and
    the retry is measured and sent under that route's own ``physical_context``.
    """
    def predicate(request: Any) -> bool:
        failed_context = failed.physical_context
        current_context = request.physical_context
        return bool(
            request.candidate_measurement_kind == "canonical_json_v1"
            and request.provider == failed.provider
            and request.model == failed.model
            and request.max_completion_tokens <= failed.max_completion_tokens  # the retry's ceiling is the failed allowance
            and (not same_round or (current_context is not None and failed_context is not None
                                    and current_context.round_id == failed_context.round_id))
            and request.candidate_raw_sha256 != failed.candidate_raw_sha256
            and request.candidate_context_size_bytes is not None
            and int(request.candidate_context_size_bytes) < int(failed.candidate_context_size_bytes)
        )

    return predicate


def _emit_overflow_retry_skipped(ctx: _RoundModelCallContext, reason: str, *, rung: Optional[str] = None) -> None:
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "type": "context_overflow_retry_skipped",
        "round": ctx.round_idx,
        "route_fp": str(getattr(ctx.context_fit_plan, "route_fp", "") or ""),
        "reason": reason,
        **({"rung": rung} if rung else {}),
    })


ROUTING_RECEIPTS_HEADER = "[ROUTING_RECEIPTS]"


def _append_routing_receipts(ctx: _RoundModelCallContext) -> bool:
    """Show the model, at this boundary, the routing acts recorded for the owner message(s) it
    answers and the recorded outcome of its own acts whose tool result was unconfirmed
    (``ToolContext.unconfirmed_routing_acts``, read by exact ids even where no owner message
    carries them): one append-only host row in the projection every reader shares
    (``routing_wait.routing_act_fields``), only when it changed what the model already read.
    """
    tool_ctx = ctx.tools._ctx
    meta = getattr(tool_ctx, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    delivery = getattr(tool_ctx, "last_owner_delivery", None)
    ids = [str(meta.get("client_message_id") or ""),
           str(delivery.get("client_message_id") or "") if isinstance(delivery, dict) else ""]
    ids = [value for index, value in enumerate(ids) if value and value not in ids[:index]]
    own = [ref for ref in getattr(tool_ctx, "unconfirmed_routing_acts", None) or [] if isinstance(ref, dict)]
    root = (getattr(tool_ctx, "budget_drive_root", "") or meta.get("budget_drive_root") or ctx.drive_root
            or getattr(tool_ctx, "drive_root", None))
    if not (ids or own) or not root or str(meta.get("delegation_role") or "") == "subagent":
        return False
    try:
        from ouroboros.project_dialogue import _ANNOTATIONS_NAME, _latest_annotations_by_token
        from ouroboros.routing_wait import routing_act_fields, routing_act_lines as lines

        root = pathlib.Path(str(root))
        rows = _latest_annotations_by_token(root / "logs" / _ANNOTATIONS_NAME)
        acts = [routing_act_fields(root, key[0], row) for key, row in rows.items() if key[0] in ids]
        for ref in own:  # shown once settled: its unconfirmed tool result was already read
            key = (str(ref.get("client_message_id") or ""), str(ref.get("routing_token") or ""))
            act = routing_act_fields(root, key[0], rows.get(key) or ref) if key[0] not in ids else None
            if act and act["status"] != "unconfirmed" and act not in acts:
                acts.append(act)
    except Exception:
        log.debug("routing receipts unreadable at the model boundary", exc_info=True)
        return False
    current = lines(acts)
    shown: Optional[List[str]] = None
    for message in reversed(ctx.messages):
        text = message.get("content") if isinstance(message, dict) and message.get("role") == "user" else None
        if isinstance(text, str) and text.startswith(ROUTING_RECEIPTS_HEADER):
            shown = [line for line in text.splitlines() if line.startswith("- ")]
            break
    if shown is None:  # what the turn's own metadata already showed it at start
        contract = meta.get("routing_contract") if isinstance(meta.get("routing_contract"), dict) else {}
        startup = contract.get("message_routing_acts") or (
            [contract["message_routing_receipt"]] if isinstance(contract.get("message_routing_receipt"), dict) else [])
        shown = lines([act for act in startup if isinstance(act, dict)])
    if not current or current == shown:
        return False
    ctx.messages.append({"role": "user", "content": "\n".join([
        f"{ROUTING_RECEIPTS_HEADER} Routing acts recorded so far for the owner message(s) this turn answers "
        "and late outcomes of its own unconfirmed acts, oldest first — receipts from the routing rail; "
        "each row identifies the work and its recorded outcome.",
        *current,
        "Not listed: another turn's act under its own agent-steer id, or an act keyed only to a task id; "
        "neither links to this message."])})
    return True


def _project_wake_input(ctx: _RoundModelCallContext, *, overflowed: bool = False) -> bool:
    """Deliver a wake's observation by its exact source when the whole one cannot fit.

    Re-decided at every round boundary until a response has consumed the wake
    input, on that round's actual request: tool-inclusive measurement (the tools
    and route of THIS call, so a grown tool set or a switched route is measured
    again), the answer reserve and one clock line against a KNOWN capacity.
    Unknown capacity sends the whole input, as any call does; a real provider
    overflow may apply the delivery before consumption, as the one retry the
    strict-shrink predicate admits only when it is actually smaller. Once a
    response followed the input, the transcript is history and stays as sent.
    Only the delivered first user message changes; the task text, the owner
    corpus and the stored source keep the complete observation (``consciousness_wake``).
    """
    tool_ctx = ctx.tools._ctx
    meta = getattr(tool_ctx, "task_metadata", None)
    wake = meta.get("wake_observation") if isinstance(meta, dict) else None
    if not isinstance(wake, dict) or not wake.get("projection_text"):
        return False
    index = next((i for i, message in enumerate(ctx.messages) if message.get("role") == "user"), None)
    if index is None or any(isinstance(message, dict) and message.get("role") == "assistant"
                            for message in ctx.messages[index + 1:]):
        return False  # a response already followed the wake input: consumed history, never rewritten
    if not hasattr(tool_ctx, "_wake_input_full_message"):
        tool_ctx._wake_input_full_message = copy.deepcopy(ctx.messages[index])
    full = tool_ctx._wake_input_full_message
    delivery = {"role": "user", "content": str(wake["projection_text"])}
    if len(json.dumps(delivery, ensure_ascii=False)) >= len(json.dumps(full, ensure_ascii=False)):
        return False  # a pointer must actually save space
    projected = overflowed
    if not overflowed:
        # Re-evaluate the FULL input on each actual route, even when the last
        # unconsumed candidate used a pointer. Measurement never rewrites history.
        current = ctx.messages[index]
        ctx.messages[index] = full
        try:
            disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
        finally:
            ctx.messages[index] = current
        measurement = disposition.measurement if disposition is not None else None
        from datetime import datetime, timezone
        from ouroboros.send_clock import main_clock_policy, render_clock_note
        from ouroboros.utils import estimate_tokens

        policy = main_clock_policy(meta, task_type=ctx.task_type)
        clock = estimate_tokens(render_clock_note(policy, datetime.now(timezone.utc))) if policy else 0
        projected = bool(measurement is not None and measurement.capacity_total_tokens is not None
                         and measurement.estimated_input_tokens + measurement.response_reserve_tokens + clock
                         > measurement.capacity_total_tokens)
    from ouroboros.context_fit import extract_plain_text_from_content

    desired = delivery if projected else full
    # Cross-family fallbacks copy the transcript, then restore it on failure.
    # Projection belongs to THESE bytes, never a flag on the shared tool context.
    if extract_plain_text_from_content(ctx.messages[index].get("content")) == extract_plain_text_from_content(desired.get("content")):
        return False
    ctx.messages[index] = delivery if projected else copy.deepcopy(full)
    invalidate_task_cache_splits(ctx.task_id)
    _loop().seal_task_transcript(ctx.messages)
    tool_ctx.messages = ctx.messages
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "wake_input_by_source" if projected else "wake_input_inline",
        "round": ctx.round_idx, "after_overflow": overflowed,
        "composition": wake.get("composition"), "source_sha256": (wake.get("source") or {}).get("sha256"),
    })
    return True


def _fit_route_tool_ceiling(ctx: _RoundModelCallContext) -> bool:
    """Keep the resident schemas within the acting route's physical ceiling (OpenAI: 128).

    In place, before measurement, so the fit, the priced candidate and the send carry
    one list and discovery reports true residency. Names the actor loaded through
    enable_tools in this run, and names left out earlier, stay; the newly left-out
    names reach the actor as a fact.
    Called by every Main round and by a wait's reprepare; True when the list changed.
    """
    from ouroboros.provider_models import tool_schema_limit
    from ouroboros.tool_policy import fit_tool_schemas_to_limit, route_tool_limit_notice

    schemas, limit = ctx.tool_schemas, tool_schema_limit(ctx.active_model, use_local=ctx.active_use_local)
    if limit is None or schemas is None or len(schemas) <= limit:
        return False
    earlier = frozenset(getattr(ctx.tools._ctx, "_route_left_out_tool_names", ()) or ())
    loaded = frozenset(getattr(ctx.tools._ctx, "_actor_loaded_tool_names", ()) or ())
    total = len(schemas)
    schemas[:], left_out = fit_tool_schemas_to_limit(schemas, limit, keep=earlier | loaded)
    ctx.tools._ctx._route_left_out_tool_names = earlier | set(left_out)
    invalidate_task_cache_splits(ctx.task_id)
    _loop()._append_or_merge_user_message(
        ctx.messages, route_tool_limit_notice(ctx.active_model, limit, total, left_out))
    return True


def _call_round_model(ctx: _RoundModelCallContext) -> Tuple[Any, float, str]:
    return output_recovery.call(ctx, _call_round_model_once)


def _call_round_model_once(ctx: _RoundModelCallContext) -> Tuple[Any, float, str]:
    """Measure, dispatch, and recover a provider refusal or known-window prediction of no result room."""
    from ouroboros.loop_llm_call import timeout_repeat_blocked
    if timeout_repeat_blocked(ctx.accumulated_usage):
        return None, None, ctx.active_context_mode
    selected = getattr(ctx.tools._ctx, "_authored_memory_plan", None)
    if (selected is not None and selected.core_sha256 == getattr(ctx.context_fit_plan, "core_sha256", None)
            and selected.route_fp == getattr(ctx.context_fit_plan, "route_fp", None)):
        ctx.context_fit_plan = selected
        ctx.tools._ctx._authored_memory_plan = None
    ctx.accumulated_usage.pop("_last_llm_failure", None)
    ctx.accumulated_usage.pop("_acting_route_failure", None)
    from ouroboros.primary_route_observation import observe_primary_route
    observation = observe_primary_route(ctx)
    if observation:
        _loop()._append_or_merge_user_message(ctx.messages, observation)
    facts = getattr(ctx.tools._ctx, "_route_facts_pending", "")
    if facts and ctx.defer_resource_wait is None:  # the acting route's first own round after a switch
        ctx.tools._ctx._route_facts_pending = ""
        _loop()._append_or_merge_user_message(ctx.messages, facts)
    _fit_route_tool_ceiling(ctx)
    _append_routing_receipts(ctx)
    from ouroboros.project_placement import append_placement_notice
    append_placement_notice(ctx.tools._ctx, ctx.messages)
    _project_wake_input(ctx)
    disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    msg, cost = _loop()._dispatch_round_model(
        ctx,
        disposition,
        attempt_cap=ctx.attempt_cap,
    )
    if msg is None and TIMEOUT_REPEAT_KEY in ctx.accumulated_usage:
        return msg, cost, ctx.active_context_mode  # its timeout repeat is spent: no image or size retry either
    from ouroboros.vision_routing import retry_refused_image_round  # the capture is read right after the send
    retried = None if msg is not None else retry_refused_image_round(ctx, _loop().last_physical_attempt_capture())
    if retried is not None:
        msg, retry_cost = retried
        cost = float(cost or 0.0) + float(retry_cost or 0.0)
    refused = ctx.accumulated_usage.pop(REFUSED_CANDIDATE_KEY, None)  # includes a pre-send-refused image retry
    if msg is not None or str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") != "context_overflow":
        return msg, cost, ctx.active_context_mode
    ctx.accumulated_usage[CONTEXT_REFUSAL_ROUND_KEY] = ctx.round_idx  # this round's size refusal, passes or not

    # Snapshot immediately: a reclaim summarizer is itself physically receipted
    # and would otherwise replace the failed Main candidate in the ContextVar.
    # A refusal before dispatch left no capture: compare with the refused candidate, never an earlier round's.
    failed_capture = _refused_candidate(refused) if refused else _loop().last_physical_attempt_capture()
    if disposition is None:
        return msg, cost, ctx.active_context_mode
    if isinstance(ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict):
        _emit_overflow_retry_skipped(ctx, "round_holds_unresolved_attempt")
        return msg, cost, ctx.active_context_mode
    # A wake's stored-source delivery is model-free and already applied: it earns the first retry.
    prepared = _project_wake_input(ctx, overflowed=True)
    return _recover_context_overflow(ctx, failed_capture, cost, prepared=prepared)


# Size recovery (owner revision, 20.10.2026): a known-window prediction of no result room or the provider's
# typed size refusal starts two model-free passes, each earning one strictly smaller check/retry: the
# first replaces superseded host state and bodies a usable turn already showed; the second, only after a
# further failed check/refusal, lowers memory granularity (books too when memory falls short of the excess), then
# changes books before addressing eligible unseen bodies, a just-chosen one last. Each pass runs once
# per round whatever the route, so a configured fallback receives the view as applied and walks nothing
# again. The fallback chain and the end of the path follow in the caller.
REFUSAL_PASSES: Tuple[Tuple[str, ...], ...] = (("host_copies", "bodies"), ("floor", "books", "unseen_bodies"))
WAKE_DELIVERY_STEP = "wake_delivery"  # the change applied before the passes, when there was one
#: The round needing size recovery (``context_renewal`` reads it); a fallback's other failure cannot cancel it.
CONTEXT_REFUSAL_ROUND_KEY = "_context_refusal_round"


def _refusal_window(ctx: _RoundModelCallContext, fit: Any,
                    refused: Any = None) -> Tuple[Optional[int], int, int, int]:
    """``(stated limit or None, the refused input, the return-after-fold slack, the reply it reserved)``.

    A provider's stated limit (OpenAI, OpenRouter, Anthropic and Gemini wordings) with its own count of the
    refused request when it states one, or a local serving instance's exact window and input (``refused``:
    its pre-send ``context_window_tokens`` and ``input_tokens``), measure it; otherwise the refused estimate
    stands. The slack is the owner's return-after-fold eighth (L33 2=A) of the stated window,
    else of the known window, else of the larger of the reading and the largest input accepted here.
    """
    from ouroboros.capability_evidence import _parse_ctx_limit_number, _parse_ctx_requested
    from ouroboros.context_fit import reclaim_low_water_margin

    measurement = fit.measurement
    message = str(ctx.accumulated_usage.get("_last_llm_provider_message") or "")
    stated = _parse_ctx_limit_number(message) or None
    # The provider's own count of the refused request beats the estimate when it states both; a stated limit
    # beside a COLD estimate measures nothing (chars/4 under-counts code and Cyrillic): unknown, as before.
    stated_input, stated_reply, stated_total = _parse_ctx_requested(message) if stated else (0, 0, 0)
    reply = stated_reply or measurement.reply_allowance_tokens
    provider_size = stated_input or max(0, stated_total - reply)  # a bare total already holds its reply
    if stated and not provider_size and measurement.anchor_input_tokens is None:
        stated = None
    size = provider_size or measurement.estimated_input_tokens
    exact_window, exact_input = (int(getattr(refused, key, 0) or 0) for key in ("context_window_tokens", "input_tokens"))
    if exact_window and exact_input:  # the instance's own counts, with the reply its exact request reserved
        stated, size = exact_window, exact_input
        reply = int(getattr(refused, "max_completion_tokens", 0) or reply)
    window = stated or measurement.capacity_total_tokens or max(
        measurement.window_reading_tokens or 0, measurement.accepted_input_tokens or 0) or None
    return stated, size, reclaim_low_water_margin(None, window) if window else 0, reply


def _refusal_target(ctx: _RoundModelCallContext, fit: Any, refused: Any = None) -> Tuple[Optional[int], int]:
    """``(tokens the next candidate sheds, the slack in them)`` measured on this refusal; None: nothing measures it.

    A stated limit L (``_refusal_window``) gives ``input + reply - L`` with the provider's own count of the
    refused request, or an anchored estimate (a cold one beside a limit measures nothing). Without one, the
    anchored estimate E over the largest input this model accepted here, A, gives ``E - A``; E at or below A
    contradicts the reading, and a cold E compares with nothing: unknown. Both add the return-after-fold
    slack. The refused input is recorded, so the facts state no optimistic room for this model again.
    """
    from ouroboros.context_anchor import record_refused_input

    measurement = fit.measurement
    predicted_free = getattr(refused, "predicted_free_tokens", None)
    if predicted_free is not None:  # a prediction never becomes a provider-refused size in the task's history
        slack = refused.predicted_slack_tokens
        return max(0, -predicted_free) + slack, slack
    stated, size, slack, reply = _refusal_window(ctx, fit, refused)
    record_refused_input(ctx.accumulated_usage, ctx.context_fit_plan, size)  # the provider's count when it stated one
    if stated:
        return max(0, size + reply - stated) + slack, slack
    accepted = measurement.accepted_input_tokens
    if measurement.anchor_input_tokens is None or not accepted or measurement.estimated_input_tokens <= accepted:
        return None, slack
    return measurement.estimated_input_tokens - accepted + slack, slack


def _recover_context_overflow(ctx: _RoundModelCallContext, failed_capture: Any, cost: float, *,
                              prepared: bool = False) -> Tuple[Any, float, str]:
    """Run the refusal passes: each pass that changed the candidate earns one strictly smaller retry.

    ``prepared`` says the caller already changed the candidate without a model (the wake's
    stored-source delivery): that change is retried first, then the passes.
    """
    def _skipped(reason: str) -> Tuple[Any, float, str]:
        _emit_overflow_retry_skipped(ctx, reason)
        return None, cost, ctx.active_context_mode

    if not _failed_capture_is_comparable(failed_capture):
        return _skipped("failed_candidate_not_comparable")
    done = _loop()._context_overflow_retries(ctx.tools._ctx)
    fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    if fit is None:
        return None, cost, ctx.active_context_mode
    target, slack = _refusal_target(ctx, fit, failed_capture)
    real = getattr(failed_capture, "predicted_free_tokens", None) is None  # the round's start: a provider's, not an estimate
    for steps in ((WAKE_DELIVERY_STEP,),) * bool(prepared) + REFUSAL_PASSES:
        if steps != (WAKE_DELIVERY_STEP,):
            key = (fit.measurement.round_id, "+".join(steps))  # once per round, never re-armed for a route
            if key in done:
                continue
            done.add(key)
            if not _apply_refusal_pass(ctx, steps, fit, target, slack, real):
                continue
        fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True)
        if fit is None:
            return None, cost, ctx.active_context_mode
        try:
            retry_msg, retry_cost = _loop()._dispatch_round_model(
                ctx, fit, attempt_cap=1,
                candidate_predicate=_strict_context_shrink_predicate(failed_capture),
                max_tokens=int(getattr(failed_capture, "max_completion_tokens", 0) or 0) or None,
            )
        except PhysicalAttemptPreconditionFailed:
            _emit_overflow_retry_skipped(ctx, "context_candidate_not_strictly_smaller", rung="+".join(steps))
            continue  # ``fit`` already measures the changed view the next pass starts from
        cost = float(cost or 0.0) + float(retry_cost or 0.0)
        if retry_msg is not None or str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") != "context_overflow":
            return retry_msg, cost, ctx.active_context_mode  # answered, or another kind: ordinary recovery
        if isinstance(ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict):
            return _skipped("round_holds_unresolved_attempt")
        # Refused again: the next pass must shrink below THIS candidate, sized by THIS refusal.
        refused = ctx.accumulated_usage.pop(REFUSED_CANDIDATE_KEY, None)
        failed_capture = _refused_candidate(refused) if refused else _loop().last_physical_attempt_capture()
        if not _failed_capture_is_comparable(failed_capture):
            return _skipped("failed_candidate_not_comparable")
        fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True) or fit
        target, slack = _refusal_target(ctx, fit, failed_capture)
    return None, cost, ctx.active_context_mode


def _apply_refusal_pass(ctx: _RoundModelCallContext, steps: Tuple[str, ...], fit: Any,
                        target: Optional[int], slack: Optional[int] = None, refused: bool = False) -> bool:
    """Apply a pass's steps in order until together they shed ``target``; True when the candidate changed.

    Each step is sized by what the earlier steps of the pass left of the target: host copies that
    already freed the room leave every body in place, and memory the floor narrowed enough addresses
    no unseen body (the gradual order the owner asked for). An unknown target lets each step take its
    whole category. A just-chosen body that fits alone goes last, only in a round a provider's refusal
    started (``refused``) and only for what the late rescue's other steps could not answer: the measured
    target left, or nothing changed against an unknown one (owner 2A).
    """
    changed, remaining = False, target
    for step in steps:
        if remaining is not None and remaining <= 0:
            break
        if getattr(ctx.context_fit_plan, "window_record", None) is not None:
            fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True) or fit
        shed = _apply_refusal_step(ctx, step, fit, remaining, slack)
        if shed is not None:
            changed = True
            remaining = None if remaining is None else remaining - shed
    if refused and "unseen_bodies" in steps and (not changed if remaining is None else remaining > 0):
        changed |= _run_emergency_address_pass(ctx, fit, rung="chosen_bodies", goal_tokens=remaining).status == "applied"
    return changed


def _apply_refusal_step(ctx: _RoundModelCallContext, step: str, fit: Any, target: Optional[int],
                        slack: Optional[int] = None) -> Optional[int]:
    """Apply one step of a refusal pass to the live transcript: the tokens it shed, None when unchanged.

    Bodies are addressed oldest first until ``target`` (the whole category when it is unknown). The floor
    narrows memory by the measured excess only, never by the return-after-fold slack: memory is continuity
    (BIBLE P1), and narrowing it for slack showed this room's own rows by address. Without a target it
    narrows by the deficit against a known window, and only with neither by everything it can give. Books
    become navigation before any eligible unseen body leaves; a just-chosen body leaves last.
    Memory alone satisfying the whole goal leaves the books and bodies untouched.
    """
    if step in ("host_copies", "bodies", "unseen_bodies"):
        goal = None if step == "host_copies" else target
        receipt = _run_emergency_address_pass(ctx, fit, rung=step, goal_tokens=goal)
        return max(0, int(receipt.reclaimed_tokens or 0)) if receipt.status == "applied" else None
    if step not in ("floor", "books") or (step == "books" and ctx.active_context_mode != "max"):
        return None  # outside Max the books already show as navigation
    current = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True) if step == "books" else fit
    before = (current or fit).measurement.estimated_input_tokens

    def shed_since_before(default: int) -> int:
        after = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True)
        return max(0, before - after.measurement.estimated_input_tokens) if after is not None else default

    def books_to_navigation(saved: int) -> Optional[int]:
        mode = ctx.active_context_mode
        _reproject_actual_overflow_low(ctx)
        return shed_since_before(saved) if ctx.active_context_mode != mode else None

    if step == "books":
        return books_to_navigation(0)
    # The measured excess, else the deficit against a known window, else everything memory can give.
    slack = _refusal_window(ctx, fit)[2] if slack is None else slack
    excess = max(1, target - slack) if target else (fit.measurement.capacity_deficit_tokens or WHOLE_CATEGORY)
    coarsened = _coarsen_memory_view(ctx, fit, saving=excess)
    saved = shed_since_before(0) if coarsened else 0
    if target and saved >= excess:
        return saved  # the next step handles any remaining slack
    shed = books_to_navigation(saved)
    return shed if shed is not None else (saved if coarsened else None)
