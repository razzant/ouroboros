"""The per-round model call: context-fit identification, measurement and memory,
dispatch, the main-context reclaim, overflow-retry predicates, the cross-model
fallback chain and context-fit plan rebinding. Extracted from loop.py (v7 L-B
split); loop.py re-exports every name."""

from __future__ import annotations

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
from ouroboros.context_budget import ContextReclaimRequest
from ouroboros.context_compaction import context_reclaim_transcript_sha256
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import REBOUND_PHYSICAL_CONTEXT_KEY, REFUSED_CANDIDATE_KEY, TRANSPORT_DEATHS_KEY, _TRANSPORT_DEATH_RETRIES
from ouroboros.loop_round_limits import _coarsen_memory_view, _run_emergency_address_pass
from ouroboros.loop_tool_execution import prune_reclaim_trace_refs, reclaim_negative_memo, reclaim_trace_refs
from ouroboros.observability import new_execution_id
from ouroboros.tools.registry import ToolRegistry
from ouroboros.transcript_prefix import sanction_rewrite
from ouroboros.usage_accounting import PhysicalAttemptContext, PhysicalAttemptPreconditionFailed, invalidate_task_cache_splits


log = logging.getLogger("ouroboros.loop")
# The typed, temporary non-success of a round whose resource refusal no route recovered and
# no owner wait follows (inline Presence, or a chain stopped by its own fence): nothing more is sent.
RESOURCE_REFUSAL_KEY = "resource_refusal"
# A round failure that owns a wait episode on its own route when no configured route answers.
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


def _snapshot_context_fit_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in usage.items() if key.startswith("_context_")}


def _restore_context_fit_usage(
    usage: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> None:
    for key in tuple(usage):
        if key.startswith("_context_"):
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


def _route_facts_text(tool_ctx: Any, *, model: str, use_local: bool, role: str, failed_model: str,
                      failure: Dict[str, Any], waiter: Any) -> str:
    """Facts after a host-driven switch away from the primary; the acting model decides what follows."""
    from ouroboros.model_slots import route_binding
    from ouroboros.provider_models import provider_for_model

    primary = getattr(tool_ctx, "primary_route", None)
    if not isinstance(primary, dict) or not primary.get("model"):
        return ""

    def label(name: str, local: bool, route_role: str) -> str:
        account = route_binding(name, local, route_role, overrides=waiter.overrides if waiter else None)[2]
        managed = not local and provider_for_model(name) == "claudexor"
        account_text = f"; account {account or 'Auto'}" if managed else ""
        return f"{name}{' (local)' if local else ''} (role {route_role}{account_text})"

    reset = str(failure.get("reset_at") or "")
    return (f"[ROUTE FACTS] This task's primary route is {label(primary['model'], bool(primary['use_local']), primary['role'])}; "
            f"{label(model, use_local, role)} answered this round after {failed_model} failed with "
            f"{failure.get('failure_code') or 'an error'} at {failure.get('ts') or 'an unrecorded time'}"
            f"{f' (its engine dated a reset at {reset})' if reset else ''}. Facts, not an instruction: stay on this "
            "route, return with switch_model(primary=\"return\") (the next real request tests the primary; a refusal "
            "there moves on to configured routes again), or return and wait for it with switch_model(primary=\"wait\"). "
            "No catalog check or timer shows whether the primary serves now.")


def _run_cross_model_fallback_chain(
    *, llm, ctx, tools, messages, active_model, active_use_local, tool_schemas,
    active_effort, max_retries, drive_logs, task_id, round_idx, event_queue,
    accumulated_usage, task_type, emit_progress, context_fit_plan,
    active_context_mode,
) -> tuple:
    """Try the configured routes other than the acting binding before any wait begins.

    An eligible unknown outcome moves on to the next route with facts-only recovery
    input (a real, possibly charged generation); an ineligible one (an accepted operation still readable,
    inline Presence) or a spent deadline stops the walk. A round whose own failure is an
    outage or an unknown outcome keeps that failure as its wait when none answers.
    """
    from ouroboros import fallback_cooldown as _fcd
    from ouroboros.loop_transport import append_unknown_recovery_input
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option, task_model_binding, route_binding
    from ouroboros.model_wait import current_model_wait
    from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS as _cooldown_kinds
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
    entry_failure = next((dict(ref) for ref in reversed(accumulated_usage.get("llm_call_refs") or [])
                          if isinstance(ref, dict) and ref.get("failure_code")), {})
    # A primary resource refusal returned here, not waited: the configured routes come
    # first, and the owner question is that refusal's own wait, asked after them.
    deferred, tools._ctx._deferred_resource_refusal = getattr(tools._ctx, "_deferred_resource_refusal", None), None
    owner_question = (task_type != "presence" and deferred is not None and not own_wait
                      and waiter is not None and waiter.waits_allowed)
    deferred_candidate = None  # The route that actually refused; never replay an unrelated primary.
    rows = _route_candidates(tools._ctx, active_model, active_use_local,
                             str(getattr(context_fit_plan, "model_role", "") or ""), waiter)
    tried = []
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
                attempt_cap=attempt_cap,
                model_role=fallback_role,
                emit_progress=emit_progress,
                defer_resource_wait=(task_type == "presence" or owner_question
                                     or _route_follows(rows[index + 1:])),
            )
        tried.append(fallback_model)
        resident = (None if tool_schemas is None else list(tool_schemas),
                    getattr(tools._ctx, "_route_left_out_tool_names", None))
        msg, _cost, candidate_mode = _loop()._call_round_model(candidate_call)
        if deferred is None and msg is None:
            # Each fallback clears the transient context slot before its own send. Keep the
            # first actual resource refusal for this chain: a later bad request cannot
            # erase evidence that an allowed route refused before generation.
            deferred = getattr(tools._ctx, "_deferred_resource_refusal", None)
            if deferred is not None:
                deferred_candidate = candidate_call
            owner_question = (task_type != "presence" and deferred is not None and not own_wait
                              and waiter is not None and waiter.waits_allowed)
        if msg is not None:
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
            tools._ctx._route_facts_pending = "" if is_primary else _route_facts_text(
                tools._ctx, model=fallback_model, use_local=fallback_use_local, role=fallback_role,
                failed_model=entry_model, failure={**entry_failure, "failure_code": entry_kind or
                                                   entry_failure.get("failure_code")}, waiter=waiter)
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
        if _walk_fenced(tools._ctx, accumulated_usage) or str(
                accumulated_usage.get("_last_llm_error_kind") or "") == "llm_output_exhausted":
            break  # an exhausted candidate answered: its outcome is the round's, no further route is dialed
        _cooled(fallback_model, fallback_use_local, fallback_role)
        previous_model, previous_tag = fallback_model, ftag
    fenced = msg is None and _walk_fenced(tools._ctx, accumulated_usage)
    if deferred is None:
        deferred = getattr(tools._ctx, "_deferred_resource_refusal", None)
    owner_question = (task_type != "presence" and deferred is not None and not own_wait
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
            tools._ctx._route_facts_pending = _route_facts_text(
                tools._ctx, model=active_model, use_local=active_use_local, role=role,
                failed_model=entry_model, failure={**entry_failure, "failure_code": entry_kind}, waiter=waiter)
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
    previous = route[:2]
    model, use_local, effort = _loop()._apply_runtime_overrides(ctx, *route)
    role, ctx.active_role_override = getattr(ctx, "active_role_override", None), None
    if role and getattr(ctx, "primary_route", None):
        from ouroboros.primary_route_observation import record_primary_return_request
        record_primary_return_request(ctx, model, use_local, role)
    if (model, use_local) != previous or role:
        context_fit_plan, active_context_mode = _loop()._rebind_context_fit_plan(
            context_fit_plan, tools, messages, model=model, use_local=use_local, preferred_mode=preferred_mode,
            tool_schemas=tool_schemas, **({"model_role": role, "model_route": {}} if role else {}))
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
    from ouroboros.capability_evidence import is_known
    from ouroboros.context import _context_fit_route
    from ouroboros.context_fit import _failed_route_evidence, _route_calibration_ratio, main_output_reserve_tokens
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
    ratio = _route_calibration_ratio(
        None,  # canonical evidence root (one observation store)
        str(getattr(evidence, "route_fp", "") or ""),
        str(route.get("model") or model),
    )
    known_window = is_known(evidence, require_fresh=True)
    window_tokens = int(getattr(evidence, "window_tokens", 0) or 0)
    output_reserve = main_output_reserve_tokens(use_local=bool(route.get("use_local", use_local)), evidence=evidence)
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
    ).reproject_for_route(  # the memory view re-rendered for this route's window, from the same capture
        window_tokens=window_tokens, known_window=known_window, ratio=ratio, output_reserve=output_reserve,
        tool_schemas=tool_schemas, start_mode=start_mode, current_messages=messages)
    mode = rebound.initial_mode
    projected_prompt_tokens = rebound.projected_tokens_with_tools(mode, tool_schemas)
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
                "projected_prompt_tokens": projected_prompt_tokens,
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
    usage["_context_target_total_tokens"] = measurement.target_total_tokens
    usage["_context_capacity_total_tokens"] = measurement.capacity_total_tokens
    usage["_context_target_deficit_tokens"] = measurement.target_deficit_tokens
    usage["_context_capacity_deficit_tokens"] = measurement.capacity_deficit_tokens
    usage["_context_reclaim_goal_tokens"] = measurement.reclaim_goal_tokens
    usage["_context_target_miss"] = disposition.action == "send_target_miss"
    usage["_context_automatic_pass_used"] = disposition.automatic_pass_used
    usage["_context_predicted_capacity_miss"] = disposition.predicted_capacity_miss


def _measure_round_main_fit(
    ctx: _RoundModelCallContext,
    *,
    automatic_pass_used: bool,
) -> Any:
    plan = ctx.context_fit_plan
    if plan is None or str(ctx.active_model or "") != str(getattr(plan, "model", "") or ""):
        return None
    from ouroboros.context_fit import measure_main_fit

    rendered_mode = str(ctx.active_context_mode) if str(ctx.active_context_mode) in {"max", "low", "nano"} else "max"
    disposition = measure_main_fit(
        plan,
        ctx.messages,
        ctx.tool_schemas,
        profile=_main_context_profile(plan, rendered_mode),
        rendered_mode=rendered_mode,
        round_id=_context_fit_round_id(ctx),
        automatic_pass_used=automatic_pass_used,
        reasoning_effort=ctx.active_effort,
    )
    _remember_main_fit(ctx, disposition)
    return disposition


def _physical_context_for_fit(disposition: Any) -> PhysicalAttemptContext:
    measurement = disposition.measurement
    return PhysicalAttemptContext(
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
    )


def _measure_main_context_view(plan, messages, schemas, mode, effort, round_id) -> dict:
    """Measure an authored candidate without changing Main's dispatch policy.

    Predictions inform the next ordinary reclaim/send; neither capacity nor
    economy estimates veto a useful authored view. No route probe or model runs.
    """
    from ouroboros.context_fit import estimate_context_prompt_tokens, measure_main_fit
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS

    if plan is None:
        return {"accepted": True, "strict_bound_proven": False,
                "estimated_input_tokens": estimate_context_prompt_tokens(messages, schemas, reasoning_effort=effort),
                "response_reserve_tokens": MAIN_LOOP_MAX_TOKENS, "capacity_total_tokens": None,
                "measurement_basis": "cold_estimate", "reason": "main_route_capacity_unknown"}
    disposition = measure_main_fit(
        plan, messages, schemas, profile=_main_context_profile(plan, mode),
        rendered_mode=mode, round_id=round_id, reasoning_effort=effort,
    )
    return {"accepted": True, "strict_bound_proven": False,
            "predicted_capacity_miss": disposition.predicted_capacity_miss,
            **asdict(disposition.measurement)}


def _fit_key(fit: Any) -> Tuple[str, str]:
    return (fit.measurement.route_fp, fit.measurement.round_id)


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
    from ouroboros.loop_messages import append_context_facts

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
    from ouroboros.acceptance_settlement import expose_acceptance_feedback

    import copy
    from ouroboros.loop_delivery import completion_observation, completion_feedback
    trace = getattr(ctx.tools._ctx, "_execution_trace", None) or {}
    observation = completion_observation(ctx.tools._ctx, trace)
    feedback_snapshot = copy.deepcopy({key: trace.get(key) for key in (
        "review_runs", "acceptance_review_outcome", "acceptance_preparation")})
    ctx.tools._ctx._completion_observation = observation

    def observe_feedback(sent):
        expose_acceptance_feedback(trace, sent, str(ctx.task_id))
        expose_acceptance_feedback(feedback_snapshot, sent, str(ctx.task_id))
        observation.update(feedback=copy.deepcopy(completion_feedback(feedback_snapshot)),
                           preparation=copy.deepcopy(feedback_snapshot.get("acceptance_preparation") or {}))

    with contextlib.ExitStack() as binding:
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
            # Inline Presence keeps its bounded typed-death repeat; every other round's new
            # generation after an unknown outcome is the round recovery (configured routes first).
            transport_death_retries=(_TRANSPORT_DEATH_RETRIES if attempt_cap is None
                                     and ctx.task_type == "presence" else 0),
            stop_retry_check=(lambda: transport_repeat_stop_requested(ctx.tools._ctx, mailbox_peek=mailbox_peek)) if attempt_cap is None else None,
            allow_server_web_search=_loop()._server_web_allowed_by_task(ctx.tools._ctx),
            physical_context=(_physical_context_for_fit(disposition) if disposition is not None else None),
            candidate_predicate=candidate_predicate, model_role=role, model_account_override=account,
            processing_preference=task_processing_preference(
                {"task_metadata": getattr(ctx.tools._ctx, "task_metadata", {})}, model_role=role),
            # The loop's own active-turn slot: a reprepared send keeps this exact
            # owner because the slot survives kwargs deep-copying by identity.
            model_turn_state=getattr(ctx.tools._ctx, "model_turn_state", None),
            model_context_observer=observe_feedback,
            send_clock_policy=main_clock_policy(
                getattr(ctx.tools._ctx, "task_metadata", {}), task_type=ctx.task_type),
            **({"max_tokens": int(max_tokens)} if max_tokens else {}),
        )
    ctx.accumulated_usage.pop(REBOUND_PHYSICAL_CONTEXT_KEY, None)  # consumed by the call's later attempts, if any
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
    # Prediction alone never starts a helper pass (owner decision 7A): the measurement is
    # recorded as facts; only the provider's actual refusal opens the recovery ladder.
    from ouroboros.loop_messages import append_context_facts

    append_context_facts(ctx)
    disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    from ouroboros.send_clock import MainSendClock

    if disposition is not None and _fit_key(disposition) in _loop()._context_reclaim_passes(ctx.tools._ctx):
        disposition = _measure_after_reclaim(ctx)
    from ouroboros.loop_llm_call import _prepare_main_messages

    # Any vision work belongs to this actual reprepare, outside both the Main
    # fit probe and the failed attempt's physical precondition/measurement —
    # and outside Main's clock: a helper send is not a Main request.
    with bind_physical_attempt_context(None), MainSendClock(None).bound():
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


def _run_main_reclaim(
    ctx: _RoundModelCallContext,
    disposition: Any,
    *,
    minimum_goal_tokens: int = 0,
    provider_refused: bool = False,
) -> Any:
    measurement = disposition.measurement
    key = _fit_key(disposition)
    passes = _loop()._context_reclaim_passes(ctx.tools._ctx)
    if key in passes:
        return None
    economic_deficit = int(measurement.target_deficit_tokens or 0)
    physical_deficit = int(measurement.capacity_deficit_tokens or 0)
    # A soft economic target cannot veto useful relief of physical pressure.
    deficit = 0 if minimum_goal_tokens else physical_deficit or economic_deficit
    request = ContextReclaimRequest(
        route_fp=measurement.route_fp,
        round_id=measurement.round_id,
        transcript_sha256=context_reclaim_transcript_sha256(ctx.messages),
        measurement_basis=measurement.measurement_basis,
        measurement_density=measurement.measurement_density,
        reclaim_goal_tokens=max(
            int(measurement.reclaim_goal_tokens),
            max(0, int(minimum_goal_tokens)),
        ),
        allow_partial_shrink=True,
    )
    rebuilt, receipt, usage = _loop().compact_tool_history_llm(
        ctx.messages,
        request=request,
        drive_root=pathlib.Path(ctx.drive_root or ctx.drive_logs.parent),
        task_id=ctx.task_id,
        negative_memo=reclaim_negative_memo(ctx.tools._ctx),
        trace_refs_by_tool_call_id=reclaim_trace_refs(ctx.tools._ctx),
        exposed_units=(getattr(ctx.tools._ctx, "_last_context_observation", {}) or {}).get("exposed_units", []),
        automatic_deficit_tokens=deficit,
        provider_refused=provider_refused,
    )
    passes.add(key)
    # The checkpoint is written only after non-empty selection and immediately
    # before map/fold, so it also covers a post-summary binding mismatch.
    if receipt.checkpoint_ref:
        _loop()._context_reclaim_materializations(ctx.tools._ctx).add(key)
    if usage:
        _loop()._account_compaction_usage(ctx.accumulated_usage, usage, ctx.event_queue, ctx.task_id)
    if receipt.status == "applied":
        invalidate_task_cache_splits(ctx.task_id)
        ctx.messages[:] = rebuilt
        sanction_rewrite(ctx.tools._ctx, "compaction")
        ctx.tools._ctx.messages = ctx.messages
        _loop().seal_task_transcript(ctx.messages)
        prune_reclaim_trace_refs(ctx.tools._ctx, ctx.messages)
    # Low-water facts: the pass is deficit-triggered but sized to land below the
    # boundary, so the landing is re-measured on the SAME fit basis as the trigger
    # and "reached the boundary" stays distinct from "achieved the margin"
    # (reclaimed == deficit is AT the boundary, not below it).
    requested_margin = int(request.reclaim_goal_tokens) - deficit
    landed = measurement
    if receipt.status == "applied":
        try:
            remeasured = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True)
        except Exception:  # telemetry only: an unmeasurable landing must not fail the pass
            log.debug("Post-reclaim fit measurement unavailable", exc_info=True)
            remeasured = None
        landed = remeasured.measurement if remeasured is not None else None
    boundary = [value for value in (
        landed.target_total_tokens, landed.capacity_total_tokens) if value is not None] if landed else []
    # None = the landing could not be measured (unknown), never "not reached".
    headroom = (min(boundary) - (landed.estimated_input_tokens + landed.response_reserve_tokens)
                if boundary else None)
    tool_ctx = ctx.tools._ctx
    previous_round = getattr(tool_ctx, "_context_reclaim_last_pass_round", None)
    tool_ctx._context_reclaim_last_pass_round = int(ctx.round_idx)
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "type": "context_reclaim",
        "checkpoint_kind": "context_reclaim_automatic",
        "round": ctx.round_idx,
        "route_fp": measurement.route_fp,
        "round_id": measurement.round_id,
        "status": receipt.status,
        "reclaim_goal_tokens": request.reclaim_goal_tokens,
        "reclaimed_tokens": receipt.reclaimed_tokens,
        "goal_reached": receipt.goal_reached,
        "checkpoint_ref": receipt.checkpoint_ref,
        "reclaim_fit": receipt.fit,
        "deficit_tokens": deficit,
        "requested_margin_tokens": requested_margin,
        "achieved_headroom_tokens": headroom,
        "boundary_reached": None if headroom is None else headroom >= 0,
        "below_boundary": None if headroom is None else headroom >= max(1, requested_margin),
        "rounds_since_previous_pass": (
            int(ctx.round_idx) - int(previous_round) if previous_round is not None else None),
    })
    if (receipt.fit or {}).get("reason") == "automatic_reclaim_unreachable":
        # One anchored notice per route/boundary; repeated impossible rounds
        # update the existing checkpoint rail rather than growing the transcript.
        marker = (f"[Context reclaim facts: {measurement.route_fp}; "
                  f"target={measurement.target_total_tokens}; capacity={measurement.capacity_total_tokens}]")
        if not any(message.get("role") == "user" and isinstance(message.get("content"), str)
                   and message["content"].startswith(marker) for message in ctx.messages):
            ctx.messages.append({"role": "user", "content": (
                f"{marker} At round {ctx.round_idx}, on the {measurement.measurement_basis} estimate "
                f"(density {measurement.measurement_density}), removing all eligible exposed sources could "
                f"free at most {receipt.fit['maximum_reclaim_tokens']} tokens, below the triggering "
                f"deficit of {deficit}. Input was {measurement.estimated_input_tokens} tokens plus "
                f"{measurement.response_reserve_tokens} reserved for output. The host kept earlier records "
                "and unconsumed sources and skipped the helper call. These are estimates, not a provider refusal. "
                "Choose how to reshape your working view or recover sources; later measurements are in the "
                "task checkpoints. This is a host fact, not an owner instruction.")})
    return receipt


def _measure_after_reclaim(ctx: _RoundModelCallContext) -> Any:
    """Suppress a second pass while reporting whether a summarizer actually ran."""
    disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True)
    if disposition is None:
        return None
    key = _fit_key(disposition)
    used = key in _loop()._context_reclaim_materializations(ctx.tools._ctx)
    if disposition.automatic_pass_used != used:
        disposition = replace(disposition, automatic_pass_used=used)
        _remember_main_fit(ctx, disposition)
    return disposition


def _reproject_actual_overflow_low(ctx: _RoundModelCallContext) -> None:
    """An actual overflow lowers Max to task-local Low; Low stays Low and Nano stays Nano (never raised)."""
    if ctx.active_context_mode != "max" or ctx.context_fit_plan is None:
        return
    ctx.messages[:] = ctx.context_fit_plan.reproject_transcript(ctx.messages, "low")
    invalidate_task_cache_splits(ctx.task_id)
    ctx.active_context_mode = "low"
    ctx.tools._ctx.messages = ctx.messages
    ctx.tools._ctx.active_context_mode = "low"
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_fit_low_retry",
        "round": ctx.round_idx,
        "route_fp": str(getattr(ctx.context_fit_plan, "route_fp", "") or ""),
        "preferred_mode": str(getattr(ctx.context_fit_plan, "preferred_mode", "") or ""),
        "effective_mode": "low",
        "owner_visible": True,
    })


def _refused_candidate(facts: Dict[str, Any]) -> Any:
    """The local lane's pre-dispatch refusal as the comparison candidate of its own round."""
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


def _strict_context_shrink_predicate(failed: Any) -> Callable[[Any], bool]:
    """Admit only a strictly smaller candidate of the same provider/model and round.

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
            and current_context is not None
            and failed_context is not None
            and current_context.round_id == failed_context.round_id
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
    """Show the model, at this boundary, the routing acts recorded for the owner message(s) it answers.

    The decision turn's metadata holds only what was recorded before it started;
    an act taken since — this turn's own promote or steer, a picker click, another
    lane on the same message — reaches the next request as one append-only host
    row, read from the existing receipt rail (``project_dialogue`` annotations) and
    only when it changed what the model has already read. Facts, not a ban. An act
    recorded under its own synthetic ``agent-steer:`` id, or keyed only to a task,
    carries no link to the message and is named as not listed rather than guessed.
    """
    tool_ctx = ctx.tools._ctx
    meta = getattr(tool_ctx, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    delivery = getattr(tool_ctx, "last_owner_delivery", None)
    ids = [str(meta.get("client_message_id") or ""),
           str(delivery.get("client_message_id") or "") if isinstance(delivery, dict) else ""]
    ids = [value for index, value in enumerate(ids) if value and value not in ids[:index]]
    root = meta.get("budget_drive_root") or ctx.drive_root or getattr(tool_ctx, "drive_root", None)
    if not ids or not root or str(meta.get("delegation_role") or "") == "subagent":
        return False
    try:
        from ouroboros.project_dialogue import _ANNOTATIONS_NAME, _latest_annotations_by_token

        rows = [row for (message_id, _token), row in _latest_annotations_by_token(
            pathlib.Path(str(root)) / "logs" / _ANNOTATIONS_NAME).items() if message_id in ids]
    except Exception:
        log.debug("routing receipts unreadable at the model boundary", exc_info=True)
        return False
    # One line per act from the receipt fields every reader shares, oldest first.
    where = lambda label, target: (  # noqa: E731
        f"{label} ({target})" if label and target and label != target else (label or target or "no target"))
    lines = lambda acts: [  # noqa: E731
        f"- {act.get('action') or '?'} → {where(str(act.get('target_label') or ''), str(act.get('target') or ''))}: "
        f"{act.get('status') or 'unknown'}, recorded {act.get('ts') or '?'}"
        for act in sorted(acts, key=lambda row: str(row.get("ts") or ""))]
    current = lines(rows)
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
        f"{ROUTING_RECEIPTS_HEADER} Routing acts recorded so far for the owner message(s) this turn answers, "
        "oldest first — receipts from the routing rail; facts, not a ban: another act stays your choice.",
        *current,
        "Not listed: an act recorded under its own agent-steer id (a steer after this message was already "
        "routed, a scope bind) or keyed only to a task id; it carries no link to this message."])})
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
    """Measure, dispatch, and recover one Main round.

    The measurement before the send is recorded as facts (``_remember_main_fit``; the
    per-round facts line shows them to the actor); a predicted deficit alone never starts
    a helper pass or a mode downgrade. Only the provider's typed refusal of the actual
    request opens the ordered recovery (``_recover_context_overflow``).
    """
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
    _project_wake_input(ctx)
    disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    if disposition is not None and _fit_key(disposition) in _loop()._context_reclaim_passes(ctx.tools._ctx):
        disposition = _measure_after_reclaim(ctx)  # a pass this round already ran: report it, never repeat it

    msg, cost = _loop()._dispatch_round_model(
        ctx,
        disposition,
        attempt_cap=ctx.attempt_cap,
    )
    refused = ctx.accumulated_usage.pop(REFUSED_CANDIDATE_KEY, None)  # a local pre-dispatch refusal's own facts
    from ouroboros.vision_routing import retry_refused_image_round  # the capture is read right after the send
    retried = None if msg is not None else retry_refused_image_round(ctx, _loop().last_physical_attempt_capture())
    if retried is not None:  # the retry ran: success stays on this route; a failure recovers as before
        return (*retried, ctx.active_context_mode)
    if msg is not None or str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") != "context_overflow":
        return msg, cost, ctx.active_context_mode

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


# Ordered recovery after a typed context refusal; unconfirmed exposure is a late
# source-only rescue, after ordinary same-model recovery and before configured fallback.
# Each rung is applied at most once per route/round and is followed by one strictly
# smaller retry against the latest refused candidate; prediction alone never opens it.
# The configured fallback chain and the named refusal with Continue follow in the caller.
OVERFLOW_RUNGS: Tuple[str, ...] = ("host_copies", "bodies", "helper", "memory", "low", "unseen_bodies")
WAKE_DELIVERY_STEP = "wake_delivery"  # the change applied before the ladder, when there was one


def _recover_context_overflow(ctx: _RoundModelCallContext, failed_capture: Any, cost: float, *,
                              prepared: bool = False) -> Tuple[Any, float, str]:
    """Walk the refusal ladder: each step that changed the candidate earns one strictly smaller retry.

    ``prepared`` says the caller already changed the candidate without a model (the
    wake's stored-source delivery): that change is retried first, then the rungs.
    """
    def _skipped(reason: str) -> Tuple[Any, float, str]:
        _emit_overflow_retry_skipped(ctx, reason)
        return None, cost, ctx.active_context_mode

    if not _failed_capture_is_comparable(failed_capture):
        return _skipped("failed_candidate_not_comparable")
    rungs = _loop()._context_overflow_retries(ctx.tools._ctx)
    fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    if fit is None:
        return None, cost, ctx.active_context_mode
    for rung in ((WAKE_DELIVERY_STEP,) if prepared else ()) + OVERFLOW_RUNGS:
        if rung != WAKE_DELIVERY_STEP:
            key = (fit.measurement.route_fp, fit.measurement.round_id, rung)
            if key in rungs:
                continue
            rungs.add(key)
            if not _apply_overflow_rung(ctx, rung, fit):
                continue
        fit = _measure_after_reclaim(ctx)
        if fit is None:
            return None, cost, ctx.active_context_mode
        try:
            retry_msg, retry_cost = _loop()._dispatch_round_model(
                ctx, fit, attempt_cap=1,
                candidate_predicate=_strict_context_shrink_predicate(failed_capture),
                max_tokens=int(getattr(failed_capture, "max_completion_tokens", 0) or 0) or None,
            )
        except PhysicalAttemptPreconditionFailed:
            _emit_overflow_retry_skipped(ctx, "context_candidate_not_strictly_smaller", rung=rung)
            continue
        cost = float(cost or 0.0) + float(retry_cost or 0.0)
        if retry_msg is not None or str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") != "context_overflow":
            return retry_msg, cost, ctx.active_context_mode  # answered, or another kind: ordinary recovery
        if isinstance(ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict):
            return _skipped("round_holds_unresolved_attempt")
        # Refused again: later rungs must shrink below THIS candidate, not the first one.
        refused = ctx.accumulated_usage.pop(REFUSED_CANDIDATE_KEY, None)
        latest = _refused_candidate(refused) if refused else _loop().last_physical_attempt_capture()
        if not _failed_capture_is_comparable(latest):
            return _skipped("failed_candidate_not_comparable")
        failed_capture = latest
        fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False) or fit
    return None, cost, ctx.active_context_mode


def _apply_overflow_rung(ctx: _RoundModelCallContext, rung: str, fit: Any) -> bool:
    """Apply one rung to the live transcript; True when the candidate actually changed."""
    if rung in ("host_copies", "bodies", "unseen_bodies"):
        return _run_emergency_address_pass(ctx, fit, rung=rung).status == "applied"
    if rung == "memory":
        return _coarsen_memory_view(ctx, fit)
    if rung == "helper":
        from ouroboros.context_fit import reclaim_low_water_margin

        landed = fit.measurement
        _loop()._context_reclaim_passes(ctx.tools._ctx).discard(_fit_key(fit))  # the refusal reopens the helper
        receipt = _loop()._run_main_reclaim(ctx, fit, minimum_goal_tokens=max(
            1, reclaim_low_water_margin(landed.target_total_tokens, landed.capacity_total_tokens)),
            provider_refused=True)
        return receipt is not None and receipt.status == "applied"
    if rung == "low":
        before = ctx.active_context_mode
        _reproject_actual_overflow_low(ctx)
        return ctx.active_context_mode != before
    return False
