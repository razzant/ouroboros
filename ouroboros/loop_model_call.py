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
from typing import Any, Callable, Dict, List, Optional, Tuple
from ouroboros import task_pacing
from ouroboros.context_budget import ContextReclaimRequest
from ouroboros.context_compaction import context_reclaim_transcript_sha256
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY, _TRANSPORT_DEATH_RETRIES
from ouroboros.loop_tool_execution import prune_reclaim_trace_refs, reclaim_negative_memo, reclaim_trace_refs
from ouroboros.observability import new_execution_id
from ouroboros.tools.registry import ToolRegistry
from ouroboros.loop_memory import (
    _defer_memory_refusal, _main_frame_bytes, _may_repair_main_memory,
    _recover_deferred_memory_refusal, _prepare_first_main_memory, _memory_recovery_held,
    _fit_existing_refused_memory,
    _reproject_actual_overflow_low as _reproject_actual_overflow_low,
)
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


def _resume_memory_refusal(tools: Any, messages: list, route: tuple, usage: dict, schemas: list) -> tuple:
    """Finish one deferred memory repair within the existing route recovery."""
    model, use_local, plan, mode = route
    if not getattr(tools._ctx, "_deferred_memory_refusal", None) or _memory_recovery_held(usage):
        return None, *route
    previous_fit = _snapshot_context_fit_usage(usage)
    recovered, msg, _cost = _recover_deferred_memory_refusal(tools)
    if recovered is not None and msg is None and usage.get("_last_llm_error_kind") == "context_overflow":
        msg, recovered.active_model, recovered.active_use_local, recovered.context_fit_plan, recovered.active_context_mode = (
            _loop()._run_cross_model_fallback_chain(
                llm=recovered.llm, ctx=tools._ctx, tools=tools, messages=recovered.messages,
                active_model=recovered.active_model, active_use_local=recovered.active_use_local,
                tool_schemas=schemas, active_effort=recovered.active_effort, max_retries=recovered.max_retries,
                drive_logs=recovered.drive_logs, task_id=recovered.task_id, round_idx=recovered.round_idx,
                event_queue=recovered.event_queue, accumulated_usage=usage, task_type=recovered.task_type,
                emit_progress=recovered.emit_progress or (lambda *_a, **_k: None),
                context_fit_plan=recovered.context_fit_plan, active_context_mode=recovered.active_context_mode,
                recovery_only=True))
    if recovered is not None and msg is not None:
        from ouroboros.model_wait import current_model_wait
        from ouroboros.model_slots import route_binding
        waiter = current_model_wait()
        ctx, primary = tools._ctx, getattr(tools._ctx, "primary_route", None)
        role = str(getattr(recovered.context_fit_plan, "model_role", "") or "main")
        binding = route_binding(recovered.active_model, recovered.active_use_local, role,
                                overrides=waiter.overrides if waiter else None)
        is_primary = bool(primary and binding == route_binding(primary["model"], bool(primary["use_local"]),
                          primary["role"], overrides=waiter.overrides if waiter else None))
        ctx._route_facts_pending = "" if is_primary else _route_facts_text(
            ctx, model=recovered.active_model, use_local=recovered.active_use_local, role=role,
            failed_model=model, failure={"failure_code": "context_overflow"}, waiter=waiter)
        if not is_primary:
            ctx.route_wait_on_primary = False
        return msg, *_adopt_fallback_route(ctx, tools, recovered.active_model, recovered.active_use_local,
            messages, recovered.messages, recovered.context_fit_plan, recovered.active_context_mode,
            schemas, usage, handover_from_model=model, handover_reason="context_overflow")
    # Failed repair never adopts its private view. Restore only projection
    # facts; actual error, cost, physical capture and unknown custody remain.
    tools._ctx.context_fit_plan, tools._ctx.messages = plan, messages
    tools._ctx.active_model, tools._ctx.active_use_local = model, use_local
    tools._ctx.active_context_mode = mode
    _restore_context_fit_usage(usage, previous_fit)
    return None, *route


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


def _cool_refused_route(model: str, use_local: bool, role: str, usage: dict, waiter: Any) -> None:
    """Apply the existing cooldown policy to the full refused binding."""
    from ouroboros import fallback_cooldown
    from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS
    from ouroboros.model_slots import route_binding
    if str(usage.get("_last_llm_error_kind") or "") in _COOLDOWN_ERROR_KINDS:
        fallback_cooldown.mark_cooldown(*route_binding(model, use_local, role,
            overrides=waiter.overrides if waiter else None))


def _disclose_route_unknown(messages: list, usage: dict, model: str) -> None:
    """A new configured-route generation carries the still-unknown attempt facts."""
    from ouroboros.loop_transport import append_unknown_recovery_input
    if usage.get("_pending_transport_outcome") or usage.get("_last_llm_error_kind") == "provider_outcome_unknown":
        append_unknown_recovery_input(messages, usage, dict(usage.get("_pending_transport_outcome") or {}),
            lead=f"The previous attempt ended without a usable answer; the configured route {model} continues.",
            continuation="configured_route")


def _run_cross_model_fallback_chain(
    *, llm, ctx, tools, messages, active_model, active_use_local, tool_schemas,
    active_effort, max_retries, drive_logs, task_id, round_idx, event_queue,
    accumulated_usage, task_type, emit_progress, context_fit_plan,
    active_context_mode, recovery_only=False,
) -> tuple:
    """Try the configured routes other than the acting binding before any wait begins.

    An eligible unknown outcome moves on to the next route with facts-only recovery
    input (a real, possibly charged generation); an ineligible one (an accepted operation still readable,
    inline Presence) or a spent deadline stops the walk. A round whose own failure is an
    outage or an unknown outcome keeps that failure as its wait when none answers.
    """
    from ouroboros import fallback_cooldown as _fcd
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option, task_model_binding, route_binding
    from ouroboros.model_wait import current_model_wait
    from ouroboros.provider_models import provider_for_model

    waiter = current_model_wait()
    _cool_refused_route(active_model, active_use_local, str(getattr(context_fit_plan, "model_role", "") or "main"), accumulated_usage, waiter)
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
        _disclose_route_unknown(messages, accumulated_usage, fallback_model)
        # Cross-FAMILY fallback must not replay the primary's
        # provider-private reasoning to a different family (the GLM->Claude
        # 400 "Invalid signature" death); the SSOT sanitizer no-ops same-family.
        fallback_messages = LLMClient.sanitize_reasoning_on_model_switch(messages, active_model, fallback_model)
        # Bind exact route evidence and choose its deterministic projection
        # BEFORE physical dispatch: the fallback's first request must not
        # inherit the failed primary route's Max projection/fingerprint. It
        # then uses the ordinary confirmed-overflow recovery owner.
        candidate_plan, candidate_mode = _loop()._rebind_context_fit_plan(
            context_fit_plan,
            tools,
            fallback_messages,
            model=fallback_model,
            use_local=fallback_use_local,
            preferred_mode=str(
                active_context_mode if recovery_only else getattr(context_fit_plan, "preferred_mode", "") or active_context_mode
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
                recovery_only=recovery_only,
                model_role=fallback_role,
                emit_progress=emit_progress,
                defer_resource_wait=(task_type == "presence" or owner_question
                                     or _route_follows(rows[index + 1:])),
            )
        tried.append(fallback_model)
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
        if _walk_fenced(tools._ctx, accumulated_usage):
            break
        _cool_refused_route(fallback_model, fallback_use_local, fallback_role, accumulated_usage, waiter)
        previous_model, previous_tag = fallback_model, ftag
    if (msg is None and entry_kind == "context_overflow"
            and accumulated_usage.get("_last_llm_error_kind") == "transport_unavailable"
            and not _memory_recovery_held(accumulated_usage)):
        # Only the finished fallback walk is classified here. A later repaired
        # acting-route send owns its own outcome, including a genuine outage.
        accumulated_usage["_last_llm_error_kind"] = "context_overflow"
    fenced = msg is None and _walk_fenced(tools._ctx, accumulated_usage)
    if deferred is None:
        deferred = getattr(tools._ctx, "_deferred_resource_refusal", None)
    owner_question = (task_type != "presence" and deferred is not None and not own_wait
                      and waiter is not None and waiter.waits_allowed)
    accumulated_usage.pop(RESOURCE_REFUSAL_KEY, None)
    if msg is not None:
        tools._ctx._deferred_memory_refusal = None
    elif getattr(tools._ctx, "_deferred_memory_refusal", None) is not None and not _memory_recovery_held(accumulated_usage):
        msg, active_model, active_use_local, context_fit_plan, active_context_mode = _resume_memory_refusal(
            tools, messages, (active_model, active_use_local, context_fit_plan, active_context_mode),
            accumulated_usage, tool_schemas)
        if msg is None and _memory_recovery_held(accumulated_usage):
            return msg, active_model, active_use_local, context_fit_plan, active_context_mode
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
        _disclose_route_unknown(retry_call.messages, accumulated_usage, retry_call.active_model)
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
            preferred_mode=retry_call.active_context_mode, tool_schemas=tool_schemas,
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
        if msg is None and not _walk_fenced(ctx, usage) and (kind in _ROUND_WAIT_KINDS or usage.get("_pending_transport_outcome")):
            # The round's own outage or unknown outcome owns its wait: a later
            # candidate's failure never re-aims it (nor the probe's expected route).
            outstanding = usage.get("_pending_transport_outcome") or pending
            usage["_last_llm_error_kind"] = "provider_outcome_unknown" if outstanding else kind
            if outstanding:
                usage["_pending_transport_outcome"] = outstanding
    if msg is not None:
        ctx._deferred_memory_refusal = None
    elif getattr(ctx, "_deferred_memory_refusal", None) is not None and not _memory_recovery_held(usage):
        msg, model, use_local, context_fit_plan, active_context_mode = _resume_memory_refusal(
            tools, limit_ctx.messages, (model, use_local, context_fit_plan, active_context_mode),
            usage, limit_ctx.tool_schemas)
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
    preferred_mode: str,
    tool_schemas: List[Dict[str, Any]],
    model_role: str = "",
    model_route: Optional[Dict[str, Any]] = None,
    credential_profile_id: Optional[str] = None,
) -> Tuple[Any, str]:
    if plan is None or not all(
        hasattr(plan, name) for name in ("max_projection", "low_projection", "core_sha256")
    ):
        raise RuntimeError(
            "CONTEXT_FIT_REBUILD_FAILED: immutable context core is unavailable for route switch"
        )
    from ouroboros.capability_evidence import is_known
    from ouroboros.context import _context_fit_route
    from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS, OWNER_NANO_TARGET_TOKENS
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
    output_reserve = main_output_reserve_tokens(use_local=bool(route.get("use_local", use_local)))

    def project(projection: Any) -> Any:
        content_json, estimated = projection.system_content_json, projection.estimated_tokens
        memory_facts = dict(getattr(projection, "memory_facts", {}) or {})
        snapshot = getattr(plan, "chronicle_state_json", "")
        template = getattr(plan, "system_templates_json", {}).get(projection.mode)
        if snapshot and template:
            from ouroboros.chronicle_view import render_system_view
            from ouroboros.context_fit import estimate_context_prompt_tokens

            content = render_system_view(
                json.loads(template), snapshot, mode=projection.mode,
                window_tokens=window_tokens if known_window else 0, calibration_ratio=ratio,
                output_reserve_tokens=output_reserve,
                task={**plan.context_task, "owner_context_mode": plan.preferred_mode},
                facts_out=memory_facts,
            )
            content_json = json.dumps(content, ensure_ascii=False, sort_keys=True)
            estimated = estimate_context_prompt_tokens([
                {"role": "system", "content": content},
                {"role": "user", "content": json.loads(projection.user_content_json or plan.user_content_json)},
            ])
        calibrated = int(int(estimated or 0) * ratio)
        nano = projection.mode == "nano"
        reserve = NANO_MIN_HEADROOM_TOKENS if nano else output_reserve
        capacity = min(OWNER_NANO_TARGET_TOKENS, window_tokens) if nano else window_tokens
        fits = (
            calibrated + reserve <= capacity
            if known_window else None
        )
        return replace(
            projection,
            system_content_json=content_json,
            estimated_tokens=estimated,
            calibrated_tokens=calibrated,
            calibration_ratio=ratio,
            fits_known_window=fits,
            memory_facts=memory_facts,
        )

    max_projection = project(plan.max_projection)
    low_projection = project(plan.low_projection)
    nano_projection = (
        project(plan.nano_projection)
        if getattr(plan, "nano_projection", None) is not None else None
    )
    # A route/account change does not become a new owner context-mode choice.
    # The active mode also survives cold continuation in its existing route record.
    preferred = str(getattr(plan, "preferred_mode", "") or preferred_mode)
    preferred = preferred if preferred in {"max", "low", "nano"} else "max"
    modes = {"max": 0, "low": 1, "nano": 2}
    initial_mode = max((mode for mode in (
        preferred, preferred_mode, getattr(plan, "rendered_mode", ""),
        getattr(tools._ctx, "active_context_mode", ""),
    ) if mode in modes), key=modes.get)
    rebound = replace(
        plan,
        preferred_mode=preferred,
        initial_mode=initial_mode,
        rendered_mode=initial_mode,
        model=str(route.get("model") or model),
        provider=str(route.get("provider") or ""),
        route_fp=str(getattr(evidence, "route_fp", "") or ""),
        status=str(getattr(evidence, "status", "") or ""),
        stale=bool(getattr(evidence, "stale", False)),
        window_tokens=window_tokens,
        output_reserve_tokens=output_reserve,
        max_projection=max_projection,
        low_projection=low_projection,
        nano_projection=nano_projection,
        model_role=task["model_role"],
        model_route={
            "source": str(getattr(evidence, "source_id", "") or ""),
            "model": parse_claudexor_model(str(route.get("model") or model))[1],
            "credentialProfileId": str(getattr(evidence, "credential_profile_id", "") or ""),
            "accountFingerprint": str(getattr(evidence, "account_fingerprint", "") or ""),
        } if route.get("provider") == "claudexor" else {},
        evidence_source=str(getattr(evidence, "source", "") or ""),
    )
    mode = initial_mode
    projected_prompt_tokens = rebound.projected_tokens_with_tools(mode, tool_schemas)
    messages[:] = rebound.reproject_transcript(messages, mode)
    invalidate_task_cache_splits(getattr(tools._ctx, "task_id", ""))
    tools._ctx.context_fit_plan = rebound
    tools._ctx.messages = messages
    tools._ctx.active_context_mode = mode
    try:
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
    # The first call's already prepared vision/executor additions, with its
    # sizing clock. Reprojection reuses this observation without another helper.
    prepared_main_frame: Optional[List[Dict[str, Any]]] = None
    recovery_only: bool = False  # Reuse the final reduced view, without another preparation cycle.


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
    if rendered_mode == "nano":
        return "owner_nano"
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
    _refresh_main_marks(ctx)
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


def _refresh_main_marks(ctx: _RoundModelCallContext) -> None:
    """Carry current explicit marks into every shared Main route's measured view."""
    plan = ctx.context_fit_plan
    if (not getattr(plan, "chronicle_state_json", "")
            or TRANSPORT_DEATHS_KEY in ctx.accumulated_usage
            or ctx.accumulated_usage.get("_pending_transport_outcome")
            or ctx.accumulated_usage.get("_last_llm_error_kind") == "provider_outcome_unknown"):
        return
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.tool_access import canonical_data_root
    from sqlite3 import Error as SQLiteError

    snapshot = json.loads(plan.chronicle_state_json)
    root = canonical_data_root(ctx.tools._ctx)
    status = None
    try:
        marks = ChronicleStore(root).active_marks(str(snapshot.get("focus", "1")))
    except (OSError, ValueError, SQLiteError) as exc:
        marks = snapshot.get("marks", [])
        status = {"kind": "active_marks_unavailable", "cause": type(exc).__name__}
    refreshed = plan.with_active_marks(marks, status)
    if refreshed is plan:
        return
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = refreshed
    ctx.messages[:] = refreshed.reproject_transcript(ctx.messages, ctx.active_context_mode)
    ctx.tools._ctx.messages = ctx.messages
    invalidate_task_cache_splits(ctx.task_id)
    sanction_rewrite(ctx.tools._ctx, "memory_marks")
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_marks_refreshed", "round": ctx.round_idx,
        "active_mark_ids": [mark["id"] for mark in marks], "source_status": status or {"kind": "observed"},
        "core_sha256": refreshed.core_sha256,
    })


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
) -> Tuple[Any, float]:
    from ouroboros.model_wait import current_model_wait
    from ouroboros.loop_transport import emit_model_substitution, transport_repeat_stop_requested
    from ouroboros.owner_mailbox import OwnerMailboxPeek
    from ouroboros.send_clock import main_clock_policy

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
            prepare_main_context=(lambda prepared: _prepare_first_main_memory(ctx, prepared))
                if ctx.round_idx == 1 and not ctx.accumulated_usage.get("rounds")
                and getattr(plan, "chronicle_state_json", "") else None,
        )
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
    observed = ctx.accumulated_usage.get("_model_route")
    if (plan is not None and isinstance(observed, dict)
            and observed != getattr(plan, "model_route", {})):
        # A response/error may expose an Auto rotation after preparation. Withdraw
        # the prior account's capacity before another physical call is prepared.
        ctx.context_fit_plan, ctx.active_context_mode = _loop()._rebind_context_fit_plan(
            ctx.context_fit_plan, ctx.tools, ctx.messages, model=ctx.active_model,
            use_local=ctx.active_use_local, preferred_mode=ctx.active_context_mode,
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
        preferred_mode=ctx.active_context_mode, tool_schemas=ctx.tool_schemas,
        model_role=role, model_route=observed or {},
        credential_profile_id=kwargs.get("model_account_override"))
    ctx.active_model, ctx.active_use_local = model, use_local
    ctx.tools._ctx.active_model = model
    ctx.tools._ctx.active_use_local = use_local
    trace = getattr(ctx.tools._ctx, "_execution_trace", {})
    _pending_model_wait_handover(
        ctx.tools._ctx,
        from_model=previous_model,
        to_model=str(model or ""),
        tool_calls=len(trace.get("tool_calls") or []) if isinstance(trace, dict) else 0,
    )
    _project_wake_input(ctx)
    disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    from ouroboros.send_clock import MainSendClock

    if (disposition is not None and disposition.action == "reclaim_once"
            and not (ctx.round_idx == 1 and not ctx.accumulated_usage.get("rounds")
                     and getattr(ctx.context_fit_plan, "chronicle_state_json", ""))):
        if _fit_key(disposition) not in _loop()._context_reclaim_passes(ctx.tools._ctx):
            with bind_physical_attempt_context(None), MainSendClock(None).bound():
                _loop()._run_main_reclaim(ctx, disposition)
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
        if ctx.round_idx == 1 and not ctx.accumulated_usage.get("rounds") and getattr(ctx.context_fit_plan, "chronicle_state_json", ""):
            kwargs["messages"], prepared_context = _prepare_first_main_memory(ctx, kwargs["messages"])
        else:
            prepared_context = _physical_context_for_fit(disposition) if disposition else None
    from ouroboros.llm_claudexor import cache_key_for_model
    from ouroboros.provider_models import provider_for_model
    kwargs["cache_affinity"] = "" if use_local else cache_key_for_model(model)
    kwargs["allow_server_web_search"] = (_loop()._server_web_allowed_by_task(ctx.tools._ctx)
                                         and not use_local and provider_for_model(model) != "claudexor")
    if provider_for_model(model) == "claudexor":
        kwargs["bypass_response_cache"] = False
    return PreparedModelCall(kwargs, prepared_context,
                             current_physical_attempt_predicate())


def _run_main_reclaim(
    ctx: _RoundModelCallContext,
    disposition: Any,
    *,
    minimum_goal_tokens: int = 0,
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



def _failed_capture_is_comparable(capture: Any) -> bool:
    return bool(
        capture is not None
        and capture.state in {"dispatched", "settled", "unresolved"}
        and capture.candidate_measurement_kind == "canonical_json_v1"
        and capture.candidate_raw_sha256
        and capture.candidate_context_size_bytes is not None
        and capture.physical_context is not None
    )


def _strict_context_shrink_predicate(failed: Any) -> Callable[[Any], bool]:
    def predicate(request: Any) -> bool:
        failed_context = failed.physical_context
        current_context = request.physical_context
        return bool(
            request.candidate_measurement_kind == "canonical_json_v1"
            and request.provider == failed.provider
            and request.model == failed.model
            and request.max_completion_tokens == failed.max_completion_tokens
            and current_context is not None
            and failed_context is not None
            and current_context.route_fp == failed_context.route_fp
            and current_context.round_id == failed_context.round_id
            and request.candidate_raw_sha256 != failed.candidate_raw_sha256
            and request.candidate_context_size_bytes is not None
            and int(request.candidate_context_size_bytes) < int(failed.candidate_context_size_bytes)
        )

    return predicate


def _emit_overflow_retry_skipped(ctx: _RoundModelCallContext, reason: str) -> None:
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_overflow_retry_skipped",
        "round": ctx.round_idx,
        "route_fp": str(getattr(ctx.context_fit_plan, "route_fp", "") or ""),
        "reason": reason,
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


def _call_round_model(ctx: _RoundModelCallContext) -> Tuple[Any, float, str]:
    """Measure, optionally reclaim, dispatch, and recover one Main round."""
    facts = getattr(ctx.tools._ctx, "_route_facts_pending", "")
    if facts and ctx.defer_resource_wait is None:  # the acting route's first own round after a switch
        ctx.tools._ctx._route_facts_pending = ""
        _loop()._append_or_merge_user_message(ctx.messages, facts)
    if ctx.defer_resource_wait is None:
        ctx.tools._ctx._deferred_memory_refusal = None
        ctx.tools._ctx._context_refusal_captures = {}
    _append_routing_receipts(ctx)
    _project_wake_input(ctx)
    disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    initial_memory = (disposition is not None and ctx.round_idx == 1 and not ctx.accumulated_usage.get("rounds")
                      and bool(getattr(ctx.context_fit_plan, "chronicle_state_json", "")))
    if disposition is not None and not initial_memory and not getattr(ctx, "recovery_only", False):
        key = _fit_key(disposition)
        already_reclaimed = key in _loop()._context_reclaim_passes(ctx.tools._ctx)
        if disposition.action == "reclaim_once" and not already_reclaimed:
            _loop()._run_main_reclaim(ctx, disposition)
            already_reclaimed = True
        if already_reclaimed:
            disposition = _measure_after_reclaim(ctx)

    captures = getattr(ctx.tools._ctx, "_context_refusal_captures", {})
    prior = captures.get(_fit_key(disposition)) if disposition is not None and getattr(ctx, "recovery_only", False) else None
    try:
        msg, cost = _loop()._dispatch_round_model(ctx, disposition, attempt_cap=ctx.attempt_cap,
            **({"candidate_predicate": _strict_context_shrink_predicate(prior)} if prior is not None else {}))
    except PhysicalAttemptPreconditionFailed:
        _emit_overflow_retry_skipped(ctx, "context_candidate_not_strictly_smaller")
        return None, 0.0, ctx.active_context_mode
    if getattr(ctx, "recovery_only", False):
        return msg, cost, ctx.active_context_mode
    if msg is not None or str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") != "context_overflow":
        return msg, cost, ctx.active_context_mode

    # Snapshot immediately: a reclaim summarizer is itself physically receipted
    # and would otherwise replace the failed Main candidate in the ContextVar.
    failed_capture = _loop().last_physical_attempt_capture()
    if disposition is not None and _failed_capture_is_comparable(failed_capture):
        captures[_fit_key(disposition)] = failed_capture
        ctx.tools._ctx._context_refusal_captures = captures
    refused_frame_bytes = _main_frame_bytes(ctx)
    refused_facts = ctx.context_fit_plan.projection(ctx.active_context_mode).memory_facts if ctx.context_fit_plan else {}
    refused_digest_ids = tuple(refused_facts.get("selected_digest_ids") or [])
    refused_memory_bytes = refused_facts.get("rendered_memory_bytes")
    if disposition is None:
        return msg, cost, ctx.active_context_mode

    def _skipped(reason: str) -> Tuple[Any, float, str]:
        _emit_overflow_retry_skipped(ctx, reason)
        return msg, cost, ctx.active_context_mode

    if isinstance(ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict):
        return _skipped("round_holds_unresolved_attempt")
    _project_wake_input(ctx, overflowed=True)
    _fit_existing_refused_memory(ctx)
    reclaim_key = _fit_key(disposition)
    overflow_fit = (
        _measure_after_reclaim(ctx)
        if reclaim_key in _loop()._context_reclaim_passes(ctx.tools._ctx)
        else _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    )
    if overflow_fit is None:
        return msg, cost, ctx.active_context_mode
    key = _fit_key(overflow_fit)
    if key not in _loop()._context_reclaim_materializations(ctx.tools._ctx):
        # A skipped economic pass did not consume the physical recovery work.
        _loop()._context_reclaim_passes(ctx.tools._ctx).discard(key)
    if key not in _loop()._context_reclaim_passes(ctx.tools._ctx):
        # The provider proved the prediction short by an unknown amount: request a
        # low-water-sized pass, never a token-sized one, so the single strict-shrink
        # retry has real headroom (the goal already carries the margin when the
        # measurement itself found a deficit).
        from ouroboros.context_fit import reclaim_low_water_margin

        landed = overflow_fit.measurement
        _loop()._run_main_reclaim(ctx, overflow_fit, minimum_goal_tokens=max(
            1, reclaim_low_water_margin(landed.target_total_tokens, landed.capacity_total_tokens)))
        overflow_fit = _measure_after_reclaim(ctx)
        if overflow_fit is None:
            return msg, cost, ctx.active_context_mode

    retries = _loop()._context_overflow_retries(ctx.tools._ctx)
    if key in retries:
        return _skipped("route_round_retry_already_used")
    if not _failed_capture_is_comparable(failed_capture):
        return _skipped("failed_candidate_not_comparable")
    if (ctx.active_context_mode == "max" or _may_repair_main_memory(ctx)) and _main_frame_bytes(ctx) >= refused_frame_bytes:
        # New biography work and book navigation wait for configured routes.
        # An already shorter Max/history view keeps its immediate retry below.
        _defer_memory_refusal(ctx, failed_capture, refused_digest_ids=refused_digest_ids,
                              refused_memory_bytes=refused_memory_bytes)
        return msg, cost, ctx.active_context_mode
    retries.add(key)
    try:
        retry_msg, retry_cost = _loop()._dispatch_round_model(
            ctx,
            overflow_fit,
            attempt_cap=1,
            candidate_predicate=_strict_context_shrink_predicate(
                failed_capture,
            ),
        )
    except PhysicalAttemptPreconditionFailed:
        _defer_memory_refusal(ctx, failed_capture, refused_digest_ids=refused_digest_ids,
                              refused_memory_bytes=refused_memory_bytes)
        return _skipped("context_candidate_not_strictly_smaller")
    if retry_msg is None and ctx.accumulated_usage.get("_last_llm_error_kind") == "context_overflow":
        _defer_memory_refusal(ctx, _loop().last_physical_attempt_capture())
    return retry_msg, retry_cost, ctx.active_context_mode
