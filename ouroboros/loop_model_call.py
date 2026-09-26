"""The per-round model call: context-fit identification, measurement and memory,
dispatch, the main-context reclaim, overflow-retry predicates, the cross-model
fallback chain and context-fit plan rebinding. Extracted from loop.py (v7 L-B
split); loop.py re-exports every name."""

from __future__ import annotations

from ouroboros.config import runtime_setting

import logging
import contextlib
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
from ouroboros.transcript_prefix import sanction_rewrite
from ouroboros.usage_accounting import PhysicalAttemptContext, PhysicalAttemptPreconditionFailed, invalidate_task_cache_splits


log = logging.getLogger("ouroboros.loop")
# The typed, temporary non-success of a round whose resource refusal no route recovered and
# no owner wait follows (inline Presence, or a chain stopped by its own fence): nothing more is sent.
RESOURCE_REFUSAL_KEY = "resource_refusal"
_CHAIN_STOP_KINDS = ("provider_outcome_unknown", "deadline_exhausted", "transport_unavailable")


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


def _run_cross_model_fallback_chain(
    *, llm, ctx, tools, messages, active_model, active_use_local, tool_schemas,
    active_effort, max_retries, drive_logs, task_id, round_idx, event_queue,
    accumulated_usage, task_type, emit_progress, context_fit_plan,
    active_context_mode,
) -> tuple:
    """Try fallbacks; unknown dispatch stops the chain."""
    from ouroboros import fallback_cooldown as _fcd
    from ouroboros.config import fallback_candidate_targets
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option, parse_fallback_chain, task_model_binding
    from ouroboros.model_wait import current_model_wait
    from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS as _cooldown_kinds
    from ouroboros.provider_models import provider_for_model

    def _cooled(model: str, use_local: bool) -> None:
        if str(accumulated_usage.get("_last_llm_error_kind") or "") in _cooldown_kinds:
            _fcd.mark_cooldown(model, use_local)

    _cooled(active_model, active_use_local)
    primary_context_usage = _snapshot_context_fit_usage(accumulated_usage)
    fallback_use_local = runtime_setting("USE_LOCAL_FALLBACK", "").lower() in ("true", "1")
    attempt_cap = _fcd.attempts_per_model()
    waiter = current_model_wait()
    configured_chain = parse_fallback_chain()
    # A primary quota refusal returned here, not waited: the configured routes come
    # first, and the owner question is that refusal's own wait, asked after them.
    deferred, tools._ctx._deferred_resource_refusal = getattr(tools._ctx, "_deferred_resource_refusal", None), None
    owner_question = (task_type != "presence" and deferred is not None
                      and waiter is not None and waiter.waits_allowed)
    deferred_candidate = None  # The route that actually refused; never replay an unrelated primary.
    candidates, tried = fallback_candidate_targets(active_model), []
    # The notice names the model that was actually just tried. `active_model`
    # stays the primary until a candidate succeeds, so a second switch would
    # otherwise read "primary -> B" beside B's predecessor's failure reason.
    previous_model, previous_tag = active_model, " (local)" if active_use_local else ""
    msg = None
    # ABI-4: the candidate ladder arrives as typed ResolvedModelTarget values;
    # `.model_id` is read once here and crosses to strings only at the LLM
    # transport boundary (the chat API's model parameter). The local-vs-remote
    # dispatch lane is the single global USE_LOCAL_FALLBACK flag above (the
    # pre-existing chain contract): the ladder's `provider_route` stays the ""
    # sentinel rather than fabricating a per-candidate fact nothing consumes.
    for index, candidate in enumerate(candidates):
        fallback_model = candidate.model_id
        # The role belongs to the configured row, before active-model removal
        # and deduplication. Those filters must not shift its account binding.
        fallback_role = f"fallback:{configured_chain.index(fallback_model)}"
        if _fcd.is_cooling_down(fallback_model, fallback_use_local):
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
                      f"{account_note}"
                      f"{f'; reason: {reason}' if reason else ''}{'; pinned account: siblings were not tried' if account_route and fallback_account else ''}",
                      incident={"task_incident": "model_lane_switch", "toast_once": f"{task_id}:model_lane_switch:{round_idx}:{fallback_model}"})
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
                                     or _route_follows(candidates[index + 1:], fallback_use_local)),
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
            owner_question = (task_type != "presence" and deferred is not None
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
            break
        tools._ctx.context_fit_plan = context_fit_plan
        tools._ctx.messages = messages
        tools._ctx.active_context_mode = active_context_mode
        _restore_context_fit_usage(accumulated_usage, primary_context_usage)
        if str(accumulated_usage.get("_last_llm_error_kind") or "") in _CHAIN_STOP_KINDS:
            break
        _cooled(fallback_model, fallback_use_local)
        previous_model, previous_tag = fallback_model, ftag
    if deferred is None:
        deferred = getattr(tools._ctx, "_deferred_resource_refusal", None)
    owner_question = (task_type != "presence" and deferred is not None
                      and waiter is not None and waiter.waits_allowed)
    accumulated_usage.pop(RESOURCE_REFUSAL_KEY, None)
    if msg is None and owner_question and str(accumulated_usage.get("_last_llm_error_kind") or "") not in _CHAIN_STOP_KINDS:
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
        # An owner may select a DIFFERENT account on either the primary or a
        # fallback model. Rebind the retained call before measurement and physical
        # send; the pre-wait account's fingerprint/capacity is not evidence for B.
        from ouroboros.model_slots import task_model_binding

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
        else:
            active_model, active_use_local = retry_call.active_model, retry_call.active_use_local
            context_fit_plan = retry_call.context_fit_plan
    elif msg is None and deferred is not None:  # the refused primary buys no forced final either
        accumulated_usage[RESOURCE_REFUSAL_KEY] = deferred.terminal(
            fallbacks_tried=tried, owner_wait="not_asked" if owner_question else "not_allowed")
    return (
        msg,
        active_model,
        active_use_local,
        context_fit_plan,
        active_context_mode,
    )


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
    from ouroboros.context_fit import _failed_route_evidence, _route_calibration_ratio
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

    def project(projection: Any) -> Any:
        calibrated = int(int(projection.estimated_tokens or 0) * ratio)
        nano = projection.mode == "nano"
        reserve = NANO_MIN_HEADROOM_TOKENS if nano else int(plan.output_reserve_tokens or 0)
        capacity = min(OWNER_NANO_TARGET_TOKENS, window_tokens) if nano else window_tokens
        fits = (
            calibrated + reserve <= capacity
            if known_window else None
        )
        return replace(
            projection,
            calibrated_tokens=calibrated,
            calibration_ratio=ratio,
            fits_known_window=fits,
        )

    max_projection = project(plan.max_projection)
    low_projection = project(plan.low_projection)
    nano_projection = (
        project(plan.nano_projection)
        if getattr(plan, "nano_projection", None) is not None else None
    )
    preferred = preferred_mode if preferred_mode in {"low", "max", "nano"} else "max"
    initial_mode = preferred
    rebound = replace(
        plan,
        preferred_mode=preferred,
        initial_mode=initial_mode,
        model=str(route.get("model") or model),
        provider=str(route.get("provider") or ""),
        route_fp=str(getattr(evidence, "route_fp", "") or ""),
        status=str(getattr(evidence, "status", "") or ""),
        stale=bool(getattr(evidence, "stale", False)),
        window_tokens=window_tokens,
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


def _route_follows(candidates: List[Any], use_local: bool) -> bool:
    from ouroboros import fallback_cooldown

    return any(not fallback_cooldown.is_cooling_down(item.model_id, use_local) for item in candidates)


def _fallback_route_follows(ctx: _RoundModelCallContext) -> bool:
    """Whether this primary round's fallback chain would still dial a configured route."""
    from ouroboros.config import fallback_candidate_targets

    if bool(getattr(ctx.tools._ctx, "exact_model_route", False)) or isinstance(
            ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict):
        return False
    use_local = runtime_setting("USE_LOCAL_FALLBACK", "").lower() in ("true", "1")
    return _route_follows(fallback_candidate_targets(ctx.active_model), use_local)


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
    from ouroboros.loop_transport import (
        emit_model_effort_mismatch, emit_model_substitution,
        managed_transport_continuation, transport_repeat_stop_requested,
    )
    from ouroboros.owner_mailbox import OwnerMailboxPeek

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
    from ouroboros.acceptance_settlement import expose_acceptance_feedback

    observe_feedback = lambda sent: expose_acceptance_feedback(
        getattr(ctx.tools._ctx, "_execution_trace", {}), sent, str(ctx.task_id))
    with contextlib.ExitStack() as binding:
        deferral = None
        if waiter is not None:
            binding.enter_context(waiter.register_reprepare(role, lambda kwargs: _reprepare_waiting_main(ctx, kwargs)))
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
            transport_death_retries=(_TRANSPORT_DEATH_RETRIES if attempt_cap is None
                                     and not managed_transport_continuation(ctx.tools._ctx) else 0),
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
        )
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
    emit_model_effort_mismatch(ctx.accumulated_usage, task_id=ctx.task_id,
                               emit_progress=getattr(ctx, "emit_progress", None))
    emit_model_substitution(ctx.accumulated_usage, task_id=ctx.task_id,
                            emit_progress=getattr(ctx, "emit_progress", None))
    call = ctx.accumulated_usage.get("_last_llm_call_meta")
    execution_id = ctx.accumulated_usage.get("execution_id")
    if (result[0] is not None and isinstance(call, dict) and call is not previous_call
            and execution_id and call.get("execution_id") == execution_id
            and call.get("round_id") == f"{execution_id}:round:{ctx.round_idx}"
            and call.get("llm_call_id")):
        call["usable_solve_response"] = True
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
    disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    if disposition is not None and disposition.action == "reclaim_once":
        if _fit_key(disposition) not in _loop()._context_reclaim_passes(ctx.tools._ctx):
            with bind_physical_attempt_context(None):
                _loop()._run_main_reclaim(ctx, disposition)
        disposition = _measure_after_reclaim(ctx)
    from ouroboros.loop_llm_call import _prepare_main_messages

    # Any vision work belongs to this actual reprepare, outside both the Main
    # fit probe and the failed attempt's physical precondition/measurement.
    with bind_physical_attempt_context(None):
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
    return PreparedModelCall(kwargs, _physical_context_for_fit(disposition) if disposition else None,
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
    })
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
    if ctx.active_context_mode == "low" or ctx.context_fit_plan is None:
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
        "type": "context_overflow_retry_skipped",
        "round": ctx.round_idx,
        "route_fp": str(getattr(ctx.context_fit_plan, "route_fp", "") or ""),
        "reason": reason,
    })


def _call_round_model(ctx: _RoundModelCallContext) -> Tuple[Any, float, str]:
    """Measure, optionally reclaim, dispatch, and recover one Main round."""
    disposition = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    if disposition is not None:
        key = _fit_key(disposition)
        already_reclaimed = key in _loop()._context_reclaim_passes(ctx.tools._ctx)
        if disposition.action == "reclaim_once" and not already_reclaimed:
            _loop()._run_main_reclaim(ctx, disposition)
            already_reclaimed = True
        if already_reclaimed:
            disposition = _measure_after_reclaim(ctx)

    msg, cost = _loop()._dispatch_round_model(
        ctx,
        disposition,
        attempt_cap=ctx.attempt_cap,
    )
    if msg is not None or str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") != "context_overflow":
        return msg, cost, ctx.active_context_mode

    # Snapshot immediately: a reclaim summarizer is itself physically receipted
    # and would otherwise replace the failed Main candidate in the ContextVar.
    failed_capture = _loop().last_physical_attempt_capture()
    if disposition is None:
        return msg, cost, ctx.active_context_mode

    def _skipped(reason: str) -> Tuple[Any, float, str]:
        _emit_overflow_retry_skipped(ctx, reason)
        return msg, cost, ctx.active_context_mode

    if isinstance(ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict):
        return _skipped("round_holds_unresolved_attempt")
    _reproject_actual_overflow_low(ctx)
    reclaim_key = _fit_key(disposition)
    overflow_fit = (
        _measure_after_reclaim(ctx)
        if reclaim_key in _loop()._context_reclaim_passes(ctx.tools._ctx)
        else _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    )
    if overflow_fit is None:
        return msg, cost, ctx.active_context_mode
    key = _fit_key(overflow_fit)
    if key not in _loop()._context_reclaim_passes(ctx.tools._ctx):
        _loop()._run_main_reclaim(ctx, overflow_fit, minimum_goal_tokens=1)
        overflow_fit = _measure_after_reclaim(ctx)
        if overflow_fit is None:
            return msg, cost, ctx.active_context_mode

    retries = _loop()._context_overflow_retries(ctx.tools._ctx)
    if key in retries:
        return _skipped("route_round_retry_already_used")
    if not _failed_capture_is_comparable(failed_capture):
        return _skipped("failed_candidate_not_comparable")
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
        return _skipped("context_candidate_not_strictly_smaller")
    return retry_msg, retry_cost, ctx.active_context_mode
