"""LLM tool loop: call model, execute tools, repeat until final response."""

from __future__ import annotations

import functools  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
import json  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
import hashlib  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
import os
import queue
import pathlib
import time  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from dataclasses import dataclass, field, replace  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ouroboros.llm import LLMClient, normalize_reasoning_effort, add_usage  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros import task_pacing  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.config import adaptive_quorum, get_context_mode, get_light_model, get_review_enforcement, get_task_review_mode, resolve_effort  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.outcomes import ACCEPTANCE_ACCEPTED, ACCEPTANCE_BYPASS_REASON_BY_RAIL, ACCEPTANCE_BYPASS_REASONS, ACCEPTANCE_DECISION_STATUSES, ACCEPTANCE_FINALIZED_UNACCEPTED, ACCEPTANCE_REVISION_REQUESTED, REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE, REASON_DELIVERY_CONTROL_DEGRADED, REASON_OWNER_REQUESTED_FINALIZATION, RESULT_INFRA_FAILED, extract_final_answer, latest_agent_defined_verification, latest_unreconciled_failed_verification, latest_unreconciled_masked_verification, reviewable_effect_projection, should_nudge_verification, turn_has_reviewable_effects  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.observability import new_execution_id  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.tool_policy import CAPABILITY_OMISSION_HEADER, format_capability_omissions, initial_tool_schemas, list_non_core_tools, swarm_router_turn  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context import build_user_content  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.context_budget import ContextReclaimRequest  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.context_compaction import compact_tool_history_llm, context_reclaim_transcript_sha256  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.deadline_utils import parse_deadline_ts, utc_now  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.utils import estimate_tokens, sanitize_tool_result_for_log, truncate_review_artifact  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.usage_accounting import (
    BudgetExceeded,
    PhysicalAttemptContext,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    PhysicalAttemptPreconditionFailed,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    invalidate_task_cache_splits,
    last_physical_attempt_capture,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
)
from ouroboros.task_finalization import (  # noqa: F401 -- historical import surface for the L-B leaves
    TERMINAL_ORIGIN_HOST_NOTICE,
    TERMINAL_ORIGIN_HOST_SALVAGE,
    TERMINAL_ORIGIN_MODEL_FINAL,
)
from supervisor.owner_stop import (
    _mark_owner_stop_control_drained,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    _narrow_round_deadline,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    _owner_stop_control_is_current,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    _owner_stop_window_elapsed,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
)

from ouroboros.loop_tool_execution import (
    StatefulToolExecutor,
    handle_tool_calls,
    prune_reclaim_trace_refs,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    reclaim_negative_memo,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    reclaim_trace_refs,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
)
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY, call_llm_with_retry, emit_llm_usage_event, forced_response_is_incomplete, forced_response_parts, provider_no_call_source  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.delegate_hold import (
    close_hold as _delegate_hold_close,
    hold_step as _delegate_hold_step,
    latch_after_unknown as _delegate_hold_latch,
)
from ouroboros.loop_transport import (
    TransportWaitEpisode,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    end_episode_budget as _end_episode_budget,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    fallback_chain_allowed as _fallback_chain_allowed,
    finalize_now_transport_terminal as _finalize_now_transport_terminal,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    last_assistant_text as _last_assistant_text,
    provider_terminal_fallback_text as _provider_terminal_fallback_text,
    reconcile_transport_wait as _reconcile_transport_wait,
    task_deadline_epoch as _task_deadline_epoch,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    transport_wait_step as _transport_wait_step,
)
from ouroboros.pricing import estimate_cost_optional  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves

log = logging.getLogger(__name__)


def _handle_text_response(
    content: Optional[str],
    llm_trace: Dict[str, Any],
    accumulated_usage: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Handle LLM response without tool calls (final response)."""
    safe_content = sanitize_tool_result_for_log(content or "")
    if safe_content.strip():
        llm_trace["reasoning_notes"].append(safe_content.strip())
        accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_MODEL_FINAL
    return safe_content, accumulated_usage, llm_trace


# Bounded staleness for the two DECIDING cost surfaces (ceiling check,
# milestone note): a round can block 900s in wait_tasks while children spend,
# and the pacing refresh covers only deadline-less tasks — such a round pays
# ONE real projection read, never per-round (e4a87344).


# Closed set of typed acceptance reasons (v6.78.0): every value is a fact
# the host already computed or the exit branch's name; none derives from model
# prose. `unspecified` is only the fail-closed fallback for a forgotten reason.


# The host-forced acceptance-review checklist (module constant for the size
# gate). v6.60.0 adds the explicit SCOPE-CUT question — a silent/unjustified
# narrowing is a high-severity finding, which under blocking enforcement
# becomes a typed obligation.


# D18a: `context_fit` owns the transcript seal; re-exported here because the
# historical `from ouroboros.loop import seal_task_transcript` import surface
# and `_loop().seal_task_transcript(...)` in loop_model_call.py address it here.
from ouroboros.context_fit import messages_carry_native_images, seal_task_transcript  # noqa: F401
from ouroboros.nanny_pacing import (
    _nanny_burn_phrase,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    _nanny_metered_since_delegate_activity,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    _nanny_reminder_due,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    _note_nanny_delegate_activity,
)


def _setup_dynamic_tools(tools_registry, tool_schemas, messages):
    """Attach list/enable tool handlers and mutate the active schema list."""
    enabled_extra: set = set()
    active_tool_names = {
        name for schema in tool_schemas
        if (name := str(schema.get("function", {}).get("name") or "").strip())
    }

    def _handle_list_tools(ctx=None, **kwargs):
        omissions = (
            tools_registry.capability_omissions()
            if hasattr(tools_registry, "capability_omissions")
            else []
        )
        non_core = [
            t for t in list_non_core_tools(tools_registry)
            if t["name"] not in active_tool_names
        ]
        if not non_core:
            if not omissions:
                return "All tools are already in your active set."
            lines = ["All currently discovered tools are already in your active set.", ""]
            lines.extend(format_capability_omissions(omissions))
            return "\n".join(lines)
        lines = [f"**{len(non_core)} additional tools available** (use `enable_tools` to activate):\n"]
        for t in non_core:
            lines.append(f"- **{t['name']}**: {t['description'][:120]}")
        if omissions:
            lines.extend(format_capability_omissions(
                omissions, header="\n" + CAPABILITY_OMISSION_HEADER,
            ))
        return "\n".join(lines)

    def _handle_enable_tools(ctx=None, tools: str = "", **kwargs):
        names = [n.strip() for n in tools.split(",") if n.strip()]
        enabled, hidden, not_found = [], [], []
        for name in names:
            schema = tools_registry.get_schema_by_name(name)
            if schema and name not in active_tool_names:
                tool_schemas.append(schema)
                invalidate_task_cache_splits(getattr(ctx, "task_id", ""))
                enabled_extra.add(name)
                active_tool_names.add(name)
                enabled.append(f"{name} (registered late)")
            elif name in active_tool_names:
                enabled.append(f"{name} (already active)")
            else:
                # A policy-filtered tool is distinct from an unknown name.
                reason = (
                    tools_registry.policy_hidden_reason(name)
                    if hasattr(tools_registry, "policy_hidden_reason") else None
                )
                if reason:
                    hidden.append(f"{name} — {reason}")
                else:
                    not_found.append(name)
        parts = []
        if enabled:
            parts.append(
                "✅ Tools are registered in the active capability envelope: "
                + ", ".join(enabled)
            )
        if hidden:
            parts.append(
                "🚫 Hidden by policy (the tool exists but this task cannot use it): "
                + "; ".join(hidden)
            )
        if not_found:
            parts.append(f"❌ Not found: {', '.join(not_found)}")
        return "\n".join(parts) if parts else "No tools specified."

    tools_registry.override_handler("list_available_tools", _handle_list_tools)
    tools_registry.override_handler("enable_tools", _handle_enable_tools)

    non_core_count = len(list_non_core_tools(tools_registry))
    if non_core_count > 0:
        _append_or_merge_user_message(
            messages,
            (
                "[SYSTEM NOTICE]\n"
                f"You have {len(tool_schemas)} core tools loaded. "
                f"There are {non_core_count} additional tools available "
                f"(use `list_available_tools` to see them, `enable_tools` to activate). "
                f"Core tools cover most tasks. Enable extras only when needed."
            ),
        )
    omissions = (
        tools_registry.capability_omissions()
        if hasattr(tools_registry, "capability_omissions")
        else []
    )
    if omissions:
        _append_or_merge_user_message(
            messages,
            "[SYSTEM NOTICE]\n" + "\n".join(format_capability_omissions(omissions)),
        )

    return tool_schemas, enabled_extra


def _provider_unavailable_result(
    ctx: _RoundLimitContext, *, error_kind: str = "provider_unavailable",
    wait_cause: str = "", waited_sec: float = 0.0, interactive: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Salvage provider failure without an unsafe retry. ``wait_cause`` is the
    transport-wait episode's latched cause (survives later overwrites of the
    mutable ``_last_llm_error_kind``); ``waited_sec``/``interactive`` keep the
    terminal text honest for zero-wait and such turns."""
    kind = str(error_kind or "")
    # Terminal precedence is decided here alone (the text helper takes both flags as given): a
    # round record (a granted transport-death repeat, no usable response since) leaves an attempt
    # unresolved and outranks the wait terminal, which in turn outranks the overflow salvage.
    record = isinstance(ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)
    is_transport_wait = wait_cause == "transport_unavailable"
    is_context_overflow = kind == "context_overflow" and not (record or is_transport_wait)
    is_deadline_exhausted = kind == "deadline_exhausted" or str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") == "deadline_exhausted"
    llm_trace = getattr(ctx, "llm_trace", None)
    llm_trace = llm_trace if isinstance(llm_trace, dict) else {}
    candidate = _live_delivery_candidate(ctx)
    salvaged = candidate.full_text if candidate is not None else _last_assistant_text(ctx.messages)
    if candidate is None and not salvaged and ctx.drive_root is not None:
        try:
            from ouroboros.observability import latest_llm_response_text
            salvaged = latest_llm_response_text(pathlib.Path(ctx.drive_root), ctx.task_id) or ""
        except Exception:
            log.debug("latest_llm_response_text salvage failed", exc_info=True)
    if salvaged:
        fallback = salvaged
    else:
        fallback = _provider_terminal_fallback_text(
            ctx.accumulated_usage, is_context_overflow=is_context_overflow,
            is_transport_wait=is_transport_wait, waited_sec=waited_sec,
            interactive=interactive,
            is_deadline_exhausted=is_deadline_exhausted,
        )
    if is_context_overflow:
        text, usage, llm_trace = _forced_fallback_result(
            ctx, llm_trace, fallback, "context_overflow",
            source="context_overflow_local_salvage",
        )
        usage.update(
            execution_status=RESULT_INFRA_FAILED,
            reason_code="llm_api_error",
            _last_llm_error_kind="context_overflow",
        )
        return text, usage, llm_trace
    if is_transport_wait:
        # No-resend terminal over dead egress.
        ctx.accumulated_usage.update(execution_status=RESULT_INFRA_FAILED, reason_code="provider_unavailable")
        text, usage, llm_trace = _forced_fallback_result(
            ctx, llm_trace, fallback, reason_code="provider_unavailable",
            source="provider_outcome_unknown_no_resend" if record else "transport_unavailable_no_resend",
        )
        if usage.get("reason_code") == "provider_unavailable":
            usage["execution_status"] = RESULT_INFRA_FAILED
        return text, usage, llm_trace
    no_call, wall = provider_no_call_source(ctx.accumulated_usage, is_deadline_exhausted)
    if no_call:
        if wall:
            _finalize_forced_services(ctx, llm_trace)
            _drain_forced_owner_directives(ctx, llm_trace)
        text, usage, llm_trace = _forced_fallback_result(
            ctx, llm_trace, fallback, reason_code="provider_unavailable",
            source=no_call, provider_terminal=wall,
        )
        if usage.get("execution_status") is not None:
            usage.update(execution_status=RESULT_INFRA_FAILED, reason_code="provider_unavailable")
        return text, usage, llm_trace
    prompt = (
        "[DEADLINE] Primary model work reached the owner deadline. Produce the best final answer now from verified work and state what remains undone."
        if is_deadline_exhausted else
        "[PROVIDER_UNAVAILABLE] The model provider failed to return a usable response. "
        "The task is being INTERRUPTED by this outage, not completed. Summarize the "
        "verified work so far and state plainly what remains undone."
    )
    text, usage, llm_trace = _forced_final_answer(
        ctx, prompt=prompt, fallback_text=fallback,
        reason_code="deadline_local" if is_deadline_exhausted else "provider_unavailable",
        provider_terminal=not is_deadline_exhausted,
    )
    if not is_deadline_exhausted and usage.get("reason_code") == "provider_unavailable":
        usage["execution_status"] = RESULT_INFRA_FAILED
    return text, usage, llm_trace


def _apply_runtime_overrides(
    ctx: Any,
    active_model: str,
    active_use_local: bool,
    active_effort: str,
) -> Tuple[str, bool, str]:
    """Apply one-shot per-round model/locality/effort overrides from tool ctx."""
    if ctx.active_model_override:
        active_model = ctx.active_model_override
        ctx.active_model_override = None
    if getattr(ctx, "active_use_local_override", None) is not None:
        active_use_local = ctx.active_use_local_override
        ctx.active_use_local_override = None
    if ctx.active_effort_override:
        active_effort = normalize_reasoning_effort(ctx.active_effort_override, default=active_effort)
        ctx.active_effort_override = None
    return active_model, active_use_local, active_effort


def _apply_overrides_and_regate_mode(ctx, active_model, active_use_local, active_effort, active_context_mode):
    """Apply per-round overrides; route rebind never predicts a mode change."""
    active_model, active_use_local, active_effort = _apply_runtime_overrides(
        ctx, active_model, active_use_local, active_effort,
    )
    return active_model, active_use_local, active_effort, active_context_mode


def _resolve_loop_max_rounds() -> int:
    from ouroboros.config import SETTINGS_DEFAULTS

    default = int(SETTINGS_DEFAULTS["OUROBOROS_MAX_ROUNDS"])
    try:
        return max(1, int(os.environ.get("OUROBOROS_MAX_ROUNDS", str(default))))
    except (ValueError, TypeError):
        log.warning("Invalid OUROBOROS_MAX_ROUNDS, defaulting to %s", default)
        return default


def run_llm_loop(
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    llm: LLMClient,
    drive_logs: pathlib.Path,
    emit_progress: Callable[..., None],
    incoming_messages: queue.Queue,
    task_type: str = "",
    task_id: str = "",
    budget_remaining_usd: Optional[float] = None,
    event_queue: Optional[queue.Queue] = None,
    initial_effort: str = "medium",
    drive_root: Optional[pathlib.Path] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Run the tool loop."""
    ctx = tools._ctx
    ctx._delivery_candidate, ctx._delivery_candidate_revision = None, 0
    ctx._delivery_control_required = False
    ctx._delivery_evidence_revision, ctx._delivery_evidence_fingerprint = 0, ""
    _initialize_owner_directives(ctx, messages)
    task_model_override = str(getattr(ctx, "task_model_override", "") or "").strip()
    active_model = task_model_override or llm.default_model()
    active_effort = initial_effort
    if getattr(ctx, "task_use_local_override", None) is not None:
        active_use_local = bool(ctx.task_use_local_override)
    else:
        active_use_local = os.environ.get("USE_LOCAL_MAIN", "").lower() in ("true", "1")
    # Unknown routes get one honest call; no synthetic short-window capacity.
    _preferred_context_mode = get_context_mode()
    context_fit_plan = getattr(ctx, "context_fit_plan", None)
    if (context_fit_plan is not None
            and str(getattr(context_fit_plan, "preferred_mode", "")) == _preferred_context_mode):
        active_context_mode = str(getattr(context_fit_plan, "initial_mode", "") or _preferred_context_mode)
    else:
        active_context_mode = _preferred_context_mode
    llm_trace: Dict[str, Any] = {"reasoning_notes": [], "tool_calls": []}
    accumulated_usage: Dict[str, Any] = {"_task_attempt": getattr(ctx, "task_attempt", None)}
    tools._ctx._accumulated_usage = ctx._accumulated_usage = accumulated_usage
    invalidate_task_cache_splits(task_id or getattr(ctx, "task_id", ""))  # rebuilt attempt = new prefix
    max_retries = 3
    cost_ceiling = _resolve_task_cost_ceiling(ctx, budget_remaining_usd)
    if cost_ceiling.root_cap_usd is not None:
        # A resumed/late-started tree member must see tree spend before its
        # first pacing surface, not a process-local empty stash.
        _loop_tree_accounting(refresh=True, max_age_sec=0.0)
    from ouroboros.tools import tool_discovery as _td
    _td.set_registry(tools)

    tool_schemas = initial_tool_schemas(tools)
    tool_schemas, _enabled_extra_tools = _setup_dynamic_tools(tools, tool_schemas, messages)
    tools._ctx.event_queue = event_queue
    tools._ctx.task_id = task_id
    tools._ctx.messages = messages
    stateful_executor = StatefulToolExecutor()
    exit_ctx = _LoopExitContext(
        tools, drive_root, task_id, event_queue, drive_logs, accumulated_usage, llm_trace,
    )
    _owner_msg_seen: set = set()
    MAX_ROUNDS = _resolve_loop_max_rounds()
    MAX_ROUNDS = min(MAX_ROUNDS, int(getattr(ctx, "inline_max_rounds", MAX_ROUNDS)))
    round_idx = 0
    free_redial = False
    transport_wait = None
    limit_ctx: Optional[_RoundLimitContext] = None
    try:
        while True:
            if free_redial:
                # A transport-wait redial re-enters the SAME logical round.
                free_redial = False
            else:
                round_idx += 1

            ctx = tools._ctx
            _prev_active_route = (active_model, active_use_local)
            _prev_active_model = active_model
            active_model, active_use_local, active_effort, active_context_mode = _apply_overrides_and_regate_mode(
                ctx, active_model, active_use_local, active_effort, active_context_mode,
            )
            if (active_model, active_use_local) != _prev_active_route:
                context_fit_plan, active_context_mode = _rebind_context_fit_plan(
                    context_fit_plan, tools, messages, model=active_model,
                    use_local=active_use_local, preferred_mode=_preferred_context_mode,
                    tool_schemas=tool_schemas,
                )
            if active_model != _prev_active_model:
                # Cross-FAMILY switch_model / per-task override: strip the prior
                # family's provider-private reasoning blocks so the new family
                # does not 400 on a foreign signature (same family = no-op).
                _sanitized = LLMClient.sanitize_reasoning_on_model_switch(messages, _prev_active_model, active_model)
                if _sanitized is not messages:
                    messages[:] = _sanitized
            ctx.active_context_mode = active_context_mode
            ctx.active_model = active_model
            ctx.active_effort = active_effort
            ctx.active_use_local = active_use_local

            # One forced-wrap-up context per round: consumed by the round-limit
            # path and supervisor finalize_now control path below.
            limit_ctx = _RoundLimitContext(
                messages, llm, active_model, active_effort, max_retries, drive_logs,
                task_id, round_idx, event_queue, accumulated_usage, task_type,
                active_use_local, MAX_ROUNDS, drive_root=drive_root, llm_trace=llm_trace,
                incoming_messages=incoming_messages, owner_msg_seen=_owner_msg_seen, tool_schemas=tool_schemas)
            _finalize_limit_ctx(limit_ctx, tools, llm_trace)
            if round_idx > MAX_ROUNDS:
                # Live hold: a paid [ROUND_LIMIT] dial would be a resend (no wake receipt) — no-call unknown terminal.
                if _delegate_hold_close(tools, drive_logs=drive_logs, task_id=task_id,
                    detail="round_limit") == "active":
                    accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
                    text, accumulated_usage, forced_trace = _handle_provider_unavailable(limit_ctx, error_kind="provider_outcome_unknown")
                else:
                    text, accumulated_usage, forced_trace = _handle_round_limit(limit_ctx)
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            # Tuple, not a sum: an APPENDED short owner message must also read as new input (final-pair fable F1).
            _pre_drain_sig = (len(messages), len(str(messages[-1].get("content") or "")))
            _controls = _drain_incoming_messages(
                messages, incoming_messages, drive_root, task_id, event_queue,
                _owner_msg_seen, owner_ctx=ctx)
            if _delegate_hold_step(
                    tools, controls=_controls, messages=messages, drive_logs=drive_logs,
                    task_id=task_id, emit_progress=emit_progress,
                    new_input=(len(messages), len(str(messages[-1].get("content") or ""))) != _pre_drain_sig) == "terminal":
                # The no-call decision reads the usage record, not the error_kind argument — stamp first.
                accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
                text, accumulated_usage, forced_trace = _handle_provider_unavailable(
                    limit_ctx, error_kind="provider_outcome_unknown")
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace
            # Per-round early exit: supervisor finalize_now, else loop-local real-deadline finalize.
            _early_final = _maybe_early_finalize(
                limit_ctx, tools, _controls, transport_episode=transport_wait)
            if _early_final is not None:
                text, accumulated_usage, forced_trace = _early_final
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            # Typed soft landing (v6.91): the ledger fence stays the untouched
            # backstop; an exhausted ceiling wraps up BEFORE spending a round.
            _soft_land = _soft_land_exhausted_ceiling(limit_ctx, cost_ceiling)
            if _soft_land is not None:
                text, accumulated_usage, forced_trace = _soft_land
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            _checkpoint_injected = _inject_round_checkpoints(
                round_idx=round_idx, max_rounds=MAX_ROUNDS, messages=messages, accumulated_usage=accumulated_usage,
                emit_progress=emit_progress, tools=tools, event_queue=event_queue, task_id=task_id,
                drive_logs=drive_logs, budget_remaining_usd=budget_remaining_usd, cost_ceiling=cost_ceiling)

            messages, _compaction_usage = _run_round_compaction(
                messages,
                _CompactionRoundContext(
                    tools=tools, drive_root=drive_root, drive_logs=drive_logs,
                    task_id=task_id, round_idx=round_idx,
                    event_queue=event_queue, emit_progress=emit_progress))
            if tools._ctx.messages is not messages:
                tools._ctx.messages = messages
            limit_ctx.messages = messages  # WA2: provider-death finalize must salvage the COMPACTED transcript
            if _compaction_usage:
                _account_compaction_usage(accumulated_usage, _compaction_usage, event_queue, task_id)

            seal_task_transcript(messages)

            msg, cost, active_context_mode = _call_round_model(
                _RoundModelCallContext(
                    llm=llm,
                    messages=messages,
                    tools=tools,
                    context_fit_plan=context_fit_plan,
                    active_model=active_model,
                    tool_schemas=tool_schemas,
                    active_effort=active_effort,
                    max_retries=max_retries,
                    drive_logs=drive_logs,
                    task_id=task_id,
                    round_idx=round_idx,
                    event_queue=event_queue,
                    accumulated_usage=accumulated_usage,
                    task_type=task_type,
                    active_use_local=active_use_local,
                    active_context_mode=active_context_mode,
                    drive_root=drive_root,
                )
            )
            tools._ctx._current_llm_call_meta = dict(accumulated_usage.get("_last_llm_call_meta") or {})

            last_error_kind = str(accumulated_usage.get("_last_llm_error_kind") or "")
            transport_wait = _reconcile_transport_wait(
                transport_wait, ctx, msg_present=msg is not None, error_kind=last_error_kind,
                drive_logs=drive_logs, task_id=task_id, model=active_model, emit_progress=emit_progress)
            if msg is None and _fallback_chain_allowed(ctx, last_error_kind, transport_wait, accumulated_usage):
                _episode_before_chain = transport_wait is not None
                (
                    msg,
                    active_model,
                    active_use_local,
                    context_fit_plan,
                    active_context_mode,
                ) = _run_cross_model_fallback_chain(
                    llm=llm, ctx=ctx, tools=tools, messages=messages, active_model=active_model,
                    active_use_local=active_use_local, tool_schemas=tool_schemas, active_effort=active_effort,
                    max_retries=max_retries, drive_logs=drive_logs, task_id=task_id, round_idx=round_idx,
                    event_queue=event_queue, accumulated_usage=accumulated_usage, task_type=task_type,
                    emit_progress=emit_progress, context_fit_plan=context_fit_plan,
                    active_context_mode=active_context_mode)
                # Post-chain reconcile with the FRESH kind: a MID-chain outage
                # latches too (see reconcile_transport_wait's docstring).
                transport_wait = _reconcile_transport_wait(
                    transport_wait, ctx, msg_present=msg is not None,
                    error_kind=str(accumulated_usage.get("_last_llm_error_kind") or ""),
                    drive_logs=drive_logs, task_id=task_id, model=active_model,
                    emit_progress=emit_progress, after_local_pass=_episode_before_chain)
            if msg is None and transport_wait is not None and _transport_wait_step(
                transport_wait, tools=tools,
                error_kind=str(accumulated_usage.get("_last_llm_error_kind") or ""),
                drive_root=drive_root, drive_logs=drive_logs, task_id=task_id, model=active_model,
                emit_progress=emit_progress, incoming_messages=incoming_messages, owner_msg_seen=_owner_msg_seen):
                free_redial = True
                continue
            if msg is None and _delegate_hold_latch(
                    tools, error_kind=last_error_kind, drive_logs=drive_logs,
                    task_id=task_id, emit_progress=emit_progress):  # hold latched -> next round top parks
                continue
            if msg is None:
                # Exact actor routes skip generic substitution and fail as infrastructure.
                text, accumulated_usage, forced_trace = _handle_provider_unavailable(
                    limit_ctx,
                    error_kind=str(accumulated_usage.get("_last_llm_error_kind") or "provider_unavailable"),
                    wait_cause=transport_wait.wait_cause if transport_wait is not None else "",
                    waited_sec=transport_wait.waited_sec if transport_wait is not None else 0.0,
                    interactive=transport_wait.interactive if transport_wait is not None else False)
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            from ouroboros.openai_chat_dispatch import CUSTOM_RECEIPTS_USAGE_KEY

            tool_calls = msg.get("tool_calls") or []
            tools._ctx._request_wire_custom_receipts = accumulated_usage.pop(
                CUSTOM_RECEIPTS_USAGE_KEY,
                (),
            )
            content = msg.get("content")
            _latch_final_answer_marker(llm_trace, content, current_tool_calls=tool_calls)
            # Every metered response counts as nanny progress.
            _note_nanny_delegate_activity(tools._ctx, round_idx, accumulated_usage, [])
            if not tool_calls:
                final_result = _no_tool_final_answer(
                    content, limit_ctx, llm_trace, tools, incoming_messages,
                    _owner_msg_seen, emit_progress,
                )
                if final_result is None:
                    continue
                return final_result

            if getattr(tools._ctx, "_skill_finalization_injected", False):
                tools._ctx._skill_finalization_injected = False
            assistant_msg = dict(msg)
            assistant_msg.setdefault("role", "assistant")
            messages.append(assistant_msg)

            _emit_round_progress(content, msg, emit_progress, llm_trace)

            handle_tool_calls(
                tool_calls, tools, drive_logs, task_id, stateful_executor,
                messages, llm_trace, emit_progress
            )

            # Nanny-economics baseline (poltergeist phase B): mark the
            # round's metered progress; re-baseline when it touched a
            # delegated run. Exact tool-call transitions — no log scans.
            _note_nanny_delegate_activity(
                tools._ctx, round_idx, accumulated_usage, tool_calls,
            )

            _prepare_post_tool_budget_context(
                tools, limit_ctx, llm_trace, active_model, active_use_local, active_effort,
            )
            budget_result = _check_budget_limits(
                limit_ctx,
                budget_remaining_usd,
                cost_ceiling=cost_ceiling,
            )
            if budget_result is not None:
                text, accumulated_usage, budget_trace = budget_result
                _merge_finalization_trace(llm_trace, budget_trace)
                return text, accumulated_usage, llm_trace

    except BudgetExceeded as exc:
        _delegate_hold_close(tools, drive_logs=drive_logs, task_id=task_id, detail="budget")
        return _handle_budget_exceeded(
            exc, exit_ctx, limit_ctx=limit_ctx, episode=transport_wait)
    finally:
        # No stale active latch behind an in-process exit (a crash skips this frame, keeping the latch for recovery).
        _delegate_hold_close(tools, drive_logs=drive_logs, task_id=task_id, detail="loop_exit")
        _cleanup_loop_resources(stateful_executor, exit_ctx)

# The v7 L-B split: the members below moved into cohesive leaves (module-size
# boundary). This tree keeps the FULL re-export surface: the tip consumer set
# (production callers and tests) still addresses every moved name at its
# historical ouroboros.loop binding, and the D33 call-time handle reads of the
# sibling leaves resolve through this module as the family rendezvous. The
# oracle's later L3-trimmed surface (RETIRED_FROM_LOOP) is a consumer-rebind
# wave, not part of the byte-preserving relocation (see LEDGER_CORRECTIONS,
# D01 lane).
from ouroboros.loop_messages import (  # noqa: E402, F401 -- intentional public re-exports
    _emit_checkpoint_event,
    _extract_plain_text_from_content,
    _append_or_merge_user_message,
    _evict_stale_image_blocks,
    _append_or_merge_user_content,
    _owner_marked_content,
    _record_owner_directive,
    _initialize_owner_directives,
    _visible_round_text,
    _emit_round_progress,
)
from ouroboros.loop_acceptance import (  # noqa: E402, F401 -- intentional public re-exports
    _task_acceptance_eligible,
    _begin_task_acceptance_fence,
    _end_task_acceptance_fence,
    _supersede_delivery_acceptance_binding,
    _supersede_task_acceptance_for_owner_followup,
    _task_acceptance_owner_generation_changed,
    _supersede_task_acceptance_for_evidence_change,
    _task_acceptance_subtree_snapshot,
    _mark_root_acceptance_checkpoint,
    _latch_final_answer_marker,
    _server_web_allowed_by_task,
    ACCEPTANCE_REASON_UNSPECIFIED,
    ACCEPTANCE_DECISION_REASONS,
    _set_acceptance_decision,
    _collect_acceptance_obligations,
    _reopen_obligation_row,
    _open_acceptance_obligations,
    _dispose_obligations_on_clean_pass,
    _format_obligations_clause,
    _record_forced_acceptance_bypass,
    terminalize_dangling_revision,
)
from ouroboros.loop_acceptance_review import (  # noqa: E402, F401 -- intentional public re-exports
    _ACCEPTANCE_REVIEW_CHECKLIST,
    _TaskAcceptanceContext,
    _acceptance_dialogue_quorum,
    _attach_dialogue_to_host_run,
    _mark_agent_acceptance_runs_advisory,
    _latest_agent_acceptance_evidence,
    _build_host_acceptance_evidence,
    _execute_task_acceptance_panel,
    _record_host_acceptance_run,
    _set_applied_host_acceptance_impact,
    _apply_task_acceptance_result,
    _record_acceptance_infra_failure,
    _prior_acceptance_run,
    _direct_context_fence_state,
    _disposition_reason_sha256,
    _refuse_identical_acceptance,
    _run_task_acceptance_review_once,
    _total_paid_acceptance_cycles,
    _RETRIEVING_ACCESS_DISCLOSURE,
    _acceptance_delivery_slots,
    _retrieving_packet_projection,
    _skip_task_acceptance_for_launch_reason,
    acceptance_retrieving_work_order,
    acceptance_dialogue_history,
    acceptance_paid_identity,
    bind_acceptance_paid_identity,
)
from ouroboros.loop_round_limits import (  # noqa: E402, F401 -- intentional public re-exports
    _CompactionRoundContext,
    _drain_incoming_messages,
    _context_reclaim_passes,
    _context_reclaim_materializations,
    _context_overflow_retries,
    _run_round_compaction,
    _RoundLimitContext,
    _account_compaction_usage,
    _handle_round_limit,
    _handle_forced_finalization,
    _handle_owner_stop_finalization,
    _handle_provider_unavailable,
    _maybe_deadline_local_finalize,
    _maybe_early_finalize,
    _finalize_limit_ctx,
)
from ouroboros.loop_nudges import (  # noqa: E402, F401 -- intentional public re-exports
    _skill_names_touched_by_trace,
    _skill_finalization_message,
    _force_plan_decision,
    _force_plan_reminder,
    _force_plan_disclosure,
    _build_recent_tool_trace,
    _maybe_inject_self_check,
    _maybe_inject_time_budget_milestone,
    _maybe_inject_cost_budget_milestone,
    _maybe_inject_nanny_economics_reminder,
    _inject_round_checkpoints,
    _forced_delegation_note,
    _nanny_finalization_message,
    _maybe_inject_finalization_nudges,
    _answer_protocol_active,
    _contract_expected_output,
)
from ouroboros.loop_model_call import (  # noqa: E402, F401 -- intentional public re-exports
    _adopt_fallback_route,
    _snapshot_context_fit_usage,
    _restore_context_fit_usage,
    _run_cross_model_fallback_chain,
    _rebind_context_fit_plan,
    _RoundModelCallContext,
    _context_fit_round_id,
    _main_context_profile,
    _remember_main_fit,
    _measure_round_main_fit,
    _physical_context_for_fit,
    _dispatch_round_model,
    _run_main_reclaim,
    _measure_after_reclaim,
    _reproject_actual_overflow_low,
    _failed_capture_is_comparable,
    _strict_context_shrink_predicate,
    _emit_overflow_retry_skipped,
    _call_round_model,
)
from ouroboros.loop_budget import (  # noqa: E402, F401 -- intentional public re-exports
    _check_budget_limits,
    _resolve_task_cost_ceiling,
    _TREE_ACCOUNTING_MAX_STALE_SEC,
    _loop_tree_accounting,
    _soft_land_exhausted_ceiling,
    _service_finalization_evidence,
    _LoopExitContext,
    _handle_budget_exceeded,
    _cleanup_loop_resources,
    _service_identity_projection,
    _finalize_task_services,
    _prepare_post_tool_budget_context,
)
from ouroboros.loop_delivery import (  # noqa: E402, F401 -- intentional public re-exports
    DeliveryCandidate,
    _swarm_handoff_attempt,
    _compute_subagent_handoff,
    _delivery_evidence_state,
    _unaccepted_delivery_binding,
    _delivery_acceptance_binding,
    _publish_delivery_candidate,
    _replace_delivery_candidate,
    _ensure_explicit_acceptance_binding,
    _forced_unaccepted_binding,
    _live_delivery_candidate,
    _current_delivery_candidate,
    _degrade_retained_delivery_candidate,
    _merge_finalization_trace,
    _delivery_control_prompt,
    _delivery_replace_required,
    _delivery_keep_allowed,
    _arm_delivery_control,
    _hold_delivery_for_skill_action,
    _parse_delivery_control_object,
    _parse_delivery_control_body,
    _classify_parsed_delivery_control,
    _resolve_forced_delivery_control_body,
    _resolve_delivery_control,
    _CHILD_ABSORPTION_HOLD_CONTROL,
    _DELIVERY_HOLD_CONTROLS,
    _SKILL_ACTION_HOLD_CONTROL,
    _compose_delivery_suffix,
    _no_tool_final_answer,
)
from ouroboros.loop_forced_finalization import (  # noqa: E402, F401 -- intentional public re-exports
    _load_direct_child_results,
    _direct_child_results,
    _child_disposition_state,
    _project_child_result_dispositions,
    _record_forced_finalization,
    _forced_orphan_note,
    _claimed_child_dispositions,
    _undispositioned_children,
    _undecided_children_listing,
    _maybe_enforce_child_absorption_gate,
    _run_forced_children_acceptance,
    _enforce_swarm_actions,
    _finalize_forced_services,
    _drain_forced_owner_directives,
    _call_forced_model_once,
    _publish_model_forced_candidate,
    _publish_stale_forced_candidate,
    _forced_fallback_result,
    _forced_swarm_router_result,
    _resolve_forced_delivery_control,
    _forced_final_answer,
    _FORCED_BEST_EFFORT_TAIL,
    _prepare_forced_prompt,
)
