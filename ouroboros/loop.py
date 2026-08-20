"""LLM tool loop: call model, execute tools, repeat until final response."""

from __future__ import annotations

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
from ouroboros.outcomes import ACCEPTANCE_ACCEPTED, ACCEPTANCE_BYPASS_REASON_BY_RAIL, ACCEPTANCE_BYPASS_REASONS, ACCEPTANCE_DECISION_STATUSES, ACCEPTANCE_FINALIZED_UNACCEPTED, ACCEPTANCE_REVISION_REQUESTED, REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE, REASON_DELIVERY_CONTROL_DEGRADED, REASON_OWNER_REQUESTED_FINALIZATION, RESULT_INFRA_FAILED, extract_final_answer, latest_agent_defined_verification, latest_unreconciled_failed_verification, latest_unreconciled_masked_verification, reviewable_effect_projection, should_nudge_verification, turn_has_reviewable_effects  # noqa: F401 -- moved readers import via the L-B leaves; the loop surface keeps these bindings
from ouroboros.observability import new_execution_id  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.tool_policy import CAPABILITY_OMISSION_HEADER, format_capability_omissions, initial_tool_schemas, list_non_core_tools, swarm_router_turn  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context import build_user_content  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.context_budget import ContextReclaimRequest  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.context_compaction import compact_tool_history_llm, context_reclaim_transcript_sha256  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.deadline_utils import parse_deadline_ts, utc_now  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.utils import estimate_tokens, truncate_review_artifact  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.usage_accounting import (
    BudgetExceeded,
    PhysicalAttemptContext,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    PhysicalAttemptPreconditionFailed,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    last_physical_attempt_capture,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
)

from ouroboros.loop_tool_execution import (
    StatefulToolExecutor,
    handle_tool_calls,
    prune_reclaim_trace_refs,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    reclaim_negative_memo,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
    reclaim_trace_refs,  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
)
from ouroboros.loop_llm_call import call_llm_with_retry, emit_llm_usage_event  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves
from ouroboros.pricing import estimate_cost_optional  # noqa: F401 -- the loop module keeps its historical import surface for the L-B leaves

# Backward-compat alias for source-inspecting/monkeypatched tests.
_call_llm_with_retry = call_llm_with_retry

log = logging.getLogger(__name__)


def _handle_text_response(
    content: Optional[str],
    llm_trace: Dict[str, Any],
    accumulated_usage: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Handle LLM response without tool calls (final response)."""
    if content and content.strip():
        llm_trace["reasoning_notes"].append(content.strip())
    return (content or ""), accumulated_usage, llm_trace


def seal_task_transcript(
    messages: List[Dict[str, Any]],
    keep_active: int = 5,
    min_prefix_tokens: int = 2048,
) -> None:
    """Mark one stable old tool-result boundary for provider prompt caching."""
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # Flatten the old sealed boundary before choosing a new one.
            msg["content"] = _extract_plain_text_from_content(content)

    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool"
    ]
    if len(tool_indices) <= keep_active:
        return

    seal_candidate_idx = tool_indices[-(keep_active + 1)]

    prefix_text_len = sum(
        len(_extract_plain_text_from_content(m.get("content", "")))
        for m in messages[: seal_candidate_idx + 1]
    )
    prefix_tokens = prefix_text_len // 4  # rough 4-chars-per-token estimate

    if prefix_tokens < min_prefix_tokens:
        return

    candidate = messages[seal_candidate_idx]
    plain_text = str(candidate.get("content", ""))
    if not plain_text.strip():
        # Anthropic 400s on cache_control attached to an empty text block; never seal
        # an empty tool output as the cache anchor (turns the whole task unanswerable).
        plain_text = "(no tool output)"
    candidate["content"] = [
        {
            "type": "text",
            "text": plain_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _setup_dynamic_tools(tools_registry, tool_schemas, messages):
    """Attach list/enable tool handlers and mutate the active schema list."""
    enabled_extra: set = set()
    active_tool_names = {
        str(schema.get("function", {}).get("name") or "").strip()
        for schema in tool_schemas
        if str(schema.get("function", {}).get("name") or "").strip()
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
                enabled_extra.add(name)
                active_tool_names.add(name)
                enabled.append(f"{name} (registered late)")
            elif name in active_tool_names:
                enabled.append(f"{name} (already active)")
            else:
                # F3 (2026-08-10 saga): a policy-filtered tool is not "Not found" —
                # answer with the typed reason so the agent stops guessing names.
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
    emit_progress: Callable[[str], None],
    incoming_messages: queue.Queue,
    task_type: str = "",
    task_id: str = "",
    budget_remaining_usd: Optional[float] = None,
    event_queue: Optional[queue.Queue] = None,
    initial_effort: str = "medium",
    drive_root: Optional[pathlib.Path] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Run the LLM-with-tools loop and return final text, usage, and trace."""
    ctx = tools._ctx
    ctx._delivery_candidate = None
    ctx._delivery_candidate_revision = 0
    ctx._delivery_control_required = False
    ctx._delivery_evidence_revision = 0
    ctx._delivery_evidence_fingerprint = ""
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
    if (
        context_fit_plan is not None
        and str(getattr(context_fit_plan, "preferred_mode", "")) == _preferred_context_mode
    ):
        active_context_mode = str(getattr(context_fit_plan, "initial_mode", "") or _preferred_context_mode)
    else:
        active_context_mode = _preferred_context_mode
    llm_trace: Dict[str, Any] = {"reasoning_notes": [], "tool_calls": []}
    accumulated_usage: Dict[str, Any] = {}
    # Published as a live reference so blocking tools (wait_task/wait_tasks/
    # delegate_wait) can read RECORDED per-send facts — e.g. the APPLIED
    # prompt-cache TTL (`_last_prompt_cache_ttl`) behind the cache-horizon
    # disclosure — without a second, route-derived predictor.
    tools._ctx._accumulated_usage = accumulated_usage
    max_retries = 3
    cost_ceiling = _resolve_task_cost_ceiling(ctx, budget_remaining_usd)
    if cost_ceiling.root_cap_usd is not None:
        # Loop-start seed (one rare ledger read): a resumed/late-started member
        # of a spending tree must see the real tree number before its first
        # pacing surface, not a process-local empty stash.
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
    round_idx = 0
    limit_ctx: Optional[_RoundLimitContext] = None
    try:
        while True:
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
                # A cross-FAMILY switch_model / per-task override: strip the
                # prior family's provider-private reasoning blocks from the
                # history so the new family does not 400 on a signature it
                # cannot validate (safe — loses only reasoning continuity).
                # Same family is a no-op.
                _sanitized = LLMClient.sanitize_reasoning_on_model_switch(messages, _prev_active_model, active_model)
                if _sanitized is not messages:
                    messages[:] = _sanitized
            ctx.active_context_mode = active_context_mode  # switch_model re-binds the fit plan from this round's mode
            ctx.active_model = active_model  # publish the round's REAL model (incl. switch_model / per-task override) so tools (native screenshot vision-routing) don't read the stale global OUROBOROS_MODEL env

            # One forced-wrap-up context per round: consumed by the round-limit
            # path and the supervisor finalize_now control path below.
            limit_ctx = _RoundLimitContext(
                messages, llm, active_model, active_effort, max_retries, drive_logs,
                task_id, round_idx, event_queue, accumulated_usage, task_type,
                active_use_local, MAX_ROUNDS, drive_root=drive_root,
                incoming_messages=incoming_messages, owner_msg_seen=_owner_msg_seen,
            )
            _finalize_limit_ctx(limit_ctx, tools, llm_trace)
            if round_idx > MAX_ROUNDS:
                text, accumulated_usage, forced_trace = _handle_round_limit(limit_ctx)
                _merge_finalization_trace(llm_trace, forced_trace)
                return text, accumulated_usage, llm_trace

            _controls = _drain_incoming_messages(
                messages,
                incoming_messages,
                drive_root,
                task_id,
                event_queue,
                _owner_msg_seen,
                owner_ctx=ctx,
            )
            # Early-exit per round: supervisor finalize_now, else loop-local real-
            # deadline finalize (headless runs that get no finalize_now) — finalize
            # best-effort rather than be killed mid-step with nothing.
            _early_final = _maybe_early_finalize(limit_ctx, tools, _controls)
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
                    tools=tools,
                    drive_root=drive_root,
                    drive_logs=drive_logs,
                    task_id=task_id,
                    round_idx=round_idx,
                    event_queue=event_queue,
                    emit_progress=emit_progress,
                ),
            )
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

            if msg is None:
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
                if msg is None:
                    # Provider-death: salvage the useful workspace state like the
                    # forced rails do, but terminalize as an infra failure — an
                    # outage interrupts the task, it never completes it.
                    text, accumulated_usage, forced_trace = _handle_provider_unavailable(limit_ctx)
                    _merge_finalization_trace(llm_trace, forced_trace)
                    return text, accumulated_usage, llm_trace

            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            _latch_final_answer_marker(llm_trace, content, current_tool_calls=tool_calls)
            # F12: EVERY LLM response marks metered nanny progress (expensive
            # no-tool rounds count); the delegate BASELINE moves post-tools only.
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

            # Nanny-economics baseline (poltergeist phase B): mark this round's
            # metered progress, and re-baseline when the round touched a
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
        return _handle_budget_exceeded(exc, exit_ctx, limit_ctx=limit_ctx)
    finally:
        _cleanup_loop_resources(stateful_executor, exit_ctx)


# The v7 L-B split: the members below moved into cohesive leaves (module-size
# boundary), and this block binds the ones the loop family still addresses here.
# The L3 package (spec 4.3-15) spent the TEMPORARY half of that facade: every
# moved name was classified by who reads it, the loop-private test imports were
# re-homed to their leaf owners, and a name whose only reader is its own leaf
# left this surface for good. What is left survives for one of exactly two
# reasons -- `run_llm_loop` below calls it, or a SIBLING leaf reads it through
# the D33 call-time handle (`_loop().X`), for which this module is the family's
# one rendezvous binding. Retiring those would not remove a seam; it would trade
# one shared seam for a mesh of sibling handles, and cross-leaf monkeypatching
# would have to learn which leaf a name landed in. The retired names are pinned
# as absent in tests/test_loop_owner_facades.py::RETIRED_FROM_LOOP: adding a
# re-export back "for convenience" restores a second address for one object.
from ouroboros.loop_messages import (  # noqa: E402, F401 -- intentional public re-exports
    _emit_checkpoint_event,
    _extract_plain_text_from_content,
    _append_or_merge_user_message,
    _owner_marked_content,
    _record_owner_directive,
    _initialize_owner_directives,
    _last_assistant_text,
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
    _set_acceptance_decision,
    _collect_acceptance_obligations,
    _open_acceptance_obligations,
    _dispose_obligations_on_clean_pass,
    _format_obligations_clause,
    _record_forced_acceptance_bypass,
)
from ouroboros.loop_acceptance_review import (  # noqa: E402, F401 -- intentional public re-exports
    _run_task_acceptance_review_once,
)
from ouroboros.loop_round_limits import (  # noqa: E402, F401 -- intentional public re-exports
    _CompactionRoundContext,
    _task_deadline_epoch,
    _drain_incoming_messages,
    _context_reclaim_passes,
    _context_reclaim_materializations,
    _context_overflow_retries,
    _run_round_compaction,
    _RoundLimitContext,
    _account_compaction_usage,
    _handle_round_limit,
    _handle_forced_finalization,
    _handle_provider_unavailable,
    _maybe_early_finalize,
    _finalize_limit_ctx,
)
from ouroboros.loop_nudges import (  # noqa: E402, F401 -- intentional public re-exports
    _force_plan_decision,
    _force_plan_reminder,
    _force_plan_disclosure,
    _note_nanny_delegate_activity,
    _inject_round_checkpoints,
    _forced_delegation_note,
    _maybe_inject_finalization_nudges,
)
from ouroboros.loop_model_call import (  # noqa: E402, F401 -- intentional public re-exports
    _run_cross_model_fallback_chain,
    _rebind_context_fit_plan,
    _RoundModelCallContext,
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
    _finalize_task_services,
    _prepare_post_tool_budget_context,
)
from ouroboros.loop_delivery import (  # noqa: E402, F401 -- intentional public re-exports
    DeliveryCandidate,
    _swarm_handoff_attempt,
    _delivery_evidence_state,
    _publish_delivery_candidate,
    _replace_delivery_candidate,
    _forced_unaccepted_binding,
    _live_delivery_candidate,
    _current_delivery_candidate,
    _degrade_retained_delivery_candidate,
    _merge_finalization_trace,
    _delivery_replace_required,
    _arm_delivery_control,
    _parse_delivery_control_object,
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
    _maybe_enforce_child_absorption_gate,
    _enforce_swarm_actions,
    _finalize_forced_services,
    _forced_fallback_result,
    _forced_swarm_router_result,
    _forced_final_answer,
)
