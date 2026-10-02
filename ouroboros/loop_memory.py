"""Published-memory fitting and source-bound repair for Main calls.

This leaf prepares complete memory views without buying an economic-target
rewrite. After the round recovery owner has tried configured routes, a definite
refusal may use the existing consolidator for a useful source-bound reduction.
Routing, retry authority and final physical-candidate proof remain in
``loop_model_call``; retained observations never replace those contracts.
"""
from __future__ import annotations

import contextlib
import copy
import json
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed, invalidate_task_cache_splits

if TYPE_CHECKING:
    from ouroboros.loop_model_call import _RoundModelCallContext


def _loop():
    from ouroboros import loop
    return loop


def _model_calls():
    from ouroboros import loop_model_call
    return loop_model_call


def _memory_recovery_held(usage: dict) -> bool:
    """A permitted new-route generation does not release older paid custody."""
    return bool(usage.get("_pending_transport_outcome") or TRANSPORT_DEATHS_KEY in usage
                or usage.get("_last_llm_error_kind") in {
                    "provider_outcome_unknown", "deadline_exhausted", "budget_exhausted"})


@dataclass(frozen=True)
class _DeferredMemoryRefusal:
    call: _RoundModelCallContext
    capture: Any
    refused_digest_ids: tuple | None = None
    refused_memory_bytes: int | None = None


def _defer_memory_refusal(ctx: _RoundModelCallContext, capture: Any, *,
                          refused_digest_ids=None, refused_memory_bytes=None) -> None:
    """Keep the acting route's paid need while configured alternatives are tried."""
    if (getattr(ctx.tools._ctx, "_deferred_memory_refusal", None) is not None
            or not (_may_repair_main_memory(ctx) or ctx.active_context_mode == "max")
            or _memory_recovery_held(ctx.accumulated_usage)
            or not _model_calls()._failed_capture_is_comparable(capture)):
        return
    ctx.tools._ctx._deferred_memory_refusal = _DeferredMemoryRefusal(
        replace(ctx, messages=copy.deepcopy(ctx.messages)), capture, refused_digest_ids, refused_memory_bytes)


def _main_frame_bytes(ctx: _RoundModelCallContext) -> int:
    """Compare complete canonical frames; the send gate proves physical shrink."""
    from ouroboros.context_budget import canonical_context_json

    messages = ctx.messages
    if ctx.prepared_main_frame is not None:
        messages = ctx.context_fit_plan.reproject_transcript(ctx.prepared_main_frame, ctx.active_context_mode)
    return len(canonical_context_json({"messages": messages, "tools": ctx.tool_schemas}).encode("utf-8"))


def _may_repair_main_memory(ctx: _RoundModelCallContext) -> bool:
    plan = ctx.context_fit_plan
    if not getattr(plan, "chronicle_state_json", ""):
        return False
    metadata = getattr(ctx.tools._ctx, "task_metadata", {}) or {}
    return (ctx.task_type not in {"presence", "review", "summarize"}
            and plan.context_task.get("delegation_role") != "subagent"
            and metadata.get("delegation_role") != "subagent"
            and not json.loads(plan.chronicle_state_json).get("is_child"))


def _fit_existing_refused_memory(ctx: _RoundModelCallContext) -> None:
    """Use ready meaningful alternatives without changing books or buying work."""
    if not getattr(ctx.context_fit_plan, "chronicle_state_json", ""):
        return
    plan = replace(ctx.context_fit_plan, context_task={
        **ctx.context_fit_plan.context_task, "memory_refusal_recovery": True,
        "refused_memory_bytes": ctx.context_fit_plan.projection(ctx.active_context_mode).memory_facts.get(
            "rendered_memory_bytes")})
    prepared = ctx.prepared_main_frame if ctx.prepared_main_frame is not None else ctx.messages
    plan = plan.fit_prepared_memory(prepared, ctx.tool_schemas, ctx.active_context_mode,
                                   reasoning_effort=ctx.active_effort)
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.messages[:] = plan.reproject_transcript(ctx.messages, ctx.active_context_mode)
    ctx.tools._ctx.messages = ctx.messages
    invalidate_task_cache_splits(ctx.task_id)


def _repair_refused_main_memory(ctx: _RoundModelCallContext, failed: Any, *,
                                refused_digest_ids=None, refused_memory_bytes=None) -> bool:
    """Buy only a useful source-bound reduction after the allowed routes refused."""
    import hashlib
    from ouroboros import consolidator
    from ouroboros.chronicle_view import refresh_chronicle_snapshot
    from ouroboros.send_clock import MainSendClock
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.utils import read_text
    from ouroboros.usage_accounting import current_usage_scope, usage_scope, bind_physical_attempt_context

    if _memory_recovery_held(ctx.accumulated_usage):
        return False
    root = canonical_data_root(ctx.tools._ctx)
    before = _main_frame_bytes(ctx)
    if refused_memory_bytes is None:
        refused_memory_bytes = ctx.context_fit_plan.projection(ctx.active_context_mode).memory_facts.get("rendered_memory_bytes")
    captured_plan = replace(ctx.context_fit_plan,
        context_task={**ctx.context_fit_plan.context_task, "memory_refusal_recovery": True,
                      "refused_memory_bytes": refused_memory_bytes})
    source = captured_plan.chronicle_state_json
    refused_facts = ctx.context_fit_plan.projection(ctx.active_context_mode).memory_facts
    demand = {"purpose": "actual_context_refusal", "rendered_mode": ctx.active_context_mode,
              "refused_digest_ids": sorted(refused_digest_ids if refused_digest_ids is not None
                                           else refused_facts.get("selected_digest_ids") or []),
              "route_fingerprint": captured_plan.route_fp,
              "failed_candidate_sha256": failed.candidate_raw_sha256,
              "failed_candidate_context_bytes": failed.candidate_context_size_bytes}

    def fits():
        snapshot = refresh_chronicle_snapshot(source, root)
        plan = replace(captured_plan, chronicle_state_json=snapshot,
            core_sha256=(hashlib.sha256((captured_plan.core_sha256 + snapshot).encode("utf-8")).hexdigest()
                         if snapshot != source else captured_plan.core_sha256))
        prepared = (plan.reproject_transcript(ctx.prepared_main_frame, ctx.active_context_mode)
                    if ctx.prepared_main_frame is not None else ctx.messages)
        plan = plan.fit_prepared_memory(prepared, ctx.tool_schemas, ctx.active_context_mode,
                                       reasoning_effort=ctx.active_effort)
        ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
        ctx.messages[:] = plan.reproject_transcript(ctx.messages, ctx.active_context_mode)
        ctx.tools._ctx.messages = ctx.messages
        facts = plan.projection(ctx.active_context_mode).memory_facts
        demand.update(rendered_memory_tokens=facts.get("rendered_memory_tokens"),
                      memory_budget_tokens=facts.get("requested_memory_tokens"),
                      complete_frame_bytes=_main_frame_bytes(ctx))
        # The actual rejected frame establishes a need, not a guessed capacity.
        # Stop at a useful whole-request reduction; do not chase working margin.
        return demand["complete_frame_bytes"] < before

    if fits():
        return True  # A concurrently published interpretation already helps.
    scope = current_usage_scope()
    with (usage_scope(replace(scope, category="consolidation", source="context_refusal_memory"))
          if scope is not None else contextlib.nullcontext()), bind_physical_attempt_context(None), MainSendClock(None).bound():
        usage = consolidator.consolidate(
            root / "logs/chat.jsonl", root / "memory/dialogue_blocks.json", root / "memory/dialogue_meta.json",
            ctx.llm, read_text(root / "memory/identity.md") if (root / "memory/identity.md").exists() else "",
            knowledge_context=ctx.tools._ctx, compact_chronicle=True, pressure_fits=fits,
            room_registry_root=root, represented_only=True, fitting_demand=demand)
    if usage:
        _loop()._account_compaction_usage(ctx.accumulated_usage, usage, ctx.event_queue, ctx.task_id)
    useful = fits()
    errors = (usage or {}).get("_consolidation_errors", [])
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_memory_prepared", "round": ctx.round_idx,
        "route_fp": captured_plan.route_fp, "fitting_demand": demand,
        "memory_view": ctx.context_fit_plan.projection(ctx.active_context_mode).memory_facts,
        "published_records": int((usage or {}).get("_blocks_written") or 0), "errors": errors})
    for error in errors:
        if error.get("kind") in {"budget_exhausted", "provider_outcome_unknown"}:
            ctx.accumulated_usage["_last_llm_error_kind"] = error["kind"]
            return False
    invalidate_task_cache_splits(ctx.task_id)
    return useful


def _recover_deferred_memory_refusal(tools: Any) -> tuple:
    """Use a source-bound smaller view after configured alternatives were tried."""
    calls = _model_calls()
    pending = getattr(tools._ctx, "_deferred_memory_refusal", None)
    tools._ctx._deferred_memory_refusal = None
    if pending is None:
        return None, None, 0.0
    ctx, failed = pending.call, pending.capture
    refused_digest_ids = pending.refused_digest_ids
    refused_memory_bytes = pending.refused_memory_bytes
    total_cost = 0.0
    while not _memory_recovery_held(ctx.accumulated_usage) and calls._failed_capture_is_comparable(failed):
        # A released fallback failure does not replace this route's definite
        # refusal. New helper/dispatch outcomes below remain authoritative.
        ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
        repaired = _may_repair_main_memory(ctx) and _repair_refused_main_memory(
            ctx, failed, refused_digest_ids=refused_digest_ids, refused_memory_bytes=refused_memory_bytes)
        if _memory_recovery_held(ctx.accumulated_usage):
            break
        if not repaired:
            if ctx.active_context_mode != "max":
                if calls._strict_context_shrink_predicate(pending.capture)(failed):
                    return ctx, None, total_cost  # A repaired owner-Low view may fit another route.
                break
            # Only exhausted meaningful Max recovery releases resident books.
            calls._reproject_actual_overflow_low(ctx)
        disposition = calls._measure_after_reclaim(ctx)
        if disposition is None:
            break
        try:
            message, cost = _loop()._dispatch_round_model(ctx, disposition, attempt_cap=1,
                candidate_predicate=calls._strict_context_shrink_predicate(failed))
        except PhysicalAttemptPreconditionFailed:
            calls._emit_overflow_retry_skipped(ctx, "context_candidate_not_strictly_smaller")
            break
        total_cost = None if total_cost is None or cost is None else total_cost + cost
        if message is not None or ctx.accumulated_usage.get("_last_llm_error_kind") != "context_overflow":
            return ctx, message, total_cost
        current = _loop().last_physical_attempt_capture()
        if (not calls._failed_capture_is_comparable(current)
                or not calls._strict_context_shrink_predicate(failed)(current)):
            break
        # A new definite refusal may need a further representation. Progress
        # is the actual smaller frame plus another source-bound view, never a
        # retry counter, fictional round, advertised capacity or unchanged send.
        failed = current
        refused_digest_ids = refused_memory_bytes = None  # This plan produced the newly refused frame.
        if not repaired:
            return ctx, None, total_cost  # Configured routes can use this final reduced view.
    return None, None, total_cost


def _prepare_first_main_memory(ctx: _RoundModelCallContext, prepared: list):
    """Fit published meanings after the complete first-request frame exists.

    A soft target miss does not commission a rewrite of the old biography.
    Existing meanings may occupy working headroom during the lazy transition;
    response reserve, physical measurements and actual-overflow recovery stay.
    """
    import datetime
    calls = _model_calls()
    from ouroboros.context_fit import measure_main_fit
    from ouroboros.send_clock import main_clock_policy, render_clock_note

    plan, mode = ctx.context_fit_plan, ctx.active_context_mode
    policy = main_clock_policy(getattr(ctx.tools._ctx, "task_metadata", {}), task_type=ctx.task_type)
    clock = ([{"role": "user", "content": render_clock_note(policy, datetime.datetime.now(datetime.timezone.utc))}]
             if policy is not None else [])
    # An unresolved Main attempt still owns its candidate. Preparation is not
    # permission to rewrite/repeat that attempt or spend a second helper call.
    held = (TRANSPORT_DEATHS_KEY in ctx.accumulated_usage
            or ctx.accumulated_usage.get("_pending_transport_outcome")
            or ctx.accumulated_usage.get("_last_llm_error_kind") == "provider_outcome_unknown")
    if not held:
        plan = plan.fit_prepared_memory(prepared, ctx.tool_schemas, mode,
            reasoning_effort=ctx.active_effort, clock_messages=clock)
        ctx.context_fit_plan = plan
        ctx.tools._ctx.context_fit_plan = plan
        ctx.messages[:] = plan.reproject_transcript(ctx.messages, mode)
        ctx.tools._ctx.messages = ctx.messages
        prepared = plan.reproject_transcript(prepared, mode)
        invalidate_task_cache_splits(ctx.task_id)
    disposition = measure_main_fit(plan, [*prepared, *clock], ctx.tool_schemas,
        profile=calls._main_context_profile(plan, mode), rendered_mode=mode,
        round_id=calls._context_fit_round_id(ctx), automatic_pass_used=True, reasoning_effort=ctx.active_effort)
    # No working history was summarized here: first-turn sources are unconsumed.
    disposition = replace(disposition, automatic_pass_used=False)
    calls._remember_main_fit(ctx, disposition)
    ctx.prepared_main_frame = copy.deepcopy([*prepared, *clock])
    return prepared, calls._physical_context_for_fit(disposition)


def _reproject_actual_overflow_low(ctx: _RoundModelCallContext) -> None:
    if ctx.active_context_mode != "max" or ctx.context_fit_plan is None:
        return
    ctx.context_fit_plan = replace(ctx.context_fit_plan, rendered_mode="low")
    ctx.messages[:] = ctx.context_fit_plan.reproject_transcript(ctx.messages, "low")
    invalidate_task_cache_splits(ctx.task_id)
    ctx.active_context_mode = "low"
    ctx.tools._ctx.messages = ctx.messages
    ctx.tools._ctx.active_context_mode = "low"
    ctx.tools._ctx.context_fit_plan = ctx.context_fit_plan
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "context_fit_low_retry",
        "round": ctx.round_idx,
        "route_fp": str(getattr(ctx.context_fit_plan, "route_fp", "") or ""),
        "preferred_mode": str(getattr(ctx.context_fit_plan, "preferred_mode", "") or ""),
        "effective_mode": "low",
        "owner_visible": True,
    })
