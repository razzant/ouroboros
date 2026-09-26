"""Budget rails of the main loop: the per-round budget check, cost ceilings and
tree accounting, the soft landing, the loop-exit context, the budget-exceeded
handler, resource cleanup, service finalization and the post-tool budget
context. Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import queue

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from ouroboros import budget_pause
from ouroboros import task_pacing
from ouroboros.loop_transport import TransportWaitEpisode, end_episode_budget as _end_episode_budget
from ouroboros.tools.registry import ToolRegistry
from ouroboros.context_fit import messages_carry_native_images
from ouroboros.usage_accounting import BudgetExceeded


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_delivery import DeliveryCandidate

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_round_limits import _RoundLimitContext


log = logging.getLogger("ouroboros.loop")


def _loop():
    """The parent loop module, read at call time.

    The loop's members stay monkeypatch-addressable at their historical
    ``ouroboros.loop`` bindings (tests rebind them there), so this leaf
    resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import loop

    return loop


def _check_budget_limits(
    ctx: "_RoundLimitContext",
    budget_remaining_usd: Optional[float],
    cost_ceiling: Optional["task_pacing.CostCeiling"] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    accumulated_usage = ctx.accumulated_usage
    raw_task_cost = accumulated_usage.get("cost")
    task_cost = float(raw_task_cost) if raw_task_cost is not None else None

    if budget_remaining_usd is not None and budget_remaining_usd <= 0:
        finish_reason = "🚫 Task rejected. Total budget exhausted. Please increase TOTAL_BUDGET in settings."
        accumulated_usage["execution_status"] = "failed"
        accumulated_usage["reason_code"] = "budget_exhausted"
        if ctx.round_idx <= 1 and not accumulated_usage.get("rounds"):
            trace = ctx.llm_trace if isinstance(ctx.llm_trace, dict) else {}
            tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
            suffix = (
                _loop()._force_plan_disclosure(tool_ctx, trace, forced_reason="budget_exhausted")
                if tool_ctx is not None else ""
            )
            _loop()._record_forced_finalization(
                ctx,
                trace,
                reason_code="budget_exhausted",
                source="host_budget_rejection_before_work",
                candidate=None,
            )
            return _loop()._compose_delivery_suffix(finish_reason, suffix), accumulated_usage, trace
        # After real work the exhaustion is an exact PAUSE (#1196), not a wrap-up call.
        budget_pause.request_pause(ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED, scope="global",
                                   reason_text=finish_reason)
        return _loop()._forced_final_answer(
            ctx,
            prompt=(
                "[BUDGET LIMIT] Total budget exhausted. Produce your best final answer NOW "
                "from the verified work so far; clearly mark anything unverified or "
                "incomplete. An honest best-effort result is the expected outcome here."
            ),
            fallback_text=finish_reason,
            reason_code="budget_exhausted",
        )
    if cost_ceiling is None or cost_ceiling.state != task_pacing.COST_CEILING_ACTIVE:
        return None
    # A tree-capped ceiling is the ROOT's money (its fence covers the tree); a
    # global-share ceiling is global money.
    pause_scope = "root" if cost_ceiling.root_cap_usd is not None else "global"
    tree_info = _loop()._loop_tree_accounting(refresh=True, max_age_sec=_loop()._TREE_ACCOUNTING_MAX_STALE_SEC)
    tree_cost = tree_info.get("accounted_usd") if isinstance(tree_info, dict) else None
    deciding, spend_basis = task_pacing.resolve_deciding_spend(
        tree_cost_usd=tree_cost,
        task_cost_usd=task_cost,
        root_cap_usd=cost_ceiling.root_cap_usd,
    )
    ceiling_usd = cost_ceiling.ceiling_usd
    prompt_estimate = int(accumulated_usage.get("_context_prompt_estimate") or 0)
    global_remaining = _wrapup_global_remaining() if prompt_estimate > 0 and not ctx.active_use_local else None
    wrapup_fits = None
    # Owner Q10 (#1196): after an explicit Resume of a GRACEFUL pause the
    # last-fit rail no longer demands room for TWO reservations. The owner's
    # Resume spent exactly that early margin, so the one call that fits in the
    # already-authorized remainder is admitted instead of re-pausing on the
    # very number that paused the task; the ledger fence at the full cap still
    # arbitrates every send. A stop that needs no wrap-up room at all
    # (``wrapup_fits is False``) is unchanged.
    last_fit_relaxed = bool(getattr(getattr(getattr(ctx, "tools", None), "_ctx", None),
                                    "_budget_resume_last_fit_relaxed", False))
    if prompt_estimate > 0 and (global_remaining is not None or (cost_ceiling.root_cap_usd is not None and deciding is not None)):
        finish_reason = task_pacing.wrapup_last_fit_text(deciding, cost_ceiling, global_remaining)
        forced_prompt = f"[BUDGET LIMIT] {finish_reason} {_loop()._FORCED_BEST_EFFORT_TAIL}"
        request_args = dict(model=ctx.active_model, prompt_tokens=prompt_estimate,
                            use_local=ctx.active_use_local)
        balances = dict(root_cap_usd=cost_ceiling.root_cap_usd, deciding_usd=deciding,
                        global_remaining_usd=global_remaining)
        wrapup_args = dict(**request_args, **balances)
        wrapup_fits = task_pacing.wrapup_reservation_fits(**wrapup_args)
        two_fit = _second_reservation_fits(ctx, wrapup_args, wrapup_fits, relaxed=last_fit_relaxed)
        server_web = _loop()._server_web_allowed_by_task(getattr(getattr(ctx, "tools", None), "_ctx", None))
        if wrapup_fits is False or two_fit is False or (
            wrapup_fits is True and messages_carry_native_images(ctx.messages)
        ):
            # Pre-screen only: every proxy stop, and every image prompt the proxy
            # understates, is priced exactly on a COPY of the transcript (no
            # service finalization) before anything destructive happens.
            probe_messages = [dict(message) for message in ctx.messages]
            _loop()._append_or_merge_user_message(probe_messages, forced_prompt)
            probe = task_pacing.prospective_wrapup_attempt_request(
                llm=ctx.llm, messages=probe_messages, model=ctx.active_model,
                reasoning_effort=ctx.active_effort, tools=ctx.tool_schemas,
                allow_server_web_search=server_web, prompt_tokens=prompt_estimate,
            )
            wrapup_args = dict(request=probe, **balances)
            wrapup_fits = task_pacing.wrapup_reservation_fits(**wrapup_args)
            two_fit = _second_reservation_fits(ctx, wrapup_args, wrapup_fits, relaxed=last_fit_relaxed)
        if wrapup_fits is False or two_fit is False:
            # The exact probe confirmed a stop: finalize services and prepare the
            # candidate that will be dispatched (forced augmentations included).
            trace = ctx.llm_trace if isinstance(ctx.llm_trace, dict) else {}
            tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
            presence_arm = (getattr(tools_ctx, "_presence_forced_declaration", None),
                            getattr(tools_ctx, "_presence_forced_pending", None))
            priced_prompt = _loop()._prepare_forced_prompt(ctx, forced_prompt, trace)
            prospective_messages = [dict(message) for message in ctx.messages]
            _loop()._append_or_merge_user_message(prospective_messages, priced_prompt)
            wrapup_request, send_messages = task_pacing.prepared_wrapup_candidate(
                ctx, prospective_messages, allow_server_web_search=server_web,
            )
            wrapup_args = dict(request=wrapup_request, **balances)
            wrapup_fits = task_pacing.wrapup_reservation_fits(**wrapup_args)
            if wrapup_fits is False:
                accumulated_usage["cost_stop_spend_basis"] = spend_basis
                accumulated_usage["cost_stop_rail"] = "wrapup_reservation_last_fit"
                budget_pause.request_pause(
                    ctx, rail=budget_pause.RAIL_WRAPUP_LAST_FIT, scope=pause_scope,
                    reason_text=task_pacing.wrapup_unaffordable_text(deciding, cost_ceiling, global_remaining))
                return _loop()._forced_fallback_result(
                    ctx, trace, task_pacing.wrapup_unaffordable_text(deciding, cost_ceiling, global_remaining),
                    "budget_exhausted", source="budget_wrapup_unaffordable",
                )
            if wrapup_fits is True and _second_reservation_fits(
                ctx, wrapup_args, wrapup_fits, relaxed=last_fit_relaxed,
            ) is False:
                accumulated_usage["cost_stop_spend_basis"] = spend_basis
                accumulated_usage["cost_stop_rail"] = "wrapup_reservation_last_fit"
                budget_pause.request_pause(ctx, rail=budget_pause.RAIL_WRAPUP_LAST_FIT,
                                           scope=pause_scope, reason_text=finish_reason)
                return _loop()._forced_final_answer(
                    ctx, prompt=priced_prompt, _prompt_prepared=True,
                    fallback_text=finish_reason, reason_code="budget_exhausted",
                    _initial_messages=send_messages, _admitted_request=wrapup_request,
                )
            # Repricing admitted ordinary work after all. No forced call is committed, so the
            # Presence arm the prepared prompt set is withdrawn: it would silence the ordinary reply.
            if tools_ctx is not None:
                tools_ctx._presence_forced_declaration, tools_ctx._presence_forced_pending = presence_arm
    if deciding is not None and ceiling_usd is not None and deciding > ceiling_usd:
        if spend_basis == task_pacing.SPEND_BASIS_TREE:
            spent_text = (
                f"Task tree spent ${deciding:.3f} (ledger-accounted incl. in-flight holds, "
                f"subagents included; own calls ${task_cost:.3f})"
                if task_cost is not None
                else f"Task tree spent ${deciding:.3f} (ledger-accounted incl. in-flight holds)"
            )
        elif spend_basis == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN:
            spent_text = (
                f"Task spent ${deciding:.3f} on its OWN calls (the tree-accounted total "
                "is unavailable right now, so subagent spend is not included — this is a "
                "lower bound)"
            )
        else:
            spent_text = f"Task spent ${deciding:.3f}"
        cap_text = (
            f"; the hard tree cap is ${cost_ceiling.root_cap_usd:.2f}"
            if cost_ceiling.root_cap_usd is not None else ""
        )
        finish_reason = (
            f"{spent_text}, over the in-task cost ceiling ${ceiling_usd:.2f}{cap_text}. "
            "Budget exhausted."
        )
        accumulated_usage["cost_stop_spend_basis"] = spend_basis
        budget_pause.request_pause(ctx, rail=budget_pause.RAIL_GRACEFUL_CEILING,
                                   scope=pause_scope, reason_text=finish_reason)
        return _loop()._forced_final_answer(
            ctx,
            prompt=f"[BUDGET LIMIT] {finish_reason} {_loop()._FORCED_BEST_EFFORT_TAIL}",
            fallback_text=finish_reason,
            reason_code="budget_exhausted",
        )
    return None


def _second_reservation_fits(ctx: "_RoundLimitContext", wrapup_args: Dict[str, Any],
                             wrapup_fits: Optional[bool], *, relaxed: bool) -> Optional[bool]:
    """The two-reservation (last-fit) probe, or ``None`` when it does not decide.

    Priced only while ONE reservation fits (otherwise the harder stop already
    decides). Under the Q10 relaxation the probe still runs, so the admitted
    early call is DISCLOSED on the usage record, but it no longer stops the
    task: the one affordable call is placed and the ledger fence keeps binding.
    """
    if wrapup_fits is not True:
        return None
    second = task_pacing.wrapup_reservation_fits(**wrapup_args, reservation_count=2)
    if relaxed and second is False:
        ctx.accumulated_usage["budget_resume_last_fit_admitted"] = {
            "round_idx": int(ctx.round_idx), "reservations_affordable": 1,
            "basis": "owner_resume_relaxed_last_fit"}
        return None
    return second


def _resolve_task_cost_ceiling(
    ctx: Any, budget_remaining_usd: Optional[float],
) -> "task_pacing.CostCeiling":
    """Return and retain the task's once-resolved cost stop."""
    disclosed = getattr(ctx, "_cost_ceiling", None)
    if isinstance(disclosed, task_pacing.CostCeiling):
        return disclosed
    resolved = task_pacing.resolve_task_cost_ceiling(ctx, budget_remaining_usd)
    setattr(ctx, "_cost_ceiling", resolved)
    return resolved


def _wrapup_global_remaining() -> Optional[float]:
    """Read one fresh wallet observation for this phase's candidate comparisons."""
    from ouroboros.usage_accounting import current_usage_scope, usage_projection

    scope = current_usage_scope()
    if scope is None or not scope.task_id:
        return None
    try:
        projection = usage_projection(scope.drive_root, global_limit_usd=scope.global_limit_usd,
                                      include_roots=False)
        if projection.get("integrity_degraded"):
            return None
        remaining = projection.get("remaining_known_usd")
        return None if remaining is None else float(remaining)
    except Exception:
        log.debug("Wrap-up global budget observation unavailable", exc_info=True)
        return None


_TREE_ACCOUNTING_MAX_STALE_SEC = 120.0


def _loop_tree_accounting(
    *, refresh: bool, max_age_sec: float = 30.0, strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return nullable, bounded-stale spend for the current task's root tree.

    ``strict`` is the money reader's contract (``refresh_root_accounting``):
    one fresh successful observation or ``None``, never the display cache.
    """
    try:
        from ouroboros.usage_accounting import (
            current_usage_scope,
            last_root_accounting,
            refresh_root_accounting,
        )

        scope = current_usage_scope()
        if scope is None or not scope.root_task_id:
            return None
        if refresh:
            return refresh_root_accounting(scope.drive_root, scope.root_task_id,
                                           max_age_sec=max_age_sec, strict=strict)
        return last_root_accounting(scope.root_task_id)
    except Exception:
        log.debug("Tree accounting telemetry unavailable", exc_info=True)
        return None


def _soft_land_exhausted_ceiling(
    limit_ctx: "_RoundLimitContext",
    cost_ceiling: "task_pacing.CostCeiling",
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Pause a root cap at/below the planning margin, even before its first
    call: owner Resume may use already-authorized headroom. Only an actor
    without exact continuation retains the priced legacy soft landing."""
    if cost_ceiling.state != task_pacing.COST_CEILING_EXHAUSTED_SOFT_LAND:
        return None
    cap_text = (
        f"${cost_ceiling.root_cap_usd:.2f}"
        if cost_ceiling.root_cap_usd is not None else "the per-task tree cap"
    )
    margin_text = (
        f"${cost_ceiling.planning_margin_usd:.2f}"
        if cost_ceiling.planning_margin_usd is not None else "the wrap-up planning margin"
    )
    soft_land_reason = (
        f"Per-task tree cap {cap_text} leaves no working room above the "
        f"wrap-up planning margin ({margin_text}). Budget exhausted."
    )
    budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_SOFT_LAND, scope="root",
                               reason_text=soft_land_reason)
    trace = limit_ctx.llm_trace if isinstance(limit_ctx.llm_trace, dict) else {}
    priced_prompt = _loop()._prepare_forced_prompt(
        limit_ctx, f"[BUDGET LIMIT] {soft_land_reason} {_loop()._FORCED_BEST_EFFORT_TAIL}", trace,
    )
    prospective = [dict(message) for message in limit_ctx.messages]
    _loop()._append_or_merge_user_message(prospective, priced_prompt)
    request, send_messages = task_pacing.prepared_wrapup_candidate(
        limit_ctx, prospective, allow_server_web_search=_loop()._server_web_allowed_by_task(
            getattr(getattr(limit_ctx, "tools", None), "_ctx", None)),
    )
    tree_info = _loop()._loop_tree_accounting(refresh=True, max_age_sec=0.0)
    deciding, spend_basis = task_pacing.resolve_deciding_spend(
        tree_cost_usd=tree_info.get("accounted_usd") if isinstance(tree_info, dict) else None,
        task_cost_usd=float(limit_ctx.accumulated_usage.get("cost") or 0.0),
        root_cap_usd=cost_ceiling.root_cap_usd,
    )
    limit_ctx.accumulated_usage["cost_stop_spend_basis"] = spend_basis
    if task_pacing.wrapup_reservation_fits(
        request=request, root_cap_usd=cost_ceiling.root_cap_usd, deciding_usd=deciding or 0.0,
        global_remaining_usd=_wrapup_global_remaining() if not limit_ctx.active_use_local else None,
        use_local=limit_ctx.active_use_local,
    ) is False:
        limit_ctx.accumulated_usage["cost_stop_rail"] = "wrapup_reservation_last_fit"
        return _loop()._forced_fallback_result(
            limit_ctx, trace, soft_land_reason, "budget_exhausted",
            source="budget_wrapup_unaffordable",
        )
    return _loop()._forced_final_answer(
        limit_ctx, prompt=priced_prompt, _prompt_prepared=True,
        fallback_text=soft_land_reason, reason_code="budget_exhausted",
        _initial_messages=send_messages, _admitted_request=request,
    )


def _service_finalization_evidence(llm_trace: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return the stable, answer-relevant part of service finalization events."""

    rows: list[Dict[str, Any]] = []
    stable_fields = (
        "service_id",
        "name",
        "task_id",
        "lifecycle",
        "backend",
        "pid",
        "port",
        "artifact_outputs",
        "artifact_output_failed",
        "artifact_audit_gap",
        "log_finalization",
    )
    for event in llm_trace.get("verification_events") or []:
        if not isinstance(event, dict) or str(event.get("kind") or "") not in {
            "services_stopped",
            "services_kept",
            "service_finalization_error",
        }:
            continue
        services = []
        for service in event.get("services") or []:
            if not isinstance(service, dict):
                continue
            services.append({
                key: service.get(key)
                for key in stable_fields
                if service.get(key) not in (None, "", [], {})
            })
        rows.append({
            "kind": str(event.get("kind") or ""),
            "services": services,
            "error": str(event.get("error") or ""),
        })
    return rows


@dataclass
class _LoopExitContext:
    tools: ToolRegistry
    drive_root: Optional[pathlib.Path]
    task_id: str
    event_queue: Optional[queue.Queue]
    drive_logs: pathlib.Path
    accumulated_usage: Dict[str, Any]
    llm_trace: Dict[str, Any]
    trace_ctx: Any = None
    previous_execution_trace: Any = None

    def attach_exception_evidence(self, exc: Exception) -> None:
        """The caller owns terminal projection; this loop owns its evidence.

        Keep the same in-memory objects on the original exception so a lifecycle
        failure cannot erase a completed multi-round trace. A failed attachment
        leaves the caller's explicit unknown projection, never a new exception.
        """
        try:
            setattr(exc, "_ouroboros_loop_usage", self.accumulated_usage)
            setattr(exc, "_ouroboros_loop_trace", self.llm_trace)
        except Exception:
            log.debug("Loop exception evidence could not be attached", exc_info=True)


def _handle_budget_exceeded(
    exc: BudgetExceeded,
    ctx: _LoopExitContext,
    *,
    limit_ctx: Optional[_RoundLimitContext] = None,
    episode: Optional[TransportWaitEpisode] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Apply the physical-attempt dispatch rail without spending a wrap-up call."""
    if episode is not None:
        # Budget rail fired mid transport-wait: close the episode's durable story.
        _end_episode_budget(
            episode, ctx.drive_logs, ctx.task_id,
            limit_ctx.active_model if limit_ctx is not None else "")
    physical_calls: Optional[int] = None
    try:
        from ouroboros.usage_accounting import usage_breakdown

        budget_root = (
            getattr(ctx.tools._ctx, "budget_drive_root", None)
            or ctx.drive_root
            or getattr(ctx.tools._ctx, "drive_root", None)
        )
        if budget_root is not None:
            attempt_evidence = usage_breakdown(
                pathlib.Path(budget_root), task_id=str(ctx.task_id),
            )
            physical_calls = int(attempt_evidence.get("physical_calls") or 0)
            if attempt_evidence.get("integrity_degraded"):
                physical_calls = None
    except Exception:
        log.exception("Could not inspect task attempts after budget rail")
    direct_chat = bool(getattr(ctx.tools._ctx, "is_direct_chat", False))
    replay_safe = physical_calls == 0 and not direct_chat
    scope = str(getattr(exc, "limit_scope", "global") or "global")
    resource_limit = {
        "status": "paused_before_dispatch" if replay_safe else "resource_limited",
        "scope": scope,
        "root_task_id": str(getattr(exc, "root_task_id", "") or ""),
        "physical_calls": physical_calls,
        "replay_safe": replay_safe,
        "auto_resume": False,
        "resume_policy": (
            "increase_or_reset_budget_then_retry"
            if direct_chat
            else ("manual_same_generation" if replay_safe else "cancel_or_new_run")
        ),
    }
    if replay_safe:
        raise exc
    if limit_ctx is not None:
        # A refused dispatch after real work: exact pause (#1196). Ineligible
        # actors fall through to the terminal projection below, loudly.
        limit_ctx.tools = ctx.tools
        limit_ctx.llm_trace = ctx.llm_trace
        budget_pause.request_pause(
            limit_ctx, rail=budget_pause.RAIL_DISPATCH_REFUSED, scope=scope,
            reason_text=str(exc), root_task_id=resource_limit["root_task_id"])
        resource_limit["exact_pause_unavailable"] = str(
            ctx.accumulated_usage.get("exact_pause_unavailable") or "")
    ctx.accumulated_usage["execution_status"] = "failed"
    ctx.accumulated_usage["reason_code"] = "budget_exhausted"
    ctx.accumulated_usage["resource_limit"] = resource_limit
    ctx.llm_trace["resource_limit"] = resource_limit
    _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "budget_scope_paused",
        "owner_visible": True,
        "toast_once": f"{ctx.task_id}:budget-paused:{scope}",
        **resource_limit,
    })
    if (
        scope == "root"
        and ctx.event_queue is not None
        and not direct_chat
    ):
        try:
            ctx.event_queue.put_nowait({
                "type": "budget_root_fence",
                "task_id": ctx.task_id,
                "root_task_id": resource_limit["root_task_id"],
                "resource_limit": resource_limit,
            })
        except Exception:
            log.error("Could not publish root budget fence for %s", ctx.task_id, exc_info=True)
    # A physical budget rail is terminal for this execution. Finalize task
    # services before testing or creating a DeliveryCandidate so no
    # pre-teardown answer is published on stale service/output evidence. The
    # outer cleanup repeats this helper only as an idempotent safety net.
    if limit_ctx is not None:
        limit_ctx.tools = ctx.tools
        limit_ctx.llm_trace = ctx.llm_trace
        _loop()._finalize_forced_services(limit_ctx, ctx.llm_trace)
    else:
        _loop()._finalize_task_services(ctx)
    candidate_seen: Optional[DeliveryCandidate] = None
    if limit_ctx is not None:
        # The exception can arrive after a substantive answer entered a service
        # re-loop. Re-read the live evidence now; the round-start snapshot alone
        # cannot prove that candidate is still current.
        limit_ctx.tools = ctx.tools
        limit_ctx.llm_trace = ctx.llm_trace
        candidate_seen = _loop()._live_delivery_candidate(limit_ctx)
        current_candidate = _loop()._current_delivery_candidate(limit_ctx, ctx.llm_trace)
        if current_candidate is not None:
            return _loop()._forced_fallback_result(
                limit_ctx,
                ctx.llm_trace,
                current_candidate.full_text,
                "budget_exhausted",
                source="budget_host_fallback",
                retained_source="budget_preserve",
                retained_control="budget_preserve",
            )
        if candidate_seen is not None:
            candidate_seen.degraded = True
            candidate_seen.degraded_reason = "budget_exhausted"
            candidate_seen.finalization_control = "budget_stale_rejected"
            _loop()._publish_delivery_candidate(ctx.tools, candidate_seen, ctx.llm_trace)
    latched = str(ctx.llm_trace.get("best_valid_final_answer") or "").strip()
    latched_is_current = (
        latched
        and len(ctx.llm_trace.get("tool_calls") or [])
        <= int(ctx.llm_trace.get("best_valid_final_answer_tools") or 0)
    )
    if latched_is_current:
        ctx.accumulated_usage["_best_effort_extracted"] = True
        if limit_ctx is not None:
            return _loop()._forced_fallback_result(
                limit_ctx,
                ctx.llm_trace,
                latched,
                "budget_exhausted",
                source="budget_latched_fallback",
            )
        return latched, ctx.accumulated_usage, ctx.llm_trace
    if candidate_seen is not None and limit_ctx is not None:
        return _loop()._forced_fallback_result(
            limit_ctx,
            ctx.llm_trace,
            candidate_seen.full_text,
            "budget_exhausted",
            source="budget_stale_candidate_preserved",
        )
    message = (
        "🚫 Model budget exhausted before another model dispatch. Increase or reset "
        "the global/root budget, then retry or resume the request. Starting a new run "
        "before changing the budget will hit the same limit."
        if direct_chat
        else (
            "🚫 Resource limit reached before another model dispatch. The task was not "
            "auto-resumed; cancel it or start a new run unless the recorded checkpoint "
            "is explicitly replay-safe."
        )
    )
    if limit_ctx is not None:
        return _loop()._forced_fallback_result(
            limit_ctx,
            ctx.llm_trace,
            message,
            "budget_exhausted",
            source="budget_host_fallback",
        )
    return message, ctx.accumulated_usage, ctx.llm_trace


def _cleanup_loop_resources(
    stateful_executor: Any,
    ctx: _LoopExitContext,
) -> None:
    """Release attempt-scoped executors, services, and delegated runs."""
    if ctx.trace_ctx is not None:
        ctx.trace_ctx._execution_trace = ctx.previous_execution_trace
    # This attempt's tool-future quiescence rows end with it (#1196): a pause
    # only got here after they settled, and any other exit is on the task's own
    # control rail; the registry prunes nothing across attempts by itself.
    try:
        budget_pause.forget_tool_scope(ctx.tools._ctx)
    except Exception:
        log.debug("Tool-future registry scope could not be dropped", exc_info=True)
    if stateful_executor:
        try:
            from ouroboros.tools.browser import cleanup_browser

            stateful_executor.submit(cleanup_browser, ctx.tools._ctx).result(timeout=5)
        except Exception:
            log.debug("Browser cleanup on executor thread failed or timed out", exc_info=True)
        try:
            stateful_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            log.warning("Failed to shutdown stateful executor", exc_info=True)
    _loop()._finalize_task_services(ctx)
    # The full DeliveryCandidate is loop-local: only its compact
    # hash/revision projection remains in llm_trace after this cleanup. Clear
    # it after the idempotent teardown safety net so cleanup cannot erase the
    # only complete answer before service evidence lands.
    ctx.tools._ctx._delivery_candidate = None
    ctx.tools._ctx._delivery_control_required = False
    if ctx.drive_root is None or not ctx.task_id:
        return
    if getattr(ctx.tools._ctx, "_budget_pausing", False):
        # The task is NOT terminal: its delegated runs stay under its custody
        # (observed and stop-requested on the durable pause row); the periodic
        # sweep keeps covering them. A terminal reconciliation here would
        # misstate a nonterminal task as ended.
        return
    try:
        from ouroboros.delegate_custody import custody_root, release_task_runs

        # A delegated run is a resource this task HOLDS, like a service or
        # an executor: a terminalized parent leaving one running has a
        # mutating process nothing is watching. The durable reconciler still
        # covers a worker dying before here; this is the ordinary path. The
        # run is cancelled only behind a DELIBERATE owner terminal (B1-A); the
        # durable result is usually not written yet at this point, so a live
        # run is left to the next sweep, which re-reads it.
        release_task_runs(custody_root(ctx.tools._ctx), ctx.task_id)
    except Exception:
        log.debug("Failed to release delegated runs for task %s", ctx.task_id, exc_info=True)


def _service_identity_projection(service: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded identity used to deduplicate idempotent teardown observations."""

    fields = (
        "service_id",
        "name",
        "task_id",
        "lifecycle",
        "backend",
        "pid",
        "port",
        "artifact_outputs",
        "artifact_output_failed",
        "artifact_audit_gap",
        "log_finalization",
    )
    return {
        key: service.get(key)
        for key in fields
        if service.get(key) not in (None, "", [], {})
    }


def _finalize_task_services(ctx: _LoopExitContext) -> bool:
    """Finalize newly observed task services and record answer-bound evidence.

    Returns True only when a new stopped/kept/error observation was added.  The
    same helper is safe both immediately before acceptance and from ``finally``.
    """

    if ctx.drive_root is None or not ctx.task_id:
        return False
    try:
        from ouroboros.tools.services import stop_task_services

        finalized = stop_task_services(ctx.tools._ctx)
        seen = getattr(ctx.tools._ctx, "_service_finalization_signatures", None)
        if not isinstance(seen, set):
            seen = set()
            ctx.tools._ctx._service_finalization_signatures = seen
        fresh = []
        for service in finalized:
            if not isinstance(service, dict):
                continue
            signature = hashlib.sha256(json.dumps(
                _service_identity_projection(service),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")).hexdigest()
            if signature in seen:
                continue
            seen.add(signature)
            fresh.append(service)
        stopped = [service for service in fresh if service.get("lifecycle") != "kept"]
        kept = [service for service in fresh if service.get("lifecycle") == "kept"]
        if stopped:
            _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
                "checkpoint_kind": "services_stopped",
                "services": stopped,
            })
            ctx.llm_trace.setdefault("verification_events", []).append({
                "kind": "services_stopped",
                "services": stopped,
            })
        if kept:
            _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
                "checkpoint_kind": "services_kept",
                "services": kept,
            })
            ctx.llm_trace.setdefault("verification_events", []).append({
                "kind": "services_kept",
                "services": kept,
            })
        return bool(stopped or kept)
    except Exception as exc:
        log.debug("Failed to stop task services", exc_info=True)
        event = {
            "kind": "service_finalization_error",
            "services": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
        signature = hashlib.sha256(json.dumps(
            event, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        seen = getattr(ctx.tools._ctx, "_service_finalization_signatures", None)
        if not isinstance(seen, set):
            seen = set()
            ctx.tools._ctx._service_finalization_signatures = seen
        if signature in seen:
            return False
        seen.add(signature)
        ctx.llm_trace.setdefault("verification_events", []).append(event)
        return True


def _finish_tool_round_budget(
    ctx: _RoundLimitContext, budget_remaining_usd: Optional[float],
    cost_ceiling: "task_pacing.CostCeiling", active_model: str,
    active_use_local: bool, active_effort: str, tool_calls: Optional[list[Dict[str, Any]]] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Complete the existing budget decision after warm or cold owner input.

    Only a freshly executed tool batch advances the nanny's metered baseline;
    cold continuation owes the budget decision, never execution of old tools.
    """
    if tool_calls is not None:
        _loop()._note_nanny_delegate_activity(ctx.tools._ctx, ctx.round_idx, ctx.accumulated_usage, tool_calls)
    _loop()._prepare_post_tool_budget_context(
        ctx.tools, ctx, ctx.llm_trace, active_model, active_use_local, active_effort)
    result = _loop()._check_budget_limits(ctx, budget_remaining_usd, cost_ceiling=cost_ceiling)
    if result is None:
        return None
    text, usage, trace = result
    _loop()._merge_finalization_trace(ctx.llm_trace, trace)
    return text, usage, ctx.llm_trace


def _finish_no_tool_round_budget(
    ctx: _RoundLimitContext, budget_remaining_usd: Optional[float],
    cost_ceiling: "task_pacing.CostCeiling",
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """The SAME budget decision after a no-tool round that did not finish.

    A round whose answer was not accepted continues to spend, whether or not it
    called tools — but only the tool tail ever reached `_check_budget_limits`,
    so a task that kept re-answering under review feedback could run past the
    ceiling untouched (#1223). This is the tool tail's own comparison and
    nothing else: no metered-baseline bookkeeping (no tools ran) and no
    `_prepare_post_tool_budget_context`, whose delivery-control arming belongs
    to a tool batch's effects. The current delivery candidate is untouched, so a
    eligible budget exit checkpoints the answer the round produced. Its cold
    continuation must return to this tail without tool-only control arming.
    """
    ctx.budget_tail = "no_tool"
    result = _loop()._check_budget_limits(ctx, budget_remaining_usd, cost_ceiling=cost_ceiling)
    if result is None:
        return None
    text, usage, trace = result
    _loop()._merge_finalization_trace(ctx.llm_trace, trace)
    return text, usage, ctx.llm_trace


def _prepare_post_tool_budget_context(
    tools: ToolRegistry,
    limit_ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    active_model: str,
    active_use_local: bool,
    active_effort: str,
) -> None:
    """Refresh candidate evidence and the actual route before budget wrap-up.

    The evidence read is the typed one (`_delivery_evidence_state`): an
    unreadable fingerprint is UNKNOWN, so a retained answer is never lost here.
    """

    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if isinstance(candidate, _loop().DeliveryCandidate):
        hold_control = (
            candidate.finalization_control
            if candidate.finalization_control in _loop()._DELIVERY_HOLD_CONTROLS
            else ""
        )
        # The absorption gate stays open while undispositioned children remain:
        # arming JSON there would recreate the conflicting-instruction round.
        absorption_gate_open = (
            hold_control == _loop()._CHILD_ABSORPTION_HOLD_CONTROL
            and bool(_loop()._undispositioned_children(limit_ctx))
        )
        evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(
            tools, limit_ctx, llm_trace,
        )
        if (
            candidate.evidence_revision != evidence_revision
            or candidate.evidence_fingerprint != evidence_fingerprint
        ):
            if absorption_gate_open:
                _loop()._hold_delivery_for_skill_action(
                    tools, llm_trace, control=_loop()._CHILD_ABSORPTION_HOLD_CONTROL,
                )
            else:
                _loop()._arm_delivery_control(
                    tools,
                    limit_ctx,
                    llm_trace,
                    control="effect_revision_required",
                )
        elif hold_control == _loop()._SKILL_ACTION_HOLD_CONTROL:
            _loop()._arm_delivery_control(
                tools,
                limit_ctx,
                llm_trace,
                control="skill_revision_required",
            )
        # An absorption hold with unchanged evidence keeps holding: only
        # dispositions close the gate, and a disposition changes evidence.
    # Cross-model fallback can adopt a different route during this round.
    limit_ctx.active_model = active_model
    limit_ctx.active_use_local = active_use_local
    limit_ctx.active_effort = active_effort
