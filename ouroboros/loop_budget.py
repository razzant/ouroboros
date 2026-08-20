"""Budget rails of the main loop: the per-round budget check, cost ceilings and
tree accounting, the soft landing, the loop-exit context, the budget-exceeded
handler, resource cleanup, service finalization and the post-tool budget
context. Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

import json
import hashlib
import queue
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import logging

from ouroboros import task_pacing
from ouroboros.tools.registry import ToolRegistry
from ouroboros.usage_accounting import BudgetExceeded

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_delivery import DeliveryCandidate
    from ouroboros.loop_round_limits import _RoundLimitContext


# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
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
    """Return a final-response tuple when budget limits require stopping.

    ``cost_ceiling`` is the typed in-task stop resolved ONCE at loop start
    (``task_pacing.resolve_cost_ceiling``). Only an ``active`` ceiling stops
    here; ``exhausted_soft_land`` fires at the top of the round. The deciding
    spend is the root subtree's ledger-accounted number when a root cap exists
    (the fence counts the TREE, not own calls); own cost is the DISCLOSED
    fallback and the diagnostic. Unknown spend never becomes $0. The two axes
    are INDEPENDENT (v6.91 fix): ``budget_remaining_usd`` None only means no
    finite GLOBAL budget exists (TOTAL_BUDGET unset — the GAIA-shaped run) and
    must not silence a live per-task ROOT CAP; with neither, the ceiling
    resolves ``disabled`` and the whole cost axis stays silent, as before."""
    accumulated_usage = ctx.accumulated_usage
    raw_task_cost = accumulated_usage.get("cost")
    task_cost = float(raw_task_cost) if raw_task_cost is not None else None

    if budget_remaining_usd is not None and budget_remaining_usd <= 0:
        finish_reason = "🚫 Task rejected. Total budget exhausted. Please increase TOTAL_BUDGET in settings."
        accumulated_usage["execution_status"] = "failed"
        accumulated_usage["reason_code"] = "budget_exhausted"
        if ctx.round_idx <= 1:
            trace = ctx.llm_trace if isinstance(ctx.llm_trace, dict) else {}
            router_result = _loop()._forced_swarm_router_result(ctx, trace, "budget_exhausted")
            if router_result is not None:
                return router_result
            tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
            suffix = (
                _loop()._force_plan_disclosure(
                    tool_ctx, trace, forced_reason="budget_exhausted",
                )
                if tool_ctx is not None else ""
            )
            # This early rejection is a forced sink like every other: nothing
            # was produced, but a queued/headless root still OWED a panel, and
            # returning without the record left `not_eligible / run_count=0` —
            # indistinguishable from "no panel was warranted". Pure ledger
            # write: no panel, no model round, no fence.
            _loop()._record_forced_finalization(
                ctx,
                trace,
                reason_code="budget_exhausted",
                source="host_budget_rejection_before_work",
                candidate=None,
            )
            return _loop()._compose_delivery_suffix(finish_reason, suffix), accumulated_usage, trace
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
    # The pre-v6.91 per-task soft "[COST NOTE]" is gone: since v6.64.0 the same
    # settings key hard-fences the whole TREE at the ledger, so an own-cost note
    # keyed to it could never fire before the fence (proven live: silent through
    # two tree deaths). The v6.56.0 latched milestones are the designed nudge.

    if cost_ceiling is None or cost_ceiling.state != task_pacing.COST_CEILING_ACTIVE:
        return None
    tree_info = _loop()._loop_tree_accounting(
        refresh=True, max_age_sec=_loop()._TREE_ACCOUNTING_MAX_STALE_SEC,
    )
    tree_cost = tree_info.get("accounted_usd") if isinstance(tree_info, dict) else None
    deciding, spend_basis = task_pacing.resolve_deciding_spend(
        tree_cost_usd=tree_cost,
        task_cost_usd=task_cost,
        root_cap_usd=cost_ceiling.root_cap_usd,
    )
    ceiling_usd = cost_ceiling.ceiling_usd
    if deciding is not None and ceiling_usd is not None and deciding > ceiling_usd:
        if spend_basis == task_pacing.SPEND_BASIS_TREE:
            spent_text = (
                f"Task tree spent ${deciding:.3f} (ledger-accounted incl. in-flight holds, "
                f"subagents included; own calls ${task_cost:.3f})"
                if task_cost is not None
                else f"Task tree spent ${deciding:.3f} (ledger-accounted incl. in-flight holds)"
            )
        elif spend_basis == task_pacing.SPEND_BASIS_OWN_TREE_UNKNOWN:
            # Stopping on a disclosed lower bound beats not stopping at all, but
            # the substitution is stated, never silent (BIBLE P1).
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
        # The basis rides the usage record too, so a later reader can tell a
        # tree-decided stop from an own-cost stand-in without parsing prose.
        accumulated_usage["cost_stop_spend_basis"] = spend_basis
        return _loop()._forced_final_answer(
            ctx,
            prompt=(
                f"[BUDGET LIMIT] {finish_reason} Produce your best final answer now from "
                "the verified work so far; clearly mark anything unverified or incomplete. "
                "An honest best-effort result is the expected outcome here, not a failure."
            ),
            fallback_text=finish_reason,
            reason_code="budget_exhausted",
        )
    # The old round-gated "[INFO] ... Wrap up if possible" nudge is replaced by
    # the latched cost milestones in task_pacing (transport: _inject_round_checkpoints).

    return None


def _resolve_task_cost_ceiling(
    ctx: Any, budget_remaining_usd: Optional[float],
) -> "task_pacing.CostCeiling":
    """The typed in-task cost stop, resolved ONCE at loop start.

    The root cap comes from the bound usage scope — the SAME
    ``OUROBOROS_PER_TASK_COST_USD``-derived value the ledger fence enforces
    (agent.py wires it as ``UsageScope.root_limit_usd``), so the graceful stop
    and the fence can never disagree about the cap."""
    root_cap = None
    try:
        from ouroboros.usage_accounting import current_usage_scope

        scope = current_usage_scope()
        root_cap = getattr(scope, "root_limit_usd", None) if scope is not None else None
    except Exception:
        log.debug("Usage scope unavailable for cost ceiling resolution", exc_info=True)
    return task_pacing.resolve_cost_ceiling(
        budget_remaining_usd,
        task_pacing.resolve_budget_profile(ctx),
        root_cap_usd=root_cap,
    )


# Bounded staleness for the two DECIDING cost surfaces (ceiling check and
# milestone note). The free stash is refreshed by every dispatch under this
# root — at most one round old, zero reads — but ONE round can block 900s in
# wait_tasks while children spend (the shape both dead waves had), and the
# pacing refresh only covers deadline-less tasks, so a round outliving this
# bound pays for exactly one real projection read. Never per-round (see the
# usage_accounting telemetry note and the e4a87344 contention class).
_TREE_ACCOUNTING_MAX_STALE_SEC = 120.0


def _loop_tree_accounting(
    *, refresh: bool, max_age_sec: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """The root subtree's accounted spend for the CURRENT task's tree (nullable).

    Reads the reserve-time scope telemetry for free; ``refresh=True`` may do one
    real ledger projection read when the stash is older than ``max_age_sec``.
    Callers: loop start / 600s pacing note / 15-round checkpoint (cache-breaking
    surfaces, small max_age), plus the two DECIDING surfaces (ceiling check +
    milestone note) with the wider ``_TREE_ACCOUNTING_MAX_STALE_SEC`` bound —
    free while rounds are shorter than the bound, since every dispatch refreshes
    the stash. Never an unconditional per-round read (usage_accounting notes,
    e4a87344). Only meaningful under a root cap; returns None otherwise (unknown
    is represented, never $0)."""
    try:
        from ouroboros.usage_accounting import (
            current_usage_scope,
            last_root_accounting,
            refresh_root_accounting,
        )

        scope = current_usage_scope()
        if scope is None or not scope.root_task_id or scope.root_limit_usd is None:
            return None
        if refresh:
            return refresh_root_accounting(scope.drive_root, scope.root_task_id, max_age_sec=max_age_sec)
        return last_root_accounting(scope.root_task_id)
    except Exception:
        log.debug("Tree accounting telemetry unavailable", exc_info=True)
        return None


def _soft_land_exhausted_ceiling(
    limit_ctx: "_RoundLimitContext",
    cost_ceiling: "task_pacing.CostCeiling",
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Typed soft landing (v6.91): a root cap at or below the planning margin
    leaves no working room — enter the existing graceful best-effort wrap-up
    BEFORE spending a work round; never run uncapped (the pre-typed shape
    resolved this to the same None as "unlimited"). The ledger fence stays the
    untouched backstop. Returns the forced-final tuple, or None when the
    ceiling is not in the ``exhausted_soft_land`` state."""
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
    return _loop()._forced_final_answer(
        limit_ctx,
        prompt=(
            f"[BUDGET LIMIT] {soft_land_reason} Produce your best final answer "
            "NOW from the verified work so far; clearly mark anything unverified "
            "or incomplete. An honest best-effort result is the expected outcome "
            "here, not a failure."
        ),
        fallback_text=soft_land_reason,
        reason_code="budget_exhausted",
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


def _handle_budget_exceeded(
    exc: BudgetExceeded,
    ctx: _LoopExitContext,
    *,
    limit_ctx: Optional[_RoundLimitContext] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Apply the physical-attempt dispatch rail without spending a wrap-up call."""
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
        and not bool(getattr(ctx.tools._ctx, "is_direct_chat", False))
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
    # A physical budget rail is terminal for this execution.  Finalize task
    # services before testing or creating a DeliveryCandidate so no pre-teardown
    # answer can be published against stale service/output evidence.  The loop's
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
    """Release executor, task services, and mailbox after every loop exit."""
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
    # The full DeliveryCandidate is intentionally loop-local. Only its compact
    # hash/revision projection remains in llm_trace after this cleanup. Clear it
    # after the idempotent teardown safety net so cleanup cannot erase the only
    # complete answer before service evidence is collected.
    ctx.tools._ctx._delivery_candidate = None
    ctx.tools._ctx._delivery_control_required = False
    if ctx.drive_root is None or not ctx.task_id:
        return
    try:
        from ouroboros.delegate_custody import custody_root, release_task_runs

        # A delegated run is a resource this task HOLDS, like a service or an executor:
        # a terminalized parent that leaves one running has a mutating process nothing
        # is watching. The durable reconciler still covers a worker that dies before
        # reaching here; this is the ordinary path.
        release_task_runs(custody_root(ctx.tools._ctx), ctx.task_id)
    except Exception:
        log.debug("Failed to release delegated runs for task %s", ctx.task_id, exc_info=True)
    try:
        from ouroboros.owner_mailbox import cleanup_task_mailbox

        cleanup_task_mailbox(ctx.drive_root, ctx.task_id)
    except Exception:
        log.debug("Failed to cleanup task mailbox", exc_info=True)


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


def _prepare_post_tool_budget_context(
    tools: ToolRegistry,
    limit_ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    active_model: str,
    active_use_local: bool,
    active_effort: str,
) -> None:
    """Refresh candidate evidence and the actual route before budget wrap-up."""

    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if isinstance(candidate, _loop().DeliveryCandidate):
        skill_action_pending = (
            candidate.finalization_control == "skill_action_or_revision_required"
        )
        evidence_revision, evidence_fingerprint = _loop()._delivery_evidence_state(
            tools, limit_ctx, llm_trace,
        )
        if (
            candidate.evidence_revision != evidence_revision
            or candidate.evidence_fingerprint != evidence_fingerprint
        ):
            _loop()._arm_delivery_control(
                tools,
                limit_ctx,
                llm_trace,
                control="effect_revision_required",
            )
        elif skill_action_pending:
            _loop()._arm_delivery_control(
                tools,
                limit_ctx,
                llm_trace,
                control="skill_revision_required",
            )
    # Cross-model fallback can adopt a different route during this round.
    limit_ctx.active_model = active_model
    limit_ctx.active_use_local = active_use_local
    limit_ctx.active_effort = active_effort
