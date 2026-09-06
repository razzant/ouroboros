"""Resolution of a task's terminal event into durable truth and delivery.

Owns the authoritative terminal cost projection, the lifecycle-fault lane for a
terminal that arrived without a usable result, the durable-write fault lane,
and the single dispatch that delivers the final answer and releases the slot.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict

from ouroboros.cost_projection import carry_cost_meta, with_cost_aliases
from ouroboros.outcomes import infra_failed_axes, normalize_outcome_axes
from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
from ouroboros.task_finalization import send_provider_death_notice
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_REJECTED_DUPLICATE,
    load_task_result,
    write_task_result,
)
from ouroboros.utils import append_jsonl, truncate_for_log, utc_now_iso
from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID
from supervisor.message_bus import notification_chat_route, row_chat_identity


def _events():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import events

    return events


log = logging.getLogger(__name__)


def _authoritative_terminal_cost(
    task_id: str, task: Dict[str, Any], result: Dict[str, Any], evt: Dict[str, Any], drive_root: pathlib.Path,
) -> Dict[str, Any]:
    """Project one terminal task/root from the physical-attempt authority."""
    from ouroboros.cost_projection import honest_accounted_amount
    from supervisor.state import reconstruct_task_cost

    authority_root = pathlib.Path(task.get("budget_drive_root") or drive_root)
    projection = reconstruct_task_cost(task_id, fields=True, drive_root=authority_root)
    from ouroboros.task_results import resolve_task_lineage

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    root_id = str(result.get("root_task_id") or task.get("root_task_id") or evt.get("root_task_id") or "")
    parent_id = str(result.get("parent_task_id") or task.get("parent_task_id") or evt.get("parent_task_id") or "")
    lineage = resolve_task_lineage(
        task_id,
        metadata=metadata,
        root_task_id=root_id,
        parent_task_id=parent_id,
        delegation_role=(
            result.get("delegation_role")
            or task.get("delegation_role")
            or evt.get("delegation_role")
        ),
        original_task_id=(
            result.get("original_task_id")
            or task.get("original_task_id")
            or evt.get("original_task_id")
        ),
        timeout_retry_from=(
            result.get("timeout_retry_from")
            or task.get("timeout_retry_from")
            or evt.get("timeout_retry_from")
        ),
    )
    is_root = bool(lineage["is_root_task"])
    if is_root and projection.get("cost_accounting_status") == "available":
        try:
            from ouroboros.usage_accounting import usage_breakdown

            subtree = usage_breakdown(
                authority_root,
                root_task_id=str(lineage["root_task_id"] or task_id),
            )
            subtree_final = bool(subtree.get("cost_final"))
            subtree_amount = honest_accounted_amount(subtree)
            projection.update({
                "accounted_upper_bound_usd_with_children": (
                    round(subtree_amount, 6) if subtree_amount is not None else None
                ),
                "cost_with_children_partial": not subtree_final,
                "cost_final": bool(projection.get("cost_final") and subtree_final),
                # THIRD site of the same class: `non_final_rows` is `cost_final`'s
                # DISCLOSED CAUSE and rides with it by contract (task_results.py), but
                # the root branch narrowed `cost_final` against the SUBTREE and then
                # left the row count describing this task alone — so a root turned
                # non-final purely by a child's open row reported a cause of 0, a flag
                # no reader could reconstruct.
                "non_final_rows": int(subtree.get("non_final_rows") or 0),
            })
        except Exception:
            log.error("Root subtree cost authority unavailable for %s", task_id, exc_info=True)
            projection.update({
                "cost_accounting_status": "unavailable", "cost_final": False,
                "cost_accounting_error": "ledger_unavailable",
                "accounted_upper_bound_usd": None,
                "accounted_upper_bound_usd_with_children": None,
                "cost_with_children_partial": True,
            })
    elif not is_root:
        from ouroboros.cost_projection import resolve_cost_pair

        present, rollup = resolve_cost_pair(
            result, "accounted_upper_bound_usd_with_children", "cost_usd_with_children")
        if not present:
            _, rollup = resolve_cost_pair(
                evt, "accounted_upper_bound_usd_with_children", "cost_usd_with_children")
        projection["accounted_upper_bound_usd_with_children"] = rollup
        projection["cost_with_children_partial"] = bool(
            result.get("cost_with_children_partial", evt.get("cost_with_children_partial", True))
        )
    checkpoint = result.get("root_phase_checkpoint")
    post_status = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
    if is_root and post_task_synthesis_is_open(post_status):
        projection["cost_final"] = False
        projection["cost_with_children_partial"] = True
    # SSOT cost naming (C2/ABI-3): every branch above writes the honest names
    # directly (Ф3.1 fix-round: producers no longer touch the retired
    # spellings), and this outer seam stays as the idempotent invariant guard
    # — it re-normalizes amounts and would strip any retired key a future
    # mutation leaked. This is deliberately the LAST statement.
    return with_cost_aliases(projection)


def _task_done_review_projection(
    result: Dict[str, Any], event: Dict[str, Any],
) -> Dict[str, Any]:
    """Select the compact persisted reviewer view for one terminal event."""
    value = result.get("review_projection")
    if not isinstance(value, dict):
        value = event.get("review_projection")
    return value if isinstance(value, dict) and value.get("panels") else {}


# Single-shot registry for the provider-death owner notification. The old gate
# (`and task`, a live RUNNING row) also swallowed every reaper-delivered terminal:
# the reaper loop pops RUNNING before its task_done dispatches (regression tests:
# test_supervisor_reaper_notification.py). Process-local: after a restart the
# worst case is one repeated notification, never a lost one.
_PROVIDER_DEATH_NOTIFIED: set[str] = set()


def _maybe_notify_provider_death(
    ctx: Any,
    task_id: Any,
    task: Dict[str, Any],
    final_task_result: Dict[str, Any],
    task_done_event: Dict[str, Any],
) -> None:
    """Provider-death honesty (P1): tell the owner a root task terminalized by a
    provider outage was NOT completed — the historical shape was 95 minutes of
    silence behind a result claiming "completed". Runs AFTER the task-done
    bookkeeping (cleanup never depends on chat delivery) and registers the id in
    the single-shot registry only after a SUCCESSFUL send, so a raising send is
    retried by a later dispatch instead of being lost. Never raises."""
    if not (
        task_id
        and str(task_id) not in _events()._PROVIDER_DEATH_NOTIFIED
        and str(
            task.get("delegation_role") or final_task_result.get("delegation_role") or ""
        ) != "subagent"
        and str(task_done_event.get("reason_code") or "") == "provider_unavailable"
        and str(task_done_event.get("status") or "") == STATUS_FAILED
    ):
        return
    # Membership, not truthiness: an outage notice for a hidden-partition root
    # used to be dropped here, so the incident left no owner-visible trace at all.
    notify_chat = notification_chat_route(task_done_event.get("chat_id"))
    if notify_chat is None:
        return
    try:
        # Promise only what works: the resume endpoint serves budget-paused
        # PENDING tasks (task_lifecycle.resume_budget_paused_task), never a
        # failed terminal — "resume" here was a false owner promise.
        if not send_provider_death_notice(
            ctx, notify_chat, task_id, final_task_result,
        ):
            return
    except Exception:
        log.warning(
            "Provider-death owner notification failed for %s", task_id, exc_info=True,
        )
        return
    _events()._PROVIDER_DEATH_NOTIFIED.add(str(task_id))


def _finish_task_done_dispatch(
    evt: Dict[str, Any],
    ctx: Any,
    *,
    task_id: Any,
    worker_id: Any,
    task: Dict[str, Any],
    final_task_result: Dict[str, Any],
    task_done_event: Dict[str, Any],
) -> None:
    """Notify lineage, release queue state, and preserve terminal compatibility."""

    from ouroboros.project_dialogue import (
        append_terminal_task_projection,
        enqueue_project_completion_summary,
    )

    # This seam is shared by the normal task_done path AND the lifecycle-fault
    # resolver, so open owner-quiz/hurry projections settle on EVERY dispatched
    # terminal transition (ingress lazy-heal covers producers that bypass it,
    # e.g. orphaned-RUNNING reconciliation).
    if not bool(evt.get("_ephemeral")):
        try:
            from supervisor.queue_transitions import reconcile_terminal_task_projections

            reconcile_terminal_task_projections(ctx.DRIVE_ROOT, str(task_id))
        except Exception:
            log.debug("terminal projection reconcile failed for %s", task_id, exc_info=True)

    append_terminal_task_projection(
        ctx.DRIVE_ROOT, str(task_id or ""), task, final_task_result, task_done_event,
    )

    enqueue_project_completion_summary(
        ctx.DRIVE_ROOT, evt, str(task_id or ""), task, final_task_result, task_done_event,
    )

    if task_id and str(task.get("delegation_role") or "") == "subagent":
        effective_result = (
            final_task_result
            or load_task_result(ctx.DRIVE_ROOT, str(task_id or ""))
            or {}
        )
        from supervisor.subagent_task_truth import enrich_task_done_event

        _envelope = enrich_task_done_event(task_done_event, effective_result)
        # Membership, not truthiness (C4): chat 0 real, negative A2A.
        chat_id = notification_chat_route(
            _events()._bound_project_chat_id(
                ctx, task_id, task.get("parent_task_id"), task.get("root_task_id")
            ) or None,
            task.get("chat_id"),
        )
        if chat_id is not None:
            status = str(
                effective_result.get("status")
                or evt.get("status")
                or STATUS_COMPLETED
            )
            status_display = {
                STATUS_COMPLETED: ("✅", "completed", "completed"),
                STATUS_FAILED: ("❌", "failed", "failed"),
                STATUS_REJECTED_DUPLICATE: ("⚠️", "rejected", "rejected"),
                STATUS_CANCELLED: ("⏹️", STATUS_CANCELLED, STATUS_CANCELLED),
                STATUS_INTERRUPTED: ("⏹️", STATUS_INTERRUPTED, STATUS_INTERRUPTED),
            }.get(status, ("ℹ️", status or "done", status or "finished"))
            icon, subagent_event, verb = status_display
            result_text = str(effective_result.get("result") or "")
            trace_text = str(effective_result.get("trace_summary") or "")
            constraint = effective_result.get("task_constraint")
            constraint = constraint if isinstance(constraint, dict) else {}
            # The seed keeps the frame's long-standing shape (the honest name
            # always present, null when unknown) even for a terminal event that
            # carried no cost field at all.
            _cost_meta = carry_cost_meta(
                {"accounted_upper_bound_usd": None, **task_done_event})
            progress_meta = {
                "subagent_event": subagent_event,
                "subagent_task_id": str(task_id or ""),
                "root_task_id": str(task.get("root_task_id") or ""),
                "parent_task_id": str(task.get("parent_task_id") or ""),
                "delegation_role": "subagent",
                "subagent_role": str(task.get("role") or ""),
                "write_surface": str(constraint.get("surface") or ""),
                "status": status,
                # C2/C12 (ABI-3): the honest cost names plus EVERY openness/
                # integrity marker accounting recorded. The VALUES come from the
                # cost SSOT (`_cost_meta` above) so a marker added there arrives
                # here too; the KEYS stay literal because a ChatOutbound frame's
                # key set must be statically checkable (tests/test_contracts.py)
                # — and `tests/test_cost_projection.py` fails if this literal
                # ever stops covering the SSOT.
                "accounted_upper_bound_usd": _cost_meta.get("accounted_upper_bound_usd"),
                "cost_with_children_partial": _cost_meta.get("cost_with_children_partial"),
                "unknown_unmetered": _cost_meta.get("unknown_unmetered"),
                "non_final_rows": _cost_meta.get("non_final_rows"),
                "reserved_usd": _cost_meta.get("reserved_usd"),
                "unresolved_upper_bound_usd": _cost_meta.get("unresolved_upper_bound_usd"),
                "ledger_integrity_degraded": _cost_meta.get("ledger_integrity_degraded"),
                "cost_accounting_error": _cost_meta.get("cost_accounting_error"),
                "cost_accounting_status": str(
                    task_done_event.get("cost_accounting_status") or "unavailable"
                ),
                "cost_final": bool(task_done_event.get("cost_final", False)),
                "result": truncate_for_log(result_text, 4000),
                "result_truncated": len(result_text) > 4000,
                "trace_summary": truncate_for_log(trace_text, 4000),
                "trace_summary_truncated": len(trace_text) > 4000,
                "error": truncate_for_log(str(effective_result.get("error") or ""), 1000),
                "artifact_status": str(effective_result.get("artifact_status") or ""),
                # The terminal frame carries the route so the finished card's chip can be
                # rebuilt on replay, and the completion-seam EVIDENCE (below) so the chip
                # upgrades from the neutral "dispatched" decision to what actually ran.
                "executor_route": str(effective_result.get("executor_route") or ""),
            }
            if isinstance(_envelope.get("execution_evidence"), dict):
                progress_meta["execution_evidence"] = _envelope["execution_evidence"]
            if _envelope.get("actual_substrate"):
                progress_meta["actual_substrate"] = str(_envelope["actual_substrate"])
            if isinstance(task_done_event.get("outcome_axes"), dict):
                progress_meta["outcome_axes"] = task_done_event["outcome_axes"]
            if task_done_event.get("reason_code"):
                progress_meta["reason_code"] = str(task_done_event["reason_code"])
            if "review_projection" in task_done_event:
                progress_meta["review_projection"] = task_done_event["review_projection"]
            ctx.send_with_budget(
                chat_id,
                f"{icon} Subagent {task_id} {verb} ({task.get('role') or 'researcher'}).",
                is_progress=True,
                task_id=str(task_id or ""),
                progress_meta=progress_meta,
            )

    from supervisor.queue import _queue_lock, clear_acceptance_fence_for_root

    with _queue_lock:
        if task_id:
            ctx.RUNNING.pop(str(task_id), None)
            # A child's settled result is the parent's cue to START integrating,
            # so settlement counts as the PARENT's own progress. Without this
            # stamp a coordinator blocked in wait_tasks was idle-killed exactly
            # when its last child delivered (the completed child instantly left
            # RUNNING, so _subtree_progressing went dark and only the grace
            # window remained). Own progress also lets the existing spare
            # machinery (resolve_grace_episode_for_spared_task) withdraw an
            # outstanding finalization-grace episode on the next enforce tick.
            # A one-shot event per child terminal — unlike subtree narration,
            # it cannot re-arm/flicker episodes. `task` is {} for reaper-delivered
            # terminals (RUNNING popped before dispatch), so fall back to the
            # durable result for the parent id — same shape the notification
            # gate handles.
            parent_meta = ctx.RUNNING.get(str(
                task.get("parent_task_id")
                or final_task_result.get("parent_task_id") or ""
            ))
            if isinstance(parent_meta, dict):
                parent_meta["last_progress_at"] = _events().time.time()
        if worker_id in ctx.WORKERS and ctx.WORKERS[worker_id].busy_task_id == task_id:
            # A `reaping` slot is OWNED — by the reaper or by an in-flight
            # cancellation custody. Its owner confirms process death and then
            # respawns or releases; freeing the slot from here would hand a
            # mid-kill process back to assignment.
            if not getattr(ctx.WORKERS[worker_id], "reaping", False):
                ctx.WORKERS[worker_id].busy_task_id = None
    if task_id:
        try:
            clear_acceptance_fence_for_root(str(task_id))
        except Exception:
            log.warning(
                "Failed to clear terminal task acceptance fence for %s",
                task_id,
                exc_info=True,
            )
        try:
            from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

            # Pending-cancel and reaper task_done arrive AFTER the row left
            # PENDING/RUNNING, so `task` is {} here: the tree identity falls
            # back to the event stamp, then the durable result.
            clear_budget_root_fence_for_settled_tree({
                "id": str(task_id or ""),
                "root_task_id": str(
                    (task if isinstance(task, dict) else {}).get("root_task_id")
                    or (task_done_event or {}).get("root_task_id")
                    or (final_task_result or {}).get("root_task_id")
                    or ""
                ),
            })
        except Exception:
            log.warning("Failed to release budget root fence for %s", task_id, exc_info=True)
    ctx.persist_queue_snapshot(reason="task_done")
    try:
        ctx.bridge.push_log(task_done_event)
    except Exception:
        log.warning(
            "Failed to forward task_done to live logs (card may not finalize)",
            exc_info=True,
        )

    if bool(evt.get("_ephemeral")):
        # An ephemeral direct-chat decision turn shows its failure inline —
        # no duplicate provider-outage owner ping.
        return
    _events()._maybe_notify_provider_death(ctx, task_id, task, final_task_result, task_done_event)
    try:
        results_dir = pathlib.Path(ctx.DRIVE_ROOT) / "task_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        result_file = results_dir / f"{task_id}.json"
        if not result_file.exists():
            write_task_result(
                ctx.DRIVE_ROOT,
                str(task_id or ""),
                STATUS_FAILED,
                reason_code="missing_task_result",
                outcome_axes=infra_failed_axes(
                    "missing_task_result", review_trigger="supervisor_fallback"
                ),
                result="",
                **({
                    key: task_done_event[key]
                    for key in ("total_rounds", "prompt_tokens", "completion_tokens")
                    if key in task_done_event
                }),
                # C12: the accounting fields come from the cost SSOT, so a marker
                # added there (reserved/unresolved/ledger integrity) reaches this
                # fallback result too instead of being dropped by a stale list.
                **carry_cost_meta(task_done_event),
                ts=evt.get("ts", ""),
            )
    except Exception as exc:
        log.warning("Failed to store task result in events: %s", exc)
    if task_id:
        try:
            from supervisor.terminal_delivery import cleanup_settled_owner_mailbox

            cleanup_settled_owner_mailbox(ctx.DRIVE_ROOT, str(task_id), task)
        except Exception:
            log.warning("Failed to cleanup terminal owner mailbox for %s", task_id, exc_info=True)


def _resolve_lifecycle_fault(
    evt: Dict[str, Any], ctx: Any, evt_status: str, *, detail: str = "",
) -> None:
    """Give a refused ``task_done`` an OWNER, or the worker slot wedges.

    Refusing the publication is right — the incident published a cancel latch as
    a terminal — but a refusal alone leaves the task in RUNNING with its worker
    still marked busy and nothing scheduled to finish it. Two cases:

    - A durable cancel intent (or a legacy ``cancel_requested`` latch) exists:
      cancellation custody and the watchdog already own this task, so the row
      stays exactly where it is and they settle it honestly.
    - Nothing owns it: the event is a genuine lifecycle bug, so the task is
      TERMINALIZED as ``failed`` with a typed reason and the slot is released.
      A wedged worker costs strictly more than an honest infra failure.

    ``detail`` overrides the default event-status wording — the durable-result
    fault (AR2-3) refuses an event whose OWN status looks settled.
    """
    task_id = str(evt.get("task_id") or "").strip()
    if not task_id:
        return
    try:
        from ouroboros.cancel_intents import cancel_pending

        if cancel_pending(ctx.DRIVE_ROOT, task_id):
            log.info(
                "task_done lifecycle fault for %s left to cancellation custody (cancel pending)",
                task_id,
            )
            return
    except Exception:
        log.debug("lifecycle-fault cancel-pending check failed for %s", task_id, exc_info=True)
    detail = detail or (
        f"Worker published a non-settled task_done ({evt_status!r}) and no cancellation "
        "owns this task; the supervisor terminalized it so the slot is not wedged."
    )
    # Capture the RUNNING row BEFORE the dispatch below pops it: it carries the
    # routing facts (chat/lineage/type) the terminal frame needs.
    task_row: Dict[str, Any] = {}
    try:
        running = getattr(ctx, "RUNNING", None)
        meta = running.get(task_id) if isinstance(running, dict) else None
        if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
            task_row = dict(meta["task"])
    except Exception:
        task_row = {}
    # GR4-3: the synthetic terminal fires the SAME assisted-update hooks the
    # normal task_done path reaches — an orphaned managed-update transaction or
    # a held assisted writer gate would otherwise survive a lifecycle-fault
    # terminal until an unrelated task released them.
    try:
        event_metadata = evt.get("metadata")
        task_metadata = (
            task_row.get("metadata")
            if isinstance(task_row.get("metadata"), dict)
            else event_metadata if isinstance(event_metadata, dict) else None
        )
        from supervisor.update_merge import (
            abort_orphaned_assisted_tx,
            release_assisted_writer_gate_after_task,
        )

        abort_orphaned_assisted_tx(str(task_id), task_metadata)
        release_assisted_writer_gate_after_task(task_metadata)
    except Exception:
        log.debug("assisted-merge orphan watchdog failed (lifecycle fault)", exc_info=True)
    stored: Dict[str, Any] = {}
    try:
        from ouroboros.task_results import STATUS_FAILED, write_task_result

        write_task_result(
            ctx.DRIVE_ROOT, task_id, STATUS_FAILED,
            reason_code="task_done_lifecycle_fault",
            result=detail,
            outcome_axes=infra_failed_axes(
                "task_done_lifecycle_fault", review_trigger="supervisor_terminal",
            ),
        )
        stored = load_task_result(ctx.DRIVE_ROOT, task_id) or {}
    except Exception:
        # GR3-6: durable persistence FAILED — retain lifecycle ownership. The
        # row stays in RUNNING and the slot stays busy: releasing them over a
        # non-settled durable truth would recreate the exact wedge this seam
        # closes (task invisible, nothing scheduled to finish it). The next
        # fault/watchdog pass retries.
        log.error(
            "Failed to terminalize lifecycle-fault task %s; retaining lifecycle "
            "ownership (no slot release)", task_id, exc_info=True,
        )
        return
    # GR3-6: the synthetic terminal goes through the NORMAL dispatch seam —
    # terminal UI frame, acceptance-fence clearing, campaign/project hooks,
    # RUNNING/slot bookkeeping, snapshot — instead of the old private partial
    # copy (RUNNING pop + slot clear only), which resolved nothing owner-visible.
    status = str(stored.get("status") or "failed")
    task_type = str(evt.get("task_type") or task_row.get("type") or "")
    task_done_event: Dict[str, Any] = {
        "ts": utc_now_iso(),
        "type": "task_done",
        "task_id": task_id,
        "task_type": task_type,
        "chat_id": row_chat_identity(
            _events()._bound_project_chat_id(
                ctx, task_id, task_row.get("parent_task_id"), task_row.get("root_task_id")
            ) or None,
            evt.get("chat_id"), task_row.get("chat_id"), stored.get("chat_id"),
            default=HIDDEN_CHAT_ID,
        ),
        "status": status,
        "reason_code": str(stored.get("reason_code") or "task_done_lifecycle_fault"),
        "outcome_axes": normalize_outcome_axes(stored),
    }
    try:
        task_done_event.update(_events()._authoritative_terminal_cost(
            task_id, task_row, stored, evt, pathlib.Path(ctx.DRIVE_ROOT),
        ))
    except Exception:
        log.debug("lifecycle-fault cost projection failed for %s", task_id, exc_info=True)
    try:
        append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", task_done_event)
    except Exception:
        log.warning("Failed to log lifecycle-fault task_done to events.jsonl", exc_info=True)
    if task_type == "evolution":
        _events()._handle_evolution_task_done(
            ctx, evt=evt, task_id=task_id, task=task_row,
            task_done_event=task_done_event,
            outcome_axes=task_done_event.get("outcome_axes") or {},
            cost=task_done_event.get("accounted_upper_bound_usd"),
            rounds=task_done_event.get("total_rounds"),
        )
    # GR4-3: the cooperative-checkpoint hooks fire for the synthetic terminal
    # exactly as the normal path fires them — a lifecycle-fault root would
    # otherwise never checkpoint its coop tree, and a faulted last subagent
    # would never trigger the tree-quiescence checkpoint.
    try:
        if task_row and str(task_row.get("delegation_role") or "") != "subagent":
            _events()._checkpoint_coop_roots_on_root_done(ctx, task_row, task_id)
    except Exception:
        log.debug("coop root-done checkpoint failed (lifecycle fault)", exc_info=True)
    _events()._finish_task_done_dispatch(
        evt, ctx,
        task_id=task_id, worker_id=evt.get("worker_id"),
        task=task_row, final_task_result=stored, task_done_event=task_done_event,
    )
    try:
        if task_row and str(task_row.get("delegation_role") or "") == "subagent":
            _events()._maybe_checkpoint_coop_on_tree_quiescence(ctx, task_row, task_id)
    except Exception:
        log.debug("coop quiescence checkpoint failed (lifecycle fault)", exc_info=True)


def _task_done_durable_fault(evt: Dict[str, Any], ctx: Any, task_id: Any) -> bool:
    """AR2-3 / GR2-3 (§8-A1): validate ``task_done`` through the DURABLE result.

    UNCONDITIONAL for every non-ephemeral task_done: the durable post-copy-back
    result must be settled (or the formalized ``interrupted`` transient),
    regardless of what the event's own status field says. The original AR2-3
    check gated on a settled event CLAIM — and the PRIMARY producer
    (``agent_task_pipeline``) emits task_done with a blank status, so ordinary
    completions bypassed validation entirely; a blank-status event over a
    running/absent row sailed through to publication. A blank status is now
    validated exactly like a settled claim: the worker asserted "done" and the
    disk must agree. Refused + forensic row; the existing fault-resolution
    path decides slot fate. Two exemptions stand: ephemeral turns (their event
    IS their terminal outcome — no durable lifecycle) and an ``interrupted``
    event status (its owner is the snapshot restore/requeue path). Never
    raises.
    """
    try:
        if bool(evt.get("_ephemeral")) or not task_id:
            return False
        evt_status = str(evt.get("status") or "").strip().lower()
        from ouroboros.task_results import STATUS_INTERRUPTED
        from ouroboros.task_status import SETTLED_STATUSES

        if evt_status == STATUS_INTERRUPTED:
            return False  # formalized transient: the restore/requeue path owns the row
        if evt_status and evt_status not in SETTLED_STATUSES:
            return False  # non-settled claims were already refused at the gate
        try:
            durable_status = str(
                (load_task_result(ctx.DRIVE_ROOT, str(task_id)) or {}).get("status") or ""
            ).strip().lower()
        except Exception:
            # An unreadable row is not proof of a fault; fail open toward the
            # ordinary dispatch (its own missing-result fallback still runs).
            log.debug("task_done durable validation read failed for %s", task_id, exc_info=True)
            return False
        if durable_status in SETTLED_STATUSES or durable_status == STATUS_INTERRUPTED:
            return False
        log.error(
            "task_done for %s claims settled %r but the durable result is %r; "
            "refused (durable lifecycle fault)",
            task_id, evt_status or "(blank)", durable_status or "absent",
        )
        try:
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "events.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "task_done_invalid_status",
                    "task_id": str(task_id),
                    "status": evt_status,
                    "durable_status": durable_status,
                    "worker_id": evt.get("worker_id"),
                },
            )
        except Exception:
            log.debug("task_done_invalid_status record failed", exc_info=True)
        _events()._resolve_lifecycle_fault(
            evt, ctx, evt_status,
            detail=(
                f"Worker published task_done claiming settled {evt_status or '(blank)'!r} "
                f"while the durable result is {durable_status or 'absent'!r} (not settled) "
                "and no cancellation owns this task; the supervisor terminalized it so the "
                "slot is not wedged."
            ),
        )
        return True
    except Exception:
        log.debug("task_done durable validation failed open for %s", task_id, exc_info=True)
        return False


def _handle_task_done(evt: Dict[str, Any], ctx: Any) -> None:
    # Phase A1.7: ``task_done`` asserts a SETTLED outcome. A non-settled status
    # (the incident's shape: the cancel latch published as a terminal) is a
    # durable LIFECYCLE FAULT — recorded loudly, RUNNING/worker state NOT
    # released (the row stays visible for custody/watchdog to settle honestly),
    # never a crash. Two deliberate exemptions: ephemeral direct-chat decision
    # turns (no durable task-result lifecycle — their event IS their terminal
    # outcome), and ``interrupted`` — the FORMALIZED transient the update/restart
    # teardown publishes for this generation (A1.11): its owner is the snapshot
    # restore/requeue path, and the effective-status orphan reconcile terminal-
    # izes a retry-less leftover, so it can never wedge the way the latch did.
    # The durable half of the same law (AR2-3) runs after the child copy-back:
    # a SETTLED event claim over a NON-settled durable row is refused too.
    _evt_status = str(evt.get("status") or "").strip().lower()
    if _evt_status and not bool(evt.get("_ephemeral")):
        from ouroboros.task_results import STATUS_INTERRUPTED as _INTERRUPTED
        from ouroboros.task_status import SETTLED_STATUSES as _SETTLED

        if _evt_status not in _SETTLED and _evt_status != _INTERRUPTED:
            log.error(
                "task_done with non-settled status %r for %s refused (lifecycle fault)",
                _evt_status, evt.get("task_id"),
            )
            try:
                ctx.append_jsonl(
                    ctx.DRIVE_ROOT / "logs" / "events.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "task_done_invalid_status",
                        "task_id": str(evt.get("task_id") or ""),
                        "status": _evt_status,
                        "worker_id": evt.get("worker_id"),
                    },
                )
            except Exception:
                log.debug("task_done_invalid_status record failed", exc_info=True)
            _events()._resolve_lifecycle_fault(evt, ctx, _evt_status)
            return
    task_id = evt.get("task_id")
    wid = evt.get("worker_id")
    meta = ctx.RUNNING.get(str(task_id or ""), {}) if task_id else {}
    task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
    event_metadata = evt.get("metadata")
    task_metadata = (
        task.get("metadata")
        if isinstance(task.get("metadata"), dict)
        else event_metadata if isinstance(event_metadata, dict) else None
    )
    if task_id:
        try:
            from supervisor.update_merge import (
                abort_orphaned_assisted_tx,
                release_assisted_writer_gate_after_task,
            )

            abort_orphaned_assisted_tx(str(task_id), task_metadata)
            release_assisted_writer_gate_after_task(task_metadata)
        except Exception:
            log.debug("assisted-merge orphan watchdog failed", exc_info=True)
    task_type = str(evt.get("task_type") or task.get("type") or "")

    final_task_result: Dict[str, Any] = {}
    if task_id:
        try:
            from ouroboros.headless import (
                copy_child_task_result,
                finalize_task_artifacts,
                task_is_readonly_subagent,
            )

            if task:
                copy_child_task_result(ctx.DRIVE_ROOT, task)
            # AR2-3 (§8-A1): task_done is validated through the DURABLE result,
            # not the event's own status claim. The read sits AFTER the child
            # copy-back (split-drive tasks settle on the child drive first) and
            # BEFORE artifact finalization, which would default-stamp a
            # fabricated ``completed`` row for a workspace task that never
            # wrote one — exactly the shape this refusal must catch.
            if _events()._task_done_durable_fault(evt, ctx, task_id):
                return
            if task:
                if not task_is_readonly_subagent(task):
                    finalize_task_artifacts(ctx.DRIVE_ROOT, task)
                if str(task.get("delegation_role") or "") != "subagent":
                    _events()._checkpoint_coop_roots_on_root_done(ctx, task, str(task_id or ""))
        except Exception as exc:
            try:
                from ouroboros.headless import ARTIFACT_STATUS_FAILED
                from ouroboros.outcomes import artifact_bundle_from_result

                existing = load_task_result(ctx.DRIVE_ROOT, str(task_id)) or {}
                # GR2-3b: annotate ONLY a row that exists. The old fallback
                # defaulted a MISSING row's status to "completed" — a copy-back
                # exception then minted a fabricated completion that the
                # monotonic guard defended and the durable validation below
                # would read back as settled. A task with no durable result
                # stays absent here and is judged by the fault seam instead.
                if existing and str(existing.get("status") or ""):
                    fields = {
                        "artifact_status": ARTIFACT_STATUS_FAILED,
                        "artifact_error": f"{type(exc).__name__}: {exc}",
                        "artifact_finalized_at": utc_now_iso(),
                    }
                    provisional = {**existing, **fields}
                    fields["artifact_bundle"] = artifact_bundle_from_result(provisional)
                    write_task_result(
                        ctx.DRIVE_ROOT,
                        str(task_id),
                        str(existing.get("status") or ""),
                        **fields,
                    )
            except Exception:
                pass
            log.warning("Failed to finalize headless artifacts for task %s", task_id, exc_info=True)
            # GR2-3b: an exception on the copy-back path must not SKIP the
            # durable validation — the incident shape is precisely a task_done
            # whose durable truth never landed. (When the exception came from
            # artifact finalization AFTER a passed validation, this re-check is
            # an idempotent read that passes again.)
            if _events()._task_done_durable_fault(evt, ctx, task_id):
                return
        try:
            final_task_result = load_task_result(ctx.DRIVE_ROOT, str(task_id)) or {}
        except Exception:
            final_task_result = {}

    outcome_axes = normalize_outcome_axes({**evt, **(final_task_result if isinstance(final_task_result, dict) else {})})
    reason_code = final_task_result.get("reason_code") or evt.get("reason_code")
    artifact_status = final_task_result.get("artifact_status") or evt.get("artifact_status")
    terminal_cost = _events()._authoritative_terminal_cost(
        str(task_id or ""), task,
        final_task_result if isinstance(final_task_result, dict) else {}, evt,
        pathlib.Path(ctx.DRIVE_ROOT),
    )
    eff_cost = terminal_cost.get("accounted_upper_bound_usd")
    eff_rounds = terminal_cost.get("total_rounds")
    task_done_event = {
        "ts": evt.get("ts", utc_now_iso()),
        "type": "task_done",
        "task_id": task_id,
        "task_type": task_type,
        "chat_id": row_chat_identity(
            _events()._bound_project_chat_id(
                ctx, task_id,
                (final_task_result.get("parent_task_id") if isinstance(final_task_result, dict) else "") or evt.get("parent_task_id"),
                (final_task_result.get("root_task_id") if isinstance(final_task_result, dict) else "") or evt.get("root_task_id"),
            ) or None,
            evt.get("chat_id"),
            (final_task_result.get("chat_id") if isinstance(final_task_result, dict) else None),
            default=HIDDEN_CHAT_ID,
        ),
        "status": str(final_task_result.get("status") or evt.get("status") or ""),
        "outcome_axes": outcome_axes,
        "reason_code": reason_code,
        "artifact_status": artifact_status,
        **terminal_cost,
    }
    if bool(evt.get("ephemeral_decision") or evt.get("_ephemeral")):
        task_done_event["ephemeral_decision"] = True
    if str(evt.get("typed_routing_action") or "").strip():
        task_done_event["typed_routing_action"] = str(evt.get("typed_routing_action") or "").strip()
    artifact_bundle = final_task_result.get("artifact_bundle") if isinstance(final_task_result, dict) else None
    if not isinstance(artifact_bundle, dict):
        artifact_bundle = evt.get("artifact_bundle")
    if isinstance(artifact_bundle, dict):
        task_done_event["artifact_bundle"] = artifact_bundle
    review_status = final_task_result.get("review_status") if isinstance(final_task_result, dict) else None
    if not isinstance(review_status, dict):
        review_status = evt.get("review_status")
    if isinstance(review_status, dict):
        task_done_event["review_status"] = review_status
    if review_projection := _events()._task_done_review_projection(final_task_result, evt):
        task_done_event["review_projection"] = review_projection
    try:
        append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", task_done_event)
    except Exception:
        log.warning("Failed to log task_done to events.jsonl", exc_info=True)

    if task_type == "evolution":
        _events()._handle_evolution_task_done(
            ctx,
            evt=evt,
            task_id=task_id,
            task=task,
            task_done_event=task_done_event,
            outcome_axes=outcome_axes,
            cost=eff_cost,
            rounds=eff_rounds,
        )

    _events()._finish_task_done_dispatch(
        evt,
        ctx,
        task_id=task_id,
        worker_id=wid,
        task=task,
        final_task_result=final_task_result,
        task_done_event=task_done_event,
    )

    # v6.91 tree-quiescence coop checkpoint: MUST run after the dispatch
    # bookkeeping above removed this terminal child from RUNNING, or the
    # finishing child still counts live and "zero live members" is never true.
    if task_id and str(task.get("delegation_role") or "") == "subagent":
        _events()._maybe_checkpoint_coop_on_tree_quiescence(ctx, task, str(task_id))
