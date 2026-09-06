"""Handing a pending task to a free worker, and refusing the ones that must not run.

Admission order is the queue's; what this adds is the per-task gates: a cancelled
pending row is settled rather than dispatched, an evolution task without live
campaign authority is cancelled rather than started, and a repo-writing task waits
while the writer gate is closed.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any, Dict

from supervisor.queue import _queue_lock


def _pool():
    """The parent module, read at call time.

    The pool owns the repo/drive roots, its size, the worker table, the shared
    PENDING/RUNNING refs and the crash clock, and ``init`` REBINDS them.
    Reading them through the module is what keeps one binding: a from-import
    here would freeze the value this module saw at import time (the
    owner-approved D18/D33 mechanical exception).
    """
    from supervisor import workers

    return workers


log = logging.getLogger(__name__)


def _evolution_assignment_error(task: Dict[str, Any]) -> str:
    """Return the exact authority error for an evolution task about to run."""
    if str(task.get("type") or "") != "evolution":
        return ""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    tx = metadata.get("evolution_transaction")
    tx = tx if isinstance(tx, dict) else {}
    task_id = str(task.get("id") or "")
    if str(tx.get("task_id") or "") != task_id:
        return "task_mismatch"
    from supervisor.evolution_lifecycle import check_evolution_authority

    try:
        authority = check_evolution_authority(
            campaign_id=str(tx.get("campaign_id") or ""),
            transaction_id=str(tx.get("transaction_id") or ""),
            task_id=task_id,
            require_uncommitted=True,
        )
    except Exception:
        log.warning("Evolution assignment authority check failed", exc_info=True)
        return "authority_check_failed"
    return "" if authority.get("ok") else str(authority.get("reason") or "unknown")


def _cancel_unauthorized_evolution(task: Dict[str, Any], reason: str) -> bool:
    """Terminally cancel a stale restored/retried evolution task."""
    task_id = str(task.get("id") or "")
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    try:
        write_task_result(
            _pool().DRIVE_ROOT,
            task_id,
            STATUS_CANCELLED,
            reason_code="evolution_authority_missing",
            authority_reason=str(reason or "unknown"),
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            result=f"Evolution authority is no longer active ({reason or 'unknown'}).",
        )
    except Exception:
        log.debug("Failed to cancel unauthorized evolution task %s", task_id, exc_info=True)
        return False
    _pool()._emit_task_done_terminal(
        task, task_id, "cancelled", reason_code="evolution_authority_missing",
    )
    _pool().append_jsonl(
        _pool().DRIVE_ROOT / "logs" / "events.jsonl",
        {
            "ts": _pool().utc_now_iso(), "type": "evolution_assignment_rejected",
            "task_id": task_id, "reason": str(reason or "unknown"),
        },
    )
    return True


def assign_tasks() -> None:
    from supervisor import queue
    from supervisor.state import budget_remaining, EVOLUTION_BUDGET_RESERVE
    with _queue_lock:
        st = _pool().load_state()
        # Cancellation/terminal custody wins before validating rows left in the
        # queue.  Then quarantine every malformed depth before budget, lease, or
        # capacity filters can leave it waiting indefinitely.
        if not _pool()._drop_cancelled_pending():
            log.error(
                "Task assignment blocked: cancellation authority or custody "
                "state is indeterminate",
            )
            queue.persist_queue_snapshot(reason="cancellation_authority_indeterminate")
            return
        _pool()._retry_terminalization_pending_for_assignment(queue)
        invalid_ids, unresolved_invalid_ids = _pool()._quarantine_invalid_pending_depths()
        unresolved_invalid_id_set = set(unresolved_invalid_ids)

        if invalid_ids:
            queue.persist_queue_snapshot(reason="invalid_task_depth")
        if unresolved_invalid_ids:
            log.error(
                "Invalid-depth rows deferred until terminal custody is available; continuing assignment for other tasks: %s",
                ", ".join(unresolved_invalid_ids),
            )
        try:
            remaining = budget_remaining(st, strict=True)
        except Exception:
            log.error("Task assignment blocked: monetary authority unavailable")
            return
        if remaining <= 0:
            planned = []
            for task in _pool().PENDING:
                if _pool()._invalid_depth_deferred(task, unresolved_invalid_id_set):
                    continue
                if isinstance(task.get("_budget_pause"), dict):
                    continue
                task_id = str(task.get("id") or "")
                cost_fields = _pool().reconstruct_task_cost(
                    task_id, fields=True,
                    drive_root=pathlib.Path(task.get("budget_drive_root") or _pool().DRIVE_ROOT),
                )
                if cost_fields.get("cost_accounting_status") != "available":
                    log.error("Budget pause blocked: task attempt history unavailable for %s", task_id)
                    return
                retry_lineage = bool(
                    int(task.get("_attempt") or 1) > 1
                    or task.get("original_task_id") or task.get("timeout_retry_from")
                )
                replay_safe = (
                    int(cost_fields.get("total_rounds") or 0) == 0
                    and not bool(cost_fields.get("ledger_integrity_degraded"))
                    and not retry_lineage
                )
                pause = {
                    "status": "paused_before_dispatch" if replay_safe else "resource_limited",
                    "scope": "global",
                    "physical_calls": int(cost_fields.get("total_rounds") or 0),
                    "replay_safe": replay_safe,
                    "auto_resume": False,
                    "resume_policy": "manual_same_generation" if replay_safe else "cancel_or_new_run",
                    "paused_at": _pool().utc_now_iso(),
                }
                planned.append((task, pause, cost_fields))
            newly_paused, terminal_ids = [], []
            for task, pause, cost_fields in planned:
                task_id = str(task.get("id") or "")
                result_root = pathlib.Path(task.get("budget_drive_root") or _pool().DRIVE_ROOT)
                try:
                    from ouroboros.task_results import STATUS_FAILED, STATUS_SCHEDULED, write_task_result

                    if pause["replay_safe"]:
                        task["_budget_pause"] = pause
                        newly_paused.append(task_id)
                        write_task_result(
                            result_root, task_id, STATUS_SCHEDULED,
                            reason_code="budget_exhausted", resource_limit=pause,
                        )
                    else:
                        write_task_result(
                            result_root, task_id, STATUS_FAILED,
                            reason_code="budget_exhausted", resource_limit=pause,
                            result="Budget exhausted after prior dispatch; cancel or start a new run.",
                            **cost_fields,
                        )
                        _pool()._emit_task_done_terminal(
                            task, task_id, "failed", reason_code="budget_exhausted",
                            cost_fields=cost_fields,
                        )
                        terminal_ids.append(task_id)
                except Exception:
                    log.error("Failed to project budget stop for %s", task_id, exc_info=True)
            if terminal_ids:
                terminal = set(terminal_ids)
                _pool().PENDING[:] = [task for task in _pool().PENDING if str(task.get("id") or "") not in terminal]
            if newly_paused or terminal_ids:
                _pool().append_jsonl(
                    _pool().DRIVE_ROOT / "logs" / "events.jsonl",
                    {
                        "ts": _pool().utc_now_iso(),
                        "type": "budget_tasks_paused",
                        "scope": "global",
                        "task_ids": newly_paused,
                        "resource_limited_task_ids": terminal_ids,
                        "auto_resume": False,
                    },
                )
                if st.get("owner_chat_id"):
                    _pool().send_with_budget(
                        int(st["owner_chat_id"]),
                        "🚫 Model budget reached. Queued tasks are paused before dispatch; "
                        "raising the limit does not resume them automatically.",
                    )
                queue.persist_queue_snapshot(reason="budget_paused_before_dispatch")
            return

        # Evolution is hard-blocked in light runtime mode at the assignment
        # chokepoint too: a task restored from a snapshot or created before the
        # mode switch must never actually run. Cancel them terminally.
        from supervisor.evolution_lifecycle import evolution_block_reason
        evo_block = evolution_block_reason()
        blocked_ids = _pool()._drop_assignable_evolution_tasks(unresolved_invalid_id_set) if evo_block else []
        if blocked_ids:
            from ouroboros.task_results import STATUS_CANCELLED, write_task_result
            for tid in blocked_ids:
                try:
                    write_task_result(
                        _pool().DRIVE_ROOT, tid, STATUS_CANCELLED,
                        result="Evolution is disabled in light runtime mode.",
                    )
                except Exception:
                    log.debug("Failed to cancel light-mode evolution task %s", tid, exc_info=True)
            if st.get("owner_chat_id"):
                _pool().send_with_budget(int(st["owner_chat_id"]), evo_block)
            queue.persist_queue_snapshot(reason="evolution_blocked_light")

        from ouroboros.project_lease import candidate_is_leasable, running_project_ids
        from ouroboros.config import get_max_active_subagents_per_root


        for w in _pool().WORKERS.values():
            if w.busy_task_id is None and not getattr(w, "reaping", False) and _pool().PENDING:
                # One-writer-per-project lease: recompute per assignment so a
                # task assigned in THIS loop pass immediately occupies its lane.
                leased = running_project_ids(_pool().RUNNING.values())
                # Find first suitable task (skip over-budget evolution tasks
                # and project-leased candidates)
                chosen_idx = None
                for i, candidate in enumerate(_pool().PENDING):
                    if _pool()._invalid_depth_deferred(candidate, unresolved_invalid_id_set):
                        continue
                    if not _pool().repo_writer_task_allowed(candidate):
                        continue
                    if isinstance(candidate.get("_budget_pause"), dict):
                        continue
                    root_task_id = str(candidate.get("root_task_id") or "").strip()
                    if root_task_id in queue.BUDGET_ROOT_FENCES:
                        continue
                    if str(candidate.get("type") or "") == "evolution" and remaining < EVOLUTION_BUDGET_RESERVE:
                        continue
                    if not candidate_is_leasable(candidate, leased):
                        continue
                    if str(candidate.get("delegation_role") or "") == "subagent":
                        root_task_id = str(candidate.get("root_task_id") or "")
                        if (
                            _pool()._running_subagent_count(root_task_id) >= get_max_active_subagents_per_root()
                            and not _pool()._assignment_depth_reservation_admits(candidate)
                        ):
                            continue
                    chosen_idx = i
                    break
                if chosen_idx is None:
                    # Project-leased rows wait; over-budget evolution rows are cleaned.
                    if remaining < EVOLUTION_BUDGET_RESERVE:
                        dropped_ids = _pool()._drop_assignable_evolution_tasks(unresolved_invalid_id_set)
                        if dropped_ids:
                            queue.persist_queue_snapshot(reason="evolution_dropped_budget")
                    continue
                task = _pool().PENDING.pop(chosen_idx)
                depth_error = _pool()._normalize_pending_task_depth(task)
                if depth_error:
                    if _pool()._terminalize_invalid_pending_depth(task, depth_error):
                        queue.persist_queue_snapshot(reason="invalid_task_depth")
                        continue
                    # Keep failed terminalization in queue custody for retry.
                    _pool().PENDING.insert(chosen_idx, task)
                    log.error(
                        "Assignment blocked: invalid task depth could not be terminalized for %s",
                        task.get("id"),
                    )
                    break
                evolution_error = _pool()._evolution_assignment_error(task)
                if evolution_error:
                    if _pool()._cancel_unauthorized_evolution(task, evolution_error):
                        queue.persist_queue_snapshot(reason="evolution_authority_rejected")
                    else:
                        _pool().PENDING.insert(chosen_idx, task)
                    continue
                if str(task.get("delegation_role") or "") == "subagent" and str(task.get("drive_root") or ""):
                    try:
                        from ouroboros.task_results import STATUS_RUNNING, write_task_result
                        from ouroboros.tools.control_delegation import stamp_task_assignment_depth
                        from ouroboros.config import get_max_subagent_depth

                        # Assignment is the first host-visible execution fact. Stamp
                        # the worker payload and canonical result from one projection.
                        _depth_fields = stamp_task_assignment_depth(
                            task, max_depth=get_max_subagent_depth(),
                        )
                        write_task_result(
                            _pool().DRIVE_ROOT,
                            str(task.get("id") or ""),
                            STATUS_RUNNING,
                            parent_task_id=task.get("parent_task_id"),
                            root_task_id=task.get("root_task_id"),
                            session_id=task.get("session_id"),
                            actor_id=task.get("actor_id"),
                            delegation_role=task.get("delegation_role"),
                            project_id=task.get("project_id"),
                            role=task.get("role"),
                            description=task.get("description"),
                            objective=task.get("objective") or task.get("description"),
                            expected_output=task.get("expected_output"),
                            constraints=task.get("constraints"),
                            context=task.get("context"),
                            memory_mode=task.get("memory_mode"),
                            drive_root=task.get("drive_root"),
                            child_drive_root=task.get("child_drive_root") or task.get("drive_root"),
                            budget_drive_root=task.get("budget_drive_root"),
                            task_constraint=task.get("task_constraint"),
                            **_depth_fields,
                            # INTENT ONLY. This mirror is written at ASSIGNMENT, one
                            # step before the worker dispatches and resolves the
                            # child; naming `effective_model_lane`/`model` here wrote
                            # whatever the record happened to hold, which on a retry
                            # is the PREVIOUS attempt's resolution and on a fresh
                            # child is nothing at all.
                            model_lane=task.get("model_lane"),
                            requested_model_lane=task.get("requested_model_lane"),
                            parent_model_lane=task.get("parent_model_lane"),
                            requested_executor=task.get("requested_executor"),
                            task_group_id=task.get("task_group_id"),
                            task_group=task.get("task_group"),
                            subagent_envelope=task.get("subagent_envelope"),
                            configured_subagent=task.get("configured_subagent"),
                            parent_cognitive_route=task.get("parent_cognitive_route"),
                            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
                            result="Subagent assigned to a worker.",
                        )
                    except Exception:
                        log.debug("Failed to mirror running subagent status", exc_info=True)
                w.busy_task_id = task["id"]
                w.in_q.put(task)
                now_ts = time.time()
                _pool().RUNNING[task["id"]] = {
                    "task": dict(task), "worker_id": w.wid,
                    "started_at": now_ts, "last_heartbeat_at": now_ts,
                    "soft_sent": False, "attempt": int(task.get("_attempt") or 1),
                }
                task_type = str(task.get("type") or "")
                if task_type in ("evolution", "review"):
                    st = _pool().load_state()
                    if st.get("owner_chat_id"):
                        emoji = '🧬' if task_type == 'evolution' else '🔎'
                        _pool().send_with_budget(
                            int(st["owner_chat_id"]),
                            f"{emoji} {task_type.capitalize()} task {task['id']} started.",
                        )
                queue.persist_queue_snapshot(reason="assign_task")
