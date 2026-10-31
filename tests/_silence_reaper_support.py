"""Historical timeout successors for the lineage readers that still consume them.

Before #1376 the idle rail killed a silent attempt and admitted its successor: a new
physical id for a root (an evolution or subagent attempt kept its id), the predecessor
``interrupted`` and ``superseded_by`` it, the successor bound to its room. No runtime
producer creates one now, but stored results and restored queue rows keep them, and
Pause/Resume, Stop, selection and Main still resolve them to their logical root.

``historical_timeout_retry`` writes that stored state from the dead attempt's own RUNNING
row and saved work, and admits the successor the way a restored snapshot row is admitted
(``restoring_snapshot``); ``historical_successor`` does the same from a task the caller
already holds; ``historical_unsaved_pause_hold`` keeps the pause intent the same
kill left for an owner-Pause seed with nothing saved (``unsaved_pause_intent``, the reason a
direct turn's unsaved pause still records). All are fixture data for those readers, not a
current producer; none replays restore-time proofs.
"""
import pathlib
import uuid

from supervisor import queue, workers


def _ended_attempt(clock, task_id):
    """The silence that once ended an attempt, and its RUNNING row leaving the pool."""
    idle = max(float(queue.get_task_idle_timeout_sec()), float(queue.get_per_call_timeout_ceiling_sec()) + 120.0)
    clock[0] += idle + float(queue.FINALIZATION_GRACE_SEC) + 2
    meta = workers.RUNNING.pop(task_id)
    slot = workers.WORKERS.get(meta.get("worker_id"))
    if slot is not None and slot.busy_task_id == task_id:
        slot.busy_task_id = None
    return meta


def historical_timeout_retry(clock, task_id="held"):
    meta = _ended_attempt(clock, task_id)
    task = dict(meta["task"])
    same_id = task.get("type") == "evolution" or task.get("delegation_role") == "subagent"
    return historical_successor(task, int(meta["attempt"]), task_id if same_id else uuid.uuid4().hex[:8], meta)


def historical_successor(task, attempt, retry_id, meta=None):
    """The stored state of ``task``'s ``attempt`` and of its successor ``retry_id`` (the same id or a
    new one), the successor admitted as a restored row. ``meta``: the dead attempt's RUNNING row
    when its clocks were carried; a caller holding only the task passes none."""
    from ouroboros.model_wait import fold_running_clock, stamp_execution_handoff
    from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
    from ouroboros.projects_registry import bind_task_to_project, project_binding_for_task
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import utc_now_iso
    from ouroboros.working_checkpoint import attach_recovery
    from supervisor.events_budget import fold_consumed_resume
    from supervisor.queue_transitions import clear_acceptance_fence_for_attempt

    task_id = str(task["id"])
    at = utc_now_iso()
    retried = {**task, "original_task_id": task_id, "id": retry_id, "_attempt": attempt + 1,
               "timeout_retry_from": task_id, "timeout_retry_at": at}
    attach_recovery(pathlib.Path(task.get("budget_drive_root") or queue.DRIVE_ROOT), retried,
                    source_task_id=task_id, from_attempt=attempt, cause="idle_timeout", prior_task=task)
    fold_consumed_resume(meta, queue.DRIVE_ROOT)
    fold_running_clock(retried, meta)
    stamp_execution_handoff(retried, meta)
    retried.pop("_budget_pause_resume", None)  # #1729: it continued saved work, never a spent grant
    retried.pop("_resume_selection", None)
    clear_acceptance_fence_for_attempt(task_id, task_attempt=attempt)
    if retry_id != task_id:
        write_task_result(
            queue.DRIVE_ROOT, retry_id, "scheduled", reason_code="idle_timeout_retry_scheduled",
            outcome_axes=terminal_outcome_axes(lifecycle="scheduled", execution="pending",
                                               reason_code="idle_timeout_retry_scheduled",
                                               review_trigger="supervisor_terminal"),
            supersedes_task_id=task_id, original_task_id=task_id, result="Retry scheduled after idle_timeout.",
            parent_task_id=task.get("parent_task_id"), root_task_id=task.get("root_task_id") or task_id,
            delegation_role=task.get("delegation_role"), timeout_retry_from=task_id, timeout_retry_at=at,
            description=task.get("description"), context=task.get("context"),
            workspace_root=task.get("workspace_root"), workspace_mode=task.get("workspace_mode"),
            memory_mode=task.get("memory_mode"),
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {})
    write_task_result(
        queue.DRIVE_ROOT, task_id, "interrupted", reason_code="idle_timeout_retry",
        outcome_axes=terminal_outcome_axes(lifecycle="interrupted", execution=EXECUTION_INFRA_FAILED,
                                           reason_code="idle_timeout_retry", review_trigger="supervisor_terminal"),
        superseded_by=retry_id if retry_id != task_id else "", retry_task_id=retry_id,
        result="Task killed by idle_timeout. Retrying.")
    admitted = queue.enqueue_task(retried, front=True, restoring_snapshot=True)
    assert not admitted.get("_admission_blocked"), admitted
    basis = task.get("_project_admission")
    if retry_id != task_id and not (isinstance(basis, dict) and basis.get("project") is None):
        predecessor = project_binding_for_task(queue.DRIVE_ROOT, task_id) or {}
        project_id = str((basis or {}).get("project_id") or "") if basis is not None else str(
            predecessor.get("project_id") or "")
        if project_id:  # the successor joined its predecessor's room under the same origin
            from supervisor.worker_promotion import _origin_from_mapping

            origin = _origin_from_mapping(task, absent="mid_task_no_origin")
            if isinstance(predecessor.get("source_ref"), dict):
                origin = {"ref": dict(predecessor["source_ref"])}
                if isinstance(predecessor.get("source_text"), str):
                    origin["text"] = predecessor["source_text"]
            elif predecessor.get("origin_absent"):
                origin = {"absent": str(predecessor["origin_absent"])}
            bind_task_to_project(queue.DRIVE_ROOT, retry_id, project_id, origin=origin, admission_basis=basis)
    return admitted


def historical_unsaved_pause_hold(clock, task_id="held"):
    """The SAME id as its next attempt, held as pause intent over its dead attempt's unsaved owner seed."""
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.model_wait import fold_running_clock, stamp_execution_handoff
    from ouroboros.task_results import resolve_task_lineage
    from ouroboros.working_checkpoint import attach_recovery
    from supervisor.events_budget import HOLD_UNSAVED_PAUSE, hold_budget_row

    meta = _ended_attempt(clock, task_id)
    task, attempt = dict(meta["task"]), int(meta["attempt"])
    root = pathlib.Path(task.get("budget_drive_root") or queue.DRIVE_ROOT)
    seed = budget_pause_row(root, task_id)
    assert (seed.get("reason"), seed.get("source_ref"), int(seed.get("task_attempt") or 0)) == ("owner", None, attempt)
    held = {key: value for key, value in task.items() if key not in {"_budget_pause", "_budget_pause_resume"}}
    held["_attempt"] = attempt + 1
    attach_recovery(root, held, source_task_id=task_id, from_attempt=attempt, cause="idle_timeout", prior_task=task)
    fold_running_clock(held, meta)
    stamp_execution_handoff(held, meta)
    actor = resolve_task_lineage(task_id, **{key: task.get(key) for key in (
        "metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id",
        "timeout_retry_from")})["is_root_task"]
    with queue._queue_lock:
        hold_budget_row(held, reason=HOLD_UNSAVED_PAUSE, result_root=root,
                        detail="stopped while pausing, before the pause was saved; " + (
                            "Resume continues only work already saved" if held.get("_working_recovery")
                            else "the owner's Resume starts it again from its request" if actor
                            else "nothing was saved to continue: cancel it or start a new run"),
                        extra={"root_task_id": str(task.get("root_task_id") or task_id),
                               "pause_id": str(seed.get("pause_id") or ""), "task_attempt": attempt,
                               "pause_generation": int(seed.get("pause_generation") or 0)})
        workers.PENDING.append(held)
        queue.sort_pending()
    return held
