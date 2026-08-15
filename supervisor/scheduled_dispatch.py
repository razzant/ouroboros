"""One schedule-fire adapter: resolve the placement AT FIRE TIME, then enqueue.

A schedule template stores ``project_id`` only — ``schedule_contract.RESERVED_TEMPLATE_FIELDS``
rejects any template that tries to persist a placement. That is deliberate (RWS v2 §7): a
placement persisted months ago in a cron row is a fact about a world that has since moved,
so the template stores the STABLE thing (which project this work belongs to) and the
volatile thing (where that project currently lives) is resolved when the schedule fires,
through the SAME ``workspace_admission`` path ordinary ``/api/tasks`` creation uses.

Resolution and insertion cannot be one instant, so the resolved placement is bound to a
FENCE — the project's ``routing_generation`` plus the connection's trust identity — which
``supervisor/queue.py::enqueue_task`` revalidates under ``_queue_lock`` immediately before
appending to PENDING (RWSB2-05). A project deleted, rebound, or a connection retired in
that window yields a typed refusal that lands in the schedule's failure state, never a
stale placement that would be discovered later on the wrong host.

This module owns NO state and NO lifecycle: it is the adapter between a firing schedule
row and the existing queue authority.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable
from typing import Any, Dict

log = logging.getLogger(__name__)


def _system_repo_dir() -> pathlib.Path:
    """The Ouroboros system repo, read from the worker pool that owns it.

    The queue holds no repo root, and adding one would duplicate a fact that
    ``supervisor.workers`` already owns and initializes.
    """
    from supervisor import workers

    return pathlib.Path(workers.REPO_DIR)


def seal_scheduled_placement(
    task: Dict[str, Any], *, drive_root: pathlib.Path, system_repo_dir: pathlib.Path | None = None
) -> str:
    """Resolve + seal the firing task's placement. Returns "" or a typed reason code.

    The project id is validated EXACTLY at fire time (a project may have been renamed,
    deleted, or never have existed when the row was written), and a malformed id is
    refused rather than normalized — normalizing would let ``"PROD"`` quietly become the
    live ``prod`` project's work.
    """
    from ouroboros.project_facts import explicit_project_id_ok
    from ouroboros.projects_registry import get_reserved_project
    from ouroboros.workspace_admission import (
        PLACEMENT_FENCE_KEY,
        placement_fence_for,
        placement_preflight_summary,
        resolve_room_workspace,
        seal_admitted_placement,
    )

    raw_project_id = str(task.get("project_id") or "")
    project_id = raw_project_id.strip()
    if not project_id:
        return ""  # a placement-less scheduled task: unchanged behavior
    if not explicit_project_id_ok(raw_project_id):
        return "invalid_project_id"
    try:
        project = get_reserved_project(drive_root, project_id)
    except Exception:
        log.warning("schedule fire: project lookup failed for %s", project_id, exc_info=True)
        return "project_lookup_failed"
    if project is None:
        return "project_not_found"
    ref, error = resolve_room_workspace(
        drive_root=drive_root,
        system_repo_dir=system_repo_dir if system_repo_dir is not None else _system_repo_dir(),
        project_id=project_id,
    )
    if error:
        return "workspace_unusable"
    if ref is not None:
        seal_admitted_placement(
            task, ref,
            preflight_summary=placement_preflight_summary(ref, project_id=str(project_id or "")),
        )
    # The fence is bound even for a file-less project: "this work was scheduled for THIS
    # generation of THIS project" is a routing fact independent of whether the project has
    # a workspace at all.
    task[PLACEMENT_FENCE_KEY] = placement_fence_for(drive_root, project_id, ref)
    return ""


def prepare_scheduled_child_drive(task: Dict[str, Any], *, drive_root: pathlib.Path) -> None:
    """Memory-fork parity with `/api/tasks`: a project/workspace scheduled task runs on an
    ISOLATED child drive, with the canonical root kept for budget and status."""
    if not str(task.get("project_id") or "").strip() and not str(task.get("workspace_root") or "").strip():
        return
    try:
        from ouroboros.headless import prepare_task_drive

        child_drive = prepare_task_drive(
            drive_root, str(task["id"]), "forked", project_id=str(task.get("project_id") or "")
        )
    except Exception:
        log.warning("schedule fire: child drive fork failed for %s", task.get("id"), exc_info=True)
        return
    if child_drive is None:
        return
    task["drive_root"] = str(child_drive)
    task["child_drive_root"] = str(child_drive)
    task["budget_drive_root"] = str(drive_root)
    metadata = task.setdefault("metadata", {})
    metadata["child_drive_root"] = str(child_drive)
    metadata["budget_drive_root"] = str(drive_root)


def dispatch_scheduled_task(
    task: Dict[str, Any],
    *,
    drive_root: pathlib.Path,
    schedule_id: str,
    schedule_name: str,
    enqueue: Callable[[Dict[str, Any]], Dict[str, Any]],
    system_repo_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """Resolve the placement, persist the SCHEDULED record, and enqueue the fired task.

    A placement refusal terminalizes the fired task and returns the queue's own
    ``_admission_blocked`` shape, so the caller's existing
    ``task_lifecycle.record_scheduled_admission`` projects it into the schedule's failure
    state — one refusal path, whether the refusal came from admission or from a fence.
    """
    from ouroboros.task_results import STATUS_FAILED, STATUS_SCHEDULED, write_task_result

    reason = seal_scheduled_placement(
        task, drive_root=drive_root, system_repo_dir=system_repo_dir
    )
    if reason:
        try:
            write_task_result(
                drive_root,
                str(task["id"]),
                STATUS_FAILED,
                reason_code=reason,
                result=f"Scheduled task was not queued: {reason}.",
                cost_usd=0.0,
                **_schedule_result_identity(task, schedule_id, schedule_name),
            )
        except Exception:
            log.warning("schedule fire: failed to terminalize %s", task.get("id"), exc_info=True)
        return {**task, "_admission_blocked": reason, "_project_id": str(task.get("project_id") or "")}
    prepare_scheduled_child_drive(task, drive_root=drive_root)
    try:
        write_task_result(
            drive_root,
            str(task["id"]),
            STATUS_SCHEDULED,
            result="Scheduled task queued.",
            **_schedule_result_identity(task, schedule_id, schedule_name),
        )
    except Exception:
        log.debug("Failed to persist scheduled task result before enqueue", exc_info=True)
    return enqueue(task)


def _schedule_result_identity(
    task: Dict[str, Any], schedule_id: str, schedule_name: str
) -> Dict[str, Any]:
    """The identity a fired schedule's durable record must carry EITHER WAY.

    A refusal keeps the same identity fields as a success: a cron failure the owner cannot
    trace back to its project and schedule is not diagnosable.
    """
    return {
        "root_task_id": str(task["id"]),
        "actor_id": "scheduler",
        "delegation_role": "root",
        "project_id": str(task.get("project_id") or ""),
        "description": str(task.get("description") or task.get("text") or ""),
        "expected_output": str(task.get("expected_output") or ""),
        "constraints": str(task.get("constraints") or ""),
        "context": str(task.get("context") or ""),
        "allowed_resources": task.get("allowed_resources") if isinstance(task.get("allowed_resources"), dict) else {},
        "deadline_at": str(task.get("deadline_at") or ""),
        "task_contract": task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {},
        "metadata": dict(task.get("metadata") or {}),
        "schedule_id": schedule_id,
        "schedule_name": schedule_name,
    }
