"""One-writer-per-project lease helpers (multi-project, v6.32.0).

Pure functions consumed by ``supervisor/workers.py::assign_tasks`` under the
queue lock: a PENDING task whose ``project_id`` is already RUNNING is skipped
this assignment pass (projects serialize internally; parallelism happens
BETWEEN projects and via subagent swarms WITHIN a task).

``project_id == ""`` means "no lane": ordinary unscoped tasks never serialize
against each other. Subagents carry their parent's stored ``project_id`` but
hold no lease of their own — the parent task IS the project's writer and its
swarm must not deadlock against itself, so only top-level (non-subagent)
tasks count as lane occupants.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Set


def _as_task(item: Any) -> Any:
    """Unwrap the supervisor RUNNING meta shape ({"task": {...}, ...}) to the
    task dict; pass a bare task dict through unchanged."""
    if isinstance(item, dict) and isinstance(item.get("task"), dict):
        return item["task"]
    return item


def _task_project_id(task: Any) -> str:
    task = _as_task(task)
    if not isinstance(task, dict):
        return ""
    return str(task.get("project_id") or "").strip()


def _is_lane_occupant(task: Any) -> bool:
    """Top-level project-scoped tasks occupy the lane; subagents do not."""
    task = _as_task(task)
    if not isinstance(task, dict):
        return False
    if str(task.get("delegation_role") or "") == "subagent":
        return False
    return bool(_task_project_id(task))


def running_project_ids(running: Iterable[Any]) -> Set[str]:
    """Project ids currently holding a writer lease.

    ``running`` is the supervisor's RUNNING mapping values (or any iterable of
    task dicts); read under the queue lock by the caller.
    """
    out: Set[str] = set()
    for task in running or ():
        if _is_lane_occupant(task):
            out.add(_task_project_id(task))
    return out


def cold_project_holders(pending: Iterable[Any]) -> Dict[str, Set[str]]:
    """Accepted cold external writers keep their original top-level project lane."""
    from ouroboros.sleep_custody import carrier_receipt

    holders: Dict[str, Set[str]] = {}
    for task in pending:
        if _is_lane_occupant(task) and carrier_receipt(task).get("retains_project_lane"):
            holders.setdefault(_task_project_id(task), set()).add(str(task.get("id") or ""))
    return holders


def candidate_is_leasable(candidate: Dict[str, Any], running_ids: Set[str], *,
                         cold_holders: Dict[str, Set[str]] | None = None) -> bool:
    """True when ``candidate`` may be assigned now under the one-writer rule."""
    if not _is_lane_occupant(candidate):
        return True
    project = _task_project_id(candidate)
    return project not in running_ids and not ((cold_holders or {}).get(project, set())
                                               - {str(candidate.get("id") or "")})


def task_lane_project_state(running: Any, pending: Any, tid: Any) -> dict:
    """Scope and exact carrier (including absence) for a conversion rollback.

    Walks the same RUNNING map and PENDING list as ``mark_task_project``, so the two
    can never disagree about where a task's in-memory scope lives. The caller MUST
    hold the queue lock from this read through the mark. Carrier values are replaced,
    never mutated, so the saved evidence remains the original preparation's."""
    key = str(tid or "")
    if not key:
        return {}
    meta = running.get(key) if hasattr(running, "get") else None
    rtask = _as_task(meta) if isinstance(meta, dict) else None
    if isinstance(rtask, dict):
        return {k: rtask[k] for k in ("project_id", "_project_admission") if k in rtask}
    for item in (pending or ()):
        ptask = _as_task(item)
        if isinstance(ptask, dict) and str(ptask.get("id") or "") == key:
            return {k: ptask[k] for k in ("project_id", "_project_admission") if k in ptask}
    return {}


def mark_task_project(running: Any, pending: Any, tid: Any, pid: Any, *, authority: str = "",
                      project: dict | None = None, restore: dict | None = None) -> bool:
    """Mark PENDING work only, under Q, after its durable binding commits.

    RUNNING owns its old execution lease until it leaves that execution. Folder
    and admission evidence are refreshed together at its next assignment.
    """
    key = str(tid or "")
    scope = str(pid or "").strip()
    if not key or (not scope and authority != "binding"):
        return False
    rows = []
    if hasattr(running, "get") and running.get(key) is not None:
        return False
    for item in (pending or ()):
        ptask = _as_task(item)
        if isinstance(ptask, dict) and str(ptask.get("id") or "") == key:
            rows.append(ptask)
    if authority != "binding" and any(
        str(row.get("project_id") or "").strip() not in ("", scope) for row in rows
    ):
        return False
    updated = False
    if project is not None:
        from ouroboros.project_admission import project_admission_basis, validate_project_admission

        basis = validate_project_admission(project_admission_basis(scope, project, frozen=True))
    for row in rows:
        row["project_id"] = scope
        if restore is not None:
            for field in ("project_id", "_project_admission"):
                row.pop(field, None)
                if field in restore:
                    row[field] = restore[field]
        elif project is not None:
            row["_project_admission"] = basis
            row.pop("_project_scope_none", None)
        updated = True
    return updated


__all__ = [
    "candidate_is_leasable",
    "cold_project_holders",
    "mark_task_project",
    "running_project_ids",
    "task_lane_project_state",
]
