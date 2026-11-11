"""One-writer-per-project lease helpers (multi-project, v6.32.0).

Pure functions consumed by ``supervisor/workers.py::assign_tasks`` under the
queue lock: a PENDING task whose ``project_id`` is already RUNNING is skipped
this assignment pass (projects serialize internally; parallelism happens
BETWEEN projects and via subagent swarms WITHIN a task).

``project_id == ""`` means "no lane": ordinary unscoped tasks never serialize
against each other. Subagents carry their parent's stored ``project_id`` but
hold no lease of their own — the parent task IS the project's writer and its
swarm must not deadlock against itself, so only top-level (non-subagent)
tasks count as lane occupants; the lane is their tree's (``lane_holders``). A
conversation-lane turn (an owner's room turn, a wake, a Presence body) holds no
lease either: it answers beside the room's work, so a Resume or Continue it
grants starts at once.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Set

from ouroboros.task_carrier import LANE_CONVERSATION


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
    task dicts); read under the queue lock by the caller. A conversation-lane
    turn is known by its RUNNING row's ``lane``: a bare task dict always counts.
    """
    out: Set[str] = set()
    for task in running or ():
        conversation = isinstance(task, dict) and task.get("lane") == LANE_CONVERSATION  # its RUNNING row's carrier
        if _is_lane_occupant(task) and not conversation:
            out.add(_task_project_id(task))
    return out


def _tree(task: Any) -> str:
    """The tree a row belongs to: its logical root (a root's retry keeps the original id)."""
    task = _as_task(task)
    return str(task.get("root_task_id") or task.get("id") or "") if isinstance(task, dict) else ""


def _asleep(task: Dict[str, Any]) -> bool:
    """Parked in its own model sleep, or granted back from it and not started yet."""
    marker = task.get("_budget_pause") or (task.get("_budget_pause_resume") or {}).get("pause")
    return isinstance(marker, dict) and marker.get("reason") == "sleep"


def lane_holders(running: Iterable[Any], pending: Iterable[Any]) -> Dict[str, Set[str]]:
    """The trees (root ids) holding each project lane; the caller holds the queue lock. A running
    top-level writer holds its lane. A queued one keeps it while its tree lives: its accepted cold
    custody, or, asleep (a sleep is its wait), a running or claimed member; with members only queued,
    a lane no other tree holds, the first queued such tree keeping it, so a tree that released its
    lane takes it back only once it is free and no two trees wait on each other."""
    from ouroboros.sleep_custody import carrier_receipt

    running, pending = list(running or ()), [task for task in pending or () if isinstance(task, dict)]
    holders: Dict[str, Set[str]] = {}
    for item in running:
        if _is_lane_occupant(item) and not (isinstance(item, dict) and item.get("lane") == LANE_CONVERSATION):
            holders.setdefault(_task_project_id(item), set()).add(_tree(item))
    sleepers = [task for task in pending if _is_lane_occupant(task)
                and (_asleep(task) or carrier_receipt(task).get("retains_project_lane"))]
    acting, queued, waiting = {_tree(item) for item in running}, {}, {}  # waiting: project -> first queued-only tree
    for task in pending if sleepers else ():
        queued.setdefault(_tree(task), set()).add(str(task.get("id") or ""))
    for task in sleepers:
        tree, project = _tree(task), _task_project_id(task)
        if carrier_receipt(task).get("retains_project_lane") or tree in acting:
            holders.setdefault(project, set()).add(tree)
        elif queued.get(tree, set()) - {str(task.get("id") or "")}:
            waiting.setdefault(project, tree)
    for project, tree in waiting.items():
        holders.setdefault(project, {tree})
    return holders


def candidate_is_leasable(candidate: Dict[str, Any], running_ids: Set[str], *,
                         holders: Dict[str, Set[str]] | None = None) -> bool:
    """True when ``candidate`` may be assigned now under the one-writer rule: never beside a lane
    another tree holds (``lane_holders``), and a top-level task never beside a running writer."""
    project = _task_project_id(candidate)
    if project and (holders or {}).get(project, set()) - {_tree(candidate)}:
        return False
    return not _is_lane_occupant(candidate) or project not in running_ids


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
    "lane_holders",
    "mark_task_project",
    "running_project_ids",
    "task_lane_project_state",
]
