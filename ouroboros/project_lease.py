"""One-writer lanes: a Project's and a recorded external folder's (multi-project, v6.32.0).

Pure functions the assignment pass (``supervisor/dispatch_choice``, ``worker_assignment``) reads under the
queue lock: a PENDING task skips this pass while another tree holds its Project's lane or its folder's.
Parallelism happens between Projects with different folders, between folders, and through subagent swarms
WITHIN a task; independent pooled work in one external folder waits its turn, whichever room it belongs to.

``project_id == ""`` means "no lane": ordinary unscoped tasks never serialize against each other, folder or
not (a bare workspace task is given its folder's derived Project at admission). Subagents carry their parent's
stored ``project_id`` but hold no lease of their own — the parent task IS the writer and its swarm must not
deadlock against itself, so only top-level (non-subagent) tasks count as lane occupants; the lane is their
tree's (``lane_holders``). A conversation-lane turn (an owner's room turn, a wake, a Presence body) holds no
lease either: it answers beside the room's work, so a Resume or Continue it grants starts at once.

A folder lane compares the string a row records (``workspace_root`` in ``external`` mode, or a pending move's
projection ``_placement_folder``) as admission already normalized it: no filesystem, registry or history read,
no nested folders. It orders background starts; it promises no protection against every file race.
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


def lane_folder(task: Any) -> str:
    """The external folder whose lane a lane occupant holds, as its row records it, else ``""``. A pending move's
    projection (``_placement_folder``, ``project_placement.placement_folder``) stands for the folder its readmission
    will select."""
    task = _as_task(task)
    if not _is_lane_occupant(task):
        return ""
    if "_placement_folder" in task:
        return str(task.get("_placement_folder") or "")
    return str(task.get("workspace_root") or "") if task.get("workspace_mode") == "external" else ""


def _lanes(task: Any) -> tuple:
    """The lanes a lane occupant holds: its Project's id, and ``("folder", path)`` for its recorded folder."""
    folder = lane_folder(task)
    return ((_task_project_id(task),) + ((("folder", folder),) if folder else ())) if _is_lane_occupant(task) else ()


def _asleep(task: Dict[str, Any]) -> bool:
    """Parked in its own model sleep, or granted back from it and not started yet."""
    marker = task.get("_budget_pause") or (task.get("_budget_pause_resume") or {}).get("pause")
    return isinstance(marker, dict) and marker.get("reason") == "sleep"


def lane_holders(running: Iterable[Any], pending: Iterable[Any]) -> Dict[Any, Set[str]]:
    """The trees (root ids) holding each lane (a Project's id, ``("folder", path)``); the caller holds the
    queue lock. A running top-level writer holds its lanes, and ``("folder", path, "acting")`` names the
    trees whose running, claimed or chosen row is in that folder now. A queued one keeps its lanes while its
    tree lives: its accepted cold custody, or, asleep (a sleep is its wait), a running or claimed member.
    A tree whose members are only queued takes its lanes back together or not at all, the first queued such
    tree first: only when no other tree holds any of them, so a tree that released its lanes takes them back
    only once all are free, no two trees wait on each other, and a free folder is never kept for a tree
    whose Project another tree holds."""
    from ouroboros.sleep_custody import carrier_receipt

    running, pending = list(running or ()), [task for task in pending or () if isinstance(task, dict)]
    holders: Dict[Any, Set[str]] = {}
    for item in running:
        if not (isinstance(item, dict) and item.get("lane") == LANE_CONVERSATION):
            for lane in _lanes(item):
                holders.setdefault(lane, set()).add(_tree(item))
                if isinstance(lane, tuple):
                    holders.setdefault(lane + ("acting",), set()).add(_tree(item))
    sleepers = [task for task in pending if _is_lane_occupant(task)
                and (_asleep(task) or carrier_receipt(task).get("retains_project_lane"))]
    acting, queued, returning = {_tree(item) for item in running}, {}, []  # returning: queued-only trees, in order
    for task in pending if sleepers else ():
        queued.setdefault(_tree(task), set()).add(str(task.get("id") or ""))
    for task in sleepers:
        tree = _tree(task)
        if carrier_receipt(task).get("retains_project_lane") or tree in acting:
            for lane in _lanes(task):
                holders.setdefault(lane, set()).add(tree)
        elif queued.get(tree, set()) - {str(task.get("id") or "")}:
            returning.append((tree, _lanes(task)))
    for tree, lanes in returning:
        if not any(lane in holders for lane in lanes):
            holders.update((lane, {tree}) for lane in lanes)
    return holders


def _folder_holders(candidate: Any, holders: Dict[Any, Set[str]]) -> Set[str]:
    """The other trees ``candidate``'s folder waits for. A tree that does not hold the folder waits for every
    holder, a cold one included; one already holding it (a sleeper returning) waits only for those acting
    in it now, never for another merely cold holder, so earlier holders of one folder never lock each other."""
    folder, tree = lane_folder(candidate), _tree(candidate)
    held = holders.get(("folder", folder), set()) if folder else set()
    others = held - {tree}
    return others & holders.get(("folder", folder, "acting"), set()) if tree in held else others


def candidate_is_leasable(candidate: Dict[str, Any], running_ids: Set[str], *,
                         holders: Dict[Any, Set[str]] | None = None) -> bool:
    """True when ``candidate`` may be assigned now under the one-writer rule: never beside a lane
    another tree holds (``lane_holders``; its folder's by ``_folder_holders``), and a top-level task
    never beside a running writer of its Project."""
    project = _task_project_id(candidate)
    if project and (holders or {}).get(project, set()) - {_tree(candidate)}:
        return False
    if _folder_holders(candidate, holders or {}):
        return False
    return not _is_lane_occupant(candidate) or project not in running_ids


def folder_projects(rows: Iterable[Any]) -> Dict[tuple, str]:
    """Each ``(tree, recorded folder)`` of the queue's rows (RUNNING metas or tasks) to that tree's Project, the
    first row first: the Project a folder wait names its holder by (``lane_wait``), built once per read."""
    projects: Dict[tuple, str] = {}
    for row in map(_as_task, rows or ()):
        if lane_folder(row):
            projects.setdefault((_tree(row), lane_folder(row)), _task_project_id(row))
    return projects


def lane_wait(candidate: Dict[str, Any], holders: Dict[Any, Set[str]], projects: Dict[tuple, str]) -> Dict[str, Any]:
    """The tree that keeps lane occupant ``candidate`` out now, ``{lane, task_id, project_id}`` (``lane``
    ``"project"`` or ``"folder"``, the latter with its ``folder``), else ``{}``; reads and changes nothing. Its
    Project's lane first, the nearer cause; the first blocker in sorted order; a folder holder's Project from
    ``projects`` (``folder_projects``)."""
    if not _is_lane_occupant(candidate):
        return {}
    project, folder = _task_project_id(candidate), lane_folder(candidate)
    blockers = sorted(holders.get(project, set()) - {_tree(candidate)})
    if blockers:
        return {"lane": "project", "task_id": blockers[0], "project_id": project}
    blockers = sorted(_folder_holders(candidate, holders))
    if not blockers:
        return {}
    return {"lane": "folder", "task_id": blockers[0], "project_id": projects.get((blockers[0], folder), ""),
            "folder": folder}


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
    "folder_projects",
    "lane_folder",
    "lane_holders",
    "lane_wait",
    "mark_task_project",
    "running_project_ids",
    "task_lane_project_state",
]
