"""SUBAGENT ADMISSION: may this parent spawn another child, and how deep?

Two bounds, asked together because they fail differently. The CAP counts what is
already live under a root — a parent at its cap must wait, not be refused forever —
while the DEPTH reservation asks whether a new level can still be opened at all. A
tree that passes the cap and fails the depth check is not the same situation as the
reverse, and answering with one number would hide that.

"Active" is deliberately liveness, not status: a task whose row says finished but
whose worker is still winding down still occupies a slot, and counting it as free is
how a cap silently becomes a suggestion.
"""

from __future__ import annotations

from typing import Any, Dict
from ouroboros.config import (
    MAX_ACTIVE_SUBAGENTS_HARD_CAP,
)


def _is_active_subagent_task(task: Dict[str, Any], root_task_id: str) -> bool:
    if str(task.get("root_task_id") or "") != root_task_id:
        return False
    return str(task.get("delegation_role") or "") == "subagent"


def _active_subagent_count(root_task_id: str, pending: list, running: dict) -> int:
    count = 0
    for task in pending:
        if isinstance(task, dict) and _is_active_subagent_task(task, root_task_id):
            count += 1
    for meta in running.values():
        task = meta.get("task") if isinstance(meta, dict) else None
        if isinstance(task, dict) and _is_active_subagent_task(task, root_task_id):
            count += 1
    return count


def _task_own_id(task: Dict[str, Any]) -> str:
    return str(task.get("id") or task.get("task_id") or "").strip()


def _iter_tree_subagent_tasks(root_task_id: str, pending: list, running: dict):
    for task in pending:
        if isinstance(task, dict) and _is_active_subagent_task(task, root_task_id):
            yield task
    for meta in running.values():
        task = meta.get("task") if isinstance(meta, dict) else None
        if isinstance(task, dict) and _is_active_subagent_task(task, root_task_id):
            yield task


def _depth_reservation_admits(
    root_task_id: str, parent_id: Any, pending: list, running: dict, max_active: int
) -> bool:
    """FR2 depth-aware reservation: when the tree is at the per-root active cap,
    still admit a child whose parent is a RUNNING subagent that has NO active
    direct child yet — one reserved direct child per running subagent — so a deep
    cooperative build is not starved by a wide first level. Bounded by a hard
    ceiling (2x the cap, capped at the documented per-root hard max
    ``config.MAX_ACTIVE_SUBAGENTS_HARD_CAP`` = 500) so the
    reservation can never unbound the tree; structural depth/max_children gates
    still apply on top."""
    parent = str(parent_id or "").strip()
    if not parent:
        return False
    parent_running = any(
        _task_own_id(t) == parent
        for meta in running.values()
        if isinstance(meta, dict) and isinstance((t := meta.get("task")), dict) and _is_active_subagent_task(t, root_task_id)
    )
    if not parent_running:
        return False
    direct_children = sum(
        1 for t in _iter_tree_subagent_tasks(root_task_id, pending, running)
        if str(t.get("parent_task_id") or "").strip() == parent
    )
    if direct_children >= 1:
        return False
    hard_ceiling = min(MAX_ACTIVE_SUBAGENTS_HARD_CAP, 2 * max(1, int(max_active)))
    return _active_subagent_count(root_task_id, pending, running) < hard_ceiling


def _subagent_cap_blocks(root_task_id: str, parent_id: Any, pending: list, running: dict, max_active: int) -> bool:
    """A subagent schedule is rejected when the tree is at the per-root active cap AND
    the FR2 depth-aware reservation does not admit it."""
    return (
        _active_subagent_count(root_task_id, pending, running) >= max_active
        and not _depth_reservation_admits(root_task_id, parent_id, pending, running, max_active)
    )


__all__ = [
    "_is_active_subagent_task",
    "_active_subagent_count",
    "_task_own_id",
    "_iter_tree_subagent_tasks",
    "_depth_reservation_admits",
    "_subagent_cap_blocks",
]
