"""Host-listed independent roots, as a worker reads them (owner decision 6C/6=A).

A task may message any active independent root the host lists -- pooled,
Swarm, project or headless (``chat_id`` 0) roots included -- and the list is
read from the supervisor's own durable projections, never rebuilt by the
worker: pooled roots from ``state/queue_snapshot.json`` (the same file the
cancel and scheduling receipts already read, dated by
``task_status.queue_snapshot_observation``), direct-chat roots from the small
fragment the supervisor main loop writes off its actor locks
(``supervisor/direct_roots.py``).  No roster service, cache or timer: each
read is one pair of small JSON reads.

The 40-row cap is presentation only -- the addressability gate consults every
row -- and the cut is disclosed in the note.  The note is appended to the
transcript as a ``[System task message]`` TAIL row only when the roster
changed since the last note, never merged into a row the model already saw.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.task_status import _load_queue_snapshot, queue_snapshot_observation
from ouroboros.utils import read_json_dict

#: The supervisor's off-lock projection of live direct-chat roots.
DIRECT_ROOTS_FRAGMENT = pathlib.Path("state") / "direct_roots.json"
#: How many rows the transcript note shows; the gate reads all of them.
ROSTER_NOTE_CAP = 40
#: Attribute the last note's fingerprint is parked on, per execution slot.
FINGERPRINT_ATTR = "_peer_roster_fingerprint"


def _compact_root(task_id: str, task: Dict[str, Any], *, status: str, direct: bool = False) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "title": str(task.get("title") or "").strip(),
        "chat_id": task.get("chat_id"),
        "project_id": str(task.get("project_id") or ""),
        "status": status,
        "drive_root": str(task.get("child_drive_root") or task.get("drive_root") or ""),
        "direct_chat": direct,
    }


def independent_roots(drive_root: pathlib.Path) -> Dict[str, Any]:
    """Every host-listed active independent root, compact (id, title, room, status).

    Pooled rows come from the queue snapshot (running, then pending); a
    subagent -- a row with a parent or the subagent role -- is not an
    independent root.  ``incomplete`` is the ONE aggregate fact: the snapshot
    is missing, unreadable or stale, or the direct-roots fragment could not
    read every actor.  Never raises.
    """
    root = pathlib.Path(drive_root)
    snapshot = _load_queue_snapshot(root)
    observation = queue_snapshot_observation(snapshot)
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    if not (snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid")):
        for status_key in ("running", "pending"):
            for row in snapshot.get(status_key) or []:
                if not isinstance(row, dict):
                    continue
                task = row.get("task") if isinstance(row.get("task"), dict) else {}
                task_id = str(row.get("id") or task.get("id") or "").strip()
                if (
                    not task_id or task_id in seen
                    or str(task.get("parent_task_id") or "").strip()
                    or str(task.get("delegation_role") or "") == "subagent"
                ):
                    continue
                seen.add(task_id)
                rows.append(_compact_root(task_id, task, status=status_key))
    fragment = read_json_dict(root / DIRECT_ROOTS_FRAGMENT) or {}
    for row in fragment.get("roots") or []:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or "").strip()
        if task_id and task_id not in seen:
            seen.add(task_id)
            rows.append(_compact_root(task_id, row, status="running", direct=True))
    return {
        "roots": rows,
        "incomplete": bool(fragment.get("incomplete")) or not bool(observation.get("fresh")),
        "queue_snapshot": observation,
    }


def host_listed_independent_root(drive_root: pathlib.Path, task_id: str) -> Optional[Dict[str, Any]]:
    """The roster row for ``task_id``, or None when the host lists no such root."""
    wanted = str(task_id or "").strip()
    if not wanted:
        return None
    try:
        return next(
            (row for row in independent_roots(drive_root)["roots"] if row["task_id"] == wanted),
            None,
        )
    except Exception:
        return None


def roster_fingerprint(roster: Dict[str, Any], *, exclude: str = "") -> tuple:
    return tuple(sorted(
        (row["task_id"], row["title"], str(row.get("chat_id")), row["project_id"], row["status"])
        for row in roster.get("roots") or [] if row["task_id"] != exclude
    ))


def render_roster_note(roster: Dict[str, Any], *, exclude: str = "") -> str:
    """The compact TAIL note: id, title, room per root; the cut and gaps disclosed."""
    rows = [row for row in roster.get("roots") or [] if row["task_id"] != exclude]
    shown = rows[:ROSTER_NOTE_CAP]
    lines = [
        "[System task message]",
        "[INDEPENDENT_ROOTS] Active independent tasks the host lists. You may message "
        "any of them with steer_task(task_id, message); it arrives as a message from "
        "THIS task (never as owner text) and files cannot be attached to it.",
    ]
    for row in shown:
        room = f"project={row['project_id']}" if row["project_id"] else f"chat={row.get('chat_id')}"
        title = row["title"] or "(untitled)"
        lines.append(f"- {row['task_id']} · {title} · {room} · {row['status']}")
    if not shown:
        lines.append("- (none)")
    if len(rows) > len(shown):
        lines.append(f"…and {len(rows) - len(shown)} more not shown.")
    if roster.get("incomplete"):
        lines.append("(roster incomplete: a host projection was stale or unreadable at this read)")
    return "\n".join(lines)


def maybe_append_roster_note(ctx: Any, messages: List[Dict[str, Any]], drive_root: Any) -> bool:
    """Append the roster note for an independent root when the roster CHANGED.

    Direct chat turns already receive the host manifest in their metadata and
    subagents address their tree through forward_to_worker/escalate, so only a
    pooled independent root gets the note.  Called after compaction and before
    the send, as a tail append: a row the model already saw is never rewritten
    (``_append_or_merge_user_content`` refuses to merge into a sent row).
    """
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    if bool(getattr(ctx, "is_direct_chat", False)) or str(metadata.get("delegation_role") or "") == "subagent":
        return False
    task_id = str(getattr(ctx, "task_id", "") or "")
    canonical = pathlib.Path(str(
        metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "") or drive_root or ""
    ) or ".")
    try:
        roster = independent_roots(canonical)
    except Exception:
        return False
    fingerprint = roster_fingerprint(roster, exclude=task_id)
    if fingerprint == getattr(ctx, FINGERPRINT_ATTR, None):
        return False
    setattr(ctx, FINGERPRINT_ATTR, fingerprint)
    from ouroboros.loop_messages import _append_or_merge_user_content

    _append_or_merge_user_content(messages, render_roster_note(roster, exclude=task_id), slot=ctx)
    return True
