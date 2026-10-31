"""One boot conversion of pre-selection queue carriers; no execution authority.

The format stamp describes the document, including an empty queue. Old names
are interpreted here once; ordinary readers use work_stop's current policy.
The pure converter also accepts stored documents so lost queue locators become
addressable again. Restore still checks terminal/cancel/source authority.
"""

import copy
import re

from ouroboros.work_stop import (
    SELECTION_KEY, STOP_REASONS, current_stop, reason_policy, refuse_continuation,
    resume_selection, stop_record,
)

FORMAT_KEY = "work_stop_format"
FORMAT_VERSION = 1
_AWAITING = "root_fence_member_pending_selection"
_POINTS = ("_budget_pause", "_budget_pause_resume", "_owner_wait_resume", "_working_recovery", SELECTION_KEY)


def _door(record):
    if isinstance(record, dict):
        if "stop_cause" in record:
            record.setdefault("door", record.pop("stop_cause"))
        for value in record.values():
            if isinstance(value, dict):
                _door(value)


def _by(hold):
    if isinstance(hold.get("by"), dict):
        return hold["by"]
    return {"actor": "ouroboros" if hold.get("actor") == "mind" else hold.get("actor", "host"),
            "task_id": hold.get("actor_task_id") or hold.get("task_id", ""), "text": hold.get("text", ""),
            "at": hold.get("at") or hold.get("held_at", "")}


def _normalize_task(task, fences, changes):
    before = copy.deepcopy(task)
    for key in (*_POINTS, "_budget_pause_hold"):
        _door(task.get(key))
    hold = task.get("_budget_pause_hold")
    if isinstance(hold, dict):
        reason = str(hold.get("reason") or "")
        by = _by(hold)
        task.pop("_budget_pause_hold")
        current = current_stop(task, fences)
        prose = reason not in STOP_REASONS and bool(reason) and not re.fullmatch(r"[a-z0-9_:]+", reason)
        if prose:
            changes.append({"task_id": task.get("id"), "action": "actor_text_archived", "text": reason})
        if hold.get("selected"):
            # A newer pause supersedes the old choice. Remaining identities are
            # revalidated by ordinary Q9 dispatch, never by its historical reason.
            same_fence = not current or (current.get("scope") == "tree" and hold.get("fence_id")
                                        and hold["fence_id"] == current.get("fence_id"))
            if not task.get("_budget_pause") and same_fence and hold.get("selected_by"):
                task[SELECTION_KEY] = resume_selection(
                    hold, selected_by=hold["selected_by"], selected_at=hold.get("selected_at", ""), by=by)
            elif not current and task.get("root_task_id", task.get("id")) != task.get("id"):
                task["_budget_pause_hold"] = stop_record(_AWAITING, held_at=hold.get("held_at", ""), by=by)
        elif hold.get("detail") == "root_resume_generation_stale":
            # U1/#2067 copied an already released cause over a current Pause.
            if not current:
                task["_budget_pause_hold"] = stop_record(_AWAITING, held_at=hold.get("held_at", ""), by=by)
        elif prose:
            pass  # Actor text is history; only a current marker/fence can stop work.
        elif reason_policy(reason)["cause"] == "refusal":
            refuse_continuation(task, hold)
        else:
            task["_budget_pause_hold"] = {**hold, "release": hold.get("release", reason_policy(reason)["release"]),
                                         "by": by}
            for key in ("selected", "selected_at", "selected_by", "actor", "actor_task_id", "text", "cause"):
                task["_budget_pause_hold"].pop(key, None)
    if task != before:
        changes.append({"task_id": task.get("id"), "action": "carriers_normalized"})


def _ghost(document):
    from ouroboros.budget_pause import exact_pause_marker
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, resolve_task_lineage

    if document.get("status") in _TRULY_TERMINAL_STATUSES | {"cancel_requested"}:
        return None
    pause, limit = document.get("budget_pause") or {}, document.get("resource_limit") or {}
    if pause.get("state") not in {"pausing", "paused", "resume_granted"} and limit.get("status") != "budget_hold":
        return None
    from supervisor.queue_snapshot import _PENDING_FIELDS

    task = {key: copy.deepcopy(document[key]) for key in _PENDING_FIELDS if document.get(key) is not None}
    task.update(id=document["task_id"], type=document.get("type") or "task",
                chat_id=document.get("chat_id", 0), _attempt=document.get("task_attempt") or 1)
    lineage = resolve_task_lineage(task["id"], **{key: document.get(key) for key in (
        "metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")})
    task.update({key: lineage[key] for key in ("root_task_id", "parent_task_id")})
    if pause.get("state") in {"pausing", "paused", "resume_granted"}:
        task["_budget_pause"] = exact_pause_marker(pause, default_root=task["root_task_id"])
    if limit.get("status") == "budget_hold":
        task["_budget_pause_hold"] = {key: copy.deepcopy(value) for key, value in limit.items()
                                      if key not in {"status", "resume_policy", "auto_resume", "exact_continuation"}}
    return task


def normalize_snapshot(snapshot, documents=()):
    """Mutate a legacy snapshot, returning journal facts. Repeat calls do nothing.

    Documents are observations, never modified. Missing rows acquire no grant or
    no-dispatch proof; an explicit Resume must recheck their actual source.
    """
    if snapshot.get(FORMAT_KEY) == FORMAT_VERSION:
        return []
    documents = {row["task_id"]: row for row in documents if row.get("task_id")}
    changes = []
    tasks = [row["task"] for lane in ("pending", "running") for row in snapshot.get(lane) or []
             if isinstance(row, dict) and isinstance(row.get("task"), dict)]
    present = {task.get("id") for task in tasks}
    for task_id, document in documents.items():
        if task_id not in present and (task := _ghost(document)) is not None:
            snapshot.setdefault("pending", []).append({"id": task_id, "task": task})
            tasks.append(task)
            present.add(task_id)
            changes.append({"task_id": task_id, "action": "queue_locator_restored"})
    budget = {f["root_task_id"]: f for f in snapshot.get("budget_root_fences") or []
              if isinstance(f, dict) and f.get("root_task_id")}
    for task in tasks:
        root = str(task.get("root_task_id") or task.get("id") or "")
        fences = {"budget": budget, "owner": (documents.get(root) or {}).get("owner_pause") or {}}
        _normalize_task(task, fences, changes)
        if any(c["task_id"] == task.get("id") and c["action"] == "queue_locator_restored" for c in changes):
            if not current_stop(task, fences):
                task["_budget_pause_hold"] = stop_record(_AWAITING, held_at="")
    snapshot[FORMAT_KEY] = FORMAT_VERSION
    snapshot["pending_count"] = len(snapshot.get("pending") or [])
    return changes


def normalize_for_restore(snapshot, root):
    """Boot adapter: observe current results, journal conversions, publish via Q."""
    if snapshot.get(FORMAT_KEY) == FORMAT_VERSION or snapshot.get("_snapshot_invalid") or snapshot.get("_snapshot_missing"):
        return False
    from ouroboros.task_results import list_task_results, load_task_result
    from ouroboros.utils import append_jsonl, utc_now_iso

    documents = list_task_results(root, statuses=["requested", "scheduled", "running", "interrupted"], strict=True)
    known = {row["task_id"] for row in documents}
    roots = {str(task.get("root_task_id") or task.get("id") or "")
             for lane in ("pending", "running") for row in snapshot.get(lane) or []
             if isinstance(row, dict) and isinstance(task := row.get("task"), dict)}
    documents += [document for tid in roots - known if tid and (document := load_task_result(root, tid, strict=True))]
    changes = normalize_snapshot(snapshot, documents)
    append_jsonl(root / "logs" / "supervisor.jsonl", {
        "ts": utc_now_iso(), "type": "work_stop_format_normalized", "format": FORMAT_VERSION, "changes": changes})
    return True
