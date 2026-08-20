"""Recurring schedules: the durable file, the skill sync, and what they enqueue.

Owns state/scheduled_tasks.json and the periodic reconciliation of skill-declared
schedules into it, then turns a schedule that is due into a queued task — skipping
any whose previous run is still pending or running.

The sync throttle is this module's own clock, not queue state: the writer and the
reader are both here.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import time
import uuid
from typing import Any, Dict, List
from ouroboros.contracts.task_contract import build_task_contract, normalize_allowed_resources
from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS, schedule_slug
from ouroboros.skill_loader import skill_identity_collision_names
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso
from supervisor.task_lifecycle import record_scheduled_admission
from supervisor.schedule_time import (
    next_cron_time as _next_cron_time,
    once_due as _once_due,
    parse_schedule_time as _parse_schedule_time,
    prune_consumed_once_records as _prune_consumed_once,
    record_last_error as _record_last_error,
    schedule_next_run as _schedule_next_run,
    timezone_for_schedule as _timezone_for_schedule,
)


def _queue():
    """The parent module, read at call time.

    The queue owns PENDING/RUNNING, the drive root, the liveness settings and the lock that guards them, and ``init``/``init_queue_refs`` REBIND those names. Reading them through the module is what keeps one binding: a from-import here would freeze the value this module saw at import time.
    """
    from supervisor import queue

    return queue


log = logging.getLogger(__name__)


_last_skill_schedule_sync: float = 0.0


_SKILL_SCHEDULE_SYNC_INTERVAL_SEC: float = 60.0


def _scheduled_tasks_path(drive_root: pathlib.Path | None = None) -> pathlib.Path:
    return pathlib.Path(drive_root or _queue().DRIVE_ROOT) / _queue().SCHEDULED_TASKS_FILE


def list_scheduled_tasks(drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Return the persisted scheduled task table."""
    data = read_json_dict(_scheduled_tasks_path(drive_root)) or {}
    if not isinstance(data, dict):
        data = {}
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        data["tasks"] = []
    data.setdefault("schema_version", 1)
    return data


def _write_scheduled_tasks(data: Dict[str, Any], drive_root: pathlib.Path | None = None) -> None:
    path = _scheduled_tasks_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data, trailing_newline=True)


def upsert_scheduled_task(record: Dict[str, Any], *, drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Create or replace a scheduled task record."""
    with _queue()._queue_lock:
        data = list_scheduled_tasks(drive_root)
        tasks = [item for item in data.get("tasks") or [] if isinstance(item, dict)]
        incoming = dict(record)
        schedule_id = str(incoming.get("id") or "").strip() or uuid.uuid4().hex[:8]
        incoming["id"] = schedule_id
        incoming.setdefault("enabled", True)
        incoming.setdefault("created_at", utc_now_iso())
        incoming["updated_at"] = utc_now_iso()
        if not incoming.get("next_run_at"):
            incoming["next_run_at"] = _schedule_next_run(incoming)
        tasks = [item for item in tasks if str(item.get("id") or "") != schedule_id]
        tasks.append(incoming)
        data["tasks"] = tasks
        _write_scheduled_tasks(data, drive_root)
        return incoming


def remove_scheduled_task(schedule_id: str, *, drive_root: pathlib.Path | None = None) -> bool:
    """Remove a scheduled task record by id."""
    wanted = str(schedule_id or "").strip()
    if not wanted:
        return False
    with _queue()._queue_lock:
        data = list_scheduled_tasks(drive_root)
        tasks = [item for item in data.get("tasks") or [] if isinstance(item, dict)]
        kept = [item for item in tasks if str(item.get("id") or "") != wanted]
        if len(kept) == len(tasks):
            return False
        data["tasks"] = kept
        _write_scheduled_tasks(data, drive_root)
        return True


def sync_skill_schedules(skills: List[Any], *, drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Sync reviewed skill manifest scheduled_tasks into the core schedule table."""
    with _queue()._queue_lock:
        data = list_scheduled_tasks(drive_root)
        tasks = [item for item in data.get("tasks") or [] if isinstance(item, dict)]
        by_id = {str(item.get("id") or ""): dict(item) for item in tasks}
        touched: list[str] = []
        blocked_skill_names = {
            str(getattr(skill, "name", "") or "") for skill in skills
            if bool(getattr(skill, "identity_collision", False))
        }
        changed = False
        for skill in skills:
            if bool(getattr(skill, "identity_collision", False)):
                # Preserve prior rows: a collision is not a removed/runnable skill.
                continue
            manifest = getattr(skill, "manifest", None)
            for spec in list(getattr(manifest, "scheduled_tasks", []) or []):
                if not isinstance(spec, dict):
                    continue
                name = str(spec.get("name") or "").strip()
                cron = str(spec.get("cron") or "").strip()
                if not name or not cron:
                    continue
                schedule_id = schedule_slug("skill", str(getattr(skill, "name", "")), name)
                touched.append(schedule_id)
                # Schedule readiness plus the supervised_task permission.
                try:
                    from ouroboros.skill_readiness import skill_readiness_for_execution
                    schedule_ready = skill_readiness_for_execution(pathlib.Path(drive_root or _queue().DRIVE_ROOT), skill).ready
                except Exception:
                    log.debug("skill schedule readiness probe failed for %s", getattr(skill, "name", ""), exc_info=True)
                    schedule_ready = False
                schedule_ready = schedule_ready and "supervised_task" in set(
                    getattr(manifest, "permissions", []) or []
                )
                record = by_id.get(schedule_id, {})
                trigger = {"type": "cron", "expr": cron}
                timing_changed = (
                    dict(record.get("trigger") or {}) != trigger
                    or str(record.get("timezone") or "") != str(spec.get("timezone") or "")
                    or str(record.get("skill_content_hash") or "") != str(getattr(skill, "content_hash", ""))
                )
                next_record = {
                    **record,
                    "id": schedule_id,
                    "name": f"{getattr(skill, 'name', '')}/{name}",
                    "description": str(spec.get("description") or f"Scheduled skill task {getattr(skill, 'name', '')}/{name}"),
                    "enabled": bool(schedule_ready),
                    "timezone": str(spec.get("timezone") or ""),
                    "trigger": trigger,
                    "task": {
                        "type": "task",
                        "text": (
                            f"Run reviewed scheduled skill task `{getattr(skill, 'name', '')}/{name}`. "
                            "Use skill_exec or the reviewed extension surface as appropriate, then report outcome."
                        ),
                        "metadata": {
                            "source": "skill_scheduled_task",
                            "skill": str(getattr(skill, "name", "")),
                            "scheduled_task": name,
                        },
                    },
                    "source": "skill_manifest",
                    "skill": str(getattr(skill, "name", "")),
                    "skill_content_hash": str(getattr(skill, "content_hash", "")),
                    "updated_at": utc_now_iso(),
                }
                if timing_changed or not next_record.get("next_run_at"):
                    next_record["next_run_at"] = _schedule_next_run(next_record)
                if next_record != record:
                    by_id[schedule_id] = next_record
                    changed = True
        for schedule_id, record in list(by_id.items()):
            if (
                str(record.get("source") or "") == "skill_manifest"
                and str(record.get("skill") or "") not in blocked_skill_names
                and schedule_id not in touched
            ):
                by_id.pop(schedule_id, None)
                changed = True
        if changed:
            data["tasks"] = list(by_id.values())
            _write_scheduled_tasks(data, drive_root)
        return {"changed": changed, "skill_schedule_ids": touched}


def resync_skill_schedules(drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Mirror discovered manifest schedules after skill lifecycle changes."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_loader import discover_skills

    root = pathlib.Path(drive_root or _queue().DRIVE_ROOT)
    return sync_skill_schedules(
        discover_skills(root, repo_path=get_skills_repo_path()),
        drive_root=root,
    )


def _schedule_running_or_queued(schedule_id: str) -> bool:
    if not schedule_id:
        return False
    for task in _queue().PENDING:
        meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if str(meta.get("schedule_id") or "") == schedule_id:
            return True
    for meta in _queue().RUNNING.values():
        task = meta.get("task") if isinstance(meta, dict) else None
        task_meta = task.get("metadata") if isinstance(task, dict) and isinstance(task.get("metadata"), dict) else {}
        if str(task_meta.get("schedule_id") or "") == schedule_id:
            return True
    return False


def _task_from_schedule(record: Dict[str, Any]) -> Dict[str, Any]:
    template = dict(record.get("task") or {})
    owner_chat_id = _queue().load_state().get("owner_chat_id") or 0
    task_id = uuid.uuid4().hex[:8]
    session_id = str(template.get("session_id") or f"schedule-{record.get('id') or task_id}")
    raw_metadata = template.get("metadata") if isinstance(template.get("metadata"), dict) else {}
    metadata = {
        key: value for key, value in dict(raw_metadata).items()
        if key not in RESERVED_TEMPLATE_FIELDS
    }
    task = {
        "id": task_id,
        "type": "task",
        "text": str(template.get("text") or template.get("description") or record.get("description") or record.get("name") or "Scheduled task"),
        "description": str(template.get("description") or template.get("text") or record.get("description") or record.get("name") or "Scheduled task"),
        "chat_id": template.get("chat_id") if template.get("chat_id") not in (None, "") else owner_chat_id,
        "priority": int(template["priority"]) if str(template.get("priority") or "").strip().lstrip("-").isdigit() else None,
        "root_task_id": task_id,
        "session_id": session_id,
        "actor_id": "scheduler",
        "delegation_role": "root",
        "metadata": metadata,
    }
    for key in ("attachments", "context", "expected_output", "constraints", "deadline_at"):
        if key in template:
            task[key] = template[key]
    allowed_resources = normalize_allowed_resources(template.get("allowed_resources") or metadata.get("allowed_resources") or {})
    if allowed_resources:
        task["allowed_resources"] = allowed_resources
    existing_contract = template.get("task_contract") if isinstance(template.get("task_contract"), dict) else {}
    if existing_contract:
        task["task_contract"] = existing_contract
    task["task_contract"] = build_task_contract(task)
    task["metadata"]["schedule_id"] = str(record.get("id") or "")
    task["metadata"]["schedule_name"] = str(record.get("name") or "")
    task["metadata"]["schedule_trigger"] = dict(record.get("trigger") or {})
    task["metadata"]["task_contract"] = task["task_contract"]
    if allowed_resources:
        task["metadata"]["allowed_resources"] = allowed_resources
    if task.get("deadline_at"):
        task["metadata"]["deadline_at"] = task.get("deadline_at")
    task["metadata"].setdefault("source", "scheduled_task")
    return task


def check_scheduled_tasks() -> None:
    """Queue due cron/on-idle schedules using the normal supervisor queue."""
    global _last_skill_schedule_sync
    with _queue()._queue_lock:
        now_monotonic = time.monotonic()
        if now_monotonic - _last_skill_schedule_sync >= _SKILL_SCHEDULE_SYNC_INTERVAL_SEC:
            _last_skill_schedule_sync = now_monotonic
            try:
                resync_skill_schedules(_queue().DRIVE_ROOT)
            except Exception:
                log.debug("Failed to sync skill schedules during scheduler tick", exc_info=True)
        data = list_scheduled_tasks()
        changed = False
        collision_names = None
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        for record in list(data.get("tasks") or []):
            if not isinstance(record, dict) or not record.get("enabled", True):
                continue
            schedule_id = str(record.get("id") or "").strip()
            if not schedule_id:
                record["id"] = uuid.uuid4().hex[:8]
                schedule_id = str(record["id"])
                changed = True
            trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
            trigger_type = str(trigger.get("type") or "cron").strip().lower()
            if _schedule_running_or_queued(schedule_id):
                continue
            tz = _timezone_for_schedule(record)
            now = now_utc.astimezone(tz)
            expr = ""
            if trigger_type == "once":
                # One-shot (B2b W=A): fires once at/after run_at via the same admission path
                # as cron, then is marked done below. A consumed receipt (non-empty completed_at)
                # NEVER re-fires even re-enabled from UI; re-arm = gateway upsert, fresh run_at.
                if record.get("completed_at"):
                    continue
                due, once_error = _once_due(trigger, tz, now)
                if once_error:
                    changed = _record_last_error(record, once_error) or changed
                    continue
                if not due:
                    continue
            elif trigger_type != "cron":
                changed = _record_last_error(record, f"unsupported trigger type: {trigger_type}") or changed
                continue
            else:
                expr = str(trigger.get("expr") or record.get("cron") or "").strip()
                if not expr:
                    changed = _record_last_error(record, "missing cron expression") or changed
                    continue
                next_run = _parse_schedule_time(record.get("next_run_at"), tz)
                if next_run is None:
                    try:
                        next_run = _next_cron_time(expr, now - datetime.timedelta(minutes=1))
                        record["next_run_at"] = next_run.isoformat()
                        changed = True
                    except Exception as exc:
                        changed = _record_last_error(record, f"{type(exc).__name__}: {exc}") or changed
                        continue
                if next_run > now:
                    continue
            if str(record.get("source") or "") == "skill_manifest":
                if collision_names is None:
                    collision_names = skill_identity_collision_names(_queue().DRIVE_ROOT)
                if str(record.get("skill") or "") in collision_names:
                    continue
            task = _task_from_schedule(record)
            try:
                from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

                write_task_result(
                    _queue().DRIVE_ROOT,
                    str(task["id"]),
                    STATUS_SCHEDULED,
                    root_task_id=str(task["id"]),
                    actor_id="scheduler",
                    delegation_role="root",
                    description=str(task.get("description") or task.get("text") or ""),
                    expected_output=str(task.get("expected_output") or ""),
                    constraints=str(task.get("constraints") or ""),
                    context=str(task.get("context") or ""),
                    allowed_resources=task.get("allowed_resources") if isinstance(task.get("allowed_resources"), dict) else {},
                    deadline_at=str(task.get("deadline_at") or ""),
                    task_contract=task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {},
                    result="Scheduled task queued.",
                    metadata=dict(task.get("metadata") or {}),
                    schedule_id=schedule_id,
                    schedule_name=str(record.get("name") or ""),
                )
            except Exception:
                log.debug("Failed to persist scheduled task result before enqueue", exc_info=True)
            admitted = _queue().enqueue_task(task)
            record["last_run_at"] = now.isoformat()
            record["last_task_id"] = task["id"]
            record_scheduled_admission(task, admitted, record)
            if trigger_type == "once":
                if not (isinstance(admitted, dict) and admitted.get("_admission_blocked")):
                    # Consumed ONLY when admission succeeded (durable receipt, never re-fired); a
                    # refused admission left the record enabled with last_error → next tick retries.
                    record["enabled"] = False
                    record["completed_at"] = now.isoformat()
                    record["next_run_at"] = ""
            else:
                try:
                    record["next_run_at"] = _next_cron_time(expr, now).isoformat()
                except Exception as exc:
                    record["last_error"] = f"{type(exc).__name__}: {exc}"
            changed = True
        # Consumed one-shot receipts age out past the unified GC retention (DEVELOPMENT
        # Runtime Cleanup SSOT; enabled records are never pruned — see the helper).
        from ouroboros.retention import age_cutoff, get_gc_retention_days

        kept, pruned = _prune_consumed_once(list(data.get("tasks") or []),
                                            age_cutoff(get_gc_retention_days()))
        if pruned:
            data["tasks"], changed = kept, True
        if changed:
            _write_scheduled_tasks(data)
            _queue().persist_queue_snapshot(reason="scheduled_tasks")
