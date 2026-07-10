"""Supervisor task queue, persistence, timeouts, and evolution scheduling."""

from __future__ import annotations

import datetime
import json
import logging
import math
import pathlib
import queue as _stdqueue  # noqa: F401 — re-exported for the test suite's reap-queue isolation
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from supervisor.state import (
    load_state, append_jsonl, atomic_write_text,
    QUEUE_SNAPSHOT_PATH, budget_remaining, EVOLUTION_BUDGET_RESERVE, reconstruct_task_cost,
)
from supervisor.message_bus import send_with_budget
from ouroboros.config import (
    FINALIZATION_GRACE_DEFAULT_SEC,
    get_finalization_grace_sec,
    get_per_call_timeout_ceiling_sec,
    get_task_abs_ceiling_sec,
    get_task_idle_timeout_sec,
)
from ouroboros.contracts.task_contract import attach_task_contract, build_task_contract, normalize_allowed_resources
from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS, schedule_slug
from ouroboros.outcomes import normalize_outcome_axes, terminal_outcome_axes
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso
from supervisor.evolution_lifecycle import (
    _read_evolution_campaign,
    _write_evolution_campaign,
    begin_evolution_transaction,
    build_evolution_task_text,
    evolution_block_reason,
    notify_owner_cycle_outcome,
    pause_evolution_campaign,
    start_evolution_campaign,
)

log = logging.getLogger(__name__)


DRIVE_ROOT: pathlib.Path = pathlib.Path.home() / "Ouroboros" / "data"
SOFT_TIMEOUT_SEC: int = 600
HARD_TIMEOUT_SEC: int = 1800
HEARTBEAT_STALE_SEC: int = 120
QUEUE_MAX_RETRIES: int = 1
FINALIZATION_GRACE_SEC: int = FINALIZATION_GRACE_DEFAULT_SEC
SCHEDULED_TASKS_FILE = pathlib.Path("state") / "scheduled_tasks.json"
# BUG3: pause a campaign whose objective fails to absorb after this many reviewed cycles.
# Mirrors the consecutive-failures threshold; keyed on the objective fingerprint, not failures.
OBJECTIVE_REPEAT_CAP: int = 3


def _task_deadline_ts(task: Dict[str, Any]) -> float:
    raw = str(task.get("deadline_at") or "").strip()
    if not raw:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        raw = str(metadata.get("deadline_at") or "").strip()
    if not raw:
        contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
        raw = str(contract.get("deadline_at") or "").strip()
    if not raw:
        return 0.0
    try:
        parsed = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return float(parsed.timestamp())
    except Exception:
        return 0.0


def init(drive_root: pathlib.Path, soft_timeout: int, hard_timeout: int) -> None:
    global DRIVE_ROOT, SOFT_TIMEOUT_SEC, HARD_TIMEOUT_SEC, FINALIZATION_GRACE_SEC
    DRIVE_ROOT = drive_root
    SOFT_TIMEOUT_SEC = soft_timeout
    HARD_TIMEOUT_SEC = hard_timeout
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec()


def refresh_timeouts_from_settings(settings: dict) -> None:
    """Hot-reload soft/hard timeouts independently, ignoring bad values."""
    global SOFT_TIMEOUT_SEC, HARD_TIMEOUT_SEC, FINALIZATION_GRACE_SEC
    soft_raw = settings.get("OUROBOROS_SOFT_TIMEOUT_SEC")
    if soft_raw is not None:
        try:
            SOFT_TIMEOUT_SEC = int(soft_raw)
        except (TypeError, ValueError):
            pass
    hard_raw = settings.get("OUROBOROS_HARD_TIMEOUT_SEC")
    if hard_raw is not None:
        try:
            HARD_TIMEOUT_SEC = int(hard_raw)
        except (TypeError, ValueError):
            pass
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec(settings)


# Set by workers.init_queue_refs().
PENDING: List[Dict[str, Any]] = []
RUNNING: Dict[str, Dict[str, Any]] = {}
QUEUE_SEQ_COUNTER_REF: Dict[str, int] = {"value": 0}

# Guards PENDING/RUNNING mutations across main loop, direct chat, watchdog.
_queue_lock = threading.RLock()
_last_skill_schedule_sync: float = 0.0
_SKILL_SCHEDULE_SYNC_INTERVAL_SEC: float = 60.0

# Variant A off-loop worker reaper lives in supervisor/task_reaper.py (extracted for
# module size). Re-export the thin names the enforce path and tests use; monkeypatching
# these queue-module names still works because the enforce path references them here.
from supervisor.task_reaper import (  # noqa: E402,F401 — re-exported for enforce path + tests
    ensure_reaper_started as _ensure_reaper_started,
    reap_queue as _reap_queue,
    reap_timed_out_task as _reap_timed_out_task,
)


def init_queue_refs(pending: List[Dict[str, Any]], running: Dict[str, Dict[str, Any]],
                    seq_counter_ref: Dict[str, int]) -> None:
    """Bind queue structures owned by workers.py."""
    global PENDING, RUNNING, QUEUE_SEQ_COUNTER_REF
    PENDING = pending
    RUNNING = running
    QUEUE_SEQ_COUNTER_REF = seq_counter_ref


def _task_priority(task_type: str) -> int:
    t = str(task_type or "").strip().lower()
    if t in ("task", "review", "deep_self_review"):
        return 0
    if t == "evolution":
        return 1
    return 2


def _queue_sort_key(task: Dict[str, Any]) -> Tuple[int, int]:
    _pr = task.get("priority")
    pr = int(_pr) if _pr is not None else _task_priority(str(task.get("type") or ""))
    _seq = task.get("_queue_seq")
    seq = int(_seq) if _seq is not None else 0
    return pr, seq


def sort_pending() -> None:
    """Sort pending queue by priority and insertion sequence."""
    PENDING.sort(key=_queue_sort_key)


def drain_all_pending() -> list:
    """Drain pending tasks during crash-storm cleanup; caller holds _queue_lock."""
    drained = list(PENDING)
    PENDING.clear()
    persist_queue_snapshot(reason="drain_all_pending")
    return drained


def enqueue_task(task: Dict[str, Any], front: bool = False) -> Dict[str, Any]:
    """Add task to PENDING (thread-safe: HTTP handlers enqueue concurrently
    with the supervisor main loop, so the mutation must hold the queue lock)."""
    t = dict(task)
    attach_task_contract(t)
    with _queue_lock:
        QUEUE_SEQ_COUNTER_REF["value"] += 1
        seq = QUEUE_SEQ_COUNTER_REF["value"]
        t.setdefault("priority", _task_priority(str(t.get("type") or "")))
        _att = t.get("_attempt")
        t.setdefault("_attempt", int(_att) if _att is not None else 1)
        t["_queue_seq"] = -seq if front else seq
        t["queued_at"] = utc_now_iso()
        PENDING.append(t)
        sort_pending()
    return t


def queue_has_task_type(task_type: str) -> bool:
    """Return whether this task type is pending or running."""
    tt = str(task_type or "")
    if any(str(t.get("type") or "") == tt for t in PENDING):
        return True
    for meta in RUNNING.values():
        task = meta.get("task") if isinstance(meta, dict) else None
        if isinstance(task, dict) and str(task.get("type") or "") == tt:
            return True
    return False


def _scheduled_tasks_path(drive_root: pathlib.Path | None = None) -> pathlib.Path:
    return pathlib.Path(drive_root or DRIVE_ROOT) / SCHEDULED_TASKS_FILE


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
    with _queue_lock:
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
    with _queue_lock:
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
    with _queue_lock:
        data = list_scheduled_tasks(drive_root)
        tasks = [item for item in data.get("tasks") or [] if isinstance(item, dict)]
        by_id = {str(item.get("id") or ""): dict(item) for item in tasks}
        touched: list[str] = []
        changed = False
        for skill in skills:
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
                # SSOT: a skill schedule is enabled only when the skill is fully
                # ready to execute (review/grants/deps/enablement), then layered
                # with the schedule-specific supervised_task requirement. This
                # keeps schedule readiness identical to execution readiness.
                try:
                    from ouroboros.skill_readiness import skill_readiness_for_execution

                    schedule_ready = skill_readiness_for_execution(
                        pathlib.Path(drive_root or DRIVE_ROOT), skill
                    ).ready
                except Exception:
                    log.debug(
                        "skill schedule readiness probe failed for %s",
                        getattr(skill, "name", ""),
                        exc_info=True,
                    )
                    schedule_ready = False
                schedule_ready = schedule_ready and (
                    "supervised_task" in set(getattr(manifest, "permissions", []) or [])
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
        # Drop schedules whose source skill/scheduled_task no longer exists
        # (skill deleted, renamed, or scheduled_task removed). Leaving disabled
        # tombstones around would accumulate stale rows in the active table.
        for schedule_id, record in list(by_id.items()):
            if str(record.get("source") or "") == "skill_manifest" and schedule_id not in touched:
                by_id.pop(schedule_id, None)
                changed = True
        if changed:
            data["tasks"] = list(by_id.values())
            _write_scheduled_tasks(data, drive_root)
        return {"changed": changed, "skill_schedule_ids": touched}


def resync_skill_schedules(drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Discover skills and mirror their manifest schedules into the core table.

    Convenience wrapper over ``sync_skill_schedules`` so skill lifecycle paths
    (toggle/grants/reconcile/delete/review/marketplace) reflect payload, grant,
    and enablement changes promptly instead of waiting for the periodic tick.
    """
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_loader import discover_skills

    root = pathlib.Path(drive_root or DRIVE_ROOT)
    return sync_skill_schedules(
        discover_skills(root, repo_path=get_skills_repo_path()),
        drive_root=root,
    )


# Cron/timezone schedule helpers live in supervisor/schedule_time.py (P7
# module-size relief); imported under their historical private names.
from supervisor.schedule_time import (  # noqa: E402
    next_cron_time as _next_cron_time,
    parse_schedule_time as _parse_schedule_time,
    schedule_next_run as _schedule_next_run,
    timezone_for_schedule as _timezone_for_schedule,
)


def _schedule_running_or_queued(schedule_id: str) -> bool:
    if not schedule_id:
        return False
    for task in PENDING:
        meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if str(meta.get("schedule_id") or "") == schedule_id:
            return True
    for meta in RUNNING.values():
        task = meta.get("task") if isinstance(meta, dict) else None
        task_meta = task.get("metadata") if isinstance(task, dict) and isinstance(task.get("metadata"), dict) else {}
        if str(task_meta.get("schedule_id") or "") == schedule_id:
            return True
    return False


def _task_from_schedule(record: Dict[str, Any]) -> Dict[str, Any]:
    template = dict(record.get("task") or {})
    owner_chat_id = load_state().get("owner_chat_id") or 0
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
    with _queue_lock:
        now_monotonic = time.monotonic()
        if now_monotonic - _last_skill_schedule_sync >= _SKILL_SCHEDULE_SYNC_INTERVAL_SEC:
            _last_skill_schedule_sync = now_monotonic
            try:
                resync_skill_schedules(DRIVE_ROOT)
            except Exception:
                log.debug("Failed to sync skill schedules during scheduler tick", exc_info=True)
        data = list_scheduled_tasks()
        changed = False
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
            if trigger_type != "cron":
                record["last_error"] = f"unsupported trigger type: {trigger_type}"
                changed = True
                continue
            expr = str(trigger.get("expr") or record.get("cron") or "").strip()
            if not expr:
                record["last_error"] = "missing cron expression"
                changed = True
                continue
            next_run = _parse_schedule_time(record.get("next_run_at"), tz)
            if next_run is None:
                try:
                    next_run = _next_cron_time(expr, now - datetime.timedelta(minutes=1))
                    record["next_run_at"] = next_run.isoformat()
                    changed = True
                except Exception as exc:
                    record["last_error"] = f"{type(exc).__name__}: {exc}"
                    changed = True
                    continue
            if next_run > now:
                continue
            task = _task_from_schedule(record)
            try:
                from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

                write_task_result(
                    DRIVE_ROOT,
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
            enqueue_task(task)
            record["last_run_at"] = now.isoformat()
            record["last_task_id"] = task["id"]
            record["failure_count"] = int(record.get("failure_count") or 0)
            record["last_error"] = ""
            try:
                record["next_run_at"] = _next_cron_time(expr, now).isoformat()
            except Exception as exc:
                record["last_error"] = f"{type(exc).__name__}: {exc}"
            changed = True
        if changed:
            _write_scheduled_tasks(data)
            persist_queue_snapshot(reason="scheduled_tasks")


def _task_drive_for_task(task: Dict[str, Any], task_id: str) -> pathlib.Path:
    """Active drive of a running task (child drive for forked/workspace tasks,
    canonical otherwise) — where its mailbox and observability actually live.
    Resolution mirrors forward_to_worker: task fields, then the result record."""
    task = task if isinstance(task, dict) else {}
    child = str(task.get("child_drive_root") or task.get("drive_root") or "").strip()
    if not child:
        try:
            from ouroboros.task_results import load_task_result
            record = load_task_result(pathlib.Path(DRIVE_ROOT), str(task_id)) or {}
            child = str(record.get("child_drive_root") or record.get("headless_child_drive_root") or record.get("drive_root") or "").strip()
        except Exception:
            child = ""
    return pathlib.Path(child) if child else pathlib.Path(DRIVE_ROOT)


def _kept_service_pids() -> "set[int]":
    """PIDs of deliberately-kept (session-scope) services to spare from a worker
    tree-kill on cancel/hard-timeout. Best-effort; never raises."""
    try:
        from ouroboros.process_custody import live_kept_service_pids
        return live_kept_service_pids(pathlib.Path(DRIVE_ROOT))
    except Exception:
        return set()


def persist_queue_snapshot(reason: str = "") -> None:
    """Persist queue snapshot for restart/recovery diagnostics.

    Snapshots PENDING/RUNNING under the queue lock: iterating the live dicts
    while HTTP handlers mutate them raised "dictionary changed size during
    iteration" in the supervisor loop (counted toward its crash limit).
    """
    with _queue_lock:
        pending_items = [dict(t) for t in PENDING]
        running_items = [
            (task_id, dict(meta) if isinstance(meta, dict) else {})
            for task_id, meta in RUNNING.items()
        ]
        # Honest worker-pool counts from the ACTUAL pool (not the configured max): the live
        # pool can be smaller (a crash-storm/direct-chat fallback clears WORKERS) and a slot
        # mid-reap is popped from RUNNING but NOT assignable. Surface the real assignable-idle
        # count so the context queue digest never falsely advertises a free worker slot.
        try:
            from supervisor import workers as _workers_mod

            _ws = list(_workers_mod.WORKERS.values())
            worker_total = len(_ws)
            reaping_count = sum(1 for _w in _ws if getattr(_w, "reaping", False))
            assignable_idle_workers = sum(
                1 for _w in _ws
                if getattr(_w, "busy_task_id", None) is None and not getattr(_w, "reaping", False)
            )
        except Exception:
            worker_total = 0
            reaping_count = 0
            assignable_idle_workers = 0
    pending_rows = []
    for t in pending_items:
        pending_rows.append({
            "id": t.get("id"), "type": t.get("type"), "priority": t.get("priority"),
            "attempt": t.get("_attempt"), "queued_at": t.get("queued_at"),
            "queue_seq": t.get("_queue_seq"),
            "task": {
                "id": t.get("id"), "type": t.get("type"), "chat_id": t.get("chat_id"),
                "text": t.get("text"), "priority": t.get("priority"),
                "depth": t.get("depth"), "description": t.get("description"),
                "objective": t.get("objective"), "title": t.get("title"),
                "expected_output": t.get("expected_output"),
                "constraints": t.get("constraints"), "role": t.get("role"),
                "context": t.get("context"), "parent_task_id": t.get("parent_task_id"),
                "root_task_id": t.get("root_task_id"), "session_id": t.get("session_id"),
                "actor_id": t.get("actor_id"), "delegation_role": t.get("delegation_role"),
                "workspace_root": t.get("workspace_root"), "workspace_mode": t.get("workspace_mode"),
                "project_id": t.get("project_id"),
                "allowed_resources": t.get("allowed_resources"), "deadline_at": t.get("deadline_at"),
                "task_contract": t.get("task_contract"),
                "model_lane": t.get("model_lane"),
                "requested_model_lane": t.get("requested_model_lane"),
                "effective_model_lane": t.get("effective_model_lane"),
                "model": t.get("model"),
                "use_local_model": t.get("use_local_model"),
                "task_group_id": t.get("task_group_id"),
                "task_group": t.get("task_group"),
                "subagent_envelope": t.get("subagent_envelope"),
                "memory_mode": t.get("memory_mode"), "drive_root": t.get("drive_root"),
                "child_drive_root": t.get("child_drive_root"),
                "budget_drive_root": t.get("budget_drive_root"),
                "task_constraint": t.get("task_constraint"),
                "metadata": t.get("metadata"),
                "_attempt": t.get("_attempt"), "review_reason": t.get("review_reason"),
                "review_source_task_id": t.get("review_source_task_id"),
            },
        })
    running_rows = []
    now = time.time()
    for task_id, meta in running_items:
        task = meta.get("task") if isinstance(meta, dict) else {}
        started = float(meta.get("started_at") or 0.0) if isinstance(meta, dict) else 0.0
        hb = float(meta.get("last_heartbeat_at") or 0.0) if isinstance(meta, dict) else 0.0
        running_rows.append({
            "id": task_id, "type": task.get("type"), "priority": task.get("priority"),
            "attempt": meta.get("attempt"), "worker_id": meta.get("worker_id"),
            "runtime_sec": round(max(0.0, now - started), 2) if started > 0 else 0.0,
            "heartbeat_lag_sec": round(max(0.0, now - hb), 2) if hb > 0 else None,
            "soft_sent": bool(meta.get("soft_sent")), "task": task,
        })
    payload = {
        "ts": utc_now_iso(),
        "reason": reason,
        "pending_count": len(pending_items), "running_count": len(running_items),
        "reaping_count": reaping_count,
        "worker_total": worker_total,
        "assignable_idle_workers": assignable_idle_workers,
        "pending": pending_rows, "running": running_rows,
    }
    try:
        atomic_write_text(QUEUE_SNAPSHOT_PATH, json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        log.warning("Failed to persist queue snapshot (reason=%s)", reason, exc_info=True)
        pass


def parse_iso_to_ts(iso_ts: str) -> Optional[float]:
    """Parse ISO timestamp to Unix time."""
    txt = str(iso_ts or "").strip()
    if not txt:
        return None
    try:
        return datetime.datetime.fromisoformat(txt.replace("Z", "+00:00")).timestamp()
    except Exception:
        log.debug("Failed to parse ISO timestamp: %s", txt, exc_info=True)
        return None


def restore_pending_from_snapshot(max_age_sec: int = 900) -> int:
    """Restore recent pending tasks from queue snapshot."""
    if PENDING:
        return 0
    try:
        if not QUEUE_SNAPSHOT_PATH.exists():
            return 0
        snap = json.loads(QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if not isinstance(snap, dict):
            return 0
        ts = str(snap.get("ts") or "")
        ts_unix = parse_iso_to_ts(ts)
        if ts_unix is None:
            return 0
        if (time.time() - ts_unix) > max_age_sec:
            return 0
        from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED, load_task_result
        restored = 0
        skipped_terminal = 0
        for row in (snap.get("pending") or []):
            task = row.get("task") if isinstance(row, dict) else None
            if not isinstance(task, dict):
                continue
            chat_id = task.get("chat_id")
            if not task.get("id") or chat_id is None or chat_id == "":
                continue
            # Do not resurrect a task that already reached a terminal/cancelled
            # outcome on disk — restoring it would re-create a "ghost" pending
            # entry that nothing should run.
            try:
                existing = load_task_result(DRIVE_ROOT, str(task.get("id")))
                existing_status = str(existing.get("status") or "") if existing else ""
                # Terminal OR cancel-intent — both must not be resurrected as pending.
                if existing_status in _TRULY_TERMINAL_STATUSES or existing_status == STATUS_CANCEL_REQUESTED:
                    skipped_terminal += 1
                    continue
            except Exception:
                log.debug("Snapshot restore terminal-status check failed for %s", task.get("id"), exc_info=True)
            enqueue_task(task)
            restored += 1
        if restored > 0 or skipped_terminal > 0:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "queue_restored_from_snapshot",
                    "restored_pending": restored,
                    "skipped_terminal": skipped_terminal,
                },
            )
        if restored > 0:
            persist_queue_snapshot(reason="queue_restored")
        return restored
    except Exception:
        log.warning("Failed to restore pending queue from snapshot", exc_info=True)
        return 0


def _emit_cancel_task_done(
    task: Optional[Dict[str, Any]],
    task_id: str,
    *,
    cost_usd: float = 0.0,
    total_rounds: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Emit a task_done event after a cancel so the UI live card resolves.
    Covers both the agent-tool path (_handle_cancel_task) and the HTTP path.
    Cost fields carry reconstructed totals so a cancelled evolution cycle records
    its real spend in the campaign tally instead of zeros."""
    try:
        from supervisor import workers
        chat_id = int((task or {}).get("chat_id") or 0) if isinstance(task, dict) else 0
        if chat_id:
            workers.get_event_q().put({
                "type": "task_done",
                "task_id": str(task_id),
                "task_type": str((task or {}).get("type") or ""),
                "chat_id": chat_id,
                "status": "cancelled",
                "outcome_axes": terminal_outcome_axes(lifecycle="cancelled", execution="cancelled", reason_code="cancelled", review_trigger="supervisor_terminal"),
                "cost_usd": cost_usd,
                "total_rounds": total_rounds,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "metadata": (task or {}).get("metadata") if isinstance((task or {}).get("metadata"), dict) else {},
            })
    except Exception:
        log.debug("Failed to emit task_done for cancelled task %s", task_id, exc_info=True)


def _is_workspace_task_record(record: Dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return bool(str(record.get("workspace_root") or "").strip() or str(metadata.get("workspace_root") or "").strip())


def _cancel_result_fields(
    task: Dict[str, Any] | None,
    *,
    existing: Dict[str, Any] | None = None,
    result: str,
    **fields: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {**fields, "result": result}
    if not (_is_workspace_task_record(task) or _is_workspace_task_record(existing)):
        return payload
    try:
        from ouroboros.headless import ARTIFACT_STATUS_MISSING
        from ouroboros.outcomes import artifact_bundle_from_result

        base: Dict[str, Any] = {}
        if isinstance(existing, dict):
            base.update(existing)
        if isinstance(task, dict):
            base.update(task)
        payload["artifact_status"] = ARTIFACT_STATUS_MISSING
        payload.setdefault("artifact_error", "Task cancelled before workspace patch finalization.")
        base.update(payload)
        base["status"] = "cancelled"
        base["artifact_status"] = ARTIFACT_STATUS_MISSING
        base.pop("artifact_bundle", None)
        bundle = artifact_bundle_from_result(base)
        payload["artifact_bundle"] = bundle
        axes = normalize_outcome_axes(base)
        artifact_axis = dict(axes.get("artifacts") or {})
        artifact_axis["status"] = ARTIFACT_STATUS_MISSING
        axes["artifacts"] = artifact_axis
        payload["outcome_axes"] = axes
    except Exception:
        log.debug("Failed to build cancelled artifact fields for task %s", (task or existing or {}).get("id") or (task or existing or {}).get("task_id"), exc_info=True)
    return payload


def cancel_task_by_id(task_id: str) -> bool:
    """Cancel a pending or running task by id."""
    from supervisor import workers

    with _queue_lock:
        for i, t in enumerate(list(PENDING)):
            if t["id"] == task_id:
                PENDING.pop(i)
                try:
                    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result
                    existing = load_task_result(DRIVE_ROOT, task_id) or {}
                    write_task_result(
                        DRIVE_ROOT, task_id, STATUS_CANCELLED,
                        **_cancel_result_fields(
                            t,
                            existing=existing,
                            result="Task cancelled by user/agent request.",
                        ),
                    )
                except Exception:
                    pass
                _emit_cancel_task_done(t, task_id)
                persist_queue_snapshot(reason="cancel_pending")
                return True

        for w in workers.WORKERS.values():
            if w.busy_task_id == task_id:
                meta = RUNNING.pop(task_id, None) or {}
                task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
                # Reconstruct real cost from durable llm_usage: the worker is about
                # to be killed without finalizing, so the rollup/task_done would
                # otherwise record zeros for a cancelled (e.g. evolution) cycle.
                c_cost, c_rounds, c_prompt, c_completion = reconstruct_task_cost(str(task_id))
                try:
                    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result
                    existing = load_task_result(DRIVE_ROOT, task_id) or {}
                    write_task_result(
                        DRIVE_ROOT, task_id, STATUS_CANCELLED,
                        **_cancel_result_fields(
                            task,
                            existing=existing,
                            cost_usd=c_cost,
                            total_rounds=c_rounds,
                            prompt_tokens=c_prompt,
                            completion_tokens=c_completion,
                            result="Running task cancelled and worker terminated.",
                        ),
                    )
                except Exception:
                    pass
                _emit_cancel_task_done(
                    task, task_id,
                    cost_usd=c_cost, total_rounds=c_rounds,
                    prompt_tokens=c_prompt, completion_tokens=c_completion,
                )
                from ouroboros.platform_layer import kill_pid_tree
                # Tree-kill the worker (a bare terminate() can orphan its
                # foreground subprocess tree), but spare deliberately-kept
                # services: a cancel is neither a session change nor a panic.
                _keep = _kept_service_pids()
                if w.proc.pid:
                    kill_pid_tree(w.proc.pid, exclude_pids=_keep)
                elif w.proc.is_alive():
                    w.proc.terminate()
                w.proc.join(timeout=5)
                if w.proc.is_alive() and w.proc.pid:
                    kill_pid_tree(w.proc.pid, exclude_pids=_keep)
                    w.proc.join(timeout=2)
                try:
                    from ouroboros.tools.services import archive_task_service_logs
                    archive_task_service_logs(pathlib.Path(DRIVE_ROOT), str(task_id), task)
                except Exception:
                    log.debug("Failed to archive service logs for cancelled task %s", task_id, exc_info=True)
                workers.respawn_worker(w.wid)
                # Free a cancelled subagent's child drive now (the worker is dead);
                # otherwise it lingers until the next startup prune + retention.
                if str(task.get("delegation_role") or "") == "subagent":
                    try:
                        from ouroboros.headless import remove_subagent_task_drive
                        remove_subagent_task_drive(DRIVE_ROOT, str(task_id))
                    except Exception:
                        log.debug("Failed to remove cancelled subagent drive for %s", task_id, exc_info=True)
                persist_queue_snapshot(reason="cancel_running")
                return True

        # Cancel arrived after the task already left pending/running (e.g. the
        # worker finished in the window between the cancel_requested latch and
        # this teardown). Finalize a lingering cancel-intent so the task ends as
        # terminal `cancelled`, not stuck forever at `cancel_requested`.
        try:
            from ouroboros.task_results import (
                STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, load_task_result, write_task_result,
            )
            existing = load_task_result(DRIVE_ROOT, task_id) or {}
            if str(existing.get("status") or "") == STATUS_CANCEL_REQUESTED:
                write_task_result(
                    DRIVE_ROOT, task_id, STATUS_CANCELLED,
                    **_cancel_result_fields(
                        existing,
                        existing=existing,
                        result="Task cancelled (finished before supervisor teardown).",
                    ),
                )
                _emit_cancel_task_done(existing, task_id)
                persist_queue_snapshot(reason="cancel_finalize")
                return True
        except Exception:
            log.debug("Cancel finalize-on-miss failed for %s", task_id, exc_info=True)
    return False


def cancel_running_evolution_tasks(reason: str = "evolution stopped") -> List[str]:
    """Cancel any RUNNING evolution task so ``/evolve stop`` ends the live cycle.

    Pending evolution tasks are pruned by the callers; this covers the worker
    that is already mid-cycle. Reuses :func:`cancel_task_by_id`, so the task ends
    as terminal ``cancelled`` (kill_pid_tree, no re-enqueue) and a cancelled
    ``task_done`` resolves the UI card — the normal success finalizer never runs.
    Returns the cancelled task ids.
    """
    cancelled: List[str] = []
    for task_id, meta in list(RUNNING.items()):
        if not isinstance(meta, dict):
            continue
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        if str(task.get("type") or "") != "evolution":
            continue
        try:
            if cancel_task_by_id(task_id):
                cancelled.append(task_id)
        except Exception:
            log.warning(
                "Failed to cancel running evolution task %s (%s)", task_id, reason, exc_info=True
            )
    return cancelled


def enforce_task_timeouts() -> None:
    """Enforce soft/hard timeouts for running tasks.

    Holds the queue lock for the whole pass: RUNNING pops and worker respawn
    decisions raced with HTTP cancel handlers (double respawn → orphaned
    worker; wrong-task dequeue). The RLock keeps nested respawn/assign calls
    re-entrant.
    """
    # Avoid circular dependency during module load.
    from supervisor import workers

    if not RUNNING:
        return
    now = time.time()
    st = load_state()
    owner_chat_id = int(st.get("owner_chat_id") or 0)

    with _queue_lock:
        _enforce_task_timeouts_locked(workers, now, owner_chat_id, st)


def _is_descendant_of(task: Dict[str, Any], ancestor_id: str) -> bool:
    """True if `task` is in the subtree rooted at ancestor_id. Cheap in-memory (no I/O):
    root_task_id == ancestor_id (covers the common root-orchestrator case even when an
    INTERMEDIATE parent has already left RUNNING — a grandchild whose parent finished is
    still a descendant of the root), OR the parent_task_id chain (via RUNNING metas)
    reaches ancestor_id (covers a mid-tree ancestor while the chain is intact).
    """
    if not isinstance(task, dict) or not ancestor_id:
        return False
    if str(task.get("root_task_id") or "") == ancestor_id:
        return True
    cur = task
    hops = 0
    while isinstance(cur, dict) and hops < 25:
        pid = str(cur.get("parent_task_id") or "")
        if not pid:
            return False
        if pid == ancestor_id:
            return True
        nxt = RUNNING.get(pid)
        cur = nxt.get("task") if isinstance(nxt, dict) and isinstance(nxt.get("task"), dict) else None
        hops += 1
    return False


def _subtree_progressing(task_id: str, now: float, idle_timeout: float) -> bool:
    """True if any RUNNING descendant of task_id made real progress within idle_timeout.

    In-memory walk over RUNNING only (NO I/O — this runs under the queue lock): keeps a
    productively-waiting orchestrator alive while its children work, instead of a flat
    wall-clock kill. Descendant freshness uses last_progress_at (real progress), not the
    bare liveness heartbeat.
    """
    if not task_id:
        return False
    for tid, m in list(RUNNING.items()):
        if tid == task_id or not isinstance(m, dict):
            continue
        if not _is_descendant_of(m.get("task") if isinstance(m.get("task"), dict) else {}, task_id):
            continue
        # Real progress only (NOT the bare 30s liveness heartbeat): a child that merely
        # pings but makes no progress must not keep its ancestor alive.
        lp = float(m.get("last_progress_at") or m.get("started_at") or 0.0)
        if lp and (now - lp) < idle_timeout:
            return True
    return False


def _has_live_descendant(task_id: str) -> bool:
    """True if any LIVE (RUNNING or PENDING) task is a descendant of task_id (in-memory, no
    I/O). Used to recognise an orchestrator at kill time so it is NOT blind-retried — a
    blind retry would replay the plan and re-spawn the whole subtree (the timeout storm).
    PENDING is included: a parent can time out while its children are merely QUEUED (worker
    saturation / project lease), and those queued children are still its live subtree.
    """
    if not task_id:
        return False
    for tid, m in list(RUNNING.items()):
        if tid == task_id or not isinstance(m, dict):
            continue
        if _is_descendant_of(m.get("task") if isinstance(m.get("task"), dict) else {}, task_id):
            return True
    for t in list(PENDING):
        if not isinstance(t, dict) or str(t.get("id") or "") == task_id:
            continue
        if _is_descendant_of(t, task_id):
            return True
    return False


def _has_pending_descendant(task_id: str) -> bool:
    """True if any PENDING (queued, not yet assigned) task is a descendant of task_id. A
    parent whose children are merely WAITING for worker capacity (saturation / project lease)
    is not idle/stuck — keep it alive (bounded by the absolute ceiling) so it can integrate
    them once they run, instead of killing it and orphaning the queued subtree."""
    if not task_id:
        return False
    for t in list(PENDING):
        if not isinstance(t, dict) or str(t.get("id") or "") == task_id:
            continue
        if _is_descendant_of(t, task_id):
            return True
    return False


def _enforce_task_timeouts_locked(
    workers: Any, now: float, owner_chat_id: int, st: Dict[str, Any]
) -> None:
    for task_id, meta in list(RUNNING.items()):
        if not isinstance(meta, dict):
            continue
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        started_at = float(meta.get("started_at") or 0.0)
        if started_at <= 0:
            continue
        last_hb = float(meta.get("last_heartbeat_at") or started_at)
        runtime_sec = max(0.0, now - started_at)
        hb_lag_sec = max(0.0, now - last_hb)
        hb_stale = hb_lag_sec >= HEARTBEAT_STALE_SEC
        _wid = meta.get("worker_id")
        worker_id = int(_wid) if _wid is not None else -1
        task_type = str(task.get("type") or "")
        _att = meta.get("attempt")
        if _att is None:
            _att = task.get("_attempt")
        attempt = int(_att) if _att is not None else 1

        effective_soft = 3000 if task_type == "deep_self_review" else SOFT_TIMEOUT_SEC
        deadline_ts = _task_deadline_ts(task)
        deadline_reached = bool(deadline_ts and now >= deadline_ts)

        # Activity-based liveness (owner decision + BIBLE P5): keep a task alive while it
        # makes REAL progress (its own last_progress_at) OR has a progressing subtree —
        # NOT merely while its 30s process heartbeat ticks. The flat BLANKET wall-clock
        # (HARD_TIMEOUT_SEC) is gone so a productively-waiting orchestrator is never killed
        # mid-flight. The HARD (unconditional, activity-independent) axes are: an explicit
        # deadline_at (a deliberate cap — often a caller's timeout_sec, honored promptly even
        # while progressing), the absolute ceiling, and the budget axis (enforced elsewhere).
        # INVARIANT: idle_timeout MUST stay >= the per-call timeout ceiling. A single
        # legitimate tool/LLM call can run up to that ceiling without emitting a
        # between-rounds progress event (heartbeats are NOT progress, by design), so if
        # idle fired below the ceiling a leaf making one long-but-real call would be killed
        # mid-work. Keep this max() coupling on any future change to either knob.
        idle_timeout = max(
            float(get_task_idle_timeout_sec()),
            float(get_per_call_timeout_ceiling_sec()) + 120.0,
        )
        # deep_self_review runs a single long 1M-context LLM call with NO intermediate
        # progress events (no tool loop), so the idle timer governs it from started_at.
        # Preserve its prior ~60min tolerance (the retired effective_hard=3600) so a
        # legitimately long review is not idle-killed mid-call.
        if task_type == "deep_self_review":
            idle_timeout = max(idle_timeout, 3600.0)
        abs_ceiling = float(get_task_abs_ceiling_sec())
        # "Real progress" only — NOT the unconditional 30s liveness heartbeat (which would
        # keep a wedged-before-first-round task alive). A task that has never made real
        # progress is measured from started_at.
        last_progress_at = float(meta.get("last_progress_at") or started_at)
        idle_sec = max(0.0, now - last_progress_at)
        subtree_progressing = _subtree_progressing(task_id, now, idle_timeout)
        # Keep an orchestrator alive while it (a) makes own progress, (b) has a freshly
        # progressing RUNNING descendant, OR (c) has a QUEUED descendant still waiting for a
        # worker — killing it then would orphan the queued subtree. Only the abs ceiling /
        # explicit deadline / budget are unconditional.
        progressing = idle_sec < idle_timeout or subtree_progressing or _has_pending_descendant(task_id)
        ceiling_reached = runtime_sec >= abs_ceiling

        if runtime_sec >= effective_soft and not bool(meta.get("soft_sent")):
            # v6.57.0 — suppress the owner-facing soft heartbeat while a descendant is
            # actively progressing: a parent WAITING on a live child (grandchild busy)
            # is not idle in any way the owner needs to see ("idle=243s. Continuing."
            # while a grandchild generated the deliverable — the site-presentation
            # incident). Crucially `soft_sent` is NOT latched here, so the message can
            # still fire later if the task becomes genuinely idle (a real reason to look).
            # The durable liveness/kill axes are unaffected — this gates only the message.
            if subtree_progressing:
                pass
            else:
                meta["soft_sent"] = True
                if owner_chat_id:
                    send_with_budget(
                        owner_chat_id,
                        f"⏱️ Task {task_id} running for {int(runtime_sec)}s. "
                        f"type={task_type}, heartbeat_lag={int(hb_lag_sec)}s, idle={int(idle_sec)}s. Continuing.",
                    )

        # Hard axes (deadline_at, abs ceiling) stop the task regardless of activity; the
        # idle/subtree gate only spares a task that has NO explicit deadline and is still
        # progressing. This honors an explicit/caller deadline promptly while never letting
        # the removed blanket wall-clock kill a productively-waiting orchestrator.
        if not ceiling_reached and not deadline_reached and progressing:
            continue

        if ceiling_reached:
            terminal_reason = "absolute_ceiling"
        elif deadline_reached:
            terminal_reason = "deadline"
        else:
            terminal_reason = "idle_timeout"
        finalization_requested_at = float(meta.get("finalization_requested_at") or 0.0)
        if finalization_requested_at <= 0 and FINALIZATION_GRACE_SEC > 0:
            meta["finalization_requested_at"] = now
            meta["finalization_reason"] = terminal_reason
            RUNNING[task_id] = meta
            # Typed finalize_now control -> cooperative tool-less final answer
            # inside the grace window. Written to the task's ACTIVE drive (the
            # one the loop drains; child drive for forked/workspace tasks).
            try:
                from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
                write_owner_message(_task_drive_for_task(task, str(task_id)), terminal_reason, str(task_id), kind=KIND_FINALIZE_NOW)
            except Exception:
                log.debug("Failed to write finalize_now control for %s", task_id, exc_info=True)
            if owner_chat_id:
                send_with_budget(
                    owner_chat_id,
                    f"⏳ Task {task_id} reached {terminal_reason}; allowing "
                    f"{FINALIZATION_GRACE_SEC}s finalization grace before hard stop.",
                )
            try:
                from supervisor import workers as _workers_mod
                _workers_mod.get_event_q().put({
                    "type": "send_message",
                    "chat_id": int(task.get("chat_id") or owner_chat_id or 0),
                    "text": (
                        f"⏳ Task {task_id} reached {terminal_reason}. "
                        "Finalize artifacts/results now; supervisor will stop the task after the grace window."
                    ),
                    "format": "markdown",
                    "is_progress": True,
                    "task_id": str(task_id),
                    "ts": utc_now_iso(),
                })
            except Exception:
                log.debug("Failed to emit finalization grace warning for %s", task_id, exc_info=True)
            continue
        if finalization_requested_at > 0 and now - finalization_requested_at < FINALIZATION_GRACE_SEC:
            continue

        # NOTE: the "worker self-finalized at the idle boundary" case is handled by the
        # reaper's POST-KILL terminal re-check (which kills+joins the process FIRST, then
        # honors an on-disk terminal result and emits an idempotent task_done). We do NOT
        # short-circuit here: freeing the slot inline without killing the still-possibly-
        # running process would let assign_tasks reuse it mid-flight and could drop the
        # terminal event, leaving the live card unresolved.

        # Variant A: hand the ENTIRE teardown to the background reaper so the loop tick
        # stays fast AND — critically — the terminal result write + retry enqueue happen
        # only AFTER the reaper has killed/joined the old process (a still-alive worker can
        # no longer race a concurrently-assigned retry; for a subagent the retry reuses the
        # same id/drive). Decisions that need live RUNNING state (orchestrator -> no blind
        # retry; the retry id) are frozen HERE under the lock and passed in the job.
        RUNNING.pop(task_id, None)
        proc_handle = None
        if worker_id in workers.WORKERS:
            w = workers.WORKERS[worker_id]
            if w.busy_task_id == task_id:
                w.busy_task_id = None
            # Mark reaping under the lock so assign_tasks and the crash detector both skip
            # this slot until the reaper installs a fresh worker.
            w.reaping = True
            proc_handle = w.proc

        # NOTE: the "no blind retry of an orchestrator with live descendants" guarantee is
        # TIMEOUT-REAPING-specific (this path). The worker-CRASH path
        # (workers._ensure_workers_healthy_locked) has its own signal-vs-attempt retry
        # semantics and is intentionally not gated here; a crashed-orchestrator storm is a
        # separate, rarer concern than the flat-wall-clock timeout storm this batch targets.
        orchestrator = _has_live_descendant(task_id)
        will_retry = (
            attempt <= QUEUE_MAX_RETRIES
            and isinstance(task, dict)
            and not deadline_reached
            and not ceiling_reached
            and not orchestrator
        )
        # A stopped evolution campaign breaks the auto-retry chain. `st` is the live state
        # loaded this tick, so this reflects the current owner decision.
        if will_retry and task_type == "evolution" and not bool(st.get("evolution_mode_enabled")):
            will_retry = False
        retry_task_id = ""
        if will_retry:
            retry_task_id = task_id if str(task.get("delegation_role") or "") == "subagent" else uuid.uuid4().hex[:8]

        _ensure_reaper_started()
        _reap_queue.put({
            "worker_id": worker_id,
            "proc": proc_handle,
            "task_id": str(task_id),
            "task": task,
            "task_type": task_type,
            "terminal_reason": terminal_reason,
            "attempt": attempt,
            "owner_chat_id": owner_chat_id,
            "runtime_sec": runtime_sec,
            "hb_lag_sec": hb_lag_sec,
            "hb_stale": hb_stale,
            "deadline_reached": deadline_reached,
            "ceiling_reached": ceiling_reached,
            "orchestrator": orchestrator,
            "will_retry": will_retry,
            "retry_task_id": retry_task_id,
        })
        persist_queue_snapshot(reason="task_timeout_reap_queued")




def queue_deep_self_review_task(reason: str, model: str = "", force: bool = False, chat_id: Optional[int] = None) -> Optional[str]:
    """Queue a deep self-review task.

    ``chat_id`` targets a specific chat (e.g. the external transport chat that ran
    ``/review``) so the queued ack and the task results return to the requester
    instead of always defaulting to the web owner's ``owner_chat_id``.
    """
    st = load_state()
    target_chat_id = chat_id if chat_id else st.get("owner_chat_id")
    if not target_chat_id:
        return None
    if (not force) and queue_has_task_type("deep_self_review"):
        return None
    tid = uuid.uuid4().hex[:8]
    enqueue_task({
        "id": tid,
        "type": "deep_self_review",
        "chat_id": int(target_chat_id),
        "text": reason or "Deep self-review",
        "model": model,
    })
    persist_queue_snapshot(reason="deep_self_review_enqueued")
    send_with_budget(int(target_chat_id), f"🔎 Deep self-review queued: {tid} ({reason})")
    return tid


def get_evolution_status_snapshot() -> Dict[str, Any]:
    """Return a non-mutating evolution scheduling snapshot."""
    st = load_state()
    enabled = bool(st.get("evolution_mode_enabled"))
    owner_chat_id = int(st.get("owner_chat_id") or 0)
    consecutive_failures = int(st.get("evolution_consecutive_failures") or 0)
    remaining = round(float(budget_remaining(st)), 2)
    queued_task = next((t for t in PENDING if str(t.get("type") or "") == "evolution"), None)
    running_task = next(
        (
            (meta.get("task") if isinstance(meta, dict) else None)
            for meta in RUNNING.values()
            if isinstance(meta, dict)
            and isinstance(meta.get("task"), dict)
            and str(meta["task"].get("type") or "") == "evolution"
        ),
        None,
    )
    status = "disabled"
    detail = "Evolution mode is off."

    campaign = _read_evolution_campaign()
    active_tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
    restart_blocked = bool(
        active_tx
        and str(active_tx.get("commit_sha") or "").strip()
        and (bool(active_tx.get("restart_required")) or not bool(active_tx.get("restart_verified")))
    )

    if restart_blocked:
        status = "waiting_for_restart_verify"
        detail = "Waiting for restart verification before the next absorbed evolution cycle."
    elif isinstance(running_task, dict):
        status = "running"
        detail = "Evolution task is running now."
    elif isinstance(queued_task, dict):
        status = "queued"
        detail = "Evolution task is queued and waiting for a worker."
    elif consecutive_failures >= 3:
        status = "paused_failures"
        detail = (
            f"Paused after {consecutive_failures} consecutive failures. "
            "Use Evolve again after investigating the failure."
        )
    elif enabled and not owner_chat_id:
        status = "waiting_for_owner_chat"
        detail = "Waiting for the first owner chat binding before scheduling evolution."
    elif enabled and remaining < EVOLUTION_BUDGET_RESERVE:
        status = "budget_blocked"
        detail = (
            f"Budget reserve active: ${remaining:.2f} remaining, "
            f"${EVOLUTION_BUDGET_RESERVE:.0f} reserved for conversations."
        )
    elif enabled and (PENDING or RUNNING):
        status = "waiting_for_idle"
        detail = "Waiting for active tasks to finish before the next evolution cycle."
    elif enabled:
        status = "idle_ready"
        detail = "Idle and ready to queue the next evolution cycle."
    elif remaining < EVOLUTION_BUDGET_RESERVE and str(st.get("last_evolution_task_at") or "").strip():
        status = "budget_stopped"
        detail = (
            f"Evolution auto-stopped because only ${remaining:.2f} remains, "
            f"below the ${EVOLUTION_BUDGET_RESERVE:.0f} conversation reserve."
        )

    return {
        "enabled": enabled,
        "status": status,
        "detail": detail,
        "campaign": campaign,
        "cycle": int(st.get("evolution_cycle") or 0),
        "owner_chat_bound": bool(owner_chat_id),
        "last_task_at": str(st.get("last_evolution_task_at") or ""),
        "consecutive_failures": consecutive_failures,
        # Unbounded budget (supervisor not initialized / TOTAL_BUDGET<=0)
        # is float('inf'), which strict JSON cannot carry — surface None so
        # /api/state stays serializable on onboarding installs.
        "budget_remaining_usd": remaining if math.isfinite(remaining) else None,
        "budget_reserve_usd": float(EVOLUTION_BUDGET_RESERVE),
        "pending_count": len(PENDING),
        "running_count": len(RUNNING),
        "queued_task_id": str((queued_task or {}).get("id") or ""),
        "running_task_id": str((running_task or {}).get("id") or ""),
    }


def _deliver_pending_owner_report() -> None:
    """Deliver a WS-13.5 owner report staged by a worker-side absorb/abandon.

    verify_restart runs in the worker (no live message bus), so it stages
    ``pending_owner_report`` on the campaign; we deliver it here in the SERVER
    process (where the bus is initialized) and clear it. Runs every supervisor
    tick. Best-effort; never raises into the tick.
    """
    try:
        campaign = _read_evolution_campaign()
        report = campaign.get("pending_owner_report")
        if not isinstance(report, dict):
            return
        notify_owner_cycle_outcome(campaign, report)  # reuses the message builder
        campaign.pop("pending_owner_report", None)
        _write_evolution_campaign(campaign)
    except Exception:
        log.debug("failed to deliver pending owner report", exc_info=True)


def enqueue_evolution_task_if_needed() -> None:
    """Queue evolution only when idle, enabled, within budget, and not failure-paused."""
    _deliver_pending_owner_report()
    if PENDING or RUNNING:
        return
    st = load_state()
    if not bool(st.get("evolution_mode_enabled")):
        return
    owner_chat_id = st.get("owner_chat_id")
    if not owner_chat_id:
        return
    campaign = _read_evolution_campaign()
    active_tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
    if (
        active_tx
        and str(active_tx.get("commit_sha") or "").strip()
        and (bool(active_tx.get("restart_required")) or not bool(active_tx.get("restart_verified")))
    ):
        return

    # Defensive net: light mode must never run evolution even if the flag was
    # left enabled (e.g. carried across a restart into light mode). Disable and
    # pause once; entry points already refuse new starts up front.
    from supervisor.state import update_state

    def _disable_evolution(live: Dict[str, Any]) -> None:
        live["evolution_mode_enabled"] = False

    block = evolution_block_reason()
    if block:
        pause_evolution_campaign("blocked in light runtime mode")
        update_state(_disable_evolution)
        send_with_budget(int(owner_chat_id), block)
        return

    consecutive_failures = int(st.get("evolution_consecutive_failures") or 0)
    if consecutive_failures >= 3:
        pause_evolution_campaign("paused after consecutive failures")
        update_state(_disable_evolution)
        send_with_budget(
            int(owner_chat_id),
            f"🧬⚠️ Evolution paused: {consecutive_failures} consecutive failures. "
            f"Use /evolve start to resume after investigating the issue."
        )
        return

    # BUG3: pause if the SAME objective has been re-proposed and no-op'd
    # OBJECTIVE_REPEAT_CAP times without ever absorbing. This is a SEPARATE breaker from
    # consecutive_failures above: that counter is reset to 0 by ANY non-failing cycle
    # (events.py), so it cannot catch a self-maintenance loop where a blocked objective is
    # re-proposed NON-consecutively (interleaved with other no_op work). The per-objective
    # count is keyed on the same canonical fingerprint the transaction stamps, accumulates
    # across non-consecutive recurrence, and is cleared only on a genuine absorb.
    from ouroboros.evolution_fingerprint import canonical_objective_fingerprint

    _objective_repeat_counts = campaign.get("objective_repeat_counts") or {}
    _active_objective_fp = canonical_objective_fingerprint(str(campaign.get("objective") or ""))
    _objective_repeats = int(_objective_repeat_counts.get(_active_objective_fp, 0)) if _active_objective_fp else 0
    if _objective_repeats >= OBJECTIVE_REPEAT_CAP:
        pause_evolution_campaign("paused: objective re-proposed without ever absorbing")
        update_state(_disable_evolution)
        send_with_budget(
            int(owner_chat_id),
            f"🧬⚠️ Evolution paused: the current objective ran {_objective_repeats} reviewed "
            f"cycles WITHOUT ever being absorbed — it keeps getting re-proposed and never lands "
            f"(a self-maintenance loop, not progress). A plain resume won't help; use "
            f"/evolve start with a DIFFERENT objective."
        )
        return

    remaining = budget_remaining(st)
    if remaining < EVOLUTION_BUDGET_RESERVE:
        pause_evolution_campaign("budget reserve reached")
        update_state(_disable_evolution)
        send_with_budget(int(owner_chat_id), f"💸 Evolution stopped: ${remaining:.2f} remaining (reserve ${EVOLUTION_BUDGET_RESERVE:.0f} for conversations).")
        return
    cycle = int(st.get("evolution_cycle") or 0) + 1
    campaign = start_evolution_campaign(source="idle_evolution")
    tid = uuid.uuid4().hex[:8]
    transaction = begin_evolution_transaction(tid, cycle=cycle, campaign=campaign)
    from ouroboros.contracts.task_contract import attach_task_contract

    task = {
        "id": tid, "type": "evolution",
        "chat_id": int(owner_chat_id),
        "text": build_evolution_task_text(cycle),
        "metadata": {"evolution_transaction": transaction},
    }
    attach_task_contract(task)
    enqueue_task(task)

    def _record_cycle(live: Dict[str, Any]) -> None:
        live["evolution_cycle"] = cycle
        live["last_evolution_task_at"] = utc_now_iso()

    update_state(_record_cycle)
    # The generic "Evolution task <id> started." lifecycle message (workers.py)
    # already announces the cycle start, so no extra enqueue bubble here.
