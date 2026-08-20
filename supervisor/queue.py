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
    QUEUE_SNAPSHOT_PATH, budget_remaining, EVOLUTION_BUDGET_RESERVE,
    reconstruct_task_cost as reconstruct_task_cost,
)
from supervisor.message_bus import send_with_budget
from ouroboros.config import (
    DATA_DIR,
    FINALIZATION_GRACE_DEFAULT_SEC,
    get_finalization_grace_sec,
    get_per_call_timeout_ceiling_sec,
    get_task_abs_ceiling_sec,
    get_task_idle_timeout_sec,
)
from ouroboros.contracts.task_contract import attach_task_contract, build_task_contract, normalize_allowed_resources
from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS, schedule_slug
from ouroboros.skill_loader import skill_identity_collision_names
from ouroboros.outcomes import terminal_outcome_axes
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso
from supervisor.evolution_lifecycle import (
    _read_evolution_campaign,
    begin_evolution_transaction,
    build_evolution_task_text,
    disable_evolution_authority,
    disable_evolution_projection,
    deliver_pending_owner_report,
    evolution_block_reason,
    notify_owner_cycle_outcome,
    pause_evolution_campaign,
    start_evolution_campaign,  # noqa: F401 -- historical queue API re-export
)
from supervisor.task_lifecycle import (  # noqa: F401 -- public queue API re-exports
    BUDGET_ROOT_FENCES, apply_budget_root_admission_fence, cancel_task_by_id,
    clear_acceptance_fence_for_root, record_scheduled_admission,
    resume_budget_paused_task, restore_queue_fences, transition_acceptance_fence,
)

log = logging.getLogger(__name__)


DRIVE_ROOT: pathlib.Path = pathlib.Path(DATA_DIR)
SOFT_TIMEOUT_SEC: int = 600
HARD_TIMEOUT_SEC: int = 1800
HEARTBEAT_STALE_SEC: int = 120
QUEUE_MAX_RETRIES: int = 1
FINALIZATION_GRACE_SEC: int = FINALIZATION_GRACE_DEFAULT_SEC
SCHEDULED_TASKS_FILE = pathlib.Path("state") / "scheduled_tasks.json"
# BUG3: pause a campaign whose objective fails to absorb after this many reviewed cycles.
# Mirrors the consecutive-failures threshold; keyed on the objective fingerprint, not failures.
OBJECTIVE_REPEAT_CAP: int = 3
_timeout_deprecation_emitted: bool = False


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
    global DRIVE_ROOT, SOFT_TIMEOUT_SEC, HARD_TIMEOUT_SEC, FINALIZATION_GRACE_SEC, QUEUE_SNAPSHOT_PATH
    DRIVE_ROOT = drive_root
    QUEUE_SNAPSHOT_PATH = drive_root / "state" / "queue_snapshot.json"
    legacy_keys = []
    if int(soft_timeout) != 600:
        legacy_keys.append("OUROBOROS_SOFT_TIMEOUT_SEC")
    if int(hard_timeout) != 1800:
        legacy_keys.append("OUROBOROS_HARD_TIMEOUT_SEC")
    SOFT_TIMEOUT_SEC, HARD_TIMEOUT_SEC = 600, 1800
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec()
    BUDGET_ROOT_FENCES.clear()
    _emit_timeout_deprecation_once(legacy_keys)


def refresh_timeouts_from_settings(settings: dict) -> None:
    """Hot-reload active liveness settings; accept retired keys as typed no-ops."""
    global FINALIZATION_GRACE_SEC
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec(settings)
    legacy_keys = []
    if str(settings.get("OUROBOROS_SOFT_TIMEOUT_SEC", "600")) != "600":
        legacy_keys.append("OUROBOROS_SOFT_TIMEOUT_SEC")
    if str(settings.get("OUROBOROS_HARD_TIMEOUT_SEC", "1800")) != "1800":
        legacy_keys.append("OUROBOROS_HARD_TIMEOUT_SEC")
    _emit_timeout_deprecation_once(legacy_keys)


def _emit_timeout_deprecation_once(keys: List[str]) -> None:
    global _timeout_deprecation_emitted
    if _timeout_deprecation_emitted or not keys:
        return
    _timeout_deprecation_emitted = True
    append_jsonl(
        pathlib.Path(DRIVE_ROOT) / "logs" / "events.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "deprecated_settings_ignored",
            "keys": list(keys),
            "remove_in": "7.0.0",
            "replacement": "current task-liveness and shared planning-cutoff policies",
        },
    )


# Set by workers.init_queue_refs().
PENDING: List[Dict[str, Any]] = []
RUNNING: Dict[str, Dict[str, Any]] = {}
QUEUE_SEQ_COUNTER_REF: Dict[str, int] = {"value": 0}
ACCEPTANCE_FENCES: Dict[str, Dict[str, Any]] = {}
ADMISSION_RESERVATIONS: Dict[str, str] = {}

# Guards PENDING/RUNNING mutations across main loop, direct chat, watchdog.
_queue_lock = threading.RLock()
_last_skill_schedule_sync: float = 0.0
_SKILL_SCHEDULE_SYNC_INTERVAL_SEC: float = 60.0

from supervisor.task_admission import (  # noqa: E402,F401 - public queue API
    release_task_admission,
    reserve_task_admission,
)

# Variant A off-loop worker reaper lives in supervisor/task_reaper.py (module size); re-export
# the thin names the enforce path and tests use — monkeypatching these queue names still works.
from supervisor.task_reaper import (  # noqa: E402,F401 — re-exported for enforce path + tests
    ensure_reaper_started as _ensure_reaper_started,
    reap_queue as _reap_queue,
    reap_timed_out_task as _reap_timed_out_task,
    request_finalization_grace as _request_finalization_grace,
    resolve_grace_episode_for_spared_task as _resolve_grace_episode_for_spared_task,
)


def init_queue_refs(pending: List[Dict[str, Any]], running: Dict[str, Dict[str, Any]],
                    seq_counter_ref: Dict[str, int]) -> None:
    """Bind queue structures owned by workers.py."""
    global PENDING, RUNNING, QUEUE_SEQ_COUNTER_REF
    PENDING = pending
    RUNNING = running
    QUEUE_SEQ_COUNTER_REF = seq_counter_ref
    ADMISSION_RESERVATIONS.clear()


def _task_priority(task_type: str) -> int:
    t = str(task_type or "").strip().lower()
    if t in ("skill_publish", "task", "review", "deep_self_review"):
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


def enqueue_task(
    task: Dict[str, Any], front: bool = False, *, restoring_snapshot: bool = False,
) -> Dict[str, Any]:
    """Add task to PENDING (thread-safe: HTTP handlers enqueue concurrently
    with the supervisor main loop, so the mutation must hold the queue lock)."""
    t = dict(task)
    attach_task_contract(t)
    with _queue_lock:
        require_unique_id = bool(t.pop("_require_unique_task_id", False))
        require_worker_pool = bool(t.pop("_require_worker_pool", False))
        admission_token = str(t.pop("_admission_token", "") or "")
        task_id = str(t.get("id") or "").strip()
        reserved_token = str(ADMISSION_RESERVATIONS.get(task_id) or "")
        if reserved_token and admission_token != reserved_token:
            # A reservation owns this id until its request either enqueues or releases it.
            # Tokenless internal callers and competing ingress must not consume/collide with it.
            t["_admission_blocked"] = "admission_reservation_owned"
            return t
        if require_worker_pool:
            try:
                from supervisor import workers

                disabled_reason = str(workers._WORKER_POOL_DISABLED_REASON or "")
                worker_count = len(workers.WORKERS)
            except Exception:
                disabled_reason = "state_unavailable"
                worker_count = 0
            if disabled_reason or worker_count <= 0:
                if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                    ADMISSION_RESERVATIONS.pop(task_id, None)
                t["_admission_blocked"] = "worker_pool_unavailable"
                t["_worker_pool_disabled_reason"] = disabled_reason or "no_workers"
                return t
        if admission_token and reserved_token != admission_token:
            t["_admission_blocked"] = "admission_reservation_lost"
            return t
        if require_unique_id and task_id:
            live_duplicate = task_id in RUNNING or any(
                isinstance(row, dict) and str(row.get("id") or "") == task_id
                for row in PENDING
            )
            if live_duplicate:
                if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                    ADMISSION_RESERVATIONS.pop(task_id, None)
                t["_admission_blocked"] = "duplicate_task_id"
                return t
            try:
                from ouroboros.task_results import load_task_result

                if load_task_result(DRIVE_ROOT, task_id):
                    if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                        ADMISSION_RESERVATIONS.pop(task_id, None)
                    t["_admission_blocked"] = "duplicate_task_id"
                    return t
            except Exception:
                log.warning("Fresh task-id lookup failed for %s", task_id, exc_info=True)
                t["_admission_blocked"] = "task_id_lookup_failed"
                if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                    ADMISSION_RESERVATIONS.pop(task_id, None)
                return t
        project_id = str(t.get("project_id") or "").strip()
        if project_id:
            try:
                from ouroboros.projects_registry import get_reserved_project

                project = get_reserved_project(DRIVE_ROOT, project_id)
                lifecycle = str((project or {}).get("lifecycle") or "active")
                if project is not None and lifecycle != "active":
                    t["_admission_blocked"] = "project_routing_fence"
                    t["_project_lifecycle"] = lifecycle
                    t["_project_id"] = project_id
                    if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                        ADMISSION_RESERVATIONS.pop(task_id, None)
                    return t
            except Exception:
                log.warning("Project admission check failed for %s", project_id, exc_info=True)
                t["_admission_blocked"] = "project_routing_fence_lookup_failed"
                t["_project_id"] = project_id
                if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                    ADMISSION_RESERVATIONS.pop(task_id, None)
                return t
        root_id = str(t.get("root_task_id") or "").strip()
        if root_id and not restoring_snapshot and apply_budget_root_admission_fence(t, root_id):
            if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                ADMISSION_RESERVATIONS.pop(task_id, None)
            return t
        fence = ACCEPTANCE_FENCES.get(root_id) if root_id else None
        if isinstance(fence, dict) and str(fence.get("status") or "") in {"active", "sealed"}:
            t["_admission_blocked"] = "task_acceptance_fence"
            t["_acceptance_fence_token"] = str(fence.get("token") or "")
            t["_acceptance_fence_status"] = str(fence.get("status") or "active")
            if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                ADMISSION_RESERVATIONS.pop(task_id, None)
            return t
        QUEUE_SEQ_COUNTER_REF["value"] += 1
        seq = QUEUE_SEQ_COUNTER_REF["value"]
        t.setdefault("priority", _task_priority(str(t.get("type") or "")))
        _att = t.get("_attempt")
        t.setdefault("_attempt", int(_att) if _att is not None else 1)
        t["_queue_seq"] = -seq if front else seq
        t["queued_at"] = utc_now_iso()
        if admission_token:
            t["_admission_owner_token"] = admission_token
        PENDING.append(t)
        sort_pending()
        if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
            ADMISSION_RESERVATIONS.pop(task_id, None)
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
                    schedule_ready = skill_readiness_for_execution(pathlib.Path(drive_root or DRIVE_ROOT), skill).ready
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

    root = pathlib.Path(drive_root or DRIVE_ROOT)
    return sync_skill_schedules(
        discover_skills(root, repo_path=get_skills_repo_path()),
        drive_root=root,
    )


# Cron/timezone schedule helpers live in supervisor/schedule_time.py (P7
# module-size relief); imported under their historical private names.
from supervisor.schedule_time import (  # noqa: E402
    next_cron_time as _next_cron_time,
    once_due as _once_due,
    parse_schedule_time as _parse_schedule_time,
    prune_consumed_once_records as _prune_consumed_once, record_last_error as _record_last_error,
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
                    collision_names = skill_identity_collision_names(DRIVE_ROOT)
                if str(record.get("skill") or "") in collision_names:
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
            admitted = enqueue_task(task)
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


def persist_queue_snapshot(reason: str = "") -> bool:
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
        acceptance_fences = [dict(row) for row in ACCEPTANCE_FENCES.values()]
        budget_root_fences = [dict(row) for row in BUDGET_ROOT_FENCES.values()]
        # Honest worker-pool counts from the ACTUAL pool (not the configured max): the live
        # pool can be smaller (a crash-storm/direct-chat fallback clears WORKERS) and a slot
        # mid-reap is popped from RUNNING but NOT assignable. Surface the real assignable-idle
        # count so the context queue digest never falsely advertises a free worker slot.
        try:
            from supervisor import workers as _workers_mod

            _ws = list(_workers_mod.WORKERS.values())
            worker_total = len(_ws)
            worker_pool_disabled_reason = str(
                getattr(_workers_mod, "_WORKER_POOL_DISABLED_REASON", "") or ""
            )
            reaping_count = sum(1 for _w in _ws if getattr(_w, "reaping", False))
            assignable_idle_workers = sum(
                1 for _w in _ws
                if getattr(_w, "busy_task_id", None) is None and not getattr(_w, "reaping", False)
            )
        except Exception:
            worker_total = 0
            worker_pool_disabled_reason = "unknown"
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
                # Scheduling INTENT survives a restart and is all a PENDING child has;
                # `parent_model_lane` and the F9 admission fact `required_model_lane`
                # above all (R2-3). Pinned to SUBAGENT_INTENT_FIELDS by test_model_slot.
                "model_lane": t.get("model_lane"), "parent_model_lane": t.get("parent_model_lane"),
                "requested_model_lane": t.get("requested_model_lane"),
                "required_model_lane": t.get("required_model_lane"), "requested_executor": t.get("requested_executor"),
                "effective_model_lane": t.get("effective_model_lane"),
                "model": t.get("model"), "use_local_model": t.get("use_local_model"),
                "effective_executor": t.get("effective_executor"), "tool_profile": t.get("tool_profile"),
                "executor_route": t.get("executor_route"), "reasoning_effort": t.get("reasoning_effort"),
                "capability_delta": t.get("capability_delta"),
                "task_group_id": t.get("task_group_id"),
                "task_group": t.get("task_group"),
                "subagent_envelope": t.get("subagent_envelope"),
                "memory_mode": t.get("memory_mode"), "drive_root": t.get("drive_root"),
                "child_drive_root": t.get("child_drive_root"),
                "budget_drive_root": t.get("budget_drive_root"),
                "task_constraint": t.get("task_constraint"),
                "metadata": t.get("metadata"), "origin_message_ref": t.get("origin_message_ref"),
                "origin_message_text": t.get("origin_message_text"), "_attempt": t.get("_attempt"),
                "review_reason": t.get("review_reason"), "review_source_task_id": t.get("review_source_task_id"),
                "_budget_pause": t.get("_budget_pause"),
                "budget_resumed_at": t.get("budget_resumed_at"),
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
        "worker_pool_disabled_reason": worker_pool_disabled_reason,
        "assignable_idle_workers": assignable_idle_workers,
        "acceptance_fences": acceptance_fences,
        "budget_root_fences": budget_root_fences,
        "pending": pending_rows, "running": running_rows,
    }
    try:
        atomic_write_text(QUEUE_SNAPSHOT_PATH, json.dumps(payload, ensure_ascii=False, indent=2))
        return True
    except Exception:
        log.warning("Failed to persist queue snapshot (reason=%s)", reason, exc_info=True)
        return False


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
        from ouroboros.task_results import (
            _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED, STATUS_CANCELLED,
            load_task_result, write_task_result,
        )
        raw_fences = snap.get("acceptance_fences", [])
        raw_budget_fences = snap.get("budget_root_fences", [])
        snapshot_pending = [
            row.get("task")
            for row in (snap.get("pending") or [])
            if isinstance(row, dict) and isinstance(row.get("task"), dict)
        ]
        fenced_roots, malformed_fences, malformed_budget_fences = restore_queue_fences(raw_fences, raw_budget_fences)
        if malformed_budget_fences:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "queue_restore_invalid_budget_root_fences",
                 "action": "fail_closed_no_restore"},
            )
            return 0
        if malformed_fences:
            affected = [str(task.get("id") or "") for task in snapshot_pending if task.get("id")]
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "queue_restore_invalid_acceptance_fences",
                    "affected_task_ids": affected,
                    "action": "fail_closed_no_restore",
                },
            )
            try:
                for task in snapshot_pending:
                    task_id = str(task.get("id") or "")
                    if task_id:
                        existing = load_task_result(DRIVE_ROOT, task_id) or {}
                        write_task_result(
                            DRIVE_ROOT,
                            task_id,
                            STATUS_CANCELLED,
                            **_cancel_result_fields(
                                task,
                                existing=existing,
                                result="Task was not restored because its acceptance-fence snapshot was invalid.",
                            ),
                        )
            except Exception:
                log.warning("Failed to terminalize tasks from invalid acceptance-fence snapshot", exc_info=True)
            return 0

        pending_by_id = {
            str(task.get("id") or ""): task for task in snapshot_pending if str(task.get("id") or "")
        }
        restored = 0
        skipped_terminal = 0
        skipped_fenced: list[str] = []
        blocked_restore: list[str] = []
        for task in snapshot_pending:
            chat_id = task.get("chat_id")
            if not task.get("id") or chat_id is None or chat_id == "":
                continue
            fenced = False
            for fenced_root in fenced_roots:
                if str(task.get("root_task_id") or "") == fenced_root:
                    fenced = True
                    break
                current = task
                seen: set[str] = set()
                while isinstance(current, dict):
                    parent_id = str(current.get("parent_task_id") or "")
                    if not parent_id or parent_id in seen:
                        break
                    if parent_id == fenced_root:
                        fenced = True
                        break
                    seen.add(parent_id)
                    current = pending_by_id.get(parent_id)
                if fenced:
                    break
            if fenced:
                task_id = str(task.get("id") or "")
                skipped_fenced.append(task_id)
                try:
                    existing = load_task_result(DRIVE_ROOT, task_id) or {}
                    write_task_result(
                        DRIVE_ROOT,
                        task_id,
                        STATUS_CANCELLED,
                        **_cancel_result_fields(
                            task,
                            existing=existing,
                            result="Task was not restored after restart because its root had entered acceptance review.",
                        ),
                    )
                except Exception:
                    log.warning("Failed to terminalize fenced snapshot task %s", task_id, exc_info=True)
                continue
            # Never resurrect a terminal/cancelled task as a ghost pending entry.
            # AR2-10 (§8-A1): the intent projection is consulted UNDER the queue lock at
            # restore — the "no active intent" read and the enqueue form one serialized step
            # against assignment/drop (same invariant as the pre-assignment consult). Boot-time
            # and contention-free; _queue_lock is an RLock, so enqueue_task stays re-entrant.
            with _queue_lock:
                skip_revival = False
                try:
                    existing = load_task_result(DRIVE_ROOT, str(task.get("id")))
                    existing_status = str(existing.get("status") or "") if existing else ""
                    # Terminal OR cancel-intent — both must not be resurrected as
                    # pending. Intent lives in the durable projection (phase A);
                    # the status check covers legacy latch files.
                    if existing_status in _TRULY_TERMINAL_STATUSES or existing_status == STATUS_CANCEL_REQUESTED:
                        skip_revival = True
                    else:
                        from ouroboros.cancel_intents import has_active_intent

                        if has_active_intent(DRIVE_ROOT, str(task.get("id"))):
                            # Left for cancellation custody/watchdog to settle —
                            # never a pending revival racing its own teardown.
                            skip_revival = True
                except Exception:
                    log.debug("Snapshot restore terminal-status check failed for %s", task.get("id"), exc_info=True)
                if skip_revival:
                    skipped_terminal += 1
                    continue
                # These tasks already existed when the root pause was snapshotted.
                # Restore them behind the root marker; only new admission is fenced.
                admitted = enqueue_task(task, restoring_snapshot=True)
            if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
                blocked_restore.append(str(task.get("id") or ""))
                continue
            restored += 1
        if skipped_fenced:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "queue_restore_skipped_acceptance_fence",
                    "task_ids": skipped_fenced,
                    "root_task_ids": sorted(fenced_roots),
                },
            )
        if restored > 0 or skipped_terminal > 0 or blocked_restore:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "queue_restored_from_snapshot",
                    "restored_pending": restored,
                    "skipped_terminal": skipped_terminal,
                    "blocked_admission": blocked_restore,
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
    cost_fields: Optional[Dict[str, Any]] = None,
    status: str = "cancelled",
) -> None:
    """Emit a task_done event after a cancel so the UI live card resolves.
    Covers both the agent-tool path (_handle_cancel_task) and the HTTP path.
    ``status`` carries the STORED terminal truth: when a worker wrote its own
    natural result just before the kill, the card must resolve to THAT outcome
    rather than be left unresolved until a reload.
    ``cost_fields`` is the caller's accounting authority — a reconstructed
    ledger projection or a CONFIRMED pre-start zero. An absent projection emits
    an honest nullable unknown; the old default fabricated a final $0 for every
    cancel (Poltergeist A1.10, owner 10=B)."""
    try:
        from supervisor import workers
        chat_id = int((task or {}).get("chat_id") or 0) if isinstance(task, dict) else 0
        workers.get_event_q().put({
                "type": "task_done",
                "task_id": str(task_id),
                "task_type": str((task or {}).get("type") or ""),
                "chat_id": chat_id,
                "status": status,
                "outcome_axes": terminal_outcome_axes(
                    lifecycle=status, execution=status, reason_code=status,
                    review_trigger="supervisor_terminal",
                ),
                **(cost_fields or {
                    "cost_accounting_status": "unavailable", "cost_final": False,
                    "cost_usd": None,
                }),
                "metadata": (task or {}).get("metadata") if isinstance((task or {}).get("metadata"), dict) else {},
        })
    except Exception:
        log.debug("Failed to emit task_done for cancelled task %s", task_id, exc_info=True)


# Cancellation custody and the terminal-cancel result-field builder live in
# supervisor.task_lifecycle (module-size boundary); re-exported so
# `supervisor.queue` stays the single import surface for callers.
from supervisor.task_lifecycle import (  # noqa: E402, F401 -- intentional public re-exports
    CANCEL_ALREADY_SETTLED,
    CANCEL_CANCELLED,
    CANCEL_FAILED,
    CANCEL_NOT_FOUND,
    _CANCEL_TERMINALIZED,
    _cancel_result_fields,
    cancel_task_custody,
    task_has_live_ownership,
    task_subtree_is_live,
)
from supervisor.queue_transitions import (  # noqa: E402, F401 -- intentional public re-exports
    evolution_stop_report,
    stop_evolution_tasks,
)


def _cancel_task_by_id_single(task_id: str) -> bool:
    """Boolean facade for the pre-v6.82 single-task callers."""
    return cancel_task_custody(task_id) in {CANCEL_CANCELLED, CANCEL_ALREADY_SETTLED}


# Evolution-stop transitions (GR2-13) live in supervisor.queue_transitions
# (module-size boundary); re-exported below with the other transition helpers
# so `supervisor.queue` stays the single import surface for callers.


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
    # ONE typed owner-stop predicate before every generic timeout-grace consumer (S3
    # §12.2 item 8): a task whose owner-requested finalization intent is still OPEN is
    # bypassed whole — no spare-withdraw, no spare-clock reset, no second grace episode,
    # no expiry kill, no RUNNING.pop, no reaper enqueue, no retry scheduling. The hold
    # deliberately outlives the grace deadline (the expiry window): the deadline gates only
    # the sweep's arm-vs-feed-custody decision in supervisor/owner_stop.py +
    # sweep_cancel_intents; the intent stays the one owner will and custody stays the only killer.
    from supervisor.owner_stop import running_owner_stop_tasks

    owner_stop_held = running_owner_stop_tasks(
        DRIVE_ROOT, grace_sec=FINALIZATION_GRACE_SEC,
    )
    for task_id, meta in list(RUNNING.items()):
        if not isinstance(meta, dict):
            continue
        if str(task_id) in owner_stop_held:
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

        deadline_ts = _task_deadline_ts(task)
        deadline_reached = bool(deadline_ts and now >= deadline_ts)

        idle_timeout = max(
            float(get_task_idle_timeout_sec()),
            float(get_per_call_timeout_ceiling_sec()) + 120.0,
        )
        # deep_self_review runs a single long 1M-context LLM call with NO intermediate
        # progress events (no tool loop), so the idle timer governs it from started_at;
        # its prior ~60min tolerance is preserved so it is not idle-killed mid-call.
        if task_type == "deep_self_review":
            idle_timeout = max(idle_timeout, 3600.0)
        abs_ceiling = float(get_task_abs_ceiling_sec())
        last_progress_at = float(meta.get("last_progress_at") or started_at)
        idle_sec = max(0.0, now - last_progress_at)
        subtree_progressing = _subtree_progressing(task_id, now, idle_timeout)
        own_progress = idle_sec < idle_timeout
        # B3 external-wait lease: a held delegate_wait window over a live delegated run
        # is legitimate silence (hard-bounded by events._handle_external_wait_lease);
        # it spares ONLY this idle rail — ceiling/deadline/budget/cancel never consult it.
        lease_ts = meta.get("external_wait_lease_until")
        # Keep an orchestrator alive on own progress, a freshly progressing RUNNING descendant,
        # a QUEUED descendant (a kill would orphan the queued subtree), or a live external-wait
        # lease; only abs ceiling / explicit deadline / budget are unconditional.
        progressing = (own_progress or subtree_progressing or _has_pending_descendant(task_id)
                       or (isinstance(lease_ts, (int, float)) and float(lease_ts) > now))
        ceiling_reached = runtime_sec >= abs_ceiling

        # Hard axes (deadline_at, abs ceiling) stop the task regardless of activity; the
        # idle/subtree gate only spares a still-progressing task with NO explicit deadline —
        # an explicit/caller deadline is honored promptly, while no blanket wall-clock kills
        # a productively-waiting orchestrator.
        if not ceiling_reached and not deadline_reached and progressing:
            # An outstanding episode outlives this reprieve or is withdrawn by it; the rule
            # (own progress answers the request, sparing only suspends its clock) lives with
            # the rest of the episode mechanics in task_reaper. The latch is checked here so
            # the drive resolution (which may read the result record) stays off the no-episode path.
            if meta.get("finalization_requested_at") and _resolve_grace_episode_for_spared_task(
                _task_drive_for_task(task, str(task_id)), str(task_id), meta,
                chat_id=int(task.get("chat_id") or owner_chat_id or 0),
                own_progress=own_progress, now=now,
            ):
                RUNNING[task_id] = meta
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
            # The control's msg_id IS the episode's identity: it is what the
            # symmetric withdraw revokes, so the latch and the mailbox control
            # can never name different episodes.
            meta["finalization_control_msg_id"] = _request_finalization_grace(
                _task_drive_for_task(task, str(task_id)), str(task_id), terminal_reason,
                chat_id=int(task.get("chat_id") or owner_chat_id or 0),
                stamp=int(now),
            )
            RUNNING[task_id] = meta
            continue
        if finalization_requested_at > 0 and now - finalization_requested_at < FINALIZATION_GRACE_SEC:
            continue

        # NOTE: "worker self-finalized at the idle boundary" is handled by the reaper's
        # POST-KILL terminal re-check (kill+join FIRST, then honor an on-disk terminal
        # result, idempotent task_done). No short-circuit here: freeing the slot inline
        # would let assign_tasks reuse it mid-flight and could drop the terminal event.

        # Variant A: hand the ENTIRE teardown to the background reaper so the loop tick
        # stays fast and the terminal write + retry enqueue happen only AFTER kill/join
        # (no race with a concurrently-assigned retry; a subagent retry reuses id/drive).
        # Live-RUNNING decisions (orchestrator -> no blind retry; retry id) freeze HERE.
        if task_type == "evolution":
            from supervisor.evolution_lifecycle import update_evolution_transaction
            if not update_evolution_transaction(task_id, dispatch_status="reaping"):
                log.warning("Evolution timeout teardown deferred: reaping state was not durable for %s", task_id)
                continue
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
        # An ACTIVE cancel intent (immediate policy, or a finalize intent already
        # CLAIMED by custody — open finalize intents never reach here, the hold
        # above skips them) must never spawn a retry clone: a new-uuid retry
        # escapes the intent (keyed by the old id) and CANCELLED_ROOT_FENCES,
        # restarting work the owner stopped.
        if will_retry:
            from ouroboros.cancel_intents import has_active_intent

            will_retry = not has_active_intent(DRIVE_ROOT, str(task_id))
        retry_task_id = ""
        if will_retry:
            same_id = task_type == "evolution" or str(task.get("delegation_role") or "") == "subagent"
            retry_task_id = task_id if same_id else uuid.uuid4().hex[:8]

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
            "incident_toast_once": f"{task_id}:{terminal_reason}:{int(finalization_requested_at or now)}",
        })
        persist_queue_snapshot(reason="task_timeout_reap_queued")


def queue_deep_self_review_task(reason: str, model: str = "", force: bool = False, chat_id: Optional[int] = None) -> Optional[str]:
    """Queue a deep self-review task.

    ``chat_id`` targets a specific chat (e.g. the external transport chat that ran
    ``/review``) so the queued ack and the task results return to the requester
    instead of always defaulting to the web owner's ``owner_chat_id``.
    """
    target_chat_id = chat_id if chat_id else load_state().get("owner_chat_id")
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


def get_evolution_status_snapshot(*, budget_projection: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a non-mutating evolution scheduling snapshot.

    ``budget_projection``: optional pre-computed global usage projection from a
    caller that already replayed the ledger this request (``/api/state``), so the
    snapshot does not replay it again. Default ``None`` keeps the self-computing,
    strict fail-closed behavior — a caller whose own computation FAILED must pass
    nothing, so the paused-evolution disclosure still comes from this snapshot.
    """
    st = load_state()
    enabled = bool(st.get("evolution_mode_enabled"))
    owner_chat_id = int(st.get("owner_chat_id") or 0)
    consecutive_failures = int(st.get("evolution_consecutive_failures") or 0)
    try:
        remaining: Optional[float] = round(float(budget_remaining(st, strict=True, projection=budget_projection)), 2)
        accounting_available = True
    except Exception:
        remaining = None
        accounting_available = False
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
    elif not accounting_available:
        status = "accounting_unavailable"
        detail = "Cost accounting is unavailable; evolution dispatch is paused without changing the campaign."
    elif consecutive_failures >= 3:
        status = "paused_failures"
        detail = (
            f"Paused after {consecutive_failures} consecutive failures. "
            "Use Evolve again after investigating the failure."
        )
    elif enabled and not owner_chat_id:
        status = "waiting_for_owner_chat"
        detail = "Waiting for the first owner chat binding before scheduling evolution."
    elif enabled and remaining is not None and remaining < EVOLUTION_BUDGET_RESERVE:
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
    elif remaining is not None and remaining < EVOLUTION_BUDGET_RESERVE and str(st.get("last_evolution_task_at") or "").strip():
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
        "cost_accounting_status": "available" if accounting_available else "unavailable",
        # Unbounded budget (supervisor not initialized / TOTAL_BUDGET<=0)
        # is float('inf'), which strict JSON cannot carry — surface None so
        # /api/state stays serializable on onboarding installs.
        "budget_remaining_usd": remaining if remaining is not None and math.isfinite(remaining) else None,
        "budget_reserve_usd": float(EVOLUTION_BUDGET_RESERVE),
        "pending_count": len(PENDING),
        "running_count": len(RUNNING),
        "queued_task_id": str((queued_task or {}).get("id") or ""),
        "running_task_id": str((running_task or {}).get("id") or ""),
    }


def _deliver_pending_owner_report() -> None:
    deliver_pending_owner_report(notify_owner_cycle_outcome)


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
    from supervisor.state import update_state
    has_authority = all(str(campaign.get(key) or "").strip() for key in ("id", "source"))
    if campaign.get("status") != "active" or not has_authority:
        disable_evolution_authority("bare_flag_disabled", campaign_id=str(campaign.get("id") or ""))
        send_with_budget(
            int(owner_chat_id),
            "🧬 Evolution stayed off: the enable flag had no active campaign authority. Use /evolve start to begin a fresh campaign.",
        )
        return
    active_tx = campaign.get("active_transaction") if isinstance(campaign.get("active_transaction"), dict) else {}
    if active_tx and (
        str(active_tx.get("commit_sha") or "").strip()
        or str(active_tx.get("dispatch_status") or "") == "reaping"
    ):
        return

    # Defensive net: light mode must never run evolution even if the flag was
    # left enabled (e.g. carried across a restart into light mode). Disable and
    # pause once; entry points already refuse new starts up front.
    block = evolution_block_reason()
    if block:
        pause_evolution_campaign("blocked in light runtime mode")
        disable_evolution_projection()
        send_with_budget(int(owner_chat_id), block)
        return

    consecutive_failures = int(st.get("evolution_consecutive_failures") or 0)
    if consecutive_failures >= 3:
        pause_evolution_campaign("paused after consecutive failures")
        disable_evolution_projection()
        send_with_budget(
            int(owner_chat_id),
            f"🧬⚠️ Evolution paused: {consecutive_failures} consecutive failures. "
            f"Use /evolve start to resume after investigating the issue."
        )
        return

    # BUG3: pause if the SAME objective has been re-proposed and no-op'd OBJECTIVE_REPEAT_CAP
    # times without ever absorbing. This is a SEPARATE breaker from consecutive_failures
    # above: that counter is reset to 0 by ANY non-failing cycle (events.py), so it cannot
    # catch a self-maintenance loop where a blocked objective is re-proposed NON-consecutively
    # (interleaved with other no_op work). The per-objective count is keyed on the same
    # canonical fingerprint the transaction stamps, accumulates across non-consecutive
    # recurrence, and is cleared only on a genuine absorb.
    from ouroboros.evolution_fingerprint import canonical_objective_fingerprint

    _objective_repeat_counts = campaign.get("objective_repeat_counts") or {}
    _active_objective_fp = canonical_objective_fingerprint(str(campaign.get("objective") or ""))
    _objective_repeats = int(_objective_repeat_counts.get(_active_objective_fp, 0)) if _active_objective_fp else 0
    if _objective_repeats >= OBJECTIVE_REPEAT_CAP:
        pause_evolution_campaign("paused: objective re-proposed without ever absorbing")
        disable_evolution_projection()
        send_with_budget(
            int(owner_chat_id),
            f"🧬⚠️ Evolution paused: the current objective ran {_objective_repeats} reviewed "
            f"cycles WITHOUT ever being absorbed — it keeps getting re-proposed and never lands "
            f"(a self-maintenance loop, not progress). A plain resume won't help; use "
            f"/evolve start with a DIFFERENT objective."
        )
        return

    try:
        remaining = budget_remaining(st, strict=True)
    except Exception:
        log.error("Evolution scheduling deferred: cost accounting unavailable", exc_info=True)
        append_jsonl(DRIVE_ROOT / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "evolution_accounting_unavailable",
            "action": "dispatch_deferred", "owner_visible": True,
        })
        return
    if remaining < EVOLUTION_BUDGET_RESERVE:
        pause_evolution_campaign("budget reserve reached")
        disable_evolution_projection()
        send_with_budget(int(owner_chat_id), f"💸 Evolution stopped: ${remaining:.2f} remaining (reserve ${EVOLUTION_BUDGET_RESERVE:.0f} for conversations).")
        return
    cycle = int(st.get("evolution_cycle") or 0) + 1
    tid = uuid.uuid4().hex[:8]
    transaction = begin_evolution_transaction(tid, cycle=cycle, campaign=campaign)
    if not transaction:
        disable_evolution_authority("transaction_attach_failed", campaign_id=str(campaign.get("id") or ""), task_id=tid)
        send_with_budget(
            int(owner_chat_id),
            "🧬 Evolution stayed off: the campaign changed before its next task could be attached. Start it again when ready.",
        )
        return
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
