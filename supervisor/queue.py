"""Supervisor task queue, persistence, timeouts, and evolution scheduling."""

from __future__ import annotations

import datetime  # noqa: F401
import json  # noqa: F401
import logging
import math  # noqa: F401
import os
import pathlib
import queue as _stdqueue  # noqa: F401 — re-exported for the test suite's reap-queue isolation
import threading
import time  # noqa: F401
import uuid  # noqa: F401
from typing import Any, Dict, List, Optional, Tuple

from supervisor.state import (
    load_state, append_jsonl, atomic_write_text,  # noqa: F401
    budget_remaining, EVOLUTION_BUDGET_RESERVE,  # noqa: F401
    reconstruct_task_cost as reconstruct_task_cost,
)
from supervisor.message_bus import send_with_budget  # noqa: F401
from ouroboros.config import (
    DATA_DIR,
    FINALIZATION_GRACE_DEFAULT_SEC,
    get_finalization_grace_sec,
    get_per_call_timeout_ceiling_sec,  # noqa: F401
    get_task_abs_ceiling_sec,  # noqa: F401
    get_task_idle_timeout_sec,  # noqa: F401
)
from ouroboros.contracts.task_contract import attach_task_contract, build_task_contract, normalize_allowed_resources  # noqa: F401
from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS, schedule_slug  # noqa: F401
from ouroboros.skill_loader import skill_identity_collision_names  # noqa: F401
from ouroboros.outcomes import terminal_outcome_axes
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso  # noqa: F401
from supervisor.evolution_lifecycle import (
    _read_evolution_campaign,  # noqa: F401
    begin_evolution_transaction,  # noqa: F401
    build_evolution_task_text,  # noqa: F401
    disable_evolution_authority,  # noqa: F401
    disable_evolution_projection,  # noqa: F401
    deliver_pending_owner_report,  # noqa: F401
    evolution_block_reason,  # noqa: F401
    notify_owner_cycle_outcome,  # noqa: F401
    pause_evolution_campaign,  # noqa: F401
    start_evolution_campaign,  # noqa: F401 -- historical queue API re-export
)
from supervisor.task_lifecycle import (  # noqa: F401 -- public queue API re-exports
    BUDGET_ROOT_FENCES, apply_budget_root_admission_fence, cancel_task_by_id,
    clear_acceptance_fence_for_root, record_scheduled_admission,
    resume_budget_paused_task, restore_queue_fences, transition_acceptance_fence,
)

log = logging.getLogger(__name__)

# Queue responsibilities owned by their own modules (module-size boundary): the
# durable snapshot, the liveness rails, recurring schedules, and the evolution
# cycle's admission. Each reads the queue's rebound state through a handle back
# to this module — see the handle docstring in any of them — and each is
# re-imported here so `supervisor.queue` stays the single public import surface.
from supervisor.queue_snapshot import (  # noqa: F401 -- supervisor/queue.py facade re-exports
    _kept_service_pids,
    parse_iso_to_ts,
    persist_queue_snapshot,
    restore_pending_from_snapshot,
)
from supervisor.queue_timeouts import (  # noqa: F401 -- supervisor/queue.py facade re-exports
    _enforce_task_timeouts_locked,
    _has_live_descendant,
    _has_pending_descendant,
    _is_descendant_of,
    _subtree_progressing,
    _task_deadline_ts,
    _task_drive_for_task,
    enforce_task_timeouts,
)
from supervisor.queue_schedules import (  # noqa: F401 -- supervisor/queue.py facade re-exports
    _schedule_running_or_queued,
    _scheduled_tasks_path,
    _task_from_schedule,
    _write_scheduled_tasks,
    check_scheduled_tasks,
    list_scheduled_tasks,
    remove_scheduled_task,
    resync_skill_schedules,
    sync_skill_schedules,
    upsert_scheduled_task,
)
from supervisor.queue_evolution import (  # noqa: F401 -- supervisor/queue.py facade re-exports
    _deliver_pending_owner_report,
    enqueue_evolution_task_if_needed,
    get_evolution_status_snapshot,
    queue_deep_self_review_task,
)


DRIVE_ROOT: pathlib.Path = pathlib.Path(DATA_DIR)
HEARTBEAT_STALE_SEC: int = 120
QUEUE_MAX_RETRIES: int = 1
FINALIZATION_GRACE_SEC: int = FINALIZATION_GRACE_DEFAULT_SEC
SCHEDULED_TASKS_FILE = pathlib.Path("state") / "scheduled_tasks.json"
# BUG3: pause a campaign whose objective fails to absorb after this many reviewed cycles.
# Mirrors the consecutive-failures threshold; keyed on the objective fingerprint, not failures.
OBJECTIVE_REPEAT_CAP: int = 3
_timeout_deprecation_emitted: bool = False


# The three retired liveness keys and the default each one announced. A settings
# document no longer carries them — `load_settings` drops every RETIRED_SETTING_KEY
# before any reader sees it — so an environment variable is the only way a
# non-default value can still exist, and the only place worth looking.
RETIRED_LIVENESS_ENV_DEFAULTS = (
    ("OUROBOROS_SOFT_TIMEOUT_SEC", "600"),
    ("OUROBOROS_HARD_TIMEOUT_SEC", "1800"),
    ("OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC", "120"),
)


def init(drive_root: pathlib.Path) -> None:
    """Bind the queue to its drive and read the live liveness settings.

    The three retired timeout keys are no longer parameters: nothing passed one
    that any rail read, and the deprecation notice they exist to raise is a fact
    about the ENVIRONMENT, which this function can read for itself.
    """
    global DRIVE_ROOT, FINALIZATION_GRACE_SEC
    DRIVE_ROOT = drive_root
    legacy_keys = [
        key for key, default in RETIRED_LIVENESS_ENV_DEFAULTS
        if str(os.environ.get(key, default)) != default
    ]
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec()
    BUDGET_ROOT_FENCES.clear()
    _emit_timeout_deprecation_once(legacy_keys)


def refresh_timeouts_from_settings(settings: dict) -> None:
    """Hot-reload the one liveness setting a reload can change.

    The retired keys are NOT probed here: `load_settings` removes them from the
    document, so a reader that looked for them would be asking a question the
    settings surface can no longer answer either way. Their one surviving source
    is the environment, which a reload does not change and `init` already read.
    """
    global FINALIZATION_GRACE_SEC
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec(settings)


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


# Cron/timezone schedule helpers live in supervisor/schedule_time.py (P7
# module-size relief); imported under their historical private names.
from supervisor.schedule_time import (  # noqa: E402
    next_cron_time as _next_cron_time,  # noqa: F401
    parse_schedule_time as _parse_schedule_time,  # noqa: F401
    schedule_next_run as _schedule_next_run,  # noqa: F401
    timezone_for_schedule as _timezone_for_schedule,  # noqa: F401
)


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
