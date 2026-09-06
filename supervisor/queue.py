"""Supervisor task queue, persistence, timeouts, and evolution scheduling."""

from __future__ import annotations

import logging
import math
import pathlib
import queue as _stdqueue  # noqa: F401 — re-exported for the test suite's reap-queue isolation
import threading
import time  # noqa: F401 -- facade name tests read (queue.time)
import uuid
from typing import Any, Dict, List, Optional, Tuple

from supervisor.state import (
    load_state,
    append_jsonl,  # noqa: F401 -- queue_snapshot leaf reads it via the _queue() handle
    atomic_write_text,  # noqa: F401 -- queue_snapshot leaf reads it via the _queue() handle
    budget_remaining, EVOLUTION_BUDGET_RESERVE,
    reconstruct_task_cost as reconstruct_task_cost,
)
from supervisor.message_bus import (
    coerce_chat_identity,  # noqa: F401 -- queue_timeouts leaf reads it via the _queue() handle
    notification_chat_route,
    send_with_budget,
)
from ouroboros.config import (
    DATA_DIR,
    FINALIZATION_GRACE_DEFAULT_SEC,
    get_finalization_grace_sec,
    get_per_call_timeout_ceiling_sec,  # noqa: F401 -- queue_timeouts leaf reads it via the _queue() handle
    get_task_abs_ceiling_sec,  # noqa: F401 -- queue_timeouts leaf reads it via the _queue() handle
    get_task_idle_timeout_sec,  # noqa: F401 -- queue_timeouts leaf reads it via the _queue() handle
)
from ouroboros.contracts.task_contract import attach_task_contract, build_task_contract, normalize_allowed_resources  # noqa: F401
from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS, schedule_slug  # noqa: F401
from ouroboros.skill_loader import skill_identity_collision_names  # noqa: F401
from ouroboros.outcomes import terminal_outcome_axes
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso  # noqa: F401
from supervisor.evolution_lifecycle import (  # noqa: F401 -- public queue API and lazy scheduler dependencies
    _deliver_pending_owner_report,
    _read_evolution_campaign,
    begin_evolution_transaction,
    build_evolution_task_text,
    disable_evolution_authority,
    disable_evolution_projection,
    deliver_pending_owner_report,
    enqueue_evolution_task_if_needed,
    evolution_block_reason,
    notify_owner_cycle_outcome,
    pause_evolution_campaign,
    start_evolution_campaign,  # noqa: F401 -- historical queue API re-export
)
from supervisor.task_lifecycle import (  # noqa: F401 -- public queue API re-exports
    BUDGET_ROOT_FENCES, apply_budget_root_admission_fence, cancel_task_by_id,
    clear_acceptance_fence_for_root,
    resume_budget_paused_task, restore_queue_fences, transition_acceptance_fence,
)
log = logging.getLogger(__name__)


DRIVE_ROOT: pathlib.Path = pathlib.Path(DATA_DIR)
# The queue snapshot path has ONE authority (MIGRATION row 1030, D18): this
# module. init() rebinds it per drive root; the queue_snapshot leaf reads it
# through the _queue() handle.
QUEUE_SNAPSHOT_PATH: pathlib.Path = DRIVE_ROOT / "state" / "queue_snapshot.json"
HEARTBEAT_STALE_SEC: int = 120
QUEUE_MAX_RETRIES: int = 1
FINALIZATION_GRACE_SEC: int = FINALIZATION_GRACE_DEFAULT_SEC
SCHEDULED_TASKS_FILE = pathlib.Path("state") / "scheduled_tasks.json"
# BUG3: pause a campaign whose objective fails to absorb after this many reviewed cycles.
# Mirrors the consecutive-failures threshold; keyed on the objective fingerprint, not failures.
OBJECTIVE_REPEAT_CAP: int = 3


def init(drive_root: pathlib.Path) -> None:
    global DRIVE_ROOT, FINALIZATION_GRACE_SEC, QUEUE_SNAPSHOT_PATH
    DRIVE_ROOT = drive_root
    QUEUE_SNAPSHOT_PATH = drive_root / "state" / "queue_snapshot.json"
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec()
    BUDGET_ROOT_FENCES.clear()


def refresh_timeouts_from_settings(settings: dict) -> None:
    """Hot-reload the active liveness settings.

    The flat wall-clock pair this once also had to absorb (soft/hard) is retired
    in 7.0: load_settings strips the keys, so there is no stored value left to
    accept, warn about, or lie about honoring.
    """
    global FINALIZATION_GRACE_SEC
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec(settings)


# Set by workers.init_queue_refs().
PENDING: List[Dict[str, Any]] = []
RUNNING: Dict[str, Dict[str, Any]] = {}
QUEUE_SEQ_COUNTER_REF: Dict[str, int] = {"value": 0}
ACCEPTANCE_FENCES: Dict[str, Dict[str, Any]] = {}
ADMISSION_RESERVATIONS: Dict[str, str] = {}

# Guards PENDING/RUNNING mutations across main loop, direct chat, watchdog.
_queue_lock = threading.RLock()
from supervisor.task_admission import (  # noqa: E402,F401 - public queue API
    coerce_queue_order, prefer_terminalization_retry_rows, record_scheduled_admission,
    reject_invalid_task_depth, release_task_admission, restore_invalid_depth_admission,
    restore_terminalization_retry, restore_terminalization_retry_rows,
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
    pr = coerce_queue_order(task.get("priority"), _task_priority(str(task.get("type") or "")))
    seq = coerce_queue_order(task.get("_queue_seq"))
    return pr, seq


def sort_pending() -> None:
    """Sort pending queue by priority and insertion sequence."""
    PENDING.sort(key=_queue_sort_key)


def drain_all_pending(*, persist: bool = True) -> list:
    """Drain pending tasks; optionally defer snapshot persistence until custody settles."""
    drained = list(PENDING)
    PENDING.clear()
    if persist:
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
        if require_unique_id and task_id:
            # Exact-id ownership wins over malformed-depth replay.
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
                if load_task_result(DRIVE_ROOT, task_id, strict=True):
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
        retry = restore_terminalization_retry(t, pending=PENDING, running=RUNNING, queue_seq_counter_ref=QUEUE_SEQ_COUNTER_REF, sort_pending=sort_pending) if restoring_snapshot else None
        if retry:
            return retry
        if reject_invalid_task_depth(t, reservations=ADMISSION_RESERVATIONS, admission_token=admission_token):
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
        t["priority"] = coerce_queue_order(t.get("priority"), _task_priority(str(t.get("type") or "")))
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
    once_due as _once_due,  # noqa: F401
    parse_schedule_time as _parse_schedule_time,  # noqa: F401
    prune_consumed_once_records as _prune_consumed_once, record_last_error as _record_last_error,  # noqa: F401
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
                # The tree identity survives even though the row already left
                # PENDING/RUNNING: the fence-release seam resolves the root
                # from the event when the queue no longer holds the task.
                "root_task_id": str((task or {}).get("root_task_id") or "") if isinstance(task, dict) else "",
                "task_type": str((task or {}).get("type") or ""),
                "chat_id": chat_id,
                "status": status,
                "outcome_axes": terminal_outcome_axes(
                    lifecycle=status, execution=status, reason_code=status,
                    review_trigger="supervisor_terminal",
                ),
                **(cost_fields or {
                    "cost_accounting_status": "unavailable", "cost_final": False,
                    # ABI-3: honest name only — the retired alias is read-only.
                    "accounted_upper_bound_usd": None,
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
    drive_cancel_intent_scope,
    task_has_live_ownership,
    task_subtree_is_live,
)
from supervisor.queue_transitions import (  # noqa: E402, F401 -- intentional public re-exports
    evolution_stop_report,
    stop_evolution_tasks,
    sweep_orphaned_budget_fences,
)


def _cancel_task_by_id_single(task_id: str) -> bool:
    """Boolean facade for the pre-v6.82 single-task callers."""
    return cancel_task_custody(task_id) in {CANCEL_CANCELLED, CANCEL_ALREADY_SETTLED}


# Evolution-stop transitions (GR2-13) live in supervisor.queue_transitions
# (module-size boundary); re-exported below with the other transition helpers
# so `supervisor.queue` stays the single import surface for callers.


def queue_deep_self_review_task(reason: str, model: str = "", force: bool = False, chat_id: Optional[int] = None) -> Optional[str]:
    """Queue a deep self-review task.

    ``chat_id`` targets a specific chat (e.g. the external transport chat that ran
    ``/review``) so the queued ack and the task results return to the requester
    instead of always defaulting to the web owner's ``owner_chat_id``.
    """
    # Membership, not truthiness: a review asked for from the hidden partition
    # is answered there, not silently re-routed to the owner's main chat.
    target_chat_id = notification_chat_route(chat_id, load_state().get("owner_chat_id"))
    if target_chat_id is None:
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
    # Typed SYSTEM row: an acknowledgement is never a task's answer, and the bench
    # trajectory reader takes the last UNTYPED outbound row as one.
    send_with_budget(int(target_chat_id), f"🔎 Deep self-review queued: {tid} ({reason})", role="system", system_type="deep_self_review_queued")
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


# v7next F1 (D08): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from supervisor.queue_schedules import (  # noqa: E402, F401 -- intentional public re-exports
    _SKILL_SCHEDULE_SYNC_INTERVAL_SEC,
    _last_skill_schedule_sync,
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
from supervisor.queue_snapshot import (  # noqa: E402, F401 -- intentional public re-exports
    _kept_service_pids,
    _retained_daemon_pids,
    parse_iso_to_ts,
    persist_queue_snapshot,
    restore_pending_from_snapshot,
)
from supervisor.queue_timeouts import (  # noqa: E402, F401 -- intentional public re-exports
    _enforce_task_timeouts_locked,
    _has_live_descendant,
    _has_pending_descendant,
    _is_descendant_of,
    _subtree_progressing,
    _task_deadline_ts,
    _task_drive_for_task,
    enforce_task_timeouts,
)
