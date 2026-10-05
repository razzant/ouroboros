"""Supervisor task queue, persistence, timeouts, and evolution scheduling."""

from __future__ import annotations

import contextlib
import contextvars
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
from ouroboros.consciousness_authority import apply_consciousness_authority, is_consciousness_origin
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


# Whether THIS process owns the supervisor's live maps (``init`` ran): a process that merely
# imports the module sees empty maps, which prove nothing (``task_settlement_liveness``).
INITIALIZED = False


def init(drive_root: pathlib.Path) -> None:
    global DRIVE_ROOT, FINALIZATION_GRACE_SEC, INITIALIZED, QUEUE_SNAPSHOT_PATH
    DRIVE_ROOT = drive_root
    INITIALIZED = True
    QUEUE_SNAPSHOT_PATH = drive_root / "state" / "queue_snapshot.json"
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec()
    BUDGET_ROOT_FENCES.clear()
    # A previous process's direct-chat turns must not outlive it in the roster,
    # and this clear is the last moment their ids exist: the roster is taken over
    # here and handed to snapshot restore below, which fences them like any other
    # row the stop caught.
    from supervisor.direct_roots import take_direct_roots

    PRIOR_DIRECT_ROOTS.clear()
    PRIOR_DIRECT_ROOTS.update(take_direct_roots(drive_root))


def refresh_timeouts_from_settings(settings: dict) -> None:
    """Hot-reload the active liveness settings.

    The flat wall-clock pair this once also had to absorb (soft/hard) is retired
    in 7.0: load_settings strips the keys, so there is no stored value left to
    accept, warn about, or lie about honoring.
    """
    global FINALIZATION_GRACE_SEC
    FINALIZATION_GRACE_SEC = get_finalization_grace_sec(settings)


# The previous process's direct-chat roots, taken from `state/direct_roots.json`
# by init above and consumed once by snapshot restore. A process-local handover
# of the SAME fragment, never a second store.
PRIOR_DIRECT_ROOTS: Dict[str, Any] = {}

# Set by workers.init_queue_refs().
PENDING: List[Dict[str, Any]] = []
RUNNING: Dict[str, Dict[str, Any]] = {}
QUEUE_SEQ_COUNTER_REF: Dict[str, int] = {"value": 0}
ACCEPTANCE_FENCES: Dict[str, Dict[str, Any]] = {}
ADMISSION_RESERVATIONS: Dict[str, str] = {}

# Guards PENDING/RUNNING mutations across main loop, direct chat, watchdog.
_queue_lock = threading.RLock()
from supervisor.task_admission import (  # noqa: E402,F401 - public queue API
    coerce_queue_order, enqueue_with_admission_receipt, prefer_terminalization_retry_rows,
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


# A root's billing a caller resolved before taking ``_queue_lock``: (task id, binding).
_PREPARED_ROOT_BILLING: contextvars.ContextVar = contextvars.ContextVar("prepared_root_billing", default=None)


def prepare_root_billing(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The whole-work binding of a ROOT about to be admitted, resolved OFF the queue lock.

    The lookup may read the money ledger (its own cross-process lock; in a cold
    process a full parse), so a caller that holds ``_queue_lock`` for its own
    admission transaction resolves this first, inside ``prepared_root_billing``.
    ``None`` for a child: nothing to resolve.
    """
    root_id = str(task.get("id") or "").strip()
    if not root_id or str(task.get("root_task_id") or root_id) != root_id:
        return None
    from ouroboros.config import runtime_setting
    from ouroboros.usage_admission import task_billing_fields

    limit = float(runtime_setting("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
    return task_billing_fields(task, root_id, limit if limit > 0 else None,
                               task.get("budget_drive_root") or DRIVE_ROOT, pin_initial=True,
                               persist_initial=False)


@contextlib.contextmanager
def prepared_root_billing(task: Dict[str, Any]):
    """Resolve a root's billing BEFORE the caller takes ``_queue_lock``.

    ``enqueue_task`` called inside the block for the same task uses the result
    instead of reading the ledger under the lock. Written as a context manager
    so the call shape ``enqueue_task(task, ...)`` stays the same everywhere.
    """
    token = _PREPARED_ROOT_BILLING.set((str(task.get("id") or "").strip(), prepare_root_billing(task)))
    try:
        yield
    finally:
        _PREPARED_ROOT_BILLING.reset(token)


def enqueue_task(
    task: Dict[str, Any], front: bool = False, *, restoring_snapshot: bool = False,
    consciousness_window: Optional[Dict[str, Any]] = None, continuation: bool = False,
    project_admission: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add task to PENDING (thread-safe: HTTP handlers enqueue concurrently
    with the supervisor main loop, so the mutation must hold the queue lock).

    A root's whole-work binding is resolved OFF the queue lock: here before the
    lock is taken, or earlier by a caller that holds the lock for its own
    admission transaction (``prepared_root_billing``).

    ``consciousness_window``: an allowance the caller already read OFF the queue lock
    (the scheduler). ``continuation``: host-derived only (a follow-up row's own
    ``continuation_of``) — a named continuation is not a spontaneous start, so it is
    outside the consciousness concurrency cap; money (the allowance) still applies."""
    t = dict(task)
    if not restoring_snapshot:
        t.pop("_project_scope_none", None)  # Only this host admission can attest absence.
        t.pop("_consciousness_continuation", None)  # never a caller-supplied marker
        if continuation:
            t["_consciousness_continuation"] = True
    attach_task_contract(apply_consciousness_authority(t))
    # The allowance read takes the cross-process ledger lock: read it BEFORE the queue
    # lock so a contended ledger never stalls every queue reader; only the live-root
    # count and the append must be one transaction with the lock (the window gates
    # starts — a window stale by milliseconds changes nothing).
    if consciousness_window is None and not restoring_snapshot:
        consciousness_window = consciousness_admission_window(t)
    # The whole-work binding of a root may read the money ledger too (its own
    # cross-process lock; in a cold process a full parse): resolve it here, off
    # the queue lock, and only attach it inside. A caller whose admission
    # transaction already holds the lock resolved it earlier (``prepared_root_billing``).
    prepared = _PREPARED_ROOT_BILLING.get()
    if prepared is not None and prepared[0] == str(t.get("id") or "").strip():
        billing = prepared[1]
    else:
        billing = None if restoring_snapshot else prepare_root_billing(t)
    project_id = str(t.get("project_id") or "").strip()
    # The host preparation basis survives queue snapshots and retries. Legacy
    # tasks lacking one are checked against current authority without claiming
    # that their historical preparation generation is known.
    has_project_admission = project_admission is not None or "_project_admission" in t
    project_admission = t.get("_project_admission") if project_admission is None else project_admission
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
                from ouroboros.routing_wait import is_own_admission_stub
                from ouroboros.task_results import load_task_result
                stored = load_task_result(DRIVE_ROOT, task_id, strict=True)
                # The emitted promote stub (#1160) belongs to THIS admission token:
                # its own enqueue reads around it, any other row still owns the id.
                if stored and not is_own_admission_stub(stored, admission_token):
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

                pool_state = workers._worker_pool_execution_state()
            except Exception:
                pool_state = {"available": False, "disabled_reason": "state_unavailable"}
            if not pool_state["available"]:
                if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                    ADMISSION_RESERVATIONS.pop(task_id, None)
                t["_admission_blocked"] = "worker_pool_unavailable"
                t["_worker_pool_disabled_reason"] = pool_state["disabled_reason"]
                return t
        consciousness_block = None if restoring_snapshot else _consciousness_admission_block(t, consciousness_window)
        if consciousness_block is not None:
            t["_admission_blocked"], t["_admission_detail"] = consciousness_block
            if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                ADMISSION_RESERVATIONS.pop(task_id, None)
            return t
        if admission_token and reserved_token != admission_token:
            t["_admission_blocked"] = "admission_reservation_lost"
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
        if billing is not None:
            from ouroboros.usage_admission import UNAVAILABLE_GROUP_PREFIX

            if str(billing["billing_group_id"]).startswith(UNAVAILABLE_GROUP_PREFIX):
                t["_admission_blocked"] = "billing_authority_unavailable"
                return t
            t.setdefault("metadata", {})["billing_group"] = {k: v for k, v in billing.items()
                                                             if k.startswith("billing_group_")}
        QUEUE_SEQ_COUNTER_REF["value"] += 1
        seq = QUEUE_SEQ_COUNTER_REF["value"]
        t["priority"] = coerce_queue_order(t.get("priority"), _task_priority(str(t.get("type") or "")))
        _att = t.get("_attempt")
        t.setdefault("_attempt", int(_att) if _att is not None else 1)
        t["_queue_seq"] = -seq if front else seq
        t["queued_at"] = utc_now_iso()
        if admission_token:
            t["_admission_owner_token"] = admission_token
        from contextlib import nullcontext
        from supervisor.followup_policy import scheduled_start
        from ouroboros.project_admission import host_unscoped
        from ouroboros.projects_registry import (
            ProjectAdmissionError, project_admission_guard, project_admission_view,
            task_project_membership, validate_project_admission,
        )

        lookup = "project_routing_fence_lookup_failed"  # the authority whose read failed
        try:
            if has_project_admission:
                validate_project_admission(project_admission)
            if project_admission is None or project_admission["project"] is None:
                unscoped = restoring_snapshot and host_unscoped(t)
                project_id, known_room = task_project_membership(DRIVE_ROOT, t)
                if project_id and unscoped:
                    # Host-attested absence is the admitted scope: a later binding
                    # never retargets it (hold release applies the same rule).
                    raise ProjectAdmissionError("project_routing_fence_changed",
                                                "The task's original unscoped assignment changed.")
                if project_id:
                    t["project_id"] = project_id
                    if project_admission is None:
                        project_admission = project_admission_view(
                            DRIVE_ROOT, project_id, allow_unregistered=not known_room, frozen=True)
                        project_admission["legacy_basis"] = True
                    elif known_room:
                        raise ProjectAdmissionError("project_routing_fence_changed", "The registered Project is missing.")
            if project_admission is not None and project_admission["project_id"] != project_id:
                raise ProjectAdmissionError("project_routing_fence_changed", "The prepared Project scope changed.")
            # Preparation/allowance reads precede these short control locks.
            # Restoring custody is not a new admission or permission to launch.
            lookup = "followup_control_wait"
            with (nullcontext(True) if restoring_snapshot else scheduled_start(DRIVE_ROOT, t)) as allowed:
                lookup = "project_routing_fence_lookup_failed"
                if not allowed:
                    t["_admission_blocked"] = "followup_control_wait"
                    return t
                if project_id:
                    with project_admission_guard(DRIVE_ROOT, project_admission):
                        # Recovery keeps the resource actually admitted, without a
                        # transient derived-selection census or current-folder substitution.
                        t["_project_admission"] = {key: value for key, value in project_admission.items()
                                                   if key != "workspace_claims"}
                        t["_project_admission"]["frozen"] = True
                        PENDING.append(t)
                else:
                    if not restoring_snapshot and not has_project_admission:
                        t["_project_scope_none"] = True
                    PENDING.append(t)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            t.update(_admission_blocked=getattr(exc, "reason", lookup),
                     _admission_detail=str(exc), _project_id=project_id,
                     _project_lifecycle=getattr(exc, "lifecycle", ""),
                     _admission_never_admitted=True)
            if admission_token and ADMISSION_RESERVATIONS.get(task_id) == admission_token:
                ADMISSION_RESERVATIONS.pop(task_id, None)
            return t
        sort_pending()
        if not restoring_snapshot:
            # A live fresh admission is positive host evidence. Retry/resume
            # authority belongs to its existing owner; restore never backfills.
            t["admitted_dispatch"] = "possible" if (
                int(t.get("_attempt") or 1) > 1 or t.get("original_task_id")
                or t.get("timeout_retry_from") or t.get("_owner_wait_resume")
                or t.get("_budget_pause_resume")
                or t.get("admitted_dispatch") == "possible"
            ) else "none"
        if ADMISSION_RESERVATIONS.get(task_id) == admission_token:
            ADMISSION_RESERVATIONS.pop(task_id, None)
    return t


def ensure_control_task_result(task_id: str) -> Dict[str, Any]:
    """Seed absent pooled lifecycle authority for a control, never from an ID alone.

    Queue membership owns the facts for every admission producer (review,
    evolution, assisted update and restore included). Keep the lock through
    create-only publication; a concurrent worker/terminal writer wins unchanged.
    Billing and direct-chat operations never create lifecycle rows here.
    """
    from ouroboros.task_results import load_task_result, resolve_task_lineage, write_task_result

    with _queue_lock:
        meta = RUNNING.get(task_id)
        task = meta.get("task") if isinstance(meta, dict) else None
        status = "running" if isinstance(task, dict) else "scheduled"
        if not isinstance(task, dict):
            task = next((row for row in PENDING if row.get("id") == task_id), None)
        if not isinstance(task, dict) or task.get("id") != task_id or task.get("_admission_blocked"):
            raise ValueError("control requires an admitted pooled task")
        root = pathlib.Path(task.get("budget_drive_root") or DRIVE_ROOT)
        existing = load_task_result(root, task_id, strict=True)
        if existing is not None:
            return existing
        fields = {key: task[key] for key in (
            "type", "chat_id", "metadata", "task_contract", "root_task_id", "parent_task_id",
            "delegation_role", "project_id", "workspace_root", "workspace_mode", "memory_mode",
            "budget_drive_root", "queued_at", "admitted_dispatch", "_admission_owner_token",
            "origin_message_text", "origin_message_ref", "objective", "title", "suggested_name",
            "original_task_id", "timeout_retry_from", "deadline_at", "root_cost_ceiling_usd",
            "billing_group", "task_constraint", "objective_author", "owner_corpus", "task_group_id", "task_group",
        ) if key in task}
        fields["root_task_id"] = resolve_task_lineage(task_id, **{
            key: task.get(key) for key in ("metadata", "root_task_id", "parent_task_id", "delegation_role",
                                          "original_task_id", "timeout_retry_from")})["root_task_id"]
        fields["description"] = task.get("description") or task.get("text") or ""
        # The host attempt key the assignment mirror and the executor's start copy write:
        # a split root's copyback authenticates its terminal time against it (terminal_time).
        fields["task_attempt"] = int((meta.get("attempt") if status == "running" else 0) or task.get("_attempt") or 1)
        if status == "running":
            fields["started_at"] = meta.get("started_at")
        return write_task_result(root, task_id, status, create_only=True, strict_existing_dict=True, **fields)


def live_consciousness_root_count() -> int:
    """Live PENDING+RUNNING roots that consciousness started (its origin marker on the
    task metadata; subagents are their root's business). Sibling of ``queue_has_task_type``."""
    def _counts(task: Any) -> bool:
        return (
            isinstance(task, dict)
            and str(task.get("delegation_role") or "root") == "root"
            and is_consciousness_origin(task.get("metadata"))
            and not task.get("_consciousness_continuation")  # a named continuation is not spontaneous
        )

    live = sum(1 for task in PENDING if _counts(task))
    return live + sum(
        1 for meta in RUNNING.values() if isinstance(meta, dict) and _counts(meta.get("task"))
    )


def _consciousness_root(task: Dict[str, Any]) -> bool:
    return (is_consciousness_origin(task.get("metadata"))
            and str(task.get("delegation_role") or "root") == "root")


def consciousness_admission_window(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The allowance a consciousness-started root's admission reads BEFORE Q; None otherwise."""
    if not _consciousness_root(task):
        return None
    from ouroboros.consciousness_allowance import allowance_window

    return allowance_window(DRIVE_ROOT)


def _consciousness_admission_block(task: Dict[str, Any], window: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, str]]:
    """The ONE admission door for the roots consciousness starts (owner decisions В11/В18).

    Called under the queue lock, so the live count and the admission are one
    transaction. Returns ``(reason, detail)`` in the queue's existing refusal
    vocabulary when a consciousness-origin ROOT may not start — the concurrency
    cap over live roots with the same marker (``OUROBOROS_CONSCIOUSNESS_MAX_TASKS``,
    0 = never) or the rolling-24h allowance (``OUROBOROS_CONSCIOUSNESS_DAILY_USD``,
    0 = consciousness may not spend; an unreadable ledger refuses honestly as
    ``allowance_unknown``) — and ``None`` when it may. Subagents are bounded by
    their root's own cap and the per-root child cap, never counted twice; a
    snapshot restore re-admits already-admitted work and is not gated here. ``window``
    is the allowance the caller read before taking the lock; read here only when absent.
    """
    if not _consciousness_root(task):
        return None
    from ouroboros.config import get_consciousness_max_tasks
    from ouroboros.consciousness_allowance import (
        STATUS_AVAILABLE, STATUS_UNKNOWN, allowance_window,
    )

    max_tasks = get_consciousness_max_tasks()
    live = live_consciousness_root_count()
    if live >= max_tasks and not task.get("_consciousness_continuation"):
        return ("consciousness_task_limit", (
            f"{live} of {max_tasks} consciousness-started tasks already live"
            if max_tasks else "OUROBOROS_CONSCIOUSNESS_MAX_TASKS=0: consciousness never starts tasks"
        ))
    if window is None:
        window = allowance_window(DRIVE_ROOT)
    if window["status"] == STATUS_UNKNOWN:
        return ("consciousness_allowance_unknown",
                f"the usage ledger could not be read: {window.get('error') or 'unknown error'}")
    if window["status"] != STATUS_AVAILABLE:
        if not window["limit_usd"]:
            return ("consciousness_allowance_exhausted",
                    "OUROBOROS_CONSCIOUSNESS_DAILY_USD=0: consciousness may not spend")
        at_least = " (at least)" if window["unknown_unmetered"] else ""
        return ("consciousness_allowance_exhausted", (
            f"${window['accounted_usd']:.2f}{at_least} of ${window['limit_usd']:.2f} "
            f"spent in the last 24 h; resets at {window['resets_at'] or 'unknown'}"
        ))
    return None


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
    task_settlement_interlock,
    task_settlement_liveness,
)


def _cancel_task_by_id_single(task_id: str) -> bool:
    """Boolean facade for the pre-v6.82 single-task callers."""
    return cancel_task_custody(task_id) in {CANCEL_CANCELLED, CANCEL_ALREADY_SETTLED}


# Evolution-stop transitions (GR2-13) live in supervisor.queue_transitions
# (module-size boundary); re-exported below with the other transition helpers
# so `supervisor.queue` stays the single import surface for callers.


def queue_deep_self_review_task(reason: str, model: str = "", force: bool = False, chat_id: Optional[int] = None,
                                origin: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Queue a deep self-review task.

    ``chat_id`` targets a specific chat (e.g. the external transport chat that ran
    ``/review``) so the queued ack and the task results return to the requester
    instead of always defaulting to the web owner's ``owner_chat_id``. ``origin`` is
    the requester's consciousness origin (a wake-up or its tree), stamped on the
    root so the admission door and the ledger see it; empty for the owner.
    """
    # Membership, not truthiness: a review asked for from the hidden partition
    # is answered there, not silently re-routed to the owner's main chat.
    target_chat_id = notification_chat_route(chat_id, load_state().get("owner_chat_id"))
    if target_chat_id is None:
        return None
    if (not force) and queue_has_task_type("deep_self_review"):
        return None
    tid = uuid.uuid4().hex[:8]
    admitted = enqueue_with_admission_receipt({
        "id": tid,
        "type": "deep_self_review",
        "chat_id": int(target_chat_id),
        "text": reason or "Deep self-review",
        "model": model,
        "_require_worker_pool": True,
        **({"metadata": dict(origin)} if origin else {}),
    })
    if admitted.get("_admission_blocked"):
        reason = admitted.get("_worker_pool_disabled_reason") or admitted["_admission_blocked"]
        detail = str(admitted.get("_admission_detail") or "")
        hint = f" {detail}." if detail else " Use /restart to restore the worker pool."
        send_with_budget(
            int(target_chat_id),
            f"Deep self-review could not be queued: {reason}.{hint}",
            role="system", system_type="deep_self_review_unavailable",
        )
        return None
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
    from supervisor.state import control_value

    enabled_known, enabled = control_value(st, "evolution_mode_enabled")
    enabled = bool(enabled)
    owner_chat_id = int(st.get("owner_chat_id") or 0)
    consecutive_failures = int(st.get("evolution_consecutive_failures") or 0)
    try:
        # A status snapshot is a display read: without a supplied projection it rides the last validated snapshot.
        remaining: Optional[float] = round(float(budget_remaining(st, strict=True, projection=budget_projection, allow_stale=True)), 2)
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

    if not enabled_known:
        status = "state_unknown"  # never read an unknown control as on or off (#1307)
        detail = "Evolution control is unknown: runtime state is unavailable or recovering from a backup."
    elif restart_blocked:
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
        "enabled": enabled if enabled_known else None,
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
    SCHEDULE_ACTIONS,
    ScheduleLockTimeout,
    ScheduleRefused,
    ScheduleStoreUnreadable,
    check_scheduled_tasks,
    list_scheduled_tasks,
    load_schedule_store,
    resync_skill_schedules,
    schedule_activity_projection,
    schedule_tool_projection,
    schedule_lifecycle_status,
    schedule_transaction,
    sync_skill_schedules,
)
from supervisor.schedule_lifecycle import (  # noqa: E402, F401 -- intentional public re-exports
    mutate_scheduled_task,
    remove_scheduled_task,
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
