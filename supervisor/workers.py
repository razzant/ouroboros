"""Worker lifecycle, health, and direct-chat handling for the supervisor."""

from __future__ import annotations
import logging
log = logging.getLogger(__name__)

# Pool responsibilities owned by their own modules (module-size boundary):
# promotion, the chat lanes, crash detection, spawn/respawn lifecycle and task
# assignment. Each reads the pool's rebound state through a handle back to this
# module — see the handle docstring in any of them — and each is re-imported
# here so `supervisor.workers` stays the single public import surface. The
# lifecycle serializer comes back as an ordinary import: a decorator is applied
# at import time, so it is the one name a call-time handle cannot carry.
from supervisor.worker_promotion import (  # noqa: F401 -- supervisor/workers.py facade re-exports
    _admit_promoted_workspace,
    _canonical_promoted_repair_constraint,
    _fail_promoted_task_loudly,
    _origin_from_mapping,
    _origin_from_task_record,
    _promote_duplicate_reason,
    _promoted_force_plan_metadata,
    _report_binding_failure,
    ensure_project_scope,
    promote_chat_to_task,
)
from supervisor.worker_chat_lane import (  # noqa: F401 -- supervisor/workers.py facade re-exports
    _broadcast_task_named,
    _handle_chat_direct_locked,
    _run_chat_task,
    auto_resume_after_restart,
    handle_chat_direct,
    handle_chat_ephemeral,
)
from supervisor.worker_health import (  # noqa: F401 -- supervisor/workers.py facade re-exports
    _emit_task_done_terminal,
    _ensure_workers_healthy_locked,
    ensure_workers_healthy,
    terminal_task_metadata,
)
from supervisor.worker_pool_lifecycle import (  # noqa: F401 -- supervisor/workers.py facade re-exports
    _WORKER_LIFECYCLE_LOCK,
    _first_worker_boot_event_since,
    _first_worker_event_since,
    _kill_survivors,
    _record_worker_pids,
    _serialized_worker_lifecycle,
    _verify_worker_sha_after_spawn,
    _worker_pids_path,
    _write_failure_result,
    kill_workers_for_update,
    reap_orphaned_workers,
    respawn_worker,
)
from supervisor.worker_assignment import (  # noqa: F401 -- supervisor/workers.py facade re-exports
    _cancel_unauthorized_evolution,
    _evolution_assignment_error,
    assign_tasks,
)

# The child process's own entry point, its root binding, its log sink filter
# and its crash record live in supervisor.worker_process: none of it reads pool
# state, because in that process none of it exists. Re-imported here so the
# spawn/respawn sites and the historical ``supervisor.workers`` names keep one
# surface; the dependency is one-way.
from supervisor.worker_process import (  # noqa: F401 -- supervisor/workers.py facade re-exports
    WORKER_LOG_SINK_SUPPRESSED_TYPES,
    _bind_worker_repo_root,
    _current_custody_session_id,
    _log_worker_crash,
    _prepare_worker_task_runtime,
    worker_main,
)

import json  # noqa: F401
import multiprocessing as mp
import os
import pathlib
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

from supervisor.state import load_state, append_jsonl, reconstruct_task_cost  # noqa: F401
from supervisor.message_bus import coerce_chat_identity, send_with_budget  # noqa: F401
from ouroboros.config import DATA_DIR, REPO_DIR as CONFIG_REPO_DIR
from ouroboros.outcomes import EXECUTION_FAILED, EXECUTION_INFRA_FAILED, terminal_outcome_axes  # noqa: F401
from ouroboros.utils import utc_now_iso


REPO_DIR: pathlib.Path = pathlib.Path(CONFIG_REPO_DIR)
DRIVE_ROOT: pathlib.Path = pathlib.Path(DATA_DIR)
MAX_WORKERS: int = 10
HEARTBEAT_STALE_SEC: int = 120
QUEUE_MAX_RETRIES: int = 1
BRANCH_DEV: str = "ouroboros"
BRANCH_STABLE: str = "ouroboros-stable"

_CTX = None
_LAST_SPAWN_TIME: float = 0.0  # grace period: don't count dead workers right after spawn
_SPAWN_GRACE_SEC: float = 90.0  # workers need up to ~60s to init (spawn + pip)

# macOS + Windows default to spawn; Linux keeps fork.
#
# fork() from the long-lived, multi-threaded supervisor is unsafe on macOS: the
# child inherits dead Mach ports, and the first network call that resolves
# system proxies (SCDynamicStoreCopyProxies via _scproxy / httpx / requests)
# SIGSEGVs on the child side of fork pre-exec. macOS therefore uses spawn, like
# Windows. Linux proxy lookup reads env only (no Mach/GCD), so fork stays the
# default there for fast worker startup. ``worker_main`` is a module-level
# target (picklable) and re-derives all state from argv, so spawn is safe; the
# PyInstaller bootloader provides multiprocessing.freeze_support() for frozen
# builds. Override with OUROBOROS_WORKER_START_METHOD when diagnosing.
_DEFAULT_WORKER_START_METHOD = "fork" if sys.platform.startswith("linux") else "spawn"
_WORKER_START_METHOD = str(os.environ.get("OUROBOROS_WORKER_START_METHOD", _DEFAULT_WORKER_START_METHOD) or _DEFAULT_WORKER_START_METHOD).strip().lower()
if _WORKER_START_METHOD not in {"fork", "spawn", "forkserver"}:
    _WORKER_START_METHOD = _DEFAULT_WORKER_START_METHOD


def _get_ctx():
    """Return the multiprocessing context for workers."""
    global _CTX
    if _CTX is None:
        _CTX = mp.get_context(_WORKER_START_METHOD)
    return _CTX


def init(repo_dir: pathlib.Path, drive_root: pathlib.Path, max_workers: int,
         branch_dev: str = "ouroboros", branch_stable: str = "ouroboros-stable") -> None:
    """Bind the worker pool to its repo, drive, size and branch defaults."""
    global REPO_DIR, DRIVE_ROOT, MAX_WORKERS, BRANCH_DEV, BRANCH_STABLE
    REPO_DIR = repo_dir
    DRIVE_ROOT = drive_root
    MAX_WORKERS = max_workers
    BRANCH_DEV = branch_dev
    BRANCH_STABLE = branch_stable

    from supervisor import queue
    queue.init(drive_root)
    queue.init_queue_refs(PENDING, RUNNING, QUEUE_SEQ_COUNTER_REF)

@dataclass
class Worker:
    wid: int
    proc: mp.Process
    in_q: Any
    busy_task_id: Optional[str] = None
    # Variant A (off-loop reaping): set under _queue_lock when a timed-out task's heavy
    # teardown (kill/join/archive/respawn) is handed to the background reaper. The slot
    # is unavailable for assignment until respawn_worker() installs a fresh Worker.
    reaping: bool = False


_EVENT_Q = None
_EVENT_Q_MANAGER = None
_EVENT_Q_GENERATION = ""
_EVENT_Q_LOCK = threading.Lock()
_EVENT_Q_SHUTDOWN = False
_WORKER_POOL_DISABLED_REASON = ""


def get_event_q():
    """Return the process-lifetime supervisor event bus, creating it lazily.

    Worker-pool generations are replaceable; the producers that publish onto
    this bus (direct chat, consciousness, active turns, and workers) are not.
    Rotating the queue during a pool respawn strands those producers on an
    undrained queue, so only a new server process creates a new bus.
    """
    global _EVENT_Q, _EVENT_Q_MANAGER, _EVENT_Q_GENERATION
    with _EVENT_Q_LOCK:
        if _EVENT_Q_SHUTDOWN:
            raise RuntimeError("supervisor event bus is shutting down")
        if _EVENT_Q is None:
            # A raw multiprocessing.Queue has an asynchronous feeder and its
            # pipe can be corrupted when a worker is force-killed mid-frame.
            # A manager-backed queue serializes synchronously in the producer
            # and isolates each producer connection, so replacing/killing a
            # worker generation cannot wedge the process-lifetime bus.
            _EVENT_Q_MANAGER = _get_ctx().Manager()
            _EVENT_Q = _EVENT_Q_MANAGER.Queue()
            _EVENT_Q_GENERATION = f"{os.getpid()}:{uuid.uuid4().hex[:12]}"
            try:
                from ouroboros.process_custody import record_process

                manager_proc = getattr(_EVENT_Q_MANAGER, "_process", None)
                manager_pid = int(getattr(manager_proc, "pid", 0) or 0)
                if manager_pid:
                    record_process(
                        DRIVE_ROOT,
                        pid=manager_pid,
                        cmd="multiprocessing SyncManager",
                        purpose="supervisor_event_queue_manager",
                        scope="session",
                        reap_process_group=False,
                    )
            except Exception:
                log.warning("Failed to custody-track event queue manager", exc_info=True)
            try:
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "event_queue_generation_started",
                        "generation": _EVENT_Q_GENERATION,
                        "server_pid": os.getpid(),
                        "start_method": _WORKER_START_METHOD,
                    },
                )
            except Exception:
                log.debug("Failed to record event queue generation", exc_info=True)
    return _EVENT_Q


def shutdown_event_q() -> None:
    """Stop the manager on graceful exit; custody reaps it after a hard exit."""
    global _EVENT_Q, _EVENT_Q_MANAGER, _EVENT_Q_GENERATION, _EVENT_Q_SHUTDOWN
    with _EVENT_Q_LOCK:
        _EVENT_Q_SHUTDOWN = True
        manager = _EVENT_Q_MANAGER
        _EVENT_Q = None
        _EVENT_Q_MANAGER = None
        _EVENT_Q_GENERATION = ""
    if manager is not None:
        try:
            manager.shutdown()
        except Exception:
            log.debug("Event queue manager shutdown failed", exc_info=True)


def event_queue_generation() -> str:
    """Stable diagnostic identity for the current server-process event bus."""
    get_event_q()
    return _EVENT_Q_GENERATION


WORKERS: Dict[int, Worker] = {}
PENDING: List[Dict[str, Any]] = []
RUNNING: Dict[str, Dict[str, Any]] = {}
CRASH_TS: List[float] = []
QUEUE_SEQ_COUNTER_REF: Dict[str, int] = {"value": 0}

# Shared queue lock; queue.py owns the canonical definition.
from supervisor.queue import _queue_lock


def worker_pool_admission_state(ctx: Any = None) -> Dict[str, Any]:
    """Return the user-facing managed-task executor admission state.

    A busy or reaping pool is still a valid queue target.  Only an explicitly
    disabled pool, or a genuinely absent pool after supervisor readiness, is
    unavailable.  Internal boot/update recovery may enqueue before an initial
    spawn and therefore does not use this user-ingress predicate.
    """
    pool = getattr(ctx, "WORKERS", WORKERS) if ctx is not None else WORKERS
    with _queue_lock:
        disabled_reason = str(_WORKER_POOL_DISABLED_REASON or "")
        worker_count = len(pool)
    update_reason = repo_writer_admission_closed()
    available = worker_count > 0 and not disabled_reason and not update_reason
    return {
        "available": available,
        "reason_code": "" if available else "worker_pool_unavailable",
        "disabled_reason": disabled_reason or update_reason or ("no_workers" if not worker_count else ""),
        "worker_count": worker_count,
    }


def ensure_worker_pool_started(n: int = 0, *, allow_disabled_restart: bool = False) -> bool:
    """Start an absent pool; only explicit internal recovery may clear disablement."""
    with _queue_lock:
        # Update admission can be closed while the one authorized assisted
        # resolver is already running. That does not make an existing healthy
        # pool absent and must never trigger a second full-pool spawn.
        if WORKERS and not _WORKER_POOL_DISABLED_REASON:
            return True
    state = worker_pool_admission_state()
    if state["available"]:
        return True
    if state["disabled_reason"] not in {"", "no_workers"} and not allow_disabled_restart:
        return False
    spawn_workers(n)
    return True


_chat_agent = None
# Serializes every direct-chat caller; _chat_agent has mutable per-call state.
import threading as _threading
_chat_agent_lock = _threading.Lock()
_ephemeral_chat_lock = _threading.Lock()
_repo_writer_gate_lock = _threading.Lock()
_repo_writer_gate_reason = ""


def close_repo_writer_admission(reason: str) -> None:
    """Stop new in-process chat turns from entering the managed checkout."""
    global _repo_writer_gate_reason
    with _repo_writer_gate_lock:
        _repo_writer_gate_reason = str(reason or "managed update")


def open_repo_writer_admission(expected_reason: str = "") -> bool:
    """Open the process-local gate, optionally only for the exact current owner."""
    global _repo_writer_gate_reason
    with _repo_writer_gate_lock:
        if expected_reason and _repo_writer_gate_reason != expected_reason:
            return False
        _repo_writer_gate_reason = ""
    return True


def repo_writer_admission_closed() -> str:
    with _repo_writer_gate_lock:
        reason = _repo_writer_gate_reason
    if reason:
        return reason
    # The in-memory latch disappears on restart; the transaction marker does
    # not. Let that durable state close the same gate during boot recovery.
    try:
        from supervisor.update_merge import active_update_tx

        tx = active_update_tx()
        if tx:
            return f"managed_update_tx:{tx.get('phase') or 'unknown'}"
    except Exception:
        log.warning("Could not read durable managed-update admission state", exc_info=True)
        return "managed_update_tx:unreadable"
    return ""


def repo_writer_task_allowed(task: Dict[str, Any]) -> bool:
    """Only the tx-authorized resolver may dispatch while the gate is closed."""
    if not repo_writer_admission_closed():
        return True
    try:
        from supervisor.update_merge import authorized_assisted_task

        return bool(authorized_assisted_task(
            str(task.get("id") or ""),
            task.get("metadata") if isinstance(task.get("metadata"), dict) else None,
        ))
    except Exception:
        return False


def drain_repo_writers(timeout: float = 30.0) -> List[str]:
    """Wait for the two existing in-process writer lanes after admission closes."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    blocked: List[str] = []
    for label, lock in (("direct_chat", _chat_agent_lock), ("ephemeral_chat", _ephemeral_chat_lock)):
        remaining = max(0.0, deadline - time.monotonic())
        if not lock.acquire(timeout=remaining):
            blocked.append(label)
            continue
        lock.release()
    return blocked


def _repo_writer_turn_allowed(chat_id: int) -> bool:
    reason = repo_writer_admission_closed()
    if not reason:
        return True
    try:
        send_with_budget(
            chat_id,
            "🔒 An update is using the repository. Try this message again when it finishes.",
        )
    except Exception:
        log.debug("Could not report managed-update writer gate", exc_info=True)
    return False


def _get_chat_agent():
    global _chat_agent
    if _chat_agent is None:
        if not getattr(sys, 'frozen', False):
            sys.path.insert(0, str(REPO_DIR))
        from ouroboros.agent import make_agent
        _chat_agent = make_agent(
            repo_dir=str(REPO_DIR),
            drive_root=str(DRIVE_ROOT),
            event_queue=get_event_q(),
        )
    return _chat_agent


def chat_turn_liveness():
    """(busy, task_id, last_activity_ts) of the in-process direct-chat turn — read
    WITHOUT taking _chat_agent_lock (a wedged turn holds that lock for its whole
    duration, so the watchdog must never block on it). The supervisor liveness
    watchdog (WS3) reads this to spot a heartbeat-silent direct turn, which is
    in-process and therefore invisible to the worker RUNNING heartbeat table."""
    agent = _chat_agent
    if agent is None or not getattr(agent, "_busy", False):
        return (False, None, None)
    return (True, getattr(agent, "_current_task_id", None), getattr(agent, "_last_activity_ts", None))


_WORKER_PIDS_FILENAME = "worker_pids.json"


@_serialized_worker_lifecycle
def spawn_workers(n: int = 0) -> None:
    global _CTX, _WORKER_POOL_DISABLED_REASON
    global _LAST_SPAWN_TIME
    with _queue_lock:
        if WORKERS:
            raise RuntimeError(
                "spawn_workers requires an empty pool; stop the current workers "
                "or use respawn_worker for one slot"
            )
    # Never hold the queue's process-local threading.RLock across fork: a child
    # would inherit it owned by a vanished thread.  The dedicated lifecycle lock
    # is not used by worker code and serializes competing full-pool starts/kills.
    reap_orphaned_workers()
    _CTX = mp.get_context(_WORKER_START_METHOD)
    event_q = get_event_q()
    events_path = DRIVE_ROOT / "logs" / "events.jsonl"
    try:
        events_offset = int(events_path.stat().st_size)
    except Exception:
        events_offset = 0

    count = n or MAX_WORKERS
    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "worker_spawn_start",
            "start_method": _WORKER_START_METHOD,
            "count": count,
            "event_queue_generation": event_queue_generation(),
            "event_queue_transport": "manager",
        },
    )
    new_workers: Dict[int, Worker] = {}
    try:
        for i in range(count):
            in_q = _CTX.Queue()
            proc = _CTX.Process(target=worker_main,
                               args=(i, in_q, event_q, str(REPO_DIR), str(DRIVE_ROOT),
                                     _current_custody_session_id()))
            proc.daemon = True
            proc.start()
            new_workers[i] = Worker(wid=i, proc=proc, in_q=in_q, busy_task_id=None)
    except Exception:
        for worker in new_workers.values():
            try:
                worker.proc.terminate()
                worker.proc.join(timeout=2)
            except Exception:
                pass
        raise
    with _queue_lock:
        if WORKERS:
            for worker in new_workers.values():
                try:
                    worker.proc.terminate()
                    worker.proc.join(timeout=2)
                except Exception:
                    pass
            raise RuntimeError("worker pool appeared during serialized startup")
        WORKERS.update(new_workers)
        _WORKER_POOL_DISABLED_REASON = ""
        _LAST_SPAWN_TIME = time.time()
    _record_worker_pids()
    # Verify asynchronously so spawn does not block the supervisor loop.
    threading.Thread(target=_verify_worker_sha_after_spawn, args=(events_offset,), daemon=True).start()


@_serialized_worker_lifecycle
def kill_workers(
    force: bool = True,
    *,
    result_reason: str = "Worker process crashed (crash storm). Task was not completed.",
    terminal_status: str = "",
    archive_service_logs: bool = True,
    disable_reason: str = "",
    preserve_pending: bool = False,
) -> None:
    global _WORKER_POOL_DISABLED_REASON
    from supervisor import queue
    with _queue_lock:
        if disable_reason:
            _WORKER_POOL_DISABLED_REASON = str(disable_reason)
            # Publish the admission fence before slow process-tree teardown so
            # concurrent ingress can refuse without starting project/workspace
            # side effects while workers are being joined.
            queue.persist_queue_snapshot(reason="worker_pool_disabling")
        cleared_running = len(RUNNING)
        from ouroboros.platform_layer import kill_pid_tree
        for w in WORKERS.values():
            if w.proc.pid:
                kill_pid_tree(w.proc.pid)
            elif w.proc.is_alive():
                w.proc.terminate()
        for w in WORKERS.values():
            w.proc.join(timeout=3)
        _kill_survivors()
        WORKERS.clear()
        orphaned_ids = []
        drained_ids = []
        try:
            done_status = terminal_status or "failed"
            running_task_ids = set(RUNNING)
            interrupted_roots = {
                str((meta.get("task") or {}).get("root_task_id") or task_id)
                for task_id, meta in RUNNING.items()
                if isinstance(meta, dict)
            }
            for task_id in list(RUNNING):
                meta = RUNNING.get(task_id) or {}
                task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
                try:
                    persisted = _write_failure_result(task_id, reason=result_reason, status=terminal_status)
                    if archive_service_logs:
                        try:
                            from ouroboros.tools.services import archive_task_service_logs
                            archive_task_service_logs(pathlib.Path(DRIVE_ROOT), str(task_id), task)
                        except Exception:
                            log.debug("Failed to archive service logs for task %s", task_id, exc_info=True)
                except Exception:
                    log.warning("Failed to write failure result for running task %s", task_id, exc_info=True)
                    persisted = done_status
                if _emit_task_done_terminal(task, str(task_id), persisted or done_status):
                    orphaned_ids.append(task_id)
            if preserve_pending:
                kept = []
                for task in PENDING:
                    parent_id = str(task.get("parent_task_id") or "")
                    root_id = str(task.get("root_task_id") or "")
                    if parent_id and (parent_id in running_task_ids or root_id in interrupted_roots):
                        tid = str(task.get("id") or "")
                        if tid:
                            persisted = _write_failure_result(
                                tid,
                                reason="Parent task was interrupted before this child started.",
                                status="cancelled",
                            )
                            if _emit_task_done_terminal(task, tid, persisted or "cancelled"):
                                drained_ids.append(tid)
                        continue
                    kept.append(task)
                PENDING[:] = kept
            else:
                drained = queue.drain_all_pending()
                for task in drained:
                    tid = task.get("id")
                    if tid:
                        try:
                            persisted = _write_failure_result(tid, reason=result_reason, status=terminal_status)
                        except Exception:
                            log.warning("Failed to write failure result for pending task %s", tid, exc_info=True)
                            persisted = done_status
                        if _emit_task_done_terminal(task, str(tid), persisted or done_status):
                            drained_ids.append(tid)
                        else:
                            PENDING.append(task)
            if orphaned_ids or drained_ids:
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "zombie_prevention_cleanup",
                        "orphaned_running": orphaned_ids,
                        "drained_pending": drained_ids,
                    },
                )
        except Exception:
            log.warning("Zombie prevention cleanup failed", exc_info=True)
        for terminal_id in orphaned_ids:
            RUNNING.pop(str(terminal_id), None)
    queue.persist_queue_snapshot(reason="kill_workers")
    if cleared_running:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "running_cleared_on_kill", "count": cleared_running,
                "force": force,
            },
        )


def _drop_cancelled_pending() -> None:
    """Remove pending tasks cancelled/finished between scheduling and assignment
    so a cancelled subagent never actually starts. Caller holds _queue_lock.

    The pre-assignment consult of the durable cancel-intent projection (phase A):
    a task with an active intent (or a legacy ``cancel_requested`` latch file) is
    settled as ``cancelled`` with a reconstructed — usually confirmed pre-start
    zero — cost, never assigned to a worker.

    It settles under the SAME rules ``cancel_task_custody`` follows, because it
    cannot call custody (the caller already holds ``_queue_lock``, which custody
    takes): the STORED status decides the outcome (a task that completed keeps
    its result), a durable write that FAILS leaves the intent active for the
    watchdog instead of publishing a cancellation that was never persisted, and
    the intent's ``requested_by`` stamps ``parent_decision`` at the OUTCOME — a
    parent whose reminder is silenced by that field would otherwise keep nagging
    about a child it cancelled itself.
    """
    if not PENDING:
        return
    try:
        from ouroboros.task_results import (
            STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, _TRULY_TERMINAL_STATUSES,
            load_task_result, write_task_result,
        )
    except Exception:
        return
    try:
        from ouroboros.cancel_intents import (
            active_intents, claim_intent, release_claim, settle_intent,
        )

        intents = active_intents(DRIVE_ROOT)
    except Exception:
        intents = {}
        settle_intent = claim_intent = release_claim = None
    try:
        from supervisor.task_lifecycle import _intent_outcome_fields
    except Exception:
        def _intent_outcome_fields(_intent):  # type: ignore[misc]
            return {}
    survivors: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for t in PENDING:
        tid = str(t.get("id") or "")
        status = ""
        if tid:
            try:
                existing = load_task_result(DRIVE_ROOT, tid)
                status = str((existing or {}).get("status") or "")
            except Exception:
                status = ""
        if tid and (status == STATUS_CANCEL_REQUESTED or tid in intents):
            # AR2-2 settle-owner unity: the drop CLAIMS the intent before it
            # settles, the same fence custody holds. A REFUSED claim means a
            # live custody attempt owns this teardown (it claimed on a capture
            # miss while the enqueue raced): the task still leaves the queue —
            # it must not be assigned — but the claim owner writes the terminal,
            # settles, and emits; a parallel settle here is the double-settle
            # the fence exists to stop.
            claim: Dict[str, Any] = {}
            if claim_intent is not None and tid in intents:
                try:
                    claim = claim_intent(DRIVE_ROOT, tid, owner="pending_drop") or {}
                except Exception:
                    claim = {"claim_refused": True}
                if claim.get("claim_refused"):
                    dropped.append(tid)
                    continue
            intent = claim or intents.get(tid) or {}
            try:
                cost_fields = reconstruct_task_cost(tid, fields=True)
            except Exception:
                cost_fields = {"cost_accounting_status": "unavailable",
                               "cost_final": False, "cost_usd": None}
            stored: Dict[str, Any] = {}
            write_failed = False
            try:
                stored = write_task_result(
                    DRIVE_ROOT, tid, STATUS_CANCELLED,
                    result="Cancelled before start.", **cost_fields,
                    **_intent_outcome_fields(intent),
                ) or {}
            except Exception:
                write_failed = True
                log.debug("Failed to finalize cancelled pending task %s", tid, exc_info=True)
            if write_failed:
                # Nothing durable happened. The task leaves the queue (it must
                # not be assigned) but the intent stays ACTIVE — the held claim
                # is RELEASED so the watchdog re-feeds custody, which writes the
                # real terminal; a settle+task_done here would publish a
                # cancellation that is not on disk.
                if release_claim is not None and claim.get("request_id"):
                    try:
                        release_claim(
                            DRIVE_ROOT, tid, error="pending-drop persistence failed",
                            expected_generation=claim.get("generation"),
                            request_id=str(claim.get("request_id") or ""),
                        )
                    except Exception:
                        log.debug("pending-drop claim release failed for %s", tid, exc_info=True)
                dropped.append(tid)
                continue
            stored_status = str(stored.get("status") or STATUS_CANCELLED)
            if settle_intent is not None and tid in intents:
                try:
                    # Fenced by this drop's OWN claim: a settle from a claim that
                    # was taken over is a no-op plus a forensic row. A
                    # scope=cascade intent — including one durably WIDENED after
                    # this loop's snapshot was read — is refused ATOMICALLY
                    # inside the settle (GR3-1: the cascade postcondition is its
                    # only settle owner) and this drop's claim is auto-released
                    # in the same write so the watchdog re-feeds the cascade.
                    settle_intent(
                        DRIVE_ROOT, tid,
                        outcome="cancelled" if stored_status == STATUS_CANCELLED else "already_settled",
                        detail=("dropped before assignment" if stored_status == STATUS_CANCELLED
                                else stored_status),
                        expected_generation=claim.get("generation"),
                        request_id=str(claim.get("request_id") or ""),
                    )
                except Exception:
                    log.debug("Failed to settle cancel intent for pending %s", tid, exc_info=True)
            # The STORED status, never a blanket "cancelled": the monotonic guard
            # refuses our write when the task settled on its own, and the card
            # must resolve to what actually happened (completion wins).
            _emit_task_done_terminal(t, tid, stored_status, cost_fields=cost_fields)
            dropped.append(tid)
            continue
        if status in _TRULY_TERMINAL_STATUSES:
            dropped.append(tid)
            continue
        survivors.append(t)
    if dropped:
        PENDING[:] = survivors
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "pending_cancelled_dropped", "task_ids": dropped},
        )
