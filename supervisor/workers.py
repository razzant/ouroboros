"""Worker lifecycle, health, and direct-chat handling for the supervisor."""

from __future__ import annotations
import logging
log = logging.getLogger(__name__)

import json
import multiprocessing as mp
import os
import pathlib
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from supervisor.state import load_state, append_jsonl, reconstruct_task_cost
from supervisor.message_bus import coerce_chat_identity, send_with_budget
from ouroboros.config import DATA_DIR, REPO_DIR as CONFIG_REPO_DIR
from ouroboros.outcomes import EXECUTION_FAILED, EXECUTION_INFRA_FAILED, terminal_outcome_axes
from ouroboros.utils import utc_now_iso


REPO_DIR: pathlib.Path = pathlib.Path(CONFIG_REPO_DIR)
DRIVE_ROOT: pathlib.Path = pathlib.Path(DATA_DIR)
MAX_WORKERS: int = 10
SOFT_TIMEOUT_SEC: int = 600
HARD_TIMEOUT_SEC: int = 1800
HEARTBEAT_STALE_SEC: int = 120
QUEUE_MAX_RETRIES: int = 1
TOTAL_BUDGET_LIMIT: float = 0.0
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
         soft_timeout: int, hard_timeout: int, total_budget_limit: float,
         branch_dev: str = "ouroboros", branch_stable: str = "ouroboros-stable") -> None:
    global REPO_DIR, DRIVE_ROOT, MAX_WORKERS, SOFT_TIMEOUT_SEC, HARD_TIMEOUT_SEC
    global TOTAL_BUDGET_LIMIT, BRANCH_DEV, BRANCH_STABLE
    REPO_DIR = repo_dir
    DRIVE_ROOT = drive_root
    MAX_WORKERS = max_workers
    SOFT_TIMEOUT_SEC = soft_timeout
    HARD_TIMEOUT_SEC = hard_timeout
    TOTAL_BUDGET_LIMIT = total_budget_limit
    BRANCH_DEV = branch_dev
    BRANCH_STABLE = branch_stable

    from supervisor import queue
    queue.init(drive_root, soft_timeout, hard_timeout)
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
_WORKER_LIFECYCLE_LOCK = threading.RLock()


def _serialized_worker_lifecycle(fn):
    def wrapped(*args, **kwargs):
        with _WORKER_LIFECYCLE_LOCK:
            return fn(*args, **kwargs)

    return wrapped


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


def _origin_from_mapping(mapping: Any, *, absent: str) -> dict:
    """Typed binding origin from an event/metadata mapping (ref passed BY VALUE
    from chat ingress; ``absent`` is the closed-enum reason when none rode along)."""
    source = mapping if isinstance(mapping, dict) else {}
    ref = source.get("origin_message_ref") or source.get("source_ref")
    if isinstance(ref, dict) and ref:
        text = source.get("origin_message_text") or source.get("source_text")
        origin = {"ref": dict(ref)}
        if isinstance(text, str) and text:
            origin["text"] = text
        return origin
    return {"absent": absent}


def _origin_from_task_record(task_id: str) -> Optional[dict]:
    """Ingress-captured origin from the persisted task record.

    A QUEUED task's ctx.task_metadata does not carry the origin (only the task
    dict/record does), so the mid-run ensure_project_scope bind falls back to
    the durable record — mirroring the UI convert path's _owner_task_origin."""
    try:
        # Child-merging reader: a forked/workspace root persists its RUNNING
        # record on its CHILD drive; the effective-status SSOT merges it (same
        # reason gateway/projects.py::_owner_task_origin uses it).
        from ouroboros.task_status import load_effective_task_result

        record = load_effective_task_result(DRIVE_ROOT, task_id) or {}
        ref = record.get("origin_message_ref")
        text = record.get("origin_message_text")
        if isinstance(ref, dict) and ref and isinstance(text, str) and text.strip():
            return {"ref": dict(ref), "text": text}
    except Exception:
        log.debug("origin task-record lookup failed for %s", task_id, exc_info=True)
    return None


def _report_binding_failure(task_id: str, project_id: str, exc: Exception, *, path: str) -> None:
    """A failed durable bind is LOUD (BIBLE P1: silent linkage loss is memory
    loss): warning log + typed events.jsonl row; the task itself keeps running."""
    log.warning("bind_task_to_project failed for %s/%s (%s)", task_id, project_id, path, exc_info=True)
    try:
        append_jsonl(DRIVE_ROOT / "logs" / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "project_binding_failed",
            "task_id": str(task_id or ""),
            "project_id": str(project_id or ""),
            "bind_path": path,
            "error": f"{type(exc).__name__}: {exc}",
        })
    except Exception:
        log.debug("project_binding_failed event write failed", exc_info=True)


def _canonical_promoted_repair_constraint(value: Any) -> tuple[Optional[dict], str]:
    """Pin and validate the authority envelope for a promoted skill repair."""
    from ouroboros.contracts.skill_payload_policy import resolve_constrained_payload_path
    from ouroboros.contracts.task_constraint import TaskConstraint, normalize_task_constraint

    constraint = normalize_task_constraint(value)
    if constraint is None or constraint.mode != "skill_repair":
        return None, ""
    canonical = TaskConstraint(
        mode="skill_repair",
        skill_name=constraint.skill_name,
        payload_root=constraint.payload_root,
        allow_enable=False,
        allow_review=True,
    )
    try:
        payload_dir = resolve_constrained_payload_path(DRIVE_ROOT, canonical, ".")
    except (TypeError, ValueError):
        return None, "invalid_skill_repair_constraint"
    if not payload_dir.is_dir():
        return None, "skill_repair_payload_missing"
    return {
        "mode": canonical.mode,
        "skill_name": canonical.skill_name,
        "payload_root": canonical.payload_root,
        "allow_enable": False,
        "allow_review": True,
    }, ""


def _promote_duplicate_reason(task_id: str, ctx: Any) -> str:
    """Fail closed if a promoted id is already live, durable, or uncheckable."""
    pending = getattr(ctx, "PENDING", PENDING)
    running = getattr(ctx, "RUNNING", RUNNING)
    with _queue_lock:
        live_duplicate = any(
            isinstance(row, dict) and str(row.get("id") or "") == task_id
            for row in list(pending or [])
        ) or task_id in (running or {})
    try:
        from ouroboros.task_results import load_task_result

        stored_duplicate = bool(
            load_task_result(getattr(ctx, "DRIVE_ROOT", DRIVE_ROOT), task_id)
        )
    except Exception:
        log.warning("promote: duplicate-id lookup failed for %s", task_id, exc_info=True)
        return "task_id_lookup_failed"
    return "duplicate_task_id" if live_duplicate or stored_duplicate else ""


def _promoted_force_plan_metadata(evt: dict) -> dict:
    if evt.get("force_plan") is not True:
        return {}
    source = str(evt.get("force_plan_source") or "operator").strip() or "operator"
    return {"metadata": {"force_plan": True, "force_plan_source": source}}


def promote_chat_to_task(evt: dict, ctx: Any) -> dict:
    """Enqueue a first-class pooled owner task from a conversation-lane promote.
    The task carries the originating ``chat_id`` (its live card and replies
    land in that thread) and the optional ``project_id`` scope; it competes for
    the project writer lease like any other top-level project task.
    """
    from ouroboros.contracts.task_contract import attach_task_contract

    tid = str(evt.get("task_id") or uuid.uuid4().hex[:16])
    admission_token = str(evt.get("routing_token") or "").strip()
    objective = str(evt.get("objective") or "").strip()
    if not objective:
        return {"status": "needs_manual_target", "reason": "empty_objective", "task_id": tid}
    # Reject before project/source/workspace side effects. enqueue_task repeats
    # the check atomically for the tiny race before queue insertion.
    duplicate_reason = _promote_duplicate_reason(tid, ctx)
    if duplicate_reason:
        return {
            "status": "needs_manual_target",
            "reason": duplicate_reason,
            "task_id": tid,
        }

    evt = dict(evt)
    source_note = str(evt.get("_source_note") or "")
    effective_pid = str(evt.get("project_id") or "")
    repair_constraint, constraint_error = _canonical_promoted_repair_constraint(
        evt.get("task_constraint")
    )
    if constraint_error:
        return {
            "status": "needs_manual_target",
            "reason": constraint_error,
            "task_id": tid,
        }
    try:
        chat_id = int(evt.get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    if not chat_id:
        st = ctx.load_state()
        try:
            chat_id = int(st.get("owner_chat_id") or 0)
        except (TypeError, ValueError):
            chat_id = 0
    expected_output = str(evt.get("expected_output") or "").strip()
    text = objective if not expected_output else f"{objective}\n\nExpected output: {expected_output}"
    # Short human title the model coined at card creation (owner P1) — reused as the
    # project name on a later "turn into project" conversion; never the bare task id.
    title = str(evt.get("title") or "").strip()[:80]
    task = {
        "id": tid,
        "type": "task",
        "chat_id": chat_id,
        "text": text,
        "description": objective,
        "objective": objective,
        "expected_output": expected_output,
        "title": title,
        "source": "promote_chat_to_task",
        "_require_unique_task_id": True,
        "_require_worker_pool": True,
        "_admission_token": admission_token,
        "promotion_admission_token": admission_token,
        **_promoted_force_plan_metadata(evt),
    }
    if repair_constraint is not None:
        # Must be present before attach_task_contract so the managed root task
        # enters execution with its confined repair profile, never ephemeral.
        task["task_constraint"] = repair_constraint
    # Ingress-captured origin identity rides the task record (post-hoc UI convert
    # reads it from the persisted result — never re-derived from content).
    if isinstance(evt.get("source_ref"), dict) and evt.get("source_ref"):
        task["origin_message_ref"] = dict(evt["source_ref"])
        if isinstance(evt.get("source_text"), str) and evt.get("source_text"):
            task["origin_message_text"] = evt["source_text"]
    pid = str(evt.get("project_id") or "").strip()
    if pid:
        # Deletion closes admission before cancellation/quiescence begins. Check
        # the durable lifecycle before creating child drives or staging uploads;
        # enqueue_task repeats this check atomically under the queue lock.
        try:
            from ouroboros.projects_registry import get_reserved_project

            existing_project = get_reserved_project(DRIVE_ROOT, pid)
            existing_lifecycle = str((existing_project or {}).get("lifecycle") or "active")
            if existing_project is not None and existing_lifecycle != "active":
                return {
                    "status": "needs_manual_target",
                    "reason": "project_routing_fence",
                    "project_lifecycle": existing_lifecycle,
                    "task_id": tid,
                }
        except Exception:
            log.warning("promote: project admission lookup failed for %s", pid, exc_info=True)
            return {
                "status": "needs_manual_target",
                "reason": "project_routing_fence_lookup_failed",
                "task_id": tid,
            }
        task["project_id"] = pid
        # When the model is CREATING a named project (project_name set), pass the
        # human display name so the project isn't named after its bare id (v6.33.0).
        project_display_name = str(evt.get("project_name") or "").strip()
        try:
            from ouroboros.projects_registry import bind_task_to_project, create_project, touch_project

            project = create_project(
                DRIVE_ROOT, pid, name=project_display_name, origin="promote_chat_to_task",
            )
            touch_project(DRIVE_ROOT, pid)
            # Bind the task to its project (durable task->project map). Without this
            # the task is project-scoped only in its own metadata; the frontend (via
            # all_task_bindings in /api/state) and the mailbox follow-up router
            # (project_chat_for_task) can't recognise it as a project task, so it
            # surfaces in the main chat with a stray "turn into project" button (P2).
            try:
                # Absence semantics by PROVENANCE (structural, never keyword):
                # a chat-born event carries client_message_id, so a missing ref
                # there is a producer BUG (grep-able producer_missing_ref); an
                # event from a context with no owner message (headless/scheduled/
                # consciousness promote) is a DESIGNED absence.
                absent_reason = (
                    "producer_missing_ref"
                    if str(evt.get("client_message_id") or "").strip()
                    and not evt.get("origin_suppressed")
                    else "mid_task_no_origin"
                )
                bind_task_to_project(
                    DRIVE_ROOT,
                    tid,
                    pid,
                    (project or {}).get("chat_id"),
                    origin=_origin_from_mapping(evt, absent=absent_reason),
                )
            except Exception as exc:
                _report_binding_failure(tid, pid, exc, path="promote_chat_to_task")
                return {
                    "status": "needs_manual_target",
                    "reason": "project_binding_failed",
                    "task_id": tid,
                }
            # The promoted task runs in the PROJECT thread: route its live card +
            # owner mailbox to the project's chat_id (not the main chat it was
            # promoted from) so follow-ups steer to it via
            # _route_project_chat_to_running_task and its progress is visible in
            # the project panel.
            try:
                proj_chat = int((project or {}).get("chat_id") or 0)
            except (TypeError, ValueError):
                proj_chat = 0
            if proj_chat:
                task["chat_id"] = proj_chat
                # The agent just created/bound this project server-side (no client
                # round-trip, unlike the UI "Turn into project" flow). Tell the
                # frontend so it refreshes projectChatIds NOW — otherwise this new
                # project's live frames render in the main chat until the periodic
                # /api/state poll catches up (≤20s) and isMyThread misclassifies them.
                try:
                    from supervisor.message_bus import get_bridge

                    get_bridge().broadcast({"type": "projects_changed", "project_id": pid, "chat_id": proj_chat})
                except Exception:
                    log.debug("promote: projects_changed broadcast failed for %s", pid, exc_info=True)
        except Exception:
            log.warning("promote: project registration failed for %s", pid, exc_info=True)
            return {
                "status": "needs_manual_target",
                "reason": "project_registration_failed",
                "task_id": tid,
            }
    # Workspace admission (v6.58.0 SSOT + the Q10=A auto-provision) lives in one
    # helper so this entry point stays readable and under the method gate.
    workspace_outcome = _admit_promoted_workspace(evt, ctx, task, pid=pid, tid=tid)
    if workspace_outcome is not None:
        return workspace_outcome
    attachment_uploads = (
        evt.get("attachment_uploads") if isinstance(evt.get("attachment_uploads"), list) else []
    )
    if attachment_uploads:
        try:
            from ouroboros.artifacts import stage_task_attachments
            from ouroboros.gateway.tasks import _render_attachment_lines

            attachment_root = pathlib.Path(str(task.get("drive_root") or DRIVE_ROOT))
            manifest = stage_task_attachments(attachment_root, tid, attachment_uploads)
            rendered = _render_attachment_lines(manifest)
            if rendered:
                task["text"] = f"{task['text']}\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]"
                task["attachment_images"] = [item for item in manifest if item.get("is_image")]
        except Exception:
            log.warning("promote: attachment staging failed for %s", tid, exc_info=True)
    attach_task_contract(task)
    admitted = ctx.enqueue_task(task)
    if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
        return {
            "status": "needs_manual_target",
            "reason": str(admitted.get("_admission_blocked") or "admission_fence"),
            "project_lifecycle": str(admitted.get("_project_lifecycle") or ""),
            "task_id": tid,
        }
    # A positive promote confirmation is allowed only after the durable queue
    # projection exists.  The event handler writes the scheduled task result
    # after the routing receipt; keeping that last step outside this function
    # makes the result itself the cross-process admission receipt.
    persist_snapshot = getattr(ctx, "persist_queue_snapshot", None)
    if not callable(persist_snapshot):
        return {
            "status": "needs_manual_target",
            "reason": "queue_snapshot_persist_unavailable",
            "task_id": tid,
            "admission_started": True,
        }
    try:
        if persist_snapshot(reason="promote_chat_to_task") is False:
            return {
                "status": "needs_manual_target",
                "reason": "queue_snapshot_persist_failed",
                "task_id": tid,
                "admission_started": True,
            }
    except Exception:
        log.warning("promote: queue snapshot persist failed for %s", tid, exc_info=True)
        return {
            "status": "needs_manual_target",
            "reason": "queue_snapshot_persist_failed",
            "task_id": tid,
            "admission_started": True,
        }
    # v6.82 (P5) disclosed residual: a PROMOTED root carries the host-attested
    # `cancelable` marker from its first RUNNING relay, not from enqueue — the
    # promote path emits no owner-facing progress frame of its own, and minting a
    # marker-only bubble would either add chat noise or bypass the canonical
    # message seam (tests/test_heartbeat_presentation.py). While it is still
    # PENDING the Dashboard Activity row cancels it; the card action appears once
    # it starts.
    outcome = {"status": "scheduled", "task_id": tid}
    if effective_pid:
        outcome["project_id"] = effective_pid
    if source_note:
        outcome["source_note"] = source_note
    return outcome


def _admit_promoted_workspace(evt: dict, ctx: Any, task: dict, *, pid: str, tid: str) -> Optional[dict]:
    """Bind the promoted task's active workspace, or return a failure outcome.

    Extracted verbatim from ``promote_chat_to_task`` (v6.90.x submarine unwind) to
    keep that function under the hard method gate; the admission SEQUENCE is
    unchanged. Returns ``None`` when the task was bound (or legitimately has no
    workspace) and mutates ``task`` in place; returns a ``needs_manual_target``
    outcome dict when admission must fail LOUDLY.
    """
    # v6.58.0 (slice 1) — the promote path admits a workspace through the SAME SSOT
    # as /api/tasks. A task born in a project ROOM defaults to the room's registered
    # working_dir (workspace="none" on the event opts out); a SET-but-broken
    # working_dir fails LOUDLY here — never a silent workspace-less task that would
    # resolve to the self_modification profile over the system repo.
    from ouroboros.workspace_admission import (
        GIT_INIT_REQUIRED,
        WORKSPACE_NONE,
        bounded_workspace_preflight,
        compose_workspace_block,
        resolve_room_workspace,
    )

    # Q10=A (owner, 2026-08-08): a project promoted with NO working folder gets one
    # AUTO-PROVISIONED via the existing ensure_project_workspace seam (an idempotent
    # standalone git repo under the durable subagent_projects root — passes the same
    # validate_workspace_root SSOT below). This binds the task's real tree as its
    # active workspace, fixing path/cwd confinement, the external tool profile and
    # the one-writer lease for the file-less project class (the submarine shape).
    # STRICTLY empty-only: a NON-EMPTY working_dir — valid or broken — is never
    # blind-ensured over (a broken one must LOUD-FAIL through resolve_room_workspace,
    # the v6.58.0 invariant, not be papered over with a fresh empty repo). The
    # workspace="none" sentinel still opts out entirely. Docs are NOT part of this
    # decision: since D-ARCH (2026-08-08) the doc matrix keys on project membership
    # and the owner mode, so binding a workspace here never drags ARCHITECTURE.md
    # out of a max context.
    if (
        pid
        and not str(evt.get("workspace_root") or "").strip()
        and str(evt.get("workspace") or "").strip().lower() != WORKSPACE_NONE
    ):
        provisioned_now = ""
        try:
            from ouroboros.projects_registry import get_project as _get_project_entry

            _existing_wd = str((_get_project_entry(DRIVE_ROOT, pid) or {}).get("working_dir") or "").strip()
        except Exception:
            # Registry read failure: do NOT provision (a blind ensure here could
            # mint a fresh empty repo over a project whose working_dir merely
            # failed to load). resolve_room_workspace re-reads and decides.
            _existing_wd = "unreadable"
            log.warning("promote: project working_dir lookup failed for %s", pid, exc_info=True)
        if not _existing_wd:
            try:
                from ouroboros.projects_registry import ensure_project_workspace

                provisioned_now = str(ensure_project_workspace(DRIVE_ROOT, pid, REPO_DIR) or "")
            except Exception:
                provisioned_now = ""
                log.warning("promote: workspace auto-provisioning raised for %s", pid, exc_info=True)
            if not provisioned_now:
                # Bind-or-fail (v6.58.0): falling through to a workspace-less
                # self_modification-profile task over the system repo is exactly
                # the silent degradation the admission SSOT exists to kill.
                _halt_promoted_task_loudly(
                    ctx, task,
                    f"project {pid!r} has no working folder and auto-provisioning one failed; "
                    "see the supervisor log (ensure_project_workspace)",
                )
                return {
                    "status": "needs_manual_target",
                    "reason": "workspace_provisioning_failed",
                    "task_id": tid,
                }
            task.setdefault("metadata", {})["workspace_autoprovisioned"] = True

    resolved_ws, ws_error, ws_decision = resolve_room_workspace(
        drive_root=DRIVE_ROOT,
        system_repo_dir=REPO_DIR,
        project_id=pid,
        explicit_workspace=str(evt.get("workspace_root") or "").strip(),
        workspace_sentinel=str(evt.get("workspace") or ""),
        # WHICH thread's room this task was born in (A7). A thread that branched
        # off works in its own checkout, and its tasks have to be admitted into
        # THAT folder or they take the project folder's writer lane and queue
        # behind it — branching would buy the owner a second copy of their files
        # and no concurrency at all. Read from the EVENT: `task["chat_id"]` is
        # rewritten to the project's own chat further up when a project is bound
        # here, so by this point it can no longer name the room.
        room_chat_id=evt.get("chat_id"),
    )
    if ws_decision:
        # A12: the owner's folder is untracked. STOP before queueing (never auto-init
        # in someone else's folder) and hand the decision up typed, with the same
        # plain-language offer the gateway surface serves.
        _halt_promoted_task_loudly(
            ctx, task, str(ws_decision.get("message") or ""),
            reason_code=GIT_INIT_REQUIRED,
            banner="GIT_INIT_REQUIRED",
            advice=(
                "Say yes in Projects → this project to start tracking the folder, or "
                "re-promote with workspace='none' for a folder-less task."
            ),
        )
        return {
            "status": "needs_manual_target",
            "reason": GIT_INIT_REQUIRED,
            "decision": ws_decision,
            "task_id": tid,
        }
    if ws_error:
        _halt_promoted_task_loudly(ctx, task, ws_error)
        return {"status": "needs_manual_target", "reason": "workspace_unusable", "task_id": tid}
    if resolved_ws:
        task["workspace_root"] = resolved_ws
        task["workspace_mode"] = "external"
        task["memory_mode"] = "forked"
        # The lease lane keys off task["project_id"]: for a project room it is already
        # set; for a bare workspace promote, resolve it (registry-first → derived hash)
        # so one folder is one serialized lane on EVERY entry path (slice 0 invariant).
        if not str(task.get("project_id") or "").strip():
            try:
                from ouroboros.project_facts import resolve_project_id as _resolve_pid

                derived_pid = _resolve_pid({"workspace_root": resolved_ws})
                if derived_pid:
                    task["project_id"] = derived_pid
            except Exception:
                log.debug("promote: project_id derivation failed for %s", tid, exc_info=True)
        # Memory-fork parity with /api/tasks: the room task runs on an ISOLATED child
        # drive (forked seed), with the canonical root kept for budget/status.
        try:
            from ouroboros.headless import prepare_task_drive

            child_drive = prepare_task_drive(
                DRIVE_ROOT, tid, "forked", project_id=str(task.get("project_id") or "")
            )
            if child_drive is not None:
                task["drive_root"] = str(child_drive)
                task["budget_drive_root"] = str(DRIVE_ROOT)
        except Exception:
            log.warning("promote: child drive fork failed for %s", tid, exc_info=True)
        # Preflight parity, HARD-CAPPED: this runs on the supervisor event-drain
        # thread, so the git/toolchain snapshot gets a bounded window and degrades
        # to a disclosed skip note instead of stalling event delivery.
        preflight_summary = bounded_workspace_preflight(resolved_ws)
        metadata = task.setdefault("metadata", {})
        metadata["workspace_root"] = resolved_ws
        metadata["workspace_preflight"] = preflight_summary
        task["text"] = (
            f"{task['text']}\n\n[HEADLESS_WORKSPACE]\n"
            + compose_workspace_block(
                workspace_root=resolved_ws,
                workspace_mode="external",
                memory_mode="forked",
                workspace_preflight=preflight_summary,
            )
            + "[END_HEADLESS_WORKSPACE]"
        )
    return None


def _halt_promoted_task_loudly(
    ctx: Any,
    task: dict,
    ws_error: str,
    *,
    reason_code: str = "workspace_unusable",
    banner: str = "WORKSPACE_UNUSABLE",
    advice: str = (
        "Fix the project's working folder (Projects → this project) or re-promote with "
        "workspace='none' for a folder-less task."
    ),
) -> None:
    """v6.58.0 loud-fail invariant: a room task whose workspace is SET-but-unusable
    is terminally FAILED at admission with a visible card + chat message — never
    silently admitted workspace-less (which would run the self_modification profile
    over the system repo). Never raises.

    The banner/reason are parameters because A12 added a SECOND reason a promoted
    task legitimately does not start: the folder is untracked and the owner has not
    answered the git offer yet. That is not a breakage, and labelling it
    `workspace_unusable` would tell the owner their folder is broken when the only
    thing missing is their answer."""
    tid = str(task.get("id") or "")
    chat_id = 0
    try:
        chat_id = int(task.get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    message = f"⚠️ {banner}: task {tid} was NOT started — {ws_error} {advice}"
    try:
        from ouroboros.task_results import STATUS_FAILED, write_task_result

        write_task_result(
            DRIVE_ROOT, tid, STATUS_FAILED,
            reason_code=reason_code,
            result=message,
            description=str(task.get("description") or ""),
            chat_id=chat_id,
            project_id=str(task.get("project_id") or ""),
        )
    except Exception:
        log.warning("promote loud-fail: task_result write failed for %s", tid, exc_info=True)
    try:
        if chat_id:
            ctx.send_with_budget(chat_id, message)
    except Exception:
        log.debug("promote loud-fail: chat message failed for %s", tid, exc_info=True)


def _running_task_workspace(ctx: Any, task_id: str) -> str:
    """The workspace_root of a LIVE task, read from the supervisor RUNNING map.

    A self-scoping task is by definition still running, so the in-memory record is
    the authority — its persisted result does not exist yet. Never raises."""
    try:
        running = getattr(ctx, "RUNNING", None)
        row = running.get(task_id) if isinstance(running, dict) else None
        task = row.get("task") if isinstance(row, dict) else None
        return str((task or {}).get("workspace_root") or "").strip()
    except Exception:
        log.debug("_running_task_workspace failed for %s", task_id, exc_info=True)
        return ""


def ensure_project_scope(evt: dict, ctx: Any) -> None:
    """Create/attach the registry project for an in-task ensure_project_scope call
    and bind the CURRENT (already-running) task to it, then broadcast so the UI moves
    the card into the project thread. Mirrors the project-registration half of
    promote_chat_to_task, but for a task that already exists (the worker has already
    set ctx.project_id locally; this makes it durable + visible)."""
    tid = str(evt.get("task_id") or "").strip()
    pid = str(evt.get("project_id") or "").strip()
    if not tid or not pid:
        return
    name = str(evt.get("project_name") or "").strip()
    try:
        from ouroboros.projects_registry import bind_task_to_project, create_project, touch_project

        project = create_project(DRIVE_ROOT, pid, name=name, origin="ensure_project_scope")
        touch_project(DRIVE_ROOT, pid)
        # A11, mirroring the UI conversion: a task that self-scopes mid-run hands the
        # new project the folder it is ALREADY working in. Otherwise the project is
        # born placeless and its next task provisions a different empty tree, which
        # is the same silent move the card conversion used to make.
        try:
            from ouroboros.projects_registry import adopt_task_workspace

            _adopted, _adopt_error = adopt_task_workspace(
                DRIVE_ROOT, pid, _running_task_workspace(ctx, tid), system_repo_dir=REPO_DIR
            )
            if _adopt_error:
                log.warning("ensure_project_scope: %s", _adopt_error)
        except Exception:
            log.debug("ensure_project_scope: workspace adoption failed for %s", pid, exc_info=True)
        try:
            proj_chat = int((project or {}).get("chat_id") or 0)
        except (TypeError, ValueError):
            proj_chat = 0
        origin = _origin_from_mapping(evt, absent="mid_task_no_origin")
        if "absent" in origin:
            # Queued tasks carry no origin in ctx.task_metadata — the live
            # RUNNING task dict does (and covers forked/workspace roots whose
            # running record lives on a CHILD drive, scope-review r2 advisory).
            running = getattr(ctx, "RUNNING", None)
            row = running.get(tid) if isinstance(running, dict) else None
            task_row = row.get("task") if isinstance(row, dict) else None
            candidate = _origin_from_mapping(task_row, absent="mid_task_no_origin")
            if "ref" in candidate and "text" in candidate:
                origin = candidate
        if "absent" in origin:
            # Last resort: the durable task record on the canonical drive
            # (scope-review r1 critical: the mid-run "make this a project
            # named X" path must keep the start message).
            origin = _origin_from_task_record(tid) or origin
        try:
            bind_task_to_project(DRIVE_ROOT, tid, pid, proj_chat or None, origin=origin)
        except Exception as exc:
            _report_binding_failure(tid, pid, exc, path="ensure_project_scope")
        # Make the one-writer-per-project lease recognize THIS already-running task
        # as a lane occupant: project_lease reads task["project_id"] from the
        # supervisor RUNNING map, which (unlike the promote path that sets it at
        # build time) is NOT set for a mid-flight self-scope. Without this, a task
        # that self-scopes to project X would not hold X's lane and a concurrent
        # X task could be assigned and write the same project. SSOT helper shared
        # with the UI api_project_from_task convert path so the two cannot drift.
        try:
            from ouroboros.project_lease import mark_task_project
            from ouroboros.projects_registry import project_working_dirs

            running = getattr(ctx, "RUNNING", None)
            pending = getattr(ctx, "PENDING", None)
            if isinstance(running, dict):
                # The project->folder map goes in because marking PINS the lane of a
                # RUNNING task: pinned without it, a task that named no folder freezes
                # (pid, "") while every later candidate for the same project resolves
                # to ("", folder) and is admitted alongside it.
                with _queue_lock:
                    mark_task_project(
                        running, pending, tid, pid, project_working_dirs(DRIVE_ROOT)
                    )
        except Exception:
            log.debug("ensure_project_scope: RUNNING project_id update failed for %s", tid, exc_info=True)
        if proj_chat:
            try:
                from supervisor.message_bus import get_bridge

                get_bridge().broadcast({"type": "projects_changed", "project_id": pid, "chat_id": proj_chat})
            except Exception:
                log.debug("ensure_project_scope: projects_changed broadcast failed for %s", pid, exc_info=True)
    except ValueError as exc:
        # The registry-wide chat-id reservation (X1) and the tombstone reservation
        # both refuse a project creation by raising ValueError. That refusal is the
        # LOUD half of "a project collision is refused loudly" — swallowing it into
        # the generic log.debug below turned an unresolvable identity clash into a
        # task that silently keeps running unscoped, which is the exact silent
        # merge the reservation exists to prevent.
        log.error(
            "ensure_project_scope REFUSED for %s: %s — the task stays unscoped; "
            "pick a different project id",
            pid, exc,
        )
    except Exception:
        log.debug("ensure_project_scope: project registration failed for %s", pid, exc_info=True)


def handle_chat_direct(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    with _chat_agent_lock:
        if not _repo_writer_turn_allowed(chat_id):
            return
        _handle_chat_direct_locked(
            chat_id,
            text,
            image_data,
            task_constraint=task_constraint,
            task_metadata=task_metadata,
        )


def _handle_chat_direct_locked(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    from supervisor.state import budget_remaining, load_state
    try:
        remaining = budget_remaining(load_state(), strict=True)
    except Exception:
        send_with_budget(chat_id, "⚠️ Cost accounting is unavailable. Task was not dispatched; retry after ledger recovery.")
        return
    if remaining <= 0:
        try:
            send_with_budget(chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.")
        except Exception:
            pass
        return
        
    _run_chat_task(
        _get_chat_agent(), chat_id, text, image_data,
        task_constraint=task_constraint, task_metadata=task_metadata, ephemeral=False,
    )


def _broadcast_task_named(msg: dict) -> None:
    """Bridge broadcast callback for the proactive namer (kept tiny + fail-soft)."""
    try:
        from supervisor.message_bus import get_bridge

        get_bridge().broadcast(msg)
    except Exception:
        log.debug("task_named broadcast failed", exc_info=True)


def _run_chat_task(
    agent: Any,
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
    *,
    ephemeral: bool = False,
) -> None:
    """Build the direct-chat task and run it on the given agent, draining events.

    ``ephemeral`` marks a SHORT-LIVED same-route turn (run on a separate agent
    instance while the shared chat agent is busy): it carries _ephemeral_turn so
    the task pipeline skips long-term memory / reflection / evolution writes."""
    try:
        from ouroboros.contracts.task_contract import attach_task_contract

        task = {
            "id": uuid.uuid4().hex[:8],
            "type": "task",
            "chat_id": chat_id,
            "text": text,
            "_is_direct_chat": True,
        }
        if ephemeral:
            task["_ephemeral_turn"] = True
        if task_constraint:
            task["task_constraint"] = dict(task_constraint)
        if task_metadata:
            task["metadata"] = dict(task_metadata)
            # The ingress-captured origin identity rides on the TASK RECORD so a
            # later post-hoc "Turn into project" reads it from the persisted
            # result instead of re-deriving identity from content.
            _origin_ref = task_metadata.get("origin_message_ref")
            if isinstance(_origin_ref, dict) and _origin_ref:
                task["origin_message_ref"] = dict(_origin_ref)
                _origin_text = task_metadata.get("origin_message_text")
                if isinstance(_origin_text, str) and _origin_text:
                    task["origin_message_text"] = _origin_text
            # Project-thread conversations scope the direct lane to the
            # project's memory (knowledge/journal/workpad sections).
            pid = str(task_metadata.get("project_id") or "").strip()
            if pid:
                task["project_id"] = pid
                # A real project-thread conversation task is bound to its project so
                # the frontend (all_task_bindings) recognises it and never offers a
                # stray "turn into project" button (P2). Ephemeral same-route turns
                # are transient decisions — never bound.
                if not ephemeral:
                    try:
                        from ouroboros.projects_registry import bind_task_to_project
                        bind_task_to_project(
                            DRIVE_ROOT, task["id"], pid, chat_id,
                            origin=_origin_from_mapping(task_metadata, absent="mid_task_no_origin"),
                        )
                    except Exception as exc:
                        _report_binding_failure(task["id"], pid, exc, path="direct_project_turn")
        if image_data:
            # image_data is (base64, mime) or (base64, mime, caption). The caption
            # still seeds task['text'] (and the legacy inline image path below) so a
            # caption-only message keeps working even when nothing stages.
            task["image_base64"] = image_data[0]
            task["image_mime"] = image_data[1]
            if len(image_data) > 2 and image_data[2]:
                task["image_caption"] = image_data[2]
                if not text:
                    task["text"] = image_data[2]
        # v6.52.0 (P1, full desktop unify): route the WHOLE desktop attachment set
        # (any type) through the shared staging substrate so the agent gets EVERY
        # attachment — images natively via attachment_images + non-images via the
        # read_file(root='artifact_store', path='attachments/...') manifest — exactly
        # like the CLI/API/GAIA path. The uploads are resolved from data/uploads/ in
        # ws._chat_attachment_uploads and carried as task['metadata'] (like force_plan).
        # On a non-empty manifest we DROP the legacy inline image_base64 so the same
        # image is not double-injected; on absent/empty uploads (older clients, the
        # single-image base64 seam) the legacy inline path above stays untouched.
        meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        uploads = meta.get("chat_attachment_uploads")
        if uploads:
            from ouroboros.artifacts import stage_task_attachments
            from ouroboros.gateway.tasks import _render_attachment_lines

            manifest = stage_task_attachments(DRIVE_ROOT, str(task["id"]), uploads)
            if manifest:
                task["drive_root"] = str(DRIVE_ROOT)
                task["attachment_images"] = [m for m in manifest if m.get("is_image")]
                rendered = _render_attachment_lines(manifest)
                if rendered:
                    task["text"] = f"{task.get('text') or ''}\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]"
                task.pop("image_base64", None)
                task.pop("image_mime", None)
        if not task["text"]:
            task["text"] = "(image attached)" if image_data else ""
        # Cluster B: proactively coin a project name for a fresh MAIN-CHAT direct card
        # (not an ephemeral decision turn, not an already-bound project-thread task) so
        # the card shows a human title up front and turn-into-project reuses it.
        if not ephemeral and not task.get("project_id"):
            from ouroboros.project_naming import spawn_proactive_namer

            spawn_proactive_namer(
                DRIVE_ROOT, str(task["id"]), task["text"], broadcast=_broadcast_task_named
            )
        attach_task_contract(task)
        events = agent.handle_task(task)
        for e in events:
            get_event_q().put(e)
    except Exception as e:
        import traceback
        err_msg = f"⚠️ Error: {type(e).__name__}: {e}"
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "direct_chat_error",
                "error": repr(e),
                "traceback": str(traceback.format_exc())[:2000],
            },
        )
        try:
            send_with_budget(chat_id, err_msg)
        except Exception:
            log.debug("Suppressed exception", exc_info=True)


def handle_chat_ephemeral(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> None:
    """The "turn = decision" path (v6.33.0 WS10): when the shared chat agent is
    busy, a new main-chat message runs as a SHORT-LIVED turn on a SEPARATE agent
    instance — bypassing _chat_agent_lock so it never freezes/injects into the
    running turn, while keeping the SAME ROUTE (same make_agent config: model /
    mode / effort, not a cheaper lane). Ephemeral turns are serialized among
    themselves and are barred from long-term memory/reflection/evolution writes."""
    from supervisor.state import budget_remaining, load_state
    try:
        remaining = budget_remaining(load_state(), strict=True)
    except Exception:
        send_with_budget(chat_id, "⚠️ Cost accounting is unavailable. Task was not dispatched; retry after ledger recovery.")
        return
    if remaining <= 0:
        try:
            send_with_budget(chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.")
        except Exception:
            pass
        return
    if not getattr(sys, 'frozen', False):
        sys.path.insert(0, str(REPO_DIR))
    from ouroboros.agent import make_agent

    with _ephemeral_chat_lock:
        if not _repo_writer_turn_allowed(chat_id):
            return
        agent = make_agent(repo_dir=str(REPO_DIR), drive_root=str(DRIVE_ROOT), event_queue=get_event_q())
        _run_chat_task(
            agent, chat_id, text, image_data,
            task_constraint=task_constraint, task_metadata=task_metadata, ephemeral=True,
        )


def auto_resume_after_restart() -> None:
    """Auto-resume after a recent restart when scratchpad still has work."""
    try:
        owner_restart_flag = DRIVE_ROOT / "state" / "owner_restart_no_resume.flag"
        if owner_restart_flag.exists():
            owner_restart_flag.unlink(missing_ok=True)
            panic_compat_flag = DRIVE_ROOT / "state" / "panic_stop.flag"
            try:
                if panic_compat_flag.read_text(encoding="utf-8").strip() == "owner_restart_no_resume":
                    panic_compat_flag.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except Exception:
                log.debug("Failed to consume owner restart compatibility flag", exc_info=True)
            log.info("Owner restart flag detected — skipping auto-resume.")
            return

        # Panic/owner-restart flags suppress auto-resume and are consumed.
        panic_flag = DRIVE_ROOT / "state" / "panic_stop.flag"
        if panic_flag.exists():
            panic_flag.unlink(missing_ok=True)
            log.info("Panic flag detected — skipping auto-resume.")
            return

        st = load_state()
        chat_id = st.get("owner_chat_id")
        if not chat_id:
            return

        restart_verify_path = DRIVE_ROOT / "state" / "pending_restart_verify.json"
        recent_restart = False
        if restart_verify_path.exists():
            recent_restart = True
        else:
            sup_log = DRIVE_ROOT / "logs" / "supervisor.jsonl"
            if sup_log.exists():
                try:
                    lines = sup_log.read_text(encoding="utf-8").strip().split("\n")
                    for line in reversed(lines[-20:]):
                        if not line.strip():
                            continue
                        evt = json.loads(line)
                        if evt.get("type") in ("launcher_start", "restart"):
                            recent_restart = True
                            break
                except Exception:
                    log.debug("Suppressed exception", exc_info=True)

        if not recent_restart:
            return

        scratchpad_path = DRIVE_ROOT / "memory" / "scratchpad.md"
        if not scratchpad_path.exists():
            return

        scratchpad = scratchpad_path.read_text(encoding="utf-8")
        stripped = scratchpad.strip()
        if not stripped or stripped == "# Scratchpad" or "(empty" in stripped.lower():
            content_lines = [
                ln.strip() for ln in stripped.splitlines()
                if ln.strip() and not ln.strip().startswith("#") and ln.strip() != "- (empty)"
            ]
            content_lines = [ln for ln in content_lines if not ln.startswith("UpdatedAt:")]
            if not content_lines:
                return

        time.sleep(2)  # Let everything initialize
        agent = _get_chat_agent()
        if not agent._busy:
            import threading
            threading.Thread(
                target=handle_chat_direct,
                args=(int(chat_id),
                      "[auto-resume after restart] Continue your work. Read scratchpad and identity — they contain context of what you were doing.",
                      None),
                daemon=True,
            ).start()
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "auto_resume_triggered",
                },
            )
    except Exception as e:
        append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(),
            "type": "auto_resume_error",
            "error": repr(e),
        })

# Log types the worker sink does NOT forward: each already reaches the dashboard
# live via a dedicated EVENT_Q sibling/handler, so forwarding the worker's
# append_jsonl copy too would double-broadcast (and task_checkpoint would also be
# re-persisted to events.jsonl by _handle_log_event, a double file write).
WORKER_LOG_SINK_SUPPRESSED_TYPES = frozenset({
    "tool_call", "llm_round", "task_checkpoint", "task_done", "llm_usage",
})


def _current_custody_session_id() -> str:
    """Server-side custody session id to hand to spawned workers (best-effort)."""
    try:
        from ouroboros.process_custody import current_custody_session_id
        return current_custody_session_id()
    except Exception:
        return ""


def _bind_worker_repo_root(repo_dir: str, drive_root: str = "") -> None:
    """Point git_ops' roots at the repo and data dir this worker was told to serve.

    ``git_ops.REPO_DIR`` is a module global with no env fallback, and ``git_ops.init()`` is never
    called at boot, so a worker inherits the hardcoded ``~/Ouroboros/repo`` default. Under the
    spawn start method (macOS/Windows) the child re-imports the module and gets that default even
    when it serves a checkout somewhere else — and ``update_merge._update_tx_marker_path()``
    resolves through it, so the worker's managed-update tool gate would read ANOTHER repo's
    transaction. Bind it from the ``repo_dir`` this worker already receives.

    ``DRIVE_ROOT`` moves with it: the same re-import leaves it on the default data dir, so a
    worker serving a custom install would write git_ops' rescue snapshots and logs under an
    unrelated home directory. Both values are handed to this process; the branch names and
    REMOTE_URL are NOT, which is also why this is a direct assignment rather than
    ``git_ops.init()`` — init() would overwrite them with its own defaults, silently retargeting
    an install whose branches differ. They keep whatever the child imported.
    """
    import pathlib as _pl

    from supervisor import git_ops as _git_ops

    _git_ops.REPO_DIR = _pl.Path(repo_dir)
    if drive_root:
        _git_ops.DRIVE_ROOT = _pl.Path(drive_root)


def _prepare_worker_task_runtime() -> None:
    """Load the managed-update authorization path before a live merge can conflict."""
    import supervisor.update_merge  # noqa: F401


def worker_main(wid: int, in_q: Any, out_q: Any, repo_dir: str, drive_root: str,
                custody_session_id: str = "") -> None:
    import os as _os
    # Mark this process as a worker BEFORE importing the agent/LLM stack so the
    # central network-transport policy disables system proxy resolution
    # (trust_env=False) for every HTTP client created here. This is the
    # fork-safety guard (no _scproxy/SCDynamicStoreCopyProxies on the child side
    # of fork) and a clean default for spawned workers too.
    _os.environ["OUROBOROS_IN_WORKER"] = "1"
    # Before ANY import that resolves the update-tx marker through git_ops (see
    # _bind_worker_repo_root): a spawned child would otherwise gate on the hardcoded default repo.
    _bind_worker_repo_root(repo_dir, drive_root)
    # Adopt the server's custody session id. Under the 'spawn' start method this
    # process re-imported process_custody and minted a fresh _SESSION_ID; without
    # adopting the server's id, every service/process this worker records looks
    # foreign to the server's reaper and gets killed at the next reap tick —
    # even a still-running task's services. Passed as an arg (not env) so it
    # cannot survive a server re-exec. See process_custody.adopt_session_id.
    if custody_session_id:
        try:
            from ouroboros.process_custody import adopt_session_id
            adopt_session_id(custody_session_id)
        except Exception:
            pass
    from ouroboros.platform_layer import create_new_session
    create_new_session()
    # Lifeline: if the supervisor dies abruptly, this worker is reparented to
    # init and would keep running LLM rounds invisibly — group-suicide instead.
    try:
        from ouroboros.process_custody import start_parent_lifeline

        start_parent_lifeline(label=f"worker-{wid}")
    except Exception:
        pass
    # Stream this worker's append_jsonl log lines to the dashboard Logs panel.
    # The WS log sink lives only in the main process, so without this every
    # worker-task log line (queued/evolution/review/subagent) is written to file
    # but never broadcast live — the "not all logs arrive" gap. Forward over the
    # existing EVENT_Q -> _handle_log_event -> push_log path. Suppress types that
    # already arrive live via a dedicated sibling event (tool_call/llm_round/
    # task_checkpoint) or are appended in the main process (task_done/llm_usage)
    # to avoid double broadcast and (for task_checkpoint) a double file write.
    try:
        from ouroboros.utils import emit_log_event, set_log_sink

        def _worker_log_sink(obj: Any) -> None:
            if isinstance(obj, dict) and str(obj.get("type") or "") in WORKER_LOG_SINK_SUPPRESSED_TYPES:
                return
            emit_log_event(out_q, obj, log_label="worker log")

        set_log_sink(_worker_log_sink)
    except Exception:
        pass
    import sys as _sys
    import traceback as _tb
    import pathlib as _pathlib
    if not getattr(_sys, 'frozen', False):
        _sys.path.insert(0, repo_dir)
    _drive = _pathlib.Path(drive_root)
    # Spawned workers must pin the runtime-mode baseline from the parent env;
    # forked workers inherit it. This keeps the elevation ratchet consistent.
    try:
        from ouroboros.config import initialize_runtime_mode_baseline
        initialize_runtime_mode_baseline()
    except Exception:
        # Non-fatal: save_settings still has env-var fallback gating.
        try:
            _log_worker_crash(wid, _drive, "init_baseline", None, _tb.format_exc())
        except Exception:
            pass
    try:
        from ouroboros.config import get_skills_repo_path, load_settings as _load_settings
        from ouroboros.extension_loader import reload_all as _reload_extensions

        pytest_default_real_data_dir = (
            "pytest" in _sys.modules
            and not _os.environ.get("OUROBOROS_DATA_DIR")
            and _drive.resolve(strict=False) == (_pathlib.Path.home() / "Ouroboros" / "data").resolve(strict=False)
        )
        if pytest_default_real_data_dir:
            try:
                from ouroboros.utils import append_jsonl, utc_now_iso
                append_jsonl(_drive / "logs" / "supervisor.jsonl", {
                    "ts": utc_now_iso(),
                    "type": "worker_extension_reload_skipped",
                    "worker_id": wid,
                    "reason": "pytest_default_real_data_dir",
                })
            except Exception:
                pass
        else:
            _repo_path = get_skills_repo_path()
            _reload_extensions(_drive, _load_settings, repo_path=_repo_path or None)
    except Exception:
        try:
            _log_worker_crash(wid, _drive, "extension_reload", None, _tb.format_exc())
        except Exception:
            pass
    try:
        from ouroboros.agent import make_agent
        agent = make_agent(repo_dir=repo_dir, drive_root=drive_root, event_queue=out_q)
    except Exception as _e:
        _log_worker_crash(wid, _drive, "make_agent", _e, _tb.format_exc())
        return
    try:
        _prepare_worker_task_runtime()
        from ouroboros.utils import append_jsonl as _append_jsonl
        from ouroboros.utils import get_git_info as _get_git_info
        from ouroboros.utils import utc_now_iso as _utc_now_iso

        _branch, _sha = _get_git_info(_pathlib.Path(repo_dir))
        _append_jsonl(_drive / "logs" / "events.jsonl", {
            "ts": _utc_now_iso(), "type": "worker_ready", "worker_id": wid,
            "pid": _os.getpid(), "git_branch": _branch, "git_sha": _sha,
        })
    except Exception as _e:
        _log_worker_crash(wid, _drive, "worker_ready", _e, _tb.format_exc())
    while True:
        try:
            task = in_q.get()
            if task is None or task.get("type") == "shutdown":
                break
            task_drive_root = str(task.get("drive_root") or drive_root)
            if task_drive_root != str(drive_root):
                task_agent = make_agent(
                    repo_dir=repo_dir,
                    drive_root=task_drive_root,
                    event_queue=out_q,
                    budget_drive_root=str(task.get("budget_drive_root") or drive_root),
                )
                events = task_agent.handle_task(task)
            else:
                events = agent.handle_task(task)
            for e in events:
                e2 = dict(e)
                e2["worker_id"] = wid
                out_q.put(e2)
        except Exception as _e:
            _log_worker_crash(wid, _drive, "handle_task", _e, _tb.format_exc())


def _write_failure_result(
    task_id: str,
    reason: str = "Worker process crashed (crash storm). Task was not completed.",
    status: str = "",
) -> str:
    """Write failure result for a crashed/orphaned task.

    Returns the FINAL persisted status: if the task already reached a terminal
    state, the monotonic guard preserves it and that existing status is returned
    (so the UI event matches disk); otherwise the written failure status.
    """
    if not task_id:
        return ""
    try:
        from ouroboros.task_results import (
            STATUS_FAILED, STATUS_COMPLETED, STATUS_REJECTED_DUPLICATE,
            STATUS_CANCELLED, load_task_result, write_task_result,
        )
        # STATUS_INTERRUPTED is not final; it is written before requeue.
        _FINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_REJECTED_DUPLICATE, STATUS_CANCELLED}
        existing = load_task_result(DRIVE_ROOT, task_id)
        if existing and existing.get("status") in _FINAL_STATUSES:
            return str(existing.get("status") or "")
        final_status = status or STATUS_FAILED
        # Reconstruct from durable llm_usage so an abnormally-finalized task does
        # not record zero cost/rounds (understating per-task + campaign metrics).
        f_cost_fields = reconstruct_task_cost(str(task_id), fields=True)
        write_task_result(
            DRIVE_ROOT,
            task_id,
            final_status,
            result=reason,
            reason_code="worker_terminal_failure" if final_status == STATUS_FAILED else str(final_status or ""),
            outcome_axes=terminal_outcome_axes(
                lifecycle=final_status,
                execution=EXECUTION_INFRA_FAILED if final_status == STATUS_FAILED else str(final_status or ""),
                reason_code="worker_terminal_failure" if final_status == STATUS_FAILED else str(final_status or ""),
                review_trigger="worker_terminal",
            ),
            **f_cost_fields,
        )
        return final_status
    except Exception:
        log.warning("Failed to write failure result for task %s", task_id, exc_info=True)
        raise


def terminal_task_metadata(task_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Project ONLY lifecycle-relevant metadata onto a terminal task_done event.

    Terminal events reach chat logs and the UI, so arbitrary task metadata
    (workspace paths, secret-bearing fields) must not ride along. Exactly two
    consumers need fields here: the evolution campaign tally reads
    ``evolution_transaction``, and the assisted-merge watchdog / writer-gate
    release in events._handle_task_done reads ``managed_update`` (its
    authority_fingerprint) — a reaped resolver task would otherwise leave the
    update tx orphaned and the writer gate latched until restart."""
    meta = task_metadata if isinstance(task_metadata, dict) else {}
    out: Dict[str, Any] = {}
    for key in ("evolution_transaction", "managed_update"):
        value = meta.get(key)
        if isinstance(value, dict):
            out[key] = dict(value)
    return out


def _emit_task_done_terminal(
    task: Optional[Dict[str, Any]],
    task_id: str,
    status: str = "failed",
    *,
    reason_code: str = "",
    cost_fields: Optional[Dict[str, Any]] = None,
) -> bool:
    """Emit a task_done event so the UI resolves the live card when a task is
    torn down outside the normal completion path (crash storm, kill, hard
    timeout). Without this the spinner spins forever on these paths.

    ``cost_fields`` is one whole ``reconstruct_task_cost(fields=True)`` projection,
    taken opaquely (as ``queue._emit_cancel_task_done`` already takes it) rather
    than re-declared field by field. Three times a key was added to that
    projection and a hand-maintained mirror here was missed; a signature that
    names no cost field cannot be missed again. Callers with no reconstructed
    cost pass nothing and the event says so instead of reporting zeros as fact."""
    if not task_id:
        return False
    try:
        chat_id = int((task or {}).get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    status = status or "failed"
    # Caller reason_code wins; budget_exhausted -> EXECUTION_FAILED below, not infra-failure.
    reason_code = reason_code or ("worker_terminal_failure" if status == "failed" else status)
    task_metadata = (task or {}).get("metadata")
    task_metadata = task_metadata if isinstance(task_metadata, dict) else {}
    terminal_metadata = terminal_task_metadata(task_metadata)
    try:
        # Only the four keys whose EMISSION RULE differs are read by name: the
        # accounting verdict always rides, the two disclosure flags ride only
        # when they have something to disclose, and everything else rides only
        # when the accounting is available -- so an unavailable projection never
        # publishes its `None` placeholders as if they were measurements.
        projection: Dict[str, Any] = dict(cost_fields or {})
        emitted: Dict[str, Any] = {
            "cost_accounting_status": str(projection.pop("cost_accounting_status", "") or "unavailable"),
            "cost_final": bool(projection.pop("cost_final", False)),
        }
        accounting_error = projection.pop("cost_accounting_error", "")
        if accounting_error:
            emitted["cost_accounting_error"] = accounting_error
        if projection.pop("ledger_integrity_degraded", False):
            emitted["ledger_integrity_degraded"] = True
        if emitted["cost_accounting_status"] == "available":
            # Verbatim, unenumerated: cost_final's disclosed cause (non_final_rows)
            # rides here today for free, and so will the next field added upstream.
            emitted.update(projection)
        get_event_q().put({
            "type": "task_done",
            "task_id": str(task_id),
            "task_type": str((task or {}).get("type") or ""),
            "chat_id": chat_id,
            "status": status,
            "outcome_axes": terminal_outcome_axes(
                lifecycle=status,
                execution=(EXECUTION_FAILED if reason_code == "budget_exhausted" else EXECUTION_INFRA_FAILED) if status == "failed" else status,
                reason_code=reason_code,
                review_trigger="worker_terminal",
            ),
            "reason_code": reason_code,
            **({"metadata": terminal_metadata} if terminal_metadata else {}),
            **emitted,
        })
        return True
    except Exception:
        log.warning("Failed to emit terminal task_done for %s", task_id, exc_info=True)
        return False


def _log_worker_crash(wid: int, drive_root: pathlib.Path, phase: str, exc: Exception, tb: str) -> None:
    """Best-effort worker-side crash logging."""
    import os as _os
    try:
        path = drive_root / "logs" / "supervisor.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = json.dumps({
            "ts": utc_now_iso(),
            "type": "worker_crash",
            "worker_id": wid,
            "pid": _os.getpid(),
            "phase": phase,
            "error": repr(exc),
            "traceback": str(tb)[:3000],
        }, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        log.debug("Suppressed exception", exc_info=True)


def _first_worker_event_since(
    offset_bytes: int, event_type: str = "worker_boot"
) -> Optional[Dict[str, Any]]:
    """Read the first event of one worker lifecycle type after a file offset."""
    path = DRIVE_ROOT / "logs" / "events.jsonl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            safe_offset = offset_bytes if 0 <= offset_bytes <= size else 0
            f.seek(safe_offset)
            data = f.read().decode("utf-8", errors="replace")
    except Exception:
        log.debug("Suppressed exception", exc_info=True)
        return None

    for line in data.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            evt = json.loads(raw)
        except Exception:
            log.debug("Suppressed exception in loop", exc_info=True)
            continue
        if isinstance(evt, dict) and str(evt.get("type") or "") == event_type:
            return evt
    return None


def _first_worker_boot_event_since(offset_bytes: int) -> Optional[Dict[str, Any]]:
    return _first_worker_event_since(offset_bytes, "worker_boot")


def _verify_worker_sha_after_spawn(events_offset: int, timeout_sec: float = 90.0) -> None:
    """Verify newly spawned workers booted at expected current_sha."""
    st = load_state()
    expected_sha = str(st.get("current_sha") or "").strip()
    if not expected_sha:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "worker_sha_verify_skipped",
                "reason": "missing_current_sha",
            },
        )
        return

    deadline = time.time() + max(float(timeout_sec), 1.0)
    boot_evt = None
    while time.time() < deadline:
        boot_evt = _first_worker_boot_event_since(events_offset)
        if boot_evt is not None:
            break
        time.sleep(0.25)

    if boot_evt is None:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "worker_sha_verify_timeout",
                "expected_sha": expected_sha,
            },
        )
        return

    observed_sha = str(boot_evt.get("git_sha") or "").strip()
    ok = bool(observed_sha and observed_sha == expected_sha)
    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "worker_sha_verify",
            "ok": ok,
            "expected_sha": expected_sha,
            "observed_sha": observed_sha,
            "worker_pid": boot_evt.get("pid"),
        },
    )
    if not ok and st.get("owner_chat_id"):
        send_with_budget(
            int(st["owner_chat_id"]),
            f"⚠️ Worker SHA mismatch after spawn: expected {expected_sha[:8]}, got {(observed_sha or 'unknown')[:8]}",
        )


_WORKER_PIDS_FILENAME = "worker_pids.json"


def _worker_pids_path() -> pathlib.Path:
    return DRIVE_ROOT / "state" / _WORKER_PIDS_FILENAME


def _record_worker_pids() -> None:
    """Persist current worker PIDs so a later server instance can reap any that
    survive an abrupt restart. Workers run in their own ``os.setsid`` session, so
    when the parent server dies they are reparented to init and outlive it."""
    try:
        from ouroboros.utils import atomic_write_json
        recs = [{"pid": int(w.proc.pid)} for w in WORKERS.values() if w.proc.pid]
        atomic_write_json(
            _worker_pids_path(),
            {"server_pid": os.getpid(), "ts": utc_now_iso(), "workers": recs},
            trailing_newline=True,
        )
    except Exception:
        log.debug("Failed to record worker pids", exc_info=True)
    # Write-through into the custody ledger (SSOT for the generation reaper);
    # worker_pids.json stays as the legacy session-leader reap path.
    try:
        from ouroboros.process_custody import record_process

        for w in WORKERS.values():
            if w.proc.pid:
                record_process(
                    DRIVE_ROOT,
                    pid=int(w.proc.pid),
                    cmd=f"ouroboros-worker-{w.wid}",
                    purpose=f"worker:{w.wid}",
                    scope="session",
                )
    except Exception:
        log.debug("Failed to ledger worker pids", exc_info=True)


def reap_orphaned_workers() -> int:
    """Kill leftover worker process groups left by a PRIOR server instance.

    ``kill_workers`` only walks the in-memory ``WORKERS`` dict, so workers
    orphaned by an abrupt restart (reparented to init, ~one Python interpreter
    each) were never reaped and accumulated across restarts. On startup we read
    the prior pid record and force-kill any that are still alive AND verifiably
    ours — cmdline matches this interpreter/multiprocessing and the process is
    its own session leader (``pgid == pid``) — which guards against PID reuse and
    bounds the group kill to the worker's own setsid session."""
    try:
        from ouroboros.utils import read_json_dict
        from ouroboros.platform_layer import (
            force_kill_pid,
            kill_process_group_id,
            process_command,
            process_group_id,
        )
    except Exception:
        return 0
    data = read_json_dict(_worker_pids_path()) or {}
    prior = data.get("workers") or []
    if not isinstance(prior, list) or not prior:
        return 0
    current = {w.proc.pid for w in WORKERS.values() if w.proc.pid}
    killed: List[int] = []
    for rec in prior:
        try:
            pid = int((rec or {}).get("pid") or 0)
        except (TypeError, ValueError):
            continue
        if not pid or pid in current or pid == os.getpid():
            continue
        cmd = process_command(pid)
        if not cmd:
            continue  # already dead
        if sys.executable not in cmd and "multiprocessing" not in cmd:
            continue  # PID reused by an unrelated process — do not touch it
        pgid = process_group_id(pid)
        if pgid and pgid == pid:
            kill_process_group_id(pgid)  # the worker's own setsid session
        force_kill_pid(pid)
        killed.append(pid)
    if killed:
        try:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "orphaned_workers_reaped", "pids": killed},
            )
        except Exception:
            log.debug("Failed to log orphaned worker reap", exc_info=True)
    return len(killed)


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


@_serialized_worker_lifecycle
def kill_workers_for_update(*, result_reason: str, terminal_status: str = "interrupted") -> List[str]:
    """Stop the current pool and return anything whose death could not be proven."""
    from ouroboros.platform_layer import kill_pid_tree

    with _queue_lock:
        fenced = list(WORKERS.values())
    teardown_error = ""
    try:
        kill_workers(
            result_reason=result_reason,
            terminal_status=terminal_status,
            disable_reason="managed_update",
            preserve_pending=True,
        )
    except Exception as exc:
        teardown_error = f"teardown:{type(exc).__name__}: {exc}"
    survivors: List[str] = []
    for worker in fenced:
        try:
            if worker.proc.is_alive() and worker.proc.pid:
                kill_pid_tree(worker.proc.pid)
                worker.proc.join(timeout=3)
            if worker.proc.is_alive():
                survivors.append(f"worker:{worker.proc.pid or worker.wid}")
        except Exception as exc:
            survivors.append(f"worker:{worker.wid}:{type(exc).__name__}")
    if teardown_error:
        survivors.append(teardown_error)
    return survivors


def _kill_survivors() -> None:
    """Force-kill any workers and their entire descendant trees."""
    from ouroboros.platform_layer import kill_pid_tree
    for w in WORKERS.values():
        pid = w.proc.pid
        if pid is None:
            continue
        if w.proc.is_alive():
            kill_pid_tree(pid)
            w.proc.join(timeout=2)


@_serialized_worker_lifecycle
def respawn_worker(wid: int) -> bool:
    """Replace one owned slot without forking under the queue RLock.

    The lifecycle lock makes the two-phase check/start/swap mutually exclusive
    with full-pool shutdown/start.  The identity check after ``proc.start()``
    prevents a replacement from being installed if the slot was removed while
    the queue lock was released.
    """
    with _queue_lock:
        old = WORKERS.get(wid)
    if old is None:
        return False
    ctx = _get_ctx()
    in_q = ctx.Queue()
    proc = ctx.Process(target=worker_main,
                       args=(wid, in_q, get_event_q(), str(REPO_DIR), str(DRIVE_ROOT),
                             _current_custody_session_id()))
    proc.daemon = True
    try:
        proc.start()
    except Exception:
        try:
            in_q.close()
            in_q.cancel_join_thread()
        except Exception:
            pass
        raise
    installed = False
    with _queue_lock:
        if WORKERS.get(wid) is old:
            WORKERS[wid] = Worker(wid=wid, proc=proc, in_q=in_q, busy_task_id=None)
            installed = True
    if not installed:
        try:
            from ouroboros.platform_layer import kill_pid_tree

            if proc.pid:
                kill_pid_tree(proc.pid)
            elif proc.is_alive():
                proc.terminate()
            proc.join(timeout=2)
        finally:
            try:
                in_q.close()
                in_q.cancel_join_thread()
            except Exception:
                pass
        return False
    # Close the crashed worker's old queue now that nothing can route to it,
    # otherwise its file descriptors / semaphores leak on every respawn.
    if old is not None and getattr(old, "in_q", None) is not None:
        try:
            old.in_q.close()
            old.in_q.cancel_join_thread()
        except Exception:
            log.debug("Failed to close old worker queue on respawn", exc_info=True)
    _record_worker_pids()
    # Do not reset _LAST_SPAWN_TIME here; respawn grace would hide crash storms.
    return True


def _drop_cancelled_pending() -> None:
    """Remove pending tasks cancelled/finished between scheduling and assignment
    so a cancelled subagent never actually starts. Caller holds _queue_lock."""
    if not PENDING:
        return
    try:
        from ouroboros.task_results import (
            STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, _TRULY_TERMINAL_STATUSES,
            load_task_result, write_task_result,
        )
    except Exception:
        return
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
        if status == STATUS_CANCEL_REQUESTED:
            try:
                write_task_result(DRIVE_ROOT, tid, STATUS_CANCELLED, result="Cancelled before start.")
            except Exception:
                log.debug("Failed to finalize cancelled pending task %s", tid, exc_info=True)
            _emit_task_done_terminal(t, tid, "cancelled")
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


def _evolution_assignment_error(task: Dict[str, Any]) -> str:
    """Return the exact authority error for an evolution task about to run."""
    if str(task.get("type") or "") != "evolution":
        return ""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    tx = metadata.get("evolution_transaction")
    tx = tx if isinstance(tx, dict) else {}
    task_id = str(task.get("id") or "")
    if str(tx.get("task_id") or "") != task_id:
        return "task_mismatch"
    from supervisor.evolution_lifecycle import check_evolution_authority

    try:
        authority = check_evolution_authority(
            campaign_id=str(tx.get("campaign_id") or ""),
            transaction_id=str(tx.get("transaction_id") or ""),
            task_id=task_id,
            require_uncommitted=True,
        )
    except Exception:
        log.warning("Evolution assignment authority check failed", exc_info=True)
        return "authority_check_failed"
    return "" if authority.get("ok") else str(authority.get("reason") or "unknown")


def _cancel_unauthorized_evolution(task: Dict[str, Any], reason: str) -> bool:
    """Terminally cancel a stale restored/retried evolution task."""
    task_id = str(task.get("id") or "")
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    try:
        write_task_result(
            DRIVE_ROOT,
            task_id,
            STATUS_CANCELLED,
            reason_code="evolution_authority_missing",
            authority_reason=str(reason or "unknown"),
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            result=f"Evolution authority is no longer active ({reason or 'unknown'}).",
        )
    except Exception:
        log.debug("Failed to cancel unauthorized evolution task %s", task_id, exc_info=True)
        return False
    _emit_task_done_terminal(
        task, task_id, "cancelled", reason_code="evolution_authority_missing",
    )
    append_jsonl(
        DRIVE_ROOT / "logs" / "events.jsonl",
        {
            "ts": utc_now_iso(), "type": "evolution_assignment_rejected",
            "task_id": task_id, "reason": str(reason or "unknown"),
        },
    )
    return True


def assign_tasks() -> None:
    from supervisor import queue
    from supervisor.state import budget_remaining, EVOLUTION_BUDGET_RESERVE
    with _queue_lock:
        st = load_state()
        try:
            remaining = budget_remaining(st, strict=True)
        except Exception:
            log.error("Task assignment blocked: monetary authority unavailable")
            return
        if remaining <= 0:
            planned = []
            for task in PENDING:
                if isinstance(task.get("_budget_pause"), dict):
                    continue
                task_id = str(task.get("id") or "")
                cost_fields = reconstruct_task_cost(
                    task_id, fields=True,
                    drive_root=pathlib.Path(task.get("budget_drive_root") or DRIVE_ROOT),
                )
                if cost_fields.get("cost_accounting_status") != "available":
                    log.error("Budget pause blocked: task attempt history unavailable for %s", task_id)
                    return
                retry_lineage = bool(
                    int(task.get("_attempt") or 1) > 1
                    or task.get("original_task_id") or task.get("timeout_retry_from")
                )
                replay_safe = (
                    int(cost_fields.get("total_rounds") or 0) == 0
                    and not bool(cost_fields.get("ledger_integrity_degraded"))
                    and not retry_lineage
                )
                pause = {
                    "status": "paused_before_dispatch" if replay_safe else "resource_limited",
                    "scope": "global",
                    "physical_calls": int(cost_fields.get("total_rounds") or 0),
                    "replay_safe": replay_safe,
                    "auto_resume": False,
                    "resume_policy": "manual_same_generation" if replay_safe else "cancel_or_new_run",
                    "paused_at": utc_now_iso(),
                }
                planned.append((task, pause, cost_fields))
            newly_paused, terminal_ids = [], []
            for task, pause, cost_fields in planned:
                task_id = str(task.get("id") or "")
                result_root = pathlib.Path(task.get("budget_drive_root") or DRIVE_ROOT)
                try:
                    from ouroboros.task_results import STATUS_FAILED, STATUS_SCHEDULED, write_task_result

                    if pause["replay_safe"]:
                        task["_budget_pause"] = pause
                        newly_paused.append(task_id)
                        write_task_result(
                            result_root, task_id, STATUS_SCHEDULED,
                            reason_code="budget_exhausted", resource_limit=pause,
                        )
                    else:
                        write_task_result(
                            result_root, task_id, STATUS_FAILED,
                            reason_code="budget_exhausted", resource_limit=pause,
                            result="Budget exhausted after prior dispatch; cancel or start a new run.",
                            **cost_fields,
                        )
                        _emit_task_done_terminal(
                            task, task_id, "failed", reason_code="budget_exhausted",
                            cost_fields=cost_fields,
                        )
                        terminal_ids.append(task_id)
                except Exception:
                    log.error("Failed to project budget stop for %s", task_id, exc_info=True)
            if terminal_ids:
                terminal = set(terminal_ids)
                PENDING[:] = [task for task in PENDING if str(task.get("id") or "") not in terminal]
            if newly_paused or terminal_ids:
                append_jsonl(
                    DRIVE_ROOT / "logs" / "events.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "budget_tasks_paused",
                        "scope": "global",
                        "task_ids": newly_paused,
                        "resource_limited_task_ids": terminal_ids,
                        "auto_resume": False,
                    },
                )
                if st.get("owner_chat_id"):
                    send_with_budget(
                        int(st["owner_chat_id"]),
                        "🚫 Model budget reached. Queued tasks are paused before dispatch; "
                        "raising the limit does not resume them automatically.",
                    )
                queue.persist_queue_snapshot(reason="budget_paused_before_dispatch")
            return

        # Drop tasks cancelled after scheduling but before assignment.
        _drop_cancelled_pending()

        # Evolution is hard-blocked in light runtime mode at the assignment
        # chokepoint too: a task restored from a snapshot or created before the
        # mode switch must never actually run. Cancel them terminally.
        from supervisor.evolution_lifecycle import evolution_block_reason
        evo_block = evolution_block_reason()
        if evo_block and any(str(t.get("type") or "") == "evolution" for t in PENDING):
            blocked_ids = [str(t.get("id") or "") for t in PENDING if str(t.get("type") or "") == "evolution"]
            PENDING[:] = [t for t in PENDING if str(t.get("type") or "") != "evolution"]
            from ouroboros.task_results import STATUS_CANCELLED, write_task_result
            for tid in blocked_ids:
                try:
                    write_task_result(
                        DRIVE_ROOT, tid, STATUS_CANCELLED,
                        result="Evolution is disabled in light runtime mode.",
                    )
                except Exception:
                    log.debug("Failed to cancel light-mode evolution task %s", tid, exc_info=True)
            if st.get("owner_chat_id"):
                send_with_budget(int(st["owner_chat_id"]), evo_block)
            queue.persist_queue_snapshot(reason="evolution_blocked_light")

        from ouroboros.project_lease import candidate_is_leasable, pin_task_lane, running_project_lanes
        from ouroboros.config import get_max_active_subagents_per_root

        # project_id -> registered working_dir, read ONCE per assignment pass.
        # A task scoped POST-HOC (mark_task_project) carries no workspace_root of
        # its own; without this map its lane would not match the room task
        # already writing the SAME project folder, and both would be admitted as
        # top-level writers. The lease itself stays filesystem-free under the
        # queue lock, so the registry read lives here. An unreadable registry is None, NOT {}: "no project has a
        # folder" is an answer the lane may narrow on, "the folders are unknown" is not, and collapsing the two let a
        # folder-bearing candidate slip past a narrow lane and a second writer into the folder (I3).
        # project_working_dirs answers None for a registry that EXISTS and does not parse — no exception is raised on
        # that path — so this arm only covers a raise on the way there.
        try:
            from ouroboros.projects_registry import project_working_dirs

            _project_workspaces = project_working_dirs(DRIVE_ROOT)
        except Exception:
            log.debug("assign_tasks: project working_dir map unavailable", exc_info=True)
            _project_workspaces = None

        def _running_subagent_count(root_task_id: str) -> int:
            if not root_task_id:
                return 0
            count = 0
            for meta in RUNNING.values():
                task = meta.get("task") if isinstance(meta, dict) else None
                if (
                    isinstance(task, dict)
                    and str(task.get("delegation_role") or "") == "subagent"
                    and str(task.get("root_task_id") or "") == root_task_id
                ):
                    count += 1
            return count

        def _assignment_depth_reservation_admits(candidate: dict) -> bool:
            root_task_id = str(candidate.get("root_task_id") or "")
            parent_id = str(candidate.get("parent_task_id") or "").strip()
            if not root_task_id or not parent_id:
                return False
            parent_running = any(
                str((meta.get("task") if isinstance(meta, dict) else {}).get("id") or "") == parent_id
                and str((meta.get("task") if isinstance(meta, dict) else {}).get("root_task_id") or "") == root_task_id
                and str((meta.get("task") if isinstance(meta, dict) else {}).get("delegation_role") or "") == "subagent"
                for meta in RUNNING.values()
            )
            if not parent_running:
                return False
            direct_running_children = sum(
                1 for meta in RUNNING.values()
                if isinstance(meta, dict)
                and isinstance(meta.get("task"), dict)
                and str(meta["task"].get("root_task_id") or "") == root_task_id
                and str(meta["task"].get("delegation_role") or "") == "subagent"
                and str(meta["task"].get("parent_task_id") or "").strip() == parent_id
            )
            return direct_running_children < 1

        for w in WORKERS.values():
            if w.busy_task_id is None and not getattr(w, "reaping", False) and PENDING:
                # One-writer-per-WORKING-FOLDER lease: recompute per
                # assignment so a task assigned in THIS loop pass immediately
                # occupies its lane. The lane key is the FOLDER, so two threads
                # of one project in the SAME folder still serialize while a
                # worktree-branched thread runs concurrently. A task with no
                # workspace_root of its own resolves through
                # _project_workspaces to its project's registered folder.
                leased = running_project_lanes(RUNNING.values(), _project_workspaces)
                # Find first suitable task (skip over-budget evolution tasks
                # and project-leased candidates)
                chosen_idx = None
                for i, candidate in enumerate(PENDING):
                    if not repo_writer_task_allowed(candidate):
                        continue
                    if isinstance(candidate.get("_budget_pause"), dict):
                        continue
                    root_task_id = str(candidate.get("root_task_id") or "").strip()
                    if root_task_id in queue.BUDGET_ROOT_FENCES:
                        continue
                    if str(candidate.get("type") or "") == "evolution" and remaining < EVOLUTION_BUDGET_RESERVE:
                        continue
                    if not candidate_is_leasable(candidate, leased, _project_workspaces):
                        continue
                    if str(candidate.get("delegation_role") or "") == "subagent":
                        root_task_id = str(candidate.get("root_task_id") or "")
                        if (
                            _running_subagent_count(root_task_id) >= get_max_active_subagents_per_root()
                            and not _assignment_depth_reservation_admits(candidate)
                        ):
                            continue
                    chosen_idx = i
                    break
                if chosen_idx is None:
                    # Nothing assignable: project-leased tasks WAIT in PENDING
                    # for the next pass; only over-budget evolution tasks are
                    # cleaned out.
                    if remaining < EVOLUTION_BUDGET_RESERVE and any(
                        str(t.get("type") or "") == "evolution" for t in PENDING
                    ):
                        PENDING[:] = [t for t in PENDING if str(t.get("type") or "") != "evolution"]
                        queue.persist_queue_snapshot(reason="evolution_dropped_budget")
                    continue
                task = PENDING.pop(chosen_idx)
                evolution_error = _evolution_assignment_error(task)
                if evolution_error:
                    if _cancel_unauthorized_evolution(task, evolution_error):
                        queue.persist_queue_snapshot(reason="evolution_authority_rejected")
                    else:
                        PENDING.insert(chosen_idx, task)
                    continue
                if str(task.get("delegation_role") or "") == "subagent" and str(task.get("drive_root") or ""):
                    try:
                        from ouroboros.task_results import STATUS_RUNNING, write_task_result
                        write_task_result(
                            DRIVE_ROOT,
                            str(task.get("id") or ""),
                            STATUS_RUNNING,
                            parent_task_id=task.get("parent_task_id"),
                            root_task_id=task.get("root_task_id"),
                            session_id=task.get("session_id"),
                            actor_id=task.get("actor_id"),
                            delegation_role=task.get("delegation_role"),
                            project_id=task.get("project_id"),
                            role=task.get("role"),
                            description=task.get("description"),
                            objective=task.get("objective") or task.get("description"),
                            expected_output=task.get("expected_output"),
                            constraints=task.get("constraints"),
                            context=task.get("context"),
                            memory_mode=task.get("memory_mode"),
                            drive_root=task.get("drive_root"),
                            child_drive_root=task.get("child_drive_root") or task.get("drive_root"),
                            budget_drive_root=task.get("budget_drive_root"),
                            task_constraint=task.get("task_constraint"),
                            # INTENT ONLY. This mirror is written at ASSIGNMENT, one
                            # step before the worker dispatches and resolves the
                            # child; naming `effective_model_lane`/`model` here wrote
                            # whatever the record happened to hold, which on a retry
                            # is the PREVIOUS attempt's resolution and on a fresh
                            # child is nothing at all.
                            model_lane=task.get("model_lane"),
                            requested_model_lane=task.get("requested_model_lane"),
                            parent_model_lane=task.get("parent_model_lane"),
                            requested_executor=task.get("requested_executor"),
                            task_group_id=task.get("task_group_id"),
                            task_group=task.get("task_group"),
                            subagent_envelope=task.get("subagent_envelope"),
                            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
                            result="Subagent assigned to a worker.",
                        )
                    except Exception:
                        log.debug("Failed to mirror running subagent status", exc_info=True)
                w.busy_task_id = task["id"]
                w.in_q.put(task)
                now_ts = time.time()
                running_record = dict(task)
                # PIN the writer lane at the RUNNING transition (T0R2-7). Derived
                # on demand, a mid-run edit of this record — the post-hoc project
                # conversion is the live one — silently moved the task into a
                # different lane, releasing the folder it is still writing in and
                # admitting a second writer onto it. The SAME project->folder map
                # the admission check used must be handed to the pin: pinning
                # without it freezes (project_id, "") for a task that was compared
                # as ("", registered_folder), so the folder reads as unheld to the
                # very next candidate and the pin itself admits a second writer.
                pin_task_lane(running_record, _project_workspaces)
                RUNNING[task["id"]] = {
                    "task": running_record, "worker_id": w.wid,
                    "started_at": now_ts, "last_heartbeat_at": now_ts,
                    "soft_sent": False, "attempt": int(task.get("_attempt") or 1),
                }
                task_type = str(task.get("type") or "")
                if task_type in ("evolution", "review"):
                    st = load_state()
                    if st.get("owner_chat_id"):
                        emoji = '🧬' if task_type == 'evolution' else '🔎'
                        send_with_budget(
                            int(st["owner_chat_id"]),
                            f"{emoji} {task_type.capitalize()} task {task['id']} started.",
                        )
                queue.persist_queue_snapshot(reason="assign_task")

def ensure_workers_healthy() -> None:
    """Detect dead workers, finalize/requeue their tasks, respawn.

    Runs under the queue lock: the RUNNING pops and respawn decisions here
    raced with HTTP cancel handlers (double respawn → orphaned worker, and
    "dict changed size" crashes in concurrent iteration). RLock keeps the
    nested enqueue/respawn/persist calls re-entrant.
    """
    from supervisor import queue
    # Workers need init time after spawn.
    if (time.time() - _LAST_SPAWN_TIME) < _SPAWN_GRACE_SEC:
        return
    with _queue_lock:
        respawn_ids, disable_pool = _ensure_workers_healthy_locked(queue)
    if disable_pool:
        # Every lifecycle operation takes lifecycle -> queue lock.  Calling
        # kill_workers while still holding queue lock would invert that order
        # against a concurrent respawn and deadlock.
        kill_workers(disable_reason="worker_crash_storm")
        CRASH_TS.clear()
        return
    for wid in respawn_ids:
        try:
            respawn_worker(wid)
        except Exception:
            log.warning("Failed to respawn crashed worker %d", wid, exc_info=True)
            with _queue_lock:
                slot = WORKERS.get(wid)
                if slot is not None:
                    slot.reaping = False
    if respawn_ids:
        queue.persist_queue_snapshot(reason="worker_respawn_after_crash")


def _ensure_workers_healthy_locked(queue: Any) -> tuple[List[int], bool]:
    busy_crashes = 0
    dead_detections = 0
    crashed_tasks = []
    respawn_ids: List[int] = []
    for wid, w in list(WORKERS.items()):
        # Variant A: a slot marked `reaping` is owned end-to-end by the background reaper
        # (kill -> join -> archive -> respawn). Its proc is expected to die mid-reap, so the
        # crash detector must NOT also respawn it — that double-respawn would orphan a live
        # worker process. The reaper installs a fresh Worker (reaping=False) when done.
        if getattr(w, "reaping", False):
            continue
        if not w.proc.is_alive():
            # Reserve the dead slot before the main loop releases the queue lock
            # to start its replacement. assign_tasks skips reaping slots.
            w.reaping = True
            dead_detections += 1
            if w.busy_task_id is not None:
                busy_crashes += 1
            exitcode = w.proc.exitcode
            meta = RUNNING.get(w.busy_task_id, {}) if w.busy_task_id else {}
            task_info = meta.get("task", {}) if isinstance(meta, dict) else {}
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "worker_dead_detected",
                    "worker_id": wid,
                    "exitcode": exitcode,
                    "busy_task_id": w.busy_task_id,
                    "task_type": task_info.get("type") if isinstance(task_info, dict) else None,
                    "task_description": (task_info.get("description", "") or "")[:200] if isinstance(task_info, dict) else None,
                    "uptime_sec": round(time.time() - meta["started_at"]) if isinstance(meta, dict) and meta.get("started_at") else None,
                    "attempt": meta.get("attempt") if isinstance(meta, dict) else None,
                    "signal": -exitcode if isinstance(exitcode, int) and exitcode < 0 else None,
                },
            )
            if w.busy_task_id and isinstance(meta, dict) and meta.get("task"):
                crashed_tasks.append({"task_id": w.busy_task_id, "task_type": task_info.get("type") if isinstance(task_info, dict) else None})
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "worker_crash_task_dump",
                        "worker_id": wid,
                        "task": meta["task"],
                        "started_at": meta.get("started_at"),
                        "last_heartbeat_at": meta.get("last_heartbeat_at"),
                        "attempt": meta.get("attempt"),
                    },
                )
            if w.busy_task_id and w.busy_task_id in RUNNING:
                meta = RUNNING.pop(w.busy_task_id) or {}
                try:
                    from ouroboros.tools.services import archive_task_service_logs
                    task_for_roots = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
                    archive_task_service_logs(pathlib.Path(DRIVE_ROOT), str(w.busy_task_id), task_for_roots)
                except Exception:
                    log.debug("Failed to archive service logs for task %s", w.busy_task_id, exc_info=True)
                task = meta.get("task") if isinstance(meta, dict) else None
                if isinstance(task, dict):
                    task_type = str(task.get("type") or "")
                    # A negative exitcode means the worker died from a signal
                    # (SIGSEGV/SIGBUS/SIGABRT/SIGKILL). These are deterministic
                    # infrastructure crashes: retrying the same runtime path
                    # reproduces them and only burns budget, so they are terminal
                    # for EVERY task type (not just deep_self_review).
                    is_crash_signal = isinstance(exitcode, int) and exitcode < 0
                    crash_signal = -exitcode if is_crash_signal else None
                    chat_id = coerce_chat_identity(task.get("chat_id"), 0)
                    attempt = int(task.get("_attempt") or 1)
                    # Reconstruct cost/rounds from durable llm_usage for any
                    # abnormal-termination rollup below (worker died pre-finalize,
                    # so the event would otherwise carry zeros).
                    r_cost_fields = reconstruct_task_cost(str(w.busy_task_id), fields=True)

                    # Already terminal via inline/direct-chat path? Leave it.
                    already_done = False
                    existing_status = ""
                    try:
                        from ouroboros.task_results import load_task_result, _TRULY_TERMINAL_STATUSES
                        existing = load_task_result(DRIVE_ROOT, str(w.busy_task_id))
                        if existing and str(existing.get("status") or "") in _TRULY_TERMINAL_STATUSES:
                            already_done = True
                            existing_status = str(existing.get("status") or "")
                            log.info(
                                "Skipping requeue for task %s — already in terminal state: %s",
                                w.busy_task_id, existing.get("status"),
                            )
                    except Exception:
                        log.debug("Failed to check existing result for %s", w.busy_task_id, exc_info=True)

                    if already_done:
                        # Terminal on disk but the worker died — its normal task_done
                        # event may have been lost with it. Emit an (idempotent)
                        # terminal event so the live card resolves instead of
                        # spinning until reconnect/history reconciliation.
                        _emit_task_done_terminal(task, str(w.busy_task_id), existing_status or "completed")
                    elif is_crash_signal or attempt > QUEUE_MAX_RETRIES:
                        deep = task_type == "deep_self_review"
                        if is_crash_signal:
                            log.warning(
                                "Task %s worker crashed with signal %s — terminal (no retry)",
                                w.busy_task_id, crash_signal,
                            )
                            result_text = (
                                f"❌ {'Deep self-review ' if deep else ''}worker process crashed "
                                f"(signal {crash_signal}). This is an infrastructure/platform crash "
                                "and is not retried automatically. "
                                + (
                                    "Use /restart and then /review to retry after a clean restart."
                                    if deep else
                                    "Use /restart and try again; if it recurs it is a platform-level issue."
                                )
                            )
                            reason_code = "worker_crash_signal"
                        else:
                            log.warning(
                                "Task %s exceeded crash retry limit (%d/%d) — marking failed",
                                w.busy_task_id, attempt, QUEUE_MAX_RETRIES,
                            )
                            result_text = (
                                f"❌ Task failed after {attempt} crash(es) (exit {exitcode}). "
                                "Worker process died repeatedly — likely a platform-level issue. "
                                "Please try again or use a different approach."
                            )
                            reason_code = "worker_crash_retry_exhausted"
                        try:
                            from ouroboros.task_results import STATUS_FAILED, write_task_result
                            write_task_result(
                                DRIVE_ROOT, str(w.busy_task_id), STATUS_FAILED,
                                result=result_text,
                                reason_code=reason_code,
                                outcome_axes=terminal_outcome_axes(lifecycle=STATUS_FAILED, execution=EXECUTION_INFRA_FAILED, reason_code=reason_code, review_trigger="worker_terminal"),
                                crash_signal=crash_signal,
                                crash_exitcode=exitcode if isinstance(exitcode, int) else None,
                                **r_cost_fields,
                            )
                        except Exception:
                            log.debug("Failed to write failed status for %s", w.busy_task_id, exc_info=True)
                        # Message before task_done: otherwise the UI may close the card first.
                        try:
                            if is_crash_signal and deep:
                                user_msg = (
                                    f"❌ Deep self-review failed: worker process crashed (signal {crash_signal}). "
                                    "This is a known platform fork-safety limitation. "
                                    "Please use `/restart` and then `/review` to retry with a fresh process."
                                )
                            elif is_crash_signal:
                                user_msg = (
                                    f"❌ Task `{str(w.busy_task_id)[:8]}` failed: worker process crashed "
                                    f"(signal {crash_signal}). This is an infrastructure crash and was not retried."
                                )
                            else:
                                user_msg = (
                                    f"❌ Task `{str(w.busy_task_id)[:8]}` failed after {attempt} crash(es). "
                                    "Worker process crashed repeatedly. Please try again."
                                )
                            incident_task_id = str(w.busy_task_id or "")
                            send_with_budget(
                                chat_id,
                                user_msg,
                                is_progress=True,
                                task_id=incident_task_id,
                                progress_meta={
                                    "task_incident": reason_code,
                                    "toast_once": f"{incident_task_id}:{reason_code}:{attempt}",
                                },
                            )
                        except Exception:
                            log.debug("Failed to send failure message for %s", w.busy_task_id, exc_info=True)
                        _emit_task_done_terminal(
                            task, str(w.busy_task_id), "failed",
                            reason_code=reason_code, cost_fields=r_cost_fields,
                        )
                    elif task_type == "evolution" and not bool(load_state().get("evolution_mode_enabled")):
                        # Evolution was stopped: do not resurrect a dead evolution
                        # worker into another cycle (mirrors the hard-timeout gate
                        # in queue.enforce_task_timeouts).
                        try:
                            from ouroboros.task_results import STATUS_CANCELLED, write_task_result
                            write_task_result(
                                DRIVE_ROOT, str(w.busy_task_id), STATUS_CANCELLED,
                                result="Evolution worker died after the campaign was stopped; not retried.",
                                reason_code="evolution_stopped_no_retry",
                                outcome_axes=terminal_outcome_axes(lifecycle=STATUS_CANCELLED, execution="cancelled", reason_code="evolution_stopped_no_retry", review_trigger="worker_terminal"),
                                **r_cost_fields,
                            )
                        except Exception:
                            log.debug("Failed to write cancelled status for %s", w.busy_task_id, exc_info=True)
                        _emit_task_done_terminal(
                            task, str(w.busy_task_id), "cancelled",
                            cost_fields=r_cost_fields,
                        )
                    else:
                        task = dict(task)
                        task["_attempt"] = attempt + 1
                        try:
                            from ouroboros.task_results import STATUS_INTERRUPTED, write_task_result
                            write_task_result(
                                DRIVE_ROOT, str(w.busy_task_id), STATUS_INTERRUPTED,
                                result=f"Worker process died mid-task (attempt {attempt}). Retrying.",
                                **r_cost_fields,
                            )
                        except Exception:
                            log.debug("Failed to write interrupted status for %s", w.busy_task_id, exc_info=True)
                        admitted = queue.enqueue_task(task, front=True)
                        admission_block = (
                            str(admitted.get("_admission_blocked") or "")
                            if isinstance(admitted, dict) else ""
                        )
                        if admission_block:
                            reason_code = "worker_crash_retry_admission_blocked"
                            try:
                                from ouroboros.task_results import STATUS_FAILED, write_task_result
                                write_task_result(
                                    DRIVE_ROOT,
                                    str(w.busy_task_id),
                                    STATUS_FAILED,
                                    result=(
                                        "Worker crashed and its retry was blocked by the active "
                                        f"{admission_block} admission fence."
                                    ),
                                    reason_code=reason_code,
                                    outcome_axes=terminal_outcome_axes(
                                        lifecycle=STATUS_FAILED,
                                        execution=EXECUTION_INFRA_FAILED,
                                        reason_code=reason_code,
                                        review_trigger="worker_terminal",
                                    ),
                                    **r_cost_fields,
                                )
                            except Exception:
                                log.debug(
                                    "Failed to terminalize admission-blocked retry for %s",
                                    w.busy_task_id,
                                    exc_info=True,
                                )
                            _emit_task_done_terminal(
                                task,
                                str(w.busy_task_id),
                                "failed",
                                reason_code=reason_code,
                                cost_fields=r_cost_fields,
                            )
            respawn_ids.append(wid)

    now = time.time()
    alive_now = sum(1 for w in WORKERS.values() if w.proc.is_alive())
    if dead_detections:
        # Only count busy crashes or all-workers-dead as storm signals.
        if busy_crashes > 0 or alive_now == 0:
            CRASH_TS.extend([now] * max(1, dead_detections))
        else:
            CRASH_TS.clear()

    CRASH_TS[:] = [t for t in CRASH_TS if (now - t) < 60.0]
    disable_pool = len(CRASH_TS) >= 3
    if disable_pool:
        # Do not execv on crash storms; keep direct-chat mode alive.
        st = load_state()
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "crash_storm_detected",
                "crash_count": len(CRASH_TS),
                "worker_count": len(WORKERS),
                "crashed_tasks": crashed_tasks,
            },
        )
        if st.get("owner_chat_id"):
            send_with_budget(
                int(st["owner_chat_id"]),
                "⚠️ Frequent worker crashes. Multiprocessing workers disabled, "
                "continuing in direct-chat mode (threading).",
                is_progress=True,
                progress_meta={
                    "task_incident": "worker_crash_storm",
                    "toast_once": f"worker-crash-storm:{int(min(CRASH_TS) if CRASH_TS else now)}",
                },
            )
    return respawn_ids, disable_pool
