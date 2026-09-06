"""Worker lifecycle, health, and direct-chat handling for the supervisor."""

from __future__ import annotations
import logging
log = logging.getLogger(__name__)

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

from supervisor.state import load_state, append_jsonl, reconstruct_task_cost
from supervisor.message_bus import send_with_budget
from supervisor.message_bus import coerce_chat_identity  # noqa: F401 -- worker_health leaf reads it via the _pool() handle
from ouroboros.config import DATA_DIR, REPO_DIR as CONFIG_REPO_DIR, WORKER_SPAWN_GRACE_SEC
from ouroboros.depth_evidence import parse_task_depth
from ouroboros.review_owner_custody import (
    reconcile_confirmed_dead_review_owner as _reconcile_confirmed_dead_review_owner_for_root,
)
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
_SPAWN_GRACE_SEC: float = WORKER_SPAWN_GRACE_SEC  # config SSOT; leaves read it through _pool()

# Defaults: spawn on macOS + Windows, forkserver on Linux (G13); the env key
# OUROBOROS_WORKER_START_METHOD overrides either. fork() from the long-lived,
# multi-threaded supervisor is unsafe on every OS: a child forked while another
# server thread holds a per-module import lock keeps that lock with no thread
# left to release it and wedges on its first import of that module (the
# deadlock colab_bootstrap.py names; macOS adds dead Mach ports and a _scproxy
# SIGSEGV pre-exec). forkserver forks every worker from ONE bare single-threaded
# server process, so no lock is inherited, and a warm child still confirms in
# seconds (mock lane, fork -> forkserver: startup 2.4-3.3 -> 3.5-4.9s, respawn
# 2.3 -> 2.5-3.2s; window 90s). worker_main is module-level and re-derives state
# from argv (its lifeline: spawner sentinel, not ppid), so spawn/forkserver/frozen re-exec are safe.
_DEFAULT_WORKER_START_METHOD = "forkserver" if sys.platform.startswith("linux") else "spawn"
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


def direct_chat_turn(task_id: str = "") -> Optional[Dict[str, Any]]:
    """The in-process direct-chat turn as a queue-shaped task record, or None.

    The ownership predicate (``task_has_live_ownership``), the owner-control
    ingresses (cancel, hurry, decisions) and the graceful-stop episode resolve
    a live direct turn through THIS one reader, so the durable running mirror
    and the owner controls can never disagree about it again: a turn the
    task list shows as running is addressable, and a turn that is not
    addressable is not shown as live (the class the rc.7 QA regress hit —
    ``running`` + cancel 404 + spend still growing). Read WITHOUT the
    chat-agent lock, like ``chat_turn_liveness``: a wedged turn holds that
    lock for its whole duration. ``task_id`` narrows the answer to that turn;
    empty answers whichever direct turn is live. An ephemeral decision turn
    (not ``_accepting_owner_messages``) is transport control, never an
    owner-addressable task, and writes no durable running row either.
    """
    agent = _chat_agent
    if agent is None or not getattr(agent, "_busy", False):
        return None
    current = str(getattr(agent, "_current_task_id", "") or "")
    if not current or (task_id and current != str(task_id)):
        return None
    if not getattr(agent, "_accepting_owner_messages", False):
        return None
    metadata = getattr(agent, "_current_task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    text = str(getattr(agent, "_current_task_text", "") or "")
    record = {
        "id": current,
        "type": "task",
        "chat_id": int(getattr(agent, "_current_chat_id", 0) or 0),
        "project_id": str(metadata.get("project_id") or ""),
        "title": str(metadata.get("title") or ""),
        "suggested_name": str(metadata.get("suggested_name") or ""),
        "objective": text,
        "text": text,
        "metadata": dict(metadata),
        "_is_direct_chat": True,
        "_started_at": float(getattr(agent, "_task_started_ts", 0.0) or 0.0),
    }
    stamps = getattr(agent, "_direct_turn_stamps", None)
    if isinstance(stamps, dict) and str(stamps.get("_task_id") or "") == current:
        record.update({key: value for key, value in stamps.items() if key != "_task_id"})
    return record


def arm_direct_chat_turn(
    task_id: str, arm: Any, *, latch_key: str = "stop_control_msg_id", **extra_stamps: Any,
) -> Optional[Dict[str, Any]]:
    """Atomically arm an owner control against the live direct turn.

    The turn's completion path flips ``_busy``/``_accepting_owner_messages``
    under the agent's owner-message admission lock; a liveness read followed
    by an unlocked control write could therefore arm a turn that ended in
    between — a control and an owner toast over an answer that already
    landed. So the re-read, the write (``arm(turn)`` returns the control's
    msg_id, or "" when nothing was written) and the stamp all happen under
    that same lock: either the turn is live for the whole arm and ends only
    after it, or it was already gone and NOTHING is written. Returns the
    armed record, or None when the turn was gone. ``latch_key`` names the
    stamp the control's msg_id lands under (the immediate stop and the
    graceful owner-stop episode keep separate latches, so one never hides
    the other); ``extra_stamps`` ride along under the same lock."""
    agent = _chat_agent
    if agent is None:
        return None
    lock = getattr(agent, "_owner_message_admission_lock", None)
    with (lock if lock is not None else _threading.Lock()):
        turn = direct_chat_turn(task_id)
        if turn is None:
            return None
        written = arm(turn)
        if written:
            stamp_direct_chat_turn(task_id, **{latch_key: written, **extra_stamps})
            turn.update({latch_key: written, **extra_stamps})
        return turn


def stamp_direct_chat_turn(task_id: str, **fields: Any) -> bool:
    """Record owner-control state on the live direct turn — the direct-chat
    twin of the latch a pooled task keeps on its RUNNING row (for example the
    armed owner-stop control id, so a sweep tick re-arms idempotently instead
    of re-toasting). Stamps belong to ONE turn id and vanish with it. Returns
    False when that turn is not live (nothing to stamp)."""
    agent = _chat_agent
    if agent is None or direct_chat_turn(task_id) is None:
        return False
    stamps = getattr(agent, "_direct_turn_stamps", None)
    if not isinstance(stamps, dict) or str(stamps.get("_task_id") or "") != str(task_id):
        stamps = {"_task_id": str(task_id)}
        agent._direct_turn_stamps = stamps
    stamps.update(fields)
    return True


def _stage_promoted_initial_attachments(
    evt: dict, task: dict, tid: str, *, inherited_manifest: Any = None,
) -> tuple[list[dict], Optional[dict]]:
    """Stage before any durable project/workspace/task admission side effect."""

    uploads = evt.get("attachment_uploads")
    uploads = uploads if isinstance(uploads, list) else []
    inherited = inherited_manifest if isinstance(inherited_manifest, list) else []
    if not uploads and not inherited:
        return [], None
    manifest: list[dict] = []
    try:
        from ouroboros.artifacts import (
            attachment_manifest_all_rejected,
            materialize_inherited_attachment_manifest,
            remove_staged_attachments,
            stage_task_attachments,
        )
        from ouroboros.gateway.tasks import _render_attachment_lines

        inherited_rows, inherited_error = materialize_inherited_attachment_manifest(
            inherited, DRIVE_ROOT, tid,
        )
        if inherited_error:
            failed_inherited = [
                {
                    **{
                        key: row[key] for key in ("ordinal", "label")
                        if isinstance(row, dict) and key in row
                    },
                    "status": "rejected",
                    "reason": "inherited_source_unavailable",
                }
                for row in inherited
            ]
            return failed_inherited, {
                "status": "needs_manual_target",
                "reason": "attachment_admission_rejected",
                "detail": f"Inherited attachment materialization failed: {inherited_error}",
                "attachment_manifest": failed_inherited,
                "task_id": tid,
            }
        upload_rows = stage_task_attachments(DRIVE_ROOT, tid, uploads) if uploads else []
        # Preserve the private cleanup ownership carried by the staging helper so
        # every later admission refusal remains atomic even after composing
        # inherited and newly-uploaded inputs.
        manifest = inherited_rows if inherited_rows else upload_rows
        if inherited_rows and upload_rows:
            inherited_rows.extend(upload_rows)
            inherited_owned = getattr(inherited_rows, "_cleanup_owned_paths", None)
            upload_owned = getattr(upload_rows, "_cleanup_owned_paths", None)
            if isinstance(inherited_owned, set) and isinstance(upload_owned, set):
                inherited_owned.update(upload_owned)
        for ordinal, row in enumerate(manifest):
            if isinstance(row, dict):
                row["ordinal"] = ordinal
        rendered = _render_attachment_lines(manifest)
        # В25c partial-default; a FULLY-rejected set stays atomic (the task
        # would start with none of its declared material).
        if attachment_manifest_all_rejected(manifest):
            remove_staged_attachments(manifest)
            from ouroboros.headless import remove_subagent_task_drive

            remove_subagent_task_drive(DRIVE_ROOT, tid)
            return manifest, {
                "status": "needs_manual_target",
                "reason": "attachment_admission_rejected",
                "detail": rendered,
                "attachment_manifest": manifest,
                "task_id": tid,
            }
        if rendered:
            task["attachments"] = manifest
            task["attachment_images"] = [
                item for item in manifest
                if str(item.get("status") or "staged") == "staged" and item.get("is_image")
            ]
        return manifest, None
    except Exception:
        log.warning("promote: attachment staging failed for %s", tid, exc_info=True)
        if manifest:
            try:
                from ouroboros.artifacts import remove_staged_attachments

                remove_staged_attachments(manifest)
            except Exception:
                log.debug("promote: partial composed attachment cleanup failed", exc_info=True)
        declared = [*inherited, *uploads]
        manifest = [
            {
                "ordinal": index,
                "status": "rejected",
                "reason": "staging_unavailable",
                "label": str(item.get("label") or item.get("display_name") or f"attachment {index + 1}")
                if isinstance(item, dict) else f"attachment {index + 1}",
            }
            for index, item in enumerate(declared)
        ]
        return manifest, {
            "status": "needs_manual_target",
            "reason": "attachment_admission_rejected",
            "detail": "\n".join(
                f"- {row['label']}: rejected (reason=staging_unavailable, ordinal={row['ordinal']})"
                for row in manifest
            ),
            "attachment_manifest": manifest,
            "task_id": tid,
        }


def _reject_promoted_after_attachment_stage(
    outcome: dict, manifest: list[dict],
) -> dict:
    """Central idempotent cleanup for every non-scheduled post-stage exit."""

    if manifest:
        try:
            from ouroboros.artifacts import remove_staged_attachments

            remove_staged_attachments(manifest)
        except Exception:
            log.debug("promote: staged attachment cleanup failed", exc_info=True)
    return outcome


def _apply_presence_promotion_authority(
    evt: dict, task: dict, *, objective: str, expected_output: str,
) -> list[dict]:
    """Preserve inherited Presence authority while rebinding the new root."""

    presence = evt.get("presence") if isinstance(evt.get("presence"), dict) else None
    if not presence:
        return []
    task["_presence_origin"] = True
    task["source"] = "presence_promote"
    task.setdefault("metadata", {})["presence"] = dict(presence)
    contract = evt.get("task_contract") if isinstance(evt.get("task_contract"), dict) else {}
    inherited_manifest = [
        dict(row) for row in (contract.get("attachment_manifest") or [])
        if isinstance(row, dict)
    ]
    promoted_contract = dict(contract)
    promoted_contract.update({
        "task_type": "task",
        "objective": objective,
        "expected_output": expected_output,
        "attachment_manifest": [],
    })
    promoted_contract.pop("lineage", None)
    task["task_contract"] = promoted_contract
    return inherited_manifest


def _relocate_promoted_attachments(task: dict, tid: str, manifest: list[dict]) -> bool:
    """Move a pre-admitted manifest into the selected child drive, if any."""

    staged = [row for row in manifest if row.get("status") == "staged" and row.get("abs_path")]
    if not staged:
        return True
    target_root = pathlib.Path(str(task.get("drive_root") or DRIVE_ROOT))
    source_dir = pathlib.Path(str(staged[0]["abs_path"])).parent
    try:
        from ouroboros.artifacts import task_artifact_dir_path

        target_dir = task_artifact_dir_path(target_root, tid, create=False) / "attachments"
        if source_dir.resolve(strict=False) == target_dir.resolve(strict=False):
            return True
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        source_dir.replace(target_dir)
        from ouroboros.artifacts import rebase_staged_attachment_manifest

        rebase_staged_attachment_manifest(manifest, source_dir, target_dir)
        return True
    except Exception:
        log.warning("promote: attachment relocation failed for %s", tid, exc_info=True)
        return False


def _promoted_scheduled_outcome(task: dict, admitted: Any, tid: str) -> dict:
    """Carry the exact admitted contract to the canonical result writer."""

    admitted_contract = (
        admitted.get("task_contract")
        if isinstance(admitted, dict)
        and isinstance(admitted.get("task_contract"), dict)
        else task.get("task_contract")
    )
    return {
        "status": "scheduled",
        "task_id": tid,
        "_admitted_task_contract": dict(admitted_contract or {}),
    }


def _announce_created_project(project: Any, tid: str, task: Any = None) -> None:
    """Agent-initiated creation (owner 2=A): owe the durable Main "project
    started" row, only when THIS call actually created the project
    (``created is True`` from ``create_project``; idempotent replays stay
    silent). Fail-soft — a cosmetic row must never reject the promotion or
    the mid-task scope call."""
    if not isinstance(project, dict) or project.get("created") is not True:
        return
    try:
        from ouroboros.project_dialogue import announce_project_started

        announce_project_started(DRIVE_ROOT, project, tid, task=task)
    except Exception:
        log.debug("project started row failed for %s", project.get("id"), exc_info=True)


from supervisor.log_addressing import TurnEventQueue as _TurnEventQueue  # noqa: E402, F401


# Log types the worker sink does NOT forward: each already reaches the dashboard
# live via a dedicated EVENT_Q sibling/handler, so forwarding the worker's
# append_jsonl copy too would double-broadcast (and task_checkpoint would also be
# re-persisted to events.jsonl by _handle_log_event, a double file write).
# The second group publishes the SAME type twice at the producer (a durable
# append plus an emit_log_event live sibling of the identical type): the live
# copy is kept and the forwarded append copy is dropped.

# The server process runs the same suppression discipline over its raw
# append_jsonl->sink broadcasts (server.py installs the wrapper): every type
# here has a dedicated ctx.bridge.push_log at its supervisor handler, so the
# sink copy would be the second delivery of the same event. This is the
# exactly-once contract test_log_forwarding pins with the production sink
# installed. The set is a superset of the worker list because the direct-chat
# agent and Background Consciousness run inside the server process and append
# the same worker-shaped rows there.
from supervisor.worker_process import WORKER_LOG_SINK_SUPPRESSED_TYPES  # noqa: E402 -- moved span, needed at import time below

SERVER_LOG_SINK_SUPPRESSED_TYPES = WORKER_LOG_SINK_SUPPRESSED_TYPES | frozenset({
    "budget_scope_paused", "task_metrics_event", "review_late_result",
    "task_cost_finalized", "skill_exec_finished", "skill_exec_failed",
    "task_cancel_cascade_noop", "task_cancel_cascade_error",
})


_TERMINALIZATION_RETRY_FIELD = "_terminalization_retry"
_TERMINALIZATION_RETRY_STATUSES = frozenset({"failed", "cancelled", "interrupted"})
_CANCEL_INTENT_AUTHORITY_HOLD_FIELD = "_cancel_intent_authority_hold"


def _terminalization_retry_spec(task: Any) -> Optional[Dict[str, Any]]:
    """Return the bounded retry contract carried by a non-dispatchable row."""
    if not isinstance(task, dict):
        return None
    raw = task.get(_TERMINALIZATION_RETRY_FIELD)
    if not isinstance(raw, dict):
        return None
    status = str(raw.get("status") or "failed").strip() or "failed"
    return {
        "reason": str(raw.get("reason") or "Worker shutdown terminalization was not durable.")[:2000],
        "status": status if status in _TERMINALIZATION_RETRY_STATUSES else "failed",
        "trigger": str(raw.get("trigger") or "terminalization_retry").strip() or "terminalization_retry",
        "reconcile_delegate_custody": bool(raw.get("reconcile_delegate_custody", True)),
    }


def _terminalization_retry_request(task: Any) -> Optional[Dict[str, Any]]:
    """Return the exact terminalization request, finalizing interrupted custody."""
    spec = _terminalization_retry_spec(task)
    if spec is None:
        return None
    if spec["status"] == "interrupted":
        spec["status"] = "failed"
        spec["reason"] = (
            f"{spec['reason']} Original shutdown status was interrupted; "
            "the retained terminalization custody was finalized as failed."
        )
    return spec


def _retry_claim_identity(task: Any, intent: Any) -> bool:
    """Prove that a marker and claimed intent name the same custody request."""
    if not isinstance(task, dict) or not isinstance(intent, dict):
        return False
    raw = task.get(_TERMINALIZATION_RETRY_FIELD)
    request_id = str(raw.get("claim_request_id") or "") if isinstance(raw, dict) else ""
    owner = str(raw.get("claim_owner") or "") if isinstance(raw, dict) else ""
    if not request_id or owner != "pending_drop":
        return False
    if request_id != str(intent.get("request_id") or ""):
        return False
    try:
        int(raw.get("claim_generation"))
        int(raw.get("claim_pid"))
    except (TypeError, ValueError, OverflowError):
        return False
    return str(intent.get("claim_owner") or "") == owner


def _retry_claim_matches(task: Any, intent: Any) -> bool:
    """Prove that a refused claim is this process's exact marker claim."""
    if not _retry_claim_identity(task, intent):
        return False
    raw = task.get(_TERMINALIZATION_RETRY_FIELD)
    try:
        return (
            int(raw.get("claim_generation")) == int(intent.get("generation"))
            and int(raw.get("claim_pid")) == int(intent.get("claim_pid")) == os.getpid()
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _prepare_retry_claim(task: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """Hold or safely recover a pending-drop claim before terminal retry.

    A live claim owned by another process is never touched.  If the marker names
    an older claim whose owner is abandoned, ``claim_intent`` performs the
    atomic generation-fenced takeover; the marker is updated before any later
    failure can retain it.  The returned claim stays held until the terminal
    result and event are durable; ``None`` means custody cannot be proven.
    """
    if not task_id:
        return None
    spec = _terminalization_retry_spec(task)
    if spec is None or not str(spec.get("trigger") or "").startswith("pending_cancel"):
        return {}
    try:
        from ouroboros.cancel_intents import (
            active_intents, claim_intent, claim_is_abandoned,
        )
        intents = active_intents(DRIVE_ROOT, strict=True)
    except Exception:
        log.debug("Pending-drop claim read failed for retry %s", task_id, exc_info=True)
        return None
    intent = intents.get(task_id) if isinstance(intents, dict) else None
    if not isinstance(intent, dict):
        return {}
    if intent.get("state") == "requested":
        try:
            claim = claim_intent(DRIVE_ROOT, task_id, owner="pending_drop") or {}
        except Exception:
            log.warning("Pending-drop claim acquisition failed for retry %s", task_id, exc_info=True)
            return None
        if not claim or claim.get("claim_refused"):
            try:
                current = active_intents(DRIVE_ROOT, strict=True).get(task_id)
            except Exception:
                return None
            return {} if not isinstance(current, dict) else None
        task.update(_attach_retry_claim(task, claim))
        return dict(claim)
    if intent.get("state") != "claimed":
        return None
    claim = intent
    if not _retry_claim_matches(task, intent):
        identity = _retry_claim_identity(task, intent)
        raw_marker = task.get(_TERMINALIZATION_RETRY_FIELD)
        has_claim_metadata = isinstance(raw_marker, dict) and any(
            key in raw_marker
            for key in ("claim_request_id", "claim_owner", "claim_generation", "claim_pid")
        )
        if not identity and has_claim_metadata:
            return None
        try:
            abandoned = bool(claim_is_abandoned(intent))
        except Exception:
            log.debug("Pending-drop claim liveness failed for retry %s", task_id, exc_info=True)
            return None
        if not abandoned:
            return None
        try:
            claim = claim_intent(DRIVE_ROOT, task_id, owner="pending_drop") or {}
        except Exception:
            log.warning("Pending-drop claim takeover failed for retry %s", task_id, exc_info=True)
            return None
        if not claim or claim.get("claim_refused"):
            try:
                current = active_intents(DRIVE_ROOT, strict=True).get(task_id)
            except Exception:
                return None
            return {} if not (isinstance(current, dict) and current.get("state") == "claimed") else None
        task.update(_attach_retry_claim(task, claim))
    return dict(claim)


def _attach_retry_claim(task: Dict[str, Any], claim: Any) -> Dict[str, Any]:
    """Copy a retry row and attach only its current fenced claim metadata."""
    retained = dict(task)
    raw = retained.get(_TERMINALIZATION_RETRY_FIELD)
    if not isinstance(raw, dict) or not isinstance(claim, dict):
        return retained
    request_id = str(claim.get("request_id") or "")
    if not request_id:
        return retained
    marker = dict(raw)
    marker["claim_request_id"] = request_id
    marker["claim_owner"] = str(claim.get("claim_owner") or "pending_drop")
    try:
        marker["claim_generation"] = int(claim.get("generation"))
    except (TypeError, ValueError, OverflowError):
        marker.pop("claim_generation", None)
    try:
        marker["claim_pid"] = int(claim.get("claim_pid"))
    except (TypeError, ValueError, OverflowError):
        marker.pop("claim_pid", None)
    retained[_TERMINALIZATION_RETRY_FIELD] = marker
    return retained


def _attach_retry_event_state(task: Dict[str, Any], event_published: Optional[bool]) -> Dict[str, Any]:
    """Copy a retry row while recording whether its terminal event was sent."""
    retained = dict(task)
    if event_published is None:
        return retained
    raw = retained.get(_TERMINALIZATION_RETRY_FIELD)
    if not isinstance(raw, dict):
        return retained
    marker = dict(raw)
    if event_published or marker.get("event_published"):
        marker["event_published"] = True
    retained[_TERMINALIZATION_RETRY_FIELD] = marker
    return retained


def _make_terminalization_retry_task(
    task: Dict[str, Any], task_id: str, *, reason: str, status: str, trigger: str,
    reconcile_delegate_custody: bool = True, claim: Optional[Dict[str, Any]] = None,
    event_published: Optional[bool] = None,
) -> Dict[str, Any]:
    """Copy a killed task into durable, non-dispatchable terminalization custody."""
    retry = dict(task) if isinstance(task, dict) else {}
    retry["id"] = str(task_id)
    if retry.get("chat_id") is None or retry.get("chat_id") == "":
        # Snapshot restore requires a concrete chat identity; zero is the
        # explicit no-chat route used by headless tasks.
        retry["chat_id"] = 0
    raw_attempt = retry.get("_attempt")
    if raw_attempt is not None:
        try:
            retry["_attempt"] = max(1, int(raw_attempt))
        except (TypeError, ValueError, OverflowError):
            retry.pop("_attempt", None)
    normalized_status = str(status or "failed").strip() or "failed"
    retry_spec = {
        "reason": str(reason or "Worker shutdown terminalization was not durable.")[:2000],
        "status": normalized_status if normalized_status in _TERMINALIZATION_RETRY_STATUSES else "failed",
        "trigger": str(trigger or "terminalization_retry").strip() or "terminalization_retry",
        "reconcile_delegate_custody": bool(reconcile_delegate_custody),
    }
    if event_published:
        retry_spec["event_published"] = True
    retry[_TERMINALIZATION_RETRY_FIELD] = retry_spec
    return _attach_retry_event_state(_attach_retry_claim(retry, claim), event_published)


def _retain_terminalization_retry_task(
    task: Dict[str, Any], task_id: str, *, reason: str, status: str, trigger: str,
    reconcile_delegate_custody: bool = True, claim: Optional[Dict[str, Any]] = None,
    event_published: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return retry custody without replacing an existing stronger marker."""
    if isinstance(task, dict) and isinstance(task.get(_TERMINALIZATION_RETRY_FIELD), dict):
        return _attach_retry_event_state(_attach_retry_claim(task, claim), event_published)
    return _make_terminalization_retry_task(
        task,
        task_id,
        reason=reason,
        status=status,
        trigger=trigger,
        reconcile_delegate_custody=reconcile_delegate_custody,
        claim=claim,
        event_published=event_published,
    )


def _audit_delegate_terminal_custody(
    task_id: str, trigger: str, *, enabled: bool = True,
) -> None:
    """Best-effort delegated-run reconciliation before terminal publication."""
    if not enabled:
        return
    try:
        from ouroboros import delegate_terminal

        audit = delegate_terminal.terminal_reconcile_task(
            DRIVE_ROOT, task_id, trigger=trigger,
        )
        delegate_terminal.record_terminal_reconciliation(
            DRIVE_ROOT, task_id, audit,
        )
    except Exception:
        log.warning(
            "Terminal delegate reconciliation failed for %s", task_id,
            exc_info=True,
        )


def _terminalization_status_accepted(
    persisted: Any, requested: str, *, allow_interrupted: bool,
) -> bool:
    """Accept only a durable terminal status (or the intentional interrupted marker)."""
    from ouroboros.task_results import STATUS_INTERRUPTED, _TRULY_TERMINAL_STATUSES

    persisted_status = str(persisted or "").strip()
    requested_status = str(requested or "failed").strip() or "failed"
    return persisted_status in _TRULY_TERMINAL_STATUSES or (
        allow_interrupted
        and requested_status == STATUS_INTERRUPTED
        and persisted_status == STATUS_INTERRUPTED
    )


def _settle_terminalization_task(
    task: Dict[str, Any], *, reason: str, status: str, trigger: str,
    reconcile_delegate_custody: bool = True, allow_interrupted: bool = False,
    event_already_published: bool = False,
) -> bool:
    """Write and publish one shutdown outcome, retaining custody on any failure."""
    task_id = str(task.get("id") or "").strip()
    if not task_id:
        return False
    requested_status = str(status or "failed").strip() or "failed"
    try:
        _audit_delegate_terminal_custody(
            task_id, trigger, enabled=reconcile_delegate_custody,
        )
        persisted = _write_failure_result(
            task_id, reason=reason, status=requested_status,
        )
    except Exception:
        log.warning(
            "Failed to write failure result for task %s", task_id,
            exc_info=True,
        )
        return False
    if not _terminalization_status_accepted(
        persisted, requested_status, allow_interrupted=allow_interrupted,
    ):
        log.warning(
            "Task %s did not reach an acceptable shutdown status during kill: %r",
            task_id,
            persisted,
        )
        return False
    if event_already_published:
        return True
    try:
        if not _emit_task_done_terminal(task, task_id, str(persisted).strip()):
            log.warning("Failed to emit terminal task_done for %s", task_id)
            return False
    except Exception:
        log.warning("Failed to emit terminal task_done for %s", task_id, exc_info=True)
        return False
    return True


def _settle_terminalization_retry_task(
    task: Dict[str, Any], claim: Optional[Dict[str, Any]] = None,
) -> bool:
    """Retry a retained marker using its own immutable terminal contract."""
    request = _terminalization_retry_request(task)
    if request is None:
        return False
    if claim and claim.get("request_id"):
        try:
            from ouroboros.cancel_intents import claim_still_owned

            if not claim_still_owned(DRIVE_ROOT, str(task.get("id") or ""), claim):
                return False
        except Exception:
            log.debug("Pending-drop claim ownership check failed", exc_info=True)
            return False
    settled = _settle_terminalization_task(
        task,
        reason=request["reason"],
        status=request["status"],
        trigger=request["trigger"],
        reconcile_delegate_custody=request["reconcile_delegate_custody"],
        allow_interrupted=False,
        event_already_published=bool(
            isinstance(task.get(_TERMINALIZATION_RETRY_FIELD), dict)
            and task[_TERMINALIZATION_RETRY_FIELD].get("event_published")
        ),
    )
    if not settled:
        return False
    if not claim or not claim.get("request_id"):
        return True
    task.update(_attach_retry_event_state(task, True))
    try:
        from ouroboros.cancel_intents import active_intents, release_claim

        if release_claim(
            DRIVE_ROOT,
            str(task.get("id") or ""),
            error="terminalization retry completed",
            expected_generation=claim.get("generation"),
            request_id=str(claim.get("request_id") or ""),
        ):
            return True
        current = active_intents(DRIVE_ROOT, strict=True).get(str(task.get("id") or ""))
    except Exception:
        log.warning("Pending-drop claim release failed after retry %s", task.get("id"), exc_info=True)
        return False
    return not (isinstance(current, dict) and current.get("state") == "claimed")


def _retry_terminalization_pending() -> Tuple[List[str], List[str]]:
    """Retry retained shutdown rows before any assignment can inspect them."""
    terminalized: List[str] = []
    unresolved: List[str] = []
    retry_state_changed = False
    survivors: List[Dict[str, Any]] = []
    for task in list(PENDING):
        spec = _terminalization_retry_spec(task)
        if spec is None:
            survivors.append(task)
            continue
        raw_marker = task.get(_TERMINALIZATION_RETRY_FIELD) if isinstance(task, dict) else None
        marker_before = dict(raw_marker) if isinstance(raw_marker, dict) else None
        task_id = str(task.get("id") or "").strip() if isinstance(task, dict) else ""
        retry_claim = _prepare_retry_claim(task, task_id)
        if retry_claim is None:
            unresolved.append(task_id or "<missing-task-id>")
            survivors.append(task)
        else:
            if not task_id or not _settle_terminalization_retry_task(task, retry_claim):
                unresolved.append(task_id or "<missing-task-id>")
                survivors.append(task)
            else:
                terminalized.append(task_id)
        current_marker = task.get(_TERMINALIZATION_RETRY_FIELD) if isinstance(task, dict) else None
        if marker_before != (dict(current_marker) if isinstance(current_marker, dict) else None):
            retry_state_changed = True
    PENDING[:] = survivors
    if retry_state_changed and unresolved:
        try:
            from supervisor import queue

            if queue.persist_queue_snapshot(reason="terminalization_retry_state") is not True:
                log.warning("Failed to persist terminalization retry state")
        except Exception:
            log.warning("Failed to persist terminalization retry state", exc_info=True)
    return terminalized, unresolved


def _retry_terminalization_pending_for_assignment(queue: Any) -> None:
    """Settle shutdown custody and persist the changed queue before assignment."""
    terminalized, unresolved = _retry_terminalization_pending()
    if terminalized:
        queue.persist_queue_snapshot(reason="terminalization_retry_settled")
    if unresolved:
        log.error(
            "Shutdown terminalization rows remain deferred; continuing assignment for other tasks: %s",
            ", ".join(unresolved),
        )


_WORKER_PIDS_FILENAME = "worker_pids.json"


def _reconcile_confirmed_dead_review_owner(owner_pid: int) -> None:
    _reconcile_confirmed_dead_review_owner_for_root(DRIVE_ROOT, owner_pid)


from supervisor.worker_pool_lifecycle import _serialized_worker_lifecycle  # noqa: E402 -- moved span, decorator read at import time below


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
    events_cursor, spawned_at = events_log_cursor(), time.time()

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
            # Unassignable until the readiness seam observes this child's worker_ready row.
            new_workers[i] = Worker(wid=i, proc=proc, in_q=in_q, busy_task_id=None, reaping=True)
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
    # Readiness + SHA verification run off-loop so spawn does not block the supervisor loop.
    threading.Thread(target=_verify_worker_sha_after_spawn, args=(dict(new_workers), events_cursor, 1, spawned_at), daemon=True).start()


@_serialized_worker_lifecycle
def kill_workers(
    force: bool = True,
    *,
    result_reason: str = "Worker process crashed (crash storm). Task was not completed.",
    terminal_status: str = "",
    archive_service_logs: bool = True,
    disable_reason: str = "",
    preserve_pending: bool = False,
    preserve_running_task_ids: Optional[set[str]] = None,
    reconcile_delegate_custody: bool = True,
) -> bool:
    global _WORKER_POOL_DISABLED_REASON
    from supervisor import queue
    with _queue_lock:
        if disable_reason:
            _WORKER_POOL_DISABLED_REASON = str(disable_reason)
            # Publish the admission fence before slow process-tree teardown so
            # concurrent ingress can refuse without starting project/workspace
            # side effects while workers are being joined.
            try:
                fence_persisted = queue.persist_queue_snapshot(reason="worker_pool_disabling") is not False
            except Exception:
                fence_persisted = False
                log.warning("Failed to persist worker-pool disable fence", exc_info=True)
            if not fence_persisted:
                # No worker or queue mutation happened yet.  Re-open the local
                # pool admission so an aborted managed update can recover it.
                _WORKER_POOL_DISABLED_REASON = ""
                log.error("Worker shutdown blocked: disable fence was not durable")
                return False
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
        for w in WORKERS.values():
            try:
                if w.proc.pid and not w.proc.is_alive():
                    _reconcile_confirmed_dead_review_owner(int(w.proc.pid))
            except Exception:
                log.debug(
                    "Could not prove worker %s dead for review reconciliation",
                    w.wid,
                    exc_info=True,
                )
        WORKERS.clear()
        orphaned_ids = []
        drained_ids = []
        terminalization_retry_ids = []
        cleanup_ok = True
        try:
            done_status = terminal_status or "failed"
            preserve_running = set(preserve_running_task_ids or ())
            running_task_ids = set(RUNNING) - preserve_running
            interrupted_roots = {
                str((meta.get("task") or {}).get("root_task_id") or task_id)
                for task_id, meta in RUNNING.items()
                if isinstance(meta, dict) and task_id not in preserve_running
            }

            def _settle_killed_pending(
                task: Dict[str, Any], *, reason: str, status: str, trigger: str,
            ) -> bool:
                """Return true only after pending custody is durably terminal."""
                return _settle_terminalization_task(
                    task,
                    reason=reason,
                    status=status,
                    trigger=trigger,
                    reconcile_delegate_custody=reconcile_delegate_custody,
                )

            def _retain_killed_pending(
                task: Dict[str, Any], *, reason: str, status: str, trigger: str,
            ) -> Dict[str, Any]:
                """Keep failed pending settlement in non-dispatchable custody."""
                task_id = str(task.get("id") or "").strip()
                retry = _retain_terminalization_retry_task(
                    task,
                    task_id,
                    reason=reason,
                    status=status,
                    trigger=trigger,
                    reconcile_delegate_custody=reconcile_delegate_custody,
                )
                terminalization_retry_ids.append(task_id or "<missing-task-id>")
                return retry

            def _settle_existing_retry(task: Dict[str, Any]) -> Optional[bool]:
                """Process a marker without allowing the current kill reason to replace it."""
                if _terminalization_retry_spec(task) is None:
                    return None
                task_id = str(task.get("id") or "").strip()
                retry_claim = _prepare_retry_claim(task, task_id)
                if retry_claim is None:
                    terminalization_retry_ids.append(task_id or "<missing-task-id>")
                    return False
                if _settle_terminalization_retry_task(task, retry_claim):
                    return True
                terminalization_retry_ids.append(task_id or "<missing-task-id>")
                return False

            for task_id in list(RUNNING):
                meta = RUNNING.get(task_id) or {}
                task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
                existing_retry = next(
                    (
                        row for row in PENDING
                        if isinstance(row, dict) and str(row.get("id") or "") == str(task_id)
                        and _terminalization_retry_spec(row) is not None
                    ),
                    None,
                )
                if existing_retry is not None:
                    if _settle_existing_retry(existing_retry):
                        PENDING.remove(existing_retry)
                        orphaned_ids.append(str(task_id))
                    RUNNING.pop(str(task_id), None)
                    continue
                if task_id in preserve_running:
                    successor = dict(task)
                    successor["_attempt"] = int(meta.get("attempt") or task.get("_attempt") or 1) + 1
                    try:
                        from ouroboros.owner_hurry import retry_reset

                        retry_reset(
                            queue._task_drive_for_task(task, str(task_id)),
                            DRIVE_ROOT, str(task_id), reason="planned_restart_requeue",
                        )
                    except Exception:
                        log.debug(
                            "Planned-restart retry reset failed for %s", task_id,
                            exc_info=True,
                        )
                    PENDING.insert(0, successor)
                    RUNNING.pop(str(task_id), None)
                    continue
                if _settle_terminalization_task(
                    task,
                    reason=result_reason,
                    status=done_status,
                    trigger="worker_pool_kill",
                    reconcile_delegate_custody=reconcile_delegate_custody,
                    allow_interrupted=True,
                ):
                    if archive_service_logs:
                        try:
                            from ouroboros.tools.services import archive_task_service_logs

                            archive_task_service_logs(pathlib.Path(DRIVE_ROOT), str(task_id), task)
                        except Exception:
                            log.debug("Failed to archive service logs for task %s", task_id, exc_info=True)
                    orphaned_ids.append(task_id)
                else:
                    # The worker is already dead, but the durable outcome or its
                    # terminal event is not proven. Move the row to a
                    # non-dispatchable retry custody that survives snapshot/boot.
                    retry_task = _retain_terminalization_retry_task(
                        task,
                        str(task_id),
                        reason=result_reason,
                        status=done_status,
                        trigger="worker_pool_kill",
                        reconcile_delegate_custody=reconcile_delegate_custody,
                    )
                    RUNNING.pop(str(task_id), None)
                    replaced = False
                    for index, row in enumerate(PENDING):
                        if isinstance(row, dict) and str(row.get("id") or "") == str(task_id):
                            merged = dict(row)
                            merged.update(retry_task)
                            PENDING[index] = merged
                            replaced = True
                            break
                    if not replaced:
                        PENDING.append(retry_task)
                    terminalization_retry_ids.append(str(task_id))
            if preserve_pending:
                kept = []
                for task in PENDING:
                    retry_outcome = _settle_existing_retry(task)
                    if retry_outcome is not None:
                        if retry_outcome:
                            drained_ids.append(str(task.get("id") or ""))
                        else:
                            kept.append(task)
                        continue
                    if str(task.get("id") or "") in preserve_running:
                        kept.append(task)
                        continue
                    parent_id = str(task.get("parent_task_id") or "")
                    root_id = str(task.get("root_task_id") or "")
                    if parent_id and (parent_id in running_task_ids or root_id in interrupted_roots):
                        tid = str(task.get("id") or "")
                        if _settle_killed_pending(
                            task,
                            reason="Parent task was interrupted before this child started.",
                            status="cancelled",
                            trigger="pending_parent_interrupted",
                        ):
                            drained_ids.append(tid)
                        else:
                            kept.append(_retain_killed_pending(
                                task,
                                reason="Parent task was interrupted before this child started.",
                                status="cancelled",
                                trigger="pending_parent_interrupted",
                            ))
                        continue
                    kept.append(task)
                PENDING[:] = kept
            else:
                # Keep the previous snapshot authoritative until every drained
                # row has either durable terminal custody or been requeued.
                drained = queue.drain_all_pending(persist=False)
                for task in drained:
                    tid = str(task.get("id") or "").strip()
                    retry_outcome = _settle_existing_retry(task)
                    if retry_outcome is not None:
                        if retry_outcome:
                            drained_ids.append(tid)
                        else:
                            PENDING.append(task)
                        continue
                    if _settle_killed_pending(
                        task,
                        reason=result_reason,
                        status=done_status,
                        trigger="pending_pool_kill",
                    ):
                        drained_ids.append(tid)
                    else:
                        # No id, failed durable write, or failed notification:
                        # retain non-dispatchable custody so a later supervisor
                        # pass can retry without starting the task.
                        PENDING.append(_retain_killed_pending(
                            task,
                            reason=result_reason,
                            status=done_status,
                            trigger="pending_pool_kill",
                        ))
            if orphaned_ids or drained_ids or terminalization_retry_ids:
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "zombie_prevention_cleanup",
                        "orphaned_running": orphaned_ids,
                        "drained_pending": drained_ids,
                        "terminalization_retry": terminalization_retry_ids,
                    },
                )
        except Exception:
            cleanup_ok = False
            log.warning("Zombie prevention cleanup failed", exc_info=True)
        for terminal_id in orphaned_ids:
            RUNNING.pop(str(terminal_id), None)
    try:
        snapshot_ok = queue.persist_queue_snapshot(reason="kill_workers") is not False
    except Exception:
        snapshot_ok = False
        log.warning("Failed to persist queue snapshot after worker shutdown", exc_info=True)
    if not snapshot_ok:
        log.error("Worker shutdown completed without a durable final queue snapshot")
    if cleared_running:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "running_cleared_on_kill", "count": cleared_running,
                "force": force,
            },
        )
    return bool(cleanup_ok and snapshot_ok)


def _persist_pending_terminalization_retries(task_ids: List[str]) -> None:
    """Record retained pending-terminal rows and make their custody restart-safe."""
    if not task_ids:
        return
    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {"ts": utc_now_iso(), "type": "pending_terminal_event_retry", "task_ids": task_ids},
    )
    try:
        from supervisor import queue

        if not queue.persist_queue_snapshot(reason="pending_terminal_event_retry"):
            log.warning("Failed to persist pending terminal-event retry custody")
    except Exception:
        log.warning(
            "Failed to persist pending terminal-event retry custody",
            exc_info=True,
        )


def _read_pending_cancel_intents(
    active_intents: Any, assignment: Dict[str, bool],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Read the live cancel projection, preserving authority failure."""
    try:
        current = active_intents(DRIVE_ROOT, strict=True)
    except Exception:
        assignment["safe"] = False
        log.error(
            "Cancel-intent authority became unreadable during pending cleanup; "
            "assignment is blocked",
            exc_info=True,
        )
        return None
    if not isinstance(current, dict):
        assignment["safe"] = False
        log.error(
            "Cancel-intent authority returned a non-object projection; "
            "assignment is blocked",
        )
        return None
    return current


def _settle_cancelled_pending_row(
    task: Dict[str, Any],
    task_id: str,
    marker: Optional[Dict[str, Any]],
    current_intents: Dict[str, Dict[str, Any]],
    *,
    pending_tail: List[Dict[str, Any]],
    survivors: List[Dict[str, Any]],
    dropped: List[str],
    terminalization_retry_ids: List[str],
    assignment: Dict[str, bool],
    active_intents: Any,
    claim_intent: Any,
    release_claim: Any,
    settle_intent: Any,
    intent_outcome_fields: Any,
    write_task_result: Any,
    status_cancelled: str,
) -> bool:
    """Settle one pre-assignment cancellation; return True to abort the pass.

    The caller holds ``_queue_lock`` and owns the ordered pending partition.
    Keeping those containers explicit preserves the existing whole-pass abort
    semantics without giving this leaf a second queue owner.
    """

    def _release_pending_claim(claim: Dict[str, Any], error: str) -> bool:
        """Prove that this drop's claim is no longer live after a failed step."""
        request_id = str(claim.get("request_id") or "")
        if not request_id:
            return True
        try:
            if release_claim is not None and release_claim(
                DRIVE_ROOT, task_id, error=error,
                expected_generation=claim.get("generation"), request_id=request_id,
            ):
                return True
        except Exception:
            log.warning("pending-drop claim release failed for %s", task_id, exc_info=True)
        if active_intents is None:
            return False
        current = _read_pending_cancel_intents(active_intents, assignment)
        if current is None:
            return False
        current = current.get(task_id)
        if not isinstance(current, dict) or current.get("state") != "claimed":
            return True
        # A changed live claim belongs to another custody owner; it is not proof
        # that this drop released its own claim.
        return False

    def _pending_claim_is_ours(claim: Dict[str, Any]) -> bool:
        """Allow publication only while an unresolved claim is still ours."""
        if not claim or not claim.get("request_id") or active_intents is None:
            return False
        current = _read_pending_cancel_intents(active_intents, assignment)
        if current is None:
            return False
        current = current.get(task_id)
        try:
            return bool(
                isinstance(current, dict)
                and current.get("state") == "claimed"
                and str(current.get("request_id") or "") == str(claim.get("request_id") or "")
                and int(current.get("generation") or 0) == int(claim.get("generation") or 0)
                and int(current.get("claim_pid") or 0) == os.getpid()
                and str(current.get("claim_owner") or "") == "pending_drop"
            )
        except (TypeError, ValueError, OverflowError):
            return False

    def _pending_claim_is_released(claim: Dict[str, Any]) -> bool:
        """Observe that a failed settle already removed the claim."""
        if not claim or not claim.get("request_id") or active_intents is None:
            return not claim or not claim.get("request_id")
        current = _read_pending_cancel_intents(active_intents, assignment)
        if current is None:
            return False
        current = current.get(task_id)
        return not (isinstance(current, dict) and current.get("state") == "claimed")

    # AR2-2 settle-owner unity: claim before writing. A marker may carry the
    # exact claim that this same pending-drop process still owns; recognize
    # that fenced refusal instead of dropping its custody.
    claim: Dict[str, Any] = {}
    if task_id in current_intents:
        if claim_intent is None:
            survivors.append(task)
            return False
        try:
            claim = claim_intent(DRIVE_ROOT, task_id, owner="pending_drop") or {}
        except Exception:
            # A raised claim write/read is not evidence that another custody
            # owner won. Retain this row and stop the pass so a transiently
            # unreadable projection cannot lose queue custody or dispatch it.
            assignment["safe"] = False
            log.error(
                "Pending-drop claim authority is unreadable for %s; "
                "assignment is blocked",
                task_id,
                exc_info=True,
            )
            survivors.append(task)
            survivors.extend(pending_tail)
            return True
        if not isinstance(claim, dict):
            assignment["safe"] = False
            log.error(
                "Pending-drop claim returned a non-object for %s; "
                "assignment is blocked",
                task_id,
            )
            survivors.append(task)
            survivors.extend(pending_tail)
            return True
        if claim.get("claim_refused"):
            if marker is not None and _retry_claim_matches(task, claim):
                pass
            elif marker is not None:
                survivors.append(task)
                return False
            else:
                # A different live custody owner is waiting for the queue lock
                # this pass holds. Keep the row for that owner and abort dispatch.
                assignment["safe"] = False
                survivors.append(task)
                survivors.extend(pending_tail)
                return True
    if task_id in current_intents and not claim:
        # The projection may have changed after the snapshot. Without an owned
        # claim this drop cannot fence its settle, so defer the whole pass.
        assignment["safe"] = False
        log.info(
            "Pending cancellation custody changed during claim for %s; "
            "deferring the assignment pass",
            task_id,
        )
        survivors.append(task)
        survivors.extend(pending_tail)
        return True
    intent = claim or current_intents.get(task_id) or {}
    try:
        cost_fields = reconstruct_task_cost(task_id, fields=True)
    except Exception:
        cost_fields = {
            "cost_accounting_status": "unavailable",
            "cost_final": False,
            # ABI-3: honest name only — nothing emits the retired alias.
            "accounted_upper_bound_usd": None,
        }
    try:
        stored = write_task_result(
            DRIVE_ROOT, task_id, status_cancelled,
            strict_existing_dict=True,
            result="Cancelled before start.", **cost_fields,
            **intent_outcome_fields(intent),
        ) or {}
        stored_status = str(stored.get("status") or "").strip()
        if not stored_status:
            raise ValueError("cancel result writer returned no durable status")
    except Exception:
        log.debug("Failed to finalize cancelled pending task %s", task_id, exc_info=True)
        released = _release_pending_claim(claim, "pending-drop persistence failed")
        if released:
            dropped.append(task_id)
        else:
            survivors.append(_retain_terminalization_retry_task(
                task, task_id,
                reason="Cancelled pending task result is not durable and its claim needs recovery.",
                status=status_cancelled, trigger="pending_cancel_result",
                reconcile_delegate_custody=False, claim=claim,
            ))
            terminalization_retry_ids.append(task_id)
        return False
    intent_present = task_id in current_intents
    # Presence of an active projection is the custody fact; availability of the
    # settle helper cannot turn a live claim into a settled one.
    intent_settled = not intent_present
    claim_released = intent_settled
    if intent_present and settle_intent is not None:
        try:
            # The durable scope is re-read by settle_intent; cascade refusal may
            # auto-release this exact claim and is handled as replayable below.
            settled_row = settle_intent(
                DRIVE_ROOT, task_id,
                outcome=("cancelled" if stored_status == status_cancelled else "already_settled"),
                detail=("dropped before assignment" if stored_status == status_cancelled else stored_status),
                expected_generation=claim.get("generation"),
                request_id=str(claim.get("request_id") or ""),
            )
            intent_settled = settled_row is not None
        except Exception:
            intent_settled = False
            log.debug("Failed to settle cancel intent for pending %s", task_id, exc_info=True)
        if not intent_settled:
            claim_released = _pending_claim_is_released(claim)
    raw_marker = task.get(_TERMINALIZATION_RETRY_FIELD)
    event_already_published = bool(
        isinstance(raw_marker, dict) and raw_marker.get("event_published")
    )
    claim_unresolved = not intent_settled and not claim_released
    event_published = event_already_published
    can_publish = not claim_unresolved or _pending_claim_is_ours(claim)
    if can_publish and not event_already_published:
        try:
            event_published = bool(
                _emit_task_done_terminal(task, task_id, stored_status, cost_fields=cost_fields)
            )
        except Exception:
            event_published = False
            log.warning(
                "Failed to emit terminal task_done for cancelled pending task %s",
                task_id, exc_info=True,
            )
    if claim_unresolved and can_publish:
        claim_released = _release_pending_claim(
            claim, "pending-drop intent settlement failed",
        )
        claim_unresolved = not intent_settled and not claim_released
    if claim_unresolved:
        survivors.append(_retain_terminalization_retry_task(
            task, task_id,
            reason="Cancelled pending task has an unresolved intent claim.",
            status=status_cancelled, trigger="pending_cancel_intent",
            reconcile_delegate_custody=False, claim=claim,
            event_published=event_published,
        ))
        terminalization_retry_ids.append(task_id)
    elif event_published:
        dropped.append(task_id)
    else:
        survivors.append(_retain_terminalization_retry_task(
            task, task_id,
            reason="Cancelled pending task has an unpublished terminal event.",
            status=status_cancelled, trigger="pending_cancel_event",
            reconcile_delegate_custody=False, claim=claim,
            event_published=False,
        ))
        terminalization_retry_ids.append(task_id)
    return False


def _drop_cancelled_pending() -> bool:
    """Remove pending tasks cancelled/finished between scheduling and assignment
    so a cancelled subagent never actually starts. Caller holds _queue_lock.

    The pre-assignment consult of the durable cancel-intent projection (phase A):
    a task with an active intent (or a legacy ``cancel_requested`` latch file) is
    settled as ``cancelled`` with a reconstructed — usually confirmed pre-start
    zero — cost, never assigned to a worker.  Return ``False`` when the
    projection cannot be read authoritatively; the caller must then abort the
    assignment pass before selecting a worker.

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
        return True
    try:
        from ouroboros.task_results import (
            STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, _TRULY_TERMINAL_STATUSES,
            load_task_result, write_task_result,
        )
    except Exception:
        log.error(
            "Pending task-result authority imports are unavailable; assignment is blocked",
            exc_info=True,
        )
        return False
    assignment = {"safe": True}
    try:
        from ouroboros.cancel_intents import (
            active_intents, claim_intent, release_claim, settle_intent,
        )
        if not isinstance(active_intents(DRIVE_ROOT, strict=True), dict):
            raise TypeError("cancel-intent authority returned a non-object projection")
    except Exception:
        log.error(
            "Cancel-intent authority is unreadable; pending assignment is blocked",
            exc_info=True,
        )
        return False
    try:
        from supervisor.task_lifecycle import _intent_outcome_fields
    except Exception:
        def _intent_outcome_fields(_intent):  # type: ignore[misc]
            return {}


    survivors: List[Dict[str, Any]] = []
    dropped: List[str] = []
    terminalization_retry_ids: List[str] = []
    hold_state_changed = False
    pending_rows = list(PENDING)
    for index, t in enumerate(pending_rows):
        if not assignment["safe"]:
            survivors.extend(pending_rows[index:])
            break
        tid = str(t.get("id") or "") if isinstance(t, dict) else ""
        authority_hold = bool(
            isinstance(t, dict)
            and isinstance(t.get(_CANCEL_INTENT_AUTHORITY_HOLD_FIELD), dict)
        )
        marker = _terminalization_retry_spec(t)
        marker_trigger = str((t.get(_TERMINALIZATION_RETRY_FIELD) or {}).get("trigger") or "") if isinstance(t, dict) else ""
        # Existing shutdown custody owns its outcome.  Only the cancellation
        # markers below may re-enter this drop path to recover an intent claim.
        if marker is not None and not marker_trigger.startswith("pending_cancel"):
            survivors.append(t)
            continue
        status = ""
        authority_error = False
        if tid:
            try:
                existing = load_task_result(DRIVE_ROOT, tid, strict=True)
                status = str((existing or {}).get("status") or "")
            except Exception:
                authority_error = True
        if authority_error:
            if authority_hold:
                assignment["safe"] = False
                log.error(
                    "Pending cancel-authority hold remains: task-result authority "
                    "is unreadable for %s",
                    tid or "<missing-task-id>",
                )
                survivors.extend(pending_rows[index:])
                break
            if marker is not None or not tid:
                survivors.append(t)
            else:
                survivors.append(_make_terminalization_retry_task(
                    t, tid,
                    reason="Pending task result authority is unreadable; dispatch is blocked.",
                    status="failed", trigger="pending_result_authority",
                    reconcile_delegate_custody=False,
                ))
                terminalization_retry_ids.append(tid)
            continue
        current_intents = _read_pending_cancel_intents(active_intents, assignment)
        if current_intents is None:
            # Keep this row and every row not yet examined.  The caller will
            # abort the assignment pass; no ordinary pending row may cross the
            # dispatch boundary while cancellation authority is unknown.
            survivors.extend(pending_rows[index:])
            break
        if authority_hold:
            if not tid:
                assignment["safe"] = False
                survivors.extend(pending_rows[index:])
                break
            hold_state_changed = True
            if status in _TRULY_TERMINAL_STATUSES:
                dropped.append(tid)
                continue
            # The projection and result are now authoritative.  Remove only the
            # restore-time nonterminal hold; an active intent or legacy latch
            # falls through into the ordinary cancellation-custody path below.
            t = dict(t)
            t.pop(_CANCEL_INTENT_AUTHORITY_HOLD_FIELD, None)
        if marker is not None and not (
            status == STATUS_CANCEL_REQUESTED
            or (current_intents is not None and tid in current_intents)
        ):
            survivors.append(t)
            continue
        if tid and (
            status == STATUS_CANCEL_REQUESTED or tid in current_intents
        ):
            abort_assignment = _settle_cancelled_pending_row(
                t,
                tid,
                marker,
                current_intents,
                pending_tail=pending_rows[index + 1:],
                survivors=survivors,
                dropped=dropped,
                terminalization_retry_ids=terminalization_retry_ids,
                assignment=assignment,
                active_intents=active_intents,
                claim_intent=claim_intent,
                release_claim=release_claim,
                settle_intent=settle_intent,
                intent_outcome_fields=_intent_outcome_fields,
                write_task_result=write_task_result,
                status_cancelled=STATUS_CANCELLED,
            )
            if abort_assignment:
                break
            continue
        if status in _TRULY_TERMINAL_STATUSES and marker is None:
            dropped.append(tid)
            continue
        survivors.append(t)
    PENDING[:] = survivors
    if hold_state_changed:
        try:
            from supervisor import queue

            persisted = queue.persist_queue_snapshot(
                reason="cancel_intent_authority_hold_resolved",
            )
        except Exception:
            persisted = False
            log.error(
                "Failed to persist resolved cancel-intent authority hold",
                exc_info=True,
            )
        if persisted is not True:
            # A resumed ordinary row must not cross dispatch until removal of
            # its restart-visible hold is durable.  Restore the pre-pass queue;
            # any terminal side effects are monotonic and will be observed on
            # the next pass.
            PENDING[:] = pending_rows
            assignment["safe"] = False
    if dropped:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {"ts": utc_now_iso(), "type": "pending_cancelled_dropped", "task_ids": dropped},
        )
    _persist_pending_terminalization_retries(terminalization_retry_ids)
    return bool(assignment["safe"])


def _normalize_pending_task_depth(task: Dict[str, Any]) -> str:
    """Normalize a pending depth, or return the typed-ingress error."""
    try:
        task["depth"] = parse_task_depth(task.get("depth"), default=0)
    except (TypeError, ValueError) as exc:
        return str(exc)
    return ""


def _terminalize_invalid_pending_depth(task: Dict[str, Any], detail: str) -> bool:
    """Give a bypassed pending row terminal custody before any worker dispatch."""
    task_id = str(task.get("id") or "").strip()
    if not task_id:
        return False
    result_root = pathlib.Path(task.get("budget_drive_root") or DRIVE_ROOT)
    raw_depth = task.get("depth")
    if raw_depth is not None and not isinstance(raw_depth, (str, int, float, bool)):
        raw_depth = repr(raw_depth)[:200]
    try:
        from ouroboros.task_results import STATUS_FAILED, write_task_result

        try:
            cost_fields = reconstruct_task_cost(task_id, fields=True, drive_root=result_root)
        except Exception:
            cost_fields = {
                "cost_accounting_status": "unavailable",
                "cost_final": False,
                # ABI-3: honest name only — nothing emits the retired alias.
                "accounted_upper_bound_usd": None,
            }
        stored = write_task_result(
            result_root,
            task_id,
            STATUS_FAILED,
            strict_existing_dict=True,
            reason_code="invalid_task_depth",
            result=f"Task was not dispatched: {str(detail)[:500]}",
            # Keep the rejected value as evidence, never as an executable depth.
            depth=0,
            raw_task_depth=raw_depth,
            invalid_task_depth=True,
            parent_task_id=task.get("parent_task_id"),
            root_task_id=task.get("root_task_id"),
            delegation_role=task.get("delegation_role"),
            chat_id=task.get("chat_id"),
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            **cost_fields,
        )
        if str((stored or {}).get("status") or "") != STATUS_FAILED:
            return False
    except Exception:
        log.warning("Failed to terminalize invalid pending task %s", task_id, exc_info=True)
        return False
    # Durable custody is authoritative; notification and diagnostics are
    # best-effort and must not make an already-terminal row look uncommitted.
    _emit_task_done_terminal(
        task, task_id, "failed", reason_code="invalid_task_depth", cost_fields=cost_fields,
    )
    try:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "pending_invalid_task_depth",
                "task_id": task_id,
                "raw_task_depth": raw_depth,
            },
        )
    except Exception:
        log.debug("Failed to record invalid pending depth for %s", task_id, exc_info=True)
    return True


def _quarantine_invalid_pending_depths() -> tuple[list[str], list[str]]:
    """Settle malformed pending rows before budget or capacity filters run."""
    terminalized: list[str] = []
    unresolved: list[str] = []
    for index in range(len(PENDING) - 1, -1, -1):
        task = PENDING[index]
        if not isinstance(task, dict):
            continue
        if _terminalization_retry_spec(task) is not None:
            # A shutdown custody row has an explicit retry contract; depth
            # quarantine must not rewrite it before its intended outcome lands.
            continue
        detail = _normalize_pending_task_depth(task)
        if not detail:
            continue
        task_id = str(task.get("id") or "").strip()
        if _terminalize_invalid_pending_depth(task, detail):
            PENDING.pop(index)
            if task_id:
                terminalized.append(task_id)
        else:
            unresolved.append(task_id or "<missing-task-id>")
    if terminalized:
        try:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "pending_invalid_task_depth_quarantined",
                    "task_ids": terminalized,
                },
            )
        except Exception:
            log.debug("Failed to record invalid pending-depth quarantine", exc_info=True)
    return terminalized, unresolved


def _invalid_depth_deferred(task: dict, deferred_ids: set[str]) -> bool:
    task_id = str(task.get("id") or "").strip()
    return (
        _terminalization_retry_spec(task) is not None
        or (task_id or "<missing-task-id>") in deferred_ids
    )


def _drop_assignable_evolution_tasks(deferred_ids: set[str]) -> list[str]:
    """Remove policy-blocked evolution rows while retaining deferred custody."""
    blocked_ids = []
    kept = []
    for task in PENDING:
        if str(task.get("type") or "") == "evolution" and not _invalid_depth_deferred(task, deferred_ids):
            blocked_ids.append(str(task.get("id") or ""))
        else:
            kept.append(task)
    PENDING[:] = kept
    return blocked_ids


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


def _worker_crash_storm_detected(
    *, busy_crashes: int, dead_detections: int, crashed_tasks: List[Dict[str, Any]]
) -> bool:
    now = time.time()
    alive_now = sum(1 for worker in WORKERS.values() if worker.proc.is_alive())
    if dead_detections:
        # Only count busy crashes or all-workers-dead as storm signals.
        if busy_crashes > 0 or alive_now == 0:
            CRASH_TS.extend([now] * max(1, dead_detections))
        else:
            CRASH_TS.clear()

    CRASH_TS[:] = [stamp for stamp in CRASH_TS if (now - stamp) < 60.0]
    if len(CRASH_TS) < 3:
        return False

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
    return True


# v7next F1 (D08): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from supervisor.worker_assignment import (  # noqa: E402, F401 -- intentional public re-exports
    _cancel_unauthorized_evolution,
    _evolution_assignment_error,
    assign_tasks,
)
from supervisor.worker_chat_lane import (  # noqa: E402, F401 -- intentional public re-exports
    _broadcast_task_named,
    _handle_chat_direct_locked,
    _run_chat_task,
    auto_resume_after_restart,
    handle_chat_direct,
    handle_chat_ephemeral,
)
from supervisor.worker_health import (  # noqa: E402, F401 -- intentional public re-exports
    _emit_task_done_terminal,
    _ensure_workers_healthy_locked,
    ensure_workers_healthy,
    terminal_task_metadata,
)
from supervisor.worker_pool_lifecycle import (  # noqa: E402, F401 -- intentional public re-exports
    _WORKER_LIFECYCLE_LOCK,
    _first_worker_event_since,
    _kill_survivors,
    _record_worker_pids,
    _serialized_worker_lifecycle,
    _verify_worker_sha_after_spawn,
    _worker_pids_path,
    _write_failure_result,
    events_log_cursor,
    kill_workers_for_update,
    reap_orphaned_workers,
    respawn_worker,
)
from supervisor.worker_process import (  # noqa: E402, F401 -- intentional public re-exports
    WORKER_LOG_SINK_SUPPRESSED_TYPES,
    _bind_worker_repo_root,
    _current_custody_session_id,
    _log_worker_crash,
    _prepare_worker_task_runtime,
    worker_main,
)
from supervisor.worker_promotion import (  # noqa: E402, F401 -- intentional public re-exports
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
