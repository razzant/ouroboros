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
from ouroboros.depth_evidence import parse_task_depth
from ouroboros.review_owner_custody import (
    reconcile_confirmed_dead_review_owner as _reconcile_confirmed_dead_review_owner_for_root,
)
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
    # X3 (owner 11=B): the repair is admitted against ONE exact payload state.
    # An unreadable payload cannot anchor a hash chain — fail closed here, not
    # after the task has already spent rounds.
    try:
        from ouroboros.skill_loader import compute_content_hash

        base_content_hash = compute_content_hash(payload_dir)
    except Exception:
        return None, "skill_repair_payload_unreadable"
    return {
        "mode": canonical.mode,
        "skill_name": canonical.skill_name,
        "payload_root": canonical.payload_root,
        "allow_enable": False,
        "allow_review": True,
        "_base_content_hash": base_content_hash,
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
            load_task_result(
                getattr(ctx, "DRIVE_ROOT", DRIVE_ROOT), task_id, strict=True,
            )
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
    attachment_manifest: list[dict] = []
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
    inherited_attachment_manifest = _apply_presence_promotion_authority(
        evt, task, objective=objective, expected_output=expected_output,
    )
    attachment_manifest, attachment_rejection = _stage_promoted_initial_attachments(
        evt, task, tid, inherited_manifest=inherited_attachment_manifest,
    )
    if attachment_rejection is not None:
        return attachment_rejection
    if repair_constraint is not None:
        # X3: bind the admission hash to the REAL task id, durably, before the
        # task exists anywhere else — every payload write CAS-checks this chain.
        # FAIL CLOSED, like the unreadable-payload branch above: a repair admitted
        # without its binding CAS-checks nothing (every later check no-ops), which
        # is precisely the drift-blind repair this mechanism replaces.
        _base_content_hash = str(repair_constraint.pop("_base_content_hash", "") or "")
        try:
            from ouroboros.skill_repair_admission import record_repair_admission

            record_repair_admission(
                DRIVE_ROOT, str(repair_constraint.get("skill_name") or ""),
                task_id=tid, base_content_hash=_base_content_hash)
        except Exception:
            log.warning("Failed to record skill repair admission for %s", tid, exc_info=True)
            return _reject_promoted_after_attachment_stage({
                "status": "needs_manual_target",
                "reason": "skill_repair_admission_unwritable",
                "task_id": tid,
            }, attachment_manifest)
        # Must be present before attach_task_contract so the managed root task
        # enters execution with its confined repair profile, never ephemeral.
        task["task_constraint"] = repair_constraint
    # Ingress-captured origin identity rides the task record (post-hoc UI convert
    # reads it from the persisted result — never re-derived from content).
    if isinstance(evt.get("source_ref"), dict) and evt.get("source_ref"):
        task["origin_message_ref"] = dict(evt["source_ref"])
        if isinstance(evt.get("source_text"), str) and evt.get("source_text"):
            task["origin_message_text"] = evt["source_text"]
    if isinstance(evt.get("predecessor_authority_source"), dict):
        task["predecessor_authority_source"] = dict(evt["predecessor_authority_source"])
    # Owner Surface Fact: the promoting turn's sending-surface fact lands in
    # METADATA (the renderer reads task["metadata"]["client_surface"]), never a
    # top-level key — and metadata may not exist yet (only force_plan creates it).
    if isinstance(evt.get("client_surface"), dict) and evt.get("client_surface"):
        task.setdefault("metadata", {})["client_surface"] = dict(evt["client_surface"])
    pid = str(evt.get("project_id") or "").strip()
    if pid:
        # Deletion closes admission before cancellation/quiescence begins. Check
        # the durable lifecycle before creating projects or child drives;
        # enqueue_task repeats this check atomically under the queue lock.
        try:
            from ouroboros.projects_registry import get_reserved_project

            existing_project = get_reserved_project(DRIVE_ROOT, pid)
            existing_lifecycle = str((existing_project or {}).get("lifecycle") or "active")
            if existing_project is not None and existing_lifecycle != "active":
                return _reject_promoted_after_attachment_stage({
                    "status": "needs_manual_target",
                    "reason": "project_routing_fence",
                    "project_lifecycle": existing_lifecycle,
                    "task_id": tid,
                }, attachment_manifest)
        except Exception:
            log.warning("promote: project admission lookup failed for %s", pid, exc_info=True)
            return _reject_promoted_after_attachment_stage({
                "status": "needs_manual_target",
                "reason": "project_routing_fence_lookup_failed",
                "task_id": tid,
            }, attachment_manifest)
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
                return _reject_promoted_after_attachment_stage({
                    "status": "needs_manual_target",
                    "reason": "project_binding_failed",
                    "task_id": tid,
                }, attachment_manifest)
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
            if evt.get("_source_created") and not (project or {}).get("created"):
                # The source-resolution half of THIS promote registered the
                # project off-loop (_prepare_promote_source_off_loop) — same
                # agent-initiated creation, so the announce gate honors it.
                project = {**(project or {}), "created": True}
            _announce_created_project(project, tid, task=task)
        except Exception:
            log.warning("promote: project registration failed for %s", pid, exc_info=True)
            return _reject_promoted_after_attachment_stage({
                "status": "needs_manual_target",
                "reason": "project_registration_failed",
                "task_id": tid,
            }, attachment_manifest)
    # Workspace admission (v6.58.0 SSOT + the Q10=A auto-provision) lives in one
    # helper so this entry point stays readable and under the method gate.
    workspace_outcome = _admit_promoted_workspace(evt, ctx, task, pid=pid, tid=tid)
    if workspace_outcome is not None:
        return _reject_promoted_after_attachment_stage(
            workspace_outcome, attachment_manifest,
        )
    if not _relocate_promoted_attachments(task, tid, attachment_manifest):
        return _reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": "attachment_admission_rejected",
            "detail": "Attachment staging could not be finalized (reason=staging_unavailable).",
            "attachment_manifest": [
                {
                    "ordinal": row.get("ordinal", index),
                    "status": "rejected",
                    "reason": "staging_unavailable",
                    "label": str(row.get("label") or f"attachment {index + 1}"),
                }
                for index, row in enumerate(attachment_manifest)
            ],
            "task_id": tid,
        }, attachment_manifest)
    if attachment_manifest:
        from ouroboros.gateway.tasks import _render_attachment_lines

        rendered = _render_attachment_lines(attachment_manifest)
        task["text"] = f"{task['text']}\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]"
        public_manifest = [dict(row) for row in attachment_manifest]
        task["attachments"] = public_manifest
        task["attachment_images"] = [row for row in public_manifest if row.get("is_image")]
        if isinstance(task.get("task_contract"), dict):
            task["task_contract"]["attachment_manifest"] = public_manifest
    attach_task_contract(task)
    admitted = ctx.enqueue_task(task)
    if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
        return _reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": str(admitted.get("_admission_blocked") or "admission_fence"),
            "project_lifecycle": str(admitted.get("_project_lifecycle") or ""),
            "task_id": tid,
        }, attachment_manifest)
    # A positive promote confirmation is allowed only after the durable queue
    # projection exists.  The event handler writes the scheduled task result
    # after the routing receipt; keeping that last step outside this function
    # makes the result itself the cross-process admission receipt.
    persist_snapshot = getattr(ctx, "persist_queue_snapshot", None)
    if not callable(persist_snapshot):
        return _reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": "queue_snapshot_persist_unavailable",
            "task_id": tid,
            "admission_started": True,
        }, attachment_manifest)
    try:
        if persist_snapshot(reason="promote_chat_to_task") is False:
            return _reject_promoted_after_attachment_stage({
                "status": "needs_manual_target",
                "reason": "queue_snapshot_persist_failed",
                "task_id": tid,
                "admission_started": True,
            }, attachment_manifest)
    except Exception:
        log.warning("promote: queue snapshot persist failed for %s", tid, exc_info=True)
        return _reject_promoted_after_attachment_stage({
            "status": "needs_manual_target",
            "reason": "queue_snapshot_persist_failed",
            "task_id": tid,
            "admission_started": True,
        }, attachment_manifest)
    # v6.82 (P5) disclosed residual: a PROMOTED root carries the host-attested
    # `cancelable` marker from its first RUNNING relay, not from enqueue — the
    # promote path emits no owner-facing progress frame of its own, and minting a
    # marker-only bubble would either add chat noise or bypass the canonical
    # message seam (tests/test_heartbeat_presentation.py). While it is still
    # PENDING the Dashboard Activity row cancels it; the card action appears once
    # it starts.
    # A project root may execute from a forked child drive.  Its budget-root
    # result therefore receives this admitted contract before worker startup.
    outcome = _promoted_scheduled_outcome(task, admitted, tid)
    if attachment_manifest:
        outcome["attachment_manifest"] = [dict(row) for row in attachment_manifest]
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
                _fail_promoted_task_loudly(
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

    resolved_ws, ws_error = resolve_room_workspace(
        drive_root=DRIVE_ROOT,
        system_repo_dir=REPO_DIR,
        project_id=pid,
        explicit_workspace=str(evt.get("workspace_root") or "").strip(),
        workspace_sentinel=str(evt.get("workspace") or ""),
    )
    if ws_error:
        _fail_promoted_task_loudly(ctx, task, ws_error)
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


def _fail_promoted_task_loudly(ctx: Any, task: dict, ws_error: str) -> None:
    """v6.58.0 loud-fail invariant: a room task whose workspace is SET-but-unusable
    is terminally FAILED at admission with a visible card + chat message — never
    silently admitted workspace-less (which would run the self_modification profile
    over the system repo). Never raises."""
    tid = str(task.get("id") or "")
    chat_id = 0
    try:
        chat_id = int(task.get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    message = (
        f"⚠️ WORKSPACE_UNUSABLE: task {tid} was NOT started — {ws_error} "
        "Fix the project's working folder (Projects → this project) or re-promote with "
        "workspace='none' for a folder-less task."
    )
    try:
        from ouroboros.task_results import STATUS_FAILED, write_task_result

        write_task_result(
            DRIVE_ROOT, tid, STATUS_FAILED,
            reason_code="workspace_unusable",
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

            running = getattr(ctx, "RUNNING", None)
            pending = getattr(ctx, "PENDING", None)
            if isinstance(running, dict):
                with _queue_lock:
                    mark_task_project(running, pending, tid, pid)
        except Exception:
            log.debug("ensure_project_scope: RUNNING project_id update failed for %s", tid, exc_info=True)
        if proj_chat:
            try:
                from supervisor.message_bus import get_bridge

                get_bridge().broadcast({"type": "projects_changed", "project_id": pid, "chat_id": proj_chat})
            except Exception:
                log.debug("ensure_project_scope: projects_changed broadcast failed for %s", pid, exc_info=True)
        running = getattr(ctx, "RUNNING", None)
        row = running.get(tid) if isinstance(running, dict) else None
        _announce_created_project(
            project, tid, task=row.get("task") if isinstance(row, dict) else None,
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
        from ouroboros.usage_ledger import DISPLAY_LOCK_TIMEOUT_SEC

        remaining = budget_remaining(
            load_state(), strict=True,
            lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC,
            allow_stale=True,
        )
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


from supervisor.log_addressing import TurnEventQueue as _TurnEventQueue  # noqa: E402


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
    task: Optional[dict] = None
    client_msg_id = ""
    if task_metadata:
        _cmid_ref = task_metadata.get("origin_message_ref")
        if isinstance(_cmid_ref, dict):
            client_msg_id = str(_cmid_ref.get("client_message_id") or "")
        if not client_msg_id:
            client_msg_id = str(task_metadata.get("client_message_id") or "")
    kind = "ephemeral_decision" if ephemeral else "direct_chat"
    task: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "type": "task",
        "chat_id": chat_id,
        "text": text,
        "_is_direct_chat": True,
    }
    try:
        from ouroboros.contracts.task_contract import attach_task_contract

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
            from ouroboros.artifacts import (
                attachment_manifest_all_rejected,
                attachment_manifest_has_rejections,
                stage_task_attachments,
            )
            from ouroboros.gateway.tasks import _render_attachment_lines

            manifest = stage_task_attachments(DRIVE_ROOT, str(task["id"]), uploads)
            rendered = _render_attachment_lines(manifest)
            # Partial staging is the default (В25c, capinv-447); a FULLY-rejected
            # set stays atomic — the task would start with none of its material.
            if attachment_manifest_all_rejected(manifest):
                from ouroboros.artifacts import remove_staged_attachments

                remove_staged_attachments(manifest)
                send_with_budget(
                    chat_id,
                    f"⚠️ Task not started: every attachment was rejected.\n{rendered}",
                )
                return
            if attachment_manifest_has_rejections(manifest):
                send_with_budget(
                    chat_id,
                    "⚠️ Some declared attachments could not be staged; the task "
                    f"starts with the rest.\n{rendered}",
                )
            if manifest:
                manifest = [dict(row) for row in manifest]
                task["drive_root"] = str(DRIVE_ROOT)
                task["attachments"] = manifest
                task["attachment_images"] = [
                    m for m in manifest
                    if str(m.get("status") or "staged") == "staged" and m.get("is_image")
                ]
                if rendered:
                    task["text"] = f"{task.get('text') or ''}\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]"
                task.pop("image_base64", None)
                task.pop("image_mime", None)
        # A rejected initial UI task must leave no partial project assignment.
        # Bind only after all declared attachments have passed admission.
        pid = str(task.get("project_id") or "").strip()
        if pid and not ephemeral:
            try:
                from ouroboros.projects_registry import bind_task_to_project

                bind_task_to_project(
                    DRIVE_ROOT, task["id"], pid, chat_id,
                    origin=_origin_from_mapping(
                        task_metadata or {}, absent="mid_task_no_origin",
                    ),
                )
            except Exception as exc:
                _report_binding_failure(task["id"], pid, exc, path="direct_project_turn")
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

        pid = str(task.get("project_id") or "")

        from supervisor.active_activity import track_direct_activity

        with track_direct_activity(
            activity_id=str(task["id"]),
            chat_id=int(chat_id or 0),
            client_message_id=client_msg_id,
            project_id=pid,
            kind=kind,
            phase="thinking",
        ):
            # Announce the authoritative start immediately (owner decision 2A):
            # the client's `Sending...` retires on this frame, not on a socket
            # echo, and the frame carries the activity<->client_message_id link
            # so even a turn that fails before its first LLM round concludes
            # cleanly via its keyed error final.
            try:
                from supervisor.message_bus import get_bridge

                get_bridge().send_chat_action(
                    int(chat_id or 0),
                    "typing",
                    activity_id=str(task["id"]),
                    client_message_id=client_msg_id,
                    phase="thinking",
                    kind=kind,
                )
            except Exception:
                log.debug("Direct-turn start typing announce failed", exc_info=True)
            # The turn's live emits (loop_llm_call and friends publish
            # straight to the agent's event queue DURING handle_task) and its
            # returned events are both drained after the registry entry is
            # gone: route them through the turn-scoped addressing proxy.
            turn_queue = _TurnEventQueue(get_event_q(), task["id"], chat_id)
            prev_queue = getattr(agent, "_event_queue", None)
            agent._event_queue = turn_queue
            try:
                events = agent.handle_task(task)
            finally:
                agent._event_queue = prev_queue
            for e in events:
                get_event_q().put(turn_queue.stamp(e))
    except Exception as e:
        import traceback
        err_msg = f"⚠️ Error: {type(e).__name__}: {e}"
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "direct_chat_error",
                "task_id": str(task.get("id") or ""),
                "chat_id": int(chat_id or 0),
                "error": repr(e),
                "traceback": str(traceback.format_exc())[:2000],
            },
        )
        try:
            # Key the error final with the turn's activity id so the client
            # concludes exactly this turn (active set, 4A) instead of leaving
            # its `Sending.../Thinking...` state to an unkeyed sweep. If the
            # failure happened before the start announce was broadcast, the
            # client has no activity<->client_message_id link yet, so announce
            # it first: the keyed final right after then retires both the
            # activity and its linked `Sending...` submission.
            failed_task_id = str(task.get("id") or "") if isinstance(task, dict) else ""
            if failed_task_id and client_msg_id:
                try:
                    from supervisor.message_bus import get_bridge

                    get_bridge().send_chat_action(
                        int(chat_id or 0),
                        "typing",
                        activity_id=failed_task_id,
                        client_message_id=client_msg_id,
                        phase="thinking",
                        kind=kind,
                    )
                except Exception:
                    log.debug("Failed-turn typing announce failed", exc_info=True)
            send_with_budget(
                chat_id,
                err_msg,
                task_id=failed_task_id,
                progress_meta={"task_terminal_status": "failed"},
            )
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
        from ouroboros.usage_ledger import DISPLAY_LOCK_TIMEOUT_SEC

        remaining = budget_remaining(
            load_state(), strict=True,
            lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC,
            allow_stale=True,
        )
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
# The second group publishes the SAME type twice at the producer (a durable
# append plus an emit_log_event live sibling of the identical type): the live
# copy is kept and the forwarded append copy is dropped.
WORKER_LOG_SINK_SUPPRESSED_TYPES = frozenset({
    "tool_call", "llm_round", "task_checkpoint", "task_done", "llm_usage",
    "provider_incomplete_response", "llm_empty_response", "provider_body_error",
    "review_cycles_exhausted", "plan_review_advisory_open",
})

# The server process runs the same suppression discipline over its raw
# append_jsonl->sink broadcasts (server.py installs the wrapper): every type
# here has a dedicated ctx.bridge.push_log at its supervisor handler, so the
# sink copy would be the second delivery of the same event. This is the
# exactly-once contract test_log_forwarding pins with the production sink
# installed. The set is a superset of the worker list because the direct-chat
# agent and Background Consciousness run inside the server process and append
# the same worker-shaped rows there.
SERVER_LOG_SINK_SUPPRESSED_TYPES = WORKER_LOG_SINK_SUPPRESSED_TYPES | frozenset({
    "budget_scope_paused", "task_metrics_event", "review_late_result",
    "task_cost_finalized", "skill_exec_finished", "skill_exec_failed",
    "task_cancel_cascade_noop", "task_cancel_cascade_error",
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


def _demote_inherited_rotating_log_handlers() -> None:
    """Only the supervisor rotates ``server.log``; workers follow the live file.

    Under fork every worker inherits the server's ``RotatingFileHandler`` and
    rotates the SAME file on its own byte count. With 64 workers the renames
    race each other and the log degenerates into a handful of tiny files (r8,
    2026-09-04: ``server.log`` 2 KB, ``.1`` 198 B, ``.2`` 739 B — hours of
    forensics gone). A ``WatchedFileHandler`` on the same path writes to
    whatever the supervisor's rotation currently calls ``server.log``.
    """
    import logging as _logging
    from logging.handlers import RotatingFileHandler, WatchedFileHandler

    root = _logging.getLogger()
    for handler in list(root.handlers):
        if not isinstance(handler, RotatingFileHandler):
            continue
        try:
            replacement = WatchedFileHandler(
                handler.baseFilename, encoding=getattr(handler, "encoding", None) or "utf-8",
            )
            replacement.setFormatter(handler.formatter)
            replacement.setLevel(handler.level)
            for log_filter in list(handler.filters):
                replacement.addFilter(log_filter)
            root.removeHandler(handler)
            root.addHandler(replacement)
            handler.close()  # this process's fd copy only; the parent's is untouched
        except Exception:
            pass


def worker_main(wid: int, in_q: Any, out_q: Any, repo_dir: str, drive_root: str,
                custody_session_id: str = "") -> None:
    import os as _os
    # Mark this process as a worker BEFORE importing the agent/LLM stack so the
    # central network-transport policy disables system proxy resolution
    # (trust_env=False) for every HTTP client created here. This is the
    # fork-safety guard (no _scproxy/SCDynamicStoreCopyProxies on the child side
    # of fork) and a clean default for spawned workers too.
    _os.environ["OUROBOROS_IN_WORKER"] = "1"
    _demote_inherited_rotating_log_handlers()
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
    # existing EVENT_Q -> _handle_log_event -> push_log path, suppressing
    # WORKER_LOG_SINK_SUPPRESSED_TYPES (see the constant's comment for the
    # exactly-once rationale per group).
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
            return


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
        existing = load_task_result(DRIVE_ROOT, task_id, strict=True)
        if existing and existing.get("status") in _FINAL_STATUSES:
            return str(existing.get("status") or "")
        final_status = status or STATUS_FAILED
        # Reconstruct from durable llm_usage so an abnormally-finalized task does
        # not record zero cost/rounds (understating per-task + campaign metrics).
        f_cost_fields = reconstruct_task_cost(str(task_id), fields=True)
        stored = write_task_result(
            DRIVE_ROOT,
            task_id,
            final_status,
            strict_existing_dict=True,
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
        persisted_status = str((stored or {}).get("status") or "").strip()
        if (
            not isinstance(stored, dict)
            or str(stored.get("task_id") or "") != str(task_id)
            or not persisted_status
        ):
            raise ValueError(
                f"failure result writer returned invalid durable identity for {task_id}"
            )
        return persisted_status
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


_LEDGERED_WORKER_PIDS: set[tuple[int, int]] = set()


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
    # worker_pids.json stays as the legacy session-leader reap path.  Each
    # (slot, pid) is ledgered ONCE per server generation: the row survives
    # every ledger rewrite while the pid is alive and same-session, and
    # ``record_process`` costs two ``ps`` subprocesses per pid — re-recording
    # all 64 slots on every respawn held the cancel/reap path ~40 s on a busy
    # host (Tier-2 load repro, 14k host processes) and appended 64 duplicate
    # rows per respawn.
    try:
        from ouroboros.process_custody import record_process

        for w in WORKERS.values():
            key = (int(w.wid), int(w.proc.pid or 0))
            if not key[1] or key in _LEDGERED_WORKER_PIDS:
                continue
            record_process(
                DRIVE_ROOT,
                pid=key[1],
                cmd=f"ouroboros-worker-{w.wid}",
                purpose=f"worker:{w.wid}",
                scope="session",
            )
            _LEDGERED_WORKER_PIDS.add(key)
    except Exception:
        log.debug("Failed to ledger worker pids", exc_info=True)


def _reconcile_confirmed_dead_review_owner(owner_pid: int) -> None:
    _reconcile_confirmed_dead_review_owner_for_root(DRIVE_ROOT, owner_pid)


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


@_serialized_worker_lifecycle
def kill_workers_for_update(*, result_reason: str, terminal_status: str = "interrupted") -> List[str]:
    """Stop the current pool and return anything whose death could not be proven."""
    from ouroboros.platform_layer import kill_pid_tree

    with _queue_lock:
        fenced = list(WORKERS.values())
    teardown_error = ""
    try:
        kill_ok = kill_workers(
            result_reason=result_reason,
            terminal_status=terminal_status,
            disable_reason="managed_update",
            preserve_pending=True,
        )
        if kill_ok is False:
            teardown_error = "teardown:queue_snapshot_persist_failed"
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
            else:
                _reconcile_confirmed_dead_review_owner(
                    int(getattr(worker.proc, "pid", 0) or 0)
                )
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
            "cost_usd": None,
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
                "cost_usd": None,
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


def assign_tasks() -> None:
    from supervisor import queue
    from supervisor.state import budget_remaining, EVOLUTION_BUDGET_RESERVE
    with _queue_lock:
        st = load_state()
        # Cancellation/terminal custody wins before validating rows left in the
        # queue.  Then quarantine every malformed depth before budget, lease, or
        # capacity filters can leave it waiting indefinitely.
        if not _drop_cancelled_pending():
            log.error(
                "Task assignment blocked: cancellation authority or custody "
                "state is indeterminate",
            )
            queue.persist_queue_snapshot(reason="cancellation_authority_indeterminate")
            return
        _retry_terminalization_pending_for_assignment(queue)
        invalid_ids, unresolved_invalid_ids = _quarantine_invalid_pending_depths()
        unresolved_invalid_id_set = set(unresolved_invalid_ids)

        if invalid_ids:
            queue.persist_queue_snapshot(reason="invalid_task_depth")
        if unresolved_invalid_ids:
            log.error(
                "Invalid-depth rows deferred until terminal custody is available; continuing assignment for other tasks: %s",
                ", ".join(unresolved_invalid_ids),
            )
        try:
            from ouroboros.usage_ledger import DISPLAY_LOCK_TIMEOUT_SEC

            # The loop ticks this gate constantly; reserve_attempt stays the
            # exact monetary authority, so this pre-check rides the last
            # validated snapshot under ledger-write contention rather than
            # stalling assignment behind the 45s monetary lock.
            remaining = budget_remaining(
                st, strict=True,
                lock_timeout_sec=DISPLAY_LOCK_TIMEOUT_SEC,
                allow_stale=True,
            )
        except Exception:
            log.error("Task assignment blocked: monetary authority unavailable")
            return
        if remaining <= 0:
            planned = []
            for task in PENDING:
                if _invalid_depth_deferred(task, unresolved_invalid_id_set):
                    continue
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

        # Evolution is hard-blocked in light runtime mode at the assignment
        # chokepoint too: a task restored from a snapshot or created before the
        # mode switch must never actually run. Cancel them terminally.
        from supervisor.evolution_lifecycle import evolution_block_reason
        evo_block = evolution_block_reason()
        blocked_ids = _drop_assignable_evolution_tasks(unresolved_invalid_id_set) if evo_block else []
        if blocked_ids:
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

        from ouroboros.project_lease import candidate_is_leasable, running_project_ids
        from ouroboros.config import get_max_active_subagents_per_root


        for w in WORKERS.values():
            if w.busy_task_id is None and not getattr(w, "reaping", False) and PENDING:
                # One-writer-per-project lease: recompute per assignment so a
                # task assigned in THIS loop pass immediately occupies its lane.
                leased = running_project_ids(RUNNING.values())
                # Find first suitable task (skip over-budget evolution tasks
                # and project-leased candidates)
                chosen_idx = None
                for i, candidate in enumerate(PENDING):
                    if _invalid_depth_deferred(candidate, unresolved_invalid_id_set):
                        continue
                    if not repo_writer_task_allowed(candidate):
                        continue
                    if isinstance(candidate.get("_budget_pause"), dict):
                        continue
                    root_task_id = str(candidate.get("root_task_id") or "").strip()
                    if root_task_id in queue.BUDGET_ROOT_FENCES:
                        continue
                    if str(candidate.get("type") or "") == "evolution" and remaining < EVOLUTION_BUDGET_RESERVE:
                        continue
                    if not candidate_is_leasable(candidate, leased):
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
                    # Project-leased rows wait; over-budget evolution rows are cleaned.
                    if remaining < EVOLUTION_BUDGET_RESERVE:
                        dropped_ids = _drop_assignable_evolution_tasks(unresolved_invalid_id_set)
                        if dropped_ids:
                            queue.persist_queue_snapshot(reason="evolution_dropped_budget")
                    continue
                task = PENDING.pop(chosen_idx)
                depth_error = _normalize_pending_task_depth(task)
                if depth_error:
                    if _terminalize_invalid_pending_depth(task, depth_error):
                        queue.persist_queue_snapshot(reason="invalid_task_depth")
                        continue
                    # Keep failed terminalization in queue custody for retry.
                    PENDING.insert(chosen_idx, task)
                    log.error(
                        "Assignment blocked: invalid task depth could not be terminalized for %s",
                        task.get("id"),
                    )
                    break
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
                        from ouroboros.tools.control_delegation import stamp_task_assignment_depth
                        from ouroboros.config import get_max_subagent_depth

                        # Assignment is the first host-visible execution fact. Stamp
                        # the worker payload and canonical result from one projection.
                        _depth_fields = stamp_task_assignment_depth(
                            task, max_depth=get_max_subagent_depth(),
                        )
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
                            **_depth_fields,
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
                            configured_subagent=task.get("configured_subagent"),
                            parent_cognitive_route=task.get("parent_cognitive_route"),
                            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
                            result="Subagent assigned to a worker.",
                        )
                    except Exception:
                        log.debug("Failed to mirror running subagent status", exc_info=True)
                w.busy_task_id = task["id"]
                w.in_q.put(task)
                now_ts = time.time()
                RUNNING[task["id"]] = {
                    "task": dict(task), "worker_id": w.wid,
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


def _worker_rss_mb(pid: int) -> Optional[int]:
    """Resident set size of a worker process in MiB from /proc, or None."""
    try:
        with open(f"/proc/{int(pid)}/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, TypeError):
        return None
    return None


def _worker_rss_limit_mb() -> int:
    """Per-worker RSS ceiling. A single task's context + tool outputs stays in
    single-digit GiB; a runaway (a model that keeps appending huge tool output,
    or a pathological workspace load) can balloon a worker to hundreds of GiB
    and trigger a GLOBAL OOM that kills the isolate and every other lane at
    once (CyberGym r11, 2026-09-04: one python3 worker reached 373 GiB RSS ->
    host OOM -> launcher torn down -> 1152 in-flight tasks written off). The
    watchdog kills the single offender (its task ends as an infra crash, no
    retry) long before the host is threatened. 0/invalid disables it."""
    raw = str(os.environ.get("OUROBOROS_WORKER_RSS_LIMIT_MB") or "").strip()
    if not raw:
        return 24000  # unset -> default ceiling
    try:
        return max(0, int(raw))  # explicit 0 disables the watchdog
    except ValueError:
        return 24000


def _ensure_workers_healthy_locked(queue: Any) -> tuple[List[int], bool]:
    busy_crashes = 0
    dead_detections = 0
    crashed_tasks = []
    respawn_ids: List[int] = []
    _rss_limit_mb = _worker_rss_limit_mb()
    for wid, w in list(WORKERS.items()):
        # The reaper owns marked slots through replacement; never double-respawn them.
        if getattr(w, "reaping", False):
            continue
        # Memory watchdog: SIGKILL a single runaway worker before its balloon
        # OOMs the whole host. The kill surfaces next tick as a signal death,
        # which the crash path below finalizes as a terminal infra failure (no
        # retry — a retry would balloon again). Only busy workers are checked;
        # an idle worker at rest is never a runaway.
        if (
            _rss_limit_mb > 0
            and w.busy_task_id is not None
            and w.proc.is_alive()
            and (w.proc.pid or 0)
        ):
            _rss = _worker_rss_mb(int(w.proc.pid))
            if _rss is not None and _rss > _rss_limit_mb:
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "worker_memory_exceeded",
                        "worker_id": wid,
                        "pid": int(w.proc.pid),
                        "rss_mb": _rss,
                        "limit_mb": _rss_limit_mb,
                        "busy_task_id": w.busy_task_id,
                    },
                )
                log.error(
                    "Worker %d (task %s) RSS %d MiB exceeds %d MiB — killing to protect the host",
                    wid, w.busy_task_id, _rss, _rss_limit_mb,
                )
                try:
                    w.proc.kill()
                except Exception:
                    log.debug("Failed to kill over-memory worker %d", wid, exc_info=True)
                continue
        if not w.proc.is_alive():
            # Reserve the dead slot before the queue lock is released.
            w.reaping = True
            _reconcile_confirmed_dead_review_owner(
                int(getattr(w.proc, "pid", 0) or 0)
            )
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
                    # Signal crashes are terminal infrastructure failures for every task type.
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
                        from ouroboros.delegate_recovery import reconcile_unrecoverable_task
                        reconcile_unrecoverable_task(DRIVE_ROOT, str(w.busy_task_id))
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
                        from ouroboros.delegate_recovery import reconcile_unrecoverable_task
                        reconcile_unrecoverable_task(DRIVE_ROOT, str(w.busy_task_id))
                    else:
                        task = dict(task)
                        task["_attempt"] = attempt + 1
                        from ouroboros.delegate_recovery import prepare_worker_crash_handoff
                        recovery_handoff = prepare_worker_crash_handoff(
                            DRIVE_ROOT, task, old_attempt=attempt, new_attempt=attempt + 1,
                            worker_id=wid,
                            exitcode=exitcode if isinstance(exitcode, int) else None,
                        )
                        try:
                            from ouroboros.task_results import STATUS_INTERRUPTED, write_task_result
                            write_task_result(
                                DRIVE_ROOT, str(w.busy_task_id), STATUS_INTERRUPTED,
                                result=f"Worker process died mid-task (attempt {attempt}). Retrying.",
                                **r_cost_fields,
                            )
                        except Exception:
                            log.debug("Failed to write interrupted status for %s", w.busy_task_id, exc_info=True)
                        try:
                            from ouroboros.owner_hurry import retry_reset

                            retry_reset(
                                queue._task_drive_for_task(task, str(w.busy_task_id)),
                                DRIVE_ROOT, str(w.busy_task_id),
                                reason="worker_crash_requeue",
                            )
                        except Exception:
                            log.debug("Crash-requeue retry reset failed for %s", w.busy_task_id, exc_info=True)
                        admitted = queue.enqueue_task(task, front=True)
                        admission_block = (
                            str(admitted.get("_admission_blocked") or "")
                            if isinstance(admitted, dict) else ""
                        )
                        if admission_block:
                            from ouroboros.delegate_recovery import veto_worker_retry_handoff
                            veto_worker_retry_handoff(
                                DRIVE_ROOT, str(w.busy_task_id), recovery_handoff, admission_block,
                            )
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

    disable_pool = _worker_crash_storm_detected(
        busy_crashes=busy_crashes,
        dead_detections=dead_detections,
        crashed_tasks=crashed_tasks,
    )
    return respawn_ids, disable_pool
