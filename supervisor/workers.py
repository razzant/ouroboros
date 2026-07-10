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
from supervisor.message_bus import send_with_budget
from ouroboros.outcomes import EXECUTION_FAILED, EXECUTION_INFRA_FAILED, terminal_outcome_axes
from ouroboros.utils import utc_now_iso


REPO_DIR: pathlib.Path = pathlib.Path.home() / "Ouroboros" / "repo"
DRIVE_ROOT: pathlib.Path = pathlib.Path.home() / "Ouroboros" / "data"
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


def get_event_q():
    """Return EVENT_Q, creating it lazily."""
    global _EVENT_Q
    if _EVENT_Q is None:
        _EVENT_Q = _get_ctx().Queue()
    return _EVENT_Q


WORKERS: Dict[int, Worker] = {}
PENDING: List[Dict[str, Any]] = []
RUNNING: Dict[str, Dict[str, Any]] = {}
CRASH_TS: List[float] = []
QUEUE_SEQ_COUNTER_REF: Dict[str, int] = {"value": 0}

# Shared queue lock; queue.py owns the canonical definition.
from supervisor.queue import _queue_lock


_chat_agent = None
# Serializes every direct-chat caller; _chat_agent has mutable per-call state.
import threading as _threading
_chat_agent_lock = _threading.Lock()


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


def promote_chat_to_task(evt: dict, ctx: Any) -> None:
    """Enqueue a first-class pooled owner task from a conversation-lane promote.

    The task carries the originating ``chat_id`` (its live card and replies
    land in that thread) and the optional ``project_id`` scope; it competes for
    the project writer lease like any other top-level project task.
    """
    from ouroboros.contracts.task_contract import attach_task_contract

    tid = str(evt.get("task_id") or uuid.uuid4().hex[:8])
    objective = str(evt.get("objective") or "").strip()
    if not objective:
        return
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
    }
    pid = str(evt.get("project_id") or "").strip()
    if pid:
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
                bind_task_to_project(DRIVE_ROOT, tid, pid, (project or {}).get("chat_id"))
            except Exception:
                log.debug("promote: bind_task_to_project failed for %s/%s", tid, pid, exc_info=True)
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
            log.debug("promote: project registration failed for %s", pid, exc_info=True)
    # v6.58.0 (slice 1) — the promote path admits a workspace through the SAME SSOT
    # as /api/tasks. A task born in a project ROOM defaults to the room's registered
    # working_dir (workspace="none" on the event opts out); a SET-but-broken
    # working_dir fails LOUDLY here — never a silent workspace-less task that would
    # resolve to the self_modification profile over the system repo.
    from ouroboros.workspace_admission import bounded_workspace_preflight, compose_workspace_block, resolve_room_workspace

    resolved_ws, ws_error = resolve_room_workspace(
        drive_root=DRIVE_ROOT,
        system_repo_dir=REPO_DIR,
        project_id=pid,
        explicit_workspace=str(evt.get("workspace_root") or "").strip(),
        workspace_sentinel=str(evt.get("workspace") or ""),
    )
    if ws_error:
        _fail_promoted_task_loudly(ctx, task, ws_error)
        return
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
    attach_task_contract(task)
    ctx.enqueue_task(task)
    # v6.40 "name ANY task card": the agent already coined `title` here (zero extra LLM
    # call), so persist it as suggested_name + emit task_named so the promoted card shows
    # the human title up front exactly like a proactively-named direct-chat card, and a
    # later turn-into-project reuses it. Same-status (SCHEDULED) write — merges, never
    # regresses; fail-soft.
    if title:
        try:
            from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

            write_task_result(DRIVE_ROOT, tid, STATUS_SCHEDULED, suggested_name=title)
            _broadcast_task_named({"type": "task_named", "task_id": tid, "suggested_name": title})
        except Exception:
            log.debug("promote: suggested_name persist/broadcast failed for %s", tid, exc_info=True)


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
        try:
            bind_task_to_project(DRIVE_ROOT, tid, pid, proj_chat or None)
        except Exception:
            log.debug("ensure_project_scope: bind failed for %s/%s", tid, pid, exc_info=True)
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
    if budget_remaining(load_state()) <= 0:
        try:
            from supervisor.message_bus import get_bridge
            get_bridge().send_message(chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.")
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
                        bind_task_to_project(DRIVE_ROOT, task["id"], pid, chat_id)
                    except Exception:
                        log.debug("bind_task_to_project failed for direct project task %s/%s", task["id"], pid, exc_info=True)
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
            from supervisor.message_bus import get_bridge
            get_bridge().send_message(chat_id, err_msg)
        except Exception:
            log.debug("Suppressed exception", exc_info=True)


# Serializes ephemeral same-route turns so concurrent main-chat messages each get
# their own response in order, without ever touching the running locked turn.
_ephemeral_chat_lock = _threading.Lock()


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
    if budget_remaining(load_state()) <= 0:
        try:
            from supervisor.message_bus import get_bridge
            get_bridge().send_message(chat_id, "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.")
        except Exception:
            pass
        return
    if not getattr(sys, 'frozen', False):
        sys.path.insert(0, str(REPO_DIR))
    from ouroboros.agent import make_agent

    with _ephemeral_chat_lock:
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


def worker_main(wid: int, in_q: Any, out_q: Any, repo_dir: str, drive_root: str,
                custody_session_id: str = "") -> None:
    import os as _os
    # Mark this process as a worker BEFORE importing the agent/LLM stack so the
    # central network-transport policy disables system proxy resolution
    # (trust_env=False) for every HTTP client created here. This is the
    # fork-safety guard (no _scproxy/SCDynamicStoreCopyProxies on the child side
    # of fork) and a clean default for spawned workers too.
    _os.environ["OUROBOROS_IN_WORKER"] = "1"
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
    while True:
        try:
            task = in_q.get()
            if task is None or task.get("type") == "shutdown":
                break
            task_drive_root = str(task.get("drive_root") or drive_root)
            if task_drive_root != str(drive_root):
                task_agent = make_agent(repo_dir=repo_dir, drive_root=task_drive_root, event_queue=out_q)
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
        f_cost, f_rounds, f_prompt, f_completion = reconstruct_task_cost(str(task_id))
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
            cost_usd=f_cost,
            total_rounds=f_rounds,
            prompt_tokens=f_prompt,
            completion_tokens=f_completion,
        )
        return final_status
    except Exception:
        log.warning("Failed to write failure result for task %s", task_id, exc_info=True)
        return status or "failed"


def _emit_task_done_terminal(
    task: Optional[Dict[str, Any]],
    task_id: str,
    status: str = "failed",
    *,
    reason_code: str = "",
    cost_usd: float = 0.0,
    total_rounds: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Emit a task_done event so the UI resolves the live card when a task is
    torn down outside the normal completion path (crash storm, kill, hard
    timeout). Without this the spinner spins forever on these paths.

    Cost fields carry reconstructed totals so an evolution campaign tally fed
    from this terminal event records real spend instead of zeros; callers that
    have no reconstructed cost leave them at 0."""
    if not task_id:
        return
    try:
        chat_id = int((task or {}).get("chat_id") or 0)
    except (TypeError, ValueError):
        chat_id = 0
    if not chat_id:
        return
    status = status or "failed"
    # Caller reason_code wins; budget_exhausted -> EXECUTION_FAILED below, not infra-failure.
    reason_code = reason_code or ("worker_terminal_failure" if status == "failed" else status)
    try:
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
            "cost_usd": cost_usd,
            "total_rounds": total_rounds,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })
    except Exception:
        log.debug("Failed to emit terminal task_done for %s", task_id, exc_info=True)


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


def _first_worker_boot_event_since(offset_bytes: int) -> Optional[Dict[str, Any]]:
    """Read first worker_boot event after a file offset."""
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
        if isinstance(evt, dict) and str(evt.get("type") or "") == "worker_boot":
            return evt
    return None


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


def spawn_workers(n: int = 0) -> None:
    global _CTX, _EVENT_Q
    # Reap any workers left orphaned by a prior/abrupt server exit before we
    # spawn fresh ones, so process groups do not accumulate across restarts.
    reap_orphaned_workers()
    # Fresh context ensures workers use current code.
    _CTX = mp.get_context(_WORKER_START_METHOD)
    _EVENT_Q = _CTX.Queue()
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
        },
    )
    WORKERS.clear()
    for i in range(count):
        in_q = _CTX.Queue()
        proc = _CTX.Process(target=worker_main,
                           args=(i, in_q, _EVENT_Q, str(REPO_DIR), str(DRIVE_ROOT),
                                 _current_custody_session_id()))
        proc.daemon = True
        proc.start()
        WORKERS[i] = Worker(wid=i, proc=proc, in_q=in_q, busy_task_id=None)
    global _LAST_SPAWN_TIME
    _LAST_SPAWN_TIME = time.time()
    _record_worker_pids()
    # Verify asynchronously so spawn does not block the supervisor loop.
    threading.Thread(target=_verify_worker_sha_after_spawn, args=(events_offset,), daemon=True).start()


def kill_workers(
    force: bool = True,
    *,
    result_reason: str = "Worker process crashed (crash storm). Task was not completed.",
    terminal_status: str = "",
    archive_service_logs: bool = True,
) -> None:
    from supervisor import queue
    with _queue_lock:
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
        try:
            done_status = terminal_status or "failed"
            orphaned_ids = []
            for task_id in list(RUNNING):
                try:
                    meta = RUNNING.get(task_id) or {}
                    task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else {}
                    persisted = _write_failure_result(task_id, reason=result_reason, status=terminal_status)
                    if archive_service_logs:
                        try:
                            from ouroboros.tools.services import archive_task_service_logs
                            archive_task_service_logs(pathlib.Path(DRIVE_ROOT), str(task_id), task)
                        except Exception:
                            log.debug("Failed to archive service logs for task %s", task_id, exc_info=True)
                    _emit_task_done_terminal(task, str(task_id), persisted or done_status)
                    orphaned_ids.append(task_id)
                except Exception:
                    log.warning("Failed to write failure result for running task %s", task_id, exc_info=True)
            drained = queue.drain_all_pending()
            drained_ids = []
            for task in drained:
                tid = task.get("id")
                if tid:
                    try:
                        persisted = _write_failure_result(tid, reason=result_reason, status=terminal_status)
                        _emit_task_done_terminal(task, str(tid), persisted or done_status)
                        drained_ids.append(tid)
                    except Exception:
                        log.warning("Failed to write failure result for pending task %s", tid, exc_info=True)
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
        RUNNING.clear()
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


def respawn_worker(wid: int) -> None:
    ctx = _get_ctx()
    in_q = ctx.Queue()
    proc = ctx.Process(target=worker_main,
                       args=(wid, in_q, get_event_q(), str(REPO_DIR), str(DRIVE_ROOT),
                             _current_custody_session_id()))
    proc.daemon = True
    proc.start()
    # Swap under _queue_lock (an RLock — safe even when the caller already holds
    # it) so a concurrent assign_tasks cannot enqueue into the slot mid-swap.
    with _queue_lock:
        old = WORKERS.get(wid)
        WORKERS[wid] = Worker(wid=wid, proc=proc, in_q=in_q, busy_task_id=None)
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


def assign_tasks() -> None:
    from supervisor import queue
    from supervisor.state import budget_remaining, EVOLUTION_BUDGET_RESERVE
    with _queue_lock:
        st = load_state()
        remaining = budget_remaining(st)
        if remaining <= 0:
            # Budget exhausted: PENDING has no timeout backstop. Terminally fail stranded
            # tasks with an OBSERVABLE budget_exhausted result + task_done so waiters resolve.
            _stranded, PENDING[:] = list(PENDING), []
            if _stranded:
                from ouroboros.task_results import fail_tasks
                _bmsg = "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings."
                fail_tasks(DRIVE_ROOT, _stranded, reason_code="budget_exhausted", result=_bmsg)
                for _t in _stranded:  # resolve the live UI/SSE card (no-ops without a chat_id)
                    _emit_task_done_terminal(_t, str(_t.get("id") or ""), reason_code="budget_exhausted")
                if st.get("owner_chat_id"):
                    send_with_budget(int(st["owner_chat_id"]), _bmsg)
                queue.persist_queue_snapshot(reason="budget_exhausted")
            return  # Stop assigning ALL tasks if budget is completely exhausted

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

        from ouroboros.project_lease import candidate_is_leasable, running_project_ids
        from ouroboros.config import get_max_active_subagents_per_root

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
                # One-writer-per-project lease: recompute per assignment so a
                # task assigned in THIS loop pass immediately occupies its lane.
                leased = running_project_ids(RUNNING.values())
                # Find first suitable task (skip over-budget evolution tasks
                # and project-leased candidates)
                chosen_idx = None
                for i, candidate in enumerate(PENDING):
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
                            model_lane=task.get("model_lane"),
                            requested_model_lane=task.get("requested_model_lane"),
                            effective_model_lane=task.get("effective_model_lane"),
                            model=task.get("model"),
                            use_local_model=task.get("use_local_model"),
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
        _ensure_workers_healthy_locked(queue)


def _ensure_workers_healthy_locked(queue: Any) -> None:
    busy_crashes = 0
    dead_detections = 0
    crashed_tasks = []
    for wid, w in list(WORKERS.items()):
        # Variant A: a slot marked `reaping` is owned end-to-end by the background reaper
        # (kill -> join -> archive -> respawn). Its proc is expected to die mid-reap, so the
        # crash detector must NOT also respawn it — that double-respawn would orphan a live
        # worker process. The reaper installs a fresh Worker (reaping=False) when done.
        if getattr(w, "reaping", False):
            continue
        if not w.proc.is_alive():
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
                    chat_id = int(task.get("chat_id") or 1)
                    attempt = int(task.get("_attempt") or 1)
                    # Reconstruct cost/rounds from durable llm_usage for any
                    # abnormal-termination rollup below (worker died pre-finalize,
                    # so the event would otherwise carry zeros).
                    r_cost, r_rounds, r_prompt, r_completion = reconstruct_task_cost(str(w.busy_task_id))

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
                                cost_usd=r_cost,
                                total_rounds=r_rounds,
                                prompt_tokens=r_prompt,
                                completion_tokens=r_completion,
                            )
                        except Exception:
                            log.debug("Failed to write failed status for %s", w.busy_task_id, exc_info=True)
                        # Message before task_done: otherwise the UI may close the card first.
                        try:
                            from supervisor.message_bus import get_bridge
                            bridge = get_bridge()
                            if bridge is not None:
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
                                bridge.send_message(chat_id, user_msg)
                        except Exception:
                            log.debug("Failed to send failure message for %s", w.busy_task_id, exc_info=True)
                        try:
                            get_event_q().put({
                                "type": "task_done",
                                "task_id": str(w.busy_task_id),
                                "task_type": task_type,
                                "chat_id": chat_id,
                                "status": "failed",
                                "reason_code": reason_code,
                                "outcome_axes": terminal_outcome_axes(lifecycle="failed", execution=EXECUTION_INFRA_FAILED, reason_code=reason_code, review_trigger="worker_terminal"),
                                "cost_usd": r_cost,
                                "total_rounds": r_rounds,
                                "prompt_tokens": r_prompt,
                                "completion_tokens": r_completion,
                            })
                        except Exception:
                            log.debug("Failed to emit terminal event for %s", w.busy_task_id, exc_info=True)
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
                                cost_usd=r_cost,
                                total_rounds=r_rounds,
                                prompt_tokens=r_prompt,
                                completion_tokens=r_completion,
                            )
                        except Exception:
                            log.debug("Failed to write cancelled status for %s", w.busy_task_id, exc_info=True)
                        _emit_task_done_terminal(
                            task, str(w.busy_task_id), "cancelled",
                            cost_usd=r_cost, total_rounds=r_rounds,
                            prompt_tokens=r_prompt, completion_tokens=r_completion,
                        )
                    else:
                        task = dict(task)
                        task["_attempt"] = attempt + 1
                        try:
                            from ouroboros.task_results import STATUS_INTERRUPTED, write_task_result
                            write_task_result(
                                DRIVE_ROOT, str(w.busy_task_id), STATUS_INTERRUPTED,
                                result=f"Worker process died mid-task (attempt {attempt}). Retrying.",
                                cost_usd=r_cost,
                                total_rounds=r_rounds,
                                prompt_tokens=r_prompt,
                                completion_tokens=r_completion,
                            )
                        except Exception:
                            log.debug("Failed to write interrupted status for %s", w.busy_task_id, exc_info=True)
                        queue.enqueue_task(task, front=True)
            respawn_worker(wid)
            queue.persist_queue_snapshot(reason="worker_respawn_after_crash")

    now = time.time()
    alive_now = sum(1 for w in WORKERS.values() if w.proc.is_alive())
    if dead_detections:
        # Only count busy crashes or all-workers-dead as storm signals.
        if busy_crashes > 0 or alive_now == 0:
            CRASH_TS.extend([now] * max(1, dead_detections))
        else:
            CRASH_TS.clear()

    CRASH_TS[:] = [t for t in CRASH_TS if (now - t) < 60.0]
    if len(CRASH_TS) >= 3:
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
            )
        kill_workers()
        CRASH_TS.clear()
