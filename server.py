"""Self-editable Starlette/uvicorn entry point for UI and supervisor runtime."""

import asyncio
import base64
import json
import logging
import socket
import subprocess

import os
import pathlib
import sys
import threading
import time
import uuid
from ouroboros.utils import read_json_dict, utc_now_iso
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.routing import Route, Mount

import uvicorn

from ouroboros.server_control import (
    execute_panic_stop as _execute_panic_stop_impl,
    restart_current_process as _restart_current_process_impl,
)
from ouroboros.server_auth import (
    NetworkAuthGate,
    get_network_auth_startup_warning,
    validate_network_auth_configuration,
)
from ouroboros.server_entrypoint import find_free_port, parse_server_args, write_port_file
from ouroboros.server_web import NoCacheStaticFiles, make_index_page, resolve_web_dir
from ouroboros.usage_accounting import ensure_legacy_imported
from ouroboros.gateway import collect_routes
from ouroboros.gateway import settings as _gateway_settings
from ouroboros.gateway.ws import (
    broadcast_ws,
    broadcast_ws_sync,
    close_all_ws,
    has_ws_clients as _has_ws_clients,
    set_event_loop as _set_ws_event_loop,
)

REPO_DIR = pathlib.Path(os.environ.get("OUROBOROS_REPO_DIR", pathlib.Path(__file__).parent))
DATA_DIR = pathlib.Path(os.environ.get("OUROBOROS_DATA_DIR",
    pathlib.Path.home() / "Ouroboros" / "data"))
DEFAULT_HOST = os.environ.get("OUROBOROS_SERVER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("OUROBOROS_SERVER_PORT", "8765"))
PORT_FILE = DATA_DIR / "state" / "server_port"

sys.path.insert(0, str(REPO_DIR))
if not os.environ.get("OUROBOROS_AGENT_PYTHON"):
    _agent_python = sys.executable
    if isinstance(_agent_python, str) and _agent_python:
        os.environ["OUROBOROS_AGENT_PYTHON"] = _agent_python

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_pytest_default_real_data_dir = (
    "pytest" in sys.modules
    and not os.environ.get("OUROBOROS_DATA_DIR")
    and DATA_DIR == pathlib.Path.home() / "Ouroboros" / "data"
)
if _pytest_default_real_data_dir:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, handlers=[logging.StreamHandler()])
else:
    _log_dir = DATA_DIR / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    # Only this process rotates (workers swap the inherited handler for a
    # WatchedFileHandler, see supervisor.workers). 2 MB x 3 held minutes of a
    # 64-lane campaign's log; 64 MB x 5 keeps a day's forensics.
    _file_handler = RotatingFileHandler(
        _log_dir / "server.log", maxBytes=64 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, handlers=[_file_handler, logging.StreamHandler()])


from ouroboros.observability import SecretRedactingLogFilter as _SecretRedactingLogFilter

for _handler in logging.getLogger().handlers:
    _handler.addFilter(_SecretRedactingLogFilter())
# httpx logs each request URL at INFO; polling transports put credentials in
# the URL path, so even redacted lines are noise at this level.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("server")

RESTART_EXIT_CODE = 42
PANIC_EXIT_CODE = 99
_restart_requested = threading.Event()
# Set only when the OWNER asked for the restart (the chat Restart button, and the
# control endpoints that restart on the owner's behalf). The single fact the
# re-exec needs to decide whether the runtime-mode ratchet pin rides along.
_owner_restart_requested = threading.Event()
_planned_delegate_restart_transaction_id = ""
_LAUNCHER_MANAGED = str(os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER", "") or "").strip() == "1"

# Captured in main() for Settings LAN-reachability metadata.
_BIND_HOST = DEFAULT_HOST


def _has_active_evolution_transaction() -> bool:
    try:
        path = DATA_DIR / "state" / "evolution_campaign.json"
        if not path.is_file():
            return False
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return False
        if raw.get("status") not in {"active", "paused"}:
            return False
        tx = raw.get("active_transaction")
        return isinstance(tx, dict) and not str(tx.get("commit_sha") or "").strip()
    except Exception:
        return False


def _installed_skill_names():
    """Names of skills currently installed ON DISK (disk-derived, not in-memory).

    Passed to the process-custody reaper so it can tell which skill-companion
    orphans are safe to reap (owner uninstalled). Disk-derived so it is correct
    independent of in-memory extension-reload timing; returns None on any failure
    so the reaper fails toward KEEP (never mass-kills live skills' companions).
    """
    try:
        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_loader import discover_skills

        names = {s.name for s in discover_skills(DATA_DIR, repo_path=get_skills_repo_path())}
        # Coalesce an EMPTY result to None ("unknown"), NOT "everything
        # uninstalled": discover_skills returns [] without raising when the skills
        # dir is momentarily unavailable; treating that as an empty install set
        # would let an enforced reap mass-kill live companions. None ⇒ keep-all.
        return names or None
    except Exception:
        log.debug("Could not compute installed skill names for custody reaper", exc_info=True)
        return None


def _restart_current_process(host: str, port: int) -> None:
    _restart_current_process_impl(
        host, port, repo_dir=REPO_DIR, log=log,
        owner_initiated=_owner_restart_requested.is_set(),
    )

from ouroboros.config import (
    SETTINGS_DEFAULTS,
    SettingsIntegrityError,
    load_settings, save_settings, verify_settings_integrity,
    apply_settings_to_env as _apply_settings_to_env,
)
from ouroboros.server_runtime import (
    apply_runtime_provider_defaults,
    has_startup_ready_provider,
    needs_local_model_autostart,
    setup_remote_if_configured,
    ws_heartbeat_loop,
)

_supervisor_ready = threading.Event()
_supervisor_error: Optional[str] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_supervisor_thread: Optional[threading.Thread] = None
_consciousness: Any = None


def _describe_bg_consciousness_state(requested_enabled: bool) -> dict:
    snapshot = _consciousness.status_snapshot() if _consciousness else {}
    running = bool(snapshot.get("running"))
    paused = bool(snapshot.get("paused"))
    next_wakeup_sec = int(snapshot.get("next_wakeup_sec") or 0)
    idle_reason = str(snapshot.get("last_idle_reason") or "")
    detail = "Background consciousness is off."
    status = "disabled"

    if requested_enabled and running and paused:
        status = "paused"
        detail = "Paused while another foreground task is active."
    elif requested_enabled and running and idle_reason == "thinking":
        status = "running"
        detail = "Background consciousness is thinking now."
    elif requested_enabled and running and idle_reason == "budget_blocked":
        status = "budget_blocked"
        detail = "Background consciousness hit its budget allocation and is waiting."
    elif requested_enabled and running:
        status = "running"
        detail = (
            "Background consciousness is idle between wakeups."
            + (f" Next wakeup in {next_wakeup_sec}s." if next_wakeup_sec > 0 else "")
        )
    elif requested_enabled:
        status = "stopped"
        detail = "Enabled in state, but the background thread is not running."

    if idle_reason == "error_backoff" and snapshot.get("last_error"):
        status = "error_backoff"
        detail = f"Waiting to retry after an internal error: {snapshot['last_error']}"

    return {
        "enabled": requested_enabled,
        "status": status,
        "detail": detail,
        **snapshot,
    }


def _start_supervisor_if_needed(settings: dict) -> bool:
    """Start the supervisor once when runtime providers become available."""
    global _supervisor_thread, _supervisor_error
    if not has_startup_ready_provider(settings):
        return False
    if _supervisor_thread and _supervisor_thread.is_alive():
        return False
    _supervisor_error = None
    _supervisor_thread = threading.Thread(
        target=_run_supervisor,
        args=(settings,),
        daemon=True,
        name="supervisor-main",
    )
    _supervisor_thread.start()
    return True


def _task_belongs_to_chat(ctx: Any, task_id: str, task_obj: Dict[str, Any], chat_id: int) -> bool:
    try:
        if int(task_obj.get("chat_id") or 0) == int(chat_id or 0):
            return True
    except (TypeError, ValueError):
        pass
    try:
        from ouroboros.projects_registry import project_chat_for_task

        return int(project_chat_for_task(ctx.DRIVE_ROOT, task_id) or 0) == int(chat_id or 0)
    except Exception:
        return False


def _active_direct_root(ctx: Any) -> Dict[str, Any]:
    """Snapshot the one in-process direct root without creating queue state."""
    try:
        agent = ctx.get_chat_agent()
        lock = getattr(agent, "_owner_message_admission_lock", None)
        if lock is None:
            return {}
        with lock:
            task_id = str(getattr(agent, "_current_task_id", "") or "").strip()
            if (
                not getattr(agent, "_busy", False)
                or not getattr(agent, "_accepting_owner_messages", False)
                or not task_id
            ):
                return {}
            metadata = getattr(agent, "_current_task_metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            return {
                "task_id": task_id,
                "status": "running",
                "title": _clip_marked(metadata.get("title"), 120),
                "objective": _clip_marked(getattr(agent, "_current_task_text", ""), 600),
                "project_id": str(metadata.get("project_id") or ""),
                "chat_id": int(getattr(agent, "_current_chat_id", 0) or 0),
                "started_at": float(getattr(agent, "_task_started_ts", 0.0) or 0.0),
                "steerable": True,
                "direct_chat": True,
            }
    except Exception:
        return {}


def _addressable_root_tasks(ctx: Any, chat_id: Optional[int] = None) -> list:
    """Compact RUNNING+PENDING owner-root manifest, without choosing a target."""
    out: list = []
    seen: set[str] = set()

    def _add(task_id: Any, task_obj: Any, status: str, started_at: Any = None) -> None:
        tid = str(task_id or "").strip()
        if not tid or tid in seen or not isinstance(task_obj, dict):
            return
        if task_obj.get("_is_direct_chat") or str(task_obj.get("delegation_role") or "") == "subagent":
            return
        if chat_id is not None and not _task_belongs_to_chat(ctx, tid, task_obj, int(chat_id or 0)):
            return
        objective = str(
            task_obj.get("objective") or task_obj.get("description") or task_obj.get("text") or ""
        ).strip()
        out.append({
            "task_id": tid,
            "status": status,
            "title": _clip_marked(task_obj.get("title"), 120),
            "objective": _clip_marked(objective, 600),
            "project_id": str(task_obj.get("project_id") or ""),
            "started_at": started_at,
            "steerable": True,
        })
        seen.add(tid)

    for tid, running in list(getattr(ctx, "RUNNING", {}).items()):
        if not isinstance(running, dict):
            continue
        task_obj = running.get("task") if isinstance(running.get("task"), dict) else running
        _add(tid, task_obj, "running", running.get("started_at"))
    for pending in list(getattr(ctx, "PENDING", []) or []):
        if isinstance(pending, dict):
            _add(pending.get("id"), pending, "pending", pending.get("queued_at"))
    direct = _active_direct_root(ctx)
    if direct and str(direct.get("task_id") or "") not in seen:
        if chat_id is None or int(direct.get("chat_id") or 0) == int(chat_id or 0):
            out.append(direct)
    return out


def _stage_mailbox_attachments(
    ctx: Any,
    task_drive: pathlib.Path,
    task_id: str,
    task_metadata: Any,
    image_data: Any = None,
) -> tuple[str, list, str]:
    """Stage one routed turn's files into the existing task artifact store.

    Returns ``(attachment_note, staged_manifest, rendered_report)`` — the manifest is kept so a
    refused admission (the cancel-pending re-check inside the mailbox
    transaction) can remove exactly the files this call staged (GR2-9).
    """
    metadata = task_metadata if isinstance(task_metadata, dict) else {}
    uploads = list(metadata.get("chat_attachment_uploads") or [])
    temp_source: Optional[pathlib.Path] = None
    if image_data and not uploads:
        # Non-Web transports may carry an inline image rather than an uploaded
        # path. Materialise it only long enough for the canonical staging helper
        # to copy it into the addressed task's artifact store.
        try:
            raw = base64.b64decode(str(image_data[0] or ""), validate=True)
            if raw and len(raw) <= 50 * 1024 * 1024:
                mime = str(image_data[1] or "image/jpeg").lower()
                suffix = ".png" if "png" in mime else ".webp" if "webp" in mime else ".jpg"
                temp_source = pathlib.Path(ctx.DRIVE_ROOT) / "uploads" / f"routed-{uuid.uuid4().hex}{suffix}"
                temp_source.parent.mkdir(parents=True, exist_ok=True)
                with temp_source.open("xb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                uploads.append({"path": str(temp_source), "label": "owner image"})
        except Exception:
            log.warning("Unable to stage routed inline image for task %s", task_id, exc_info=True)
    try:
        if not uploads:
            return "", [], ""
        from ouroboros.artifacts import stage_task_attachments
        from ouroboros.gateway.tasks import _render_attachment_lines

        manifest = stage_task_attachments(task_drive, task_id, uploads)
        rendered = _render_attachment_lines(manifest)
        note = f"\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]" if rendered else ""
        return note, manifest, rendered
    finally:
        if temp_source is not None:
            try:
                temp_source.unlink(missing_ok=True)
            except OSError:
                log.debug("Unable to remove routed attachment staging source", exc_info=True)


def _route_project_chat_to_running_task(
    ctx: Any,
    chat_id: int,
    message: str,
    client_message_id: str = "",
    *,
    task_metadata: Any = None,
    image_data: Any = None,
) -> str:
    """Deliver a Project follow-up to the sole RUNNING/PENDING root mailbox.

    Multi-project (v6.32.0): a focused project room with exactly ONE active pooled
    task IS that task's context, so a follow-up is delivered to it as a TRANSPORT
    invariant (the loop drains the mailbox every round) — there is no routing CHOICE
    to make. But when the room has ZERO or MORE THAN ONE steerable task, picking a
    target is a JUDGMENT, and code must never make it mechanically (BIBLE P5 LLM-first,
    v6.34.0 WS1): this returns "" so the message flows to the decision turn, where the
    agent sees `current_chat.running_tasks` and chooses `steer_task` / `promote_chat_to_task`.
    Returns the delivered task id, or "" (no delivery — fall through to the decision lane).

    A chat is a project thread by REGISTRY membership, not a bare numeric range —
    large external-transport (Telegram-style) chat ids must not be misclassified and
    have their owner messages swallowed.
    """
    try:
        if not _project_id_for_registered_chat(ctx, chat_id):
            return ""
    except Exception:
        return ""
    try:
        steerable = _addressable_root_tasks(ctx, chat_id)
        # Exactly one candidate => unambiguous transport. Zero or many => a routing
        # decision the AGENT must make (P5/WS1), so do not deliver here.
        if len(steerable) != 1:
            return ""
        candidate = steerable[0]
        tid = str(candidate["task_id"])
        direct_agent = None
        direct_lock = None
        if candidate.get("direct_chat"):
            direct_agent = ctx.get_chat_agent()
            direct_lock = getattr(direct_agent, "_owner_message_admission_lock", None)
            if direct_lock is None:
                return ""
        task_obj: Dict[str, Any] = {}
        running = getattr(ctx, "RUNNING", {}).get(tid)
        if isinstance(running, dict):
            task_obj = running.get("task") if isinstance(running.get("task"), dict) else running
        if not task_obj:
            task_obj = next(
                (row for row in list(getattr(ctx, "PENDING", []) or []) if str(row.get("id") or "") == tid),
                {},
            )
        from ouroboros.project_dialogue import routing_target_label

        target_label = routing_target_label(
            ctx.DRIVE_ROOT,
            "mailbox_delivery",
            tid,
            task=task_obj or candidate,
            project_id=str((task_obj or candidate).get("project_id") or ""),
        )
        from ouroboros.owner_mailbox import write_owner_message
        from supervisor.queue import (
            ACCEPTANCE_FENCES,
            _queue_lock,
            _task_drive_for_task,
            persist_queue_snapshot,
        )

        # Active drive (child drive for forked/workspace tasks) — mirror
        # forward_to_worker / steer_task so the mailbox lands where the task
        # actually drains it, not the canonical root. A stable msg_id derived from
        # client_message_id makes this 1:1 delivery idempotent — a WebSocket retry of
        # the same message can't double-deliver (drain_owner_entries dedups by msg_id),
        # matching steer_task's contract.
        direct_lock_held = False
        queue_lock_held = False
        fence_generation_changed = False
        active_fence = None
        if direct_lock is not None:
            direct_lock.acquire()
            direct_lock_held = True
            if not (
                getattr(direct_agent, "_busy", False)
                and getattr(direct_agent, "_accepting_owner_messages", False)
                and str(getattr(direct_agent, "_current_task_id", "") or "") == tid
            ):
                direct_lock.release()
                direct_lock_held = False
                return ""
        task_drive = pathlib.Path(ctx.DRIVE_ROOT) if direct_lock_held else _task_drive_for_task(task_obj, tid)
        msg_id = f"{client_message_id}:{tid}" if client_message_id else None
        staged_manifest: list = []
        attachment_report = ""
        message_written = False
        cancel_refused_in_txn = False

        def _drop_staged_inputs() -> None:
            # GR2-9: the admission was refused, so the files staged for this
            # message must not linger in the dying task's artifact store.
            if not staged_manifest:
                return
            try:
                from ouroboros.artifacts import remove_staged_attachments

                remove_staged_attachments(staged_manifest)
            except Exception:
                log.debug("staged-attachment cleanup failed for %s", tid, exc_info=True)

        try:
            # GR2-9 ordering: check cancellation BEFORE staging — the old order
            # copied the owner's files into the artifact store of a task whose
            # cancellation was already pending, then refused the message. The
            # cheap up-front check runs off the lock; the transactional
            # re-checks below still run and remove the staged inputs on refusal.
            from ouroboros.cancel_intents import cancel_pending

            if cancel_pending(ctx.DRIVE_ROOT, tid):
                log.info("Mailbox follow-up refused for %s: cancel pending (pre-staging)", tid)
                return ""
            attachment_note, staged_manifest, attachment_report = _stage_mailbox_attachments(
                ctx, task_drive, tid, task_metadata, image_data,
            )
            if direct_lock_held:
                # AR2-6 (fable): the direct-agent lane used to skip the
                # cancel-pending admission check the queue lane makes below — a
                # direct turn whose cancellation is pending must not accept a
                # new owner message either. Same predicate, same honest
                # fall-through to the direct chat lane.
                if cancel_pending(ctx.DRIVE_ROOT, tid):
                    log.info("Mailbox follow-up refused for %s: cancel pending (direct lane)", tid)
                    _drop_staged_inputs()
                    return ""
            if not direct_lock_held:
                _queue_lock.acquire()
                queue_lock_held = True
                live_meta = getattr(ctx, "RUNNING", {}).get(tid)
                still_pending = any(
                    isinstance(row, dict) and str(row.get("id") or "") == tid
                    for row in list(getattr(ctx, "PENDING", []) or [])
                )
                if live_meta is None and not still_pending:
                    return ""
                # Phase A: a task whose cancellation is PENDING must not accept a
                # new owner message — same refusal the steer_task route makes,
                # checked inside this admission transaction. Falling through to
                # the direct lane is the honest outcome: the follow-up is
                # answered in chat instead of handed to a dying task.
                if cancel_pending(ctx.DRIVE_ROOT, tid):
                    log.info("Mailbox follow-up refused for %s: cancel pending", tid)
                    cancel_refused_in_txn = True
                    return ""
                fence_root = str(task_obj.get("root_task_id") or tid)
                active_fence = ACCEPTANCE_FENCES.get(fence_root)
                if isinstance(active_fence, dict) and str(active_fence.get("status") or "") == "sealed":
                    return ""
            if not write_owner_message(
                task_drive, f"{message}{attachment_note}", tid, msg_id=msg_id,
                client_surface=(
                    dict(task_metadata["client_surface"])
                    if isinstance(task_metadata, dict) and isinstance(task_metadata.get("client_surface"), dict)
                    else None
                ),
                attachment_manifest=staged_manifest if staged_manifest else None,
            ):
                return ""
            message_written = True
            if direct_lock_held:
                direct_agent._owner_message_generation = int(
                    getattr(direct_agent, "_owner_message_generation", 0) or 0
                ) + 1
            else:
                if isinstance(active_fence, dict) and str(active_fence.get("status") or "") == "active":
                    active_fence["owner_message_generation"] = int(
                        active_fence.get("owner_message_generation") or 0
                    ) + 1
                    fence_generation_changed = True
        finally:
            if queue_lock_held:
                _queue_lock.release()
            if direct_lock_held:
                direct_lock.release()
            if cancel_refused_in_txn:
                # After the lock release: unlinking staged files is file I/O the
                # global queue lock should not wait on.
                _drop_staged_inputs()
            elif staged_manifest and not message_written:
                _drop_staged_inputs()
        if fence_generation_changed:
            persist_queue_snapshot(reason="acceptance_fence_owner_message")
        if isinstance(task_metadata, dict) and staged_manifest:
            task_metadata["_attachment_manifest"] = [
                dict(item) for item in staged_manifest if isinstance(item, dict)
            ]
            task_metadata["_attachment_report"] = attachment_report
        if isinstance(task_metadata, dict):
            task_metadata["_routing_target_label"] = target_label
        if attachment_report:
            try:
                ctx.send_with_budget(
                    chat_id,
                    f"📎 Attachment staging report for {target_label or 'Task'}:\n"
                    f"{attachment_report}",
                )
            except Exception:
                log.debug("Mailbox attachment report notice failed for %s", tid, exc_info=True)
        return tid
    except Exception:
        log.debug("Mailbox follow-up routing failed; falling back to direct lane", exc_info=True)
    return ""


def _owner_evolution_stop(ctx: Any, chat_id: int) -> str:
    """The ``/evolve off`` stop transaction; returns the final status wording.

    Cancels live evolution work BEFORE the terminal campaign close:
    ``complete_evolution_campaign`` runs the per-cycle worktree cleanup, which
    skips while a task still holds the shared worktree — so the running cycle
    must be gone first. PENDING evolution tasks go through the SAME durable
    intent + typed custody (GR2-13); the old in-place prune left them with no
    intent, no terminal result and no ``task_done``, and a stop with still-live
    leftovers was declared clean.
    """
    stop_incomplete = False
    try:
        from supervisor.queue import evolution_stop_report, stop_evolution_tasks
        from ouroboros.post_task_evolution import drop_pending_request

        # Fast path: drop any queued post-task promotion so it cannot re-arm on
        # the next boot tick (the evolution_owner_stopped flag is the durable backstop).
        drop_pending_request(ctx.DRIVE_ROOT)
        stopped = stop_evolution_tasks("disabled via owner chat")
        ctx.sort_pending()
        ctx.persist_queue_snapshot(reason="evolve_off")
        stop_lines, stop_incomplete = evolution_stop_report(stopped)
        for line in stop_lines:
            ctx.send_with_budget(chat_id, line)
    except Exception:
        log.warning("Evolution stop transaction failed", exc_info=True)
        stop_incomplete = True
    try:
        from supervisor.evolution_lifecycle import complete_evolution_campaign

        if stop_incomplete:
            # GR3-3: an INCOMPLETE stop must not close the campaign — a terminal
            # "stopped" over still-live evolution work declares a clean ending
            # that did not happen. The campaign stays open; the durable
            # evolution_owner_stopped flag already blocks new cycles, and the
            # owner-stop backstop (supervisor/events.py, on the live task's own
            # settle) closes the campaign once nothing is live.
            log.warning(
                "Evolution stop is incomplete; campaign left open for the "
                "settle-time owner-stop backstop",
            )
        else:
            # Terminal close (not a resumable pause): /evolve start mints a FRESH
            # campaign rather than resurrecting this one.
            complete_evolution_campaign("disabled via owner chat", status="stopped")
    except Exception:
        log.warning("Failed to update evolution campaign state", exc_info=True)
    if stop_incomplete:
        return ("OFF (mode disabled) — but the stop is INCOMPLETE: see the "
                "still-live task(s) above. The campaign stays open until they "
                "settle. Post-task auto-evolution stays paused until /evolve start")
    return "OFF — post-task auto-evolution also paused until /evolve start"


def _clip_marked(value: str, limit: int) -> str:
    """Clip a routing/recognition string but NEVER silently: an explicit omission
    marker keeps a decision-context field honest (no silent ``[:N]`` truncation of a
    cognitive/routing artifact — DEVELOPMENT.md). The marker + the full task_id keep
    enough signal for the agent to disambiguate the steer target."""
    s = str(value or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f" …[+{len(s) - limit} chars omitted]"


def _chat_running_tasks(ctx: Any, chat_id: int) -> list:
    """Structural snapshot of the owner's RUNNING root tasks in THIS chat (id +
    objective + recency). The decision turn reads this from runtime context to
    pick a steer_task target by its own judgment — code only exposes the state,
    it never auto-chooses (BIBLE P5). Direct in-process turns and subagents are
    not pooled RUNNING tasks and are excluded."""
    return [row for row in _addressable_root_tasks(ctx, chat_id) if row.get("status") == "running"]


def _task_result_ground_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded typed projection of one task result for a routing/promote turn:
    identity, outcome, and WHERE THE WORK LIVES (workspace facts + artifact refs).
    Never raw result text — a router turn that reconstructs prior work from chat
    memory instead of these facts invents false premises (the saga's "continue"
    promotion rebuilt a finished game from scratch)."""
    bundle = row.get("artifact_bundle") if isinstance(row.get("artifact_bundle"), dict) else {}
    artifacts = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), list) else []
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    preflight = meta.get("workspace_preflight") if isinstance(meta.get("workspace_preflight"), dict) else {}
    git = preflight.get("git") if isinstance(preflight.get("git"), dict) else {}
    task_id = str(row.get("task_id") or row.get("id") or "")
    human_label = _clip_marked(
        row.get("title") or row.get("objective") or row.get("description") or task_id, 120,
    )
    out = {
        "task_id": task_id,
        "status": str(row.get("status") or ""),
        "title": _clip_marked(row.get("title"), 120),
        "objective": _clip_marked(row.get("objective") or row.get("description"), 300),
        "project_id": str(row.get("project_id") or ""),
        "reason_code": str(row.get("reason_code") or ""),
        "workspace_root": str(row.get("workspace_root") or ""),
        "workspace_mode": str(row.get("workspace_mode") or ""),
        "artifact_status": str(row.get("artifact_status") or ""),
        "artifact_refs": [
            str(item.get("path") or item.get("name") or "")
            for item in artifacts[:8] if isinstance(item, dict)
        ],
        "authority_source": {
            "kind": "task_result",
            "task_id": task_id,
            "human_label": human_label,
            "tool": "get_task_result",
            "arguments": {"task_id": task_id, "include_authority": True},
        },
    }
    if git:
        out["workspace_git_at_start"] = {
            "head": str(git.get("head") or ""),
            "branch": str(git.get("branch") or ""),
            "dirty": bool(git.get("dirty")),
        }
    return out


def _latest_project_task_result(ctx: Any, project_id: str) -> Optional[Dict[str, Any]]:
    """Newest task result bound to ``project_id`` WITHOUT replaying the whole
    store (DEVELOPMENT "Projection over replay"). The registry row's durable
    ``last_task_result_id`` pointer (stamped at project-task finalization) is
    read FIRST — one direct file fetch, immune to how many newer foreign
    results exist. Only when the pointer is absent or stale (missing/
    unparseable/foreign file) does the fallback run: the bounded newest-64
    mtime scan, then — for pre-pointer projects only — a disclosed full scan
    of the store (the lazy self-heal for rows finalized before the pointer
    existed; with zero matching results nothing is written back, so it repeats
    per lookup until a matching result exists). Only the ABSENT-pointer case
    writes the pointer back: a non-empty pointer that failed to resolve is
    usually a split-drive result in flight (finalization stamps the pointer
    before the canonical copy-back lands), so overwriting it from the scan
    would permanently regress it to an older result — serve the scan hit and
    let the pointer resolve itself. The steady state needs no
    ouroboros/context_budget.py threshold enrollment (that table guards
    recurring full-store replays)."""
    from ouroboros.projects_registry import get_project, update_project
    from ouroboros.task_results import load_task_result, task_results_dir
    from ouroboros.utils import read_json_dict

    try:
        pointer = str((get_project(ctx.DRIVE_ROOT, project_id) or {}).get(
            "last_task_result_id") or "").strip()
    except Exception:
        pointer = ""
    if pointer:
        pointed = load_task_result(ctx.DRIVE_ROOT, pointer)
        if isinstance(pointed, dict) and str(pointed.get("project_id") or "") == project_id:
            return pointed
        log.debug(
            "project last-task-result pointer for %r is stale (%s); "
            "falling back to the bounded scan", project_id, pointer,
        )

    paths = list(task_results_dir(ctx.DRIVE_ROOT, create=False).glob("*.json"))
    try:
        paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        paths.sort(key=lambda path: path.name, reverse=True)
    row = None
    for path in paths[:64]:
        candidate = read_json_dict(path)
        if candidate is not None and str(candidate.get("project_id") or "") == project_id:
            row = candidate
            break
    if row is None and len(paths) > 64:
        log.info(
            "project last-task-result: %r missed the bounded scan; running the "
            "full-store self-heal scan (%d files)", project_id, len(paths),
        )
        for path in paths[64:]:
            candidate = read_json_dict(path)
            if candidate is not None and str(candidate.get("project_id") or "") == project_id:
                row = candidate
                break
    if row is not None and not pointer:
        try:
            update_project(ctx.DRIVE_ROOT, project_id, last_task_result_id=str(
                row.get("task_id") or row.get("id") or ""))
        except Exception:
            log.debug("project last-task-result pointer write-back failed", exc_info=True)
    return row


def _main_routing_manifest(ctx: Any) -> Dict[str, Any]:
    """Bounded canonical facts for one Main-chat LLM routing decision."""
    from ouroboros.projects_registry import list_projects
    from ouroboros.task_results import list_task_results
    from ouroboros.utils import iter_jsonl_objects

    projects = [{
        "project_id": str(row.get("id") or ""),
        "name": _clip_marked(row.get("name"), 120),
        "chat_id": int(row.get("chat_id") or 0),
        "lifecycle": str(row.get("lifecycle") or "active"),
        # Registry-canonical working folder: the router turn's ground truth for
        # where a project's work lives (Q8-A).
        "working_dir": str(row.get("working_dir") or ""),
    } for row in list_projects(ctx.DRIVE_ROOT)]
    roots = _addressable_root_tasks(ctx, None)

    all_results = list_task_results(ctx.DRIVE_ROOT)
    all_results.sort(key=lambda row: str(row.get("ts") or row.get("updated_at") or ""), reverse=True)
    finals = [_task_result_ground_truth(row) for row in all_results[:16]]

    dialogue_rows: list = []
    chat_paths = sorted(
        (pathlib.Path(ctx.DRIVE_ROOT) / "archive").glob("chat_*.jsonl"),
        key=lambda path: path.name,
    )[-2:] + [pathlib.Path(ctx.DRIVE_ROOT) / "logs" / "chat.jsonl"]
    for path in chat_paths:
        for row in iter_jsonl_objects(path):
            text = str(row.get("text") or "").strip()
            if text:
                dialogue_rows.append({
                    "ts": str(row.get("ts") or ""),
                    "direction": str(row.get("direction") or ""),
                    "chat_id": int(row.get("chat_id") or 1),
                    "text": _clip_marked(text, 500),
                    "task_id": str(row.get("task_id") or ""),
                    "client_message_id": str(row.get("client_message_id") or ""),
                })
    dialogue = dialogue_rows[-20:]
    return {
        "projects": projects[:40],
        "root_tasks": roots[:40],
        "final_results": finals,
        "recent_canonical_dialogue": dialogue,
        "omissions": {
            "projects": max(0, len(projects) - 40),
            "root_tasks": max(0, len(roots) - 40),
            "final_results": max(0, len(all_results) - 16),
            "dialogue_rows": max(0, len(dialogue_rows) - 20),
        },
    }


def _decision_turn_metadata(ctx: Any, chat_id: int, client_message_id: str, task_metadata: Any) -> Any:
    """Enrich a chat turn's metadata with the structural facts the decision turn
    needs: the RUNNING tasks in THIS chat (so it can steer_task the right one
    instead of spawning a duplicate) and the originating message id (for idempotent
    steer delivery). P5-clean: surfaces state only; the agent picks the target by
    judgment among answer / steer_task / promote_chat_to_task / route_to_project."""
    md = dict(task_metadata) if isinstance(task_metadata, dict) else {}
    swarm_intent = bool(md.get("force_plan"))
    addressable_here = _addressable_root_tasks(ctx, chat_id)
    running_here = [row for row in addressable_here if row.get("status") == "running"]
    project_id = str(md.get("project_id") or "").strip() or _project_id_for_registered_chat(
        ctx, chat_id,
    )
    is_main_lane = not bool(project_id)
    try:
        # Every non-Project owner transport is the Main lane.  External transports
        # commonly use a real provider chat id rather than Web's numeric ``1``;
        # keying this decision to ``chat_id == 1`` made their canonical router see
        # neither Projects nor globally addressable roots.
        main_manifest = _main_routing_manifest(ctx) if is_main_lane else {}
        if main_manifest and not (
            main_manifest.get("projects") or main_manifest.get("root_tasks")
        ):
            main_manifest = {}
    except Exception:
        log.warning("Unable to build Main routing manifest", exc_info=True)
        main_manifest = {"error": "routing_manifest_unavailable"} if is_main_lane else {}
    if not swarm_intent and not addressable_here and not client_message_id and not main_manifest:
        return task_metadata
    if addressable_here:
        md["current_chat"] = {
            "chat_id": int(chat_id or 0),
            "running_tasks": running_here,
            "addressable_root_tasks": addressable_here,
        }
    if main_manifest:
        md["main_routing_manifest"] = main_manifest
    if project_id:
        # Ground truth for a project-room "continue" decision (Q8-A): the thread's
        # most recent task result as a bounded typed projection. Without it the
        # router turn has only chat memory about where prior work lives.
        try:
            row = _latest_project_task_result(ctx, project_id)
            if row is not None:
                md["project_last_task_result"] = _task_result_ground_truth(row)
        except Exception:
            log.debug("project last-task-result projection failed", exc_info=True)
    if client_message_id:
        md["client_message_id"] = client_message_id
    option_roots = (
        list(main_manifest.get("root_tasks") or [])
        if is_main_lane and isinstance(main_manifest, dict)
        else addressable_here
    )
    manual_options = [] if swarm_intent else [
        {
            "action": "steer_task",
            "task_id": row["task_id"],
            "status": row["status"],
            "title": row.get("title") or row.get("objective"),
            "project_id": str(row.get("project_id") or ""),
        }
        for row in option_roots
        if isinstance(row, dict) and row.get("task_id")
    ]
    if not swarm_intent and is_main_lane and isinstance(main_manifest, dict):
        manual_options.extend({
            "action": "new_task_in_project",
            "project_id": str(row.get("project_id") or ""),
            "project_name": str(row.get("name") or row.get("project_id") or "Project"),
            "label": f"New task in {str(row.get('name') or 'Project')}",
        } for row in list(main_manifest.get("projects") or []) if isinstance(row, dict))
    elif project_id and not swarm_intent:
        manual_options.append({
            "action": "new_task_in_project",
            "project_id": project_id,
            "label": "New task in Project",
        })
    routing_contract = {
        "llm_first": True,
        "source_lane": "main" if is_main_lane else "project",
        "valid_actions": (
            (["promote_chat_to_task", "route_to_project"] if is_main_lane else ["promote_chat_to_task"])
            if swarm_intent else
            [
                "answer_inline", "steer_task", "promote_chat_to_task", "route_to_project",
                "needs_manual_target",
            ]
        ),
        "on_uncertain_or_invalid_target": (
            "promote_chat_to_task" if swarm_intent else "needs_manual_target"
        ),
        "manual_options": manual_options,
    }
    if not swarm_intent:
        routing_contract["manual_target_tool"] = {"name": "route_to_project", "project_id": ""}
    md["routing_contract"] = routing_contract
    return md


def _supervisor_loop_stalled(last_tick: float, now: float, deadline_sec: int) -> bool:
    """True when the supervisor loop has not published a liveness tick within the
    deadline (WS3). deadline_sec<=0 disables the watchdog."""
    return deadline_sec > 0 and (now - last_tick) > deadline_sec


def _chat_turn_wedged(busy: bool, last_activity_ts, now: float, deadline_sec: int) -> bool:
    """True when an IN-PROCESS direct-chat turn is busy but its liveness tick has been
    silent past the deadline (WS3). ``last_activity_ts is None`` => the turn has not
    started its liveness loop yet (not wedged). deadline_sec<=0 disables the check."""
    if not busy or last_activity_ts is None or deadline_sec <= 0:
        return False
    return (now - last_activity_ts) > deadline_sec


def _alert_chat_turn_wedge(task_id, gap: float) -> None:
    """WS3: a direct-chat turn is heartbeat-silent. New messages still get answered
    (WS10 ephemeral decision turns), but a hung IN-PROCESS turn cannot be killed and
    still holds the chat-agent lock, so admission cannot be freed in-process (full
    kill-ability via out-of-process direct chat was deferred per owner). Surface it +
    recommend /restart, which is the safe full recovery."""
    from supervisor.state import append_jsonl, load_state
    try:
        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "chat_turn_wedge",
            "task_id": str(task_id or ""), "silent_sec": round(gap, 1),
        })
    except Exception:
        log.debug("chat-turn wedge log failed", exc_info=True)
    try:
        owner_chat = int((load_state() or {}).get("owner_chat_id") or 0)
        if owner_chat:
            from supervisor.message_bus import send_with_budget
            send_with_budget(
                owner_chat,
                f"⚠️ A chat turn looks wedged (~{int(gap)}s with no heartbeat). New messages "
                "still get answered, but the stuck turn can't be cleared in-process — /restart "
                "to fully recover it.",
                is_progress=True,
                task_id=str(task_id or ""),
                progress_meta={
                    "task_incident": "chat_turn_wedge",
                    "toast_once": f"{task_id or 'direct-chat'}:chat_turn_wedge",
                },
            )
    except Exception:
        log.debug("chat-turn wedge owner alert failed", exc_info=True)


def _start_supervisor_liveness_watchdog(liveness: list, stop_event=None) -> None:
    """Dedicated daemon thread (NOT inside the supervisor loop, so it fires even when
    that loop stalls). It ALERTS the owner on two silent-wedge classes — a supervisor
    loop stall (new-message intake starvation) and a heartbeat-silent in-process
    direct-chat turn — converting a multi-hour silent wedge into an immediate signal.
    It deliberately does NOT kill a hung thread or free the chat-agent lock: the wedged
    turn holds that lock for its whole duration, so in-process admission-freeing is
    unsafe (out-of-process direct chat for full kill-ability was deferred per owner);
    WS10 ephemeral decision turns keep the chat responsive meanwhile. ``stop_event`` is
    a PER-GENERATION token: when the supervisor loop that owns ``liveness`` exits (incl.
    the crash-storm death path, which never sets the global restart flag), it is set so
    this watchdog stops watching a now-stale liveness list (no false post-revival alert)."""
    from ouroboros.config import get_supervisor_liveness_deadline_sec

    deadline = get_supervisor_liveness_deadline_sec()
    if deadline <= 0:
        return

    def _watch() -> None:
        from supervisor.state import append_jsonl, load_state
        interval = min(15, max(1, deadline // 3))
        loop_alerted = False
        wedged_task = None
        while not _restart_requested.is_set() and not (stop_event is not None and stop_event.is_set()):
            time.sleep(interval)
            # ONE clock: both halves measure an ELAPSED GAP against stamps taken on
            # the monotonic clock (the loop-liveness tick here, and the chat-turn
            # heartbeat in agent.py), so a wall-clock jump — NTP step, DST/timezone
            # change, manual set, VM resume — can neither fabricate a stall/wedge
            # nor mask a real one on either half.
            now = time.monotonic()
            # (1) Supervisor loop stall — new-message intake starvation.
            if _supervisor_loop_stalled(liveness[0], now, deadline):
                if not loop_alerted:
                    gap = now - liveness[0]
                    log.error(
                        "Supervisor loop STALLED ~%.0fs — new-message intake starved (WS10 "
                        "ephemeral chat still answers); investigate a blocking step.", gap,
                    )
                    try:
                        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
                            "ts": utc_now_iso(), "type": "supervisor_loop_stall", "stalled_sec": round(gap, 1),
                        })
                    except Exception:
                        log.debug("loop-stall log failed", exc_info=True)
                    try:
                        owner_chat = int((load_state() or {}).get("owner_chat_id") or 0)
                        if owner_chat:
                            from supervisor.message_bus import send_with_budget
                            send_with_budget(
                                owner_chat,
                                f"⚠️ My supervisor loop stalled for ~{int(gap)}s — new messages may be "
                                "delayed. I recover on the next tick or a restart; investigating.",
                                is_progress=True,
                                progress_meta={
                                    "task_incident": "supervisor_loop_stall",
                                    # pid disambiguates server GENERATIONS: the monotonic stamp alone can
                                    # repeat at a similar uptime offset across restarts, and the browser's
                                    # toast-dedupe set outlives this process while the page stays open.
                                    "toast_once": f"supervisor-loop-stall:{os.getpid()}:{int(liveness[0])}",
                                },
                            )
                    except Exception:
                        log.debug("loop-stall owner alert failed", exc_info=True)
                    loop_alerted = True
            else:
                loop_alerted = False
            # (1b) Event-bus SyncManager liveness — a dead manager crashes the
            # loop's get_nowait() with BrokenPipeError and takes the whole
            # campaign down (CyberGym r11/r12/r13). Log it loudly the moment it
            # dies so the cause is captured before the loop hits the pipe.
            try:
                from supervisor.workers import _event_q_manager_alive, _EVENT_Q_MANAGER
                if _EVENT_Q_MANAGER is not None and not _event_q_manager_alive():
                    log.error(
                        "Event-queue SyncManager is DEAD (generation %s) — the next "
                        "event-queue read will fail; the supervisor loop rebuilds the "
                        "bus in place on that read.",
                        getattr(__import__("supervisor.workers", fromlist=["_EVENT_Q_GENERATION"]), "_EVENT_Q_GENERATION", "?"),
                    )
            except Exception:
                log.debug("event-queue manager liveness probe failed", exc_info=True)
            # (2) In-process direct-chat turn wedge — a heartbeat-silent busy turn.
            try:
                from supervisor.workers import chat_turn_liveness
                busy, turn_task, turn_ts = chat_turn_liveness()
            except Exception:
                busy, turn_task, turn_ts = (False, None, None)
            if _chat_turn_wedged(busy, turn_ts, now, deadline):
                if wedged_task != turn_task:  # alert once per wedged turn
                    _alert_chat_turn_wedge(turn_task, now - (turn_ts or now))
                    wedged_task = turn_task
            elif not busy:
                wedged_task = None

    threading.Thread(target=_watch, name="supervisor-liveness-watchdog", daemon=True).start()


_LAST_CANCEL_INTENT_SWEEP = [0.0]
_LAST_USAGE_RECONCILE = [0.0]


def _periodic_supervisor_maintenance(last_custody_reap: list, last_review_reconcile: list) -> None:
    """Throttled periodic upkeep extracted from the supervisor loop: cancel-intent
    watchdog and pending child-ref promotion replay (every 20s), custody reap of
    orphaned task-scoped processes (every 600s) + review-job zombie reconcile
    (every 300s) + abandoned unresolved usage-attempt write-off (every 300s).
    Each cadence gates itself via its own last-run marker."""
    if time.time() - _LAST_CANCEL_INTENT_SWEEP[0] > 20:
        _LAST_CANCEL_INTENT_SWEEP[0] = time.time()
        try:
            # Phase A watchdog: re-feed open durable cancel intents into custody
            # (the ONE settle owner) so a lost control event can no longer wedge
            # a cancellation forever — the Poltergeist incident class.
            from supervisor.task_lifecycle import sweep_cancel_intents

            outcomes = sweep_cancel_intents()
            if outcomes:
                log.info("Cancel-intent watchdog settled: %s", outcomes)
        except Exception:
            log.debug("Cancel-intent watchdog sweep failed", exc_info=True)
        try:
            # Phase A2/F7: re-enqueue terminal answers registered as OWED whose
            # send never got confirmed (a crash between settle and send used to
            # lose the owner's answer forever — the incident class itself).
            from supervisor.terminal_delivery import replay_pending_deliveries

            replay_pending_deliveries(DATA_DIR)
        except Exception:
            log.debug("Pending terminal-delivery replay failed", exc_info=True)
        try:
            from ouroboros.observability import retry_pending_child_ref_promotions

            retry_pending_child_ref_promotions(DATA_DIR)
        except Exception:
            log.debug("Pending child-ref promotion retry failed", exc_info=True)
    if time.time() - last_custody_reap[0] > 600:
        last_custody_reap[0] = time.time()
        try:
            from ouroboros.process_custody import reap_orphaned_processes
            from supervisor.queue import RUNNING as _running_tasks

            live_tasks = set(_running_tasks.keys())
            reap_orphaned_processes(
                DATA_DIR, running_task_ids=live_tasks,
                live_owner_skills=_installed_skill_names(),
            )
            # A delegated Claudexor run is an orphan under exactly the same predicate:
            # its owning task is no longer running. It has no pid, so the process
            # reaper cannot see it — but it is still spending quota and still writing.
            _reconcile_delegated_runs(live_tasks)
            _cursor_refresh_settled_terminals()
        except Exception:
            log.debug("Periodic custody reap failed", exc_info=True)
    if time.time() - last_review_reconcile[0] > 300:
        last_review_reconcile[0] = time.time()
        _periodic_zombie_reconcile()
    if time.time() - _LAST_USAGE_RECONCILE[0] > 300:
        _LAST_USAGE_RECONCILE[0] = time.time()
        try:
            # Backstop for the unresolved-row lifecycle: rows whose task never
            # reached a terminal cost projection (crashes, pre-feature ledgers,
            # unattributed rows) are written off at their bound past the TTL.
            from ouroboros.usage_reconcile import reconcile_abandoned_unresolved_attempts

            outcome = reconcile_abandoned_unresolved_attempts(DATA_DIR)
            if outcome.get("terminalized"):
                log.info(
                    "Usage reconcile wrote off %d abandoned unresolved attempt(s)",
                    len(outcome["terminalized"]),
                )
        except Exception:
            log.debug("Periodic usage-attempt reconcile failed", exc_info=True)


def _reconcile_delegated_runs(running_task_ids: set) -> None:
    """Settle or cancel delegated runs whose owning task is gone (startup + tick)."""
    try:
        from ouroboros.claudexor_daemon import ensure_owned_gateway
        from ouroboros.delegate_custody import reconcile_orphaned_runs
        from ouroboros.delegate_recovery import recoverable_task_ids

        # The tick runs on the supervisor loop thread: a daemon sitting in its
        # recovery-only admission window must not hold that thread for the default
        # admission wait — skip-until-next-sweep is this caller's normal posture.
        outcomes = reconcile_orphaned_runs(
            DATA_DIR, running_task_ids=running_task_ids,
            gateway_factory=lambda: ensure_owned_gateway(admission_wait_sec=0),
            recoverable_task_ids=recoverable_task_ids(DATA_DIR),
        )
        if outcomes:
            log.info("Delegated-run reconciliation handled %d orphan(s): %s", len(outcomes), outcomes)
            # A run settled by this sweep may belong to a task that already wrote
            # its terminal result with a non-empty unreconciled disclosure — the
            # stored projection then lies forever (nanny-leaf S1). Audit-only
            # refresh; never cancels.
            from ouroboros.delegate_terminal import refresh_terminal_reconciliation

            for tid in {str(o.get("task_id") or "") for o in outcomes
                        if o.get("task_id") and (o.get("settled") or str(
                            o.get("action") or "") in (
                                "absent", "cancelled", "invocation_retired"))}:
                try:
                    refresh_terminal_reconciliation(DATA_DIR, tid)
                except Exception:
                    log.debug("Sweep terminal-result refresh failed for %s", tid, exc_info=True)
    except Exception:
        log.debug("Delegated-run reconciliation failed", exc_info=True)


def _cursor_refresh_settled_terminals() -> None:
    """Cursor-driven pass: runs settled OUTSIDE a generation's reconcile
    outcomes (terminal-boundary settlements, earlier generations) never
    reappear in the orphan sweep, so their tasks' stored evidence would stay
    stale forever. Bounded to newly appended custody rows per tick. At BOOT
    this runs AFTER the D1a backfill (see ``_startup_custody_sweep``), so a
    same-generation heal keeps its pinned ``boot_backfill`` attribution and
    the cursor's change-gated pass advances past it without a second write.
    """
    try:
        from ouroboros.delegate_terminal import refresh_recently_settled_terminals

        refreshed = refresh_recently_settled_terminals(DATA_DIR)
        if refreshed:
            log.info("Cursor refresh healed %d stale terminal result(s)", refreshed)
    except Exception:
        log.debug("Cursor terminal-refresh pass failed", exc_info=True)


def _startup_custody_sweep() -> None:
    """Both custody surfaces, swept once per generation at supervisor startup.

    Nothing is running yet, so every ledgered process and every open delegated run is
    by definition ownerless: the generation that was watching them did not survive.
    """
    try:
        from ouroboros.process_custody import reap_orphaned_processes

        reaped = reap_orphaned_processes(DATA_DIR, live_owner_skills=_installed_skill_names())
        if reaped:
            log.info("Process custody reaper killed %d orphaned process(es): %s", len(reaped), reaped)
    except Exception:
        log.debug("Process custody startup reap failed", exc_info=True)
    _reconcile_delegated_runs(set())
    try:
        # D1a boot backfill, ONCE per generation and AFTER the orphan reconcile
        # (so this generation's settlements are already visible to the audit):
        # a run settled in a PREVIOUS generation never appears in any current
        # pass's outcomes, so the sweep-side refresh above can never reach its
        # task's stored disclosure — the backfill joins from the stored terminal
        # results instead and heals every generation-crossing stale row.
        from ouroboros.delegate_terminal import backfill_terminal_reconciliations

        refreshed = backfill_terminal_reconciliations(DATA_DIR)
        if refreshed:
            log.info("Boot custody backfill refreshed %d stored disclosure(s): %s",
                     len(refreshed), refreshed)
    except Exception:
        log.debug("Boot custody-disclosure backfill failed", exc_info=True)
    _cursor_refresh_settled_terminals()
    try:
        # Phase A boot migration: legacy ``cancel_requested`` status latches
        # become ordinary durable cancel intents; the supervisor watchdog then
        # drives each through custody to a real settled outcome.
        from ouroboros.cancel_intents import migrate_legacy_cancel_latches

        migrated = migrate_legacy_cancel_latches(DATA_DIR)
        if migrated:
            log.info("Migrated %d legacy cancel latch(es) to durable intents: %s",
                     len(migrated), migrated)
    except Exception:
        log.debug("Legacy cancel-latch migration failed", exc_info=True)
    try:
        # Boot half of the durable terminal outbox: an answer that was registered
        # as owed but whose send never completed (crash between settle and send)
        # is re-enqueued exactly once — the delivered registry suppresses a copy
        # that actually landed.
        from supervisor.terminal_delivery import replay_pending_deliveries

        replay_pending_deliveries(DATA_DIR)
    except Exception:
        log.debug("Boot replay of pending terminal deliveries failed", exc_info=True)


def _prune_delegated_snapshots() -> None:
    """C1 delegated execution snapshots: GC cross-checked against custody.

    A snapshot stays while its run is open/undisposed OR a pending invocation
    names it; everything else (disposed, closed, refused) is torn down with its
    pinned baseline ref. Fail-soft like every startup prune step — the guard
    lives here so the startup sequence never dies on a GC error.

    FAIL-CLOSED on an unreadable custody log (CR1-1): the keep-set comes from
    replaying the custody rows, and ``_iter_rows`` swallows its own OSError —
    right for the fail-soft readers, but here an unreadable log replays as
    "no open runs", the keep-set goes EMPTY, and the prune destroys every
    live snapshot with the child's only copy of its work. GC may delete only
    over PROVEN settled && patch_disposed; an UNKNOWN custody state skips the
    destructive prune entirely and says so loudly."""
    try:
        from ouroboros import delegate_custody as _delegate_custody
        from ouroboros import subagent_worktrees as _snap_worktrees
        from supervisor.state import append_jsonl

        if _delegate_custody.custody_log_unreadable(DATA_DIR):
            log.warning(
                "Delegated snapshot prune SKIPPED: custody event log exists but "
                "cannot be read, so open snapshots are unknowable (fail-closed)")
            if not append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "delegated_snapshot_prune_skipped",
                "reason": "custody_log_unreadable",
            }):
                # CR2-2: the log is unwritable too — the promised durable row
                # could not land. Escalate loudly; the skip itself already
                # protects the open snapshots, so this stays fail-soft.
                log.error(
                    "Delegated snapshot prune skip could NOT be recorded durably: "
                    "the delegated_snapshot_prune_skipped row was not written "
                    "(custody event log unwritable). Open snapshots remain "
                    "protected by the skip itself.")
            return
        snapshot_report = _snap_worktrees.prune_execution_snapshots(
            _delegate_custody.open_snapshot_ids(DATA_DIR))
        if snapshot_report.get("removed"):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "delegated_snapshot_prune",
                "report": snapshot_report,
            })
    except Exception:
        log.debug("Delegated execution snapshot prune failed", exc_info=True)


def _scoped_task_metadata(project_id: str, task_metadata: Any) -> Any:
    """Bind a chat frame's task_metadata to the thread's project via chat_id (the
    SSOT). A registered project chat scopes to its OWN project, overriding any
    client-supplied project_id; a non-project chat DROPS an untrusted client
    project_id (work is scoped to a project only via the promote_chat_to_task tool,
    never a raw ws frame). Prevents a stale/malformed frame (chat_id A + project_id
    B) from rendering in A while loading/writing project B's memory."""
    if project_id:
        return {**(task_metadata or {}), "project_id": project_id}
    if task_metadata and task_metadata.get("project_id"):
        return {k: v for k, v in task_metadata.items() if k != "project_id"}
    return task_metadata


def _owner_binding_chat_id(ctx: Any, chat_id: int, is_external_transport: bool) -> int:
    """The owner's canonical chat for owner-targeted notices (restart, supervisor
    death, consciousness). External transports bind to their own chat; a WEB owner
    always binds to MAIN (1), never a project panel — so if the first post-reset
    web message lands in a project room, owner notices still reach main."""
    if not is_external_transport and _project_id_for_registered_chat(ctx, chat_id):
        return 1
    try:
        return int(chat_id or 0)
    except (TypeError, ValueError):
        return 0


def _project_id_for_registered_chat(ctx: Any, chat_id: int) -> str:
    """Return the registered project id for a project chat_id, else ``""``.

    NOT an isolation gate (full project awareness, v6.32.0): the one mind notices
    EVERY human message via inject_observation, project rooms included. This just
    classifies a chat as a project thread so the message is scoped to that project
    (task_metadata.project_id) and routed to its panel. This active-only lookup is
    paired with ``_reserved_project_for_chat`` for deleting/tombstoned IDs, so a
    reserved chat cannot be resurrected through ordinary routing.
    """
    try:
        from ouroboros.projects_registry import list_projects

        cid = int(chat_id or 0)
        for project in list_projects(ctx.DRIVE_ROOT):
            try:
                if int(project.get("chat_id") or 0) == cid:
                    return str(project.get("id") or "").strip()
            except (TypeError, ValueError):
                continue
    except Exception:
        log.debug("Project chat_id lookup failed", exc_info=True)
    return ""


def _reserved_project_for_chat(ctx: Any, chat_id: int) -> Dict[str, Any]:
    try:
        from ouroboros.projects_registry import list_reserved_projects

        cid = int(chat_id or 0)
        for project in list_reserved_projects(ctx.DRIVE_ROOT):
            try:
                if int(project.get("chat_id") or 0) == cid:
                    return dict(project)
            except (TypeError, ValueError):
                continue
    except Exception:
        log.debug("Reserved Project chat lookup failed", exc_info=True)
    return {}


def _record_routing_receipt(
    bridge: Any,
    ctx: Any,
    *,
    chat_id: int,
    client_message_id: str,
    action: str,
    target: str = "",
    target_label: str = "",
    status: str,
    persist: bool = True,
    options: Optional[list] = None,
    detail: str = "",
    attachment_manifest: Optional[list] = None,
) -> None:
    """Emit a typed bubble-free ack and optionally persist its presentation state."""
    if target and not str(target_label or "").strip():
        from ouroboros.project_dialogue import routing_target_label

        target_label = routing_target_label(ctx.DRIVE_ROOT, action, target)
    if persist:
        try:
            from ouroboros.project_dialogue import append_chat_annotation

            append_chat_annotation(
                ctx.DRIVE_ROOT,
                client_message_id,
                action=action,
                target=target,
                target_label=target_label,
                status=status,
                detail=detail,
                attachment_manifest=attachment_manifest,
            )
        except Exception:
            log.debug("Routing annotation append failed", exc_info=True)
    try:
        ack = getattr(bridge, "send_routing_ack", None)
        if callable(ack):
            ack_kwargs = {
                "client_message_id": client_message_id,
                "action": action,
                "target": target,
                "target_label": target_label,
                "status": status,
            }
            if options is not None:
                ack_kwargs["options"] = options
            if attachment_manifest is not None:
                ack_kwargs["attachment_manifest"] = attachment_manifest
            ack(
                chat_id,
                **ack_kwargs,
            )
        else:
            broadcast = getattr(bridge, "broadcast", None)
            if callable(broadcast):
                payload = {
                    "type": "message_annotation",
                    "annotation_type": "routing_ack",
                    "chat_id": int(chat_id or 0),
                    "client_message_id": str(client_message_id or ""),
                    "action": action,
                    "target": target,
                    "target_label": target_label,
                    "status": status,
                    "suppress_bubble": True,
                }
                if options is not None:
                    payload["options"] = options
                if attachment_manifest is not None:
                    payload["attachment_manifest"] = attachment_manifest
                broadcast(payload)
    except Exception:
        log.debug("Routing receipt broadcast failed", exc_info=True)


def _route_owner_message(bridge: Any, ctx: Any, incoming: Dict[str, Any]) -> None:
    """Route one non-command owner message through the canonical decision lane."""
    chat_id = int(incoming["chat_id"])
    text = str(incoming.get("text") or "")
    image_caption = str(incoming.get("image_caption") or "")
    client_message_id = str(incoming.get("client_message_id") or "")
    image_data = incoming.get("image_data")
    task_constraint = incoming.get("task_constraint")
    task_metadata = incoming.get("task_metadata")
    from ouroboros.contracts.task_constraint import normalize_task_constraint

    normalized_constraint = normalize_task_constraint(task_constraint)
    if normalized_constraint and normalized_constraint.mode == "skill_repair":
        # Repair is already a typed, narrowly confined task request. Sending it
        # through the conversation decision lane would combine skill_repair with
        # _ephemeral_turn: ephemeral hides the repair mutators while heal mode
        # blocks promotion. Promote it directly without weakening either policy.
        # DELIBERATE: task_metadata (incl. any client_surface fact) is dropped on
        # this branch — a repair task's objective is a fixed UI action and the
        # sending surface adds nothing to it (same treatment as force_plan here).
        from supervisor.events import _handle_promote_chat_to_task

        ctx.consciousness.inject_observation(
            f"Message from my human: {incoming.get('log_text') or ''}"
        )
        task_id = uuid.uuid4().hex[:16]
        event = {
            "type": "promote_chat_to_task",
            "task_id": task_id,
            "routing_token": uuid.uuid4().hex,
            "objective": text or image_caption,
            "chat_id": chat_id,
            "client_message_id": client_message_id,
            "task_constraint": task_constraint,
            "routed_from_main": True,
        }
        origin_ref = incoming.get("origin_message_ref")
        if isinstance(origin_ref, dict) and origin_ref:
            event["source_ref"] = origin_ref
            event["source_text"] = str(incoming.get("log_text") or "")
        else:
            event["origin_suppressed"] = True
        try:
            outcome = _handle_promote_chat_to_task(event, ctx)
        except Exception:
            log.warning("Direct skill-repair promotion failed", exc_info=True)
            outcome = {
                "status": "needs_manual_target",
                "reason": "repair_promotion_failed",
                "task_id": task_id,
            }
        outcome = outcome if isinstance(outcome, dict) else {"status": "scheduled", "task_id": task_id}
        outcome_status = str(outcome.get("status") or "needs_manual_target")
        if outcome_status == "scheduled":
            try:
                ctx.send_with_budget(
                    chat_id,
                    f"✅ Repair task {task_id} was accepted and durably scheduled.",
                )
            except Exception:
                log.debug("Repair promotion success notification failed", exc_info=True)
        else:
            reason = str(outcome.get("reason") or outcome_status)
            try:
                ctx.send_with_budget(
                    chat_id,
                    f"⚠️ Repair task was not started ({reason}). Please retry from the skill card.",
                )
            except Exception:
                log.debug("Repair promotion refusal notification failed", exc_info=True)
        return
    reserved_project = _reserved_project_for_chat(ctx, chat_id)
    project_id = (
        str(reserved_project.get("id") or "")
        if str((reserved_project or {}).get("lifecycle") or "active") == "active"
        else ""
    )
    if reserved_project and not project_id:
        _record_routing_receipt(
            bridge,
            ctx,
            chat_id=chat_id,
            client_message_id=client_message_id,
            action="project_route",
            target=str(reserved_project.get("id") or ""),
            status="project_unavailable",
        )
        return
    ctx.consciousness.inject_observation(f"Message from my human: {incoming.get('log_text') or ''}")
    task_metadata = _scoped_task_metadata(project_id, task_metadata)
    swarm_intent = bool(
        isinstance(task_metadata, dict) and task_metadata.get("force_plan")
    )
    # The turn's origin identity rides UNCONDITIONALLY (not only when the
    # decision lane runs): a bare direct turn with no projects/roots yet — the
    # first-ever project creation — must still carry it so promote/route/bind
    # receive the ref by value.
    origin_ref = incoming.get("origin_message_ref")
    if isinstance(origin_ref, dict) and origin_ref:
        task_metadata = {
            **(task_metadata or {}),
            "origin_message_ref": origin_ref,
            "origin_message_text": str(incoming.get("log_text") or ""),
        }
    else:
        # A suppressed (never-logged) message has a DESIGNED absence of origin;
        # downstream binders must not classify it as a producer bug.
        task_metadata = {**(task_metadata or {}), "origin_suppressed": True}
    # Owner Surface Fact channel fallback: a non-web ingress (telegram/skill
    # transports) carries no browser observables, but its channel IS the
    # surface fact. Host-stamped here, never overwriting a real descriptor;
    # source=="web" stays an honest absence (an old SPA sends no fact), and a
    # synthetic A2A chat (negative id) is machine traffic — no owner sent it,
    # so it must never wear an owner_client fact.
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id as _is_a2a

    _ingress_source = str(incoming.get("source") or "web")
    if (
        _ingress_source != "web"
        and not _is_a2a(chat_id)
        and not isinstance(task_metadata.get("client_surface"), dict)
    ):
        task_metadata = {**task_metadata, "client_surface": {"channel": _ingress_source}}
    if project_id and not swarm_intent:
        routed_to_task = _route_project_chat_to_running_task(
            ctx,
            chat_id,
            text or image_caption,
            client_message_id,
            task_metadata=task_metadata,
            image_data=image_data,
        )
        if routed_to_task:
            _record_routing_receipt(
                bridge,
                ctx,
                chat_id=chat_id,
                client_message_id=client_message_id,
                action="mailbox_delivery",
                target=routed_to_task,
                target_label=(
                    str(task_metadata.get("_routing_target_label") or "")
                    if isinstance(task_metadata, dict) else ""
                ),
                status="delivered",
                detail=(
                    str(task_metadata.get("_attachment_report") or "")
                    if isinstance(task_metadata, dict) else ""
                ),
                attachment_manifest=(
                    list(task_metadata.get("_attachment_manifest") or [])
                    if isinstance(task_metadata, dict) else None
                ),
            )
            return

    global_roots = _addressable_root_tasks(ctx, None)
    try:
        from ouroboros.projects_registry import list_projects

        has_projects = bool(list_projects(ctx.DRIVE_ROOT))
    except Exception:
        log.warning("Unable to inspect Projects for owner routing", exc_info=True)
        has_projects = True
    needs_decision_lane = swarm_intent or bool(project_id) or has_projects or bool(global_roots)
    if needs_decision_lane:
        task_metadata = _decision_turn_metadata(ctx, chat_id, client_message_id, task_metadata)
    agent = ctx.get_chat_agent()

    def _run_direct() -> None:
        try:
            ctx.handle_chat_direct(
                chat_id,
                text or image_caption,
                image_data,
                task_constraint=task_constraint,
                task_metadata=task_metadata,
            )
        finally:
            ctx.consciousness.resume()

    if needs_decision_lane or agent._busy:
        threading.Thread(
            target=ctx.handle_chat_ephemeral,
            args=(chat_id, text or image_caption, image_data),
            kwargs={"task_constraint": task_constraint, "task_metadata": task_metadata},
            daemon=True,
        ).start()
    else:
        ctx.consciousness.pause()
        threading.Thread(target=_run_direct, daemon=True).start()


def _process_bridge_updates(bridge, offset: int, ctx: Any) -> int:
    from supervisor.message_bus import coerce_chat_identity

    updates = bridge.get_updates(offset=offset, timeout=1)
    for upd in updates:
        offset = int(upd["update_id"]) + 1
        msg = upd.get("message") or {}
        if not msg:
            continue

        chat_id = coerce_chat_identity((msg.get("chat") or {}).get("id"), 1)
        user_id = coerce_chat_identity((msg.get("from") or {}).get("id"), chat_id or 1)
        text = str(msg.get("text") or "")
        source = str(msg.get("source") or "web")
        sender_label = str(msg.get("sender_label") or "")
        sender_session_id = str(msg.get("sender_session_id") or "")
        client_message_id = str(msg.get("client_message_id") or "")
        transport = msg.get("transport") if isinstance(msg.get("transport"), dict) else {}
        image_base64 = str(msg.get("image_base64") or "")
        image_mime = str(msg.get("image_mime") or "image/jpeg")
        image_caption = str(msg.get("image_caption") or "")
        suppress_chat_log = bool(msg.get("suppress_chat_log"))
        task_constraint = msg.get("task_constraint") if isinstance(msg.get("task_constraint"), dict) else None
        task_metadata = msg.get("task_metadata") if isinstance(msg.get("task_metadata"), dict) else None
        image_data = (image_base64, image_mime, image_caption) if image_base64 else None
        log_text = text or image_caption or ("(image attached)" if image_base64 else "")
        now_iso = utc_now_iso()
        if not client_message_id:
            # Some owner transports have no client-generated id.  Give the
            # canonical row a deterministic host id before logging/routing so a
            # typed non-bubble acknowledgement cannot be silently dropped and a
            # replay of the same inbound update remains idempotent.
            identity = json.dumps(
                {
                    "source": source,
                    "session": sender_session_id,
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "update_id": int(upd.get("update_id") or 0),
                    "text": text,
                    "caption": image_caption,
                    "transport": transport,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            client_message_id = f"host-{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex}"

        st = ctx.load_state()
        owner_id = st.get("owner_id")
        lowered = text.strip().lower()
        is_slash_command = lowered.startswith("/")
        is_external_transport = source != "web"
        external_identity_present = (not is_external_transport) or (chat_id > 0 and user_id > 0)
        # Global owner = primary chat for outbound notices (web on desktop, the
        # first transport on headless Colab). Bound once, on the first message.
        if owner_id is None and external_identity_present:
            owner_id = user_id

        from supervisor.message_bus import log_chat

        # Origin identity is captured HERE, where the host writes the canonical
        # row (BIBLE P2: identity by value, never re-derived from content
        # downstream). Only a row that is actually logged mints a ref — a
        # suppressed message must not reference a non-existent canonical row.
        origin_message_ref: Optional[Dict[str, Any]] = None
        if not suppress_chat_log:
            log_chat(
                "in",
                chat_id,
                user_id,
                log_text,
                ts=now_iso,
                source=source,
                sender_label=sender_label,
                sender_session_id=sender_session_id,
                client_message_id=client_message_id,
                transport=transport,
                client_surface=(
                    task_metadata.get("client_surface")
                    if isinstance(task_metadata, dict) and isinstance(task_metadata.get("client_surface"), dict)
                    else None
                ),
            )
            from ouroboros.project_dialogue import build_owner_message_ref

            origin_message_ref = build_owner_message_ref(
                chat_id=chat_id,
                client_message_id=client_message_id,
                ts=now_iso,
                text=log_text,
            )
            if source != "web":
                bridge.broadcast({
                    "type": "photo" if image_base64 else "chat",
                    "role": "user",
                    "content": text,
                    "caption": image_caption,
                    "image_base64": image_base64,
                    "mime": image_mime,
                    "ts": now_iso,
                    "source": source,
                    "sender_label": sender_label,
                    "sender_session_id": sender_session_id,
                    "client_message_id": client_message_id,
                    "transport": transport,
                    "chat_id": chat_id,
                })
        def _stamp_owner_activity(live: dict) -> None:
            if live.get("owner_id") is None and external_identity_present:
                live["owner_id"] = user_id
                live["owner_chat_id"] = _owner_binding_chat_id(ctx, chat_id, is_external_transport)
            live["last_owner_message_at"] = now_iso

        ctx.update_state(_stamp_owner_activity)

        if not text and not image_base64:
            continue

        if is_external_transport and is_slash_command:
            if not external_identity_present:
                ctx.send_with_budget(chat_id, "⚠️ Command ignored: this transport did not provide owner identity.")
                continue
            owner_ext_id = st.get("owner_external_id")
            owner_ext_chat_id = st.get("owner_external_chat_id")
            if owner_ext_id is None:
                def _bind_external_owner(live: dict) -> None:
                    if live.get("owner_external_id") is None:
                        live["owner_external_id"] = user_id
                        live["owner_external_chat_id"] = chat_id
                        live["owner_external_bound_at"] = now_iso

                ctx.update_state(_bind_external_owner)
                ctx.send_with_budget(chat_id, "✅ Owner chat registered. Send the command again to execute it.")
                continue
            try:
                owner_ext_id_int = int(owner_ext_id or 0)
                owner_ext_chat_id_int = int(owner_ext_chat_id or 0)
            except (TypeError, ValueError):
                owner_ext_id_int = 0
                owner_ext_chat_id_int = 0
            if owner_ext_id_int != user_id or owner_ext_chat_id_int != chat_id:
                ctx.send_with_budget(chat_id, "⚠️ Command ignored: this transport is not the bound owner chat.")
                continue

        if lowered.startswith("/panic"):
            ctx.send_with_budget(chat_id, "🛑 PANIC: killing everything. App will close.")
            _execute_panic_stop(ctx.consciousness, ctx.kill_workers)
        elif lowered.startswith("/restart"):
            ctx.send_with_budget(chat_id, "♻️ Restarting.")
            ok, restart_msg = _safe_restart_serialized(
                ctx.safe_restart,
                reason="owner_restart",
                unsynced_policy="rescue_and_reset",
            )
            if not ok:
                ctx.send_with_budget(chat_id, f"⚠️ Restart cancelled: {restart_msg}")
                continue
            state_dir = DATA_DIR / "state"
            owner_restart_flag = state_dir / "owner_restart_no_resume.flag"
            stable_skip_flag = state_dir / "panic_stop.flag"
            try:
                state_dir.mkdir(parents=True, exist_ok=True)
                owner_restart_flag.write_text("owner_restart", encoding="utf-8")
                # Pair owner flag with panic_stop for stable-build auto-resume compatibility.
                stable_skip_flag.write_text("owner_restart_no_resume", encoding="utf-8")
            except Exception:
                owner_restart_flag.unlink(missing_ok=True)
                stable_skip_flag.unlink(missing_ok=True)
                log.warning("Failed to write owner restart no-resume flag", exc_info=True)
                ctx.send_with_budget(chat_id, "⚠️ Restart cancelled: could not write restart state.")
                continue
            try:
                ctx.kill_workers(
                    force=True,
                    terminal_status="cancelled",
                    result_reason="Owner restart stopped this task before process restart.",
                    **_managed_update_pending_kwargs(),
                )
            except Exception:
                owner_restart_flag.unlink(missing_ok=True)
                stable_skip_flag.unlink(missing_ok=True)
                log.warning("Restart cancelled because worker shutdown failed", exc_info=True)
                try:
                    ctx.send_with_budget(chat_id, "⚠️ Restart cancelled: failed to stop workers.")
                except Exception:
                    pass
                continue
            try:
                ctx.send_with_budget(chat_id, "Stopping active task. New settings apply to the next message.")
            except Exception:
                log.warning("Failed to send owner restart stop notice; continuing restart", exc_info=True)
            _request_restart_exit(owner=True)
        elif lowered == "/review" or lowered.startswith("/review "):
            # Target the requesting chat so the ack and results return to the
            # external transport owner, not the default web owner_chat_id.
            ctx.queue_deep_self_review_task(reason="owner:/review", force=True, chat_id=chat_id)
        elif lowered.startswith("/evolve"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "on"
            turn_on = action not in ("off", "stop", "0")
            objective = ""
            if turn_on and len(parts) > 2:
                objective = text.split(None, 2)[2].strip()
            if turn_on:
                from supervisor.evolution_lifecycle import evolution_block_reason, start_evolution_campaign
                from supervisor.state import update_state as _evo_update_state

                block = evolution_block_reason()
                if block:
                    ctx.send_with_budget(chat_id, block)
                    continue
                # GR4-6: clear the durable owner-stop flag BEFORE the campaign is
                # minted — the old order (campaign first, flag cleared in the later
                # save_state below) left a window where the owner-stop backstop,
                # fired by an old evolution task settling, read flag=True +
                # campaign=active and closed the FRESH campaign. Owner-authorized
                # clear (the owner is explicitly starting evolution). GR5-1: the
                # prior value is captured FIRST so a failed start can restore it.
                _prior_owner_stop = bool(ctx.load_state().get("evolution_owner_stopped"))
                _evo_update_state(lambda live: live.__setitem__("evolution_owner_stopped", False))
                try:
                    if not start_evolution_campaign(objective, source="owner_chat"):
                        raise RuntimeError("campaign write was refused")
                except Exception:
                    log.warning("Failed to start evolution campaign", exc_info=True)
                    # GR5-1: the start FAILED, so the pre-mint clear was not an
                    # owner-authorized state change after all. Restore the CAPTURED
                    # prior value — leaving it cleared would let the post-task
                    # promotion pipeline (apply_pending_request reads the flag)
                    # autonomously re-arm evolution the owner believes is off, and
                    # an unconditional True would invent a stop that never happened.
                    _evo_update_state(lambda live, _v=_prior_owner_stop: live.__setitem__(
                        "evolution_owner_stopped", _v))
                    ctx.send_with_budget(chat_id, "⚠️ Evolution stayed OFF: campaign state could not be created.")
                    continue
            st2 = ctx.load_state()
            st2["evolution_mode_enabled"] = bool(turn_on)
            if turn_on:
                st2["evolution_consecutive_failures"] = 0
            # Owner stop is AUTHORITATIVE against the post-task promotion pipeline: the
            # durable evolution_owner_stopped flag (read by apply_pending_request) blocks an
            # autonomous re-arm until the owner /evolve starts again. Set True on stop,
            # cleared (False) on turn_on — the only owner-authorized clear.
            st2["evolution_owner_stopped"] = (not turn_on)
            # Owner-initiated evolution must not inherit a stale post-task one-shot
            # autostop, which would disable the owner's campaign after one cycle.
            st2["post_task_autostop"] = False
            ctx.save_state(st2)
            ctx.send_with_budget(
                chat_id,
                f"🧬 Evolution campaign: {'ON' if turn_on else _owner_evolution_stop(ctx, chat_id)}",
            )
        elif lowered.startswith("/bg"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "status"
            if action in ("start", "on", "1"):
                result = ctx.consciousness.start()
                _bg_s = ctx.load_state()
                _bg_s["bg_consciousness_enabled"] = True
                ctx.save_state(_bg_s)
                ctx.send_with_budget(chat_id, f"🧠 {result}")
            elif action in ("stop", "off", "0"):
                result = ctx.consciousness.stop()
                _bg_s = ctx.load_state()
                _bg_s["bg_consciousness_enabled"] = False
                ctx.save_state(_bg_s)
                ctx.send_with_budget(chat_id, f"🧠 {result}")
            else:
                bg_status = "running" if ctx.consciousness.is_running else "stopped"
                ctx.send_with_budget(chat_id, f"🧠 Background consciousness: {bg_status}")
        elif lowered.startswith("/status"):
            from supervisor.state import status_text
            from supervisor.queue import SOFT_TIMEOUT_SEC, HARD_TIMEOUT_SEC

            status = status_text(ctx.WORKERS, ctx.PENDING, ctx.RUNNING, SOFT_TIMEOUT_SEC, HARD_TIMEOUT_SEC)
            ctx.send_with_budget(chat_id, status)
        else:
            _route_owner_message(
                bridge,
                ctx,
                {
                    "chat_id": chat_id,
                    "text": text,
                    "image_caption": image_caption,
                    "client_message_id": client_message_id,
                    "image_data": image_data,
                    "task_constraint": task_constraint,
                    "task_metadata": task_metadata,
                    "log_text": log_text,
                    "origin_message_ref": origin_message_ref,
                    "source": source,
                },
            )
    return offset


def _runtime_branch_defaults() -> tuple[str, str]:
    branch_dev = "ouroboros"
    branch_stable = "ouroboros-stable"
    if not _LAUNCHER_MANAGED:
        return branch_dev, branch_stable
    try:
        from supervisor import git_ops as git_ops_module
        if hasattr(git_ops_module, "managed_branch_defaults"):
            return git_ops_module.managed_branch_defaults(REPO_DIR)
    except Exception:
        pass
    return branch_dev, branch_stable


def _bootstrap_supervisor_repo(settings: dict, git_ops_module=None):
    if git_ops_module is None:
        from supervisor import git_ops as git_ops_module

    branch_dev, branch_stable = _runtime_branch_defaults()

    git_ops_module.init(
        repo_dir=REPO_DIR,
        drive_root=DATA_DIR,
        remote_url="",
        branch_dev=branch_dev,
        branch_stable=branch_stable,
    )
    git_ops_module.ensure_repo_present()
    setup_remote_if_configured(settings, log)

    if _LAUNCHER_MANAGED:
        # An in-flight managed-update assisted merge intentionally leaves MERGE_HEAD + the partly
        # resolved merge in the live worktree (over pre_update_sha). Use the NON-destructive
        # rescue_and_block policy so the bootstrap restart does not reset/clean that merge state
        # away before finalize_managed_update_on_boot / _recover_assisted_on_boot can resume it.
        try:
            from supervisor.update_merge import active_update_tx

            _managed_update_active = bool(active_update_tx())
        except Exception:
            _managed_update_active = False
        block = _has_active_evolution_transaction() or _managed_update_active
        policy = "rescue_and_block" if block else "rescue_and_reset"
        ok, msg = _safe_restart_serialized(
            git_ops_module.safe_restart,
            reason="bootstrap",
            unsynced_policy=policy,
        )
        if not ok and policy == "rescue_and_block":
            try:
                from supervisor.evolution_lifecycle import pause_evolution_campaign
                from supervisor.state import load_state, save_state

                st = load_state()
                st["evolution_mode_enabled"] = False
                save_state(st)
                pause_evolution_campaign(f"bootstrap blocked to protect active evolution transaction: {msg}")
            except Exception:
                log.debug("Failed to pause evolution after blocked bootstrap", exc_info=True)
        return ok, msg

    log.info("Local-dev server start detected — skipping bootstrap git reset.")
    deps_ok, deps_msg = git_ops_module.sync_runtime_dependencies(reason="bootstrap_local_dev")
    if not deps_ok:
        return False, f"Failed local-dev deps sync: {deps_msg}"

    import_result = git_ops_module.import_test()
    if import_result.get("ok"):
        return True, "OK: local-dev bootstrap"
    return False, f"Local-dev import test failed (rc={import_result.get('returncode', -1)})"


def _periodic_zombie_reconcile() -> None:
    """Heal zombie 'running' records on a supervisor cadence.

    A worker that died mid-review (crash / SIGKILL / manual stop) leaves
    ``review_job.json`` at status=running forever in headless/no-UI runs, where
    the boot and ``GET /api/extensions`` reconciles never fire; the same death
    leaves ``task_results/<id>.json`` at running. Both reconciles are
    liveness-gated (pid-dead / queue-empty + worker-boot evidence), so a live
    review or task is never touched.
    """
    try:
        from ouroboros.skill_review_runner import reconcile_stale_review_jobs
        reconcile_stale_review_jobs(DATA_DIR)
    except Exception:
        log.debug("Periodic skill review-job reconcile failed", exc_info=True)
    try:
        from ouroboros.task_status import reconcile_orphaned_running_tasks
        reconcile_orphaned_running_tasks(DATA_DIR)
    except Exception:
        log.debug("Periodic orphaned running-task reconcile failed", exc_info=True)
    try:
        from ouroboros.projects_registry import reconcile_projects
        reconcile_projects(DATA_DIR)
    except Exception:
        log.debug("Project registry reconcile failed", exc_info=True)
    _resume_interrupted_project_deletions()


def _resume_interrupted_project_deletions() -> None:
    try:
        from supervisor.task_lifecycle import resume_project_deletions

        resume_project_deletions(DATA_DIR)
    except Exception:
        log.debug("Project deletion recovery failed", exc_info=True)


def _startup_worktree_prune() -> None:
    """Startup hygiene: prune orphaned subagent worktrees (after the custody sweep)."""
    from supervisor.state import append_jsonl

    try:
        from ouroboros import subagent_worktrees

        worktree_report = subagent_worktrees.prune_orphans()
        if worktree_report.get("removed"):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "subagent_worktree_prune",
                "report": worktree_report,
            })
    except Exception:
        log.debug("Subagent worktree prune failed", exc_info=True)


def _startup_prune_sweeps() -> None:
    """Startup hygiene: prune stale task drives/trees and orphaned temp files."""
    from supervisor.state import append_jsonl

    try:
        from ouroboros.headless import prune_headless_task_drives, prune_task_drives, prune_task_trees
        from ouroboros.utils import sweep_stale_temp_files

        prune_report = prune_headless_task_drives(DATA_DIR)
        task_drive_report = prune_task_drives(DATA_DIR)
        # Ephemeral task-tree coordination ledgers age out with their terminal root.
        prune_task_trees(DATA_DIR)
        # Reap orphaned atomic-write temp files (.*.tmp.*) left by a hard kill.
        sweep_stale_temp_files(DATA_DIR)
        if (
            prune_report.get("pruned")
            or prune_report.get("errors")
            or task_drive_report.get("pruned")
            or task_drive_report.get("errors")
        ):
            append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "headless_task_drive_prune",
                "report": prune_report,
                "task_drives": task_drive_report,
            })
    except Exception:
        log.debug("Headless task drive prune failed", exc_info=True)


def _run_supervisor(settings: dict) -> None:
    """Initialize and run the supervisor loop. Called in a background thread."""
    global _supervisor_error, _supervisor_thread, _consciousness

    _apply_settings_to_env(settings)

    # Revival must drop the prior consciousness and cached event-queue binding.
    if _consciousness is not None:
        try:
            _consciousness.stop()
        except Exception:
            log.debug("Failed to stop previous consciousness instance", exc_info=True)
        _consciousness = None
    try:
        from supervisor import workers as _workers_mod

        _workers_mod._chat_agent = None
    except Exception:
        log.debug("Failed to reset cached chat agent", exc_info=True)

    try:
        ensure_legacy_imported(pathlib.Path(DATA_DIR))

        from supervisor.message_bus import init as bus_init
        from supervisor.message_bus import LocalChatBridge

        bridge = LocalChatBridge(settings)
        bridge._broadcast_fn = broadcast_ws_sync

        from ouroboros.utils import set_log_sink
        from supervisor.events import make_server_log_sink

        set_log_sink(make_server_log_sink(bridge, pathlib.Path(DATA_DIR)))

        bus_init(
            drive_root=DATA_DIR,
            total_budget_limit=float(settings.get("TOTAL_BUDGET", SETTINGS_DEFAULTS["TOTAL_BUDGET"])),
            budget_report_every=10,
            chat_bridge=bridge,
        )

        from supervisor.state import init as state_init, init_state, load_state, save_state, update_state
        from supervisor.state import append_jsonl, update_budget_from_usage, rotate_chat_log_if_needed, rotate_jsonl_log_if_needed
        state_init(DATA_DIR, float(settings.get("TOTAL_BUDGET", SETTINGS_DEFAULTS["TOTAL_BUDGET"])))
        init_state()

        from supervisor.git_ops import safe_restart
        ok, msg = _bootstrap_supervisor_repo(settings)
        if not ok:
            log.error("Supervisor bootstrap failed: %s", msg)

        from supervisor.queue import (
            enqueue_task, enforce_task_timeouts, enqueue_evolution_task_if_needed,
            persist_queue_snapshot, restore_pending_from_snapshot,
            cancel_task_by_id, queue_deep_self_review_task, sort_pending,
        )
        from supervisor.workers import (
            init as workers_init, get_event_q, WORKERS, PENDING, RUNNING,
            spawn_workers, kill_workers, assign_tasks, ensure_workers_healthy,
            handle_chat_direct, handle_chat_ephemeral, _get_chat_agent, auto_resume_after_restart,
        )

        max_workers = int(settings.get("OUROBOROS_MAX_WORKERS", 10))
        soft_timeout = int(settings.get("OUROBOROS_SOFT_TIMEOUT_SEC", 600))
        hard_timeout = int(settings.get("OUROBOROS_HARD_TIMEOUT_SEC", 1800))

        # Managed manifest branch defaults must drive worker commit/restart flows too.
        _workers_branch_dev, _workers_branch_stable = _runtime_branch_defaults()
        workers_init(
            repo_dir=REPO_DIR, drive_root=DATA_DIR, max_workers=max_workers,
            soft_timeout=soft_timeout, hard_timeout=hard_timeout,
            total_budget_limit=float(settings.get("TOTAL_BUDGET", SETTINGS_DEFAULTS["TOTAL_BUDGET"])),
            branch_dev=_workers_branch_dev, branch_stable=_workers_branch_stable,
        )

        from supervisor.events import dispatch_event
        from supervisor.message_bus import send_with_budget
        from ouroboros.consciousness import BackgroundConsciousness
        import types
        import queue as _queue_mod

        restored_pending = restore_pending_from_snapshot()
        kill_workers(preserve_pending=True)
        spawn_workers(max_workers)
        persist_queue_snapshot(reason="startup")
        try:
            from ouroboros.delegate_recovery import pre_adopt_planned_handoffs

            pre_adopt_planned_handoffs(DATA_DIR, list(PENDING))
        except Exception:
            log.debug("Planned delegate pre-adoption failed", exc_info=True)
        _resume_interrupted_project_deletions()
        # Original startup order preserved: drive prunes, custody sweep (reap
        # orphaned processes), THEN worktree prune.
        _startup_prune_sweeps()
        _startup_custody_sweep()
        _startup_worktree_prune()

        _prune_delegated_snapshots()

        try:
            from ouroboros.observability import prune_observability_blobs
            from ouroboros.tools.services import prune_service_logs

            observability_report = prune_observability_blobs(DATA_DIR)
            service_report = prune_service_logs(DATA_DIR)
            if (
                observability_report.get("enabled")
                or observability_report.get("manifest_count")
                or observability_report.get("blob_count")
                or observability_report.get("deleted_manifests")
                or observability_report.get("deleted_blobs")
                or observability_report.get("errors")
                or service_report.get("deleted_dirs")
                or service_report.get("deleted_files")
                or service_report.get("errors")
            ):
                append_jsonl(DATA_DIR / "logs" / "events.jsonl", {
                    "ts": utc_now_iso(),
                    "type": "runtime_artifact_prune",
                    "observability": observability_report,
                    "services": service_report,
                })
        except Exception:
            log.debug("Runtime artifact prune failed", exc_info=True)

        if restored_pending > 0:
            st_boot = load_state()
            if st_boot.get("owner_chat_id"):
                send_with_budget(int(st_boot["owner_chat_id"]),
                    f"♻️ Restored pending queue from snapshot: {restored_pending} tasks.")

        auto_resume_after_restart()

        def _get_owner_chat_id() -> Optional[int]:
            try:
                st = load_state()
                cid = st.get("owner_chat_id")
                return int(cid) if cid else None
            except Exception:
                return None

        _consciousness = BackgroundConsciousness(
            drive_root=DATA_DIR, repo_dir=REPO_DIR,
            event_queue=get_event_q(), owner_chat_id_fn=_get_owner_chat_id,
        )

        _bg_st = load_state()
        if _bg_st.get("bg_consciousness_enabled"):
            _consciousness.start()
            log.info("Background consciousness auto-restored from saved state.")

        branch_dev, branch_stable = _runtime_branch_defaults()
        _event_ctx = types.SimpleNamespace(
            DRIVE_ROOT=DATA_DIR, REPO_DIR=REPO_DIR,
            BRANCH_DEV=branch_dev, BRANCH_STABLE=branch_stable,
            bridge=bridge, WORKERS=WORKERS, PENDING=PENDING, RUNNING=RUNNING,
            MAX_WORKERS=max_workers,
            send_with_budget=send_with_budget, load_state=load_state, save_state=save_state,
            update_state=update_state,
            update_budget_from_usage=update_budget_from_usage, append_jsonl=append_jsonl,
            enqueue_task=enqueue_task, cancel_task_by_id=cancel_task_by_id,
            queue_deep_self_review_task=queue_deep_self_review_task, persist_queue_snapshot=persist_queue_snapshot,
            safe_restart=safe_restart, kill_workers=kill_workers, spawn_workers=spawn_workers,
            sort_pending=sort_pending, consciousness=_consciousness,
            soft_timeout=soft_timeout, hard_timeout=hard_timeout,
            get_chat_agent=_get_chat_agent, handle_chat_direct=handle_chat_direct,
            handle_chat_ephemeral=handle_chat_ephemeral, request_restart=_request_restart_exit,
        )
    except Exception as exc:
        _supervisor_error = f"Supervisor init failed: {exc}"
        _consciousness = None
        log.critical("Supervisor initialization failed", exc_info=True)
        _supervisor_ready.set()
        _supervisor_thread = None
        return

    _supervisor_ready.set()
    log.info("Supervisor ready.")

    offset = 0
    crash_count = 0
    _last_custody_reap = [time.time()]
    _last_review_job_reconcile = [time.time()]
    # WS3: a dedicated watchdog thread (outside this loop, so it fires even if the
    # loop stalls) surfaces a wedge as an observable signal + owner alert instead
    # of silent hours; the loop publishes a liveness tick each iteration. The tick
    # is MONOTONIC: it is only ever read as an elapsed gap, so a wall-clock jump
    # must not turn a healthy loop into a phantom stall (nor hide a real one).
    _loop_liveness = [time.monotonic()]
    _watchdog_stop = threading.Event()  # per-generation: stops the watchdog when THIS loop exits
    _start_supervisor_liveness_watchdog(_loop_liveness, _watchdog_stop)
    while not _restart_requested.is_set():
        try:
            _loop_liveness[0] = time.monotonic()
            rotate_chat_log_if_needed(DATA_DIR)
            # progress.jsonl rotates on the same supervisor tick (v6.90.x P2); its
            # readers (history backfill, SSE replay, api_logs_tail, TB ATIF) are
            # archive-chain-aware.
            rotate_jsonl_log_if_needed(DATA_DIR, "progress.jsonl", "progress")
            ensure_workers_healthy()

            event_q = get_event_q()
            while True:
                try:
                    evt = event_q.get_nowait()
                except _queue_mod.Empty:
                    break
                except (BrokenPipeError, EOFError, FileNotFoundError) as _bus_exc:
                    # The SyncManager backing the event bus died (OOM / stray
                    # kill / corrupted connection). Crashing the loop here took
                    # down the whole 64-lane campaign (CyberGym r11/r12/r13,
                    # 2026-09-04/05). Rebuild the bus in place and keep going:
                    # queued events are re-derivable (task_done is re-emitted
                    # by the worker's own terminal path; heartbeats/checkpoints
                    # are periodic), so the loss is bounded to the dead bus's
                    # in-flight backlog.
                    from supervisor.workers import revive_event_q_if_dead

                    revived = revive_event_q_if_dead()
                    if revived is None:
                        raise  # not a dead-manager shape — surface it
                    log.error(
                        "Supervisor event bus rebuilt after manager death (%s); "
                        "dropped the dead bus's in-flight backlog",
                        _bus_exc,
                    )
                    event_q = revived
                    break
                if evt.get("type") == "restart_request":
                    _handle_restart_in_supervisor(evt, _event_ctx)
                    continue
                dispatch_event(evt, _event_ctx)

            if _restart_requested.is_set():
                break

            # WS3: intake new bridge messages EARLY — before the heavy steps
            # (enforce_task_timeouts / assign_tasks / evolution) — so a later
            # blocking step can never starve new-message intake (the wedge class
            # where no task_received fired for hours until a full restart).
            offset = _process_bridge_updates(bridge, offset, _event_ctx)

            enforce_task_timeouts()
            try:
                from supervisor.queue import check_scheduled_tasks
                check_scheduled_tasks()
            except Exception:
                log.warning("Scheduled task check failed", exc_info=True)
            _periodic_supervisor_maintenance(_last_custody_reap, _last_review_job_reconcile)
            # Loop-tick restart drain (no sleep, events keep flowing): while
            # draining a deferred restart, skip starting new work the restart
            # deadline would immediately chop (evolution / pending project tasks).
            if not _check_pending_restart_drain(_event_ctx):
                try:
                    from ouroboros.post_task_evolution import apply_pending_request
                    from supervisor import state as _pte_state

                    apply_pending_request(_pte_state.DRIVE_ROOT)
                except Exception:
                    log.debug("Post-task evolution apply failed", exc_info=True)
                enqueue_evolution_task_if_needed()
                assign_tasks()
            if _restart_requested.is_set():
                break  # restart just triggered (drain done) — exit without assigning new work (bridge intake already ran early this iteration)
            persist_queue_snapshot(reason="main_loop")

            crash_count = 0
            time.sleep(0.5)

        except Exception as exc:
            crash_count += 1
            log.error("Supervisor loop crash #%d: %s", crash_count, exc, exc_info=True)
            if crash_count >= 3:
                # Visible death: previously the loop returned with
                # _supervisor_ready still set and no _supervisor_error, so
                # tasks silently stopped being assigned with a healthy-looking
                # /api/state. Record the failure and tell the owner.
                _supervisor_error = f"Supervisor loop died after 3 consecutive crashes: {exc}"
                _supervisor_ready.clear()
                log.critical("Supervisor exceeded max retries: %s", _supervisor_error)
                try:
                    st = load_state()
                    if st.get("owner_chat_id"):
                        send_with_budget(
                            int(st["owner_chat_id"]),
                            "🛑 Supervisor loop died after repeated crashes; tasks are no "
                            "longer being assigned. Saving settings or restarting the app "
                            f"will revive it. Last error: {exc}",
                        )
                except Exception:
                    log.debug("Failed to notify owner about supervisor death", exc_info=True)
                _watchdog_stop.set()  # this generation is dead — stop its liveness watchdog
                return
            time.sleep(min(30, 2 ** crash_count))
    _watchdog_stop.set()  # loop exited (restart) — stop this generation's watchdog
    _supervisor_thread = None


# Deferred restart-drain state (multi-project, v6.32.0). The drain MUST NOT
# sleep on the supervisor loop thread (it is the only thread that processes
# heartbeats / task_done and shrinks RUNNING). Instead a restart with live
# tasks is recorded here and re-checked every loop tick, so events keep
# flowing and the drain actually observes tasks finishing.
_pending_restart: Dict[str, Any] = {}


def _live_running_task_ids(ctx: Any) -> list:
    """RUNNING task ids with a fresh heartbeat — structured facts only.

    Heartbeat staleness belongs to the generic supervisor queue, not to the
    planning-scout wait policy.  The latter intentionally waits until terminal
    state or its shared cutoff even when a scout heartbeat is stale.
    """
    from supervisor.queue import HEARTBEAT_STALE_SEC

    now = time.time()
    live = []
    for tid, meta in dict(ctx.RUNNING or {}).items():
        if not isinstance(meta, dict):
            continue
        try:
            hb = float(meta.get("last_heartbeat_at") or 0.0)
        except (TypeError, ValueError):
            hb = 0.0
        if hb and (now - hb) < HEARTBEAT_STALE_SEC:
            live.append(str(tid))
    return live


def _handle_restart_in_supervisor(evt: Dict[str, Any], ctx: Any) -> None:
    """Handle agent restart request: drain live tasks across loop ticks, then
    graceful shutdown + exit(42). Never sleeps on the dispatch thread."""
    st = ctx.load_state()
    if st.get("owner_chat_id"):
        ctx.send_with_budget(
            int(st["owner_chat_id"]),
            f"♻️ Restart requested by agent: {evt.get('reason')}",
        )
    from ouroboros.config import get_restart_drain_max_sec

    max_wait = get_restart_drain_max_sec()
    live = _live_running_task_ids(ctx) if max_wait > 0 else []
    if live:
        # Defer: re-checked each tick by _check_pending_restart_drain so the
        # loop keeps draining events (heartbeats advance, RUNNING shrinks).
        _pending_restart.clear()
        _pending_restart.update({
            "reason": str(evt.get("reason") or "agent_restart_request"),
            "deadline": time.time() + min(max_wait, 1800),
            "evolution_restart": bool(evt.get("evolution_restart")),
        })
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                f"⏳ Restart drain: waiting up to {max_wait}s for running task(s) "
                f"{', '.join(sorted(live))} to finish.",
            )
        return
    _perform_supervisor_restart(
        ctx, restart_reason=str(evt.get("reason") or "agent_restart_request"),
        evolution_restart=bool(evt.get("evolution_restart")),
    )


def _check_pending_restart_drain(ctx: Any) -> bool:
    """Loop-tick hook: complete a deferred restart once tasks drain or the
    deadline passes (proceeds fail-closed). Returns True while STILL draining, so
    the loop can skip starting new work that the restart would immediately chop."""
    if not _pending_restart:
        return False
    live = _live_running_task_ids(ctx)
    if live and time.time() < float(_pending_restart.get("deadline") or 0.0):
        return True  # keep draining — events still flow each tick
    pending = dict(_pending_restart)
    _pending_restart.clear()
    _perform_supervisor_restart(
        ctx, restart_reason=str(pending.get("reason") or "agent_restart_request"),
        evolution_restart=bool(pending.get("evolution_restart")),
    )
    # Still "quiescing" this tick: _perform_supervisor_restart sets up the exit
    # (or fail-closed pauses) and returns to the loop — the process exits on the
    # next `while not _restart_requested` check. Returning True keeps the caller
    # from starting new enqueue/assign work on this final pre-exit tick.
    return True


def _perform_supervisor_restart(
    ctx: Any, *, restart_reason: str = "agent_restart_request",
    evolution_restart: bool = False,
) -> None:
    """Graceful shutdown + exit(42) (the post-drain tail; never sleeps)."""
    st = ctx.load_state()
    marker = read_json_dict(
        pathlib.Path(ctx.DRIVE_ROOT) / "state" / "pending_restart_verify.json"
    ) or {}
    claim = (
        marker.get("evolution_claim")
        if evolution_restart and marker.get("reason") == restart_reason
        else {}
    )
    claim = claim if isinstance(claim, dict) else {}
    if evolution_restart and not claim:
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                "🧬 Restart cancelled: the exact evolution restart receipt is missing.",
            )
        return
    if claim:
        from supervisor.evolution_lifecycle import check_evolution_authority

        authority = check_evolution_authority(
            str(claim.get("campaign_id") or ""),
            str(claim.get("transaction_id") or ""),
            str(claim.get("task_id") or ""),
            commit_sha=str(claim.get("commit_sha") or ""),
        )
        if not authority.get("ok"):
            if st.get("owner_chat_id"):
                ctx.send_with_budget(
                    int(st["owner_chat_id"]),
                    "🧬 Restart cancelled: evolution authority changed "
                    f"({authority.get('reason') or 'unknown'}).",
                )
            return
        expected_sha = str(claim.get("commit_sha") or "")
        try:
            head_proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ctx.REPO_DIR),
                check=False,
                capture_output=True,
                text=True,
            )
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(ctx.REPO_DIR),
                check=False,
                capture_output=True,
                text=True,
            )
            head = head_proc.stdout.strip() if head_proc.returncode == 0 else ""
            clean = status_proc.returncode == 0 and not status_proc.stdout.strip()
        except Exception:
            head = ""
            clean = False
        if not expected_sha or head != expected_sha or not clean:
            if st.get("owner_chat_id"):
                ctx.send_with_budget(
                    int(st["owner_chat_id"]),
                    "🧬 Restart cancelled: the live checkout no longer matches "
                    "the exact reviewed evolution commit.",
                )
            return
    ok, msg = _safe_restart_serialized(
        ctx.safe_restart,
        reason="agent_restart_request",
        unsynced_policy="rescue_and_block",
    )
    if not ok:
        try:
            from supervisor.evolution_lifecycle import pause_evolution_campaign

            st["evolution_mode_enabled"] = False
            ctx.save_state(st)
            pause_evolution_campaign(f"agent restart blocked to protect local changes: {msg}")
        except Exception:
            log.debug("Failed to pause evolution after blocked agent restart", exc_info=True)
        if st.get("owner_chat_id"):
            ctx.send_with_budget(int(st["owner_chat_id"]), f"⚠️ Restart skipped: {msg}")
        return
    cleanup_status, cleanup_reason = _shutdown_task_cleanup_args(restart_requested=True)
    global _planned_delegate_restart_transaction_id
    _planned_delegate_restart_transaction_id = ""
    planned_handoffs: set[str] = set()
    restart_transaction_id = uuid.uuid4().hex
    try:
        from ouroboros.delegate_recovery import prepare_planned_restart_handoffs

        planned_handoffs = prepare_planned_restart_handoffs(
            ctx.DRIVE_ROOT, ctx.RUNNING,
            restart_transaction_id=restart_transaction_id,
        )
    except Exception:
        log.debug("Planned self-restart delegate handoff preparation failed", exc_info=True)
    restart_kill_kwargs = _managed_update_pending_kwargs()
    if planned_handoffs:
        _planned_delegate_restart_transaction_id = restart_transaction_id
        restart_kill_kwargs["preserve_pending"] = True
    ctx.kill_workers(
        force=True,
        terminal_status=cleanup_status,
        result_reason=cleanup_reason,
        preserve_running_task_ids=planned_handoffs,
        **restart_kill_kwargs,
    )
    st2 = ctx.load_state()
    st2["session_id"] = uuid.uuid4().hex
    ctx.save_state(st2)
    ctx.persist_queue_snapshot(reason="pre_restart_exit")
    _request_restart_exit()


def _request_restart_exit(owner: bool = False) -> None:
    """Signal server shutdown with restart exit code.

    ``owner`` is the ONE fact the re-exec needs: an owner-initiated restart
    re-reads the runtime mode from settings, an agent- or supervisor-initiated
    one keeps inheriting the boot pin (see server_control.restart_current_process).
    """
    if owner:
        _owner_restart_requested.set()
    _restart_requested.set()


def _managed_update_pending_kwargs() -> dict:
    """Preserve queued work while a durable tx or its pre-tx quiesce owns restart."""
    try:
        from ouroboros.delegate_recovery import has_planned_restart_handoffs

        if (
            has_planned_restart_handoffs(DATA_DIR)
            and _restart_requested.is_set()
            and not _owner_restart_requested.is_set()
        ):
            return {"preserve_pending": True}
        from supervisor.update_merge import active_update_tx

        if active_update_tx():
            return {"preserve_pending": True}
        from supervisor.workers import repo_writer_admission_closed, worker_pool_admission_state

        gate = repo_writer_admission_closed()
        disabled = str(worker_pool_admission_state().get("disabled_reason") or "")
        if gate.startswith("managed_update:") or disabled == "managed_update":
            return {"preserve_pending": True}
        return {}
    except Exception:
        return {"preserve_pending": True}


def _safe_restart_serialized(safe_restart_fn, *, reason: str, unsynced_policy: str):
    """Serialize checkout/reset with update apply; only a landed update may restart."""
    from supervisor import git_ops
    from supervisor.update_merge import (
        acquire_update_lock,
        read_update_tx_strict,
        release_update_lock,
    )

    try:
        lock_fh = acquire_update_lock()
    except RuntimeError:
        return False, "Managed update is changing the checkout; restart was deferred."
    try:
        status, tx = read_update_tx_strict()
        if status == "corrupt":
            return False, "Managed update state is unreadable; restart was deferred."
        if status == "absent" and not git_ops._clear_update_intent():
            return False, (
                "An update intent marker with no update transaction could not be removed; "
                "restart was deferred rather than applying an orphaned update."
            )
        allowed_phases = {"pending_boot_smoke", "applying_replace"}
        if status == "valid" and str(tx.get("phase") or "") not in allowed_phases:
            return False, "Managed update merge is still being resolved; restart was deferred."
        return safe_restart_fn(reason=reason, unsynced_policy=unsynced_policy)
    finally:
        release_update_lock(lock_fh)


def _wait_for_supervisor_update_finalize() -> bool:
    """Wait for a real init outcome; slow dependency sync is not a failed boot."""
    _supervisor_ready.wait()
    return not bool(_supervisor_error)


def _boot_managed_update_tasks() -> None:
    """Finalize a pending update, restart after rollback, then refresh its feed."""
    try:
        from supervisor.git_ops import compute_managed_update_status
        from supervisor.update_merge import finalize_managed_update_on_boot

        result = finalize_managed_update_on_boot(
            supervisor_ready=_wait_for_supervisor_update_finalize()
        )
        stash_note = str(result.get("stash_note") or "")
        if stash_note:
            # Q1=C disclosure contract: a stash restore that conflicted keeps the
            # entry and the OWNER must see the exact recovery command, not only
            # the supervisor log.
            try:
                from supervisor.message_bus import send_with_budget
                from supervisor.state import load_state as _load_state

                owner_chat = int((_load_state() or {}).get("owner_chat_id") or 0)
                if owner_chat:
                    send_with_budget(owner_chat, f"📦 Managed update: {stash_note}")
            except Exception:
                log.debug("stash note owner notification failed", exc_info=True)
        if result.get("rolled_back") is True:
            # This generation imported the rejected candidate. Preserve queued roots
            # through shutdown, then exec the restored code instead of limping on.
            from supervisor.workers import close_repo_writer_admission

            close_repo_writer_admission("managed_update:rollback_restart")
            _request_restart_exit()
            return
        update_status = compute_managed_update_status(fetch=True)
        broadcast_ws_sync({
            "type": "update_status_ready",
            "available": bool(update_status.get("available")),
            "check_ok": update_status.get("check_ok"),
        })
    except Exception:
        log.debug("boot managed-update tasks failed", exc_info=True)


def _shutdown_task_cleanup_args(restart_requested: bool) -> tuple[str, str]:
    """Return ``(terminal_status, result_reason)`` for tasks torn down by a
    graceful server shutdown.

    A graceful shutdown — a requested restart (exit 42) or an external
    stop/restart signal (SIGTERM/SIGINT) — is not a worker crash storm, so a
    still-running task is finalized as ``cancelled`` with an honest reason
    instead of the default crash-storm text the supervisor uses for real
    worker deaths.
    """
    if restart_requested:
        reason = (
            "Server restarted before this task finished; the task was "
            "interrupted by the restart, not a worker crash."
        )
    else:
        reason = (
            "Server shut down (external stop/restart signal) before this task "
            "finished; the task was interrupted, not a worker crash."
        )
    return "cancelled", reason


def _shutdown_supervisor_event_bus() -> None:
    try:
        from supervisor.workers import shutdown_event_q

        shutdown_event_q()
    except Exception:
        pass


def _drain_task_done_finalizations(timeout_sec: float = 120.0) -> bool:
    """Preserve queued copy-back/promotions before shutdown removes workers."""

    try:
        from supervisor.events import drain_task_done_finalizations

        return drain_task_done_finalizations(timeout_sec=timeout_sec)
    except Exception:
        log.warning("Deferred task finalization drain failed", exc_info=True)
        return False


def _execute_panic_stop(consciousness, kill_workers_fn) -> None:
    _execute_panic_stop_impl(
        consciousness,
        kill_workers_fn,
        data_dir=DATA_DIR,
        panic_exit_code=PANIC_EXIT_CODE,
        log=log,
    )

APP_START = time.time()


def _sync_gateway_settings_module() -> None:
    """Keep legacy server.* monkeypatch tests wired to gateway.settings."""
    _gateway_settings.load_settings = load_settings
    _gateway_settings.save_settings = save_settings
    _gateway_settings._apply_settings_to_env = _apply_settings_to_env
    _gateway_settings.apply_runtime_provider_defaults = apply_runtime_provider_defaults


async def api_settings_get(request):
    _sync_gateway_settings_module()
    return await _gateway_settings.api_settings_get(request)


async def api_settings_post(request):
    _sync_gateway_settings_module()
    return await _gateway_settings.api_settings_post(request)

web_dir = resolve_web_dir(REPO_DIR)
web_dir.mkdir(parents=True, exist_ok=True)
index_page = make_index_page(web_dir)

routes = [
    Route("/", endpoint=index_page),
    *collect_routes(
        data_dir=DATA_DIR,
        settings_handlers={
            "api_onboarding": _gateway_settings.api_onboarding,
            "api_settings_get": api_settings_get,
            "api_settings_post": api_settings_post,
        },
    ),
    Mount("/static", app=NoCacheStaticFiles(directory=str(web_dir)), name="static"),
]

from contextlib import asynccontextmanager, suppress


def _run_startup_task_recovery(
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    *,
    skip_live_data: bool,
) -> None:
    """Reconcile durable task phases once, after the prior process is gone."""
    if skip_live_data:
        return
    try:
        from ouroboros.task_status import reconcile_orphaned_running_tasks

        reconcile_orphaned_running_tasks(drive_root)
    except Exception:
        log.warning("Orphaned running-task reconciliation at startup failed", exc_info=True)
    try:
        from ouroboros.agent_task_pipeline import recover_pending_root_post_task_synthesis

        recover_pending_root_post_task_synthesis(drive_root, repo_dir)
    except Exception:
        log.warning("Root post-task synthesis recovery at startup failed", exc_info=True)


@asynccontextmanager
async def lifespan(app):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    _set_ws_event_loop(_event_loop)
    ws_heartbeat_task = asyncio.create_task(
        ws_heartbeat_loop(_has_ws_clients, broadcast_ws),
        name="ws-heartbeat",
    )

    settings, provider_defaults_changed, _provider_default_keys = apply_runtime_provider_defaults(load_settings())
    # Persist the boot normalization only for an install that ALREADY has a
    # settings file. Creating it here would make the server — which now starts
    # BEFORE first-run onboarding on every host — the author of the first bytes
    # of settings.json, and every fresh-install proof is gated on that file being
    # absent until the owner's own onboarding save (the wizard's `light` safety
    # coverage, install-time agent presets). Nothing is lost: the values are
    # applied in-process below, and the completion save persists the same
    # normalization. Mirror of launcher._prepare_first_run_settings.
    from ouroboros.config import SETTINGS_PATH as _settings_path

    if provider_defaults_changed and _settings_path.exists():
        save_settings(settings, allow_elevation=True)
    _apply_settings_to_env(settings)
    # Pin boot-time runtime-mode after env apply; save_settings compares to this owner baseline.
    from ouroboros.config import initialize_runtime_mode_baseline
    initialize_runtime_mode_baseline()
    has_local = needs_local_model_autostart(settings)
    lifespan_drive_root = pathlib.Path(
        app.state.drive_root
        if hasattr(app, "state") and hasattr(app.state, "drive_root")
        else DATA_DIR
    )
    default_real_data_dir = pathlib.Path.home() / "Ouroboros" / "data"
    pytest_default_real_data_dir = (
        (bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules)
        and lifespan_drive_root == default_real_data_dir
        and not os.environ.get("OUROBOROS_DATA_DIR")
    )

    # Source-mode must seed native skills too, matching packaged launcher layout.
    try:
        if pytest_default_real_data_dir:
            log.info("Skipping native skills bootstrap against real DATA_DIR during pytest")
        else:
            from ouroboros.launcher_bootstrap import ensure_data_skills_seeded
            ensure_data_skills_seeded()
    except Exception:
        log.warning("Native skills bootstrap failed", exc_info=True)

    # Boot-reconcile the project registry BEFORE /api/state and context-building
    # can rely on registered_project_chat_ids (the multi-project isolation SSOT):
    # register any pre-existing data/projects/<id>/ store whose row is missing, so
    # an inherited project's raw chat is partitioned from turn one (not only after
    # the 300s periodic tick). Idempotent and never prunes.
    try:
        if not pytest_default_real_data_dir:
            from ouroboros.projects_registry import reconcile_projects
            reconcile_projects(lifespan_drive_root)
    except Exception:
        log.warning("Project registry boot reconcile failed", exc_info=True)

    if has_startup_ready_provider(settings):
        _start_supervisor_if_needed(settings)
    else:
        _supervisor_ready.set()
        log.info("No supported provider or local routing configured. Supervisor not started.")

    # P2: finalize a pending managed merge update (post-boot smoke / boot-loop rollback)
    # and run a one-shot boot-time update check (check-on-restart) so the main-screen
    # Update badge reflects availability. Both run OFF the startup critical path and
    # fail-soft — a missing managed remote / offline boot simply yields no badge.
    threading.Thread(
        target=_boot_managed_update_tasks, daemon=True, name="boot-managed-update",
    ).start()

    if has_local and settings.get("LOCAL_MODEL_SOURCE"):
        from ouroboros.local_model_autostart import auto_start_local_model
        threading.Thread(
            target=auto_start_local_model, args=(settings,),
            daemon=True, name="local-model-autostart",
        ).start()

    host_service_task = None
    host_service_server = None
    extension_reconcile_task = None
    try:
        from ouroboros.event_bus import init_global_event_bus
        from ouroboros.extension_companion import init_global_supervisor
        from ouroboros.gateway.host_service import (
            DEFAULT_HOST_SERVICE_HOST,
            create_host_service_app,
            host_service_port,
        )

        init_global_event_bus().set_loop(_event_loop)
        init_global_supervisor(lifespan_drive_root)
        host_service_app = create_host_service_app(lifespan_drive_root)
        host_port = host_service_port()
        # Probe the port first: uvicorn's Server.startup() calls sys.exit(1) on a
        # bind error, and SystemExit raised inside an asyncio task escapes
        # run_forever and takes down the WHOLE main server (a stale prior
        # instance still holding the port is exactly the realistic trigger).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _probe:
            _probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                _probe.bind((DEFAULT_HOST_SERVICE_HOST, host_port))
            except OSError as bind_exc:
                raise RuntimeError(
                    f"Host Service port {host_port} is busy: {bind_exc}"
                ) from bind_exc
        host_service_config = uvicorn.Config(
            host_service_app,
            host=DEFAULT_HOST_SERVICE_HOST,
            port=host_port,
            log_level="warning",
        )
        host_service_server = uvicorn.Server(host_service_config)
        host_service_task = asyncio.create_task(
            host_service_server.serve(),
            name="host-service-api",
        )
        log.info("Host Service API listening on %s:%d", DEFAULT_HOST_SERVICE_HOST, host_port)
    except Exception:
        log.warning("Failed to start Host Service API", exc_info=True)

    try:
        from ouroboros.skill_review_runner import reconcile_stale_review_jobs

        if pytest_default_real_data_dir:
            log.info("Skipping stale skill-review reconciliation against real DATA_DIR during pytest")
        else:
            reconcile_stale_review_jobs(lifespan_drive_root)
    except Exception:
        log.warning("Stale skill-review reconciliation at startup failed", exc_info=True)

    # Startup-only: after the prior process generation is gone, finalize orphaned
    # RUNNING results and resolve an indeterminate post-task synthesis phase.
    # The periodic zombie sweep intentionally does not perform this recovery.
    _run_startup_task_recovery(
        lifespan_drive_root,
        REPO_DIR,
        skip_live_data=pytest_default_real_data_dir,
    )

    # Reload enabled+reviewed extensions across restarts.
    try:
        from ouroboros.config import (
            get_skills_repo_path,
            load_settings as _load_settings,
        )
        from ouroboros.extension_loader import reload_all as _reload_extensions
        from ouroboros.extension_loader import set_ws_broadcaster as _set_extension_ws_broadcaster
        _set_extension_ws_broadcaster(broadcast_ws_sync)
        repo_path = get_skills_repo_path()
        if pytest_default_real_data_dir:
            log.info("Skipping extension reload_all against real DATA_DIR during pytest")
        else:
            _reload_extensions(lifespan_drive_root, _load_settings, repo_path=repo_path or None)
    except Exception:
        log.error("Extension reload_all at startup failed", exc_info=True)

    try:
        from ouroboros.mcp_client import (
            reconfigure_from_settings as _mcp_reconfigure_startup,
            refresh_all_background as _mcp_refresh_background_startup,
        )
        _mcp_reconfigure_startup(settings)
        _mcp_refresh_background_startup(reason="startup")
    except Exception:
        log.warning("MCP startup reconfigure failed", exc_info=True)

    try:
        from ouroboros.config import get_skills_repo_path
        from ouroboros.config import load_settings as _load_settings
        from ouroboros.extension_reconcile_queue import extension_reconcile_pickup_loop

        if pytest_default_real_data_dir:
            log.info("Skipping extension reconcile pickup against real DATA_DIR during pytest")
        else:
            extension_reconcile_task = asyncio.create_task(
                extension_reconcile_pickup_loop(
                    lifespan_drive_root,
                    _load_settings,
                    repo_path_getter=lambda: get_skills_repo_path() or None,
                ),
                name="extension-reconcile-pickup",
            )
    except Exception:
        log.warning("Failed to start extension reconcile pickup task", exc_info=True)

    try:
        yield
    finally:
        if extension_reconcile_task is not None:
            extension_reconcile_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(extension_reconcile_task, timeout=30)
        if host_service_server is not None:
            try:
                host_service_server.should_exit = True
            except Exception:
                pass
        if host_service_task is not None:
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(host_service_task, timeout=5)
            if not host_service_task.done():
                host_service_task.cancel()
                with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    await asyncio.wait_for(host_service_task, timeout=2)
        ws_heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await ws_heartbeat_task

        log.info("Server shutting down...")
        try:
            from ouroboros.local_model import get_manager
            get_manager().stop_server()
        except Exception:
            pass
        try:
            from ouroboros.tools.shell import kill_all_tracked_subprocesses
            kill_all_tracked_subprocesses()
        except Exception:
            pass
        try:
            from ouroboros.workspace_executor import kill_all_foreground
            kill_all_foreground(lifespan_drive_root)
        except Exception:
            pass
        try:
            from ouroboros.tools.services import kill_all_services
            kill_all_services(lifespan_drive_root)
        except Exception:
            pass
        try:
            from ouroboros.extension_companion import get_global_supervisor
            supervisor = get_global_supervisor()
            if supervisor is not None:
                supervisor.stop_all()
        except Exception:
            pass
        _drain_task_done_finalizations()
        try:
            restart_requested = _restart_requested.is_set()
            # Record an explicit shutdown cause so a task interrupted by the
            # shutdown is never later read as a worker crash storm.
            try:
                from ouroboros.utils import append_jsonl, utc_now_iso
                append_jsonl(
                    lifespan_drive_root / "logs" / "supervisor.jsonl",
                    {
                        "ts": utc_now_iso(),
                        "type": "server_shutdown",
                        "cause": "restart_requested" if restart_requested else "external_signal",
                        "restart_exit": restart_requested,
                    },
                )
            except Exception:
                log.debug("Failed to record server_shutdown event", exc_info=True)
            from supervisor.workers import kill_workers
            cleanup_status, cleanup_reason = _shutdown_task_cleanup_args(restart_requested)
            kill_workers(
                force=True,
                terminal_status=cleanup_status,
                result_reason=cleanup_reason,
                **_managed_update_pending_kwargs(),
            )
        except Exception:
            pass
        try:
            from supervisor.message_bus import get_bridge
            get_bridge().shutdown()
        except Exception:
            pass
        _shutdown_supervisor_event_bus()


app = NetworkAuthGate(Starlette(routes=routes, lifespan=lifespan))
app.app.state.drive_root = pathlib.Path(DATA_DIR)  # type: ignore[attr-defined]
app.app.state.repo_dir = pathlib.Path(REPO_DIR)  # type: ignore[attr-defined]
app.app.state.broadcast_ws_sync = broadcast_ws_sync  # type: ignore[attr-defined]
app.app.state.app_start = APP_START  # type: ignore[attr-defined]
app.app.state.supervisor_ready_event = _supervisor_ready  # type: ignore[attr-defined]
app.app.state.get_supervisor_error = lambda: _supervisor_error  # type: ignore[attr-defined]
app.app.state.describe_bg_consciousness_state = _describe_bg_consciousness_state  # type: ignore[attr-defined]
app.app.state.request_restart = _request_restart_exit  # type: ignore[attr-defined]
app.app.state.runtime_branch_defaults = _runtime_branch_defaults  # type: ignore[attr-defined]
app.app.state.bind_host = _BIND_HOST  # type: ignore[attr-defined]
app.app.state.port_file = PORT_FILE  # type: ignore[attr-defined]
app.app.state.default_port = DEFAULT_PORT  # type: ignore[attr-defined]
app.app.state.start_supervisor_if_needed = _start_supervisor_if_needed  # type: ignore[attr-defined]


_ACTUAL_BOUND_PORT: Optional[int] = None


def _actual_bound_port() -> int:
    """Port the server actually bound (set in main(); DEFAULT_PORT before that)."""
    return _ACTUAL_BOUND_PORT if _ACTUAL_BOUND_PORT else DEFAULT_PORT


def _emergency_process_cleanup(*, port_sweep: bool = True) -> None:
    """Kill child processes, workers, companions, and runtime port holders."""
    _drain_task_done_finalizations()
    try:
        from ouroboros.tools.shell import kill_all_tracked_subprocesses
        kill_all_tracked_subprocesses()
    except Exception:
        pass
    try:
        from ouroboros.workspace_executor import kill_all_foreground
        kill_all_foreground(DATA_DIR, wait=False)
    except Exception:
        pass
    try:
        from ouroboros.tools.services import kill_all_services
        kill_all_services(DATA_DIR, wait=False)
    except Exception:
        pass
    try:
        from supervisor.workers import kill_workers
        if _restart_requested.is_set():
            # A restart that hung past the uvicorn shutdown timeout still reaches
            # here; finalize running tasks as an honest interrupted-by-restart,
            # not a worker crash storm.
            cleanup_status, cleanup_reason = _shutdown_task_cleanup_args(True)
            kill_workers(
                force=True,
                archive_service_logs=False,
                terminal_status=cleanup_status,
                result_reason=cleanup_reason,
                **_managed_update_pending_kwargs(),
            )
        else:
            kill_workers(
                force=True,
                archive_service_logs=False,
                **_managed_update_pending_kwargs(),
            )
    except Exception:
        pass
    import multiprocessing
    from ouroboros.platform_layer import force_kill_pid, kill_process_on_port
    for child in multiprocessing.active_children():
        try:
            force_kill_pid(child.pid)
        except (ProcessLookupError, PermissionError):
            pass
        # Reap the Process object so it does not linger as a zombie / keep
        # active_children non-empty if the main process exits before it dies.
        try:
            child.join(timeout=2)
        except Exception:
            pass
    if port_sweep:
        # Sweep the ACTUALLY bound port (find_free_port may have moved off
        # DEFAULT_PORT); the old hardcoded 8765/8766 pair could kill an
        # unrelated process on a custom-port install.
        kill_process_on_port(_actual_bound_port())
    try:
        from ouroboros.extension_companion import panic_kill_all
        from ouroboros.gateway.host_service import host_service_port
        panic_kill_all()
        if port_sweep:
            kill_process_on_port(host_service_port())
    except Exception:
        pass

def main() -> int:
    # A benchmark-owned child may receive an integrity pin from its parent.
    # Verify the exact bytes before even resolving the saved bind host; a
    # malformed/replaced snapshot must not be converted into product defaults.
    try:
        verify_settings_integrity()
    except SettingsIntegrityError:
        log.error("isolated settings integrity verification failed")
        return 2
    try:
        saved_host = str(load_settings().get("OUROBOROS_SERVER_HOST") or "").strip()
    except Exception:
        saved_host = ""
    default_host = os.environ.get("OUROBOROS_SERVER_HOST", "").strip() or saved_host or DEFAULT_HOST
    args = parse_server_args(default_host, DEFAULT_PORT)
    global _BIND_HOST
    _BIND_HOST = args.host
    app.app.state.bind_host = args.host  # type: ignore[attr-defined]
    auth_warning = get_network_auth_startup_warning(args.host)
    if auth_warning:
        log.warning(auth_warning)
    auth_error = validate_network_auth_configuration(args.host)
    if auth_error:
        log.error(auth_error)
        return 2
    actual_port = find_free_port(args.host, args.port)
    if actual_port != args.port:
        log.info("Port %d busy on %s, using %d instead", args.port, args.host, actual_port)
    global _ACTUAL_BOUND_PORT
    _ACTUAL_BOUND_PORT = actual_port
    write_port_file(PORT_FILE, actual_port)
    log.info("Starting Ouroboros server on %s:%d", args.host, actual_port)
    config = uvicorn.Config(
        app,
        host=args.host,
        port=actual_port,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
    server = uvicorn.Server(config)
    _uvicorn_exited = threading.Event()

    def _check_restart():
        """Monitor restart signal, then shut down uvicorn."""
        while not _restart_requested.is_set():
            time.sleep(0.5)
        log.info("Restart requested — closing WebSocket clients and shutting down server.")

        loop = _event_loop
        if loop:
            try:
                future = asyncio.run_coroutine_threadsafe(close_all_ws(), loop)
                future.result(timeout=3)
            except Exception:
                pass

        server.should_exit = True

        # Force-exit only if uvicorn never returns; direct-server mode needs cleanup/re-exec time.
        force_exit_timeout_sec = 5 if _LAUNCHER_MANAGED else 30
        if _uvicorn_exited.wait(timeout=force_exit_timeout_sec):
            return
        log.warning(
            "Uvicorn did not exit within %ss — running emergency cleanup before os._exit(%d)",
            force_exit_timeout_sec,
            RESTART_EXIT_CODE,
        )
        _emergency_process_cleanup()
        os._exit(RESTART_EXIT_CODE)

    threading.Thread(target=_check_restart, daemon=True).start()

    try:
        server.run()
    finally:
        _uvicorn_exited.set()

    if _restart_requested.is_set():
        log.info("Exiting with code %d (restart signal).", RESTART_EXIT_CODE)
        _emergency_process_cleanup(port_sweep=False)
        if not _LAUNCHER_MANAGED:
            if _planned_delegate_restart_transaction_id:
                from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV

                os.environ[PLANNED_RESTART_TRANSACTION_ENV] = (
                    _planned_delegate_restart_transaction_id
                )
            _restart_current_process(args.host, actual_port)
        os._exit(RESTART_EXIT_CODE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
