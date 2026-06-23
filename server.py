"""Self-editable Starlette/uvicorn entry point for UI and supervisor runtime."""

import asyncio
import json
import logging
import socket

import os
import pathlib
import sys
import threading
import time
import uuid
from ouroboros.utils import utc_now_iso
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
    _file_handler = RotatingFileHandler(
        _log_dir / "server.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, handlers=[_file_handler, logging.StreamHandler()])
log = logging.getLogger("server")

RESTART_EXIT_CODE = 42
PANIC_EXIT_CODE = 99
_restart_requested = threading.Event()
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
    _restart_current_process_impl(host, port, repo_dir=REPO_DIR, log=log)

from ouroboros.config import (
    load_settings, save_settings, apply_settings_to_env as _apply_settings_to_env,
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


def _route_project_chat_to_running_task(ctx: Any, chat_id: int, message: str, client_message_id: str = "") -> str:
    """Deliver a PROJECT-room follow-up to its running pooled task's mailbox — ONLY
    for the unambiguous 1:1 case.

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
        from ouroboros.projects_registry import registered_project_chat_ids

        if int(chat_id or 0) not in registered_project_chat_ids(ctx.DRIVE_ROOT):
            return ""
    except Exception:
        return ""
    try:
        steerable: list = []
        for tid, running in list(ctx.RUNNING.items()):
            if not isinstance(running, dict):
                continue
            task_obj = running.get("task") if isinstance(running.get("task"), dict) else running
            if int(task_obj.get("chat_id") or 0) != int(chat_id or 0):
                # A post-hoc "Turn into project" task keeps its original (main)
                # chat_id on the live object but belongs to this project thread;
                # match it via the durable binding so follow-ups still steer it.
                try:
                    from ouroboros.projects_registry import project_chat_for_task

                    if int(project_chat_for_task(ctx.DRIVE_ROOT, tid) or 0) != int(chat_id or 0):
                        continue
                except Exception:
                    continue
            if task_obj.get("_is_direct_chat"):
                continue
            if str(task_obj.get("delegation_role") or "") == "subagent":
                continue
            steerable.append((str(tid), task_obj))
        # Exactly one candidate => unambiguous transport. Zero or many => a routing
        # decision the AGENT must make (P5/WS1), so do not deliver here.
        if len(steerable) != 1:
            return ""
        tid, task_obj = steerable[0]
        from ouroboros.owner_mailbox import write_owner_message
        from supervisor.queue import _task_drive_for_task

        # Active drive (child drive for forked/workspace tasks) — mirror
        # forward_to_worker / steer_task so the mailbox lands where the task
        # actually drains it, not the canonical root. A stable msg_id derived from
        # client_message_id makes this 1:1 delivery idempotent — a WebSocket retry of
        # the same message can't double-deliver (drain_owner_entries dedups by msg_id),
        # matching steer_task's contract.
        task_drive = _task_drive_for_task(task_obj, tid)
        msg_id = f"{client_message_id}:{tid}" if client_message_id else None
        write_owner_message(task_drive, message, tid, msg_id=msg_id)
        return tid
    except Exception:
        log.debug("Mailbox follow-up routing failed; falling back to direct lane", exc_info=True)
    return ""


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
    out: list = []
    try:
        from ouroboros.projects_registry import project_chat_for_task
    except Exception:
        project_chat_for_task = None  # type: ignore
    try:
        for tid, running in list(ctx.RUNNING.items()):
            if not isinstance(running, dict):
                continue
            task_obj = running.get("task") if isinstance(running.get("task"), dict) else running
            if not isinstance(task_obj, dict):
                continue
            if task_obj.get("_is_direct_chat") or str(task_obj.get("delegation_role") or "") == "subagent":
                continue
            same = False
            try:
                same = int(task_obj.get("chat_id") or 0) == int(chat_id or 0)
            except (TypeError, ValueError):
                same = False
            if not same and project_chat_for_task is not None:
                try:
                    same = int(project_chat_for_task(ctx.DRIVE_ROOT, str(tid)) or 0) == int(chat_id or 0)
                except Exception:
                    same = False
            if not same:
                continue
            objective = str(
                task_obj.get("objective") or task_obj.get("description") or task_obj.get("text") or ""
            ).strip()
            out.append({
                "task_id": str(tid),
                "status": "running",
                "title": _clip_marked(task_obj.get("title"), 120),
                "objective": _clip_marked(objective, 600),
                "project_id": str(task_obj.get("project_id") or ""),
                "started_at": running.get("started_at"),
                "steerable": True,
            })
    except Exception:
        log.debug("running-tasks snapshot failed", exc_info=True)
    return out


def _decision_turn_metadata(ctx: Any, chat_id: int, client_message_id: str, task_metadata: Any) -> Any:
    """Enrich a chat turn's metadata with the structural facts the decision turn
    needs: the RUNNING tasks in THIS chat (so it can steer_task the right one
    instead of spawning a duplicate) and the originating message id (for idempotent
    steer delivery). P5-clean: surfaces state only; the agent picks the target by
    judgment among answer / steer_task / promote_chat_to_task / route_to_project."""
    running_here = _chat_running_tasks(ctx, chat_id)
    if not running_here and not client_message_id:
        return task_metadata
    md = dict(task_metadata) if isinstance(task_metadata, dict) else {}
    if running_here:
        md["current_chat"] = {"chat_id": int(chat_id or 0), "running_tasks": running_here}
    if client_message_id:
        md["client_message_id"] = client_message_id
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
            from supervisor.message_bus import get_bridge
            get_bridge().send_message(
                owner_chat,
                f"⚠️ A chat turn looks wedged (~{int(gap)}s with no heartbeat). New messages "
                "still get answered, but the stuck turn can't be cleared in-process — /restart "
                "to fully recover it.",
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
            now = time.time()
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
                            from supervisor.message_bus import get_bridge
                            get_bridge().send_message(
                                owner_chat,
                                f"⚠️ My supervisor loop stalled for ~{int(gap)}s — new messages may be "
                                "delayed. I recover on the next tick or a restart; investigating.",
                            )
                    except Exception:
                        log.debug("loop-stall owner alert failed", exc_info=True)
                    loop_alerted = True
            else:
                loop_alerted = False
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


def _periodic_supervisor_maintenance(last_custody_reap: list, last_review_reconcile: list) -> None:
    """Throttled periodic upkeep extracted from the supervisor loop: custody reap
    of orphaned task-scoped processes (every 600s) + review-job zombie reconcile
    (every 300s). Each cadence gates itself via its own last-run marker."""
    if time.time() - last_custody_reap[0] > 600:
        last_custody_reap[0] = time.time()
        try:
            from ouroboros.process_custody import reap_orphaned_processes
            from supervisor.queue import RUNNING as _running_tasks

            reap_orphaned_processes(
                DATA_DIR, running_task_ids=set(_running_tasks.keys()),
                live_owner_skills=_installed_skill_names(),
            )
        except Exception:
            log.debug("Periodic custody reap failed", exc_info=True)
    if time.time() - last_review_reconcile[0] > 300:
        last_review_reconcile[0] = time.time()
        _periodic_zombie_reconcile()


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
    (task_metadata.project_id) and routed to its panel. Includes ARCHIVED projects
    so the classification stays consistent; archiving is a UI-visibility concern
    only (web/app.js filters it).
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
        image_data = (
            (image_base64, image_mime, image_caption)
            if image_base64
            else None
        )
        log_text = text or image_caption or ("(image attached)" if image_base64 else "")
        now_iso = utc_now_iso()

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

        if not suppress_chat_log:
            log_chat(
                "in",
                chat_id,
                user_id,
                log_text,
                source=source,
                sender_label=sender_label,
                sender_session_id=sender_session_id,
                client_message_id=client_message_id,
                transport=transport,
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
        # Atomic owner-binding + activity stamp: the old load→(log/broadcast)→save
        # span could overwrite concurrent budget/state writers with stale data.
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
            # External transports authorize slash commands against a SEPARATE
            # owner-external slot, so the local web owner (1/1) can never lock out
            # a real Telegram owner. The first external slash binds it (TOFU) and
            # asks for a resend instead of executing.
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
            ok, restart_msg = ctx.safe_restart(reason="owner_restart", unsynced_policy="rescue_and_reset")
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
            _request_restart_exit()
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
                from supervisor.evolution_lifecycle import evolution_block_reason

                block = evolution_block_reason()
                if block:
                    ctx.send_with_budget(chat_id, block)
                    continue
            st2 = ctx.load_state()
            st2["evolution_mode_enabled"] = bool(turn_on)
            if turn_on:
                st2["evolution_consecutive_failures"] = 0
            # Owner-initiated evolution must not inherit a stale post-task one-shot
            # autostop, which would disable the owner's campaign after one cycle.
            st2["post_task_autostop"] = False
            ctx.save_state(st2)
            try:
                from supervisor.evolution_lifecycle import pause_evolution_campaign, start_evolution_campaign

                if turn_on:
                    start_evolution_campaign(objective, source="owner_chat")
                else:
                    pause_evolution_campaign("disabled via owner chat")
            except Exception:
                log.warning("Failed to update evolution campaign state", exc_info=True)
            if not turn_on:
                # Cancel the live evolution worker too — pruning PENDING alone
                # leaves a mid-cycle task running (and eligible for retry).
                from supervisor.queue import cancel_running_evolution_tasks

                cancelled = cancel_running_evolution_tasks("disabled via owner chat")
                ctx.PENDING[:] = [t for t in ctx.PENDING if str(t.get("type")) != "evolution"]
                ctx.sort_pending()
                ctx.persist_queue_snapshot(reason="evolve_off")
                if cancelled:
                    ctx.send_with_budget(
                        chat_id,
                        f"🛑 Cancelled running evolution task(s): {', '.join(cancelled)}",
                    )
            ctx.send_with_budget(chat_id, f"🧬 Evolution campaign: {'ON' if turn_on else 'OFF'}")
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
            project_id = _project_id_for_registered_chat(ctx, chat_id)
            # Full project awareness (v6.32.0): the one mind notices EVERY human
            # message, including in a project room — project history is part of its
            # continuous awareness (BIBLE P1), not a separate isolated stream.
            ctx.consciousness.inject_observation(f"Message from my human: {log_text}")
            task_metadata = _scoped_task_metadata(project_id, task_metadata)
            routed_to_task = _route_project_chat_to_running_task(
                ctx, chat_id, text or image_caption, client_message_id
            )
            if routed_to_task:
                ctx.send_with_budget(
                    chat_id,
                    f"📨 Forwarded to the running task {routed_to_task} "
                    "(it will see this on its next round).",
                )
                continue
            task_metadata = _decision_turn_metadata(ctx, chat_id, client_message_id, task_metadata)
            agent = ctx.get_chat_agent()

            def _run_constrained_or_resume(cid, txt, img, constraint, metadata, resume_consciousness: bool):
                try:
                    ctx.handle_chat_direct(
                        cid,
                        txt,
                        img,
                        task_constraint=constraint,
                        task_metadata=metadata,
                    )
                finally:
                    if resume_consciousness:
                        ctx.consciousness.resume()

            if agent._busy:
                # turn = decision (v6.33.0 WS10; WS1 v6.34.0): a new message — MAIN chat
                # OR a PROJECT room — NEVER injects into or blocks on the running turn,
                # and it is NEVER mechanically auto-spawned into a duplicate task. It runs
                # as a SHORT-LIVED SAME-ROUTE decision turn on a separate agent instance
                # (project-scoped via task_metadata.project_id, seeing current_chat.running_tasks),
                # so the one mind decides per-message by JUDGMENT — answer inline, steer_task
                # a running task, or promote_chat_to_task new parallel work (P5/BIBLE LLM-first).
                threading.Thread(
                    target=ctx.handle_chat_ephemeral,
                    args=(chat_id, text or image_caption, image_data),
                    kwargs={"task_constraint": task_constraint, "task_metadata": task_metadata},
                    daemon=True,
                ).start()
            else:
                ctx.consciousness.pause()
                threading.Thread(
                    target=_run_constrained_or_resume,
                    args=(chat_id, text or image_caption, image_data, task_constraint, task_metadata, True),
                    daemon=True,
                ).start()
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
        ok, msg = git_ops_module.safe_restart(reason="bootstrap", unsynced_policy=policy)
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


def _run_supervisor(settings: dict) -> None:
    """Initialize and run the supervisor loop. Called in a background thread."""
    global _supervisor_error, _supervisor_thread, _consciousness

    _apply_settings_to_env(settings)

    # Supervisor revival (e.g. settings POST after a loop death) must not leak
    # the previous generation: the old BackgroundConsciousness daemon thread
    # would keep burning budget unreachable by /bg stop, and the cached direct
    # chat agent stays bound to the OLD event queue (messages to a dead queue).
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
        from supervisor.message_bus import init as bus_init
        from supervisor.message_bus import LocalChatBridge

        bridge = LocalChatBridge(settings)
        bridge._broadcast_fn = broadcast_ws_sync

        from ouroboros.utils import set_log_sink
        set_log_sink(bridge.push_log)

        bus_init(
            drive_root=DATA_DIR,
            total_budget_limit=float(settings.get("TOTAL_BUDGET", 10.0)),
            budget_report_every=10,
            chat_bridge=bridge,
        )

        from supervisor.state import init as state_init, init_state, load_state, save_state, update_state
        from supervisor.state import append_jsonl, update_budget_from_usage, rotate_chat_log_if_needed
        state_init(DATA_DIR, float(settings.get("TOTAL_BUDGET", 10.0)))
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
            total_budget_limit=float(settings.get("TOTAL_BUDGET", 10.0)),
            branch_dev=_workers_branch_dev, branch_stable=_workers_branch_stable,
        )

        from supervisor.events import dispatch_event
        from supervisor.message_bus import send_with_budget
        from ouroboros.consciousness import BackgroundConsciousness
        import types
        import queue as _queue_mod

        kill_workers()
        spawn_workers(max_workers)
        restored_pending = restore_pending_from_snapshot()
        persist_queue_snapshot(reason="startup")
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
        try:
            from ouroboros.process_custody import reap_orphaned_processes

            reaped = reap_orphaned_processes(DATA_DIR, live_owner_skills=_installed_skill_names())
            if reaped:
                log.info("Process custody reaper killed %d orphaned process(es): %s", len(reaped), reaped)
        except Exception:
            log.debug("Process custody startup reap failed", exc_info=True)

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
    # of silent hours; the loop publishes a liveness tick each iteration.
    _loop_liveness = [time.time()]
    _watchdog_stop = threading.Event()  # per-generation: stops the watchdog when THIS loop exits
    _start_supervisor_liveness_watchdog(_loop_liveness, _watchdog_stop)
    while not _restart_requested.is_set():
        try:
            _loop_liveness[0] = time.time()
            rotate_chat_log_if_needed(DATA_DIR)
            ensure_workers_healthy()

            event_q = get_event_q()
            while True:
                try:
                    evt = event_q.get_nowait()
                except _queue_mod.Empty:
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

    Heartbeat-staleness reuses the swarm SSOT getter
    (``config.get_plan_task_swarm_heartbeat_stale_sec``) so there is one
    definition of "a worker is still alive", not a scattered magic cutoff.
    """
    from ouroboros.config import get_plan_task_swarm_heartbeat_stale_sec

    stale_sec = get_plan_task_swarm_heartbeat_stale_sec()
    now = time.time()
    live = []
    for tid, meta in dict(ctx.RUNNING or {}).items():
        if not isinstance(meta, dict):
            continue
        try:
            hb = float(meta.get("last_heartbeat_at") or 0.0)
        except (TypeError, ValueError):
            hb = 0.0
        if hb and (now - hb) < stale_sec:
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
        })
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                f"⏳ Restart drain: waiting up to {max_wait}s for running task(s) "
                f"{', '.join(sorted(live))} to finish.",
            )
        return
    _perform_supervisor_restart(ctx)


def _check_pending_restart_drain(ctx: Any) -> bool:
    """Loop-tick hook: complete a deferred restart once tasks drain or the
    deadline passes (proceeds fail-closed). Returns True while STILL draining, so
    the loop can skip starting new work that the restart would immediately chop."""
    if not _pending_restart:
        return False
    live = _live_running_task_ids(ctx)
    if live and time.time() < float(_pending_restart.get("deadline") or 0.0):
        return True  # keep draining — events still flow each tick
    _pending_restart.clear()
    _perform_supervisor_restart(ctx)
    # Still "quiescing" this tick: _perform_supervisor_restart sets up the exit
    # (or fail-closed pauses) and returns to the loop — the process exits on the
    # next `while not _restart_requested` check. Returning True keeps the caller
    # from starting new enqueue/assign work on this final pre-exit tick.
    return True


def _perform_supervisor_restart(ctx: Any) -> None:
    """Graceful shutdown + exit(42) (the post-drain tail; never sleeps)."""
    st = ctx.load_state()
    ok, msg = ctx.safe_restart(
        reason="agent_restart_request", unsynced_policy="rescue_and_block",
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
    ctx.kill_workers(force=True, terminal_status=cleanup_status, result_reason=cleanup_reason)
    st2 = ctx.load_state()
    st2["session_id"] = uuid.uuid4().hex
    ctx.save_state(st2)
    ctx.persist_queue_snapshot(reason="pre_restart_exit")
    _request_restart_exit()


def _request_restart_exit() -> None:
    """Signal server shutdown with restart exit code."""
    _restart_requested.set()


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
            "api_claude_code_status": _gateway_settings.api_claude_code_status,
            "api_claude_code_install": _gateway_settings.api_claude_code_install,
            "api_settings_get": api_settings_get,
            "api_settings_post": api_settings_post,
        },
    ),
    Mount("/static", app=NoCacheStaticFiles(directory=str(web_dir)), name="static"),
]

from contextlib import asynccontextmanager, suppress


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
    if provider_defaults_changed:
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
    def _boot_managed_update_tasks():
        try:
            _supervisor_ready.wait(timeout=60)
            from supervisor.git_ops import compute_managed_update_status
            from supervisor.update_merge import finalize_managed_update_on_boot

            # A HEALTHY boot only — _supervisor_ready is also set on supervisor INIT FAILURE
            # (alongside _supervisor_error), so gate on the error too or finalize would clear a
            # pending update as "finalized" on a failed boot, defeating the boot-loop rollback.
            finalize_managed_update_on_boot(
                supervisor_ready=_supervisor_ready.is_set() and not _supervisor_error
            )
            status = compute_managed_update_status(fetch=True)
            # Persist the boot check-on-restart result so the passive Update pill can show
            # availability without a network fetch on every poll (P2 2F: boot fetches once,
            # the badge reads this cache; no periodic polling). A passive
            # compute_managed_update_status(fetch=False) bails before resolving the official
            # ref, so without this cache the pill stays hidden after a restart.
            try:
                from supervisor.state import update_state
                from ouroboros.utils import utc_now_iso

                def _cache_update_status(s):
                    s["managed_update_cache"] = {
                        "available": bool(status.get("available")),
                        "safe_to_apply": bool(status.get("safe_to_apply")),
                        "latest_sha": status.get("latest_sha") or "",
                        "latest_short_sha": status.get("latest_short_sha") or "",
                        "latest_message": status.get("latest_message") or "",
                        "behind": int(status.get("behind") or 0),
                        "ahead": int(status.get("ahead") or 0),
                        "checked_at": utc_now_iso(),
                    }

                update_state(_cache_update_status)
            except Exception:
                log.debug("boot managed-update cache failed", exc_info=True)
        except Exception:
            log.debug("boot managed-update tasks failed", exc_info=True)

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

    # Durably finalize orphaned RUNNING task results (worker died / SIGKILL /
    # manual stop) so a zombie cannot masquerade as still-running across restart.
    # Liveness-gated inside the projection; never touches a still-live task.
    try:
        from ouroboros.task_status import reconcile_orphaned_running_tasks

        if not pytest_default_real_data_dir:
            reconcile_orphaned_running_tasks(lifespan_drive_root)
    except Exception:
        log.warning("Orphaned running-task reconciliation at startup failed", exc_info=True)

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
            kill_workers(force=True, terminal_status=cleanup_status, result_reason=cleanup_reason)
        except Exception:
            pass
        try:
            from supervisor.message_bus import get_bridge
            get_bridge().shutdown()
        except Exception:
            pass


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
            )
        else:
            kill_workers(force=True, archive_service_logs=False)
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
            _restart_current_process(args.host, actual_port)
        os._exit(RESTART_EXIT_CODE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
