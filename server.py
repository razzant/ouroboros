"""Self-editable Starlette/uvicorn entry point for UI and supervisor runtime."""

import asyncio
import base64  # noqa: F401
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

from ouroboros.server_process import (  # noqa: F401
    DATA_DIR,
    _owner_restart_requested,
    _request_restart_exit,
    _restart_requested,
    _supervisor_stop,
    log,
)
from ouroboros.server_routing_context import (  # noqa: F401
    _active_direct_root,
    _addressable_root_tasks,
    _chat_running_tasks,
    _clip_marked,
    _decision_turn_metadata,
    _latest_project_task_result,
    _main_routing_manifest,
    _owner_binding_chat_id,
    _project_id_for_registered_chat,
    _reserved_project_for_chat,
    _scoped_task_metadata,
    _task_belongs_to_chat,
    _task_result_ground_truth,
)
from ouroboros.server_owner_routing import (  # noqa: F401
    _owner_evolution_stop,
    _record_routing_receipt,
    _route_owner_message,
    _route_project_chat_to_running_task,
    _stage_mailbox_attachments,
)
from ouroboros.server_liveness import (  # noqa: F401
    _alert_chat_turn_wedge,
    _chat_turn_wedged,
    _start_supervisor_liveness_watchdog,
    _supervisor_loop_stalled,
)
from ouroboros.server_maintenance import (  # noqa: F401
    _LAST_CANCEL_INTENT_SWEEP,
    _installed_skill_names,
    _periodic_supervisor_maintenance,
    _periodic_zombie_reconcile,
    _prune_delegated_snapshots,
    _reconcile_delegated_runs,
    _resume_interrupted_project_deletions,
    _run_startup_task_recovery,
    _startup_retired_settings_notice,
    _startup_custody_sweep,
    _startup_prune_sweeps,
    _startup_worktree_prune,
)
from ouroboros.server_restart import (  # noqa: F401
    _live_running_task_ids,
    _managed_update_pending_kwargs,
    _safe_restart_serialized,
    _shutdown_supervisor_event_bus,
    _shutdown_task_cleanup_args,
)

REPO_DIR = pathlib.Path(os.environ.get("OUROBOROS_REPO_DIR", pathlib.Path(__file__).parent))
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


from ouroboros.observability import SecretRedactingLogFilter as _SecretRedactingLogFilter

for _handler in logging.getLogger().handlers:
    _handler.addFilter(_SecretRedactingLogFilter())
# httpx logs each request URL at INFO; polling transports put credentials in
# the URL path, so even redacted lines are noise at this level.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

RESTART_EXIT_CODE = 42
PANIC_EXIT_CODE = 99
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
    _supervisor_stop.clear()  # in-process revival after a teardown-stopped generation
    _supervisor_thread = threading.Thread(
        target=_run_supervisor,
        args=(settings,),
        daemon=True,
        name="supervisor-main",
    )
    _supervisor_thread.start()
    return True


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

            status = status_text(ctx.WORKERS, ctx.PENDING, ctx.RUNNING)
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

        # Managed manifest branch defaults must drive worker commit/restart flows too.
        _workers_branch_dev, _workers_branch_stable = _runtime_branch_defaults()
        workers_init(
            repo_dir=REPO_DIR, drive_root=DATA_DIR, max_workers=max_workers,
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
        _startup_retired_settings_notice(settings)

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
    while not _restart_requested.is_set() and not _supervisor_stop.is_set():
        try:
            _loop_liveness[0] = time.monotonic()
            rotate_chat_log_if_needed(DATA_DIR)
            # progress.jsonl rotates on the same supervisor tick (v6.90.x P2); its
            # readers (history backfill, SSE replay, api_logs_tail, TB ATIF) are
            # archive-chain-aware.
            rotate_jsonl_log_if_needed(DATA_DIR, "progress.jsonl", "progress")
            # CPL4-C1..C4 rotation train: the remaining unbounded hot logs rotate
            # on the same tick with the same rotator. events.jsonl went LAST
            # behind chain-aware custody readers (delegate_custody replay + fault
            # scan, complete_custody_rows, the settled-terminal chain cursor,
            # legacy usage import, swarm rollup, worker boot verify) — rows keep
            # replaying from archive/events_*.jsonl; tools/supervisor/
            # task_reflections readers are tail-bounded and archive-backfilled
            # (memory.read_jsonl_tail, api_logs_tail).
            rotate_jsonl_log_if_needed(DATA_DIR, "events.jsonl", "events")
            rotate_jsonl_log_if_needed(DATA_DIR, "tools.jsonl", "tools")
            rotate_jsonl_log_if_needed(DATA_DIR, "supervisor.jsonl", "supervisor")
            rotate_jsonl_log_if_needed(DATA_DIR, "task_reflections.jsonl", "task_reflections")
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
            if _supervisor_stop.is_set() or _restart_requested.is_set():
                # The shutdown/restart tore the bus down under this tick (a
                # Manager proxy raising BrokenPipe/EOF): not a crash, no alarm.
                log.info("Supervisor loop exiting on shutdown: %s", exc)
                break
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
                break  # this generation is dead: the shared exit below stops its watchdog
            # Backoff on the stop event, not time.sleep, so a shutdown is prompt.
            _supervisor_stop.wait(min(30, 2 ** crash_count))
    _watchdog_stop.set()  # every ordinary exit (restart, shutdown, crash death) stops this generation's watchdog
    _supervisor_thread = None


# Deferred restart-drain state (multi-project, v6.32.0). The drain MUST NOT
# sleep on the supervisor loop thread (it is the only thread that processes
# heartbeats / task_done and shrinks RUNNING). Instead a restart with live
# tasks is recorded here and re-checked every loop tick, so events keep
# flowing and the drain actually observes tasks finishing.
_pending_restart: Dict[str, Any] = {}


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
        try:
            from ouroboros.update_letter import refresh_after_check

            refresh_after_check(update_status)
        except Exception:
            log.debug("boot update letter refresh failed", exc_info=True)
        broadcast_ws_sync({
            "type": "update_status_ready",
            "available": bool(update_status.get("available")),
            "check_ok": update_status.get("check_ok"),
        })
    except Exception:
        log.debug("boot managed-update tasks failed", exc_info=True)


def _execute_panic_stop(consciousness, kill_workers_fn) -> None:
    _execute_panic_stop_impl(
        consciousness,
        kill_workers_fn,
        data_dir=DATA_DIR,
        panic_exit_code=PANIC_EXIT_CODE,
        log=log,
        bound_port=_actual_bound_port(),
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


@asynccontextmanager
async def lifespan(app):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    _set_ws_event_loop(_event_loop)
    ws_heartbeat_task = asyncio.create_task(
        ws_heartbeat_loop(_has_ws_clients, broadcast_ws),
        name="ws-heartbeat",
    )

    # Boot APPLIES the provider normalization in-process and persists nothing
    # (mirror of launcher_onboarding.prepare_first_run_settings). Two
    # normalizations, two re-derivations: the VOCABULARY one is the read seam
    # every reader applies (config.normalize_settings_raw); the PROVIDER one is
    # re-derived by every consumer that needs the route — the task-start
    # projection, the settings GET and onboarding reads, the context-fit route
    # resolver — so a start-time write would only make boot a second author of
    # settings.json with no reader that needs it.
    settings, _provider_defaults_changed, _provider_default_keys = apply_runtime_provider_defaults(load_settings())
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

    _supervisor_stop.clear()  # a fresh lifespan owns a fresh generation (symmetric with the teardown set)
    if has_startup_ready_provider(settings):
        _start_supervisor_if_needed(settings)
    else:
        _supervisor_ready.set()
        log.info("No supported provider or local routing configured. Supervisor not started.")

    # P2: finalize a pending managed merge update (post-boot smoke / boot-loop rollback)
    # and run a one-shot boot-time update check (check-on-restart) so the main-screen
    # Update badge reflects availability. Both run OFF the startup critical path and
    # fail-soft — a missing managed remote / offline boot simply yields no badge.
    # Local autostart goes FIRST: a local-only install's boot check may write its update
    # letter through the local model, and the git fetch ahead of that call is the head
    # start the model server gets (no readiness wait — a letter that still finds the
    # model loading fails typed and is rewritten by the next check).
    if has_local and settings.get("LOCAL_MODEL_SOURCE"):
        from ouroboros.local_model_autostart import auto_start_local_model
        threading.Thread(
            target=auto_start_local_model, args=(settings,),
            daemon=True, name="local-model-autostart",
        ).start()

    threading.Thread(
        target=_boot_managed_update_tasks, daemon=True, name="boot-managed-update",
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
        _supervisor_stop.set()  # first: the loop must know a teardown owns what follows
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
        # Let the loop leave its current tick BEFORE workers are killed and the
        # bridge/Manager go down: a tick still running would otherwise respawn
        # a killed worker or meet BrokenPipe/EOF. Bounded well inside the
        # launcher's force-exit budget; the stop flag already suppresses the
        # crash counter if the join times out.
        supervisor_thread = _supervisor_thread
        if supervisor_thread is not None and supervisor_thread.is_alive():
            supervisor_thread.join(timeout=2)
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
