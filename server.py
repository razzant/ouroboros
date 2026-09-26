"""Self-editable Starlette/uvicorn entry point for UI and supervisor runtime."""

import asyncio
import base64  # noqa: F401
import json
import logging
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
from ouroboros.server_control import (execute_panic_stop as _execute_panic_stop_impl,
                                      restart_current_process as _restart_current_process_impl)
from ouroboros.startup_historical_audit import audit as _historical_audit
from ouroboros.server_auth import (
    NetworkAuthGate,
    get_network_auth_startup_warning,
    validate_network_auth_configuration,
)
from ouroboros.server_entrypoint import bound_service_socket, find_free_port, parse_server_args, write_port_file
from ouroboros.server_web import NoCacheStaticFiles, make_index_page, resolve_web_dir
from ouroboros.usage_accounting import ensure_legacy_imported
from ouroboros.task_finalization import host_operation_reply_kwargs
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
    _exit_signalled,
    _SignalStopServer,
    _embedded_uvicorn_server,
    log,
)
from ouroboros.server_routing_context import (  # noqa: F401
    _active_direct_roots,
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
    main_lane_routing_metadata,
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
    drain_worker_events, flush_budget_projection,
)
from ouroboros.server_maintenance import (  # noqa: F401
    _LAST_CANCEL_INTENT_SWEEP,
    _migrate_startup_cancel_latches,
    _startup_worker_pids,
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
    _perform_owner_restart,
    _safe_restart_serialized,
    _shutdown_supervisor_event_bus,
    _shutdown_task_cleanup_args,
    _stop_owned_daemon_for_new_pin,
    _stop_owned_work,
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
_LAUNCHER_MANAGED_REPO_DIR = str(os.environ.get("OUROBOROS_MANAGED_REPO_DIR", "") or "").strip()

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


def _launcher_managed_repo_matches() -> bool:
    if not _LAUNCHER_MANAGED: return False
    if not _LAUNCHER_MANAGED_REPO_DIR: return (REPO_DIR / ".git" / "ouroboros-managed.json").is_file()
    try:
        return pathlib.Path(_LAUNCHER_MANAGED_REPO_DIR).resolve(strict=False) == REPO_DIR.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False


def _restart_current_process(host: str, port: int) -> None:
    # Every direct restart reaches this seam, including an assisted update whose
    # native waits were already moved to PENDING before its resolver ran.
    try:
        from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV, arm_active_planned_restart_transaction
        from ouroboros.server_restart import _RESTARTABLE_UPDATE_PHASES
        from supervisor.update_merge import read_update_tx_strict

        if any((DATA_DIR / "state" / name).exists() for name in ("owner_restart_no_resume.flag", "panic_stop.flag")):
            os.environ.pop(PLANNED_RESTART_TRANSACTION_ENV, None)
        else:
            status, tx = read_update_tx_strict()
            if status == "valid" and tx.get("phase") in _RESTARTABLE_UPDATE_PHASES:
                arm_active_planned_restart_transaction(DATA_DIR)
    except Exception:
        log.warning("Direct restart transaction could not be armed; continuation remains unconfirmed", exc_info=True)
    _restart_current_process_impl(
        host, port, repo_dir=REPO_DIR, log=log,
        owner_initiated=_owner_restart_requested.is_set(),
    )

from ouroboros.config import (
    SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC,
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

_supervisor_ready = threading.Event()  # a live generation finished init: the API's `supervisor_ready`
_supervisor_init_done = threading.Event()  # init reached an outcome (ready OR `_supervisor_error`): boot waiters
_supervisor_error: Optional[str] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_supervisor_thread: Optional[threading.Thread] = None
_consciousness: Any = None


def _clock_of(iso_value: Any) -> str:
    """``HH:MM`` in the server's local time for an ISO instant (``?`` when absent)."""
    from ouroboros.deadline_utils import parse_deadline_ts

    parsed = parse_deadline_ts(str(iso_value or ""))
    return parsed.astimezone().strftime("%H:%M") if parsed is not None else "?"


def _describe_bg_consciousness_state(requested_enabled: bool) -> dict:
    """Project the alarm clock's snapshot into one honest status + detail line."""
    snapshot = _consciousness.status_snapshot() if _consciousness else {}
    outcome = str(snapshot.get("last_wake_outcome") or "")
    next_at = _clock_of(snapshot.get("next_wake_at"))
    if not requested_enabled:
        status, detail = "disabled", "Background consciousness is off."
    elif not snapshot:
        status, detail = "stopped", "Enabled in state, but the alarm clock was not constructed (supervisor init failed)."
    elif snapshot.get("live_wake_task_id"):
        status, detail = "thinking", f"Wake-up {snapshot['live_wake_task_id']} is running as a Main turn."
    elif outcome == "skipped:waiting_for_first_conversation":
        status, detail = "waiting_for_first_conversation", "No owner chat is bound yet; the first conversation binds it."
    elif outcome == "skipped:allowance_exhausted":
        spent, daily = snapshot.get("spent_24h_usd"), snapshot.get("daily_usd")
        status = "allowance_exhausted"
        at_least = "at least " if int(snapshot.get("unknown_unmetered") or 0) > 0 else ""
        degraded = "; ledger integrity degraded" if snapshot.get("integrity_degraded") else ""
        detail = (f"Daily allowance spent ({at_least}${float(spent or 0):.2f} of ${float(daily or 0):.2f} in the last 24 h{degraded}); "
                  f"next check at {_clock_of(snapshot.get('allowance_resets_at')) if snapshot.get('allowance_resets_at') else next_at}.")
    elif outcome == "skipped:allowance_unknown":
        status, detail = "allowance_unknown", f"The usage ledger could not be read ({snapshot.get('last_error') or 'unknown error'}); retry at {next_at}."
    elif outcome.startswith("rejected:"):
        status, detail = "wake_rejected", f"The last wake-up was refused ({outcome.split(':', 1)[1]}); next attempt at {next_at}."
    elif outcome == "failed":
        status, detail = "wake_failed", f"The last wake-up failed ({snapshot.get('last_error') or 'runner error'}); next attempt at {next_at}, backing off."
    else:
        status, detail = "sleeping", f"Sleeping until {next_at}."
        if snapshot.get("pending_reason"):
            detail += f" Early wake pending: {snapshot['pending_reason']}."
    return {**snapshot, "enabled": requested_enabled, "status": status, "detail": detail}


def _start_supervisor_if_needed(settings: dict) -> bool:
    """Start the supervisor once when runtime providers become available."""
    global _supervisor_thread, _supervisor_error
    if not has_startup_ready_provider(settings):
        return False
    if _supervisor_thread and _supervisor_thread.is_alive():
        return False
    if _exit_signalled.is_set():
        return False  # the process is exiting: no revival behind the teardown
    _supervisor_error = None
    _supervisor_stop.clear()  # in-process revival after a teardown-stopped generation
    _supervisor_ready.clear()  # readiness is THIS generation's: Starting, not a stale Online, until init succeeds
    _supervisor_init_done.clear()
    _supervisor_thread = threading.Thread(
        target=_supervisor_generation,
        args=(settings,),
        daemon=True,
        name="supervisor-main",
    )
    _supervisor_thread.start()
    return True


def _supervisor_generation(settings: dict) -> None:
    """Thread body: re-check the exit latch, then run one supervisor generation.

    Admission (`_start_supervisor_if_needed`) and this thread start are separate steps,
    so a settings save can pass the latch check a moment before SIGTERM; a generation
    that starts anyway must end here, before its startup kill/spawn would run behind
    the teardown's `kill_workers` (#1142).
    """
    global _supervisor_thread
    if _exit_signalled.is_set():
        _supervisor_thread = None
        return
    _run_supervisor(settings)


def _preserve_unprocessed_updates(bridge, updates, consumed: int) -> int:
    """Best effort, never raises: hand the tail behind the ``consumed``-th update back to the bridge, ids
    intact, for this process's next read; what is not kept is logged as lost.
    Memory only: it dies with process exit; accepted rows outlive it."""
    tail, requeue, kept = list(updates[consumed:]), getattr(bridge, "requeue_updates", None), 0
    try:
        if tail and callable(requeue):
            kept = int(requeue(tail) or 0)
    except Exception as exc:
        log.error("Bridge %s hand-back raised: %s", type(bridge).__name__, exc, exc_info=True)
    if kept < len(tail):
        log.error("Bridge %s cannot take back %d unprocessed update(s); they are lost", type(bridge).__name__, len(tail) - kept)
    return kept


def _process_bridge_updates(bridge, offset: int, ctx: Any) -> int:
    updates = bridge.get_updates(offset=offset, timeout=1)
    cursor = [0]  # updates taken up so far, the one in flight included
    try:
        return _handle_bridge_update_batch(bridge, updates, offset, ctx, cursor)
    except Exception as exc:
        # The failing update is the crash the loop accounts for; the ones behind it were
        # only dequeued, never handled, and come back next tick (the hand-back never raises).
        failed = (updates[cursor[0] - 1] if 0 < cursor[0] <= len(updates) else {}).get("update_id")
        kept = _preserve_unprocessed_updates(bridge, updates, cursor[0])
        log.error("Bridge update %s failed: %s; %d later update(s) handed back to the bridge", failed, exc, kept)
        raise


def _handle_bridge_update_batch(bridge, updates, offset: int, ctx: Any, cursor: list) -> int:
    from supervisor.message_bus import coerce_chat_identity

    for upd in updates:
        cursor[0] += 1
        offset = int(upd["update_id"]) + 1
        msg = upd.get("message") or {}
        if not msg:
            continue
        # get_updates may return several already-queued messages. Rebind the
        # transport per message rather than using the last route in the batch.
        if hasattr(bridge, "activate_update_transport"):
            bridge.activate_update_transport(msg)

        chat_id = coerce_chat_identity((msg.get("chat") or {}).get("id"), 1)
        user_id = coerce_chat_identity((msg.get("from") or {}).get("id"), chat_id or 1)
        text = str(msg.get("text") or "")
        source = str(msg.get("source") or "web")
        sender_session_id = str(msg.get("sender_session_id") or "")
        client_message_id = str(msg.get("client_message_id") or "")
        transport = msg.get("transport") if isinstance(msg.get("transport"), dict) else {}
        image_base64 = str(msg.get("image_base64") or "")
        image_mime = str(msg.get("image_mime") or "image/jpeg")
        image_caption = str(msg.get("image_caption") or "")
        task_constraint = msg.get("task_constraint") if isinstance(msg.get("task_constraint"), dict) else None
        task_metadata = msg.get("task_metadata") if isinstance(msg.get("task_metadata"), dict) else None
        image_data = (image_base64, image_mime, image_caption) if image_base64 else None
        log_text = text or image_caption or ("(image attached)" if image_base64 else "(file attached)" if (task_metadata or {}).get("chat_attachment_uploads") else "")
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

        from supervisor.message_bus import record_inbound_message

        # The same writer mints ordinary ingress and validates preaccepted
        # skill deliveries; the latter already have their one canonical row.
        origin_message_ref = record_inbound_message(
            bridge, msg, chat_id=chat_id, user_id=user_id,
            client_message_id=client_message_id, text=log_text, ts=now_iso,
        )
        if task_metadata or msg.get("accepted_source_ref"):
            task_metadata = {k: v for k, v in (task_metadata or {}).items() if k != "_host_operation"}
            if msg.get("accepted_source_ref"):
                task_metadata["_host_operation"] = True
        reply_source = origin_message_ref if msg.get("accepted_source_ref") else None
        def reply(body: str, status: str = "completed") -> None:
            ctx.send_with_budget(chat_id, body, **host_operation_reply_kwargs(reply_source, status), role="system", system_type="command_reply")
        def _stamp_owner_activity(live: dict) -> None:
            if live.get("owner_id") is None and external_identity_present:
                live["owner_id"] = user_id
                live["owner_chat_id"] = _owner_binding_chat_id(ctx, chat_id, is_external_transport)
            live["last_owner_message_at"] = now_iso

        ctx.update_state(_stamp_owner_activity)

        if not text and not image_base64 and not (task_metadata or {}).get("chat_attachment_uploads"):
            continue

        if is_external_transport and is_slash_command:
            if not external_identity_present:
                reply("⚠️ Command ignored: this transport did not provide owner identity.", "failed")
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
                reply("✅ Owner chat registered. Send the command again to execute it.")
                continue
            try:
                owner_ext_id_int = int(owner_ext_id or 0)
                owner_ext_chat_id_int = int(owner_ext_chat_id or 0)
            except (TypeError, ValueError):
                owner_ext_id_int = 0
                owner_ext_chat_id_int = 0
            if owner_ext_id_int != user_id or owner_ext_chat_id_int != chat_id:
                reply("⚠️ Command ignored: this transport is not the bound owner chat.", "failed")
                continue

        if lowered.startswith("/panic"):
            reply("🛑 PANIC: killing everything. App will close.", "")
            # Never hand back a volatile tail before Panic: even a nonblocking
            # callback could perform I/O or delay the hard stop, and this memory
            # cannot survive the exit. Accepted ingress rows remain on disk.
            _execute_panic_stop(ctx.consciousness, ctx.kill_workers)
            return offset  # Never drain another already-queued message after Panic.
        elif lowered.startswith("/restart"):
            reply("♻️ Restarting.", "")
            ok, restart_msg = _perform_owner_restart(ctx, reply)
            if not ok:
                reply(f"⚠️ Restart cancelled: {restart_msg}", "failed")
                continue
            _preserve_unprocessed_updates(bridge, updates, cursor[0])  # best effort; this generation handles nothing more
            return offset  # Remaining accepted rows stay durable; no replay is promised.
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
                    reply(block, "failed")
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
                    reply("⚠️ Evolution stayed OFF: campaign state could not be created.", "failed")
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
            # An owner's stop carries no agent source (absent = owner-placed, sticky against
            # toggle_evolution); an owner's start drops a stale one with the flag.
            st2.pop("evolution_stop_source", None)
            # Owner-initiated evolution must not inherit a stale post-task one-shot
            # autostop, which would disable the owner's campaign after one cycle.
            st2["post_task_autostop"] = False
            ctx.save_state(st2)
            reply(f"🧬 Evolution campaign: {'ON' if turn_on else _owner_evolution_stop(ctx, chat_id)}")
        elif lowered.startswith("/bg"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "status"
            if action in ("start", "on", "1"):
                result = ctx.consciousness.start()
                _bg_s = ctx.load_state()
                _bg_s["bg_consciousness_enabled"] = True
                ctx.save_state(_bg_s)
                reply(f"🧠 {result}")
            elif action in ("stop", "off", "0"):
                result = ctx.consciousness.stop()
                _bg_s = ctx.load_state()
                _bg_s["bg_consciousness_enabled"] = False
                ctx.save_state(_bg_s)
                reply(f"🧠 {result}")
            else:
                described = _describe_bg_consciousness_state(bool(ctx.load_state().get("bg_consciousness_enabled")))
                reply(f"🧠 Background consciousness: {described['status']} — {described['detail']}")
        elif lowered.startswith("/status"):
            from supervisor.state import status_text

            status = status_text(ctx.WORKERS, ctx.PENDING, ctx.RUNNING)
            reply(status)
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
    if _LAUNCHER_MANAGED:
        try:
            from supervisor import git_ops as git_ops_module
            return git_ops_module.managed_branch_defaults(REPO_DIR)
        except Exception:
            pass
    return "ouroboros", "ouroboros-stable"


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

    if _launcher_managed_repo_matches():
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

    if _LAUNCHER_MANAGED:
        log.warning("Managed marker lacks matching repository identity; skipping destructive bootstrap for %s.", REPO_DIR)

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

    # Revival must drop the prior consciousness. Native turns own fresh agents.
    if _consciousness is not None:
        try:
            _consciousness.stop()
        except Exception:
            log.debug("Failed to stop previous consciousness instance", exc_info=True)
        _consciousness = None
    prior_worker_pids: set[int] | None = None
    try:
        ensure_legacy_imported(pathlib.Path(DATA_DIR))

        from supervisor.message_bus import LocalChatBridge, init as bus_init

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
        from supervisor.direct_roots import publish_direct_roots
        from supervisor.workers import (
            init as workers_init, get_event_q, WORKERS, PENDING, RUNNING,
            spawn_workers, kill_workers, assign_tasks, ensure_workers_healthy,
            handle_chat_direct, auto_resume_after_restart,
        )

        max_workers = int(settings.get("OUROBOROS_MAX_WORKERS", 10))

        # Managed manifest branch defaults must drive worker commit/restart flows too.
        _workers_branch_dev, _workers_branch_stable = _runtime_branch_defaults()
        workers_init(
            repo_dir=REPO_DIR, drive_root=DATA_DIR, max_workers=max_workers,
            branch_dev=_workers_branch_dev, branch_stable=_workers_branch_stable,
        )

        from supervisor.message_bus import send_with_budget
        from ouroboros.consciousness import BackgroundConsciousness
        import types

        _migrate_startup_cancel_latches(DATA_DIR)
        prior_worker_pids = _startup_worker_pids(DATA_DIR)
        interrupted_running: list = []
        restored_pending = restore_pending_from_snapshot(terminalized=interrupted_running)
        kill_workers(preserve_pending=True)
        spawn_workers(max_workers)
        from ouroboros.server_process import record_applied_restart_settings
        record_applied_restart_settings({"OUROBOROS_MAX_WORKERS": max_workers,
                                        "OUROBOROS_SKILLS_REPO_PATH": settings.get("OUROBOROS_SKILLS_REPO_PATH", "")})
        persist_queue_snapshot(reason="startup")
        try:
            from ouroboros.delegate_recovery import pre_adopt_planned_handoffs

            pre_adopt_planned_handoffs(DATA_DIR, list(PENDING))
        except Exception:
            log.debug("Planned delegate pre-adoption failed", exc_info=True)
        _startup_custody_sweep()
        recovered_files = _run_startup_task_recovery(
            DATA_DIR, REPO_DIR, skip_live_data=_pytest_default_real_data_dir,
            prior_worker_pids=prior_worker_pids,
        )
        _resume_interrupted_project_deletions()
        _startup_prune_sweeps(preserve_task_sources=bool(
            recovered_files["unresolved"] or recovered_files["protected"] or recovered_files["errors"]))
        _startup_worktree_prune()

        _prune_delegated_snapshots()

        if restored_pending > 0 or interrupted_running:
            st_boot = load_state()
            if st_boot.get("owner_chat_id"):
                # The second clause states an INTENT, not an outcome: restore only
                # fences an interrupted task with a durable cancel intent, and
                # cancellation custody writes its terminal result a watchdog
                # window later (task_lifecycle._INTENT_WATCHDOG_MIN_AGE_SEC).
                notice = ["♻️"]
                if restored_pending > 0:
                    notice.append(f"Restored pending queue from snapshot: {restored_pending} tasks.")
                if interrupted_running:
                    count = len(interrupted_running)
                    notice.append(
                        f"Cancelling {count} task{'' if count == 1 else 's'} that "
                        f"{'was' if count == 1 else 'were'} still running when the server stopped."
                    )
                send_with_budget(int(st_boot["owner_chat_id"]), " ".join(notice), role="system", system_type="startup_notice")
        _startup_retired_settings_notice(settings)

        auto_resume_after_restart()

        def _get_owner_chat_id() -> Optional[int]:
            try:
                return int((load_state() or {}).get("owner_chat_id") or 0) or None
            except Exception:
                return None

        _consciousness = BackgroundConsciousness(
            drive_root=DATA_DIR, repo_dir=REPO_DIR, owner_chat_id_fn=_get_owner_chat_id,
            routing_metadata_fn=lambda cid: main_lane_routing_metadata(_event_ctx, cid))  # _event_ctx is built below

        if load_state().get("bg_consciousness_enabled"):
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
            handle_chat_direct=handle_chat_direct,
            request_restart=_request_restart_exit,
        )
    except Exception as exc:
        _supervisor_error = f"Supervisor init failed: {exc}"
        _consciousness = None
        log.critical("Supervisor initialization failed", exc_info=True)
        try:
            # Provider-configured lifespan normally relies on this supervisor
            # owner for boot recovery. If initialization itself fails, keep the
            # same custody pass instead of serving with orphan RUNNING rows.
            recovery_pids = prior_worker_pids
            if recovery_pids is None:
                try:
                    recovery_pids = _startup_worker_pids(DATA_DIR)
                except Exception:
                    recovery_pids = None
            _run_startup_task_recovery(
                DATA_DIR, REPO_DIR, skip_live_data=_pytest_default_real_data_dir,
                prior_worker_pids=recovery_pids,
            )
        except Exception:
            log.critical("Startup recovery after supervisor initialization failure failed", exc_info=True)
        _supervisor_ready.clear()  # never reached its loop: the API must not paint Online over the error
        _supervisor_init_done.set()
        _supervisor_thread = None
        return

    _supervisor_ready.set()
    _supervisor_init_done.set()
    log.info("Supervisor ready.")
    _historical_audit.start(DATA_DIR, REPO_DIR)

    offset = 0
    crash_count = 0
    _last_custody_reap = [time.time()]
    _last_review_job_reconcile = [time.time()]
    # WS3: a dedicated watchdog thread (outside this loop, so it fires even if the
    # loop stalls) surfaces a wedge as an observable signal + owner alert instead
    # of silent hours; the loop publishes a liveness tick at each tick PHASE. The
    # tick is MONOTONIC: it is only ever read as an elapsed gap, so a wall-clock
    # jump must not turn a healthy loop into a phantom stall (nor hide a real one).
    from ouroboros.server_liveness import loop_phase_facts
    _loop_liveness = [time.monotonic(), {}, time.thread_time(), None]  # slots: server_liveness.py
    _watchdog_stop = threading.Event()  # per-generation: stops the watchdog when THIS loop exits
    _start_supervisor_liveness_watchdog(_loop_liveness, _watchdog_stop)
    while not _restart_requested.is_set() and not _supervisor_stop.is_set() and not _exit_signalled.is_set():
        try:
            _loop_liveness[1], _loop_liveness[0] = loop_phase_facts(_loop_liveness, "events", new_tick=True), time.monotonic()
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

            # One BOUNDED events batch (count + time; the remainder waits for the next
            # turn), so a producer that keeps the queue non-empty cannot hide intake.
            backlog = drain_worker_events(
                get_event_q(), _event_ctx, _loop_liveness, on_restart=_handle_restart_in_supervisor,
            )

            if _restart_requested.is_set():
                flush_budget_projection(_event_ctx)  # this turn's drained llm_usage still reaches state.json
                break

            # WS3: intake new bridge messages EARLY — before the heavy steps
            # (enforce_task_timeouts / assign_tasks / evolution) — so a later
            # blocking step can never starve new-message intake (the wedge class
            # where no task_received fired for hours until a full restart).
            offset = _process_bridge_updates(bridge, offset, _event_ctx)
            # The one budget-projection write of this turn (llm_usage events only mark it dirty).
            flush_budget_projection(_event_ctx)

            _loop_liveness[1], _loop_liveness[0] = loop_phase_facts(_loop_liveness, "maintenance"), time.monotonic()
            enforce_task_timeouts()
            try:
                from supervisor.queue import check_scheduled_tasks
                check_scheduled_tasks()
            except Exception:
                log.warning("Scheduled task check failed", exc_info=True)
            _periodic_supervisor_maintenance(
                _last_custody_reap, _last_review_job_reconcile, stop_event=_watchdog_stop,
                on_orphans_healed=lambda count: _consciousness and _consciousness.notify(f"orphans_healed:{count}"),
            )
            _loop_liveness[1], _loop_liveness[0] = loop_phase_facts(_loop_liveness, "assign"), time.monotonic()
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
            publish_direct_roots(_event_ctx.DRIVE_ROOT)
            if _consciousness is not None:
                try:
                    _consciousness.tick(time.time())
                except Exception:
                    log.warning("Consciousness alarm tick failed", exc_info=True)

            crash_count = 0
            if not backlog:
                time.sleep(0.5)  # a turn that hit its events bound drains the backlog at full speed

        except Exception as exc:
            if _supervisor_stop.is_set() or _restart_requested.is_set() or _exit_signalled.is_set():
                # A shutdown-torn Manager proxy is not a supervisor crash.
                log.info("Supervisor loop exiting on shutdown: %s", exc)
                break
            crash_count += 1
            log.error("Supervisor loop crash #%d: %s", crash_count, exc, exc_info=True)
            if crash_count >= 3:
                # Clear readiness and notify: a dead loop must not look healthy.
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
                            role="system", system_type="supervisor_failure")
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
            role="system", system_type="restart_notice")
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
                role="system", system_type="restart_notice")
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
                role="system", system_type="restart_notice")
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
                    role="system", system_type="restart_notice")
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
                    role="system", system_type="restart_notice")
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
            ctx.send_with_budget(int(st["owner_chat_id"]), f"⚠️ Restart skipped: {msg}", role="system", system_type="restart_notice")
        return
    cleanup_status, cleanup_reason = _shutdown_task_cleanup_args(restart_requested=True)
    global _planned_delegate_restart_transaction_id
    _planned_delegate_restart_transaction_id = ""
    planned_handoffs: set[str] = set()
    restart_transaction_id = uuid.uuid4().hex
    try:
        from ouroboros.delegate_recovery import prepare_planned_restart_handoffs
        from ouroboros.owner_wait import prepare_owner_wait_handoffs

        planned_handoffs = prepare_planned_restart_handoffs(
            ctx.DRIVE_ROOT, ctx.RUNNING,
            restart_transaction_id=restart_transaction_id,
            additional_task_ids=prepare_owner_wait_handoffs(
                ctx.DRIVE_ROOT, ctx.RUNNING, restart_transaction_id),
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
    _supervisor_init_done.wait()
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
                    send_with_budget(owner_chat, f"📦 Managed update: {stash_note}", role="system", system_type="managed_update_notice")
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

def _startup_owner_command(command: str):
    """Bind only process verbs while the normal command consumer is absent."""
    if command not in {"/panic", "/restart"}:
        return None

    def execute():
        # Onboarding may finish after HTTP admission. A newly live consumer
        # owns the ordinary command rather than two concurrent control paths.
        from supervisor.message_bus import try_get_bridge
        bridge = try_get_bridge()
        if _supervisor_thread and _supervisor_thread.is_alive() and bridge is not None:
            bridge.ui_send(command, broadcast=False)
            return
        from supervisor import state, workers, git_ops
        from types import SimpleNamespace
        state.init(DATA_DIR)
        if command == "/panic":
            _execute_panic_stop(_consciousness, workers.kill_workers)
            return
        branch_dev, branch_stable = _runtime_branch_defaults()
        git_ops.init(REPO_DIR, DATA_DIR, "", branch_dev, branch_stable)
        context = SimpleNamespace(safe_restart=git_ops.safe_restart,
                                  RUNNING=workers.RUNNING, kill_workers=workers.kill_workers)
        ok, message = _perform_owner_restart(context)
        if not ok:
            log.error("Startup owner restart cancelled: %s", message)

    return execute


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

from contextlib import ExitStack, asynccontextmanager, suppress


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

    if not _exit_signalled.is_set():
        _supervisor_stop.clear()  # a fresh lifespan owns a fresh generation (symmetric with the teardown set)
    if has_startup_ready_provider(settings):
        _start_supervisor_if_needed(settings)
    else:
        _supervisor_ready.set()
        _supervisor_init_done.set()
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

    if not pytest_default_real_data_dir:
        from ouroboros.claudexor_daemon import warm_owned_daemon
        warm_owned_daemon()  # provisioned homes only; one background ensure, off the startup path

    host_service_task = None
    host_service_server = None
    host_service_listener = ExitStack()
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
        # Bind before starting the asyncio task: uvicorn's bind-error SystemExit
        # otherwise escapes run_forever and kills the main server. Keep that
        # socket, removing the former probe-close/rebind race as well.
        host_socket = host_service_listener.enter_context(bound_service_socket(
            lifespan_drive_root, "host_service", DEFAULT_HOST_SERVICE_HOST, host_port))
        host_service_config = uvicorn.Config(
            host_service_app,
            host=DEFAULT_HOST_SERVICE_HOST,
            port=host_port,
            log_level="warning",
        )
        host_service_server = _embedded_uvicorn_server(host_service_config)
        host_service_task = asyncio.create_task(
            host_service_server.serve(sockets=[host_socket]),
            name="host-service-api",
        )
        host_service_task.add_done_callback(lambda _task: host_service_listener.close())
        log.info("Host Service API listening on %s:%d", DEFAULT_HOST_SERVICE_HOST, host_port)
    except Exception:
        host_service_listener.close()
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
    if not has_startup_ready_provider(settings):
        _run_startup_task_recovery(
            lifespan_drive_root, REPO_DIR, skip_live_data=pytest_default_real_data_dir,
            prior_worker_pids=None if pytest_default_real_data_dir else _startup_worker_pids(lifespan_drive_root),
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
        _historical_audit.stop()
        log.info("Server shutting down...")
        # Let the loop leave its current tick BEFORE workers are killed and the
        # bridge/Manager go down: a tick still running would otherwise respawn
        # a killed worker or meet BrokenPipe/EOF. Bounded well inside the
        # launcher's force-exit budget; the stop flag already suppresses the
        # crash counter if the join times out.
        supervisor_thread = _supervisor_thread
        if supervisor_thread is not None and supervisor_thread.is_alive():
            supervisor_thread.join(timeout=2)
        # Terminal custody FIRST: this is the teardown's one irreversible durable
        # write and every wait below it is best effort (ARCHITECTURE, Shutdown).
        try:
            restart_requested = _restart_requested.is_set()
            from supervisor.workers import kill_workers
            cleanup_status, cleanup_reason = _shutdown_task_cleanup_args(restart_requested)
            kill_workers(
                force=True,
                terminal_status=cleanup_status,
                result_reason=cleanup_reason,
                **_restart_cleanup_kwargs(),
                **_managed_update_pending_kwargs(),
            )
            # Record an explicit shutdown cause so a task interrupted by the shutdown is
            # never later read as a worker crash storm. Diagnostic, so it runs AFTER the
            # custody write: append_jsonl waits up to two seconds for the log lock, and
            # that wait must never spend the force-exit budget on unterminalized workers.
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
        except Exception:
            pass
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
        host_service_listener.close()
        ws_heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await ws_heartbeat_task

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
        if _restart_requested.is_set():
            try:
                # A planned restart whose landed checkout pins another engine ends
                # the owned daemon here so the next generation starts on that pin.
                _stop_owned_daemon_for_new_pin()
            except Exception:
                log.critical("Planned restart: engine pin check raised; the owned daemon is left serving",
                             exc_info=True)
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
app.app.state.startup_owner_command = _startup_owner_command
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


def _restart_cleanup_kwargs() -> dict:
    """Keep owner Restart from re-opening daemon custody after its stop."""
    if _owner_restart_requested.is_set():
        return {"reconcile_delegate_custody": False}
    return {}


def _emergency_process_cleanup(*, port_sweep: bool = True) -> None:
    """Kill child processes, workers, companions, and runtime port holders."""
    _historical_audit.stop()  # forced path may skip lifespan's finally; stop never waits
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
        cleanup_kwargs = _restart_cleanup_kwargs()
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
                **cleanup_kwargs,
                **_managed_update_pending_kwargs(),
            )
        else:
            kill_workers(
                force=True,
                archive_service_logs=False,
                **cleanup_kwargs,
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
    env_host = os.environ.get("OUROBOROS_SERVER_HOST", "").strip()
    default_host = env_host or saved_host or DEFAULT_HOST
    args = parse_server_args(default_host, DEFAULT_PORT)
    host_source = "cli" if args.host_explicit else (
        ("launcher" if _LAUNCHER_MANAGED else "environment") if env_host else "settings")
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
    config = uvicorn.Config(
        app,
        host=args.host,
        port=actual_port,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=20,
        # Bound the open HTTP/WS drain so the lifespan teardown (terminal custody) starts inside
        # the launcher's stop budget instead of leaving terminalization to the next boot (#1142).
        timeout_graceful_shutdown=SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC,
    )
    server = _SignalStopServer(config)
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
        with bound_service_socket(DATA_DIR, "main", args.host, actual_port,
                                  server_host_source=host_source) as listener:
            actual_port = _ACTUAL_BOUND_PORT = listener.getsockname()[1]
            write_port_file(PORT_FILE, actual_port)
            log.info("Starting Ouroboros server on %s:%d", args.host, actual_port)
            server.run(sockets=[listener])
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
