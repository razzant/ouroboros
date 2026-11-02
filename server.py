"""Self-editable Starlette/uvicorn entry point for UI and supervisor runtime."""

import asyncio
import base64  # noqa: F401
import json
import subprocess as subprocess
import os
import pathlib
import sys
import threading
import time
import uuid
# Before any import resolves a path: the executing checkout is the implicit repository and the data root is
# settled; an explicit value wins, and APP_ROOT alone never silently leaves historical memory (#1239).
from ouroboros.launch_roots import enter_direct, same_location  # noqa: E402 — standard library only
os.environ.update(enter_direct(os.path.dirname(os.path.abspath(__file__))))
from ouroboros.utils import read_json_dict as read_json_dict, utc_now_iso  # noqa: E402
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn
from ouroboros.server_control import (PanicIngress, execute_panic_stop as _execute_panic_stop_impl,
                                      exit_after_shutdown,
                                      restart_current_process as _restart_current_process_impl)  # noqa: F401 — restart lifecycle binding
from ouroboros.owned_shutdown import (begin_owned_stop, finish_unconfirmed_stops, start_inherited_import,
                                       stop_owned_work)
from ouroboros.server_auth import (
    NetworkAuthGate,
    get_network_auth_startup_warning,
    validate_network_auth_configuration,
)
from ouroboros.server_entrypoint import (ServerAdmissionRefused, admit_data_root, bound_service_socket,
                                         find_free_port, parse_server_args, reserved_service_ports, write_port_file)
from ouroboros.launcher_bootstrap import automatic_launch_allowed
from ouroboros import body_adoption, process_scope, server_boot, store_switch
from ouroboros.server_web import ServedUI
from ouroboros.config import resolve_port_file
from ouroboros.process_logging import configure_process_logging
from ouroboros.task_finalization import host_operation_reply_kwargs
from ouroboros import usage_store  # the boot import of the retired journal (run_startup_phase below)
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
    DATA_DIR, _owner_restart_requested,
    _request_restart_exit, _restart_requested,
    _supervisor_stop, _exit_signalled,
    _SignalStopServer, _embedded_uvicorn_server,
    capture_server_source_baseline, server_stop_cause, log,
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
    drain_worker_events, flush_budget_projection, publish_supervisor_load,
)
from ouroboros.server_liveness import BOOT, start_boot_coordinator, wait_event
from ouroboros.server_maintenance import (  # noqa: F401
    bind_body_lifespan,
    _LAST_CANCEL_INTENT_SWEEP,
    _migrate_startup_cancel_latches,
    _startup_worker_pids,
    _installed_skill_names,
    _periodic_supervisor_maintenance,
    _periodic_zombie_reconcile,
    _prune_delegated_snapshots,
    _prune_worktrees,
    _reconcile_delegated_runs,
    _resume_interrupted_project_deletions,
    _run_startup_task_recovery,
    _startup_retired_settings_notice,
    _startup_environment_review_notice,
    _startup_review_pool_notice,
    _startup_custody_sweep,
    _startup_prune_sweeps,
)
from ouroboros.ui_translation import start_background as _start_ui_translation
from supervisor.workspace_preparation import start_generation as _start_preparation_generation
from supervisor.event_observation import pending_idle_observations
from ouroboros import server_restart as _restart_lifecycle
from ouroboros.server_restart import (  # noqa: F401
    _begin_task_teardown, _live_running_task_ids, _managed_update_pending_kwargs,
    _perform_owner_restart, _safe_restart_serialized,
    _shutdown_supervisor_event_bus, _shutdown_task_cleanup_args,
    _stop_owned_daemon_for_new_pin, _stop_owned_work,
)

REPO_DIR = pathlib.Path(os.environ["OUROBOROS_REPO_DIR"])
capture_server_source_baseline(REPO_DIR)
DEFAULT_HOST = os.environ.get("OUROBOROS_SERVER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("OUROBOROS_SERVER_PORT", "8765"))
PORT_FILE = resolve_port_file(DATA_DIR, hosting_data_dir=DATA_DIR)

sys.path.insert(0, str(REPO_DIR))
if isinstance(sys.executable, str) and sys.executable:  # this host's interpreter, never an inherited (moved-app) one
    os.environ["OUROBOROS_AGENT_PYTHON"] = sys.executable

# Logging is configured in main(), not at import: a spawn/forkserver worker re-imports
# this module as ``__mp_main__`` and configures itself in ``worker_main``
# (ouroboros/process_logging.py), so importing the server never attaches handlers.
_pytest_default_real_data_dir = (
    "pytest" in sys.modules
    and not os.environ.get("OUROBOROS_DATA_DIR")
    and same_location(DATA_DIR, pathlib.Path.home() / "Ouroboros" / "data")
)

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
    return _restart_lifecycle._restart_current_process(sys.modules[__name__], host, port)

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
_bootstrap_ok: Optional[bool] = None  # this generation's `_bootstrap_supervisor_repo` outcome; None until it ran
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_supervisor_thread: Optional[threading.Thread] = None
_consciousness: Any = None
# This generation's extensions registered their event-bus subscribers: the supervisor's first tick
# (an overdue note) waits for it. Set outside a lifespan, cleared when one starts, set by its boot step.
_extensions_ready = threading.Event()
_extensions_ready.set()
_dialogue_bridge: Any = None  # the chat bridge published at the store's publication (`dialogue_ready`)
_dialogue_lock = threading.Lock()


from ouroboros.consciousness_view import _clock_of as _clock_of, describe_consciousness, describe_consciousness_state


def _describe_bg_consciousness_state(requested_enabled: bool | None) -> dict:
    return describe_consciousness_state(_consciousness.status_snapshot() if _consciousness else {}, requested_enabled)


def _start_supervisor_if_needed(settings: dict) -> bool:
    """Start the supervisor once when runtime providers become available."""
    global _supervisor_thread, _supervisor_error, _bootstrap_ok
    if not has_startup_ready_provider(settings):
        return False
    if _supervisor_thread and _supervisor_thread.is_alive():
        return False
    if _exit_signalled.is_set():
        return False  # the process is exiting: no revival behind the teardown
    _supervisor_error = None
    _bootstrap_ok = None
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
    if _exit_signalled.is_set() or not _await_extensions_ready():
        _supervisor_thread = None
        return
    _run_supervisor(settings)


def _await_extensions_ready(*stops: threading.Event) -> bool:
    """Wait until this generation's extensions registered their event-bus subscribers; False when
    the generation ends first. The supervisor's first scheduler tick may consume an overdue note
    and the bus keeps no history, so a subscriber attached later (Telegram) would miss it."""
    return wait_event(_extensions_ready, _exit_signalled, _supervisor_stop, _restart_requested, *stops)


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


def _process_bridge_updates(bridge, offset: int, ctx: Any, *, timeout: float = 1) -> int:
    updates = bridge.get_updates(offset=offset, timeout=timeout)
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
    from ouroboros.chat_uploads import attachment_placeholder

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
        log_text = text.strip() or image_caption.strip() or attachment_placeholder(image_base64, task_metadata)
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

        lowered = text.strip().lower()
        is_slash_command = lowered.startswith("/")
        is_external_transport = source != "web"
        external_identity_present = (not is_external_transport) or (chat_id > 0 and user_id > 0)
        # Emergency Stop Invariant (#1307): the local door stops before any state read,
        # chat record, state update or reply; an external transport needs a POSITIVELY
        # bound owner (one lock-free read) and never registers a stranger from an unknown slot.
        panic = lowered.startswith("/panic")
        if panic and not is_external_transport:
            _execute_panic_stop(ctx.consciousness, ctx.kill_workers)
            return offset  # Never drain another already-queued message after Panic.
        from supervisor.state import StateUnavailable, control_value

        st = ctx.load_state()
        if getattr(bridge, "panic", None) is not None:
            bridge.panic.observe_owner(st)
        from ouroboros.server_control import external_owner_binding

        ext_known, owner_ext_id, bound_pair = external_owner_binding(st)
        if panic and external_identity_present and ext_known and owner_ext_id is not None and (
                bound_pair == (user_id, chat_id)):
            _execute_panic_stop(ctx.consciousness, ctx.kill_workers)
            return offset

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
            ctx.send_with_budget(chat_id, body, **host_operation_reply_kwargs(reply_source, status, client_message_id), role="system", system_type="command_reply")
        def _stamp_owner_activity(live: dict) -> None:
            # Global owner = primary chat for outbound notices (web on desktop, the first
            # transport on headless Colab), bound once — only into a slot KNOWN to be empty.
            if control_value(live, "owner_id") == (True, None) and external_identity_present:
                live["owner_id"] = user_id
                live["owner_chat_id"] = _owner_binding_chat_id(ctx, chat_id, is_external_transport)
            live["last_owner_message_at"] = now_iso

        try:
            ctx.update_state(_stamp_owner_activity)
        except StateUnavailable as exc:
            log.warning("Owner activity not stamped: %s", exc)  # the message itself still proceeds

        if not text and not image_base64 and not (task_metadata or {}).get("chat_attachment_uploads"):
            continue

        if is_external_transport and is_slash_command:
            if not external_identity_present:
                reply("⚠️ Command ignored: this transport did not provide owner identity.", "failed")
                continue
            if not ext_known:
                reply("⚠️ Command ignored: the bound owner chat is unknown right now (runtime state "
                      "is unavailable or recovering), so no command runs and no chat is registered.", "failed")
                continue
            if owner_ext_id is None:
                bound = {"ok": False}

                def _bind_external_owner(live: dict) -> None:
                    if control_value(live, "owner_external_id") == (True, None):
                        live["owner_external_id"] = user_id
                        live["owner_external_chat_id"] = chat_id
                        live["owner_external_bound_at"] = now_iso
                        bound["ok"] = True

                try:
                    saved_owner = ctx.update_state(_bind_external_owner)
                    if getattr(bridge, "panic", None) is not None:
                        bridge.panic.observe_owner(saved_owner)
                except StateUnavailable:
                    pass
                reply("✅ Owner chat registered. Send the command again to execute it." if bound["ok"] else
                      "⚠️ Command ignored: the owner chat could not be registered right now.",
                      "completed" if bound["ok"] else "failed")
                continue
            if bound_pair != (user_id, chat_id):
                reply("⚠️ Command ignored: this transport is not the bound owner chat.", "failed")
                continue

        if lowered.startswith("/restart"):
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
            # `/review <row>` names the executor (the Web selector sends it);
            # a bare `/review` (Telegram) runs on the Main model.
            parts = text.split(None, 1)
            ctx.queue_deep_self_review_task(reason="owner:/review", force=True, chat_id=chat_id, client_message_id=client_message_id,
                                            reviewer=parts[1].strip() if len(parts) > 1 else "", origin={"work_kind": "owner"})
        elif lowered.startswith("/evolve"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "on"
            turn_on = action not in ("off", "stop", "0")
            objective = ""
            if turn_on and len(parts) > 2:
                objective = text.split(None, 2)[2].strip()
            if turn_on:
                from supervisor.events_runtime_controls import owner_evolution_start

                refusal = owner_evolution_start(objective, source="owner_chat")
                reply(refusal or "🧬 Evolution campaign: ON", "failed" if refusal else "completed")
            else:
                reply(f"🧬 Evolution campaign: {_owner_evolution_stop(ctx, chat_id)}")
        elif lowered.startswith("/bg"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "status"
            if action in ("start", "on", "1", "stop", "off", "0"):
                from supervisor.events_runtime_controls import persist_consciousness_choice

                on = action in ("start", "on", "1")
                result = ctx.consciousness.start() if on else ctx.consciousness.stop()
                reply(f"🧠 {result}{persist_consciousness_choice(on)}")
            else:
                from supervisor.state import control_value

                known, enabled = control_value(ctx.load_state(), "bg_consciousness_enabled")
                described = _describe_bg_consciousness_state(bool(enabled) if known else None)
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
                    "source": source, "received_at": str(msg.get("received_at") or ""),
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

    # This tree is inside an adoption transition (landed, returned or still open): never reset it.
    adoption_boot = body_adoption.holds_checkout(DATA_DIR, REPO_DIR)
    if _launcher_managed_repo_matches() and not adoption_boot:
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
        from ouroboros.server_restart import _noticed_bootstrap  # its failure, not a deferral, is told
        ok, msg = _safe_restart_serialized(
            _noticed_bootstrap(git_ops_module.safe_restart),
            reason="bootstrap",
            unsynced_policy=policy,
        )
        if not ok and policy == "rescue_and_block":
            try:
                from supervisor.evolution_lifecycle import disable_evolution_projection, pause_evolution_campaign

                disable_evolution_projection()
                pause_evolution_campaign(f"bootstrap blocked to protect active evolution transaction: {msg}")
            except Exception:
                log.debug("Failed to pause evolution after blocked bootstrap", exc_info=True)
        return ok, msg

    if _LAUNCHER_MANAGED and not adoption_boot:
        log.warning("Managed marker lacks matching repository identity; skipping destructive bootstrap for %s.", REPO_DIR)

    log.info("Local-dev server start detected — skipping bootstrap git reset.")
    deps_ok, deps_msg = git_ops_module.sync_runtime_dependencies(reason="bootstrap_local_dev")
    if not deps_ok:
        return False, f"Failed local-dev deps sync: {deps_msg}"

    import_result = git_ops_module.import_test()
    if import_result.get("ok"):
        return True, "OK: local-dev bootstrap"
    return False, f"Local-dev import test failed (rc={import_result.get('returncode', -1)})"


def _initialize_runtime_state(settings: dict, *, stop_requested=None) -> None:
    """Initialize before ingress; unavailable controls stay unknown while independent work continues."""
    from supervisor.state import init as state_init, init_state

    state_init(DATA_DIR, float(settings.get("TOTAL_BUDGET", SETTINGS_DEFAULTS["TOTAL_BUDGET"])),
               stop_requested=stop_requested)
    boot_state = init_state()
    if boot_state.quality not in {"current", "recovered"}:
        log.critical("Runtime state is %s (%s): owner binding, evolution and consciousness "
                     "controls stay unknown until it is readable", boot_state.quality, boot_state.reason)


def _dialogue_cutover(root: pathlib.Path) -> None:
    """The dialogue store takes over the live chat/progress files once per root (recent client ids
    dedup before the first accepted message); a failure is this step's, and the dialogue still opens:
    its writer seam repeats an unfinished cutover before it stores a row, and readers answer
    "still importing" until one completes."""
    from ouroboros.dialogue_import import cutover

    BOOT.step_started("dialogue_cutover")
    try:
        cutover(root, initialize_journal=True)  # the boot alone chooses an existing file's journal
    except Exception as exc:
        log.warning("Dialogue cutover did not complete; the first stored row repeats it", exc_info=True)
        BOOT.step_ended("dialogue_cutover", error=f"{type(exc).__name__}: {exc}")
        return
    BOOT.step_ended("dialogue_cutover")


def _open_dialogue(settings: dict, *, stop_requested=None):
    """Publish the chat bridge and bus once per process: from then on an owner message is
    accepted durably (its canonical chat row is written and the turn waits for its executor),
    which is ``dialogue_ready``. The boot runs this at the store's publication, before any
    repair; the supervisor attaches to the published bridge instead of creating a second one
    (a direct supervisor start without a boot creates it here)."""
    global _dialogue_bridge
    from supervisor.message_bus import LocalChatBridge, init as bus_init, try_get_bridge

    with _dialogue_lock:
        if _dialogue_bridge is not None and try_get_bridge() is _dialogue_bridge:
            return _dialogue_bridge
        _initialize_runtime_state(settings, stop_requested=stop_requested)
        from supervisor.state import load_state

        bridge = LocalChatBridge(settings)
        bridge.startup_owner_command = _startup_owner_command
        bridge.panic = PanicIngress(_startup_owner_command("/panic"))
        bridge.panic.observe_owner(load_state())
        bridge._broadcast_fn = broadcast_ws_sync

        from ouroboros.utils import set_log_sink
        from supervisor.events import make_server_log_sink

        set_log_sink(make_server_log_sink(bridge, pathlib.Path(DATA_DIR)))

        bus_init(
            drive_root=DATA_DIR, budget_report_every=10, chat_bridge=bridge,
            total_budget_limit=float(settings.get("TOTAL_BUDGET", SETTINGS_DEFAULTS["TOTAL_BUDGET"])),
        )
        _dialogue_bridge = bridge
        BOOT.dialogue_ready()
        log.info("Dialogue ready: owner messages are accepted.")
        return bridge


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
    _watchdog_stop = threading.Event()  # per-generation: set on EVERY exit of this generation
    try:
        BOOT.step_started("supervisor")  # tasks_ready is this generation's; its sub-steps follow
        # Watch startup stalls; even a failed watchdog start publishes an init outcome.
        from ouroboros.server_liveness import loop_phase_facts, note_supervisor_ready, run_startup_phase
        _loop_liveness = [time.monotonic(), {}, time.thread_time(), None]  # slots: server_liveness.py
        _loop_liveness[1], _loop_liveness[0] = loop_phase_facts(_loop_liveness, "startup", new_tick=True), time.monotonic()
        _start_supervisor_liveness_watchdog(_loop_liveness, _watchdog_stop)
        run_startup_phase(_loop_liveness, "startup:usage_store", lambda: usage_store.migrate_from_journal(pathlib.Path(DATA_DIR)))
        run_startup_phase(_loop_liveness, "startup:current_store", lambda: store_switch.require_published(DATA_DIR))  # published before any worker or recovery write
        from supervisor.state import control_is, load_state, save_state, update_state
        from supervisor.state import append_jsonl, update_budget_from_usage, rotate_jsonl_log_if_needed
        # The dialogue opened at the store's publication; this generation attaches to it.
        bridge = _open_dialogue(settings, stop_requested=lambda stop=_watchdog_stop: any(e.is_set() for e in (stop, _supervisor_stop, _restart_requested, _exit_signalled)))

        from supervisor.git_ops import safe_restart
        ok, msg = run_startup_phase(_loop_liveness, "startup:bootstrap", lambda: _bootstrap_supervisor_repo(settings))
        global _bootstrap_ok
        _bootstrap_ok = bool(ok)
        if not ok:
            log.error("Supervisor bootstrap failed: %s", msg)
        from ouroboros.store_import import note_current_stamp
        note_current_stamp(DATA_DIR)  # the stamp the bootstrap just wrote: the legacy detector's observation

        from supervisor.queue import (
            enqueue_task, enforce_task_timeouts, enqueue_evolution_task_if_needed,
            persist_queue_snapshot, restore_pending_from_snapshot,
            queue_deep_self_review_task, sort_pending,
        )
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

        prior_worker_pids = _startup_worker_pids(DATA_DIR)
        interrupted_running: list = []
        restored_pending = restore_pending_from_snapshot(terminalized=interrupted_running)
        _start_preparation_generation(stop=lambda stop=_watchdog_stop: any(e.is_set() for e in (stop, _supervisor_stop, _restart_requested, _exit_signalled)))
        kill_workers(preserve_pending=True)
        spawn_workers()
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
        recovered_files = run_startup_phase(_loop_liveness, "startup:task_recovery", lambda: _run_startup_task_recovery(
            DATA_DIR, REPO_DIR, skip_live_data=_pytest_default_real_data_dir,
            prior_worker_pids=prior_worker_pids,
        ))
        _resume_interrupted_project_deletions()
        _startup_prune_sweeps(preserve_task_sources=bool(
            recovered_files["unresolved"] or recovered_files["protected"] or recovered_files["errors"]), recovery_report=recovered_files)

        if restored_pending > 0 or interrupted_running:
            st_boot = load_state()
            if st_boot.get("owner_chat_id"):
                # The second clause states an INTENT, not an outcome: restore only
                # fences an interrupted task with a durable cancel intent, and
                # cancellation custody writes its terminal result a watchdog
                # window later (task_lifecycle._INTENT_WATCHDOG_MIN_AGE_SEC).
                notice = ["♻️"]
                if restored_pending > 0:
                    notice.append(f"Restored pending queue: {restored_pending} tasks.")
                if interrupted_running:
                    count = len(interrupted_running)
                    notice.append(
                        f"Settling {count} task{'' if count == 1 else 's'} that "
                        f"{'was' if count == 1 else 'were'} still running when the server stopped."
                    )
                send_with_budget(int(st_boot["owner_chat_id"]), " ".join(notice), role="system", system_type="startup_notice")
        _startup_retired_settings_notice(settings)
        _startup_environment_review_notice()
        _startup_review_pool_notice(settings)
        store_switch.startup_notice(DATA_DIR)
        from ouroboros.upgrade_notices import startup_upgrade_notices
        startup_upgrade_notices(settings)

        auto_resume_after_restart()

        def _get_owner_chat_id() -> Optional[int]:
            try:
                return int((load_state() or {}).get("owner_chat_id") or 0) or None
            except Exception:
                return None

        _consciousness = BackgroundConsciousness(
            drive_root=DATA_DIR, repo_dir=REPO_DIR, owner_chat_id_fn=_get_owner_chat_id,
            routing_metadata_fn=lambda cid: main_lane_routing_metadata(_event_ctx, cid))  # _event_ctx is built below

        # A boot grant needs a KNOWN True and no unconsumed Panic flag (#1307).
        from ouroboros.installation_stop import installation_panic
        if control_is(load_state(), "bg_consciousness_enabled", True) and not installation_panic(DATA_DIR):
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
            enqueue_task=enqueue_task,
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
        BOOT.step_ended("supervisor", error=_supervisor_error)
        _supervisor_thread = None
        _watchdog_stop.set()  # a generation that died in init has no loop to watch
        return

    _supervisor_ready.set()
    _supervisor_init_done.set()
    BOOT.step_ended("supervisor")
    note_supervisor_ready()

    offset = 0
    crash_count = 0
    _last_custody_reap = [0.0]  # recovered checking copies become eligible off-loop after readiness
    _last_review_job_reconcile = [time.time()]
    # The watchdog was started before startup recovery; never start another here.

    while not _restart_requested.is_set() and not _supervisor_stop.is_set() and not _exit_signalled.is_set():
        try:
            _loop_liveness[1], _loop_liveness[0] = loop_phase_facts(_loop_liveness, "events", new_tick=True), time.monotonic()
            # The dialogue chains (chat, progress) live in state/dialog.sqlite and were rotated one last
            # time by the cutover; the remaining unbounded hot logs rotate on this tick with the one rotator.
            # events.jsonl went LAST behind chain-aware custody readers (the store import's custody fold, legacy usage
            # import, swarm rollup, worker boot verify) — rows keep replaying from archive/events_*.jsonl; tools/supervisor/
            # task_reflections readers are tail-bounded and archive-backfilled (memory.read_jsonl_tail, api_logs_tail).
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
            # where no task_received fired for hours until a full restart). A drain that stopped
            # at its bound only polls: the idle wait would steal its backlog's second.
            offset = _process_bridge_updates(bridge, offset, _event_ctx, timeout=0 if backlog else 1)
            # The one budget-projection write of this turn (llm_usage events only mark it dirty).
            flush_budget_projection(_event_ctx)

            _loop_liveness[1], _loop_liveness[0] = loop_phase_facts(_loop_liveness, "maintenance"), time.monotonic()
            enforce_task_timeouts(pending_events=pending_idle_observations(get_event_q()))
            _periodic_supervisor_maintenance(  # includes the off-loop schedule duty
                _last_custody_reap, _last_review_job_reconcile, stop_event=_watchdog_stop,
            )
            _loop_liveness[1], _loop_liveness[0] = loop_phase_facts(_loop_liveness, "assign"), time.monotonic()
            # The requester's finalization starts the drain; its accepted intent
            # already holds independent starts. Existing trees still finish.
            draining = _check_pending_restart_drain(_event_ctx)
            if not draining and not _restart_lifecycle.agent_restart_pending():
                try:
                    from ouroboros.post_task_evolution import apply_pending_request
                    from supervisor import state as _pte_state

                    apply_pending_request(_pte_state.DRIVE_ROOT)
                except Exception:
                    log.debug("Post-task evolution apply failed", exc_info=True)
                enqueue_evolution_task_if_needed()
            if _restart_requested.is_set():
                break  # restart just triggered (drain done) — exit without assigning new work (bridge intake already ran early this iteration)
            assign_tasks()
            if persist_queue_snapshot(reason="main_loop") is False:  # None: a saved queue kept as it is
                raise RuntimeError("queue publication failed")
            if _consciousness is not None and not draining and not _restart_lifecycle.agent_restart_pending():
                try:
                    _consciousness.tick(time.time())
                except Exception:
                    log.warning("Consciousness alarm tick failed", exc_info=True)

            crash_count = 0
            publish_supervisor_load(get_event_q(), DATA_DIR)  # one row a minute, never under Q; never raises
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
    return _restart_lifecycle._handle_restart_in_supervisor(sys.modules[__name__], evt, ctx)


def _check_pending_restart_drain(ctx: Any) -> bool:
    return _restart_lifecycle._check_pending_restart_drain(sys.modules[__name__], ctx)


def _perform_supervisor_restart(ctx: Any, *, restart_reason="agent_restart_request", evolution_restart=False) -> None:
    return _restart_lifecycle._perform_supervisor_restart(
        sys.modules[__name__], ctx, restart_reason=restart_reason, evolution_restart=evolution_restart)


def _wait_for_supervisor_update_finalize() -> bool:
    """Wait for a real init outcome; slow dependency sync is not a failed boot."""
    _supervisor_init_done.wait()
    return not bool(_supervisor_error)


def _boot_managed_update_tasks() -> None:
    """Finalize a pending update, restart after rollback, then refresh its feed."""
    try:
        from ouroboros.gateway.update_progress import observe_boot
        from supervisor.git_ops import advance_recovery_point, compute_managed_update_status
        from supervisor.update_merge import active_update_tx, finalize_managed_update_on_boot

        ready = _wait_for_supervisor_update_finalize()
        # Observation only: the finalizer's result and every decision below stay its own.
        result = observe_boot(finalize_managed_update_on_boot, ready=ready, bootstrap_ok=_bootstrap_ok is True)
        # A failed bootstrap (dependency sync or the import test of this tree) is not a
        # ready generation even when the supervisor thread survived it: the adoption
        # of a tree that does not import is returned, not finalized.
        body_adoption.settle_on_boot(DATA_DIR, REPO_DIR, supervisor_ready=ready and _bootstrap_ok is True)
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
        if result.get("rolled_back") is True and active_update_tx():
            # Completed restore custody survives re-exec; the restored generation
            # clears it without another checkout or restart.
            from supervisor.workers import close_repo_writer_admission

            close_repo_writer_admission("managed_update:rollback_restart")
            _request_restart_exit()
            return
        advance_recovery_point(ready and _bootstrap_ok is True and result.get("rolled_back") is not True)
        update_status = compute_managed_update_status(fetch=True)
        try:
            from ouroboros.update_letter import refresh_after_check, runtime_status

            refresh_after_check(runtime_status(update_status))
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
    )

def _startup_owner_command(command: str, *, send_kwargs=None, source="web", user_id=0, chat_id=0, reply=None):
    """Bind the emergency owner directly; startup Restart retains normal routing."""
    verb = str(command).strip().lower()
    if verb not in {"/panic", "/restart"}:
        return None
    from supervisor.message_bus import try_get_bridge

    if (verb == "/restart" and _supervisor_thread and _supervisor_thread.is_alive()
            and _supervisor_ready.is_set() and try_get_bridge() is not None):
        return None  # Preserve the ready transport's ordinary metadata/acceptance path.

    def execute():
        from supervisor import workers

        if verb == "/panic":
            _execute_panic_stop(_consciousness, workers.kill_workers)
            return True
        if source != "web":
            from ouroboros.server_control import external_owner_binding
            from supervisor.state import load_state

            known, owner, pair = external_owner_binding(load_state())
            if not (known and owner is not None and user_id > 0 and chat_id > 0 and pair == (user_id, chat_id)):
                return False  # Existing intake still owns refusal or first-time binding.
        # Onboarding may finish after HTTP admission. A newly live consumer
        # owns the ordinary command rather than two concurrent control paths.
        bridge = try_get_bridge()
        if (_supervisor_thread and _supervisor_thread.is_alive()
                and _supervisor_ready.is_set() and bridge is not None):
            if source == "web" and not (send_kwargs or {}).get("accepted_source_ref"):
                bridge.ui_send(command, **({"broadcast": False, **(send_kwargs or {})}))
            else:
                bridge.enqueue_local_message(command, source=source, user_id=user_id,
                                             chat_id=chat_id, **(send_kwargs or {}))
            return True
        from supervisor import state, git_ops
        from types import SimpleNamespace
        state.init(DATA_DIR)
        branch_dev, branch_stable = _runtime_branch_defaults()
        git_ops.init(REPO_DIR, DATA_DIR, "", branch_dev, branch_stable)
        context = SimpleNamespace(safe_restart=git_ops.safe_restart,
                                  RUNNING=workers.RUNNING, kill_workers=workers.kill_workers)
        if reply is not None:
            reply("♻️ Restarting.", "")
        ok, message = _perform_owner_restart(context, reply)
        if not ok:
            log.error("Startup owner restart cancelled: %s", message)
            if reply is not None:
                reply(f"⚠️ Restart cancelled: {message}", "failed")
        return True

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

served_ui = ServedUI(REPO_DIR)  # `/` and `/static`; the lifespan applies OUROBOROS_UI_DIR per generation
index_page = served_ui.index_page

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
    *served_ui.mounts(),  # public selected /auth, builtin /onboarding/static, selected /static last
]

from contextlib import ExitStack, asynccontextmanager, suppress
from types import SimpleNamespace


@asynccontextmanager
async def lifespan(app):
    """Serve first: before the yield only what admits the data root, applies the settings and the
    served UI and opens the store's import window; everything whose cost depends on the data, the
    installed extensions or child processes is a named step of the boot coordinator, which runs
    on its own thread while the socket already answers (``/api/health`` carries the phases)."""
    global _event_loop, _dialogue_bridge
    BOOT.reset()
    _extensions_ready.clear()  # this generation's extensions have not registered yet
    _supervisor_ready.clear()  # readiness is this generation's: its recovery or its supervisor proves it
    _dialogue_bridge = None
    lifespan_drive_root = pathlib.Path(getattr(getattr(app, "state", None), "drive_root", DATA_DIR))
    default_real_data_dir = pathlib.Path.home() / "Ouroboros" / "data"
    pytest_default_real_data_dir = (
        (bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules)
        and same_location(lifespan_drive_root, default_real_data_dir)
        and not os.environ.get("OUROBOROS_DATA_DIR")
    )
    if not pytest_default_real_data_dir:  # an imported ASGI app is admitted here; main() already was
        admit_data_root(lifespan_drive_root)
        process_scope.initialize_installation_scope(lifespan_drive_root)  # idempotent after main(); first for an imported app
    _event_loop = asyncio.get_running_loop()
    _set_ws_event_loop(_event_loop)
    ws_heartbeat_task = asyncio.create_task(
        ws_heartbeat_loop(_has_ws_clients, broadcast_ws),
        name="ws-heartbeat",
    )

    # Boot APPLIES the provider normalization in-process and persists none of it
    # (mirror of launcher_onboarding.prepare_first_run_settings); its read does
    # persist the two compatibility upgrades (``migrate=True``). Two
    # normalizations, two re-derivations: the VOCABULARY one is the read seam
    # every reader applies (config.normalize_settings_raw); the PROVIDER one is
    # re-derived by every consumer that needs the route — the task-start
    # projection, the settings GET and onboarding reads, the context-fit route
    # resolver — so writing it at start would only make boot a second author.
    settings, _provider_defaults_changed, _provider_default_keys = apply_runtime_provider_defaults(load_settings(migrate=True))
    _apply_settings_to_env(settings)
    served_ui.apply(app, settings)  # this generation's main interface, before any request is served
    # Pin boot-time runtime-mode after env apply; save_settings compares to this owner baseline.
    from ouroboros.config import initialize_runtime_mode_baseline
    initialize_runtime_mode_baseline()
    has_local = needs_local_model_autostart(settings)

    if not pytest_default_real_data_dir:
        store_switch.start(lifespan_drive_root, repo_dir=REPO_DIR, request_restart=_request_restart_exit)  # a pending switch, then the import owner off the ready path (P1)

    if not _exit_signalled.is_set():
        _supervisor_stop.clear()  # a fresh lifespan owns a fresh generation (symmetric with the teardown set)
    body_stop = threading.Event()  # lifespan ownership survives a cognitive supervisor revival; the boot coordinator's stop
    start_body = (lambda: None) if pytest_default_real_data_dir else bind_body_lifespan(
        lifespan_drive_root, body_stop, collection_ready=_supervisor_ready.is_set)
    # A provider-ready boot starts the supervisor once the store published and the extensions registered.
    startup_provider_ready = has_startup_ready_provider(settings)
    if not startup_provider_ready:  # readiness follows the store's publication (store_switch.when_ready below)
        _supervisor_init_done.set()
        log.info("No supported provider or local routing configured. Supervisor not started.")

    boot = SimpleNamespace(host_service_task=None, host_service_server=None, host_service_listener=ExitStack(),
                           extension_reconcile_task=None)

    # --- at the store's publication (on the import thread): the dialogue first, repairs after ---
    def _dialogue() -> None:
        """Owner messages are accepted durably from here on (T2: a new chat before old repairs). A
        providerless generation has no executor for them: its dialogue opens with the supervisor a
        later settings save starts."""
        if startup_provider_ready and not body_stop.is_set() and not pytest_default_real_data_dir:
            _dialogue_cutover(lifespan_drive_root)
            _open_dialogue(settings, stop_requested=lambda: any(
                e.is_set() for e in (body_stop, _supervisor_stop, _restart_requested, _exit_signalled)))

    recovery = None
    if not startup_provider_ready:  # after the store's publication: the recovery writes, then readiness (P1)
        recovery = lambda: not body_stop.is_set() and _run_startup_task_recovery(  # noqa: E731
            lifespan_drive_root, REPO_DIR, skip_live_data=pytest_default_real_data_dir,
            prior_worker_pids=None if pytest_default_real_data_dir else _startup_worker_pids(lifespan_drive_root),
        )

    def _after_extensions() -> None:
        """The body runtime and the supervisor (or the providerless recovery and readiness), only
        once the extensions registered their bus subscribers: the first tick may consume an
        overdue note, and a subscriber attached later never sees it."""
        if body_stop.is_set() or not _await_extensions_ready(body_stop):
            return
        start_body()
        if recovery is not None:
            recovery()
            _supervisor_ready.set()
        else:
            _start_supervisor_if_needed(settings)

    store_switch.when_ready(lifespan_drive_root, _dialogue, _after_extensions)  # never on this thread: it starts the coordinator next

    # --- the boot coordinator's steps, in the order the lifespan ran them before it served ---
    live = not pytest_default_real_data_dir  # the real data root: every step; a bare pytest run: none that touches it

    def _usage_store() -> None:  # the one journal import, on every door (providerless included)
        if not live:
            return
        try:
            usage_store.migrate_from_journal(lifespan_drive_root, initialize_journal=True)
        except Exception:
            log.critical("Usage store import failed at startup; money reads report it unavailable "
                         "until it succeeds", exc_info=True)
            raise

    def _extensions() -> None:  # reload enabled+reviewed extensions across restarts
        try:
            from ouroboros.config import get_skills_repo_path, load_settings as _load_settings
            from ouroboros.extension_loader import reload_all as _reload_extensions, set_ws_broadcaster as _set_extension_ws_broadcaster
            _set_extension_ws_broadcaster(broadcast_ws_sync)
            repo_path = get_skills_repo_path()
            if not live:
                log.info("Skipping extension reload_all against real DATA_DIR during pytest")
            else:
                _reload_extensions(lifespan_drive_root, _load_settings, repo_path=repo_path or None)
        finally:
            _extensions_ready.set()  # loaded or not, the subscribers that exist are registered: the supervisor may start

    async def _start_extension_reconcile() -> None:
        from ouroboros.config import get_skills_repo_path, load_settings as _load_settings
        from ouroboros.extension_reconcile_queue import extension_reconcile_pickup_loop

        boot.extension_reconcile_task = asyncio.create_task(
            extension_reconcile_pickup_loop(
                lifespan_drive_root,
                _load_settings,
                repo_path_getter=lambda: get_skills_repo_path() or None,
            ),
            name="extension-reconcile-pickup",
        )

    steps = [
        ("owned_stops", lambda: live and finish_unconfirmed_stops(lifespan_drive_root)),  # before any extension/replacement process (§9)
        ("inherited_records", lambda: live and start_inherited_import(lifespan_drive_root)),  # the one-time disk walk, off the ready path
        ("usage_store", _usage_store),
        ("skills", lambda: live and server_boot.seed_skills(lifespan_drive_root, settings, provider_ready=startup_provider_ready)),
        ("projects", lambda: live and server_boot.reconcile_projects(lifespan_drive_root)),
        ("log_listings", lambda: live and server_boot.remember_log_listings(lifespan_drive_root)),
        ("services", lambda: server_boot.start_background_services(
            settings, local_model=has_local, boot_managed_update=_boot_managed_update_tasks, warm_daemon=live)),
        ("host_service", lambda: server_boot.on_loop(_event_loop, server_boot.serve_host_service(lifespan_drive_root, boot))),
        ("review_jobs", lambda: live and server_boot.reconcile_review_jobs(lifespan_drive_root)),
        ("extensions", _extensions),
        ("ui_translation", lambda: live and _start_ui_translation(lifespan_drive_root, _supervisor_stop)),  # after the skills registered their tables
        ("mcp", server_boot.mcp_startup),
        ("extension_reconcile", lambda: live and server_boot.on_loop(_event_loop, _start_extension_reconcile())),
        ("store", lambda: live and server_boot.wait_store_published(lifespan_drive_root, body_stop)),
        ("supervisor", lambda: server_boot.wait_supervisor_outcome(_supervisor_ready, lambda: _supervisor_error, body_stop)),
    ]
    start_boot_coordinator(steps, stop=body_stop)
    BOOT.serving()

    try:
        yield
    finally:
        _supervisor_stop.set()  # first: the loop must know a teardown owns what follows
        body_stop.set()  # the body runtime and the boot coordinator stop with the generation
        from ouroboros.delegate_collection import stop_body_runtime
        stop_body_runtime(lifespan_drive_root, wait=False)
        from ouroboros.maintenance import child as maintenance_child
        maintenance_child.stop()
        begin_owned_stop(lifespan_drive_root, cause=server_stop_cause())  # the grace starts here; cause and pending stops are recorded before any wait
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
            cleanup_status, cleanup_reason = _begin_task_teardown(restart_requested)
            kill_workers(
                force=True,
                terminal_status=cleanup_status,
                result_reason=cleanup_reason,
                stop_source="server_shutdown",
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
                        "cause": server_stop_cause(restart_requested),
                        "restart_exit": restart_requested,
                    },
                )
            except Exception:
                log.debug("Failed to record server_shutdown event", exc_info=True)
        except Exception:
            pass
        if _restart_requested.is_set():
            try:
                _stop_owned_daemon_for_new_pin()
            except Exception:
                log.critical("Planned restart: engine pin check raised; the owned daemon is left serving",
                             exc_info=True)
        extension_reconcile_task = boot.extension_reconcile_task
        if extension_reconcile_task is not None:
            extension_reconcile_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(extension_reconcile_task, timeout=30)
        if boot.host_service_server is not None:
            try:
                boot.host_service_server.should_exit = True
            except Exception:
                pass
        host_service_task = boot.host_service_task
        if host_service_task is not None:
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(host_service_task, timeout=5)
            if not host_service_task.done():
                host_service_task.cancel()
                with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    await asyncio.wait_for(host_service_task, timeout=2)
        boot.host_service_listener.close()
        ws_heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await ws_heartbeat_task

        try:
            from ouroboros.local_model import get_manager
            get_manager().stop_server()
        except Exception:
            pass
        stop_owned_work(lifespan_drive_root)  # the generation's one bounded stop; a started one is joined
        try:
            from ouroboros.extension_companion import get_global_supervisor
            supervisor = get_global_supervisor()
            if supervisor is not None:
                supervisor.stop_all()
        except Exception:
            pass
        try:
            from supervisor.message_bus import get_bridge
            get_bridge().shutdown()
        except Exception:
            pass
        _shutdown_supervisor_event_bus()


app = NetworkAuthGate(Starlette(routes=routes, lifespan=lifespan), served_ui=served_ui)
app.app.state.drive_root = pathlib.Path(DATA_DIR)  # type: ignore[attr-defined]
app.app.state.repo_dir = pathlib.Path(REPO_DIR)  # type: ignore[attr-defined]
app.app.state.broadcast_ws_sync = broadcast_ws_sync  # type: ignore[attr-defined]
app.app.state.app_start = APP_START  # type: ignore[attr-defined]
app.app.state.supervisor_ready_event = _supervisor_ready  # type: ignore[attr-defined]
app.app.state.supervisor_init_done_event = _supervisor_init_done  # type: ignore[attr-defined]
app.app.state.get_supervisor_error = lambda: _supervisor_error  # type: ignore[attr-defined]
app.app.state.describe_bg_consciousness_state = _describe_bg_consciousness_state  # type: ignore[attr-defined]
app.app.state.describe_consciousness = lambda: describe_consciousness(_consciousness)  # type: ignore[attr-defined]
app.app.state.startup_owner_command = _startup_owner_command
app.app.state.request_restart = _request_restart_exit  # type: ignore[attr-defined]
app.app.state.runtime_branch_defaults = _runtime_branch_defaults  # type: ignore[attr-defined]
app.app.state.bind_host = _BIND_HOST  # type: ignore[attr-defined]
app.app.state.port_file = PORT_FILE  # type: ignore[attr-defined]
app.app.state.default_port = DEFAULT_PORT  # type: ignore[attr-defined]
app.app.state.start_supervisor_if_needed = _start_supervisor_if_needed  # type: ignore[attr-defined]


def _restart_cleanup_kwargs() -> dict:
    """Keep owner Restart from re-opening daemon custody after its stop; every
    application stop keeps saved work and accepted queue for the next boot (#1563)."""
    if _owner_restart_requested.is_set():
        return {"reconcile_delegate_custody": False, "retain_saved_work": True}
    return {"retain_saved_work": True}


def _emergency_process_cleanup() -> dict:
    """Kill child processes, workers and companions; the kill census. The main and
    Host Service listeners close with this process: no port is swept."""
    from ouroboros.maintenance import child as maintenance_child
    from ouroboros.storage import count as storage_count
    from ouroboros.storage_free import free as storage_free
    maintenance_child.stop()
    storage_count.stop()
    storage_free.stop()
    begin_owned_stop(DATA_DIR, cause=server_stop_cause())  # the grace starts here; cause and pending stops are recorded before any wait
    worker_exits = None  # the pool's own PID census; None until kill_workers ran to its end
    try:
        from supervisor.workers import kill_workers, last_worker_exit_census
        cleanup_kwargs = _restart_cleanup_kwargs()
        if _restart_requested.is_set():
            # A restart that hung past the uvicorn shutdown timeout still reaches
            # here; finalize running tasks as an honest interrupted-by-restart,
            # not a worker crash storm.
            cleanup_status, cleanup_reason = _shutdown_task_cleanup_args(True)
            cleanup_kwargs.update(terminal_status=cleanup_status, result_reason=cleanup_reason,
                                  stop_source="server_shutdown")
        kill_workers(force=True, archive_service_logs=False,
                     **cleanup_kwargs, **_managed_update_pending_kwargs())
        worker_exits = last_worker_exit_census()
    except Exception:
        pass
    if _restart_requested.is_set():
        try:
            _stop_owned_daemon_for_new_pin()
        except Exception:
            log.critical("Planned restart: engine pin check raised; the owned daemon is left serving",
                         exc_info=True)
    owned_stop = stop_owned_work(DATA_DIR)  # the same one stop: joined until it completes or its deadline
    import multiprocessing
    from ouroboros.platform_layer import force_kill_pid
    for child in multiprocessing.active_children():
        try:
            force_kill_pid(child.pid)
        except (ProcessLookupError, PermissionError):
            pass
        try:  # reap it: a zombie would keep active_children non-empty past this exit
            child.join(timeout=2)
        except Exception:
            pass
    if _restart_requested.is_set():  # only the restart that carried an adoption can arm it; never a wait
        body_adoption.arm(DATA_DIR, worker_exits=worker_exits,
                          live_children=multiprocessing.active_children() + maintenance_child.live_children()
                          + storage_count.live_children() + storage_free.live_children(),
                          owned_stop=owned_stop, owner_restart=_owner_restart_requested.is_set())
    try:
        from ouroboros.extension_companion import panic_kill_all
        panic_kill_all()
    except Exception:
        pass
    return {"worker_exits": worker_exits, "owned_stop": owned_stop,
            "live_children": multiprocessing.active_children() + maintenance_child.live_children()
            + storage_count.live_children() + storage_free.live_children()}

def main() -> int:
    from ouroboros.delegate_recovery import observe_restart_parent
    # A Windows direct successor first waits for its parent, which holds the data root until it exits;
    # admission then follows that exit, and a Panic exit ends the successor before any write.
    if observe_restart_parent().get("exit_code") == PANIC_EXIT_CODE:
        configure_process_logging(drive_logs=None)
        log.critical("The previous server generation exited by Panic; its restart successor stops")
        return PANIC_EXIT_CODE
    # Admission before any write: one server per actual data root (server_entrypoint).
    if not _pytest_default_real_data_dir:
        try:
            admit_data_root(DATA_DIR)
        except ServerAdmissionRefused as refusal:
            configure_process_logging(drive_logs=None)
            log.error("Server not started: %s", refusal)
            return 2
    # Sole server.log writer; tests on the real default data root retain only the stream handler.
    configure_process_logging(drive_logs=None if _pytest_default_real_data_dir else DATA_DIR / "logs")
    if not _pytest_default_real_data_dir:  # a pending Panic sweep ends and the installation marker is planted before any child (§9)
        process_scope.initialize_installation_scope(DATA_DIR)
    if not automatic_launch_allowed(os.environ.get("OUROBOROS_LAUNCH_INTENT", "owner"), DATA_DIR, log):
        return 0
    # Verify a benchmark parent's exact integrity pin before resolving the saved bind host;
    # a malformed/replaced snapshot must never become product defaults.
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
    server = _SignalStopServer(None)  # the config follows the port choice; the stop watcher must not
    server.watch_launcher_stop()  # wait for it: a launcher quit or exit from here on ends this start
    actual_port = find_free_port(args.host, args.port, reserved=reserved_service_ports())
    if actual_port != args.port:
        log.info("Port %d busy on %s, using %d instead", args.port, args.host, actual_port)
    if _exit_signalled.is_set():
        log.info("Stop requested before the server started serving; exiting.")
        return 0
    server.config = uvicorn.Config(
        app,
        host=args.host,
        port=actual_port,
        log_level="warning",
        log_config=None,  # uvicorn loggers propagate to the root handlers
        ws_ping_interval=20,
        ws_ping_timeout=20,
        # Leave time for terminal custody inside the launcher stop budget (#1142).
        timeout_graceful_shutdown=SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC,
    )
    _uvicorn_exited = threading.Event()

    def _check_restart():
        """Own final cleanup and transfer, whether or not uvicorn returns."""
        while not _restart_requested.wait(0.5):
            if _uvicorn_exited.is_set():
                return
        log.info("Restart requested — closing WebSocket clients and shutting down server.")

        loop = _event_loop
        if loop:
            try:
                future = asyncio.run_coroutine_threadsafe(close_all_ws(), loop)
                future.result(timeout=3)
            except Exception:
                pass

        server.should_exit = True

        # This bounds the graceful wait; the owned-work stop below has its own
        # deadline inside the launcher's grace, then this thread exits.
        force_exit_timeout_sec = 5 if _LAUNCHER_MANAGED else 30
        if not _uvicorn_exited.wait(timeout=force_exit_timeout_sec):
            log.warning("Uvicorn did not exit within %ss; finishing cleanup before restart",
                        force_exit_timeout_sec)
        try:
            # Our listeners close on exit/exec; port sweeps add no ownership proof.
            store_switch.switch_at_restart(DATA_DIR, REPO_DIR, _emergency_process_cleanup())  # the seam: after every reader is dead, before the exit
            if not _LAUNCHER_MANAGED:
                if _planned_delegate_restart_transaction_id:
                    from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV

                    os.environ[PLANNED_RESTART_TRANSACTION_ENV] = _planned_delegate_restart_transaction_id
                _restart_current_process(args.host, actual_port)
        except Exception:
            log.exception("Restart failed; cleanup or transfer is unconfirmed, custody retained")
            return exit_after_shutdown(1)  # Panic retains termination even when cleanup/transfer failed.
        exit_after_shutdown(RESTART_EXIT_CODE)

    restart_thread = threading.Thread(target=_check_restart, daemon=True)
    restart_thread.start()

    try:
        with bound_service_socket(DATA_DIR, "main", args.host, actual_port,
                                  server_host_source=host_source) as listener:
            actual_port = listener.getsockname()[1]
            write_port_file(PORT_FILE, actual_port)
            log.info("Starting Ouroboros server on %s:%d", args.host, actual_port)
            server.run(sockets=[listener])
    finally:
        _uvicorn_exited.set()
        if _restart_requested.is_set():
            restart_thread.join()  # uvicorn returning does not complete cleanup or direct re-exec

    return 0


def executable_main() -> None:
    """Command boundary: unwind lifecycle cleanup before bypassing capture joins."""
    import signal
    from ouroboros.platform_layer import SHUTDOWN_SIGNALS

    exit_signal = None

    def stop(signum, _frame):
        # Uvicorn replays here after its lifespan. Never interrupt main's final
        # restart join; before serving, its existing stop latch ends startup.
        nonlocal exit_signal
        exit_signal = signum
        _exit_signalled.set()
        _supervisor_stop.set()

    handlers = {}
    try:
        signals = list(SHUTDOWN_SIGNALS)
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        for signum in signals:
            handlers[signum] = signal.signal(signum, stop)
        code = main()
        if exit_signal is not None:
            code = 128 + exit_signal
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (OSError, ValueError):
            code = code or 120  # Broken/closed output cannot restore the executor join.
        exit_after_shutdown(code)  # Accepted Panic retains its writes and exit.
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    executable_main()
