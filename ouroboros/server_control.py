"""Process-control helpers for the self-editable server entrypoint."""

from __future__ import annotations

import os
import json
import pathlib
import sys
from typing import Any


def external_owner_binding(state: dict) -> tuple[bool, Any, tuple[int, int]]:
    """Ordinary commands use this current state observation, never Panic's cached pair."""
    from supervisor.state import control_value

    user_known, owner = control_value(state, "owner_external_id")
    chat_known, chat = control_value(state, "owner_external_chat_id")
    try:
        pair = (int(owner or 0), int(chat or 0))
    except (TypeError, ValueError):
        pair = (0, 0)
    return user_known and chat_known, owner, pair


def dispatch_accepted_restart(bridge, text: str, *, callback=None, **message) -> None:
    """Schedule accepted Restart after its transport captured the canonical receipt."""
    import logging
    import threading

    callback = callback or getattr(bridge, "startup_owner_command", None)
    if text.strip().lower() != "/restart" or not callable(callback):
        bridge.enqueue_local_message(text, **message)
        return
    from ouroboros.task_finalization import host_operation_reply_kwargs
    from supervisor.message_bus import send_with_budget

    def reply(body, status=""):
        bridge.activate_update_transport(message)
        send_with_budget(message["chat_id"], body, role="system", system_type="command_reply",
                         **host_operation_reply_kwargs(message.get("accepted_source_ref"), status))

    identity = {key: message[key] for key in ("source", "user_id", "chat_id")}
    action = callback(text, **identity, reply=reply,
                      send_kwargs={key: value for key, value in message.items() if key not in identity})
    if action is None:
        bridge.enqueue_local_message(text, **message)
        return

    def execute():
        try:
            if not action():  # A changed/unknown owner uses ordinary authenticated intake.
                bridge.enqueue_local_message(text, **message)
        except Exception:
            logging.getLogger(__name__).exception("Accepted startup Restart failed; its canonical receipt remains retained")

    threading.Thread(target=execute, name="startup-owner-restart", daemon=True).start()


class PanicIngress:
    """One bridge generation's emergency door, independent of ordinary intake.

    A positively observed external binding is immutable until owner Reset. Keep
    that pair through state outages; never derive one from an unknown/empty slot.
    Reset closes this generation before disk work. Readers already holding its
    old cell cannot republish into the new, closed cell.
    """

    def __init__(self, stop=None):
        self._stop = stop
        self._owner = [True, None]

    def observe_owner(self, state: dict) -> None:
        from supervisor.state import control_value

        cell = self._owner
        if not cell[0] or cell[1] is not None or not isinstance(state, dict) or not state.get("initialization_id"):
            return
        user_known, user = control_value(state, "owner_external_id")
        chat_known, chat = control_value(state, "owner_external_chat_id")
        if user_known and chat_known and type(user) is int and type(chat) is int and user > 0 and chat > 0:
            cell[1] = (user, chat)

    def invalidate_owner(self) -> None:
        self._owner = [False, None]

    def request(self, text: str, *, source="web", user_id=0, chat_id=0) -> bool:
        if self._stop is None or not str(text).strip().lower().startswith("/panic"):
            return False
        pair = self._owner[1]
        if source != "web" and (pair is None or pair != (user_id, chat_id)):
            return False
        import threading

        # Do not wait for the supervisor, the chat ingress lock, a fresh state
        # read, or a shared executor slot. Existing emergency owners do the stop.
        threading.Thread(target=self._stop, name="panic-ingress", daemon=True).start()
        return True


def _spawn_restart_successor(
    argv: list[str], env: dict[str, str], repo_dir: pathlib.Path, *,
    new_process_group: bool = True,
) -> None:
    from ouroboros.config import DATA_DIR
    from ouroboros.process_custody import spawn_supervised
    from ouroboros.delegate_recovery import (
        PLANNED_RESTART_TRANSACTION_ENV, WINDOWS_RESTART_PARENT_HANDLE_ENV,
        bind_windows_restart_successor, open_windows_restart_parent,
    )

    root = pathlib.Path(DATA_DIR)
    env.pop(WINDOWS_RESTART_PARENT_HANDLE_ENV, None)
    transaction_id = env.get(PLANNED_RESTART_TRANSACTION_ENV, "")
    observing = os.name == "nt" and not new_process_group and bool(transaction_id)
    kernel, parent_handle = open_windows_restart_parent(root, transaction_id) if observing else (None, 0)
    try:
        popen_kwargs = {}
        if observing:
            import subprocess

            startupinfo = subprocess.STARTUPINFO()
            startupinfo.lpAttributeList = {"handle_list": [parent_handle]}
            popen_kwargs = {"startupinfo": startupinfo, "close_fds": True}
            env[WINDOWS_RESTART_PARENT_HANDLE_ENV] = str(parent_handle)
        proc = spawn_supervised(
            argv,
            drive_root=root,
            # The replacement is the next server generation. Session scope would
            # make its startup reap treat it as a foreign-session process.
            purpose="server_restart_fallback",
            scope="daemon",
            cwd=str(repo_dir),
            env=env,
            new_process_group=new_process_group,
            **popen_kwargs,
        )
        if observing:
            try:
                bind_windows_restart_successor(root, transaction_id, proc.pid)
            except Exception:
                proc.terminate()
                proc.wait(timeout=5)
                raise
    finally:
        if parent_handle:
            kernel.CloseHandle(parent_handle)


def restart_current_process(
    host: str,
    port: int,
    *,
    repo_dir: pathlib.Path,
    log: Any,
    owner_initiated: bool = False,
) -> None:
    """Transfer direct server mode to a replacement process.

    ``owner_initiated`` marks the restart the OWNER asked for (the chat Restart
    button, and the control endpoints that restart on the owner's behalf). Only
    that restart drops the inherited runtime-mode ratchet pin, so the child
    re-pins from ``load_settings()``; an agent- or supervisor-initiated restart
    keeps inheriting it exactly as before.
    """
    env = os.environ.copy()
    desired_host = str(host)
    try:
        from ouroboros.config import load_settings
        desired_host = (
            str(os.environ.get("OUROBOROS_SERVER_HOST") or "").strip()
            or str(load_settings().get("OUROBOROS_SERVER_HOST") or "").strip()
            or desired_host
        )
    except Exception:
        desired_host = str(host)
    # Keep real env/argv overrides, but do not turn a Settings-derived host into
    # an env pin: the next owner save must still apply on the following restart.
    env["OUROBOROS_SERVER_PORT"] = str(port)
    env.pop("OUROBOROS_MANAGED_BY_LAUNCHER", None)
    env.pop("OUROBOROS_MANAGED_REPO_DIR", None)
    if owner_initiated:
        # The ratchet pin is exported so a CHILD inherits the parent's baseline
        # and cannot widen its own scope. Carried across an owner restart it also
        # pinned the mode the owner just raised in Settings and pressed Restart to
        # apply: the replacement re-pinned the OLD baseline from this env and the
        # new mode never took effect, on this restart or any later one. Dropping
        # the key here makes the child re-pin from load_settings() — the file only
        # the owner can author. Agent/supervisor restarts keep inheriting it.
        from ouroboros.config import BOOT_RUNTIME_MODE_ENV_KEY

        env.pop(BOOT_RUNTIME_MODE_ENV_KEY, None)
    raw_argv = sys.argv
    try:
        saved = json.loads(os.environ.get("OUROBOROS_SERVER_REEXEC_ARGV_JSON", "") or "[]")
        if isinstance(saved, list) and saved and all(isinstance(item, str) and item for item in saved):
            raw_argv = saved
    except Exception:
        raw_argv = sys.argv
    argv = [sys.executable, *raw_argv]
    from ouroboros import platform_layer

    if platform_layer.IS_WINDOWS:
        # Windows CRT exec is spawn-plus-exit; use the custodied spawn directly.
        # Keep its console group so Ctrl+C still reaches the replacement.
        log.info("Starting replacement direct server mode on %s:%d", desired_host, port)
        try:
            _spawn_restart_successor(argv, env, repo_dir, new_process_group=False)
        except Exception:
            log.exception("Spawned restart fallback failed; no successor was started.")
            raise
        log.info("Spawned replacement server process for Windows direct restart.")
        return

    log.info("Re-executing direct server mode on %s:%d", desired_host, port)
    try:
        os.execvpe(sys.executable, argv, env)
    except Exception:
        log.exception("Direct re-exec failed; attempting spawned restart fallback.")
        try:
            _spawn_restart_successor(argv, env, repo_dir)
            log.info("Spawned replacement server process after exec failure.")
        except Exception:
            log.exception("Spawned restart fallback failed; no successor was started.")
            raise


def execute_panic_stop(
    consciousness: Any,
    kill_workers_fn,
    *,
    data_dir: pathlib.Path,
    panic_exit_code: int,
    log: Any,
    bound_port: int | None = None,
) -> None:
    """Request owned emergency stops, write panic flag, hard-exit; disclose unknowns.

    ``bound_port`` is the main port the server actually bound. The caller owns
    that fact and passes it in; this leaf does not reach back into the server
    module for it. Omitted (or falsy), the sweep falls back to the default
    install port — see the sweep below.

    Owner-local request-only APIs signal captured children before settlement.
    Normal admission pins attested daemon identities; ``stop_outcome`` later
    settles custody and attempts same-home CLI shutdown for unresolved targets.
    The shared daemon survives ordinary close outside the Windows launcher Job,
    so that Job is not a Panic backstop.
    Unconfirmed shutdown remains disclosed with custody retained; names or
    recycled descriptor ports never authorize signalling an unrelated process.
    """
    import threading
    import time
    from ouroboros.startup_historical_audit import audit

    audit.stop()
    requests, settlements = {}, {}
    request_threads = []

    def attempt(name, fn, *, settle=False, native=False):
        def run():
            try:
                value = fn()
                (settlements if settle else requests)[name] = value
            except Exception as exc:
                (settlements if settle else requests)[name] = {"requested": False,
                    "error": f"{type(exc).__name__}: {exc}"}
                if settle and name == "daemon":
                    try:
                        _record_unconfirmed_daemon_stop(data_dir, exc)
                    except Exception as disclosure_error:
                        settlements[name]["disclosure_error"] = type(disclosure_error).__name__
        if native:
            run()  # retained multiprocessing owner; no application callbacks
            return
        (settlements if settle else requests)[name] = "unfinished"
        thread = threading.Thread(target=run, name=f"panic-{name}", daemon=True)
        try:
            thread.start()
        except RuntimeError as exc:
            (settlements if settle else requests)[name] = {"requested": False, "error": str(exc)}
            return  # failure of one owner cannot skip the remaining native children
        if not settle:
            request_threads.append(thread)

    import multiprocessing

    from ouroboros.claudexor_daemon import get_owned_daemon
    from ouroboros.extension_companion import panic_kill_all
    from ouroboros.gateway.host_service import host_service_port
    from ouroboros.local_model import get_manager
    from ouroboros.mcp_task_sessions import stop_scope as stop_mcp_bridges
    from ouroboros.platform_layer import kill_process_on_port
    from ouroboros.tools.services import kill_all_services
    from ouroboros.tools.shell import kill_all_tracked_subprocesses
    from ouroboros.workspace_executor import kill_all_foreground
    from supervisor.worker_pool_lifecycle import kill_worker_tree

    # Singleton creation can wait on a startup lock; only already-owned handles
    # belong to this immediate phase. Unknown attachments stay explicitly unproved.
    model, daemon = get_manager(create=False), get_owned_daemon(create=False)
    if model is not None:
        attempt("local-model", lambda: model.panic_stop(request_only=True))
    if daemon is not None:
        attempt("daemon", lambda: daemon.panic_stop(request_only=True))
    else:
        requests["daemon"] = {"requested": False, "error": "no captured daemon identity"}
    attempt("commands", lambda: kill_all_tracked_subprocesses(request_only=True))
    attempt("executors", lambda: kill_all_foreground(data_dir, request_only=True))
    attempt("services", lambda: kill_all_services(data_dir, request_only=True))
    attempt("mcp-browser", lambda: stop_mcp_bridges(data_dir, request_only=True))
    attempt("companions", lambda: panic_kill_all(request_only=True))
    children = multiprocessing.active_children()
    for child in children:
        attempt(f"child-{child.pid}", lambda child=child: kill_worker_tree(
            child.pid, panic_process=child), native=True)

    request_deadline = time.monotonic() + .25
    for thread in request_threads:
        thread.join(timeout=max(0, request_deadline - time.monotonic()))

    # Complete bounded native requests BEFORE settlement helpers can kill this
    # server via its ports. Worker cooperation is never the root-stop authority.
    backstop_deadline = time.monotonic() + 1
    for child in children:
        backstop = getattr(child, "_ouroboros_stop_backstop", None)
        if backstop is not None and backstop.ident is not None:
            backstop.join(timeout=max(0, backstop_deadline - time.monotonic()))

    # Each owner had an independent request attempt; unfinished callbacks remain
    # unconfirmed. Existing owners settle trees/custody independently; neither
    # their launch nor this process's exit is proof that another process died.
    if consciousness is not None:
        attempt("consciousness", consciousness.stop, settle=True)
    if model is not None:
        attempt("local-model", model.stop_server, settle=True)
    attempt("daemon", lambda: get_owned_daemon().stop_outcome(), settle=True)
    attempt("commands", kill_all_tracked_subprocesses, settle=True)
    attempt("executors", lambda: kill_all_foreground(data_dir, wait=False), settle=True)
    attempt("services", lambda: kill_all_services(data_dir, wait=False), settle=True)
    attempt("mcp-browser", lambda: stop_mcp_bridges(data_dir), settle=True)
    attempt("companions", panic_kill_all, settle=True)
    # Workers received private lifeline requests above. Root-first tree cleanup
    # here would destroy their local child ownership before those requests run.
    # Queue/custody reconciliation belongs to the following supervisor boot.
    attempt("main-port", lambda: kill_process_on_port(bound_port or 8765), settle=True)
    attempt("host-port", lambda: kill_process_on_port(host_service_port()), settle=True)

    flag_written = _bounded(lambda: _write_panic_flag(data_dir), 2.0)
    controls_written = _bounded(lambda: _persist_panic_controls(data_dir), 2.0)
    _bounded(lambda: log.critical("PANIC STOP: requests=%s; settlement=%s; flag persisted=%s; controls=%s; "
                                  "hard exit %d, unresolved custody retained.",
                                  requests, settlements, flag_written, controls_written, panic_exit_code), 0.5)
    os._exit(panic_exit_code)


def _bounded(fn, timeout_sec: float) -> bool:
    """Run one best-effort Panic write on a daemon thread and wait at most
    ``timeout_sec``: a stalled disk or lock can never hold the exit. True only
    when it finished without raising."""
    import threading

    done = threading.Event()

    def _run() -> None:
        try:
            fn()
            done.set()
        except Exception:
            pass

    threading.Thread(target=_run, name="panic-bounded-write", daemon=True).start()
    return done.wait(timeout_sec)


def _write_panic_flag(data_dir: pathlib.Path) -> None:
    panic_flag = data_dir / "state" / "panic_stop.flag"
    panic_flag.parent.mkdir(parents=True, exist_ok=True)
    panic_flag.write_text("panic", encoding="utf-8")


def _persist_panic_controls(data_dir: pathlib.Path) -> None:
    """Panic is an owner stop: disable evolution and consciousness as KNOWN controls
    (short lock, never unlocked), record the evolution stop intent, close the campaign
    without git cleanup and drop a queued promotion. Each step is independent."""
    from ouroboros.post_task_evolution import drop_pending_request
    from supervisor import state
    from supervisor.evolution_lifecycle import complete_evolution_campaign, record_evolution_stop_intent

    failures = []
    for step in (
        lambda: state.update_state(_panic_controls, confirm=PANIC_CONTROL_KEYS, lock_timeout_sec=0.5),
        lambda: record_evolution_stop_intent("panic", "panic stop"),
        # cleanup_worktree=False: Panic never runs git stash/reset; the flag + boot reconcile own it.
        lambda: complete_evolution_campaign("panic stop", status="stopped", cleanup_worktree=False),
        lambda: drop_pending_request(data_dir),
    ):
        try:
            if step() is False:
                failures.append("write returned False")
        except Exception as exc:
            failures.append(type(exc).__name__)
    if failures:
        raise OSError(f"Panic controls unconfirmed: {failures}")


def _panic_controls(st: dict) -> None:
    st.update(evolution_mode_enabled=False, bg_consciousness_enabled=False,
              evolution_owner_stopped=True, post_task_autostop=False)
    st.pop("evolution_stop_source", None)  # an owner stop: no agent source may un-stick it


PANIC_CONTROL_KEYS = ("evolution_mode_enabled", "bg_consciousness_enabled", "evolution_owner_stopped",
                      "evolution_stop_source", "post_task_autostop")


def _record_unconfirmed_daemon_stop(data_dir: pathlib.Path, exc: BaseException) -> None:
    from ouroboros.utils import append_jsonl, utc_now_iso

    append_jsonl(data_dir / "logs" / "supervisor.jsonl", {
        "ts": utc_now_iso(), "type": "process_stop_unconfirmed",
        "purpose": "claudexor_daemon", "reason": f"stop raised {type(exc).__name__}",
    })
