"""What runs INSIDE a worker child process, from entry to crash record.

The pool that spawns workers and the code a worker runs are different worlds:
nothing here reads the pool's state, because none of it exists in this process.
The entry point binds the repo and drive roots it was told to serve, installs
the log sink that streams this worker's lines back over the event queue, runs
the task, and records a crash the parent would otherwise never see.

``worker_main`` stays a module-level function so it remains picklable: on
platforms that spawn rather than fork, the child re-imports it by name.
"""

from __future__ import annotations

import logging
import json
import pathlib
from typing import Any
from ouroboros.observability import stamp_finalization_enqueue
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


# Log types the worker sink does NOT forward: each already reaches the dashboard
# live via a dedicated EVENT_Q sibling/handler, so forwarding the worker's
# append_jsonl copy too would double-broadcast (and task_checkpoint would also be
# re-persisted to events.jsonl by _handle_log_event, a double file write).


WORKER_LOG_SINK_SUPPRESSED_TYPES = frozenset({
    # The durable start/wait-ended rows share the live frame's payload (#1316).
    "tool_call", "tool_call_started", "tool_call_timeout",
    "llm_round", "task_checkpoint", "task_done", "llm_usage",
    "provider_incomplete_response", "llm_empty_response", "provider_body_error",
    "review_cycles_exhausted", "plan_review_advisory_open",
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


def _adopt_published_extensions(pool_drive_root: str) -> None:
    """Catch up with the server's extension generation before the task runs.

    A worker loads extensions ONCE, at spawn, so a skill enabled after that was
    invisible to every task this process served until the pool respawned: the
    model calling the fresh tool got "Unknown tool" while /api/extensions
    truthfully reported it live. This is the natural point to notice — the task
    is about to materialize its tool catalog — and the steady state costs one
    small JSON read, with a bounded reload only when the generations differ.

    The root is the POOL's, never the task's: a subagent or headless task
    carries its own forked ``drive_root``, which has no extension registry of
    its own and would read as "nothing published". This process loaded its
    extensions from the pool root at spawn, and the server publishes there, so
    that is the only root the two generations are comparable in.
    """
    try:
        from ouroboros.config import get_skills_repo_path, load_settings
        from ouroboros.extension_reconcile_queue import adopt_published_extension_generation

        adopt_published_extension_generation(
            pathlib.Path(pool_drive_root),
            load_settings,
            repo_path=get_skills_repo_path() or None,
        )
    except Exception:
        log.debug("extension generation adoption failed", exc_info=True)


def spawn_worker_process(ctx, wid, in_q, out_q, repo_dir, drive_root):
    """Bind a private emergency channel to this exact worker before it starts."""
    import socket
    import weakref
    from functools import partial

    parent, child = socket.socketpair()
    try:
        parent.setblocking(False)
        proc = ctx.Process(target=worker_main, args=(wid, in_q, out_q, str(repo_dir), str(drive_root),
                                                   _current_custody_session_id(), child))
        proc.daemon = True
        proc.start()
    except BaseException:
        parent.close()
        raise
    finally:
        child.close()
    proc._ouroboros_stop_socket = parent
    proc._ouroboros_stop_request = partial(request_worker_stop, weakref.proxy(proc))
    weakref.finalize(proc, parent.close)  # fallback for failed/abandoned pool installation
    return proc


def request_worker_stop(proc):
    """Ask the worker's owners first; force our native child to stop after a bound.

    This Process still owns its native Popen. It is NOT an attached PID/watch:
    Process.kill retains multiprocessing's native exited-child checks, including
    on Darwin, and works before the child installs or can service its lifeline.

    Panic budget: local child owners get 250 ms (``request_worker_owned_stops``),
    the lifeline bounds its callback at 500 ms, and this backstop requests the
    native kill at 750 ms whatever the lifeline did.
    """
    import threading

    native_kill = proc._popen.kill  # retain the owned native child even if the pool drops its Process
    backstop = getattr(proc, "_ouroboros_stop_backstop", None)
    native_request = getattr(proc, "_ouroboros_stop_native_request", None)
    if native_request is None:
        native_request = {"requested": False, "error": "native backstop not completed"}
        proc._ouroboros_stop_native_request = native_request

    def force_stop():
        try:
            native_kill()  # same native operation as Process.kill; no join or application callback
            native_request.update(requested=True, error="")
        except Exception as exc:
            native_request.update(requested=False, error=f"{type(exc).__name__}: {exc}")

    if backstop is None:
        # Give held child owners their bounded request phase before root death.
        # Retain the native child owner until its request; no PID rediscovery.
        backstop = threading.Timer(0.75, force_stop)
        backstop.daemon = True
        proc._ouroboros_stop_backstop = backstop
        try:
            backstop.start()
        except RuntimeError:
            force_stop()  # inability to start a timer cannot remove the native stop
    try:
        proc._ouroboros_stop_socket.send(b"!")
        requested, error = True, ""
    except OSError as exc:
        requested, error = False, type(exc).__name__
    return {"pid": proc.pid, "requested": requested, "scope": "worker_owners",
            "confirmation": "unconfirmed", "error": error,
            "root_backstop": "armed_native_owner", "native_request": native_request,
            "limit": "unpublished spawns/unresponsive child owners unconfirmed"}


def close_worker_stop_channel(proc):
    """Retire parent IPC without cancelling a still-owed native stop request."""
    channel = getattr(proc, "_ouroboros_stop_socket", None)
    if channel is not None:
        channel.close()
    backstop = getattr(proc, "_ouroboros_stop_backstop", None)
    if backstop is not None and proc.exitcode is not None:
        backstop.cancel()
        if backstop.ident is not None:
            backstop.join(timeout=1)
        proc._ouroboros_stop_backstop = None


def request_worker_owned_stops(drive_root):
    """Attempt local owner requests independently within one 250 ms budget.

    Only loaded modules can hold local owners. Do not import an absent owner
    (or wait for an in-flight import) during Emergency Stop. An unpublished
    Popen or an unresponsive owner remains unconfirmed, never a death proof.
    """
    import sys
    import threading
    import time

    deadline = time.monotonic() + 0.25
    results, threads = {}, []

    def request(module_name, method, *, manager=False):
        try:
            module = sys.modules.get(module_name)
            if module is None:
                results[module_name] = {"requested": False, "error": "owner module unavailable"}
                return
            owner = getattr(module, method)
            if manager:
                owner = owner(create=False)
                value = owner.panic_stop(request_only=True) if owner is not None else []
            else:
                value = owner(request_only=True)
                if module_name == "ouroboros.tools.shell_process":
                    # Give returning spawns a bounded chance to publish; already
                    # held children received their requests before this wait.
                    while module._spawning_subprocesses and time.monotonic() < deadline:
                        time.sleep(0.001)
                    value.extend(owner(request_only=True))
                    if module._spawning_subprocesses:
                        value.append({"requested": False, "error": "spawn still unpublished"})
            results[module_name] = value
        except BaseException as exc:
            results[module_name] = {"requested": False, "error": type(exc).__name__}

    for module_name, method, manager in (
        ("ouroboros.tools.shell_process", "kill_all_tracked_subprocesses", False),
        ("ouroboros.workspace_executor", "kill_all_foreground", False),
        ("ouroboros.tools.services", "kill_all_services", False),
        ("ouroboros.extension_companion", "panic_kill_all", False),
        ("ouroboros.claudexor_daemon", "get_owned_daemon", True),
        ("ouroboros.local_model", "get_manager", True),
    ):
        results[module_name] = {"requested": False, "error": "owner request unfinished"}
        thread = threading.Thread(target=request, args=(module_name, method),
                                  kwargs={"manager": manager}, daemon=True)
        try:
            thread.start()
            threads.append(thread)
        except RuntimeError:
            continue
    for thread in threads:
        thread.join(timeout=max(0, deadline - time.monotonic()))
    return results


def _configure_worker_logging() -> None:
    """Give a real pool child its own process logging (stream handler, redaction,
    crash hooks) whatever module the parent ran as ``__main__``; ``worker_main``
    run inside another process leaves logging to that host process."""
    import multiprocessing

    if multiprocessing.parent_process() is not None:
        from ouroboros.process_logging import configure_process_logging

        configure_process_logging(drive_logs=None)


def worker_main(wid: int, in_q: Any, out_q: Any, repo_dir: str, drive_root: str,
                custody_session_id: str = "", stop_socket=None) -> None:
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
    try:
        _configure_worker_logging()
    except Exception:
        import traceback as _tb_boot  # logging itself failed: stderr is all that is left

        _tb_boot.print_exc()
    # Entry progress precedes extension loading and agent construction. If logging
    # fails, the parent retains the ordinary readiness window rather than losing the child.
    try:
        from ouroboros.utils import append_jsonl, utc_now_iso

        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "worker_starting",
            "worker_id": wid, "pid": _os.getpid(), "phase": "entry",
        })
    except Exception:
        log.debug("Worker entry progress unavailable", exc_info=True)
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
    # Lifeline: if the supervisor dies abruptly, this worker would keep running
    # LLM rounds invisibly — group-suicide instead. It watches the spawner's
    # parent sentinel, not the ppid: under forkserver the parent is the
    # forkserver, which outlives a dead supervisor while any worker lives.
    try:
        from ouroboros.process_custody import start_parent_lifeline

        start_parent_lifeline(label=f"worker-{wid}", stop_socket=stop_socket,
                              before_exit=lambda: request_worker_owned_stops(drive_root))
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
    # Every worker must pin the runtime-mode baseline. Spawn and forkserver do
    # not inherit live parent memory, so the pin travels through the
    # parent-exported OUROBOROS_BOOT_RUNTIME_MODE environment key (config.py
    # _resolve_baseline_from_env). This keeps the elevation ratchet consistent.
    try:
        from ouroboros.config import initialize_runtime_mode_baseline
        initialize_runtime_mode_baseline()
    except Exception as _e:
        # Non-fatal: save_settings still has env-var fallback gating.
        try:
            _log_worker_crash(wid, _drive, "init_baseline", _e, _tb.format_exc())
        except Exception:
            pass
    # Guards BOTH extension entry points below: the spawn-time load and the
    # per-task generation adoption must refuse the same real-data-dir case.
    extensions_owned = True
    try:
        from ouroboros.config import get_skills_repo_path, load_settings as _load_settings
        from ouroboros.extension_loader import reload_all as _reload_extensions

        pytest_default_real_data_dir = (
            "pytest" in _sys.modules
            and not _os.environ.get("OUROBOROS_DATA_DIR")
            and _drive.resolve(strict=False) == (_pathlib.Path.home() / "Ouroboros" / "data").resolve(strict=False)
        )
        if pytest_default_real_data_dir:
            extensions_owned = False
            try:
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
    except Exception as _e:
        try:
            _log_worker_crash(wid, _drive, "extension_reload", _e, _tb.format_exc())
        except Exception:
            pass
    try:
        from ouroboros.agent import make_agent
        from functools import partial
        from ouroboros.owner_wait import worker_owner_wait
        agent = make_agent(repo_dir=repo_dir, drive_root=drive_root, event_queue=out_q)
        agent.owner_wait_callback = partial(worker_owner_wait, wid, in_q, out_q)
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
            if extensions_owned:
                _adopt_published_extensions(drive_root)
            if task_drive_root != str(drive_root):
                task_agent = make_agent(
                    repo_dir=repo_dir,
                    drive_root=task_drive_root,
                    event_queue=out_q,
                    budget_drive_root=str(task.get("budget_drive_root") or drive_root),
                )
                task_agent.owner_wait_callback = agent.owner_wait_callback
                events = task_agent.handle_task(task)
            else:
                events = agent.handle_task(task)
            for e in events:
                e2 = dict(e)
                e2["worker_id"] = wid
                if (
                    e2.get("type") == "task_done"
                    and e2.get("task_id") == task.get("id")
                    and not task.get("_is_direct_chat")
                ):
                    # Earlier frames (including the answer) keep their order.
                    # Do not release this slot until its first save attempt ends.
                    try:
                        from ouroboros.headless import prepare_terminal_task_files

                        prepared = prepare_terminal_task_files(_drive, task)
                        if prepared.get("error"):
                            log.warning("Terminal file preparation for %s: %s", task.get("id"), prepared["error"])
                    except Exception:
                        # File publication must not enter the model's crash-retry rail.
                        log.exception("Terminal file preparation failed for %s", task.get("id"))
                    e2["_files_prepared_attempt"] = int(task.get("_attempt") or 1)
                out_q.put(stamp_finalization_enqueue(e2))
        except Exception as _e:
            _log_worker_crash(wid, _drive, "handle_task", _e, _tb.format_exc())
            return


def _log_worker_crash(wid: int, drive_root: pathlib.Path, phase: str, exc: Exception, tb: str) -> None:
    """Record a worker-side crash in this process's log and in ``supervisor.jsonl``.

    The row goes through the shared appender (lock, live sink, test-data guard);
    when that write fails while the process dies, one stderr line is the last
    trace left.
    """
    import os as _os
    import sys as _sys
    entry = {
        "ts": utc_now_iso(),
        "type": "worker_crash",
        "worker_id": wid,
        "pid": _os.getpid(),
        "phase": phase,
        "error": repr(exc),
        "traceback": str(tb)[:3000],
    }
    try:
        if exc is not None:
            log.error("Worker %s crashed during %s", wid, phase, exc_info=exc)
        else:
            log.error("Worker %s crashed during %s:\n%s", wid, phase, tb)
    except Exception:
        pass
    try:
        from ouroboros.utils import append_jsonl

        if append_jsonl(drive_root / "logs" / "supervisor.jsonl", entry):
            return
    except Exception:
        pass
    try:
        print(json.dumps(entry, ensure_ascii=False), file=_sys.stderr, flush=True)
    except Exception:
        pass
