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
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


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


def worker_main(wid: int, in_q: Any, out_q: Any, repo_dir: str, drive_root: str,
                custody_session_id: str = "") -> None:
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
