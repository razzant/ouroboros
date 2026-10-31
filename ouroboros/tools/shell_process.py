"""Process-execution substrate shared by every command-running tool.

Owns the tracked subprocess registry, the one admission/publication protocol
its producers start children through (``tracked_start``) and the restart
exit's own finalization scope (``exit_finalization``), its panic-time
request and the ordinary stop's tree kill, the
PYTHONPATH scrubbing that keeps an external workspace from shadow-importing
Ouroboros, the single normalized per-command timeout resolution (which the
loop's outer wait reuses, ``command_timeout_for_call``), the return-code and
bounded stdout/stderr rendering, and the probe that decides whether a
resolved cwd is reachable through the workspace executor. The
run_command/run_script handlers and the tool descriptors stay with
``tools/shell.py``.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, List, Mapping, Optional

from ouroboros.platform_layer import kill_process_tree, request_process_tree_kill, scrub_repo_from_pythonpath, subprocess_new_group_kwargs
from ouroboros.config import SETTINGS_DEFAULTS, load_settings, runtime_environ
from ouroboros.owned_shutdown import stop_begun
from ouroboros.tools.registry import ToolContext
from ouroboros.deadline_utils import deadline_remaining_sec
from ouroboros.workspace_executor import executor_ref_from_ctx
from ouroboros.workspace_executor import map_host_path as executor_map_host_path

log = logging.getLogger(__name__)

# Tracked process groups let panic kill descendant trees too. Publication and the Panic request
# never take a lock: a cleanup holding ``_subprocess_lock`` cannot keep a started child unnamed.
_active_subprocesses: set = set()
_subprocess_lock = threading.Lock()
_panic_requested = False
_spawning_subprocesses = 0  # admitted starts not yet published, or published into a stop not yet handed over
_start_settled = threading.Condition()  # guards only that count and its admission check; never held across I/O
_FINALIZING = threading.local()  # the restart exit's thread inside ``exit_finalization``
_RUN_SHELL_DEFAULT_TIMEOUT_SEC = 360


class CommandAdmissionClosed(RuntimeError):
    """Emergency Stop or this server generation's begun stop refused a tracked start before its Popen."""

    process_not_started = True


@contextmanager
def tracked_start():
    """Admit one tracked child and count it from before its Popen until its start settled.

    The yielded ``publish(proc)`` registers the handle Popen returned; producers call it before any
    durable registration, callback or reader can fail. A child published after Emergency Stop or the
    generation's stop door began is stopped here by its own starter, outside every lock, before the
    count is released, so the ordinary stop's wait (``kill_all_tracked_subprocesses``) never ends while
    that handover is still owed; inside ``exit_finalization`` only Emergency Stop refuses or stops it. A
    refused or failed start always settles. Never wraps a caller's Popen in a lock: the caller keeps its
    own Popen, operation-start gate and error provenance.

    Publication adds the handle, then reads the stop facts; a stop sets its fact, then snapshots. So a
    child is in the stop's snapshot or its publisher saw the stop, with no lock between them; the count
    is admitted under ``_start_settled``, which the stop's wait reads after its door began."""
    global _spawning_subprocesses
    finalizing = getattr(_FINALIZING, "active", False)
    with _start_settled:
        refusal = ("Emergency Stop has retired command admission" if _panic_requested else
                   "this server generation's stop has begun: command admission is closed"
                   if stop_begun() and not finalizing else "")
        if not refusal:
            _spawning_subprocesses += 1
    if refusal:
        raise CommandAdmissionClosed(refusal)
    stopping: list = []

    def publish(proc):
        _active_subprocesses.add(proc)
        if _panic_requested or (stop_begun() and not finalizing):
            stopping.append(proc)
        return proc

    try:
        yield publish
    finally:
        try:
            for proc in stopping:  # Panic only requests; the ordinary stop's tree kill is synchronous
                request_process_tree_kill(proc) if _panic_requested else _kill_process_group(proc)
        except Exception:
            log.warning("A child published into a stop was not stopped by its starter; it stays tracked",
                        exc_info=True)
        finally:
            with _start_settled:
                _spawning_subprocesses -= 1
                _start_settled.notify_all()


@contextmanager
def exit_finalization():
    """The restart exit's own finalization (``store_switch.switch_at_restart``), entered after the
    generation's one stop finished: tracked starts on THIS thread pass that stop's closed admission and
    publication under their own timeouts. Emergency Stop still refuses and ends them; other threads, and
    this one after the block, stay refused."""
    previous = getattr(_FINALIZING, "active", False)
    _FINALIZING.active = True
    try:
        yield
    finally:
        _FINALIZING.active = previous


def _tracked_subprocess_run(cmd, **kwargs):
    """subprocess.run replacement with process-tree tracking. When capturing TEXT
    output, decode tolerantly (errors='replace') so binary stdout/stderr (a MIPS
    interpreter, a DOOM framebuffer, raw bytes) surfaces as readable text instead
    of raising UnicodeDecodeError and collapsing the whole call into a
    shell_error."""
    timeout = kwargs.pop("timeout", None)
    record_path = None
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("errors", "replace")
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    kwargs.update(subprocess_new_group_kwargs())
    from ouroboros.process_scope import record_task_group, scoped_child_env, task_launch_target
    from ouroboros.tool_custody import invocation_binding

    binding = invocation_binding()
    target = task_launch_target(binding)
    if target is not None:
        # The task scope marker rides THIS launch's env only (never os.environ): the
        # child and whatever it leaves behind belong to the task until the task ends
        # (DEVELOPMENT "Process Custody Rule"); the command's own process group is
        # recorded below, which is what reaches a platform binary on macOS.
        kwargs["env"] = scoped_child_env(kwargs["env"] if kwargs.get("env") is not None else os.environ, target)
    from ouroboros.tool_custody import external_handoff
    confirmed = external_handoff("shell_process")
    proc = None
    try:
        from ouroboros.owner_pause import operation_start

        with tracked_start() as publish, operation_start():
            try:
                proc = subprocess.Popen(cmd, **kwargs)
            except (OSError, ValueError) as exc:
                exc.process_not_started = True
                raise
            publish(proc)  # before the spawning owner can exit
        from ouroboros.workspace_executor import _register_process
        if target is not None:  # task scope only; the foreground receipt keeps its own condition below
            record_task_group(*target, proc.pid)
        if binding.get("drive_root") and binding.get("task_id"):
            record_path = _register_process(pathlib.Path(binding["drive_root"]), {
                "record_type": "foreground", "executor_type": "local", "executor_id": "host",
                "host_pid": proc.pid})
            if record_path is not None:
                confirmed()
    except BaseException:
        if proc is None:
            confirmed()
        raise
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.wait(timeout=5)
        raise
    finally:
        if proc.poll() is not None:
            confirmed()
            from ouroboros.workspace_executor import _forget_process
            _active_subprocesses.discard(proc)
            _forget_process(record_path)


def _kill_process_group(proc):
    """Kill a subprocess tree."""
    kill_process_tree(proc)


def kill_all_tracked_subprocesses(*, request_only=False, deadline=None):
    """Request owned kills without locks (Panic), or stop the tracked children (ordinary cleanup).

    The ordinary stop kills its snapshot through the tree owner. With the generation stop's
    ``deadline`` (admission is already closed) it also waits, until then, for starts still in flight,
    stops what they published and confirms each exit; without one it adds no wait. Only handles whose
    exit it observed leave the registry: a concurrent publication or an unconfirmed kill keeps its
    name. Returns ``None`` when every child it addressed exited, else the live ``pids`` and the
    ``unpublished`` starts it could not see, which no durable record names."""
    global _panic_requested
    if request_only:
        _panic_requested = True
        return [request_process_tree_kill(proc) for proc in _active_subprocesses.copy()]
    procs = list(_active_subprocesses.copy())  # whole-set copies and updates: no publisher waits on this stop
    for proc in procs:
        if proc.poll() is None:  # a reaped handle's PID may already name another process
            _kill_process_group(proc)
    if deadline is not None:
        with _start_settled:
            _start_settled.wait_for(lambda: not _spawning_subprocesses, max(0.0, deadline - time.monotonic()))
        late = [proc for proc in _active_subprocesses.copy() if proc not in procs]
        for proc in late:  # its starter already handed it to the tree kill; this stop confirms the exit
            if proc.poll() is None:
                _kill_process_group(proc)
        procs += late
    alive = [proc for proc in procs if not _exit_observed(proc, deadline)]
    _active_subprocesses.difference_update([proc for proc in procs if proc not in alive])
    with _start_settled:
        unpublished = _spawning_subprocesses
    if alive or unpublished:
        return {"state": "unconfirmed", "pids": sorted(proc.pid for proc in alive), "unpublished": unpublished}
    return None


def _exit_observed(proc, deadline) -> bool:
    try:
        if deadline is not None:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        return proc.poll() is not None
    except subprocess.TimeoutExpired:
        return False


def unsettled_tracked_children() -> Optional[dict]:
    """What a joined stop must still disclose, without a wait: tracked children still alive and starts
    still in flight (a publication after the stop ran is stopped by its starter, not confirmed)."""
    alive = sorted(proc.pid for proc in _active_subprocesses.copy() if proc.poll() is None)
    with _start_settled:
        unpublished = _spawning_subprocesses
    return {"state": "unconfirmed", "pids": alive, "unpublished": unpublished} if alive or unpublished else None


def _shell_env_for_cwd(ctx: ToolContext, work_dir: pathlib.Path) -> dict:
    """For a command whose cwd is OUTSIDE the Ouroboros system repo (an external
    workspace / target project, e.g. SWE-bench dig-direct ``/app``), return an
    env copy with the repo dir scrubbed from ``PYTHONPATH`` so the target cannot
    shadow-import Ouroboros's own modules (R2). ``ctx.repo_dir`` stays pinned to
    the Ouroboros repo even in workspace mode, so this is the authoritative
    in-repo test. Every child receives the task environment; in-repo commands
    retain the repository Python path. A cwd inside the bound body candidate gets
    the candidate's own isolated environment instead (``body_candidate``): its
    code must never resolve the serving data root or repository."""
    from ouroboros import body_candidate

    isolated = body_candidate.process_environment(ctx, work_dir)
    if isolated is not None:
        return isolated
    env = runtime_environ()
    try:
        system_repo = pathlib.Path(getattr(ctx, "repo_dir")).resolve(strict=False)
        wd = pathlib.Path(work_dir).resolve(strict=False)
    except Exception:
        return env
    try:
        in_repo = wd == system_repo or wd.is_relative_to(system_repo)
    except AttributeError:  # pragma: no cover - py<3.9
        in_repo = str(wd) == str(system_repo) or str(wd).startswith(str(system_repo) + os.sep)
    if in_repo:
        return env
    return scrub_repo_from_pythonpath(env, system_repo)


def _resolve_effective_timeout(
    default_timeout_sec: int,
    ctx: ToolContext | None = None,
    override_sec: int | None = None,
) -> int:
    """Resolve the effective per-command timeout as ONE normalized pipeline:
    resolve the REQUESTED value from a single precedence chain (per-call
    ``override_sec`` > env ``OUROBOROS_TOOL_TIMEOUT_SEC`` > settings.json > config
    ``SETTINGS_DEFAULTS`` > the in-code last-resort ``default_timeout_sec``), then
    apply the per-call ceiling, then clamp toward the remaining task-deadline budget
    (60s floor when a deadline exists), then floor at 1s. The outer budget loop
    remains the hard deadline enforcer.

    Hygiene fix (SSOT): the prior code skipped an env/settings value EQUAL to the
    config default (``!= default_setting``), so ``OUROBOROS_TOOL_TIMEOUT_SEC=600``
    (= the SETTINGS_DEFAULTS value) silently fell through to the in-code 360s default.
    The configured value is now honored regardless of equality, and env/settings
    values no longer BYPASS the ceiling/deadline clamp. RELEASE NOTE: installs that
    relied on the buggy effective 360s now get the configured 600s — a foreground
    command may hold the task longer (still bounded by ceiling + task deadline).
    """
    from ouroboros.config import get_per_call_timeout_ceiling_sec

    # 1. Resolve the REQUESTED timeout from a single precedence chain.
    requested: int | None = None
    if override_sec is not None:
        try:
            ov = int(override_sec)
        except (TypeError, ValueError):
            ov = 0
        if ov > 0:
            requested = ov
    if requested is None:
        raw = str(os.environ.get("OUROBOROS_TOOL_TIMEOUT_SEC", "") or "").strip()
        if raw:
            try:
                v = int(raw)
                if v > 0:
                    requested = v
            except ValueError:
                pass
    if requested is None:
        try:
            settings_val = int(load_settings().get("OUROBOROS_TOOL_TIMEOUT_SEC") or 0)
            if settings_val > 0:
                requested = settings_val
        except Exception:
            pass
    if requested is None:
        cfg_default = int(SETTINGS_DEFAULTS.get("OUROBOROS_TOOL_TIMEOUT_SEC") or 0)
        requested = cfg_default if cfg_default > 0 else int(default_timeout_sec)

    # 2. Per-call ceiling.
    effective = min(requested, get_per_call_timeout_ceiling_sec())

    # 3. Clamp toward the remaining task-deadline budget (60s floor when a deadline exists).
    if ctx is not None:
        remaining = deadline_remaining_sec(ctx)
        if remaining > 0:
            effective = int(max(60, min(effective, remaining * 0.5)))

    # 4. Floor at 1s.
    return max(1, int(effective))


# The handlers that bound their child with ``_resolve_effective_timeout``. The loop's outer
# wait resolves the same value here and adds its settlement margin, so whatever sets the
# bound (override, environment, settings, default, ceiling or task deadline) the child's own
# cleanly reported timeout fires first. The margin covers ordinary settlement, not work after
# the child: verify_and_record's executor probes can outlast it, and that result then settles
# late (``late_tool_results``) (#1376). Only these three are aligned here: skill_exec,
# extensions and MCP keep their own child bounds and outer windows.
COMMAND_TIMEOUT_TOOLS = frozenset({"run_command", "run_script", "verify_and_record"})


def command_timeout_for_call(tool_name: str, args: Mapping[str, Any], ctx: ToolContext | None = None) -> Optional[int]:
    """The child timeout ``tool_name``'s handler will use for dispatched ``args``, or ``None``
    when the tool does not bound a child through this resolution. Dispatch has already folded
    the ``timeout`` alias into ``timeout_sec`` (``tool_resolution._normalize_tool_call_args``)."""
    if tool_name not in COMMAND_TIMEOUT_TOOLS:
        return None
    return _resolve_effective_timeout(_RUN_SHELL_DEFAULT_TIMEOUT_SEC, ctx, override_sec=args.get("timeout_sec"))


# R5 (node-runtime sprint): the render moved to the typed process-facts SSOT
# (signal naming, lived_ms/resolved_runtime disclosure); the historical private
# spelling stays importable here for call sites and tests.
from ouroboros.tools.process_facts import describe_returncode as _describe_returncode  # noqa: E402, F401 — historical private spelling for call sites and tests


def _format_process_output(stdout: str, stderr: str, *, limit: int = 50_000) -> str:
    """Mask diagnostic copies before bounds can split a selected secret."""
    from ouroboros.tools.process_facts import redact_process_data

    stdout_text = redact_process_data(str(stdout or ""))
    stderr_text = redact_process_data(str(stderr or ""))
    parts: List[str] = []
    if stdout_text.strip():
        parts.append(f"STDOUT:\n{stdout_text}")
    if stderr_text.strip():
        parts.append(f"STDERR:\n{stderr_text}")
    rendered = "\n\n".join(parts) if parts else "STDOUT:\n(empty)"
    if len(rendered) > limit:
        rendered = rendered[: limit // 2] + "\n...(truncated)...\n" + rendered[-limit // 2 :]
    return rendered


def _executor_can_run_cwd(ctx: ToolContext, work_dir: pathlib.Path) -> bool:
    executor_ref = executor_ref_from_ctx(ctx)
    if executor_ref is None:
        return False
    try:
        executor_map_host_path(executor_ref, pathlib.Path(work_dir).resolve(strict=False))
        return True
    except Exception:
        return False

def _publish_finished_process_facts(ctx, res, started_ts) -> int:
    """Typed process facts (R5) for a returned child, measured structurally —
    never re-derived from prose. Returns the child's lifetime in ms."""
    import time

    from ouroboros.tools.process_facts import (
        active_resolved_runtime,
        publish_process_facts,
    )

    lived_ms = max(0, int((time.monotonic() - started_ts) * 1000))
    publish_process_facts(
        returncode=getattr(res, "returncode", None),
        started_ts=started_ts,
        resolved_runtime=active_resolved_runtime(ctx),
    )
    return lived_ms


def _publish_unfinished_process_facts(
    ctx, started_ts, *, timed_out: bool = False, spawn_error: BaseException | None = None,
) -> None:
    """Typed facts for a child with no returncode (timeout / pre-exec failure).

    There is no exit code to publish here — that is exactly the case the typed
    family names instead of leaving silent: ``timed_out`` (with the host kill
    that enforces the deadline) for a child the deadline stopped, and
    ``pre_exec_failure`` carrying the platform's exception class for a child
    that never reached exec. Duration and the attested substituted runtime ride
    along as before.

    An ``OSError`` from the spawn is what a PRE-EXEC failure looks like
    (FileNotFoundError, PermissionError, ...): the command never ran and the
    exception class is the honest typed cause. Any other exception is a
    host-side failure AROUND the run and claims no pre-exec cause."""
    from ouroboros.tools.process_facts import (
        active_resolved_runtime,
        publish_process_facts,
    )

    publish_process_facts(started_ts=started_ts,
                          resolved_runtime=active_resolved_runtime(ctx),
                          timed_out=timed_out,
                          # The deadline is enforced BY the host killing the
                          # subprocess tree, so a timeout is always also a host
                          # kill — the one fact a Windows kill can still carry.
                          killed_by_host=timed_out,
                          pre_exec_failure=(
                              type(spawn_error).__name__
                              if isinstance(spawn_error, OSError) else ""))
