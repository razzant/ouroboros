"""Process-execution substrate shared by every command-running tool.

Owns the tracked subprocess registry and its panic-time tree kill, the
PYTHONPATH scrubbing that keeps an external workspace from shadow-importing
Ouroboros, the single normalized per-command timeout resolution, the
return-code and bounded stdout/stderr rendering, and the probe that decides
whether a resolved cwd is reachable through the workspace executor. The
run_command/run_script handlers and the tool descriptors stay with
``tools/shell.py``.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import threading
from typing import List

from ouroboros.platform_layer import kill_process_tree, scrub_repo_from_pythonpath, subprocess_new_group_kwargs
from ouroboros.config import SETTINGS_DEFAULTS, load_settings
from ouroboros.tools.registry import ToolContext
from ouroboros.deadline_utils import deadline_remaining_sec
from ouroboros.workspace_executor import executor_ref_from_ctx
from ouroboros.workspace_executor import map_host_path as executor_map_host_path


# Tracked process groups let panic kill descendant trees too.
_active_subprocesses: set = set()
_subprocess_lock = threading.Lock()
_RUN_SHELL_DEFAULT_TIMEOUT_SEC = 360


def _tracked_subprocess_run(cmd, **kwargs):
    """subprocess.run replacement with process-tree tracking. When capturing TEXT
    output, decode tolerantly (errors='replace') so binary stdout/stderr (a MIPS
    interpreter, a DOOM framebuffer, raw bytes) surfaces as readable text instead
    of raising UnicodeDecodeError and collapsing the whole call into a
    shell_error."""
    timeout = kwargs.pop("timeout", None)
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("errors", "replace")
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    kwargs.update(subprocess_new_group_kwargs())
    proc = subprocess.Popen(cmd, **kwargs)
    with _subprocess_lock:
        _active_subprocesses.add(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.wait(timeout=5)
        raise
    finally:
        with _subprocess_lock:
            _active_subprocesses.discard(proc)


def _kill_process_group(proc):
    """Kill a subprocess tree."""
    kill_process_tree(proc)


def kill_all_tracked_subprocesses():
    """Kill all tracked subprocess trees on panic."""
    with _subprocess_lock:
        procs = list(_active_subprocesses)
    for proc in procs:
        _kill_process_group(proc)
    with _subprocess_lock:
        _active_subprocesses.clear()


def _shell_env_for_cwd(ctx: ToolContext, work_dir: pathlib.Path) -> "dict | None":
    """For a command whose cwd is OUTSIDE the Ouroboros system repo (an external
    workspace / target project, e.g. SWE-bench dig-direct ``/app``), return an
    env copy with the repo dir scrubbed from ``PYTHONPATH`` so the target cannot
    shadow-import Ouroboros's own modules (R2). ``ctx.repo_dir`` stays pinned to
    the Ouroboros repo even in workspace mode, so this is the authoritative
    in-repo test. Returns ``None`` for commands inside the system repo (Ouroboros
    tooling legitimately imports itself) so they inherit ``os.environ``."""
    try:
        system_repo = pathlib.Path(getattr(ctx, "repo_dir")).resolve(strict=False)
        wd = pathlib.Path(work_dir).resolve(strict=False)
    except Exception:
        return None
    try:
        in_repo = wd == system_repo or wd.is_relative_to(system_repo)
    except AttributeError:  # pragma: no cover - py<3.9
        in_repo = str(wd) == str(system_repo) or str(wd).startswith(str(system_repo) + os.sep)
    if in_repo:
        return None
    return scrub_repo_from_pythonpath(dict(os.environ), system_repo)


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


# R5 (node-runtime sprint): the render moved to the typed process-facts SSOT
# (signal naming, lived_ms/resolved_runtime disclosure); the historical private
# spelling stays importable here for call sites and tests.
from ouroboros.tools.process_facts import describe_returncode as _describe_returncode  # noqa: E402, F401 — historical private spelling for call sites and tests


def _format_process_output(stdout: str, stderr: str, *, limit: int = 50_000) -> str:
    """Render bounded stdout/stderr sections."""
    stdout_text = str(stdout or "")
    stderr_text = str(stderr or "")
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
