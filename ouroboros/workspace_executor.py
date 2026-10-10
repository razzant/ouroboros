"""Host-owned external workspace execution behind semantic task executor_ref.

Completion belongs to the backend, not its host CLI. Its unique pidfile keeps
wrapper wait evidence; failed readback retains task-attributed executor custody.
Stop also requires a backend receipt. Live children survive exception unwind
in these records independently of the closed tool invocation. Root join proves
nothing about escaped descendants; pre-spawn failure starts nothing.
"""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import posixpath
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ouroboros.observability import redact_projection
from ouroboros.platform_layer import (
    IS_WINDOWS,
    bootstrap_process_path,
    kill_pid_tree, kill_process_group_id, kill_process_tree,
    pid_is_signalable, pid_provably_gone,
    process_command, process_group_id,
    request_process_tree_kill,
    scrub_repo_from_pythonpath, subprocess_new_group_kwargs,
)
from ouroboros.tool_access import path_is_relative_to
from ouroboros.utils import atomic_write_json, utc_now_iso

@dataclass(frozen=True)
class PathMapping:
    host_path: pathlib.Path
    backend_path: str

@dataclass(frozen=True)
class ExecutorRef:
    kind: str
    executor_id: str
    network: str
    mappings: tuple[PathMapping, ...]
    container_name: str = ""

@dataclass
class ExecutorResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    backend_trace: dict[str, Any] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)
    operation_outcome: str = "unknown"

@dataclass
class _ExecutorService:
    service_id: str
    task_id: str
    name: str
    executor: ExecutorRef
    cmd: list[str]
    host_cwd: pathlib.Path
    backend_cwd: str
    cwd_root: str
    outputs: list[str]
    before_outputs: dict[str, tuple[bool, int, str]]
    cwd_base: str = ""
    cwd_source: str = ""
    skill_name: str = ""
    keep_alive: bool = False
    started_at: float = field(default_factory=time.time)
    local_proc: subprocess.Popen | None = None
    backend_pid: str = ""
    backend_log_path: str = ""
    readiness: dict[str, Any] = field(default_factory=dict)
    ready: bool = False
    ready_observed_at: str = ""
    readiness_log_offset: int = 0
    readiness_log_carry: bytes = b""
    readiness_log_identity: tuple[int, int] | None = None
    durable_record_path: pathlib.Path | None = None
    env: dict[str, str] = field(default_factory=dict, repr=False)
    secret_values: tuple[str, ...] = field(default=(), repr=False)
    drive_root: pathlib.Path | None = None


_SERVICES: dict[str, _ExecutorService] = {}
_FOREGROUND: dict[subprocess.Popen, tuple[pathlib.Path | None, str]] = {}
_STATE_LOCK = threading.RLock()
_panic_requested = False
_MAX_SERVICE_LOG_TAIL_CHARS = 80_000
_READINESS_SCAN_CHUNK_BYTES = 64 * 1024
_PROCESS_STATE_DIR = "workspace_executor_processes"
_PROCESS_RECORD_OWNER = "ouroboros_workspace_executor"
_PROCESS_RECORD_SCHEMA_VERSION = 1

def executor_ref_from_ctx(ctx: Any) -> ExecutorRef | None:
    """Return a normalized executor ref from ToolContext/task metadata."""

    accessor = getattr(ctx, "workspace_executor_ref", None)
    if callable(accessor):
        raw = accessor()
    else:
        raw = getattr(ctx, "executor_ref", None)
        if not isinstance(raw, dict) or not raw:
            metadata = getattr(ctx, "task_metadata", {})
            if isinstance(metadata, dict):
                raw = metadata.get("executor_ref")
    if not isinstance(raw, dict):
        return None
    return normalize_executor_ref(raw)


def normalize_executor_ref(raw: dict[str, Any]) -> ExecutorRef | None:
    if not raw:
        return None
    kind = str(raw.get("type") or raw.get("kind") or "").strip().lower()
    if not kind:
        raise ValueError("executor_ref.type is required")
    if kind not in {"local", "docker_exec"}:
        raise ValueError(f"unsupported executor_ref.type: {kind}")
    network = str(raw.get("network") or "host").strip().lower()
    if network not in {"host", "none"}:
        raise ValueError("executor_ref.network must be 'host' or 'none'")
    if kind == "local" and network == "none":
        raise ValueError("local executor_ref cannot enforce network=none; use docker_exec")

    mappings: list[PathMapping] = []
    workspace_host = str(raw.get("workspace_host_path") or "").strip()
    workspace_backend = str(raw.get("workspace_backend_path") or "").strip()
    if workspace_host and workspace_backend:
        mappings.append(PathMapping(pathlib.Path(workspace_host).expanduser().resolve(strict=False), _normalize_backend_path(workspace_backend)))
    raw_mappings = raw.get("path_mappings") if "path_mappings" in raw else raw.get("mappings")
    if raw_mappings is None:
        raw_mappings = []
    if not isinstance(raw_mappings, list):
        raise ValueError("executor_ref.path_mappings must be a list")
    for item in raw_mappings:
        if not isinstance(item, dict):
            raise ValueError("executor_ref.path_mappings entries must be objects")
        host = str(item.get("host_path") or "").strip()
        backend = str(item.get("backend_path") or "").strip()
        if not host or not backend:
            raise ValueError("executor_ref.path_mappings entries require host_path and backend_path")
        mappings.append(PathMapping(pathlib.Path(host).expanduser().resolve(strict=False), _normalize_backend_path(backend)))
    if not mappings:
        raise ValueError("executor_ref requires at least one host/backend path mapping")
    if kind == "docker_exec" and not str(raw.get("container_name") or raw.get("container") or "").strip():
        raise ValueError("docker_exec executor_ref requires container_name")
    return ExecutorRef(
        kind=kind,
        executor_id=str(raw.get("id") or raw.get("container_name") or uuid.uuid4().hex[:12]),
        network=network,
        mappings=tuple(_dedupe_mappings(mappings)),
        container_name=str(raw.get("container_name") or raw.get("container") or "").strip(),
    )


def _normalize_backend_path(path_text: str) -> str:
    normalized = str(path_text or "").replace("\\", "/").strip()
    if not normalized.startswith("/"):
        raise ValueError("executor_ref backend_path must be an absolute backend path")
    raw_parts = [part for part in normalized.split("/") if part]
    if any(part in {".", ".."} for part in raw_parts):
        raise ValueError("executor_ref backend_path must not contain traversal segments")
    normalized = posixpath.normpath(normalized)
    if normalized in {"", ".", "/"}:
        raise ValueError("executor_ref backend_path must not be empty or backend root")
    return normalized


def _dedupe_mappings(mappings: list[PathMapping]) -> list[PathMapping]:
    seen: set[tuple[str, str]] = set()
    result: list[PathMapping] = []
    for mapping in mappings:
        key = (str(mapping.host_path), mapping.backend_path.rstrip("/"))
        if key in seen:
            continue
        seen.add(key)
        result.append(mapping)
    result.sort(key=lambda item: len(str(item.host_path)), reverse=True)
    return result


def map_host_path(executor: ExecutorRef, path: pathlib.Path) -> str:
    host = pathlib.Path(path).expanduser().resolve(strict=False)
    for mapping in executor.mappings:
        if not path_is_relative_to(host, mapping.host_path):
            continue
        rel = host.relative_to(mapping.host_path).as_posix()
        base = mapping.backend_path.rstrip("/")
        return base if not rel or rel == "." else f"{base}/{rel}"
    raise ValueError(f"path is outside executor mappings: {host}")


def map_backend_path(executor: ExecutorRef, path_text: str) -> pathlib.Path:
    return map_backend_path_lexical(executor, path_text).resolve(strict=False)


def map_backend_path_lexical(executor: ExecutorRef, path_text: str) -> pathlib.Path:
    """Map a backend path without resolving descendant symlinks.

    Policy callers need the lexical spelling to detect a path that originated
    under a configured output root before canonicalization can move it into a
    different allowed root.  Execution and ordinary path consumers continue to
    use ``map_backend_path`` and receive the resolved host path.
    """
    normalized = str(path_text or "").replace("\\", "/").rstrip("/")
    if not normalized.startswith("/"):
        raise ValueError(f"backend path is not absolute: {path_text}")
    for mapping in sorted(executor.mappings, key=lambda item: len(item.backend_path.rstrip("/")), reverse=True):
        base = str(mapping.backend_path or "").replace("\\", "/").rstrip("/")
        if not base:
            continue
        if normalized != base and not normalized.startswith(base + "/"):
            continue
        rel_text = normalized[len(base):].lstrip("/")
        rel_parts = [part for part in rel_text.split("/") if part and part != "."]
        if any(part == ".." for part in rel_parts):
            raise ValueError(f"backend path escapes executor mapping: {path_text}")
        return mapping.host_path.joinpath(*rel_parts)
    raise ValueError(f"path is outside executor backend mappings: {path_text}")


def execute(
    ctx: Any,
    cmd: list[str],
    cwd: pathlib.Path,
    timeout_sec: int,
    *,
    env_overlay: "dict[str, str] | None" = None,
    target_env: "dict[str, str] | None" = None,
) -> ExecutorResult:
    """Run in the configured backend. Host/interpreter ``env_overlay`` applies
    only locally: host paths/PATH must not leak into Docker. Explicit target_env
    is separate and reaches either backend, through inert aliases in Docker.
    In a body candidate or system copy, Docker must map the source copy and its sibling ``.env`` directory (else refused)."""
    executor = executor_ref_from_ctx(ctx)
    if executor is None:
        raise ValueError("no executor_ref configured")
    bootstrap_process_path()
    cwd_path = pathlib.Path(cwd).resolve(strict=False)
    backend_cwd = map_host_path(executor, cwd_path)
    from ouroboros.body_candidate import executor_environment
    candidate_env = executor_environment(ctx, cwd_path, executor=executor, map_path=map_host_path)
    if executor.kind == "local":
        return _execute_local(
            executor, cmd, cwd_path, timeout_sec,
            drive_root=_drive_root_from_ctx(ctx), local_env=candidate_env,
            env_overlay=env_overlay, **({"target_env": target_env} if target_env else {}),
        )
    return _execute_docker(executor, cmd, backend_cwd, timeout_sec, drive_root=_drive_root_from_ctx(ctx),
                           **({"target_env": overlay_env(candidate_env or {}, target_env), "replace_env": True}
                              if candidate_env is not None else {"target_env": target_env} if target_env else {}))


def _system_repo_dir() -> str | None:
    """Resolve the Ouroboros system repo dir for PYTHONPATH scrubbing. Executor
    backends always run EXTERNAL-workspace commands, so this repo entry is the
    one to strip (R2). Env first (set at server startup), then config."""
    repo = (os.environ.get("OUROBOROS_REPO_DIR") or "").strip()
    if repo:
        return repo
    try:
        from ouroboros import config

        return str(config.REPO_DIR)
    except Exception:
        return None


def overlay_env(base: "dict[str, str]", env_overlay: "dict[str, str] | None") -> dict[str, str]:
    """Overlay executor/companion environments: Windows keys ignore case, POSIX do not.
    Replace existing casing instead of duplicating Path/PATH. Plain-host PATH
    dedupe remains in ``process_interpreters.apply_env_path_prepend``.
    """
    env = dict(base)
    for key, value in (env_overlay or {}).items():
        if IS_WINDOWS:
            for existing in [k for k in env if k.upper() == key.upper() and k != key]:
                del env[existing]
        env[key] = str(value)
    return env


def _execute_local(
    executor: ExecutorRef,
    cmd: list[str],
    cwd: pathlib.Path,
    timeout_sec: int,
    *,
    drive_root: pathlib.Path | None,
    env_overlay: "dict[str, str] | None" = None,
    target_env: "dict[str, str] | None" = None,
    local_env: "dict[str, str] | None" = None,
) -> ExecutorResult:
    if _panic_requested:
        raise RuntimeError("Emergency Stop has retired executor admission")
    started = time.monotonic()
    from ouroboros.owner_pause import operation_start
    from ouroboros.settings_integrity import runtime_environ

    base_env = local_env if local_env is not None else scrub_repo_from_pythonpath(runtime_environ(), _system_repo_dir())
    process_env = overlay_env(overlay_env(base_env, target_env), env_overlay)
    with operation_start():
        try:
            proc = subprocess.Popen(
                [str(part) for part in cmd],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                errors="replace",
                env=process_env,
                **subprocess_new_group_kwargs(),
            )
        except (OSError, ValueError) as exc:
            exc.process_not_started = True
            raise
    _FOREGROUND[proc] = (None, executor.kind)
    if _panic_requested:
        request_process_tree_kill(proc)
    record_path = _register_process(
        drive_root,
        {
            "record_type": "foreground",
            "executor_type": executor.kind,
            "executor_id": executor.executor_id,
            "host_pid": proc.pid,
            "cwd": str(cwd),
            "cmd": _redacted_cmd(cmd),
        },
    )
    _FOREGROUND[proc] = (record_path, executor.kind)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return ExecutorResult(
            proc.returncode,
            stdout or "",
            stderr or "",
            _trace(executor, str(cwd), cmd, proc.returncode, started),
            [str(part) for part in cmd],
            operation_outcome="completed",
        )
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        proc.wait(timeout=5)
        raise
    finally:
        _FOREGROUND.pop(proc, None)
        if proc.poll() is not None:
            _forget_process(record_path)  # Exceptions cannot discard a live child.


def _execute_docker(
    executor: ExecutorRef,
    cmd: list[str],
    backend_cwd: str,
    timeout_sec: int,
    *,
    drive_root: pathlib.Path | None,
    target_env: "dict[str, str] | None" = None,
    replace_env: bool = False,
) -> ExecutorResult:
    if _panic_requested:
        raise RuntimeError("Emergency Stop has retired executor admission")
    if executor.network == "none":
        _assert_docker_network_none(executor.container_name)
    pidfile = f"/tmp/ouroboros-exec-{uuid.uuid4().hex}.pid"
    prefix = f"OUROBOROS_PROCESS_ENV_{uuid.uuid4().hex}_"
    aliases = {key: f"{prefix}{index}" for index, key in enumerate(target_env or {})}
    command = _docker_env_command(shlex.join(str(part) for part in cmd), aliases, replace_env=replace_env)
    exec_payload = shlex.quote(f"exec {command}")
    quoted_pidfile = shlex.quote(pidfile)
    wrapper = (
        f"rm -f {quoted_pidfile}; "
        "if command -v setsid >/dev/null 2>&1; then "
        f"setsid sh -c {exec_payload} & "
        "else "
        f"sh -c {exec_payload} & "
        "fi; "
        f"pid=$!; echo $pid > {quoted_pidfile}; "
        "wait $pid; rc=$?; "
        # The same owned pidfile retains the wrapper's positive wait evidence.
        # Absence (including a CLI that died before launch) proves nothing.
        f"echo completed > {quoted_pidfile}; "
        "exit $rc"
    )
    docker_cmd = [
        "docker",
        "exec",
        *[part for alias in aliases.values() for part in ("--env", alias)],
        "--workdir",
        backend_cwd,
        executor.container_name,
        "sh",
        "-lc",
        wrapper,
    ]
    started = time.monotonic()
    from ouroboros.owner_pause import operation_start

    with operation_start():
        try:
            proc = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                stdin=subprocess.DEVNULL,
                **({"env": {**os.environ, **{aliases[key]: value for key, value in target_env.items()}}} if target_env else {}),
                **subprocess_new_group_kwargs(),
            )
        except (OSError, ValueError) as exc:
            exc.process_not_started = True
            raise
    _FOREGROUND[proc] = (None, executor.kind)
    if _panic_requested:
        request_process_tree_kill(proc)
    record_path = _register_process(
        drive_root,
        {
            "record_type": "foreground",
            "executor_type": executor.kind,
            "executor_id": executor.executor_id,
            "host_pid": proc.pid,
            "container_name": executor.container_name,
            "backend_pidfile": pidfile,
            "backend_cwd": backend_cwd,
            "cmd": _redacted_cmd(cmd),
        },
    )
    _FOREGROUND[proc] = (record_path, executor.kind)
    cleanup_confirmed = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        cleanup_confirmed = _docker_exec_completed(executor.container_name, pidfile)
    except subprocess.TimeoutExpired:
        cleanup_confirmed = _cleanup_docker_exec_timeout(executor.container_name, pidfile)
        kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        raise
    finally:
        _FOREGROUND.pop(proc, None)
        if cleanup_confirmed:
            _retire_docker_completion(record_path)
    return ExecutorResult(proc.returncode, stdout or "", stderr or "", _trace(executor, backend_cwd, cmd, proc.returncode, started), [str(part) for part in cmd],
                          operation_outcome="completed" if cleanup_confirmed else "unknown")


def _docker_exec_completed(container_name: str, pidfile: str) -> bool:
    """Read the backend wrapper's wait fact, never infer it from CLI exit."""
    quoted = shlex.quote(pidfile)
    try:
        proc = subprocess.run(
            ["docker", "exec", container_name, "sh", "-lc",
             f'[ "$(cat {quoted})" = completed ] && printf completed'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5,
        )
        return proc.returncode == 0 and proc.stdout == "completed"
    except Exception:
        return False


def _retire_docker_completion(record_path: pathlib.Path | None) -> bool:
    """Persist the observed fact before deleting its backend receipt.

    An ambiguous deletion is retried from this same record without signaling
    a potentially reused host PID. Missing host authority retains the marker.
    """
    record = _load_process_record(record_path) if record_path else None
    if not record:
        return False
    try:
        if record.get("backend_completed") is not True:
            record["backend_completed"] = True
            atomic_write_json(record_path, record, trailing_newline=True)
        if not pid_provably_gone(int(record.get("host_pid") or 0)):
            return False  # Backend completion does not join a still-live CLI.
        proc = subprocess.run(["docker", "exec", record["container_name"], "sh", "-lc",
            "rm -f -- " + shlex.quote(record["backend_pidfile"])],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        if proc.returncode != 0:
            return False
        record_path.unlink(missing_ok=True)
        from ouroboros.owned_shutdown import forget_executor_process

        forget_executor_process(record_path)
        return True
    except Exception:
        return False


def _cleanup_docker_exec_timeout(container_name: str, pidfile: str) -> bool:
    shell = _docker_exec_pidfile_stop_shell(pidfile)
    try:
        bootstrap_process_path()
        proc = subprocess.run(
            ["docker", "exec", container_name, "sh", "-lc", shell],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=5,
        )
        return proc.returncode == 0 and proc.stdout == "completed"
    except Exception:
        return False

def _docker_exec_pidfile_stop_shell(pidfile: str) -> str:
    quoted_pidfile = shlex.quote(pidfile)
    return (
        f"pid=$(cat {quoted_pidfile} 2>/dev/null || true); "
        'case "$pid" in completed) printf completed; exit $?;; '
        "''|*[!0-9]*) exit 1;; esac; "
        "kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null || true; "
        "sleep 0.5; "
        "kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true; "
        "if kill -0 -$pid 2>/dev/null || kill -0 $pid 2>/dev/null; then exit 1; fi; "
        f"printf completed > {quoted_pidfile} && printf completed"
    )

def _docker_record_stop_shell(record: dict[str, Any]) -> str:
    pidfile = str(record.get("backend_pidfile") or "").strip()
    if pidfile:
        return _docker_exec_pidfile_stop_shell(pidfile)
    backend_pid = str(record.get("backend_pid") or "").strip()
    return _docker_service_stop_shell(backend_pid) if backend_pid else ""

def _dispatch_docker_record_cleanup(record: dict[str, Any]) -> bool:
    container = str(record.get("container_name") or "").strip()
    shell = _docker_record_stop_shell(record)
    if not container or not shell:
        return True
    try:
        bootstrap_process_path()
        proc = subprocess.run(
            ["docker", "exec", container, "sh", "-lc", shell],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=5,
        )
        if proc.returncode != 0:
            return False
        # The bounded shell dispatch is only an attempt.  Service records carry
        # a backend PID, so panic/``wait=False`` cleanup performs the same
        # explicit terminal probe before allowing the durable record to go.
        backend_pid = str(record.get("backend_pid") or "").strip()
        if backend_pid:
            return _docker_pid_state(container, backend_pid) == "exited"
        return proc.stdout == "completed"
    except Exception:
        return False

def _drive_root_from_ctx(ctx: Any) -> pathlib.Path | None:
    drive_root = getattr(ctx, "drive_root", None)
    if drive_root in (None, ""):
        return None
    try:
        return pathlib.Path(drive_root).resolve(strict=False)
    except Exception:
        return None

def _state_dir(drive_root: pathlib.Path | None) -> pathlib.Path | None:
    if drive_root is None:
        return None
    try:
        path = pathlib.Path(drive_root).resolve(strict=False) / "state" / _PROCESS_STATE_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None

def _safe_record_id(prefix: str) -> str:
    text = f"{prefix}-{uuid.uuid4().hex}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _services_snapshot() -> list[_ExecutorService]:
    with _STATE_LOCK:
        return list(_SERVICES.values())


def _register_process(drive_root: pathlib.Path | None, payload: dict[str, Any]) -> pathlib.Path | None:
    from ouroboros.tool_custody import invocation_binding, retain_unconfirmed_host_operation

    binding = invocation_binding()
    payload = {**{key: value for key, value in binding.items() if key != "drive_root"}, **payload}
    state_dir = _state_dir(pathlib.Path(binding["drive_root"]) if binding.get("drive_root") else drive_root)
    if state_dir is None:
        if binding.get("task_id"):
            retain_unconfirmed_host_operation("executor_custody_unavailable")
        return None
    record_id = _safe_record_id(str(payload.get("record_type") or "process"))
    host_command_sha256 = ""
    try:
        host_pid = int(payload.get("host_pid") or 0)
    except (TypeError, ValueError):
        host_pid = 0
    if host_pid > 0:
        host_command_sha256 = _process_command_sha256(host_pid)
    path = state_dir / f"{record_id}.json"
    record = {
        "schema_version": _PROCESS_RECORD_SCHEMA_VERSION,
        "owner": _PROCESS_RECORD_OWNER,
        "id": record_id,
        "created_at": utc_now_iso(),
        **payload,
    }
    if host_command_sha256:
        record["host_command_sha256"] = host_command_sha256
    try:
        atomic_write_json(path, record, trailing_newline=True)
    except Exception:
        retain_unconfirmed_host_operation("executor_custody_write_failed")
        return None
    from ouroboros.owned_shutdown import record_executor_process

    if not record_executor_process(path, record):  # the exit reads the ownership set, never a walk
        retain_unconfirmed_host_operation("executor_custody_write_failed")
    return path


def _register_service_process(drive_root: pathlib.Path | None, record: _ExecutorService) -> pathlib.Path | None:
    return _register_process(
        drive_root,
        {
            "record_type": "service",
            "service_id": record.service_id,
            "task_id": record.task_id,
            "name": record.name,
            "executor_type": record.executor.kind,
            "executor_id": record.executor.executor_id,
            "host_pid": int(record.backend_pid) if record.executor.kind == "local" and str(record.backend_pid).isdigit() else 0,
            "container_name": record.executor.container_name,
            "backend_pid": record.backend_pid,
            "backend_log_path": record.backend_log_path,
            "backend_cwd": record.backend_cwd,
            "host_cwd": str(record.host_cwd),
            "cwd_root": record.cwd_root,
            "cwd_base": record.cwd_base,
            "cwd_source": record.cwd_source,
            "skill_name": record.skill_name,
            "cmd": _redacted_cmd(record.cmd, record.secret_values),
            "keep_alive": bool(record.keep_alive),
        },
    )


def _forget_process(record_path: pathlib.Path | None) -> None:
    if record_path is None:
        return
    try:
        record_path.unlink(missing_ok=True)
    except Exception:
        return  # still on disk, so the ownership set keeps naming it
    from ouroboros.owned_shutdown import forget_executor_process

    forget_executor_process(record_path)


def _load_process_record(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _process_command_sha256(pid: int) -> str:
    try:
        command = process_command(int(pid))
    except Exception:
        command = ""
    if not command:
        return ""
    return hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()


def _host_pid_matches_record(record: dict[str, Any]) -> bool:
    try:
        host_pid = int(record.get("host_pid") or 0)
    except (TypeError, ValueError):
        return False
    if host_pid <= 0:
        return False
    expected = str(record.get("host_command_sha256") or "").strip()
    if not expected:
        # Windows has no POSIX command capture; macOS ps can also miss a spawn.
        # Owner/schema/id were validated. Fallback requires signalability:
        # pid_is_alive includes EPERM, which grants no right to kill. This
        # protects foreign processes only for non-root (root can signal all).
        return pid_is_signalable(host_pid)
    return _process_command_sha256(host_pid) == expected


def _valid_process_record(path: pathlib.Path, record: dict[str, Any], *, check_identity: bool = True) -> bool:
    if record.get("owner") != _PROCESS_RECORD_OWNER:
        return False
    try:
        if int(record.get("schema_version") or 0) != _PROCESS_RECORD_SCHEMA_VERSION:
            return False
    except (TypeError, ValueError):
        return False
    record_id = str(record.get("id") or "").strip()
    if record_id != path.stem:
        return False
    record_type = str(record.get("record_type") or "")
    executor_type = str(record.get("executor_type") or "")
    if record_type not in {"foreground", "service"} or executor_type not in {"local", "docker_exec"}:
        return False
    if executor_type == "docker_exec":
        container_name = str(record.get("container_name") or "").strip()
        if not container_name:
            return False
        pidfile = str(record.get("backend_pidfile") or "").strip()
        backend_pid = str(record.get("backend_pid") or "").strip()
        if record_type == "foreground":
            if not pidfile.startswith("/tmp/ouroboros-exec-") or not pidfile.endswith(".pid"):
                return False
        elif not backend_pid.isdigit():
            return False
    else:
        if check_identity and not _host_pid_matches_record(record):
            return False
    return True


def _owned_process_records(drive_root: pathlib.Path | None, record_type: str) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    """Records the ownership set names, plus this process's in-memory foreground records: no tree walk."""
    from ouroboros.owned_shutdown import executor_record_paths

    paths = executor_record_paths(drive_root, record_type) if drive_root is not None else []
    with _STATE_LOCK:
        paths += [path for path, _kind in _FOREGROUND.copy().values() if path is not None]
    records: list[tuple[pathlib.Path, dict[str, Any]]] = []
    for path in dict.fromkeys(paths):
        record = _load_process_record(path)
        if record is not None and record.get("record_type") == record_type and _valid_process_record(path, record):
            records.append((path, record))
    return records


def _stop_record_process(path: pathlib.Path, record: dict[str, Any], *, wait: bool = True) -> bool:
    """One durable record's typed kill; True once dispatched (Docker: once its backend receipt confirms)."""
    if record.get("executor_type") != "docker_exec":
        _kill_host_pid(record.get("host_pid"))
        return True
    if record.get("record_type") == "foreground" and record.get("backend_completed") is True:
        return _retire_docker_completion(path)
    dispatched = _kill_docker_record(record, wait=wait)
    if dispatched and record.get("record_type") == "foreground" and record.get("backend_pidfile"):
        dispatched = _retire_docker_completion(path)
    return dispatched


def _settle_durable_record(path: pathlib.Path, record: dict[str, Any], *, wait: bool) -> dict[str, Any]:
    dispatched = _stop_record_process(path, record, wait=wait)
    if dispatched:
        _forget_process(path)
    state = "cleanup_pending" if record.get("executor_type") == "docker_exec" and not dispatched else "stopped"
    if record.get("record_type") == "foreground":
        return {"record_type": "foreground", "id": record.get("id"), "executor_type": record.get("executor_type"),
                "cleanup_dispatched": dispatched, "state": state}
    return {"record_type": "service", "service_id": record.get("service_id"), "name": record.get("name"),
            "task_id": record.get("task_id"), "state": state,
            "executor": {"id": record.get("executor_id"), "type": record.get("executor_type")},
            "cleanup_dispatched": dispatched, "durable_cleanup": True}


def _kill_host_pid(host_pid: Any) -> None:
    try:
        pid = int(host_pid)
    except (TypeError, ValueError):
        return
    if pid <= 0:
        return
    if not IS_WINDOWS:
        pgid = process_group_id(pid)
        if pgid:
            kill_process_group_id(pgid)
    kill_pid_tree(pid)


def _kill_docker_record(record: dict[str, Any], *, wait: bool = True) -> bool:
    if not wait:
        return _dispatch_docker_record_cleanup(record)
    container = str(record.get("container_name") or "").strip()
    if not container:
        return True
    pidfile = str(record.get("backend_pidfile") or "").strip()
    if pidfile:
        return _cleanup_docker_exec_timeout(container, pidfile)
    backend_pid = str(record.get("backend_pid") or "").strip()
    if backend_pid:
        try:
            bootstrap_process_path()
            proc = subprocess.run(
                ["docker", "exec", container, "sh", "-lc", _docker_service_stop_shell(backend_pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
        except Exception:
            return False
        if proc.returncode != 0:
            return False
        if _docker_pid_state(container, backend_pid) != "exited":
            return False
    return True


def _docker_pid_state(container_name: str, backend_pid: str) -> str:
    """Probe Docker and preserve ``unknown`` when the probe is inconclusive."""

    pid = str(backend_pid or "").strip()
    if not pid:
        return "unknown"
    try:
        bootstrap_process_path()
        proc = subprocess.run(
            [
                "docker",
                "exec",
                str(container_name),
                "sh",
                "-lc",
                f"kill -0 {shlex.quote(pid)} 2>/dev/null && echo running || echo exited",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    raw_output = proc.stdout or ""
    if isinstance(raw_output, bytes):
        raw_output = raw_output.decode("utf-8", errors="replace")
    output = str(raw_output).strip()
    if output == "running":
        return "running"
    if output == "exited":
        return "exited"
    return "unknown"


def kill_all_foreground(drive_root: pathlib.Path | None = None, *, wait: bool = True, request_only: bool = False) -> list[dict[str, Any]]:
    """Kill durable executor foreground processes for panic/shutdown paths."""
    global _panic_requested
    if request_only:
        _panic_requested = True
        requested = []
        for proc, (_path, kind) in _FOREGROUND.copy().items():
            requested.append({"executor_type": kind, **request_process_tree_kill(proc)})
            if kind == "docker_exec":
                requested.append({"requested": False, "scope": "backend",
                                  "error": "container process requires executor settlement"})
        return requested
    return [_settle_durable_record(path, record, wait=wait)
            for path, record in _owned_process_records(drive_root, "foreground")]


def _assert_docker_network_none(container_name: str) -> None:
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.HostConfig.NetworkMode}}", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker inspect failed for executor container {container_name}: {proc.stderr.strip()}")
    mode = (proc.stdout or "").strip().strip('"').lower()
    if mode != "none":
        raise RuntimeError(f"executor_ref.network=none requires Docker NetworkMode=none, got {mode!r}")


def _trace(executor: ExecutorRef, cwd: str, cmd: list[str], returncode: int | None, started: float) -> dict[str, Any]:
    return {
        "executor_id": executor.executor_id,
        "executor_type": executor.kind,
        "network": executor.network,
        "cwd": cwd,
        "cmd": _redacted_cmd(cmd),
        "returncode": returncode,
        "elapsed_sec": round(max(0.0, time.monotonic() - started), 3),
        "ts": utc_now_iso(),
    }


def _redacted_cmd(cmd: list[str], secret_values: tuple[str, ...] = ()) -> list[str]:
    redacted = redact_projection(_service_diagnostic([str(part) for part in cmd], secret_values)).value
    return [str(part) for part in redacted] if isinstance(redacted, list) else []


def _service_diagnostic(value: Any, secret_values: tuple[str, ...]) -> Any:
    if not secret_values:
        return value
    # Explicit selections need exact-value masking in addition to the normal
    # executor projection. Keep this dependency with that optional capability.
    from ouroboros.secret_masking import redact_known_values

    return redact_known_values(value, secret_values)


def service_key(ctx: Any, name: str) -> str:
    task_id = str(getattr(ctx, "task_id", "") or "manual")
    return f"{task_id}:{name}"


def start_service(
    ctx: Any,
    *,
    name: str,
    cmd: list[str],
    host_cwd: pathlib.Path,
    cwd_root: str,
    readiness: dict[str, Any],
    outputs: list[str],
    before_outputs: dict[str, tuple[bool, int, str]],
    cwd_base: str = "",
    cwd_source: str = "",
    skill_name: str = "",
    keep_alive: bool = False,
    env_overlay: "dict[str, str] | None" = None,
    env: dict[str, str] | None = None,
    secret_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    if _panic_requested:
        raise RuntimeError("Emergency Stop has retired executor admission")
    env = validate_process_env(env)
    executor = executor_ref_from_ctx(ctx)
    if executor is None:
        raise ValueError("no executor_ref configured")
    bootstrap_process_path()
    key = service_key(ctx, name)
    with _STATE_LOCK:
        existing = _SERVICES.get(key)
    if existing is not None:
        if _service_state(existing) == "running":
            return _service_payload(existing, state="running", note="already_running")
        stopped = _stop_service_record(existing)
        if stopped.get("stop_failed"):
            raise RuntimeError("previous service termination is not confirmed: " + str(stopped.get("stop_error") or "unknown"))
    backend_cwd = map_host_path(executor, host_cwd)
    from ouroboros.body_candidate import executor_environment
    candidate_env = executor_environment(ctx, host_cwd, executor=executor, map_path=map_host_path)
    record = _ExecutorService(
        service_id=key,
        task_id=str(getattr(ctx, "task_id", "") or "manual"),
        name=name,
        executor=executor,
        cmd=[str(part) for part in cmd],
        host_cwd=host_cwd,
        backend_cwd=backend_cwd,
        cwd_root=cwd_root,
        cwd_base=str(cwd_base),
        cwd_source=str(cwd_source),
        skill_name=str(skill_name),
        outputs=list(outputs),
        before_outputs=before_outputs,
        keep_alive=bool(keep_alive),
        readiness=dict(readiness or {}),
        env=env,
        secret_values=secret_values,
        drive_root=_drive_root_from_ctx(ctx),
    )
    if executor.kind == "local":
        from ouroboros.process_custody import spawn_supervised

        # A service inside the bound body candidate runs isolated; a refusal precedes any log.
        base_env = candidate_env if candidate_env is not None else _executor_service_env()
        log_path = pathlib.Path(getattr(ctx, "drive_root")) / "services" / record.task_id / f"{name}.executor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("ab")

        def publish_process(proc):
            record.local_proc = proc
            record.backend_pid = str(proc.pid)
            record.backend_log_path = str(log_path)
            _SERVICES[key] = record
            if _panic_requested:
                raise RuntimeError(f"Emergency Stop during service spawn: {request_process_tree_kill(proc)}")

        try:
            spawn_supervised(
                record.cmd, drive_root=pathlib.Path(getattr(ctx, "drive_root")),
                purpose=f"workspace_service:{name}", scope="session" if record.keep_alive else "task",
                owner_task_id=record.task_id, on_spawn=publish_process, cwd=str(host_cwd),
                stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                # Host interpreter overlay applies only to the local executor.
                env=overlay_env(overlay_env(base_env, env_overlay), env),
            )
        finally:
            log_fh.close()
    else:
        if executor.network == "none":
            _assert_docker_network_none(executor.container_name)
        log_path = f"/tmp/ouroboros-service-{record.task_id}-{name}.log"
        prefix = f"OUROBOROS_SERVICE_ENV_{uuid.uuid4().hex}_"
        target_env = overlay_env(candidate_env or {}, env)
        aliases = {key: f"{prefix}{index}" for index, key in enumerate(target_env)}
        shell = _docker_service_start_shell(record, log_path, aliases, replace_env=candidate_env is not None)
        proc = _submit_service_command(
            ["docker", "exec", *[part for alias in aliases.values() for part in ("--env", alias)],
             executor.container_name, "sh", "-lc", shell],
            # Target PATH/DOCKER_HOST/LD_PRELOAD must not reconfigure the host
            # CLI. Only inert aliases cross this hop; values stay out of argv.
            **({"env": {**os.environ, **{aliases[key]: value for key, value in target_env.items()}}} if target_env else {}),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            from ouroboros.tool_custody import retain_unconfirmed_host_operation
            retain_unconfirmed_host_operation("backend_service_start_unconfirmed")
            raise RuntimeError(_service_diagnostic(
                proc.stderr.strip() or proc.stdout.strip() or "docker service start failed", secret_values,
            ))
        lines = (proc.stdout or "").strip().splitlines()
        record.backend_pid = lines[-1].strip() if lines else ""
        if not record.backend_pid.isdigit():
            from ouroboros.tool_custody import retain_unconfirmed_host_operation
            retain_unconfirmed_host_operation("backend_service_identity_unconfirmed")
            raise RuntimeError("backend service identity was not confirmed")
        record.backend_log_path = log_path
        with _STATE_LOCK:
            _SERVICES[key] = record
    record.durable_record_path = _register_service_process(_drive_root_from_ctx(ctx), record)
    _wait_readiness(record, readiness)
    return _service_payload(record)


def _submit_service_command(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Start the Docker CLI under the Pause gate; wait outside it."""
    from ouroboros.owner_pause import operation_start

    with operation_start():
        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except (OSError, ValueError) as exc:
            exc.process_not_started = True
            raise
    try:
        stdout, stderr = proc.communicate(timeout=20)
    except Exception as exc:
        from ouroboros.tool_custody import retain_unconfirmed_host_operation
        retain_unconfirmed_host_operation("backend_service_start_unconfirmed")
        if isinstance(exc, subprocess.TimeoutExpired):
            proc.kill()
            proc.communicate()
        # The host CLI ended; no backend service receipt was obtained.
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def service_status(ctx: Any, name: str) -> dict[str, Any] | None:
    with _STATE_LOCK:
        record = _SERVICES.get(service_key(ctx, name))
    if record is None:
        return None
    return _service_payload(record)


def service_execution_facts(service_id: str) -> dict[str, Any] | None:
    """One executor service's start identity and execution state, without readiness work.

    The local backend's Popen gives the real return code. Docker gives state only:
    its ``kill -0`` probe's own exit status is not the service's, so ``returncode``
    stays ``None`` and an inconclusive probe reads ``unknown``. ``None`` = no record.
    """
    with _STATE_LOCK:
        record = _SERVICES.get(service_id)
    if record is None:
        return None
    if record.executor.kind == "local" and record.local_proc is not None:
        rc = record.local_proc.poll()
        state = "running" if rc is None else "exited"
    else:
        rc, state = None, _safe_service_state(record)
    return {"service_id": service_id, "started_at": record.started_at, "backend_pid": record.backend_pid,
            "state": state, "returncode": rc}


def service_logs(ctx: Any, name: str, tail: int) -> dict[str, Any] | None:
    with _STATE_LOCK:
        record = _SERVICES.get(service_key(ctx, name))
    if record is None:
        return None
    return {
        **_service_payload(record),
        "tail": redact_projection(_service_diagnostic(
            _read_service_tail(record, tail), record.secret_values,
        )).value,
    }


def _stop_service_record(record: _ExecutorService, *, wait: bool = True) -> dict[str, Any]:
    """Stop one owned process and finalize its local log before forgetting env.
    Unconfirmed termination keeps the record for later cleanup; Docker records only dispatch cleanup."""
    def failed(message: str) -> dict[str, Any]:
        payload = _service_payload(record)
        payload.update(stop_failed=True, cleanup_dispatched=False,
                       stop_error=_service_diagnostic(message, record.secret_values))
        return payload

    try:
        if record.executor.kind == "local":
            if record.local_proc is not None and record.local_proc.poll() is None:
                kill_process_tree(record.local_proc)
                if wait:
                    try:
                        record.local_proc.wait(timeout=5)
                    except Exception:
                        pass
            if _safe_service_state(record) != "exited":
                return failed("local service termination is not confirmed")
        else:
            proc = subprocess.run(
                ["docker", "exec", record.executor.container_name, "sh", "-lc", _docker_service_stop_shell(record.backend_pid)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=10 if wait else 5,
            )
            if proc.returncode != 0:
                return failed(proc.stderr.strip() or proc.stdout.strip() or "docker service stop failed")
            probe_state = _safe_service_state(record)
            if probe_state != "exited":
                return failed(f"docker service stop returned success but kill-0 confirmation is {probe_state}")
    except Exception as exc:
        return failed(f"{type(exc).__name__}: {exc}")
    payload = _service_payload(record, state="stopped")
    if record.executor.kind == "docker_exec":
        payload["cleanup_dispatched"] = True
    elif record.drive_root is not None:
        from ouroboros.tools.services import _finalize_service_log_for_drive

        payload["log_finalization"] = _finalize_service_log_for_drive(
            record.drive_root, record, log_path=pathlib.Path(record.backend_log_path),
        )
    with _STATE_LOCK:
        _SERVICES.pop(record.service_id, None)
    _forget_process(record.durable_record_path)
    return payload


def stop_service(ctx: Any, name: str) -> dict[str, Any] | None:
    with _STATE_LOCK:
        record = _SERVICES.get(service_key(ctx, name))
    if record is None:
        return None
    payload = _stop_service_record(record)
    if not payload.get("stop_failed"):
        payload["_before_outputs"] = record.before_outputs
    return payload


def stop_task_services(ctx: Any) -> list[dict[str, Any]]:
    task_id = str(getattr(ctx, "task_id", "") or "manual")
    kept = [
        _service_payload(record, state=_safe_service_state(record), note="keep_alive")
        for record in _services_snapshot()
        if record.task_id == task_id
        and bool(getattr(record, "keep_alive", False))
    ]
    for item in kept:
        item["lifecycle"] = "kept"
    names = [
        record.name
        for record in _services_snapshot()
        if record.task_id == task_id
        and not bool(getattr(record, "keep_alive", False))
    ]
    stopped: list[dict[str, Any]] = []
    for name in names:
        try:
            payload = stop_service(ctx, name)
            if payload is not None:
                payload["lifecycle"] = "stopped"
                stopped.append(payload)
        except Exception:
            pass
    return [*kept, *stopped]


def kill_all_services(
    drive_root: pathlib.Path | None = None,
    *,
    wait: bool = True,
    request_only: bool = False,
    durable: bool = True,
) -> list[dict[str, Any]]:
    """``durable=False``: in-memory services only (the exit stop settles durable records itself)."""
    global _panic_requested
    if request_only:
        _panic_requested = True
        return [
            {"service_id": record.service_id, **(
                request_process_tree_kill(record.local_proc) if record.local_proc is not None
                else {"requested": False, "scope": "backend", "error": "backend-only process requires executor settlement"}
            )}
            for record in _SERVICES.copy().values()
        ]
    stopped = [_stop_service_record(record, wait=wait) for record in _services_snapshot()]
    if durable:
        stopped.extend(_kill_durable_service_records(drive_root, wait=wait))
    return stopped


def _kill_durable_service_records(drive_root: pathlib.Path | None, *, wait: bool = True) -> list[dict[str, Any]]:
    memory_paths = {record.durable_record_path for record in _services_snapshot() if record.durable_record_path is not None}
    return [_settle_durable_record(path, record, wait=wait)
            for path, record in _owned_process_records(drive_root, "service") if path not in memory_paths]


def validate_process_env(value: Any) -> dict[str, str]:
    """Validate an explicit env map without coercing or exposing its values."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("env must be an object of string names and string values")
    for key, item in value.items():
        if not isinstance(key, str) or not key or "=" in key or "\x00" in key:
            raise ValueError("env names must be nonempty strings without '=' or NUL")
        if not isinstance(item, str) or "\x00" in item:
            raise ValueError("env values must be strings without NUL")
    return dict(value)


def resolve_process_env(
    env: Any = None, env_from_settings: Any = None, *, settings: dict[str, Any] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Resolve admitted references over literal env, retaining Settings secrecy.

    The caller owns authority to supply settings. Ordinary literal values and
    ordinary Settings fields are not secrets merely because they are selected.
    """
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.secret_masking import MASKED_SECRET_SETTING_KEYS, is_custom_secret_setting_key

    literal, refs = validate_process_env(env), validate_process_env(env_from_settings)
    selected, secrets = {}, {}
    for name, reference in refs.items():
        if not reference or settings is None or reference not in settings:
            raise ValueError(f"env_from_settings: setting for {name!r} is missing")
        value = settings[reference]
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"env_from_settings: setting for {name!r} must be a string without NUL")
        key = name.upper() if IS_WINDOWS else name
        selected[key] = value
        secrets[key] = (reference in MASKED_SECRET_SETTING_KEYS
                        or is_custom_secret_setting_key(reference, known_setting_keys=SETTINGS_DEFAULTS))
    return overlay_env(literal, selected), tuple(value for key, value in selected.items() if secrets[key] and value)


def service_env() -> dict[str, str]:
    """Compatible minimal host environment; redaction belongs to diagnostics."""
    allowed_exact = {
        "PATH",
        "HOME",
        "USERPROFILE",
        # Login name, not a secret: keychain/identity lookups need it (a CLI
        # whose credentials are keyed by account name finds nothing without it).
        "USER",
        "LOGNAME",
        "USERNAME",
        "APPDATA",
        "LOCALAPPDATA",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "NODE_PATH",
        "SystemRoot",
        "SYSTEMROOT",
        "WINDIR",
        "windir",
        "COMSPEC",
        "ComSpec",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "NUMBER_OF_PROCESSORS",
        "PROGRAMDATA",
        "ProgramData",
        "ProgramFiles",
        "PROGRAMFILES",
        "ProgramFiles(x86)",
        "PROGRAMFILES(X86)",
    }
    allowed_casefold = {key.casefold() for key in allowed_exact}
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.casefold() not in allowed_casefold and not key.startswith("LC_"):
            continue
        env[key] = str(value)
    return env


def _executor_service_env() -> dict[str, str]:
    # External-workspace service: strip the Ouroboros repo from PYTHONPATH so the
    # target project cannot shadow-import Ouroboros's own modules (R2).
    return scrub_repo_from_pythonpath(service_env(), _system_repo_dir())


def _docker_env_command(command: str, aliases: dict[str, str] | None, *, replace_env: bool = False) -> str:
    """Expand selected values only in the target, never in host argv or shell code."""
    if not aliases:
        return command
    unset = "-i" if replace_env else shlex.join([part for alias in aliases.values() for part in ("-u", alias)])
    assignments = " ".join(f'{shlex.quote(key)}="${{{alias}}}"' for key, alias in aliases.items())
    if replace_env:
        assignments = 'PATH="$PATH" ' + assignments  # backend executable search, never the host PATH
    return f"env {unset} -- {assignments} {command}"


def _docker_service_start_shell(record: _ExecutorService, log_path: str, aliases: dict[str, str] | None = None,
                                *, replace_env: bool = False) -> str:
    command = _docker_env_command(shlex.join(record.cmd), aliases, replace_env=replace_env)
    exec_payload = shlex.quote(f"exec {command}")
    quoted_cwd = shlex.quote(record.backend_cwd)
    quoted_log = shlex.quote(log_path)
    return (
        f"cd {quoted_cwd} && "
        "if command -v setsid >/dev/null 2>&1; then "
        f"nohup setsid sh -c {exec_payload} > {quoted_log} 2>&1 & echo $!; "
        "else "
        f"nohup sh -c {exec_payload} > {quoted_log} 2>&1 & echo $!; "
        "fi"
    )


def _docker_service_stop_shell(backend_pid: str) -> str:
    pid = str(backend_pid or "").strip()
    if pid.isdigit():
        return (
            f"pid={pid}; "
            "kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null || true; "
            "sleep 0.5; "
            "kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true; "
            "if kill -0 -$pid 2>/dev/null || kill -0 $pid 2>/dev/null; then exit 1; fi"
        )
    quoted_pid = shlex.quote(pid)
    return (
        f"kill -TERM {quoted_pid} 2>/dev/null || true; "
        f"if kill -0 {quoted_pid} 2>/dev/null; then exit 1; fi"
    )


def _service_payload(record: _ExecutorService, *, state: str | None = None, note: str = "") -> dict[str, Any]:
    actual_state = state if state is not None else _safe_service_state(record)
    if actual_state == "running":
        _refresh_executor_service_readiness(record)
        # Readiness scanning and the state probe are separate operations.  A
        # process can settle during the scan, so re-poll before serializing a
        # running+ready claim (for both local and Docker backends).
        actual_state = _safe_service_state(record)
    else:
        record.ready = False
    if actual_state != "running":
        record.ready = False
    payload = {
        "service_id": record.service_id,
        "name": record.name,
        "task_id": record.task_id,
        "state": actual_state,
        "ready": bool(record.ready),
        "ready_observed_at": getattr(record, "ready_observed_at", "") or None,
        "executor": {
            "id": record.executor.executor_id,
            "type": record.executor.kind,
            "network": record.executor.network,
        },
        "backend_pid": record.backend_pid,
        "backend_cwd": record.backend_cwd,
        "host_cwd": str(record.host_cwd),
        "cwd_root": record.cwd_root,
        "cwd_base": record.cwd_base,
        "cwd_source": record.cwd_source,
        "skill_name": record.skill_name,
        "cmd": _redacted_cmd(record.cmd, getattr(record, "secret_values", ())),
        "outputs": list(record.outputs),
        "keep_alive": bool(record.keep_alive),
        "backend_log_path": record.backend_log_path,
        "uptime_sec": round(max(0.0, time.time() - record.started_at), 3),
        "ts": utc_now_iso(),
    }
    if note:
        payload["note"] = _service_diagnostic(note, getattr(record, "secret_values", ()))
    return payload


def _service_state(record: _ExecutorService) -> str:
    if record.executor.kind == "local":
        proc = record.local_proc
        if proc is None:
            return "exited"
        try:
            return "running" if proc.poll() is None else "exited"
        except Exception:
            return "unknown"
    try:
        proc = subprocess.run(
            ["docker", "exec", record.executor.container_name, "sh", "-lc", f"kill -0 {shlex.quote(record.backend_pid)} 2>/dev/null && echo running || echo exited"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except Exception:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    raw_output = proc.stdout or ""
    if isinstance(raw_output, bytes):
        raw_output = raw_output.decode("utf-8", errors="replace")
    output = str(raw_output).strip()
    if output == "running":
        return "running"
    if output == "exited":
        return "exited"
    return "unknown"


def _safe_service_state(record: _ExecutorService) -> str:
    """Never let an inconclusive state probe discard cleanup custody."""

    try:
        state = _service_state(record)
    except Exception:
        return "unknown"
    return state if state in {"running", "exited", "unknown"} else "unknown"


def _read_service_tail(record: _ExecutorService, chars: int) -> str:
    limit = max(1, min(int(chars or 8000), _MAX_SERVICE_LOG_TAIL_CHARS))
    if record.executor.kind == "local":
        path = pathlib.Path(record.backend_log_path)
        if not path.exists():
            return ""
        with path.open("rb") as fh:
            fh.seek(max(0, path.stat().st_size - limit))
            return fh.read(limit).decode("utf-8", errors="replace")
    proc = subprocess.run(
        ["docker", "exec", record.executor.container_name, "sh", "-lc", f"tail -c {limit} {shlex.quote(record.backend_log_path)} 2>/dev/null || true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=10,
    )
    return proc.stdout or ""


def _wait_readiness(record: _ExecutorService, readiness: dict[str, Any]) -> None:
    contains = str(readiness.get("log_contains") or readiness.get("stdout_contains") or "").strip()
    timeout = min(max(float(readiness.get("timeout_sec") or 0), 0.0), 25.0)
    if not contains:
        record.ready = True
        record.ready_observed_at = getattr(record, "ready_observed_at", "") or utc_now_iso()
        return
    deadline = time.time() + timeout
    while time.time() <= deadline:
        if _executor_readiness_marker_observed(record, contains):
            record.ready = True
            record.ready_observed_at = getattr(record, "ready_observed_at", "") or utc_now_iso()
            return
        if _service_state(record) != "running":
            return
        time.sleep(0.2)


def _executor_readiness_marker_observed(record: _ExecutorService, marker: str) -> bool:
    if record.executor.kind == "local":
        return _read_local_service_marker(record, pathlib.Path(record.backend_log_path), marker)
    return _read_docker_service_marker(record, marker)


def _refresh_executor_service_readiness(record: _ExecutorService) -> None:
    """Refresh a pending readiness probe without rereading the display tail."""

    if record.ready:
        return
    contains = str(record.readiness.get("log_contains") or record.readiness.get("stdout_contains") or "").strip()
    if not contains:
        record.ready = True
        record.ready_observed_at = record.ready_observed_at or utc_now_iso()
        return
    if _executor_readiness_marker_observed(record, contains):
        record.ready = True
        record.ready_observed_at = record.ready_observed_at or utc_now_iso()


def _read_docker_service_marker(record: _ExecutorService, marker: str) -> bool:
    """Read one bounded unseen remote-log chunk and scan it with carry bytes."""

    needle = str(marker).encode("utf-8")
    if not needle:
        return True
    offset = int(getattr(record, "readiness_log_offset", 0) or 0)
    identity = getattr(record, "readiness_log_identity", None)
    carry = bytes(getattr(record, "readiness_log_carry", b"") or b"")
    for attempt in range(2):
        path = shlex.quote(record.backend_log_path)
        shell = (
            f"meta=$(stat -c '%d:%i:%s' {path} 2>/dev/null) || exit 2; "
            "printf '%s\\n' \"$meta\"; "
            f"tail -c +{offset + 1} {path} 2>/dev/null | head -c {_READINESS_SCAN_CHUNK_BYTES}"
        )
        try:
            proc = subprocess.run(
                ["docker", "exec", record.executor.container_name, "sh", "-lc", shell],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
        except Exception:
            return False
        if proc.returncode != 0:
            return False
        output = bytes(proc.stdout or b"")
        header, sep, chunk = output.partition(b"\n")
        if not sep:
            return False
        try:
            dev_text, ino_text, size_text = header.decode("ascii").split(":", 2)
            remote_identity = (int(dev_text), int(ino_text))
            remote_size = int(size_text)
        except (ValueError, UnicodeDecodeError):
            return False
        if identity != remote_identity or remote_size < offset:
            # Rotation/truncation is a new stream.  Same-inode same-size rewrites
            # remain advisory, as they are not distinguishable from ordinary
            # append-only logs without imposing a second state machine.
            identity = remote_identity
            offset = 0
            carry = b""
            record.readiness_log_identity = identity
            record.readiness_log_offset = 0
            record.readiness_log_carry = b""
            if attempt == 0:
                continue
        record.readiness_log_identity = remote_identity
        record.readiness_log_offset = offset + len(chunk)
        window = carry + chunk
        carry_size = max(0, len(needle) - 1)
        record.readiness_log_carry = window[-carry_size:] if carry_size else b""
        return needle in window
    return False


def _read_local_service_marker(record: Any, path: pathlib.Path, marker: str) -> bool:
    """Incrementally scan a local service log, preserving marker-boundary bytes."""

    needle = str(marker).encode("utf-8")
    if not needle:
        return True
    try:
        file_stat = path.stat()
        identity = (int(file_stat.st_dev), int(file_stat.st_ino))
        offset = int(getattr(record, "readiness_log_offset", 0) or 0)
        if getattr(record, "readiness_log_identity", None) != identity or file_stat.st_size < offset:
            record.readiness_log_identity = identity
            record.readiness_log_offset = 0
            record.readiness_log_carry = b""
        with path.open("rb") as fh:
            fh.seek(int(getattr(record, "readiness_log_offset", 0) or 0))
            carry = bytes(getattr(record, "readiness_log_carry", b"") or b"")
            carry_size = max(0, len(needle) - 1)
            while True:
                chunk = fh.read(64 * 1024)
                if not chunk:
                    break
                record.readiness_log_offset = int(getattr(record, "readiness_log_offset", 0) or 0) + len(chunk)
                window = carry + chunk
                if needle in window:
                    record.readiness_log_carry = window[-carry_size:] if carry_size else b""
                    return True
                carry = window[-carry_size:] if carry_size else b""
            record.readiness_log_carry = carry
    except OSError:
        return False
    return False
