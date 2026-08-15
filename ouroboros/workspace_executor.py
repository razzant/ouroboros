"""Host-owned workspace execution backends for external task workspaces.

The task contract stays semantic. This module consumes an operator/runtime
``executor_ref`` from task metadata and routes process execution into the
declared backend when present.
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
    kill_pid_tree,
    kill_process_group_id,
    kill_process_tree,
    pid_is_alive,
    process_command,
    process_group_id,
    scrub_repo_from_pythonpath,
    subprocess_new_group_kwargs,
)
from ouroboros.tool_access import path_is_relative_to
from ouroboros.utils import atomic_write_json, utc_now_iso
from ouroboros.workspace_ref import (
    SshWorkspaceRef,
    normalize_remote_root,
    workspace_ref_for,
)

# Typed unavailability code for the ssh executor kind, meaning EXACTLY ONE thing:
# no remote workspace broker is registered in THIS process (a CLI, a worker before
# the server lifespan, a build without the transport). It is never a fallback and
# never a stand-in for a remote refusal — a registered broker's failures keep their
# own `RemoteWorkspaceError` codes, because "the remote said no" and "this process
# can reach no remote at all" are different answers the owner must tell apart.
SSH_EXECUTOR_UNAVAILABLE = "SSH_EXECUTOR_UNAVAILABLE"


class SshExecutorUnavailableError(RuntimeError):
    """Raised when an ssh execution is requested in a process with no broker.

    Carries the owner ACTION for its code, out of the same register every other
    remote refusal reads (``remote_refusal_actions``). This one reaches the MODEL as
    prose — ``dispatch_prepare`` folds it into ``PreparedCall.unavailable``, which
    lands in the task transcript verbatim — so the reader least able to look a code
    up was the one getting a sentence with no next step in it.
    """

    code = SSH_EXECUTOR_UNAVAILABLE

    def __init__(self, detail: str = ""):
        from ouroboros.remote_refusal_actions import REFUSAL_ACTIONS

        message = SSH_EXECUTOR_UNAVAILABLE
        if detail:
            message = f"{SSH_EXECUTOR_UNAVAILABLE}: {detail}"
        self.action = REFUSAL_ACTIONS.get(SSH_EXECUTOR_UNAVAILABLE.lower(), "retry")
        super().__init__(message)


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
    # ssh projection fields (derived from the persisted WorkspaceRef; an ssh
    # ExecutorRef is never stored — WorkspaceRef is the only durable placement).
    connection_id: str = ""
    remote_root: str = ""

@dataclass
class ExecutorResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    backend_trace: dict[str, Any] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)

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
    ready: bool = False
    durable_record_path: pathlib.Path | None = None


_SERVICES: dict[str, _ExecutorService] = {}
_FOREGROUND: dict[str, pathlib.Path] = {}
_STATE_LOCK = threading.RLock()
_MAX_SERVICE_LOG_TAIL_CHARS = 80_000
_PROCESS_STATE_DIR = "workspace_executor_processes"
_PROCESS_RECORD_OWNER = "ouroboros_workspace_executor"
_PROCESS_RECORD_SCHEMA_VERSION = 1

def executor_ref_from_ctx(ctx: Any) -> ExecutorRef | None:
    """Return the executor projection for a ToolContext/task record.

    RWS3-03 derivation order: a sealed ``WorkspaceRef`` in task metadata is the
    placement authority — an ssh ref yields a DERIVED ssh ExecutorRef (nothing
    stores an ssh executor_ref independently). A local ref (and the legacy
    no-ref case) keeps today's behavior: the operator/runtime ``executor_ref``
    from context/metadata, or None for plain host execution.
    """

    sealed = workspace_ref_for(ctx)
    if isinstance(sealed, SshWorkspaceRef):
        return executor_ref_from_workspace_ref(sealed)
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


def executor_ref_from_workspace_ref(ref: SshWorkspaceRef) -> ExecutorRef:
    """Project the persisted ssh placement into the immutable executor facade ref.

    Public because the Home CHANNELS (attachment staging, media import, the snapshot
    bridge) hold a sealed ``SshWorkspaceRef`` without holding a ``ToolContext`` — task
    admission has a placement long before there is a tool context to read it from.
    Reaching for the private name from those call sites would have made this seam
    private in name only."""

    return ExecutorRef(
        kind="ssh",
        executor_id=ref.workspace_id,
        network="host",
        mappings=(),
        connection_id=ref.connection_id,
        remote_root=ref.remote_root,
    )


def normalize_executor_ref(raw: dict[str, Any]) -> ExecutorRef | None:
    if not raw:
        return None
    kind = str(raw.get("type") or raw.get("kind") or "").strip().lower()
    if not kind:
        raise ValueError("executor_ref.type is required")
    if kind not in {"local", "docker_exec", "ssh"}:
        raise ValueError(f"unsupported executor_ref.type: {kind}")
    network = str(raw.get("network") or "host").strip().lower()
    if network not in {"host", "none"}:
        raise ValueError("executor_ref.network must be 'host' or 'none'")
    if kind == "local" and network == "none":
        raise ValueError("local executor_ref cannot enforce network=none; use docker_exec")
    if kind == "ssh":
        if network == "none":
            raise ValueError("ssh executor_ref cannot enforce network=none")
        connection_id = str(raw.get("connection_id") or "").strip()
        if not connection_id:
            raise ValueError("ssh executor_ref requires connection_id")
        remote_root = normalize_remote_root(raw.get("remote_root"))
        return ExecutorRef(
            kind="ssh",
            executor_id=str(raw.get("id") or raw.get("workspace_id") or connection_id),
            network=network,
            mappings=(),
            connection_id=connection_id,
            remote_root=remote_root,
        )

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
        return (mapping.host_path.joinpath(*rel_parts)).resolve(strict=False)
    raise ValueError(f"path is outside executor backend mappings: {path_text}")


def covers(executor: ExecutorRef | None, path: pathlib.Path) -> bool:
    """Whether the executor's path mappings cover host ``path``.

    The ONE coverage predicate every process surface routes through: True means
    "run this cwd through the executor backend", False (including any mapping
    failure — non-coverage is never an error here) means "fall back to host
    execution"."""
    if executor is None:
        return False
    try:
        map_host_path(executor, pathlib.Path(path).resolve(strict=False))
    except Exception:
        return False
    return True


def covering_host_path(executor: ExecutorRef | None, backend_path_text: str) -> pathlib.Path | None:
    """Guard-side backend→host projection of ``covers``: the host path a
    backend-absolute path maps to when the executor covers it, else ``None``
    (never raises), so shell guards can judge backend tokens by host policy."""
    if executor is None:
        return None
    try:
        return map_backend_path(executor, backend_path_text)
    except Exception:
        return None


def ensure_execution_cwd(executor: ExecutorRef | None, resolved_cwd: pathlib.Path, *, cwd_root: str = "") -> pathlib.Path:
    """Materialize a resolved process cwd — the ONLY place a cwd directory is
    created (``resolve_shell_cwd`` is a pure fact resolver). Called once per
    operation at the execution boundary, after resolution and before any
    exists-check or process start.

    Policy (the resolver's former side effects, unchanged): only the task-owned
    data roots (task_drive / artifact_store) and the unnamed-deliverables
    container are auto-created; every other cwd must already exist. Local and
    docker_exec backends both see host paths (docker mounts the host mapping),
    so host-side mkdir is authoritative for both; a remote backend would own
    materialization here instead — hence the executor parameter."""
    path = pathlib.Path(resolved_cwd)
    label = str(cwd_root or "")
    if label in {"task_drive", "artifact_store"}:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"could not create {label} cwd {path}: {exc}") from exc
    elif label == "user_files":
        from ouroboros.tool_access import _deliverables_root

        if path.resolve(strict=False) == _deliverables_root():
            path.mkdir(parents=True, exist_ok=True)
    return path


def ssh_workspace_ref_payload(executor: ExecutorRef) -> dict[str, str]:
    """Rebuild the sealed placement payload an ssh ExecutorRef was projected FROM.

    The projection is lossless by construction (`_executor_ref_from_workspace_ref`),
    so this is a round-trip, not a second placement authority: the broker is
    addressed by the same descriptor `workspace_admission` sealed.
    """
    if executor.kind != "ssh":
        raise ValueError("ssh_workspace_ref_payload is for ssh placement")
    return SshWorkspaceRef(
        connection_id=executor.connection_id,
        remote_root=executor.remote_root,
        workspace_id=executor.executor_id,
    ).to_payload()


def _remote_service(executor: ExecutorRef, phase: str):
    """The broker facade, or the honest "this process has no transport" refusal.

    `SSH_EXECUTOR_UNAVAILABLE` survives synthesis for exactly ONE condition: no
    broker is registered in this process (a build without the ssh transport, or a
    process that never ran the server lifespan). A registered broker's own
    failures are typed `RemoteWorkspaceError`s and are NOT flattened into this
    code — "the remote refused" and "this build cannot reach any remote" are
    different answers and the owner must be able to tell them apart.
    """
    from ouroboros.remote_workspace import get_remote_workspace_service
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    try:
        return get_remote_workspace_service()
    except RemoteWorkspaceError as exc:
        raise SshExecutorUnavailableError(
            f"no remote workspace broker is registered in this process "
            f"({phase}, connection_id={executor.connection_id}): {exc}"
        ) from exc


def prepare_native_operation(
    executor: ExecutorRef | None,
    tool: str,
    *,
    args: Any = None,
    blobs: Any = None,
    task_id: str = "",
    parent_task_id: str = "",
    project_id: str = "",
    request_id: str = "",
    operation_id: str = "",
    deadline_ms: int | None = None,
) -> Any:
    """ONE bundled prepare for a NATIVE (ssh) operation — the §3.1 step-2 door.

    The target answers with every fact the authorize phase needs in a single
    round trip (cwd kind, git toplevel, interpreter candidates, protected-path
    stats, path canonicalizations) plus the token binding of the
    target-canonical execution args; there is deliberately no per-fact RPC
    (codex #17). Local and docker placements prepare on Home and never call
    this.

    Returns the broker's :class:`~ouroboros.remote_workspace.PreparedRemoteCall`.
    """
    if executor is None or executor.kind != "ssh":
        raise ValueError("prepare_native_operation is for ssh placement; local/docker prepare on Home")
    return _remote_service(executor, "prepare").prepare(
        ssh_workspace_ref_payload(executor),
        request_id=str(request_id or uuid.uuid4().hex[:16]),
        operation_id=str(operation_id or uuid.uuid4().hex[:16]),
        tool=str(tool),
        args=dict(args or {}),
        blobs=dict(blobs or {}),
        deadline_ms=deadline_ms,
        task_id=str(task_id or ""),
        parent_task_id=str(parent_task_id or ""),
        project_id=str(project_id or ""),
    )


def fetch_native_blob(
    executor: ExecutorRef | None,
    blob_id: str,
    *,
    max_bytes: int,
    task_id: str = "",
) -> bytes:
    """Fetch ONE content-addressed blob a prepared operation declared.

    On-demand only, and only for a blob some manifest named: a snapshot's manifest
    arrives first and its entries are what authorize each fetch, so nothing is
    transferred that Home has not already been told the size and hash of."""
    if executor is None or executor.kind != "ssh":
        raise ValueError("fetch_native_blob is for native (ssh) placements")
    return bytes(
        _remote_service(executor, "import").fetch_blob(
            ssh_workspace_ref_payload(executor),
            str(blob_id),
            max_bytes=int(max_bytes),
            task_id=str(task_id or ""),
        )
    )


def abort_prepared_operation(
    executor: ExecutorRef | None,
    prepared: Any,
    *,
    task_id: str = "",
    reason: str = "denied",
) -> bool:
    """Withdraw a prepared native operation that must NOT be executed.

    The third member of the prepare/execute pair, and the one a Home channel needs
    when its own pre-execution check fails: an operation the target has prepared but
    Home refuses must be told so, or the target holds a reserved token and its
    staged blobs until expiry."""
    if executor is None or executor.kind != "ssh":
        raise ValueError("abort_prepared_operation is for native (ssh) prepared operations")
    return bool(
        _remote_service(executor, "authorize").abort_prepared(
            ssh_workspace_ref_payload(executor),
            prepared,
            task_id=str(task_id or ""),
            reason=str(reason or "denied"),
        )
    )


def execute_prepared(
    executor: ExecutorRef | None,
    prepared: Any,
    *,
    canonical_args: Any = None,
    task_id: str = "",
    timeout_sec: float | None = None,
    import_kind: str = "",
    import_context: Any = None,
) -> Any:
    """Execute a token-bound prepared native operation; return its envelope.

    The narrow sibling of ``execute``: the authorized token goes to the target,
    which revalidates the binding and the path confinement as execution
    INTEGRITY — not as a second policy authority. Local/docker keep ``execute``
    unchanged.

    The return value is the target's :class:`ToolExecutionEnvelope`, already
    Home-imported by the transport (declared blobs verified, published through the
    artifact authority, remote provenance kept in the private receipt). It is
    deliberately NOT flattened into an ``ExecutorResult``: the envelope's ``text``
    IS the operation's model-facing answer — for a native read it is the file, and
    for a process it is the ONE rendered process result both placements produce
    (`workspace_native_contract.render_process_result_text`). A returncode-shaped
    projection would throw that text away and make the dispatcher re-render, badly,
    what the target already rendered correctly.

    ``canonical_args`` defaults to the prepared call's own ``execution_args``:
    the argv Home authorized IS the argv that runs, and passing anything else is
    the caller explicitly re-declaring what it authorized (the target rejects a
    mismatch against the token binding).

    ``import_kind``/``import_context`` name which CLOSED import channel the result
    belongs to (`remote_protocol.IMPORT_CHANNELS`). The default is the ordinary tool
    result; a Home channel whose result means something else — an attachment set
    whose manifest must be checked against what Home authorized — declares its own,
    because the import contract is what decides how the returned bytes are believed.
    """
    if executor is None or executor.kind != "ssh":
        raise ValueError("execute_prepared is for native (ssh) prepared operations; local/docker use execute()")
    if prepared is None or not getattr(prepared, "prepared_token", ""):
        raise ValueError("execute_prepared requires a bound PreparedRemoteCall")
    args = canonical_args if canonical_args is not None else getattr(prepared, "execution_args", {})
    return _remote_service(executor, "execute").execute_prepared(
        ssh_workspace_ref_payload(executor),
        prepared,
        canonical_args=dict(args or {}),
        task_id=str(task_id or ""),
        timeout_sec=timeout_sec,
        import_kind=str(import_kind or ""),
        import_context=import_context,
    )


def execute(ctx: Any, cmd: list[str], cwd: pathlib.Path, timeout_sec: int) -> ExecutorResult:
    executor = executor_ref_from_ctx(ctx)
    if executor is None:
        raise ValueError("no executor_ref configured")
    if executor.kind == "ssh":
        # NOT an unavailability: this door takes a Home `cwd` and a Home argv, so
        # an ssh placement reaching it would mean the dispatch skipped the native
        # route — a wiring error, not a missing transport. It is refused the same
        # way `execute_prepared` refuses a local ref: the two doors are exclusive,
        # and `SSH_EXECUTOR_UNAVAILABLE` keeps meaning only "no broker here".
        raise ValueError(
            "ssh placement executes prepared native operations; route the "
            f"operation through execute_prepared() (connection_id={executor.connection_id})"
        )
    bootstrap_process_path()
    cwd_path = pathlib.Path(cwd).resolve(strict=False)
    backend_cwd = map_host_path(executor, cwd_path)
    if executor.kind == "local":
        return _execute_local(executor, cmd, cwd_path, timeout_sec, drive_root=_drive_root_from_ctx(ctx))
    return _execute_docker(executor, cmd, backend_cwd, timeout_sec, drive_root=_drive_root_from_ctx(ctx))


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


def _execute_local(
    executor: ExecutorRef,
    cmd: list[str],
    cwd: pathlib.Path,
    timeout_sec: int,
    *,
    drive_root: pathlib.Path | None,
) -> ExecutorResult:
    started = time.time()
    proc = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        errors="replace",
        env=scrub_repo_from_pythonpath(dict(os.environ), _system_repo_dir()),
        **subprocess_new_group_kwargs(),
    )
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
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return ExecutorResult(
            proc.returncode,
            stdout or "",
            stderr or "",
            _trace(executor, str(cwd), cmd, proc.returncode, started),
            [str(part) for part in cmd],
        )
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        proc.wait(timeout=5)
        raise
    finally:
        _forget_process(record_path)


def _execute_docker(
    executor: ExecutorRef,
    cmd: list[str],
    backend_cwd: str,
    timeout_sec: int,
    *,
    drive_root: pathlib.Path | None,
) -> ExecutorResult:
    if executor.network == "none":
        _assert_docker_network_none(executor.container_name)
    pidfile = f"/tmp/ouroboros-exec-{uuid.uuid4().hex}.pid"
    command = shlex.join(str(part) for part in cmd)
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
        f"rm -f {quoted_pidfile}; "
        "exit $rc"
    )
    docker_cmd = [
        "docker",
        "exec",
        "--workdir",
        backend_cwd,
        executor.container_name,
        "sh",
        "-lc",
        wrapper,
    ]
    started = time.time()
    proc = subprocess.Popen(
        docker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        stdin=subprocess.DEVNULL,
        **subprocess_new_group_kwargs(),
    )
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
    cleanup_confirmed = True
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        cleanup_confirmed = _cleanup_docker_exec_timeout(executor.container_name, pidfile)
        kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        raise
    finally:
        if cleanup_confirmed:
            _forget_process(record_path)
    return ExecutorResult(proc.returncode, stdout or "", stderr or "", _trace(executor, backend_cwd, cmd, proc.returncode, started), [str(part) for part in cmd])


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
        return proc.returncode == 0
    except Exception:
        return False

def _docker_exec_pidfile_stop_shell(pidfile: str) -> str:
    quoted_pidfile = shlex.quote(pidfile)
    return (
        f"pid=$(cat {quoted_pidfile} 2>/dev/null || true); "
        "case \"$pid\" in ''|*[!0-9]*) exit 0;; esac; "
        "kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null || true; "
        "sleep 0.5; "
        "kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true; "
        f"rm -f {quoted_pidfile}"
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
        return proc.returncode == 0
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
    state_dir = _state_dir(drive_root)
    if state_dir is None:
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
        if record.get("record_type") == "foreground":
            with _STATE_LOCK:
                _FOREGROUND[record_id] = path
        return path
    except Exception:
        return None


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
            "cmd": _redacted_cmd(record.cmd),
            "keep_alive": bool(record.keep_alive),
        },
    )


def _forget_process(record_path: pathlib.Path | None) -> None:
    if record_path is None:
        return
    try:
        with _STATE_LOCK:
            _FOREGROUND.pop(record_path.stem, None)
        record_path.unlink(missing_ok=True)
    except Exception:
        pass


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
        # No command line could be captured at register time. This is always the
        # case on Windows, where platform_layer.process_command() is POSIX-only
        # and returns "". Without this fallback the record would be permanently
        # unvalidatable, so kill_all_foreground/_services would never dispatch
        # taskkill for it (the worktree/service cleanup leak). Fall back to a
        # liveness check; owner/schema/id are already verified by the caller
        # (_valid_process_record). The PID-reuse hardening via command-hash
        # comparison still applies on POSIX, where a command line is available.
        return pid_is_alive(host_pid)
    return _process_command_sha256(host_pid) == expected


def _valid_process_record(path: pathlib.Path, record: dict[str, Any]) -> bool:
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
        if not _host_pid_matches_record(record):
            return False
    return True


def _iter_process_records(drive_root: pathlib.Path | None = None) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    roots: list[pathlib.Path] = []
    if drive_root is not None:
        state_dir = _state_dir(drive_root)
        if state_dir is not None:
            roots.append(state_dir)
        try:
            state_root = pathlib.Path(drive_root).resolve(strict=False) / "state"
            if state_root.exists():
                roots.extend(path for path in state_root.rglob(_PROCESS_STATE_DIR) if path.is_dir())
        except Exception:
            pass
    with _STATE_LOCK:
        roots.extend(path.parent for path in _FOREGROUND.values())
    seen: set[pathlib.Path] = set()
    records: list[tuple[pathlib.Path, dict[str, Any]]] = []
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for path in root.glob("*.json"):
            record = _load_process_record(path)
            if record is not None and _valid_process_record(path, record):
                records.append((path, record))
    return records


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
    return True


def kill_all_foreground(drive_root: pathlib.Path | None = None, *, wait: bool = True) -> list[dict[str, Any]]:
    """Kill durable executor foreground processes for panic/shutdown paths."""

    killed: list[dict[str, Any]] = []
    for path, record in _iter_process_records(drive_root):
        if record.get("record_type") != "foreground":
            continue
        cleanup_dispatched = True
        if record.get("executor_type") == "docker_exec":
            cleanup_dispatched = _kill_docker_record(record, wait=wait)
        _kill_host_pid(record.get("host_pid"))
        if cleanup_dispatched:
            _forget_process(path)
        killed.append(
            {
                "record_type": "foreground",
                "id": record.get("id"),
                "executor_type": record.get("executor_type"),
                "cleanup_dispatched": cleanup_dispatched,
                "state": (
                    "cleanup_pending"
                    if record.get("executor_type") == "docker_exec" and not cleanup_dispatched
                    else "stopped"
                ),
            }
        )
    return killed


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
        "elapsed_sec": round(max(0.0, time.time() - started), 3),
        "ts": utc_now_iso(),
    }


def _redacted_cmd(cmd: list[str]) -> list[str]:
    redacted = redact_projection([str(part) for part in cmd]).value
    return [str(part) for part in redacted] if isinstance(redacted, list) else []


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
) -> dict[str, Any]:
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
        _forget_process(existing.durable_record_path)
        with _STATE_LOCK:
            _SERVICES.pop(key, None)
    backend_cwd = map_host_path(executor, host_cwd)
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
    )
    if executor.kind == "local":
        log_path = pathlib.Path(getattr(ctx, "drive_root")) / "services" / record.task_id / f"{name}.executor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("ab")
        from ouroboros.process_custody import spawn_supervised

        proc = spawn_supervised(
            record.cmd,
            drive_root=pathlib.Path(getattr(ctx, "drive_root")),
            purpose=f"workspace_service:{name}",
            scope="session" if record.keep_alive else "task",
            owner_task_id=record.task_id,
            cwd=str(host_cwd),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=_executor_service_env(),
        )
        log_fh.close()
        record.local_proc = proc
        record.backend_pid = str(proc.pid)
        record.backend_log_path = str(log_path)
    else:
        if executor.network == "none":
            _assert_docker_network_none(executor.container_name)
        log_path = f"/tmp/ouroboros-service-{record.task_id}-{name}.log"
        shell = _docker_service_start_shell(record, log_path)
        proc = subprocess.run(
            ["docker", "exec", executor.container_name, "sh", "-lc", shell],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "docker service start failed")
        record.backend_pid = (proc.stdout or "").strip().splitlines()[-1].strip()
        record.backend_log_path = log_path
    record.durable_record_path = _register_service_process(_drive_root_from_ctx(ctx), record)
    with _STATE_LOCK:
        _SERVICES[key] = record
    _wait_readiness(record, readiness)
    return _service_payload(record)


def service_status(ctx: Any, name: str) -> dict[str, Any] | None:
    with _STATE_LOCK:
        record = _SERVICES.get(service_key(ctx, name))
    if record is None:
        return None
    return _service_payload(record)


def service_logs(ctx: Any, name: str, tail: int) -> dict[str, Any] | None:
    with _STATE_LOCK:
        record = _SERVICES.get(service_key(ctx, name))
    if record is None:
        return None
    return {
        **_service_payload(record),
        "tail": str(redact_projection(_read_service_tail(record, tail)).value),
    }


def stop_service(ctx: Any, name: str) -> dict[str, Any] | None:
    key = service_key(ctx, name)
    with _STATE_LOCK:
        record = _SERVICES.get(key)
    if record is None:
        return None
    if record.executor.kind == "local":
        if record.local_proc is not None and record.local_proc.poll() is None:
            kill_process_tree(record.local_proc)
            try:
                record.local_proc.wait(timeout=5)
            except Exception:
                pass
    else:
        try:
            proc = subprocess.run(
                ["docker", "exec", record.executor.container_name, "sh", "-lc", _docker_service_stop_shell(record.backend_pid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            payload = _service_payload(record)
            payload["stop_failed"] = True
            payload["stop_error"] = f"{type(exc).__name__}: {exc}"
            return payload
        if proc.returncode != 0:
            payload = _service_payload(record)
            payload["stop_failed"] = True
            payload["stop_error"] = proc.stderr.strip() or proc.stdout.strip() or "docker service stop failed"
            return payload
    with _STATE_LOCK:
        _SERVICES.pop(key, None)
    _forget_process(record.durable_record_path)
    payload = _service_payload(record, state="stopped")
    payload["_before_outputs"] = record.before_outputs
    return payload


def stop_task_services(ctx: Any) -> list[dict[str, Any]]:
    task_id = str(getattr(ctx, "task_id", "") or "manual")
    kept = [
        _service_payload(record, state=_service_state(record), note="keep_alive")
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
) -> list[dict[str, Any]]:
    records = _services_snapshot()
    stopped: list[dict[str, Any]] = []
    for record in records:
        try:
            if record.executor.kind == "local":
                if record.local_proc is not None and record.local_proc.poll() is None:
                    kill_process_tree(record.local_proc)
                    if wait:
                        try:
                            record.local_proc.wait(timeout=5)
                        except Exception:
                            pass
                with _STATE_LOCK:
                    _SERVICES.pop(record.service_id, None)
                _forget_process(record.durable_record_path)
                stopped.append(_service_payload(record, state="stopped"))
            else:
                proc = subprocess.run(
                    ["docker", "exec", record.executor.container_name, "sh", "-lc", _docker_service_stop_shell(record.backend_pid)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10 if wait else 5,
                )
                payload = _service_payload(record, state="stopped" if proc.returncode == 0 else _service_state(record))
                payload["cleanup_dispatched"] = proc.returncode == 0
                if proc.returncode == 0:
                    with _STATE_LOCK:
                        _SERVICES.pop(record.service_id, None)
                    _forget_process(record.durable_record_path)
                else:
                    payload["stop_failed"] = True
                    payload["stop_error"] = proc.stderr.strip() or proc.stdout.strip() or "docker service stop failed"
                stopped.append(payload)
        except Exception as exc:
            payload = _service_payload(record)
            payload["stop_failed"] = True
            payload["stop_error"] = f"{type(exc).__name__}: {exc}"
            stopped.append(payload)
    stopped.extend(_kill_durable_service_records(drive_root, wait=wait))
    return stopped


def _kill_durable_service_records(drive_root: pathlib.Path | None, *, wait: bool = True) -> list[dict[str, Any]]:
    stopped: list[dict[str, Any]] = []
    memory_paths = {record.durable_record_path for record in _services_snapshot() if record.durable_record_path is not None}
    for path, record in _iter_process_records(drive_root):
        if record.get("record_type") != "service" or path in memory_paths:
            continue
        cleanup_dispatched = True
        if record.get("executor_type") == "docker_exec":
            cleanup_dispatched = _kill_docker_record(record, wait=wait)
        else:
            _kill_host_pid(record.get("host_pid"))
        if cleanup_dispatched:
            _forget_process(path)
        stopped.append(
            {
                "record_type": "service",
                "service_id": record.get("service_id"),
                "name": record.get("name"),
                "task_id": record.get("task_id"),
                "state": (
                    "cleanup_pending"
                    if record.get("executor_type") == "docker_exec" and not cleanup_dispatched
                    else "stopped"
                ),
                "executor": {
                    "id": record.get("executor_id"),
                    "type": record.get("executor_type"),
                },
                "cleanup_dispatched": cleanup_dispatched,
                "durable_cleanup": True,
            }
        )
    return stopped


def _executor_service_env() -> dict[str, str]:
    allowed_exact = {
        "PATH",
        "HOME",
        "USERPROFILE",
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
        try:
            if redact_projection(str(value)).records:
                continue
        except Exception:
            continue
        env[key] = str(value)
    # External-workspace service: strip the Ouroboros repo from PYTHONPATH so the
    # target project cannot shadow-import Ouroboros's own modules (R2).
    return scrub_repo_from_pythonpath(env, _system_repo_dir())


def _docker_service_start_shell(record: _ExecutorService, log_path: str) -> str:
    command = shlex.join(record.cmd)
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
            "kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true"
        )
    quoted_pid = shlex.quote(pid)
    return f"kill -TERM {quoted_pid} 2>/dev/null || true"


def _service_payload(record: _ExecutorService, *, state: str | None = None, note: str = "") -> dict[str, Any]:
    actual_state = state or _service_state(record)
    payload = {
        "service_id": record.service_id,
        "name": record.name,
        "task_id": record.task_id,
        "state": actual_state,
        "ready": bool(record.ready),
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
        "cmd": _redacted_cmd(record.cmd),
        "outputs": list(record.outputs),
        "keep_alive": bool(record.keep_alive),
        "backend_log_path": record.backend_log_path,
        "uptime_sec": round(max(0.0, time.time() - record.started_at), 3),
        "ts": utc_now_iso(),
    }
    if note:
        payload["note"] = note
    return payload


def _service_state(record: _ExecutorService) -> str:
    if record.executor.kind == "local":
        proc = record.local_proc
        return "running" if proc is not None and proc.poll() is None else "exited"
    proc = subprocess.run(
        ["docker", "exec", record.executor.container_name, "sh", "-lc", f"kill -0 {shlex.quote(record.backend_pid)} 2>/dev/null && echo running || echo exited"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    return "running" if "running" in (proc.stdout or "") else "exited"


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
        return
    deadline = time.time() + timeout
    while time.time() <= deadline:
        if contains in _read_service_tail(record, 20_000):
            record.ready = True
            return
        if _service_state(record) != "running":
            return
        time.sleep(0.2)
