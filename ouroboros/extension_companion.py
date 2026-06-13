"""Supervision for long-lived extension companion processes."""

from __future__ import annotations

import logging
import os
import pathlib
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ouroboros.platform_layer import IS_WINDOWS, assign_pid_to_job, close_job, create_kill_on_close_job, kill_process_on_port, kill_process_tree, merge_hidden_kwargs, subprocess_new_group_kwargs, terminate_job, terminate_process_tree
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

_SERVER_PROCESS_PID = int(os.environ.get("OUROBOROS_SERVER_PROCESS_PID") or "-1")
_GLOBAL_SUPERVISOR: Optional["CompanionSupervisor"] = None
_COMPANION_BASE_ENV_KEYS = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "HOME", "USERPROFILE"}


def _companion_base_env() -> Dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _COMPANION_BASE_ENV_KEYS
    }


def _drain_companion_pipe(pipe, cap: int, buf: bytearray, overflow_flag: Dict[str, bool], label: str) -> None:
    """Keep draining forever; preserve only the first ``cap`` bytes."""
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                return
            remaining = cap - len(buf)
            if remaining > 0:
                buf.extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow_flag[label] = True
    except (OSError, ValueError):
        return


@dataclass
class CompanionDescriptor:
    skill_name: str
    name: str
    command: List[str]
    cwd: pathlib.Path
    env: Dict[str, str]
    ports: List[int] = field(default_factory=list)
    restart_policy: str = "on_failure"
    max_restarts: int = 5
    restart_window_sec: float = 300.0
    stdout_cap: int = 2 * 1024 * 1024
    stderr_cap: int = 2 * 1024 * 1024


@dataclass
class CompanionRuntime:
    descriptor: CompanionDescriptor
    process: subprocess.Popen
    started_at: float
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    overflow: Dict[str, bool] = field(default_factory=lambda: {"stdout": False, "stderr": False})
    restart_times: List[float] = field(default_factory=list)
    job_handle: Any = None


def init_server_process_pid(pid: Optional[int] = None) -> None:
    global _SERVER_PROCESS_PID
    _SERVER_PROCESS_PID = int(pid or os.getpid())
    os.environ["OUROBOROS_SERVER_PROCESS_PID"] = str(_SERVER_PROCESS_PID)


def is_server_process() -> bool:
    return os.getpid() == _SERVER_PROCESS_PID


def _port_is_available(port: int) -> bool:
    if not port:
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


class CompanionSupervisor:
    """Owns process lifecycle for extension companions in the server process."""

    def __init__(self, data_dir: pathlib.Path):
        self.data_dir = pathlib.Path(data_dir)
        self._lock = threading.RLock()
        self._runtimes: Dict[str, CompanionRuntime] = {}
        self._restart_history: Dict[str, List[float]] = {}

    def _key(self, skill_name: str, name: str) -> str:
        return f"{skill_name}:{name}"

    def start(self, descriptor: CompanionDescriptor) -> bool:
        """Start a companion process if this is the main server process."""
        if not is_server_process():
            log.debug("Skipping companion start outside server process: %s/%s", descriptor.skill_name, descriptor.name)
            return False
        key = self._key(descriptor.skill_name, descriptor.name)
        with self._lock:
            existing = self._runtimes.get(key)
            if existing and existing.process.poll() is None:
                return True
            for port in descriptor.ports:
                if not _port_is_available(port):
                    raise RuntimeError(f"port {port} is already in use")
            popen_kwargs: Dict[str, Any] = {
                "cwd": str(descriptor.cwd),
                "env": {**_companion_base_env(), **dict(descriptor.env)},
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if not IS_WINDOWS:
                popen_kwargs.update(subprocess_new_group_kwargs())
            popen_kwargs = merge_hidden_kwargs(popen_kwargs)
            proc = subprocess.Popen(descriptor.command, **popen_kwargs)  # noqa: S603
            # Write-through into the custody ledger (daemon scope). Companions
            # survive clean restarts (reconcile re-spawns them), but the reaper
            # now reaps a companion entry when its owner skill is uninstalled OR
            # the entry is from a foreign generation (see process_custody). The
            # launcher-facing extension_companions.json contract stays untouched.
            try:
                from ouroboros.config import DATA_DIR as _data_dir
                from ouroboros.process_custody import record_process

                record_process(
                    pathlib.Path(_data_dir),
                    pid=proc.pid,
                    cmd=list(descriptor.command),
                    purpose=f"companion:{descriptor.skill_name}:{descriptor.name}",
                    scope="daemon",
                )
            except Exception:
                log.debug("companion custody record failed", exc_info=True)
            job_handle = None
            if IS_WINDOWS and os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER") != "1":
                job_handle = create_kill_on_close_job()
                if job_handle is None or not assign_pid_to_job(job_handle, proc.pid):
                    if job_handle is not None:
                        close_job(job_handle)
                    kill_process_tree(proc)
                    raise RuntimeError("failed to assign companion process to Windows Job Object")
            runtime = CompanionRuntime(
                descriptor=descriptor,
                process=proc,
                started_at=time.monotonic(),
                job_handle=job_handle,
            )
            self._runtimes[key] = runtime
            self._start_drainers(runtime)
            threading.Thread(
                target=self._monitor_runtime,
                args=(key, runtime),
                daemon=True,
                name=f"companion-monitor-{descriptor.skill_name}-{descriptor.name}",
            ).start()
            self._write_runtime_snapshot()
            return True

    def _start_drainers(self, runtime: CompanionRuntime) -> None:
        for label, pipe, cap, buf in (
            ("stdout", runtime.process.stdout, runtime.descriptor.stdout_cap, runtime.stdout),
            ("stderr", runtime.process.stderr, runtime.descriptor.stderr_cap, runtime.stderr),
        ):
            if pipe is None:
                continue
            threading.Thread(
                target=_drain_companion_pipe,
                args=(
                    pipe,
                    cap,
                    buf,
                    runtime.overflow,
                    label,
                ),
                daemon=True,
                name=f"companion-{label}-{runtime.descriptor.skill_name}-{runtime.descriptor.name}",
            ).start()

    def _monitor_runtime(self, key: str, runtime: CompanionRuntime) -> None:
        returncode = runtime.process.wait()
        descriptor = runtime.descriptor
        should_restart = False
        with self._lock:
            current = self._runtimes.get(key)
            if current is runtime and descriptor.restart_policy == "on_failure" and returncode != 0:
                now = time.monotonic()
                history = [
                    ts for ts in self._restart_history.get(key, [])
                    if now - ts <= descriptor.restart_window_sec
                ]
                if len(history) < descriptor.max_restarts:
                    history.append(now)
                    self._restart_history[key] = history
                    self._runtimes.pop(key, None)
                    should_restart = True
                else:
                    log.warning(
                        "companion %s/%s exceeded restart limit",
                        descriptor.skill_name,
                        descriptor.name,
                    )
            elif current is runtime:
                self._runtimes.pop(key, None)
        self._write_runtime_snapshot()
        if should_restart:
            time.sleep(0.5)
            try:
                self.start(descriptor)
            except Exception:
                log.warning("failed to restart companion %s/%s", descriptor.skill_name, descriptor.name, exc_info=True)
        if runtime.job_handle is not None:
            close_job(runtime.job_handle)
    def stop(self, skill_name: str, name: str, timeout_sec: float = 5.0) -> None:
        key = self._key(skill_name, name)
        with self._lock:
            runtime = self._runtimes.pop(key, None)
        if not runtime:
            return
        self._terminate_runtime(runtime, timeout_sec=timeout_sec)
        self._write_runtime_snapshot()

    def stop_skill(self, skill_name: str, timeout_sec: float = 5.0) -> None:
        for runtime in list(self.snapshot().values()):
            if runtime.get("skill_name") == skill_name:
                self.stop(skill_name, str(runtime.get("name") or ""), timeout_sec=timeout_sec)

    def stop_all(self, timeout_sec: float = 5.0) -> None:
        for runtime in list(self.snapshot().values()):
            self.stop(str(runtime["skill_name"]), str(runtime["name"]), timeout_sec=timeout_sec)

    def panic_kill_all(self) -> None:
        with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
        for runtime in runtimes:
            try:
                if runtime.job_handle is not None:
                    terminate_job(runtime.job_handle)
                kill_process_tree(runtime.process)
                for port in runtime.descriptor.ports:
                    kill_process_on_port(port)
            finally:
                if runtime.job_handle is not None:
                    close_job(runtime.job_handle)
        self._write_runtime_snapshot()

    def _terminate_runtime(self, runtime: CompanionRuntime, *, timeout_sec: float) -> None:
        proc = runtime.process
        try:
            if proc.poll() is None:
                if runtime.job_handle is not None:
                    terminate_job(runtime.job_handle)
                else:
                    terminate_process_tree(proc)
                deadline = time.monotonic() + max(0.1, timeout_sec)
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if proc.poll() is None:
                    kill_process_tree(proc)
            for port in runtime.descriptor.ports:
                kill_process_on_port(port)
        finally:
            if runtime.job_handle is not None:
                close_job(runtime.job_handle)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                key: self._runtime_snapshot(rt)
                for key, rt in self._runtimes.items()
            }

    @staticmethod
    def _runtime_snapshot(rt: CompanionRuntime) -> Dict[str, Any]:
        return {
            "skill_name": rt.descriptor.skill_name,
            "name": rt.descriptor.name,
            "pid": rt.process.pid,
            "returncode": rt.process.poll(),
            "ports": list(rt.descriptor.ports),
            "started_at_monotonic": rt.started_at,
            "updated_at": utc_now_iso(),
        }

    def _write_runtime_snapshot(self) -> None:
        try:
            atomic_write_json(self.data_dir / "state" / "extension_companions.json", self.snapshot())
        except Exception:
            log.debug("Failed to persist companion runtime snapshot", exc_info=True)


def init_global_supervisor(data_dir: pathlib.Path) -> CompanionSupervisor:
    global _GLOBAL_SUPERVISOR
    init_server_process_pid()
    _GLOBAL_SUPERVISOR = CompanionSupervisor(data_dir)
    return _GLOBAL_SUPERVISOR


def get_global_supervisor() -> Optional[CompanionSupervisor]:
    return _GLOBAL_SUPERVISOR


def snapshot_processes() -> Dict[str, Dict[str, Any]]:
    if _GLOBAL_SUPERVISOR is None:
        return {}
    return _GLOBAL_SUPERVISOR.snapshot()


def panic_kill_all() -> None:
    if _GLOBAL_SUPERVISOR is not None:
        _GLOBAL_SUPERVISOR.panic_kill_all()


__all__ = [
    "CompanionDescriptor", "CompanionSupervisor", "init_global_supervisor",
    "init_server_process_pid", "is_server_process", "panic_kill_all", "snapshot_processes",
]
