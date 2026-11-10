"""Supervision for long-lived extension companion processes."""

from __future__ import annotations

import hmac
import logging
import os
import pathlib
import socket
import subprocess
import threading
import time
import uuid
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from ouroboros.platform_layer import (
    IS_WINDOWS,
    assign_pid_to_job,
    close_job,
    create_kill_on_close_job,
    kill_process_on_port,
    kill_process_tree,
    merge_hidden_kwargs,
    request_process_tree_kill,
    subprocess_new_group_kwargs,
    terminate_job,
    terminate_process_tree,
)
from ouroboros.provider_models import MODEL_PROVIDER_CREDENTIAL_KEYS
from ouroboros.usage_accounting import record_unmetered_external_dispatch
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

_SERVER_PROCESS_PID = int(os.environ.get("OUROBOROS_SERVER_PROCESS_PID") or "-1")
_GLOBAL_SUPERVISOR: Optional["CompanionSupervisor"] = None
# The login identity rides along: CLIs a companion may call (gh, claude, codex,
# cursor-agent) key their keychain/credential lookups on it (parity with
# workspace_executor.service_env()).
_COMPANION_BASE_ENV_KEYS = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "HOME", "USERPROFILE",
                            "USER", "LOGNAME", "USERNAME"}


def _companion_base_env() -> Dict[str, str]:
    from ouroboros.process_scope import installation_scope_entries

    return {**{key: value for key, value in os.environ.items() if key.upper() in _COMPANION_BASE_ENV_KEYS},
            **installation_scope_entries()}


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
    # Non-empty once a failed start or a stop retires this exact runtime: it is
    # never restarted or replaced, and stays owned until its death is observed.
    retiring: str = ""


def init_server_process_pid(pid: Optional[int] = None) -> None:
    global _SERVER_PROCESS_PID
    _SERVER_PROCESS_PID = int(pid or os.getpid())
    os.environ["OUROBOROS_SERVER_PROCESS_PID"] = str(_SERVER_PROCESS_PID)


def is_server_process() -> bool:
    return os.getpid() == _SERVER_PROCESS_PID


def _await_exit(proc: subprocess.Popen, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    return proc.poll() is not None


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


def _setting_text(value: Any) -> bytes:
    return ("" if value is None else str(value)).strip().encode("utf-8")


def _carries(env: Mapping[str, str], expected: Mapping[str, Any]) -> bool:
    """Whether ``env`` holds each value the caller read (``expected``).

    Surrounding blanks aside, an empty value is an absent one, as ``get_settings_state`` reports it missing; the
    values compare in constant time and nothing here names them."""
    return all(hmac.compare_digest(_setting_text(env.get(key)), _setting_text(value))
               for key, value in expected.items())


def _as_read(outcome: str, env: Dict[str, str], expected: Optional[Mapping[str, Any]]) -> str:
    """``outcome``, or ``settings_changed`` when ``env`` lacks a value the caller read (``expected``)."""
    return outcome if expected is None or _carries(env, expected) else "settings_changed"


class CompanionSupervisor:
    """Owns process lifecycle for extension companions in the server process."""

    def __init__(self, data_dir: pathlib.Path):
        self.data_dir = pathlib.Path(data_dir)
        self._lock = threading.RLock()
        self._runtimes: Dict[str, CompanionRuntime] = {}
        self._restart_history: Dict[str, List[float]] = {}
        # The descriptor each companion last started with, kept past a crash so a settings refresh can compare
        # against it; an explicit stop or Panic forgets it, so nothing stopped by a lifecycle control is revived.
        self._last_descriptors: Dict[str, CompanionDescriptor] = {}
        # Companions a settings refresh stopped and has not replaced (its read failed, the spawn was refused, the
        # stop is unconfirmed): the next refresh starts one even when Settings read as before. Any start or
        # lifecycle stop forgets it; an absence a crash left stays the monitor's.
        self._unreplaced: set[str] = set()
        self._panic_requested = False

    def _key(self, skill_name: str, name: str) -> str:
        return f"{skill_name}:{name}"

    def start(self, descriptor: CompanionDescriptor) -> bool:
        """Start a companion process if this is the main server process."""
        if not is_server_process():
            log.debug("Skipping companion start outside server process: %s/%s", descriptor.skill_name, descriptor.name)
            return False
        key = self._key(descriptor.skill_name, descriptor.name)
        with self._lock:
            if self._panic_requested:
                return False
            existing = self._runtimes.get(key)
            if existing is not None and existing.process.poll() is None:
                if existing.retiring:
                    raise RuntimeError(f"companion {key} is retained until its death is observed "
                                       f"({existing.retiring}); replacement refused")
                return True
            if existing is not None:
                self._settle_runtime(key, existing)
            for port in descriptor.ports:
                if not _port_is_available(port):
                    raise RuntimeError(f"port {port} is already in use")
            from ouroboros.process_custody import lifeline_parent_env

            popen_kwargs: Dict[str, Any] = {
                "cwd": str(descriptor.cwd),
                "env": {**_companion_base_env(), **dict(descriptor.env), **lifeline_parent_env()},
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if not IS_WINDOWS:
                popen_kwargs.update(subprocess_new_group_kwargs())
            popen_kwargs = merge_hidden_kwargs(popen_kwargs)
            proc = subprocess.Popen(descriptor.command, **popen_kwargs)  # noqa: S603
            # The existing runtime owner publishes the positive identity before
            # accounting, custody or snapshot writes can block this lock.
            runtime = CompanionRuntime(descriptor, proc, time.monotonic())
            self._runtimes[key] = runtime
            job_handle = None
            if IS_WINDOWS and os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER") != "1":
                job_handle = create_kill_on_close_job()
                if job_handle is None or not assign_pid_to_job(job_handle, proc.pid):
                    if job_handle is not None:
                        close_job(job_handle)  # never attached: an empty Job proves no kill
                    self._retire_failed_start(key, runtime, "Windows Job assignment failed")
                    raise RuntimeError("failed to assign companion process to Windows Job Object")
            runtime.job_handle = job_handle
            if self._panic_requested:
                request_process_tree_kill(proc, job_handle=job_handle)
                return False
            if any(
                str(descriptor.env.get(key) or "").strip()
                for key in MODEL_PROVIDER_CREDENTIAL_KEYS
            ):
                try:
                    system_task = f"extension:{descriptor.skill_name}"
                    record_unmetered_external_dispatch(
                        f"extension:companion:{uuid.uuid4().hex}",
                        drive_root=self.data_dir,
                        provider="external-extension",
                        task_id=system_task,
                        root_task_id=system_task,
                        category="external_skill",
                        source=(
                            f"extension_companion:{descriptor.skill_name}:"
                            f"{descriptor.name}"
                        ),
                        non_task_operation=True,  # a host companion: no task lineage to read
                    )
                except Exception:
                    self._retire_failed_start(key, runtime, "cost disclosure failed")
                    raise
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
            self._start_drainers(runtime)
            try:
                from ouroboros.extension_health import clear_companion_restart_exhausted

                clear_companion_restart_exhausted(
                    self.data_dir, descriptor.skill_name, descriptor.name,
                )
            except Exception:
                log.debug("Failed to clear companion terminal health", exc_info=True)
            self._last_descriptors[key] = descriptor
            self._unreplaced.discard(key)
            self._start_monitor(key, runtime)
            self._write_runtime_snapshot()
            return True

    def refresh_blocker(self, skill_name: str, name: str, expected: Optional[Mapping[str, Any]] = None) -> str:
        """The answer a refresh gives from what this supervisor holds, or "" when it must read settings.

        ``panic``, ``not_started``, ``retained``; with ``expected`` (the caller's read of every compared key) also
        ``unchanged`` while the running companion carries it and ``not_running`` while one a crash left carries
        it (restarting that is the monitor's). A companion a refresh stopped and has not replaced never answers
        here."""
        key = self._key(skill_name, name)
        with self._lock:
            if self._panic_requested:
                return "panic"
            runtime = self._runtimes.get(key)
            if runtime is None and key not in self._last_descriptors:
                return "not_started"
            if runtime is not None and runtime.retiring and runtime.process.poll() is None:
                return "retained"
            if expected is None:
                return ""
            if runtime is not None:
                alive = not runtime.retiring and runtime.process.poll() is None
                return "unchanged" if alive and _carries(runtime.descriptor.env, expected) else ""
            if key in self._unreplaced:
                return ""
            return "not_running" if _carries(self._last_descriptors[key].env, expected) else ""

    def refresh(
        self,
        skill_name: str,
        name: str,
        build_env: Callable[[], Dict[str, str]],
        *,
        compare_keys: Iterable[str],
        still_valid: Callable[[], bool],
        admission: AbstractContextManager[Any] = nullcontext(),
        expected: Optional[Mapping[str, Any]] = None,
        timeout_sec: float = 5.0,
        _attempts: int = 2,
    ) -> str:
        """Respawn one companion whose settings-derived environment changed; otherwise do nothing.

        Only ``compare_keys`` (the manifest's ``env_from_settings``) decide a change: the Host Service token,
        PATH, PYTHONPATH and other service values never restart a companion. A retained or unconfirmed
        runtime is never replaced. The old process's death is observed before the environment is built
        again from the settings as they read then and the new one starts, so a save made while it stopped
        is the one applied. ``still_valid``, that environment, Panic and the start are one step inside
        ``admission`` (the caller's lock its owner-state writers take), never the stop: an Off saved during
        the stop is read there, one saved later finds a process to stop. Nothing here holds the supervisor's
        lock while waiting. ``expected`` holds the values its caller read: when the environment this refresh kept or
        started carries others, Settings moved after that read and the outcome is ``settings_changed`` (the newer
        read is the one applied; read again). A companion this refresh stopped stays owed a start until one
        succeeds, so a later refresh starts it even when Settings read as the stopped one carried them.
        Outcomes: ``unchanged``, ``not_running``, ``not_started``, ``restarted``, ``started``, ``settings_changed``,
        ``retained``, ``stop_unconfirmed``, ``refused``, ``panic``.
        """
        key = self._key(skill_name, name)
        keys = tuple(dict.fromkeys(str(item) for item in compare_keys))
        with self._lock:
            if self._panic_requested:
                return "panic"
            runtime = self._runtimes.get(key)
            baseline = runtime.descriptor if runtime is not None else self._last_descriptors.get(key)
        if baseline is None:
            return "not_started"
        if runtime is not None and runtime.retiring:
            if not self._settle_runtime(key, runtime):
                return "retained"
            runtime = None
        env = build_env()
        running = runtime is not None and runtime.process.poll() is None
        with self._lock:
            unreplaced = key in self._unreplaced
        if [env.get(k) for k in keys] == [baseline.env.get(k) for k in keys] and (running or not unreplaced):
            return _as_read("unchanged" if running else "not_running", env, expected)
        if runtime is not None:
            with self._lock:
                runtime.retiring = runtime.retiring or "settings refresh"
                self._unreplaced.add(key)
            try:
                self._terminate_runtime(runtime, timeout_sec=timeout_sec)
            finally:
                settled = self._settle_runtime(key, runtime)
                self._write_runtime_snapshot()
            if not settled:
                return "stop_unconfirmed"
        with admission:
            if not still_valid():
                return "refused"
            env = build_env()
            with self._lock:
                if self._panic_requested:
                    return "panic"
            if not self.start(replace(baseline, env=dict(env))):
                return "panic" if self._panic_requested else "not_started"
        with self._lock:
            current = self._runtimes.get(key)
        if current is not None and [current.descriptor.env.get(k) for k in keys] != [env.get(k) for k in keys]:
            # A crash restart of the previous environment won the start: replace it once more.
            if _attempts > 1:
                return self.refresh(skill_name, name, build_env, compare_keys=keys, still_valid=still_valid,
                                    admission=admission, expected=expected, timeout_sec=timeout_sec,
                                    _attempts=_attempts - 1)
            return "stop_unconfirmed"
        return _as_read("restarted" if running else "started", env, expected)

    def _start_monitor(self, key: str, runtime: CompanionRuntime) -> None:
        threading.Thread(
            target=self._monitor_runtime,
            args=(key, runtime),
            daemon=True,
            name=f"companion-monitor-{runtime.descriptor.skill_name}-{runtime.descriptor.name}",
        ).start()

    def _retire_failed_start(self, key: str, runtime: CompanionRuntime, reason: str) -> None:
        """Kill a published runtime whose start failed; keep it owned until death.

        A process may outlive the kill request. Its exact runtime stays visible to
        stop, Panic and the launcher snapshot, refuses replacement, and the monitor
        settles it (Job included) once death is observed.
        """
        runtime.retiring = reason
        try:
            kill_process_tree(runtime.process)
        except Exception:
            log.exception("Companion kill after failed start failed: %s", key)
        if not self._settle_runtime(key, runtime):
            try:
                self._start_monitor(key, runtime)
            except Exception:
                log.exception("Companion %s is retained without a death monitor", key)
        self._write_runtime_snapshot()

    def _release_job(self, runtime: CompanionRuntime) -> None:
        """Close this runtime's Job exactly once; later readers observe None."""
        with self._lock:
            job, runtime.job_handle = runtime.job_handle, None
        if job is not None and (problem := close_job(job)):
            log.warning("Companion %s/%s Job close: %s", runtime.descriptor.skill_name,
                        runtime.descriptor.name, problem)

    def _settle_runtime(self, key: str, runtime: CompanionRuntime) -> bool:
        """After observed death, release the exact runtime's Job, then its entry."""
        if runtime.process.poll() is None:
            return False
        self._release_job(runtime)
        with self._lock:
            if self._runtimes.get(key) is runtime:
                self._runtimes.pop(key, None)
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
        restart_exhausted = False
        with self._lock:
            current = self._runtimes.get(key)
            if (current is runtime and not runtime.retiring and not self._panic_requested
                    and descriptor.restart_policy == "on_failure" and returncode != 0):
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
                    restart_exhausted = True
                    log.warning(
                        "companion %s/%s exceeded restart limit",
                        descriptor.skill_name,
                        descriptor.name,
                    )
            elif current is runtime:
                self._runtimes.pop(key, None)
        if restart_exhausted:
            try:
                from ouroboros.extension_health import record_companion_restart_exhausted

                record_companion_restart_exhausted(
                    self.data_dir,
                    descriptor.skill_name,
                    descriptor.name,
                    returncode=returncode,
                )
            except Exception:
                log.debug("Failed to persist companion terminal health", exc_info=True)
            finally:
                # Publish durable failure before the live snapshot loses the
                # exited process. Readers that observe an empty snapshot can
                # therefore already recover the terminal reason from health.
                with self._lock:
                    if self._runtimes.get(key) is runtime:
                        self._runtimes.pop(key, None)
        # Death is observed: the dead runtime's Job closes before any replacement.
        self._release_job(runtime)
        self._write_runtime_snapshot()
        if should_restart:
            time.sleep(0.5)
            try:
                with self._lock:
                    # A refresh or stop may have superseded this monitor during its backoff.
                    # Check and start together; an older crash must not replace a newer registration.
                    if self._last_descriptors.get(key) is descriptor:
                        self.start(descriptor)
            except Exception:
                log.warning("failed to restart companion %s/%s", descriptor.skill_name, descriptor.name, exc_info=True)

    def stop(self, skill_name: str, name: str, timeout_sec: float = 5.0) -> None:
        """Terminate the exact runtime; its owner is removed only after observed death."""
        key = self._key(skill_name, name)
        with self._lock:
            self._last_descriptors.pop(key, None)
            self._unreplaced.discard(key)
            runtime = self._runtimes.get(key)
            if runtime is None:
                return
            runtime.retiring = runtime.retiring or "stop requested"
        try:
            self._terminate_runtime(runtime, timeout_sec=timeout_sec)
        finally:
            if not self._settle_runtime(key, runtime):
                log.warning("Companion %s is retained: its death was not observed after stop", key)
            self._write_runtime_snapshot()

    def stop_skill(self, skill_name: str, timeout_sec: float = 5.0) -> None:
        with self._lock:
            for key, descriptor in list(self._last_descriptors.items()):
                if descriptor.skill_name == skill_name:
                    self._last_descriptors.pop(key, None)
                    self._unreplaced.discard(key)
        for runtime in list(self.snapshot().values()):
            if runtime.get("skill_name") == skill_name:
                self.stop(skill_name, str(runtime.get("name") or ""), timeout_sec=timeout_sec)

    def stop_all(self, timeout_sec: float = 5.0) -> None:
        with self._lock:
            self._last_descriptors.clear()
            self._unreplaced.clear()
        for runtime in list(self.snapshot().values()):
            self.stop(str(runtime["skill_name"]), str(runtime["name"]), timeout_sec=timeout_sec)

    def panic_kill_all(self, *, request_only: bool = False) -> list[dict[str, Any]] | None:
        self._panic_requested = True
        self._last_descriptors.clear()
        self._unreplaced.clear()
        if request_only:
            return [request_process_tree_kill(runtime.process, job_handle=runtime.job_handle)
                    for runtime in self._runtimes.copy().values()]
        with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
        for runtime in runtimes:
            try:
                with self._lock:  # a settling monitor cannot close the Job mid-use
                    if runtime.job_handle is not None:
                        terminate_job(runtime.job_handle)
                kill_process_tree(runtime.process)
                for port in runtime.descriptor.ports:
                    kill_process_on_port(port)
            finally:
                self._release_job(runtime)
        self._write_runtime_snapshot()

    def _terminate_runtime(self, runtime: CompanionRuntime, *, timeout_sec: float) -> None:
        """Request termination; the Job stays owned until stop observes death."""
        proc = runtime.process
        if proc.poll() is None:
            with self._lock:  # a settling monitor cannot close the Job mid-use
                job = runtime.job_handle
                if job is not None:
                    terminate_job(job)
            if job is None:
                terminate_process_tree(proc)
            if not _await_exit(proc, max(0.1, timeout_sec)):
                kill_process_tree(proc)
                _await_exit(proc, 1.0)  # observe the forced death instead of assuming it
        for port in runtime.descriptor.ports:
            kill_process_on_port(port)

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
            # Non-empty: a retained failed start or unconfirmed stop, not a running companion.
            "retiring": rt.retiring,
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


def panic_kill_all(*, request_only: bool = False) -> list[dict[str, Any]] | None:
    if _GLOBAL_SUPERVISOR is not None:
        return _GLOBAL_SUPERVISOR.panic_kill_all(request_only=request_only)
    return [] if request_only else None


__all__ = [
    "CompanionDescriptor", "CompanionSupervisor", "init_global_supervisor",
    "init_server_process_pid", "is_server_process", "panic_kill_all", "snapshot_processes",
]


def companion_spawn_env(
    spec: Dict[str, Any], token: str, *, env_allow: Any, granted_upper: Any,
    skill: str, skill_dir: Optional[pathlib.Path], state_dir: pathlib.Path,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """The env one staged companion is spawned with (fix-round-6).

    Built only inside ``PluginAPIImpl._publish_registrations``'s post-swap attach, after
    the generation fence admitted the publication: the settings-derived
    values (``_scrub_env`` -> ``load_settings``, a lock-free read that
    persists nothing), the manifest env overlay, the Host
    Service bridge URL/token and the isolated-dep PYTHONPATH are all
    resolved HERE, so the pre-fence descriptor build stays purely
    computational.
    """
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
    from ouroboros.deployment_policy import child_policy_env, fixed_settings
    from ouroboros.extension_isolated_deps import _isolated_python_site_dirs
    from ouroboros.gateway.host_service import DEFAULT_HOST_SERVICE_HOST, host_service_port
    from ouroboros.node_runtime import prepend_skill_node_emergency_path, skill_manifest_owns_path
    from ouroboros.process_custody import PARENT_PID_ENV
    from ouroboros.tools.skill_exec import _scrub_env

    # Case-aware merge (delta finding D2-8): a manifest "Path" must REPLACE
    # the allowlisted "PATH" on Windows, never sit next to it — duplicate
    # case-variant env keys make CreateProcess-era spawns fail or pick an
    # undefined winner. Same contract as the executor-local service lane.
    from ouroboros.workspace_executor import overlay_env

    # Under a deployment policy, a manifest's env never stands in for a value it fixes, or for its file and pin.
    reserved = set(FORBIDDEN_SKILL_SETTINGS) | {"HOST_SERVICE_TOKEN", "HOST_SERVICE_URL", PARENT_PID_ENV,
                                                *child_policy_env(), *fixed_settings()}
    env = overlay_env(
        _scrub_env(
            list(env_allow), state_dir, skill,
            granted_keys=list(granted_upper), settings=settings,
        ),
        {
            str(key): str(value)
            for key, value in (spec.get("env") or {}).items()
            if str(key).upper() not in reserved
        },
    )
    env["HOST_SERVICE_URL"] = f"http://{DEFAULT_HOST_SERVICE_HOST}:{host_service_port()}"
    env["HOST_SERVICE_TOKEN"] = token
    site_dirs = [] if skill_dir is None else [
        str(path) for path in _isolated_python_site_dirs(skill_dir)
    ]
    if site_dirs:
        inherited = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join([*site_dirs, inherited] if inherited else site_dirs)
    if str(spec.get("runtime") or "").strip() in {"node", "npm"} and not skill_manifest_owns_path(spec):
        # T14 emergency PATH prepend, the other half of the argv rewrite in
        # register_companion_process: descriptor env keys win over the
        # supervisor's `_companion_base_env` merge, so the prepend reaches
        # the child (and the PATH it would otherwise inherit) and survives
        # supervisor restarts.
        prepend_skill_node_emergency_path(env, fallback_path=os.environ.get("PATH", ""))
    return env
