"""Supervision for long-lived extension companion processes."""

from __future__ import annotations

import hmac
import logging
import os
import pathlib
import socket
import subprocess
import sys
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


def _companion_base_env(descriptor_env=None) -> Dict[str, str]:
    from ouroboros.node_runtime import forward_playwright_node
    from ouroboros.process_scope import installation_scope_entries

    env = {**{key: value for key, value in os.environ.items() if key.upper() in _COMPANION_BASE_ENV_KEYS},
           **installation_scope_entries()}
    from ouroboros.workspace_executor import overlay_env

    env = overlay_env(env, descriptor_env)
    forward_playwright_node(env)  # refresh a saved automatic selection after the manifest's case-aware overlay
    return env


def _interpreter_command(command: List[str], *, windows: bool = IS_WINDOWS, executable: str = sys.executable,
                         base: str = getattr(sys, "_base_executable", "") or sys.executable,
                         ) -> tuple[List[str], Dict[str, str]]:
    """The companion as the started process itself. A Windows venv's ``python.exe`` is a redirector that runs
    the base interpreter as its CHILD: the supervisor's PID, Job and kill would hold the redirector, not the
    companion. As ``multiprocessing`` does (bpo-35797), start the base interpreter and name the venv in
    ``__PYVENV_LAUNCHER__``: the same venv interpreter, one process."""
    if (windows and command and os.path.normcase(command[0]) == os.path.normcase(executable)
            and os.path.normcase(base) != os.path.normcase(executable)):
        return [base, *command[1:]], {"__PYVENV_LAUNCHER__": executable}
    return list(command), {}


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
    skill_dir: Optional[str] = None  # installed payload, independent of the staged cwd
    exited_cleanly: bool = field(default=False, init=False, repr=False, compare=False)  # this start only
    started_utc: str = field(default="", init=False, repr=False, compare=False)


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
    # Non-empty once a failed start, or a stop or refresh while it still runs, retires this exact runtime:
    # it is never restarted or replaced, and stays owned until its death is observed.
    retiring: str = ""
    # Set by its monitor once that death is observed; its entry may stay until the failure it left is recorded.
    exited: bool = False
    stop_unconfirmed: bool = False  # a completed stop attempt already found this process still alive

    @property
    def started_utc(self) -> str:
        return self.descriptor.started_utc

    @property
    def death_known(self) -> bool:
        """Its monitor observed its death, or an earlier poll (a snapshot's, before any monitor ran) already cached
        the exit code: read, never polled. Such a runtime counts as missing, not as running or awaiting exit; a
        stand-in without ``returncode`` answers through its monitor alone."""
        return self.exited or getattr(self.process, "returncode", None) is not None


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
        # Each start owns a descriptor copy. Only the current one may open a terminal failure.
        self._last_descriptors: Dict[str, CompanionDescriptor] = {}
        # Membership is a settings refresh's replacement claim; its value is the observed start time of
        # a running predecessor, or empty if refresh only found an absence.
        # A successful start or lifecycle stop clears the claim and its cause together.
        self._unreplaced: dict[str, str] = {}
        self._panic_requested = False

    def _key(self, skill_name: str, name: str) -> str:
        return f"{skill_name}:{name}"

    def start(self, descriptor: CompanionDescriptor, only_if: Optional[Callable[[], bool]] = None) -> bool:
        """Start a companion process if this is the main server process. ``only_if``, asked under this supervisor's
        lock together with the start, may decline it: False then, as for Panic."""
        if not is_server_process():
            log.debug("Skipping companion start outside server process: %s/%s", descriptor.skill_name, descriptor.name)
            return False
        key = self._key(descriptor.skill_name, descriptor.name)
        with self._lock:
            if self._panic_requested or (only_if is not None and not only_if()):
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

            argv, launcher_env = _interpreter_command(list(descriptor.command))
            popen_kwargs: Dict[str, Any] = {
                "cwd": str(descriptor.cwd),
                "env": {**_companion_base_env(descriptor.env), **lifeline_parent_env(), **launcher_env},
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if not IS_WINDOWS:
                popen_kwargs.update(subprocess_new_group_kwargs())
            popen_kwargs = merge_hidden_kwargs(popen_kwargs)
            proc = subprocess.Popen(argv, **popen_kwargs)  # noqa: S603
            # The existing runtime owner publishes the positive identity before
            # accounting, custody or snapshot writes can block this lock.
            runtime = CompanionRuntime(replace(descriptor), proc, time.monotonic())  # its own copy names this start
            runtime.descriptor.started_utc = utc_now_iso()
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
            # Now the companion's last start, named before its close: a late monitor of any earlier runtime (one a
            # failed start settled included) records nothing over it or over its later death.
            prior = self._last_descriptors.get(key)
            replacing = key in self._unreplaced or (prior is not None and not prior.exited_cleanly)
            self._last_descriptors[key] = runtime.descriptor
            self._unreplaced.pop(key, None)
            self._write_runtime_snapshot()
        # Close the preceding failure as of this successful start, before its own monitor can publish a new one.
        # Waiting for the health writer must not hold the supervisor's lock.
        if not self._close_failure(runtime, "start") and replacing:
            self._notify_health_changed()
        self._start_monitor(key, runtime)
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
        read is the one applied; read again). Once refresh takes over replacement, a later refresh can finish it
        even when Settings read as the stopped companion carried them; a monitor cannot use that old environment.
        Outcomes: ``unchanged``, ``not_running``, ``not_started``, ``restarted``, ``started``, ``settings_changed``,
        ``retained``, ``stop_unconfirmed``, ``refused``, ``panic``.
        """
        key = self._key(skill_name, name)
        keys = tuple(dict.fromkeys(str(item) for item in compare_keys))
        with self._lock:
            if self._panic_requested:
                return "panic"
            runtime = self._runtimes.get(key)
            baseline = self._last_descriptors.get(key)
            retrying = key in self._unreplaced
        if runtime is not None and runtime.retiring:
            if not self._settle_runtime(key, runtime):
                return "retained"
            runtime = None
        if baseline is None:
            return "not_started"
        # An absent runtime with a claim already needs replacement even on the same settings. Read them only
        # inside final admission so a failed read cannot bypass the replacement failure path on retry.
        env = None if runtime is None and retrying else build_env()
        with self._lock:
            if self._panic_requested:
                return "panic"
            current = self._runtimes.get(key)
            owns_baseline = self._last_descriptors.get(key) is baseline and (current is None or current is runtime)
            if owns_baseline:
                if current is not None and current.retiring:
                    return "retained"
                running = current is not None and current.process.poll() is None
                if (env is not None and (running or key not in self._unreplaced)
                        and [env.get(k) for k in keys] == [baseline.env.get(k) for k in keys]
                ):
                    return _as_read("unchanged" if running else "not_running", env, expected)
                if running:  # only an intended stop, never a death the refresh merely found
                    current.retiring = "settings refresh"
                replaced_start = self._unreplaced.setdefault(key, current.started_utc if running else "")
        if not owns_baseline:
            # A different start (possibly a crash monitor) or a Stop won while settings were read.
            return (self.refresh(skill_name, name, build_env, compare_keys=keys, still_valid=still_valid,
                                 admission=admission, expected=expected, timeout_sec=timeout_sec,
                                 _attempts=_attempts - 1) if _attempts > 1 else "not_started")
        if current is not None:
            try:
                self._terminate_runtime(current, timeout_sec=timeout_sec)
            finally:
                settled = self._finish_stop(key, current)
            if not settled:
                return "stop_unconfirmed"

        def owns_replacement() -> bool:
            return (not self._panic_requested and key in self._unreplaced
                    and self._last_descriptors.get(key) is baseline)

        replacing = False
        try:
            with admission:
                if not still_valid():
                    return "refused"
                replacing = True
                env = build_env()
                started = self.start(replace(baseline, env=dict(env)), only_if=owns_replacement)
                replacing = False
                if not started and self._panic_requested:
                    return "panic"
        except Exception:
            # The start this replacement stands in for: one it stopped, or one whose clean exit it found (no
            # monitor tells of that). A death it merely found stays its monitor's to tell.
            replaced = replaced_start or (baseline.started_utc if baseline.exited_cleanly else "")
            if replacing and replaced:
                # Journal this failed replacement even if a later start won. The health writer rechecks ownership.
                from ouroboros.extension_health import record_companion_refresh_failure

                record_companion_refresh_failure(
                    self.data_dir, skill_name, name, still_current=owns_replacement, replaced_start=replaced,
                    retained=lambda: self.retained_companions(skill_name, baseline.skill_dir),
                )
            raise
        with self._lock:
            current = self._runtimes.get(key)
        if not started or (current is not None and [current.descriptor.env.get(k) for k in keys] !=
                           [env.get(k) for k in keys]):
            # A reconcile or another refresh can still win admission. Re-read what that start carries.
            if _attempts > 1:
                return self.refresh(skill_name, name, build_env, compare_keys=keys, still_valid=still_valid,
                                    admission=admission, expected=expected, timeout_sec=timeout_sec,
                                    _attempts=_attempts - 1)
            return "stop_unconfirmed" if started else "not_started"
        return _as_read("restarted" if running else "started", env, expected)

    def close_recovered_failures(self, skill_name: str, names: Iterable[str], by: str = "reconcile") -> None:
        """Close, as of its start, the recorded terminal failure of each named companion whose current runtime has
        no known death: that start's own close may have been lost (a health write that timed out). Without a runtime
        nothing closes."""
        for name in names:
            with self._lock:
                runtime = self._runtimes.get(self._key(skill_name, name))
            if runtime is not None:
                self._close_failure(runtime, by)

    def _close_failure(self, runtime: CompanionRuntime, by: str) -> bool:
        """A successful start closes the preceding failure even if the new process already exited cleanly. Its
        monitor has not started yet. A later reconcile can repair a missed close only with a still-live runtime.
        Both recheck identity inside the health writer, without polling or holding the supervisor's lock."""
        descriptor = runtime.descriptor
        try:
            from ouroboros.extension_health import clear_companion_restart_exhausted

            return clear_companion_restart_exhausted(
                self.data_dir, descriptor.skill_name, descriptor.name, by=by, closed_at=runtime.started_utc,
                retained=lambda: self.retained_companions(descriptor.skill_name, descriptor.skill_dir),
                still_current=lambda: (
                    not self._panic_requested and not runtime.retiring and (by == "start" or not runtime.death_known)
                    and self._last_descriptors.get(self._key(descriptor.skill_name, descriptor.name)) is descriptor
                ),
            )
        except Exception:
            log.debug("Failed to clear companion terminal health", exc_info=True)
            return False

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
        if not self._finish_stop(key, runtime):
            try:
                self._start_monitor(key, runtime)
            except Exception:
                log.exception("Companion %s is retained without a death monitor", key)

    def _finish_stop(self, key: str, runtime: CompanionRuntime) -> bool:
        """Publish the stop outcome. A page with no pending exits needs a hint to discover its first."""
        settled = self._settle_runtime(key, runtime)
        self._write_runtime_snapshot()
        if not settled and is_server_process():
            with self._lock:
                if runtime.stop_unconfirmed or runtime.death_known or self._runtimes.get(key) is not runtime:
                    return settled
                runtime.stop_unconfirmed = True
            self._notify_health_changed()
        return settled

    @staticmethod
    def _notify_health_changed() -> None:
        from ouroboros.extension_health import _notify_health_changed

        _notify_health_changed()

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
        # Intent at observation determines whether this was a crash. A later stop can withdraw the restart,
        # not turn an already observed crash into a clean or intentional exit.
        unexpected = returncode != 0 and not runtime.retiring and not self._panic_requested
        observed_at = utc_now_iso()
        descriptor = runtime.descriptor
        restart_times = None
        terminal = False
        exhausted = False
        with self._lock:
            runtime.exited = True
            current = self._runtimes.get(key)
            descriptor.exited_cleanly = returncode == 0 and not runtime.retiring
            if unexpected and not self._panic_requested and self._last_descriptors.get(key) is descriptor:
                if descriptor.restart_policy == "on_failure":
                    now = time.monotonic()
                    history = [ts for ts in self._restart_history.get(key, [])
                               if now - ts <= descriptor.restart_window_sec]
                    exhausted = len(history) >= descriptor.max_restarts
                    if not exhausted and key not in self._unreplaced:
                        restart_times = [*history, now]
                terminal = restart_times is None
            if current is runtime and not terminal:
                self._runtimes.pop(key)
        if terminal:
            if exhausted:
                log.warning("companion %s exceeded restart limit", key)
            self._record_failure(key, runtime, returncode, observed_at, exhausted=exhausted)
        # Death is observed: the dead runtime's Job closes before any replacement.
        self._release_job(runtime)
        self._write_runtime_snapshot()
        if runtime.stop_unconfirmed and key in self._unreplaced and self._last_descriptors.get(key) is descriptor:
            self._notify_health_changed()  # pending exit became pending replacement
        if restart_times is not None:
            def admit_restart() -> bool:
                # The window was measured at the exit. Charge it under start's lock after backoff.
                if (self._panic_requested or key in self._unreplaced
                        or self._last_descriptors.get(key) is not descriptor):
                    return False
                self._restart_history[key] = restart_times
                return True

            time.sleep(0.5)
            attempted_at = utc_now_iso()
            try:
                started = self.start(descriptor, only_if=admit_restart)
            except Exception:
                log.warning("failed to restart companion %s/%s", descriptor.skill_name, descriptor.name, exc_info=True)
                self._record_failure(key, runtime, returncode, observed_at, restart_attempted_at=attempted_at)
                return
            if not started:
                self._record_failure(key, runtime, returncode, observed_at)
                return
        if unexpected and not terminal:
            try:
                from ouroboros.extension_health import record_companion_exit

                record_companion_exit(self.data_dir, descriptor.skill_name, descriptor.name, returncode=returncode,
                                      observed_at=observed_at)
            except Exception:
                log.debug("Failed to journal a companion exit", exc_info=True)

    def _record_failure(self, key: str, runtime: CompanionRuntime, returncode: int, observed_at: str,
                        *, exhausted: bool = False, restart_attempted_at: str = "") -> None:
        """Journal this death; only an unreplaced, still-registered start can leave a current failure."""
        descriptor = runtime.descriptor
        try:
            from ouroboros.extension_health import record_companion_exit, record_companion_restart_exhausted

            facts = dict(
                returncode=returncode,
                observed_at=observed_at,
                started_at=runtime.started_utc,  # this start closed any earlier failure, recorded or not
                retained=lambda: self.retained_companions(descriptor.skill_name, descriptor.skill_dir),
                # Refresh can take restart ownership without changing the fact of the death. Stop or a successful
                # replacement, in contrast, removes this start's authority to change current health.
                still_current=lambda: (not self._panic_requested and not runtime.retiring
                                       and self._last_descriptors.get(key) is descriptor),
            )
            args = (self.data_dir, descriptor.skill_name, descriptor.name)
            if exhausted or restart_attempted_at:
                record_companion_restart_exhausted(*args, **facts, restart_failed=bool(restart_attempted_at),
                                                   restart_attempted_at=restart_attempted_at)
            else:
                record_companion_exit(*args, **facts)
        except Exception:
            log.debug("Failed to persist companion terminal health", exc_info=True)
        finally:
            # Publish before removing the slot if the monitor still owns it. A refresh may already have settled it.
            with self._lock:
                if self._runtimes.get(key) is runtime:
                    self._runtimes.pop(key, None)

    def stop(self, skill_name: str, name: str, timeout_sec: float = 5.0) -> None:
        """Terminate the exact runtime; its owner is removed only after observed death."""
        key = self._key(skill_name, name)
        with self._lock:
            self._last_descriptors.pop(key, None)
            self._unreplaced.pop(key, None)
            runtime = self._runtimes.get(key)
            if runtime is None:
                return
            if runtime.process.poll() is None:  # a stop intent; a death found here stays its monitor's to tell
                runtime.retiring = runtime.retiring or "stop requested"
        try:
            self._terminate_runtime(runtime, timeout_sec=timeout_sec)
        finally:
            if not self._finish_stop(key, runtime):
                log.warning("Companion %s is retained: its death was not observed after stop", key)

    def stop_skill(self, skill_name: str, timeout_sec: float = 5.0) -> None:
        with self._lock:
            for key, descriptor in list(self._last_descriptors.items()):
                if descriptor.skill_name == skill_name:
                    self._last_descriptors.pop(key, None)
                    self._unreplaced.pop(key, None)
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

    def companion_ownership(self, skill_name: str, names: Iterable[str]) -> tuple[List[str], Dict[str, str]]:
        """Missing registrations and retained reasons, from this skill's entries without polling processes: one
        whose death is known is missing, even while its entry waits for its monitor or that failure's record."""
        missing, retained = [], {}
        with self._lock:
            for name in names:
                runtime = self._runtimes.get(self._key(skill_name, name))
                if runtime is None or runtime.death_known:
                    missing.append(name)
                elif runtime.retiring:
                    retained[name] = runtime.retiring
        return missing, retained

    def companions_allow_known_good(self, skill_name: str, names: Iterable[str]) -> bool:
        """Already-observed starts only: a clean exit is valid start history, an unknown absence is not.

        Like retained custody, this is a passive sample. Never poll or wait for the supervisor lock
        while the health writer holds its lock. An incompletely published replacement is not a good start.
        """
        for name in names:
            key = self._key(skill_name, name)
            started = self._last_descriptors.get(key)
            runtime = self._runtimes.get(key)
            if started is None or key in self._unreplaced:
                return False
            if runtime is None:
                if not started.exited_cleanly:
                    return False
            elif (runtime.descriptor is not started or runtime.retiring
                  or (runtime.death_known and not started.exited_cleanly
                      and getattr(runtime.process, "returncode", None) != 0)):
                return False
            if self._last_descriptors.get(key) is not started:
                return False
        return not self._panic_requested

    def retained_companions(self, skill_name: str, skill_dir: Optional[str]) -> Dict[str, str]:
        """Owned registrations awaiting exit for these installed files, even after their bundle was disposed."""
        # A passive observation, not a spawn decision: an unrelated slow start must not block listing.
        return {
            rt.descriptor.name: rt.retiring for rt in self._runtimes.copy().values()
            if rt.retiring and not rt.death_known
            and rt.descriptor.skill_name == skill_name and rt.descriptor.skill_dir == skill_dir
        }

    def observed_companion_starts(
        self, skill_name: str, skill_dir: str, names: Iterable[str],
    ) -> tuple[dict[str, str], bool, bool]:
        """Last successful starts, unsettled replacements and unconfirmed readiness for this publication.

        Borrow only already-observed facts, without waiting for the supervisor lock or querying a process. No start
        of these files (published before its spawn, a declined start), or one a Stop forgot or a newer start replaced
        during this read, confirms nothing. Only the last start's own runtime is its process: ``start`` names a new
        one last after its custody write, so until then a replacement stays pending. Panic revokes every start.
        """
        starts, pending, unconfirmed = {}, False, False
        for name in names:
            key = self._key(skill_name, name)
            started = self._last_descriptors.get(key)
            runtime = self._runtimes.get(key)
            replacing = key in self._unreplaced
            if self._panic_requested:
                continue
            if started is None or started.skill_dir != skill_dir or self._last_descriptors.get(key) is not started:
                unconfirmed |= not self._panic_requested  # Panic's clear is no observation
                continue
            own = runtime is not None and runtime.descriptor is started
            awaiting_exit = own and not runtime.death_known
            if replacing:
                starts[name] = started.started_utc
                pending |= not awaiting_exit
            elif awaiting_exit:
                starts[name] = started.started_utc
            else:
                # The start stays a fact once its process ended: a clean exit revives no failure that start closed.
                starts[name] = started.started_utc
                unconfirmed |= not (started.exited_cleanly or (
                    own and not runtime.retiring and getattr(runtime.process, "returncode", None) == 0))
        return starts, pending, unconfirmed

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
    manifest_env = {str(key): str(value) for key, value in (spec.get("env") or {}).items()
                    if str(key).upper() not in reserved}
    env = overlay_env(_scrub_env(list(env_allow), state_dir, skill, granted_keys=list(granted_upper),
                                 settings=settings), manifest_env)
    from ouroboros.node_runtime import keep_explicit_driver_node

    keep_explicit_driver_node(env, manifest_env)  # the manifest's driver Node is its own, even one equal to ours
    env["HOST_SERVICE_URL"] = f"http://{DEFAULT_HOST_SERVICE_HOST}:{host_service_port()}"
    env["HOST_SERVICE_TOKEN"] = token
    site_dirs = [] if skill_dir is None else [
        str(path) for path in _isolated_python_site_dirs(skill_dir)
    ]
    if site_dirs:
        inherited = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join([*site_dirs, inherited] if inherited else site_dirs)
    if str(spec.get("runtime") or "").strip() in {"node", "npm"} and not skill_manifest_owns_path(spec):
        # T14 PATH prepend (the pin's Node first, else an emergency only), the
        # other half of the argv rewrite in register_companion_process: an npm
        # launcher's shebang finds the reviewed Node. Descriptor env keys win
        # over the supervisor's `_companion_base_env` merge, so the prepend
        # reaches the child and survives supervisor restarts.
        prepend_skill_node_emergency_path(env, fallback_path=os.environ.get("PATH", ""))
    return env
