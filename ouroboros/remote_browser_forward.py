"""Hardened task-owned OpenSSH local forwards for remote loopback services."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import pathlib
import re
import secrets
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ouroboros.platform_layer import kill_process_group_id, kill_process_tree
from ouroboros.process_custody import spawn_supervised

_SAFE_ALIAS = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:@-]{0,254}$")
_BIND_RACE_MARKERS = (
    "address already in use",
    "cannot listen to port",
    "could not request local forwarding",
)
_DEFAULT_CONTROL_PORTS = frozenset({8765, 8766, 8767})


class BrowserForwardError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrowserForward:
    forward_id: str
    connection_id: str
    task_id: str
    remote_port: int
    local_port: int
    url: str
    origin: str
    task_token: str
    config_sha256: str


@dataclass
class _LiveForward:
    record: BrowserForward
    process: subprocess.Popen[Any]


@dataclass
class _PendingForward:
    connection_id: str
    task_id: str
    process: subprocess.Popen[Any]


class SSHBrowserForwardManager:
    """Own non-multiplexed `ssh -N -L` children below one broker generation."""

    def __init__(
        self,
        drive_root: pathlib.Path,
        *,
        ssh_binary: str = "ssh",
        config_runner: Callable[[list[str], Mapping[str, str]], bytes] | None = None,
        process_spawner: Callable[..., subprocess.Popen[Any]] = spawn_supervised,
        connector: Callable[..., Any] = socket.create_connection,
        control_ports: set[int] | frozenset[int] | None = None,
        max_bind_attempts: int = 3,
    ) -> None:
        self.drive_root = pathlib.Path(drive_root).resolve(strict=False)
        self.ssh_binary = str(ssh_binary or "ssh")
        self._config_runner = config_runner or self._run_config
        self._process_spawner = process_spawner
        self._connector = connector
        self._control_ports = frozenset(control_ports or _DEFAULT_CONTROL_PORTS)
        self._max_bind_attempts = max(1, min(int(max_bind_attempts), 5))
        self._lock = threading.RLock()
        self._live: dict[str, _LiveForward] = {}
        self._pending: dict[int, _PendingForward] = {}
        # PANIC CUSTODY: panic reads it as a snapshot without taking the ordinary
        # manager lock, so it also covers a child between spawn and publication in
        # ``_pending``. Keyed by ``id()`` and pruned by ``_terminate``/
        # ``_terminate_many`` — the same functions that end the process — because it
        # was append-only with no removal anywhere, so every forward ever opened kept
        # a dead ``Popen`` and its ``stderr`` pipe object reachable for the whole life
        # of the broker. This is the same defect as the broker's own
        # ``_panic_transports`` and it is fixed the same way; custody must end when
        # custody ends.
        self._panic_processes: dict[int, subprocess.Popen[Any]] = {}
        # NOT the same shape: these are TOMBSTONES, written on the normal exit path
        # precisely so they outlive the thing they name — ``open()`` reads them to
        # refuse a forward for a task or connection whose owner is already gone.
        # Bounded by the distinct task/connection ids of one server generation, and
        # deliberately not pruned: evicting one would re-permit a forward for a dead
        # owner, which is the guard they exist to be.
        self._closed_tasks: set[str] = set()
        self._closed_connections: set[str] = set()
        self._closed = False

    def open(
        self,
        connection: Mapping[str, Any],
        *,
        remote_port: int,
        task_id: str,
    ) -> BrowserForward:
        alias, remote_port = _validated_connection_fields(
            connection.get("ssh_alias"),
            remote_port,
        )
        connection_id = str(connection.get("id") or "").strip()
        if not connection_id or not task_id:
            raise BrowserForwardError("connection_id and task_id are required")
        with self._lock:
            if self._closed or task_id in self._closed_tasks or connection_id in self._closed_connections:
                raise BrowserForwardError("browser forward owner is already closed")
        child_env = _ssh_child_env()
        last_error = "local forward did not become ready"
        for _attempt in range(self._max_bind_attempts):
            local_port, probe = self._reserve_loopback_port()
            try:
                command = self._command(alias, local_port, remote_port)
                initial_digest = self._validated_config_digest(command, child_env)
                fresh_digest = self._validated_config_digest(command, child_env)
                if fresh_digest != initial_digest:
                    raise BrowserForwardError("SSH effective config changed while the forward was prepared")
                probe.close()
                process = self._process_spawner(
                    command,
                    drive_root=self.drive_root,
                    purpose=f"remote_browser_forward:{connection_id}:{task_id}",
                    scope="task",
                    owner_task_id=task_id,
                    new_process_group=True,
                    required_custody=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=child_env,
                )
                self._panic_processes[id(process)] = process
                if self._closed:
                    if (
                        process.poll() is None
                        and not kill_process_group_id(process.pid)
                    ):
                        try:
                            process.kill()
                        except Exception:
                            pass
                    raise BrowserForwardError(
                        "browser forward owner closed during startup"
                    )
                with self._lock:
                    if self._closed or task_id in self._closed_tasks or connection_id in self._closed_connections:
                        _terminate(process)
                        raise BrowserForwardError("browser forward owner closed during startup")
                    self._pending[id(process)] = _PendingForward(
                        connection_id,
                        task_id,
                        process,
                    )
                ready, error = self._await_ready(process, local_port)
                if not ready:
                    with self._lock:
                        self._pending.pop(id(process), None)
                    _terminate(process)
                    last_error = error
                    if any(marker in str(error or "").casefold() for marker in _BIND_RACE_MARKERS):
                        continue
                    raise BrowserForwardError(error)
                origin = f"http://127.0.0.1:{local_port}"
                record = BrowserForward(
                    forward_id=secrets.token_urlsafe(24),
                    connection_id=connection_id,
                    task_id=task_id,
                    remote_port=remote_port,
                    local_port=local_port,
                    url=origin + "/",
                    origin=origin,
                    task_token=secrets.token_urlsafe(32),
                    config_sha256=fresh_digest,
                )
                with self._lock:
                    self._pending.pop(id(process), None)
                    if self._closed or task_id in self._closed_tasks or connection_id in self._closed_connections:
                        _terminate(process)
                        raise BrowserForwardError("browser forward owner closed during startup")
                    self._live[record.forward_id] = _LiveForward(record, process)
                return record
            finally:
                probe.close()
        raise BrowserForwardError(f"SSH local port lost a bounded bind race: {last_error}")

    def close(self, forward_id: str) -> bool:
        with self._lock:
            live = self._live.pop(str(forward_id), None)
        if live is None:
            return False
        _terminate(live.process, self._panic_processes)
        return True

    def close_task(self, task_id: str) -> int:
        task_id = str(task_id)
        with self._lock:
            self._closed_tasks.add(task_id)
            live = [self._live.pop(key) for key, item in list(self._live.items()) if item.record.task_id == task_id]
            pending = [
                self._pending.pop(key).process for key, item in list(self._pending.items()) if item.task_id == task_id
            ]
        return _terminate_many(live, pending, self._panic_processes)

    def close_connection(self, connection_id: str) -> int:
        connection_id = str(connection_id)
        with self._lock:
            self._closed_connections.add(connection_id)
            live = [
                self._live.pop(key)
                for key, item in list(self._live.items())
                if item.record.connection_id == connection_id
            ]
            pending = [
                self._pending.pop(key).process
                for key, item in list(self._pending.items())
                if item.connection_id == connection_id
            ]
        return _terminate_many(live, pending, self._panic_processes)

    def close_all(self) -> int:
        with self._lock:
            self._closed = True
            live = list(self._live.values())
            pending = [item.process for item in self._pending.values()]
            self._live.clear()
            self._pending.clear()
        return _terminate_many(live, pending, self._panic_processes)

    def panic_close_all(self) -> int:
        """Drop browser SSH children without waiting on process cleanup."""

        self._closed = True
        # One atomic snapshot, no lock: ``tuple(dict.values())`` completes inside a
        # single C call, so a concurrent register or discard cannot be seen half-done.
        processes = tuple(self._panic_processes.values())
        if self._lock.acquire(blocking=False):
            try:
                self._live.clear()
                self._pending.clear()
            finally:
                self._lock.release()
        for process in processes:
            if process.poll() is not None:
                continue
            if not kill_process_group_id(process.pid):
                try:
                    process.kill()
                except Exception:
                    pass
        # Panic is terminal for this manager, so custody is released with everything
        # else — AFTER the loop, never before: a register emptied first would be a
        # panic that reached nothing.
        self._panic_processes.clear()
        return len(processes)

    def records(self) -> list[BrowserForward]:
        """Live forward records. TEST-FACING: no production reader (see
        tests/test_seam_producers.py::SEAM_WITHOUT_PRODUCER)."""

        with self._lock:
            return [live.record for live in self._live.values()]

    def _validated_config_digest(
        self,
        command: list[str],
        child_env: Mapping[str, str],
    ) -> str:
        raw = self._config_runner([command[0], "-G", *command[1:]], child_env)
        forward = command[command.index("-L") + 1]
        _reject_hostile_config(
            raw,
            child_env=child_env,
            expected_local_forward=_forward_tuple(forward),
        )
        return hashlib.sha256(raw).hexdigest()

    def _reserve_loopback_port(self) -> tuple[int, socket.socket]:
        for _attempt in range(20):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
            if port not in self._control_ports:
                return port, probe
            probe.close()
        raise BrowserForwardError("could not reserve a non-control loopback port")

    def _command(
        self,
        alias: str,
        local_port: int,
        remote_port: int,
    ) -> list[str]:
        overrides = (
            "BatchMode=yes",
            "ExitOnForwardFailure=yes",
            "ControlMaster=no",
            "ControlPath=none",
            "ControlPersist=no",
            "ForwardAgent=no",
            "ForwardX11=no",
            "ForwardX11Trusted=no",
            "PermitLocalCommand=no",
            "RemoteCommand=none",
            "RequestTTY=no",
            "SessionType=none",
            "Tunnel=no",
        )
        command = [self.ssh_binary, "-n", "-N", "-T", "-S", "none"]
        for value in overrides:
            command.extend(["-o", value])
        command.extend(
            [
                "-L",
                f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
                alias,
            ]
        )
        return command

    def _await_ready(
        self,
        process: subprocess.Popen[Any],
        local_port: int,
    ) -> tuple[bool, str]:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False, _stderr_text(process) or "SSH forward exited before ready"
            try:
                connection = self._connector(("127.0.0.1", local_port), timeout=0.1)
            except OSError:
                time.sleep(0.03)
                continue
            try:
                connection.close()
            except Exception:
                pass
            return True, ""
        return False, "SSH forward startup timed out"

    @staticmethod
    def _run_config(
        command: list[str],
        child_env: Mapping[str, str],
    ) -> bytes:
        proc = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            env=dict(child_env),
        )
        if proc.returncode:
            raise BrowserForwardError(proc.stderr.decode("utf-8", errors="replace") or "ssh -G failed")
        return bytes(proc.stdout)


def _reject_hostile_config(
    raw: bytes,
    *,
    child_env: Mapping[str, str],
    expected_local_forward: tuple[str, int, str, int],
) -> None:
    config: dict[str, list[str]] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        key, separator, value = line.strip().partition(" ")
        if not separator:
            continue
        config.setdefault(key.casefold(), []).append(value.strip())
    forbidden = {
        "remoteforward",
        "dynamicforward",
        "setenv",
    }
    present = sorted(key for key in forbidden if config.get(key))
    local_forwards = config.get("localforward", [])
    if len(local_forwards) != 1 or _forward_tuple(local_forwards[0]) != expected_local_forward:
        present.append("localforward")
    sendenv_patterns = [pattern for value in config.get("sendenv", []) for pattern in value.split()]
    for env_name in child_env:
        selected = False
        for pattern in sendenv_patterns:
            negate = pattern.startswith("-")
            candidate = pattern[1:] if negate else pattern
            if candidate and fnmatch.fnmatchcase(env_name, candidate):
                selected = not negate
        if selected:
            present.append("sendenv")
            break
    tunnel = " ".join(config.get("tunnel", [])).casefold()
    remote_command = " ".join(config.get("remotecommand", [])).strip().casefold()
    local_command = " ".join(config.get("localcommand", [])).strip().casefold()
    permit_local = " ".join(config.get("permitlocalcommand", [])).casefold()
    if tunnel not in {"", "no", "false"}:
        present.append("tunnel")
    if remote_command not in {"", "none"}:
        present.append("remotecommand")
    if local_command not in {"", "none"}:
        present.append("localcommand")
    if permit_local not in {"", "no", "false"}:
        present.append("permitlocalcommand")
    if present:
        raise BrowserForwardError(
            "SSH alias has forbidden forwarding/command/environment effects: "
            + ", ".join(sorted(set(present)))
            + ". Use a dedicated safe alias."
        )


def _forward_tuple(value: str) -> tuple[str, int, str, int]:
    endpoints = str(value or "").split()
    if len(endpoints) == 1:
        endpoints = endpoints[0].split(":")
        if len(endpoints) == 4:
            endpoints = [":".join(endpoints[:2]), ":".join(endpoints[2:])]
    if len(endpoints) != 2:
        return ("", 0, "", 0)

    parsed: list[tuple[str, int]] = []
    for endpoint in endpoints:
        host, separator, raw_port = endpoint.rpartition(":")
        host = host.strip("[]")
        if not separator:
            return ("", 0, "", 0)
        try:
            port = int(raw_port)
        except ValueError:
            return ("", 0, "", 0)
        parsed.append((host, port))
    return (*parsed[0], *parsed[1])


def _validated_connection_fields(alias_value: Any, port_value: Any) -> tuple[str, int]:
    alias = str(alias_value or "").strip()
    if not _SAFE_ALIAS.fullmatch(alias):
        raise BrowserForwardError("ssh_alias is invalid or option-shaped")
    try:
        port = int(port_value)
    except (TypeError, ValueError) as exc:
        raise BrowserForwardError("remote service port is invalid") from exc
    if not 1 <= port <= 65535:
        raise BrowserForwardError("remote service port must be 1..65535")
    return alias, port


def _ssh_child_env() -> dict[str, str]:
    allowed = {
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "SSH_AUTH_SOCK",
        "TMPDIR",
        "SYSTEMROOT",
        "WINDIR",
    }
    return {key: value for key, value in os.environ.items() if key in allowed and value}


def _stderr_text(process: subprocess.Popen[Any]) -> str:
    stream = getattr(process, "stderr", None)
    if stream is None:
        return ""
    try:
        data = stream.read(16_384)
    except Exception:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data or "")


def _terminate_many(
    live: list[_LiveForward],
    pending: list[subprocess.Popen[Any]],
    register: dict[int, subprocess.Popen[Any]] | None = None,
) -> int:
    processes = [item.process for item in live] + pending
    for process in processes:
        _terminate(process, register)
    return len(processes)


def _terminate(
    process: subprocess.Popen[Any],
    register: dict[int, subprocess.Popen[Any]] | None = None,
) -> None:
    # Ending panic custody belongs HERE, in the function that ends the process, rather
    # than beside each of the four close paths that call it — the register grew
    # unboundedly for exactly as long as it did because "and remember to un-register"
    # was a per-call-site obligation. Discarded FIRST, so a kill that raises still
    # ends custody: the child is condemned either way.
    if register is not None:
        register.pop(id(process), None)
    try:
        kill_process_tree(process)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    try:
        process.wait(timeout=3)
    except Exception:
        pass
