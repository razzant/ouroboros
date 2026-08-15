"""The forwarding exemption, and the aliases it refuses (RWS v2 §3.2).

Transferred from the donor, split at the lane boundary: the forward MANAGER cases
live here — the real `ssh -L` child, its `ssh -G` validation, its custody
registration and its panic teardown.

The CONSUMER side landed separately and is covered by
`tests/test_browser_remote_forward.py`: `tools/browser.py::_resolve_placement_url`
opening a forward for a remote task's loopback URL, the rewrite onto the local end,
the per-request foreign-origin block, and the typed refusals. `remote_file_bridge`
does not exist and is not planned for v1 — a `file://` across the placement boundary
is a DEFERRED capability (see the ARCHITECTURE deferral list), so the donor's
`test_remote_file_bridge_*` cases have no code to exercise here and were not
transferred.

Loopback HTTP/WS forwarding is the ONE named exemption to "every byte crosses
through the transfer service", so its guards carry the weight the transfer
service would otherwise carry: required custody, exactly the recorded `-L` and
nothing else, no reuse of an owner control socket, publication only after the
forward is proven ready, and closure on task end, connection teardown and panic.
"""

from __future__ import annotations

import io
import subprocess
import threading

import pytest

from ouroboros.remote_browser_forward import (
    BrowserForwardError,
    SSHBrowserForwardManager,
)

_SAFE_CONFIG = b"""
hostname example.invalid
user deploy
tunnel false
remotecommand none
permitlocalcommand no
controlmaster auto
controlpersist 600
sendenv LANG
sendenv LC_*
"""


def _config_for_command(command):
    forward = command[command.index("-L") + 1].split(":")
    local_host, local_port, remote_host, remote_port = forward
    return _SAFE_CONFIG + (
        f"localforward [{local_host}]:{local_port} "
        f"[{remote_host}]:{remote_port}\n"
    ).encode()


class _Connection:
    def close(self):
        return None


class _Process:
    _next_pid = 9000

    def __init__(self, *, returncode=None, stderr=b""):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = returncode
        self.stderr = io.BytesIO(stderr)
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        del timeout
        return self.returncode


def test_forward_uses_required_custody_exact_loopback_and_no_multiplexing(
    tmp_path,
):
    spawned = []
    config_calls = []

    def _spawn(command, **kwargs):
        spawned.append((command, kwargs))
        return _Process()

    def _config(command, child_env):
        config_calls.append((command, dict(child_env)))
        return _config_for_command(command)

    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=_config,
        process_spawner=_spawn,
        connector=lambda *args, **kwargs: _Connection(),
    )
    record = manager.open(
        {"id": "connection", "ssh_alias": "safe-alias"},
        remote_port=4321,
        task_id="task",
    )

    command, kwargs = spawned[0]
    assert kwargs["required_custody"] is True
    assert kwargs["new_process_group"] is True
    assert kwargs["owner_task_id"] == "task"
    assert "-N" in command and "-T" in command and "-S" in command
    assert command[command.index("-S") + 1] == "none"
    assert command.count("-L") == 1
    assert not any(part == "-D" for part in command)
    assert command[-1] == "safe-alias"
    assert (
        command[command.index("-L") + 1]
        == f"127.0.0.1:{record.local_port}:127.0.0.1:4321"
    )
    assert record.origin == f"http://127.0.0.1:{record.local_port}"
    assert record.url == record.origin + "/"
    assert record.task_token
    assert len(config_calls) == 2
    assert all(
        config_command == [command[0], "-G", *command[1:]]
        for config_command, _child_env in config_calls
    )
    assert all(
        config_env == kwargs["env"]
        for _config_command, config_env in config_calls
    )
    assert manager.close(record.forward_id) is True


@pytest.mark.parametrize(
    "line",
    [
        b"localforward 127.0.0.1:1 127.0.0.1:2\n",
        b"remoteforward 1 127.0.0.1:2\n",
        b"dynamicforward 1080\n",
        b"tunnel point-to-point\n",
        b"remotecommand touch /tmp/pwned\n",
        b"localcommand touch /tmp/pwned\n",
        b"permitlocalcommand yes\n",
        b"setenv SECRET=value\n",
    ],
)
def test_hostile_effective_alias_is_rejected_before_spawn(tmp_path, line):
    spawned = []
    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=lambda command, _child_env: _config_for_command(command) + line,
        process_spawner=lambda *args, **kwargs: spawned.append((args, kwargs)),
    )
    with pytest.raises(BrowserForwardError, match="forbidden"):
        manager.open(
            {"id": "connection", "ssh_alias": "hostile"},
            remote_port=8080,
            task_id="task",
        )
    assert spawned == []


def test_standard_or_unretained_sendenv_patterns_are_allowed(tmp_path):
    spawned = []

    def _spawn(command, **kwargs):
        spawned.append((command, kwargs))
        return _Process()

    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=lambda command, _child_env: (
            _config_for_command(command) + b"sendenv UNRETAINED_*\n"
        ),
        process_spawner=_spawn,
        connector=lambda *args, **kwargs: _Connection(),
    )
    record = manager.open(
        {"id": "connection", "ssh_alias": "safe"},
        remote_port=8080,
        task_id="task",
    )
    assert spawned
    assert manager.close(record.forward_id) is True


@pytest.mark.parametrize("retained_name", ["HOME", "PATH", "SSH_AUTH_SOCK"])
def test_sendenv_matching_retained_child_env_is_rejected(
    tmp_path,
    monkeypatch,
    retained_name,
):
    monkeypatch.setenv(retained_name, f"/retained/{retained_name.lower()}")
    spawned = []
    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=lambda command, _child_env: (
            _config_for_command(command)
            + f"sendenv {retained_name}\n".encode()
        ),
        process_spawner=lambda *args, **kwargs: spawned.append((args, kwargs)),
    )
    with pytest.raises(BrowserForwardError, match="sendenv"):
        manager.open(
            {"id": "connection", "ssh_alias": "unsafe-sendenv"},
            remote_port=8080,
            task_id="task",
        )
    assert spawned == []


@pytest.mark.parametrize(
    "alias",
    [
        "-oProxyCommand=evil",
        "safe alias",
        "safe\nHost evil",
        "",
        "../unsafe",
    ],
)
def test_option_shaped_or_hostile_alias_is_rejected(tmp_path, alias):
    config_calls = []
    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=lambda command, _child_env: (
            config_calls.append(command) or _config_for_command(command)
        ),
    )
    with pytest.raises(BrowserForwardError, match="ssh_alias"):
        manager.open(
            {"id": "connection", "ssh_alias": alias},
            remote_port=8080,
            task_id="task",
        )
    assert config_calls == []


def test_config_digest_change_between_probe_and_spawn_fails_closed(tmp_path):
    config_calls = 0

    def _changing_config(command, _child_env):
        nonlocal config_calls
        config_calls += 1
        return _config_for_command(command) + (
            b"user changed\n" if config_calls == 2 else b""
        )

    spawned = []
    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=_changing_config,
        process_spawner=lambda *args, **kwargs: spawned.append((args, kwargs)),
    )
    with pytest.raises(BrowserForwardError, match="config changed"):
        manager.open(
            {"id": "connection", "ssh_alias": "safe"},
            remote_port=8080,
            task_id="task",
        )
    assert spawned == []


def test_ephemeral_bind_race_is_retried_and_only_ready_url_is_published(
    tmp_path,
    monkeypatch,
):
    processes = [
        _Process(
            returncode=255,
            stderr=b"bind [127.0.0.1]: Address already in use\n",
        ),
        _Process(),
    ]
    spawned = []

    def _spawn(command, **kwargs):
        spawned.append((command, kwargs))
        return processes[len(spawned) - 1]

    monkeypatch.setattr(
        "ouroboros.remote_browser_forward.kill_process_tree",
        lambda process: process.kill(),
    )
    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=lambda command, _child_env: _config_for_command(command),
        process_spawner=_spawn,
        connector=lambda *args, **kwargs: _Connection(),
    )
    ports = iter([41001, 41002])

    class _Probe:
        def close(self):
            return None

    monkeypatch.setattr(
        manager,
        "_reserve_loopback_port",
        lambda: (next(ports), _Probe()),
    )
    record = manager.open(
        {"id": "connection", "ssh_alias": "safe"},
        remote_port=8080,
        task_id="task",
    )
    first_port = int(spawned[0][0][spawned[0][0].index("-L") + 1].split(":")[1])
    assert len(spawned) == 2
    assert processes[0].killed is True
    assert first_port == 41001
    assert record.local_port == 41002


def test_task_connection_and_global_cleanup_terminate_owned_children(
    tmp_path,
    monkeypatch,
):
    processes = []

    def _spawn(*args, **kwargs):
        del args, kwargs
        process = _Process()
        processes.append(process)
        return process

    monkeypatch.setattr(
        "ouroboros.remote_browser_forward.kill_process_tree",
        lambda process: process.kill(),
    )
    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=lambda command, _child_env: _config_for_command(command),
        process_spawner=_spawn,
        connector=lambda *args, **kwargs: _Connection(),
    )
    manager.open(
        {"id": "one", "ssh_alias": "safe-one"},
        remote_port=8001,
        task_id="task-a",
    )
    manager.open(
        {"id": "one", "ssh_alias": "safe-one"},
        remote_port=8002,
        task_id="task-b",
    )
    manager.open(
        {"id": "two", "ssh_alias": "safe-two"},
        remote_port=8003,
        task_id="task-c",
    )

    assert manager.close_task("task-a") == 1
    assert manager.close_connection("one") == 1
    assert manager.close_all() == 1
    assert all(process.killed for process in processes)
    assert manager.records() == []


def test_panic_cleanup_kills_a_forward_still_in_startup(tmp_path, monkeypatch):
    process = _Process()
    entered = threading.Event()
    release = threading.Event()
    outcome = []

    monkeypatch.setattr(
        "ouroboros.remote_browser_forward.kill_process_tree",
        lambda child: child.kill(),
    )
    monkeypatch.setattr(
        "ouroboros.remote_browser_forward.os.getpgid",
        lambda _pid: (_ for _ in ()).throw(ProcessLookupError()),
    )
    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=lambda command, _child_env: _config_for_command(command),
        process_spawner=lambda *args, **kwargs: process,
    )

    def _blocked_ready(child, port):
        del child, port
        entered.set()
        release.wait(2)
        return False, "closed"

    monkeypatch.setattr(manager, "_await_ready", _blocked_ready)

    def _open():
        try:
            manager.open(
                {"id": "connection", "ssh_alias": "safe"},
                remote_port=8080,
                task_id="task",
            )
        except Exception as exc:
            outcome.append(exc)

    thread = threading.Thread(target=_open)
    thread.start()
    assert entered.wait(1)
    assert manager.panic_close_all() == 1
    assert process.killed is True
    release.set()
    thread.join(2)
    assert outcome and isinstance(outcome[0], BrowserForwardError)
    assert manager.records() == []


def test_panic_cleanup_does_not_wait_for_held_manager_lock(tmp_path):
    manager = SSHBrowserForwardManager(tmp_path)
    process = _Process()
    manager._panic_processes[id(process)] = process
    entered = threading.Event()
    release = threading.Event()

    def hold_lock():
        with manager._lock:
            entered.set()
            release.wait(2)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert entered.wait(1)
    started = __import__("time").monotonic()
    assert manager.panic_close_all() == 1
    elapsed = __import__("time").monotonic() - started
    release.set()
    holder.join(1)

    assert elapsed < 0.2
    assert process.killed is True


def test_process_returned_after_panic_is_killed_before_registration(tmp_path):
    process = _Process()
    manager = None

    def spawn(*_args, **_kwargs):
        assert manager is not None
        manager.panic_close_all()
        return process

    manager = SSHBrowserForwardManager(
        tmp_path,
        config_runner=lambda command, _child_env: _config_for_command(command),
        process_spawner=spawn,
    )

    with pytest.raises(BrowserForwardError, match="closed during startup"):
        manager.open(
            {"id": "connection", "ssh_alias": "safe"},
            remote_port=8080,
            task_id="task",
        )
    assert process.killed is True
    assert manager.records() == []


def test_real_config_runner_surfaces_ssh_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ouroboros.remote_browser_forward.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=255,
            stdout=b"",
            stderr=b"bad config",
        ),
    )
    manager = SSHBrowserForwardManager(tmp_path)
    with pytest.raises(BrowserForwardError, match="bad config"):
        manager._run_config(["ssh", "-G", "bad"], {"HOME": str(tmp_path)})
