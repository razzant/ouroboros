"""The actual main/watchdog composition retains cleanup before either transfer."""
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("managed", [True, False], ids=["launcher", "direct"])
@pytest.mark.parametrize("uvicorn_returns", [True, False], ids=["returned", "held"])
def test_restart_waits_for_cleanup_and_transfers_once(monkeypatch, tmp_path, managed, uvicorn_returns):
    import server
    from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV

    cleanup_entered, release_cleanup, transfer, release_server = (threading.Event() for _ in range(4))
    calls, failures = [], []
    class DrainEvent(threading.Event):
        def wait(self, timeout=None):
            return super().wait(0.02 if timeout in (5, 30) else timeout)

    class FakeServer:
        def __init__(self, config):
            self.should_exit = False

        def watch_launcher_stop(self):
            pass

        def run(self, *, sockets):
            assert sockets[0].getsockname() == ("127.0.0.1", 9123)
            server._restart_requested.set()
            if not uvicorn_returns:
                assert release_server.wait(5)

    @contextmanager
    def listener(*args, **kwargs):
        yield SimpleNamespace(getsockname=lambda: ("127.0.0.1", 9123))

    def cleanup(**kwargs):
        calls.append(("cleanup", kwargs))
        cleanup_entered.set()
        assert release_cleanup.wait(5)
        calls.append(("cleanup_finished", None))

    def reexec(host, port, **kwargs):
        assert calls[-1][0] == "cleanup_finished"
        assert (host, port) == ("127.0.0.1", 9123)
        assert kwargs["owner_initiated"] is True
        assert server.os.environ[PLANNED_RESTART_TRANSACTION_ENV] == "planned-same-attempt"
        calls.append(("reexec", None))

    def exit_process(code):
        assert code == server.RESTART_EXIT_CODE
        calls.append(("exit", code))
        transfer.set()

    monkeypatch.setattr(server, "threading", SimpleNamespace(Event=DrainEvent, Thread=threading.Thread))
    monkeypatch.setattr(server, "_restart_requested", threading.Event())
    owner = threading.Event()
    owner.set()
    monkeypatch.setattr(server, "_owner_restart_requested", owner)
    monkeypatch.setattr(server, "_planned_delegate_restart_transaction_id", "planned-same-attempt")
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", managed)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_event_loop", None)
    monkeypatch.setattr(server, "load_settings", lambda: {})
    monkeypatch.setattr(server, "verify_settings_integrity", lambda: None)
    monkeypatch.setattr(server, "parse_server_args", lambda *a: SimpleNamespace(host="127.0.0.1", port=9123, host_explicit=False))
    monkeypatch.setattr(server, "find_free_port", lambda host, port: port)
    monkeypatch.setattr(server, "get_network_auth_startup_warning", lambda host: "")
    monkeypatch.setattr(server, "validate_network_auth_configuration", lambda host: "")
    monkeypatch.setattr(server, "bound_service_socket", listener)
    monkeypatch.setattr(server, "write_port_file", lambda *a: None)
    monkeypatch.setattr(server.uvicorn, "Config", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_SignalStopServer", FakeServer)
    monkeypatch.setattr(server, "_emergency_process_cleanup", cleanup)
    monkeypatch.setattr(server, "_restart_current_process_impl", reexec)
    monkeypatch.setattr("supervisor.update_merge.read_update_tx_strict", lambda: ("absent", {}))
    monkeypatch.setattr(server.os, "_exit", exit_process)
    monkeypatch.delenv(PLANNED_RESTART_TRANSACTION_ENV, raising=False)

    def run():
        try:
            server.main()
        except BaseException as exc:
            failures.append(exc)

    main = threading.Thread(target=run)
    main.start()
    try:
        assert cleanup_entered.wait(5)
        assert not transfer.is_set() and main.is_alive()
        release_cleanup.set()
        assert transfer.wait(5)
    finally:
        release_cleanup.set()
        release_server.set()
        main.join(5)
    assert not main.is_alive() and failures == []
    assert calls == [("cleanup", {"port_sweep": False}), ("cleanup_finished", None),
                     *([] if managed else [("reexec", None)]), ("exit", server.RESTART_EXIT_CODE)]


def test_restart_fallback_runs_pin_handoff_before_retained_executor_cleanup(monkeypatch, tmp_path):
    import server

    calls = []
    requested = threading.Event()
    requested.set()
    monkeypatch.setattr(server, "_restart_requested", requested)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server._historical_audit, "stop", lambda: None)
    monkeypatch.setattr(server, "_managed_update_pending_kwargs", lambda: {"preserve_pending": True})
    monkeypatch.setattr(server, "_restart_cleanup_kwargs", lambda: {})
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: calls.append(("workers", kw)))
    monkeypatch.setattr(server, "_stop_owned_daemon_for_new_pin", lambda: calls.append(("pin", {})))
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda: calls.append(("shell", {})))
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda root, **kw: calls.append(("foreground", kw)))
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda root, **kw: calls.append(("services", kw)))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: calls.append(("companions", {})))
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda port: pytest.fail("restart must not kill an unrelated listener by port"))
    server._emergency_process_cleanup(port_sweep=False)
    assert [name for name, kw in calls] == ["workers", "pin", "shell", "foreground", "services", "companions"]
    assert calls[0][1]["preserve_pending"] is True
    assert calls[3][1] == calls[4][1] == {"wait": False}


def test_direct_restart_uses_custodied_spawn_on_windows_and_exec_on_posix(monkeypatch, tmp_path):
    from ouroboros import config, delegate_recovery, platform_layer, process_custody, server_control
    from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    base_interpreter = str(tmp_path / "base-python.exe")
    monkeypatch.setattr(sys, "_base_executable", base_interpreter)
    monkeypatch.setattr(sys, "argv", ["server.py", "--port", "9123"])
    monkeypatch.setenv(PLANNED_RESTART_TRANSACTION_ENV, "planned-same-attempt")
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_MANAGED_REPO_DIR", "old-repo")
    monkeypatch.delenv("OUROBOROS_SERVER_REEXEC_ARGV_JSON", raising=False)
    calls = []
    handle = 0x123456789
    kernel = SimpleNamespace(CloseHandle=lambda value: calls.append(("close_handle", value)))
    monkeypatch.setattr(delegate_recovery, "open_windows_restart_parent",
                        lambda root, tx: (kernel, handle) if (root, tx) == (
                            tmp_path, "planned-same-attempt") else pytest.fail("wrong restart transaction"))
    monkeypatch.setattr(delegate_recovery, "bind_windows_restart_successor",
                        lambda root, tx, pid: calls.append(("bind", root, tx, pid)))

    def spawn(cmd, **kwargs):
        calls.append(("spawn", cmd, kwargs))
        return SimpleNamespace(pid=24680)

    def execvpe(executable, argv, env):
        calls.append(("exec", executable, argv, env))

    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    monkeypatch.setattr(server_control.os, "execvpe", execvpe)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    server_control.restart_current_process("127.0.0.1", 9123, repo_dir=tmp_path,
                                           log=logging.getLogger("test"))
    _, cmd, kwargs = calls.pop(0)
    assert calls == ([
        ("bind", tmp_path, "planned-same-attempt", 24680), ("close_handle", handle)
    ] if os.name == "nt" else [])
    if os.name == "nt":
        assert kwargs["startupinfo"].lpAttributeList == {"handle_list": [handle]}
        assert kwargs["close_fds"] is True
        assert kwargs["env"][delegate_recovery.WINDOWS_RESTART_PARENT_HANDLE_ENV] == str(handle)
    calls.clear()
    assert cmd == [base_interpreter, "server.py", "--port", "9123"]
    assert kwargs["env"]["__PYVENV_LAUNCHER__"] == sys.executable
    assert kwargs["drive_root"] == tmp_path
    assert kwargs["purpose"] == "server_restart_fallback"
    assert kwargs["scope"] == "daemon"
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["new_process_group"] is False
    assert kwargs["env"][PLANNED_RESTART_TRANSACTION_ENV] == "planned-same-attempt"
    assert kwargs["env"]["OUROBOROS_SERVER_PORT"] == "9123"
    assert "OUROBOROS_MANAGED_BY_LAUNCHER" not in kwargs["env"]
    assert "OUROBOROS_MANAGED_REPO_DIR" not in kwargs["env"]

    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    server_control.restart_current_process("127.0.0.1", 9123, repo_dir=tmp_path,
                                           log=logging.getLogger("test"))
    assert len(calls) == 1 and calls[0][0] == "exec"
    _, executable, argv, env = calls[0]
    assert executable == sys.executable
    assert argv == [sys.executable, "server.py", "--port", "9123"]
    assert env[PLANNED_RESTART_TRANSACTION_ENV] == "planned-same-attempt"
    assert env["OUROBOROS_SERVER_PORT"] == "9123"


@pytest.mark.parametrize("failure", ["spawn", "bind"])
def test_windows_direct_restart_spawn_or_binding_failure_cannot_ack(monkeypatch, tmp_path, failure):
    """Portable Popen seam: the native STARTUPINFO/HANDLE transfer needs Windows CI."""
    from ouroboros import config, delegate_recovery, process_custody, server_control

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_control, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(subprocess, "STARTUPINFO", lambda: SimpleNamespace(), raising=False)
    tx = {"transaction_id": "planned", "status": "prepared", "supervisor_pid": os.getpid(),
          "task_ids": ["wait-a"]}
    delegate_recovery._write_restart_transaction(tmp_path, tx)
    handle = 0x123456789  # Above DWORD: startup handle list and env must retain all bits.
    kernel = SimpleNamespace(closed=[], CloseHandle=lambda value: kernel.closed.append(value))
    monkeypatch.setattr(delegate_recovery, "open_windows_restart_parent", lambda *_: (kernel, handle))

    child = SimpleNamespace(pid=24680, terminated=False, waited=False)
    child.terminate = lambda: setattr(child, "terminated", True)
    child.wait = lambda timeout: setattr(child, "waited", timeout == 5)

    def spawn(_argv, **kwargs):
        assert kwargs["new_process_group"] is False
        assert kwargs["startupinfo"].lpAttributeList == {"handle_list": [handle]}
        assert kwargs["close_fds"] is True
        assert kwargs["env"][delegate_recovery.WINDOWS_RESTART_PARENT_HANDLE_ENV] == str(handle)
        if failure == "spawn":
            raise OSError("spawn refused")
        return child

    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    if failure == "bind":
        monkeypatch.setattr(delegate_recovery, "bind_windows_restart_successor",
                            lambda *_: (_ for _ in ()).throw(OSError("binding refused")))
    with pytest.raises(OSError, match="spawn refused" if failure == "spawn" else "binding refused"):
        server_control._spawn_restart_successor(
            [sys.executable, "server.py"],
            {delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV: "planned"},
            tmp_path, new_process_group=False,
        )
    assert kernel.closed == [handle]
    assert child.terminated is (failure == "bind")
    assert child.waited is (failure == "bind")
    assert delegate_recovery._read_restart_transaction(tmp_path, "planned")["status"] == "prepared"


def test_windows_direct_restart_parent_and_successor_identity_are_bound(monkeypatch, tmp_path):
    from ouroboros import delegate_recovery, platform_layer

    tx = {"transaction_id": "planned", "status": "prepared", "supervisor_pid": os.getpid(),
          "task_ids": ["wait-a"]}
    delegate_recovery._write_restart_transaction(tmp_path, tx)
    from ouroboros.utils import atomic_write_json
    atomic_write_json(delegate_recovery._active_restart_transaction_path(tmp_path),
                      {"transaction_id": "planned", "supervisor_pid": os.getpid()})

    handle = 0x123456789
    birth = (1 << 32) | 12345

    class Kernel:
        def __init__(self):
            self.closed = []

        def OpenProcess(self, access, inherit, pid):
            assert (access, inherit, pid) == (0x101000, True, os.getpid())
            return handle

        def GetProcessId(self, handle):
            assert handle == 0x123456789
            return os.getpid()

        def GetProcessTimes(self, handle, created, *_rest):
            created._obj.dwLowDateTime = 12345
            created._obj.dwHighDateTime = 1
            return True

        def CloseHandle(self, handle):
            self.closed.append(handle)

    kernel = Kernel()
    monkeypatch.setattr(delegate_recovery, "_windows_restart_kernel32", lambda: kernel)
    assert delegate_recovery.open_windows_restart_parent(tmp_path, "planned") == (kernel, handle)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda pid: "win-filetime:67890")
    delegate_recovery.bind_windows_restart_successor(tmp_path, "planned", 24680)
    observed = delegate_recovery._read_restart_transaction(tmp_path, "planned")
    assert observed["direct_spawn_parent_birth"] == f"win-filetime:{birth}"
    assert (observed["direct_spawn_successor_pid"], observed["direct_spawn_successor_birth"]) == (
        24680, "win-filetime:67890")
    assert observed["status"] == "prepared" and kernel.closed == []


def test_windows_direct_restart_never_treats_same_pid_token_as_exec(monkeypatch, tmp_path):
    from ouroboros import delegate_recovery, platform_layer

    delegate_recovery._write_restart_transaction(tmp_path, {
        "transaction_id": "planned", "status": "prepared", "supervisor_pid": os.getpid(),
        "task_ids": ["wait-a"],
    })
    monkeypatch.setenv(delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV, "planned")
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    delegate_recovery._ack_direct_exec_successor(tmp_path)
    assert delegate_recovery._read_restart_transaction(tmp_path, "planned")["status"] == "prepared"


@pytest.mark.parametrize("failure", ["", "fallback", "cleanup", "reexec"],
                         ids=["reexec-ok", "spawn-fallback-ok", "cleanup-error", "reexec-error"])
@pytest.mark.parametrize("uvicorn_returns", [False, True], ids=["held", "returned"])
def test_direct_watchdog_physically_transfers_or_fails(tmp_path, failure, uvicorn_returns):
    """Thread outcomes are physical exec or nonzero exit, never success/hang after an error."""
    script = tmp_path / "direct_reexec.py"
    receipt = tmp_path / "successor.json"
    script.write_text('''
from contextlib import contextmanager
import json, os, pathlib, sys, threading
from types import SimpleNamespace

receipt = pathlib.Path(sys.argv[1])
failure, uvicorn_returns = sys.argv[2], sys.argv[3] == "returned"
if os.environ.get("OURO_TEST_SUCCESSOR"):
    from ouroboros.delegate_recovery import (PLANNED_RESTART_TRANSACTION_ENV,
        _ack_direct_exec_successor, _read_restart_transaction)
    from ouroboros.config import DATA_DIR
    token = os.environ.get(PLANNED_RESTART_TRANSACTION_ENV)
    _ack_direct_exec_successor(DATA_DIR)
    receipt.write_text(json.dumps({
        "pid": os.getpid(), "previous_pid": int(os.environ["OURO_TEST_SUCCESSOR"]),
        "transaction": token,
        "transaction_status": _read_restart_transaction(DATA_DIR, token or "").get("status"),
        "ack_source": _read_restart_transaction(DATA_DIR, token or "").get("ack_source"),
        "port": os.environ.get("OUROBOROS_SERVER_PORT"),
        "cleanup": os.environ.get("OURO_TEST_CLEANUP"),
    }), encoding="utf-8")
    sys.exit(0)

import server
import supervisor.update_merge
from ouroboros import delegate_recovery
from ouroboros.utils import atomic_write_json
transaction = {"transaction_id": "physical-handoff", "status": "prepared",
               "supervisor_pid": os.getpid(), "task_ids": ["held-wait"]}
delegate_recovery._write_restart_transaction(server.DATA_DIR, transaction)
active = delegate_recovery._active_restart_transaction_path(server.DATA_DIR)
active.parent.mkdir(parents=True, exist_ok=True)
atomic_write_json(active, {"transaction_id": "physical-handoff", "supervisor_pid": os.getpid()})
class DrainEvent(threading.Event):
    def wait(self, timeout=None):
        return super().wait(0.02 if timeout == 30 else timeout)
class HeldServer:
    def __init__(self, config):
        self.should_exit = False
    def watch_launcher_stop(self):
        pass
    def run(self, **kwargs):
        server._restart_requested.set()
        if not uvicorn_returns:
            threading.Event().wait(20)
            raise AssertionError("the direct watchdog never replaced the held server")
@contextmanager
def listener(*args, **kwargs):
    yield SimpleNamespace(getsockname=lambda: ("127.0.0.1", 9123))
def cleanup(**kwargs):
    assert kwargs == {"port_sweep": False}
    if failure == "cleanup":
        raise OSError("fixture cleanup failure")
    os.environ["OURO_TEST_CLEANUP"] = "completed"
    os.environ["OURO_TEST_SUCCESSOR"] = str(os.getpid())
def reexec_failure(*args, **kwargs):
    raise OSError("fixture reexec failure")

server.threading = SimpleNamespace(Event=DrainEvent, Thread=threading.Thread)
server._restart_requested = threading.Event()
server._event_loop = None
server._LAUNCHER_MANAGED = False
server._planned_delegate_restart_transaction_id = "physical-handoff"
server.verify_settings_integrity = lambda: None
server.load_settings = lambda: {}
server.parse_server_args = lambda *args: SimpleNamespace(host="127.0.0.1", port=9123, host_explicit=False)
server.find_free_port = lambda host, port: port
server.get_network_auth_startup_warning = lambda host: ""
server.validate_network_auth_configuration = lambda host: ""
server.bound_service_socket = listener
server.write_port_file = lambda *args: None
server.uvicorn.Config = lambda *args, **kwargs: None
server._SignalStopServer = HeldServer
server._emergency_process_cleanup = cleanup
if failure in {"reexec", "fallback"} or (os.name == "nt" and not failure):
    from ouroboros import server_control, process_custody
    if failure in {"reexec", "fallback"}:
        server_control.os.execvpe = reexec_failure
    if failure == "reexec":
        process_custody.spawn_supervised = reexec_failure
    elif os.name != "nt":
        spawn = process_custody.spawn_supervised
        def joined_spawn(*args, **kwargs):
            child = spawn(*args, **kwargs)
            assert child.wait(timeout=10) == 0
            return child
        process_custody.spawn_supervised = joined_spawn
supervisor.update_merge.read_update_tx_strict = lambda: ("absent", {})
sys.exit(server.main())
''', encoding="utf-8")
    env = os.environ.copy()
    env.pop("OUROBOROS_SERVER_REEXEC_ARGV_JSON", None)
    env.pop("OURO_TEST_SUCCESSOR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, str(script), str(receipt), failure,
                             "returned" if uvicorn_returns else "held"], env=env,
                            capture_output=True, text=True, timeout=15)
    if failure in {"cleanup", "reexec"}:
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Restart failed; cleanup or transfer is unconfirmed" in result.stderr
        assert "fixture " + failure + " failure" in result.stderr
        assert not receipt.exists()
        return
    assert result.returncode == (42 if failure == "fallback" or (os.name == "nt" and not failure) else 0), result.stdout + result.stderr
    if os.name == "nt":
        deadline = time.monotonic() + 10
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    observed = json.loads(receipt.read_text(encoding="utf-8"))
    assert observed["transaction"] == "physical-handoff"
    assert observed["port"] == "9123" and observed["cleanup"] == "completed"
    assert (observed["pid"] == observed["previous_pid"]) is (not failure and os.name != "nt")
    if os.name == "nt":
        assert (observed["transaction_status"], observed["ack_source"]) == (
            "normal_exit_acknowledged", "windows_direct_parent_handle")
    elif not failure:
        assert (observed["transaction_status"], observed["ack_source"]) == (
            "normal_exit_acknowledged", "direct_exec_successor")
    else:
        assert observed["transaction_status"] == "prepared"
