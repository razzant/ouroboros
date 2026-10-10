"""The actual main/watchdog composition retains cleanup before either transfer."""
from contextlib import contextmanager
import json
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
    monkeypatch.setattr(server, "_managed_update_pending_kwargs", lambda: {"preserve_pending": True})
    monkeypatch.setattr(server, "_restart_cleanup_kwargs", lambda: {})
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: calls.append(("workers", kw)))
    monkeypatch.setattr(server, "_stop_owned_daemon_for_new_pin", lambda: calls.append(("pin", {})))
    monkeypatch.setattr(server, "stop_owned_work", lambda root: calls.append(("owned-stop", {"root": root})))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: calls.append(("companions", {})))
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda port: pytest.fail("restart must not kill an unrelated listener by port"))
    server._emergency_process_cleanup(port_sweep=False)
    assert [name for name, kw in calls] == ["workers", "pin", "owned-stop", "companions"]
    assert calls[0][1]["preserve_pending"] is True
    assert calls[2][1] == {"root": tmp_path}


@pytest.mark.parametrize("failure", ["", "fallback", "cleanup", "reexec"],
                         ids=["reexec-ok", "spawn-fallback-ok", "cleanup-error", "reexec-error"])
@pytest.mark.parametrize("uvicorn_returns", [False, True], ids=["held", "returned"])
def test_direct_watchdog_physically_transfers_or_fails(tmp_path, failure, uvicorn_returns):
    """Thread outcomes are physical exec or nonzero exit, never success/hang after an error."""
    script = tmp_path / "direct reexec.py"
    receipt = tmp_path / "successor receipt.json"
    script.write_text('''
from contextlib import contextmanager
import json, os, pathlib, subprocess, sys, threading, time
from types import SimpleNamespace

receipt = pathlib.Path(sys.argv[1])
failure, uvicorn_returns = sys.argv[2], sys.argv[3] == "returned"
if os.environ.get("OURO_TEST_SUCCESSOR"):
    from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV
    if os.getpid() != int(os.environ["OURO_TEST_SUCCESSOR"]):
        # A spawned successor reports only after the test saw its caller exit:
        # one that dies with the caller leaves no receipt.
        caller_exited = receipt.with_name(receipt.name + ".caller-exited")
        deadline = time.monotonic() + 20
        while not caller_exited.exists():
            if time.monotonic() > deadline:
                sys.exit(3)
            time.sleep(0.05)
    staged = receipt.with_name(receipt.name + ".tmp")
    staged.write_text(json.dumps({
        "pid": os.getpid(), "previous_pid": int(os.environ["OURO_TEST_SUCCESSOR"]),
        "transaction": os.environ.get(PLANNED_RESTART_TRANSACTION_ENV),
        "port": os.environ.get("OUROBOROS_SERVER_PORT"),
        "cleanup": os.environ.get("OURO_TEST_CLEANUP"),
        "argv": sys.argv[1:],
    }), encoding="utf-8")
    os.replace(staged, receipt)  # the test polls for the receipt: never show it half-written
    sys.exit(0)

import server
import supervisor.update_merge
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
from ouroboros import server_control, process_custody
if failure in {"reexec", "fallback"}:
    server_control.os.execvpe = reexec_failure
if failure == "reexec":
    process_custody.spawn_supervised = reexec_failure
elif os.name == "nt" or failure == "fallback":
    # Windows replaces by spawning. The successor gets its own stdio so the test
    # sees THIS process exit first; the successor then has to survive it to report.
    spawn = process_custody.spawn_supervised
    def detached_spawn(*args, **kwargs):
        log = open(receipt.with_name(receipt.name + ".successor.log"), "w", encoding="utf-8")
        kwargs.update(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        return spawn(*args, **kwargs)
    process_custody.spawn_supervised = detached_spawn
supervisor.update_merge.read_update_tx_strict = lambda: ("absent", {})
sys.exit(server.main())
''', encoding="utf-8")
    env = os.environ.copy()
    env.pop("OUROBOROS_SERVER_REEXEC_ARGV_JSON", None)
    env.pop("OURO_TEST_SUCCESSOR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    arguments = [str(receipt), failure, "returned" if uvicorn_returns else "held",
                 'argument with "quotes" and a trailing\\']
    result = subprocess.run([sys.executable, str(script), *arguments], env=env,
                            capture_output=True, text=True, timeout=15)
    if failure in {"cleanup", "reexec"}:
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Restart failed; cleanup or transfer is unconfirmed" in result.stderr
        assert "fixture " + failure + " failure" in result.stderr
        assert not receipt.exists()
        return
    spawned = os.name == "nt" or failure == "fallback"
    assert result.returncode == (42 if spawned else 0), result.stdout + result.stderr
    if spawned:
        receipt.with_name(receipt.name + ".caller-exited").write_text("", encoding="utf-8")
        deadline = time.monotonic() + 20
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        successor_log = receipt.with_name(receipt.name + ".successor.log")
        assert receipt.exists(), "the spawned successor did not outlive its caller: " + (
            successor_log.read_text(encoding="utf-8") if successor_log.exists() else "no successor log")
    observed = json.loads(receipt.read_text(encoding="utf-8"))
    assert observed["transaction"] == "physical-handoff"
    assert observed["port"] == "9123" and observed["cleanup"] == "completed"
    assert observed["argv"] == arguments
    assert (observed["pid"] == observed["previous_pid"]) is (not spawned)


@pytest.mark.parametrize("platform,failure", [
    ("nt", ""), ("nt", "spawn"),
    ("posix", ""), ("posix", "exec"), ("posix", "spawn"),
])
def test_direct_transfer_uses_platform_process_primitive(monkeypatch, tmp_path, platform, failure):
    """Windows must bypass CRT exec; POSIX keeps exec and its spawn fallback."""
    from ouroboros import config, server_control, process_custody

    class Transferred(BaseException):
        pass

    calls, captured = [], {}
    argv = ["server path.py", "", 'argument with "quotes"', "trailing\\"]
    environment = {"OUROBOROS_SERVER_HOST": "127.0.0.1", "S1_TOKEN": "same generation",
                   "OUROBOROS_MANAGED_BY_LAUNCHER": "1", "OUROBOROS_MANAGED_REPO_DIR": "old"}

    def execvpe(executable, command, env):
        calls.append("exec")
        assert platform != "nt", "Windows restart must never enter CRT exec"
        captured.update(argv=command, env=env)
        if failure:
            raise OSError("exec failed")
        raise Transferred()

    def spawn(command, **kwargs):
        calls.append("spawn")
        captured.update(argv=command, **kwargs)
        if failure == "spawn":
            raise OSError("spawn failed")
        return SimpleNamespace(pid=1234)

    # Select the existing platform flag without changing pathlib's host OS.
    monkeypatch.setattr(server_control, "IS_WINDOWS", platform == "nt", raising=False)
    monkeypatch.setattr(server_control, "os", SimpleNamespace(
        name=platform, environ=environment, execvpe=execvpe))
    monkeypatch.setattr(server_control, "sys", SimpleNamespace(executable="python", argv=argv))
    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    logger = SimpleNamespace(info=lambda *a: None, exception=lambda *a: None)

    def transfer():
        server_control.restart_current_process("127.0.0.1", 9123, repo_dir=tmp_path, log=logger)

    if failure == "spawn":
        with pytest.raises(OSError, match="spawn failed"):
            transfer()
    elif platform == "posix" and not failure:
        with pytest.raises(Transferred):
            transfer()
    else:
        transfer()
    assert calls == (["spawn"] if platform == "nt" else ["exec"] + (["spawn"] if failure else []))
    assert captured["argv"] == ["python", *argv]
    assert captured["env"] == {"OUROBOROS_SERVER_HOST": "127.0.0.1",
                               "S1_TOKEN": "same generation", "OUROBOROS_SERVER_PORT": "9123"}
    if "spawn" in calls:
        # Windows keeps the console group (CTRL+C); POSIX isolates the fallback's session.
        assert captured["new_process_group"] is (platform != "nt")
        assert captured["scope"] == "daemon"
        assert captured["drive_root"] == tmp_path and captured["cwd"] == str(tmp_path)
