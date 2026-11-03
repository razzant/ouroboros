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


@pytest.mark.parametrize("phase", ["publication", "custody", "success", "early-cleanup", "posix-exec"])
@pytest.mark.parametrize("uvicorn_returns", [True, False], ids=["returned", "held"])
@pytest.mark.parametrize("panic_entry", ["executor", "ingress", "failed-ingress"])
def test_actual_restart_watcher_leaves_termination_to_panic(tmp_path, phase, uvicorn_returns, panic_entry):
    """Panic persists before physical exit 99, including when the watcher raises.

    Run the real main/watcher, spawn consumer and Panic owner in a disposable
    interpreter. Native child handles, exec and unrelated stop owners are doubles;
    the process exit, Panic flag and disabled state are real. The ingress case
    holds its callback before execute_panic_stop can claim termination itself.
    Failed ingress start runs the same executor synchronously in the calling
    thread. The POSIX case must keep this interpreter alive instead of entering
    exec after Panic was accepted during cleanup.
    """
    script = tmp_path / "watcher_panic.py"
    ready, release = tmp_path / "watcher-ready", tmp_path / "release-panic"
    script.write_text('''
from contextlib import contextmanager
import os, pathlib, sys, threading, time
from types import SimpleNamespace
import server
import supervisor.workers  # normal server startup establishes the worker import graph
from ouroboros import platform_layer, process_custody, server_control
from ouroboros.startup_historical_audit import audit
from supervisor import state

phase, uvicorn_returns, panic_entry = sys.argv[1], sys.argv[2] == "returned", sys.argv[3]
ready, release = map(pathlib.Path, sys.argv[4:])
panic_paused = threading.Event()

def pause_panic():
    panic_paused.set()
    deadline = time.monotonic() + 10
    while not release.exists():
        if time.monotonic() > deadline:
            raise AssertionError("test did not release Panic")
        time.sleep(.01)

write_flag = server_control._write_panic_flag
def paused_flag(root):
    pause_panic()
    write_flag(root)

def start_panic():
    def stop():
        if panic_entry != "executor":
            pause_panic()  # accepted, but the executor has not run even its first line
        server_control.execute_panic_stop(None, lambda: None,
            data_dir=server.DATA_DIR, panic_exit_code=99, log=server.log)

    if panic_entry == "failed-ingress":
        start = threading.Thread.start
        def failed_start(thread):
            if thread.name == "panic-ingress":
                raise RuntimeError("test ingress thread start failed")
            return start(thread)
        threading.Thread.start = failed_start
        # The ingress caller and restart watcher are independent threads.
        threading.Thread(target=lambda: server_control.PanicIngress(stop).request("/panic"), daemon=True).start()
    elif panic_entry == "ingress":
        assert server_control.PanicIngress(stop).request("/panic")
    else:
        # The emergency caller is a daemon, so returning main could exit 0.
        threading.Thread(target=stop, daemon=True).start()
    assert panic_paused.wait(10)

audit.stop = pause_panic if phase == "early-cleanup" else lambda: None
# Audit is independent of persistence; hold the actual writer in every phase.
server_control._write_panic_flag = paused_flag
import multiprocessing
import ouroboros.claudexor_daemon, ouroboros.extension_companion
import ouroboros.local_model, ouroboros.tools.shell, ouroboros.tools.services
import ouroboros.workspace_executor, supervisor.update_merge
multiprocessing.active_children = lambda: []
ouroboros.claudexor_daemon.get_owned_daemon = lambda **kw: SimpleNamespace(stop_outcome=lambda: True)
ouroboros.local_model.get_manager = lambda **kw: None
ouroboros.tools.shell.kill_all_tracked_subprocesses = lambda **kw: []
ouroboros.tools.services.kill_all_services = lambda *a, **kw: []
ouroboros.workspace_executor.kill_all_foreground = lambda *a, **kw: []
ouroboros.extension_companion.panic_kill_all = lambda **kw: []
platform_layer.kill_process_on_port = lambda port: None
platform_layer.request_process_tree_kill = lambda proc: {"requested": True, "pid": proc.pid}
process_custody.kill_process_tree = lambda proc: None

class Successor:
    pid = 4242
    returncode = None
    def __init__(self, *a, **kw):
        if phase == "publication":
            start_panic()
    def wait(self, **kw):
        self.returncode = 0
        return self.returncode
    def poll(self):
        return self.returncode
process_custody.subprocess = SimpleNamespace(Popen=Successor)
def record(*a, **kw):
    if phase in {"custody", "success"}:
        start_panic()
    if phase == "custody":
        raise OSError("test custody write failed during Panic")
process_custody.record_process = record
server_control.IS_WINDOWS = phase != "posix-exec"  # portable branches; no native Win32 claim
if phase == "posix-exec":
    # An unexpected exec destroys Panic's thread. Exit the disposable interpreter
    # with a distinct code to make that loss observable to the outer consumer.
    server_control.os.execvpe = lambda *a: os._exit(88)

transfer = server._restart_current_process_impl
def restart(*a, **kw):
    try:
        return transfer(*a, **kw)
    finally:
        ready.write_text("watcher reached final exit")
server._restart_current_process_impl = restart
def cleanup(**kw):
    if phase == "posix-exec":
        start_panic()
        ready.write_text("watcher reached POSIX transfer after accepted Panic")
    if phase == "early-cleanup":
        start_panic()
        ready.write_text("watcher reached cleanup failure")
        raise OSError("test cleanup failed as Panic began")

class DrainEvent(threading.Event):
    def wait(self, timeout=None):
        return super().wait(.02 if timeout == 30 else timeout)
class TestServer:
    def __init__(self, config):
        self.should_exit = False
    def watch_launcher_stop(self):
        pass
    def run(self, **kw):
        server._restart_requested.set()
        if not uvicorn_returns:
            threading.Event().wait(20)
            raise AssertionError("neither Panic nor watcher exited")
@contextmanager
def listener(*a, **kw):
    yield SimpleNamespace(getsockname=lambda: ("127.0.0.1", 9123))

state.init_state()
assert state.update_state(lambda st: st.update(evolution_mode_enabled=True, bg_consciousness_enabled=True),
                          confirm=("evolution_mode_enabled", "bg_consciousness_enabled")) is not False
server.threading = SimpleNamespace(Event=DrainEvent, Thread=threading.Thread)
server._restart_requested = threading.Event()
server._event_loop = None
server._LAUNCHER_MANAGED = False
server._planned_delegate_restart_transaction_id = ""
server.verify_settings_integrity = lambda: None
server.load_settings = lambda: {}
server.parse_server_args = lambda *a: SimpleNamespace(host="127.0.0.1", port=9123, host_explicit=False)
server.find_free_port = lambda host, port, **_: port
server.get_network_auth_startup_warning = lambda host: ""
server.validate_network_auth_configuration = lambda host: ""
server.bound_service_socket = listener
server.write_port_file = lambda *a: None
server.uvicorn.Config = lambda *a, **kw: None
server._SignalStopServer = TestServer
server._emergency_process_cleanup = cleanup
supervisor.update_merge.read_update_tx_strict = lambda: ("absent", {})
sys.exit(server.main())
''', encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["OUROBOROS_DATA_DIR"] = str(tmp_path / "data")
    env["OUROBOROS_SETTINGS_PATH"] = str(tmp_path / "data" / "settings.json")
    env.pop("OUROBOROS_PLANNED_RESTART_TRANSACTION_ID", None)
    # The parent waits for ready before reading output. A traceback during spawn
    # must not fill a pipe and block the restart watcher before it publishes ready.
    output_path = tmp_path / "watcher-output.log"
    with output_path.open("w", encoding="utf-8") as output_file, subprocess.Popen(
        [sys.executable, str(script), phase, "returned" if uvicorn_returns else "held", panic_entry,
         str(ready), str(release)], env=env, stdout=output_file, stderr=subprocess.STDOUT, text=True,
    ) as proc:
        try:
            # Includes a cold server import and durable state initialization.
            deadline = time.monotonic() + 45
            while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.01)
            assert ready.exists(), output_path.read_text(encoding="utf-8")
            # Panic is held before persistence. Neither explicit watcher exits nor
            # main's implicit exit 0 may terminate this interpreter in that window.
            assert not (tmp_path / "data/state/panic_stop.flag").exists()
            try:
                proc.wait(timeout=.15)
            except subprocess.TimeoutExpired:
                pass
            else:
                pytest.fail(f"watcher exited {proc.returncode} before Panic persistence:\n"
                            + output_path.read_text(encoding="utf-8"))
            release.write_text("continue", encoding="utf-8")
            proc.wait(timeout=10)
            assert proc.returncode == 99, output_path.read_text(encoding="utf-8")
        finally:
            release.touch()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
    assert (tmp_path / "data/state/panic_stop.flag").read_text(encoding="utf-8") == "panic"
    persisted = json.loads((tmp_path / "data/state/state.json").read_text(encoding="utf-8"))
    assert persisted["evolution_mode_enabled"] is False
    assert persisted["bg_consciousness_enabled"] is False
    assert persisted["evolution_owner_stopped"] is True


@pytest.mark.parametrize("managed", [True, False], ids=["launcher", "direct"])
@pytest.mark.parametrize("uvicorn_returns", [True, False], ids=["returned", "held"])
def test_restart_waits_for_cleanup_and_transfers_once(monkeypatch, tmp_path, managed, uvicorn_returns):
    import server
    from ouroboros import server_restart
    from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV
    from tests._work_stop_helpers import live_restart

    transaction = live_restart(tmp_path, monkeypatch)
    transaction_id = transaction["transaction_id"]
    monkeypatch.setattr(server_restart, "_LAST_RETURN_TX", {"id": transaction_id})

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
        assert server.os.environ[PLANNED_RESTART_TRANSACTION_ENV] == transaction_id
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
    monkeypatch.setattr(server, "_planned_delegate_restart_transaction_id", transaction_id)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", managed)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_event_loop", None)
    monkeypatch.setattr(server, "load_settings", lambda: {})
    monkeypatch.setattr(server, "verify_settings_integrity", lambda: None)
    monkeypatch.setattr(server, "parse_server_args", lambda *a: SimpleNamespace(host="127.0.0.1", port=9123, host_explicit=False))
    monkeypatch.setattr(server, "find_free_port", lambda host, port, **_: port)
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
    assert calls == [("cleanup", {}), ("cleanup_finished", None),
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
    server._emergency_process_cleanup()
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
    from ouroboros.config import DATA_DIR
    from ouroboros.delegate_recovery import (PLANNED_RESTART_TRANSACTION_ENV, _ack_direct_exec_successor,
                                             _read_restart_transaction, observe_restart_parent)
    if os.getpid() != int(os.environ["OURO_TEST_SUCCESSOR"]):
        # A spawned successor reports only after the test saw its caller exit:
        # one that dies with the caller leaves no receipt.
        caller_exited = receipt.with_name(receipt.name + ".caller-exited")
        deadline = time.monotonic() + 20
        while not caller_exited.exists():
            if time.monotonic() > deadline:
                sys.exit(3)
            time.sleep(0.05)
    token, parent = os.environ.get(PLANNED_RESTART_TRANSACTION_ENV), observe_restart_parent()
    _ack_direct_exec_successor(DATA_DIR)  # the successor's own proof of the prepared transaction
    proof = _read_restart_transaction(DATA_DIR, "physical-handoff")
    staged = receipt.with_name(receipt.name + ".tmp")
    staged.write_text(json.dumps({
        "pid": os.getpid(), "previous_pid": int(os.environ["OURO_TEST_SUCCESSOR"]),
        "transaction": token, "parent": parent,
        "proof": {key: proof.get(key) for key in ("status", "ack_source", "successor_pid")},
        "port": os.environ.get("OUROBOROS_SERVER_PORT"),
        "cleanup": os.environ.get("OURO_TEST_CLEANUP"),
        "argv": sys.argv[1:],
    }), encoding="utf-8")
    os.replace(staged, receipt)  # the test polls for the receipt: never show it half-written
    sys.exit(0)

import server
import supervisor.update_merge
from supervisor.restart_retention import prepare_restart_returns
# The real producer: a prepared row expecting 42 and its active stamp, as a planned restart writes them.
prepare_restart_returns(server.DATA_DIR, {}, [], transaction_id="physical-handoff")
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
    assert kwargs == {}
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
server.find_free_port = lambda host, port, **_: port
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
supervisor.update_merge.read_update_tx_strict = lambda: ("absent", {})
sys.exit(server.main())
''', encoding="utf-8")
    env = os.environ.copy()
    env.pop("OUROBOROS_SERVER_REEXEC_ARGV_JSON", None)
    env.pop("OURO_TEST_SUCCESSOR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["OUROBOROS_DATA_DIR"] = str(tmp_path / "data")  # this case's own transaction
    arguments = [str(receipt), failure, "returned" if uvicorn_returns else "held",
                 'argument with "quotes" and a trailing\\']
    stdout, stderr = tmp_path / "stdout.log", tmp_path / "stderr.log"
    # The real Windows handoff retains stdio. Pipe EOF therefore waits for the
    # successor, which itself waits for caller-exited below. Files let run() wait
    # only for the caller's physical exit, without changing the spawn boundary.
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, str(script), *arguments], env=env,
                                stdout=out, stderr=err, timeout=45)
    errors = stderr.read_text(encoding="utf-8", errors="replace")
    output = stdout.read_text(encoding="utf-8", errors="replace") + errors
    if failure in {"cleanup", "reexec"}:
        assert result.returncode == 1, output
        assert "Restart failed; cleanup or transfer is unconfirmed" in errors
        assert "fixture " + failure + " failure" in errors
        assert not receipt.exists()
        return
    spawned = os.name == "nt" or failure == "fallback"
    assert result.returncode == (42 if spawned else 0), output
    if spawned:
        receipt.with_name(receipt.name + ".caller-exited").write_text("", encoding="utf-8")
        deadline = time.monotonic() + 20
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert receipt.exists(), "the spawned successor did not outlive its caller: " + (
            stdout.read_text(encoding="utf-8", errors="replace") + stderr.read_text(encoding="utf-8", errors="replace"))
    observed = json.loads(receipt.read_text(encoding="utf-8"))
    assert observed["transaction"] == "physical-handoff"
    assert observed["port"] == "9123" and observed["cleanup"] == "completed"
    assert observed["argv"] == arguments
    assert (observed["pid"] == observed["previous_pid"]) is (not spawned)
    if os.name == "nt":  # native only: the inherited handle observed this exact caller's exit 42
        assert observed["parent"]["pid"] == observed["previous_pid"] and observed["parent"]["exit_code"] == 42
        assert observed["proof"] == {"status": "normal_exit_acknowledged",
                                     "ack_source": "windows_direct_parent_handle", "successor_pid": observed["pid"]}
    elif spawned:  # a POSIX fallback spawn proves itself by the exact custody receipt its caller bound
        assert observed["parent"] == {}
        assert observed["proof"] == {"status": "normal_exit_acknowledged",
                                     "ack_source": "direct_spawn_successor", "successor_pid": None}
    else:
        assert observed["proof"] == {"status": "normal_exit_acknowledged",
                                     "ack_source": "direct_exec_successor", "successor_pid": None}


@pytest.mark.parametrize("platform,failure", [
    ("nt", ""), ("nt", "spawn"),
    ("posix", ""), ("posix", "exec"), ("posix", "spawn"),
])
def test_direct_transfer_uses_platform_process_primitive(monkeypatch, tmp_path, platform, failure):
    """Windows must bypass CRT exec; POSIX keeps exec and its spawn fallback."""
    from ouroboros import config, platform_layer, server_control, process_custody

    class Transferred(BaseException):
        pass

    calls, captured, handles = [], {}, []
    startupinfo = object()

    @contextmanager
    def parent_handle():
        handles.append("opened")
        try:
            yield 77, {"startupinfo": startupinfo, "close_fds": True}
        finally:
            handles.append("closed")

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
    monkeypatch.setattr(platform_layer, "inheritable_self_handle", parent_handle)
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
    assert handles == (["opened", "closed"] if platform == "nt" else [])
    assert captured["argv"] == ["python", *argv]
    assert captured["env"] == {"OUROBOROS_SERVER_HOST": "127.0.0.1",
                               "S1_TOKEN": "same generation", "OUROBOROS_SERVER_PORT": "9123",
                               **({"OUROBOROS_PLANNED_RESTART_PARENT_HANDLE": "77"} if platform == "nt" else {})}
    if platform == "nt":
        assert captured["startupinfo"] is startupinfo and captured["close_fds"] is True
    if "spawn" in calls:
        # Windows keeps the console group (CTRL+C); POSIX isolates the fallback's session.
        assert captured["new_process_group"] is (platform != "nt")
        assert captured["scope"] == "daemon"
        assert captured["drive_root"] == tmp_path and captured["cwd"] == str(tmp_path)


def test_posix_transfer_observes_panic_accepted_during_its_last_log(monkeypatch, tmp_path):
    """Logging can yield after preparation; the Panic check belongs next to exec."""
    from ouroboros import config, process_custody, server_control

    monkeypatch.setattr(server_control, "_restart_stop_requested", False)
    monkeypatch.setattr(server_control, "IS_WINDOWS", False)
    monkeypatch.setattr(server_control, "os", SimpleNamespace(
        environ={}, execvpe=lambda *a: pytest.fail("accepted Panic must prevent exec")))
    monkeypatch.setattr(server_control, "sys", SimpleNamespace(executable="python", argv=["server.py"]))
    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(process_custody, "spawn_supervised",
                        lambda *a, **kw: pytest.fail("accepted Panic must not trigger a fallback"))

    def accept_panic(*a):
        server_control._restart_stop_requested = True

    log = SimpleNamespace(info=accept_panic, exception=lambda *a: pytest.fail("no transfer should be attempted"))
    server_control.restart_current_process("127.0.0.1", 9123, repo_dir=tmp_path, log=log)
    assert server_control._restart_stop_requested


@pytest.mark.parametrize("parent_exit", [42, 99, None], ids=["restart", "panic", "no-parent"])
def test_a_restart_successor_observes_its_parent_before_data_admission(tmp_path, parent_exit):
    """A successor's real main() against a real parent holding the data root's kernel lock.

    The parent outlives the successor's admission wait. Observing its inherited parent first,
    the successor neither times out nor writes the root while that parent lives: after exit 42
    it is admitted and logs, after Panic's 99 it stops unadmitted and without a server log.
    Without an inherited parent, ordinary contention is still refused within the bound. Only
    the Win32 handle wait is a portable double; its native behaviour is Windows CI's.
    """
    root, release, observing, exited = (tmp_path / name for name in ("data", "release", "observing", "exited"))
    holder_script, successor_script = tmp_path / "holder.py", tmp_path / "successor.py"
    holder_script.write_text('''
import os, pathlib, sys, time
from ouroboros.server_entrypoint import admit_data_root
admit_data_root(pathlib.Path(sys.argv[1]), wait_sec=0)
print("admitted", os.getpid(), flush=True)
while not pathlib.Path(sys.argv[2]).exists():
    time.sleep(0.02)
os._exit(int(sys.argv[3]))
''', encoding="utf-8")
    successor_script.write_text('''
import os, pathlib, sys, time
import server
from ouroboros import platform_layer, runtime_limits
observing, exited = map(pathlib.Path, sys.argv[1:3])
# Far shorter than the parent's remaining life; the production bound returns once it has exited.
production_wait, runtime_limits.SERVER_ADMISSION_WAIT_SEC = runtime_limits.SERVER_ADMISSION_WAIT_SEC, 0.2
print("successor", os.getpid(), flush=True)

def await_process_handle(handle):  # the inherited-handle wait and exit read, without Win32
    observing.write_text(str(handle), encoding="utf-8")
    while not exited.exists():
        time.sleep(0.02)
    runtime_limits.SERVER_ADMISSION_WAIT_SEC = production_wait  # Windows may release a dead holder's lock late
    return {"pid": handle, "birth": "fixture", "exit_code": int(exited.read_text(encoding="utf-8"))}

def serving_checks(*_args):
    print("serving checks reached", flush=True)
    return False  # main() ends here, after admission and logging

platform_layer.await_process_handle = await_process_handle
server.automatic_launch_allowed = serving_checks
sys.exit(server.main())
''', encoding="utf-8")
    env = os.environ.copy()
    for key in ("OUROBOROS_PLANNED_RESTART_TRANSACTION_ID", "OUROBOROS_PLANNED_RESTART_PARENT_HANDLE",
                "OUROBOROS_PLANNED_RESTART_BINDING_HANDLE"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]), OUROBOROS_DATA_DIR=str(root),
               OUROBOROS_SETTINGS_PATH=str(root / "settings.json"))
    holder = subprocess.Popen([sys.executable, str(holder_script), str(root), str(release), str(parent_exit or 0)],
                              env=env, stdout=subprocess.PIPE, text=True)
    successor = None
    try:
        holder_line = holder.stdout.readline().split()
        assert holder_line[:1] == ["admitted"], holder_line
        if parent_exit is not None:
            env["OUROBOROS_PLANNED_RESTART_PARENT_HANDLE"] = str(holder.pid)
        successor = subprocess.Popen([sys.executable, str(successor_script), str(observing), str(exited)],
                                     env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if parent_exit is not None:
            deadline = time.monotonic() + 60
            while not observing.exists():  # a successor admitted first is refused before it gets here
                assert successor.poll() is None, successor.communicate(timeout=10)[0]
                assert time.monotonic() < deadline, "the successor never awaited its parent"
                time.sleep(0.02)
            time.sleep(0.6)  # three admission waits while the parent still holds the root
            assert successor.poll() is None and holder.poll() is None
            release.touch()
            assert holder.wait(timeout=10) == parent_exit
            exited.with_suffix(".tmp").write_text(str(parent_exit), encoding="utf-8")
            os.replace(exited.with_suffix(".tmp"), exited)
        output, _ = successor.communicate(timeout=60)
    finally:
        release.touch()
        for proc in (successor, holder):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=10)
        holder.stdout.close()
    stamp = (root / "state" / "server.lock").read_text(encoding="ascii").strip()
    if parent_exit is None:
        assert successor.returncode == 2 and "another Ouroboros server serves this data directory" in output, output
        assert stamp == f"pid={holder_line[1]}"
    elif parent_exit == 99:
        assert successor.returncode == 99 and "exited by Panic" in output, output
        assert "serving checks reached" not in output
        assert stamp == f"pid={holder_line[1]}" and not (root / "logs" / "server.log").exists()
    else:
        assert successor.returncode == 0 and "serving checks reached" in output, output
        successor_pids = [line.split()[1] for line in output.splitlines() if line.startswith("successor ")]
        assert [f"pid={pid}" for pid in successor_pids] == [stamp] and (root / "logs" / "server.log").exists()
