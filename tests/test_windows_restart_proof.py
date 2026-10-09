"""A Windows direct planned restart: the successor proves its exact parent's exit 42.

The portable tests drive the transaction policy through its real consumers, the
launch wiring and Panic's publication race with fake Win32 primitives; they
prove no native handle behaviour. test_windows_restart_native runs the real
primitives and is skipped off Windows.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import delegate_recovery as recovery
from ouroboros import platform_layer, server_control
from tests.test_owner_wait_restart import windows_successor

HANDLE_SOURCE = "windows_direct_parent_handle"


class Gateway:
    def get_run(self, run_id):
        assert run_id == "run-1"
        return {"id": run_id, "state": "running"}

    def close(self):
        pass


@pytest.fixture
def planned_session(tmp_path, monkeypatch):
    """A sleeping configured session the old generation prepared; this process then succeeds it."""
    import ouroboros.claudexor_daemon as daemon
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_work_order import work_order_fingerprint
    from ouroboros.utils import atomic_write_json
    from tests.test_available_subagents_runtime import _session_row, _settings, _snapshot

    custody._CUSTODY.clear()
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task = {"id": "child1", "_attempt": 1, "configured_subagent": snapshot, "drive_root": str(tmp_path),
            "task_constraint": {}, "task_contract": {"objective": "Build", "expected_output": "Patch"}}
    custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="child1", route_id="codex", selected_subagent_id="session-builder",
        config_fingerprint=snapshot["config_fingerprint"],
        authority_fingerprint=recovery.authority_fingerprint_from_task(task),
        work_order_fingerprint=work_order_fingerprint(task),
    ))
    (tmp_path / "state" / "delegate_supervision").mkdir(parents=True)
    atomic_write_json(tmp_path / "state" / "delegate_supervision" / "child1.json",
                      {"schema": 1, "run_id": "run-1", "status": "sleeping", "journal_cursor": 7})
    assert recovery.prepare_planned_restart_handoffs(
        tmp_path, {"child1": {"task": task, "attempt": 1, "worker_id": 3}}) == {"child1"}
    handoff = recovery._read(tmp_path, "child1")
    parent_pid = int(handoff["supervisor_pid"])
    monkeypatch.setattr(recovery.os, "getpid", lambda: parent_pid + 100_000)  # the spawned successor
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    monkeypatch.delenv(recovery.PLANNED_RESTART_TRANSACTION_ENV, raising=False)
    return SimpleNamespace(root=tmp_path, task=task, successor_task={**task, "_attempt": 2}, parent_pid=parent_pid,
                           transaction_id=handoff["restart_transaction_id"])


def windows_successor_of(case, monkeypatch, parent=None, binding=None):
    monkeypatch.setattr(recovery, "IS_WINDOWS", True)
    monkeypatch.setenv(recovery.PLANNED_RESTART_TRANSACTION_ENV, case.transaction_id)
    windows_successor(monkeypatch, case.root, case.transaction_id, parent, binding)


def test_pre_adoption_accepts_the_proved_parent_although_its_number_now_names_another_process(
    planned_session, monkeypatch,
):
    case = planned_session
    windows_successor_of(case, monkeypatch)
    probed = []
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: probed.append(pid) or True)
    assert recovery.pre_adopt_planned_handoffs(case.root, [case.successor_task]) == {"child1"}
    transaction = recovery._read_restart_transaction(case.root, case.transaction_id)
    assert transaction["status"] == "normal_exit_acknowledged" and transaction["ack_source"] == HANDLE_SOURCE
    assert transaction["supervisor_pid"] == case.parent_pid  # the old generation keeps its own number
    assert transaction["successor_pid"] == case.parent_pid + 100_000
    assert probed == []  # the handle already observed that exact exit; a reused number proves nothing
    assert recovery._read(case.root, "child1")["status"] == "pre_adopted"


def test_worker_adoption_keeps_its_checks_and_never_requires_the_successor_pid(planned_session, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.tools.registry import ToolContext

    case = planned_session
    windows_successor_of(case, monkeypatch)
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: False)
    assert recovery.pre_adopt_planned_handoffs(case.root, [case.successor_task]) == {"child1"}
    monkeypatch.setattr(recovery.os, "getpid", lambda: case.parent_pid + 200_000)  # a worker process
    monkeypatch.setattr(delegate, "exact_start", lambda *_a, **_k: pytest.fail("a planned handoff adopts, never POSTs"))
    ctx = ToolContext(repo_dir=case.root, drive_root=case.root, task_id="child1",
                      task_metadata={"drive_root": str(case.root)}, task_constraint=normalize_task_constraint({}),
                      task_contract=case.task["task_contract"])
    assert recovery.adopt_handoff(ctx, case.successor_task)["status"] == "adopted"
    stale = {**case.successor_task, "_attempt": 3}
    recovery._write(case.root, {**recovery._read(case.root, "child1"), "status": "pre_adopted"})
    assert recovery.adopt_handoff(ctx, stale)["status"] == "recovery_required"  # attempt checks still apply


@pytest.mark.parametrize("fault", [
    "token_only", "observation_failed", "exit_1", "panic_99", "other_parent_pid", "other_parent_birth",
    "binding_failed", "other_successor_pid", "other_successor_birth", "transaction_replaced",
])
def test_pre_adoption_vetoes_every_unproven_windows_successor(planned_session, monkeypatch, caplog, fault):
    case = planned_session
    parent = {"pid": case.parent_pid, "birth": "win-filetime:1", "exit_code": 42}
    successor = case.parent_pid + 100_000
    binding = {"supervisor_birth": "win-filetime:1", "successor_pid": successor, "successor_birth": "win-filetime:2"}
    if fault == "observation_failed":
        parent = OSError("the process handle is invalid")
    elif fault in {"exit_1", "panic_99"}:
        parent["exit_code"] = 1 if fault == "exit_1" else 99
    elif fault == "other_parent_pid":
        parent["pid"] += 1
    elif fault == "other_parent_birth":
        parent["birth"] = "win-filetime:9"
    elif fault == "binding_failed":
        binding = {}
    elif fault == "other_successor_pid":
        binding["successor_pid"] = successor + 1
    elif fault == "other_successor_birth":
        binding["successor_birth"] = "win-filetime:9"
    elif fault == "transaction_replaced":
        binding = {**binding, "status": "normal_exit_acknowledged", "exit_code": 42, "ack_source": "launcher_waitpid"}
    windows_successor_of(case, monkeypatch, parent, binding)
    if fault == "token_only":
        monkeypatch.delenv(recovery.PLANNED_RESTART_PARENT_ENV)
    if fault == "transaction_replaced":  # a reader that is not the bound successor never re-acknowledges
        monkeypatch.setattr(recovery, "_pid_alive", lambda pid: True)
    assert recovery.pre_adopt_planned_handoffs(case.root, [case.successor_task]) == set()
    expected = "previous_supervisor_generation_not_dead" if fault == "transaction_replaced" else (
        "restart_normal_exit_unproven")
    assert recovery._read(case.root, "child1")["veto_reason"] == expected
    assert "continuation unproven" in caplog.text


def test_same_pid_token_still_proves_a_posix_exec_but_never_a_windows_spawn(planned_session, monkeypatch):
    case = planned_session
    monkeypatch.setattr(recovery.os, "getpid", lambda: case.parent_pid)  # exec keeps the number
    monkeypatch.setattr(recovery, "_restart_parent", None)
    for windows, status in ((True, "prepared"), (False, "normal_exit_acknowledged")):
        monkeypatch.setattr(recovery, "IS_WINDOWS", windows)
        monkeypatch.setenv(recovery.PLANNED_RESTART_TRANSACTION_ENV, case.transaction_id)
        recovery._ack_direct_exec_successor(case.root)
        assert recovery._read_restart_transaction(case.root, case.transaction_id)["status"] == status
    assert recovery._read_restart_transaction(case.root, case.transaction_id)["ack_source"] == "direct_exec_successor"


def test_a_launcher_acknowledged_generation_keeps_its_liveness_check(planned_session, monkeypatch):
    case = planned_session
    assert recovery.acknowledge_observed_restart_exit(case.root, supervisor_pid=case.parent_pid, exit_code=42)
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: pid == case.parent_pid)
    assert recovery.pre_adopt_planned_handoffs(case.root, [case.successor_task]) == set()
    assert recovery._read(case.root, "child1")["veto_reason"] == "previous_supervisor_generation_not_dead"


def test_the_parent_is_observed_once_and_its_handle_never_reaches_children(monkeypatch, caplog):
    waits = []
    monkeypatch.setattr(recovery, "_restart_parent", None)
    monkeypatch.setenv(recovery.PLANNED_RESTART_PARENT_ENV, "77")
    monkeypatch.setattr(platform_layer, "await_process_handle",
                        lambda handle: waits.append(handle) or {"pid": 5, "birth": "b", "exit_code": 42})
    assert recovery.observe_restart_parent() == {"pid": 5, "birth": "b", "exit_code": 42}
    assert recovery.PLANNED_RESTART_PARENT_ENV not in os.environ
    assert recovery.observe_restart_parent()["exit_code"] == 42 and waits == [77]
    monkeypatch.setattr(recovery, "_restart_parent", None)
    monkeypatch.setenv(recovery.PLANNED_RESTART_PARENT_ENV, "not a handle")
    assert recovery.observe_restart_parent() == {}
    assert "restart parent handle could not be observed" in caplog.text


def test_binding_names_the_live_parent_and_its_spawned_successor(tmp_path, monkeypatch):
    recovery._write_restart_transaction(tmp_path, {"transaction_id": "tx", "status": "prepared",
                                                   "supervisor_pid": os.getpid(), "task_ids": ["a"]})
    births = {os.getpid(): "win-filetime:1", 4242: "win-filetime:2"}
    monkeypatch.setattr(platform_layer, "process_start_time", lambda pid: births.get(pid, ""))
    recovery.bind_restart_successor(tmp_path, "tx", 4242)
    row = recovery._read_restart_transaction(tmp_path, "tx")
    assert (row["status"], row["supervisor_birth"], row["successor_pid"], row["successor_birth"]) == (
        "prepared", "win-filetime:1", 4242, "win-filetime:2")
    with pytest.raises(ValueError):  # a successor already gone has no birth to bind
        recovery.bind_restart_successor(tmp_path, "tx", 4343)
    with pytest.raises(ValueError):
        recovery.bind_restart_successor(tmp_path, "absent", 4242)


# --- the parent's launch wiring -------------------------------------------------------------

class FakeHandle(int):
    def __new__(cls, value, calls):
        handle = super().__new__(cls, value)
        handle.calls = calls
        return handle

    def Close(self):
        self.calls.append("close")


def windows_transfer(monkeypatch, tmp_path, *, token="tx-1", base=r"C:\Python\python.exe", handle_error=None,
                     spawn_error=None, bind_error=None):
    """Run restart_current_process as a Windows direct parent with fake handle, spawn and binding."""
    from ouroboros import config, process_custody

    calls, captured, logged = [], {}, []
    environment = {"OUROBOROS_SERVER_HOST": "127.0.0.1",
                   **({recovery.PLANNED_RESTART_TRANSACTION_ENV: token} if token else {})}

    @contextlib.contextmanager
    def inheritable():
        if handle_error:
            raise handle_error
        calls.append("open")
        handle = FakeHandle(77, calls)
        try:
            yield handle, {"startupinfo": "only-77", "close_fds": True}
        finally:
            handle.Close()

    def spawn(argv, **kwargs):
        calls.append("spawn")
        captured.update(argv=list(argv), **kwargs)
        if spawn_error:
            raise spawn_error
        return SimpleNamespace(pid=4242, terminate=lambda: calls.append("terminate"),
                               kill=lambda: calls.append("kill"))

    def bind(root, transaction_id, pid):
        calls.append(("bind", transaction_id, pid))
        if bind_error:
            raise bind_error

    monkeypatch.setattr(server_control, "IS_WINDOWS", True)
    monkeypatch.setattr(server_control, "os", SimpleNamespace(
        name="nt", environ=environment, execvpe=lambda *a: pytest.fail("Windows never enters CRT exec")))
    monkeypatch.setattr(server_control, "sys", SimpleNamespace(
        executable=r"C:\venv\Scripts\python.exe", _base_executable=base, argv=["server.py"]))
    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(platform_layer, "inheritable_self_handle", inheritable)
    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    monkeypatch.setattr(recovery, "bind_restart_successor", bind)
    log = SimpleNamespace(info=lambda *a: None, exception=lambda message, *a: logged.append(message % a))
    error = None
    try:
        server_control.restart_current_process("127.0.0.1", 9123, repo_dir=tmp_path, log=log)
    except Exception as exc:
        error = exc
    return SimpleNamespace(calls=calls, captured=captured, logged=logged, error=error)


def test_planned_windows_spawn_passes_only_the_parent_handle_and_binds_the_real_successor(monkeypatch, tmp_path):
    run = windows_transfer(monkeypatch, tmp_path)
    assert run.error is None
    assert run.calls == ["open", "spawn", "close", ("bind", "tx-1", 4242)]  # closed before the parent exits
    captured = run.captured
    assert captured["argv"] == [r"C:\Python\python.exe", "server.py"]  # the base interpreter is the bound PID
    assert captured["env"]["__PYVENV_LAUNCHER__"] == r"C:\venv\Scripts\python.exe"  # ...running the venv
    assert captured["env"][recovery.PLANNED_RESTART_PARENT_ENV] == "77"
    assert (captured["startupinfo"], captured["close_fds"]) == ("only-77", True)
    assert not {"stdin", "stdout", "stderr"}.intersection(captured)  # STARTUPINFO owns all three
    assert captured["new_process_group"] is False  # the same console group, as exec would keep
    assert captured["on_spawn"] is server_control._hold_restart_successor
    assert captured["scope"] == "daemon"


def test_a_plain_interpreter_needs_no_venv_redirect(monkeypatch, tmp_path):
    run = windows_transfer(monkeypatch, tmp_path, base=r"C:\venv\Scripts\python.exe")
    assert run.error is None
    assert run.captured["argv"][0] == r"C:\venv\Scripts\python.exe"
    assert "__PYVENV_LAUNCHER__" not in run.captured["env"]


def test_a_failed_binding_keeps_the_custodied_successor_serving_and_says_so(monkeypatch, tmp_path):
    run = windows_transfer(monkeypatch, tmp_path, bind_error=OSError("disk full"))
    assert run.error is None  # the watcher still exits 42; the successor keeps the install available
    assert run.calls == ["open", "spawn", "close", ("bind", "tx-1", 4242)]
    assert "terminate" not in run.calls and "kill" not in run.calls
    assert run.logged == ["Restart binding for successor 4242 failed: it is not stopped, "
                          "but prepared handoffs stay unresumed"]


def test_a_failed_spawn_or_custody_record_still_fails_the_transfer(monkeypatch, tmp_path):
    run = windows_transfer(monkeypatch, tmp_path,
                           spawn_error=RuntimeError("spawned process could not enter durable custody"))
    assert isinstance(run.error, RuntimeError)  # the watcher exits 1; spawn_supervised already killed the child
    assert run.calls == ["open", "spawn", "close"]  # never bound, the handle still closed
    assert run.logged == ["Spawned restart fallback failed; no successor was started."]


def test_an_unavailable_parent_handle_spawns_the_successor_without_a_proof(monkeypatch, tmp_path):
    run = windows_transfer(monkeypatch, tmp_path, handle_error=OSError("access denied"))
    assert run.error is None and run.calls == ["spawn"]
    assert "startupinfo" not in run.captured and run.captured["argv"][0] == r"C:\Python\python.exe"
    assert run.captured["env"]["__PYVENV_LAUNCHER__"] == r"C:\venv\Scripts\python.exe"
    assert recovery.PLANNED_RESTART_PARENT_ENV not in run.captured["env"]
    assert run.logged == ["Restart parent handle unavailable; the successor cannot prove its continuation"]


def test_a_restart_without_a_transaction_still_holds_the_venv_server_for_panic(monkeypatch, tmp_path, stop_requests):
    run = windows_transfer(monkeypatch, tmp_path, token="")
    assert run.error is None and run.calls == ["spawn"]
    assert "startupinfo" not in run.captured and run.captured["argv"][0] == r"C:\Python\python.exe"
    assert run.captured["env"]["__PYVENV_LAUNCHER__"] == r"C:\venv\Scripts\python.exe"
    assert recovery.PLANNED_RESTART_PARENT_ENV not in run.captured["env"]
    run.captured["on_spawn"](SimpleNamespace(pid=4242))
    assert server_control.stop_restart_successor() == [{"pid": 4242, "requested": True}]
    assert stop_requests == [4242]


# --- Panic and the successor --------------------------------------------------------------

@pytest.fixture
def stop_requests(monkeypatch):
    requests = []
    monkeypatch.setattr(server_control, "_restart_successors", [])
    monkeypatch.setattr(server_control, "_restart_stop_requested", False)
    monkeypatch.setattr(platform_layer, "request_process_tree_kill",
                        lambda proc: requests.append(proc.pid) or {"pid": proc.pid, "requested": True})
    return requests


def test_panic_after_publication_stops_the_held_successor(stop_requests):
    server_control._hold_restart_successor(SimpleNamespace(pid=4242))
    assert server_control.stop_restart_successor() == [{"pid": 4242, "requested": True}]
    assert stop_requests == [4242]


def test_panic_before_publication_stops_the_successor_as_it_is_published(stop_requests):
    assert server_control.stop_restart_successor() == []
    server_control._hold_restart_successor(SimpleNamespace(pid=4242))
    assert stop_requests == [4242]


def test_panic_between_publication_and_its_check_still_stops_the_successor(stop_requests, monkeypatch):
    class PanicLandsHere(list):
        def append(self, proc):
            super().append(proc)
            server_control.stop_restart_successor()

    monkeypatch.setattr(server_control, "_restart_successors", PanicLandsHere())
    server_control._hold_restart_successor(SimpleNamespace(pid=4242))
    assert stop_requests == [4242, 4242]  # a duplicate request is harmless; a missed one is not


@pytest.fixture
def real_restart_spawn(monkeypatch, tmp_path):
    """A sleeping real child through the production spawn fallback, always reaped."""
    from ouroboros import config

    monkeypatch.setattr(server_control, "IS_WINDOWS", False)
    monkeypatch.setattr(server_control.os, "execvpe", lambda *a: (_ for _ in ()).throw(OSError("no exec")))
    monkeypatch.setenv("OUROBOROS_SERVER_REEXEC_ARGV_JSON", json.dumps(["-c", "import time; time.sleep(30)"]))
    monkeypatch.delenv(recovery.PLANNED_RESTART_TRANSACTION_ENV, raising=False)
    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    log = SimpleNamespace(info=lambda *a: None, exception=lambda *a: None)
    try:
        yield lambda: server_control.restart_current_process("127.0.0.1", 9123, repo_dir=tmp_path, log=log)
    finally:
        server_control.stop_restart_successor()
        for successor in server_control._restart_successors:
            successor.wait(timeout=10)


@pytest.mark.serial
@pytest.mark.parametrize("phase", ["before", "during", "after"])
def test_panic_around_real_spawn_publication_leaves_no_successor_running(real_restart_spawn, monkeypatch, phase):
    if phase == "before":
        assert server_control.stop_restart_successor() == []
    elif phase == "during":
        class PanicLandsHere(list):
            def append(self, proc):
                super().append(proc)
                server_control.stop_restart_successor()

        monkeypatch.setattr(server_control, "_restart_successors", PanicLandsHere())
    real_restart_spawn()
    if phase == "after":
        [request] = server_control.stop_restart_successor()
        assert request["requested"] is True
    [successor] = server_control._restart_successors
    assert successor.wait(timeout=10) is not None


@pytest.mark.serial
def test_panic_stops_a_real_successor_while_its_custody_write_is_blocked(real_restart_spawn, monkeypatch):
    from ouroboros import process_custody

    entered, release = threading.Event(), threading.Event()
    failures = []

    def blocked_record(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        raise OSError("custody write failed after the stop")

    def spawn():
        try:
            real_restart_spawn()
        except Exception as exc:
            failures.append(exc)

    monkeypatch.setattr(process_custody, "record_process", blocked_record)
    thread = threading.Thread(target=spawn)
    thread.start()
    try:
        assert entered.wait(5)
        [successor] = server_control._restart_successors
        [request] = server_control.stop_restart_successor()
        assert request["requested"] is True
        assert successor.wait(timeout=5) is not None
        assert thread.is_alive()  # stop neither joins the spawn nor waits for its I/O
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert len(failures) == 1
    with pytest.raises(RuntimeError, match="durable custody"):
        raise failures[0]


@pytest.mark.parametrize("exit_code", [99, 42, 1, None])
def test_a_successor_that_observes_a_panic_exit_stops_before_it_serves(monkeypatch, exit_code):
    import server

    class Serving(Exception):
        pass

    observed = {} if exit_code is None else {"pid": 1, "birth": "b", "exit_code": exit_code}
    monkeypatch.setattr(recovery, "observe_restart_parent", lambda: observed)
    monkeypatch.setattr(server, "configure_process_logging", lambda **_kw: None)
    monkeypatch.setattr(server, "automatic_launch_allowed", lambda *_a: (_ for _ in ()).throw(Serving()))
    if exit_code == 99:
        assert server.main() == server.PANIC_EXIT_CODE
    else:  # any other exit, or no inherited parent, boots as before (continuation alone stays unproven)
        with pytest.raises(Serving):
            server.main()


# --- the platform primitives with fake Win32 ----------------------------------------------

@pytest.mark.parametrize("wait_fails", [False, True])
def test_awaiting_the_parent_reads_its_identity_and_always_closes_the_handle(monkeypatch, wait_fails):
    calls = []

    def wait(handle, timeout):
        calls.append(("wait", handle, timeout))
        if wait_fails:
            raise OSError("WAIT_FAILED")
        return 0

    monkeypatch.setitem(sys.modules, "_winapi", SimpleNamespace(
        INFINITE=0xFFFFFFFF, WAIT_OBJECT_0=0, WaitForSingleObject=wait, GetExitCodeProcess=lambda handle: 42,
        CloseHandle=lambda handle: calls.append(("close", handle))))
    monkeypatch.setattr(platform_layer, "_kernel32", SimpleNamespace(GetProcessId=lambda handle: 4242), raising=False)
    monkeypatch.setattr(platform_layer, "_windows_process_start_time",
                        lambda pid=0, handle=None: f"win-filetime:{handle}")
    if wait_fails:
        with pytest.raises(OSError):
            platform_layer.await_process_handle(77)
    else:
        assert platform_layer.await_process_handle(77) == {"pid": 4242, "birth": "win-filetime:77", "exit_code": 42}
    assert calls == [("wait", 77, 0xFFFFFFFF), ("close", 77)]
