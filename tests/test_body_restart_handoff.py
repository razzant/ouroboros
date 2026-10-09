"""The real captured cold-bootstrap helper transfers a Windows restart proof.

The package hook and captured helper source run unchanged. Git and Win32 calls
are doubled; the final process uses the real delegate-recovery consumer. This
is portable bootstrap evidence, not native Windows handle-inheritance evidence.
"""
from __future__ import annotations

import builtins
import ctypes
import ctypes.wintypes
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import delegate_recovery as recovery, platform_layer


class _BootstrapExit(BaseException):
    def __init__(self, code):
        self.code = code


def _cold_boot(tmp_path, monkeypatch, *, fault="", absent_streams=False, with_transaction=True):
    """Run __init__ -> captured main -> switch -> handover -> final recovery.

    Three identities are deliberately distinct: the original server (101), the
    first successor/cold helper (202), and the final serving process (303).
    """
    repo = Path(__file__).resolve().parents[1]
    serving, helper, data = (tmp_path / name for name in ("serving", "captured", "data"))
    (serving / "ouroboros").mkdir(parents=True)
    (serving / ".git").mkdir()
    helper.mkdir()
    hook = serving / "ouroboros/__init__.py"
    hook.write_bytes((repo / "ouroboros/__init__.py").read_bytes())
    captured = helper / "switch.py"
    captured.write_bytes((repo / "ouroboros/body_switch.py").read_bytes())
    (serving / ".git/ouroboros-body-adoption").write_text(str(helper), encoding="utf-8")
    old, new = "a" * 40, "b" * 40
    (helper / "handoff.json").write_text(json.dumps({
        "id": "cold-test", "phase": "armed", "repo_dir": str(serving),
        "data_dir": str(data), "old": old, "cand": new, "branch": "ouroboros", "switch": [],
    }), encoding="utf-8")
    transaction = {
        "transaction_id": "cold-restart", "status": "prepared", "supervisor_pid": 101,
        "supervisor_birth": "win-filetime:1", "successor_pid": 202,
        "successor_birth": "win-filetime:2", "task_ids": [],
    }
    if fault == "helper_birth":
        transaction["successor_birth"] = "win-filetime:999"
    if with_transaction:
        recovery._write_restart_transaction(data, transaction)
    env = {recovery.PLANNED_RESTART_PARENT_ENV: "77"}
    if with_transaction:
        env[recovery.PLANNED_RESTART_TRANSACTION_ENV] = "cold-restart"
    calls, launches, final, trace = [], [], {}, []
    state = SimpleNamespace(head=old, inherited=set())
    helper_thread = threading.get_ident()
    gate_entered, writer_closed, child_done = (threading.Event() for _ in range(3))
    children, child_errors = [], []
    identities = {77: (101, 1), -1: (202, 2), 500: (303, 3)}
    closed = set()
    parent_exit = 99 if fault == "panic" else 42

    def close(handle):
        closed.add(int(handle))
        calls.append(("close", int(handle)))
        if int(handle) == 89:
            writer_closed.set()

    class Handle(int):
        def Close(self):
            close(self)

    class StartupInfo(SimpleNamespace):
        def copy(self):
            return StartupInfo(**{**vars(self), "lpAttributeList": {
                "handle_list": list(self.lpAttributeList["handle_list"])}})

    def duplicate(current, raw, target, access, inherit, options):
        assert current == target == -1 and inherit is True
        result = int(raw) + 100
        if int(raw) in identities:
            identities[result] = identities[int(raw)]
        calls.append(("duplicate", int(raw), result))
        return result

    def wait(handle, _timeout):
        phase = "helper" if threading.get_ident() == helper_thread else "final"
        if phase == "final" and int(handle) not in state.inherited:
            raise OSError("the original parent HANDLE was not inherited by the final process")
        if phase == "helper":
            assert gate_entered.wait(5), "final recovery never reached the binding barrier"
            assert not child_done.is_set() and not final, "final recovery outran helper binding"
            if fault == "helper_death":
                # Model the owner's process disappearing before its binding
                # write. Its sole pipe writer closes; EOF still proves no bind.
                raise _BootstrapExit(137)
        calls.append(("wait", phase, int(handle)))
        return 0

    def read(handle, _size):
        assert int(handle) in state.inherited, "the binding barrier itself must be inherited"
        calls.append(("gate-enter", int(handle)))
        gate_entered.set()
        if not writer_closed.wait(5):
            pytest.fail("final recovery remained blocked after the helper's binding attempt")
        assert 89 in closed
        calls.append(("gate", int(handle)))
        raise BrokenPipeError("binding writer closed")

    api = SimpleNamespace(
        GetCurrentProcess=lambda: -1, DuplicateHandle=duplicate, DUPLICATE_SAME_ACCESS=2,
        CloseHandle=close, CreatePipe=lambda *_a: (88, 89), ReadFile=read,
        WaitForSingleObject=wait, WAIT_OBJECT_0=0, INFINITE=0xFFFFFFFF,
        GetExitCodeProcess=lambda handle: parent_exit,
        STARTF_USESTDHANDLES=0x100, FILE_TYPE_CHAR=2, GetFileType=lambda handle: 1,
    )

    def process_times(handle, created, *_rest):
        created._obj.dwLowDateTime = identities[int(handle)][1]
        created._obj.dwHighDateTime = 0
        return 1

    def process_id(handle):
        return identities[int(handle)][0]

    kernel = SimpleNamespace(GetProcessId=process_id, GetProcessTimes=process_times)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_kw: kernel, raising=False)
    monkeypatch.setattr(platform_layer, "ctypes", ctypes, raising=False)
    monkeypatch.setattr(platform_layer, "_kernel32", kernel, raising=False)
    monkeypatch.setitem(sys.modules, "_winapi", api)
    crt = SimpleNamespace(get_osfhandle=lambda fd: fd + 10, locking=lambda *_a: None, LK_NBLCK=1)
    monkeypatch.setitem(sys.modules, "msvcrt", crt)

    def git(command, **_kwargs):
        assert command[0] == "git"
        args = command[1:]
        if args[:2] == ["rev-parse", "--verify"]:
            out = state.head
        elif args[:2] == ["rev-parse", "--absolute-git-dir"]:
            out = str(serving / ".git")
        elif args[0] == "symbolic-ref":
            out = "ouroboros"
        elif args[0] == "config":
            out = "true"
        elif args[0] == "update-ref":
            state.head, out = new, ""
        else:
            assert args[0] in {"cat-file", "status"}, args
            out = ""
        return SimpleNamespace(returncode=0, stdout=(out + "\n").encode(), stderr=b"")

    def final_main(child_env):
        try:
            parent = recovery.observe_restart_parent()  # server.main's first restart consumer
            if parent.get("exit_code") != 99:
                recovery._ack_direct_exec_successor(data)
            final.update(parent=parent, env=child_env,
                         transaction=recovery._read_restart_transaction(data, "cold-restart"))
        except BaseException as exc:
            child_errors.append(exc)
        finally:
            child_done.set()

    def final_wait():
        children[0].join(5)
        assert not children[0].is_alive(), "test-owned final recovery thread did not finish"
        if child_errors:
            raise child_errors[0]
        return 99 if final["parent"].get("exit_code") == 99 else 0

    def popen(argv, **kwargs):
        launches.append((list(argv), kwargs))
        startup = kwargs.get("startupinfo")
        state.inherited = set(startup.lpAttributeList["handle_list"]) if startup else set()
        child_env = dict(kwargs.get("env", env))
        monkeypatch.setattr(recovery, "os", SimpleNamespace(environ=child_env, getpid=lambda: 303))
        monkeypatch.setattr(recovery, "IS_WINDOWS", True)
        monkeypatch.setattr(recovery, "_restart_parent", None)
        monkeypatch.setattr(platform_layer, "process_start_time", lambda pid: "win-filetime:3" if pid == 303 else "")
        child = threading.Thread(target=final_main, args=(child_env,), name="test-final-restart-consumer")
        children.append(child)
        child.start()  # the real consumer can run BEFORE Popen returns to the helper
        return SimpleNamespace(pid=303, _handle=500, wait=final_wait)

    proxy_os = SimpleNamespace(**vars(os))
    proxy_os.environ, proxy_os.getpid = env, lambda: 202
    proxy_os._exit = lambda code: (_ for _ in ()).throw(_BootstrapExit(code))
    proxy_os.pipe = lambda: (88, 89)
    proxy_os.close = close
    proxy_os.set_inheritable = lambda *_a: None
    proxy_os.set_handle_inheritable = lambda *_a: None
    replace = os.replace

    def replace_metadata(source, destination):
        if Path(destination).name == "cold-restart.json":
            assert gate_entered.is_set() and not child_done.is_set()
            assert not final and not writer_closed.is_set(), "the metadata write must still fence final recovery"
            calls.append(("binding-write",))
            if fault == "metadata":
                calls.append(("binding-write-failed",))
                raise OSError("disk full for restart metadata")
        return replace(source, destination)

    proxy_os.replace = replace_metadata
    proxy_sys = SimpleNamespace(**vars(sys))
    proxy_sys.platform = "win32"
    proxy_sys.executable = r"C:\venv with spaces\Scripts\python.exe"
    proxy_sys._base_executable = r"C:\Python\python.exe"
    proxy_sys.orig_argv = [proxy_sys.executable, "server.py", "space argument", ""]
    proxy_sys.stdout, proxy_sys.stderr = io.StringIO(), io.StringIO()
    for fd, name in enumerate(("stdin", "stdout", "stderr")):
        setattr(proxy_sys, "__" + name + "__", SimpleNamespace(fileno=lambda fd=fd: fd))
    if absent_streams:
        proxy_sys.stderr = proxy_sys.__stdin__ = proxy_sys.__stderr__ = None
    proxy_subprocess = SimpleNamespace(**vars(subprocess))
    proxy_subprocess.run, proxy_subprocess.Popen = git, popen
    proxy_subprocess.Handle, proxy_subprocess.STARTUPINFO = Handle, StartupInfo
    proxy_subprocess.STARTF_USESTDHANDLES = 0x100
    imports = {"os": proxy_os, "sys": proxy_sys, "subprocess": proxy_subprocess,
               "_winapi": api, "msvcrt": crt}

    def import_module(name, *args, **kwargs):
        assert not name.startswith(("ouroboros", "supervisor")), "captured helper imported mutable body code"
        return imports.get(name) or builtins.__import__(name, *args, **kwargs)

    def execute_helper(code, namespace):
        namespace["__builtins__"] = cold_builtins
        builtins.exec(code, namespace)

    cold_builtins = {**vars(builtins), "__import__": import_module, "exec": execute_helper}

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code.co_filename == str(captured):
            trace.append(frame.f_code.co_name)

    previous_profile = sys.getprofile()
    try:
        sys.setprofile(profile)
        with pytest.raises(_BootstrapExit) as stopped:
            builtins.exec(compile(hook.read_bytes(), str(hook), "exec"), {
                "__file__": str(hook), "__name__": "ouroboros", "__builtins__": cold_builtins,
            })
    finally:
        sys.setprofile(previous_profile)
        released = writer_closed.is_set()
        if not released:
            close(89)  # teardown only: never leave a failed test's reader waiting
        for child in children:
            child.join(5)
            assert not child.is_alive(), "test-owned final recovery thread survived cleanup"
        if children:
            assert released, "the helper did not release its sole binding writer"
        if child_errors:
            raise child_errors[0]
    return SimpleNamespace(final=final, calls=calls, launches=launches, trace=trace,
                           exit_code=stopped.value.code, head=state.head,
                           handoff=json.loads((helper / "handoff.json").read_text()),
                           stderr=proxy_sys.stderr.getvalue() if proxy_sys.stderr else "")


@pytest.mark.parametrize("absent_streams", [False, True], ids=["redirected", "absent-stdin-stderr"])
def test_real_cold_hook_binds_the_final_serving_generation(tmp_path, monkeypatch, absent_streams):
    run = _cold_boot(tmp_path, monkeypatch, absent_streams=absent_streams)
    assert run.exit_code == 0
    assert {"main", "_switch", "_handover", "_wait_as_parent"} <= set(run.trace)
    assert run.handoff["phase"] == "switched" and run.head == "b" * 40
    assert run.final["parent"] == {"pid": 101, "birth": "win-filetime:1", "exit_code": 42}
    row = run.final["transaction"]
    assert (row["status"], row["successor_pid"], row["successor_birth"]) == (
        "normal_exit_acknowledged", 303, "win-filetime:3")
    assert row["supervisor_pid"] == 101 and row["ack_source"] == "windows_direct_parent_handle"
    argv, options = run.launches[0]
    assert argv == [r"C:\Python\python.exe", "server.py", "space argument", ""]
    assert options["env"]["__PYVENV_LAUNCHER__"] == r"C:\venv with spaces\Scripts\python.exe"
    assert options["close_fds"] is True and options.get("creationflags", 0) == 0
    assert not {"stdin", "stdout", "stderr"}.intersection(options)
    startup = options["startupinfo"]
    assert (startup.hStdInput, startup.hStdOutput, startup.hStdError) == (
        (0, 111, 0) if absent_streams else (110, 111, 112))
    parent_handle = int(options["env"][recovery.PLANNED_RESTART_PARENT_ENV])
    gate_handle = int(options["env"]["OUROBOROS_PLANNED_RESTART_BINDING_HANDLE"])
    assert set(startup.lpAttributeList["handle_list"]) == {
        parent_handle, gate_handle, 111, *(() if absent_streams else (110, 112)),
    }
    assert not any(key.endswith("_HANDLE") for key in run.final["env"])
    assert any(call[0] == "gate" for call in run.calls)
    boundaries = [call[0] for call in run.calls]
    assert boundaries.index("gate-enter") < boundaries.index("binding-write") < boundaries.index("gate")


@pytest.mark.parametrize("fault", ["metadata", "helper_birth"])
def test_cold_helper_does_not_invent_recovery_when_rebinding_fails(tmp_path, monkeypatch, caplog, fault):
    run = _cold_boot(tmp_path, monkeypatch, fault=fault)
    assert run.exit_code == 0  # the already started successor remains available
    assert run.final["parent"]["exit_code"] == 42
    assert run.final["transaction"]["status"] == "prepared"
    assert run.final["transaction"]["successor_pid"] == 202
    assert "continuation unproven" in caplog.text
    assert any(call[0] == "gate" for call in run.calls)  # the error also releases the child


@pytest.mark.parametrize("fault,expected_exit", [("", 0), ("panic", 99)])
def test_cold_helper_observes_parent_without_continuations(tmp_path, monkeypatch, fault, expected_exit):
    run = _cold_boot(tmp_path, monkeypatch, fault=fault, with_transaction=False)
    assert run.exit_code == expected_exit
    assert run.final["parent"]["exit_code"] == (99 if fault else 42)
    assert not run.final["transaction"]
    assert any(call[0] == "gate" for call in run.calls)
    assert not any(call[0] == "binding-write" for call in run.calls)


def test_cold_helper_keeps_original_parent_panic_absolute(tmp_path, monkeypatch):
    run = _cold_boot(tmp_path, monkeypatch, fault="panic")
    assert run.exit_code == 99
    assert run.final["parent"] == {"pid": 101, "birth": "win-filetime:1", "exit_code": 99}
    assert run.final["transaction"]["status"] == "prepared"
    assert run.final["transaction"]["successor_pid"] == 202


def test_helper_death_releases_eof_without_proving_the_unwritten_binding(tmp_path, monkeypatch, caplog):
    run = _cold_boot(tmp_path, monkeypatch, fault="helper_death")
    assert run.exit_code == 137
    assert run.final["parent"] == {"pid": 101, "birth": "win-filetime:1", "exit_code": 42}
    assert run.final["transaction"]["status"] == "prepared"
    assert run.final["transaction"]["successor_pid"] == 202
    assert "continuation unproven" in caplog.text
    assert any(call[0] == "gate" for call in run.calls)
