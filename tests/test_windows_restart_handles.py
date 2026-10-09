"""Portable API doubles for restart handle ownership and wait-result handling.

Native handle inheritance is exercised separately by test_windows_restart_native.
"""
import ast
import inspect
import os
from types import SimpleNamespace
import subprocess
import sys

import pytest

from ouroboros import platform_layer


@pytest.mark.parametrize("wait_result", [0x80, 0x102, 0xFFFFFFFF], ids=["abandoned", "timeout", "failed"])
def test_parent_wait_requires_a_signaled_process_before_reading_exit(monkeypatch, wait_result):
    calls = []
    monkeypatch.setitem(sys.modules, "_winapi", SimpleNamespace(
        INFINITE=0xFFFFFFFF, WAIT_OBJECT_0=0,
        WaitForSingleObject=lambda handle, timeout: calls.append(("wait", handle, timeout)) or wait_result,
        GetExitCodeProcess=lambda handle: pytest.fail("an unsignaled wait is not exit proof"),
        CloseHandle=lambda handle: calls.append(("close", handle)),
    ))
    monkeypatch.setattr(platform_layer, "_kernel32", SimpleNamespace(
        GetProcessId=lambda handle: pytest.fail("identity must follow a signaled wait")), raising=False)

    with pytest.raises(OSError, match="did not signal"):
        platform_layer.await_process_handle(77)
    assert calls == [("wait", 77, 0xFFFFFFFF), ("close", 77)]


@pytest.fixture
def windows_handles(monkeypatch):
    calls = []

    class Handle(int):
        def Close(self):
            calls.append(("close", int(self)))

    class StartupInfo(SimpleNamespace):
        def copy(self):
            return StartupInfo(**{**vars(self), "lpAttributeList": {
                "handle_list": list(self.lpAttributeList["handle_list"])}})

    def duplicate(current, raw, target, access, inherit, options):
        assert current == target == -1 and access == 0 and inherit is True and options == 2
        calls.append(("duplicate", raw))
        return raw + 100

    def open_process(access, inherit, pid):
        assert (access, inherit, pid) == (0x101000, True, os.getpid())  # wait/query, never terminate rights
        return 77

    api = SimpleNamespace(OpenProcess=open_process, GetCurrentProcess=lambda: -1,
                          DuplicateHandle=duplicate, DUPLICATE_SAME_ACCESS=2,
                          CloseHandle=lambda handle: calls.append(("close", handle)),
                          STARTF_USESTDHANDLES=0x100, FILE_TYPE_CHAR=2,
                          GetFileType=lambda handle: 1)
    monkeypatch.setitem(sys.modules, "_winapi", api)
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda fd: fd + 10))
    monkeypatch.setattr(platform_layer, "sys", SimpleNamespace(**{
        f"__{name}__": SimpleNamespace(fileno=lambda fd=fd: fd)
        for fd, name in enumerate(("stdin", "stdout", "stderr"))}))
    monkeypatch.setattr(subprocess, "Handle", Handle, raising=False)
    monkeypatch.setattr(subprocess, "STARTUPINFO", StartupInfo, raising=False)
    monkeypatch.setattr(subprocess, "STARTF_USESTDHANDLES", 0x100, raising=False)
    return calls, api


def test_startupinfo_failure_closes_the_opened_inheritable_handle(monkeypatch, windows_handles):
    calls, _api = windows_handles
    failure = OSError("STARTUPINFO unavailable")

    def startupinfo(**kwargs):
        raise failure

    monkeypatch.setattr(subprocess, "STARTUPINFO", startupinfo)
    with pytest.raises(OSError) as raised:
        with platform_layer.inheritable_self_handle():
            pytest.fail("setup should fail before publishing handles")
    assert raised.value is failure
    assert calls == [("close", 77)]


def test_partial_stdio_setup_closes_every_acquired_handle(monkeypatch, windows_handles):
    calls, api = windows_handles
    original = api.DuplicateHandle

    def duplicate(*args):
        if args[1] == 11:
            raise OSError("cannot duplicate stdout")
        return original(*args)

    monkeypatch.setattr(api, "DuplicateHandle", duplicate)
    with pytest.raises(OSError, match="cannot duplicate stdout"):
        with platform_layer.inheritable_self_handle():
            pytest.fail("setup should fail before publishing handles")
    assert calls == [("duplicate", 10), ("close", 110), ("close", 77)]


def _windows_popen_consumer(api):
    """Run the installed CPython Windows Popen methods, replacing only its OS APIs.

    The host interpreter selects POSIX methods on Darwin/Linux. Compile the
    unselected Windows methods from that same stdlib source to inspect the actual
    CreateProcess boundary there too; this is not native Windows evidence.
    """
    source = ast.parse(inspect.getsource(subprocess))
    popen = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Popen")
    windows = next(node for node in popen.body if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Name) and node.test.id == "_mswindows")
    methods = [node for node in windows.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_get_handles", "_make_inheritable", "_filter_handle_list", "_execute_child"}]
    assert len(methods) == 4
    namespace = {**vars(subprocess), "_winapi": api, "msvcrt": sys.modules["msvcrt"]}
    exec(compile(ast.Module(body=methods, type_ignores=[]), subprocess.__file__, "exec"), namespace)
    consumer = type("WindowsPopenConsumer", (), {node.name: namespace[node.name] for node in methods})()
    consumer._close_pipe_fds = lambda *handles: None
    return consumer


@pytest.mark.parametrize("stdio_mode", ["redirected", "absent-stdin-stderr", "closed-os-fd"])
def test_actual_popen_preserves_stdio_at_createprocess(monkeypatch, windows_handles, stdio_mode):
    calls, api = windows_handles
    absent = stdio_mode != "redirected"
    if stdio_mode == "absent-stdin-stderr":
        monkeypatch.setattr(platform_layer.sys, "__stdin__", None)

        def closed():
            raise ValueError("I/O operation on closed file")

        monkeypatch.setattr(platform_layer.sys, "__stderr__", SimpleNamespace(fileno=closed))
    elif stdio_mode == "closed-os-fd":
        # An externally closed fd can leave the Python stream object present.
        # CPython's _Py_get_osfhandle raises OSError for INVALID_HANDLE_VALUE:
        # https://github.com/python/cpython/blob/f08d3c437bc5f41973ddd11f4cfc3d9fe04010f7/Python/fileutils.c#L2246-L2252
        def get_osfhandle(fd):
            if fd != 1:
                raise OSError(9, "Bad file descriptor")
            return 11

        monkeypatch.setattr(sys.modules["msvcrt"], "get_osfhandle", get_osfhandle)

    created = []

    def createprocess(executable, args, process_security, thread_security, inherit, flags, env, cwd, info):
        created.append(info)
        assert inherit == 1 and flags == 0  # retains the caller's console group
        return 500, 501, 4242, 4243

    api.CreateProcess = createprocess
    consumer = _windows_popen_consumer(api)
    with platform_layer.inheritable_self_handle() as (parent, kwargs):
        assert parent == 77 and not {"stdin", "stdout", "stderr"}.intersection(kwargs)
        handles = consumer._get_handles(None, None, None)
        assert handles == (-1,) * 6  # no synthesized Popen pipes or standard-handle overrides
        arguments = {name: None for name in inspect.signature(consumer._execute_child).parameters}
        arguments.update(args=["python.exe", "server.py"], close_fds=kwargs["close_fds"], pass_fds=(),
                         startupinfo=kwargs["startupinfo"], creationflags=0, shell=False,
                         **dict(zip(("p2cread", "p2cwrite", "c2pread", "c2pwrite", "errread", "errwrite"), handles)))
        consumer._execute_child(**arguments)
        [info] = created
        assert (info.hStdInput, info.hStdOutput, info.hStdError) == ((0, 111, 0) if absent else (110, 111, 112))
        assert info.dwFlags & 0x100
        assert set(info.lpAttributeList["handle_list"]) == ({77, 111} if absent else {77, 110, 111, 112})
        assert not any(row[0] == "close" and row[1] != 501 for row in calls)
    assert {row[1] for row in calls if row[0] == "close"} == ({77, 111, 501} if absent else {77, 110, 111, 112, 501})
