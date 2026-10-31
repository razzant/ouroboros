"""Handoff/Panic interleavings: every native boundary is replaced before fake PIDs exist."""
import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from ouroboros import launcher_background as background, launcher_bootstrap as boot, platform_layer


@pytest.fixture
def native_guard(monkeypatch):
    attempts = []
    for name in ("OUROBOROS_PLANNED_RESTART_PARENT_HANDLE", "OUROBOROS_PLANNED_RESTART_BINDING_HANDLE"):
        monkeypatch.delenv(name, raising=False)

    def denied(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("unexpected native effect in fake-PID test")

    originals = []

    def block(module, name):
        if hasattr(module, name):
            originals.append(getattr(module, name))
            monkeypatch.setattr(module, name, denied)

    for name in ("kill", "killpg", "_exit", "execl", "execle", "execlp", "execlpe", "execve", "execv",
                 "execvp", "execvpe", "posix_spawn", "posix_spawnp", "system", "popen", "startfile",
                 "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
        block(os, name)
    block(signal, "pidfd_send_signal")
    # Replacing __init__ also fences previously imported Popen class aliases.
    for name in ("__init__", "terminate", "kill", "send_signal"):
        block(subprocess.Popen, name)
    process_names = ("terminate_job", "request_process_tree_kill", "kill_process_tree", "force_kill_pid",
                     "kill_process_on_port", "kill_pid_tree", "terminate_process_group_id", "kill_process_group_id",
                     "terminate_process_tree", "signal_pid", "deliver_signal", "open_path_external",
                     "request_native_attention", "inheritable_self_handle", "await_process_handle")
    for name in process_names:
        block(platform_layer, name)
    kernel = getattr(platform_layer, "_kernel32", None)
    for name in ("TerminateProcess", "TerminateJobObject", "GenerateConsoleCtrlEvent", "CreateProcessW"):
        block(kernel, name)
    winapi = sys.modules.get("_winapi") or ModuleType("_winapi")
    # Proactor loops own real I/O completion handles, including the pytest phase
    # loops. Give only asyncio its original CloseHandle; product/subprocess calls
    # and imported aliases must still be denied, even for a swallowed failure.
    asyncio_windows = sys.modules.get("asyncio.windows_events")
    asyncio_winapi = SimpleNamespace(**vars(winapi)) if asyncio_windows is not None else None
    for name in ("TerminateProcess", "CreateProcess", "OpenProcess", "CloseHandle", "GenerateConsoleCtrlEvent",
                 "WaitForSingleObject", "GetExitCodeProcess", "DuplicateHandle"):
        block(winapi, name)
        monkeypatch.setattr(winapi, name, denied, raising=False)
        if asyncio_winapi is not None and name != "CloseHandle":
            setattr(asyncio_winapi, name, denied)
    if asyncio_windows is not None:
        monkeypatch.setattr(asyncio_windows, "_winapi", asyncio_winapi)
    for name in ("WinDLL", "OleDLL"):
        block(ctypes, name)
    # Replace imported function aliases before any fake PID/handle test body.
    for name, module in list(sys.modules.items()):
        if module is not None and (name == "launcher" or name.startswith(("ouroboros.", "supervisor."))):
            for attr, value in list(vars(module).items()):
                if any(value is original for original in originals):
                    monkeypatch.setattr(module, attr, denied)

    class Native:
        def __getattr__(self, name):
            return denied

    monkeypatch.setattr(ctypes, "WinDLL", denied, raising=False)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        kernel32=Native(), user32=Native(), shell32=Native(), ntdll=Native()), raising=False)
    guarded_kernel = Native()
    # These are the layer's only non-process/console kernel32 calls. Keep real
    # locking for the dialogue store and append_jsonl; every other call denies.
    if kernel is not None:
        for name in ("LockFileEx", "UnlockFileEx"):
            setattr(guarded_kernel, name, getattr(kernel, name))
    monkeypatch.setattr(platform_layer, "_kernel32", guarded_kernel, raising=False)
    monkeypatch.setattr(background, "NativeAppearance", lambda: SimpleNamespace())
    monkeypatch.setattr(background, "WindowChrome", lambda: SimpleNamespace())
    monkeypatch.setattr(background, "native_notifier", lambda *a: SimpleNamespace())
    monkeypatch.setattr(background, "_active", None)
    monkeypatch.setitem(sys.modules, "_winapi", winapi)
    yield denied
    assert attempts == [], f"unexpected effects, including swallowed calls: {attempts}"


def test_native_guard_preserves_windows_file_lock_calls(monkeypatch):
    calls = []
    kernel = SimpleNamespace(
        LockFileEx=lambda *args: calls.append(("lock", args)) or 1,
        UnlockFileEx=lambda *args: calls.append(("unlock", args)) or 1,
    )

    class Overlapped(ctypes.Structure):
        _fields_ = [("Offset", ctypes.c_uint32), ("OffsetHigh", ctypes.c_uint32)]

    with monkeypatch.context() as patch:
        patch.setattr(platform_layer, "_kernel32", kernel, raising=False)
        patch.setattr(platform_layer, "_OVERLAPPED", Overlapped, raising=False)
        patch.setattr(platform_layer, "IS_WINDOWS", True)
        patch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda fd: fd + 1000))
        with contextlib.contextmanager(native_guard.__wrapped__)(patch):
            platform_layer.file_lock_exclusive_nb(41)
            platform_layer.file_unlock(41)
        assert [name for name, _ in calls] == ["lock", "unlock"]
        assert calls[0][1][:-1] == (1041, 3, 0, platform_layer._WIN32_LOCK_LENGTH, 0)
        assert calls[1][1][:-1] == (1041, 0, platform_layer._WIN32_LOCK_LENGTH, 0)
        for _, args in calls:
            overlap = args[-1]._obj
            assert (overlap.OffsetHigh << 32) | overlap.Offset == platform_layer._WIN32_LOCK_OFFSET


def test_native_guard_denies_effects_and_aliases_but_preserves_asyncio_close(monkeypatch):
    calls = []

    def record(name):
        def invoke(*args, **kwargs):
            calls.append((name, args, kwargs))
        return invoke

    kernel_names = ("TerminateProcess", "TerminateJobObject", "GenerateConsoleCtrlEvent", "CreateProcessW",
                    "OpenProcess", "CloseHandle", "GetConsoleProcessList", "GetProcessTimes", "GetExitCodeProcess",
                    "GetProcessId", "GetCurrentProcess", "IsProcessInJob", "QueryInformationJobObject",
                    "CreateJobObjectW", "SetInformationJobObject", "AssignProcessToJobObject", "UnknownNativeEffect")
    winapi_names = ("TerminateProcess", "CreateProcess", "OpenProcess", "CloseHandle", "GenerateConsoleCtrlEvent",
                    "WaitForSingleObject", "GetExitCodeProcess", "DuplicateHandle")
    kernel = SimpleNamespace(**{name: record(name) for name in (*kernel_names, "LockFileEx", "UnlockFileEx")})
    winapi = ModuleType("_winapi")
    for name in winapi_names:
        setattr(winapi, name, record(name))
    real_close = winapi.CloseHandle
    asyncio_windows = ModuleType("asyncio.windows_events")
    asyncio_windows._winapi = winapi
    aliases = ModuleType("ouroboros._native_guard_probe")
    aliases.close = real_close
    aliases.terminate = kernel.TerminateProcess
    aliases.kill = os.kill
    aliases.signal_pid = platform_layer.signal_pid
    popen_alias = subprocess.Popen

    with monkeypatch.context() as patch:
        patch.setattr(platform_layer, "_kernel32", kernel, raising=False)
        patch.setitem(sys.modules, "_winapi", winapi)
        patch.setitem(sys.modules, "asyncio.windows_events", asyncio_windows)
        patch.setitem(sys.modules, aliases.__name__, aliases)
        with pytest.raises(AssertionError, match="unexpected effects, including swallowed calls"):
            with contextlib.contextmanager(native_guard.__wrapped__)(patch) as denied:
                asyncio_windows._winapi.CloseHandle(901)
                effects = [getattr(platform_layer._kernel32, name) for name in kernel_names]
                effects += [getattr(sys.modules["_winapi"], name) for name in winapi_names]
                effects += [getattr(asyncio_windows._winapi, name) for name in winapi_names if name != "CloseHandle"]
                effects += [getattr(getattr(ctypes.windll, library), name)
                            for library in ("kernel32", "user32", "shell32", "ntdll") for name in kernel_names]
                effects += [aliases.close, aliases.terminate, aliases.kill, aliases.signal_pid,
                            os.kill, platform_layer.signal_pid, popen_alias.__init__, ctypes.WinDLL]
                for effect in effects:
                    assert effect is denied
                    with pytest.raises(AssertionError, match="unexpected native effect in fake-PID test"):
                        effect(42)
        assert calls == [("CloseHandle", (901,), {})]


@pytest.fixture
def handoff(native_guard, monkeypatch, tmp_path):
    from tests.test_launcher_source_handoff import context
    from ouroboros.delegate_recovery import PLANNED_RESTART_PARENT_ENV

    events, spawns, owned, handles = [], [], [], []
    startup = object()

    @contextlib.contextmanager
    def inherited():
        handles.append("open")
        try:
            yield 501, {"startupinfo": startup, "close_fds": True}
        finally:
            handles.append("closed")

    def stop(proc, **kwargs):
        if not any(proc is item for item in owned) or kwargs:
            native_guard("unexpected process stop", proc, kwargs)
        events.append(("stop", proc.pid))
        return {"requested": True}

    def leave(code):
        if code not in (0, 1, 99):
            native_guard("unexpected exit", code)
        events.append(("exit", code))

    monkeypatch.setattr(boot, "_BOOTSTRAP_COMMANDS", [])
    monkeypatch.setattr(boot, "source_launcher_importable", lambda *a, **kw: True)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "inherited_console_kwargs", lambda: {})
    monkeypatch.setattr(platform_layer, "inheritable_self_handle", inherited)
    monkeypatch.setattr(platform_layer, "request_process_tree_kill", stop)
    monkeypatch.setattr(os, "_exit", leave)
    monkeypatch.setattr(background, "indicator_class", lambda: None)
    monkeypatch.setattr(background, "request_tray_cleanup", lambda: None)
    bg = background.Background(lambda: events.append("ordinary quit"), lambda: 1, threading.Event())

    def run():
        boot.exec_source_launcher(
            context(tmp_path), [], release_lock=lambda: events.append("release"), acquire_lock=lambda: False,
            before_release=lambda: events.append("remove icon"), on_failure=lambda: events.append("restore"),
            desktop=True, macos=False, cancelled=bg.shutdown.is_set)

    def install_spawn(action):
        def checked(command, **kwargs):
            expected = [context(tmp_path).embedded_python, str(context(tmp_path).repo_dir / "launcher.py")]
            if command != expected or spawns:
                native_guard("unexpected process launch", command, kwargs)
            assert kwargs["close_fds"] is True
            spawns.append((command, kwargs))
            proc = action(command, **kwargs)
            owned.append(proc)
            return proc
        monkeypatch.setattr(boot.subprocess, "Popen", checked)

    run.spawn = install_spawn
    run.spawns, run.handles = spawns, handles
    run.expected_spawns = 1

    def inheritance():
        kwargs = spawns[0][1]
        assert kwargs["startupinfo"] is startup
        assert kwargs["env"][PLANNED_RESTART_PARENT_ENV] == "501"
        return kwargs["env"]

    run.inheritance = inheritance
    yield bg, events, run
    assert len(spawns) == run.expected_spawns
    if handles:  # the rejected wait-only patch never entered the inheritance helper
        assert handles == ["open", "closed"]


@pytest.fixture
def successor(native_guard, monkeypatch):
    """Run the real launcher admission and existing observation with a fake inherited handle."""
    import launcher
    from ouroboros import delegate_recovery

    def run(env, parent_exit):
        observed, started = [], []

        def observe(handle):
            if handle != 501:
                native_guard("unexpected inherited handle", handle)
            observed.append(handle)
            return {"pid": 45, "birth": "fake-parent", "exit_code": parent_exit}

        class StartupReached(Exception):
            pass

        def startup(argv):
            started.append(argv)
            raise StartupReached

        with monkeypatch.context() as patch:
            patch.setattr(launcher, "IS_WINDOWS", True)
            patch.setattr(launcher, "parse_launch_options", startup)
            patch.setattr(delegate_recovery, "_restart_parent", None)
            patch.setattr(platform_layer, "await_process_handle", observe)
            patch.delenv(delegate_recovery.PLANNED_RESTART_BINDING_ENV, raising=False)
            parent = env.get(delegate_recovery.PLANNED_RESTART_PARENT_ENV)
            if parent:
                patch.setenv(delegate_recovery.PLANNED_RESTART_PARENT_ENV, parent)
            else:
                patch.delenv(delegate_recovery.PLANNED_RESTART_PARENT_ENV, raising=False)
            try:
                launcher.main([])
            except StartupReached:
                pass
            assert delegate_recovery.PLANNED_RESTART_PARENT_ENV not in os.environ
            assert delegate_recovery.observe_restart_parent().get("exit_code") == parent_exit
        assert observed == ([501] if parent else []), "the real observer consumes the handle once"
        return bool(started)

    return run


@pytest.mark.parametrize("refused", [False, True])
def test_panic_exits_while_spawn_is_stuck_and_late_successor_refuses(handoff, successor, monkeypatch, refused):
    from ouroboros import server_control

    bg, events, run = handoff
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    owners = []

    def spawn(*args, **kwargs):
        entered.set()
        release.wait()  # deliberately stuck until the test's finally, never a native child
        if refused:
            raise OSError("mock spawn refusal")
        return SimpleNamespace(pid=41)

    def persist(*args, **kwargs):
        owners.append(threading.current_thread())
        events.append("persisted")
        # Use the old hook if testing the rejected patch: it must fail this case.
        kwargs.get("exit_fn", os._exit)(99)
        exited.set()

    run.spawn(spawn)
    monkeypatch.setattr(server_control, "execute_panic_stop", persist)
    monkeypatch.setattr("ouroboros.panic_source.record_source", lambda *a, **kw: None)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(3)
        bg.request_panic()  # must return while Popen is still blocked
        assert bg.shutdown.is_set() and bg.panic.is_set() and bg.quit.is_set()
        assert exited.wait(3), "Panic must reach exit99 while Popen remains blocked"
        assert thread.is_alive() and not release.is_set()
        assert [e for e in events if isinstance(e, tuple)] == [("exit", 99)]
        assert not owners[0].daemon
        if not refused:
            # Model a process created before Popen returns: its own production
            # admission observes parent 99 and refuses before starting work.
            assert not successor(run.inheritance(), 99)
    finally:
        release.set()
        thread.join(5)
        for owner in owners:
            owner.join(5)
    assert not thread.is_alive() and all(not owner.is_alive() for owner in owners)
    assert [e for e in events if isinstance(e, tuple)] == (
        [("exit", 99)] if refused else [("exit", 99), ("stop", 41)])
    run.inheritance()
    if not refused:
        assert boot.source_handoff.successor.pid == 41  # retained; a stop request proves no exit
    assert "ordinary quit" not in events and "restore" not in events


def test_successful_spawn_cannot_exit_over_pending_panic_persistence(handoff, monkeypatch):
    from ouroboros import server_control

    bg, events, run = handoff
    writing, release = threading.Event(), threading.Event()
    owners = []

    def persist(*args, **kwargs):
        owners.append(threading.current_thread())
        writing.set()
        assert release.wait(5)
        events.append("persisted")
        kwargs.get("exit_fn", os._exit)(99)

    def spawn(*args, **kwargs):
        bg.request_panic()  # the successful Popen interleaving from the independent review
        assert writing.wait(3)
        return SimpleNamespace(pid=44)

    run.spawn(spawn)
    monkeypatch.setattr(server_control, "execute_panic_stop", persist)
    monkeypatch.setattr("ouroboros.panic_source.record_source", lambda *a, **kw: None)
    try:
        run()
        assert ("stop", 44) in events, "the successful spawn stays under Panic custody"
        assert not any(isinstance(e, tuple) and e[0] == "exit" for e in events)
    finally:
        release.set()
        for owner in owners:
            owner.join(5)
    assert all(not owner.is_alive() for owner in owners)
    assert events[-2:] == ["persisted", ("exit", 99)]


@pytest.mark.parametrize("cancelled", [False, True])
def test_successful_spawn_commits_normally_or_stops_its_captured_child(handoff, successor, cancelled):
    bg, events, run = handoff

    def spawn(*args, **kwargs):
        if cancelled:
            bg.shutdown.set()
        return SimpleNamespace(pid=42)

    run.spawn(spawn)
    run()
    assert [e for e in events if isinstance(e, tuple)] == ([("stop", 42)] if cancelled else [("exit", 0)])
    if not cancelled:
        assert successor(run.inheritance(), 0), "normal transfer still reaches startup"


def test_panic_between_publication_and_terminal_commit_stops_the_successor(handoff, monkeypatch):
    bg, events, run = handoff
    run.spawn(lambda *a, **kw: SimpleNamespace(pid=43))
    leave = boot.source_handoff.leave

    def race(code, cancelled):
        boot.source_handoff.claim_stop()
        bg.shutdown.set()
        boot.stop_bootstrap_commands()
        leave(code, cancelled)

    monkeypatch.setattr(boot.source_handoff, "leave", race)
    run()
    assert [e for e in events if isinstance(e, tuple)] == [("stop", 43)]


def test_cancellation_during_parent_handle_setup_closes_admission(handoff, monkeypatch):
    bg, events, run = handoff
    inherited = platform_layer.inheritable_self_handle

    @contextlib.contextmanager
    def cancel():
        with inherited() as value:
            bg.shutdown.set()
            yield value

    monkeypatch.setattr(platform_layer, "inheritable_self_handle", cancel)
    run.expected_spawns = 0
    run()
    assert not any(isinstance(e, tuple) for e in events), "no spawn, stop or terminal transfer"


def test_source_without_parent_handle_keeps_old_launch_behavior(successor):
    assert successor({}, None)
