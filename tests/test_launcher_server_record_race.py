"""The launcher retires one server record from two threads without losing the race on Windows.

stop_agent (window close) and the lifecycle thread (child exit) both retire state/server_process.json once
the server is reaped. Windows refuses a delete while another thread still reads the file (WinError 32), which
logged a traceback and made stop_agent report an unconfirmed stop. The fakes below give every platform those
sharing semantics: a read that the other caller overlaps stays open until a delete is attempted.
"""

import contextlib
import json
import pathlib
import threading

import pytest


@pytest.fixture
def launcher_root(monkeypatch, tmp_path):
    import launcher

    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    monkeypatch.setattr(launcher, "DATA_DIR", data)
    return launcher, data


def _retire_from_both_threads(launcher, data, monkeypatch):
    record_path = data / "state" / "server_process.json"
    record_path.write_text(json.dumps({"pid": 4242}), encoding="utf-8")
    readers: list = []
    first_reading, second_entering = threading.Event(), threading.Event()
    second_reading, delete_attempted = threading.Event(), threading.Event()
    real_read, real_unlink = pathlib.Path.read_text, pathlib.Path.unlink

    def read_text(self, *args, **kwargs):
        if self != record_path:
            return real_read(self, *args, **kwargs)
        readers.append(threading.get_ident())
        try:
            if not first_reading.is_set():
                first_reading.set()
                second_entering.wait(timeout=5)
                second_reading.wait(timeout=1)  # serialized callers never overlap: this wait just expires
            else:
                second_reading.set()
                delete_attempted.wait(timeout=5)
            return real_read(self, *args, **kwargs)
        finally:
            readers.remove(threading.get_ident())

    def unlink(self, *args, **kwargs):
        if self == record_path:
            delete_attempted.set()
            if readers:
                raise PermissionError(32, "The process cannot access the file because it is being used")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text)
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    results: dict = {}

    def retire(reason, entering=None):
        if entering is not None:
            entering.set()
        results[reason] = launcher._cleanup_recorded_server_group_for_pid(4242, reason)

    lifecycle = threading.Thread(target=retire, args=("agent_exit",))
    lifecycle.start()
    assert first_reading.wait(timeout=5)
    closer = threading.Thread(target=retire, args=("stop_agent", second_entering))
    closer.start()
    lifecycle.join(timeout=15)
    closer.join(timeout=15)
    return results, record_path


def test_both_retirers_of_one_record_succeed_under_windows_sharing(launcher_root, monkeypatch):
    launcher, data = launcher_root
    results, record_path = _retire_from_both_threads(launcher, data, monkeypatch)

    assert results == {"agent_exit": True, "stop_agent": True}
    assert not record_path.exists()


def test_without_the_lock_the_overlapping_read_loses_a_retirer(launcher_root, monkeypatch):
    launcher, data = launcher_root
    monkeypatch.setattr(launcher, "_server_record_retire_lock", contextlib.nullcontext())
    results, record_path = _retire_from_both_threads(launcher, data, monkeypatch)

    assert results == {"agent_exit": False, "stop_agent": True}, "the fake must reproduce the WinError 32 race"
    assert not record_path.exists()


def test_a_record_of_another_server_stays(launcher_root):
    launcher, data = launcher_root
    record_path = data / "state" / "server_process.json"
    record_path.write_text(json.dumps({"pid": 1111}), encoding="utf-8")

    assert launcher._cleanup_recorded_server_group_for_pid(4242, "agent_exit") is True
    assert record_path.exists()


@pytest.mark.parametrize("stopped", [True, False])
@pytest.mark.parametrize("forced", [False, True])
def test_recycle_and_lifecycle_join_one_generation_retirement(launcher_root, monkeypatch, stopped, forced):
    """The real recycle and monitor paths overlap while the old engine is still stopping."""
    import types
    import ouroboros.server_restart as restart

    launcher, _ = launcher_root
    exited, stopping, release, monitor_retiring = (threading.Event() for _ in range(4))
    calls, starts, results = [], [], []
    first = types.SimpleNamespace(pid=4242, returncode=0)
    first.stdin = types.SimpleNamespace(write=lambda _: None if forced else exited.set(), flush=lambda: None, close=lambda: None)

    def wait(**_):
        if forced and threading.current_thread().name == "closer" and not stopping.is_set():
            import subprocess
            raise subprocess.TimeoutExpired("fixture-server", 0)
        return exited.wait(timeout=5)

    first.wait = wait
    first.poll = lambda: 0 if exited.is_set() else None
    second = types.SimpleNamespace(pid=4243, returncode=0, poll=lambda: 0)
    second.wait = lambda **_: launcher._shutdown_event.set()
    monkeypatch.setattr(launcher, "_shutdown_event", threading.Event())
    monkeypatch.setattr(launcher, "_agent_restart_requested", threading.Event())
    monkeypatch.setattr(launcher, "_agent_proc", first)
    monkeypatch.setattr(launcher, "_agent_job", None)
    monkeypatch.setattr(launcher, "_agent_start", None)
    monkeypatch.setattr(launcher, "_agent_stopping", None)
    monkeypatch.setattr(launcher, "_generation_port", None)
    monkeypatch.setattr(launcher, "kill_process_tree", lambda *a, **kw: None)
    monkeypatch.setattr(launcher, "_retained_shared_daemon_pids", lambda: set())
    monkeypatch.setattr(launcher, "_pre_generation_cleanup", lambda: [])
    monkeypatch.setattr(launcher, "automatic_launch_allowed", lambda *a: True)
    monkeypatch.setattr(launcher, "_clear_port_file", lambda: True)
    monkeypatch.setattr(launcher, "update_external_host", lambda *a: None)
    monkeypatch.setattr(launcher, "_await_generation_endpoint", lambda *a, **kw: None)
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    monkeypatch.setattr(launcher, "_cleanup_recorded_server_group_for_pid", lambda *a: True)

    def start(_):
        proc = first if not starts else second
        starts.append(proc.pid)
        return proc

    def stop(_):
        calls.append(threading.current_thread().name)
        if len(calls) > 1 and not release.is_set():
            return False  # the daemon's actual _stopping response to a duplicate call
        stopping.set()
        exited.set()  # the server can exit while forced engine cleanup is still held
        assert release.wait(timeout=5)
        return stopped

    original = launcher._retire_agent

    def retire(proc, reason):
        if reason == "agent_exit" and proc is first:
            monitor_retiring.set()
        return original(proc, reason)

    monkeypatch.setattr(launcher, "start_agent", start)
    monkeypatch.setattr(restart, "_stop_owned_daemon", stop)
    monkeypatch.setattr(launcher, "_retire_agent", retire)
    # Start the closer first and hold its engine stop. The monitor must join it, not fail.
    closer = threading.Thread(target=lambda: results.append(launcher._request_agent_restart()), name="closer")
    closer.start()
    assert stopping.wait(timeout=5)
    monitor = threading.Thread(target=launcher.agent_lifecycle_loop, name="monitor")
    monitor.start()
    try:
        assert monitor_retiring.wait(timeout=5)
        monitor.join(timeout=0.2)
        joined = monitor.is_alive()
    finally:
        release.set()
        closer.join(timeout=5)
        monitor.join(timeout=5)
    assert not closer.is_alive() and not monitor.is_alive()
    assert joined, "in-progress retirement must not abort the requested successor"
    assert calls.count("closer") == 1
    assert starts == ([4242, 4243] if stopped else [4242])
    assert calls.count("monitor") == (1 if stopped else 0), "only the successor has another retirement"
