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
