"""The informational Win32 creation token neither signals nor enrolls a process."""
from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import pytest

from ouroboros import platform_layer, process_custody, server_process


@pytest.mark.skipif(os.name != "nt", reason="native Win32 API is checked by the Windows CI lane")
def test_native_windows_current_process_binding(tmp_path):
    binding = server_process.record_service_binding(tmp_path, "main", "127.0.0.1", 3000, pid=os.getpid())
    assert binding["fingerprint"]["source"] == "windows_creation_time"
    assert int(binding["fingerprint"]["creation_time"]) > 0
    assert server_process.service_binding_is_live(binding)
    server_process.clear_service_binding(tmp_path, "main", binding)
    assert server_process.read_service_bindings(tmp_path) == {}


@pytest.fixture
def kernel32(monkeypatch):
    facts = {"creation": 133700000000000001, "handle": 123, "ok": True, "calls": []}

    class Function:
        def __init__(self, call):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    def open_process(access, inherit, pid):
        facts["calls"].append(("open", access, inherit, pid))
        return facts["handle"]

    def times(handle, created, *_rest):
        facts["calls"].append(("times", handle))
        if facts.get("raises"):
            raise RuntimeError("native API unavailable")
        created._obj.dwLowDateTime = facts["creation"] & 0xFFFFFFFF
        created._obj.dwHighDateTime = facts["creation"] >> 32
        return facts["ok"]

    api = SimpleNamespace(OpenProcess=Function(open_process), GetProcessTimes=Function(times),
                          CloseHandle=Function(lambda handle: facts["calls"].append(("close", handle))))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: api, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    return facts


def test_creation_filetime_uses_limited_query_and_closes_handle(kernel32):
    assert server_process._windows_service_creation_time(456) == str(kernel32["creation"])
    assert kernel32["calls"] == [("open", 0x1000, False, 456), ("times", 123), ("close", 123)]


@pytest.mark.parametrize("change,error,closed", [
    ({"handle": 0}, OSError, False),
    ({"ok": False}, OSError, True),
    ({"creation": 0}, ValueError, True),
    ({"raises": True}, RuntimeError, True),
])
def test_native_identity_failure_is_explicit_and_closes_owned_handle(kernel32, change, error, closed):
    kernel32.update(change)
    with pytest.raises(error):
        server_process._windows_service_creation_time(456)
    assert (("close", 123) in kernel32["calls"]) is closed


def test_windows_binding_dead_reused_and_unreadable_processes(tmp_path, monkeypatch, kernel32):
    monkeypatch.setattr(server_process, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(process_custody, "_fingerprint_matches",
                        lambda _row: pytest.fail("informational Windows identity must not change kill fingerprints"))
    binding = server_process.record_service_binding(tmp_path, "main", "127.0.0.1", 3000, pid=456)
    assert binding["fingerprint"] == {"source": "windows_creation_time", "creation_time": str(kernel32["creation"])}
    assert server_process.service_binding_is_live(binding)
    kernel32["creation"] += 1
    assert not server_process.service_binding_is_live(binding)
    kernel32["handle"] = 0
    with pytest.raises(OSError):
        server_process.service_binding_is_live(binding)
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda _pid: False)
    before = len(kernel32["calls"])
    assert not server_process.service_binding_is_live(binding)
    assert len(kernel32["calls"]) == before
    assert not (tmp_path / "state/process_ledger.jsonl").exists()


def test_failed_windows_registration_does_not_publish_a_false_binding(tmp_path, monkeypatch, kernel32):
    monkeypatch.setattr(server_process, "os", SimpleNamespace(name="nt"))
    kernel32["handle"] = 0
    with pytest.raises(OSError):
        server_process.record_service_binding(tmp_path, "host_service", "127.0.0.1", 3001, pid=456)
    assert server_process.read_service_bindings(tmp_path) == {}
