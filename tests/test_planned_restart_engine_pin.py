"""Application Exit/Restart stops the owned engine, independently of the next pin.

The old pin-comparison cases are superseded by the application-ending contract;
cold-start storage compatibility remains the runtime's own admission decision.
"""
import inspect
import json
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as owned, claudexor_runtime, config, server_restart


def _rows(root):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize("version", ["3.9.8", "3.10.0", "4.3.0", ""])
def test_application_stop_never_compares_or_provisions_an_engine(tmp_path, monkeypatch, version):
    calls = []
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(
        stop_outcome=lambda **kw: calls.append(kw) or "stopped", engine_version=version))
    monkeypatch.setattr(claudexor_runtime, "load_runtime_pin", lambda **kw: pytest.fail("stop reads no pin"))
    monkeypatch.setattr(owned, "read_owned_gateway", lambda **kw: pytest.fail("stop has no network wait"))
    assert server_restart._stop_owned_daemon("Restart") is True
    assert calls == [{"force": True}]
    assert not (tmp_path / "state" / "panic_stop.flag").exists()


@pytest.mark.parametrize("outcome,confirmed", [("stopped", True), ("nothing_to_stop", True), ("unconfirmed", False)])
def test_application_stop_preserves_physical_outcome(tmp_path, monkeypatch, caplog, outcome, confirmed):
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(stop_outcome=lambda **kw: outcome))
    assert server_restart._stop_owned_daemon("Exit") is confirmed
    assert ("unconfirmed" in caplog.text) is not confirmed


def test_raising_stop_keeps_an_explicit_unconfirmed_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    def fail(**kw):
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(stop_outcome=fail))
    assert server_restart._stop_owned_daemon("Exit") is False
    row = _rows(tmp_path)[-1]
    assert row["type"] == "process_stop_unconfirmed" and row["purpose"] == owned.CUSTODY_PURPOSE


def test_unprovisioned_application_stop_does_not_start_a_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: owned.OwnedClaudexorDaemon())
    monkeypatch.setattr(owned, "ensure_owned_gateway", lambda **kw: pytest.fail("stop started an engine"))
    assert server_restart._stop_owned_daemon("Exit") is True


@pytest.mark.parametrize("restart", [False, True], ids=["quit", "restart"])
def test_emergency_cleanup_ends_the_engine_on_both_doors(tmp_path, monkeypatch, restart):
    import threading
    import server
    calls = []
    flags = threading.Event()
    if restart:
        flags.set()
    monkeypatch.setattr(server, "_restart_requested", flags)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_managed_update_pending_kwargs", lambda: {})
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: calls.append("workers"))
    monkeypatch.setattr(server, "_stop_owned_daemon", lambda label: calls.append("engine") or True)
    monkeypatch.setattr(server, "stop_owned_work", lambda root: calls.append("owned-stop"))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.body_adoption.arm", lambda *a, **kw: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    server._emergency_process_cleanup()
    assert calls == ["workers", "engine", "owned-stop"]


def test_lifespan_ends_engine_after_worker_custody_before_optional_waits():
    import server
    source = inspect.getsource(server.lifespan)
    kill = source.index("kill_workers(\n                force=True,\n                terminal_status=cleanup_status")
    stop = source.index('_stop_owned_daemon("Server shutdown")')
    assert kill < stop < source.index("if extension_reconcile_task is not None:")
    assert "_stop_owned_daemon_for_new_pin" not in source


@pytest.mark.parametrize("version,other,below", [
    ("4.3.0", "4.3.1", True), ("4.3.1", "4.3.0", False), ("4.3.1", "4.3.1", False),
    ("4.3.0", "v4.3.1", False), ("4.3.0", "4.3.1-rc.1", False), ("4.3.0", "4.3", False), ("4.3.0", "", False),
    ("4.3.0", "4.3.1\n", False), ("4.3.0", "\u0664.\u0663.\u0661", False), ("v4.3.0", "4.3.1", False),
])
def test_release_order_is_the_engines_plain_semver_only(version, other, below):
    assert claudexor_runtime.release_below(version, other) is below
