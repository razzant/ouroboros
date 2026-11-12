"""Panic makes physical requests before manager settlement or persistence.

Real processes are confined to disposable test roots, never the installation.
The HTTP fixture models the authenticated engine handshake, not a real daemon.
"""
from __future__ import annotations

import concurrent.futures
import logging
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as daemon
from ouroboros import local_model, process_custody, server_control
from ouroboros import platform_layer as platform
from tests.test_daemon_stop_diagnostics import authenticated_home  # noqa: F401
from tests.test_panic_owned_requests import owners  # noqa: F401



def _child(root, purpose):
    return process_custody.spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        drive_root=root, purpose=purpose, scope="daemon" if purpose == daemon.CUSTODY_PURPOSE else "session")


@pytest.mark.usefixtures("authenticated_home", "owners")
def test_public_panic_requests_attached_and_local_children_before_blocked_settlement(
    tmp_path, monkeypatch, caplog,
):
    manager = daemon.OwnedClaudexorDaemon()
    model = local_model.LocalModelManager()
    procs = [_child(tmp_path, purpose) for purpose in (daemon.CUSTODY_PURPOSE, "local_model_server")]
    model._proc = procs[1]
    manager.ensure_running()  # real authenticated fixture + real measured ledger + OS target capture
    assert manager._proc is None and len(manager._panic_targets) == 1
    monkeypatch.setattr(daemon, "_MANAGER", manager)
    monkeypatch.setattr(local_model, "_manager", model)
    monkeypatch.setattr(platform, "kill_process_on_port", lambda _: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr(server_control.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _: None)
    receipt_calls, flag_calls = [], []
    real_request = platform.request_process_tree_kill

    def observe_request(*args, **kwargs):
        result = real_request(*args, **kwargs)
        receipt_calls.append(result)
        return result

    def persist_after_requests(_root):
        expected = {procs[1].pid} if platform.IS_MACOS else {p.pid for p in procs}
        assert {r["pid"] for r in receipt_calls if r["requested"]} == expected
        if platform.IS_MACOS:
            assert any(r["pid"] == procs[0].pid and not r["requested"] for r in receipt_calls)
        assert manager._lock.locked() and model._lock.locked()
        for proc in procs:
            if proc.pid in expected:
                assert proc.wait(timeout=2) is not None
        flag_calls.append(True)

    monkeypatch.setattr(platform, "request_process_tree_kill", observe_request)
    monkeypatch.setattr(server_control, "_write_panic_flag", persist_after_requests)
    locks = [manager._lock, model._lock, daemon._MANAGER_LOCK, local_model._manager_lock]
    prior_threads = set(threading.enumerate())
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        for lock in locks:
            lock.acquire()
        started = time.monotonic()
        with caplog.at_level(logging.CRITICAL):
            future = pool.submit(server_control.execute_panic_stop, SimpleNamespace(stop=lambda: None),
                                 lambda **_: None, data_dir=tmp_path, panic_exit_code=99,
                                 log=logging.getLogger("panic-request-test"))
            with pytest.raises(SystemExit):
                future.result(timeout=3)
        assert time.monotonic() - started < 3 and flag_calls == [True]
        assert "unfinished" in caplog.text and "unresolved custody retained" in caplog.text
        assert process_custody.ledger_path(tmp_path).exists()
    finally:
        for lock in reversed(locks):
            if lock.locked():
                lock.release()
        pool.shutdown(wait=True)
        for proc in procs:
            real_request(proc)
            proc.wait(timeout=5)
        for thread in set(threading.enumerate()) - prior_threads:
            if thread.name.startswith("panic-"):
                thread.join(timeout=10)
                assert not thread.is_alive()


@pytest.mark.usefixtures("authenticated_home")
def test_attached_request_uses_pinned_identity_without_rereading_custody(
    tmp_path, monkeypatch,
):
    manager = daemon.OwnedClaudexorDaemon()
    proc = _child(tmp_path, daemon.CUSTODY_PURPOSE)
    try:
        manager.ensure_running()
        assert manager._panic_targets
        def unavailable(*_args, **_kwargs):
            raise AssertionError("Panic may not read its attachment proof or wait for CLI")
        monkeypatch.setattr(daemon, "verify_owned_home", unavailable)
        monkeypatch.setattr(process_custody, "process_stop_snapshot", unavailable)
        monkeypatch.setattr(manager, "_request_operator_stop", unavailable)
        manager._lock.acquire()
        try:
            receipts = manager.panic_stop(request_only=True)
            assert receipts[0]["pid"] == proc.pid
            if platform.IS_MACOS:
                assert not receipts[0]["requested"] and "signalable identity" in receipts[0]["error"]
                assert proc.poll() is None
                platform.request_process_tree_kill(proc)  # ordinary owned child remains signalable
            else:
                assert receipts[0]["requested"]
            assert proc.wait(timeout=2) is not None
        finally:
            manager._lock.release()
        # The retained OS identity cannot follow an exited target into a new process.
        again = manager.panic_stop(request_only=True)
        assert not again[0]["requested"] and "error" in again[0]
    finally:
        platform.request_process_tree_kill(proc)
        proc.wait(timeout=5)


@pytest.mark.usefixtures("authenticated_home")
def test_unmeasured_attached_custody_never_becomes_signal_authority(tmp_path, monkeypatch):
    manager = daemon.OwnedClaudexorDaemon()
    proc = _child(tmp_path, daemon.CUSTODY_PURPOSE)
    try:
        row = process_custody.process_stop_snapshot(tmp_path, {daemon.CUSTODY_PURPOSE})[0]
        row["fingerprint"] = {"cmd_sha256": row["fingerprint"]["cmd_sha256"]}
        monkeypatch.setattr(process_custody, "process_stop_snapshot", lambda *_: [row])
        manager.ensure_running()
        receipts = manager.panic_stop(request_only=True)
        assert not receipts[0]["requested"] and "no measured" in receipts[0]["error"]
        assert proc.poll() is None
    finally:
        platform.request_process_tree_kill(proc)
        proc.wait(timeout=5)


def test_request_failures_are_receipts_not_clean_stops(monkeypatch):
    proc = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    # The mocked POSIX surface exists on every host; no numeric PID/group is ever really signalled.
    monkeypatch.setattr(platform.os, "getpgid", lambda _: (_ for _ in ()).throw(PermissionError("refused")),
                        raising=False)
    monkeypatch.setattr(platform.os, "killpg", lambda *_: pytest.fail("a refused lookup must not signal"),
                        raising=False)
    monkeypatch.setattr(platform.os, "kill", lambda *_: pytest.fail("a refused lookup must not signal"))
    receipt = platform.request_process_tree_kill(proc)
    assert receipt == {"pid": 123, "requested": False, "scope": "process", "error": "PermissionError: refused"}


def test_windows_job_refusal_still_requests_exact_handle_without_taskkill(monkeypatch):
    calls = []
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    monkeypatch.setattr(platform, "terminate_job", lambda _: "Job termination unconfirmed")
    monkeypatch.setitem(sys.modules, "_winapi", SimpleNamespace(
        TerminateProcess=lambda handle, code: calls.append((handle, code))))
    proc = SimpleNamespace(pid=123, _handle=987)
    receipt = platform.request_process_tree_kill(proc, job_handle=456)
    assert calls == [(987, 1)]
    assert receipt == {"pid": 123, "requested": True, "scope": "process", "error": "Job termination unconfirmed"}


@pytest.mark.skipif(not platform.IS_MACOS, reason="native Darwin exit watch")
def test_darwin_exit_watch_refusal_stays_invalid_after_repeated_requests(tmp_path):
    proc = _child(tmp_path, "watch-test")
    target = platform.capture_process_stop_target(proc.pid)
    try:
        proc.kill()
        proc.wait(timeout=2)
        first = platform.request_process_tree_kill(target)
        assert not first["requested"] and target["handle"].closed
        second = platform.request_process_tree_kill(target)
        assert not second["requested"]
    finally:
        platform.request_process_tree_kill(proc)
        proc.wait(timeout=5)
        target["handle"].close()


def test_failed_panic_control_write_is_reported_after_independent_steps(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr("supervisor.state.update_state", lambda *a, **k: False)
    monkeypatch.setattr("supervisor.evolution_lifecycle.record_evolution_stop_intent", lambda *a: called.append("intent"))
    monkeypatch.setattr("supervisor.evolution_lifecycle.complete_evolution_campaign", lambda *a, **k: called.append("campaign"))
    monkeypatch.setattr("ouroboros.post_task_evolution.drop_pending_request", lambda *a: called.append("pending"))
    with pytest.raises(OSError, match="unconfirmed"):
        server_control._persist_panic_controls(tmp_path)
    assert called == ["intent", "campaign", "pending"]


@pytest.mark.usefixtures("authenticated_home")
def test_panic_settlement_does_not_request_a_successor_daemon(tmp_path, monkeypatch):
    manager = daemon.OwnedClaudexorDaemon()
    original = _child(tmp_path, daemon.CUSTODY_PURPOSE)
    successor = None
    try:
        manager.ensure_running()
        receipt = manager.panic_stop(request_only=True)[0]
        if platform.IS_MACOS:
            assert not receipt["requested"]
            platform.request_process_tree_kill(original)
        else:
            assert receipt["requested"]
        original.wait(timeout=2)
        successor = _child(tmp_path, daemon.CUSTODY_PURPOSE)
        monkeypatch.setattr(manager, "_classify_liveness", lambda **_: (object(), "running", ""))
        monkeypatch.setattr(manager, "_request_operator_stop", lambda: pytest.fail("must not chase successor endpoint"))
        assert manager.stop_outcome() == "unconfirmed"
        assert successor.poll() is None
        retained = process_custody.process_stop_snapshot(tmp_path, {daemon.CUSTODY_PURPOSE})
        assert any(row["pid"] == successor.pid for row in retained)
    finally:
        for proc in (original, successor):
            if proc is not None:
                platform.request_process_tree_kill(proc)
                proc.wait(timeout=5)
