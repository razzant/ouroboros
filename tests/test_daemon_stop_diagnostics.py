"""Panic reports partial custody; authenticated legacy attach repairs only its own marker."""
import http.server
import json
import logging
import os
import pathlib
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as daemon, process_custody as custody, config
from ouroboros.gateways.claudexor import ClaudexorUnavailable


def _rows(root):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def authenticated_home(tmp_path, monkeypatch):
    from ouroboros import claudexor_runtime
    from tests.test_claudexor_owned_daemon import _write_descriptor

    requests = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append(self.headers.get("Authorization"))
            ok = self.headers.get("Authorization") == "Bearer tok-owned"
            body = json.dumps({"compatible": True, "protocolMajor": 3,
                               "engine": {"version": "3.9.8", "sha": "a" * 40}}).encode()
            self.send_response(200 if ok else 401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    _write_descriptor(daemon.owned_config_dir(), port=server.server_port)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(
        pin=SimpleNamespace(version="3.9.8", build_sha="a" * 40)))
    try:
        yield server, requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


@pytest.mark.serial
def test_authenticated_legacy_attach_creates_marker_without_replacing_it(authenticated_home, tmp_path):
    server, requests = authenticated_home
    manager = daemon.OwnedClaudexorDaemon()
    assert not daemon.ownership_marker_path().exists()
    assert manager.ensure_running().port == server.server_port
    marker = daemon.ownership_marker_path()
    assert daemon.verify_owned_home(require_marker=True) == ""
    first = marker.read_bytes()
    assert requests == ["Bearer tok-owned"]
    assert manager.ensure_running().port == server.server_port
    assert marker.read_bytes() == first
    # Authenticated but unledgered is visible, and never PID/name/port kill authority.
    assert manager.stop() is False
    assert _rows(tmp_path)[-1]["type"] == "process_stop_unconfirmed"


@pytest.mark.serial
@pytest.mark.parametrize("marker", ["{broken", "[]", "{}", '{"owner":"other"}',
    '{"owner":"ouroboros","data_dir":"/other-install"}'])
def test_attach_refuses_existing_invalid_or_foreign_marker_before_handshake(authenticated_home, marker):
    _, requests = authenticated_home
    path = daemon.ownership_marker_path()
    path.write_text(marker)
    with pytest.raises(ClaudexorUnavailable, match="marker"):
        daemon.OwnedClaudexorDaemon().ensure_running()
    assert path.read_text() == marker
    assert requests == []


@pytest.mark.serial
def test_failed_authentication_does_not_create_marker(authenticated_home, monkeypatch):
    from ouroboros import claudexor_runtime
    _, requests = authenticated_home
    token = daemon.owned_descriptor_path().parent / "token"
    token.write_text("wrong-token")
    class MissingRuntime:
        def ensure(self):
            raise claudexor_runtime.ClaudexorRuntimeError("fixture_no_runtime", "not installed")
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: MissingRuntime())
    with pytest.raises(ClaudexorUnavailable, match="not installed"):
        daemon.OwnedClaudexorDaemon().ensure_running()
    assert requests == ["Bearer wrong-token"]
    assert not daemon.ownership_marker_path().exists()


def test_marker_creation_preserves_concurrent_foreign_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    original = pathlib.Path.open
    path = daemon.ownership_marker_path()
    foreign = '{"owner":"foreign","data_dir":"/foreign"}'
    def racing_open(self, mode="r", *args, **kwargs):
        if self == path and mode == "x":
            with original(self, "w") as out:
                out.write(foreign)
        return original(self, mode, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "open", racing_open)
    with pytest.raises(ClaudexorUnavailable):
        daemon._write_ownership_marker()
    assert path.read_text() == foreign


@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="POSIX measured custody fixture")
def test_panic_reports_missing_marker_and_preserves_live_process(tmp_path, monkeypatch, caplog):
    from tests.test_server_control_panic_daemon import _run_panic
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    proc = custody.spawn_supervised([sys.executable, "-c", "import time; time.sleep(60)"],
        drive_root=tmp_path, purpose=daemon.CUSTODY_PURPOSE, scope="daemon")
    manager = daemon.OwnedClaudexorDaemon()
    try:
        before = custody.ledger_path(tmp_path).read_bytes()
        with caplog.at_level(logging.CRITICAL):
            assert _run_panic(monkeypatch, tmp_path, daemon_stop=manager.stop)
        assert proc.poll() is None
        assert custody.ledger_path(tmp_path).read_bytes() == before
        assert "stop unconfirmed" in caplog.text
        rows = _rows(tmp_path)
        assert len(rows) == 1 and rows[0]["type"] == "process_stop_unconfirmed"
        assert "marker" in rows[0]["reason"]
    finally:
        proc.kill()
        proc.wait(timeout=5)


@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="POSIX measured custody fixture")
def test_partial_stop_is_false_and_preserves_unconfirmed_root(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    procs = [custody.spawn_supervised([sys.executable, "-c", "import time; time.sleep(60)"],
        drive_root=tmp_path, purpose=daemon.CUSTODY_PURPOSE, scope="daemon") for _ in range(2)]
    daemon._write_ownership_marker()
    manager = daemon.OwnedClaudexorDaemon()
    monkeypatch.setattr(manager, "_alive_endpoint", lambda **_kw: object())
    from ouroboros import platform_layer
    original = platform_layer.kill_pid_tree
    monkeypatch.setattr(platform_layer, "kill_pid_tree", lambda pid: original(pid) if pid == procs[0].pid else None)
    monkeypatch.setattr(custody, "kill_process_group_id", lambda _pgid: None)
    real_stop = custody.stop_ledgered_processes
    monkeypatch.setattr(custody, "stop_ledgered_processes", lambda root, purposes, **kw: real_stop(root, purposes, timeout_sec=0.05, **kw))
    try:
        with caplog.at_level(logging.CRITICAL):
            assert manager.stop() is False
        procs[0].wait(timeout=5)
        assert procs[1].poll() is None
        assert [r["pid"] for r in custody._read_ledger(tmp_path)] == [procs[1].pid]
        rows = _rows(tmp_path)
        assert [r["type"] for r in rows] == ["process_stopped", "process_stop_unconfirmed"]
        assert "stop unconfirmed" in caplog.text
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


def test_each_ledger_row_gets_its_own_exit_window(tmp_path, monkeypatch):
    clock = [0.0]
    rows = [{"pid": pid, "pgid": 0, "purpose": daemon.CUSTODY_PURPOSE, "scope": "daemon"} for pid in (111, 222)]
    custody._rewrite_ledger(tmp_path, rows)
    monkeypatch.setattr(custody.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(custody.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(custody, "_fingerprint_matches", lambda entry, require_measured=False:
        True if require_measured else clock[0] < (0.1 if entry["pid"] == 111 else 0.2))
    monkeypatch.setattr("ouroboros.platform_layer.collect_descendant_pids", lambda _: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_pid_tree", lambda _: None)
    assert custody.stop_ledgered_processes(tmp_path, {daemon.CUSTODY_PURPOSE}, timeout_sec=0.1) == [111, 222]
    assert custody._read_ledger(tmp_path) == []
    assert clock[0] == pytest.approx(0.2)


def test_stop_returns_on_lock_timeout_without_waiting_for_owner(tmp_path, monkeypatch, caplog):
    from ouroboros.gateways import claudexor
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(claudexor, "SHORT_POLL_TIMEOUT_SEC", 0.05)
    manager = daemon.OwnedClaudexorDaemon()
    manager._lock.acquire()
    results, done = [], threading.Event()
    def stop():
        results.append(manager.stop())
        done.set()
    thread = threading.Thread(target=stop)
    try:
        with caplog.at_level(logging.CRITICAL):
            thread.start()
            assert done.wait(2), "Stop must return while the other caller still holds the lock"
        assert manager._lock.locked()
        assert results == [False]
        assert "lock unavailable" in caplog.text
        assert _rows(tmp_path)[-1]["type"] == "process_stop_unconfirmed"
    finally:
        manager._lock.release()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_empty_stop_is_quiet(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert daemon.OwnedClaudexorDaemon().stop() is False
    assert not caplog.records
    assert _rows(tmp_path) == []


@pytest.mark.parametrize("raw", [b"{partial", b'{"pid":"unreadable","purpose":"claudexor_daemon"}\n'])
def test_unreadable_custody_is_disclosed_without_claiming_success(tmp_path, monkeypatch, caplog, raw):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    path = custody.ledger_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    with caplog.at_level(logging.CRITICAL):
        assert daemon.OwnedClaudexorDaemon().stop() is False
    assert path.read_bytes() == raw
    assert "ledger unreadable" in caplog.text


def test_panic_records_stop_exception_and_continues(tmp_path, monkeypatch):
    from tests.test_server_control_panic_daemon import _run_panic
    def fail():
        raise RuntimeError("fixture stop failed")
    assert _run_panic(monkeypatch, tmp_path, daemon_stop=fail)
    assert _rows(tmp_path)[-1] == {
        "ts": _rows(tmp_path)[-1]["ts"], "type": "process_stop_unconfirmed",
        "purpose": daemon.CUSTODY_PURPOSE, "reason": "stop raised RuntimeError",
    }


@pytest.mark.serial
@pytest.mark.skipif(os.name == "nt", reason="POSIX exact process identity")
def test_real_legacy_daemon_attach_then_stop_uses_token_and_ledger(tmp_path, monkeypatch):
    from ouroboros import claudexor_runtime
    from tests.test_claudexor_custody_lifetime import _DAEMON
    from tests.test_claudexor_owned_daemon import _write_descriptor

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    proc = custody.spawn_supervised(
        [sys.executable, "-u", "-c", _DAEMON.replace("fixture-token", "tok-owned")],
        drive_root=tmp_path, purpose=daemon.CUSTODY_PURPOSE, scope="session",
        stdout=subprocess.PIPE, text=True,
    )
    try:
        port = int(proc.stdout.readline())
        _write_descriptor(daemon.owned_config_dir(), port=port)
        monkeypatch.setattr(custody, "_SESSION_ID", "new-generation")
        monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(
            pin=SimpleNamespace(version="3.9.8", build_sha="a" * 40)))
        manager = daemon.OwnedClaudexorDaemon()
        assert manager._proc is None
        assert manager.ensure_running().port == port
        assert daemon.verify_owned_home(require_marker=True) == ""
        assert manager.stop() is True
        proc.wait(timeout=5)
        assert custody._read_ledger(tmp_path) == []
        assert [row["type"] for row in _rows(tmp_path)] == ["process_stopped"]
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        proc.stdout.close()
