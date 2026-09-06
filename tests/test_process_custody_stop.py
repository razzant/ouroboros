"""Confirmed stop retains failed signals and concurrent ledger evidence."""
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

from ouroboros import process_custody as custody
from ouroboros import claudexor_daemon as daemon


def test_strict_identity_does_not_substitute_later_measurements(monkeypatch):
    row = {"pid": 123, "fingerprint": {"start_time_boot": "old.boot", "cmd_sha256": "old"}}
    monkeypatch.setattr(custody, "pid_is_alive", lambda _: True)
    monkeypatch.setattr(custody, "pid_is_zombie", lambda _: False)
    start = Mock(side_effect=["", "different.boot"])
    command = Mock(side_effect=["", "different"])
    monkeypatch.setattr(custody, "process_start_time", start)
    monkeypatch.setattr(custody, "_live_cmd_sha256", command)
    assert not custody._fingerprint_matches(row, require_measured=True)
    assert start.call_count == 1
    assert command.call_count == 0


def test_rewrite_preserves_concurrent_registration_and_opaque_bytes(tmp_path):
    first = {"pid": 123, "scope": "daemon", "purpose": "old"}
    successor = {"pid": 123, "scope": "daemon", "purpose": "new"}
    unrelated = {"pid": 456, "scope": "session"}
    opaque = b'{"unfinished":\xff\n'
    path = custody.ledger_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(json.dumps(first).encode() + b"\n" + opaque)
    assert custody.append_jsonl(path, successor)
    assert custody.append_jsonl(path, unrelated)
    custody._rewrite_ledger(tmp_path, [], previous=[first])
    raw = path.read_bytes()
    assert opaque in raw
    assert b'"purpose": "old"' not in raw
    assert custody._read_ledger(tmp_path) == [successor, unrelated]


def test_rewrite_without_append_lock_preserves_every_byte(tmp_path, monkeypatch):
    row = {"pid": 123}
    path = custody.ledger_path(tmp_path)
    path.parent.mkdir(parents=True)
    raw = json.dumps(row).encode() + b"\n"
    path.write_bytes(raw)
    monkeypatch.setattr("ouroboros.platform_layer.acquire_exclusive_file_lock", lambda *_a, **_kw: None)
    custody._rewrite_ledger(tmp_path, [], previous=[row])
    assert path.read_bytes() == raw


@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX group fixture")
def test_failed_stop_retains_live_identity_and_does_not_publish_stopped(tmp_path, monkeypatch):
    proc = custody.spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"], drive_root=tmp_path,
        purpose=daemon.CUSTODY_PURPOSE, scope="daemon",
    )
    try:
        before = custody.ledger_path(tmp_path).read_bytes()
        monkeypatch.setattr(custody, "kill_process_group_id", lambda _: None)
        monkeypatch.setattr("ouroboros.platform_layer.kill_pid_tree", lambda _: None)
        assert custody.stop_ledgered_processes(tmp_path, {daemon.CUSTODY_PURPOSE}, timeout_sec=0) == []
        assert proc.poll() is None
        assert custody.ledger_path(tmp_path).read_bytes() == before
        assert not (tmp_path / "logs" / "supervisor.jsonl").exists()
    finally:
        proc.kill()
        proc.wait(timeout=5)


@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX group fixture")
def test_successful_stop_only_prunes_the_observed_identity(tmp_path, monkeypatch):
    proc = custody.spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"], drive_root=tmp_path,
        purpose=daemon.CUSTODY_PURPOSE, scope="daemon",
    )
    path = custody.ledger_path(tmp_path)
    opaque = b'{"unfinished":\xff\n'
    with path.open("ab") as stream:
        stream.write(opaque)
    original_kill = custody.kill_process_group_id
    appended = {"pid": 99999999, "scope": "session", "purpose": "concurrent"}
    def kill_and_append(pgid):
        assert custody.append_jsonl(path, appended)
        original_kill(pgid)
    monkeypatch.setattr(custody, "kill_process_group_id", kill_and_append)
    try:
        assert custody.stop_ledgered_processes(tmp_path, {daemon.CUSTODY_PURPOSE}) == [proc.pid]
        proc.wait(timeout=5)
        assert custody._read_ledger(tmp_path) == [appended]
        assert opaque in path.read_bytes()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def test_unconfirmed_self_started_child_handle_is_retained(monkeypatch):
    proc = Mock(pid=123)
    proc.poll.return_value = None
    proc.wait.side_effect = subprocess.TimeoutExpired("fixture", 5)
    manager = daemon.OwnedClaudexorDaemon()
    manager._proc = proc
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_tree", lambda _: None)
    assert manager._terminate_child() is False
    assert manager._proc is proc


def test_unconfirmed_startup_child_cannot_be_replaced(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from ouroboros import claudexor_runtime, config
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    manager = daemon.OwnedClaudexorDaemon()
    manager._proc = Mock(pid=123)
    manager._proc.poll.return_value = None
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(manager, "_classify_liveness", lambda: (None, "stale", ""))
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(ensure=lambda: pytest.fail("runtime provisioning before child exit")))
    monkeypatch.setattr(custody, "spawn_supervised", lambda *_a, **_kw: pytest.fail("duplicate spawn"))
    with pytest.raises(ClaudexorUnavailable) as caught:
        manager.ensure_running()
    assert caught.value.code == "daemon_stop_unconfirmed"
    assert manager._proc.pid == 123


@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX detached harness fixture")
def test_stop_ledgered_process_ends_detached_harness_children(tmp_path):
    from ouroboros.platform_layer import force_kill_pid
    import time
    command = [sys.executable, "-c", (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); "
        "print(p.pid,flush=True); time.sleep(60)"
    )]
    proc = custody.spawn_supervised(command, drive_root=tmp_path,
        purpose=daemon.CUSTODY_PURPOSE, scope="daemon", stdout=subprocess.PIPE, text=True)
    child = int(proc.stdout.readline())
    try:
        assert custody.stop_ledgered_processes(tmp_path, {daemon.CUSTODY_PURPOSE}) == [proc.pid]
        proc.wait(timeout=5)
        deadline = time.monotonic() + 2
        while custody.pid_is_alive(child) and not custody.pid_is_zombie(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not custody.pid_is_alive(child) or custody.pid_is_zombie(child)
    finally:
        force_kill_pid(child)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        proc.stdout.close()
