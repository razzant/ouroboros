"""Confirmed stop retains failed signals and concurrent ledger evidence."""
import json
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from ouroboros import process_custody as custody
from ouroboros import claudexor_daemon as daemon


def test_reaper_prunes_superseded_pid_rows_in_one_sweep(tmp_path):
    pid = 2**31 - 1
    assert not custody.pid_is_alive(pid)
    first = {"pid": pid, "pgid": 0, "scope": "session", "purpose": "old"}
    latest = {**first, "purpose": "latest"}
    path = custody.ledger_path(tmp_path)
    assert custody.append_jsonl(path, first)
    assert custody.append_jsonl(path, latest)
    assert custody._read_ledger(tmp_path) == [latest]
    assert custody.reap_orphaned_processes(tmp_path) == []
    assert path.read_bytes() == b""


@pytest.mark.parametrize("purpose", ["latest", "service:\u0085name", "service:\u2028name", "service:\u2029name"])
def test_reaper_compacts_duplicate_rows_without_losing_live_survivor(tmp_path, purpose):
    first = {"pid": os.getpid(), "pgid": 0, "scope": "session", "purpose": "old",
             "session_id": custody.current_custody_session_id()}
    latest = {**first, "purpose": purpose}
    path = custody.ledger_path(tmp_path)
    for row in (first, latest, latest):
        assert custody.append_jsonl(path, row)
    assert custody._read_ledger_strict(tmp_path) == (True, [latest])
    assert custody.reap_orphaned_processes(tmp_path) == []
    assert [json.loads(line) for line in path.read_bytes().splitlines()] == [latest]
    assert custody._read_ledger(tmp_path) == [latest]


@pytest.mark.parametrize("identical", [False, True])
def test_reaper_preserves_same_pid_append_and_opaque_bytes(tmp_path, monkeypatch, identical):
    pid = 2**31 - 1
    assert not custody.pid_is_alive(pid)
    first = {"pid": pid, "pgid": 0, "scope": "session", "purpose": "old"}
    latest = {**first, "purpose": "latest"}
    appended = latest if identical else {**latest, "purpose": "concurrent"}
    path = custody.ledger_path(tmp_path)
    for row in (first, latest):
        assert custody.append_jsonl(path, row)
    opaque = b'{"unfinished":\xff\n[]\n{"pid":"not-a-pid"}\n'
    with path.open("ab") as stream:
        stream.write(opaque)
    observed_end = path.stat().st_size
    original = custody._fingerprint_matches
    appended_bytes = []

    def check_and_append(entry):
        assert entry == latest
        assert custody.append_jsonl(path, appended)
        appended_bytes.append(path.read_bytes()[observed_end:])
        return original(entry)

    monkeypatch.setattr(custody, "_fingerprint_matches", check_and_append)
    before = path.read_bytes()
    assert custody.reap_orphaned_processes(tmp_path) == []
    # Preserve the writer's actual bytes, including native text-mode newlines.
    expected_tail, = appended_bytes
    assert json.loads(expected_tail) == appended
    assert len(before) == observed_end
    assert path.read_bytes() == opaque + expected_tail
    assert custody._read_ledger(tmp_path) == [appended]


def test_reaper_defers_compaction_after_a_concurrent_prefix_rewrite(tmp_path, monkeypatch):
    pid = 2**31 - 1
    assert not custody.pid_is_alive(pid)
    first = {"pid": pid, "pgid": 0, "scope": "session", "purpose": "old"}
    latest = {**first, "purpose": "latest"}
    path = custody.ledger_path(tmp_path)
    for row in (first, latest):
        assert custody.append_jsonl(path, row)
    original = custody._fingerprint_matches
    rewritten = []

    def check_and_rewrite(entry):
        custody._rewrite_ledger(tmp_path, [latest])
        rewritten.append(path.read_bytes())
        return original(entry)

    monkeypatch.setattr(custody, "_fingerprint_matches", check_and_rewrite)
    assert custody.reap_orphaned_processes(tmp_path) == []
    assert path.read_bytes() == rewritten[0]
    assert custody._read_ledger(tmp_path) == [latest]


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
    previous = path.read_bytes()
    assert custody.append_jsonl(path, successor)
    assert custody.append_jsonl(path, unrelated)
    custody._rewrite_ledger(tmp_path, [], previous=previous)
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
    custody._rewrite_ledger(tmp_path, [], previous=raw)
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
        monkeypatch.setattr("ouroboros.platform_layer.force_kill_pid", lambda _: None)
        assert custody.stop_ledgered_processes(tmp_path, {daemon.CUSTODY_PURPOSE}, timeout_sec=0) == []
        assert proc.poll() is None
        assert custody.ledger_path(tmp_path).read_bytes() == before
        assert not (tmp_path / "logs" / "supervisor.jsonl").exists()
    finally:
        proc.kill()
        proc.wait(timeout=5)


@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX group fixture")
@pytest.mark.parametrize("purpose", ["mcp_task_session:held", "service:held"])
def test_reused_group_number_cannot_signal_a_real_foreign_process(tmp_path, monkeypatch, purpose):
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=True)
    try:
        row = {"pid": foreign.pid, "pgid": foreign.pid, "scope": "task",
               "purpose": purpose, "owner_task": "gone", "session_id": "old-boot",
               "fingerprint": {"start_time": "old boot", "cmd_sha256": "old command"}}
        assert custody.append_jsonl(custody.ledger_path(tmp_path), row)
        signals = []
        monkeypatch.setattr(custody, "kill_process_group_id", lambda pgid, **kw: signals.append(pgid))
        unconfirmed = []
        assert custody.stop_group_custody(tmp_path, lambda _: True, timeout_sec=0,
                                          unconfirmed=unconfirmed) == []
        assert unconfirmed and "identity unconfirmed" in unconfirmed[0]
        if purpose.startswith("service:"):
            quiet, blockers = custody.quiesce_custodied_services(tmp_path, timeout_sec=0)
            assert quiet is False and blockers == [f"custody_service:{foreign.pid}:identity_unconfirmed"]
        assert custody.reap_orphaned_processes(tmp_path, running_task_ids=set()) == []
        assert signals == [] and foreign.poll() is None
        assert custody._read_ledger(tmp_path) == [row]
    finally:
        foreign.kill()
        foreign.wait(timeout=5)


@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX group fixture")
def test_reaper_retains_group_when_signal_helper_swallows_failure(tmp_path, monkeypatch):
    proc = custody.spawn_supervised(
        [sys.executable, "-c", "import time; time.sleep(60)"], drive_root=tmp_path,
        purpose="mcp_task_session:held", scope="task", owner_task_id="gone")
    try:
        before = custody._read_ledger(tmp_path)
        monkeypatch.setattr(custody, "kill_process_group_id", lambda *a, **kw: None)
        assert custody.reap_orphaned_processes(tmp_path, running_task_ids=set()) == []
        assert proc.poll() is None
        assert custody._read_ledger(tmp_path) == before
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
        manager.ensure_running(startup_wait_sec=0)
    assert caught.value.code == "daemon_starting"
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
