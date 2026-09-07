"""A dead service leader is not a writer; its living group members still are."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import process_containment, process_custody
from ouroboros.platform_layer import force_kill_pid


@pytest.mark.parametrize("states,expected", [
    ("123 Z\n123 Z+\n", False),
    ("123 Z\n123 S\n", True),
    ("123 D\n", True),
    ("123 T\n", True),
    ("123\n", True),
    ("999 S\n", True),
])
def test_service_group_liveness_inspects_every_member(monkeypatch, states, expected):
    monkeypatch.setattr(process_containment._pl, "process_group_is_alive", lambda _: True)
    monkeypatch.setattr(process_containment.subprocess, "run", lambda *_a, **_k: SimpleNamespace(
        returncode=0, stdout=states))
    assert process_containment.process_group_has_live_members(123) is expected


def test_unreadable_service_group_is_not_proven_quiet(monkeypatch):
    monkeypatch.setattr(process_containment._pl, "process_group_is_alive", lambda _: True)
    monkeypatch.setattr(process_containment.subprocess, "run", lambda *_a, **_k: SimpleNamespace(
        returncode=1, stdout=""))
    assert process_containment.process_group_has_live_members(123) is True


@pytest.mark.serial
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux waitid WNOWAIT zombie fixture")
@pytest.mark.parametrize("live_child", [False, True])
def test_update_quiesces_zombie_leader_without_losing_live_child(tmp_path, live_child):
    script = (
        "import subprocess,sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL) if sys.argv[1] == 'yes' else None\n"
        "print(child.pid if child else 0, flush=True)\n"
        "sys.stdin.read()\n"
    )
    leader = process_custody.spawn_supervised(
        [sys.executable, "-c", script, "yes" if live_child else "no"],
        drive_root=tmp_path, purpose="service:zombie-fixture", scope="session",
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    child_pid = 0
    try:
        child_pid = int(leader.stdout.readline())
        leader.stdin.close()
        # Wait for EXIT without reaping: no timing guess, and the OS table
        # deliberately still contains the zombie that used to block updates.
        os.waitid(os.P_PID, leader.pid, os.WEXITED | os.WNOWAIT)
        assert process_containment.pid_is_zombie(leader.pid)
        assert process_containment.process_group_has_live_members(leader.pid) is live_child
        assert process_custody.quiesce_custodied_services(tmp_path) == (True, [])
        assert process_custody._read_ledger(tmp_path) == []
        assert not process_containment.process_group_has_live_members(leader.pid)
    finally:
        if child_pid:
            force_kill_pid(child_pid)
        if not leader.stdin.closed:
            leader.stdin.close()
        leader.wait(timeout=10)
        leader.stdout.close()


def test_same_generation_zombie_record_is_pruned(monkeypatch, tmp_path):
    entry = {"pid": 123, "pgid": 123, "purpose": "service:exited", "scope": "session",
             "session_id": process_custody.current_custody_session_id()}
    monkeypatch.setattr(process_custody, "_read_ledger", lambda _: [entry])
    monkeypatch.setattr(process_custody, "pid_is_alive", lambda _: True)
    monkeypatch.setattr(process_custody, "pid_is_zombie", lambda _: True)
    monkeypatch.setattr(process_custody, "process_group_has_live_members", lambda _: False)
    survivors = []
    monkeypatch.setattr(process_custody, "_rewrite_ledger", lambda _, rows, **_kw: survivors.extend(rows))
    assert process_custody.reap_orphaned_processes(tmp_path) == []
    assert survivors == []
