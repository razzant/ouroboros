"""A dead service leader is not a writer; its living group members still are."""

import os
import subprocess
import sys
import time
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
    real_isdir = process_containment.os.path.isdir
    monkeypatch.setattr(process_containment.os.path, "isdir", lambda path: False if path == "/proc" else real_isdir(path))
    monkeypatch.setattr(process_containment._pl, "process_group_is_alive", lambda _: True)
    monkeypatch.setattr(process_containment.subprocess, "run", lambda *_a, **_k: SimpleNamespace(
        returncode=0, stdout=states))
    assert process_containment.process_group_has_live_members(123) is expected


def test_unreadable_service_group_is_not_proven_quiet(monkeypatch):
    real_isdir = process_containment.os.path.isdir
    monkeypatch.setattr(process_containment.os.path, "isdir", lambda path: False if path == "/proc" else real_isdir(path))
    monkeypatch.setattr(process_containment._pl, "process_group_is_alive", lambda _: True)
    monkeypatch.setattr(process_containment.subprocess, "run", lambda *_a, **_k: SimpleNamespace(
        returncode=1, stdout=""))
    assert process_containment.process_group_has_live_members(123) is True


@pytest.mark.parametrize("census,expected", [(None, True), ({}, True), ({123: True}, True), ({123: False}, False)])
def test_linux_owner_uses_confirming_census_without_spawning(monkeypatch, census, expected):
    pc = process_containment
    real_isdir = pc.os.path.isdir
    monkeypatch.setattr(pc.os.path, "isdir", lambda path: True if path == "/proc" else real_isdir(path))
    monkeypatch.setattr(pc._pl, "IS_WINDOWS", False)
    monkeypatch.setattr(pc._pl, "process_group_is_alive", lambda _: True)
    monkeypatch.setattr(pc, "process_groups_have_live_members", lambda groups: census if groups == {123} else pytest.fail())
    monkeypatch.setattr(pc.subprocess, "run", lambda *_a, **_k: pytest.fail("Linux needs no ps child"))
    assert pc.process_group_has_live_members(123) is expected


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
        deadline = time.monotonic() + 5
        while process_containment.process_group_has_live_members(leader.pid) is not live_child:
            assert time.monotonic() < deadline, "group census never converged"
            time.sleep(0.01)
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
    assert process_custody.append_jsonl(process_custody.ledger_path(tmp_path), entry)
    monkeypatch.setattr(process_custody, "pid_is_alive", lambda _: True)
    monkeypatch.setattr(process_custody, "pid_is_zombie", lambda _: True)
    monkeypatch.setattr(process_custody, "process_group_has_live_members", lambda _: False)
    survivors = []
    monkeypatch.setattr(process_custody, "_rewrite_ledger", lambda _, rows, **_kw: survivors.extend(rows))
    assert process_custody.reap_orphaned_processes(tmp_path) == []
    assert survivors == []


@pytest.mark.parametrize("live_or_unknown", [False, True])
def test_test_group_cleanup_requires_positive_quiet_census(monkeypatch, live_or_unknown):
    from ouroboros import platform_layer
    from tests._shared import reap_test_process_group

    signals, waits, inspected = [], [], []
    # The existing best-effort primitive has no success result, including on
    # EPERM. Model that contract without replacing the host's os.killpg.
    monkeypatch.setattr(platform_layer, "kill_process_group_id", signals.append)
    def census(pgid):
        inspected.append(pgid)
        return live_or_unknown
    monkeypatch.setattr(process_containment, "process_group_has_live_members", census)
    supervisor = SimpleNamespace(pid=123, wait=lambda **kw: waits.append(kw))
    if live_or_unknown:
        with pytest.raises(AssertionError, match="live or unknown members"):
            reap_test_process_group(supervisor, timeout_sec=0)
    else:
        reap_test_process_group(supervisor, timeout_sec=0)
    assert signals == inspected == [123]
    assert waits == [{"timeout": 0}]


@pytest.mark.skipif(os.name == "nt", reason="test-owned POSIX process group")
def test_test_group_cleanup_reaps_live_helper(tmp_path):
    from tests._shared import reap_test_process_group

    # The helper shares the new session/group but is not our direct child.
    script = (
        "import subprocess,sys,time\n"
        "helper=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        "print(helper.pid,flush=True)\n"
        "time.sleep(60)\n"
    )
    supervisor = subprocess.Popen([sys.executable, "-c", script], start_new_session=True,
                                  stdout=subprocess.PIPE, text=True)
    try:
        helper_pid = int(supervisor.stdout.readline())
        assert process_containment.process_group_has_live_members(supervisor.pid)
        reap_test_process_group(supervisor)
        assert not process_containment.process_group_has_live_members(supervisor.pid)
        assert not process_custody.pid_is_alive(helper_pid) or process_containment.pid_is_zombie(helper_pid)
    finally:
        reap_test_process_group(supervisor)
        supervisor.stdout.close()
