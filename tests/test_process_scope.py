"""Scope markers, recorded groups and the one census/reap (issue #1772, Directive 1).

Real children are confined to this test process's own spawns and ended by exact pid;
nothing here reads or signals another installation. The macOS platform-binary case is
the incident shape: ``/bin/sleep`` shows no environment under ``ps -E`` with SIP, so it
is ended through its recorded group, while a node child is ended through the marker.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from ouroboros import platform_layer as _pl
from ouroboros import process_census as pc
from ouroboros import process_scope as ps

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the environment census is POSIX-only")


def _sleeper(env=None, *, argv=None, new_session=True) -> subprocess.Popen:
    env = {**os.environ, **(env or {})}
    return subprocess.Popen(argv or [sys.executable, "-c", "import time; time.sleep(120)"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                            start_new_session=new_session)


def _end(*procs: subprocess.Popen) -> None:
    for proc in procs:
        if proc.poll() is None:
            try:
                os.kill(proc.pid, 9)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not ps.measure_birth(pid):
            return True
        time.sleep(0.05)
    return not ps.measure_birth(pid)


def _group(proc: subprocess.Popen) -> dict:
    return {"pgid": os.getpgid(proc.pid), "leader_pid": proc.pid, "leader_birth": ps.measure_birth(proc.pid)}


# --------------------------------------------------------------------------- markers

def test_marker_grammar_and_roles():
    for prefix in (ps.INSTALL_SCOPE_PREFIX, ps.TASK_SCOPE_PREFIX, ps.RUN_SCOPE_PREFIX):
        name = ps.mint_marker(prefix)
        assert ps.is_scope_name(name) and name.startswith(prefix) and len(name) == len(prefix) + 32
    assert ps.mint_marker(ps.TASK_SCOPE_PREFIX) != ps.mint_marker(ps.TASK_SCOPE_PREFIX)
    assert not ps.is_scope_name("OURO_TASK_SCOPE_/tmp/root")  # a path is never a marker
    assert not ps.is_scope_name("OURO_PROC_CONTAINER_" + "a" * 32)  # the gate token is another grammar
    with pytest.raises(ValueError):
        ps.mint_marker("lower_scope_")


def test_shedding_keeps_the_installation_marker_and_drops_shorter_lived_ones():
    install, task, run = (ps.mint_marker(p) for p in (ps.INSTALL_SCOPE_PREFIX, ps.TASK_SCOPE_PREFIX, ps.RUN_SCOPE_PREFIX))
    env = {install: "1", task: "1", run: "1", "PATH": "/bin", "OURO_PROC_CONTAINER_" + "f" * 32: "1"}
    shed = ps.shed_scopes(env)
    assert install in shed and task not in shed and run not in shed
    assert shed["PATH"] == "/bin" and "OURO_PROC_CONTAINER_" + "f" * 32 in shed  # non-grammar keys untouched
    assert ps.scope_entries(env) == {install: "1", task: "1", run: "1"}
    assert ps.installation_scope_entries(env) == {install: "1"}
    assert ps.scope_entries({task: "0"}) == {}  # the value is exactly "1"


def test_installation_marker_is_persisted_stable_and_unique_per_root(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert ps.installation_marker(a, create=False) == ""
    first = ps.installation_marker(a)
    assert first.startswith(ps.INSTALL_SCOPE_PREFIX) and ps.installation_marker(a) == first
    record = json.loads(ps.installation_scope_path(a).read_text(encoding="utf-8"))
    assert record["marker"] == first and record["created_at"]
    assert ps.installation_marker(b) != first
    env: dict = {}
    assert ps.plant_installation_scope(a, env) == first and env == {first: "1"}


# --------------------------------------------------------------------------- census and reap

def test_census_reads_pid_pgid_birth_and_visible_markers():
    marker = ps.mint_marker(ps.TASK_SCOPE_PREFIX)
    child = _sleeper({marker: "1"})
    try:
        rows = ps.census([marker])
        assert rows is not None and child.pid in rows
        row = rows[child.pid]
        assert row.pgid == child.pid and row.ppid == os.getpid()
        assert row.birth == ps.measure_birth(child.pid) and row.epoch > 0
        assert marker in (row.markers or set()), "a non-platform child's marker is visible"
        assert marker not in (rows[os.getpid()].markers or set())
    finally:
        _end(child)


@pytest.mark.skipif(not _pl.IS_MACOS, reason="native macOS process observation")
def test_on_macos_a_platform_binary_keeps_its_identity_with_either_marker_visibility():
    marker = ps.mint_marker(ps.TASK_SCOPE_PREFIX)
    child = _sleeper({marker: "1"}, argv=["/bin/sleep", "120"])
    try:
        rows = ps.census([marker])
        assert rows is not None and child.pid in rows
        row = rows[child.pid]
        assert row.pgid == child.pid and row.birth == ps.measure_birth(child.pid)
        assert row.markers in (set(), {marker})  # visibility depends on this host's protection policy
    finally:
        _end(child)


def test_end_scope_ends_a_platform_binary_through_its_group_and_a_node_child_through_the_marker(tmp_path):
    node = _pl.resolve_bundled_node()
    if not node:
        pytest.skip("no bundled node for the marker-visible child")
    marker = ps.mint_marker(ps.TASK_SCOPE_PREFIX)
    # The incident shape of a task's own command: a shell in its own group starts a
    # background /bin/sleep and exits; the sleep stays in the recorded group.
    shell = _sleeper({marker: "1"}, argv=["/bin/sh", "-c", "/bin/sleep 120 >/dev/null 2>&1 & /bin/sleep 1"])
    group = _group(shell)  # recorded while the leader lives, as the spawn seams do
    shell.wait(timeout=5)  # the shell is gone; its background child keeps the group
    rows = ps.census([marker])
    orphan = [row.pid for row in rows.values() if row.pgid == group["pgid"] and row.pid != shell.pid]
    assert len(orphan) == 1, "the background sleep stays in the exited shell's group"
    orphan_pid = orphan[0]
    # A node child that left the group (setsid) but carries the marker.
    runaway = _sleeper({marker: "1"}, argv=[node, "-e", "setTimeout(() => {}, 120000)"])
    bystander = _sleeper({ps.mint_marker(ps.TASK_SCOPE_PREFIX): "1"})
    try:
        outcome = ps.end_scope(marker=marker, groups=[group], exclude={os.getpid(), 1}, grace_sec=0.5)
        assert outcome["coverage"] == ps.COVERAGE_POSIX and not outcome["census_failed"]
        assert set(outcome["ended"]) == {orphan_pid, runaway.pid}
        assert outcome["survivors"] == [] and outcome["groups_reused"] == []
        assert _gone(orphan_pid) and _gone(runaway.pid)
        assert bystander.poll() is None, "another scope's child is untouched"
    finally:
        _end(runaway, bystander)
        try:
            os.kill(orphan_pid, 9)
        except ProcessLookupError:
            pass


def test_a_reused_or_emptied_group_is_never_signalled():
    bystander = _sleeper()
    try:
        reused = {"pgid": os.getpgid(bystander.pid), "leader_pid": bystander.pid,
                  "leader_birth": "Mon Jan  1 00:00:00 2001"}  # not the recorded leader: the id was reused
        gone = {"pgid": 2 ** 22 - 7, "leader_pid": 2 ** 22 - 7, "leader_birth": "Mon Jan  1 00:00:00 2001"}
        outcome = ps.end_scope(groups=[reused, gone], exclude={os.getpid(), 1})
        assert outcome["groups_reused"] == [reused["pgid"]] and outcome["groups_gone"] == [gone["pgid"]]
        assert outcome["ended"] == [] and bystander.poll() is None
        # The same group with its true leader birth IS this scope's and ends member by member.
        true = _group(bystander)
        outcome = ps.end_scope(groups=[true], exclude={os.getpid(), 1}, grace_sec=0.2)
        assert outcome["ended"] == [bystander.pid] and _gone(bystander.pid)
    finally:
        _end(bystander)


def test_a_member_whose_identity_changed_is_not_signalled(monkeypatch):
    child = _sleeper()
    try:
        group = _group(child)
        real = ps.read_identities

        def reborn(pids):
            found = real(pids)
            if child.pid in found:  # the pid names another process now (its birth differs)
                found[child.pid] = ps._Identity(ps.IDENTITY_ALIVE, "Mon Jan  1 00:00:00 2001", child.pid)
            return found

        monkeypatch.setattr(ps, "read_identities", reborn)
        outcome = ps.end_scope(groups=[group], exclude={os.getpid(), 1}, grace_sec=0.2)
        assert outcome["ended"] == [] and child.poll() is None, "a re-measured birth that differs refuses the signal"
        assert outcome["already_gone"] == [child.pid], "confirmed reused: not ours any more"
    finally:
        _end(child)


def test_an_unreadable_table_is_unknown_never_an_empty_scope(monkeypatch):
    child = _sleeper()
    try:
        monkeypatch.setattr(ps, "census", lambda markers=(): None)
        outcome = ps.end_scope(groups=[_group(child)], exclude={os.getpid(), 1})
        assert outcome["census_failed"] is True and outcome["ended"] == [] and child.poll() is None
    finally:
        _end(child)


def test_windows_reports_unsupported_coverage(monkeypatch):
    monkeypatch.setattr(_pl, "IS_WINDOWS", True)
    outcome = ps.end_scope(marker=ps.mint_marker(ps.TASK_SCOPE_PREFIX))
    assert outcome["coverage"] == ps.COVERAGE_UNSUPPORTED and outcome["ended"] == [] and not outcome["census_failed"]
    assert ps.census([]) is None


# --------------------------------------------------------------------------- task scope records

def test_task_scope_marker_is_minted_once_and_rotated_only_after_a_cleanup_started(tmp_path):
    first = ps.task_scope_marker(tmp_path, "t1")
    assert first.startswith(ps.TASK_SCOPE_PREFIX) and ps.task_scope_marker(tmp_path, "t1") == first
    assert ps.task_scope_env(tmp_path, "t1") == {first: "1"}
    assert ps.task_scope_env(tmp_path, "") == {} and ps.task_scope_env(None, "t1") == {}
    record = ps.task_scope_record(tmp_path, "t1")
    assert record["task_id"] == "t1" and record["groups"] == [] and record["minted_epoch"] > 0
    path = ps.task_scope_path(tmp_path, "t1")
    path.write_text(json.dumps({**record, "groups": [{"pgid": 7, "leader_pid": 7, "leader_birth": "x"}],
                                "cleanup": {"state": "running", "started_epoch": time.time()}}), encoding="utf-8")
    rotated = ps.task_scope_marker(tmp_path, "t1")
    assert rotated != first, "a successor's children never share an ending attempt's id"
    assert ps.task_scope_record(tmp_path, "t1")["groups"] == [{"pgid": 7, "leader_pid": 7, "leader_birth": "x"}], \
        "the groups the old record still owed are carried"
    path.write_text(json.dumps({**ps.task_scope_record(tmp_path, "t1"), "known_members": [{"pid": 9, "birth": "y"}],
                                "cleanup": {"state": "pending", "reason": "unverified"}}), encoding="utf-8")
    assert ps.task_scope_marker(tmp_path, "t1") not in (first, rotated)
    assert ps.task_scope_record(tmp_path, "t1")["known_members"] == [{"pid": 9, "birth": "y"}], "known members are carried too"
    assert ps.task_scope_records(tmp_path) == [ps.task_scope_record(tmp_path, "t1")]


def test_record_task_group_keeps_live_groups_and_prunes_emptied_ones(tmp_path):
    live, short = _sleeper(), _sleeper()
    try:
        entry = ps.record_task_group(tmp_path, "t2", short.pid)
        assert entry["pgid"] == short.pid and entry["leader_birth"] == ps.measure_birth(short.pid)
        _end(short)  # the group empties
        ps.record_task_group(tmp_path, "t2", live.pid)
        groups = ps.task_scope_record(tmp_path, "t2")["groups"]
        assert [group["pgid"] for group in groups] == [live.pid], "an emptied group is pruned on the next append"
        assert ps.record_task_group(tmp_path, "", live.pid) is None
    finally:
        _end(live, short)


# --------------------------------------------------------------------------- fix round 1

def test_a_leader_that_already_exited_still_gets_its_group_recorded_and_its_child_ended(tmp_path):
    """Directive 1 / Fable finding 1: the seam records AFTER Popen, and a shell such as
    ``sh -c "nohup X & echo $!"`` exits in milliseconds; a zombie refuses ``getpgid`` on
    macOS. The group is recorded from the zombie-visible table (or structurally, pgid ==
    pid, for a leader already reaped), and the task end still reaches the ``&`` child."""
    launches = []
    for _ in range(2):
        shell = subprocess.Popen(["/bin/sh", "-c", "/bin/sleep 120 >/dev/null 2>&1 & echo $!"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True, start_new_session=True)
        launches.append((shell, int(shell.stdout.read().strip().splitlines()[-1])))
    zombie_shell, zombie_child = launches[0]
    reaped_shell, reaped_child = launches[1]
    try:
        time.sleep(0.3)  # the shells exited; the first stays a zombie (unwaited), the second is reaped below
        reaped_shell.wait(timeout=5)
        zombie = ps.record_task_group(tmp_path, "t-zombie", zombie_shell.pid)
        assert zombie is not None and zombie["pgid"] == zombie_shell.pid and zombie["leader_birth"], \
            "a zombie leader's group and birth are still read from the table"
        reaped = ps.record_task_group(tmp_path, "t-reaped", reaped_shell.pid)
        assert reaped is not None and reaped["pgid"] == reaped_shell.pid and reaped["leader_birth"] == "", \
            "a reaped leader's group is structural (it led the group that bears its pid)"
        for group, child in ((zombie, zombie_child), (reaped, reaped_child)):
            outcome = ps.end_scope(groups=[group], exclude={os.getpid(), 1}, grace_sec=0.5)
            assert outcome["ended"] == [child] and outcome["settled"] is True, outcome
            assert _gone(child)
    finally:
        _end(zombie_shell, reaped_shell)
        for _shell, child in launches:
            try:
                os.kill(child, 9)
            except ProcessLookupError:
                pass


def test_a_denied_or_failed_signal_is_unverified_never_gone_and_only_denied_is_final(monkeypatch):
    for result, reason, final in ((_pl.SIGNAL_DENIED, ps.UNVERIFIED_DENIED, True), (_pl.SIGNAL_FAILED, ps.UNVERIFIED_FAILED, False)):
        child = _sleeper()
        try:
            group = _group(child)
            monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: result)
            outcome = ps.end_scope(groups=[group], exclude={os.getpid(), 1}, grace_sec=0.2)
            assert outcome["ended"] == [] and outcome["already_gone"] == [] and outcome["survivors"] == []
            assert outcome["unverified"] == [{"pid": child.pid, "birth": group["leader_birth"], "reason": reason}]
            assert outcome["settled"] is final, "EPERM is final and disclosed; a failed signal is retried"
            assert child.poll() is None
        finally:
            monkeypatch.undo()
            _end(child)


def test_an_unreadable_identity_is_unverified_and_never_signalled(monkeypatch):
    child = _sleeper()
    try:
        group = _group(child)
        real = ps.read_identities
        monkeypatch.setattr(ps, "read_identities", lambda pids: {
            **real(pids), **({child.pid: ps._Identity(ps.IDENTITY_UNREADABLE)} if child.pid in set(pids) else {})})
        signalled = []
        monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: signalled.append(pid) or _pl.SIGNAL_SENT)
        outcome = ps.end_scope(groups=[group], exclude={os.getpid(), 1}, grace_sec=0.2)
        assert signalled == [], "no identity, no signal"
        assert outcome["unverified"] == [{"pid": child.pid, "birth": group["leader_birth"], "reason": ps.UNVERIFIED_IDENTITY}]
        assert outcome["ended"] == [] and outcome["settled"] is False and child.poll() is None
    finally:
        _end(child)


def test_an_identity_that_turns_unreadable_after_the_signal_is_unverified_not_ended(monkeypatch):
    child = _sleeper()
    try:
        group = _group(child)
        real = ps.read_identities
        calls = []

        def flaky(pids):
            calls.append(set(pids))
            found = real(pids)
            if len(calls) > 1 and child.pid in found:  # the reader stalls once the signal is out
                found[child.pid] = ps._Identity(ps.IDENTITY_UNREADABLE)
            return found

        monkeypatch.setattr(ps, "read_identities", flaky)
        monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: _pl.SIGNAL_SENT)  # accepted, nothing dies
        outcome = ps.end_scope(groups=[group], exclude={os.getpid(), 1}, grace_sec=0.2)
        assert outcome["ended"] == [] and outcome["survivors"] == []
        assert [item["reason"] for item in outcome["unverified"]] == [ps.UNVERIFIED_IDENTITY]
        assert outcome["settled"] is False and child.poll() is None
    finally:
        _end(child)


def test_a_captured_pid_missing_from_a_partial_census_is_asked_by_identity_never_called_gone(monkeypatch):
    """Astra delta 3: Linux omits the row of a ``/proc/<pid>/stat`` it could not read; the
    known pid is then read by the typed reader — unreadable stays unverified, confirmed
    absence or another birth discharges it, the same birth alive is signalled."""
    birth = "Sat Oct 10 20:10:23 2026"
    monkeypatch.setattr(ps, "census", lambda markers=(): {4000: ps._Row(4000, 1, 4000, birth, 1.0)})  # no row for 4242
    signalled = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: signalled.append(pid) or _pl.SIGNAL_SENT)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    answers = {}
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        int(pid): answers.get(int(pid), ps._Identity(pc.IDENTITY_ABSENT)) for pid in pids})
    captured = [{"pid": 4242, "birth": birth}]
    answers[4242] = ps._Identity(ps.IDENTITY_UNREADABLE)
    outcome = ps.end_scope(captured=captured, exclude={os.getpid(), 1}, grace_sec=0.0)
    assert outcome["unverified"] == [{"pid": 4242, "birth": birth, "reason": ps.UNVERIFIED_IDENTITY}]
    assert outcome["captured_gone"] == [] and outcome["settled"] is False and signalled == []
    answers[4242] = ps._Identity(ps.IDENTITY_ALIVE, "Mon Jan  1 00:00:00 2001", 4242)  # another process now
    outcome = ps.end_scope(captured=captured, exclude={os.getpid(), 1}, grace_sec=0.0)
    assert outcome["captured_gone"] == [4242] and outcome["settled"] is True and signalled == []
    answers[4242] = ps._Identity(ps.IDENTITY_ALIVE, birth, 4242)  # ours, alive, merely absent from the table
    outcome = ps.end_scope(captured=captured, exclude={os.getpid(), 1}, grace_sec=0.0)
    assert signalled == [4242] and outcome["ended"] == [4242] and outcome["settled"] is True
    del answers[4242]  # positive absence
    outcome = ps.end_scope(captured=captured, exclude={os.getpid(), 1}, grace_sec=0.0)
    assert outcome["captured_gone"] == [4242] and outcome["unverified"] == [] and outcome["settled"] is True


def test_a_failed_census_carries_every_unresolved_known_identity_as_unverified(monkeypatch):
    """Astra delta2 1: when the table cannot be read, the recorded identities this pass did
    not resolve are owed, never dropped: they ride the outcome, and so the record."""
    birth = "Sat Oct 10 20:10:23 2026"
    monkeypatch.setattr(ps, "census", lambda markers=(): None)
    outcome = ps.end_scope(marker=ps.mint_marker(ps.TASK_SCOPE_PREFIX),
                           captured=[{"pid": 4242, "birth": birth}, {"pid": 4243, "birth": birth}], exclude={os.getpid(), 1})
    assert outcome["census_failed"] is True and outcome["settled"] is False and outcome["ended"] == []
    assert outcome["unverified"] == [{"pid": 4242, "birth": birth, "reason": ps.UNVERIFIED_CENSUS},
                                     {"pid": 4243, "birth": birth, "reason": ps.UNVERIFIED_CENSUS}]
    assert ps._known_members(outcome) == [{"pid": 4242, "birth": birth}, {"pid": 4243, "birth": birth}]


@pytest.mark.parametrize("later", ["ended", "gone"])
def test_an_unverified_entry_is_dropped_once_a_later_scan_ends_or_confirms_the_pid(monkeypatch, later):
    """Fable delta2 NIT 2: a captured pid unreadable on scan 1 and ended (or confirmed gone)
    on the rescan that another member's kill pass earns owes nothing; a stale unverified
    entry would cost one more sweep."""
    birth, other = "Sat Oct 10 20:10:23 2026", "Mon Jan  1 00:00:00 2001"
    marker = ps.mint_marker(ps.TASK_SCOPE_PREFIX)
    scans = []

    def census(markers=()):  # scan 1: a marker member, no row for the captured pid; later: its row (or a replacement's)
        scans.append(1)
        if len(scans) == 1:
            return {4300: ps._Row(4300, 1, 4300, birth, 1.0, markers={marker})}
        return {4242: ps._Row(4242, 1, 4242, birth if later == "ended" else other, 1.0)}

    monkeypatch.setattr(ps, "census", census)
    answers = iter([ps.IDENTITY_UNREADABLE])  # the first read (the missing captured pid) is unreadable; then alive

    def identities(pids):
        state = next(answers, ps.IDENTITY_ALIVE)
        return {int(pid): (ps._Identity(state) if state == ps.IDENTITY_UNREADABLE else ps._Identity(state, birth, int(pid)))
                for pid in pids}

    monkeypatch.setattr(ps, "read_identities", identities)
    signalled = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: signalled.append(pid) or _pl.SIGNAL_SENT)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    outcome = ps.end_scope(marker=marker, captured=[{"pid": 4242, "birth": birth}], exclude={os.getpid(), 1}, grace_sec=0.0)
    assert outcome["unverified"] == [], "resolved on the rescan: nothing left to disclose"
    assert (outcome["ended"], outcome["captured_gone"]) == (([4242, 4300], []) if later == "ended" else ([4300], [4242]))
    assert outcome["settled"] is True and signalled == ([4300, 4242] if later == "ended" else [4300])


@pytest.mark.parametrize("failure", ["unreadable", "signal_failed"])
def test_resolving_a_stale_birth_never_discharges_another_birth_on_the_same_pid(monkeypatch, failure):
    """Astra delta3 1: a recorded identity (P, A) whose child exited is confirmed gone while a
    NEW scope member reused P with birth B and could not be ended (identity unreadable, or
    its signal failed). Resolution is per identity: A's resolution never erases B's
    unverified evidence, so the pass stays unsettled and B rides into the next one."""
    old, new = "Sun Oct 11 12:00:00 2026", "Mon Oct 12 12:00:00 2026"
    marker = ps.mint_marker(ps.INSTALL_SCOPE_PREFIX)
    monkeypatch.setattr(ps, "census", lambda markers=(): {4242: ps._Row(4242, 1, 4242, new, ps.birth_epoch(new), markers={marker})})
    if failure == "unreadable":
        monkeypatch.setattr(ps, "read_identities", lambda pids: {int(pid): ps._Identity(ps.IDENTITY_UNREADABLE) for pid in pids})
        result, reason = _pl.SIGNAL_SENT, ps.UNVERIFIED_IDENTITY
    else:
        monkeypatch.setattr(ps, "read_identities", lambda pids: {int(pid): ps._Identity(ps.IDENTITY_ALIVE, new, int(pid)) for pid in pids})
        result, reason = _pl.SIGNAL_FAILED, ps.UNVERIFIED_FAILED
    signalled = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: signalled.append(pid) or result)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    monkeypatch.setattr(ps, "_RESCAN_SETTLE_SEC", 0)
    outcome = ps.end_scope(marker=marker, captured=[{"pid": 4242, "birth": old}], exclude={os.getpid(), 1},
                           grace_sec=0.0, term_first=False, rescans=1)
    assert outcome["captured_gone"] == [4242] and outcome["ended"] == [], "A is gone; nothing was ended"
    assert outcome["unverified"] == [{"pid": 4242, "birth": new, "reason": reason}], "B keeps its own evidence"
    assert outcome["settled"] is False and ps._known_members(outcome) == [{"pid": 4242, "birth": new}]
    assert signalled == ([] if failure == "unreadable" else [4242])


def test_each_birth_recorded_for_one_pid_is_judged_on_its_own(monkeypatch):
    """Two custodians may both owe pid P under different births (a pending record's known
    member and a task record's). A table without P asks the reader once and judges each
    birth: unreadable keeps both; alive with B confirms A gone and ends B."""
    old, new = "Sun Oct 11 12:00:00 2026", "Mon Oct 12 12:00:00 2026"
    captured = [{"pid": 4242, "birth": old}, {"pid": 4242, "birth": new}]
    monkeypatch.setattr(ps, "census", lambda markers=(): {})
    signalled = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: signalled.append(pid) or _pl.SIGNAL_SENT)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {int(pid): ps._Identity(ps.IDENTITY_UNREADABLE) for pid in pids})
    outcome = ps.end_scope(captured=captured, exclude={os.getpid(), 1}, grace_sec=0.0)
    assert outcome["unverified"] == [{"pid": 4242, "birth": old, "reason": ps.UNVERIFIED_IDENTITY},
                                     {"pid": 4242, "birth": new, "reason": ps.UNVERIFIED_IDENTITY}]
    assert outcome["settled"] is False and signalled == []
    monkeypatch.setattr(ps, "read_identities", lambda pids: {int(pid): ps._Identity(ps.IDENTITY_ALIVE, new, int(pid)) for pid in pids})
    outcome = ps.end_scope(captured=captured, exclude={os.getpid(), 1}, grace_sec=0.0)
    assert (outcome["captured_gone"], outcome["ended"], outcome["unverified"]) == ([4242], [4242], [])
    assert outcome["settled"] is True and signalled == [4242]


def test_an_entrys_iso_opened_at_converts_to_the_lstart_floor_and_junk_to_no_floor():
    """The engine writes ``opened_at`` as ISO 8601 (JS ``toISOString``: milliseconds, ``Z``);
    the floor is that instant in the lstart spelling the census compares, whole seconds, a
    naive time read as UTC; anything else is no floor."""
    epoch = ps.birth_epoch("Mon Oct 12 12:00:07 2026")
    import datetime as dt

    utc = dt.datetime.fromtimestamp(epoch, dt.timezone.utc)
    assert ps._iso_floor(utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")) == "Mon Oct 12 12:00:07 2026"
    assert ps._iso_floor(utc.isoformat()) == "Mon Oct 12 12:00:07 2026"
    assert ps._iso_floor(utc.replace(tzinfo=None).isoformat()) == "Mon Oct 12 12:00:07 2026"
    assert ps._iso_floor(utc.astimezone(dt.timezone(dt.timedelta(hours=2))).isoformat()) == "Mon Oct 12 12:00:07 2026"
    assert ps._iso_floor("") == ps._iso_floor(None) == ps._iso_floor("not a time") == ps._iso_floor(12) == ""


def test_a_proc_stat_read_failure_other_than_absence_is_unreadable(monkeypatch):
    def controlled(path, *args, **kwargs):
        if str(path) == "/proc/4242/stat":
            raise PermissionError(13, "denied")
        if str(path) == "/proc/4243/stat":
            raise FileNotFoundError(2, "gone")
        raise AssertionError(path)

    monkeypatch.setattr(pc, "open", controlled, raising=False)  # the census module's own reads only
    assert pc._proc_identity(4242).state == pc.IDENTITY_UNREADABLE, "denied is not absent"
    assert pc._proc_identity(4243).state == pc.IDENTITY_ABSENT


def test_rotation_carries_the_previous_markers_so_an_old_setsid_child_is_still_named(tmp_path):
    first = ps.task_scope_marker(tmp_path, "t-rot")
    old = _sleeper({first: "1"})  # a marker-visible child of the first attempt
    try:
        path = ps.task_scope_path(tmp_path, "t-rot")
        record = ps.task_scope_record(tmp_path, "t-rot")
        path.write_text(json.dumps({**record, "cleanup": {"state": "pending", "reason": "census_failed"}}), encoding="utf-8")
        second = ps.task_scope_marker(tmp_path, "t-rot")
        rotated = ps.task_scope_record(tmp_path, "t-rot")
        assert second != first and rotated["previous_markers"] == [first]
        assert ps._record_markers(rotated) == [second, first]
        outcome = ps.end_scope(markers=ps._record_markers(rotated), exclude={os.getpid(), 1}, grace_sec=0.2)
        assert outcome["ended"] == [old.pid] and _gone(old.pid)
    finally:
        _end(old)


def test_the_census_reader_carries_no_scope_marker_and_pins_the_c_locale(monkeypatch):
    install, task = ps.mint_marker(ps.INSTALL_SCOPE_PREFIX), ps.mint_marker(ps.TASK_SCOPE_PREFIX)
    monkeypatch.setenv(install, "1")
    monkeypatch.setenv(task, "1")
    env = pc._ps_env()
    assert not ps.scope_entries(env), "the ps child never lists itself as a member"
    assert env["LC_ALL"] == "C" and env["LANG"] == "C"


def test_proc_stat_rows_need_no_ps_and_spell_births_like_lstart():
    data = (b"1234 (a comm) with) parens) S 1 1234 1234 0 -1 4194560 100 0 0 0 5 3 0 0 20 0 1 0 98765 1000000 50 "
            b"18446744073709551615 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")
    state, ppid, pgid, epoch = pc.parse_proc_stat(data, boot_epoch=1_700_000_000, hz=100)
    assert (state, ppid, pgid, epoch) == ("S", 1, 1234, 1_700_000_987.0), "btime + starttime // CLK_TCK, as procps prints it"
    text = pc.lstart_text(epoch)
    assert pc._LSTART_RE.match(text) and pc.birth_epoch(text) == epoch
    assert pc._state_letter("Z") == pc.IDENTITY_ZOMBIE and pc._state_letter("Ss") == pc.IDENTITY_ALIVE


def test_the_lstart_spelling_is_english_under_a_non_english_lc_time():
    """grok 3: the engine writes the C-locale spelling; a server whose LC_TIME is German
    must still spell (and compare) a ``/proc`` birth the same way."""
    import locale

    saved = locale.setlocale(locale.LC_TIME)
    epoch = 1_700_000_987.0
    try:
        try:
            locale.setlocale(locale.LC_TIME, "de_DE.UTF-8")
        except locale.Error:
            pytest.skip("no non-English LC_TIME locale on this host")
        if time.strftime("%a", time.localtime(epoch)) in pc._WEEKDAYS:
            pytest.skip("this host's de_DE spelling is indistinguishable from English")
        text = pc.lstart_text(epoch)
        weekday, month = text.split()[:2]
        assert weekday in pc._WEEKDAYS and month in pc._MONTH_NAMES, text
        assert pc._LSTART_RE.match(text) and pc.birth_epoch(text) == epoch
    finally:
        locale.setlocale(locale.LC_TIME, saved)
