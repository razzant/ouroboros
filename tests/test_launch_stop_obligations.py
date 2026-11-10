"""What a launch still owes survives every writer that records it, and a launch is a launch without its birth.

Each stop records what it could not confirm (``note_owed``), from what it measured, even when it cannot take the
ledger lock; an in-memory owner and the launch's record step can run side by side. A launch whose leader birth could
not be read at registration keeps its launch marker as its reach (``_launch_stop_shared``: controlled identities, or
the real runner that leaves a marker-carrying server behind).
"""

from __future__ import annotations

import os
import pathlib
import threading
import time

import pytest

from tests._launch_stop_shared import (
    _CHILD,
    _MARK_A,
    _OLD_BIRTH,
    _PATIENCE,
    _PID,
    _controlled,
    _held,
    _launch_row,
    _owed,
    _reader,
)
from tests._cases import case_folders, chosen_rows

pytestmark = [
              pytest.mark.skipif(os.name == "nt", reason="the measured stop is POSIX; Windows keeps its native kill")]


class _Held:
    """The in-memory owner's Popen over the controlled table: exited once its pid is gone."""

    def __init__(self, pid, rows):
        self.pid, self._rows, self.returncode = pid, rows, None

    def poll(self):
        if self.returncode is None and self.pid not in self._rows:
            self.returncode = -9
        return self.returncode

    def wait(self, timeout=None):
        return self.poll()


@pytest.mark.parametrize(("order", "consumer"), [
    ("memory-then-ledger", "next-start retry"),
    ("ledger-then-memory", "next-generation reaper"),
])
def test_two_parallel_exit_stops_of_one_launch_keep_what_either_could_not_end(tmp_path, monkeypatch, order, consumer):
    """Launch A's runner L has a child C in its own group whose binary hides its environment. Two writers of
    ``_stop_records`` stop A at once: the in-memory owner reaches C through L's chain while the daemon roots are
    unreadable, so C stays owed and held, and it ends L; the record step, given the row read before either began,
    meets an unreadable table and owes L. They finish in either order under free locks. The next start then ends C,
    never retiring A while C lives."""
    from ouroboros import owned_shutdown, process_census, process_custody, process_scope
    from ouroboros.tools import services

    rows = {_PID: process_census._Row(_PID, 2, _PID, _OLD_BIRTH, 123.0, "alive", {_MARK_A}),
            _CHILD: process_census._Row(_CHILD, _PID, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}
    world = _controlled(tmp_path, monkeypatch, rows)
    world.roots_unknown, world.table_lost = True, True
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda root: None if world.roots_unknown else set())

    def census(*args, **kwargs):  # the record step's table read fails during the exit; the owner's does not
        world.censuses.append(threading.current_thread().name)
        return None if world.table_lost and threading.current_thread().name == "owned-stop-services" else dict(rows)

    monkeypatch.setattr(process_scope, "census", census)
    monkeypatch.setattr("ouroboros.platform_layer.request_process_tree_kill", lambda proc: world.kills.append(proc.pid))
    row = _launch_row("t-a")
    world.spawned(row)
    proc = _Held(_PID, rows)
    proc._ouroboros_custody = dict(row)
    record = services.ServiceRecord(name="a", service_id="t-a:a", task_id="t-a", cmd=["runner"], cwd=str(tmp_path),
                                    log_path=tmp_path / "a.log", proc=proc, pgid=_PID, leader_birth=_OLD_BIRTH,
                                    scope_marker=_MARK_A, drive_root=str(tmp_path))
    monkeypatch.setattr(services, "_SERVICES", {record.service_id: record})
    real_note, first_done, writers = process_custody.note_owed, threading.Event(), []
    first = "owned-stop-memory" if order == "memory-then-ledger" else "owned-stop-services"

    def in_order(*args, **kwargs):  # both writers run; the gate only decides whose final answer lands first
        mine = threading.current_thread().name == first
        if not mine:
            assert first_done.wait(10), "the other writer never recorded"
        try:
            return real_note(*args, **kwargs)
        finally:
            writers.append((threading.current_thread().name, [item["pid"] for item in args[2]]))
            if mine:
                first_done.set()

    monkeypatch.setattr(process_custody, "note_owed", in_order)
    entries = owned_shutdown.owned_records(tmp_path)  # what the exit stop read before either stop began
    remaining = owned_shutdown._stop_records(tmp_path, entries, time.monotonic() + 10, owners=[
        ("memory", lambda: services.kill_all_services(tmp_path, wait=True, durable=False))])

    assert _PID not in rows and _CHILD in rows and remaining, "L ended; C is still owed"
    answers = {name: pids for name, pids in writers}
    # each owes what it could not end; the record step may already have read C, if the owner's write came first
    assert _PID in answers["owned-stop-services"] and _CHILD in answers["owned-stop-memory"], writers
    monkeypatch.setattr(process_custody, "note_owed", real_note)
    monkeypatch.setattr(services, "_SERVICES", {})  # that server exited; its in-memory owner is gone
    world.roots_unknown, world.table_lost = False, False
    if consumer == "next-start retry":
        owned_shutdown.finish_unconfirmed_stops(tmp_path)
    else:
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())

    assert _CHILD not in rows, "what either stop could not end is ended by the next start, never retired with it"
    if consumer == "next-start retry":  # its end confirmed, nothing it owed lingers
        assert owned_shutdown.owned_records(tmp_path) == []
    else:
        assert process_custody._read_ledger(tmp_path) == []


@pytest.mark.parametrize("current,judged,observed,stale,expected", [
    # a stop discharges what it measured and left out, keeps what another stop added, takes its own answer
    ([{"pid": 7, "birth": "b"}], [{"pid": 7, "birth": "b"}], [], False, []),
    # held status survives its answer
    ([{"pid": 8, "birth": "b", "held": True}], (), [{"pid": 8, "birth": "b"}], False,
     [{"pid": 8, "birth": "b", "held": True}]),
    # another birth on the same pid is another identity; a stale copy adds without authority
    ([], (), [{"pid": 9, "birth": "b"}], True, [{"pid": 9, "birth": "b", "held": True}]),
])
def test_only_a_stop_that_judged_an_identity_discharges_it(current, judged, observed, stale, expected):
    from ouroboros.process_custody import owed_after

    assert owed_after(current, judged, observed, stale=stale) == expected


@pytest.fixture
def scripts(tmp_path):
    from tests._launch_stop_shared import write_scripts

    return write_scripts(tmp_path)


def _births_unreadable(monkeypatch):
    """Every birth read of the registration fails (a transient ``ps`` failure); returns the restore."""
    from ouroboros import process_census, process_scope

    real_birth, real_group = process_census.birth_text, process_scope.record_task_group
    monkeypatch.setattr(process_census, "birth_text", lambda pid: "")
    monkeypatch.setattr(process_scope, "record_task_group",
                        lambda *args, **kwargs: {**(real_group(*args, **kwargs) or {}), "leader_birth": ""})

    def restore():
        monkeypatch.setattr(process_census, "birth_text", real_birth)
        monkeypatch.setattr(process_scope, "record_task_group", real_group)

    return restore


@pytest.mark.parametrize("consumer", ["reaper", "quiesce", "body candidate"])
def test_a_launch_whose_birth_could_not_be_read_is_still_ended_and_held_by_its_marker(
        tmp_path, monkeypatch, scripts, consumer):
    """The registration could not read the leader's birth, so the row keeps its launch marker without one. The
    runner then exits and its server, in another session, still carries the marker: the next generation's sweep
    and the update quiesce end it, and the custody readers hold the launch while it lives."""
    from ouroboros import body_candidate, process_custody
    from ouroboros.presence_continuation import _custody_blockers
    from tests._launch_stop_shared import _PATIENCE, _answers, _ledgered_service, _same_process_alive, _settle

    restore = _births_unreadable(monkeypatch)
    leader, (port, pid, birth) = _ledgered_service(tmp_path, scripts, "kept", "separate-exit", scope="task",
                                                   owner_task="t")
    restore()
    try:
        assert leader.wait(timeout=_PATIENCE) == 0
        [row] = [item for item in process_custody._read_ledger(tmp_path) if item["purpose"] == "service:kept"]
        assert row["scope_marker"] and not row.get("scope_birth")
        if consumer == "reaper":
            monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
            process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())
        elif consumer == "quiesce":
            assert process_custody.quiesce_custodied_services(tmp_path) == (True, [])
        elif consumer == "presence":
            assert any(item["kind"] == "owned_process" for item in _custody_blockers(tmp_path, "t"))
        else:
            assert body_candidate.live_lineage_processes(tmp_path, "t")
        if consumer in ("reaper", "quiesce"):
            assert not _answers(port) and not _same_process_alive(pid, birth)
    finally:
        _settle(leader, identities=[(pid, birth)])


def test_an_executor_service_whose_birth_could_not_be_read_keeps_its_marker_for_the_exit_stop(
        tmp_path, monkeypatch, scripts):
    """The executor's durable record of that launch keeps the marker too, and the exit stop's record step ends
    the server the runner left before it forgets the record."""
    from ouroboros import owned_shutdown, workspace_executor
    from tests._launch_stop_shared import (
        _PATIENCE,
        _answers,
        _end,
        _executor_ctx,
        _executor_start,
        _same_process_alive,
        _server,
    )

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path)
    restore = _births_unreadable(monkeypatch)
    _executor_start(ctx, workspace, scripts, tmp_path / "demo.port", "demo", mode="separate-exit")
    restore()
    port, pid, birth = _server(tmp_path / "demo.port")
    with workspace_executor._STATE_LOCK:
        record = workspace_executor._SERVICES.pop("exec-task:demo")
    try:
        record.local_proc.wait(timeout=_PATIENCE)
        root = owned_shutdown.installation_root(ctx.drive_root)
        entry = next(item for item in owned_shutdown.owned_records(root)
                     if item.get("record_path") == str(record.durable_record_path))

        stopped = owned_shutdown._stop_entry(root, entry, time.monotonic() + _PATIENCE, set())

        assert (stopped, _answers(port), _same_process_alive(pid, birth)) == (True, False, False)
        assert not record.durable_record_path.exists()
    finally:
        _end(pid, birth)


@pytest.mark.parametrize(("reader", "case"), [
    ("presence", "marker carrier"),
    ("cold sleep", "hidden live leader"),
    ("continuation", "command-changed leader"),
    ("body candidate", "birth-refuted pid"),
])
def test_every_reader_holds_an_unmeasured_launch_by_its_marker_and_only_a_birth_refutes_its_leader(
        tmp_path, monkeypatch, reader, case):
    """The four custody readers over a launch row whose leader birth the registration could not read: a process
    that kept its marker holds it; a live process on its leader's pid that shows no marker may be that leader
    hiding its environment, so the launch is unknown, never settled; a command-only change after an ``exec``
    (same pid, same birth) is still that unmeasured leader — unknown, not ended; only a measured BIRTH that
    differs refutes the pid as another process. Controlled identities, no OS process signalled."""
    from ouroboros import process_census, process_custody
    from tests._launch_stop_shared import _CARRIER, _held, _reader

    rows = ({_CARRIER: process_census._Row(_CARRIER, 1, _CARRIER, _OLD_BIRTH, 123.0, "alive", {_MARK_A})}
            if case == "marker carrier" else {_PID: process_census._Row(_PID, 2, _PID, _OLD_BIRTH, 123.0, "alive", set())})
    read, owner = _reader(tmp_path, monkeypatch, reader)
    world = _controlled(tmp_path, monkeypatch, rows)
    mismatch = {"command-changed leader": ["cmd_sha256"], "birth-refuted pid": ["start_time"]}.get(case)
    if mismatch is not None:
        monkeypatch.setattr(process_custody, "measured_identity_mismatch",
                            lambda row: {"mismatches": mismatch} if int(row["pid"]) in rows else {})
    world.spawned({**_launch_row(owner), "scope_birth": ""})

    found = read()

    expected = {"marker carrier": "live", "hidden live leader": "unknown",
                "command-changed leader": "unknown", "birth-refuted pid": "none"}[case]
    assert _held(found, _PID) == expected, found
    assert world.signals == [] and world.kills == []


def test_one_census_with_a_marker_read_answers_within_its_own_bound(monkeypatch):
    """A slow table read leaves the targeted marker read (macOS ``ps -E -p``) only what is left of the census's
    one bound: a stalled second read cannot double the time a shutdown's grace share allowed it."""
    import subprocess
    import sys

    from ouroboros import process_census

    if os.path.isdir("/proc"):
        pytest.skip("Linux reads /proc: no ps subprocess")
    real_run, calls = process_census.subprocess.run, []

    def slow(argv, *args, timeout=None, **kwargs):
        calls.append((list(argv), timeout))
        if "-E" in argv:  # the marker read stalls until its bound
            time.sleep(timeout or 0)
            raise subprocess.TimeoutExpired(argv, timeout)
        time.sleep(0.6 * (timeout or 0))  # the table answers, slowly
        return real_run(argv, *args, timeout=timeout, **kwargs)

    monkeypatch.setattr(process_census.subprocess, "run", slow)
    began = time.monotonic()
    rows = process_census.census(["OURO_LAUNCH_SCOPE_" + "d" * 32], env_rows=lambda table: [os.getpid()], timeout=1.0)
    elapsed = time.monotonic() - began

    assert rows is None, "the marker read failed: the table is unknown, never clean"
    assert any("-E" in argv for argv, _bound in calls), calls
    assert elapsed < 1.25, (elapsed, calls, sys.platform)


@pytest.mark.parametrize("leader", ["birth recorded", "birth unrecorded"])
def test_identity_reads_that_never_answer_at_shutdown_keep_the_one_grace(tmp_path, monkeypatch, scripts, leader):
    """The table answers but every identity read (``ps -p``) blocks until its bound. A held service whose leader
    birth was recorded then has no identity to signal by: nothing is signalled and it keeps its custody for the
    next start. One whose birth was never recorded is unmeasured, so it gets its native group kill, inside the
    one grace: its birth read takes only the grace's share. Either way the exit leaves inside that grace. This is
    the ``ps`` reader's contract, selected on every POSIX host (``_read_through_ps``); a stall in Linux's active
    ``/proc`` reader has no such bound and is NOT covered here."""
    import subprocess

    from ouroboros import owned_shutdown, process_census
    from ouroboros.tools import services
    from tests._launch_stop_shared import _end, _read_through_ps, _registry, _same_process_alive, _server, _start

    _read_through_ps(monkeypatch)
    registry = _registry(tmp_path, monkeypatch)
    leader_pid = _start(registry, scripts, tmp_path / "held.port", "held", mode="same")["pid"]
    _port, pid, birth = _server(tmp_path / "held.port")
    [record] = services._SERVICES.values()
    if leader == "birth unrecorded":
        record.leader_birth = ""
    elif not record.leader_birth:
        record.leader_birth = process_census.measure_birth(leader_pid)
    budget = 3.0
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: budget)
    real_run, real_killpg, kills, reads = process_census.subprocess.run, os.killpg, [], []
    began = time.monotonic()

    def blocked(argv, *args, timeout=None, **kwargs):
        if list(argv)[:1] == ["ps"] and "-p" in argv and "-E" not in argv:
            reads.append(timeout)
            time.sleep(min(timeout or 6.0, 6.0))
            raise subprocess.TimeoutExpired(argv, timeout)
        return real_run(argv, *args, timeout=timeout, **kwargs)

    monkeypatch.setattr(process_census.subprocess, "run", blocked)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: kills.append((time.monotonic() - began, pgid, sig))
                        or real_killpg(pgid, sig))
    before = set(threading.enumerate())
    try:
        outcome = owned_shutdown.stop_owned_work(tmp_path / "data")
        returned = time.monotonic() - began
        for thread in set(threading.enumerate()) - before:  # what this stop started, finished before judging it
            if thread.name.startswith("owned-stop-"):
                thread.join(timeout=30)
        monkeypatch.setattr(process_census.subprocess, "run", real_run)

        assert returned < budget + 1.0 and reads and outcome["state"] == "unconfirmed", (returned, reads, outcome)
        native = [at for at, pgid, sig in kills if pgid == leader_pid and sig]
        if leader == "birth recorded":
            assert not native and _same_process_alive(pid, birth), "nothing measured it, so nothing signalled it"
        else:
            assert native and native[0] < budget, (kills, reads)
    finally:
        monkeypatch.undo()
        _end(pid, birth)
        _end(leader_pid, process_census.measure_birth(leader_pid))


def test_a_contended_note_owed_keeps_a_child_another_stop_published_in_the_ledger(tmp_path, monkeypatch):
    """Two stops of one launch publish what each still owes. Writer A, holding the ledger append lock, publishes child
    C; writer B's answer names only the leader. B must not drop C from the canonical ledger, what cold sleep, Continue
    and Presence read: its read and append share one lock hold, so when B cannot take the lock it appends nothing."""
    from ouroboros import owned_shutdown, process_census, process_custody
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.utils import append_jsonl, jsonl_append_lock_path

    rows = {_PID: process_census._Row(_PID, 2, _PID, _OLD_BIRTH, 123.0, "alive", {_MARK_A})}
    world = _controlled(tmp_path, monkeypatch, rows)
    launch = _launch_row("t-a")  # the ledger row, owing nothing yet
    world.spawned(launch)
    child = [{"pid": _CHILD, "birth": _OLD_BIRTH}]
    lock_path = jsonl_append_lock_path(process_custody.ledger_path(tmp_path))
    held = acquire_exclusive_file_lock(lock_path, timeout_sec=5.0)  # writer A holds it across B's attempt
    assert held is not None
    try:
        a_row = {"ts": "t", **{k: v for k, v in launch.items() if k != "ts"}, "known_members": child}
        assert append_jsonl(process_custody.ledger_path(tmp_path), a_row, lock_held=True, lock_timeout_sec=0.0)
        owned_shutdown.record_ledgered_process(tmp_path, a_row, judged=())  # A, under the lock, published C
        result = {}

        def writer_b():  # B's answer names only the leader
            result["row"] = process_custody.note_owed(
                tmp_path, launch, [{"pid": _PID, "birth": _OLD_BIRTH}], lock_timeout_sec=0.3)

        b = threading.Thread(target=writer_b, name="writer-B")
        b.start()
        b.join(10)
    finally:
        release_exclusive_file_lock(lock_path, held)

    assert result.get("row") is None, "B could not take the lock, so it overwrote nothing"
    [latest] = [r for r in process_custody._read_ledger_records(tmp_path, strict=True)[1]
                if r.get("scope_marker") == _MARK_A]
    assert [m["pid"] for m in latest.get("known_members") or []] == [_CHILD], "the ledger still owes C"


def test_a_host_service_is_not_retired_while_a_parallel_stop_owes_a_live_child(tmp_path, monkeypatch):
    """The in-memory service owner cannot trace child C once C left the leader's group and hid its marker, so its
    own pass reports the leader exited. The launch's ledgered record step observed C and persisted it as owed.
    The owner reconciles against that persisted debt: it keeps the service record and its live log, finalizing
    neither while C is alive."""
    from ouroboros import process_census, process_custody
    from ouroboros.tools import services

    rows = {_CHILD: process_census._Row(_CHILD, 1, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}  # orphan, no marker
    world = _controlled(tmp_path, monkeypatch, rows)
    spawn = _launch_row("t-a")  # the memory owner's view of the launch: no child yet
    world.spawned(spawn)
    owed = process_custody.note_owed(tmp_path, spawn, [{"pid": _CHILD, "birth": _OLD_BIRTH}])  # the ledger pass persists C
    assert owed and [m["pid"] for m in owed["known_members"]] == [_CHILD]

    log = tmp_path / "svc.log"
    log.write_text("live log\n", encoding="utf-8")
    proc = _Held(_PID, rows)  # the leader is gone (not in the table)
    proc._ouroboros_custody = dict(spawn)  # the owner still holds the pre-child row
    record = services.ServiceRecord(name="a", service_id="t-a:a", task_id="t-a", cmd=["runner"], cwd=str(tmp_path),
                                    log_path=log, proc=proc, pgid=_PID, leader_birth=_OLD_BIRTH,
                                    scope_marker=_MARK_A, drive_root=str(tmp_path))
    monkeypatch.setattr(services, "_SERVICES", {record.service_id: record})

    stopped = services.kill_all_services(tmp_path, wait=True, durable=False)

    assert _CHILD in rows, "C is still alive"
    assert record.service_id in services._SERVICES, "the service record is kept, not retired"
    assert log.exists() and log.read_text(encoding="utf-8") == "live log\n", "its live log is not finalized"
    assert any(item.get("stop_failed") for item in stopped), stopped
    [latest] = [r for r in process_custody._read_ledger(tmp_path) if r.get("scope_marker") == _MARK_A]
    assert [m["pid"] for m in latest.get("known_members") or []] == [_CHILD], "the launch still owes C"


def _runner_and_hidden_child():
    """Runner L of launch A, and its child C in a group of its own whose binary hides its environment."""
    from ouroboros import process_census

    return {_PID: process_census._Row(_PID, 2, _PID, _OLD_BIRTH, 123.0, "alive", {_MARK_A}),
            _CHILD: process_census._Row(_CHILD, _PID, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}


def _in_memory_service(tmp_path, monkeypatch, owner, rows, row):
    """The exit process's own in-memory record of launch A (``owner``: host service or local executor service),
    its Popen holding L and its live log; returns (record, its registry, the stop function its owner calls)."""
    from ouroboros import workspace_executor
    from ouroboros.tools import services

    log = tmp_path / "a.log"
    log.write_text("live log\n", encoding="utf-8")
    proc = _Held(_PID, rows)
    proc._ouroboros_custody = dict(row)
    if owner == "host service":
        record = services.ServiceRecord(name="a", service_id="t-a:a", task_id="t-a", cmd=["runner"], cwd=str(tmp_path),
                                        log_path=log, proc=proc, pgid=_PID, leader_birth=_OLD_BIRTH,
                                        scope_marker=_MARK_A, drive_root=str(tmp_path))
        monkeypatch.setattr(services, "_SERVICES", {record.service_id: record})
        monkeypatch.setattr(workspace_executor, "_SERVICES", {})
        return record, services._SERVICES, (services, "_stop_host_record")
    from ouroboros.workspace_executor_paths import ExecutorRef

    record = workspace_executor._ExecutorService(
        service_id="t-a:a", task_id="t-a", name="a", executor=ExecutorRef("local", "local", "host", ()),
        cmd=["runner"], host_cwd=tmp_path, backend_cwd=str(tmp_path), cwd_root="active_workspace", outputs=[],
        before_outputs={}, local_proc=proc, backend_pid=str(_PID), backend_log_path=str(log), drive_root=tmp_path,
        leader_birth=_OLD_BIRTH, scope_marker=_MARK_A)
    monkeypatch.setattr(workspace_executor, "_SERVICES", {record.service_id: record})
    monkeypatch.setattr(services, "_SERVICES", {})
    return record, workspace_executor._SERVICES, (workspace_executor, "_stop_service_record")


@pytest.mark.parametrize("owner", ["host service", "local executor service"])
def test_a_launch_held_in_memory_is_not_retired_around_what_its_record_step_observed(tmp_path, monkeypatch, owner):
    """The generation's exit (``stop_owned_work``) over a service launch this process holds in memory: its in-memory
    owner and the launch's ledgered record step run in parallel, each with a census of its own. Schedule: the record
    step's census runs first and reaches runner L's child C through L's chain while the daemon roots are unreadable,
    so C is held, and it ends L; the owner's census runs only after that, when nothing traces C any more (another
    group, a hidden environment, L gone); the record step's final answer lands only after the owner decided. C was
    already durable when L died: the record, its live log and its name stay while C lives, the exit is not reported
    clean, and the next start ends C."""
    import json as _json

    from ouroboros import owned_shutdown, process_custody, process_scope
    from ouroboros.tools import services
    from tests._launch_stop_shared import _PATIENCE

    rows = _runner_and_hidden_child()
    world = _controlled(tmp_path, monkeypatch, rows)
    roots = {"unknown": True}
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda root: None if roots["unknown"] else set())
    row = _launch_row("t-a", purpose="service:a" if owner == "host service" else "workspace_service:a")
    world.spawned(row)
    record, registry, (module, stop_name) = _in_memory_service(tmp_path, monkeypatch, owner, rows, row)
    log = tmp_path / "a.log"

    owner_threads, decided = set(), threading.Event()
    real_kill_all = services.kill_all_services
    real_census, real_note, real_stop = process_scope.census, process_custody.note_owed, getattr(module, stop_name)

    def memory_owner(*args, **kwargs):
        owner_threads.add(threading.get_ident())
        return real_kill_all(*args, **kwargs)

    def census(*args, **kwargs):  # the owner's census runs only once the record step ended L
        if threading.get_ident() in owner_threads:
            deadline = time.monotonic() + _PATIENCE
            while _PID in rows:
                assert time.monotonic() < deadline, "the record step never ended L"
                time.sleep(0.01)
        return real_census(*args, **kwargs)

    def owner_stop(*args, **kwargs):
        try:
            return real_stop(*args, **kwargs)
        finally:
            decided.set()

    def note(*args, **kwargs):  # the record step's final answer lands only after the owner decided
        if threading.get_ident() not in owner_threads:
            assert decided.wait(_PATIENCE), "the in-memory owner never decided"
        return real_note(*args, **kwargs)

    monkeypatch.setattr(services, "kill_all_services", memory_owner)
    monkeypatch.setattr(process_scope, "census", census)
    monkeypatch.setattr(module, stop_name, owner_stop)
    monkeypatch.setattr(process_custody, "note_owed", note)

    outcome = owned_shutdown.stop_owned_work(tmp_path)

    assert _PID not in rows and _CHILD in rows, "L ended; C is still alive"
    assert record.service_id in registry, "the service record and its name are kept while C lives"
    assert log.exists() and log.read_text(encoding="utf-8") == "live log\n", "its live log is not finalized"
    assert outcome["state"] == "unconfirmed" and outcome["memory_unconfirmed"].get("services"), outcome
    [latest] = [r for r in process_custody._read_ledger(tmp_path) if r.get("scope_marker") == _MARK_A]
    assert [m["pid"] for m in latest.get("known_members") or []] == [_CHILD], "the launch still owes C"
    assert _CHILD not in [pid for pid, _birth in world.signals], "held: nothing told C from a daemon root"
    events = [_json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()] \
        if (tmp_path / "logs" / "events.jsonl").exists() else []
    assert not [item for event in events if event.get("type") == "services_shutdown_cleanup"
                for item in event["services"] if (item.get("log_finalization") or {}).get("deleted_live_log")]

    monkeypatch.setattr(services, "kill_all_services", real_kill_all)
    monkeypatch.setattr(process_scope, "census", real_census)
    monkeypatch.setattr(process_custody, "note_owed", real_note)
    monkeypatch.setattr(module, stop_name, real_stop)
    registry.clear()  # that server exited; its in-memory owner is gone
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    roots["unknown"] = False
    owned_shutdown.finish_unconfirmed_stops(tmp_path)

    assert _CHILD not in rows, "the next start ends what the exit could not"
    assert owned_shutdown.owned_records(tmp_path) == []


# The body candidate counts any record the exit stamped ``unconfirmed_since`` as live, so it reads a stop's
# observation only after a stop that stamps nothing (the reaper's).
_CONTENDED_CASES = [*((stop, "presence") for stop in ("exit record step", "reaper", "quiesce", "service owner",
                                                      "service owner after the grace")),
                    *(("exit record step", reader) for reader in ("cold sleep", "continuation", "next start")),
                    ("reaper", "body candidate"), ("service owner", "next start")]


@pytest.mark.parametrize("stop,reader", chosen_rows(_CONTENDED_CASES, 0, 8, 9))
def test_a_child_a_stop_observed_while_the_ledger_lock_was_held_stays_owed_for_every_later_reader(
        tmp_path, monkeypatch, stop, reader):
    """A stop of launch A reaches runner L's child C through L's chain while the daemon roots are unreadable, so
    C is held and owed, and it ends L: from then on only that stop's observation names C (another group, a hidden
    environment, no parent chain). Another holder keeps the ledger's append lock across each of that stop's
    publications (it waits its bound and cannot reconcile). C stays owed anyway: the custody readers
    (cold sleep, Continue, Presence, the body candidate) hold the launch while C lives, and the next start ends
    C. Nothing signals C while nothing tells it from a daemon root. Past a shutdown's one grace the exit's
    in-memory owner takes one lock attempt, no wait, and still records C."""
    from ouroboros import owned_shutdown, process_custody
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.tools import services
    from ouroboros.utils import jsonl_append_lock_path
    from tests._launch_stop_shared import _held, _reader

    read, task = _reader(tmp_path, monkeypatch, "presence" if reader == "next start" else reader)
    rows = _runner_and_hidden_child()
    world = _controlled(tmp_path, monkeypatch, rows)
    roots = {"unknown": True}
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda root: None if roots["unknown"] else set())
    row = _launch_row(task, purpose="service:a")
    world.spawned(row)
    lock_path = jsonl_append_lock_path(process_custody.ledger_path(tmp_path))
    real_publish, published = process_custody._publish_owed, []

    def contended(*args, **kwargs):  # another holder keeps the ledger's append lock across this publication
        holder = acquire_exclusive_file_lock(lock_path, timeout_sec=5.0)
        assert holder is not None
        try:
            published.append(real_publish(*args, **kwargs))
        finally:
            release_exclusive_file_lock(lock_path, holder)
        return published[-1]

    monkeypatch.setattr(process_custody, "_publish_owed", contended)
    if stop == "exit record step":
        owned_shutdown._stop_records(tmp_path, owned_shutdown.owned_records(tmp_path), time.monotonic() + 10)
    elif stop == "reaper":
        process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())
    elif stop == "quiesce":
        assert process_custody.quiesce_custodied_services(tmp_path, timeout_sec=0)[0] is False
    else:
        record, _registry, _stop = _in_memory_service(tmp_path, monkeypatch, "host service", rows, row)
        if stop == "service owner after the grace":  # the exit's one grace already ran out
            owned_shutdown._GENERATION_STOP.deadline = time.monotonic() - 1.0
        began = time.monotonic()
        [payload] = services.kill_all_services(tmp_path, wait=True, durable=False)
        assert payload.get("stop_failed") and record.service_id in services._SERVICES, payload
        if stop == "service owner after the grace":
            assert time.monotonic() - began < 1.0, "no lock wait beyond the grace"
    monkeypatch.setattr(process_custody, "_publish_owed", real_publish)

    assert _PID not in rows and _CHILD in rows, "L ended; C is still alive"
    assert published and all(answer == (None, True) for answer in published), "durable, never reconciled under the lock"
    assert _CHILD not in [pid for pid, _birth in world.signals], "held: nothing told C from a daemon root"
    roots["unknown"] = False
    if reader != "next start":
        found = read()
        assert _held(found, _PID) == "live", found
        assert _CHILD in rows
        return
    services._SERVICES.clear()  # that server exited; its in-memory owner is gone
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    if stop == "exit record step":
        owned_shutdown.finish_unconfirmed_stops(tmp_path)
    else:
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())
    assert _CHILD not in rows, "the next start ends what the contended stop observed"


# --------------------------------------------------------------------------- damaged custody, controlled identities


_DAMAGES = ["pending file", "pending directory", "document after folding"]


def _damage(root, damage: str, undo: list) -> None:
    from ouroboros import owned_shutdown

    pending = root / "state" / owned_shutdown.PENDING_DIRNAME
    if damage == "document after folding":
        assert owned_shutdown._update(root, lambda document: False), "the fold needs the lock"
        assert not list(pending.glob("*.json"))
        path = owned_shutdown.owned_processes_path(root)
        intact = path.read_bytes()
        path.write_bytes(b"{torn")
        undo.append(lambda: path.write_bytes(intact))
    elif damage == "pending directory":  # a path that cannot be listed, whoever runs this (root reads through modes)
        moved = pending.with_name(pending.name + ".moved")
        pending.rename(moved)
        pending.write_bytes(b"")

        def restore() -> None:
            pending.unlink()
            moved.rename(pending)

        undo.append(restore)
    else:
        for target in list(pending.glob("*.json")):
            intact = target.read_bytes()
            target.write_bytes(b"{torn")
            undo.append(lambda target=target, intact=intact: target.write_bytes(intact))


def _contended_stop(tmp_path, monkeypatch, task):
    """Launch A with C held, its stop's publications meeting a held ledger lock: C lands only as a pending copy."""
    from ouroboros import owned_shutdown, process_census, process_custody
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.utils import jsonl_append_lock_path
    from tests._launch_stop_shared import _CHILD, _MARK_A, _OLD_BIRTH, _PID, _controlled, _launch_row

    rows = {_PID: process_census._Row(_PID, 2, _PID, _OLD_BIRTH, 123.0, "alive", {_MARK_A}),
            _CHILD: process_census._Row(_CHILD, _PID, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}
    world = _controlled(tmp_path, monkeypatch, rows)
    roots = {"unknown": True}
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda root: None if roots["unknown"] else set())
    world.spawned(_launch_row(task, purpose="service:a"))
    lock_path = jsonl_append_lock_path(process_custody.ledger_path(tmp_path))
    target, real = owned_shutdown.owned_records(tmp_path), process_custody.note_owed
    holder = acquire_exclusive_file_lock(lock_path, timeout_sec=5.0)
    assert holder is not None
    try:
        with monkeypatch.context() as held:  # no lock wait in this stop; its exit stamp would only wait out the deadline
            held.setattr(process_custody, "_lock_wait", lambda: 0.0)
            held.setattr(process_custody, "note_owed", lambda *a, **k: real(*a, **{**k, "lock_timeout_sec": 0.0}))
            held.setattr(owned_shutdown, "_stamp", lambda *a, **k: [])
            owned_shutdown._stop_records(tmp_path, target, time.monotonic() + _PATIENCE)
    finally:
        release_exclusive_file_lock(lock_path, holder)
    assert _PID not in rows and _CHILD in rows, "L ended; C is still alive"
    assert not [row for row in process_custody._read_ledger(tmp_path) if row.get("known_members")], (
        "the lock was held: only the pending copy names C")
    assert list((tmp_path / "state" / owned_shutdown.PENDING_DIRNAME).glob("*.json"))
    return rows, world, roots


@pytest.mark.parametrize("damage", chosen_rows(_DAMAGES, 0, 2))
@pytest.mark.parametrize("reader", ["presence", "continuation", "cold sleep", "body candidate"])
def test_custody_a_contended_stop_wrote_reads_as_unknown_while_it_cannot_be_read(tmp_path, monkeypatch, reader, damage):
    """The stop succeeded in writing C to the existing pending channel because the ledger lock was held. The pending
    file (or its directory, or the document it was folded into) then cannot be read, and the ledger row names only L,
    which is gone. No reader may release the launch: unavailable is not empty."""
    from tests._launch_stop_shared import _CHILD, _PID

    read, task = _reader(tmp_path, monkeypatch, reader)
    rows, _world, roots = _contended_stop(tmp_path, monkeypatch, task)
    roots["unknown"] = False
    undo: list = []
    _damage(tmp_path, damage, undo)
    try:
        found = read()
        assert _held(found, _PID) in ("unknown", "live") and found, f"{reader} released a launch it cannot read: {found}"
        assert _held(found, _PID) == "unknown", found
    finally:
        for restore in undo:
            restore()
    assert _held(read(), _PID) == "live", "readable again, what it owes holds it"
    assert _CHILD in rows, "reading signals nothing"


def test_no_stop_confirms_a_launch_whose_custody_it_cannot_read(subtests, tmp_path, monkeypatch):
    """The same damage under the stops. L is gone and nothing traces C, so each stop's own answer is clean; it still
    keeps the launch, and after the damage clears the next start ends C and retires it."""
    case_paths = case_folders(tmp_path)
    for damage in _DAMAGES:
        for consumer in ["exit record step", "reaper", "quiesce", "service owner"]:
            tmp_path = next(case_paths)
            with subtests.test(damage=damage, consumer=consumer), monkeypatch.context() as monkeypatch:
                from ouroboros import owned_shutdown, process_custody
                from ouroboros.tools import services
                from tests._launch_stop_shared import _CHILD, _MARK_A, _PID, _launch_row

                rows, world, roots = _contended_stop(tmp_path, monkeypatch, "t-a")
                roots["unknown"] = False
                launch = f"launch-{_MARK_A.rsplit('_', 1)[-1]}"
                undo: list = []
                _damage(tmp_path, damage, undo)
                try:
                    if consumer == "exit record step":
                        monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
                        outcome = owned_shutdown.stop_owned_work(tmp_path)
                        assert outcome["state"] == "unconfirmed", outcome
                        if damage != "document after folding":  # with no readable document the exit reads no target at all
                            assert launch in outcome["unconfirmed"], f"the record stays while C cannot be read: {outcome}"
                    elif consumer == "service owner":
                        held, memory, _stop = _in_memory_service(tmp_path, monkeypatch, "host service", rows, _launch_row(
                            "t-a", purpose="service:a"))
                        [payload] = services.kill_all_services(tmp_path, wait=True, durable=False)
                        assert payload["stop_failed"] is True and payload["process_cleanup"]["custody"] == "unavailable", payload
                        assert held.service_id in memory and (tmp_path / "a.log").exists(), "its record, name and live log stay"
                        memory.clear()  # that server exited; its in-memory owner is gone
                    elif consumer == "reaper":
                        process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())
                        assert [row for row in process_custody._read_ledger(tmp_path) if row.get("scope_marker") == _MARK_A]
                    else:
                        ok, blockers = process_custody.quiesce_custodied_services(tmp_path, timeout_sec=0)
                        assert ok is False and blockers, "an update cannot begin over custody it cannot read"
                        assert [row for row in process_custody._read_ledger(tmp_path) if row.get("scope_marker") == _MARK_A]
                finally:
                    for restore in undo:
                        restore()
                assert _CHILD in rows and _PID not in rows
                monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
                entries = [entry for entry in owned_shutdown.owned_records(tmp_path) if owned_shutdown._is_stop_target(entry)]
                assert entries, "the record stayed for the next start"
                assert owned_shutdown._stop_records(tmp_path, entries, time.monotonic() + _PATIENCE) == []
                assert _CHILD not in rows, "the next start ends what the contended stop observed"
                assert world.signals[-1][0] == _CHILD


def test_a_pending_copy_a_fold_deleted_under_a_strict_read_is_in_the_document(tmp_path, monkeypatch):
    """A healthy fold between a strict reader's listing and its read of a pending file is not damage: the document it
    reads next holds that copy, so the reader neither fails nor loses it."""
    from ouroboros import owned_shutdown
    from tests._launch_stop_shared import _CHILD, _OLD_BIRTH, _PID, _launch_row

    (tmp_path / "state").mkdir()
    row = {**_launch_row("t-a"), "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH}]}
    assert owned_shutdown.record_ledgered_process(tmp_path, row, contended=True)
    folded, real_read = [], pathlib.Path.read_text

    def read_text(self, *args, **kwargs):
        if self.parent.name == owned_shutdown.PENDING_DIRNAME and not folded:
            folded.append(self.name)
            assert owned_shutdown._update(tmp_path, lambda document: False)  # the fold, right after the listing
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text)
    [record] = owned_shutdown.owned_records(tmp_path, strict=True)

    assert folded and [item["pid"] for item in record["ledger_entry"]["known_members"]] == [_CHILD]
    assert record["host_pid"] == _PID


@pytest.mark.parametrize("reader", ["presence", "continuation"])
def test_a_later_write_cannot_replace_an_unreadable_set_and_forget_what_only_it_names(tmp_path, monkeypatch, reader):
    """C reached the document only through a contended stop's pending copy, so nothing else names it. The document then
    cannot be read. A later registration, a stop stamp and an exit must not rebuild it from nothing: its bytes stay,
    the launch stays unknown to every reader, new entries wait in the pending channel, and once the document reads
    again C is still owed, the waiting entries fold in, and the next start ends C."""
    from ouroboros import owned_shutdown, process_custody
    from tests._launch_stop_shared import _CHILD, _MARK_B, _OLD_BIRTH, _OTHER, _PID, _launch_row

    read, task = _reader(tmp_path, monkeypatch, reader)
    rows, world, roots = _contended_stop(tmp_path, monkeypatch, task)
    roots["unknown"] = False
    undo: list = []
    _damage(tmp_path, "document after folding", undo)
    document = owned_shutdown.owned_processes_path(tmp_path)
    assert document.read_bytes() == b"{torn"

    world.spawned(_launch_row("t-b", marker=_MARK_B, birth=_OLD_BIRTH, purpose="service:b", pid=_OTHER))
    owned_shutdown._stamp(tmp_path, "stop_requested_at", owned_shutdown._is_stop_target, timeout_sec=0.0)
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    outcome = owned_shutdown.stop_owned_work(tmp_path)

    assert document.read_bytes() == b"{torn", "no write replaced what it could not read"
    assert outcome["state"] == "unconfirmed", outcome
    assert _held(read(), _PID) == "unknown", "unavailable custody is not empty custody"
    assert process_custody._read_ledger(tmp_path) and _CHILD in rows, "reading and refusing signal nothing"

    for restore in undo:
        restore()
    assert _held(read(), _PID) == "live", "readable again, it still owes C"
    assert owned_shutdown._update(tmp_path, lambda current: False), "the waiting entries fold in now"
    assert {f"launch-{_MARK_B.rsplit('_', 1)[-1]}"} <= {entry["record_id"] for entry in owned_shutdown.owned_records(
        tmp_path, strict=True)}
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    targets = [entry for entry in owned_shutdown.owned_records(tmp_path) if entry["ledger_entry"]["pid"] == _PID]
    assert owned_shutdown._stop_records(tmp_path, targets, time.monotonic() + _PATIENCE) == []
    assert _CHILD not in rows, "the next stop ends what the document alone named"


def test_a_row_another_stop_appended_keeps_a_launch_a_sweep_just_confirmed(tmp_path, monkeypatch):
    """A sweep's pass confirmed launch A and judged the D it owed, so what that pass appended goes with A, while A's
    last row is still the one it left. Another stop's row, landing after that pass and before the compaction, names C:
    A stays in both stores, the readers hold it, and the next sweep ends C."""
    from ouroboros import owned_shutdown, process_census, process_custody
    from ouroboros.presence_continuation import _custody_blockers
    from tests._launch_stop_shared import _CARRIER, _CHILD, _OLD_BIRTH, _PID, _controlled, _launch_row

    rows = {_CHILD: process_census._Row(_CHILD, 1, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}  # orphan, nothing traces it
    world = _controlled(tmp_path, monkeypatch, rows)
    row = {**_launch_row("t-a", purpose="service:a"), "known_members": [{"pid": _CARRIER, "birth": _OLD_BIRTH}]}
    world.spawned(row)
    real_rewrite, late = process_custody._rewrite_ledger, []

    def rewrite(*args, **kwargs):
        if kwargs.get("settled") and not late:  # the sweep's pass finished; its compaction has not begun
            late.append(process_custody.note_owed(tmp_path, row, [{"pid": _CHILD, "birth": _OLD_BIRTH}]))
        return real_rewrite(*args, **kwargs)

    monkeypatch.setattr(process_custody, "_rewrite_ledger", rewrite)
    process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())

    assert late and late[0] is not None, "the pass confirmed, then the other stop published"
    assert _CHILD in rows and [(item["pid"]) for item in late[0]["known_members"]] == [_CHILD]
    assert all(names == [_CHILD] for names in _owed(tmp_path, _PID)), "both stores still name it"
    assert _held(_custody_blockers(tmp_path, "t-a"), _PID) == "live"
    monkeypatch.setattr(process_custody, "_rewrite_ledger", real_rewrite)

    process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())

    assert _CHILD not in rows and _owed(tmp_path, _PID) == ([], []), "the next sweep ends it, and the launch leaves"
    assert owned_shutdown.owned_records(tmp_path) == []
