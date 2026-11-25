"""One selection of a launch's custody for every consumer.

A launch's debt can live in the canonical ledger, the ownership document, or a pending copy, and those stores lag each
other by design. Every consumer therefore combines them by LAUNCH before it filters by owner: a stale copy cannot hide
what the ledger names, a compacted ledger cannot hide what only a pending copy names, and a store that cannot be read
is unknown, never absent. Real consumers (cold sleep, Continue, Presence, the body candidate, log prune, the reaper, the
update quiesce, the exit), over a real runner L and server C where the schedule can be staged and controlled identities
where only what is read matters (``_launch_stop_shared``).
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time

import pytest

from tests._launch_stop_shared import (
    _CHILD,
    _HEX_A,
    _HEX_B,
    _MARK_A,
    _MARK_B,
    _OLD_BIRTH,
    _OTHER,
    _PATIENCE,
    _PID,
    _answers,
    _controlled,
    _deny_signals_to,
    _held,
    _launch_row,
    _owed,
    _reader,
    _registry,
    _runner_cmd,
    _same_process_alive,
    _server,
    _settle,
    write_scripts,
)
from tests.test_launch_stop_contract import HIDDEN, _host, _leader_gone, _roots_unknown
from tests._cases import case_folders, chosen_rows

pytestmark = [
              pytest.mark.skipif(os.name == "nt", reason="the measured stop is POSIX; Windows keeps its native kill")]

READERS = ["presence", "continuation", "cold sleep", "body candidate"]


@pytest.fixture
def scripts(tmp_path):
    return write_scripts(tmp_path)


def _ownership_writes_fail(monkeypatch) -> dict:
    """The ownership document and pending channel refuse every write; the ledger still takes its append."""
    from ouroboros import owned_shutdown

    state, real = {"failing": True}, owned_shutdown.atomic_write_json

    def write(path, *args, **kwargs):
        if state["failing"]:
            raise OSError("the state directory refuses this write")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(owned_shutdown, "atomic_write_json", write)
    return state


# --------------------------------------------------------------------------- ledger names more than the copy


def _ledger_only_debt(tmp_path, monkeypatch, scripts, task, *, survivor="denied"):
    """A real stop's ledger append names C while both ownership writes fail: L exits, the set's copy still names only L,
    with no stop stamp on it, and C lives on (its signal denied, or held while the daemon roots are unknown). Returns
    what the test needs and the undo of the denial."""
    from ouroboros import owned_shutdown
    from ouroboros.tools import services

    data = tmp_path / "data"
    ctx, record, registry, log, leader, server = _host(tmp_path, monkeypatch, scripts, task=task)
    roots = _roots_unknown(monkeypatch) if survivor == "held" else None
    undeny = _deny_signals_to(monkeypatch, {server[1]}) if survivor == "denied" else (lambda: None)
    writes = _ownership_writes_fail(monkeypatch)
    stopped = services._stop_service(ctx, name="a")
    writes["failing"] = False
    assert stopped.startswith("⚠️ SERVICE_STOP_ERROR"), stopped
    _leader_gone(leader)
    record.proc.wait(timeout=_PATIENCE)
    ledger, copy = _owed(data, leader)
    assert server[1] in ledger and server[1] not in copy, "the ledger alone names C"
    [entry] = [item for item in owned_shutdown.owned_records(data) if (item.get("ledger_entry") or {}).get("pid") == leader]
    assert not entry.get("unconfirmed_since") and not entry.get("stop_requested_at")
    if roots is not None:
        roots["unknown"] = False
    return data, ctx, record, registry, log, leader, server, undeny


@pytest.mark.parametrize("survivor", ["denied", "held"])
@pytest.mark.parametrize("reader", chosen_rows(READERS, 0, 3))
def test_a_stale_copy_cannot_hide_what_the_ledger_names_from_any_reader(tmp_path, monkeypatch, scripts, reader, survivor):
    """Whichever store a reader used to start from, the launch holds while C lives; reading signals nothing."""
    (tmp_path / "data").mkdir()
    read, task = _reader(tmp_path / "data", monkeypatch, reader)
    _data, _ctx, record, _registry_map, _log, _leader, (port, pid, birth), _undeny = _ledger_only_debt(
        tmp_path, monkeypatch, scripts, task, survivor=survivor)
    try:
        found = read()
        assert found, f"{reader} released a launch whose ledger row still names a live child"
        assert _answers(port) and _same_process_alive(pid, birth), "reading signals nothing"
    finally:
        _settle(record.proc, identities=[(pid, birth)])


def test_a_stale_copy_keeps_the_candidate_and_the_log_of_a_launch_that_still_owes(tmp_path, monkeypatch, scripts):
    """The three things the body candidate's answer authorizes: handing the candidate over, retiring its checkout, and
    pruning the service's live log. None happens while the ledger names a live child behind a stale copy."""
    from ouroboros import body_candidate
    from ouroboros.tools import services

    data, ctx, record, _registry_map, log, _leader, (_port, pid, birth), _undeny = _ledger_only_debt(
        tmp_path, monkeypatch, scripts, "t-a")
    monkeypatch.setattr(services, "_SERVICES", {})  # that worker's memory is gone; only custody speaks now
    try:
        monkeypatch.setattr(body_candidate, "_owner_is_terminal", lambda ctx, task_id: True)
        with pytest.raises(body_candidate.CandidateRefused) as refused:
            body_candidate._transfer(ctx, {"task_id": "t-a", "candidate_id": "c-1"}, "t-b", via="resume")
        assert "CANDIDATE_OWNER_PROCESSES_LIVE" in str(refused.value.args) or refused.value.code == "CANDIDATE_OWNER_PROCESSES_LIVE"

        report = services.prune_service_logs(data, retention_days=0, now=time.time() + 60)

        assert report["retained_files"] == 1 and log.exists(), report
    finally:
        _settle(record.proc, identities=[(pid, birth)])


@pytest.mark.parametrize("stop", ["next start", "reaper", "quiesce"])
def test_every_stop_reads_what_only_the_ledger_names_and_ends_it(tmp_path, monkeypatch, scripts, stop):
    """The stale copy does not narrow a stop either: each ends C from what the ledger names, and retires the launch."""
    from ouroboros import owned_shutdown, process_custody

    data, _ctx, record, registry, _log, _leader, (port, pid, birth), undeny = _ledger_only_debt(
        tmp_path, monkeypatch, scripts, "t-a")
    undeny()
    registry.clear()  # that worker died: no in-memory owner
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: _PATIENCE)
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    try:
        if stop == "next start":
            owned_shutdown._stamp(data, "stop_requested_at", owned_shutdown._is_stop_target)
            finished = owned_shutdown.finish_unconfirmed_stops(data)
            assert finished["confirmed"] == 1 and finished["unconfirmed"] == 0, finished
        elif stop == "exit":
            assert owned_shutdown.stop_owned_work(data)["state"] == "completed"
        elif stop == "reaper":
            monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
            process_custody.reap_orphaned_processes(data, running_task_ids=set())
        else:
            assert process_custody.quiesce_custodied_services(data) == (True, [])
        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert owned_shutdown.owned_records(data, strict=True) == [], "a confirmed stop retires the launch"
    finally:
        _settle(record.proc, identities=[(pid, birth)])


# --------------------------------------------------------------------------- a copy names what the ledger no longer holds


class _Compaction:
    """The next generation's sweep A reads the table before L starts C and pauses before its signal. An ordinary stop B
    then observes L -> C and pauses before writing it. A ends L, confirms, and compacts the launch from both stores. B
    then publishes while another holder keeps the ledger's append lock across all of it: C lands as a pending copy with
    no canonical row left for it, and B's report is unconfirmed (C refuses its signal). ``publish()`` runs that last
    part. C's denial is only there to keep C alive; ``allow()`` lifts it."""

    def __init__(self, tmp_path, monkeypatch, scripts, task):
        from ouroboros import platform_layer, process_custody
        from ouroboros.tools import services

        self.data, self.monkeypatch = tmp_path / "data", monkeypatch
        registry = _registry(tmp_path, monkeypatch, task_id=task)
        port_file, trigger = tmp_path / "a.port", tmp_path / "start-the-child"
        started = json.loads(registry.execute("start_service", {
            "name": "a", "cmd": _runner_cmd(scripts, port_file, "separate", *HIDDEN, f"late={trigger}"), "readiness": {}}))
        self.leader, self.ctx = started["pid"], registry._ctx
        self.record = services._SERVICES[services._service_key(self.ctx, "a")]
        self.go_signal, self.go_stop, self.done = threading.Event(), threading.Event(), threading.Event()
        self.results, self.denied, self.identities = {}, set(), []
        a_at_signal, b_at_ahead = threading.Event(), threading.Event()
        real_deliver, real_ahead = platform_layer.deliver_signal, process_custody.write_ahead

        def deliver(target, *, graceful):
            if threading.current_thread().name == "sweep":
                a_at_signal.set()
                assert self.go_signal.wait(_PATIENCE)
            if int(target) in self.denied:
                return platform_layer.SIGNAL_DENIED
            return real_deliver(target, graceful=graceful)

        def ahead(*args, **kwargs):
            if threading.current_thread().name == "stopper":
                b_at_ahead.set()
                assert self.go_stop.wait(_PATIENCE)
            return real_ahead(*args, **kwargs)

        def stopper():
            try:
                self.results["b"] = services._stop_service(self.ctx, name="a")
            finally:
                self.done.set()

        monkeypatch.setattr(platform_layer, "deliver_signal", deliver)
        monkeypatch.setattr(process_custody, "write_ahead", ahead)
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        self.sweep = threading.Thread(target=lambda: process_custody.reap_orphaned_processes(
            self.data, running_task_ids=set()), name="sweep")
        self.stopper = threading.Thread(target=stopper, name="stopper")
        self.sweep.start()
        assert a_at_signal.wait(_PATIENCE), "A read the table and reached its signal with no child to see"
        trigger.write_text("now", encoding="utf-8")
        self.port, self.pid, self.birth = _server(port_file)
        self.identities.append((self.pid, self.birth))
        self.denied.add(self.pid)
        self.stopper.start()
        assert b_at_ahead.wait(_PATIENCE), "B observed L -> C"
        self.go_signal.set()
        self.sweep.join(_PATIENCE)
        assert not self.sweep.is_alive()

    def publish(self, *, folded=False):
        from ouroboros import owned_shutdown, process_custody
        from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
        from ouroboros.utils import jsonl_append_lock_path

        assert not _rows_of(self.data, self.leader) and _owed(self.data, self.leader) == ([], []), "A retired the launch"
        lock_path = jsonl_append_lock_path(process_custody.ledger_path(self.data))
        holder = acquire_exclusive_file_lock(lock_path, timeout_sec=_PATIENCE)
        assert holder is not None
        real_wait = process_custody._lock_wait
        self.monkeypatch.setattr(process_custody, "_lock_wait", lambda: 0.0)  # its one bounded wait, already spent
        try:
            self.go_stop.set()
            assert self.done.wait(_PATIENCE), "B never returned"
        finally:
            self.monkeypatch.setattr(process_custody, "_lock_wait", real_wait)
            release_exclusive_file_lock(lock_path, holder)
        self.stopper.join(_PATIENCE)
        assert self.results["b"].startswith("⚠️ SERVICE_STOP_ERROR"), self.results["b"]
        assert not _rows_of(self.data, self.leader), "no canonical row"
        pending = self.data / "state" / owned_shutdown.PENDING_DIRNAME
        assert list(pending.glob("launch-*.json")), "only a pending copy names C"
        if folded:
            assert owned_shutdown._update(self.data, lambda document: False)
            assert not list(pending.glob("launch-*.json"))
        assert self.pid in _owed(self.data, self.leader)[1]

    def allow(self):
        self.denied.clear()

    def close(self):
        self.go_signal.set()
        self.go_stop.set()
        for thread in (self.sweep, self.stopper):
            if thread.ident is not None:
                thread.join(_PATIENCE)
        _settle(self.record.proc, identities=self.identities)


def _rows_of(data, leader):
    from ouroboros import process_custody

    return [row for row in process_custody._read_ledger(data) if row.get("pid") == leader]


@pytest.mark.parametrize("where", ["pending copy", "folded into the document"])
@pytest.mark.parametrize("reader", chosen_rows(READERS, 0, 3))
def test_every_reader_holds_a_launch_only_a_copy_names_after_compaction(tmp_path, monkeypatch, scripts, reader, where):
    """That schedule, then each reader. C was published, so unlike the late child nobody observed yet, the launch must
    hold even though its canonical row is gone; reading signals nothing."""
    (tmp_path / "data").mkdir()
    read, task = _reader(tmp_path / "data", monkeypatch, reader)
    schedule = _Compaction(tmp_path, monkeypatch, scripts, task)
    try:
        schedule.publish(folded=where != "pending copy")
        assert _held(read(), schedule.leader) == "live"
        assert _answers(schedule.port) and _same_process_alive(schedule.pid, schedule.birth), "reading signals nothing"
    finally:
        schedule.close()


@pytest.mark.parametrize("stop", ["reaper", "quiesce", "exit"])
def test_the_stops_select_and_retire_a_launch_only_a_copy_names(tmp_path, monkeypatch, scripts, stop):
    """Reaper and quiesce used to select from the ledger alone: C stayed named and alive with nobody to end it. Each now
    ends it and leaves no record behind, with no ledger row to compact."""
    from ouroboros import owned_shutdown, process_custody
    from ouroboros.tools import services

    schedule = _Compaction(tmp_path, monkeypatch, scripts, "t-a")
    data = schedule.data
    try:
        schedule.publish()
        schedule.allow()
        services._SERVICES.clear()  # that worker died: no in-memory owner
        monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: _PATIENCE)
        monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
        if stop == "reaper":
            process_custody.reap_orphaned_processes(data, running_task_ids=set())
        elif stop == "quiesce":
            assert process_custody.quiesce_custodied_services(data) == (True, [])
        else:
            assert owned_shutdown.stop_owned_work(data)["state"] == "completed"
        assert not _answers(schedule.port) and not _same_process_alive(schedule.pid, schedule.birth), (
            f"{stop} never selected the launch")
        assert owned_shutdown.owned_records(data, strict=True) == [] and _owed(data, schedule.leader) == ([], [])
    finally:
        schedule.close()


# --------------------------------------------------------------------------- controlled identities: what is read


def _copy_only(tmp_path, monkeypatch, task, *, pending=True, ledger_file=True):
    """Launch A: its ledger row gone (or the ledger never written), only an ownership copy names C, alive in the table."""
    from ouroboros import owned_shutdown, process_census, process_custody

    rows = {_CHILD: process_census._Row(_CHILD, 1, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}
    world = _controlled(tmp_path, monkeypatch, rows)
    row = {**_launch_row(task, purpose="service:a"), "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH}]}
    if ledger_file:
        process_custody.ledger_path(tmp_path).write_bytes(b"")
    assert owned_shutdown.record_ledgered_process(tmp_path, row, contended=pending)
    return rows, world, row


def test_a_launch_only_a_copy_names_holds_with_an_empty_or_missing_ledger(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for ledger_file in [True, False]:
        for pending in [True, False]:
            for reader in READERS:
                tmp_path = next(case_paths)
                with subtests.test(ledger_file=ledger_file, pending=pending, reader=reader), monkeypatch.context() as monkeypatch:
                    read, task = _reader(tmp_path, monkeypatch, reader)
                    rows, world, _row = _copy_only(tmp_path, monkeypatch, task, pending=pending, ledger_file=ledger_file)
                    assert _held(read(), _PID) == "live"
                    assert world.signals == [] and _CHILD in rows


@pytest.mark.parametrize("reader", ["presence", "continuation", "cold sleep"])
def test_another_owners_copy_only_launch_does_not_hold_this_reader(tmp_path, monkeypatch, reader):
    """Combining by launch comes before the owner filter, and the owner filter still applies."""
    read, _task = _reader(tmp_path, monkeypatch, reader)
    _copy_only(tmp_path, monkeypatch, "someone-else")
    assert _held(read(), _PID) == "none"


def test_a_store_a_copy_only_launch_might_live_in_is_unknown_not_absent(subtests, tmp_path, monkeypatch):
    """Nothing else names this launch, so nothing can say it is not this reader's: the answer is a blocker for every
    owner until the store reads again. Readable again, the launch holds."""
    case_paths = case_folders(tmp_path)
    for case in ["pending file unreadable", "document unreadable", "pending directory unreadable"]:
        for reader in READERS:
            tmp_path = next(case_paths)
            with subtests.test(case=case, reader=reader), monkeypatch.context() as monkeypatch:
                from ouroboros import owned_shutdown

                read, task = _reader(tmp_path, monkeypatch, reader)
                rows, _world, _row = _copy_only(tmp_path, monkeypatch, task)
                pending_dir = tmp_path / "state" / owned_shutdown.PENDING_DIRNAME
                undo = []
                if case == "pending file unreadable":
                    [target] = list(pending_dir.glob("launch-*.json"))
                    intact = target.read_bytes()
                    target.write_bytes(b"{torn")
                    undo.append(lambda: target.write_bytes(intact))
                elif case == "pending directory unreadable":
                    moved = pending_dir.with_name(pending_dir.name + ".moved")
                    pending_dir.rename(moved)
                    pending_dir.write_bytes(b"")

                    def restore():
                        pending_dir.unlink()
                        moved.rename(pending_dir)

                    undo.append(restore)
                else:
                    path = owned_shutdown.owned_processes_path(tmp_path)
                    path.write_bytes(b"{torn")
                    undo.append(path.unlink)
                try:
                    assert read(), f"{reader} called unreadable custody empty"
                finally:
                    for restore_store in undo:
                        restore_store()
                assert _held(read(), _PID) == "live"
                assert _CHILD in rows


@pytest.mark.parametrize("reader", ["presence", "stop"])
def test_a_damaged_copy_of_a_launch_another_store_attributes_stays_with_that_launch(tmp_path, monkeypatch, reader):
    """The ledger says whose launch B is, so its one damaged copy is B's owner's unknown alone: it blocks neither another
    owner's reader nor a stop of another launch."""
    from ouroboros import owned_shutdown, process_custody
    from ouroboros.presence_continuation import _custody_blockers

    world = _controlled(tmp_path, monkeypatch, {})
    mine, theirs = _launch_row("t-a", purpose="service:a"), _launch_row("t-b", marker=_MARK_B, purpose="service:b", pid=_OTHER)
    world.spawned(mine)
    world.spawned(theirs)
    pending = tmp_path / "state" / owned_shutdown.PENDING_DIRNAME
    pending.mkdir(parents=True, exist_ok=True)
    (pending / f"launch-{_HEX_B}.damaged.json").write_text("{torn", encoding="utf-8")

    if reader == "presence":
        assert _custody_blockers(tmp_path, "t-a") == []
        assert _held(_custody_blockers(tmp_path, "t-b"), _OTHER) == "unknown"
    else:
        assert process_custody._owned_copies(tmp_path, [theirs]) is None, "B's own stop cannot read its copy: unknown"
        assert process_custody.end_service_rows(tmp_path, [mine])[id(mine)]["exit_confirmed"] is True


def test_a_stop_does_not_read_a_damaged_copy_of_a_launch_it_does_not_stop(tmp_path, monkeypatch):
    """Nothing attributes that launch, but a stop asks about its own launches only: it ends what it was given."""
    from ouroboros import owned_shutdown, process_custody

    world = _controlled(tmp_path, monkeypatch, {})
    mine = _launch_row("t-a", purpose="service:a")
    world.spawned(mine)
    pending = tmp_path / "state" / owned_shutdown.PENDING_DIRNAME
    pending.mkdir(parents=True, exist_ok=True)
    (pending / f"launch-{_HEX_B}.damaged.json").write_text("{torn", encoding="utf-8")

    assert process_custody.end_service_rows(tmp_path, [mine])[id(mine)]["exit_confirmed"] is True
    process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())
    assert not [row for row in process_custody._read_ledger(tmp_path) if row.get("scope_marker") == _MARK_A]


def test_quiesce_refuses_an_update_over_a_launch_it_cannot_read(tmp_path, monkeypatch):
    """Quiesce answers "is it safe to replace the code": a copy nobody can attribute may name a live process."""
    from ouroboros import owned_shutdown, process_custody

    _controlled(tmp_path, monkeypatch, {})
    process_custody.ledger_path(tmp_path).write_bytes(b"")
    pending = tmp_path / "state" / owned_shutdown.PENDING_DIRNAME
    pending.mkdir(parents=True, exist_ok=True)
    damaged = pending / f"launch-{_HEX_B}.damaged.json"
    damaged.write_text("{torn", encoding="utf-8")

    ok, blockers = process_custody.quiesce_custodied_services(tmp_path, timeout_sec=0)
    assert ok is False and blockers, blockers

    damaged.unlink()
    assert process_custody.quiesce_custodied_services(tmp_path, timeout_sec=0) == (True, [])


def test_a_leftover_executor_record_reads_what_its_launch_owes_from_every_store(tmp_path, monkeypatch, scripts):
    """The control stop over a dead worker's durable record has neither memory nor a ledger row naming C: only a pending
    copy does. Its one pass still takes C as input, by identity, and only then forgets the record."""
    from ouroboros import owned_shutdown, process_custody
    from ouroboros.tools import services
    from tests.test_launch_stop_contract import _dead_worker

    data = tmp_path / "data"
    data.mkdir()
    _ctx, record, leader, (port, pid, birth), _root = _dead_worker(tmp_path, monkeypatch, scripts)
    try:
        [row] = [item for item in process_custody._read_ledger(data) if item["pid"] == leader]
        assert owned_shutdown.record_ledgered_process(
            data, {**row, "known_members": [{"pid": pid, "birth": birth, "held": True}]}, contended=True)
        assert _owed(data, leader) == ([], [pid])

        [stopped] = services.kill_all_services(data, wait=True)

        assert stopped["state"] == "stopped" and not record.durable_record_path.exists(), stopped
        assert not _answers(port) and not _same_process_alive(pid, birth)
    finally:
        _settle(record.local_proc, identities=[(pid, birth)])


# --------------------------------------------------------------------------- batching


def test_a_launch_on_a_child_drive_is_read_from_its_own_ledger(subtests, tmp_path, monkeypatch):
    """The ownership set names the launch's drive, so the debt only that drive's ledger names (a stale copy) is read
    there, once, whichever root the reader was given. Without its ledger it is unknown, not settled."""
    case_paths = case_folders(tmp_path)
    for reader in READERS:
        tmp_path = next(case_paths)
        with subtests.test(reader=reader), monkeypatch.context() as monkeypatch:
            from ouroboros import owned_shutdown, process_census, process_custody
            from ouroboros.utils import append_jsonl

            read, task = _reader(tmp_path, monkeypatch, reader)
            rows = {_CHILD: process_census._Row(_CHILD, 1, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}
            _controlled(tmp_path, monkeypatch, rows)
            process_custody.ledger_path(tmp_path).write_bytes(b"")
            drive = tmp_path / "task_drives" / "child-task"
            (drive / "state").mkdir(parents=True)
            row = _launch_row(task, purpose="service:a")
            assert append_jsonl(process_custody.ledger_path(drive), row)
            assert owned_shutdown.record_ledgered_process(drive, row)
            assert owned_shutdown.installation_root(drive) == owned_shutdown.installation_root(tmp_path)
            assert _held(read(), _PID) == "none", "nothing is owed yet"
            assert append_jsonl(process_custody.ledger_path(drive), {**row, "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH}]})
            reads, real = [], process_custody._read_ledger_records

            def counted(path, **kwargs):
                reads.append(pathlib.Path(path).resolve())
                return real(path, **kwargs)

            monkeypatch.setattr(process_custody, "_read_ledger_records", counted)
            found = read()

            assert _held(found, _PID) == "live", found
            assert reads.count(drive.resolve()) == 1, reads
            process_custody.ledger_path(drive).write_bytes(b"\xff\xfe not a row\n")
            assert _held(read(), _PID) == "unknown", "that drive's ledger cannot be read: unknown, never settled"


def test_a_healthy_read_keeps_one_identity_read_and_no_census(subtests, tmp_path, monkeypatch):
    """A live measured leader answers from one batched identity read: the combined selection adds no census, no second
    read of a store, nothing per launch."""
    case_paths = case_folders(tmp_path)
    for reader in READERS:
        tmp_path = next(case_paths)
        with subtests.test(reader=reader), monkeypatch.context() as monkeypatch:
            from ouroboros import owned_shutdown, process_census, process_custody, process_scope

            read, task = _reader(tmp_path, monkeypatch, reader)
            launches = [(_PID + index, "%032x" % (0xa0 + index)) for index in range(3)]
            rows = {pid: process_census._Row(pid, 2, pid, _OLD_BIRTH, 123.0, "alive", set()) for pid, _hex in launches}
            world = _controlled(tmp_path, monkeypatch, rows)
            for pid, value in launches:
                world.spawned(_launch_row(task, marker="OURO_LAUNCH_SCOPE_" + value, purpose="service:a", pid=pid))
            identity_reads, real_identities = [], process_scope.read_identities

            def identities(pids, **kwargs):
                listed = sorted(pids)
                identity_reads.append(listed)
                return real_identities(listed, **kwargs)

            monkeypatch.setattr(process_scope, "read_identities", identities)
            documents, real_document = [], owned_shutdown._read_document
            monkeypatch.setattr(owned_shutdown, "_read_document",
                                lambda *a, **k: documents.append(1) or real_document(*a, **k))
            ledgers, real_ledger = [], process_custody._read_ledger_records
            monkeypatch.setattr(process_custody, "_read_ledger_records",
                                lambda *a, **k: ledgers.append(1) or real_ledger(*a, **k))
            del world.censuses[:]

            found = read()

            assert found and world.censuses == [], "a live measured leader needs no table"
            assert identity_reads == [sorted(pid for pid, _hex in launches)], identity_reads
            assert len(documents) == 1, "one read of the ownership set"
            assert len(ledgers) == 1, "one read of the one ledger these launches live in"


def test_one_census_answers_a_batch_of_dead_leaders(tmp_path, monkeypatch):
    """Selected from either store, every launch whose leader cannot answer shares one table."""
    from ouroboros import process_census, process_custody
    from ouroboros.presence_continuation import _custody_blockers

    rows = {_CHILD: process_census._Row(_CHILD, 1, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}
    world = _controlled(tmp_path, monkeypatch, rows)
    process_custody.ledger_path(tmp_path).write_bytes(b"")
    world.spawned(_launch_row("t-a", purpose="service:a"))
    _second = {**_launch_row("t-a", marker=_MARK_B, purpose="service:b", pid=_OTHER),
               "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH}]}
    from ouroboros import owned_shutdown

    assert owned_shutdown.record_ledgered_process(tmp_path, _second, contended=True)
    del world.censuses[:]

    found = _custody_blockers(tmp_path, "t-a")

    assert _held(found, _OTHER) == "live" and _held(found, _PID) == "none", found
    assert len(world.censuses) == 1, world.censuses


# --------------------------------------------------------------------------- writers


def test_a_late_pending_copy_with_another_fingerprint_spelling_cannot_erase_what_a_launch_owes(tmp_path, monkeypatch):
    """A launch's record is its marker: a copy rebuilt without the registration's fingerprint (an executor record's row
    once the ledger lost it) still folds into the stored launch, never replaces it."""
    from ouroboros import owned_shutdown
    from tests._launch_stop_shared import _CARRIER

    (tmp_path / "state").mkdir()
    stored = {**_launch_row("t-a", purpose="workspace_service:a"), "fingerprint": {"start_time": "birth"},
              "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH}]}
    assert owned_shutdown.record_ledgered_process(tmp_path, stored)
    rebuilt = {**_launch_row("t-a", purpose="workspace_service:a"), "known_members": [{"pid": _CARRIER, "birth": _OLD_BIRTH}]}
    assert owned_shutdown.record_ledgered_process(tmp_path, rebuilt, contended=True)
    assert owned_shutdown._update(tmp_path, lambda document: False)

    [record] = owned_shutdown.owned_records(tmp_path, strict=True)
    assert {item["pid"] for item in record["ledger_entry"]["known_members"]} == {_CHILD, _CARRIER}


def test_a_document_that_holds_no_set_is_invalid_not_absent(tmp_path):
    """Only a missing file reads as an empty set. A JSON ``null`` (or any non-set) stays unknown to a strict reader, and
    no writer replaces it."""
    from ouroboros import owned_shutdown

    (tmp_path / "state").mkdir()
    path = owned_shutdown.owned_processes_path(tmp_path)
    assert owned_shutdown.owned_records(tmp_path, strict=True) == []
    for content in (b"null\n", b"[]\n", b'{"records": null}\n'):
        path.write_bytes(content)
        with pytest.raises(ValueError):
            owned_shutdown.owned_records(tmp_path, strict=True)
        assert owned_shutdown._update(tmp_path, lambda document: True) is False
        assert path.read_bytes() == content, "not rebuilt from nothing"


def test_a_stop_during_registration_can_write_to_the_copy_after_compaction(tmp_path, monkeypatch):
    """A stop whose Popen has no row yet resolves the launch by marker in the document too, not only in the ledger: its
    observation is durable although the compaction left the document's copy as the launch's only row."""
    from ouroboros import owned_shutdown, process_custody

    _controlled(tmp_path, monkeypatch, {})
    process_custody.ledger_path(tmp_path).write_bytes(b"")
    row = _launch_row("t-a", purpose="service:a")
    assert owned_shutdown.record_ledgered_process(tmp_path, row)
    child = [{"pid": _CHILD, "birth": _OLD_BIRTH}]

    assert process_custody.write_ahead((tmp_path, {"pid": _PID, "scope_marker": _MARK_A}), child) is True
    assert _owed(tmp_path, _PID) == ([_CHILD], [_CHILD])
    assert _HEX_A in {record["record_id"].rsplit("-", 1)[-1] for record in owned_shutdown.owned_records(tmp_path, strict=True)}


def test_the_exit_adds_debt_only_the_ledger_names_to_its_one_stop(tmp_path, monkeypatch):
    """The stop that retired a launch's record finished; another stop's later row names C but its ownership write failed.
    No record names the launch, so the exit's targets would miss it and report itself complete. The exit reads that
    debt, its one stop ends C and retires it, and a launch the ledger names with nothing owed adds nothing to the exit."""
    from ouroboros import owned_shutdown, process_census, process_custody
    from ouroboros.utils import append_jsonl

    rows = {_CHILD: process_census._Row(_CHILD, 1, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}
    _controlled(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: _PATIENCE)
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    settled = _launch_row("t-b", marker=_MARK_B, purpose="service:b", pid=_OTHER)
    assert append_jsonl(process_custody.ledger_path(tmp_path), settled)
    row = {**_launch_row("t-a", purpose="service:a"), "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH}]}
    assert append_jsonl(process_custody.ledger_path(tmp_path), row)
    assert owned_shutdown.owned_records(tmp_path, strict=True) == []

    outcome = owned_shutdown.stop_owned_work(tmp_path)

    assert _CHILD not in rows, outcome
    assert outcome["state"] == "completed" and outcome["targets"] == 1, outcome
    assert owned_shutdown.owned_records(tmp_path, strict=True) == []


@pytest.mark.parametrize("case", ["owed, copies refused", "ledger malformed", "set unreadable"])
def test_the_exit_never_reports_completion_over_ledger_debt_it_could_not_count(tmp_path, monkeypatch, case):
    """Debt only the ledger names reaches the exit's stop through its ownership copy. When neither the document nor the
    pending channel takes that copy for the whole exit, or the ledger or the set cannot be read, the exit cannot count
    that debt: its answer is unconfirmed with the launch's key or the reason, the other stop owners still run, and a
    later door's join keeps that answer after the writes recover; the ledger keeps C for the next start's sweep. A copy
    that is written lets the one stop end C, and with nothing owed a refused write changes nothing."""
    from ouroboros import owned_shutdown, process_census, process_custody
    from ouroboros.utils import append_jsonl

    rows = {_CHILD: process_census._Row(_CHILD, 1, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}
    _controlled(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: _PATIENCE)
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    ran, real_owners = [], owned_shutdown._memory_owners
    monkeypatch.setattr(owned_shutdown, "_memory_owners", lambda root, deadline: [
        *real_owners(root, deadline), ("probe", lambda: ran.append("probe"))])
    ledger = process_custody.ledger_path(tmp_path)
    assert append_jsonl(ledger, _launch_row("t-b", marker=_MARK_B, purpose="service:b", pid=_OTHER))
    if case != "nothing owed, copies refused":
        assert append_jsonl(ledger, {**_launch_row("t-a", purpose="service:a"),
                                     "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH}]})
    writes = _ownership_writes_fail(monkeypatch)
    writes["failing"] = "refused" in case
    aside = ledger.with_name("ledger-aside")
    if case == "ledger unreadable":
        ledger.rename(aside)
        ledger.mkdir()  # a read of the ledger now fails with the OS's own error
    elif case == "ledger malformed":
        ledger.rename(aside)
        ledger.write_bytes(b"{torn\n")
    elif case == "set unreadable":
        owned_shutdown.owned_processes_path(tmp_path).write_bytes(b"{torn")

    outcome = owned_shutdown.stop_owned_work(tmp_path)

    assert ran == ["probe"], "every other stop owner still ran"
    if case in ("owed, copy written", "nothing owed, copies refused"):
        assert outcome["state"] == "completed" and not outcome.get("ledger_unconfirmed"), outcome
        assert outcome["targets"] == (1 if case == "owed, copy written" else 0), outcome
        assert (_CHILD in rows) is (case == "nothing owed, copies refused"), rows
        return
    assert outcome["state"] == "unconfirmed", f"the exit claimed completion over debt it could not count: {outcome}"
    assert _CHILD in rows, "this exit does not stop what it could not count"
    if case == "owed, copies refused":
        assert outcome["ledger_unconfirmed"] == {"records": [f"launch-{_HEX_A}"]}, outcome
        logged = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text(
            encoding="utf-8").splitlines() if '"owned_stop_unconfirmed"' in line]
        assert logged[-1]["ledger"] == outcome["ledger_unconfirmed"], logged
    else:  # the reason, in either report field
        assert "cannot be read" in json.dumps(outcome) or "ownership_set_unreadable" in outcome["unconfirmed"], outcome
    writes["failing"] = False
    if case == "ledger unreadable":
        ledger.rmdir()
        aside.rename(ledger)
    elif case == "ledger malformed":
        ledger.unlink()
        aside.rename(ledger)
    elif case == "set unreadable":
        owned_shutdown.owned_processes_path(tmp_path).unlink()
    joined = owned_shutdown.stop_owned_work(tmp_path)  # a later door: the stores answer again, the exit did not change
    assert joined["state"] == "unconfirmed" and joined.get("ledger_unconfirmed") == outcome.get("ledger_unconfirmed")
    assert _owed(tmp_path, _PID)[0] == [_CHILD], "the ledger still names C"
    process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())  # the next start's sweep
    assert _CHILD not in rows, "the next start's sweep ends what the exit could not"


# Pids above every host's pid range: a missed fake can signal nothing real.
_LEADER, _REUSED, _UNREAD, _LEGACY, _HELD = 999_999_901, 999_999_902, 999_999_903, 999_999_904, 999_999_905
_EXECED, _REFUTED, _GONE, _UNCONFIRMED = 999_999_906, 999_999_907, 999_999_908, 999_999_909
_BIRTH = "Mon Nov 24 10:00:00 2026"


def _kept(rows, *, blind="", notes=(), unknown=()):
    from types import SimpleNamespace

    from ouroboros import launch_custody

    return launch_custody._KeptLaunches(
        SimpleNamespace(blind=blind, unknown={id(row) for row in unknown}, readable=True), rows,
        {id(row): [{"record_id": row["rid"], "host_pid": row["pid"]}] for row in rows}, {}, {}, [], [*notes])


def _row(pid, rid, birth="", **fingerprint):
    return {"pid": pid, "rid": rid, "scope_birth": birth, "purpose": "service:web", "scope": "session",
            "fingerprint": fingerprint}


def test_the_kept_spare_proves_leaders_keeps_mains_retention_rule_and_names_every_gap(monkeypatch):
    """Every OS observation is a double. A leader with a recorded birth is judged by it in one batched read (spared
    whatever its command became after an ``exec``; another birth is a reused pid); one without, by its strict measured
    fingerprint, refuted only by a measured birth mismatch, ended only by a provable exit (a liveness probe that cannot
    see it proves nothing). Every other leader is unknown and named in ``blind``; it is spared only as main's retention
    rule (``_fingerprint_matches``) spares a recorded row, never by its live pid alone. Every store gap the selection
    reports, a damaged copy no store attributes included, is in ``blind`` too; a selection that fails to read spares
    nothing and says so."""
    from ouroboros import launch_custody, platform_layer, process_census, process_containment, process_custody
    from ouroboros.process_census import IDENTITY_ALIVE, IDENTITY_UNREADABLE, _Identity

    rows = [_row(_LEADER, "a", _BIRTH, cmd_sha256="argv-before-exec"), _row(_REUSED, "b", "Mon Nov 24 10:00:01 2026"),
            _row(_UNREAD, "c", "Mon Nov 24 10:00:02 2026", start_time="t0", cmd_sha256="h0"), _row(_LEGACY, "d"),
            _row(_HELD, "e", start_time="t1", cmd_sha256="h1"), _row(_EXECED, "f", start_time="t2", cmd_sha256="h2"),
            _row(_REFUTED, "g", start_time="t3", cmd_sha256="h3"), _row(_GONE, "h", start_time="t4", cmd_sha256="h4"),
            _row(_UNCONFIRMED, "i", start_time="t5", cmd_sha256="h5")]
    reads, strict, retained = [], [], []
    measured = {_EXECED: ["cmd_sha256"], _REFUTED: ["start_time"]}
    monkeypatch.setattr(process_census, "read_identities", lambda pids: reads.append(sorted(pids)) or {
        _LEADER: _Identity(IDENTITY_ALIVE, _BIRTH), _REUSED: _Identity(IDENTITY_ALIVE, "Tue Nov 25 08:00:00 2026"),
        _UNREAD: _Identity(IDENTITY_UNREADABLE)})
    monkeypatch.setattr(platform_layer, "pid_provably_gone", lambda pid: pid == _GONE)
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: pid not in (_GONE, _UNCONFIRMED))
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda pid: False)

    def fingerprint(row, require_measured=False):
        (strict if require_measured else retained).append(row["pid"])
        return row["pid"] == _HELD or (not require_measured and row["pid"] == _UNREAD)

    monkeypatch.setattr(process_custody, "_fingerprint_matches", fingerprint)
    monkeypatch.setattr(process_custody, "measured_identity_mismatch",
                        lambda row: {"mismatches": measured[row["pid"]]} if row["pid"] in measured else {})
    damaged = "a pending ownership copy (launch-x) cannot be read, and no other store says whose launch it is"
    monkeypatch.setattr(launch_custody, "_kept_launches", lambda root: _kept(rows, blind=damaged, notes=[damaged]))
    spare = launch_custody.kept_service_spare("/nonexistent-drive")
    assert reads == [[_LEADER, _REUSED, _UNREAD]]
    assert strict == [_LEGACY, _HELD, _EXECED, _REFUTED, _UNCONFIRMED] and retained == [_UNREAD, _EXECED, _UNCONFIRMED]
    assert spare.pids == {_LEADER, _UNREAD, _HELD}, "no leader is spared by its live pid alone"
    assert spare.blind == (f"{damaged}; kept launch(es) pid-999999903, pid-999999904, pid-999999906, pid-999999909 "
                           "may run: their leader's identity is unknown")
    monkeypatch.setattr(launch_custody, "_kept_launches", lambda root: _kept(
        rows[:1], blind=launch_custody._SET_UNREADABLE, notes=[launch_custody._SET_UNREADABLE]))
    assert launch_custody.kept_service_spare("/nonexistent-drive") == ({_LEADER}, launch_custody._SET_UNREADABLE)

    def unreadable(root):
        raise OSError("the state directory cannot be listed")

    monkeypatch.setattr(launch_custody, "_kept_launches", unreadable)
    assert launch_custody.kept_service_spare("/nonexistent-drive") == (set(), "kept services could not be read: OSError")


def test_an_unreadable_kept_executor_record_is_a_gap_and_a_forgotten_one_is_not(tmp_path, monkeypatch):
    from ouroboros import launch_custody

    (tmp_path / "torn.json").write_text("{", encoding="utf-8")
    records = [{"record_id": rid, "kind": "service", "record_path": str(tmp_path / name)}
               for rid, name in (("svc-torn", "torn.json"), ("svc-gone", "gone.json"))]
    monkeypatch.setattr(launch_custody, "read_custody", lambda root, strict=True, every_drive=False:
                        launch_custody.LaunchCustody(True, [], {}, set(), "", b"", records))
    assert launch_custody._kept_launches(tmp_path).notes == [
        "executor record svc-torn cannot be read: it may be a kept service"]


def _secondary_drive(tmp_path):
    fork = tmp_path / "state" / "headless_tasks" / "producer" / "data"
    fork.mkdir(parents=True)
    return fork


def _launch(pid, nonce, purpose, scope):
    """A ledger row as ``process_custody.record_process`` writes it for a launch with its own marker."""
    return {"pid": pid, "pgid": pid, "purpose": purpose, "scope": scope, "owner_task": "producer", "session_id": "s",
            "fingerprint": {}, "scope_marker": "OURO_LAUNCH_SCOPE_" + nonce * 32, "scope_birth": _BIRTH}


@pytest.mark.parametrize("gap", ["set unreadable", "ledger line torn", "pending copy damaged"])
def test_a_store_gap_never_empties_what_reads_and_is_named_beside_it(tmp_path, gap):
    """Real files in a temporary store; no process is read. One store may be all that names a launch, so a gap in any
    of them leaves every launch the others name in the selection, beside a note: an unreadable set still yields the
    launch only its pending copy names (on a task's own drive, read there), a ledger with a line nothing parses still
    yields the launch only its other line names, and a damaged pending copy no store attributes is named as such."""
    from ouroboros import launch_custody, owned_shutdown
    from ouroboros.process_custody import ledger_path

    row = _launch(_LEADER, "a", "service:kept-page", "session")
    pending = tmp_path / "state" / owned_shutdown.PENDING_DIRNAME
    if gap == "set unreadable":
        assert owned_shutdown.record_ledgered_process(_secondary_drive(tmp_path), row, contended=True)
        owned_shutdown.owned_processes_path(tmp_path).write_text("{", encoding="utf-8")
        note = launch_custody._SET_UNREADABLE
    elif gap == "ledger line torn":  # only the ledger names it
        ledger_path(tmp_path).parent.mkdir(parents=True)
        ledger_path(tmp_path).write_text(json.dumps(row) + "\n{torn\n", encoding="utf-8")
        note = "the process ledger cannot be read whole"
    else:
        assert owned_shutdown.record_ledgered_process(tmp_path, row)
        note = (f"a pending ownership copy (launch-{'d' * 32}) cannot be read, and no other store says whose launch "
                "it is")
    if gap != "ledger line torn":
        pending.mkdir(parents=True, exist_ok=True)
        (pending / f"launch-{'d' * 32}.torn.json").write_text("{", encoding="utf-8")
    custody = launch_custody.read_custody(tmp_path, every_drive=True)
    [found] = custody.rows
    assert found["scope_marker"] == row["scope_marker"]
    assert custody.readable is (gap != "ledger line torn") and custody.blind == ("" if gap == "ledger line torn" else note)
    assert (id(found) in custody.unknown) is (gap != "pending copy damaged"), "its stores were not read whole"
    kept = launch_custody._kept_launches(tmp_path)
    assert [launch["pid"] for launch in kept.rows] == [_LEADER] and kept.notes == [note]


@pytest.mark.parametrize("store", ["readable", "document unreadable"])
def test_panic_ends_the_unheld_kept_host_launch_and_its_owed_member_through_the_exit_stop(tmp_path, monkeypatch,
                                                                                          store):
    """The real exit stop on the inert OS model (``_controlled``: a pid table, measured signals, no OS process is read
    or signalled). A kept host launch on a task's own drive, its leader reparented, owes a member an earlier pass
    recorded that has left its group; neither shows a marker. Panic stamps every owned stop first (an earlier stamp
    stays) and hands the exit's own record stop exactly that launch, which signals its leader and that member by their
    recorded births, judges what the launch owes and retires its record: ``completed``. The launch ``tools.services``
    holds here (its launch id decides, not its pid), a local-executor launch (its executor's durable settle owns it)
    and a task process stay untouched. With the document unreadable, the launch only a pending copy names still ends,
    but its custody cannot be reconciled: the answer is ``unknown`` and its stamped record stays for the next start.
    It proves the POSIX measured route only: no Windows native stop and no executor alias owner runs here."""
    from types import SimpleNamespace

    from ouroboros import launch_custody, owned_shutdown, process_census, process_custody
    from ouroboros.tools import services
    from ouroboros.utils import append_jsonl
    from tests._launch_stop_shared import _CARRIER

    task_pid, member = _PID + 3, {"pid": _CHILD, "birth": _BIRTH}
    rows = {pid: process_census._Row(pid, 1, pid, _BIRTH, 123.0, "alive", set())
            for pid in (_PID, _CHILD, _CARRIER, _OTHER, task_pid)}  # reparented, each its own group, no marker shown
    world = _controlled(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(launch_custody, "_PANIC_ACCOUNTING_SEC", _PATIENCE)  # a bound for a loaded host, not a pace
    fork, unreadable = _secondary_drive(tmp_path), store == "document unreadable"

    def spawned(drive, row):  # what record_process writes: the drive's ledger row, then its ownership record
        assert append_jsonl(process_custody.ledger_path(drive), row)
        assert owned_shutdown.record_ledgered_process(drive, row)

    kept, held = _launch(_PID, "a", "service:kept-page", "session"), _launch(_CARRIER, "b", "service:held-page", "session")
    executor, task = _launch(_OTHER, "c", "workspace_service:api", "session"), _launch(task_pid, "e", "service:x", "task")
    for row in (held, executor, task):
        spawned(tmp_path, row)
    spawned(fork, kept)
    assert process_custody.note_owed(fork, kept, [member]) is not None  # an earlier pass recorded what it owes
    task_id, kept_id = "launch-" + "e" * 32, "launch-" + "a" * 32
    [earlier] = owned_shutdown._stamp(tmp_path, "stop_requested_at", lambda entry: entry["record_id"] == task_id)
    if unreadable:  # its copy waits as a pending file; then the document turns unreadable
        assert owned_shutdown.record_ledgered_process(fork, process_custody.latest_launch_row(fork, kept), contended=True)
        owned_shutdown.owned_processes_path(tmp_path).write_text("{", encoding="utf-8")
    assert _owed(fork, _PID) == ([_CHILD], [_CHILD]) and world.signals == []
    monkeypatch.setattr(services, "_SERVICES", {"producer:held-page": SimpleNamespace(
        scope_marker=held["scope_marker"], proc=SimpleNamespace(pid=_GONE))})

    outcome = launch_custody.panic_stop_kept(tmp_path)

    assert sorted(world.signals) == [(_PID, _BIRTH), (_CHILD, _BIRTH)], "the launch and its owed member, by birth"
    assert set(rows) == {_CARRIER, _OTHER, task_pid} and (world.kills, world.native) == ([], [])
    assert _owed(fork, _PID)[0] == [], "its ledger owes nothing once the stop judged that member"
    records = {record["record_id"]: record for record in owned_shutdown.owned_records(tmp_path)}
    if unreadable:
        assert outcome["state"] == "unknown" and outcome["unconfirmed"] == [kept_id], outcome
        assert outcome["note"].startswith(launch_custody._SET_UNREADABLE), outcome
        assert list(records) == [kept_id] and records[kept_id]["stop_requested_at"], "kept for the next start"
    else:
        assert outcome == {"state": "completed", "targets": 1, "unconfirmed": []}
        assert set(records) == {"launch-" + "b" * 32, "launch-" + "c" * 32, task_id}, "its record retired"
        assert all(record["stop_requested_at"] for record in records.values()), "every owned stop stamped"
        assert records[task_id]["stop_requested_at"] == earlier["stop_requested_at"], "an earlier stamp stays"
        assert _owed(fork, _PID) == ([], [])
