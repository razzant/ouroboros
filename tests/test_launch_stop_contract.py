"""One launch, every stop: what a stop reached only through a parent chain is durable before that stop signals, a pass
discharges only what it judged, and custody that cannot be read confirms nothing.

Real consumers over real processes wherever the schedule can be staged: a runner L whose server C leads another
session and shows no launch marker, so after L dies only what a stop recorded names C (``_launch_stop_shared``). C
answering a connection and its measured identity are the observations. The damaged-store matrix, where a real process
adds nothing to what is read, is in ``test_launch_stop_obligations``.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import threading
import time

import pytest

from tests._launch_stop_shared import (
    _PATIENCE,
    _answers,
    _end,
    _executor_ctx,
    _executor_start,
    _held,
    _ledgered_service,
    _owed,
    _payload,
    _reader,
    _registry,
    _runner_cmd,
    _same_process_alive,
    _server,
    _settle,
    _start,
    write_scripts,
)

pytestmark = [
              pytest.mark.skipif(os.name == "nt", reason="the measured stop is POSIX; Windows keeps its native kill")]

HIDDEN = ("hidden=1",)


@pytest.fixture
def scripts(tmp_path):
    return write_scripts(tmp_path)


def _roots_unknown(monkeypatch) -> dict:
    """The installation's daemon roots cannot be read until ``state["unknown"]`` clears: a chain-only member is held."""
    from ouroboros import process_custody

    state = {"unknown": True}
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda root: None if state["unknown"] else set())
    return state


def _leader_gone(pid: int) -> None:
    from ouroboros.process_census import IDENTITY_ALIVE, read_identities

    deadline = time.monotonic() + _PATIENCE
    while read_identities([pid])[pid].state == IDENTITY_ALIVE:
        assert time.monotonic() < deadline, f"leader {pid} never ended"
        time.sleep(0.02)


def _host(tmp_path, monkeypatch, scripts, name="a", task="stop-task"):
    """(ctx, record, its registry, its log, leader pid, the server's identity) of a real host service."""
    from ouroboros.tools import services

    registry = _registry(tmp_path, monkeypatch, task_id=task)
    leader = _start(registry, scripts, tmp_path / f"{name}.port", name, options=HIDDEN)["pid"]
    record = services._SERVICES[services._service_key(registry._ctx, name)]
    return registry._ctx, record, services._SERVICES, pathlib.Path(record.log_path), leader, _server(tmp_path / f"{name}.port")


def _executor(tmp_path, monkeypatch, scripts, name="a", task="exec-task"):
    """The same over a real local executor service."""
    from ouroboros import workspace_executor as executor

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path, task_id=task)
    _executor_start(ctx, workspace, scripts, tmp_path / f"{name}.port", name, options=HIDDEN)
    record = executor._SERVICES[f"{task}:{name}"]
    return (ctx, record, executor._SERVICES, pathlib.Path(record.backend_log_path), record.local_proc.pid,
            _server(tmp_path / f"{name}.port"))


BACKENDS = {"host service": _host, "local executor service": _executor}


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_a_batched_exit_answer_cannot_discharge_what_an_ordinary_stop_published_after_its_input(
        tmp_path, monkeypatch, scripts, backend):
    """The shutdown takes its input (no child) and pauses before its census. An ordinary ``stop_service`` then observes
    L -> C with the daemon roots unreadable, holds C, ends L and records C. The shutdown's census runs now and cannot
    find C, so its own answer is clean; C is new custody it never judged. Result, log and name stay, the readers hold
    the launch, and the retry ends C and retires it all."""
    from ouroboros import process_scope
    from ouroboros.loop_budget import _service_identity_projection
    from ouroboros.presence_continuation import _custody_blockers
    from ouroboros.tools import services

    data = tmp_path / "data"
    ctx, record, registry, log, leader, (port, pid, birth) = BACKENDS[backend](tmp_path, monkeypatch, scripts)
    roots = _roots_unknown(monkeypatch)
    at_census, stopped, real_census, result = threading.Event(), threading.Event(), process_scope.census, {}

    def census(*args, **kwargs):
        if threading.current_thread().name == "exit-owner":
            at_census.set()
            assert stopped.wait(_PATIENCE)
        return real_census(*args, **kwargs)

    monkeypatch.setattr(process_scope, "census", census)
    exit_owner = threading.Thread(target=lambda: result.update(stopped=services.kill_all_services(
        data, wait=True, durable=False)), name="exit-owner")
    exit_owner.start()
    try:
        assert at_census.wait(_PATIENCE), "the shutdown never reached its census"
        first = services._stop_service(ctx, name="a")
        assert first.startswith("⚠️ SERVICE_STOP_ERROR") and _payload(first)["stop_failed"] is True, first
        _leader_gone(leader)
        stopped.set()
        exit_owner.join(_PATIENCE)
        assert not exit_owner.is_alive()

        assert _answers(port) and _same_process_alive(pid, birth), "C lives"
        [payload] = result["stopped"]
        assert payload.get("stop_failed") is True and payload.get("cleanup_dispatched") is False, payload
        assert _service_identity_projection(payload)["process_cleanup"]["custody"] == "owed", "the task's result says why"
        assert record.service_id in registry, "the name stays with the launch while C lives"
        assert log.exists() and "log_finalization" not in payload, "its live log is not finalized"
        assert all(pid in names for names in _owed(data, leader)), "ledger row and ownership copy still name C"
        assert all(item.get("held") for item in record.known_members), "held: nothing told C from a daemon root"
        assert _same_process_alive(pid, birth), "the shutdown never signalled what it did not judge"
        roots["unknown"] = False
        assert _held(_custody_blockers(data, ctx.task_id), leader) == "live"

        retired = _payload(services._stop_service(ctx, name="a"))

        assert retired["process_cleanup"]["exit_confirmed"] is True and "stop_failed" not in retired, retired
        assert retired["log_finalization"]["deleted_live_log"] is True and not log.exists()
        assert record.service_id not in registry and not _answers(port) and not _same_process_alive(pid, birth)
        assert _held(_custody_blockers(data, ctx.task_id), leader) == "none"
        assert _owed(data, leader) == ([], [])
    finally:
        stopped.set()
        exit_owner.join(_PATIENCE)
        _end(pid, birth)


def _signals_see_custody(monkeypatch, read) -> list:
    """What ``read()`` names at the moment each signal of a stop is delivered: the OS boundary a chain breaks at."""
    from ouroboros import platform_layer

    seen, real = [], platform_layer.deliver_signal

    def deliver(target, *, graceful):
        seen.append((int(target), read()))
        return real(target, graceful=graceful)

    monkeypatch.setattr(platform_layer, "deliver_signal", deliver)
    return seen


def _dead_worker(tmp_path, monkeypatch, scripts, task="exec-task"):
    """A real local executor service whose worker died: no in-memory owner, its JSON and ledger records on disk."""
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor

    ctx, record, registry, _log, leader, server = _executor(tmp_path, monkeypatch, scripts, task=task)
    with executor._STATE_LOCK:
        registry.pop(record.service_id)
    return ctx, record, leader, server, owned_shutdown.installation_root(ctx.drive_root)


@pytest.mark.parametrize("consumer", ["ordinary stop", "quiesce", "leftover record"])
def test_no_stop_signals_before_what_only_a_chain_reached_is_durable(tmp_path, monkeypatch, scripts, consumer):
    """Write-ahead, seen at the signal itself: every stop has C in the launch's custody before it signals anything that
    could break L's chain, so what it observed cannot go with it. With the roots readable the same pass ends L and C,
    and nothing stays owed."""
    from ouroboros import owned_shutdown, process_custody
    from ouroboros.tools import services

    data = tmp_path / "data"
    data.mkdir()
    if consumer in ("reaper", "quiesce"):
        proc, (port, pid, birth) = _ledgered_service(tmp_path, scripts, "a", scope="task", owner_task="t", drive=data,
                                                     options=HIDDEN)
        leader, cleanup = proc.pid, proc
    elif consumer == "local executor stop":
        ctx, record, _registry_map, _log, leader, (port, pid, birth) = _executor(tmp_path, monkeypatch, scripts)
        cleanup = record.local_proc
    elif consumer in ("dead worker's records", "leftover record"):
        ctx, record, leader, (port, pid, birth), root = _dead_worker(tmp_path, monkeypatch, scripts)
        cleanup = record.local_proc
    else:
        ctx, record, _registry_map, _log, leader, (port, pid, birth) = _host(tmp_path, monkeypatch, scripts)
        cleanup = record.proc
    seen = _signals_see_custody(monkeypatch, lambda: _owed(data, leader))
    try:
        if consumer == "reaper":
            monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
            process_custody.reap_orphaned_processes(data, running_task_ids=set())
        elif consumer == "quiesce":
            assert process_custody.quiesce_custodied_services(data) == (True, [])
        elif consumer == "exit owner":
            services.kill_all_services(data, wait=True, durable=False)
        elif consumer == "leftover record":  # the control stop settles a durable executor record no process holds
            stopped = services.kill_all_services(data, wait=True)
            assert [item["state"] for item in stopped] == ["stopped"], stopped
            assert not record.durable_record_path.exists()
        elif consumer == "task end":
            [payload] = services.stop_task_services(ctx)
            assert payload["lifecycle"] == "stopped", payload
        elif consumer == "exit record step":
            with services._LOCK:
                services._SERVICES.clear()  # that worker died: no in-memory owner in the exiting server
            root = owned_shutdown.installation_root(data)
            targets = [entry for entry in owned_shutdown.owned_records(root) if owned_shutdown._is_stop_target(entry)]
            assert owned_shutdown._stop_records(root, targets, time.monotonic() + _PATIENCE) == []
        elif consumer == "dead worker's records":
            targets = [entry for entry in owned_shutdown.owned_records(root) if owned_shutdown._is_stop_target(entry)]
            assert owned_shutdown._stop_records(root, targets, time.monotonic() + _PATIENCE) == []
        else:
            stopped = _payload(services._stop_service(ctx, name="a"))
            assert stopped["process_cleanup"]["exit_confirmed"] is True, stopped

        assert seen, "the stop signalled something"
        target, (ledger, copy) = seen[0]
        assert pid in ledger and pid in copy, f"the first signal (to {target}) broke a chain nothing durable named"
        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert all(pid not in names for names in _owed(data, leader)), "ended: it owes nothing"
        if consumer in ("reaper", "quiesce"):  # a sweep's confirmed launch leaves with what its own pass wrote
            assert not [row for row in process_custody._read_ledger(data) if row["pid"] == leader]
        if consumer != "leftover record":  # that stop settles the executor record alone; the row's own step comes later
            assert not [entry for entry in owned_shutdown.owned_records(data)
                        if (entry.get("ledger_entry") or {}).get("pid") == leader], "a confirmed stop retires its record"
    finally:
        _settle(cleanup, identities=[(pid, birth)])


def test_a_stop_killed_right_after_its_signal_leaves_the_child_named_for_the_next_start(tmp_path, monkeypatch, scripts):
    """The launcher's force-exit lands on the stop right after its signal ended L: nothing after that signal runs, no
    answer, no publication, no memory left. C, which only L's chain reached, is already named: the readers hold the
    launch and the next generation's sweep ends it."""
    from ouroboros import platform_layer, process_custody
    from ouroboros.presence_continuation import _custody_blockers
    from ouroboros.tools import services

    class Died(BaseException):
        """Not an exception any stop handles: the process is gone."""

    data = tmp_path / "data"
    ctx, record, registry, _log, leader, (port, pid, birth) = _host(tmp_path, monkeypatch, scripts)
    real = platform_layer.deliver_signal

    def deliver(target, *, graceful):
        if int(target) != leader:
            return platform_layer.SIGNAL_DENIED  # L is the one that dies first here, whatever order the pass took
        real(target, graceful=False)
        raise Died

    monkeypatch.setattr(platform_layer, "deliver_signal", deliver)
    try:
        with pytest.raises(Died):
            services._stop_service(ctx, name="a")
        monkeypatch.setattr(platform_layer, "deliver_signal", real)
        record.proc.wait(timeout=_PATIENCE)
        registry.clear()  # that process is gone, and every memory of it

        assert _answers(port) and _same_process_alive(pid, birth) and all(pid in names for names in _owed(data, leader))
        assert _held(_custody_blockers(data, ctx.task_id), leader) == "live"
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        process_custody.reap_orphaned_processes(data, running_task_ids=set())

        assert not _answers(port) and not _same_process_alive(pid, birth), "the next start ends what the dead stop observed"
        assert _held(_custody_blockers(data, ctx.task_id), leader) == "none"
    finally:
        monkeypatch.setattr(platform_layer, "deliver_signal", real)
        _settle(record.proc, identities=[(pid, birth)])


@pytest.mark.parametrize("reader", ["presence", "continuation", "cold sleep"])
def test_a_dead_workers_local_service_has_one_decision_and_every_reader_sees_what_it_observed(
        tmp_path, monkeypatch, scripts, reader):
    """Real ``_stop_records`` over a dead worker's executor record and ledger row of one launch. One census decides for
    both: C goes in the launch's custody, not on one alias, so neither record can retire around it, the required
    readers hold the launch while L is gone, and the executor record itself carries no debt only it names. The next
    start ends C and retires both."""
    from ouroboros import owned_shutdown, process_scope

    data = tmp_path / "data"
    data.mkdir()
    read, task = _reader(data, monkeypatch, reader)
    ctx, record, leader, (port, pid, birth), root = _dead_worker(tmp_path, monkeypatch, scripts, task=task)
    durable = record.durable_record_path
    roots = _roots_unknown(monkeypatch)
    calls, real_census = [], process_scope.census
    monkeypatch.setattr(process_scope, "census", lambda *a, **k: calls.append(threading.current_thread().name)
                        or real_census(*a, **k))
    try:
        targets = [entry for entry in owned_shutdown.owned_records(root) if owned_shutdown._is_stop_target(entry)]
        assert len(targets) == 2, targets
        remaining = owned_shutdown._stop_records(root, targets, time.monotonic() + _PATIENCE)
        monkeypatch.setattr(process_scope, "census", real_census)
        record.local_proc.wait(timeout=_PATIENCE)  # L's real parent died: whoever adopted it reaps it, no corpse stays

        assert _answers(port) and _same_process_alive(pid, birth), "C lives, held"
        roots["unknown"] = False
        assert _held(read(), leader) == "live", "the readers hold the launch through what it still owes"
        assert _same_process_alive(pid, birth), "reading signals nothing"
        assert remaining == sorted(entry["record_id"] for entry in targets), "neither record retires around C"
        assert durable.exists() and all(pid in names for names in _owed(data, leader))
        assert "known_members" not in json.loads(durable.read_text(encoding="utf-8")), "no debt only this record names"
        assert set(calls) == {"owned-stop-services"}, f"one pass decides for both records: {calls}"

        finished = owned_shutdown.finish_unconfirmed_stops(root)

        assert finished["retried"] == 2 and finished["unconfirmed"] == 0, finished
        assert not _answers(port) and not _same_process_alive(pid, birth) and not durable.exists()
        assert [entry["record_id"] for entry in owned_shutdown.owned_records(root)
                if owned_shutdown._is_stop_target(entry)] == []
        assert _held(read(), leader) == "none"
    finally:
        _settle(record.local_proc, identities=[(pid, birth)])


def _block_registration(monkeypatch, at: str, data) -> tuple[threading.Event, threading.Event]:
    """Pause the starter's ``record_process`` at one point: both durable stores written (before the custody assignment),
    the ledger appended but not the ownership set, or before the ledger append. Returns (reached, release)."""
    from ouroboros import owned_shutdown, process_custody

    reached, release = threading.Event(), threading.Event()

    def pause():
        if threading.current_thread().name == "starter":
            reached.set()
            assert release.wait(_PATIENCE)

    if at == "after both durable stores":
        real = process_custody.record_process

        def record(*args, **kwargs):
            row = real(*args, **kwargs)
            pause()
            return row

        monkeypatch.setattr(process_custody, "record_process", record)
    elif at == "after the ledger append":
        real_set = owned_shutdown.record_ledgered_process

        def publish(*args, **kwargs):
            pause()
            return real_set(*args, **kwargs)

        monkeypatch.setattr(owned_shutdown, "record_ledgered_process", publish)
    else:
        real_append, ledger = process_custody.append_jsonl, process_custody.ledger_path(data)

        def append(path, *args, **kwargs):
            if pathlib.Path(path) == ledger:
                pause()
            return real_append(path, *args, **kwargs)

        monkeypatch.setattr(process_custody, "append_jsonl", append)
    return reached, release


@pytest.mark.parametrize("backend", list(BACKENDS))
@pytest.mark.parametrize("at", ["after both durable stores", "after the ledger append", "before the ledger append"])
def test_a_stop_during_registration_resolves_the_launch_durably_or_leaves_its_chain_alone(
        tmp_path, monkeypatch, scripts, at, backend):
    """The exit begins while ``start_service`` is still in ``record_process``: the memory record is published, the
    Popen has no ``_ouroboros_custody`` yet. Once a row exists the stop finds it by its launch: C is durable and a late
    registration cannot erase it from either store, so the next start ends it. With no row yet nothing could carry C,
    so nothing is signalled, L's chain stays, and a retry after registration ends it all. Either way the exit says
    unconfirmed."""
    from ouroboros import body_candidate, owned_shutdown
    from ouroboros import workspace_executor as executor
    from ouroboros.presence_continuation import _custody_blockers
    from ouroboros.tools import services

    data = tmp_path / "data"
    port_file, started = tmp_path / "a.port", {}
    cmd = _runner_cmd(scripts, port_file, "separate", *HIDDEN)
    if backend == "host service":
        registry = _registry(tmp_path, monkeypatch)
        ctx, memory = registry._ctx, services._SERVICES
        begin = functools.partial(registry.execute, "start_service", {"name": "a", "cmd": cmd, "readiness": {}})
    else:
        monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
        ctx, workspace = _executor_ctx(tmp_path)
        memory = executor._SERVICES
        begin = functools.partial(services._start_service, ctx, name="a", cmd=cmd, cwd=str(workspace), readiness={})
    roots = _roots_unknown(monkeypatch)
    reached, release = _block_registration(monkeypatch, at, data)
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    starter = threading.Thread(target=lambda: started.update(text=begin()), name="starter")
    starter.start()
    assert reached.wait(_PATIENCE), "the start never reached registration"
    port, pid, birth = _server(port_file)
    [record] = memory.values()
    proc = record.proc if backend == "host service" else record.local_proc
    leader = proc.pid
    assert not hasattr(proc, "_ouroboros_custody")
    try:
        outcome = owned_shutdown.stop_owned_work(data)

        assert (outcome["state"] == "unconfirmed"
                and record.service_id in outcome["memory_unconfirmed"]["services"]["services"]), outcome
        assert _answers(port) and record.service_id in memory
        if at == "before the ledger append":
            assert proc.poll() is None, "nothing durable could carry C, so its chain is left intact"
        else:
            _leader_gone(leader)
            assert pid in _owed(data, leader)[0], "the stop found the launch's row by its marker"
        release.set()
        starter.join(_PATIENCE)
        assert not starter.is_alive() and hasattr(proc, "_ouroboros_custody")
        roots["unknown"] = False
        if at == "before the ledger append":
            retried = _payload(services._stop_service(ctx, name="a"))
            assert retried["process_cleanup"]["exit_confirmed"] is True, retried
        else:
            assert all(pid in names for names in _owed(data, leader)), "registration landed after C and did not erase it"
            assert _held(_custody_blockers(data, record.task_id), leader) == "live"
            assert body_candidate.live_lineage_processes(data, record.task_id), "the body candidate reads the set's copy"
            memory.clear()  # that server exited; its in-memory owner is gone
            monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
            finished = owned_shutdown.finish_unconfirmed_stops(data)
            assert finished["unconfirmed"] == 0, finished
        assert not _answers(port) and not _same_process_alive(pid, birth) and proc.poll() is not None
    finally:
        release.set()
        starter.join(_PATIENCE)
        _settle(proc, identities=[(pid, birth)])


def test_a_stop_during_registration_with_no_chain_to_keep_ends_the_launch_and_the_start_still_answers(
        tmp_path, monkeypatch, scripts):
    """Nothing only a chain reached (the server stays in the runner's group): the stop ends the launch while its
    registration has not reached the ledger, with nothing to carry and no false failure. The start returns its own
    terminal status, not an error from the record that left."""
    from ouroboros.tools import services

    data = tmp_path / "data"
    registry = _registry(tmp_path, monkeypatch)
    reached, release = _block_registration(monkeypatch, "before the ledger append", data)
    port_file, started = tmp_path / "a.port", {}
    starter = threading.Thread(target=lambda: started.update(text=registry.execute("start_service", {
        "name": "a", "cmd": _runner_cmd(scripts, port_file, "same"), "readiness": {}})), name="starter")
    starter.start()
    assert reached.wait(_PATIENCE)
    port, pid, birth = _server(port_file)
    [record] = services._SERVICES.values()
    try:
        stopped = _payload(services._stop_service(registry._ctx, name="a"))

        assert stopped["process_cleanup"]["exit_confirmed"] is True and "stop_failed" not in stopped, stopped
        assert not _answers(port) and not _same_process_alive(pid, birth) and record.service_id not in services._SERVICES
        release.set()
        starter.join(_PATIENCE)
        assert json.loads(started["text"])["state"] == "exited", started
    finally:
        release.set()
        starter.join(_PATIENCE)
        _settle(record.proc, identities=[(pid, birth)])


def test_a_stop_discharges_what_another_stop_recorded_once_it_is_provably_gone(tmp_path, monkeypatch):
    """Another stop recorded C, and C has since died. A stop that never judged C reads its identity once: gone, so it
    owes nothing and the launch retires now, where a live C would keep it (the parallel-stop tests above)."""
    from ouroboros import process_census, process_custody
    from ouroboros.tools import services
    from tests._launch_stop_shared import _CHILD, _MARK_A, _OLD_BIRTH, _PID, _controlled, _launch_row

    rows = {_PID: process_census._Row(_PID, 2, _PID, _OLD_BIRTH, 123.0, "alive", {_MARK_A})}
    world = _controlled(tmp_path, monkeypatch, rows)
    row = _launch_row("t-a", purpose="service:a")
    world.spawned(row)
    assert process_custody.note_owed(tmp_path, row, [{"pid": _CHILD, "birth": _OLD_BIRTH}])

    class Held:
        pid, returncode, _ouroboros_custody = _PID, None, row

        def poll(self):
            if _PID not in rows and self.returncode is None:
                self.returncode = -9
            return self.returncode

        def wait(self, timeout=None):
            return self.poll()

    log = tmp_path / "a.log"
    log.write_text("live log\n", encoding="utf-8")
    record = services.ServiceRecord(name="a", service_id="t-a:a", task_id="t-a", cmd=["runner"], cwd=str(tmp_path),
                                    log_path=log, proc=Held(), pgid=_PID, leader_birth=_OLD_BIRTH,
                                    scope_marker=_MARK_A, drive_root=str(tmp_path))
    monkeypatch.setattr(services, "_SERVICES", {record.service_id: record})

    [payload] = services.kill_all_services(tmp_path, wait=True, durable=False)

    assert "stop_failed" not in payload and record.service_id not in services._SERVICES, payload
    assert _owed(tmp_path, _PID) == ([], [])


def test_one_launchs_failed_publication_costs_a_shutdown_only_that_launch(tmp_path, monkeypatch, scripts):
    """The batched shutdown publishes before it signals, so an unexpected failure there must stay with its launch. A's
    publication raises: A keeps its chain and its record, unconfirmed; sibling B, in the same pass, still ends and
    retires. A retry after the fault ends A."""
    from ouroboros import process_custody
    from ouroboros.tools import services

    data = tmp_path / "data"
    registry = _registry(tmp_path, monkeypatch)
    pid_a = _start(registry, scripts, tmp_path / "a.port", "a", options=HIDDEN)["pid"]
    _start(registry, scripts, tmp_path / "b.port", "b", mode="same")
    port_a, server_a, birth_a = _server(tmp_path / "a.port")
    port_b, server_b, birth_b = _server(tmp_path / "b.port")
    record_a = services._SERVICES[services._service_key(registry._ctx, "a")]
    record_b = services._SERVICES[services._service_key(registry._ctx, "b")]
    real, faults = process_custody._reconcile_owed, []

    def faulty(drive_root, entry, *args, **kwargs):
        if int(entry.get("pid") or 0) == pid_a:
            faults.append(args)
            raise OSError("the disk answered oddly")
        return real(drive_root, entry, *args, **kwargs)

    monkeypatch.setattr(process_custody, "_reconcile_owed", faulty)
    try:
        by_id = {item["service_id"]: item for item in services.kill_all_services(data, wait=True, durable=False)}

        assert faults, "A's publication was attempted"
        assert by_id[record_a.service_id]["stop_failed"] is True and by_id[record_a.service_id]["process_cleanup"]["custody"] == "unavailable"
        assert record_a.proc.poll() is None and _answers(port_a), "nothing durable could carry its child: its chain stays"
        assert "stop_failed" not in by_id[record_b.service_id] and record_b.service_id not in services._SERVICES
        assert not _answers(port_b) and not _same_process_alive(server_b, birth_b)

        monkeypatch.setattr(process_custody, "_reconcile_owed", real)
        retried = _payload(services._stop_service(registry._ctx, name="a"))

        assert retried["process_cleanup"]["exit_confirmed"] is True and not _answers(port_a), retried
    finally:
        monkeypatch.setattr(process_custody, "_reconcile_owed", real)
        _settle(record_a.proc, record_b.proc, identities=[(server_a, birth_a), (server_b, birth_b)])


def test_overlapping_ordinary_stops_finalize_a_service_once(tmp_path, monkeypatch, scripts):
    """Two ``stop_service`` calls whose passes overlap both measure the end; the one that retires the record finalizes
    its log, and the other finds it gone. No lock held across either pass: neither waits on the other's grace."""
    from ouroboros import process_scope
    from ouroboros.tools import services

    ctx, record, registry, log, _leader, (port, pid, birth) = _host(tmp_path, monkeypatch, scripts)
    barrier, real_census, results = threading.Barrier(2), process_scope.census, []

    def census(*args, **kwargs):
        if threading.current_thread().name.startswith("stopper"):
            try:
                barrier.wait(_PATIENCE)  # each pass reached its census while the other is in its own
            except threading.BrokenBarrierError:
                pass
        return real_census(*args, **kwargs)

    monkeypatch.setattr(process_scope, "census", census)
    stoppers = [threading.Thread(target=lambda: results.append(services._stop_service(ctx, name="a")),
                                 name=f"stopper-{index}") for index in range(2)]
    try:
        for stopper in stoppers:
            stopper.start()
        for stopper in stoppers:
            stopper.join(_PATIENCE)
        assert not any(stopper.is_alive() for stopper in stoppers)

        finished = [json.loads(text) for text in results if text.startswith("{")]
        assert len(finished) == 1 and finished[0]["log_finalization"]["deleted_live_log"] is True, results
        assert [text for text in results if not text.startswith("{")] == ["⚠️ SERVICE_NOT_FOUND: a"], results
        assert record.service_id not in registry and not log.exists() and not _answers(port)
    finally:
        barrier.abort()
        _settle(record.proc, identities=[(pid, birth)])


def test_a_stop_writing_ahead_after_the_other_stop_of_its_launch_finished_still_leaves_no_record(
        tmp_path, monkeypatch, scripts):
    """The exit's two stops of one held launch, both clean, in the order that re-creates the ownership record: the
    in-memory owner's census sees L -> C and pauses; the record step ends both, confirms and retires its record; the
    owner then writes C ahead (a new record), finds C gone and confirms. Whichever publishes last retires what it
    left, so the exit reports nothing it does not owe."""
    from ouroboros import owned_shutdown, process_custody, process_scope
    from ouroboros.tools import services

    data = tmp_path / "data"
    _ctx, record, registry, log, leader, (port, pid, birth) = _host(tmp_path, monkeypatch, scripts)
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: _PATIENCE)
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    owner_saw, step_done, owner_threads = threading.Event(), threading.Event(), set()
    real_kill_all, real_census, real_ledgered = services.kill_all_services, process_scope.census, owned_shutdown._stop_ledgered
    real_ahead = process_custody.write_ahead

    def memory_owner(*args, **kwargs):
        owner_threads.add(threading.get_ident())
        return real_kill_all(*args, **kwargs)

    def census(*args, **kwargs):
        if threading.get_ident() not in owner_threads:
            assert owner_saw.wait(_PATIENCE), "the owner never read the table"
        return real_census(*args, **kwargs)

    def ahead(*args, **kwargs):
        if threading.get_ident() in owner_threads:
            owner_saw.set()
            assert step_done.wait(_PATIENCE), "the record step never finished"
        return real_ahead(*args, **kwargs)

    def ledgered(*args, **kwargs):
        try:
            return real_ledgered(*args, **kwargs)
        finally:
            step_done.set()

    monkeypatch.setattr(services, "kill_all_services", memory_owner)
    monkeypatch.setattr(process_scope, "census", census)
    monkeypatch.setattr(process_custody, "write_ahead", ahead)
    monkeypatch.setattr(owned_shutdown, "_stop_ledgered", ledgered)
    try:
        outcome = owned_shutdown.stop_owned_work(data)

        assert outcome["state"] == "completed" and outcome["unconfirmed"] == [] and not outcome["memory_unconfirmed"], outcome
        assert owned_shutdown.owned_records(data, strict=True) == []
        assert record.service_id not in registry and not log.exists(), "its log was finalized by the owner that retired it"
        assert not _answers(port) and not _same_process_alive(pid, birth) and record.proc.poll() is not None
    finally:
        step_done.set()
        owner_saw.set()
        _settle(record.proc, identities=[(pid, birth)])


def test_a_stop_that_never_observed_a_late_child_is_not_ordered_against_the_stop_that_did(tmp_path, monkeypatch, scripts):
    """What write-ahead does and does not order. The shutdown A reads the table while L has no child and pauses before
    its signal. L then starts C (its own session, no marker), and an ordinary stop B observes L -> C and pauses before
    writing it. A signals L: its rescan cannot find C and nothing names C yet, so A confirms what it saw and retires
    the record, as a lone stop would around a process started after its last table read. B's observation still lands:
    custody names C from then on, whichever record is gone, every reader holds the launch, and a later stop ends C.
    A's discharge rule is not involved: C was never in A's input and was not published when A reconciled."""
    from ouroboros import body_candidate, platform_layer, process_custody
    from ouroboros.presence_continuation import _custody_blockers
    from ouroboros.tools import services

    data = tmp_path / "data"
    registry, port_file, trigger = _registry(tmp_path, monkeypatch), tmp_path / "a.port", tmp_path / "start-the-child"
    started = json.loads(registry.execute("start_service", {
        "name": "a", "cmd": _runner_cmd(scripts, port_file, "separate", *HIDDEN, f"late={trigger}"), "readiness": {}}))
    record = services._SERVICES[services._service_key(registry._ctx, "a")]
    leader, ctx = started["pid"], registry._ctx
    roots = _roots_unknown(monkeypatch)
    a_at_signal, go_signal, b_at_ahead, a_done = (threading.Event() for _ in range(4))
    real_deliver, real_ahead, results = platform_layer.deliver_signal, process_custody.write_ahead, {}

    def deliver(target, *, graceful):
        if threading.current_thread().name == "exit-owner":
            a_at_signal.set()
            assert go_signal.wait(_PATIENCE)
        return real_deliver(target, graceful=graceful)

    def ahead(*args, **kwargs):
        if threading.current_thread().name == "stopper":
            b_at_ahead.set()
            assert a_done.wait(_PATIENCE)
        return real_ahead(*args, **kwargs)

    def exit_owner():
        try:
            results["a"] = services.kill_all_services(data, wait=True, durable=False)
        finally:
            a_done.set()

    monkeypatch.setattr(platform_layer, "deliver_signal", deliver)
    monkeypatch.setattr(process_custody, "write_ahead", ahead)
    threads = [threading.Thread(target=exit_owner, name="exit-owner"),
               threading.Thread(target=lambda: results.update(b=services._stop_service(ctx, name="a")), name="stopper")]
    threads[0].start()
    identities = []
    try:
        assert a_at_signal.wait(_PATIENCE), "A read the table and reached its signal with no child to see"
        trigger.write_text("now", encoding="utf-8")
        port, pid, birth = _server(port_file)
        identities = [(pid, birth)]
        threads[1].start()
        assert b_at_ahead.wait(_PATIENCE), "B observed L -> C"
        assert _owed(data, leader) == ([], []), "observed, not yet published"
        go_signal.set()
        for thread in threads:
            thread.join(_PATIENCE)
        assert not any(thread.is_alive() for thread in threads)

        [payload] = results["a"]
        assert payload["process_cleanup"]["exit_confirmed"] is True and "stop_failed" not in payload, payload
        assert record.service_id not in services._SERVICES, "A retired the record on what it observed: the disclosed limit"
        assert results["b"].startswith("⚠️ SERVICE_STOP_ERROR"), results["b"]
        assert _answers(port) and _same_process_alive(pid, birth), "C lives, held"

        assert all(pid in names for names in _owed(data, leader)), "B's observation names C, with no record left to carry it"
        roots["unknown"] = False
        assert _held(_custody_blockers(data, record.task_id), leader) == "live"
        assert body_candidate.live_lineage_processes(data, record.task_id), "the set's copy names it too"
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        process_custody.reap_orphaned_processes(data, running_task_ids=set())

        assert not _answers(port) and not _same_process_alive(pid, birth), "a later stop ends what custody still names"
        assert _held(_custody_blockers(data, record.task_id), leader) == "none"
    finally:
        trigger.write_text("now", encoding="utf-8")
        a_at_signal.set()
        go_signal.set()
        a_done.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(_PATIENCE)
        _settle(record.proc, identities=identities)


def test_emergency_stop_during_registration_still_has_its_immediate_handle(tmp_path, monkeypatch, scripts):
    """Panic reads the published handle while registration is paused: one native kill, no table read, no custody read."""
    from ouroboros import process_custody, process_scope
    from ouroboros.tools import services

    data = tmp_path / "data"
    registry = _registry(tmp_path, monkeypatch)
    reached, release = _block_registration(monkeypatch, "before the ledger append", data)
    monkeypatch.setattr(services, "_panic_requested", False)
    monkeypatch.setattr("ouroboros.workspace_executor._panic_requested", False)
    port_file = tmp_path / "a.port"
    starter = threading.Thread(target=lambda: registry.execute("start_service", {
        "name": "a", "cmd": _runner_cmd(scripts, port_file, "same"), "readiness": {}}), name="starter")
    starter.start()
    assert reached.wait(_PATIENCE)
    _port, pid, birth = _server(port_file)
    [record] = services._SERVICES.values()
    monkeypatch.setattr(process_scope, "census", lambda *a, **k: pytest.fail("Panic reads no process table"))
    monkeypatch.setattr(process_custody, "_read_ledger_records", lambda *a, **k: pytest.fail("Panic reads no custody"))
    try:
        [requested] = services.kill_all_services(data, request_only=True)
        assert requested["service_id"] == record.service_id and requested.get("requested") is True, requested
        assert record.proc.wait(timeout=_PATIENCE) is not None
    finally:
        monkeypatch.undo()
        release.set()
        starter.join(_PATIENCE)
        _settle(record.proc, identities=[(pid, birth)])


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_the_spawn_callback_writes_each_registry_under_its_lock(tmp_path, monkeypatch, scripts, backend):
    """Readers iterate these dicts under their locks; an unlocked insert from an admitted spawn raises in the middle of
    a shutdown's stop."""
    from ouroboros import workspace_executor as executor
    from ouroboros.tools import services

    module, held = (services, lambda: services._LOCK.locked()) if backend == "host service" else (
        executor, lambda: executor._STATE_LOCK._is_owned())
    unlocked = []

    class Spy(dict):
        def __setitem__(self, key, value):
            if not held():
                unlocked.append(key)
            super().__setitem__(key, value)

    monkeypatch.setattr(module, "_SERVICES", Spy())
    ctx, record, _registry_map, _log, _leader, (port, pid, birth) = BACKENDS[backend](tmp_path, monkeypatch, scripts)
    try:
        assert unlocked == [], "the registry was written without its lock"
    finally:
        _settle(record.proc if backend == "host service" else record.local_proc, identities=[(pid, birth)])
