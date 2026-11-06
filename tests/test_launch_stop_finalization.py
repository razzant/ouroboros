"""One finalization per launch, by the stop that won it, with the service's name held until it is done.

Overlapping stops of one service each measure the same end. Only one of them finalizes: the first to claim that exact
record object. The name stays taken while it does, because the log path and the outputs are reused by a new start of
the same task and name; once a newer service holds the name, a late stop of the old object touches nothing of it. No
lock is held across a process wait. Real services, real stops; a gate at the finalization a late stop would reach.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from tests._launch_stop_shared import (
    _PATIENCE,
    _answers,
    _executor_ctx,
    _executor_start,
    _payload,
    _runner_cmd,
    _same_process_alive,
    _server,
    _settle,
    write_scripts,
)
from tests.test_launch_stop_contract import BACKENDS, _host

pytestmark = [
              pytest.mark.skipif(os.name == "nt", reason="the measured stop is POSIX; Windows keeps its native kill")]


@pytest.fixture
def scripts(tmp_path):
    return write_scripts(tmp_path)


def _proc(record):
    return record.proc if hasattr(record, "proc") else record.local_proc


def _pause_finalization_of(monkeypatch, thread_name: str):
    """That thread stops right before it would finalize a service log. Returns (reached, go)."""
    from ouroboros.tools import services

    reached, go, real = threading.Event(), threading.Event(), services._finalize_service_log_for_drive

    def finalize(*args, **kwargs):
        if threading.current_thread().name == thread_name:
            reached.set()
            assert go.wait(_PATIENCE), "never released"
        return real(*args, **kwargs)

    monkeypatch.setattr(services, "_finalize_service_log_for_drive", finalize)
    return reached, go


def _restart(ctx, tmp_path, scripts, backend, port_file):
    from ouroboros.tools import services

    arguments = {"name": "a", "cmd": _runner_cmd(scripts, port_file, "same"),
                 "readiness": {"log_contains": "READY", "timeout_sec": 20}}
    if backend != "host service":
        arguments["cwd"] = str(tmp_path / "project")
    return services._start_service(ctx, **arguments)


def _new_log_has_output(log) -> bool:
    return log.exists() and "READY" in log.read_text(encoding="utf-8")


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_a_late_stop_of_the_old_object_leaves_the_newer_service_alone(tmp_path, monkeypatch, scripts, backend):
    """Stops A and B both confirm the end of the same old record. B wins: it finalizes and frees the name. A new service
    starts under that task and name, reusing the log path. A resumes: it must neither finalize the path nor remove the
    new record. The newer service stays registered with its live log, and its own stop later works."""
    from ouroboros import process_scope
    from ouroboros.tools import services

    ctx, record, registry, log, _leader, (_port, pid, birth) = BACKENDS[backend](tmp_path, monkeypatch, scripts)
    real_end, a_ended, go_a = process_scope.end_held_processes, threading.Event(), threading.Event()
    finalized_by = []
    real_finalize = services._finalize_service_log_for_drive

    def end(*args, **kwargs):
        answers = real_end(*args, **kwargs)
        if threading.current_thread().name == "late-a":  # A measured the end; everything after is its decision
            a_ended.set()
            assert go_a.wait(_PATIENCE)
        return answers

    def finalize(*args, **kwargs):
        finalized_by.append(threading.current_thread().name)
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(process_scope, "end_held_processes", end)
    monkeypatch.setattr(services, "_finalize_service_log_for_drive", finalize)
    results: dict = {}
    me = threading.current_thread().name
    late_a = threading.Thread(target=lambda: results.update(a=services._stop_service(ctx, name="a")), name="late-a")
    late_a.start()
    new_identity = None
    try:
        assert a_ended.wait(_PATIENCE), "A never measured the end"
        won = _payload(services._stop_service(ctx, name="a"))
        assert won["process_cleanup"]["exit_confirmed"] is True and "stop_failed" not in won, won
        assert finalized_by == [me], finalized_by
        assert record.service_id not in registry and not log.exists()

        started = json.loads(_restart(ctx, tmp_path, scripts, backend, tmp_path / "new.port"))
        newer = registry[record.service_id]
        assert newer is not record and started["state"] == "running", started
        new_identity = _server(tmp_path / "new.port")
        assert _new_log_has_output(log), "the new service writes the same log path"

        go_a.set()
        late_a.join(_PATIENCE)
        assert not late_a.is_alive()

        assert registry.get(record.service_id) is newer, "the late stop removed the newer service's record"
        assert _new_log_has_output(log), "the late stop finalized the newer service's live log"
        assert finalized_by == [me], f"finalized twice: {finalized_by}"
        assert _proc(newer).poll() is None and _answers(new_identity[0])
        assert results["a"] == "⚠️ SERVICE_NOT_FOUND: a", results["a"]

        stopped = _payload(services._stop_service(ctx, name="a"))
        assert stopped["process_cleanup"]["exit_confirmed"] is True, stopped
        assert stopped["log_finalization"]["deleted_live_log"] is True and record.service_id not in registry
        assert not _answers(new_identity[0])
    finally:
        go_a.set()
        late_a.join(_PATIENCE)
        for item in list(registry.values()):
            _settle(_proc(item))
        _settle(_proc(record), identities=[(pid, birth), *([new_identity[1:]] if new_identity else [])])


def test_a_late_host_stop_never_turns_to_an_executor_service_that_took_its_name(tmp_path, monkeypatch, scripts):
    """The same A/B/late-A order across backends. A selects host launch L and pauses after its process pass; B ends L,
    finalizes it and frees the name; the same task starts that name again on the local executor, as the start path
    allows. A resumes and finds L retired: the call stays bound to the launch it selected, so it answers that nothing is
    left to stop and the replacement receives no stop. A stop that selected no host record still stops the executor's."""
    from ouroboros import process_scope
    from ouroboros import workspace_executor as executor
    from ouroboros.tools import services

    ctx, record, registry, log, _leader, (_port, pid, birth) = _host(tmp_path, monkeypatch, scripts, task="swap-task")
    real_end, a_ended, go_a = process_scope.end_held_processes, threading.Event(), threading.Event()
    executor_stops, real_executor_stop = [], services.executor_stop_service

    def end(*args, **kwargs):
        answers = real_end(*args, **kwargs)
        if threading.current_thread().name == "late-a":  # A measured the end; everything after is its decision
            a_ended.set()
            assert go_a.wait(_PATIENCE)
        return answers

    def executor_stop(*args, **kwargs):
        executor_stops.append(threading.current_thread().name)
        return real_executor_stop(*args, **kwargs)

    monkeypatch.setattr(process_scope, "end_held_processes", end)
    monkeypatch.setattr(services, "executor_stop_service", executor_stop)
    results: dict = {}
    late_a = threading.Thread(target=lambda: results.update(a=services._stop_service(ctx, name="a")), name="late-a")
    late_a.start()
    replacement = new_identity = None
    try:
        assert a_ended.wait(_PATIENCE), "A never measured the end"
        won = _payload(services._stop_service(ctx, name="a"))
        assert won["process_cleanup"]["exit_confirmed"] is True and "stop_failed" not in won, won
        assert record.service_id not in registry and not log.exists()

        executor_ctx, workspace = _executor_ctx(tmp_path, task_id="swap-task")
        started = _executor_start(executor_ctx, workspace, scripts, tmp_path / "new.port", "a", mode="same")
        replacement = executor._SERVICES[record.service_id]  # the same task and name, now the executor's
        new_identity = _server(tmp_path / "new.port")
        assert started["state"] == "running" and _answers(new_identity[0]), started

        go_a.set()
        late_a.join(_PATIENCE)
        assert not late_a.is_alive()

        assert executor._SERVICES.get(record.service_id) is replacement, "the late host stop removed the replacement"
        assert replacement.local_proc.poll() is None and _answers(new_identity[0]), "the late host stop ended it"
        assert "late-a" not in executor_stops, f"the late host stop turned to the executor: {executor_stops}"
        assert results["a"] == "⚠️ SERVICE_NOT_FOUND: a", results["a"]

        stopped = _payload(services._stop_service(executor_ctx, name="a"))  # no host record: the executor's own stop
        assert stopped["process_cleanup"]["exit_confirmed"] is True, stopped
        assert record.service_id not in executor._SERVICES and not _answers(new_identity[0])
    finally:
        go_a.set()
        late_a.join(_PATIENCE)
        if replacement is not None:
            _settle(replacement.local_proc)
        _settle(record.proc, identities=[(pid, birth), *([new_identity[1:]] if new_identity else [])])


@pytest.mark.parametrize("consumer", ["stop_service", "shutdown"])
@pytest.mark.parametrize("backend", list(BACKENDS))
def test_the_name_stays_held_until_the_winner_finished_finalizing(tmp_path, monkeypatch, scripts, backend, consumer):
    """A start of the same name while the winner is finalizing is refused: its log must not be opened and then deleted
    under it, nor a new record be overwritten. Once the winner finished the name is free and the start succeeds."""
    from ouroboros.tools import services

    data = tmp_path / "data"
    ctx, record, registry, log, _leader, (_port, pid, birth) = BACKENDS[backend](tmp_path, monkeypatch, scripts)
    reached, go = _pause_finalization_of(monkeypatch, "winner")
    results: dict = {}

    def stop():
        if consumer == "stop_service":
            results["winner"] = services._stop_service(ctx, name="a")
        else:
            results["winner"] = services.kill_all_services(data, wait=True, durable=False)

    winner = threading.Thread(target=stop, name="winner")
    winner.start()
    spawned, new_identity = [], None
    try:
        assert reached.wait(_PATIENCE), "the winner never reached its finalization"
        assert record.service_id in registry, "the winner holds the name through its finalization"
        before = log.read_bytes()
        real_spawn = "ouroboros.process_custody.spawn_supervised"
        with monkeypatch.context() as patch:
            patch.setattr(real_spawn, lambda *a, **k: spawned.append(a) or pytest.fail("a start beside a finalization"))
            refused = _restart(ctx, tmp_path, scripts, backend, tmp_path / "early.port")
        assert not spawned and refused.startswith("⚠️ SERVICE_"), refused
        assert registry.get(record.service_id) is record and log.read_bytes() == before

        go.set()
        winner.join(_PATIENCE)
        assert not winner.is_alive() and record.service_id not in registry and not log.exists(), results

        started = json.loads(_restart(ctx, tmp_path, scripts, backend, tmp_path / "new.port"))
        assert started["state"] == "running" and registry[record.service_id] is not record, started
        new_identity = _server(tmp_path / "new.port")
        assert _new_log_has_output(log) and _answers(new_identity[0])
    finally:
        go.set()
        winner.join(_PATIENCE)
        for item in list(registry.values()):
            _settle(_proc(item))
        _settle(_proc(record), identities=[(pid, birth), *([new_identity[1:]] if new_identity else [])])


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_a_second_stop_during_the_winners_finalization_finds_nothing_to_do(tmp_path, monkeypatch, scripts, backend):
    """It neither finalizes again nor frees the name under the winner."""
    from ouroboros.tools import services

    ctx, record, registry, log, _leader, (_port, pid, birth) = BACKENDS[backend](tmp_path, monkeypatch, scripts)
    reached, go = _pause_finalization_of(monkeypatch, "winner")
    results: dict = {}
    winner = threading.Thread(target=lambda: results.update(winner=services._stop_service(ctx, name="a")), name="winner")
    winner.start()
    try:
        assert reached.wait(_PATIENCE)

        late = services._stop_service(ctx, name="a")

        assert late == "⚠️ SERVICE_NOT_FOUND: a", late
        assert registry.get(record.service_id) is record and log.exists(), "still the winner's"
        go.set()
        winner.join(_PATIENCE)
        assert _payload(results["winner"])["log_finalization"]["deleted_live_log"] is True
        assert record.service_id not in registry and not log.exists()
        assert not _same_process_alive(pid, birth)
    finally:
        go.set()
        winner.join(_PATIENCE)
        _settle(_proc(record), identities=[(pid, birth)])


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_the_name_is_released_when_finalization_fails(tmp_path, monkeypatch, scripts, backend):
    """A winner whose finalization raises still frees the name: a held name must not outlive its stop."""
    from ouroboros.tools import services

    ctx, record, registry, _log, _leader, (_port, pid, birth) = BACKENDS[backend](tmp_path, monkeypatch, scripts)

    def boom(*_args, **_kwargs):
        raise OSError("the disk answered oddly")

    monkeypatch.setattr(services, "_finalize_service_log_for_drive", boom)
    try:
        with pytest.raises(OSError):
            services._stop_service(ctx, name="a")
        assert record.service_id not in registry
    finally:
        _settle(_proc(record), identities=[(pid, birth)])


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_a_declared_output_is_registered_while_the_name_is_still_held(tmp_path, monkeypatch, scripts, backend):
    """Outputs are the last part of a stop's finalization: a new start must not write them under the old stop's copy."""
    from ouroboros.tools import services, shell

    ctx, record, registry, _log, _leader, (_port, pid, birth) = BACKENDS[backend](tmp_path, monkeypatch, scripts)
    record.outputs = ["out.txt"]
    seen = []

    def outputs(*_args, **_kwargs):
        seen.append(registry.get(record.service_id) is record)
        return "", False, False

    monkeypatch.setattr(shell, "_register_process_outputs", outputs)
    monkeypatch.setattr(services, "_working_tree_changes", lambda cwd: set())
    try:
        stopped = _payload(services._stop_service(ctx, name="a"))
        assert stopped["artifact_output_failed"] is False and seen == [True], (stopped, seen)
        assert record.service_id not in registry
    finally:
        _settle(_proc(record), identities=[(pid, birth)])
