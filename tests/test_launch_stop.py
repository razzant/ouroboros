"""A service stop ends what its launch can still be traced to, and says so only when it is true.

``start_service``/``stop_service`` through the tool registry, task-end ``stop_task_services``, the
shutdown/update ``kill_all_services``, the local executor and Emergency Stop's real caller
(``_launch_stop_shared``: the service and how it is observed; ``test_launch_stop_recovery``: what the
custody rows and durable records do when no in-memory owner is left).

POSIX only: Windows keeps its native tree kill and has no census to measure with.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tests._launch_stop_shared import (
    _PATIENCE,
    HIDDEN_TOP_RUNNER,
    RUNNER,
    _answers,
    _count_census,
    _deny_signals_to,
    _end,
    _executor_ctx,
    _executor_start,
    _ledgered_service,
    _order_spy,
    _payload,
    _pid_from,
    _read_through_ps,
    _registry,
    _runner_cmd,
    _same_process_alive,
    _server,
    _settle,
    _start,
    _stranger,
    _wait_exited,
    write_scripts,
)

pytestmark = [
              pytest.mark.skipif(os.name == "nt", reason="the measured stop is POSIX; Windows keeps its native kill")]


@pytest.fixture
def scripts(tmp_path):
    return write_scripts(tmp_path)


TERM_RECORDER = r'''
import os, signal, socket, sys, time
def note(*_):
    open(sys.argv[2], "w").close()  # received TERM, and keeps serving
signal.signal(signal.SIGTERM, note)
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
sock.listen(16)
tmp = sys.argv[1] + ".tmp"
with open(tmp, "w") as handle:
    handle.write(f"{sock.getsockname()[1]} {os.getpid()}")
os.replace(tmp, sys.argv[1])
while True:
    conn, _ = sock.accept()
    conn.sendall(b"pong\n")
    conn.close()
'''


def _assert_group_killed_before_its_scan(events: list, leader: int) -> None:
    """A durable record's native request: only its identity read comes before its group's kill. A held
    service still unreaped may have scanned earlier on its own request, in the order it always had."""
    read, kill = events.index(("read", str(leader))), events.index(("kill", -leader))
    assert read < kill and all(kind != "scan" for kind, _ in events[read:kill]), events


def _quiet_other_panic_owners(monkeypatch) -> None:
    """Only the owners this test is not about (the set ``test_server_control_panic_daemon`` neutralizes)."""
    from ouroboros import server_control
    from ouroboros.maintenance import child as packer
    from ouroboros.startup_historical_audit import audit
    from ouroboros.storage_free import free as storage_free

    monkeypatch.setattr(audit, "stop", lambda: None)
    monkeypatch.setattr(packer, "stop", lambda: None)
    monkeypatch.setattr(storage_free, "stop", lambda: None)
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda **kw: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda **kw: [])
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **kw: SimpleNamespace(
        panic_stop=lambda **kw: [], stop_server=lambda: None))
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon", lambda **kw: SimpleNamespace(
        panic_stop=lambda *, request_only: [], stop_outcome=lambda: None))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _root: None)
    monkeypatch.setattr(server_control, "_restart_stop_requested", False)


def test_stop_service_ends_a_server_its_runner_put_in_another_group_and_spares_the_sibling(tmp_path, monkeypatch, scripts):
    registry = _registry(tmp_path, monkeypatch)
    started = _start(registry, scripts, tmp_path / "demo.port", "demo")
    _start(registry, scripts, tmp_path / "sibling.port", "sibling")
    port, pid, birth = _server(tmp_path / "demo.port")
    sibling_port, sibling_pid, sibling_birth = _server(tmp_path / "sibling.port")
    try:
        assert os.getpgid(pid) != started["pgid"], "the server must lead its own group, outside the runner's"
        assert _answers(port) and _answers(sibling_port)

        stopped = _payload(registry.execute("stop_service", {"name": "demo"}))

        assert not _answers(port), "the separately grouped server must not keep serving after stop_service"
        assert not _same_process_alive(pid, birth)
        assert stopped["state"] == "exited" and stopped["returncode"] == -15  # the runner died of our TERM
        cleanup = stopped["process_cleanup"]
        assert cleanup["exit_confirmed"] is True and pid in cleanup["ended"] and started["pid"] in cleanup["ended"]
        assert not cleanup["survivors"] and not cleanup["unverified"] and "stop_failed" not in stopped
        assert stopped["log_finalization"]["deleted_live_log"] is True
        assert registry.execute("service_status", {"name": "demo"}).startswith("⚠️ SERVICE_NOT_FOUND")
        assert _answers(sibling_port) and _same_process_alive(sibling_pid, sibling_birth), "another service is never reached"
    finally:
        registry.execute("stop_service", {"name": "sibling"})
        _end(pid, birth)
        _end(sibling_pid, sibling_birth)
    assert not _answers(sibling_port)


def test_a_denied_member_keeps_record_log_and_outputs_refuses_replacement_and_a_retry_ends_it(tmp_path, monkeypatch, scripts):
    registry = _registry(tmp_path, monkeypatch)
    task_drive = tmp_path / "data" / "task_drives" / "stop-task"
    task_drive.mkdir(parents=True)
    started = json.loads(registry.execute("start_service", {
        "name": "demo", "cmd": _runner_cmd(scripts, tmp_path / "demo.port", "separate", "out=report.txt"),
        "cwd": str(task_drive), "outputs": ["report.txt"], "readiness": {"log_contains": "READY", "timeout_sec": 20}}))
    assert started["state"] == "running", started
    port, pid, birth = _server(tmp_path / "demo.port")
    deadline = time.monotonic() + _PATIENCE
    while not (task_drive / "report.txt").exists():  # written just before READY
        assert time.monotonic() < deadline
        time.sleep(0.02)
    log = tmp_path / "data" / "services" / "stop-task" / "demo.log"
    artifact = tmp_path / "data" / "task_results" / "artifacts" / "stop-task" / "report.txt"
    try:
        allow = _deny_signals_to(monkeypatch, {pid})
        failed_text = registry.execute("stop_service", {"name": "demo"})

        assert failed_text.startswith("⚠️ SERVICE_STOP_ERROR (stop_service)")
        assert _answers(port) and _same_process_alive(pid, birth), "a denied signal is not an exit"
        failed = _payload(failed_text)
        assert failed["stop_failed"] is True and failed["cleanup_dispatched"] is False
        assert failed["state"] == "exited", "the runner's own exit is reported beside the cleanup"
        assert failed["process_cleanup"]["exit_confirmed"] is False
        assert {"pid": pid, "birth": birth, "reason": "signal_denied"} in failed["process_cleanup"]["unverified"]
        assert "artifact_outputs" not in failed and "log_finalization" not in failed
        assert log.exists() and not artifact.exists()
        status = json.loads(registry.execute("service_status", {"name": "demo"}))
        assert status["process_cleanup"]["exit_confirmed"] is False  # the record is still there, saying why

        replaced = registry.execute_result("start_service", {"name": "demo", "cmd": _runner_cmd(
            scripts, tmp_path / "replacement.port")})
        assert (replaced.status, replaced.code) == ("blocked", "LEGACY_BLOCKED")
        assert replaced.text.startswith("⚠️ SERVICE_ALREADY_RUNNING: demo") and replaced.meta["process_not_started"]
        assert not (tmp_path / "replacement.port").exists() and _answers(port)

        allow()  # the runner is gone; the server is still owed by its pid and birth
        stopped = _payload(registry.execute("stop_service", {"name": "demo"}))

        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert stopped["process_cleanup"]["exit_confirmed"] is True and pid in stopped["process_cleanup"]["ended"]
        assert "cannot be ruled out" in stopped["process_cleanup"]["residual"], "its leader ended in the first attempt"
        assert started["pid"] not in stopped["process_cleanup"]["ended"]  # it ended in the first attempt
        assert "ARTIFACT_OUTPUTS" in stopped["artifact_outputs"] and artifact.read_text(encoding="utf-8") == "generated"
        assert stopped["log_finalization"]["deleted_live_log"] is True and not log.exists()
        assert json.loads(registry.execute("start_service", {
            "name": "demo", "cmd": _runner_cmd(scripts, tmp_path / "replacement.port"),
            "readiness": {"log_contains": "READY", "timeout_sec": 20}}))["state"] == "running"
        replacement = _server(tmp_path / "replacement.port")
        registry.execute("stop_service", {"name": "demo"})
        assert not _answers(replacement[0])
    finally:
        _end(pid, birth)


def test_an_unreadable_process_table_signals_nothing_and_the_retry_ends_the_tree(tmp_path, monkeypatch, scripts):
    from ouroboros import process_scope

    registry = _registry(tmp_path, monkeypatch)
    started = _start(registry, scripts, tmp_path / "demo.port", "demo")
    port, pid, birth = _server(tmp_path / "demo.port")
    real_census = process_scope.census
    try:
        monkeypatch.setattr(process_scope, "census", lambda *_a, **_k: None)
        failed = _payload(registry.execute("stop_service", {"name": "demo"}))

        assert failed["stop_failed"] is True and failed["process_cleanup"]["census_failed"] is True
        assert failed["state"] == "running" and _answers(port), "no table, no signal: the tree stays whole"
        monkeypatch.setattr(process_scope, "census", real_census)
        stopped = _payload(registry.execute("stop_service", {"name": "demo"}))

        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert {started["pid"], pid} <= set(stopped["process_cleanup"]["ended"])
    finally:
        monkeypatch.setattr(process_scope, "census", real_census)
        registry.execute("stop_service", {"name": "demo"})
        _end(pid, birth)


def test_a_leader_gone_before_the_stop_leaves_its_group_and_its_marked_orphan_to_the_stop(tmp_path, monkeypatch, scripts):
    registry = _registry(tmp_path, monkeypatch)
    _start(registry, scripts, tmp_path / "same.port", "same", mode="same-exit")
    _start(registry, scripts, tmp_path / "separate.port", "separate", mode="separate-exit")
    port, pid, birth = _server(tmp_path / "same.port")
    orphan_port, orphan_pid, orphan_birth = _server(tmp_path / "separate.port")
    try:
        for name in ("same", "separate"):  # the runner's own exit is observed before the stop
            _wait_exited(registry, name)
        assert os.getpgid(orphan_pid) == orphan_pid and _answers(orphan_port), "reparented, in its own group"

        same = _payload(registry.execute("stop_service", {"name": "same"}))
        separate = _payload(registry.execute("stop_service", {"name": "separate"}))

        assert not _answers(port) and not _same_process_alive(pid, birth), "its group outlives the leader"
        assert not _answers(orphan_port) and not _same_process_alive(orphan_pid, orphan_birth), "its marker names it"
        assert pid in same["process_cleanup"]["ended"] and orphan_pid in separate["process_cleanup"]["ended"]
        for payload in (same, separate):
            assert payload["process_cleanup"]["exit_confirmed"] is True and "cannot be ruled out" in payload[
                "process_cleanup"]["residual"], "the leader was gone before the stop: its chain is broken"
            assert payload["process_cleanup"]["reach"] == ["group", "descendants", "marker"]
        assert separate["state"] == "exited" and separate["returncode"] == 0
    finally:
        _end(pid, birth)
        _end(orphan_pid, orphan_birth)


def _legacy(registry, name: str):
    """The in-memory record of a launch without its identity (an older record's shape): no launch marker
    and no measured leader birth; once its runner is reaped, that birth can no longer be read either."""
    from ouroboros.tools import services

    record = services._SERVICES[services._service_key(registry._ctx, name)]
    record.scope_marker, record.leader_birth = "", ""
    return record


@pytest.mark.parametrize("group", ["kept-member", "emptied"])
def test_a_legacy_record_whose_leader_exited_stops_by_its_group_and_discloses_what_it_cannot_trace(
        tmp_path, monkeypatch, scripts, group):
    """Its group, bounded by the record's start, is all its stop can still trace. An unreadable table or a
    denied member keeps the record owed for a retry; then the stop ends that reach, retires the record,
    frees the name and says what it cannot rule out. Never a stop that can never complete."""
    from ouroboros import process_scope

    registry = _registry(tmp_path, monkeypatch)
    _start(registry, scripts, tmp_path / "demo.port", "demo", mode="same-exit" if group == "kept-member" else "separate-exit")
    _start(registry, scripts, tmp_path / "sibling.port", "sibling")
    port, pid, birth = _server(tmp_path / "demo.port")
    sibling_port, sibling_pid, sibling_birth = _server(tmp_path / "sibling.port")
    record = _legacy(registry, "demo")
    real_census = process_scope.census
    try:
        _wait_exited(registry, "demo")
        assert record.proc.returncode == 0
        monkeypatch.setattr(process_scope, "census", lambda *_a, **_k: None)
        unknown = registry.execute("stop_service", {"name": "demo"})
        monkeypatch.setattr(process_scope, "census", real_census)
        assert unknown.startswith("⚠️ SERVICE_STOP_ERROR") and _payload(unknown)["process_cleanup"]["census_failed"]
        if group == "kept-member":
            allow = _deny_signals_to(monkeypatch, {pid})
            assert _payload(registry.execute("stop_service", {"name": "demo"}))["stop_failed"] is True
            assert registry.execute_result("start_service", {"name": "demo", "cmd": _runner_cmd(
                scripts, tmp_path / "early.port")}).code == "LEGACY_BLOCKED" and _answers(port)
            allow()

        stopped = _payload(registry.execute("stop_service", {"name": "demo"}))

        cleanup = stopped["process_cleanup"]
        assert "stop_failed" not in stopped and cleanup["exit_confirmed"] is True
        assert "cannot be ruled out" in cleanup["residual"]
        assert cleanup["reach"] == ["group", "descendants", *(["owed"] if group == "kept-member" else [])]  # no marker
        assert stopped["log_finalization"]["deleted_live_log"] is True
        if group == "kept-member":
            assert pid in cleanup["ended"] and not _answers(port) and not _same_process_alive(pid, birth)
        else:
            assert _answers(port), "outside its group and unmarked: untraceable, exactly what the residual says"
        assert registry.execute("service_status", {"name": "demo"}).startswith("⚠️ SERVICE_NOT_FOUND")
        _start(registry, scripts, tmp_path / "again.port", "demo")  # the name is free again
        again = _server(tmp_path / "again.port")
        assert _payload(registry.execute("stop_service", {"name": "demo"}))["process_cleanup"]["exit_confirmed"]
        assert not _answers(again[0])
        assert _answers(sibling_port) and _same_process_alive(sibling_pid, sibling_birth), "another service is kept"
    finally:
        monkeypatch.setattr(process_scope, "census", real_census)
        registry.execute("stop_service", {"name": "sibling"})
        for identity in ((pid, birth), (sibling_pid, sibling_birth)):
            _end(*identity)


def test_task_end_and_the_local_executor_retire_a_legacy_record_whose_leader_exited(tmp_path, monkeypatch, scripts):
    from ouroboros import workspace_executor
    from ouroboros.tools import services
    from ouroboros.tools.services import stop_task_services

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry = _registry(tmp_path, monkeypatch)
    _start(registry, scripts, tmp_path / "legacy.port", "legacy", mode="same-exit")
    _start(registry, scripts, tmp_path / "kept.port", "kept", keep_alive=True)
    ctx, workspace = _executor_ctx(tmp_path)
    _executor_start(ctx, workspace, scripts, tmp_path / "local.port", "local", mode="same-exit")
    servers = {name: _server(tmp_path / f"{name}.port") for name in ("legacy", "kept", "local")}
    _legacy(registry, "legacy")
    local = workspace_executor._SERVICES["exec-task:local"]
    local.scope_marker, local.leader_birth = "", ""
    try:
        _wait_exited(registry, "legacy")
        assert local.local_proc.wait(timeout=_PATIENCE) == 0

        results = {item["name"]: item for item in stop_task_services(registry._ctx)}
        executor = _payload(services._stop_service(ctx, name="local"))

        for name, payload in (("legacy", results["legacy"]), ("local", executor)):
            port, pid, birth = servers[name]
            assert pid in payload["process_cleanup"]["ended"] and not _answers(port), name
            assert "cannot be ruled out" in payload["process_cleanup"]["residual"] and "stop_failed" not in payload
        assert results["legacy"]["lifecycle"] == "stopped" and executor["state"] == "stopped"
        assert "exec-task:local" not in workspace_executor._SERVICES and not local.durable_record_path.exists()
        assert results["kept"]["lifecycle"] == "kept" and _answers(servers["kept"][0])
    finally:
        registry.execute("stop_service", {"name": "kept"})
        for _port, pid, birth in servers.values():
            _end(pid, birth)


def test_task_end_stops_the_tree_keeps_kept_services_and_reports_an_unconfirmed_end(tmp_path, monkeypatch, scripts):
    from ouroboros.tools.services import stop_task_services

    registry = _registry(tmp_path, monkeypatch)
    _start(registry, scripts, tmp_path / "demo.port", "demo")
    _start(registry, scripts, tmp_path / "held.port", "held")
    _start(registry, scripts, tmp_path / "kept.port", "kept", keep_alive=True)
    port, pid, birth = _server(tmp_path / "demo.port")
    held_port, held_pid, held_birth = _server(tmp_path / "held.port")
    kept_port, kept_pid, kept_birth = _server(tmp_path / "kept.port")
    allow = _deny_signals_to(monkeypatch, {held_pid})
    try:
        results = {item["name"]: item for item in stop_task_services(registry._ctx)}

        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert results["demo"]["lifecycle"] == "stopped" and pid in results["demo"]["process_cleanup"]["ended"]
        assert _answers(held_port) and results["held"]["lifecycle"] == "cleanup_pending"
        assert results["held"]["stop_failed"] is True
        assert results["kept"]["lifecycle"] == "kept" and _answers(kept_port)
    finally:
        allow()
        for name in ("held", "kept"):
            registry.execute("stop_service", {"name": name})
        for item in ((pid, birth), (held_pid, held_birth), (kept_pid, kept_birth)):
            _end(*item)
    assert not _answers(held_port) and not _answers(kept_port)


@pytest.mark.parametrize("door", ["explicit_stop", "shutdown"])
def test_explicit_stop_offers_term_first_while_shutdown_kills_at_once(tmp_path, monkeypatch, scripts, door):
    """The causal difference between the doors: a TERM-ignoring member hears TERM only from a stop that
    can afford its grace; shutdown and update send KILL at once, inside their own allowance."""
    from ouroboros.tools.services import kill_all_services

    (scripts / "recorder.py").write_text(TERM_RECORDER, encoding="utf-8")
    heard = tmp_path / "heard-term"
    runner = scripts / "runner.py"
    runner.write_text(RUNNER.replace("[sys.executable, server, port_file]",
                                     f"[sys.executable, server, port_file, {str(heard)!r}]"), encoding="utf-8")
    registry = _registry(tmp_path, monkeypatch)
    json.loads(registry.execute("start_service", {
        "name": "deaf", "cmd": [sys.executable, str(runner), str(scripts / "recorder.py"), str(tmp_path / "deaf.port"),
                                "separate"], "readiness": {"log_contains": "READY", "timeout_sec": 20}}))
    port, pid, birth = _server(tmp_path / "deaf.port")
    try:
        if door == "explicit_stop":
            payload = _payload(registry.execute("stop_service", {"name": "deaf"}))
        else:
            payload = next(item for item in kill_all_services(tmp_path / "data", wait=True) if item["name"] == "deaf")

        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert payload["process_cleanup"]["exit_confirmed"] is True and pid in payload["process_cleanup"]["ended"]
        assert heard.exists() is (door == "explicit_stop")
    finally:
        _end(pid, birth)


def test_emergency_stop_requests_and_settles_host_and_executor_services_with_no_scan_before_a_kill(
        tmp_path, monkeypatch, scripts):
    """Emergency Stop's request and settle steps never enter the measured stop, read no process table
    before their first signal and wait for nothing: host services, held executor services and a
    durable executor record whose in-memory owner is gone (its group is killed before any snapshot)."""
    from ouroboros import process_scope, workspace_executor
    from ouroboros.tools import services

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry = _registry(tmp_path, monkeypatch)
    _start(registry, scripts, tmp_path / "host.port", "host", mode="same")
    ctx, workspace = _executor_ctx(tmp_path)
    _executor_start(ctx, workspace, scripts, tmp_path / "held.port", "held", mode="same")
    _executor_start(ctx, workspace, scripts, tmp_path / "durable.port", "durable", mode="same")
    with workspace_executor._STATE_LOCK:
        durable = workspace_executor._SERVICES.pop("exec-task:durable")
    servers = {name: _server(tmp_path / f"{name}.port") for name in ("host", "held", "durable")}

    def refuse(*_a, **_k):
        raise AssertionError("Emergency Stop never enters the measured stop")

    try:
        monkeypatch.setattr(process_scope, "end_scope", refuse)
        events = _order_spy(monkeypatch)
        requested = services.kill_all_services(tmp_path / "data", request_only=True)

        assert [kind for kind, _ in events] and all(kind == "kill" for kind, _ in events), events
        assert {item.get("service_id") for item in requested if item.get("requested")} >= {
            "stop-task:host", "exec-task:held"}
        events.clear()
        services.kill_all_services(tmp_path / "data", wait=False)

        _assert_group_killed_before_its_scan(events, durable.local_proc.pid)
        for name, (port, pid, birth) in servers.items():  # same-group servers die with their groups
            deadline = time.monotonic() + _PATIENCE
            while _answers(port) or _same_process_alive(pid, birth):
                assert time.monotonic() < deadline, f"{name} outlived Emergency Stop"
                time.sleep(0.05)
    finally:
        durable.local_proc.wait(timeout=_PATIENCE)
        for port, pid, birth in servers.values():
            _end(pid, birth)


def test_shutdown_ends_many_services_in_one_shared_pass_and_attributes_each_end(tmp_path, monkeypatch, scripts):
    from ouroboros.tools.services import kill_all_services

    registry = _registry(tmp_path, monkeypatch)
    names = ("alpha", "bravo", "charlie")
    for name in names:
        _start(registry, scripts, tmp_path / f"{name}.port", name)
    servers = {name: _server(tmp_path / f"{name}.port") for name in names}
    calls = _count_census(monkeypatch)
    allow = _deny_signals_to(monkeypatch, {servers["bravo"][1]})
    try:
        payloads = {item["name"]: item for item in kill_all_services(tmp_path / "data", wait=True)}

        assert 1 <= len(calls) <= 2, "one shared scan and rescan for every service, never one pass per service"
        for name in ("alpha", "charlie"):
            port, pid, birth = servers[name]
            assert not _answers(port) and not _same_process_alive(pid, birth)
            cleanup = payloads[name]["process_cleanup"]
            assert cleanup["exit_confirmed"] is True and pid in cleanup["ended"]
            assert not {servers[other][1] for other in names if other != name} & set(cleanup["ended"])
        assert _answers(servers["bravo"][0]) and payloads["bravo"]["stop_failed"] is True
        assert {"pid": servers["bravo"][1], "birth": servers["bravo"][2], "reason": "signal_denied"} in payloads[
            "bravo"]["process_cleanup"]["unverified"]
    finally:
        allow()
        registry.execute("stop_service", {"name": "bravo"})
        for port, pid, birth in servers.values():
            _end(pid, birth)


def test_an_unreadable_daemon_ledger_holds_chain_only_members_owed_until_it_reads_again(tmp_path, monkeypatch, scripts):
    """While nothing tells a chain-only member from a daemon root it is held unsignalled, and the stop that ends
    its leader keeps it owed, in the record and on the launch's ledger row; a renewed failure holds it again. Once
    the ledger reads, an ordinary member ends, while a recorded daemon root and its own child stay and are no
    longer owed. A service whose daemon sits below it, with the ledger readable, is spared the same way."""
    from ouroboros.process_custody import ledger_path, record_process
    from ouroboros.tools import services

    registry = _registry(tmp_path, monkeypatch)

    def launch(name, *options):
        json.loads(registry.execute("start_service", {
            "name": name, "cmd": _runner_cmd(scripts, tmp_path / f"{name}.port", "separate", *options),
            "readiness": {"log_contains": "READY", "timeout_sec": 20}}))
        return _server(tmp_path / f"{name}.port")

    servers = {"rooted": launch("rooted", f"daemon={tmp_path / 'daemon.pid'}", f"daemon_child={tmp_path / 'below.pid'}"),
               "plain": launch("plain", f"daemon={tmp_path / 'plain.pid'}"),
               "readable": launch("readable", f"daemon={tmp_path / 'readable.pid'}")}
    daemon, below, plain = (_pid_from(tmp_path / name) for name in ("daemon.pid", "below.pid", "plain.pid"))
    readable_daemon = _pid_from(tmp_path / "readable.pid")
    for root in (daemon, readable_daemon):  # recorded installation daemon roots; ``plain`` is the same shape, unrecorded
        record_process(tmp_path / "data", pid=root[0], cmd=["daemon"], purpose="claudexor_daemon", scope="daemon")
    ledger = ledger_path(tmp_path / "data")
    try:
        spared = _payload(registry.execute("stop_service", {"name": "readable"}))
        assert spared["process_cleanup"]["exit_confirmed"] is True and _same_process_alive(*readable_daemon)
        assert readable_daemon[0] not in spared["process_cleanup"]["ended"], "a live daemon root below it is spared"

        ledger.write_bytes(ledger.read_bytes() + b"{broken\n")
        for name in ("rooted", "plain"):
            first = _payload(registry.execute("stop_service", {"name": name}))
            port, pid, birth = servers[name]
            assert first["stop_failed"] is True and not _answers(port) and not _same_process_alive(pid, birth)
        for identity in (daemon, below, plain):
            assert _same_process_alive(*identity), "nothing tells it from a daemon root: never signalled"
        owed = {"pid": plain[0], "birth": plain[1], "held": True}
        assert owed in services._SERVICES[services._service_key(registry._ctx, "plain")].known_members
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if '"service:plain"' in line]
        assert owed in rows[-1]["known_members"], "the launch's ledger row carries it to any later custody"

        again = _payload(registry.execute("stop_service", {"name": "plain"}))
        assert again["stop_failed"] is True and _same_process_alive(*plain), "still unknown: still held, still owed"
        assert {"pid": plain[0], "birth": plain[1], "reason": "spared_roots_unknown"} in again["process_cleanup"]["unverified"]

        ledger.write_bytes(ledger.read_bytes().replace(b"{broken\n", b""))
        ended = _payload(registry.execute("stop_service", {"name": "plain"}))
        assert ended["process_cleanup"]["exit_confirmed"] is True and plain[0] in ended["process_cleanup"]["ended"]
        assert not _same_process_alive(*plain)
        kept = _payload(registry.execute("stop_service", {"name": "rooted"}))
        assert kept["process_cleanup"]["exit_confirmed"] is True and "stop_failed" not in kept
        assert _same_process_alive(*daemon) and _same_process_alive(*below), "a daemon root and its subtree stay"
        assert not {daemon[0], below[0]} & set(kept["process_cleanup"]["ended"])
    finally:
        for identity in (*servers.values(), daemon, below, plain, readable_daemon):
            _end(identity[-2], identity[-1])


def test_local_executor_stop_confirms_every_member_keeps_owed_identities_durably_and_refuses_overlap(
        tmp_path, monkeypatch, scripts):
    from ouroboros import owned_shutdown, process_census, process_custody, workspace_executor
    from ouroboros.tools import services

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path)
    # Its server sheds the launch marker (``hidden``): only the runner's parent chain reaches it on every reader. A
    # marker the table can read (Linux reads each environment) names the server later, so it would owe no write first.
    _executor_start(ctx, workspace, scripts, tmp_path / "demo.port", "demo", options=("hidden=1",))
    port, pid, birth = _server(tmp_path / "demo.port")
    record = workspace_executor._SERVICES["exec-task:demo"]
    durable = record.durable_record_path
    try:
        table = process_census.census([record.scope_marker])
        assert record.scope_marker not in (table[pid].markers or set()), "chain-only"
        _deny_signals_to(monkeypatch, {pid})
        with monkeypatch.context() as write:  # every custody write refuses once: nothing could name the chain-only child
            write.setattr(process_custody, "append_jsonl", lambda *_a, **_k: False)
            write.setattr(owned_shutdown, "_write_pending", lambda *_a, **_k: False)
            unrecorded = _payload(services._stop_service(ctx, name="demo"))
        assert unrecorded["stop_failed"] is True and unrecorded["process_cleanup"]["custody"] == "unavailable"
        assert record.local_proc.poll() is None and _answers(port), "nothing was signalled: its chain is left intact"
        assert {"pid": pid, "birth": birth, "reason": "custody_unwritten"} in unrecorded["process_cleanup"]["unverified"]
        failed = _payload(services._stop_service(ctx, name="demo"))

        assert failed["stop_failed"] is True and _answers(port), "the root's exit does not settle its members"
        assert "custody" not in failed["process_cleanup"] or failed["process_cleanup"]["custody"] is None
        [row] = [row for row in process_custody._read_ledger(tmp_path / "data") if row["pid"] == record.local_proc.pid]
        assert {"pid": pid, "birth": birth} in row["known_members"], "the launch's own custody carries it"
        assert "known_members" not in json.loads(durable.read_text(encoding="utf-8")), "no debt only this record names"
        replaced = services._start_service(ctx, name="demo", cmd=_runner_cmd(scripts, tmp_path / "again.port"),
                                           cwd=str(workspace))
        assert "previous service termination is not confirmed" in replaced and not (tmp_path / "again.port").exists()

        monkeypatch.undo()
        monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
        stopped = _payload(services._stop_service(ctx, name="demo"))

        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert stopped["state"] == "stopped" and pid in stopped["process_cleanup"]["ended"]
        assert not durable.exists() and "exec-task:demo" not in workspace_executor._SERVICES
    finally:
        _end(pid, birth)


def test_a_durable_executor_record_without_its_owner_ends_the_separately_grouped_child(tmp_path, monkeypatch, scripts):
    """The record a lost in-memory owner left: the update's waited stop measures it by its launch
    identity and confirms the whole reach before forgetting it (Emergency Stop: the test above)."""
    from ouroboros import workspace_executor

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path)
    _executor_start(ctx, workspace, scripts, tmp_path / "demo.port", "demo")
    port, pid, birth = _server(tmp_path / "demo.port")
    with workspace_executor._STATE_LOCK:
        record = workspace_executor._SERVICES.pop("exec-task:demo")  # its owner is gone; the durable record stays
    saved = json.loads(record.durable_record_path.read_text(encoding="utf-8"))
    try:
        assert saved["scope_marker"] == record.scope_marker and saved["host_birth"] == record.leader_birth
        results = workspace_executor.kill_all_services(ctx.drive_root, wait=True)

        assert not _answers(port) and not _same_process_alive(pid, birth)
        mine = [item for item in results if item.get("service_id") == "exec-task:demo"]
        assert mine and mine[0]["state"] == "stopped" and mine[0]["cleanup_dispatched"] is True
        assert not record.durable_record_path.exists()
    finally:
        record.local_proc.wait(timeout=_PATIENCE)
        _end(pid, birth)


def test_a_marker_free_census_reads_no_environment(monkeypatch):
    """A table without markers drops ``ps -E`` on macOS (on a busy host its per-process argument reads
    can cost seconds), and an owned stop's marker read covers only the rows it names; Linux reads
    ``/proc`` environments only when a marker is asked for."""
    from ouroboros import process_census

    if os.path.isdir("/proc"):
        pytest.skip("Linux reads /proc/<pid>/environ only when a marker is asked for")
    calls = []
    real = process_census.subprocess.run

    def spy(argv, *args, **kwargs):
        calls.append(list(argv))
        if "-E" in argv:  # the marker table is the expensive one: its argv is the fact, not its run
            return subprocess.CompletedProcess(argv, 1, "", "")
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(process_census.subprocess, "run", spy)
    rows = process_census.census([])
    me = rows[os.getpid()]
    assert (me.ppid, me.pgid) == (os.getppid(), os.getpgrp()) and me.birth and me.markers == set()
    assert calls and all("-E" not in argv for argv in calls)
    marker = "OURO_TASK_SCOPE_" + "0" * 32
    assert process_census.census([marker]) is None and "-E" in calls[-1] and "-A" in calls[-1]
    calls.clear()
    monkeypatch.setattr(process_census.subprocess, "run", real)
    from ouroboros.process_census import census

    targeted = census([marker], env_rows=lambda rows: [os.getpid()])  # an owned stop's orphan read names its pids
    assert targeted[os.getpid()].markers == set()
    assert {row.markers is None for pid, row in targeted.items() if pid != os.getpid()} == {True}


def test_an_owed_identity_already_gone_or_reused_frees_the_retry_and_touches_nothing_else(tmp_path, monkeypatch, scripts):
    """A retry whose owed identities are absent or name another process now resolves them as gone: a shape
    or a type slip there would turn the stop's own retry into an exception. The stranger is never signalled."""
    from ouroboros.tools import services

    registry = _registry(tmp_path, monkeypatch)
    _start(registry, scripts, tmp_path / "demo.port", "demo")
    port, pid, birth = _server(tmp_path / "demo.port")
    stranger, stranger_birth = _stranger(birth)
    allow = _deny_signals_to(monkeypatch, {pid})
    try:
        assert _payload(registry.execute("stop_service", {"name": "demo"}))["stop_failed"] is True
        allow()
        os.kill(pid, signal.SIGKILL)  # it ends by other means between the attempts
        deadline = time.monotonic() + _PATIENCE
        while _same_process_alive(pid, birth):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        record = services._SERVICES[services._service_key(registry._ctx, "demo")]
        assert {"pid": pid, "birth": birth} in record.known_members
        record.known_members.append({"pid": stranger.pid, "birth": "Thu Jan  1 00:00:00 2015"})  # its pid was reused

        stopped = _payload(registry.execute("stop_service", {"name": "demo"}))

        assert stopped["process_cleanup"]["exit_confirmed"] is True and "stop_failed" not in stopped
        assert stranger.poll() is None and _same_process_alive(stranger.pid, stranger_birth)
    finally:
        _settle(stranger, identities=[(pid, birth)])


def test_a_gone_identity_stays_with_the_launch_that_named_it(tmp_path, scripts):
    from ouroboros.process_census import measure_birth
    from ouroboros.process_scope import end_owned_processes

    launches = {name: _ledgered_service(tmp_path, scripts, name) for name in ("first", "second")}
    gone = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
    gone_birth = measure_birth(gone.pid)
    gone.kill()
    gone.wait(timeout=_PATIENCE)
    try:
        outcomes = end_owned_processes([
            {"key": "first", "pid": launches["first"][0].pid, "birth": measure_birth(launches["first"][0].pid),
             "known": [{"pid": gone.pid, "birth": gone_birth}]},
            {"key": "second", "pid": launches["second"][0].pid, "birth": measure_birth(launches["second"][0].pid)},
        ], graceful=False)

        assert gone.pid in outcomes["first"]["captured_gone"]
        assert gone.pid not in outcomes["second"]["captured_gone"]
        assert outcomes["first"]["exit_confirmed"] is True and outcomes["second"]["exit_confirmed"] is True
        for leader, (port, pid, birth) in launches.values():
            assert not _answers(port) and not _same_process_alive(pid, birth)
    finally:
        for leader, (_port, pid, birth) in launches.values():
            _settle(leader, identities=[(pid, birth)])


def test_emergency_stop_through_its_caller_signals_every_service_before_any_table_read(tmp_path, monkeypatch, scripts):
    """``execute_panic_stop`` itself, on real host and executor services, a durable record among them: its
    service threads send each group's first kill before any table scan and never enter the measured stop."""
    from ouroboros import process_scope, server_control, workspace_executor

    class _Exited(RuntimeError):
        pass

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry = _registry(tmp_path, monkeypatch)
    host = _start(registry, scripts, tmp_path / "host.port", "host", mode="same")
    ctx, workspace = _executor_ctx(tmp_path)
    _executor_start(ctx, workspace, scripts, tmp_path / "held.port", "held", mode="same")
    _executor_start(ctx, workspace, scripts, tmp_path / "durable.port", "durable", mode="same")
    held = workspace_executor._SERVICES["exec-task:held"].local_proc.pid
    with workspace_executor._STATE_LOCK:
        durable = workspace_executor._SERVICES.pop("exec-task:durable")
    servers = {name: _server(tmp_path / f"{name}.port") for name in ("host", "held", "durable")}
    _quiet_other_panic_owners(monkeypatch)
    monkeypatch.setattr(server_control.os, "_exit", lambda code: (_ for _ in ()).throw(_Exited(code)))
    entered, by_thread = [], {}
    for name in ("end_owned_processes", "end_held_processes"):
        monkeypatch.setattr(process_scope, name, lambda *a, _name=name, **k: entered.append(_name) or {})
    _order_spy(monkeypatch, by_thread)
    before = set(threading.enumerate())
    try:
        with pytest.raises(_Exited):
            server_control.execute_panic_stop(
                consciousness=SimpleNamespace(stop=lambda: None), kill_workers_fn=lambda **_k: None,
                data_dir=tmp_path / "data", panic_exit_code=99, log=SimpleNamespace(critical=lambda *a, **k: None))
        for thread in set(threading.enumerate()) - before:
            if thread.name.startswith("panic-"):
                thread.join(timeout=30)
                assert not thread.is_alive(), "a test-owned settlement did not finish"

        requested = by_thread["panic-services"]
        first_scan = next((index for index, (kind, _) in enumerate(requested) if kind == "scan"), len(requested))
        for group in (host["pid"], held):  # the Popen's own request: no read of any kind before it
            assert requested.index(("kill", -group)) < first_scan, requested
        _assert_group_killed_before_its_scan(requested, durable.local_proc.pid)
        assert entered == [], "Emergency Stop never enters the measured stop"
        for name, (port, pid, birth) in servers.items():
            deadline = time.monotonic() + _PATIENCE
            while _answers(port) or _same_process_alive(pid, birth):
                assert time.monotonic() < deadline, f"{name} outlived Emergency Stop"
                time.sleep(0.05)
    finally:
        durable.local_proc.wait(timeout=_PATIENCE)
        for _port, pid, birth in servers.values():
            _end(pid, birth)


def test_the_settle_step_requests_a_held_executor_service_before_it_reads_any_durable_record(tmp_path, monkeypatch, scripts):
    """Its old order: a hard exit may follow at any moment, so what a Popen already identifies goes first."""
    from ouroboros import workspace_executor

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path)
    _executor_start(ctx, workspace, scripts, tmp_path / "held.port", "held", mode="same")
    _executor_start(ctx, workspace, scripts, tmp_path / "durable.port", "durable", mode="same")
    held = workspace_executor._SERVICES["exec-task:held"].local_proc.pid
    with workspace_executor._STATE_LOCK:
        durable = workspace_executor._SERVICES.pop("exec-task:durable")
    servers = [_server(tmp_path / f"{name}.port") for name in ("held", "durable")]
    events = _order_spy(monkeypatch)
    try:
        workspace_executor.kill_all_services(ctx.drive_root, wait=False)

        first_kill = next(index for index, (kind, value) in enumerate(events) if kind == "kill" and value in (held, -held))
        assert first_kill < events.index(("read", str(durable.local_proc.pid))), events
    finally:
        durable.local_proc.wait(timeout=_PATIENCE)
        monkeypatch.undo()
        for _port, pid, birth in servers:
            _end(pid, birth)


# --------------------------------------------------------------------------- what a census cannot read


HIDDEN_LEAF_RUNNER = r'''
import subprocess, sys
child = subprocess.Popen(["/bin/sleep", "120"], stdin=subprocess.DEVNULL, start_new_session=True)
open(sys.argv[1], "w").write(str(child.pid))
print("READY", flush=True)
'''

# The sleeper sheds every scope marker (the RUNNER's ``hidden``): what SIP's ``ps -E`` shows of Apple's ``/bin/sleep``
# on a developer Mac, and on a host that shows its environment (a SIP-off runner, Linux ``/proc``) all the same.
CHAIN_ONLY_RUNNER = r'''
import os, pathlib, re, subprocess, sys, time
shed = {key: value for key, value in os.environ.items()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*_SCOPE_[0-9a-f]{32}", key)}
child = subprocess.Popen(["/bin/sleep", "120"], start_new_session=True, env=shed)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
print("READY", flush=True)
while True:
    time.sleep(60)
'''


def test_a_marker_carrier_below_a_shell_that_hides_its_environment_ends_after_its_leader_left(tmp_path, monkeypatch, scripts):
    """The runner exits and leaves ``/bin/sh -c 'server; :'`` in its own session: launchd's child hides its
    environment on macOS, the server below it visibly carries the launch marker. The marker read covers what lies
    below an orphan born since the launch, not only launchd's direct children."""
    registry = _registry(tmp_path, monkeypatch)
    (scripts / "hidden_top.py").write_text(HIDDEN_TOP_RUNNER, encoding="utf-8")
    json.loads(registry.execute("start_service", {
        "name": "demo", "cmd": [sys.executable, str(scripts / "hidden_top.py"), str(scripts / "server.py"),
                                str(tmp_path / "demo.port")], "readiness": {"log_contains": "READY", "timeout_sec": 20}}))
    port, pid, birth = _server(tmp_path / "demo.port")
    try:
        _wait_exited(registry, "demo")
        assert _answers(port)

        cleanup = _payload(registry.execute("stop_service", {"name": "demo"}))["process_cleanup"]

        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert pid in cleanup["ended"] and cleanup["exit_confirmed"] is True and "cannot be ruled out" in cleanup["residual"]
    finally:
        _end(pid, birth)


def test_a_gone_leader_discloses_the_orphan_its_marker_cannot_name(tmp_path, monkeypatch, scripts):
    """``/bin/sleep`` in its own session after its runner exited: on macOS its environment is hidden, so neither the
    chain nor the marker reaches it. The stop ends what it can trace and says what it cannot rule out."""
    registry = _registry(tmp_path, monkeypatch)
    (scripts / "hidden_leaf.py").write_text(HIDDEN_LEAF_RUNNER, encoding="utf-8")
    json.loads(registry.execute("start_service", {
        "name": "leaf", "cmd": [sys.executable, str(scripts / "hidden_leaf.py"), str(tmp_path / "leaf.pid")],
        "readiness": {"log_contains": "READY", "timeout_sec": 20}}))
    leaf, birth = _pid_from(tmp_path / "leaf.pid")
    try:
        _wait_exited(registry, "leaf")
        stopped = _payload(registry.execute("stop_service", {"name": "leaf"}))

        cleanup = stopped["process_cleanup"]
        assert "stop_failed" not in stopped and cleanup["exit_confirmed"] is True
        assert "cannot be ruled out" in cleanup["residual"]
        if _same_process_alive(leaf, birth):  # what the residual names: no chain, and a marker its binary hides
            assert sys.platform == "darwin" and leaf not in cleanup["ended"]
        else:
            assert leaf in cleanup["ended"]
    finally:
        _end(leaf, birth)


@pytest.mark.parametrize("executor", [False, True], ids=["host-service", "local-executor"])
def test_a_child_held_on_one_failed_daemon_ledger_read_stays_owed_and_the_retry_ends_it(
        tmp_path, monkeypatch, scripts, executor):
    """The reviewers' native shape: a separately grouped ``/bin/sleep`` the census reaches only through its parent
    chain (it carries no launch marker: ``CHAIN_ONLY_RUNNER``). One daemon-ledger read fails: the stop ends its runner
    and holds it, owed by pid and birth; the next stop, with the ledger readable, ends it instead of forgetting it."""
    from ouroboros import process_custody, workspace_executor
    from ouroboros.process_census import census
    from ouroboros.tools import services

    (scripts / "chain_only.py").write_text(CHAIN_ONLY_RUNNER, encoding="utf-8")
    command = [sys.executable, str(scripts / "chain_only.py"), str(tmp_path / "child.pid")]
    registry = _registry(tmp_path, monkeypatch)
    ctx = registry._ctx
    if executor:
        monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
        ctx, workspace = _executor_ctx(tmp_path)
        started = _payload(services._start_service(ctx, name="probe", cmd=command, cwd=str(workspace),
                                                   readiness={"log_contains": "READY", "timeout_sec": 20}))
        record = workspace_executor._SERVICES["exec-task:probe"]
        proc = record.local_proc
    else:
        started = _payload(registry.execute("start_service", {"name": "probe", "cmd": command,
                                                              "readiness": {"log_contains": "READY", "timeout_sec": 20}}))
        record = services._SERVICES["stop-task:probe"]
        proc = record.proc
    assert started["state"] == "running", started
    child, birth = _pid_from(tmp_path / "child.pid")
    try:
        rows = census([record.scope_marker])
        assert rows[child].ppid == proc.pid and record.scope_marker not in (rows[child].markers or set()), "chain-only"
        with monkeypatch.context() as failed_ledger:
            failed_ledger.setattr(process_custody, "live_daemon_root_pids",
                                  lambda *_a, **_k: (_ for _ in ()).throw(OSError("one failed daemon-ledger read")))
            first = _payload(services._stop_service(ctx, name="probe"))
        assert first["stop_failed"] and _same_process_alive(child, birth), first
        assert {"pid": child, "birth": birth, "reason": "spared_roots_unknown"} in first["process_cleanup"]["unverified"]
        assert {"pid": child, "birth": birth, "held": True} in record.known_members
        assert proc.wait(timeout=_PATIENCE) is not None, "its leader ended in that first stop"

        second = _payload(services._stop_service(ctx, name="probe"))

        assert second["process_cleanup"]["exit_confirmed"] is True and child in second["process_cleanup"]["ended"]
        assert not _same_process_alive(child, birth)
    finally:
        _end(child, birth)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=_PATIENCE)


def test_a_process_table_that_never_answers_at_shutdown_still_gets_the_native_kill_inside_the_one_grace(
        tmp_path, monkeypatch, scripts):
    """A deterministic stalled reader: every table read (``ps -A``) blocks until its bound, or 6 s. The exit
    stop's measured pass takes half of what is left of the one grace, then the held service gets its native group
    kill inside that grace; nothing waits per service. Emergency Stop reads no table before its first signal
    (the caller-order test above). This is the ``ps`` reader's contract, selected on every POSIX host
    (``_read_through_ps``); a stall in Linux's active ``/proc`` reader has no such bound and is NOT covered here."""
    from ouroboros import owned_shutdown, process_census

    _read_through_ps(monkeypatch)
    registry = _registry(tmp_path, monkeypatch)
    leader = _start(registry, scripts, tmp_path / "held.port", "held", mode="same")["pid"]
    port, pid, birth = _server(tmp_path / "held.port")
    budget = 3.0
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: budget)
    real_run, real_killpg, bounds, kills = process_census.subprocess.run, os.killpg, [], []

    def stalled(argv, *args, timeout=None, **kwargs):
        if list(argv)[:1] == ["ps"] and "-A" in argv:
            bounds.append(timeout)
            time.sleep(min(timeout or 6.0, 6.0))
            if timeout is None or timeout <= 6.0:
                raise subprocess.TimeoutExpired(argv, timeout)
        return real_run(argv, *args, timeout=timeout, **kwargs)

    monkeypatch.setattr(process_census.subprocess, "run", stalled)
    began = time.monotonic()
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: kills.append((time.monotonic() - began, pgid, sig))
                        or real_killpg(pgid, sig))
    before = set(threading.enumerate())
    try:
        outcome = owned_shutdown.stop_owned_work(tmp_path / "data")
        returned = time.monotonic() - began

        first = next((at for at, pgid, sig in kills if pgid == leader and sig), None)
        assert first is not None and first < budget, (kills, bounds)
        assert returned < budget + 1.0 and bounds and all(bound is not None and bound <= budget for bound in bounds)
        assert outcome["state"] == "unconfirmed", "a table that never answered confirms nothing"
        deadline = time.monotonic() + _PATIENCE
        while _answers(port) or _same_process_alive(pid, birth):  # the same-group server goes with its group
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        for thread in set(threading.enumerate()) - before:
            if thread.name.startswith("owned-stop-"):
                thread.join(timeout=30)
        monkeypatch.undo()
        _end(pid, birth)
