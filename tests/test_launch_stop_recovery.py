"""A launch that outlives its leader or its in-memory owner stays in custody until its stop is confirmed.

The periodic reaper, the update quiesce, the exit stop's record steps and the generation stop, over
ledger rows and durable executor records whose leader may be gone or its pid reused
(``_launch_stop_shared``: the service and how it is observed); both custody stores keeping a launch
under its own key, and the readers that release custody reading what it still owes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from tests._launch_stop_shared import (
    _CARRIER,
    _CHILD,
    _HEX_A,
    _HEX_B,
    _MARK_A,
    _MARK_B,
    _NEW_BIRTH,
    _OLD_BIRTH,
    _OTHER,
    _PATIENCE,
    _PID,
    HIDDEN_TOP_RUNNER,
    _answers,
    _controlled,
    _count_census,
    _deny_signals_to,
    _end,
    _executor_ctx,
    _executor_start,
    _held,
    _launch_row,
    _ledgered_service,
    _payload,
    _pid_from,
    _reader,
    _registry,
    _runner_cmd,
    _same_process_alive,
    _server,
    _settle,
    _start,
    _stranger,
    _view,
    write_scripts,
)
from tests._cases import case_folders

pytestmark = [
              pytest.mark.skipif(os.name == "nt", reason="the measured stop is POSIX; Windows keeps its native kill")]


@pytest.fixture
def scripts(tmp_path):
    return write_scripts(tmp_path)


def _leader_pid_reused_by(root, leader, stranger) -> dict:
    """The state a recycled pid leaves on disk: this launch's row now names a live stranger's pid and group,
    under the leader's recorded birth and command."""
    from ouroboros import owned_shutdown
    from ouroboros.process_custody import ledger_path

    path = ledger_path(root)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    row = next(item for item in rows if item["pid"] == leader.pid)
    row.update(pid=stranger.pid, pgid=stranger.pid)
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    owned_shutdown.record_ledgered_process(root, row, judged=())  # the launch's own record: its copy follows the row
    return row


@pytest.mark.parametrize("mode", ["same-exit", "separate-exit"])
def test_the_exit_stop_ends_what_a_gone_leader_left_before_forgetting_the_record(
        tmp_path, monkeypatch, scripts, mode):
    from ouroboros import owned_shutdown, workspace_executor

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path)
    _executor_start(ctx, workspace, scripts, tmp_path / "demo.port", "demo", mode=mode)
    port, pid, birth = _server(tmp_path / "demo.port")
    with workspace_executor._STATE_LOCK:
        record = workspace_executor._SERVICES.pop("exec-task:demo")
    try:
        record.local_proc.wait(timeout=_PATIENCE)  # the leader exited and was reaped: its group or its orphan remains
        root = owned_shutdown.installation_root(ctx.drive_root)
        entry = next(item for item in owned_shutdown.owned_records(root)
                     if item.get("record_path") == str(record.durable_record_path))

        assert owned_shutdown._stop_entry(root, entry, time.monotonic() + _PATIENCE, set()) is True

        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert not record.durable_record_path.exists()
    finally:
        _end(pid, birth)


def test_update_quiesce_ends_several_ledgered_services_in_one_pass(tmp_path, monkeypatch, scripts):
    from ouroboros.process_custody import ledger_path, quiesce_custodied_services

    launched = [_ledgered_service(tmp_path, scripts, name) for name in ("kept-a", "kept-b", "kept-c")]
    launched.append(_ledgered_service(tmp_path, scripts, "gone-d", mode="separate-exit"))
    calls = _count_census(monkeypatch)
    try:
        assert launched[-1][0].wait(timeout=_PATIENCE) == 0  # one leader already gone: its marker still names its orphan
        ok, blockers = quiesce_custodied_services(tmp_path)

        assert (ok, blockers) == (True, [])
        assert 1 <= len(calls) <= 2, "one shared pass for every target, never one per row"
        for leader, (port, pid, birth) in launched:
            assert not _answers(port) and not _same_process_alive(pid, birth)
        assert all(leader.wait(timeout=_PATIENCE) in (-9, 0) for leader, _ in launched)
        assert '"service:' not in ledger_path(tmp_path).read_text(encoding="utf-8")
    finally:
        for leader, (_port, pid, birth) in launched:
            if leader.poll() is None:
                leader.kill()
                leader.wait(timeout=_PATIENCE)
            _end(pid, birth)


@pytest.mark.parametrize("mode", ["separate", "separate-exit"])
def test_the_exit_stop_ends_a_kept_service_row_by_its_launch_identity(tmp_path, scripts, mode):
    """A kept service whose in-memory owner is gone: the exit stop's ledgered-row step ends its whole
    reach, live leader or orphan, before it forgets the row (an older row keeps its group stop)."""
    from ouroboros import owned_shutdown

    leader, (port, pid, birth) = _ledgered_service(tmp_path, scripts, "kept", mode)
    try:
        if mode.endswith("-exit"):
            assert leader.wait(timeout=_PATIENCE) == 0
        root = owned_shutdown.installation_root(tmp_path)
        entry = next(item for item in owned_shutdown.owned_records(root) if item.get("host_pid") == leader.pid)

        assert owned_shutdown._stop_entry(root, entry, time.monotonic() + _PATIENCE, set()) is True

        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert all(item.get("host_pid") != leader.pid for item in owned_shutdown.owned_records(root))
    finally:
        if leader.poll() is None:
            leader.kill()
        leader.wait(timeout=_PATIENCE)
        _end(pid, birth)


def test_the_exit_stop_ends_every_kept_service_row_in_one_shared_pass(tmp_path, monkeypatch, scripts):
    from ouroboros import owned_shutdown

    launched = [_ledgered_service(tmp_path, scripts, name) for name in ("kept-a", "kept-b", "kept-c")]
    root = owned_shutdown.installation_root(tmp_path)
    leaders = {leader.pid for leader, _ in launched}
    entries = [item for item in owned_shutdown.owned_records(root) if item.get("host_pid") in leaders]
    calls = _count_census(monkeypatch)
    try:
        assert len(entries) == 3
        remaining = owned_shutdown._stop_records(root, entries, time.monotonic() + _PATIENCE)

        assert remaining == [] and 1 <= len(calls) <= 2, "one census pass shared by every row, not one per row"
        for leader, (port, pid, birth) in launched:
            assert not _answers(port) and not _same_process_alive(pid, birth)
        assert not leaders & {item.get("host_pid") for item in owned_shutdown.owned_records(root)}
    finally:
        for leader, (_port, pid, birth) in launched:
            if leader.poll() is None:
                leader.kill()
            leader.wait(timeout=_PATIENCE)
            _end(pid, birth)


@pytest.mark.parametrize(("owner", "mode"), [("gone-task", "separate"), ("previous-generation", "separate-exit")])
def test_the_reaper_ends_a_stale_service_by_its_launch_identity_and_keeps_an_unconfirmed_row(
        tmp_path, monkeypatch, scripts, mode, owner):
    """A task service whose task is gone, or a kept service a previous server generation left."""
    from ouroboros import process_custody
    from ouroboros.process_custody import ledger_path, reap_orphaned_processes

    leader, (port, pid, birth) = _ledgered_service(tmp_path, scripts, "orphaned", mode,
                                                   scope="task" if owner == "gone-task" else "session",
                                                   owner_task="gone-task")
    try:
        if mode.endswith("-exit"):
            assert leader.wait(timeout=_PATIENCE) == 0
        if owner == "previous-generation":
            monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        allow = _deny_signals_to(monkeypatch, {pid})
        reap_orphaned_processes(tmp_path, running_task_ids=set())

        assert _answers(port) and '"service:orphaned"' in ledger_path(tmp_path).read_text(encoding="utf-8")
        allow()
        reaped = reap_orphaned_processes(tmp_path, running_task_ids=set())

        assert pid in reaped and not _answers(port) and not _same_process_alive(pid, birth)
        assert '"service:orphaned"' not in ledger_path(tmp_path).read_text(encoding="utf-8")
        events = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
        assert '"process_reaped"' in events and '"launch identity"' in events
    finally:
        if leader.poll() is None:
            leader.kill()
        leader.wait(timeout=_PATIENCE)
        _end(pid, birth)


@pytest.mark.parametrize("later", ["reaper", "exit stop"])
@pytest.mark.parametrize("scope", ["task", "session"])
def test_the_reaper_keeps_a_dead_launch_row_while_its_owner_lives_and_a_later_stop_ends_its_orphan(
        tmp_path, monkeypatch, scripts, scope, later):
    """Custody of a launch does not end with its leader: a sweep during the owner's life signals nothing and
    forgets nothing, so the stop that comes later still knows the marker its orphan carries."""
    from ouroboros import owned_shutdown, process_custody
    from ouroboros.process_custody import ledger_path, reap_orphaned_processes

    leader, (port, pid, birth) = _ledgered_service(tmp_path, scripts, "owned", "separate-exit", scope=scope,
                                                   owner_task="running")
    try:
        assert leader.wait(timeout=_PATIENCE) == 0

        assert reap_orphaned_processes(tmp_path, running_task_ids={"running"}) == []
        assert _answers(port), "a sweep never signals the service of a live owner"
        assert '"service:owned"' in ledger_path(tmp_path).read_text(encoding="utf-8"), "its marker stays in custody"

        if later == "reaper":
            if scope == "session":
                monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
            assert pid in reap_orphaned_processes(tmp_path, running_task_ids=set())
        else:
            root = owned_shutdown.installation_root(tmp_path)
            entry = next(item for item in owned_shutdown.owned_records(root) if item.get("host_pid") == leader.pid)
            assert owned_shutdown._stop_entry(root, entry, time.monotonic() + _PATIENCE, set()) is True

        assert not _answers(port) and not _same_process_alive(pid, birth)
    finally:
        _settle(leader, identities=[(pid, birth)])


def test_a_stopped_launch_leaves_its_row_to_the_next_sweep_without_a_reap_claim(tmp_path, monkeypatch, scripts):
    """The real tool's row after a confirmed stop: nothing is left, so its owner's end retires the row
    quietly; a ``process_reaped`` row or a reaped pid would claim an end that sweep did not cause."""
    from ouroboros.process_custody import ledger_path, reap_orphaned_processes

    registry = _registry(tmp_path, monkeypatch)
    started = _start(registry, scripts, tmp_path / "demo.port", "demo", mode="same")
    port, pid, birth = _server(tmp_path / "demo.port")
    data = tmp_path / "data"
    try:
        assert _payload(registry.execute("stop_service", {"name": "demo"}))["process_cleanup"]["exit_confirmed"] is True

        assert reap_orphaned_processes(data, running_task_ids={"stop-task"}) == []
        assert '"service:demo"' in ledger_path(data).read_text(encoding="utf-8")
        assert reap_orphaned_processes(data, running_task_ids=set()) == []

        assert '"service:demo"' not in ledger_path(data).read_text(encoding="utf-8")
        events = data / "logs" / "supervisor.jsonl"
        assert not events.exists() or '"process_reaped"' not in events.read_text(encoding="utf-8")
        assert started["pid"] and not _answers(port)
    finally:
        _end(pid, birth)


@pytest.mark.parametrize("consumer", ["reaper", "quiesce", "exit stop"])
def test_a_reused_leader_pid_never_retires_a_launch_before_its_marker_is_read(tmp_path, monkeypatch, scripts, consumer):
    from ouroboros import owned_shutdown
    from ouroboros.process_custody import ledger_path, quiesce_custodied_services, reap_orphaned_processes

    leader, (port, pid, birth) = _ledgered_service(tmp_path, scripts, "orphaned", "separate-exit", scope="task",
                                                   owner_task="owner")
    assert leader.wait(timeout=_PATIENCE) == 0
    stranger, stranger_birth = _stranger(birth)
    row = _leader_pid_reused_by(tmp_path, leader, stranger)
    assert row["scope_birth"] != stranger_birth
    try:
        if consumer == "reaper":
            assert reap_orphaned_processes(tmp_path, running_task_ids={"owner"}) == []
            assert _answers(port) and '"service:orphaned"' in ledger_path(tmp_path).read_text(encoding="utf-8")
            reaped = reap_orphaned_processes(tmp_path, running_task_ids=set())
            assert pid in reaped and stranger.pid not in reaped
        elif consumer == "quiesce":
            assert quiesce_custodied_services(tmp_path) == (True, [])
        else:
            root = owned_shutdown.installation_root(tmp_path)
            entry = next(item for item in owned_shutdown.owned_records(root) if item.get("host_pid") == stranger.pid)
            assert owned_shutdown._stop_entry(root, entry, time.monotonic() + _PATIENCE, set()) is True

        assert not _answers(port) and not _same_process_alive(pid, birth), "the marker still names the orphan"
        assert stranger.poll() is None and _same_process_alive(stranger.pid, stranger_birth), "never its new owner"
        if consumer == "exit stop":  # the exit stop retires its ownership record; the next start's reaper the row
            assert all(item.get("host_pid") != stranger.pid for item in owned_shutdown.owned_records(tmp_path))
        else:
            assert '"service:orphaned"' not in ledger_path(tmp_path).read_text(encoding="utf-8")
        assert '"process_not_ours"' in (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    finally:
        _settle(leader, stranger, identities=[(pid, birth)])


@pytest.mark.parametrize(("consumer", "leader_state"), [("update", "gone"), ("exit stop", "reused")])
def test_a_durable_executor_launch_ends_what_its_gone_or_reused_leader_left(
        tmp_path, monkeypatch, scripts, consumer, leader_state):
    from ouroboros import owned_shutdown, workspace_executor

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path)
    _executor_start(ctx, workspace, scripts, tmp_path / "demo.port", "demo", mode="separate-exit")
    port, pid, birth = _server(tmp_path / "demo.port")
    with workspace_executor._STATE_LOCK:
        record = workspace_executor._SERVICES.pop("exec-task:demo")
    assert record.local_proc.wait(timeout=_PATIENCE) == 0
    path, stranger, stranger_birth = record.durable_record_path, None, ""
    try:
        if leader_state == "reused":
            stranger, stranger_birth = _stranger(record.leader_birth)
            path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "host_pid": stranger.pid}),
                            encoding="utf-8")
        if consumer == "update":
            results = workspace_executor.kill_all_services(ctx.drive_root, wait=True)
            mine = [item for item in results if item.get("service_id") == "exec-task:demo"]
            assert mine and mine[0]["state"] == "stopped" and mine[0]["cleanup_dispatched"] is True
        else:
            root = owned_shutdown.installation_root(ctx.drive_root)
            entry = next(item for item in owned_shutdown.owned_records(root) if item.get("record_path") == str(path))
            assert owned_shutdown._stop_entry(root, entry, time.monotonic() + _PATIENCE, set()) is True

        assert not _answers(port) and not _same_process_alive(pid, birth)
        assert not path.exists()
        if stranger is not None:
            assert stranger.poll() is None and _same_process_alive(stranger.pid, stranger_birth)
    finally:
        _settle(stranger, identities=[(pid, birth)])


@pytest.mark.parametrize(("ledger", "consumer"), [
    ("daemon row", "quiesce"),
    ("corrupt", "reaper"),
    ("missing", "exit stop"),
])
def test_a_child_drive_stop_spares_the_installation_daemon_and_holds_it_while_its_ledger_is_unknown(
        tmp_path, scripts, consumer, ledger):
    """The installation's ledger names its daemon, whichever drive the service row lives in. When that ledger
    is corrupt or missing, nothing tells a member only the parent chain reached from a daemon root: it stays
    unsignalled and the stop unconfirmed, while what carries the launch marker still ends."""
    from ouroboros import owned_shutdown
    from ouroboros.process_census import measure_birth
    from ouroboros.process_custody import (
        ledger_path,
        quiesce_custodied_services,
        reap_orphaned_processes,
        record_process,
    )

    data, drive = tmp_path / "data", tmp_path / "data" / "task_drives" / "child"
    drive.mkdir(parents=True)
    pidfile = tmp_path / "daemon.pid"
    leader, (port, pid, birth) = _ledgered_service(tmp_path, scripts, "svc", "separate", scope="task", owner_task="gone",
                                                   drive=drive, options=[f"daemon={pidfile}"])
    daemon = int(pidfile.read_text(encoding="utf-8"))
    daemon_birth = measure_birth(daemon)
    if ledger != "missing":
        record_process(data, pid=daemon, cmd=["daemon"], purpose="claudexor_daemon", scope="daemon")
    if ledger == "corrupt":
        with ledger_path(data).open("ab") as handle:
            handle.write(b"{broken\n")
    try:
        if consumer == "quiesce":
            ok, blockers = quiesce_custodied_services(drive)
            confirmed = (ok, blockers) == (True, [])
            assert confirmed or blockers == [f"custody_service:{leader.pid}"]
        elif consumer == "reaper":
            reap_orphaned_processes(drive, running_task_ids=set())
            confirmed = '"service:svc"' not in ledger_path(drive).read_text(encoding="utf-8")
        else:
            root = owned_shutdown.installation_root(drive)
            entry = next(item for item in owned_shutdown.owned_records(root) if item.get("host_pid") == leader.pid)
            confirmed = owned_shutdown._stop_entry(root, entry, time.monotonic() + _PATIENCE, set())

        assert not _answers(port) and not _same_process_alive(pid, birth), "its marker names the server either way"
        assert _same_process_alive(daemon, daemon_birth), "a daemon root is never signalled, or held while unknown"
        assert confirmed is (ledger == "daemon row")
    finally:
        _settle(leader, identities=[(pid, birth), (daemon, daemon_birth)])


def test_the_generation_stop_ends_many_services_in_shared_passes_and_reports_an_unconfirmed_one(
        tmp_path, monkeypatch, scripts):
    from ouroboros import owned_shutdown

    registry = _registry(tmp_path, monkeypatch)
    names = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    leaders = {name: _start(registry, scripts, tmp_path / f"{name}.port", name)["pid"] for name in names}
    servers = {name: _server(tmp_path / f"{name}.port") for name in names}
    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: 60.0)
    calls = _count_census(monkeypatch)
    allow = _deny_signals_to(monkeypatch, {servers["bravo"][1]})
    try:
        outcome = owned_shutdown.stop_owned_work(tmp_path / "data")

        for name in names:
            port, pid, birth = servers[name]
            if name != "bravo":
                assert not _answers(port) and not _same_process_alive(pid, birth), name
        assert _answers(servers["bravo"][0]), "a denied member is not an exit"
        assert 1 <= len(calls) <= 4, "the in-memory owner's pass and the rows' pass, never one per service"
        [bravo] = [entry["record_id"] for entry in owned_shutdown.owned_records(tmp_path / "data")
                   if (entry.get("ledger_entry") or {}).get("pid") == leaders["bravo"]]
        assert bravo.startswith("launch-"), "a launch's record is its own, never its leader pid's"
        assert outcome["state"] == "unconfirmed" and outcome["unconfirmed"] == [bravo]
        assert outcome["memory_unconfirmed"]["services"] == {"state": "unconfirmed", "services": ["stop-task:bravo"]}
    finally:
        allow()
        registry.execute("stop_service", {"name": "bravo"})
        for _port, pid, birth in servers.values():
            _end(pid, birth)


@pytest.mark.parametrize(("mode", "consumer"), [("same-exit", "quiesce"), ("separate-exit", "reaper")])
def test_a_legacy_row_whose_leader_exited_keeps_its_old_group_stop_and_is_never_held_forever(
        tmp_path, scripts, consumer, mode):
    """A row without a launch identity (an older row's shape) keeps the stop it always had: a group that
    outlived its leader is ended through that group, an emptied one is forgotten, and a sibling whose owner
    still runs is kept by the reaper. What left the group unmarked was never traceable from such a row."""
    from ouroboros.process_custody import (
        ledger_path,
        quiesce_custodied_services,
        reap_orphaned_processes,
        spawn_supervised,
    )

    leader = spawn_supervised(_runner_cmd(scripts, tmp_path / "legacy.port", mode), drive_root=tmp_path,
                              purpose="service:legacy", scope="task", owner_task_id="gone", stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    sibling, (sibling_port, sibling_pid, sibling_birth) = (None, (0, 0, "")) if consumer == "quiesce" else (
        _ledgered_service(tmp_path, scripts, "sibling", scope="task", owner_task="running"))
    try:
        assert leader.stdout.readline().strip() == b"READY" and leader.wait(timeout=_PATIENCE) == 0
        port, pid, birth = _server(tmp_path / "legacy.port")
        assert "scope_marker" not in ledger_path(tmp_path).read_text(encoding="utf-8").splitlines()[0]

        if consumer == "quiesce":
            assert quiesce_custodied_services(tmp_path) == (True, [])
        else:
            reap_orphaned_processes(tmp_path, running_task_ids={"running"})

        assert '"service:legacy"' not in ledger_path(tmp_path).read_text(encoding="utf-8")
        assert _answers(port) is (mode == "separate-exit"), "its group is its whole reach"
        if sibling is not None:
            assert _answers(sibling_port) and _same_process_alive(sibling_pid, sibling_birth)
            assert '"service:sibling"' in ledger_path(tmp_path).read_text(encoding="utf-8")
    finally:
        _settle(leader, sibling, identities=[(pid, birth), (sibling_pid, sibling_birth)])


# --------------------------------------------------------------------------- what a stop still owes, handed on


@pytest.mark.parametrize("consumer", ["reaper", "quiesce", "exit-stop"])
def test_a_denied_chain_member_stays_owed_on_the_row_and_the_next_pass_of_each_consumer_retries_it(
        tmp_path, monkeypatch, consumer):
    """The reviewers' table-and-signal doubles (no OS process is signalled, files stay in ``tmp_path``): a child
    the parent chain reached is denied while its leader dies. The real consumer keeps it owed on the launch's row,
    in the ledger and in the ownership set alike, so its next pass signals it again instead of retiring the row."""
    from ouroboros import owned_shutdown, platform_layer, process_census, process_custody, process_scope

    leader, child = 880001, 880002
    birth = "Mon Nov 2 12:00:00 2026"
    marker = "OURO_LAUNCH_SCOPE_" + "a" * 32
    rows = {
        leader: process_census._Row(leader, 2, leader, birth, 123.0, "alive", {marker}),
        child: process_census._Row(child, leader, child, birth, 123.0, "alive", set()),
    }
    signals, denied = [], {child}

    def identities(pids):
        return {pid: process_census._Identity("alive", rows[pid].birth, rows[pid].pgid)
                if pid in rows else process_census._Identity("absent") for pid in pids}

    def signal(pid, *, graceful):
        signals.append(pid)
        if pid in denied:
            return platform_layer.SIGNAL_DENIED
        rows.pop(pid, None)
        if pid == leader and child in rows:
            rows[child].ppid = 1
        return platform_layer.SIGNAL_SENT

    monkeypatch.setattr(process_custody, "IS_WINDOWS", False)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    monkeypatch.setattr(process_custody, "_fingerprint_matches", lambda row, **kw: row["pid"] in rows)
    monkeypatch.setattr(process_custody, "_service_group_survives_leader", lambda row: False)
    monkeypatch.setattr(process_custody, "measured_identity_mismatch", lambda row: {})
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda root: set())
    monkeypatch.setattr(process_custody, "live_daemon_root_pids", lambda *a, **kw: set())
    monkeypatch.setattr(process_scope, "census", lambda *a, **kw: dict(rows))
    monkeypatch.setattr(process_scope, "read_identities", identities)
    monkeypatch.setattr(process_scope, "birth_epoch", lambda value: 123.0 if value else 0.0)
    monkeypatch.setattr(platform_layer, "deliver_signal", signal)
    monkeypatch.setattr(process_scope, "_RESCAN_SETTLE_SEC", 0)
    monkeypatch.setattr(owned_shutdown, "_exit_proven", lambda pid: pid not in rows)

    entry = {"pid": leader, "pgid": leader, "purpose": "service:review-probe", "scope": "task",
             "owner_task": "finished", "session_id": "previous-generation", "scope_birth": birth,
             "scope_marker": marker, "fingerprint": {}}
    path = process_custody.ledger_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert owned_shutdown.record_ledgered_process(tmp_path, entry)

    def stop():
        if consumer == "reaper":
            return process_custody.reap_orphaned_processes(tmp_path, running_task_ids=set())
        if consumer == "quiesce":
            return process_custody.quiesce_custodied_services(tmp_path, timeout_sec=0)
        record = next(row for row in owned_shutdown.owned_records(tmp_path) if row["host_pid"] == leader)
        return owned_shutdown._stop_entry(tmp_path, record, time.monotonic() + 5, set())

    stop()
    assert leader not in rows and child in rows
    owed = {"pid": child, "birth": birth}
    latest = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][-1]
    assert latest["known_members"] == [owed], "the ledger row carries it"
    record = next(row for row in owned_shutdown.owned_records(tmp_path) if row["host_pid"] == leader)
    assert record["ledger_entry"]["known_members"] == [owed], "and so does the ownership set"
    denied.clear()

    stop()

    assert child not in rows and signals.count(child) == 2, "a previously measured, denied member is retried"
    if consumer == "exit-stop":
        assert all(row["host_pid"] != leader for row in owned_shutdown.owned_records(tmp_path))
    else:
        assert '"service:review-probe"' not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("first,then", [("reaper", "reaper"), ("exit stop", "exit stop"), ("reaper", "exit stop")])
def test_a_held_child_survives_renewed_ledger_failure_and_any_later_consumer_ends_it(tmp_path, scripts, first, then):
    """A torn installation ledger: the first stop ends the leader and holds its chain-only child; a renewed
    failure holds it again; once the ledger reads, whichever consumer comes next ends it from the row."""
    from ouroboros import owned_shutdown
    from ouroboros.process_custody import ledger_path, reap_orphaned_processes

    pidfile = tmp_path / "chain.pid"
    leader, (port, pid, birth) = _ledgered_service(tmp_path, scripts, "svc", "separate", scope="task", owner_task="gone",
                                                   options=[f"daemon={pidfile}"])
    child = _pid_from(pidfile)
    path = ledger_path(tmp_path)
    path.write_bytes(path.read_bytes() + b"{broken\n")
    root = owned_shutdown.installation_root(tmp_path)

    def stop(consumer) -> bool:
        if consumer == "reaper":
            reap_orphaned_processes(tmp_path, running_task_ids=set())
            return '"service:svc"' not in path.read_text(encoding="utf-8")
        entry = next(item for item in owned_shutdown.owned_records(root) if item.get("host_pid") == leader.pid)
        return owned_shutdown._stop_records(root, [entry], time.monotonic() + _PATIENCE) == []

    try:
        assert stop(first) is False
        assert leader.wait(timeout=_PATIENCE) == -9 and not _answers(port) and _same_process_alive(*child)
        assert stop(first) is False and _same_process_alive(*child), "still unknown: held again, still owed"
        path.write_bytes(path.read_bytes().replace(b"{broken\n", b""))

        assert stop(then) is True

        assert not _same_process_alive(*child)
    finally:
        _settle(leader, identities=[(pid, birth), child])


def test_a_denied_chain_only_child_stays_owed_on_the_row_and_the_next_sweep_ends_it(tmp_path, monkeypatch, scripts):
    """A readable ledger, one denied signal: the row carries the identity, so the leader's death does not erase
    the only custody that named it."""
    from ouroboros.process_custody import ledger_path, reap_orphaned_processes

    pidfile = tmp_path / "chain.pid"
    leader, (port, pid, birth) = _ledgered_service(tmp_path, scripts, "svc", "separate", scope="task", owner_task="gone",
                                                   options=[f"daemon={pidfile}"])
    child = _pid_from(pidfile)
    allow = _deny_signals_to(monkeypatch, {child[0]})
    try:
        reap_orphaned_processes(tmp_path, running_task_ids=set())
        assert '"service:svc"' in ledger_path(tmp_path).read_text(encoding="utf-8")
        assert leader.wait(timeout=_PATIENCE) == -9 and _same_process_alive(*child)
        allow()

        reaped = reap_orphaned_processes(tmp_path, running_task_ids=set())

        assert child[0] in reaped and not _same_process_alive(*child)
        assert '"service:svc"' not in ledger_path(tmp_path).read_text(encoding="utf-8")
    finally:
        allow()
        _settle(leader, identities=[(pid, birth), child])


def test_what_a_host_stop_still_owes_reaches_the_next_generations_sweep(tmp_path, monkeypatch, scripts):
    """The in-memory record dies with its server; the launch's ledger row is the custody left. It carries the
    held identity the last stop could not judge, so the next generation's sweep ends it once the ledger reads."""
    from ouroboros import process_custody
    from ouroboros.process_custody import ledger_path, reap_orphaned_processes
    from ouroboros.tools import services

    registry = _registry(tmp_path, monkeypatch)
    pidfile = tmp_path / "plain.pid"
    json.loads(registry.execute("start_service", {
        "name": "demo", "cmd": _runner_cmd(scripts, tmp_path / "demo.port", "separate", f"daemon={pidfile}"),
        "readiness": {"log_contains": "READY", "timeout_sec": 20}}))
    port, pid, birth = _server(tmp_path / "demo.port")
    child = _pid_from(pidfile)
    data = tmp_path / "data"
    ledger = ledger_path(data)
    try:
        ledger.write_bytes(ledger.read_bytes() + b"{broken\n")
        assert _payload(registry.execute("stop_service", {"name": "demo"}))["stop_failed"] is True
        assert _same_process_alive(*child) and not _answers(port)
        services._SERVICES.clear()  # its server generation is gone
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")
        ledger.write_bytes(ledger.read_bytes().replace(b"{broken\n", b""))

        reaped = reap_orphaned_processes(data, running_task_ids=set())

        assert child[0] in reaped and not _same_process_alive(*child)
    finally:
        _end(pid, birth)
        _end(*child)


def test_the_next_generations_sweep_ends_the_marker_carrier_a_kept_row_left_below_a_hidden_shell(
        tmp_path, monkeypatch, scripts):
    """A kept service carries no task marker, so its row is its last custody; the server below an
    environment-hiding shell still shows the launch marker, and the sweep reads it."""
    from ouroboros import process_custody
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, SCOPE_VALUE, mint_marker
    from ouroboros.process_custody import ledger_path, reap_orphaned_processes, spawn_supervised

    (scripts / "hidden_top.py").write_text(HIDDEN_TOP_RUNNER, encoding="utf-8")
    marker = mint_marker(LAUNCH_SCOPE_PREFIX)
    leader = spawn_supervised([sys.executable, str(scripts / "hidden_top.py"), str(scripts / "server.py"),
                               str(tmp_path / "kept.port")], drive_root=tmp_path, purpose="service:kept",
                              scope="session", owner_task_id="t", scope_marker=marker,
                              env={**os.environ, marker: SCOPE_VALUE}, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    assert leader.stdout.readline().strip() == b"READY" and leader.wait(timeout=_PATIENCE) == 0
    port, pid, birth = _server(tmp_path / "kept.port")
    try:
        monkeypatch.setattr(process_custody, "_SESSION_ID", "next-generation")

        reaped = reap_orphaned_processes(tmp_path, running_task_ids=set())

        assert pid in reaped and not _answers(port) and not _same_process_alive(pid, birth)
        assert '"service:kept"' not in ledger_path(tmp_path).read_text(encoding="utf-8")
    finally:
        _settle(leader, identities=[(pid, birth)])


@pytest.mark.parametrize(("consumer", "order", "b_row"), [
    ("reaper", "owed-then-reuse", "launch"),
    ("quiesce", "reuse-then-late-owed", "unmarked"),
    ("exit-stop", "owed-then-reuse", "launch"),
])
def test_an_old_launchs_owed_child_outlives_a_new_process_that_reuses_its_dead_leaders_pid(
        tmp_path, monkeypatch, consumer, order, b_row):
    """Launch A owes a denied child after its leader died; the kernel then hands A's leader pid to B, a launch or an
    older unmarked spawn, whose spawn appends its row under that pid. A's owed row lands before B's, or late, after
    it (a stop of A that finished after B started). Both custody stores keep A's obligation and B's custody under
    their own keys; the next pass of each consumer ends A's child, records the stranger on A's pid only after that
    end, and the reaper leaves B alone while B's owner runs."""
    from ouroboros import owned_shutdown, process_census, process_custody

    rows = {_PID: process_census._Row(_PID, 2, _PID, _OLD_BIRTH, 123.0, "alive", {_MARK_A}),
            _CHILD: process_census._Row(_CHILD, _PID, _CHILD, _OLD_BIRTH, 123.0, "alive", set())}
    world = _controlled(tmp_path, monkeypatch, rows)
    world.denied.add(_CHILD)
    world.spawned(_launch_row("finished"))
    late, real_note_owed = [], process_custody.note_owed
    if order == "reuse-then-late-owed":
        monkeypatch.setattr(process_custody, "note_owed", lambda *a, **kw: late.append((a, kw)))

    def stop():
        if consumer == "reaper":
            return process_custody.reap_orphaned_processes(tmp_path, running_task_ids={"running"})
        if consumer == "quiesce":
            return process_custody.quiesce_custodied_services(tmp_path, timeout_sec=0)
        return owned_shutdown._stop_records(tmp_path, owned_shutdown.owned_records(tmp_path), time.monotonic() + 5)

    stop()
    assert _PID not in rows and _CHILD in rows, "A's leader ended, its denied child is owed"
    world.denied.clear()
    rows[_PID] = process_census._Row(_PID, 2, _PID, _NEW_BIRTH, 123.0, "alive", {_MARK_B} if b_row == "launch" else set())
    b = _launch_row("running", marker=_MARK_B if b_row == "launch" else "", birth=_NEW_BIRTH, purpose="service:new",
                    session=process_custody._SESSION_ID)
    world.spawned(b)
    if order == "reuse-then-late-owed":
        assert len(late) == 1
        monkeypatch.setattr(process_custody, "note_owed", real_note_owed)
        real_note_owed(*late[0][0], **late[0][1])

    ledger, records = _view(tmp_path)
    a_id, b_id = "launch-" + _HEX_A, ("launch-" + _HEX_B if b_row == "launch" else f"pid-{_PID}")
    assert [row.get("known_members") for row in ledger if row["purpose"] == "service:old"] == [
        [{"pid": _CHILD, "birth": _OLD_BIRTH}]], "A's owed child stays in the ledger"
    assert [row["owner_task"] for row in ledger if row["purpose"] == "service:new"] == ["running"], "and B's row"
    assert {rid: rec["ledger_entry"].get("known_members") for rid, rec in records.items()
            if rec["purpose"] == "service:old"} == {a_id: [{"pid": _CHILD, "birth": _OLD_BIRTH}]}
    assert [rid for rid, rec in records.items() if rec["purpose"] == "service:new"] == [b_id]

    stop()

    assert _CHILD not in rows, "A's custody outlives the reuse of its leader's pid: its owed child is ended"
    ledger, records = _view(tmp_path)
    assert a_id not in records, "A's record goes once its end is confirmed"
    if consumer != "exit-stop":  # the exit stop leaves its ledger rows to the next sweep's compaction
        assert not [row for row in ledger if row.get("scope_marker") == _MARK_A]
    if consumer == "reaper":  # B's owner runs: B stays, and the stranger on A's pid is recorded after A's end
        assert rows[_PID].birth == _NEW_BIRTH and (_PID, _NEW_BIRTH) not in world.signals and _PID not in world.kills
        assert b_id in records and [row["purpose"] for row in ledger] == ["service:new"], "B's custody is kept"
        events = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()]
        [refuted] = [event for event in events if event.get("type") == "process_not_ours"]
        assert refuted["record_id"] == a_id and refuted["pid"] == _PID


@pytest.mark.parametrize("late", ["before-compaction", "after-unlock", "after-forget"])
def test_a_late_row_of_the_same_launch_keeps_its_custody_across_a_compaction_that_dropped_the_launch(
        tmp_path, monkeypatch, late):
    """A compaction drops launch A's observed rows (its end was confirmed) while a stop of A that finished later
    records what A still owes (``note_owed``: one appended row, then its ownership record). That row lands before
    the compaction takes the lock, while it holds it, between its unlock and its forgetting, or after both: the
    ledger and the set end up naming it either way, and an unmarked row keeps its pid key throughout."""
    import threading

    from ouroboros import owned_shutdown, process_custody, utils
    from ouroboros.utils import append_jsonl

    path = process_custody.ledger_path(tmp_path)
    path.parent.mkdir(parents=True)
    legacy = {"pid": _OTHER, "pgid": _OTHER, "purpose": "worker:0", "scope": "session", "fingerprint": {}}
    for row in (_launch_row("finished"), legacy):
        assert append_jsonl(path, {"ts": "t0", **row}) and owned_shutdown.record_ledgered_process(tmp_path, row)
    _ok, entries, previous = process_custody._read_ledger_records(tmp_path, strict=True)
    [a_entry] = [entry for entry in entries if entry.get("scope_marker") == _MARK_A]
    owed = [{"pid": _CHILD, "birth": _OLD_BIRTH}]
    landed = []

    def land():  # once: what that later stop of A records
        if not landed:
            landed.append(process_custody.note_owed(tmp_path, a_entry, owed))

    forgetting = "forget_ledgered_rows" if hasattr(owned_shutdown, "forget_ledgered_rows") else "forget_ledgered_pids"
    real_forget, real_replace, waiting = getattr(owned_shutdown, forgetting), utils.replace_atomic, []

    def forget(drive_root, rows):
        if late == "after-unlock":
            land()
        result = real_forget(drive_root, rows)
        if late == "after-forget":
            land()
        return result

    def replace(tmp, target):  # inside the compaction's lock: the late append waits for it
        if late == "during-compaction" and not waiting:
            waiting.append(threading.Thread(target=land))
            waiting[0].start()
            time.sleep(0.2)
        return real_replace(tmp, target)

    if late == "before-compaction":
        land()
    monkeypatch.setattr(owned_shutdown, forgetting, forget)
    monkeypatch.setattr(utils, "replace_atomic", replace)
    process_custody._rewrite_ledger(tmp_path, [entry for entry in entries if entry is not a_entry], previous=previous)
    for thread in waiting:
        thread.join(5)

    assert landed and landed[0] is not None
    ledger, records = _view(tmp_path)
    assert [row.get("known_members") for row in ledger if row.get("scope_marker") == _MARK_A] == [owed]
    assert [rec["ledger_entry"].get("known_members") for rec in records.values()
            if rec["ledger_entry"].get("scope_marker") == _MARK_A] == [owed], "the set still names A's late row"
    assert [rid for rid, rec in records.items() if rec["purpose"] == "worker:0"] == [f"pid-{_OTHER}"]
    _ok, entries, previous = process_custody._read_ledger_records(tmp_path, strict=True)
    process_custody._rewrite_ledger(tmp_path, [entry for entry in entries if entry.get("pid") != _OTHER],
                                    previous=previous)  # the unmarked row's end: its pid record goes with it
    ledger, records = _view(tmp_path)
    assert len(path.read_text().splitlines()) == len(ledger) == 1 and set(records) == {"launch-" + _HEX_A}


def test_an_older_pid_spelling_of_a_launch_record_and_its_pending_registrations_stay_that_launchs_own(tmp_path):
    """The ownership set an earlier build wrote names launch A ``pid-<pid>``; a contended registration of the
    unmarked process that reused that pid waits as a pending file under the same name, and so does a late row of
    A in the old spelling. Every read takes A's record as A's own (its earliest stop stamp kept, the child its late
    copy owes added as held: a pending copy may be older than what is stored) and B's apart; one locked update
    stores both. Matching neither of A's ledger rows, that record outlives their compaction until a stop retires
    it; forgetting its own exact row leaves B."""
    from ouroboros import owned_shutdown, process_custody

    root = tmp_path
    a = {**_launch_row("finished"), "fingerprint": {"start_time": _OLD_BIRTH}}
    b = {**_launch_row("running", marker="", purpose="service:new"), "fingerprint": {"start_time": _NEW_BIRTH}}
    a_late = {**a, "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH}]}
    old_id, a_id = f"pid-{_PID}", "launch-" + _HEX_A

    def record(row, record_id, stop=None):  # the shape record_ledgered_process stores
        return {"record_id": record_id, "kind": "supervised", "host_pid": _PID, "birth": row["fingerprint"]["start_time"],
                "purpose": row["purpose"], "scope": row["scope"], "owner_task": row["owner_task"],
                "drive_root": str(root.resolve()), "record_path": str(process_custody.ledger_path(root.resolve())),
                "ledger_entry": row, "stop_requested_at": stop, "unconfirmed_since": None}

    path = owned_shutdown.owned_processes_path(root)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "records": {old_id: record(a, old_id, "2026-11-02T12:00:00+00:00")}}))
    pending = root / "state" / owned_shutdown.PENDING_DIRNAME
    pending.mkdir()
    (pending / f"{old_id}.1.json").write_text(json.dumps(record(b, old_id)))  # B's contended registration
    (pending / f"{old_id}.2.json").write_text(json.dumps(record(a_late, old_id)))  # A's late row, old spelling

    a_folded = {**a, "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH, "held": True}]}
    for read in range(2):
        records = {entry["record_id"]: entry for entry in owned_shutdown.owned_records(root, strict=True)}
        named = [rec["ledger_entry"] for rec in records.values()]
        assert len(named) == 2 and a_folded in named and b in named, "both A and B stay named"
        assert records[a_id]["ledger_entry"] == a_folded
        assert records[a_id]["stop_requested_at"] == "2026-11-02T12:00:00+00:00"
        assert records[old_id]["ledger_entry"] == b
        if not read:
            assert owned_shutdown._update(root, lambda document: False)  # folds and stores both under their ids
            assert not list(pending.iterdir()) and set(json.loads(path.read_text())["records"]) == {a_id, old_id}
    other = tmp_path / "both"  # A under both spellings: its own record keeps its row, the earliest stamp stays
    owned_shutdown.owned_processes_path(other).parent.mkdir(parents=True)
    owned_shutdown.owned_processes_path(other).write_text(json.dumps({"records": {
        old_id: record(a, old_id, "2026-11-02T11:00:00+00:00"), a_id: record(a_late, a_id, "2026-11-02T12:00:00+00:00")}}))
    [merged] = owned_shutdown.owned_records(other, strict=True)
    assert (merged["record_id"], merged["ledger_entry"]) == (a_id, a_late)
    assert merged["stop_requested_at"] == "2026-11-02T11:00:00+00:00"
    assert owned_shutdown.forget_ledgered_rows(root, [{"ts": "t", **a}, {"ts": "t", **a_late}])
    assert {entry["record_id"] for entry in owned_shutdown.owned_records(root, strict=True)} == {a_id, old_id}
    assert owned_shutdown.forget_ledgered_rows(root, [{"ts": "t", **a_folded}])
    assert [entry["ledger_entry"] for entry in owned_shutdown.owned_records(root, strict=True)] == [b]
    assert process_custody.custody_record_id({"pid": _PID, "scope_marker": "OURO_TASK_SCOPE_" + _HEX_A}) == old_id


_READER_CASES = {  # rows alive, A's and its sibling's owed children, unreadable identities; held answers; censuses
    "owed child": ({_CHILD}, [_CHILD], [], set(), "live", "none", 1),
    "marker only": ({_CARRIER}, [], [], set(), "live", "none", 1),
    "unknown": ({_CHILD}, [_CHILD], [], set(), "unknown", "unknown", 1),
    "reused pid": ({_PID}, [], [], set(), "none", "none", 1),
    "live leader": ({_PID, _OTHER}, [], [], set(), "live", "live", 0),
    "mixed": ({_PID, _CHILD}, [], [_CHILD], set(), "live", "live", 1),
    "unreadable leader": ({_PID}, [], [], {_PID}, "unknown", "unknown", 1),
}


def test_a_reader_holds_a_launch_by_what_it_still_owes_never_by_the_process_on_its_leaders_pid(subtests, tmp_path, monkeypatch):
    """The four readers that release custody (Presence's lend, a cold sleep, a Continue's writers, a body
    candidate's transfer and GC) over launch A and a sibling launch of the same owner. A leader measured alive with
    its recorded birth holds its launch with no process table; a gone leader's launch is held by a live owed child
    or a process that kept its marker, read in ONE table for every such launch; an unreadable table or leader
    identity holds it as unknown (never as a running writer); a stranger that reused its leader pid is not it.
    Controlled identities, no OS process signalled."""
    case_paths = case_folders(tmp_path)
    for case in list(_READER_CASES):
        for reader in ["presence", "cold sleep", "continuation", "body candidate"]:
            tmp_path = next(case_paths)
            with subtests.test(case=case, reader=reader), monkeypatch.context() as monkeypatch:
                from ouroboros import owned_shutdown, process_census

                alive, a_owed, sibling_owed, unreadable, expect_a, expect_sibling, censuses = _READER_CASES[case]
                row = {_PID: (2, _NEW_BIRTH if case == "reused pid" else _OLD_BIRTH, set()), _OTHER: (2, _OLD_BIRTH, {_MARK_B}),
                       _CHILD: (1, _OLD_BIRTH, set()), _CARRIER: (1, _OLD_BIRTH, {_MARK_A})}
                rows = {pid: process_census._Row(pid, row[pid][0], pid, row[pid][1], 123.0, "alive", row[pid][2]) for pid in alive}
                read, owner = _reader(tmp_path, monkeypatch, reader)
                world = _controlled(tmp_path, monkeypatch, rows)
                world.unknown, world.unreadable = case in ("unknown", "unreadable leader"), unreadable
                owed = [{"pid": pid, "birth": _OLD_BIRTH} for pid in a_owed]
                world.spawned({**_launch_row(owner), "known_members": owed})
                world.spawned({**_launch_row(owner, marker=_MARK_B, purpose="service:sibling", pid=_OTHER),
                               "known_members": [{"pid": pid, "birth": _OLD_BIRTH} for pid in sibling_owed]})
                assert len(owned_shutdown.owned_records(tmp_path)) == 2

                found = read()

                assert _held(found, _PID) == expect_a, found
                if reader != "body candidate" or "unknown" not in (expect_a, expect_sibling):  # a body's -1 names no record
                    assert _held(found, _OTHER) == expect_sibling, found
                assert len(world.censuses) == censuses, "a measured live leader needs no table; the others share ONE"
                assert world.signals == [] and world.kills == []


@pytest.mark.parametrize("case", ["owed child", "reused pid"])
def test_a_body_candidate_stays_with_its_owner_while_its_launch_owes_a_child_through_transfer_and_gc(
        tmp_path, monkeypatch, case):
    """The same reading at a candidate's two release points: its transfer to the next task and its GC verdict."""
    from ouroboros import body_candidate, owned_shutdown, process_census
    from ouroboros.task_results import write_task_result
    from tests.body_candidate_support import isolate, make_ctx, make_serving

    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    first = make_ctx(serving, data, "root-first")
    body_candidate.prepare(first)
    write_task_result(data, "root-first", "completed", result="done")
    rows = ({_CHILD: process_census._Row(_CHILD, 1, _CHILD, _OLD_BIRTH, 123.0, "alive", set())} if case == "owed child"
            else {_PID: process_census._Row(_PID, 2, _PID, _NEW_BIRTH, 123.0, "alive", set())})
    world = _controlled(data, monkeypatch, rows)
    world.spawned({**_launch_row("root-first"),
                   "known_members": [{"pid": _CHILD, "birth": _OLD_BIRTH}] if case == "owed child" else []})
    assert owned_shutdown.owned_records(data)
    row = body_candidate.find("root-first")

    verdict = body_candidate.retention_verdict(row, expired=True, data_dir=data)
    if case == "owed child":
        assert verdict["reason"] == "owner_processes_live" and not verdict["remove"]
        with pytest.raises(body_candidate.CandidateRefused) as refused:
            body_candidate._transfer(make_ctx(serving, data, "root-second"), row, "root-second", via="test")
        assert refused.value.code == "CANDIDATE_OWNER_PROCESSES_LIVE" and body_candidate.find("root-first")
    else:
        assert verdict["reason"] != "owner_processes_live"
        moved = body_candidate._transfer(make_ctx(serving, data, "root-second"), row, "root-second", via="test")
        assert moved["task_id"] == "root-second"
    assert world.signals == [] and world.kills == []
