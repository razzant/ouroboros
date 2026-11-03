"""Panic ends every live member of the installation scope by identity (owner 4A, #1772).

Today's owner requests come first and unchanged; the census is one more bounded request
attempt that never delays the hard exit, works with the daemon already dead, leaves a
durable pending record for whatever it could not finish, and never signals the server,
its launcher, pid 1 or another installation's members. Real children belong to this test
process and are ended by exact pid.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import platform_layer as _pl
from ouroboros import process_scope as ps

pytestmark = [pytest.mark.serial, pytest.mark.skipif(os.name == "nt", reason="the environment census is POSIX-only")]

REPO = pathlib.Path(__file__).resolve().parents[1]


def _sleeper(env=None, *, argv=None) -> subprocess.Popen:
    return subprocess.Popen(argv or [sys.executable, "-c", "import time; time.sleep(120)"],
                            env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)


def _end(*procs) -> None:
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


def test_the_census_ends_marker_members_and_recorded_task_groups_with_no_daemon_anywhere(tmp_path):
    marker = ps.installation_marker(tmp_path)
    member = _sleeper({marker: "1"})  # a worker/command/service of this installation (env visible)
    platform_child = _sleeper(argv=["/bin/sleep", "120"])  # a task command's platform-binary leftover
    ps.record_task_group(tmp_path, "t-ended", platform_child.pid)
    foreign = _sleeper({ps.mint_marker(ps.INSTALL_SCOPE_PREFIX): "1"})  # another installation
    try:
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert outcome["requested"] is True and not outcome["census_failed"]
        assert set(outcome["ended"]) == {member.pid, platform_child.pid}
        assert outcome["survivors"] == [] and outcome["recorded_groups"] == 1
        assert outcome["residual"] == ps.PANIC_CENSUS_RESIDUAL  # the engine's harness shells are disclosed
        assert outcome["engine_index"]["status"] == "absent" and outcome["settled"] is True, "no index: nothing owed to it"
        assert _gone(member.pid) and _gone(platform_child.pid)
        assert foreign.poll() is None, "another installation's member is never signalled"
        assert not ps.panic_sweep_pending_path(tmp_path).exists(), "a finished census owes nothing"
    finally:
        _end(member, platform_child, foreign)


def test_without_a_planted_marker_the_census_requests_nothing(tmp_path):
    assert ps.panic_scope_census(tmp_path) == {"requested": False, "reason": "no installation marker planted"}
    assert ps.complete_pending_panic_sweep(tmp_path) is None


def test_the_census_excludes_the_server_its_launcher_and_pid_one(tmp_path, monkeypatch):
    marker = ps.installation_marker(tmp_path)
    birth = "Sat Oct 10 20:10:23 2026"
    rows = {
        1: ps._Row(1, 0, 1, birth, 1.0, markers={marker}),
        4000: ps._Row(4000, 1, 4000, birth, 1.0, markers={marker}),   # the launcher
        4001: ps._Row(4001, 4000, 4001, birth, 1.0, markers={marker}),  # the server
        4002: ps._Row(4002, 4001, 4002, birth, 1.0, markers={marker}),  # a worker
    }
    signalled = []
    monkeypatch.setattr(ps, "census", lambda markers=(): dict(rows))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        int(pid): ps._Identity(ps.IDENTITY_ALIVE, "Sat Oct 10 20:10:23 2026", int(pid)) for pid in pids if int(pid) in rows})
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: signalled.append((pid, graceful)) or _pl.SIGNAL_SENT)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    outcome = ps.panic_scope_census(tmp_path, exclude={4001, 4000, 1})
    assert signalled == [(4002, False)], "KILL by identity, only the worker"
    assert outcome["ended"] == [4002] and outcome["settled"] is True


def test_a_stalled_census_leaves_a_durable_pending_record_that_the_launcher_completes(tmp_path, monkeypatch):
    marker = ps.installation_marker(tmp_path)
    member = _sleeper({marker: "1"})
    release = threading.Event()
    real_census = ps.census

    def stalled(markers=()):
        release.wait(10)
        return None  # the table never came back: unknown, never clean

    monkeypatch.setattr(ps, "census", stalled)
    results = []
    thread = threading.Thread(target=lambda: results.append(ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})),
                              daemon=True)
    try:
        thread.start()
        deadline = time.monotonic() + 5
        while not ps.panic_sweep_pending_path(tmp_path).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ps.panic_sweep_pending_path(tmp_path).exists(), "the record is written BEFORE the census"
        assert thread.is_alive() and member.poll() is None
        release.set()
        thread.join(timeout=5)
        assert results and results[0]["census_failed"] is True
        assert ps.panic_sweep_pending_path(tmp_path).exists(), "an unfinished census keeps its obligation"
        monkeypatch.setattr(ps, "census", real_census)
        completed = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1}, installation=True)
        assert completed is not None and completed["ended"] == [member.pid] and _gone(member.pid)
        assert not ps.panic_sweep_pending_path(tmp_path).exists()
        assert ps.complete_pending_panic_sweep(tmp_path) is None
    finally:
        release.set()
        _end(member)


class _ExitCalled(RuntimeError):
    pass


def _panic(monkeypatch, tmp_path, *, census):
    """Neutralize the other owners; keep the census as the one real request under test."""
    from ouroboros import server_control
    from ouroboros.startup_historical_audit import audit

    monkeypatch.setattr(audit, "stop", lambda: None)
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda **kw: [])
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **kw: [])
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **kw: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda **kw: [])
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **kw: None)
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon",
                        lambda **kw: None if kw.get("create") is False else SimpleNamespace(stop_outcome=lambda: "nothing_to_stop"))
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _root: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr(ps, "panic_scope_census", census)
    monkeypatch.setattr(server_control.os, "_exit", lambda code: (_ for _ in ()).throw(_ExitCalled(code)))
    diagnostics = []

    def report(message, *args):
        diagnostics.append({"requests": copy.deepcopy(args[0]), "settlements": copy.deepcopy(args[1])})
        logging.getLogger(__name__).critical(message, *args)

    started = time.monotonic()
    with pytest.raises(_ExitCalled) as exit_info:
        server_control.execute_panic_stop(consciousness=None, kill_workers_fn=lambda **kw: None,
                                          data_dir=tmp_path, panic_exit_code=120, log=SimpleNamespace(critical=report))
    assert exit_info.value.args == (120,)
    return diagnostics[0], time.monotonic() - started


def test_panic_runs_the_census_as_one_more_request_after_the_owner_requests(tmp_path, monkeypatch):
    calls = []

    def census(data_dir, **kw):
        calls.append(pathlib.Path(data_dir))
        return {"requested": True, "ended": [7], "survivors": [], "census_failed": False}

    diagnostics, _elapsed = _panic(monkeypatch, tmp_path, census=census)
    assert calls == [tmp_path]
    assert diagnostics["requests"]["scope-census"]["ended"] == [7]
    assert diagnostics["requests"]["daemon"] == {"requested": False, "error": "no captured daemon identity"}
    assert "scope-census" not in diagnostics["settlements"], "a request attempt, never a settlement to wait on"


def test_a_census_that_stalls_never_delays_the_panic_exit(tmp_path, monkeypatch):
    release = threading.Event()

    def census(data_dir, **kw):
        release.wait(30)
        return {"requested": True}

    try:
        diagnostics, elapsed = _panic(monkeypatch, tmp_path, census=census)
        assert diagnostics["requests"]["scope-census"] == "unfinished"
        assert elapsed < 5.0, "the request join is the existing quarter second; nothing waits on the census"
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "panic-scope-census":
                thread.join(timeout=5)


def test_a_failing_census_is_disclosed_and_the_other_requests_still_run(tmp_path, monkeypatch):
    def census(data_dir, **kw):
        raise RuntimeError("table unreadable")

    diagnostics, _elapsed = _panic(monkeypatch, tmp_path, census=census)
    assert diagnostics["requests"]["scope-census"] == {"requested": False, "error": "RuntimeError: table unreadable"}
    assert "commands" in diagnostics["requests"] and "daemon" in diagnostics["settlements"]


def test_launcher_completion_and_known_only_startup_keep_their_admission_and_exit_order():
    launcher = (REPO / "launcher.py").read_text(encoding="utf-8")
    panic_branch = launcher[launcher.index("if exit_code == PANIC_EXIT_CODE:"):]
    completion = panic_branch.index('name="panic-scope-completion"')
    assert completion < panic_branch.index('_kill_orphaned_children(reason="panic_stop")')
    assert "join(timeout=" in panic_branch[completion:completion + 400], "bounded: the app's exit never waits on a stalled census"
    assert "exclude\": {os.getpid(), os.getppid(), 1}" in panic_branch.replace("'", '"'), "the launcher and its parent are excluded"
    server = (REPO / "server.py").read_text(encoding="utf-8")
    main = server[server.index("def main() -> int:"):]
    assert (main.index("admit_data_root(DATA_DIR)") < main.index("process_scope.initialize_installation_scope(DATA_DIR)")
            < main.index("automatic_launch_allowed("))
    # Ordinary doors never sweep the installation scope: close, quit, planned and manual Restart.
    for path in ("ouroboros/server_restart.py", "ouroboros/owned_shutdown.py"):
        text = (REPO / path).read_text(encoding="utf-8")
        assert "panic_scope_census" not in text and "complete_pending_panic_sweep" not in text


def test_the_daemon_spawn_env_keeps_the_installation_marker_and_sheds_task_and_run_markers(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests.test_claudexor_owned_daemon import _point_owned_home, _UnpublishedChild

    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", sys.executable)
    monkeypatch.setattr(owned, "_SPAWN_WAIT_SEC", 0.05)
    monkeypatch.setattr(owned, "_SPAWN_POLL_SEC", 0.01)
    install, task, run = (ps.mint_marker(p) for p in (ps.INSTALL_SCOPE_PREFIX, ps.TASK_SCOPE_PREFIX, ps.RUN_SCOPE_PREFIX))
    for name in (install, task, run):
        monkeypatch.setenv(name, "1")
    captured = {}

    def _capture_spawn(command, **kwargs):
        captured["env"] = kwargs["env"]
        return _UnpublishedChild(424245)

    monkeypatch.setattr(_pl, "kill_process_tree", lambda proc: proc.terminate())
    import ouroboros.process_custody as custody_mod

    monkeypatch.setattr(custody_mod, "spawn_supervised", _capture_spawn)
    with pytest.raises(ClaudexorUnavailable):
        owned.OwnedClaudexorDaemon().ensure_running()
    env = captured["env"]
    assert env.get(install) == "1", "the installation-owned daemon stays in the installation's scope"
    assert task not in env and run not in env, "inheritance is not a lifetime transfer"


def test_the_allowlisted_service_env_and_mcp_stdio_env_keep_the_installation_marker(monkeypatch):
    from ouroboros import mcp_client
    from ouroboros.tools.services import _service_env
    from ouroboros.workspace_executor import _executor_service_env, service_env

    # The unmarked control is explicit even when pytest itself is a scoped service.
    for name in ps.scope_entries(os.environ):
        monkeypatch.delenv(name)
    install, task = ps.mint_marker(ps.INSTALL_SCOPE_PREFIX), ps.mint_marker(ps.TASK_SCOPE_PREFIX)
    captured = []
    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "StdioServerParameters", lambda **kwargs: captured.append(kwargs) or SimpleNamespace(**kwargs))
    monkeypatch.setattr(mcp_client, "stdio_client", lambda params, **kw: SimpleNamespace(params=params))
    cfg = mcp_client.normalize_server_config({"id": "local", "transport": "stdio", "command": "python3", "args": []})

    for build in (service_env, _service_env, _executor_service_env):  # nothing planted: byte-identical allowlist
        assert not ps.scope_entries(build())
    mcp_client._transport_factory(cfg)
    assert "env" not in captured[-1], "no marker planted: the SDK default environment, as before"

    monkeypatch.setenv(install, "1")
    monkeypatch.setenv(task, "1")
    for build in (service_env, _service_env, _executor_service_env):
        env = build()
        assert env.get(install) == "1" and task not in env, "a service keeps the installation marker only"
    mcp_client._transport_factory(cfg)
    assert captured[-1]["env"] == {install: "1"}, "a stdio server rides in the installation's scope"


def test_markers_are_names_only_and_never_logged_or_persisted_with_environment_bytes(tmp_path, caplog):
    for module in ("process_scope.py", "process_census.py"):
        source = (REPO / "ouroboros" / module).read_text(encoding="utf-8")
        assert not re.search(r"log\.\w+\([^)]*(?:rest|words|entries|environ)", source), f"no environment bytes in logs: {module}"
    marker = ps.installation_marker(tmp_path)
    member = _sleeper({marker: "1"})
    try:
        with caplog.at_level(logging.DEBUG):
            outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert outcome["ended"] == [member.pid]
        assert marker not in caplog.text
    finally:
        _end(member)


def test_the_imported_asgi_lifespan_initializes_the_installation_scope_after_admission_before_children():
    """Architecture §1 names the imported-app route (`server:app` under an ASGI runner):
    its lifespan admits the data root itself, so it plants the marker and completes a
    pending sweep there too, before the supervisor spawns anything."""
    import ast

    tree = ast.parse((REPO / "server.py").read_text(encoding="utf-8"))
    lifespan = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan")
    calls = sorted((node.lineno, node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
                   for node in ast.walk(lifespan) if isinstance(node, ast.Call))
    line = {name: lineno for lineno, name in reversed(calls)}  # first occurrence wins
    assert line["admit_data_root"] < line["initialize_installation_scope"] < line["_start_supervisor_if_needed"]


def test_initialize_installation_scope_is_idempotent_without_resuming_old_broad_discovery(tmp_path):
    marker = ps.installation_marker(tmp_path)
    leftover = _sleeper({marker: "1"})
    try:
        ps.panic_sweep_pending_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        ps.panic_sweep_pending_path(tmp_path).write_text('{"requested_at": "x", "excluded": []}', encoding="utf-8")
        env: dict = {}
        planted = ps.initialize_installation_scope(tmp_path)
        assert planted == marker and os.environ.get(marker) == "1"
        assert leftover.poll() is None, "no saved identity: an old Panic cannot enrol this process at startup"
        assert not ps.panic_sweep_pending_path(tmp_path).exists()
        row = json.loads((tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()[-1])
        assert row["scope"] == "known_members" and "not a full cleanup" in row["residual"]
        assert ps.initialize_installation_scope(tmp_path) == marker, "idempotent"
        assert ps.plant_installation_scope(tmp_path, env) == marker and env == {marker: "1"}
    finally:
        _end(leftover)
        os.environ.pop(marker, None)


def test_a_survivor_keeps_the_pending_record_until_a_completion_settles_it(tmp_path, monkeypatch):
    marker = ps.installation_marker(tmp_path)
    member = _sleeper({marker: "1"})
    real_deliver = _pl.deliver_signal
    try:
        monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: _pl.SIGNAL_SENT)  # accepted, nothing dies
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert [row["pid"] for row in outcome["survivors"]] == [member.pid]
        assert outcome["settled"] is False and outcome["pending_retired"] is False and outcome["recorded"] is True
        pending = __import__("json").loads(ps.panic_sweep_pending_path(tmp_path).read_text(encoding="utf-8"))
        assert [row["pid"] for row in pending["last_outcome"]["survivors"]] == [member.pid] and pending["attempts"] == 1
        journal = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
        assert journal.count('"panic_scope_census"') == 1 and marker not in journal
        monkeypatch.setattr(_pl, "deliver_signal", real_deliver)
        completed = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1})
        assert completed["ended"] == [member.pid] and completed["settled"] is True and completed["pending_retired"] is True
        assert _gone(member.pid) and not ps.panic_sweep_pending_path(tmp_path).exists()
    finally:
        _end(member)


def test_a_journal_append_that_returns_false_keeps_the_pending_record_until_a_completion_is_recorded(
        tmp_path, monkeypatch):
    """Astra (whole review, coupling): ``append_jsonl`` reports a failed write by returning
    ``False``, not by raising. A settled Panic outcome whose journal row never landed is no
    receipt: the pending record stays, and only a later completion whose row lands retires
    it. A directory occupies the journal's path, so every real append attempt fails.
    Synthetic pid, mocked signals."""
    marker = ps.installation_marker(tmp_path)
    birth, pid = "Sat Oct 10 20:10:23 2026", 4242
    rows = {pid: ps._Row(pid, 1, pid, birth, ps.birth_epoch(birth), markers={marker})}
    monkeypatch.setattr(ps, "census", lambda markers=(): dict(rows))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        int(p): ps._Identity(ps.IDENTITY_ALIVE, birth, int(p)) for p in pids if int(p) in rows})

    def kill(p, *, graceful):
        rows.pop(p, None)  # the member dies of the signal
        return _pl.SIGNAL_SENT

    monkeypatch.setattr(_pl, "deliver_signal", kill)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    journal, pending = tmp_path / "logs" / "supervisor.jsonl", ps.panic_sweep_pending_path(tmp_path)
    journal.mkdir(parents=True)
    outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
    assert outcome["ended"] == [pid] and outcome["settled"] is True
    assert outcome["recorded"] is False and outcome["pending_retired"] is False
    assert pending.exists() and json.loads(pending.read_text(encoding="utf-8"))["attempts"] == 1, \
        "an unrecorded outcome keeps the obligation"
    journal.rmdir()  # the journal is writable again
    completed = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1})
    assert completed["settled"] is True and completed["recorded"] is True and completed["pending_retired"] is True
    assert not pending.exists()
    assert [(row["type"], row["phase"]) for row in map(json.loads, journal.read_text(encoding="utf-8").splitlines())] \
        == [("panic_scope_census", "completion")], "the retiring completion is the recorded one"


@pytest.mark.parametrize("failure", ["raises", "not_true"])
def test_a_journal_append_that_raises_or_answers_anything_but_true_keeps_the_pending_record(
        tmp_path, monkeypatch, failure):
    """Only a real ``True`` is a receipt; an exception (or any other answer) keeps the record
    for the next completion, which retires it once its own row lands."""
    ps.installation_marker(tmp_path)
    monkeypatch.setattr(ps, "census", lambda markers=(): {})  # a readable, empty table: settled

    def unrecorded(*_args, **_kwargs):
        if failure == "raises":
            raise OSError("journal unavailable")
        return 1  # truthy, yet not the writer's True

    real_append = ps.append_jsonl
    monkeypatch.setattr(ps, "append_jsonl", unrecorded)
    outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
    pending = ps.panic_sweep_pending_path(tmp_path)
    assert outcome["settled"] is True and outcome["recorded"] is False and pending.exists()
    monkeypatch.setattr(ps, "append_jsonl", real_append)
    completed = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1})
    assert completed["recorded"] is True and completed["pending_retired"] is True and not pending.exists()


def test_completion_reads_the_saved_exclusions_and_adds_the_current_controller_and_its_parent(tmp_path, monkeypatch):
    marker = ps.installation_marker(tmp_path)
    birth = "Sat Oct 10 20:10:23 2026"
    me, parent = os.getpid(), os.getppid()
    rows = {pid: ps._Row(pid, 1, pid, birth, 1.0, markers={marker}) for pid in (4000, 4002, me, parent)}
    ps.panic_sweep_pending_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    ps.panic_sweep_pending_path(tmp_path).write_text(
        __import__("json").dumps({"requested_at": "x", "excluded": [4000]}), encoding="utf-8")  # the old server saved 4000
    signalled = []
    monkeypatch.setattr(ps, "census", lambda markers=(): dict(rows))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        int(pid): ps._Identity(ps.IDENTITY_ALIVE, birth, int(pid)) for pid in pids if int(pid) in rows})
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: signalled.append(pid) or _pl.SIGNAL_SENT)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    outcome = ps.complete_pending_panic_sweep(tmp_path, installation=True)  # launcher completion, no explicit exclusions
    assert signalled == [4002], "the saved exclusion, this controller and its parent are never signalled"
    assert outcome["ended"] == [4002] and outcome["pending_retired"] is True


def test_the_engines_run_scope_index_members_are_ended_by_identity_and_its_state_is_disclosed(tmp_path):
    ps.installation_marker(tmp_path)
    run_marker = ps.mint_marker(ps.RUN_SCOPE_PREFIX)
    captured = _sleeper(argv=["/bin/sleep", "120"])          # a captured harness descendant (env invisible)
    group_leader = _sleeper(argv=["/bin/sleep", "120"])      # the harness root's own group
    marked = _sleeper({run_marker: "1"})                     # a run member the census can see by marker
    stranger = _sleeper({ps.mint_marker(ps.RUN_SCOPE_PREFIX): "1"})  # a run of another engine: not in this index
    index = ps.engine_run_scopes_path(tmp_path)
    index.parent.mkdir(parents=True, exist_ok=True)
    try:
        facts = ps.read_identities([captured.pid, group_leader.pid])
        index.write_text(__import__("json").dumps({"schema_version": 1, "entries": [{
            "run_id": "run-1", "marker": run_marker, "state": "running", "scope_start": facts[group_leader.pid].birth,
            "root_groups": [{"pgid": group_leader.pid, "leader_pid": group_leader.pid, "leader_birth": facts[group_leader.pid].birth}],
            "captured": [{"pid": captured.pid, "pgid": captured.pid, "birth": facts[captured.pid].birth}],
        }]}), encoding="utf-8")
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert outcome["engine_index"]["status"] == "read" and outcome["engine_index"]["entries"] == 1
        assert set(outcome["ended"]) == {captured.pid, group_leader.pid, marked.pid}
        assert stranger.poll() is None and outcome["settled"] is True
        assert ps.PANIC_CENSUS_RESIDUAL in outcome["residual"]
    finally:
        _end(captured, group_leader, marked, stranger)


@pytest.mark.parametrize("body,status,settles", [
    (None, "absent", True),
    ("{not json", "unreadable", False),
    ('{"schema_version": 2, "entries": []}', "unsupported_schema", False),
    ('[]', "unreadable", False),
    ('{"schema_version": 1}', "invalid", False),
    ('{"schema_version": 1, "entries": "broken"}', "invalid", False),
    ('{"schema_version": 1, "entries": []}', "read", True),
])
def test_the_engine_index_is_disclosed_never_read_as_empty_and_only_absent_or_valid_lets_a_pass_settle(
        tmp_path, body, status, settles):
    ps.installation_marker(tmp_path)
    index = ps.engine_run_scopes_path(tmp_path)
    index.parent.mkdir(parents=True, exist_ok=True)
    if body is not None:
        index.write_text(body, encoding="utf-8")
    entries, disclosure = ps.engine_scope_index(tmp_path)
    assert entries == [] and disclosure["status"] == status
    outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
    assert outcome["engine_index"]["status"] == status, "the Panic facts carry the index's state"
    assert outcome["settled"] is settles and outcome["pending_retired"] is settles
    assert ps.panic_sweep_pending_path(tmp_path).exists() is not settles, "an unknown index keeps the obligation"


def test_an_unreadable_engine_index_keeps_the_pending_record_and_the_completion_reaches_the_recorded_member(tmp_path):
    """Astra delta 1: the index decides completion. Its read fails on the Panic pass while
    it names a platform-binary descendant the census cannot see by marker; the obligation
    stays, and once the index reads, the completion ends that member."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root reads a mode-0 file")
    ps.installation_marker(tmp_path)
    captured = _sleeper(argv=["/bin/sleep", "120"])
    index = ps.engine_run_scopes_path(tmp_path)
    index.parent.mkdir(parents=True, exist_ok=True)
    try:
        birth = ps.birth_text(captured.pid)
        index.write_text(json.dumps({"schema_version": 1, "entries": [{
            "run_id": "r", "marker": ps.mint_marker(ps.RUN_SCOPE_PREFIX), "state": "running", "scope_start": birth,
            "root_groups": [], "captured": [{"pid": captured.pid, "pgid": captured.pid, "birth": birth}]}]}), encoding="utf-8")
        index.chmod(0)
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert outcome["engine_index"]["status"] == "unreadable" and outcome["ended"] == [] and captured.poll() is None
        assert outcome["settled"] is False and outcome["pending_retired"] is False
        assert ps.panic_sweep_pending_path(tmp_path).exists(), "retryable: the record stays"
        index.chmod(0o600)
        completed = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1}, installation=True)
        assert completed["engine_index"]["status"] == "read" and completed["ended"] == [captured.pid] and _gone(captured.pid)
        assert completed["settled"] is True and completed["pending_retired"] is True
        assert not ps.panic_sweep_pending_path(tmp_path).exists()
    finally:
        try:
            index.chmod(0o600)
        except OSError:
            pass
        _end(captured)


def test_a_malformed_index_row_is_disclosed_the_valid_rows_are_acted_on_and_the_pass_stays_pending(tmp_path):
    ps.installation_marker(tmp_path)
    member = _sleeper(argv=["/bin/sleep", "120"])
    index = ps.engine_run_scopes_path(tmp_path)
    index.parent.mkdir(parents=True, exist_ok=True)
    try:
        birth = ps.birth_text(member.pid)
        good = {"run_id": "good", "marker": ps.mint_marker(ps.RUN_SCOPE_PREFIX), "state": "running", "scope_start": birth,
                "root_groups": [{"pgid": member.pid, "leader_pid": member.pid, "leader_birth": birth,
                                 "leader_identity": {"ignored": True}}], "captured": []}
        bad = {"run_id": "bad", "marker": ps.mint_marker(ps.RUN_SCOPE_PREFIX), "state": "running", "scope_start": birth,
               "root_groups": [], "captured": [{"pid": 4242, "pgid": "not-a-number", "birth": birth}]}
        index.write_text(json.dumps({"schema_version": 1, "entries": [bad, good, "junk"]}), encoding="utf-8")
        entries, disclosure = ps.engine_scope_index(tmp_path)  # never raises
        assert [entry["run_id"] for entry in entries] == ["good"] and disclosure["status"] == "invalid"
        assert [(row["index"], row["run_id"], row["reason"]) for row in disclosure["malformed"]] == [
            (0, "bad", "captured.pgid malformed"), (2, "", "entry is not an object")]
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert outcome["ended"] == [member.pid] and _gone(member.pid), "the valid row is acted on"
        assert outcome["settled"] is False and ps.panic_sweep_pending_path(tmp_path).exists(), "invalid cannot complete"
    finally:
        _end(member)


def test_a_null_engine_leader_birth_never_vouches_for_a_live_leader_while_the_floor_still_names_the_orphans(tmp_path):
    """Astra delta2 2 / Fable delta2 NIT 1: for an engine root group whose ``leader_birth``
    is null, the floor (``scope_start``) bounds the members' age and is never the leader's
    identity: a live ``pid == pgid`` cannot be verified by the host, so the group is skipped
    (the engine's contract) even when that leader was born in the floor's second; with no
    live leader, the members at or after the floor belong to the scope."""
    ps.installation_marker(tmp_path)
    leader = _sleeper(argv=["/bin/sleep", "120"])  # a live root born in the very second the floor names
    shell = subprocess.Popen(["/bin/sh", "-c", "/bin/sleep 120 >/dev/null 2>&1 & echo $!"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True, start_new_session=True)
    orphan = int(shell.stdout.read().strip().splitlines()[-1])
    index = ps.engine_run_scopes_path(tmp_path)
    index.parent.mkdir(parents=True, exist_ok=True)
    try:
        leader_birth, shell_birth = ps.birth_text(leader.pid), ps.birth_text(shell.pid)
        assert leader_birth and shell_birth
        shell.wait(timeout=5)  # reaped: the group it led keeps its & child and no live pid == pgid

        def entry(run_id, pgid, floor):
            return {"run_id": run_id, "marker": ps.mint_marker(ps.RUN_SCOPE_PREFIX), "state": "running",
                    "scope_start": floor, "root_groups": [{"pgid": pgid, "leader_pid": pgid, "leader_birth": None}],
                    "captured": []}

        index.write_text(json.dumps({"schema_version": 1, "entries": [
            entry("live-leader", leader.pid, leader_birth), entry("leaderless", shell.pid, shell_birth)]}), encoding="utf-8")
        groups = ps._engine_scope_targets(ps.engine_scope_index(tmp_path)[0])[1]
        assert [(group["leader_birth"], group["floor"]) for group in groups] == [("", leader_birth), ("", shell_birth)], \
            "null stays null; the floor rides separately"
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert leader.poll() is None and leader.pid not in outcome["ended"] and leader.pid in outcome["groups_reused"], \
            "a live pid == pgid the host cannot verify: skipped and disclosed, never signalled on the floor"
        assert outcome["ended"] == [orphan] and _gone(orphan), "the leaderless group's child at or after the floor is the scope's"
    finally:
        _end(leader)
        try:
            os.kill(orphan, 9)
        except ProcessLookupError:
            pass


def test_a_null_scope_start_falls_back_to_the_entrys_opened_at_as_the_members_age_floor(tmp_path):
    """Astra delta3 2: schema 1 allows ``scope_start: null``; the engine's contract then names
    the ISO ``opened_at`` as the floor. For two leaderless groups (reaped shells, ``&``
    children) with ``leader_birth`` and ``scope_start`` null: a member born before the
    entry opened is spared and disclosed; one born at or after the floor is the scope's and
    is ended. Both ISO spellings the engine may write (``Z``, ``+00:00``) convert."""
    import datetime as dt

    ps.installation_marker(tmp_path)

    def leaderless():
        shell = subprocess.Popen(["/bin/sh", "-c", "/bin/sleep 120 >/dev/null 2>&1 & echo $!"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True, start_new_session=True)
        child = int(shell.stdout.read().strip().splitlines()[-1])
        birth = ps.birth_text(shell.pid)
        assert birth
        shell.wait(timeout=5)  # reaped: no live pid == pgid remains
        return shell.pid, child, ps.birth_epoch(birth)

    older_group, older, older_epoch = leaderless()
    member_group, member, member_epoch = leaderless()
    def iso(epoch):
        return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(timespec="milliseconds")
    index = ps.engine_run_scopes_path(tmp_path)
    index.parent.mkdir(parents=True, exist_ok=True)
    try:
        def entry(run_id, pgid, opened_at):
            return {"run_id": run_id, "marker": ps.mint_marker(ps.RUN_SCOPE_PREFIX), "state": "running",
                    "opened_at": opened_at, "scope_start": None,
                    "root_groups": [{"pgid": pgid, "leader_pid": pgid, "leader_birth": None}], "captured": []}

        opened_late = iso(older_epoch + 3600).replace("+00:00", "Z")  # the entry opened an hour after that child was born
        opened_at_birth = iso(member_epoch)                            # opened in the second the shell was born
        index.write_text(json.dumps({"schema_version": 1, "entries": [
            entry("older", older_group, opened_late), entry("member", member_group, opened_at_birth)]}), encoding="utf-8")
        groups = ps._engine_scope_targets(ps.engine_scope_index(tmp_path)[0])[1]
        assert [(group["leader_birth"], group["floor"]) for group in groups] == [
            ("", ps.lstart_text(older_epoch + 3600)), ("", ps.lstart_text(member_epoch))], \
            "leader_birth stays empty; opened_at becomes the lstart-form floor"
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert outcome["groups_reused"] == [older_group] and older not in outcome["ended"] and ps.measure_birth(older), \
            "born before the entry opened: not the scope's, skipped and disclosed"
        assert outcome["ended"] == [member] and _gone(member), "at or after the opened_at floor: ended"
        assert outcome["settled"] is True
    finally:
        for pid in (older, member):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


def test_a_known_member_rides_the_pending_record_into_the_completion(tmp_path, monkeypatch):
    """Astra delta 2 (Panic): a member whose identity could not be read at Panic is carried by
    pid and birth; the completion reaches it once its marker is no longer visible."""
    marker = ps.installation_marker(tmp_path)
    member = _sleeper({marker: "1"})
    try:
        birth = ps.measure_birth(member.pid)
        real_read = ps.read_identities
        monkeypatch.setattr(ps, "read_identities", lambda pids: {
            **real_read(pids), **({member.pid: ps._Identity(ps.IDENTITY_UNREADABLE)} if member.pid in set(pids) else {})})
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert outcome["unverified"] == [{"pid": member.pid, "birth": birth, "reason": ps.UNVERIFIED_IDENTITY}]
        assert outcome["pending_retired"] is False
        pending = json.loads(ps.panic_sweep_pending_path(tmp_path).read_text(encoding="utf-8"))
        assert pending["known_members"] == [{"pid": member.pid, "birth": birth}]
        monkeypatch.setattr(ps, "read_identities", real_read)
        real_census = ps.census

        def hidden(markers=()):  # it exec'd a platform binary: same pid and birth, no visible marker
            rows = real_census(markers)
            if rows and member.pid in rows:
                row = rows[member.pid]
                rows[member.pid] = ps._Row(row.pid, row.ppid, row.pgid, row.birth, row.epoch, row.state, set())
            return rows

        monkeypatch.setattr(ps, "census", hidden)
        completed = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1})
        assert completed["ended"] == [member.pid] and _gone(member.pid) and completed["pending_retired"] is True
    finally:
        _end(member)


def test_known_identities_survive_a_failed_census_on_the_pending_record(tmp_path, monkeypatch):
    """Astra delta2 1 (Panic pending record): Panic saves a hidden member it could not verify;
    a completion that cannot read the table keeps the saved identity; the next completion
    reads and ends it. Synthetic pid, mocked signals."""
    marker = ps.installation_marker(tmp_path)
    birth, pid = "Sat Oct 10 20:10:23 2026", 4242
    signals = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda p, *, graceful: signals.append(p) or _pl.SIGNAL_SENT)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    def row(markers):
        return {pid: ps._Row(pid, 1, pid, birth, ps.birth_epoch(birth), markers=set(markers))}
    monkeypatch.setattr(ps, "census", lambda markers=(): row([marker]))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {int(p): ps._Identity(ps.IDENTITY_UNREADABLE) for p in pids})
    first = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
    pending = ps.panic_sweep_pending_path(tmp_path)
    assert first["pending_retired"] is False
    assert json.loads(pending.read_text(encoding="utf-8"))["known_members"] == [{"pid": pid, "birth": birth}]
    monkeypatch.setattr(ps, "census", lambda markers=(): None)  # the launcher's broad completion meets a stalled table
    failed = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1}, installation=True)
    assert failed["census_failed"] is True and failed["pending_retired"] is False
    assert [item["reason"] for item in failed["unverified"]] == [ps.UNVERIFIED_CENSUS]
    assert json.loads(pending.read_text(encoding="utf-8"))["known_members"] == [{"pid": pid, "birth": birth}], \
        "a failed census keeps the known identity"
    monkeypatch.setattr(ps, "census", lambda markers=(): row([]))  # no marker left to see
    monkeypatch.setattr(ps, "read_identities", lambda pids: {int(p): ps._Identity(ps.IDENTITY_ALIVE, birth, int(p)) for p in pids})
    recovered = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1})
    assert signals == [pid] and recovered["ended"] == [pid] and recovered["pending_retired"] is True
    assert not pending.exists()


def test_panic_reads_the_known_members_task_cleanups_still_owe_and_carries_them_until_resolved(tmp_path, monkeypatch):
    """Astra delta2 3: a hidden escaped child a task cleanup saved by pid and birth is Panic's
    to end as well; while it stays unresolved, Panic's pending record carries it instead of
    retiring, and the completion ends it once its identity reads."""
    from ouroboros.task_results import write_task_result

    ps.installation_marker(tmp_path)
    marker = ps.task_scope_marker(tmp_path, "t-owed")
    child = _sleeper({marker: "1"})  # an escaped child (no recorded group) the task's cleanup could not verify
    try:
        birth = ps.measure_birth(child.pid)
        monkeypatch.setattr(ps, "note_task_terminal_transition", lambda *a, **k: None)
        write_task_result(tmp_path, "t-owed", "completed", result="done")
        real_read = ps.read_identities
        monkeypatch.setattr(ps, "read_identities", lambda pids: {
            **real_read(pids), **({child.pid: ps._Identity(ps.IDENTITY_UNREADABLE)} if child.pid in set(pids) else {})})
        assert ps.settle_task_scope(tmp_path, "t-owed")["settled"] is False
        assert ps.task_scope_record(tmp_path, "t-owed")["known_members"] == [{"pid": child.pid, "birth": birth}]
        real_census = ps.census

        def hidden(markers=()):  # it exec'd a platform binary: no marker, same pid and birth
            rows = real_census(markers)
            if rows and child.pid in rows:
                row = rows[child.pid]
                rows[child.pid] = ps._Row(row.pid, row.ppid, row.pgid, row.birth, row.epoch, row.state, set())
            return rows

        monkeypatch.setattr(ps, "census", hidden)
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert outcome["task_known"] == 1 and outcome["pending_retired"] is False and child.poll() is None
        assert [item["pid"] for item in outcome["unverified"]] == [child.pid], "owed and still unverified: Panic does not retire"
        pending = json.loads(ps.panic_sweep_pending_path(tmp_path).read_text(encoding="utf-8"))
        assert pending["known_members"] == [{"pid": child.pid, "birth": birth}]
        monkeypatch.setattr(ps, "read_identities", real_read)
        completed = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1})
        assert completed["ended"] == [child.pid] and _gone(child.pid) and completed["pending_retired"] is True
    finally:
        _end(child)


@pytest.mark.parametrize("failure", ["unreadable", "signal_failed"])
def test_a_stale_known_birth_confirmed_gone_never_retires_the_record_owing_the_current_birth(tmp_path, monkeypatch, failure):
    """Astra delta3 1 (records): a task cleanup saved (P, A) unverified; that child exited and
    a new installation member reused P with birth B. Panic confirms A gone and cannot end B
    (identity unreadable, or its signal failed): B's evidence stays, the pending record is
    kept with B, and the completion ends B by identity before retiring. Synthetic pids,
    mocked signals."""
    from ouroboros.task_results import write_task_result

    install = ps.installation_marker(tmp_path)
    old, new, pid = "Sun Oct 11 12:00:00 2026", "Mon Oct 12 12:00:00 2026", 4242
    def row(birth, marker):
        return {pid: ps._Row(pid, 1, pid, birth, ps.birth_epoch(birth), markers={marker})}
    signals = []
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    monkeypatch.setattr(ps, "_RESCAN_SETTLE_SEC", 0)
    monkeypatch.setattr(ps, "note_task_terminal_transition", lambda *a, **k: None)
    task_marker = ps.task_scope_marker(tmp_path, "t-stale")
    write_task_result(tmp_path, "t-stale", "completed", result="done")
    monkeypatch.setattr(ps, "census", lambda markers=(): row(old, task_marker))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {int(p): ps._Identity(ps.IDENTITY_UNREADABLE) for p in pids})
    assert ps.settle_task_scope(tmp_path, "t-stale")["settled"] is False
    assert ps.task_scope_record(tmp_path, "t-stale")["known_members"] == [{"pid": pid, "birth": old}]
    monkeypatch.setattr(ps, "census", lambda markers=(): row(new, install))  # A exited; B reused its pid
    if failure == "unreadable":
        result, reason = _pl.SIGNAL_SENT, ps.UNVERIFIED_IDENTITY
    else:
        monkeypatch.setattr(ps, "read_identities", lambda pids: {int(p): ps._Identity(ps.IDENTITY_ALIVE, new, int(p)) for p in pids})
        result, reason = _pl.SIGNAL_FAILED, ps.UNVERIFIED_FAILED
    monkeypatch.setattr(_pl, "deliver_signal", lambda p, *, graceful: signals.append(p) or result)
    panic = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
    pending = ps.panic_sweep_pending_path(tmp_path)
    assert panic["task_known"] == 1 and panic["captured_gone"] == [pid] and panic["ended"] == []
    assert panic["unverified"] == [{"pid": pid, "birth": new, "reason": reason}], "B's own evidence survives A's resolution"
    assert panic["settled"] is False and panic["pending_retired"] is False and pending.exists()
    assert json.loads(pending.read_text(encoding="utf-8"))["known_members"] == [{"pid": pid, "birth": new}]
    monkeypatch.setattr(ps, "read_identities", lambda pids: {int(p): ps._Identity(ps.IDENTITY_ALIVE, new, int(p)) for p in pids})
    monkeypatch.setattr(_pl, "deliver_signal", lambda p, *, graceful: signals.append(p) or _pl.SIGNAL_SENT)
    completed = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1})
    assert completed["ended"] == [pid] and completed["unverified"] == [] and completed["pending_retired"] is True
    assert not pending.exists() and signals[-1] == pid


def test_a_denied_signal_is_disclosed_as_final_and_the_panic_record_retires(tmp_path, monkeypatch):
    marker = ps.installation_marker(tmp_path)
    member = _sleeper({marker: "1"})
    try:
        monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: _pl.SIGNAL_DENIED)
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert [item["reason"] for item in outcome["unverified"]] == [ps.UNVERIFIED_DENIED] and outcome["ended"] == []
        assert outcome["settled"] is True and outcome["pending_retired"] is True and member.poll() is None
        assert not ps.panic_sweep_pending_path(tmp_path).exists(), "no later pass of this installation can change EPERM"
    finally:
        _end(member)


def test_a_newer_schema_entry_is_never_acted_on(tmp_path):
    ps.installation_marker(tmp_path)
    member = _sleeper()
    try:
        birth = ps.birth_text(member.pid)
        index = ps.engine_run_scopes_path(tmp_path)
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(__import__("json").dumps({"schema_version": 2, "entries": [{
            "run_id": "r", "captured": [{"pid": member.pid, "pgid": member.pid, "birth": birth}]}]}), encoding="utf-8")
        outcome = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
        assert member.poll() is None and outcome["ended"] == [] and outcome["engine_index"]["status"] == "unsupported_schema"
    finally:
        _end(member)
