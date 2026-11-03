"""A task's own commands belong to the task (owner 2A, issue #1772, Directive 1).

Whatever a command leaves running lives until the task's accepted truly-terminal
transition, then is ended by scope: the task marker on the launch env (never the worker's
``os.environ``) plus the command's own recorded process group, which is what reaches a
platform binary whose environment macOS hides. A pause, a retained hold, a handoff or a
reopenable placeholder is not terminal; a keep service is handed to the installation and
survives the task end but not Panic; concurrent turns are isolated; the sweep executes an
ended task's obligation on positive durable evidence only. Real children belong to this
test process and are ended by exact pid.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import platform_layer as _pl
from ouroboros import process_scope as ps
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.shell_process import _active_subprocesses, _tracked_subprocess_run

pytestmark = [pytest.mark.serial, pytest.mark.skipif(os.name == "nt", reason="the environment census is POSIX-only")]


@contextmanager
def _bound(root, task_id):
    """The context-bound invocation binding a tool launch runs under (``tool_custody.invocation_binding``).

    A real invocation of a task without a durable row (``task_lifecycle_bound=False``): its external
    duty stays local here; the durable claim path is pinned in ``tests/test_invocation_ownership.py``."""
    from ouroboros.owner_pause import _TOOL_OPERATION
    from ouroboros.tool_custody import invocation_outcome

    source = SimpleNamespace(task_id=task_id, root_task_id=task_id, drive_root=root, task_metadata={},
                             task_lifecycle_bound=False)
    token = _TOOL_OPERATION.set((source, "run_command", invocation_outcome(source, "run_command", f"op-{task_id}")))
    try:
        yield source
    finally:
        _TOOL_OPERATION.reset(token)


def _run(root, task_id, script, timeout=10):
    with _bound(root, task_id):
        return _tracked_subprocess_run(["/bin/sh", "-c", script], cwd=str(root), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, timeout=timeout)


def _leftover(root, task_id, launch="/bin/sleep 120"):
    """A command whose background child outlives the call; returns the child's exact pid."""
    res = _run(root, task_id, f"nohup {launch} >/dev/null 2>&1 & echo $!")
    pid = int(res.stdout.strip().splitlines()[-1])
    assert _alive(pid), "the background child outlives its command"
    assert not any(proc.pid == pid for proc in list(_active_subprocesses)), "nothing tracks it in memory"
    return pid


def _alive(pid) -> bool:
    return bool(ps.measure_birth(pid))


def _gone(pid, timeout=8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _end_pids(*pids) -> None:
    for pid in pids:
        try:
            os.kill(int(pid), 9)
        except (ProcessLookupError, PermissionError):
            pass


@pytest.fixture(autouse=True)
def _own_cleanup_threads():
    yield
    ps.wait_for_cleanups(15)


@pytest.mark.parametrize("task_id", ["t-end", "manual"])
def test_a_commands_background_child_lives_until_the_task_ends_then_is_ended_by_its_group(tmp_path, task_id):
    inherited = ps.scope_entries(os.environ, prefix=ps.TASK_SCOPE_PREFIX)
    pid = _leftover(tmp_path, task_id)
    try:
        record = ps.task_scope_record(tmp_path, task_id)
        assert record["marker"].startswith(ps.TASK_SCOPE_PREFIX)
        assert [group["pgid"] for group in record["groups"]] == [os.getpgid(pid)], "the command's own group is recorded"
        assert ps.scope_entries(os.environ, prefix=ps.TASK_SCOPE_PREFIX) == inherited, "launch leaves the worker's task markers unchanged"
        write_task_result(tmp_path, task_id, "running", result="working")
        ps.wait_for_cleanups()
        assert _alive(pid), "a running row is not an end"
        write_task_result(tmp_path, task_id, "completed", result="done")
        ps.wait_for_cleanups()
        assert _gone(pid), "the accepted terminal transition ends the task's leftovers"
        facts = load_task_result(tmp_path, task_id)["process_scope"]
        assert pid in facts["ended"] and facts["survivors"] == [] and facts["coverage"] == ps.COVERAGE_POSIX
        assert facts["reason"] == "terminal_transition" and facts["marker_role"] == "task"
        assert ps.task_scope_record(tmp_path, task_id) is None, "the obligation is discharged"
    finally:
        _end_pids(pid)


@pytest.mark.skipif(not _pl.IS_MACOS, reason="native macOS recorded-group cleanup")
def test_the_platform_binary_leftover_is_ended_even_when_its_marker_is_hidden(tmp_path, monkeypatch):
    from dataclasses import replace

    pid = _leftover(tmp_path, "t-sip")
    try:
        record = ps.task_scope_record(tmp_path, "t-sip")
        marker = record["marker"]
        census = ps.census
        def hidden_marker(markers=()):
            rows = census(markers)
            if rows is not None and pid in rows:
                rows[pid] = replace(rows[pid], markers=set())
            return rows
        monkeypatch.setattr(ps, "census", hidden_marker)  # only this child's environment observation
        rows = ps.census([marker])
        assert marker not in rows[pid].markers
        assert pid not in ps.scope_members(rows, marker=marker)["members"]
        assert pid in ps.scope_members(rows, marker=marker, groups=record["groups"])["members"]
        write_task_result(tmp_path, "t-sip", "completed", result="done")
        ps.wait_for_cleanups()
        assert _gone(pid), "its recorded group is the source that reaches it"
    finally:
        _end_pids(pid)


def test_a_non_platform_child_carries_the_task_marker_and_a_setsid_escape_is_still_ended(tmp_path):
    node = _pl.resolve_bundled_node()
    if not node:
        pytest.skip("no bundled node for the marker-visible child")
    # The runaway leaves the command's group (setsid): only the marker names it.
    script = (f"import subprocess; p=subprocess.Popen([{node!r},'-e','setTimeout(()=>{{}},120000)'],"
              "start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print(p.pid)")
    res = _run(tmp_path, "t-marker", f"{sys.executable} -c {subprocess.list2cmdline([script])}")
    pid = int(res.stdout.strip().splitlines()[-1])
    try:
        marker = ps.task_scope_record(tmp_path, "t-marker")["marker"]
        rows = ps.census([marker])
        assert marker in (rows[pid].markers or set()), "a non-platform child's marker is visible"
        assert rows[pid].pgid == pid, "it left the command's group"
        write_task_result(tmp_path, "t-marker", "completed", result="done")
        ps.wait_for_cleanups()
        assert _gone(pid)
        assert pid in load_task_result(tmp_path, "t-marker")["process_scope"]["ended"]
    finally:
        _end_pids(pid)


def test_pause_retention_and_a_reopenable_placeholder_are_not_terminal(tmp_path):
    from ouroboros.task_results import reopen_reconciled_presence_placeholder

    pid = _leftover(tmp_path, "t-hold")
    try:
        for status in ("running", "interrupted", "running"):
            write_task_result(tmp_path, "t-hold", status, result="held")
            ps.wait_for_cleanups()
            assert _alive(pid), f"{status} keeps the task's processes"
        assert ps.settle_task_scopes(tmp_path, set()) == [], "the sweep needs positive terminal evidence"
        assert _alive(pid)
        # The orphan reconciler's failed mark on a lost presence turn: a placeholder its
        # re-run reopens, never an end.
        write_task_result(tmp_path, "t-hold", "failed", reason_code="orphaned_running_after_worker_restart",
                          status_reconciled_from="running", metadata={"source": "presence"}, result="lost")
        ps.wait_for_cleanups()
        assert _alive(pid) and ps.task_scope_record(tmp_path, "t-hold") is not None
        assert ps.settle_task_scopes(tmp_path, set()) == [] and _alive(pid)
        assert reopen_reconciled_presence_placeholder(tmp_path, "t-hold")
        write_task_result(tmp_path, "t-hold", "completed", result="done")
        ps.wait_for_cleanups()
        assert _gone(pid), "the deliberate end ends it"
    finally:
        _end_pids(pid)


def test_concurrent_turns_are_isolated(tmp_path):
    first, second = _leftover(tmp_path, "t-one"), _leftover(tmp_path, "t-two")
    try:
        one, two = ps.task_scope_record(tmp_path, "t-one"), ps.task_scope_record(tmp_path, "t-two")
        assert one["marker"] != two["marker"]
        write_task_result(tmp_path, "t-one", "completed", result="done")
        ps.wait_for_cleanups()
        assert _gone(first) and _alive(second), "ending one turn leaves the other's child alone"
        write_task_result(tmp_path, "t-two", "cancelled", result="stopped")
        ps.wait_for_cleanups()
        assert _gone(second)
    finally:
        _end_pids(first, second)


def test_the_sweep_ends_a_dead_tasks_members_and_spares_live_paused_and_unknown_owners(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "note_task_terminal_transition", lambda *a, **k: None)  # the worker died at its write
    dead, live, paused, unknown = (_leftover(tmp_path, tid) for tid in ("t-dead", "t-live", "t-paused", "t-unknown"))
    try:
        write_task_result(tmp_path, "t-dead", "failed", result="worker crashed after the write")
        write_task_result(tmp_path, "t-live", "completed", result="post-task cognition still running")
        write_task_result(tmp_path, "t-paused", "running", result="paused")
        settled = ps.settle_task_scopes(tmp_path, lambda: {"t-live"})
        assert [row["task_id"] for row in settled] == ["t-dead"] and dead in settled[0]["ended"]
        assert _gone(dead)
        assert _alive(live) and _alive(paused) and _alive(unknown), "live, paused and unknown owners keep theirs"
        assert ps.task_scope_record(tmp_path, "t-dead") is None
        assert all(ps.task_scope_record(tmp_path, tid) for tid in ("t-live", "t-paused", "t-unknown"))
        settled = ps.settle_task_scopes(tmp_path, set())  # the live owner ended: its obligation is next
        assert [row["task_id"] for row in settled] == ["t-live"] and _gone(live)
    finally:
        _end_pids(dead, live, paused, unknown)


def test_the_server_sweeps_outstanding_obligations_at_startup(tmp_path, monkeypatch):
    from ouroboros import post_task_checkpoint, server_maintenance as sm
    from supervisor import active_activity, queue, workers

    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(post_task_checkpoint, "POST_TASK_SYNTHESIS_INFLIGHT", {})
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", active_activity.DirectActivityRegistry())
    monkeypatch.setattr(ps, "note_task_terminal_transition", lambda *a, **k: None)
    pid = _leftover(tmp_path, "t-boot")
    try:
        write_task_result(tmp_path, "t-boot", "completed", result="done before the crash")
        sm._startup_custody_sweep()
        assert _gone(pid) and ps.task_scope_record(tmp_path, "t-boot") is None
    finally:
        _end_pids(pid)


@pytest.mark.parametrize("task_id", ["t-keep", "manual"])
def test_a_keep_service_sheds_the_task_marker_survives_the_task_end_and_dies_on_panic(tmp_path, monkeypatch, task_id):
    from ouroboros import config as cfg
    from ouroboros.tools.registry import ToolRegistry

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    install = ps.installation_marker(tmp_path)
    monkeypatch.setenv(install, "1")  # the server planted it; services inherit it through the allowlist
    repo, drive = tmp_path / "repo", tmp_path
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = task_id
    cmd = [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(120)"]
    kept = __import__("json").loads(registry.execute("start_service", {
        "name": "kept", "cmd": cmd, "keep_alive": True, "readiness": {"log_contains": "READY", "timeout_sec": 5}}))
    owned = __import__("json").loads(registry.execute("start_service", {
        "name": "owned", "cmd": cmd, "readiness": {"log_contains": "READY", "timeout_sec": 5}}))
    try:
        task_marker = ps.task_scope_record(drive, task_id)["marker"]
        rows = ps.census([install, task_marker])
        assert rows[kept["pid"]].markers == {install}, "a keep service keeps the installation marker only"
        assert rows[owned["pid"]].markers == {install, task_marker}, "a task service carries both"
        assert [group["pgid"] for group in ps.task_scope_record(drive, task_id)["groups"]] == [owned["pid"]]
        write_task_result(drive, task_id, "completed", result="done")
        ps.wait_for_cleanups()
        assert _gone(owned["pid"]) and _alive(kept["pid"]), "the keep service survives the task end"
        outcome = ps.panic_scope_census(drive, exclude={os.getpid(), 1})
        assert kept["pid"] in outcome["ended"] and _gone(kept["pid"]), "Panic ends it by the installation marker"
    finally:
        _end_pids(kept["pid"], owned["pid"])
        from ouroboros.tools import services

        services._SERVICES.clear()


@pytest.mark.parametrize("task_id", ["t-exec", "manual"])
def test_an_executor_foreground_command_carries_the_marker_and_records_its_group(tmp_path, task_id):
    from ouroboros.workspace_executor import ExecutorRef, _execute_local

    executor = ExecutorRef(kind="local", executor_id="local", network="host", mappings=())
    with _bound(tmp_path, task_id):
        result = _execute_local(
            executor, [sys.executable, "-c", "import os; print(sorted(k for k in os.environ if k.startswith('OURO_TASK_SCOPE_')))"],
            tmp_path, 20, drive_root=tmp_path)
    record = ps.task_scope_record(tmp_path, task_id)
    assert result.returncode == 0 and record["marker"] in result.stdout
    assert len(record["groups"]) == 1 and record["groups"][0]["leader_birth"]


def test_cleanup_threads_are_named_and_bounded(tmp_path):
    pid = _leftover(tmp_path, "t-thread")
    try:
        before = {thread.name for thread in threading.enumerate()}
        write_task_result(tmp_path, "t-thread", "completed", result="done")
        ps.wait_for_cleanups()
        assert _gone(pid)
        assert not any(name.startswith("task-scope-cleanup") and name not in before for name in
                       {thread.name for thread in threading.enumerate() if thread.is_alive()})
    finally:
        _end_pids(pid)


def test_an_unverified_member_keeps_the_record_and_the_sweep_settles_it_later_writing_the_result_once(tmp_path, monkeypatch):
    """Astra 1-2: a failed signal is unverified, never ended; the record stays with its
    group evidence for the sweep; an unchanged outcome is recorded once, not on every
    sweep; the retirement waits for a settled, recorded outcome."""
    from ouroboros import task_store

    pid = _leftover(tmp_path, "t-denied")
    real_deliver = _pl.deliver_signal
    try:
        monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: _pl.SIGNAL_FAILED)
        write_task_result(tmp_path, "t-denied", "completed", result="done")
        ps.wait_for_cleanups()
        assert _alive(pid), "a failed signal ends nothing and claims nothing"
        record = ps.task_scope_record(tmp_path, "t-denied")
        assert record["cleanup"]["state"] == "pending" and record["cleanup"]["reason"] == "unverified"
        assert [group["pgid"] for group in record["groups"]] == [os.getpgid(pid)], "the group evidence stays"
        facts = load_task_result(tmp_path, "t-denied")["process_scope"]
        assert facts["settled"] is False and facts["unverified"][0]["pid"] == pid and facts["ended"] == []
        writes = []
        real_patch = task_store.patch_document
        monkeypatch.setattr(task_store, "patch_document", lambda *a, **k: writes.append(a[1]) or real_patch(*a, **k))
        ps.settle_task_scopes(tmp_path, set())
        ps.settle_task_scopes(tmp_path, set())
        assert writes == [], "an unchanged outcome is not rewritten on every sweep"
        assert ps.task_scope_record(tmp_path, "t-denied")["cleanup"]["attempts"] == 3
        monkeypatch.setattr(_pl, "deliver_signal", real_deliver)
        settled = ps.settle_task_scopes(tmp_path, set())
        assert [row["task_id"] for row in settled] == ["t-denied"] and _gone(pid)
        assert ps.task_scope_record(tmp_path, "t-denied") is None, "retired only once settled and recorded"
        assert load_task_result(tmp_path, "t-denied")["process_scope"]["settled"] is True
        assert load_task_result(tmp_path, "t-denied")["status"] == "completed", "no status writer touched the row"
    finally:
        _end_pids(pid)


# --------------------------------------------------------------------------- fix round 2

def test_a_denied_signal_is_final_disclosed_on_the_result_and_retires_the_record(tmp_path, monkeypatch):
    """Fable delta NIT 2: EPERM (a root child, ``sudo -n … &``) cannot be ended by any later
    pass of this installation: disclosed with pid, birth and reason, the obligation retired."""
    pid = _leftover(tmp_path, "t-root")
    try:
        monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: _pl.SIGNAL_DENIED)
        write_task_result(tmp_path, "t-root", "completed", result="done")
        ps.wait_for_cleanups()
        facts = load_task_result(tmp_path, "t-root")["process_scope"]
        assert facts["unverified"] == [{"pid": pid, "birth": ps.measure_birth(pid), "reason": ps.UNVERIFIED_DENIED}]
        assert facts["settled"] is True and facts["ended"] == [] and _alive(pid)
        assert ps.task_scope_record(tmp_path, "t-root") is None, "nothing a later pass could change"
    finally:
        _end_pids(pid)


def test_a_denied_member_beside_an_unverified_one_keeps_the_record_with_both_known(tmp_path, monkeypatch):
    denied, failed = _leftover(tmp_path, "t-mixed"), _leftover(tmp_path, "t-mixed")
    try:
        monkeypatch.setattr(_pl, "deliver_signal",
                            lambda pid, *, graceful: _pl.SIGNAL_DENIED if pid == denied else _pl.SIGNAL_FAILED)
        write_task_result(tmp_path, "t-mixed", "completed", result="done")
        ps.wait_for_cleanups()
        record = ps.task_scope_record(tmp_path, "t-mixed")
        assert record["cleanup"]["reason"] == "unverified"
        assert {item["pid"] for item in record["known_members"]} == {denied, failed}, "both ride into the next pass"
        facts = load_task_result(tmp_path, "t-mixed")["process_scope"]
        assert facts["settled"] is False
        assert {item["reason"] for item in facts["unverified"]} == {ps.UNVERIFIED_DENIED, ps.UNVERIFIED_FAILED}
    finally:
        _end_pids(denied, failed)


def _marker_child(root, task_id) -> subprocess.Popen:
    """A marker-visible child outside any recorded group (a setsid escape): first the
    marker, later only its known identity, names it."""
    marker = ps.task_scope_marker(root, task_id)
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], env={**os.environ, marker: "1"},
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def _unreadable_for(monkeypatch, pid):
    real = ps.read_identities
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        **real(pids), **({pid: ps._Identity(ps.IDENTITY_UNREADABLE)} if pid in set(pids) else {})})
    return real


def _hide_marker(monkeypatch, pid, *, birth=None):
    """The census row of ``pid`` as after an exec into a platform binary (no visible marker),
    or, with ``birth``, as a replacement process that took the pid."""
    real = ps.census

    def hidden(markers=()):
        rows = real(markers)
        if rows and pid in rows:
            row = rows[pid]
            rows[pid] = ps._Row(row.pid, row.ppid, row.pgid, birth or row.birth, row.epoch, row.state, set())
        return rows

    monkeypatch.setattr(ps, "census", hidden)


def test_a_known_member_is_carried_into_the_next_pass_and_ended_once_its_identity_reads(tmp_path, monkeypatch):
    """Astra delta 2: a member observed but left unverified (its identity read failed) is an
    input of the next pass by pid and birth; losing marker visibility in between (an exec
    into a platform binary) never discharges it; continued unreadability keeps the record."""
    proc = _marker_child(tmp_path, "t-known")
    try:
        birth = ps.measure_birth(proc.pid)
        real_read = _unreadable_for(monkeypatch, proc.pid)
        write_task_result(tmp_path, "t-known", "completed", result="done")
        ps.wait_for_cleanups()
        record = ps.task_scope_record(tmp_path, "t-known")
        assert record["known_members"] == [{"pid": proc.pid, "birth": birth}] and record["cleanup"]["reason"] == "unverified"
        _hide_marker(monkeypatch, proc.pid)  # it exec'd a platform binary: no marker, same pid and birth
        assert ps.settle_task_scopes(tmp_path, set()) and _alive(proc.pid), "still unreadable: still owed"
        record = ps.task_scope_record(tmp_path, "t-known")
        assert record is not None and record["known_members"] == [{"pid": proc.pid, "birth": birth}]
        monkeypatch.setattr(ps, "read_identities", real_read)
        settled = ps.settle_task_scopes(tmp_path, set())
        assert settled and proc.pid in settled[0]["ended"] and _gone(proc.pid), "the known identity reaches it"
        assert ps.task_scope_record(tmp_path, "t-known") is None
    finally:
        _end_pids(proc.pid)
        proc.wait(timeout=5)


def test_known_identities_survive_a_failed_census_and_the_next_readable_pass_ends_them(tmp_path, monkeypatch):
    """Astra delta2 1 (task record): pass 1 saves a hidden member it could not verify; pass 2
    cannot read the table at all — the saved identity must survive, never be replaced by that
    pass's empty list; pass 3 reads and ends it. Synthetic pid, mocked signals."""
    birth, pid = "Sat Oct 10 20:10:23 2026", 4242
    marker = ps.task_scope_marker(tmp_path, "t-table")
    monkeypatch.setattr(ps, "note_task_terminal_transition", lambda *a, **k: None)
    write_task_result(tmp_path, "t-table", "completed", result="done")
    signals = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda p, *, graceful: signals.append(p) or _pl.SIGNAL_SENT)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    row = lambda markers: {pid: ps._Row(pid, 1, pid, birth, ps.birth_epoch(birth), markers=set(markers))}
    monkeypatch.setattr(ps, "census", lambda markers=(): row([marker]))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {int(p): ps._Identity(ps.IDENTITY_UNREADABLE) for p in pids})
    first = ps.settle_task_scope(tmp_path, "t-table")
    assert first["settled"] is False and ps.task_scope_record(tmp_path, "t-table")["known_members"] == [{"pid": pid, "birth": birth}]
    monkeypatch.setattr(ps, "census", lambda markers=(): None)  # a ps timeout is evidence of nothing
    failed = ps.settle_task_scope(tmp_path, "t-table")
    record = ps.task_scope_record(tmp_path, "t-table")
    assert failed["census_failed"] is True and failed["settled"] is False and record["cleanup"]["reason"] == "census_failed"
    assert record["known_members"] == [{"pid": pid, "birth": birth}], "a failed census keeps the known identity"
    assert [item["reason"] for item in failed["unverified"]] == [ps.UNVERIFIED_CENSUS]
    monkeypatch.setattr(ps, "census", lambda markers=(): row([]))  # it exec'd a platform binary: no marker left
    monkeypatch.setattr(ps, "read_identities", lambda pids: {int(p): ps._Identity(ps.IDENTITY_ALIVE, birth, int(p)) for p in pids})
    recovered = ps.settle_task_scope(tmp_path, "t-table")
    assert signals == [pid] and recovered["ended"] == [pid] and recovered["settled"] is True, "the saved identity reaches it"
    assert ps.task_scope_record(tmp_path, "t-table") is None


def test_a_known_member_whose_birth_changed_is_spared_and_discharged(tmp_path, monkeypatch):
    proc = _marker_child(tmp_path, "t-reborn")
    try:
        birth = ps.measure_birth(proc.pid)
        real_read = _unreadable_for(monkeypatch, proc.pid)
        write_task_result(tmp_path, "t-reborn", "completed", result="done")
        ps.wait_for_cleanups()
        assert ps.task_scope_record(tmp_path, "t-reborn")["known_members"] == [{"pid": proc.pid, "birth": birth}]
        monkeypatch.setattr(ps, "read_identities", real_read)
        _hide_marker(monkeypatch, proc.pid, birth="Mon Jan  1 00:00:00 2001")  # a replacement process took the pid
        signalled = []
        monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: signalled.append(pid) or _pl.SIGNAL_SENT)
        settled = ps.settle_task_scopes(tmp_path, set())
        assert settled and settled[0]["captured_gone"] == [proc.pid] and settled[0]["settled"] is True
        assert signalled == [] and _alive(proc.pid), "a replacement process on the same pid is never signalled"
        assert ps.task_scope_record(tmp_path, "t-reborn") is None, "discharged: ours is gone"
    finally:
        _end_pids(proc.pid)
        proc.wait(timeout=5)
