"""Direct restart composition with production custody/transactions/restore.

Only process/platform observations are simulated; this is not native Windows
qualification and never launches a server, daemon, or provider.
"""
from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import pytest

from ouroboros import config, server_control
from ouroboros import delegate_recovery as dr
from ouroboros import process_custody as pc
from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
from supervisor.restart_retention import fresh_return_transaction, prepare_restart_returns
from tests.test_restart_saved_work import _install_queue, _pool_events, _queued, _working_run

pytestmark = pytest.mark.serial


@pytest.fixture
def direct_restart(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    write_task_result(tmp_path, "queued", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append(_queued("queued", root_task_id="queued"))
    case = SimpleNamespace(root=tmp_path, queue=queue, workers=workers, parent=os.getpid(),
                           pid=912345, current=os.getpid(), birth="spawn-birth", command="python server.py",
                           env={}, child_env={}, calls=[], killed=[], failure="", on_spawn=None)
    proc = SimpleNamespace(pid=case.pid, wait=lambda timeout=None: -9)

    def popen(argv, **kwargs):
        case.calls.append("spawn")
        case.child_env.update(kwargs["env"])
        if case.failure == "spawn":
            raise OSError("fixture spawn failed")
        if case.on_spawn:
            case.on_spawn()
        return proc

    def execvpe(*args):
        case.calls.append("exec")
        raise OSError("fixture exec failed")

    monkeypatch.setattr(dr, "os", SimpleNamespace(environ=case.env, getpid=lambda: case.current))
    monkeypatch.setattr(server_control, "os", SimpleNamespace(environ=case.env, execvpe=execvpe, getpid=lambda: case.parent))
    monkeypatch.setattr(server_control, "sys", SimpleNamespace(executable="python", argv=["server.py"]))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(pc, "subprocess", SimpleNamespace(Popen=popen))
    monkeypatch.setattr(pc, "subprocess_new_group_kwargs", lambda: {})
    monkeypatch.setattr(pc, "process_group_id", lambda pid: pid)
    monkeypatch.setattr(pc, "process_start_time", lambda pid: case.birth)
    monkeypatch.setattr(pc, "process_start_time_legacy", lambda pid: case.birth)
    monkeypatch.setattr(pc, "process_command", lambda pid: case.command)
    monkeypatch.setattr(pc, "pid_is_alive", lambda pid: pid == case.pid)
    monkeypatch.setattr(pc, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pc, "kill_process_tree", case.killed.append)
    append = pc.append_jsonl
    monkeypatch.setattr(pc, "append_jsonl", lambda *a, **kw: False if case.failure == "custody" else append(*a, **kw))

    def prepare():
        assert prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING,
                                       transaction_id="direct-return") == {"saved-root"}
        assert dr.arm_active_planned_restart_transaction(tmp_path) == "direct-return"
        workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                             stop_source="server_shutdown", retain_saved_work=True)
        # Accepted work survives a long stop; this is the real source snapshot.
        snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
        snapshot["ts"] = "2000-01-01T00:00:00+00:00"
        queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snapshot))
        workers.PENDING.clear()
        workers.RUNNING.clear()

    def transfer(windows=True):
        monkeypatch.setattr(server_control, "IS_WINDOWS", windows)
        server_control.restart_current_process("127.0.0.1", 8765, repo_dir=tmp_path,
                                               log=logging.getLogger(__name__))

    def successor():
        case.current = case.pid
        case.env.clear()
        case.env.update(case.child_env)

    def restore():
        queue.restore_pending_from_snapshot()
        return {row["id"]: row for row in workers.PENDING}

    case.prepare, case.transfer, case.successor, case.restore = prepare, transfer, successor, restore
    return case


@pytest.mark.parametrize("windows", [True, False], ids=["windows-branch", "posix-spawn-fallback"])
def test_direct_spawn_returns_saved_work_and_runnable_queue_once(direct_restart, windows):
    case = direct_restart
    case.prepare()
    case.transfer(windows)
    case.successor()
    rows = case.restore()
    assert set(rows) == {"saved-root", "queued"}
    assert all(budget_hold_fact(row) is None for row in rows.values())
    assert rows["saved-root"]["_working_recovery"]["cause"] == "restart"
    from ouroboros.working_checkpoint import recovery_source_for_task

    saved = recovery_source_for_task(case.root, rows["saved-root"])
    assert saved["messages"][-1] == {"role": "tool", "tool_call_id": "call_a", "content": "done a"}
    tx = dr._read_restart_transaction(case.root, "direct-return")
    assert tx["supervisor_pid"] == case.parent != case.pid
    assert tx["ack_source"] == "direct_spawn_successor"
    [receipt] = [json.loads(line) for line in pc.ledger_path(case.root).read_text().splitlines()]
    assert tx["direct_spawn_successor"] == receipt
    assert receipt["pid"] == case.pid and receipt["scope"] == "daemon"
    assert tx["returns_restored_at"] and fresh_return_transaction(case.root) == {}
    assert dr.PLANNED_RESTART_TRANSACTION_ENV not in case.env
    assert case.calls == (["spawn"] if windows else ["exec", "spawn"])


@pytest.mark.parametrize("fault", ["wrong-pid", "source-pid", "recycled-pid", "wrong-command", "unknown-birth",
                                   "wrong-active", "stale-preparation", "consumed", "no-token",
                                   "wrong-source", "spawn", "custody"])
def test_direct_spawn_cannot_return_work_without_its_exact_fresh_receipt(direct_restart, fault):
    case = direct_restart
    case.prepare()
    if fault == "wrong-source":
        tx = dr._read_restart_transaction(case.root, "direct-return")
        dr._write_restart_transaction(case.root, {**tx, "supervisor_pid": case.parent + 1})
    if fault in {"spawn", "custody"}:
        case.failure = fault
        with pytest.raises((OSError, RuntimeError), match="spawn failed|durable custody"):
            case.transfer()
        assert len(case.killed) == (fault == "custody")
    else:
        case.transfer()
    case.successor()
    if fault == "wrong-pid":
        case.current += 1
    elif fault == "source-pid":
        case.current = case.parent
    elif fault == "recycled-pid":
        case.birth = "reused-pid-birth"
    elif fault == "wrong-command":
        case.command = "python unrelated.py"
    elif fault == "unknown-birth":
        case.birth = ""
    elif fault in {"wrong-active", "stale-preparation"}:
        active = dr._active_restart_transaction_path(case.root)
        row = json.loads(active.read_text())
        row["transaction_id" if fault == "wrong-active" else "prepared_at"] = "another"
        active.write_text(json.dumps(row))
    elif fault == "consumed":
        tx = dr._read_restart_transaction(case.root, "direct-return")
        dr._write_restart_transaction(case.root, {**tx, "returns_restored_at": "already-restored"})
    elif fault == "no-token":
        case.env.clear()
    rows = case.restore()
    assert set(rows) == {"saved-root", "queued"}
    assert all(budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK for row in rows.values())
    assert dr._read_restart_transaction(case.root, "direct-return")["status"] == "prepared"
    assert dr.PLANNED_RESTART_TRANSACTION_ENV not in case.env


@pytest.mark.parametrize("control", ["prior-hold", "stop", "panic"])
def test_direct_spawn_keeps_owner_controls(direct_restart, control):
    from ouroboros.cancel_intents import request_cancel
    from supervisor.events_budget import HOLD_OWNER_RESTART, hold_budget_row

    case = direct_restart
    if control == "prior-hold":
        hold_budget_row(case.workers.PENDING[0], reason=HOLD_OWNER_RESTART, result_root=case.root)
    case.prepare()
    case.transfer()
    case.successor()
    if control == "stop":
        request_cancel(case.root, "saved-root", reason="owner stop", source="owner", requested_by="owner")
    elif control == "panic":
        (case.root / "state" / "panic_stop.flag").write_text("panic")
    rows = case.restore()
    if control == "prior-hold":
        assert budget_hold_fact(rows["queued"])["reason"] == HOLD_OWNER_RESTART
        assert budget_hold_fact(rows["saved-root"]) is None
    elif control == "stop":
        assert "saved-root" not in rows
        assert budget_hold_fact(rows["queued"]) is None
    else:
        assert all(budget_hold_fact(row) is not None for row in rows.values())


def test_early_spawned_successor_waits_for_custody_publication(direct_restart, monkeypatch):
    import threading

    from ouroboros import platform_layer

    case = direct_restart
    case.prepare()
    entering, done = threading.Event(), threading.Event()
    observed = []
    parent_thread = threading.get_ident()
    monkeypatch.setattr(dr.os, "getpid", lambda: case.parent if threading.get_ident() == parent_thread else case.pid)
    acquire = platform_layer.acquire_exclusive_file_lock

    def lock(path, **kwargs):
        if threading.get_ident() != parent_thread:
            entering.set()
        return acquire(path, **kwargs)

    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", lock)

    def boot():
        try:
            observed.append(fresh_return_transaction(case.root))
        finally:
            done.set()

    child = threading.Thread(target=boot)

    def early_boot():
        child.start()
        assert entering.wait(2), "successor must take the transaction lock"
        assert not done.wait(.05), "Popen has returned neither custody nor a transaction receipt yet"

    case.on_spawn = early_boot
    try:
        case.transfer()
    finally:
        child.join(3)
    assert not child.is_alive()
    assert observed and observed[0].get("ack_source") == "direct_spawn_successor"
    case.successor()
    monkeypatch.setattr(dr.os, "getpid", lambda: case.current)
    case.env.pop(dr.PLANNED_RESTART_TRANSACTION_ENV, None)  # boot already consumed it
    assert all(budget_hold_fact(row) is None for row in case.restore().values())


def test_failed_transaction_publication_keeps_successful_spawn_unacknowledged(direct_restart, monkeypatch):
    case = direct_restart
    case.prepare()
    write = dr._write_restart_transaction

    def fail_binding(root, row):
        if "direct_spawn_successor" in row:
            raise OSError("fixture full transaction disk")
        return write(root, row)

    monkeypatch.setattr(dr, "_write_restart_transaction", fail_binding)
    case.transfer()  # the process did start, even though its return grant cannot persist
    case.successor()
    assert all(budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK for row in case.restore().values())
    assert dr._read_restart_transaction(case.root, "direct-return")["status"] == "prepared"
    assert case.killed == []


@pytest.mark.parametrize("fault", ["wrong-pid", "recycled-pid", "stale-preparation"])
def test_acknowledged_spawn_cannot_lend_its_return_to_a_later_process(direct_restart, fault):
    case = direct_restart
    case.prepare()
    case.transfer()
    case.successor()
    assert fresh_return_transaction(case.root)["ack_source"] == "direct_spawn_successor"
    if fault == "wrong-pid":
        case.current += 1
    elif fault == "recycled-pid":
        case.birth = "later-process"
    else:
        active = dr._active_restart_transaction_path(case.root)
        active.write_text(json.dumps({**json.loads(active.read_text()), "prepared_at": "later-restart"}))
    assert all(budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK for row in case.restore().values())
