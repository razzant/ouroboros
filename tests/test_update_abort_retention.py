"""An update that stays in this process must return its stopped saved work.

Exercise the update doors, retention, queue persistence, pool startup/readiness,
and assignment themselves. Only process creation/termination and thread scheduling
are simulated; a fake process emits the same readiness observation as its OS peer.
"""
from __future__ import annotations

import json
import queue as stdqueue
import time
from types import SimpleNamespace

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.task_results import load_task_result, write_task_result
from supervisor.events_budget import budget_hold_fact
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_restart_retention import _pool_events, _queued
from tests.test_restart_saved_work import _working_run

pytestmark = pytest.mark.serial


class _Process:
    def __init__(self, pid):
        self.pid, self.alive, self.exitcode = pid, True, None

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.alive, self.exitcode = False, -15


class _InlineThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def _system(tmp_path, monkeypatch):
    from ouroboros.gateway import control
    from supervisor import git_ops, worker_pool_lifecycle, worker_process
    from ouroboros.utils import append_jsonl

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "_git_dir", lambda: repo / ".git")
    monkeypatch.setattr(workers, "REPO_DIR", repo)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    monkeypatch.setattr(workers, "MAX_WORKERS", 1)
    monkeypatch.setattr(workers, "mp", SimpleNamespace(get_context=lambda _method: SimpleNamespace(Queue=stdqueue.Queue)))
    monkeypatch.setattr(workers, "threading", SimpleNamespace(Thread=_InlineThread))
    processes, launches = {}, []

    def process(wid, commands):
        proc = _Process(700000 + len(processes))
        processes[proc.pid] = proc
        return proc

    def spawn(_ctx, wid, commands, _events, _repo, root):
        proc = process(wid, commands)
        launches.append(commands)
        append_jsonl(root / "logs" / "events.jsonl", {
            "type": "worker_ready", "worker_id": wid, "pid": proc.pid, "git_sha": "test-sha"})
        return proc

    for module in (workers, worker_pool_lifecycle):
        monkeypatch.setattr(module, "kill_worker_tree", lambda pid, **_kw: processes[pid].terminate())
    monkeypatch.setattr(worker_process, "spawn_worker_process", spawn)

    def saved(task_id="saved"):
        admitted = queue.enqueue_task(_queued(task_id, root_task_id=task_id))
        assert not admitted.get("_admission_blocked"), admitted
        queue.PENDING.remove(admitted)
        _working_run(tmp_path, workers, task_id)
        workers.RUNNING[task_id]["task"] = dict(admitted)
        wid = workers.RUNNING[task_id]["worker_id"]
        commands = stdqueue.Queue()
        workers.WORKERS[wid] = workers.Worker(wid, process(wid, commands), commands, busy_task_id=task_id)

    return SimpleNamespace(queue=queue, state=state, workers=workers, control=control,
                           processes=processes, launches=launches, saved=saved)


def _assisted(system, monkeypatch):
    """An already-restored checkout still owns its real durable rollback marker."""
    from supervisor import git_ops, update_merge

    tx = {"task_id": "resolver", "phase": "assisted_resolution", "pre_update_sha": "a" * 40,
          "pre_update_branch": "ouroboros", "base_sha": "b" * 40, "target_sha": "c" * 40,
          "stash_restored": True}
    metadata = {"managed_update": {"authority_fingerprint": update_merge.assisted_authority_fingerprint(tx)}}
    update_merge.write_update_tx(tx)
    system.workers.close_repo_writer_admission(update_merge.assisted_writer_gate_reason(tx))

    def git_capture(command, **_kw):
        if command == ["git", "rev-parse", "--verify", "HEAD"]:
            return 0, tx["pre_update_sha"], ""
        if command == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, tx["pre_update_branch"], ""
        raise AssertionError(f"unexpected Git call: {command!r}")

    monkeypatch.setattr(git_ops, "git_capture", git_capture)
    return update_merge, tx, metadata


def _assert_returned(system, root, task_id="saved"):
    assert task_id not in system.workers.RUNNING, "aborted update stranded retained RUNNING"
    [task] = [row for row in system.workers.PENDING if row["id"] == task_id]
    assert task["_attempt"] == 2
    assert budget_hold_fact(task) is None
    ctx, _ = _loop_ctx(root, task_id, attempt=2)
    frozen = wc.load_recovery(ctx, task["_working_recovery"])
    assert frozen["working"]["boundary"] == "pre_effect"
    return task


def _scope_saved(system, root):
    from ouroboros.projects_registry import create_project, project_scope_admission

    folder = root / "prepared-project"
    folder.mkdir()
    create_project(root, "project", working_dir=str(folder))
    task = system.workers.RUNNING["saved"]["task"]
    task.update(project_id="project", workspace_root=str(folder),
                _project_admission=project_scope_admission(root, project_id="project"))
    write_task_result(root, "saved", "running", project_id="project", workspace_root=str(folder))
    return task


def test_gateway_abort_returns_saved_work_and_assigns_once(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    system.saved()
    assert system.control._quiesce_repo_writers("auto_merge") == []
    assert system.workers.RUNNING["saved"]["retained_for_boot"] is True

    system.control._respawn_workers_after_failed_update()

    task = _assert_returned(system, tmp_path)
    handoff = dict(task["_working_recovery"])
    system.control._respawn_workers_after_failed_update()
    assert _assert_returned(system, tmp_path)["_working_recovery"] == handoff
    assert len(system.launches) == 1
    system.workers.assign_tasks()
    [commands] = system.launches
    sent = commands.get_nowait()
    assert sent["id"] == "saved" and sent["_attempt"] == 2
    assert sent["_working_recovery"] == handoff
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert commands.empty() and not system.workers.PENDING
    assert system.workers.RUNNING["saved"]["attempt"] == 2


@pytest.mark.parametrize("door", ["watchdog", "resolver_release", "early_rollback"])
def test_assisted_abort_doors_return_saved_work(tmp_path, monkeypatch, door):
    system = _system(tmp_path, monkeypatch)
    system.saved()
    assert system.control._quiesce_repo_writers("assisted") == []
    update_merge, _tx, metadata = _assisted(system, monkeypatch)
    # The assisted resolver's startup is allowed during an ACTIVE update. It
    # must not consume the saved tasks or release their project leases early.
    assert system.workers.ensure_worker_pool_started(n=1, allow_disabled_restart=True)
    system.workers.assign_tasks()
    assert system.workers.RUNNING["saved"]["retained_for_boot"] is True
    assert not system.workers.PENDING and system.launches[0].empty()
    if door == "watchdog":
        result = update_merge.abort_orphaned_assisted_tx("resolver", metadata)
        assert result["rolled_back"], result
    elif door == "resolver_release":
        # Worker rollback already cleared the tx; its done event releases the
        # exact assisted latch owned by its immutable metadata.
        assert update_merge.clear_update_tx()
        assert update_merge.release_assisted_writer_gate_after_task(metadata)
    else:
        ok, message = update_merge.rollback_managed_update("early_abort")
        assert ok, message
    assert not system.workers.repo_writer_admission_closed()
    _assert_returned(system, tmp_path)
    system.workers.assign_tasks()
    sent = system.launches[0].get_nowait()
    assert sent["id"] == "saved" and sent["_attempt"] == 2
    assert system.launches[0].empty()


def test_unknown_old_process_is_not_recovered_by_the_retention_marker(tmp_path, monkeypatch):
    from supervisor import worker_pool_lifecycle

    system = _system(tmp_path, monkeypatch)
    system.saved()
    for module in (system.workers, worker_pool_lifecycle):
        monkeypatch.setattr(module, "kill_worker_tree", lambda _pid, **_kw: None)
    blockers = system.control._quiesce_repo_writers("auto_merge")
    assert blockers and any(item.startswith("worker:") for item in blockers)
    assert system.workers.RUNNING["saved"]["retained_for_boot"] is True
    system.control._respawn_workers_after_failed_update()
    assert not system.launches and not system.workers.PENDING
    assert system.workers.repo_writer_admission_closed()


@pytest.mark.parametrize("source_failure", ["corrupt", "missing", "unreadable"])
def test_failed_source_conversion_releases_lease_but_remains_explicitly_held(tmp_path, monkeypatch, source_failure):
    from ouroboros.project_lease import running_project_ids

    system = _system(tmp_path, monkeypatch)
    system.saved()
    system.workers.RUNNING["saved"]["task"]["project_id"] = "saved-project"
    assert running_project_ids(system.workers.RUNNING.values()) == {"saved-project"}
    assert system.control._quiesce_repo_writers("auto_merge") == []
    # Source bytes disappear after successful retention; no clean retry may
    # replace the interrupted thought, even though its OS process is dead.
    checkpoint = wc.checkpoint_path(tmp_path, "saved", 1)
    if source_failure == "missing":
        checkpoint.unlink()
    elif source_failure == "corrupt":
        checkpoint.write_text("corrupt")
    else:
        original_read = type(checkpoint).read_bytes
        def read_bytes(path):
            if path == checkpoint:
                raise OSError("injected checkpoint disk failure")
            return original_read(path)
        monkeypatch.setattr(type(checkpoint), "read_bytes", read_bytes)
    system.control._respawn_workers_after_failed_update()
    assert "saved" not in system.workers.RUNNING
    [held] = system.workers.PENDING
    assert held["id"] == "saved" and budget_hold_fact(held)
    assert not running_project_ids(system.workers.RUNNING.values())
    system.workers.assign_tasks()
    assert all(commands.empty() for commands in system.launches)
    before = json.dumps(held, sort_keys=True)
    system.control._respawn_workers_after_failed_update()
    assert json.dumps(system.workers.PENDING[0], sort_keys=True) == before


@pytest.mark.parametrize("when", ["before_quiesce", "after_quiesce"])
def test_owner_stop_is_never_returned_by_abort(tmp_path, monkeypatch, when):
    from ouroboros.cancel_intents import request_cancel

    system = _system(tmp_path, monkeypatch)
    system.saved()
    if when == "before_quiesce":
        request_cancel(tmp_path, "saved", reason="owner stop", source="owner", requested_by="owner")
    assert system.control._quiesce_repo_writers("auto_merge") == []
    if when == "after_quiesce":
        request_cancel(tmp_path, "saved", reason="owner stop", source="owner", requested_by="owner")
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert all(commands.empty() for commands in system.launches)
    assert "saved" not in system.workers.RUNNING
    assert load_task_result(tmp_path, "saved")["status"] in {"cancelled", "interrupted"}


@pytest.mark.parametrize("when", ["before_quiesce", "after_quiesce"])
def test_owner_pause_is_never_released_by_abort(tmp_path, monkeypatch, when):
    from supervisor.owner_pause_control import request_owner_pause

    system = _system(tmp_path, monkeypatch)
    system.saved()
    if when == "before_quiesce":
        assert request_owner_pause("saved", request_id="pause-press")["ok"]
    assert system.control._quiesce_repo_writers("auto_merge") == []
    if when == "after_quiesce":
        assert request_owner_pause("saved", request_id="pause-press")["ok"]
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert all(commands.empty() for commands in system.launches)
    assert "saved" not in system.workers.RUNNING
    assert any(row["id"] == "saved" for row in system.workers.PENDING)


def test_failed_snapshot_never_opens_writers_and_retry_buys_one_successor(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    system.saved()
    assert system.control._quiesce_repo_writers("auto_merge") == []
    original_write = system.queue.atomic_write_text

    def fail_write(path, *args, **kwargs):
        if path == system.queue.QUEUE_SNAPSHOT_PATH:
            raise OSError("injected snapshot disk failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(system.queue, "atomic_write_text", fail_write)
    system.control._respawn_workers_after_failed_update()
    assert system.workers.repo_writer_admission_closed()
    assert not system.launches
    monkeypatch.setattr(system.queue, "atomic_write_text", original_write)
    system.control._respawn_workers_after_failed_update()
    task = _assert_returned(system, tmp_path)
    assert task["_attempt"] == 2 and len(system.launches) == 1


@pytest.mark.parametrize("replacement", ["different_attempt", "different_object", "foreign_id"])
def test_abort_only_returns_the_captured_attempt(tmp_path, monkeypatch, replacement):
    system = _system(tmp_path, monkeypatch)
    system.saved()
    assert system.control._quiesce_repo_writers("auto_merge") == []
    meta = system.workers.RUNNING["saved"]
    if replacement == "different_attempt":
        meta["attempt"] = meta["task"]["_attempt"] = 9
    elif replacement == "different_object":
        system.workers.RUNNING["saved"] = {**meta, "task": dict(meta["task"])}
    else:
        system.saved("foreign")
        system.workers.RUNNING["foreign"]["retained_for_boot"] = True
    system.control._respawn_workers_after_failed_update()
    assert system.workers.repo_writer_admission_closed()
    if replacement == "foreign_id":
        assert "foreign" in system.workers.RUNNING
        assert not any(row["id"] == "foreign" for row in system.workers.PENDING)
    else:
        assert "saved" in system.workers.RUNNING
        assert not system.workers.PENDING


def test_active_assisted_transaction_refuses_gateway_abort_recovery(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    system.saved()
    assert system.control._quiesce_repo_writers("assisted") == []
    update_merge, _tx, metadata = _assisted(system, monkeypatch)
    system.control._respawn_workers_after_failed_update()
    assert system.workers.RUNNING["saved"]["retained_for_boot"] is True
    assert not system.workers.PENDING and not system.launches
    assert system.workers.repo_writer_admission_closed()
    assert not update_merge.release_assisted_writer_gate_after_task(metadata)
    assert update_merge.abort_orphaned_assisted_tx("resolver", {})["acted"] is False


def test_the_same_update_can_still_return_saved_work_on_real_restart_restore(tmp_path, monkeypatch):
    from ouroboros import delegate_recovery
    from tests.test_restart_saved_work import _ack

    system = _system(tmp_path, monkeypatch)
    system.saved()
    assert system.control._quiesce_repo_writers("auto_merge") == []
    active = json.loads(delegate_recovery._active_restart_transaction_path(tmp_path).read_text())
    _ack(tmp_path, active["transaction_id"])
    system.workers.RUNNING.clear()
    assert system.queue.restore_pending_from_snapshot() == 1
    _assert_returned(system, tmp_path)
    assert system.workers.ensure_worker_pool_started(allow_disabled_restart=True)
    # Process-local admission disappears on an actual application restart.
    system.workers.open_repo_writer_admission()
    system.workers.assign_tasks()
    assert system.launches[0].get_nowait()["id"] == "saved"


def test_assisted_failed_snapshot_can_be_retried_by_the_real_release_hook(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    system.saved()
    assert system.control._quiesce_repo_writers("assisted") == []
    update_merge, _tx, metadata = _assisted(system, monkeypatch)
    original_write = system.queue.atomic_write_text

    def fail_write(path, *args, **kwargs):
        if path == system.queue.QUEUE_SNAPSHOT_PATH:
            raise OSError("injected snapshot disk failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(system.queue, "atomic_write_text", fail_write)
    update_merge.abort_orphaned_assisted_tx("resolver", metadata)
    assert not update_merge.active_update_tx(), "rollback cleared its real durable marker"
    assert system.workers.repo_writer_admission_closed()
    system.workers.assign_tasks()
    assert all(commands.empty() for commands in system.launches)
    monkeypatch.setattr(system.queue, "atomic_write_text", original_write)
    assert update_merge.release_assisted_writer_gate_after_task(metadata)
    _assert_returned(system, tmp_path)
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert len(system.launches) == 1
    assert system.launches[0].get_nowait()["id"] == "saved"
    assert system.launches[0].empty()


@pytest.mark.parametrize("observation", ["alive", "unreadable"])
def test_abort_rechecks_the_captured_process_after_successful_quiescence(tmp_path, monkeypatch, observation):
    system = _system(tmp_path, monkeypatch)
    system.saved()
    [old] = system.processes.values()
    assert system.control._quiesce_repo_writers("auto_merge") == []
    if observation == "alive":
        old.alive = True
    else:
        def unreadable():
            raise OSError("process observation unavailable")
        monkeypatch.setattr(old, "is_alive", unreadable)
    system.control._respawn_workers_after_failed_update()
    assert system.workers.repo_writer_admission_closed()
    assert not system.launches and not system.workers.PENDING
    assert system.workers.RUNNING["saved"]["retained_for_boot"] is True


def test_active_assisted_pool_dispatches_only_its_authorized_resolver(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    system.saved()
    assert system.control._quiesce_repo_writers("assisted") == []
    _update_merge, _tx, metadata = _assisted(system, monkeypatch)
    for task_id in ("ordinary", "resolver"):
        write_task_result(tmp_path, task_id, "scheduled", chat_id=0)
        system.workers.PENDING.append(_queued(task_id, metadata=metadata if task_id == "resolver" else {}))
    assert system.workers.ensure_worker_pool_started(n=1, allow_disabled_restart=True)
    system.workers.assign_tasks()
    assert system.launches[0].get_nowait()["id"] == "resolver"
    assert set(system.workers.RUNNING) == {"saved", "resolver"}
    assert system.workers.RUNNING["saved"]["retained_for_boot"] is True
    assert [row["id"] for row in system.workers.PENDING] == ["ordinary"]


def test_abort_preserves_an_existing_queue_hold_and_releases_the_project_lease(tmp_path, monkeypatch):
    from ouroboros.project_lease import running_project_ids
    from supervisor.events_budget import HOLD_OWNER_RESTART, hold_budget_row

    system = _system(tmp_path, monkeypatch)
    system.saved()
    _scope_saved(system, tmp_path)
    held = _queued("held", project_id="project")
    write_task_result(tmp_path, "held", "scheduled", chat_id=0)
    hold_budget_row(held, reason=HOLD_OWNER_RESTART, result_root=tmp_path)
    hold = dict(budget_hold_fact(held))
    system.workers.PENDING.append(held)
    assert system.control._quiesce_repo_writers("auto_merge") == []
    assert running_project_ids(system.workers.RUNNING.values()) == {"project"}
    system.control._respawn_workers_after_failed_update()
    assert not running_project_ids(system.workers.RUNNING.values())
    assert budget_hold_fact(held) == hold
    system.workers.assign_tasks()
    assert system.launches[0].get_nowait()["id"] == "saved"
    assert running_project_ids(system.workers.RUNNING.values()) == {"project"}
    assert system.workers.PENDING == [held] and budget_hold_fact(held) == hold


def test_a_deadline_elapsed_during_update_never_dispatches_a_successor(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    system = _system(tmp_path, monkeypatch)
    system.saved()
    now = time.time()
    system.workers.RUNNING["saved"]["task"]["deadline_at"] = datetime.fromtimestamp(now + 10, timezone.utc).isoformat()
    assert system.control._quiesce_repo_writers("auto_merge") == []
    monkeypatch.setattr(time, "time", lambda: now + 20)
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert all(commands.empty() for commands in system.launches)
    assert "saved" not in system.workers.RUNNING


def test_unreadable_money_after_abort_still_refuses_assignment(tmp_path, monkeypatch):
    from ouroboros import usage_store

    system = _system(tmp_path, monkeypatch)
    system.saved()
    assert system.control._quiesce_repo_writers("auto_merge") == []
    ledger = tmp_path / usage_store.STORE_REL
    # Enqueue created the SQLite authority. Fail only its OS connection leaf;
    # no changed budget, journal replacement, gate stub or cache manipulation.
    previous = ledger.read_bytes()
    original_connect = usage_store.sqlite3.connect
    def denied(database, *args, **kwargs):
        if str(database).split("?")[0] == ledger.as_uri():
            raise usage_store.sqlite3.OperationalError("injected money-store open failure")
        return original_connect(database, *args, **kwargs)
    monkeypatch.setattr(usage_store.sqlite3, "connect", denied)
    system.control._respawn_workers_after_failed_update()
    _assert_returned(system, tmp_path)
    system.workers.assign_tasks()
    assert all(commands.empty() for commands in system.launches)
    assert "saved" not in system.workers.RUNNING
    assert ledger.read_bytes() == previous


@pytest.mark.parametrize("failure", ["missing_basis", "missing_folder", "unreadable_registry"])
def test_saved_project_recovery_revalidates_its_original_identity(tmp_path, monkeypatch, failure):
    from ouroboros import projects_registry

    system = _system(tmp_path, monkeypatch)
    system.saved()
    task = _scope_saved(system, tmp_path)
    assert system.control._quiesce_repo_writers("auto_merge") == []
    if failure == "missing_basis":
        task.pop("_project_admission")
    elif failure == "missing_folder":
        (tmp_path / "prepared-project").rmdir()
    else:
        projects_registry._registry_path(tmp_path).write_text("{unreadable")
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert all(commands.empty() for commands in system.launches)
    assert "saved" not in system.workers.RUNNING
    [held] = [row for row in system.workers.PENDING if row["id"] == "saved"]
    assert held.get("_project_admission_restore_hold") or held.get("_terminalization_retry")


def test_a_child_never_outlives_a_parent_whose_saved_source_became_unreadable(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    system.saved("parent")
    system.saved("child")
    child = system.workers.RUNNING["child"]["task"]
    child.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    write_task_result(tmp_path, "child", "running", parent_task_id="parent", root_task_id="parent")
    assert system.control._quiesce_repo_writers("auto_merge") == []
    wc.checkpoint_path(tmp_path, "parent", 1).write_text("broken")
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    assert not system.workers.RUNNING
    rows = {task["id"]: task for task in system.workers.PENDING}
    assert set(rows) == {"parent", "child"}
    assert all(budget_hold_fact(row) for row in rows.values())
    assert all(commands.empty() for commands in system.launches)


def test_abort_handoff_reaches_the_real_loop_without_replaying_unknown_effects(tmp_path, monkeypatch):
    from ouroboros import delegate_custody, loop
    from ouroboros.external_runs import observe_task_runs
    from tests.test_context_fit_v664 import _plan
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    system = _system(tmp_path, monkeypatch)
    system.saved()
    original = _registry(tmp_path, monkeypatch, "saved", 1)
    messages = _plan().messages_for("max") + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "done", "type": "function", "function": {"name": "save", "arguments": "{}"}},
            {"id": "inflight", "type": "function", "function": {"name": "save", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "done", "content": "Saved object 42"}]
    limit = SimpleNamespace(tools=original, messages=messages, llm_trace={"tool_calls": []},
                            accumulated_usage={"cost": 3.0}, round_idx=7, tool_schemas=[],
                            owner_msg_seen=set(), budget_tail="tool")
    assert wc.save_round(limit, "pre_effect")
    # A durable POST intent with no returned run id remains unknown. The loop
    # reads actual custody; neither this test nor recovery marks it terminated.
    assert delegate_custody.record_start_requested(
        tmp_path, task_id="saved", root_task_id="saved", invocation_id="unknown-invocation",
        route="test-route", request={"prompt": "external effect"})
    before = observe_task_runs(tmp_path, "saved", request_stop=False)
    assert before["runs"][0]["state"] == "stop_unknown"
    assert system.control._quiesce_repo_writers("auto_merge") == []
    system.control._respawn_workers_after_failed_update()
    system.workers.assign_tasks()
    sent = system.launches[0].get_nowait()
    assert sent["_working_recovery"]["cause"] == "update_aborted"
    assert observe_task_runs(tmp_path, "saved", request_stop=False)["runs"] == before["runs"]
    registry = _registry(tmp_path, monkeypatch, "saved", 2)
    registry._ctx.working_recovery = sent["_working_recovery"]
    seen, effects, custody_at_model = [], [], []
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *_a, **_k: pytest.fail("no paid provider call"))
    monkeypatch.setattr(registry, "execute_result", lambda *_a, **_k: effects.append(True) or pytest.fail("no tool replay"))

    class ModelReached(BaseException):
        """Stop this test at the restored prompt, before ordinary task finalization."""

    def provider_reply(call, _disposition, **_kwargs):
        seen.extend(call.messages)
        custody_at_model.extend(observe_task_runs(tmp_path, "saved", request_stop=False)["runs"])
        raise ModelReached()

    monkeypatch.setattr(loop, "_dispatch_round_model", provider_reply)
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["task_id"] = "saved"
    with pytest.raises(ModelReached):
        loop.run_llm_loop(**kwargs)
    assert not effects
    assert registry._ctx._accumulated_usage["cost"] == 3.0
    assert any(row.get("content") == "Saved object 42" for row in seen)
    unknown = next(row["content"] for row in seen if row.get("tool_call_id") == "inflight")
    assert "UNKNOWN" in unknown and "NOT re-executed" in unknown
    notice = next(str(row.get("content")) for row in seen
                  if "continued from its working checkpoint" in str(row.get("content")))
    assert "unknown-invocation: stop_unknown" in notice
    assert "update was aborted before restart" in notice
    assert custody_at_model == before["runs"]
    assert not wc.checkpoint_path(tmp_path, "saved", 1).exists()
