"""Native owner waits survive every cleanup of one confirmed planned restart.

The OS process is inert; checkpoint storage, restart transactions, both cleanup
passes, result writes and queue restoration are the production implementations.
No provider, browser, server or daemon is started by these tests.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import queue as stdqueue
from itertools import count
import threading
import time
from types import SimpleNamespace

from tests.worker_pool_support import affirm_ready

import pytest

from ouroboros import delegate_recovery, owner_wait, server_restart
from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import atomic_write_json
from supervisor import queue, workers
from tests._store_shared import result_signature, unreadable_result


class InertProcess:
    """A handle whose termination never reaches an operating-system PID."""

    _pids = count(2147483000)

    def __init__(self):
        self.pid = next(self._pids)
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def join(self, timeout=None):
        pass


def kill_inert(pid, **kwargs):
    """The pool's tree kill for an inert handle: terminate the slot its fake PID names."""
    for slot in workers.WORKERS.values():
        if slot.proc.pid == pid:
            slot.proc.terminate()


@pytest.fixture
def restart_case(tmp_path, monkeypatch):
    from supervisor import task_lifecycle, update_merge, worker_pool_lifecycle

    monkeypatch.setattr(worker_pool_lifecycle, "kill_worker_tree", kill_inert)

    pending, running = [], {}
    # The inert pool must not inherit a prior test's manager-backed event bus.
    event_q = stdqueue.Queue()
    monkeypatch.setattr(workers, "get_event_q", lambda: event_q)
    for module in (workers, queue):
        monkeypatch.setattr(module, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", running)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    budget_fences = {}
    for module in (queue, task_lifecycle):
        monkeypatch.setattr(module, "BUDGET_ROOT_FENCES", budget_fences)
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "")
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: None)
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    requested, owner_requested = threading.Event(), threading.Event()
    requested.set()
    monkeypatch.setattr(server_restart, "_restart_requested", requested)
    monkeypatch.setattr(server_restart, "_owner_restart_requested", owner_requested)
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    monkeypatch.delenv(delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV, raising=False)

    task_id, attempt, started = "native-owner-a", 3, time.time() - 4000
    task = {"id": task_id, "type": "task", "chat_id": 1, "_attempt": attempt,
            "text": "Continue the saved draft after its owner answers.", "depth": 0}
    write_task_result(tmp_path, task_id, "running", total_rounds=7,
                      accounted_upper_bound_usd=2.5, result="The draft was saved.")
    ctx = SimpleNamespace(task_id=task_id, task_attempt=attempt, drive_root=tmp_path,
                          budget_drive_root=str(tmp_path), task_started_at=started,
                          _owner_wait_requested="quiz-a", _owner_directives=[{"text": task["text"]}],
                          active_model="fixture-model", active_effort="high",
                          active_use_local=False, active_context_mode="max")
    messages = [{"role": "assistant", "tool_calls": [{"id": "save-1"}]},
                {"role": "tool", "tool_call_id": "save-1", "content": "Saved draft object 42"}]
    trace = {"tool_calls": [{"name": "save_draft", "result": "object 42"}]}
    wait = owner_wait.checkpoint_owner_wait(ctx, messages, trace, {"cost": 2.5}, 7, [], {"owner-msg-1"})
    wait = owner_wait.set_owner_wait(tmp_path, task_id, {**wait, "state": "waiting"})
    running[task_id] = {"task": task, "worker_id": 0, "attempt": attempt,
                        "started_at": started, "last_heartbeat_at": time.time(), "owner_wait": wait}
    process = InertProcess()
    monkeypatch.setattr(workers, "WORKERS", {
        0: affirm_ready(workers.Worker(0, process, SimpleNamespace(), busy_task_id=task_id, active_capacity=False)),
    })
    # Each case owns its event frames. The process-lifetime bus is a SyncManager child that an
    # earlier test's real stop path can leave dead in this worker; a terminal frame that cannot
    # publish keeps its task in custody by design, which then reads as a stop that did not happen.
    events = stdqueue.Queue()
    monkeypatch.setattr(workers, "get_event_q", lambda: events)
    return SimpleNamespace(root=tmp_path, task_id=task_id, attempt=attempt, started=started, task=task,
                           ctx=ctx, wait=wait, process=process, transaction_id="native-restart-tx", events=events)


def first_cleanup(case):
    pending_ids = [row["id"] for row in workers.PENDING]
    native = owner_wait.prepare_owner_wait_handoffs(case.root, workers.RUNNING, case.transaction_id)
    selected = delegate_recovery.prepare_planned_restart_handoffs(
        case.root, workers.RUNNING, restart_transaction_id=case.transaction_id,
        additional_task_ids=native,
    )
    assert selected == native == {case.task_id}
    # As server._perform_supervisor_restart: the same transaction names the runnable queue (#1563).
    from supervisor.restart_retention import prepare_restart_returns

    prepare_restart_returns(case.root, workers.RUNNING, list(workers.PENDING),
                            transaction_id=case.transaction_id, owner_wait_ids=selected)
    assert workers.kill_workers(
        terminal_status="cancelled", result_reason="Planned self-restart",
        preserve_pending=True, preserve_running_task_ids=selected,
        reconcile_delegate_custody=False, archive_service_logs=False, retain_saved_work=True,
    )
    assert not case.process.is_alive() and not workers.RUNNING
    assert sorted(row["id"] for row in workers.PENDING) == sorted(pending_ids + [case.task_id])
    successor = next(row for row in workers.PENDING if row["id"] == case.task_id)
    assert successor["_attempt"] == case.attempt
    assert successor["_owner_wait_resume"]["started_at"] == case.started
    assert load_task_result(case.root, case.task_id)["status"] == "running"
    return successor


ACK_SOURCES = {"launcher": "launcher_waitpid", "direct_exec": "direct_exec_successor",
               "windows_spawn": "windows_direct_parent_handle"}


def acknowledge(case, monkeypatch, transport, parent=None, binding=None):
    """Leave the transaction prepared for the successor's reader, except the launcher's own wait."""
    if transport == "launcher":
        assert delegate_recovery.acknowledge_observed_restart_exit(
            case.root, supervisor_pid=os.getpid(), exit_code=42,
        )
        return
    # The production restore predicate consumes this one-shot token.
    monkeypatch.setenv(delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV, case.transaction_id)
    monkeypatch.setattr(delegate_recovery, "IS_WINDOWS", transport == "windows_spawn")
    if transport == "windows_spawn":
        windows_successor(monkeypatch, case.root, case.transaction_id, parent, binding)


def windows_successor(monkeypatch, root, transaction_id, parent=None, binding=None):
    """This process as a Windows successor of the row's supervisor, which binds the row while the successor waits."""
    from ouroboros import platform_layer

    real_birth = platform_layer.process_start_time
    births = {os.getpid(): "win-filetime:2"}
    monkeypatch.setattr(platform_layer, "process_start_time", lambda pid: births.get(pid) or real_birth(pid))
    monkeypatch.setattr(delegate_recovery, "_restart_parent", None)
    monkeypatch.setenv(delegate_recovery.PLANNED_RESTART_PARENT_ENV, "77")

    def await_parent(handle):
        assert handle == 77
        row = delegate_recovery._read_restart_transaction(root, transaction_id)
        assert row["status"] == "prepared" and "successor_pid" not in row  # bound only after the spawn
        assert row["expected_exit_code"] == 42  # the real producer's preparation, never a bare fixture row
        row.update(binding if binding is not None else {
            "supervisor_birth": "win-filetime:1", "successor_pid": os.getpid(), "successor_birth": "win-filetime:2"})
        delegate_recovery._write_restart_transaction(root, row)
        if isinstance(parent, Exception):
            raise parent
        return dict(parent) if parent is not None else {
            "pid": int(row["supervisor_pid"]), "birth": "win-filetime:1", "exit_code": 42}

    monkeypatch.setattr(platform_layer, "await_process_handle", await_parent)


def restore_stale_snapshot(case):
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    snapshot["ts"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    atomic_write_json(queue.QUEUE_SNAPSHOT_PATH, snapshot)
    workers.PENDING.clear()  # A new supervisor starts with empty process memory.
    return queue.restore_pending_from_snapshot(max_age_sec=900)


def test_native_handoff_is_visible_to_subsequent_cleanup(restart_case):
    first_cleanup(restart_case)
    assert delegate_recovery.has_planned_restart_handoffs(restart_case.root), (
        "A prepared native owner-wait transaction must be visible without a delegated-run row"
    )
    assert server_restart._managed_update_pending_kwargs() == {"preserve_pending": True}


@pytest.mark.parametrize("transport", ACK_SOURCES)
def test_native_wait_survives_second_cleanup_and_old_snapshot(restart_case, monkeypatch, transport):
    case = restart_case
    first_cleanup(case)
    kwargs = server_restart._managed_update_pending_kwargs()
    status, reason = server_restart._shutdown_task_cleanup_args(restart_requested=True)
    assert workers.kill_workers(terminal_status=status, result_reason=reason,
                                reconcile_delegate_custody=False, archive_service_logs=False, **kwargs)
    acknowledge(case, monkeypatch, transport)
    assert restore_stale_snapshot(case) == 1, {
        "second_cleanup_kwargs": kwargs,
        "task_status": load_task_result(case.root, case.task_id)["status"],
        "snapshot": json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text()),
    }
    restored = workers.PENDING[0]
    assert restored["id"] == case.task_id and restored["_attempt"] == case.attempt
    assert restored["_owner_wait_resume"]["started_at"] == case.started
    saved = owner_wait.load_owner_wait(case.ctx, restored["_owner_wait_resume"])
    assert saved["round_idx"] == 7 and saved["usage"] == {"cost": 2.5}
    assert saved["messages"][-1]["content"] == "Saved draft object 42"
    assert load_task_result(case.root, case.task_id)["total_rounds"] == 7


@pytest.mark.parametrize("transport", ACK_SOURCES)
def test_observed_restart_restores_real_native_source_past_snapshot_age(restart_case, monkeypatch, transport):
    case = restart_case
    first_cleanup(case)
    acknowledge(case, monkeypatch, transport)
    assert restore_stale_snapshot(case) == 1
    handoff = workers.PENDING[0]["_owner_wait_resume"]
    assert owner_wait.load_owner_wait(case.ctx, handoff)["trace"]["tool_calls"][0]["result"] == "object 42"
    transaction = delegate_recovery._read_restart_transaction(case.root, case.transaction_id)
    assert transaction["status"] == "normal_exit_acknowledged"
    assert transaction["ack_source"] == ACK_SOURCES[transport]


@pytest.mark.parametrize("fault", ["token_only", "observation_failed", "exit_1", "panic_99", "other_parent_pid",
                                   "other_parent_birth", "binding_failed", "other_successor"])
def test_windows_successor_without_exact_parent_proof_holds_the_wait(restart_case, monkeypatch, caplog, fault):
    """Only the inherited parent's own exit 42 and the bound identities return the wait (S1 keeps it held)."""
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact

    case = restart_case
    first_cleanup(case)
    parent = {"pid": os.getpid(), "birth": "win-filetime:1", "exit_code": 42}
    binding = None
    if fault == "observation_failed":
        parent = OSError("the process handle is invalid")
    elif fault in {"exit_1", "panic_99"}:
        parent["exit_code"] = 1 if fault == "exit_1" else 99
    elif fault == "other_parent_pid":
        parent["pid"] = os.getpid() + 1
    elif fault == "other_parent_birth":
        parent["birth"] = "win-filetime:9"
    elif fault == "binding_failed":
        binding = {}  # the parent exited 42 but never recorded the successor
    elif fault == "other_successor":
        binding = {"supervisor_birth": "win-filetime:1", "successor_pid": os.getpid(), "successor_birth": "win-filetime:9"}
    acknowledge(case, monkeypatch, "windows_spawn", parent, binding)
    if fault == "token_only":  # the same PID and the token, but no inherited parent
        monkeypatch.delenv(delegate_recovery.PLANNED_RESTART_PARENT_ENV)
    restore_stale_snapshot(case)
    [row] = workers.PENDING
    assert budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK and not row.get("_owner_wait_resume")
    assert row["_working_recovery"]["source_ref"] == case.wait["source_ref"]  # the exact wait source, held
    assert delegate_recovery._read_restart_transaction(case.root, case.transaction_id)["status"] == "prepared"
    assert "continuation unproven, its handoffs are not resumed" in caplog.text


@pytest.mark.parametrize("control", ["restart", "unacknowledged", "panic", "stop_child", "bad_child_source"])
def test_preserved_question_parent_keeps_its_saved_running_child(restart_case, monkeypatch, control):
    """Separate owner-question custody counts as surviving ancestry during both kills."""
    from ouroboros import working_checkpoint
    from ouroboros.cancel_intents import request_cancel
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _working_run

    case = restart_case
    case.task["root_task_id"] = case.task_id
    _working_run(case.root, workers, "saved-child", parent=case.task_id)
    workers.WORKERS[1] = affirm_ready(workers.Worker(1, InertProcess(), SimpleNamespace(), busy_task_id="saved-child"))
    if control == "stop_child":
        request_cancel(case.root, "saved-child", reason="owner stop", source="owner", requested_by="owner")
    if control == "bad_child_source":
        working_checkpoint.checkpoint_path(case.root, "saved-child", 1).write_bytes(b"broken source")
    native = owner_wait.prepare_owner_wait_handoffs(case.root, workers.RUNNING, case.transaction_id)
    selected = delegate_recovery.prepare_planned_restart_handoffs(
        case.root, workers.RUNNING, restart_transaction_id=case.transaction_id, additional_task_ids=native)
    assert selected == native == {case.task_id}
    prepare_restart_returns(case.root, workers.RUNNING, workers.PENDING,
                            transaction_id=case.transaction_id, owner_wait_ids=selected)
    assert workers.kill_workers(
        terminal_status="cancelled", result_reason="Planned self-restart",
        preserve_pending=True, preserve_running_task_ids=selected,
        reconcile_delegate_custody=False, archive_service_logs=False, retain_saved_work=True)
    assert [row["id"] for row in workers.PENDING] == [case.task_id]
    assert set(workers.RUNNING) == (set() if control == "stop_child" else {"saved-child"})
    if control == "stop_child":
        assert load_task_result(case.root, "saved-child")["status"] == "cancelled"
    else:
        assert load_task_result(case.root, "saved-child")["status"] == "running"
    terminal = [frame["status"] for frame in list(case.events.queue)
                if frame.get("type") == "task_done" and frame.get("task_id") == "saved-child"]
    assert terminal == (["cancelled"] if control == "stop_child" else [])
    # The lifespan's second cleanup sees the parent already queued and the child
    # still carried in RUNNING for recovery from its last working checkpoint.
    assert workers.kill_workers(
        terminal_status="cancelled", result_reason="Server shutdown.", preserve_pending=True,
        reconcile_delegate_custody=False, archive_service_logs=False, retain_saved_work=True)
    if control != "unacknowledged":
        acknowledge(case, monkeypatch, "launcher")
    if control == "panic":
        (case.root / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    workers.RUNNING.clear()
    workers.PENDING.clear()
    queue.restore_pending_from_snapshot()
    rows = {row["id"]: row for row in workers.PENDING}
    child_survives = control not in {"stop_child", "bad_child_source"}
    assert set(rows) == {case.task_id} | ({"saved-child"} if child_survives else set())
    if child_survives:
        child = rows["saved-child"]
        assert child["_attempt"] == 2 and child["parent_task_id"] == case.task_id
        source = working_checkpoint.recovery_source_for_task(case.root, child)
        assert source["messages"][-1]["content"] == "done a"
    if control in {"unacknowledged", "panic"}:
        assert all(budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK for row in rows.values())
        commands = stdqueue.Queue()
        workers.WORKERS[0] = affirm_ready(workers.Worker(0, InertProcess(), commands))
        monkeypatch.setattr(workers, "load_state", lambda: {})
        workers.assign_tasks()
        assert commands.empty() and not workers.RUNNING
    else:
        parent = rows[case.task_id]
        assert parent["_owner_wait_resume"]["source_ref"] == case.wait["source_ref"]
        assert owner_wait.load_owner_wait(case.ctx, parent["_owner_wait_resume"])["round_idx"] == 7
        assert all(budget_hold_fact(row) is None for row in rows.values())
        from supervisor import state

        commands = stdqueue.Queue()
        workers.WORKERS.update({wid: affirm_ready(workers.Worker(wid, InertProcess(), commands)) for wid in (0, 1)})
        monkeypatch.setattr(workers, "load_state", lambda: {})
        monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 100.0)
        workers.assign_tasks()
        assert {commands.get_nowait()["id"] for _ in rows} == set(rows)
        assert set(workers.RUNNING) == set(rows)
        # Remaining input belongs only to the spare's quiescence protocol.
        assert all(commands.get_nowait()["type"] == "worker_quiescence"
                   for _ in range(commands.qsize()))


@pytest.mark.parametrize("refusal", ["unacknowledged", "spent_wait", "panic", "owner_restart", "bad_source"])
def test_old_snapshot_never_revives_unproven_or_spent_wait(restart_case, monkeypatch, refusal):
    """Owner 2026-10-08 (#1563): a crash-like (unacknowledged) stop or a Panic keeps the
    exact question-wait source HELD for an explicit Resume — never dispatched by itself;
    the owner's acknowledged Restart returns the wait it named; a spent wait or an
    unreadable source never revives."""
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact

    case = restart_case
    first_cleanup(case)
    if refusal != "unacknowledged":
        acknowledge(case, monkeypatch, "launcher")
    if refusal == "spent_wait":
        owner_wait.set_owner_wait(case.root, case.task_id, {**case.wait, "state": "resumed"},
                                  expected_wait_id=case.wait["wait_id"])
    elif refusal in {"panic", "owner_restart"}:
        name = "panic_stop.flag" if refusal == "panic" else "owner_restart_no_resume.flag"
        (case.root / "state" / name).write_text(refusal)
    elif refusal == "bad_source":
        path = task_artifact_dir_path(case.root, case.task_id) / case.wait["source_ref"]["path"]
        path.write_bytes(b"broken source")
    restored = restore_stale_snapshot(case)
    if refusal == "spent_wait":
        assert restored == 0 and not workers.PENDING
        return
    [row] = workers.PENDING
    if refusal == "bad_source":
        from ouroboros.work_stop import continuation_refusal
        assert restored == 1 and continuation_refusal(row) and not row.get("_resume_selection")
        assert row["_owner_wait_resume"]["source_ref"] == case.wait["source_ref"]
        assert path.read_bytes() == b"broken source", "holding must not rewrite the original"
        commands = stdqueue.Queue()
        workers.WORKERS[0] = affirm_ready(workers.Worker(0, InertProcess(), commands))
        monkeypatch.setattr(workers, "load_state", lambda: {})
        workers.assign_tasks()
        assert commands.empty() and not workers.RUNNING
        return
    if refusal == "owner_restart":  # the owner's acknowledged Restart returns the wait it named
        assert row["_owner_wait_resume"]["source_ref"] == case.wait["source_ref"]
        assert budget_hold_fact(row) is None
        return
    assert budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK, "held for an explicit Resume"
    assert budget_hold_fact(row)["door"] == ("panic" if refusal == "panic" else "app_stop")
    assert not row.get("_owner_wait_resume"), "never the cold wait handoff the stop refused"
    assert row["_working_recovery"]["source_ref"] == case.wait["source_ref"]
    assert row["_attempt"] == case.attempt + 1
    commands = stdqueue.Queue()
    workers.WORKERS[0] = affirm_ready(workers.Worker(0, InertProcess(), commands))
    monkeypatch.setattr(workers, "load_state", lambda: {})
    workers.assign_tasks()
    assert commands.empty() and not workers.RUNNING, "nothing runs before the owner's Resume"


def test_running_projection_preserves_the_native_continuation_before_cold_load(restart_case):
    from ouroboros.agent import OuroborosAgent

    case = restart_case
    task = first_cleanup(case)
    before = read_actor_source_bytes(case.root, case.task_id, case.wait["source_ref"])
    # _prepare_task_context writes this row before run_llm_loop loads the source.
    agent = SimpleNamespace(env=SimpleNamespace(drive_root=case.root), _task_started_ts=case.started)
    OuroborosAgent._persist_running_record(agent, task)
    row = load_task_result(case.root, case.task_id)
    saved = owner_wait.load_owner_wait(case.ctx, task["_owner_wait_resume"])
    assert row["owner_wait"] == case.wait
    assert row["total_rounds"] == 7 and row["accounted_upper_bound_usd"] == 2.5
    assert read_actor_source_bytes(case.root, case.task_id, case.wait["source_ref"]) == before
    assert saved["messages"][-1]["content"] == "Saved draft object 42"
    assert dt.datetime.fromisoformat(row["started_at"]).timestamp() == pytest.approx(case.started)


def test_running_projection_cannot_turn_consumed_wait_back_into_a_resume(restart_case):
    from ouroboros.agent import OuroborosAgent

    case = restart_case
    task = first_cleanup(case)
    owner_wait.set_owner_wait(case.root, case.task_id, {**case.wait, "state": "resumed"},
                              expected_wait_id=case.wait["wait_id"])
    agent = SimpleNamespace(env=SimpleNamespace(drive_root=case.root), _task_started_ts=case.started)
    OuroborosAgent._persist_running_record(agent, task)
    with pytest.raises(ValueError, match="not an active"):
        owner_wait.load_owner_wait(case.ctx, task["_owner_wait_resume"])
    assert load_task_result(case.root, case.task_id)["owner_wait"]["state"] == "resumed"


@pytest.mark.parametrize("remaining", [100.0, 0.0])
def test_child_budget_fence_allows_only_restored_owner_continuation(restart_case, monkeypatch, remaining):
    from ouroboros import usage_accounting as accounting
    from supervisor import events_budget, state, task_lifecycle

    case = restart_case
    case.task.update(root_task_id=case.task_id, delegation_role="root")
    sibling = {"id": "not-started-child", "type": "task", "chat_id": 1, "depth": 1,
               "root_task_id": case.task_id, "parent_task_id": case.task_id,
               "delegation_role": "subagent", "text": "Unstarted independent work"}
    assert not queue.enqueue_task(sibling).get("_admission_blocked")
    write_task_result(case.root, "spent-child", "failed", root_task_id=case.task_id,
                      parent_task_id=case.task_id, reason_code="budget_exhausted")
    events_budget._handle_budget_root_fence({
        "type": "budget_root_fence", "task_id": "spent-child", "task_type": "task",
        "resource_limit": {"scope": "root", "root_task_id": case.task_id},
    }, SimpleNamespace(DRIVE_ROOT=case.root, RUNNING=workers.RUNNING,
                       persist_queue_snapshot=queue.persist_queue_snapshot,
                       bridge=SimpleNamespace(push_log=lambda event: None)))
    assert workers.RUNNING[case.task_id]["owner_wait"]["state"] == "waiting"
    fence = dict(queue.BUDGET_ROOT_FENCES[case.task_id])
    first_cleanup(case)
    acknowledge(case, monkeypatch, "launcher")

    # A new supervisor restores the snapshot, not leftovers in process maps.
    workers.PENDING.clear()
    workers.RUNNING.clear()
    workers.WORKERS.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    queue.ACCEPTANCE_FENCES.clear()
    queue.ADMISSION_RESERVATIONS.clear()
    queue.QUEUE_SEQ_COUNTER_REF["value"] = 0
    assert queue.restore_pending_from_snapshot() == 2
    assert queue.BUDGET_ROOT_FENCES is task_lifecycle.BUDGET_ROOT_FENCES
    assert queue.BUDGET_ROOT_FENCES == {case.task_id: fence}

    commands = stdqueue.Queue()
    workers.WORKERS[0] = affirm_ready(workers.Worker(0, InertProcess(), commands))
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda task: True)
    monkeypatch.setattr(state, "budget_remaining", lambda *args, **kwargs: remaining)
    workers.assign_tasks()
    assert set(workers.RUNNING) == {case.task_id}
    sent = commands.get_nowait()
    assert commands.empty() and sent["id"] == case.task_id
    assert sent["_attempt"] == case.attempt
    assert workers.RUNNING[case.task_id]["started_at"] == case.started
    handoff = sent["_owner_wait_resume"]
    assert handoff["source_ref"] == case.wait["source_ref"]
    assert owner_wait.load_owner_wait(case.ctx, handoff)["round_idx"] == 7
    if remaining > 0:
        assert [row["id"] for row in workers.PENDING] == [sibling["id"]]
    else:
        # Ordinary work follows its existing global-budget pause/terminal rail.
        stopped = load_task_result(case.root, sibling["id"])
        assert stopped["reason_code"] == "budget_exhausted"
        assert stopped["resource_limit"]["scope"] == "global"
    assert queue.BUDGET_ROOT_FENCES == {case.task_id: fence}
    assert json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())["budget_root_fences"] == [fence]
    fresh = queue.enqueue_task({**sibling, "id": "fresh-child"})
    assert fresh["_admission_blocked"] == "root_budget_fence"
    with accounting.usage_scope(accounting.UsageScope(
        drive_root=case.root, task_id=case.task_id, root_task_id=case.task_id,
        global_limit_usd=100.0, root_limit_usd=100.0,
    )):
        with pytest.raises(accounting.BudgetExceeded) as refused:
            accounting.reserve_attempt(accounting.AttemptRequest(
                model="fixture", provider="openai", reservation_usd=1.0))
        assert refused.value.limit_scope == "root"


def restore_project_wait(case, monkeypatch, *, registry_outage=True):
    """Real planned-restart handoff and restore, with an optional admission outage."""
    from ouroboros import projects_registry as registry
    from supervisor import state, git_ops

    state.init(case.root)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", case.root)
    room = registry.create_project(case.root, "target")
    case.task.update(project_id="target", chat_id=room["chat_id"], admitted_dispatch="possible",
                     _project_admission=registry.project_admission_view(case.root, "target", frozen=True))
    first_cleanup(case)
    acknowledge(case, monkeypatch, "launcher")
    path = registry._registry_path(case.root)
    original = path.read_bytes()
    if registry_outage:
        path.write_text("{torn", encoding="utf-8")
    assert restore_stale_snapshot(case) == (0 if registry_outage else 1)
    [restored] = workers.PENDING
    assert bool(restored.get("_project_admission_restore_hold")) is registry_outage
    path.write_bytes(original)
    return restored


@pytest.mark.serial
@pytest.mark.parametrize("owner_restart", [False, True])
def test_project_wait_recovers_once_after_registry_hold(restart_case, monkeypatch, owner_restart):
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored = restore_project_wait(case, monkeypatch)
    if owner_restart:  # owner 2026-10-08 (quiz d2f7532b): the owner's Restart returns the work it named
        from ouroboros import delegate_recovery
        tx = delegate_recovery._read_restart_transaction(
            case.root, restored["_owner_wait_resume"]["restart_transaction_id"])
        tx.update(door="owner_restart", generation="previous", door_applied_at="")
        delegate_recovery._write_restart_transaction(case.root, tx)
        delegate_recovery.consume_owner_restart(case.root, tx)
    handoff = dict(restored["_owner_wait_resume"])
    original = read_actor_source_bytes(case.root, case.task_id, handoff["source_ref"])
    sent = worker(SimpleNamespace(root=case.root), monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [case.task_id]
    assert sent[0]["_attempt"] == case.attempt
    assert sent[0]["_owner_wait_resume"] == handoff
    assert sent[0]["admitted_dispatch"] == "possible"
    assert not sent[0].get("_project_admission_restore_hold")
    assert read_actor_source_bytes(case.root, case.task_id, handoff["source_ref"]) == original
    assert owner_wait.load_owner_wait(case.ctx, handoff)["round_idx"] == 7


@pytest.mark.serial
@pytest.mark.parametrize("fault", ["result", "source", "transaction"])
def test_project_wait_unreadable_authority_stays_same_id_then_recovers(restart_case, monkeypatch, fault):
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored = restore_project_wait(case, monkeypatch, registry_outage=fault != "result")
    signature = result_signature(case.root, case.task_id)
    if fault == "result":
        unreadable = unreadable_result(monkeypatch, case.task_id)  # the store refuses the row
    else:
        path = (delegate_recovery._restart_transaction_path(case.root, case.transaction_id) if fault == "transaction"
                else task_artifact_dir_path(case.root, case.task_id) / case.wait["source_ref"]["path"])
        original = path.read_bytes()
        path.write_bytes(b"{torn")
    sent = worker(SimpleNamespace(root=case.root), monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert not sent and workers.PENDING == [restored]
    assert restored["_project_admission_restore_hold"] and not restored.get("_terminalization_retry")
    assert restored["admitted_dispatch"] == "possible"
    if fault == "result":
        unreadable["on"] = False
    else:
        path.write_bytes(original)
    assert result_signature(case.root, case.task_id) == signature  # the hold wrote nothing over the authority
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [case.task_id]
    assert sent[0]["_owner_wait_resume"]["source_ref"] == case.wait["source_ref"]
    assert owner_wait.load_owner_wait(case.ctx, sent[0]["_owner_wait_resume"])["round_idx"] == 7


@pytest.mark.serial
@pytest.mark.parametrize("refusal", ["unacknowledged", "spent_wait", "owner_restart", "deadline", "ceiling",
                                   "ceiling_unreadable_transaction", "stop"])
def test_project_wait_positive_refusal_uses_existing_custody(restart_case, monkeypatch, refusal):
    from ouroboros.cancel_intents import request_cancel
    from supervisor import queue_timeouts
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored = restore_project_wait(case, monkeypatch)
    deadline_jobs = stdqueue.Queue()
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "_reap_queue", deadline_jobs)
    monkeypatch.setattr(queue_timeouts, "_pending_deadline_jobs", {})
    if refusal == "unacknowledged":
        transaction = delegate_recovery._read_restart_transaction(case.root, case.transaction_id)
        delegate_recovery._write_restart_transaction(case.root, {**transaction, "status": "prepared"})
    elif refusal == "spent_wait":
        owner_wait.set_owner_wait(case.root, case.task_id, {**case.wait, "state": "resumed"},
                                  expected_wait_id=case.wait["wait_id"])
    elif refusal == "owner_restart":
        from tests._work_stop_helpers import live_restart
        live_restart(case.root, monkeypatch)
    elif refusal == "deadline":
        restored["deadline_at"] = "2000-01-01T00:00:00Z"
    elif refusal.startswith("ceiling"):
        monkeypatch.setattr("ouroboros.config.get_task_abs_ceiling_sec", lambda: 1)
        if refusal == "ceiling_unreadable_transaction":
            delegate_recovery._restart_transaction_path(case.root, case.transaction_id).write_bytes(b"{torn")
    else:
        request_cancel(case.root, case.task_id, reason="owner stopped")
    sent = worker(SimpleNamespace(root=case.root), monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    if refusal == "deadline":
        # Expiry retains custody until the off-loop reaper durably settles it.
        assert not sent and workers.PENDING == [restored]
        job = deadline_jobs.get_nowait()
        assert job["kind"] == "pending_deadline"
        queue_timeouts._reap_pending_deadline(job)
        deadline_jobs.task_done()
    assert deadline_jobs.empty()
    assert not sent and not workers.PENDING
    stored = load_task_result(case.root, case.task_id)
    assert stored["status"] == ("failed" if refusal == "deadline" else "cancelled")
    assert stored.get("admission_outcome") != "never_admitted"
    if refusal not in {"deadline", "stop"}:
        assert stored["cancel_origin"]["reason"] == "server_shutdown"
        assert stored["cancel_origin"]["source"] == "snapshot_restore"


def _expired_project_wait(case, monkeypatch):
    """The restored acknowledged wait past its explicit deadline, with the reaper's queue captured."""
    from supervisor import queue_timeouts

    restored = restore_project_wait(case, monkeypatch)
    jobs = stdqueue.Queue()
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "_reap_queue", jobs)
    monkeypatch.setattr(queue_timeouts, "_pending_deadline_jobs", {})
    restored["deadline_at"] = "2000-01-01T00:00:00Z"
    return restored, jobs


@pytest.mark.serial
@pytest.mark.parametrize("unknown", ["transaction", "source", "store"])
def test_project_wait_deadline_keeps_custody_while_its_authority_is_unreadable(restart_case, monkeypatch, unknown):
    """#939: the deadline settles only the exact continuation it can still read; unknown authority
    keeps the hold (no marker, start or deletion) until it reads again, then the deadline binds."""
    from supervisor import queue_timeouts
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored, jobs = _expired_project_wait(case, monkeypatch)
    transaction = delegate_recovery._restart_transaction_path(case.root, case.transaction_id)
    source = task_artifact_dir_path(case.root, case.task_id) / case.wait["source_ref"]["path"]
    saved = {"transaction": transaction.read_bytes(), "source": source.read_bytes()}
    unreadable = unreadable_result(monkeypatch, case.task_id) if unknown == "store" else None
    if unknown != "store":
        (transaction if unknown == "transaction" else source).write_bytes(b"{torn")
    sent = worker(SimpleNamespace(root=case.root), monkeypatch)
    for _ in range(2):
        queue.enforce_task_timeouts()
        workers.assign_tasks()
    assert not sent and workers.PENDING == [restored] and jobs.empty()
    assert restored["_project_admission_restore_hold"] and not restored.get("_terminalization_retry")
    if unreadable:
        unreadable["on"] = False
    else:
        (transaction if unknown == "transaction" else source).write_bytes(saved[unknown])
    assert load_task_result(case.root, case.task_id)["status"] == "running"
    workers.assign_tasks()
    workers.assign_tasks()
    queue_timeouts._reap_pending_deadline(jobs.get_nowait())
    assert not sent and not workers.PENDING
    stored = load_task_result(case.root, case.task_id)
    assert stored["status"] == "failed" and stored["reason_code"] == "deadline"
    assert read_actor_source_bytes(case.root, case.task_id, case.wait["source_ref"]) == saved["source"]


@pytest.mark.serial
@pytest.mark.parametrize("control", ["owner_pause", "stop", "saved_terminal", "live_owner"])
def test_project_wait_deadline_rechecks_controls_before_settlement(restart_case, monkeypatch, control):
    """A control arriving after the deadline marker and before the reaper's write wins."""
    from ouroboros.cancel_intents import request_cancel
    from supervisor import queue_timeouts
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored, jobs = _expired_project_wait(case, monkeypatch)
    sent = worker(SimpleNamespace(root=case.root), monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert restored["_terminalization_retry"]["trigger"] == "deadline" and jobs.qsize() == 1
    if control == "owner_pause":
        assert request_owner_pause(case.task_id, request_id="pause-before-deadline")["ok"]
    elif control == "stop":
        request_cancel(case.root, case.task_id, reason="owner stopped")
    elif control == "saved_terminal":
        write_task_result(case.root, case.task_id, "completed", result="Saved answer")
    else:
        workers.WORKERS[0].busy_task_id = case.task_id
    queue_timeouts._reap_pending_deadline(jobs.get_nowait())
    stored = load_task_result(case.root, case.task_id)
    assert not sent and stored.get("reason_code") != "deadline"
    if control == "saved_terminal":
        assert not workers.PENDING and stored["status"] == "completed" and stored["result"] == "Saved answer"
        return
    assert workers.PENDING == [restored] and stored["status"] == "running"
    if control == "stop":
        assert queue.cancel_task_custody(case.task_id) in {queue.CANCEL_CANCELLED, queue.CANCEL_ALREADY_SETTLED}
        assert load_task_result(case.root, case.task_id)["status"] == "cancelled" and not workers.PENDING


@pytest.mark.serial
def test_project_wait_refusal_retains_custody_until_cancel_intent_is_durable(restart_case, monkeypatch):
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored = restore_project_wait(case, monkeypatch)
    transaction = delegate_recovery._read_restart_transaction(case.root, case.transaction_id)
    delegate_recovery._write_restart_transaction(case.root, {**transaction, "status": "prepared"})
    sent = worker(SimpleNamespace(root=case.root), monkeypatch)
    with monkeypatch.context() as patch:
        def unavailable(*args, **kwargs):
            raise OSError("synthetic cancel intent write failure")
        patch.setattr("ouroboros.cancel_intents.request_cancel", unavailable)
        workers.assign_tasks()
        workers.assign_tasks()
    assert workers.PENDING == [restored] and not sent
    assert restored["_project_admission_restore_hold"] and not restored.get("_terminalization_retry")
    assert load_task_result(case.root, case.task_id)["status"] == "running"
    workers.assign_tasks()
    workers.assign_tasks()
    assert not workers.PENDING and not sent
    assert load_task_result(case.root, case.task_id)["status"] == "cancelled"
