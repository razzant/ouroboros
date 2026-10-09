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
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import delegate_recovery, owner_wait, server_restart
from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import atomic_write_json
from supervisor import queue, workers


class InertProcess:
    """A handle whose termination never reaches an operating-system PID."""

    pid = None

    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def join(self, timeout=None):
        pass


@pytest.fixture
def restart_case(tmp_path, monkeypatch):
    from supervisor import task_lifecycle, update_merge

    pending, running = [], {}
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
        0: workers.Worker(0, process, SimpleNamespace(), busy_task_id=task_id, active_capacity=False),
    })
    return SimpleNamespace(root=tmp_path, task_id=task_id, attempt=attempt, started=started,
                           task=task, ctx=ctx, wait=wait, process=process, transaction_id="native-restart-tx")


def first_cleanup(case):
    pending_ids = [row["id"] for row in workers.PENDING]
    native = owner_wait.prepare_owner_wait_handoffs(case.root, workers.RUNNING, case.transaction_id)
    selected = delegate_recovery.prepare_planned_restart_handoffs(
        case.root, workers.RUNNING, restart_transaction_id=case.transaction_id,
        additional_task_ids=native,
    )
    assert selected == native == {case.task_id}
    assert workers.kill_workers(
        terminal_status="cancelled", result_reason="Planned self-restart",
        preserve_pending=True, preserve_running_task_ids=selected,
        reconcile_delegate_custody=False, archive_service_logs=False,
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
def test_windows_successor_without_exact_parent_proof_resumes_nothing(restart_case, monkeypatch, caplog, fault):
    """Only the inherited parent's own exit 42 and the bound identities authorize the wait."""
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
    assert restore_stale_snapshot(case) == 0
    assert not workers.PENDING
    assert delegate_recovery._read_restart_transaction(case.root, case.transaction_id)["status"] == "prepared"
    assert "continuation unproven, its handoffs are not resumed" in caplog.text


@pytest.mark.parametrize("refusal", ["unacknowledged", "spent_wait", "panic", "owner_restart", "bad_source"])
def test_old_snapshot_never_revives_unproven_or_spent_wait(restart_case, monkeypatch, refusal):
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
    assert restore_stale_snapshot(case) == 0
    assert not workers.PENDING


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
    workers.WORKERS[0] = workers.Worker(0, InertProcess(), commands)
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
def test_project_wait_recovers_once_after_registry_hold(restart_case, monkeypatch):
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored = restore_project_wait(case, monkeypatch)
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
    if fault == "result":
        path = case.root / "task_results" / (case.task_id + ".json")
    elif fault == "transaction":
        path = delegate_recovery._restart_transaction_path(case.root, case.transaction_id)
    else:
        path = task_artifact_dir_path(case.root, case.task_id) / case.wait["source_ref"]["path"]
    original = path.read_bytes()
    path.write_bytes(b"{torn")
    sent = worker(SimpleNamespace(root=case.root), monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert not sent and workers.PENDING == [restored]
    assert restored["_project_admission_restore_hold"] and not restored.get("_terminalization_retry")
    assert restored["admitted_dispatch"] == "possible" and path.read_bytes() == b"{torn"
    path.write_bytes(original)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [case.task_id]
    assert sent[0]["_owner_wait_resume"]["source_ref"] == case.wait["source_ref"]
    assert owner_wait.load_owner_wait(case.ctx, sent[0]["_owner_wait_resume"])["round_idx"] == 7


@pytest.mark.serial
@pytest.mark.parametrize("refusal", ["unacknowledged", "spent_wait", "panic", "owner_restart", "deadline", "ceiling",
                                   "ceiling_unreadable_transaction", "stop"])
def test_project_wait_positive_refusal_uses_existing_custody(restart_case, monkeypatch, refusal):
    from ouroboros.cancel_intents import request_cancel
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored = restore_project_wait(case, monkeypatch)
    if refusal == "unacknowledged":
        transaction = delegate_recovery._read_restart_transaction(case.root, case.transaction_id)
        delegate_recovery._write_restart_transaction(case.root, {**transaction, "status": "prepared"})
    elif refusal == "spent_wait":
        owner_wait.set_owner_wait(case.root, case.task_id, {**case.wait, "state": "resumed"},
                                  expected_wait_id=case.wait["wait_id"])
    elif refusal in {"panic", "owner_restart"}:
        name = "panic_stop.flag" if refusal == "panic" else "owner_restart_no_resume.flag"
        (case.root / "state" / name).write_text(refusal, encoding="utf-8")
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
    assert not sent and not workers.PENDING
    stored = load_task_result(case.root, case.task_id)
    assert stored["status"] == ("failed" if refusal == "deadline" else "cancelled")
    assert stored.get("admission_outcome") != "never_admitted"
    if refusal not in {"deadline", "stop"}:
        assert stored["cancel_origin"]["reason"] == "server_shutdown"
        assert stored["cancel_origin"]["source"] == "snapshot_restore"


@pytest.mark.serial
def test_project_wait_refusal_retains_custody_until_cancel_intent_is_durable(restart_case, monkeypatch):
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored = restore_project_wait(case, monkeypatch)
    (case.root / "state/panic_stop.flag").write_text("panic", encoding="utf-8")
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
