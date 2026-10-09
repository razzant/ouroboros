"""Planned delegated restart through cleanup, snapshot, assignment and adoption.

Only process, gateway and Win32 observations are doubles. Queue admission,
custody, transaction proof, cleanup, restore and worker assignment are real.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody, delegate_recovery as recovery
from ouroboros.subagent_work_order import work_order_fingerprint
from ouroboros.task_results import load_task_result
from ouroboros.utils import atomic_write_json
from supervisor import queue, workers
from tests.test_available_subagents_runtime import _session_row, _settings, _snapshot
from tests.test_owner_wait_restart import InertProcess, windows_successor
from tests.test_project_hold_recovery import accepted, worker
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.test_windows_restart_proof import Gateway

pytestmark = pytest.mark.serial


@pytest.fixture
def prepared(host, tmp_path, monkeypatch, request):  # noqa: F811
    import ouroboros.claudexor_daemon as daemon

    scoped = request.param
    task = (accepted(host, tmp_path, tid="child1") if scoped else queue.enqueue_task({
        "id": "child1", "type": "task", "chat_id": 1, "text": "Continue the saved work", "depth": 0,
        "workspace_root": str(tmp_path), "drive_root": str(host.root),
    }))
    task["configured_subagent"] = _snapshot(_settings(_session_row()), "session-builder")
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert len(sent) == 1 and not host.pending
    task = host.running["child1"]["task"]
    from ouroboros.agent import OuroborosAgent

    OuroborosAgent._persist_running_record(SimpleNamespace(env=SimpleNamespace(drive_root=host.root)), task)
    assert task["admitted_dispatch"] == "possible"
    assert load_task_result(host.root, "child1")["admitted_dispatch"] == "possible"
    process = InertProcess()
    monkeypatch.setattr(workers, "WORKERS", {
        0: workers.Worker(0, process, SimpleNamespace(), busy_task_id="child1", active_capacity=False),
    })
    monkeypatch.setattr(custody, "_CUSTODY", {})
    snapshot = task["configured_subagent"]
    custody.record_started(host.root, custody.RunCustody(
        run_id="run-1", task_id="child1", route_id="codex", selected_subagent_id="session-builder",
        config_fingerprint=snapshot["config_fingerprint"],
        authority_fingerprint=recovery.authority_fingerprint_from_task(task),
        work_order_fingerprint=work_order_fingerprint(task),
    ))
    atomic_write_json(host.root / "state/delegate_supervision/child1.json", {
        "schema": 1, "run_id": "run-1", "status": "sleeping", "journal_cursor": 7,
    })
    selected = recovery.prepare_planned_restart_handoffs(host.root, host.running)
    assert selected == {"child1"}
    handoff = recovery._read(host.root, "child1")
    assert workers.kill_workers(
        terminal_status="cancelled", result_reason="Planned self-restart", preserve_pending=True,
        preserve_running_task_ids=selected, reconcile_delegate_custody=False, archive_service_logs=False,
    )
    assert not process.is_alive() and not host.running
    assert host.pending[0]["_attempt"] == handoff["new_attempt"]
    assert queue.persist_queue_snapshot()
    monkeypatch.setattr(recovery.os, "getpid", lambda: handoff["supervisor_pid"] + 100_000)
    monkeypatch.setattr(recovery, "IS_WINDOWS", True)
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    monkeypatch.setenv(recovery.PLANNED_RESTART_TRANSACTION_ENV, handoff["restart_transaction_id"])
    windows_successor(monkeypatch, host.root, handoff["restart_transaction_id"])
    return SimpleNamespace(host=host, task=task, handoff=handoff)


def restore(case):
    case.host.pending.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert case.host.pending[0]["_project_admission_restore_hold"]


@pytest.mark.parametrize("prepared", [False, True], indirect=True, ids=["unscoped", "project"])
@pytest.mark.parametrize("adopt_before_restore", [False, True], ids=["startup-order", "pre-adopted-snapshot"])
def test_exact_planned_handoff_reaches_worker_once(prepared, monkeypatch, adopt_before_restore):
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.subagent_bootstrap import _adopt_recovery_handoff
    from ouroboros.tools.registry import ToolContext
    import ouroboros.tools.delegate as delegate

    case = prepared
    if adopt_before_restore:
        assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    restore(case)
    if not adopt_before_restore:
        assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    sent = worker(case.host, monkeypatch)
    workers.assign_tasks()
    assert len(sent) == 1, [row.get("_project_admission_restore_hold") for row in case.host.pending]
    workers.assign_tasks()
    assert len(sent) == 1 and not case.host.pending
    task = sent[0]
    assert task["_attempt"] == case.handoff["new_attempt"]
    assert task["configured_subagent"] == case.task["configured_subagent"]
    assert task["workspace_root"] == case.task["workspace_root"]
    monkeypatch.setattr(recovery.os, "getpid", lambda: case.handoff["supervisor_pid"] + 200_000)
    monkeypatch.setattr(delegate, "exact_start", lambda *_a, **_k: pytest.fail("must adopt, never start another run"))
    ctx = ToolContext(repo_dir=case.host.root, drive_root=case.host.root, task_id="child1",
                      task_metadata={"drive_root": str(case.host.root)},
                      task_constraint=normalize_task_constraint(task.get("task_constraint")),
                      task_contract=task.get("task_contract"))
    result = json.loads(_adopt_recovery_handoff(ctx, task))
    assert result["status"] == "configured_session_started"
    assert result["recovery"]["status"] == "adopted" and result["run_id"] == "run-1"
    assert recovery._read(case.host.root, "child1")["status"] == "adopted"


@pytest.mark.parametrize("prepared", [False, True], indirect=True, ids=["unscoped", "project"])
@pytest.mark.parametrize("fault", ["bad_proof", "ordinary_restart", "panic"])
def test_unproven_or_owner_stopped_restart_never_assigns(prepared, monkeypatch, fault):
    case = prepared
    if fault == "bad_proof":
        windows_successor(monkeypatch, case.host.root, case.handoff["restart_transaction_id"], binding={})
    else:
        flag = "owner_restart_no_resume.flag" if fault == "ordinary_restart" else "panic_stop.flag"
        (case.host.root / "state" / flag).write_text("stop", encoding="utf-8")
    restore(case)
    assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == set()
    sent = worker(case.host, monkeypatch)
    workers.assign_tasks()
    assert not sent and not case.host.running


@pytest.mark.parametrize("prepared", [False, True], indirect=True, ids=["unscoped", "project"])
@pytest.mark.parametrize("fault", [
    "reserved", "adopted", "vetoed", "worker_crash", "wrong_task", "wrong_attempt", "authority",
    "actor", "config", "work_order", "tx_missing", "tx_unreadable", "tx_prepared", "tx_identity",
    "tx_parent", "tx_exit", "tx_membership", "owner_restart", "panic", "owner_hold", "budget", "acceptance",
])
def test_assignment_rechecks_exact_pre_adoption_and_independent_fences(prepared, monkeypatch, fault):
    case = prepared
    restore(case)
    assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    row = recovery._read(case.host.root, "child1")
    task = case.host.pending[0]
    transaction_id = row["restart_transaction_id"]
    if fault in {"reserved", "adopted", "vetoed"}:
        row["status"] = fault
    elif fault == "worker_crash":
        row["cause"] = recovery.CAUSE_WORKER_CRASH
    elif fault == "wrong_task":
        row["task_id"] = "another-task"
    elif fault == "wrong_attempt":
        row["new_attempt"] += 1
    elif fault in {"authority", "actor", "config", "work_order"}:
        key = {"authority": "authority_fingerprint", "actor": "selected_subagent_id",
               "config": "config_fingerprint", "work_order": "work_order_fingerprint"}[fault]
        row[key] = "changed"
    elif fault.startswith("tx_"):
        transaction = recovery._read_restart_transaction(case.host.root, transaction_id)
        path = recovery._restart_transaction_path(case.host.root, transaction_id)
        if fault == "tx_missing":
            path.unlink()
        elif fault == "tx_unreadable":
            path.write_text("{torn", encoding="utf-8")
        else:
            key, value = {"tx_prepared": ("status", "prepared"), "tx_identity": ("transaction_id", "another"),
                          "tx_parent": ("supervisor_pid", -1), "tx_exit": ("exit_code", 99),
                          "tx_membership": ("task_ids", [])}[fault]
            transaction[key] = value
            atomic_write_json(path, transaction)
    elif fault in {"owner_restart", "panic"}:
        flag = "owner_restart_no_resume.flag" if fault == "owner_restart" else "panic_stop.flag"
        (case.host.root / "state" / flag).write_text("stop", encoding="utf-8")
    elif fault == "owner_hold":
        task["_owner_hold"] = {"reason": "owner"}
    elif fault == "budget":
        task["_budget_pause"] = {"reason": "budget"}
    elif fault == "acceptance":
        queue.ACCEPTANCE_FENCES[str(task.get("root_task_id") or "")] = {"status": "active"}
    atomic_write_json(recovery._path(case.host.root, "child1"), row)
    if fault in {"owner_hold", "budget", "acceptance"}:
        assert recovery.planned_handoff_resume_allowed(case.host.root, task)
    sent = worker(case.host, monkeypatch)
    workers.assign_tasks()
    assert not sent and not case.host.running
    assert len(case.host.pending) == 1
