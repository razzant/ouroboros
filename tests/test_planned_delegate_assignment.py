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
from tests.test_owner_wait_restart import InertProcess, kill_inert, windows_successor
from tests.test_project_hold_recovery import accepted, worker
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.test_windows_restart_proof import Gateway
from tests.test_pending_deadline_continuations import expiry  # noqa: F401
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial


@pytest.fixture
def prepared(host, tmp_path, monkeypatch, request):  # noqa: F811
    import ouroboros.claudexor_daemon as daemon

    options = request.param if isinstance(request.param, tuple) else (request.param, False)
    scoped, timed = options[:2]
    saved = len(options) > 2 and options[2] in {"saved", "saved_unreadable"}
    if timed:
        from datetime import datetime, timezone
        import time

        enqueue = queue.enqueue_task
        deadline = datetime.fromtimestamp(time.time() + 600, timezone.utc).isoformat()
        timed_enqueue = lambda row, **kwargs: enqueue({**row, "deadline_at": deadline}, **kwargs)
        monkeypatch.setattr(queue, "enqueue_task", timed_enqueue)
        monkeypatch.setattr(host.ctx, "enqueue_task", timed_enqueue)
    task = (accepted(host, tmp_path, tid="child1") if scoped else queue.enqueue_task({
        "id": "child1", "type": "task", "chat_id": 1, "text": "Continue the saved work", "depth": 0,
        "workspace_root": str(tmp_path), "drive_root": str(host.root),
    }))
    if timed:
        assert task["deadline_at"] == deadline
    task["configured_subagent"] = _snapshot(_settings(_session_row()), "session-builder")
    if saved:
        from ouroboros.contracts.task_contract import build_task_contract
        from ouroboros.tools.control_delegation import stamp_depth_provenance

        task["delegation_role"] = "subagent"
        task["parent_cognitive_route"] = {"model": "same-model", "effort": "high"}
        # This is a newly admitted child, not a legacy row lacking depth authority.
        contract = build_task_contract(task)
        contract["delegation_budget"]["depth_remaining"] = 3
        task["task_contract"], _ = stamp_depth_provenance(
            contract, attempted_depth=task["depth"], max_depth=3)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert len(sent) == 1 and not host.pending
    task = host.running["child1"]["task"]
    from ouroboros.agent import OuroborosAgent

    if saved:
        from ouroboros import subagents
        from ouroboros.agent_dispatch import resolve_dispatch_axes

        # A real first worker resolves its profile before starting the leaf and
        # recording the authority fingerprint; restore must compare that binding.
        with monkeypatch.context() as first_dispatch:
            first_dispatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
            first_dispatch.setattr(subagents, "route_health", lambda *_a, **_kw: ("", ""))
            assert resolve_dispatch_axes(task).executor == "harness"
    OuroborosAgent._persist_running_record(SimpleNamespace(env=SimpleNamespace(drive_root=host.root)), task)
    assert task["admitted_dispatch"] == "possible"
    assert load_task_result(host.root, "child1")["admitted_dispatch"] == "possible"
    from ouroboros.contracts.task_contract import build_task_contract
    initial_authority = build_task_contract(task)
    process = InertProcess()
    from supervisor import worker_pool_lifecycle
    for module in (workers, worker_pool_lifecycle):  # an inert handle's fake PID reaches no OS process
        monkeypatch.setattr(module, "kill_worker_tree", kill_inert)
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
    if saved:
        from tests.test_working_checkpoint import _saved_working_state

        _saved_working_state(host.root, monkeypatch, "child1")
    if len(options) > 2 and options[2] == "resumed":
        from ouroboros.budget_pause import STATE_RESUMED, set_budget_pause

        # The loop has consumed its explicit Resume and is now supervising run-1.
        grant = {"grant_id": "spent-resume", "authority": "explicit_resume", "generation": 1,
                 "consumed_at": "2026-10-20T00:00:00Z", "selected_by": "owner"}
        source = {"path": "saved-source", "sha256": "a" * 64}
        set_budget_pause(host.root, "child1", {"state": STATE_RESUMED, "pause_id": "prior-pause",
            "source_ref": source, "grant": grant, "task_attempt": 1, "resume_generation": 1})
        task["_budget_pause_resume"] = {"pause_id": "prior-pause", "grant_id": "spent-resume",
            "source_ref": source, "authority": "explicit_resume", "grant_generation": 1,
            "pause": {"exact_continuation": True, "checkpoint": {"pause_id": "prior-pause"}}}
    selected = recovery.prepare_planned_restart_handoffs(host.root, host.running)
    assert selected == {"child1"}
    handoff = recovery._read(host.root, "child1")
    if saved:
        from supervisor.restart_retention import prepare_restart_returns

        assert prepare_restart_returns(host.root, host.running, host.pending,
            transaction_id=handoff["restart_transaction_id"]) == {"child1"}
        if options[2] == "saved_unreadable":
            from ouroboros.working_checkpoint import checkpoint_path

            checkpoint_path(host.root, "child1", task["_attempt"]).write_text("{unreadable", encoding="utf-8")
    assert workers.kill_workers(
        terminal_status="cancelled", result_reason="Planned self-restart", preserve_pending=True,
        preserve_running_task_ids=selected, reconcile_delegate_custody=False, archive_service_logs=False,
    )
    assert not process.is_alive() and not host.running
    cleanups = [json.loads(line) for line in (host.root / "logs/supervisor.jsonl").read_text().splitlines()
                if json.loads(line).get("type") == "zombie_prevention_cleanup"]
    assert cleanups[-1]["preserved_successors"] == ["child1"]
    assert host.pending[0]["_attempt"] == handoff["new_attempt"]
    if len(options) > 2 and options[2] == "resumed":
        assert host.pending[0]["_budget_pause_resume"]["grant_id"] == "spent-resume"
        assert workers.kill_workers(terminal_status="cancelled", result_reason="Second stop door",
                                    reconcile_delegate_custody=False, archive_service_logs=False)
        assert len(host.pending) == 1
    assert queue.persist_queue_snapshot()
    monkeypatch.setattr(recovery.os, "getpid", lambda: handoff["supervisor_pid"] + 100_000)
    monkeypatch.setattr(recovery, "IS_WINDOWS", True)
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    monkeypatch.setenv(recovery.PLANNED_RESTART_TRANSACTION_ENV, handoff["restart_transaction_id"])
    windows_successor(monkeypatch, host.root, handoff["restart_transaction_id"])
    return SimpleNamespace(host=host, task=task, handoff=handoff, initial_authority=initial_authority)


def restore(case):
    case.host.pending.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert case.host.pending[0]["_project_admission_restore_hold"]


@pytest.mark.parametrize("prepared", [False, True, (False, False, "resumed"), (True, False, "resumed")],
                         indirect=True, ids=["unscoped", "project", "resumed-unscoped", "resumed-project"])
@pytest.mark.parametrize("adopt_before_restore", [False, True], ids=["startup-order", "pre-adopted-snapshot"])
def test_exact_planned_handoff_reaches_worker_once(prepared, monkeypatch, adopt_before_restore):
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.subagent_bootstrap import _adopt_recovery_handoff
    from ouroboros.tools.registry import ToolContext
    import ouroboros.tools.delegate as delegate

    case = prepared
    assert "_working_recovery" not in case.host.pending[0], "no saved cognition keeps the existing handoff path"
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


@pytest.mark.parametrize("prepared", [False, True], indirect=True)
@pytest.mark.parametrize("reason", ["stale_queue_row_grant_consumed", "stale_resume_grant_superseded"])
@pytest.mark.parametrize("basis", ["valid", "foreign", "unreadable"])
def test_legacy_spent_hold_is_redecided_on_first_restore(prepared, monkeypatch, reason, basis):
    from supervisor.events_budget import hold_budget_row
    from ouroboros.work_stop import continuation_refusal

    case = prepared
    task = case.host.pending[0]
    hold_budget_row(task, reason=reason, extra={"pause_id": "prior-pause", "grant_id": "spent-resume"})
    assert continuation_refusal(task)["grant_state"] in {"consumed", "superseded"}
    assert "_budget_pause_resume" not in task and "_budget_pause_consumed" not in task
    path = recovery._path(case.host.root, "child1")
    if basis == "foreign":
        row = recovery._read(case.host.root, "child1")
        row["new_attempt"] += 1
        atomic_write_json(path, row)
    elif basis == "unreadable":
        path.write_text("{torn", encoding="utf-8")
    assert queue.persist_queue_snapshot()
    case.host.pending.clear()
    count = queue.restore_pending_from_snapshot()
    if basis == "foreign":
        assert count == 0 and not case.host.pending
        from ouroboros.cancel_intents import has_active_intent
        assert has_active_intent(case.host.root, "child1")
        return
    assert count == 1
    if basis == "unreadable":
        assert continuation_refusal(case.host.pending[0])["reason"] == reason
        return
    assert not continuation_refusal(case.host.pending[0])
    assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    sent = worker(case.host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert len(sent) == 1 and sent[0]["_attempt"] == case.handoff["new_attempt"]
    assert custody.open_runs(case.host.root)[0].run_id == "run-1"


@pytest.mark.parametrize("prepared", [False, True], indirect=True, ids=["unscoped", "project"])
@pytest.mark.parametrize("fault", ["bad_proof", "ordinary_restart", "panic"])
def test_unproven_or_owner_stopped_restart_never_assigns(prepared, monkeypatch, fault):
    case = prepared
    if fault == "bad_proof":
        windows_successor(monkeypatch, case.host.root, case.handoff["restart_transaction_id"], binding={})
    elif fault == "ordinary_restart":
        # A new manual Restart supersedes the plan through its own transaction;
        # an unbound old flag cannot rename an already prepared update.
        recovery.prepare_owner_restart(case.host.root)
        monkeypatch.setattr(recovery, "_PROCESS_GENERATION", "successor")
    else:
        (case.host.root / "state/panic_stop.flag").write_text("panic", encoding="utf-8")
    restore(case)
    if fault == "panic":
        from ouroboros.work_stop import current_stop
        assert current_stop(case.host.pending[0])["door"] == "panic"
        assert current_stop(case.host.pending[0])["release"] == "resume"
        # The installation input is no longer needed by the adoption decision.
        (case.host.root / "state/panic_stop.flag").unlink()
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
        row["holders"][0][key] = "changed"
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
    elif fault == "owner_restart":
        from tests._work_stop_helpers import live_restart
        live_restart(case.host.root, monkeypatch)
    elif fault == "panic":
        from ouroboros.work_stop import stop_record
        task["_budget_pause_hold"] = stop_record("panic_hold", held_at="test")
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


# --- one planned restart transaction carrying every returning kind --------------------------


@pytest.mark.parametrize("prepared", [(False, True), (True, True)], indirect=True,
                         ids=["unscoped", "project"])
@pytest.mark.parametrize("entry", ["tick", "assignment"])
def test_held_planned_handoff_expires_without_dispatch(prepared, expiry, monkeypatch, entry):  # noqa: F811
    """The deadline is part of admission authority, not edited after the handoff was prepared."""
    case = prepared
    assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    restore(case)
    task = case.host.pending[0]
    assert task["admitted_dispatch"] == "possible" and recovery.planned_handoff_resume_allowed(case.host.root, task)
    now = queue._task_deadline_ts(task) + 1
    monkeypatch.setattr("time.time", lambda: now)
    sent = worker(case.host, monkeypatch)
    # Keep the real terminal event/outbox path visible after installing the worker double.
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=expiry.frames.append))
    queue.enforce_task_timeouts() if entry == "tick" else workers.assign_tasks()
    workers.assign_tasks()
    expiry.reap()
    assert not sent and not queue.PENDING and not queue.RUNNING and not case.host.attempts
    assert load_task_result(case.host.root, "child1")["reason_code"] == "deadline"
    assert [e["status"] for e in expiry.frames if e.get("type") == "task_done"] == ["failed"]


@pytest.mark.parametrize("prepared", [(True, True)], indirect=True, ids=["project"])
@pytest.mark.parametrize("fault", ["transaction", "handoff", "pause", "governing"])
def test_held_planned_expiry_keeps_unproven_authority(prepared, expiry, monkeypatch, fault):  # noqa: F811
    case = prepared
    assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    restore(case)
    task = case.host.pending[0]
    now = queue._task_deadline_ts(task) + 1
    monkeypatch.setattr("time.time", lambda: now)
    if fault == "transaction":
        recovery._restart_transaction_path(case.host.root, case.handoff["restart_transaction_id"]).write_text("{torn")
    elif fault == "handoff":
        recovery._path(case.host.root, "child1").write_text("{torn")
    elif fault == "pause":
        task["_owner_hold"] = {"reason": "owner"}
    else:
        task["_working_recovery"] = {"source_task_id": "child1", "from_attempt": 0, "cause": "restart"}
        assert recovery.planned_handoff_resume_allowed(case.host.root, task)
    sent = worker(case.host, monkeypatch)
    queue.enforce_task_timeouts()
    workers.assign_tasks()
    expiry.reap()
    assert not sent and not case.host.attempts and queue.PENDING == [task]
    assert load_task_result(case.host.root, "child1").get("reason_code") != "deadline"


def _expire_held(case, expiry, monkeypatch, entry="tick"):  # noqa: F811
    """Elapse the admitted deadline, run one entry, assignment and the reaper; return puts and task_done frames."""
    now = queue._task_deadline_ts(case.host.pending[0]) + 1
    monkeypatch.setattr("time.time", lambda: now)
    sent = worker(case.host, monkeypatch)
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=expiry.frames.append))
    queue.enforce_task_timeouts() if entry == "tick" else workers.assign_tasks()
    workers.assign_tasks()
    expiry.reap()
    assert load_task_result(case.host.root, "child1").get("reason_code") != "deadline"
    return sent, [(e.get("status"), e.get("reason_code")) for e in expiry.frames if e.get("type") == "task_done"]


@pytest.mark.parametrize("prepared", [(False, True), (True, True)], indirect=True, ids=["unscoped", "project"])
@pytest.mark.parametrize("entry", ["tick", "assignment"])
@pytest.mark.parametrize("producer", ["launch_claimed", "worker_started"])
def test_held_planned_expiry_needs_its_attempt_unclaimed(prepared, expiry, monkeypatch, entry,  # noqa: F811
                                                         producer):
    """The planned grant expires only its never-handed attempt: the launcher's pre-put stamp or
    the worker's own start record of that attempt keeps the existing custody."""
    from ouroboros.agent import OuroborosAgent
    from supervisor.task_admission import record_project_dispatch_possible

    case = prepared
    assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    restore(case)
    task = case.host.pending[0]
    if producer == "launch_claimed":
        assert record_project_dispatch_possible(task)
    else:
        OuroborosAgent._persist_running_record(SimpleNamespace(env=SimpleNamespace(drive_root=case.host.root)), task)
    assert recovery.planned_handoff_resume_allowed(case.host.root, task)
    sent, done = _expire_held(case, expiry, monkeypatch, entry)
    assert not sent and not done and not case.host.attempts and queue.PENDING == [task]
    assert task["_project_admission_restore_hold"]


@pytest.mark.parametrize("prepared", [(False, True), (True, True)], indirect=True, ids=["unscoped", "project"])
@pytest.mark.parametrize("entry", ["tick", "assignment"])
@pytest.mark.parametrize("carrier", ["working", "pause", "owner_wait"])
def test_expired_planned_grant_never_stands_in_for_a_carrier(prepared, expiry, monkeypatch, entry,  # noqa: F811
                                                              carrier):
    """Expiry judges the carrier the loop resumes, as dispatch does; an invalid one has no planned fallback."""
    case = prepared
    restore(case)
    assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    task = case.host.pending[0]
    task.update({
        "working": {"_working_recovery": {"source_task_id": "child1", "from_attempt": 0, "cause": "restart"}},
        "pause": {"_budget_pause_resume": {"pause_id": "p-old", "grant_id": "g-old"}},
        "owner_wait": {"_owner_wait_resume": {"wait_id": "w-old", "restart_transaction_id": "another"}},
    }[carrier])
    assert recovery.planned_handoff_resume_allowed(case.host.root, task)
    sent, done = _expire_held(case, expiry, monkeypatch, entry)
    assert not sent and not case.host.attempts and not queue.RUNNING
    if carrier != "owner_wait":
        assert not done and queue.PENDING == [task] and task["_project_admission_restore_hold"]
    else:  # a refused cold wait follows its own shutdown custody, never the deadline
        assert done in ([], [("cancelled", "cancelled")])


@pytest.mark.parametrize("prepared", [(False, True), (True, True)], indirect=True, ids=["unscoped", "project"])
@pytest.mark.parametrize("change", ["panic", "owner_restart", "stop", "vetoed"])
@pytest.mark.parametrize("during", [False, True], ids=["before-tick", "during-custody"])
def test_held_planned_expiry_rereads_controls_and_handoff(prepared, expiry, monkeypatch, change, during):  # noqa: F811
    """Panic, a no-resume restart, Stop or a vetoed handoff wins before marking or while custody runs off Q."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.work_stop import stop_record

    case = prepared
    root = case.host.root
    assert recovery.pre_adopt_planned_handoffs(root, case.host.pending) == {"child1"}
    restore(case)
    task = case.host.pending[0]

    def alter(*_a, **_k):
        if change == "panic":
            task["_budget_pause_hold"] = stop_record("panic_hold", held_at="test")
        elif change == "owner_restart":
            from tests._work_stop_helpers import live_restart
            live_restart(root, monkeypatch)
        elif change == "stop":
            request_cancel(root, "child1", source="test", reason="Stop the held planned task")
        else:
            recovery.veto_handoff(root, "child1", "test_veto")

    if during:
        monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", alter)
    else:
        alter()
    sent, done = _expire_held(case, expiry, monkeypatch)
    assert not sent and not done and not case.host.attempts and queue.PENDING == [task]
    assert bool(task.get("_terminalization_retry")) is (during or change == "stop")


@pytest.fixture
def composed(host, tmp_path, monkeypatch):  # noqa: F811
    """As server._perform_supervisor_restart: owner wait, planned delegate session, S1 saved work and queue.

    The Project-scoped configured session is held for Project verification at restore; the
    others ride the same transaction. The successor is a simulated Windows spawn.
    """
    import time

    import ouroboros.claudexor_daemon as daemon
    from ouroboros import owner_wait
    from ouroboros.agent import OuroborosAgent
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_retention import _queued
    from tests.test_restart_saved_work import _working_run

    task = accepted(host, tmp_path, tid="child1")
    task["configured_subagent"] = _snapshot(_settings(_session_row()), "session-builder")
    worker(host, monkeypatch)
    workers.assign_tasks()
    task = host.running["child1"]["task"]
    OuroborosAgent._persist_running_record(SimpleNamespace(env=SimpleNamespace(drive_root=host.root)), task)
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
    # A native owner question parked mid-run.
    started = time.time() - 600
    asker = {"id": "asker", "type": "task", "chat_id": 1, "_attempt": 3, "text": "Ask, then continue.", "depth": 0}
    write_task_result(host.root, "asker", "running", total_rounds=7)
    ctx = SimpleNamespace(task_id="asker", task_attempt=3, drive_root=host.root, budget_drive_root=str(host.root),
                          task_started_at=started, _owner_wait_requested="quiz-a",
                          _owner_directives=[{"text": asker["text"]}], active_model="fixture-model",
                          active_effort="high", active_use_local=False, active_context_mode="max")
    wait = owner_wait.checkpoint_owner_wait(ctx, [{"role": "assistant", "content": "asked"}], {}, {}, 7, [], set())
    wait = owner_wait.set_owner_wait(host.root, "asker", {**wait, "state": "waiting"})
    host.running["asker"] = {"task": asker, "worker_id": 1, "attempt": 3, "started_at": started,
                             "last_heartbeat_at": time.time(), "owner_wait": wait}
    # S1 saved work and the formerly runnable queue.
    _working_run(host.root, workers, "saved-root")
    write_task_result(host.root, "queued", STATUS_SCHEDULED, chat_id=0)
    host.pending.append(_queued("queued", root_task_id="queued"))
    monkeypatch.setattr(workers, "WORKERS", {
        wid: workers.Worker(wid, InertProcess(), SimpleNamespace(), busy_task_id=tid, active_capacity=False)
        for wid, tid in enumerate(("child1", "asker", "saved-root"))})
    transaction_id = "composed-restart"
    owner_waits = owner_wait.prepare_owner_wait_handoffs(host.root, host.running, transaction_id)
    planned = recovery.prepare_planned_restart_handoffs(
        host.root, host.running, restart_transaction_id=transaction_id, additional_task_ids=owner_waits)
    assert owner_waits == {"asker"} and planned == {"asker", "child1"}
    assert prepare_restart_returns(host.root, host.running, list(host.pending), transaction_id=transaction_id,
                                   owner_wait_ids=owner_waits) == {"saved-root"}
    assert workers.kill_workers(
        force=True, terminal_status="cancelled", result_reason="Planned self-restart", stop_source="server_shutdown",
        preserve_running_task_ids=planned, retain_saved_work=True, preserve_pending=True,
        reconcile_delegate_custody=False, archive_service_logs=False)
    assert queue.persist_queue_snapshot(reason="pre_restart_exit")
    parent_pid = recovery._read_restart_transaction(host.root, transaction_id)["supervisor_pid"]
    monkeypatch.setattr(recovery.os, "getpid", lambda: parent_pid + 100_000)  # the spawned successor
    monkeypatch.setattr(recovery, "IS_WINDOWS", True)
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    monkeypatch.setenv(recovery.PLANNED_RESTART_TRANSACTION_ENV, transaction_id)
    return SimpleNamespace(host=host, transaction_id=transaction_id, parent_pid=parent_pid, wait=wait)


def _free_workers(host, monkeypatch, count=4):  # noqa: F811
    from supervisor import state

    sent = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None))
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(workers, "WORKERS", {wid: affirm_ready(SimpleNamespace(
        wid=wid, busy_task_id=None, reaping=False, in_q=SimpleNamespace(put=lambda row: sent.append(row) if row.get("type") != "worker_quiescence" else None))) for wid in range(count)})
    return sent


@pytest.mark.parametrize("variant", ["exit42", "exit99", "later_process", "intermediate_restart",
                                     "stale_preparation", "consumed"])
def test_one_restart_returns_saved_work_planned_session_owner_wait_and_project_hold(composed, monkeypatch,
                                                                                    variant):
    """S1 returns saved work/queue only to the exact successor; retained carriers use their own gates."""
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
    from supervisor.restart_retention import fresh_return_transaction

    case = composed
    root = case.host.root
    parent, binding = {"pid": case.parent_pid, "birth": "win-filetime:1",
                       "exit_code": 99 if variant == "exit99" else 42}, None
    if variant == "intermediate_restart":
        # The bound successor restarted before acknowledging (a boot resuming a pending store
        # switch): the next generation observes ITS exit 42, never the prepared parent's.
        bound = case.parent_pid + 100_000
        parent = {"pid": bound, "birth": "win-filetime:2", "exit_code": 42}
        binding = {"supervisor_birth": "win-filetime:1", "successor_pid": bound, "successor_birth": "win-filetime:2"}
    windows_successor(monkeypatch, root, case.transaction_id, parent=parent, binding=binding)
    if variant == "intermediate_restart":
        monkeypatch.setattr(recovery.os, "getpid", lambda: case.parent_pid + 200_000)
    if variant in {"stale_preparation", "consumed"}:
        if variant == "consumed":
            row = recovery._read_restart_transaction(root, case.transaction_id)
            recovery._write_restart_transaction(root, {**row, "returns_restored_at": "an-earlier-boot"})
        else:
            active = recovery._active_restart_transaction_path(root)
            atomic_write_json(active, {**json.loads(active.read_text()), "prepared_at": "a-later-preparation"})
    if variant == "later_process":
        # The bound successor acknowledged at its boot, then a later process restores.
        assert fresh_return_transaction(root)["ack_source"] == recovery.WINDOWS_PARENT_ACK
        monkeypatch.setattr(recovery.os, "getpid", lambda: case.parent_pid + 200_000)
    case.host.pending.clear()
    queue.restore_pending_from_snapshot()
    pre_adopted = recovery.pre_adopt_planned_handoffs(root, case.host.pending)
    rows = {row["id"]: row for row in case.host.pending}
    assert set(rows) == {"child1", "asker", "saved-root", "queued"}
    assert rows["child1"]["_project_admission_restore_hold"]  # Project custody, never an unverified replay
    transaction = recovery._read_restart_transaction(root, case.transaction_id)
    sent = _free_workers(case.host, monkeypatch)
    for _ in range(3):
        workers.assign_tasks()
    dispatched = sorted(row["id"] for row in sent)
    if variant == "exit42":
        assert transaction["status"] == "normal_exit_acknowledged"
        assert transaction["ack_source"] == recovery.WINDOWS_PARENT_ACK and transaction["returns_restored_at"]
        assert pre_adopted == {"child1"}
        assert dispatched == ["asker", "child1", "queued", "saved-root"]  # each exactly once
        by_id = {row["id"]: row for row in sent}
        assert by_id["asker"]["_owner_wait_resume"]["wait_id"] == case.wait["wait_id"]
        assert by_id["saved-root"]["_working_recovery"]["cause"] == "restart"
        assert not by_id["child1"].get("_project_admission_restore_hold")
        assert recovery.planned_handoff_resume_allowed(root, by_id["child1"])
        assert fresh_return_transaction(root) == {}, "the returns are consumed by this one boot"
        return
    # S1's saved work and queue are never lent to an unproven, stale, consumed or later reader.
    for task_id in ("saved-root", "queued"):
        assert budget_hold_fact(rows[task_id])["reason"] == HOLD_SAVED_WORK
        assert task_id not in dispatched
    if variant == "later_process":
        assert transaction["status"] == "normal_exit_acknowledged" and not transaction.get("returns_restored_at")
        # The owner wait keeps its own gate; D1's holder needs the exact Windows successor.
        assert pre_adopted == set()
        assert dispatched == ["asker"]
        by_id = {row["id"]: row for row in sent}
        assert by_id["asker"]["_owner_wait_resume"]["wait_id"] == case.wait["wait_id"]
        assert rows["child1"]["_project_admission_restore_hold"]
        assert not recovery.planned_handoff_resume_allowed(root, rows["child1"])
        assert recovery._read(root, "child1")["veto_reason"] == "restart_successor_generation_mismatch"
        assert set(row["id"] for row in case.host.pending) == {"child1", "saved-root", "queued"}
        assert fresh_return_transaction(root) == {}
        return
    assert transaction["status"] == "prepared" and pre_adopted == set()
    assert recovery._read(root, "child1")["veto_reason"] == "restart_normal_exit_unproven"
    assert budget_hold_fact(rows["asker"])["reason"] == HOLD_SAVED_WORK and not rows["asker"].get("_owner_wait_resume")
    assert dispatched == []


@pytest.mark.parametrize("prepared", [True], indirect=True, ids=["project"])
@pytest.mark.parametrize("carrier", ["working", "pause", "owner_wait"])
def test_planned_authority_never_stands_in_for_the_loops_own_carrier(prepared, monkeypatch, carrier):
    """The loop resumes its first present carrier; an unprovable one keeps the Project hold.

    Producer trace: a preserved successor copies its dispatched row (workers.kill_workers
    ``dict(task)``), and dispatch pops neither an older saved-work locator nor a consumed
    pause grant; an owner wait is added by the same planned restart. Planned authority
    releases only a row that carries none of them (#1716 governing-carrier seam).
    """
    case = prepared
    restore(case)
    assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    task = case.host.pending[0]
    task.update({
        "working": {"_working_recovery": {"source_task_id": "child1", "from_attempt": 0, "cause": "restart"}},
        "pause": {"_budget_pause_resume": {"pause_id": "p-old", "grant_id": "g-old"}},
        "owner_wait": {"_owner_wait_resume": {"wait_id": "w-old", "restart_transaction_id": "another"}},
    }[carrier])
    assert recovery.planned_handoff_resume_allowed(case.host.root, task)  # the planned proof itself is valid
    sent = worker(case.host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert not sent and not case.host.running
    if carrier != "owner_wait":  # a refused cold wait follows its own shutdown custody
        [held] = case.host.pending
        assert held["_project_admission_restore_hold"]
