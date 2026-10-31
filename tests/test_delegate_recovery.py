"""Saved supervision adopts local identity before observing a leaf engine."""
import json
from types import SimpleNamespace

import pytest

from tests.test_configured_session_restore import InputReached
from tests.test_delegate_pending_recovery import _pending_handoff
from tests.test_planned_delegate_assignment import prepared as prepared, restore  # noqa: F401
from tests.test_swarm_host_admission import host  # noqa: F401
from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("door", ["self_restart", "managed_update"])
def test_planned_restart_doors_name_saved_delegates_as_work_returns(tmp_path, monkeypatch, door,
                                                                  stub_delegate_engines):
    """Native waits use their own carrier; saved delegates still need a working return."""
    import server
    from ouroboros import body_adoption, delegate_recovery as recovery, owner_wait
    from ouroboros.gateway import control
    from supervisor import git_ops, queue, restart_retention as retention, workers
    from tests.test_working_checkpoint import _saved_working_state

    _saved_working_state(tmp_path, monkeypatch, "child1")
    running = {tid: {"task": {"id": tid, "_attempt": 1}, "attempt": 1} for tid in ("child1", "native-wait")}
    monkeypatch.setattr(workers, "RUNNING", running)
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "direct_chat_turns", lambda: [])
    monkeypatch.setattr(owner_wait, "prepare_owner_wait_handoffs", lambda *_a: {"native-wait"})
    monkeypatch.setattr(recovery, "prepare_planned_restart_handoffs", lambda *_a, **_k: set(running))
    prepare_returns = retention.prepare_restart_returns
    observed = []
    def prepared(*args, **kwargs):
        returns = prepare_returns(*args, **kwargs)
        transaction = recovery._read_restart_transaction(tmp_path, kwargs["transaction_id"])
        observed.append((kwargs["owner_wait_ids"], returns, transaction))
        raise InputReached  # Stop before any process teardown or restart request.
    monkeypatch.setattr(retention, "prepare_restart_returns", prepared)
    if door == "self_restart":
        monkeypatch.setattr(body_adoption, "bind_restart", lambda fn, *_a: fn)
        monkeypatch.setattr(server, "_safe_restart_serialized", lambda *_a, **_k: (True, "ok"))
        monkeypatch.setattr(server, "_begin_task_teardown", lambda **_k: ("cancelled", "fixture restart"))
        monkeypatch.setattr(server, "_planned_delegate_restart_transaction_id", "")
        monkeypatch.setattr(server, "_request_restart_exit", lambda: pytest.fail("no real restart"))
        monkeypatch.setattr(queue, "name_queue_returns", lambda pending, _reason: pending)
        ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[],
                              load_state=lambda: {}, safe_restart=lambda **_k: (True, "ok"),
                              kill_workers=lambda **_k: pytest.fail("no process teardown"))
        with pytest.raises(InputReached):
            server._perform_supervisor_restart(ctx)
    else:
        monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(workers, "close_repo_writer_admission", lambda *_a: None)
        monkeypatch.setattr(workers, "open_repo_writer_admission", lambda: None)
        monkeypatch.setattr(workers, "drain_repo_writers", lambda: [])
        monkeypatch.setattr(workers, "kill_workers_for_update", lambda **_k: pytest.fail("no process teardown"))
        monkeypatch.setattr(control.update_progress, "advance", lambda *_a: None)
        monkeypatch.setattr(retention, "capture_update_returns", lambda *_a, **_k: {"rows": {}})
        with pytest.raises(InputReached):
            control._stop_repo_writers("update")
    waits, returns, transaction = observed[0]
    assert waits == {"native-wait"} and returns == {"child1"}
    assert transaction["return_ids"] == ["child1"] and set(transaction["task_ids"]) == set(running)
    assert not stub_delegate_engines


def _bound(root, task, invocation, *, terminal=False):
    from ouroboros import delegate_custody as custody, delegate_recovery as recovery

    row = recovery._read(root, task["id"])
    assert custody.record_started(root, custody.RunCustody(
        run_id="bound-run", invocation_id=invocation, task_id=task["id"], route_id="codex",
        **{key: row[key] for key in ("selected_subagent_id", "config_fingerprint", "authority_fingerprint",
                                    "work_order_fingerprint", "snapshot_id", "execution_root",
                                    "baseline_sha", "target_root")}))
    if terminal:
        assert custody.emit(root, custody.SETTLED, {"run_id": "bound-run", "task_id": task["id"], "state": "succeeded"})


@pytest.mark.parametrize("transition", ["pending", "bound", "settled"])
def test_saved_pending_handoff_keeps_reservation_and_addressed_late_fate(
        tmp_path, monkeypatch, stub_delegate_engines, transition):
    custody, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, "nanny")
    task["_working_recovery"] = {}  # Invalid locator is still saved cognition, never a replay permit.
    row = recovery._read(tmp_path, "nanny")
    row.update(journal_cursor=17, interaction_acknowledged_ids=["earlier-q"],
               coordination_cursor={"child": "v1"}, pending_wake={"wake_id": "kept", "payload": {
                   "status": "waiting_on_user", "supervision_wake_id": "kept"}})
    recovery._write(tmp_path, row)
    if transition != "pending":
        _bound(tmp_path, task, invocation, terminal=transition == "settled")
    monkeypatch.setattr("ouroboros.tools.delegate.exact_start", lambda *_a, **_k: pytest.fail("automatic pending POST"))
    ctx = SimpleNamespace(task_id="nanny", task_attempt=2, drive_root=tmp_path, budget_drive_root=tmp_path)
    adopted = recovery.adopt_handoff(ctx, task)
    assert adopted["status"] == "adopted" and adopted["liveness"] == "unobserved"
    assert not stub_delegate_engines
    stored = recovery._read(tmp_path, "nanny")
    assert stored["status"] == "adopted" and stored["pending_invocation_id"] == invocation
    if transition == "pending":
        assert adopted["pending_invocation_id"] == adopted["retry_of"] == invocation
        assert not adopted.get("run_id")
        assert not (tmp_path / "state/delegate_supervision/nanny.json").exists()
        assert stored["journal_cursor"] == 17 and stored["pending_wake"]["wake_id"] == "kept"
        _bound(tmp_path, task, invocation)
        from ouroboros.delegate_supervision import supervised_wait, _load_state
        result = supervised_wait(ctx, "bound-run", wait_once=lambda *_a, **_k: pytest.fail("saved wake comes first"))
        assert json.loads(result.text)["supervision_wake_id"] == "kept"
        assert _load_state(ctx, "bound-run")["journal_cursor"] == 17
    else:
        assert adopted["run_id"] == "bound-run"
        assert adopted["settled"] is (transition == "settled")
        assert recovery._read(tmp_path, "nanny")["run_id"] == "bound-run"
    assert not stub_delegate_engines


@pytest.mark.parametrize("transition", ["live", "settled"])
def test_saved_run_handoff_accepts_settlement_since_reservation(tmp_path, monkeypatch, transition):
    custody, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, "nanny")
    _bound(tmp_path, task, invocation)
    row = recovery._read(tmp_path, "nanny")
    recovery._write(tmp_path, {**row, "pending_invocation_id": "", "run_id": "bound-run"})
    task["_working_recovery"] = {}
    if transition == "settled":
        custody.emit(tmp_path, custody.SETTLED, {"task_id": "nanny", "run_id": "bound-run", "state": "succeeded"})
    monkeypatch.setattr(recovery, "_run_probe_error", lambda *_a: pytest.fail("no pre-restore probe"))
    ctx = SimpleNamespace(task_id="nanny", task_attempt=2, drive_root=tmp_path, budget_drive_root=tmp_path)
    result = recovery.adopt_handoff(ctx, task)
    assert result["status"] == "adopted" and result["run_id"] == "bound-run"
    assert result["settled"] is (transition == "settled")


@pytest.mark.parametrize("fault", ["attempt", "authority", "holder", "generation", "cause"])
def test_saved_carrier_does_not_relax_local_handoff_authority(tmp_path, monkeypatch, fault):
    custody, recovery, task, _, _, _, _ = _pending_handoff(tmp_path, "nanny")
    task["_working_recovery"] = {}
    row = recovery._read(tmp_path, "nanny")
    if fault == "attempt":
        task["_attempt"] += 1
    elif fault == "authority":
        row["authority_fingerprint"] = "foreign"
    elif fault == "holder":
        row["snapshot_id"] = "foreign"
    elif fault == "generation":
        row["supervisor_pid"] += 1
    else:
        row["cause"] = "owner_restart"
    recovery._write(tmp_path, row)
    reconciled = []
    monkeypatch.setattr(custody, "reconcile_task_runs", lambda *_a, **_k: reconciled.append(True))
    ctx = SimpleNamespace(task_id="nanny", task_attempt=task["_attempt"], drive_root=tmp_path, budget_drive_root=tmp_path)
    assert recovery.adopt_handoff(ctx, task)["status"] == "recovery_required"
    assert recovery._read(tmp_path, "nanny")["status"] == "vetoed" and reconciled == [True]


@pytest.mark.parametrize("fault", ["none", "no_resume", "panic", "transaction", "attempt", "holder", "old_generation"])
@pytest.mark.parametrize("fate", ["pending", "bound", "settled"])
def test_startup_pre_adoption_retains_local_checks_without_engine_observation(
        tmp_path, monkeypatch, stub_delegate_engines, fault, fate):
    custody, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, "nanny")
    if fate != "pending":
        _bound(tmp_path, task, invocation, terminal=fate == "settled")
    task["_working_recovery"] = {}
    row = recovery._read(tmp_path, "nanny")
    row.update(cause=recovery.CAUSE_PLANNED_SELF_RESTART, restart_transaction_id="tx", supervisor_pid=123)
    recovery._write(tmp_path, row)
    transaction = {"transaction_id": "tx", "supervisor_pid": 123, "status": "normal_exit_acknowledged",
                   "exit_code": 42, "task_ids": ["nanny"]}
    if fault == "transaction":
        transaction["exit_code"] = 1
    elif fault == "no_resume":
        recovery.prepare_owner_restart(tmp_path)
        monkeypatch.setattr(recovery, "_PROCESS_GENERATION", "successor")
    elif fault == "panic":
        # Startup restores the queue before pre-adoption; the work's stop,
        # not the consumed installation input, owns the decision here.
        from supervisor.restart_retention import hold_stopped_queue

        hold_stopped_queue(task, tmp_path, "panic")
    elif fault == "attempt":
        task["_attempt"] += 1
    elif fault == "holder":
        recovery._write(tmp_path, {**row, "snapshot_id": "foreign"})
    recovery._write_restart_transaction(tmp_path, transaction)
    monkeypatch.delenv(recovery.PLANNED_RESTART_TRANSACTION_ENV, raising=False)
    monkeypatch.setattr(recovery, "_pid_alive", lambda _pid: fault == "old_generation")
    monkeypatch.setattr(custody, "reconcile_task_runs", lambda *_a, **_k: None)
    adopted = recovery.pre_adopt_planned_handoffs(tmp_path, [task])
    assert not stub_delegate_engines
    if fault == "none":
        assert adopted == {"nanny"}
        assert recovery._read(tmp_path, "nanny")["status"] == "pre_adopted"
        assert recovery.planned_handoff_resume_allowed(tmp_path, task)
    else:
        assert not adopted and recovery._read(tmp_path, "nanny")["status"] == "vetoed"
        if fault in {"no_resume", "panic"}:
            assert recovery._read(tmp_path, "nanny")["veto_reason"] == "startup_work_stopped"


@pytest.mark.parametrize("prepared", [(False, False, "saved"), (True, False, "saved")],
                         indirect=True, ids=["unscoped", "project"])
@pytest.mark.parametrize("adopt_before_restore", [False, True], ids=["startup-order", "pre-adopted-snapshot"])
def test_planned_saved_task_crosses_startup_admission_bootstrap_and_real_loop(
        prepared, monkeypatch, stub_delegate_engines, adopt_before_restore):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros import claudexor_daemon, delegate_recovery as recovery, loop, subagent_bootstrap
    from ouroboros.working_checkpoint import checkpoint_path
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from supervisor import queue, workers
    from tests.test_project_hold_recovery import worker
    from tests.test_working_checkpoint import _registry
    from tests.test_loop_transport_wait import _loop_kwargs

    case = prepared
    root = case.host.root
    # The fixture saved cognition and prepared its return BEFORE the stop/ack;
    # adding a checkpoint after the stop correctly leaves an application hold.
    source = case.host.pending[0]["_working_recovery"]
    assert source["from_attempt"] == case.handoff["old_attempt"]
    assert source["source_task_id"] == "child1" and source["cause"] == "restart"
    assert read_actor_source_bytes(root, "child1", source["source_ref"]) == checkpoint_path(
        root, "child1", case.handoff["old_attempt"]).read_bytes()
    def no_engine(*_args, **_kwargs):
        pytest.fail("startup cannot admit the leaf before restoring its owner")
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", no_engine)
    if adopt_before_restore:
        assert recovery.pre_adopt_planned_handoffs(root, case.host.pending) == {"child1"}
        assert queue.persist_queue_snapshot()
    restore(case)
    assert case.host.pending[0]["_working_recovery"] == source
    if not adopt_before_restore:
        assert recovery.pre_adopt_planned_handoffs(root, case.host.pending) == {"child1"}
    assert recovery._read(root, "child1")["status"] == "pre_adopted"
    sent = worker(case.host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert len(sent) == 1 and not case.host.pending, json.dumps(case.host.pending, default=str, indent=2)
    task = sent[0]
    # Preserve the exact task/work-order binding recorded before the handoff.
    registry = _registry(root, monkeypatch, "child1", task["_attempt"])
    ctx = registry._ctx
    ctx.working_recovery = task["_working_recovery"]
    dispatch = resolve_dispatch_axes(task)
    from ouroboros.contracts.task_contract import build_task_contract
    assert build_task_contract(task) == case.initial_authority
    assert not recovery._successor_binding_mismatch(case.handoff, task)
    ctx._configured_startup_wake = subagent_bootstrap.bootstrap_before_context(ctx, task, dispatch)
    adoption = recovery._read(root, "child1")
    assert adoption["status"] == "adopted", json.dumps(adoption, default=str, indent=2)
    assert json.loads(ctx._configured_startup_wake)["recovery"]["run_id"] == "run-1"
    assert not stub_delegate_engines
    def input_reached(call, *_a, **_kw):
        assert not stub_delegate_engines, "no engine observation before restored cognition"
        text = str(call.messages)
        assert "Saved object 42" in text and "unobserved" in text and "run-1" in text
        raise InputReached
    monkeypatch.setattr(loop, "_dispatch_round_model", input_reached)
    kwargs = _loop_kwargs(root, registry, [])
    kwargs["task_id"] = "child1"
    with pytest.raises(InputReached):
        loop.run_llm_loop(**kwargs)
    # InputReached exits the loop: its ordinary cleanup may attach to observe
    # retained work, but must not ensure, provision, or start an engine.
    assert stub_delegate_engines == ["attach"]


@pytest.mark.parametrize("prepared", [(False, False, "saved_unreadable"), (True, False, "saved_unreadable")],
                         indirect=True, ids=["unscoped", "project"])
def test_planned_failed_cognition_freeze_keeps_source_gap_without_engine_admission(
        prepared, monkeypatch, stub_delegate_engines):
    from ouroboros import claudexor_daemon, delegate_recovery as recovery
    from ouroboros.working_checkpoint import recovery_source_for_task
    from supervisor import workers
    from tests.test_project_hold_recovery import worker

    case = prepared
    source = case.host.pending[0]["_working_recovery"]
    assert source == {"source_task_id": "child1", "from_attempt": case.handoff["old_attempt"], "cause": "restart"}
    assert not recovery_source_for_task(case.host.root, case.host.pending[0])
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway",
                        lambda: pytest.fail("an unreadable source cannot become a fresh leaf start"))
    restore(case)
    assert recovery.pre_adopt_planned_handoffs(case.host.root, case.host.pending) == {"child1"}
    sent = worker(case.host, monkeypatch)
    workers.assign_tasks()
    assert not sent and case.host.pending[0]["_working_recovery"] == source
    assert recovery._read(case.host.root, "child1")["status"] == "pre_adopted"
    assert not stub_delegate_engines


def test_unreadable_custody_keeps_saved_reservation_without_observation_or_veto(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current

    custody, recovery, task, _, _, _, _ = _pending_handoff(tmp_path, "nanny")
    task["_working_recovery"] = {}
    def unreadable(*_a, **_kw):
        raise OSError("controlled unavailable custody")
    monkeypatch.setattr(current, "task_snapshot", unreadable)
    monkeypatch.setattr(custody, "reconcile_task_runs", lambda *_a, **_k: pytest.fail("missing observation cannot veto"))
    ctx = SimpleNamespace(task_id="nanny", task_attempt=2, drive_root=tmp_path, budget_drive_root=tmp_path)
    result = recovery.adopt_handoff(ctx, task)
    assert result == {"status": "recovery_required", "reason": "custody_unreadable"}
    assert recovery._read(tmp_path, "nanny")["status"] == "reserved"


@pytest.mark.parametrize("binding", ["after_restart", "before_restart", "settled_before_restart"])
@pytest.mark.parametrize("aged_gc", [False, True])
def test_pending_wait_sources_survive_aged_gc_and_second_planned_restart(
        tmp_path, monkeypatch, binding, aged_gc, stub_delegate_engines):
    import os
    from ouroboros import delegate_supervision as ds
    from ouroboros.delegate_state_sweep import sweep_settled_delegate_state
    from ouroboros.working_checkpoint import flush_content_acks
    from tests.test_delegate_state_sweep import _OLD, _write_row
    from tests.test_sleep_questions import _deliver

    _, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, "nanny")
    task["_working_recovery"] = {}
    row = recovery._read(tmp_path, "nanny")
    payload = {"status": "waiting_on_user", "supervision_wake_id": "kept",
               "pending_interactions": [{"interaction_id": "still-unanswered", "question": "Which path?"}]}
    sources = {"journal_cursor": 17, "mailbox_acknowledged_ids": ["prior-mail"],
               "interaction_acknowledged_ids": ["prior-question"],
               "coordination_cursor": {"children": {"child": {"status": "completed"}}},
               "checkpoint": {"reason": "Review retained work", "consumed": True},
               "pending_wake": {"wake_id": "kept", "attempt_key": "1", "payload": payload,
                                "interaction_ids": ["still-unanswered"]}}
    recovery._write(tmp_path, {**row, **sources})
    ctx = SimpleNamespace(task_id="nanny", task_attempt=2, drive_root=tmp_path, budget_drive_root=tmp_path)
    assert recovery.adopt_handoff(ctx, task)["status"] == "adopted"
    adopted = recovery._read(tmp_path, "nanny")
    # A referenced transaction must survive beside the live pending source.
    adopted["restart_transaction_id"] = "first-tx"
    recovery._write(tmp_path, adopted)
    old_transaction = _write_row(tmp_path, "delegate_recovery_transactions", "first-tx", {"schema": 1})
    os.utime(recovery._path(tmp_path, "nanny"), (_OLD, _OLD))
    if aged_gc:
        assert not sweep_settled_delegate_state(tmp_path)["removed"]
    assert old_transaction.exists()
    assert not (tmp_path / "state/delegate_supervision/nanny.json").exists()
    if binding != "after_restart":
        _bound(tmp_path, task, invocation, terminal=binding == "settled_before_restart")
    # Exercise the planned selector too: an unbound adoption has no supervision
    # file, so selecting only sleeping files would silently miss its second stop.
    assert recovery.prepare_planned_restart_handoffs(
        tmp_path, {"nanny": {"task": task, "attempt": 2}}, restart_transaction_id="second-tx") == {"nanny"}
    reserved = recovery._read(tmp_path, "nanny")
    for key, value in sources.items():
        assert reserved[key] == value
    assert (reserved["old_attempt"], reserved["new_attempt"]) == (2, 3)
    if binding == "after_restart":
        assert recovery.prepare_planned_restart_handoffs(
            tmp_path, {"nanny": {"task": task, "attempt": 2}}, restart_transaction_id="second-tx") == {"nanny"}
        assert recovery._read(tmp_path, "nanny")["pending_wake"] == sources["pending_wake"]
    assert recovery.acknowledge_observed_restart_exit(tmp_path, supervisor_pid=os.getpid(), exit_code=42)
    task["_attempt"] = ctx.task_attempt = 3
    assert recovery.pre_adopt_planned_handoffs(tmp_path, [task]) == {"nanny"}
    assert recovery.adopt_handoff(ctx, task)["status"] == "adopted"
    if binding == "after_restart":
        _bound(tmp_path, task, invocation)
    result = ds.supervised_wait(ctx, "bound-run", wait_once=lambda *_a, **_k: pytest.fail("retained wake must replay first"))
    assert json.loads(result.text)["supervision_wake_id"] == "kept"
    state = ds._load_state(ctx, "bound-run")
    for key, value in sources.items():
        assert state[key] == value
    assert state["run_id"] == "bound-run"
    _deliver(ctx, result.text, monkeypatch, tool="delegate_wait", meta=result.meta)
    assert ds._load_state(ctx, "")["pending_wake"]
    assert ctx._sleep_interaction_seen == {"bound-run": ["still-unanswered"]}
    flush_content_acks(ctx)
    state = ds._load_state(ctx, "bound-run")
    assert not state["pending_wake"]
    assert state["interaction_acknowledged_ids"] == ["prior-question", "still-unanswered"]
    assert not stub_delegate_engines


@pytest.mark.parametrize("fault", ["task", "attempt", "authority", "holder", "custody_read", "row_read"])
def test_second_handoff_never_replaces_unproven_pending_wait_source(tmp_path, monkeypatch, fault):
    from ouroboros import delegate_custody_current as current

    _, recovery, task, _, _, _, _ = _pending_handoff(tmp_path, "nanny")
    task["_working_recovery"] = {}
    ctx = SimpleNamespace(task_id="nanny", task_attempt=2, drive_root=tmp_path, budget_drive_root=tmp_path)
    assert recovery.adopt_handoff(ctx, task)["status"] == "adopted"
    row = recovery._read(tmp_path, "nanny")
    row["journal_cursor"] = 17
    if fault in {"authority", "holder"}:
        row["authority_fingerprint" if fault == "authority" else "snapshot_id"] = "foreign"
    elif fault == "attempt":
        row["new_attempt"] = 9
    recovery._write(tmp_path, row)
    path = recovery._path(tmp_path, "nanny")
    if fault == "task":
        path.write_text(json.dumps({**row, "task_id": "foreign"}), encoding="utf-8")
    elif fault == "row_read":
        path.write_text("{torn", encoding="utf-8")
    elif fault == "custody_read":
        def unreadable(*_a, **_k):
            raise OSError("controlled unavailable custody")
        monkeypatch.setattr(current, "task_snapshot", unreadable)
    before = path.read_bytes()
    assert recovery.prepare_handoff(tmp_path, task, cause=recovery.CAUSE_WORKER_CRASH,
                                    old_attempt=2, new_attempt=3) == {}
    assert not recovery.prepare_planned_restart_handoffs(
        tmp_path, {"nanny": {"task": task, "attempt": 2}}, restart_transaction_id="never-prepared")
    assert path.read_bytes() == before
    assert not (tmp_path / "state/delegate_supervision/nanny.json").exists()


@pytest.mark.parametrize("path", ["adopted_pending", "recovery_required", "evidence_unreadable"])
def test_restored_handoff_grounds_prior_physical_runs_in_local_custody(tmp_path, monkeypatch, path,
                                                                     stub_delegate_engines):
    """Settled run A plus adopted pending B is physical history, never a fresh-start nudge."""
    from ouroboros import delegate_custody as custody, delegate_evidence, delegate_recovery as recovery
    from ouroboros import subagent_bootstrap
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.subagent_work_order import work_order_fingerprint
    from tests.test_configured_session_restore import _restored_registry, _task
    from tests.test_restored_readonly_zero_run import _receipt_args

    registry = _restored_registry(tmp_path, monkeypatch, "working")
    ctx = registry._ctx
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    task = _task(ctx, "working")
    fields = {"task_id": task["id"], "route": "codex",
              "selected_subagent_id": task["configured_subagent"]["selected_subagent_id"],
              "config_fingerprint": task["configured_subagent"]["config_fingerprint"],
              "authority_fingerprint": recovery.authority_fingerprint_from_task(task),
              "work_order_fingerprint": work_order_fingerprint(task)}
    for invocation in ("invoke-a", "invoke-b"):  # A ran and failed; B's start response was lost.
        assert custody.record_start_requested(tmp_path, run_id="", invocation_id=invocation, idempotency_key=invocation,
                                              max_seconds=60, request={"prompt": "canonical"}, **fields)
    assert custody.emit(tmp_path, custody.STARTED, {**fields, "invocation_id": "invoke-a", "run_id": "run-a"})
    assert custody.emit(tmp_path, custody.SETTLED, {**fields, "invocation_id": "invoke-a", "run_id": "run-a",
                                                  "state": "failed", "model": "stub-model"})
    handoff = recovery.prepare_worker_crash_handoff(tmp_path, task, old_attempt=1, new_attempt=2,
                                                    worker_id=1, exitcode=1)
    assert handoff["pending_invocation_id"] == "invoke-b" and not handoff["run_id"]
    dispatch = resolve_dispatch_axes(task)
    if path == "recovery_required":
        recovery._write(tmp_path, {**handoff, "snapshot_id": "foreign"})  # A proved different holder.
        monkeypatch.setattr(custody, "reconcile_task_runs", lambda *_a, **_k: None)
    elif path == "evidence_unreadable":
        def unreadable(*_a, **_k):
            raise OSError("controlled unreadable custody evidence")
        monkeypatch.setattr(delegate_evidence, "task_execution_evidence", unreadable)
    receipt = json.loads(subagent_bootstrap.bootstrap_before_context(ctx, task, dispatch))
    bootstrap = ctx._configured_actor_bootstrap
    fact = subagent_bootstrap.actor_first_unresolved_fact(ctx, task_id=task["id"], drive_root=tmp_path)
    note = subagent_bootstrap.configured_actor_finalization_message(ctx, task_id=task["id"], fallback_root=tmp_path)
    assert "Start the exact assigned session now" not in note, "no paid duplicate over known history"
    if path == "evidence_unreadable":
        assert receipt["status"] == "configured_session_continuation"
        assert bootstrap["physical_started"] is False and bootstrap["zero_run_evidence_status"] == "unknown"
        assert fact["reason"] == "zero_run_evidence_unavailable"
    else:
        assert receipt["status"] == {"adopted_pending": "configured_session_continuation",
                                     "recovery_required": "configured_session_recovery_wake"}[path]
        assert bootstrap["physical_started"] is True
        assert fact["reason"] == "delegated_runs_failed_without_success"
        assert registry.get_schema_by_name("verify_and_record") is None
        assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in registry.execute("verify_and_record", _receipt_args())
    assert not stub_delegate_engines


# A bad OLD holder (foreign binding, unreadable fate) no longer decides the fate of a valid sleeping
# sibling: holders are proven and preserved one by one (tests/test_delegate_plural_recovery.py).
@pytest.mark.parametrize("fate", ["refused", "bound_settled", "refused_running"])
@pytest.mark.parametrize("door", ["worker_crash", "planned_restart"])
def test_resolved_pending_reservation_yields_wait_sources_to_current_supervision(
        tmp_path, monkeypatch, stub_delegate_engines, fate, door):
    """Adopted pending B resolved without a restoring wait; replacement C owns supervision."""
    from ouroboros import delegate_supervision as ds
    from tests.test_working_checkpoint import _registry

    custody, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, "nanny")
    task["_working_recovery"] = {}
    ctx = _registry(tmp_path, monkeypatch, "nanny", 2)._ctx
    assert recovery.adopt_handoff(ctx, task)["status"] == "adopted"
    row = recovery._read(tmp_path, "nanny")
    binding = {key: row[key] for key in ("selected_subagent_id", "config_fingerprint", "authority_fingerprint",
                                         "work_order_fingerprint", "snapshot_id", "execution_root",
                                         "baseline_sha", "target_root")}
    if fate == "bound_settled":
        _bound(tmp_path, task, invocation, terminal=True)  # Bound and settled, never waited on.
    else:
        assert custody.emit(tmp_path, custody.START_FAILED, {"task_id": "nanny", "invocation_id": invocation,
                                                             "definite": True})
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-c", invocation_id="invoke-c", task_id="nanny", route_id="codex", **binding))
    if fate != "refused_running":
        def worker_dies(*_a, **_k):
            raise InputReached
        with pytest.raises(InputReached):
            ds.supervised_wait(ctx, "run-c", since_seq=23, wait_once=worker_dies)
        assert ds._load_state(ctx, "run-c")["status"] == "sleeping"
    engines, before = list(stub_delegate_engines), recovery._path(tmp_path, "nanny").read_bytes()
    unrecoverable = []
    monkeypatch.setattr(recovery, "reconcile_unrecoverable_task", lambda *_a: unrecoverable.append(True))
    if door == "worker_crash":
        preserved = {"nanny"} if recovery.prepare_worker_crash_handoff(
            tmp_path, task, old_attempt=2, new_attempt=3, worker_id=1, exitcode=1) else set()
    else:
        preserved = recovery.prepare_planned_restart_handoffs(
            tmp_path, {"nanny": {"task": task, "attempt": 2}}, restart_transaction_id="tx")
    current = recovery._read(tmp_path, "nanny")
    if fate in {"refused", "bound_settled"} or (fate == "refused_running" and door == "worker_crash"):
        assert preserved == {"nanny"} and not unrecoverable
        assert (current["status"], current["run_id"], current["pending_invocation_id"]) == ("reserved", "run-c", "")
        assert current["journal_cursor"] == (0 if fate == "refused_running" else 23)
        assert (current["old_attempt"], current["new_attempt"]) == (2, 3)
    else:
        # A planned stop keeps only sleeping supervision; an unproven fate keeps its reservation.
        assert not preserved and recovery._path(tmp_path, "nanny").read_bytes() == before
        assert unrecoverable == ([True] if door == "worker_crash" else [])
    assert stub_delegate_engines == engines


def test_crash_handoff_does_not_graft_an_old_run_wait_onto_never_waited_holder(
        tmp_path, monkeypatch, stub_delegate_engines):
    """A resolved B exposes C, not A's retained cursor, question or wake."""
    from ouroboros import delegate_supervision as ds
    from tests.test_working_checkpoint import _registry

    custody, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, "nanny")
    task["_working_recovery"] = {}
    ctx = _registry(tmp_path, monkeypatch, "nanny", 2)._ctx
    assert recovery.adopt_handoff(ctx, task)["status"] == "adopted"
    row = recovery._read(tmp_path, "nanny")
    binding = {key: row[key] for key in ("selected_subagent_id", "config_fingerprint", "authority_fingerprint",
                                       "work_order_fingerprint", "snapshot_id", "execution_root",
                                       "baseline_sha", "target_root")}
    assert custody.emit(tmp_path, custody.START_FAILED,
                        {"task_id": "nanny", "invocation_id": invocation, "definite": True})
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-c", invocation_id="invoke-c", task_id="nanny", route_id="codex", **binding))
    ds._save_state(ctx, {"run_id": "run-a", "status": "sleeping", "journal_cursor": 991,
                        "pending_wake": {"wake_id": "wake-a", "payload": {"run_id": "run-a"}},
                        "interaction_acknowledged_ids": ["question-a"],
                        "coordination_cursor": {"child-terminal": "already-acknowledged"}})
    handoff = recovery.prepare_worker_crash_handoff(
        tmp_path, task, old_attempt=2, new_attempt=3, worker_id=1, exitcode=1)
    assert handoff["run_id"] == "run-c"
    assert handoff["journal_cursor"] == 0
    assert handoff["pending_wake"] == {}
    assert handoff["interaction_acknowledged_ids"] == []
    assert handoff["coordination_cursor"] == {"child-terminal": "already-acknowledged"}
    assert not stub_delegate_engines


def _pending_after_waited_run(tmp_path, monkeypatch):
    """Attempt 1 waited run A out (question, then terminal, both ACKed) and lost pending B's start.

    The first crash handoff (inside ``_pending_handoff``) therefore reads A's supervision file.
    """
    from ouroboros import delegate_custody as custody, delegate_interactions, delegate_supervision as ds

    custody._CUSTODY.clear()
    monkeypatch.setattr(delegate_interactions, "_REPORTED_INTERACTIONS", {})
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-a", invocation_id="invoke-a", task_id="nanny", route_id="codex"))
    first = SimpleNamespace(task_id="nanny", task_attempt=1, drive_root=tmp_path, budget_drive_root=tmp_path)
    ds._save_state(first, {"schema": 1, "run_id": "run-a",  # A child's terminal, acknowledged earlier.
                           "coordination_cursor": {"children": {"child": {"status": "completed"}}}})

    def reports(**fields):
        return lambda *_a, **_k: json.dumps({"run_id": "run-a", **fields})

    question = ds.supervised_wait(first, "run-a", wait_once=reports(
        status="waiting_on_user", last_seq=990,
        pendingInteractions=[{"interactionId": "question-a", "question": "Which path?"}]))
    assert ds.acknowledge_pending_wake(first, question.text)
    assert custody.emit(tmp_path, custody.SETTLED, {"run_id": "run-a", "task_id": "nanny", "state": "succeeded"})
    terminal = ds.supervised_wait(first, "run-a", wait_once=reports(status="terminal", state="succeeded", last_seq=991))
    assert ds.acknowledge_pending_wake(first, terminal.text)
    acked = ds._load_state(first, "run-a")
    assert (acked["status"], acked["journal_cursor"], acked["interaction_acknowledged_ids"]) == (
        "awake", 991, ["question-a"])
    _, recovery, task, _, invocation, _, _ = _pending_handoff(tmp_path, "nanny")
    task["_working_recovery"] = {}
    ctx = SimpleNamespace(task_id="nanny", task_attempt=2, drive_root=tmp_path, budget_drive_root=tmp_path)
    return acked, recovery, task, invocation, ctx


def test_pending_retry_waits_from_its_own_start_not_an_earlier_runs_acked_wait(
        tmp_path, monkeypatch, stub_delegate_engines):
    """Pending B's exact retry is neither answered with A's terminal nor observed from A's cursor (#1862)."""
    from ouroboros import delegate_interactions, delegate_supervision as ds

    acked, recovery, task, invocation, ctx = _pending_after_waited_run(tmp_path, monkeypatch)
    handoff = recovery._read(tmp_path, "nanny")
    assert (handoff["pending_invocation_id"], handoff["run_id"]) == (invocation, "")
    assert (handoff["journal_cursor"], handoff["pending_wake"], handoff["interaction_acknowledged_ids"]) == (0, {}, [])
    assert handoff["coordination_cursor"] == acked["coordination_cursor"], "the acked child cursor is task-scoped"
    assert recovery.adopt_handoff(ctx, task)["retry_of"] == invocation
    _bound(tmp_path, task, invocation)  # The model's exact retry_of binds B.
    observed = []

    def observe(_ctx, run_id, _tick, cursor, **_kw):
        observed.append((run_id, cursor))
        raise InputReached

    with pytest.raises(InputReached):
        ds.supervised_wait(ctx, "bound-run", wait_once=observe)
    assert observed == [("bound-run", 0)], "B's own observation comes first, from B's own start"
    state = ds._load_state(ctx, "bound-run")
    assert (state["journal_cursor"], state["interaction_acknowledged_ids"]) == (0, [])
    assert not any(state.get(key) for key in ("pending_wake", "last_wake", "last_acknowledged_wake"))
    assert state["coordination_cursor"] == acked["coordination_cursor"]
    assert "bound-run" not in delegate_interactions._REPORTED_INTERACTIONS
    assert set(stub_delegate_engines) <= {"gateway"}, "the loop's transport constructor only, no engine lifecycle"


def test_second_crash_keeps_the_same_pending_reservations_deferred_wait(
        tmp_path, monkeypatch, stub_delegate_engines):
    """Control: B's proven reservation carries its own deferred wait through a second handoff."""
    from ouroboros import delegate_supervision as ds

    _, recovery, task, invocation, ctx = _pending_after_waited_run(tmp_path, monkeypatch)
    assert recovery.adopt_handoff(ctx, task)["retry_of"] == invocation
    own = {"journal_cursor": 17, "interaction_acknowledged_ids": ["own-question"],
           "pending_wake": {"wake_id": "kept", "attempt_key": "2", "payload": {
               "status": "waiting_on_user", "supervision_wake_id": "kept"}}}
    recovery._write(tmp_path, {**recovery._read(tmp_path, "nanny"), **own})
    second = recovery.prepare_worker_crash_handoff(
        tmp_path, task, old_attempt=2, new_attempt=3, worker_id=1, exitcode=1)
    assert (second["pending_invocation_id"], second["run_id"]) == (invocation, "")
    assert {key: second[key] for key in own} == own, "neither reset nor replaced by A's file"
    task["_attempt"] = ctx.task_attempt = 3
    assert recovery.adopt_handoff(ctx, task)["retry_of"] == invocation
    _bound(tmp_path, task, invocation)
    replay = ds.supervised_wait(ctx, "bound-run", wait_once=lambda *_a, **_k: pytest.fail("own wake first"))
    assert json.loads(replay.text)["supervision_wake_id"] == "kept"
    assert ds._load_state(ctx, "bound-run")["journal_cursor"] == 17
    assert set(stub_delegate_engines) <= {"gateway"}
