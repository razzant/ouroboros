"""Plural recovery follows each saved binding through independent outcomes."""
from types import SimpleNamespace
import json
import os

import pytest

from ouroboros import delegate_custody as custody, delegate_recovery as recovery
from ouroboros import delegate_supervision as supervision


def task_and_context(root):
    task = {"id": "root", "drive_root": str(root), "_attempt": 2,
            "task_contract": {"objective": "supervise independent leaves"}}
    ctx = SimpleNamespace(task_id="root", task_attempt=2, drive_root=root, budget_drive_root=root, task_metadata=task)
    return task, ctx


def add_holder(root, task, name, *, pending=False, settled=False):
    binding = {"selected_subagent_id": f"row-{name}", "config_fingerprint": f"config-{name}",
               "work_order_fingerprint": f"work-{name}",
               "authority_fingerprint": recovery.authority_fingerprint_from_task(task)}
    assert custody.record_start_requested(root, task_id=task["id"], invocation_id=f"inv-{name}",
                                          request={"prompt": f"assignment {name}"}, route="codex", **binding)
    if not pending:
        run = custody.RunCustody(task_id=task["id"], run_id=name, invocation_id=f"inv-{name}", route_id="codex", **binding)
        assert custody.record_started(root, run)
        if settled:
            assert custody.emit(root, custody.SETTLED, {"task_id": task["id"], "run_id": name, "state": "succeeded"})
    return binding


def reserve(root, task, **kwargs):
    return recovery.prepare_handoff(root, task, old_attempt=1, new_attempt=2,
                                    cause=kwargs.pop("cause", recovery.CAUSE_WORKER_CRASH), **kwargs)


@pytest.mark.parametrize("door", ["startup", "worker"])
@pytest.mark.parametrize("answer", ["queued", "running", "succeeded", "foreign", "unnamed", "no_state",
                                    "foreign_store", "read_error", "unsaved", "terminal_unsaved"])
def test_adoption_publishes_the_first_proved_state_of_an_accepted_start(tmp_path, monkeypatch, door, answer):
    from ouroboros import delegate_custody_current as current
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from tests._delegated_transport_shared import Engine, _bind_without_state, _rows
    from tests._owner_pause_ended_helpers import REQUEST, _decide, _pending

    task, ctx = task_and_context(tmp_path)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    start = _pending(tmp_path, "root", "lost", tree="root", selected_subagent_id="actor",
                     config_fingerprint="config", work_order_fingerprint="work",
                     authority_fingerprint=recovery.authority_fingerprint_from_task(task))
    assert _decide(tmp_path, start)
    engine = Engine(tmp_path)
    state = "succeeded" if answer == "terminal_unsaved" else answer if answer in {"queued", "succeeded"} else "running"
    job = engine.accept("lost", REQUEST, state=state)
    # A pending reservation may acquire its run identity between generations.
    reserve(tmp_path, task, cause=recovery.CAUSE_PLANNED_SELF_RESTART, restart_transaction_id="tx")
    recovery._write_restart_transaction(tmp_path, {"transaction_id": "tx", "status": "normal_exit_acknowledged",
        "supervisor_pid": os.getpid(), "exit_code": 42, "task_ids": ["root"]})
    with engine.gateway() as gateway:
        run = _bind_without_state(tmp_path, gateway, engine, engine.commands[job]["run"])
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", engine.gateway)
    monkeypatch.setattr("ouroboros.tools.delegate.exact_start", lambda *_a: pytest.fail("bound invocation replayed"))
    detail = engine._detail(job)
    if answer in {"foreign", "unnamed"}:
        detail["summary"] = {"state": "running", **({"runId": "another-run"} if answer == "foreign" else {})}
        monkeypatch.setattr(engine, "_detail", lambda _job: detail)
    if answer == "no_state":
        detail["summary"].pop("state")
        monkeypatch.setattr(engine, "_detail", lambda _job: detail)
    if answer == "foreign_store":
        monkeypatch.setattr(engine, "_named", lambda _request: {"storeId": "replacement"})
    if answer == "read_error":
        def unavailable(_job):
            raise OSError("run unreadable")
        monkeypatch.setattr(engine, "_detail", unavailable)
    if answer in {"unsaved", "terminal_unsaved"}:
        emit = custody.emit
        refused = custody.START_STATE_OBSERVED if answer == "unsaved" else custody.TERMINAL_OBSERVED
        monkeypatch.setattr(custody, "emit", lambda root, kind, row, **kw:
                            False if kind == refused else emit(root, kind, row, **kw))

    proved = answer in {"queued", "running", "succeeded"}
    if door == "startup":
        assert recovery.pre_adopt_planned_handoffs(tmp_path, [task]) == {"root"}  # includes reserved for orphan protection
        assert recovery.planned_handoff_resume_allowed(tmp_path, task) is proved
        assert recovery._read(tmp_path, "root")["holders"][0]["status"] == ("pre_adopted" if proved else "deferred")
    else:
        receipt = recovery.adopt_handoff(ctx, task)
        assert receipt["status"] == ("adopted" if proved else "recovery_required")
        if not proved:
            assert receipt["reason"] == "run_unobserved"
        elif answer == "succeeded":
            assert receipt["terminal_state"] == "succeeded"
    observed = current.run(tmp_path, run.run_id)
    assert observed.resume_start_state == (state if proved or answer == "terminal_unsaved" else "unknown")
    fence = read_fence(tmp_path, "root")
    if answer in {"queued", "running"}:
        assert fence["active_delegated_start"]["run_id"] == run.run_id
        assert fence["active_delegated_start"]["state"] == answer
    else:
        assert not fence
    assert bool(observed.terminal_source) is (answer == "succeeded")
    assert not _rows(tmp_path, custody.SETTLED) and not engine.creates and not engine.cancels
    if answer == "succeeded":
        from ouroboros.delegate_collection import collect_terminal, retained_payload
        if door == "startup":
            monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway",
                                lambda: pytest.fail("successor reopened the engine after startup retained the terminal"))
            with monkeypatch.context() as unavailable:
                def unreadable(*_args):
                    raise OSError("terminal source unavailable")
                unavailable.setattr("ouroboros.delegate_collection._read_source", unreadable)
                assert recovery.adopt_handoff(ctx, task)["status"] == "recovery_required"
            assert recovery.adopt_handoff(ctx, task)["terminal_state"] == "succeeded"
        with engine.gateway() as gateway:
            monkeypatch.setattr(gateway, "get_run", lambda *_a, **_kw: pytest.fail("retained terminal queried again"))
            collected = collect_terminal(tmp_path, run.run_id, gateway=gateway)
        assert collected["settlement"]["settled"] and collected["output_retention"]["complete"]
        assert retained_payload(tmp_path, current.run(tmp_path, run.run_id))["primary_output"]["text"] == engine.commands[job]["text"]


def test_a_saved_ordinary_run_admits_past_one_failed_startup_probe_and_adopts_on_the_worker_recheck(
        tmp_path, monkeypatch):
    """Ordinary saved work (never an accepted unconfirmed start) is not left reserved for ever by
    one failed liveness probe: the deferred holder keeps its orphan protection while the row admits
    the successor, and the worker boot's own adoption re-probes the physical run. No new start,
    money or Stop is issued."""
    from ouroboros import delegate_custody_current as current
    from ouroboros.task_results import write_task_result
    from tests._delegated_transport_shared import _rows

    task, ctx = task_and_context(tmp_path)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    add_holder(tmp_path, task, "live")
    assert current.run(tmp_path, "live").resume_start_state is None  # ordinary: no accepted start risk
    reserve(tmp_path, task, cause=recovery.CAUSE_PLANNED_SELF_RESTART, restart_transaction_id="tx")
    recovery._write_restart_transaction(tmp_path, {"transaction_id": "tx", "status": "normal_exit_acknowledged",
        "supervisor_pid": os.getpid(), "exit_code": 42, "task_ids": ["root"]})

    def unreachable(*_a, **_kw):
        raise OSError("engine unreachable during the startup probe")
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", unreachable)
    assert recovery.pre_adopt_planned_handoffs(tmp_path, [task]) == {"root"}  # kept, for orphan protection
    row = recovery._read(tmp_path, "root")
    assert row["holders"][0]["status"] == "deferred" and row["holders"][0]["deferred_reason"] == "run_unobserved"
    assert row["status"] == "pre_adopted", "admission is not the startup probe's hostage; the worker re-checks"
    assert recovery.planned_handoff_resume_allowed(tmp_path, task)
    # The subsequent success: the worker boot's own re-check reads the ordinary run live.
    detail = {"summary": {"runId": "live", "jobId": "job-live", "state": "running", "effectiveAccess": "readonly"},
              "lastSeq": 2, "pendingInteractions": []}
    gateway = SimpleNamespace(store_identity="store-a", close=lambda: None, get_run=lambda *_a, **_kw: detail)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: gateway)
    receipt = recovery.adopt_handoff(ctx, task)
    assert receipt["status"] == "adopted" and receipt["run_id"] == "live"
    assert not _rows(tmp_path, custody.SETTLED) and not _rows(tmp_path, custody.START_STATE_OBSERVED)


def test_a_recovered_start_gates_adoption_until_its_first_state_is_proved_then_admits(tmp_path, monkeypatch):
    """An accepted unconfirmed start keeps the startup gate shut while its first state is unproved
    (one failed probe included); a genuine later answer proves it and admits the successor."""
    from ouroboros import delegate_custody_current as current
    from ouroboros.task_results import write_task_result
    from tests._delegated_transport_shared import Engine, _bind_without_state, _rows
    from tests._owner_pause_ended_helpers import REQUEST, _decide, _pending

    task, ctx = task_and_context(tmp_path)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    start = _pending(tmp_path, "root", "lost", tree="root", selected_subagent_id="actor",
                     config_fingerprint="config", work_order_fingerprint="work",
                     authority_fingerprint=recovery.authority_fingerprint_from_task(task))
    assert _decide(tmp_path, start)
    engine = Engine(tmp_path)
    job = engine.accept("lost", REQUEST, state="running")
    reserve(tmp_path, task, cause=recovery.CAUSE_PLANNED_SELF_RESTART, restart_transaction_id="tx")
    recovery._write_restart_transaction(tmp_path, {"transaction_id": "tx", "status": "normal_exit_acknowledged",
        "supervisor_pid": os.getpid(), "exit_code": 42, "task_ids": ["root"]})
    with engine.gateway() as gateway:
        run = _bind_without_state(tmp_path, gateway, engine, engine.commands[job]["run"])
    assert run.resume_start_state == "unknown"

    def unreachable(*_a, **_kw):
        raise OSError("engine unreachable during the startup probe")
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", unreachable)
    assert recovery.pre_adopt_planned_handoffs(tmp_path, [task]) == {"root"}  # kept, for orphan protection
    row = recovery._read(tmp_path, "root")
    assert row["holders"][0]["status"] == "deferred" and row["status"] == "reserved"
    assert not recovery.planned_handoff_resume_allowed(tmp_path, task), "the startup gate stays shut"
    # The genuine later answer proves the first state; the reserved row is re-checked and admits.
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", engine.gateway)
    monkeypatch.setattr("ouroboros.tools.delegate.exact_start", lambda *_a: pytest.fail("bound invocation replayed"))
    assert recovery.pre_adopt_planned_handoffs(tmp_path, [task]) == {"root"}
    row = recovery._read(tmp_path, "root")
    assert row["holders"][0]["status"] == "pre_adopted" and row["status"] == "pre_adopted"
    assert recovery.planned_handoff_resume_allowed(tmp_path, task)
    assert current.run(tmp_path, run.run_id).resume_start_state == "running"
    assert not _rows(tmp_path, custody.SETTLED) and not engine.creates and not engine.cancels


def test_root_recovers_live_pending_and_settled_wake_before_configured_return(tmp_path, monkeypatch):
    from ouroboros.subagent_bootstrap import bootstrap_before_context

    task, ctx = task_and_context(tmp_path)
    task["_working_recovery"] = {}
    add_holder(tmp_path, task, "live")
    add_holder(tmp_path, task, "pending", pending=True)
    add_holder(tmp_path, task, "done", settled=True)
    state = supervision._load_state(ctx, "done")
    state.update(status="wake_pending", journal_cursor=19, pending_wake={"wake_id": "kept", "attempt_key": "1",
                 "payload": {"status": "terminal", "run_id": "done", "supervision_wake_id": "kept"}})
    supervision._save_state(ctx, state)
    row = reserve(tmp_path, task)
    assert row["schema"] == 2 and len(row["holders"]) == 3
    monkeypatch.setattr("ouroboros.tools.delegate.exact_start", lambda *_a, **_k: pytest.fail("saved work replayed automatically"))
    monkeypatch.setattr(recovery, "_run_probe_error", lambda *_a: pytest.fail("saved liveness observed automatically"))
    receipt = json.loads(bootstrap_before_context(ctx, task, None))
    assert {h.get("run_id") or h.get("pending_invocation_id") for h in receipt["recovery"]["holders"]} == {"live", "inv-pending", "done"}
    assert all(h["status"] == "adopted" for h in receipt["recovery"]["holders"])
    assert next(h["wake"] for h in receipt["recovery"]["holders"] if h.get("wake"))["supervision_wake_id"] == "kept"
    assert supervision.supervision_checkpoint(ctx)["runs"]["done"]["journal_cursor"] == 19


def test_wrong_holder_proof_does_not_adopt_it_or_discard_its_sibling(tmp_path):
    task, ctx = task_and_context(tmp_path)
    task["_working_recovery"] = {}
    add_holder(tmp_path, task, "a")
    add_holder(tmp_path, task, "b")
    row = reserve(tmp_path, task)
    row["holders"][0]["execution_root"] = "/wrong-binding"
    recovery._write(tmp_path, row)
    result = recovery.adopt_handoff(ctx, task)
    assert [h["status"] for h in result["holders"]] == ["refused", "adopted"]
    assert len(custody.open_runs(tmp_path)) == 2


def test_pending_replays_use_each_invocations_original_binding(tmp_path, monkeypatch):
    from ouroboros.delegate_shared import delegate_result

    task, ctx = task_and_context(tmp_path)
    for name in ("a", "b"):
        add_holder(tmp_path, task, name, pending=True)
    reserve(tmp_path, task)
    posts = []

    def replay(_ctx, prompt, spec):
        invocation = custody.invocation_record(tmp_path, spec["retry_of"])
        assert prompt == invocation["request"]["prompt"]
        posts.append((spec["retry_of"], invocation["selected_subagent_id"], prompt))
        run_id = spec["retry_of"].removeprefix("inv-")
        assert custody.record_started(tmp_path, custody.RunCustody(task_id="root", run_id=run_id,
            invocation_id=spec["retry_of"], route_id="codex", **{key: invocation.get(key, "") for key in (
                "selected_subagent_id", "config_fingerprint", "work_order_fingerprint", "authority_fingerprint")}))
        return delegate_result({"status": "started", "run_id": run_id})

    monkeypatch.setattr("ouroboros.tools.delegate.exact_start", replay)
    monkeypatch.setattr(recovery, "_run_probe_error", lambda *_a: "")
    result = recovery.adopt_handoff(ctx, task)
    assert all(h["status"] == "adopted" for h in result["holders"])
    assert posts == [("inv-a", "row-a", "assignment a"), ("inv-b", "row-b", "assignment b")]


def test_orphan_protection_names_handles_instead_of_every_run_of_the_task(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_reconcile as reconcile

    task, _ctx = task_and_context(tmp_path)
    add_holder(tmp_path, task, "kept")
    add_holder(tmp_path, task, "pending", pending=True)
    reserve(tmp_path, task)
    add_holder(tmp_path, task, "unreserved")
    seen = []
    monkeypatch.setattr(reconcile, "_reconcile_each", lambda _root, runs, _factory, **kwargs:
                        seen.extend(run.run_id for run in runs) or [])
    reconcile.reconcile_orphaned_runs(tmp_path, running_task_ids=set(), recoverable_handles=recovery.recoverable_handles(tmp_path))
    assert seen == ["unreserved"]


@pytest.mark.parametrize("bad", [None, "birth", "handle"])
def test_windows_generation_proof_applies_to_each_holder(tmp_path, monkeypatch, bad):
    from ouroboros import platform_layer

    task, _ctx = task_and_context(tmp_path)
    task["_working_recovery"] = {}
    for name in ("a", "b"):
        add_holder(tmp_path, task, name)
    row = reserve(tmp_path, task, cause=recovery.CAUSE_PLANNED_SELF_RESTART, restart_transaction_id="tx")
    pid = os.getpid()
    tx = {"schema": 1, "transaction_id": "tx", "status": "prepared", "expected_exit_code": 42,
          "supervisor_pid": pid, "supervisor_birth": "parent", "successor_pid": pid,
          "successor_birth": "child", "task_ids": ["root"], "prepared_at": "stable"}
    recovery._write_restart_transaction(tmp_path, tx)
    recovery._write_restart_transaction(tmp_path, {"transaction_id": "active", **{k: tx[k] for k in (
        "supervisor_pid", "prepared_at")}})
    # The active locator names the transaction, not itself.
    recovery._active_restart_transaction_path(tmp_path).write_text(json.dumps({
        "transaction_id": "tx", "supervisor_pid": pid, "prepared_at": "stable"}))
    monkeypatch.setattr(recovery, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: "child")
    parent = {} if bad == "handle" else {"pid": pid, "birth": "wrong" if bad == "birth" else "parent",
                                        "exit_code": 1 if bad == "exit" else 42}
    monkeypatch.setattr(recovery, "observe_restart_parent", lambda: parent)
    monkeypatch.setenv(recovery.PLANNED_RESTART_TRANSACTION_ENV, "tx")
    monkeypatch.setattr(recovery, "veto_handoff", lambda _root, _tid, reason: {"reason": reason})
    kept = recovery.pre_adopt_planned_handoffs(tmp_path, [task])
    assert kept == ({"root"} if bad is None else set())
    if bad is None:
        assert all(h["status"] == "pre_adopted" for h in recovery._read(tmp_path, "root")["holders"])
    assert len(row["holders"]) == 2


def test_handoff_migration_reads_without_effect_and_preserves_unknown(tmp_path):
    task, ctx = task_and_context(tmp_path)
    add_holder(tmp_path, task, "old")
    row = reserve(tmp_path, task)
    legacy = {**{k: v for k, v in row.items() if k not in {"holders", "_single_base"}}, "schema": 1}
    path = recovery._path(tmp_path, "root")
    path.write_text(json.dumps(legacy))
    original = path.read_bytes()
    migrated = recovery._read(tmp_path, "root")
    assert migrated["schema"] == 2 and len(migrated["holders"]) == 1
    assert path.read_bytes() == original
    recovery._write(tmp_path, migrated)
    assert json.loads(path.read_text())["schema"] == 2
    path.write_text('{"schema":2,"holders":null}')
    assert recovery.adopt_handoff(ctx, task)["reason"] == "handoff_unreadable"


@pytest.mark.parametrize(("planned", "fault"), [(False, "unreadable_fate"), (True, "foreign_binding")])
def test_unknown_old_pending_holder_preserves_its_valid_sleeping_sibling(tmp_path, monkeypatch, fault, planned):
    task, ctx = task_and_context(tmp_path)
    task["_working_recovery"] = {}
    add_holder(tmp_path, task, "old", pending=True)
    reserve(tmp_path, task)
    assert recovery.adopt_handoff(ctx, task)["status"] == "adopted"
    assert custody.emit(tmp_path, custody.START_FAILED, {"task_id": "root", "invocation_id": "inv-old", "definite": True})
    if fault == "foreign_binding":
        row = recovery._read(tmp_path, "root")
        row["holders"][0]["snapshot_id"] = "foreign"
        recovery._write(tmp_path, row)
    else:
        monkeypatch.setattr(custody, "invocation_record", lambda *_a, **_k: (_ for _ in ()).throw(OSError("unreadable fate")))
    add_holder(tmp_path, task, "current")
    state = supervision._load_state(ctx, "current")
    state.update(status="sleeping", journal_cursor=23)
    supervision._save_state(ctx, state)
    monkeypatch.setattr(recovery, "reconcile_unrecoverable_task", lambda *_a: pytest.fail("sibling discarded"))
    if planned:
        assert recovery.prepare_planned_restart_handoffs(tmp_path, {"root": {"task": task, "attempt": 2}},
                                                         restart_transaction_id="tx") == {"root"}
    else:
        assert recovery.prepare_worker_crash_handoff(tmp_path, task, old_attempt=2, new_attempt=3, worker_id=1, exitcode=1)
    holders = recovery._read(tmp_path, "root")["holders"]
    assert len(holders) == 2
    assert next(h for h in holders if h.get("pending_invocation_id") == "inv-old")["status"] == "deferred"
    current = next(h for h in holders if h.get("run_id") == "current")
    assert current["status"] == "reserved" and current["journal_cursor"] == 23
