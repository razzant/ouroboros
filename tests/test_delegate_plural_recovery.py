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


@pytest.mark.parametrize("bad", [None, "birth", "exit", "handle"])
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


@pytest.mark.parametrize("fault", ["foreign_binding", "unreadable_fate"])
@pytest.mark.parametrize("planned", [False, True])
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
