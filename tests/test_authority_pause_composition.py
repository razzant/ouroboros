"""Joined authority pauses: child isolation and exact body custody across restart.

Only external transport observations are scripted. Admission, pause publication,
queue restore, BodyRuntime acceptance and Resume use their production owners.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause as bp, usage_accounting as ua
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx, _supervisor_ctx
from tests.test_billing_authority_unavailable import BINDING, _loop_driver, _request, _run, _task_scope
from tests.test_billing_group import _scope, _spend

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("authority", ["readable", "unreadable", "group_exhausted", "global_exhausted"])
def test_child_pause_keeps_sibling_independent_and_resume_rechecks_original_money(tmp_path, monkeypatch, authority):
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import patch_document
    from supervisor.events import _handle_budget_pause

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    binding = {**BINDING, "billing_group_limit_usd": 2.0}
    write_task_result(tmp_path, "P", "running", billing_group=binding)
    write_task_result(tmp_path, "R", "running", billing_group=binding)
    if authority == "group_exhausted":
        _spend(tmp_path, _scope(tmp_path, "P", "P", group="P", group_limit=2.0, root_limit=2.0), 2.0)
    # The root's whole-work binding turns corrupt: every scope that rereads it meets an unreadable authority.
    patch_document(tmp_path, "R", lambda row: {**row, "billing_group": {"billing_group_id": "P"}})
    task = {"id": "child", "root_task_id": "R", "parent_task_id": "R", "type": "task",
            "chat_id": 0, "_attempt": 1, "budget_drive_root": str(tmp_path)}
    sibling = {**task, "id": "sibling"}
    write_task_result(tmp_path, "child", "running", root_task_id="R", parent_task_id="R")
    write_task_result(tmp_path, "sibling", "running", root_task_id="R", parent_task_id="R")
    workers.RUNNING.update({tid: {"task": row, "worker_id": wid, "attempt": 1}
                            for tid, row, wid in (("child", task, 0), ("sibling", sibling, 1))})
    workers.WORKERS.update({wid: SimpleNamespace(busy_task_id=tid) for wid, tid in ((0, "child"), (1, "sibling"))})
    sibling_before = copy.deepcopy(workers.RUNNING["sibling"])
    loop, registry, sends = _loop_driver(tmp_path, monkeypatch, "child")
    registry._ctx.root_task_id = "R"
    registry._ctx.task_metadata = {"root_task_id": "R", "parent_task_id": "R"}
    with ua.usage_scope(_task_scope(tmp_path, "child", "R")), pytest.raises(bp.BudgetPauseRequested) as paused:
        _run(loop, registry, tmp_path, "child")
    assert sends == ["child"] and paused.value.pause["scope"] == "task"
    _handle_budget_pause({**bp.pause_event(task, paused.value.pause), "worker_id": 0},
                         _supervisor_ctx(tmp_path, workers, q, [], []))
    bp.end_dispatch_fence("child")
    assert q.BUDGET_ROOT_FENCES == {}
    assert workers.RUNNING["sibling"] == sibling_before
    assert [row["id"] for row in workers.PENDING] == ["child"]
    # Even the unaffected running sibling cannot buy new work under unreadable authority.
    with ua.usage_scope(_task_scope(tmp_path, "sibling", "R")), pytest.raises(ua.BillingAuthorityUnavailable):
        ua.reserve_attempt(_request())
    if authority != "unreadable":
        patch_document(tmp_path, "R", lambda row: {**row, "billing_group": binding})
    if authority == "global_exhausted":
        monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 0.0)
    result = q.resume_budget_paused_task("child")
    errors = {"unreadable": "billing_authority_unavailable", "group_exhausted": "root_hard_cap_exhausted",
              "global_exhausted": "budget_still_exhausted"}
    if authority in errors:
        assert result["error"] == errors[authority], result
        assert "_budget_pause_resume" not in workers.PENDING[0]
    else:
        assert result["ok"], result
        with ua.usage_scope(_task_scope(tmp_path, "sibling", "R")):
            reservation = ua.reserve_attempt(_request())
            assert reservation.scope.billing_group_id == "P"
            ua.release_attempt(reservation, reason="test_unsent")
        assert workers.PENDING[0]["_budget_pause_resume"]["pause"]["root_task_id"] == "R"
    assert workers.RUNNING["sibling"] == sibling_before and q.BUDGET_ROOT_FENCES == {}


class LiveGateway:
    """No provider/daemon: the already-admitted external session remains live."""
    def get_run(self, run_id, **kwargs):
        return {"summary": {"runId": run_id, "state": "running"}, "lastSeq": 1}

    def close(self):
        pass


def _park_with_body(tmp_path, monkeypatch, direct):
    from ouroboros import delegate_collection as body, sleep_custody as sc
    from supervisor.events_budget import install_exact_budget_pause
    from tests.test_cold_external_custody import _setup

    env = _setup(tmp_path, monkeypatch, direct=direct)
    # Undo only the helper's fake caretaking and snapshot writer, keeping real authority.
    monkeypatch.setattr(sc, "_runtime", body.body_runtime)
    from supervisor.queue_snapshot import persist_queue_snapshot
    monkeypatch.setattr(env.q, "persist_queue_snapshot", persist_queue_snapshot)
    runtime = body.start_body_runtime(tmp_path, gateway_factory=LiveGateway, max_workers=2)
    env.runtime = runtime
    def host_tick(_ctx):
        sc.reconcile(env.q, direct=[env.task] if direct else ())
        return ""
    monkeypatch.setattr(bp, "_hold_control_reason", host_tick)
    try:
        with pytest.raises(bp.BudgetPauseRequested) as parked:
            bp.request_pause(env.limit, rail=bp.RAIL_BILLING_AUTHORITY, scope="task",
                             reason_text="billing authority unreadable", root_task_id="sleeper")
        row = parked.value.pause
        assert row["external_custody"]["runtime"]["runtime_id"] == runtime.runtime_id
        install_exact_budget_pause(_supervisor_ctx(tmp_path, env.workers, env.q, [], []), "sleeper",
                                   bp.exact_pause_marker(row)["checkpoint"], evt=bp.pause_event(env.task, row))
        bp.end_dispatch_fence("sleeper")
        return env, bp.budget_pause_row(tmp_path, "sleeper")
    except BaseException:
        runtime.close(wait=True)
        raise


@pytest.mark.parametrize("direct", [False, True])
def test_cold_queue_restore_rebinds_actual_body_and_returns_same_live_session(tmp_path, monkeypatch, direct):
    from ouroboros import delegate_collection as body, delegate_custody_current as current, owner_wait, sleep_custody as sc
    from ouroboros.project_lease import cold_project_holders, candidate_is_leasable
    from supervisor.sleep_wake import wake_ready_sleepers

    env, row = _park_with_body(tmp_path, monkeypatch, direct)
    first_runtime = env.runtime
    try:
        assert env.q.persist_queue_snapshot(reason="authority_restart")
        first_runtime.close(wait=True)
        assert not first_runtime.thread.is_alive()
        env.workers.PENDING.clear()
        assert env.q.restore_pending_from_snapshot() == 1
        parked = env.workers.PENDING[0]
        assert parked["id"] == "sleeper" and sc.carrier_receipt(parked)
        assert wake_ready_sleepers(env.q) == []
        assert env.q.resume_budget_paused_task("sleeper")["error"] == "external_runs_unsettled"
        holders = cold_project_holders(env.workers.PENDING)
        assert holders == {"project": {"sleeper"}}
        assert not candidate_is_leasable({"id": "foreign", "project_id": "project"}, set(), cold_holders=holders)
        env.runtime = body.start_body_runtime(tmp_path, gateway_factory=LiveGateway, max_workers=2)
        assert env.runtime.runtime_id != first_runtime.runtime_id
        assert sc.reconcile(env.q) == {"sleeper"}
        assert bp.budget_pause_row(tmp_path, "sleeper")["external_custody"] == row["external_custody"]
        assert env.q.resume_budget_paused_task("sleeper")["ok"]
        ctx, _ = _loop_ctx(tmp_path, "sleeper", direct=direct)
        ctx.budget_pause_resume = parked["_budget_pause_resume"]
        saved = bp.load_budget_pause(ctx)
        monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
        messages = []
        bp.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, messages, {}, {}, set(), budget_remaining_usd=5.0)
        assert messages[0]["content"] == "original intent"
        assert bp.budget_pause_row(tmp_path, "sleeper")["state"] == "resumed"
        assert not current.run(tmp_path, "run-one").settled
        assert current.run(tmp_path, "run-one").invocation_id == "invoke-one"
    finally:
        env.runtime.close(wait=True)
        assert not env.runtime.thread.is_alive()


@pytest.mark.parametrize("change", ["task_attempt", "pause_generation", "root_task_id", "project_id", "source_ref",
                                    "foreign_run", "changed_invocation", "unknown_custody"])
def test_resume_refuses_changed_or_unknown_accepted_binding(tmp_path, monkeypatch, change):
    from ouroboros import delegate_custody as custody, delegate_custody_current as current, sleep_custody as sc
    env, original = _park_with_body(tmp_path, monkeypatch, False)
    try:
        row = copy.deepcopy(original)
        if change in {"task_attempt", "pause_generation"}:
            row[change] = int(row[change]) + 1
        elif change == "source_ref":
            row[change] = {**row[change], "sha256": "0" * 64}
        elif change in {"root_task_id", "project_id"}:
            row[change] = "foreign"
        elif change == "foreign_run":
            assert custody.emit(tmp_path, custody.STARTED, {"run_id": "another", "invocation_id": "another-invocation",
                "task_id": "sleeper", "root_task_id": "sleeper", "route": "route", "source": "delegate_start"})
        elif change == "changed_invocation":
            real = current.run
            def changed(root, rid):
                entry = real(root, rid)
                if entry is not None:
                    entry.invocation_id = "foreign-invocation"
                return entry
            monkeypatch.setattr(current, "run", changed)
        else:
            real = current.task_snapshot
            def unknown(root, tid):
                if tid == "sleeper":
                    raise OSError("synthetic custody reader failure")
                return real(root, tid)
            monkeypatch.setattr(current, "task_snapshot", unknown)
        if row != original:
            bp.set_budget_pause(tmp_path, "sleeper", row, expected_pause_id=original["pause_id"])
            assert not sc.accepted_receipt("sleeper", row)
        result = env.q.resume_budget_paused_task("sleeper")
        assert not result["ok"], (change, result)
        assert "_budget_pause_resume" not in env.workers.PENDING[0]
        assert bp.budget_pause_row(tmp_path, "sleeper")["state"] == "paused"
    finally:
        env.runtime.close(wait=True)
        assert not env.runtime.thread.is_alive()


def test_real_owner_pause_reaches_retained_ordinary_run_but_not_started_critic(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody, delegate_custody_current as current
    from ouroboros.gateways import claudexor
    from supervisor.owner_pause_control import request_owner_pause, _record_settled_member

    env, _row = _park_with_body(tmp_path, monkeypatch, False)
    controls = []
    class Gateway(LiveGateway):
        def handshake(self):
            pass

        def cancel_run(self, run_id, reason=""):
            controls.append((run_id, reason))
            return {"accepted": True, "status": "accepted"}
    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    try:
        assert custody.emit(tmp_path, custody.STARTED, {"run_id": "critic", "invocation_id": "critic-invocation",
            "task_id": "sleeper", "root_task_id": "sleeper", "route": "route", "source": "review_substrate",
            "access": "readonly"})
        assert request_owner_pause("sleeper", request_id="actual-owner-pause")["ok"]
        for _ in range(2):
            _record_settled_member(env.q, env.workers.PENDING[0])
        assert [run_id for run_id, _reason in controls] == ["run-one"]
        assert not current.run(tmp_path, "critic").settled
        assert not env.q.resume_budget_paused_task("sleeper")["ok"]
        assert "_budget_pause_resume" not in env.workers.PENDING[0]
    finally:
        env.runtime.close(wait=True)
        assert not env.runtime.thread.is_alive()
