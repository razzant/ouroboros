"""F12: real SQLite save contention must retain the same technical continuation."""
from __future__ import annotations

import copy
import json
import sqlite3

import pytest

from ouroboros import budget_pause as bp, store
from ouroboros.artifacts import read_actor_source_bytes
from tests._budget_pause_exact_helpers import _loop_ctx
from tests._store_journal_shared import use_journal
from tests.test_store import document, publish


def _contended_save(root, monkeypatch, stage, transaction, *, on_hold=None):
    """Hold a real BEGIN/COMMIT blocker until request_pause publishes its hold."""
    real_save = bp.set_budget_pause
    attempts, failures, holds, blockers = [], [], [], []

    def save(path, task_id, row, *args, **kwargs):
        selected = bool(row.get("source_ref")) == (stage == "checkpoint")
        if row.get("state") == bp.STATE_PAUSING and selected:
            attempts.append(row)
            if len(attempts) == 1:
                conn = sqlite3.connect(store.store_path(root), isolation_level=None)
                blockers.append(conn)
                conn.execute("BEGIN IMMEDIATE" if transaction == "begin" else "BEGIN")
                conn.execute("SELECT * FROM meta").fetchall()
            try:
                with store.acquisition_budget(0):
                    return real_save(path, task_id, row, *args, **kwargs)
            except store.StoreBusy as exc:
                failures.append(exc)
                assert exc.reason == "contention" and exc.rollback_confirmed is True
                raise
        with store.acquisition_budget(0):
            return real_save(path, task_id, row, *args, **kwargs)

    def release():
        while blockers:
            blockers.pop().close()

    def held(ctx, row):
        if row.get("state") != bp.STATE_PAUSING:
            return
        holds.append(copy.deepcopy(row))
        assert bp.dispatch_fenced(ctx.task_id) and ctx._budget_pausing
        assert len(holds) == 1, "contention must release on the first hold"
        release()
        if on_hold:
            on_hold(ctx)

    monkeypatch.setattr(bp, "set_budget_pause", save)
    monkeypatch.setattr(bp, "_publish_hold", held)
    monkeypatch.setattr(bp, "_HOLD_POLL_SEC", 0.001)
    return attempts, failures, holds, release


@pytest.mark.parametrize("rail", [bp.RAIL_BILLING_AUTHORITY, bp.RAIL_CONTROL_AUTHORITY])
@pytest.mark.parametrize("stage", ["seed", "checkpoint"])
@pytest.mark.parametrize("transaction", ["begin", "commit"])
def test_real_save_contention_releases_to_same_snapshot(tmp_path, monkeypatch, rail, stage, transaction):
    if transaction == "commit":  # a reader holds only a rollback-journal COMMIT; a WAL COMMIT waits for none
        use_journal(monkeypatch, "PERSIST")
    publish(tmp_path, document("member", status="running"))
    ctx, limit = _loop_ctx(tmp_path, "member")
    messages, trace = copy.deepcopy(limit.messages), copy.deepcopy(limit.llm_trace)
    attempts, failures, holds, release = _contended_save(tmp_path, monkeypatch, stage, transaction)
    real_prepare, prepared = bp._exact_continuation_row, []

    def prepare(*args, **kwargs):
        row = real_prepare(*args, **kwargs)
        prepared.append(row)
        return row

    monkeypatch.setattr(bp, "_exact_continuation_row", prepare)
    try:
        with pytest.raises(bp.BudgetPauseRequested) as parked:
            bp.request_pause(limit, rail=rail, scope="task", reason_text="unreadable authority")
        row = parked.value.pause
        assert len(failures) == len(holds) == len(prepared) == 1
        assert len(attempts) == 2 and attempts[0] is attempts[1]
        assert row["pause_id"] == attempts[0]["pause_id"] and row["pause_generation"] == 1
        assert row["source_ref"] == prepared[0]["source_ref"]
        assert bp.budget_pause_row(tmp_path, "member") == row
        state = json.loads(read_actor_source_bytes(tmp_path, "member", row["source_ref"]))
        assert state["messages"] == messages and state["trace"] == trace
        assert row["resume_point"]["unanswered_tool_call_ids"] == ["call_b"]
        assert row["resume_point"]["unanswered_policy"] == "not_re_executed_execution_unknown"
        assert limit.accumulated_usage["cost"] == 1.25
        assert limit.accumulated_usage["reason_code"] == "authority_paused"
        assert "exact_pause_unavailable" not in limit.accumulated_usage
        assert "budget_pause_hold" not in limit.accumulated_usage
    finally:
        release()
        bp.end_dispatch_fence("member")


@pytest.mark.parametrize("stage", ["seed", "checkpoint"])
def test_real_loop_keeps_authority_interruption_resumable_after_contention(tmp_path, monkeypatch, stage):
    from ouroboros import usage_accounting as ua
    from tests.test_billing_authority_unavailable import _loop_driver, _run, _task_scope

    publish(tmp_path, document("member", status="running", billing_group={"billing_group_id": "member"}))
    attempts, failures, holds, release = _contended_save(tmp_path, monkeypatch, stage, "begin")
    loop, registry, sends = _loop_driver(tmp_path, monkeypatch, "member")
    try:
        with ua.usage_scope(_task_scope(tmp_path, "member", "member")), pytest.raises(bp.BudgetPauseRequested) as parked:
            _run(loop, registry, tmp_path, "member")
        assert sends == ["member"] and len(failures) == len(holds) == 1
        assert attempts[0] is attempts[1]
        state = json.loads(read_actor_source_bytes(tmp_path, "member", parked.value.pause["source_ref"]))
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][0]["content"] == "keep this thought"
    finally:
        release()
        bp.end_dispatch_fence("member")


@pytest.mark.parametrize("stage", ["seed", "checkpoint"])
@pytest.mark.parametrize("control", ["cancelled", "panic", "deadline"])
def test_contention_hold_obeys_real_task_controls(tmp_path, monkeypatch, stage, control):
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.model_wait import ModelWaitInterrupted, task_model_wait_scope

    publish(tmp_path, document("member", status="running"))
    ctx, limit = _loop_ctx(tmp_path, "member")
    task = {"id": "member", "_attempt": 1}
    requested = [""]

    def stop(_ctx):
        if control == "cancelled":
            request_cancel(tmp_path, "member", source="contention-test")
        elif control == "panic":
            requested[0] = "panic"
        else:
            task["deadline_at"] = "2000-01-01T00:00:00+00:00"

    attempts, failures, holds, release = _contended_save(tmp_path, monkeypatch, stage, "begin", on_hold=stop)
    try:
        with task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None, worker_slot_held=True,
                                   owner_control=lambda: requested[0]):
            with pytest.raises(ModelWaitInterrupted) as stopped:
                bp.request_pause(limit, rail=bp.RAIL_BILLING_AUTHORITY, scope="task", reason_text="unreadable")
        assert stopped.value.control_reason == control
        assert len(attempts) == len(failures) == len(holds) == 1
        assert bp.dispatch_fenced("member")  # a control never reopens paid dispatch
        assert limit.accumulated_usage["exact_pause_unavailable"] == "hold_ended_by_control"
        row = bp.budget_pause_row(tmp_path, "member")
        if stage == "checkpoint":
            assert row["state"] == bp.STATE_ABANDONED
            assert row["abandon_reason"] == f"hold_ended_by_control:{control}"
        else:
            assert row == {}  # no seed was committed
    finally:
        release()
        bp.end_dispatch_fence("member")


@pytest.mark.parametrize("stage", ["seed", "checkpoint"])
@pytest.mark.parametrize("reason,rollback", [("contention", True), ("contention", False),
                                           ("contention", None), ("unknown", True), ("permission", True)])
def test_only_positive_rolled_back_contention_retries(tmp_path, monkeypatch, stage, reason, rollback):
    publish(tmp_path, document("member", status="running"))
    _ctx, limit = _loop_ctx(tmp_path, "member")
    real_save, attempts = bp.set_budget_pause, []
    retry = reason == "contention" and rollback is True

    def save(root, tid, row, *args, **kwargs):
        if row.get("state") == bp.STATE_PAUSING and bool(row.get("source_ref")) == (stage == "checkpoint"):
            attempts.append(row)
            if len(attempts) <= 2:
                exc = store.StoreBusy("typed save failure", reason=reason)
                if rollback is not None:
                    exc.rollback_confirmed = rollback
                raise exc
        return real_save(root, tid, row, *args, **kwargs)

    monkeypatch.setattr(bp, "set_budget_pause", save)
    monkeypatch.setattr(bp, "_HOLD_POLL_SEC", 0.001)
    try:
        if retry:
            with pytest.raises(bp.BudgetPauseRequested):
                bp.request_pause(limit, rail=bp.RAIL_CONTROL_AUTHORITY, scope="task", reason_text="unreadable")
            assert len(attempts) == 3 and all(row is attempts[0] for row in attempts)
        else:
            assert bp.request_pause(limit, rail=bp.RAIL_CONTROL_AUTHORITY, scope="task", reason_text="unreadable") is None
            assert len(attempts) == 1 and not bp.dispatch_fenced("member")
            assert limit.accumulated_usage["exact_pause_unavailable"] == (
                bp.HOLD_PAUSE_RECORD_UNWRITABLE if stage == "seed" else bp.HOLD_CHECKPOINT_UNWRITABLE)
            assert "budget_pause_hold" not in limit.accumulated_usage
    finally:
        bp.end_dispatch_fence("member")


@pytest.mark.serial
@pytest.mark.parametrize("fault", ["before_publication", "ambiguous_after_publication"])
def test_custody_retry_keeps_one_capture_and_the_original_live_run(tmp_path, monkeypatch, fault):
    from ouroboros import owner_wait, sleep_custody as sc
    from tests.test_cold_external_custody import _setup

    real_control = bp._hold_control_reason
    env = _setup(tmp_path, monkeypatch)
    captures, observations, fields, writes = [], [], [], []
    real_source, real_observe, real_fields, real_replace = (
        owner_wait.store_continuation_source, bp.observe_task_runs, sc.request_fields, sc._replace)

    def capture(*args):
        source = real_source(*args)
        captures.append(copy.deepcopy(source))
        return source

    def observe(*args, **kwargs):
        observations.append(kwargs.get("stop_policy"))
        return real_observe(*args, **kwargs)

    def request_fields(ctx, row):
        result = real_fields(ctx, row)
        fields.append(copy.deepcopy(result))
        return result

    def replace(root, tid, expected, replacement):
        if replacement.get("external_custody", {}).get("state") == "requested":
            writes.append(replacement)
            if len(writes) == 1:
                if fault == "ambiguous_after_publication":
                    real_replace(root, tid, expected, replacement)
                exc = store.StoreBusy("publication observation unavailable", reason=(
                    "contention" if fault == "before_publication" else "unknown"))
                exc.rollback_confirmed = fault == "before_publication"
                raise exc
        return real_replace(root, tid, expected, replacement)

    def host_tick(ctx):
        sc.reconcile(env.q)
        return real_control(ctx)

    monkeypatch.setattr(owner_wait, "store_continuation_source", capture)
    monkeypatch.setattr(bp, "observe_task_runs", observe)
    monkeypatch.setattr(sc, "request_fields", request_fields)
    monkeypatch.setattr(sc, "_replace", replace)
    monkeypatch.setattr(bp, "_hold_control_reason", host_tick)
    try:
        with pytest.raises(bp.BudgetPauseRequested) as paused:
            bp.request_pause(env.limit, rail=bp.RAIL_BILLING_AUTHORITY, scope="task", reason_text="unreadable")
        row = paused.value.pause
        assert len(captures) == len(observations) == len(fields) == 1
        assert observations == [bp.STOP_POLICY_OBSERVE]  # no custody cancel/stop effects
        assert row["source_ref"] == captures[0]
        assert row["external_custody"]["binding"] == fields[0]["binding"]
        assert row["external_custody"]["runs"] == [["run-one", "invoke-one"]] == fields[0]["runs"]
        assert sc.accepted_receipt("sleeper", row)
        assert len(writes) == (2 if fault == "before_publication" else 1)
        assert all(row is writes[0] for row in writes)
        assert env.limit.accumulated_usage["cost"] == 1.25
    finally:
        bp.end_dispatch_fence("sleeper")
