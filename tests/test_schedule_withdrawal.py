"""Withdrawal has one retention clock and one audited remover; accepted work survives."""
import copy
import datetime

import pytest

from supervisor import queue_schedules as store, schedule_occurrence as occurrence
from supervisor.schedule_time import prune_consumed_once_records
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_consciousness_schedule_controls import _events
from tests.test_schedule_occurrence import q, _row, _rows  # noqa: F401


def withdraw(host):
    return host.queue.mutate_scheduled_task("delete", "s1", actor="owner", reason="Finished " + "x" * 600)


def test_delete_keeps_the_row_and_restore_cancels_withdrawal(q):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    assert withdraw(q)["status"] == "withdrawn"
    row = _rows(q)["s1"]
    assert not row["enabled"] and row["delete_requested_by"] == "owner"
    assert len(row["delete_reason"]) == 500
    q.queue.check_scheduled_tasks()
    assert not q.pending and "s1" in _rows(q)
    assert q.queue.mutate_scheduled_task("restore", "s1", actor="owner", reason="Again")["status"] == "updated"
    assert not any(k in _rows(q)["s1"] for k in ("delete_requested_at", "delete_requested_by", "delete_reason"))
    q.queue.check_scheduled_tasks()
    assert len(q.pending) == 1


def test_restore_of_consumed_withdrawal_keeps_history(q):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    q.queue.check_scheduled_tasks()
    before = copy.deepcopy(_rows(q)["s1"])
    withdraw(q)
    result = q.queue.mutate_scheduled_task("restore", "s1", actor="owner", reason="Keep history")
    row = _rows(q)["s1"]
    assert result["status"] == "updated" and not row.get("delete_requested_at")
    assert not row["enabled"] and row["completed_at"] == before["completed_at"]
    assert row["occurrence"] == before["occurrence"] and len(q.pending) == 1


def test_age_gate_precedes_owed_and_settlement_reads(q, monkeypatch):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    withdraw(q)
    calls = []
    monkeypatch.setattr(occurrence, "owed", lambda *a, **k: calls.append("owed") or False)
    monkeypatch.setattr(occurrence, "deletion_settled", lambda *a, **k: calls.append("settled") or True)
    q.queue.check_scheduled_tasks()
    assert "s1" in _rows(q) and calls == []
    monkeypatch.setattr("ouroboros.retention.get_gc_retention_days", lambda: 0)
    q.queue.check_scheduled_tasks()
    assert "s1" not in _rows(q) and calls == ["owed", "settled"]
    events = [e for e in _events(q.root) if e.get("action") == "remove"]
    assert [e["phase"] for e in events] == ["intent", "outcome"]
    assert events[0]["operation_id"] == events[1]["operation_id"]
    q.queue.check_scheduled_tasks()
    assert [e for e in _events(q.root) if e.get("action") == "remove"] == events


@pytest.mark.parametrize("owed,settled", [(True, True), (None, True), (False, False), (False, True)])
def test_remover_requires_known_unowed_and_settled_work(q, monkeypatch, owed, settled):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    withdraw(q)
    monkeypatch.setattr("ouroboros.retention.get_gc_retention_days", lambda: 0)
    monkeypatch.setattr(occurrence, "owed", lambda *a, **k: owed)
    monkeypatch.setattr(occurrence, "deletion_settled", lambda *a, **k: settled)
    q.queue.check_scheduled_tasks()
    assert ("s1" not in _rows(q)) is (owed is False and settled)


def test_remove_intent_failure_keeps_the_withdrawal(q, monkeypatch):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    withdraw(q)
    monkeypatch.setattr("ouroboros.retention.get_gc_retention_days", lambda: 0)
    audit = store._audit_schedule_mutation
    monkeypatch.setattr(store, "_audit_schedule_mutation", lambda **kw: False)
    q.queue.check_scheduled_tasks()
    assert "s1" in _rows(q)
    monkeypatch.setattr(store, "_audit_schedule_mutation", audit)
    q.queue.check_scheduled_tasks()
    assert "s1" not in _rows(q)


def test_consumed_gc_does_not_borrow_the_withdrawal_clock():
    row = {"enabled": False, "trigger": {"type": "once"}, "completed_at": "2000-01-01T00:00:00+00:00"}
    withdrawn = {**row, "delete_requested_at": "2026-10-21T00:00:00+00:00"}
    cutoff = datetime.datetime(2026, 10, 20, tzinfo=datetime.timezone.utc).timestamp()
    assert prune_consumed_once_records([row, withdrawn], cutoff) == ([withdrawn], 1)


def test_withdrawn_occurrence_republishes_and_settles_before_removal(q, monkeypatch):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    q.queue.check_scheduled_tasks()
    task = q.pending.pop()
    token = task["metadata"]["schedule_occurrence"]["token"]
    withdraw(q)
    monkeypatch.setattr("ouroboros.retention.get_gc_retention_days", lambda: 0)
    q.queue.check_scheduled_tasks()
    assert [t["id"] for t in q.pending] == [task["id"]]
    assert _rows(q)["s1"]["occurrence"]["token"] == token
    q.pending.clear()
    write_task_result(q.root, task["id"], "completed", result="Finished")
    q.queue.check_scheduled_tasks()
    assert "s1" not in _rows(q)
    assert load_task_result(q.root, task["id"])["status"] == "completed"


def test_unstarted_claim_blocks_collection_even_when_it_owes_no_task(q):  # noqa: F811
    _row(q, intent={"kind": "system_repo"})
    withdraw(q)
    with store.schedule_transaction(q.root):
        data = store.load_schedule_store(q.root)
        row = data["tasks"][0]
        row["occurrence"] = {"phase": "claimed", "token": "never-admitted", "task_id": "unstarted"}
        assert occurrence.owed(row, drive_root=q.root) is False
        assert occurrence.deletion_settled(row, drive_root=q.root) is True
        cutoff = datetime.datetime.now(datetime.timezone.utc).timestamp() + 1
        assert not store._remove_withdrawn(data, q.root, cutoff) and data["tasks"] == [row]
        row.pop("occurrence")  # reconciliation has now released the claim
        assert store._remove_withdrawn(data, q.root, cutoff) and data["tasks"] == []
