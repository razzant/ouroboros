"""Durable producers survive a busy current store; repair consumes their retained sources."""
import contextlib
import json
from pathlib import Path
import sqlite3
import time

import pytest

from ouroboros import maintenance, maintenance_jobs, plan_collection_job, store, subagent_metrics
from ouroboros import delegate_supervision_state as supervision
from ouroboros.review_records import ReviewActorRecord, ReviewRequest
from ouroboros.review_substrate import ReviewCoordinator
from ouroboros.task_results import write_task_result
from tests.test_plan_record import _participant, _record
from tests.test_subagent_metrics import event


@contextlib.contextmanager
def busy_store(root, exclusive=False):
    # A separate real SQLite connection holds the lock throughout the producer call.
    with contextlib.closing(sqlite3.connect(store.store_path(root), isolation_level=None)) as conn:
        conn.execute("BEGIN EXCLUSIVE" if exclusive else "BEGIN IMMEDIATE")
        try:
            yield
        finally:
            conn.execute("ROLLBACK")


def test_plan_manifest_stays_durable_when_its_wake_cannot_be_written(tmp_path):
    write_task_result(tmp_path, "task", "completed", plan_record=_record([_participant("seat")]))
    maintenance.run_job(tmp_path, plan_collection_job.JOB)
    coordinator = ReviewCoordinator(drive_root=tmp_path)
    request = ReviewRequest(surface="plan_review", goal="review", task_id="task")
    actor = ReviewActorRecord(slot_id="seat", model="test/model", status="ok", raw_text="retained answer")
    with busy_store(tmp_path):
        started = time.monotonic()
        ref = coordinator._persist_producer_outcome(request, actor, "busy", "plan_review", {"answer": "kept"})
        elapsed = time.monotonic() - started
    assert ref and json.loads(Path(ref["manifest_ref"]["path"]).read_text(encoding="utf-8"))["producer_complete"]
    assert elapsed < 1, "the wake must not spend the producer's normal 10-second store budget"
    # The next ordinary dispatch check replays the missed wake without a task edit or repair.
    assert plan_collection_job.JOB in maintenance.due(tmp_path)
    assert maintenance.run_job(tmp_path, plan_collection_job.JOB)["state"] == "done"
    assert maintenance.run_job(tmp_path, plan_collection_job.JOB) is None
    assert coordinator._persist_producer_outcome(request, actor, "free", "plan_review", {"answer": "kept"})
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.get_row(plan_collection_job.CHANGED, "task")["producer"][1] == "free_response"


@pytest.mark.parametrize("arrival", ["before-transaction", "during-transaction"])
def test_plan_wake_flush_preserves_a_racing_publication(tmp_path, monkeypatch, arrival):
    from tests.test_child_duty_rest import measured_rest

    write_task_result(tmp_path, "task", "completed", plan_record=_record([_participant("seat")]))
    maintenance.run_job(tmp_path, plan_collection_job.JOB)
    old, new = ["task", "old", "digest-old"], ["task", "new", "digest-new"]
    with busy_store(tmp_path):
        plan_collection_job.producer_published(tmp_path, old)
    open_store, put_row, arrived = store.open_store, store.Txn.put_row, []

    def publish():
        if not arrived:
            arrived.append(True)
            plan_collection_job.producer_published(tmp_path, new)

    @contextlib.contextmanager
    def opening(root, **kwargs):
        if kwargs.get("write") and arrival == "before-transaction":
            publish()
        with open_store(root, **kwargs) as txn:
            yield txn

    def writing(txn, set_name, key, facts):
        result = put_row(txn, set_name, key, facts)
        if set_name == plan_collection_job.CHANGED and arrival == "during-transaction":
            publish()
        return result

    with monkeypatch.context() as race:
        race.setattr(store, "open_store", opening)
        race.setattr(store.Txn, "put_row", writing)
        assert plan_collection_job.JOB in maintenance.due(tmp_path)
    assert arrived == [True]
    assert plan_collection_job.JOB in maintenance.due(tmp_path)
    with open_store(tmp_path, write=False) as txn:
        assert txn.get_row(plan_collection_job.CHANGED, "task") == {"producer": new}
    assert maintenance.run_job(tmp_path, plan_collection_job.JOB)["state"] == "done"
    measured_rest(tmp_path, plan_collection_job.JOB)


def test_supervision_save_does_not_wait_for_the_retirement_store(tmp_path):
    with store.open_store(tmp_path, write=True):
        pass
    path = tmp_path / "state/delegate_supervision/task.json"
    view = supervision.run_view(supervision.read(path), "run")
    view["journal_cursor"] = 3
    with busy_store(tmp_path):
        started = time.monotonic()
        supervision.save_view(path, tmp_path, "task", view)
        elapsed = time.monotonic() - started
    assert supervision.read(path)["runs"]["run"]["journal_cursor"] == 3
    assert elapsed < 1
    assert maintenance_jobs.job(tmp_path, "runtime_retirement") is None
    # Startup discovery owns a missed artifact notification.
    from ouroboros import maintenance_retirement
    maintenance_retirement.record_startup(tmp_path, gaps=True)
    assert maintenance.run_job(tmp_path, "runtime_retirement")["state"] == "done"
    view["journal_cursor"] = 4
    supervision.save_view(path, tmp_path, "task", view)
    assert supervision.read(path)["runs"]["run"]["journal_cursor"] == 4
    assert maintenance_jobs.job(tmp_path, "runtime_retirement")["arrival"]


@pytest.mark.parametrize("exclusive", [False, True], ids=["busy-writer", "busy-reader"])
def test_metrics_append_survives_a_busy_projection_and_repairs_once(tmp_path, monkeypatch, exclusive):
    # PERSIST makes the EXCLUSIVE case block reads too, covering the pre-append lookup.
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    subagent_metrics.publish(tmp_path, event("before"))
    with busy_store(tmp_path, exclusive):
        started = time.monotonic()
        subagent_metrics.publish(tmp_path, event("busy"))
        elapsed = time.monotonic() - started
    events = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["producer_id"] for row in events if row.get("type") == subagent_metrics.EVENT] == ["before", "busy"]
    assert elapsed < 1
    maintenance_jobs.record(tmp_path, subagent_metrics.JOB, state="pending")
    assert maintenance.run_job(tmp_path, subagent_metrics.JOB)["state"] == "done"
    subagent_metrics.publish(tmp_path, event("free"))
    subagent_metrics.publish(tmp_path, event("busy"))
    facts = subagent_metrics.read(tmp_path, "worker")
    assert facts["rows"][0]["dispatched"] == 3
    assert facts["coverage"]["stream"] == "complete"
    assert maintenance.run_job(tmp_path, subagent_metrics.JOB) is None
