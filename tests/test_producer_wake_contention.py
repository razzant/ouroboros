"""Durable producers survive a busy current store; repair consumes their retained sources."""
import contextlib
import errno
import json
import logging
import os
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
def test_metrics_append_survives_a_busy_projection_and_repairs_once(tmp_path, monkeypatch, caplog, exclusive):
    # PERSIST makes the EXCLUSIVE case block reads too, covering the pre-append lookup.
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    asked, open_store, job = [], store.open_store, subagent_metrics.JOB
    monkeypatch.setattr(maintenance.child, "request", lambda: asked.append(True))
    caplog.set_level(logging.WARNING, logger=subagent_metrics.__name__)
    owed, taken = tmp_path / "state/subagent_metrics.owed", tmp_path / "state/subagent_metrics.taken"

    def warnings():
        return [record for record in caplog.records if record.name == subagent_metrics.__name__]

    def marks():
        return owed.exists(), taken.exists()

    def stream():
        seats = subagent_metrics.seat_facts(tmp_path, [("worker", "test/model")])["coverage"]["stream"]
        assert seats == subagent_metrics.read(tmp_path, "worker")["coverage"]["stream"]
        return seats

    def revision():
        with open_store(tmp_path, write=False) as txn:
            return txn.revision()

    def refused(producer):
        with busy_store(tmp_path, exclusive):
            started = time.monotonic()
            subagent_metrics.publish(tmp_path, event(producer, task_id="task-" + producer))
            return time.monotonic() - started

    # A free store folds the publication inline: nothing is marked, deferred, asked for or logged.
    subagent_metrics.publish(tmp_path, event("before"))
    assert maintenance.run_job(tmp_path, job) is None and not asked and not warnings()
    assert marks() == (False, False) and stream() == "complete"
    # A held store: the event is in the journal, and the wake the store refused is a mark beside it.
    assert refused("busy") < 1
    events = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["producer_id"] for row in events if row.get("type") == subagent_metrics.EVENT] == ["before", "busy"]
    assert marks() == (True, False) and asked
    # The held store is told nothing more: one line names the event.
    [line] = warnings()
    assert "task=task-busy" in line.getMessage() and "producer=busy" in line.getMessage() and line.exc_info is None
    # Readers show the fold that waits while the store's own row still says complete.
    assert stream() == "pending"
    # A look in another process: nothing this module could remember carries the debt, and it keeps none.
    remembered = [value for name, value in vars(subagent_metrics).items()
                  if not name.startswith("__") and isinstance(value, (set, dict, list))]
    for value in remembered:
        value.clear()
    assert job in maintenance.due(tmp_path) and not remembered
    # A store held again: the pass takes the mark, fails at its own commit and writes no report after it.
    writes, before = [], revision()

    @contextlib.contextmanager
    def counting(root, **kwargs):
        writes.append(bool(kwargs.get("write")))
        with open_store(root, **kwargs) as txn:
            yield txn

    with monkeypatch.context() as held, busy_store(tmp_path), store.acquisition_budget(0):
        held.setattr(store, "open_store", counting)
        with pytest.raises(store.StoreBusy):
            maintenance.run_job(tmp_path, job)
    assert writes.count(True) == 1 and revision() == before
    assert marks() == (False, True) and job in maintenance.due(tmp_path) and stream() == "pending"
    # A refusal while a taken mark waits joins it at the next capture. One that arrives after that capture
    # leaves a new mark: the pass's end drops only the mark it took. Under the journal's append lock the mark
    # is left once, before the event; without the lock the append may land after a capture, so it is left again.
    order = []

    def traced(name):
        call = getattr(subagent_metrics, name)

        def run(*args, **kwargs):
            order.append(name)
            return call(*args, **kwargs)
        return run

    for name in ("_mark", "append_jsonl", "_publish_suffix"):
        monkeypatch.setattr(subagent_metrics, name, traced(name))
    with monkeypatch.context() as unlocked:
        unlocked.setattr(subagent_metrics, "_journal_lock", lambda root: contextlib.nullcontext(False))
        refused("late")
    assert order == ["_mark", "append_jsonl", "_mark", "_publish_suffix", "_mark"] and marks() == (True, True)
    catch_up = subagent_metrics.catch_up

    def refused_after_capture(root, **kwargs):
        assert marks() == (False, True)
        del order[:]
        refused("later")
        assert order == ["_mark", "append_jsonl", "_publish_suffix", "_mark"] and marks() == (True, True)
        return catch_up(root, **kwargs)

    with monkeypatch.context() as racing:
        racing.setattr(subagent_metrics, "catch_up", refused_after_capture)
        assert maintenance.run_job(tmp_path, job)["state"] == "done"
    assert marks() == (True, False) and job in maintenance.due(tmp_path) and stream() == "pending"
    assert maintenance.run_job(tmp_path, job)["state"] == "done"
    assert marks() == (False, False) and maintenance.run_job(tmp_path, job) is None and stream() == "complete"
    assert subagent_metrics.read(tmp_path, "worker")["rows"][0]["dispatched"] == 4
    requests = len(asked)
    subagent_metrics.publish(tmp_path, event("free"))
    subagent_metrics.publish(tmp_path, event("busy", task_id="task-busy"))
    facts = subagent_metrics.read(tmp_path, "worker")
    assert facts["rows"][0]["dispatched"] == 5
    assert facts["coverage"]["stream"] == "complete"
    assert maintenance.run_job(tmp_path, job) is None
    assert len(asked) == requests and len(warnings()) == 3 and marks() == (False, False)


def test_a_metrics_mark_wakes_no_failed_row_and_an_unreadable_one_proves_nothing(tmp_path, monkeypatch):
    job, owed = subagent_metrics.JOB, tmp_path / "state/subagent_metrics.owed"

    def stream():
        return subagent_metrics.read(tmp_path, "worker")["coverage"]["stream"]

    subagent_metrics.publish(tmp_path, event())
    assert job not in maintenance.due(tmp_path) and stream() == "complete"
    owed.touch()
    assert job in maintenance.due(tmp_path) and stream() == "pending"
    # A failed row rests on its source alone: a torn tail must not wake the child at every look.
    maintenance_jobs.record(tmp_path, job, state="failed", source=subagent_metrics._source_stamp(tmp_path))
    assert job not in maintenance.due(tmp_path)
    maintenance_jobs.record(tmp_path, job, state="done")
    assert job in maintenance.due(tmp_path)
    # A mark that cannot be read proves nothing: a look takes no duty from it, a reader no complete stream.
    stat = os.stat

    def unreadable(path, *args, **kwargs):
        if path == owed:
            raise PermissionError(errno.EACCES, "unreadable", str(path))
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", unreadable)
    assert job not in maintenance.due(tmp_path) and stream() == "pending"


def test_a_metrics_mark_is_taken_only_under_the_journals_append_lock(tmp_path):
    """The pass renames the mark under the append lock a publisher holds from its mark to its append, so it never
    takes a mark whose event is still to come: while another holder has the lock, the mark stays where it is."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.utils import jsonl_append_lock_path

    owed, taken = tmp_path / "state/subagent_metrics.owed", tmp_path / "state/subagent_metrics.taken"
    owed.parent.mkdir(parents=True)
    owed.touch()
    lock_path = jsonl_append_lock_path(tmp_path / "logs/events.jsonl")
    lock = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0, poll_sec=0.01, owner_aware_stale=True)
    assert lock is not None
    try:
        subagent_metrics._take(tmp_path)
        assert (owed.exists(), taken.exists()) == (True, False), "a publisher between its mark and its append holds it"
    finally:
        release_exclusive_file_lock(lock_path, lock)
    subagent_metrics._take(tmp_path)
    assert (owed.exists(), taken.exists()) == (False, True), "with the lock free the pass takes the mark"
