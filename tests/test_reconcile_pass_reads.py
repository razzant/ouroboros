"""The 300 s reconcile pass reads what it can act on, and reads it once (FINAL_PLAN E2).

tests/test_cost_live_set.py holds the pass flat on a booted server as the live set grows and at rest over the
largest set; these are the same laws at unit size, each on its own mechanism:

- the periodic orphan sweep never projects a row the queue owns: its own write guard
  (``orphan_reconcile_write_guard``) refuses that row whatever the projection says, so the server
  excludes the queue's ids unread (``server_maintenance.queue_owned_task_ids``);
- what the sweep does read, it reads once per pass: the candidates' documents in one batch
  (``obligations.result_rows``), one queue document, one events tail (``task_status._SweepReads``);
- the preparation recovery reads a row again only after a write moved its stamp
  (``task_store.owing_documents``): a receipt no claim marked owes it nothing until then.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import obligations, store, task_results, task_status, task_store
from ouroboros.task_results import STATUS_REQUESTED, STATUS_SCHEDULED, load_task_result, write_task_result
from ouroboros.utils import append_jsonl
from tests._shared import write_queue_document


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _document_reads(monkeypatch) -> list:
    """Every task document the code under test reads: one entry per id, batched or alone."""
    reads: list = []
    batch, single = task_store.documents_by_ids, task_store.read_document

    def batched(root, task_ids):
        task_ids = list(task_ids)
        reads.extend(task_ids)
        return batch(root, task_ids)

    def alone(root, task_id, **kwargs):
        reads.append(task_id)
        return single(root, task_id, **kwargs)

    monkeypatch.setattr(task_store, "documents_by_ids", batched)
    monkeypatch.setattr(task_store, "read_document", alone)
    monkeypatch.setattr(task_results, "read_document", alone)
    return reads


# ---- one batch, gaps kept ------------------------------------------------------------------------------------------

def test_a_pass_reads_its_members_in_one_batch(tmp_path, monkeypatch):
    ids = [f"run{index}" for index in range(5)]
    for task_id in ids:
        write_task_result(tmp_path, task_id, "running")
    batches, singles = [], []
    batch, single = task_store.documents_by_ids, task_store.read_document
    monkeypatch.setattr(task_store, "documents_by_ids", lambda root, task_ids: batches.append(list(task_ids))
                        or batch(root, task_ids))
    monkeypatch.setattr(task_results, "read_document", lambda root, task_id, **kwargs: singles.append(task_id)
                        or single(root, task_id, **kwargs))
    rows = list(obligations.result_rows(tmp_path, "nonterminal"))
    assert sorted(row["task_id"] for row in rows) == ids
    assert batches == [ids] and singles == [], "five members, one read"


def test_a_damaged_member_stays_a_repair_gap_and_hides_no_other(tmp_path):
    write_task_result(tmp_path, "good", "running")
    write_task_result(tmp_path, "bad", "running")
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("UPDATE task_bodies SET body='{' WHERE task_id='bad'")
    rows = list(obligations.result_rows(tmp_path, "nonterminal"))
    assert [row["task_id"] for row in rows] == ["good"]
    assert obligations.members(tmp_path, "unknowns")["task:bad"]["reason"] in {"ValueError", "JSONDecodeError"}


# ---- the queue's own rows are not the sweep's ------------------------------------------------------------------------

def test_the_periodic_sweep_excludes_what_the_queue_owns_unread(monkeypatch):
    from ouroboros import projects_registry, skill_review_runner
    from ouroboros import server_maintenance as sm
    from supervisor import queue, workers

    monkeypatch.setattr(queue, "INITIALIZED", True)
    monkeypatch.setattr(queue, "RUNNING", {"r1": {"task": {"id": "r1"}}})
    monkeypatch.setattr(queue, "PENDING", [{"id": "p1"}])
    monkeypatch.setattr(workers, "WORKERS", {7: SimpleNamespace(busy_task_id="b1"), 8: SimpleNamespace(busy_task_id=None)})
    assert sm.queue_owned_task_ids() == {"r1", "p1", "b1"}
    seen: dict = {}
    monkeypatch.setattr(task_status, "reconcile_orphaned_running_tasks", lambda root, **kwargs: seen.update(kwargs) or 0)
    monkeypatch.setattr(skill_review_runner, "reconcile_stale_review_jobs", lambda root: None)
    monkeypatch.setattr(projects_registry, "reconcile_projects", lambda root: None)
    monkeypatch.setattr(sm, "_resume_interrupted_project_deletions", lambda: None)
    sm._periodic_zombie_reconcile()
    assert seen["exclude_task_ids"] == {"r1", "p1", "b1"} and seen["boot"] is False
    assert seen["write_guard"] is not None, "the exclusion stands only beside the guard that refuses those rows"
    monkeypatch.setattr(queue, "INITIALIZED", False)
    assert sm.queue_owned_task_ids() == frozenset(), "with no supervisor here, its guard refuses nothing"


@pytest.mark.parametrize("live", [1, 20])
def test_a_sweep_over_the_queue_s_rows_reads_no_document(tmp_path, monkeypatch, live):
    """Equal at 1 and 20 rows the queue owns: none is read (the same law the live-set axis holds)."""
    ids = [f"queued{index}" for index in range(live)]
    for task_id in ids:
        write_task_result(tmp_path, task_id, "running")
    reads = _document_reads(monkeypatch)
    assert task_status.reconcile_orphaned_running_tasks(tmp_path, exclude_task_ids=frozenset(ids), boot=False) == 0
    assert reads == []
    assert {load_task_result(tmp_path, task_id)["status"] for task_id in ids} == {"running"}


# ---- what the sweep reads, it reads once ---------------------------------------------------------------------------

def _orphan(root, task_id: str, now: float) -> None:
    """A pooled row whose worker restarted after its last event, the queue fresh and without it: a proven orphan."""
    from ouroboros.task_status import execution_owner_record

    owner = execution_owner_record(root, {"id": task_id, "_attempt": 1}, "pooled")
    write_task_result(root, task_id, "running", task_attempt=1, ts=_stamp(now - 200), execution_owner=owner)


@pytest.mark.parametrize("orphans", [1, 4])
def test_a_sweep_reads_the_queue_document_and_the_events_tail_once(tmp_path, monkeypatch, orphans):
    now = time.time()
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    for index in range(orphans):
        _orphan(tmp_path, f"orphan{index}", now)
    write_queue_document(tmp_path / "state" / "queue_snapshot.json", {"ts": _stamp(now), "pending": [], "running": []})
    append_jsonl(tmp_path / "logs" / "events.jsonl", {"ts": _stamp(now - 100), "type": "worker_boot"})
    queue_reads, tails = [], []
    real_queue, real_tail = task_status._load_queue_snapshot, task_status.iter_jsonl_objects
    monkeypatch.setattr(task_status, "_load_queue_snapshot", lambda root: queue_reads.append(root) or real_queue(root))
    monkeypatch.setattr(task_status, "iter_jsonl_objects", lambda *a, **k: tails.append(1) or real_tail(*a, **k))
    assert task_status.reconcile_orphaned_running_tasks(tmp_path, boot=False) == orphans
    # One read decides every row; each write then checks the queue afresh under its own row lock.
    assert len(queue_reads) == 1 + orphans, queue_reads
    assert len(tails) == 1
    assert {load_task_result(tmp_path, f"orphan{index}")["reason_code"] for index in range(orphans)} == {
        "orphaned_running_after_worker_restart"}


def test_an_orphan_under_an_active_cancel_intent_is_left_to_its_custody(tmp_path):
    from ouroboros.cancel_intents import request_cancel

    now = time.time()
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    _orphan(tmp_path, "cancelling", now)
    _orphan(tmp_path, "plain", now)
    write_queue_document(tmp_path / "state" / "queue_snapshot.json", {"ts": _stamp(now), "pending": [], "running": []})
    append_jsonl(tmp_path / "logs" / "events.jsonl", {"ts": _stamp(now - 100), "type": "worker_boot"})
    assert request_cancel(tmp_path, "cancelling", reason="owner stop", allow_settled_target=True)
    assert task_status.reconcile_orphaned_running_tasks(tmp_path, boot=False) == 1
    assert load_task_result(tmp_path, "cancelling")["status"] == "running"
    assert load_task_result(tmp_path, "plain")["status"] == "failed"


# ---- the preparation recovery reads a row once per stamp ------------------------------------------------------------

def _owes(document) -> bool:
    return bool(document.get("marked")) or document.get("status") == STATUS_REQUESTED


def test_an_unchanged_row_is_read_once_per_stamp(tmp_path, monkeypatch):
    write_task_result(tmp_path, "plain", STATUS_SCHEDULED)
    write_task_result(tmp_path, "marked", STATUS_SCHEDULED, marked=True)
    write_task_result(tmp_path, "asked", STATUS_REQUESTED)
    write_task_result(tmp_path, "done", "completed")
    reads = _document_reads(monkeypatch)
    fetched: list = []
    real_fetch = task_store._fetch
    monkeypatch.setattr(task_store, "_fetch", lambda txn, where="", params=(): fetched.append(params) or real_fetch(txn, where, params))
    first = task_store.owing_documents(tmp_path, (STATUS_REQUESTED, STATUS_SCHEDULED), _owes)
    assert [row["task_id"] for row in first] == ["asked", "marked"]
    assert len(fetched) == 1 and reads == []
    first[0]["status"] = "mutated by its reader"
    again = task_store.owing_documents(tmp_path, (STATUS_REQUESTED, STATUS_SCHEDULED), _owes)
    assert len(fetched) == 1, "no row moved: no body is read"
    assert [(row["task_id"], row["status"]) for row in again] == [("asked", STATUS_REQUESTED), ("marked", STATUS_SCHEDULED)]
    write_task_result(tmp_path, "plain", STATUS_SCHEDULED, marked=True)  # the write that makes it owed
    moved = task_store.owing_documents(tmp_path, (STATUS_REQUESTED, STATUS_SCHEDULED), _owes)
    assert [row["task_id"] for row in moved] == ["asked", "marked", "plain"]
    assert len(fetched) == 2 and '"plain"' in fetched[-1][0] and '"marked"' not in fetched[-1][0]
    write_task_result(tmp_path, "asked", "completed")  # it left the selection
    assert [row["task_id"] for row in task_store.owing_documents(
        tmp_path, (STATUS_REQUESTED, STATUS_SCHEDULED), _owes)] == ["marked", "plain"]


def test_a_read_that_fails_keeps_nothing(tmp_path, monkeypatch):
    write_task_result(tmp_path, "plain", STATUS_SCHEDULED)
    real_fetch = task_store._fetch

    def broken(*_args, **_kwargs):
        raise store.StoreError("store unavailable")

    monkeypatch.setattr(task_store, "_fetch", broken)
    with pytest.raises(store.StoreError):
        task_store.owing_documents(tmp_path, (STATUS_SCHEDULED,), _owes)
    fetched: list = []
    monkeypatch.setattr(task_store, "_fetch", lambda txn, where="", params=(): fetched.append(params) or real_fetch(txn, where, params))
    task_store.owing_documents(tmp_path, (STATUS_SCHEDULED,), _owes)
    assert len(fetched) == 1, "the failed read left no stamp to rest on"


@pytest.mark.parametrize("receipts", [1, 20])
def test_the_preparation_recovery_rests_over_unmarked_receipts(tmp_path, monkeypatch, receipts):
    """Equal at 1 and 20 SCHEDULED receipts no claim marked: the second pass reads no body."""
    from supervisor import workspace_preparation as prep

    for index in range(receipts):
        write_task_result(tmp_path, f"receipt{index}", STATUS_SCHEDULED)
    assert prep.recover_interrupted_preparations(tmp_path) == {
        "settled": [], "retained": [], "admitted": [], "uncertain": [], "live": [], "errors": []}
    fetched: list = []
    real_fetch = task_store._fetch
    monkeypatch.setattr(task_store, "_fetch", lambda txn, where="", params=(): fetched.append(params) or real_fetch(txn, where, params))
    prep.recover_interrupted_preparations(tmp_path)
    assert fetched == []
