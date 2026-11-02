"""Successful receipt reads release Q; publication still uses current authority."""
import copy
import inspect
import threading

import pytest

from ouroboros import task_results
from supervisor import queue_schedules as schedules, schedule_occurrence as occurrences
from tests.test_schedule_occurrence import _row, _rows, q as _q
from tests.test_schedule_responsiveness import automatic_pass, run_off_thread
from tests.test_g1_followup_policy import world as _world

q, world = _q, _world


def _save(q, row):
    with schedules.schedule_transaction(q.root):
        data = schedules.load_schedule_store(q.root)
        data["tasks"] = [row]
        schedules._write_scheduled_tasks(data, q.root)


@pytest.mark.parametrize("consumer", ["reconcile", "admission", "repair", "gc"])
def test_successful_receipt_read_allows_unrelated_queue_progress(q, monkeypatch, consumer):
    _row(q, intent={"kind": "system_repo"})
    if consumer == "admission":
        row = _rows(q)["s1"]
        claimed = occurrences.claim(row, row["trigger"]["run_at"])
        _save(q, row)
        prepared = occurrences.prepare(claimed)
        tid = claimed["occurrence"]["task_id"]
        target = lambda: occurrences.admit([prepared])
    else:
        automatic_pass()
        task = q.pending.pop()
        tid = task["id"]
        row = _rows(q)["s1"]
        if consumer == "reconcile":
            # A relationship awaiting resolution cannot run binding repair.
            row["followup_relation"] = {"kind": "unknown"}
        elif consumer == "repair":
            row.pop("occurrence")  # pre-occurrence legacy binding, still owed
            q.pending.append(task)
        elif consumer == "gc":
            task_results.write_task_result(q.root, tid, "completed", result="Finished")
            row.pop("occurrence")
            row.update(completed_at="2000-01-01T00:00:00+00:00", last_outcome={"task_id": tid},
                       followup_relation={"kind": "unknown"})
        _save(q, row)
        target = automatic_pass

    entered, release = threading.Event(), threading.Event()
    real_read = task_results.load_task_result
    observed = []

    def slow_read(root, task_id, **kw):
        result = real_read(root, task_id, **kw)  # successful read, NOT acquisition contention
        receipt_read = any((frame.function == "_read_back" and frame.filename.endswith("schedule_occurrence.py"))
                           or (frame.function == "load" and frame.filename.endswith("task_ownership.py"))
                           for frame in inspect.stack())
        if str(task_id) == tid and receipt_read and not entered.is_set():
            observed.append(q.queue._queue_lock._is_owned())
            entered.set()
            assert release.wait(10), "test failed to release its receipt reader"
        return result

    def unrelated():
        q.queue.enqueue_task({"id": "unrelated", "type": "task", "text": "Fixture",
                              "chat_id": 0, "_project_scope_none": True}, restoring_snapshot=True)
        assert q.queue.persist_queue_snapshot(reason="receipt_read_consumer") is True

    monkeypatch.setattr(task_results, "load_task_result", slow_read)
    reader, done, errors = run_off_thread(target, "receipt-consumer")
    competitor = None
    try:
        assert entered.wait(5), f"{consumer} never read its receipt"
        competitor, progressed, competitor_errors = run_off_thread(unrelated, "unrelated-enqueue")
        assert progressed.wait(2), f"{consumer} held Q during a successful receipt read"
        assert not done.is_set() and not release.is_set()
        assert not competitor_errors
        assert any(t["id"] == "unrelated" for t in q.pending)
        assert observed == [False]
    finally:
        release.set()
        reader.join(10)
        if competitor is not None:
            competitor.join(10)
    assert not errors and done.is_set()
    if consumer == "admission":
        assert [t["id"] for t in q.pending].count(tid) == 1
    elif consumer == "gc":
        assert "s1" not in _rows(q)


def _after_receipt(monkeypatch, tid, mutate):
    """A writer lands after the coherent body/stamp read, before Q is reacquired."""
    real_read, called = task_results.load_task_result, []

    def read(root, task_id, **kw):
        result = real_read(root, task_id, **kw)
        if task_id == tid and kw.get("signed") and not called:
            called.append(True)
            mutate()
        return result

    monkeypatch.setattr(task_results, "load_task_result", read)
    return called


@pytest.mark.parametrize("consumer", ["scan", "admit"])
@pytest.mark.parametrize("mutation", ["owner_hold", "cancelled"])
def test_changed_receipt_defers_without_replaying_or_persisting_a_false_hold(q, monkeypatch, consumer, mutation):
    _row(q, intent={"kind": "system_repo"})
    automatic_pass()
    task = q.pending.pop()
    before = _rows(q)["s1"]
    prepared = occurrences.prepare(occurrences.view(before, task))

    def mutate():
        assert not q.queue._queue_lock._is_owned()
        # Both writers can land on an unsent accepted task without changing its
        # schedule row. This does not pretend a worker ran without its row mark.
        task_results.write_task_result(q.root, task["id"],
            "cancelled" if mutation == "cancelled" else "scheduled",
            **({"_owner_hold": {"source": "owner", "revision": "new"}} if mutation == "owner_hold" else {}))

    called = _after_receipt(monkeypatch, task["id"], mutate)
    (automatic_pass if consumer == "scan" else lambda: occurrences.admit([prepared]))()
    assert called and not q.pending and _rows(q)["s1"] == before
    automatic_pass()
    automatic_pass()
    if mutation == "cancelled":
        assert not q.pending
        assert task_results.load_task_result(q.root, task["id"])["status"] == "cancelled"
    else:
        [restored] = q.pending
        assert restored["id"] == task["id"]
        assert restored["metadata"]["schedule_occurrence"]["token"] == before["occurrence"]["token"]
        assert restored["_owner_hold"]["revision"] == "new"
        assert occurrences.record_dispatch_possible(restored) is False


@pytest.mark.parametrize("mutation", ["edit", "disable", "delete", "restart", "return", "stop"])
def test_fresh_admission_rechecks_controls_after_receipt_observation(q, monkeypatch, mutation):
    from supervisor.followup_policy import record_restart

    _row(q, intent={"kind": "system_repo"})
    row = _rows(q)["s1"]
    claimed = occurrences.claim(row, row["trigger"]["run_at"])
    _save(q, row)
    prepared = occurrences.prepare(claimed)
    tid = claimed["occurrence"]["task_id"]
    stopped = threading.Event()

    def mutate():
        assert not q.queue._queue_lock._is_owned()
        if mutation == "edit":
            changed = copy.deepcopy(_rows(q)["s1"])
            changed["trigger"]["run_at"] = "2099-01-01T00:00:00+00:00"
            q.queue.upsert_scheduled_task(changed)
        elif mutation in {"disable", "delete"}:
            q.queue.mutate_scheduled_task(mutation, "s1", actor="owner", reason="Fixture")
        elif mutation == "restart":
            record_restart(q.root)
        elif mutation == "return":
            with q.queue._queue_lock:
                monkeypatch.setattr(q.queue, "RETURNS_NAMED", True)
        else:
            stopped.set()

    called = _after_receipt(monkeypatch, tid, mutate)
    occurrences.admit([prepared], stop=stopped.is_set)
    assert called and not q.pending
    assert not task_results.load_task_result(q.root, tid)
    if mutation in {"restart", "return", "stop"}:
        assert _rows(q)["s1"]["occurrence"]["token"] == claimed["occurrence"]["token"]
        monkeypatch.setattr(q.queue, "RETURNS_NAMED", False)
        stopped.clear()
        automatic_pass()
        automatic_pass()
        assert [t["id"] for t in q.pending] == [tid]
    else:
        automatic_pass()
        assert not q.pending


def test_delete_during_observation_preserves_the_already_accepted_occurrence(q, monkeypatch):
    _row(q, intent={"kind": "system_repo"})
    automatic_pass()
    task = q.pending.pop()
    called = _after_receipt(monkeypatch, task["id"], lambda: q.queue.mutate_scheduled_task(
        "delete", "s1", actor="owner", reason="Fixture"))
    automatic_pass()
    assert called and not q.pending
    assert _rows(q)["s1"]["delete_requested_at"]
    automatic_pass()
    automatic_pass()
    [restored] = q.pending
    assert restored["id"] == task["id"]
    assert restored["metadata"]["schedule_occurrence"] == task["metadata"]["schedule_occurrence"]
    assert occurrences.record_dispatch_possible(restored) is True


def test_original_stop_during_receipt_observation_still_blocks_related_admission(world, monkeypatch):
    from tests.test_g1_followup_policy import register, row, stop
    from supervisor import queue

    _, root, pending = world
    registered = register(world)
    automatic_pass()
    task = pending.pop()
    prepared = occurrences.prepare(occurrences.view(row(root, registered["id"]), task))
    called = _after_receipt(monkeypatch, task["id"], lambda: stop(root))
    occurrences.admit([prepared])
    assert called and not pending
    assert row(root, registered["id"])["followup_hold"]["reason"] == "origin_stopped"
    queue.check_scheduled_tasks()
    assert not pending


@pytest.mark.parametrize("writer", ["body", "handoff", "card"])
def test_receipt_signature_tracks_authority_writers_but_not_display_only_cards(q, writer):
    from ouroboros import task_store, task_activity_facts

    _row(q, intent={"kind": "system_repo"})
    automatic_pass()
    task = q.pending.pop()
    row = _rows(q)["s1"]
    reads = occurrences.prepare_receipts([row], drive_root=q.root)
    if writer == "body":
        task_results.write_task_result(q.root, task["id"], "scheduled", _owner_hold={"source": "owner"})
    elif writer == "handoff":
        task_store.patch_handoff(q.root, task["id"], "model_waits", "wait",
                                 lambda _: {"state": "waiting"})
        assert task_results.load_task_result(q.root, task["id"])["model_waits"]["wait"] == {"state": "waiting"}
    else:
        task_activity_facts.record_narration(q.root, {"task_id": task["id"], "narration": True,
            "content": "Display only", "ts": "2026-10-25T00:00:00+00:00"})
        assert task_results.load_task_result(q.root, task["id"])["last_line"] == "Display only"
    with q.queue._queue_lock:
        if writer == "card":
            assert occurrences.reconcile(row, reads=reads)[0] == "republish"
        else:
            assert occurrences.reconcile(row, reads=reads)[0] == "retry"


def test_busy_signature_defers_then_republishes_same_task_once(q, monkeypatch):
    from ouroboros import store, task_store

    _row(q, intent={"kind": "system_repo"})
    automatic_pass()
    task = q.pending.pop()
    before = _rows(q)["s1"]
    real = task_store.signature

    def busy(root, tid, **kw):
        if q.queue._queue_lock._is_owned():
            assert store._WAIT_BUDGET.get() == 0
            raise store.StoreBusy("fixture held store", reason="contention")
        return real(root, tid, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(task_store, "signature", busy)
        automatic_pass()
    assert not q.pending and _rows(q)["s1"] == before
    automatic_pass()
    automatic_pass()
    assert [t["id"] for t in q.pending] == [task["id"]]


def test_admission_uses_committed_document_without_second_body_read(q, monkeypatch):
    _row(q, intent={"kind": "system_repo"})
    written = []
    real_write, real_read = task_results.write_task_result, task_results.load_task_result

    def write(root, tid, status, **kw):
        stored = real_write(root, tid, status, **kw)
        if kw.get("schedule_admission"):
            written.append(tid)
        return stored

    def read(root, tid, **kw):
        assert not (tid in written and q.queue._queue_lock._is_owned()), "full receipt readback under Q"
        return real_read(root, tid, **kw)

    monkeypatch.setattr(task_results, "write_task_result", write)
    monkeypatch.setattr(task_results, "load_task_result", read)
    automatic_pass()
    [task] = q.pending
    assert written == [task["id"]]
    assert real_read(q.root, task["id"])["schedule_admission"]["token"] == _rows(q)["s1"]["occurrence"]["token"]


@pytest.mark.parametrize("evidence", ["missing", "unreadable", "busy", "changed"])
def test_retention_uses_the_same_strict_receipt_observation(q, monkeypatch, evidence):
    from tests._store_shared import delete_result
    from tests.test_schedule_responsiveness import held_store

    tid = "past"
    task_results.write_task_result(q.root, tid, "completed", result="Finished")
    row = {"id": "s1", "enabled": False, "trigger": {"type": "once"},
           "completed_at": "2000-01-01T00:00:00+00:00", "last_task_id": tid,
           "last_outcome": {"task_id": tid}, "followup_relation": {"kind": "unknown"}}
    _save(q, row)
    with monkeypatch.context() as patch:
        if evidence == "missing":
            # Bare absence this process never observed (task rows have no GC): the limited
            # legacy compatibility of a row with no occurrence, not proven settlement.
            delete_result(q.root, tid)
        elif evidence == "changed":
            _after_receipt(patch, tid, lambda: task_results.write_task_result(
                q.root, tid, "completed", result="Enriched terminal result"))
        elif evidence == "unreadable":
            real_read = task_results.load_task_result

            def read(root, task_id, **kw):
                if task_id == tid and kw.get("signed"):
                    raise ValueError("unreadable receipt")
                return real_read(root, task_id, **kw)

            patch.setattr(task_results, "load_task_result", read)
        if evidence == "busy":
            with held_store(q.root):
                automatic_pass()
        else:
            automatic_pass()
    assert ("s1" not in _rows(q)) is (evidence == "missing")
    if evidence != "missing":
        assert _rows(q)["s1"] == row
        automatic_pass()
        assert "s1" not in _rows(q)


@pytest.mark.parametrize("failure", ["none", "refused", "lost_reply"])
def test_binding_repair_publishes_committed_receipt_before_queue_binding(world, monkeypatch, failure):
    from ouroboros.store import StoreBusy
    from tests.test_g1_followup_policy import register

    _, root, pending = world
    register(world)
    automatic_pass()
    [task] = pending
    row = schedules.load_schedule_store(root)["tasks"][0]
    receipt = task_results.load_task_result(root, task["id"])["schedule_admission"]
    frozen = copy.deepcopy(receipt["task"])
    # An older admitted task still carries the relationship which the schedule
    # owner has since resolved. Repair may change only this unsent token.
    frozen["metadata"]["followup_relation"] = {"kind": "unknown"}
    task["metadata"]["followup_relation"] = {"kind": "unknown"}
    task_results.write_task_result(root, task["id"], "scheduled",
        metadata=frozen["metadata"], schedule_admission={**receipt, "task": frozen})
    real_write, attempts = task_results.write_task_result, []

    def write(root, tid, status, **kw):
        if kw.get("schedule_admission") and not attempts:
            attempts.append(tid)
            if failure == "refused":
                return False
            stored = real_write(root, tid, status, **kw)
            if failure == "lost_reply":
                raise StoreBusy("committed binding reply lost", reason="contention")
            return stored
        return real_write(root, tid, status, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(task_results, "write_task_result", write)
        automatic_pass()
    assert attempts == [task["id"]]
    from supervisor.followup_policy import scheduled_start

    if failure != "none":
        assert task["metadata"]["followup_relation"]["kind"] == "unknown"
        with scheduled_start(root, task) as allowed:
            assert allowed is False
        automatic_pass()
    assert task["metadata"]["followup_relation"] == row["followup_relation"]
    assert len(pending) == 1
    stored = task_results.load_task_result(root, task["id"])
    assert stored["schedule_admission"]["token"] == receipt["token"]
    assert stored["schedule_admission"]["task"]["metadata"] == task["metadata"]
    with scheduled_start(root, task) as allowed:
        assert allowed is True


@pytest.mark.parametrize("interruption", ["none", "origin_stop", "refused", "lost_reply"])
def test_unqueued_binding_repair_keeps_identity_controls_and_recovery(world, monkeypatch, interruption):
    from ouroboros.store import StoreBusy
    from tests.test_g1_followup_policy import register, stop
    from supervisor.followup_policy import scheduled_start

    _, root, pending = world
    register(world)
    automatic_pass()
    task = pending.pop()
    receipt = task_results.load_task_result(root, task["id"])["schedule_admission"]
    frozen = copy.deepcopy(receipt["task"])
    frozen["metadata"]["followup_relation"] = {"kind": "unknown"}
    task_results.write_task_result(root, task["id"], "scheduled",
        metadata=frozen["metadata"], schedule_admission={**receipt, "task": frozen})
    if interruption == "origin_stop":
        stop(root)
    real_write, attempts = task_results.write_task_result, []

    def write(root, tid, status, **kw):
        if kw.get("schedule_admission") and not attempts:
            attempts.append(tid)
            if interruption == "refused":
                return False
            stored = real_write(root, tid, status, **kw)
            if interruption == "lost_reply":
                raise StoreBusy("committed binding reply lost", reason="contention")
            return stored
        return real_write(root, tid, status, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(task_results, "write_task_result", write)
        automatic_pass()
    assert attempts == [task["id"]]
    if interruption != "none":
        assert not pending
        automatic_pass()
    if interruption == "origin_stop":
        assert not pending
        row = schedules.load_schedule_store(root)["tasks"][0]
        assert row["followup_hold"]["reason"] == "origin_stopped"
        return
    assert len(pending) == 1 and pending[0]["id"] == task["id"]
    assert pending[0]["text"] == task["text"]
    stored = task_results.load_task_result(root, task["id"])["schedule_admission"]
    assert stored["token"] == receipt["token"] and stored["task"] == pending[0]
    with scheduled_start(root, pending[0]) as allowed:
        assert allowed is True
