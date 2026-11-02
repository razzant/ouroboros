"""Pending plan discovery reuses bounded SQLite collection without borrowing acceptance authority."""
from __future__ import annotations

import copy
import json

import pytest

from ouroboros import evidence_packs, late_review_marks, plan_record, store, store_import, task_store
from tests._store_journal_shared import store_state
from tests import test_late_review_marks as late
from tests.test_late_review_marks import council as council
from tests.test_plan_record import _participant, _record
from tests.test_plan_task_tool import harness as harness
from tests.test_review_collection_index import _combined_schema, _legacy_store, _old_replace, _operation
from tests.test_store import document, publish
from tests.test_store_cutover import _plan, captured_statements


def _pending(**fields):
    return {**_participant("seat"), **fields}


def _flags(root, task_id):
    with store.open_store(root, write=False) as txn:
        return txn.row(task_id, "review_collection_pending", "plan_collection_pending", "write_seq")


@pytest.mark.parametrize(("raw", "expected"), [
    (None, False), ({}, False), (plan_record.empty_record(), False),
    (_record([_pending()]), True),
    (_record([_pending(owner_marked=False, dispatched=False, unresolved=True)]), True),
    (_record([_pending()], [_participant("later", status="answered")], revisions=2), True),
    *[(_record([_participant("seat", status=status)]), False)
      for status in ("answered", "unparseable", "failed", "not_dispatched")],
    ("broken", True), ({"schema_version": 99, "asks": []}, True),
    ({"schema_version": 3, "asks": "broken"}, True),
    ({"schema_version": 3, "asks": [None]}, True),
    ({"schema_version": 3, "asks": [{"participants": {}}]}, True),
    (_record([{"seat_id": "unknown", "status": "future_status"}]), True),
])
def test_plan_selection_is_pure_and_conservative(tmp_path, monkeypatch, raw, expected):
    def forbidden(*_args, **_kwargs):
        pytest.fail("selection must not read sources or fully validate/copy history")

    before = copy.deepcopy(raw)
    with monkeypatch.context() as patch:
        patch.setattr(plan_record, "validate_record", forbidden)
        patch.setattr(plan_record, "read_document", forbidden)
        assert plan_record.collection_pending(raw) is expected
    assert raw == before
    row = document("task", "running", plan_record=raw)
    publish(tmp_path, row)  # Unknown shape is retained, never a writer failure.
    assert _flags(tmp_path, "task") == (0, int(expected), 1)
    assert task_store.read_document(tmp_path, "task") == row


@pytest.mark.parametrize("tier", [store.TIER_ENFORCED, store.TIER_NAME])
def test_schema_three_false_acceptance_zero_never_hides_pending_plan_after_old_write(tmp_path, tier):
    original = document("task", plan_record=_record([_pending()]), result="whole historical answer")
    _legacy_store(tmp_path, original, owned_root=tmp_path, tier=tier, shape="v3")
    before = store_state(store.store_path(tmp_path))  # rows and any write's trace; a first WAL open's header aside
    assert task_store.read_document(tmp_path, "task") == original
    assert store_state(store.store_path(tmp_path)) == before, "passive reads must not upgrade"
    store_import.upgrade_current_schema(tmp_path)
    assert _flags(tmp_path, "task") == (0, None, 1)
    assert list(task_store.plan_collection_candidates(tmp_path)) == ["task"]
    assert _flags(tmp_path, "task") == (0, 1, 1)
    with store.open_store(tmp_path, write=True) as txn:
        # This actual predecessor knows acceptance and owner questions, but not plan selection.
        _old_replace(txn.conn, original, known=("has_owner_questions", "review_collection_pending"))
        assert txn.meta("schema_version") == store.SCHEMA_VERSION
    assert _flags(tmp_path, "task") == (0, None, 2)
    assert list(task_store.plan_collection_candidates(tmp_path)) == ["task"]
    assert _flags(tmp_path, "task") == (0, 1, 2)
    assert task_store.read_document(tmp_path, "task") == original


def test_actual_plan_writes_set_selection_until_the_last_pending_seat_settles(tmp_path):
    publish(tmp_path, document("task", "running", plan_record=_record()))
    ask = _record([_pending(), _participant("critic", owner_marked=False)])["asks"][0]
    plan_record.append_ask(tmp_path, "task", ask)
    assert _flags(tmp_path, "task") == (0, 1, 2)
    plan_record.update_participant(tmp_path, "task", ask_id="a1", seat_id="seat", facts={"status": "answered"})
    assert _flags(tmp_path, "task") == (0, 1, 3)
    plan_record.update_participant(tmp_path, "task", ask_id="a1", seat_id="critic", facts={"status": "failed"})
    assert _flags(tmp_path, "task") == (0, 0, 4)
    plan_record.update_participant(tmp_path, "task", ask_id="a1", seat_id="critic", facts={"status": "pending"})
    assert _flags(tmp_path, "task") == (0, 0, 5)
    assert list(plan_record.pending_participants(plan_record.load_plan_record(tmp_path, "task"))) == []


def test_schema_three_first_child_write_honors_budget_before_callback(tmp_path):
    child = tmp_path / "child"
    original = document("task", plan_record=_record())
    _legacy_store(child, original, owned_root=tmp_path, shape="v3")
    calls = []

    def change(row):
        with store.open_store(child, write=False) as txn:
            _combined_schema(txn)
        calls.append(row)
        return {**row, "plan_record": _record([_pending()])}

    with store.open_store(child, write=True):
        with store.acquisition_budget(0), pytest.raises(store.StoreBusy):
            task_store.patch_document(child, "task", change)
        with store.acquisition_budget(0):
            assert list(task_store.plan_collection_candidates(child)) == []
    assert calls == []
    with store.open_store(child, write=False) as txn:
        assert txn.meta("schema_version") == 3
    task_store.patch_document(child, "task", change)
    assert calls == [original] and _flags(child, "task") == (0, 1, 2)


def test_schema_three_interrupted_upgrade_rolls_back_body_indexes_and_metadata(tmp_path, monkeypatch):
    original = document("task", plan_record=_record([_pending()]))
    _legacy_store(tmp_path, original, owned_root=tmp_path, shape="v3")
    real = store.Txn.set_meta

    def interrupted(txn, key, value):
        real(txn, key, value)
        if key == "schema_version":
            assert txn.row("task", "plan_collection_pending") is None
            raise OSError("after DDL, before commit")

    with monkeypatch.context() as patch:
        patch.setattr(store.Txn, "set_meta", interrupted)
        with pytest.raises(OSError):
            store_import.upgrade_current_schema(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == 3
        assert "plan_collection_pending" not in {r["name"] for r in txn.conn.execute("PRAGMA table_info(tasks)")}
        assert "tasks_plan_collection" not in {r["name"] for r in txn.conn.execute("PRAGMA index_list(tasks)")}
        assert txn.load("task") == original and txn.row("task", "write_seq") == 1
    store_import.upgrade_current_schema(tmp_path)
    store_import.upgrade_current_schema(tmp_path)
    assert _flags(tmp_path, "task") == (0, None, 1)


@pytest.mark.parametrize("touch", ["body", "handoff"])
def test_off_lock_classification_cas_preserves_a_concurrent_pending_plan(tmp_path, monkeypatch, touch):
    _legacy_store(tmp_path, document("task", plan_record=_record()), owned_root=tmp_path, shape="v3")
    store_import.upgrade_current_schema(tmp_path)
    predicate = plan_record.collection_pending
    touched = []

    def classify(raw):
        if not touched:
            touched.append(True)
            # A real other-connection COMMIT fails if the classifier still holds its read lock.
            with monkeypatch.context() as patch:
                patch.setattr(plan_record, "collection_pending", predicate)
                if touch == "body":
                    with store.open_store(tmp_path, write=True, wait_sec=0) as txn:
                        txn.store_task(document("task", plan_record=_record([_pending()])))
                else:
                    task_store.patch_handoff(tmp_path, "task", "model_waits", "wait", lambda _: {"state": "closed"})
        return predicate(raw)

    monkeypatch.setattr(plan_record, "collection_pending", classify)
    assert list(task_store.plan_collection_candidates(tmp_path)) == []
    assert _flags(tmp_path, "task") == (0, 1 if touch == "body" else None, 2)
    assert list(task_store.plan_collection_candidates(tmp_path)) == (["task"] if touch == "body" else [])
    assert _flags(tmp_path, "task") == (0, 1 if touch == "body" else 0, 2)


def test_known_plan_work_uses_its_own_index_before_unknown_history(tmp_path, monkeypatch):
    _legacy_store(tmp_path, *(document(f"old-{i}", plan_record=_record()) for i in range(65)),
                  owned_root=tmp_path, shape="v3")
    task_store.patch_document(tmp_path, "z-plan", lambda _: document("z-plan", plan_record=_record([_pending()])))
    task_store.patch_document(tmp_path, "acceptance", lambda _: document("acceptance", review_operations={"op": _operation()}))
    seen = []

    def first():
        candidates = iter(task_store.plan_collection_candidates(tmp_path))
        seen.append(next(candidates))
        candidates.close()

    sql = captured_statements(monkeypatch, first)
    assert seen == ["z-plan"] and not any("task_bodies" in statement for statement, _ in sql)
    selected = [(statement, params) for statement, params in sql if "plan_collection_pending" in statement]
    assert selected
    assert all(any("tasks_plan_collection" in step for step in _plan(tmp_path, statement, params))
               for statement, params in selected)
    assert _flags(tmp_path, "old-0") == (0, None, 1)
    assert "plan_collection_pending" not in store.FLAG_COLUMNS.values()
    publish(tmp_path / "packer", document("quiet"))
    with store.open_store(tmp_path / "packer", write=True) as txn:
        txn.conn.execute("UPDATE tasks SET plan_collection_pending=1 WHERE task_id='quiet'")
    with store.open_store(tmp_path / "packer", write=False) as txn:
        assert evidence_packs.task_packable(txn, "quiet"), "selector is not a new retention duty"


def test_stopped_plan_classification_keeps_unknown_rows_for_next_pass(tmp_path, monkeypatch):
    _legacy_store(tmp_path, document("one"), document("two"), owned_root=tmp_path, shape="v3")
    store_import.upgrade_current_schema(tmp_path)
    predicate, stopped = plan_record.collection_pending, []

    def classify(raw):
        stopped.append(True)
        return predicate(raw)

    monkeypatch.setattr(plan_record, "collection_pending", classify)
    assert list(task_store.plan_collection_candidates(tmp_path, stop=lambda: bool(stopped))) == []
    assert _flags(tmp_path, "one") == (0, None, 1) and _flags(tmp_path, "two") == (0, None, 1)
    assert list(task_store.plan_collection_candidates(tmp_path)) == []
    assert _flags(tmp_path, "one") == (0, 0, 1) and _flags(tmp_path, "two") == (0, 0, 1)


def test_real_late_collector_revisits_only_pending_plan_after_legacy_history(council, monkeypatch):
    root = council.h.drive
    with store.open_store(root, write=True) as txn:
        for i in range(130):
            txn.store_task(document(f"old-{i:03}", plan_record=_record([_participant("done", status="answered")]),
                                    result="retained answer" * 500))
        txn.conn.execute("UPDATE tasks SET plan_collection_pending=NULL WHERE task_id LIKE 'old-%'")
    commits, original_commit, totals = [], store.commit, {}

    def committed(conn):
        # total_changes is cumulative on a cached connection: this transaction's rows are the step.
        total = conn.total_changes
        commits.append(total - totals.get(id(conn), (conn, 0))[1])
        totals[id(conn)] = (conn, total)
        return original_commit(conn)

    store.close_connections()  # fresh connections count from zero

    with monkeypatch.context() as patch:
        patch.setattr(store, "commit", committed)
        assert late_review_marks.replay_owed_late_reviews(root)["settled"] == 0
        assert late_review_marks._collect_ended_plan_asks(root, None) == 0  # child-owned collection
    assert sum(changes > 0 for changes in commits) == 3, "legacy classification commits by bounded batch, not per row"
    assert _flags(root, "old-000") == (0, 0, 1)
    assert _flags(root, late.PLAN_TASK)[1] == 1, "a running author retains its pending seat"
    with monkeypatch.context() as lossy:
        lost = late._lose_settlement_callbacks(lossy)
        council.transport.release("m/a")
        late._until(lambda: lost)
    assert late._owed_rows(root) == {}
    late._end(council.h)
    reads, body_text = [], store.Txn.body_text

    def body(txn, task_id):
        assert not task_id.startswith("old-"), "periodic collection reread closed history"
        reads.append(task_id)
        return body_text(txn, task_id)

    with monkeypatch.context() as patch:
        patch.setattr(store.Txn, "body_text", body)
        statements = captured_statements(patch, lambda: late_review_marks._collect_ended_plan_asks(root, None))
    assert reads and set(reads) == {late.PLAN_TASK}
    assert late._participant(council.h, "s1")["status"] == "answered"
    assert len(late._plan_marks(root)) == 1 and council.transport.calls.count("m/a") == 1
    selected = [(sql, params) for sql, params in statements if "SELECT task_id FROM tasks" in sql and "plan_collection_pending" in sql]
    assert selected and all(any("tasks_plan_collection" in step for step in _plan(root, sql, params)) for sql, params in selected)
    before = _flags(root, "old-000")
    late_review_marks._collect_ended_plan_asks(root, None)
    assert len(late._plan_marks(root)) == 1 and _flags(root, "old-000") == before


@pytest.mark.parametrize("after_candidate", [False, True])
@pytest.mark.parametrize("error", [OSError, ValueError])
def test_lazy_selection_failure_preserves_real_settlement_and_remaining_pending_seats(
        council, monkeypatch, caplog, after_candidate, error):
    root = council.h.drive
    with monkeypatch.context() as lossy:
        lost = late._lose_settlement_callbacks(lossy)
        council.transport.release("m/a")
        late._until(lambda: lost)
    late._end(council.h)
    assert late._participant(council.h, "s1")["status"] == "pending" and late._owed_rows(root) == {}
    candidates = task_store.plan_collection_candidates

    def interrupted(*args, **kwargs):
        if after_candidate:
            selected = iter(candidates(*args, **kwargs))
            try:
                yield next(selected)  # The real collector settles this seat before iteration fails.
            finally:
                selected.close()
        raise error("candidate iteration unavailable")

    with monkeypatch.context() as fault:
        fault.setattr(task_store, "plan_collection_candidates", interrupted)
        settled = late_review_marks._collect_ended_plan_asks(root, None)
    assert settled == int(after_candidate)
    assert "plan records are unreadable; their pending seats stay pending" in caplog.text
    assert late._participant(council.h, "s1")["status"] == ("answered" if after_candidate else "pending")
    assert all(late._participant(council.h, seat)["status"] == "pending" for seat in ("s2", "s3"))
    assert _flags(root, late.PLAN_TASK)[1] == 1
    assert late_review_marks._collect_ended_plan_asks(root, None) == int(not after_candidate)
    [mark] = late._plan_marks(root)
    assert mark["review"]["subject"]["seat_id"] == "s1"
    assert late._participant(council.h, "s1")["status"] == "answered"
    assert late_review_marks._collect_ended_plan_asks(root, None) == 0
    assert late._plan_marks(root) == [mark] and council.transport.calls.count("m/a") == 1


@pytest.mark.parametrize("discovery", ["initial", "changed", "legacy"])
@pytest.mark.parametrize("read", ["plan", "terminal", "collection"])
@pytest.mark.parametrize("failure", ["busy", "invalid"])
def test_failed_plan_examination_retries_contention_and_rests_on_invalid_input(council, monkeypatch, discovery, read, failure):
    from ouroboros import maintenance, maintenance_jobs, plan_collection_job, task_results
    from ouroboros.tools import plan_ask
    from tests.test_child_duty_rest import measured_rest

    root = council.h.drive
    if discovery == "changed":
        assert maintenance.run_job(root, "ended_plan_asks")["state"] == "done"
        measured_rest(root, "ended_plan_asks")
    with monkeypatch.context() as lossy:
        lost = late._lose_settlement_callbacks(lossy)
        council.transport.release()
        late._until(lambda: len(lost) == 3)
    maintenance.due(root)  # drain any contended producer wake before selecting the examination's token
    late._end(council.h)
    with store.open_store(root, write=True) as txn:
        if discovery == "legacy":
            txn.delete_row(plan_collection_job.CHANGED, late.PLAN_TASK)  # an older writer queued no token
        wake = txn.get_row(plan_collection_job.CHANGED, late.PLAN_TASK)
    assert bool(wake) is (discovery != "legacy")
    faults = []

    def busy(*args, **kwargs):
        faults.append(True)
        error = store.StoreBusy if failure == "busy" else {
            "plan": plan_record.PlanRecordInvalid, "terminal": ValueError, "collection": OSError,
        }[read]
        raise error("this child pass cannot read the retained plan")

    monkeypatch.setattr(plan_ask, "dispatch_ask", lambda *a, **kw: pytest.fail("collection never buys another ask"))
    owner, attribute = {"plan": (plan_record, "read_document"),
                        "terminal": (task_results, "load_task_result"),
                        "collection": (plan_ask, "collect_pending")}[read]
    with monkeypatch.context() as fault:
        fault.setattr(owner, attribute, busy)
        state = "pending" if failure == "busy" else "done"
        assert maintenance.run_job(root, "ended_plan_asks") == {"state": state, "items": 0}
        if failure == "invalid":
            for _ in range(3):
                measured_rest(root, "ended_plan_asks")
    assert faults == [True]
    assert _flags(root, late.PLAN_TASK)[1] == 1
    assert all(late._participant(council.h, seat)["status"] == "pending" for seat in ("s1", "s2", "s3"))
    assert late._plan_marks(root) == []
    with store.open_store(root, write=False) as txn:
        assert txn.get_row(plan_collection_job.CHANGED, late.PLAN_TASK) == (wake if failure == "busy" else None)
    assert maintenance_jobs.job(root, "ended_plan_asks")["state"] == state

    # Contention retries at the next pass; a deterministic failure waits for the task's next write.
    if failure == "invalid":
        measured_rest(root, "ended_plan_asks")
        task_store.patch_document(root, late.PLAN_TASK, lambda row: {**row, "result": "repaired input"})
    assert maintenance.run_job(root, "ended_plan_asks") == {"state": "done", "items": 3}
    assert _flags(root, late.PLAN_TASK)[1] == 0
    assert all(late._participant(council.h, seat)["status"] == "answered" for seat in ("s1", "s2", "s3"))
    marks = late._plan_marks(root)
    assert {mark["review"]["subject"]["seat_id"] for mark in marks} == {"s1", "s2", "s3"}
    with store.open_store(root, write=False) as txn:
        assert txn.get_row(plan_collection_job.CHANGED, late.PLAN_TASK) is None
    measured_rest(root, "ended_plan_asks")
    assert late._plan_marks(root) == marks and sorted(council.transport.calls) == ["m/a", "m/b", "m/c"]


@pytest.mark.parametrize("publication", ["free", "busy", "lost-at-exit"])
def test_retained_plan_replies_recover_missed_wakes_without_new_task_writes(council, monkeypatch, publication):
    import threading
    import time
    from contextlib import nullcontext

    from ouroboros import maintenance, maintenance_jobs, maintenance_retirement, plan_collection_job
    from ouroboros.tools import plan_ask
    from tests.test_child_duty_rest import measured_rest
    from tests.test_producer_wake_contention import busy_store

    root = council.h.drive
    late._end(council.h)
    assert maintenance.run_job(root, plan_collection_job.JOB) == {"state": "done", "items": 0}
    # The first observation may publish newly discovered in-flight custody facts once.
    late._until(lambda: maintenance.run_job(root, plan_collection_job.JOB) is None)
    measured_rest(root, plan_collection_job.JOB)
    sequence = _flags(root, late.PLAN_TASK)[2]
    publish_wake, gate, elapsed, requested = plan_collection_job.producer_published, threading.Lock(), [], []
    publishing = threading.local()

    def publish(root, token):
        with gate, busy_store(root) if publication == "busy" else nullcontext():
            started = time.monotonic()
            publishing.active = True
            try:
                if publication != "lost-at-exit":  # death after durable reply, before a wake can survive
                    publish_wake(root, token)
                elapsed.append(time.monotonic() - started)
            finally:
                publishing.active = False

    with monkeypatch.context() as lossy:
        lost = late._lose_settlement_callbacks(lossy)
        lossy.setattr(plan_collection_job, "producer_published", publish)
        lossy.setattr(maintenance.child, "request",
                      lambda: requested.append(True) if getattr(publishing, "active", False) else None)
        council.transport.release()
        late._until(lambda: len(lost) == 3)
    assert len(elapsed) == 3 and max(elapsed) < 1
    assert len(requested) == (0 if publication == "lost-at-exit" else 3)
    assert _flags(root, late.PLAN_TASK) == (0, 1, sequence), "no task write rescued the publications"
    assert maintenance_jobs.job(root, plan_collection_job.JOB)["state"] == "done"
    if publication == "lost-at-exit":
        measured_rest(root, plan_collection_job.JOB)
        maintenance_retirement.record_startup(root, gaps=False)
    else:
        # The server's dispatch check must persist the wake before starting another process.
        assert plan_collection_job.JOB in maintenance.due(root)
        with store.open_store(root, write=False) as txn:
            assert txn.get_row(plan_collection_job.CHANGED, late.PLAN_TASK)
    monkeypatch.setattr(plan_ask, "dispatch_ask", lambda *a, **kw: pytest.fail("collection never buys another ask"))
    assert maintenance.run_job(root, plan_collection_job.JOB) == {"state": "done", "items": 3}
    assert all(late._participant(council.h, seat)["status"] == "answered" for seat in ("s1", "s2", "s3"))
    assert {mark["review"]["subject"]["seat_id"] for mark in late._plan_marks(root)} == {"s1", "s2", "s3"}
    measured_rest(root, plan_collection_job.JOB)
    maintenance_retirement.record_startup(root, gaps=False)
    measured_rest(root, plan_collection_job.JOB)  # boot does not wake settled history
    assert sorted(council.transport.calls) == ["m/a", "m/b", "m/c"]


def test_invalid_plan_rests_until_its_record_is_repaired(tmp_path):
    from ouroboros import maintenance, plan_collection_job
    from tests.test_child_duty_rest import measured_rest

    publish(tmp_path, document("invalid", plan_record={"schema_version": 99, "asks": []}))
    assert _flags(tmp_path, "invalid")[1] == 1
    assert maintenance.run_job(tmp_path, plan_collection_job.JOB) == {"state": "done", "items": 0}
    for _ in range(5):
        measured_rest(tmp_path, plan_collection_job.JOB)
    assert _flags(tmp_path, "invalid")[1] == 1, "examination does not clear plan debt"
    task_store.patch_document(tmp_path, "invalid", lambda row: {**row, "plan_record": _record()})
    assert _flags(tmp_path, "invalid")[1] == 0
    measured_rest(tmp_path, plan_collection_job.JOB)


def test_storeless_roundtrip_and_reimport_preserve_plan_and_derive_selector(tmp_path):
    original = document("task", plan_record=_record([_pending()]), result="exact historical text")
    source, restored = tmp_path / "source", tmp_path / "restored"
    publish(source, original)
    store_import.export_facts(source)
    exported = source / "task_results/task.json"
    copied = restored / "task_results/task.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(exported.read_bytes())
    store_import.ensure_current_store(restored, with_children=False)
    assert task_store.read_document(restored, "task") == original and _flags(restored, "task")[1] == 1
    closed = copy.deepcopy(original)
    closed["plan_record"]["asks"][0]["participants"][0]["status"] = "answered"
    copied.write_text(json.dumps(closed), encoding="utf-8")
    store_import.ensure_current_store(restored, with_children=False)
    assert task_store.read_document(restored, "task") == original  # boot only opens the store
    store_import.run_legacy_reconcile(restored)
    assert task_store.read_document(restored, "task") == closed and _flags(restored, "task")[1] == 0
