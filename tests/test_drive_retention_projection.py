"""Drive pruning selects small current facts before reading possible candidates."""
from __future__ import annotations

import json
import os
import time
import contextlib
from datetime import datetime, timezone

import pytest

from ouroboros import headless, retention, store, store_import, task_store
from ouroboros.task_results import write_task_result
from tests.test_review_collection_index import _legacy_store, _old_replace
from tests.test_store import document, publish
from tests.test_store_import import write_result

NOW = datetime(2026, 10, 16, tzinfo=timezone.utc).timestamp()
OLD = "2026-10-01T00:00:00+00:00"
YOUNG = "2026-10-16T00:00:00+00:00"
FUTURE = "2026-11-01T00:00:00+00:00"


def _layout(root, kind, task_id):
    if kind == "headless":
        drive = headless.prepare_task_drive(root, task_id, "empty")
        return drive.parent, drive, headless.prune_headless_task_drives
    drive = root / "task_drives" / task_id
    drive.mkdir(parents=True)
    return drive, drive, headless.prune_task_drives


def _flag(root, task_id):
    with store.open_store(root, write=False) as txn:
        return txn.row(task_id, "drive_retention_at", "write_seq")


def _body_spy(monkeypatch):
    real, reads = store.Txn.body_text, []

    def body(txn, task_id):
        raw = real(txn, task_id)
        reads.append((task_id, 0 if raw is None else len(raw)))
        return raw

    monkeypatch.setattr(store.Txn, "body_text", body)
    return reads


@pytest.mark.parametrize("kind", ["headless", "direct"])
@pytest.mark.parametrize("count,size", [(3, 128), (70, 100_000)])
def test_real_writer_young_and_nonterminal_bodies_are_not_read_on_either_pass(
        tmp_path, monkeypatch, kind, count, size):
    root = tmp_path / "data"
    for i in range(count):
        task_id = f"task-{i:03}"
        _directory, drive, prune = _layout(root, kind, task_id)
        write_task_result(root, task_id, "running" if i % 2 else "completed",
                          ts=YOUNG, result="x" * size, child_drive_root=str(drive))
    with store.open_store(root, write=False) as txn:
        before = txn.revision()
    reads = _body_spy(monkeypatch)
    for _pass in range(2):
        report = prune(root, retention_days=7, now=NOW, live=lambda _: False)
        assert report["scanned"] == count
        assert not report["pruned"] and not report["errors"]
        assert len(report["skipped"]) == count
    assert reads == [], "cost must not grow with skipped body size or count"
    with store.open_store(root, write=False) as txn:
        assert txn.revision() == before, "known hints add no periodic writes"


@pytest.mark.parametrize("kind", ["headless", "direct"])
@pytest.mark.parametrize("fields,removed", [
    ({"artifact_finalized_at": FUTURE, "completed_at": OLD, "ts": OLD}, False),
    ({"artifact_finalized_at": "bad", "completed_at": OLD, "ts": FUTURE}, True),
    ({"artifact_finalized_at": "", "completed_at": "bad",
      "finished_at": "2026-10-01T00:00:00", "ts": FUTURE}, True),
    ({"artifact_finalized_at": OLD, "completed_at": FUTURE, "ts": FUTURE}, True),
    ({"artifact_finalized_at": "bad", "completed_at": "bad",
      "finished_at": "bad", "ts": "bad"}, True),
    ({"ts": YOUNG}, False),
])
def test_exact_precedence_naive_utc_invalid_dates_and_layout_mtime(
        tmp_path, kind, fields, removed):
    root = tmp_path / "data"
    directory, drive, prune = _layout(root, kind, "task")
    publish(root, document("task", child_drive_root=str(drive), **fields))
    old = NOW - 20 * 86400
    os.utime(directory, (old, old))
    report = prune(root, retention_days=7, now=NOW, live=lambda _: False)
    assert not report["errors"], report
    assert bool(report["pruned"]) is removed
    assert directory.exists() is not removed


@pytest.mark.parametrize("kind", ["headless", "direct"])
def test_zero_days_and_cancelled_subagent_exemption_remain_real_settlements(tmp_path, kind):
    root = tmp_path / "data"
    rows = []
    for task_id, status, role, date in (
        ("before-now", "completed", "root", OLD),
        ("cancelled-child", "cancelled", "subagent", FUTURE),
        ("cancelled-root", "cancelled", "root", FUTURE),
    ):
        _directory, drive, prune = _layout(root, kind, task_id)
        rows.append(document(task_id, status, delegation_role=role,
                             child_drive_root=str(drive), ts=date))
    publish(root, *rows)
    report = prune(root, retention_days=0, now=NOW, live=lambda _: False)
    assert {row["task_id"] for row in report["pruned"]} == {"before-now", "cancelled-child"}
    assert report["skipped"] == [{"task_id": "cancelled-root", "reason": "younger_than_retention"}]


@pytest.mark.parametrize("kind", ["headless", "direct"])
def test_hint_race_is_rechecked_against_current_body_before_settlement(tmp_path, monkeypatch, kind):
    root = tmp_path / "data"
    directory, drive, prune = _layout(root, kind, "task")
    write_task_result(root, "task", "completed", ts=OLD, child_drive_root=str(drive))
    real = task_store.drive_retention_facts

    def select_then_enrich(*args, **kwargs):
        facts = real(*args, **kwargs)
        write_task_result(root, "task", "completed", artifact_finalized_at=FUTURE)
        return facts

    monkeypatch.setattr(task_store, "drive_retention_facts", select_then_enrich)
    report = prune(root, retention_days=7, now=NOW, live=lambda _: False)
    assert directory.is_dir() and not report["pruned"]
    assert report["skipped"] == [{"task_id": "task", "reason": "younger_than_retention"}]


@pytest.mark.parametrize("tier", [store.TIER_ENFORCED, store.TIER_NAME])
def test_schema_upgrade_is_lazy_and_old_writer_replace_invalidates_hint(tmp_path, monkeypatch, tier):
    root = tmp_path / "data"
    directory, drive, prune = _layout(root, "direct", "task")
    row = document("task", ts=YOUNG, child_drive_root=str(drive), result="old answer" * 1000)
    _legacy_store(root, row, owned_root=tmp_path, tier=tier, shape="v3")
    reads = _body_spy(monkeypatch)
    store_import.upgrade_current_schema(root)
    assert _flag(root, "task") == (None, 1)
    assert reads == [], "new projection must not backfill bodies during DDL"
    report = prune(root, retention_days=7, now=NOW, live=lambda _: False)
    assert not report["pruned"] and len(reads) == 1
    assert _flag(root, "task") == (retention.task_result_retention_at(row), 1)
    reads.clear()
    prune(root, retention_days=7, now=NOW, live=lambda _: False)
    assert reads == []
    # A real predecessor REPLACE uses only its own columns. Meta remains current,
    # yet the replacement must lose the new scalar instead of retaining old age.
    newer = {**row, "artifact_finalized_at": OLD}
    with store.open_store(root, write=True) as txn:
        _old_replace(txn.conn, newer, known=(
            "has_owner_questions", "review_collection_pending", "plan_collection_pending"))
        assert txn.meta("schema_version") == store.SCHEMA_VERSION
    assert _flag(root, "task") == (None, 2)
    store_import.upgrade_current_schema(root)
    assert _flag(root, "task") == (None, 2)
    assert prune(root, retention_days=7, now=NOW, live=lambda _: False)["pruned"]
    assert not directory.exists()


def test_legacy_classification_is_off_lock_and_cas_keeps_a_racing_writer(tmp_path, monkeypatch):
    root = tmp_path / "data"
    directory, drive, prune = _layout(root, "direct", "task")
    row = document("task", ts=OLD, child_drive_root=str(drive))
    _legacy_store(root, row, owned_root=tmp_path, shape="v3")
    store_import.upgrade_current_schema(root)
    original, touched = retention.task_result_retention_at, []

    def classify(document):
        if not touched:
            touched.append(True)
            # Another connection's actual write cannot commit under a held read lock.
            with monkeypatch.context() as patch:
                patch.setattr(retention, "task_result_retention_at", original)
                with store.open_store(root, write=True, wait_sec=0) as txn:
                    txn.store_task({**document, "artifact_finalized_at": FUTURE})
        return original(document)

    monkeypatch.setattr(retention, "task_result_retention_at", classify)
    report = prune(root, retention_days=7, now=NOW, live=lambda _: False)
    assert touched and directory.is_dir() and not report["pruned"]
    assert _flag(root, "task") == (
        original({**row, "artifact_finalized_at": FUTURE}), 2)


def test_imported_projection_and_quarantined_missing_invalid_outcomes(tmp_path):
    root = tmp_path / "data"
    for task_id in ("admitted", "missing", "legacy", "invalid"):
        _directory, drive, prune = _layout(root, "direct", task_id)
        if task_id == "admitted":
            write_result(root, document(task_id, ts=YOUNG, child_drive_root=str(drive)))
        elif task_id == "legacy":
            path = root / "task_results" / f"{task_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"task_id": task_id, "status": "completed"}), encoding="utf-8")
    store_import.ensure_current_store(root, with_children=False)
    task_store.patch_document(root, "invalid", lambda _: document("invalid", status="", ts=YOUNG))
    report = prune(root, retention_days=7, now=NOW, live=lambda _: False)
    assert not report["pruned"]
    assert {row["task_id"] for row in report["errors"]} == {"legacy", "invalid"}
    assert {row["task_id"]: row["reason"] for row in report["skipped"]} == {
        "admitted": "younger_than_retention", "missing": "task_not_terminal"}
    assert _flag(root, "admitted")[0] == retention.task_result_retention_at(document("admitted", ts=YOUNG))


@pytest.mark.parametrize("kind", ["headless", "direct"])
def test_unavailable_selector_falls_back_to_strict_real_pruner(tmp_path, monkeypatch, kind):
    root = tmp_path / "data"
    directory, drive, prune = _layout(root, kind, "task")
    write_task_result(root, "task", "completed", ts=OLD, child_drive_root=str(drive))

    def unavailable(*_args, **_kwargs):
        raise store.StoreBusy("selector unavailable")

    monkeypatch.setattr(task_store, "drive_retention_facts", unavailable)
    report = prune(root, retention_days=7, now=NOW, live=lambda _: False)
    assert report["pruned"] and not report["errors"] and not directory.exists()


@pytest.mark.parametrize("kind", ["headless", "direct"])
def test_budget_counts_settlements_only_cursor_wraps_and_stop_defers(tmp_path, kind):
    root = tmp_path / "data"
    for task_id, date in (("a", OLD), ("b", YOUNG), ("c", OLD)):
        _directory, drive, prune = _layout(root, kind, task_id)
        write_task_result(root, task_id, "completed", ts=date, child_drive_root=str(drive))
    first = prune(root, retention_days=7, now=NOW, live=lambda _: False, budget=1, after="b")
    assert [row["task_id"] for row in first["pruned"]] == ["c"]
    assert first["cursor"] == "c" and first["deferred"] == ["a"]
    assert first["skipped"] == [{"task_id": "b", "reason": "younger_than_retention"}]
    closed = prune(root, retention_days=7, now=NOW, live=lambda _: False, stop=lambda: True, after="c")
    assert not closed["pruned"] and closed["deferred"] == ["a"]
    assert closed["cursor"] == "c"
    last = prune(root, retention_days=7, now=NOW, live=lambda _: False, budget=1, after="c")
    assert [row["task_id"] for row in last["pruned"]] == ["a"]


def test_stopped_legacy_classification_leaves_null_without_body_reads(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _directory, drive, prune = _layout(root, "direct", "task")
    row = document("task", ts=OLD, child_drive_root=str(drive))
    _legacy_store(root, row, owned_root=tmp_path, shape="v3")
    store_import.upgrade_current_schema(root)
    reads = _body_spy(monkeypatch)
    report = prune(root, retention_days=7, now=NOW, live=lambda _: False, stop=lambda: True)
    assert report["deferred"] == ["task"] and not report["pruned"]
    assert reads == [] and _flag(root, "task") == (None, 1)


@pytest.mark.parametrize("damage", ["missing_body", "invalid_json", "future_schema"])
def test_unclassified_or_missing_bodies_reach_existing_strict_reader(tmp_path, monkeypatch, damage):
    root = tmp_path / "data"
    directory, drive, prune = _layout(root, "direct", "task")
    write_task_result(root, "task", "completed", ts=YOUNG, child_drive_root=str(drive))
    with store.open_store(root, write=True) as txn:
        if damage == "missing_body":
            txn.conn.execute("DELETE FROM task_bodies WHERE task_id='task'")
        else:
            raw = "{bad" if damage == "invalid_json" else json.dumps(
                {**document("task", ts=YOUNG), "_schema_version": 999})
            txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='task'", (raw,))
            txn.conn.execute("UPDATE tasks SET drive_retention_at=NULL WHERE task_id='task'")
    reads = _body_spy(monkeypatch)
    report = prune(root, retention_days=7, now=NOW, live=lambda _: False)
    assert directory.exists() and not report["pruned"]
    assert reads and (report["errors"] or report["skipped"])
    if damage == "future_schema":
        # Store admission, not this selector, owns schema policy. A raw edit of
        # an already-admitted row is not re-quarantined by the existing reader.
        assert _flag(root, "task")[0] is None
        assert report["skipped"] == [{"task_id": "task", "reason": "younger_than_retention"}]
    else:
        assert not any(row["reason"] == "younger_than_retention" for row in report["skipped"])


@pytest.mark.parametrize("count,size", [(64, 16_384), (512, 65_536)])
def test_real_selection_cost_is_batched_and_independent_of_skipped_body_bytes(tmp_path, monkeypatch, count, size):
    root = tmp_path / "data"
    for n in range(count):
        _directory, drive, prune = _layout(root, "direct", f"task-{n:04}")
        write_task_result(root, f"task-{n:04}", "completed", ts=YOUNG,
                          result="x" * size, child_drive_root=str(drive))
    real_open, transactions = task_store.open_store, []

    @contextlib.contextmanager
    def observed(*args, **kwargs):
        transactions.append(kwargs.get("write", False))
        with real_open(*args, **kwargs) as txn:
            yield txn

    reads = _body_spy(monkeypatch)
    monkeypatch.setattr(task_store, "open_store", observed)
    started, cpu = time.perf_counter(), time.process_time()
    report = prune(root, retention_days=7, now=NOW, live=lambda _: False)
    elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu
    assert len(report["skipped"]) == count and not report["errors"]
    assert not reads and not any(transactions)
    assert len(transactions) == 2 * ((count + 63) // 64)
    print(json.dumps({"tasks": count, "payload_bytes_each": size,
                      "body_reads": len(reads), "body_chars_read": sum(row[1] for row in reads),
                      "selector_transactions": len(transactions), "elapsed": elapsed, "cpu": cpu}))
