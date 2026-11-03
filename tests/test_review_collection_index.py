"""The task store's ``review_collection_pending`` column through schema upgrades, old
writers, first child writes and export/import.

The column is 1 exactly while a task holds an open review operation pointer of any surface
(``obligations.review_collection_pending``): the evidence packer keeps such a task's loose files
and the maintenance pass retires a dead controller's pointer. The retired task acceptance's
closed pointers, panels and late receipts select nothing; its open pointers are open until retired.
"""
from __future__ import annotations

import copy
import json
import re

import pytest

from ouroboros import obligations, store, store_import, task_store
from tests._store_journal_shared import store_state
from tests.test_store import document, publish

_SCHEMA_SHAPES = {"v1": (), "owner-v2": ("has_owner_questions",),
                  "review-v2": ("review_collection_pending",)}


def _operation(state="dispatched", **fields):
    return {"surface": "task_acceptance", "state": state, "retry_key": "wave", **fields}


@pytest.mark.parametrize("surface", ["task_acceptance", "multi_model_review", "plan_review"])
@pytest.mark.parametrize("state,pending", [
    ("preparing", True), ("retained", True), ("dispatched", True), ("unpublished", False), ("closed", False),
    ("collected", False), ("preparation_unknown", False), ("preparation_refused", False), ("unrecognized", False),
])
def test_selection_is_an_open_pointer_of_any_surface(surface, state, pending):
    row = document("task", review_operations={"owner": {**_operation(state), "surface": surface}})
    assert obligations.review_collection_pending(row) is pending
    row["review_operations"]["owner"]["control_only"] = True
    assert obligations.review_collection_pending(row) is False
    row["review_operations"]["owner"] = "unreadable"
    assert obligations.review_collection_pending(row) is True
    panel = {"surface": "task_acceptance", "authority": "host_root", "actors": [{"operation_state": "in_flight"}]}
    assert obligations.review_collection_pending(document("task", review_projection={"panels": [panel]})) is False


def _legacy_store(root, *rows, owned_root, tier=store.TIER_ENFORCED, shape="v1"):
    """Actual v1, either independent v2, or v3, with that writer's column list."""
    assert root.resolve().is_relative_to(owned_root.resolve()), "fixture store must stay inside its tmp_path"
    path = store.store_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    known = ("has_owner_questions", "review_collection_pending") if shape == "v3" else _SCHEMA_SHAPES[shape]
    schema = store._SCHEMA
    for column, index in (
        ("has_owner_questions", "CREATE INDEX tasks_has_owner_questions ON tasks(has_owner_questions) WHERE has_owner_questions=1;\n"),
        ("review_collection_pending", store.REVIEW_COLLECTION_INDEX),
        ("plan_collection_pending", store.PLAN_COLLECTION_INDEX),
    ):
        if column not in known:
            schema = re.sub(rf",\s*{column} INTEGER", "", schema).replace(index, "")
    schema = re.sub(r",\s*drive_retention_at TEXT", "", schema)
    schema = re.sub(r",\s*terminal_seq INTEGER", "", schema).replace(store.TERMINAL_SEQ_INDEX, "")  # schema 8's column
    conn = store.connect(path, tier, create=True)
    try:
        conn.executescript(schema)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert columns & {"has_owner_questions", "review_collection_pending"} == set(known)
        assert "plan_collection_pending" not in columns
        conn.execute(f"PRAGMA application_id={store._APPLICATION_ID[tier]}")
        conn.execute("BEGIN IMMEDIATE")
        for row in rows:
            _old_replace(conn, row, known=known)
        txn = store.Txn(conn)
        txn.set_meta("schema_version", 3 if shape == "v3" else (1 if shape == "v1" else 2))
        txn.set_meta("lock_tier", tier)
        txn.set_meta("phase", store.PUBLISHED)
        store.commit(conn)
    finally:
        conn.close()


def _old_replace(conn, row, *, known=()):
    columns = store.typed_columns(row)
    for column in ("has_owner_questions", "review_collection_pending", "plan_collection_pending", "drive_retention_at"):
        if column not in known:
            columns.pop(column)
    previous = conn.execute("SELECT write_seq FROM tasks WHERE task_id=?", (row["task_id"],)).fetchone()
    columns["write_seq"] = (previous[0] if previous else 0) + 1
    conn.execute("INSERT OR REPLACE INTO task_bodies(task_id,body) VALUES (?,?)",
                 (row["task_id"], store.encode_document(row)))
    names = ("task_id", *columns)
    conn.execute(f"INSERT OR REPLACE INTO tasks ({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                 (row["task_id"], *columns.values()))


def _flag(root, task_id):
    with store.open_store(root, write=False) as txn:
        return txn.row(task_id, "review_collection_pending", "write_seq")


def _combined_schema(txn):
    assert txn.meta("schema_version") == store.SCHEMA_VERSION
    columns = {row["name"]: row for row in txn.conn.execute("PRAGMA table_info(tasks)")}
    assert all(columns[name]["notnull"] == 0 for name in (
        "has_owner_questions", "review_collection_pending", "plan_collection_pending", "terminal_seq"))
    indexes = {row["name"] for row in txn.conn.execute("PRAGMA index_list(tasks)")}
    assert {"tasks_has_owner_questions", "tasks_review_collection", "tasks_plan_collection", "tasks_terminal_seq"} <= indexes


def _question_document(task_id):
    return document(task_id, result="The complete answer stays intact", owner_quiz={"q": {
        "quiz_id": "q", "state": "open", "question": "Keep this format?", "options": ["Yes", "No"]}})


@pytest.mark.parametrize("tier", [store.TIER_ENFORCED, store.TIER_NAME])
@pytest.mark.parametrize("shape", ["v1", "owner-v2"])
def test_upgrade_then_old_writer_resets_unknown_without_history_loss(tmp_path, tier, shape):
    closed = document("closed", review_operations={"op": _operation("collected")}, result="full old answer")
    _legacy_store(tmp_path, closed, owned_root=tmp_path, tier=tier, shape=shape)
    store_import.upgrade_current_schema(tmp_path)
    assert _flag(tmp_path, "closed") == (None, 1)
    assert list(task_store.review_collection_candidates(tmp_path)) == []
    assert _flag(tmp_path, "closed") == (0, 1)
    newer = {**closed, "review_operations": {"op": _operation("dispatched")}}  # an open pointer, old writer
    with store.open_store(tmp_path, write=True) as txn:
        _old_replace(txn.conn, newer, known=_SCHEMA_SHAPES[shape])
    assert _flag(tmp_path, "closed") == (None, 2)
    store_import.upgrade_current_schema(tmp_path)
    assert list(task_store.review_collection_candidates(tmp_path)) == ["closed"]
    assert _flag(tmp_path, "closed") == (1, 2)
    assert task_store.read_document(tmp_path, "closed") == newer


@pytest.mark.parametrize("shape", _SCHEMA_SHAPES)
def test_passive_read_leaves_old_schema_and_first_child_write_upgrades_before_mutator(tmp_path, monkeypatch, shape):
    child = tmp_path / "child"
    original = _question_document("child")
    _legacy_store(child, original, owned_root=tmp_path, shape=shape)
    before = store_state(store.store_path(child))  # rows and any write's trace; a first WAL open's header aside
    assert task_store.read_document(child, "child") == original
    assert store_state(store.store_path(child)) == before
    monkeypatch.setattr(store_import, "_reconcile", lambda *_: pytest.fail("first write must not replay/import history"))
    calls = []
    def mutate(row):
        calls.append(row["result"])
        with store.open_store(child, write=False) as txn:
            _combined_schema(txn)
            assert txn.row("child", "has_owner_questions", "write_seq") == (1, 1)
        return {**row, "review_operations": {"op": _operation()}}
    task_store.patch_document(child, "child", mutate)
    # The current writer derives the column: an open pointer (an old acceptance one) is open until retired.
    assert calls == [original["result"]] and _flag(child, "child") == (1, 2)
    assert task_store.read_document(child, "child")["owner_quiz"] == original["owner_quiz"]
    assert set(obligations.members(child, "owner_questions")) == {"child"}


@pytest.mark.parametrize("shape", _SCHEMA_SHAPES)
def test_interrupted_schema_upgrade_rolls_back_and_retries(tmp_path, monkeypatch, shape):
    _legacy_store(tmp_path, _question_document("task"), owned_root=tmp_path, shape=shape)
    with store.open_store(tmp_path, write=False) as txn:
        columns = {r["name"] for r in txn.conn.execute("PRAGMA table_info(tasks)")}
        indexes = {r["name"] for r in txn.conn.execute("PRAGMA index_list(tasks)")}
        raw = txn.body_text("task")
    real = store_import._upgrade_review_collection
    def fail(txn):
        real(txn)  # the preceding owner-flag derivation and both DDLs must roll back together
        _combined_schema(txn)
        assert txn.row("task", "has_owner_questions") == 1
        raise OSError("interrupted before the combined transaction commits")
    with monkeypatch.context() as patch:
        patch.setattr(store_import, "_upgrade_review_collection", fail)
        with pytest.raises(OSError):
            store_import.upgrade_current_schema(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == (1 if shape == "v1" else 2)
        assert {r["name"] for r in txn.conn.execute("PRAGMA table_info(tasks)")} == columns
        assert {r["name"] for r in txn.conn.execute("PRAGMA index_list(tasks)")} == indexes
        assert txn.body_text("task") == raw and txn.row("task", "write_seq") == 1
    store_import.upgrade_current_schema(tmp_path)
    store_import.upgrade_current_schema(tmp_path)
    assert _flag(tmp_path, "task") == (0 if shape == "review-v2" else None, 1)
    with store.open_store(tmp_path, write=False) as txn:
        _combined_schema(txn)
        assert txn.row("task", "has_owner_questions") == 1 and txn.body_text("task") == raw


@pytest.mark.parametrize("resume", ["writer", "boot"])  # the collector's own resume left with task acceptance
def test_partial_export_upgrade_does_not_publish_a_complete_schema_version(tmp_path, resume):
    original = {**_question_document("task"), "review_operations": {"op": _operation()}}
    _legacy_store(tmp_path, original, owned_root=tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("phase", store.EXPORTING)
    assert store_import.ensure_current_store(tmp_path, with_children=False)["status"] == "switch_pending"
    with store.open_store(tmp_path, write=False, service=True) as txn:
        columns = {row["name"] for row in txn.conn.execute("PRAGMA table_info(tasks)")}
        assert "has_owner_questions" in columns and "review_collection_pending" not in columns
        assert txn.meta("schema_version") == 1
        assert txn.load("task") == original and txn.row("task", "write_seq") == 1
    with pytest.raises(store.StoreRetired):
        task_store.patch_document(tmp_path, "task", lambda _: pytest.fail("export is not writable"))
    with store.open_store(tmp_path, write=True, service=True) as txn:
        txn.set_meta("phase", store.PUBLISHED)
    calls = []
    if resume == "writer":
        def mutate(row):
            with store.open_store(tmp_path, write=False) as txn:
                _combined_schema(txn)
            calls.append(row)
            return row
        assert task_store.patch_document(tmp_path, "task", mutate) == original
        assert calls == [original]
    elif resume == "boot":
        store_import.ensure_current_store(tmp_path, with_children=False)
    with store.open_store(tmp_path, write=False) as txn:
        _combined_schema(txn)
        assert txn.load("task") == original
        assert txn.row("task", "has_owner_questions", "write_seq") == (1, 2 if resume == "writer" else 1)


@pytest.mark.parametrize("shape", _SCHEMA_SHAPES)
def test_first_touch_schema_owner_honors_zero_wait_and_never_calls_mutator(tmp_path, monkeypatch, shape):
    _legacy_store(tmp_path, document("task"), owned_root=tmp_path, shape=shape)
    with store.open_store(tmp_path, write=True):
        with store.acquisition_budget(0), pytest.raises(store.StoreBusy):
            store_import.upgrade_current_schema(tmp_path)
        with store.acquisition_budget(0), pytest.raises(store.StoreBusy):
            task_store.patch_document(tmp_path, "task", lambda _: pytest.fail("not acquired"))
    assert task_store.patch_document(tmp_path, "task", lambda row: row)["task_id"] == "task"


@pytest.mark.parametrize("shape", _SCHEMA_SHAPES)
@pytest.mark.parametrize("boot", [False, True], ids=["first-touch-upgrade", "boot"])
def test_combined_upgrade_keeps_question_duty_and_lazy_review_selection(tmp_path, shape, boot):
    from ouroboros import evidence_packs

    asked, quiet = _question_document("asked"), document("quiet", result="closed history")
    _legacy_store(tmp_path, asked, quiet, owned_root=tmp_path, shape=shape)
    with store.open_store(tmp_path, write=False) as txn:
        bodies = {tid: txn.body_text(tid) for tid in ("asked", "quiet")}
    if boot:
        store_import.ensure_current_store(tmp_path, with_children=False)
    else:
        store_import.upgrade_current_schema(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        _combined_schema(txn)
        assert txn.row("asked", "has_owner_questions") == 1
        assert txn.row("quiet", "has_owner_questions") == 0
        for tid, raw in bodies.items():
            assert txn.body_text(tid) == raw and txn.row(tid, "write_seq") == 1
            assert txn.row(tid, "review_collection_pending") == (0 if shape == "review-v2" else None)
    assert set(obligations.members(tmp_path, "owner_questions")) == {"asked"}
    assert "review_collection_pending" not in store.FLAG_COLUMNS.values()
    # The upgraded rows pack (NULL or 0: no body with the selector recorded an open review); the
    # owner-question duty grants no authority to block packing. A selector of 1 does, and only because
    # it names a review that may still read the task's loose handles (evidence pack idiom).
    with store.open_store(tmp_path, write=False) as txn:
        assert evidence_packs.task_packable(txn, "quiet") and evidence_packs.task_packable(txn, "asked")
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET review_collection_pending=1 WHERE task_id='quiet'")
    with store.open_store(tmp_path, write=False) as txn:
        assert not evidence_packs.task_packable(txn, "quiet") and evidence_packs.task_packable(txn, "asked")
    task_store.patch_document(tmp_path, "asked", lambda row: {**row, "owner_quiz": {"q": {"state": "closed"}}})
    assert obligations.members(tmp_path, "owner_questions") == {}


def test_storeless_export_import_and_changed_legacy_files_keep_documents_and_owe_no_collection(tmp_path):
    root = tmp_path / "original"
    row = document("task", review_operations={"op": _operation("unpublished")}, result="exact evidence")
    publish(root, row)
    store_import.export_facts(root)
    exported = root / "task_results" / "task.json"
    restored = tmp_path / "restored"
    (restored / "task_results").mkdir(parents=True)
    (restored / "task_results" / "task.json").write_bytes(exported.read_bytes())
    store_import.ensure_current_store(restored, with_children=False)
    # An imported old acceptance pointer is no collection debt (the column derives 0).
    assert task_store.read_document(restored, "task") == row and _flag(restored, "task")[0] == 0
    changed = copy.deepcopy(row)
    changed["review_operations"]["op"]["state"] = "collected"
    (restored / "task_results" / "task.json").write_text(json.dumps(changed), encoding="utf-8")
    store_import.ensure_current_store(restored, with_children=False)
    assert task_store.read_document(restored, "task") == row  # boot leaves frozen files to maintenance
    store_import.run_legacy_reconcile(restored)
    assert task_store.read_document(restored, "task") == changed and _flag(restored, "task")[0] == 0
