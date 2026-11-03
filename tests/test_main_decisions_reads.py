"""Main reads its questions in one display batch and shares lineage generations.

The batch must preserve the full reader's forms, strict admission and per-member
gaps, including when a damaged document makes SQLite's JSON projection fail.
"""
from __future__ import annotations

import json

import pytest

from ouroboros import projects_registry, store, task_results, task_store
from ouroboros.gateway.main_decisions import main_decisions_snapshot
from ouroboros.owner_quiz import record_asked, record_publication, reconcile_terminal
from tests._store_shared import seed_result
from tests.test_task_reads import BULK, METADATA_BULK


def _ask(root, task_id, **fields):
    seed_result(root, task_id, {"task_id": task_id, "status": "completed", "chat_id": 1,
                               **BULK, "metadata": METADATA_BULK, **fields})
    record_asked(root, task_id, quiz_id="format", question="Which format?", options=["PDF", "HTML"],
                 option_details=["Printable", "Searchable"], recommended_index=1, chat_id=1)
    record_publication(root, task_id, "format", 1)
    reconcile_terminal(root, task_id)


@pytest.mark.parametrize("count", [1, 30])
def test_question_batch_and_lineage_generations_preserve_full_forms(tmp_path, monkeypatch, count):
    project = projects_registry.create_project(tmp_path, "converted", name="Converted room")
    seed_result(tmp_path, "root", {"status": "completed", "chat_id": 1})
    projects_registry.bind_task_to_project(tmp_path, "root", project["id"], origin={"absent": "system"})
    tasks = [f"author-{index}" for index in range(count)]
    parents = [f"parent-{index}" for index in range(count)]
    for task_id, parent in zip(tasks, parents):
        seed_result(tmp_path, parent, {"status": "completed", "chat_id": 1, "parent_task_id": "root"})
        _ask(tmp_path, task_id, parent_task_id=parent, root_task_id="root",
             owner_wait={"quiz_id": "format", "state": "expired_terminal"})
    monkeypatch.setattr(projects_registry, "_TASK_LINEAGE", {})
    batches, opened, generations, resolvers = [], [], [], []
    real_views, real_open = task_store.display_documents, task_store.open_store
    real_lineage, real_resolver = projects_registry._read_lineage, projects_registry.RoomResolver

    def views(root, ids, versions=None):
        batches.append(set(ids))
        return real_views(root, ids, versions)

    def opened_store(*args, **kwargs):
        opened.append(args[0])
        return real_open(*args, **kwargs)

    def lineage(root, ids, **kwargs):
        generations.append(set(ids))
        return real_lineage(root, ids, **kwargs)

    def resolver(*args, **kwargs):
        resolvers.append(kwargs)
        return real_resolver(*args, **kwargs)

    def full_read_forbidden(*args, **kwargs):
        pytest.fail("Main read a full task body for a question")

    with monkeypatch.context() as reading:
        reading.setattr(task_store, "display_documents", views)
        reading.setattr(task_store, "open_store", opened_store)
        reading.setattr(task_results, "load_task_result", full_read_forbidden)
        reading.setattr(projects_registry, "_read_lineage", lineage)
        reading.setattr(projects_registry, "RoomResolver", resolver)
        snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert batches == [set(tasks)] and opened == [tmp_path]
    assert resolvers == [{"strict": True}]
    assert generations == [set(tasks), {*parents, "root"}]
    assert snapshot["current_questions_complete"] is True and snapshot["current_questions_gaps"] == []
    assert len(snapshot["current_questions"]) == count
    assert {row["project_chat_id"] for row in snapshot["current_questions"]} == {project["chat_id"]}

    def full_documents(root, ids, versions=None):
        return {tid: json.dumps(task_results.load_task_result(root, tid, strict=True)) for tid in ids}

    monkeypatch.setattr(task_store, "display_documents", full_documents)
    assert main_decisions_snapshot(tmp_path, [], {}) == snapshot


@pytest.mark.parametrize("damage", ["json", "utf8", "identity", "status", "quiz", "missing"])
def test_bad_member_keeps_its_gap_and_healthy_questions(tmp_path, monkeypatch, damage):
    for task_id in ("bad", "good"):
        _ask(tmp_path, task_id)
    before = main_decisions_snapshot(tmp_path, [], {})
    with store.open_store(tmp_path, write=True) as txn:
        original = txn.body_text("bad")
        if damage == "missing":
            txn.conn.execute("DELETE FROM task_bodies WHERE task_id='bad'")
        elif damage == "utf8":
            txn.conn.execute("UPDATE task_bodies SET body=CAST(X'7B22FF' AS TEXT) WHERE task_id='bad'")
        else:
            document = json.loads(original)
            if damage != "json":
                document.update({"identity": {"task_id": "other"}, "status": {"status": None},
                                 "quiz": {"owner_quiz": "unreadable"}}[damage])
            txn.conn.execute("UPDATE task_bodies SET body=? WHERE task_id='bad'",
                             ("unreadable" if damage == "json" else json.dumps(document),))
    # Cold lineage must also recover the healthy member after a failed batch.
    monkeypatch.setattr(projects_registry, "_TASK_LINEAGE", {})
    with store.open_store(tmp_path, write=False) as txn:
        revision = txn.revision()
    snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert snapshot["current_questions_complete"] is False
    assert snapshot["current_questions_gaps"] == [{"reason": "question_source_unavailable", "task_id": "bad"}]
    assert snapshot["current_questions"] == [row for row in before["current_questions"] if row["task_id"] == "good"]
    assert snapshot["main_attention_complete"] is True and snapshot["main_attention_gaps"] == []
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.revision() == revision, "a passive read never repairs its sources"
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("INSERT OR REPLACE INTO task_bodies (task_id,body) VALUES ('bad',?)", (original,))
    assert main_decisions_snapshot(tmp_path, [], {}) == before


def test_failed_batch_preserves_strict_per_member_recovery(tmp_path, monkeypatch):
    _ask(tmp_path, "author")
    expected = main_decisions_snapshot(tmp_path, [], {})

    def busy(*args, **kwargs):
        raise store.StoreBusy("batch unavailable")

    monkeypatch.setattr(task_store, "display_documents", busy)
    assert main_decisions_snapshot(tmp_path, [], {}) == expected
