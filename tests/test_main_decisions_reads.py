"""Main reads its questions in one display batch and shares lineage generations.

The batch must preserve the full reader's forms, strict admission and per-member
gaps, including when a damaged document makes SQLite's JSON projection fail.
"""
from __future__ import annotations

import json

import pytest

from ouroboros import project_admission, projects_registry, store, task_results, task_store
from ouroboros.gateway.main_decisions import main_decisions_snapshot
from ouroboros.owner_quiz import record_asked, record_publication, reconcile_terminal
from tests._queue_state import gateway, read_state  # noqa: F401
from tests._store_shared import seed_result
from tests.test_task_reads import BULK, METADATA_BULK


def _count_registry_parses(patching):
    """Every physical read-and-parse of the Project registry, counted where each reader opens it
    (``project_admission._registry_snapshot``): what the Main decisions branch of /api/state pays per Project
    lookup. The endpoint's other readers are a different cost and are not counted here."""
    parses, real = [], project_admission._registry_snapshot

    def counted(*args, **kwargs):
        parses.append(kwargs.get("allow_missing"))
        return real(*args, **kwargs)

    patching.setattr(project_admission, "_registry_snapshot", counted)
    return parses


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
        parses = _count_registry_parses(reading)
        snapshot = main_decisions_snapshot(tmp_path, [], {})
    assert len(parses) == 1, "one registry parse names every question's Project, never one per bound task"
    assert batches == [set(tasks)] and opened == [tmp_path]
    assert [kwargs["strict"] for kwargs in resolvers] == [True]
    assert generations == [set(tasks), {*parents, "root"}]
    assert snapshot["current_questions_complete"] is True and snapshot["current_questions_gaps"] == []
    assert len(snapshot["current_questions"]) == count
    assert {row["project_chat_id"] for row in snapshot["current_questions"]} == {project["chat_id"]}

    def full_documents(root, ids, versions=None):
        return {tid: json.dumps(task_results.load_task_result(root, tid, strict=True)) for tid in ids}

    monkeypatch.setattr(task_store, "display_documents", full_documents)
    assert main_decisions_snapshot(tmp_path, [], {}) == snapshot


@pytest.mark.parametrize("damage", ["json", "quiz", "missing"])
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


def test_attention_rows_share_one_strict_snapshot_and_batched_lineage_generations(tmp_path, monkeypatch):
    """The attention loop's room reads ride ONE strict resolver construction and warmed lineage
    generations for a whole census page of offered rows — never a per-row read (N+1). Each row lands
    at the nearest durable binding its ancestry confirms."""
    from ouroboros.task_results import write_task_result

    project = projects_registry.create_project(tmp_path, "converted", name="Converted room")
    count = 30
    helpers = [f"helper-{index}" for index in range(count)]
    write_task_result(tmp_path, "root", "running", chat_id=1, root_task_id="root", delegation_role="root")
    write_task_result(tmp_path, "middle", "running", chat_id=1, root_task_id="root",
                      parent_task_id="root", delegation_role="subagent")
    projects_registry.bind_task_to_project(tmp_path, "middle", project["id"], origin={"absent": "system"})
    rows = []
    for task_id in helpers:
        write_task_result(tmp_path, task_id, "running", chat_id=1, root_task_id="root",
                          parent_task_id="middle", delegation_role="subagent")
        rows.append({"activity_id": task_id, "chat_id": 1, "phase": "budget_paused", "pause_cause": "budget",
                     "can_resume": True, "cancelable": False, "parent_task_id": "middle", "root_task_id": "root"})
    monkeypatch.setattr(projects_registry, "_TASK_LINEAGE", {})
    generations, resolvers = [], []
    real_lineage, real_resolver = projects_registry._read_lineage, projects_registry.RoomResolver

    def lineage(root, ids, **kwargs):
        generations.append(set(ids))
        return real_lineage(root, ids, **kwargs)

    def resolver(*args, **kwargs):
        resolvers.append(kwargs)
        return real_resolver(*args, **kwargs)

    with monkeypatch.context() as reading:
        reading.setattr(projects_registry, "_read_lineage", lineage)
        reading.setattr(projects_registry, "RoomResolver", resolver)
        parses = _count_registry_parses(reading)
        snapshot = main_decisions_snapshot(tmp_path, rows, {})
    assert len(parses) == 1, "30 actions bound through one ancestor share the snapshot's one registry parse"
    assert [kwargs["strict"] for kwargs in resolvers] == [True], "one strict construction for the whole snapshot"
    assert generations == [set(helpers), {"middle", "root"}], "warmed by generation, never per row"
    assert snapshot["main_attention_complete"] is True and snapshot["main_attention_gaps"] == []
    assert len(snapshot["main_attention"]) == count
    assert {row["project_id"] for row in snapshot["main_attention"]} == {project["id"]}
    assert {row["chat_id"] for row in snapshot["main_attention"]} == {project["chat_id"]}


def test_the_main_decisions_branch_of_one_state_get_parses_the_registry_once(gateway, monkeypatch):  # noqa: F811
    """One isolated ``GET /api/state`` over the real gateway with 30 held helpers of one converted ancestor, each
    owing the owner a Resume. Only the Main decisions branch is counted, from its entry to its return: the
    endpoint's neighbouring readers are another cost and stay as they are."""
    from ouroboros.gateway import main_decisions
    from ouroboros.task_results import write_task_result

    root = gateway.root
    project = projects_registry.create_project(root, "converted", name="Converted room")
    write_task_result(root, "root", "running", chat_id=1, root_task_id="root", delegation_role="root")
    write_task_result(root, "middle", "running", chat_id=1, root_task_id="root", parent_task_id="root",
                      delegation_role="subagent")
    projects_registry.bind_task_to_project(root, "middle", project["id"], origin={"absent": "system"})
    for index in range(30):
        task_id = f"helper-{index}"
        write_task_result(root, task_id, "running", chat_id=1, root_task_id="root", parent_task_id="middle",
                          delegation_role="subagent")
        gateway.pending.append({"id": task_id, "chat_id": 1, "budget_drive_root": str(root), "parent_task_id": "middle",
                                "root_task_id": "root", "delegation_role": "subagent",
                                "_budget_pause_hold": {"reason": "owner_restart_hold", "selected": False,
                                                       "dispatchable": False}})
    gateway.running["root"] = {"task": {"id": "root", "root_task_id": "root", "chat_id": 1,
                                        "budget_drive_root": str(root), "_attempt": 3}, "attempt": 3, "started_at": 1.0}
    branches, real = [], main_decisions.main_decisions_snapshot

    def measured(*args, **kwargs):
        with monkeypatch.context() as counting:
            branches.append(_count_registry_parses(counting))
            return real(*args, **kwargs)

    monkeypatch.setattr(main_decisions, "main_decisions_snapshot", measured)
    payload = read_state(gateway)
    assert payload["main_attention_complete"] is True and payload["main_attention_gaps"] == []
    assert len(payload["main_attention"]) == 30
    assert {row["project_id"] for row in payload["main_attention"]} == {project["id"]}
    assert [len(parses) for parses in branches] == [1], "the branch's registry parses, per call"
