"""A room's read costs what its own tasks cost, not one store connection per task (issue #1771).

Under load every call that gives up the interpreter lock (a SQLite connection or statement, a
``stat``, an ``open``) waits milliseconds to get it back, so a page's latency is the number of
such calls. The room rule walks every row's task: a process that just started reads their lineage
in one batch per generation of ancestors, so a first read of a room with 30 tasks opens the store
as often as one with 3.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sqlite3

from tests._store_shared import seed_result
from tests.test_chat_history_paging import isolated_runtime, request, row, write  # noqa: F401


def _counted(monkeypatch):
    counts = collections.Counter()
    connect, stat, path_stat, path_open = sqlite3.connect, os.stat, pathlib.Path.stat, pathlib.Path.open

    def counted_connect(*args, **kwargs):
        counts["connect"] += 1
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(lambda _statement: counts.update(["statement"]))
        return connection

    def tally(name, real):
        def call(*args, **kwargs):
            counts[name] += 1
            return real(*args, **kwargs)
        return call

    monkeypatch.setattr(sqlite3, "connect", counted_connect)
    monkeypatch.setattr(os, "stat", tally("stat", stat))
    monkeypatch.setattr(pathlib.Path, "stat", tally("stat", path_stat))
    monkeypatch.setattr(pathlib.Path, "open", tally("open", path_open))
    return counts


def _answered_room(root, tasks, ancestors=0):
    """``tasks`` answered rows of the room; each task has a chain of ``ancestors`` parent tasks the rows do
    not name, which the room rule finds only through the store's lineage."""
    from ouroboros.projects_registry import create_project

    chat = create_project(root, "lineage", name="Lineage")["chat_id"]
    write(root / "archive" / "chat_20260901T000000.jsonl", [row(index, chat_id=chat, direction="out",
                                                                text=f"older {index}", task_id=f"task-{index}")
                                                            for index in range(tasks)])
    write(root / "logs" / "chat.jsonl", [row(index, chat_id=chat, direction="out", text=f"answer {index}",
                                             task_id=f"task-{index}") for index in range(tasks)])
    for index in range(tasks):
        chain = [f"task-{index}", *(f"up{level}-{index}" for level in range(1, ancestors + 1))]
        for task_id, parent in zip(chain, [*chain[1:], ""]):
            seed_result(root, task_id, {"task_id": task_id, "status": "completed", "chat_id": chat,
                                        "parent_task_id": parent, "root_task_id": chain[-1] if parent else ""})
    return chat


def test_a_first_read_after_start_reads_its_tasks_lineage_in_batches(tmp_path, monkeypatch):
    """The room rule walks each row's task: a process that just started reads their lineage once per
    generation of ancestors, not one store connection per task (cold and warm reads are counted apart)."""
    from ouroboros import projects_registry

    cold, warm = {}, {}
    for tasks in (3, 30):
        root = tmp_path / str(tasks)
        chat = _answered_room(root, tasks, ancestors=2)  # each row's task has a parent and a grandparent
        monkeypatch.setattr(projects_registry, "_TASK_LINEAGE", {})  # nothing read yet in this process
        monkeypatch.setattr(projects_registry, "_ROOM_ANSWERS", {})
        with monkeypatch.context() as patch:
            cold[tasks] = _counted(patch)
            status, page = request(root, chat_id=str(chat))
        assert status == 200 and len([row for row in page["messages"] if row.get("history_id")]) == 2 * tasks
        with monkeypatch.context() as patch:
            warm[tasks] = _counted(patch)
            assert request(root, chat_id=str(chat))[0] == 200
    assert cold[30]["connect"] <= cold[3]["connect"] + 1, cold
    assert warm[30]["connect"] <= warm[3]["connect"] and warm[30]["statement"] <= warm[3]["statement"], warm


def test_a_delta_reads_its_new_tasks_lineage_in_batches(tmp_path, monkeypatch):
    """New tasks a delta brings are as cold as a first read's: their rows replay through the same batches."""
    from ouroboros import projects_registry

    costs = {}
    for tasks in (3, 30):
        root = tmp_path / str(tasks)
        chat = _answered_room(root, 1)
        status, page = request(root, chat_id=str(chat))
        assert status == 200 and page["after_cursor"]
        with (root / "logs" / "chat.jsonl").open("a", encoding="utf-8") as handle:
            for index in range(tasks):
                task_id = f"new-{index}"
                seed_result(root, task_id, {"task_id": task_id, "status": "running", "chat_id": chat})
                handle.write(json.dumps(row(index, chat_id=chat, direction="out", text=f"news {index}",
                                            task_id=task_id)) + "\n")
        monkeypatch.setattr(projects_registry, "_TASK_LINEAGE", {})
        monkeypatch.setattr(projects_registry, "_ROOM_ANSWERS", {})
        with monkeypatch.context() as patch:
            costs[tasks] = _counted(patch)
            status, delta = request(root, chat_id=str(chat), cursor=page["after_cursor"])
        assert status == 200 and len([r for r in delta["messages"] if r.get("history_id")]) == tasks
    assert costs[30]["connect"] <= costs[3]["connect"] + 1, costs


def test_remembered_bound_children_read_their_parents_lineage_in_one_batch(tmp_path, monkeypatch):
    """A legacy child row names no parent: the anchor pass takes it from the task's lineage and then asks the
    parent's room. The room rule stops at the child's binding, so the warm before the rows reads no parent;
    the pass reads its parents together, and 30 such children open the store as often as 3."""
    from ouroboros import projects_registry
    from ouroboros.projects_registry import bind_task_to_project, create_project

    costs = {}
    for tasks in (3, 30):
        root = tmp_path / str(tasks)
        project = create_project(root, "bound", name="Bound")
        children = [f"child-{index}" for index in range(tasks)]
        write(root / "logs" / "chat.jsonl", [row(index, chat_id=project["chat_id"], direction="out",
                                                 text=f"child answer {index}", task_id=child)
                                             for index, child in enumerate(children)])
        for index, child in enumerate(children):
            seed_result(root, f"par-{index}", {"task_id": f"par-{index}", "status": "completed", "chat_id": 1})
            seed_result(root, child, {"task_id": child, "status": "completed", "chat_id": project["chat_id"],
                                      "parent_task_id": f"par-{index}", "root_task_id": f"par-{index}"})
            bind_task_to_project(root, child, project["id"], origin={"absent": "system"})
        monkeypatch.setattr(projects_registry, "_TASK_LINEAGE", {})
        monkeypatch.setattr(projects_registry, "_ROOM_ANSWERS", {})
        assert set(projects_registry._read_lineage(root, children)) == set(children)  # remembered; parents are not
        reads, real = [], projects_registry._read_lineage
        with monkeypatch.context() as patch:
            patch.setattr(projects_registry, "_read_lineage", lambda *args, **kwargs: reads.append(1) or real(*args, **kwargs))
            status, page = request(root, chat_id=str(project["chat_id"]))
        assert status == 200 and len([r for r in page["messages"] if r.get("history_id")]) == tasks
        costs[tasks] = len(reads)
    assert costs[30] == costs[3], costs  # one read of the parents together, not one per child
