"""A room's read costs what its own tasks cost, not one store connection per task (issue #1771).

Under load every call that gives up the interpreter lock (a SQLite connection or statement, a
``stat``, an ``open``) waits milliseconds to get it back, so a page's latency is the number of
such calls. The room rule walks every row's task: a process that just started reads their lineage
in one batch per generation of ancestors, so a first read of a room with 30 tasks opens the store
as often as one with 3.
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import pathlib
import sqlite3
from types import SimpleNamespace

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


def test_hidden_endings_share_the_pages_visible_result_batch(tmp_path, monkeypatch):
    from ouroboros import task_store
    from ouroboros.terminal_projection import _project_row

    current = {"retried": {"status": "failed"}, "open": {"status": "running"},
               "finishing": {"status": "completed", "root_phase_checkpoint": {"post_task_synthesis": "running"}}}
    receipts = []
    for tid, fields in current.items():
        receipts.append(_project_row(tid, {"root_task_id": tid, "task_attempt": 0, "status": "completed"}, {}, {"chat_id": 1}))
        seed_result(tmp_path, tid, {"task_attempt": 1, "chat_id": 1, **fields})
    write(tmp_path / "logs" / "chat.jsonl", receipts)
    write(tmp_path / "logs" / "progress.jsonl", [row(1, "progress", task_id="visible")])
    seed_result(tmp_path, "elsewhere", {"status": "completed", "chat_id": 2})
    batches, fetch = [], task_store.display_documents

    def counted(root, ids, versions=None):
        batches.append(set(ids))
        return fetch(root, ids, versions)

    monkeypatch.setattr(task_store, "display_documents", counted)
    status, page = request(tmp_path)
    assert status == 200
    assert batches == [set(current) | {"visible"}]
    by_id = {r["task_id"]: r for r in page["messages"] if r.get("historical_terminal")}
    assert set(by_id) == set(current)
    assert by_id["retried"]["current_outcome"]["status"] == "failed"
    for tid, phase in (("open", "unfinished"), ("finishing", "finalizing")):
        assert by_id[tid]["task_phase"] == phase and by_id[tid]["outcome_final"] is False
        assert by_id[tid]["historical_terminal"]["status"] == "completed" and "current_outcome" not in by_id[tid]


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


def test_a_page_resolves_only_the_receipts_its_owner_messages_can_wear(tmp_path, monkeypatch):
    """A page asks the routing-receipt journal about the owner messages it can show (its selected owner rows; on
    the newest page, its Project's origins), never the install's other id-only receipts, and a room showing none
    reads no journal: neither a room of answers and agent notes carrying ids nor an empty delta of a Project with a
    retained origin. Each message resolves alone, so every page keeps the bytes the whole-journal read gave it:
    Main's newest page and an older one, a Project whose retained origin copy wears its receipt with its row gone,
    also when the page's read of its Project's refs failed or predates the binding (the copy then asks for itself).
    A page asks an id as its row looks it up (``str``): one another writer saved as a number is asked and worn."""
    from ouroboros import chat_annotations, dialogue_store, project_dialogue
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.dialogue_provenance import recorded_author
    from ouroboros.gateway import history
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.projects_registry import bind_task_to_project, create_project

    alpha, beta, gamma, delta = (create_project(tmp_path, slug, name=slug.title())["chat_id"]
                                 for slug in ("alpha", "beta", "gamma", "delta"))
    gone = {"direction": "in", "chat_id": 1, "client_message_id": "origin-gone", "ts": "2026-09-01T00:00:00Z",
            "text": "Start alpha"}
    ref, words = owner_row_source(gone)  # never stored: the binding's exact copy is all that is left of it
    bind_task_to_project(tmp_path, "alpha-root", "alpha", origin={"ref": ref, "text": words, "snapshot": {"projectable": True}})
    owners = {room: [row(index, chat_id=chat, client_message_id=f"{room}-{index}") for index in range(count)]
              for room, chat, count in (("main", 1, 6), ("alpha", alpha, 3), ("beta", beta, 40))}
    ineligible = [row(0, chat_id=gamma, direction="out", text="An answer alone", client_message_id="gamma-out"),
                  row(1, chat_id=gamma, client_message_id="gamma-note", system_type="agent_note_received",
                      agent_note_text="A program's words", declared_author=recorded_author("Scout"))]
    unnamed = row(2, chat_id=gamma, text="An owner's words saved without an id", client_message_id=None)
    numbered = row(3, chat_id=delta, text="Words another writer saved under a number", client_message_id=4242)
    write(tmp_path / "logs" / "chat.jsonl",
          [*(entry for rows in owners.values() for entry in rows), *ineligible, unnamed, numbered])
    for rows in (*owners.values(), ineligible):  # id-only receipts, as older installs wrote them: the store places each
        for entry in rows:
            chat_annotations.append_chat_annotation(tmp_path, entry["client_message_id"], action="steer_task",
                                                    status="delivered", routing_token="act")
    chat_annotations.append_chat_annotation(tmp_path, "4242", action="steer_task", status="delivered",  # addressed, as
                                            routing_token="act", source_chat_id=delta,  # this tree writes them
                                            source_row_address=format_address(row_address(numbered)))
    chat_annotations.append_chat_annotation(tmp_path, "origin-gone", action="ensure_project_scope", status="delivered",
                                            routing_token="act", source_chat_id=1, source_row_address=format_address(ref))
    status, first = request(tmp_path, n_human="2")  # imports the rows; its cursor reads the older page below them
    assert status == 200 and first["has_more"]

    asked, resolved, parsed = [], [], []
    index, sources, lines, whole = (chat_annotations._annotation_index, dialogue_store.annotation_sources,
                                    chat_annotations.iter_jsonl_objects, chat_annotations.addressed_chat_annotations)

    def asking(path, **kwargs):
        asked.append(kwargs.get("message_ids"))
        return index(path, **kwargs)

    def resolving(root, ids):
        resolved.append(set(ids))
        return sources(root, resolved[-1])

    def parsing(path, *args, **kwargs):
        parsed.append(path)
        return lines(path, *args, **kwargs)

    monkeypatch.setattr(chat_annotations, "_annotation_index", asking)
    monkeypatch.setattr(dialogue_store, "annotation_sources", resolving)
    monkeypatch.setattr(chat_annotations, "iter_jsonl_objects", parsing)

    def read(params, *, unscoped=False, refs=None):
        for seen in (asked, resolved, parsed):
            seen.clear()
        with monkeypatch.context() as patch:
            if unscoped:  # the read every page made before: the whole journal, every id-only receipt resolved
                patch.setattr(chat_annotations, "addressed_chat_annotations", lambda root, **_kwargs: whole(root))
            if refs is not None:  # the page's own read of its Project's origin refs
                patch.setattr(project_dialogue, "source_refs_for_project", refs)
            response = asyncio.run(history.make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params=params)))
        assert response.status_code == 200
        return response.body, list(asked), set().union(*resolved), len(parsed)

    main, alpha_ids, beta_ids = ({entry["client_message_id"] for entry in owners[room]} for room in ("main", "alpha", "beta"))
    pages = {"main": {}, "main older": {"cursor": first["next_cursor"]}, "alpha": {"chat_id": str(alpha)},
             "gamma": {"chat_id": str(gamma)}, "delta": {"chat_id": str(delta)}}
    for name, params in pages.items():
        body, page_asked, page_resolved, page_parsed = read(params)
        old_body, old_asked, old_resolved, old_parsed = read(params, unscoped=True)
        assert body == old_body, name  # the same receipts on the same rows, byte for byte
        assert (old_asked, old_parsed) == ([None], 1) and beta_ids <= old_resolved, name  # the whole journal, resolved
        shown = {message["client_message_id"]: message.get("chat_annotation", {}).get("status")
                 for message in json.loads(body)["messages"] if message["role"] == "user"}
        if name == "gamma":  # an answer and an agent's note carry ids and receipts, yet neither wears one; an owner
            assert "gamma-note" in shown and len(shown) == 2 and not any(shown.values()), shown  # row's null id is none
            assert (page_asked, page_parsed, page_resolved) == ([set()], 0, set()), "no owner row: no journal read"
            continue
        [ids] = page_asked  # one question per page, about its own owner messages only
        assert ids is not None and page_parsed == 1, f"{name}: the page read every receipt of the install"
        assert shown and set(shown.values()) == {"delivered"} and set(shown) <= ids, name
        assert not ids & beta_ids and page_resolved == ids - {"origin-gone", "4242"}, name  # addressed: none placed
        if name == "main older":
            newer = {message["client_message_id"] for message in first["messages"] if message["role"] == "user"}
            assert ids < main and newer and not ids & newer, "the newer page's rows are not asked again"
        else:
            assert ids == {"main": main, "alpha": alpha_ids | {"origin-gone"}, "delta": {"4242"}}[name], name
        assert name != "alpha" or "origin-gone" in shown, "the retained origin copy wears its receipt"

    # The Project's delta with nothing new renders no origin: it asks about nobody and opens no journal.
    after = json.loads(read({"chat_id": str(alpha)})[0])["after_cursor"]
    body, page_asked, page_resolved, page_parsed = read({"chat_id": str(alpha), "cursor": after})
    assert json.loads(body)["messages"] == [] and (page_asked, page_parsed, page_resolved) == ([set()], 0, set())

    def unreadable(*_args, **_kwargs):
        raise OSError("the Project's refs cannot be read this time")

    # The page's refs miss the binding (a failed read, or one taken before the binding landed); the origin its own
    # read renders still wears its receipt, asked for after the page's rows, as the whole-journal read gave it.
    for name, refs in (("refs unreadable", unreadable), ("refs before the binding", lambda *_args, **_kwargs: [])):
        body, page_asked, _resolved, page_parsed = read({"chat_id": str(alpha)}, refs=refs)
        assert body == read({"chat_id": str(alpha)}, unscoped=True, refs=refs)[0], name
        shown = {message["client_message_id"]: message.get("chat_annotation", {}).get("status")
                 for message in json.loads(body)["messages"] if message["role"] == "user"}
        assert shown.get("origin-gone") == "delivered", f"{name}: the retained copy lost its receipt"
        assert (page_asked, page_parsed) == ([alpha_ids, {"origin-gone"}], 2), name
