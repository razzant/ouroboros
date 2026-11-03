"""History, the logs tail and the task-event transports are addressed reads of the dialogue store.

A room's page, Main's page, a task's log tail and its event replay do the same SQL work whatever
other rooms wrote (statements and VM steps, ``tests.test_dialogue_addressed_readers.metered``);
the archive import's frontier is disclosed and never read past (never "no older rows"); a Project
room finds the rows of its bound tasks' descendants by the store's lineage.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_import as di, dialogue_store as ds
from ouroboros.gateway import history_paging, logs
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.task_events import _TaskEventCursorFollower, iter_task_events
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import write_task_result
from tests._dialogue import seed
from tests.test_chat_history_paging import isolated_runtime  # noqa: F401  (autouse: state/queue roots, no lifecycle row)
from tests.test_dialogue_addressed_readers import metered


def ts(minute: int) -> str:
    return f"2026-10-01T00:{minute:02d}:00Z"


def history(root, **params):
    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params=params)))
    return response.status_code, json.loads(response.body)


def logs_tail(root, **params):
    request = SimpleNamespace(path_params={"name": "progress"}, query_params=params,
                              app=SimpleNamespace(state=SimpleNamespace(drive_root=root)))
    response = logs._logs_tail(request)
    return response.status_code, json.loads(response.body)


def events_v2(root, task_id, cursor=None):
    follower = _TaskEventCursorFollower(root, task_id, cursor or {"v": 2, "seq": 0, "view": "", "positions": {}})
    follower.refresh_view()
    return list(follower.read_events())


def own_room(root):
    """A Project with a bound task, five answers and five narration rows, beside five Main rows."""
    project = create_project(root, "own", name="Own")
    chat = project["chat_id"]
    write_task_result(root, "own-task", "running", chat_id=chat)
    bind_task_to_project(root, "own-task", project["id"], origin={"absent": "system"})
    for index in range(5):
        ds.append(root, "chat", {"ts": ts(index), "chat_id": chat, "direction": "out", "task_id": "own-task",
                                 "text": f"own answer {index}"})
        ds.append(root, "progress", {"ts": ts(index), "chat_id": chat, "task_id": "own-task", "content": f"own step {index}"})
        ds.append(root, "chat", {"ts": ts(index), "chat_id": 1, "direction": "in", "text": f"main {index}",
                                 "client_message_id": f"main-{index}"})
    return chat


def foreign_rooms(root, rooms: int, rows: int):
    """``rooms`` other Projects with ``rows`` answers and narration rows each: other rooms' history."""
    for room in range(rooms):
        chat = create_project(root, f"foreign-{room}", name=f"Foreign {room}")["chat_id"]
        for index in range(rows):
            ds.append(root, "chat", {"ts": ts(index), "chat_id": chat, "direction": "out",
                                     "task_id": f"foreign-task-{room}", "text": f"foreign {room} {index}"})
            ds.append(root, "progress", {"ts": ts(index), "chat_id": chat, "task_id": f"foreign-task-{room}",
                                         "content": f"foreign step {room} {index}"})


READERS = {
    "room": lambda root, chat: history(root, chat_id=str(chat)),
    "main": lambda root, chat: history(root),
    "logs": lambda root, chat: logs_tail(root, limit="20", task_id="own-task"),
    "events_v1": lambda root, chat: iter_task_events(root, "own-task"),
    "events_v2": lambda root, chat: events_v2(root, "own-task"),
}


def _check(reader, result):
    if reader in ("room", "main"):
        status, page = result
        assert status == 200 and page["window"]["complete"] is True
        texts = {row["text"] for row in page["messages"]}
        own = {f"own answer {i}" for i in range(5)} | {f"own step {i}" for i in range(5)}
        assert texts == (own if reader == "room" else {f"main {i}" for i in range(5)})
    elif reader == "logs":
        status, payload = result
        assert status == 200 and [row["content"] for row in payload["entries"]] == [f"own step {i}" for i in range(5)]
    else:
        rows = [row["data"].get("content") or row["data"].get("text") for row in result if row["type"] in ("progress", "message")]
        assert sorted(rows) == sorted([f"own step {i}" for i in range(5)] + [f"own answer {i}" for i in range(5)])


@pytest.mark.parametrize("reader", sorted(READERS))
def test_a_reader_does_equal_sql_work_whatever_other_rooms_wrote(tmp_path, monkeypatch, reader):
    """Two foreign Projects with one or thirty rows each: the room's page, Main's page (every chat no
    Project owns), the task's log tail and both event transports run the same statements and the same
    VM steps. Main's lens enumerates the store's rooms (one row per chat, never history), so the
    number of rooms is fixed here; the Project room's cost is also equal with one or thirty rooms."""
    counts = []
    for rows in (1, 30):
        root = tmp_path / str(rows)
        chat = own_room(root)
        foreign_rooms(root, 2, rows)
        with metered(monkeypatch) as measured:
            result = READERS[reader](root, chat)
        _check(reader, result)
        counts.append(dict(measured))
    assert counts[0] == counts[1] and counts[0]["steps"] > 0, counts


def test_a_project_page_does_equal_sql_work_with_one_or_thirty_other_rooms(tmp_path, monkeypatch):
    counts = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        chat = own_room(root)
        foreign_rooms(root, rooms, 10)
        with metered(monkeypatch) as measured:
            status, page = history(root, chat_id=str(chat))
        assert status == 200 and [row["text"] for row in page["messages"] if not row.get("is_progress")] == [
            f"own answer {index}" for index in range(5)]
        counts.append(dict(measured))
    assert counts[0] == counts[1] and counts[0]["steps"] > 0, counts


def test_the_import_frontier_is_disclosed_and_never_read_past(tmp_path, monkeypatch):
    """Before the archive job ran, a recent page that reaches the frontier says so (one whose quota
    and window end above it does not); the older page that would cross it answers 503 migrating
    with its cursor; after the import the same cursor reads the archived rows and the recent page
    is complete."""
    def row(minute, text):
        return {"ts": ts(minute), "chat_id": 1, "direction": "in", "text": text, "client_message_id": text}

    seed(tmp_path, archives={"chat_20260901T000000.jsonl": [row(0, "archived 0"), row(1, "archived 1")]},
         live={"chat": [row(2, "live 2"), row(3, "live 3")]})
    di.cutover(tmp_path)
    with monkeypatch.context() as narrow:
        narrow.setattr(history_paging, "_RECENT_WINDOW_BYTES", 1)
        status, bounded = history(tmp_path, n_human="1")
    assert status == 200 and [m["text"] for m in bounded["messages"]] == ["live 3"]
    assert "chat_importing" not in bounded["window"]["truncated_by"], "the page did not reach the frontier"
    assert bounded["has_more"] is True

    status, recent = history(tmp_path)
    assert status == 200 and [m["text"] for m in recent["messages"]] == ["live 2", "live 3"]
    assert "chat_importing" in recent["window"]["truncated_by"] and recent["window"]["complete"] is False
    assert recent["has_more"] is True and recent["next_cursor"]
    frontier = recent["coverage"]["spans"]["chat"]["from"]
    assert frontier == (tmp_path / "archive" / "chat_20260901T000000.jsonl").stat().st_size
    assert "progress_importing" not in recent["window"]["truncated_by"], "the progress stream has no archive"

    status, older = history(tmp_path, cursor=recent["next_cursor"])
    assert status == 503 and older["reason_code"] == "migrating"
    assert older["next_cursor"] == recent["next_cursor"] and older["has_more"] is True and older["messages"] == []

    assert di.run_dialogue_import(tmp_path)["state"] == "done"
    status, older = history(tmp_path, cursor=recent["next_cursor"])
    assert status == 200 and [m["text"] for m in older["messages"]] == ["archived 0", "archived 1"]
    assert older["has_more"] is False and older["coverage"]["spans"]["chat"]["from"] == 0
    status, complete = history(tmp_path)
    assert status == 200 and complete["window"] == {"complete": True, "truncated_by": []}
    assert [m["text"] for m in complete["messages"]] == ["archived 0", "archived 1", "live 2", "live 3"]


def test_an_older_cursor_minted_before_the_import_reads_on_after_it(tmp_path):
    """A page cursor from the older body (its coordinates are the store's) names a range below the
    frontier: the replay says still importing until the job ran, then replays the same rows."""
    def row(minute, text):
        return {"ts": ts(minute), "chat_id": 1, "direction": "in", "text": text, "client_message_id": text}

    seed(tmp_path, archives={"chat_20260901T000000.jsonl": [row(0, "archived 0"), row(1, "archived 1")]},
         live={"chat": [row(2, "live 2")]})
    assert di.run_dialogue_import(tmp_path)["state"] == "done"
    status, page = history(tmp_path, n_human="1")
    status, older = history(tmp_path, cursor=page["next_cursor"])
    assert status == 200 and [m["text"] for m in older["messages"]] == ["archived 1"]
    frozen = older["page_cursor"]
    # The same conversation on a root whose archive job has not run yet (an upgrade in progress).
    other = tmp_path / "other"
    seed(other, archives={"chat_20260901T000000.jsonl": [row(0, "archived 0"), row(1, "archived 1")]},
         live={"chat": [row(2, "live 2")]})
    di.cutover(other)
    status, waiting = history(other, cursor=frozen)
    assert status == 503 and waiting["reason_code"] == "migrating" and waiting["next_cursor"] == frozen
    assert di.run_dialogue_import(other)["state"] == "done"
    status, replayed = history(other, cursor=frozen)
    assert status == 200 and [m["history_id"] for m in replayed["messages"]] == [m["history_id"] for m in older["messages"]]


def test_task_events_v2_waits_at_the_frontier_and_resumes_after_the_import(tmp_path):
    """A pending segment the task may have rows in is disclosed once and never crossed; the
    cursor stays before it, and the import's rows arrive at their coordinates afterwards."""
    seed(tmp_path, archives={"progress_20260901T000000.jsonl": [
        {"ts": ts(0), "task_id": "root", "content": "archived step"}]})
    write_task_result(tmp_path, "root", "running", ts=ts(0))
    di.cutover(tmp_path)
    ds.append(tmp_path, "progress", {"ts": ts(1), "task_id": "root", "content": "live step"})
    first = events_v2(tmp_path, "root")
    gaps = [row for row in first if row["type"] == "history_gap"]
    assert [gap["reason"] for gap in gaps] == ["still_importing"] and gaps[0]["source"] == "progress"
    assert [row["data"]["content"] for row in first if row["type"] == "progress"] == []
    position = first[-1]["cursor"]["positions"][str(tmp_path)]["progress"]
    assert position == 0, "nothing below the frontier is skipped"
    assert di.run_dialogue_import(tmp_path)["state"] == "done"
    resumed = events_v2(tmp_path, "root", first[-1]["cursor"])
    assert [row["data"]["content"] for row in resumed if row["type"] == "progress"] == ["archived step", "live step"]
    assert not any(row["type"] == "history_gap" for row in resumed)
    assert resumed[-1]["cursor"]["positions"][str(tmp_path)]["progress"] == ds.stream_end(tmp_path, "progress")
    events = iter_task_events(tmp_path, "root")
    assert [row["data"]["content"] for row in events if row["type"] == "progress"] == ["archived step", "live step"]
    assert [row["line"] for row in events if row["type"] == "progress"] == [1, 2]


def test_task_events_v1_discloses_the_frontier_once_and_omits_unsettled_lines(tmp_path):
    seed(tmp_path, archives={"progress_20260901T000000.jsonl": [
        {"ts": ts(0), "task_id": "root", "content": "archived step"}]})
    write_task_result(tmp_path, "root", "running", ts=ts(0))
    di.cutover(tmp_path)
    ds.append(tmp_path, "progress", {"ts": ts(1), "task_id": "root", "content": "live step"})
    from ouroboros.gateway.task_events import _TaskEventFollower

    follower = _TaskEventFollower(tmp_path, "root")
    rows = follower.full_merge()
    assert [row["data"].get("content") for row in rows if row["type"] == "progress"] == ["live step"]
    assert all("line" not in row for row in rows if row["type"] == "progress"), "an unsettled count is no number"
    notices = follower.take_notices()
    assert [notice["reason"] for notice in notices] == ["still_importing"] and follower.take_notices() == []
    assert di.run_dialogue_import(tmp_path)["state"] == "done"
    rows = follower.full_merge()
    assert [row["line"] for row in rows if row["type"] == "progress"] == [1, 2]
    assert follower.take_notices() == []


def test_a_project_room_finds_its_bound_tasks_descendants_by_the_stores_lineage(tmp_path):
    """A grandchild's narration the supervisor recorded under Main with its task id alone belongs
    to the room its root was turned into; Main keeps none of it."""
    project = create_project(tmp_path, "room", name="Room")
    write_task_result(tmp_path, "par", "running", root_task_id="par", chat_id=1)
    bind_task_to_project(tmp_path, "par", project["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "kid", "running", delegation_role="subagent", parent_task_id="par", root_task_id="par")
    write_task_result(tmp_path, "grandkid", "running", delegation_role="subagent", parent_task_id="kid", root_task_id="par")
    ds.append(tmp_path, "progress", {"ts": ts(1), "chat_id": 1, "task_id": "grandkid", "content": "deep work",
                                     "is_progress": True, "role": "system"})
    ds.append(tmp_path, "progress", {"ts": ts(2), "chat_id": 1, "task_id": "stranger", "content": "someone else"})
    status, room = history(tmp_path, chat_id=str(project["chat_id"]), n_human="0")
    assert status == 200 and [row["text"] for row in room["messages"] if row.get("is_progress")] == ["deep work"]
    status, main = history(tmp_path, n_human="0")
    assert status == 200 and [row["text"] for row in main["messages"] if row.get("is_progress")] == ["someone else"]
