"""``after_cursor``: an open window reads only the rows stored since its last read.

The newest page names where new arrivals start; passed back as ``cursor`` it returns exactly
the rows appended after that read (rotation included) and the next ``after_cursor``, never
older rows, and refuses (409) when the room's view changed or too much arrived to replay.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ouroboros import dialogue_import, dialogue_store as ds
from ouroboros.gateway import history_paging
from ouroboros.gateway.history import make_chat_history_endpoint

TS = "2026-10-16T10:00:{:02d}Z"


def _append(root, name, rows):
    for row in rows:
        ds.append(root, name, row)


def _say(root, *texts, start=0):
    _append(root, "chat", [{"ts": TS.format(start + index), "direction": "in", "chat_id": 1, "text": text}
                           for index, text in enumerate(texts)])


def _read(root, cursor=None):
    params = {"chat_id": "1", **({"cursor": cursor} if cursor else {})}
    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params=params)))
    return response.status_code, json.loads(response.body)


def test_the_newest_page_names_the_start_of_new_arrivals_and_a_delta_returns_only_them(tmp_path):
    _say(tmp_path, "first", "second")
    _append(tmp_path, "progress", [{"ts": TS.format(2), "content": "working", "task_id": "t1", "chat_id": 1}])
    status, page = _read(tmp_path)
    assert status == 200 and [row["text"] for row in page["messages"]] == ["first", "second", "working"]
    assert page["after_cursor"]

    status, empty = _read(tmp_path, page["after_cursor"])
    assert status == 200 and empty["messages"] == [] and empty["has_more"] is False and empty["after_cursor"]

    _say(tmp_path, "third", start=3)
    _append(tmp_path, "progress", [{"ts": TS.format(4), "content": "still working", "task_id": "t1", "chat_id": 1}])
    status, delta = _read(tmp_path, empty["after_cursor"])
    assert status == 200 and [row["text"] for row in delta["messages"]] == ["third", "still working"]
    assert delta["has_more"] is False and delta["next_cursor"] is None
    # The same identities the full page would give: a window merges by history_id.
    _, again = _read(tmp_path)
    by_id = {row["history_id"]: row["text"] for row in again["messages"]}
    assert all(by_id[row["history_id"]] == row["text"] for row in delta["messages"])
    # An older page continues older history; it does not move where new arrivals start.
    assert "after_cursor" not in _read(tmp_path, page["page_cursor"])[1]


def test_new_arrivals_an_older_body_appended_are_found_after_the_next_cutover(tmp_path):
    """Rows a store-unaware body wrote to ``logs/chat.jsonl`` (a downgrade ran) join the store at
    the next boot's cutover, at their chain place: the window's delta finds them like any arrival."""
    _say(tmp_path, "before the downgrade")
    _, page = _read(tmp_path)
    live = tmp_path / "logs" / "chat.jsonl"
    live.parent.mkdir(parents=True, exist_ok=True)
    with live.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": TS.format(1), "direction": "in", "chat_id": 1, "text": "written by the older body"}) + "\n")
    dialogue_import.cutover(tmp_path)
    _say(tmp_path, "after the upgrade", start=2)
    status, delta = _read(tmp_path, page["after_cursor"])
    assert status == 200
    assert [row["text"] for row in delta["messages"]] == ["written by the older body", "after the upgrade"]


def test_a_changed_room_view_or_a_long_absence_takes_the_newest_page_again(tmp_path, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    _say(tmp_path, "hello")
    _, page = _read(tmp_path)
    project = create_project(tmp_path, "room", name="Room")
    bind_task_to_project(tmp_path, "some-task", project["id"], origin={"absent": "system"})
    status, refused = _read(tmp_path, page["after_cursor"])
    assert status == 409 and refused["reason_code"] == "history_view_changed"

    _, page = _read(tmp_path)
    monkeypatch.setattr(history_paging, "_DELTA_CEILING_BYTES", 64)
    _say(tmp_path, *(f"message {index}" for index in range(5)), start=1)
    status, refused = _read(tmp_path, page["after_cursor"])
    assert status == 409 and refused["reason_code"] == "history_delta_reset"
    assert _read(tmp_path)[0] == 200


def test_a_new_owner_message_annotation_takes_the_newest_page_again(tmp_path):
    """Annotations change older owner rows in place (no new row): a window cannot be brought up
    to date by new arrivals once they changed, so ``after`` refuses and the newest page answers."""
    from ouroboros.project_dialogue import append_chat_annotation

    _append(tmp_path, "chat", [{"ts": TS.format(0), "direction": "in", "chat_id": 1, "text": "route me",
                                "client_message_id": "msg-1"}])
    _, page = _read(tmp_path)
    status, quiet = _read(tmp_path, page["after_cursor"])
    assert status == 200 and quiet["messages"] == []  # nothing changed: the delta answers
    append_chat_annotation(tmp_path, "msg-1", action="promote_chat_to_task", status="scheduled")
    status, refused = _read(tmp_path, quiet["after_cursor"])
    assert status == 409 and refused["reason_code"] == "history_delta_reset"
    status, fresh = _read(tmp_path)
    assert status == 200 and fresh["messages"][0]["chat_annotation"]["status"] == "scheduled"
    assert _read(tmp_path, fresh["after_cursor"])[0] == 200


def test_new_arrivals_in_a_project_name_their_newest_message(tmp_path):
    from ouroboros.projects_registry import create_project

    chat = create_project(tmp_path, "arrivals", name="Arrivals")["chat_id"]

    def project_read(cursor=None):
        params = {"chat_id": str(chat), **({"cursor": cursor} if cursor else {})}
        response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params=params)))
        return response.status_code, json.loads(response.body)

    _append(tmp_path, "chat", [{"ts": TS.format(0), "direction": "out", "chat_id": chat, "text": "first answer"}])
    _, page = project_read()
    status, quiet = project_read(page["after_cursor"])
    assert status == 200 and "latest_message" not in quiet["window"]  # no new message: the window keeps its own
    _append(tmp_path, "chat", [{"ts": TS.format(1), "direction": "out", "chat_id": chat, "text": "second answer"},
                               {"ts": TS.format(2), "direction": "in", "chat_id": chat, "text": "owner reply"}])
    status, delta = project_read(quiet["after_cursor"])
    newest = next(row for row in delta["messages"] if row["text"] == "second answer")
    assert status == 200 and delta["window"]["latest_message"] == {"history_id": newest["history_id"],
                                                                     "out_of_order": False}
