"""``around=task:<id>``: the page around one task, wherever it lies in the room's history.

The work list of a room names its work; a press goes to that work's message even when the
message is far older than the loaded window (DESIGN "Room work status"). The read lands on the
task's earliest row of the room -- one indexed probe of the task's own rows in the dialogue store,
never a walk of the room from the present -- holds a quarter of each quota before it and the rest
after it, and says what it found: ``found`` (and where), ``moved`` (the task belongs to another
room) or ``absent``; while the archive still imports, an unfound task is not yet known absent.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ouroboros import dialogue_store as ds
from ouroboros.gateway import history_paging
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.store import StoreError
from tests import _dialogue as dialogue_test
from tests.test_dialogue_addressed_readers import metered

TS = "2026-10-13T{:02d}:{:02d}:{:02d}Z"


def _ts(index):
    return TS.format(index // 3600, index // 60 % 60, index % 60)


def _append(root, name, rows):
    for row in rows:
        ds.append(root, name, row)


def _read(root, **params):
    query = {"chat_id": "1", **{key: str(value) for key, value in params.items()}}
    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params=query)))
    return response.status_code, json.loads(response.body)


def _say(root, first, count, chat_id=1, prefix="m"):
    _append(root, "chat", [{"ts": _ts(first + index), "direction": "in", "chat_id": chat_id,
                            "text": f"{prefix}{first + index}"} for index in range(count)])


def _history(root):
    """Owner chatter, then a task far back (its owner message, its reply and narration), then a long
    present that a newest page cannot reach past."""
    _say(root, 0, 40)
    _append(root, "chat", [
        {"ts": _ts(40), "direction": "in", "chat_id": 1, "text": "please audit the release"},
        {"ts": _ts(42), "direction": "out", "chat_id": 1, "task_id": "t-old", "text": "Auditing the release now."},
    ])
    _append(root, "progress", [{"ts": _ts(43 + index), "chat_id": 1, "task_id": "t-old", "content": f"step {index}"}
                               for index in range(3)])
    _say(root, 50, 400)
    _append(root, "progress", [{"ts": _ts(460 + index), "chat_id": 1, "task_id": "t-new", "content": f"now {index}"}
                               for index in range(3)])


def _texts(page):
    return [row.get("text") or row.get("content") for row in page["messages"]]


def test_a_task_far_older_than_the_newest_page_is_read_in_one_page_around_it(tmp_path):
    _history(tmp_path)
    status, newest = _read(tmp_path, n_human=8, n_progress=4)
    assert status == 200 and "Auditing the release now." not in _texts(newest)  # out of reach of the present

    status, page = _read(tmp_path, around="task:t-old", n_human=8, n_progress=4)
    assert status == 200
    anchor = page["anchor"]
    reply = next(row for row in page["messages"] if row.get("text") == "Auditing the release now.")
    assert anchor == {"state": "found", "task_id": "t-old", "history_id": reply["history_id"]}
    chat = [row["text"] for row in page["messages"] if row["history_id"].startswith("chat:") and row.get("text")]
    at = chat.index("Auditing the release now.")
    # A quarter of the 8-row quota before the anchor (its owner message among them), the rest after.
    assert chat[:at] == ["m39", "please audit the release"]
    assert chat[at + 1:] == ["m50", "m51", "m52", "m53", "m54"]
    assert {"step 0", "step 1", "step 2"} <= set(_texts(page))  # the task's own narration, around its first row
    assert "now 0" not in _texts(page)  # never today's narration under a three-day-old conversation
    assert page["has_more"] is True and page["next_cursor"] and "after_cursor" not in page
    assert page["window"]["complete"] is False

    # The page re-reads as itself, and Load more continues older from its lower edge.
    status, again = _read(tmp_path, cursor=page["page_cursor"], n_human=8, n_progress=4)
    assert status == 200 and _texts(again) == _texts(page)
    status, older = _read(tmp_path, cursor=page["next_cursor"])
    assert status == 200 and _texts(older)[-1] == "m38" and "please audit the release" not in _texts(older)


def test_a_task_with_no_rows_is_absent_and_a_bad_address_is_refused(tmp_path):
    _history(tmp_path)
    status, page = _read(tmp_path, around="task:t-never")
    assert status == 200
    assert page["anchor"] == {"state": "absent", "task_id": "t-never"}
    assert page["messages"] == [] and page["page_cursor"] is None and page["has_more"] is False
    # The working read of the same room is untouched by the refusal paths below.
    assert _read(tmp_path, around="task:t-old")[1]["anchor"]["state"] == "found"
    for bad in ("t-old", "message:t-old", "task:", "task:a b"):
        status, body = _read(tmp_path, around=bad)
        assert status == 400 and body["reason_code"] == "history_around_invalid", bad
    _, newest = _read(tmp_path)
    status, body = _read(tmp_path, around="task:t-old", cursor=newest["page_cursor"])
    assert status == 400 and body["reason_code"] == "history_around_invalid"


def test_a_source_that_cannot_be_read_says_so_and_the_newest_page_still_reads(tmp_path, monkeypatch):
    _history(tmp_path)

    def unreadable(*_args, **_kwargs):
        raise StoreError("dialogue store unavailable")

    monkeypatch.setattr(history_paging.HistorySource, "task_pivot", unreadable)
    status, body = _read(tmp_path, around="task:t-old")
    assert status == 503 and body["reason_code"] == "history_source_unavailable" and body["messages"] == []
    assert _read(tmp_path)[0] == 200


def test_a_task_bound_to_a_project_has_moved_from_main_and_is_found_in_its_project(tmp_path):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    _history(tmp_path)
    project = create_project(tmp_path, "audit", name="Audit")
    bind_task_to_project(tmp_path, "t-old", project["id"], origin={"absent": "system"})
    status, page = _read(tmp_path, around="task:t-old")
    assert status == 200 and page["anchor"] == {"state": "moved", "task_id": "t-old"} and page["messages"] == []
    status, page = _read(tmp_path, around="task:t-old", chat_id=project["chat_id"])
    assert status == 200 and page["anchor"]["state"] == "found"
    assert "Auditing the release now." in _texts(page)


def test_a_task_that_only_narrated_lands_on_its_narration_and_names_the_missing_conversation(tmp_path):
    _history(tmp_path)
    _append(tmp_path, "progress", [{"ts": _ts(470), "chat_id": 1, "task_id": "t-quiet", "content": "thinking"}])
    status, page = _read(tmp_path, around="task:t-quiet", n_human=8, n_progress=4)
    assert status == 200 and page["anchor"]["state"] == "found"
    assert page["anchor"]["history_id"].startswith("progress:") and "thinking" in _texts(page)
    assert "chat_around_unpositioned" in page["window"]["truncated_by"]
    # The chain it starts pages narration on: Load more continues older narration, never the present.
    assert page["has_more"] is True and page["coverage"]["spans"]["chat"] is None
    status, older = _read(tmp_path, cursor=page["next_cursor"])
    assert status == 200 and "now 1" in _texts(older) and not any(row["history_id"].startswith("chat:")
                                                                     for row in older["messages"])


def test_a_task_waiting_on_its_model_lands_on_its_waiting_card_never_another_room(tmp_path):
    """Its only row is the model wait its card shows: that row is in this room, so it is where the task is."""
    _history(tmp_path)
    _append(tmp_path, "progress", [{"ts": _ts(472), "chat_id": 1, "task_id": "t-waits", "type": "task_model_wait",
                                    "wait_id": "w-1", "revision": 1, "content": "waiting for the model"}])
    status, page = _read(tmp_path, around="task:t-waits")
    assert status == 200 and page["anchor"]["state"] == "found"
    assert page["anchor"]["history_id"].startswith("progress:")
    assert any(row.get("task_id") == "t-waits" for row in page["messages"]), "the waiting card is on the page"


def test_a_task_whose_only_row_main_hides_is_absent_never_another_room(tmp_path):
    """Main hides a task's terminal projection, yet its room rule keeps it: nothing to show, nothing moved."""
    _history(tmp_path)
    _append(tmp_path, "chat", [{"ts": _ts(473), "direction": "out", "chat_id": 1, "task_id": "t-silent",
                                "type": "task_summary", "summary_kind": "terminal_result_projection", "text": "Done."}])
    assert _read(tmp_path, around="task:t-silent")[1]["anchor"] == {"state": "absent", "task_id": "t-silent"}


def _said(task, index, **extra):
    return {"ts": _ts(index), "direction": "out", "chat_id": 1, "task_id": task, "text": f"{task} {index}", **extra}


def test_the_task_is_found_by_its_own_rows_whatever_the_room_holds(tmp_path, monkeypatch):
    """The pivot is one indexed probe of the task's rows: the same SQL work with ten or four hundred
    rows of other work around it, and the page lands on the same place."""
    counts, landed = [], []
    for others in (10, 400):
        root = tmp_path / str(others)
        _append(root, "chat", [_said("t-other", index) for index in range(others)])
        _append(root, "chat", [_said("t-x", others), _said("t-x", others + 1)])
        _append(root, "chat", [_said("t-later", others + 2 + index) for index in range(10)])
        reader = history_paging.HistorySource(root, "chat", history_paging.RoomLens((("chat_id", 1),)))
        with metered(monkeypatch) as measured:
            state, row = reader.task_pivot("t-x", lambda row: True)
        counts.append(dict(measured))
        landed.append((state, row["text"]))
    assert landed == [("found", "t-x 10"), ("found", "t-x 400")]
    assert counts[0] == counts[1] and counts[0]["statements"] > 0, counts


def test_the_page_after_a_task_reads_on_across_archives(tmp_path):
    dialogue_test.seed(tmp_path, archives={
        "chat_20261013T110000.jsonl": [_said("t-a", index) for index in range(6)] + [_said("t-x", 6)],
        "chat_20261013T120000.jsonl": [_said("t-b", index) for index in range(7, 10)]},
        live={"chat": [_said("t-c", index) for index in range(10, 30)]})
    dialogue_test.import_all(tmp_path)
    status, page = _read(tmp_path, around="task:t-x", n_human=8, n_progress=0)
    assert status == 200 and page["anchor"]["state"] == "found"
    assert _texts(page) == ["t-a 4", "t-a 5", "t-x 6", "t-b 7", "t-b 8", "t-b 9", "t-c 10", "t-c 11"]
    status, again = _read(tmp_path, cursor=page["page_cursor"], n_human=8, n_progress=0)
    assert status == 200 and _texts(again) == _texts(page)


def test_an_unfound_task_is_still_importing_never_absent_until_the_archive_is_in(tmp_path):
    """A task whose rows lie in an archive segment the import has not reached is not known absent:
    the read answers 503 migrating; once imported, the same read lands on it."""
    dialogue_test.seed(tmp_path, archives={"chat_20261013T110000.jsonl": [_said("t-x", 1)]},
                       live={"chat": [_said("t-c", 2)]})
    from ouroboros import dialogue_import
    dialogue_import.cutover(tmp_path)  # the live file is in; the archive is planned, not imported yet
    status, body = _read(tmp_path, around="task:t-x")
    assert status == 503 and body["reason_code"] == "migrating" and body["messages"] == []
    assert _read(tmp_path, around="task:t-c")[1]["anchor"]["state"] == "found"  # found above the frontier
    dialogue_test.import_all(tmp_path)
    status, page = _read(tmp_path, around="task:t-x")
    assert status == 200 and page["anchor"]["state"] == "found" and "t-x 1" in _texts(page)


def test_an_around_page_is_not_the_present(tmp_path, monkeypatch):
    """The live lifecycle row and current-work evidence belong to the newest page only."""
    from ouroboros.gateway import history

    _history(tmp_path)
    lifecycle = {"ts": _ts(999), "role": "system", "text": "Restarting…", "system_type": "lifecycle"}
    monkeypatch.setattr(history, "_active_lifecycle_row", lambda matches: dict(lifecycle))
    assert "Restarting…" in _texts(_read(tmp_path)[1])
    status, page = _read(tmp_path, around="task:t-old")
    assert status == 200 and "Restarting…" not in _texts(page)
