"""A window that applies ``repairs`` (``?repair=settled``, CLIENT_CONTRACT "Reading history") keeps reading what
arrived when a task it shows settles: the ``after`` read sends that task's rows the window holds again, projected
as the newest page projects them now, instead of refusing (409 ``history_delta_reset``/``task_settled``) and making
the window read the whole newest page again. A task the page shows through another task's document or rows, a
window that shows an owner's send still pending, or a client that did not ask keeps the refusal.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

from ouroboros import dialogue_store as ds
from ouroboros.gateway import history_attribution, history_paging
from ouroboros.gateway.history import make_chat_history_endpoint
from tests._dialogue import healthy_main_fixture  # noqa: F401 -- the "healthy_main" fixture
from tests._store_shared import seed_result

TS = "2026-10-16T10:00:{:02d}Z"
QUESTION = {"quiz_id": "q1", "question": "Which?", "options": ["a", "b"]}


def _read(root, cursor=None, *, repair=True, chat=1, recorded=False):
    params = {"chat_id": str(chat), **({"cursor": cursor} if cursor else {}), **({"repair": "settled"} if repair else {})}
    endpoint = make_chat_history_endpoint(root)
    if recorded:
        endpoint = history_attribution.recorded(endpoint)
    response = asyncio.run(endpoint(SimpleNamespace(query_params=params, method="GET")))
    return response.status_code, json.loads(response.body)


def _cursor(token):
    return json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))


def _task(root, task_id, status, index, **facts):
    seed_result(root, task_id, {"task_id": task_id, "status": status, "chat_id": 1, "ts": TS.format(index), **facts})


def _asking(root, index=0):
    """A running task with an open question, its narration and its words."""
    _task(root, "asker", "running", index, owner_quiz={"q1": {**QUESTION, "state": "open"}})
    for stream, row in (
            ("chat", {"type": "quiz", "direction": "out", "text": "Which?", "quiz": {**QUESTION, "state": "open"}}),
            ("progress", {"content": "working on it"}),
            ("chat", {"direction": "out", "text": "while you decide"})):
        ds.append(root, stream, {"ts": TS.format(index), "chat_id": 1, "task_id": "asker", **row})


def _ids(rows, task="asker"):
    return [row["history_id"] for row in rows if row.get("task_id") == task and row.get("history_id")]


def test_a_settled_task_is_answered_with_its_held_rows_as_the_newest_page_shows_them(tmp_path):
    _asking(tmp_path)
    _task(tmp_path, "done", "completed", 1)
    ds.append(tmp_path, "chat", {"ts": TS.format(1), "direction": "out", "chat_id": 1, "task_id": "done", "text": "ok"})
    status, page = _read(tmp_path)
    held = _cursor(page["after_cursor"])["held"]
    assert status == 200 and set(held) == {"asker", "done"}
    shown = _ids(page["messages"])
    assert len(shown) == 3

    # The question closes and the task settles with no new row: its rows say so again, nothing arrived.
    _task(tmp_path, "asker", "completed", 0, owner_quiz={"q1": {**QUESTION, "state": "closed"}})
    status, repaired = _read(tmp_path, page["after_cursor"], recorded=True)
    counted = history_attribution.drain()["groups"]
    assert status == 200 and repaired["messages"] == [], repaired
    assert _ids(repaired["repairs"]) == shown and "latest_message" not in repaired["window"]
    status, newest = _read(tmp_path)
    assert [row for row in newest["messages"] if row.get("history_id") in shown] == repaired["repairs"]
    question = next(row for row in repaired["repairs"] if row.get("msg_type") == "quiz")
    assert question["quiz"]["state"] == "closed"
    assert [group.get("repairs") for group in counted] == ["settled"]

    # The window is up to date at the new cursor: the next read is quiet.
    status, quiet = _read(tmp_path, repaired["after_cursor"])
    assert status == 200 and quiet["messages"] == [] and "repairs" not in quiet


def test_without_the_ask_a_settled_task_still_takes_the_newest_page(tmp_path):
    _asking(tmp_path)
    status, legacy = _read(tmp_path, repair=False)
    assert status == 200 and "held" not in _cursor(legacy["after_cursor"])
    _task(tmp_path, "asker", "completed", 0)
    status, refused = _read(tmp_path, legacy["after_cursor"])  # asked only now: the cursor holds no rows
    assert (status, refused["detail"]) == (409, "task_settled")
    _task(tmp_path, "asker", "running", 0)
    _, page = _read(tmp_path)
    _task(tmp_path, "asker", "completed", 0)
    status, refused = _read(tmp_path, page["after_cursor"], repair=False)  # held rows, but not asked on this read
    assert (status, refused["detail"]) == (409, "task_settled")


def test_what_arrived_with_a_settling_task_stays_the_arrival_and_the_newest_message(tmp_path):
    _asking(tmp_path)
    _, page = _read(tmp_path)
    _task(tmp_path, "asker", "completed", 0)
    ds.append(tmp_path, "chat", {"ts": TS.format(9), "direction": "out", "chat_id": 1, "task_id": "asker",
                                 "text": "the final answer"})
    status, read = _read(tmp_path, page["after_cursor"])
    assert status == 200 and [row["text"] for row in read["messages"]] == ["the final answer"]
    assert _ids(read["repairs"]) == _ids(page["messages"])
    assert read["window"]["latest_message"]["history_id"] == read["messages"][0]["history_id"]
    # The rows the window now holds of it include the arrival: a later change sends them all again.
    _task(tmp_path, "asker", "completed", 0, title="renamed after it settled")
    status, again = _read(tmp_path, read["after_cursor"])
    assert status == 200 and _ids(again["repairs"]) == _ids(page["messages"]) + _ids(read["messages"])


def test_an_unrelated_change_sends_nothing_again(tmp_path):
    _asking(tmp_path)
    _task(tmp_path, "elsewhere", "running", 1)
    _, page = _read(tmp_path)
    _task(tmp_path, "elsewhere", "completed", 1)  # not shown in this window
    _task(tmp_path, "asker", "running", 0, title="still running")  # running: its rows carry its facts
    status, quiet = _read(tmp_path, page["after_cursor"])
    assert status == 200 and quiet["messages"] == [] and "repairs" not in quiet


def test_a_task_shown_through_another_tasks_document_keeps_the_refusal(tmp_path):
    _task(tmp_path, "old", "failed", 0, retry_task_id="new")
    _task(tmp_path, "new", "completed", 1)
    ds.append(tmp_path, "chat", {"ts": TS.format(0), "direction": "out", "chat_id": 1, "task_id": "old", "text": "x"})
    _asking(tmp_path, 2)
    _, page = _read(tmp_path)
    assert set(_cursor(page["after_cursor"])["held"]) == {"asker"}  # neither the retried task nor its successor
    _task(tmp_path, "new", "completed", 1, title="settled again")
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")


def test_a_window_showing_a_pending_send_keeps_the_refusal_until_its_next_newest_page(tmp_path, monkeypatch):
    import supervisor.message_ingress as ingress

    _asking(tmp_path)
    pending = {"on": True}
    monkeypatch.setattr(ingress, "delivery_facts", lambda row, chat: {"ingress_pending": True} if pending["on"] else {})
    ds.append(tmp_path, "chat", {"ts": TS.format(3), "direction": "in", "chat_id": 1, "text": "do it",
                                 "client_message_id": "c1", "ingress_accepted": True})
    _, page = _read(tmp_path)
    assert "held" not in _cursor(page["after_cursor"])
    pending["on"] = False  # entered: no row says so, and the window still shows it pending
    _task(tmp_path, "asker", "completed", 0)
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")
    _, fresh = _read(tmp_path)
    assert "asker" in _cursor(fresh["after_cursor"])["held"]


def test_a_held_row_that_cannot_be_read_refuses(tmp_path, monkeypatch):
    _asking(tmp_path)
    _, page = _read(tmp_path)
    _task(tmp_path, "asker", "completed", 0)
    rows_at = ds.rows_at
    monkeypatch.setattr(ds, "rows_at", lambda root, stream, seqs: dict(list(rows_at(root, stream, seqs).items())[1:]))
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"], refused["stream"]) == (409, "replay_gap", "chat")


def test_a_window_too_large_to_name_its_rows_reads_on_without_them(tmp_path, monkeypatch):
    _asking(tmp_path)
    _, plain = _read(tmp_path, repair=False)
    monkeypatch.setattr(history_paging, "_CURSOR_CHARS", len(plain["after_cursor"]) + 4)
    status, page = _read(tmp_path)
    assert status == 200 and "held" not in _cursor(page["after_cursor"])
    _task(tmp_path, "asker", "completed", 0)
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")


def test_a_held_task_keeps_the_version_its_rows_were_projected_at(tmp_path):
    _asking(tmp_path)
    _, page = _read(tmp_path)
    seen = _cursor(page["after_cursor"])["tasks"]["asker"]
    _task(tmp_path, "asker", "running", 0, title="later")
    ds.append(tmp_path, "progress", {"ts": TS.format(5), "chat_id": 1, "task_id": "asker", "content": "more"})
    status, going = _read(tmp_path, page["after_cursor"])
    assert status == 200 and [row["text"] for row in going["messages"] if row.get("is_progress")] == ["more"]
    assert _cursor(going["after_cursor"])["tasks"]["asker"] == seen  # its earlier rows still show that version


def test_the_cursor_names_held_rows_only_below_its_boundary(tmp_path):
    _asking(tmp_path)
    _, page = _read(tmp_path)
    cursor = _cursor(page["after_cursor"])
    seqs = history_paging._held_seqs(cursor["held"]["asker"], cursor["upper"])
    assert [len(seqs[0]), len(seqs[1])] == [2, 1]
    for broken in ({**cursor, "held": {"asker": [[cursor["upper"]["chat"]], []]}},
                   {**cursor, "held": {"asker": [[5, 0], []]}},
                   {**cursor, "held": {"stranger": [[], []]}}):
        token = base64.urlsafe_b64encode(json.dumps(broken).encode()).decode().rstrip("=")
        assert _read(tmp_path, token)[0] == 400


def test_a_task_not_held_once_in_a_window_is_never_held_in_it(tmp_path):
    _task(tmp_path, "old", "failed", 0, retry_task_id="new")
    _task(tmp_path, "new", "running", 1)
    ds.append(tmp_path, "chat", {"ts": TS.format(0), "direction": "out", "chat_id": 1, "task_id": "old", "text": "x"})
    _, page = _read(tmp_path)
    assert "new" in _cursor(page["after_cursor"])["tasks"] and _cursor(page["after_cursor"])["held"] == {}
    ds.append(tmp_path, "chat", {"ts": TS.format(2), "direction": "out", "chat_id": 1, "task_id": "new", "text": "y"})
    status, going = _read(tmp_path, page["after_cursor"])
    # Its own new row would be all the window holds of it: the old task's rows show its truth too.
    assert status == 200 and _cursor(going["after_cursor"])["held"] == {}


class _Reads:
    versions, replica_results = {}, set()

    def overlaid(self, tasks):
        return set(tasks)


def _row(task, seq, **extra):
    return {"task_id": task, "role": "assistant", "history_id": f"h{seq}",
            "history_position": {"source": "chat", "offset": seq}, **extra}


def test_held_rows_leave_out_tasks_the_page_shows_through_another_or_without_a_row():
    page = {"lens": {"chat": history_paging.RoomLens((("appearance", "room:1"),))}, "after": False,
            "stripped": {"h4"}}
    messages = [_row("plain", 1), _row("plain", 2, is_progress=True, history_position={"source": "progress", "offset": 7}),
                _row("plain", 3, role="user"),  # an owner's row: never held
                _row("stripped", 4),  # a child's row shown without its lineage
                _row("receipt", 5, historical_terminal={"status": "completed"}),
                _row("pinned", 6, system_type="project_started"),
                _row("reviewer", 8, presentation_owner_task_id="owner"), _row("owner", 9),
                _row("grouped", 10, review_group={"attempts": [{"task_id": "attempt"}]}), _row("attempt", 11),
                {"task_id": "carried_only", "role": "system", "system_type": "task_evidence"}]
    shown = {"plain", "stripped", "receipt", "reviewer", "owner", "grouped", "attempt", "carried_only"}
    assert history_paging._held_rows(page, messages, shown, _Reads()) == {"plain": [[1], [7]]}
    place = {**page, "lens": {"chat": history_paging.RoomLens((("chat_id", 9),))}}
    assert history_paging._held_rows(place, messages, shown, _Reads()) is None  # a place room holds none
