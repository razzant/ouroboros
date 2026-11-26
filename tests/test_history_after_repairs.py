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

import pytest

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
    import zlib

    packed = token.startswith("z")  # a cursor naming held rows travels deflated
    raw = base64.urlsafe_b64decode(token[packed:] + "=" * (-len(token[packed:]) % 4))
    return json.loads(zlib.decompress(raw) if packed else raw)


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


def test_held_rows_travel_packed_and_a_window_too_large_even_so_reads_on_without_them(tmp_path, monkeypatch):
    for index in range(40):
        ds.append(tmp_path, "progress", {"ts": TS.format(1), "chat_id": 1, "task_id": "asker", "content": f"step {index}"})
    _asking(tmp_path)
    _, plain = _read(tmp_path, repair=False)
    status, page = _read(tmp_path)
    assert status == 200 and page["after_cursor"].startswith("z") and plain["after_cursor"].startswith("e")
    assert len(_cursor(page["after_cursor"])["held"]["asker"][1]) == 41
    _task(tmp_path, "asker", "completed", 0)
    status, repaired = _read(tmp_path, page["after_cursor"])
    assert status == 200 and len(repaired["repairs"]) == 2 + 41
    _task(tmp_path, "asker", "running", 0)
    monkeypatch.setattr(history_paging, "_CURSOR_CHARS", len(plain["after_cursor"]) + 4)
    encode = history_paging.encode_cursor  # rows that do not fit even packed: none held, the arrivals go on
    monkeypatch.setattr(history_paging, "encode_cursor", lambda value, packed=False: "z" * 10_000 if packed else encode(value))
    status, page = _read(tmp_path)
    assert status == 200 and "held" not in _cursor(page["after_cursor"])
    _task(tmp_path, "asker", "completed", 0)
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")


def test_a_packed_cursor_that_inflates_past_its_bound_is_invalid(tmp_path, monkeypatch):
    import zlib

    _asking(tmp_path)
    _, page = _read(tmp_path)
    monkeypatch.setattr(history_paging, "_UNPACKED_CURSOR_BYTES", 64)
    token = "z" + base64.urlsafe_b64encode(zlib.compress(json.dumps(_cursor(page["after_cursor"])).encode())).decode()
    assert _read(tmp_path, token.rstrip("="))[0] == 400


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
    answer = {"type": "quiz_answer", "task_id": "plain", "history_position": {"source": "chat", "offset": 12}}
    page = {"lens": {"chat": history_paging.RoomLens((("appearance", "room:1"),))}, "after": False,
            "stripped": {"h4"}, "selections": {"chat": ([answer], 0, set()), "progress": ([], 0, set())}}
    messages = [_row("plain", 1), _row("plain", 2, is_progress=True, history_position={"source": "progress", "offset": 7}),
                _row("plain", 3, role="user"),  # an owner's row: never held
                _row("stripped", 4),  # a child's row shown without its lineage
                _row("receipt", 5, historical_terminal={"status": "completed"}),
                _row("pinned", 6, system_type="project_started"), _row("waits", 13, system_type="task_model_wait"),
                _row("reviewer", 8, presentation_owner_task_id="owner"), _row("owner", 9),
                _row("grouped", 10, review_group={"attempts": [{"task_id": "attempt"}]}), _row("attempt", 11),
                {"task_id": "carried_only", "role": "system", "system_type": "task_evidence"}]
    shown = {"plain", "stripped", "receipt", "pinned", "waits", "reviewer", "owner", "grouped", "attempt", "carried_only"}
    # The answer the page drew no row of is still evidence its question's row was projected with.
    assert history_paging._held_rows(page, messages, shown, _Reads()) == {"plain": [[1, 12], [7]]}  # a fixed row: no facts
    place = {**page, "lens": {"chat": history_paging.RoomLens((("chat_id", 9),))}}
    assert history_paging._held_rows(place, messages, shown, _Reads()) is None  # a place room holds none


def test_a_question_answered_on_the_page_keeps_its_answer_when_its_task_settles(tmp_path):
    """The page draws the answered question, not its answer row; that row is still what the question's row was
    projected with, so a repair reads it again and shows the question as the newest page does."""
    answered = {**QUESTION, "state": "answered", "answered_index": 1}
    _task(tmp_path, "asker", "running", 0, owner_quiz={"q1": answered})
    ds.append(tmp_path, "chat", {"ts": TS.format(0), "chat_id": 1, "task_id": "asker", "type": "quiz", "direction": "out",
                                 "text": "Which?", "quiz": {**QUESTION, "state": "open"}})
    ds.append(tmp_path, "chat", {"ts": TS.format(1), "chat_id": 1, "task_id": "asker", "direction": "system",
                                 "type": "quiz_answer", "source": "owner_quiz_answer", "text": "b", "quiz": answered})
    ds.append(tmp_path, "progress", {"ts": TS.format(2), "chat_id": 1, "task_id": "asker", "content": "go on with b"})
    _, page = _read(tmp_path)
    question = next(row for row in page["messages"] if row.get("msg_type") == "quiz")
    assert question["quiz"].get("answer_row_address"), question["quiz"]
    _task(tmp_path, "asker", "completed", 0, owner_quiz={"q1": answered})
    status, repaired = _read(tmp_path, page["after_cursor"])
    assert status == 200 and repaired["messages"] == []
    status, newest = _read(tmp_path)
    again = {row["history_id"] for row in repaired["repairs"]}
    assert [row for row in newest["messages"] if row.get("history_id") in again] == repaired["repairs"]
    assert next(row for row in repaired["repairs"] if row.get("msg_type") == "quiz")["quiz"] == question["quiz"]


def _as_newest_shows(root, read):
    """Every stored row ``read`` carries (sent again or arrived) is exactly the row a fresh newest page shows."""
    _status, newest = _read(root)
    shown = {row.get("history_id"): row for row in newest["messages"] if row.get("history_id")}
    for row in [*read.get("repairs", ()), *read["messages"]]:
        if row.get("history_id"):
            assert shown.get(row["history_id"]) == row, row["history_id"]


def test_rows_sent_again_keep_their_stored_order_before_what_arrived(tmp_path):
    first, second = {**QUESTION, "state": "answered", "answered_index": 0}, {**QUESTION, "state": "answered", "answered_index": 1}
    _task(tmp_path, "asker", "running", 0, owner_quiz={"q1": first})
    answer = {"chat_id": 1, "task_id": "asker", "direction": "system", "type": "quiz_answer", "source": "owner_quiz_answer"}
    ds.append(tmp_path, "chat", {"ts": TS.format(0), "chat_id": 1, "task_id": "asker", "type": "quiz", "direction": "out",
                                 "text": "Which?", "quiz": {**QUESTION, "state": "open"}})
    ds.append(tmp_path, "chat", {**answer, "ts": TS.format(1), "text": "a", "quiz": first})
    _, page = _read(tmp_path)
    ds.append(tmp_path, "chat", {**answer, "ts": TS.format(1), "text": "b", "quiz": second})  # a later answer, same time
    _task(tmp_path, "asker", "completed", 0, owner_quiz={"q1": second})
    status, read = _read(tmp_path, page["after_cursor"])
    assert status == 200 and read["repairs"]
    _as_newest_shows(tmp_path, read)


def test_words_revised_in_place_send_a_settling_window_to_its_newest_page(tmp_path):
    _asking(tmp_path)
    ds.append(tmp_path, "chat", {"ts": TS.format(1), "chat_id": 1, "direction": "out", "text": "draft caption"})
    _, page = _read(tmp_path)
    caption = next(row for row in page["messages"] if row.get("text") == "draft caption")
    ds.revise_text(tmp_path, "chat", caption["history_position"]["offset"], "corrected caption")
    status, quiet = _read(tmp_path, page["after_cursor"])
    assert status == 200 and "held" not in _cursor(quiet["after_cursor"])  # nothing to refuse yet; nothing held
    _task(tmp_path, "asker", "completed", 0)
    status, refused = _read(tmp_path, quiet["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")
    _, fresh = _read(tmp_path)  # the newest page names the revision it saw: settlements are answered again
    _task(tmp_path, "asker", "completed", 0, title="changed after it settled")
    status, repaired = _read(tmp_path, fresh["after_cursor"])
    assert status == 200 and repaired["repairs"]


def test_a_child_and_its_parent_are_never_held(tmp_path):
    _task(tmp_path, "parent", "running", 0)
    _task(tmp_path, "child", "running", 1, parent_task_id="parent", root_task_id="parent")
    ds.append(tmp_path, "progress", {"ts": TS.format(0), "chat_id": 1, "task_id": "parent", "content": "plan"})
    ds.append(tmp_path, "chat", {"ts": TS.format(1), "chat_id": 1, "task_id": "child", "parent_task_id": "parent",
                                 "root_task_id": "parent", "delegation_role": "subagent", "direction": "out",
                                 "text": "child answer"})
    _asking(tmp_path, 2)
    _, page = _read(tmp_path)
    assert {"parent", "child"} <= set(_cursor(page["after_cursor"])["tasks"])
    assert set(_cursor(page["after_cursor"])["held"]) == {"asker"}  # the newest page places children: only it
    _task(tmp_path, "child", "completed", 1, parent_task_id="parent", root_task_id="parent")
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")


def test_a_repair_whose_projection_failed_part_way_refuses(tmp_path, monkeypatch):
    import ouroboros.tool_call_log as tool_call_log

    _asking(tmp_path)
    _, page = _read(tmp_path)
    _task(tmp_path, "asker", "completed", 0)
    original, fail = tool_call_log.replay_evidence_for_tasks, {"on": True}

    def evidence(*args, **kwargs):
        if fail["on"]:
            raise OSError("evidence unreadable")
        return original(*args, **kwargs)
    monkeypatch.setattr(tool_call_log, "replay_evidence_for_tasks", evidence)
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")
    fail["on"] = False
    status, repaired = _read(tmp_path, page["after_cursor"])
    assert status == 200 and repaired["repairs"]


def test_a_task_that_turns_into_a_retry_while_it_settles_refuses(tmp_path):
    _asking(tmp_path)
    _, page = _read(tmp_path)
    _task(tmp_path, "retry", "running", 1)
    _task(tmp_path, "asker", "failed", 0, retry_task_id="retry")  # its row now shows its successor's truth
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")


def test_a_window_that_stops_holding_keeps_the_versions_its_rows_showed(tmp_path, monkeypatch):
    import supervisor.message_ingress as ingress

    _asking(tmp_path)
    _, page = _read(tmp_path)
    seen = _cursor(page["after_cursor"])["tasks"]["asker"]
    monkeypatch.setattr(ingress, "delivery_facts", lambda row, chat: {"ingress_pending": True})
    ds.append(tmp_path, "chat", {"ts": TS.format(3), "direction": "in", "chat_id": 1, "text": "do it",
                                 "client_message_id": "c1", "ingress_accepted": True})
    _task(tmp_path, "asker", "running", 0, title="later")
    ds.append(tmp_path, "progress", {"ts": TS.format(4), "chat_id": 1, "task_id": "asker", "content": "more"})
    status, going = _read(tmp_path, page["after_cursor"])
    cursor = _cursor(going["after_cursor"])
    assert status == 200 and "held" not in cursor and cursor["tasks"]["asker"] == seen
    _task(tmp_path, "asker", "completed", 0)
    status, refused = _read(tmp_path, going["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")  # still owed: the window's old rows never saw it


def test_a_settled_tasks_carrier_comes_with_its_rows_not_as_an_arrival(tmp_path, monkeypatch):
    import ouroboros.tool_call_log as tool_call_log

    _task(tmp_path, "talker", "running", 0)
    ds.append(tmp_path, "chat", {"ts": TS.format(0), "chat_id": 1, "task_id": "talker", "direction": "out", "text": "hi"})
    original = tool_call_log.replay_evidence_for_tasks
    monkeypatch.setattr(tool_call_log, "replay_evidence_for_tasks", lambda root, tasks: {
        **original(root, tasks), **({"talker": {"observations": [{"tool": "x"}], "coverage": {"shown": 1, "matched": 1}}}
                                   if "talker" in tasks else {})})
    _, page = _read(tmp_path)
    assert any(row.get("system_type") == "task_evidence" for row in page["messages"])
    _task(tmp_path, "talker", "completed", 0)
    status, read = _read(tmp_path, page["after_cursor"])
    assert status == 200 and read["messages"] == []
    assert {row.get("system_type") or "speech" for row in read["repairs"]} == {"task_evidence", "speech"}


def test_a_routing_receipt_without_a_row_address_holds_nothing():
    page = {"lens": {"chat": history_paging.RoomLens((("appearance", "room:1"),))}, "after": False,
            "selections": {"chat": ([], 0, set()), "progress": ([], 0, set())}}
    owner = {"role": "user", "task_id": "", "client_message_id": "c1"}
    plain = [_row("plain", 1)]
    weak = [*plain, {**owner, "chat_annotation": {"status": "delivered"}}]
    placed = [*plain, {**owner, "chat_annotation": {"status": "delivered", "source_row_address": "row:2@t#x"}}]
    assert history_paging._held_rows(page, weak, {"plain"}, _Reads()) is None
    assert history_paging._held_rows(page, placed, {"plain"}, _Reads()) == {"plain": [[1], []]}


def test_a_broken_packed_cursor_is_invalid_and_none_is_issued_past_its_inflate_bound(tmp_path, monkeypatch):
    _asking(tmp_path)
    assert _read(tmp_path, "zAAAA")[0] == 400  # not deflate data: 400, never a server error
    _, plain = _read(tmp_path, repair=False)
    monkeypatch.setattr(history_paging, "_UNPACKED_CURSOR_BYTES", 64)  # it could not read back what it packed
    status, page = _read(tmp_path)
    assert status == 200 and page["after_cursor"] == plain["after_cursor"]


def test_a_read_receipt_placed_only_by_its_client_id_holds_nothing(tmp_path):
    """An older receipt (written before receipts named their row) is placed by finding its client id among the
    dialogue's rows; a stamped read is no different: that placement is the dialogue's, not the journal's."""
    from ouroboros.chat_annotations import append_chat_annotation

    _asking(tmp_path)
    ds.append(tmp_path, "chat", {"ts": TS.format(3), "direction": "in", "chat_id": 1, "text": "faster please",
                                 "client_message_id": "c1"})
    append_chat_annotation(tmp_path, "c1", action="steer_task", target="asker", status="read", source_chat_id=1)
    _, page = _read(tmp_path)
    owner = next(row for row in page["messages"] if row.get("client_message_id") == "c1")
    assert owner["chat_annotation"] == {"source_row_address": owner["row_address"], "action": "steer_task",
                                        "target": "asker", "status": "read"}  # the wire receipt as before
    assert "held" not in _cursor(page["after_cursor"])
    _task(tmp_path, "asker", "completed", 0)
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")


@pytest.mark.parametrize("meanwhile", ["settlements_read", "projected"])
def test_words_revised_while_a_settlement_is_answered_send_the_window_to_its_newest_page(tmp_path, monkeypatch, meanwhile):
    """The revision is taken before the settlements are read; one that moves before the answer leaves (while they are
    read, or once the page is projected) is a change of words the repaired rows do not carry: the read refuses as the
    settlement alone once did, and the newest page shows them."""
    from ouroboros.gateway import place_history

    _asking(tmp_path)
    ds.append(tmp_path, "chat", {"ts": TS.format(1), "chat_id": 1, "direction": "out", "text": "draft caption"})
    _, page = _read(tmp_path)
    caption = next(row for row in page["messages"] if row.get("text") == "draft caption")
    _task(tmp_path, "asker", "completed", 0)
    owner, name = (history_paging, "_settled_since") if meanwhile == "settlements_read" else (
        place_history, "annotate_place_history")  # the last step of the projection, before the cursor is named
    original = getattr(owner, name)

    def revised_meanwhile(*args, **kwargs):
        ds.revise_text(tmp_path, "chat", caption["history_position"]["offset"], "corrected caption")
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, name, revised_meanwhile)
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused.get("detail")) == (409, "task_settled"), refused
    monkeypatch.setattr(owner, name, original)
    _, newest = _read(tmp_path)
    assert "corrected caption" in [row.get("text") for row in newest["messages"]]


def test_a_newest_page_whose_projection_failed_part_way_names_no_arrivals_cursor(tmp_path, monkeypatch):
    """A newest page that lost part of its task projection is no window to continue: without ``after_cursor`` the
    client reads the newest page again (CLIENT_CONTRACT "New arrivals"), which projects it whole once it can."""
    import ouroboros.tool_call_log as tool_call_log

    _asking(tmp_path)
    _task(tmp_path, "done", "completed", 1)
    ds.append(tmp_path, "chat", {"ts": TS.format(1), "direction": "out", "chat_id": 1, "task_id": "done", "text": "ok"})
    original, fail = tool_call_log.replay_evidence_for_tasks, {"on": True}

    def evidence(*args, **kwargs):
        if fail["on"]:
            raise OSError("evidence unreadable")
        return original(*args, **kwargs)
    monkeypatch.setattr(tool_call_log, "replay_evidence_for_tasks", evidence)
    status, partial = _read(tmp_path)
    assert status == 200 and partial["messages"] and partial["page_cursor"]
    assert "after_cursor" not in partial  # nothing to continue from: the next read of arrivals is a newest page
    # The window says what it lacks (a client owes the room a whole read again: ``*_projection_failed``).
    assert {"chat_projection_failed", "progress_projection_failed"} <= set(partial["window"]["truncated_by"])
    assert partial["window"]["complete"] is False
    assert all("projection_failed" in span["gaps"] for span in partial["coverage"]["spans"].values())
    fail["on"] = False
    status, whole = _read(tmp_path)
    assert status == 200 and whole["after_cursor"] and not any("projection" in cause for cause in whole["window"]["truncated_by"])
    done = next(row for row in whole["messages"] if row.get("task_id") == "done")
    assert done.get("task_terminal_status") == "completed" != next(
        row for row in partial["messages"] if row.get("task_id") == "done").get("task_terminal_status")


def _shown_by_a_notice_and_its_current_work(root, monkeypatch):
    """A running task this room shows only through a notice fixed to it (a steer not delivered) and the carrier of
    its current work; the notice shows none of its facts, and once it settles the carrier is gone."""
    import ouroboros.tool_call_log as tool_call_log

    _asking(root)
    _task(root, "runner", "running", 1)
    ds.append(root, "chat", {"ts": TS.format(1), "chat_id": 1, "direction": "system", "type": "steer_not_delivered",
                             "task_id": "runner", "text": "Runner · not delivered"})
    live = {"on": True}
    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *_, **__: [
        {"activity_id": "runner", "chat_id": 1, "kind": "managed_task", "phase": "working"}] if live["on"] else [])
    original = tool_call_log.replay_evidence_for_tasks
    monkeypatch.setattr(tool_call_log, "replay_evidence_for_tasks", lambda root, tasks: {
        **original(root, tasks), **({"runner": {"observations": [{"tool": "x"}], "coverage": {"shown": 1, "matched": 1}}}
                                   if "runner" in tasks else {})})
    _, page = _read(root)
    rows = [row.get("system_type") for row in page["messages"] if row.get("task_id") == "runner"]
    assert rows == ["steer_not_delivered", "task_evidence"]
    live["on"] = False
    _task(root, "runner", "completed", 1)
    return page


def test_a_task_shown_only_by_a_fixed_notice_and_its_work_is_never_held(tmp_path, monkeypatch):
    page = _shown_by_a_notice_and_its_current_work(tmp_path, monkeypatch)
    cursor = _cursor(page["after_cursor"])
    assert "runner" in cursor["tasks"] and "runner" not in cursor["held"] and "asker" in cursor["held"]
    status, refused = _read(tmp_path, page["after_cursor"])
    assert (status, refused["detail"]) == (409, "task_settled")  # as before the repairs: its newest page
    _, newest = _read(tmp_path)
    assert "runner" not in _cursor(newest["after_cursor"])["tasks"]  # its notice alone names no version to watch


def test_a_repair_that_leaves_its_task_unshown_refuses_instead_of_repeating(tmp_path, monkeypatch):
    """Rows sent again that show none of their task's facts would carry its old version on: the same repair on every
    read. A window that held such rows (a cursor naming them, as an older server's did) takes its newest page."""
    page = _shown_by_a_notice_and_its_current_work(tmp_path, monkeypatch)
    cursor = _cursor(page["after_cursor"])
    notice = next(row for row in page["messages"] if row.get("system_type") == "steer_not_delivered")
    cursor["held"]["runner"] = [[notice["history_position"]["offset"]], []]
    status, refused = _read(tmp_path, history_paging.encode_cursor(cursor, packed=True))
    assert (status, refused.get("detail")) == (409, "task_settled"), refused.get("repairs")
