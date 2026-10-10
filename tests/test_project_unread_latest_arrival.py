"""A Project room is read at the message that ARRIVED last, not the newest ``ts``.

The history window is ordered and tailed by event time, while the unread
revision advances when a standalone message arrives. A terminal answer the
outbox delivers late keeps its original ``ts``: the event-time tail can leave
it out of the recent window entirely, or sort it above messages that arrived
before it. A recent Project read therefore names the physically newest
standalone message it read (``window.latest_message``) without moving or adding
a row, and names nothing it cannot vouch for (DESIGN "Project unread dot").

Deliveries run the real ``events_chat_delivery`` seam into a real registered
Project, so every counted row is one the sidebar revision counted; reads run the
real history endpoint.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros.gateway import history, history_paging
from ouroboros.projects_registry import create_project, get_project
from ouroboros.utils import append_jsonl
from supervisor import events_chat_delivery as delivery
from supervisor import message_bus

CHILD = {"id": "kid-1", "delegation_role": "subagent", "parent_task_id": "root-1",
         "root_task_id": "root-1", "subagent_role": "researcher"}


def ts(minute: int) -> str:
    return f"2026-09-28T12:{minute:02d}:00.000000+00:00"


@pytest.fixture
def room(tmp_path, monkeypatch):
    from supervisor import queue, state

    for module, names in ((state, ("DRIVE_ROOT", "STATE_PATH", "STATE_LAST_GOOD_PATH", "STATE_LOCK_PATH")),
                          (queue, ("DRIVE_ROOT", "QUEUE_SNAPSHOT_PATH"))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    state.init(tmp_path)
    queue.init(tmp_path)
    monkeypatch.setattr(history, "_active_lifecycle_row", lambda _filter: None)
    project = create_project(tmp_path, "racer")
    chat_id = int(project["chat_id"])
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = lambda _frame: None
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7, "session_id": "s"})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    host = SimpleNamespace(
        DRIVE_ROOT=tmp_path, append_jsonl=append_jsonl, bridge=bridge,
        send_with_budget=message_bus.send_with_budget,
        RUNNING={"root-1": {"task": {"id": "root-1"}}, "kid-1": {"task": dict(CHILD)}},
    )
    chat_log = tmp_path / "logs" / "chat.jsonl"

    def deliver(text: str, minute: int, task_id: str = "root-1", **fields) -> None:
        # A replayed outbox row carries its original ``ts`` exactly like this one.
        delivery._handle_send_message({"type": "send_message", "chat_id": chat_id, "task_id": task_id,
                                       "text": text, "ts": ts(minute), **fields}, host)

    def read(**params) -> dict:
        response = asyncio.run(history.make_chat_history_endpoint(tmp_path)(
            SimpleNamespace(query_params={"chat_id": str(chat_id), **params})))
        assert response.status_code == 200, response.body
        return json.loads(response.body)

    def row_id(text: str) -> str:
        """The physical identity of the one stored row carrying ``text``."""
        offset = 0
        for line in chat_log.read_bytes().splitlines(keepends=True):
            try:
                if json.loads(line).get("text") == text:
                    return f"chat:{offset}"
            except ValueError:
                pass
            offset += len(line)
        raise AssertionError(f"no stored row {text!r}")

    return SimpleNamespace(
        chat_id=chat_id, deliver=deliver, read=read, row_id=row_id, chat_log=chat_log, host=host, root=tmp_path,
        revision=lambda: int(get_project(tmp_path, "racer")["visible_revision"]),
    )


def texts(payload: dict) -> list[str]:
    return [row["text"] for row in payload["messages"] if not row.get("is_progress") and row.get("text")]


def test_a_late_answer_below_the_window_is_named_not_moved(room):
    for minute in (10, 11, 12):
        room.deliver(f"reply {minute}", minute)
    room.deliver("late final", 1)
    assert room.revision() == 4, "the late answer is a new conversation message"

    recent = room.read(n_human="3")
    assert texts(recent) == ["reply 10", "reply 11", "reply 12"], "the event-time window is unchanged"
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("late final"), "out_of_order": True}
    older = room.read(cursor=recent["next_cursor"])
    assert "late final" in texts(older), "the first older page shows the named message"
    assert "latest_message" not in older["window"], "only a recent read names the newest arrival"


def test_ordinary_truncation_names_the_bottom_message(room):
    """Old rows leaving the window is no reason to withhold the newest message."""
    for minute in range(1, 6):
        room.deliver(f"reply {minute}", minute)
    recent = room.read(n_human="2")
    assert "quota" in recent["window"]["truncated_by"]
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("reply 5"), "out_of_order": False}


def test_a_late_answer_inside_the_window_sorts_above_messages_that_arrived_first(room):
    room.deliver("reply 5", 5)
    message_bus.log_chat("in", room.chat_id, 7, "owner asks", ts=ts(9))
    room.deliver("late final", 3)
    recent = room.read()
    assert texts(recent) == ["late final", "reply 5", "owner asks"], "placement stays chronological"
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("late final"), "out_of_order": True}


def test_card_content_and_progress_after_the_answer_are_never_the_newest_message(room):
    room.deliver("root answer", 5)
    before = room.revision()
    room.deliver("child final", 6, task_id="kid-1")
    room.deliver("custody", 7, role="system", system_type="custody_notice", progress_meta={
        "card_row": "timeline", "card_row_id": "final:root-1:x:custody_notice"})
    room.deliver("late review", 1, role="system", system_type="acceptance_late_settlement",
                 progress_meta={"card_row": "reviews", "card_row_id": "acceptance-late:k"})
    room.deliver("narration", 8, is_progress=True)
    assert room.revision() == before, "none of these is a conversation message"
    assert room.read()["window"]["latest_message"] == {
        "history_id": room.row_id("root answer"), "out_of_order": False}


def salvage_receipt(room, task: dict, minute: int) -> str:
    """Deliver the real host-salvage ``terminal_incident`` receipt; answer its text."""
    from supervisor.terminal_delivery import project_terminal_result_event

    event = project_terminal_result_event(
        room.root, task, task["id"], result_text=f"raw output of {task['id']}", terminal_origin="host_salvage",
        provider_notice=f"{task['id']} stopped by an outage.",
        base_event={"type": "send_message", "chat_id": room.chat_id, "task_id": task["id"],
                    "text": f"raw output of {task['id']}", "ts": ts(minute)})
    assert event["system_type"] == "terminal_incident"
    delivery._handle_send_message(event, room.host)
    return event["text"]


def test_a_root_terminal_incident_is_a_standalone_message_and_the_newest_arrival(room):
    room.deliver("root answer", 5)
    before = room.revision()
    receipt = salvage_receipt(room, {"id": "root-1"}, 6)
    assert room.revision() == before + 1
    recent = room.read()
    (row,) = [row for row in recent["messages"] if row.get("system_type") == "terminal_incident"]
    assert row["role"] == "system" and not row.get("card_row"), "shown alone in the feed"
    assert recent["window"]["latest_message"] == {"history_id": room.row_id(receipt), "out_of_order": False}


def test_a_childs_terminal_incident_and_a_progress_incident_are_never_the_newest_message(room):
    room.deliver("root answer", 5)
    before = room.revision()
    salvage_receipt(room, CHILD, 6)
    room.deliver("worker lost", 7, is_progress=True, progress_meta={
        "task_incident": "worker_lost", "toast_once": "root-1:worker_lost"})
    assert room.revision() == before, "both change a card, not the conversation"
    assert room.read()["window"]["latest_message"] == {
        "history_id": room.row_id("root answer"), "out_of_order": False}


def test_an_unreadable_row_after_the_newest_readable_one_leaves_its_arrival_unknown(room):
    room.deliver("reply 1", 1)
    with room.chat_log.open("ab") as stream:
        stream.write(b'{"direction": "out", "text": "torn\n')
    recent = room.read()
    assert "chat_malformed_jsonl" in recent["window"]["truncated_by"]
    assert recent["window"]["latest_message"] is None
    room.deliver("reply 2", 2)
    assert room.read()["window"]["latest_message"] == {"history_id": room.row_id("reply 2"), "out_of_order": False}, \
        "an unreadable row before the newest message does not hide it"


def test_a_projection_that_fails_part_way_leaves_the_arrival_unknown_and_discloses_the_gap(room, monkeypatch):
    """Every row after a projection fault is missing: the last one projected is not the newest arrival."""
    room.deliver("reply 1", 1)
    room.deliver("reply 2", 2)
    copy = history._copy_task_summary_metadata

    def fault(rec, entry):
        if entry.get("text") == "reply 2":
            raise RuntimeError("injected projection fault")
        return copy(rec, entry)

    monkeypatch.setattr(history, "_copy_task_summary_metadata", fault)
    partial = room.read()
    assert texts(partial) == ["reply 1"], "the rows after the fault are not in the response"
    assert "chat_projection_failed" in partial["window"]["truncated_by"]
    assert partial["coverage"]["spans"]["chat"]["gaps"] == ["projection_failed"], "the span is not clean"
    assert partial["window"]["latest_message"] is None, "reply 1 is not the message that arrived last"
    monkeypatch.setattr(history, "_copy_task_summary_metadata", copy)
    healed = room.read()
    assert "chat_projection_failed" not in healed["window"]["truncated_by"]
    assert healed["window"]["latest_message"] == {"history_id": room.row_id("reply 2"), "out_of_order": False}


def test_a_live_line_its_writer_has_not_finished_leaves_the_arrival_unknown_until_it_is(room):
    """The read freezes before the unfinished line, which may be the newest message."""
    room.deliver("reply 1", 1)
    clean = room.read()
    assert "chat_incomplete_live_line" not in clean["window"]["truncated_by"], "a clean end of the chat is no gap"
    assert clean["window"]["latest_message"] == {"history_id": room.row_id("reply 1"), "out_of_order": False}

    line = json.dumps({"ts": ts(2), "direction": "out", "chat_id": room.chat_id, "text": "reply 2"}).encode() + b"\n"
    with room.chat_log.open("ab") as stream:
        stream.write(line[:-6])
    unfinished = room.read()
    assert texts(unfinished) == ["reply 1"], "the page holds only complete rows"
    assert "chat_incomplete_live_line" in unfinished["window"]["truncated_by"]
    assert unfinished["window"]["latest_message"] is None, "the previous message is not the one that arrived last"
    frozen = room.chat_log.stat().st_size - len(line[:-6])
    span = unfinished["coverage"]["spans"]["chat"]
    assert span["to"] == unfinished["coverage"]["upper"]["chat"] == frozen and span["gaps"] == [], \
        "History coverage: every byte up to the frozen boundary was delivered; the unfinished line lies after it"

    with room.chat_log.open("ab") as stream:
        stream.write(line[-6:])
    finished = room.read()
    assert "chat_incomplete_live_line" not in finished["window"]["truncated_by"]
    assert finished["window"]["latest_message"] == {"history_id": room.row_id("reply 2"), "out_of_order": False}


def test_a_frozen_page_keeps_the_unfinished_line_it_froze_before_and_never_names_an_arrival(room):
    """A page replay reads the recent read's frozen boundary: the unfinished line beyond it stays a
    disclosed gap, and what arrived after it cannot be seen, so the replay names no older reply."""
    room.deliver("reply 1", 1)
    line = json.dumps({"ts": ts(2), "direction": "out", "chat_id": room.chat_id, "text": "reply 2"}).encode() + b"\n"
    with room.chat_log.open("ab") as stream:
        stream.write(line[:-6])
    unfinished = room.read()
    assert unfinished["window"]["latest_message"] is None
    with room.chat_log.open("ab") as stream:
        stream.write(line[-6:])

    replayed = room.read(cursor=unfinished["page_cursor"])
    assert texts(replayed) == ["reply 1"], "the frozen page is unchanged"
    assert "chat_incomplete_live_line" in replayed["window"]["truncated_by"], "the gap it froze before is kept"
    assert replayed["coverage"]["spans"]["chat"]["gaps"] == [], "and its span stays clean"
    assert replayed["window"]["latest_message"] is None, "reply 1 is not the message that arrived last"

    clean = room.read()
    assert clean["window"]["latest_message"] == {"history_id": room.row_id("reply 2"), "out_of_order": False}
    room.deliver("reply 3", 3)
    stale = room.read(cursor=clean["page_cursor"])
    assert "chat_incomplete_live_line" not in stale["window"]["truncated_by"], "a clean boundary has no gap"
    assert stale["window"]["latest_message"] is None, "reply 3 arrived beyond the boundary this replay reads"


def test_an_unreadable_chat_source_leaves_the_arrival_unknown(room, monkeypatch):
    room.deliver("reply 1", 1)
    readable = history_paging.HistorySource

    def refuse_chat(path, source, *frozen):
        if source == "chat":
            raise PermissionError("chat.jsonl")
        return readable(path, source, *frozen)

    monkeypatch.setattr(history_paging, "HistorySource", refuse_chat)
    window = room.read()["window"]
    assert "chat_source_unavailable" in window["truncated_by"]
    assert window["latest_message"] is None


def test_main_names_nothing(room):
    message_bus.send_with_budget(1, "a Main reply", task_id="root-1")
    response = asyncio.run(history.make_chat_history_endpoint(message_bus.DATA_DIR)(
        SimpleNamespace(query_params={})))
    assert "latest_message" not in json.loads(response.body)["window"]


def test_a_room_read_while_the_project_registry_is_unreadable_names_no_arrival_and_main_reads_on(room):
    """Unclassified, the room is read through Main's lens: its arrival is unknown, never absent
    (which the bottom would decide). Main is known without the registry and reads as before."""
    room.deliver("root answer", 5)
    registry = room.root / "state" / "projects.json"
    intact = registry.read_bytes()
    registry.write_bytes(intact[: len(intact) // 2])
    try:
        assert room.read()["window"]["latest_message"] is None
        main = asyncio.run(history.make_chat_history_endpoint(room.root)(SimpleNamespace(query_params={})))
        assert main.status_code == 200 and "latest_message" not in json.loads(main.body)["window"]
    finally:
        registry.write_bytes(intact)
    assert room.read()["window"]["latest_message"] == {"history_id": room.row_id("root answer"), "out_of_order": False}


def test_one_malformed_registry_row_leaves_only_the_room_it_may_hold_unknown(room):
    """A row the strict read refuses cannot unclassify the rooms the readable rows name."""
    room.deliver("root answer", 5)
    registry = room.root / "state" / "projects.json"
    data = json.loads(registry.read_text(encoding="utf-8"))
    registry.write_text(json.dumps({**data, "projects": [*data["projects"], {"chat_id": room.chat_id + 1}]}),
                        encoding="utf-8")
    recent = room.read()
    assert "root answer" in texts(recent)
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("root answer"), "out_of_order": False}
    other = asyncio.run(history.make_chat_history_endpoint(room.root)(
        SimpleNamespace(query_params={"chat_id": str(room.chat_id + 1)})))
    assert json.loads(other.body)["window"]["latest_message"] is None, "the malformed row's room is unknown"


@pytest.fixture
def narrow_recent(monkeypatch):
    """A recent read that stops as soon as its quota is met (a large chat's window)."""
    monkeypatch.setattr(history_paging, "_TAIL_WINDOW_START_BYTES", 64)


def child_words(room, count: int, minute: int = 20) -> None:
    for index in range(count):
        room.deliver(f"child note {index}", minute + index, task_id="kid-1")


def test_a_message_before_a_recent_quota_of_card_rows_is_named_from_one_bounded_read(room, narrow_recent):
    """A child's words and the owner's own messages fill the quota; the answer before them is still the newest."""
    room.deliver("root answer", 5)
    child_words(room, 6)
    message_bus.log_chat("in", room.chat_id, 7, "owner asks", ts=ts(40))
    assert room.revision() == 1, "only the root answer is a conversation message"
    recent = room.read(n_human="2")
    assert "root answer" not in texts(recent), "the answer lies before the recent selection"
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("root answer"), "out_of_order": True}, \
        "named, and read only on screen: this read cannot place it relative to the bottom"


def test_a_room_whose_chat_holds_no_standalone_message_names_nothing(room, narrow_recent):
    child_words(room, 6)
    message_bus.log_chat("in", room.chat_id, 7, "owner asks", ts=ts(40))
    recent = room.read(n_human="2")
    assert recent["has_more"], "the recent selection did not reach the start of the chat"
    assert "latest_message" not in recent["window"], "proven: no standalone message exists to read"


def test_a_bound_reached_before_the_newest_message_leaves_its_arrival_unknown(room, narrow_recent, monkeypatch):
    room.deliver("root answer", 5)
    child_words(room, 8)
    monkeypatch.setattr(history_paging, "_READ_CEILING_BYTES", 1)
    assert room.read(n_human="2")["window"]["latest_message"] is None


def test_an_unreadable_row_between_the_newest_message_and_the_recent_read_leaves_it_unknown(room, narrow_recent):
    room.deliver("root answer", 5)
    with room.chat_log.open("ab") as stream:
        stream.write(b'{"direction": "out", "text": "torn\n')
    child_words(room, 6)
    assert room.read(n_human="2")["window"]["latest_message"] is None


def test_a_skill_review_after_a_late_answer_does_not_stand_in_for_it(room):
    """Skill reviews are appended beside the conversation and never count as unread."""
    room.deliver("reply 5", 5)
    room.deliver("late final", 1)
    append_jsonl(room.chat_log, {"ts": ts(9), "direction": "system", "type": "skill_review", "chat_id": room.chat_id,
                                 "skill": "weather", "status": "pass", "text": "Skill review: `weather` — status=pass"})
    assert room.revision() == 2
    assert room.read()["window"]["latest_message"] == {
        "history_id": room.row_id("late final"), "out_of_order": True}


def legacy_child_final(room, text: str, minute: int, task_id: str = "kid-old") -> None:
    """A child's final written before chat rows carried lineage: only its task result names the child."""
    from ouroboros.task_results import write_task_result

    write_task_result(room.root, task_id, "completed", delegation_role="subagent", parent_task_id="root-1",
                      root_task_id="root-1", role="researcher")
    append_jsonl(room.chat_log, {"ts": ts(minute), "direction": "out", "chat_id": room.chat_id,
                                 "text": text, "task_id": task_id})


def test_a_legacy_child_final_its_task_result_names_is_never_the_newest_message(room):
    """History recovers the child's lineage from its task result and shows the final in the child's card."""
    room.deliver("root answer", 5)
    legacy_child_final(room, "legacy child final", 6)
    recent = room.read()
    (child,) = [row for row in recent["messages"] if row.get("text") == "legacy child final"]
    assert child["delegation_role"] == "subagent" and child["parent_task_id"] == "root-1", "shown in the child's card"
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("root answer"), "out_of_order": False}


@pytest.mark.parametrize("old_result", [False, True], ids=["missing-result", "pre-upgrade-result"])
def test_a_legacy_child_final_uses_the_progress_lineage_the_client_reads(room, old_result):
    """A pre-upgrade result may be quarantined; the page still knows the child's parent."""
    room.deliver("root answer", 5)
    lineage = {"delegation_role": "subagent", "parent_task_id": "root-1",
               "root_task_id": "root-1", "subagent_task_id": "kid-old"}
    append_jsonl(room.chat_log, {"ts": ts(6), "direction": "out", "chat_id": room.chat_id,
                                 "text": "legacy child final", "task_id": "kid-old"})
    append_jsonl(room.root / "logs" / "progress.jsonl", {
        "ts": ts(4), "chat_id": room.chat_id, "task_id": "root-1", "content": "Child scheduled",
        "subagent_event": "scheduled", **lineage})
    if old_result:
        path = room.root / "task_results" / "kid-old.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": "kid-old", "status": "completed", **lineage}), encoding="utf-8")
    recent = room.read()
    assert any(row.get("subagent_task_id") == "kid-old" for row in recent["messages"])
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("root answer"), "out_of_order": False}
    if old_result:
        assert not path.exists(), "the unstamped result was quarantined, not used as current lineage"
    # The scheduling row's task_id names the parent, and a delivery from that
    # same child still stands alone. Neither may be mistaken for child speech.
    assert room.host.bridge.send_photo(room.chat_id, b"\x89PNG\r\n\x1a\n", caption="child photo", task_id="kid-old")[0]
    recent = room.read()
    photo = next(row for row in recent["messages"] if row.get("system_type") == "photo")
    assert recent["window"]["latest_message"] == {"history_id": photo["history_id"], "out_of_order": False}


def durable_child(room, task_id: str = "kid-1") -> None:
    """The child's lineage as its task result persists it, recovered for every row the child wrote."""
    from ouroboros.task_results import write_task_result

    write_task_result(room.root, task_id, "running", delegation_role="subagent", parent_task_id="root-1",
                      root_task_id="root-1", role="researcher")


def test_a_photo_a_child_delivers_is_shown_alone_and_is_the_newest_arrival(room):
    """A child's words are card content; what it delivers (a photo, a file, links, a question) is
    its own bubble in the conversation, which the producer counts."""
    durable_child(room)
    room.deliver("root answer", 5)
    room.deliver("child note", 6, task_id="kid-1")
    before = room.revision()
    assert room.host.bridge.send_photo(room.chat_id, b"\x89PNG\r\n\x1a\n", caption="child photo", task_id="kid-1")[0]
    assert room.revision() == before + 1, "the producer counts the child's photo"
    recent = room.read()
    (photo,) = [row for row in recent["messages"] if row.get("system_type") == "photo"]
    assert photo["delegation_role"] == "subagent", "its lineage is known, and it is still shown alone"
    assert recent["window"]["latest_message"] == {"history_id": photo["history_id"], "out_of_order": False}, \
        "the root answer on screen is not the photo that arrived after it"


def test_a_question_a_child_asked_before_the_recent_selection_is_named_by_the_bounded_search(room, narrow_recent):
    durable_child(room)
    room.deliver("root answer", 5)
    assert room.host.bridge.send_quiz(room.chat_id, "q-kid", "Ship it?", ["Yes", "No"],
                                      assumption="I ship after review", task_id="kid-1")[0]
    child_words(room, 6)
    message_bus.log_chat("in", room.chat_id, 7, "owner asks", ts=ts(40))
    recent = room.read(n_human="2")
    assert "Ship it?" not in texts(recent), "the question lies before the recent selection"
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("Ship it?"), "out_of_order": True}


def test_a_legacy_child_final_before_the_recent_selection_does_not_stand_in_for_the_root_answer(room, narrow_recent):
    room.deliver("root answer", 5)
    legacy_child_final(room, "legacy child final", 6)
    child_words(room, 6)
    message_bus.log_chat("in", room.chat_id, 7, "owner asks", ts=ts(40))
    recent = room.read(n_human="2")
    assert "legacy child final" not in texts(recent), "both lie before the recent selection"
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("root answer"), "out_of_order": True}


def follow(room, cursor):
    """Every older page from ``cursor`` down to the start of the chat."""
    pages = []
    while cursor:
        pages.append(room.read(cursor=cursor))
        cursor = pages[-1]["next_cursor"]
    return pages


def test_older_pages_carry_a_bounded_search_on_to_the_newest_message(room, narrow_recent, monkeypatch):
    """The recent read's bound came first. Its older pages continue the search: the one holding the
    newest standalone message names it, relative to the chain's frozen boundary; none other does."""
    room.deliver("first answer", 1)
    room.deliver("root answer", 5)
    child_words(room, 8)
    message_bus.log_chat("in", room.chat_id, 7, "owner asks", ts=ts(40))
    monkeypatch.setattr(history_paging, "_READ_CEILING_BYTES", 1)
    recent = room.read(n_human="2")
    answer = room.row_id("root answer")
    assert recent["window"]["latest_message"] is None, "unknown: the bound came before the answer"
    assert int(answer.split(":")[1]) < recent["window"]["latest_before"] < room.chat_log.stat().st_size, \
        "but searched: nothing newer lies at or after latest_before"
    replay = room.read(cursor=recent["page_cursor"])
    assert replay["window"]["latest_message"] is None and "latest_before" not in replay["window"]

    pages = follow(room, recent["next_cursor"])
    named = [page["window"].get("latest_message", "nothing") for page in pages]
    holding = next(page for page in pages if page["window"].get("latest_message"))
    assert [name for name in named if name != "nothing"] == [{"history_id": answer, "out_of_order": True}], \
        ("the pages above it hold no standalone message, and the pages below it name nothing", named)
    assert named[0] == "nothing" and named[-1] == "nothing", named
    assert "root answer" in texts(holding) and holding["coverage"]["upper"] == recent["coverage"]["upper"]
    assert room.read(cursor=holding["page_cursor"])["window"]["latest_message"] == {
        "history_id": answer, "out_of_order": True}, "a restored page names it again"
    assert "first answer" in texts(pages[-1])


def test_an_unreadable_row_interrupts_the_continued_search(room, narrow_recent, monkeypatch):
    room.deliver("root answer", 5)
    with room.chat_log.open("ab") as stream:
        stream.write(b'{"direction": "out", "text": "torn\n')
    child_words(room, 8)
    monkeypatch.setattr(history_paging, "_READ_CEILING_BYTES", 1)
    recent = room.read(n_human="2")
    assert recent["window"]["latest_message"] is None and recent["window"]["latest_before"] > 0
    pages = follow(room, recent["next_cursor"])
    assert any("root answer" in texts(page) for page in pages)
    assert not any(page["window"].get("latest_message") for page in pages), \
        "past an unreadable row the answer may not be the newest message"


@pytest.mark.parametrize("torn", [False, True], ids=["clean", "unreadable"])
def test_a_continued_search_reaching_the_clean_start_of_the_chat_proves_no_message_is_there(
        room, narrow_recent, monkeypatch, torn):
    """A room holding only a child's words and the owner's own messages (a revision once counted a
    child's words): the older page that reaches the start of the chat with none says so, positively,
    so the bottom decides. Past an unreadable row nothing is proven."""
    if torn:
        room.chat_log.parent.mkdir(parents=True, exist_ok=True)
        with room.chat_log.open("ab") as stream:
            stream.write(b'{"direction": "out", "text": "torn\n')
    child_words(room, 8)
    message_bus.log_chat("in", room.chat_id, 7, "owner asks", ts=ts(40))
    monkeypatch.setattr(history_paging, "_READ_CEILING_BYTES", 1)
    recent = room.read(n_human="2")
    assert recent["window"]["latest_message"] is None and recent["window"]["latest_before"] > 0
    pages = follow(room, recent["next_cursor"])
    assert not any(page["window"].get("latest_message") for page in pages)
    assert [page["window"].get("latest_absent", False) for page in pages] == [False] * (len(pages) - 1) + [not torn]
    assert room.read(cursor=pages[-1]["page_cursor"])["window"].get("latest_absent", False) is not torn, \
        "a restored page says it again"


def test_a_search_that_found_the_newest_message_is_not_continued(room, narrow_recent, monkeypatch):
    room.deliver("first answer", 1)
    room.deliver("root answer", 5)
    child_words(room, 6)
    recent = room.read(n_human="2")
    assert recent["window"]["latest_message"] == {"history_id": room.row_id("root answer"), "out_of_order": True}
    assert "latest_before" not in recent["window"]
    monkeypatch.setattr(history_paging, "_READ_CEILING_BYTES", 1)
    assert not any("latest_message" in page["window"] for page in follow(room, recent["next_cursor"]))


def test_the_arrival_search_keeps_its_own_bound_below_the_page_ceiling(room, narrow_recent, monkeypatch):
    """Naming the newest arrival is paid on every recent read: it stops at its own small
    bound and passes the fact on, while the pages themselves read to their quota."""
    room.deliver("root answer", 5)
    child_words(room, 8)
    monkeypatch.setattr(history_paging, "_ARRIVAL_SEARCH_BYTES", 1)
    recent = room.read(n_human="2")
    assert recent["window"]["latest_message"] is None and recent["window"]["latest_before"] > 0
    older = room.read(cursor=recent["next_cursor"])
    assert len([text for text in texts(older) if text.startswith("child note")]) == 2, \
        "the page ceiling, not the arrival bound, governs an ordinary older page"


def test_a_cursor_carries_its_quiet_fact_as_a_boolean_and_older_cursors_still_read(room):
    import base64

    for minute in range(3):
        room.deliver(f"reply {minute}", minute)
    cursor = room.read(n_human="1")["next_cursor"]
    state = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    encode = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")  # noqa: E731
    legacy = {key: value for key, value in state.items() if key != "quiet"}
    assert texts(room.read(cursor=encode(legacy))) == texts(room.read(cursor=cursor)), "a saved older cursor reads on"
    response = asyncio.run(history.make_chat_history_endpoint(room.root)(SimpleNamespace(query_params={
        "chat_id": str(room.chat_id), "cursor": encode({**state, "quiet": "yes"})})))
    assert response.status_code == 400
