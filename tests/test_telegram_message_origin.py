"""Source room/work labels through registered bridge handlers and the actual Bot API encoder."""
from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from tests.test_telegram_reply_references import bridge, _state  # noqa: F401 - shared recording Bot API wire
from ouroboros.task_results import write_task_result
from ouroboros.projects_registry import create_project, update_project
from supervisor import message_bus
from supervisor.message_origin import message_origin


@pytest.fixture
def wired(bridge, monkeypatch):
    b = bridge
    data = b.api.state_dir / "data"
    data.mkdir(exist_ok=True)
    subscribers = {}
    b.api.register_supervised_task = lambda *args, **kwargs: None
    b.api.subscribe_event = lambda topic, handler: subscribers.update({topic: handler})
    b.api.register_route = lambda *args, **kwargs: None
    b.api.register_settings_section = lambda *args, **kwargs: None
    b.api.on_unload = lambda *args: None
    monkeypatch.setattr(b.plugin, "register_miniapp", lambda api: False)
    b.plugin.register(b.api)
    monkeypatch.setattr(message_bus, "DATA_DIR", data)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 42})
    monkeypatch.setattr(message_bus, "_chat_media_download_url", lambda *args: ("", ""))
    events = []

    def publish(topic, event):
        events.append((topic, event))
        if topic in subscribers:
            asyncio.run(subscribers[topic](event))

    monkeypatch.setattr(message_bus, "publish_event", publish)
    bus = message_bus.LocalChatBridge({})
    return SimpleNamespace(**vars(b), data=data, bus=bus, events=events, subscribers=subscribers)


def tasks(w, *, depth=1):
    write_task_result(w.data, "root-work", "running", title="Build export", chat_id=1)
    parent = "root-work"
    for index in range(depth):
        tid = f"worker-{index}"
        write_task_result(w.data, tid, "running", title=f"Reader {index}", chat_id=1,
                          parent_task_id=parent, root_task_id="root-work", delegation_role="subagent")
        parent = tid
    return parent


def test_main_direct_and_named_work_are_distinct(wired):
    w = wired
    write_task_result(w.data, "direct", "running", _is_direct_chat=True, title="Hello", chat_id=1)
    w.bus.send_message(1, "Direct answer", task_id="direct")
    tasks(w)
    w.bus.send_message(1, "Work answer", task_id="root-work")
    texts = [data["text"] for method, data in w.calls if method == "sendMessage"]
    assert texts == ["Main chat\n\nDirect answer", "Main chat\nTask: Build export\n\nWork answer"]
    assert w.api.logs == []


def test_nested_worker_room_rename_and_move_use_each_events_room(wired):
    w = wired
    child = tasks(w, depth=2)
    project = create_project(w.data, "source-export", name="Source export")
    for chat in (1, project["chat_id"]):
        w.bus.send_message(chat, "Read the writer", task_id=child,
                           progress_meta={"system_type": "proactive_message"})
    update_project(w.data, project["id"], name="New name")
    w.bus.send_message(project["chat_id"], "Renamed", task_id=child,
                       progress_meta={"system_type": "proactive_message"})
    texts = [data["text"] for method, data in w.calls if method == "sendMessage"]
    assert texts[0].startswith("Main chat\nTask: Build export › Worker: Reader 0 › Worker: Reader 1\n\n")
    assert texts[1].startswith("Project: Source export\nTask: Build export")
    assert texts[2].startswith("Project: New name\nTask: Build export")


def test_worker_lifecycle_edit_keeps_bubble_and_current_project(wired):
    w = wired
    child = tasks(w)
    project = create_project(w.data, "cards", name="Cards")
    for state, text in (("scheduled", "Queued"), ("running", "Working")):
        w.bus.send_message(project["chat_id"], text, task_id=child, is_progress=True,
                           progress_meta={"subagent_event": state, "subagent_task_id": child})
        update_project(w.data, project["id"], name="Cards now")
    assert [method for method, _ in w.calls] == ["sendMessage", "editMessageText"]
    assert w.calls[0][1]["text"].startswith("Project: Cards\n")
    assert w.calls[1][1]["text"].startswith("Project: Cards now\n")
    assert w.calls[1][1]["message_id"] == "101"


def test_label_is_literal_and_multipart_body_and_references_survive(wired):
    w = wired
    title = "<b>😀 *R&D*</b>"
    project = create_project(w.data, "literal", name=title)
    body = "**α & β** 😀\n" * 700
    w.bus.send_message(project["chat_id"], body, parse_mode="Markdown", task_id="missing")
    assert len(w.calls) > 2
    stripped = []
    for method, data in w.calls:
        assert method == "sendMessage" and w.transport._u16len(data["text"]) <= 4096
        assert data["text"].startswith("<b>Project: &lt;b&gt;😀 *R&amp;D*&lt;/b&gt;\nTask: missing</b>\n\n")
        stripped.append(data["text"].split("</b>\n\n", 1)[1])
        mid = 100 + len(stripped)
        assert _state(w.plugin)._reply_reference(w.api, "", 42, mid) == "task:missing"
    expected = w.transport.markdown_to_telegram_html(body.strip())  # existing outbound handler trims its body
    assert "".join(w.transport._telegram_html_to_plain(part) for part in stripped) == w.transport._telegram_html_to_plain(expected)


@pytest.mark.parametrize("kind", ["photo", "video", "document", "links"])
def test_media_and_links_carry_origin_and_exact_reference(wired, kind):
    w = wired
    child = tasks(w)
    if kind == "photo":
        w.bus.send_photo(1, b"photo", caption="Caption", task_id=child)
    elif kind == "video":
        w.bus.send_video(1, b"video", caption="Caption", task_id=child)
    elif kind == "document":
        w.bus.send_document(1, b"document", filename="report.txt", caption="Caption", task_id=child)
    else:
        w.bus.send_links(1, [{"label": "Read", "url": "https://example.com"}], title="Links", task_id=child)
    assert len(w.calls) == 1
    method, data = w.calls[0]
    field = "text" if kind == "links" else "caption"
    assert "Main chat\nTask: Build export › Worker: Reader 0" in data[field]
    event = w.events[-1][1]
    assert event["task_id"] == child and event.get("row_address")
    assert _state(w.plugin)._reply_reference(w.api, "", 42, 101) == event["row_address"]
    assert w.api.logs == []


def test_long_caption_kept_as_ordered_text_before_media(wired):
    w = wired
    child = tasks(w)
    caption = "A 😀 & B " * 400
    w.bus.send_photo(1, b"photo", caption=caption, task_id=child)
    assert w.calls[-1][0] == "sendPhoto"
    assert w.transport._u16len(w.calls[-1][1]["caption"]) <= 1024
    text = "".join(w.transport._telegram_html_to_plain(data["text"].split("</b>\n\n", 1)[1])
                   for method, data in w.calls if method == "sendMessage")
    assert text == caption
    ref = w.events[-1][1]["row_address"]
    assert all(_state(w.plugin)._reply_reference(w.api, "", 42, 101 + i) == ref for i in range(len(w.calls)))


def test_quiz_edit_retains_origin_after_fresh_client(wired):
    w = wired
    child = tasks(w)
    q = sys.modules[w.plugin.__package__ + ".lib.telegram_quiz"]
    event = {"chat_id": 1, "task_id": child, "quiz_id": "choose", "question": "Choose?",
             "options": [{"label": "A"}], "message_origin": message_origin(w.data, 1, child)}
    asyncio.run(w.subscribers["chat.quiz"](event))
    asyncio.run(w.subscribers["chat.quiz_state"]({"task_id": child, "quiz_id": "choose", "state": "closed", "closed_reason": "Done"}))
    assert [method for method, _ in w.calls] == ["sendMessage", "editMessageText"]
    assert w.calls[0][1]["text"].startswith("Main chat\nTask: Build export › Worker: Reader 0\n\n")
    assert w.calls[1][1]["text"].startswith("Main chat\nTask: Build export › Worker: Reader 0\n\n")
    assert q.quiz_for_message(w.api, "", 42, 101)["task_id"] == child


def test_missing_cyclic_and_deep_lineage_do_not_invent_main(wired, monkeypatch):
    w = wired
    child = tasks(w, depth=9)
    origin = message_origin(w.data, 55555, child)
    assert origin["room"] == {"kind": "unknown"} and origin["incomplete"]
    event = {"message_origin": origin}
    label = sys.modules[w.plugin.__package__ + ".lib.telegram_origin"].origin_label(event, "en")
    assert label.startswith("Chat\n…") and "Task: Build export" in label and label.count("Worker:") == 2 and len(label) < 240
    assert message_origin(w.data, 0)["room"]["kind"] == "unknown"
    assert message_origin(None, 1, "direct", event={"_is_direct_chat": True})["tasks"] == []
    assert w.plugin.origin_label({"chat_id": 444}, "en") == ""
    import supervisor.message_origin as source
    monkeypatch.setattr(source, "project_of_room", lambda *args: (_ for _ in ()).throw(OSError("unreadable")))
    assert message_origin(w.data, 55555)["room"]["kind"] == "unknown"


@pytest.mark.parametrize("answer_kind", ["callback", "reply", "closed"])
@pytest.mark.parametrize("question", ["Q" * 3300, "Long question 😀\n" * 800], ids=["short", "overflow"])
def test_native_quiz_answer_preserves_heading_without_polluter(wired, answer_kind, question):
    w = wired
    child = tasks(w)
    q = sys.modules[w.plugin.__package__ + ".lib.telegram_quiz"]
    event = {"chat_id": 1, "task_id": child, "quiz_id": "native", "question": question,
             "options": [{"label": "A"}], "message_origin": message_origin(w.data, 1, child)}
    asyncio.run(w.subscribers["chat.quiz"](event))
    token = q.mint_token(child, "native")
    record = q.quiz_for_token(w.api, "", token)
    heading = "Main chat\nTask: Build export › Worker: Reader 0\n\n"
    for method, data in w.calls:
        assert method == "sendMessage" and data["chat_id"] == "42"
        assert data["text"].startswith(heading) and data["text"].count(heading) == 1
        assert w.transport._u16len(data["text"]) <= 4096
    assert record["text"].startswith(heading) and record["text"].count(heading) == 1
    client = w.plugin._telegram_client(w.api)  # the poller's long-lived client, no event

    async def post(*args):
        if answer_kind == "closed":
            return 409, {"state": "closed", "outcome": "closed", "closed_reason": "Finished"}
        return 200, {"status": "answered", "state": "answered", "accepted": True,
                     "answered_index": 0, "comment": "😀" * 200}

    async def act():
        if answer_kind == "callback":
            await q.answer_from_callback(w.api, client, f"qz:{token}:0", cb_id="tap", request_key="tap", lang="en", post=post)
        else:
            await q.answer_from_reply(w.api, client, record, "Answer", chat_id=42, request_key="reply", lang="en", post=post)
        await client.send_message(42, "Unrelated response", parse_mode="")

    asyncio.run(act())
    edits = [data for method, data in w.calls if method == "editMessageText"]
    assert edits and edits[-1]["text"].startswith("Main chat\nTask: Build export › Worker: Reader 0\n\n")
    assert edits[-1]["text"].count(heading) == 1
    assert w.transport._u16len(edits[-1]["text"]) <= 4096
    assert w.calls[-1][1]["text"] == "Unrelated response"
    assert client.origin_label == ""


def test_quiz_creation_keeps_event_origin_and_published_answer_edits_once(wired, monkeypatch):
    w = wired
    child = tasks(w)
    clients = []
    real_client = w.plugin._telegram_client

    def client(*args, **kwargs):
        value = real_client(*args, **kwargs)
        clients.append((value, value.origin_label))
        return value

    monkeypatch.setattr(w.plugin, "_telegram_client", client)
    asyncio.run(w.subscribers["chat.quiz"]({
        "chat_id": 1, "task_id": child, "quiz_id": "published", "question": "Already answered?",
        "options": [{"label": "A"}], "state": "answered", "answered_index": 0,
        "message_origin": message_origin(w.data, 1, child)}))
    heading = "Main chat\nTask: Build export › Worker: Reader 0\n\n"
    assert [method for method, _ in w.calls] == ["sendMessage", "editMessageText"]
    assert all(data["text"].startswith(heading) and data["text"].count(heading) == 1 for _, data in w.calls)
    assert w.calls[-1][1]["text"] == heading + "Question: Already answered?\n1. A\nAnswered: 1. A"
    assert clients[0][1] and all(value.origin_label == original for value, original in clients)
    assert w.api.logs == []


def test_quiz_heading_counts_in_the_answered_edit_budget(wired):
    w = wired
    child = tasks(w)
    q = w.plugin.telegram_quiz
    heading = "Main chat\nTask: Build export › Worker: Reader 0\n\n"
    hint = q.hint("en")
    # Without the origin this fits exactly with the edit reserve; with it the
    # card must use the compact answer door, not spend that reserve on chrome.
    size = 4096 - q.answered_edit_reserve("en") - w.transport._u16len("Question: \n1. A\n" + hint)
    event = {"chat_id": 1, "task_id": child, "quiz_id": "budget", "question": "Q" * size,
             "options": [{"label": "A"}], "message_origin": message_origin(w.data, 1, child)}
    asyncio.run(w.subscribers["chat.quiz"](event))
    assert len(w.calls) == 2
    assert "reply_markup" not in w.calls[0][1] and "reply_markup" in w.calls[1][1]
    assert w.calls[0][1]["text"] == heading + "Question: " + "Q" * size + "\n1. A"
    record = q.quiz_for_message(w.api, "", 42, 102)
    assert record["text"] == heading + "1. A"
    assert w.api.logs == []


def test_audio_document_fallback_does_not_repeat_long_caption(wired):
    w = wired
    child = tasks(w)
    real = w.transport.TelegramClient.call

    async def call(client, method, **kwargs):
        if method == "sendAudio":
            raise w.transport.TelegramRequestRejected("Unsupported audio", status_code=400)
        return await real(client, method, **kwargs)

    from unittest.mock import patch
    caption = "Caption 😀 " * 150
    with patch.object(w.transport.TelegramClient, "call", call):
        w.bus.send_document(1, b"audio", filename="a.mp3", caption=caption, mime="audio/mpeg", task_id=child)
    assert [method for method, _ in w.calls] == ["sendMessage", "sendDocument"]
    text = w.transport._telegram_html_to_plain(w.calls[0][1]["text"].split("</b>\n\n", 1)[1])
    assert text == caption


def test_labelled_multipart_keeps_the_native_owner_reply(wired):
    from tests.test_telegram_reply_references import _accept_owner

    w = wired
    ref = _accept_owner(w, 10)
    _accept_owner(w, 11)
    event = {"chat_id": 1, "task_id": "root-work", "reply_to": ref, "text": "Reply " * 2000,
             "message_origin": {"room": {"kind": "main"}, "tasks": [{"id": "root-work", "name": "Export", "worker": False}]}}
    asyncio.run(w.subscribers["chat.outbound"](event))
    assert len(w.calls) > 2
    assert all(data.get("reply_to_message_id") == 10 for _, data in w.calls)
    assert w.api.logs == []


def test_early_worker_and_cycle_keep_unknown_ancestry_explicit(wired):
    w = wired
    early = message_origin(w.data, 1, event={"subagent_task_id": "early-child"})
    assert early["tasks"] == [{"id": "early-child", "name": "", "worker": True}] and early["incomplete"]
    tasks(w)
    write_task_result(w.data, "cycle", "running", parent_task_id="cycle", root_task_id="root-work", delegation_role="subagent")
    cyclic = message_origin(w.data, 1, "cycle")
    assert cyclic["incomplete"] and cyclic["tasks"][0]["name"] == "Build export"


def test_long_heading_is_bounded_but_authored_text_is_not(wired):
    w = wired
    title = "Широкая длинная задача 😀 " * 3
    write_task_result(w.data, "long", "running", title=title, chat_id=1)
    w.bus.send_message(1, "Complete body\nsecond line", task_id="long")
    text = w.calls[0][1]["text"]
    assert "…\n\nComplete body\nsecond line" in text and len(text.split("\n\n")[0]) < 80


def test_card_fallback_has_own_origin_and_reference(wired, monkeypatch):
    from ouroboros.card_store import write_card
    from supervisor import card_delivery

    w = wired
    child = tasks(w)
    record = write_card(w.data, chat_id=1, task_id=child, title="Overview", fallback_text="Whole fallback",
                        kind="components", body={"components": [{"type": "markdown", "text": "Hello"}]})
    monkeypatch.setattr(card_delivery, "publish_event", message_bus.publish_event)
    card_delivery.send_card(w.bus, record, 1, task_id=child, interactive=False)
    assert w.calls[-1][1]["text"] == "Main chat\nTask: Build export › Worker: Reader 0\n\nWhole fallback"
    assert _state(w.plugin)._reply_reference(w.api, "", 42, 101).startswith("row:1@")


def test_origin_does_not_change_mirror_or_child_final_policy(wired):
    w = wired
    child = tasks(w)
    origin = message_origin(w.data, 1, child)
    asyncio.run(w.subscribers["chat.outbound"]({"chat_id": 1, "task_id": child, "text": "Private child report",
                "delegation_role": "subagent", "message_origin": origin}))
    assert w.calls == []
    (w.api.state_dir / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": "telegram_only"}), encoding="utf-8")
    w.bus.send_message(1, "Web-only", task_id="root-work")
    assert w.calls == []
