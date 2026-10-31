"""Telegram references are transport facts; the host owns their meaning and routing."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests.test_telegram_format_parity import _load_skill
from tests.test_telegram_routing_notices import phone  # noqa: F401 -- the owner's chat behind the real bus

# The shared native-skill loader replaces its package entries in sys.modules.
pytestmark = pytest.mark.serial


class Api:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.logs = []

    def get_state_dir(self):
        return str(self.state_dir)

    def get_runtime_info(self):
        return {"data_dir": str(self.state_dir / "data")}

    def get_settings(self, _keys):
        return {"TELEGRAM_BOT_TOKEN": "test-token"}

    def log(self, level, message):
        self.logs.append((level, message))


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    plugin, telegram_api = _load_skill()
    api = Api(tmp_path)
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    calls, replies = [], []

    class Wire:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, url, *, data, files):
            calls.append((url.rsplit("/", 1)[-1], dict(data or {})))
            if replies:
                value = replies.pop(0)
                if isinstance(value, Exception):
                    raise value
                return httpx.Response(value[0], json=value[1])
            mid = (data or {}).get("message_id") or (100 + len(calls))
            return httpx.Response(200, json={"ok": True, "result": {"message_id": int(mid)}})

    monkeypatch.setattr(telegram_api.httpx, "AsyncClient", Wire)
    return SimpleNamespace(plugin=plugin, transport=telegram_api, api=api, calls=calls, replies=replies)


def _state(plugin):
    import sys
    return sys.modules[plugin.__package__ + ".lib.telegram_state"]


def test_every_chunk_id_is_retained_across_reload_and_nonlast_resolves(bridge):
    b = bridge
    state = _state(b.plugin)
    record = {"row_address": "row:1@2026-10-16T12:00:00Z#123456789abc", "task_id": "work-1", "kind": "message"}
    client = b.transport.TelegramClient("test-token", on_message=lambda chat, mid: state._remember_message(b.api, chat, mid, record))
    last = asyncio.run(client.send_message(42, "word " * 2000, parse_mode=""))
    assert len(b.calls) >= 3 and last == 100 + len(b.calls)
    plugin, _transport = _load_skill()  # no in-memory map survives a plugin restart
    restored = _state(plugin)
    for mid in range(101, last + 1):
        assert restored._reply_reference(Api(b.api.state_dir), 42, mid) == record["row_address"]
    assert restored._reply_reference(b.api, 43, 101) == ""
    assert restored._reply_reference(b.api, 42, 9999) == ""


def test_confirmed_chunk_survives_later_unknown_send(bridge):
    b = bridge
    seen = []
    b.replies.extend([(200, {"ok": True, "result": {"message_id": 51}}), httpx.ReadTimeout("offline")])
    client = b.transport.TelegramClient("test-token", on_message=lambda chat, mid: seen.append((chat, mid)))
    with pytest.raises(b.transport.TelegramTransportError):
        asyncio.run(client.send_message(42, "x" * 5000, parse_mode=""))
    assert seen == [(42, 51)] and len(b.calls) == 2


def test_fallback_media_keyboard_and_edit_share_confirmed_message_observer(bridge):
    b = bridge
    seen = []
    client = b.transport.TelegramClient("test-token", on_message=lambda chat, mid: seen.append((chat, mid)), reply_to_message_id=17)
    b.replies.extend([(400, {"ok": False}), (200, {"ok": True, "result": {"message_id": 61}})])

    async def run():
        await client.send_message(42, "**Hello**")
        await client.send_photo(42, "eA==")
        await client.send_document(42, b"file")
        await client.send_audio(42, b"file", "a.mp3")
        await client.send_message_with_inline_keyboard(42, "Choose", [[{"text": "A", "callback_data": "a"}]])
        await client.call("sendVideo", data={"chat_id": "42"})
        assert await client.edit_message_text(42, 61, "Updated")
        b.replies.append((400, {"ok": False, "description": "Bad Request: message is not modified"}))
        assert await client.edit_message_text(42, 61, "Updated")

    asyncio.run(run())
    assert [mid for _, mid in seen] == [61, 103, 104, 105, 106, 107, 61, 61]
    for method, data in b.calls:
        assert (data.get("reply_to_message_id") == 17) is method.startswith("send")
    assert b.calls[1][1]["text"] == "Hello"


def test_message_reference_retention_is_bounded_and_work_cards_use_task_address(bridge, monkeypatch):
    state, api = _state(bridge.plugin), bridge.api
    monkeypatch.setattr(state, "_MAX_MESSAGE_REFS", 3)
    for mid in range(1, 6):
        state._remember_message(api, 42, mid, {"row_address": f"row-{mid}", "kind": "work_card", "task_id": f"task-{mid}"})
    state._remember_message(api, 42, 3, {"row_address": "updated-row", "kind": "message"})
    assert list(state._message_refs(api)) == ["42:4", "42:5", "42:3"]
    assert state._reply_reference(api, 42, 1) == ""
    assert state._reply_reference(api, 42, 4) == "task:task-4"
    assert state._reply_reference(api, 42, 3) == "updated-row"


def _owner(mid=50, text="Explain this", *, reply=0, chat=42):
    return {"message_id": mid, "chat": {"id": chat, "type": "private"}, "from": {"id": chat},
            "text": text, **({"reply_to_message": {"message_id": reply}} if reply else {})}


def _host_posts(b, monkeypatch):
    posts = []

    async def post(_api, path, payload):
        posts.append((path, payload))
        return 202, {"ok": True, "status": "queued"}

    monkeypatch.setattr(b.plugin, "_host_post", post)
    return posts


def _receive(b, message, *, update_id=70):
    asyncio.run(b.plugin._handle_owner_message(b.api, b.plugin._telegram_client(b.api),
                {"message": message}, update_id, "full_access", "en"))


@pytest.mark.parametrize("kind,address,expected", [
    ("message", "row:1@time#123456789abc", "row:1@time#123456789abc"),
    ("work_card", "row:1@time#123456789abc", "task:work-1"),
    ("message", "", "task:work-1"),
    ("completion", "row:1@later#123456789abc", "row:1@later#123456789abc"),
])
def test_reply_to_mirrored_work_goes_through_normal_ingress_as_reference(bridge, monkeypatch, kind, address, expected):
    b = bridge
    event = {"text": "Explanation " * 1000, "task_id": "work-1", "chat_id": 1,
             "kind": kind, "row_address": address}
    asyncio.run(b.plugin._make_outbound(b.api)(event))
    assert len(b.calls) > 1
    posts = _host_posts(b, monkeypatch)
    _receive(b, _owner(reply=101))  # the FIRST chunk, not the last one
    path, payload = posts[0]
    assert path == "/chat/inject" and payload["reply_to"] == expected
    assert payload["text"] == "Explain this" and payload["chat_id"] == 42
    assert "task_id" not in payload and "task_metadata" not in payload
    assert payload["client_message_id"] == "tg:42:50"
    _receive(b, _owner(reply=101), update_id=71)
    assert posts[1][1]["client_message_id"] == payload["client_message_id"]
    assert b.api.logs == []


def test_unmapped_or_other_chat_reply_has_no_reference_and_no_notice(bridge, monkeypatch):
    b = bridge
    _state(b.plugin)._remember_message(b.api, 43, 101, {"row_address": "other-chat"})
    posts = _host_posts(b, monkeypatch)
    _receive(b, _owner(reply=101))
    _receive(b, _owner(51, reply=999))
    assert all("reply_to" not in payload for _, payload in posts)
    assert b.calls == [] and b.api.logs == []


def _accept_owner(b, mid, *, text="Same owner words", chat=42):
    """Use the real canonical ingress writer; owner lookup must never use text."""
    from supervisor.message_bus import accept_local_message
    from ouroboros.chat_chain import format_address, row_address

    state = _state(b.plugin)
    state._owner_message_seen(b.api, chat, mid)
    fields = state._owner_ingress_fields(b.api, chat, _owner(mid, text, chat=chat), mid)
    receiver = SimpleNamespace(enqueue_local_message=lambda *args, **kwargs: None)
    row, _ = accept_local_message(receiver, b.api.state_dir / "data", text,
                                 chat_id=chat, user_id=chat, source="skill:telegram", **fields)
    return {"address": format_address(row_address(row)), "chat_id": chat, "role": "user",
            "author_label": "You", "kind": "message", "snippet": text, "ts": row["ts"], "task_id": ""}


@pytest.mark.parametrize("target", ["earlier", "latest", "other_chat", "unmapped", "system", "unresolved"])
def test_native_reply_only_for_nonlatest_owner_from_same_telegram_chat(bridge, target):
    b = bridge
    earlier = _accept_owner(b, 10)
    latest = _accept_owner(b, 11)  # deliberately identical text; identity decides
    other = _accept_owner(b, 12, chat=43)
    reference = {"earlier": earlier, "latest": latest, "other_chat": other,
                 "unmapped": {**earlier, "address": "row:1@unknown#000000000000"},
                 "system": {**earlier, "role": "system"},
                 "unresolved": {**earlier, "unresolved": True}}[target]
    # Reload before the first answer: pending ingress associations also survive restart.
    b.plugin, _ = _load_skill()
    asyncio.run(b.plugin._make_outbound(b.api)({"chat_id": 1, "text": "Answer " * 1000,
                "row_address": "answer-row", "reply_to": reference}))
    assert len(b.calls) > 1
    for _, data in b.calls:
        assert data.get("reply_to_message_id") == (10 if target == "earlier" else None)
    assert b.api.logs == []
    if target == "earlier":
        assert _state(b.plugin)._message_ref(b.api, 42, 10)["row_address"] == earlier["address"]


def test_local_owner_command_breaks_native_reply_adjacency(bridge, monkeypatch):
    b = bridge
    ref = _accept_owner(b, 10)
    _receive(b, _owner(11, "/help"))
    b.calls.clear()
    asyncio.run(b.plugin._make_outbound(b.api)({"text": "Earlier answer", "reply_to": ref}))
    assert b.calls[0][1]["reply_to_message_id"] == 10


def test_latest_owner_identity_survives_reference_eviction(bridge, monkeypatch):
    b = bridge
    state = _state(b.plugin)
    monkeypatch.setattr(state, "_MAX_MESSAGE_REFS", 2)
    ref = _accept_owner(b, 10)
    _accept_owner(b, 11)
    # A late binding refreshes an old record; it must not make it the latest input.
    record = {**state._message_ref(b.api, 42, 10), "row_address": ref["address"]}
    state._remember_message(b.api, 42, 10, record)
    state._remember_message(b.api, 42, 100, {"row_address": "bot-row"})
    assert state._message_ref(b.api, 42, 11) == {}
    assert state._native_reply_id(b.api, 42, {"reply_to": ref}) == 10


@pytest.mark.parametrize("surface,extra", [
    ("photo", {"image_base64": "eA=="}),
    ("video", {"video_base64": "eA=="}),
    ("document", {"file_base64": "eA==", "filename": "note.txt"}),
    ("document", {"file_base64": "eA==", "filename": "note.mp3", "mime": "audio/mpeg"}),
    ("links", {"title": "Read this", "actions": [{"label": "Source", "url": "https://example.com"}]}),
])
def test_every_mirrored_media_surface_retains_reference_and_native_reply(bridge, surface, extra):
    b = bridge
    ref = _accept_owner(b, 10)
    _accept_owner(b, 11)
    event = {"chat_id": 1, "row_address": "host-media-row", "task_id": "work", "reply_to": ref, **extra}
    asyncio.run(getattr(b.plugin, "_make_" + surface)(b.api)(event))
    assert len(b.calls) == 1 and b.calls[0][1]["reply_to_message_id"] == 10
    assert _state(b.plugin)._reply_reference(b.api, 42, 101) == "host-media-row"
    assert b.api.logs == []


def test_long_quiz_keeps_answer_door_and_earlier_parts_carry_references(bridge, monkeypatch):
    b = bridge
    event = {"chat_id": 1, "row_address": "question-row", "quiz_id": "quiz", "task_id": "task",
             "question": "Long question " * 900, "options": [{"label": "A"}, {"label": "B"}]}
    asyncio.run(b.plugin._make_quiz(b.api)(event))
    assert len(b.calls) > 2
    last = 100 + len(b.calls)
    state = _state(b.plugin)
    assert all(state._reply_reference(b.api, 42, mid) == "question-row" for mid in range(101, last + 1))
    assert state._message_ref(b.api, 42, 101)["kind"] == "question"
    posts = _host_posts(b, monkeypatch)
    _receive(b, _owner(50, "Please explain", reply=101))
    assert posts[0][0] == "/chat/inject" and posts[0][1]["reply_to"] == "question-row"
    _receive(b, _owner(51, "B", reply=last))
    assert posts[1][0] == "/chat/decision"
    assert posts[1][1]["decision_id"] == "quiz:task:quiz" and posts[1][1]["comment"] == "B"


def _quiet(b, **settings):
    b.plugin.merge_settings(b.api.state_dir, {"TELEGRAM_SILENT_MODE": "on", **settings})
    return b.plugin._make_outbound(b.api)


def _note(text, task_id="episode-1", **fields):
    """A progress note as the agent publishes its narration (``ouroboros/agent.py``)."""
    return {"text": text, "task_id": task_id, "is_progress": True, "role": "assistant",
            "system_type": "model_narration", **fields}


def test_silent_mode_replaces_progress_only_within_its_episode_and_reply_origin(bridge):
    b = bridge
    handle = _quiet(b, TELEGRAM_MIRROR_PROGRESS="on")

    async def run():
        await handle(_note("First", row_address="row-first"))
        await handle(_note("Update", row_address="row-update"))
        await handle(_note("Continue", "episode-2", row_address="row-continue"))
        await handle(_note("Other origin", "episode-2", row_address="row-origin",
                           reply_to={"address": "other-owner"}))

    asyncio.run(run())
    assert [method for method, _ in b.calls] == ["sendMessage", "editMessageText", "sendMessage", "sendMessage"]
    assert _state(b.plugin)._reply_reference(b.api, 42, 101) == "row-update"
    assert _state(b.plugin)._reply_reference(b.api, 42, 103) == "row-continue"
    assert _state(b.plugin)._reply_reference(b.api, 42, 104) == "row-origin"


def test_silent_mode_starts_new_message_when_native_reply_anchor_changes(bridge):
    b = bridge
    handle = _quiet(b, TELEGRAM_MIRROR_PROGRESS="on")
    ref = _accept_owner(b, 10)
    event = _note("First note", "same-episode", reply_to=ref)
    asyncio.run(handle(event))
    _state(b.plugin)._owner_message_seen(b.api, 42, 11)
    asyncio.run(handle({**event, "text": "More about the earlier request"}))
    assert [method for method, _ in b.calls] == ["sendMessage", "sendMessage"]
    assert "reply_to_message_id" not in b.calls[0][1] and b.calls[1][1]["reply_to_message_id"] == 10


_REPLIES = ["Starting: the page template first, then the three longest notes.",
            "Found it: long tables break at the page edge; wrapping rows, not shrinking the font.",
            "Done: the PDF export works and long tables wrap."]


@pytest.mark.parametrize("silent", ["on", "off"])
def test_every_authored_message_and_the_final_answer_stay_their_own_messages(bridge, silent):
    b = bridge
    b.plugin.merge_settings(b.api.state_dir, {"TELEGRAM_SILENT_MODE": silent})
    handle = b.plugin._make_outbound(b.api)

    async def run():
        await handle({"text": _REPLIES[0], "task_id": "t1", "system_type": "proactive_message"})
        await handle({"text": _REPLIES[1], "task_id": "t1", "system_type": "proactive_message"})
        await handle({"text": _REPLIES[2], "task_id": "t1"})

    asyncio.run(run())
    assert [method for method, _ in b.calls] == ["sendMessage"] * 3
    assert [data["text"] for _, data in b.calls] == _REPLIES
    # Silent mode means no sound; it never means fewer messages.
    assert [data.get("disable_notification") for _, data in b.calls] == [True if silent == "on" else None] * 3
    assert b.plugin._get_silent_msg(b.api, 42) == 0


def test_progress_replaces_progress_but_never_a_reply_or_the_final_answer(bridge):
    b = bridge
    handle = _quiet(b, TELEGRAM_MIRROR_PROGRESS="on")
    events = [{"text": _REPLIES[0], "task_id": "t1", "system_type": "proactive_message"},
              _note("Reading the page template.", "t1"), _note("Building a trial PDF.", "t1"),
              {"text": _REPLIES[1], "task_id": "t1", "system_type": "proactive_message"},
              _note("Checking table wrapping.", "t1"), {"text": _REPLIES[2], "task_id": "t1"}]

    async def run():
        for event in events:
            await handle(event)

    asyncio.run(run())
    assert [method for method, _ in b.calls] == [
        "sendMessage", "sendMessage", "editMessageText", "sendMessage", "sendMessage", "sendMessage"]
    edit = b.calls[2][1]
    assert edit["message_id"] == "102" and edit["text"] == "Building a trial PDF."
    visible = {}
    for index, (method, data) in enumerate(b.calls, 101):
        visible[int(data["message_id"]) if method == "editMessageText" else index] = data["text"]
    assert list(visible.values()) == [_REPLIES[0], "Building a trial PDF.", _REPLIES[1],
                                      "Checking table wrapping.", _REPLIES[2]]
    assert all(data.get("disable_notification") is True for method, data in b.calls if method == "sendMessage")
    assert b.plugin._get_silent_msg(b.api, 42) == 0, "the final answer ends the progress run"


def test_a_split_progress_note_is_never_replaced(bridge):
    """Editing the last part of a long note would erase that part: only a one-message note is replaceable."""
    b = bridge
    handle = _quiet(b, TELEGRAM_MIRROR_PROGRESS="on")
    long_note = "Reading every chapter. " * 400
    asyncio.run(handle(_note(long_note, "t1")))
    parts = len(b.calls)
    assert parts > 1
    asyncio.run(handle(_note("Next step.", "t1")))
    assert [method for method, _ in b.calls] == ["sendMessage"] * (parts + 1)
    assert b.calls[-1][1]["text"] == "Next step."
    asyncio.run(handle(_note(long_note, "t1")))  # nor does a split note replace a short one
    assert "editMessageText" not in [method for method, _ in b.calls]


def test_a_message_tracked_before_this_version_is_never_replaced(bridge):
    """An older bridge tracked authored replies; with no progress fact on it that reply stays."""
    b = bridge
    handle = _quiet(b, TELEGRAM_MIRROR_PROGRESS="on")
    _state(b.plugin)._remember_message(b.api, 42, 77, {
        "row_address": "old-reply", "chat_id": 1, "task_id": "t1", "kind": "message",
        "reply_to": "", "native_reply_id": 0})
    b.plugin._set_silent_msg(b.api, 42, 77)
    asyncio.run(handle(_note("Checking the result.", "t1")))
    assert [method for method, _ in b.calls] == ["sendMessage"]
    asyncio.run(handle({"text": "Same task, same origin: still its own message.", "task_id": "t1"}))
    assert [method for method, _ in b.calls] == ["sendMessage", "sendMessage"]


def test_questions_files_and_host_rows_keep_their_sound_in_silent_mode(bridge):
    b = bridge
    b.plugin.merge_settings(b.api.state_dir, {"TELEGRAM_SILENT_MODE": "on"})
    asyncio.run(b.plugin._make_outbound(b.api)({
        "text": "The task stopped: the working folder can't be used.", "role": "system",
        "system_type": "terminal_incident", "task_id": "t1"}))
    asyncio.run(b.plugin._make_quiz(b.api)({
        "chat_id": 1, "quiz_id": "q", "task_id": "t1", "question": "Which name?",
        "options": [{"label": "A"}, {"label": "B"}]}))
    asyncio.run(b.plugin._make_document(b.api)({"chat_id": 1, "file_base64": "eA==", "filename": "notes.pdf"}))
    assert [method for method, _ in b.calls] == ["sendMessage", "sendMessage", "sendDocument"]
    assert all("disable_notification" not in data for _, data in b.calls)


def _host_note(text, task_id="episode-1", **fields):
    """A technical progress note in the host's voice, as the agent publishes it (``ouroboros/agent.py``)."""
    return {**_note(text, task_id, **fields), "role": "system", "system_type": "host_progress"}


def test_silent_mode_mutes_and_replaces_the_hosts_technical_progress_but_not_its_problems(bridge):
    """Owner's choice, 27 Oct: A, every message muted. The host's typed technical notes are progress like the
    model's own: muted, and the next note of the episode replaces the last. A typed incident is a problem: it
    keeps its sound, is never replaced and leaves the progress run as it was."""
    b = bridge
    handle = _quiet(b, TELEGRAM_MIRROR_PROGRESS="on")

    async def run():
        await handle(_host_note("Checkpoint 3 at round 12"))
        await handle(_host_note("Running the tests"))
        await handle(_host_note("Fallback: switching model lane", task_incident="model_lane_switch"))
        await handle(_note("Reading the failing test first."))

    asyncio.run(run())
    assert [(method, data.get("disable_notification")) for method, data in b.calls] == [
        ("sendMessage", True), ("editMessageText", None), ("sendMessage", None), ("editMessageText", None)]
    assert b.calls[1][1]["message_id"] == b.calls[3][1]["message_id"] == "101"  # the first note, form-encoded


def test_the_agents_host_progress_reaches_the_phone_muted_and_replaced(phone, monkeypatch):  # noqa: F811
    """The real chain: ``OuroborosAgent._emit_progress`` → the supervisor's send_message delivery → the bus →
    this bridge. In Silent mode the host's technical note rings no sound and the next one replaces it; the
    incident note the same producer types keeps its sound."""
    from supervisor import events_chat_delivery
    from tests.test_progress_narration_voice import _agent, _emit

    phone.plugin.merge_settings(phone.api.state_dir, {"TELEGRAM_SILENT_MODE": "on", "TELEGRAM_MIRROR_PROGRESS": "on"})
    monkeypatch.setattr(events_chat_delivery, "_bound_project_chat_id", lambda *_: 0)
    phone.ctx.RUNNING["task-1"] = {"task": {"id": "task-1", "_attempt": 0}}
    agent, events = _agent()
    _emit(agent, "Checkpoint 3 at round 12")
    _emit(agent, "Running the tests")
    _emit(agent, "Fallback: switching model lane", incident={"task_incident": "model_lane_switch", "toast_once": "lane"})
    for _ in range(3):
        assert events_chat_delivery._handle_send_message(events.get_nowait(), phone.ctx) is True
    assert [(method, data.get("disable_notification")) for method, data in phone.calls] == [
        ("sendMessage", True), ("editMessageText", None), ("sendMessage", None)]


def test_silent_mode_setting_explains_sound_not_replacement(tmp_path):
    from tests.test_telegram_format_parity import _load_skill as load

    plugin, _transport = load()
    sections = []
    api = SimpleNamespace(
        get_state_dir=lambda: str(tmp_path), get_runtime_info=lambda: {"data_dir": str(tmp_path / "data")},
        register_supervised_task=lambda *a, **k: None, subscribe_event=lambda *a, **k: None,
        register_route=lambda *a, **k: None, log=lambda *a, **k: None,
        register_settings_section=lambda name, **kwargs: sections.append(kwargs))
    plugin.register_miniapp = lambda _api: None
    plugin.register(api)
    fields = [field for component in sections[0]["schema"]["components"] if component.get("type") == "form"
              for field in component["fields"]]
    (silent,) = [field for field in fields if field["name"] == "TELEGRAM_SILENT_MODE"]
    assert silent["label"] == "Silent mode"
    assert [option["label"] for option in silent["options"]] == ["Off", "On (without sound)"]
    assert silent["help"] == (
        "On sends replies, the final answer and progress without sound; a progress update replaces "
        "the one just before it. Questions, files and problems keep their sound.")


def test_completion_notifier_maps_persisted_summary_and_keeps_episodes_separate(bridge, monkeypatch):
    import sys
    from ouroboros.chat_chain import format_address, row_address

    b = bridge
    notifier = sys.modules[b.plugin.__package__ + ".lib.telegram_notifier"]
    summary = {"type": "task_summary", "summary_kind": "terminal_root_projection", "chat_id": 1,
               "task_id": "episode-1", "ts": "2026-10-16T12:00:00Z", "outcome_phase": "done"}
    monkeypatch.setattr(notifier, "_summary_ids_in_tail", lambda _api: [("episode-1", summary)])
    state = {"notified_task_ids": []}
    asyncio.run(notifier._check_tasks_notify(b.api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))
    summary2 = {**summary, "task_id": "episode-2"}
    monkeypatch.setattr(notifier, "_summary_ids_in_tail", lambda _api: [("episode-2", summary2)])
    asyncio.run(notifier._check_tasks_notify(b.api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, state, "en"))
    assert [method for method, _ in b.calls] == ["sendMessage", "sendMessage"]
    mapping = _state(b.plugin)
    assert mapping._reply_reference(b.api, 42, 101) == "task:episode-1"
    assert mapping._message_ref(b.api, 42, 101)["row_address"] == format_address(row_address(summary))
    assert mapping._reply_reference(b.api, 42, 102) == "task:episode-2"


def test_relay_success_silent_failure_one_line_and_its_paired_system_row_only_when_no_ack_carried_it(bridge):
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX, routing_refusal_cause

    b = bridge
    handle = b.plugin._make_outbound(b.api)
    ack = {"annotation_type": "routing_ack", "action": "steer_task", "client_message_id": "tg:42:10",
           "target": "work-1", "target_label": "Parser", "status": "delivered", "cause": ""}
    asyncio.run(handle(ack))
    assert not b.calls
    cause = routing_refusal_cause("steer_task", "rejected", "cancel_pending")
    refused = {**ack, "status": "rejected", "cause": cause, "routing_notice_carrier": "routing_ack"}
    asyncio.run(handle(refused))
    assert len(b.calls) == 1 and b.calls[0][1]["text"] == f"Parser · {cause}"
    assert "\n" not in b.calls[0][1]["text"]
    # The ACK is the carrier for an act that wears no owner message too; its paired System row is skipped
    # only when it says so, and a row that carries the notice itself (or any standalone one) still arrives.
    asyncio.run(handle({**refused, "client_message_id": AGENT_RECEIPT_ID_PREFIX + "1"}))
    row = {"role": "system", "system_type": "steer_not_delivered", "text": f"Parser · {cause}"}
    asyncio.run(handle({**row, "routing_notice_carrier": "routing_ack"}))
    assert [data["text"] for _, data in b.calls] == [f"Parser · {cause}"] * 2
    asyncio.run(handle({**row, "routing_notice_carrier": "system"}))
    asyncio.run(handle(row))
    assert len(b.calls) == 4 and all(method == "sendMessage" for method, _ in b.calls)


def test_relay_quotes_the_owners_words_and_files_above_the_note_verbatim(bridge):
    first = {"text": "Please deploy *now*", "attachments": [], "source": {"chat_id": 1, "label": "Main"}}
    second = {"text": "", "attachments": [{"name": "config.json"}], "source": {"chat_id": 7, "label": "Alpha"}}
    event = {"role": "assistant", "text": "Apply this constraint.", "chat_id": 7, "task_id": "receiver",
             "system_type": "relay_entry", "kind": "relay_entry",
             "relay": {"relay_id": "r", **first, "quotes": [first, second]}}
    asyncio.run(bridge.plugin._make_outbound(bridge.api)(event))
    [(method, data)] = bridge.calls
    assert method == "sendMessage"
    assert data["text"] == ("Your message in Main:\nPlease deploy *now*\n\n"
                            "Your message in Alpha:\nAttached: config.json\n\nApply this constraint.")
    assert not data.get("parse_mode"), "his words are sent verbatim, never re-read as markup"
    bridge.calls.clear()
    asyncio.run(bridge.plugin._make_outbound(bridge.api)({**event, "relay": {"relay_id": "r2", "text": "", "quotes": []}}))
    assert not bridge.calls, "a note between works without his words stays in its room, off the phone"


_MAP_FAULTS = [("write_text", 28), ("write_text", 13), ("replace", 28), ("replace", 13)]  # ENOSPC, EACCES


def _fail_reference_map(monkeypatch, operation, code):
    original = getattr(Path, operation)

    def fail(path, *args, **kwargs):
        if path.name == "message_refs.tmp":
            raise OSError(code, "injected derived-map failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, fail)


@pytest.mark.parametrize("operation,code", _MAP_FAULTS)
def test_owner_message_reaches_host_when_reference_map_cannot_be_saved(bridge, monkeypatch, operation, code):
    posts = _host_posts(bridge, monkeypatch)
    _fail_reference_map(monkeypatch, operation, code)
    _receive(bridge, _owner(text="Deliver my actual request"))
    [(path, payload)] = posts
    assert path == "/chat/inject" and payload["text"] == "Deliver my actual request"
    assert payload["client_message_id"] == "tg:42:50"
    assert bridge.api.logs and all(level == "warning" and "test-token" not in line
                                   and str(bridge.api.state_dir) not in line for level, line in bridge.api.logs)


@pytest.mark.parametrize("operation,code", _MAP_FAULTS)
def test_every_outbound_chunk_is_sent_when_reference_map_cannot_be_saved(bridge, monkeypatch, operation, code):
    text = "All words must reach the owner. " * 700
    expected = bridge.transport._chunk_raw_text(text.strip())  # the bridge strips before chunking
    assert len(expected) > 2
    _fail_reference_map(monkeypatch, operation, code)
    asyncio.run(bridge.plugin._make_outbound(bridge.api)({"text": text, "chat_id": 1, "task_id": "reply-work",
                "markdown": False, "row_address": "row:1@2026-10-17T08:00:00Z#123456789abc"}))
    assert [data["text"] for method, data in bridge.calls if method == "sendMessage"] == expected
    assert len(bridge.api.logs) == len(expected) and {level for level, _ in bridge.api.logs} == {"warning"}
    assert _state(bridge.plugin)._reply_reference(bridge.api, 42, 101) == ""


@pytest.mark.parametrize("notify,salvage", [("on", ""), ("off", "Preserved draft")])
def test_project_status_reply_names_work_not_the_unshown_main_answer(bridge, monkeypatch, notify, salvage):
    from ouroboros.chat_references import resolve_reply
    from ouroboros.task_results import write_task_result
    from ouroboros.chat_chain import format_address, row_address
    from supervisor import message_bus
    from tests.test_chat_attachments import _web_bridge

    b = bridge
    root = b.api.state_dir / "data"
    _web_bridge(root, monkeypatch)
    write_task_result(root, "finished-work", "completed", title="The finished work")
    row = message_bus.log_chat("system", 1, 0, "Project / Done", record_type="project_completion_summary",
                              message_meta={"task_id": "finished-work", "completion_answer": "AUTHOR ANSWER NOT SENT TO TELEGRAM"})
    address = format_address(row_address(row))
    b.plugin.merge_settings(b.api.state_dir, {"TELEGRAM_NOTIFY_TASKS": notify})
    event = {"role": "system", "system_type": "project_completion_summary", "kind": "project_completion_summary",
             "chat_id": 1, "task_id": "finished-work", "row_address": address, "text": "Project / Done",
             "completion_answer": row["completion_answer"],
             "completion_notice": {"neutral": True, "detail": "", "salvage": salvage}}
    asyncio.run(b.plugin._make_outbound(b.api)(event))
    assert len(b.calls) == 1 and "AUTHOR ANSWER" not in b.calls[0][1]["text"]
    posts = _host_posts(b, monkeypatch)
    _receive(b, _owner(reply=101))
    assert posts[0][0] == "/chat/inject" and posts[0][1]["reply_to"] == "task:finished-work"
    assert "task_id" not in posts[0][1]  # a reference, never a routing request
    assert resolve_reply(root, posts[0][1]["reply_to"])["kind"] == "work_card"
    assert resolve_reply(root, address)["snippet"] == row["completion_answer"]  # Main's separate visible answer
