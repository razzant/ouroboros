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
    client = b.transport.TelegramClient("test-token", on_message=lambda chat, mid: state._remember_message(b.api, "", chat, mid, record))
    last = asyncio.run(client.send_message(42, "word " * 2000, parse_mode=""))
    assert len(b.calls) >= 3 and last == 100 + len(b.calls)
    plugin, _transport = _load_skill()  # no in-memory map survives a plugin restart
    restored = _state(plugin)
    for mid in range(101, last + 1):
        assert restored._reply_reference(Api(b.api.state_dir), "", 42, mid) == record["row_address"]
    assert restored._reply_reference(b.api, "", 43, 101) == ""
    assert restored._reply_reference(b.api, "", 42, 9999) == ""


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
        state._remember_message(api, "", 42, mid, {"row_address": f"row-{mid}", "kind": "work_card", "task_id": f"task-{mid}"})
    state._remember_message(api, "", 42, 3, {"row_address": "updated-row", "kind": "message"})
    assert list(state._message_refs(api, "")) == ["42:4", "42:5", "42:3"]
    assert state._reply_reference(api, "", 42, 1) == ""
    assert state._reply_reference(api, "", 42, 4) == "task:task-4"
    assert state._reply_reference(api, "", 42, 3) == "updated-row"


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


def test_a_reply_to_a_bot_message_no_map_knows_is_an_ordinary_unquoted_turn_here(bridge, monkeypatch):
    """An evicted answer, /help or any bot text the maps never held: the Reply is this chat's ordinary turn,
    without a quote, a guessed room or a refusal (a command sent that way keeps its dispatch, as typed)."""
    b = bridge
    _state(b.plugin)._remember_message(b.api, "", 43, 101, {"row_address": "other-chat"})  # another chat's number
    posts = _host_posts(b, monkeypatch)
    _receive(b, _owner(reply=101))
    _receive(b, _owner(51, reply=999))
    _receive(b, _owner(52, "/status", reply=999))
    assert [(path, payload["text"], payload["chat_id"], "reply_to" in payload) for path, payload in posts] == [
        ("/chat/inject", "Explain this", 42, False), ("/chat/inject", "Explain this", 42, False),
        ("/chat/inject", "/status", 42, False)]
    assert [payload["client_message_id"] for _path, payload in posts] == ["tg:42:50", "tg:42:51", "tg:42:52"]
    assert all("owner_message_id" not in payload["transport"] for _path, payload in posts), "no exact turn"
    assert b.calls == [] and b.api.logs == [], "nothing is refused and no note is due: no old record holds them"


def _accept_owner(b, mid, *, text="Same owner words", chat=42):
    """Use the real canonical ingress writer; owner lookup must never use text."""
    from supervisor.message_bus import accept_local_message
    from ouroboros.chat_chain import format_address, row_address

    state = _state(b.plugin)
    state._owner_message_seen(b.api, "", chat, mid)
    fields = state._owner_ingress_fields(b.api, "", chat, _owner(mid, text, chat=chat), mid)
    receiver = SimpleNamespace(enqueue_local_message=lambda *args, **kwargs: None)
    row, _ = accept_local_message(receiver, b.api.state_dir / "data", text,
                                 chat_id=chat, user_id=chat, source="skill:telegram", **fields)
    return {"address": format_address(row_address(row)), "chat_id": chat, "role": "user",
            "author_label": "You", "kind": "message", "snippet": text, "ts": row["ts"], "task_id": ""}


@pytest.mark.parametrize("target", ["earlier", "other_chat", "unresolved"])
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
        assert _state(b.plugin)._message_ref(b.api, "", 42, 10)["row_address"] == earlier["address"]


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
    record = {**state._message_ref(b.api, "", 42, 10), "row_address": ref["address"]}
    state._remember_message(b.api, "", 42, 10, record)
    state._remember_message(b.api, "", 42, 100, {"row_address": "bot-row"})
    assert state._message_ref(b.api, "", 42, 11) == {}
    assert state._native_reply_id(b.api, "", 42, {"reply_to": ref}) == 10


@pytest.mark.parametrize("surface,extra", [
    ("photo", {"image_base64": "eA=="}),
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
    assert _state(b.plugin)._reply_reference(b.api, "", 42, 101) == "host-media-row"
    assert b.api.logs == []


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
    assert _state(b.plugin)._reply_reference(b.api, "", 42, 101) == "row-update"
    assert _state(b.plugin)._reply_reference(b.api, "", 42, 103) == "row-continue"
    assert _state(b.plugin)._reply_reference(b.api, "", 42, 104) == "row-origin"


def test_silent_mode_starts_new_message_when_native_reply_anchor_changes(bridge):
    b = bridge
    handle = _quiet(b, TELEGRAM_MIRROR_PROGRESS="on")
    ref = _accept_owner(b, 10)
    event = _note("First note", "same-episode", reply_to=ref)
    asyncio.run(handle(event))
    _state(b.plugin)._owner_message_seen(b.api, "", 42, 11)
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
    assert b.plugin._get_silent_msg(b.api, "", 42) == 0


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
    assert b.plugin._get_silent_msg(b.api, "", 42) == 0, "the final answer ends the progress run"


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
    _state(b.plugin)._remember_message(b.api, "", 42, 77, {
        "row_address": "old-reply", "chat_id": 1, "task_id": "t1", "kind": "message",
        "reply_to": "", "native_reply_id": 0})
    b.plugin._set_silent_msg(b.api, "", 42, 77)
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
        await handle(_host_note("Fallback: trying other/model", task_incident="model_lane_attempt"))
        await handle(_host_note("Fallback: other/model is now the active route; it answered."))
        await handle(_note("Reading the failing test first."))

    asyncio.run(run())
    assert [(method, data.get("disable_notification")) for method, data in b.calls] == [
        ("sendMessage", True), ("editMessageText", None), ("sendMessage", None), ("editMessageText", None), ("editMessageText", None)]
    assert b.calls[1][1]["message_id"] == b.calls[3][1]["message_id"] == b.calls[4][1]["message_id"] == "101"  # the first note, form-encoded


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
    _emit(agent, "Fallback: trying other/model", incident={"task_incident": "model_lane_attempt", "toast_once": "lane"})
    _emit(agent, "Fallback: other/model is now the active route; it answered.")  # ordinary progress: Silent mutes it
    for _ in range(4):
        assert events_chat_delivery._handle_send_message(events.get_nowait(), phone.ctx) is True
    assert [(method, data.get("disable_notification")) for method, data in phone.calls] == [
        ("sendMessage", True), ("editMessageText", None), ("sendMessage", None), ("editMessageText", None)]


def test_with_silent_mode_off_mirrored_route_rows_keep_the_phones_own_sound(phone, monkeypatch):  # noqa: F811
    """The same real chain with Silent mode off and progress mirrored, the owner's other existing choice: every
    note, the walk's ordinary outcome row included, is its own message with the phone's own notification. The
    outcome row adds no incident or toast; it is not promised silence."""
    from supervisor import events_chat_delivery
    from tests.test_progress_narration_voice import _agent, _emit

    phone.plugin.merge_settings(phone.api.state_dir, {"TELEGRAM_SILENT_MODE": "off", "TELEGRAM_MIRROR_PROGRESS": "on"})
    monkeypatch.setattr(events_chat_delivery, "_bound_project_chat_id", lambda *_: 0)
    phone.ctx.RUNNING["task-1"] = {"task": {"id": "task-1", "_attempt": 0}}
    agent, events = _agent()
    _emit(agent, "Running the tests")
    _emit(agent, "Fallback: trying other/model", incident={"task_incident": "model_lane_attempt", "toast_once": "lane"})
    _emit(agent, "Fallback: other/model is now the active route; it answered.")
    for _ in range(3):
        assert events_chat_delivery._handle_send_message(events.get_nowait(), phone.ctx) is True
    assert [(method, data.get("disable_notification")) for method, data in phone.calls] == [("sendMessage", None)] * 3
    assert phone.calls[-1][1]["text"].endswith("other/model is now the active route; it answered.")


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
    assert mapping._reply_reference(b.api, "", 42, 101) == "task:episode-1"
    assert mapping._message_ref(b.api, "", 42, 101)["row_address"] == format_address(row_address(summary))
    assert mapping._reply_reference(b.api, "", 42, 102) == "task:episode-2"


def test_relay_success_silent_failure_one_line_and_its_system_row_only_when_no_ack_carried_it(bridge):
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
    # An ACK that answers no chat row (an act wearing no owner message) names its System row the carrier and adds
    # no line; a host act's row beside a carrier ACK is skipped only when it says so, and a row that carries the
    # notice itself (or any standalone one) still arrives.
    asyncio.run(handle({**refused, "client_message_id": AGENT_RECEIPT_ID_PREFIX + "1",
                        "routing_notice_carrier": "system"}))
    row = {"role": "system", "system_type": "steer_not_delivered", "text": f"Parser · {cause}"}
    asyncio.run(handle({**row, "system_type": "task_not_started", "routing_notice_carrier": "routing_ack"}))
    assert [data["text"] for _, data in b.calls] == [f"Parser · {cause}"]
    asyncio.run(handle({**row, "routing_notice_carrier": "system"}))
    asyncio.run(handle(row))
    assert len(b.calls) == 3 and all(method == "sendMessage" for method, _ in b.calls)


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
    assert _state(bridge.plugin)._reply_reference(bridge.api, "", 42, 101) == ""


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


# --- the owner's own words and the bridge's own maps decide what a Reply names ------------------------------------

def test_a_reply_to_the_owners_own_words_is_never_refused_and_quotes_his_accepted_row(bridge, monkeypatch):
    b = bridge
    earlier = _accept_owner(b, 10)
    (b.api.state_dir / "message_refs.json").unlink()  # the map let it go; the accepted row stays
    posts = _host_posts(b, monkeypatch)
    _receive(b, {**_owner(60, "and on Windows too"), "reply_to_message": {"message_id": 10, "from": {"id": 42}}})
    _receive(b, _owner(11, "/help"), update_id=71)  # his own words the host never accepted
    _receive(b, {**_owner(61, "about that"), "reply_to_message": {"message_id": 11, "from": {"id": 42}}}, update_id=72)
    assert [payload.get("reply_to") for _path, payload in posts] == [earlier["address"], None]
    assert [method for method, _data in b.calls] == ["sendMessage"], "only /help's own text: nothing refused"


def test_a_reply_to_a_card_only_the_card_index_still_names_reaches_its_helper(bridge, monkeypatch):
    b = bridge
    _state(b.plugin)._set_subagent_msg(b.api, "", 42, "helper-1", 77)
    posts = _host_posts(b, monkeypatch)
    _receive(b, _owner(60, "about the helper", reply=77))
    [(_path, payload)] = posts
    assert payload["reply_to"] == "task:helper-1"
    facts = payload["transport"]
    assert (facts["reply_child_task_id"], facts["reply_card_message_id"], facts["owner_message_id"]) == ("helper-1", 77, 60)
    assert "bot_id" not in facts and b.calls == []  # a token naming no bot


def test_panic_names_what_the_map_holds_and_never_waits_on_the_store(bridge, monkeypatch):
    """Panic's own door keeps its reading: a Reply on it is what this bot's map names, read without the store."""
    b = bridge
    state = _state(b.plugin)
    state._remember_message(b.api, "", 42, 10, {"direction": "in", "chat_id": 42, "kind": "message",
                                                "client_message_id": "tg:42:10"})
    monkeypatch.setattr("supervisor.message_bus.accepted_chat_message",
                        lambda *_args, **_kwargs: pytest.fail("Panic reads no accepted row"))
    panic = {"message_id": 11, "text": "/panic", "chat": {"id": 42}, "from": {"id": 42},
             "reply_to_message": {"message_id": 10, "from": {"id": 42}}}
    assert state._owner_ingress_fields(b.api, "", 42, panic, 5) == {"client_message_id": "tg:42:11"}


# --- the owner bot's exact Reply: its Project's own turn, its strict anchor, its refusals ---------------------------

class _OwnerBotApi(Api):
    """Bot 123 (a numeric token id, as every real token has) with its host data beside its skill state."""

    def __init__(self, state_dir, data_dir):
        super().__init__(state_dir)
        self.data_dir = Path(data_dir)

    def get_runtime_info(self):
        return {"data_dir": str(self.data_dir)}

    def get_settings(self, _keys):
        return {"TELEGRAM_BOT_TOKEN": "123:test"}


@pytest.fixture(params=["all", "telegram_only"])
def owner_bot(tmp_path, monkeypatch, request):
    """The owner's private chat 42 with bot 123 behind the real bus, ingress, owner router and room rule, in each
    mirror mode. ``calls`` holds every Bot API request as ``(method, form, message id)``; ``replies`` queues a refusal
    for the next one. A Project's converted root began from the owner's Telegram message; its helper is admitted."""
    from ouroboros import projects_registry as pr
    from ouroboros.event_bus import CHAT_OUTBOUND
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result
    from supervisor import message_bus, queue

    from tests.test_chat_references import seed_bundled_bridge

    plugin, transport = _load_skill()
    data, skill = tmp_path / "data", tmp_path / "skill"
    data.mkdir()
    skill.mkdir()
    seed_bundled_bridge(data)  # only the bundled bridge's Replies have their room decided
    (skill / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": request.param}),
                                         encoding="utf-8")
    api = _OwnerBotApi(skill, data)
    calls, replies = [], []

    class Wire:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, url, *, data, files):
            form = dict(data or {})
            if replies:
                status, body = replies.pop(0)
                calls.append((url.rsplit("/", 1)[-1], form, 0))
                return httpx.Response(status, json=body)
            mid = int(form.get("message_id") or 0) or 100 + len(calls)  # an edit answers with its own message
            calls.append((url.rsplit("/", 1)[-1], form, mid))
            return httpx.Response(200, json={"ok": True, "result": {"message_id": mid}})

    monkeypatch.setattr(transport.httpx, "AsyncClient", Wire)
    handle = plugin._make_outbound(api)
    bus = message_bus.LocalChatBridge({})
    bus._broadcast_fn = lambda _frame: None

    events = []

    def publish(topic, event):
        events.append((topic, dict(event)))
        if topic == CHAT_OUTBOUND:
            asyncio.run(handle(dict(event)))

    for name, value in (("DATA_DIR", data), ("get_bridge", lambda: bus), ("load_state", lambda: {"owner_id": 42}),
                        ("publish_event", publish)):
        monkeypatch.setattr(message_bus, name, value)
    for name, value in (("DRIVE_ROOT", str(data)), ("RUNNING", {}), ("PENDING", []),
                        ("ADMISSION_RESERVATIONS", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    queued, accepted = [], []
    receiver = SimpleNamespace(enqueue_local_message=lambda text, **payload: queued.append({"text": text, **payload}))

    async def inject(_api, payload):  # /chat/inject's named acceptance; the chat is bound as the supervisor's update does
        bus.activate_update_transport({"chat_id": payload["chat_id"], "transport": payload["transport"]})
        row, _ = message_bus.accept_local_message(
            receiver, data, payload["text"], chat_id=payload["chat_id"], user_id=payload["user_id"],
            source="skill:telegram", sender_label=payload["sender_label"], transport=payload["transport"],
            client_message_id=payload["client_message_id"],
            **({"task_metadata": {"reply_to": payload["reply_to"]}} if payload.get("reply_to") else {}))
        accepted.append(row)

    monkeypatch.setattr(plugin, "_inject", inject)

    def says(mid, text, *, reply=0):
        """One owner message in chat 42, a Reply to bot message ``reply`` when given; its accepted row or None."""
        message = {"message_id": mid, "date": 2_000_000_000, "chat": {"id": 42, "type": "private"},
                   "from": {"id": 42, "first_name": "Owner"}, "text": text,
                   **({"reply_to_message": {"message_id": reply, "from": {"id": 123, "is_bot": True}}} if reply else {})}
        before = len(accepted)
        asyncio.run(plugin._handle_owner_message(api, plugin._telegram_client(api), {"message": message}, mid,
                                                 "full_access", "en"))
        return accepted[-1] if len(accepted) > before else None

    project = pr.create_project(data, "export-room", name="Export room")
    opening = says(40, "Plan the export as a project")
    origin = build_owner_message_ref(chat_id=42, client_message_id=opening["client_message_id"], ts=opening["ts"],
                                     text=opening["text"])
    write_task_result(data, "project-root", "running", root_task_id="project-root", chat_id=1, origin_message_ref=origin)
    pr.bind_task_to_project(data, "project-root", project["id"], origin={"absent": "system"})
    write_task_result(data, "helper", "running", delegation_role="subagent", parent_task_id="project-root",
                      root_task_id="project-root", chat_id=1)
    return SimpleNamespace(plugin=plugin, api=api, bus=bus, data=data, calls=calls, replies=replies, says=says,
                           queued=queued, accepted=accepted, project=project, origin=origin, state=_state(plugin),
                           events=events, mode=request.param, receiver=receiver)


def _helper_card(chain, event, text, *, fail_edit=False):
    """The helper's lifecycle card as its producers frame it, from its own task record (``subagent_message_meta``:
    lineage included), through the real send seam; its message id, or 0 when nothing reached Telegram."""
    from ouroboros.subagent_messages import subagent_message_meta
    from ouroboros.task_results import load_task_result
    from supervisor.message_bus import send_with_budget

    if fail_edit:  # its bubble is gone at Telegram: the card posts a fresh one below
        chain.replies.append((400, {"ok": False, "error_code": 400, "description": "Bad Request: message to edit not found"}))
    before = len(chain.calls)
    meta = subagent_message_meta(load_task_result(chain.data, "helper"), task_id="helper", event=event)
    send_with_budget(chain.project["chat_id"], text, is_progress=True, task_id="helper", progress_meta=meta)
    return chain.calls[-1][2] if len(chain.calls) > before else 0


def _sent_card(chain) -> dict:
    """The last helper card event the bus published, as the bridge received it."""
    return [event for _topic, event in chain.events if event.get("subagent_event")][-1]


_ROOT_ADDRESS = {"kind": "telegram", "conversation_id": "42", "bot_id": "123", "host_address": "root"}


def _dispatch(chain, row, monkeypatch):
    """The accepted row through the owner router as server.py hands it over: the turns it started, its source."""
    from ouroboros import server_owner_routing as routing

    started = []

    def elsewhere(*_args, **_kwargs):
        pytest.fail("the owner bot's exact Reply is never a running work's input")

    monkeypatch.setattr(routing, "_route_project_chat_to_running_task", elsewhere)
    monkeypatch.setattr(routing, "_route_main_followup", elsewhere)
    monkeypatch.setattr(routing, "addressable_roots", lambda _ctx: SimpleNamespace(rows=lambda *_a: [], gap=None))
    monkeypatch.setattr(routing, "_decision_turn_metadata", lambda _ctx, _room, _cmid, meta, _roots, **_kw: meta)
    [incoming] = [item for item in chain.queued if item["client_message_id"] == row["client_message_id"]]
    source = incoming["accepted_source_ref"]
    ctx = SimpleNamespace(DRIVE_ROOT=chain.data, handle_chat_direct=lambda *args, **kwargs: started.append((args, kwargs)))
    routing._route_owner_message(chain.bus, ctx, {
        "chat_id": 42, "text": row["text"], "log_text": row["text"], "image_caption": "", "image_data": None,
        "client_message_id": row["client_message_id"], "task_constraint": None, "origin_message_ref": source,
        "accepted_source_row": incoming["accepted_source_row"],
        "task_metadata": {**(incoming.get("task_metadata") or {}), "_host_operation": True},
        "source": "skill:telegram", "received_at": row["ts"]})
    return started, source


def _answer(chain, turn, source, text, room=None, metadata=None):
    """That turn's own answer through the real send seam: what reached Telegram. ``metadata``: what the router
    started the turn with (its own address among it), as its running task holds it."""
    from supervisor import queue
    from supervisor.message_bus import send_with_budget

    queue.RUNNING[turn] = {"task": {"id": turn, "origin_message_ref": source, "metadata": dict(metadata or {})}}
    before = len(chain.calls)
    send_with_budget(room or chain.project["chat_id"], text, task_id=turn, progress_meta={"origin_message_ref": source})
    return chain.calls[before:]


def test_a_reply_about_a_helper_runs_in_its_project_and_quotes_that_helpers_card(owner_bot, monkeypatch):
    """Both producers publish for real; the card was edited, then posted again below, then left the card index."""
    from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_SALVAGE
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.message_bus import send_with_budget
    from supervisor.terminal_delivery import project_terminal_result_event

    chain = owner_bot
    room = chain.project["chat_id"]
    old_card = _helper_card(chain, "scheduled", "🗓️ Scheduled subagent helper (reviewer): check the tables")
    assert _sent_card(chain)["transport"] == _ROOT_ADDRESS, "its root's bot and chat, and no anchor, reach the bridge"
    method, form, _mid = chain.calls[-1]
    assert (method, form["chat_id"], "reply_to_message_id" in form) == ("sendMessage", "42", False)
    assert _helper_card(chain, "running", "🟡 reviewer — running") == old_card, "an edit keeps its bubble"
    card = _helper_card(chain, "progress", "🟡 reviewer — reading the tables", fail_edit=True)
    assert card not in (0, old_card)
    chain.state._save_subagent_state(chain.api, {})  # the card index let it go; the larger reply map still holds it
    write_task_result(chain.data, "project-root", "completed", root_task_id="project-root", chat_id=1,
                      origin_message_ref=chain.origin)  # a helper's own notice reaches the phone once its root ended
    incident = project_terminal_result_event(chain.data, load_task_result(chain.data, "helper"), "helper",
                                             result_text="partial tables", terminal_origin=TERMINAL_ORIGIN_HOST_SALVAGE)
    assert (incident["chat_id"], incident["system_type"]) == (room, "terminal_incident")
    send_with_budget(incident["chat_id"], incident["text"], task_id="helper", role="system",
                     system_type=incident["system_type"], progress_meta=incident.get("progress_meta"))
    method, _form, notice = chain.calls[-1]
    assert method == "sendMessage" and notice not in (old_card, card)
    assert chain.events[-1][1]["transport"] == _ROOT_ADDRESS, "the helper's notice too, and nothing more"

    for mid, target, quoted in ((60, notice, card), (61, old_card, old_card)):
        row = chain.says(mid, f"Question {mid} about the helper", reply=target)
        assert row["reply_to"]["telegram_room_chat_id"] == room, "a confirmed Project target: its own turn there"
        assert row["transport"]["reply_child_task_id"] == "helper"
        [(args, kwargs)], source = _dispatch(chain, row, monkeypatch)
        assert args[0] == room and kwargs["task_metadata"]["project_id"] == chain.project["id"]
        assert kwargs["task_metadata"]["origin_message_ref"] == source
        [(sent, form, _mid)] = _answer(chain, f"turn-{mid}", source, f"Answer to {mid}",
                                       metadata=kwargs["task_metadata"])
        assert form["text"].endswith(f"Answer to {mid}")
        # On the helper's card he asked about (the notice's helper now, or the very card he replied to), strictly.
        assert (sent, form["reply_to_message_id"], form["allow_sending_without_reply"]) == ("sendMessage", quoted, False)


def test_a_reply_to_a_retired_project_card_is_refused_once_and_a_reply_to_that_refusal_is_ordinary(owner_bot, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import projects_registry as pr
    from ouroboros.gateway.host_service import _accepted_inbound, _operation_state

    chain = owner_bot
    card = _helper_card(chain, "scheduled", "🗓️ Scheduled subagent helper (reviewer): check the tables")
    pr.begin_project_deletion(chain.data, chain.project["id"])
    pr.complete_project_deletion(chain.data, chain.project["id"])
    row = chain.says(62, "Is it still running?", reply=card)
    assert row["reply_to"]["address"] == "task:helper" and row["reply_to"]["telegram_room_chat_id"] is None
    later = {"kind": "telegram", "conversation_id": "42", "bot_id": "999"}  # another bot's later message in this chat
    chain.bus.activate_update_transport({"chat_id": 42, "transport": later})
    started, _source = _dispatch(chain, row, monkeypatch)
    assert started == [], "never Main, never the retired room"
    [ack] = [event for _topic, event in chain.events if event.get("annotation_type") == "routing_ack"]
    assert ack["transport"] == {**_ROOT_ADDRESS, "host_address": "reply"}, "it answers in the bot and chat he wrote to"
    assert chain.bus.chat_transport(42) == later, "the chat's binding stays its own"
    method, form, refusal = chain.calls[-1]
    assert (method, form["text"], form["reply_to_message_id"]) == (
        "sendMessage", "Not started: the message or card you replied to is unavailable", 62)
    # The named operation ends where it was accepted, refused as its one notice said, after any restart too.
    host = SimpleNamespace(data_dir=chain.data)
    state = _operation_state(host, _accepted_inbound(host, 42, row["client_message_id"]))
    assert (state["status"], state["text"]) == ("failed", "Not started: the message or card you replied to is unavailable")
    assert len([call for call in chain.calls if call[1].get("text") == state["text"]]) == 1, "a single visible notice"
    before = len(chain.calls)
    again = chain.says(63, "Then do it in Main", reply=refusal)
    assert again is not None and not again.get("reply_to"), "the refusal names no row: this chat's ordinary turn, unquoted"
    assert chain.calls[before:] == [], "nothing refused and nothing noted"


def test_a_reply_to_ordinary_words_stays_this_chats_turn_with_its_usual_quote(owner_bot):
    from ouroboros.chat_references import exact_reply_transport
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result
    from supervisor.message_bus import send_with_budget

    chain = owner_bot
    write_task_result(chain.data, "main-turn", "running", chat_id=42, origin_message_ref=chain.origin)
    send_with_budget(42, "Here is the plan.", task_id="main-turn")
    row = chain.says(64, "and on Windows too", reply=chain.calls[-1][2])
    assert row["reply_to"]["task_id"] == "main-turn" and row["reply_to"]["telegram_room_chat_id"] == 0
    assert exact_reply_transport(row) == {}, "its turn keeps this chat's binding and the ordinary quote"
    write_task_result(chain.data, "next-turn", "running", chat_id=42, origin_message_ref=build_owner_message_ref(
        chat_id=42, client_message_id=row["client_message_id"], ts=row["ts"], text=row["text"]))
    send_with_budget(42, "On Windows too.", task_id="next-turn")
    method, form, _mid = chain.calls[-1]
    assert method == "sendMessage" and "reply_to_message_id" not in form  # his latest message: answered unquoted


# --- where a turn's deliveries go: the proven address of the owner message behind them --------------------------

def _began_with(chain, client_message_id, transport):
    """The Project root now began with this owner message, as the bundled bridge's /chat/inject accepted it."""
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result
    from supervisor import message_bus

    row, _ = message_bus.accept_local_message(
        chain.receiver, chain.data, "Plan the import as a project", chat_id=42, user_id=42, source="skill:telegram",
        sender_label="Telegram (Owner)", transport=transport, client_message_id=client_message_id)
    origin = build_owner_message_ref(chat_id=42, client_message_id=client_message_id, ts=row["ts"], text=row["text"])
    write_task_result(chain.data, "project-root", "running", root_task_id="project-root", chat_id=1,
                      origin_message_ref=origin)


@pytest.mark.parametrize("opening, bot_fact", [
    ("tg:123:42:41", {"bot_id": "123"}),  # this bot took it
    ("tg:999:42:41", {"bot_id": "999"}),  # another bot took it: only that bot carries the pair
    ("tg:123:42:41", {}),                 # accepted before the bridge sent its bot: the bot-qualified id proves it
    ("tg:42:41", {}),                     # a botless id proves no bot, and none is guessed
])
def test_a_helpers_pair_goes_only_through_the_bot_and_chat_that_took_its_works_first_message(owner_bot, opening,
                                                                                           bot_fact):
    chain = owner_bot
    _began_with(chain, opening, {"kind": "telegram", "conversation_id": "42", "sender_label": "Telegram (Owner)",
                                 **bot_fact})
    card = _helper_card(chain, "scheduled", "🗓️ Scheduled subagent helper (reviewer): check the tables")
    bot = opening.split(":")[1] if opening.count(":") == 3 else ""
    assert _sent_card(chain)["transport"] == ({**_ROOT_ADDRESS, "bot_id": bot} if bot else {})
    # The common mirror delivers it as before the address existed; Telegram-only carries it only where it was asked.
    assert bool(card) is (chain.mode == "all" or bot == "123")


@pytest.mark.parametrize("pressed, quoted", [
    ("old_card", 90),       # a card saved before this version and no longer in the card index: that very card
    ("notice", 90),         # a notice about its helper: the helper's card, however old its record
    ("old_notice", 90),     # a notice saved before it named its helper, its card gone from the index: that card
    ("lonely_notice", 66),  # no card of that helper is known any more: his own message, nothing guessed
    ("root_row", 66),       # the root's own row beside the root's completion card: no helper, the owner's message
])
def test_a_reply_to_an_old_card_or_its_notice_is_answered_on_what_the_maps_still_hold(owner_bot, monkeypatch,
                                                                                    pressed, quoted):
    chain = owner_bot
    room = chain.project["chat_id"]
    work = "project-root" if pressed == "root_row" else "helper"
    if pressed != "lonely_notice":  # the record shape the bridge saved for a card before 1.4.18
        chain.state._remember_message(chain.api, "123", 42, 90, {
            "row_address": "", "chat_id": room, "task_id": work, "kind": "work_card", "reply_to": "",
            "native_reply_id": 0})
    chain.state._remember_message(chain.api, "123", 42, 91, {
        "row_address": "", "chat_id": room, "task_id": work, "kind": "terminal_incident", "reply_to": "",
        "native_reply_id": 0, **({"child_task_id": work} if pressed in ("notice", "lonely_notice") else {})})
    row = chain.says(66, "What happened there?", reply=90 if pressed == "old_card" else 91)
    assert row["reply_to"]["address"] == f"task:{work}" and row["reply_to"]["telegram_room_chat_id"] == room
    [(args, kwargs)], source = _dispatch(chain, row, monkeypatch)
    assert args[0] == room
    [(sent, form, _mid)] = _answer(chain, f"turn-{pressed}", source, "It stopped on the tables.",
                                   metadata=kwargs["task_metadata"])
    assert (sent, form["reply_to_message_id"], form["allow_sending_without_reply"]) == ("sendMessage", quoted, False)


@pytest.mark.parametrize("turn", ["live", "saved"])
def test_an_exact_replys_turn_takes_its_own_address_first_and_leaves_the_rooms_binding(owner_bot, monkeypatch, turn):
    from ouroboros.event_bus import CHAT_OUTBOUND, CHAT_PHOTO, CHAT_QUIZ
    from ouroboros.owner_quiz import record_asked
    from ouroboros.task_results import write_task_result
    from supervisor import queue
    from supervisor.message_bus import send_with_budget

    chain = owner_bot
    room = chain.project["chat_id"]
    card = _helper_card(chain, "scheduled", "🗓️ Scheduled subagent helper (reviewer): check the tables")
    bound = {"kind": "slack", "conversation_id": "C1"}  # another connector's binding of the Project room
    chain.bus.activate_update_transport({"chat_id": room, "transport": bound})
    row = chain.says(67, "Which tables?", reply=card)
    [(args, kwargs)], source = _dispatch(chain, row, monkeypatch)
    assert args[0] == room
    task_id, begun = f"turn-{turn}", dict(kwargs["task_metadata"])  # the turn carries its address from its start
    if turn == "live":
        queue.RUNNING[task_id] = {"task": {"id": task_id, "origin_message_ref": source, "metadata": begun}}
    else:  # after a restart: its stored result keeps what it began with
        write_task_result(chain.data, task_id, "completed", chat_id=room, origin_message_ref=source, metadata=begun)
    before, published = len(chain.calls), len(chain.events)
    origin = {"progress_meta": {"origin_message_ref": source}} if turn == "live" else {}  # the live turn's own frame
    send_with_budget(room, "The orders tables.", task_id=task_id, **origin)
    chain.bus.send_photo(room, b"\x89PNG\r\n\x1a\n" + b"0" * 16, caption="the tables", task_id=task_id)
    if turn == "live":
        record_asked(chain.data, task_id, quiz_id="q1", question="Orders or users?", options=["Orders", "Users"],
                     assumption="orders meanwhile")
        assert chain.bus.send_quiz(room, "q1", "Orders or users?", [{"label": "Orders"}, {"label": "Users"}],
                                   assumption="orders meanwhile", task_id=task_id)[0]
    exact = {**_ROOT_ADDRESS, "host_address": "reply",
             "exact_reply": {"owner_message_id": 67, "card_message_id": card, "child_task_id": "helper"}}
    sent = {topic: event["transport"] for topic, event in chain.events[published:]
            if topic in (CHAT_OUTBOUND, CHAT_PHOTO, CHAT_QUIZ)}
    assert sent == {CHAT_OUTBOUND: exact, CHAT_PHOTO: exact, **({CHAT_QUIZ: exact} if turn == "live" else {})}
    assert chain.bus.chat_transport(room) == bound, "the room keeps its own binding"
    method, form, _mid = chain.calls[before]
    assert (method, form["reply_to_message_id"], form["allow_sending_without_reply"]) == ("sendMessage", card, False)


@pytest.mark.parametrize("root_state", ["live", "saved"])
def test_d8_helper_uses_its_project_reply_roots_telegram_destination(owner_bot, monkeypatch, root_state):
    from ouroboros.event_bus import CHAT_OUTBOUND, CHAT_PHOTO, CHAT_QUIZ
    from ouroboros.owner_quiz import record_asked
    from ouroboros.subagent_messages import subagent_message_meta
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import queue
    from supervisor.message_bus import send_with_budget

    chain = owner_bot
    room = chain.project["chat_id"]
    slack = {"kind": "slack", "conversation_id": "C1"}
    chain.bus.activate_update_transport({"chat_id": room, "transport": slack})
    card = _helper_card(chain, "scheduled", "🗓️ Scheduled subagent helper (reviewer): check the tables")
    [(_, kwargs)], source = _dispatch(chain, chain.says(67, "Which tables?", reply=card), monkeypatch)
    metadata = dict(kwargs["task_metadata"])
    if root_state == "live":
        queue.RUNNING["reply-parent"] = {"task": {"id": "reply-parent", "metadata": metadata, "chat_id": room}}
    else:
        write_task_result(chain.data, "reply-parent", "completed", chat_id=room,
                          origin_message_ref=source, metadata=metadata)
    write_task_result(chain.data, "reply-child", "running", chat_id=room, delegation_role="subagent",
                      root_task_id="reply-parent", parent_task_id="reply-parent")

    chain.plugin.merge_settings(Path(chain.api.get_state_dir()), {"TELEGRAM_MIRROR_PROGRESS": "on"})
    start, sent_before = len(chain.events), len(chain.calls)
    send_with_budget(room, "Reading the cells.", is_progress=True, task_id="reply-child")
    meta = subagent_message_meta(load_task_result(chain.data, "reply-child"), task_id="reply-child", event="scheduled")
    send_with_budget(room, "🗓️ Scheduled subagent reply-child (reader): read cells", is_progress=True,
                     task_id="reply-child", progress_meta=meta)
    chain.bus.send_photo(room, b"\x89PNG\r\n\x1a\n" + b"0" * 16, caption="the cells", task_id="reply-child")
    record_asked(chain.data, "reply-child", quiz_id="q-child", question="Which columns?", options=["A", "B"],
                 assumption="A meanwhile")
    assert chain.bus.send_quiz(room, "q-child", "Which columns?", [{"label": "A"}, {"label": "B"}],
                               assumption="A meanwhile", task_id="reply-child")[0]
    events = [(topic, event) for topic, event in chain.events[start:] if topic in (CHAT_OUTBOUND, CHAT_PHOTO, CHAT_QUIZ)]
    assert [topic for topic, _event in events] == [CHAT_OUTBOUND, CHAT_OUTBOUND, CHAT_PHOTO, CHAT_QUIZ]
    bare = {**_ROOT_ADDRESS, "host_address": "reply"}
    for topic, event in events:
        destination = event["transport"]
        assert destination.get("telegram_root", destination) == bare
        if topic == CHAT_PHOTO:
            asyncio.run(chain.plugin._make_photo(chain.api)(event))
        elif topic == CHAT_QUIZ:
            asyncio.run(chain.plugin._make_quiz(chain.api)(event))
    sent = chain.calls[sent_before:]
    assert [method for method, _form, _mid in sent] == ["sendMessage", "sendMessage", "sendPhoto", "sendMessage"]
    assert all(form["chat_id"] == "42" and "reply_to_message_id" not in form for _method, form, _mid in sent)
    assert chain.bus.chat_transport(room) == slack


def test_long_quiz_keeps_answer_door_and_earlier_parts_carry_references(owner_bot, monkeypatch):
    """The question as the bus publishes it (``LocalChatBridge.send_quiz``): every part names its stored row, so a
    Reply to an earlier part quotes the question as an ordinary reference here (its card is the answer's door),
    and a Reply to the keyboard message is its answer."""
    from ouroboros.event_bus import CHAT_QUIZ
    from ouroboros.owner_quiz import record_asked

    chain = owner_bot
    room = chain.project["chat_id"]
    chain.bus.activate_update_transport({"chat_id": room, "transport": {"kind": "telegram", "conversation_id": "42"}})
    question = "Long question " * 900
    record_asked(chain.data, "project-root", quiz_id="q-long", question=question, options=["A", "B"],
                 assumption="A meanwhile")
    assert chain.bus.send_quiz(room, "q-long", question, [{"label": "A"}, {"label": "B"}], assumption="A meanwhile",
                               task_id="project-root")[0]
    [event] = [event for topic, event in chain.events if topic == CHAT_QUIZ]
    assert event["row_address"].startswith(f"row:{room}@"), "the publication names its stored row"
    before = len(chain.calls)
    asyncio.run(chain.plugin._make_quiz(chain.api)(event))
    sent = chain.calls[before:]
    assert len(sent) > 2
    quiz, heading = chain.plugin.telegram_quiz, chain.plugin.origin_label(event, "en") + "\n"
    explained = []
    for method, form, _mid in sent:  # every part: this chat, its heading once, within Telegram's UTF-16 limit
        assert method == "sendMessage" and form["chat_id"] == "42"
        assert form["text"].startswith(heading) and form["text"].count(heading) == 1
        assert chain.plugin._u16len(form["text"]) <= 4096
        if "reply_markup" not in form:
            explained.append(form["text"][len(heading):])
    expected = quiz.render_quiz_text(question, ["A", "B"], "", "A meanwhile")
    assert "".join("".join(part.split()) for part in explained) == "".join(expected.split()), "the whole text"
    assert sent[-1][1]["text"] == heading + "1. A\n2. B\n" + quiz.hint("en") and "reply_markup" in sent[-1][1]
    parts = [mid for _method, _form, mid in sent]
    assert quiz.quiz_for_message(chain.api, "123", 42, parts[-1])["text"] == heading + "1. A\n2. B"
    assert {chain.state._reply_reference(chain.api, "123", 42, mid) for mid in parts} == {event["row_address"]}
    row = chain.says(70, "Please explain", reply=parts[0])
    assert row["reply_to"]["address"] == event["row_address"] and row["reply_to"]["kind"] == "question"
    assert row["reply_to"]["telegram_room_chat_id"] == 0, "an earlier part is this chat's ordinary reference"
    posts = []

    async def post(_api, path, payload):
        posts.append((path, payload))
        return 200, {"ok": True}

    monkeypatch.setattr(chain.plugin, "_host_post", post)
    assert chain.says(71, "B", reply=parts[-1]) is None, "the keyboard message's Reply is its answer, no turn"
    assert [(path, payload["decision_id"], payload["comment"]) for path, payload in posts] == [
        ("/chat/decision", "quiz:project-root:q-long", "B")]


def test_an_exact_replys_progress_goes_where_its_answer_does_without_its_anchor(owner_bot, monkeypatch):
    """Its room is another connector's: the turn's progress still reaches the phone through the bot and chat that took
    the Reply, as a progress note (no strict anchor, so Silent mode's replacement and the progress toggle hold)."""
    from ouroboros.event_bus import CHAT_OUTBOUND
    from supervisor import queue
    from supervisor.message_bus import send_with_budget

    chain = owner_bot
    room = chain.project["chat_id"]
    chain.plugin.merge_settings(Path(chain.api.get_state_dir()), {"TELEGRAM_MIRROR_PROGRESS": "on"})
    card = _helper_card(chain, "scheduled", "🗓️ Scheduled subagent helper (reviewer): check the tables")
    row = chain.says(68, "How far is it?", reply=card)
    [(_args, kwargs)], source = _dispatch(chain, row, monkeypatch)
    queue.RUNNING["turn-progress"] = {"task": {"id": "turn-progress", "origin_message_ref": source,
                                               "metadata": dict(kwargs["task_metadata"])}}
    chain.bus.activate_update_transport({"chat_id": room, "transport": {"kind": "slack", "conversation_id": "C1"}})
    before, published = len(chain.calls), len(chain.events)
    send_with_budget(room, "Reading the orders table.", is_progress=True, task_id="turn-progress")
    [event] = [event for topic, event in chain.events[published:] if topic == CHAT_OUTBOUND]
    assert event["transport"] == {**_ROOT_ADDRESS, "host_address": "reply"}, "its bot and chat, never the anchor"
    [(method, form, _mid)] = chain.calls[before:]
    assert (method, form["chat_id"], "reply_to_message_id" in form) == ("sendMessage", "42", False)


def test_a_helpers_card_reaches_the_phone_beside_another_connectors_binding_of_its_room(owner_bot):
    chain = owner_bot
    room = chain.project["chat_id"]
    slack = {"kind": "slack", "conversation_id": "C1"}
    chain.bus.activate_update_transport({"chat_id": room, "transport": slack})
    card = _helper_card(chain, "scheduled", "🗓️ Scheduled subagent helper (reviewer): check the tables")
    assert _sent_card(chain)["transport"] == {**slack, "telegram_root": _ROOT_ADDRESS}, "the binding keeps its card"
    assert card and chain.calls[-1][1]["chat_id"] == "42", "and the phone gets it where the work's first message was"
    assert chain.bus.chat_transport(room) == slack


def test_an_exact_replys_completion_notice_goes_where_it_was_asked_and_never_through_another_bot(owner_bot,
                                                                                               monkeypatch):
    import sys

    from ouroboros.task_results import write_task_result

    chain = owner_bot
    notifier = sys.modules[chain.plugin.__package__ + ".lib.telegram_notifier"]
    card = _helper_card(chain, "scheduled", "🗓️ Scheduled subagent helper (reviewer): check the tables")
    row = chain.says(69, "Finish it", reply=card)
    [(_args, kwargs)], source = _dispatch(chain, row, monkeypatch)
    write_task_result(chain.data, "turn-done", "completed", chat_id=chain.project["chat_id"], origin_message_ref=source,
                      metadata=dict(kwargs["task_metadata"]))
    from ouroboros import dialogue_store

    dialogue_store.append(chain.data, "chat", {  # the root's terminal summary row, as the tail reads it
        "ts": "2026-11-21T00:00:00+00:00", "type": "task_summary", "direction": "system", "task_id": "turn-done",
        "chat_id": chain.project["chat_id"], "text": "Done.", "summary_kind": notifier._ROOT_SUMMARY_KIND,
        "outcome_final": True})
    pushed = []

    async def push(_api, chat_id, _text, *, trust_env=False, event=None):
        pushed.append((chat_id, event))
        return "sent", None

    real = notifier._push_notification
    monkeypatch.setattr(notifier, "_push_notification", push)
    asyncio.run(notifier._check_tasks_notify(chain.api, {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_NOTIFY_TASKS": "on"}, 42,
                                             {"notified_task_ids": []}, "en"))
    [(chat_id, event)] = pushed
    assert (chat_id, event["transport"]) == (42, {**_ROOT_ADDRESS, "host_address": "reply"})
    before = len(chain.calls)
    outcome, _exc = asyncio.run(real(chain.api, 42, "Done.", event={
        **event, "transport": {**event["transport"], "bot_id": "999"}}))
    assert (outcome, chain.calls[before:]) == ("skipped", []), "another bot never carries that turn's notice"


def test_typing_for_a_proven_address_shows_only_in_its_bot_and_chat(owner_bot):
    chain = owner_bot
    room = chain.project["chat_id"]
    exact = {**_ROOT_ADDRESS, "host_address": "reply", "exact_reply": {"owner_message_id": 60}}
    targets = chain.plugin._typing_cue(chain.api)._targets
    assert targets({"chats": [{"chat_id": room, "transport": exact}]}) == {42}
    assert targets({"chats": [{"chat_id": room, "transport": {**exact, "bot_id": "999"}}]}) == set()
    assert targets({"chats": [{"chat_id": room, "transport": {**exact, "conversation_id": "43"}}]}) == set()
    ordinary = {"kind": "telegram", "conversation_id": "42"}  # an ordinary turn's binding types as before
    assert targets({"chats": [{"chat_id": 42, "transport": ordinary}]}) == {42}
    cue = chain.plugin._typing_cue(chain.api)
    cue._rescope()  # as each read does: the token whose bot chose the chat
    asyncio.run(cue._send(42))  # its own client, of that token, sends the action
    method, form, _mid = chain.calls[-1]
    assert (method, form["chat_id"], form["action"]) == ("sendChatAction", "42", "typing")
