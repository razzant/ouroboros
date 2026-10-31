import asyncio, importlib.util, json, sys, types
from pathlib import Path

import pytest


def _load_plugin():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    pkg = types.ModuleType("tg_sub_test"); pkg.__path__ = [str(root)]
    sys.modules["tg_sub_test"] = pkg
    spec = importlib.util.spec_from_file_location("tg_sub_test.plugin", root / "plugin.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m


class _Api:
    def __init__(self, sd): self.sd = Path(sd)
    def get_state_dir(self): return str(self.sd)
    def get_settings(self, keys): return {"TELEGRAM_BOT_TOKEN": "token"}
    def log(self, *a, **k): pass


class _Rec:
    sent = []; edited = []; _id = [1000]
    muted = []
    def __init__(self, token, **kwargs): self.on_message = kwargs.get("on_message")
    async def send_message(self, chat_id, text, parse_mode="HTML", *, disable_notification=False):
        _Rec._id[0] += 1; _Rec.sent.append((chat_id, text, parse_mode, _Rec._id[0]))
        if disable_notification:
            _Rec.muted.append(_Rec._id[0])
        if self.on_message: self.on_message(chat_id, _Rec._id[0])
        return _Rec._id[0]
    async def edit_message_text(self, chat_id, message_id, text, parse_mode="HTML"):
        _Rec.edited.append((chat_id, message_id, text, parse_mode))
        if self.on_message: self.on_message(chat_id, message_id)
        return True
    async def send_chat_action(self, chat_id, action="typing"): pass


def _setup(tmp_path, monkeypatch, **settings):
    plugin = _load_plugin()
    sd = tmp_path / "state"; sd.mkdir(parents=True, exist_ok=True)
    base = {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": "all"}; base.update(settings)
    (sd / "settings.json").write_text(json.dumps(base), encoding="utf-8")
    monkeypatch.setattr(plugin, "TelegramClient", _Rec)
    _Rec.sent = []; _Rec.edited = []; _Rec._id = [1000]
    _Rec.muted = []
    return plugin, _Api(sd)


@pytest.mark.parametrize("silent", ["off", "on"])
def test_subagent_gets_one_edited_bubble(tmp_path, monkeypatch, silent):
    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_SILENT_MODE=silent)
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "Subagent abc scheduled (researcher)", "is_progress": True,
                        "subagent_event": "scheduled", "subagent_task_id": "abc123", "subagent_role": "researcher"}))
    assert len(_Rec.sent) == 1 and not _Rec.edited
    first_id = _Rec.sent[0][3]
    # A helper's card is technical progress: Silent mode sends it without sound.
    assert _Rec.muted == ([first_id] if silent == "on" else [])
    asyncio.run(handle({"text": "Subagent abc completed (researcher)", "is_progress": True,
                        "subagent_event": "completed", "subagent_task_id": "abc123", "subagent_role": "researcher", "cost_usd": 0.04}))
    assert len(_Rec.sent) == 1                       # NO second message
    assert len(_Rec.edited) == 1 and _Rec.edited[0][1] == first_id
    assert "done" in _Rec.edited[0][2] and "$0.04" in _Rec.edited[0][2]


def test_generic_progress_hidden_by_default(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch)
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "Thinking...", "is_progress": True}))
    assert not _Rec.sent and not _Rec.edited


def test_mirror_progress_off_gates_firehose(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_MIRROR_PROGRESS="off")
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "Thinking...", "is_progress": True}))
    assert not _Rec.sent and not _Rec.edited         # explicit opt-out drops it


@pytest.mark.parametrize("mirror", ["off", "on"])
def test_a_cards_placed_row_is_a_progress_note_here(tmp_path, monkeypatch, mirror):
    """The web shows a card's host rows inside the task card; this transport has none, so a
    root's own placed row (a late review settlement, a custody note) follows the owner's
    progress toggle. A child's row still waits for its root, toggle or not."""
    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_MIRROR_PROGRESS=mirror)
    handle = plugin._make_outbound(api)
    row = {"text": "Reviewers rejected the answer after it was delivered.", "role": "system",
           "system_type": "acceptance_late_settlement", "card_row": "reviews", "task_id": "root-1"}
    asyncio.run(handle(row))
    child = {**row, "delegation_role": "subagent", "root_task_id": "root-done"}
    asyncio.run(handle(child))  # its root's outcome is unreadable: the row waits
    assert [sent[1] for sent in _Rec.sent] == ([row["text"]] if mirror == "on" else [])
    # Once the root has finished, a child's late row goes as it always did, toggle or not.
    from ouroboros.task_results import task_result_path

    data = tmp_path / "data"
    api.get_runtime_info = lambda: {"data_dir": str(data)}
    task_result_path(data, "root-done").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    _Rec.sent = []
    asyncio.run(handle(child))
    assert [sent[1] for sent in _Rec.sent] == [row["text"]]


@pytest.mark.parametrize("mirror", [None, "off", "on"])
def test_wrap_up_receipt_row_goes_while_other_card_rows_follow_the_toggle(tmp_path, monkeypatch, mirror):
    """#1564: the host types Wrap up's saved-work row as a terminal receipt, so it reaches
    this chat with progress mirroring off; an untyped custody row and progress do not."""
    plugin, api = _setup(tmp_path, monkeypatch, **({} if mirror is None else {"TELEGRAM_MIRROR_PROGRESS": mirror}))
    handle = plugin._make_outbound(api)
    row = {"role": "system", "system_type": "custody_notice", "card_row": "timeline", "task_id": "root-1"}
    receipt = {**row, "text": "Work saved at Wrap up:\n- Subtask kid (Stopped): its text is saved",
               "card_row_id": "cascade:root-1:ci_1:retained_work", "terminal_receipt": "retained_work"}
    custody = {**row, "text": "Open delegated execution: run-x.", "card_row_id": "final:root-1:abc:custody_notice"}
    for event in (receipt, custody, {"text": "Thinking...", "is_progress": True}):
        asyncio.run(handle(event))
    sent = [entry[1] for entry in _Rec.sent]
    assert sent[0] == receipt["text"]
    assert (custody["text"] in sent) is (mirror == "on")
    if mirror != "on":
        assert sent == [receipt["text"]]


def test_real_reply_is_sent(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch)
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "Here is your answer", "is_progress": False}))
    assert len(_Rec.sent) == 1 and "answer" in _Rec.sent[0][1]


@pytest.mark.parametrize("notify", ["off", "on"])
@pytest.mark.parametrize("silent", [None, "off", "on"], ids=["default", "off", "on"])
def test_a_terminal_answer_reaches_telegram_alone_and_keeps_every_reply(tmp_path, monkeypatch, silent, notify):
    """The real terminal delivery sends the model's answer alone, as its own message: the
    host notice stays a field of the result and never becomes a second message, and
    Silent mode mutes the replies without replacing any of them; additional completion
    notices have their own opt-in toggle."""
    from collections import deque
    import queue

    from ouroboros.task_finalization import prepare_terminal_send_event, register_final_answer_owed
    from ouroboros.tools.control_runtime import _send_user_message
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery, message_bus
    from supervisor.terminal_delivery import pending_deliveries

    settings = {"TELEGRAM_NOTIFY_TASKS": notify}
    if silent is not None:
        settings["TELEGRAM_SILENT_MODE"] = silent
    plugin, api = _setup(tmp_path, monkeypatch, **settings)
    handle = plugin._make_outbound(api)
    published, frames = [], []

    def publish(_topic, event):
        published.append(dict(event))
        asyncio.run(handle(event))

    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", publish)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    host = types.SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl,
                                 send_with_budget=message_bus.send_with_budget)
    task = {"id": "native-reply", "type": "task", "chat_id": 1, "_is_direct_chat": True}
    events = queue.Queue()
    tool_ctx = types.SimpleNamespace(current_chat_id=1, task_id=task["id"], task_metadata={},
                                     event_queue=events, pending_events=[], drive_root=tmp_path,
                                     drive_logs=lambda: tmp_path / "logs")
    ongoing = ["I found the source.", "I am checking the result."]
    for text in ongoing:
        assert "sent to owner chat" in _send_user_message(tool_ctx, text)
        delivery._handle_send_message(events.get_nowait(), host)
    assert not tool_ctx.pending_events
    assert all(row["system_type"] == "proactive_message" for row in published)
    assert all("task_terminal_status" not in row for row in published)
    assert _Rec.edited == []

    answer, notice = "Complete model answer: λ", "Plan review stayed open; inspect the task details."
    event = prepare_terminal_send_event(tmp_path, task, answer, {
        "terminal_origin": "model_final", "terminal_host_notice": notice,
    }, {"type": "send_message", "task_id": task["id"], "chat_id": 1,
        "text": answer, "log_text": answer, "format": "markdown"}, presence=False)
    assert "terminal_host_notice" not in event
    register_final_answer_owed(task, event, env_drive_root=tmp_path)
    delivery_id = event["delivery_id"]
    delivery._handle_send_message(event, host)
    sent_counts = len(_Rec.sent), len(_Rec.edited), len(published)
    delivery._handle_send_message(event, host)  # Buffered final after its live copy.
    assert sent_counts == (len(_Rec.sent), len(_Rec.edited), len(published))
    assert event["text"] == event["log_text"] == answer and event["delivery_id"] == delivery_id
    assert [(row["role"], row["content"]) for row in frames] == [
        *(('assistant', text) for text in ongoing), ('assistant', answer),
    ]
    assert pending_deliveries(tmp_path) == []
    visible = {row[3]: row[1] for row in _Rec.sent}
    assert list(visible.values()) == [*ongoing, answer]
    assert all(notice not in text for text in visible.values())
    assert _Rec.muted == (list(visible) if silent == "on" else [])
    # A host notice keeps its sound and a later reply is again its own message.
    delivery._handle_send_message({"type": "send_message", "chat_id": 1,
                                   "text": "Separate host receipt", "role": "system",
                                   "system_type": "terminal_incident"}, host)
    receipt_id = _Rec.sent[-1][3]
    assert receipt_id not in _Rec.muted
    assert "sent to owner chat" in _send_user_message(tool_ctx, "A later ongoing reply.")
    delivery._handle_send_message(events.get_nowait(), host)
    assert _Rec.sent[-1][1] == "A later ongoing reply." and _Rec.edited == []
    assert plugin._get_silent_msg(api, 42) == 0


def test_routing_refusal_speaks_in_the_third_person(tmp_path, monkeypatch):
    """astra A-miss1: the mirror's routing refusal names no narrator."""
    plugin, api = _setup(tmp_path, monkeypatch)
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"annotation_type": "routing_ack", "status": "needs_manual_target",
                        "options": [{"task_id": "t1", "label": "Fix the parser"}]}))
    (message,) = [row[1] for row in _Rec.sent]
    lines = message.splitlines()
    assert lines[0] == "Choose a target for the last message:"
    assert "1. Fix the parser" in lines
    assert not any(line.startswith("I ") for line in lines) and "I couldn't" not in message


@pytest.mark.parametrize("silent", ["off", "on"])
def test_a_landed_receipt_stays_silent_unless_a_file_was_not_attached(tmp_path, monkeypatch, silent):
    """A landed act publishes nothing here either; only a file the host could not attach earns one
    plain line, sent fresh: it never edits or takes over silent mode's tracked answer."""
    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_SILENT_MODE=silent)
    plugin._set_silent_msg(api, 42, 555)
    handle = plugin._make_outbound(api)
    ack = {"annotation_type": "routing_ack", "status": "delivered", "action": "steer_task", "target": "t1",
           "attachment_manifest": [{"ordinal": 0, "label": "a.png", "status": "staged"}]}
    asyncio.run(handle(ack))
    assert _Rec.sent == []
    ack["attachment_manifest"].append({"ordinal": 1, "label": "b.pdf", "status": "rejected", "reason": "source_missing"})
    asyncio.run(handle(ack))
    assert [(row[1], row[2]) for row in _Rec.sent] == [("Not attached to the task: b.pdf", "")]
    assert _Rec.edited == [] and plugin._get_silent_msg(api, 42) == 555


def test_the_not_attached_line_speaks_the_install_language(tmp_path, monkeypatch):
    """The bridge composes these lines itself (the not-attached line and the picker's own lines), so
    one message reads its own text table in the install's language; target labels and file names
    stay as the host and the owner gave them."""
    from ouroboros import i18n_memory as memory

    plugin, api = _setup(tmp_path, monkeypatch)
    data = tmp_path / "data"
    plugin.telegram_i18n.configure(data)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    memory.update_memory(data, "ru", lambda doc: (memory.apply_generated(doc, {
        "code:tg.menu.attachments_not_attached": {"text": "Не приложено к задаче: {names}"},
        "code:tg.menu.routing_choose_target": {"text": "Куда отправить последнее сообщение:"},
        "code:tg.menu.routing_more_targets": {"text": "…и ещё {count} в веб-чате."},
        "code:tg.menu.routing_reply_with_number": {"text": "Ответьте номером или выберите вариант в веб-чате."},
    }, model="test"), doc)[1], create=True)
    handle = plugin._make_outbound(api)
    rejected = [{"ordinal": 0, "label": "отчёт.pdf", "status": "rejected", "reason": "source_missing"}]
    asyncio.run(handle({"annotation_type": "routing_ack", "status": "delivered", "target": "t1",
                        "attachment_manifest": rejected}))
    assert [row[1] for row in _Rec.sent] == ["Не приложено к задаче: отчёт.pdf"]
    asyncio.run(handle({"annotation_type": "routing_ack", "status": "needs_manual_target", "attachment_manifest": rejected,
                        "options": [{"task_id": f"t{n}", "label": f"Work {n}"} for n in range(1, 10)]}))
    assert _Rec.sent[-1][1].splitlines() == [
        "Куда отправить последнее сообщение:", *[f"{n}. Work {n}" for n in range(1, 9)], "…и ещё 1 в веб-чате.",
        "Ответьте номером или выберите вариант в веб-чате.", "Не приложено к задаче: отчёт.pdf"]


def test_subagent_cards_off_hides_activity(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_SUBAGENT_CARDS="off")
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "Subagent abc scheduled", "is_progress": True,
                        "subagent_event": "scheduled", "subagent_task_id": "abc"}))
    assert not _Rec.sent and not _Rec.edited




def test_markdown_hint_false_sends_plain(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch)
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "use * and a_b_c.txt", "is_progress": False, "markdown": False}))
    assert _Rec.sent and _Rec.sent[0][2] == ""        # plain → no markdown→HTML mangling


def test_markdown_hint_absent_or_true_sends_html(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch)
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "**bold**", "is_progress": False}))            # no markdown key
    assert _Rec.sent and _Rec.sent[0][2] == "HTML"
    _Rec.sent.clear()
    asyncio.run(handle({"text": "**bold**", "is_progress": False, "markdown": True}))
    assert _Rec.sent and _Rec.sent[0][2] == "HTML"




def test_subagent_card_shows_work_commentary(tmp_path, monkeypatch):
    # Complaint #2: the card must carry the live work note (like the UI), not
    # just a bare status line.
    plugin, api = _setup(tmp_path, monkeypatch)
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "I will read lines 710 to 765 of ouroboros/llm.py.",
                        "is_progress": True, "subagent_event": "progress",
                        "subagent_task_id": "27f40120", "subagent_role": "planning-scout-1"}))
    assert len(_Rec.sent) == 1
    card = _Rec.sent[0][1]
    assert "planning-scout-1" in card
    assert "I will read lines 710 to 765" in card  # the actual commentary


def test_subagent_card_preserves_short_note_byte_for_byte():
    _load_plugin()
    from tg_sub_test.lib.telegram_state import _subagent_card_text

    card = _subagent_card_text(
        {
            "text": "reading the current implementation",
            "subagent_task_id": "27f40120-full-task-id",
            "subagent_role": "planning-scout-1",
        },
        "progress",
        "en",
    )

    assert card == (
        "🟡 planning-scout-1 — running [27f40120]\n"
        "reading the current implementation"
    )


def test_subagent_card_discloses_long_note_and_full_task_ref():
    _load_plugin()
    from ouroboros.utils import truncate_review_artifact
    from tg_sub_test.lib.telegram_state import _subagent_card_text

    task_id = "27f40120-full-task-id"
    note = "x" * 1000
    preview = truncate_review_artifact(note, 700)

    card = _subagent_card_text(
        {
            "text": note,
            "subagent_task_id": task_id,
            "subagent_role": "planning-scout-1",
        },
        "progress",
        "en",
    )

    assert card == (
        f"🟡 planning-scout-1 — running [27f40120]\n{preview}\n"
        f"Full ref: task {task_id}"
    )
    assert "OMISSION NOTE: truncated at 700 chars; original length 1000" in card


def test_jsonl_tail_uses_shared_parser_and_discloses_byte_prefix(tmp_path):
    _load_plugin()
    from tg_sub_test.lib.telegram_state import _jsonl_tail

    path = tmp_path / "events.jsonl"
    path.write_text(
        "ignored partial line\n"
        + json.dumps({"i": 1}) + "\n"
        + json.dumps(["not-a-dict"]) + "\n"
        + json.dumps({"i": 2}) + "\n",
        encoding="utf-8",
    )

    rows, omitted = _jsonl_tail(path, max_entries=10, tail_bytes=30)

    assert rows == [{"i": 2}]
    assert omitted is True


def test_jsonl_tail_discloses_entry_prefix(tmp_path):
    _load_plugin()
    from tg_sub_test.lib.telegram_state import _jsonl_tail

    path = tmp_path / "events.jsonl"
    path.write_text(
        "".join(json.dumps({"i": i}) + "\n" for i in range(3)),
        encoding="utf-8",
    )

    rows, omitted = _jsonl_tail(path, max_entries=2, tail_bytes=1024)

    assert rows == [{"i": 1}, {"i": 2}]
    assert omitted is True


def test_load_runtime_state_uses_validated_loopback_port(monkeypatch):
    _load_plugin()
    state = sys.modules["tg_sub_test.lib.telegram_state"]
    seen = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"spent_usd": 12.5, "budget_pct": 25.0}

    class Client:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url):
            seen["url"] = url
            return Response()

    class Api:
        @staticmethod
        def get_runtime_info():
            return {"server_port": "9123"}

    monkeypatch.setattr(state.httpx, "AsyncClient", Client)

    result = asyncio.run(state._load_runtime_state(Api()))

    assert result == {"spent_usd": 12.5, "budget_pct": 25.0}
    assert seen["url"] == "http://127.0.0.1:9123/api/state"
    assert seen["kwargs"] == {
        "timeout": 2.0,
        "follow_redirects": False,
        "trust_env": False,
    }


def test_load_runtime_state_rejects_invalid_port_before_http(monkeypatch):
    _load_plugin()
    state = sys.modules["tg_sub_test.lib.telegram_state"]

    class Client:
        def __init__(self, **_kwargs):
            raise AssertionError("HTTP client must not be created for an invalid port")

    class Api:
        def __init__(self, port):
            self.port = port

        def get_runtime_info(self):
            return {"server_port": self.port}

    monkeypatch.setattr(state.httpx, "AsyncClient", Client)

    for port in (0, 65_536, "9123/path", None):
        assert asyncio.run(state._load_runtime_state(Api(port))) == {}


def test_recent_thoughts_discloses_long_entry_and_reconstructible_source(tmp_path):
    from ouroboros import dialogue_store as ds

    plugin = _load_plugin()
    data = tmp_path / "data"
    timestamp = "2026-07-30T10:11:12Z"
    task_id = "task-abcdef123456"
    ds.append(data, "progress", {"ts": timestamp, "task_id": task_id, "text": "x" * 300})

    class Api:
        @staticmethod
        def get_runtime_info():
            return {"data_dir": str(data)}

    text = plugin._load_recent_thoughts(Api())

    assert "OMISSION NOTE: truncated at 100 chars; original length 300" in text
    assert f"Full ref: data/state/dialog.sqlite (progress) @ {timestamp} task {task_id}" in text
    assert text.endswith("Showing up to 4 recent entries. Full source: data/state/dialog.sqlite (progress)")


def test_recent_thoughts_preserves_short_entry_and_discloses_list_projection(tmp_path):
    from ouroboros import dialogue_store as ds

    plugin = _load_plugin()
    data = tmp_path / "data"
    note = "plain short entry byte intact"
    ds.append(data, "progress", {"text": note})

    class Api:
        @staticmethod
        def get_runtime_info():
            return {"data_dir": str(data)}

    assert plugin._load_recent_thoughts(Api()) == (
        f"• {note}\n\n"
        "Showing up to 4 recent entries. Full source: data/state/dialog.sqlite (progress)"
    )


def test_subagent_progress_edits_one_bubble_not_a_flood(tmp_path, monkeypatch):
    # Complaint #1: a stream of progress notes for ONE subagent must edit a
    # single bubble in place (the note changes each time), never post new ones.
    plugin, api = _setup(tmp_path, monkeypatch)
    handle = plugin._make_outbound(api)
    for note in ("reading llm.py retry logic", "searching the recovery loop", "examining vision params"):
        asyncio.run(handle({"text": note, "is_progress": True, "subagent_event": "progress",
                            "subagent_task_id": "27f40120", "subagent_role": "planning-scout-1"}))
    assert len(_Rec.sent) == 1     # ONE bubble — not a flood
    assert len(_Rec.edited) == 2   # subsequent notes edit in place
    assert "examining vision params" in _Rec.edited[-1][2]  # shows the latest note


def test_edit_message_not_modified_treated_as_success():
    # Identical-text edits ("message is not modified") must read as success so
    # the card render never falls back to posting a duplicate bubble.
    plugin = _load_plugin()
    client = plugin.TelegramClient("token")

    async def _raise(msg):
        raise RuntimeError(msg)

    async def call_not_modified(method, **kw):
        await _raise("Telegram API editMessageText returned HTTP 400: Bad Request: message is not modified")

    async def call_real_error(method, **kw):
        await _raise("Telegram API editMessageText returned HTTP 400: Bad Request: message to edit not found")

    client.call = call_not_modified
    assert asyncio.run(client.edit_message_text(42, 100, "x", parse_mode="")) is True
    client.call = call_real_error
    assert asyncio.run(client.edit_message_text(42, 100, "x", parse_mode="")) is False


def test_helper_rejection_card_carries_the_owner_sentence(tmp_path, monkeypatch):
    """#1403: the real rejection producer's frame, merged the way the bridge publishes
    chat.outbound (text + progress_meta), reaches the unchanged skill's helper card with
    the owner sentence; the parent model's instruction stays in the stored result."""
    from types import SimpleNamespace

    from supervisor.events_schedule_task import _reject_schedule_task
    from tests._store_shared import read_result

    frames = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path / "drive", send_with_budget=lambda chat_id, text, **kw: frames.append(
        {"chat_id": chat_id, "text": text, "is_progress": kw.get("is_progress", False), "task_id": kw.get("task_id", ""),
         "role": kw["role"], "system_type": kw["system_type"], **kw.get("progress_meta", {})}))
    detail = "Subagent not scheduled: the worker pool is unavailable. Do the work inline yourself, or retry after /restart."
    _reject_schedule_task(ctx, tid="kid12345", chat_id=42, delegation_role="subagent", parent_id="root1",
                          root_task_id="root1", role="researcher", result_fields={}, detail=detail,
                          fallback_message="⚠️ Task kid12345 not scheduled: worker pool unavailable.")
    assert read_result(tmp_path / "drive", "kid12345")["result"] == detail
    plugin, api = _setup(tmp_path, monkeypatch)
    asyncio.run(plugin._make_outbound(api)(frames[0]))
    card = _Rec.sent[0][1]
    assert "Task kid12345 not scheduled: worker pool unavailable." in card
    assert "inline yourself" not in card and "/restart" not in card


@pytest.mark.parametrize("owner_sentence", [False, True])
@pytest.mark.parametrize("abandoned", [False, True])
@pytest.mark.parametrize("persistence", ["saved", "failed", "busy"])
@pytest.mark.parametrize("delegation_role", ["subagent", "root"])
def test_rejection_preserves_persistence_qualification(tmp_path, monkeypatch, owner_sentence,
                                                     abandoned, persistence, delegation_role):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ouroboros.store import StoreBusy
    from supervisor import events_schedule_task, task_admission

    detail = "Subagent not scheduled: worker pool unavailable. Do the work inline yourself."
    owner = "Task kid12345 not scheduled: worker pool unavailable."
    addendum = (" Another owner now holds this request; nothing was overwritten." if abandoned else
                " The refusal was not recorded (scheduled_result_authority_unknown); recovery keeps the request.")
    # The owner's sentence keeps the uncertainty and the kept request, without the internal code.
    owner_addendum = addendum.replace(" (scheduled_result_authority_unknown)", "")
    refusal = Mock(side_effect={"saved": None, "failed": OSError("store unavailable"),
                                "busy": StoreBusy("current store busy: database is locked", reason="contention")}[persistence])
    monkeypatch.setattr(task_admission, "persist_never_admitted_refusal", refusal)
    preparation = SimpleNamespace(token="claim-1", refusal_fields=lambda: {},
                                  abandon=Mock(return_value=abandoned), after_refusal=Mock())
    sent = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, send_with_budget=lambda *a, **kw: sent.append((a, kw)))
    kwargs = dict(tid="kid12345", chat_id=42, delegation_role=delegation_role, parent_id="root1",
                  root_task_id="root1", role="researcher", result_fields={}, detail=detail,
                  fallback_message="⚠️ " + owner if owner_sentence else "", preparation=preparation)
    if persistence == "busy":
        with pytest.raises(StoreBusy):
            events_schedule_task._reject_schedule_task(ctx, **kwargs)
    else:
        assert events_schedule_task._reject_schedule_task(ctx, **kwargs) is (persistence == "saved")
    refusal.assert_called_once()
    assert refusal.call_args.kwargs["result"] == detail, "the parent model keeps the full producer detail"
    assert refusal.call_args.kwargs["admission_token"] == "claim-1"
    assert preparation.abandon.call_count == (persistence == "failed")
    assert preparation.after_refusal.call_count == (persistence == "saved")
    if persistence == "busy" or (delegation_role == "root" and not owner_sentence):
        assert sent == []
        return
    failed = persistence == "failed"
    expected = owner + (owner_addendum if failed else "") if owner_sentence else detail + (addendum if failed else "")
    ((chat_id, text), fields), = sent
    if owner_sentence:
        assert "scheduled_result_authority_unknown" not in text, "owner words carry no internal code"
    assert chat_id == 42 and text == "⚠️ " + expected
    assert fields["role"] == "system"
    if delegation_role == "subagent":
        assert fields["system_type"] == "subagent_admission_notice"
        assert fields["progress_meta"]["error"] == expected, "the web card reads this public error field"
        assert fields["task_id"] == "root1"
    else:
        assert fields["system_type"] == "task_admission_notice"
