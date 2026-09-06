import asyncio, importlib.util, json, sys, types
from pathlib import Path


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
    def __init__(self, token, **_kwargs): pass
    async def send_message(self, chat_id, text, parse_mode="HTML"):
        _Rec._id[0] += 1; _Rec.sent.append((chat_id, text, parse_mode, _Rec._id[0])); return _Rec._id[0]
    async def edit_message_text(self, chat_id, message_id, text, parse_mode="HTML"):
        _Rec.edited.append((chat_id, message_id, text, parse_mode)); return True
    async def send_chat_action(self, chat_id, action="typing"): pass


def _setup(tmp_path, monkeypatch, **settings):
    plugin = _load_plugin()
    sd = tmp_path / "state"; sd.mkdir(parents=True, exist_ok=True)
    base = {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": "all"}; base.update(settings)
    (sd / "settings.json").write_text(json.dumps(base), encoding="utf-8")
    monkeypatch.setattr(plugin, "TelegramClient", _Rec)
    _Rec.sent = []; _Rec.edited = []; _Rec._id = [1000]
    return plugin, _Api(sd)


def test_subagent_gets_one_edited_bubble(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch)
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "Subagent abc scheduled (researcher)", "is_progress": True,
                        "subagent_event": "scheduled", "subagent_task_id": "abc123", "subagent_role": "researcher"}))
    assert len(_Rec.sent) == 1 and not _Rec.edited
    first_id = _Rec.sent[0][3]
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


def test_real_reply_is_sent(tmp_path, monkeypatch):
    plugin, api = _setup(tmp_path, monkeypatch)
    handle = plugin._make_outbound(api)
    asyncio.run(handle({"text": "Here is your answer", "is_progress": False}))
    assert len(_Rec.sent) == 1 and "answer" in _Rec.sent[0][1]


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
    plugin = _load_plugin()
    data = tmp_path / "data"
    log = data / "logs" / "progress.jsonl"
    log.parent.mkdir(parents=True)
    timestamp = "2026-07-30T10:11:12Z"
    task_id = "task-abcdef123456"
    log.write_text(
        json.dumps({"ts": timestamp, "task_id": task_id, "text": "x" * 300}) + "\n",
        encoding="utf-8",
    )

    class Api:
        @staticmethod
        def get_runtime_info():
            return {"data_dir": str(data)}

    text = plugin._load_recent_thoughts(Api())

    assert "OMISSION NOTE: truncated at 100 chars; original length 300" in text
    assert f"Full ref: data/logs/progress.jsonl @ {timestamp} task {task_id}" in text
    assert text.endswith("Showing up to 4 recent entries. Full source: data/logs/progress.jsonl")


def test_recent_thoughts_preserves_short_entry_and_discloses_list_projection(tmp_path):
    plugin = _load_plugin()
    data = tmp_path / "data"
    log = data / "logs" / "progress.jsonl"
    log.parent.mkdir(parents=True)
    note = "plain short entry byte intact"
    log.write_text(json.dumps({"text": note}) + "\n", encoding="utf-8")

    class Api:
        @staticmethod
        def get_runtime_info():
            return {"data_dir": str(data)}

    assert plugin._load_recent_thoughts(Api()) == (
        f"• {note}\n\n"
        "Showing up to 4 recent entries. Full source: data/logs/progress.jsonl"
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
