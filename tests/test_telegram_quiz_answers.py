"""Answering a quiz card from Telegram (#472): buttons and replies reach the
host's ONE decision ingress; outcomes are toasted honestly."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_plugin():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    package = types.ModuleType("telegram_quiz_test")
    package.__path__ = [str(root)]
    sys.modules["telegram_quiz_test"] = package
    spec = importlib.util.spec_from_file_location("telegram_quiz_test.plugin", root / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Api:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.logs = []

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, keys):
        return {"TELEGRAM_BOT_TOKEN": "token"}

    def get_skill_token(self):
        return types.SimpleNamespace(use_in_request=lambda: "skill-token")

    def log(self, level, message, **fields):
        self.logs.append((level, message))


class Client:
    updates: list = []

    def __init__(self, token, **_kwargs):
        self.sent, self.panels, self.edits, self.toasts = [], [], [], []

    async def call(self, method, **kwargs):
        return {"ok": True, "result": {}}

    async def get_updates(self, offset):
        return list(Client.updates)

    async def send_message(self, chat_id, text, parse_mode="HTML"):
        self.sent.append((chat_id, text))
        return 1

    async def send_message_with_inline_keyboard(self, chat_id, text, keyboard, parse_mode="HTML"):
        self.panels.append((chat_id, text, keyboard))
        return 555

    async def edit_message_text_with_inline_keyboard(self, chat_id, message_id, text, keyboard, parse_mode="HTML"):
        self.edits.append((chat_id, message_id, text, keyboard))
        return True

    async def answer_callback_query(self, callback_query_id, *, text=""):
        self.toasts.append((callback_query_id, text))


_EVENT = {
    "chat_id": 1, "quiz_id": "q1", "task_id": "task-1", "question": "Which db?",
    "options": [{"label": "sqlite"}, {"label": "postgres"}], "stake": "", "assumption": "sqlite meanwhile",
    "transport": {},
}


def _settings(tmp_path, **extra):
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", **extra}), encoding="utf-8")


def _send_card(plugin, tmp_path, monkeypatch, *, wait_for_answer=False):
    _settings(tmp_path)
    monkeypatch.setattr(plugin, "TelegramClient", Client)
    api = Api(tmp_path)
    asyncio.run(plugin._make_quiz(api)({**_EVENT, "wait_for_answer": wait_for_answer}))
    return api


def _run_poller(plugin, api, monkeypatch, posts, *, reply=(200, {"ok": True, "state": "answered"})):
    async def fake_post(_api, path, body):
        posts.append((path, body))
        return reply

    injected = []

    async def fake_inject(_api, payload):
        injected.append(payload)

    async def stop_sleep(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(plugin, "_host_post", fake_post)
    monkeypatch.setattr(plugin, "_inject", fake_inject)
    monkeypatch.setattr(plugin.asyncio, "sleep", stop_sleep)
    monkeypatch.setattr(plugin, "TelegramClient", Client)
    try:
        asyncio.run(plugin._make_poller(api)())
    except asyncio.CancelledError:
        pass
    return injected


def test_quiz_card_carries_one_button_per_option_and_remembers_the_card(tmp_path, monkeypatch):
    plugin = _load_plugin()
    api = _send_card(plugin, tmp_path, monkeypatch)
    state = json.loads((tmp_path / "quiz_state.json").read_text(encoding="utf-8"))
    (token, record), = state["quizzes"].items()
    assert record == {
        "task_id": "task-1", "quiz_id": "q1", "chat_id": 42, "message_id": 555,
        "options": ["sqlite", "postgres"],
        "text": "Question: Which db?\n1. sqlite\n2. postgres\nContinuing meanwhile: sqlite meanwhile",
    }
    assert token == plugin.telegram_quiz.mint_token("task-1", "q1")
    keyboard = plugin.telegram_quiz.quiz_keyboard(token, ["sqlite", "postgres"])
    assert [row[0]["callback_data"] for row in keyboard] == [f"qz:{token}:0", f"qz:{token}:1"]
    assert all(len(row[0]["callback_data"].encode("utf-8")) <= 64 for row in keyboard)
    assert [row[0]["text"] for row in keyboard] == ["1. sqlite", "2. postgres"]
    assert api.logs == []


def test_open_question_is_a_plain_message_with_a_reply_address(tmp_path, monkeypatch):
    plugin = _load_plugin()
    _settings(tmp_path)
    monkeypatch.setattr(plugin, "TelegramClient", Client)
    api = Api(tmp_path)
    asyncio.run(plugin._make_quiz(api)({**_EVENT, "options": [], "wait_for_answer": True}))
    first = _LAST_CLIENT[-1]
    assert first.panels == []
    assert first.sent == [(42, "Question: Which db?\nWaiting for your answer; Stop and the task deadline still apply."
                                "\nReply to this message with your answer.")]
    record = plugin.telegram_quiz.quiz_for_message(api, 42, 1)
    assert record and record["options"] == []
    Client.updates = [{"update_id": 100, "message": {
        "message_id": 1001, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
        "text": "Try a different database", "reply_to_message": {"message_id": 1},
    }}]
    posts = []
    assert _run_poller(plugin, api, monkeypatch, posts) == []
    assert posts == [("/chat/decision", {
        "request_id": "tg:100", "decision_id": "quiz:task-1:q1", "comment": "Try a different database",
    })]


def test_tapped_option_reaches_the_decision_ingress_and_settles_the_card(tmp_path, monkeypatch):
    plugin = _load_plugin()
    api = _send_card(plugin, tmp_path, monkeypatch)
    token = plugin.telegram_quiz.mint_token("task-1", "q1")
    Client.updates = [{"update_id": 7, "callback_query": {
        "id": "cb", "data": f"qz:{token}:1",
        "message": {"message_id": 555, "chat": {"id": 42, "type": "private"}}, "from": {"id": 42},
    }}]
    posts = []
    injected = _run_poller(plugin, api, monkeypatch, posts,
                           reply=(200, {"ok": True, "state": "answered", "answered_index": 1}))

    assert posts == [("/chat/decision", {
        "request_id": "tg:7", "decision_id": "quiz:task-1:q1", "option_index": 1,
    })]
    assert injected == [], "an answer is a decision, never a new chat turn"
    # The toast and the settled card come from the last constructed client.
    last = _LAST_CLIENT[-1]
    assert last.toasts == [("cb", "✅ Answer delivered to the task.")]
    assert last.edits == [(42, 555,
                           "Question: Which db?\n1. sqlite\n2. postgres\nContinuing meanwhile: sqlite meanwhile"
                           "\nAnswered: 2. postgres", [])]


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("answer_path", ["callback", "reply"])
def test_settled_quiz_drops_only_required_wait_copy(tmp_path, monkeypatch, required, answer_path):
    plugin = _load_plugin()
    api = _send_card(plugin, tmp_path, monkeypatch, wait_for_answer=required)
    assert ("Waiting for your answer" in _LAST_CLIENT[-1].panels[0][1]) is required
    token = plugin.telegram_quiz.mint_token("task-1", "q1")
    if answer_path == "callback":
        Client.updates = [{"update_id": 70, "callback_query": {
            "id": "cb", "data": f"qz:{token}:1", "from": {"id": 42},
            "message": {"message_id": 555, "chat": {"id": 42, "type": "private"}},
        }}]
    else:
        Client.updates = [{"update_id": 70, "message": {
            "message_id": 600, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
            "text": "Keep the prepared choice", "reply_to_message": {"message_id": 555},
        }}]
    posts = []
    assert not _run_poller(plugin, api, monkeypatch, posts,
                           reply=(200, {"ok": True, "state": "answered", "answered_index": 1}))
    assert len(posts) == 1 and posts[0][0] == "/chat/decision"
    text = _LAST_CLIENT[-1].edits[0][2]
    assert "Waiting for your answer" not in text
    assert ("Continuing meanwhile: sqlite meanwhile" in text) is not required
    assert "Answered: " in text
    assert _LAST_CLIENT[-1].edits[0][3] == []


@pytest.mark.parametrize("answer_text", ["Use mysql instead", "  Use mysql instead\n", "2", "`/panic`", "The command is /panic"])
def test_reply_to_the_card_is_the_owners_own_answer(tmp_path, monkeypatch, answer_text):
    plugin = _load_plugin()
    api = _send_card(plugin, tmp_path, monkeypatch)
    Client.updates = [{"update_id": 8, "message": {
        "message_id": 600, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
        "text": answer_text, "reply_to_message": {"message_id": 555},
    }}]
    posts = []
    injected = _run_poller(plugin, api, monkeypatch, posts)

    assert posts == [("/chat/decision", {
        "request_id": "tg:8", "decision_id": "quiz:task-1:q1", "comment": answer_text,
    })]
    assert injected == []
    last = _LAST_CLIENT[-1]
    assert last.sent == [(42, "✅ Answer delivered to the task.")]
    assert last.edits[0][2].endswith("\nAnswered: " + answer_text)


@pytest.mark.parametrize("command", ["/menu", "/language", "/help"])
def test_reply_to_quiz_keeps_existing_local_command_precedence(tmp_path, monkeypatch, command):
    plugin = _load_plugin()
    api = _send_card(plugin, tmp_path, monkeypatch)
    Client.updates = [{"update_id": 88, "message": {
        "message_id": 601, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
        "text": command, "reply_to_message": {"message_id": 555},
    }}]
    posts = []
    injected = _run_poller(plugin, api, monkeypatch, posts)
    assert posts == [] and injected == []
    assert _LAST_CLIENT[-1].sent or _LAST_CLIENT[-1].panels


def test_reply_to_an_ordinary_message_is_a_normal_chat_turn(tmp_path, monkeypatch):
    plugin = _load_plugin()
    api = _send_card(plugin, tmp_path, monkeypatch)
    Client.updates = [{"update_id": 9, "message": {
        "message_id": 601, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
        "text": "and this?", "reply_to_message": {"message_id": 12},
    }}]
    posts = []
    injected = _run_poller(plugin, api, monkeypatch, posts)
    assert posts == []
    assert [p["text"] for p in injected] == ["and this?"]


def test_late_or_unknown_answers_are_toasted_honestly(tmp_path, monkeypatch):
    plugin = _load_plugin()
    api = _send_card(plugin, tmp_path, monkeypatch)
    token = plugin.telegram_quiz.mint_token("task-1", "q1")

    def _callback(update_id, data):
        return {"update_id": update_id, "callback_query": {
            "id": f"cb{update_id}", "data": data,
            "message": {"message_id": 555, "chat": {"id": 42, "type": "private"}}, "from": {"id": 42},
        }}

    # Already answered by the web card with option 0: the loser learns the winner.
    Client.updates = [_callback(10, f"qz:{token}:1")]
    posts = []
    _run_poller(plugin, api, monkeypatch, posts,
                reply=(409, {"ok": False, "error": "already_answered", "state": "answered", "answered_index": 0}))
    last = _LAST_CLIENT[-1]
    assert last.toasts == [("cb10", "This question was already answered.")]
    assert last.edits[0][2].endswith("\nAnswered: 1. sqlite")

    # The task had finished, but the card outlived it (В17a=A): the host records
    # the answer AND delivers it into the card's chat, so the tap succeeds and
    # the card settles exactly as an ordinary answer does.
    Client.updates = [_callback(11, f"qz:{token}:1")]
    _run_poller(plugin, api, monkeypatch, [],
                reply=(200, {"ok": True, "state": "answered", "answered_index": 1,
                             "answered_after_terminal": True, "forwarded": True}))
    last = _LAST_CLIENT[-1]
    assert last.toasts == [("cb11", "✅ The task had already finished — your answer "
                                   "was delivered to the chat.")]
    assert last.edits[0][2].endswith("\nAnswered: 2. postgres")
    assert last.edits[0][3] == []  # the keyboard goes, as for any answer

    # A card whose chat has no owner turn to start (machine/hidden): recorded,
    # never claimed as delivered.
    Client.updates = [_callback(16, f"qz:{token}:1")]
    _run_poller(plugin, api, monkeypatch, [],
                reply=(200, {"ok": True, "state": "answered", "answered_index": 1,
                             "answered_after_terminal": True, "forwarded": False,
                             "reason_code": "hidden_chat"}))
    assert _LAST_CLIENT[-1].toasts == [
        ("cb16", "✅ Answer recorded. The task had already finished and this card "
                 "has no chat to deliver it to."),
    ]

    # A genuinely settled card (already answered by another surface) still 409s.
    Client.updates = [_callback(17, f"qz:{token}:1")]
    _run_poller(plugin, api, monkeypatch, [], reply=(409, {"ok": False, "state": "expired_terminal"}))
    assert _LAST_CLIENT[-1].toasts == [("cb17", "This question has expired — the task moved on.")]

    # Unknown to the host.
    Client.updates = [_callback(12, f"qz:{token}:0")]
    _run_poller(plugin, api, monkeypatch, [], reply=(404, {"error": "quiz not found"}))
    assert _LAST_CLIENT[-1].toasts == [("cb12", "This question is no longer known to Ouroboros.")]

    # Unknown token / out-of-range index: nothing is posted at all.
    Client.updates = [_callback(13, "qz:deadbeef0000:0"), _callback(14, f"qz:{token}:7")]
    posts = []
    _run_poller(plugin, api, monkeypatch, posts)
    assert posts == []
    assert [t for _cb, t in _LAST_CLIENT[-1].toasts] == ["This question is no longer known to Ouroboros."] * 2


def test_non_owner_tap_never_reaches_the_ingress(tmp_path, monkeypatch):
    plugin = _load_plugin()
    api = _send_card(plugin, tmp_path, monkeypatch)
    token = plugin.telegram_quiz.mint_token("task-1", "q1")
    Client.updates = [{"update_id": 15, "callback_query": {
        "id": "cb", "data": f"qz:{token}:0",
        "message": {"message_id": 555, "chat": {"id": 42, "type": "private"}}, "from": {"id": 99},
    }}]
    posts = []
    _run_poller(plugin, api, monkeypatch, posts)
    assert posts == []
    assert _LAST_CLIENT[-1].toasts == [("cb", "Not authorized")]


def test_remembered_cards_are_bounded(tmp_path):
    plugin = _load_plugin()
    api = Api(tmp_path)
    for index in range(60):
        plugin.telegram_quiz.remember_quiz(api, f"tok{index}", {"task_id": "t", "quiz_id": f"q{index}",
                                                                 "chat_id": 42, "message_id": index, "options": []})
    state = json.loads((tmp_path / "quiz_state.json").read_text(encoding="utf-8"))
    assert len(state["quizzes"]) == 50
    assert "tok0" not in state["quizzes"] and "tok59" in state["quizzes"]
    assert plugin.telegram_quiz.quiz_for_message(api, 42, 59)["quiz_id"] == "q59"
    assert plugin.telegram_quiz.quiz_for_message(api, 42, 0) is None


_LAST_CLIENT: list = []
_original_init = Client.__init__


def _tracking_init(self, token, **kwargs):
    _original_init(self, token, **kwargs)
    _LAST_CLIENT.append(self)


Client.__init__ = _tracking_init


@pytest.mark.parametrize("command", ["/panic", "/restart", "/status"])
@pytest.mark.parametrize("reply_to_quiz", [False, True])
def test_owner_commands_keep_dispatch_when_replying_to_quiz(tmp_path, monkeypatch, command, reply_to_quiz):
    plugin = _load_plugin()
    api = _send_card(plugin, tmp_path, monkeypatch)
    _settings(tmp_path, TELEGRAM_COMMAND_MODE="full_access")
    message = {"message_id": 602, "chat": {"id": 42, "type": "private"},
               "from": {"id": 42}, "text": command}
    if reply_to_quiz:
        message["reply_to_message"] = {"message_id": 555}
    Client.updates = [{"update_id": 89, "message": message}]
    posts = []
    injected = _run_poller(plugin, api, monkeypatch, posts)
    assert posts == []
    assert [row["text"] for row in injected] == [command]


def test_recommended_option_is_starred_in_the_button_caption(tmp_path, monkeypatch):
    plugin = _load_plugin()
    _settings(tmp_path)
    monkeypatch.setattr(plugin, "TelegramClient", Client)
    api = Api(tmp_path)
    event = {**_EVENT, "options": [{"label": "sqlite"}, {"label": "postgres", "detail": "scales", "recommended": True}]}
    asyncio.run(plugin._make_quiz(api)(event))
    state = json.loads((tmp_path / "quiz_state.json").read_text(encoding="utf-8"))
    (token, record), = state["quizzes"].items()
    assert record["options"] == ["sqlite", "★ postgres"]
    assert "1. sqlite\n2. ★ postgres" in record["text"]
    keyboard = plugin.telegram_quiz.quiz_keyboard(token, record["options"])
    assert [row[0]["text"] for row in keyboard] == ["1. sqlite", "2. ★ postgres"]


# --- Full card: project, host facts, option details, localization, overflow ---

_HINT_EN = "Tap an option, or reply to this message with your own answer."
_HINT_RU = "Нажмите вариант или ответьте на это сообщение своим текстом."
_FULL_EVENT = {
    **_EVENT,
    "project_name": "Alpha site",
    "host_facts": "Asked by task task-1, started by your message of 2026-09-25 00:21.",
    "stake": "Whether the data survives a restart.",
    "options": [{"label": "sqlite", "detail": "one file, nothing to run"},
                {"label": "postgres", "detail": "scales, needs a server", "recommended": True}],
}


def _recording_client(plugin):
    """The REAL TelegramClient with only the HTTP call recorded, so the
    production chunker and keyboard encoding are what the test observes."""
    calls: list = []

    class Recording(plugin.TelegramClient):
        async def call(self, method, *, data=None, files=None, timeout=30):
            calls.append((method, dict(data or {})))
            return {"ok": True, "result": {"message_id": 700 + len(calls)}}

    return Recording, calls


def _send_full_card(plugin, tmp_path, monkeypatch, event, **settings):
    _settings(tmp_path, **settings)
    recording, calls = _recording_client(plugin)
    monkeypatch.setattr(plugin, "TelegramClient", recording)
    api = Api(tmp_path)
    asyncio.run(plugin._make_quiz(api)(event))
    assert api.logs == []
    state = json.loads((tmp_path / "quiz_state.json").read_text(encoding="utf-8"))
    (token, record), = state["quizzes"].items()
    return api, calls, token, record


def _keyboard_calls(calls):
    return [(m, d) for m, d in calls if m == "sendMessage" and "reply_markup" in d]


def _plain_calls(calls):
    return [(m, d) for m, d in calls if m == "sendMessage" and "reply_markup" not in d]


@pytest.mark.parametrize("lang,expected_body,hint_text", [
    ("en",
     "Project: Alpha site\n"
     "Asked by task task-1, started by your message of 2026-09-25 00:21.\n"
     "Question: Which db?\n"
     "At stake: Whether the data survives a restart.\n"
     "1. sqlite — one file, nothing to run\n"
     "2. ★ postgres — scales, needs a server\n"
     "Continuing meanwhile: sqlite meanwhile",
     _HINT_EN),
    ("ru",
     "Проект: Alpha site\n"
     "Asked by task task-1, started by your message of 2026-09-25 00:21.\n"
     "Вопрос: Which db?\n"
     "Что на кону: Whether the data survives a restart.\n"
     "1. sqlite — one file, nothing to run\n"
     "2. ★ postgres — scales, needs a server\n"
     "Пока продолжаю так: sqlite meanwhile",
     _HINT_RU),
], ids=["en", "ru"])
def test_short_card_is_one_message_with_project_facts_details_and_star(
        tmp_path, monkeypatch, lang, expected_body, hint_text):
    plugin = _load_plugin()
    _api, calls, token, record = _send_full_card(
        plugin, tmp_path, monkeypatch, _FULL_EVENT, TELEGRAM_LANGUAGE=lang)
    assert len(calls) == 1 and calls[0][0] == "sendMessage"
    data = calls[0][1]
    assert data["text"] == f"{expected_body}\n{hint_text}"
    assert "parse_mode" not in data, "the authored text is sent verbatim"
    buttons = json.loads(data["reply_markup"])["inline_keyboard"]
    assert [row[0]["text"] for row in buttons] == ["1. sqlite", "2. ★ postgres"]
    assert [row[0]["callback_data"] for row in buttons] == [f"qz:{token}:0", f"qz:{token}:1"]
    assert record["message_id"] == 701
    assert record["options"] == ["sqlite", "★ postgres"]
    assert record["text"] == expected_body


@pytest.mark.parametrize("lang,waiting", [
    ("en", "Waiting for your answer; Stop and the task deadline still apply."),
    ("ru", "Жду вашего ответа; Stop и срок задачи по-прежнему действуют."),
], ids=["en", "ru"])
def test_waiting_line_is_localized_and_dropped_from_the_settled_text(tmp_path, monkeypatch, lang, waiting):
    plugin = _load_plugin()
    _api, calls, _token, record = _send_full_card(
        plugin, tmp_path, monkeypatch, {**_FULL_EVENT, "wait_for_answer": True}, TELEGRAM_LANGUAGE=lang)
    assert waiting in calls[0][1]["text"]
    assert waiting not in record["text"]
    assert record["text"].splitlines()[-1] == "2. ★ postgres — scales, needs a server"


def test_old_event_without_project_facts_or_details_keeps_the_plain_card(tmp_path, monkeypatch):
    plugin = _load_plugin()
    _api, calls, _token, record = _send_full_card(plugin, tmp_path, monkeypatch, dict(_EVENT))
    assert calls[0][1]["text"] == (
        "Question: Which db?\n1. sqlite\n2. postgres\nContinuing meanwhile: sqlite meanwhile\n" + _HINT_EN)
    assert "Project" not in record["text"] and " — " not in record["text"]


def _long_question():
    # Multi-paragraph, with astral characters that count twice in UTF-16.
    paragraph = ("Why this matters 🧭: " + "the migration keeps every row intact. " * 20).rstrip()
    return "\n\n".join(f"{index}. {paragraph}" for index in range(8))


def test_long_card_is_sent_in_ordered_plain_parts_then_one_keyboard_message(tmp_path, monkeypatch):
    plugin = _load_plugin()
    quiz = plugin.telegram_quiz
    event = {**_FULL_EVENT, "question": _long_question()}
    _api, calls, token, record = _send_full_card(plugin, tmp_path, monkeypatch, event)
    body = quiz.render_quiz_text(
        event["question"], ["sqlite", "postgres"], event["stake"], event["assumption"],
        project_name="Alpha site", host_facts=event["host_facts"],
        option_details=["one file, nothing to run", "scales, needs a server"], recommended_index=1)
    assert quiz._u16len(f"{body}\n{_HINT_EN}") > quiz._TELEGRAM_TEXT_LIMIT

    plain = [d["text"] for _m, d in _plain_calls(calls)]
    keyboards = _keyboard_calls(calls)
    assert len(plain) >= 2 and len(keyboards) == 1
    assert calls[-1] == keyboards[0], "the keyboard message comes last"
    assert all(quiz._u16len(part) <= quiz._TELEGRAM_TEXT_LIMIT for part in plain)
    assert all("parse_mode" not in d for _m, d in calls)
    assert "\n".join(plain) == body, "no authored character is lost across parts"

    compact = "Project: Alpha site\n1. sqlite\n2. ★ postgres"
    assert keyboards[0][1]["text"] == f"{compact}\n{_HINT_EN}"
    assert record["message_id"] == 700 + len(calls)
    assert record["text"] == compact
    assert record["options"] == ["sqlite", "★ postgres"]
    buttons = json.loads(keyboards[0][1]["reply_markup"])["inline_keyboard"]
    assert [row[0]["callback_data"] for row in buttons] == [f"qz:{token}:0", f"qz:{token}:1"]


def test_single_long_line_question_keeps_every_authored_character(tmp_path, monkeypatch):
    plugin = _load_plugin()
    quiz = plugin.telegram_quiz
    question = " ".join(f"word{index}🧭" for index in range(700))  # ~5 600 UTF-16 units, one line
    event = {**_FULL_EVENT, "question": question}
    _api, calls, _token, _record = _send_full_card(plugin, tmp_path, monkeypatch, event)
    plain = [d["text"] for _m, d in _plain_calls(calls)]
    assert len(plain) >= 2 and len(_keyboard_calls(calls)) == 1
    assert all(quiz._u16len(part) <= quiz._TELEGRAM_TEXT_LIMIT for part in plain)
    # The chunker breaks a single long line at a space and the message boundary
    # stands in for that space; every non-whitespace character survives in order.
    sent = "".join("".join(part.split()) for part in plain)
    assert question.replace(" ", "") in sent
    assert "".join(sent.split()) == "".join(quiz.render_quiz_text(
        question, ["sqlite", "postgres"], event["stake"], event["assumption"],
        project_name="Alpha site", host_facts=event["host_facts"],
        option_details=["one file, nothing to run", "scales, needs a server"],
        recommended_index=1).split())


def test_overflowing_card_answers_through_its_keyboard_message(tmp_path, monkeypatch):
    plugin = _load_plugin()
    event = {**_FULL_EVENT, "question": _long_question()}
    api, calls, token, record = _send_full_card(plugin, tmp_path, monkeypatch, event)
    keyboard_id = record["message_id"]
    first_part_id = 701
    assert keyboard_id != first_part_id

    # A tap resolves to the card identity and settles the keyboard message.
    Client.updates = [{"update_id": 21, "callback_query": {
        "id": "cb", "data": f"qz:{token}:1", "from": {"id": 42},
        "message": {"message_id": keyboard_id, "chat": {"id": 42, "type": "private"}},
    }}]
    posts = []
    _run_poller(plugin, api, monkeypatch, posts,
                reply=(200, {"ok": True, "state": "answered", "answered_index": 1}))
    assert posts == [("/chat/decision", {
        "request_id": "tg:21", "decision_id": "quiz:task-1:q1", "option_index": 1,
    })]
    assert _LAST_CLIENT[-1].edits == [(42, keyboard_id,
                                       "Project: Alpha site\n1. sqlite\n2. ★ postgres\nAnswered: 2. ★ postgres",
                                       [])]

    # A reply to the keyboard message is the owner's own answer ...
    Client.updates = [{"update_id": 22, "message": {
        "message_id": 900, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
        "text": "postgres, but later", "reply_to_message": {"message_id": keyboard_id},
    }}]
    posts = []
    assert _run_poller(plugin, api, monkeypatch, posts) == []
    assert posts == [("/chat/decision", {
        "request_id": "tg:22", "decision_id": "quiz:task-1:q1", "comment": "postgres, but later",
    })]

    # ... while a reply to an earlier explanation part is an ordinary owner message.
    Client.updates = [{"update_id": 23, "message": {
        "message_id": 901, "chat": {"id": 42, "type": "private"}, "from": {"id": 42},
        "text": "about paragraph 3", "reply_to_message": {"message_id": first_part_id},
    }}]
    posts = []
    injected = _run_poller(plugin, api, monkeypatch, posts)
    assert posts == []
    assert [row["text"] for row in injected] == ["about paragraph 3"]


def test_record_without_details_from_an_older_skill_still_answers(tmp_path, monkeypatch):
    plugin = _load_plugin()
    _settings(tmp_path)
    api = Api(tmp_path)
    token = plugin.telegram_quiz.mint_token("task-1", "q1")
    plugin.telegram_quiz.remember_quiz(api, token, {
        "task_id": "task-1", "quiz_id": "q1", "chat_id": 42, "message_id": 555,
        "options": ["sqlite", "★ postgres"], "text": "Question: Which db?\n1. sqlite\n2. ★ postgres",
    })
    Client.updates = [{"update_id": 24, "callback_query": {
        "id": "cb", "data": f"qz:{token}:0", "from": {"id": 42},
        "message": {"message_id": 555, "chat": {"id": 42, "type": "private"}},
    }}]
    posts = []
    _run_poller(plugin, api, monkeypatch, posts,
                reply=(200, {"ok": True, "state": "answered", "answered_index": 0}))
    assert posts[0][1]["option_index"] == 0
    assert _LAST_CLIENT[-1].edits[0][2].endswith("\nAnswered: 1. sqlite")


class _CardClient:
    """Records what send_quiz_card sends; message ids count up from 500."""

    def __init__(self):
        self.sent, self.next_id = [], 500

    async def send_message(self, chat_id, text, parse_mode="HTML"):
        self.sent.append(("plain", text)); self.next_id += 1
        return self.next_id

    async def send_message_with_inline_keyboard(self, chat_id, text, keyboard, parse_mode="HTML"):
        self.sent.append(("keyboard", text)); self.next_id += 1
        return self.next_id


def _send_card_direct(body, hint="tap"):
    import asyncio
    from skills.telegram.lib import telegram_quiz as quiz

    client = _CardClient()
    message_id, overflowed = asyncio.run(quiz.send_quiz_card(
        client, 42, body=body, compact="1. a\n2. b", hint_text=hint, keyboard=[[{"text": "1. a", "callback_data": "x"}]]))
    return client, message_id, overflowed


def test_card_that_fits_only_without_its_answered_edit_goes_out_as_parts():
    """Both directions: a card leaves room for "Answered: <echo>", or it is split (Opus L1)."""
    from skills.telegram.lib import telegram_quiz as quiz

    fits = "q" * (quiz._TELEGRAM_TEXT_LIMIT - quiz._ANSWERED_EDIT_RESERVE - len("\ntap"))
    client, message_id, overflowed = _send_card_direct(fits)
    assert not overflowed and [kind for kind, _ in client.sent] == ["keyboard"]
    assert quiz._u16len(client.sent[0][1]) + quiz._ANSWERED_EDIT_RESERVE <= quiz._TELEGRAM_TEXT_LIMIT

    barely = fits + "q" * 8  # fits the message limit alone, not with the answered edit
    assert quiz._u16len(f"{barely}\ntap") <= quiz._TELEGRAM_TEXT_LIMIT
    client, message_id, overflowed = _send_card_direct(barely)
    assert overflowed and [kind for kind, _ in client.sent] == ["plain", "keyboard"]
    assert client.sent[0][1] == barely and message_id == 502


def test_card_length_is_measured_in_utf16_units_not_code_points():
    """Astral characters count double in Telegram's limit; len() would let this card through (Opus L2)."""
    from skills.telegram.lib import telegram_quiz as quiz

    budget = quiz._TELEGRAM_TEXT_LIMIT - quiz._ANSWERED_EDIT_RESERVE - len("\ntap")
    body = "\U0001F600" * (budget // 2 + 4)  # fewer code points than the budget, more UTF-16 units
    assert len(body) < budget < quiz._u16len(body)
    client, _message_id, overflowed = _send_card_direct(body)
    assert overflowed and [kind for kind, _ in client.sent] == ["plain", "keyboard"]
    assert client.sent[0][1] == body
