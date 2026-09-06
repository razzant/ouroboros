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


def _send_card(plugin, tmp_path, monkeypatch):
    _settings(tmp_path)
    monkeypatch.setattr(plugin, "TelegramClient", Client)
    api = Api(tmp_path)
    asyncio.run(plugin._make_quiz(api)(dict(_EVENT)))
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


@pytest.mark.parametrize("answer_text", ["Use mysql instead", "  Use mysql instead\n", "2"])
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

    # Task settled: expired.
    Client.updates = [_callback(11, f"qz:{token}:1")]
    _run_poller(plugin, api, monkeypatch, [], reply=(409, {"ok": False, "state": "expired_terminal"}))
    last = _LAST_CLIENT[-1]
    assert last.toasts == [("cb11", "This question has expired — the task moved on.")]
    assert last.edits == []

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
