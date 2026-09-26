"""TZ-2 B2: a quiz card already sent to Telegram follows the host's live
``chat.quiz_state`` facts — Telegram has no reload, so without them a card kept
its buttons and its waiting line after a web answer, a closed wait or the task's
end. The lifecycle only moves forward, as on the web card."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "skills" / "telegram"


def _load_plugin():
    package = types.ModuleType("telegram_quiz_state_test")
    package.__path__ = [str(_ROOT)]
    sys.modules["telegram_quiz_state_test"] = package
    spec = importlib.util.spec_from_file_location("telegram_quiz_state_test.plugin", _ROOT / "plugin.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Api:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.logs = []
        self.subscriptions = []

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, keys):
        return {"TELEGRAM_BOT_TOKEN": "token"}

    def log(self, level, message, **fields):
        self.logs.append((level, message))


class Client:
    instances: list = []
    edit_result = True

    def __init__(self, token, **_kwargs):
        self.sent, self.panels, self.edits, self.toasts = [], [], [], []
        Client.instances.append(self)

    async def send_message(self, chat_id, text, parse_mode="HTML"):
        self.sent.append((chat_id, text))
        return 1

    async def send_message_with_inline_keyboard(self, chat_id, text, keyboard, parse_mode="HTML"):
        self.panels.append((chat_id, text, keyboard))
        return 555

    async def edit_message_text_with_inline_keyboard(self, chat_id, message_id, text, keyboard, parse_mode="HTML"):
        self.edits.append((chat_id, message_id, text, keyboard))
        return Client.edit_result

    async def answer_callback_query(self, callback_query_id, *, text=""):
        self.toasts.append((callback_query_id, text))


_QUIZ = {
    "chat_id": 1, "quiz_id": "q1", "task_id": "task-1", "question": "Which db?",
    "options": [{"label": "sqlite"}, {"label": "postgres"}], "stake": "", "assumption": "sqlite meanwhile",
    "transport": {},
}
_WAITING_LINE = "Waiting for your answer"


@pytest.fixture
def card(tmp_path, monkeypatch):
    """Send one card to Telegram; return a helper that applies host lifecycle facts."""
    plugin = _load_plugin()
    Client.instances, Client.edit_result = [], True
    monkeypatch.setattr(plugin, "TelegramClient", Client)
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    api = Api(tmp_path)

    def send(**overrides):
        asyncio.run(plugin._make_quiz(api)({**_QUIZ, **overrides}))

    def apply(state, **fields):
        """One ``chat.quiz_state`` event, as ``LocalChatBridge.send_quiz_state`` publishes it."""
        before = len(Client.instances)
        asyncio.run(plugin._make_quiz_state(api)({
            "type": "quiz_state", "quiz_id": "q1", "task_id": "task-1", "state": state,
            "ts": "2026-09-25T20:00:00+00:00", "chat_id": 1, "transport": {},
            "topic": "chat.quiz_state", **fields,
        }))
        return [edit for client in Client.instances[before:] for edit in client.edits]

    ns = types.SimpleNamespace(plugin=plugin, api=api, send=send, apply=apply,
                               token=plugin.telegram_quiz.mint_token("task-1", "q1"))
    ns.record = lambda: plugin.telegram_quiz.quiz_for_token(api, ns.token)
    return ns


def test_web_answer_by_option_settles_the_telegram_card(card):
    card.send()
    assert card.apply("answered", answered_index=1) == [(42, 555,
        "Question: Which db?\n1. sqlite\n2. postgres\nContinuing meanwhile: sqlite meanwhile"
        "\nAnswered: 2. postgres", [])]
    assert card.record()["state"] == "answered"
    assert card.api.logs == []


def test_web_answer_in_the_owners_own_words_settles_the_card_verbatim(card):
    card.send(wait_for_answer=True)
    (edit,) = card.apply("answered", comment="neither — use duckdb")
    assert edit[2] == "Question: Which db?\n1. sqlite\n2. postgres\nAnswered: neither — use duckdb"
    assert edit[3] == []
    # An option with a note keeps both; a very long note is echoed like a Telegram reply.
    (edit,) = card.apply("answered", answered_index=0, comment="x" * 300)
    assert edit[2].endswith("\nAnswered: 1. sqlite — " + "x" * 200 + "…")


def test_closed_wait_drops_the_waiting_line_and_keeps_the_card_answerable(card):
    card.send(wait_for_answer=True)
    assert _WAITING_LINE in Client.instances[0].panels[0][1]
    # An `open` that closes nothing is not an edit.
    assert card.apply("open") == [] and card.apply("open", wait_for_answer=True) == []
    (edit,) = card.apply("open", wait_for_answer=False)
    assert edit[:2] == (42, 555)
    assert _WAITING_LINE not in edit[2]
    assert edit[2] == ("Question: Which db?\n1. sqlite\n2. postgres"
                       "\nThe task continued; an answer is still accepted."
                       "\nTap an option, or reply to this message with your own answer.")
    assert edit[3] == card.plugin.telegram_quiz.quiz_keyboard(card.token, ["sqlite", "postgres"])
    assert "state" not in card.record(), "an open card has nothing settled to remember"


def test_closed_wait_on_an_open_question_keeps_the_reply_address(card):
    card.send(options=[], wait_for_answer=True)
    (edit,) = card.apply("open", wait_for_answer=False)
    assert edit == (42, 1, "Question: Which db?\nThe task continued; an answer is still accepted."
                           "\nReply to this message with your answer.", [])


def test_expired_card_stops_waiting_but_still_takes_a_late_answer(card):
    """В17a=A: the host accepts a late answer as the owner's message, and the web
    card stays answerable (ANSWERABLE_QUIZ_STATES), so the buttons stay too."""
    card.send(wait_for_answer=True)
    (edit,) = card.apply("expired_terminal")
    assert _WAITING_LINE not in edit[2]
    assert edit[2].endswith("\nThe task finished; a late answer is accepted as your message."
                            "\nTap an option, or reply to this message with your own answer.")
    assert edit[3] == card.plugin.telegram_quiz.quiz_keyboard(card.token, ["sqlite", "postgres"])
    assert card.record()["state"] == "expired_terminal"
    # A stale closed-wait fact never turns "the task finished" back into "continued".
    assert card.apply("open", wait_for_answer=False) == []
    # A late answer still settles it.
    (edit,) = card.apply("answered", answered_index=0)
    assert edit[2].endswith("\nAnswered: 1. sqlite") and edit[3] == []


def test_superseded_card_becomes_a_read_only_record(card):
    card.send()
    (edit,) = card.apply("superseded")
    assert edit[2].endswith("\nContinuing meanwhile: sqlite meanwhile\nReplaced by a newer question.")
    assert edit[3] == []
    assert card.apply("expired_terminal") == [] and card.apply("open", wait_for_answer=False) == []


@pytest.mark.parametrize("later", [
    {"state": "open", "wait_for_answer": False}, {"state": "expired_terminal"}, {"state": "superseded"},
])
def test_answered_card_is_never_rolled_back(card, later):
    card.send(wait_for_answer=True)
    card.apply("answered", answered_index=1)
    fields = dict(later)
    assert card.apply(fields.pop("state"), **fields) == []
    assert card.record()["state"] == "answered"
    # The host's answer itself is idempotent: re-applying it is the same edit.
    (edit,) = card.apply("answered", answered_index=1)
    assert edit[2].endswith("\nAnswered: 2. postgres")


def test_an_answer_given_in_telegram_also_blocks_a_later_rollback(card):
    card.send(wait_for_answer=True)
    client = Client("token")

    async def post(_api, path, body):
        return 200, {"ok": True, "state": "answered", "answered_index": 1}

    asyncio.run(card.plugin.telegram_quiz.answer_from_callback(
        card.api, client, f"qz:{card.token}:1", cb_id="cb", update_id=7, lang="en", post=post))
    assert client.edits[0][2].endswith("\nAnswered: 2. postgres")
    assert card.record()["state"] == "answered"
    assert card.apply("expired_terminal") == []
    assert card.apply("open", wait_for_answer=False) == []


def test_unknown_or_anonymous_question_is_a_no_op(card):
    card.send()
    assert card.apply("answered", quiz_id="q-other", answered_index=0) == []
    assert card.apply("expired_terminal", task_id="") == []
    assert card.apply("answered", quiz_id="", task_id="") == []
    assert card.apply("mystery") == []
    assert card.api.logs == []


def test_failed_edit_is_logged_and_not_retried(card):
    card.send()
    Client.edit_result = False
    assert len(card.apply("expired_terminal")) == 1
    assert card.api.logs == [("warning", "Telegram quiz card edit failed (expired_terminal).")]


def test_card_follows_the_bridge_language(card):
    (card.api.state_dir / "settings.json").write_text(
        json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_LANGUAGE": "ru"}), encoding="utf-8")
    card.send(wait_for_answer=True)
    (edit,) = card.apply("answered", answered_index=0)
    assert edit[2].endswith("\nОтвет: 1. sqlite")


def test_the_manifest_declares_every_topic_register_subscribes(tmp_path):
    plugin = _load_plugin()
    api = Api(tmp_path)
    api.register_supervised_task = lambda *args, **kwargs: None
    api.subscribe_event = lambda topic, handler: api.subscriptions.append(topic)
    api.register_route = api.register_settings_section = lambda *args, **kwargs: None
    plugin.register_miniapp = lambda _api: None
    plugin.register(api)
    assert "chat.quiz_state" in api.subscriptions
    manifest = (_ROOT / "SKILL.md").read_text(encoding="utf-8")
    line = next(row for row in manifest.splitlines() if row.startswith("subscribe_events:"))
    declared = [topic.strip() for topic in line.split("[", 1)[1].rstrip("]").split(",")]
    assert declared == api.subscriptions


@pytest.mark.parametrize("record_state,event,expected", [
    (None, {"state": "answered", "answered_index": 5}, ("Q\nAnswered: 6.", [])),
    (None, {"state": "answered"}, ("Q\nAnswered.", [])),
    (None, {"state": "answered", "answered_index": True}, ("Q\nAnswered.", [])),
    (None, {"state": "superseded"}, ("Q\nReplaced by a newer question.", [])),
    (None, {"state": "open", "wait_for_answer": False},
     ("Q\nThe task continued; an answer is still accepted.\nReply to this message with your answer.", [])),
    ("superseded", {"state": "expired_terminal"}, None),
    ("expired_terminal", {"state": "open", "wait_for_answer": False}, None),
    ("answered", {"state": "superseded"}, None),
    (None, {"state": "unknown"}, None),
])
def test_lifecycle_edit_is_a_pure_forward_only_decision(record_state, event, expected):
    """The decision needs no Telegram client: text and keyboard from the card and the fact."""
    telegram_quiz = _load_plugin().telegram_quiz
    record = {"task_id": "t", "quiz_id": "q", "chat_id": 42, "message_id": 9, "options": [], "text": "Q"}
    if record_state:
        record["state"] = record_state
    assert telegram_quiz.lifecycle_edit(record, event, "en") == expected


def _fact(**fields):
    return {"type": "quiz_state", "quiz_id": "q1", "task_id": "task-1", "state": "answered",
            "ts": "2026-09-25T20:00:00+00:00", "chat_id": 1, "transport": {},
            "topic": "chat.quiz_state", **fields}


def _all_edits():
    return [edit for client in Client.instances for edit in client.edits]


def test_a_lifecycle_fact_that_outruns_the_cards_creation_is_applied_after_it(card, monkeypatch):
    """Finding 5: the host can publish ``quiz_state`` (a web answer) while the card's
    send is still in flight and nothing is remembered yet; the fact was dropped as an
    unknown card and the card kept its buttons. Creation and lifecycle edits are
    serialized per card, and a fact that arrives during creation is retained and
    applied once the card is remembered."""
    plugin, api = card.plugin, card.api
    gate = asyncio.Event()

    async def slow_send(self, chat_id, text, keyboard, parse_mode="HTML"):
        self.panels.append((chat_id, text, keyboard))
        await gate.wait()
        return 555

    monkeypatch.setattr(Client, "send_message_with_inline_keyboard", slow_send)

    async def scenario():
        creation = asyncio.ensure_future(plugin._make_quiz(api)(dict(_QUIZ)))
        await asyncio.sleep(0)
        assert card.record() is None  # the send is in flight, nothing remembered yet
        fact = asyncio.ensure_future(plugin._make_quiz_state(api)(_fact(answered_index=1)))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(creation, fact)

    asyncio.run(scenario())
    assert _all_edits() == [(42, 555,
        "Question: Which db?\n1. sqlite\n2. postgres\nContinuing meanwhile: sqlite meanwhile"
        "\nAnswered: 2. postgres", [])]
    assert card.record()["state"] == "answered"
    assert card.api.logs == []


def test_concurrent_edits_are_serialized_so_an_expiry_never_lands_over_an_answer(card, monkeypatch):
    """Finding 5: two facts in flight finished out of order — an expiry edit landing
    after the answered edit restored the buttons while the stored state said
    ``answered``. Per-card serialization makes each edit re-read the stored state
    before it is sent: the expiry lands first, the answer settles the card last."""
    card.send(wait_for_answer=True)
    plugin, api = card.plugin, card.api
    gate = asyncio.Event()
    sent = []  # every edit in the order Telegram would receive it (one log, not per client)

    async def slow_edit(self, chat_id, message_id, text, keyboard, parse_mode="HTML"):
        if keyboard:  # the expiry edit (buttons restored) is the slow one
            await gate.wait()
        sent.append((chat_id, message_id, text, keyboard))
        return True

    monkeypatch.setattr(Client, "edit_message_text_with_inline_keyboard", slow_edit)

    async def scenario():
        expiry = asyncio.ensure_future(plugin._make_quiz_state(api)(_fact(state="expired_terminal")))
        await asyncio.sleep(0)
        answer = asyncio.ensure_future(plugin._make_quiz_state(api)(_fact(answered_index=1)))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(expiry, answer)

    asyncio.run(scenario())
    assert [bool(edit[3]) for edit in sent] == [True, False]  # expiry first, then the answer
    assert sent[-1][2].endswith("\nAnswered: 2. postgres")
    assert card.record()["state"] == "answered"


@pytest.mark.parametrize("path", ["callback", "reply"])
def test_a_telegram_answer_waits_for_an_in_flight_lifecycle_edit_and_lands_last(card, monkeypatch, path):
    """Finding A1: the owner's own Telegram answer settled the card OUTSIDE the card
    lock, so an expiry edit already past its state read (``follow_lifecycle`` holds
    the lock while its edit is on the wire) landed after the answered edit and put
    the buttons back while the stored state said ``answered``. The answer takes the
    same lock: the toast never waits, the expiry lands first, the answered edit last."""
    card.send(wait_for_answer=True)
    plugin, api = card.plugin, card.api
    gate = asyncio.Event()
    sent = []  # every edit in the order Telegram would receive it

    async def slow_edit(self, chat_id, message_id, text, keyboard, parse_mode="HTML"):
        if keyboard:  # the expiry edit (buttons restored) is the slow one
            await gate.wait()
        sent.append((chat_id, message_id, text, keyboard))
        return True

    monkeypatch.setattr(Client, "edit_message_text_with_inline_keyboard", slow_edit)
    client = Client("token")

    async def post(_api, _path, body):
        return 200, {"ok": True, "state": "answered", "answered_index": 1}

    async def answer():
        if path == "callback":
            await plugin.telegram_quiz.answer_from_callback(
                api, client, f"qz:{card.token}:1", cb_id="cb", update_id=7, lang="en", post=post)
        else:
            await plugin.telegram_quiz.answer_from_reply(
                api, client, card.record(), "2. postgres", chat_id=42, update_id=7, lang="en", post=post)

    async def scenario():
        expiry = asyncio.ensure_future(plugin._make_quiz_state(api)(_fact(state="expired_terminal")))
        await asyncio.sleep(0)
        assert card.record()["state"] == "expired_terminal" and sent == []  # stored; its edit is in flight
        answered = asyncio.ensure_future(answer())
        await asyncio.sleep(0)
        assert client.toasts or client.sent  # the outcome is told at once, never behind the lock
        gate.set()
        await asyncio.gather(expiry, answered)

    asyncio.run(scenario())
    assert [bool(edit[3]) for edit in sent] == [True, False]  # expiry first, then the answer
    assert sent[-1][2].endswith("\nAnswered: 2. postgres")
    assert card.record()["state"] == "answered"
    assert card.api.logs == []
