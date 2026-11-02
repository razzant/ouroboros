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


def test_closed_question_states_its_closure_and_reason_without_controls_or_an_answer(card):
    card.send(wait_for_answer=True)
    closure = {"closed_reason": "The migration was cancelled.", "closed_actor": "model",
               "question_delivery": {"kind": "closure", "state": "owed"}}
    # A closure is never an answer: stray answer fields on the fact are not shown.
    (edit,) = card.apply("closed", answered_index=0, comment="yes", **closure)
    assert edit == (42, 555, "Question: Which db?\n1. sqlite\n2. postgres"
                             "\nQuestion closed.\nThe migration was cancelled.", [])
    assert card.record()["state"] == "closed"
    # Later or out-of-order facts never reopen, expire, replace or answer it.
    for later in ({"state": "open", "wait_for_answer": False}, {"state": "expired_terminal"},
                  {"state": "superseded"}, {"state": "answered", "answered_index": 1}):
        fields = dict(later)
        assert card.apply(fields.pop("state"), **fields) == []
    assert card.record()["state"] == "closed"
    # The discharged closure is re-published: the same edit, idempotent.
    closure["question_delivery"] = {"kind": "closure", "state": "delivered"}
    assert card.apply("closed", **closure) == [edit]
    assert card.api.logs == []


def test_an_answered_card_is_never_closed(card):
    card.send()
    card.apply("answered", answered_index=1)
    assert card.apply("closed", closed_reason="Not needed any more.") == []
    assert card.record()["state"] == "answered"


@pytest.mark.parametrize("path", ["callback", "reply"])
def test_a_telegram_answer_refused_as_closed_says_so_and_settles_the_card(card, path):
    card.send()
    client = Client("token")

    async def post(_api, _path, body):
        return 409, {"ok": False, "error": "quiz_closed", "state": "closed", "decision_id": body["decision_id"],
                     "closed_reason": "Answered in the Project instead.", "closed_actor": "owner"}

    if path == "callback":
        asyncio.run(card.plugin.telegram_quiz.answer_from_callback(
            card.api, client, f"qz:{card.token}:0", cb_id="cb", update_id=8, lang="en", post=post))
        assert client.toasts == [("cb", "Question closed.")]
    else:
        asyncio.run(card.plugin.telegram_quiz.answer_from_reply(
            card.api, client, card.record(), "sqlite", chat_id=42, update_id=8, lang="en", post=post))
    assert client.sent == [(42, "Question closed.")]  # a refused answer is a problem, by tap or by text
    assert client.edits == [(42, 555, "Question: Which db?\n1. sqlite\n2. postgres"
                             "\nContinuing meanwhile: sqlite meanwhile"
                             "\nQuestion closed.\nAnswered in the Project instead.", [])]
    assert card.record()["state"] == "closed"


@pytest.mark.parametrize("state,fields,tail,buttons", [
    ("closed", {"closed_reason": "Superseded by the new plan."}, "\nQuestion closed.\nSuperseded by the new plan.", False),
    ("answered", {"answered_index": 0}, "\nAnswered: 1. sqlite", False),
    ("expired_terminal", {}, "\nThe task finished; a late answer is accepted as your message."
                             "\nTap an option, or reply to this message with your own answer.", True),
])
def test_a_queued_ask_published_after_its_question_settled_shows_that_state(card, state, fields, tail, buttons):
    """A queued ask can reach Telegram after the question was answered or closed: the
    publication carries the canonical state, so the new card settles on it at once."""
    card.send(state=state, **fields)
    (edit,) = _all_edits()
    assert edit[2] == "Question: Which db?\n1. sqlite\n2. postgres\nContinuing meanwhile: sqlite meanwhile" + tail
    assert bool(edit[3]) is buttons
    assert card.record()["state"] == state
    Client.instances.clear()
    card.send(quiz_id="q2")  # an open publication is sent as it is, with no edit
    assert _all_edits() == []


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


def test_card_follows_the_install_language(card, tmp_path, monkeypatch):
    """The card's own lines come from the install's translation memory under the one
    interface language — not from a bridge-private setting or a shipped dictionary."""
    from ouroboros import i18n_memory as memory

    root = tmp_path / "data"
    card.plugin.telegram_i18n.configure(root)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    memory.update_memory(root, "ru", lambda doc: (memory.apply_generated(doc, {
        "code:tg.quiz.answered_line": {"text": "Ответ: {answer}"}}, model="test"), doc)[1], create=True)
    card.send(wait_for_answer=True)
    (edit,) = card.apply("answered", answered_index=0)
    assert edit[2].endswith("\nОтвет: 1. sqlite")
    # A line the memory lacks stays English rather than blank or guessed.
    assert "Question: Which db?" in edit[2]


def test_a_translated_receipt_never_pushes_the_answered_edit_past_one_message(card, tmp_path, monkeypatch):
    """The answered edit appends the recorded answer and the delivery line to the card. When the
    install's language makes those lines long, a card that would have fit as one message goes out
    the long way instead (the whole question as plain parts, then the compact keyboard message),
    so the later edit fits Telegram's limit and no question text is cut. Measured in UTF-16 units
    with the real renderer, astral characters included."""
    from ouroboros import i18n_memory as memory

    quiz = card.plugin.telegram_quiz
    root = tmp_path / "data"
    card.plugin.telegram_i18n.configure(root)
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    saved = "Ответ сохранён 📨 и ждёт, пока задача продолжит работу; " * 6
    memory.update_memory(root, "ru", lambda doc: (memory.apply_generated(doc, {
        "code:tg.quiz.answered_line": {"text": "Ответ, записанный хостом 🗂: {answer}"},
        "code:tg.quiz.status_saved": {"text": saved.strip()}}, model="test"), doc)[1], create=True)

    def body(question):
        return quiz.render_quiz_text(question, ["sqlite", "postgres"], "", "sqlite meanwhile", lang="ru",
                                     option_details=["", ""], recommended_index=None)
    # The largest question the English-sized reserve still sends as ONE message, astral-heavy.
    budget = quiz._TELEGRAM_TEXT_LIMIT - quiz._ANSWERED_EDIT_RESERVE - quiz._u16len("\n" + quiz.hint("ru"))
    filler = budget - quiz._u16len(body(""))
    question = "🙂" * (filler // 2) + "q" * (filler % 2)
    assert quiz._u16len(f"{body(question)}\n{quiz.hint('ru')}") + quiz._ANSWERED_EDIT_RESERVE == quiz._TELEGRAM_TEXT_LIMIT

    card.send(question=question)
    words = "Используй postgres, но с ночными бэкапами 💾"
    (edit,) = card.apply("answered", comment=words, question_delivery=_owed("awaiting_recipient_read"))
    assert quiz._u16len(edit[2]) <= quiz._TELEGRAM_TEXT_LIMIT
    assert edit[2].endswith(f"\nОтвет, записанный хостом 🗂: {words}\n{saved.strip()}")  # the answer is never cut here
    client = Client.instances[0]  # the card's own sender; the edit came through a later client
    assert client.sent == [(42, body(question))]  # the whole question, as rendered, before the keyboard message
    assert [text for _, text, _ in client.panels] == [f"1. sqlite\n2. postgres\n{quiz.hint('ru')}"]


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
    (None, {"state": "closed", "answered_index": 0, "comment": "yes"}, ("Q\nQuestion closed.", [])),
    ("expired_terminal", {"state": "closed", "closed_reason": "x" * 300}, ("Q\nQuestion closed.\n" + "x" * 200 + "…", [])),
    ("closed", {"state": "answered", "answered_index": 0}, None),
    ("answered", {"state": "closed", "closed_reason": "late"}, None),
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
        # A tap's toast is told at once, never behind the lock; a text answer's receipt is the card.
        assert client.toasts if path == "callback" else client.sent == []
        gate.set()
        await asyncio.gather(expiry, answered)

    asyncio.run(scenario())
    assert [bool(edit[3]) for edit in sent] == [True, False]  # expiry first, then the answer
    assert sent[-1][2].endswith("\nAnswered: 2. postgres")
    assert card.record()["state"] == "answered"
    assert client.sent == [] and card.api.logs == []


_SAVED = "Answer saved · the task reads it when it continues"
_BASE = "Question: Which db?\n1. sqlite\n2. postgres\nContinuing meanwhile: sqlite meanwhile"


def _owed(reason=""):
    return {"kind": "answer", "state": "owed", **({"reason_code": reason} if reason else {})}


_DELIVERED = {"kind": "answer", "state": "delivered"}


@pytest.mark.parametrize("winner,shown", [
    ({"comment": "neither — use duckdb"}, "neither — use duckdb"),
    ({"answered_index": 1, "comment": "with nightly backups"}, "2. postgres — with nightly backups"),
], ids=["comment", "option-comment"])
def test_a_delivery_only_fact_keeps_the_recorded_winner(card, winner, shown):
    """The read acknowledgement publishes the delivery alone: absence never clears the winner,
    and a confirmed delivery removes the pending line while the answer stays."""
    card.send()
    (edit,) = card.apply("answered", question_delivery=_owed("awaiting_recipient_read"), **winner)
    assert edit[2] == f"{_BASE}\nAnswered: {shown}\n{_SAVED}"
    (edit,) = card.apply("answered", question_delivery=_DELIVERED)
    assert edit[2] == f"{_BASE}\nAnswered: {shown}"
    record = card.record()
    assert {key: record.get(key) for key in ("answered_index", "comment")} == {
        "answered_index": winner.get("answered_index"), "comment": winner.get("comment")}


def test_a_stale_owed_fact_never_takes_back_a_confirmed_delivery(card):
    card.send()
    card.apply("answered", answered_index=1, question_delivery=_DELIVERED)
    (edit,) = card.apply("answered", answered_index=1, question_delivery=_owed("awaiting_recipient_read"))
    assert edit[2] == f"{_BASE}\nAnswered: 2. postgres"
    assert card.record()["question_delivery"] == _DELIVERED


@pytest.mark.parametrize("order", ["full-then-sparse", "sparse-then-full"])
def test_facts_that_outrun_the_cards_creation_keep_the_winner_and_the_delivery(card, order):
    """Two answered facts can both arrive before the card is remembered; the retained fact
    keeps the winner one of them names and the furthest delivery either reports."""
    full = _fact(answered_index=1, question_delivery=_owed("awaiting_recipient_read"))
    sparse = _fact(question_delivery=_DELIVERED)
    for fact in ([full, sparse] if order == "full-then-sparse" else [sparse, full]):
        asyncio.run(card.plugin._make_quiz_state(card.api)(fact))
    assert card.record() is None and _all_edits() == []  # retained, nothing to edit yet
    card.send()
    assert _all_edits() == [(42, 555, f"{_BASE}\nAnswered: 2. postgres", [])]
    assert card.record()["answered_index"] == 1
    assert card.api.logs == []


def test_host_answer_and_its_read_reach_the_card_through_the_real_producers(card, tmp_path, monkeypatch):
    """The host's own decision ingress, mailbox read and acknowledgement drive the card: the
    tap's toast says the answer waits for the task, and the read keeps the recorded winner."""
    import queue
    from types import SimpleNamespace

    import supervisor.message_bus as mb
    from ouroboros.gateway import task_decision as td
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from supervisor.events_runtime_controls import _handle_owner_message_injected
    from tests.test_question_answer_routing import runtime

    host = tmp_path / "host"
    host.mkdir()
    runtime(host, monkeypatch)
    published = []
    monkeypatch.setattr(mb, "publish_event", lambda topic, event: published.append((topic, dict(event))))
    card.send(task_id="asker", quiz_id="q")
    token = card.plugin.telegram_quiz.mint_token("asker", "q")
    client = Client("token")

    async def post(_api, path, body):
        assert path == "/chat/decision"
        return await td.answer_decision(host, body, source="skill:telegram")

    async def deliver_facts():
        facts = [event for topic, event in published if topic == "chat.quiz_state"]
        published.clear()
        for fact in facts:
            await card.plugin._make_quiz_state(card.api)(fact)
        return facts

    asyncio.run(card.plugin.telegram_quiz.answer_from_callback(
        card.api, client, f"qz:{token}:1", cb_id="cb", update_id=90, lang="en", post=post))
    asyncio.run(deliver_facts())
    assert client.toasts == [("cb", f"✅ {_SAVED}.")]
    expected = f"{_BASE}\nAnswered: 2. postgres"
    assert client.edits[-1][2] == f"{expected}\n{_SAVED}"

    events = queue.Queue()
    ctx = SimpleNamespace(task_id="asker", task_metadata={}, drive_root=host, budget_drive_root=host)
    _drain_incoming_messages([], queue.Queue(), host, "asker", events, set(), owner_ctx=ctx)
    acks = []
    while not events.empty():
        event = events.get_nowait()
        acks.extend([event] if event.get("quiz_ack") else [])
    assert acks, "the read is acknowledged"
    for event in acks:
        _handle_owner_message_injected(event, SimpleNamespace(DRIVE_ROOT=host, bridge=mb.get_bridge()))
    facts = asyncio.run(deliver_facts())
    assert facts and facts[-1]["question_delivery"]["state"] == "delivered"
    assert facts[-1].get("answered_index") == 1, "the acknowledgement names the recorded winner"
    edits = [edit for instance in Client.instances for edit in instance.edits]
    assert edits[-1][2] == expected
    record = card.plugin.telegram_quiz.quiz_for_token(card.api, token)
    assert record["question_delivery"]["state"] == "delivered" and record["answered_index"] == 1
    assert card.api.logs == []


_PENDING = "Answer recorded · delivery pending"


def _owed_ingress(card, tmp_path, monkeypatch, failure=None):
    """The host's real decision ingress for one question (``asker``/``q``) whose answer delivery keeps
    failing: the history write raises, or the task's mailbox refuses the copy. ``answer`` runs one
    Telegram tap or reply through it and returns the host's ``(status, payload)``."""
    import supervisor.message_bus as mb
    from ouroboros.gateway import task_decision as td
    from ouroboros.owner_quiz import quiz_states
    from tests.test_question_answer_routing import runtime

    host = tmp_path / "host"
    host.mkdir()
    runtime(host, monkeypatch)
    monkeypatch.setattr(mb, "publish_event", lambda *_a, **_k: None)  # the card hears only its own answer
    if failure == "history":
        def unavailable(*_a, **_k):
            raise OSError("history store unavailable")
        monkeypatch.setattr(td, "_record_quiz_answer_history", unavailable)
    elif failure == "mailbox":
        monkeypatch.setattr("ouroboros.owner_mailbox.write_owner_message", lambda *_a, **_k: False)
    card.send(task_id="asker", quiz_id="q")
    quiz, client, replies = card.plugin.telegram_quiz, Client("token"), []
    token = quiz.mint_token("asker", "q")

    async def post(_api, path, body):
        assert path == "/chat/decision"
        replies.append(await td.answer_decision(host, body, source="skill:telegram"))
        return replies[-1]

    def answer(path, given, update_id):
        if path == "tap":
            asyncio.run(quiz.answer_from_callback(card.api, client, f"qz:{token}:{given}", cb_id="cb",
                                                  update_id=update_id, lang="en", post=post))
        else:
            asyncio.run(quiz.answer_from_reply(card.api, client, quiz.quiz_for_token(card.api, token), given,
                                               chat_id=42, update_id=update_id, lang="en", post=post))
        return replies[-1]

    def web(**answered):
        return asyncio.run(td.answer_decision(
            host, {"request_id": "web-1", "decision_id": "quiz:asker:q", **answered}, source="web"))

    return types.SimpleNamespace(client=client, answer=answer, web=web, block=lambda: quiz_states(host, "asker")["q"])


@pytest.mark.parametrize("failure", ["history", "mailbox"])
@pytest.mark.parametrize("path,earlier,given,shown", [
    ("tap", {"option_index": 0}, 1, "1. sqlite"),
    ("tap", {"comment": "Use duckdb"}, 1, "Use duckdb"),
    ("tap", {"option_index": 0, "comment": "only with nightly backups"}, 0, "1. sqlite — only with nightly backups"),
    ("reply", {"option_index": 0}, "Use duckdb", "1. sqlite"),
    ("reply", {"comment": "Use duckdb"}, "Use mysql", "Use duckdb"),
    ("reply", {"option_index": 1, "comment": "with nightly backups"}, "with nightly backups",
     "2. postgres — with nightly backups"),
], ids=["tap-option", "tap-comment", "tap-option-comment", "reply-option", "reply-comment", "reply-option-comment"])
def test_a_later_answer_refused_behind_the_winners_failed_delivery_is_a_conflict(
        card, tmp_path, monkeypatch, failure, path, earlier, given, shown):
    """#2109: before refusing a later answer the host retries the earlier winner's owed delivery, and when
    that retry fails the later request gets a 503 carrying the EARLIER winner. Through the real ingress it is
    a conflict: the card keeps the recorded answer and its pending delivery, a tap is also told by its toast,
    either answer gets the one line naming the winner, and the winner stays recorded."""
    host = _owed_ingress(card, tmp_path, monkeypatch, failure)
    assert host.web(**earlier)[0] == 503  # recorded, its delivery owed
    winner = host.block()
    status, payload = host.answer(path, given, 91)
    assert status == 503 and payload["state"] == "answered"  # the winner's retried delivery failed again
    assert host.client.edits == [(42, 555, f"{_BASE}\nAnswered: {shown}\n{_PENDING}", [])]
    assert host.client.sent == [(42, f"Not accepted: this question was already answered. Recorded answer: {shown}")]
    assert host.client.toasts == ([("cb", "This question was already answered.")] if path == "tap" else [])
    keys = ("request_id", "answered_index", "comment")
    assert [host.block().get(key) for key in keys] == [winner.get(key) for key in keys]


@pytest.mark.parametrize("failure", ["history", "mailbox"])
@pytest.mark.parametrize("path,earlier,given,shown", [
    ("tap", {"option_index": 1}, 1, "2. postgres"),
    ("reply", {"comment": "Use duckdb"}, "Use duckdb", "Use duckdb"),
    ("tap", None, 1, "2. postgres"),
    ("reply", None, "Use duckdb", "Use duckdb"),
], ids=["tap-same-answer", "reply-same-answer", "tap-same-update", "reply-same-update"])
def test_the_same_answer_behind_a_failed_delivery_stays_accepted(
        card, tmp_path, monkeypatch, failure, path, earlier, given, shown):
    """No false changed answer: the same answer given on the web card, or this very update answered again,
    is the recorded winner, so its 503 stays the receipt of an answer recorded with its delivery pending."""
    host = _owed_ingress(card, tmp_path, monkeypatch, failure)
    if earlier is None:
        assert host.answer(path, given, 91)[0] == 503
        for told in (host.client.edits, host.client.toasts, host.client.sent):
            told.clear()
    else:
        assert host.web(**earlier)[0] == 503
    status, payload = host.answer(path, given, 91)
    assert status == 503 and payload["state"] == "answered"
    assert host.client.edits == [(42, 555, f"{_BASE}\nAnswered: {shown}\n{_PENDING}", [])]
    assert (host.client.toasts, host.client.sent) == (([("cb", f"✅ {_PENDING}.")] if path == "tap" else []), [])


@pytest.mark.parametrize("path,given", [("tap", 1), ("reply", "Use duckdb")])
def test_an_answer_the_host_could_not_record_stays_unknown(card, tmp_path, monkeypatch, path, given):
    """Genuinely unknown stays unknown: the ingress could not record the answer, its 503 names no winner,
    the card stays open and the owner is asked to send the same answer again."""
    host = _owed_ingress(card, tmp_path, monkeypatch)

    def unavailable(*_a, **_k):
        raise OSError("question store unavailable")

    monkeypatch.setattr("ouroboros.owner_quiz.record_answered", unavailable)
    status, payload = host.answer(path, given, 93)
    assert status == 503 and payload["state"] == "open" and not {"answered_index", "comment"} & set(payload)
    line = "The answer was not confirmed (HTTP 503), so it may not be recorded. Send the same answer again later."
    assert host.client.edits == [] and host.block()["state"] == "open"
    assert (host.client.toasts, host.client.sent) == ([("cb", line)] if path == "tap" else [], [(42, line)])
