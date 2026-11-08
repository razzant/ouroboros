"""A Telegram message id means something only within (numeric bot id, chat id) (#2557).

The real chain on synthetic data: the shipped poller and bridge handlers, real HTTP on loopback,
the real Host Service and its canonical acceptance in an isolated data root. Only the Bot API
server is a fake, numbering each bot's updates and messages on its own, as Telegram does: in a
private chat the chat id is the owner's for every bot, so two bots' numbers collide. Assertions
read what the host stored and what each bot was asked on the wire. Nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ouroboros.gateway.host_service import create_host_service_app
from tests._dialogue import chat_rows
from tests.test_chat_attachments import _web_bridge
from tests.test_host_service_api import _seed_token
from tests.test_telegram_api_base_loopback import _BotServer
from tests.test_telegram_proxy import _Api, _load

# The shared native-skill loader replaces its package entries in sys.modules; the host's globals are patched.

A, A_ROTATED, B = "111111:SECRET-A", "111111:ROTATED-A", "222222:SECRET-B"
A_ID, B_ID = "111111", "222222"
QUESTION = {"chat_id": 1, "task_id": "asker", "quiz_id": "q", "question": "Which database?",
            "options": [{"label": "sqlite"}, {"label": "postgres"}]}


class _Bot:
    """One bot at one server: its own update queue, message numbers and call log."""

    def __init__(self, first_message: int):
        self.queue, self.next_update, self.next_message = [], 1, first_message
        self.calls, self.sent = [], set()

    def push(self, update: dict) -> int:
        update_id, self.next_update = self.next_update, self.next_update + 1
        self.queue.append({"update_id": update_id, **update})
        return update_id

    def methods(self, name: str) -> list[dict]:
        return [form for method, form in self.calls if method == name]


class _BotApi(_BotServer):
    """The cloud's documented answers for any token, each bot keeping its own numbers."""

    def __init__(self):
        super().__init__()
        self.bots: dict[str, _Bot] = {}
        self.first_message = {A_ID: 100, B_ID: 200}
        self.after_send = None

    def bot(self, token: str) -> _Bot:
        head = str(token).split(":", 1)[0]
        return self.bots.setdefault(head, _Bot(self.first_message.get(head, 300)))

    def respond(self, http_method, path, form):
        token, method = path[len("/bot"):].split("/", 1)
        bot = self.bot(token)
        bot.calls.append((method, form))
        result: object = True
        if method == "getMe":
            result = {"id": int(token.split(":")[0]) if token.split(":")[0].isdigit() else 1, "is_bot": True}
        elif method == "getUpdates":
            offset = int(form.get("offset") or 0)
            bot.queue = [update for update in bot.queue if update["update_id"] >= offset]
            result = list(bot.queue)
        elif method == "editMessageText":
            if int(form["message_id"]) not in bot.sent:
                return 400, json.dumps({"ok": False, "error_code": 400, "description":
                                        "Bad Request: message to edit not found"}).encode(), "application/json"
            result = {"message_id": int(form["message_id"])}
        elif method.startswith("send") and method != "sendChatAction":
            mid, bot.next_message = bot.next_message, bot.next_message + 1
            bot.sent.add(mid)
            result = {"message_id": mid}
            if self.after_send is not None:
                self.after_send()
        return 200, json.dumps({"ok": True, "result": result}).encode(), "application/json"


def _owner(mid: int, text: str, *, reply: int = 0) -> dict:
    return {"message": {"message_id": mid, "date": int(time.time()) + 5, "text": text,
                        "chat": {"id": 42, "type": "private"}, "from": {"id": 42, "first_name": "Owner"},
                        **({"reply_to_message": {"message_id": reply}} if reply else {})}}


def _tap(callback_id: str, data: str, mid: int) -> dict:
    return {"callback_query": {"id": callback_id, "data": data, "from": {"id": 42},
                               "message": {"message_id": mid, "chat": {"id": 42, "type": "private"}}}}


async def _until(condition, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


class _World:
    def __init__(self, tmp_path: Path, monkeypatch, bridge):
        self.plugin, _api, self.notifier = _load()
        monkeypatch.setattr(self.plugin, "_HONOR_ENV_PROXIES", False)
        self.data, self.bridge = tmp_path, bridge
        self.state = tmp_path / "state" / "skills" / "telegram"
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
        self.api = _Api(self.state, {"TELEGRAM_BOT_TOKEN": A})
        self.api.get_runtime_info = lambda: {"data_dir": str(tmp_path)}
        self.bots: _BotApi | None = None

    def use(self, token: str) -> None:
        self.api.granted["TELEGRAM_BOT_TOKEN"] = token

    def serve(self, server: _BotServer) -> None:
        self.plugin.merge_settings(self.state, {"TELEGRAM_API_BASE": server.origin})

    def run(self, scenario) -> None:
        async def main():
            async with _BotApi() as self.bots:
                self.serve(self.bots)
                await scenario()
        asyncio.run(main())

    async def deliver(self, token: str, *updates: dict, server: _BotApi | None = None) -> None:
        """A poller of ``token`` runs until it has handled ``updates`` and saved its batch."""
        self.use(token)
        bot = (server or self.bots).bot(token)
        seen = len(bot.methods("getUpdates"))
        for update in updates:
            bot.push(update)
        poller = asyncio.create_task(self.plugin._poller(self.api))
        try:
            await _until(lambda: len(bot.methods("getUpdates")) >= seen + 2)
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller

    def forget_cursor(self) -> None:
        """The batch was never confirmed: the next poller is given it again and asks the host."""
        (self.state / "poll_offset.json").unlink()

    def rows(self) -> list[dict]:
        return [row for row in chat_rows(self.data) if row.get("source") == "skill:telegram"]

    def ids(self) -> list[str]:
        return [row["client_message_id"] for row in self.rows()]

    def texts(self) -> list[str]:
        return [row["text"] for row in self.rows()]

    def dispatched(self) -> int:
        return self.bridge._inbox.qsize()

    def refs(self) -> dict:
        return json.loads((self.state / "message_refs.json").read_text(encoding="utf-8"))

    def address(self, index: int = -1) -> str:
        from ouroboros.chat_chain import format_address, row_address
        return format_address(row_address(self.rows()[index]))

    def reference(self, index: int) -> dict:
        row = self.rows()[index]
        return {"address": self.address(index), "chat_id": 42, "role": "user", "author_label": "You",
                "kind": "message", "snippet": row["text"], "ts": row["ts"], "task_id": ""}

    def outbound(self, event: dict):
        return self.plugin._make_outbound(self.api)(event)


@contextlib.contextmanager
def _host(world: _World):
    import uvicorn

    _seed_token(world.data, skill="telegram", token="skill-token", permissions=["inject_chat"])
    app = create_host_service_app(world.data, bridge_getter=lambda: world.bridge)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline, "the host service did not start"
        time.sleep(0.01)
    try:
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(10)


@pytest.fixture
def world(tmp_path, monkeypatch):
    bridge, _frames = _web_bridge(tmp_path, monkeypatch)
    item = _World(tmp_path, monkeypatch, bridge)
    with _host(item) as port:
        monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", str(port))
        yield item


@pytest.fixture
def quiz_world(tmp_path, monkeypatch):
    """The same chain with a real open question of a running task behind /chat/decision."""
    from tests.test_question_answer_routing import runtime

    _app, bridge, _queue, _task = runtime(tmp_path, monkeypatch)
    item = _World(tmp_path, monkeypatch, bridge)
    with _host(item) as port:
        monkeypatch.setenv("OUROBOROS_HOST_SERVICE_PORT", str(port))
        item.token = item.plugin.telegram_quiz.mint_token("asker", "q")
        yield item


def _quiz_block(world: _World) -> dict:
    from ouroboros.owner_quiz import quiz_states
    return quiz_states(world.data, "asker")["q"]


# --- 1-4: another bot's numbers are its own; the host's dedup is untouched --------------------

def test_another_bot_with_the_same_numbers_and_other_words_is_a_new_dispatched_row(world):
    async def scenario():
        await world.deliver(A, _owner(50, "words to the first bot"))
        await world.deliver(B, _owner(50, "words to the new bot"))

    world.run(scenario)
    assert world.texts() == ["words to the first bot", "words to the new bot"]
    assert world.ids() == [f"tg:{A_ID}:42:50", f"tg:{B_ID}:42:50"]
    assert world.dispatched() == 2
    assert not world.bots.bot(B).methods("sendMessage"), "no 'update failed' notice: nothing was refused"


def test_another_bot_with_the_same_numbers_and_the_same_words_is_not_merged(world):
    async def scenario():
        await world.deliver(A, _owner(50, "да"))
        await world.deliver(B, _owner(50, "да"))

    world.run(scenario)
    assert world.ids() == [f"tg:{A_ID}:42:50", f"tg:{B_ID}:42:50"]
    assert world.dispatched() == 2


def test_a_replay_of_the_new_bots_update_is_exactly_one_row(world):
    async def scenario():
        await world.deliver(A, _owner(50, "first"))
        await world.deliver(B, _owner(50, "second"))
        world.forget_cursor()
        await world.deliver(B, _owner(50, "second"))

    world.run(scenario)
    assert world.ids() == [f"tg:{A_ID}:42:50", f"tg:{B_ID}:42:50"]
    assert world.dispatched() == 2
    assert not world.bots.bot(B).methods("sendMessage")


def test_the_same_bot_id_with_other_words_is_still_refused_by_the_host(world):
    async def scenario():
        await world.deliver(B, _owner(50, "second"))
        world.forget_cursor()
        await world.deliver(B, _owner(50, "not the same message"))

    world.run(scenario)
    assert world.texts() == ["second"] and world.dispatched() == 1
    (notice,) = world.bots.bot(B).methods("sendMessage")
    assert "42" == notice["chat_id"]
    assert any("Telegram update processing failed" in message for _level, message in world.api.logs)


# --- 5: a rotated token and another server keep the identity ----------------------------------

def test_a_rotated_token_keeps_the_identity_and_rejoins_its_message(world):
    async def scenario():
        await world.deliver(A, _owner(50, "before rotation"))
        world.forget_cursor()
        await world.deliver(A_ROTATED, _owner(50, "before rotation"), _owner(51, "after rotation"))

    world.run(scenario)
    assert world.ids() == [f"tg:{A_ID}:42:50", f"tg:{A_ID}:42:51"]
    assert world.dispatched() == 2
    assert "SECRET" not in (world.state / "message_refs.json").read_text(encoding="utf-8")


def test_another_bot_api_server_keeps_the_identity_and_rejoins_its_message(world):
    async def scenario():
        await world.deliver(A, _owner(50, "on the first server"))
        async with _BotApi() as other:
            world.serve(other)
            await world.deliver(A, _owner(50, "on the first server"), _owner(51, "on the second"), server=other)

    world.run(scenario)
    assert world.ids() == [f"tg:{A_ID}:42:50", f"tg:{A_ID}:42:51"]
    assert world.dispatched() == 2


def test_a_token_naming_no_bot_keeps_the_botless_grammar_in_its_own_explicit_branch(world):
    async def scenario():
        await world.deliver("not-numeric", _owner(50, "fixture token"))

    world.run(scenario)
    assert world.ids() == ["tg:42:50"]
    assert list(world.refs()["messages"]) == ["42:50"] and "bots" not in world.refs()


# --- 6: no anchor crosses bots ------------------------------------------------------------------

def test_a_reply_never_takes_a_reference_from_another_bots_message(world):
    async def steps():
        await world.deliver(A, _owner(10, "the first owner message"))
        await world.outbound({"chat_id": 1, "text": "Answer of the first bot", "row_address": world.address(0)})
        await world.deliver(A, _owner(11, "a reply to the first bot", reply=100))
        await world.deliver(B, _owner(12, "a reply to the new bot's message 100", reply=100))

    world.run(steps)
    by_text = {row["text"]: row for row in world.rows()}
    assert by_text["a reply to the first bot"]["reply_to"]["address"] == world.address(0)
    assert not by_text["a reply to the new bot's message 100"].get("reply_to")


def test_native_reply_and_latest_owner_belong_to_the_sending_bot(world):
    async def steps():
        await world.deliver(A, _owner(900, "A early"), _owner(901, "A latest"))
        await world.deliver(B, _owner(5, "B early"), _owner(6, "B latest"))
        early_a, early_b, latest_b = world.reference(0), world.reference(2), world.reference(3)
        world.use(A)
        await world.outbound({"chat_id": 1, "text": "to A's early", "reply_to": early_a})
        world.use(B)
        await world.outbound({"chat_id": 1, "text": "A's row, B sends", "reply_to": early_a})
        await world.outbound({"chat_id": 1, "text": "to B's early", "reply_to": early_b})
        await world.outbound({"chat_id": 1, "text": "to B's latest", "reply_to": latest_b})

    world.run(steps)
    sent_a = world.bots.bot(A).methods("sendMessage")
    sent_b = {form["text"]: form for form in world.bots.bot(B).methods("sendMessage")}
    assert [form.get("reply_to_message_id") for form in sent_a] == ["900"]
    assert "reply_to_message_id" not in sent_b["A's row, B sends"]
    assert sent_b["to B's early"]["reply_to_message_id"] == "5"
    assert "reply_to_message_id" not in sent_b["to B's latest"], "B's own latest, below A's old numbers"


def test_subagent_card_and_silent_progress_edit_only_their_own_bots_bubble(world):
    card = {"subagent_event": "running", "subagent_task_id": "child-1", "chat_id": 1, "text": "working"}
    note = {"task_id": "t1", "is_progress": True, "role": "assistant", "system_type": "model_narration", "chat_id": 1}

    async def steps():
        world.plugin.merge_settings(world.state, {"TELEGRAM_SILENT_MODE": "on", "TELEGRAM_MIRROR_PROGRESS": "on"})
        for token, step in ((A, "one"), (B, "two"), (A, "three"), (B, "four")):
            world.use(token)
            await world.outbound({**card, "text": f"card {step}"})
        for token, step in ((A, "one"), (A, "two"), (B, "three")):
            world.use(token)
            await world.outbound({**note, "text": f"note {step}"})

    world.run(steps)
    bot_a, bot_b = world.bots.bot(A), world.bots.bot(B)
    def note_of(form):  # a card leads with its status header
        return form["text"].rsplit("\n", 1)[-1]

    assert [(note_of(form), form["message_id"]) for form in bot_a.methods("editMessageText")] == [
        ("card three", "100"), ("note two", "101")]
    assert [(note_of(form), form["message_id"]) for form in bot_b.methods("editMessageText")] == [("card four", "200")]
    assert [note_of(form) for form in bot_b.methods("sendMessage")] == ["card two", "note three"]


# --- quiz cards: one question, a card at each bot, A→B→A ---------------------------------------

def _closed() -> dict:
    return {"type": "quiz_state", "quiz_id": "q", "task_id": "asker", "state": "closed", "closed_reason": "no longer needed"}


def test_a_lifecycle_edit_goes_to_the_current_bots_own_card_and_a_return_finds_its_earlier_card(quiz_world):
    world = quiz_world

    async def steps():
        make_quiz, follow = world.plugin._make_quiz(world.api), world.plugin._make_quiz_state(world.api)
        world.use(A)
        await make_quiz(QUESTION)
        world.use(B)
        await make_quiz(QUESTION)
        await follow(_closed())
        assert [form["message_id"] for form in world.bots.bot(B).methods("editMessageText")] == ["200"]
        assert not world.bots.bot(A).methods("editMessageText")
        world.use(A)  # A→B→A: the return finds A's own card, not B's number
        await follow(_closed())

    world.run(steps)
    assert [form["message_id"] for form in world.bots.bot(A).methods("editMessageText")] == ["100"]
    assert [form["message_id"] for form in world.bots.bot(B).methods("editMessageText")] == ["200"]


def test_a_reply_answers_only_the_polling_bots_card_and_settles_that_card(quiz_world):
    world = quiz_world

    async def steps():
        make_quiz = world.plugin._make_quiz(world.api)
        world.use(A)
        await make_quiz(QUESTION)
        world.use(B)
        await make_quiz(QUESTION)
        # B's own message 100 is no card, even if A's card has that number here.
        world.bots.bot(B).sent.add(100)
        await world.deliver(B, _owner(30, "just a remark", reply=100))
        assert world.texts() == ["just a remark"] and _quiz_block(world)["state"] == "open"
        await world.deliver(B, _owner(31, "postgres, in my words", reply=200))

    world.run(steps)
    assert _quiz_block(world)["comment"] == "postgres, in my words"
    (edit,) = world.bots.bot(B).methods("editMessageText")
    assert edit["message_id"] == "200"
    assert not world.bots.bot(A).methods("editMessageText")


def test_a_fact_while_another_bot_is_current_waits_for_a_card_of_that_bot_and_edits_no_foreign_number(quiz_world):
    """Facts name the question, cards are the bots': a fact B cannot show stays retained for B's own
    card, A's earlier card is untouched meanwhile, and A's next fact still finds and settles it."""
    world = quiz_world

    async def steps():
        make_quiz, follow = world.plugin._make_quiz(world.api), world.plugin._make_quiz_state(world.api)
        world.use(A)
        await make_quiz(QUESTION)
        world.use(B)
        await follow(_closed())
        assert not world.bots.bot(A).methods("editMessageText")
        assert not world.bots.bot(B).methods("editMessageText")
        await make_quiz(QUESTION)  # its card shows the retained fact at once
        assert [form["message_id"] for form in world.bots.bot(B).methods("editMessageText")] == ["200"]
        world.use(A)
        await follow(_closed())

    world.run(steps)
    assert [form["message_id"] for form in world.bots.bot(A).methods("editMessageText")] == ["100"]
    assert len(world.bots.bot(B).methods("editMessageText")) == 1


def test_a_later_answer_settles_the_other_bots_card_on_its_own_number(quiz_world):
    world = quiz_world

    async def steps():
        make_quiz = world.plugin._make_quiz(world.api)
        world.use(A)
        await make_quiz(QUESTION)
        world.use(B)
        await make_quiz(QUESTION)
        await world.deliver(B, _tap("cb-b", f"qz:{world.token}:1", 200))
        await world.deliver(A, _tap("cb-a", f"qz:{world.token}:0", 100))

    world.run(steps)
    assert _quiz_block(world)["answered_index"] == 1
    assert [form["message_id"] for form in world.bots.bot(B).methods("editMessageText")] == ["200"]
    assert [form["message_id"] for form in world.bots.bot(A).methods("editMessageText")] == ["100"]


def test_equal_update_ids_of_two_bots_never_confirm_a_different_answer_through_the_real_decision_ingress(quiz_world):
    """``owner_quiz.record_answered`` answers an equal request id as a duplicate before it looks at
    the answer: unscoped, B's other choice would be told it was recorded."""
    world = quiz_world

    async def steps():
        make_quiz = world.plugin._make_quiz(world.api)
        world.use(A)
        await make_quiz(QUESTION)
        world.use(B)
        await make_quiz(QUESTION)
        await world.deliver(A, _tap("cb-same", f"qz:{world.token}:0", 100))
        await world.deliver(B, _tap("cb-same", f"qz:{world.token}:1", 200))

    world.run(steps)
    (toast,) = world.bots.bot(B).methods("answerCallbackQuery")
    assert not toast.get("text", "").startswith("✅"), "B's other choice was not recorded"
    block = _quiz_block(world)
    assert block["answered_index"] == 0 and block["request_id"] == f"tg:{A_ID}:cb:cb-same"
    (lost,) = world.bots.bot(B).methods("sendMessage")[-1:]
    assert "sqlite" in lost["text"], "B is told the recorded answer"


# --- 7: Settings may already name another bot ---------------------------------------------------

def test_a_decision_request_names_the_tap_on_every_server_not_one_servers_queue_number(quiz_world):
    """telegram_poll_binding: each server numbers its own queue; callback and message ids are the
    same on every server. The same tap read again on another server is the same request; another
    tap at an equal update id on another server is not."""
    world = quiz_world

    async def steps():
        world.use(A)
        await world.plugin._make_quiz(world.api)(QUESTION)
        await world.deliver(A, _tap("cb-one", f"qz:{world.token}:0", 100))
        async with _BotApi() as other:
            world.serve(other)
            other.bot(A).next_update = 40
            world.forget_cursor()  # its handled batch would confirm the tap before any request
            await world.deliver(A, _tap("cb-one", f"qz:{world.token}:0", 100), server=other)
            (again,) = other.bot(A).methods("answerCallbackQuery")
            assert again["text"].startswith("✅"), "the same tap on another queue number"
        async with _BotApi() as third:
            world.serve(third)  # its update id 1 equals the first server's; this is another tap
            await world.deliver(A, _tap("cb-two", f"qz:{world.token}:1", 100), server=third)
            (refused,) = third.bot(A).methods("answerCallbackQuery")
            assert not refused["text"].startswith("✅"), "an equal queue number is not the same request"

    world.run(steps)
    block = _quiz_block(world)
    assert block["answered_index"] == 0 and block["request_id"] == f"tg:{A_ID}:cb:cb-one"


def test_a_reply_answer_is_named_by_its_bot_chat_and_message(quiz_world):
    world = quiz_world

    async def steps():
        world.use(A)
        await world.plugin._make_quiz(world.api)(QUESTION)
        await world.deliver(A, _owner(31, "in my words", reply=100))

    world.run(steps)
    assert _quiz_block(world)["request_id"] == f"tg:{A_ID}:m:42:31"


def test_settings_already_naming_another_bot_do_not_move_the_running_pollers_ids(world):
    async def steps():
        world.use(A)
        bot_a = world.bots.bot(A)
        poller = asyncio.create_task(world.plugin._poller(world.api))
        try:
            await _until(lambda: bot_a.methods("getUpdates"))
            world.use(B)  # saved while it polls; it keeps polling with the token it started with
            seen = len(bot_a.methods("getUpdates"))
            bot_a.push(_owner(50, "polled by the first bot"))
            await _until(lambda: len(bot_a.methods("getUpdates")) >= seen + 2)
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller
        await world.outbound({"chat_id": 1, "text": "sent by the new token", "row_address": "row-b"})

    world.run(steps)
    assert world.ids() == [f"tg:{A_ID}:42:50"]
    refs = world.refs()["bots"]
    assert refs["messages"][f"{A_ID}:42:50"]["client_message_id"] == f"tg:{A_ID}:42:50"
    assert refs["messages"][f"{B_ID}:42:200"]["row_address"] == "row-b"
    assert refs["latest_owner"] == {f"{A_ID}:42": 50}


def test_a_token_saved_during_a_long_send_never_moves_its_parts(world):
    async def steps():
        world.use(A)
        world.bots.after_send = lambda: world.use(B)
        await world.outbound({"chat_id": 1, "text": "All words must reach the owner. " * 700,
                              "markdown": False, "row_address": "long-row"})

    world.run(steps)
    parts = world.bots.bot(A).methods("sendMessage")
    assert len(parts) > 2 and not world.bots.bot(B).methods("sendMessage")
    refs = world.refs()["bots"]["messages"]
    assert len(refs) == len(parts) and all(key.startswith(f"{A_ID}:42:") for key in refs)
    assert {record["row_address"] for record in refs.values()} == {"long-row"}


def test_the_notifier_scopes_its_mirror_to_the_token_it_sends_with(world):
    async def steps():
        world.use(B)
        outcome = await world.notifier._push_notification(world.api, 42, "done", event={
            "chat_id": 1, "task_id": "work", "kind": "task_summary", "row_address": "summary-row"})
        assert outcome == ("sent", None)

    world.run(steps)
    assert world.refs()["bots"]["messages"][f"{B_ID}:42:200"]["row_address"] == "summary-row"
    assert "messages" not in world.refs()


def test_the_panic_path_names_the_bot_that_polled_it(world):
    inject = AsyncMock()
    world.plugin._inject = inject
    world.plugin._POLL_GENERATION_STARTED = 1

    async def steps():
        world.use(B)
        client = world.plugin._telegram_client(world.api)
        await world.plugin._handle_owner_message(world.api, client, {
            "message": {"message_id": 7, "date": 100, "text": "/panic", "chat": {"id": 42}, "from": {"id": 42}},
        }, 1, "full_access", "en")

    world.run(steps)
    assert inject.await_args.args[1]["client_message_id"] == f"tg:{B_ID}:42:7"


# --- 8: what was stored stays readable and is never rewritten; a botless record anchors nothing --

_LEGACY_REFS = {"messages": {"42:7": {"direction": "in", "chat_id": 42, "kind": "message", "client_message_id": "tg:42:7"},
                             "42:8": {"row_address": "old-row", "chat_id": 1, "task_id": "", "kind": "message",
                                      "reply_to": "", "native_reply_id": 0}},
                "latest_owner": {"42": 7}}


def _seed_legacy_refs(world: _World) -> None:
    (world.state / "message_refs.json").write_text(json.dumps(_LEGACY_REFS), encoding="utf-8")


def test_stored_rows_and_references_stay_as_they_were_and_no_bot_inherits_them(world):
    from ouroboros.chat_references import resolve_reply
    from supervisor.message_bus import accepted_chat_message

    transport = {"kind": "telegram", "conversation_id": "42", "sender_label": "Telegram (Owner)"}

    async def steps():
        _seed_legacy_refs(world)
        status, _body = await world.plugin._host_post(world.api, "/chat/inject", {
            "client_message_id": "tg:42:7", "text": "stored before the upgrade", "chat_id": 42, "user_id": 42,
            "source": "telegram", "sender_label": "Telegram (Owner)", "transport": transport})
        assert status == 202
        await world.deliver(A, _owner(7, "the same number, other words"), _owner(9, "a reply to an old message", reply=8))
        await world.deliver(B, _owner(7, "stored before the upgrade"))

    world.run(steps)
    assert world.ids() == ["tg:42:7", f"tg:{A_ID}:42:7", f"tg:{A_ID}:42:9", f"tg:{B_ID}:42:7"]
    stored = accepted_chat_message(world.data, 42, "tg:42:7")
    assert stored["text"] == "stored before the upgrade"
    assert resolve_reply(world.data, world.address(0))["snippet"] == "stored before the upgrade"
    refs = world.refs()
    assert refs["messages"] == _LEGACY_REFS["messages"] and refs["latest_owner"] == _LEGACY_REFS["latest_owner"]
    assert not world.rows()[2].get("reply_to"), "a matching number proves no ownership: no reference"
    (note,) = world.bots.bot(A).methods("sendMessage")
    assert note["text"].startswith("Sent, but not linked to the earlier work")


def test_a_reply_with_nothing_old_at_its_number_or_known_as_its_own_gets_no_note(world):
    async def steps():
        _seed_legacy_refs(world)
        world.use(A)
        world.bots.bot(A).next_message = 8  # A's own message 8, at the number of the stored record
        await world.outbound({"chat_id": 1, "text": "A's own message", "row_address": "row-own"})
        await world.deliver(A, _owner(20, "a reply to A's own", reply=8), _owner(21, "a reply to nothing known", reply=1234))

    world.run(steps)
    assert world.texts() == ["a reply to A's own", "a reply to nothing known"]
    assert [form["text"] for form in world.bots.bot(A).methods("sendMessage")] == ["A's own message"]


def test_a_replayed_batch_across_the_upgrade_repeats_its_turn_once_and_only_once(world):
    """The disclosed boundary: an unconfirmed batch accepted under the old id format is one new
    row after the upgrade, since the old id may be another bot's. From then on the dedup holds."""
    async def steps():
        status, _body = await world.plugin._host_post(world.api, "/chat/inject", {
            "client_message_id": "tg:42:50", "text": "in flight at the upgrade", "chat_id": 42, "user_id": 42,
            "source": "telegram", "sender_label": "Telegram (Owner)", "transport": {}})
        assert status == 202
        await world.deliver(A, _owner(50, "in flight at the upgrade"))
        world.forget_cursor()
        await world.deliver(A, _owner(50, "in flight at the upgrade"))

    world.run(steps)
    assert world.ids() == ["tg:42:50", f"tg:{A_ID}:42:50"]
    assert world.dispatched() == 2


def test_botless_bubbles_are_not_edited_by_a_named_bot(world):
    async def steps():
        (world.state / "subagent_state.json").write_text(json.dumps({"42:child-1": 100}), encoding="utf-8")
        (world.state / "silent_state.json").write_text(json.dumps({"42": 100}), encoding="utf-8")
        _seed_legacy_refs(world)
        world.plugin.merge_settings(world.state, {"TELEGRAM_SILENT_MODE": "on", "TELEGRAM_MIRROR_PROGRESS": "on"})
        world.use(A)
        world.bots.bot(A).sent.add(100)
        await world.outbound({"subagent_event": "running", "subagent_task_id": "child-1", "chat_id": 1, "text": "card"})
        await world.outbound({"task_id": "t1", "is_progress": True, "role": "assistant", "text": "note",
                              "system_type": "model_narration", "chat_id": 1})

    world.run(steps)
    assert not world.bots.bot(A).methods("editMessageText")
    assert json.loads((world.state / "subagent_state.json").read_text(encoding="utf-8"))["42:child-1"] == 100


_CANARY = "fixture-secret-canary-value"


def test_a_secret_request_no_longer_remembered_has_no_protection_left(world):
    """The disclosed limit: the local hold exists only while a record says a card was there. The
    file keeps its last 50 cards; past that, nothing distinguishes the reply."""
    async def steps():
        await world.deliver(A, _owner(90, "a reply to an evicted card", reply=77))

    world.run(steps)
    assert world.texts() == ["a reply to an evicted card"]


def _seed_legacy_card(world: _World, token: str, *, secret: bool, message_id: int = 77) -> None:
    (world.state / "quiz_state.json").write_text(json.dumps({"quizzes": {token: {
        "task_id": "asker", "quiz_id": "q", "chat_id": 42, "message_id": message_id,
        "options": [] if secret else ["sqlite", "postgres"], "text": "Which database?",
        **({"secret": True} if secret else {})}}}), encoding="utf-8")


@pytest.mark.parametrize(("words", "token"), [
    pytest.param(_CANARY, A, id='words-sender-unknown-same-bot'),
    pytest.param("/status", B, id='command-sender-unknown-new-bot'),
])
def test_a_reply_to_a_stored_secret_request_never_reaches_the_host(world, token, words):
    """Whatever it holds, a command included, as a reply to a remembered secret request always was."""
    async def steps():
        _seed_legacy_card(world, "old-secret", secret=True)
        await world.deliver(token, _owner(90, words, reply=77))

    world.run(steps)
    assert world.rows() == [] and world.dispatched() == 0
    sent = world.bots.bot(token).methods("sendMessage")
    assert len(sent) == 1 and sent[0]["text"].startswith("Not sent: a secret goes only into the protected form")
    assert words not in sent[0]["text"]
    assert not world.bots.bot(token).methods("editMessageText")
    stored = json.loads((world.state / "quiz_state.json").read_text(encoding="utf-8"))
    assert stored["quizzes"]["old-secret"]["message_id"] == 77 and stored["quizzes"]["old-secret"]["secret"] is True


def test_this_bots_own_message_at_a_secret_requests_number_is_no_ambiguity(world):
    async def steps():
        _seed_legacy_card(world, "old-secret", secret=True, message_id=200)
        world.use(B)
        await world.outbound({"chat_id": 1, "text": "B's own message 200", "row_address": "own-row"})
        await world.deliver(B, _owner(90, "an ordinary reply", reply=200))

    world.run(steps)
    assert world.texts() == ["an ordinary reply"]


@pytest.mark.parametrize(("body", "token"), [
    pytest.param({"text": "postgres, I think"}, A, id='words-sender-unknown-same-bot'),
    pytest.param({"caption": "a caption", "photo": [{"file_id": "p"}]}, B, id='captioned-sender-unknown-new-bot'),
    pytest.param({"photo": [{"file_id": "p"}]}, A, id='bare-photo-sender-unknown-same-bot'),
    pytest.param({"document": {"file_id": "d", "file_name": "notes.txt", "mime_type": "text/plain", "file_size": 10}}, B, id='file-sender-unknown-new-bot'),
])
def test_a_reply_at_a_stored_cards_number_is_held_here_with_a_route_to_the_question(quiz_world, token, body):
    """Neither an answer (its sender is unknown) nor a new turn: nothing reaches the task or a model."""
    world = quiz_world
    message = _owner(90, "", reply=77)
    message["message"].pop("text")
    message["message"].update(body)

    async def steps():
        _seed_legacy_card(world, world.token, secret=False)
        await world.deliver(token, message)

    world.run(steps)
    assert world.rows() == [] and world.dispatched() == 0
    assert _quiz_block(world)["state"] == "open" and not _quiz_block(world).get("comment")
    (held,) = world.bots.bot(token).methods("sendMessage")
    assert held["text"].startswith("Not sent: this may answer an earlier question.")
    assert not world.bots.bot(token).methods("editMessageText")
    assert not world.bots.bot(token).methods("getFile"), "no attachment is fetched for a held reply"


def test_a_tap_on_a_stored_card_reaches_its_question_and_edits_nothing(quiz_world):
    world = quiz_world

    async def steps():
        _seed_legacy_card(world, world.token, secret=False)
        await world.deliver(A, _tap("cb-old", f"qz:{world.token}:1", 77))

    world.run(steps)
    assert _quiz_block(world)["answered_index"] == 1 and _quiz_block(world)["request_id"] == f"tg:{A_ID}:cb:cb-old"
    (toast,) = world.bots.bot(A).methods("answerCallbackQuery")
    assert toast["text"].startswith("✅")
    assert not world.bots.bot(A).methods("editMessageText")
    stored = json.loads((world.state / "quiz_state.json").read_text(encoding="utf-8"))["quizzes"][world.token]
    assert stored["message_id"] == 77 and "state" not in stored, "the stored card keeps its bytes"


def test_an_answer_stored_under_the_old_request_id_is_not_answered_again(quiz_world):
    from ouroboros.owner_quiz import record_answered

    world = quiz_world

    async def steps():
        record_answered(world.data, "asker", quiz_id="q", option_index=1, request_id="tg:1", answer_source="skill:telegram")
        make_quiz = world.plugin._make_quiz(world.api)
        world.use(A)
        await make_quiz(QUESTION)
        await world.deliver(A, _tap("cb-replay", f"qz:{world.token}:1", 100))

    world.run(steps)
    block = _quiz_block(world)
    assert block["request_id"] == "tg:1" and block["answered_index"] == 1
    (toast,) = world.bots.bot(A).methods("answerCallbackQuery")
    assert toast["text"].startswith("✅"), "the same answer is accepted, not reported as a conflict"


# --- 9: the poll cursor's rules for a file that names no bot are unchanged -----------------------

def test_a_cursor_naming_no_bot_keeps_its_offset_and_its_handled_batch(world):
    """Its handled batch stays too (disclosed): for a file from before cursors named their bot, an
    equal message of another bot is confirmed as a replay. The offset is that file's own rule."""
    async def steps():
        world.use(A)
        bot = world.bots.bot(A)
        bot.next_update = 500
        (world.state / "poll_offset.json").write_text(json.dumps({
            "offset": 500, "base": world.bots.origin, "handled": {world.bots.origin: ["m:42:60"]}}), encoding="utf-8")
        await world.deliver(A, _owner(60, "looks handled"), _owner(61, "fresh"))

    world.run(steps)
    assert world.bots.bot(A).methods("getUpdates")[0]["offset"] == "500"
    assert world.texts() == ["fresh"] and world.ids() == [f"tg:{A_ID}:42:61"]
    assert json.loads((world.state / "poll_offset.json").read_text(encoding="utf-8"))["bot"] == A_ID


def test_a_cursor_of_another_bot_starts_the_new_bot_from_its_oldest_update(world):
    async def steps():
        world.use(B)
        (world.state / "poll_offset.json").write_text(json.dumps({
            "offset": 5000, "base": world.bots.origin, "bot": A_ID,
            "handled": {world.bots.origin: ["m:42:60"]}}), encoding="utf-8")
        await world.deliver(B, _owner(60, "the new bot's own message 60"))

    world.run(steps)
    assert world.ids() == [f"tg:{B_ID}:42:60"]
    assert world.bots.bot(B).methods("getUpdates")[0]["offset"] == "0"


@pytest.mark.parametrize("token", [A, B], ids=["sender-unknown-same-bot", "sender-unknown-new-bot"])
def test_a_command_replying_at_a_stored_cards_number_keeps_its_dispatch(quiz_world, token, monkeypatch):
    """Commands keep their command-mode dispatch in replies, as before: only an answer or a turn is held."""
    world = quiz_world
    chosen = AsyncMock(return_value="")
    monkeypatch.setattr(world.plugin, "_choose_language", chosen)

    async def steps():
        _seed_legacy_card(world, world.token, secret=False)
        await world.deliver(token, _owner(90, "/status", reply=77), _owner(91, "/language Quenya", reply=77))

    world.run(steps)
    assert world.texts() == ["/status"] and world.dispatched() == 1
    assert not world.rows()[0].get("reply_to"), "no link is guessed from the old card's number"
    assert chosen.await_count == 1 and chosen.await_args.args[1] == "Quenya"
    assert _quiz_block(world)["state"] == "open"
    assert not world.bots.bot(token).methods("sendMessage"), "nothing held, nothing refused"


# --- 10: every named bot shares the file's bounds in one order; no bot count evicts --------------

_FIVE_BOTS = (A, B, "333333:SECRET-C", "444444:SECRET-D", "555555:SECRET-E")
_SECRET_QUESTION = {**QUESTION, "options": [], "secret_request": {
    "plugin_name": "telegram", "fields": [{"id": "botToken", "setting_key": "TELEGRAM_BOT_TOKEN"}]}}


@pytest.fixture
def maps(tmp_path):
    """The two bounded maps alone: their shipped readers and writers on an isolated state folder."""
    plugin, _api, _notifier = _load()

    def read(name: str) -> dict:
        return json.loads((tmp_path / name).read_text(encoding="utf-8"))

    return SimpleNamespace(api=_Api(tmp_path, {}), quiz=plugin.telegram_quiz,
                           state=sys.modules[plugin.__package__ + ".lib.telegram_state"],
                           refs=lambda: read("message_refs.json"), cards=lambda: read("quiz_state.json"))


def test_five_bots_with_one_secret_request_each_all_keep_its_local_hold(world):
    """Five cards are five of the newest 50, whichever bots sent them: after four more bots sent
    theirs, a reply to the first bot's secret request still stays here, as do the others'."""
    cards = {}

    async def steps():
        make_quiz = world.plugin._make_quiz(world.api)
        for index, token in enumerate(_FIVE_BOTS):
            world.use(token)
            await make_quiz({**_SECRET_QUESTION, "quiz_id": f"secret-{index}"})
            (cards[token],) = world.bots.bot(token).sent
        for token in _FIVE_BOTS:
            await world.deliver(token, _owner(90, _CANARY, reply=cards[token]))

    world.run(steps)
    assert [cards[token] for token in _FIVE_BOTS] == [100, 200, 300, 300, 300], "each bot numbers its own"
    assert world.rows() == [] and world.dispatched() == 0, "no reply reached the host"
    for token in _FIVE_BOTS:
        _card, held = world.bots.bot(token).methods("sendMessage")
        assert held["text"].startswith("Not sent: a secret goes only into the protected form"), token
        assert _CANARY not in held["text"]
        assert not world.bots.bot(token).methods("editMessageText")


def test_five_bots_with_one_reference_each_all_keep_it(maps):
    for bot in ("1", "2", "3", "4", "5"):
        maps.state._remember_message(maps.api, bot, 42, 1, {"row_address": f"row-{bot}"})
    maps.state._remember_message(maps.api, "2", 42, 2, {"row_address": "again"})
    assert [maps.state._message_ref(maps.api, bot, 42, 1)["row_address"] for bot in "12345"] == [
        "row-1", "row-2", "row-3", "row-4", "row-5"]
    assert list(maps.refs()["bots"]["messages"]) == ["1:42:1", "2:42:1", "3:42:1", "4:42:1", "5:42:1", "2:42:2"]


def test_cards_of_all_bots_share_the_newest_fifty_in_one_order(maps):
    """One question has a card at each bot, under one token. Interleaved, the oldest card of all
    goes first, an update makes its card the newest, and the bound counts every bot together. The
    cards saved before this file named its bots are neither counted nor touched."""
    quiz, (bot_a, bot_b, bot_c) = maps.quiz, ("111111", "222222", "333333")
    legacy = {"quizzes": {"old": {"task_id": "t", "quiz_id": "old", "chat_id": 42, "message_id": 7, "options": []}}}
    (maps.api.state_dir / "quiz_state.json").write_text(json.dumps(legacy), encoding="utf-8")

    def card(bot: str, index: int) -> None:
        quiz.remember_quiz(maps.api, bot, f"tok{index}", {
            "task_id": "t", "quiz_id": f"q{index}", "chat_id": 42, "options": [],
            "message_id": {bot_a: 100, bot_b: 200, bot_c: 300}[bot] + index})

    for index in range(17):  # 51 cards: tok0 at A, B and C, then tok1 at each, ...
        for bot in (bot_a, bot_b, bot_c):
            card(bot, index)
    assert quiz.quiz_for_token(maps.api, bot_a, "tok0") is None, "the oldest card of all went first"
    assert quiz.quiz_for_token(maps.api, bot_b, "tok0")["message_id"] == 200
    quiz.remember_state(maps.api, bot_b, "tok0", quiz.quiz_for_token(maps.api, bot_b, "tok0"), "closed")
    card(bot_a, 17)

    stored = maps.cards()
    assert len(stored["bots"]) == quiz._MAX_REMEMBERED == 50
    assert list(stored["bots"])[:2] == [f"{bot_a}:tok1", f"{bot_b}:tok1"]
    assert list(stored["bots"])[-2:] == [f"{bot_b}:tok0", f"{bot_a}:tok17"]
    assert quiz.quiz_for_token(maps.api, bot_c, "tok0") is None, "the next oldest: B's own was updated"
    assert quiz.quiz_for_token(maps.api, bot_b, "tok0")["state"] == "closed"
    assert quiz.quiz_for_message(maps.api, bot_a, 42, 101)["quiz_id"] == "q1"
    assert quiz.quiz_for_message(maps.api, bot_a, 42, 100) is None, "evicted: no other bot's card stands in"
    assert quiz.quiz_for_message(maps.api, bot_a, 42, 200) is None, "B's card, at a number A never sent"
    assert quiz.quiz_for_message(maps.api, bot_c, 42, 316)["quiz_id"] == "q16"
    assert json.dumps(stored["quizzes"]) == json.dumps(legacy["quizzes"]), "the old map keeps its bytes"


def test_references_of_all_bots_share_the_newest_thousand_in_one_order(maps):
    """Each bot numbers its own messages, so the same numbers recur under every bot. Interleaved,
    the oldest reference of all goes first, an update makes its reference the newest, and the
    bound counts every bot together. The map saved before it named its bots is neither counted
    nor touched, and no named bot reads it."""
    state, bots, limit = maps.state, ("111111", "222222", "333333"), maps.state._MAX_MESSAGE_REFS
    legacy = {"messages": {"42:1": {"row_address": "old-row", "chat_id": 1}}, "latest_owner": {"42": 1}}
    (maps.api.state_dir / "message_refs.json").write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    for mid in range(1, limit // len(bots) + 2):  # 1002 owner messages: A 1, B 1, C 1, A 2, ...
        for bot in bots:
            state._remember_message(maps.api, bot, 42, mid, {"row_address": f"{bot}-{mid}", "direction": "in"})
    assert state._message_ref(maps.api, bots[0], 42, 1) == {} and state._message_ref(maps.api, bots[1], 42, 1) == {}
    state._remember_message(maps.api, bots[2], 42, 1, {"row_address": "updated", "direction": "in"})
    state._remember_message(maps.api, bots[0], 42, 335, {"row_address": "newest"})

    stored = maps.refs()
    assert len(stored["bots"]["messages"]) == limit
    assert list(stored["bots"]["messages"])[:2] == [f"{bots[1]}:42:2", f"{bots[2]}:42:2"]
    assert list(stored["bots"]["messages"])[-2:] == [f"{bots[2]}:42:1", f"{bots[0]}:42:335"]
    assert state._message_ref(maps.api, bots[0], 42, 2) == {}, "the next oldest: C's 1 was updated"
    assert state._message_ref(maps.api, bots[1], 42, 2)["row_address"] == f"{bots[1]}-2"
    assert state._message_ref(maps.api, bots[2], 42, 1)["row_address"] == "updated"
    assert list(state._message_refs(maps.api, bots[1]))[:2] == ["42:2", "42:3"]
    assert stored["bots"]["latest_owner"] == {f"{bot}:42": 334 for bot in bots}
    assert json.dumps(stored["messages"]) == json.dumps(legacy["messages"]), "the old map keeps its bytes"
    assert stored["latest_owner"] == legacy["latest_owner"]
    assert list(state._message_refs(maps.api, "")) == ["42:1"], "a token naming no bot still reads only the old map"
