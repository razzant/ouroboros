"""The outbound journal (skills/telegram/lib/telegram_outbox.py) against a Bot API origin on loopback.

The shipped ``register()`` wiring, handlers, client, HTTPX and journal run; only the Bot API origin (an asyncio
server on 127.0.0.1 named by TELEGRAM_API_BASE) and the journal's clock are substituted. The origin refuses
connections (its listener closed), closes a connection after reading a request, answers 429 with retry_after,
a 5xx or a refusal, or hangs; one test kills a real child process during a request. Nothing leaves the
machine; no owner token or setting is read.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import sys
import time
import urllib.parse
from pathlib import Path

import pytest

from tests.test_telegram_proxy import _Api, _TOKEN, _load

_REPO = Path(__file__).resolve().parents[1]
_MESSAGE_METHODS = frozenset({"sendMessage", "sendPhoto", "sendVideo", "sendDocument", "sendAudio", "editMessageText"})
_BOT = _TOKEN.split(":", 1)[0]
_LONG = " ".join(f"word{index}" for index in range(1200))  # three Telegram messages


class _Origin:
    """A Bot API origin: each message request takes the next ``script`` action (``ok`` once it runs out):
    ``close`` (read the request, answer nothing), ``hang``, ``429:<seconds>`` or an HTTP status to refuse with.
    ``close_after`` stops listening after that many answered message requests."""

    def __init__(self, script=(), *, close_after=None):
        self.script, self.close_after = list(script), close_after
        self.requests: list[tuple[str, str, dict, bytes]] = []  # (method, path, form, body)
        self.answered, self.next_id, self.port = 0, 500, 0
        self.hanging = asyncio.Event()
        self._server, self._handlers = None, set()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def sent(self, method="sendMessage") -> list[dict]:
        return [form for name, _path, form, _body in self.requests if name == method]

    def methods(self) -> list[str]:
        return [name for name, *_rest in self.requests]

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_exc):
        self._server.close()
        for handler in list(self._handlers):  # a hung request ends with its origin
            handler.cancel()
        await asyncio.gather(*self._handlers, return_exceptions=True)

    async def _serve(self, reader, writer):
        self._handlers.add(asyncio.current_task())
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            first, *lines = head.decode("latin-1").split("\r\n")
            headers = dict(line.lower().split(": ", 1) for line in lines if ": " in line)
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            path = urllib.parse.urlsplit(first.split(" ")[1]).path
            method = path.rsplit("/", 1)[-1]
            form = {}
            if headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
                form = {key: values[-1] for key, values in urllib.parse.parse_qs(body.decode()).items()}
            self.requests.append((method, path, form, body))
            action = self.script.pop(0) if method in _MESSAGE_METHODS and self.script else "ok"
            if action == "close":
                return
            if action == "hang":
                self.hanging.set()
                await asyncio.Event().wait()
            status, payload = self._answer(method, form, action)
            writer.write(f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                         "Connection: close\r\n\r\n".encode() + payload)
            await writer.drain()
            if method in _MESSAGE_METHODS and status == 200:
                self.answered += 1
                if self.close_after is not None and self.answered >= self.close_after:
                    self._server.close()  # a later request finds nothing listening: it never left
        finally:
            writer.close()
            self._handlers.discard(asyncio.current_task())

    def _answer(self, method: str, form: dict, action: str) -> tuple[int, bytes]:
        if action.startswith("429:"):
            wait = int(action.split(":", 1)[1])
            body = {"ok": False, "error_code": 429, "description": f"Too Many Requests: retry after {wait}",
                    "parameters": {"retry_after": wait}}
            return 429, json.dumps(body).encode()
        if action != "ok":
            return int(action), json.dumps({"ok": False, "error_code": int(action), "description": "No"}).encode()
        if method == "editMessageText":
            result = {"message_id": int(form.get("message_id") or 0), "chat": {"id": 42}}
        elif method in _MESSAGE_METHODS:
            self.next_id += 1
            result = {"message_id": self.next_id, "chat": {"id": 42}}
        else:
            result = True
        return 200, json.dumps({"ok": True, "result": result}).encode()


class _Registrar(_Api):
    """The PluginAPI ``register()`` meets; it keeps the subscriptions and supervised tasks registered."""

    def __init__(self, state_dir, token=_TOKEN):
        super().__init__(state_dir, {"TELEGRAM_BOT_TOKEN": token})
        self.subscriptions, self.tasks = {}, {}

    def subscribe_event(self, topic, handler):
        self.subscriptions[topic] = handler

    def register_supervised_task(self, name, factory, **_policy):
        self.tasks[name] = factory

    def register_route(self, *_args, **_kwargs):
        pass

    def register_settings_section(self, *_args, **_kwargs):
        pass

    def on_unload(self, _callback):
        pass


def _not_delivered(bridge) -> str:
    """The Delivery issues list of Settings' Telegram form, as its route answers it."""
    return bridge.module("telegram_connection")._outcomes_words(bridge.outbox.summary(), time.time())


def _closed_base() -> str:
    """An origin with nothing listening: every request to it fails before it leaves."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


class _Bridge:
    """One load of the shipped skill on ``state``: its own modules, PluginAPI, journal and sender."""

    def __init__(self, state: Path, base: str, *, token=_TOKEN, **settings):
        state.mkdir(exist_ok=True)
        self.state, self.settings = state, settings
        self.serve(base)
        self.plugin, self.telegram_api, _notifier = _load()
        self.plugin._HONOR_ENV_PROXIES = False
        self.plugin.register_miniapp = lambda _api: False
        self.api = _Registrar(state, token)
        self.plugin.register(self.api)
        self.outbox = self.plugin.outbox_for(self.api)
        self.now = 1_800_000_000.0
        self.outbox.clock = lambda: self.now
        self.sender = None

    def module(self, name: str):
        return sys.modules[f"{self.plugin.__package__}.lib.{name}"]

    def serve(self, base: str) -> None:
        """Save ``base`` as the Bot API server; the next request goes there (as a Save in Settings does)."""
        (self.state / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_API_BASE": base,
                                                              **self.settings}), encoding="utf-8")

    async def publish(self, topic: str, event: dict) -> None:
        await self.api.subscriptions[topic](event)

    def start(self) -> None:
        self.sender = asyncio.ensure_future(self.api.tasks["outbox"]())

    async def stop(self) -> None:
        if self.sender is not None:
            self.sender.cancel()
            await asyncio.gather(self.sender, return_exceptions=True)

    def waiting(self) -> list[dict]:
        return [record for record in self.outbox._ops.values() if record["state"] in {"pending", "retry_wait", "sending"}]

    async def drain(self, timeout: float = 10.0) -> None:
        """Let the sender run every waiting operation, moving the clock to each one's due time."""
        if self.sender is None:
            self.start()
        await _until(lambda: self.outbox.summary()["loaded"])  # a new load reads what waited at the last stop
        deadline = time.monotonic() + timeout
        while self.waiting() or self.outbox._fifo().locked():  # until the last run has settled its files
            assert time.monotonic() < deadline, f"still waiting: {self.waiting()}"
            record, wait = self.outbox._next()
            if record is None and wait:  # the oldest one waits: its time comes
                self.now += wait
            self.outbox.wake()
            await asyncio.sleep(0.01)

    def records(self) -> list[dict]:
        return sorted(self.outbox._ops.values(), key=lambda record: record["seq"])

    def files(self) -> list[str]:
        root = self.state / "outbox"
        return sorted(path.name for path in root.iterdir()) if root.exists() else []


async def _until(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


def _run(scenario):
    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


def test_a_reply_that_never_left_waits_and_arrives_once_when_telegram_answers(tmp_path):
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        await bridge.publish("chat.outbound", {"text": "nothing mirrored", "is_progress": True, "transport": {}})
        assert bridge.files() == []  # a message the bridge does not send is never journaled
        await bridge.publish("chat.outbound", {"text": "The answer.", "transport": {}, "row_address": "row:a"})
        [record] = bridge.records()
        assert (record["state"], record["parts"][0]["state"], record["reason"]) == ("retry_wait", "retry_wait", "ConnectError")
        fields = bridge.plugin._connection_fields(bridge.api)
        assert fields["delivery_status"] == "1 waiting to be sent (can't reach Telegram); next try in 5 s"
        assert _not_delivered(bridge) == "No recent delivery issues."  # what waits is no issue, nor arrived yet
        assert "The answer" not in json.dumps({key: value for key, value in fields.items() if key != "bridge"})
        assert any("Telegram delivery waits (ConnectError)" in message for _level, message in bridge.api.logs)
        async with _Origin() as origin:
            bridge.serve(origin.base)
            await bridge.drain()
        assert [form["text"] for form in origin.sent()] == ["The answer."]
        assert bridge.records() == [] and bridge.files() == []  # delivered: nothing is kept
        assert bridge.plugin._connection_fields(bridge.api)["delivery_status"] == "Nothing waits to be sent"
        assert ("info", "Telegram delivery resumed after ConnectError.") in bridge.api.logs
        await bridge.stop()
    _run(scenario)


@pytest.mark.parametrize("answer", ["close", "502"])
def test_an_answer_that_may_follow_delivery_is_unknown_and_never_sent_again(tmp_path, answer):
    async def scenario():
        async with _Origin([answer]) as origin:
            bridge = _Bridge(tmp_path / "state", origin.base)
            await bridge.publish("chat.outbound", {"text": "Maybe delivered.", "transport": {}})
            await bridge.publish("chat.outbound", {"text": "Next one.", "transport": {}})
            bridge.now += 7 * 24 * 3600
            bridge.start()
            await asyncio.sleep(0.2)  # the sender is free to act: nothing of the first is due
        assert [form["text"] for form in origin.sent()] == ["Maybe delivered.", "Next one."]
        [record] = bridge.records()
        assert (record["state"], record["parts"][0]["state"], record["held"]) == ("unknown", "unknown", True)
        # No row in the app holds these words: their only copy stays, never sent again; the record has no words.
        assert bridge.files() == [f"{record['id']}.event.json", f"{record['id']}.json"]
        assert "Maybe delivered" not in (bridge.state / "outbox" / f"{record['id']}.json").read_text(encoding="utf-8")
        assert bridge.plugin._connection_fields(bridge.api)["delivery_status"] == (
            "1 recent delivery issue (see Delivery issues)")  # never "did not arrive": it may have
        assert _not_delivered(bridge).endswith(" · Reply\nMay not have arrived (no clear answer from Telegram; not sent "
                                               "again). Only copy kept on this computer (Kept copies).")
        await bridge.stop()
    _run(scenario)


def test_flood_control_waits_exactly_its_retry_after(tmp_path):
    async def scenario():
        async with _Origin(["429:7"]) as origin:
            bridge = _Bridge(tmp_path / "state", origin.base)
            await bridge.publish("chat.outbound", {"text": "Slow down.", "transport": {}})
            assert bridge.outbox.summary()["cause"] == "HTTP 429"
            assert bridge.outbox.summary()["retry_in_sec"] == 7
            bridge.now += 6
            assert bridge.outbox._next()[0] is None  # not before Telegram's own wait
            await bridge.drain()
        assert [form["text"] for form in origin.sent()] == ["Slow down.", "Slow down."]
        assert origin.next_id == 501 and bridge.records() == []
        await bridge.stop()
    _run(scenario)


def test_a_long_reply_resumes_at_its_first_unsent_part(tmp_path):
    async def scenario():
        async with _Origin(close_after=1) as first, _Origin() as second:
            bridge = _Bridge(tmp_path / "state", first.base)
            await bridge.publish("chat.outbound", {"text": _LONG, "transport": {}})
            [record] = bridge.records()
            assert [part["state"] for part in record["parts"]] == ["confirmed", "retry_wait"]
            bridge.serve(second.base)
            await bridge.drain()
        chunks = bridge.telegram_api.message_chunks(_LONG, "HTML")
        assert len(chunks) == 3
        assert [form["text"] for form in first.sent()] == chunks[:1]  # the delivered part is never resent
        assert [form["text"] for form in second.sent()] == chunks[1:]
        await bridge.stop()
    _run(scenario)


def test_media_links_and_a_quiz_card_wait_in_order_and_arrive_once(tmp_path):
    quiz = {"quiz_id": "q1", "task_id": "task-1", "question": "Which db?", "transport": {},
            "options": [{"label": "sqlite"}, {"label": "postgres"}]}

    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        await bridge.publish("chat.photo", {"image_base64": base64.b64encode(b"PNG-BYTES").decode(),
                                            "caption": "chart", "transport": {}})
        await bridge.publish("chat.document", {"file_base64": base64.b64encode(b"DOC-BYTES").decode(),
                                               "filename": "notes.txt", "transport": {}})
        await bridge.publish("chat.links", {"title": "Links", "transport": {},
                                            "actions": [{"label": "Docs", "url": "https://example.org/docs"}]})
        await bridge.publish("chat.quiz", quiz)
        # The host's answer arrives while the card itself still waits: kept for the card (no outbox record of its
        # own), it is applied once the card exists.
        await bridge.publish("chat.quiz_state", {"quiz_id": "q1", "task_id": "task-1", "state": "answered",
                                                 "answered_index": 1})
        assert [record["state"] for record in bridge.records()] == ["retry_wait", "pending", "pending", "pending"]
        async with _Origin() as origin:
            bridge.serve(origin.base)
            await bridge.drain()
        assert origin.methods() == ["sendPhoto", "sendDocument", "sendMessage", "sendMessage", "editMessageText"]
        assert b"PNG-BYTES" in origin.requests[0][3] and b"DOC-BYTES" in origin.requests[1][3]
        links, card = origin.sent()
        assert "https://example.org/docs" in links["reply_markup"] and "Which db?" in card["text"]
        quiz_module = bridge.module("telegram_quiz")
        remembered = quiz_module.quiz_for_token(bridge.api, _BOT, quiz_module.mint_token("task-1", "q1"))
        assert remembered["message_id"] == 504 and origin.sent("editMessageText")[0]["message_id"] == "504"
        assert bridge.records() == [] and bridge.files() == []
        await bridge.stop()
    _run(scenario)


@pytest.mark.parametrize("fact,tail", [
    ({"state": "answered", "answered_index": 1}, "\nAnswered: 2. postgres"),
    ({"state": "closed", "closed_reason": "Plan changed."}, "\nQuestion closed.\nPlan changed."),
], ids=["answered", "closed"])
@pytest.mark.parametrize("restart", [False, True], ids=["same-load", "new-load"])
def test_a_settled_edit_that_waits_is_retried_when_its_card_is_created_again(tmp_path, fact, tail, restart):
    """The host's fact reaches the bridge before its card exists (kept for the card), the card is sent and its
    settling edit meets flood control. The retry runs the card's handler again on its saved event, which
    creates the card again: that never takes back the remembered state, so the same edit goes out, once, even
    after a restart took what the first load kept in memory."""
    quiz = {"quiz_id": "q1", "task_id": "task-1", "question": "Which db?", "transport": {},
            "options": [{"label": "sqlite"}, {"label": "postgres"}]}

    async def scenario():
        async with _Origin(["ok", "429:3"]) as origin:
            bridge = _Bridge(tmp_path / "state", origin.base)
            await bridge.publish("chat.quiz_state", {"quiz_id": "q1", "task_id": "task-1", **fact})
            assert origin.requests == [] and bridge.files() == []  # no card yet: no outbox record; kept for the card
            await bridge.publish("chat.quiz", quiz)
            [record] = bridge.records()
            assert [(part["method"], part["state"]) for part in record["parts"]] == [
                ("sendMessage", "confirmed"), ("editMessageText", "retry_wait")]
            if restart:  # new modules: what the first load kept in memory is gone
                bridge = _Bridge(tmp_path / "state", origin.base)
            await bridge.drain()
        assert origin.methods() == ["sendMessage", "editMessageText", "editMessageText"]  # the card itself once
        refused, retried = origin.sent("editMessageText")
        assert retried == refused and retried["text"].endswith(tail)
        assert json.loads(retried["reply_markup"]) == {"inline_keyboard": []}  # no buttons left to tap
        quiz_module = bridge.module("telegram_quiz")
        remembered = quiz_module.quiz_for_token(bridge.api, _BOT, quiz_module.mint_token("task-1", "q1"))
        assert remembered["state"] == fact["state"] and remembered.get("answered_index") == fact.get("answered_index")
        assert bridge.records() == [] and bridge.files() == []  # delivered whole
        await bridge.stop()
    _run(scenario)


def test_a_waiting_reply_keeps_its_native_reply_source_and_label(tmp_path):
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        state = bridge.module("telegram_state")
        for owner_message in (10, 11):  # the owner's earlier and latest messages in this chat
            state._remember_message(bridge.api, _BOT, 42, owner_message, {
                "direction": "in", "chat_id": 1, "kind": "message", "row_address": f"row:owner-{owner_message}"})
        answer = {"text": "Earlier answer.", "transport": {}, "chat_id": 1, "row_address": "row:answer",
                  "task_id": "task-a", "reply_to": {"role": "user", "address": "row:owner-10", "chat_id": 1},
                  "message_origin": {"label": "Ремонт доставки"}}
        note = {"text": "A note.", "transport": {}, "chat_id": 1, "row_address": "row:note", "task_id": "task-b"}
        await bridge.publish("chat.outbound", answer)
        await bridge.publish("chat.outbound", note)
        first, second = bridge.records()
        saved = json.loads((bridge.state / "outbox" / f"{first['id']}.event.json").read_text(encoding="utf-8"))
        assert saved == answer  # the event as published, its label and reply facts included
        assert first["source"] == {"task_id": "task-a", "row_address": "row:answer", "reply_to": "row:owner-10"}
        async with _Origin() as origin:
            bridge.serve(origin.base)
            await bridge.drain()
        sent_answer, sent_note = origin.sent()
        assert sent_answer["reply_to_message_id"] == "10" and "reply_to_message_id" not in sent_note
        ref = state._message_ref(bridge.api, _BOT, 42, 501)
        assert (ref["row_address"], ref["task_id"], ref["reply_to"], ref["native_reply_id"]) == (
            "row:answer", "task-a", "row:owner-10", 10)
        assert state._reply_reference(bridge.api, _BOT, 42, 502) == "row:note"
        await bridge.stop()
    _run(scenario)


def test_a_later_message_never_overtakes_one_that_waits(tmp_path):
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        await bridge.publish("chat.outbound", {"text": "first", "transport": {}})
        async with _Origin() as origin:
            bridge.serve(origin.base)  # Telegram is back, but the first message is not due yet
            await bridge.publish("chat.outbound", {"text": "second", "transport": {}})
            assert origin.requests == [] and [record["state"] for record in bridge.records()] == ["retry_wait", "pending"]
            await bridge.drain()
        assert [form["text"] for form in origin.sent()] == ["first", "second"]
        await bridge.stop()
    _run(scenario)


def test_an_ended_sender_never_holds_later_messages_behind_what_waits(tmp_path):
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        await bridge.publish("chat.outbound", {"text": "first", "transport": {}})
        bridge.start()
        await asyncio.sleep(0.1)  # the sender runs and sleeps until the first message is due
        await bridge.stop()  # its restarts spent: nothing runs what waits until the next load
        assert bridge.outbox._sender_down
        async with _Origin() as origin:
            bridge.serve(origin.base)
            await bridge.publish("chat.outbound", {"text": "second", "transport": {}})
        assert [form["text"] for form in origin.sent()] == ["second"]
        assert [record["state"] for record in bridge.records()] == ["retry_wait"]  # kept for the next load
    _run(scenario)


def test_a_stop_that_lands_with_a_wake_ends_the_sender(tmp_path):
    """Unload or restart cancels the sender; a wake in the same turn of the loop never keeps it alive."""
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        await bridge.publish("chat.outbound", {"text": "waits for its retry", "transport": {}})
        bridge.start()
        await _until(lambda: bridge.outbox.summary()["loaded"])
        await asyncio.sleep(0.05)  # it sleeps until the retry is due, or until woken
        bridge.outbox.wake()
        bridge.sender.cancel()
        await asyncio.wait_for(asyncio.gather(bridge.sender, return_exceptions=True), timeout=5)
        assert bridge.outbox._sender_down
    _run(scenario)


def test_a_new_load_delivers_what_waited_at_the_last_stop(tmp_path):
    async def scenario():
        before = _Bridge(tmp_path / "state", _closed_base())
        await before.publish("chat.outbound", {"text": "Survives the restart.", "transport": {}})
        async with _Origin() as origin:
            after = _Bridge(tmp_path / "state", origin.base)  # new modules, PluginAPI and journal on the same state
            await after.drain()
        assert [form["text"] for form in origin.sent()] == ["Survives the restart."]
        assert after.files() == []
        await after.stop()
    _run(scenario)


_CHILD = """
import asyncio, sys
sys.path.insert(0, sys.argv[1])
from tests.test_telegram_outbox import _Bridge, _LONG
from pathlib import Path

async def main():
    bridge = _Bridge(Path(sys.argv[2]), sys.argv[3])
    await bridge.publish("chat.outbound", {"text": _LONG, "transport": {}})

asyncio.run(main())
"""


_CHILD_BUSY = """
import asyncio, sys
sys.path.insert(0, sys.argv[1])
from tests.test_telegram_outbox import _Bridge, _until
from pathlib import Path

async def main():
    bridge = _Bridge(Path(sys.argv[2]), sys.argv[3])
    first = asyncio.ensure_future(bridge.publish("chat.outbound", {"text": "first, hangs", "transport": {}}))
    await _until(lambda: bridge.outbox._fifo().locked() and bridge.records())
    await bridge.publish("chat.outbound", {"text": "nothing mirrored", "is_progress": True, "transport": {}})
    await bridge.publish("chat.outbound", {"text": "second, waits", "transport": {}})
    print("ACCEPTED", flush=True)
    await first

asyncio.run(main())
"""


def test_an_event_waiting_behind_a_hung_request_is_saved_before_it_waits_and_survives_a_kill(tmp_path):
    """Process boundary: the second event is on disk before it waits for the first one's hung request, so a
    kill loses neither; the filtered event in between is never saved, so it can never become a send."""
    async def scenario():
        state = tmp_path / "state"
        async with _Origin(["hang"]) as origin:
            child = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _CHILD_BUSY, str(_REPO), str(state), origin.base, cwd=str(_REPO),
                stdout=asyncio.subprocess.PIPE, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            try:
                await asyncio.wait_for(origin.hanging.wait(), timeout=30)
                assert (await asyncio.wait_for(child.stdout.readline(), timeout=30)).strip() == b"ACCEPTED"
            finally:
                child.kill()
                await child.wait()
            saved = sorted((json.loads(path.read_text(encoding="utf-8")) for path in (state / "outbox").glob("*.json")
                            if not path.name.endswith(".event.json")), key=lambda record: record["seq"])
            assert [(record["state"], [part["state"] for part in record["parts"]]) for record in saved] == [
                ("sending", ["sending"]), ("pending", [])]
            bridge = _Bridge(state, origin.base)
            await bridge.drain()
        assert [form["text"] for form in origin.sent()] == ["first, hangs", "second, waits"]
        [first] = bridge.records()
        assert [(part["state"], part.get("reason")) for part in first["parts"]] == [("unknown", "interrupted")]
        await bridge.stop()
    _run(scenario)


def test_a_part_in_flight_when_the_process_is_killed_is_unknown_and_the_rest_arrive(tmp_path):
    """Process boundary: SIGKILL (TerminateProcess) during the second part's request leaves the saved
    ``sending`` record on disk; the next load never sends that part again and delivers the third."""
    async def scenario():
        state = tmp_path / "state"
        async with _Origin(["ok", "hang"]) as origin:
            child = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _CHILD, str(_REPO), str(state), origin.base, cwd=str(_REPO),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            try:
                await asyncio.wait_for(origin.hanging.wait(), timeout=30)
            finally:
                child.kill()
                await child.wait()
            [on_disk] = [json.loads(path.read_text(encoding="utf-8")) for path in (state / "outbox").glob("*.json")
                         if not path.name.endswith(".event.json")]
            assert [part["state"] for part in on_disk["parts"]] == ["confirmed", "sending"]
            bridge = _Bridge(state, origin.base)
            bridge.start()
            await _until(lambda: len(origin.sent()) == 3)
            await _until(lambda: not bridge.waiting())
        chunks = bridge.telegram_api.message_chunks(_LONG, "HTML")
        assert [form["text"] for form in origin.sent()] == chunks  # each part reached Telegram exactly once
        [record] = bridge.records()
        assert [(part["state"], part.get("reason")) for part in record["parts"]] == [
            ("confirmed", None), ("unknown", "interrupted"), ("confirmed", None)]
        assert record["state"] == "unknown"
        await bridge.stop()
    _run(scenario)


def _failing_journal(bridge, *, after: int = 0, cleanup: bool = True) -> None:
    """From the journal write after the first ``after`` on, the disk refuses; so does every removal unless ``cleanup``."""
    module = bridge.module("telegram_outbox")
    real_write, real_unlink, writes = module._write_private, module._unlink, []

    def write(path, content):
        writes.append(path.name)
        if len(writes) > after:
            raise PermissionError("read-only state")
        real_write(path, content)

    def unlink(path):
        if not cleanup:
            raise PermissionError("read-only state")
        real_unlink(path)

    module._write_private, module._unlink = write, unlink


def test_a_message_the_outbox_cannot_save_is_not_sent_and_the_status_says_so(tmp_path):
    async def scenario():
        async with _Origin() as origin:
            bridge = _Bridge(tmp_path / "state", origin.base)
            _failing_journal(bridge)
            await bridge.publish("chat.outbound", {"text": "Unsaved.", "transport": {}})
        assert origin.requests == [] and bridge.records() == []  # fails closed: nothing without a saved record
        assert bridge.plugin._connection_fields(bridge.api)["delivery_status"] == (
            "Degraded: the outbox can't save (PermissionError), so nothing new is sent. "
            "1 not sent because the outbox couldn't hold them. 1 recent delivery issue (see Delivery issues)")
        # The disk took no record of it either: it is named for this load only, and no copy is claimed.
        assert _not_delivered(bridge).endswith(" · Reply\nNot sent: the outbox couldn't save it. No copy was kept.")
        assert any("was not sent: the outbox could not hold it" in message for _level, message in bridge.api.logs)
    _run(scenario)


def test_a_saved_message_never_goes_out_while_its_sending_state_cannot_be_saved(tmp_path):
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        await bridge.publish("chat.outbound", {"text": "Saved once.", "transport": {}})
        assert [record["state"] for record in bridge.records()] == ["retry_wait"]
        async with _Origin() as origin:
            bridge.serve(origin.base)
            _failing_journal(bridge, cleanup=False)  # and nothing written before can be removed
            bridge.start()
            bridge.now += 5
            bridge.outbox.wake()
            await _until(lambda: bridge.outbox.summary()["fault"] == "PermissionError")
            await asyncio.sleep(0.2)
            assert origin.requests == []  # its retry could not save "sending" first: nothing left
            assert bridge.plugin._connection_fields(bridge.api)["delivery_status"].startswith(
                "Degraded: the outbox can't save (PermissionError)")
            await bridge.stop()
            after = _Bridge(tmp_path / "state", origin.base)  # a new load, with a disk that works again
            await after.drain()
        assert [form["text"] for form in origin.sent()] == ["Saved once."]
        await after.stop()
    _run(scenario)


def test_a_delivered_part_whose_outcome_cannot_be_saved_is_never_sent_again(tmp_path):
    async def scenario():
        async with _Origin() as origin:
            bridge = _Bridge(tmp_path / "state", origin.base)
            _failing_journal(bridge, after=3, cleanup=False)  # event, custody and sending; no outcome
            await bridge.publish("chat.outbound", {"text": "Delivered once.", "transport": {}})
            assert [form["text"] for form in origin.sent()] == ["Delivered once."]
            assert bridge.plugin._connection_fields(bridge.api)["delivery_status"].startswith("Degraded")
            after = _Bridge(tmp_path / "state", origin.base)
            await after.drain()
        assert [form["text"] for form in origin.sent()] == ["Delivered once."]
        [record] = after.records()
        assert [(part["state"], part.get("reason")) for part in record["parts"]] == [("unknown", "interrupted")]
        await after.stop()
    _run(scenario)


def test_an_edit_whose_answer_was_lost_is_never_followed_by_another_request(tmp_path):
    async def scenario():
        async with _Origin(["ok", "close"]) as origin:
            bridge = _Bridge(tmp_path / "state", origin.base, TELEGRAM_SILENT_MODE="on", TELEGRAM_MIRROR_PROGRESS="on")
            progress = {"is_progress": True, "task_id": "t1", "transport": {}}
            await bridge.publish("chat.outbound", {"text": "step one", **progress})
            await bridge.publish("chat.outbound", {"text": "step two", **progress})  # replaces step one by an edit
        assert origin.methods() == ["sendMessage", "editMessageText"]  # no plain-text edit, no fresh message
        assert origin.sent("editMessageText")[0]["message_id"] == "501"
        [record] = bridge.records()
        assert [(part["method"], part["state"]) for part in record["parts"]] == [("editMessageText", "unknown")]
        assert _not_delivered(bridge).endswith(
            " · Progress note\nMay not have arrived (no clear answer from Telegram; not sent again).")
        assert not record.get("held") and f"{record['id']}.event.json" not in bridge.files()  # telemetry: not kept
    _run(scenario)


def test_refused_expired_and_changed_operations_are_never_sent(tmp_path):
    async def scenario():
        async with _Origin(["403"]) as origin:
            refused = _Bridge(tmp_path / "refused", origin.base)
            await refused.publish("chat.outbound", {"text": "Blocked.", "transport": {}})
            assert [(record["state"], record["parts"][0]["status"]) for record in refused.records()] == [("refused", 403)]
            assert ("error", "Telegram outbound error: Telegram API sendMessage returned HTTP 403.") in refused.api.logs

        expired = _Bridge(tmp_path / "expired", _closed_base())
        await expired.publish("chat.outbound", {"text": "Too late.", "transport": {}})
        expired.now += 24 * 3600 + 1
        async with _Origin() as late:
            expired.serve(late.base)
            await expired.drain()
        assert late.requests == [] and [record["reason"] for record in expired.records()] == ["expired"]

        async with _Origin(close_after=1) as first, _Origin() as second:
            changed = _Bridge(tmp_path / "changed", first.base)
            await changed.publish("chat.outbound", {"text": _LONG, "transport": {}})
            [record] = changed.records()
            saved = changed.state / "outbox" / f"{record['id']}.event.json"
            saved.write_text(json.dumps({"text": "other words " + _LONG, "transport": {}}), encoding="utf-8")
            changed.serve(second.base)
            await changed.drain()
        assert second.requests == []  # the saved part no longer matches: nothing else is sent
        assert [(record["state"], record["reason"]) for record in changed.records()] == [("failed", "changed")]
        for bridge in (refused, expired, changed):
            await bridge.stop()
    _run(scenario)


def test_what_no_bot_took_yet_goes_out_whole_through_the_token_in_force(tmp_path):
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        await bridge.publish("chat.outbound", {"text": _LONG, "transport": {}})
        bridge.api.granted["TELEGRAM_BOT_TOKEN"] = "654321:OTHER-BOT-TOKEN"
        assert bridge.plugin.binding_for(bridge.api).read_now(bridge.api)
        async with _Origin(close_after=1) as first, _Origin() as second:
            bridge.serve(first.base)  # the new bot's first part arrives, then its connection fails again
            bridge.start()
            bridge.now += 5
            bridge.outbox.wake()
            await _until(lambda: len(first.sent()) == 1 and bridge.waiting() and not bridge.outbox._fifo().locked())
            bridge.serve(second.base)
            await bridge.drain()
        chunks = bridge.telegram_api.message_chunks(_LONG, "HTML")
        assert [form["text"] for form in first.sent() + second.sent()] == chunks
        assert {path.split("/")[1] for _method, path, _form, _body in first.requests + second.requests} == {
            "bot654321:OTHER-BOT-TOKEN"}
        assert bridge.records() == []  # delivered whole: the old bot never took a part of it
        await bridge.stop()
    _run(scenario)


def test_a_replaced_bot_never_continues_what_the_old_one_started(tmp_path):
    async def scenario():
        async with _Origin(close_after=1) as first, _Origin() as second:
            bridge = _Bridge(tmp_path / "state", first.base)
            await bridge.publish("chat.outbound", {"text": _LONG, "transport": {}})
            bridge.api.granted["TELEGRAM_BOT_TOKEN"] = "654321:OTHER-BOT-TOKEN"
            assert bridge.plugin.binding_for(bridge.api).read_now(bridge.api)  # the new token is accepted
            bridge.serve(second.base)
            await bridge.drain()
        assert len(first.sent()) == 1 and second.requests == []
        [record] = bridge.records()
        assert (record["state"], record["reason"], record["bot"]) == ("failed", "token_changed", _BOT)
        await bridge.stop()
    _run(scenario)


@pytest.mark.parametrize("limit", ["count", "bytes"])
def test_concurrent_publications_reserve_capacity_before_writing(tmp_path, monkeypatch, limit):
    async def scenario():
        bridge = _Bridge(tmp_path / "state", _closed_base())
        module = bridge.module("telegram_outbox")
        if limit == "bytes":
            monkeypatch.setattr(module, "MAX_ACTIVE_BYTES", 256)
        event = {"text": "Queued.", "transport": {}}
        size = len(json.dumps(event, ensure_ascii=False).encode())
        expected = 500 if limit == "count" else 256 // size
        await bridge.publish("chat.outbound", event)  # a due head holds the queue
        await asyncio.gather(*(bridge.publish("chat.outbound", event) for _ in range(520)))
        assert len(bridge.waiting()) == expected
        assert sum(r["payload_bytes"] for r in bridge.waiting()) == expected * size
        assert bridge.outbox.summary()["not_sent"] == 521 - expected
        assert bridge.outbox._capacity == {} and bridge.outbox._reserved == set()
        after = _Bridge(tmp_path / "state", _closed_base())
        after.outbox._load()
        assert len(after.waiting()) == expected  # the bound is durable too
    _run(scenario)


@pytest.mark.parametrize("fail_write", [False, True], ids=["saved", "write-failed"])
def test_cancelled_admission_keeps_its_reservation_until_the_writer_finishes(tmp_path, monkeypatch, fail_write):
    import threading

    async def scenario():
        async with _Origin() as origin:
            bridge = _Bridge(tmp_path / "state", origin.base)
            module = bridge.module("telegram_outbox")
            monkeypatch.setattr(module, "MAX_ACTIVE", 1)
            bridge.start()
            await _until(lambda: bridge.outbox.summary()["loaded"])
            entered, release = threading.Event(), threading.Event()
            original = bridge.outbox._write_payload

            def write(record, payload):
                entered.set()
                assert release.wait(10)
                original(record, payload)
                if fail_write:
                    raise OSError("synthetic write failure after payload")

            monkeypatch.setattr(bridge.outbox, "_write_payload", write)
            event = {"image_base64": base64.b64encode(b"image").decode(), "transport": {}}
            sending = asyncio.create_task(bridge.publish("chat.photo", event))
            await _until(entered.is_set)
            sending.cancel()
            try:
                await bridge.publish("chat.photo", event)
                assert len(bridge.outbox._capacity) == 1 and not sending.done()
                assert bridge.outbox.summary()["not_sent"] == 1
                assert origin.requests == []
            finally:
                release.set()
                result = await asyncio.gather(sending, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError)
            assert bridge.outbox._capacity == {}
            if not fail_write:
                await _until(lambda: bridge.outbox.summary()["delivered"] == 1)
            await bridge.stop()
            after = _Bridge(tmp_path / "state", origin.base)
            if fail_write:
                # No saved bytes remain; each refused photo (the failed write, the one refused while it was
                # reserved) is named by its metadata only, never sent.
                assert not [name for name in bridge.files() if name.endswith(".event.json")]
                assert [(r["state"], r["reason"], r.get("held")) for r in bridge.records()] == [
                    ("failed", "journal OSError", None), ("failed", "full", None)]
                await after.publish("chat.photo", event)  # the failed reservation has been released
            await after.drain()
            assert origin.methods() == ["sendPhoto"]
            await after.stop()
    _run(scenario)


@pytest.mark.parametrize("topic,event,methods", [
    ("chat.outbound", {"text": "**Hello**"}, ["sendMessage", "sendMessage"]),
    ("chat.links", {"title": _LONG, "actions": [{"label": "Docs", "url": "https://example.org"}]},
     ["sendMessage"] * 4),
    ("chat.document", {"file_base64": base64.b64encode(b"audio").decode(), "filename": "voice.mp3"},
     ["sendAudio", "sendDocument"]),
])
def test_successful_fallback_replaces_its_refusal_after_reload(tmp_path, topic, event, methods):
    async def scenario():
        async with _Origin(["400", "429:2"]) as origin:
            bridge = _Bridge(tmp_path / "state", origin.base)
            await bridge.publish(topic, {**event, "transport": {}})
            [record] = bridge.records()
            assert record["parts"][1]["replaces"] == 0
            if topic == "chat.document":  # saved by 1.4.13, before fallback links existed
                path = bridge.outbox._record_path(record)
                legacy = json.loads(path.read_text())
                legacy["parts"][1].pop("replaces")
                path.write_text(json.dumps(legacy))
            after = _Bridge(tmp_path / "state", origin.base)
            await after.drain()
            assert origin.methods() == methods + methods[-1:]
            assert after.outbox.summary()["delivered"] == 1 and after.records() == []
            assert after.plugin._connection_fields(after.api)["delivery_status"] == "Nothing waits to be sent"
            await after.stop()
    _run(scenario)


@pytest.mark.parametrize("script,state", [(["400", "close"], "unknown"), (["400", "ok", "403"], "refused")])
def test_fallback_does_not_hide_an_unknown_replacement_or_another_lost_part(tmp_path, script, state):
    async def scenario():
        async with _Origin(script) as origin:
            bridge = _Bridge(tmp_path / "state", origin.base)
            await bridge.publish("chat.outbound", {"text": _LONG, "transport": {}})
            [record] = bridge.records()
            assert record["state"] == state and bridge.outbox.summary()["delivered"] == 0
            before = list(origin.requests)
            after = _Bridge(tmp_path / "state", origin.base)
            await after.drain()
            assert origin.requests == before  # no retry of an unknown or refused part
            assert after.records()[0]["state"] == state
            await after.stop()
    _run(scenario)


def test_pre_request_failure_on_an_empty_queue_is_visible_but_filtered_events_are_not(tmp_path):
    async def scenario():
        async with _Origin() as origin:
            bridge = _Bridge(tmp_path / "state", origin.base, TELEGRAM_PROXY="not-a-proxy")
            await bridge.publish("chat.outbound", {"text": "Cannot start.", "transport": {}})
            assert origin.requests == []
            assert bridge.outbox.summary()["failed"] == 1
            assert _not_delivered(bridge).endswith(
                " · Reply\nStopped before sending. Only copy kept on this computer (Kept copies).")
            after = _Bridge(tmp_path / "state", origin.base, TELEGRAM_MIRROR_MODE="telegram_only")
            after.outbox._load()
            await after.publish("chat.outbound", {"text": "Filtered web event.", "transport": {"source": "web"}})
            assert after.outbox.summary()["failed"] == 1 and len(after.records()) == 1
            assert origin.requests == []
    _run(scenario)


def test_quiz_waiting_for_its_card_lock_survives_cancellation_and_reload(tmp_path):
    async def scenario():
        async with _Origin() as origin:
            bridge = _Bridge(tmp_path / "state", origin.base)
            quiz = bridge.module("telegram_quiz")
            lock = quiz.card_lock(quiz.mint_token("task-1", "q1"))
            await lock.acquire()
            task = asyncio.create_task(bridge.publish("chat.quiz", {
                "task_id": "task-1", "quiz_id": "q1", "question": "Keep this question?", "options": []}))
            await _until(lambda: bool(bridge.waiting()))
            assert origin.requests == []
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            lock.release()
            after = _Bridge(tmp_path / "state", origin.base)
            await after.drain()
            assert len(origin.sent()) == 1 and "Keep this question?" in origin.sent()[0]["text"]
            await after.stop()
    _run(scenario)


def test_refused_progress_edits_are_replaced_by_their_new_message(tmp_path):
    async def scenario():
        async with _Origin(["ok", "400", "400", "ok"]) as origin:
            bridge = _Bridge(tmp_path / "state", origin.base, TELEGRAM_SILENT_MODE="on", TELEGRAM_MIRROR_PROGRESS="on")
            progress = {"is_progress": True, "task_id": "t1", "transport": {}}
            await bridge.publish("chat.outbound", {"text": "step one", **progress})
            await bridge.publish("chat.outbound", {"text": "step two", **progress})
            assert origin.methods() == ["sendMessage", "editMessageText", "editMessageText", "sendMessage"]
            assert origin.sent()[-1]["text"] == "step two"
            assert bridge.outbox.summary()["delivered"] == 2 and bridge.records() == []
    _run(scenario)


@pytest.mark.parametrize("restart", [False, True], ids=["same-load", "reload"])
@pytest.mark.parametrize("wait_answer", ["ok", "close"], ids=["confirmed-prefix", "unknown-prefix"])
def test_a_wait_ended_edit_replays_its_presentation_before_the_new_host_answer(tmp_path, monkeypatch,
                                                                           restart, wait_answer):
    """Real host producer -> initial card -> wait-ended edit gets 429 -> host answer -> retry. The answer edit
    also gets 429 and reloads: both earlier parts keep their exact identities, unknown never repeats, and the last
    card shows the authoritative answer with no buttons. No lifecycle event is published to rescue the retry."""
    from ouroboros import owner_quiz
    from tests.test_telegram_outbox_copies import _bridge, _Producer

    data = tmp_path / "data"

    async def scenario():
        produce = _Producer(monkeypatch, data)
        owner_quiz.record_asked(data, "work-1", quiz_id="q1", question="Which db?",
                               options=["sqlite", "postgres"], wait_for_answer=True, chat_id=1)
        assert produce.bus.send_quiz(1, "q1", "Which db?", [{"label": "sqlite"}, {"label": "postgres"}],
                                     task_id="work-1", wait_for_answer=True)[0]
        asked = produce._one("chat.quiz")
        assert owner_quiz.mark_wait_ended(data, "work-1", "q1")
        async with _Origin(["ok", "429:3", wait_answer, "429:3"]) as origin:
            bridge = _bridge(data, origin.base)
            await bridge.publish(*asked)
            [original] = bridge.records()
            assert [p["state"] for p in original["parts"]] == ["confirmed", "retry_wait"]
            assert owner_quiz.record_answered(data, "work-1", quiz_id="q1", option_index=1, request_id="web-1")
            await bridge.stop()
            if restart:
                bridge = _bridge(data, origin.base)
            await bridge.outbox._recover()
            [record] = bridge.records()
            bridge.now += 60
            await bridge.outbox._resume(record)
            assert record["state"] == "retry_wait", record
            assert [p["state"] for p in record["parts"]] == [
                "confirmed", "confirmed" if wait_answer == "ok" else "unknown", "retry_wait"]
            await bridge.stop()
            bridge = _bridge(data, origin.base)  # the newer answer's edit has its own pending presentation too
            await bridge.drain()
            assert origin.methods() == ["sendMessage"] + ["editMessageText"] * 4
            wait, wait_retry, answer, answer_retry = origin.sent("editMessageText")
            assert wait == wait_retry and "Answered:" not in wait["text"]
            assert answer == answer_retry and "\nAnswered: 2. postgres" in answer["text"]
            assert "Waiting for your answer" not in answer["text"]
            assert json.loads(answer["reply_markup"]) == {"inline_keyboard": []}
            quiz = bridge.module("telegram_quiz")
            card = quiz.quiz_for_token(bridge.api, _BOT, quiz.mint_token("work-1", "q1"))
            assert (card["state"], card["answered_index"]) == ("answered", 1)
            if wait_answer == "close":
                [record] = bridge.records()
                assert record["state"] == "unknown" and record["parts"][1]["attempts"] == 2
                assert record["parts"][0]["attempts"] == 1
            else:
                assert bridge.records() == []
            print("WAIT_ENDED_FINAL", json.dumps({"methods": origin.methods(), "last_card": answer_retry,
                                                  "outcomes": [r["state"] for r in bridge.records()]}))
            await bridge.stop()
    _run(scenario)
