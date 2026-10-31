"""The Telegram bridge's one typing refresher (#1766, DESIGN §9).

One ``sendChatAction`` decays on Telegram clients in ~5s while a reply is prepared
for minutes. These tests pin the refresher against a scripted host projection and
a recording Telegram client: renewal below the client expiry while the host says a
reply is prepared, one action per Telegram chat however many roots back it, per-chat
spacing that no event burst beats, zero state reads while idle, bounded retry of
unverified reads, Timik232's transient backoff (429 never busy-loops) and
permanent-rejection stop, the message bracket that keeps an answer from reviving
the cue it ends (and what it cannot undo), unload, reload, and the plugin wiring
through the bridge's own mirror policy. Timing is patched down.
"""

import asyncio
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import pytest


def _load():
    """Load plugin (which imports the lib modules) and return the typing module."""
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    pkg = types.ModuleType("tg_kt"); pkg.__path__ = [str(root)]; sys.modules["tg_kt"] = pkg
    spec = importlib.util.spec_from_file_location("tg_kt.plugin", root / "plugin.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return sys.modules["tg_kt.lib.telegram_typing"]


REFRESH = 0.06


@pytest.fixture
def mod(monkeypatch):
    module = _load()
    monkeypatch.setattr(module, "_REFRESH_SEC", REFRESH)
    monkeypatch.setattr(module, "_MIN_READ_GAP_SEC", 0.005)
    monkeypatch.setattr(module, "TELEGRAM_RETRY_INITIAL_SEC", 0.02)
    monkeypatch.setattr(module, "next_telegram_retry_delay", lambda current: min(current * 2, 0.08))
    return module


def _api_module():
    return sys.modules["tg_kt.lib.telegram_api"]


class _Rec:
    """Recording client: every action and message in order, optional failures or gates."""
    def __init__(self, failures=None, gate=None):
        self.calls = []
        self.failures = list(failures or [])
        self.gate = gate

    @property
    def actions(self):
        return [chat for kind, chat in self.calls if kind == "typing"]

    async def send_chat_action(self, chat_id, action="typing"):
        self.calls.append(("typing", chat_id))
        if self.gate is not None:
            await self.gate.wait()
        if self.failures:
            raise self.failures.pop(0)


class _Host:
    """A scripted reply-preparation projection; ``chats`` are already Telegram targets."""
    def __init__(self, chats=(), complete=True):
        self.chats, self.complete, self.fail, self.reads = set(chats), complete, False, 0

    def read(self):
        self.reads += 1
        if self.fail:
            raise OSError("state unavailable")
        return {"complete": self.complete, "chats": [{"chat_id": c, "transport": {}} for c in self.chats]}


def _cue(mod, host, client, warnings=None):
    return mod.TypingCue(read_state=host.read, targets=lambda state: {row["chat_id"] for row in state["chats"]},
                         client_factory=lambda: client, warn=(warnings if warnings is not None else []).append)


async def _until(condition, timeout=2.0):
    """Wait for what the cue does, not for a guessed pause: its reads hop through a worker thread, which
    on a coarse timer (Windows ticks about every 15.6 ms) takes longer than any short fixed sleep."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the cue never got there"
        await asyncio.sleep(0.005)


async def _start(cue, until=None):
    """Start the cue; with ``until``, wait for its first cycle to show it (a read, an action on the wire)."""
    task = asyncio.get_running_loop().create_task(cue.run())
    await (_until(until) if until is not None else asyncio.sleep(0.01))
    return task


async def _stop(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _deliver(cue, chat, rec, *, delivered=True):
    """What the plugin does around one message: bracket it, record it, mark it delivered."""
    async with cue.message(chat) as mark:
        rec.calls.append(("message", chat))
        if delivered:
            mark()


def test_renews_while_prepared_one_action_per_chat_and_idles_with_zero_reads(mod):
    host, rec = _Host({7}), _Rec()

    async def scenario():
        cue = _cue(mod, host, rec)
        empty_projected, targets = [], cue._targets

        def projected(state):  # the verified-empty read the cue goes idle on
            chats = targets(state)
            if state.get("complete") and not chats:
                empty_projected.append(True)
            return chats
        cue._targets = projected
        task = await _start(cue, until=lambda: rec.actions.count(7) >= 4)
        try:
            assert set(rec.actions) == {7}
            host.chats = set()
            await _until(lambda: empty_projected)  # one verified-empty read ends it
            reads, actions = host.reads, len(rec.actions)
            await asyncio.sleep(0.3)
            assert host.reads == reads, "idle reads nothing until a wake"
            assert len(rec.actions) == actions
            for _ in range(20):  # a burst of wakes shares reads
                cue.wake()
            await _until(lambda: host.reads > reads)
            await asyncio.sleep(0.05)
            assert reads < host.reads <= reads + 3
        finally:
            await _stop(task)

    asyncio.run(scenario())


def test_an_event_burst_never_beats_the_per_chat_spacing(mod, monkeypatch):
    """Wakes cut a pause short and re-read, but each Telegram chat still gets at most one
    action per ``_REFRESH_SEC`` (the old cue renewed on every coalesced read)."""
    monkeypatch.setattr(mod, "_REFRESH_SEC", 0.2)
    monkeypatch.setattr(mod, "_MIN_READ_GAP_SEC", 0.01)
    host, rec = _Host({7, 8}), _Rec()

    async def scenario():
        cue = _cue(mod, host, rec)
        task = await _start(cue)
        started = time.monotonic()
        while time.monotonic() - started < 1.0:
            cue.wake()  # e.g. chat.typing / chat.outbound rows for every progress line
            await asyncio.sleep(0.005)
        elapsed = time.monotonic() - started
        await _stop(task)
        assert host.reads > 20, "the burst did re-read"
        for chat in (7, 8):
            assert 3 <= rec.actions.count(chat) <= int(elapsed / 0.2) + 2, rec.calls

    asyncio.run(scenario())


def test_a_wake_rediscovers_preparation_and_reload_reads_once_at_start(mod):
    host, rec = _Host(), _Rec()

    async def scenario():
        cue = _cue(mod, host, rec)
        task = await _start(cue, until=lambda: host.reads >= 1)
        assert host.reads == 1 and rec.actions == []
        host.chats = {5}  # e.g. a warm resume: the host now prepares, an event wakes the cue
        cue.wake()
        await _until(lambda: 5 in rec.actions)
        await _stop(task)
        reads = host.reads
        cue.wake()  # unloaded: a late wake is a no-op
        await asyncio.sleep(0.05)
        assert host.reads == reads
        fresh, before = _cue(mod, host, rec), len(rec.actions)  # a reload: no event at all
        task = await _start(fresh, until=lambda: len(rec.actions) > before)
        assert host.reads > reads and rec.actions[before:] == [5]
        await _stop(task)

    asyncio.run(scenario())


def test_unverified_reads_send_nothing_retry_boundedly_then_idle(mod):
    host, rec = _Host({3}), _Rec()

    async def scenario():
        first_sent = asyncio.Event()
        send = rec.send_chat_action

        async def send_then_fail(chat_id, action="typing"):
            await send(chat_id, action)
            host.fail = True  # before another read, rather than racing the renewal clock
            first_sent.set()

        rec.send_chat_action = send_then_fail
        cue = _cue(mod, host, rec)
        task = await _start(cue)
        await asyncio.wait_for(first_sent.wait(), timeout=2)
        actions = len(rec.actions)
        await asyncio.sleep(REFRESH * 40)
        assert len(rec.actions) == actions, "a failed read renews nothing"
        reads = host.reads
        await asyncio.sleep(0.5)
        assert host.reads == reads, "the bounded retry ended; the cue idles"
        host.fail, host.complete, host.chats = False, False, set()
        cue.wake()
        await asyncio.sleep(0.05)
        assert host.reads == reads + 1, "an idle cue does not retry an incomplete empty read"
        await _stop(task)

    asyncio.run(scenario())


def test_transient_failure_backs_off_then_recovers(mod):
    TelegramTransportError = _api_module().TelegramTransportError
    host, rec = _Host({7}), _Rec(failures=[TelegramTransportError("dead network")])

    async def scenario():
        task = await _start(_cue(mod, host, rec))
        await asyncio.sleep(REFRESH * 8)
        assert len(rec.actions) >= 3, "the first failure backed off, then the cadence resumed"
        await _stop(task)

    asyncio.run(scenario())


def test_transient_rejection_429_backs_off_without_busy_loop(mod):
    """Regression for the review finding: a transient TelegramRequestRejected (429/5xx)
    paces like any transient failure instead of retrying with zero delay."""
    TelegramRequestRejected = _api_module().TelegramRequestRejected
    rejection = TelegramRequestRejected("Telegram API sendChatAction returned HTTP 429.", status_code=429)
    host, rec = _Host({7}), _Rec(failures=[rejection] * 100)

    async def scenario():
        task = await _start(_cue(mod, host, rec))
        await asyncio.sleep(0.4)
        assert 2 <= len(rec.actions) <= 8, rec.calls
        await _stop(task)

    asyncio.run(scenario())


def test_permanent_rejection_mutes_that_chat_until_a_message_reaches_it(mod):
    TelegramRequestRejected = _api_module().TelegramRequestRejected
    rejection = TelegramRequestRejected("Telegram API sendChatAction returned HTTP 403.", status_code=403)
    host, rec, warnings = _Host({9, 10}), _Rec(failures=[rejection]), []

    async def scenario():
        cue = _cue(mod, host, rec, warnings)
        task = await _start(cue, until=lambda: rec.actions.count(10) >= 3)
        try:
            assert rec.actions.count(9) == 1, rec.calls
            assert len(warnings) == 1 and "stopped" in warnings[0]
            before = rec.actions.count(10)
            await _deliver(cue, 9, rec, delivered=False)
            await _until(lambda: rec.actions.count(10) >= before + 2)  # chat 10's renewals are the clock
            assert rec.actions.count(9) == 1, "a message that did not arrive unmutes nothing"
            await _deliver(cue, 9, rec)
            await _until(lambda: rec.actions.count(9) >= 2)
        finally:
            await _stop(task)

    asyncio.run(scenario())


def test_a_chat_muted_by_rejection_idles_the_cue_until_a_message_reaches_it(mod):
    TelegramRequestRejected = _api_module().TelegramRequestRejected
    rejection = TelegramRequestRejected("Telegram API sendChatAction returned HTTP 403.", status_code=403)
    host, rec = _Host({9}), _Rec(failures=[rejection])

    async def scenario():
        cue = _cue(mod, host, rec)
        task = await _start(cue, until=lambda: rec.actions)
        await asyncio.sleep(0.1)  # the rejection settles
        reads = host.reads
        await asyncio.sleep(0.2)
        assert host.reads == reads and rec.actions == [9], "a muted-only chat keeps no reader busy"
        await _deliver(cue, 9, rec)  # the bracket wakes the cue after a delivered message
        await _until(lambda: rec.actions.count(9) >= 2)
        await _stop(task)

    asyncio.run(scenario())


def test_a_delivered_message_lets_the_next_read_renew_at_once(mod, monkeypatch):
    """A message clears the chat's indicator on the client: a still-prepared reply gets its
    action back on the next read instead of a full spacing later; a failed one keeps the spacing.
    The spacing is far longer than the wait for the renewal, so a cue that waited it out would fail."""
    monkeypatch.setattr(mod, "_REFRESH_SEC", 5.0)
    host, rec = _Host({4}), _Rec()

    async def scenario():
        cue = _cue(mod, host, rec)
        task = await _start(cue, until=lambda: rec.actions)
        assert rec.actions == [4]
        await _deliver(cue, 4, rec, delivered=False)
        await asyncio.sleep(0.1)
        assert rec.actions == [4], "an undelivered message keeps the per-chat spacing"
        await _deliver(cue, 4, rec)
        await _until(lambda: len(rec.actions) > 1)
        assert rec.calls[-2:] == [("message", 4), ("typing", 4)], rec.calls
        await _stop(task)

    asyncio.run(scenario())


def test_an_answer_waits_for_an_in_flight_action_and_is_never_followed_by_one(mod):
    """Delayed send ordering: the answer waits (bounded) for the action already on the
    wire, no action starts while it goes, and a read that began before it sends nothing after it."""
    host, rec = _Host({4}), _Rec(gate=asyncio.Event())

    async def scenario():
        cue = _cue(mod, host, rec)
        task = await _start(cue, until=lambda: rec.actions)
        assert rec.actions == [4]  # in flight, held by the gate
        answer = asyncio.get_running_loop().create_task(_deliver(cue, 4, rec))
        await asyncio.sleep(0.02)
        assert not answer.done(), "the answer waits for the action already sent"
        host.chats = set()  # the host already handed the answer off
        rec.gate.set()
        await asyncio.wait_for(answer, 1)
        await asyncio.sleep(REFRESH * 3)
        assert rec.calls == [("typing", 4), ("message", 4)], rec.calls
        await _stop(task)

    asyncio.run(scenario())


def test_no_action_starts_while_a_message_is_on_its_way(mod, monkeypatch):
    monkeypatch.setattr(mod, "_REFRESH_SEC", 0.02)
    host, rec = _Host({4}), _Rec()

    async def scenario():
        cue = _cue(mod, host, rec)
        task = await _start(cue, until=lambda: rec.actions)
        async with cue.message(4) as mark:
            before = len(rec.actions)
            for _ in range(5):
                cue.wake()
                await asyncio.sleep(0.02)
            assert len(rec.actions) == before, rec.calls
            rec.calls.append(("message", 4))
            mark()
        await _until(lambda: len(rec.calls) >= 3)
        assert rec.calls[:3] == [("typing", 4), ("message", 4), ("typing", 4)], rec.calls
        await _stop(task)

    asyncio.run(scenario())


class _Telegram:
    """Telegram's side of the wire: an action counts as accepted when the server gets to
    it, whatever the client later does locally. ``release`` lets the server answer."""
    def __init__(self):
        self.log, self.server = [], threading.Event()

    def accepted_after(self, kind):
        """Accepted after ``kind`` by the log's order: two stamps within one tick of the clock compare equal
        (Python 3.10's monotonic clock ticks every 15.6 ms on Windows)."""
        kinds = [k for k, _ in self.log]
        return "accepted" in kinds and kind in kinds and kinds.index("accepted") > kinds.index(kind)

    async def send_chat_action(self, chat_id, action="typing"):
        self.log.append(("written", time.monotonic()))
        threading.Thread(target=lambda: (self.server.wait(5), self.log.append(("accepted", time.monotonic()))),
                         daemon=True).start()
        await asyncio.get_running_loop().run_in_executor(None, self.server.wait, 5)


def test_a_hung_action_delays_the_answer_only_for_the_bound_and_may_still_land_after_it(mod, monkeypatch):
    """The order wait is bounded; past it the client stops waiting (local cancel), and the
    request Telegram already received can still be accepted AFTER the answer. That residual
    is real: the cue only promises it starts no new action there afterwards."""
    monkeypatch.setattr(mod, "_ORDER_WAIT_SEC", 0.1)
    host, telegram = _Host({4}), _Telegram()

    async def scenario():
        cue = mod.TypingCue(read_state=host.read, targets=lambda s: {r["chat_id"] for r in s["chats"]},
                            client_factory=lambda: telegram, warn=lambda _t: None)
        task = await _start(cue, until=lambda: 4 in cue._inflight)
        inflight = cue._inflight[4]
        host.chats = set()  # the answer was handed off
        started = time.monotonic()
        async with cue.message(4) as mark:
            telegram.log.append(("message", time.monotonic()))
            mark()
        assert time.monotonic() - started < 0.5, "the answer waited only for the bound"
        await asyncio.sleep(0)
        assert inflight.cancelled(), "the client stopped waiting for the hung action"
        telegram.server.set()  # Telegram processes the request it already had
        await _until(lambda: any(kind == "accepted" for kind, _ in telegram.log))  # on its own server thread
        assert telegram.accepted_after("message"), "an accepted action after the answer stays possible"
        assert [kind for kind, _ in telegram.log].count("written") == 1, "no new action started after the answer"
        await _stop(task)

    asyncio.run(scenario())


def test_a_read_begun_before_a_message_never_sends_after_it(mod):
    rec, release, entered, reads = _Rec(), threading.Event(), threading.Event(), []

    async def scenario():

        def slow_read():
            # Only the read that began before the final still saw the reply prepared.
            reads.append(1)
            if len(reads) > 1:
                return {"complete": True, "chats": []}
            entered.set()
            release.wait(2)
            return {"complete": True, "chats": [{"chat_id": 6, "transport": {}}]}

        cue = mod.TypingCue(read_state=slow_read, targets=lambda s: {r["chat_id"] for r in s["chats"]},
                            client_factory=lambda: rec, warn=lambda _t: None)
        task = asyncio.get_running_loop().create_task(cue.run())
        await asyncio.get_running_loop().run_in_executor(None, entered.wait, 2)
        await _deliver(cue, 6, rec)  # the final lands while that read is still out
        release.set()
        await asyncio.sleep(0.05)
        assert rec.calls == [("message", 6)], "the stale read was fenced"
        assert len(reads) >= 2, "the message woke a fresh read"
        await _stop(task)
        release.set()

    asyncio.run(scenario())


def test_unload_cancels_in_flight_sends(mod):
    host, rec = _Host({8}), _Rec(gate=asyncio.Event())

    async def scenario():
        cue = _cue(mod, host, rec)
        task = await _start(cue, until=lambda: cue._inflight)
        inflight = list(cue._inflight.values())
        assert inflight and not inflight[0].done()
        await _stop(task)
        await asyncio.sleep(0)
        assert inflight[0].cancelled() and cue._inflight == {}

    asyncio.run(scenario())


class _PluginApi:
    def __init__(self, tmp, settings=None):
        self._tmp, self.logs = tmp, []
        state = tmp / "state" / "skills" / "telegram"
        state.mkdir(parents=True, exist_ok=True)
        if settings is not None:
            import json
            (state / "settings.json").write_text(json.dumps(settings))

    def get_state_dir(self):
        return str(self._tmp / "state" / "skills" / "telegram")

    def get_settings(self, keys):
        return {}

    def get_runtime_info(self):
        return {"data_dir": str(self._tmp)}

    def log(self, *args, **kwargs):
        self.logs.append(args)


class _Cue:
    """Records what the plugin asks of the cue."""
    def __init__(self):
        self.events = []

    def wake(self):
        self.events.append("wake")

    def message(self, chat_id):
        cue = self

        class Bracket:
            async def __aenter__(self):
                cue.events.append(("before", chat_id))
                return lambda: cue.events.append(("delivered", chat_id))

            async def __aexit__(self, *exc):
                cue.events.append(("after", chat_id))
                return False

        return Bracket()


def test_typing_events_only_wake_and_outbound_text_is_bracketed(mod, tmp_path, monkeypatch):
    plugin = sys.modules["tg_kt.plugin"]
    api = _PluginApi(tmp_path, {"TELEGRAM_CHAT_ID": "77", "TELEGRAM_MIRROR_MODE": "all"})
    calls = []

    class Client:
        async def send_chat_action(self, chat_id, action="typing"):
            calls.append(("typing", chat_id))

        async def send_message(self, chat_id, text, parse_mode=""):
            calls.append(("message", chat_id))
            return 1

    monkeypatch.setattr(plugin, "_telegram_client", lambda *_a, **_k: Client())
    cue = _Cue()

    async def scenario():
        await plugin._make_typing(cue)({"chat_id": 1, "activity_id": "a", "transport": {}})
        assert calls == [] and cue.events == ["wake"], "a start is a wake, never itself a cue"
        await plugin._make_outbound(api, cue)({"chat_id": 1, "task_id": "a", "text": "the answer", "transport": {}})
        assert calls == [("message", 77)]
        assert cue.events[1:] == [("before", 77), ("delivered", 77), ("after", 77)]
        await plugin._make_outbound(api, cue)({"chat_id": 1, "task_id": "a", "text": "tick", "is_progress": True})
        assert len(cue.events) == 4 and calls == [("message", 77)], "a dropped progress row touches nothing"

    asyncio.run(scenario())


def test_a_quiz_card_is_bracketed_like_a_reply(mod, tmp_path, monkeypatch):
    plugin = sys.modules["tg_kt.plugin"]
    quiz = sys.modules["tg_kt.lib.telegram_quiz"]
    api = _PluginApi(tmp_path, {"TELEGRAM_CHAT_ID": "77", "TELEGRAM_MIRROR_MODE": "all"})
    cue = _Cue()

    async def card(_client, chat_id, **_kw):
        cue.events.append(("card", chat_id))
        return 5, False

    async def nothing(*_a, **_k):
        return None

    monkeypatch.setattr(plugin, "_telegram_client", lambda *_a, **_k: object())
    monkeypatch.setattr(quiz, "send_quiz_card", card)
    monkeypatch.setattr(quiz, "remember_quiz", lambda *_a, **_k: None)
    monkeypatch.setattr(quiz, "apply_retained_fact", nothing)

    async def scenario():
        await plugin._make_quiz(api, cue)({"chat_id": 1, "task_id": "t", "quiz_id": "q", "question": "Which?",
                                           "options": ["a", "b"], "wait_for_answer": True, "transport": {}})
        assert cue.events == [("before", 77), ("card", 77), ("delivered", 77), ("after", 77)], cue.events

    asyncio.run(scenario())


@pytest.mark.parametrize("settings,transport,expected", [
    ({"TELEGRAM_CHAT_ID": "77", "TELEGRAM_MIRROR_MODE": "all"}, {}, {77}),
    ({"TELEGRAM_CHAT_ID": "77", "TELEGRAM_MIRROR_MODE": "telegram_only"}, {}, set()),
    ({"TELEGRAM_CHAT_ID": "77", "TELEGRAM_MIRROR_MODE": "telegram_only"},
     {"kind": "telegram", "conversation_id": "77"}, {77}),
    ({}, {"kind": "telegram", "conversation_id": "88"}, {88}),
    ({}, {}, set()),
])
def test_the_cue_maps_host_chats_through_the_bridges_mirror_policy(mod, tmp_path, settings, transport, expected):
    plugin = sys.modules["tg_kt.plugin"]
    cue = plugin._typing_cue(_PluginApi(tmp_path, settings))
    state = {"complete": True, "chats": [{"chat_id": 1, "transport": transport}, {"chat_id": 4242, "transport": transport}]}
    assert cue._targets(state) == expected
