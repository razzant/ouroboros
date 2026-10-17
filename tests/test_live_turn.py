"""The live voice-call driver against fakes of the lanes it meets.

``FakeEngine`` implements ``LiveEngine`` (the engines lane), ``FakeHub`` the
``CallHub`` contract (the audio lane) and ``FakeLedger`` the usage ledger. The
call runs on its own thread exactly as a turn does; the tests feed peer events
and engine events from outside and read what reached the hub, the engine, the
chat and the task row.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import live_turn
from ouroboros.call_protocol import (
    STATE_CLOSED, STATE_CLOSING, STATE_CONNECTING, STATE_LIVE, STATE_RECONNECTING,
    CallSpec, PeerAudio, PeerControl, PeerInfo, PeerJoined, PeerLeft,
)
from ouroboros.gateways.realtime_engine import (
    ENGINE_GOOGLE, SPEAKER_OUROBOROS, SPEAKER_OWNER,
    AudioIn, AudioOut, Caption, Close, ContextItem, EngineError, GoAway, Interrupt, PlayedAudio,
    Ready, ResponseDone, ResponseStarted, ResumeCheckpoint, SpeechStarted, ToolCall, ToolCallCancelled, ToolResult,
    UpdateTools,
)
from ouroboros.usage_accounting import BudgetExceeded

_END = object()
OWNER = PeerInfo(peer_id="owner-1", role="owner", mic=True, speaker=True, label="Anton")


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


def _spec(**overrides) -> CallSpec:
    fields = dict(call_id="call-1", chat_id=1, engine=ENGINE_GOOGLE, model="gemini-3.8-live", voice="Kore",
                  opened_by="owner", presence_binding_id="", max_minutes=0, label="Test call", client_message_id="cm-1")
    fields.update(overrides)
    return CallSpec(**fields)


class FakeEngine:
    """``LiveEngine``: records the session config and every command; the test pushes events."""

    auto_ready = True

    def __init__(self) -> None:
        self.config = None
        self.sent: list = []
        self.closed: list = []
        self._queue = None
        self._loop = None

    async def open(self, config) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self.config = config  # last: ``Harness.start`` returns on it, and a test may push at once
        if self.auto_ready:
            self._queue.put_nowait(Ready({"input_rate": 16000, "output_rate": 24000}))  # Gemini Live's wire rates

    async def send(self, command) -> None:
        self.sent.append(command)

    async def events(self):
        while True:
            event = await self._queue.get()
            if event is _END:
                return
            yield event

    async def close(self, reason: str) -> None:
        self.closed.append(reason)

    def push(self, event) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def end_stream(self) -> None:
        self.push(_END)

    def sent_of(self, kind):
        return [command for command in self.sent if isinstance(command, kind)]


class FakeLLM:
    def __init__(self) -> None:
        self.engines: list = []
        self.routes: list = []
        self.fail_opens = 0

    def open_live(self, route):
        if self.fail_opens:
            self.fail_opens -= 1
            raise RuntimeError("engine unreachable")
        self.routes.append(route)
        self.engines.append(FakeEngine())
        return self.engines[-1]


class FakeHub:
    """``CallHub``: the test feeds peer events; the driver's output is recorded."""

    def __init__(self) -> None:
        self.inbox: queue.Queue = queue.Queue()
        self.played: list = []
        self.flushes = 0
        self.states: list = []
        self.captions: list = []
        self.rates = None
        self.present = [OWNER]
        self.closed: list = []
        self.activity: list = []
        self.words: list = []  # (state, detail, action) of every state that carried words
        self.ended_by_owner = False

    def set_engine_rates(self, input_rate, output_rate) -> None:
        self.rates = (input_rate, output_rate)

    def next_event(self, timeout_s):
        try:
            return self.inbox.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def play(self, pcm, item_id) -> None:
        self.played.append((item_id, pcm))

    def flush(self) -> int:
        self.flushes += 1
        return self.flushes

    def send_state(self, state, reason="", *, phase="", detail="", action="") -> None:
        self.states.append((state, reason))
        if detail or action:
            self.words.append((state, detail, action))

    def send_activity(self, activity, label="") -> None:
        self.activity.append((activity, label))

    def send_caption(self, speaker, text, final, item_id="") -> None:
        self.captions.append((speaker, text, final, item_id))

    def peers(self):
        return tuple(self.present)

    def close(self, reason, **words) -> None:
        self.closed.append(reason)

    def feed(self, event) -> None:
        self.inbox.put(event)

    def control(self, kind, **fields) -> None:
        self.feed(PeerControl(peer_id="owner-1", role="owner", message={"type": kind, **fields}))


class FakeLedger:
    def __init__(self) -> None:
        self.reserved: list = []
        self.settled: list = []
        self.abandoned: list = []
        self.refused: list = []
        self.warmed: list = []
        self.refuse = False

    def warm_quote(self, **fields):
        self.warmed.append((fields, threading.current_thread().name))
        return None

    def reserve_live_response(self, **fields):
        if self.refuse:
            raise BudgetExceeded("the call's budget is spent")
        attempt = {"key": fields["response_key"], **fields}
        self.reserved.append(attempt)
        return attempt

    def record_refused_live_response(self, **fields):
        self.refused.append(fields)

    def settle_live_response(self, attempt, *, usage, raw_usage, status) -> None:
        self.settled.append((attempt["key"], status, dict(usage)))

    def abandon_live_response(self, attempt, *, reason) -> None:
        self.abandoned.append((attempt["key"], reason))


def _schema(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}


class Harness:
    """One call on its own thread, wired to the fakes; ``tool_hook`` is what a tool does."""

    def __init__(self, monkeypatch, tmp_path, *, spec=None, task=None) -> None:
        from ouroboros import loop, loop_budget
        from supervisor import message_bus

        self.root = tmp_path
        self.task = task
        (tmp_path / "logs").mkdir(exist_ok=True)
        self.spec = spec or _spec()
        self.hub, self.llm, self.ledger = FakeHub(), FakeLLM(), FakeLedger()
        self.schemas = [_schema("knowledge_read"), _schema("web_search"), _schema("enable_tools")]
        self.progress: list = []
        self.broadcasts: list = []
        self.batches: list = []
        self.tool_hook = None
        self.result = None
        self.error = None
        monkeypatch.setattr(live_turn, "_hub_for", lambda call_id: self.hub if call_id == self.spec.call_id else None)
        monkeypatch.setattr(live_turn, "_ledger", lambda: self.ledger)
        monkeypatch.setattr(live_turn, "initial_call_schemas", lambda tools, spec=None: list(self.schemas))
        monkeypatch.setattr(loop, "_setup_dynamic_tools",
                            lambda tools, schemas, messages, context_mode="max": (list(schemas), None))
        monkeypatch.setattr(loop, "handle_tool_calls", self._handle_tool_calls)
        monkeypatch.setattr(loop, "_measure_tool_result_batch", lambda ctx, messages, schemas, *, round_id: None)
        monkeypatch.setattr(loop_budget, "_cleanup_loop_resources", lambda executor, exit_ctx: None)
        monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(
            broadcast=self.broadcasts.append, send_chat_action=lambda *a, **k: None))
        self.ctx = SimpleNamespace(task_attempt=1, active_context_mode="nano", drive_root=tmp_path, task_id="t-call")
        self.tools = SimpleNamespace(_ctx=self.ctx)
        self.messages = [{"role": "system", "content": "You are Ouroboros, on a call."},
                         {"role": "user", "content": "[Live call] A live voice call is open in this room."}]
        self.incoming: queue.Queue = queue.Queue()
        self.events: queue.Queue = queue.Queue()
        from ouroboros.utils import utc_now_iso

        self.context_captured_at = utc_now_iso()  # as the agent records it before the context reads the room

    def emit_progress(self, text, **kwargs) -> None:
        self.progress.append((text, kwargs))

    def _handle_tool_calls(self, tool_calls, tools, drive_logs, task_id, executor, messages, llm_trace,
                           emit_progress, *, fit_candidate, tool_schemas) -> None:
        self.batches.append([call["id"] for call in tool_calls])
        for call in tool_calls:
            if self.tool_hook is not None:
                self.tool_hook(call, messages, tool_schemas)
            else:
                messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                                 "content": f"result of {call['function']['name']}"})

    def start(self) -> FakeEngine:
        def _run():
            try:
                self.result = live_turn.run_live_turn(
                    messages=self.messages, tools=self.tools, llm=self.llm, drive_logs=self.root / "logs",
                    emit_progress=self.emit_progress, incoming_messages=self.incoming, task_id="t-call",
                    task_type="task", budget_remaining_usd=10.0, event_queue=self.events, drive_root=self.root,
                    spec=self.spec, task=self.task, context_captured_at=self.context_captured_at)
            except BaseException as exc:  # surfaced to the test thread
                self.error = exc

        self.thread = threading.Thread(target=_run, name="call-under-test", daemon=True)
        self.thread.start()
        assert _wait_for(lambda: bool(self.llm.engines) and self.llm.engines[-1].config is not None)
        return self.llm.engines[-1]

    def engine(self) -> FakeEngine:
        return self.llm.engines[-1]

    def stop(self) -> None:
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

        assert write_owner_message(self.root, "stop", "t-call", msg_id="owner-stop", kind=KIND_FINALIZE_NOW)

    def finish(self, timeout=15.0):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the call did not end"
        if self.error is not None:
            raise self.error
        return self.result

    def counts_heard(self) -> int:
        return sum(str(m.get("content")).startswith("[Heard in the call]") for m in self.messages)

    def chat_rows(self) -> list:
        path = self.root / "logs" / "chat.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def task_row(self) -> dict:
        from ouroboros.task_results import load_task_result

        return load_task_result(self.root, "t-call") or {}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Build calls; whatever the test leaves running is stopped and joined here."""
    created: list = []

    def make(**kwargs) -> Harness:
        created.append(Harness(monkeypatch, tmp_path, **kwargs))
        return created[-1]

    yield make
    for h in created:
        thread = getattr(h, "thread", None)
        if thread is not None and thread.is_alive():
            h.stop()
            thread.join(15)
    assert _wait_for(lambda: not any(t.name.startswith("call-") for t in threading.enumerate()), 5)


# --- opening -------------------------------------------------------------------


def test_live_state_waits_for_provider_confirmation(harness, monkeypatch):
    monkeypatch.setattr(FakeEngine, "auto_ready", False)
    h = harness()
    engine = h.start()
    assert h.hub.states == [(STATE_CONNECTING, "")]
    engine.push(Ready({"input_rate": 24000, "output_rate": 24000}))
    assert _wait_for(lambda: h.hub.states[-1] == (STATE_LIVE, ""))
    h.stop()
    h.finish()


def test_explicit_owner_bye_ends_the_call_but_meeting_bye_does_not(harness):
    h = harness()
    h.start()
    h.hub.feed(PeerControl("meeting", "meeting", {"type": "bye"}))
    time.sleep(.05)
    assert h.thread.is_alive()
    h.hub.control("bye")
    text, usage, _ = h.finish()
    assert usage["live_call"]["ended"] == "owner_bye" and "utterances" in text


def test_the_session_opens_with_the_fitted_instructions_and_the_nano_view(harness):
    h = harness()
    engine = h.start()
    config = engine.config
    assert config.instructions == "You are Ouroboros, on a call."
    assert [schema["function"]["name"] for schema in config.tools] == ["knowledge_read", "web_search", "enable_tools"]
    assert config.route.engine == ENGINE_GOOGLE and config.route.model == "gemini-3.8-live" and config.route.voice == "Kore"
    assert config.native_search is True and config.resume_handle == ""  # Gemini's own search: on by default
    assert config.history == ({"role": "user", "content": "[Live call] A live voice call is open in this room."},)
    assert h.llm.routes[0].engine == ENGINE_GOOGLE
    assert _wait_for(lambda: (STATE_LIVE, "") in h.hub.states)
    assert h.hub.states[0] == (STATE_CONNECTING, "") and h.hub.rates == (16000, 24000)
    h.stop()
    text, usage, trace = h.finish()
    assert [state for state, _ in h.hub.states] == [STATE_CONNECTING, STATE_LIVE, STATE_CLOSING, STATE_CLOSED]
    assert engine.sent_of(Close) and engine.closed == ["owner_stop"] and h.hub.closed == ["owner_stop"]


def test_a_missing_hub_is_a_typed_failure_without_an_engine(harness, monkeypatch, tmp_path):
    h = harness(spec=_spec(call_id="nobody-opened-this"))
    monkeypatch.setattr(live_turn, "_hub_for", lambda call_id: None)
    text, usage, trace = live_turn.run_live_turn(
        messages=h.messages, tools=h.tools, llm=h.llm, drive_logs=tmp_path / "logs", emit_progress=h.emit_progress,
        incoming_messages=h.incoming, task_id="t-call", task_type="task", budget_remaining_usd=1.0,
        event_queue=h.events, drive_root=tmp_path, spec=h.spec)
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "live_call_hub_missing"
    assert usage["terminal_origin"] == "host_notice" and h.llm.engines == [] and "could not run" in text


def test_an_engine_that_cannot_open_ends_the_call_as_a_typed_failure(harness, monkeypatch, tmp_path):
    h = harness()
    h.llm.fail_opens = 1
    thread = threading.Thread(target=lambda: setattr(h, "result", live_turn.run_live_turn(
        messages=h.messages, tools=h.tools, llm=h.llm, drive_logs=tmp_path / "logs", emit_progress=h.emit_progress,
        incoming_messages=h.incoming, task_id="t-call", task_type="task", budget_remaining_usd=1.0,
        event_queue=h.events, drive_root=tmp_path, spec=h.spec)), daemon=True)
    thread.start()
    thread.join(15)
    assert not thread.is_alive()
    text, usage, _trace = h.result
    assert usage["reason_code"] == "live_engine_unavailable" and usage["execution_status"] == "infra_failed"
    assert h.hub.states[-1] == (STATE_CLOSED, "engine_unavailable") and h.hub.closed == ["engine_unavailable"]


# --- audio, captions, barge-in ------------------------------------------------------


def test_audio_flows_both_ways_and_a_muted_peer_is_not_heard(harness, monkeypatch, tmp_path):
    h = harness()
    engine = h.start()
    h.hub.feed(PeerAudio(peer_id="owner-1", role="owner", pcm=b"\x01\x02" * 100))
    assert _wait_for(lambda: engine.sent_of(AudioIn))
    assert engine.sent_of(AudioIn)[0] == AudioIn(pcm=b"\x01\x02" * 100, rate=16000)
    engine.push(AudioOut(pcm=b"\x09" * 50, rate=24000, item_id="item-a"))
    assert _wait_for(lambda: h.hub.played == [("item-a", b"\x09" * 50)])
    engine.push(AudioOut(pcm=b"\x0a" * 50, rate=24000, item_id="item-a"))
    assert _wait_for(lambda: len(h.hub.played) == 2)
    # Frames carry no item id: the item is named once, by an empty caption, before its audio.
    assert h.hub.captions == [(SPEAKER_OUROBOROS, "", False, "item-a")]
    h.hub.control("mute", muted=True)
    time.sleep(0.05)
    h.hub.feed(PeerAudio(peer_id="owner-1", role="owner", pcm=b"\x03" * 10))
    h.hub.control("mute", muted=False)
    h.hub.feed(PeerAudio(peer_id="owner-1", role="owner", pcm=b"\x04" * 10))
    assert _wait_for(lambda: len(engine.sent_of(AudioIn)) == 2)
    assert [command.pcm for command in engine.sent_of(AudioIn)] == [b"\x01\x02" * 100, b"\x04" * 10]
    h.stop()
    h.finish()


def test_final_captions_become_rows_of_the_room_without_an_owner_turn(harness, monkeypatch, tmp_path):
    h = harness()
    engine = h.start()
    engine.push(Caption(speaker=SPEAKER_OWNER, text="what is", final=False, item_id="u1"))
    engine.push(Caption(speaker=SPEAKER_OWNER, text="what is the weather", final=True, item_id="u1"))
    assert _wait_for(lambda: len(h.chat_rows()) == 1)
    assert h.hub.captions[0] == (SPEAKER_OWNER, "what is", False, "u1")
    row = h.chat_rows()[0]
    assert row["direction"] == "in" and row["text"] == "what is the weather" and row["source"] == "call"
    assert row["task_id"] == "t-call" and row["client_message_id"] == "call:call-1:u1" and row["sender_label"] == "voice"
    assert _wait_for(lambda: h.broadcasts)  # the echo follows the row on the call's thread
    assert h.broadcasts[0]["role"] == "user" and h.broadcasts[0]["source"] == "call" and h.broadcasts[0]["content"] == "what is the weather"
    assert h.events.empty() and h.incoming.empty()  # the owner's words opened no turn
    engine.push(ResponseStarted(item_id="a1", response_id="resp-1"))
    engine.push(AudioOut(pcm=b"\x01" * 10, rate=24000, item_id="a1"))
    engine.push(Caption(speaker=SPEAKER_OUROBOROS, text="Sunny.", final=True, item_id="a1"))
    assert _wait_for(lambda: (SPEAKER_OUROBOROS, "Sunny.", True, "a1") in h.hub.captions)
    time.sleep(0.05)
    assert h.progress == []  # not written until the peer confirms playback (or the grace ends)
    h.hub.control("played", item_id="a1", played_ms=800)
    assert _wait_for(lambda: len(h.progress) == 1)
    text, kwargs = h.progress[0]
    assert text == "Sunny." and kwargs["narration"] is True
    assert kwargs["_progress_meta_override"]["spoken"] == {"call_id": "call-1", "item_id": "a1", "status": "played", "played_ms": 800}
    assert engine.sent_of(PlayedAudio) == [PlayedAudio(item_id="a1", played_ms=800)]
    assert [m["role"] for m in h.messages[2:]] == ["user", "assistant"] and all(m["_call_row"] for m in h.messages[2:])
    h.stop()
    h.finish()


def test_a_spoken_row_is_written_as_generated_when_no_playback_ack_arrives(harness, monkeypatch, tmp_path):
    monkeypatch.setattr(live_turn, "SPOKEN_ACK_GRACE_S", 0.3)
    h = harness()
    engine = h.start()
    engine.push(Caption(speaker=SPEAKER_OUROBOROS, text="Unheard.", final=True, item_id="a9"))
    assert _wait_for(lambda: len(h.progress) == 1)
    assert h.progress[0][1]["_progress_meta_override"]["spoken"]["status"] == "generated"
    h.stop()
    h.finish()


def test_barge_in_flushes_the_peers_and_interrupts_the_item(harness, monkeypatch, tmp_path):
    h = harness()
    engine = h.start()
    engine.push(AudioOut(pcm=bytes(9600), rate=24000, item_id="a1"))   # 200 ms delivered
    assert _wait_for(lambda: len(h.hub.played) == 1)
    h.hub.control("played", item_id="a1", played_ms=205)   # the mark runs past the audio
    assert _wait_for(lambda: engine.sent_of(PlayedAudio))
    engine.push(SpeechStarted())
    assert _wait_for(lambda: engine.sent_of(Interrupt))
    # The cut stays inside the delivered audio; the provider refuses one beyond it.
    assert h.hub.flushes == 1 and engine.sent_of(Interrupt) == [Interrupt(item_id="a1", played_ms=200)]
    engine.push(AudioOut(pcm=b"\x02" * 10, rate=24000, item_id="a1"))   # late audio of the cut item
    engine.push(Caption(speaker=SPEAKER_OUROBOROS, text="I was say", final=True, item_id="a1"))
    assert _wait_for(lambda: len(h.progress) == 1)
    assert h.progress[0][1]["_progress_meta_override"]["spoken"]["status"] == "interrupted"
    assert len(h.hub.played) == 1
    h.stop()
    h.finish()


def test_barge_in_after_final_caption_retains_playback_cut(harness):
    h = harness()
    engine = h.start()
    engine.push(AudioOut(bytes(48000 * 3), 24000, "a1"))
    engine.push(Caption(SPEAKER_OUROBOROS, "A three second sentence.", True, "a1"))
    h.hub.control("played", item_id="a1", played_ms=1000)
    assert _wait_for(lambda: engine.sent_of(PlayedAudio))
    assert h.progress == []  # A final generated caption is not completed playback.
    engine.push(SpeechStarted())
    assert _wait_for(lambda: len(h.progress) == 1)
    assert h.progress[0][1]["_progress_meta_override"]["spoken"]["status"] == "interrupted"
    assert engine.sent_of(Interrupt) == [Interrupt("a1", 1000)]
    h.hub.control("played", item_id="a1", played_ms=1100)
    h.stop()
    h.finish()
    assert len(h.progress) == 1


def test_barge_in_truncates_all_unplayed_items_of_a_response(harness):
    h = harness()
    engine = h.start()
    for item in ("commentary", "answer"):
        engine.push(AudioOut(bytes(48000 * 3), 24000, item))
        engine.push(Caption(SPEAKER_OUROBOROS, item, True, item))
    h.hub.control("played", item_id="commentary", played_ms=1000)
    assert _wait_for(lambda: engine.sent_of(PlayedAudio))
    engine.push(SpeechStarted())
    assert _wait_for(lambda: len(h.progress) == 2)
    assert {(c.item_id, c.played_ms) for c in engine.sent_of(Interrupt)} == {("commentary", 1000), ("answer", 0)}
    assert all(p[1]["_progress_meta_override"]["spoken"]["status"] == "interrupted" for p in h.progress)
    h.stop()
    h.finish()


# --- tools -----------------------------------------------------------------------


def test_tool_worker_keeps_call_contextvars_and_forwards_earlier_message_updates(harness, monkeypatch):
    import contextvars

    witness = contextvars.ContextVar("live_tool_scope", default="missing")
    original = live_turn._LiveCall.run
    def scoped_run(self):
        witness.set("call-settings-and-usage")  # bound on the turn thread, as the agent's usage scope is
        return original(self)
    monkeypatch.setattr(live_turn._LiveCall, "run", scoped_run)
    h = harness()
    def hook(call, messages, schemas):
        assert witness.get() == "call-settings-and-usage"
        messages[1]["content"] = "A review updated an existing message."
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "done"})
    h.tool_hook = hook
    engine = h.start()
    engine.push(ToolCall("scoped", "web_search", "{}"))
    assert _wait_for(lambda: engine.sent_of(ToolResult))
    assert any(c.text == "A review updated an existing message." for c in engine.sent_of(ContextItem))
    h.stop()
    h.finish()


def test_tool_calls_run_as_one_serialized_batch_and_every_change_goes_back(harness, monkeypatch, tmp_path):
    h = harness()
    overlap = {"max": 0, "now": 0}
    gate = threading.Lock()

    def hook(call, messages, schemas):
        with gate:
            overlap["now"] += 1
            overlap["max"] = max(overlap["max"], overlap["now"])
        time.sleep(0.05)
        messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                         "content": [{"type": "text", "text": f"saw {call['function']['name']}"}]})
        if call["function"]["name"] == "web_search":
            messages.append({"role": "user", "content": "[Dialogue update] the owner typed meanwhile"})
        with gate:
            overlap["now"] -= 1

    h.tool_hook = hook
    engine = h.start()
    engine.push(ToolCall(call_id="c1", name="knowledge_read", arguments="{}"))
    engine.push(ToolCall(call_id="c2", name="web_search", arguments='{"q": "x"}'))
    assert _wait_for(lambda: len(engine.sent_of(ToolResult)) == 2)
    assert overlap["max"] == 1 and sum(len(b) for b in h.batches) == 2
    results = {r.call_id: r for r in engine.sent_of(ToolResult)}
    assert results["c1"].output == "saw knowledge_read" and results["c2"].output == "saw web_search"
    forwarded = [c for c in engine.sent_of(ContextItem) if c.source == "tool_batch:user"]
    assert forwarded == [ContextItem(text="[Dialogue update] the owner typed meanwhile", source="tool_batch:user")]
    assistant_rows = [m for m in h.messages if m.get("tool_calls")]
    assert {tc["id"] for row in assistant_rows for tc in row["tool_calls"]} == {"c1", "c2"}
    h.stop()
    text, usage, trace = h.finish()
    assert usage["live_call"]["tool_calls"] == 2 and h.task_row()["live_call"]["executed_tool_calls"] == ["c1", "c2"]


def test_tool_growth_reaches_the_session_as_an_update(harness, monkeypatch, tmp_path):
    h = harness()

    def hook(call, messages, schemas):
        schemas.append(_schema("read_file"))  # what enable_tools does to the resident list
        messages.append({"role": "tool", "tool_call_id": call["id"], "name": "enable_tools", "content": "enabled: read_file"})

    h.tool_hook = hook
    engine = h.start()
    engine.push(ToolCall(call_id="c-grow", name="enable_tools", arguments='{"names": ["read_file"]}'))
    assert _wait_for(lambda: engine.sent_of(UpdateTools))
    names = [s["function"]["name"] for s in engine.sent_of(UpdateTools)[0].tools]
    assert names == ["knowledge_read", "web_search", "enable_tools", "read_file"]
    assert engine.sent_of(ToolResult)[0].call_id == "c-grow"
    h.stop()
    h.finish()


def test_owner_stop_mid_tool_closes_once_the_running_tool_settles(harness, monkeypatch, tmp_path):
    h = harness()
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def hook(call, messages, schemas):
        entered.set()
        assert release.wait(10)
        messages.append({"role": "tool", "tool_call_id": call["id"], "name": "web_search", "content": "late"})
        done.set()

    h.tool_hook = hook
    engine = h.start()
    engine.push(ToolCall(call_id="c-slow", name="web_search", arguments="{}"))
    assert entered.wait(10)
    h.stop()
    assert _wait_for(lambda: (STATE_CLOSING, "owner_stop") in h.hub.states)
    assert not done.is_set() and h.thread.is_alive()
    release.set()
    text, usage, _trace = h.finish()
    assert done.is_set() and usage["live_call"]["ended"] == "owner_stop"
    assert engine.closed == ["owner_stop"] and h.hub.states[-1] == (STATE_CLOSED, "owner_stop")


def test_a_withdrawn_tool_call_does_not_run(harness, monkeypatch, tmp_path):
    h = harness()
    started = threading.Event()
    release = threading.Event()

    def hook(call, messages, schemas):
        started.set()
        assert release.wait(10)
        messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"], "content": "ok"})

    h.tool_hook = hook
    engine = h.start()
    engine.push(ToolCall(call_id="c-run", name="web_search", arguments="{}"))
    assert started.wait(10)
    engine.push(ToolCall(call_id="c-later", name="knowledge_read", arguments="{}"))
    engine.push(ToolCallCancelled(call_ids=("c-later",)))
    time.sleep(0.1)
    release.set()
    assert _wait_for(lambda: engine.sent_of(ToolResult))
    time.sleep(0.2)
    assert [c for b in h.batches for c in b] == ["c-run"]
    h.stop()
    h.finish()


# --- the ledger and the budget ----------------------------------------------------------


def test_responses_are_reserved_and_settled(harness, monkeypatch, tmp_path):
    h = harness()
    engine = h.start()
    engine.push(ResponseStarted(item_id="a1", response_id="r1"))
    engine.push(ResponseDone(item_id="a1", status="completed", usage={"text_input_tokens": 10, "audio_output_tokens": 20},
                             raw_usage={"raw": 1}, response_id="r1"))
    assert _wait_for(lambda: h.ledger.settled)
    assert h.ledger.reserved[0]["response_key"] == "r1" and h.ledger.reserved[0]["rendered_mode"] == "nano"
    assert h.ledger.reserved[0]["call_id"] == "call-1" and h.ledger.reserved[0]["task_id"] == "t-call"
    assert h.ledger.settled == [("r1", "completed", {"text_input_tokens": 10, "audio_output_tokens": 20})]
    h.stop()
    text, usage, _trace = h.finish()
    assert usage["live_responses"] == 1 and usage["live_usage"]["text_input_tokens"] == 10
    assert usage["live_usage"]["audio_output_tokens"] == 20 and h.ledger.abandoned == []


def test_a_budget_refusal_closes_the_call_and_propagates(harness, monkeypatch, tmp_path):
    h = harness()
    h.ledger.refuse = True
    engine = h.start()
    engine.push(ResponseStarted(item_id="a1", response_id="r1"))
    h.thread.join(15)
    assert not h.thread.is_alive() and isinstance(h.error, BudgetExceeded)
    assert h.hub.states[-1] == (STATE_CLOSED, "budget_exceeded") and engine.closed == ["budget_exceeded"]
    assert getattr(h.error, "_ouroboros_loop_usage", None) is not None


def test_the_minutes_ceiling_ends_the_call(harness, monkeypatch, tmp_path):
    monkeypatch.setattr(live_turn, "call_minutes_ceiling", lambda spec: 0.5 / 60)  # half a second, in minutes
    h = harness()
    h.start()
    text, usage, _trace = h.finish()
    assert usage["live_call"]["ended"] == "max_minutes" and "max_minutes" in text
    assert h.hub.closed == ["max_minutes"]


def test_an_empty_room_ends_the_call_after_the_grace(harness, monkeypatch, tmp_path):
    monkeypatch.setattr(live_turn, "PEER_GRACE_S", 0.3)
    h = harness()
    h.start()
    h.hub.present = []
    h.hub.feed(PeerLeft(peer_id="owner-1", reason="bye"))
    text, usage, _trace = h.finish()
    assert usage["live_call"]["ended"] == "peers_left"


def test_a_rejoin_cancels_the_empty_room_grace(harness, monkeypatch, tmp_path):
    monkeypatch.setattr(live_turn, "PEER_GRACE_S", 0.4)
    h = harness()
    h.start()
    h.hub.present = []
    h.hub.feed(PeerLeft(peer_id="owner-1"))
    time.sleep(0.1)
    h.hub.present = [OWNER]
    h.hub.feed(PeerJoined(peer=OWNER))
    time.sleep(0.6)
    assert h.thread.is_alive()
    h.stop()
    text, usage, _trace = h.finish()
    assert usage["live_call"]["ended"] == "owner_stop"


# --- reconnects -------------------------------------------------------------------


def test_a_lost_session_reopens_with_history_and_never_re_executes_a_tool(harness, monkeypatch, tmp_path):
    h = harness()
    first = h.start()
    first.push(Caption(speaker=SPEAKER_OWNER, text="look it up", final=True, item_id="u1"))
    first.push(ToolCall(call_id="c1", name="web_search", arguments="{}"))
    assert _wait_for(lambda: first.sent_of(ToolResult))
    first.push(ResponseStarted(item_id="a2", response_id="r2"))
    assert _wait_for(lambda: h.ledger.reserved)
    first.end_stream()
    assert _wait_for(lambda: len(h.llm.engines) == 2 and h.llm.engines[1].config is not None)
    second = h.engine()
    assert (STATE_RECONNECTING, "stream_ended") in h.hub.states and h.hub.states[-1] == (STATE_LIVE, "")
    assert h.ledger.abandoned == [("r2", "stream_ended")]
    history = second.config.history
    assert history[1] == {"role": "user", "content": "look it up"}
    assert history[2]["role"] == "assistant" and history[2]["tool_calls"][0]["id"] == "c1"
    assert history[3] == {"role": "tool", "content": "result of web_search", "tool_call_id": "c1", "name": "web_search"}
    assert second.config.resume_handle == ""
    second.push(ToolCall(call_id="c1", name="web_search", arguments="{}"))  # the provider asks again
    assert _wait_for(lambda: second.sent_of(ToolResult))
    assert second.sent_of(ToolResult)[0].output == "result of web_search" and h.batches == [["c1"]]
    first.push(AudioOut(pcm=b"\x05", rate=24000, item_id="stale"))  # the dead session's late event
    time.sleep(0.1)
    assert h.hub.played == []
    h.stop()
    text, usage, _trace = h.finish()
    assert usage["live_call"]["reconnects"] == 1 and h.task_row()["live_call"]["reconnects"] == 1


def test_go_away_resumes_with_the_providers_handle(harness, monkeypatch, tmp_path):
    monkeypatch.delenv("OUROBOROS_CALL_NATIVE_SEARCH", raising=False)
    h = harness(spec=_spec(engine=ENGINE_GOOGLE, model="gemini-3.8-live", voice="Kore"))
    first = h.start()
    assert first.config.native_search is True
    first.push(ResumeCheckpoint(handle="resume-7", resumable=True))
    first.push(GoAway(time_left_s=5))
    assert _wait_for(lambda: len(h.llm.engines) == 2 and h.llm.engines[1].config is not None)
    second = h.engine()
    assert second.config.resume_handle == "resume-7" and second.config.history == ()
    assert first.closed == ["go_away"]
    h.stop()
    h.finish()


def test_a_retryable_engine_error_reconnects_and_a_fatal_one_ends_the_call(harness):
    h = harness()
    first = h.start()
    first.push(EngineError(kind="rate_limited", message="slow down", retryable=True))
    assert _wait_for(lambda: len(h.llm.engines) == 2 and h.llm.engines[1].config is not None)
    second = h.engine()
    second.push(EngineError(kind="authentication", message="Live provider refused access.", retryable=False))
    text, usage, _trace = h.finish()
    assert usage["reason_code"] == "live_engine_error" and usage["live_call"]["ended"] == "authentication"
    assert "authentication: Live provider refused access." in text
    # The surface gets the one next step; the host's own description is never quoted as the provider's words.
    assert ("closed", "", "settings") in h.hub.words


def test_a_providers_own_sentence_reaches_the_surface_as_its_words(harness):
    h = harness()
    h.start().push(EngineError(kind="provider_closed", message="You exceeded your current quota.",
                               provider_words="You exceeded your current quota."))
    h.finish()
    assert ("closed", "You exceeded your current quota.", "retry") in h.hub.words


def test_a_session_that_cannot_be_reopened_ends_the_call(harness, monkeypatch, tmp_path):
    monkeypatch.setattr(live_turn, "RECONNECT_ATTEMPTS", 1)
    h = harness()
    first = h.start()
    h.llm.fail_opens = 1
    first.end_stream()
    text, usage, _trace = h.finish()
    assert usage["live_call"]["ended"] == "reconnect_failed" and usage["reason_code"] == "live_engine_unavailable"


def test_meeting_audio_in_the_mix_is_recorded_as_heard_never_as_the_owner(harness, monkeypatch, tmp_path):
    from supervisor import message_bus

    sent = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat_id, text, **kw: sent.append((chat_id, text, kw)))
    h = harness()
    engine = h.start()
    meeting = PeerInfo(peer_id="capture-1", role="meeting", mic=True, speaker=False, skill="meeting_audio")
    h.hub.feed(PeerJoined(meeting))
    time.sleep(0.05)
    engine.push(Caption(speaker=SPEAKER_OWNER, text="next quarter is fine", final=True, item_id="u1"))
    assert _wait_for(lambda: sent)
    chat_id, text, kw = sent[0]
    assert text == "Heard in the call: next quarter is fine" and kw["role"] == "system" and kw["narration"] is False
    assert kw["system_type"] == "call_heard" and kw["progress_meta"]["call_task_id"] == "t-call"
    assert h.chat_rows() == []  # no owner row while meeting audio is in the mix
    h.hub.feed(PeerLeft(peer_id="capture-1"))
    time.sleep(0.05)
    engine.push(Caption(speaker=SPEAKER_OWNER, text="thanks", final=True, item_id="u2"))
    assert _wait_for(lambda: len(h.chat_rows()) == 1) and h.chat_rows()[0]["text"] == "thanks"
    h.stop()
    h.finish()


def test_a_presence_call_records_both_sides_in_its_conversation_never_as_owner_rows(harness, monkeypatch, tmp_path):
    from ouroboros import presence_call

    recorded = []
    monkeypatch.setattr(presence_call, "record_call_utterance",
                        lambda root, task, **kw: recorded.append((task.get("id"), kw)) or True)
    task = {"id": "t-call", "metadata": {"presence": {"binding_id": "b" * 32}}}
    h = harness(spec=_spec(presence_binding_id="b" * 32, opened_by="skill"), task=task)
    engine = h.start()
    h.hub.feed(PeerControl(peer_id="zoom-1", role="meeting", message={"type": "participant", "label": "Elena", "speaking": True}))
    time.sleep(0.05)
    engine.push(Caption(speaker=SPEAKER_OWNER, text="hello Aika", final=True, item_id="u1"))
    assert _wait_for(lambda: recorded)
    assert recorded[0] == ("t-call", {"direction": "in", "text": "hello Aika", "speaker_label": "Elena"})
    for label in ("Elena", "Pavel"):  # two voices during one utterance: it stays unnamed
        h.hub.feed(PeerControl(peer_id="zoom-1", role="meeting", message={"type": "participant", "label": label, "speaking": True}))
    time.sleep(0.05)
    engine.push(Caption(speaker=SPEAKER_OWNER, text="and the budget", final=True, item_id="u2"))
    assert _wait_for(lambda: len(recorded) == 2) and recorded[1][1]["speaker_label"] == ""
    engine.push(Caption(speaker=SPEAKER_OUROBOROS, text="Happy to help.", final=True, item_id="a1"))
    h.hub.control("played", item_id="a1", played_ms=900)
    assert _wait_for(lambda: len(recorded) == 3)
    assert recorded[2] == ("t-call", {"direction": "out", "text": "Happy to help.", "speaker_label": ""})
    assert h.chat_rows() == []
    h.stop()
    h.finish()


# --- the owner's mailbox and the terminal ---------------------------------------------


def test_typed_mailbox_words_reach_the_session_and_are_acknowledged_after_the_checkpoint(harness, monkeypatch, tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    h = harness()
    engine = h.start()
    assert write_owner_message(tmp_path, "also check tomorrow", "t-call", msg_id="owner-typed-1")
    assert _wait_for(lambda: [c for c in engine.sent_of(ContextItem) if c.source == "mailbox"])
    item = [c for c in engine.sent_of(ContextItem) if c.source == "mailbox"][0]
    assert "also check tomorrow" in item.text
    assert _wait_for(lambda: "owner-typed-1" in (h.task_row().get("live_call") or {}).get("consumed_mailbox_ids", []))
    # Acknowledged after the checkpoint: the ACK follows the checkpoint write on the call's thread.
    assert _wait_for(lambda: list(drain_owner_entries(tmp_path, "t-call", set(), 1)) == [])
    h.stop()
    h.finish()


def test_the_terminal_is_a_host_notice_that_points_at_the_transcript(harness, monkeypatch, tmp_path):
    h = harness()
    engine = h.start()
    engine.push(Caption(speaker=SPEAKER_OWNER, text="hi", final=True, item_id="u1"))
    engine.push(Caption(speaker=SPEAKER_OUROBOROS, text="hello", final=True, item_id="a1"))
    assert _wait_for(lambda: len(h.chat_rows()) == 1)
    h.stop()
    text, usage, trace = h.finish()
    assert usage["terminal_origin"] == "host_notice"
    assert text.startswith("Live call Test call ended (owner_stop). The transcript is in this chat: 1 owner, 0 heard and 1 Ouroboros utterances")
    assert usage["live_call"]["owner"] == 1 and usage["live_call"]["ouroboros"] == 1 and usage["live_call"]["call_id"] == "call-1"
    assert trace["reasoning_notes"] == ["live_call"] and "execution_status" not in usage
    assert h.progress[-1][0] == "hello" and h.progress[-1][1]["_progress_meta_override"]["spoken"]["status"] == "generated"
    assert h.task_row()["live_call"]["utterances"] == {"owner": 1, "ouroboros": 1, "heard": 0}


def test_context_and_participant_controls_become_context_items(harness, monkeypatch, tmp_path):
    h = harness()
    engine = h.start()
    h.hub.control("context", text="slide 3 is on screen", source="screen")
    h.hub.feed(PeerControl(peer_id="meet-1", role="meeting", message={"type": "participant", "label": "Bob", "speaking": True}))
    assert _wait_for(lambda: len(engine.sent_of(ContextItem)) == 2)
    assert engine.sent_of(ContextItem)[0] == ContextItem(text="slide 3 is on screen", source="peer:owner:screen")
    assert engine.sent_of(ContextItem)[1] == ContextItem(text="Bob started speaking", source="meeting_participant")
    h.stop()
    h.finish()


@pytest.mark.parametrize("raw, expected", [("1", True), ("off", False), ("", True)])
def test_native_search_follows_the_setting_for_gemini_only(monkeypatch, raw, expected):
    from ouroboros.live_call_route import native_search_enabled

    monkeypatch.setenv("OUROBOROS_CALL_NATIVE_SEARCH", raw)
    assert native_search_enabled(_spec(engine=ENGINE_GOOGLE, model="gemini-3.8-live")) is expected


def test_the_call_view_opens_with_memory_the_web_and_the_room_chat():
    from ouroboros.live_call_route import CALL_INITIAL_TOOL_NAMES
    from ouroboros.tool_capabilities import COGNITIVE_MEMORY_TOOL_NAMES

    # Pointers in the call view are read_file addresses, and long work becomes a task, from the first open.
    assert set(CALL_INITIAL_TOOL_NAMES) == set(COGNITIVE_MEMORY_TOOL_NAMES) | {
        "web_search", "send_user_message", "read_file", "start_task"}


def test_a_call_nobody_attaches_to_ends_after_the_rejoin_grace(harness, monkeypatch, tmp_path):
    monkeypatch.setattr(live_turn, "PEER_GRACE_S", 0.2)
    h = harness()
    h.hub.present = []
    h.start()
    text, usage, _trace = h.finish()
    assert usage["live_call"]["ended"] == "peers_left"


def test_a_presence_call_opens_with_its_whole_ceiling_and_an_owner_call_with_the_nano_view():
    from ouroboros.live_call_route import initial_call_schemas

    registry = SimpleNamespace(schemas=lambda: [_schema("notes_read"), _schema("web_search"), _schema("enable_tools")])
    owner = [s["function"]["name"] for s in initial_call_schemas(registry, _spec())]
    presence = [s["function"]["name"] for s in initial_call_schemas(registry, _spec(presence_binding_id="b" * 32))]
    assert "notes_read" not in owner and {"web_search", "enable_tools"} <= set(owner)
    assert presence == ["notes_read", "web_search", "enable_tools"]
