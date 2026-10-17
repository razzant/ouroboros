"""The live voice-call driver across sessions: what a call heard, saw and was asked.

The round-trip cases of ``test_live_turn``'s harness that span a reconnect, a
connection still opening, a repeated provider request, a Presence call's rounds
and the route's instruction ceiling. The fakes and the ``harness`` fixture are
``test_live_turn``'s own.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from ouroboros import live_turn
from ouroboros.call_protocol import (
    STATE_CLOSING, STATE_CONNECTING, STATE_LIVE, PeerAudio, PeerInfo, PeerJoined, PeerLeft,
)
from ouroboros.gateways.realtime_engine import (
    ENGINE_GOOGLE, SPEAKER_OUROBOROS, SPEAKER_OWNER,
    AudioIn, AudioOut, Caption, ContextItem, GoAway, Interrupt, PlayedAudio,
    ResponseDone, ResponseStarted, SpeechStarted, ToolCall, ToolResult,
)
from ouroboros.usage_accounting import BudgetExceeded
from tests import test_live_turn as base
from tests.test_live_turn import OWNER, FakeEngine, _spec, _wait_for

harness = base.harness  # the same fixture, registered in this module


# --- what reaches the session: images, context the room heard, refused results ----------

PIXEL = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x04\x00\x00\x00\xb5\x1c\x0c\x02"
         b"\x00\x00\x00\x0bIDATx\xdac\xfc\xff\x1f\x00\x03\x03\x02\x00\xef\xa3\x08[\x00\x00\x00\x00IEND\xaeB`\x82")


def _image_row(note: str, pixels: bytes = PIXEL) -> dict:
    import base64

    return {"role": "user", "content": [{"type": "text", "text": note},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(pixels).decode()}}]}


def test_pixels_a_tool_viewed_ride_on_that_calls_result(harness):
    h = harness()

    def hook(call, messages, schemas):
        if call["id"] == "view-1":  # as vision.attach_local_image_to_context does, before the tool row
            messages.append(_image_row("[Image attached: pixel.png]"))
        messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                         "content": f"viewed by {call['id']}"})

    h.tool_hook = hook
    engine = h.start()
    engine.push(ToolCall("view-1", "web_search", "{}"))
    assert _wait_for(lambda: engine.sent_of(ToolResult))
    result, = engine.sent_of(ToolResult)
    assert result.call_id == "view-1" and result.images == (PIXEL,)
    # The image row's caption travels as context; the pixels do not travel twice.
    assert [c.text for c in engine.sent_of(ContextItem) if c.source == "tool_batch:user"] == ["[Image attached: pixel.png]"]
    engine.push(ToolCall("plain-2", "web_search", "{}"))
    assert _wait_for(lambda: len(engine.sent_of(ToolResult)) == 2)
    assert engine.sent_of(ToolResult)[1].images == ()
    # A reopened session gets the image on the user row of its history view, never dropped.
    engine.end_stream()
    assert _wait_for(lambda: len(h.llm.engines) == 2 and h.engine().config is not None)
    history = h.engine().config.history
    image_rows = [row for row in history if isinstance(row["content"], list)]
    assert len(image_rows) == 1 and image_rows[0]["content"][1]["type"] == "image_url"
    h.stop()
    h.finish()


def test_pixels_with_no_result_to_carry_them_are_a_named_omission(harness):
    h = harness()

    def hook(call, messages, schemas):
        messages.append(_image_row("[Image attached: orphan.png]"))  # the tool never wrote its own row

    h.tool_hook = hook
    engine = h.start()
    engine.push(ToolCall("view-orphan", "web_search", "{}"))
    assert _wait_for(lambda: any(c.source == "tool_batch:images_omitted" for c in engine.sent_of(ContextItem)))
    omitted, = [c for c in engine.sent_of(ContextItem) if c.source == "tool_batch:images_omitted"]
    assert "1 image(s)" in omitted.text and engine.sent_of(ToolResult) == []
    h.stop()
    h.finish()


def test_a_refused_tool_result_becomes_a_typed_failure_the_model_hears(harness, monkeypatch):
    from ouroboros.gateways.realtime_transport import LiveEngineRefusal

    h = harness()
    engine = h.start()
    original = FakeEngine.send

    async def refusing_send(self, command):
        if isinstance(command, ToolResult) and command.output.startswith("huge"):
            raise LiveEngineRefusal("wire_size", "Message of 9000000 bytes exceeds the 8388608-byte frame limit.")
        await original(self, command)

    monkeypatch.setattr(FakeEngine, "send", refusing_send)

    def hook(call, messages, schemas):
        messages.append(_image_row("[Image attached: big.png]"))
        messages.append({"role": "tool", "tool_call_id": call["id"], "name": "web_search", "content": "huge result"})

    h.tool_hook = hook
    engine.push(ToolCall("big-1", "web_search", "{}"))
    assert _wait_for(lambda: engine.sent_of(ToolResult))
    notice, = engine.sent_of(ToolResult)
    assert notice.call_id == "big-1" and notice.images == ()
    assert notice.output.startswith("⚠️ TOOL_RESULT_REFUSED (wire_size)")
    assert "11-character text and 1 image(s)" in notice.output and "frame limit" in notice.output
    assert h.thread.is_alive()  # the session continues
    h.stop()
    h.finish()


def test_a_refused_result_for_an_unknown_call_travels_as_context(harness, monkeypatch):
    from ouroboros.gateways.realtime_transport import LiveEngineRefusal

    h = harness()
    engine = h.start()
    original = FakeEngine.send

    async def refusing_send(self, command):
        if isinstance(command, ToolResult):
            raise LiveEngineRefusal("unknown_call", "No pending function call with this id.")
        await original(self, command)

    monkeypatch.setattr(FakeEngine, "send", refusing_send)
    engine.push(ToolCall("gone-1", "web_search", "{}"))
    assert _wait_for(lambda: any(c.source == "tool_result_refused" for c in engine.sent_of(ContextItem)))
    assert "TOOL_RESULT_REFUSED (unknown_call)" in [c for c in engine.sent_of(ContextItem) if c.source == "tool_result_refused"][0].text
    h.stop()
    h.finish()


def test_what_the_room_said_is_part_of_the_transcript_a_reopened_session_replays(harness, tmp_path):
    from ouroboros.live_awareness import notify_chat_row
    from ouroboros.utils import utc_now_iso

    h = harness()
    engine = h.start()
    phrase = "The meeting moved to Thursday at 14:00."
    notify_chat_row(tmp_path, {"chat_id": 1, "direction": "in", "text": phrase, "ts": utc_now_iso(),
                               "client_message_id": "typed-during-call", "source": "web"})
    assert _wait_for(lambda: any(phrase in c.text for c in engine.sent_of(ContextItem)))
    h.hub.control("context", text="slide 3 is on screen", source="screen")
    assert _wait_for(lambda: any(c.text == "slide 3 is on screen" for c in engine.sent_of(ContextItem)))
    engine.push(GoAway(time_left_s=30))
    assert _wait_for(lambda: len(h.llm.engines) == 2 and h.engine().config is not None)
    history = h.engine().config.history
    assert any(phrase in str(row["content"]) for row in history)
    assert any(row["content"] == "slide 3 is on screen" for row in history)
    # Replayed in history, delivered once: the second session is not sent them again as items.
    time.sleep(0.3)
    assert not any(phrase in c.text for c in h.engine().sent_of(ContextItem))
    rows = [m for m in h.messages if m.get("_context_source")]
    assert {m["_context_source"] for m in rows} >= {"peer:owner:screen"} and all(m["_call_row"] for m in rows)
    h.stop()
    h.finish()


def test_a_room_row_between_context_capture_and_the_listener_is_delivered(harness, tmp_path):
    from supervisor.message_bus import log_chat

    h = harness()  # the turn's context is captured here; the driver has not started
    phrase = "The owner added this while the call context was being prepared."
    assert log_chat("in", 1, 1, phrase, drive_root=tmp_path, source="web") is not None
    time.sleep(0.02)
    engine = h.start()
    assert _wait_for(lambda: any(phrase in c.text for c in engine.sent_of(ContextItem)))
    h.stop()
    h.finish()


def test_the_feed_does_not_catch_up_from_before_the_context_was_captured(harness, tmp_path):
    from supervisor.message_bus import log_chat
    from ouroboros.utils import utc_now_iso

    assert log_chat("in", 1, 1, "already in the captured context", drive_root=tmp_path, source="web") is not None
    time.sleep(0.02)
    h = harness()
    h.context_captured_at = utc_now_iso()
    time.sleep(0.02)
    engine = h.start()
    assert _wait_for(lambda: (STATE_LIVE, "") in h.hub.states)
    time.sleep(0.4)
    assert not any("already in the captured context" in c.text for c in engine.sent_of(ContextItem))
    h.stop()
    h.finish()


# --- while the connection is pending ---------------------------------------------------


def _slow_opening(monkeypatch):
    original = FakeEngine.open

    async def slow_open(self, config):
        await original(self, config)
        self.release_open = asyncio.Event()
        await self.release_open.wait()

    monkeypatch.setattr(FakeEngine, "open", slow_open)


def test_stop_during_the_connection_cancels_the_opening_and_closes_what_was_opened(harness, monkeypatch):
    _slow_opening(monkeypatch)
    h = harness()
    engine = h.start()
    assert h.hub.states == [(STATE_CONNECTING, "")]
    h.stop()
    # The opening never completes on its own, so a closed engine means the Stop cancelled it while the
    # provider connection was still pending (counted by outcome, not by the clock of a loaded machine).
    assert _wait_for(lambda: engine.closed and (STATE_CLOSING, "owner_stop") in h.hub.states, timeout=15.0)
    text, usage, _trace = h.finish()
    assert usage["live_call"]["ended"] == "owner_stop" and "execution_status" not in usage
    assert engine.closed == ["cancelled"] and engine.sent == []  # no session is paid for
    assert (STATE_LIVE, "") not in h.hub.states and h.hub.closed == ["owner_stop"]
    assert h.ledger.reserved == [] and h.ledger.abandoned == []


def test_a_stop_beside_typed_words_closes_without_waiting_for_their_send(harness, monkeypatch, tmp_path):
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

    original = FakeEngine.send

    async def held_send(self, command):  # a transport waiting to be ready again holds this send
        if isinstance(command, ContextItem) and command.source == "mailbox":
            await asyncio.Event().wait()
        await original(self, command)

    monkeypatch.setattr(FakeEngine, "send", held_send)
    assert write_owner_message(tmp_path, "one more thing", "t-call", msg_id="owner-typed-2")
    assert write_owner_message(tmp_path, "stop", "t-call", msg_id="owner-stop", kind=KIND_FINALIZE_NOW)
    h = harness()
    engine = h.start()
    text, usage, _trace = h.finish(timeout=10.0)
    assert usage["live_call"]["ended"] == "owner_stop"
    assert not [c for c in engine.sent_of(ContextItem) if c.source == "mailbox"]


def test_a_stop_after_a_send_already_waiting_on_the_transport_closes_the_call(harness, monkeypatch, tmp_path):
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

    original, sending = FakeEngine.send, threading.Event()

    async def held_send(self, command):  # a transport waiting to be ready again holds this send
        if isinstance(command, ContextItem) and command.source == "mailbox":
            sending.set()
            await asyncio.Event().wait()
        await original(self, command)

    monkeypatch.setattr(FakeEngine, "send", held_send)
    h = harness()
    engine = h.start()
    assert write_owner_message(tmp_path, "one more thing", "t-call", msg_id="owner-typed-3")
    assert sending.wait(5)  # that send has begun and waits
    assert write_owner_message(tmp_path, "stop", "t-call", msg_id="owner-stop", kind=KIND_FINALIZE_NOW)
    text, usage, _trace = h.finish()  # a Stop held behind the send would never finish
    assert usage["live_call"]["ended"] == "owner_stop" and engine.closed


def test_a_delivery_boundary_acknowledges_only_the_mail_sent_before_it(harness, monkeypatch, tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    original, gates, started = FakeEngine.send, {}, {}

    async def held_send(self, command):  # each mailbox send waits until the test releases it
        if isinstance(command, ContextItem) and command.source == "mailbox":
            word = "first" if "first" in command.text else "second"
            started[word] = True
            gate = gates.setdefault(word, asyncio.Event())
            await gate.wait()
        await original(self, command)

    def unacknowledged():
        return {entry.get("msg_id") or entry.get("id") for entry in drain_owner_entries(tmp_path, "t-call", set(), 1)}

    monkeypatch.setattr(FakeEngine, "send", held_send)
    h = harness()
    engine = h.start()
    assert write_owner_message(tmp_path, "the first words", "t-call", msg_id="mail-first")
    assert _wait_for(lambda: started.get("first"))
    assert write_owner_message(tmp_path, "the second words", "t-call", msg_id="mail-second")
    assert _wait_for(lambda: "mail-second" in h.task_row().get("live_call", {}).get("consumed_mailbox_ids", [])
                     or any("second" in str(m.get("content")) for m in h.messages))
    loop = engine._loop
    loop.call_soon_threadsafe(gates.setdefault("first", asyncio.Event()).set)
    assert _wait_for(lambda: started.get("second"))  # the first is delivered, the second is waiting
    assert _wait_for(lambda: "mail-first" not in unacknowledged())
    assert "mail-second" in unacknowledged()  # its own boundary has not been reached
    loop.call_soon_threadsafe(gates.setdefault("second", asyncio.Event()).set)
    assert _wait_for(lambda: "mail-second" not in unacknowledged())
    h.stop()
    h.finish()

def test_joins_leaves_and_context_while_connecting_wait_for_the_live_session(harness, monkeypatch):
    _slow_opening(monkeypatch)
    h = harness()
    engine = h.start()
    h.hub.present = []
    h.hub.feed(PeerLeft(peer_id="owner-1"))
    time.sleep(0.05)
    h.hub.present = [OWNER]
    h.hub.feed(PeerJoined(peer=OWNER))
    h.hub.control("context", text="seen before the session opened", source="screen")
    h.hub.feed(PeerAudio(peer_id="owner-1", role="owner", pcm=b"\x01" * 20))  # transient: not queued
    assert _wait_for(lambda: any(m.get("_context_source") == "peer:owner:screen" for m in h.messages))
    assert engine.sent == [] and engine.config.history == (
        {"role": "user", "content": "[Live call] A live voice call is open in this room."},)
    engine._loop.call_soon_threadsafe(engine.release_open.set)
    assert _wait_for(lambda: engine.sent_of(ContextItem))
    assert engine.sent_of(ContextItem) == [ContextItem(text="seen before the session opened", source="peer:owner:screen")]
    assert engine.sent_of(AudioIn) == [] and (STATE_LIVE, "") in h.hub.states
    h.stop()
    h.finish()


def test_typed_words_while_connecting_are_queued_and_acknowledged_only_once_live(harness, monkeypatch, tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    _slow_opening(monkeypatch)
    h = harness()
    engine = h.start()
    assert write_owner_message(tmp_path, "typed while connecting", "t-call", msg_id="owner-early-1")
    assert _wait_for(lambda: any("typed while connecting" in str(m.get("content")) for m in h.messages))
    time.sleep(0.3)
    assert engine.sent == []
    assert list(drain_owner_entries(tmp_path, "t-call", set(), 1)) != []  # nothing acknowledged while no engine is live
    engine._loop.call_soon_threadsafe(engine.release_open.set)
    assert _wait_for(lambda: [c for c in engine.sent_of(ContextItem) if c.source == "mailbox"])
    assert "typed while connecting" in engine.sent_of(ContextItem)[0].text
    assert _wait_for(lambda: list(drain_owner_entries(tmp_path, "t-call", set(), 1)) == [])
    h.stop()
    h.finish()


# --- provider call ids, rounds and who spoke ----------------------------------------------


def test_a_repeated_request_for_a_running_call_joins_it_and_runs_the_effect_once(harness):
    entered, release = threading.Event(), threading.Event()
    h = harness()

    def effect(call, messages, schemas):
        entered.set()
        assert release.wait(5)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "effect complete"})

    h.tool_hook = effect
    first = h.start()
    first.push(ToolCall("one-physical-effect", "web_search", "{}"))
    assert entered.wait(3)
    first.push(GoAway(time_left_s=30))
    assert _wait_for(lambda: len(h.llm.engines) == 2 and h.engine().config is not None)
    second = h.engine()
    second.push(ToolCall("one-physical-effect", "web_search", "{}"))  # the new session asks again
    time.sleep(0.15)
    release.set()
    assert _wait_for(lambda: second.sent_of(ToolResult))
    assert second.sent_of(ToolResult)[0].output == "effect complete" and first.sent_of(ToolResult) == []
    time.sleep(0.2)
    assert h.batches == [["one-physical-effect"]]
    h.stop()
    h.finish()


def test_a_presence_calls_autonomous_rounds_are_capped_per_input_and_reset_by_speech(harness):
    h = harness(spec=_spec(presence_binding_id="b" * 32, opened_by="skill"))
    h.ctx.inline_max_rounds = 1
    engine = h.start()
    engine.push(ToolCall("round-1", "web_search", "{}"))
    assert _wait_for(lambda: len(engine.sent_of(ToolResult)) == 1)
    engine.push(ToolCall("round-2", "web_search", "{}"))  # no participant input in between
    assert _wait_for(lambda: len(engine.sent_of(ToolResult)) == 2)
    refused = engine.sent_of(ToolResult)[1]
    assert refused.call_id == "round-2" and refused.output.startswith("⚠️ TOOL_ROUND_LIMIT")
    assert "1 autonomous tool round(s)" in refused.output and h.batches == [["round-1"]]
    assert h.thread.is_alive() and (STATE_CLOSING, "") not in h.hub.states  # never a session close
    engine.push(Caption(speaker=SPEAKER_OWNER, text="and the budget?", final=True, item_id="u1"))
    assert _wait_for(lambda: h.messages[-1].get("content") == "[Heard in the call] and the budget?")
    engine.push(ToolCall("round-3", "web_search", "{}"))
    assert _wait_for(lambda: len(h.batches) == 2)
    assert h.batches == [["round-1"], ["round-3"]]
    h.stop()
    h.finish()


def test_the_calls_of_one_response_are_one_presence_round_however_they_arrive(harness):
    h = harness(spec=_spec(presence_binding_id="b" * 32, opened_by="skill"))
    h.ctx.inline_max_rounds = 1
    engine = h.start()
    engine.push(ResponseStarted(item_id="i1", response_id="one-response"))
    engine.push(ToolCall("first", "web_search", "{}"))
    assert _wait_for(lambda: len(engine.sent_of(ToolResult)) == 1)
    engine.push(ToolCall("second", "web_search", "{}"))  # the same response, a later batch
    assert _wait_for(lambda: len(engine.sent_of(ToolResult)) == 2)
    assert h.batches == [["first"], ["second"]]
    assert not any(r.output.startswith("⚠️ TOOL_ROUND_LIMIT") for r in engine.sent_of(ToolResult))
    engine.push(ResponseDone(item_id="i1", status="completed", response_id="one-response"))
    engine.push(ResponseStarted(item_id="i2", response_id="next-response"))  # a further autonomous round
    engine.push(ToolCall("third", "web_search", "{}"))
    assert _wait_for(lambda: len(engine.sent_of(ToolResult)) == 3)
    assert engine.sent_of(ToolResult)[2].output.startswith("⚠️ TOOL_ROUND_LIMIT")
    h.stop()
    h.finish()

def test_an_owner_call_has_no_round_ceiling(harness):
    h = harness()
    h.ctx.inline_max_rounds = 1
    engine = h.start()
    for call_id in ("r1", "r2", "r3"):
        engine.push(ToolCall(call_id, "web_search", "{}"))
    assert _wait_for(lambda: len(engine.sent_of(ToolResult)) == 3)
    assert not any(r.output.startswith("⚠️ TOOL_ROUND_LIMIT") for r in engine.sent_of(ToolResult))
    h.stop()
    h.finish()


def test_meeting_audio_already_sent_makes_the_delayed_caption_heard_not_owner_words(harness):
    h = harness()
    engine = h.start()
    h.hub.feed(PeerJoined(PeerInfo("meeting-audio", "meeting", True, False)))
    h.hub.feed(PeerAudio("meeting-audio", "meeting", b"\x01\x00" * 240))
    assert _wait_for(lambda: engine.sent_of(AudioIn))
    h.hub.feed(PeerLeft("meeting-audio", "listen_stop"))
    h.hub.control("context", text="after-leave-marker")
    assert _wait_for(lambda: any(c.text == "after-leave-marker" for c in engine.sent_of(ContextItem)))
    engine.push(Caption(SPEAKER_OWNER, "words the room said before the capture stopped", True, "delayed"))
    assert _wait_for(lambda: any("words the room said" in str(m.get("content")) for m in h.messages))
    row, = [m for m in h.messages if "words the room said" in str(m.get("content"))]
    assert row["content"].startswith("[Heard in the call]") and h.chat_rows() == []
    # The audio that fed the engine is spent: the owner's next words are the owner's.
    engine.push(Caption(SPEAKER_OWNER, "thanks", True, "u2"))
    assert _wait_for(lambda: len(h.chat_rows()) == 1) and h.chat_rows()[0]["text"] == "thanks"
    h.stop()
    h.finish()


def test_an_input_items_speaker_follows_the_audio_that_formed_it_however_late_the_events(harness):
    h = harness()
    engine = h.start()
    second = b"\x01\x00" * 16000  # one second at Gemini Live's 16 kHz input
    h.hub.feed(PeerJoined(PeerInfo("meeting-audio", "meeting", True, False)))
    h.hub.feed(PeerAudio("meeting-audio", "meeting", second))
    h.hub.feed(PeerAudio("meeting-audio", "meeting", second))  # 0-2000 ms: the meeting
    assert _wait_for(lambda: len(engine.sent_of(AudioIn)) == 2)
    h.hub.feed(PeerLeft("meeting-audio", "listen_stop"))
    h.hub.feed(PeerAudio("owner-1", "owner", second))  # 2000-3000 ms: the owner alone
    assert _wait_for(lambda: len(engine.sent_of(AudioIn)) == 3)
    # The provider reports both meeting inputs and the owner's only now, after the meeting left.
    engine.push(SpeechStarted("m1", 0))
    engine.push(SpeechStarted("m2", 1000))
    engine.push(SpeechStarted("o1", 2100))
    for item, words in (("m2", "meeting words two"), ("o1", "owner words"), ("m1", "meeting words one")):
        engine.push(Caption(SPEAKER_OWNER, words, True, item))  # the captions land late, out of order
    assert _wait_for(lambda: len(h.chat_rows()) == 1 and h.counts_heard() == 2)
    assert h.chat_rows()[0]["text"] == "owner words"
    heard = [m["content"] for m in h.messages if str(m.get("content")).startswith("[Heard in the call]")]
    assert heard == ["[Heard in the call] meeting words two", "[Heard in the call] meeting words one"]
    h.stop()
    h.finish()

def test_barge_in_leaves_a_finished_fully_played_item_alone(harness):
    h = harness()
    engine = h.start()
    engine.push(AudioOut(bytes(48000), 24000, "done"))        # one second, finished and heard whole
    engine.push(Caption(SPEAKER_OUROBOROS, "A finished sentence.", True, "done"))
    engine.push(AudioOut(bytes(48000), 24000, "live"))        # the next item, still playing
    engine.push(Caption(SPEAKER_OUROBOROS, "", False, "live"))
    h.hub.control("played", item_id="done", played_ms=1000)
    h.hub.control("played", item_id="live", played_ms=200)
    assert _wait_for(lambda: len(engine.sent_of(PlayedAudio)) == 2)
    assert _wait_for(lambda: len(h.progress) == 1)
    assert h.progress[0][1]["_progress_meta_override"]["spoken"]["status"] == "played"
    engine.push(SpeechStarted())
    assert _wait_for(lambda: engine.sent_of(Interrupt))
    assert engine.sent_of(Interrupt) == [Interrupt("live", 200)]
    engine.push(SpeechStarted())  # the owner's next turn: nothing finished is cut again
    time.sleep(0.1)
    assert engine.sent_of(Interrupt) == [Interrupt("live", 200)]
    h.stop()
    h.finish()


def test_barge_in_still_cuts_an_item_whose_playback_caught_up_while_it_generates(harness):
    h = harness()
    engine = h.start()
    engine.push(AudioOut(bytes(4800), 24000, "gen"))  # 100 ms delivered so far, no final caption yet
    h.hub.control("played", item_id="gen", played_ms=100)
    assert _wait_for(lambda: engine.sent_of(PlayedAudio))
    engine.push(SpeechStarted())
    assert _wait_for(lambda: engine.sent_of(Interrupt))
    assert engine.sent_of(Interrupt) == [Interrupt("gen", 100)]
    engine.push(AudioOut(bytes(4800), 24000, "gen"))  # late frames of the cut item are not played
    time.sleep(0.1)
    assert len(h.hub.played) == 1
    h.stop()
    h.finish()


# --- the ledger: unknown stays unknown, refusals keep the charge, no fetch on the loop ------


def test_the_quote_is_warmed_on_the_ledger_thread_and_never_holds_the_open(harness):
    h = harness()
    release = threading.Event()
    warm = h.ledger.warm_quote

    def slow_warm(**fields):
        release.wait(10)  # a catalog fetch that has not answered yet
        return warm(**fields)

    h.ledger.warm_quote = slow_warm
    h.start()  # the engine opens while the quote is still being fetched
    assert not h.ledger.warmed
    release.set()
    assert _wait_for(lambda: h.ledger.warmed)
    (fields, thread_name), = h.ledger.warmed
    assert fields == {"engine": ENGINE_GOOGLE, "model": "gemini-3.8-live", "drive_root": h.root}
    assert thread_name.startswith("call-ledger")
    h.stop()
    h.finish()


def test_a_budget_refused_response_the_provider_billed_keeps_its_charge(harness):
    h = harness()
    h.ledger.refuse = True
    engine = h.start()
    engine.push(ResponseDone(item_id="a1", status="completed", usage={"audio_output_tokens": 20},
                             raw_usage={"output_tokens": 20}, response_id="r1"))
    h.thread.join(15)
    assert not h.thread.is_alive() and isinstance(h.error, BudgetExceeded)
    refused, = h.ledger.refused
    assert refused["response_key"] == "r1" and refused["status"] == "completed"
    assert refused["usage"] == {"audio_output_tokens": 20} and refused["raw_usage"] == {"output_tokens": 20}
    assert refused["reason"] == "the call's budget is spent" and refused["call_id"] == "call-1"


# --- the route's known instruction ceiling shapes the first open ------------------------------


def _ladder(h, full, addressed, nano=None):
    from ouroboros.context_fit import ContextFitPlan, ContextFitProjection

    projection = lambda mode, text: ContextFitProjection(mode, json.dumps(text), len(text) // 4, len(text) // 4, 1)
    h.ctx.context_fit_plan = ContextFitPlan("capture", "max", "max", "model", "openai", "route", "asserted",
        False, 128000, 32000, json.dumps("original input"), projection("max", full), projection("low", addressed),
        nano_projection=projection("nano", nano if nano is not None else addressed))
    h.messages[0]["content"] = full


def test_a_slow_ledger_write_never_holds_the_audio_and_settles_after_its_reservation(harness):
    h = harness()
    engine = h.start()
    gate = threading.Event()
    reserve = h.ledger.reserve_live_response

    def slow_reserve(**fields):  # a store busy with another writer
        gate.wait(5)
        return reserve(**fields)

    h.ledger.reserve_live_response = slow_reserve
    engine.push(ResponseStarted(item_id="a1", response_id="r1"))
    engine.push(AudioOut(bytes(4800), 24000, "a1"))
    engine.push(ResponseDone(item_id="a1", status="completed", usage={"audio_output_tokens": 3}, response_id="r1"))
    assert _wait_for(lambda: len(h.hub.played) == 1, timeout=2.0)  # the audio flows while the write waits
    assert h.ledger.reserved == [] and h.ledger.settled == []
    gate.set()
    assert _wait_for(lambda: h.ledger.settled == [("r1", "completed", {"audio_output_tokens": 3})])
    assert [attempt["key"] for attempt in h.ledger.reserved] == ["r1"]
    h.stop()
    h.finish()


def test_ledger_writes_run_inside_the_turns_bound_context(harness, monkeypatch):
    import contextvars

    admitted = contextvars.ContextVar("admitted_scope", default=None)  # what the agent binds on the turn thread
    original = live_turn.run_live_turn

    def bound(**kwargs):
        admitted.set("the task's root and group limits")
        return original(**kwargs)

    monkeypatch.setattr(live_turn, "run_live_turn", bound)
    h = harness()
    seen: list = []
    for name in ("reserve_live_response", "settle_live_response"):
        def reading(*args, _write=getattr(h.ledger, name), _name=name, **fields):
            seen.append((_name, admitted.get()))
            return _write(*args, **fields)
        setattr(h.ledger, name, reading)
    engine = h.start()
    engine.push(ResponseStarted(item_id="a1", response_id="r1"))
    engine.push(ResponseDone(item_id="a1", status="completed", usage={"audio_output_tokens": 3}, response_id="r1"))
    assert _wait_for(lambda: len(seen) == 2)
    assert sorted(seen) == [(name, "the task's root and group limits")
                            for name in ("reserve_live_response", "settle_live_response")]
    h.stop()
    h.finish()


def test_a_meeting_only_call_counts_real_time_playback_at_a_barge_in(harness):
    h = harness(spec=_spec(presence_binding_id="b" * 32, opened_by="skill"))
    engine = h.start()
    engine.push(AudioOut(bytes(4800), 24000, "done"))  # 100 ms, finished and long since played
    engine.push(Caption(SPEAKER_OUROBOROS, "A short answer.", True, "done"))
    engine.push(AudioOut(bytes(48000 * 4), 24000, "long"))  # 4 s delivered, still playing
    assert _wait_for(lambda: len(h.hub.played) == 2)
    time.sleep(0.3)  # no peer reports playback; the audio plays in real time meanwhile
    engine.push(SpeechStarted())
    assert _wait_for(lambda: engine.sent_of(Interrupt))
    (cut,) = engine.sent_of(Interrupt)  # the finished answer is left alone
    assert cut.item_id == "long" and 250 <= cut.played_ms < 4000  # cut where it is, not at zero
    h.stop()
    h.finish()


def test_later_meeting_captions_landing_first_leave_the_earlier_owner_words_the_owners(harness):
    h = harness()
    engine = h.start()
    second = b"\x01\x00" * 24000
    h.hub.feed(PeerAudio("owner-1", "owner", second))  # 0-1000 ms: the owner, before any meeting
    assert _wait_for(lambda: len(engine.sent_of(AudioIn)) == 1)
    h.hub.feed(PeerJoined(PeerInfo("meeting-audio", "meeting", True, False)))
    h.hub.feed(PeerAudio("meeting-audio", "meeting", second))
    h.hub.feed(PeerAudio("meeting-audio", "meeting", second))  # 1000-3000 ms: the meeting
    assert _wait_for(lambda: len(engine.sent_of(AudioIn)) == 3)
    for item, start in (("o1", 0), ("m1", 1000), ("m2", 2000)):
        engine.push(SpeechStarted(item, start))
    for item, words in (("m1", "meeting words one"), ("m2", "meeting words two"), ("o1", "owner words")):
        engine.push(Caption(SPEAKER_OWNER, words, True, item))  # the owner's caption lands last
    assert _wait_for(lambda: len(h.chat_rows()) == 1 and h.counts_heard() == 2)
    assert h.chat_rows()[0]["text"] == "owner words"
    h.stop()
    h.finish()


@pytest.mark.parametrize("asked_epoch,delivered", [(2, ToolResult), (1, ContextItem)])
def test_a_round_limit_answer_goes_to_the_session_that_asked_or_stays_as_context(asked_epoch, delivered):
    from types import SimpleNamespace

    from ouroboros.live_tool_batch import ToolBatchWorker

    sent: list = []

    async def send(command):
        sent.append(command)

    call = SimpleNamespace(spec=SimpleNamespace(presence_binding_id="b" * 32, call_id="c"), task_id="t",
                           ctx=SimpleNamespace(inline_max_rounds=1), epoch=2, _send=send, _checkpoint=lambda: True)
    lane = ToolBatchWorker(call)
    lane.round_origins = {"earlier-response"}  # this input's one allowed round is spent
    lane.claimed = {"late": asked_epoch}
    lane.pending = [(asked_epoch, ToolCall("late", "web_search", "{}"), "next-response")]

    async def one_take():
        task = asyncio.create_task(lane.run_forever())
        while not sent:
            await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(asyncio.wait_for(one_take(), 5))
    lane.shutdown()
    (answer,) = sent
    assert isinstance(answer, delivered)
    text = answer.output if delivered is ToolResult else answer.text
    assert "TOOL_ROUND_LIMIT" in text
    if delivered is ContextItem:
        assert answer.source == "tool_result_retained"
