"""R2 E8: fake-wire replay of the owner probe's cancellation race.

The supplied cancel-race.json lines 1472-1482 order speech_started, client
cancel/truncate, response.done(cancelled/turn_detected), then an uncorrelated
response_cancel_not_active error. IDs and spoken text here are synthetic.
"""
import asyncio
import json

import pytest

from ouroboros.contracts.live_conversation import Interrupt, LiveOptions
from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.realtime_session import JsonSession
from tests.test_grok_openai_protocol import (
    Wire, deny_network as deny_network, make, providers as providers, selected,
)


USAGE = {
    "total_tokens": 34419, "input_tokens": 33904, "output_tokens": 515,
    "input_token_details": {"text_tokens": 33794, "audio_tokens": 110,
        "image_tokens": 0, "cached_tokens": 33152,
        "cached_tokens_details": {"text_tokens": 33152, "audio_tokens": 0, "image_tokens": 0}},
    "output_token_details": {"text_tokens": 125, "audio_tokens": 390, "reasoning_tokens": 26},
}
SPEECH = {"type": "input_audio_buffer.speech_started", "event_id": "server-speech",
          "audio_start_ms": 18292, "item_id": "owner-next"}
DONE = {"type": "response.done", "event_id": "server-done", "response": {
    "id": "reply", "status": "cancelled",
    "status_details": {"type": "cancelled", "reason": "turn_detected"},
    "output": [{"id": "spoken", "type": "message", "role": "assistant", "status": "incomplete",
                "content": [{"type": "output_audio", "transcript": "Partial reply"}]}],
    "usage": USAGE,
}}
ERROR = {"type": "error", "event_id": "server-error", "error": {
    "type": "invalid_request_error", "code": "response_cancel_not_active",
    "message": "Cancellation failed: no active response found", "param": None, "event_id": None,
}}


class CancelWire(Wire):
    """Hold the truncate ACK so response.done can precede it as in the probe."""
    async def send(self, raw):
        if json.loads(raw)["type"] == "conversation.item.truncate":
            self.sent.append(json.loads(raw))
        else:
            await super().send(raw)


async def receive(adapter, wire, event):
    wire.put(event)
    return await adapter.receive()


async def start(adapter, wire):
    await adapter.open()
    assert (await adapter.receive())[0].kind == "ready"
    await receive(adapter, wire, {"type": "response.created", "response": {"id": "reply"}})
    events = await receive(adapter, wire, {"type": "response.output_audio.delta",
        "response_id": "reply", "item_id": "spoken", "delta": "AAA="})
    assert events[0].kind == "audio"


async def finish_barge_in(adapter, wire, engine):
    # A late chunk of the cancelled response must not restart playback.
    assert await receive(adapter, wire, {"type": "response.output_audio.delta",
        "response_id": "reply", "item_id": "spoken", "delta": "AAA="}) == []
    assert await receive(adapter, wire, {"type": "conversation.item.truncated",
        "item_id": "spoken", "audio_end_ms": 3820}) == []
    caption = await receive(adapter, wire, {"type": "response.output_audio_transcript.done",
        "item_id": "spoken", "transcript": "Partial reply"})
    assert caption[0].data["text"] == "Partial reply" and caption[0].data["final"]
    await receive(adapter, wire, {"type": "input_audio_buffer.speech_stopped", "item_id": "owner-next"})
    await receive(adapter, wire, {"type": "input_audio_buffer.committed", "item_id": "owner-next"})
    caption = await receive(adapter, wire, {"type": "conversation.item.input_audio_transcription.completed",
        "item_id": "owner-next", "status": "completed", "transcript": "Please continue."})
    assert caption[0].data["speaker"] == "owner" and caption[0].data["text"] == "Please continue."
    if engine == "openai":
        assert sum(e["type"] == "response.create" for e in wire.sent) == 1
    await receive(adapter, wire, {"type": "response.created", "response": {"id": "next-reply"}})
    audio = await receive(adapter, wire, {"type": "response.output_audio.delta",
        "response_id": "next-reply", "item_id": "next-spoken", "delta": "AAA="})
    assert audio[0].kind == "audio" and audio[0].data["pcm"] == b"\0\0"
    await receive(adapter, wire, {"type": "response.output_audio_transcript.done",
        "item_id": "next-spoken", "transcript": "Continuing."})
    done = await receive(adapter, wire, {"type": "response.done",
        "response": {"id": "next-reply", "status": "completed"}})
    assert [e.kind for e in done] == ["response_done"]
    assert done[0].data["status"] == "completed"
    assert adapter.renewal_ready and not wire.closed


@pytest.mark.parametrize("engine", ["openai", "grok"])
@pytest.mark.parametrize("interruptions", [True, False])
def test_vad_barge_in_skips_redundant_cancel_but_keeps_truncation(providers, engine, interruptions):
    async def run():
        adapter, wire = make(providers, engine, LiveOptions(interruptions=interruptions), wire=CancelWire(engine))
        try:
            await start(adapter, wire)
            events = await receive(adapter, wire, SPEECH)
            auto_cancel = interruptions or engine == "grok"  # Grok's VAD cannot disable interruptions.
            assert any(e.kind == "speech_started" for e in events) is auto_cancel
            # Same driver order as the probe; also cover an explicit stop when VAD interruption is off.
            await adapter.send(Interrupt())
            await adapter.send(Interrupt("spoken", 3820))
            cancels = [e for e in wire.sent if e["type"] == "response.cancel"]
            assert len(cancels) == (0 if auto_cancel else 1)
            assert wire.sent[-1] == {"type": "conversation.item.truncate", "item_id": "spoken",
                                    "content_index": 0, "audio_end_ms": 3820}
            done = await receive(adapter, wire, DONE)
            assert [e.kind for e in done] == ["response_done"]
            assert done[0].data["status"] == "interrupted"
            assert done[0].data["raw_usage"] == USAGE
            assert done[0].data["usage"]["audio_output_tokens"] == 390
            if cancels:
                assert await receive(adapter, wire, ERROR) == []
            await finish_barge_in(adapter, wire, engine)
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["openai", "grok"])
@pytest.mark.parametrize("order", ["native_order", "done_before_error", "error_before_done"])
def test_cancel_in_flight_survives_native_done_then_unlinked_error(providers, engine, order):
    async def run():
        observed = []
        adapter, wire = make(providers, engine, wire=CancelWire(engine),
            observe=lambda direction, raw: observed.append((direction, json.loads(raw))))
        try:
            await start(adapter, wire)
            observed.clear()
            # The client begins cancelling before it learns that server VAD already did so.
            if order == "native_order":
                # A send already queued behind microphone work can reach the wire
                # after speech_started. Exercise the real send lock, without a clock race.
                await adapter._send_lock.acquire()
                cancelling = asyncio.create_task(adapter.send(Interrupt("spoken", 3820)))
                try:
                    await asyncio.sleep(0)
                    await receive(adapter, wire, SPEECH)
                finally:
                    adapter._send_lock.release()
                    await cancelling
            else:
                await adapter.send(Interrupt("spoken", 3820))
                await receive(adapter, wire, SPEECH)
            assert len([e for e in wire.sent if e["type"] == "response.cancel"]) == 1
            if order == "error_before_done":
                assert await receive(adapter, wire, ERROR) == []
            done = await receive(adapter, wire, DONE)
            assert [e.kind for e in done] == ["response_done"]
            assert done[0].data["status"] == "interrupted" and done[0].data["raw_usage"] == USAGE
            if order != "error_before_done":
                assert await receive(adapter, wire, ERROR) == []
                received = [e["type"] for direction, e in observed if direction == "receive"]
                assert received[-3:] == ["input_audio_buffer.speech_started", "response.done", "error"]
            if order == "native_order":
                assert [(direction, e["type"]) for direction, e in observed] == [
                    ("receive", "input_audio_buffer.speech_started"),
                    ("send", "response.cancel"), ("send", "conversation.item.truncate"),
                    ("receive", "response.done"), ("receive", "error"),
                ]
            assert ("receive", ERROR) in observed  # Benign still reaches raw task evidence.
            assert await receive(adapter, wire, DONE) == []  # No second usage receipt.
            await finish_barge_in(adapter, wire, engine)
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["openai", "grok"])
@pytest.mark.parametrize("status", ["cancelled", "completed"])
def test_interrupt_after_response_done_only_truncates(providers, engine, status):
    async def run():
        adapter, wire = make(providers, engine, wire=CancelWire(engine))
        try:
            await start(adapter, wire)
            await receive(adapter, wire, {"type": "response.done", "response": {"id": "reply", "status": status}})
            await adapter.send(Interrupt("spoken", 3820))
            assert not any(e["type"] == "response.cancel" for e in wire.sent)
            assert await receive(adapter, wire, {"type": "conversation.item.truncated",
                "item_id": "spoken", "audio_end_ms": 3820}) == []
            # A second, unrequested truncation must still fail.
            with pytest.raises(RealtimeSetupError, match="provider_truncation"):
                await receive(adapter, wire, {"type": "conversation.item.truncated",
                    "item_id": "spoken", "audio_end_ms": 3820})
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("engine", ["openai", "grok"])
@pytest.mark.parametrize("case", ["own_cancel", "unsolicited", "vad_only", "wrong_code", "wrong_response",
                                 "wrong_event", "next_response", "duplicate", "non_error_event"])
def test_only_the_just_sent_cancel_can_make_one_error_benign(providers, engine, case):
    async def run():
        adapter, wire = make(providers, engine)
        try:
            await start(adapter, wire)
            if case == "vad_only":
                await receive(adapter, wire, SPEECH)
                await adapter.send(Interrupt())
            elif case != "unsolicited":
                await adapter.send(Interrupt())
            event = json.loads(json.dumps(ERROR))
            if case == "wrong_code":
                event["error"]["code"] = "conversation_item_truncate_invalid_audio_end_ms"
            elif case == "wrong_response":
                event["error"]["response_id"] = "another-response"
            elif case == "wrong_event":
                event["error"]["event_id"] = "another-client-event"
            elif case == "non_error_event":
                event["type"] = "session.updated"
            elif case == "next_response":
                await receive(adapter, wire, DONE)
                await receive(adapter, wire, {"type": "response.created", "response": {"id": "next-reply"}})
            elif case == "duplicate":
                assert await receive(adapter, wire, ERROR) == []
            if case == "own_cancel":
                assert await receive(adapter, wire, event) == []
                assert not wire.closed
            else:
                with pytest.raises(RealtimeSetupError, match="provider_error") as caught:
                    await receive(adapter, wire, event)
                assert event["error"]["message"] not in str(caught.value)
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("completed", [False, True])
def test_openai_cancel_error_can_correlate_to_client_event_id(providers, completed):
    async def run():
        adapter, wire = make(providers, "openai")
        try:
            await start(adapter, wire)
            await adapter.send(Interrupt())
            cancel = wire.sent[-1]
            assert cancel["type"] == "response.cancel" and cancel["response_id"] == "reply"
            if completed:
                await receive(adapter, wire, {"type": "response.done", "response": {"id": "reply", "status": "completed"}})
            error = {**ERROR, "error": {**ERROR["error"], "event_id": cancel["event_id"], "response_id": "reply"}}
            assert await receive(adapter, wire, error) == []
        finally:
            await adapter.close()
    asyncio.run(run())


def test_shared_json_reader_keeps_errors_fatal_by_default():
    async def run():
        wire = Wire()
        session = JsonSession(selected(), LiveOptions(), connect=wire.connect, observe=lambda *_: None)
        try:
            await session.connect_wire()
            wire.put(ERROR)
            with pytest.raises(RealtimeSetupError, match="provider_error"):
                await session.read()
            normal = {"type": "session.created"}
            wire.put(normal)
            assert await session.read() == normal
        finally:
            await session.close()
    asyncio.run(run())
