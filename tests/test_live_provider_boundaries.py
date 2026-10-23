"""Native-session consumer negatives; synthetic wires, no credentials or devices."""
from __future__ import annotations

import asyncio
import copy
import importlib

import pytest

from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests.test_live_conversation import ENGINES, VoiceWire, packages as packages


def selected():
    from ouroboros.contracts.realtime import RealtimeInput, canonical_json
    return RealtimeInput("synthetic", canonical_json([
        {"role": "system", "content": "Synthetic host rules."},
        {"role": "user", "content": "Keep my complete words."},
    ]), "[]", 256, "nano", "core", "view")


@pytest.mark.parametrize("name,engine", ENGINES)
def test_wrong_pcm_rate_refuses_before_send(packages, name, engine):
    from ouroboros.contracts.live_conversation import Audio, LiveOptions
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.realtime_setup import loaded_provider
    factory = loaded_provider(f"{name}:{engine}")["session"]
    wire = VoiceWire(engine, None)
    session = factory(selected(), LiveOptions(), connect=wire.connect, observe=lambda *_: None)
    with pytest.raises(RealtimeSetupError, match="audio_format"):
        asyncio.run(session.send(Audio(b"\x00\x00" * 20, 8000)))
    assert not wire.sent


def test_openai_changed_live_context_ack_refuses_response(packages):
    from ouroboros.contracts.live_conversation import ContextUpdate, LiveOptions
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.realtime_setup import loaded_provider

    class ChangedAck(VoiceWire):
        def put(self, value):
            value = copy.deepcopy(value)
            if value.get("type") == "conversation.item.done" and value["item"].get("id", "").startswith("host_item_"):
                value["item"]["content"][0]["text"] = "Provider changed the owner message."
            super().put(value)

    wire = ChangedAck("openai", None)
    session = loaded_provider("openai_realtime:openai")["session"](
        selected(), LiveOptions(), connect=wire.connect, observe=lambda *_: None)

    async def scenario():
        try:
            await session.open()
            assert (await session.receive())[0].kind == "ready"
            await session.send(ContextUpdate(({"role": "user", "content": "An exact new owner message."},)))
            with pytest.raises(RealtimeSetupError, match="item_mismatch"):
                await session.receive()
        finally:
            await session.close()
    asyncio.run(scenario())
    assert wire.closed
    assert not any(e.get("type") == "response.create" for e in wire.sent)


def test_gemini_idle_connection_does_not_rebill_same_input(packages):
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.extension_import_staging import _module_key
    module = importlib.import_module(_module_key("gemini_live"))
    wire = VoiceWire("gemini", None)
    session = module.GeminiSession(selected(), LiveOptions(), connect=wire.connect, observe=lambda *_: None)

    async def scenario():
        try:
            await session.open()
            assert (await session.receive())[0].kind == "ready"
            receiving = asyncio.create_task(session.receive())
            await asyncio.sleep(.02)
            assert not receiving.done()  # No clock-only renewal, even with a live socket read.
            events = await session.convert({"serverContent": {"turnComplete": True},
                "usageMetadata": {"promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 1300}]}})
            assert [e.kind for e in events] == ["response_started", "response_done", "renew"]
            assert not any(e.kind == "renew" for e in await session.convert(
                {"serverContent": {"turnComplete": True}}))
        finally:
            if 'receiving' in locals():
                receiving.cancel()
                await asyncio.gather(receiving, return_exceptions=True)
            await session.close()
    asyncio.run(scenario())
    assert wire.closed and len(wire.sent) == 1


def test_gemini_partial_then_late_final_keeps_complete_caption_and_audio_identity(packages):
    import base64
    from tests.test_live_provider_regressions import session as make_session

    async def scenario():
        adapter, _ = make_session("gemini")
        first = await adapter.convert({"serverContent": {
            "modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000",
                "data": base64.b64encode(b"\x00\x00" * 48).decode()}}]},
            "outputTranscription": {"text": "First words "}}})
        item = next(e.data["item_id"] for e in first if e.kind == "audio")
        boundary = await adapter.convert({"serverContent": {"turnComplete": True}})
        assert not adapter.renewal_ready
        assert not any(e.kind == "caption" and e.data["final"] for e in boundary)
        late = await adapter.convert({"serverContent": {
            "outputTranscription": {"text": "and the ending.", "finished": True}}})
        final = next(e for e in late if e.kind == "caption" and e.data["final"])
        assert final.data["item_id"] == item
        assert final.data["text"] == "First words and the ending."
        assert not any(e.kind == "response_started" for e in late)
        assert adapter.renewal_ready
    asyncio.run(scenario())


def test_gemini_expiry_with_untranscribed_audio_is_an_explicit_failure(packages):
    from tests.test_live_provider_regressions import DriverHost, driver, session as make_session, until
    from ouroboros import call_protocol as cp
    from ouroboros.contracts.realtime import RealtimeSetupError

    async def scenario():
        adapter, wire = make_session("gemini")
        host = DriverHost(adapter)
        running = asyncio.create_task(driver().run(host))
        try:
            await until(lambda: host.hub._state["state"] == cp.STATE_LIVE)
            assert host.hub.receive_audio("owner", cp.AudioFrame(
                cp.FRAME_AUDIO_IN, host.hub.epoch, 0, 24000, 0, b"\x01\x00" * 960))
            await until(lambda: any("realtimeInput" in row for row in wire.sent))
            wire.put({"voiceActivity": {"type": "ACTIVITY_START"}})
            wire.put({"goAway": {"timeLeft": "0.05s"}})
            with pytest.raises(RealtimeSetupError, match="renewal_pending_work"):
                await asyncio.wait_for(running, 1)
            assert not host.rows
        finally:
            host.hub.close("fixture_done")
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
    asyncio.run(scenario())
