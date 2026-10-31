"""Settings and call commands must reach Gemini's physical fake wire."""
from __future__ import annotations

import asyncio

import pytest

from ouroboros.contracts.live_conversation import ContextUpdate, LiveOptions, Respond, ToolOutputs
from tests.test_gemini_live_boundaries import deny_network  # noqa: F401
from tests.test_gemini_live_boundaries import gemini as gemini_fixture, make_session, receive

gemini = gemini_fixture


def tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


@pytest.mark.parametrize("pause,ms", [("short", 500), ("normal", None), ("long", 1600)])
@pytest.mark.parametrize("interruptions", [True, False])
def test_setup_honours_options_on_the_socket(gemini, pause, ms, interruptions):
    async def run():
        session, wire = make_session(gemini, LiveOptions(
            languages=("ru-RU", "en-US"), instructions="Feature then owner call instructions.",
            voice="Puck", pause=pause, interruptions=interruptions, search=True))
        try:
            await session.open()
            setup = wire.sent[0]["setup"]
            for key in ("inputAudioTranscription", "outputAudioTranscription"):
                assert setup[key] == {"languageCodes": ["ru-RU", "en-US"]}
            parts = setup["systemInstruction"]["parts"]
            assert parts[0]["text"] == "Host system rules."
            assert parts[1]["text"] == "Feature then owner call instructions."
            assert "Original owner history." in parts[2]["text"]
            assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"] == {"voiceName": "Puck"}
            vad = setup.get("realtimeInputConfig", {})
            assert vad.get("automaticActivityDetection", {}).get("silenceDurationMs") == ms
            assert vad.get("activityHandling", "START_OF_ACTIVITY_INTERRUPTS") == (
                "START_OF_ACTIVITY_INTERRUPTS" if interruptions else "NO_INTERRUPTION")
            assert {"googleSearch": {}} in setup["tools"]
            assert "thinkingConfig" not in setup["generationConfig"]
        finally:
            await session.close()
    asyncio.run(run())


@pytest.mark.parametrize("thinking,expected", [("", "low"), ("medium", "medium"), ("high", "high")])
def test_extended_thinking_uses_owner_level_and_low_default(gemini, thinking, expected):
    async def run():
        session, wire = make_session(gemini, LiveOptions(thinking=thinking), model="gemini-3.8-live-extended-thinking")
        try:
            await session.open()
            assert wire.sent[0]["setup"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": expected}
        finally:
            await session.close()
    asyncio.run(run())


def test_basic_model_and_defaults_do_not_claim_unsupported_controls(gemini):
    session, _ = make_session(gemini, LiveOptions(thinking="high"))
    setup = session.wire["events"][0]["setup"]
    assert "thinkingConfig" not in setup["generationConfig"]
    assert "realtimeInputConfig" not in setup
    assert setup["inputAudioTranscription"] == {}
    assert setup["tools"] == []
    assert "interleaved" in gemini.CATALOG.model().note("thinking")
    assert len(gemini.CATALOG.model().voices) == 30
    assert "hypothesis" in gemini.CATALOG.model("gemini-3.8-live-extended-thinking").facts.source


def test_greeting_is_a_host_fact_and_waits_for_active_generation(gemini):
    async def run():
        session, wire = make_session(gemini)
        try:
            await session.open()
            await session.receive()
            await session.send(Respond("The owner opened this call and can hear you."))
            frame = wire.sent[-1]["clientContent"]
            assert frame["turnComplete"] is True
            assert frame["turns"] == [{"role": "user", "parts": [{
                "text": "[Host fact; not owner words]\nThe owner opened this call and can hear you."}]}]
            count = len(wire.sent)
            await session.send(ContextUpdate(({"role": "user", "content": "A new message"},)))
            assert len(wire.sent) == count
            await receive(session, wire, {"serverContent": {"turnComplete": True, "interactionStatus": "IN_PROGRESS"}})
            assert len(wire.sent) == count
            await receive(session, wire, {"serverContent": {"turnComplete": True, "interactionStatus": "IDLE"}})
            assert "A new message" in wire.sent[-1]["clientContent"]["turns"][0]["parts"][0]["text"]
        finally:
            await session.close()
    asyncio.run(run())


@pytest.mark.parametrize("extended", [False, True])
def test_silent_tool_results_and_deduplicated_schema_renewal(gemini, extended):
    async def run():
        model = "gemini-3.8-live" + ("-extended-thinking" if extended else "")
        session, wire = make_session(gemini, model=model)
        try:
            await session.open()
            await session.receive()
            await receive(session, wire, {"toolCall": {"functionCalls": [
                {"id": "quiet", "name": "end_call", "args": {}},
                {"id": "normal", "name": "enable_tools", "args": {}}]}})
            schemas = (tool("read_file"), tool("read_file"))
            await session.send(ToolOutputs((
                {"role": "tool", "tool_call_id": "quiet", "content": "Done"},
                {"role": "tool", "tool_call_id": "normal", "content": "Enabled"},
            ), schemas, silent=("quiet",)))
            responses = wire.sent[-1]["toolResponse"]["functionResponses"]
            if extended:
                assert all("scheduling" not in row for row in responses)
            else:
                assert responses[0]["scheduling"] == "SILENT"
                assert responses[1]["scheduling"] == "WHEN_IDLE"
            renewal = await session.receive()
            assert [e.kind for e in renewal] == ["renew"]
            assert renewal[0].data["resume"] is False
            assert session.tools == (tool("read_file"),)
            await session.send(ToolOutputs((), (tool("read_file"),)))
            assert session.pending == []
            await session.send(ToolOutputs((), (tool("read_file"), tool("write_file"))))
            assert len(session.tools) == 2
            assert session.pending == []  # One pending renewal carries the latest schemas.
        finally:
            await session.close()
    asyncio.run(run())


def test_nonfunction_tools_refuse_typed_and_unfamiliar_functions_remain_available(gemini):
    from ouroboros.contracts.realtime import RealtimeSetupError
    with pytest.raises(RealtimeSetupError, match="tools"):
        make_session(gemini, tools=({"type": "unexpected"},))
    session, _ = make_session(gemini, tools=(tool("future_function"),))
    assert session.wire["events"][0]["setup"]["tools"][0]["functionDeclarations"][0]["name"] == "future_function"
