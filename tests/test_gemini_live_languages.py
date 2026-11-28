"""R1 G1: real host language batches through Gemini's fake wire; a recorded list waits for a fresh connection (#4426)."""
from __future__ import annotations

import asyncio
import json

import pytest

from ouroboros.contracts.live_conversation import Respond, ToolOutputs
from skills.live_conversation.driver import call_tools, recorded_call_languages, session_options
from tests._live_world import host_world as host_world_fixture
from tests.test_gemini_live_boundaries import deny_network  # noqa: F401
from tests.test_gemini_live_boundaries import gemini as gemini_fixture, Wire, audio_content, receive

host_world = host_world_fixture
gemini = gemini_fixture


def language_host(host_world, gemini, tmp_path, *, extended):
    host = host_world.host
    model = "gemini-3.8-live" + ("-extended-thinking" if extended else "")
    host.model = gemini.CATALOG.model(model)
    host.turn.native_model, host.turn.model, host.turn.route = model, "google::" + model, gemini.ROUTE
    host.call_tools = {tool.name: tool for tool in call_tools(tmp_path / "call-state")}
    host.schemas = [schema for schema in host.schemas if schema["function"]["name"] != "set_test_language"]
    host.schemas.extend({"type": "function", "function": {"name": tool.name,
        "description": tool.description, "parameters": dict(tool.parameters)}} for tool in host.call_tools.values())
    wires = {}

    def factory(selected, options, *, observe):
        wire = Wire()
        session = gemini.GeminiSession(selected, options, connect=wire.connect, observe=observe)
        wires[session] = wire
        return session
    host_world.provider["session"] = factory
    return host, wires


@pytest.mark.parametrize("extended", [True, False])
@pytest.mark.parametrize("idle_before_result", [False, True])
def test_language_batch_reaches_gemini_and_waits_for_its_next_fresh_connection(host_world, gemini, tmp_path,
                                                                              extended, idle_before_result):
    from skills.live_conversation.driver import _Call
    host, wires = language_host(host_world, gemini, tmp_path, extended=extended)
    call = _Call(host)  # publishes take_languages; no connection loop runs here

    async def consume(session):
        wire = wires[session]
        call.session = session
        call.connected.set()
        await session.receive()  # Ready after the real host's physical fit/open.
        await session.send(Respond(host.opening))
        await receive(session, wire, audio_content("Synthetic greeting."))
        await receive(session, wire, {"serverContent": {
            "turnComplete": True, "interactionStatus": "IN_PROGRESS"},
            "usageMetadata": {"promptTokensDetails": [{"modality": "TEXT", "tokenCount": 100}],
                              "thoughtsTokenCount": 7}})
        events = await receive(session, wire, {"toolCall": {"functionCalls": [{
            "name": "set_call_languages", "args": {"languages": ["ru-RU", "en-US"]}, "id": "languages-1"}]}})
        assert not session.renewal_ready
        if idle_before_result:
            await receive(session, wire, {"serverContent": {"turnComplete": True, "interactionStatus": "IDLE"}})
            assert not session.renewal_ready  # An unanswered tool still owns the connection.
        calls = [{"id": e.data["call_id"], "type": "function", "function": {
            "name": e.data["name"], "arguments": e.data["arguments"]}} for e in events if e.kind == "tool_call"]
        changes = await host.tool_batch(calls)
        assert changes.silent == ()  # recording languages leaves the turn open for the next words
        # No renewal for the list: it waits for this engine's next fresh connection or the next call.
        assert not host.state.get("refit_after_answer") and host.state["languages"] == ("ru-RU", "en-US")
        assert not host.state.get("refit")
        await session.send(ToolOutputs(tuple(changes), tuple(host.schemas), silent=changes.silent))
        responses = [event["toolResponse"]["functionResponses"] for event in wire.sent if "toolResponse" in event]
        assert len(responses) == 1
        assert responses[0][0]["name"] == "set_call_languages"
        assert "ru-RU, en-US" in responses[0][0]["response"]["output"]
        assert "only when it starts a fresh connection" in responses[0][0]["response"]["output"]
        assert responses[0][0].get("scheduling") == (None if extended else "WHEN_IDLE")
        assert not session.calls
        assert session.renewal_ready is idle_before_result
        if not idle_before_result:
            await receive(session, wire, {"serverContent": {"turnComplete": True, "interactionStatus": "IDLE"}})
        assert session.renewal_ready

    async def refitted(session):
        wire = wires[session]
        setup = wire.sent[0]["setup"]
        assert setup["inputAudioTranscription"] == {"languageCodes": ["ru-RU", "en-US"]}
        assert setup["outputAudioTranscription"] == {"languageCodes": ["ru-RU", "en-US"]}
        assert setup["sessionResumption"] == {}
        assert any(row.get("tool_call_id") == "languages-1" for row in json.loads(session.selected.messages_json))
        await session.receive()
        events = await receive(session, wire, {"toolCall": {"functionCalls": [{
            "name": "set_call_languages", "args": {"languages": ["ru-RU", "en-US"]}, "id": "languages-2"}]}})
        call = next(e.data for e in events if e.kind == "tool_call")
        changes = await host.tool_batch([{"id": call["call_id"], "type": "function", "function": {
            "name": call["name"], "arguments": call["arguments"]}}])
        await session.send(ToolOutputs(tuple(changes), tuple(host.schemas), silent=changes.silent))
        assert not host.state.get("refit") and not host.state.get("refit_after_answer")
        assert not session.pending

    async def run():
        host.state["recognition"] = ()  # what the driver records for a fresh session opened with no list
        await host.session(consume, session_options(host))
        assert recorded_call_languages(tmp_path / "call-state") == ("ru-RU", "en-US")
        # A native resume keeps the setting its session was created with: no document says a resumed Gemini session
        # accepts a changed setup, so its languageCodes stay what that session started with.
        from ouroboros.contracts.realtime import RealtimeInput
        selected = RealtimeInput(host.turn.native_model, json.dumps([{"role": "system", "content": "Host rules."}]),
                                 "[]", 256, "nano", "core", "view")
        resumed = gemini.layout(selected, session_options(host, "handle-1"))
        assert resumed[0]["setup"]["inputAudioTranscription"] == {} and resumed[0]["setup"]["sessionResumption"] == {
            "handle": "handle-1"}
        await host.session(refitted, session_options(host))  # the next fresh connection carries the list
    asyncio.run(run())


@pytest.mark.parametrize("caption_finished", [False, True])
def test_thinking_language_tool_through_driver_waits_for_playback_and_idle(host_world, gemini, tmp_path,
                                                                       caption_finished):
    from ouroboros import call_protocol as cp
    from skills.live_conversation.driver import _Call
    from tests.test_live_conversation_driver import Hub

    host, wires = language_host(host_world, gemini, tmp_path, extended=True)
    hub = Hub()
    hub.spec = host.hub.spec
    host.hub = hub

    async def run():
        call = _Call(host)
        task = asyncio.create_task(call.run())
        stage = "initial connection"

        async def until(predicate):
            async def wait():
                while not predicate():
                    if task.done():
                        await task  # Preserve a real exception rather than a timeout.
                        pytest.fail("Call ended before the language tool/refit completed")
                    await asyncio.sleep(.01)
            await asyncio.wait_for(wait(), 20)

        try:
            await until(lambda: call.connected.is_set())
            session = call.session
            wire = wires[session]
            stage = "greeting receipt"
            wire.put(audio_content("Synthetic greeting.", finished=caption_finished))
            wire.put({"serverContent": {"generationComplete": True}})
            wire.put({"serverContent": {"turnComplete": True, "interactionStatus": "IN_PROGRESS"},
                      "usageMetadata": {"promptTokenCount": 38940, "responseTokenCount": 52,
                          "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 38607},
                                                  {"modality": "AUDIO", "tokenCount": 222}],
                          "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 52}],
                          "thoughtsTokenCount": 97}})
            await until(lambda: bool(call.audio_ms) and bool(call.receipts))
            assert not call.quiet() and call.session is session
            item, duration = next(iter(call.audio_ms.items()))
            hub.events.append(cp.PeerControl("synthetic-peer", cp.PEER_OWNER,
                {"type": "played", "item_id": item, "played_ms": int(duration)}))
            stage = "greeting playback acknowledgement"
            await until(lambda: not call.unheard())
            assert not call.quiet() and call.session is session  # IN_PROGRESS still owns this response.
            wire.put({})  # The probe also received an empty native frame before the tool.
            wire.put({"toolCall": {"functionCalls": [{"name": "set_call_languages",
                "args": {"languages": ["ru-RU", "en-US"]}, "id": "languages-1"}]}})
            stage = "language tool response"
            await until(lambda: any("toolResponse" in event for event in wire.sent))
            response = next(event["toolResponse"] for event in wire.sent if "toolResponse" in event)
            assert "scheduling" not in response["functionResponses"][0]
            assert not call.quiet() and call.session is session
            wire.put({"serverContent": {"turnComplete": True, "interactionStatus": "IDLE"}})
            stage = "no reconnection for the list"
            assert "only when it starts a fresh connection" in response["functionResponses"][0]["response"]["output"]
            await until(lambda: call.quiet())
            await asyncio.sleep(.3)
            assert call.session is session and not call.renewing
            assert len([w for w in wires.values() if w.sent]) == 1, "one opened connection"
            assert host.state["languages"] == ("ru-RU", "en-US") and not host.state.get("refit_after_answer")
            # A native resume (goAway with a handle) keeps the setting its session was created with.
            stage = "resume after the list changed"
            wire.put({"sessionResumptionUpdate": {"newHandle": "synthetic-handle", "resumable": True}})
            wire.put({"goAway": {"timeLeft": "5s"}})
            await until(lambda: call.session is not None and call.session is not session and call.connected.is_set())
            setup = wires[call.session].sent[0]["setup"]
            assert setup["sessionResumption"] == {"handle": "synthetic-handle"}
            assert setup["inputAudioTranscription"] == {} and setup["outputAudioTranscription"] == {}
            assert host.state["greeted"]
            assert not any("clientContent" in event for event in wires[call.session].sent)  # no second greeting
        except Exception as exc:
            sent = [[list(event) for event in wire.sent] for wire in wires.values() if wire.sent]
            raise AssertionError(f"Gemini language scenario failed at {stage}; sent frame types={sent}") from exc
        finally:
            hub.ended_by_owner = True
            if not task.done():
                await asyncio.wait_for(task, 20)
    asyncio.run(run())
