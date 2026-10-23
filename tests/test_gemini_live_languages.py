"""R1 G1: real host language batches through Gemini's fake wire and refit."""
from __future__ import annotations

import asyncio
import json

import pytest

from ouroboros.contracts.live_conversation import Respond, ToolOutputs
from skills.live_conversation.driver import call_tools, recorded_call_languages, session_options
from tests.test_live_conversation import host_world as host_world_fixture
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
def test_language_batch_reaches_gemini_before_fresh_refit(host_world, gemini, tmp_path,
                                                       extended, idle_before_result):
    host, wires = language_host(host_world, gemini, tmp_path, extended=extended)

    async def consume(session):
        wire = wires[session]
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
        assert changes.silent == ("languages-1",)
        assert host.state["refit"] and host.state["languages"] == ("ru-RU", "en-US")
        await session.send(ToolOutputs(tuple(changes), tuple(host.schemas), silent=changes.silent))
        responses = [event["toolResponse"]["functionResponses"] for event in wire.sent if "toolResponse" in event]
        assert len(responses) == 1
        assert responses[0][0]["name"] == "set_call_languages"
        assert "ru-RU, en-US" in responses[0][0]["response"]["output"]
        assert responses[0][0].get("scheduling") == (None if extended else "SILENT")
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
        assert not host.state.get("refit")  # Recording the same languages needs no new connection.
        assert not session.pending

    async def run():
        await host.session(consume, session_options(host))
        assert recorded_call_languages(tmp_path / "call-state") == ("ru-RU", "en-US")
        host.state.pop("refit")
        await host.session(refitted, session_options(host))
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
            stage = "fresh language refit"
            await until(lambda: call.session is not None and call.session is not session and call.connected.is_set())
            setup = wires[call.session].sent[0]["setup"]
            assert setup["inputAudioTranscription"] == {"languageCodes": ["ru-RU", "en-US"]}
            assert setup["sessionResumption"] == {}
            assert host.state["greeted"]
            assert not any("clientContent" in event for event in wires[call.session].sent)  # No second greeting.
        except Exception as exc:
            sent = [[list(event) for event in wire.sent] for wire in wires.values() if wire.sent]
            raise AssertionError(f"Gemini language scenario failed at {stage}; sent frame types={sent}") from exc
        finally:
            hub.ended_by_owner = True
            if not task.done():
                await asyncio.wait_for(task, 20)
    asyncio.run(run())
