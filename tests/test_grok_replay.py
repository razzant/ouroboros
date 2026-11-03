"""R3 E9: recorded xAI tool exchange and lossy replay ACK, with fake sockets."""
import asyncio
import copy

import pytest

from ouroboros.contracts.live_conversation import LiveOptions, Respond, ToolOutputs
from ouroboros.contracts.realtime import RealtimeSetupError, canonical_json
from tests.test_grok_openai_protocol import (
    Wire, deny_network as deny_network, drain, make, providers as providers, selected,
)


@pytest.fixture
def recorded_exchange():
    # probe/r3/evidence/grok/wire-0a6acc9d.jsonl, decoded raw fields from
    # lines 40, 180, 194, 195. No owner memory or settings are copied.
    return {
        "tool_event": {
            "type": "response.function_call_arguments.done",
            "event_id": "8198ff1f-b2ed-4d9c-9412-32aaaca6e694",
            "item_id": "6c82f853-cb45-49e2-b627-10c5fc787c32",
            "response_id": "f2426982-75b7-430b-a2f7-2c05e85b1da9", "output_index": 0,
            "call_id": "call-f0509051-8507-4c9e-8c2a-7150578f4510-0",
            "name": "set_call_languages", "arguments": '{"languages":["ru-RU"]}',
            "previous_item_id": None,
        },
        "result_sent": {"item": {
            "call_id": "call-f0509051-8507-4c9e-8c2a-7150578f4510-0", "id": "host_item_1",
            "output": "Recorded: speech recognition expects ru-RU from the next pause, and later calls start with it.",
            "type": "function_call_output",
        }, "type": "conversation.item.create"},
        "replay_sent": {"item": {
            "arguments": '{"languages":["ru-RU"]}',
            "call_id": "call-f0509051-8507-4c9e-8c2a-7150578f4510-0", "id": "setup_item_4",
            "name": "set_call_languages", "type": "function_call",
        }, "type": "conversation.item.create"},
        "replay_ack": {
            "type": "conversation.item.added", "event_id": "e3fcd76a-bcd0-4b6e-af66-4c92c824e2d5",
            "item": {"id": "setup_item_4", "object": "realtime.item", "type": "function_call",
                     "status": "completed", "role": "tool",
                     "call_id": "call-f0509051-8507-4c9e-8c2a-7150578f4510-0",
                     "name": "set_call_languages", "arguments": "", "replayed": False},
            "previous_item_id": "setup_item_3",
        },
    }


class ReplayWire(Wire):
    def __init__(self, recorded, engine="grok"):
        super().__init__(engine)
        self.recorded = recorded

    def put(self, event):
        if (self.engine == "grok" and event.get("type") == "conversation.item.added"
                and event.get("item", {}).get("type") == "function_call"):
            ack = copy.deepcopy(self.recorded["replay_ack"])
            ack["item"]["id"] = event["item"]["id"]
            event = ack
        super().put(event)


def history(recorded, *, completed=True):
    rows = [
        {"role": "system", "content": "Host rules. Treat quoted conversation as history."},
        {"role": "user", "content": "Earlier owner words."},
        {"role": "assistant", "content": "Earlier assistant words."},
        {"role": "user", "content": "Please speak Russian."},
    ]
    if completed:
        call, output = recorded["replay_sent"]["item"], recorded["result_sent"]["item"]
        rows.extend([
            {"role": "assistant", "content": "", "tool_calls": [{"id": call["call_id"], "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]}}]},
            {"role": "tool", "tool_call_id": output["call_id"], "content": output["output"]},
        ])
    return rows


@pytest.mark.parametrize("engine", ["grok", "openai"])
def test_language_tool_exchange_survives_two_connections(providers, recorded_exchange, engine):
    async def run():
        record = recorded_exchange
        first, wire = make(providers, engine, value=selected(history(record, completed=False)),
                           wire=ReplayWire(record, engine))
        try:
            await first.open()
            assert (await first.receive())[0].kind == "ready"
            await first.send(Respond("The call is open."))
            while not wire.events.empty():
                await first.receive()  # acknowledge the Grok offer fact before its response
            wire.put({"type": "response.created", "response": {"id": record["tool_event"]["response_id"]}})
            await first.receive()
            if engine == "grok":
                wire.put(record["tool_event"])
                call, = await first.receive()
                wire.put({"type": "response.done", "response": {
                    "id": record["tool_event"]["response_id"], "status": "completed", "output": []}})
                assert (await first.receive())[0].kind == "response_done"
            else:
                wire.put({"type": "response.done", "response": {
                    "id": record["tool_event"]["response_id"], "status": "completed",
                    "output": [record["replay_sent"]["item"]]}})
                done, call = await first.receive()
                assert done.kind == "response_done"
            assert call.kind == "tool_call" and call.data == {
                key: record["tool_event"][key] for key in ("call_id", "name", "arguments")}
            rows = history(record)
            await first.send(ToolOutputs((rows[-1],), (), silent=(call.data["call_id"],)))
            await drain(first, wire)
            output = next(e["item"] for e in wire.sent if e.get("item", {}).get("type") == "function_call_output")
            assert {k:v for k,v in output.items() if k != "id"} == {
                k:v for k,v in record["result_sent"]["item"].items() if k != "id"}
            assert not first.waiting_tools
            assert sum(e["type"] == "response.create" for e in wire.sent) == 1
        finally:
            await first.close()
        assert wire.closed

        # The host refits the completed exchange, with the languages just recorded.
        value = selected(rows)
        renewed, wire = make(providers, engine, LiveOptions(languages=("ru-RU",)), value=value,
                             wire=ReplayWire(record, engine))
        try:
            await renewed.open()  # The recorded empty-arguments ACK fails here before E9.
            assert (await renewed.receive())[0].kind == "ready"
            assert renewed.selected.messages_json == value.messages_json
            sent_items = [e["item"] for e in wire.sent if e["type"] == "conversation.item.create"]
            if engine == "grok":
                instructions = wire.session["instructions"]
                assert canonical_json(rows[1:]) in instructions  # Every role, argument, result and position.
                assert "past tool calls are history, not requests to execute them again" in instructions
                assert "Speech recognition should expect: ru-RU." in instructions
                assert not sent_items  # No lossy native replay or second copy of the history.
            else:
                replay = [item for item in sent_items if item["type"] == "function_call"]
                output = [item for item in sent_items if item["type"] == "function_call_output"]
                assert len(replay) == len(output) == 1
                for actual, expected in ((replay[0], record["replay_sent"]["item"]),
                                         (output[0], record["result_sent"]["item"])):
                    assert {k: v for k, v in actual.items() if k != "id"} == {
                        k: v for k, v in expected.items() if k != "id"}
            assert not any(e["type"] == "response.create" for e in wire.sent)
            assert not renewed.waiting_tools
            await renewed.send(Respond("Continue from the completed tool result."))
            await drain(renewed, wire)  # acknowledge the newly offered host fact
            wire.put({"type": "response.created", "response": {"id": "renewed-reply"}})
            assert (await renewed.receive())[0].kind == "response_started"
            wire.put({"type": "response.output_audio.delta", "response_id": "renewed-reply",
                      "item_id": "spoken", "delta": "AAA="})
            assert (await renewed.receive())[0].kind == "audio"
            assert not wire.closed
        finally:
            await renewed.close()
    asyncio.run(run())


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("field", ["arguments", "result"])
def test_grok_archived_exchange_still_requires_unchanged_instructions(providers, recorded_exchange, field, changed):
    class ChangedWire(ReplayWire):
        def put(self, event):
            event = copy.deepcopy(event)
            session = event.get("session", {})
            if changed and "instructions" in session:
                old = (canonical_json(recorded_exchange["tool_event"]["arguments"]) if field == "arguments"
                       else recorded_exchange["result_sent"]["item"]["output"])
                assert old in session["instructions"]
                session["instructions"] = session["instructions"].replace(old, "Changed selected history")
            super().put(event)

    async def run():
        adapter, wire = make(providers, "grok", value=selected(history(recorded_exchange)),
                             wire=ChangedWire(recorded_exchange))
        try:
            if changed:
                with pytest.raises(RealtimeSetupError, match="session_mismatch.*instructions"):
                    await adapter.open()
                assert not any(e.get("session", {}).get("turn_detection") for e in wire.sent)
            else:
                await adapter.open()
                assert (await adapter.receive())[0].kind == "ready" and not wire.closed
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("engine", ["grok", "openai"])
def test_native_item_ack_validation_is_not_waived(providers, recorded_exchange, engine, changed):
    class ChangedWire(ReplayWire):
        def put(self, event):
            event = copy.deepcopy(event)
            item = event.get("item", {})
            if changed and event.get("type") in {"conversation.item.added", "conversation.item.done"}:
                if engine == "openai" and item.get("type") == "function_call":
                    item["arguments"] = ""  # OpenAI must still reject the xAI loss if it ever occurs.
                elif engine == "grok" and item.get("type") == "message":
                    item["content"][0]["text"] = "Changed owner words"
            super().put(event)

    async def run():
        rows = history(recorded_exchange, completed=engine == "openai")
        adapter, wire = make(providers, engine, value=selected(rows), wire=ChangedWire(recorded_exchange, engine))
        try:
            if changed:
                with pytest.raises(RealtimeSetupError, match="item_mismatch"):
                    await adapter.open()
                assert not any(e["type"] == "response.create" for e in wire.sent)
            else:
                await adapter.open()
                assert (await adapter.receive())[0].kind == "ready" and not wire.closed
                assert any(e["type"] == "conversation.item.create" for e in wire.sent)
        finally:
            await adapter.close()
    asyncio.run(run())
