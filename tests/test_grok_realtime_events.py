"""xAI's native function and caption events through fake sockets."""
import asyncio

import pytest

from ouroboros.contracts.live_conversation import ToolOutputs
from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, make, drain


@pytest.mark.parametrize("sources", [("arguments",), ("arguments", "response"), ("response", "arguments")])
def test_native_function_arguments_emit_once_per_call_id(providers, sources):
    async def run():
        adapter, wire = make(providers, "grok")
        call = {"type": "function_call", "name": "read_file", "call_id": "c", "arguments": '{"path":"notes"}'}
        arguments = {**call, "type": "response.function_call_arguments.done"}
        events = {"arguments": arguments, "response": {"type": "response.done", "response": {
            "id": "r", "status": "completed", "output": [call]}}}
        try:
            await adapter.open()
            await adapter.receive()
            calls = []
            for source in sources:
                wire.put(events[source])
                calls.extend(event.data for event in await adapter.receive() if event.kind == "tool_call")
            assert calls == [{key: call[key] for key in ("name", "call_id", "arguments")}]
            assert adapter.waiting_tools == {"c"}
            await adapter.send(ToolOutputs(({"role": "tool", "tool_call_id": "c", "content": "result"},), (), silent=("c",)))
            await drain(adapter, wire)
            assert not adapter.waiting_tools
            wire.put(arguments)  # A repeat after completion must not re-arm the tool.
            assert await adapter.receive() == []
            assert not adapter.waiting_tools
            wire.put({**arguments, "call_id": "next"})
            assert (await adapter.receive())[0].data["call_id"] == "next"
            assert adapter.waiting_tools == {"next"}
            assert not any(event["type"] == "response.create" for event in wire.sent)
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("prefix", ["response.audio_transcript", "response.output_audio_transcript"])
@pytest.mark.parametrize("final_field", ["delta", "transcript"])
def test_assistant_caption_appends_deltas_and_replaces_full_transcripts(providers, prefix, final_field):
    async def run():
        adapter, wire = make(providers, "grok")
        try:
            await adapter.open()
            await adapter.receive()
            displayed = ""
            for text in ("Hello ", "there"):
                wire.put({"type": prefix + ".delta", "item_id": "spoken", "delta": text})
                caption, = await adapter.receive()
                assert caption.kind == "caption" and not caption.data["replace"]
                assert not caption.data["final"] and caption.data["speaker"] == "ouroboros"
                displayed += caption.data["text"]
            assert displayed == "Hello there"
            wire.put({"type": prefix + ".done", "item_id": "spoken",
                      final_field: "Hello there!" if final_field == "transcript" else "!"})
            caption, = await adapter.receive()
            assert caption.data["replace"] is (final_field == "transcript")
            assert caption.data["final"]
            displayed = caption.data["text"] if caption.data["replace"] else displayed + caption.data["text"]
            assert displayed == "Hello there!"
            # A full transcript may also revise a nonfinal caption.
            caption, = adapter.convert({"type": prefix + ".delta", "item_id": "spoken",
                                       "transcript": "Revised words", "delta": "ignored fragment"})
            assert caption.data["replace"] and caption.data["text"] == "Revised words"
        finally:
            await adapter.close()
    asyncio.run(run())


def test_owner_full_transcript_keeps_replacement_and_final_status(providers):
    adapter, _ = make(providers, "grok")
    for status in ("in_progress", "completed"):
        caption, = adapter.convert({"type": "conversation.item.input_audio_transcription.completed",
            "item_id": "owner", "transcript": "Owner words", "status": status})
        assert caption.data["replace"] and caption.data["text"] == "Owner words"
        assert caption.data["final"] is (status == "completed")
