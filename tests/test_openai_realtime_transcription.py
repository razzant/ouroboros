"""An item transcription failure does not end the native spoken conversation."""
import asyncio

import pytest

from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, make


@pytest.mark.parametrize("outcome", ["failed", "completed"])
def test_input_transcription_finishes_only_its_item_and_speech_continues(providers, outcome):
    async def run():
        adapter, wire = make(providers, "openai")
        try:
            await adapter.open()
            await adapter.receive()
            for item in ("owner", "other"):
                adapter.convert({"type": "input_audio_buffer.speech_started", "item_id": item})
                adapter.convert({"type": "input_audio_buffer.speech_stopped", "item_id": item})
            assert not adapter.renewal_ready
            wire.put({"type": "conversation.item.input_audio_transcription." + outcome,
                      "item_id": "owner", "transcript": "Owner words."})
            events = await adapter.receive()
            assert not any(event.kind == "closed" for event in events)
            assert adapter.input_transcripts == {"other"}
            assert not adapter.renewal_ready and not wire.closed
            if outcome == "completed":
                assert len(events) == 1 and events[0].kind == "caption"
                assert events[0].data["text"] == "Owner words." and events[0].data["final"]
            else:
                assert events == []  # no recognized words to invent a caption from

            wire.put({"type": "conversation.item.input_audio_transcription.completed",
                      "item_id": "other", "transcript": "Next words."})
            assert (await adapter.receive())[0].data["text"] == "Next words."
            wire.put({"type": "response.created", "response": {"id": "answer"}})
            assert (await adapter.receive())[0].kind == "response_started"
            wire.put({"type": "response.output_audio.delta", "response_id": "answer",
                      "item_id": "spoken", "delta": "AAA="})
            assert (await adapter.receive())[0].kind == "audio"
            wire.put({"type": "response.output_audio_transcript.done", "item_id": "spoken", "transcript": "Answer."})
            assert (await adapter.receive())[0].kind == "caption"
            wire.put({"type": "response.done", "response": {"id": "answer", "status": "completed"}})
            assert [event.kind for event in await adapter.receive()] == ["response_done"]
            assert adapter.renewal_ready and not wire.closed
        finally:
            await adapter.close()
    asyncio.run(run())
