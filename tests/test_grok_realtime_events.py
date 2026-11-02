"""xAI's native function and caption events through fake sockets."""
import asyncio

import pytest

from ouroboros.contracts.live_conversation import ContextUpdate, ToolOutputs
from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, make, drain
from tests.test_live_conversation import host_world as host_world


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
            assert caption.data["replace"]  # the durable final row contains the entire phrase
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


def test_grok_renewal_waits_for_vad_answer_and_late_transcripts(providers):
    from skills.live_conversation.driver import _Call
    from tests.test_live_conversation_driver import Host

    async def run():
        adapter, _ = make(providers, "grok")
        call = _Call(Host())
        call.session = adapter
        assert not call.quiet(), 'an unopened connection is not ready'
        try:
            await adapter.open()
            await adapter.receive()
            assert call.quiet()
            adapter.convert({'type': 'input_audio_buffer.speech_started', 'item_id': 'owner'})
            assert not call.quiet()
            adapter.convert({'type': 'input_audio_buffer.speech_stopped', 'item_id': 'owner'})
            assert not call.quiet(), 'input_done must not permit renewal before the VAD answer'
            adapter.convert({'type': 'response.created', 'response': {'id': 'answer'}})
            assert not call.quiet()
            adapter.convert({'type': 'response.audio.delta', 'item_id': 'spoken', 'delta': 'AAA='})
            adapter.convert({'type': 'response.done', 'response': {'id': 'answer', 'status': 'completed'}})
            assert not call.quiet(), 'both transcripts are still in flight'
            adapter.convert({'type': 'conversation.item.input_audio_transcription.completed',
                             'item_id': 'owner', 'transcript': 'Question', 'status': 'in_progress'})
            assert not call.quiet()
            adapter.convert({'type': 'conversation.item.input_audio_transcription.completed',
                             'item_id': 'owner', 'transcript': 'Question?', 'status': 'completed'})
            assert not call.quiet(), 'the output transcript is still in flight'
            adapter.convert({'type': 'response.audio_transcript.done', 'item_id': 'spoken', 'transcript': 'Answer.'})
            assert call.quiet()
        finally:
            await adapter.close()
        assert not call.quiet()
    asyncio.run(run())


def test_grok_renewal_waits_for_context_ack_requested_response_and_tools(providers):
    async def run():
        adapter, wire = make(providers, 'grok')
        try:
            await adapter.open()
            await adapter.receive()
            assert adapter.renewal_ready
            adapter.convert({'type': 'response.created', 'response': {'id': 'tools'}})
            await adapter.send(ContextUpdate(({'role': 'user', 'content': 'New context'},)))
            adapter.convert({'type': 'response.done', 'response': {'id': 'tools', 'status': 'completed',
                'output': [{'type': 'function_call', 'call_id': 'c', 'name': 'attend', 'arguments': '{}'}]}})
            assert not adapter.renewal_ready  # queued context + waiting tool
            await adapter.advance_context()
            assert adapter.outstanding and not adapter.renewal_ready
            await drain(adapter, wire)
            assert adapter.want_response and adapter.waiting_tools and not adapter.renewal_ready
            await adapter.send(ToolOutputs(({'role': 'tool', 'tool_call_id': 'c', 'content': 'done'},), (), silent=('c',)))
            await drain(adapter, wire)
            assert adapter.response_requested and not adapter.renewal_ready
            adapter.convert({'type': 'response.created', 'response': {'id': 'answer'}})
            adapter.convert({'type': 'response.done', 'response': {'id': 'answer', 'status': 'completed'}})
            assert adapter.renewal_ready
            await adapter.send(ContextUpdate(({'role': 'system', 'content': 'Added instructions'},)))
            assert adapter.session_update is not None and not adapter.renewal_ready
            await drain(adapter, wire)
            assert adapter.response_requested and not adapter.renewal_ready
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize('finished', [False, True])
def test_grok_received_words_survive_finalization_in_durable_history(providers, host_world, finished):
    from ouroboros import dialogue_store
    from skills.live_conversation.driver import _Call

    async def run():
        adapter, _ = make(providers, 'grok')
        host = host_world.host
        call = _Call(host)
        call.last_session = adapter
        call.writer = asyncio.create_task(call.save_rows())
        try:
            for event in [
                {'type': 'response.audio_transcript.delta', 'item_id': 'a', 'delta': 'First '},
                {'type': 'response.audio_transcript.delta', 'item_id': 'b', 'delta': 'Separate phrase'},
                {'type': 'response.audio_transcript.delta', 'item_id': 'a', 'delta': 'phrase'},
                {'type': 'conversation.item.input_audio_transcription.completed', 'item_id': 'owner',
                 'transcript': 'Owner draft', 'status': 'in_progress'},
                {'type': 'conversation.item.input_audio_transcription.completed', 'item_id': 'owner',
                 'transcript': 'Owner revised words', 'status': 'in_progress'},
            ]:
                for caption in adapter.convert(event):
                    await call.caption(caption.data)
            if finished:
                for item, speaker, text in [('a', 'ouroboros', 'First phrase.'), ('b', 'ouroboros', 'Separate phrase.'),
                                             ('owner', 'owner', 'Owner revised words.')]:
                    kind = ('conversation.item.input_audio_transcription.completed' if speaker == 'owner'
                            else 'response.audio_transcript.done')
                    for caption in adapter.convert({'type': kind, 'item_id': item, 'transcript': text, 'status': 'completed'}):
                        await call.caption(caption.data)
                assert adapter.final_captions() == []
            else:
                assert not host.captioned, 'partial UI captions are not yet durable'
            await adapter.close()  # End, Stop and a dropped connection all use finalize after close.
            await call.finalize()
            assert adapter.final_captions() == [], 'a finalized tail must not be saved twice'
            assert len(host.captioned) == 3
            for (speaker, item), saved in host.captioned.items():
                text = {'a': 'First phrase', 'b': 'Separate phrase', 'owner': 'Owner revised words'}[item]
                expected = text + '.' if finished else '[Partial transcript] ' + text
                assert saved['text'] == expected
                # Read the actual persisted row, not just the driver's or host's cache.
                row = dialogue_store.row_at(host.ctx.drive_root, 'chat', saved['seq'])
                assert expected in row.row['text']
        finally:
            call.writer.cancel()
            await asyncio.gather(call.writer, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize('old_receipt_late', [False, True])
def test_an_old_cancelled_receipt_cannot_release_the_next_unanswered_vad_turn(providers, old_receipt_late):
    """R18-1: only response.created consumes the VAD obligation, in either receipt order."""
    from skills.live_conversation.driver import _Call
    from tests.test_live_conversation_driver import Host, until

    async def scenario():
        native, wire = make(providers, 'grok')
        host = Host()
        host.state['greeted'] = True
        call = _Call(host)
        call.session = native
        reader = None
        try:
            await native.open()
            reader = asyncio.create_task(call.provider())
            await until(call.connected.is_set)
            assert call.quiet()
            wire.put({'type': 'response.created', 'response': {'id': 'old'}})
            wire.put({'type': 'response.audio.delta', 'response_id': 'old', 'item_id': 'old-words', 'delta': 'AAA='})
            wire.put({'type': 'response.audio_transcript.done', 'response_id': 'old',
                      'item_id': 'old-words', 'transcript': 'Previous answer.'})
            wire.put({'type': 'input_audio_buffer.speech_started', 'item_id': 'new-question'})
            await until(lambda: call.input_open and 'old-words' in call.interrupted and not native.truncations)
            receipt = {'type': 'response.done', 'response': {'id': 'old', 'status': 'cancelled'}}
            if not old_receipt_late:
                wire.put(receipt)
            wire.put({'type': 'input_audio_buffer.speech_stopped', 'item_id': 'new-question'})
            wire.put({'type': 'conversation.item.input_audio_transcription.completed',
                      'item_id': 'new-question', 'status': 'completed', 'transcript': 'New question?'})
            await until(lambda: not call.input_open and not native.input_transcripts)
            assert not call.quiet()
            if old_receipt_late:
                wire.put(receipt)
                await until(lambda: any(row[:2] == ('done', 'old') for row in host.usage))
            assert not call.quiet() and native.vad_response_pending, 'a renewal must still wait for the new answer'
            wire.put({'type': 'response.created', 'response': {'id': 'new'}})
            await until(lambda: ('start', 'new') in host.usage)
            assert not call.quiet()
            wire.put({'type': 'response.done', 'response': {'id': 'new', 'status': 'completed'}})
            await until(lambda: any(row[:2] == ('done', 'new') for row in host.usage))
            assert call.quiet(), 'a finished native turn can renew'
        finally:
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            await native.close()
    asyncio.run(scenario())
