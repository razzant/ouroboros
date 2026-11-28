"""An item transcription failure does not end the native spoken conversation."""
import asyncio
import copy
from dataclasses import replace

import pytest

from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, make
from tests._live_world import host_world as host_world


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
            else:  # no recognized words: an empty final closes the item and invents none
                assert [(event.kind, event.data) for event in events] == [
                    ("caption", {"speaker": "owner", "item_id": "owner", "text": "", "final": True})]

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


@pytest.mark.parametrize('missing', [None, 'audio_split', 'duration'])
def test_transcription_usage_reaches_shared_ledger_with_its_own_model(providers, host_world, monkeypatch, missing):
    from ouroboros import live_usage, pricing_catalog, usage_accounting as ua

    host = host_world.host
    host.turn.route = replace(host.turn.route, accounting_provider='openai_realtime')
    host._connection_accounting = True
    host.connection_number = 1
    tariff = {'catalog_id': 'gpt-4o-transcribe', 'fetched_at': 'fixture', 'sha256': 'fixture', 'entry': {
        'input_cost_per_token': .002, 'input_cost_per_audio_token': .006, 'output_cost_per_token': .01}}
    def cached_quote(model, *, provider, drive_root, allow_live_fetch):
        assert (model, provider, drive_root, allow_live_fetch) == (
            'gpt-4o-transcribe', 'openai_realtime', host.ctx.drive_root, False)
        return None if missing == 'tariff' else tariff
    monkeypatch.setattr(pricing_catalog, 'get_catalog_quote', cached_quote)
    # live_usage imports the same catalog lookup, so exercise its existing pricing boundary.
    monkeypatch.setattr(live_usage, 'get_catalog_quote', cached_quote)
    monkeypatch.setenv('TOTAL_BUDGET', '0.1')
    raw = {'type': 'tokens', 'input_tokens': 13, 'output_tokens': 9, 'total_tokens': 22,
           'input_token_details': {'text_tokens': 0, 'audio_tokens': 13}}
    if missing == 'audio_split':
        raw['input_token_details'].pop('audio_tokens')
    elif missing == 'duration':
        raw = {'type': 'duration', 'seconds': 3}
    event = {'type': 'conversation.item.input_audio_transcription.completed', 'event_id': 'asr-1',
             'item_id': 'owner-1', 'content_index': 0, 'transcript': 'Owner words.'}
    if missing != 'usage':
        event['usage'] = raw
    adapter, _ = make(providers, 'openai', observe=host.observe)
    with ua.usage_scope(ua.UsageScope(drive_root=host.ctx.drive_root, task_id=host.ctx.task_id,
                                    root_task_id=host.ctx.task_id)):
        caption, = adapter.convert(event)
        assert caption.kind == 'caption' and caption.data['text'] == 'Owner words.'
        # Replay of the item, even with another event id, cannot charge twice.
        adapter.convert({**event, 'event_id': 'duplicate'})
        records = ua.read_usage_records(host.ctx.drive_root)
        assert len(records) == 1
        row = records[0]
        assert row['model'] == 'gpt-4o-transcribe' and row['provider'] == 'openai_realtime'
        assert row['source'] == 'live_transcription_usage' and row['state'] == 'settled'
        assert row['task_id'] == host.ctx.task_id and row['root_task_id'] == host.ctx.task_id
        evidence = row['cost_evidence']
        assert evidence['call_id'] == host.turn.spec.call_id and evidence['connection_key'] == '1'
        assert evidence['item_id'] == 'owner-1' and evidence['content_index'] == 0
        assert evidence['raw_usage'] == ({} if missing == 'usage' else raw)
        assert 'response_key' not in evidence and not host.responses and not host.completed
        if missing is None:
            assert row['cost_usd'] == pytest.approx(.168)
            assert row['cost_coverage'] == 'complete' and not row['cost_final']
            with pytest.raises(ua.BudgetExceeded):
                live_usage.check_live_budget(engine='openai_realtime', model=host.turn.model,
                                             task_id=host.ctx.task_id, drive_root=host.ctx.drive_root)
        else:
            assert row['cost_usd'] is None and row['cost_coverage'] == 'partial'
            assert evidence['unpriced'], 'unknown usage or tariff is never a free transcription'
        # Already-incurred spend is retained even after the limit is exhausted.
        adapter.convert({**event, 'event_id': 'asr-2', 'item_id': 'owner-2'})
        assert len(ua.read_usage_records(host.ctx.drive_root)) == 2
        adapter.convert({**event, 'type': 'response.output_audio_transcript.done', 'item_id': 'spoken'})
        assert len(ua.read_usage_records(host.ctx.drive_root)) == 2, 'output captions have no separate ASR charge'


def test_a_component_sample_without_its_model_still_lands_priced_unknown(host_world):
    """R18 M2: the contract requires the model; a sample without one never ends the call."""
    import json
    from ouroboros import usage_accounting as ua

    host = host_world.host
    host._connection_accounting = True
    host.connection_number = 1
    with ua.usage_scope(ua.UsageScope(drive_root=host.ctx.drive_root, task_id=host.ctx.task_id,
                                    root_task_id=host.ctx.task_id)):
        host.observe('usage', json.dumps({'component': 'input_transcription', 'item_id': 'owner-1',
                                          'usage': {'audio_input_tokens': 3}, 'raw_usage': {}}))
        [row] = ua.read_usage_records(host.ctx.drive_root)
    assert row['source'] == 'live_transcription_usage' and row['model'] == ''
    assert row['cost_usd'] is None and row['cost_evidence']['unpriced'], 'unknown, never free'


# Recorded on 28.11.2026 (probes/languages, gpt-realtime-2.1): the transcription echo carries the whole object with
# "prompt": null added, an omitted language reads back null, and a refusal leaves the session open.
REFUSED_CODE = {"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_value",
    "message": "Invalid value: 'zz'.", "param": "session.audio.input.transcription.language", "event_id": None}}
REFUSED_FIELD = {"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_parameter",
    "message": "The 'languages' parameter is not supported for this model.",
    "param": "session.audio.input.transcription.languages", "event_id": None}}


def language_wire(answer=None, echoed=None):
    from tests.test_grok_openai_protocol import Wire

    class Recorded(Wire):
        def put(self, event):
            if event.get("type") == "session.updated":
                transcription = event["session"].get("audio", {}).get("input", {}).get("transcription")
                if transcription is not None:
                    shaped = {"model": transcription.get("model"), "language": transcription.get("language"), "prompt": None}
                    event = {**event, "session": {**event["session"], "audio": {**event["session"]["audio"], "input": {
                        **event["session"]["audio"]["input"], "transcription": {**shaped, **(echoed or {})}}}}}
            super().put(event)

        async def send(self, raw):
            import json
            event = json.loads(raw)
            if answer is not None and str(event.get("event_id", "")).startswith("languages_"):
                self.sent.append(event)
                refusal = copy.deepcopy(answer)
                if refusal["error"]["code"] == "invalid_value":
                    refusal["error"]["event_id"] = event["event_id"]
                self.put(refusal)
                return
            await super().send(raw)
    return Recorded()


async def opened(providers, wire, languages=()):
    from ouroboros.contracts.live_conversation import LiveOptions
    adapter, wire = make(providers, "openai", LiveOptions(languages=languages), wire=wire)
    await adapter.open()
    assert (await adapter.receive())[0].kind == "ready"
    return adapter


@pytest.mark.parametrize("languages,sent,taken", [
    (("ru-RU",), {"model": "gpt-4o-transcribe", "language": "ru"}, ("ru-RU",)),
    (("ru-RU", "en-US"), {"model": "gpt-4o-transcribe"}, ()),  # several: the engine detects each utterance
    ((), {"model": "gpt-4o-transcribe"}, ()),
])
def test_languages_change_in_place_alone_on_the_wire_until_echoed(providers, languages, sent, taken):
    from ouroboros.contracts.live_conversation import ContextUpdate

    async def run():
        wire = language_wire()
        adapter = await opened(providers, wire, ("de-DE",))
        try:
            setup = [event for event in wire.sent if event["type"] == "session.update"]
            update = asyncio.create_task(adapter.update_languages(languages))
            await asyncio.sleep(0.01)
            change = wire.sent[-1]
            assert change == {"type": "session.update", "event_id": "languages_1",
                              "session": {"type": "realtime", "audio": {"input": {"transcription": sent}}}}
            assert not adapter.renewal_ready and not update.done()
            await adapter.send(ContextUpdate(({"role": "user", "content": "Room words."},)))
            assert wire.sent[-1] is change, "nothing crosses the update before its echo"
            await adapter.receive()  # the echo
            assert await update == taken
            await adapter.receive()  # the held item's acknowledgement
            assert [event["type"] for event in wire.sent[-2:]] == ["conversation.item.create", "response.create"]
            # Instructions, tools and history are not rewritten: the cached prefix stays as it was set up.
            assert [event for event in wire.sent if event["type"] == "session.update"] == [*setup, change]
            assert not any(event.get("item", {}).get("role") == "system" for event in wire.sent[len(setup) + 2:])
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("answer", [REFUSED_CODE, REFUSED_FIELD, "other language echoed"])
def test_a_refused_or_disagreeing_language_update_changes_nothing_and_keeps_the_call(providers, answer):
    async def run():
        wire = language_wire(None if isinstance(answer, str) else answer,
                             echoed={"language": "en"} if isinstance(answer, str) else None)
        adapter = await opened(providers, wire)
        try:
            update = asyncio.create_task(adapter.update_languages(("ru-RU",)))
            await asyncio.sleep(0.01)
            events = await adapter.receive()
            assert not any(event.kind == "closed" for event in events) and not wire.closed
            assert await update is None
            assert adapter.renewal_ready, "the list now goes the renewal way"
        finally:
            await adapter.close()
    asyncio.run(run())


def test_an_error_that_is_not_the_language_updates_stays_fatal(providers):
    from ouroboros.contracts.realtime import RealtimeSetupError

    async def run():
        wire = language_wire({"type": "error", "error": {"type": "server_error", "code": "server_error",
                                                         "param": None, "event_id": None}})
        adapter = await opened(providers, wire)
        try:
            update = asyncio.create_task(adapter.update_languages(("ru-RU",)))
            await asyncio.sleep(0.01)
            with pytest.raises(RealtimeSetupError):
                await adapter.receive()
            update.cancel()
        finally:
            await adapter.close()
    asyncio.run(run())


def test_a_session_not_ready_takes_no_languages(providers):
    async def run():
        adapter, wire = make(providers, "openai")
        assert await adapter.update_languages(("ru-RU",)) is None and not wire.sent
    asyncio.run(run())
