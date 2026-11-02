"""Native queues cannot lend a completed answer's exposure to later writes."""
import asyncio
import json

import pytest

from ouroboros.contracts.live_conversation import ContextUpdate, LiveOptions
from ouroboros.realtime_setup import loaded_provider
from tests.test_live_conversation import ENGINES, VoiceWire, packages as packages
from tests.test_live_provider_boundaries import selected
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('name,engine', ENGINES)
def test_queued_update_is_exposed_by_the_next_answer_only(packages, name, engine):
    async def run():
        wire = VoiceWire(engine, None)
        session = loaded_provider(name + ':' + engine)['session'](
            selected(), LiveOptions(), connect=wire.connect, observe=lambda *_: None)
        row = {'role': 'user', 'content': 'New words queued behind the answer.', '_host_context_kind':'owner_words'}
        try:
            await session.open()
            await session.receive()
            if engine == 'gemini':
                session.turn_busy = True
                session.response = 'old-answer'
                session.response_complete = False
                session.usage = {'promptTokenCount': 100}
            else:
                session.active_response = 'old-answer'
            await session.send(ContextUpdate((row,), respond=False))
            assert row not in session.held_context_rows
            wire.put({'serverContent': {'turnComplete': True}} if engine == 'gemini' else
                {'type': 'response.done', 'response': {'id':'old-answer', 'status':'completed', 'usage': {}, 'output':[]}})
            events = await session.receive()
            assert any(event.kind == 'response_done' for event in events)
            assert row not in session.response_context_rows['old-answer']
            assert row in session.held_context_rows
            session.capture_response_context('next-answer')
            assert row in session.response_context_rows['next-answer']
            assert '_host_context_kind' not in json.dumps(wire.sent)
        finally:
            await session.close()
    asyncio.run(run())


def test_prepared_catchup_prices_images_already_held_without_retroactive_projection(host_world):
    from ouroboros.realtime_setup import measure_session_wire, rows_for_engine
    from tests.test_context_image_floors import images
    host = host_world.host
    host._session_opened = True
    host.messages.extend(images(2))
    async def run():
        lease = await host.prepare_session(LiveOptions())
        async def send(command):
            pass
        lease.session.send = send
        tail = images(1)
        host.messages.extend(tail)
        host._context_changed()
        async def consume(session):
            # The first connection already holds its projected prefix. A new
            # image cannot retroactively turn those delivered pixels into text.
            # Price delivered rows without adding another fresh-fit manifest.
            delivered = [*json.loads(session.selected.messages_json), *rows_for_engine(tail, vision=True)]
            _, actual, _, _ = measure_session_wire(host, host.plan, delivered, host.schemas,
                host_world.provider['session'], LiveOptions(), _room_fact=False, fresh=False)
            assert host._prepared_view[0]['fit']['measurement']['estimated_input_tokens'] == actual.prompt_tokens_estimate
        try:
            await lease.run(consume)
        finally:
            await lease.close()
    asyncio.run(run())
