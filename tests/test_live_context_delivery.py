"""Native queues cannot lend a completed answer's exposure to later writes."""
import asyncio
import json

import pytest

from ouroboros.contracts.live_conversation import ContextUpdate, LiveOptions
from ouroboros.realtime_setup import loaded_provider
from tests.test_live_conversation import ENGINES, VoiceWire, packages as packages
from tests.test_live_provider_boundaries import selected
from tests._live_world import host_world as host_world


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


def test_prepared_catchup_cannot_borrow_a_smaller_reply_than_the_open_socket(host_world, monkeypatch):
    from dataclasses import replace
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.realtime_setup import measure_session_wire
    from skills.gemini_live.plugin import GeminiSession
    from tests._live_wire import Wire

    host = host_world.host
    host._session_opened = True
    host.plan = replace(host.plan, output_reserve_tokens=65_536)
    host.model = replace(host.model, facts=replace(host.model.facts, basis='estimator', conversation_window_tokens=92_000))
    wire, options = Wire(), LiveOptions()

    def factory(*args, **kwargs):
        return GeminiSession(*args, connect=wire.connect, **kwargs)

    monkeypatch.setitem(host_world.provider, 'session', factory)

    async def run():
        lease = await host.prepare_session(options)
        try:
            allowance = wire.sent[0]['setup']['generationConfig']['maxOutputTokens']
            free = 92_000 - max(10_000, allowance) - lease.request.prompt_tokens_estimate
            tail = {'role': 'user', 'content': 'x' * (4 * (free + 1000))}
            await host.observe_context(tail)
            _, new_setup, _, _ = measure_session_wire(lease.view, lease.view.plan,
                [*lease.view.messages, tail], lease.view.schemas, factory, options, fresh=False, _room_fact=False)
            assert new_setup.max_completion_tokens < allowance and main_context_fit_refusal(new_setup) is None

            async def consume(session):
                pytest.fail('the extra row fits only on a newly opened connection')
            with pytest.raises(RealtimeSetupError, match='prepared_stale'):
                await lease.run(consume)
            assert len(wire.sent) == 1 and host._connection_view is None
        finally:
            await lease.close()

    asyncio.run(run())
