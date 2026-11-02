# Reviewer probes of R15 (rev-astra2); silence and ready-boundary pins follow the 28.10 owner decision.
"""R15 reviewer probes. Real adapters, synthetic peers/data, no provider access."""
import asyncio
import base64
import copy
import io
import queue
from dataclasses import replace

import pytest

from tests.test_live_conversation import host_world as host_world
from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, Wire
from tests.test_gemini_live_boundaries import gemini as gemini, Wire as GeminiWire


def image_row():
    return {'role': 'user', 'content': [
        {'type': 'text', 'text': '[image: shot.png; original: /tmp/synthetic-shot.png]'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='},
         '_caption': '[image: shot.png; original: /tmp/synthetic-shot.png]',
         '_source_path': '/tmp/synthetic-shot.png'}]}


def adapter_factory(engine, providers, gemini):
    def factory(selected, options, *, observe):
        wire = GeminiWire() if engine == 'gemini' else Wire(engine)
        ctor = gemini.GeminiSession if engine == 'gemini' else providers[engine].session
        return ctor(selected, options, connect=wire.connect, observe=observe)
    return factory


@pytest.mark.parametrize('engine', ['grok', 'openai', 'gemini'])
def test_live_mailbox_image_reaches_engine_without_closing(host_world, providers, gemini, engine):
    from PIL import Image
    from ouroboros.contracts.live_conversation import LiveOptions
    from skills.live_conversation.driver import _Call
    host = host_world.host
    host.model = replace(host.model, vision=engine != 'grok')
    host_world.provider['session'] = adapter_factory(engine, providers, gemini)
    blob = io.BytesIO()
    Image.new('RGB', (2, 2), 'blue').save(blob, format='PNG')
    host.incoming = queue.Queue()
    host.incoming.put({'text': 'Look at this while we talk.', 'client_message_id': 'typed-image',
                       'image_base64': base64.b64encode(blob.getvalue()).decode(), 'image_caption': 'shot.png'})

    async def consume(session):
        await session.receive()  # ready
        call = _Call(host)
        call.session = session
        call.connected.set()
        rows = await host.mailbox()
        assert any(p.get('type') == 'image_url' for r in host.messages
                   if isinstance(r.get('content'), list) for p in r['content'])
        assert not host.renewal_due()
        await call.context_update(rows)
        sent = [r for r in session.ws.sent if r.get('type') == 'conversation.item.create'
                or 'clientContent' in r]
        assert sent
        if engine == 'grok':
            assert 'does not see images' in str(sent)
    asyncio.run(host.session(consume, LiveOptions()))


@pytest.mark.parametrize('engine', ['grok', 'openai', 'gemini'])
def test_authored_context_view_after_screenshot_keeps_call_alive(host_world, providers, gemini, engine):
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.tools.compact_context import _compact_context
    host = host_world.host
    host.model = replace(host.model, vision=engine != 'grok')
    host_world.provider['session'] = adapter_factory(engine, providers, gemini)

    async def consume(session):
        await session.receive()
        async def shoot(_host, _args):
            host.messages.append(copy.deepcopy(image_row()))
            return 'Screenshot retained.'
        host.call_tools['set_test_language'] = replace(host.call_tools['set_test_language'], handler=shoot)
        result = await host.tool_batch([{'id': 'shot', 'type': 'function', 'function': {
            'name': 'set_test_language', 'arguments': '{}'}}])
        assert ('does not see images' in str(result)) == (engine == 'grok')
        assert 'requested' in _compact_context(host.ctx, working_note='I inspected the screenshot.', keep_unit_ids=[])
        await host.apply_context_view()
        assert host.ctx._context_view_receipt['status'] == 'applied'
    asyncio.run(host.session(consume, LiveOptions()))


@pytest.mark.parametrize('resume', [False, True])
def test_peer_history_survives_reconnection(resume):
    from skills.live_conversation import driver
    from tests.test_live_conversation_driver import Host, Session
    host = Host()
    call = driver._Call(host)
    call.session = Session()
    call.connected.set()  # this probe represents an already-ready physical connection
    call.request_renewal(resuming=resume)
    host.state['resume_handle'] = 'synthetic-resume'
    async def scenario():
        await call.peers()
        assert call.ever_attached
        call.renewing = call.resume = call.provider_closed = False
        host.hub.peers = lambda: []
        call.empty_since -= driver.EMPTY_CALL_GRACE_S + 1
        with pytest.raises(driver.RealtimeSetupError, match='peers_left'):
            await call.peers()
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['grok', 'openai', 'gemini'])
def test_provider_close_words_survive_after_ready(host_world, providers, gemini, engine):
    from ouroboros.contracts.live_conversation import LiveOptions
    from skills.live_conversation.driver import _Call
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close
    host = host_world.host
    host.model = replace(host.model, vision=engine != 'grok')
    host_world.provider['session'] = adapter_factory(engine, providers, gemini)

    async def consume(session):
        await session.receive()
        async def closed():
            raise ConnectionClosedError(Close(1011, 'Your prepayment credits are depleted.'), None)
        session.read = closed
        call = _Call(host)
        call.session = session
        with pytest.raises(Exception) as caught:
            await call.provider()
        assert 'Your prepayment credits are depleted.' in str(caught.value)
    asyncio.run(host.session(consume, LiveOptions()))


def test_silent_first_language_refit_preserves_silence_when_capacity_also_requires_refit():
    from skills.live_conversation.driver import _Call
    from ouroboros.contracts.live_conversation import ToolBatch
    from tests.test_live_conversation_driver import Host, Session
    host = Host()
    call = _Call(host)
    call.session = Session()
    call.connected.set()
    async def batch(calls):
        host.state['refit'] = True
        host.due = True
        return ToolBatch([{'role': 'tool', 'tool_call_id': 'lang', 'content': 'Recorded.'}], silent=['lang'])
    host.tool_batch = batch
    async def scenario():
        await call.tool_queue.put({'id': 'lang', 'function': {'name': 'set_call_languages', 'arguments': '{}'}})
        task = asyncio.create_task(call.tools())
        try:
            await asyncio.wait_for(call.tool_queue.join(), 2)
            assert call.tool_refit.is_set()
            assert not call.pending_offers, 'A completed silent result offers nothing on either refit path.'
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('first', [False, True])
def test_offline_work_room_arithmetic_describes_same_candidate(host_world, first):
    from devtools._context_fit_session import measure_session
    from ouroboros.contracts.live_conversation import LiveOptions
    host = host_world.host
    host._session_opened = not first
    result = measure_session(host, host_world.provider['session'], LiveOptions())
    assert result['arithmetic']['start_work_room_tokens'] == result['start_work_room_tokens']
    assert result['arithmetic']['room_above_start_tokens'] == result['margin_tokens'] - result['start_work_room_tokens']
