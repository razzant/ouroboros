"""Real websocket protocols over a paused memory transport; no provider or socket."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from websockets.frames import Opcode
from websockets.http11 import Request
from websockets.server import ServerProtocol

from ouroboros.contracts.live_conversation import LiveOptions
from tests.test_gemini_live_boundaries import gemini as gemini
from tests.test_grok_openai_protocol import (
    deny_network as deny_network, providers as providers, selected,
)


class MemoryTransport(asyncio.Transport):
    """Handshake and control frames are real; flow-control callbacks are causal gates."""
    def __init__(self, client):
        self.client = client
        self.server = ServerProtocol(max_size=None)
        self.loop = asyncio.get_running_loop()
        self.frames = asyncio.Queue()
        self.request = None
        self.closed = self.paused = self.drop_pongs = False

    def write(self, data):
        self.server.receive_data(data)
        for event in self.server.events_received():
            if isinstance(event, Request):
                self.request = event
                self.server.send_response(self.server.accept(event))
            else:
                self.frames.put_nowait(event)
        self.deliver()

    def deliver(self):
        for data in self.server.data_to_send():
            if not data:
                self.loop.call_soon(self.close)
            elif not (self.drop_pongs and data[0] & 0x0f == Opcode.PONG):
                self.loop.call_soon(self.client.data_received, data)

    async def frame(self, opcode):
        async def wait():
            while True:
                frame = await self.frames.get()
                if frame.opcode == opcode:
                    return frame
        return await asyncio.wait_for(wait(), 2)

    def pause(self):
        self.paused = True
        self.client.pause_writing()

    def resume(self):
        if self.paused and not self.closed:
            self.paused = False
            self.client.resume_writing()

    def close(self):
        if not self.closed:
            self.closed = True
            self.loop.call_soon(self.client.connection_lost, None)

    abort = close
    write_eof = close

    def is_closing(self): return self.closed
    def can_write_eof(self): return False
    def set_write_buffer_limits(self, *args, **kwargs): pass
    def pause_reading(self): pass
    def resume_reading(self): pass
    def get_extra_info(self, name, default=None): return default


@pytest.fixture(params=['openai', 'grok', 'gemini'])
def provider(request, providers, gemini):
    module = gemini if request.param == 'gemini' else providers[request.param].module
    registered = {}
    module.register(SimpleNamespace(
        get_settings=lambda keys: {key: 'synthetic-credential' for key in keys},
        register_realtime_provider=lambda name, **values: registered.update(values)))
    return request.param, registered['session']


@asynccontextmanager
async def connected(provider, monkeypatch, *, interval=.01):
    name, factory = provider
    loop = asyncio.get_running_loop()
    transports = []

    async def create_connection(protocol_factory, *args, **kwargs):
        client = protocol_factory()
        assert client.ping_interval == client.ping_timeout == 20
        assert client.close_timeout == 3
        client.ping_interval = interval
        transport = MemoryTransport(client)
        transports.append(transport)
        client.connection_made(transport)
        return transport, client

    monkeypatch.setattr(loop, 'create_connection', create_connection)
    session = factory(selected(), LiveOptions(), observe=lambda *_: None)
    try:
        await session.connect_wire()
        await asyncio.sleep(0)  # let the native keepalive begin its first interval
        transport, = transports
        header = 'x-goog-api-key' if name == 'gemini' else 'Authorization'
        expected = 'synthetic-credential' if name == 'gemini' else 'Bearer synthetic-credential'
        assert transport.request.headers[header] == expected
        yield session, transport
    finally:
        for transport in transports:
            transport.resume()
        await session.close()


async def settle(tasks):
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_paused_audio_and_second_writer_coexist_with_keepalive_and_peer_ping(provider, monkeypatch, caplog):
    async def run():
        async with connected(provider, monkeypatch) as (session, transport):
            transport.pause()
            audio = asyncio.create_task(session.write({'audio': 'continuous-input'}))
            tools = asyncio.create_task(session.write({'tool_output': 'ready'}))
            try:
                await transport.frame(Opcode.TEXT)  # first writer is now waiting for drain
                await transport.frame(Opcode.PING)  # the library's keepalive writes concurrently
                transport.server.send_ping(b'provider-check')
                transport.deliver()
                assert (await transport.frame(Opcode.PONG)).data == b'provider-check'
                keepalive = getattr(session.ws, 'keepalive_task', None)
                if keepalive is None:
                    keepalive = session.ws.keepalive_ping_task
                assert not keepalive.done(), 'keepalive died while an audio send was blocked'
                assert not transport.closed
                assert not audio.done() and not tools.done()
                transport.resume()
                await asyncio.wait_for(asyncio.gather(audio, tools), 2)
                # A further ping/pong and application event prove the connection stayed useful.
                await asyncio.wait_for(await session.ws.ping(b'after-pressure'), 2)
                await session.write({'response': 'continue'})
                assert 'keepalive ping failed' not in caplog.text
                assert 'data transfer failed' not in caplog.text
            finally:
                await settle([audio, tools])
    asyncio.run(run())


def test_connector_preserves_headers_lifetime_limits_and_unbounded_messages(provider, monkeypatch):
    import websockets.asyncio.client

    original = websockets.asyncio.client.connect
    calls = []

    def capture(uri, **kwargs):
        calls.append((uri, kwargs))
        return original(uri, **kwargs)

    monkeypatch.setattr(websockets.asyncio.client, 'connect', capture)

    async def run():
        async with connected(provider, monkeypatch) as (session, transport):
            (uri, options), = calls
            assert uri.startswith('wss://')
            assert options['open_timeout'] == 30 and options['close_timeout'] == 3
            assert options['proxy'] is None  # legacy connections also went directly
            assert options['max_size'] is None
            assert options['ping_interval'] == options['ping_timeout'] == 20
            assert options['additional_headers']  # verified on the actual handshake above
            # Exceed the library's default 1 MiB receive limit on the real protocol.
            event = {'text': 'x' * (2**20 + 1)}
            transport.server.send_text(json.dumps(event).encode())
            transport.deliver()
            assert await asyncio.wait_for(session.read(), 2) == event
    asyncio.run(run())


@pytest.mark.parametrize('cancel', ['active', 'queued'])
def test_cancelled_sender_removes_only_its_drain_waiter(provider, monkeypatch, cancel):
    async def run():
        async with connected(provider, monkeypatch) as (session, transport):
            transport.pause()
            first = asyncio.create_task(session.write({'audio': 'first'}))
            second = asyncio.create_task(session.write({'audio': 'second'}))
            try:
                await transport.frame(Opcode.TEXT)
                await transport.frame(Opcode.PING)
                victim, survivor = (first, second) if cancel == 'active' else (second, first)
                victim.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await victim
                if cancel == 'active':
                    await transport.frame(Opcode.TEXT)
                assert len(session.ws.drain_waiters) == 2  # survivor and keepalive
                assert not any(waiter.done() for waiter in session.ws.drain_waiters)
                transport.resume()
                await asyncio.wait_for(survivor, 2)
                await asyncio.wait_for(await session.ws.ping(b'after-cancel'), 2)
                assert not session.ws.drain_waiters
                await session.write({'response': 'still-open'})
            finally:
                await settle([first, second])
    asyncio.run(run())


@pytest.mark.parametrize('responsive', [False, True])
def test_default_keepalive_detects_dead_peer_at_twenty_plus_twenty_seconds(provider, monkeypatch, responsive):
    async def run():
        loop = asyncio.get_running_loop()
        clock, offset = loop.time, 0
        monkeypatch.setattr(loop, 'time', lambda: clock() + offset)

        async def advance(seconds):
            nonlocal offset
            offset += seconds
            # Drain the ready/timer callbacks without sleeping through the production deadline.
            for _ in range(6):
                checkpoint = loop.create_future()
                loop.call_later(0, checkpoint.set_result, None)
                await checkpoint

        async with connected(provider, monkeypatch, interval=20) as (session, transport):
            transport.drop_pongs = not responsive
            await advance(19)
            assert transport.frames.empty() and not transport.closed
            await advance(1.1)
            await transport.frame(Opcode.PING)
            await advance(19)
            assert not transport.closed  # the pong deadline hasn't elapsed
            await advance(1.1)
            if responsive:
                assert not transport.closed
                await session.write({'response': 'still-open'})
            else:
                frame = await transport.frame(Opcode.CLOSE)
                assert frame.data[:2] == (1011).to_bytes(2, 'big')
                assert frame.data[2:] == b'keepalive ping timeout'
                await asyncio.wait_for(session.ws.wait_closed(), 2)
                assert transport.closed
    asyncio.run(run())
