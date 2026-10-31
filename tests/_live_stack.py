"""The admitted call stack over disposable stores and loopback native providers.

Only the page/device and remote provider are simulated. Built-in installation,
gateway, hub, host, skill consumer, adapters, fit, tools and stores are real.
"""
import asyncio
import inspect
import json
import queue
import socket
import hashlib
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from ouroboros import context, extension_loader, realtime_session
from ouroboros.call_peers import open_owner_hub, close_hub
from ouroboros.call_protocol import CallSpec
from ouroboros.gateway.call_ws import serve_peer
from ouroboros.memory import Memory
from ouroboros.skill_builtin import install_builtin_skill
from ouroboros.skill_loader import (find_skill, save_enabled, requested_skill_permissions,
                                    save_skill_grants, skill_state_dir_path)
from tests._live_stack_page import Page
from tests._live_stack_wire import ProviderServer

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ('live_conversation', 'gemini_live', 'openai_realtime', 'grok_realtime')
PROVIDERS = {'gemini': 'gemini_live:gemini', 'openai': 'openai_realtime:openai', 'grok': 'grok_realtime:grok'}


def install(root):
    settings = {key: 'synthetic-unused-key' for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'XAI_API_KEY')}
    for name in SKILLS:
        result = install_builtin_skill(name, drive_root=root, seed_dir=ROOT / 'skills', turn_on_when_ready=True)
        assert result['ok'], result
        save_enabled(root, name, True)
        skill = find_skill(root, name)
        permissions = requested_skill_permissions(skill.manifest.permissions, skill.manifest.subscribe_events)
        keys = list(skill.manifest.env_from_settings)
        save_skill_grants(root, name, keys, content_hash=skill.content_hash, requested_keys=keys,
                          granted_permissions=permissions, requested_permissions=permissions)
        assert extension_loader.load_extension(skill, lambda: settings, drive_root=root) is None
    from ouroboros.live_conversation import loaded_feature
    return loaded_feature('live_conversation:call')


class Stack:
    def __init__(self, state, engine):
        self.state, self.engine, self.root = state, engine, state.env.drive_root
        self.provider, self.page = ProviderServer(engine), Page()
        self.host = self.call = None
        self.mail, self.events = queue.Queue(), queue.Queue()
        self.running = None

    async def wait(self, predicate, timeout=120):
        # A hang detector, not a scenario latency assertion. Fitting, SQLite
        # and the provider loop share two workers on the loaded sprint host.
        async def pending():
            while not predicate():
                if self.running and self.running.done():
                    raise AssertionError(('call ended before scenario boundary', self.running.result()))
                assert not self.provider.errors, self.provider.errors
                await asyncio.sleep(.01)
        try:
            await asyncio.wait_for(pending(), timeout)
        except asyncio.TimeoutError:
            raise AssertionError({'boundary': 'timed out', 'blockers': self.call.quiet_blockers() if self.call else None,
                'connections': len(self.provider.connections), 'renewing': self.call.renewing if self.call else None,
                'refit': self.host.state.get('refit') if self.host else None,
                'last_native': [list(v) if self.engine == 'gemini' else v.get('type')
                                for v in self.wire.received[-8:]]}) from None

    @property
    def wire(self): return self.provider.connections[-1]

    async def quiet(self):
        await self.wait(lambda: self.call and self.call.connected.is_set() and self.call.quiet()
                        and not self.call.renewing and not self.call.tool_refit.is_set()
                        and not self.host.state.get('refit'))

    async def owner_says(self, text):
        before = len(self.wire.audio)
        item = f'owner-{len(self.page.controls)}'
        self.wire.input_start(item)
        await self.wait(lambda: len(self.wire.audio) >= before + 960)
        self.wire.input_end(item, text)
        await self.wait(lambda: any(r.get('text') == text and r.get('final') for r in self.page.controls))

    async def tool(self, cid, name, args):
        self.wire.tool(cid, name, args)
        await self.wait(lambda: any(c.get('id') == cid for c in self.host.trace.get('tool_calls', []))
                        or cid in self.wire.outputs or self.call.tool_refit.is_set())

    def transcript(self):
        from tests._dialogue import chat_rows
        return chat_rows(self.root)

    def journal(self):
        return [json.loads(v) for v in self.host.wire_path.read_text().splitlines()]


@asynccontextmanager
async def stack(state, monkeypatch, engine, *, thinking=False, languages=False):
    from websockets.asyncio.server import serve
    from ouroboros import live_conversation
    from ouroboros import pricing_catalog
    from ouroboros.contracts.live_conversation import LiveConfig
    from ouroboros.utils import utc_now_iso
    from tests._cost_driver import loopback_only

    # Native provider URLs are replaced at the sole transport boundary. No DNS or
    # connect to a non-loopback address can escape even if that patch is removed.
    connect = socket.socket.connect
    def local_only(sock, address):
        assert not isinstance(address, tuple) or address[0] in ('127.0.0.1', '::1'), 'nonlocal call test connection'
        return connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', local_only)
    monkeypatch.setattr(socket, 'getaddrinfo', loopback_only(SimpleNamespace(note_name=lambda *a: None), socket.getaddrinfo))
    # The public price feed is also a remote dependency; an explicit synthetic
    # cache leaves accounting/settlement real without any catalog network I/O.
    raw = json.dumps({'synthetic-price-row': {'litellm_provider': 'synthetic'}})
    monkeypatch.setattr(pricing_catalog, '_fetch_catalog', lambda: pricing_catalog._decode(dict(
        url=pricing_catalog.CATALOG_URL, content=raw, fetched_at=utc_now_iso(),
        sha256=hashlib.sha256(raw.encode()).hexdigest())))
    monkeypatch.setenv('OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS', 'true')
    run = Stack(state, engine)
    run.provider.thinking = thinking
    feature = install(run.root)
    config = LiveConfig(enabled=True, provider=PROVIDERS[engine],
        model='gemini-3.8-live-extended-thinking' if thinking else '', search=False, interruptions=True,
        instructions='Synthetic owner call instruction, kept verbatim.')
    config_dir = skill_state_dir_path(run.root, 'live_conversation')
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / 'conversation.json').write_text(json.dumps(asdict(config)))
    if languages:
        run.provider.script.append(('tool', 'opening-language', 'set_call_languages', {'languages': ['en-US']}))
    run.provider.script.append(('say', 'Synthetic greeting heard.'))
    spec = CallSpec('stack-' + engine, state.ctx.current_chat_id, config.provider, config.model,
                    client_message_id='stack-opening')
    turn = live_conversation.LiveConversationTurn(feature, config, spec, publish=run.events.put)
    task = {**state.task, 'model': turn.model, 'text': turn.opening,
        'origin_message_ref': {'chat_id': spec.chat_id, 'client_message_id': spec.client_message_id},
        'metadata': {'execution_transport': turn.request, 'host_fact': {'kind': 'call_open'}}}
    state.ctx.task_metadata = task['metadata']
    state.ctx.event_queue = run.events
    memory = Memory(state.env.drive_root, state.env.repo_dir)
    messages, _ = context.build_llm_messages(state.env, memory, task, ctx=state.ctx,
        tool_schemas=state.host.tools.schemas(), route_resolver=turn.resolve_route)
    real_host = live_conversation.ConversationHost
    class ObservedHost(real_host):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            run.host = self
    monkeypatch.setattr(live_conversation, 'ConversationHost', ObservedHost)
    feature_run = inspect.getclosurevars(inspect.unwrap(feature['handler'])).nonlocals['run']
    real_call = feature_run.__globals__['_Call']
    class ObservedCall(real_call):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            run.call = self
    monkeypatch.setitem(feature_run.__globals__, '_Call', ObservedCall)
    hub, fresh = open_owner_hub(turn.spec)
    assert fresh
    connector = realtime_session.connect_provider
    async with serve(run.provider.accept, '127.0.0.1', 0) as server:
        port = server.sockets[0].getsockname()[1]
        assert port not in (8797, 8798, 8799)
        monkeypatch.setattr(realtime_session, 'connect_provider',
                            lambda uri, *, headers: connector(f'ws://127.0.0.1:{port}', headers={}))
        gateway = asyncio.create_task(serve_peer(run.page, hub, role='owner', video=True))
        async def playback():
            while not hub.closed:
                run.page.tick(capture=run.page.capturing())
                await asyncio.sleep(.02)
        pump = asyncio.create_task(playback())
        run.running = asyncio.create_task(asyncio.to_thread(turn.run, state.ctx, messages, state.host.tools,
            task=task, initial_effort='', incoming_messages=run.mail, event_queue=run.events))
        try:
            await run.wait(lambda: any(v.get('text') == 'Synthetic greeting heard.' and v.get('final')
                                      for v in run.page.controls))
            await run.quiet()
            yield run
        finally:
            run.page.end()
            try:
                # A hang detector, not a timing claim: teardown joins effects, the journal and
                # settlement, which a loaded machine slows well past one scenario step.
                run.result = await asyncio.wait_for(asyncio.shield(run.running), 120)
                await asyncio.wait_for(gateway, 3)
            finally:
                close_hub(turn.spec.call_id, 'test_cleanup')
                pump.cancel(); gateway.cancel()
                await asyncio.gather(pump, gateway, return_exceptions=True)
                for name in reversed(SKILLS):
                    extension_loader.unload_extension(name)
