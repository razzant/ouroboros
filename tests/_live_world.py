"""``host_world``: a real conversation host for one live call, over its own copy of the shared installation.

The installation (``tests._memory_view_context.world``) is a tests/_worlds.py world: built once per run, cloned per test. A
parametrized ``host_world`` is the same installation under the real books, as an owner's mode and window read them; its
param is ``(owner_mode, window, basis)``. Everything else is this test's own: the fixture provider and feature, the call's
task, its tools and its first request.
"""
from __future__ import annotations

import asyncio
import functools
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._worlds import clone, world

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    silent_context_updates = True
    output_bound_tokens = 100
    renewal_ready = True

    def __init__(self, selected, options, *, observe):
        self.selected, self.options, self.observe = selected, options, observe
        self.wire = {'messages': json.loads(selected.messages_json), 'tools': json.loads(selected.tools_json),
                     'instructions': options.instructions + (' Speech recognition should expect: ' + ', '.join(options.languages) + '.' if options.languages else '')}
        self.opened = self.closed = False

    async def open(self):
        self.opened = True

    async def receive(self):
        return await asyncio.Future()

    async def close(self):
        self.closed = True


def _lay_installation(folder: Path, *, books: bool) -> None:
    from tests._memory_view_context import world as installation

    env, memory, rooms = installation(folder / 'world')
    if books:
        for relative in ('BIBLE.md', 'prompts/SYSTEM.md'):
            shutil.copyfile(ROOT / relative, env.repo_dir / relative)
        shutil.copytree(ROOT / 'docs', env.repo_dir / 'docs', dirs_exist_ok=True)
        memory.identity_path().write_text(memory._default_identity(), encoding='utf-8')
        memory.scratchpad_path().write_text(memory._default_scratchpad() + '\nCurrent project notes. ' * 180, encoding='utf-8')
    (folder / 'rooms.json').write_text(json.dumps(rooms), encoding='utf-8')


def installation_template(*, books: bool) -> Path:
    return world('live-books' if books else 'live-plain', functools.partial(_lay_installation, books=books))


def _own_installation(tmp_path: Path, *, books: bool) -> tuple:
    """``(env, memory, rooms)`` over this test's private copy."""
    from ouroboros.agent import Env
    from ouroboros.memory import Memory

    template = installation_template(books=books)
    root = clone(template / 'world', tmp_path / 'world')
    return (Env(repo_dir=root / 'repo', drive_root=root / 'drive'), Memory(drive_root=root / 'drive', repo_dir=root / 'repo'),
            json.loads((template / 'rooms.json').read_text(encoding='utf-8')))


@pytest.fixture
def host_world(tmp_path, monkeypatch, provider_catalog_offline, request):
    from ouroboros import config, context, extension_registry_state as registry
    from ouroboros.contracts.live_conversation import LiveConfig, CallTool
    from ouroboros.contracts.realtime import CatalogModel, ProviderCatalog, RouteFacts, RealtimeRoute
    from ouroboros.live_conversation import LiveConversationTurn, ConversationHost
    from ouroboros.call_protocol import CallSpec
    from ouroboros.call_peers import LiveCallHub
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_context import ToolContext
    from ouroboros.task_results import write_task_result, STATUS_RUNNING
    from tests.test_budget_fence_liveness import _publish
    from ouroboros.model_wait import task_model_wait_scope
    from supervisor import message_bus

    env, memory, rooms = _own_installation(tmp_path, books=hasattr(request, 'param'))
    owner_mode, window, basis = getattr(request, 'param', ('nano', 200000, 'provider'))
    monkeypatch.setattr(config, 'DATA_DIR', env.drive_root)
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(env.drive_root))
    monkeypatch.setattr(context, 'get_context_mode', lambda: owner_mode)
    monkeypatch.setattr(message_bus, 'DATA_DIR', env.drive_root)
    monkeypatch.setattr(message_bus, '_BRIDGE', message_bus.LocalChatBridge())
    settings = {'TEST_CREDENTIAL': 'test-credential'}
    async def language(host, arguments):
        host.state['languages'] = arguments['languages']
        return 'Language selection retained.'
    async def handler(host):
        pass
    call_tool = CallTool('set_test_language', 'Set the test language.',
        {'type': 'object', 'properties': {'languages': {'type': 'array', 'items': {'type': 'string'}}}}, language, silent=True)
    live_config = LiveConfig(enabled=True, provider='fixture:voice')
    feature = dict(skill='fixture', key='fixture:call', extension_generation='generation',
                   content_hash='fixture-content', configure=lambda: live_config, handler=handler, call_tools=(call_tool,))
    catalog = ProviderCatalog('Fixture engine', (CatalogModel('voice-test', 'Voice test',
        RouteFacts(window, 250000, 10000, basis, 'fixture', '2026-10-21'), default=True, vision=True),),
        ('TEST_CREDENTIAL',))
    sessions = []
    def factory(*args, **kw):
        value = FakeSession(*args, **kw)
        sessions.append(value)
        return value
    provider = dict(skill='fixture', extension_generation='generation', content_hash='fixture-content',
        catalog=catalog, route=RealtimeRoute('fixture_live', 'wss://example.invalid/live', 'fixture::'), session=factory)
    bundle = registry._ExtensionRegistrations(generation_digest='generation',
        realtime_providers=['fixture:voice'], conversation_features=['fixture:call'],
        api_instances=[SimpleNamespace(get_settings=lambda keys: dict(settings))])
    monkeypatch.setitem(registry._extensions, 'fixture', bundle)
    monkeypatch.setitem(registry._realtime_providers, 'fixture:voice', provider)
    monkeypatch.setitem(registry._conversation_features, 'fixture:call', feature)
    spec = CallSpec('fixture-call', rooms['alpha'], live_config.provider, '', '', client_message_id='call:fixture-call:open')
    turn = LiveConversationTurn(feature, live_config, spec, publish=lambda event: None)
    task = dict(id='fixture-call-task', type='task', text=turn.opening, _is_direct_chat=True,
        chat_id=spec.chat_id, model=turn.model, use_local_model=False,
        origin_message_ref={'chat_id': spec.chat_id, 'client_message_id': spec.client_message_id},
        metadata={'execution_transport': turn.request, 'host_fact': {'kind': 'call_open'}})
    write_task_result(env.drive_root, task['id'], STATUS_RUNNING, chat_id=spec.chat_id)
    _publish(env.drive_root)
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, branch_dev='test', task_id=task['id'],
        current_chat_id=spec.chat_id, is_direct_chat=True, task_metadata=task['metadata'], task_contract={})
    progress = []
    ctx.emit_progress_fn = lambda text, **kw: progress.append({'text': text, **kw})
    tools = ToolRegistry(env.repo_dir, env.drive_root)
    tools.set_context(ctx)
    messages, _ = context.build_llm_messages(env, memory, task, ctx=ctx, tool_schemas=tools.schemas(), route_resolver=turn.resolve_route)
    hub = LiveCallHub(turn.spec)
    with task_model_wait_scope(task=task, drive_root=env.drive_root, event_queue=None, worker_slot_held=False) as owner:
        ctx.model_wait_context, owner.tool_context = owner, ctx
        host = ConversationHost(turn, ctx, messages, tools, task, '', hub)
        yield SimpleNamespace(host=host, env=env, ctx=ctx, settings=settings, config=live_config, provider=provider,
            sessions=sessions, feature=feature, task=task, progress=progress)
        asyncio.run(host.finish())
        hub.close('fixture_done')
        registry._extensions.pop('fixture', None)
        registry._realtime_providers.pop('fixture:voice', None)
        registry._conversation_features.pop('fixture:call', None)
