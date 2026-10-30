"""The settings switch stops server-owned calls even without a page refresh."""
import asyncio
import contextvars
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros.contracts.live_conversation import LiveOptions
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('presence', [False, True])
@pytest.mark.parametrize('configuration', ['disabled', 'enabled', 'unreadable'])
def test_disable_during_preparation_prevents_later_provider_open(
        host_world, monkeypatch, tmp_path, configuration, presence):
    from ouroboros import call_peers
    from skills.live_conversation import plugin
    host = host_world.host
    host.turn.feature['key'] = 'live_conversation:call'
    if presence:
        host.turn.spec = replace(host.turn.spec, opened_by='skill', presence_binding_id='meeting-fixture')
        host.task['metadata']['presence'] = {'binding_id': 'meeting-fixture'}
        host.task.pop('origin_message_ref')
    monkeypatch.setattr(call_peers, 'get_hub', lambda _: host.hub)
    monkeypatch.setattr(call_peers, 'close_hub', lambda _, reason, **kw: host.hub.close(reason, **kw))
    routes, callbacks = {}, {}
    api = SimpleNamespace(get_state_dir=lambda: str(tmp_path / 'feature'),
        get_runtime_info=lambda: {'data_dir': str(host.ctx.drive_root)},
        register_live_conversation=lambda *a, **kw: callbacks.update(kw),
        register_settings_section=lambda *a, **kw: None,
        register_route=lambda name, handler, **kw: routes.update({name: handler}))
    plugin.register(api)

    def save(enabled):
        async def body(): return {'enabled': enabled, 'provider': host.turn.provider}
        return asyncio.run(routes['config'](SimpleNamespace(method='POST', json=body)))

    assert save(True).status_code == 200
    reads = []
    def configure():
        reads.append(True)
        return callbacks['configure']()
    host.turn.feature['configure'] = configure
    # The already-admitted turn prepares ordinary context before run() creates
    # its real host. A subsequent settings write must reach that later host.
    assert host.turn.config.enabled
    disabled = configuration == 'disabled'
    assert save(not disabled).status_code == 200
    assert callbacks['configure']().enabled is (not disabled)
    if configuration == 'unreadable':
        (tmp_path / 'feature/conversation.json').write_text('{invalid')
    opened = []
    async def handler(port):
        async def consume(session):
            opened.append(session.opened)
        await port.session(consume, LiveOptions())
    host.turn.feature['handler'] = handler
    result = host.turn.run(host.ctx, host.messages, host.tools, task=host.task, initial_effort='')
    assert bool(opened) is (not disabled), result[1]
    assert result[1]['reason_code'] == ('calls_disabled' if disabled else 'finished')
    assert len(reads) == 1  # admission-to-host check, never per-frame polling
    if disabled:
        assert not host_world.sessions
        assert host.hub._state['detail'] == 'Calls (beta) are off.'


@pytest.mark.parametrize('presence', [False, True])
def test_settings_disable_ends_background_and_presence_calls(host_world, monkeypatch, tmp_path, presence):
    from ouroboros import call_peers
    from skills.live_conversation import plugin
    host = host_world.host
    host.turn.feature['key'] = 'live_conversation:call'
    if presence:
        host.turn.spec = replace(host.turn.spec, opened_by='skill', presence_binding_id='meeting-fixture')
        host.task['metadata']['presence'] = {'binding_id': 'meeting-fixture'}
        host.task.pop('origin_message_ref')
    monkeypatch.setattr(call_peers, 'get_hub', lambda _: host.hub)
    monkeypatch.setattr(call_peers, 'close_hub', lambda _, reason, **kw: host.hub.close(reason, **kw))
    started, release, ended = threading.Event(), threading.Event(), threading.Event()
    sessions = []
    async def handler(port):
        async def consume(session):
            sessions.append(session)
            started.set()
            while not release.is_set():
                await asyncio.sleep(.01)
        try:
            await port.session(consume, LiveOptions())
        finally:
            ended.set()
    monkeypatch.setitem(host.turn.feature, 'handler', handler)
    routes = {}
    api = SimpleNamespace(get_state_dir=lambda: str(tmp_path / 'feature'),
        get_runtime_info=lambda: {'data_dir': str(host.ctx.drive_root)},
        register_live_conversation=lambda *a, **kw: None, register_settings_section=lambda *a, **kw: None,
        register_route=lambda name, handler, **kw: routes.update({name: handler}))
    plugin.register(api)
    def request(value=None):
        async def body(): return value
        return asyncio.run(routes['config'](SimpleNamespace(method='POST' if value is not None else 'GET', json=body)))
    value = {'enabled': True, 'provider': host.turn.provider}
    assert request(value).status_code == 200
    copied = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(copied.run, host.turn.run, host.ctx, host.messages, host.tools,
                               task=host.task, initial_effort='')
        try:
            assert started.wait(15)
            assert request(value).status_code == 200
            assert request({'enabled': 'invalid'}).status_code == 400
            (tmp_path / 'feature/conversation.json').write_text('{invalid')
            with pytest.raises(ValueError):
                request()  # a failed configuration read is not an owner disable
            assert not ended.wait(.15)
            response = request({**value, 'enabled': False})
            assert response.status_code == 200 and json.loads(response.body)['enabled'] is False
            assert ended.wait(5), 'successful beta disable left the server call running'
        finally:
            release.set()
            result = future.result(timeout=15)
    assert result[1]['reason_code'] == 'calls_disabled'
    assert sessions[-1].closed
    assert host.hub._state['state'] == 'closed'
    assert host.hub._state['reason'] == 'calls_disabled'
    assert host.hub._state['detail'] == 'Calls (beta) are off.'
