"""Real call HTTP admission with synthetic feature/provider and an isolated room.

The retired preview/funding protocol is deliberately absent. No socket binds a
port: TestClient reaches the same gateway route through its ASGI boundary.
"""
from dataclasses import replace

from starlette.applications import Starlette
from starlette.testclient import TestClient

from tests.test_live_conversation import host_world as host_world


def test_owner_call_door_mints_opening_id_and_rejoins_one_call(host_world, monkeypatch):
    from ouroboros.gateway.call_ws import call_routes
    from ouroboros.call_peers import close_hub, get_hub
    from supervisor import message_bus
    state = host_world
    calls, queued = [], []
    bridge = message_bus.LocalChatBridge()
    def accept(text, **values):
        calls.append((text, values))
        values.pop('dispatch')(text, task_metadata=values['task_metadata'])
    monkeypatch.setattr(bridge, 'ui_send', accept)
    monkeypatch.setattr(bridge, 'enqueue_local_message', lambda text, **values: queued.append((text, values)))
    monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
    payload = {'chat_id': state.task['chat_id'], 'client_message_id': 'untrusted-browser-id', 'feature': 'fixture:call'}
    try:
        with TestClient(Starlette(routes=call_routes())) as client:
            first = client.post('/api/call/open', json=payload)
            assert first.status_code == 200, first.text
            call_id = first.json()['call_id']
            assert first.json()['vision'] is True
            assert calls[0][0] == state.host.opening
            assert calls[0][1]['client_message_id'] == f'call:{call_id}:open'
            assert calls[0][1]['task_metadata']['host_fact'] == {'kind': 'call_open', 'call_id': call_id}
            assert queued[0][1]['task_metadata']['_realtime_turn'].model == 'fixture::voice-test'
            assert get_hub(call_id).spec.client_message_id == f'call:{call_id}:open'
            again = client.post('/api/call/open', json=payload)
            assert again.status_code == 200 and again.json()['joined'] is True
            assert again.json()['call_id'] == call_id and len(calls) == 1
    finally:
        if calls:
            close_hub(calls[0][1]['task_metadata']['execution_transport']['call_id'], 'fixture_done')


def test_call_door_reports_disabled_credential_and_model_refusals(host_world, monkeypatch):
    from ouroboros.gateway.call_ws import call_routes
    state = host_world
    payload = {'chat_id': state.task['chat_id'], 'client_message_id': 'owner-act', 'feature': 'fixture:call'}
    with TestClient(Starlette(routes=call_routes())) as client:
        monkeypatch.setitem(state.feature, 'configure', lambda: replace(state.config, enabled=False))
        assert client.post('/api/call/open', json=payload).json()['code'] == 'calls_disabled'
        monkeypatch.setitem(state.feature, 'configure', lambda: state.config)
        state.settings.clear()
        assert client.post('/api/call/open', json=payload).json()['code'] == 'credential'
        state.settings['TEST_CREDENTIAL'] = 'test-credential'
        monkeypatch.setitem(state.feature, 'configure', lambda: replace(state.config, model='unknown'))
        assert client.post('/api/call/open', json=payload).json()['code'] == 'model_unavailable'
