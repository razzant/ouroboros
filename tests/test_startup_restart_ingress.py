"""Startup Restart shares the owner, while each transport keeps its authority and facts."""
from types import SimpleNamespace
import json
import threading

import pytest
from starlette.testclient import TestClient

from tests.test_onboarding_process_controls import startup_controls as startup_controls

pytestmark = pytest.mark.serial


def _join_restarts():
    for thread in threading.enumerate():
        if thread.name == 'startup-owner-restart':
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize('ready_at_admission', [True, False])
def test_ws_typed_restart_preserves_surface_when_supervisor_becomes_ready(
        startup_controls, monkeypatch, ready_at_admission):
    from supervisor import message_bus

    obj = startup_controls
    calls, queued = [], threading.Event()
    bridge = message_bus.LocalChatBridge()
    echoes = []
    bridge._broadcast_fn = echoes.append
    enqueue = bridge.enqueue_local_message

    def capture(text, **kwargs):
        calls.append((text, kwargs))
        enqueue(text, **kwargs)
        queued.set()

    monkeypatch.setattr(bridge, 'enqueue_local_message', capture)
    monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
    monkeypatch.setattr(obj.server, '_supervisor_thread', SimpleNamespace(is_alive=lambda: True))
    if ready_at_admission:
        obj.server._supervisor_ready.set()

    def admit(command, **kwargs):
        action = obj.server._startup_owner_command(command, **kwargs)
        obj.server._supervisor_ready.set()
        return action

    obj.app.state.startup_owner_command = admit
    monkeypatch.setattr(obj.server, '_perform_owner_restart', lambda *_: pytest.fail('ready consumer owns Restart'))
    with obj.client.websocket_connect('/ws') as socket:
        socket.send_json({'type': 'chat', 'content': '/restart', 'sender_session_id': 'the-sender',
                          'client_message_id': 'the-message', 'chat_id': 23, 'project_id': 'project',
                          'client_surface': {'pywebview': True, 'viewport': {'w': 390, 'h': 900}}})
        assert queued.wait(3)
    assert len(calls) == 1
    text, kwargs = calls[0]
    assert text == '/restart' and kwargs['source'] == 'web' and kwargs['user_id'] == 1
    assert kwargs['sender_session_id'] == 'the-sender' and kwargs['client_message_id'] == 'the-message'
    assert kwargs['chat_id'] == 23 and kwargs['task_metadata']['project_id'] == 'project'
    assert kwargs['task_metadata']['client_surface']['pywebview'] is True
    assert kwargs['task_metadata']['client_surface']['viewport'] == {'w': 390, 'h': 900}
    assert kwargs['task_metadata']['client_surface']['received_at']
    path = obj.data / 'logs/chat.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1 and rows[0]['client_message_id'] == 'the-message'
    assert kwargs['accepted_source_row'] == rows[0]
    assert echoes[0]['ingress_accepted'] is True and echoes[0]['sender_session_id'] == 'the-sender'
    update = bridge.get_updates(0, timeout=0)[0]['message']
    assert message_bus.record_inbound_message(bridge, update, chat_id=23, user_id=1,
        client_message_id='the-message', text='/restart', ts=rows[0]['ts']) == kwargs['accepted_source_ref']
    assert len(path.read_text(encoding='utf-8').splitlines()) == 1, 'ready handoff must not re-log the accepted message'


@pytest.mark.parametrize('case', ['known', 'unknown', 'unbound', 'foreign', 'changed_after_admission', 'ready', 'ready_after_admission'])
def test_host_service_restart_uses_current_owner_or_preserves_normal_intake(startup_controls, monkeypatch, case):
    from ouroboros.gateway.host_service import create_host_service_app
    from supervisor import message_bus, state
    from tests.test_host_service_api import _seed_token
    from tests.test_transport_commands import Ctx

    obj = startup_controls
    live = state.ensure_state_defaults({'owner_external_id': 17, 'owner_external_chat_id': 23})
    if case == 'unknown':
        live.pop('owner_external_id')
    elif case == 'unbound':
        live.update(owner_external_id=None, owner_external_chat_id=None)
    elif case == 'foreign':
        live.update(owner_external_id=99, owner_external_chat_id=99)
    monkeypatch.setattr(state, 'load_state', lambda: dict(live))
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
    monkeypatch.setattr(message_bus, 'DATA_DIR', obj.data)
    monkeypatch.setattr(obj.server, '_supervisor_thread', SimpleNamespace(is_alive=lambda: True))
    if case == 'ready':
        obj.server._supervisor_ready.set()
    stopped = []
    monkeypatch.setattr(obj.server, '_perform_owner_restart', lambda *_: (stopped.append(True) or True, 'ok'))

    def admit(command, **kwargs):
        action = obj.server._startup_owner_command(command, **kwargs)
        if case == 'changed_after_admission':
            live.update(owner_external_id=99, owner_external_chat_id=99)
        if case == 'ready_after_admission':
            obj.server._supervisor_ready.set()
        return action

    bridge.startup_owner_command = admit
    _seed_token(obj.data, permissions=['inject_chat'])
    app = create_host_service_app(obj.data, bridge_getter=lambda: bridge)
    with TestClient(app) as client:
        payload = {'text': '/restart', 'user_id': 17, 'chat_id': 23,
                   'sender_label': 'Owner transport', 'transport': {'channel': 'telegram'}}
        if case in {'ready', 'ready_after_admission'}:
            payload['client_message_id'] = 'restart-note'
        assert client.post('/chat/inject', json=payload).status_code == 403
        assert not stopped
        response = client.post('/chat/inject', headers={'X-Skill-Token': 'token'}, json=payload)
        assert response.status_code == 202, response.text
    _join_restarts()
    if case == 'known':
        assert stopped == [True] and bridge.get_updates(0, timeout=0) == []
    else:
        assert stopped == [], 'no direct action without current matching authority'
        queued = bridge.get_updates(0, timeout=0)
        assert len(queued) == 1
        msg = queued[0]['message']
        assert msg['source'] == 'skill:test_skill' and msg['from']['id'] == 17 and msg['chat']['id'] == 23
        assert msg['sender_label'] == 'Owner transport' and msg['transport'] == {'channel': 'telegram'}
        if case in {'ready', 'ready_after_admission'}:
            assert msg['client_message_id'] == 'restart-note'
            assert msg['accepted_source_ref']['client_message_id'] == 'restart-note'
            assert msg['accepted_source_row']['client_message_id'] == 'restart-note'
        else:
            bridge.requeue_updates(queued)
            monkeypatch.setattr(message_bus, 'log_chat', lambda *_a, **_kw: None)
            ctx = Ctx(live)
            obj.server._process_bridge_updates(bridge, 0, ctx)
            assert not stopped
            expected = 'registered' if case == 'unbound' else 'unknown' if case == 'unknown' else 'not the bound owner'
            assert expected in ctx.sent[-1][1]


@pytest.mark.parametrize('checkout_ok', [True, False])
def test_correlated_restart_is_accepted_once_without_holding_ingress_or_claiming_completion(
        startup_controls, monkeypatch, checkout_ok):
    from ouroboros import server_restart
    from ouroboros.gateway.host_service import create_host_service_app
    from supervisor import message_bus, state
    from tests.test_host_service_api import _seed_token

    obj = startup_controls
    bridge = message_bus.LocalChatBridge()
    bridge.startup_owner_command = obj.server._startup_owner_command
    monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
    monkeypatch.setattr(message_bus, 'DATA_DIR', obj.data)
    monkeypatch.setattr(state, 'load_state', lambda: {'owner_external_id': 17, 'owner_external_chat_id': 23})
    entered, release = threading.Event(), threading.Event()
    calls = []

    def gate(*_args, **_kwargs):
        calls.append('checkout')
        entered.set()
        assert release.wait(5)
        return checkout_ok, 'fixture checkout refused' if not checkout_ok else 'ok'

    monkeypatch.setattr(server_restart, '_safe_restart_serialized', gate)
    monkeypatch.setattr(server_restart, '_stop_owned_work', lambda *_: [])
    _seed_token(obj.data, permissions=['inject_chat'])
    app = create_host_service_app(obj.data, bridge_getter=lambda: bridge)
    headers = {'X-Skill-Token': 'token'}
    body = {'text': '/restart', 'user_id': 17, 'chat_id': 23, 'client_message_id': 'restart-once'}
    try:
        with TestClient(app) as client:
            assert client.post('/chat/inject', headers=headers, json={**body, 'wait_for_response': True}).status_code == 400
            assert not entered.is_set(), 'validation must precede dispatch'
            accepted = client.post('/chat/inject', headers=headers, json=body)
            assert accepted.status_code == 202 and accepted.json()['operation_ref'] == '23:restart-once'
            assert entered.wait(3)
            assert message_bus._INGRESS_LOCK.acquire(timeout=0.5), 'slow checkout held the canonical ingress lock'
            message_bus._INGRESS_LOCK.release()
            assert client.get('/chat/operations/23:restart-once', headers=headers).json()['status'] == 'pending'
            repeated = client.post('/chat/inject', headers=headers, json=body)
            assert repeated.status_code == 202 and repeated.json()['rejoined'] is True
            assert client.post('/chat/inject', headers=headers, json={**body, 'text': '/status'}).status_code == 409
            assert calls == ['checkout']
        # The accepted action keeps running after its request/client disconnects.
        assert not release.is_set() and calls == ['checkout']
    finally:
        release.set()
        _join_restarts()
    with TestClient(app) as client:
        operation = client.get('/chat/operations/23:restart-once', headers=headers).json()
        assert operation['status'] == ('completed' if checkout_ok else 'failed')
        if checkout_ok:
            assert 'Restart confirmed.' in operation['text'] and 'restarted' not in operation['text'].lower()
        rejoined = client.post('/chat/inject', headers=headers, json=body)
        assert rejoined.json()['rejoined'] is True and calls == ['checkout']
    rows = [json.loads(line) for line in (obj.data / 'logs/chat.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len([row for row in rows if row.get('direction') == 'in']) == 1
    replies = [row for row in rows if row.get('origin_message_ref')]
    assert replies
    assert all(row['origin_message_ref']['client_message_id'] == 'restart-once' for row in replies)
    assert not replies[0].get('task_terminal_status'), 'initial acknowledgement never claims completion'
    assert any(row.get('task_terminal_status') == 'completed' for row in replies) is checkout_ok
    assert obj.server._restart_requested.is_set() is checkout_ok


@pytest.mark.parametrize('frame', [{'type': 'command', 'cmd': '/restart'}, {'type': 'chat', 'content': '/restart'}])
def test_ws_restart_before_bridge_is_published(startup_controls, monkeypatch, frame):
    invoked = threading.Event()
    monkeypatch.setattr(startup_controls.server, '_perform_owner_restart',
                        lambda *_: (invoked.set() or True, 'ok'))
    with startup_controls.client.websocket_connect('/ws') as socket:
        socket.send_json(frame)
        if frame['type'] == 'command':
            assert invoked.wait(3)
        else:
            assert socket.receive_json()['system_type'] == 'initialization_notice'
            assert not invoked.is_set(), 'typed chat needs its canonical acceptance owner'


def test_ws_restart_does_not_block_same_socket_panic(startup_controls, monkeypatch):
    from supervisor import message_bus

    obj = startup_controls
    started, release, panic = threading.Event(), threading.Event(), threading.Event()
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
    monkeypatch.setattr(bridge.panic, 'request', lambda text, **_: panic.set() or True if text == '/panic' else False)

    def restart(_ctx, _reply=None):
        started.set()
        assert release.wait(5)
        return True, 'ok'

    monkeypatch.setattr(obj.server, '_perform_owner_restart', restart)
    with obj.client.websocket_connect('/ws') as socket:
        try:
            socket.send_json({'type': 'command', 'cmd': '/restart'})
            assert started.wait(3)
            socket.send_json({'type': 'chat', 'content': '/panic'})
            assert panic.wait(3), 'Panic waited behind a slow restart checkout'
        finally:
            release.set()


@pytest.mark.parametrize('echo_failure', [False, True])
def test_typed_restart_keeps_ordered_receipts_and_echo_before_dispatch(startup_controls, monkeypatch, echo_failure):
    from supervisor import message_bus

    obj = startup_controls
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
    entered, release, panic, callback_seen, restarted = (threading.Event() for _ in range(5))
    echoes, observed = [], []
    original = bridge.handle_web_message

    def hold(text, **kwargs):
        if text == 'prior message':
            entered.set()
            assert release.wait(5)
        original(text, **kwargs)

    def echo(frame):
        if frame.get('ingress_accepted'):
            echoes.append(frame['client_message_id'])
            if echo_failure and frame['client_message_id'] == 'restart':
                raise RuntimeError('socket disconnected during echo')

    def admit(command, **kwargs):
        callback_seen.set()
        return obj.server._startup_owner_command(command, **kwargs)

    def restart(_ctx, _reply=None):
        rows = [json.loads(line) for line in (obj.data / 'logs/chat.jsonl').read_text(encoding='utf-8').splitlines()]
        observed.append(([row['client_message_id'] for row in rows if row.get('direction') == 'in'], list(echoes)))
        restarted.set()
        return True, 'ok'

    monkeypatch.setattr(bridge, 'handle_web_message', hold)
    monkeypatch.setattr(bridge.panic, 'request', lambda text, **_: panic.set() or True if text == '/panic' else False)
    bridge._broadcast_fn = echo
    obj.app.state.startup_owner_command = admit
    monkeypatch.setattr(obj.server, '_perform_owner_restart', restart)
    with obj.client.websocket_connect('/ws') as socket:
        try:
            socket.send_json({'type': 'chat', 'content': 'prior message', 'client_message_id': 'prior'})
            assert entered.wait(3)
            socket.send_json({'type': 'chat', 'content': '/restart', 'client_message_id': 'restart'})
            socket.send_json({'type': 'chat', 'content': '/panic'})
            assert panic.wait(3), 'Panic cannot wait behind ordinary ordered acceptance'
            assert not callback_seen.is_set(), 'typed Restart cannot bypass its canonical acceptance'
            release.set()
            assert restarted.wait(3)
        finally:
            release.set()
    _join_restarts()
    assert observed == [(['prior', 'restart'], ['prior', 'restart'])]
    queued = bridge.get_updates(0, timeout=0)
    assert [row['message']['client_message_id'] for row in queued] == ['prior']
