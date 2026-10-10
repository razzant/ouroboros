"""Owner process verbs work before onboarding without an unconsumed chat bus."""
from types import SimpleNamespace
import json
import threading

import pytest
from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient

pytestmark = pytest.mark.serial


@pytest.mark.parametrize('case', ['unmanaged', 'custom', 'missing', 'error'])
def test_runtime_branch_defaults_preserve_managed_choices_and_fallback(monkeypatch, case):
    import server
    from supervisor import git_ops

    calls = []

    def branches(repo):
        calls.append(repo)
        if case == 'error':
            raise RuntimeError('manifest unavailable')
        return 'custom-dev', 'custom-stable'

    monkeypatch.setattr(server, '_LAUNCHER_MANAGED', case != 'unmanaged')
    if case == 'missing':
        monkeypatch.delattr(git_ops, 'managed_branch_defaults')
    else:
        monkeypatch.setattr(git_ops, 'managed_branch_defaults', branches)
    expected = ('custom-dev', 'custom-stable') if case == 'custom' else ('ouroboros', 'ouroboros-stable')
    assert server._runtime_branch_defaults() == expected
    assert calls == ([server.REPO_DIR] if case in {'custom', 'error'} else [])


@pytest.fixture
def startup_controls(tmp_path, monkeypatch):
    import server
    from ouroboros.gateway.control import api_command
    from ouroboros.gateway.ws import ws_endpoint
    from supervisor import message_bus, state, workers, git_ops

    data, repo = tmp_path / 'data', tmp_path / 'repo'
    repo.mkdir(); data.mkdir()
    monkeypatch.setattr(server, 'DATA_DIR', data)
    monkeypatch.setattr(server, 'REPO_DIR', repo)
    monkeypatch.setattr(server, '_supervisor_thread', None)
    monkeypatch.setattr(server, '_supervisor_ready', threading.Event())
    monkeypatch.setattr(server, '_consciousness', None)
    monkeypatch.setattr(server, '_restart_requested', threading.Event())
    monkeypatch.setattr(server, '_owner_restart_requested', threading.Event())
    from ouroboros import server_process, server_restart
    monkeypatch.setattr(server_restart, 'DATA_DIR', data)
    monkeypatch.setattr(server_process, '_restart_requested', server._restart_requested)
    monkeypatch.setattr(server_process, '_owner_restart_requested', server._owner_restart_requested)
    monkeypatch.setattr(message_bus, '_BRIDGE', None)
    monkeypatch.setattr(message_bus, 'DATA_DIR', data)
    monkeypatch.setattr(workers, 'RUNNING', {})
    for name in ('DRIVE_ROOT', 'STATE_PATH', 'STATE_LAST_GOOD_PATH', 'STATE_LOCK_PATH'):
        monkeypatch.setattr(state, name, getattr(state, name))
    for name in ('DRIVE_ROOT', 'REPO_DIR', 'REMOTE_URL', 'BRANCH_DEV', 'BRANCH_STABLE'):
        monkeypatch.setattr(git_ops, name, getattr(git_ops, name))
    app = Starlette(routes=[Route('/api/command', api_command, methods=['POST']), WebSocketRoute('/ws', ws_endpoint)])
    app.state.startup_owner_command = server._startup_owner_command
    with TestClient(app) as client:
        yield SimpleNamespace(server=server, client=client, app=app, data=data, repo=repo)
    for thread in threading.enumerate():
        if thread.name == 'startup-owner-restart':
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.fixture(params=['absent', 'initializing_at_admission', 'initializing_after_admission', 'dead_with_stale_bridge'])
def startup_without_consumer(startup_controls, monkeypatch, request):
    """A real thread is not a consumer until it has a published bridge."""
    from supervisor import message_bus

    obj = startup_controls
    release = threading.Event()
    thread = threading.Thread(target=release.wait)
    bridge = message_bus.LocalChatBridge()
    if request.param == 'initializing_at_admission':
        thread.start()
        monkeypatch.setattr(obj.server, '_supervisor_thread', thread)

    def admit(command, **kwargs):
        action = obj.server._startup_owner_command(command, **kwargs)
        if request.param in {'initializing_after_admission', 'dead_with_stale_bridge'}:
            thread.start()
            monkeypatch.setattr(obj.server, '_supervisor_thread', thread)
            if request.param == 'dead_with_stale_bridge':
                release.set()
                thread.join(timeout=2)
                assert not thread.is_alive()
                monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
        return action

    obj.app.state.startup_owner_command = admit
    try:
        yield obj
        assert bridge.get_updates(0, timeout=0) == [], 'a stale bridge must not receive the command'
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(timeout=2)
        assert not thread.is_alive()


def test_onboarding_restart_uses_existing_no_resume_flags_and_exit_signal(startup_without_consumer, monkeypatch):
    obj = startup_without_consumer
    calls = []
    from ouroboros import server_restart
    monkeypatch.setattr(server_restart, '_safe_restart_serialized',
                        lambda fn, **kw: calls.append(('checked', kw)) or (True, 'ok'))
    monkeypatch.setattr(server_restart, '_stop_owned_work', lambda ctx: calls.append(('stopped', ctx.RUNNING)))
    response = obj.client.post('/api/command', json={'cmd': '/restart'})
    assert response.status_code == 200 and response.json() == {'status': 'ok'}
    assert calls == [('checked', {'reason': 'owner_restart', 'unsynced_policy': 'rescue_and_reset'}), ('stopped', {})]
    assert obj.server._restart_requested.is_set() and obj.server._owner_restart_requested.is_set()
    assert obj.server.RESTART_EXIT_CODE == 42
    assert (obj.data / 'state/owner_restart_no_resume.flag').read_text(encoding="utf-8") == 'owner_restart'
    assert (obj.data / 'state/panic_stop.flag').read_text(encoding="utf-8") == 'owner_restart_no_resume'
    assert not (obj.data / 'settings.json').exists()
    from supervisor import state, git_ops
    assert state.DRIVE_ROOT == git_ops.DRIVE_ROOT == obj.data
    assert git_ops.REPO_DIR == obj.repo


@pytest.mark.parametrize('thread_alive', [True, False])
@pytest.mark.parametrize('surface', ['http', 'ws', 'ws_chat', 'ws_chat_files'])
def test_restart_during_initialization_does_not_queue_to_published_bridge(
        startup_controls, monkeypatch, thread_alive, surface):
    """Publishing the bridge precedes recovery; only a ready loop can consume Restart. The composer
    sends Restart typed beside a staged file as the exact command (chat_attachments.composerText):
    the startup door still takes it, and its file stays on the one accepted row."""
    from supervisor import message_bus
    from ouroboros import chat_uploads, server_restart
    from ouroboros.gateway import ws as ws_gateway
    from tests.test_chat_attachments import PNG

    obj = startup_controls
    monkeypatch.setattr(ws_gateway, 'DATA_DIR', obj.data)
    _path, staged = chat_uploads.store_upload(PNG, 'photo.png', data_dir=obj.data, pending=True)
    release = threading.Event()
    thread = threading.Thread(target=release.wait)
    thread.start()
    monkeypatch.setattr(obj.server, '_supervisor_thread', thread)
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
    monkeypatch.setattr(server_restart, '_safe_restart_serialized', lambda *a, **kw: (True, 'ok'))
    stopped = []
    did_stop = threading.Event()
    monkeypatch.setattr(server_restart, '_stop_owned_work', lambda ctx: (stopped.append(ctx.RUNNING), did_stop.set()))
    if not thread_alive:
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
    try:
        if surface == 'http':
            response = obj.client.post('/api/command', json={'cmd': '/restart'})
            assert response.status_code == 200 and response.json() == {'status': 'ok'}
        else:
            with obj.client.websocket_connect('/ws') as socket:
                socket.send_json({'type': 'command', 'cmd': '/restart'} if surface == 'ws'
                                 else {'type': 'chat', 'content': ' /ReStArT '} if surface == 'ws_chat'
                                 else {'type': 'chat', 'content': '/restart', 'client_message_id': 'with-file', 'attachments': [
                                     {'filename': staged['upload'], 'display_name': 'photo.png', 'mime': 'image/png'}]})
                assert did_stop.wait(3), 'Restart remained behind the initializing supervisor'
        assert stopped == [{}]
        if surface == 'ws_chat_files':
            rows = [json.loads(line) for line in (obj.data / 'logs/chat.jsonl').read_text(encoding='utf-8').splitlines()]
            (row,) = [item for item in rows if item['direction'] == 'in']
            assert (row['text'], row['client_message_id']) == ('/restart', 'with-file')
            assert [ref['upload'] for ref in row['attachments']] == [staged['upload']], 'the file rides the accepted row'
            assert any(item.get('type') == 'command_reply' and item['origin_message_ref']['client_message_id'] == 'with-file'
                       for item in rows), "the door's reply answers that row"
        assert obj.server._restart_requested.wait(3) and obj.server._owner_restart_requested.is_set()
        assert (obj.data / 'state/owner_restart_no_resume.flag').read_text(encoding='utf-8') == 'owner_restart'
        assert bridge.get_updates(0, timeout=0) == []
    finally:
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_onboarding_panic_runs_real_panic_marker_and_exit_99(startup_without_consumer, monkeypatch):
    obj = startup_without_consumer
    exits, stops = [], []
    from ouroboros import server_control
    from ouroboros.gateway.host_service import host_service_port
    from ouroboros.startup_historical_audit import audit
    monkeypatch.setattr(audit, 'stop', lambda: None)
    monkeypatch.setattr('ouroboros.tools.shell.kill_all_tracked_subprocesses', lambda **kw: [])
    monkeypatch.setattr('ouroboros.workspace_executor.kill_all_foreground', lambda *a, **kw: [])
    monkeypatch.setattr('ouroboros.tools.services.kill_all_services', lambda *a, **kw: [])
    monkeypatch.setattr('ouroboros.local_model.get_manager', lambda **kw: SimpleNamespace(
        panic_stop=lambda **kw: [], stop_server=lambda: None))
    monkeypatch.setattr('supervisor.evolution_lifecycle.complete_evolution_campaign', lambda *a, **kw: {})
    monkeypatch.setattr('ouroboros.post_task_evolution.drop_pending_request', lambda *a, **kw: None)
    monkeypatch.setattr('ouroboros.extension_companion.panic_kill_all', lambda **kw: [])
    monkeypatch.setattr('multiprocessing.active_children', lambda: [SimpleNamespace(pid=4321)])
    monkeypatch.setattr('supervisor.worker_pool_lifecycle.kill_worker_tree',
                        lambda pid, panic_process=None: stops.append(('native_worker', pid)) or {'requested': True})
    monkeypatch.setattr('ouroboros.platform_layer.kill_process_on_port', lambda port: stops.append(('port', port)))
    monkeypatch.setattr('ouroboros.claudexor_daemon.get_owned_daemon',
                        lambda **kw: SimpleNamespace(
                            panic_stop=lambda *, request_only: stops.append(('daemon_request', request_only)) or [],
                            stop_outcome=lambda: stops.append(('daemon',))))
    monkeypatch.setattr('supervisor.workers.kill_workers', lambda **kw: stops.append(('workers', kw)))
    monkeypatch.setattr(obj.server, '_ACTUAL_BOUND_PORT', 19876)
    monkeypatch.setattr(server_control.os, '_exit', lambda code: exits.append(code))
    before = set(threading.enumerate())
    try:
        response = obj.client.post('/api/command', json={'cmd': '/panic'})
    finally:
        # Production hard-exits without waiting for helpers. This test retains
        # its fake owners until their independent settlement threads finish.
        for thread in set(threading.enumerate()) - before:
            if thread.name.startswith('panic-'):
                thread.join(timeout=5)
                assert not thread.is_alive()
    assert response.status_code == 200 and response.json() == {'status': 'ok'}
    assert exits == [99]
    assert (obj.data / 'state/panic_stop.flag').read_text(encoding="utf-8") == 'panic'
    assert stops[0] == ('daemon_request', True)
    assert stops.count(('daemon',)) == 1
    assert [event for event in stops if event[0] == 'native_worker'] == [('native_worker', 4321)]
    assert not [event for event in stops if event[0] == 'workers']
    assert stops.index(('native_worker', 4321)) < next(i for i, event in enumerate(stops) if event[0] == 'port')
    assert sorted(event[1] for event in stops if event[0] == 'port') == sorted([19876, host_service_port()])
    assert not (obj.data / 'settings.json').exists()
    from supervisor import state
    # Before onboarding there is no state to write: Panic never mints one (#1307). The flag is
    # the durable gate; the first real boot writes the disabled controls, then consumes it.
    # This emergency door must not wait for supervisor.state.init to rebind
    # process-global paths merely to kill the already-owned workers.
    assert state.read_state(obj.data).quality == "uninitialized"


@pytest.mark.parametrize('command', ['ordinary chat', '/review', '/evolve', '/panic later', '/restart later'])
def test_unknown_commands_are_not_accepted_without_a_consumer(startup_controls, command):
    obj = startup_controls
    response = obj.client.post('/api/command', json={'cmd': command})
    assert response.status_code == 409
    assert 'provider setup' in response.json()['error']
    assert not obj.server._restart_requested.is_set()
    assert not (obj.data / 'state/panic_stop.flag').exists()


@pytest.mark.parametrize('command', ['/panic', '/restart', 'ordinary chat'])
def test_configured_commands_keep_the_existing_bus_path(startup_controls, monkeypatch, command):
    calls = []
    monkeypatch.setattr(startup_controls.server, '_supervisor_thread', SimpleNamespace(is_alive=lambda: True))
    startup_controls.server._supervisor_ready.set()
    monkeypatch.setattr('supervisor.message_bus._BRIDGE', SimpleNamespace(ui_send=lambda *a, **kw: calls.append((a, kw))))
    response = startup_controls.client.post('/api/command', json={'cmd': command})
    assert response.status_code == 200
    assert len(calls) == 1 and calls[0][0] == (command,)
    assert calls[0][1]['task_metadata']['client_surface'] == {'channel': 'api_command'}
    assert not startup_controls.server._restart_requested.is_set()


@pytest.mark.parametrize('command', ['/panic', '/restart'])
def test_onboarding_finishing_before_background_dispatch_reuses_live_consumer(startup_controls, monkeypatch, command):
    from supervisor import message_bus

    obj = startup_controls
    release = threading.Event()
    thread = threading.Thread(target=release.wait)
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, 'log_chat', lambda *a, **kw: None)
    panic = []
    monkeypatch.setattr(obj.server, '_execute_panic_stop', lambda *a: panic.append('requested'))
    monkeypatch.setattr(obj.server, '_perform_owner_restart', lambda *a: pytest.fail('duplicate direct Restart'))

    def admit(cmd, **kwargs):
        action = obj.server._startup_owner_command(cmd, **kwargs)
        thread.start()
        monkeypatch.setattr(obj.server, '_supervisor_thread', thread)
        monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
        obj.server._supervisor_ready.set()
        return action

    obj.app.state.startup_owner_command = admit
    try:
        response = obj.client.post('/api/command', json={'cmd': command})
        assert response.status_code == 200 and response.json() == {'status': 'ok'}
        updates = bridge.get_updates(0, timeout=0)
        if command == '/panic':
            # The authenticated emergency door must not wait for the later
            # ordinary consumer to come online or enqueue a duplicate command.
            assert panic == ['requested'] and updates == []
        else:
            assert panic == [] and len(updates) == 1 and updates[0]['message']['text'] == command
            assert updates[0]['message']['task_metadata']['client_surface']['channel'] == 'api_command'
        assert bridge.get_updates(0, timeout=0) == []
        assert not obj.server._restart_requested.is_set()
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(timeout=2)
        assert not thread.is_alive()
    assert not (obj.data / 'state/panic_stop.flag').exists()


def test_onboarding_restart_refusal_keeps_core_and_source(startup_controls, monkeypatch, caplog):
    obj = startup_controls
    from ouroboros import server_restart
    monkeypatch.setattr(server_restart, '_safe_restart_serialized', lambda *a, **kw: (False, 'update is still resolving'))
    monkeypatch.setattr(server_restart, '_stop_owned_work', lambda ctx: pytest.fail('stopped before gate'))
    response = obj.client.post('/api/command', json={'cmd': '/restart'})
    assert response.status_code == 200  # accepted dispatch; failure remains explicit in lifecycle log
    assert not obj.server._restart_requested.is_set()
    assert not (obj.data / 'state/owner_restart_no_resume.flag').exists()
    assert 'update is still resolving' in caplog.text
