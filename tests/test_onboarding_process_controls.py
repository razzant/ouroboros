"""Owner process verbs work before onboarding without an unconsumed chat bus."""
from types import SimpleNamespace
import threading

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
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
    from supervisor import message_bus, state, workers, git_ops

    data, repo = tmp_path / 'data', tmp_path / 'repo'
    repo.mkdir(); data.mkdir()
    monkeypatch.setattr(server, 'DATA_DIR', data)
    monkeypatch.setattr(server, 'REPO_DIR', repo)
    monkeypatch.setattr(server, '_supervisor_thread', None)
    monkeypatch.setattr(server, '_consciousness', None)
    monkeypatch.setattr(server, '_restart_requested', threading.Event())
    monkeypatch.setattr(server, '_owner_restart_requested', threading.Event())
    from ouroboros import server_process
    monkeypatch.setattr(server_process, '_restart_requested', server._restart_requested)
    monkeypatch.setattr(server_process, '_owner_restart_requested', server._owner_restart_requested)
    monkeypatch.setattr(message_bus, '_BRIDGE', None)
    monkeypatch.setattr(workers, 'RUNNING', {})
    for name in ('DRIVE_ROOT', 'STATE_PATH', 'STATE_LAST_GOOD_PATH', 'STATE_LOCK_PATH'):
        monkeypatch.setattr(state, name, getattr(state, name))
    for name in ('DRIVE_ROOT', 'REPO_DIR', 'REMOTE_URL', 'BRANCH_DEV', 'BRANCH_STABLE'):
        monkeypatch.setattr(git_ops, name, getattr(git_ops, name))
    app = Starlette(routes=[Route('/api/command', api_command, methods=['POST'])])
    app.state.startup_owner_command = server._startup_owner_command
    with TestClient(app) as client:
        yield SimpleNamespace(server=server, client=client, data=data, repo=repo)


def test_onboarding_restart_uses_existing_no_resume_flags_and_exit_signal(startup_controls, monkeypatch):
    obj = startup_controls
    calls = []
    monkeypatch.setattr(obj.server, '_safe_restart_serialized',
                        lambda fn, **kw: calls.append(('checked', kw)) or (True, 'ok'))
    monkeypatch.setattr(obj.server, '_stop_owned_work', lambda ctx: calls.append(('stopped', ctx.RUNNING)))
    response = obj.client.post('/api/command', json={'cmd': '/restart'})
    assert response.status_code == 200 and response.json() == {'status': 'ok'}
    assert calls == [('checked', {'reason': 'owner_restart', 'unsynced_policy': 'rescue_and_reset'}), ('stopped', {})]
    assert obj.server._restart_requested.is_set() and obj.server._owner_restart_requested.is_set()
    assert obj.server.RESTART_EXIT_CODE == 42
    assert (obj.data / 'state/owner_restart_no_resume.flag').read_text() == 'owner_restart'
    assert (obj.data / 'state/panic_stop.flag').read_text() == 'owner_restart_no_resume'
    assert not (obj.data / 'settings.json').exists()
    from supervisor import state, git_ops
    assert state.DRIVE_ROOT == git_ops.DRIVE_ROOT == obj.data
    assert git_ops.REPO_DIR == obj.repo


def test_onboarding_panic_runs_real_panic_marker_and_exit_99(startup_controls, monkeypatch):
    obj = startup_controls
    exits, stops = [], []
    from ouroboros import server_control
    monkeypatch.setattr('ouroboros.tools.shell.kill_all_tracked_subprocesses', lambda: None)
    monkeypatch.setattr('ouroboros.workspace_executor.kill_all_foreground', lambda *a, **kw: None)
    monkeypatch.setattr('ouroboros.tools.services.kill_all_services', lambda *a, **kw: None)
    monkeypatch.setattr('ouroboros.local_model.get_manager', lambda: SimpleNamespace(stop_server=lambda: None))
    monkeypatch.setattr('supervisor.evolution_lifecycle.complete_evolution_campaign', lambda *a, **kw: {})
    monkeypatch.setattr('ouroboros.post_task_evolution.drop_pending_request', lambda *a, **kw: None)
    monkeypatch.setattr('ouroboros.extension_companion.panic_kill_all', lambda: None)
    monkeypatch.setattr('multiprocessing.active_children', lambda: [])
    monkeypatch.setattr('ouroboros.platform_layer.kill_process_on_port', lambda port: stops.append(('port', port)))
    monkeypatch.setattr('ouroboros.claudexor_daemon.get_owned_daemon',
                        lambda: SimpleNamespace(stop=lambda: stops.append(('daemon',))))
    monkeypatch.setattr('supervisor.workers.kill_workers', lambda **kw: stops.append(('workers', kw)))
    monkeypatch.setattr(obj.server, '_ACTUAL_BOUND_PORT', 19876)
    monkeypatch.setattr(server_control.os, '_exit', lambda code: exits.append(code))
    response = obj.client.post('/api/command', json={'cmd': '/panic'})
    assert response.status_code == 200 and response.json() == {'status': 'ok'}
    assert exits == [99]
    assert (obj.data / 'state/panic_stop.flag').read_text() == 'panic'
    assert ('daemon',) in stops and ('port', 19876) in stops
    assert not (obj.data / 'settings.json').exists()
    from supervisor import state
    assert state.DRIVE_ROOT == obj.data
    assert state.load_state()['evolution_owner_stopped'] is True


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
    monkeypatch.setattr('supervisor.message_bus._BRIDGE', SimpleNamespace(ui_send=lambda *a, **kw: calls.append((a, kw))))
    response = startup_controls.client.post('/api/command', json={'cmd': command})
    assert response.status_code == 200
    assert len(calls) == 1 and calls[0][0] == (command,)
    assert not startup_controls.server._restart_requested.is_set()


def test_onboarding_finishing_before_background_dispatch_reuses_live_consumer(startup_controls, monkeypatch):
    obj = startup_controls
    action = obj.server._startup_owner_command('/panic')
    calls = []
    monkeypatch.setattr(obj.server, '_supervisor_thread', SimpleNamespace(is_alive=lambda: True))
    monkeypatch.setattr('supervisor.message_bus._BRIDGE', SimpleNamespace(ui_send=lambda *a, **kw: calls.append((a, kw))))
    action()
    assert calls == [(('/panic',), {'broadcast': False})]
    assert not (obj.data / 'state/panic_stop.flag').exists()


def test_onboarding_restart_refusal_keeps_core_and_source(startup_controls, monkeypatch, caplog):
    obj = startup_controls
    monkeypatch.setattr(obj.server, '_safe_restart_serialized', lambda *a, **kw: (False, 'update is still resolving'))
    monkeypatch.setattr(obj.server, '_stop_owned_work', lambda ctx: pytest.fail('stopped before gate'))
    response = obj.client.post('/api/command', json={'cmd': '/restart'})
    assert response.status_code == 200  # accepted dispatch; failure remains explicit in lifecycle log
    assert not obj.server._restart_requested.is_set()
    assert not (obj.data / 'state/owner_restart_no_resume.flag').exists()
    assert 'update is still resolving' in caplog.text
