"""Cross-surface integration regressions without widening credential authority."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from ouroboros import config, local_model, server_process
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import resource_root_path


@pytest.mark.parametrize('relative', ['auth_token.json', 'profiles/alpha/auth_token.json'])
def test_ordinary_token_named_file_is_readable_in_repository_and_task_outputs(tmp_path, relative):
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    repo.mkdir()
    data.mkdir()
    secret = repo / relative
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text('ORDINARY_PROJECT_TOKEN_REPORT')
    ctx = ToolContext(repo_dir=repo, drive_root=data,
                      task_constraint=TaskConstraint(mode='local_readonly_subagent'))
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)
    result = registry.execute_result('read_file', {'root':'system_repo','path':relative})
    assert result.status == 'ok'
    assert 'ORDINARY_PROJECT_TOKEN_REPORT' in result.text
    assert 'auth_token.json' in registry.execute('list_files', {'root':'system_repo','path':str(secret.parent.relative_to(repo))})
    search = registry.execute('search_code', {'query':'ORDINARY_PROJECT_TOKEN_REPORT'})
    assert 'auth_token.json:' in search
    task_root = resource_root_path(ctx, 'task_drive')
    target = task_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('ORDINARY_TASK_OUTPUT')
    assert 'ORDINARY_TASK_OUTPUT' in registry.execute('read_file', {'root':'task_drive','path':relative})


@pytest.mark.parametrize('name', ['public.pem', 'private.key'])
def test_certificate_suffix_alone_is_not_restricted_child_authority(tmp_path, name):
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    repo.mkdir()
    data.mkdir()
    (repo / name).write_text('reviewed project certificate fixture')
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data,
                         task_constraint=TaskConstraint(mode='local_readonly_subagent')))
    assert 'reviewed project certificate fixture' in registry.execute('read_file', {'path':name})


@pytest.mark.parametrize('failure', ['metadata', 'write', 'none'])
def test_healthy_local_model_is_ready_when_informational_binding_fails(tmp_path, monkeypatch, caplog, failure):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    manager = local_model.LocalModelManager()
    proc = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    manager._proc = proc
    manager._status = 'loading'
    monkeypatch.setattr(manager, 'health_check', lambda **_: {'ok':True,'context_length':8192,'model_name':'fixture'})
    if failure == 'metadata':
        monkeypatch.setattr(server_process, 'record_service_binding', lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError('start time unavailable')))
    elif failure == 'write':
        monkeypatch.setattr(server_process, 'update_json_locked', lambda *_a, **_k: (_ for _ in ()).throw(OSError('read-only filesystem')))
    manager._wait_for_healthy(proc=proc, port=manager.port, launch_id='fixture-launch', timeout=1)
    assert manager.is_running
    assert manager._proc is proc
    assert manager.status_dict()['context_length'] == 8192
    assert manager.status_dict()['model_name'] == 'fixture'
    assert manager.status_dict()['error'] is None
    if failure != 'none':
        assert manager._service_binding is None
        assert 'binding unavailable' in caplog.text
    else:
        assert manager._service_binding['pid'] == proc.pid


def test_stale_health_response_cannot_publish_new_model_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    manager = local_model.LocalModelManager()
    previous = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    current = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    manager._proc = previous
    manager._status = 'loading'
    def health(**_):
        manager._proc = current
        return {'ok':True,'context_length':8192,'model_name':'old'}
    monkeypatch.setattr(manager, 'health_check', health)
    monkeypatch.setattr(server_process, 'record_service_binding', lambda *_a, **_k: pytest.fail('stale generation published'))
    manager._wait_for_healthy(proc=previous, port=manager.port, launch_id='fixture-launch', timeout=1)
    assert manager._status == 'loading'
    assert manager._proc is current
    assert manager._service_binding is None
