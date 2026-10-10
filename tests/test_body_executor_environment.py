"""Candidate environment at Docker argv boundary, executed by a keyless local transport double.

The command/service shell is real. No daemon/container or provider is contacted.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros import body_candidate, workspace_executor as executor
from tests.body_candidate_support import isolate, make_ctx, make_serving

pytestmark = pytest.mark.serial


@pytest.mark.parametrize('consumer', ['command', 'service'])
def test_docker_candidate_uses_mapped_clean_environment(tmp_path, monkeypatch, consumer):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    ctx = make_ctx(serving, data, 'docker-env')
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    mount = tmp_path / 'mount'
    mount.symlink_to(candidate.parent, target_is_directory=True)
    ctx.executor_ref = {'type': 'docker_exec', 'container_name': 'fixture',
        'workspace_host_path': str(candidate.parent), 'workspace_backend_path': str(mount)}
    expected = str(mount / (candidate.name + '.env') / 'data')
    script = 'import os,json; print(json.dumps(dict(os.environ)), flush=True)'
    real_popen, real_run = subprocess.Popen, subprocess.run
    commands = []

    def transport(cmd, **kwargs):
        if cmd[:2] != ['docker', 'exec']:
            return real_popen(cmd, **kwargs)
        # Model docker exec's container env: host env arrives only under named --env aliases.
        host_env = kwargs.get('env') or os.environ
        backend_env = {'PATH': os.environ['PATH'], 'HOME': '/container-home',
                       'CONTAINER_API_KEY': 'synthetic-leak', 'OUROBOROS_MANAGED_BY_LAUNCHER': '1'}
        for index, arg in enumerate(cmd):
            if arg == '--env':
                backend_env[cmd[index + 1]] = host_env[cmd[index + 1]]
        kwargs['env'] = backend_env
        if '--workdir' in cmd:
            kwargs['cwd'] = cmd[cmd.index('--workdir') + 1]
        commands.append((cmd, kwargs))
        return real_popen(['sh', '-c', cmd[-1]], **kwargs)

    monkeypatch.setattr(executor.subprocess, 'Popen', transport)
    if consumer == 'command':
        result = executor.execute(ctx, [sys.executable, '-c', script], candidate, 10)
        assert result.returncode == 0 and result.operation_outcome == 'completed', result
        env = json.loads(result.stdout)
    else:
        # Execute the real service command expansion synchronously, preserving the
        # actual start_service branch and inert-env transport without orphan processes.
        original = executor._docker_service_start_shell
        captured = {}
        def service_shell(record, log_path, aliases=None, **kwargs):
            shell = original(record, log_path, aliases, **kwargs)
            # The same generated exec payload is inside the detached launch shell.
            import shlex
            payload = shlex.split(shell)[shlex.split(shell).index('-c') + 1]
            return f'cd {__import__("shlex").quote(record.backend_cwd)} && {payload}'
        def submit(cmd, **kwargs):
            result = real_run(cmd, **kwargs)
            assert result.returncode == 0, result.stderr
            captured['env'] = json.loads(result.stdout)
            return subprocess.CompletedProcess(cmd, 0, '12345\n', '')
        monkeypatch.setattr(executor, '_docker_service_start_shell', service_shell)
        monkeypatch.setattr(executor, '_submit_service_command', submit)
        monkeypatch.setattr(executor, '_service_state', lambda record: 'exited')
        monkeypatch.setattr(executor, '_service_payload', lambda record, **kwargs: {'state': 'exited'})
        executor.start_service(ctx, name='env', cmd=[sys.executable, '-c', script], host_cwd=candidate,
                               cwd_root='system_repo', readiness={}, outputs=[], before_outputs={}, env={})
        env = captured['env']
    assert env['OUROBOROS_DATA_DIR'] == expected
    assert env['HOME'] == str(mount / (candidate.name + '.env') / 'home')
    assert env['OUROBOROS_REPO_DIR'] == str(mount / candidate.name)
    assert 'CONTAINER_API_KEY' not in env and 'OUROBOROS_MANAGED_BY_LAUNCHER' not in env
    assert not any(key.startswith(('OUROBOROS_SERVICE_ENV_', 'OUROBOROS_PROCESS_ENV_')) for key in env)
    assert commands and all('synthetic-leak' not in ' '.join(cmd) for cmd, _ in commands)


@pytest.mark.parametrize('consumer', ['command', 'service'])
def test_docker_candidate_requires_mapping_of_isolated_sibling_before_spawn(tmp_path, monkeypatch, consumer):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    ctx = make_ctx(serving, data, 'unmapped-env')
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    ctx.executor_ref = {'type': 'docker_exec', 'container_name': 'fixture',
        'workspace_host_path': str(candidate), 'workspace_backend_path': '/workspace'}
    def no_spawn(*args, **kwargs):
        raise AssertionError('a process must not start with an unmapped environment')
    monkeypatch.setattr(executor.subprocess, 'Popen', no_spawn)
    with pytest.raises(body_candidate.CandidateRefused, match='sibling .env'):
        if consumer == 'command':
            executor.execute(ctx, ['python', '-c', 'pass'], candidate, 10)
        else:
            executor.start_service(ctx, name='env', cmd=['python', '-c', 'pass'], host_cwd=candidate,
                cwd_root='system_repo', readiness={}, outputs=[], before_outputs={}, env={})
