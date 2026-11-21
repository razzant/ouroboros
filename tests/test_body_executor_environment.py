"""Candidate environment at Docker argv boundary, executed by a keyless local transport double.

The command/service shell is real. No daemon/container or provider is contacted.
"""
import os
import pathlib
import shlex
import subprocess
import sys

import pytest

from ouroboros import body_candidate, workspace_executor as executor
from ouroboros.docker_process_observer import proc_observer_function
# The executor imports this lazily; it subclasses ``subprocess.Popen`` at import, so it must be imported before
# the transport double below replaces ``subprocess.Popen`` with a function.
from ouroboros.tools import shell_process  # noqa: F401
from tests.body_candidate_support import isolate, make_ctx, make_serving


# A container's ``setsid`` (util-linux): the launched command leads its own session; same as the execution-domain
# suite's shim.
SETSID_SHIM = 'import os, sys\ntry:\n    os.setsid()\nexcept OSError:\n    pass\nos.execvp(sys.argv[1], sys.argv[1:])\n'


_DRIVER_KEYS = ('PLAYWRIGHT_NODEJS_PATH', 'OUROBOROS_PLAYWRIGHT_NODE_SELECTED', 'OUROBOROS_PINNED_NODE_CARRIED')
_SERVING_NODE = '/serving/state/cx/node/24.22.3-linux-x64.g-0123456789ab-00000000/node-standalone/bin/node'


def _pinned_node(monkeypatch, node, data):
    """The serving installation's pinned Node, whatever this host prepared: 'prepared' is one fixed host path found
    by record, 'cold' the real lookup in the isolated data root, which names a not-yet-published generation.

    Both lookups (``carry_pinned_node`` and ``reviewed_node(prepared=True)``) read the store under ``config.DATA_DIR``,
    which ``isolate`` leaves at the session's shared root: a Node another test prepared there would be carried. The
    module root is bound to this test's ``data`` so 'cold' really is cold."""
    from ouroboros import config

    monkeypatch.setattr(config, 'DATA_DIR', data)
    for name in _DRIVER_KEYS:
        monkeypatch.delenv(name, raising=False)
    if node == 'prepared':
        from types import SimpleNamespace

        from ouroboros import claudexor_runtime
        monkeypatch.setattr(claudexor_runtime, 'get_runtime_manager', lambda: SimpleNamespace(
            reviewed_node=lambda allow_install=False, prepared=False: _SERVING_NODE,
            prepared_node=lambda root: _SERVING_NODE))


@pytest.mark.parametrize('node', ['cold', 'prepared'])
@pytest.mark.parametrize('consumer', ['command', 'service'])
def test_docker_candidate_uses_mapped_clean_environment(tmp_path, monkeypatch, consumer, node):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    _pinned_node(monkeypatch, node, data)
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    ctx = make_ctx(serving, data, 'docker-env')
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    # A Docker workspace is POSIX even when its host fixture runs on Windows.
    mount = pathlib.PurePosixPath('/workspace')
    ctx.executor_ref = {'type': 'docker_exec', 'container_name': 'fixture',
        'workspace_host_path': str(candidate.parent), 'workspace_backend_path': str(mount)}
    expected = str(mount / (candidate.name + '.env') / 'data')

    def host_cwd(backend_cwd):
        relative = pathlib.PurePosixPath(backend_cwd).relative_to(mount)
        return candidate.parent.joinpath(*relative.parts)
    # Observe with a shell builtin: MSYS rewrites HOME when it launches native
    # Windows Python, a hop absent from the real Docker/container path.
    probe = ['sh', '-c', 'export -p']
    def exported_env(stdout):
        return dict(word.split('=', 1) for word in shlex.split(stdout) if '=' in word)
    real_popen = subprocess.Popen
    commands = []
    # The fake container supplies what a Linux container has and the host may lack: ``setsid`` (the wrapper
    # refuses to start without it; macOS has none) and a container /proc for the guarded completion probe's
    # group observer (macOS and Windows/MSYS have no Linux /proc). After the synchronous ``export -p`` ended,
    # that container's process table holds nothing of the command, so the observer reads an empty /proc.
    container_path = os.environ['PATH']
    if subprocess.run(['sh', '-c', 'command -v setsid'], capture_output=True).returncode != 0:
        shim_dir = tmp_path / 'container-bin'
        shim_dir.mkdir()
        (shim_dir / 'setsid').write_text(f'#!{sys.executable}\n' + SETSID_SHIM, encoding='utf-8')
        (shim_dir / 'setsid').chmod(0o755)
        container_path = f'{shim_dir}{os.pathsep}{container_path}'
    if not os.path.isdir('/proc/self'):
        container_proc = tmp_path / 'container-proc'
        container_proc.mkdir()
        monkeypatch.setattr(executor, '_docker_proc_observer_function',
                            lambda: proc_observer_function(proc_root=str(container_proc)))

    def transport(cmd, **kwargs):
        if cmd[:2] != ['docker', 'exec']:
            return real_popen(cmd, **kwargs)
        # Model docker exec's container env: host env arrives only under named --env aliases.
        host_env = kwargs.get('env') or os.environ
        backend_env = {'PATH': container_path, 'HOME': '/container-home',
                       'CONTAINER_API_KEY': 'synthetic-leak', 'OUROBOROS_MANAGED_BY_LAUNCHER': '1'}
        for index, arg in enumerate(cmd):
            if arg == '--env':
                backend_env[cmd[index + 1]] = host_env[cmd[index + 1]]
        kwargs['env'] = backend_env
        if '--workdir' in cmd:
            kwargs['cwd'] = str(host_cwd(cmd[cmd.index('--workdir') + 1]))
        commands.append((cmd, kwargs))
        return real_popen(['sh', '-c', cmd[-1]], **kwargs)

    monkeypatch.setattr(executor.subprocess, 'Popen', transport)
    if consumer == 'command':
        result = executor.execute(ctx, probe, candidate, 10)
        assert result.returncode == 0 and result.operation_outcome == 'completed', result
        # Completion came from the wrapper's real wait receipt, read once by the guarded settlement probe.
        guarded = [cmd for cmd, _ in commands if cmd[-1].startswith('case "$(cat ')]
        assert len(guarded) == 1 and guarded[0][-1].endswith(' && printf completed'), guarded
        env = exported_env(result.stdout)
    else:
        # Execute the real service command expansion synchronously, preserving the
        # actual start_service branch and inert-env transport without orphan processes.
        original = executor._docker_service_start_shell
        captured = {}
        real_submit = executor._submit_service_command
        def service_shell(record, log_path, aliases=None, **kwargs):
            shell = original(record, log_path, aliases, **kwargs)
            # The same generated exec payload is inside the detached launch shell.
            payload = shlex.split(shell)[shlex.split(shell).index('-c') + 1]
            return f'cd {shlex.quote(str(host_cwd(record.backend_cwd)))} && {payload}'
        def submit(cmd, **kwargs):
            result = real_submit(cmd, **kwargs)
            assert result.returncode == 0, result.stderr
            captured['env'] = exported_env(result.stdout)
            result.stdout = '12345\n'  # only the fixture backend PID; keep the real confirmation owner
            return result
        monkeypatch.setattr(executor, '_docker_service_start_shell', service_shell)
        monkeypatch.setattr(executor, '_submit_service_command', submit)
        monkeypatch.setattr(executor, '_service_state', lambda record: 'exited')
        monkeypatch.setattr(executor, '_service_payload', lambda record, **kwargs: {'state': 'exited'})
        # No daemon answers here: the fixture container is its own name (a service without outputs pins it first).
        monkeypatch.setattr('ouroboros.workspace_executor_paths.pinned_container', lambda ref: ref)
        executor.start_service(ctx, name='env', cmd=probe, host_cwd=candidate,
                               cwd_root='system_repo', readiness={}, outputs=[], before_outputs={}, env={})
        env = captured['env']
    assert env['OUROBOROS_DATA_DIR'] == expected
    assert env['HOME'] == str(mount / (candidate.name + '.env') / 'home')
    assert env['OUROBOROS_REPO_DIR'] == str(mount / candidate.name)
    assert 'CONTAINER_API_KEY' not in env and 'OUROBOROS_MANAGED_BY_LAUNCHER' not in env
    assert not set(_DRIVER_KEYS) & set(env)  # the container runs its own runtime, never a host Node path
    assert not any(key.startswith(('OUROBOROS_SERVICE_ENV_', 'OUROBOROS_PROCESS_ENV_')) for key in env)
    assert commands and all('synthetic-leak' not in ' '.join(cmd) for cmd, _ in commands)


@pytest.mark.parametrize('operator', ['', '/operator/node'])
@pytest.mark.parametrize('node', ['cold', 'prepared'])
def test_only_a_local_candidate_process_gets_the_host_driver_node(tmp_path, monkeypatch, node, operator):
    """R4 (Fable): the candidate environment gained the serving installation's driver Node, a host path under
    <data>/state/cx/node, and the Docker projection maps every absolute value: map_host_path refused it, so every
    Docker command and service of a bound candidate or system copy was refused (CANDIDATE_ENVIRONMENT_UNAVAILABLE).
    The production projection itself, both executors: the host Node (carried or planned, beside an operator's literal
    driver) reaches a local process only; an operator's host path is not handed to a container either."""
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    _pinned_node(monkeypatch, node, data)
    if operator:
        monkeypatch.setenv('PLAYWRIGHT_NODEJS_PATH', operator)
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    ctx = make_ctx(serving, data, 'driver-node')
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    from types import SimpleNamespace
    local = body_candidate.executor_environment(
        ctx, candidate, executor=SimpleNamespace(kind='local'), map_path=executor.map_host_path)
    selected = local.get('OUROBOROS_PLAYWRIGHT_NODE_SELECTED', '')
    carried = local.get('OUROBOROS_PINNED_NODE_CARRIED', '')
    assert local['PLAYWRIGHT_NODEJS_PATH'] == (operator or selected)
    if operator:
        assert not selected  # an operator's literal driver is never marked automatic, even beside a carried Node
    if node == 'prepared':
        assert carried == _SERVING_NODE and selected == ('' if operator else _SERVING_NODE)
    elif not operator:  # nothing prepared: the generation the serving installation's own preparation publishes
        assert not carried
        assert pathlib.Path(selected).is_relative_to(data / 'state' / 'cx' / 'node')
        assert not pathlib.Path(selected).exists()
    ctx.executor_ref = {'type': 'docker_exec', 'container_name': 'fixture',
        'workspace_host_path': str(candidate.parent), 'workspace_backend_path': '/workspace'}
    docker = body_candidate.executor_environment(
        ctx, candidate, executor=executor.executor_ref_from_ctx(ctx), map_path=executor.map_host_path)
    assert not set(_DRIVER_KEYS) & set(docker)
    assert docker['OUROBOROS_DATA_DIR'] == f'/workspace/{candidate.name}.env/data'  # still the mapped isolated roots


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
