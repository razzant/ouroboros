"""File identity and ordinary task capabilities across real registry paths."""
from __future__ import annotations

import pathlib
import shlex
import sys

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import build_resolved_resource_binding, resource_root_path
from ouroboros.tools.registry import ToolContext, ToolRegistry

pytestmark = pytest.mark.serial


@pytest.fixture
def environment(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    system = tmp_path / 'system'
    work = home / 'project'
    data = tmp_path / 'data'
    for path in (home, system, work, data):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pathlib.Path, 'home', lambda: home)
    monkeypatch.setenv('OUROBOROS_USER_FILES_ROOT', str(home))
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setenv('OUROBOROS_SAFETY_MODE', 'off')
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=work,
                      workspace_mode='external', task_id='task')
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    return reg, ctx, home, work, data


@pytest.mark.parametrize('root', ['active_workspace', 'system_repo', 'runtime_data', 'task_drive', 'artifact_store'])
def test_absolute_in_root_keeps_exact_file_identity(environment, root):
    _reg, ctx, _home, _work, _data = environment
    base = resource_root_path(ctx, root)
    target = base / 'sub' / 'new.txt'
    binding = build_resolved_resource_binding(ctx, root=root, operation='read', path=str(target))
    assert binding.target_path == target.resolve()
    relative = build_resolved_resource_binding(ctx, root=root, operation='read', path='sub/new.txt')
    assert relative.target_path == binding.target_path


def test_external_absolute_does_not_read_or_create_a_repo_mirror(environment):
    reg, ctx, home, work, _data = environment
    outside = home / 'elsewhere' / 'target.txt'
    outside.parent.mkdir()
    outside.write_text('correct', encoding='utf-8')
    # The former lstrip('/') path is occupied, so NOT_FOUND cannot hide the bug.
    mirror = work.joinpath(*outside.parts[1:])
    mirror.parent.mkdir(parents=True)
    mirror.write_text('wrong-file', encoding='utf-8')
    read = reg.execute('read_file', {'root': 'active_workspace', 'path': str(outside)})
    assert 'outside selected root' in read and 'wrong-file' not in read
    write = reg.execute('write_file', {'root': 'active_workspace', 'path': str(outside), 'content': 'changed'})
    assert 'outside selected root' in write
    assert outside.read_text() == 'correct' and mirror.read_text() == 'wrong-file'
    assert 'correct' in reg.execute('read_file', {'root': 'user_files', 'path': str(outside)})
    with pytest.raises(ValueError, match='outside selected root'):
        ctx.repo_path(str(outside))


def test_runtime_prefix_requires_a_path_boundary(environment):
    _reg, ctx, _home, _work, data = environment
    with pytest.raises(ValueError, match='outside selected root'):
        build_resolved_resource_binding(ctx, root='runtime_data', operation='read', path=str(data) + '-other/logs/x')
    assert ctx.drive_path(str(data / 'logs' / 'x')) == (data / 'logs' / 'x').resolve()


def test_runtime_legacy_alias_does_not_rewrite_an_absolute_address(environment):
    _reg, ctx, _home, _work, data = environment
    outside = pathlib.Path(data.anchor) / '.tmp-data-old' / 'data' / 'logs' / 'x'
    with pytest.raises(ValueError, match='outside selected root'):
        build_resolved_resource_binding(ctx, root='runtime_data', operation='read', path=str(outside))
    relative = build_resolved_resource_binding(ctx, root='runtime_data', operation='read',
                                                path='.tmp-data-old/data/logs/x')
    assert relative.target_path == (data / 'logs' / 'x').resolve()


@pytest.mark.skipif(sys.platform == 'win32', reason='foreign Windows syntax on a POSIX host')
def test_foreign_absolute_address_does_not_acquire_the_process_cwd(environment, monkeypatch):
    _reg, ctx, _home, work, _data = environment
    monkeypatch.chdir(work)
    with pytest.raises(ValueError, match='outside selected root'):
        ctx.repo_path('Z:/outside.txt')


@pytest.mark.parametrize('tool', ['edit_batch', 'apply_patch'])
def test_repo_only_edit_refuses_unsupported_root_before_payload_resolution(environment, tool):
    from ouroboros.loop_tool_execution import _extract_result_metadata
    reg, _ctx, _home, work, data = environment
    payload = {'root': 'skill_payload'}
    if tool == 'edit_batch':
        payload['edits'] = [{'path': 'mod.py', 'old_str': 'one', 'new_str': 'two'}]
    else:
        payload['patch'] = '*** Update File: mod.py\n-one\n+two\n'
    result = reg.execute(tool, payload)
    expected = 'EDIT_BATCH_ERROR' if tool == 'edit_batch' else 'APPLY_PATCH_BLOCKED'
    assert expected in result
    assert 'write_file/edit_text' in result and 'TOOL_ERROR' not in result
    assert _extract_result_metadata(tool, result, False)['status'] == 'edit_ops_blocked'
    assert not (data / 'skills').exists() and not (work / 'mod.py').exists()


@pytest.mark.parametrize('relative', ['settings.json', '.config/app/options.conf', 'Library/Preferences/app.conf', '.ssh/config'])
def test_root_can_read_and_edit_ordinary_config(environment, relative):
    reg, _ctx, home, _work, _data = environment
    target = home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('old configuration\n', encoding='utf-8')
    assert 'old configuration' in reg.execute('read_file', {'root': 'user_files', 'path': str(target)})
    result = reg.execute('edit_text', {'root': 'user_files', 'path': str(target),
                                    'old_str': 'old configuration', 'new_str': 'new configuration'})
    assert result.startswith('OK: edited'), result
    assert target.read_text() == 'new configuration\n'


def test_owner_key_write_and_config_symlink_remain_blocked(environment):
    reg, _ctx, home, _work, _data = environment
    key = home / '.ssh' / 'authorized_keys'
    key.parent.mkdir()
    key.write_text('owner-key', encoding='utf-8')
    result = reg.execute('write_file', {'root': 'user_files', 'path': str(key), 'content': 'changed'})
    assert 'BLOCKED' in result or 'ERROR' in result
    assert key.read_text() == 'owner-key'
    config = key.parent / 'config'
    try:
        config.symlink_to(key)
    except OSError:
        pytest.skip('symlink unavailable')
    result = reg.execute('write_file', {'root': 'user_files', 'path': str(config), 'content': 'changed'})
    assert 'BLOCKED' in result or 'ERROR' in result
    assert key.read_text() == 'owner-key'


@pytest.mark.parametrize('mode', ['light', 'advanced'])
def test_root_shell_write_authority_does_not_depend_on_cwd(environment, monkeypatch, mode):
    reg, _ctx, home, work, _data = environment
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', mode)
    target = home / 'other' / 'settings.json'
    target.parent.mkdir()
    body = 'from pathlib import Path; Path(' + repr(str(target)) + ').write_text("result")'
    result = reg.execute('run_command', {'cmd': [sys.executable, '-c', body],
                                       'cwd': str(work), 'outputs': [str(target)]})
    assert target.read_text() == 'result', result
    assert 'WORKSPACE_SHELL_BLOCKED' not in result


@pytest.mark.parametrize('body', [
    'from pathlib import Path; print(Path(PATH).read_text()); print(1 > 0)',
    'from pathlib import Path; print(Path(PATH).read_text()); print("form truncated =>")',
])
def test_light_reads_logs_with_operator_or_prose_bytes(environment, monkeypatch, body):
    reg, ctx, _home, _work, data = environment
    # The ordinary root diagnoses its own runtime; external-project shell has
    # an independent runtime-read boundary, so use the ordinary task profile.
    ctx.workspace_mode = ''
    ctx.workspace_root = None
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'light')
    path = data / 'logs' / 'sample.jsonl'
    path.parent.mkdir()
    path.write_text('LOG_CONTENT', encoding='utf-8')
    result = reg.execute('run_command', {'cmd': [sys.executable, '-c', body.replace('PATH', repr(str(path)))],
                                       'cwd': 'task_drive'})
    assert 'LOG_CONTENT' in result and 'LIGHT_MODE_BLOCKED' not in result


def test_light_mixed_log_read_and_own_output_preserves_control_boundary(environment, monkeypatch):
    reg, ctx, _home, _work, data = environment
    ctx.workspace_mode = ''
    ctx.workspace_root = None
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'light')
    path = data / 'logs' / 'sample.jsonl'
    path.parent.mkdir()
    path.write_text('LOG_CONTENT', encoding='utf-8')
    output = resource_root_path(ctx, 'task_drive') / 'output.txt'
    body = f'from pathlib import Path; text=Path({str(path)!r}).read_text(); print(len(text)>0); Path({str(output)!r}).write_text(text)'
    result = reg.execute('run_script', {'script': body, 'cwd': 'task_drive'})
    assert output.read_text() == 'LOG_CONTENT', result
    denied = reg.execute('run_command', {'cmd': [sys.executable, '-c', f'open({str(path)!r},"w").write("bad")'], 'cwd': 'task_drive'})
    assert 'LIGHT_MODE_BLOCKED' in denied and path.read_text() == 'LOG_CONTENT'
    (data / 'settings.json').write_text('{}', encoding='utf-8')
    secret = reg.execute('run_command', {'cmd': [sys.executable, '-c', f'print(open({str(data / "settings.json")!r}).read()); print(1>0)'], 'cwd': 'task_drive'})
    assert 'LIGHT_MODE_BLOCKED' in secret


@pytest.mark.parametrize('project_id', ['', 'current-project'])
def test_readonly_child_can_review_auth_sources_and_scoped_knowledge(environment, monkeypatch, project_id):
    from ouroboros import config

    reg, ctx, _home, work, data = environment
    monkeypatch.setattr(config, 'DATA_DIR', data)
    ctx.project_id = project_id
    ctx.task_constraint = TaskConstraint(mode='local_readonly_subagent')
    path = work / 'src' / 'auth' / 'login.py'
    path.parent.mkdir(parents=True)
    token = 'sk-' + 'a' * 48
    path.write_text('def login():\n    return "' + token + '"  # AUTH_SOURCE\n', encoding='utf-8')
    read = reg.execute('read_file', {'path': 'src/auth/login.py'})
    assert 'def login' in read and token not in read and '***' in read
    assert 'auth/' in reg.execute('list_files', {'path': 'src'})
    search = reg.execute('search_code', {'query': 'AUTH_SOURCE', 'path': 'src'})
    assert 'AUTH_SOURCE' in search and token not in search
    assert 'login' in reg.execute('query_code', {'op': 'symbols', 'path': 'src/auth/login.py'})
    knowledge = data / 'projects' / project_id / 'knowledge' if project_id else data / 'memory' / 'knowledge'
    knowledge.mkdir(parents=True)
    (knowledge / 'topic.md').write_text('# Topic\n\nKNOWLEDGE_READ', encoding='utf-8')
    other = data / 'projects' / 'other-project' / 'knowledge'
    other.mkdir(parents=True)
    (other / 'hidden.md').write_text('# Other\n\nOTHER_PROJECT_FACT', encoding='utf-8')
    before = sorted(str(p) for p in data.rglob('*'))
    listing = reg.execute('knowledge_list', {})
    assert 'topic' in listing and 'OTHER_PROJECT_FACT' not in listing
    assert 'KNOWLEDGE_READ' in reg.execute('knowledge_read', {'topic': 'topic'})
    assert 'OTHER_PROJECT_FACT' not in reg.execute('knowledge_read', {'topic': 'hidden'})
    assert sorted(str(p) for p in data.rglob('*')) == before
    assert reg.get_schema_by_name('knowledge_write') is None
    assert 'LOCAL_READONLY_SUBAGENT_BLOCKED' in reg.execute('write_file', {'path': 'x', 'content': 'x'})


def test_acting_shell_uses_path_identity_not_code_substrings(environment):
    from tests._typed_guard_shared import _shell_guard_text
    reg, ctx, _home, work, data = environment
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    for body in [
        "import os; print(os.environ.get('PATH', ''))",
        "print('file1.txt is just a description')",
        "print(open('.env.example').read())",
    ]:
        assert _shell_guard_text(reg, {'cmd': [sys.executable, '-c', body]}, 'advanced') is None
    actual = _shell_guard_text(reg, {'cmd': [sys.executable, '-c', f'print(open({str(data / "settings.json")!r}).read())']}, 'advanced')
    assert actual and 'SUBAGENT_SECRET_READ_BLOCKED' in actual
    outside = work.parent / 'outside.txt'
    denied = _shell_guard_text(reg, {'cmd': [sys.executable, '-c', f'open({str(outside)!r},"w").write("bad")']}, 'advanced')
    assert denied and 'WORKSPACE_SHELL_BLOCKED' in denied


def test_ssh_subject_separates_remote_payload_and_local_channels():
    from ouroboros.shell_parse import local_shell_subject
    from ouroboros.tools.shell_guards import writer_target_rows
    raw = ['ssh', '-p', '2222', '-E', '/tmp/ssh.log', 'host', 'sudo -n tee /etc/remote.conf']
    local = local_shell_subject(raw)
    assert '/etc/remote.conf' not in repr(local)
    assert '/tmp/ssh.log' in [target for _argv, targets, _body, _unknown in writer_target_rows(local) for target in targets]
    wrapped = ['sh', '-c', "ssh host 'cat /remote/source' < local.in > local.out"]
    local = local_shell_subject(wrapped)
    assert '/remote/source' not in repr(local)
    assert 'local.in' in local
    assert 'local.out' in [target for _argv, targets, _body, _unknown in writer_target_rows(local) for target in targets]
    assert raw[-1] == 'sudo -n tee /etc/remote.conf'


def test_remote_paths_are_not_child_local_writes_but_outer_redirects_are(environment):
    from tests._typed_guard_shared import _shell_guard_text
    reg, ctx, _home, work, _data = environment
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    remote = ['ssh', 'host', 'sudo -n tee /etc/remote.conf']
    assert _shell_guard_text(reg, {'cmd': remote}, 'advanced') is None
    outside = work.parent / 'outside.txt'
    redirected = ['sh', '-c', "ssh host 'cat /remote/source' > " + shlex.quote(str(outside))]
    denied = _shell_guard_text(reg, {'cmd': redirected}, 'advanced')
    assert denied and 'WORKSPACE_SHELL_BLOCKED' in denied
    log = ['ssh', '-E', str(outside), 'host', 'cat /remote/source']
    denied = _shell_guard_text(reg, {'cmd': log}, 'advanced')
    assert denied and 'WORKSPACE_SHELL_BLOCKED' in denied


@pytest.mark.skipif(sys.platform == 'win32', reason='the test SSH fixture uses a POSIX executable shim')
def test_ssh_remote_task_runs_with_original_argv(environment):
    reg, _ctx, home, work, _data = environment
    fixture = home / 'bin' / 'ssh'
    fixture.parent.mkdir()
    fixture.write_text('#!' + sys.executable + '\nimport sys\nprint("REMOTE_OK:" + sys.argv[-1])\n', encoding='utf-8')
    fixture.chmod(0o755)
    payload = 'sudo -n tee /etc/remote.conf'
    result = reg.execute('run_command', {'cmd': [str(fixture), 'fixture-host', payload], 'cwd': str(work)})
    assert 'REMOTE_OK:' + payload in result, result
    assert 'WORKSPACE_SHELL_BLOCKED' not in result


@pytest.mark.parametrize('actor', ['acting_subagent', 'local_readonly_subagent'])
@pytest.mark.parametrize('command', [
    ['grep', '-rn', 'token', 'src/'], ['rg', 'password', '.'],
    ['pytest', '-k', 'secret'], ['git', 'log', '--grep', 'credential'],
    [sys.executable, '-c', "print('token')"],
])
def test_child_search_words_are_not_credential_path_operands(environment, actor, command):
    from copy import deepcopy
    from tests._typed_guard_shared import _shell_guard_text

    reg, ctx, _home, work, _data = environment
    ctx.task_constraint = (TaskConstraint(mode=actor, surface='external_workspace', write_root=str(work))
                           if actor == 'acting_subagent' else TaskConstraint(mode=actor))
    original = deepcopy(command)
    if actor == 'local_readonly_subagent':
        # This profile exposes file inspection, not shell execution. Exercise
        # the shared predicate without inventing a shell capability for it.
        from ouroboros.tools.registry_guard_process import _subagent_shell_targets_secret
        assert not _subagent_shell_targets_secret(command, ctx=ctx, cwd=work)
        assert reg.get_schema_by_name('run_command') is None
    else:
        assert _shell_guard_text(reg, {'cmd': command, 'cwd': str(work)}, 'advanced') is None
    assert command == original


@pytest.mark.parametrize('wrapper', ['direct', 'env', 'sh', 'sh_env'])
def test_wrapped_inline_credential_read_is_blocked_at_the_same_physical_target(environment, wrapper):
    reg, ctx, _home, work, _data = environment
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    (work / '.env').write_text('FIXTURE_SECRET_MUST_NOT_REACH_OUTPUT', encoding='utf-8')
    command = [sys.executable, '-c', "print(open('.env').read())"]
    if wrapper in {'env', 'sh_env'}:
        command = ['env', *command]
    if wrapper in {'sh', 'sh_env'}:
        command = ['sh', '-c', shlex.join(command)]
    result = reg.execute_result('run_command', {'cmd': command, 'cwd': str(work)})
    assert (result.status, result.code) == ('blocked', 'SUBAGENT_SECRET_READ_BLOCKED')
    assert 'FIXTURE_SECRET_MUST_NOT_REACH_OUTPUT' not in result.text


def test_secret_paths_use_sequential_and_env_local_cwd(environment):
    from ouroboros.tools.registry_guard_process import _subagent_shell_targets_secret

    _reg, ctx, home, work, data = environment
    source = work / 'src'
    source.mkdir()
    (source / 'settings.json').write_text('ordinary project config', encoding='utf-8')
    (data / 'settings.json').write_text('runtime control fixture', encoding='utf-8')
    body = "print(open('settings.json').read())"
    python = shlex.join([sys.executable, '-c', body])
    assert _subagent_shell_targets_secret(['sh', '-c', f'cd {shlex.quote(str(data))}; {python}'], ctx=ctx, cwd=work)
    assert not _subagent_shell_targets_secret(
        ['sh', '-c', f'cd {shlex.quote(str(source))}; {python}'], ctx=ctx, cwd=data)
    assert _subagent_shell_targets_secret(['env', '-C', str(data), sys.executable, '-c', body], ctx=ctx, cwd=work)
    # env -C applies only to its command; it must not retarget a later reader.
    command = ['sh', '-c', f'env -C {shlex.quote(str(data))} true; {python}']
    assert not _subagent_shell_targets_secret(command, ctx=ctx, cwd=source)
    key = home / '.ssh' / 'id_fixture'
    key.parent.mkdir()
    key.write_text('owner-key-fixture', encoding='utf-8')
    assert _subagent_shell_targets_secret(['sh', '-c', 'true', '<', str(key)], ctx=ctx, cwd=work)


@pytest.mark.skipif(sys.platform == 'win32', reason='actual POSIX sh/env execution')
@pytest.mark.parametrize('wrapper', ['direct', 'env', 'sh'])
def test_child_inline_source_read_and_bare_word_execute_without_argv_rewrite(environment, wrapper):
    reg, ctx, _home, work, _data = environment
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    source = work / 'src' / 'auth' / 'source.py'
    source.parent.mkdir(parents=True)
    source.write_text('SOURCE_READ_OK', encoding='utf-8')
    command = [sys.executable, '-c', "print('token'); print(open('src/auth/source.py').read())"]
    if wrapper == 'env':
        command = ['env', *command]
    elif wrapper == 'sh':
        command = ['sh', '-c', shlex.join(command)]
    result = reg.execute_result('run_command', {'cmd': command, 'cwd': str(work)})
    assert 'SOURCE_READ_OK' in result.text and 'token' in result.text
    assert result.status != 'blocked'


@pytest.mark.parametrize('spelling', [
    '$OUROBOROS_DATA_DIR', '${OUROBOROS_DATA_DIR}', '%OUROBOROS_DATA_DIR%',
    '$HOME/Ouroboros/data', '${HOME}/Ouroboros/data', '%USERPROFILE%/Ouroboros/data', '~/Ouroboros/data',
])
def test_known_root_expansion_applies_to_write_targets_only(tmp_path, monkeypatch, spelling):
    from ouroboros.tools.shell_guards import runtime_data_guard_targets

    home = tmp_path / 'home'
    data = home / 'Ouroboros' / 'data'
    scratch = data / 'task_drives' / 'current'
    scratch.mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, 'home', lambda: home)
    def targets(body):
        return runtime_data_guard_targets(['sh', '-c', body], writeish=True,
            drive_root=data, work_dir=scratch, allowed_roots=[scratch])
    log = data / 'logs' / 'events.jsonl'
    assert str(log) in targets(f'printf changed > "{spelling}/logs/events.jsonl"')
    assert targets(f'cat "{spelling}/logs/events.jsonl" > "{spelling}/task_drives/current/copy.txt"') == []


@pytest.mark.parametrize('command', [
    ['sh', '-c', 'cat $HOME/.ssh/id_fixture'],
    ['sh', '-c', 'cat ${HOME}/file1.txt'],
    ['sh', '-c', 'cd $HOME && cat .ssh/id_fixture'],
    ['sh', '-c', 'env -C $HOME cat .ssh/id_fixture'],
    ['sh', '-c', 'cat $OUROBOROS_DATA_DIR/settings.json'],
    ['cmd', '/c', 'type %USERPROFILE%/.ssh/id_fixture'],
])
def test_child_known_root_credential_reads_are_blocked(environment, monkeypatch, command):
    reg, ctx, home, work, data = environment
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(data))
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    for target in (home / '.ssh/id_fixture', home / 'file1.txt', data / 'settings.json'):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('FIXTURE_SECRET_MUST_NOT_REACH_OUTPUT', encoding='utf-8')
    original = list(command)
    result = reg.execute_result('run_command', {'cmd': command, 'cwd': str(work)})
    assert (result.status, result.code) == ('blocked', 'SUBAGENT_SECRET_READ_BLOCKED')
    assert 'FIXTURE_SECRET_MUST_NOT_REACH_OUTPUT' not in result.text
    assert command == original


@pytest.mark.skipif(sys.platform == 'win32', reason='actual POSIX shell expansion')
def test_child_known_root_source_read_keeps_shell_capability(environment, monkeypatch):
    reg, ctx, home, work, _data = environment
    monkeypatch.setenv('HOME', str(home))
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    (work / 'README.md').write_text('SOURCE_READ_OK', encoding='utf-8')
    command = ['sh', '-c', 'cat "$HOME/project/README.md"; printf "token\\n"']
    result = reg.execute_result('run_command', {'cmd': command, 'cwd': str(work)})
    assert result.status == 'ok' and 'SOURCE_READ_OK' in result.text and 'token' in result.text
    assert '$HOME/project/README.md' in command[2]
