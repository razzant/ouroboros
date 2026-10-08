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


@pytest.mark.parametrize('location,relative', [
    ('repo', 'config/.env'),
    ('repo', 'deploy/credentials.json'),
    ('data', 'claudexor/profile/session/auth.json'),
])
def test_child_file_reads_match_command_reads(environment, location, relative):
    reg, ctx, _home, work, data = environment
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    target = (work if location == 'repo' else data) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('unprefixed-credential-canary', encoding='utf-8')
    command = [sys.executable, '-c', f'from pathlib import Path; print(Path({str(target)!r}).read_text())']
    read = reg.execute('read_file', {'root': 'active_workspace' if location == 'repo' else 'runtime_data', 'path': relative})
    assert 'unprefixed-credential-canary' in read
    shell = reg.execute('run_command', {'cmd': command, 'cwd': str(work)})
    assert 'unprefixed-credential-canary' in shell, shell


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
def test_payload_edit_requires_a_skill_selector(environment, tool):
    reg, _ctx, _home, work, data = environment
    payload = {'root': 'skill_payload'}
    if tool == 'edit_batch':
        payload['edits'] = [{'path': 'mod.py', 'old_str': 'one', 'new_str': 'two'}]
    else:
        payload['patch'] = '*** Update File: mod.py\n-one\n+two\n'
    result = reg.execute(tool, payload)
    assert 'SKILL_PAYLOAD_ARG_ERROR' in result and 'TOOL_ERROR' not in result
    assert not (data / 'skills').exists() and not (work / 'mod.py').exists()


@pytest.mark.parametrize('relative', ['settings.json', '.config/app/options.conf', 'Library/Preferences/app.conf', '.ssh/config'])
def test_root_can_read_and_edit_ordinary_config(environment, relative):
    reg, _ctx, home, _work, _data = environment
    target = home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    old = '"old configuration"\n' if relative.endswith('.json') else 'old configuration\n'
    target.write_text(old, encoding='utf-8')
    assert 'old configuration' in reg.execute('read_file', {'root': 'user_files', 'path': str(target)})
    result = reg.execute('edit_text', {'root': 'user_files', 'path': str(target),
                                    'old_str': 'old configuration', 'new_str': 'new configuration'})
    assert result.startswith('OK: edited'), result
    assert target.read_text() == old.replace('old configuration', 'new configuration')


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


def test_light_shell_work_preserves_the_structured_settings_write_boundary(environment, monkeypatch):
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
    changed = reg.execute('run_command', {'cmd': [sys.executable, '-c', f'open({str(path)!r},"w").write("updated")'], 'cwd': 'task_drive'})
    assert 'exit_code=0' in changed and path.read_text() == 'updated'
    (data / 'settings.json').write_text('{}', encoding='utf-8')
    settings_write = reg.execute('write_file', {'root': 'runtime_data', 'path': 'settings.json', 'content': 'updated'})
    assert 'BLOCKED' in settings_write
    assert (data / 'settings.json').read_text() == '{}'


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
    assert 'def login' in read and token in read
    assert 'auth/' in reg.execute('list_files', {'path': 'src'})
    search = reg.execute('search_code', {'query': 'AUTH_SOURCE', 'path': 'src'})
    assert 'AUTH_SOURCE' in search and token in search
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
    # A child writes knowledge and chronicle drafts in its own name; identity stays the parent's.
    assert reg.get_schema_by_name('knowledge_write') is not None
    assert reg.get_schema_by_name('update_identity') is None
    assert reg.get_schema_by_name('chronicle_write') is not None
    assert 'LOCAL_READONLY_SUBAGENT_BLOCKED' in reg.execute('write_file', {'path': 'x', 'content': 'x'})


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


@pytest.mark.parametrize('remote', [False, True])
@pytest.mark.parametrize('control', ['elevation', 'owner_key'])
def test_ssh_keeps_full_command_visible_to_supervisor_and_handler(environment, monkeypatch, remote, control):
    reg, ctx, home, work, _data = environment
    commands = {
        'elevation': [sys.executable, '-c',
                      "from ouroboros.config import save_settings; save_settings({'OUROBOROS_RUNTIME_MODE':'pro'})"],
        'owner_key': ['cat', str(home / '.ssh' / 'id_fixture')],
    }
    if control == 'owner_key':
        ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    command = commands[control]
    command = ['ssh', 'localhost', *command] if remote else command
    original = list(command)
    observed = []

    def supervisor(_name, args, *_args, **_kwargs):
        observed.append(('supervisor', list(args['cmd'])))
        return True, ''

    def handler(_ctx, cmd, _resolved_binding=None, **_kwargs):
        observed.append(('handler', list(cmd)))
        return 'accepted request'

    monkeypatch.setattr('ouroboros.safety.check_safety', supervisor)
    reg.override_handler('run_command', handler)
    result = reg.execute('run_command', {'cmd': command, 'cwd': str(work)})
    assert result == 'accepted request'
    assert observed == [('supervisor', original), ('handler', original)]
    assert command == original


@pytest.mark.parametrize('root', ['active_workspace', 'task_drive', 'artifact_store'])
@pytest.mark.parametrize('directory', ['ordinary', 'auth', 'tokens'])
def test_child_source_names_and_certificates_do_not_confer_credential_authority(environment, root, directory):
    reg, ctx, _home, _work, _data = environment
    ctx.task_constraint = TaskConstraint(mode='local_readonly_subagent')
    base = resource_root_path(ctx, root)
    source = base / 'src' / directory / 'login.py'
    source.parent.mkdir(parents=True)
    token = 'sk-' + 'a' * 48
    source.write_text('def login():\n    return "' + token + '"  # SOURCE_AVAILABLE\n', encoding='utf-8')
    result = reg.execute('read_file', {'root': root, 'path': source.relative_to(base).as_posix()})
    assert 'SOURCE_AVAILABLE' in result and token in result
    # Both public and private PEM fixture content survive whole and partial reads.
    certificate = '-----BEGIN CERTIFICATE-----\n' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' * 2 + '\n-----END CERTIFICATE-----\n'
    public = source.parent / 'public.pem'
    public.write_text(certificate, encoding='utf-8')
    from ouroboros.tools.core_file_tools import delivered_source_prefix

    public_result = reg.execute('read_file', {'root': root, 'path': public.relative_to(base).as_posix()})
    assert delivered_source_prefix(ctx.last_read_view, public_result, len(public_result)) == certificate
    private = source.parent / 'fixture.pem'
    material = 'fixture-private-material-0123456789'
    private.write_text('-----BEGIN PRIVATE KEY-----\n' + material + '\n-----END PRIVATE KEY-----\n', encoding='utf-8')
    for start_line in (1, 2):
        read = reg.execute('read_file', {'root': root, 'path': private.relative_to(base).as_posix(), 'start_line': start_line})
        assert material in read
        assert ctx.last_read_view['total_lines'] == 3


@pytest.mark.parametrize('root', ['task_drive', 'artifact_store'])
@pytest.mark.parametrize('path', ['auth/login.py', 'tokens/login.py', 'tokens.json'])
def test_child_own_task_content_is_not_a_repository_credential_store(environment, root, path):
    reg, ctx, _home, _work, _data = environment
    ctx.task_constraint = TaskConstraint(mode='local_readonly_subagent')
    target = resource_root_path(ctx, root) / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('TASK_CONTENT_AVAILABLE', encoding='utf-8')
    assert 'TASK_CONTENT_AVAILABLE' in reg.execute('read_file', {'root': root, 'path': path})
    assert path.split('/')[0] in reg.execute('list_files', {'root': root, 'path': '.'})
    # The matrix is closed under read⇒search (TZ-1 E): the child searches its own
    # task content exactly where it reads it, and the shared path policy still does
    # not mistake an ordinary auth/ path in task content for a credential store.
    assert 'TASK_CONTENT_AVAILABLE' in reg.execute('search_code', {'root': root, 'path': '.', 'query': 'TASK_CONTENT_AVAILABLE'})


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

    reg, ctx, _home, work, _data = environment
    ctx.task_constraint = (TaskConstraint(mode=actor, surface='external_workspace', write_root=str(work))
                           if actor == 'acting_subagent' else TaskConstraint(mode=actor))
    original = deepcopy(command)
    if actor == 'local_readonly_subagent':
        # This profile exposes file inspection, not shell execution.
        assert reg.get_schema_by_name('run_command') is None
    else:
        seen = []

        def handler(_ctx, cmd, _resolved_binding=None, **_kwargs):
            seen.append(list(cmd))
            return 'search dispatched'

        reg.override_handler('run_command', handler)
        assert reg.execute('run_command', {'cmd': command, 'cwd': str(work)}) == 'search dispatched'
        assert seen == [original]
    assert command == original


@pytest.mark.parametrize('wrapper', ['direct', 'env', 'sh', 'sh_env'])
@pytest.mark.parametrize('allowed', [False, True])
def test_wrapped_inline_read_uses_the_configured_supervisor(environment, monkeypatch, wrapper, allowed):
    reg, ctx, _home, work, _data = environment
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    (work / '.env').write_text('FIXTURE_SECRET_MUST_NOT_REACH_OUTPUT', encoding='utf-8')
    command = [sys.executable, '-c', "print(open('.env', encoding='utf-8').read())"]
    if wrapper in {'env', 'sh_env'}:
        command = ['env', *command]
    if wrapper in {'sh', 'sh_env'}:
        command = ['sh', '-c', shlex.join(command)]
    decisions = []

    def supervisor(_name, args, *_args, **_kwargs):
        decisions.append(list(args['cmd']))
        return allowed, 'Supervisor fixture decision'

    monkeypatch.setenv('OUROBOROS_SAFETY_MODE', 'full')
    monkeypatch.setattr('ouroboros.safety.check_safety', supervisor)
    result = reg.execute_result('run_command', {'cmd': command, 'cwd': str(work)})
    assert decisions == [command]
    if allowed:
        assert result.status == 'ok' and 'FIXTURE_SECRET_MUST_NOT_REACH_OUTPUT' in result.text
    else:
        assert (result.status, result.code) == ('blocked', 'SAFETY_VIOLATION')
        assert 'FIXTURE_SECRET_MUST_NOT_REACH_OUTPUT' not in result.text


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
    ['sh', '-c', 'cd $HOME && cat .ssh/id_fixture'],
    ['sh', '-c', 'env -C $HOME cat .ssh/id_fixture'],
    ['sh', '-c', 'cat $OUROBOROS_DATA_DIR/settings.json'],
    ['cmd', '/c', 'type %USERPROFILE%/.ssh/id_fixture'],
])
def test_child_known_root_read_reaches_the_configured_supervisor(environment, monkeypatch, command):
    reg, ctx, home, work, data = environment
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(data))
    ctx.task_constraint = TaskConstraint(mode='acting_subagent', surface='external_workspace', write_root=str(work))
    for target in (home / '.ssh/id_fixture', data / 'settings.json'):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('FIXTURE_SECRET_MUST_NOT_REACH_OUTPUT', encoding='utf-8')
    original = list(command)
    decisions = []

    def supervisor(_name, args, *_args, **_kwargs):
        decisions.append(list(args['cmd']))
        return False, 'Supervisor fixture decision'

    monkeypatch.setenv('OUROBOROS_SAFETY_MODE', 'full')
    monkeypatch.setattr('ouroboros.safety.check_safety', supervisor)
    result = reg.execute_result('run_command', {'cmd': command, 'cwd': str(work)})
    assert (result.status, result.code) == ('blocked', 'SAFETY_VIOLATION')
    assert decisions == [original]
    assert 'FIXTURE_SECRET_MUST_NOT_REACH_OUTPUT' not in result.text
    assert command == original


@pytest.mark.parametrize('name', ['draft-notes.txt', 'notes.txt'])
def test_ordinary_home_text_file_keeps_native_read_and_write(environment, name):
    reg, _ctx, home, _work, _data = environment
    target = home / name
    result = reg.execute_result('write_file', {'root': 'user_files', 'path': str(target), 'content': 'ORDINARY_TEXT_OK'})
    assert result.status == 'ok'
    assert target.read_text() == 'ORDINARY_TEXT_OK'
    read = reg.execute_result('read_file', {'root': 'user_files', 'path': str(target)})
    assert read.status == 'ok' and 'ORDINARY_TEXT_OK' in read.text


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


@pytest.mark.parametrize('mode', ['light', 'advanced', 'pro', 'cyber_pro'])
@pytest.mark.parametrize('actor', ['parent', 'local_readonly_subagent', 'acting_subagent'])
def test_parent_runtime_read_rules_follow_physical_files_across_root_labels(tmp_path, monkeypatch, mode, actor):
    from ouroboros import config
    from ouroboros.tools import vision

    repo = tmp_path / 'repo'
    data = repo / 'runtime'
    data.mkdir(parents=True)
    monkeypatch.setattr(config, 'DATA_DIR', data)
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', mode)
    monkeypatch.setenv('OUROBOROS_SAFETY_MODE', 'off')
    registry = ToolRegistry(repo, data)
    ctx = registry._ctx
    if actor != 'parent':
        ctx.task_constraint = TaskConstraint(mode=actor, surface='external_workspace', write_root=str(repo))
    sources = {'.git/HEAD': 'ref: refs/heads/synthetic\n', '.env': 'SYNTHETIC_ENV_CONTENT',
               'runtime/settings.json': '{"key": "SYNTHETIC_SETTINGS_CONTENT"}'}
    for relative, content in sources.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        assert content in registry.execute('read_file', {'path': relative})
    assert '.git/' in registry.execute('list_files', {})
    project = data / 'projects' / 'other' / 'hidden.py'
    project.parent.mkdir(parents=True)
    project.write_text('def hidden_project_fact():\n    pass\n', encoding='utf-8')
    owner = data / 'state' / 'skills' / 'sample' / 'grants.json'
    owner.parent.mkdir(parents=True)
    owner.write_text('{"value": "SYNTHETIC_OWNER_STATE"}', encoding='utf-8')
    for root, prefix in [('runtime_data', ''), ('active_workspace', 'runtime/')]:
        blocked = registry.execute('read_file', {'root': root, 'path': prefix + 'projects/other/hidden.py'})
        assert 'ACCESS_DENIED' in blocked and 'hidden_project_fact' not in blocked
        listing = registry.execute('list_files', {'root': root, 'path': prefix or '.'})
        assert 'projects/' not in listing
        viewed = registry.execute('read_file', {'root': root, 'path': prefix + 'state/skills/sample/grants.json'})
        assert ('SYNTHETIC_OWNER_STATE' in viewed) is (mode == 'cyber_pro')
        search = registry.execute('search_code', {'root': root, 'path': prefix or '.', 'query': 'SYNTHETIC'})
        assert 'SYNTHETIC_SETTINGS_CONTENT' in search
        assert ('SYNTHETIC_OWNER_STATE' in search) is (mode == 'cyber_pro')
    assert 'ACCESS_DENIED' in vision._read_file_parity_block(ctx, project)
    assert bool(vision._read_file_parity_block(ctx, owner)) is (mode != 'cyber_pro')
    # Admission precedes source hashing/parsing, including a previously cached
    # projection. A partial policy view must not overwrite the shared cache.
    from ouroboros import code_intelligence
    cached = code_intelligence.inventory_cache_path(repo, data)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text('{"synthetic": "unchanged cache"}', encoding='utf-8')
    cache_before = cached.read_bytes()
    # Source readers use bounded Path.open('rb'); read_bytes/read_text also open.
    original_open = pathlib.Path.open
    hidden = project.resolve()

    def permitted_open(path, *args, **kwargs):
        if path.resolve() == hidden:
            pytest.fail('project-store source was read before admission')
        return original_open(path, *args, **kwargs)

    # A bare-symbol owner_of reads the inventory through that same admission.
    (repo / 'ouroboros').mkdir()
    (repo / 'ouroboros' / 'domains.toml').write_text(
        '[domains]\nD01 = "Synthetic"\n\n[modules]\n"ouroboros/owned.py" = "D01"\n', encoding='utf-8')
    (repo / 'ouroboros' / 'owned.py').write_text('def owned_fact():\n    pass\n', encoding='utf-8')
    monkeypatch.setattr(pathlib.Path, 'open', permitted_open)
    for op, options in [('symbols', {}), ('digest', {}), ('structural', {'query': 'FunctionDef'}),
                        ('architecture', {'query': 'owner_of owned_fact'})]:
        result = registry.execute('query_code', {'op': op, **options})
        assert 'hidden_project_fact' not in result
        assert 'projects/other/hidden.py' not in result
        assert op != 'architecture' or 'ouroboros/owned.py -> D01 (Synthetic) [symbol_definition]' in result, result
    assert cached.read_bytes() == cache_before


def test_project_settings_source_is_readable_by_verify_guard(tmp_path):
    from ouroboros.tools.shell_guards import process_shell_guard_args
    from tests._typed_guard_shared import _shell_guard_text

    repo, data = tmp_path / "repo", tmp_path / "runtime"
    (repo / "data").mkdir(parents=True)
    data.mkdir()
    (repo / "data" / "settings.json").write_text('{"ordinary": "project fixture"}', encoding="utf-8")
    registry = ToolRegistry(repo, data)
    registry._ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(repo))
    mapped = process_shell_guard_args("verify_and_record", {"check": "cat data/settings.json", "cwd": str(repo)})
    result = _shell_guard_text(registry, mapped, "advanced")
    assert result is None, result
    assert "project fixture" in registry.execute("read_file", {"path": "data/settings.json"})


@pytest.mark.parametrize('mode', ['light', 'advanced', 'pro', 'cyber_pro'])
@pytest.mark.parametrize('actor', ['parent', 'local_readonly_subagent', 'acting_subagent'])
@pytest.mark.parametrize('fallback', [False, True])
def test_explicit_user_files_retains_parent_mode_policy(tmp_path, monkeypatch, mode, actor, fallback):
    from ouroboros import config
    from ouroboros.tools import media, vision
    from tests.test_media_tools import _patch_pypdf, _FakePage

    home = tmp_path / 'home'
    repo, data = home / 'Ouroboros/repo', home / 'Ouroboros/data'
    repo.mkdir(parents=True)
    source = data / 'projects/p1/knowledge/source.py'
    source.parent.mkdir(parents=True)
    source.write_text('def parent_project_source():\n    pass\n', encoding='utf-8')
    monkeypatch.setattr(config, 'DATA_DIR', data)
    monkeypatch.setenv('OUROBOROS_USER_FILES_ROOT', str(home))
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', mode)
    monkeypatch.setenv('OUROBOROS_SAFETY_MODE', 'off')
    registry = ToolRegistry(repo, data)
    ctx = registry._ctx
    if actor != 'parent':
        ctx.task_constraint = TaskConstraint(mode=actor, surface='external_workspace', write_root=str(repo))
    if fallback:
        monkeypatch.setattr('ouroboros.code_search_rg._rg_binary', lambda: '')
    else:
        from tests.test_code_search_rg import _install_fake_rg
        _install_fake_rg(tmp_path, monkeypatch)
    for tool, args, expected in (
        ('read_file', {'path': str(source)}, 'parent_project_source'),
        ('list_files', {'path': str(source.parent)}, 'source.py'),
        ('search_code', {'path': str(source.parent), 'query': 'parent_project_source'}, 'parent_project_source'),
        ('query_code', {'op': 'symbols', 'path': str(source.parent)}, 'parent_project_source'),
        ('query_code', {'op': 'digest', 'path': str(source.parent)}, 'parent_project_source'),
    ):
        result = registry.execute(tool, {'root': 'user_files', **args})
        assert (expected in result) is (mode == 'cyber_pro'), result
        if mode != 'cyber_pro':
            assert 'BLOCKED' in result or 'blocked' in result, result
    # The explicitly selected runtime root and the implicit media reader keep
    # their pre-existing project-store policy even in Cyber, for every actor.
    runtime = registry.execute('read_file', {'root': 'runtime_data', 'path': str(source)})
    assert 'ACCESS_DENIED' in runtime and 'parent_project_source' not in runtime
    assert 'ACCESS_DENIED' in vision._read_file_parity_block(ctx, source)
    pdf = source.parent / 'note.pdf'
    pdf.write_bytes(b'%PDF-1.4 fake')
    _patch_pypdf(monkeypatch, [_FakePage('parent_project_source')])
    assert 'ACCESS_DENIED' in media._ocr_pdf(ctx, str(pdf))


def test_query_code_user_files_child_matches_parent_read_scope(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.query_code import _query_code
    from ouroboros.tools.registry import ToolContext

    home = tmp_path / "home"
    source = home / "project"
    source.mkdir(parents=True)
    (source / "source.py").write_text("def parent_visible():\n    pass\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.py").write_text("def outside_source():\n    pass\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    for constraint in (None, TaskConstraint(mode="local_readonly_subagent")):
        ctx.task_constraint = constraint
        for op in ("symbols", "digest"):
            result = _query_code(ctx, op, root="user_files", path=str(source))
            assert "parent_visible" in result
        refused = _query_code(ctx, "symbols", root="user_files", path=str(outside))
        assert "outside the user_files home" in refused and "outside_source" not in refused
