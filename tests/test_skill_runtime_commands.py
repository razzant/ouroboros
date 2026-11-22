"""Real interpreter/toolchain contracts after normal skill review and grants."""
from __future__ import annotations

import json
import shutil
import sys

import pytest
import yaml

from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state
from ouroboros.tools import skill_exec
from ouroboros.tools.registry import ToolContext
from tests._cases import chosen_rows


PROGRAMS = {
    'python': ('py', "import sys\nfrom pathlib import Path\nprint(Path('cwd-proof.txt').read_text(encoding='utf-8'))\nprint(*sys.argv[1:],sep='\\n')\nprint('diagnostic',file=sys.stderr)\nsys.exit(7)\n"),
    'python3': ('py', "import sys\nfrom pathlib import Path\nprint(Path('cwd-proof.txt').read_text(encoding='utf-8'))\nprint(*sys.argv[1:],sep='\\n')\nprint('diagnostic',file=sys.stderr)\nsys.exit(7)\n"),
    'node': ('js', "console.log(require('node:fs').readFileSync('cwd-proof.txt','utf8')); for(const x of process.argv.slice(2))console.log(x); console.error('diagnostic'); process.exit(7);"),
    'deno': ('ts', "console.log(Deno.readTextFileSync('cwd-proof.txt')); for(const x of Deno.args)console.log(x); console.error('diagnostic'); Deno.exit(7);"),
    'go': ('go', 'package main\nimport("fmt";"os")\nfunc main(){cwd,_:=os.ReadFile("cwd-proof.txt");fmt.Println(string(cwd));for _,a:=range os.Args[1:]{fmt.Println(a)};fmt.Fprintln(os.Stderr,"diagnostic");os.Exit(7)}\n'),
    'bash': ('sh', 'printf "%s\\n" "$(cat cwd-proof.txt)" "$@"\nprintf "diagnostic\\n" >&2\nexit 7\n'),
    'ruby': ('rb', 'puts File.read("cwd-proof.txt", encoding: "UTF-8"); puts ARGV; STDERR.puts "diagnostic"; exit 7\n'),
}


def installed_script(tmp_path, monkeypatch, *, runtime='python3', body="print('ok')", permissions=(), env_keys=(), timeout=60):
    repo = tmp_path / 'repo'
    repo.mkdir(exist_ok=True)
    data = tmp_path / 'drive, with space'
    payload = data / 'skills/external/runtime_fixture'
    (payload / 'scripts').mkdir(parents=True, exist_ok=True)
    # A unique relative file proves cwd without comparing each runtime's path spelling.
    (payload / 'cwd-proof.txt').write_text(f'cwd marker {payload}', encoding='utf-8')
    suffix = PROGRAMS[runtime][0]
    script = payload / f'scripts/main.{suffix}'
    script.write_text(body)
    manifest = {'name': 'runtime_fixture', 'description': 'Runtime fixture', 'version': '1', 'type': 'script',
                'runtime': runtime, 'timeout_sec': timeout, 'permissions': list(permissions),
                'env_from_settings': list(env_keys), 'scripts': [{'name': f'main.{suffix}'}]}
    if runtime == 'go':
        manifest['env_from_settings'].append('GOMAXPROCS')
        monkeypatch.setattr(skill_exec, 'load_settings', lambda: {'GOMAXPROCS': '4'})
    (payload / 'SKILL.md').write_text('---\n' + yaml.safe_dump(manifest) + '---\nFixture\n')
    save_review_state(data, 'runtime_fixture', SkillReviewState(status='pass', content_hash=compute_content_hash(payload)))
    save_enabled(data, 'runtime_fixture', True)
    monkeypatch.setenv('OUROBOROS_SKILLS_REPO_PATH', '')
    return ToolContext(repo_dir=repo, drive_root=data), payload, script


def execute(ctx, script, args=()):
    return skill_exec._handle_skill_exec(ctx, skill='runtime_fixture', script=f'scripts/{script.name}', args=list(args))


@pytest.mark.parametrize('runtime', chosen_rows(list(PROGRAMS), 0, 4, 6))
def test_advertised_runtime_preserves_args_cwd_stderr_and_exit(tmp_path, monkeypatch, runtime):
    if not skill_exec._resolve_runtime_binary(runtime)[0]:
        pytest.skip(f'{runtime} toolchain is not installed')
    monkeypatch.setenv('GOMAXPROCS', '4')
    ctx, payload, script = installed_script(tmp_path, monkeypatch, runtime=runtime, body=PROGRAMS[runtime][1])
    args = ['hello world', ';$(literal)', 'other.go', '--literal-flag']
    result = execute(ctx, script, args)
    assert result.startswith('⚠️ SKILL_EXEC_FAILED'), result
    facts = json.loads(result[result.index('{'):])
    assert facts['exit_code'] == 7 and facts['runtime_phase'] == 'execute', facts
    assert facts['stdout'].splitlines() == [(payload / 'cwd-proof.txt').read_text(encoding='utf-8'), *args], facts
    assert facts['stderr'].strip() == 'diagnostic'
    assert not list((ctx.drive_root / 'state/skills/runtime_fixture').glob('go-exec-*'))


def test_go_compile_failure_is_separate_from_program_exit(tmp_path, monkeypatch):
    if not shutil.which('go'):
        pytest.skip('go toolchain is not installed')
    ctx, _, script = installed_script(tmp_path, monkeypatch, runtime='go', body='package main\nfunc main(\n')
    result = execute(ctx, script)
    assert result.startswith('⚠️ SKILL_EXEC_FAILED'), result
    facts = json.loads(result[result.index('{'):])
    assert facts['runtime_phase'] == 'compile' and facts['exit_code'] != 0
    assert 'syntax error' in facts['stderr']
    assert not list((ctx.drive_root / 'state/skills/runtime_fixture').glob('go-exec-*'))


def test_deno_reads_external_file_without_fs_and_writes_existing_state(tmp_path, monkeypatch):
    if not shutil.which('deno'):
        pytest.skip('deno runtime is not installed')
    outside = tmp_path / 'selected external.txt'
    outside.write_text('owner selected input')
    # An ordinary setting the skill declared (a custom key would be a grant-bound custom secret).
    monkeypatch.setattr(skill_exec, 'load_settings', lambda: {'OUROBOROS_UI_LANGUAGE': 'visible'})
    body = '''const text = await Deno.readTextFile(Deno.args[0]);
const state = Deno.env.get('OUROBOROS_SKILL_STATE_DIR');
await Deno.writeTextFile(state + '/result.txt', text + ':' + Deno.env.get('OUROBOROS_UI_LANGUAGE'));
console.log(text);
'''
    ctx, _, script = installed_script(tmp_path, monkeypatch, runtime='deno', body=body, env_keys=['OUROBOROS_UI_LANGUAGE'])
    facts = json.loads(execute(ctx, script, [str(outside)]))
    assert facts['exit_code'] == 0
    assert (ctx.drive_root / 'state/skills/runtime_fixture/result.txt').read_text() == 'owner selected input:visible'
    assert 'NotCapable' not in facts['stderr']


@pytest.mark.parametrize('permission,allowed', [('fs', False), ('fs', True), ('subprocess', False), ('subprocess', True)])
def test_deno_effects_use_reviewed_declarations(tmp_path, monkeypatch, permission, allowed):
    if not shutil.which('deno'):
        pytest.skip('deno runtime is not installed')
    output = tmp_path / 'selected output.txt'
    body = ("await Deno.writeTextFile(Deno.args[0], 'written');" if permission == 'fs' else
            "const r = await new Deno.Command(Deno.args[0], {args:['-c', \"print('child')\"]}).output(); console.log(new TextDecoder().decode(r.stdout).trim());")
    ctx, _, script = installed_script(tmp_path, monkeypatch, runtime='deno', body=body, permissions=[permission] if allowed else [])
    result = execute(ctx, script, [str(output) if permission == 'fs' else sys.executable])
    if allowed:
        facts = json.loads(result)
        assert facts['exit_code'] == 0, result
        if permission == 'fs':
            assert output.read_text() == 'written'
        else:
            assert facts['stdout'].strip() == 'child'
    else:
        assert 'SKILL_EXEC_FAILED' in result and 'NotCapable' in result
        assert not output.exists()


def test_missing_runtime_reports_unavailable_without_failing_skill_validation(tmp_path, monkeypatch):
    ctx, _, script = installed_script(tmp_path, monkeypatch, runtime='go', body=PROGRAMS['go'][1])
    monkeypatch.setattr(skill_exec, '_resolve_runtime_binary', lambda _: (None, ''))
    result = execute(ctx, script)
    assert result.startswith('⚠️ SKILL_EXEC_ERROR') and 'no available binary on PATH' in result
    assert 'not in the allowlist' not in result and 'SKILL_EXEC_FAILED' not in result
