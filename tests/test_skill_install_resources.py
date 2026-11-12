"""Declared resources/builds through the real isolated dependency owner."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import shutil
import sys
import tarfile
import threading
from types import SimpleNamespace

import pytest
import yaml

from ouroboros.marketplace import isolated_deps as deps
from ouroboros.marketplace.install_specs import install_specs_hash, normalize_install_specs
from ouroboros.skill_dependencies import skill_deps_not_ready
from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state
from tests.test_skill_runtime_commands import execute, installed_script



@contextmanager
def serve(routes):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            content = routes.get(self.path)
            if callable(content):
                content = content(f'http://127.0.0.1:{self.server.server_port}')
            if content is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html' if self.path.startswith('/simple/') else 'application/octet-stream')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def archive(files, prefix):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w:gz') as packed:
        for name, body in files.items():
            raw = body.encode() if isinstance(body, str) else body
            info = tarfile.TarInfo(f'{prefix}/{name}')
            info.size = len(raw)
            packed.addfile(info, io.BytesIO(raw))
    return out.getvalue()


def normalized(raw):
    auto, manual, warnings = normalize_install_specs(raw)
    assert not manual and not warnings, (manual, warnings)
    return auto


def resource(url, data, **extra):
    return {'kind': 'download', 'url': url, 'sha256': hashlib.sha256(data).hexdigest(),
            'size_bytes': len(data), 'target': 'resources/input.bin', **extra}


def declare_dependencies(ctx, payload, raw):
    path = payload / 'SKILL.md'
    manifest = yaml.safe_load(path.read_text().split('---')[1])
    manifest['install_specs'] = [raw]
    path.write_text('---\n' + yaml.safe_dump(manifest) + '---\nFixture\n')
    save_review_state(ctx.drive_root, 'runtime_fixture',
                      SkillReviewState(status='pass', content_hash=compute_content_hash(payload)))


def reviewed_resource(drive, name, raw):
    payload = drive / 'skills/external' / name
    payload.mkdir(parents=True, exist_ok=True)
    manifest = {'name': name, 'description': 'Install fixture', 'version': '1', 'type': 'instruction',
                'install_specs': [raw]}
    (payload / 'SKILL.md').write_text('---\n' + yaml.safe_dump(manifest) + '---\nFixture\n')
    save_review_state(drive, name, SkillReviewState(status='pass', content_hash=compute_content_hash(payload)))
    return payload


def test_large_resource_cache_survives_env_replacement_and_digest_change(tmp_path):
    first = b'\x7fELF' + b'a' * (9 * 1024 * 1024)
    second = b'\x7fELF' + b'b' * (9 * 1024 * 1024)
    with serve({'/first': first, '/second': second}) as (url, requests):
        raw = resource(url + '/first', first, version='image-one')
        skill = reviewed_resource(tmp_path, 'fixture', raw)
        original_hash = compute_content_hash(skill)
        specs = normalized(raw)
        installed = deps.install_isolated_dependencies(tmp_path, 'fixture', skill, specs)
        assert installed['installed'][0]['downloaded'] and installed['installed'][0]['installed']
        assert installed['installed'][0]['sha256'] == raw['sha256']
        assert installed['executable_ready'] is None
        assert (skill / '.ouroboros_env/resources/input.bin').read_bytes() == first
        cache = tmp_path / 'state/skills/fixture/dependency_cache/resources' / raw['sha256']
        assert cache.is_file()
        shutil.rmtree(skill / '.ouroboros_env')
        deps.install_isolated_dependencies(tmp_path, 'fixture', skill, specs)
        assert requests == ['/first'], requests
        assert compute_content_hash(skill) == original_hash  # large bytes do not become payload/Git content
        changed = resource(url + '/second', second, version='image-two')
        loaded = SimpleNamespace(name='fixture', skill_dir=skill, source='external',
                                 manifest=SimpleNamespace(raw_extra={'install_specs': [changed]}))
        assert skill_deps_not_ready(tmp_path, loaded)[1] == 'fingerprint'
        reviewed_resource(tmp_path, 'fixture', changed)
        installed = deps.install_isolated_dependencies(tmp_path, 'fixture', skill, normalized(changed))
        assert installed['specs_hash'] != install_specs_hash(specs)
        assert requests == ['/first', '/second']
        assert (skill / '.ouroboros_env/resources/input.bin').read_bytes() == second
        assert cache.is_file()
        (skill / '.ouroboros_env/resources/input.bin').unlink()
        assert deps.read_deps_state(tmp_path, 'fixture', skill)['status'] == 'stale'


def test_declared_native_source_build_runs_and_checks_actual_output(tmp_path):
    compiler = shutil.which('cc')
    if not compiler or sys.platform == 'win32':
        pytest.skip('fixture uses the POSIX C compiler; portable argv machinery has unit coverage')
    source = b'#include <stdio.h>\nint main(void){puts("native feature works");return 0;}\n'
    with serve({'/native.c': source}) as (url, _):
        raw = resource(url + '/native.c', source, target='resources/native.c',
                       steps=[{'argv': [sys.executable, '-c', "from pathlib import Path;Path('bin').mkdir()"]},
                              {'argv': [compiler, 'resources/native.c', '-o', 'bin/native']}],
                       outputs=['bin/native'], bins=['native'], check={'argv': ['native']})
        skill = reviewed_resource(tmp_path, 'native', raw)
        result = deps.install_isolated_dependencies(tmp_path, 'native', skill, normalized(raw))
    assert result['status'] == 'installed' and result['executable_ready'] is True
    assert result['installed'][0]['outputs'][0]['sha256']
    assert [row['kind'] for row in result['logs']] == ['build', 'build', 'readiness_check']


def test_failed_check_records_installed_but_not_executable_ready(tmp_path):
    # Real command/check path, with a downloaded inert file instead of a package-manager mock.
    data = b'payload'
    with serve({'/file': data}) as (url, _):
        raw = resource(url + '/file', data, check={'argv': [sys.executable, '-c', "import sys;print('feature missing',file=sys.stderr);sys.exit(3)"]})
        skill = reviewed_resource(tmp_path, 'failure', raw)
        with pytest.raises(RuntimeError, match='feature missing'):
            deps.install_isolated_dependencies(tmp_path, 'failure', skill, normalized(raw))
    state = deps.read_deps_state(tmp_path, 'failure')
    assert state['status'] == 'failed' and state['executable_ready'] is False
    assert state['installed'][0]['downloaded'] and state['installed'][0]['installed']
    assert state['logs'][-1]['returncode'] == 3


def test_platform_applicability_and_malformed_descriptors_are_explicit(tmp_path):
    valid = resource('http://example.invalid/file', b'bytes', platforms=['unmatched-os'])
    skill = reviewed_resource(tmp_path, 'platform', valid)
    result = deps.install_isolated_dependencies(tmp_path, 'platform', skill, normalized(valid))
    assert result['installed'][0]['status'] == 'not_applicable'
    assert result['installed'][0]['installed'] is False
    for delta in ({'sha256': ''}, {'size_bytes': True}, {'target': '../escape'},
                  {'steps': ['not argv']}, {'steps': [{'argv': ['true']}]}, {'platforms': 'linux'}):
        auto, manual, warnings = normalize_install_specs({**valid, **delta})
        assert not auto and manual and warnings


def _test_registry_env(monkeypatch, url):
    original = deps._installer_env
    def controlled(env_root, **kwargs):
        env = original(env_root, **kwargs)
        env.update(PIP_INDEX_URL=url + '/simple', PIP_TRUSTED_HOST='127.0.0.1',
                   npm_config_registry=url, npm_config_audit='false', npm_config_fund='false')
        return env
    monkeypatch.setattr(deps, '_installer_env', controlled)


def test_source_only_python_package_requires_opt_in_and_executes_real_feature(tmp_path, monkeypatch):
    backend = '''import pathlib,zipfile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name='fixture_source-1.0.0-py3-none-any.whl'
    with zipfile.ZipFile(pathlib.Path(wheel_directory)/name,'w') as wheel:
        wheel.writestr('fixture_source.py','def feature(value): return value * 3 + 1\\n')
        wheel.writestr('fixture_source-1.0.0.dist-info/METADATA','Metadata-Version: 2.1\\nName: fixture-source\\nVersion: 1.0.0\\n')
        wheel.writestr('fixture_source-1.0.0.dist-info/WHEEL','Wheel-Version: 1.0\\nGenerator: fixture\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n')
        wheel.writestr('fixture_source-1.0.0.dist-info/RECORD','')
    return name
'''
    source = archive({'backend.py': backend, 'pyproject.toml': '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n'}, 'fixture_source-1.0.0')
    routes = {'/fixture_source-1.0.0.tar.gz': source,
              '/simple/fixture-source/': lambda url: f'<a href="{url}/fixture_source-1.0.0.tar.gz#sha256={hashlib.sha256(source).hexdigest()}">fixture_source-1.0.0.tar.gz</a>'.encode()}
    with serve(routes) as (url, _):
        _test_registry_env(monkeypatch, url)
        ctx, payload, script = installed_script(tmp_path, monkeypatch, body='from fixture_source import feature\nprint(feature(7))')
        with pytest.raises(RuntimeError, match='pip install failed'):
            deps.install_isolated_dependencies(ctx.drive_root, 'runtime_fixture', payload,
                                              normalized({'kind': 'pip', 'package': 'fixture-source==1.0.0'}))
        spec = {'kind': 'pip', 'package': 'fixture-source==1.0.0', 'allow_source_build': True,
                'check': {'argv': ['python', '-c', 'from fixture_source import feature; assert feature(7)==22']}}
        declare_dependencies(ctx, payload, spec)
        result = deps.install_isolated_dependencies(ctx.drive_root, 'runtime_fixture', payload, normalized(spec))
        assert result['executable_ready'] is True
        assert any(row['name'] == 'fixture-source' and row['version'] == '1.0.0' for row in result['resolved_packages'])
        assert json.loads(execute(ctx, script))['stdout'].strip() == '22'


def test_npm_postinstall_opt_in_produces_a_real_feature(tmp_path, monkeypatch):
    if not shutil.which('npm') or not shutil.which('node'):
        pytest.skip('npm/node are not installed')
    name = 'fixture-postinstall'
    package = {'name': name, 'version': '1.0.0', 'main': 'generated.js', 'scripts': {'postinstall': 'node build.js'}}
    tarball = archive({'package.json': json.dumps(package),
                       'build.js': "require('fs').writeFileSync('generated.js', 'module.exports = x => x * 5;');"}, 'package')
    def metadata(url):
        return json.dumps({'name': name, 'dist-tags': {'latest': '1.0.0'}, 'versions': {'1.0.0': {
            **package, 'dist': {'tarball': url + '/fixture.tgz',
                               'integrity': 'sha512-' + base64.b64encode(hashlib.sha512(tarball).digest()).decode()}}}}).encode()
    with serve({'/' + name: metadata, '/fixture.tgz': tarball}) as (url, _):
        _test_registry_env(monkeypatch, url)
        ctx, payload, script = installed_script(tmp_path, monkeypatch, runtime='node', body="console.log(require('fixture-postinstall')(4));")
        spec = {'kind': 'npm', 'package': name + '@1.0.0',
                'check': {'argv': ['node', '-e', "if(require('fixture-postinstall')(4)!==20)process.exit(2)"]}}
        declare_dependencies(ctx, payload, spec)
        with pytest.raises(RuntimeError, match='readiness check failed'):
            deps.install_isolated_dependencies(ctx.drive_root, 'runtime_fixture', payload, normalized(spec))
        assert not (payload / '.ouroboros_env/node/node_modules/fixture-postinstall/generated.js').exists()
        # New install with explicit build authority (old failed environment is
        # kept in this test until the approved retry proves the feature works).
        spec['allow_install_scripts'] = True
        declare_dependencies(ctx, payload, spec)
        result = deps.install_isolated_dependencies(ctx.drive_root, 'runtime_fixture', payload, normalized(spec))
        assert result['executable_ready'] is True
        assert any(row['kind'] == 'npm' and row.get('version') == '1.0.0' for row in result['resolved_packages'])
        assert json.loads(execute(ctx, script))['stdout'].strip() == '20'


def test_new_install_declarations_refuse_stale_review_and_substituted_specs(tmp_path, monkeypatch):
    raw = resource('http://example.invalid/never-fetch', b'payload')
    skill = reviewed_resource(tmp_path, 'binding', raw)
    monkeypatch.setattr(deps, 'fetch_exact_file', lambda **_: pytest.fail('unreviewed declaration must not fetch'))
    with pytest.raises(RuntimeError, match='differ from the hash-covered'):
        deps.install_isolated_dependencies(tmp_path, 'binding', skill, normalized({**raw, 'target': 'resources/other'}))
    (skill / 'notes.md').write_text('changed after review')
    with pytest.raises(RuntimeError, match='fresh executable skill review'):
        deps.install_isolated_dependencies(tmp_path, 'binding', skill, normalized(raw))


@pytest.mark.parametrize('with_check', [False, True])
def test_payload_change_during_download_cannot_run_old_build_steps(tmp_path, monkeypatch, with_check):
    data = b'payload'
    with serve({'/file': data}) as (url, _):
        raw = resource(url + '/file', data)
        if with_check:
            raw['check'] = {'argv': [sys.executable, '-c', "raise SystemExit('must not run')"]}
        skill = reviewed_resource(tmp_path, 'drift', raw)
        original = deps.fetch_exact_file
        def fetch_then_edit(**kwargs):
            path = original(**kwargs)
            (skill / 'notes.md').write_text('changed while downloading')
            return path
        monkeypatch.setattr(deps, 'fetch_exact_file', fetch_then_edit)
        monkeypatch.setattr(deps, '_run', lambda *a, **kw: pytest.fail('no process after payload drift'))
        with pytest.raises(RuntimeError, match='fresh executable skill review'):
            deps.install_isolated_dependencies(tmp_path, 'drift', skill, normalized(raw))
    assert deps.read_deps_state(tmp_path, 'drift')['status'] == 'failed'
