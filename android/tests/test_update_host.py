"""Native upgrade sequencing; Android PackageManager effects are simulated here."""
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
from types import SimpleNamespace

import pytest

pytest.importorskip('fcntl', reason='The native installer executes inside Linux')
_SOURCE = Path(__file__).parents[1] / 'bootstrap' / 'update-host'
_loader = SourceFileLoader('ouroboros_android_update_host', str(_SOURCE))
_spec = spec_from_loader(_loader.name, _loader)
host = module_from_spec(_spec)
_loader.exec_module(host)


@pytest.fixture
def device(tmp_path, monkeypatch):
    app = tmp_path / 'app'
    repo, sdk, java = app / 'repo', app / 'android-sdk', app / 'jdk'
    (repo / 'android' / 'host').mkdir(parents=True)
    (repo / 'assets').mkdir()
    source_repo = _SOURCE.parents[2]
    shutil.copytree(source_repo / 'android/provision', repo / 'android/provision',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (repo / 'ouroboros').mkdir()
    shutil.copy2(source_repo / 'ouroboros/claudexor_runtime_pin.json', repo / 'ouroboros/claudexor_runtime_pin.json')
    shutil.copy2(source_repo / 'requirements-runtime.lock', repo / 'requirements-runtime.lock')
    (repo / 'android' / 'host' / 'build.py').write_text('# compiler')
    (repo / 'android' / 'host' / 'Main.java').write_text('original native body')
    (repo / 'assets' / 'icon_1024.png').write_bytes(b'icon')
    (repo / 'VERSION').write_text('7.0.0')
    (repo / 'android' / 'bootstrap').mkdir()
    for name in ('android-exec', 'android-call', 'build-apk', 'update-host', 'enter-linux', 'core-control'):
        (repo / 'android' / 'bootstrap' / name).write_text('# script ' + name)
    sdk.mkdir()
    (sdk / 'installation.json').write_text(json.dumps({
        'build_tools_version': '36.0.0', 'platform': 'android-36', 'java_home': str(java),
        'artifacts': [{'name': 'pinned', 'sha256': 'abc'}]}))
    (app / 'signing').mkdir()
    (app / 'signing' / 'host.keystore').write_bytes(b'private fixture')
    (app / 'signing' / 'host-password').write_text('fixture password')
    facts = {'installed': b'old APK', 'version': 4, 'builds': [], 'installs': [], 'starts': [], 'scripts': {}}
    certificate = b'public fixture certificate'
    digest = hashlib.sha256(certificate).hexdigest()
    args = SimpleNamespace(app_root=app, repo=None, sdk=None, java_home=None, check=False)

    def run(argv, binary=False, input=None):
        argv = [str(value) for value in argv]
        if 'cat' in argv:
            return facts['scripts'].get(argv[-1], b'')
        if '/system/bin/sh' in argv:
            facts['scripts'][argv[-1]] = input
            return b''
        if '-exportcert' in argv:
            assert '-storepass:file' in argv
            return certificate
        if 'path' in argv:
            return 'package:/data/app/test/base.apk\n' if facts['installed'] else ''
        if 'dumpsys' in argv:
            return f'versionCode={facts["version"]} minSdk=26 targetSdk=36\n'
        if str(repo / 'android' / 'host' / 'build.py') in argv:
            facts['builds'].append(argv)
            output = Path(argv[argv.index('--out') + 1])
            version = int(argv[argv.index('--version-code') + 1])
            body = (repo / 'android' / 'host' / 'Main.java').read_bytes()
            (output / 'Ouroboros.apk').write_bytes(str(version).encode() + b'|' + body)
            assert argv[argv.index('--key-alias') + 1] == 'ouroboros-host'
            assert '--keystore-pass-file' in argv
            return str(output / 'Ouroboros.apk')
        if 'start-foreground-service' in argv:
            facts['starts'].append(argv)
            assert argv[-1] == 'status'
            return 'Starting service'
        if argv[0] == 'git':
            return 'a' * 40
        raise AssertionError(argv)

    def install(argv, *, stdin, **kwargs):
        assert argv[1:4] == ['pm', 'install', '-r']
        assert argv[-2:] == ['--', '-']
        body = stdin.read()
        facts['installs'].append(body)
        facts['installed'] = body
        facts['version'] = int(body.split(b'|', 1)[0])
        return subprocess.CompletedProcess(argv, 0, stdout='Success\n', stderr='')

    monkeypatch.setattr(host, 'platform_state', lambda repo, app: (True, {}))
    monkeypatch.setattr(host, 'run', run)
    monkeypatch.setattr(host, 'signer', lambda *a: digest)
    monkeypatch.setattr(host, 'copy_installed', lambda path, dest, tool: dest.write_bytes(facts['installed']))
    monkeypatch.setattr(host.subprocess, 'run', install)
    return args, facts


def test_native_change_install_readback_and_core_only_update(device):
    args, facts = device
    first = host.update(args)
    assert first['status'] == 'installed' and first['version_code'] == 5
    assert len(facts['installs']) == len(facts['starts']) == 1
    (args.app_root / 'repo' / 'core.py').write_text('new core implementation')
    second = host.update(args)
    assert second['status'] == 'current'
    assert len(facts['builds']) == len(facts['installs']) == 1
    assert len(facts['starts']) == 2


def test_release_version_change_updates_native_version_name(device):
    args, facts = device
    host.update(args)
    (args.app_root / 'repo' / 'VERSION').write_text('7.0.1')
    result = host.update(args)
    assert result['status'] == 'installed' and result['version_name'] == '7.0.1'
    assert result['version_code'] == 6 and len(facts['builds']) == 2


def test_bootstrap_source_change_adopts_without_rebuilding_apk(device):
    args, facts = device
    host.update(args)
    source = args.app_root / 'repo' / 'android' / 'bootstrap' / 'enter-linux'
    source.write_text('# new Linux entry')
    args.check = True
    assert host.update(args)['status'] == 'build_required'
    args.check = False
    assert host.update(args)['status'] == 'current'
    assert facts['scripts']['/data/local/ouroboros-phone/bin/enter-linux'] == b'# new Linux entry'
    assert len(facts['builds']) == len(facts['installs']) == 1


def test_restoring_older_native_source_uses_higher_install_number(device):
    args, facts = device
    original = args.app_root / 'repo' / 'android' / 'host' / 'Main.java'
    host.update(args)
    original.write_text('personal edit plus official update')
    host.update(args)
    original.write_text('original native body')
    restored = host.update(args)
    assert restored['version_code'] == 7
    assert facts['installed'] == b'7|original native body'
    assert len(list((args.app_root / 'data' / 'android-builds').iterdir())) == 3


def test_check_never_builds_installs_or_starts(device):
    args, facts = device
    args.check = True
    assert host.update(args)['status'] == 'build_required'
    assert not facts['builds'] and not facts['installs'] and not facts['starts']


@pytest.mark.parametrize('returncode,stdout,stderr,absent', [
    (1, '', '', True),
    (1, '', 'Error: package service unavailable', False),
    (1, 'permission denied', '', False),
    (127, '', '', False),
])
def test_package_absence_is_distinct_from_failed_lookup(monkeypatch, returncode, stdout, stderr, absent):
    error = subprocess.CalledProcessError(returncode, ['pm', 'path', host.PACKAGE], stdout, stderr)

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(host, 'run', run)
    if absent:
        assert host.installed_package('android-exec') is None
    else:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            host.installed_package('android-exec')
        assert caught.value is error


def test_missing_package_can_install_with_existing_personal_key(device, monkeypatch):
    args, facts = device
    facts['installed'] = None
    previous_run = host.run

    def run(argv, **kwargs):
        if 'path' in argv and not facts['installed']:
            raise subprocess.CalledProcessError(1, argv, '', '')
        return previous_run(argv, **kwargs)

    monkeypatch.setattr(host, 'run', run)
    result = host.update(args)
    assert result['status'] == 'installed' and result['version_code'] == 1
    assert len(facts['builds']) == len(facts['installs']) == 1


def test_missing_personal_key_never_generates_replacement(device):
    args, facts = device
    key = args.app_root / 'signing' / 'host.keystore'
    key.unlink()
    with pytest.raises(RuntimeError, match='Restore its backup'):
        host.update(args)
    assert not key.exists() and not facts['builds'] and not facts['installs']


def test_wrong_installed_signer_preserves_package(device, monkeypatch):
    args, facts = device
    monkeypatch.setattr(host, 'signer', lambda *a: 'wrong')
    with pytest.raises(RuntimeError, match='different signing key'):
        host.update(args)
    assert facts['installed'] == b'old APK' and not facts['builds']


def test_source_changed_during_build_is_not_installed(device, monkeypatch):
    args, facts = device
    original_run = host.run

    def changing_run(argv, **kwargs):
        result = original_run(argv, **kwargs)
        if any(str(value).endswith('/host/build.py') for value in argv):
            (args.app_root / 'repo' / 'android' / 'host' / 'Main.java').write_text('concurrent edit')
        return result

    monkeypatch.setattr(host, 'run', changing_run)
    with pytest.raises(RuntimeError, match='changed during compilation'):
        host.update(args)
    assert not facts['installs']


def test_unknown_install_failure_is_not_repeated_or_recorded_current(device, monkeypatch):
    args, facts = device

    def unknown_install(argv, **kwargs):
        facts['installs'].append('unknown')
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(host.subprocess, 'run', unknown_install)
    with pytest.raises(subprocess.CalledProcessError):
        host.update(args)
    assert facts['installs'] == ['unknown'] and not facts['starts']
    assert not (args.app_root / 'data' / 'state' / 'android_host.json').exists()


def test_android_call_restores_only_connection_before_request(tmp_path, monkeypatch):
    loader = SourceFileLoader('ouroboros_android_call_recovery', str(_SOURCE.with_name('android-call')))
    module = module_from_spec(spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    attempts, starts = [], []

    class Socket:
        def connect(self, address):
            attempts.append(address)
            if len(attempts) == 1:
                raise ConnectionRefusedError('host process was replaced')

    monkeypatch.setattr(module.subprocess, 'run', lambda argv, **kw: starts.append(argv))
    module.connect(Socket(), '@ai.ouroboros.android.rpc', True)
    assert len(attempts) == 2
    assert starts == [['android-exec', 'am', 'start-foreground-service', '-n',
                       'ai.ouroboros.android/.CoreService', '-a', 'status']]


def test_android_call_custom_socket_does_not_start_host(monkeypatch):
    loader = SourceFileLoader('ouroboros_android_call_no_recovery', str(_SOURCE.with_name('android-call')))
    module = module_from_spec(spec_from_loader(loader.name, loader))
    loader.exec_module(module)

    class Socket:
        def connect(self, address):
            raise ConnectionRefusedError('absent custom service')

    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: pytest.fail('native host started'))
    with pytest.raises(ConnectionRefusedError):
        module.connect(Socket(), '/custom/socket', True)


def test_failed_tool_keeps_captured_diagnostics_without_printing_argv(monkeypatch, capsys):
    def fail(_args):
        raise subprocess.CalledProcessError(255, ['tool', 'private-input-argument'],
                                            b'\npartial tool output\n', b'\nUnknown option -\n')

    monkeypatch.setattr(host, 'update', fail)
    monkeypatch.setattr(host.sys, 'argv', ['update-host', '--check'])
    assert host.main() == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {'status': 'failed', 'error': 'Tool process failed',
                                 'returncode': 255, 'stdout': 'partial tool output',
                                 'stderr': 'Unknown option -'}
    assert 'private-input-argument' not in output


def test_platform_pin_delta_prepares_before_native_build_with_same_key(device, monkeypatch):
    args, facts = device
    host.update(args)
    repo = args.app_root / 'repo'
    manifest = repo / 'android/provision/artifacts.json'
    pins = json.loads(manifest.read_text())
    next(pin for pin in pins if pin['name'] == 'android-platform')['sha256'] = 'e' * 64
    manifest.write_text(json.dumps(pins))
    desired_before = host.source_identity(repo, args.app_root / 'android-sdk')[0]
    key = (args.app_root / 'signing/host.keystore').read_bytes()
    calls, prepared = [], []
    old_run = host.run

    def run(argv, **kwargs):
        if '--ensure-platform' in argv:
            assert not prepared
            calls.append('prepare')
            prepared.append(True)
            return ''
        if any(str(value).endswith('/host/build.py') for value in argv):
            assert prepared
            calls.append('build')
        return old_run(argv, **kwargs)

    monkeypatch.setattr(host, 'platform_state', lambda *_: (bool(prepared), {'sdk': 'new desired input'}))
    monkeypatch.setattr(host, 'run', run)
    args.check = True
    assert host.update(args)['status'] == 'platform_update_required'
    assert not calls
    args.check = False
    result = host.update(args)
    assert result['status'] == 'installed' and result['version_code'] == 6
    assert result['input_sha256'] == desired_before
    assert calls == ['prepare', 'build']
    assert (args.app_root / 'signing/host.keystore').read_bytes() == key


def test_unsatisfied_platform_recipe_cannot_claim_native_success(device, monkeypatch):
    args, facts = device
    monkeypatch.setattr(host, 'platform_state', lambda *_: (False, {'sdk': 'new'}))
    old_run = host.run
    monkeypatch.setattr(host, 'run', lambda argv, **kw: '' if '--ensure-platform' in argv else old_run(argv, **kw))
    with pytest.raises(RuntimeError, match='does not match'):
        host.update(args)
    assert facts['installed'] == b'old APK' and not facts['builds']
