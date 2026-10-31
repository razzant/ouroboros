import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
capture = runpy.run_path(str(ROOT / 'scripts/capture_linux_sources.py'))


def source_fixture(directory, package='webkit2gtk', version='2.50.4-0ubuntu0.22.04.1'):
    data = b'fixture source including Debian patches'
    source = directory / 'source.tar.xz'
    source.write_bytes(data)
    (directory / 'source.dsc').write_text(
        f'Format: 3.0 (quilt)\nSource: {package}\nVersion: {version}\nChecksums-Sha256:\n'
        f' {hashlib.sha256(data).hexdigest()} {len(data)} {source.name}\nFiles:\n')
    return source


def test_capture_downloads_exact_source_versions_and_exports_transformation_and_hashes(tmp_path, monkeypatch):
    root = tmp_path / 'linux-gtk'
    (root / 'share/doc').mkdir(parents=True)
    (root / 'share/common-licenses').mkdir()
    (root / 'share/doc/webkit-relocation.txt').write_text('dated fixture change notice')
    manifest = {'source_packages': {'webkit2gtk': '2.50.4-0ubuntu0.22.04.1'}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[0] == 'apt-get':
            source_fixture(kwargs['cwd'])
        else:
            kwargs['stdout'].write('fixture\t1\n')

    monkeypatch.setattr(capture['subprocess'], 'run', run)
    archive = tmp_path / 'sources.tar.gz'
    capture['capture'](root, archive)
    assert calls[0] == ['apt-get', 'source', '--download-only', '--only-source', 'webkit2gtk=2.50.4-0ubuntu0.22.04.1']
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert 'linux-native-sources/debian/webkit2gtk/source.dsc' in names
        assert 'linux-native-sources/build-inputs/scripts/bundle_linux_gtk.py' in names
        assert 'linux-native-sources/share/doc/webkit-relocation.txt' in names
        assert json.load(tar.extractfile('linux-native-sources/gtk-manifest.json')) == manifest
    assert archive.with_suffix('.gz.sha256').read_text().split()[0] == capture['sha256'](archive)
    stage = tmp_path / 'linux-native-sources'
    for line in (stage / 'SHA256SUMS').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        assert capture['sha256'](stage / relative) == digest
    with pytest.raises(FileExistsError):
        capture['capture'](root, archive)


def test_source_capture_refuses_wrong_version_missing_and_modified_files(tmp_path):
    check = capture['verify_source']
    with pytest.raises(RuntimeError, match='one source descriptor'):
        check(tmp_path, 'webkit2gtk', '1')
    source = source_fixture(tmp_path, version='1')
    with pytest.raises(RuntimeError, match='identity mismatch'):
        check(tmp_path, 'webkit2gtk', '2')
    check(tmp_path, 'webkit2gtk', '1')
    source.write_bytes(b'modified')
    with pytest.raises(RuntimeError, match='mismatch'):
        check(tmp_path, 'webkit2gtk', '1')
    source.unlink()
    with pytest.raises(FileNotFoundError):
        check(tmp_path, 'webkit2gtk', '1')


@pytest.mark.parametrize('versions', [('2', '2', '2'), ('2', '1', '3')])
def test_source_package_collection_preserves_all_observed_versions(monkeypatch, versions):
    bundle = runpy.run_path(str(ROOT / 'scripts/bundle_linux_gtk.py'))
    packages = dict(zip(('libmount1', 'libblkid1', 'libuuid1'), versions))
    calls = []

    def output(*args):
        calls.append(args[-1])
        assert args[:-1] == ('dpkg-query', '-W', '-f=${source:Package}\t${source:Version}')
        return 'util-linux\t' + packages[args[-1]]

    collect = bundle['collect_source_packages']
    monkeypatch.setitem(collect.__globals__, 'output', output)
    if len(set(versions)) == 1:
        assert collect(packages) == {'util-linux': '2'}
    else:
        with pytest.raises(RuntimeError, match='Conflicting source package versions') as error:
            collect(packages)
        assert json.loads(str(error.value).split(': ', 1)[1]) == {'util-linux': ['1', '2', '3']}
    assert calls == list(packages)


def test_production_and_diagnostic_use_capture_and_ci_retains_partial_failure_inputs():
    assert 'scripts/capture_linux_sources.py' in (ROOT / 'build_linux.sh').read_text()
    assert 'scripts/capture_linux_sources.py' in (ROOT / 'scripts/build_linux_native_experiment.sh').read_text()
    ci = (ROOT / '.github/workflows/ci.yml').read_text()
    step = ci.split('- name: Retain Linux native corresponding-source inputs')[1].split('- name:')[0]
    assert "if: always() && matrix.os == 'ubuntu-latest'" in step
    assert 'dist/linux-native-sources/' in step and 'dist/native-sources-linux-*.tar.gz' in step
    assert "OUROBOROS_BUNDLE_LINUX_GTK: '1'" in ci
    inputs = ci.split('- name: Install Linux GTK closure build inputs (Ubuntu 22.04)')[1].split('- name:')[0]
    assert inputs.index('sudo apt-get update') < inputs.index(
        'sudo apt-get -y --no-install-recommends dist-upgrade') < inputs.index('sudo apt-get install')


def test_source_companion_does_not_change_release_archive_selection(tmp_path):
    # Exercise the existing release consumer; its binary rules are unchanged.
    from scripts import release_proof

    binary = tmp_path / 'Ouroboros-7.7.1-linux-x86_64.tar.gz'
    binary.write_bytes(b'binary fixture')
    before = release_proof._release_assets(tmp_path, release_proof.RELEASE_ASSET_SUFFIXES)
    (tmp_path / 'native-sources-linux-7.7.1.tar.gz').write_bytes(b'source fixture')
    assert release_proof.locate_artifact(tmp_path) == binary
    assert release_proof._release_assets(tmp_path, release_proof.RELEASE_ASSET_SUFFIXES) == before
    assert 'dist/native-sources-linux-${VERSION}.tar.gz' in (ROOT / 'build_linux.sh').read_text()


@pytest.mark.skipif(os.name == 'nt', reason='Linux build scripts require POSIX shell execution')
@pytest.mark.parametrize('arch,mode,expect_native', [('aarch64', 'auto', False), ('x86_64', '0', False),
                                                   ('aarch64', '1', True)])
def test_non_target_build_keeps_host_backend_path_before_any_build(tmp_path, arch, mode, expect_native):
    (tmp_path / 'VERSION').write_text('0.0.0')
    fake = tmp_path / 'bin'
    fake.mkdir()
    for name, code in {'uname': f'echo {arch}', 'uv': 'exit 0', 'pkg-config': 'exit 1'}.items():
        script = fake / name
        script.write_text('#!/bin/sh\n' + code + '\n')
        script.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT / 'build_linux.sh')], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, 'PATH': str(fake) + os.pathsep + os.environ['PATH'],
             'OUROBOROS_BUNDLE_LINUX_GTK': mode})
    assert result.returncode == 1
    if expect_native:
        assert 'GTK/WebKit build inputs are missing' in result.stderr
    else:
        assert 'Host-backend Linux build' in result.stdout and 'python-standalone/ not found' in result.stdout


@pytest.mark.skipif(os.name == 'nt', reason='AppImage packaging is POSIX-only')
def test_explicit_legacy_appdir_packaging_is_preserved(tmp_path):
    dist = tmp_path / 'dist'
    payload = dist / 'Ouroboros'
    payload.mkdir(parents=True)
    (payload / 'Ouroboros').write_text('#!/bin/sh\nexit 0\n')
    (payload / 'Ouroboros').chmod(0o755)
    result = subprocess.run(['bash', 'scripts/build_appimage.sh', '--appdir-only'], cwd=ROOT, capture_output=True,
        text=True, env={**os.environ, 'OUROBOROS_DIST_DIR': str(dist), 'OUROBOROS_BUNDLE_LINUX_GTK': '0'})
    assert result.returncode == 0, result.stderr
    assert (dist / 'Ouroboros.AppDir/usr/lib/ouroboros/Ouroboros').is_file()
