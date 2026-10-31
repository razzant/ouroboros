#!/usr/bin/env python3
"""Retain exact Debian sources and local transformations for one GTK closure.

Requires matching, authenticated deb-src indices on the builder. Missing or
superseded sources fail the build; partial downloads remain for diagnosis.
This is retention, not a license-compliance or recipient-delivery assertion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_source(directory, package, version):
    """apt verifies archive authentication; verify the retained .dsc/file binding."""
    descriptors = list(directory.glob('*.dsc'))
    if len(descriptors) != 1:
        raise RuntimeError(f'{package}={version}: expected one source descriptor')
    text = descriptors[0].read_text()
    if f'\nSource: {package}\n' not in '\n' + text or f'\nVersion: {version}\n' not in '\n' + text:
        raise RuntimeError(f'{package}={version}: source descriptor identity mismatch')
    if '\nChecksums-Sha256:\n' not in text:
        raise RuntimeError(f'{package}={version}: no source SHA-256 section')
    checksums = text.split('\nChecksums-Sha256:\n', 1)[1]
    files = []
    for line in checksums.splitlines():
        if not line.startswith(' '):
            break
        digest, size, name = line.split()
        if Path(name).name != name:
            raise RuntimeError(f'Unsafe source filename: {name}')
        path = directory / name
        if path.stat().st_size != int(size) or sha256(path) != digest:
            raise RuntimeError(f'Source hash/size mismatch: {path}')
        files.append(name)
    if not files:
        raise RuntimeError(f'{package}={version}: no source files')


def capture(root: Path, archive: Path):
    manifest = json.loads((root / 'manifest.json').read_text())
    stage = archive.parent / 'linux-native-sources'
    stage.mkdir()  # never mix sources from different builds
    shutil.copy2(root / 'manifest.json', stage / 'gtk-manifest.json')
    for package, version in sorted(manifest['source_packages'].items()):
        directory = stage / 'debian' / package
        directory.mkdir(parents=True)
        subprocess.run(['apt-get', 'source', '--download-only', '--only-source', package + '=' + version],
                       cwd=directory, check=True)
        verify_source(directory, package, version)
    for relative in ('share/doc', 'share/common-licenses'):
        shutil.copytree(root / relative, stage / relative)
    repo = Path(__file__).resolve().parents[1]
    for relative in ('build_linux.sh', 'Ouroboros.spec', 'pyproject.toml', 'uv.lock', 'requirements-runtime.lock',
                     'scripts/bundle_linux_gtk.py', 'scripts/capture_linux_sources.py',
                     'packaging/appimage/desktop-env.sh', 'packaging/appimage/desktop-launcher.sh',
                     'packaging/appimage/AppRun', 'packaging/appimage/README.md'):
        target = stage / 'build-inputs' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, target)
    indices = stage / 'apt-release'
    indices.mkdir()
    for path in Path('/var/lib/apt/lists').glob('*InRelease'):
        shutil.copy2(path, indices / path.name)
    with (stage / 'builder-packages.tsv').open('w') as stream:
        subprocess.run(['dpkg-query', '-W'], stdout=stream, check=True)
    (stage / 'SHA256SUMS').write_text(''.join(
        sha256(path) + '  ' + path.relative_to(stage).as_posix() + '\n'
        for path in sorted(stage.rglob('*')) if path.is_file()))
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(stage, arcname='linux-native-sources')
    archive.with_suffix(archive.suffix + '.sha256').write_text(sha256(archive) + '  ' + archive.name + '\n')


if __name__ == '__main__':
    capture(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
