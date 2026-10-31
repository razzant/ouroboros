"""Small real-lane falsifiers for #1908 export facts and source reconstruction."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import test_evidence
from ouroboros.force_remove import force_rmtree
from ouroboros.test_environment import declare_scratch
from tests.test_test_storage import _repo

pytestmark = pytest.mark.serial


def _git(repo, *args, **kwargs):
    return subprocess.check_output(['git', *args], cwd=repo, **kwargs)


@pytest.mark.parametrize('fault', ['record', 'fsync', 'preview'])
@pytest.mark.parametrize('lane', ['pytest', 'node'])
@pytest.mark.parametrize('red', [False, True])
def test_export_failure_preserves_real_lane_facts(tmp_path, monkeypatch, capsys, fault, lane, red):
    from ouroboros import preflight_runner, preflight_node
    repo = _repo(tmp_path / 'source', 'def test_probe():\n    print("REAL_PYTEST_TAIL")\n    assert ' + str(not red) + '\n')
    if lane == 'node':
        web = repo / 'web/tests'
        web.mkdir(parents=True)
        (web / 'probe.test.js').write_text('const t=require("node:test"); t("probe",()=>{console.log("REAL_NODE_TAIL");' + ('throw Error("red")' if red else '') + '});')
        node = os.environ.get('STORAGE_TEST_NODE') or preflight_node.resolve_node()
        assert node and preflight_node.probe_node_version(node), 'a real executable Node is required'
        monkeypatch.setattr(preflight_node, 'resolve_node', lambda: node)
    original_record, original_fsync, original_preview = test_evidence.write_record, test_evidence.os.fsync, test_evidence.preview_file
    def record(path, value):
        if Path(path).parent.name.startswith(lane + '-') and Path(path).name == 'result.json' and fault == 'record':
            raise OSError(28, 'controlled lane record ENOSPC')
        return original_record(path, value)
    def fsync(fd):
        # Only raw stream descriptors, not source/atomic metadata publication.
        if fault == 'fsync' and any(os.fstat(fd).st_ino == path.stat().st_ino
            for path in evidence_parent.glob('run-*/' + lane + '-*/*.bin')):
            raise OSError(5, 'controlled stream fsync failure')
        return original_fsync(fd)
    def preview(path, *args, **kwargs):
        if Path(path).parent.name.startswith(lane + '-') and fault == 'preview':
            raise OSError(5, 'controlled preview read failure')
        return original_preview(path, *args, **kwargs)
    evidence_parent = test_evidence.diagnostic_directory(tmp_path / 'evidence', 'export-' + tmp_path.name)
    monkeypatch.setenv(test_evidence.EVIDENCE_ENV, str(evidence_parent))
    monkeypatch.setattr(test_evidence, 'write_record', record)
    monkeypatch.setattr(test_evidence.os, 'fsync', fsync)
    monkeypatch.setattr(test_evidence, 'preview_file', preview)
    verdict = preflight_runner.run_hermetic_pytest(repo, pytest_args=['tests', '-q', '-s'], timeout=60)
    text = capsys.readouterr().err
    evidence = Path(next(line.split(' ', 1)[1] for line in text.splitlines() if line.startswith('PREFLIGHT_EVIDENCE ')))
    result = json.loads((evidence / 'result.json').read_bytes())
    try:
        # Export faults are not pytest/Node exceptions or unconfirmed teardown.
        assert result['containment_error'] == '', result
        assert (verdict is not None) == red, verdict
        assert result['diagnostics_incomplete'] is True, result
        observed = next(item for item in result['lanes'] if item['lane'] == lane)
        assert observed['returncode'] == int(red) and observed['containment_error'] == '', observed
        assert result['cleanup'] == 'retained' and 'evidence' in result['retention_reason'], result
        assert 'EVIDENCE_INCOMPLETE' in text
        assert b'REAL_' in next(evidence.glob(lane + '-*/stdout.bin')).read_bytes()
    finally:
        # The exact run has reaped. Preserve raw bytes outside its synthetic tree.
        scratch = Path(result['scratch'])
        if scratch.exists():
            subprocess.run(['git', 'worktree', 'remove', '--force', str(scratch / 'repo')], cwd=repo, check=True)
            shutil.rmtree(scratch)


def test_disposable_recipe_recovers_index_live_untracked_modes_without_source(tmp_path, monkeypatch, capsys):
    from ouroboros.preflight_runner import run_hermetic_pytest
    envelope = tmp_path / 'ended-source'
    envelope.mkdir()
    declare_scratch(envelope, 'pytest-basetemp')
    source = _repo(envelope / 'repo', 'def test_probe():\n    assert True\n')
    dual = source / 'dual.txt'
    dual.write_bytes(b'head\n')
    _git(source, 'add', '.')
    _git(source, '-c', 'user.name=Fixture', '-c', 'user.email=f@example.invalid', 'commit', '-qm', 'head')
    head = _git(source, 'rev-parse', 'HEAD').strip().decode()
    head_tree = _git(source, 'rev-parse', 'HEAD^{tree}').strip().decode()
    dual.write_bytes(b'only-staged\x00bytes\n')
    dual.chmod(0o755)
    _git(source, 'add', 'dual.txt')
    index_tree = _git(source, 'write-tree').strip().decode()
    dual.write_bytes(b'live\x00different\n')
    dual.chmod(0o644)
    executable = source / 'new-executable'
    executable.write_bytes(b'#!/bin/sh\nprintf retained\n')
    executable.chmod(0o751)
    monkeypatch.setenv('OUROBOROS_TEST_TEMP_ROOT', str(envelope))
    destination = test_evidence.diagnostic_directory(tmp_path / 'evidence', 'recipe-' + tmp_path.name)
    monkeypatch.setenv(test_evidence.EVIDENCE_ENV, str(destination))
    assert run_hermetic_pytest(source, pytest_args=['tests', '-q'], timeout=60) is None
    evidence = Path(next(line.split(' ', 1)[1] for line in capsys.readouterr().err.splitlines() if line.startswith('PREFLIGHT_EVIDENCE '))) / 'source'
    record = json.loads((evidence / 'source.json').read_bytes())
    force_rmtree(envelope)  # The only source/object owner was this test's stopped synthetic repo.
    assert not envelope.exists(), 'read-only Git objects (Windows) kept the source alive'
    recovered = tmp_path / 'recovered'
    recovered.mkdir()
    _git(recovered, 'init', '-q')
    pack = evidence / 'base.pack'
    with pack.open('rb') as stream:
        subprocess.run(['git', 'unpack-objects', '-q'], cwd=recovered, stdin=stream, check=True)
    assert record['head'] == head and record['base']['tree'] == head_tree
    assert subprocess.run(['git', 'cat-file', '-e', head], cwd=recovered, capture_output=True).returncode != 0
    assert record['index']['tree'] == index_tree
    _git(recovered, 'read-tree', head_tree)
    _git(recovered, 'checkout-index', '-a')
    subprocess.run(['git', 'apply', '--binary', str(evidence / 'candidate.diff')], cwd=recovered, check=True)
    _git(recovered, 'read-tree', record['index']['tree'])  # Must NOT update live files.
    assert _git(recovered, 'show', ':dual.txt') == b'only-staged\x00bytes\n'
    assert (recovered / 'dual.txt').read_bytes() == b'live\x00different\n'
    if os.name != 'nt':
        assert stat.S_IMODE((recovered / 'dual.txt').stat().st_mode) == 0o644
    for item in record['untracked']:
        target = recovered / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(evidence / 'untracked' / item['path'], target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']
        assert stat.S_IMODE(target.stat().st_mode) == item['mode']
    if os.name != 'nt':
        assert stat.S_IMODE((recovered / 'new-executable').stat().st_mode) == 0o751


def test_non_git_source_still_executes_and_names_missing_recipe(tmp_path, capsys):
    from ouroboros.safe_test_supervisor import main
    root, source, evidence = tmp_path / 'scratch', tmp_path / 'no-git', test_evidence.diagnostic_directory(tmp_path / 'evidence', 'no-git-' + tmp_path.name)
    root.mkdir()
    source.mkdir()
    assert main(root, source, [sys.executable, '-c', 'print("NON_GIT_RAN")'], evidence_parent=evidence) == 0
    run = Path(next(line.split(' ', 1)[1] for line in capsys.readouterr().err.splitlines() if line.startswith('SAFE_TEST_EVIDENCE ')))
    assert b'NON_GIT_RAN' in next(run.glob('command-*/stdout.bin')).read_bytes()
    record = json.loads((run / 'source/source.json').read_bytes())
    assert record['kind'] == 'non-git' and record['source_snapshot'] == 'unavailable'


def test_ctx_raw_evidence_is_not_deliverable_and_outlives_execution_drive(tmp_path, monkeypatch):
    from ouroboros.artifacts import collect_task_artifact_records
    canonical = tmp_path / 'installation'
    drive = canonical / 'state/headless_tasks/actor/data'
    drive.mkdir(parents=True)
    # Synthetic ctx only; no real installation or task document is touched.
    monkeypatch.delenv('OUROBOROS_TEST_TEMP_ROOT', raising=False)
    path = test_evidence.evidence_directory(tmp_path, ctx=SimpleNamespace(drive_root=drive, task_id='actor'))
    (path / 'stdout.bin').write_bytes(b'raw diagnostic')
    assert collect_task_artifact_records(canonical, 'actor') == []
    assert collect_task_artifact_records(drive, 'actor') == []
    shutil.rmtree(drive)
    assert (path / 'stdout.bin').read_bytes() == b'raw diagnostic'


def _persistent_recipe(source, destination, monkeypatch):
    """Record ``source`` as a persistent owner while another git process holds its index lock."""
    lock = source / '.git/index.lock'
    lock.write_bytes(b'held by a concurrent git process')
    before = (source / '.git/index').read_bytes()
    # pytest's own scratch is declared disposable; the store premise here is persistent.
    monkeypatch.setattr(test_evidence, '_disposable_store', lambda _common: False)
    try:
        record = test_evidence.source_evidence(source, destination)
    finally:
        assert lock.read_bytes() == b'held by a concurrent git process', 'the live lock was taken or removed'
        lock.unlink()
    assert (source / '.git/index').read_bytes() == before, 'the recorded index must not be rewritten'
    assert record['base'] == {'kind': 'repository', 'git_common_dir': str((source / '.git').resolve()),
                              'tree': _git(source, 'rev-parse', 'HEAD^{tree}').strip().decode(),
                              'self_contained': False, 'alternates': []}
    pack = destination / record['index']['pack']['path']
    assert record['index']['pack']['sha256'] == hashlib.sha256(pack.read_bytes()).hexdigest()
    return record, pack


def _borrowing_base(tmp_path, source, pack):
    """The pack's own objects (read in an isolated empty store), and a store that adds them
    to the persistent owner's objects; unpack-objects skips what alternates already have."""
    stores = []
    for name, alternate in (('pack-only', None), ('recovered', source / '.git/objects')):
        store = tmp_path / name
        store.mkdir()
        _git(store, 'init', '-q')
        if alternate is not None:
            # LF bytes: Git reads the whole line as the path, and Windows text mode would add a CR.
            (store / '.git/objects/info/alternates').write_bytes(os.fsencode(alternate) + b'\n')
        with pack.open('rb') as stream:
            subprocess.run(['git', 'unpack-objects', '-q'], cwd=store, stdin=stream, check=True)
        stores.append(store)
    loose = {path.parent.name + path.name for path in (stores[0] / '.git/objects').glob('??/*')}
    return stores[1], loose


def test_persistent_recipe_packs_only_index_objects_and_reconstructs_staged_live_binary_modes(tmp_path, monkeypatch):
    source = _repo(tmp_path / 'persistent', 'def test_probe():\n    assert True\n')
    (source / 'large-unchanged.bin').write_bytes(os.urandom(256 * 1024))
    (source / 'dual.txt').write_bytes(b'head\n')
    (source / 'tool.sh').write_bytes(b'#!/bin/sh\necho tool\n')
    _git(source, 'add', '.')
    _git(source, '-c', 'user.name=Fixture', '-c', 'user.email=f@example.invalid', 'commit', '-qm', 'base')
    head_tree = _git(source, 'rev-parse', 'HEAD^{tree}').strip().decode()
    (source / 'dual.txt').write_bytes(b'only-staged\x00bytes\n')
    (source / 'added.bin').write_bytes(b'\x00\xffstaged binary\r\n')
    _git(source, 'add', 'dual.txt', 'added.bin')
    _git(source, 'update-index', '--chmod=+x', 'tool.sh')
    index_tree = _git(source, 'write-tree').strip().decode()
    (source / 'dual.txt').write_bytes(b'live\x00different\n')
    (source / 'untracked.txt').write_bytes(b'admitted\r\n')
    (source / 'untracked.txt').chmod(0o640)
    destination = tmp_path / 'evidence'
    record, pack = _persistent_recipe(source, destination, monkeypatch)
    assert record['index']['tree'] == index_tree and record['index']['pack']['relative_to'] == head_tree
    # Exactly the index-only trees/blobs: never the unchanged base, history or a commit.
    expected = {line.split()[0] for line in _git(source, 'rev-list', '--objects', index_tree, '--not', head_tree)
                .decode().splitlines()}
    recovered, loose = _borrowing_base(tmp_path, source, pack)
    assert loose == expected and record['index']['pack']['objects'] == len(expected)
    full = _git(source, 'pack-objects', '--revs', '--stdout', input=f'{head_tree}\n{index_tree}\n'.encode())
    assert record['index']['pack']['bytes'] * 20 < len(full), 'the former full-base export was not avoided'
    _git(recovered, 'read-tree', head_tree)
    _git(recovered, 'checkout-index', '-a')
    subprocess.run(['git', 'apply', '--binary', str(destination / 'candidate.diff')], cwd=recovered, check=True)
    _git(recovered, 'read-tree', index_tree)  # Must NOT update live files.
    assert _git(recovered, 'show', ':dual.txt') == b'only-staged\x00bytes\n'
    assert (recovered / 'dual.txt').read_bytes() == b'live\x00different\n'
    assert _git(recovered, 'show', ':added.bin') == b'\x00\xffstaged binary\r\n'
    assert _git(recovered, 'ls-files', '-s', 'tool.sh').startswith(b'100755 ')
    assert (recovered / 'large-unchanged.bin').read_bytes() == (source / 'large-unchanged.bin').read_bytes()
    [item] = record['untracked']
    assert item['path'] == 'untracked.txt' and (destination / 'untracked/untracked.txt').read_bytes() == b'admitted\r\n'
    if os.name != 'nt':
        assert item['mode'] == 0o640


def test_a_seat_clone_recipe_names_where_its_referenced_base_lives(tmp_path, monkeypatch):
    source = _repo(tmp_path / 'source', 'def test_probe():\n    assert True\n')
    clone = tmp_path / 'seat' / 'area'  # review_checking's seat copy: `git clone --shared`
    _git(tmp_path, 'clone', '-q', '--shared', str(source), str(clone))
    monkeypatch.setattr(test_evidence, '_disposable_store', lambda _common: False)
    base = test_evidence.source_evidence(clone, tmp_path / 'evidence')['base']
    assert (base['kind'], base['git_common_dir'], base['self_contained']) == ('repository', str((clone / '.git').resolve()), False)
    assert base['alternates'] == [str((source / '.git/objects').resolve())]


def test_persistent_unmerged_recipe_keeps_stage_blobs_missing_from_base(tmp_path, monkeypatch):
    source = _repo(tmp_path / 'persistent', 'def test_probe():\n    assert True\n')
    identity = ['-c', 'user.name=Fixture', '-c', 'user.email=f@example.invalid']
    commit = [*identity, 'commit', '-qam']
    _git(source, 'checkout', '-qb', 'side')
    (source / 'tests/test_probe.py').write_text('def test_probe():\n    assert "side"\n')
    _git(source, *commit, 'side')
    _git(source, 'checkout', '-q', '-')
    (source / 'tests/test_probe.py').write_text('def test_probe():\n    assert "ours"\n')
    _git(source, *commit, 'ours')
    # The test environment has no global identity (GIT_CONFIG_GLOBAL=/dev/null): without one a
    # host that cannot guess it (Linux, Windows) refuses the merge (128) before any stage exists.
    merged = subprocess.run(['git', *identity, 'merge', '-q', 'side'], cwd=source, capture_output=True)
    assert merged.returncode == 1, merged
    assert _git(source, 'ls-files', '-u').decode().count('\ttests/test_probe.py\n') == 3
    destination = tmp_path / 'evidence'
    record, pack = _persistent_recipe(source, destination, monkeypatch)
    assert record['index']['unmerged'] is True and record['index']['tree'] is None
    entries = (destination / 'index.entries').read_bytes()
    assert entries == _git(source, 'ls-files', '--stage', '-z')
    head_blobs = {line.split()[2] for line in _git(source, 'ls-tree', '-r', 'HEAD').decode().splitlines()}
    stages = {entry.split(b'\t', 1)[0].split()[1].decode() for entry in entries.split(b'\0') if entry}
    recovered, loose = _borrowing_base(tmp_path, source, pack)
    assert loose == stages - head_blobs and loose  # merge base and theirs; ours stays with the owner
    # The exact unmerged stages are restorable from entries plus base and pack objects.
    info = b''.join(entry + b'\0' for entry in entries.split(b'\0') if entry)
    subprocess.run(['git', 'update-index', '-z', '--index-info'], cwd=recovered, input=info, check=True)
    assert _git(recovered, 'ls-files', '--stage', '-z') == entries
