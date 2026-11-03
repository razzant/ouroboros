"""Consumers of durable test evidence after the enclosing launcher releases scratch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ouroboros.test_evidence import diagnostic_directory

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.serial


def test_native_source_can_be_recovered_after_outer_launcher_release(tmp_path):
    evidence_parent = diagnostic_directory(tmp_path / 'durable', 'source-lifetime')
    # This input is outside the launcher repository: preserve it explicitly, not
    # by pretending safe_test captures arbitrary command inputs.
    probe = tmp_path / 'test_source_probe.py'
    probe.write_text('''import os, pathlib, subprocess
from ouroboros.preflight_runner import run_hermetic_pytest

def test_native(tmp_path):
    repo = tmp_path / 'ephemeral-source'
    (repo / 'tests').mkdir(parents=True)
    path = repo / 'tests/test_original.py'
    path.write_bytes(b'def test_original():\\n    assert True\\n')
    (repo / '.gitattributes').write_text('tests/test_original.py export-ignore export-subst\\n')
    for args in (['init'], ['add', '.'], ['-c','user.name=Fixture','-c','user.email=f@example.invalid','commit','-qm','base']):
        subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)
    path.write_bytes(path.read_bytes() + b'\\n# dirty exact \\xc3\\xa9\\r\\n')
    (repo / 'admitted.bin').write_bytes(b'new\\x00input\\r\\n')
    assert run_hermetic_pytest(repo, pytest_args=['tests','-q'], timeout=60) is None
''', encoding='utf-8')
    (evidence_parent / 'probe.py').write_bytes(probe.read_bytes())
    result = subprocess.run([sys.executable, '-I', '-S', str(REPO / 'scripts/safe_test.py'),
        '--temp-parent', '/tmp' if os.name == 'posix' else str(tmp_path),
        '--evidence-dir', str(evidence_parent), '--', sys.executable, '-m', 'pytest',
        '-p', 'tests.conftest', '--rootdir', str(tmp_path), '--confcutdir', str(tmp_path),
        '-o', 'addopts=', '-q', '-s', str(probe)], cwd=REPO, capture_output=True, timeout=120)
    (evidence_parent / 'launcher.stdout').write_bytes(result.stdout)
    (evidence_parent / 'launcher.stderr').write_bytes(result.stderr)
    assert result.returncode == 0, (result.stdout + result.stderr).decode('utf-8', 'replace')
    outer = Path(next(line.split(' ', 1)[1] for line in result.stderr.decode().splitlines()
                      if line.startswith('SAFE_TEST_EVIDENCE ')))
    outer_result = json.loads((outer / 'result.json').read_bytes())
    assert not Path(outer_result['scratch']).exists(), outer_result
    sources = [path for path in outer.glob('run-*/source/source.json')]
    assert len(sources) == 1
    source = sources[0].parent
    record = json.loads(sources[0].read_bytes())
    assert not Path(record['source_repository']).exists()
    # HEAD/path alone are no longer useful. Recover actual source bytes without
    # consulting the vanished repository or another worktree/object database.
    archive = source / 'base.pack'
    assert archive.is_file(), 'committed input lost when the outer owner released its pytest basetemp'
    assert record['base']['self_contained'] is True
    recovered = tmp_path / 'recovered'
    recovered.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=recovered, check=True)
    with archive.open('rb') as stream:
        subprocess.run(['git', 'unpack-objects', '-q'], cwd=recovered, stdin=stream, check=True)
    assert subprocess.run(['git', 'cat-file', '-e', record['head']], cwd=recovered, capture_output=True).returncode != 0, 'do not archive repository history'
    subprocess.run(['git', 'read-tree', record['base']['tree']], cwd=recovered, check=True)
    subprocess.run(['git', 'checkout-index', '-a'], cwd=recovered, check=True)
    original = (recovered / 'tests/test_original.py').read_bytes()
    assert original == b'def test_original():\n    assert True\n'
    apply = subprocess.run(['git', 'apply', '--binary', '--unsafe-paths', str(source / 'candidate.diff')],
                           cwd=recovered, capture_output=True)
    assert apply.returncode == 0, apply.stderr
    expected = original + b'\n# dirty exact \xc3\xa9\r\n'
    assert (recovered / 'tests/test_original.py').read_bytes() == expected
    assert (source / 'untracked/admitted.bin').read_bytes() == b'new\x00input\r\n'
    assert record['base']['sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_atomic_record_keeps_prior_bytes_on_failed_publication(tmp_path, monkeypatch):
    from ouroboros import test_evidence, utils
    target = tmp_path / 'result.json'
    test_evidence.write_record(target, {'first': 'é\nexact'})
    before = target.read_bytes()
    assert b'\r\n' not in before and before.endswith(b'\n')
    def broken(*_args):
        raise OSError('controlled replace failure')
    monkeypatch.setattr(utils, 'replace_atomic', broken)
    with pytest.raises(OSError, match='controlled replace failure'):
        test_evidence.write_record(target, {'second': True})
    assert target.read_bytes() == before
    assert not list(tmp_path.glob('.result.json.tmp.*'))


@pytest.mark.parametrize('outcome', ['raised', 'gap'])
def test_failed_source_export_retains_untested_source(tmp_path, monkeypatch, capsys, outcome):
    from ouroboros import preflight_runner, test_evidence
    from ouroboros.force_remove import force_rmtree
    from tests.test_test_storage import _repo
    envelope = tmp_path / 'envelope'
    envelope.mkdir()
    repo = _repo(envelope / 'source', 'def test_probe():\n    assert True\n')
    monkeypatch.setenv('OUROBOROS_TEST_TEMP_ROOT', str(envelope))
    exports = []
    def failed(source, destination):
        exports.append(str(source))
        if outcome == 'gap':  # an incomplete recipe is no proof input either
            return {'gaps': [{'path': 'input.bin', 'error': 'OSError: controlled source export failure'}]}
        raise OSError('controlled source export failure')
    monkeypatch.setattr(test_evidence, 'source_evidence', failed)
    monkeypatch.setattr(preflight_runner, '_execute_pytest_pass', lambda *args: pytest.fail('must not run'))
    verdict = preflight_runner.run_hermetic_pytest(repo, pytest_args=['tests', '-q'], timeout=60)
    assert 'controlled source export failure' in verdict
    evidence = Path(next(line.split(' ', 1)[1] for line in capsys.readouterr().err.splitlines()
                         if line.startswith('PREFLIGHT_EVIDENCE ')))
    record = json.loads((evidence / 'result.json').read_bytes())
    scratch = Path(record['scratch'])
    assert record['execution'] == 'not_run' and record['cleanup'] == 'retained', record
    assert 'candidate evidence incomplete' in record['retention_reason']
    assert (scratch / 'OUROBOROS_RETAINED.txt').is_file()
    assert exports == [str(scratch / 'repo')], 'export the tested input once, never an earlier partial recipe'
    assert (scratch / 'repo/tests/test_probe.py').is_file()
    assert (envelope / 'OUROBOROS_RETAINED.txt').is_file()
    subprocess.run(['git', 'worktree', 'remove', '--force', str(scratch / 'repo')], cwd=repo, check=True)
    # No lane started. These exact synthetic roots have no worker/descendant;
    # all failed-export metadata is already outside them. The source's Git objects
    # are read-only, which a plain rmtree cannot delete on Windows.
    for root in (scratch, envelope):
        force_rmtree(root)
        assert not root.exists(), root


def test_debug_retention_keeps_real_test_verdict(tmp_path, monkeypatch, capsys):
    import shutil
    from ouroboros import preflight_runner
    from tests.test_test_storage import _repo
    repo = _repo(tmp_path / 'source', 'def test_probe():\n    assert False\n')
    monkeypatch.setenv('OUROBOROS_PREFLIGHT_KEEP', '1')
    verdict = preflight_runner.run_hermetic_pytest(repo, pytest_args=['tests', '-q'], timeout=60)
    assert verdict is not None
    evidence = Path(next(line.split(' ', 1)[1] for line in capsys.readouterr().err.splitlines()
                         if line.startswith('PREFLIGHT_EVIDENCE ')))
    record = json.loads((evidence / 'result.json').read_bytes())
    assert record['verdict'] == verdict and record['execution'] == 'executed'
    assert record['cleanup'] == 'retained' and 'debug' in record['retention_reason']
    assert record['containment_error'] == ''
    scratch = Path(record['scratch'])
    subprocess.run(['git', 'worktree', 'remove', '--force', str(scratch / 'repo')], cwd=repo, check=True)
    shutil.rmtree(scratch)  # This exact run's retained ended tree; original raw logs persist.


def test_diagnostic_names_are_stable_per_producer_but_separate_invocations(tmp_path, monkeypatch):
    monkeypatch.setenv('OUROBOROS_TEST_EVIDENCE_ROOT', str(tmp_path / 'evidence'))
    first = diagnostic_directory(tmp_path / 'worker0/test0', 'effort-settings')
    second = diagnostic_directory(tmp_path / 'worker1/test0', 'effort-settings')
    assert first != second
    (first / 'same.png').write_bytes(b'first invocation')
    (second / 'same.png').write_bytes(b'second invocation')
    assert first == diagnostic_directory(tmp_path / 'worker0/test0', 'effort-settings')
    assert (first / 'same.png').read_bytes() == b'first invocation'


def test_scratch_readonly_retry_is_windows_only_and_failed_retry_propagates(tmp_path, monkeypatch):
    from ouroboros import test_evidence
    calls = []
    # Exercise the same callback Windows rmtree invokes for read-only objects;
    # this is not a claim to execute a native NTFS sharing/ACL consumer on macOS.
    def rmtree(path, *, onerror):
        def unlink(name):
            calls.append(('retry', name))
        onerror(unlink, str(path), (PermissionError, PermissionError('readonly'), None))
    # End the temporary platform projection before other fixture teardowns run.
    with monkeypatch.context() as patch:
        patch.setattr(test_evidence.shutil, 'rmtree', rmtree)
        patch.setattr(test_evidence.os, 'chmod', lambda name, mode: calls.append(('chmod', name)))
        patch.setattr(test_evidence.os, 'name', 'nt')
        test_evidence.remove_scratch_tree(str(tmp_path))
        assert [event[0] for event in calls] == ['chmod', 'retry']
        def denied(*_args):
            raise PermissionError('still shared')
        patch.setattr(test_evidence.os, 'chmod', denied)
        with pytest.raises(PermissionError, match='still shared'):
            test_evidence.remove_scratch_tree(str(tmp_path))
