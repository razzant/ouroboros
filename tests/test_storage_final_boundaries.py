"""#1908 final falsifiers: real interrupt export and disposable caller lifetimes."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests.test_test_storage import REPO, _repo


def _interrupt_probe(probe, evidence, workers):
    probe.mkdir()
    ready = [evidence / f'ready-{i}' for i in range(max(workers, 1))]
    survivors = [evidence / f'survivor-{i}' for i in range(max(workers, 1))]
    # A child the test leaves behind: it outlives the INT-exited pytest root
    # and records the TERM that a graceful teardown owes it.
    (probe / 'survivor.py').write_text(
        'import os, signal, sys, time\nfrom pathlib import Path\n'
        'def stop(signum, _frame):\n    Path(sys.argv[1]).write_text(signal.Signals(signum).name)\n    sys.exit(0)\n'
        'signal.signal(signal.SIGTERM, stop)\nready = Path(sys.argv[2])\n'
        'pending = ready.with_suffix(".pending")\npending.write_text(sys.argv[3])\n'
        'os.replace(pending, ready)\ntime.sleep(120)\n')
    for i in range(len(ready)):
        # Separate files plus loadfile give each worker one genuinely busy test.
        (probe / f'test_wait_{i}.py').write_text(
            'import os, subprocess, sys, time\ndef test_wait():\n'
            f'    subprocess.Popen([sys.executable, {str(probe / "survivor.py")!r}, '
            f'{str(survivors[i])!r}, {str(ready[i])!r}, os.environ.get("PYTEST_XDIST_WORKER", "main")])\n'
            '    print("PYTEST_READY", flush=True)\n    time.sleep(120)\n')
    # The child's final xdist configuration is the admission authority. Memory can change
    # between this parent and run_tests; only stage a busy file for each admitted worker.
    (probe / 'conftest.py').write_text(
        'from pathlib import Path\n'
        'def pytest_configure(config):\n'
        '    if hasattr(config, "workerinput"):\n        return\n'
        '    workers = config.getoption("numprocesses", default=0) or 0\n'
        f'    for i in range(max(workers, 1), {max(workers, 1)}):\n'
        '        (Path(__file__).parent / f"test_wait_{i}.py").unlink()\n'
        f'    target = Path({str(evidence / "admitted")!r})\n'
        '    pending = target.with_suffix(".pending")\n'
        '    pending.write_text(str(workers))\n    pending.replace(target)\n')
    return ready, survivors


@pytest.mark.skipif(os.name == 'nt', reason='POSIX SIGINT delivery; Windows keeps Job teardown')
@pytest.mark.parametrize('entry,workers,repeat', [
    ('safe_test', 0, False), ('run_tests', 0, False),
    ('run_tests', 2, False), ('run_tests', 2, True),
])
def test_ctrl_c_exports_real_pytest_terminal_results(tmp_path, entry, workers, repeat):
    from ouroboros import platform_layer as pl
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.test_evidence import EVIDENCE_ENV
    from ouroboros.test_parallelism import NESTED_WORKERS_ENV

    evidence = diagnostic_directory(tmp_path / 'evidence', 'ctrl-c')
    probe = tmp_path / 'probe'
    ready, survivors = _interrupt_probe(probe, evidence, workers)
    junit = evidence / 'interrupt.xml'
    pytest_args = ['-p', 'tests.conftest', '--rootdir', str(probe), '--confcutdir', str(probe),
                   '-o', 'addopts=', '-s', '-q', f'--junitxml={junit}', str(probe)]
    if workers:
        pytest_args += ['-n', str(workers), '--dist', 'loadfile']
    # The ordinary CLI (`make test`) and the direct launcher; run_tests takes
    # its safe_test evidence parent from the environment, like any nested owner.
    command = ([sys.executable, str(REPO / 'scripts/run_tests.py'), *pytest_args] if entry == 'run_tests' else
               [sys.executable, '-I', '-S', str(REPO / 'scripts/safe_test.py'), '--evidence-dir', str(evidence),
                '--', sys.executable, '-m', 'pytest', *pytest_args])
    container = ProcessContainer()
    with (evidence / 'launcher.log').open('wb') as out:
        proc = container.spawn(command, cwd=REPO, stdout=out, stderr=subprocess.STDOUT,
                               env={**os.environ, EVIDENCE_ENV: str(evidence),
                                    NESTED_WORKERS_ENV: str(max(workers, 1))})
        try:
            # The ready files are the ordering signal; the deadline only bounds a hung launch. Two
            # xdist workers importing tests.conftest under a loaded runner took over 40 s on CI.
            deadline = time.monotonic() + 180
            admission = evidence / 'admitted'
            while not admission.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert admission.exists(), (evidence / 'launcher.log').read_text()
            admitted = int(admission.read_text())
            assert 0 <= admitted <= workers and bool(admitted) == bool(workers)
            workers = admitted
            ready, survivors = ready[:max(workers, 1)], survivors[:max(workers, 1)]
            while not all(p.exists() for p in ready) and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert all(p.exists() for p in ready), (evidence / 'launcher.log').read_text()
            if workers:
                assert {p.read_text() for p in ready} == {f'gw{i}' for i in range(workers)}, (
                    evidence / 'launcher.log').read_text()
            assert pl.signal_pid(proc.pid, signal.SIGINT)
            if repeat:
                # Every worker is busy. A second launcher INT must not cut teardown.
                time.sleep(.05)
                assert pl.signal_pid(proc.pid, signal.SIGINT)
            code = proc.wait(timeout=60)
        finally:
            assert container.reap() == ''
            container.close()
    [run] = [path for path in evidence.glob('run-*') if (path / 'result.json').exists()]
    record = json.loads((run / 'result.json').read_bytes())
    raw = next(run.glob('command-*/stdout.bin')).read_bytes()
    facts = {
        'exit': code, 'keyboard_interrupt': b'KeyboardInterrupt' in raw,
        'junit_complete': junit.is_file() and b'</testsuites>' in junit.read_bytes(),
        # safe_test's own export, or run_tests' explicit --ci-evidence-dir.
        'terminal_exits': [json.loads(path.read_bytes())['session_exit_code'] for path in run.rglob('results.json')],
        'survivors': [p.read_text() if p.exists() else 'no TERM before the forced reap' for p in survivors],
        # Observed here: run_tests reaped and reported before safe_test escalated.
        'runner_reported': entry != 'run_tests' or b'=== focused: exit 130' in raw,
        'containment_error': record['containment_error'], 'scratch_released': not Path(record['scratch']).exists(),
    }
    assert facts == {'exit': 130, 'keyboard_interrupt': True, 'junit_complete': True, 'terminal_exits': [2],
                     'survivors': ['SIGTERM'] * len(survivors), 'runner_reported': True, 'containment_error': '',
                     'scratch_released': True}, raw.decode(errors='replace')[-4000:]


@pytest.mark.skipif(os.name == 'nt', reason='symlink creation may require Windows privileges')
def test_generic_owner_records_untracked_links_without_following_or_refusing(tmp_path, capsys):
    from ouroboros.safe_test_supervisor import main

    source = _repo(tmp_path / 'source', 'def test_ok():\n    assert True\n')
    outside = tmp_path / 'outside'
    outside.write_bytes(b'UNIQUE_EXTERNAL_INPUT')
    (source / 'external').symlink_to(outside)
    (source / 'unreadable').symlink_to(tmp_path / 'absent-target')
    regular = source / 'regular'
    regular.write_bytes(b'exact\x00bytes')
    regular.chmod(0o751)
    root = tmp_path / 'scratch'
    root.mkdir()
    parent = diagnostic_directory(tmp_path / 'evidence', 'symlink')
    assert main(root, source, [sys.executable, '-c', 'print("LINK_INPUT_COMMAND_RAN")'],
                evidence_parent=parent) == 0
    run = Path(next(line.split(' ', 1)[1] for line in capsys.readouterr().err.splitlines()
                    if line.startswith('SAFE_TEST_EVIDENCE ')))
    assert b'LINK_INPUT_COMMAND_RAN' in next(run.glob('command-*/stdout.bin')).read_bytes()
    record = json.loads((run / 'source/source.json').read_bytes())
    links = {item['path']: item for item in record['untracked'] if item.get('kind') == 'symlink'}
    assert set(links) == {'external', 'unreadable'}
    assert links['external']['target'] == str(outside)
    assert all(item['contents'] == 'unrecorded (target not followed)' for item in links.values())
    assert not (run / 'source/untracked/external').exists()
    assert outside.read_bytes() == b'UNIQUE_EXTERNAL_INPUT'
    assert (run / 'source/untracked/regular').read_bytes() == b'exact\x00bytes'
    assert next(item for item in record['untracked'] if item['path'] == 'regular')['mode'] == 0o751


def test_in_checkout_evidence_is_never_its_own_input_across_repeated_and_nested_runs(tmp_path, capsys):
    from ouroboros.safe_test_supervisor import main

    # The checkout itself outlives this test's scratch, like a reviewer seat's copy.
    source = _repo(diagnostic_directory(tmp_path / 'seat', 'in-checkout') / 'area', 'def test_ok():\n    assert True\n').resolve()
    (source / 'ordinary.bin').write_bytes(b'ordinary\x00input')
    parent = source / '.review-evidence' / 'focused'  # Git does not ignore a seat's evidence folder
    nested = ('import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; '
              'from ouroboros.safe_test_supervisor import main; '
              'raise SystemExit(main(Path(sys.argv[2]), Path(sys.argv[3]), [sys.executable, "-c", "print(1)"]))')
    outer = []
    for attempt in range(2):
        root, inner_root = tmp_path / f'scratch-{attempt}', tmp_path / f'nested-{attempt}'
        root.mkdir()
        inner_root.mkdir()
        # The nested owner inherits the outer run as its evidence parent, inside the checkout.
        assert main(root, source, [sys.executable, '-c', nested, str(REPO), str(inner_root), str(source)],
                    evidence_parent=parent) == 0
        outer.append(Path(next(line.split(' ', 1)[1] for line in capsys.readouterr().err.splitlines()
                               if line.startswith('SAFE_TEST_EVIDENCE '))))
    recipes = sorted(parent.rglob('source/source.json'))
    assert len(recipes) == 4, recipes  # two outer runs, each with one nested run; no copied recipe
    for recipe in recipes:
        record = json.loads(recipe.read_bytes())
        assert [item['path'] for item in record['untracked']] == ['ordinary.bin'], record
        assert (record['source_snapshot'], record['gaps']) == ('complete', [])
        assert (recipe.parent / 'untracked/ordinary.bin').read_bytes() == b'ordinary\x00input'
        assert not (recipe.parent / 'untracked/.review-evidence').exists()
    runs = [run.relative_to(source).as_posix() for run in outer]
    assert json.loads((outer[0] / 'source/source.json').read_bytes())['excluded_outputs'] == runs[:1]
    assert json.loads((outer[1] / 'source/source.json').read_bytes())['excluded_outputs'] == sorted(runs)
    [inner] = [path.parent.parent for path in outer[1].glob('run-*/source/source.json')]
    assert json.loads((inner / 'source/source.json').read_bytes())['excluded_outputs'] == sorted(runs)
    assert json.loads((inner / 'result.json').read_bytes())['returncode'] == 0


@pytest.mark.parametrize('fault', ['input', 'recipe'])
def test_a_source_gap_is_named_the_command_runs_and_a_disposable_source_stays(tmp_path, monkeypatch, capsys, fault):
    import shutil

    from ouroboros import safe_test_supervisor, test_evidence
    from ouroboros.test_environment import RETENTION_MARKER, declare_scratch
    from tests._shared import retire_synthetic_retention

    envelope = tmp_path / 'envelope'
    envelope.mkdir()
    declare_scratch(envelope, 'pytest-basetemp')  # the source lives in another owner's disposable scratch
    source = _repo(envelope / 'source', 'def test_ok():\n    assert True\n')
    (source / 'kept.bin').write_bytes(b'kept')
    (source / 'vanishing.bin').write_bytes(b'gone')
    if fault == 'input':
        copy = shutil.copy2

        def vanishing(src, dst, **kwargs):
            if Path(src).name == 'vanishing.bin':
                raise FileNotFoundError(2, 'vanished during capture', str(src))
            return copy(src, dst, **kwargs)
        monkeypatch.setattr(test_evidence.shutil, 'copy2', vanishing)
    else:
        def too_long(*_args):
            raise OSError(63, 'File name too long')
        monkeypatch.setattr(safe_test_supervisor, 'source_evidence', too_long)
    root = tmp_path / 'scratch'
    root.mkdir()
    reasons = []
    try:
        assert safe_test_supervisor.main(root, source, [sys.executable, '-c', 'print("GAP_RAN"); raise SystemExit(3)'],
                                         evidence_parent=diagnostic_directory(tmp_path / 'evidence', 'gap-' + fault)) == 3
        err = capsys.readouterr().err
        run = Path(next(line.split(' ', 1)[1] for line in err.splitlines() if line.startswith('SAFE_TEST_EVIDENCE ')))
        result = json.loads((run / 'result.json').read_bytes())
        reasons.append('test source evidence incomplete: ' + result['source_gap'])
        assert b'GAP_RAN' in next(run.glob('command-*/stdout.bin')).read_bytes()
        assert (result['returncode'], result['execution'], result['containment_error']) == (3, 'executed', '')
        assert result['source_gap'] and 'TEST_SOURCE_INCOMPLETE' in err
        record = json.loads((run / 'source/source.json').read_bytes())
        assert record['source_snapshot'] == 'incomplete', record
        if fault == 'input':
            assert [item['path'] for item in record['gaps']] == ['vanishing.bin'] and 'vanishing.bin' in result['source_gap']
            assert [item['path'] for item in record['untracked']] == ['kept.bin']
        else:
            assert 'File name too long' in record['error'] and 'untracked' not in record
        # Reproducible scratch still goes; the uncaptured source stays with its owner.
        assert not root.exists() and result['retention_reason'] == ''
        assert (envelope / RETENTION_MARKER).read_text().splitlines() == reasons
    finally:
        retire_synthetic_retention(envelope, *reasons)


@pytest.mark.parametrize('red', [False, True])
def test_contributor_preflight_evidence_survives_real_checkout_cleanup(tmp_path, monkeypatch, red):
    import scripts.run_external_review as caller
    from ouroboros import commit_admission, preflight_runner
    from ouroboros.tools.registry import ToolContext

    repo = _repo(tmp_path / 'source', 'def test_probe():\n    print("CONTRIBUTOR_RAW")\n    assert ' + str(not red) + '\n')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    monkeypatch.setattr(caller, 'REPO', repo)
    drive = diagnostic_directory(tmp_path / 'review-drive', 'contributor')
    # External contributor runs do not inherit an outer managed test owner.
    monkeypatch.delenv('OUROBOROS_TEST_EVIDENCE_ROOT', raising=False)
    frozen_paths, evidence_paths = [], []

    def controlled_lane(ctx, **_kwargs):
        # Only the selected workload is substituted. Real caller, frozen checkout,
        # hermetic assembly, pytest, result export and checkout removal all run.
        frozen_paths.append(Path(ctx.repo_dir))
        result = preflight_runner.run_hermetic_pytest(ctx.repo_dir, ctx=ctx,
                    pytest_args=['tests', '-q', '-s'], timeout=60)
        evidence_paths.append(Path(ctx._preflight_test_evidence))
        return result

    monkeypatch.setattr(commit_admission, 'run_tests_preflight_with_proof', controlled_lane)
    _, refusal = caller._contributor_tests_preflight(ToolContext(repo_dir=repo, drive_root=drive),
        {'base_sha': head, 'head_sha': head}, review_drive_root=drive)
    assert bool(refusal) == red
    assert frozen_paths and all(not path.parent.exists() for path in frozen_paths)
    [evidence] = evidence_paths
    assert evidence.is_relative_to(drive)
    assert (evidence / 'result.json').is_file()
    assert (evidence / 'source/source.json').is_file()
    assert b'CONTRIBUTOR_RAW' in next(evidence.glob('pytest-*/stdout.bin')).read_bytes()
    assert json.loads((evidence / 'result.json').read_bytes())['cleanup'] == 'released'


@pytest.mark.parametrize('red', [False, True])
def test_operator_preflight_evidence_survives_real_checkout_cleanup(tmp_path, monkeypatch, red):
    import scripts.run_external_review as caller
    from ouroboros import preflight_runner
    from ouroboros.tools import git as git_mod
    from ouroboros.tools import review_helpers
    from tests import _contributor_packet_shared as shared
    from tests.test_external_review_operator_lane import _operator_fixture

    repo, _briefs = _operator_fixture(tmp_path, monkeypatch)
    (repo / 'tests').mkdir()
    (repo / 'tests/test_probe.py').write_text('def test_probe():\n    print("OPERATOR_RAW")\n    assert ' + str(not red) + '\n')
    shared.git(repo, 'add', 'tests/test_probe.py')
    drive = diagnostic_directory(tmp_path / 'review-drive', 'operator')
    monkeypatch.delenv('OUROBOROS_TEST_EVIDENCE_ROOT', raising=False)
    frozen_paths, evidence_paths = [], []

    def controlled_lane(ctx, **_kwargs):
        # As for the contributor lane, only the selected workload is substituted.
        frozen_paths.append(Path(ctx.repo_dir))
        result = preflight_runner.run_hermetic_pytest(ctx.repo_dir, ctx=ctx,
                    pytest_args=['tests', '-q', '-s'], timeout=60)
        evidence_paths.append(Path(ctx._preflight_test_evidence))
        return result

    monkeypatch.setattr(review_helpers, '_run_review_preflight_tests', controlled_lane)
    monkeypatch.setattr(git_mod, '_run_review_preflight_tests', controlled_lane)
    output = tmp_path / 'out'
    monkeypatch.setattr(caller.sys, 'argv', ['run_external_review.py', f'--output={output}',
                                             f'--drive-root={drive}', 'fix: one'])
    exit_code = caller.main()
    assert (exit_code != 0) == red, (output / 'outcome.json').read_text()
    assert frozen_paths and all(not path.parent.exists() for path in frozen_paths)
    [evidence] = evidence_paths
    assert evidence.is_relative_to(drive / 'test-evidence'), evidence
    assert (evidence / 'result.json').is_file() and (evidence / 'source/source.json').is_file()
    assert b'OPERATOR_RAW' in next(evidence.glob('pytest-*/stdout.bin')).read_bytes()
    assert json.loads((evidence / 'result.json').read_bytes())['cleanup'] == 'released'


@pytest.mark.parametrize('boundary', ['windows', 'birth', 'token', 'valid'])
def test_interrupt_root_keeps_identity_and_platform_guards(monkeypatch, boundary):
    from ouroboros import process_containment as module

    owner = module.ProcessContainer()
    owner._root, owner._birth = 987654, 'measured-birth'
    calls = []
    monkeypatch.setattr(module._pl, 'IS_WINDOWS', boundary == 'windows')
    monkeypatch.setattr(module._pl, 'process_start_time',
                        lambda _pid: 'reused-pid' if boundary == 'birth' else owner._birth)
    monkeypatch.setattr(module, 'pid_marker_state',
                        lambda *_: module.MARKER_UNREADABLE if boundary == 'token' else module.MARKER_MEMBER)
    monkeypatch.setattr(module._pl, 'signal_pid', lambda *args: calls.append(args) or True)
    assert owner.interrupt_root() == (boundary == 'valid')
    assert calls == ([(owner._root, signal.SIGINT)] if boundary == 'valid' else [])


def test_uncaptured_source_retention_failure_uses_enclosing_declared_owner(tmp_path, monkeypatch):
    from ouroboros import safe_test_supervisor, test_environment, test_evidence
    from tests._shared import retire_synthetic_retention

    outer, inner = tmp_path / 'outer', tmp_path / 'outer/inner'
    inner.mkdir(parents=True)
    test_environment.declare_scratch(outer, 'pytest-session')
    test_environment.declare_scratch(inner, 'pytest-basetemp')
    source = inner / 'source'
    source.mkdir()
    (source / 'unique').write_bytes(b'uncaptured input')
    retain = test_environment.retain_tree
    monkeypatch.setattr(test_environment, 'retain_tree',
                        lambda root, reason: False if Path(root) == inner else retain(root, reason))
    reason = 'test source evidence incomplete: controlled capture gap'
    try:
        test_evidence.retain_uncaptured_source(source, 'controlled capture gap')
        result = safe_test_supervisor.release_scratch(outer)
        assert result['released'] == [] and result['retained'] == [str(outer)]
        assert (source / 'unique').read_bytes() == b'uncaptured input'
        assert (outer / test_environment.RETENTION_MARKER).read_text().splitlines() == [reason]
    finally:
        retire_synthetic_retention(outer, reason)


def test_unwritable_source_retention_is_not_reported_as_preserved(tmp_path, monkeypatch, capsys):
    from ouroboros import safe_test_supervisor, test_environment
    from tests._shared import retire_synthetic_retention

    source, root = tmp_path / 'source', tmp_path / 'scratch'
    source.mkdir()
    root.mkdir()
    test_environment.declare_scratch(source, 'pytest-basetemp')
    monkeypatch.delenv('OUROBOROS_TEST_TEMP_ROOT', raising=False)
    def capture_gap(*_args):
        raise OSError('controlled capture failure')
    monkeypatch.setattr(safe_test_supervisor, 'source_evidence', capture_gap)
    retain = test_environment.retain_tree
    monkeypatch.setattr(test_environment, 'retain_tree', lambda *_args: False)
    # This owner's own root remains writable; only source-owner markers fail.
    monkeypatch.setattr(safe_test_supervisor, 'retain_tree', retain)
    evidence = diagnostic_directory(tmp_path / 'evidence', 'retention-failure')
    try:
        assert safe_test_supervisor.main(root, source, [sys.executable, '-c', 'print("MUST_NOT_RUN")'],
                                         evidence_parent=evidence) == 2
        [run] = list(evidence.glob('run-*'))
        result = json.loads((run / 'result.json').read_bytes())
        assert result['execution'] == 'not_run'
        assert 'cannot retain uncaptured source' in result['evidence_error']
        assert str(source) in result['evidence_error']
        assert root.exists() and source.exists()
        assert 'cannot retain uncaptured source' in capsys.readouterr().err
        assert not list(run.glob('command-*'))
    finally:
        retire_synthetic_retention(root, 'command not started')


def test_source_can_be_the_declared_scratch_root(tmp_path, monkeypatch):
    from ouroboros import test_environment, test_evidence
    source = tmp_path / 'source'
    source.mkdir()
    test_environment.declare_scratch(source, 'pytest-basetemp')
    monkeypatch.delenv('OUROBOROS_TEST_TEMP_ROOT', raising=False)
    assert test_evidence._disposable_owner(source) == source


def test_listed_input_disappearing_before_stat_is_a_named_gap(tmp_path, monkeypatch):
    from ouroboros import test_evidence
    source = _repo(tmp_path / 'source', 'def test_ok():\n    assert True\n')
    vanished = source / 'vanishing.bin'
    vanished.write_bytes(b'listed by git')
    original_stat = Path.stat
    def disappearing(path, *args, **kwargs):
        if path == vanished:
            raise FileNotFoundError(2, 'disappeared after listing', str(path))
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', disappearing)
    record = test_evidence.source_evidence(source, tmp_path / 'evidence')
    assert record['source_snapshot'] == 'incomplete'
    assert [gap['path'] for gap in record['gaps']] == ['vanishing.bin']
