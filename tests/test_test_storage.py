"""Real consumers of the #1908 result/scratch separation (not a disk benchmark)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ouroboros.test_evidence import diagnostic_directory
from ouroboros.test_environment import declare_scratch, retain_tree

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.serial


def _safe(tmp_path, command, *, keep=False, evidence=None):
    evidence = evidence or diagnostic_directory(tmp_path / "evidence", "storage-" + tmp_path.name)
    result = subprocess.run([sys.executable, "-I", "-S", str(REPO / "scripts/safe_test.py"),
                             "--temp-parent", "/tmp" if os.name == "posix" else str(tmp_path),
                             "--evidence-dir", str(evidence), *(["--keep"] if keep else []),
                             "--", *command], cwd=REPO, capture_output=True, text=True, timeout=120)
    paths = [Path(line.split(" ", 1)[1]) for line in result.stderr.splitlines()
             if line.startswith("SAFE_TEST_EVIDENCE ")]
    return result, paths[-1] if paths else None


@pytest.mark.parametrize("red", [False, True])
def test_known_pytest_scratch_released_with_complete_raw_output_and_diagnostics(tmp_path, red):
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "test_probe.py").write_text('''import os, pathlib, sys
from ouroboros.test_evidence import diagnostic_directory

def test_probe(tmp_path):
    (tmp_path / 'reproducible.bin').write_bytes(b'x' * 3_000_000)
    out = diagnostic_directory(tmp_path / 'evidence', 'declared')
    (out / 'screenshot.png').write_bytes(b'fake-png-diagnostic')
    (out / 'state.json').write_text('{"at":"assertion"}')
    print('STDOUT_HEAD' + 'o' * 100_000 + 'STDOUT_TAIL', flush=True)
    sys.stderr.write('STDERR_HEAD' + 'e' * 100_000 + 'STDERR_TAIL\\n')
    assert ''' + str(not red) + "\n", encoding="utf-8")
    args = [sys.executable, "-m", "pytest", "-p", "tests.conftest", "--rootdir", str(probe),
            "--confcutdir", str(probe), "-o", "addopts=", "-q", "-s", str(probe)]
    result, evidence = _safe(tmp_path, args)
    assert result.returncode == int(red), result.stderr
    record = json.loads((evidence / "result.json").read_text())
    assert not Path(record["scratch"]).exists(), record
    assert record["cleanup"]["released"] and record["containment_error"] == ""
    raw = next(evidence.glob("command-*/stdout.bin")).read_bytes()
    err = next(evidence.glob("command-*/stderr.bin")).read_bytes()
    assert b"STDOUT_HEAD" + b"o" * 100_000 + b"STDOUT_TAIL" in raw
    assert b"STDERR_HEAD" + b"e" * 100_000 + b"STDERR_TAIL" in err
    assert next(evidence.glob("declared/*/screenshot.png")).read_bytes() == b"fake-png-diagnostic"
    assert json.loads(next(evidence.glob("declared/*/state.json")).read_text()) == {"at": "assertion"}
    assert (evidence / "source/source.json").is_file()
    assert list(evidence.glob("pytest-*/results.json")), "post-test CI consumer lost its projection"


@pytest.mark.parametrize("keep", [False, True])
def test_arbitrary_unique_command_content_and_debug_are_not_declared_garbage(tmp_path, keep):
    code = "import pathlib,os; pathlib.Path(os.environ['HOME'],'unique').write_text('work'); raise SystemExit(7)"
    result, evidence = _safe(tmp_path, [sys.executable, "-c", code], keep=keep)
    assert result.returncode == 7
    record = json.loads((evidence / "result.json").read_text())
    root = Path(record["scratch"])
    assert (root / "home/unique").read_text() == "work"
    assert not record["containment_error"]
    assert (record["retention_reason"] == "debug retention") == keep
    # These are this test's new ended roots; preserve the needed unique bytes
    # before explicitly discarding our synthetic input, never scan old roots.
    (evidence / "unique.txt").write_bytes((root / "home/unique").read_bytes())
    import shutil
    shutil.rmtree(root)


def test_explicit_env_keeps_outer_containment_membership(monkeypatch):
    from tests.conftest import _isolated_child_env
    # Upper case: Windows stores every environment name upper-cased.
    monkeypatch.setenv("OURO_PROC_CONTAINER_OUTER_PROBE", "1")
    assert _isolated_child_env({})["OURO_PROC_CONTAINER_OUTER_PROBE"] == "1"
    assert _isolated_child_env({"OUROBOROS_TEST_EVIDENCE_ROOT": "/selected"})["OUROBOROS_TEST_EVIDENCE_ROOT"] == "/selected"


def test_controller_signatures_survive_a_bounded_preview(tmp_path):
    from ouroboros.preflight_runner import _controller_diagnostics, _classify_pass_result
    from ouroboros.test_evidence import preview_file
    line = b"[gw2] pytest-timeout callback invoked (300s) while running 't.py::test_x'; worker subsequently lost\n"
    (tmp_path / 'stdout.bin').write_bytes(b'H' * 300_000 + b'\n[gw2] node down: crashed\n' + line + b'T' * 300_000)
    (tmp_path / 'stderr.bin').write_bytes(b'+++++ Timeout +++++\n')
    preview = preview_file(tmp_path / 'stdout.bin')
    assert '[gw2]' not in preview
    output = _controller_diagnostics(tmp_path) + preview
    verdict = _classify_pass_result('parallel', 1, output, 8000, parallel=True)
    # The crash label and the controller's readable timeout line survive the cut. Neither that line nor a
    # banner names the cause: only the pass's own recorded crash rows do (tests/test_preflight_runner.py).
    assert 'PARALLEL_WORKER_CRASH' in verdict and 'callback invoked (300s)' in verdict
    assert 'nothing recorded why' in verdict and 'Do NOT mark' not in verdict


def test_retention_unknown_siblings_and_symlinks_are_not_released(tmp_path):
    from ouroboros.safe_test_supervisor import release_scratch
    root = tmp_path / "owned"
    known, unknown = root / "known", root / "unknown"
    known.mkdir(parents=True)
    unknown.mkdir()
    declare_scratch(known, "pytest-session")
    (unknown / "work").write_text("keep")
    retain_tree(known, "unconfirmed child")
    try:
        assert release_scratch(root)["retained"] == [str(root)]
        assert known.is_dir() and (unknown / "work").is_file()
    finally:
        # This marker models an unknown process; the fixture spawned none.
        (known / 'OUROBOROS_RETAINED.txt').unlink()


def _repo(root, source):
    root.mkdir()
    (root / "tests").mkdir()
    (root / "tests/test_probe.py").write_text(source, encoding="utf-8")
    for args in (["init"], ["add", "."], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                                             "commit", "-qm", "fixture"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


@pytest.mark.parametrize("red", [False, True])
def test_native_ctxless_green_red_keep_full_source_and_logs(tmp_path, capsys, red):
    from ouroboros.preflight_runner import run_hermetic_pytest
    repo = _repo(tmp_path / "repo", "def test_probe():\n    print('NATIVE_HEAD' + 'x' * 100_000 + 'NATIVE_TAIL')\n    assert " + str(not red) + "\n")
    (repo / "admitted-input.bin").write_bytes(b'untracked\x00exact')
    # A second dirty tracked version must survive, not merely its initial HEAD.
    with (repo / "tests/test_probe.py").open("a") as stream:
        stream.write("\n# dirty input\n")
    result = run_hermetic_pytest(repo, pytest_args=["tests", "-q", "-s"], timeout=60, max_output=300)
    assert (result is not None) == red, result
    captured = capsys.readouterr()
    evidence = Path(next(line.split(" ", 1)[1] for line in captured.err.splitlines()
                         if line.startswith("PREFLIGHT_EVIDENCE ")))
    record = json.loads((evidence / "result.json").read_text())
    assert not Path(record["scratch"]).exists()
    assert b"NATIVE_HEAD" + b"x" * 100_000 + b"NATIVE_TAIL" in next(evidence.glob("pytest-*/stdout.bin")).read_bytes()
    assert (evidence / "source/untracked/admitted-input.bin").read_bytes() == b'untracked\x00exact'
    assert b"dirty input" in (evidence / "source/candidate.diff").read_bytes()


def test_native_timeout_preserves_obtained_output(tmp_path, capsys):
    from ouroboros.preflight_runner import run_hermetic_pytest
    repo = _repo(tmp_path / "repo", "import time\ndef test_hang():\n    print('LAST_TEST_BEFORE_TIMEOUT', flush=True)\n    time.sleep(120)\n")
    result = run_hermetic_pytest(repo, pytest_args=["tests", "-q", "-s"], timeout=2)
    assert "timed out" in result and "LAST_TEST_BEFORE_TIMEOUT" in result
    evidence = Path(next(line.split(" ", 1)[1] for line in capsys.readouterr().err.splitlines()
                         if line.startswith("PREFLIGHT_EVIDENCE ")))
    assert b"LAST_TEST_BEFORE_TIMEOUT" in next(evidence.glob("pytest-*/stdout.bin")).read_bytes()
    assert json.loads(next(evidence.glob("pytest-*/result.json")).read_text())["returncode"] is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX detached child/stdio witness")
def test_direct_exit_reaps_descendant_before_log_read_and_release(tmp_path):
    marker = diagnostic_directory(tmp_path, "storage-descendant") / "child.pid"
    code = ("import os,subprocess,sys; subprocess.Popen([sys.executable,'-c',"
            f"\"import os,time;open({str(marker)!r},'w').write(str(os.getpid()));time.sleep(120)\"]); "
            "print('DIRECT_EXIT',flush=True)")
    started = time.monotonic()
    result, evidence = _safe(tmp_path, [sys.executable, "-c", code])
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 30, "waited for pipe EOF instead of direct process exit/reap"
    record = json.loads((evidence / "result.json").read_text())
    assert record["containment_error"] == "" and not Path(record["scratch"]).exists()
    if marker.exists():
        from ouroboros.platform_layer import pid_is_alive
        from ouroboros.process_containment import pid_is_zombie
        pid = int(marker.read_text())
        assert not pid_is_alive(pid) or pid_is_zombie(pid)


def test_evidence_failure_retains_source_without_changing_test_verdict(tmp_path, monkeypatch, capsys):
    from ouroboros import preflight_runner, test_evidence
    repo = _repo(tmp_path / "repo", "def test_probe():\n    assert True\n")
    original = test_evidence.write_record
    def broken(path, value):
        if Path(path).name == 'result.json' and 'verdict' in value:
            raise OSError('controlled export failure')
        return original(path, value)
    monkeypatch.setattr(test_evidence, "write_record", broken)
    assert preflight_runner.run_hermetic_pytest(repo, pytest_args=["tests", "-q"], timeout=60) is None
    text = capsys.readouterr().err
    evidence = Path(next(line.split(" ", 1)[1] for line in text.splitlines() if line.startswith("PREFLIGHT_EVIDENCE ")))
    assert evidence.is_dir(), "the observed execution's remaining diagnostics stay addressable"
    assert 'PREFLIGHT_EVIDENCE_INCOMPLETE' in text
    # The retained worktree is discoverable through Git as well as the marker.
    worktrees = subprocess.check_output(['git', 'worktree', 'list', '--porcelain'], cwd=repo, text=True)
    # Git prints C:/... on Windows: compare the resolved paths, never the spellings.
    scratch = [path for path in (Path(line.removeprefix('worktree ')) for line in worktrees.splitlines()
                                 if line.startswith('worktree ')) if path.resolve() != repo.resolve()]
    assert len(scratch) == 1 and (scratch[0].parent / 'OUROBOROS_RETAINED.txt').is_file()
    # Synthetic negative cleanup after asserting process outcome and preserving
    # its complete source/logs. This test alone created this exact registration.
    subprocess.run(['git', 'worktree', 'remove', '--force', str(scratch[0])], cwd=repo, check=True)
    import shutil
    shutil.rmtree(scratch[0].parent)


@pytest.mark.parametrize('module', ['tests.fixtures_e2e_cancellation', 'tests.system_e2e.harness'])
def test_fixture_transport_clones_keep_history_refs_and_independent_objects(tmp_path, monkeypatch, module):
    import importlib
    producer = importlib.import_module(module)
    source = _repo(tmp_path / 'source', 'def test_ok():\n    assert True\n')
    def git(root, *args):
        return subprocess.check_output(['git', *args], cwd=root).decode().strip()
    first = git(source, 'rev-parse', 'HEAD')
    git(source, 'tag', 'fixture-v1', first)
    git(source, 'branch', 'other-history', first)
    for index in range(3):
        (source / 'probe.txt').write_text(str(index))
        git(source, 'add', '.')
        git(source, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '-qm', f'fixture-{index}')
    expected = git(source, 'rev-list', 'HEAD')
    destination = tmp_path / 'destination'
    destination.mkdir()
    monkeypatch.setattr(producer, 'REPO_ROOT', source)
    clone = producer.clone_repo(destination)
    assert git(clone, 'rev-list', 'HEAD') == expected
    assert git(clone, 'rev-parse', 'refs/tags/fixture-v1') == first
    assert git(clone, 'rev-parse', 'HEAD~3') == first
    assert not (clone / '.git/objects/info/alternates').exists()
    assert (clone / '.git/shallow').exists() is False
    # This existing fixture deliberately removes origin and its remote refs.
    assert git(clone, 'for-each-ref', 'refs/remotes') == ''
    moved = tmp_path / 'source-not-available'
    source.rename(moved)
    try:
        assert git(clone, 'rev-list', 'HEAD') == expected
        git(clone, 'reset', '--hard', first)
        assert git(clone, 'rev-parse', 'HEAD') == first
    finally:
        moved.rename(source)


def test_failed_containment_setup_never_reexecutes_the_command(tmp_path, monkeypatch):
    from ouroboros import safe_test_supervisor as owner
    root = tmp_path / 'new-boundary'
    root.mkdir()
    calls = []
    class BrokenContainer:
        def spawn(self, command, **kwargs):
            calls.append('spawn')
            raise OSError('setup failed after possible creation')
        def reap(self):
            calls.append('reap')
            return 'setup outcome unknown'
        def close(self):
            calls.append('close')
    monkeypatch.setattr(owner, 'ProcessContainer', lambda **_options: BrokenContainer())
    try:
        assert owner.main(root, REPO, [sys.executable, '-c', 'raise RuntimeError("must not run")']) == 2
        assert calls == ['spawn', 'reap', 'close']
        assert (root / 'OUROBOROS_RETAINED.txt').is_file()
    finally:
        # BrokenContainer is a no-spawn synthetic owner, not a real leftover.
        (root / 'OUROBOROS_RETAINED.txt').unlink(missing_ok=True)


def test_stdlib_launcher_refuses_pending_adoption_before_package_import(tmp_path):
    import shutil
    repo = tmp_path / 'cold-copy'
    (repo / 'scripts').mkdir(parents=True)
    (repo / 'ouroboros').mkdir()
    (repo / '.git').mkdir()
    shutil.copyfile(REPO / 'scripts/safe_test.py', repo / 'scripts/safe_test.py')
    shutil.copyfile(REPO / 'ouroboros/test_environment.py', repo / 'ouroboros/test_environment.py')
    # The stdlib helper parses defaults only during execution, not this refusal.
    (repo / '.git/ouroboros-body-adoption').write_text('not-a-real-switch')
    marker = tmp_path / 'import-ran'
    (repo / 'ouroboros/__init__.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("ran")\n')
    result = subprocess.run([sys.executable, '-I', '-S', str(repo / 'scripts/safe_test.py'),
                             '--temp-parent', '/tmp' if os.name == 'posix' else str(tmp_path),
                             '--', sys.executable, '-c', 'print("must-not-run")'], capture_output=True, text=True)
    assert result.returncode == 2 and 'pending body adoption' in result.stderr
    assert not marker.exists() and not list((repo / 'ouroboros').glob('__pycache__/*'))


def test_nested_safe_test_retains_inner_evidence_after_outer_release(tmp_path):
    probe = tmp_path / 'probe'
    probe.mkdir()
    temp_parent = '/tmp' if os.name == 'posix' else str(tmp_path)
    (probe / 'test_nested.py').write_text(f'''import os, subprocess, sys
from pathlib import Path

def test_nested():
    run = subprocess.run([sys.executable, '-I', '-S', {str(REPO / 'scripts/safe_test.py')!r},
        '--temp-parent', {temp_parent!r}, '--', sys.executable, '-c', 'print("INNER_EVIDENCE_TAIL")'],
        capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    print(run.stderr, flush=True)
''')
    result, evidence = _safe(tmp_path, [sys.executable, '-m', 'pytest', '-p', 'tests.conftest',
        '--rootdir', str(probe), '--confcutdir', str(probe), '-o', 'addopts=', '-q', '-s', str(probe)])
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((evidence / 'result.json').read_text())
    assert not Path(record['scratch']).exists()
    inner = [path for path in evidence.glob('run-*') if (path / 'result.json').is_file()]
    assert len(inner) == 1
    assert b'INNER_EVIDENCE_TAIL' in next(inner[0].glob('command-*/stdout.bin')).read_bytes()
    assert not Path(json.loads((inner[0] / 'result.json').read_text())['scratch']).exists()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX SIGKILL interruption witness')
@pytest.mark.parametrize('stage', ['launcher', 'supervisor'])
def test_hard_interruption_retains_scratch_and_partial_evidence(tmp_path, stage):
    from ouroboros.process_containment import ProcessContainer, MARKER_MEMBER, pid_marker_state
    from ouroboros.platform_layer import force_kill_pid
    import shutil
    evidence_parent = diagnostic_directory(tmp_path / 'evidence', 'interrupted-' + tmp_path.name)
    output = (evidence_parent / 'launcher.log').open('wb')
    container = ProcessContainer()
    command = [sys.executable, '-I', '-S', str(REPO / 'scripts/safe_test.py'),
        '--temp-parent', '/tmp', '--evidence-dir', str(evidence_parent), '--', sys.executable,
        '-c', 'import time; print("INTERRUPT_STARTED", flush=True); time.sleep(120)']
    proc = container.spawn(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
    root = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            owners = list(evidence_parent.glob('run-*/owner.json'))
            if owners:
                evidence = owners[0].parent
                logs = list(evidence.glob('command-*/stdout.bin'))
                if logs and b'INTERRUPT_STARTED' in logs[0].read_bytes():
                    break
            time.sleep(.02)
        else:
            pytest.fail('command did not reach the observed interruption point')
        owner = json.loads(owners[0].read_text())
        root = Path(owner['scratch'])
        pid = proc.pid if stage == 'launcher' else owner['supervisor_pid']
        assert pid_marker_state(pid, container._token) == MARKER_MEMBER
        force_kill_pid(pid)
        proc.wait(timeout=10)
        assert container.reap() == ''
        assert root.exists(), 'a hard-killed owner cannot attest completed cleanup'
        assert b'INTERRUPT_STARTED' in logs[0].read_bytes()
        assert not (evidence / 'result.json').exists(), 'do not fabricate a terminal result'
    finally:
        assert container.reap() == ''
        container.close()
        proc.wait(timeout=10)
        output.close()
        # Only this controlled test's root; full partial evidence already lives
        # outside it and its own measured container has now proven teardown.
        if root is not None:
            shutil.rmtree(root)


def test_unwritable_default_evidence_falls_back_but_an_explicit_destination_never_moves(tmp_path, monkeypatch, capsys):
    from ouroboros import safe_test_supervisor as owner
    from ouroboros.test_environment import RETENTION_MARKER
    from tests._shared import retire_synthetic_retention
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    # The real default parent (`<checkout>/../.test-evidence`) cannot become a directory.
    (tmp_path / '.test-evidence').write_text('a file occupies the default evidence parent')
    root, explicit_root = tmp_path / 'scratch', tmp_path / 'explicit-scratch'
    root.mkdir()
    explicit_root.mkdir()
    try:
        (tmp_path / '.test-evidence').resolve().mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # the exact fault the owner meets below
        fallback = f'default evidence parent unwritable ({exc}); evidence kept inside retained scratch'
    monkeypatch.delenv('OUROBOROS_TEST_EVIDENCE_ROOT', raising=False)
    monkeypatch.setenv('OUROBOROS_TEST_TEMP_ROOT', str(root))  # stage two's own root, as safe_test sets it
    try:
        assert owner.main(root, checkout, [sys.executable, '-c', 'print("FALLBACK_RAN"); raise SystemExit(5)']) == 5
        err = capsys.readouterr().err
        run = Path(next(line.split(' ', 1)[1] for line in err.splitlines() if line.startswith('SAFE_TEST_EVIDENCE ')))
        assert 'SAFE_TEST_EVIDENCE_FALLBACK default evidence parent unwritable' in err
        assert run.is_relative_to(root / 'evidence') and b'FALLBACK_RAN' in next(run.glob('command-*/stdout.bin')).read_bytes()
        record = json.loads((run / 'result.json').read_text())
        assert record['returncode'] == 5 and record['evidence_fallback'] and record['cleanup']['retained'] == [str(root)]
        assert (root / RETENTION_MARKER).is_file(), 'the only copy of the evidence lives in this scratch'
        # A selected destination that cannot be created is an error, never a silent redirect.
        ran = tmp_path / 'explicit-ran'
        monkeypatch.setenv('OUROBOROS_TEST_TEMP_ROOT', str(explicit_root))
        code = owner.main(explicit_root, checkout, [sys.executable, '-c', f'open({str(ran)!r}, "w")'],
                          evidence_parent=tmp_path / '.test-evidence' / 'selected')
        err = capsys.readouterr().err
        assert code == 2 and not ran.exists() and 'SAFE_TEST_EVIDENCE_FALLBACK' not in err
        assert 'SAFE_TEST_ERROR' in err and not (explicit_root / 'evidence').exists()
    finally:
        # Both faults were manufactured above and no command outlived its reap.
        retire_synthetic_retention(root, fallback, f"persistent test evidence selected inside scratch: {(root / 'evidence').resolve()}",
                                   f'retention marker or unreadable subtree: {root / RETENTION_MARKER}')
        retire_synthetic_retention(explicit_root, 'command not started')
