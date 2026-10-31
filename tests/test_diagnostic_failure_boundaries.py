"""#1908: diagnostic failures cannot erase an already observed exit/reap."""
from __future__ import annotations

import errno
import json
import sys
from pathlib import Path

import pytest

from ouroboros.test_environment import RETENTION_MARKER


def test_native_relative_output_is_retained_at_the_child_cwd(tmp_path, monkeypatch, capsys):
    from ouroboros.preflight_runner import run_hermetic_pytest
    from ouroboros.test_evidence import diagnostic_directory
    from tests.test_test_storage import _repo

    repo = _repo(tmp_path / "repo", """import os
from pathlib import Path

def test_probe():
    out = Path(os.environ['OUROBOROS_BROWSER_EVIDENCE_OUT'])
    out.mkdir()
    (out / 'result.txt').write_text('relative diagnostic')
""")
    monkeypatch.setenv("OUROBOROS_BROWSER_EVIDENCE_OUT", "relative-shots")
    assert run_hermetic_pytest(repo, pytest_args=["tests", "-q"], timeout=60) is None
    evidence = Path(next(line.split(" ", 1)[1] for line in capsys.readouterr().err.splitlines()
                         if line.startswith("PREFLIGHT_EVIDENCE ")))
    record = json.loads((evidence / "result.json").read_text())
    root = Path(record["scratch"])
    try:
        diagnostic = root / "repo" / "relative-shots" / "result.txt"
        assert diagnostic.read_text() == "relative diagnostic"
        assert record["cleanup"] == "retained" and record["containment_error"] == ""
        saved = diagnostic_directory(tmp_path / "proof", "relative-output") / "result.txt"
        saved.write_bytes(diagnostic.read_bytes())
    finally:
        # This tiny real pass has a recorded clean reap. The saved diagnostic,
        # source and logs survive; only this test's new scratch may be retired.
        if root.exists() and record["containment_error"] == "":
            from ouroboros.test_evidence import remove_scratch_tree
            remove_scratch_tree(root)


@pytest.mark.parametrize("lane", ["pytest", "node"])
def test_result_and_stderr_write_failures_keep_the_real_lane_outcome(tmp_path, monkeypatch, lane):
    from ouroboros import preflight_node, preflight_runner, test_evidence

    repo, root = tmp_path / "repo", tmp_path / "run"
    repo.mkdir()
    root.mkdir()
    if lane == "pytest":
        (repo / "test_probe.py").write_text("def test_probe():\n    print('real pytest output')\n")
    else:
        tests = repo / "web" / "tests"
        tests.mkdir(parents=True)
        (tests / "probe.test.js").write_text("const test = require('node:test'); test('real node', () => {});\n")
    real_write = test_evidence.write_record
    state = {"full": False}
    original_stderr = sys.stderr

    class FillingStderr:
        def write(self, text):
            if state["full"]:
                raise OSError(errno.ENOSPC, "diagnostic volume full")
            return original_stderr.write(text)

        def flush(self):
            if state["full"]:
                raise OSError(errno.ENOSPC, "diagnostic volume full")
            return original_stderr.flush()

    def write_record(path, value):
        if Path(path).name == "result.json":
            state["full"] = True
            raise OSError(errno.ENOSPC, "result volume full")
        return real_write(path, value)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(test_evidence, "write_record", write_record)
            patch.setattr(sys, "stderr", FillingStderr())
            if lane == "pytest":
                result = preflight_runner._execute_pytest_pass(sys.executable, repo, root,
                    ["-o", "addopts=", "-q", "test_probe.py"], 30)
                code, output, reap = result
                errors = result.evidence_errors
                assert "1 passed" in output
            else:
                result = preflight_node.run_node_tests(repo, root, 30, 8000)
                code, reap, errors = result["returncode"], result["reap_error"], result["evidence_errors"]
                assert result["error"] is None and result["execution"] == "executed"
        assert code == 0 and reap == ""
        assert any("result volume full" in error for error in errors)
        assert (root / RETENTION_MARKER).is_file()
    finally:
        # The real lane returned a clean reap; this fixture's injected marker is retired.
        (root / RETENTION_MARKER).unlink(missing_ok=True)


def test_gate_record_and_stderr_failure_still_releases_with_a_retention_reason(tmp_path, monkeypatch):
    from ouroboros import test_evidence
    root = tmp_path / "scratch"
    root.mkdir()
    releases = []

    class FullStderr:
        def write(self, _text):
            raise OSError(errno.ENOSPC, "stderr volume full")
        def flush(self):
            raise OSError(errno.ENOSPC, "stderr volume full")

    def fail_record(*_args):
        raise OSError(errno.ENOSPC, "record volume full")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(test_evidence, "write_record", fail_record)
            patch.setattr(sys, "stderr", FullStderr())
            evidence = {"path": tmp_path, "cleanup": (None, root, None, None, ""), "evidence_errors": []}
            test_evidence.finish_gate(evidence, None, "pre_commit", lambda *args: releases.append(args))
        assert releases and "gate evidence write failed" in releases[0][-1]
        assert evidence["evidence_errors"]
        assert (root / RETENTION_MARKER).is_file()
    finally:
        (root / RETENTION_MARKER).unlink(missing_ok=True)


def test_standalone_record_and_stderr_failure_keeps_command_status(tmp_path, monkeypatch):
    from ouroboros import safe_test_supervisor, test_evidence
    from ouroboros.test_environment import isolated_environment
    from tests.test_test_storage import REPO

    root = tmp_path / "run"
    isolated_environment(root, REPO)
    original_write, original_stderr = test_evidence.write_record, sys.stderr
    state, records = {"full": False}, []

    class FillingStderr:
        def write(self, text):
            if state["full"]:
                raise OSError(errno.ENOSPC, "stderr volume full")
            return original_stderr.write(text)
        def flush(self):
            if state["full"]:
                raise OSError(errno.ENOSPC, "stderr volume full")
            return original_stderr.flush()

    def fail_result(path, value):
        if Path(path).name == "result.json":
            state["full"] = True
            records.append(dict(value))
            raise OSError(errno.ENOSPC, "record volume full")
        return original_write(path, value)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stderr", FillingStderr())
            patch.setattr(test_evidence, "write_record", fail_result)
            patch.setattr(safe_test_supervisor, "write_record", fail_result)
            code = safe_test_supervisor.main(root, REPO,
                [sys.executable, "-c", "print('real command output'); raise SystemExit(7)"],
                evidence_parent=test_evidence.diagnostic_directory(tmp_path / "evidence", "standalone-fault"))
        assert code == 7
        assert records[-1]["returncode"] == 7 and records[-1]["containment_error"] == ""
        assert records[-1]["evidence_error"] and records[-1]["retention_reason"]
        assert (root / RETENTION_MARKER).is_file()
    finally:
        (root / RETENTION_MARKER).unlink(missing_ok=True)


@pytest.mark.parametrize("relative,inside", [("../../node-shots", True), ("../../../outside", False)])
def test_node_relative_destination_uses_web_cwd(tmp_path, monkeypatch, relative, inside):
    from ouroboros.preflight_runner import _preflight_env

    root = tmp_path / "native"
    worktree = root / "repo"
    monkeypatch.setenv("OUROBOROS_BROWSER_EVIDENCE_OUT", relative)
    try:
        env = _preflight_env(root, worktree, cwd=worktree / "web")
        assert env["OUROBOROS_BROWSER_EVIDENCE_OUT"] == relative
        assert (root / RETENTION_MARKER).exists() is inside
    finally:
        (root / RETENTION_MARKER).unlink(missing_ok=True)


def test_live_display_failure_keeps_status_and_the_complete_raw_streams(tmp_path, monkeypatch):
    from ouroboros import safe_test_supervisor
    from ouroboros.test_environment import isolated_environment
    from ouroboros.test_evidence import diagnostic_directory
    from tests._shared import retire_synthetic_retention

    root, checkout = tmp_path / "run", tmp_path / "checkout"
    checkout.mkdir()
    isolated_environment(root, checkout)
    full = OSError(errno.ENOSPC, "display volume full")

    class FullDisplay:
        class buffer:
            @staticmethod
            def write(_data):
                raise full

            @staticmethod
            def flush():
                pass

        def write(self, text):
            return len(text)

        def flush(self):
            pass

    parent = diagnostic_directory(tmp_path / "evidence", "display-fault")
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(root))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stdout", FullDisplay())
            code = safe_test_supervisor.main(root, checkout, [sys.executable, "-c",
                "print('D' * 70_000 + 'DISPLAY_TAIL'); raise SystemExit(3)"], evidence_parent=parent)
        [run] = [path for path in parent.glob("run-*") if (path / "result.json").is_file()]
        record = json.loads((run / "result.json").read_text())
        assert code == 3 and record["returncode"] == 3 and record["containment_error"] == ""
        assert "display" in record["evidence_error"] and "display" in record["retention_reason"]
        assert b"D" * 70_000 + b"DISPLAY_TAIL" in next(run.glob("command-*/stdout.bin")).read_bytes()
        assert (root / RETENTION_MARKER).is_file()
    finally:
        # The display fault was injected; the command was reaped cleanly above.
        retire_synthetic_retention(root, f"test evidence incomplete: display: {full}", f"display: {full}")
