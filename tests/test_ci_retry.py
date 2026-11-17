"""scripts/ci_retry.py: one retry per failure, a first run for what a stopped session left, one verdict."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR
from ouroboros.test_parallelism import NESTED_WORKERS_ENV
from tests import ci_accepted
from tests.ci_evidence import reconcile_shards, render_annotations
from tests.ci_verdicts import render_run_verdict

REPO = Path(__file__).resolve().parents[1]
SHA, RUN = "c" * 40, "777"
PARALLEL = f"not exclusive and {LANE_EXCLUSION_EXPR}"
A, B, C, D = (f"tests/test_{name}.py::test_case" for name in "abcd")


@pytest.fixture(scope="module")
def ci_retry():
    spec = importlib.util.spec_from_file_location("ci_retry_script", REPO / "scripts" / "ci_retry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _round(evidence, k=1):
    return evidence / f"retry-{k}", evidence / f"retry-{k}" / "argsfile.txt"


# ---------------------------------------------------------------- the round's command


@pytest.mark.parametrize("pass_flags", [
    ["-n", "auto", "--dist", "loadgroup", "--max-worker-restart=0", "--timeout=300"], ["--timeout=300"],
], ids=["parallel", "serial"])
def test_the_ci_run_tests_form_keeps_every_option_but_the_lane_proof(ci_retry, tmp_path, pass_flags):
    """Exactly what ci.yml wraps; --timeout stays, the shard goes, the evidence moves into the round."""
    evidence = tmp_path / "ci-evidence" / "parallel"
    round_dir, argsfile = _round(evidence)
    command = ["python", "scripts/run_tests.py", "tests/", "-m", PARALLEL, *pass_flags, "-q", "--tb=short",
               f"--ci-evidence-dir={evidence}", "--default-suite-shard=3/8"]
    assert ci_retry.retry_command(command, argsfile=argsfile, direct=[], round_dir=round_dir) == [
        "python", "scripts/run_tests.py", f"@{argsfile}", "-m", PARALLEL, *pass_flags, "-q", "--tb=short",
        f"--ci-evidence-dir={round_dir}"]


def test_the_ui_lane_form_drops_its_lane_guard_and_keeps_its_junit_private(ci_retry, tmp_path):
    evidence, private = tmp_path / "ci-evidence" / "ui" / "host", tmp_path / "ci-private" / "ui"
    raw = tmp_path / "ci-raw" / "ui" / "host"
    round_dir, argsfile = _round(evidence, 2)
    command = ["python", "-I", "-S", "scripts/safe_test.py", "--evidence-dir", str(raw), "--", "python", "-m",
               "pytest", "tests/", "-m", "ui_browser", "--require-ui-browser", "-vv", "--tb=short", "--timeout=900",
               "--ui-browser-shard=5/8", "--session-timeout=4500", f"--junitxml={private / 'host.xml'}",
               f"--ci-evidence-dir={evidence}"]
    assert ci_retry.retry_command(command, argsfile=argsfile, direct=[], round_dir=round_dir) == [
        "python", "-I", "-S", "scripts/safe_test.py", "--evidence-dir", str(raw / "retry-2"), "--", "python", "-m",
        "pytest", f"@{argsfile}", "-m", "ui_browser", "-vv", "--tb=short", "--timeout=900",
        f"--junitxml={private / 'retry-2' / 'host.xml'}", f"--ci-evidence-dir={round_dir}"]


@pytest.mark.parametrize("attached", [False, True], ids=["separate", "attached"])
def test_a_launchers_raw_evidence_parent_moves_into_its_own_round_child(ci_retry, tmp_path, attached):
    """scripts/safe_test.py keeps each invocation's run directory beneath --evidence-dir. A round's go beneath that
    parent's retry-<k>, as its projection goes beneath DIR/retry-<k>: the relation is the relative path itself."""
    evidence, raw = tmp_path / "ci-evidence" / "ui" / "tools", tmp_path / "ci-raw" / "ui" / "tools"
    round_dir, argsfile = _round(evidence, 3)

    def spelled(path):
        return [f"--evidence-dir={path}"] if attached else ["--evidence-dir", str(path)]

    command = ["python", "-I", "-S", "scripts/safe_test.py", *spelled(raw), "--", "python", "-m", "pytest",
               "tests/test_browser_tools_smoke.py", f"--ci-evidence-dir={evidence}"]
    assert ci_retry.retry_command(command, argsfile=argsfile, direct=[], round_dir=round_dir) == [
        "python", "-I", "-S", "scripts/safe_test.py", *spelled(raw / "retry-3"), "--", "python", "-m", "pytest",
        f"@{argsfile}", f"--ci-evidence-dir={round_dir}"]


def test_separate_option_values_move_or_go_with_their_option(ci_retry, tmp_path):
    evidence = tmp_path / "parallel"
    round_dir, argsfile = _round(evidence)
    command = ["python", "-m", "pytest", "--ci-evidence-dir", str(evidence), "tests/", "--junitxml",
               str(Path("private") / "host.xml"), "--default-suite-shard", "2/4", "--session-timeout", "4500",
               "--ui-browser-shard", "1/8", "--default-suite-manifest", "--require-ui-browser",
               "-p", "no:cacheprovider", "--timeout", "300"]
    assert ci_retry.retry_command(command, argsfile=argsfile, direct=[], round_dir=round_dir) == [
        "python", "-m", "pytest", "--ci-evidence-dir", str(round_dir), f"@{argsfile}",
        "--junitxml", str(Path("private") / "retry-1" / "host.xml"), "-p", "no:cacheprovider", "--timeout", "300"]


def test_every_test_path_becomes_one_argument_file_where_the_first_stood(ci_retry, tmp_path):
    """journal-persist names its modules; an option's value is never a test path, whatever it reads."""
    evidence = tmp_path / "parallel"
    round_dir, argsfile = _round(evidence)
    command = ["python", "scripts/run_tests.py", "tests/test_store.py", "tests/test_store_import.py",
               "tests/test_store_switch.py", "-m", PARALLEL, "-n", "auto", "--dist", "loadscope",
               "--max-worker-restart=0", "--timeout=300", "-q", "--tb=short", f"--ci-evidence-dir={evidence}"]
    assert ci_retry.retry_command(command, argsfile=argsfile, direct=[], round_dir=round_dir) == [
        "python", "scripts/run_tests.py", f"@{argsfile}", "-m", PARALLEL, "-n", "auto", "--dist", "loadscope",
        "--max-worker-restart=0", "--timeout=300", "-q", "--tb=short", f"--ci-evidence-dir={round_dir}"]
    mixed = ["python", "-m", "pytest", "-q", "tests/test_a.py", "-k", "tests/test_b.py or x",
             "devtools/test_y.py::test_z", "--deselect", "tests/test_a.py::test_c", "--rootdir", "tests/",
             "-o", "testpaths=tests/", "test_free.py::test_d", "--basetemp", "tests/tmp"]
    assert ci_retry.retry_command(mixed, argsfile=argsfile, direct=["tests/test_ü.py::test_e"],
                                  round_dir=round_dir) == [
        "python", "-m", "pytest", "-q", f"@{argsfile}", "tests/test_ü.py::test_e", "-k", "tests/test_b.py or x",
        "--deselect", "tests/test_a.py::test_c", "--rootdir", "tests/", "-o", "testpaths=tests/",
        "--basetemp", "tests/tmp", f"--ci-evidence-dir={round_dir}"]
    # No test path at all: the argument file and the projection directory are appended.
    assert ci_retry.retry_command(["python", "-m", "pytest", "-q"], argsfile=argsfile, direct=[],
                                  round_dir=round_dir) == [
        "python", "-m", "pytest", "-q", f"@{argsfile}", f"--ci-evidence-dir={round_dir}"]


def test_the_argument_file_holds_printable_ascii_and_reads_the_same_in_every_locale(ci_retry, tmp_path,
                                                                                    monkeypatch):
    from _pytest.config.argparsing import PytestArgumentParser

    assert PytestArgumentParser(usage=None, extra_info={}).fromfile_prefix_chars == "@"
    nodes = ["tests/test_a.py::test_one[a b]", "tests/test_a.py::test_two[x\\u0438]", "tests/test_b.py::TestC::test_d",
             "tests/test_ü.py::test_e", "tests/test_f.py::test_g[line break]", "tests/test_h.py::test_vt[\x0b]"]
    argsfile = tmp_path / "argsfile.txt"
    direct = ci_retry.write_argsfile(argsfile, nodes)
    assert direct == nodes[3:]
    payload = argsfile.read_bytes()
    assert payload.isascii() and payload.decode("ascii").split("\n") == [*nodes[:3], ""]
    # argparse, as pytest parses `@file`, reading it as a Windows runner's cp1252 locale does (Python < 3.12).
    real_open = open
    monkeypatch.setattr(argparse, "open", lambda path, *args, **kwargs: real_open(path, encoding="cp1252"),
                        raising=False)
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("nodes", nargs="*")
    assert parser.parse_args([f"@{argsfile}", *direct]).nodes == nodes[:3] + direct
    # The hazard the split avoids: UTF-8 in the file would reach pytest as other ids.
    argsfile.write_text(nodes[3] + "\n", encoding="utf-8")
    assert parser.parse_args([f"@{argsfile}"]).nodes != [nodes[3]]


def test_run_tests_forwards_a_round_to_pytest_unchanged(ci_retry, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_tests_for_retry", REPO / "scripts" / "run_tests.py")
    run_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_tests)
    monkeypatch.delenv("OUROBOROS_TEST_EVIDENCE_ROOT", raising=False)
    calls = []
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: calls.append((label, argv)) or 0)
    evidence = tmp_path / "parallel"
    round_dir, argsfile = _round(evidence)
    command = ["python", "scripts/run_tests.py", "tests/", "-m", PARALLEL, "-n", "auto", "--dist", "loadscope",
               "--max-worker-restart=0", "--timeout=300", "-q", "--tb=short", f"--ci-evidence-dir={evidence}",
               "--default-suite-shard=2/8"]
    retried = ci_retry.retry_command(command, argsfile=argsfile, direct=["tests/test_ü.py::test_e"],
                                     round_dir=round_dir)
    assert run_tests.main(retried[2:]) == 0
    assert calls == [("focused", [run_tests.sys.executable, "-m", "pytest", *retried[2:]])]


# ---------------------------------------------------------------- rounds and verdicts


def _rows(node, outcome):
    if outcome == "crash":
        return [{"nodeid": node, "phase": "crash", "outcome": "failed", "error_type": "WorkerCrash"}]
    if outcome == "skipped":
        return [{"nodeid": node, "phase": "setup", "outcome": "skipped", "error_type": ""}]
    if outcome == "setup":  # Set up, then the session ended.
        return [{"nodeid": node, "phase": "setup", "outcome": "passed", "error_type": ""}]
    return [{"nodeid": node, "phase": "setup", "outcome": "passed", "error_type": ""},
            {"nodeid": node, "phase": "call", "outcome": outcome,
             "error_type": "AssertionError" if outcome == "failed" else ""},
            {"nodeid": node, "phase": "teardown", "outcome": "passed", "error_type": ""}]


def _session(outcomes, *, exit, assigned=None, collected=None, collection=(), export=True, journal=(), full=None,
             block="default_parallel"):
    """A session that writes its projection into the command's --ci-evidence-dir; its lane is `assigned`
    unless `full` names a lane that other shards share."""
    def play(directory):
        directory.mkdir(parents=True, exist_ok=True)
        if journal:
            (directory / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in journal),
                                                     encoding="utf-8")
        if export:
            data = {"format": 1, "session_exit_code": exit, "collection_failures": [
                        {"nodeid": node, "outcome": "failed"} for node in collection],
                    "tests_collected": len(outcomes) if collected is None else collected,
                    "reports": [row for node, outcome in outcomes.items() for row in _rows(node, outcome)],
                    "github": {"sha": SHA, "run_id": RUN, "run_attempt": "1"}}
            if assigned is not None:
                data[block] = {"full": assigned if full is None else full, "file_weights": [["tests/x.py", 1]],
                               "pass": "parallel", "shard": [1, 2], "assigned": assigned}
            (directory / "results.json").write_text(json.dumps(data), encoding="utf-8")
        return exit
    return play


class _Sessions:
    """Stands in for the wrapped command: each call plays the next session; it records what a round asked for."""

    def __init__(self, *sessions):
        self.sessions, self.requested = list(sessions), []

    def __call__(self, command):
        directory = Path(next(token.split("=", 1)[1] for token in command if token.startswith("--ci-evidence-dir=")))
        files = [Path(token[1:]) for token in command if token.startswith("@")]
        self.requested.append([node for path in files for node in path.read_text(encoding="ascii").splitlines()])
        return self.sessions.pop(0)(directory)


def _wrapped(ci_retry, monkeypatch, tmp_path, *sessions, options=()):
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("GITHUB_RUN_ID", RUN)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    evidence = tmp_path / "parallel"
    runner = _Sessions(*sessions)
    monkeypatch.setattr(ci_retry, "_run", runner)
    code = ci_retry.main(["--evidence-dir", str(evidence), *options, "--", "python", "scripts/run_tests.py", "tests/",
                          "-q", f"--ci-evidence-dir={evidence}", "--default-suite-shard=1/2"])
    assert not runner.sessions, "every planned session ran"
    return code, json.loads((evidence / "verdict.json").read_text(encoding="utf-8")), runner.requested


def test_a_test_that_passes_its_retry_is_flaky_and_does_not_block(ci_retry, monkeypatch, tmp_path, capsys):
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({A: "failed", B: "passed", C: "skipped"}, exit=1),
                                        _session({A: "passed"}, exit=0))
    assert code == 0 and requested == [[], [A]]
    assert verdict == {
        "format": 1, "pass": "parallel", "github": {"sha": SHA, "run_id": RUN, "run_attempt": "1"},
        "platform": ci_retry._system(),
        "rounds": [{"dir": ".", "exit": 1, "ran": 3}, {"dir": "retry-1", "exit": 0, "ran": 1}],
        "flaky": [{"nodeid": A, "first": {"phase": "call", "error_type": "AssertionError"}}],
        "failed": [], "accepted": [], "not_run": [], "continued": 0,
        "counts": {"passed": 1, "flaky": 1, "failed": 0, "accepted": 0, "not_run": 0, "skipped": 1},
        "verdict": "flaky", "reason": ""}
    assert f"FLAKY {A} (first: call, AssertionError)" in capsys.readouterr().out


@pytest.mark.parametrize("retry, exit, second", [("failed", 1, {"phase": "call", "error_type": "AssertionError"}),
                                                 ("skipped", 0, {"phase": "setup", "error_type": "Skipped"})])
def test_a_test_that_does_not_pass_its_retry_fails_and_blocks(ci_retry, monkeypatch, tmp_path, retry, exit, second):
    """Only a pass is a pass: a retry that skips (a green session) proves nothing either."""
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({A: "failed", B: "passed"}, exit=1), _session({A: retry}, exit=exit))
    assert code == 1 and requested == [[], [A]] and verdict["verdict"] == "red"
    assert verdict["failed"] == [{"nodeid": A, "attempts": [{"phase": "call", "error_type": "AssertionError"}, second]}]
    assert verdict["flaky"] == [] and verdict["reason"] == ""


def test_a_stopped_session_is_continued_and_its_crash_retried(ci_retry, monkeypatch, tmp_path):
    """A dead xdist worker stops the session: what it never ran gets a first run, then a retry if it fails."""
    lane = [A, B, C, D]
    code, verdict, requested = _wrapped(
        ci_retry, monkeypatch, tmp_path,
        _session({A: "passed", B: "crash"}, exit=1, assigned=lane, collected=4),
        _session({B: "passed", C: "failed", D: "passed"}, exit=1),
        _session({C: "passed"}, exit=0))
    assert code == 0 and requested == [[], [B, "tests/test_c.py", "tests/test_d.py"], [C]]
    assert verdict["verdict"] == "flaky" and verdict["continued"] == 2 and verdict["not_run"] == []
    assert verdict["flaky"] == [{"nodeid": B, "first": {"phase": "crash", "error_type": "WorkerCrash"}},
                                {"nodeid": C, "first": {"phase": "call", "error_type": "AssertionError"}}]
    assert verdict["counts"] == {"passed": 2, "flaky": 2, "failed": 0, "accepted": 0, "not_run": 0, "skipped": 0}
    assert [row["dir"] for row in verdict["rounds"]] == [".", "retry-1", "retry-2"]


def test_without_a_lane_block_a_dead_workers_crash_record_names_the_rest(ci_retry, monkeypatch, tmp_path):
    """A pass that takes no shard option (journal-persist) records no lane block; the session's crash
    record (tests/conftest.py) lists what it collected and never finished, and that is continued."""
    def crashed(directory):
        code = _session({A: "passed", B: "crash"}, exit=1, collected=4)(directory)
        data = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        data["worker_crash"] = {"worker": "gw1", "reason": "died", "collection_available": True,
                                "unfinished": [C, D]}
        (directory / "results.json").write_text(json.dumps(data), encoding="utf-8")
        return code
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path, crashed,
                                        _session({B: "passed", C: "passed", D: "passed"}, exit=0))
    assert code == 0 and requested == [[], [B, "tests/test_c.py", "tests/test_d.py"]] and verdict["continued"] == 2
    # Without that record the rest stays unknown.
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path / "unknown",
                                        _session({A: "passed", B: "crash"}, exit=1, collected=4))
    assert code == 1 and requested == [[]]
    assert verdict["reason"] == "2 collected test(s) never ran, and no lane block or crash record names them"


def test_a_continuation_asks_for_unrun_files_by_path_and_everything_else_by_id(ci_retry, monkeypatch, tmp_path):
    """The unrun rest of a crashed shard is thousands of tests, and pytest matches each node id against every
    entry of its directory (on Windows with a file-identity check per entry: run 42061846211, Windows shard 7,
    collected nothing after 34 minutes). A round names a file none of whose tests has an attempt by its path,
    and a file's remaining tests by id once one of them ran; the crashed test keeps its id."""
    x1, x2, y1, y2 = ("tests/test_x.py::test_one", "tests/test_x.py::test_two",
                      "tests/test_y.py::test_one", "tests/test_y.py::test_two")
    code, verdict, requested = _wrapped(
        ci_retry, monkeypatch, tmp_path,
        _session({x1: "passed", A: "crash"}, exit=1, assigned=[A, x1, x2, y1, y2], collected=5),
        _session({A: "passed", x2: "passed", y1: "passed", y2: "passed"}, exit=0))
    assert code == 0 and requested == [[], [A, x2, "tests/test_y.py"]]
    assert verdict["verdict"] == "flaky" and verdict["not_run"] == [] and verdict["continued"] == 3
    assert ci_retry.round_arguments([y2, y1, x2], {x1: []}) == ["tests/test_y.py", x2]


def test_a_lane_that_packs_single_tests_continues_a_shared_file_by_id(ci_retry, monkeypatch, tmp_path):
    """The UI lane packs single tests into shards, so a file's other tests belong to other shards. The round drops
    the shard option: asked by its path, the file ran them all here and reconcile-shards refused the shard for
    executing outside its assignment (run 42395824706, UI shard 3: 7 unrun tests after the session timeout, 69
    collected). A file the lane shares goes by id; one whose every lane test is pending still goes by path."""
    x1, x2, y1, y2 = ("tests/test_x.py::test_one", "tests/test_x.py::test_two",
                      "tests/test_y.py::test_one", "tests/test_y.py::test_two")
    lane = [A, x1, x2, y1, y2]  # x2 is another shard's.
    code, verdict, requested = _wrapped(
        ci_retry, monkeypatch, tmp_path,
        _session({A: "passed"}, exit=1, assigned=[A, x1, y1, y2], full=lane, block="ui_browser", collected=4),
        _session({x1: "passed", y1: "passed", y2: "passed"}, exit=0))
    assert code == 0 and requested == [[], [x1, "tests/test_y.py"]]
    assert verdict["verdict"] == "green" and verdict["not_run"] == [] and verdict["continued"] == 3
    assert ci_retry.round_arguments([x1, y1, y2], {A: []}, lane) == [x1, "tests/test_y.py"]
    # A lane that keeps files whole (or none named) asks for both files by path, as before.
    assert ci_retry.round_arguments([x1, y1, y2], {A: []}, [A, x1, y1, y2]) == ["tests/test_x.py", "tests/test_y.py"]
    assert ci_retry.round_arguments([x1, y1, y2], {A: []}) == ["tests/test_x.py", "tests/test_y.py"]


def test_a_test_that_never_runs_blocks(ci_retry, monkeypatch, tmp_path):
    """A continuation round that stops again leaves its rest for the next round; one that runs none of its
    tests ends the rounds, and what it left is NOT RUN."""
    code, verdict, requested = _wrapped(
        ci_retry, monkeypatch, tmp_path,
        _session({A: "failed"}, exit=1, assigned=[A, B, C], collected=3),
        _session({A: "passed", B: "passed"}, exit=2),
        _session({}, exit=4, export=False))
    assert code == 1 and requested == [[], [A, "tests/test_b.py", "tests/test_c.py"], ["tests/test_c.py"]]
    assert verdict["verdict"] == "red" and verdict["not_run"] == [C] and verdict["flaky"][0]["nodeid"] == A
    assert verdict["reason"] == "retry-2 ran none of its 1 test(s) (exit status 4)"
    assert verdict["rounds"][-1] == {"dir": "retry-2", "exit": 4, "ran": 0}


def test_a_retry_that_runs_nothing_leaves_the_failure_with_one_attempt(ci_retry, monkeypatch, tmp_path):
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({A: "failed", B: "passed"}, exit=1),
                                        _session({}, exit=4))
    assert code == 1 and verdict["verdict"] == "red"
    assert verdict["failed"] == [{"nodeid": A, "attempts": [{"phase": "call", "error_type": "AssertionError"}]}]
    assert "retry-1 ran none of its 1 test(s)" in verdict["reason"]


def _accepted_file(tmp_path, entries):
    path = tmp_path / "accepted_failures.json"
    path.write_text(json.dumps({"format": 1, "label": ci_accepted.LABEL, "entries": entries}), encoding="utf-8")
    return path


def test_an_accepted_failure_does_not_block_and_the_command_never_sees_the_token(ci_retry, monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "owner")
    monkeypatch.setenv("GH_TOKEN", "test-token")
    reads = []
    monkeypatch.setattr(ci_accepted, "read_issue", lambda repo, number, token: reads.append(number) or {"state": "open"})
    monkeypatch.setattr(ci_accepted, "read_events", lambda repo, number, token: [
        {"event": "labeled", "label": {"name": ci_accepted.LABEL}, "actor": {"login": "owner"}}])
    accepted = _accepted_file(tmp_path, [{"nodeid": A, "issue": 2083}, {"nodeid": C, "issue": 9}])
    code, verdict, _ = _wrapped(ci_retry, monkeypatch, tmp_path, _session({A: "failed", B: "failed"}, exit=1),
                                _session({A: "failed", B: "passed"}, exit=1), options=["--accepted", str(accepted)])
    assert code == 0 and verdict["verdict"] == "accepted" and reads == [2083]
    assert verdict["accepted"] == [{"nodeid": A, "issue": 2083}] and [row["nodeid"] for row in verdict["failed"]] == [A]
    assert verdict["counts"]["accepted"] == 1 and verdict["flaky"][0]["nodeid"] == B
    # Without the repository's token, nothing is accepted and nothing is read.
    reads.clear()
    monkeypatch.delenv("GH_TOKEN")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    code, verdict, _ = _wrapped(ci_retry, monkeypatch, tmp_path / "again", _session({A: "failed"}, exit=1),
                                _session({A: "failed"}, exit=1), options=["--accepted", str(accepted)])
    assert code == 1 and verdict["verdict"] == "red" and verdict["accepted"] == [] and reads == []


def test_the_program_is_found_on_path_as_the_calling_shell_finds_it(ci_retry, monkeypatch):
    """Windows' own lookup tries the interpreter's directory first, where a venv's python.exe runs
    as the base interpreter, so a bare `python` started the hosted Python without the venv's pytest.
    The command inherits the environment less the repository's API token (scripts/ci_retry.py `_run`)."""
    started = []
    monkeypatch.setenv("GH_TOKEN", "test-token-a")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-b")
    monkeypatch.setenv("CI_RETRY_PROBE", "kept")

    class Process:
        def __init__(self, command, env):
            assert "GH_TOKEN" not in env and "GITHUB_TOKEN" not in env
            assert env["CI_RETRY_PROBE"] == "kept"
            started.append(command)

        def wait(self):
            return 0

    monkeypatch.setattr(ci_retry.subprocess, "Popen", Process)
    monkeypatch.setattr(ci_retry.shutil, "which", {"python": "/venv/bin/python"}.get)
    assert ci_retry._run(["python", "scripts/run_tests.py"]) == 0
    assert ci_retry._run(["not-on-path", "scripts/run_tests.py"]) == 0
    assert started == [["/venv/bin/python", "scripts/run_tests.py"], ["not-on-path", "scripts/run_tests.py"]]


def test_without_any_evidence_the_command_status_stands(ci_retry, monkeypatch, tmp_path):
    code, verdict, _ = _wrapped(ci_retry, monkeypatch, tmp_path, _session({}, exit=3, export=False))
    assert code == 3 and verdict["verdict"] == "red" and verdict["rounds"] == [{"dir": ".", "exit": 3, "ran": 0}]
    assert "left neither results.json nor events.jsonl" in verdict["reason"]


def test_a_green_session_needs_no_round(ci_retry, monkeypatch, tmp_path):
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({A: "passed", B: "skipped"}, exit=0))
    assert code == 0 and requested == [[]] and verdict["verdict"] == "green" and verdict["reason"] == ""
    assert verdict["counts"] == {"passed": 1, "flaky": 0, "failed": 0, "accepted": 0, "not_run": 0, "skipped": 1}
    assert not (tmp_path / "parallel" / "retry-1").exists()


@pytest.mark.parametrize("session, reason", [
    (_session({A: "passed"}, exit=1), "the session ended with exit status 1, but no test failed or went unexecuted"),
    (_session({A: "passed", B: "setup"}, exit=2, collected=3),
     "2 collected test(s) never ran, and no lane block or crash record names them"),
    (_session({}, exit=1, export=False, journal=[
        {"event": "start", "nodeid": A}, {"event": "report", "nodeid": A, "phase": "call", "outcome": "passed"},
        {"event": "finish", "nodeid": A}, {"event": "start", "nodeid": B}]),
     "the session ended before its final export, and no lane block names its assignment"),
], ids=["unexplained", "stopped-without-lane", "killed-without-lane"])
def test_a_red_session_that_no_test_explains_blocks_without_rounds(ci_retry, monkeypatch, tmp_path, session, reason):
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path, session)
    assert code == 1 and requested == [[]] and verdict["verdict"] == "red" and reason in verdict["reason"]
    assert len(verdict["rounds"]) == 1


def test_a_test_in_flight_when_its_session_ended_is_a_failed_attempt(ci_retry, monkeypatch, tmp_path):
    journal = [{"event": "start", "nodeid": A}, {"event": "start", "nodeid": B},
               {"event": "report", "nodeid": B, "phase": "call", "outcome": "passed"}, {"event": "finish", "nodeid": B}]
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({A: "setup", B: "passed"}, exit=2, journal=journal),
                                        _session({A: "passed"}, exit=0))
    assert code == 0 and requested == [[], [A]]
    assert verdict["flaky"] == [{"nodeid": A, "first": {"phase": "crash", "error_type": "InFlight"}}]


def _lane_row(assigned):
    return {"event": "lane", "nodeid": "", "block": "default_serial",
            "github": {"sha": SHA, "run_id": RUN, "run_attempt": "1"},
            "lane": {"full": assigned, "file_weights": [["tests/x.py", 1]], "pass": "serial",
                     "shard": [1, 2], "assigned": assigned}}


def test_a_killed_session_is_continued_from_the_lane_it_journaled(ci_retry, monkeypatch, tmp_path):
    """No final export (pytest-timeout's thread method ends a whole serial session on Windows): the lane
    block journaled at collection names the assignment, the test in flight gets its retry, the rest a first run."""
    journal = [_lane_row([A, B, C]), {"event": "start", "nodeid": A},
               {"event": "report", "nodeid": A, "phase": "call", "outcome": "passed"},
               {"event": "finish", "nodeid": A}, {"event": "start", "nodeid": B}]
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({}, exit=1, export=False, journal=journal),
                                        _session({B: "passed", C: "passed"}, exit=0))
    assert code == 0 and requested == [[], [B, "tests/test_c.py"]]  # B ran once (by id); test_c.py never ran (by path)
    assert verdict["verdict"] == "flaky" and verdict["not_run"] == [] and verdict["continued"] == 1
    assert verdict["flaky"] == [{"nodeid": B, "first": {"phase": "crash", "error_type": "InFlight"}}]
    assert verdict["rounds"][0] == {"dir": ".", "exit": 1, "ran": 2}
    # The same kill without a journaled lane leaves the rest unknown: red, and nothing is retried.
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path / "unknown",
                                        _session({}, exit=1, export=False, journal=journal[1:]))
    assert code == 1 and requested == [[]] and "no lane block names its assignment" in verdict["reason"]


def test_beside_a_dead_worker_a_test_in_flight_runs_again_without_spending_its_retry(ci_retry, monkeypatch, tmp_path):
    """An owned run aborts on a dead worker while other workers are mid-test (tests/conftest.py): the crash
    explains the stop, so their tests run again as if they never ran; a bare kill makes the test in flight
    the suspect instead (test_a_killed_session_is_continued_from_the_lane_it_journaled)."""
    journal = [{"event": "start", "nodeid": A}, {"event": "start", "nodeid": B},
               {"event": "report", "nodeid": B, "phase": "crash", "outcome": "failed", "error_type": "WorkerCrash"}]
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({B: "crash"}, exit=1, assigned=[A, B, C], journal=journal),
                                        _session({A: "failed", B: "passed", C: "passed"}, exit=1),
                                        _session({A: "passed"}, exit=0))
    assert code == 0 and requested == [[], ["tests/test_a.py", B, "tests/test_c.py"], [A]]  # unrun files by path
    assert verdict["flaky"] == [{"nodeid": A, "first": {"phase": "call", "error_type": "AssertionError"}},
                                {"nodeid": B, "first": {"phase": "crash", "error_type": "WorkerCrash"}}]
    assert verdict["continued"] == 2


def test_an_internal_error_leaves_the_tests_in_flight_unblamed(ci_retry, monkeypatch, tmp_path):
    """pytest's internal error (exit status 3) comes from a hook or the xdist scheduler between tests
    (a conftest close after the reports, a scheduler assertion on Windows): the tests other workers had
    in flight run again as if they never ran, rather than failing once each for a stop they did not cause."""
    journal = [{"event": "start", "nodeid": A}, {"event": "start", "nodeid": B},
               {"event": "report", "nodeid": B, "phase": "call", "outcome": "passed"}, {"event": "finish", "nodeid": B}]
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({B: "passed"}, exit=3, assigned=[A, B, C], journal=journal),
                                        _session({A: "passed", C: "passed"}, exit=0))
    assert code == 0 and requested == [[], ["tests/test_a.py", "tests/test_c.py"]]  # both never ran: by path
    assert verdict["verdict"] == "green" and verdict["flaky"] == [] and verdict["continued"] == 2


def test_a_collection_failure_blocks_and_is_never_retried(ci_retry, monkeypatch, tmp_path):
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({A: "failed"}, exit=2, collection=["tests/test_broken.py"]),
                                        _session({A: "passed"}, exit=0))
    assert code == 1 and requested == [[], [A]] and verdict["verdict"] == "red"
    assert verdict["failed"] == [{"nodeid": "tests/test_broken.py",
                                  "attempts": [{"phase": "collect", "error_type": "CollectionError"}]}]


def test_the_round_backstop_ends_red_and_names_the_rest(ci_retry, monkeypatch, tmp_path):
    code, verdict, requested = _wrapped(ci_retry, monkeypatch, tmp_path,
                                        _session({A: "failed", B: "failed"}, exit=1), options=["--max-rounds", "0"])
    assert code == 1 and requested == [[]]
    assert verdict["reason"] == "2 test(s) still pending after 0 retry round(s)"
    assert [row["nodeid"] for row in verdict["failed"]] == [A, B]


@pytest.mark.parametrize("leftover", ["verdict.json", "retry-1"])
def test_an_evidence_directory_with_an_earlier_verdict_or_round_is_refused(ci_retry, monkeypatch, tmp_path, leftover):
    """A round that leaves no projection must never be read from an earlier invocation's."""
    evidence = tmp_path / "parallel"
    evidence.mkdir()
    if leftover == "retry-1":
        (evidence / leftover).mkdir()
    else:
        (evidence / leftover).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ci_retry, "_run", lambda command: pytest.fail("nothing may run"))
    with pytest.raises(SystemExit) as stop:
        ci_retry.main(["--evidence-dir", str(evidence), "--", "python", "-m", "pytest", "tests/"])
    assert stop.value.code == 2
    with pytest.raises(SystemExit) as stop:
        ci_retry.main(["--evidence-dir", str(tmp_path / "fresh")])  # No command after --.
    assert stop.value.code == 2


def test_the_retry_reader_counts_execution_as_the_reconciler_does(ci_retry):
    """scripts/ci_retry.py cannot import tests/ci_evidence.py (pytest below its CLI): both rules agree."""
    from tests.ci_evidence import _executed

    rows = [row for node, outcome in ((A, "passed"), (B, "failed"), (C, "skipped"), (D, "crash"))
            for row in _rows(node, outcome)] + _rows("tests/test_e.py::test_case", "setup")
    assert set(ci_retry._outcomes(rows)) == _executed(rows) == {A, B, C, D}


# ---------------------------------------------------------------- real sessions


def _entry(tmp_path, suite):
    entry = tmp_path / "run.py"
    entry.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"os.chdir({str(suite)!r})\n"
        f"os.environ.update(GITHUB_SHA={SHA!r}, GITHUB_RUN_ID={RUN!r}, GITHUB_RUN_ATTEMPT='1')\n"
        "import pytest\n"
        "raise SystemExit(pytest.main(sys.argv[1:]))\n", encoding="utf-8")
    return entry


def _command(tmp_path, suite, *options):
    return [sys.executable, "-I", "-S", str(REPO / "scripts/safe_test.py"), "--temp-parent", str(tmp_path), "--",
            sys.executable, str(_entry(tmp_path, suite)), "tests/", "-p", "tests.conftest", "-o", "addopts=",
            "--rootdir", str(suite), "--confcutdir", str(suite), "-q", *options]


def _wrapper(tmp_path, evidence, command, env=None):
    return subprocess.run(
        [sys.executable, "-I", "-S", str(REPO / "scripts/ci_retry.py"), "--evidence-dir", str(evidence), "--", *command],
        env={**os.environ, "GITHUB_SHA": SHA, "GITHUB_RUN_ID": RUN, "GITHUB_RUN_ATTEMPT": "1", **(env or {})},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("last", [True, False], ids=["in-the-last-test", "with-more-to-run"])
def test_a_session_timeout_stops_only_what_is_left(tmp_path, last):
    """pytest-timeout asks the session to stop after any test that ends past its deadline, the last one too, and
    the request alone made the exit status 1: a UI shard whose 313 tests all passed went red, crossing its
    deadline in the last one (run 42395824706, shard 2). tests/conftest.py ends such a loop as a complete one;
    with tests left, the session still stops before them."""
    suite = tmp_path / "suite"
    (suite / "tests").mkdir(parents=True)
    expires = textwrap.dedent("""
        def test_expires(request):
            import pytest_timeout
            request.config.stash[pytest_timeout.SESSION_EXPIRE_KEY] = 1.0  # The deadline passes in this test.
        """)
    other = "def test_other():\n    pass\n"
    (suite / "tests" / "test_specimen.py").write_text(other + expires if last else expires + other, encoding="utf-8")
    completed = subprocess.run(_command(tmp_path, suite, "--session-timeout=600"),
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    output = completed.stdout + completed.stderr
    if last:
        assert completed.returncode == 0, output
        assert "2 passed" in output and "session-timeout: 600.0 sec exceeded in the last test" in output
    else:
        assert completed.returncode == 1, output
        assert "1 passed" in output and "session-timeout: 600.0 sec exceeded" in output


@pytest.mark.parametrize("always", [False, True], ids=["fails-once", "fails-twice"])
def test_a_real_session_retries_its_failure_once(tmp_path, always):
    suite = tmp_path / "suite"
    (suite / "tests").mkdir(parents=True)
    (suite / "tests" / "test_specimen.py").write_text(textwrap.dedent(f"""
        from pathlib import Path
        def test_fails_once():
            marker = Path({str(tmp_path / "failed-once")!r})
            if {always!r} or not marker.exists():
                marker.write_text("", encoding="utf-8")
                assert False, "the first run fails"
        def test_passes():
            pass
    """), encoding="utf-8")
    evidence = tmp_path / "evidence"
    result = _wrapper(tmp_path, evidence, _command(tmp_path, suite, f"--ci-evidence-dir={evidence}"))
    node = "tests/test_specimen.py::test_fails_once"
    verdict = _json(evidence / "verdict.json")
    assert result.returncode == int(always), result.stdout + result.stderr
    assert verdict["verdict"] == ("red" if always else "flaky")
    assert verdict["rounds"] == [{"dir": ".", "exit": 1, "ran": 2}, {"dir": "retry-1", "exit": int(always), "ran": 1}]
    assert (evidence / "retry-1" / "argsfile.txt").read_text(encoding="ascii") == node + "\n"
    assert _json(evidence / "results.json")["session_exit_code"] == 1, "the first session's projection stays"
    assert {row["nodeid"] for row in _json(evidence / "retry-1" / "results.json")["reports"]} == {node}
    if always:
        assert verdict["failed"] == [{"nodeid": node, "attempts": [{"phase": "call", "error_type": "AssertionError"}] * 2}]
        assert f"FAILED {node} (call, AssertionError; call, AssertionError)" in result.stdout
    else:
        assert verdict["flaky"] == [{"nodeid": node, "first": {"phase": "call", "error_type": "AssertionError"}}]
        assert f"FLAKY {node} (first: call, AssertionError)" in result.stdout


_UI_ATTEMPTS = '''
import json, os, time
from pathlib import Path
import pytest
from ouroboros.test_evidence import diagnostic_directory, keep_startup_log


def attempt(name):
    seen = Path(@SEEN@) / (name + ".seen")
    first = not seen.exists()
    seen.write_text("", encoding="utf-8")
    return "first" if first else "retry"


def test_receipt_frames(tmp_path):
    label = attempt("frames")
    frames = Path(os.environ["OUROBOROS_UI_EVIDENCE_DIR"]) / "late-input-heads"
    frames.mkdir(parents=True, exist_ok=True)
    (frames / "receipt.png").write_bytes(f"receipt seen by the {label} attempt".encode())
    (frames / "results.json").write_text(json.dumps({"receipt_width": 1 if label == "first" else 2}), encoding="utf-8")
    (diagnostic_directory(tmp_path, "receipt-facts", publish=True) / "facts.json").write_text(json.dumps({"attempt": label}))
    (diagnostic_directory(tmp_path / "drive", "receipt-facts") / "drive.json").write_text("a retained neighbour")
    assert label == "retry", "the receipt is narrower than its title"


@pytest.mark.timeout(3, method="signal", func_only=True)
def test_server_readiness(tmp_path):
    label = attempt("readiness")
    (tmp_path / "data" / "logs").mkdir(parents=True)
    (tmp_path / "server-stdio.log").write_text(f"server of the {label} attempt\\n", encoding="utf-8")
    (tmp_path / "data" / "logs" / "supervisor.jsonl").write_text('{"event": "boot"}\\n', encoding="utf-8")
    with keep_startup_log(tmp_path, "server-stdio.log"):
        if label == "first":
            time.sleep(60)  # never ready: pytest-timeout's SIGALRM ends the wait
'''


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGALRM"), reason="pytest-timeout's signal method needs SIGALRM")
def test_each_rounds_raw_run_keeps_its_diagnostics_apart_from_the_proof(tmp_path):
    """#2716/#2763 as the UI lane runs: ci_retry -> safe_test --evidence-dir -> pytest. A first session that fails (a
    custom frame assertion; a readiness wait stopped by a real SIGALRM) and a round that passes stay FLAKY. Each
    invocation's run, a round's beneath retry-<k>, keeps its frames, facts and startup log past its released scratch,
    under one name with each attempt's own bytes; the export takes exactly those, at the run's relative path, and
    never their family's retained neighbour."""
    from ouroboros.test_environment import DIAGNOSTIC_OUTPUT_KEYS
    from ouroboros.test_evidence import EVIDENCE_RUN_MARKER

    suite = tmp_path / "suite"
    (suite / "tests").mkdir(parents=True)
    (suite / "tests" / "test_specimen.py").write_text(_UI_ATTEMPTS.replace("@SEEN@", repr(str(tmp_path))),
                                                      encoding="utf-8")
    evidence, raw, out = tmp_path / "ci-evidence" / "host", tmp_path / "ci-raw" / "host", tmp_path / "ci-diagnostics"
    command = _command(tmp_path, suite, f"--ci-evidence-dir={evidence}")
    command[command.index("--"):command.index("--")] = ["--evidence-dir", str(raw)]
    # As on the runner: no step-level UI destination, so each invocation's falls in its own run.
    env = {key: value for key, value in os.environ.items() if key not in DIAGNOSTIC_OUTPUT_KEYS}
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(REPO / "scripts/ci_retry.py"), "--evidence-dir", str(evidence), "--", *command],
        env={**env, "GITHUB_SHA": SHA, "GITHUB_RUN_ID": RUN, "GITHUB_RUN_ATTEMPT": "1"},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    frames, readiness = "tests/test_specimen.py::test_receipt_frames", "tests/test_specimen.py::test_server_readiness"
    verdict = _json(evidence / "verdict.json")
    assert verdict["verdict"] == "flaky" and [row["dir"] for row in verdict["rounds"]] == [".", "retry-1"]
    assert verdict["flaky"] == [{"nodeid": frames, "first": {"phase": "call", "error_type": "AssertionError"}},
                                {"nodeid": readiness, "first": {"phase": "call", "error_type": "Failed"}}]
    assert "Timeout (>3.0s) from pytest-timeout" in output
    ends = [line.split(" ", 1) for line in result.stderr.splitlines()
            if line.startswith(("SAFE_TEST_RELEASED ", "SAFE_TEST_RETAINED "))]
    assert [(end, Path(root).exists()) for end, root in ends] == [("SAFE_TEST_RELEASED", False)] * 2, output
    runs = {label: [marker.parent for marker in raw.glob(pattern + "/" + EVIDENCE_RUN_MARKER)]
            for label, pattern in (("first", "run-*"), ("retry", "retry-1/run-*"))}
    assert [len(found) for found in runs.values()] == [1, 1], runs

    exported = subprocess.run([sys.executable, "-m", "ouroboros.test_evidence", "export-diagnostics", str(raw), str(out)],
                              cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert exported.returncode == 0 and "CI_DIAGNOSTICS_EXPORTED 8 file(s)" in exported.stdout, exported.stdout
    for label, (run,) in runs.items():
        mine = out / run.relative_to(raw)
        files = {path.relative_to(mine).as_posix().split("/", 2)[0] + "/" + path.name: path
                 for path in mine.rglob("*") if path.is_file()}
        assert set(files) == {"receipt-facts/facts.json", "ui/receipt.png", "ui/results.json"} | (
            {"server-startup/server-stdio.log", "server-startup/supervisor.jsonl"} if label == "first" else set())
        assert files["ui/receipt.png"].read_bytes() == f"receipt seen by the {label} attempt".encode()
        assert _json(files["receipt-facts/facts.json"]) == {"attempt": label}
        if label == "first":  # the timed-out wait's own log, beside the data logs it had
            assert files["server-startup/server-stdio.log"].read_text(encoding="utf-8") == "server of the first attempt\n"
        assert [path.read_text(encoding="utf-8") for path in run.glob("receipt-facts/*/drive.json")] == [
            "a retained neighbour"], "it stays with its run"
    # The lane's proof holds its projections only. Read as proof, the custom results.json would be a gap that turns
    # the lane RED: why the diagnostics travel in their own artifact, which no proof download selects.
    assert not [path for path in evidence.rglob("*") if path.name in ("receipt.png", "facts.json", "server-stdio.log")]
    gaps, _ = reconcile_shards(tmp_path / "no-manifest", out, count=1, sha=SHA, run_id=RUN)
    assert any("ui/late-input-heads/results.json: invalid result projection" in gap for gap in gaps), gaps


def _reconciled(tmp_path, suite, expression, lane_key):
    """The unsharded manifest of the synthetic suite, then the shards reconciled against it."""
    manifest = tmp_path / "manifest" / "default-ci-manifest-attempt1" / lane_key
    collected = subprocess.run(
        _command(tmp_path, suite, "-m", expression, "--collect-only", "--default-suite-manifest",
                 f"--ci-evidence-dir={manifest}"),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert collected.returncode == 0, collected.stdout + collected.stderr
    return reconcile_shards(tmp_path / "manifest", tmp_path / "shards", count=1, sha=SHA, run_id=RUN,
                            lane_key=lane_key)


@pytest.mark.parametrize("death, seconds", [("os._exit(3)", None), ("time.sleep(60)", 2.0)],
                         ids=["exit", "thread-timeout"])
def test_a_dead_worker_is_continued_and_the_shard_still_reconciles(tmp_path, death, seconds):
    """--max-worker-restart=0 stops the whole session when a worker dies (an os._exit, or the thread timeout
    that ends the worker on Windows), and an owned run's controller aborts before the other workers report
    (tests/conftest.py): the lane block they handed over at collection still reaches the export, the round runs
    the crashed test again and every test the session never reached, and the shard's red session is proven by its
    verdict. The verdict and its readers keep the dead worker's own facts: a timeout only where its callback ran."""
    suite = tmp_path / "suite"
    (suite / "tests").mkdir(parents=True)
    # --dist loadscope sends the largest scope first: its first test kills its worker, at once or when its timer fires.
    (suite / "tests" / "test_a_crash.py").write_text(textwrap.dedent(f"""
        import os, time
        from pathlib import Path
        import pytest
        @pytest.mark.timeout(2, method="thread")
        def test_dies_once():
            marker = Path({str(tmp_path / "crashed-once")!r})
            if not marker.exists():
                marker.write_text("", encoding="utf-8")
                {death}
        @pytest.mark.parametrize("n", range(4))
        def test_after(n):
            pass
    """), encoding="utf-8")
    for name in "bcd":
        (suite / "tests" / f"test_{name}.py").write_text(
            "import time\nimport pytest\n@pytest.mark.parametrize('n', range(2))\n"
            "def test_slow(n):\n    time.sleep(0.5)\n", encoding="utf-8")
    evidence = tmp_path / "shards" / "default-ci-shard-1-attempt1" / "parallel"
    result = _wrapper(tmp_path, evidence, _command(
        tmp_path, suite, "-m", PARALLEL, "-n", "2", "--dist", "loadscope", "--max-worker-restart=0",
        f"--ci-evidence-dir={evidence}", "--default-suite-shard=1/1"), env={NESTED_WORKERS_ENV: "2"})
    assert result.returncode == 0, result.stdout + result.stderr
    first = _json(evidence / "results.json")
    assigned = first["default_parallel"]["assigned"]
    reached = {row["nodeid"] for row in first["reports"] if row["phase"] in ("call", "crash")}
    assert first["session_exit_code"] != 0 and len(assigned) == 11 and first["worker_crash"]
    assert set(assigned) - reached, "the stopped session left tests it never ran"
    assert not list(evidence.glob(".lane-*")), "the controller took the workers' hand-over"
    lanes = [json.loads(line) for line in (evidence / "events.jsonl").read_text(encoding="utf-8").splitlines()
             if '"event": "lane"' in line]
    assert [row["block"] for row in lanes] == ["default_parallel"] and lanes[0]["lane"] == first["default_parallel"]
    verdict = _json(evidence / "verdict.json")
    assert verdict["verdict"] == "flaky" and verdict["not_run"] == [] and verdict["failed"] == []
    died = {"phase": "crash", "error_type": "WorkerCrash", "worker": first["worker_crash"]["worker"],
            **({"timeout_callback_seconds": seconds} if seconds else {})}
    assert verdict["flaky"] == [{"nodeid": "tests/test_a_crash.py::test_dies_once", "first": died}]
    shown = "tests/test_a_crash.py::test_dies_once (first: crash, WorkerCrash" + (
        "; timeout callback invoked (2s), worker subsequently lost)" if seconds else ")")
    assert f"FLAKY {shown}" in result.stdout
    assert render_annotations([evidence]) == [f"::warning title=Flaky test (passed on its retry)::{shown}"]
    text, facts = render_run_verdict(tmp_path / "shards")
    assert f"- {shown}" in text and facts["flaky"][0].get("timeout_callback_seconds") == seconds
    # A test the other worker had in flight when the controller aborted runs again as never run.
    assert verdict["continued"] == len(set(assigned) - reached)
    gaps, notes = _reconciled(tmp_path, suite, PARALLEL, "default_parallel")
    assert gaps == [] and any("retry verdict flaky over 2 round(s): FLAKY 1, ACCEPTED 0" in note for note in notes)
    # The same red session without its verdict is no proof.
    (evidence / "verdict.json").unlink()
    gaps, _ = reconcile_shards(tmp_path / "manifest", tmp_path / "shards", count=1, sha=SHA, run_id=RUN,
                               lane_key="default_parallel")
    assert any("its session ended with exit status" in gap for gap in gaps)
    assert any("assigned but not executed" in gap for gap in gaps)


def test_a_killed_session_is_continued_from_its_journal_and_proven_by_its_verdict(tmp_path):
    """pytest-timeout's thread method ends a whole serial session on Windows: no final export at all.
    The lane block journaled at collection lets the rounds retry the test in flight and run the rest,
    and the journal with that verdict proves the shard."""
    suite = tmp_path / "suite"
    (suite / "tests").mkdir(parents=True)
    serial = "import pytest\npytestmark = pytest.mark.serial('exclusive')\n"
    (suite / "tests" / "test_a.py").write_text(serial + "@pytest.mark.parametrize('n', range(2))\n"
                                               "def test_quick(n):\n    pass\n", encoding="utf-8")
    (suite / "tests" / "test_b_killed.py").write_text(serial + textwrap.dedent(f"""
        import os
        from pathlib import Path
        def test_killed_once():
            marker = Path({str(tmp_path / "killed-once")!r})
            if not marker.exists():
                marker.write_text("", encoding="utf-8")
                os._exit(1)
    """), encoding="utf-8")
    (suite / "tests" / "test_c.py").write_text(serial + "@pytest.mark.parametrize('n', range(2))\n"
                                               "def test_later(n):\n    pass\n", encoding="utf-8")
    expression = f"exclusive and {LANE_EXCLUSION_EXPR}"
    evidence = tmp_path / "shards" / "default-ci-shard-1-attempt1" / "serial"
    result = _wrapper(tmp_path, evidence, _command(tmp_path, suite, "-m", expression,
                                                   f"--ci-evidence-dir={evidence}", "--default-suite-shard=1/1"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (evidence / "results.json").exists(), "the first session was killed before its export"
    verdict = _json(evidence / "verdict.json")
    killed = "tests/test_b_killed.py::test_killed_once"
    assert verdict["verdict"] == "flaky" and verdict["not_run"] == [] and verdict["continued"] == 2
    assert verdict["flaky"] == [{"nodeid": killed, "first": {"phase": "crash", "error_type": "InFlight"}}]
    assert (evidence / "retry-1" / "argsfile.txt").read_text(encoding="ascii").split() == [
        killed, "tests/test_c.py"]  # the killed test by id; test_c.py, which never ran, by path
    gaps, notes = _reconciled(tmp_path, suite, expression, "default_serial")
    assert gaps == [], gaps
    assert any("proof from attempt 1; assigned 5, executed 5; retry verdict flaky" in note for note in notes)
    # Without a passing verdict of this run the journal is no proof.
    for change in ({"verdict": "red"}, {"github": {"sha": SHA, "run_id": "778", "run_attempt": "1"}}, None):
        rewritten = dict(verdict, **change) if change else None
        if rewritten is None:
            (evidence / "verdict.json").unlink()
        else:
            (evidence / "verdict.json").write_text(json.dumps(rewritten), encoding="utf-8")
        gaps, _ = reconcile_shards(tmp_path / "manifest", tmp_path / "shards", count=1, sha=SHA, run_id=RUN,
                                   lane_key="default_serial")
        assert any(gap.startswith("shard 1/1: its session ended before its final export") for gap in gaps), gaps
