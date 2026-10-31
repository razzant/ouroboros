"""A failed launch stays red, and only the first teardown answer decides its scratch.

Faults are injected at the OS seam (``Popen``) and at the container's own answer; the
consumers are real: the container, the hermetic gate's lanes and safe_test's two stages.
"""
from __future__ import annotations

import errno
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import preflight_node, preflight_runner as pr, process_containment, process_lifeline, safe_test_supervisor
from ouroboros import test_environment, test_evidence
from ouroboros.process_containment import ProcessContainer
from ouroboros.process_lifeline import ProcessLifeline
from ouroboros.process_overload import OverloadWatch
from ouroboros.test_environment import RETENTION_MARKER
from tests._shared import retire_synthetic_gate_hold, retire_synthetic_retention
from tests.test_preflight_receipts import _ctx, _tiny_repo, gate_env, receipt_python  # noqa: F401 -- the gate's fixtures
from tests.test_preflight_runner import _make_repo, _pin_preflight_root

REPO = pathlib.Path(__file__).resolve().parents[1]
UNCONFIRMED = "pid 4242 was still alive after the reap"
GUARDIAN_UNSETTLED = "process lifeline did not exit after pipe closure"
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX guardian and token custody")


def _popen_refusing(monkeypatch, *needles, started=None):
    """The container's own view of ``subprocess`` refuses one argv: git and probes keep the real one."""
    def popen(argv, *args, **kwargs):
        if all(part in argv for part in needles):
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
        proc = subprocess.Popen(argv, *args, **kwargs)
        if started is not None:
            started.append(proc)
        return proc

    monkeypatch.setattr(process_containment, "subprocess", SimpleNamespace(**{
        **{name: getattr(subprocess, name) for name in dir(subprocess) if not name.startswith("__")}, "Popen": popen}))


def _first_reap_unconfirmed(monkeypatch, reason=UNCONFIRMED):
    """The first scan really runs, then answers ``reason``; every later scan is the real, empty one."""
    real, answers = ProcessContainer._reap, [reason]

    def reap(self):
        answer = real(self)
        return answers.pop() if answers else answer

    monkeypatch.setattr(ProcessContainer, "_reap", reap)


def _guardian_unsettled(monkeypatch):
    real = ProcessLifeline.finish

    def finish(self, *, released):
        real(self, released=released)
        return GUARDIAN_UNSETTLED

    monkeypatch.setattr(ProcessLifeline, "finish", finish)


# --- The container


@posix_only
def test_a_later_reap_repeats_the_first_answer_instead_of_an_empty_rescan(monkeypatch):
    container = ProcessContainer()
    with monkeypatch.context() as patch:
        patch.setattr(process_containment, "pids_with_env_marker", lambda *_a, **_k: None)
        first = container.reap()
    assert "could not be enumerated" in first
    assert container.reap() == first


def test_a_first_reap_that_never_finished_stays_unconfirmed(monkeypatch):
    container = ProcessContainer()

    def interrupted(_self):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(ProcessContainer, "_reap", interrupted)
        with pytest.raises(KeyboardInterrupt):
            container.reap()
    assert "did not finish" in container.reap()


@posix_only
def test_an_interrupt_while_stopping_the_watch_still_stops_the_tree_and_its_guardian(monkeypatch):
    real_stop, scans = OverloadWatch.stop, []

    def interrupted(watch):
        real_stop(watch)
        raise KeyboardInterrupt

    monkeypatch.setattr(OverloadWatch, "stop", interrupted)
    container = ProcessContainer(lifeline=True, overload=True)
    root = container.spawn([sys.executable, "-c", "import time; time.sleep(600)"])
    guardian = container._lifeline._proc
    with pytest.raises(KeyboardInterrupt):
        container.reap()
    root.wait(timeout=10)
    guardian.wait(timeout=10)
    real_reap = ProcessContainer._reap
    monkeypatch.setattr(ProcessContainer, "_reap", lambda self: scans.append(self) or real_reap(self))
    assert "did not finish" in container.reap() and "did not finish" in container.reap() and not scans


@posix_only
def test_a_finished_reap_is_repeated_without_a_second_scan(monkeypatch):
    container = ProcessContainer(lifeline=True, overload=True)
    root = container.spawn([sys.executable, "-c", "import time; time.sleep(600)"])
    assert container.reap() == "" and root.wait(timeout=10) is not None
    scans, real_reap = [], ProcessContainer._reap
    monkeypatch.setattr(ProcessContainer, "_reap", lambda self: scans.append(self) or real_reap(self))
    assert container.reap() == "" and not scans


def test_the_answer_of_spawns_own_reap_reaches_its_owner(monkeypatch):
    _popen_refusing(monkeypatch, "refused-workload")
    _first_reap_unconfirmed(monkeypatch)
    container = ProcessContainer()
    with pytest.raises(OSError):
        container.spawn([sys.executable, "-c", "pass", "refused-workload"])
    assert container.root_created is False and container.reap() == UNCONFIRMED


@posix_only
def test_a_guardian_started_before_a_refused_workload_is_settled_before_the_clean_answer(monkeypatch):
    started = []
    _popen_refusing(monkeypatch, "refused-workload", started=started)
    container = ProcessContainer(lifeline=True)
    with pytest.raises(OSError):
        container.spawn([sys.executable, "-c", "pass", "refused-workload"])
    [guardian] = started
    assert guardian.poll() is not None, "the guardian outlived the answer that says everything is gone"
    assert container.root_created is False and container.reap() == ""


@posix_only
def test_an_unsettled_guardian_is_the_answer_though_no_workload_ever_existed(monkeypatch):
    _popen_refusing(monkeypatch, "refused-workload")
    _guardian_unsettled(monkeypatch)
    container = ProcessContainer(lifeline=True)
    with pytest.raises(OSError):
        container.spawn([sys.executable, "-c", "pass", "refused-workload"])
    assert container.root_created is False and GUARDIAN_UNSETTLED in container.reap()


@posix_only
@pytest.mark.parametrize("fault", ["guardian_refused", "handshake_lost"])
def test_a_guardian_that_never_came_up_answers_through_the_owners_reap(monkeypatch, fault):
    started = []
    if fault == "guardian_refused":
        _popen_refusing(monkeypatch, "-B", started=started)
    else:
        _popen_refusing(monkeypatch, "never-refused", started=started)
        monkeypatch.setattr(process_lifeline, "select", SimpleNamespace(select=lambda *_a: ([], [], [])))
    container = ProcessContainer(lifeline=True)
    with pytest.raises((OSError, RuntimeError)):
        container.spawn([sys.executable, "-c", "pass"])
    assert all(proc.poll() is not None for proc in started) and len(started) == (fault == "handshake_lost")
    assert container.root_created is False and container.reap() == ""


def test_a_root_created_without_a_returned_handle_is_stopped_and_answered(monkeypatch):
    started = []
    _popen_refusing(monkeypatch, "never-refused", started=started)

    def refused(_watch):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(OverloadWatch, "start", refused)
    container = ProcessContainer(overload=True)
    with pytest.raises(RuntimeError):
        container.spawn([sys.executable, "-c", "import time; time.sleep(600)"])
    [root] = started
    assert container.root_created is True and root.poll() is not None and container.reap() == ""


# --- The hermetic gate


def _gate(tmp_path, monkeypatch, lane="pytest"):
    """A gate over a real tiny candidate whose first lane is refused its process."""
    repo = _tiny_repo(tmp_path, web=lane == "node")
    temp_root = (tmp_path / "preflight-root").resolve()
    temp_root.mkdir()
    _pin_preflight_root(monkeypatch, temp_root)
    if lane == "node":  # no working Node is needed: this lane never gets past its launch
        monkeypatch.setattr(preflight_node, "resolve_node", lambda: sys.executable)
        monkeypatch.setattr(preflight_node, "probe_node_version", lambda _node: "99.0.0")
    return repo, temp_root, _ctx(repo, tmp_path / "canonical")


def _registered(repo, temp_root) -> bool:
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=repo,
                             capture_output=True, text=True, check=True).stdout
    return f"worktree {(temp_root / 'repo').as_posix()}\n" in listing


def _refuse_lane(monkeypatch, lane):
    _popen_refusing(monkeypatch, *{"pytest": ("-m", "pytest"), "node": ("--test",)}[lane])


@posix_only
@pytest.mark.parametrize("lane", ["pytest", "node"])
def test_a_refused_lane_stays_red_and_releases_once_its_guardian_settled(tmp_path, monkeypatch, lane):
    repo, temp_root, ctx = _gate(tmp_path, monkeypatch, lane)
    _refuse_lane(monkeypatch, lane)

    verdict = pr.run_hermetic_pytest(repo, ctx=ctx)

    assert verdict and "hermetic preflight failed" in verdict and "RETAINED" not in verdict, verdict
    terminal = json.loads((pathlib.Path(ctx._preflight_test_evidence) / "result.json").read_text(encoding="utf-8"))
    assert (terminal["execution"], terminal["cleanup"], terminal["retention_reason"]) == ("not_run", "released", "")
    assert ctx._preflight_test_disclosure["outcome"] == "not_started" and not ctx._preflight_test_proof
    assert not temp_root.exists() and not _registered(repo, temp_root)


@posix_only
@pytest.mark.parametrize("lane", ["pytest", "node"])
@pytest.mark.parametrize("hold", ["first_reap_unconfirmed", "no_handle_unconfirmed", "guardian_unsettled",
                                  "explicit_keep", "gate_evidence_lost"])
def test_every_real_reason_still_holds_a_refused_lanes_tree(tmp_path, monkeypatch, lane, hold):
    repo, temp_root, ctx = _gate(tmp_path, monkeypatch, lane)
    reason = {"first_reap_unconfirmed": UNCONFIRMED, "no_handle_unconfirmed": UNCONFIRMED,
              "guardian_unsettled": GUARDIAN_UNSETTLED, "explicit_keep": "explicit preflight debug retention",
              "gate_evidence_lost": "gate evidence write failed: controlled evidence loss"}[hold]
    if hold == "no_handle_unconfirmed":  # a real root is created, stopped, and never handed over
        def refused(_watch):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(OverloadWatch, "start", refused)
    else:
        _refuse_lane(monkeypatch, lane)
    if "unconfirmed" in hold:
        _first_reap_unconfirmed(monkeypatch)
    elif hold == "guardian_unsettled":
        _guardian_unsettled(monkeypatch)
    elif hold == "explicit_keep":
        monkeypatch.setenv("OUROBOROS_PREFLIGHT_KEEP", "1")
    elif hold == "gate_evidence_lost":
        real = test_evidence.write_record

        def lost(path, value):
            if pathlib.Path(path).name == "result.json":
                raise OSError("controlled evidence loss")
            real(path, value)

        monkeypatch.setattr(test_evidence, "write_record", lost)
    try:
        verdict = pr.run_hermetic_pytest(repo, ctx=ctx)
        assert verdict and "hermetic preflight failed" in verdict, verdict
        assert not ctx._preflight_test_proof
        assert (temp_root / "repo").is_dir() and _registered(repo, temp_root)
        assert reason in (temp_root / RETENTION_MARKER).read_text(encoding="utf-8")
        if hold != "gate_evidence_lost":
            terminal = json.loads((pathlib.Path(ctx._preflight_test_evidence) / "result.json").read_text(encoding="utf-8"))
            assert terminal["cleanup"] == "retained" and reason in terminal["retention_reason"]
    finally:
        retire_synthetic_gate_hold(repo, temp_root, reason)


def test_a_lane_whose_evidence_never_opened_keeps_its_armed_reason(tmp_path, monkeypatch):
    repo, temp_root, ctx = _gate(tmp_path, monkeypatch)

    class NoEvidence:
        def __init__(self, *_a):
            raise OSError("controlled evidence loss")

    monkeypatch.setattr(test_evidence, "FileOutput", NoEvidence)
    reason = "the parallel pass did not return, so its teardown is unconfirmed"
    try:
        verdict = pr.run_hermetic_pytest(repo, ctx=ctx)
        assert verdict and "RETAINED" in verdict and (temp_root / "repo").is_dir(), verdict
    finally:
        retire_synthetic_gate_hold(repo, temp_root, reason)


# --- Retiring a hold the test itself manufactured


def _held_gate(tmp_path, marker: bytes):
    """A real worktree registered under a gate root whose marker holds ``marker``."""
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    temp_root = (tmp_path / "preflight-root").resolve()
    temp_root.mkdir()
    subprocess.run(["git", "worktree", "add", "--detach", str(temp_root / "repo"), "HEAD"],
                   cwd=repo, check=True, capture_output=True)
    (temp_root / RETENTION_MARKER).write_bytes(marker)
    return repo, temp_root


SYNTHETIC = f"{UNCONFIRMED}\nhermetic preflight: {UNCONFIRMED}\n".encode()


def test_a_synthetic_gate_hold_is_retired_with_its_worktree(tmp_path):
    repo, temp_root = _held_gate(tmp_path, SYNTHETIC)
    assert retire_synthetic_gate_hold(repo, temp_root, UNCONFIRMED) is True
    assert not (temp_root / RETENTION_MARKER).exists() and not (temp_root / "repo").exists()
    assert not _registered(repo, temp_root)


@pytest.mark.parametrize("marker", [
    pytest.param(SYNTHETIC + b"pid 77 is still alive\n", id="mixed"),
    pytest.param(b"pid 77 is still alive\n", id="changed"),
    pytest.param(b"\xff\xfe not a reason\n", id="unreadable"),
    pytest.param(SYNTHETIC, id="removal_failed"),
])
def test_a_hold_the_test_cannot_prove_synthetic_keeps_marker_and_worktree(tmp_path, marker):
    repo, temp_root = _held_gate(tmp_path, marker)
    worktree = temp_root / "repo"
    removal_failed = marker == SYNTHETIC
    if removal_failed:  # --force once cannot remove a locked worktree
        subprocess.run(["git", "worktree", "lock", str(worktree)], cwd=repo, check=True, capture_output=True)
    try:
        assert retire_synthetic_gate_hold(repo, temp_root, UNCONFIRMED) is False
        assert (temp_root / RETENTION_MARKER).read_bytes() == marker
        assert worktree.is_dir() and _registered(repo, temp_root)
    finally:  # everything above was manufactured here
        subprocess.run(["git", "worktree", "unlock", str(worktree)], cwd=repo, capture_output=True)
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, capture_output=True)
        (temp_root / RETENTION_MARKER).unlink(missing_ok=True)


# --- safe_test


def _launcher(tmp_path, *options, command=None):
    evidence = tmp_path / "evidence"
    return subprocess.run(
        [sys.executable, "-I", "-S", str(REPO / "scripts" / "safe_test.py"), "--temp-parent", str(tmp_path),
         "--evidence-dir", str(evidence), *options, "--", *(command or [str(tmp_path / "no-such-command")])],
        cwd=REPO, capture_output=True, text=True, timeout=120), evidence


def _only_result(evidence):
    [run] = [path for path in pathlib.Path(evidence).glob("run-*") if (path / "result.json").is_file()]
    return run, json.loads((run / "result.json").read_text(encoding="utf-8"))


def test_the_launcher_reports_a_missing_command_red_and_releases_its_root(tmp_path):
    result, evidence = _launcher(tmp_path)
    run, record = _only_result(evidence)
    final, root = result.stderr.splitlines()[-1].split(" ", 1)
    assert result.returncode == 2 and final == "SAFE_TEST_RELEASED" and not pathlib.Path(root).exists(), result.stderr
    assert "SAFE_TEST_ERROR FileNotFoundError" in result.stderr
    assert record["execution"] == "not_run" and "FileNotFoundError" in record["launch_error"]
    assert (record["returncode"], record["evidence_error"], record["containment_error"]) == (2, "", "")
    assert (record["retention_reason"], record["cleanup"]["retained"]) == ("", [])
    assert "FileNotFoundError" in json.loads(next(run.glob("command-*/result.json")).read_text())["launch_error"]


def test_the_launchers_explicit_keep_retains_a_missing_commands_root(tmp_path):
    result, evidence = _launcher(tmp_path, "--keep")
    _run, record = _only_result(evidence)
    final, root = result.stderr.splitlines()[-1].split(" ", 1)
    try:
        assert result.returncode == 2 and final == "SAFE_TEST_RETAINED", result.stderr
        assert record["retention_reason"] == "debug retention" and record["execution"] == "not_run"
    finally:
        retire_synthetic_retention(root, "debug retention")


def _stage_two(tmp_path, monkeypatch):
    root, checkout = tmp_path / "run", tmp_path / "checkout"
    checkout.mkdir()
    test_environment.isolated_environment(root, checkout)
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(root))
    return root, checkout, test_evidence.diagnostic_directory(tmp_path / "evidence", "launch")


COMMAND = [sys.executable, "-c", "import time; time.sleep(600)", "refused-workload"]


def _no_handle(monkeypatch):
    """The real root is created, then the container cannot hand it over."""
    def attach(*_a):
        raise RuntimeError("controlled custody failure")

    monkeypatch.setattr(ProcessLifeline, "attach", attach)


@posix_only
@pytest.mark.parametrize("fault,execution", [("refused", "not_run"), ("no_handle", "attempted")])
def test_stage_two_releases_a_failed_launch_once_its_reap_is_clean(tmp_path, monkeypatch, fault, execution):
    root, checkout, parent = _stage_two(tmp_path, monkeypatch)
    if fault == "refused":
        _popen_refusing(monkeypatch, "refused-workload")
    else:
        _no_handle(monkeypatch)
    assert safe_test_supervisor.main(root, checkout, COMMAND, evidence_parent=parent) == 2
    _run, record = _only_result(parent)
    assert (record["execution"], record["evidence_error"], record["retention_reason"]) == (execution, "", "")
    assert record["launch_error"] and not root.exists()


@posix_only
@pytest.mark.parametrize("hold,reason", [
    ("first_reap_unconfirmed", UNCONFIRMED), ("no_handle_unconfirmed", UNCONFIRMED),
    ("guardian_unsettled", GUARDIAN_UNSETTLED), ("keep", "debug retention"),
    ("evidence_lost", "evidence/cleanup incomplete: controlled evidence loss"),
])
def test_every_real_reason_still_holds_a_failed_launchs_scratch(tmp_path, monkeypatch, hold, reason):
    root, checkout, parent = _stage_two(tmp_path, monkeypatch)
    if hold == "no_handle_unconfirmed":
        _no_handle(monkeypatch)
    else:
        _popen_refusing(monkeypatch, "refused-workload")
    if "unconfirmed" in hold:
        _first_reap_unconfirmed(monkeypatch)
    elif hold == "guardian_unsettled":
        _guardian_unsettled(monkeypatch)
    elif hold == "evidence_lost":
        real = safe_test_supervisor.write_record

        def lost(path, value):
            if pathlib.Path(path).name == "result.json":
                raise OSError("controlled evidence loss")
            real(path, value)

        monkeypatch.setattr(safe_test_supervisor, "write_record", lost)
    try:
        assert safe_test_supervisor.main(root, checkout, COMMAND, evidence_parent=parent, keep=hold == "keep") == 2
        assert root.is_dir() and reason in (root / RETENTION_MARKER).read_text(encoding="utf-8")
        if hold != "evidence_lost":
            _run, record = _only_result(parent)
            assert reason in record["retention_reason"] and record["launch_error"]
    finally:
        retire_synthetic_retention(root, reason)


def test_a_source_gap_inside_the_scratch_holds_a_failed_launch(tmp_path, monkeypatch):
    root, _unused, parent = _stage_two(tmp_path, monkeypatch)
    test_environment.declare_scratch(root, "body-process")
    source = root / "source"
    source.mkdir()
    _popen_refusing(monkeypatch, "refused-workload")

    def capture_gap(*_args):
        raise OSError("controlled capture failure")

    monkeypatch.setattr(safe_test_supervisor, "source_evidence", capture_gap)
    gap = "test source evidence incomplete: recipe not recorded (OSError: controlled capture failure)"
    try:
        assert safe_test_supervisor.main(root, source, COMMAND, evidence_parent=parent) == 2
        _run, record = _only_result(parent)
        assert source.is_dir() and record["source_gap"] and "retention marker" in record["retention_reason"]
        assert gap in (root / RETENTION_MARKER).read_text(encoding="utf-8")
    finally:
        retire_synthetic_retention(root, gap, f"retention marker or unreadable subtree: {root / RETENTION_MARKER}")
