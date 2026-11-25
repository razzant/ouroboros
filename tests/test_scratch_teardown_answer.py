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

from ouroboros import preflight_runner as pr, process_containment, process_lifeline, safe_test_supervisor
from ouroboros import session_scratch, test_environment, test_evidence
from ouroboros.process_containment import ProcessContainer
from ouroboros.process_lifeline import ProcessLifeline
from ouroboros.process_overload import OverloadWatch
from ouroboros.test_environment import RETENTION_MARKER
from tests._shared import retire_synthetic_gate_hold, retire_synthetic_retention
from tests.test_preflight_receipts import _ctx, _tiny_repo, gate_env, receipt_python  # noqa: F401 -- the gate's fixtures
from tests.test_preflight_runner import _make_repo
from ouroboros import preflight_node
from tests.test_preflight_runner import _pin_preflight_root

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
@pytest.mark.parametrize("fault", ["guardian_refused", "handshake_lost", "handshake_eof", "handshake_wrong", "selector_failed"])
def test_a_guardian_that_never_came_up_answers_through_the_owners_reap(monkeypatch, fault):
    started, waits, readiness = [], [], []
    if fault == "guardian_refused":
        _popen_refusing(monkeypatch, "-B", started=started)
    elif fault in {"handshake_eof", "handshake_wrong"}:
        spawn = ProcessContainer.spawn

        def short_guardian(container, argv, **kwargs):
            guardian = any("_guardian_main" in str(arg) for arg in argv)
            if guardian:
                code = "pass" if fault == "handshake_eof" else "import sys; sys.stdout.buffer.write(b'X')"
                argv = [sys.executable, "-c", code]
            proc = spawn(container, argv, **kwargs)
            if guardian:
                started.append(proc)
            return proc

        monkeypatch.setattr(ProcessContainer, "spawn", short_guardian)
        selector_type = process_lifeline.selectors.DefaultSelector

        class ObservedSelector(selector_type):
            def select(self, timeout=None):
                events = super().select(timeout)
                readiness.append(bool(events))
                return events

        monkeypatch.setattr(process_lifeline, "selectors", SimpleNamespace(
            DefaultSelector=ObservedSelector, EVENT_READ=process_lifeline.selectors.EVENT_READ,
        ))
    else:
        from contextlib import nullcontext

        def wait(timeout):
            waits.append(timeout)
            if fault == "selector_failed":
                raise OSError(errno.EIO, "readiness wait failed")
            return []

        _popen_refusing(monkeypatch, "never-refused", started=started)
        waiter = SimpleNamespace(register=lambda *_a: None, select=wait)
        monkeypatch.setattr(process_lifeline, "selectors", SimpleNamespace(
            DefaultSelector=lambda: nullcontext(waiter), EVENT_READ=1,
        ))
    container = ProcessContainer(lifeline=True)
    with pytest.raises((OSError, RuntimeError)):
        container.spawn([sys.executable, "-c", "pass"])
    assert all(proc.poll() is not None for proc in started) and len(started) == (fault != "guardian_refused")
    assert container.root_created is False and container.reap() == ""
    if fault in {"handshake_lost", "selector_failed"}:
        assert waits == [process_lifeline._HANDSHAKE_SEC] == [10.0]
    if fault in {"handshake_eof", "handshake_wrong"}:
        assert readiness == [True], "the protocol failed after readiness, not the timeout"


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


# --- The hermetic gate


def _registered(repo, temp_root) -> bool:
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=repo,
                             capture_output=True, text=True, check=True).stdout
    return f"worktree {(temp_root / 'repo').as_posix()}\n" in listing


def _refuse_lane(monkeypatch, lane):
    _popen_refusing(monkeypatch, *{"pytest": ("-m", "pytest"), "node": ("--test",)}[lane])


@posix_only
@pytest.mark.parametrize("lane", ["node"])  # a refused pytest lane is now refused after the warm executed
def test_a_refused_lane_stays_red_and_releases_once_its_guardian_settled(tmp_path, monkeypatch, lane):
    repo, temp_root, ctx = _gate(tmp_path, monkeypatch, lane)
    _refuse_lane(monkeypatch, lane)

    verdict = pr.run_hermetic_pytest(repo, ctx=ctx)

    assert verdict and "hermetic preflight failed" in verdict and "RETAINED" not in verdict, verdict
    terminal = json.loads((pathlib.Path(ctx._preflight_test_evidence) / "result.json").read_text(encoding="utf-8"))
    assert (terminal["execution"], terminal["cleanup"], terminal["retention_reason"]) == ("not_run", "released", "")
    assert (terminal["kept"], terminal["containment_tokens"]) == ([], []), "a released run records nothing kept"
    assert ctx._preflight_test_disclosure["outcome"] == "not_started" and not ctx._preflight_test_proof
    assert not temp_root.exists() and not _registered(repo, temp_root)


@posix_only
# A refused pytest lane now follows the bytecode warm: its unconfirmed first reap and unsettled guardian are
# reported as the warm's containment failure, and with lost gate evidence it leaves a retention in tmp_path that
# the hold's retirement does not cover. Those three holds keep their node row only.
@pytest.mark.parametrize("hold,lane", [("first_reap_unconfirmed", "node"), ("no_handle_unconfirmed", "pytest"),
                                       ("no_handle_unconfirmed", "node"), ("guardian_unsettled", "node"),
                                       ("explicit_keep", "pytest"), ("explicit_keep", "node"),
                                       ("gate_evidence_lost", "node")])
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
            assert terminal["kept"] == [{"path": str(temp_root), "reasons": [terminal["retention_reason"].split("\n")[0]]}]
            assert [token.split("_")[:3] for token in terminal["containment_tokens"]] == [["OURO", "PROC", "CONTAINER"]]
    finally:
        retire_synthetic_gate_hold(repo, temp_root, reason)


@pytest.mark.parametrize("in_worktree", [False, True], ids=["test-folder", "worktree"])
def test_a_marker_beneath_a_proven_gate_root_keeps_its_folder_and_is_held_by_its_tasks_scratch_record(
        tmp_path, monkeypatch, in_worktree):
    repo = _make_repo(tmp_path, {"tests/test_plain.py": "def test_ok():\n    assert True\n"})
    parent = (tmp_path / "temp").resolve()
    monkeypatch.setattr(test_environment, "default_temp_parent", lambda: parent)
    parent.mkdir()
    monkeypatch.delenv("OUROBOROS_TEST_TEMP_ROOT", raising=False)
    run = {}
    root = test_environment.gate_root(evidence=run)
    container, worktree = root.parent, root / "repo"
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=repo, check=True, capture_output=True)
    (root / "b0").mkdir()
    test_environment.declare_scratch(root / "b0", "pytest-basetemp")
    test, other = root / "b0" / "t" / "popen-gw0" / "test_held0", root / "b0" / "t" / "popen-gw0" / "test_other0"
    other.mkdir(parents=True)
    held = worktree if in_worktree else test
    (held / "deep").mkdir(parents=True)
    reason = "1 process tree(s) started from this checkout were never proven gone"
    assert test_environment.retain_tree(held / "deep", reason)
    evidence, data = tmp_path / "evidence", (tmp_path / "data").resolve()
    evidence.mkdir()
    ctx = SimpleNamespace(task_id="t-kept", drive_root=str(data))
    tokens = ["OURO_PROC_CONTAINER_lane"]
    try:
        test_evidence.finish_gate({**run, "path": evidence, "cleanup": (repo, root, worktree, True, ""),
                                   "containment_tokens": [*tokens, ""]}, None, "post_commit", pr._release_hermetic_tree, ctx=ctx)
        result = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
        assert (result["cleanup"], result["retention_reason"]) == ("retained", reason)
        assert held.is_dir() and not other.exists(), "the folder stays, and only the folder"
        assert worktree.exists() is _registered(repo, root) is in_worktree, "a worktree goes with its registration, or stays with it"
        assert not (root / RETENTION_MARKER).exists(), "one nested marker kept the whole gate root"
        drive = data / "task_drives" / ctx.task_id
        row = json.loads((drive / "session_scratch_owners" / container.name / "scratch.json").read_text(encoding="utf-8"))
        assert row["schema_version"] == 2 and row["path"] == str(container)
        assert row["kept"] == result["kept"] == [{"path": str(held), "reasons": [reason]}]
        assert row["containment_tokens"] == result["containment_tokens"] == tokens
        assert (data / "state" / "session_scratch" / ctx.task_id / f"{container.name}.json").is_file()
        # The reader every task's scratch has: the marker holds the tree, and once it is gone the tree goes.
        assert session_scratch.release_session_scratch(drive) and held.is_dir()
        assert retire_synthetic_retention(held / "deep", reason)
        assert not session_scratch.release_session_scratch(drive) and not root.exists()
    finally:
        retire_synthetic_retention(held / "deep", reason)


def _file_gate(tmp_path, monkeypatch):
    """The real fresh allocator and finisher, but synthetic files and no worktree/process launches."""
    parent = (tmp_path / "temp").resolve()
    parent.mkdir()
    monkeypatch.setattr(test_environment, "default_temp_parent", lambda: parent)
    monkeypatch.delenv("OUROBOROS_TEST_TEMP_ROOT", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_KEEP", raising=False)
    run = {"path": tmp_path / "evidence", "containment_tokens": ["synthetic-lane", ""]}
    run["path"].mkdir()
    root = test_environment.gate_root(evidence=run)
    repo, worktree = tmp_path / "repo", root / "repo"
    repo.mkdir()
    worktree.mkdir()
    run["cleanup"] = (repo, root, worktree, False, "")
    ctx = SimpleNamespace(task_id="gate-task", drive_root=str(tmp_path / "data"))
    record = (pathlib.Path(ctx.drive_root) / "task_drives" / ctx.task_id / "session_scratch_owners"
              / root.parent.name / "scratch.json")
    return run, ctx, root, record


@pytest.mark.parametrize("hold", ["nested", "root", "debug", "release-error"])
def test_new_gate_carries_its_issued_nonce_through_finish_and_the_real_custody_reader(tmp_path, monkeypatch, hold):
    from ouroboros import owner_keep, task_custody, work_domain
    from ouroboros.workspace_preflight import release_finished_scratch
    from tests._store_shared import seed_result

    run, ctx, root, record = _file_gate(tmp_path, monkeypatch)
    container, identity = root.parent, dict(run["scratch_ownership"])
    held = root / "held" if hold == "nested" else root
    held.mkdir(exist_ok=True)
    (held / "payload").write_bytes(b"NEW-GATE")
    reason = f"synthetic gate hold: {hold}"
    if hold != "release-error":
        test_environment.retain_tree(held, reason)
    if hold == "debug":
        monkeypatch.setenv("OUROBOROS_PREFLIGHT_KEEP", "1")
    release = pr._release_hermetic_tree
    if hold == "release-error":
        def release(*_args):
            raise OSError(reason)
    test_evidence.finish_gate(run, None, "pre_commit", release, ctx=ctx)
    row = json.loads(record.read_text())
    result = json.loads((run["path"] / "result.json").read_text())
    assert {key: row[key] for key in identity} == identity
    assert row["kept"] == result["kept"] and row["containment_tokens"] == ["synthetic-lane"]
    before = record.read_bytes(), (container / session_scratch.SCRATCH_MARKER).read_bytes()
    lstat = work_domain.lstat

    def renumbered(path):
        seen = lstat(path)
        return SimpleNamespace(st_mode=seen.st_mode, st_dev=seen.st_dev + 1, st_ino=seen.st_ino) if path == container else seen

    monkeypatch.setattr(work_domain, "lstat", renumbered)
    assert session_scratch.recorded_scratch(record) == container
    # A repeated finish re-notes this same record, never replacing its nonce or former facts.
    test_evidence.finish_gate(run, None, "pre_commit", release, ctx=ctx)
    assert (record.read_bytes(), (container / session_scratch.SCRATCH_MARKER).read_bytes()) == before
    data = pathlib.Path(ctx.drive_root)
    seed_result(data, ctx.task_id, {"status": "completed"})
    monkeypatch.setattr(session_scratch, "task_hold", lambda *_a: "")  # no process census: synthetic ended owner
    owner_keep.set_owner_keep(data, ctx.task_id, keep=True, source={"basis": "test"})
    assert [h["class"] for h in release_finished_scratch(data, lambda _t: False)["holds"]] == ["owner_keep"]
    owner_keep.set_owner_keep(data, ctx.task_id, keep=False, source={"basis": "test"})
    assert release_finished_scratch(data, lambda _t: False)["released"]
    retired = task_custody.remove_custody_trash(data)
    assert [h["class"] for h in retired["holds"]] == ["producer_retained"]
    assert (held / "payload").read_bytes() == b"NEW-GATE"
    # These exact markers were made above, never old retained data or an unknown process.
    for marker in (root / RETENTION_MARKER, held / RETENTION_MARKER, container / RETENTION_MARKER):
        marker.unlink(missing_ok=True)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not container.exists()


def test_a_released_new_gate_removes_its_workspace_and_verified_outside_container(tmp_path, monkeypatch):
    run, ctx, root, record = _file_gate(tmp_path, monkeypatch)
    test_evidence.finish_gate(run, None, "pre_commit", pr._release_hermetic_tree, ctx=ctx)
    assert not root.parent.exists() and not record.exists()
    assert json.loads((run["path"] / "result.json").read_text())["cleanup"] == "released"


@pytest.mark.parametrize("fault", ["carrier-lost", "nonce-changed", "record-write"])
def test_an_unknown_or_failed_new_gate_writer_never_invents_ownership_or_removes_the_retained_tree(
        tmp_path, monkeypatch, capsys, fault):
    run, ctx, root, record = _file_gate(tmp_path, monkeypatch)
    container = root.parent
    if fault != "carrier-lost":
        test_environment.retain_tree(root, "synthetic hold")
    (root / "payload").write_bytes(b"KEPT")
    if fault == "carrier-lost":
        run.pop("scratch_ownership")
    elif fault == "nonce-changed":
        (container / session_scratch.SCRATCH_MARKER).write_bytes(b"f" * 32 + b"\n")
    else:
        write = test_evidence.write_record

        def failed(path, value):
            if pathlib.Path(path).name == "scratch.json":
                raise OSError("synthetic record write failure")
            return write(path, value)
        monkeypatch.setattr(test_evidence, "write_record", failed)
    test_evidence.finish_gate(run, None, "pre_commit", pr._release_hermetic_tree, ctx=ctx)
    assert not record.exists() and (root / "payload").read_bytes() == b"KEPT"
    assert "PREFLIGHT_EVIDENCE_INCOMPLETE" in capsys.readouterr().err
    assert not list((tmp_path / "data").rglob("scratch.json")), "no legacy fallback or root adoption"
    result = json.loads((run["path"] / "result.json").read_text())
    assert result["diagnostics_incomplete"] and result["evidence_errors"]
    reason = result["retention_reason"]
    assert reason == "synthetic hold" or reason.startswith("gate scratch identity unconfirmed:")
    assert retire_synthetic_retention(root, "synthetic hold", reason, "hermetic preflight: synthetic hold")


@pytest.mark.parametrize("mismatch", [False, True])
def test_a_former_kept_gate_record_stays_v1_byte_for_byte_and_still_holds_a_device_mismatch(
        tmp_path, monkeypatch, mismatch):
    parent = (tmp_path / "temp").resolve()
    root = parent / "obs-old-gate"
    root.mkdir(parents=True)
    (root / "payload").write_bytes(b"LEGACY")
    monkeypatch.setattr(test_environment, "default_temp_parent", lambda: parent)
    data = tmp_path / "data"
    ctx = SimpleNamespace(task_id="old-task", drive_root=str(data))
    record = data / "task_drives" / ctx.task_id / "session_scratch_owners" / root.name / "scratch.json"
    record.parent.mkdir(parents=True)
    seen = root.lstat()
    row = {"schema_version": 1, "path": str(root), "identity": [seen.st_dev + mismatch, seen.st_ino],
           "kept": [{"path": str(root), "reasons": ["former facts"]}], "containment_tokens": ["old-token"]}
    test_evidence.write_record(record, row)  # the former writer's exact shape is the negative control
    before = record.read_bytes()
    if mismatch:
        with pytest.raises(session_scratch.ScratchIdentityMismatch):
            test_evidence.record_kept_scratch(ctx, root, [], ["new-token"])
        assert session_scratch.release_session_scratch(record.parents[2])[0]["class"] == "identity_unverified"
    else:
        test_evidence.record_kept_scratch(ctx, root, [], ["new-token"])
        assert session_scratch.recorded_scratch(record) == root
    assert record.read_bytes() == before and (root / "payload").read_bytes() == b"LEGACY"
    assert not (root / session_scratch.SCRATCH_MARKER).exists(), "no opportunistic v1 migration"


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


def test_a_green_command_whose_child_wrote_bytecode_still_releases_its_root(tmp_path):
    """The launcher declares the bytecode prefix it creates: a child that compiles a module under
    `PYTHONPYCACHEPREFIX` (test processes, nested runners) leaves nothing undeclared, so the root is released."""
    code = ("import os, pathlib; pyc = pathlib.Path(os.environ['PYTHONPYCACHEPREFIX']) / 'pkg' / 'mod.cpython-310.pyc'; "
            "pyc.parent.mkdir(parents=True); pyc.write_bytes(b'0'); print(os.environ['PYTHONPYCACHEPREFIX'])")
    result, evidence = _launcher(tmp_path, command=[sys.executable, "-c", code])
    _run, record = _only_result(evidence)
    final, root = result.stderr.splitlines()[-1].split(" ", 1)
    assert result.returncode == 0 and final == "SAFE_TEST_RELEASED" and not pathlib.Path(root).exists(), result.stderr
    assert result.stdout.strip().splitlines()[-1].startswith(root) and record["retention_reason"] == ""


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
    assert (record["kept"], record["containment_tokens"]) == ([], []), "a released run records nothing kept"


@posix_only
@pytest.mark.parametrize("hold,reason", [
    ("first_reap_unconfirmed", UNCONFIRMED), ("guardian_unsettled", GUARDIAN_UNSETTLED), ("evidence_lost", "evidence/cleanup incomplete: controlled evidence loss"),
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
            assert record["kept"] == [{"path": str(root), "reasons": [record["retention_reason"].split("\n")[0]]}]
            assert [token.split("_")[:3] for token in record["containment_tokens"]] == [["OURO", "PROC", "CONTAINER"]]
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
