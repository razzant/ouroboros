"""Process custody of a prepared Update's readiness lanes (``update_checks``): each lane runs as the commit gate runs a
pass (``preflight_runner._execute_pytest_pass``: a ``ProcessContainer`` with its lifeline and overload watch, its
streams regular files), so nothing it started outlives it, a descendant in its own session included, and what it
printed survives its deadline; a tree its reap could not prove gone leaves the run incomplete and its process root
marked; the process root is removed only when nothing under it is marked retained (``test_environment``), otherwise
kept and named in the checks' record. The tracked owner's own deadline (``shell_process._tracked_subprocess_run``)
stays the answer when its kill is not yet observed."""
from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time
import uuid

import pytest

from supervisor import update_checks
from tests import test_update_checks
from tests.test_update_checks import _outcome, _toy_workspace

harness = test_update_checks.harness  # the same recorded outcomes and stand-ins

posix_only = pytest.mark.skipif(os.name == "nt", reason="session leaders, SIGTERM and ps are POSIX; Windows keeps "
                                                        "its kill-on-close Job, unproven here")


def _ours(pid: int, marker: str) -> bool:
    """``pid`` is alive and is the process this test started (its command line carries ``marker``)."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    seen = subprocess.run(["ps", "-o", "stat=,command=", "-p", str(pid)], capture_output=True, text=True)
    return marker in seen.stdout and not seen.stdout.strip().startswith("Z")


def _lane_env(tmp_path, ws):
    from ouroboros.test_environment import isolated_environment
    from ouroboros.test_parallelism import NESTED_WORKERS_ENV

    return {**isolated_environment(tmp_path / "iso", ws), NESTED_WORKERS_ENV: "1"}


@posix_only
def test_a_descendant_in_its_own_session_does_not_outlive_its_lane(tmp_path):
    """A test starts a sleeper in its own session (``start_new_session``) and returns; it reparents to init once pytest
    exits, outside pytest's process group, and keeps the lane's stdout open. The lane ends complete and green within
    its budget, and nothing it started is alive once ``run_tests`` returns."""
    marker = f"custody-sleeper-{uuid.uuid4().hex}"
    pidfile = tmp_path / "sleeper.pid"
    ws, commit = _toy_workspace(tmp_path, {"tests/test_detached.py": (
        "import pathlib, subprocess, sys\n\n\ndef test_leaves_a_session_leader():\n"
        f"    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)', {marker!r}],\n"
        "                             start_new_session=True)\n"
        f"    pathlib.Path({str(pidfile)!r}).write_text(str(child.pid))\n")})
    pid = 0
    try:
        run = update_checks.run_tests(ws, commit, ["tests/test_detached.py"], tmp_path / "scratch",
                                      _lane_env(tmp_path, ws), deadline=time.monotonic() + 20)
        pid = int(pidfile.read_text())
        survived = _ours(pid, marker)
        assert not survived, f"the lane's descendant {pid} outlived it"
        assert run["complete"] and run["ok"] and [(lane["lane"], lane["returncode"]) for lane in run["lanes"]] == [
            ("parallel", 0), ("serial", 5)]
    finally:
        if pid and _ours(pid, marker):  # only the sleeper this test started
            os.kill(pid, signal.SIGKILL)


@posix_only
def test_the_owners_deadline_and_partial_output_survive_a_kill_not_yet_observed(tmp_path, monkeypatch):
    """A child that ignores SIGTERM and whose tree kill has not landed (stood in for by a kill that does nothing) is
    still running when the owner's 5 s exit wait ends: the deadline it was given and what it printed before it stay the
    answer, never the exit wait's own timeout; the child stays tracked until its exit is observed."""
    from ouroboros.tools import shell_process

    monkeypatch.setattr(shell_process, "_kill_process_group", lambda proc: None)
    script = ("import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "print('before the deadline', flush=True); time.sleep(7)")
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as cut:
        shell_process._tracked_subprocess_run([sys.executable, "-c", script], cwd=str(tmp_path),
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
    live = [proc for proc in shell_process._active_subprocesses.copy() if proc.args[-1] == script]
    try:
        assert time.monotonic() - started >= 5  # the exit wait ran and ended without the exit
        assert cut.value.timeout == 1 and b"before the deadline" in (cut.value.stdout or b"")
        assert len(live) == 1 and live[0].poll() is None  # still tracked: Panic and the generation stop can name it
    finally:
        for proc in live:  # the toy exits by itself within seconds; reap it and drop it from the registry
            proc.wait(timeout=15)
            shell_process._active_subprocesses.discard(proc)


@posix_only
def test_a_lane_cut_at_its_deadline_keeps_what_it_printed_even_when_its_test_ignores_sigterm(tmp_path):
    """Real pytest through the lane: a test that ignores SIGTERM and outlives the budget. The lane ends at the deadline
    (124, ``partial``), what pytest printed up to the cut is kept in its file with the deadline stated after its
    stderr, and the test's process is gone."""
    pidfile = tmp_path / "stubborn.pid"
    ws, commit = _toy_workspace(tmp_path, {"tests/test_stubborn.py": (
        "import os, pathlib, signal, time\n\n\n"
        + "".join(f"def test_{n}_passes():\n    assert True\n\n\n" for n in range(5))
        + "def test_ignores_term():\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"    pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()))\n    time.sleep(120)\n")})
    run = update_checks.run_tests(ws, commit, ["tests/test_stubborn.py"], tmp_path / "scratch",
                                  _lane_env(tmp_path, ws), deadline=time.monotonic() + 8)

    import json
    folder = pathlib.Path(run["evidence"])
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert run["complete"] is False and [(lane["lane"], lane["returncode"], lane["output"])
                                         for lane in manifest["lanes"]] == [
        ("parallel", 124, "partial"), ("serial", 124, "none")]
    stdout = (folder / "parallel.stdout.txt").read_text(encoding="utf-8")
    stderr = (folder / "parallel.stderr.txt").read_text(encoding="utf-8")
    assert "....." in stdout, stdout  # pytest's progress up to the hung case, printed before the cut
    assert stderr.rstrip("\n").splitlines()[-1].startswith("timed out after ")
    assert not _ours(int(pidfile.read_text()), "")  # the test's process is gone


def _root_seen(monkeypatch):
    """Stands in for the environments, keeping the process root ``selected_tests`` made."""
    seen = []
    monkeypatch.setattr(update_checks, "_environments", lambda row, root, evidence: seen.append(root) or (
        (sys.executable, {"PYTHONUSERBASE": "g"}), (sys.executable, {"PYTHONUSERBASE": "s"})))
    return seen


def test_a_marked_process_root_is_kept_and_named_never_deleted(harness, monkeypatch):
    """A lane's conftest or nested runner that could not prove its processes gone marks the root retained
    (``test_environment.retain_tree``): the checks keep it and name it with why, their verdict unchanged."""
    from ouroboros.test_environment import RETENTION_MARKER, retain_tree

    calls, outcomes, row, scratch = harness
    outcomes["wprime"] = _outcome(rc=0)
    seen, fake = _root_seen(monkeypatch), update_checks.run_tests

    def marks(*a, **kw):
        (seen[0] / "tmp" / "nested").mkdir(parents=True, exist_ok=True)
        retain_tree(seen[0] / "tmp" / "nested", "nested preflight: still alive after a best-effort kill: 4242")
        return fake(*a, **kw)

    monkeypatch.setattr(update_checks, "run_tests", marks)
    try:
        verdict = update_checks.selected_tests(row, scratch)

        root = pathlib.Path(verdict["process_root"])
        assert verdict["ok"] is True and root == seen[0] and root.is_dir()
        assert (root / "tmp" / "nested" / RETENTION_MARKER).is_file()
        kept = verdict["process_root_release"]
        assert kept["state"] == "retained" and kept["path"] == str(root)
        assert kept["markers"] == [str(root / "tmp" / "nested" / RETENTION_MARKER)]
        assert "still alive after a best-effort kill: 4242" in kept["reason"]
    finally:
        import shutil
        if seen:
            shutil.rmtree(seen[0], ignore_errors=True)  # this test's own fixture root


def test_an_unmarked_process_root_is_removed_and_says_so(harness, monkeypatch):
    calls, outcomes, row, scratch = harness
    outcomes["wprime"] = _outcome(rc=0)
    seen = _root_seen(monkeypatch)

    verdict = update_checks.selected_tests(row, scratch)

    assert verdict["ok"] is True and not seen[0].exists()
    assert verdict["process_root_release"] == {"state": "removed", "path": str(seen[0])}


def test_a_root_marked_by_a_run_that_raised_is_kept(harness, monkeypatch):
    """The run's own exception stays the answer; the marked root is kept all the same (logged, the record lost with
    the exception)."""
    from ouroboros.test_environment import retain_tree

    calls, outcomes, row, scratch = harness
    seen = _root_seen(monkeypatch)

    def breaks(*_a, **_kw):
        retain_tree(seen[0], "a lane's tree was not proven gone")
        raise OSError("the run broke")

    monkeypatch.setattr(update_checks, "run_tests", breaks)
    try:
        with pytest.raises(OSError, match="the run broke"):
            update_checks.selected_tests(row, scratch)
        assert seen[0].is_dir()
    finally:
        import shutil
        shutil.rmtree(seen[0], ignore_errors=True)


@posix_only
def test_a_tree_its_reap_could_not_prove_gone_leaves_the_run_incomplete_and_marks_its_root(tmp_path, monkeypatch):
    """The container's reap names a member it could not prove gone: the lane says so (``containment``), the run is
    incomplete, never a pass, and the run's test root (``OUROBOROS_TEST_TEMP_ROOT``) is marked retained."""
    from ouroboros import process_containment
    from ouroboros.test_environment import RETENTION_MARKER, retention_markers
    from tests._shared import retire_synthetic_retention

    reap = process_containment.ProcessContainer.reap

    def unproven(self):
        return reap(self) or "still alive after a best-effort kill: 4242"

    monkeypatch.setattr(process_containment.ProcessContainer, "reap", unproven)
    ws, commit = _toy_workspace(tmp_path, {"tests/test_green.py": "def test_ok():\n    assert True\n"})
    root = tmp_path / "root"
    root.mkdir()
    env = {**_lane_env(tmp_path, ws), "OUROBOROS_TEST_TEMP_ROOT": str(root)}

    run = update_checks.run_tests(ws, commit, ["tests/test_green.py"], tmp_path / "scratch", env,
                                  deadline=time.monotonic() + 60)

    assert run["complete"] is False and run["ok"] is False
    assert [lane.get("containment", "") for lane in run["lanes"]] == [
        "still alive after a best-effort kill: 4242"] * 2
    assert "not proven gone" in update_checks._incomplete(run)
    assert retention_markers(root) == [root / RETENTION_MARKER]
    assert retire_synthetic_retention(root, *[  # this test's stand-in reap proves no surviving process
        f"update checks: a lane's tree was not proven gone: {why}" for why in {
            lane["containment"] for lane in run["lanes"]}])


def _stat(pid: int) -> str:
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


@posix_only
def test_a_lane_root_whose_exit_its_kill_did_not_see_is_waited_once_its_tree_is_reaped(tmp_path, monkeypatch):
    """The tracked owner's tree kill lands late (stood in for by one that does nothing), so a SIGTERM-deaf root outlives
    the owner's 5 s exit wait and stays tracked; the lane's container then reaps it. As the commit gate waits its
    direct pytest process after its reap, the lane observes that root's exit before it returns: no zombie is left, and
    no handle the registry would still name to Emergency Stop and the generation's stop."""
    from ouroboros.tools import shell_process

    monkeypatch.setattr(shell_process, "_kill_process_group", lambda proc: None)
    marker = f"late-kill-{uuid.uuid4().hex}"
    script = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "print('before the deadline', flush=True); time.sleep(60)")
    outputs = (tmp_path / "lane.stdout.txt", tmp_path / "lane.stderr.txt")

    run = update_checks.run_tracked([sys.executable, "-c", script, marker], tmp_path, 1, outputs=outputs)

    left = [proc for proc in shell_process._active_subprocesses.copy() if marker in proc.args]
    try:
        assert (run["returncode"], run["output"], run["containment"]) == (124, "partial", "")
        assert run["stdout"] == "before the deadline\n" and run["stderr"].endswith("timed out after 1s")
        assert left == [], [(proc.pid, proc.returncode, _stat(proc.pid)) for proc in left]
    finally:
        for proc in left:  # only the root this test started: reap it and drop it from the registry
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=15)
            shell_process._active_subprocesses.discard(proc)


def test_an_unproven_lane_tree_keeps_its_process_root_even_when_its_marker_cannot_be_written(tmp_path, monkeypatch):
    """The commit gate's release rule (``preflight_runner._release_hermetic_tree``): a lane's unproven reap keeps the
    tree, and that decision does not depend on the retention marker's write. The reap names a member it could not prove
    gone and the marker cannot be written (a full disk, a read-only root): the process root is kept and named retained
    with that lane's reason, never removed."""
    import shutil

    from ouroboros import process_containment, test_environment
    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, isolated_environment
    from ouroboros.test_parallelism import NESTED_WORKERS_ENV
    from tests.test_update_checks import _lane_row

    reap = process_containment.ProcessContainer.reap
    monkeypatch.setattr(process_containment.ProcessContainer, "reap",
                        lambda self: reap(self) or "still alive after a best-effort kill: 4242")
    monkeypatch.setattr(test_environment, "retain_tree", lambda _root, _reason: False)
    row, ws = _lane_row(tmp_path, monkeypatch, {"tests/test_ok.py": "def test_ok():\n    assert True\n"})
    seen = []

    def environments(_row, root, _evidence):  # as the real ones: the lanes' test root is the process root
        seen.append(root)
        env = {**isolated_environment(tmp_path / "iso", ws, source={
            key: value for key, value in os.environ.items() if key != INSTALLATION_USER_BASE_ENV}),
               NESTED_WORKERS_ENV: "1", "OUROBOROS_TEST_TEMP_ROOT": str(root)}
        return (sys.executable, env), (sys.executable, env)

    monkeypatch.setattr(update_checks, "_environments", environments)
    try:
        verdict = update_checks.selected_tests(row, tmp_path / "scratch")
        assert verdict["ok"] is False and "not proven gone" in verdict["unverified"]
        assert seen[0].is_dir(), "the process root was removed while a lane's tree was not proven gone"
        kept = verdict["process_root_release"]
        assert (kept["state"], kept["path"], kept["markers"]) == ("retained", str(seen[0]), [])
        assert "still alive after a best-effort kill: 4242" in kept["reason"]
    finally:
        if seen:
            shutil.rmtree(seen[0], ignore_errors=True)  # this test's stand-in reap proves no surviving process


# Which run of ``selected_tests`` ends up raising after its lanes' reap could not prove their tree gone, and how.
_RAISING_RUN = {"W'": 1, "W": 2, "T": 3}


@pytest.mark.parametrize("boundary", ["stop", "run.json"])
@pytest.mark.parametrize("role", list(_RAISING_RUN))
def test_a_run_that_raises_after_an_unproven_lane_keeps_its_process_root_without_a_marker(
        tmp_path, monkeypatch, caplog, role, boundary):
    """The same release rule when the run raises instead of returning: its lanes' reap names a member it could not
    prove gone, the marker cannot be written, and then a stop begins during that lane (Emergency Stop's admission
    latch, so ``run_tests`` raises the stop's typed refusal once its evidence is kept) or the run's ``run.json`` cannot
    be written. Real lanes through ``selected_tests``, for W', the comparison on W and the attribution on T: the
    exception stays the answer, unchanged, and the process root is kept, named in the log with that lane's reason,
    never removed. Before the fix the run's facts reached the release only once it returned, so the root went."""
    import json
    import shutil

    from ouroboros import process_containment, test_environment, utils
    from ouroboros.test_environment import retention_markers
    from ouroboros.tools import shell_process
    from supervisor import update_prepare
    from tests.test_update_checks import _lane_row

    counter, order = tmp_path / "count", _RAISING_RUN[role]
    # The case fails in every run but the second: W' fails it, W passes it (so it blocks and T runs it), T fails it.
    row, ws = _lane_row(tmp_path, monkeypatch, {"tests/test_ok.py": (
        "import pathlib\n\n\ndef test_ok():\n"
        f"    seen = pathlib.Path({str(counter)!r})\n"
        "    count = int(seen.read_text()) + 1 if seen.exists() else 1\n"
        "    seen.write_text(str(count))\n"
        "    assert count == 2\n")})
    runs, seen, real_run, reap, write = [], [], update_checks.run_tests, process_containment.ProcessContainer.reap, \
        utils.atomic_write_json

    def counted(*a, **kw):
        runs.append(a[1])
        return real_run(*a, **kw)

    def unproven(self):
        if len(runs) != order:
            return reap(self)
        if boundary == "stop":
            monkeypatch.setattr(shell_process, "_panic_requested", True)  # the stop begins during this lane
        return reap(self) or "still alive after a best-effort kill: 4242"

    def unwritable(path, *a, **kw):
        if boundary == "run.json" and len(runs) == order and pathlib.Path(path).name == "run.json":
            raise OSError("the run's evidence could not be written")
        return write(path, *a, **kw)

    def environments(_row, root, _evidence):  # as the real ones: the lanes' test root is the process root
        from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, isolated_environment
        from ouroboros.test_parallelism import NESTED_WORKERS_ENV

        seen.append(root)
        env = {**isolated_environment(tmp_path / "iso", ws, source={
            key: value for key, value in os.environ.items() if key != INSTALLATION_USER_BASE_ENV}),
               NESTED_WORKERS_ENV: "1", "OUROBOROS_TEST_TEMP_ROOT": str(root)}
        return (sys.executable, env), (sys.executable, env)

    monkeypatch.setattr(shell_process, "_panic_requested", False)
    monkeypatch.setattr(update_checks, "run_tests", counted)
    monkeypatch.setattr(update_checks, "_environments", environments)
    monkeypatch.setattr(process_containment.ProcessContainer, "reap", unproven)
    monkeypatch.setattr(test_environment, "retain_tree", lambda _root, _reason: False)
    monkeypatch.setattr(utils, "atomic_write_json", unwritable)
    caplog.set_level("WARNING", logger=update_checks.__name__)
    try:
        expected = (update_prepare.PrepareRefused, "Emergency Stop") if boundary == "stop" else (
            OSError, "the run's evidence could not be written")
        with pytest.raises(expected[0], match=expected[1]) as raised:
            update_checks.selected_tests(row, tmp_path / "scratch")
        if boundary == "stop":
            assert raised.value.code == "UPDATE_PREPARE_STOPPED"
        assert len(runs) == order, runs  # the run meant to raise is the one that did
        assert seen[0].is_dir(), "the process root was removed while a lane's tree was not proven gone"
        assert retention_markers(seen[0]) == []  # no marker could be written: the facts alone kept it
        kept = [record.getMessage() for record in caplog.records if "kept their process root" in record.getMessage()]
        assert len(kept) == 1 and str(seen[0]) in kept[0] and "parallel: still alive after a best-effort kill: 4242" \
            in kept[0], kept
        if boundary == "stop":  # the stop is raised once the run's evidence is kept
            manifests = [json.loads(path.read_text(encoding="utf-8"))
                         for path in (tmp_path / "scratch").glob("junit-*/run.json")]
            assert len(manifests) == order
            assert [manifest["complete"] for manifest in manifests].count(False) == 1
            assert any(lane.get("containment") for manifest in manifests for lane in manifest["lanes"])
    finally:
        if seen:
            shutil.rmtree(seen[0], ignore_errors=True)  # this test's stand-in reap proves no surviving process


def test_a_contained_run_that_raises_hands_over_its_unproven_reap_first(tmp_path, monkeypatch):
    """Inside ``run_tracked`` too: a lane whose own start raised something other than a refusal (here a stand-in
    ``OSError``) is reaped, and the reap's finding reaches ``custody`` before that exception leaves, whatever the
    marker's write did. A proven reap hands over nothing."""
    from ouroboros import process_containment, test_environment

    monkeypatch.setattr(test_environment, "retain_tree", lambda _root, _reason: False)
    monkeypatch.setattr(update_checks, "_ended", lambda *_a: (_ for _ in ()).throw(OSError("the start broke")))
    outputs, env = (tmp_path / "lane.stdout.txt", tmp_path / "lane.stderr.txt"), {
        "OUROBOROS_TEST_TEMP_ROOT": str(tmp_path)}
    held = []
    with pytest.raises(OSError, match="the start broke"):
        update_checks.run_tracked([sys.executable, "-c", "pass"], tmp_path, 5, env=env, outputs=outputs,
                                  custody=held.append)
    assert held == []  # nothing started, the real reap proved it: nothing to hand over

    monkeypatch.setattr(process_containment.ProcessContainer, "reap",
                        lambda self: "still alive after a best-effort kill: 4242")
    with pytest.raises(OSError, match="the start broke"):
        update_checks.run_tracked([sys.executable, "-c", "pass"], tmp_path, 5, env=env, outputs=outputs,
                                  custody=held.append)
    assert held == ["still alive after a best-effort kill: 4242"]
