"""One monotonic budget for a prepared Update's checks (``update_checks``): it begins just before W' is materialized
and bounds every process of W', W and T, each getting what remains (``config.UPDATE_CHECKS_TIMEOUT_SEC``, or the commit
gate's explicit ``OUROBOROS_PREFLIGHT_TIMEOUT_SEC``). A Git step or run out of it is named with its own output kept,
never a pass, a waiver or an erased blocker; W' is in the workspace again afterwards, also when a check raised; a
begun stop raises its typed refusal."""
from __future__ import annotations

import pathlib
import subprocess
import time

import pytest

from supervisor import update_checks
from tests import test_update_checks
from tests.test_update_checks import _outcome, _toy_commit, _toy_git, _toy_workspace

harness = test_update_checks.harness  # the same recorded outcomes and stand-ins


@pytest.mark.parametrize("cut", ["timeout", "spent"])
def test_a_presence_read_out_of_budget_attributes_nothing_and_keeps_its_raw_attempt(harness, monkeypatch, cut):
    """T's presence read within what remains of the budget: one that ran out of time, or never started because W
    spent the rest, names why and attributes nothing; the blocking case stands, T never runs, the Git owner's own
    words are kept as they came, and W' is in the workspace again."""
    import json

    calls, outcomes, row, scratch = harness
    outcomes["wprime"] = _outcome(failed={"tests/test_a.py::test_x": "failure:AssertionError:['new']"})
    outcomes["wcommit"] = _outcome(rc=0)
    said, restored = b"git process timed out after 1s and was terminated: git cat-file (fatal: slow disk)", []
    monkeypatch.setattr(update_checks, "_materialize", lambda *a: restored.append(a[1]))

    def read(*a, **kw):
        raise subprocess.TimeoutExpired(["git", *a[1:]], kw["timeout"], output=b"", stderr=said)

    monkeypatch.setattr("supervisor.update_prepare._run", read)
    if cut == "spent":  # W takes the rest of a one-second budget
        fake = update_checks.run_tests
        monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "1")
        monkeypatch.setattr(update_checks, "run_tests", lambda *a, **kw: (
            time.sleep(1.2) if a[1] == "wcommit" else None) or fake(*a, **kw))

    verdict = update_checks.selected_tests(row, scratch)

    target, folder = verdict["target_run"], pathlib.Path(verdict["target_run"]["presence"])
    assert verdict["ok"] is False and verdict["blocking"] == ["tests/test_a.py::test_x"] and "unverified" not in verdict
    assert [commit for commit, _targets in calls] == ["wprime", "wcommit"] and verdict["target_failing"] == []
    assert (target["complete"], target["cases"]) == (False, 0) and target["unavailable"].startswith(
        {"timeout": "reading T's files did not finish within", "spent": "the deadline passed before"}[cut])
    presence = json.loads((folder / "presence.json").read_text(encoding="utf-8"))
    assert presence["outcome"] == {"timeout": "timeout", "spent": "not_started"}[cut]
    assert (folder / "stdin.txt").read_bytes() == b"tsha:tests/test_a.py\n"
    assert (folder / "cat-file.stderr.bin").read_bytes() == (said if cut == "timeout" else b"")
    assert restored[-1] == "wprime"


def test_a_check_that_raises_after_w_prime_still_restores_it_and_raises_its_own_failure(harness, monkeypatch):
    """W' is in the workspace again after a W or T check that raised, through the Git owner outside the budget; a
    restore that fails too is logged, and the check's own exception stays the answer. A begun stop that ended T's
    presence read raises its typed refusal, never read as T lacking those files."""
    from ouroboros.tools import shell_process
    from supervisor.update_prepare import PrepareRefused

    calls, outcomes, row, scratch = harness
    outcomes["wprime"] = _outcome(failed={"tests/test_a.py::test_x": "failure:AssertionError:['new']"})
    outcomes["wcommit"] = _outcome(rc=0)
    restored, broke, fake = [], [], update_checks.run_tests

    def materialize(*a):  # W' before the selection, then the restores
        restored.append(a[1])
        if len(broke) > 1:
            raise RuntimeError("the restore failed as well")

    def w_breaks(*a, **kw):
        if a[1] == "wcommit":
            broke.append(True)
            raise OSError("W broke")
        return fake(*a, **kw)

    monkeypatch.setattr(update_checks, "_materialize", materialize)
    monkeypatch.setattr(update_checks, "run_tests", w_breaks)
    with pytest.raises(OSError, match="W broke"):
        update_checks.selected_tests(row, scratch)
    with pytest.raises(OSError, match="W broke"):  # its own restore failing changes nothing of the answer
        update_checks.selected_tests(row, scratch)
    assert restored == ["wprime"] * 4

    restored.clear()
    monkeypatch.setattr(update_checks, "_materialize", lambda *a: restored.append(a[1]))
    monkeypatch.setattr(update_checks, "run_tests", fake)
    monkeypatch.setattr("supervisor.update_prepare._run", lambda *a, **kw: subprocess.CompletedProcess(a, -15, b"",
                                                                                                        b""))
    monkeypatch.setattr(shell_process, "_panic_requested", True)  # the stop that ended the read
    with pytest.raises(PrepareRefused) as stopped:
        update_checks.selected_tests(row, scratch)
    assert stopped.value.code == "UPDATE_PREPARE_STOPPED" and restored == ["wprime", "wprime"]
    assert [commit for commit, _targets in calls][-2:] == ["wprime", "wcommit"]  # T never ran


def test_one_budget_begins_before_w_prime_and_bounds_w_and_t_with_what_remains(harness, monkeypatch):
    """One monotonic deadline, taken once the environments are ready, reaches W', W and T alike, T's presence read
    included: one batched Git process for both files, bounded by what remains. The configured budget (here the commit
    gate's explicit override) is recorded beside the verdict, apart from any run's own."""
    calls, outcomes, row, scratch = harness
    outcomes["wprime"] = _outcome(failed={"tests/test_a.py::test_x": "failure:AssertionError:['new']",
                                          "tests/test_b.py::test_y": "failure:AssertionError:['new']"})
    outcomes["wcommit"] = _outcome(rc=0)
    outcomes["tsha"] = _outcome(rc=0)
    fake, deadlines, reads = update_checks.run_tests, [], []
    monkeypatch.setattr(update_checks, "run_tests", lambda *a, **kw: deadlines.append(kw["deadline"]) or fake(*a, **kw))
    monkeypatch.setattr("supervisor.update_prepare._run", lambda *a, **kw: reads.append(
        (a[1:], kw["timeout"], kw["data"])) or subprocess.CompletedProcess(a, 0, b"blob\nblob\n", b""))
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "123")

    before = time.monotonic()
    verdict = update_checks.selected_tests(row, scratch)
    after = time.monotonic()

    assert [commit for commit, _targets in calls] == ["wprime", "wcommit", "tsha"] and len(set(deadlines)) == 1
    assert verdict["budget_seconds"] == 123 and before + 123 <= deadlines[0] <= after + 123
    assert [(args, data) for args, _timeout, data in reads] == [
        (("cat-file", "--batch-check=%(objecttype)"), b"tsha:tests/test_a.py\ntsha:tests/test_b.py\n")]
    assert 0 < reads[0][1] <= 123 and verdict["target_run"]["complete"] is True  # the remainder, not the owner's own


def test_the_budget_is_the_configured_default_unless_the_gates_explicit_override_names_another(monkeypatch):
    from ouroboros import config

    monkeypatch.delenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", raising=False)
    assert update_checks._budget() == config.UPDATE_CHECKS_TIMEOUT_SEC
    for raw, seconds in (("60", 60), (str(config.UPDATE_CHECKS_TIMEOUT_SEC * 3), config.UPDATE_CHECKS_TIMEOUT_SEC * 3),
                         ("0", config.UPDATE_CHECKS_TIMEOUT_SEC), ("junk", config.UPDATE_CHECKS_TIMEOUT_SEC)):
        monkeypatch.setenv("OUROBOROS_PREFLIGHT_TIMEOUT_SEC", raw)
        assert update_checks._budget() == seconds, raw


def test_each_run_gets_what_remains_of_one_budget_and_a_spent_one_starts_nothing(tmp_path):
    """Real pytest under one shared deadline: each run records what remained when it began (``deadline_seconds``),
    less for a later run by at least the earlier one's time; once it is spent a run neither materializes its commit
    nor starts a lane, and says so."""
    import json

    from ouroboros.test_environment import isolated_environment

    ws, first = _toy_workspace(tmp_path, {
        "tests/test_wait.py": "import time\n\n\ndef test_wait():\n    time.sleep(2)\n"})
    second, _ = _toy_commit(ws, first, {"tests/test_other.py": "def test_other():\n    assert True\n"}, branch="two")
    _toy_git(ws, "checkout", "-q", "--detach", first)
    env = isolated_environment(tmp_path / "iso", ws)
    budget = 120
    deadline = time.monotonic() + budget

    runs = [update_checks.run_tests(ws, first, ["tests/test_wait.py"], tmp_path / "scratch", env, deadline=deadline)
            for _ in range(2)]
    spent = update_checks.run_tests(ws, second, ["tests/test_other.py"], tmp_path / "scratch", env,
                                    deadline=time.monotonic() - 1)

    left = [json.loads((pathlib.Path(run["evidence"]) / "run.json").read_text(encoding="utf-8"))["deadline_seconds"]
            for run in runs]
    assert all(run["complete"] and run["ok"] for run in runs)
    assert budget >= left[0] > left[1] and left[0] - left[1] >= runs[0]["seconds"] - 0.2
    folder = pathlib.Path(spent["evidence"])
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert spent["complete"] is False and manifest["deadline_seconds"] < 0
    assert [(lane["returncode"], lane["output"]) for lane in manifest["lanes"]] == [(124, "none"), (124, "none")]
    assert (folder / "parallel.stderr.txt").read_text(encoding="utf-8") == (
        f"the deadline passed before {second} was materialized")
    assert _toy_git(ws, "rev-parse", "HEAD") == first  # never materialized


def test_materializing_takes_what_remains_and_every_other_git_step_keeps_the_owners_bound(tmp_path, monkeypatch):
    """The Git owner's own bound stays for every caller that names none; a run's materialization gets what remains of
    the checks' budget (never more than that bound), and running out of it there leaves the run unstarted, named, with
    the Git owner's own termination message kept as it came (``run.json`` ``materialization``, ``materialize.*.bin``).
    A begun stop that ended a materialization raises its typed refusal, never an ordinary failed step."""
    import json
    import subprocess as sp

    from ouroboros.tools import shell_process
    from supervisor import git_ops, update_prepare

    seen, said = [], b"git process timed out after 0.6s and was terminated: git reset (fatal: index.lock held)"

    def bounded(_cmd, *, timeout, **_kw):
        seen.append(timeout)
        return (git_ops.FETCH_TIMEOUT_RC, b"", said) if timeout < 1 else (0, b"", b"")

    monkeypatch.setattr(git_ops, "_run_git_process_bounded", bounded)
    update_prepare._run(tmp_path, "status")
    update_prepare._run(tmp_path, "status", timeout=42.5)
    with pytest.raises(sp.TimeoutExpired) as cut:
        update_prepare._run(tmp_path, "reset", "--hard", timeout=0.5)
    assert seen == [update_prepare._GIT_TIMEOUT_SEC, 42.5, 0.5] and cut.value.timeout == 0.5

    seen.clear()
    run = update_checks.run_tests(tmp_path, "c0ffee", ["tests/test_x.py"], tmp_path / "scratch",
                                  deadline=time.monotonic() + 0.6)
    folder = pathlib.Path(run["evidence"])
    kept = json.loads((folder / "run.json").read_text(encoding="utf-8"))["materialization"]
    assert len(seen) == 1 and 0 < seen[0] <= 0.6 and run["complete"] is False
    assert (kept["argv"], kept["outcome"], kept["stderr"]) == (
        ["git", "reset", "--hard", "--quiet", "c0ffee"], "timeout", "materialize.stderr.bin")
    assert (folder / "materialize.stderr.bin").read_bytes() == said  # the raw attempt, not a sentence about it
    assert (folder / "serial.stderr.txt").read_text(encoding="utf-8").startswith(
        "materializing c0ffee did not finish within") and said.decode() in run["tail"]

    monkeypatch.setattr(git_ops, "_run_git_process_bounded", lambda _cmd, **_kw: (-15, b"", b""))  # ended
    with pytest.raises(update_prepare._registry().GitFailed):  # no stop: the failed step, as before
        update_checks.run_tests(tmp_path, "c0ffee", ["tests/test_x.py"], tmp_path / "scratch")
    monkeypatch.setattr(shell_process, "_panic_requested", True)  # the stop that ended it
    with pytest.raises(update_prepare.PrepareRefused) as stopped:
        update_checks.run_tests(tmp_path, "c0ffee", ["tests/test_x.py"], tmp_path / "scratch")
    assert stopped.value.code == "UPDATE_PREPARE_STOPPED"


@pytest.mark.parametrize("returncode", [-15, -9, 1, 127])
def test_a_git_step_a_begun_stop_ended_is_that_stop_even_unchecked(tmp_path, monkeypatch, returncode):
    """A Git child the generation's stop ended answers -15 or -9, not the refused start's 127. Read unchecked (T's
    floor ``git show``, a snapshot's ``diff-files --quiet``) it was taken as Git's answer: the floor unknown, the
    checks unverified, never the stop. Any nonzero exit while a stop has begun is that stop's typed refusal; without
    a stop the same exit is returned as before."""
    from ouroboros.tools import shell_process
    from supervisor import git_ops, update_prepare

    monkeypatch.setattr(git_ops, "_run_git_process_bounded", lambda _cmd, **_kw: (returncode, b"", b"ended"))
    assert update_prepare._run(tmp_path, "diff-files", "--quiet", check=False).returncode == returncode
    with pytest.raises(ValueError, match="cannot be read"):
        update_checks._target_floor(tmp_path, "tsha")
    monkeypatch.setattr(shell_process, "_panic_requested", True)  # the stop that ended it
    for read in (lambda: update_prepare._run(tmp_path, "diff-files", "--quiet", check=False),
                 lambda: update_checks._target_floor(tmp_path, "tsha")):
        with pytest.raises(update_prepare.PrepareRefused) as stopped:
            read()
        assert stopped.value.code == "UPDATE_PREPARE_STOPPED"


def test_a_stop_that_ends_a_run_is_that_stop_once_the_run_kept_its_evidence(tmp_path, monkeypatch):
    """Real lanes. Emergency Stop begins while W' runs, once its first lane ended (that lane's measured failure kept),
    so the tracked owner refuses the next lane (125): ``run.json`` and every lane's output are written, then the stop's
    typed refusal is raised, never an incomplete run read as unverified checks. A run that completed before the stop
    began is its measured result: its failure stays a failure and nothing is raised."""
    import json
    import os

    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, isolated_environment
    from ouroboros.tools import shell_process
    from supervisor import update_prepare

    ws, commit = _toy_workspace(tmp_path, {"tests/test_t.py": (
        "def test_ok():\n    assert True\n\n\ndef test_bad():\n    assert False, 'measured before the stop'\n")})
    env = isolated_environment(tmp_path / "iso", ws, source={
        key: value for key, value in os.environ.items() if key != INSTALLATION_USER_BASE_ENV})
    real, ended = update_checks.run_tracked, []

    def stop_after(lanes):
        def tracked(*args, **kwargs):
            done = real(*args, **kwargs)
            ended.append(done["returncode"])
            if len(ended) == lanes:
                monkeypatch.setattr(shell_process, "_panic_requested", True)  # Emergency Stop begins here
            return done
        return tracked

    monkeypatch.setattr(shell_process, "_panic_requested", False)
    monkeypatch.setattr(update_checks, "run_tracked", stop_after(1))
    with pytest.raises(update_prepare.PrepareRefused) as stopped:
        update_checks.run_tests(ws, commit, ["tests/test_t.py"], tmp_path / "cut", env)
    assert stopped.value.code == "UPDATE_PREPARE_STOPPED"
    folder, = (tmp_path / "cut").glob("junit-*")
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert [(lane["lane"], lane["returncode"]) for lane in manifest["lanes"]] == [("parallel", 1), ("serial", 125)]
    assert manifest["complete"] is False and list(manifest["failed"]) == ["tests/test_t.py::test_bad"]
    assert "Emergency Stop" in (folder / "serial.stderr.txt").read_text(encoding="utf-8")
    assert "test_bad" in (folder / "parallel.stdout.txt").read_text(encoding="utf-8")

    ended.clear()
    monkeypatch.setattr(shell_process, "_panic_requested", False)
    monkeypatch.setattr(update_checks, "run_tracked", stop_after(2))  # the stop begins after the run's last lane
    done = update_checks.run_tests(ws, commit, ["tests/test_t.py"], tmp_path / "whole", env)
    assert done["complete"] is True and list(done["failed"]) == ["tests/test_t.py::test_bad"]
