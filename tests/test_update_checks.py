"""The selected checks of a prepared Update decide only on complete, case-bound evidence.

``run_tests`` itself is replaced here by recorded outcomes where the verdict is exercised (a W'
failure blocks unless the SAME case fails on W with the same signature in a complete run, and one
the official target T also fails stays blocking, only attributed; an incomplete run is unverified,
never ok), and an empty target list can never reach pytest (it would run the whole suite). What is
selected is the installation's own part, read with real Git and the real shared selector, and the
runs themselves go through real pytest in the commit gate's two lanes.
"""
from __future__ import annotations

import pathlib

import pytest

from supervisor import update_checks


def _row(tmp_path):
    return {"path": str(tmp_path / "ws"), "repo_dir": str(tmp_path / "repo"), "composition_commit": "wprime",
            "layers": {"old_commit": "head", "new_files": "W2", "old_files": "W1"}, "layers_base": {"k2": "wcommit"},
            "target": {"target_sha": "tsha"}, "environment": {"same": True}}


TOY_FLOOR = {"source": "tsha:ouroboros/test_selection.py", "sha256": "toy", "tests": {"tests/test_a.py": "toy"}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    calls = []
    outcomes = {}

    def run_tests(_workspace, commit, targets, _scratch, _env=None, _python="", record=None):
        calls.append((commit, list(targets)))
        return {**outcomes[commit], "evidence": f"junit-{commit}"}

    class Selection:
        tier, files, reasons = "selected", ("tests/test_a.py", "tests/test_b.py"), ()

        def record(self):
            return {"tier": self.tier, "files": list(self.files), "reasons": []}

    monkeypatch.setattr(update_checks, "run_tests", run_tests)
    monkeypatch.setattr(update_checks, "_materialize", lambda *_a: None)
    monkeypatch.setattr(update_checks, "_test_environment", lambda *_a: {"PYTHONUSERBASE": "u"})
    monkeypatch.setattr(update_checks, "_contribution", lambda *_a: ({"ouroboros/x.py"}, set()))
    monkeypatch.setattr("supervisor.update_prepare._run",  # T has every file these cases live in
                        lambda *_a, **_kw: type("Done", (), {"returncode": 0, "stdout": b""})())
    monkeypatch.setattr("ouroboros.test_selection.select_from_paths", lambda *_a, **_kw: Selection())
    monkeypatch.setattr(update_checks, "_target_floor", lambda *_a: dict(TOY_FLOOR))
    return calls, outcomes, _row(tmp_path), tmp_path / "scratch"


def _outcome(*, complete=True, rc=1, failed=None):
    failed = failed or {}
    return {"complete": complete, "ok": complete and rc == 0 and not failed, "failed": failed, "crashed": [],
            "cases": 3, "returncode": rc, "tail": "", "seconds": 1.0,
            "lanes": [{"lane": "parallel", "returncode": rc, "cases": 3, "seconds": 1.0}]}


def test_an_incomplete_run_is_unverified_and_never_reaches_a_baseline(harness):
    calls, outcomes, row, scratch = harness
    outcomes["wprime"] = _outcome(complete=False, rc=124)  # a deadline: no failing case in a partial report
    verdict = update_checks.selected_tests(row, scratch)
    assert verdict["ok"] is False and verdict["unverified"] and calls == [("wprime", ["tests/test_a.py",
                                                                                       "tests/test_b.py"])]


def test_only_the_same_case_with_the_same_signature_on_a_complete_w_run_is_already_failing(harness):
    calls, outcomes, row, scratch = harness
    outcomes["wprime"] = _outcome(failed={"tests/test_a.py::test_x": "failure:AssertionError:['old']",
                                          "tests/test_b.py::test_y": "error:ImportError:['new']"})
    outcomes["wcommit"] = _outcome(failed={"tests/test_a.py::test_x": "failure:AssertionError:['old']",
                                           "tests/test_b.py::test_y": "failure:AssertionError:['other cause']"})
    outcomes["tsha"] = _outcome(rc=0)  # the official target passes it here: the composition caused it
    verdict = update_checks.selected_tests(row, scratch)
    assert calls[1] == ("wcommit", ["tests/test_a.py::test_x", "tests/test_b.py::test_y"])  # exactly those cases
    assert verdict["already_failing"] == ["tests/test_a.py::test_x"]
    assert verdict["blocking"] == ["tests/test_b.py::test_y"] and verdict["ok"] is False
    assert calls[2] == ("tsha", ["tests/test_b.py::test_y"]) and verdict["target_failing"] == []


def test_a_new_failure_the_official_target_also_fails_still_blocks_and_is_attributed(harness):
    """The case passed on W (the installation before the update) and fails on W' the same way it fails on T: the
    official target alone breaks it here. That is named, never waived; only a complete W run's same-signature
    failure is exempt. T runs only for the blocking cases."""
    calls, outcomes, row, scratch = harness
    signature = "failure:AssertionError:['official']"
    outcomes["wprime"] = _outcome(failed={"tests/test_a.py::test_x": signature,
                                          "tests/test_b.py::test_y": "failure:AssertionError:['known']"})
    outcomes["wcommit"] = _outcome(failed={"tests/test_b.py::test_y": "failure:AssertionError:['known']"})
    outcomes["tsha"] = _outcome(failed={"tests/test_a.py::test_x": signature})

    verdict = update_checks.selected_tests(row, scratch)

    assert verdict["ok"] is False and verdict["blocking"] == ["tests/test_a.py::test_x"]
    assert verdict["already_failing"] == ["tests/test_b.py::test_y"]
    assert verdict["target_failing"] == ["tests/test_a.py::test_x"]  # attribution only
    assert calls[2] == ("tsha", ["tests/test_a.py::test_x"])  # exactly the blocking case on T


def test_without_a_blocking_case_the_target_never_runs(harness):
    calls, outcomes, row, scratch = harness
    outcomes["wprime"] = _outcome(failed={"tests/test_b.py::test_y": "failure:AssertionError:['known']"})
    outcomes["wcommit"] = _outcome(failed={"tests/test_b.py::test_y": "failure:AssertionError:['known']"})
    verdict = update_checks.selected_tests(row, scratch)
    assert verdict["ok"] is True and [commit for commit, _targets in calls] == ["wprime", "wcommit"]
    assert "target_failing" not in verdict


def test_an_incomplete_baseline_admits_nothing(harness):
    _calls, outcomes, row, scratch = harness
    outcomes["wprime"] = _outcome(failed={"tests/test_a.py::test_x": "failure:AssertionError:['old']"})
    outcomes["wcommit"] = _outcome(complete=False, rc=2, failed={"tests/test_a.py::test_x": "failure:AssertionError:['old']"})
    outcomes["tsha"] = _outcome(failed={"tests/test_a.py::test_x": "failure:AssertionError:['old']"})
    verdict = update_checks.selected_tests(row, scratch)
    assert verdict["ok"] is False and verdict["blocking"] == ["tests/test_a.py::test_x"]


def test_an_empty_target_list_never_runs_pytest(tmp_path):
    with pytest.raises(ValueError):
        update_checks.run_tests(pathlib.Path(tmp_path), "commit", [], tmp_path / "scratch")


def test_w_prime_runs_on_the_generation_packages_and_w_on_the_serving_ones(tmp_path):
    import sys

    venv = {"environment": {"generation": {"kind": "venv", "user_base": str(tmp_path / "venv.update-x"),
                                           "current": str(tmp_path / "venv")}}}
    assert update_checks._interpreters(venv, {"A": "1"}) == (
        str(tmp_path / "venv.update-x" / "bin" / "python"), sys.executable, {"A": "1"})
    site = {"environment": {"generation": {"kind": "user_site", "user_base": "generation", "current": "serving"}}}
    assert update_checks._interpreters(site, {"PYTHONUSERBASE": "generation"}) == (
        sys.executable, sys.executable, {"PYTHONUSERBASE": "serving"})


def test_a_real_run_binds_each_failing_case_and_a_cut_run_is_incomplete(tmp_path, monkeypatch):
    """``run_tests`` itself through real pytest in a scratch git workspace, under the isolated environment."""
    import subprocess as sp

    from ouroboros.test_environment import isolated_environment

    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "test_toy.py").write_text("def test_ok():\n    assert True\n\n\n"
                                              "def test_bad():\n    assert 1 == 2, 'old cause'\n")
    (ws / "tests" / "test_slow.py").write_text("import time\n\n\ndef test_slow():\n    time.sleep(60)\n")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "toy"]):
        sp.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=ws, check=True, capture_output=True)
    commit = sp.run(["git", "rev-parse", "HEAD"], cwd=ws, capture_output=True, text=True, check=True).stdout.strip()
    env = isolated_environment(tmp_path / "iso", ws)

    done = update_checks.run_tests(ws, commit, ["tests/test_toy.py"], tmp_path / "scratch", env)

    assert done["complete"] and done["cases"] == 2 and done["ok"] is False
    assert list(done["failed"]) == ["tests/test_toy.py::test_bad"]
    assert done["failed"]["tests/test_toy.py::test_bad"].startswith("failure:")
    monkeypatch.setattr(update_checks, "_TEST_TIMEOUT_SEC", 5)  # a deadline cuts the run: no verdict from it
    cut = update_checks.run_tests(ws, commit, ["tests/test_slow.py"], tmp_path / "scratch", env)
    assert cut["complete"] is False and cut["ok"] is False and cut["returncode"] == 124


def _toy_workspace(tmp_path, files):
    import subprocess as sp

    ws = tmp_path / "ws"
    for rel, text in files.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "toy"]):
        sp.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=ws, check=True, capture_output=True)
    return ws, sp.run(["git", "rev-parse", "HEAD"], cwd=ws, capture_output=True, text=True, check=True).stdout.strip()


def test_a_class_method_failure_replays_by_its_complete_node_id_on_the_identical_baseline(tmp_path):
    """Real pytest twice: the W run's node ids (classes, nesting, parameters) collect again on W."""
    from ouroboros.test_environment import isolated_environment

    ws, commit = _toy_workspace(tmp_path, {"tests/test_existing.py": (
        "import pytest\n\n\nclass TestExisting:\n    def test_behavior(self):\n        assert 1 == 2, 'known'\n\n"
        "    class TestNested:\n        @pytest.mark.parametrize('n', [1, 2])\n        def test_value(self, n):\n"
        "            assert n == 1, 'two'\n\n    def test_fine(self):\n        assert True\n")})
    env = isolated_environment(tmp_path / "iso", ws)

    first = update_checks.run_tests(ws, commit, ["tests/test_existing.py"], tmp_path / "scratch", env)

    assert first["complete"] and sorted(first["failed"]) == [
        "tests/test_existing.py::TestExisting::TestNested::test_value[2]",
        "tests/test_existing.py::TestExisting::test_behavior"]
    replay = update_checks.run_tests(ws, commit, sorted(first["failed"]), tmp_path / "scratch", env)
    assert replay["complete"] and replay["returncode"] == 1 and replay["cases"] == 2  # collected, not exit 4
    assert replay["failed"] == first["failed"]  # the same cases with the same signatures: already failing


def test_a_command_after_emergency_stop_is_refused_before_it_starts(tmp_path, monkeypatch):
    import sys

    from ouroboros.tools import shell_process

    marker = tmp_path / "started"
    monkeypatch.setattr(shell_process, "_panic_requested", True)
    refused = update_checks.run_tracked([sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
                                        tmp_path, 30)
    assert refused["returncode"] == 125 and "Emergency Stop" in refused["stderr"] and not marker.exists()
    monkeypatch.setattr(shell_process, "_panic_requested", False)
    ran = update_checks.run_tracked([sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"], tmp_path, 30)
    assert ran["returncode"] == 0 and marker.exists()  # admission open: an ordinary preparation step runs


def _toy_git(ws, *args):
    import subprocess as sp

    return sp.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=ws, check=True,
                  capture_output=True, text=True).stdout.strip()


def _toy_commit(ws, base, files, remove=(), branch=""):
    """A commit of ``base`` with ``files`` written and ``remove`` gone, on its own branch: ``(commit, tree)``."""
    _toy_git(ws, "checkout", "-q", "-B", branch or f"b{len(files)}{len(remove)}", base)
    for rel, text in files.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    for rel in remove:
        (ws / rel).unlink()
    _toy_git(ws, "add", "-A")
    _toy_git(ws, "commit", "-q", "--allow-empty", "-m", branch or "toy")
    return _toy_git(ws, "rev-parse", "HEAD"), _toy_git(ws, "rev-parse", "HEAD^{tree}")


@pytest.mark.parametrize("case", ["official_only", "local_cutover", "carried_tests"])
def test_the_selection_is_this_installations_part_relative_to_the_official_target(tmp_path, monkeypatch, case):
    """Real Git and the real shared selector. What an official update alone changes (a cutover path included)
    never escalates the installation's checks: the floor runs. A local cutover change keeps the selector's full
    tier. The test modules this installation's work carried run on W' as long as they exist there; one the
    adaptation removed is named, and changed files no selected test reaches are named as unknown coverage."""
    from ouroboros import test_selection

    ws = tmp_path / "ws"
    ws.mkdir()
    _toy_git(ws, "init", "-q")
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor.py": "toy floor"})
    base = {"tests/test_floor.py": "def test_floor():\n    assert True\n", "server.py": "print('p')\n",
            "ouroboros/test_selection.py": 'FLOOR_TESTS = {"tests/test_floor.py": "toy floor"}\n',
            "ouroboros/__init__.py": "", "ouroboros/core.py": "GEN = 1\n",
            "tests/test_core.py": "from ouroboros import core\n\n\ndef test_core():\n    assert core.GEN\n"}
    _toy_git(ws, "checkout", "-q", "--orphan", "official-base")
    for rel, text in base.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    _toy_git(ws, "add", "-A")
    _toy_git(ws, "commit", "-q", "-m", "P")
    p = _toy_git(ws, "rev-parse", "HEAD")
    official = {"server.py": "print('t')\n"} if case == "official_only" else {"ouroboros/core.py": "GEN = 2\n"}
    target, _ = _toy_commit(ws, p, official, branch="official")
    local = {"official_only": {},
             "local_cutover": {"server.py": "print('mine')\n"},
             "carried_tests": {"tests/test_mine.py": "from ouroboros import core\n\n\ndef test_mine():\n"
                                                     "    assert core.GEN\n",
                               "tests/test_gone.py": "def test_gone():\n    assert True\n",
                               "notes/idea.md": "an idea\n"}}[case]
    head, head_tree = _toy_commit(ws, p, local, branch="local")
    kept = {key: value for key, value in local.items() if key != "tests/test_gone.py"}
    composition, composition_tree = _toy_commit(ws, target, kept, branch="composition")
    row = {"path": str(ws), "composition_commit": composition, "target": {"target_sha": target},
           "layers": {"old_commit": head, "old_files": head_tree, "new_files": composition_tree},
           "layers_base": {"k2": head}, "environment": {"same": True}}
    ran, records = [], []
    monkeypatch.setattr(update_checks, "_test_environment", lambda *_a: {"PYTHONUSERBASE": "u"})
    monkeypatch.setattr(update_checks, "run_tests", lambda _ws, commit, targets, *_a, record=None: ran.append(
        (commit, list(targets))) or records.append(record) or {
        "complete": True, "ok": True, "failed": {}, "crashed": [], "cases": 1, "returncode": 0, "tail": "",
        "lanes": [], "seconds": 0.1, "evidence": "junit-wprime"})

    verdict = update_checks.selected_tests(row, tmp_path / "scratch")

    assert verdict["ok"] is True and ran[0][0] == composition
    # The W' run keeps the exact selection that chose its targets, and the checks record names that run's evidence.
    selection = test_selection.select_from_paths(ws, records[0]["footprint"] + [
        path for path in records[0]["local_tests"] if path not in records[0]["dropped_local_tests"]], base=target)
    assert records[0]["selection"] == selection.record() and verdict["new"]["evidence"] == "junit-wprime"
    if case == "official_only":
        assert (verdict["tier"], ran[0][1], verdict["changed"]) == ("selected", ["tests/test_floor.py"], 0)
        # The rule this replaces (every path from H, against H) would run the whole suite for an official cutover.
        legacy = update_checks._names(ws, head, composition_tree)
        assert test_selection.select_from_paths(ws, legacy, base=head).tier == "full"
    elif case == "local_cutover":
        assert verdict["tier"] == "full" and verdict["changed"] == 1  # this installation's own cutover change
    else:
        assert verdict["tier"] == "selected" and verdict["local_tests"] == 2
        assert ran[0][1] == ["tests/test_floor.py", "tests/test_mine.py"]
        assert verdict["dropped_local_tests"] == ["tests/test_gone.py"]
        assert (verdict["uncovered"], verdict["uncovered_count"]) == (["notes/idea.md"], 1)
        assert records[0]["selection"]["files"] == ran[0][1] and [list(why) for why in records[0]["selection"][
            "reasons"]] == [["tests/test_floor.py", "floor: toy floor"], ["tests/test_mine.py", "changed test module"]]


def test_a_run_goes_through_the_gates_parallel_and_serial_lanes(tmp_path):
    """Real pytest: the non-serial cases over the admitted xdist workers, then the serial ones alone; each lane's
    exit, cases and seconds are kept, and a lane none of the cases belongs to is empty, not incomplete."""
    from ouroboros.test_environment import isolated_environment

    ws, commit = _toy_workspace(tmp_path, {
        "pytest.ini": "[pytest]\nmarkers =\n    serial: one at a time\n",
        "tests/test_lanes.py": ("import pytest\n\n\ndef test_parallel():\n    assert True\n\n\n"
                                "@pytest.mark.serial\ndef test_serial():\n    assert True\n"),
        "tests/test_only_parallel.py": "def test_alone():\n    assert True\n"})
    env = isolated_environment(tmp_path / "iso", ws)

    both = update_checks.run_tests(ws, commit, ["tests/test_lanes.py"], tmp_path / "scratch", env)
    alone = update_checks.run_tests(ws, commit, ["tests/test_only_parallel.py"], tmp_path / "scratch", env)

    assert both["complete"] and both["ok"] and both["cases"] == 2
    assert [(lane["lane"], lane["returncode"], lane["cases"]) for lane in both["lanes"]] == [
        ("parallel", 0, 1), ("serial", 0, 1)]
    assert all(lane["seconds"] >= 0 for lane in both["lanes"]) and both["seconds"] >= 0
    assert alone["complete"] and alone["ok"] and [(lane["lane"], lane["returncode"]) for lane in alone["lanes"]] == [
        ("parallel", 0), ("serial", 5)]


def test_a_crashed_test_worker_leaves_the_run_incomplete_never_a_failure_a_baseline_could_excuse(tmp_path):
    """xdist reports a worker lost mid-case as an ordinary failed case and runs the rest elsewhere, exiting 1 (measured
    with xdist 3.8 and ``--max-worker-restart=0``). Such a run is incomplete: W' is unverified, never ready, and a W run
    that crashed the same way excuses nothing, since only a complete W run's same-signature failure is exempt."""
    from ouroboros.test_environment import isolated_environment

    ws, commit = _toy_workspace(tmp_path, {
        "tests/test_crash.py": "import os\n\n\ndef test_dies():\n    os._exit(1)\n",
        "tests/test_ok.py": "".join(f"def test_ok_{n}():\n    assert True\n\n\n" for n in range(8))})
    env = isolated_environment(tmp_path / "iso", ws)

    done = update_checks.run_tests(ws, commit, ["tests/test_crash.py", "tests/test_ok.py"], tmp_path / "scratch", env)

    assert done["complete"] is False and done["ok"] is False
    assert done["crashed"] == ["tests/test_crash.py::test_dies"]
    assert "a test worker crashed while running tests/test_crash.py::test_dies" in update_checks._incomplete(done)


def test_a_command_cut_at_its_deadline_keeps_what_it_printed_before_it(tmp_path):
    """POSIX keeps a child's output read before the deadline on the timeout; the result holds it and names it
    ``partial`` (``unknown`` where the platform keeps none), the deadline still stated after its stderr."""
    import os
    import sys

    script = ("import sys, time; print('before the deadline', flush=True); "
              "print('on stderr too', file=sys.stderr, flush=True); time.sleep(60)")
    cut = update_checks.run_tracked([sys.executable, "-c", script], tmp_path, 2)
    done = update_checks.run_tracked([sys.executable, "-c", "print('all of it')"], tmp_path, 30)

    assert cut["returncode"] == 124 and cut["stderr"].endswith("timed out after 2s")
    if os.name == "nt":
        assert (cut["output"], cut["stdout"]) == ("unknown", "")
    else:
        assert (cut["output"], cut["stdout"], cut["stderr"]) == (
            "partial", "before the deadline\n", "on stderr too\ntimed out after 2s")
    assert (done["returncode"], done["stdout"], done["output"]) == (0, "all of it\n", "complete")


def test_a_run_keeps_its_exact_argv_and_every_lanes_complete_output_beside_its_reports(tmp_path, monkeypatch):
    """Real pytest in the gate's lanes. Green: a passing lane's output is kept too, not only a failing one's tail.
    Red: the complete output of a failing lane, its first failure included, which the record's 2000-character
    tail no longer holds. Cut: the lane the deadline ended keeps what pytest printed before it (``partial``), and
    the lane that never started is named ``none``. ``run.json`` holds the commit, interpreter, user base, targets,
    what chose them (``record``) and each lane's exact argv beside the reports, in the run's ``evidence`` folder."""
    import json
    import os
    import sys

    from ouroboros.test_environment import isolated_environment
    from ouroboros.test_parallelism import worker_budget

    red_cases = "".join(f"def test_red_{n:02d}():\n    assert False, 'cause-{n:02d} {'x' * 120} END-{n:02d}'\n\n\n"
                        for n in range(40))
    ws, commit = _toy_workspace(tmp_path, {
        "conftest.py": "import sys\n\n\ndef pytest_sessionstart(session):\n    print('SESSION-BEGAN', flush=True)\n",
        "tests/test_green.py": "def test_ok():\n    assert True\n", "tests/test_red.py": red_cases,
        "tests/test_slow.py": "import time\n\n\ndef test_slow():\n    time.sleep(60)\n"})
    env = isolated_environment(tmp_path / "iso", ws)

    def kept(run):
        folder = pathlib.Path(run["evidence"])
        return folder, json.loads((folder / "run.json").read_text(encoding="utf-8")), {
            f"{lane}.{stream}": (folder / f"{lane}.{stream}.txt").read_text(encoding="utf-8")
            for lane in ("parallel", "serial") for stream in ("stdout", "stderr")}

    green = update_checks.run_tests(ws, commit, ["tests/test_green.py"], tmp_path / "scratch", env)
    red = update_checks.run_tests(ws, commit, ["tests/test_red.py"], tmp_path / "scratch", env,
                                  record={"selection": {"tier": "selected", "files": ["tests/test_red.py"]}})
    monkeypatch.setattr(update_checks, "_TEST_TIMEOUT_SEC", 5)
    cut = update_checks.run_tests(ws, commit, ["tests/test_slow.py"], tmp_path / "scratch", env)

    folder, manifest, output = kept(green)
    assert green["ok"] and green["tail"] == "" and "1 passed" in output["parallel.stdout"]
    assert [lane["output"] for lane in manifest["lanes"]] == ["complete", "complete"]
    folder, manifest, output = kept(red)
    assert red["complete"] and len(red["failed"]) == 40 and folder.parent == tmp_path / "scratch"
    assert "END-00" in output["parallel.stdout"] and "END-00" not in red["tail"]
    assert (manifest["commit"], manifest["python"], manifest["user_base"], manifest["targets"]) == (
        commit, sys.executable, env.get("PYTHONUSERBASE"), ["tests/test_red.py"])
    assert manifest["selection"] == {"tier": "selected", "files": ["tests/test_red.py"]}
    assert [lane["argv"] for lane in manifest["lanes"]] == [
        [sys.executable, "-m", "pytest", f"@{folder / 'targets.txt'}", "-q", "-p", "no:cacheprovider", "-o",
         "junit_family=xunit1", f"--junitxml={folder / (label + '.xml')}", *flags]
        for label, flags in update_checks._lanes(worker_budget(os.cpu_count() or 1))]
    assert manifest["targets_file"] == str(folder / "targets.txt")
    assert (folder / "targets.txt").read_text(encoding="utf-8") == "tests/test_red.py\n"
    assert (manifest["complete"], manifest["failed"], manifest["returncode"]) == (True, red["failed"], 1)
    folder, manifest, output = kept(cut)
    assert cut["complete"] is False and [(lane["returncode"], lane["output"]) for lane in manifest["lanes"]] == [
        (124, "unknown" if os.name == "nt" else "partial"), (124, "none")]
    assert os.name == "nt" or "SESSION-BEGAN" in output["parallel.stdout"]
    assert output["serial.stderr"] == "the deadline passed before this lane"


def test_node_ids_reach_every_lane_through_pytests_own_argument_file(tmp_path):
    """Real pytest 9 with its ``@file`` arguments, which each xdist worker reads again: a rerun by complete node ids
    (classes, a nested class, parameters with a space and brackets) collects exactly those cases in both lanes; each
    lane's argv names the file instead of the targets, and the file's lines are the run's exact targets."""
    import json

    from ouroboros.test_environment import isolated_environment

    ws, commit = _toy_workspace(tmp_path, {
        "pytest.ini": "[pytest]\nmarkers =\n    serial: one at a time\n",
        "tests/test_ids.py": (
            "import pytest\n\n\nclass TestOuter:\n    class TestInner:\n"
            "        @pytest.mark.parametrize('s', ['a b', 'c[d]', 'plain'])\n"
            "        def test_value(self, s):\n            assert s == 'plain', 'spaced'\n\n"
            "    @pytest.mark.serial\n    def test_alone(self):\n        assert False, 'serial'\n\n"
            "    def test_fine(self):\n        assert True\n")})
    env = isolated_environment(tmp_path / "iso", ws)

    first = update_checks.run_tests(ws, commit, ["tests/test_ids.py"], tmp_path / "scratch", env)
    failing = sorted(first["failed"])
    replay = update_checks.run_tests(ws, commit, failing, tmp_path / "scratch", env)

    assert first["complete"] and failing == ["tests/test_ids.py::TestOuter::TestInner::test_value[a b]",
                                             "tests/test_ids.py::TestOuter::TestInner::test_value[c[d]]",
                                             "tests/test_ids.py::TestOuter::test_alone"]
    assert replay["complete"] and replay["failed"] == first["failed"]  # the same cases, the same signatures
    assert [(lane["lane"], lane["returncode"], lane["cases"]) for lane in replay["lanes"]] == [
        ("parallel", 1, 2), ("serial", 1, 1)]
    folder = pathlib.Path(replay["evidence"])
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert (folder / "targets.txt").read_text(encoding="utf-8").splitlines() == manifest["targets"] == failing
    assert manifest["targets_file"] == str(folder / "targets.txt")
    assert all(f"@{folder / 'targets.txt'}" in lane["argv"] and not set(failing) & set(lane["argv"])
               for lane in manifest["lanes"])


def test_a_full_tier_reaches_pytest_through_a_command_line_windows_can_start(tmp_path, monkeypatch):
    """The real selector's full tier over this tree names every test module. Inline, one lane's command line is
    far beyond the 32767 characters Windows' CreateProcess takes (measured portably, by ``subprocess.list2cmdline``),
    so no lane could start there; through the argument file each lane's command stays well inside it, and the file
    still names every module: never fewer tests. This proves the command line, not a native Windows run."""
    import subprocess as sp

    from ouroboros import test_selection

    repo = pathlib.Path(__file__).resolve().parents[1]
    selection = test_selection.select_from_paths(repo, ["tests/conftest.py"], base="HEAD")  # shared apparatus
    started = []
    monkeypatch.setattr(update_checks, "_materialize", lambda *_a: None)  # this checkout is never reset
    monkeypatch.setattr(update_checks, "run_tracked", lambda cmd, *_a, **_kw: started.append(cmd) or {
        "returncode": 5, "stdout": "", "stderr": "", "output": "complete"})
    python = str(tmp_path / ("deep" * 40) / "python.exe")  # a long interpreter path counts as well

    run = update_checks.run_tests(tmp_path / "ws", "commit", list(selection.files), tmp_path / "scratch",
                                  python=python)

    assert selection.tier == "full" and len(selection.files) > 1000
    assert len(sp.list2cmdline([python, "-m", "pytest", *selection.files])) > 32767  # what the falsifier guards
    assert len(started) == 2 and all(len(sp.list2cmdline(cmd)) < 32767 for cmd in started)
    targets = (pathlib.Path(run["evidence"]) / "targets.txt").read_text(encoding="utf-8").splitlines()
    assert targets == list(selection.files)


def test_an_interpreter_the_system_cannot_start_leaves_the_checks_unverified(tmp_path, monkeypatch):
    """Popen raising (a missing interpreter here; on Windows also a command line it does not take) is a start refused
    before anything ran: 125 and ``none``, like the Emergency Stop refusal. The run is incomplete and the selected
    checks unverified (``checks_unverified`` for the candidate), never a generic failure of the preparation step."""
    import json

    from ouroboros.test_environment import isolated_environment

    missing = tmp_path / "gone" / "bin" / "python"
    refused = update_checks.run_tracked([str(missing), "-c", "pass"], tmp_path, 30)
    assert (refused["returncode"], refused["output"], refused["stdout"]) == (125, "none", "")
    assert refused["stderr"].startswith("not started: FileNotFoundError")

    ws, commit = _toy_workspace(tmp_path, {"tests/test_ok.py": "def test_ok():\n    assert True\n"})
    env = isolated_environment(tmp_path / "iso", ws)

    class Selection:
        tier, files, reasons = "selected", ("tests/test_ok.py",), ()

        def record(self):
            return {"tier": self.tier, "files": list(self.files), "reasons": []}

    monkeypatch.setattr(update_checks, "_contribution", lambda *_a: ({"ouroboros/x.py"}, set()))
    monkeypatch.setattr(update_checks, "_test_environment", lambda *_a: env)
    monkeypatch.setattr("ouroboros.test_selection.select_from_paths", lambda *_a, **_kw: Selection())
    monkeypatch.setattr(update_checks, "_target_floor", lambda *_a: dict(TOY_FLOOR))
    row = {"path": str(ws), "composition_commit": commit, "layers_base": {"k2": commit},
           "target": {"target_sha": commit}, "environment": {"generation": {"kind": "user_site", "python": str(missing),
                                                                             "user_base": "u", "current": "c"}}}

    verdict = update_checks.selected_tests(row, tmp_path / "scratch")

    assert verdict["ok"] is False and "125 a refused start" in verdict["unverified"]
    assert verdict["new"]["complete"] is False and verdict["blocking"] == []
    manifest = json.loads((pathlib.Path(verdict["new"]["evidence"]) / "run.json").read_text(encoding="utf-8"))
    assert manifest["python"] == str(missing)
    assert [(lane["returncode"], lane["output"]) for lane in manifest["lanes"]] == [(125, "none"), (125, "none")]


def _floor_scene(tmp_path, monkeypatch, *, selector, official=None, official_remove=(), local=None, local_remove=()):
    """Real Git and the real shared selector. P declares its floor in its own ``ouroboros/test_selection.py``; the
    official target T = P + ``official`` with ``selector`` as its declaration; this installation's H = P + ``local``
    and W' = T + ``local``. The serving process's own FLOOR_TESTS is P's older list. Returns the verdict, the runs
    and the commits."""
    from ouroboros import test_selection

    ws = tmp_path / "ws"
    ws.mkdir()
    _toy_git(ws, "init", "-q")
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor_old.py": "the serving list"})
    _toy_git(ws, "checkout", "-q", "--orphan", "official-base")
    for rel, text in {"ouroboros/test_selection.py": 'FLOOR_TESTS = {"tests/test_floor_old.py": "toy floor"}\n',
                      "tests/test_floor_old.py": "def test_floor():\n    assert True\n", "ouroboros/__init__.py": "",
                      "ouroboros/core.py": "GEN = 1\n", "notes/idea.md": "an idea\n"}.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    _toy_git(ws, "add", "-A")
    _toy_git(ws, "commit", "-q", "-m", "P")
    p = _toy_git(ws, "rev-parse", "HEAD")
    target, _ = _toy_commit(ws, p, {**(official or {}), **({"ouroboros/test_selection.py": selector} if selector else {})},
                            remove=official_remove, branch="official")
    head, head_tree = _toy_commit(ws, p, local or {}, remove=local_remove, branch="local")
    composition, composition_tree = _toy_commit(ws, target, local or {}, remove=local_remove, branch="composition")
    row = {"path": str(ws), "composition_commit": composition, "target": {"target_sha": target},
           "layers": {"old_commit": head, "old_files": head_tree, "new_files": composition_tree},
           "layers_base": {"k2": head}, "environment": {"same": True}}
    ran = []
    monkeypatch.setattr(update_checks, "_test_environment", lambda *_a: {"PYTHONUSERBASE": "u"})
    monkeypatch.setattr(update_checks, "run_tests", lambda _ws, commit, targets, *_a, record=None: ran.append(
        (commit, list(targets), record)) or {"complete": True, "ok": True, "failed": {}, "crashed": [], "cases": 1,
                                             "returncode": 0, "tail": "", "lanes": [], "seconds": 0.1,
                                             "evidence": "junit-wprime"})
    return update_checks.selected_tests(row, tmp_path / "scratch"), ran, target, composition


def test_the_floor_is_the_official_targets_own_so_an_official_rename_is_followed(tmp_path, monkeypatch):
    """T renames a floor witness (and its declaration) as an ordinary refactor. The serving process still lists the
    old name, which W' no longer has; judged by that list, every Update to T would be refused forever. The checks
    take T's floor instead: the renamed witness runs, and the record names T's declaration, its digest and values."""
    import hashlib
    import json

    renamed = 'FLOOR_TESTS = {"tests/test_floor_new.py": "toy floor"}\n'
    verdict, ran, target, composition = _floor_scene(
        tmp_path, monkeypatch, selector=renamed, official={"tests/test_floor_new.py": "def test_new():\n    assert 1\n"},
        official_remove=("tests/test_floor_old.py",), local={"notes/idea.md": "mine\n"})

    floor = {"source": f"{target}:ouroboros/test_selection.py", "tests": {"tests/test_floor_new.py": "toy floor"},
             "sha256": hashlib.sha256(json.dumps({"tests/test_floor_new.py": "toy floor"}, sort_keys=True)
                                      .encode()).hexdigest()}
    assert verdict["ok"] is True and verdict["tier"] == "selected" and verdict["floor"] == floor
    assert [(commit, targets) for commit, targets, _record in ran] == [(composition, ["tests/test_floor_new.py"])]
    assert ran[0][2]["floor"] == floor and [list(why) for why in ran[0][2]["selection"]["reasons"]] == [
        ["tests/test_floor_new.py", "floor: toy floor"]]


@pytest.mark.parametrize("change", ["removed", "edited"])
def test_a_witness_this_installation_changed_is_never_skipped(tmp_path, monkeypatch, change):
    """This installation's own work removed a witness of T's floor: the checks fail, naming exactly the missing
    paths (``checks_failed``, not unverified and not a generic step failure), and nothing runs in its place. One
    its work edited is a verifier change: the selector's full tier."""
    keep = 'FLOOR_TESTS = {"tests/test_floor_old.py": "toy floor"}\n'
    local = {"edited": {"tests/test_floor_old.py": "def test_floor():\n    assert 2\n"}, "removed": {}}[change]
    verdict, ran, _target, _composition = _floor_scene(
        tmp_path, monkeypatch, selector=keep, official={"ouroboros/core.py": "GEN = 2\n"}, local=local,
        local_remove=("tests/test_floor_old.py",) if change == "removed" else ())

    if change == "removed":
        assert verdict["ok"] is False and "unverified" not in verdict and ran == []
        assert verdict["floor_missing"] == verdict["blocking"] == ["tests/test_floor_old.py"]
        assert verdict["dropped_local_tests"] == ["tests/test_floor_old.py"]
    else:
        assert verdict["ok"] is True and verdict["tier"] == "full" and verdict["changed"] == 1
        assert ["tests/test_floor_old.py", "verification apparatus changed"] in [
            list(why) for why in ran[0][2]["selection"]["reasons"]]


@pytest.mark.parametrize("declaration", ["absent", "unparsable", "nonliteral", "annotated", "changed_later", "empty",
                                         "not_a_test_module"])
def test_a_target_floor_not_declared_in_the_shape_read_here_leaves_the_checks_unverified(tmp_path, monkeypatch,
                                                                                       declaration):
    """No fallback to the serving list and no selection without a floor: a target whose floor cannot be read as
    the one recognized declaration (a single module-level FLOOR_TESTS literal of test modules) is unverified, with
    its source and why, and nothing runs."""
    selector = {"absent": "", "unparsable": "FLOOR_TESTS = {\n", "nonliteral": "FLOOR_TESTS = dict(a='b')\n",
                "annotated": 'FLOOR_TESTS: dict = {"tests/test_floor_old.py": "toy floor"}\n',
                "changed_later": 'FLOOR_TESTS = {"tests/test_floor_old.py": "toy floor"}\n'
                                 'FLOOR_TESTS["tests/test_more.py"] = "added later"\n',
                "empty": "FLOOR_TESTS = {}\n",
                "not_a_test_module": 'FLOOR_TESTS = {"ouroboros/core.py": "not a test"}\n'}[declaration]
    verdict, ran, target, _composition = _floor_scene(
        tmp_path, monkeypatch, selector=selector, official={"notes/idea.md": "theirs\n"},
        official_remove=("ouroboros/test_selection.py",) if declaration == "absent" else ())

    source = f"{target}:ouroboros/test_selection.py"
    assert verdict["ok"] is False and ran == [] and verdict["blocking"] == []
    assert verdict["unverified"].startswith(f"the official target's required floor is unknown: {source}")
    assert verdict["floor"]["source"] == source and verdict["floor"]["error"]


def test_without_a_floor_the_selector_keeps_its_own_for_every_other_caller(tmp_path, monkeypatch):
    """The commit gate, adoption qualification and ``scripts/run_tests.py`` give no floor: this process's FLOOR_TESTS
    still decides what is missing (a ValueError, refused), what an edit escalates, and the policy digest."""
    from ouroboros import test_selection

    ws, commit = _toy_workspace(tmp_path, {"tests/test_floor_a.py": "def test_a():\n    assert True\n",
                                           "tests/test_other.py": "def test_o():\n    assert True\n"})
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor_a.py": "own", "tests/test_gone.py": "own"})
    digest = test_selection.policy_digest()

    with pytest.raises(ValueError, match="required gate floor is missing: tests/test_gone.py$"):
        test_selection.select_from_paths(ws, ["tests/test_other.py"], base=commit)
    given = test_selection.select_from_paths(ws, ["README.md"], base=commit, floor={"tests/test_other.py": "given"})

    assert given.tier == "selected" and given.files == ("tests/test_other.py",) and given.policy == digest
    assert test_selection.apparatus_change("tests/test_gone.py")
    assert not test_selection.apparatus_change("tests/test_gone.py", {"tests/test_other.py": "given"})
    assert test_selection.policy_digest() == digest


@pytest.mark.skipif(not hasattr(__import__("socket"), "AF_UNIX"), reason="AF_UNIX sockets are POSIX here")
def test_a_deep_candidate_runs_its_unix_socket_cases_and_its_process_temp_is_removed(tmp_path, monkeypatch):
    """An installation's candidate sits deep in its data root, and the checks' reports deeper in its scratch. The test
    processes take HOME, TMPDIR and caches from a short root where the commit gate keeps its own, so a forkserver child
    (the floor's lifeline case starts one) gets its socket: rooted in that scratch, it outgrew macOS's 104 bytes and
    the case failed on W' and W alike, as if already failing. Each run records that TMPDIR; the root is gone once the
    checks end, and the reports stay in the scratch."""
    import json

    deep = tmp_path / ("data-" * 12) / "update-candidates" / "update-0123456789ab"
    deep.mkdir(parents=True)
    ws, commit = _toy_workspace(deep, {"tests/test_unix.py": (
        "import multiprocessing as mp\nimport time\n\n\ndef test_a_forkserver_child_starts():\n"
        "    child = mp.get_context('forkserver').Process(target=time.sleep, args=(0,))\n"
        "    child.start()\n    child.join(60)\n    assert child.exitcode == 0\n")})
    scratch = ws.with_name(ws.name + ".env") / "update-checks"

    class Selection:
        tier, files, reasons = "selected", ("tests/test_unix.py",), ()

        def record(self):
            return {"tier": self.tier, "files": list(self.files), "reasons": []}

    monkeypatch.setattr(update_checks, "_contribution", lambda *_a: ({"ouroboros/x.py"}, set()))
    monkeypatch.setattr(update_checks, "_target_floor", lambda *_a: dict(TOY_FLOOR))
    monkeypatch.setattr("ouroboros.test_selection.select_from_paths", lambda *_a, **_kw: Selection())
    row = {"path": str(ws), "composition_commit": commit, "layers_base": {"k2": commit},
           "target": {"target_sha": commit},
           "environment": {"generation": {"kind": "user_site", "user_base": str(tmp_path / "u"), "current": "c"}}}

    verdict = update_checks.selected_tests(row, scratch)

    folder = pathlib.Path(verdict["new"]["evidence"])
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    # W' itself passes: the case is never excused as already failing on a W that failed the same environmental way.
    assert verdict["ok"] is True and verdict["new"]["ok"] and verdict["new"]["cases"] == 1
    assert verdict["already_failing"] == [] and "baseline" not in verdict
    assert len(str(scratch / "run-01234567" / "tmp")) + len("/pymp-0123abcd/listener-0123abcd") > 104  # the old root
    assert pathlib.Path(manifest["tmpdir"]).parent == pathlib.Path(verdict["process_root"]).resolve()
    assert not pathlib.Path(verdict["process_root"]).exists() and folder.parent == scratch and folder.is_dir()
