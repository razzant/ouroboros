"""Real parent/child writes and concurrent runs stay on disposable roots."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ouroboros.preflight_runner import _preflight_env
from ouroboros.settings_defaults import settings_env_keys
from ouroboros.test_environment import isolated_environment, settings_keys
from tests.group_run import kill_survivors, run_in_group

REPO = Path(__file__).resolve().parents[1]


def test_pytest_cache_belongs_to_session_boundary(pytestconfig):
    from tests.conftest import _PYTEST_ROOT

    assert Path(pytestconfig.cache._cachedir).is_relative_to(_PYTEST_ROOT)


def test_static_settings_scrub_covers_runtime_vocabulary():
    assert set(settings_env_keys()) <= settings_keys()


def test_isolation_keeps_the_platform_default_text_encoding(tmp_path):
    """UTF-8 mode forced onto every child would hide the cp1252 bugs Windows CI
    catches; isolation is about roots, never about the interpreter's encoding."""
    env = isolated_environment(tmp_path, REPO, source={})
    assert "PYTHONUTF8" not in env and "PYTHONIOENCODING" not in env
    gate = _preflight_env(tmp_path / "data", tmp_path / "repo")
    assert "PYTHONUTF8" not in gate and "PYTHONIOENCODING" not in gate


def test_chromium_download_staging_uses_disposable_temp(tmp_path):
    env = isolated_environment(tmp_path, REPO, source={"MAC_CHROMIUM_TMPDIR": "/owner-temp"})
    assert env["MAC_CHROMIUM_TMPDIR"] == env["TMPDIR"]
    assert Path(env["MAC_CHROMIUM_TMPDIR"]).is_relative_to(tmp_path)
    assert Path(env["MAC_CHROMIUM_TMPDIR"]).is_dir()


def test_scrubbed_child_reinjects_roots_but_preserves_explicit_synthetic_home(tmp_path):
    from tests.conftest import _isolated_child_env, _PYTEST_DEFAULTS

    env = _isolated_child_env({})
    for key in ("HOME", "OUROBOROS_APP_ROOT", "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
                "OUROBOROS_SUBAGENT_WORKTREE_ROOT", "OUROBOROS_DELIVERABLES_ROOT",
                "PYTHONPYCACHEPREFIX", "PYTHONUSERBASE"):
        assert env[key] == _PYTEST_DEFAULTS[key]
    empty = _isolated_child_env({key: "" for key in _PYTEST_DEFAULTS})
    for key in ("OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH", "OUROBOROS_APP_ROOT",
                "OUROBOROS_SUBAGENT_PROJECTS_ROOT", "OUROBOROS_SUBAGENT_WORKTREE_ROOT",
                "OUROBOROS_DELIVERABLES_ROOT", "PYTHONUSERBASE"):
        assert empty[key] == _PYTEST_DEFAULTS[key]
    assert empty["PYTHONDONTWRITEBYTECODE"] == empty["PYTHONPYCACHEPREFIX"] == ""
    selected = tmp_path / "synthetic-home"
    env = _isolated_child_env({"HOME": str(selected), "USERPROFILE": str(selected),
                               "OUROBOROS_DELIVERABLES_ROOT": str(tmp_path / "explicit")})
    assert env["HOME"] == str(selected)
    assert "OUROBOROS_SUBAGENT_PROJECTS_ROOT" not in env
    assert "OUROBOROS_SUBAGENT_WORKTREE_ROOT" not in env
    assert env["OUROBOROS_DELIVERABLES_ROOT"] == str(tmp_path / "explicit")


def test_git_discovery_cannot_escape_to_an_ancestor_checkout(tmp_path, monkeypatch):
    from ouroboros.workspace_admission import validate_workspace_root, WorkspaceRootError

    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    subprocess.run(["git", "init", str(ancestor)], check=True, capture_output=True)
    root = ancestor / "test-boundary"
    env = isolated_environment(root, REPO)
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                            cwd=root / "tmp", env=env, capture_output=True)
    assert result.returncode != 0
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(root))
    assert validate_workspace_root(root / "tmp", system_repo_dir=REPO,
                                   drive_root=root / "data") == root / "tmp"
    (root / "tmp" / ".git").write_text("invalid git metadata", encoding="utf-8")
    with pytest.raises(WorkspaceRootError, match="could not be resolved"):
        validate_workspace_root(root / "tmp", system_repo_dir=REPO, drive_root=root / "data")
    (root / "tmp" / ".git").unlink()
    own = root / "tmp" / "own"
    subprocess.run(["git", "init", str(own)], env=env, check=True, capture_output=True)
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                            cwd=own, env=env, check=True, capture_output=True, text=True)
    assert Path(result.stdout.strip()).resolve() == own.resolve()


def test_a_session_under_an_owner_keeps_the_runs_prefix_and_collects_with_gc_paused(tmp_path):
    """Under an owner root (safe_test, the gate) every session reads and hands on the OWNER's warmed bytecode
    prefix (a prefix of its own would start cold), while bare pytest keeps one under its own session root; and
    the conftest collects with the cyclic collector paused, freezing the collected tree afterwards."""
    owner = tmp_path / "owner"
    env = isolated_environment(owner, REPO)
    script = "import json, os, sys, tests.conftest; print(json.dumps([sys.pycache_prefix, os.environ['PYTHONPYCACHEPREFIX']]))"
    owned = subprocess.run([sys.executable, "-c", script], cwd=REPO, env={**env, "OUROBOROS_TEST_TEMP_ROOT": str(owner)},
                           text=True, capture_output=True, timeout=120)
    assert owned.returncode == 0, owned.stderr
    assert json.loads(owned.stdout.splitlines()[-1]) == [env["PYTHONPYCACHEPREFIX"]] * 2
    bare = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env, text=True, capture_output=True, timeout=120)
    assert bare.returncode == 0, bare.stderr
    prefix, handed = json.loads(bare.stdout.splitlines()[-1])
    assert prefix == handed and prefix != env["PYTHONPYCACHEPREFIX"] and prefix.endswith("pycache.noindex")
    suite, witness = tmp_path / "suite", tmp_path / "gc-at-collection.txt"
    (suite / "tests").mkdir(parents=True)
    (suite / "tests" / "test_gc.py").write_text(
        f"import gc, pathlib\npathlib.Path({str(witness)!r}).write_text(str(gc.isenabled()))\n"
        "def test_after_collection():\n    assert gc.isenabled() and gc.get_freeze_count() > 0\n", encoding="utf-8")
    run = subprocess.run([sys.executable, "-m", "pytest", "-p", "tests.conftest", "--rootdir", str(suite), "--confcutdir",
                          str(suite), "-o", "addopts=", "-q", "-p", "no:cacheprovider", str(suite / "tests")],
                         cwd=REPO, env={**env, "OUROBOROS_TEST_TEMP_ROOT": str(owner)}, text=True, capture_output=True,
                         timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    assert witness.read_text(encoding="utf-8") == "False"
_LAUNCHER = [sys.executable, "-I", "-S", str(REPO / "scripts" / "safe_test.py")]


def _final_root(stderr):
    """Exactly one final line names the launcher's root, and it is the last one."""
    lines = stderr.splitlines()
    final = [line for line in lines if line.startswith(("SAFE_TEST_RETAINED ", "SAFE_TEST_RELEASED "))]
    assert len(final) == 1 and lines[-1] == final[0], stderr
    return final[0].split(" ", 1)


def _live_conftest():
    return next(module for module in list(sys.modules.values())
                if getattr(module, "__file__", "")
                and Path(module.__file__).as_posix().endswith("tests/conftest.py")
                and hasattr(module, "_guard_test_tree_deletion"))


def _pytest_config(tmp_path, *, policy="all", basetemp=None, worker=False):
    from types import SimpleNamespace

    config = SimpleNamespace(getini=lambda name: policy, option=SimpleNamespace(basetemp=basetemp))
    if worker:
        config.workerinput = {"workerid": "gw0"}
    return config


def test_pytest_is_refused_any_deletion_of_test_trees(tmp_path, monkeypatch):
    conftest = _live_conftest()
    monkeypatch.setattr(conftest, "_SAFE_TEMP_ROOT", "")
    with pytest.raises(pytest.UsageError, match="tmp_path_retention_policy"):
        conftest._guard_test_tree_deletion(_pytest_config(tmp_path, policy="failed"))
    previous = tmp_path / "previous-basetemp"
    (previous / "test_x0").mkdir(parents=True)
    with pytest.raises(pytest.UsageError, match="fresh, never-used path"):
        conftest._guard_test_tree_deletion(_pytest_config(tmp_path, basetemp=str(previous)))
    assert (previous / "test_x0").is_dir()
    # An EMPTY existing directory is refused too: pytest's rm_rf runs on any existing
    # explicit basetemp, and emptiness never proves the process that owns it is gone.
    empty = tmp_path / "empty-basetemp"
    empty.mkdir()
    with pytest.raises(pytest.UsageError, match="already exists"):
        conftest._guard_test_tree_deletion(_pytest_config(tmp_path, basetemp=str(empty)))
    assert empty.is_dir()
    conftest._guard_test_tree_deletion(_pytest_config(tmp_path, basetemp=str(tmp_path / "fresh")))
    # Bare pytest claims one too, beneath its own fresh session root: pytest's numbered
    # basetemp would be subject to tmp_path_retention_count cleanup at exit.
    session_root = tmp_path / "session-root"
    session_root.mkdir()
    monkeypatch.setattr(conftest, "_PYTEST_ROOT", session_root)
    bare = _pytest_config(tmp_path)
    conftest._guard_test_tree_deletion(bare)
    assert bare.option.basetemp == str(session_root / "b0" / "t")
    # Every controller under the launcher claims its own never-used basetemp.
    launcher_root = tmp_path / "launcher-root"
    launcher_root.mkdir()
    monkeypatch.setattr(conftest, "_SAFE_TEMP_ROOT", str(launcher_root))
    first, second = _pytest_config(tmp_path), _pytest_config(tmp_path)
    conftest._guard_test_tree_deletion(first)
    conftest._guard_test_tree_deletion(second)
    assert first.option.basetemp == str(launcher_root / "b0" / "t")
    assert second.option.basetemp == str(launcher_root / "b1" / "t")
    assert not Path(first.option.basetemp).exists()  # pytest creates it; nothing to empty
    worker = _pytest_config(tmp_path, worker=True)
    conftest._guard_test_tree_deletion(worker)
    assert worker.option.basetemp is None


def test_basetemp_dotdot_cannot_delete_an_existing_tree(tmp_path):
    retained = tmp_path / "retained"
    retained.mkdir()
    sentinel = retained / "sentinel"
    sentinel.write_bytes(b"keep")
    spelling = tmp_path / "nonexistent" / ".." / "retained"
    with pytest.raises(pytest.UsageError, match="already exists"):
        _live_conftest()._guard_test_tree_deletion(_pytest_config(tmp_path, basetemp=str(spelling)))
    assert sentinel.read_bytes() == b"keep"


_NODE = "tests/test_test_environment.py::test_git_discovery_cannot_escape_to_an_ancestor_checkout"


def _nested_env(**overrides):
    import os

    # The nested run is its own session: when THIS test runs inside an xdist
    # worker, the inherited worker id would make the nested controller write
    # its probe as that worker and the record set would never show "controller".
    dropped = {"OUROBOROS_TEST_TEMP_ROOT", "PYTEST_XDIST_WORKER", "PYTEST_XDIST_TESTRUNUID"}
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    env.update(overrides)
    return env


def test_session_finish_retains_the_session_tree(tmp_path, monkeypatch):
    from types import SimpleNamespace

    conftest = _live_conftest()
    session_root = tmp_path / "session-root"
    (session_root / "data").mkdir(parents=True)
    monkeypatch.setattr(conftest, "_PYTEST_ROOT", session_root)
    monkeypatch.setattr(conftest, "_PYTEST_DATA_DIR", session_root / "data")
    monkeypatch.setattr(conftest, "_mock_pollution_files", lambda root: set())
    session = SimpleNamespace(config=SimpleNamespace(_ouroboros_initial_mock_pollution=set()), exitstatus=0)
    conftest.pytest_sessionfinish(session, 0)
    assert (session_root / "data").is_dir() and session.exitstatus == 0


_RETENTION_PROBE = """
import os
import pytest

@pytest.fixture(autouse=True)
def _manufactured_retention(tmp_path):
    from ouroboros.test_environment import retain_tree
    from tests._shared import retire_synthetic_retention
    held = tmp_path / "held"
    held.mkdir()
    assert retain_tree(held, "probe: stubbed reap")
    yield
    if os.environ["RETENTION_PROBE"] == "retire":
        assert retire_synthetic_retention(held, "probe: stubbed reap")
"""


@pytest.mark.parametrize("mode", ["leave", "retire", "bare"])
def test_a_marker_left_in_a_tests_tmp_path_fails_that_test_under_a_releasing_owner(tmp_path, mode):
    """#1908: one forgotten synthetic marker kept every green battery's whole root. Without an owner
    nobody fails it, and the marker still keeps its folder."""
    from tests._shared import retire_synthetic_retention

    probe, launcher_root = tmp_path / "probe", tmp_path / "launcher-root"
    probe.mkdir()
    launcher_root.mkdir()
    (probe / "retention_probe.py").write_text(_RETENTION_PROBE, encoding="utf-8")
    owner = {"TMPDIR": str(launcher_root)} if mode == "bare" else {"OUROBOROS_TEST_TEMP_ROOT": str(launcher_root)}
    env = _nested_env(PYTHONPATH=str(probe), RETENTION_PROBE=mode, **owner)
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", _NODE, "-o", "addopts=", "-q",
                                 "-p", "retention_probe"],
                                cwd=REPO, env=env, text=True, capture_output=True, timeout=300)
        output = result.stdout + result.stderr
        if mode == "leave":
            assert result.returncode == 1 and "1 passed" in output and "1 error" in output, output
            assert "ERROR at teardown of test_git_discovery_cannot_escape_to_an_ancestor_checkout" in output, output
            assert "left retention in its tmp_path" in output and "held (probe: stubbed reap)" in output, output
            assert "retire_synthetic_retention" in output, output
            assert len(list(launcher_root.glob("b*/t/*/held"))) == 1, "a marker keeps its test's folder"
        else:
            assert result.returncode == 0, output
            assert len(list(launcher_root.glob("**/t/*/held"))) == (mode == "bare"), "a retired marker holds nothing"
    finally:
        for held in launcher_root.glob("**/t/*/held"):
            retire_synthetic_retention(held, "probe: stubbed reap")  # the nested run's manufactured marker


_RELEASE_PROBE = """
import json, multiprocessing, os, pathlib, subprocess, sys, time
import pytest

PYTHON, PIDS = sys.executable, pathlib.Path(os.environ["RELEASE_PROBE_OUT"]).with_suffix(".pids")
LEAVE_A_DESCENDANT = ("import pathlib, subprocess, sys; child = subprocess.Popen([sys.executable, '-c', "
                      "'import time; time.sleep(300)']); pathlib.Path(sys.argv[1]).write_text(str(child.pid))")
seen, session_folders = {}, []


@pytest.fixture(autouse=True)
def _a_folder_with_bytes(request, tmp_path):
    seen[request.node.name] = tmp_path
    (tmp_path / "work").write_text("scratch")


@pytest.fixture(scope="session")
def session_folder(tmp_path_factory):
    session_folders.append(tmp_path_factory.mktemp("session"))
    return session_folders[0]


def test_passed(session_folder):
    assert session_folder.is_dir()


def test_failed():
    assert False


def test_skipped():
    pytest.skip("skipped is not failed")


def test_exited_child():
    subprocess.run([PYTHON, "-c", "pass"], check=True)


def test_live_child():
    subprocess.Popen([PYTHON, "-c", "import time; time.sleep(300)"])


def test_live_multiprocessing_child():
    multiprocessing.get_context("spawn").Process(target=time.sleep, args=(300,), daemon=True).start()


@pytest.mark.skipif(os.name == "nt", reason="no census there: a detached child's folder waits for the run")
def test_detached_child_left_nothing():
    subprocess.run([PYTHON, "-c", "pass"], start_new_session=True, check=True)


@pytest.mark.skipif(os.name == "nt", reason="no census there: a detached child's folder waits for the run")
def test_detached_child_left_a_descendant():
    subprocess.run([PYTHON, "-c", LEAVE_A_DESCENDANT, str(PIDS)], start_new_session=True, check=True)


@pytest.mark.parametrize("case", ["first", "second"])
def test_one_basename_for_two_cases_cut_at_thirty(case):
    pass


def test_zz_witness():
    seen.pop("test_zz_witness")
    cut = {path for name, path in seen.items() if name.startswith("test_one_basename")}
    pathlib.Path(os.environ["RELEASE_PROBE_OUT"]).write_text(json.dumps(
        {**{name: path.exists() for name, path in seen.items()}, "session": session_folders[0].is_dir(),
         "a released number stays spent": len(cut) == 2}))
"""


def test_a_passed_tests_folder_goes_with_its_processes_and_all_others_wait_for_the_runs_proof(tmp_path):
    """Live inside the run: what went, what waits. After it the owner's reap and release take the rest."""
    probe, out = tmp_path / "probe", tmp_path / "out.json"
    probe.mkdir()
    (probe / "test_release_probe.py").write_text(_RELEASE_PROBE, encoding="utf-8")
    env = {**os.environ, "RELEASE_PROBE_OUT": str(out)}
    result = run_in_group(_LAUNCHER + ["--temp-parent", str(tmp_path), "--", sys.executable, "-m", "pytest",
                                       "-p", "tests.conftest", "--rootdir", str(probe), "--confcutdir", str(probe),
                                       "-o", "addopts=", "-q", str(probe)], cwd=REPO, env=env, timeout=120)
    pids = out.with_suffix(".pids")
    survivors = kill_survivors([int(pids.read_text())]) if pids.exists() else []
    assert "1 failed" in result.stdout, result.stdout + result.stderr
    gone = {"test_passed", "test_skipped", "test_exited_child", "test_one_basename_for_two_cases_cut_at_thirty[first]"}
    waiting = {"test_failed", "test_live_child", "test_live_multiprocessing_child", "session", "a released number stays spent"}
    if os.name != "nt":
        gone, waiting = gone | {"test_detached_child_left_nothing"}, waiting | {"test_detached_child_left_a_descendant"}
    present = json.loads(out.read_text(encoding="utf-8"))
    assert {name for name, there in present.items() if there} >= waiting, present
    assert not {name for name, there in present.items() if there} & gone, present
    assert _final_root(result.stderr)[0] == "SAFE_TEST_RELEASED", result.stderr
    assert not survivors
    assert f"{len(gone) + 2} released" in result.stdout, result.stdout  # the second cut name and the witness too
