"""Real parent/child writes and concurrent runs stay on disposable roots."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ouroboros.preflight_runner import _preflight_env
from ouroboros.settings_defaults import settings_env_keys
from ouroboros.test_environment import isolated_environment, settings_keys

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.serial


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


_WRITE_ROOTS = """
import json, os, pathlib
from ouroboros import config
from supervisor import update_merge
roots = [config.DATA_DIR, pathlib.Path(config.get_subagent_projects_root()),
         pathlib.Path(config.get_subagent_worktree_root()), pathlib.Path(config.get_deliverables_root()),
         config.APP_ROOT, pathlib.Path.home(), pathlib.Path(os.environ['PYTHONUSERBASE'])]
for root in roots:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'probe').write_text('test-only')
config.save_settings({'TOTAL_BUDGET': 0})
update_merge._log_supervisor({'type': 'isolation_probe'})
print(json.dumps([str(root) for root in roots]))
"""


def test_preflight_children_and_concurrent_runs_leave_owner_sentinels_untouched(tmp_path, monkeypatch):
    owner = tmp_path / "owner"
    sentinels = []
    for name in ("data", "projects", "worktrees", "Deliverables", "home", "app", "userbase"):
        path = owner / name / "sentinel"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"unchanged")
        sentinels.append(path)
    for key, name in (("HOME", "home"), ("OUROBOROS_APP_ROOT", "app"),
                      ("OUROBOROS_DATA_DIR", "data"), ("OUROBOROS_SUBAGENT_PROJECTS_ROOT", "projects"),
                      ("OUROBOROS_SUBAGENT_WORKTREE_ROOT", "worktrees"),
                      ("OUROBOROS_DELIVERABLES_ROOT", "Deliverables"), ("PYTHONUSERBASE", "userbase")):
        monkeypatch.setenv(key, str(owner / name))
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://invalid.example")
    before = {path.relative_to(owner): path.read_bytes() for path in sentinels}

    def run(number):
        root = tmp_path / f"run-{number}"
        env = _preflight_env(root, REPO)
        assert not {"OPENAI_API_KEY", "OPENAI_COMPATIBLE_BASE_URL", "OUROBOROS_MANAGED_BY_LAUNCHER"} & env.keys()
        result = subprocess.run([sys.executable, "-c", _WRITE_ROOTS], cwd=REPO, env=env,
                                text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stderr
        paths = json.loads(result.stdout.splitlines()[-1])
        assert all(Path(path).is_relative_to(root) for path in paths)
        assert (root / "data" / "settings.json").is_file()
        assert (root / "data" / "logs" / "supervisor.jsonl").is_file()
        return set(paths)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(run, range(2))
    assert first.isdisjoint(second)
    assert {path.relative_to(owner): path.read_bytes() for path in owner.rglob("*") if path.is_file()} == before


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


def test_boundary_is_established_before_runtime_imports_in_nested_collection(tmp_path):
    root = tmp_path / "boundary"
    env = isolated_environment(root, REPO)
    result = subprocess.run([sys.executable, "-c", "import tests.conftest\n" + _WRITE_ROOTS],
                            cwd=REPO, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout.splitlines()[-1])
    assert all(Path(path).is_relative_to(root) for path in paths)


def test_safe_test_refuses_a_disposable_root_inside_the_checkout(tmp_path):
    """A nested root is not disposable: git walks up from it into THIS working tree."""
    launcher = [sys.executable, "-I", "-S", str(REPO / "scripts" / "safe_test.py")]
    probe = [str(sys.executable), "-c", "print('boundary ran')"]

    refused = subprocess.run(launcher + ["--temp-parent", str(REPO / "build"), "--"] + probe,
                             cwd=REPO, text=True, capture_output=True, timeout=120)
    assert refused.returncode == 2, refused.stdout
    assert "outside the repository working tree" in refused.stderr
    assert "boundary ran" not in refused.stdout

    accepted = subprocess.run(launcher + ["--temp-parent", str(tmp_path), "--"] + probe,
                              cwd=REPO, text=True, capture_output=True, timeout=120)
    assert accepted.returncode == 0, accepted.stderr
    assert "boundary ran" in accepted.stdout

    # Any enclosing checkout is the same hazard, not only this one.
    outer = tmp_path / "outer-checkout"
    subprocess.run(["git", "init", str(outer)], check=True, capture_output=True)
    (outer / "nested").mkdir()
    before = sorted(path.name for path in (outer / "nested").iterdir())
    refused = subprocess.run(launcher + ["--temp-parent", str(outer / "nested"), "--"] + probe,
                             cwd=REPO, text=True, capture_output=True, timeout=120)
    assert refused.returncode == 2, refused.stdout
    assert "must not be inside a Git working tree" in refused.stderr
    assert sorted(path.name for path in (outer / "nested").iterdir()) == before == []


def test_safe_test_retains_its_new_root_and_keeps_the_command_status(tmp_path):
    """Exit is no proof that descendants are gone: the launcher never deletes its tree."""
    launcher = [sys.executable, "-I", "-S", str(REPO / "scripts" / "safe_test.py")]
    probe = [sys.executable, "-c", "import os, pathlib, sys; "
             "pathlib.Path(os.environ['HOME'], 'probe').write_text('kept'); "
             "print(os.environ['OUROBOROS_TEST_TEMP_ROOT']); sys.exit(7)"]
    result = subprocess.run(launcher + ["--temp-parent", str(tmp_path), "--"] + probe,
                            cwd=REPO, text=True, capture_output=True, timeout=120)
    assert result.returncode == 7, result.stderr  # The command's own status, retention is not failure.
    retained = [line.split(" ", 1)[1] for line in result.stderr.splitlines()
                if line.startswith("SAFE_TEST_RETAINED ")]
    assert len(retained) == 1, result.stderr
    root = Path(retained[0])
    assert root.parent == tmp_path.resolve() and root.name.startswith("ob-")
    assert (root / "home" / "probe").read_text() == "kept"
    # Handed down so each pytest controller claims its own basetemp beneath it.
    assert Path(result.stdout.splitlines()[-1]) == root


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


def test_two_sequential_pytest_sessions_under_one_launcher_root_never_share_a_basetemp(tmp_path):
    """scripts/run_tests.py --sequential runs two pytest processes with the SAME env."""
    import os

    launcher_root = tmp_path / "launcher-root"
    launcher_root.mkdir()
    env = {**os.environ, "OUROBOROS_TEST_TEMP_ROOT": str(launcher_root)}
    node = "tests/test_test_environment.py::test_git_discovery_cannot_escape_to_an_ancestor_checkout"
    basetemps = []
    for _pass in range(2):
        result = subprocess.run([sys.executable, "-m", "pytest", node, "-o", "addopts=", "-q"],
                                cwd=REPO, env=env, text=True, capture_output=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        line = next(row for row in result.stdout.splitlines() if row.startswith("test session trees retained"))
        basetemps.append(Path(line.rsplit(", ", 1)[-1]))
    assert basetemps == [launcher_root / "b0" / "t", launcher_root / "b1" / "t"]
    # The second session neither reused nor emptied the first one's tree.
    for basetemp in basetemps:
        assert any(basetemp.iterdir()), basetemp


_NODE = "tests/test_test_environment.py::test_git_discovery_cannot_escape_to_an_ancestor_checkout"

_SESSION_PROBE = """
import json, os, pathlib, tempfile

def pytest_sessionstart(session):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    record = {"tmpdir": os.environ.get("TMPDIR", ""), "gettempdir": tempfile.gettempdir(),
              "session_root": str(pathlib.Path(os.environ["OUROBOROS_DATA_DIR"]).parent)}
    pathlib.Path(os.environ["SESSION_PROBE_OUT"], worker + ".json").write_text(json.dumps(record))
"""


def _nested_env(**overrides):
    import os

    # The nested run is its own session: when THIS test runs inside an xdist
    # worker, the inherited worker id would make the nested controller write
    # its probe as that worker and the record set would never show "controller".
    dropped = {"OUROBOROS_TEST_TEMP_ROOT", "PYTEST_XDIST_WORKER", "PYTEST_XDIST_TESTRUNUID"}
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    env.update(overrides)
    return env


@pytest.mark.parametrize("mode", ["launcher", "bare"])
def test_every_pytest_process_takes_a_short_sibling_session_root(tmp_path, mode):
    """A real xdist run: workers inherit the controller's TMPDIR, yet none nests beneath it."""
    pytest.importorskip("xdist")
    probe, out, parent = tmp_path / "probe", tmp_path / "out", tmp_path / "parent"
    for path in (probe, out, parent):
        path.mkdir()
    (probe / "session_probe.py").write_text(_SESSION_PROBE, encoding="utf-8")
    # Under the launcher its root is the parent; bare pytest keeps an explicitly chosen TMPDIR.
    selected = {"OUROBOROS_TEST_TEMP_ROOT": str(parent)} if mode == "launcher" else {"TMPDIR": str(parent)}
    env = _nested_env(PYTHONPATH=str(probe), SESSION_PROBE_OUT=str(out), **selected)
    result = subprocess.run([sys.executable, "-m", "pytest", _NODE, "-o", "addopts=", "-q",
                             "-n", "2", "-p", "session_probe"],
                            cwd=REPO, env=env, text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    records = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in out.glob("*.json")}
    assert set(records) == {"controller", "gw0", "gw1"}, records
    roots = {Path(record["session_root"]) for record in records.values()}
    assert len(roots) == 3 and {root.parent for root in roots} == {parent}, roots
    for record in records.values():
        expected = str(Path(record["session_root"]) / "tmp") if mode == "launcher" else str(parent)
        assert record["tmpdir"] == record["gettempdir"] == expected, record
        assert len(record["tmpdir"]) <= len(str(parent)) + len("/p12345678/tmp"), record


def test_bare_pytest_with_zero_retention_count_keeps_its_session_tree(tmp_path):
    """pytest's numbered basetemp would delete the CURRENT session tree at exit here."""
    result = subprocess.run([sys.executable, "-m", "pytest", _NODE, "-o", "addopts=", "-q",
                             "-o", "tmp_path_retention_count=0"],
                            cwd=REPO, env=_nested_env(TMPDIR=str(tmp_path)),
                            text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(row for row in result.stdout.splitlines() if row.startswith("test session trees retained"))
    root = Path(line.split(": ", 1)[1].split(", ")[0])
    assert root.parent == tmp_path, line
    kept = root / "b0" / "t"
    assert kept.is_dir() and any(kept.iterdir()), sorted(root.rglob("*"))[:20]


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
