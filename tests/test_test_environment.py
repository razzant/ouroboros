"""Real parent/child writes and concurrent runs stay on disposable roots."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from ouroboros.platform_layer import kill_process_tree
from ouroboros.preflight_runner import _preflight_env
from ouroboros.settings_defaults import settings_env_keys
from ouroboros.test_environment import isolated_environment, settings_keys
from tests.group_run import kill_survivors, run_in_group

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
    """Unknown command data remains even after reap; only declared scratch is released."""
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
    # Spotlight skips everything below it (fake app bundles among the fixtures), and it stays short:
    # nested in a session's /tmp/obs-* scratch, a pool listener below it still fits AF_UNIX's 104 bytes.
    assert root.name.endswith(".noindex") and len(root.name) == len("ob-12345678.noindex")
    assert (root / "home" / "probe").read_text() == "kept"
    # Handed down so each pytest controller claims its own basetemp beneath it.
    assert Path(result.stdout.splitlines()[-1]) == root


# The command keeps a child in a session of its own, as tests/group_run.py keeps node's runner.
# It records the SIGTERM it gets; an "obedient" command then exits, a "stubborn" one keeps running,
# so only the SIGKILL that follows can stop it.
_TREE = """\
import os, pathlib, signal, subprocess, sys, time
own = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], **own)
def term(*_):
    pathlib.Path(sys.argv[1] + ".term").write_text("TERM")
    if sys.argv[2] == "obedient":
        os._exit(0)
if os.name != "nt":
    signal.signal(signal.SIGTERM, term)
pathlib.Path(sys.argv[1]).write_text(f"{os.getpid()} {child.pid}")
time.sleep(600)
"""
_LAUNCHER = [sys.executable, "-I", "-S", str(REPO / "scripts" / "safe_test.py")]


def _final_root(stderr):
    """Exactly one final line names the launcher's root, and it is the last one."""
    lines = stderr.splitlines()
    final = [line for line in lines if line.startswith(("SAFE_TEST_RETAINED ", "SAFE_TEST_RELEASED "))]
    assert len(final) == 1 and lines[-1] == final[0], stderr
    return final[0].split(" ", 1)


def _recorded_tree(path, within=30.0):
    deadline = time.monotonic() + within
    while not (path.exists() and len(path.read_text(encoding="utf-8").split()) == 2):
        assert time.monotonic() < deadline, "the command never recorded its tree"
        time.sleep(0.05)
    return [int(pid) for pid in path.read_text(encoding="utf-8").split()]


def test_safe_test_timeout_stops_the_whole_tree_and_exits_124(tmp_path):
    """The deadline reaches a child outside the command's group and a command deaf to SIGTERM."""
    pids = tmp_path / "pids"
    result = run_in_group(_LAUNCHER + ["--temp-parent", str(tmp_path), "--timeout", "3", "--",
                                       sys.executable, "-c", _TREE, str(pids), "stubborn"], cwd=REPO, timeout=120)
    survivors = kill_survivors(_recorded_tree(pids))
    assert result.returncode == 124, result.stderr
    assert "SAFE_TEST_TIMEOUT after 3s" in result.stderr
    # Teardown was confirmed, so the declared scratch went even on this red outcome.
    assert _final_root(result.stderr)[0] == "SAFE_TEST_RELEASED", result.stderr
    if sys.platform != "win32":  # the graceful SIGTERM came before the SIGKILL that stopped it
        assert (tmp_path / "pids.term").read_text(encoding="utf-8") == "TERM"
    assert not survivors


def test_safe_test_timeout_keeps_a_finished_command_status(tmp_path):
    result = run_in_group(_LAUNCHER + ["--temp-parent", str(tmp_path), "--timeout", "60", "--",
                                       sys.executable, "-c", "import sys; sys.exit(7)"], cwd=REPO, timeout=120)
    assert result.returncode == 7, result.stderr
    assert "SAFE_TEST_TIMEOUT" not in result.stderr and _final_root(result.stderr)
    refused = run_in_group(_LAUNCHER + ["--timeout", "0", "--", sys.executable, "-c", "pass"],
                           cwd=REPO, timeout=120)
    assert refused.returncode == 2 and "positive number of seconds" in refused.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("stop", ["SIGTERM", "SIGINT"])
def test_safe_test_forwards_interrupt_then_reaps_the_whole_tree(tmp_path, stop):
    """INT lets the root finish; TERM requests tree stop; both reap detached children."""
    stop = getattr(signal, stop)
    pids = tmp_path / "pids"
    launcher = subprocess.Popen(_LAUNCHER + ["--temp-parent", str(tmp_path), "--timeout", "600", "--",
                                             sys.executable, "-c", _TREE, str(pids), "obedient"],
                                cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
    tree = []
    try:
        tree = _recorded_tree(pids)
        os.kill(launcher.pid, stop)
        os.kill(launcher.pid, stop)  # a repeated signal cannot cut the teardown short
        _, stderr = launcher.communicate(timeout=60)
    finally:
        if launcher.poll() is None:
            kill_process_tree(launcher)
            launcher.wait()
        survivors = kill_survivors(tree)  # a launcher that failed to stop the tree must not leave it behind
    assert launcher.returncode == 128 + stop, stderr
    assert _final_root(stderr)[0] == "SAFE_TEST_RELEASED", stderr
    if stop == signal.SIGTERM:
        assert (tmp_path / "pids.term").read_text(encoding="utf-8") == "TERM"
    else:
        assert not (tmp_path / "pids.term").exists(), "Ctrl-C was incorrectly translated to TERM"
    assert not survivors


def test_a_declared_launcher_releases_the_installation_pth_it_wrote_and_keeps_anything_else(tmp_path):
    """Started inside a declared session (a prepared Update's checks on a user-site installation), the launcher's own
    fresh user base gets the installation ``.pth`` (``installation_user_site``), so its command reads the declared
    site. The run releases its root with that one file; a user base holding anything else, or that file rewritten,
    holds the command's own data and keeps the root."""
    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, user_site_packages

    _installation_base(tmp_path / "installation")
    env = {**os.environ, INSTALLATION_USER_BASE_ENV: str(tmp_path / "installation"), "PYTHONNOUSERSITE": ""}
    pth = "pathlib.Path(site.getusersitepackages(), 'ouroboros-installation-user-site.pth')"
    seen = {}
    for label, then in (("released", "pass"), ("other_file", "pathlib.Path(site.getuserbase(), 'note.txt').write_text('kept')"),
                        ("rewritten", f"{pth}.write_text('import sys\\n')")):
        command = [sys.executable, "-c", f"import pathlib, site\nassert {pth}.is_file(), {pth}\n{then}\n"]
        done = run_in_group(_LAUNCHER + ["--temp-parent", str(tmp_path), "--", *command], cwd=REPO, env=env,
                            timeout=120)
        assert done.returncode == 0, done.stderr
        state, root = _final_root(done.stderr)
        seen[label] = (state, Path(root))
    assert {label: state for label, (state, _root) in seen.items()} == {
        "released": "SAFE_TEST_RELEASED", "other_file": "SAFE_TEST_RETAINED", "rewritten": "SAFE_TEST_RETAINED"}
    assert not seen["released"][1].exists()
    assert (seen["other_file"][1] / "userbase" / "note.txt").read_text(encoding="utf-8") == "kept"
    kept = Path(user_site_packages(seen["rewritten"][1] / "userbase")) / "ouroboros-installation-user-site.pth"
    assert kept.read_text(encoding="utf-8") == "import sys\n"


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
    worker = getattr(session.config, "workerinput", {}).get("workerid", "controller")
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
    # Two real workers are the scenario, so this child is explicitly allocated two slots.
    env = _nested_env(PYTHONPATH=str(probe), SESSION_PROBE_OUT=str(out), OURO_TEST_WORKER_BUDGET="2", **selected)
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


@pytest.mark.parametrize("mode", ["leave", "retire"])
def test_a_marker_left_in_a_tests_tmp_path_fails_that_test_under_a_releasing_owner(tmp_path, mode):
    """#1908: one forgotten synthetic marker kept every green battery's whole root."""
    from tests._shared import retire_synthetic_retention

    probe, launcher_root = tmp_path / "probe", tmp_path / "launcher-root"
    probe.mkdir()
    launcher_root.mkdir()
    (probe / "retention_probe.py").write_text(_RETENTION_PROBE, encoding="utf-8")
    env = _nested_env(PYTHONPATH=str(probe), RETENTION_PROBE=mode, OUROBOROS_TEST_TEMP_ROOT=str(launcher_root))
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
        else:
            assert result.returncode == 0, output
    finally:
        for held in launcher_root.glob("b*/t/*/held"):
            retire_synthetic_retention(held, "probe: stubbed reap")  # the nested run's manufactured marker


def _installation_base(root, label="installation"):
    """A user base keeping an installation's packages for this interpreter: a module, a ``.pth`` path line to another
    and a ``.pth`` startup line, plus a stdlib name it also holds."""
    from ouroboros.test_environment import user_site_packages

    site = Path(user_site_packages(root))
    for rel, text in {"gen_plain.py": f"WHERE = {label!r}\n", "gen_pth_dir/gen_pth_added.py": f"WHERE = {label!r}\n",
                      "gen_pth_hook.py": f"import sys\nsys._gen_pth_ran = {label!r}\n",
                      "tabnanny.py": f"WHERE = {label!r}\n",
                      "zz_gen.pth": "gen_pth_dir\nimport gen_pth_hook\n"}.items():
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_text(text, encoding="utf-8")
    return site


def test_an_installation_user_site_is_read_only_when_declared_and_a_bad_declaration_is_named(tmp_path):
    """Nothing declared changes nothing; a declared base that is relative or missing is named; an existing base without
    user site-packages for this interpreter (a packaged boot's ``pip --user`` the bundle already satisfied) adds
    nothing, no ``.pth``; one with them gets the ``.pth``."""
    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, installation_user_site, user_site_packages

    fresh = tmp_path / "fresh"
    assert installation_user_site({}, fresh) == "" and not fresh.exists()  # nothing declared: nothing changes
    for declared in ("relative/base", str(tmp_path / "gone")):
        with pytest.raises(ValueError, match=f"^{INSTALLATION_USER_BASE_ENV}="):
            installation_user_site({INSTALLATION_USER_BASE_ENV: declared}, fresh)
    assert installation_user_site({INSTALLATION_USER_BASE_ENV: str(tmp_path)}, fresh) == "" and not fresh.exists()
    site = _installation_base(tmp_path / "installation")
    assert installation_user_site({INSTALLATION_USER_BASE_ENV: str(tmp_path / "installation")}, fresh) == str(site)
    assert (Path(user_site_packages(fresh)) / "ouroboros-installation-user-site.pth").read_text(encoding="utf-8") == (
        f"import site; site.addsitedir({str(site)!r})\n")


def test_the_installation_pth_line_reads_the_same_path_in_any_locale_encoding(tmp_path):
    """CPython 3.10-3.12 decode a ``.pth`` in the locale's encoding when UTF-8 mode is off (Windows ships 3.10, and a
    profile path may be Cyrillic): the line is pure ASCII, so decoded as Latin-1 or cp1251 it names the same path. No
    non-UTF-8 locale needs to be installed: the bytes are decoded here as those encodings would."""
    import ast

    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, installation_user_site, user_site_packages

    base = tmp_path / "Антон Ä" / "python-userbase"
    Path(user_site_packages(base)).mkdir(parents=True)
    fresh = tmp_path / "fresh"
    site = installation_user_site({INSTALLATION_USER_BASE_ENV: str(base)}, fresh)
    data = (Path(user_site_packages(fresh)) / "ouroboros-installation-user-site.pth").read_bytes()
    assert data.isascii(), data
    for encoding in ("latin-1", "cp1251", "utf-8"):
        line = data.decode(encoding)
        assert line.startswith("import site; site.addsitedir(") and line.endswith(")\n")
        call = ast.parse(line.split("; ", 1)[1]).body[0].value
        assert ast.literal_eval(call.args[0]) == site == user_site_packages(base), encoding


def test_a_session_started_with_its_own_pythonnousersite_keeps_it_and_writes_no_pth(tmp_path):
    """A test that isolated itself (``PYTHONNOUSERSITE=1``) and starts a nested session that inherits the declaration
    keeps its choice: the declaration is still checked, but no ``.pth`` is written, so the nested session's conftest
    leaves its defaults' ``PYTHONNOUSERSITE=1`` for its children. An empty value is unset for Python: it reads."""
    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, installation_user_site, user_site_packages

    site = _installation_base(tmp_path / "installation")
    declared = {INSTALLATION_USER_BASE_ENV: str(tmp_path / "installation")}
    for value in ("1", "yes"):
        fresh = tmp_path / f"fresh-{value}"
        assert installation_user_site({**declared, "PYTHONNOUSERSITE": value}, fresh) == "" and not fresh.exists()
    with pytest.raises(ValueError, match=f"^{INSTALLATION_USER_BASE_ENV}="):
        installation_user_site({INSTALLATION_USER_BASE_ENV: str(tmp_path / "gone"), "PYTHONNOUSERSITE": "1"},
                               tmp_path / "fresh-gone")
    fresh = tmp_path / "fresh-reads"
    assert installation_user_site({**declared, "PYTHONNOUSERSITE": ""}, fresh) == str(site)
    assert (Path(user_site_packages(fresh)) / "ouroboros-installation-user-site.pth").is_file()


def test_a_declared_installation_site_reads_as_cpythons_site_processing_orders_it_and_writes_stay_fresh(tmp_path):
    """Through the fresh user base's ``.pth``, CPython's own site processing reads the declared site after that fresh
    user site and before the interpreter's own site-packages: its module, its ``.pth`` path and startup lines, the
    stdlib still first; the process's own user site, where ``pip --user`` writes, stays the fresh one. ``-s``, ``-I``
    and a child's own ``PYTHONNOUSERSITE=1`` turn all of it off."""
    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, installation_user_site, user_site_packages

    # The interpreter this suite's venv was made from: outside a venv, a user site works.
    base = Path(sys.base_prefix, "python.exe" if os.name == "nt" else f"bin/python{sys.version_info[0]}."
                                                                      f"{sys.version_info[1]}")
    python = str(base) if base.is_file() else sys.executable
    clean = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    fresh = tmp_path / "fresh"
    enabled = subprocess.run([python, "-c", "import site; print(site.ENABLE_USER_SITE)"], capture_output=True,
                             text=True, env={**clean, "PYTHONUSERBASE": str(fresh),
                                             "PYTHONNOUSERSITE": ""}).stdout.strip()
    if enabled != "True":
        pytest.skip(f"{python} enables no user site")
    declared = _installation_base(tmp_path / "installation")
    installation_user_site({INSTALLATION_USER_BASE_ENV: str(tmp_path / "installation")}, fresh)
    probe = ("import json, os, site, sys, sysconfig\n"
             "def where(name):\n"
             "    try:\n        return getattr(__import__(name), 'WHERE', 'stdlib')\n"
             "    except ImportError:\n        return None\n"
             "paths = [os.path.normcase(os.path.abspath(p or os.curdir)) for p in sys.path]\n"
             "at = lambda p: paths.index(os.path.normcase(os.path.abspath(p))) if os.path.normcase(os.path.abspath(p))"
             " in paths else None\n"
             "print(json.dumps([where('gen_plain'), where('gen_pth_added'), getattr(sys, '_gen_pth_ran', None),\n"
             "    where('tabnanny'), [at(os.path.dirname(os.__file__)), at(site.getusersitepackages()),\n"
             "    at(sys.argv[1]), at(sysconfig.get_paths()['purelib'])], site.getusersitepackages()]))\n")

    def child(*flags, **extra):
        env = {**clean, "PYTHONUSERBASE": str(fresh), "PYTHONNOUSERSITE": "", **extra}  # this conftest's too
        done = subprocess.run([python, *flags, "-c", probe, str(declared)], capture_output=True, text=True, env=env,
                              timeout=120)
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)

    plain, added, ran, shadow, order, user_site = child()
    assert (plain, added, ran, shadow) == ("installation", "installation", "installation", "stdlib")
    assert None not in order and order == sorted(order), order  # stdlib, fresh user site, declared, site-packages
    assert user_site == user_site_packages(fresh) != str(declared)
    for flags, extra in (("-s",), {}), (("-I",), {}), ((), {"PYTHONNOUSERSITE": "1"}):
        assert child(*flags, **extra)[:4] == [None, None, None, "stdlib"], (flags, extra)


def test_product_children_of_a_declared_session_read_the_installation_and_an_explicit_opt_out_stays_out(tmp_path):
    """On a packaged install the declared user base is the product's own environment: a package a dependency-changing
    Update brings may exist only there. A test process of a declared session (the real conftest, its child isolation
    installed) starts product children through ``isolated_environment`` or with an environment it cleared: each reads
    the installation through its own fresh user base, so W' does not fail "only with the update" on an import. The
    same children with an explicit ``PYTHONNOUSERSITE=1`` stay out. The gate's own environment (``_preflight_env``)
    keeps declaring nothing: its producer declares (``update_checks._user_base``)."""
    import sysconfig

    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, user_site_packages

    base = Path(sys.base_prefix, "python.exe" if os.name == "nt" else f"bin/python{sys.version_info[0]}."
                                                                      f"{sys.version_info[1]}")
    python = str(base) if base.is_file() else sys.executable
    clean = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    enabled = subprocess.run([python, "-c", "import site; print(site.ENABLE_USER_SITE)"], capture_output=True,
                             text=True, env={**clean, "PYTHONUSERBASE": str(tmp_path / "probe"),
                                             "PYTHONNOUSERSITE": ""}).stdout.strip()
    if enabled != "True":
        pytest.skip(f"{python} enables no user site")
    installation = tmp_path / "installation"
    site = Path(user_site_packages(installation))
    site.mkdir(parents=True)
    (site / "only_in_installation.py").write_text("WHERE = 'installation'\n", encoding="utf-8")
    # The suite's own packages (pytest among them) for that interpreter, through the same site.
    (site / "zz_suite_packages.pth").write_text("".join(f"{path}\n" for path in dict.fromkeys(
        (sysconfig.get_path("purelib"), sysconfig.get_path("platlib")))), encoding="utf-8")
    script = (
        "import json, os, pathlib, subprocess, sys\n"
        "import tests.conftest as harness\n"
        "harness._install_pytest_child_isolation()\n"
        "from ouroboros.test_environment import isolated_environment\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "def reads(env):\n"
        "    return subprocess.run([sys.executable, '-c', 'import only_in_installation'], env=env,\n"
        "                          capture_output=True, timeout=120).returncode == 0\n"
        "isolated = isolated_environment(root / 'isolated', pathlib.Path.cwd())\n"
        "cleared = {'PATH': os.environ.get('PATH', '')}\n"
        "from ouroboros.preflight_runner import _preflight_env\n"
        "gate = _preflight_env(root / 'gate', pathlib.Path.cwd())\n"
        "print(json.dumps({'isolated': reads(isolated),"
        " 'isolated_opt_out': reads({**isolated, 'PYTHONNOUSERSITE': '1'}), 'cleared': reads(cleared), 'cleared_opt_out': reads({**cleared, 'PYTHONNOUSERSITE': '1'}),"
        f" 'gate': [gate['PYTHONNOUSERSITE'], {INSTALLATION_USER_BASE_ENV!r} in gate]}}))\n")
    env = {**isolated_environment(tmp_path / "session", REPO), "PYTHONUSERBASE": str(installation),
           "PYTHONNOUSERSITE": "", INSTALLATION_USER_BASE_ENV: str(installation)}  # as update_checks._user_base does
    done = subprocess.run([python, "-c", script, str(tmp_path / "children")], cwd=REPO, env=env, text=True,
                          capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == {
        "isolated": True, "isolated_opt_out": False, "cleared": True, "cleared_opt_out": False, "gate": ["1", False]}


def test_the_test_conftest_reads_a_declared_installation_site_and_otherwise_keeps_its_isolated_defaults(tmp_path):
    """The real conftest, imported as a nested session imports it: without the declaration its children get the
    isolated defaults exactly as before; with one (and the empty ``PYTHONNOUSERSITE`` the checks pass) they keep the
    fresh user base and get its ``.pth``, the user site left enabled for every process that keeps the environment; a
    declared base without site-packages, or a session started with its own ``PYTHONNOUSERSITE=1``, keeps the
    defaults; an invalid one stops the session, named."""
    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, user_site_packages

    site = _installation_base(tmp_path / "installation")
    (tmp_path / "empty-userbase").mkdir()
    report = ("import json, os\nprint(json.dumps({key: os.environ.get(key) for key in "
              "('PYTHONUSERBASE', 'PYTHONNOUSERSITE')}))\n")
    seen = {}
    # Each case declares only what it names: a declaration this suite itself runs under (a prepared Update's checks)
    # never reaches the "none" case, which is a session without one.
    ambient = {key: value for key, value in os.environ.items() if key != INSTALLATION_USER_BASE_ENV}
    for label, declared, nousersite in (("none", None, "1"), ("valid", str(tmp_path / "installation"), ""),
                                        ("isolated", str(tmp_path / "installation"), "1"),
                                        ("empty", str(tmp_path / "empty-userbase"), ""),
                                        ("bad", str(tmp_path / "gone"), "")):
        env = {**isolated_environment(tmp_path / label, REPO, source=ambient), "PYTHONNOUSERSITE": nousersite}
        if declared:
            env[INSTALLATION_USER_BASE_ENV] = declared
        assert env.get(INSTALLATION_USER_BASE_ENV) == declared, label
        seen[label] = subprocess.run([sys.executable, "-c", "import tests.conftest\n" + report], cwd=REPO, env=env,
                                     text=True, capture_output=True, timeout=120)
    none, valid, isolated, empty = (json.loads(seen[label].stdout.splitlines()[-1])
                                    for label in ("none", "valid", "isolated", "empty"))
    for kept in (none, isolated, empty):
        assert kept["PYTHONNOUSERSITE"] == "1" and not Path(user_site_packages(kept["PYTHONUSERBASE"])).exists()
    assert valid["PYTHONNOUSERSITE"] == "" and Path(valid["PYTHONUSERBASE"]).is_relative_to(tmp_path / "valid")
    assert (Path(user_site_packages(valid["PYTHONUSERBASE"])) / "ouroboros-installation-user-site.pth").read_text(
        encoding="utf-8") == f"import site; site.addsitedir({str(site)!r})\n"
    assert seen["bad"].returncode != 0 and f"{INSTALLATION_USER_BASE_ENV}=" in seen["bad"].stderr
