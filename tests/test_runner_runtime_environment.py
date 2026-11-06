"""Real nested runtimes use the selected Node and a socket-capable macOS temp root."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from ouroboros import preflight_node as node, preflight_runner, test_environment
from tests.test_preflight_node import requires_node
from tests.test_preflight_receipts import _git, _tiny_repo
from tests.test_run_tests_script import run_tests as run_tests

REPO = Path(__file__).resolve().parents[1]
SHORT_TEMP = pytest.mark.skipif(sys.platform != "darwin" and not sys.platform.startswith("linux"),
                                reason="the short AF_UNIX temp-root default is macOS and Linux")
POSIX = pytest.mark.skipif(os.name == "nt", reason="the deliberate bad PATH executable is a POSIX shell fixture")


def _preferred_node_over_broken_path(tmp_path, monkeypatch):
    selected = node.resolve_node()
    assert selected
    broken = tmp_path / "broken-bin"
    broken.mkdir()
    shim = broken / "node"
    shim.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(broken) + os.pathsep + os.environ.get("PATH", ""))
    assert subprocess.run(["node", "--version"], capture_output=True, timeout=10).returncode == 91
    monkeypatch.setattr(node, "resolve_bundled_node", lambda: selected)
    return selected


def test_node_environment_keeps_other_inputs_and_does_not_require_a_runtime(tmp_path, monkeypatch):
    selected = str(tmp_path / "bundle" / "node")
    original = {"PATH": os.pathsep.join(["first", str(Path(selected).parent), "last"]), "KEEP": "value"}
    before = dict(original)
    monkeypatch.setattr(node, "resolve_node", lambda: selected)
    monkeypatch.setattr(node, "probe_node_version", lambda *_: pytest.fail("environment projection is not a health probe"))
    projected = node.node_environment(original)
    assert projected["PATH"].split(os.pathsep) == [str(Path(selected).parent), "first", "last"]
    assert projected["KEEP"] == "value" and original == before
    assert node.node_environment(projected) == projected
    monkeypatch.setattr(node, "resolve_node", lambda: None)
    assert node.node_environment(original) == original
    assert node.node_environment(original) is not original


@POSIX
def test_path_fallback_keeps_virtualenv_tools_before_the_node_directory(tmp_path, monkeypatch):
    first, later = tmp_path / "venv-bin", tmp_path / "system-bin"
    first.mkdir()
    later.mkdir()
    for directory, command, marker in (
            (first, "python3", "venv-python3"), (first, "git", "venv-git"),
            (later, "python3", "system-python3"), (later, "git", "system-git"),
            (later, "node", "system-node")):
        executable = directory / command
        executable.write_text(f'#!/bin/sh\nprintf "%s\\n" "{marker}"\n', encoding="utf-8")
        executable.chmod(0o755)
    original = {"PATH": os.pathsep.join((str(first), str(later))), "KEEP": "value"}
    monkeypatch.setenv("PATH", original["PATH"])
    monkeypatch.setattr(node, "resolve_bundled_node", lambda: None)
    selected = node.resolve_node()
    assert selected == str(later / "node")
    projected = node.node_environment(original)
    assert projected == original and projected is not original
    assert shutil.which("node", path=projected["PATH"]) == selected
    for command, marker in (("python3", "venv-python3"), ("git", "venv-git"), ("node", "system-node")):
        result = subprocess.run([command], env=projected, text=True, capture_output=True, timeout=10)
        assert (result.returncode, result.stdout.strip()) == (0, marker), result.stderr


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_temp_parent_is_short_on_macos_and_linux_and_the_platform_default_elsewhere(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(test_environment.sys, "platform", platform)
    monkeypatch.setattr(test_environment.tempfile, "gettempdir", lambda: str(tmp_path))
    assert test_environment.default_temp_parent() == (tmp_path if platform == "win32" else Path("/tmp"))
    # Session scratch alone moves out of the OS cleaners' /tmp (O3, owner Q1=A narrowed by A-1):
    # the preflight, commit-admission and safe_test parent above is unchanged.
    assert test_environment.session_scratch_parent() == (tmp_path if platform == "win32" else Path("/var/tmp"))


@SHORT_TEMP
def test_session_scratch_parent_keeps_the_deepest_nested_manager_socket_inside_the_bound(tmp_path):
    """The real witness of the four extra bytes: safe_test rooted in a session scratch under
    ``session_scratch_parent()`` (as a delegated session's instructions ask), then a real manager
    socket in both nested layouts a run makes there: an owned process temp (``c<8>``) and the pytest
    session root with its ``tmp`` (``p<8>/tmp``, the deepest: 95 of 104 bytes under ``/private/var/tmp``)."""
    parent = test_environment.session_scratch_parent().resolve()
    scratch = Path(tempfile.mkdtemp(prefix="obs-", dir=parent))
    probe = tmp_path / "probe.py"
    probe.write_text(f'''
import json, multiprocessing, os, pathlib, sys, tempfile
sys.path.insert(0, {str(REPO)!r})
from ouroboros.test_environment import declare_scratch, owned_process_temp
if __name__ == "__main__":  # a spawned manager child re-imports this file
    root = pathlib.Path(os.environ["OUROBOROS_TEST_TEMP_ROOT"])
    session = pathlib.Path(tempfile.mkdtemp(prefix="p", dir=root))  # tests/conftest.py's pytest root
    declare_scratch(session, "pytest-session")
    found = {{}}
    for layout, temp in (("process", owned_process_temp()), ("pytest", session / "tmp")):
        temp.mkdir(exist_ok=True)
        os.environ["TMPDIR"] = tempfile.tempdir = str(temp)
        with multiprocessing.Manager() as manager:
            queue = manager.Queue()
            queue.put("owned")
            assert queue.get(timeout=5) == "owned"
            found[layout] = {{"address": manager.address, "tmpdir": str(temp)}}
    print(json.dumps(found))
''', encoding="utf-8")
    try:
        result = subprocess.run([sys.executable, "-I", "-S", str(REPO / "scripts/safe_test.py"), "--temp-parent",
                                 str(scratch), "--timeout", "180", "--", sys.executable, str(probe)],
                                cwd=REPO, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        found = json.loads(result.stdout.splitlines()[-1])
        for layout in ("process", "pytest"):
            address = Path(found[layout]["address"])
            assert address.is_relative_to(scratch) and Path(found[layout]["tmpdir"]).is_relative_to(scratch), found
            assert len(os.fsencode(address)) < 104, (layout, len(os.fsencode(address)), address)
        assert "SAFE_TEST_RELEASED" in result.stderr and not any(scratch.iterdir())
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_a_nested_gate_lives_under_its_owner_and_a_top_level_gate_keeps_the_default_parent(tmp_path, monkeypatch):
    owner = (tmp_path / "owner").resolve()
    owner.mkdir()
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(owner))
    nested = test_environment.gate_root()
    assert nested.parent == owner and (nested / test_environment.SCRATCH_MARKER).is_file(), "its owner's release takes it"
    monkeypatch.delenv("OUROBOROS_TEST_TEMP_ROOT")
    parent = (tmp_path / "default").resolve()
    parent.mkdir()
    monkeypatch.setattr(test_environment, "default_temp_parent", lambda: parent)
    top = test_environment.gate_root()
    assert top.parent == parent and top.name.startswith("obs-"), "a record of a kept root needs this name"


def test_fingerprint_environment_projection_creates_no_directories(tmp_path):
    root = tmp_path / "not-created"
    env = preflight_runner._preflight_env(root, root / "repo", create=False)
    assert env["OUROBOROS_DATA_DIR"] == str(root / "data")
    assert not root.exists()


@SHORT_TEMP
@requires_node
def test_real_gate_runs_nested_node_and_manager_despite_bad_path_and_long_temp(tmp_path, monkeypatch):
    selected = _preferred_node_over_broken_path(tmp_path, monkeypatch)
    repo = _tiny_repo(tmp_path, web=True)
    node_report, manager_report = tmp_path / "node.json", tmp_path / "manager.json"
    (repo / "tests/test_parallel.py").write_text(f'''
import json, pathlib, shutil, subprocess, sys
CACHED_NODE = shutil.which('node')
def test_nested_node():
    commands = [[CACHED_NODE, '-p', 'process.execPath'], ['node', '-p', 'process.execPath'],
                [sys.executable, '-c', "import subprocess; subprocess.run(['node','-p','process.execPath'],check=True)"]]
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, (result.returncode, result.stderr)
        assert pathlib.Path(result.stdout.strip()).resolve() == pathlib.Path({selected!r}).resolve()
    pathlib.Path({str(node_report)!r}).write_text(json.dumps({{'node': result.stdout.strip()}}), encoding='utf-8')
def test_parallel_sibling():
    assert True
''', encoding="utf-8")
    (repo / "tests/test_serial.py").write_text(f'''
import json, multiprocessing, os, pathlib, pytest
@pytest.mark.serial
def test_real_manager():
    with multiprocessing.Manager() as manager:
        queue = manager.Queue()
        queue.put('owned')
        assert queue.get(timeout=5) == 'owned'
        address = manager.address
        assert len(os.fsencode(address)) < 104
        pathlib.Path({str(manager_report)!r}).write_text(json.dumps({{
            'address': address, 'tmpdir': os.environ['TMPDIR'], 'engine': os.environ.get('OUROBOROS_UI_BROWSER_ENGINE')}}), encoding='utf-8')
''', encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "real nested runtime witnesses")
    ambient = tmp_path / ("long-parent-" * 12)
    ambient.mkdir()
    for key in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(key, str(ambient))
    monkeypatch.setattr(tempfile, "tempdir", str(ambient))
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", "webkit")  # safe_test keeps a person's choice; the gate never does
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "2")  # nested slots: the gate must really run two workers
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_TESTRUNUID", raising=False)
    assert preflight_runner.run_hermetic_pytest(repo, timeout=120, max_output=12000) is None
    assert Path(json.loads(node_report.read_text(encoding="utf-8"))["node"]).resolve() == Path(selected).resolve()
    facts = json.loads(manager_report.read_text(encoding="utf-8"))
    assert Path(facts["tmpdir"]).is_relative_to(Path("/tmp").resolve())
    assert facts["engine"] is None


@SHORT_TEMP
def test_a_gate_rooted_in_the_inherited_long_temp_cannot_hold_its_nested_managers_socket(tmp_path, monkeypatch):
    """The negative of the short default: the same real gate rooted where the old non-macOS rule
    put it (the inherited TMPDIR) gives its nested multiprocessing manager a socket path past the
    AF_UNIX limit (#2110: the reporter's Linux fence and saved-pause failures). The interpreter
    refuses it, or (CPython 3.14) listens in the shared /tmp instead: never inside that parent."""
    repo = _tiny_repo(tmp_path)
    report = tmp_path / "manager.json"
    (repo / "tests/test_serial.py").write_text(
        "import json, multiprocessing, os, pathlib, pytest\n\n\n@pytest.mark.serial\ndef test_real_manager():\n"
        "    with multiprocessing.Manager() as manager:\n        assert manager.Queue() is not None\n"
        f"        pathlib.Path({str(report)!r}).write_text(json.dumps({{\n"
        "            'address': manager.address, 'tmpdir': os.environ['TMPDIR']}), encoding='utf-8')\n",
        encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "a nested manager")
    ambient = tmp_path / ("long-parent-" * 12)
    ambient.mkdir()
    for key in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(key, str(ambient))
    monkeypatch.setattr(tempfile, "tempdir", str(ambient))
    monkeypatch.delenv("OUROBOROS_TEST_TEMP_ROOT", raising=False)  # a top-level gate takes the default parent
    monkeypatch.setattr(test_environment, "default_temp_parent", lambda: Path(tempfile.gettempdir()))
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "2")
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_TESTRUNUID", raising=False)
    error = preflight_runner.run_hermetic_pytest(repo, timeout=120, max_output=12000)
    if error or not report.exists():
        assert error and "AF_UNIX path too long" in error, error
    else:
        facts = json.loads(report.read_text(encoding="utf-8"))
        print(f"NESTED_MANAGER {facts}")
        # The gate's temp was inside the long parent and its manager listened elsewhere: one healthy there disproves it.
        assert Path(facts["tmpdir"]).is_relative_to(ambient.resolve()), facts
        assert not Path(facts["address"]).resolve().is_relative_to(ambient.resolve()), facts


@POSIX
@requires_node
def test_feedback_runner_real_python_child_inherits_the_selected_node(tmp_path, monkeypatch, run_tests):
    selected = _preferred_node_over_broken_path(tmp_path, monkeypatch)
    original_path = os.environ["PATH"]
    child = ("import os,pathlib,subprocess; "
             "p=subprocess.run(['node','-p','process.execPath'],capture_output=True,text=True,timeout=10); "
             "assert p.returncode==0,(p.returncode,p.stderr); "
             f"assert pathlib.Path(p.stdout.strip()).resolve()==pathlib.Path({selected!r}).resolve(); "
             f"explicit=dict(os.environ,PATH={original_path!r}); "
             "assert subprocess.run(['node','--version'],env=explicit,timeout=10).returncode==91")
    assert run_tests._run("nested node witness", [sys.executable, "-c", child], tmp_path) == 0
    assert os.environ["PATH"] == original_path, "the runner must not mutate its caller's environment"


@SHORT_TEMP
@pytest.mark.parametrize("explicit", [False, True], ids=["default-short", "explicit-parent"])
def test_safe_launcher_default_is_short_and_explicit_parent_still_wins(tmp_path, explicit):
    ambient = tmp_path / ("long-parent-" * 12)
    ambient.mkdir()
    env = {**os.environ, "TMPDIR": str(ambient), "TEMP": str(ambient), "TMP": str(ambient)}
    probe = "import json,os,pathlib; print(json.dumps({'root':os.environ['OUROBOROS_TEST_TEMP_ROOT']}))"
    if not explicit:
        probe = ("import multiprocessing; "
                 "manager=multiprocessing.Manager(); q=manager.Queue(); q.put('owned'); "
                 "assert q.get(timeout=5)=='owned'; manager.shutdown(); " + probe)
    argv = [sys.executable, "-I", "-S", str(REPO / "scripts/safe_test.py")]
    if explicit:
        argv += ["--temp-parent", str(ambient)]
    result = subprocess.run([*argv, "--", sys.executable, "-c", probe], cwd=REPO, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    root = Path(json.loads(result.stdout.splitlines()[-1])["root"])
    assert root.parent == (ambient.resolve() if explicit else Path("/tmp").resolve())
    assert not root.exists() and f"SAFE_TEST_RELEASED {root}" in result.stderr
