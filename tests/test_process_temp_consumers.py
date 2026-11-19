"""A long-named case's real processes get a short temp for their sockets; the case keeps its own roots (#2517)."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from ouroboros.test_environment import SCRATCH_MARKER, isolated_environment, owned_process_temp

pytest_plugins = ("tests.test_ui_smoke_playwright",)
REPO = Path(__file__).resolve().parents[1]
POSIX = pytest.mark.skipif(os.name == "nt", reason="a Windows manager listens on a named pipe")
TEMP_KEYS = {"TMPDIR", "TEMP", "TMP", "MAC_CHROMIUM_TMPDIR", "PYTEST_DEBUG_TEMPROOT"}
CASE_ROOTS = ("HOME", "OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH")
AF_UNIX_LIMIT = 104 if sys.platform == "darwin" else 108  # sun_path, terminator included
SOCKET_BELOW_TEMP = len("/pymp-12345678/listener-12345678")
_EXCHANGE = """
import json, multiprocessing, sys
with multiprocessing.Manager() as manager:
    queue = manager.Queue()
    queue.put("owned")
    print(json.dumps({"exchanged": queue.get(timeout=30), "address": manager.address, "python": sys.version}))
"""


def _exchange(env, cwd=REPO) -> dict:
    """One real Manager queue exchange in exactly this environment, or what refused it."""
    result = subprocess.run([sys.executable, "-c", _EXCHANGE], cwd=cwd, env=env,
                            text=True, capture_output=True, timeout=120)
    return json.loads(result.stdout.splitlines()[-1]) if result.returncode == 0 else {"refused": result.stderr}


def test_a_process_temp_moves_only_temp_keys_and_a_projection_allocates_nothing(tmp_path):
    case, temp = tmp_path.resolve() / "case", tmp_path.resolve() / "session" / "leaf"
    plain = isolated_environment(case, REPO, source={}, create=False)
    # The gate fingerprints this projection (`_preflight_env(create=False)`): its default never moves.
    assert plain["TMPDIR"] == str(case / "tmp") and plain["GIT_CEILING_DIRECTORIES"] == str(case)
    moved = isolated_environment(case, REPO, source={}, create=False, process_temp=temp)
    assert moved == isolated_environment(case, REPO, source={}, create=False, process_temp=temp)
    assert not case.exists() and not temp.parent.exists()
    assert {key for key in plain if moved[key] != plain[key]} == {*TEMP_KEYS, "GIT_CEILING_DIRECTORIES"}
    assert {moved[key] for key in TEMP_KEYS} == {str(temp)}
    assert isolated_environment(case, REPO, source={}, process_temp=temp) == moved
    assert temp.is_dir() and (case / "home").is_dir() and not (case / "tmp").exists()


def test_git_started_in_a_process_temp_stops_there(tmp_path):
    ancestor = tmp_path / "ancestor"
    subprocess.run(["git", "init", str(ancestor)], check=True, capture_output=True)
    temp = ancestor / "session" / "leaf"
    env = isolated_environment(tmp_path / "case", REPO, process_temp=temp)
    found = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=temp, env=env, capture_output=True)
    assert found.returncode != 0, found.stdout


def test_only_a_run_owner_gives_a_declared_process_temp_and_each_call_its_own(tmp_path, monkeypatch):
    owner, ambient = tmp_path / "owner", tmp_path / "ambient"
    owner.mkdir()
    ambient.mkdir()
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(owner))
    monkeypatch.setattr(tempfile, "tempdir", str(ambient))  # An inherited or patched temp is not the owner's.
    first, second = owned_process_temp(), owned_process_temp()
    assert first != second and first.parent == second.parent == owner and not any(ambient.iterdir())
    # Declared, so the owner releases it with the run instead of keeping the run as unknown data.
    assert {(leaf / SCRATCH_MARKER).read_text(encoding="utf-8") for leaf in (first, second)} == {"process-temp\n"}
    monkeypatch.delenv("OUROBOROS_TEST_TEMP_ROOT")
    assert owned_process_temp() is None  # Bare pytest: its ambient temp is nobody's to vouch for.


@POSIX
def test_a_process_temp_gives_a_long_named_case_the_manager_its_own_root_cannot(tmp_path):
    """Runs directly under a launcher or the hermetic gate; the launcher case below brings bare runs there."""
    temp = owned_process_temp()
    if temp is None:
        pytest.skip("bare pytest keeps its layout")
    env = isolated_environment(tmp_path, REPO, process_temp=temp)
    assert all(Path(env[key]).is_relative_to(tmp_path.resolve()) for key in CASE_ROOTS)
    facts = _exchange(env)
    assert facts.get("exchanged") == "owned", facts
    # Inside the owned temp: CPython 3.14 on macOS silently falls back to the shared /tmp at 64 bytes of TMPDIR.
    assert Path(facts["address"]).is_relative_to(temp), facts
    print(f"PROCESS_TEMP_SOCKET {len(os.fsencode(facts['address']))} bytes: {facts['address']} ({facts['python']})")
    case_temp = tmp_path.resolve() / "tmp"
    if len(os.fsencode(case_temp)) + SOCKET_BELOW_TEMP >= AF_UNIX_LIMIT:
        old = _exchange(isolated_environment(tmp_path, REPO))
        refused = "AF_UNIX path too long" in old.get("refused", "")
        assert refused or not Path(old["address"]).is_relative_to(case_temp), old


@POSIX
@pytest.mark.parametrize("flags", [(), ("-n", "2")], ids=["serial", "xdist"])
def test_launcher_sessions_give_a_long_named_case_its_process_temp(tmp_path, flags):
    node = "tests/test_process_temp_consumers.py::test_a_process_temp_gives_a_long_named_case_the_manager_its_own_root_cannot"
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(REPO / "scripts" / "safe_test.py"), "--evidence-dir", str(tmp_path / "evidence"),
         "--timeout", "280", "--", sys.executable, "-m", "pytest", node, "-o", "addopts=", "-rP", *flags],
        cwd=REPO, env={**os.environ, "OURO_TEST_WORKER_BUDGET": "2"}, text=True, capture_output=True, timeout=300)
    assert result.returncode == 0 and "1 passed" in result.stdout, result.stdout + result.stderr


@POSIX
@pytest.mark.ui_browser
def test_a_long_named_cases_real_server_listens_in_its_own_process_temp(direct_server_with_data, tmp_path):
    env, owner = direct_server_with_data["env"], os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
    temp, case = Path(env["TMPDIR"]), tmp_path.resolve()
    assert all(Path(env[key]).is_relative_to(case) for key in CASE_ROOTS)
    # The owner's declared leaf; bare pytest keeps its layout.
    assert temp.parent == Path(owner).resolve() if owner else temp == case / "tmp", (temp, case)
    assert [path for path in temp.glob("pymp-*/*") if stat.S_ISSOCK(path.lstat().st_mode)], sorted(temp.rglob("*"))
    facts = _exchange(env, cwd=direct_server_with_data["repo_dir"])
    assert facts.get("exchanged") == "owned" and Path(facts["address"]).is_relative_to(temp), facts
