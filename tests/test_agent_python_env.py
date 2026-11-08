"""Regression tests for the agent interpreter handle.

The bundled app must run pytest through the same interpreter that launched
Ouroboros. Plain `pytest` or `python -m pytest` can resolve to the wrong
runtime in packaged builds.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest


def test_server_entry_names_its_own_interpreter_over_an_inherited_one(tmp_path):
    """An older launcher or a moved app can hand down a dead interpreter; the body's handle names this one."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    env = {**os.environ, "OUROBOROS_AGENT_PYTHON": str(tmp_path / "Old.app" / "python3"),
           "OUROBOROS_REPO_DIR": str(repo_root), "PYTHONPATH": str(repo_root)}
    done = subprocess.run([sys.executable, "-B", "-c", "import os, server; print(os.environ['OUROBOROS_AGENT_PYTHON'])"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == sys.executable, "the old entry kept an inherited value"


def test_preflight_test_runner_uses_sys_executable():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    source = (repo_root / "ouroboros" / "preflight_runner.py").read_text(encoding="utf-8")
    review_helpers = (repo_root / "ouroboros" / "tools" / "review_helpers.py").read_text(encoding="utf-8")
    assert '["pytest", "tests/"' not in source
    assert '"-m", "pytest"' in source
    assert "sys.executable" in source
    assert "run_hermetic_pytest" in review_helpers


def test_git_pre_push_tests_uses_sys_executable():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    source = (repo_root / "ouroboros" / "preflight_runner.py").read_text(encoding="utf-8")
    git_tools = (repo_root / "ouroboros" / "tools" / "git.py").read_text(encoding="utf-8")
    assert '["pytest", "tests/"' not in source
    assert '"-m", "pytest"' in source
    assert "sys.executable" in source
    assert "OUROBOROS_AGENT_PYTHON" in source
    assert "run_hermetic_pytest" in git_tools


def test_shell_no_longer_launches_pytest_at_all():
    """The shell validation runner retired WITH its only consumer (D10).

    `_run_validation` was `claude_code_edit`'s own `validate=True`
    implementation — one caller since the initial bundle — so it left with the
    tool. The interpreter-handle invariant this pin protected (a packaged app
    must run pytest through the interpreter that launched it, never bare
    `python`) keeps its two FIRST-CLASS homes, each pinned by its own test
    above: the preflight runner and the git pre-push tests. This pin now
    asserts the new truth at the old location: shell.py launches pytest in NO
    form, so an ad-hoc runner with the wrong interpreter cannot sneak back in
    beside process tools — a new pytest launch belongs in
    preflight_runner/run_hermetic_pytest, where the sibling pins bind it.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    source = (repo_root / "ouroboros" / "tools" / "shell.py").read_text(encoding="utf-8")
    assert '["python", "-m", "pytest"' not in source
    assert '"-m", "pytest"' not in source
    assert '"pytest"' not in source


def test_sys_executable_minus_m_pytest_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"sys.executable -m pytest --version exited {result.returncode}; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "pytest" in (result.stdout + result.stderr).lower()


def test_agent_python_env_var_points_to_usable_python():
    agent_python = os.environ.get("OUROBOROS_AGENT_PYTHON")
    if not agent_python:
        pytest.skip("OUROBOROS_AGENT_PYTHON is not set in this unit-test run")
    result = subprocess.run(
        [agent_python, "-c", "print('ok')"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "ok" in result.stdout


def test_pyproject_declares_pytest_as_runtime_dependency():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"pytest>=7.0"' in pyproject


def test_pyproject_declares_pip_as_runtime_dependency():
    """uv environments must preserve the runtime's ``python -m pip`` contract."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"pip"' in pyproject
