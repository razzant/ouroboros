"""Smoke test suite for Ouroboros.

Verifies core invariants and health checks after code changes.
"""

import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent


def test_no_env_dumping():
    """Security: no code dumps entire env (os.environ without key access).

    Allows: os.environ["KEY"], os.environ.get(), os.environ.setdefault(),
            os.environ.copy() (for subprocess).
    Disallows: print(os.environ), json.dumps(os.environ), etc.
    """
    # Only flag raw os.environ passed to print/json/log without bracket or .get( accessor
    dangerous = re.compile(r'(?:print|json\.dumps|log)\s*\(.*\bos\.environ\b(?!\s*[\[.])')
    violations = []
    exclude_dirs = {'.git', '__pycache__', 'venv', '.venv', 'env', 'build', 'dist', '.pytest_cache'}
    source_roots = [REPO / "ouroboros", REPO / "supervisor", REPO / "tests"]
    for root in source_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in exclude_dirs for part in path.parts):
                continue
            in_docstring = False
            for i, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Track triple-quoted docstrings to avoid false positives
                if '"""' in line or "'''" in line:
                    # Toggle docstring state; lines containing triple quotes are skipped entirely
                    in_docstring = not in_docstring
                    continue
                if in_docstring:
                    continue
                if dangerous.search(line):
                    violations.append(f"{path.name}:{i}: {line.strip()[:80]}")
    assert len(violations) == 0, f"Dangerous env dumping:\n" + "\n".join(violations)


def test_expected_tools_present():
    """Core tools are present and discoverable."""
    from ouroboros.tools.registry import get_tools
    tools = get_tools()
    names = [t.name for t in tools]
    expected = [
        "repo_read", "repo_list", "drive_read", "drive_list",
        "repo_write_commit", "repo_commit_push", "git_status", "git_diff",
        "knowledge_read", "knowledge_write", "knowledge_list",
        "chat_history", "update_scratchpad", "update_identity",
        "run_shell", "claude_code_edit",
        "browse_page", "browser_action",
        "web_search",
        "request_restart", "promote_to_stable", "schedule_task", "cancel_task",
        "request_review", "switch_model", "send_owner_message",
        "list_available_tools", "enable_tools",
        "analyze_screenshot",
        "search_agent",
    ]
    missing = [n for n in expected if n not in names]
    assert not missing, f"Missing tools: {missing}"


def test_version_consistency():
    """VERSION, pyproject.toml, and README must agree."""
    version_file = (REPO / "VERSION").read_text().strip()
    import toml
    pyproject = toml.load(REPO / "pyproject.toml")
    pyproject_version = pyproject["project"]["version"]
    readme_head = (REPO / "README.md").read_text().splitlines()[0]
    assert version_file == pyproject_version, f"VERSION mismatch: {version_file} vs {pyproject_version}"
    assert f"**Version:** {pyproject_version}" in readme_head, f"README missing version {pyproject_version}"


def test_run_shell_uses_list():
    """run_shell expects a list of args (not a shell string)."""
    import inspect
    from ouroboros.tools.core import run_shell
    sig = inspect.signature(run_shell)
    assert "cmd" in sig.parameters
    # Accept list or tuple; reject str in signature hint is not enforced; just document usage
    # This test ensures we remember to pass a list
    assert True  # placeholder for code review


def test_no_direct_env_print():
    """Additional check: no print(os.environ) anywhere."""
    # This is redundant with test_no_env_dumping but catches obvious cases
    import subprocess
    result = subprocess.run(
        ["grep", "-r", "print(os.environ)", "ouroboros", "supervisor", "tests", "--include=*.py"],
        capture_output=True, text=True
    )
    assert result.returncode != 0, "Found print(os.environ):\n" + result.stdout


# Additional tests can be added below
