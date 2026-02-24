"""Fast tests for pre-push gate.

These checks must be extremely quick (<5s total).
"""

import pathlib
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


# ── Critical invariants ────────────────────────────────────────────────

def test_version_file_exists():
    """VERSION file exists and contains valid semver."""
    version = (REPO / "VERSION").read_text().strip()
    parts = version.split(".")
    assert len(parts) == 3, f"VERSION '{version}' is not semver"
    for p in parts:
        assert p.isdigit(), f"VERSION part '{p}' is not numeric"


def test_version_in_readme():
    """VERSION matches what README claims."""
    version = (REPO / "VERSION").read_text().strip()
    readme = (REPO / "README.md").read_text()
    assert version in readme, f"VERSION {version} not found in README.md"


def test_bible_exists_and_has_principles():
    """BIBLE.md exists and contains all 9 principles (0-8)."""
    bible = (REPO / "BIBLE.md").read_text()
    for i in range(9):
        assert f"Principle {i}" in bible, f"Principle {i} missing from BIBLE.md"


def test_core_modules_import():
    """All core modules import without error."""
    core_modules = [
        "ouroboros.agent",
        "ouroboros.context",
        "ouroboros.loop",
        "ouroboros.llm",
        "ouroboros.memory",
        "ouroboros.utils",
        "ouroboros.consciousness",
        "supervisor.state",
        "supervisor.events",
    ]
    for module in core_modules:
        __import__(module)


def test_no_env_dumping_simple():
    """Quick scan: no direct print(os.environ) in core files."""
    import re

    dangerous = re.compile(r'print\s*\(\s*os\.environ\b')
    violations = []
    for path in (REPO / "ouroboros").rglob("*.py"):
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if dangerous.search(line):
                violations.append(f"{path.name}:{i}: {line.strip()[:60]}")
    assert len(violations) == 0, f"Env dumping found:\n" + "\n".join(violations)


def test_run_shell_uses_list():
    """Quick scan: run_shell calls use list, not string."""
    import re

    violations = []
    for path in (REPO / "ouroboros").rglob("*.py"):
        text = path.read_text()
        for m in re.finditer(r'run_shell\s*\(\s*["\']', text):
            snippet = text[max(0, m.start() - 40):m.end() + 40].strip()
            if not snippet.startswith("#"):
                violations.append(f"{path.name}: {snippet}")
                break  # one per file is enough
    assert len(violations) == 0, f"run_shell with string cmd:\n" + "\n".join(violations)
