"""Shared module accessors, ToolContext builders and fixtures for the git+review suites.

Split out of ``tests/test_git_review_pipeline.py`` when that module was
divided by theme. The definitions are verbatim so every sibling suite keeps
the exact seam it was written against; ``git_ctx``/``review_ctx`` are plain
(non-autouse) fixtures, so a suite gets them by importing them.
"""
import importlib
import os
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


import re as _re


def _critical_triad_items():
    """Parse critical triad checklist item ids from the frozen CHECKLISTS.md.

    Used to parametrize the NW-2 advisory-downgrade guardrail over EVERY
    critical item (not just ``code_quality``), so a per-item always-block
    hardcode against owner-chosen advisory enforcement (the 58a52c4 class)
    fails the suite. Falls back to a known critical pair if parsing fails so
    the guardrail never silently degrades to zero cases.
    """
    try:
        review = importlib.import_module("ouroboros.tools.review")
        section = review._load_checklist_section()
        items = []
        for line in section.splitlines():
            m = _re.match(r"^\s*\|\s*\d+\s*\|\s*([a-z0-9_]+)\s*\|.*\|\s*critical\s*\|\s*$", line)
            if m:
                items.append(m.group(1))
        # version_bump (item 8) is the incident's triad item; ensure it's present.
        if "version_bump" in items and len(items) >= 5:
            return items
    except Exception:
        pass
    return ["bible_compliance", "code_quality", "version_bump", "security_issues"]


def _get_git_module():
    return importlib.import_module("ouroboros.tools.git")


def _get_review_module():
    return importlib.import_module("ouroboros.tools.review")


def _get_registry_module():
    return importlib.import_module("ouroboros.tools.registry")


def _get_git_ops_module():
    return importlib.import_module("supervisor.git_ops")


def _make_ctx(tmp_path):
    """Create a minimal ToolContext with a temporary git repo."""
    from ouroboros.tools.registry import ToolContext
    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "logs").mkdir(parents=True)
    (drive / "locks").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), capture_output=True)
    (repo / "dummy.txt").write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "branch", "-M", "ouroboros"], cwd=str(repo), capture_output=True)
    return ToolContext(repo_dir=repo, drive_root=drive)
