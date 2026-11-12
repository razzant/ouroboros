"""Shared fixtures for review-change and diff-pulse consumer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ouroboros.tools import git as git_mod
from tests import _contributor_packet_shared as shared
from tests.review_pool_rosters import set_review_pool

GOAL = "Make the installed body's helper return the proposal's constant."
SCOPE = "ouroboros/helper.py only; the checklist and tests stay as they are."


def _brief_text(brief: dict) -> str:
    """All delivered message bodies and, for a retrieving seat, its session task."""
    parts = []
    for message in brief["messages"]:
        content = message.get("content") or ""
        if isinstance(content, list):
            parts.extend(str(block.get("text") or json.dumps(block, sort_keys=True)) for block in content)
        else:
            parts.append(str(content))
    return "\n".join(parts) + "\n" + brief["session_task"]


@pytest.fixture
def staged_body(tmp_path, monkeypatch):
    fixture = shared.init_installed_body(tmp_path)
    repo = Path(fixture["repo"])
    # A real install carries a .gitignore; without one the gate's `git add -A`
    # door would stage the one it writes and review a different tree.
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    shared.git(repo, "add", ".gitignore")
    shared.git(repo, "commit", "-q", "-m", "ignore caches")
    shared.git(repo, "cherry-pick", "--no-commit", fixture["head_sha"])
    fixture["staged_tree_sha"] = shared.git(repo, "write-tree")
    assert fixture["staged_tree_sha"] != fixture["head_tree_sha"]
    set_review_pool(monkeypatch, shared.golden_pool())
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")  # the proposal touches a protected surface
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", shared.passing_test_runner)
    return fixture
