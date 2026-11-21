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
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")  # the proposal touches a protected surface
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", shared.passing_test_runner)
    return fixture


# What the landing continuation leaves out: a paid rebuttal and a new panel it cannot carry, and the
# paths, so the reviewed composition (``review_collection.author_scope``) is what lands.
CONTINUATION_DROPS = frozenset({"review_rebuttal", "preflight_reviewer", "reviewers", "reason", "paths"})
LANDING_DECISION = {"disposition": "accepted", "rationale": "test"}


def returned_reference(result) -> dict:
    """The ``review_reference`` a commit call returned with its outcome before Git."""
    text = str(result)
    assert text.startswith("Review outcome returned before commit"), text
    # The JSON follows the prose; a collected record may append its id line after it.
    return json.JSONDecoder().raw_decode(text, text.index("\n{") + 1)[0]["review_reference"]


def land_after_review(ctx, commit_message, *, commit=None, **commit_kwargs):
    """Outside Cyber Pro every review outcome, a PASS included, returns before Git: the first call
    must hand it back, and the author's explicit continuation on its ``review_reference`` lands it."""
    commit = commit or git_mod._commit_reviewed
    reference = returned_reference(commit(ctx, commit_message, **commit_kwargs))
    carried = {key: value for key, value in commit_kwargs.items() if key not in CONTINUATION_DROPS}
    return commit(ctx, commit_message, review_reference=reference, author_disposition=dict(LANDING_DECISION), **carried)
