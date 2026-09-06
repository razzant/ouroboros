"""The throwaway history repositories the git_ops recovery suites work against.

Split out of ``tests/test_git_ops_recovery.py`` when that module was divided by theme; the
builders are verbatim, so every sibling suite starts from the same commits, remotes and
working state it was written against.
"""

from __future__ import annotations

import subprocess




def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

def _history_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@ouroboros")
    (repo / "value.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "value.txt")
    _git(repo, "commit", "-qm", "one")
    first = _git(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "-M", "ouroboros")
    (repo / "value.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "two")
    second = _git(repo, "rev-parse", "HEAD")
    return repo, first, second
