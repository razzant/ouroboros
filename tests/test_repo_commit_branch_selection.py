"""Commit preparation preserves dirty package files when the local branch is missing."""

import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.tools import git as git_tools

pytestmark = pytest.mark.serial


def _git(repo, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _repository(tmp_path, remote_count):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "feature")
    package = repo / "ouroboros"
    package.mkdir()
    tracked = package / "module.py"
    tracked.write_text("base = 1\n", encoding="utf-8")
    _git(repo, "add", "ouroboros/module.py")
    _git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-m", "fixture base")
    for remote in ("origin", "managed")[:remote_count]:
        _git(repo, "remote", "add", remote, f"https://example.invalid/{remote}/repo.git")
        _git(repo, "update-ref", f"refs/remotes/{remote}/ouroboros", "HEAD")
    tracked.write_text("staged = 2\n", encoding="utf-8")
    _git(repo, "add", "ouroboros/module.py")
    tracked.write_text("unstaged = 3\n", encoding="utf-8")
    (package / "new.txt").write_text("new work\n", encoding="utf-8")
    (repo / "outside.txt").write_text("outside work\n", encoding="utf-8")
    return repo


def _snapshot(repo):
    # Exact index and file bytes distinguish staged work from unstaged work.
    return {
        "branch": _git(repo, "symbolic-ref", "HEAD"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "refs": _git(repo, "show-ref"),
        "index": (repo / ".git" / "index").read_bytes(),
        "files": {str(path.relative_to(repo)): path.read_bytes()
                  for path in repo.rglob("*") if path.is_file() and ".git" not in path.parts},
    }


@pytest.mark.parametrize("remote_count", [0, 1, 2])
def test_missing_local_branch_preserves_candidate_without_remote_guess(tmp_path, remote_count):
    repo = _repository(tmp_path, remote_count)
    before = _snapshot(repo)

    detached, error = git_tools._prepare_review_commit_worktree(
        SimpleNamespace(repo_dir=repo, branch_dev="ouroboros"), None,
    )

    assert detached is False
    assert "GIT_BRANCH_UNAVAILABLE" in error
    assert "ouroboros" in error
    assert _snapshot(repo) == before


@pytest.mark.parametrize("remote_count", [0, 1, 2])
def test_existing_local_branch_switch_preserves_staged_and_unstaged_work(tmp_path, remote_count):
    repo = _repository(tmp_path, remote_count)
    _git(repo, "branch", "ouroboros")
    before = _snapshot(repo)
    staged = _git(repo, "show", ":ouroboros/module.py")

    detached, error = git_tools._prepare_review_commit_worktree(
        SimpleNamespace(repo_dir=repo, branch_dev="ouroboros"), None,
    )

    assert (detached, error) == (False, "")
    after = _snapshot(repo)
    assert after["branch"] == "refs/heads/ouroboros"
    assert after["head"] == before["head"]
    assert after["refs"] == before["refs"]
    assert after["files"] == before["files"]
    assert _git(repo, "show", ":ouroboros/module.py") == staged


@pytest.mark.parametrize("verified", [True, False])
def test_managed_preparation_uses_transaction_verification_only(tmp_path, monkeypatch, verified):
    from supervisor import update_merge

    transaction = {"id": "fixture-managed-merge"}
    seen = []

    def verify(value):
        seen.append(value)
        return verified, "managed refusal"

    def no_checkout(*args, **kwargs):
        raise AssertionError("managed preparation must not select another branch")

    monkeypatch.setattr(update_merge, "managed_assisted_precommit_verify", verify)
    monkeypatch.setattr(git_tools, "run_cmd", no_checkout)
    result = git_tools._prepare_review_commit_worktree(
        SimpleNamespace(repo_dir=tmp_path, branch_dev="ouroboros"), transaction,
    )
    assert seen == [transaction]
    assert result == (False, "" if verified else "managed refusal")
