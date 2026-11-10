"""Commit preparation selects the commit branch without losing work: dirty package files when the
local branch is missing, and a staged merge's MERGE_HEAD, MERGE_MODE and MERGE_MSG."""

import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.tools import git as git_tools



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


# --- a staged merge: Git's checkout drops MERGE_HEAD, MERGE_MODE and MERGE_MSG on success ---------------------

_MERGE_STATE = ("MERGE_HEAD", "MERGE_MODE", "MERGE_MSG")
_FIXTURE = ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid")


def _merge_state(repo):
    return {name: (repo / ".git" / name).read_bytes() if (repo / ".git" / name).exists() else None
            for name in _MERGE_STATE}


def _state(repo):
    """Branch or detached HEAD, refs, exact index and file bytes, and the merge state."""
    return {"head": _git(repo, "rev-parse", "--symbolic-full-name", "HEAD"), "sha": _git(repo, "rev-parse", "HEAD"),
            "refs": _git(repo, "show-ref"), "index": (repo / ".git" / "index").read_bytes(),
            "files": {str(path.relative_to(repo)): path.read_bytes()
                      for path in repo.rglob("*") if path.is_file() and ".git" not in path.parts},
            "merge": _merge_state(repo)}


def _staged_merge(root, *, on="ouroboros", detached=False, conflict=False):
    """A contributor's commit merged ``--no-ff --no-commit`` onto ``on`` (or its detached HEAD), as
    ``stage_pr_merge`` leaves it; ``conflict`` leaves base.txt unmerged instead."""
    root.mkdir(parents=True)
    _git(root, "init", "-b", on)
    (root / "base.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", "base.txt")
    _git(root, *_FIXTURE, "commit", "-m", "fixture base")
    _git(root, "checkout", "-b", "contribution")
    (root / ("base.txt" if conflict else "feature.txt")).write_text("contributed\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=Contributor", "-c", "user.email=contrib@example.invalid", "commit", "-m", "contribution")
    _git(root, "checkout", *(("--detach", on) if detached else (on,)))
    if conflict:
        (root / "base.txt").write_text("ours\n", encoding="utf-8")
        _git(root, *_FIXTURE, "commit", "-am", "ours")
    merged = subprocess.run(["git", *_FIXTURE, "merge", "--no-ff", "--no-commit", "contribution"], cwd=root,
                            capture_output=True, text=True)
    assert merged.returncode == (1 if conflict else 0), merged.stderr
    assert _merge_state(root)["MERGE_HEAD"] == (_git(root, "rev-parse", "contribution") + "\n").encode()
    return root


def test_the_current_branch_keeps_a_staged_merge_byte_for_byte(tmp_path):
    """Already on the commit branch nothing is checked out. The negative control shows what the checkout
    preparation used to run does to the same merge on this Git."""
    control = _staged_merge(tmp_path / "control")
    _git(control, "checkout", "--no-guess", "ouroboros", "--")
    assert _merge_state(control) == dict.fromkeys(_MERGE_STATE)
    repo = _staged_merge(tmp_path / "prepared")
    before = _state(repo)
    assert all(before["merge"].values())

    result = git_tools._prepare_review_commit_worktree(SimpleNamespace(repo_dir=repo, branch_dev="ouroboros"), None)

    assert result == (False, "")
    assert _state(repo) == before


@pytest.mark.parametrize("away", ["other_branch", "detached"])
def test_no_switch_runs_while_a_merge_is_staged_and_one_runs_without_it(tmp_path, away):
    """Reaching the commit branch from elsewhere needs a checkout, which would drop the merge: refused, with
    branch, refs, index, files and merge state unchanged. Once that merge is aborted the same switch runs."""
    repo = _staged_merge(tmp_path / "repo", on="elsewhere", detached=away == "detached")
    _git(repo, "branch", "ouroboros", "elsewhere")
    before = _state(repo)
    ctx = SimpleNamespace(repo_dir=repo, branch_dev="ouroboros")

    detached, error = git_tools._prepare_review_commit_worktree(ctx, None)

    assert detached is False and error.startswith("⚠️ GIT_MERGE_IN_PROGRESS_BLOCKED: a merge is in progress on "), error
    assert "Nothing was switched, staged or reviewed" in error
    assert _state(repo) == before
    _git(repo, "merge", "--abort")
    assert git_tools._prepare_review_commit_worktree(ctx, None) == (away == "detached", "")
    assert _git(repo, "symbolic-ref", "HEAD") == "refs/heads/ouroboros"


def test_an_unmerged_index_on_the_commit_branch_is_refused_without_a_checkout(tmp_path):
    """Git's checkout used to refuse a conflicted index here; preparation now refuses it itself, naming the
    unmerged paths, so conflicts are never staged as resolved, and the merge state stays."""
    repo = _staged_merge(tmp_path / "repo", conflict=True)
    before = _state(repo)

    detached, error = git_tools._prepare_review_commit_worktree(SimpleNamespace(repo_dir=repo, branch_dev="ouroboros"),
                                                                None)

    assert detached is False and error.startswith("⚠️ GIT_ERROR (checkout): Repository has unmerged paths"), error
    assert error.endswith("\nbase.txt")
    assert _state(repo) == before


@pytest.mark.parametrize("away", ["other_branch", "detached"])
def test_merge_state_read_failure_never_permits_a_branch_switch(tmp_path, monkeypatch, away):
    repo = _staged_merge(tmp_path / "repo", on="elsewhere", detached=away == "detached")
    _git(repo, "branch", "ouroboros", "elsewhere")
    before = _state(repo)
    original = git_tools.run_cmd

    def failed_probe(argv, **kwargs):
        if argv[:2] == ["git", "rev-parse"] and "MERGE_HEAD" in argv:
            raise RuntimeError("synthetic merge-state read failure")
        return original(argv, **kwargs)

    monkeypatch.setattr(git_tools, "run_cmd", failed_probe)
    detached, error = git_tools._prepare_review_commit_worktree(
        SimpleNamespace(repo_dir=repo, branch_dev="ouroboros"), None)
    assert detached is False and "Could not verify merge state" in error
    assert _state(repo) == before
