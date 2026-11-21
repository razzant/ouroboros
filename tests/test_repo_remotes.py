from __future__ import annotations
import subprocess

from tests._workspace_git import git_repository

def _init_repo(path):
    git_repository(path, unborn=True)

class FakeGitHubClient:
    def __init__(self, *, repos=None, forks=None):
        self.repos = dict(repos or {})
        self.forks = list(forks or [])
        self.created = []
    def current_login(self): return "anton"
    def repo_info(self, slug): return self.repos.get(slug)
    def viewer_forks_of(self, upstream_slug): return list(self.forks)
    def create_fork(self, upstream_slug, *, name, prefer_private):
        self.created.append((upstream_slug, name, prefer_private))
        return {"full_name": f"anton/{name}", "fork": True, "parent": {"full_name": upstream_slug}}

def test_reuses_renamed_verified_fork(tmp_path):
    from ouroboros.repo_remotes import ensure_personal_origin_target
    repo = tmp_path / "repo"; _init_repo(repo)
    client = FakeGitHubClient(forks=[{"nameWithOwner": "anton/my-ouroboros-lab", "isFork": True, "parent": {"nameWithOwner": "razzant/ouroboros"}}])
    result = ensure_personal_origin_target(repo, "token", client=client)
    assert result.ok and result.repo_slug == "anton/my-ouroboros-lab" and client.created == []

def test_name_collision_creates_alternate_fork(tmp_path):
    from ouroboros.repo_remotes import ensure_personal_origin_target
    repo = tmp_path / "repo"; _init_repo(repo)
    result = ensure_personal_origin_target(repo, "token", client=FakeGitHubClient(repos={"anton/ouroboros": {"full_name": "anton/ouroboros", "fork": False}}))
    assert result.ok and result.repo_slug == "anton/ouroboros-razzant"
    assert result.warnings == ["name_collision:anton/ouroboros"]

def test_configured_official_repo_is_rejected(tmp_path):
    from ouroboros.repo_remotes import ensure_personal_origin_target
    repo = tmp_path / "repo"; _init_repo(repo)
    result = ensure_personal_origin_target(repo, "token", configured_repo="https://github.com/razzant/ouroboros.git", client=FakeGitHubClient())
    assert result.ok is False and result.action == "official_repo_not_personal"

def test_private_fork_failure_does_not_fallback_to_public(tmp_path):
    from ouroboros.repo_remotes import ensure_personal_origin_target
    repo = tmp_path / "repo"; _init_repo(repo)
    class RefusingPrivateFork(FakeGitHubClient):
        def create_fork(self, upstream_slug, *, name, prefer_private):
            assert prefer_private is True
            raise RuntimeError("private forks are not allowed")
    result = ensure_personal_origin_target(repo, "token", client=RefusingPrivateFork())
    assert result.ok is False and "private forks are not allowed" in result.message

def test_clone_default_official_origin_is_not_a_conflict(tmp_path):
    # A plain desktop clone of the official repo leaves origin=official. Setting a
    # personal GITHUB_REPO must NOT origin_conflict — the official URL belongs on
    # the managed remote, not origin (role-based layout: managed=official, origin=personal).
    from ouroboros.repo_remotes import ensure_personal_origin_target
    repo = tmp_path / "repo"; _init_repo(repo)
    subprocess.run(["git", "remote", "add", "origin", "https://github.com/razzant/ouroboros.git"], cwd=repo, check=True, capture_output=True)
    result = ensure_personal_origin_target(repo, "token", configured_repo="anton/my-fork", client=FakeGitHubClient())
    assert result.ok and result.repo_slug == "anton/my-fork" and result.action == "configured"

def test_real_personal_origin_still_conflicts(tmp_path):
    # A deliberate NON-official origin that differs from the configured target is
    # still a conflict, so a user's real personal origin is never silently repointed.
    from ouroboros.repo_remotes import ensure_personal_origin_target
    repo = tmp_path / "repo"; _init_repo(repo)
    subprocess.run(["git", "remote", "add", "origin", "https://github.com/anton/other-repo.git"], cwd=repo, check=True, capture_output=True)
    result = ensure_personal_origin_target(repo, "token", configured_repo="anton/my-fork", client=FakeGitHubClient())
    assert result.ok is False and result.action == "origin_conflict"

def test_configure_personal_remote_rejects_official_without_autofork(tmp_path, monkeypatch):
    # B1 regression: a configured GITHUB_REPO must be validated even when
    # auto_fork is False, so origin can never be pointed at the official repo.
    from supervisor import git_ops
    repo = tmp_path / "repo"; _init_repo(repo)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    ok, _msg, slug = git_ops.configure_personal_remote("razzant/ouroboros", "token", auto_fork=False)
    assert ok is False and slug == ""

def test_configure_personal_remote_empty_slug_without_autofork_does_not_fork(tmp_path, monkeypatch):
    # auto_fork=False with no configured slug must not silently create a fork.
    from supervisor import git_ops
    repo = tmp_path / "repo"; _init_repo(repo)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    ok, msg, slug = git_ops.configure_personal_remote("", "token", auto_fork=False)
    assert ok is False and slug == "" and "Missing repo slug" in msg
