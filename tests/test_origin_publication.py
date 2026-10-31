"""Personal-origin publication: real local/bare Git, no remote service or credentials."""
from __future__ import annotations

import os
import subprocess

import pytest

from supervisor import git_ops


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


@pytest.fixture
def origin(tmp_path, monkeypatch):
    repo, remote = tmp_path / "repo", tmp_path / "origin.git"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "ouroboros")
    git(repo, "config", "user.name", "Publication test")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "push.followTags", "true")
    git(repo, "commit", "--allow-empty", "-qm", "base")
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "tag", "-a", "v-old", "-m", "old release")
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    return repo, remote


def refs(repo, prefix=""):
    return git(repo, "for-each-ref", "--format=%(refname)", *([prefix] if prefix else [])).splitlines()


def test_default_push_never_backfills_reachable_tags_even_with_ambient_follow_tags(origin):
    repo, remote = origin
    ok, message = git_ops.push_to_remote()
    assert ok and "Pushed ouroboros" in message
    assert refs(remote) == ["refs/heads/ouroboros"]
    assert git(remote, "rev-parse", "ouroboros") == git(repo, "rev-parse", "HEAD")
    assert refs(repo, "refs/tags") == ["refs/tags/v-old"]  # nothing removed locally


def test_only_operation_created_tag_is_pushed_and_later_neutral_push_stays_quiet(origin):
    repo, remote = origin
    git(repo, "commit", "--allow-empty", "-qm", "release")
    commit = git(repo, "rev-parse", "HEAD")
    git(repo, "tag", "-a", "v-new", "-m", "operation release")
    git(repo, "tag", "-a", "unrelated", "-m", "not this operation")
    ok, message = git_ops.push_to_remote(created_tag="v-new", expected_commit_sha=commit)
    assert ok and "+ tag v-new" in message
    assert refs(remote, "refs/tags") == ["refs/tags/v-new"]
    assert git(remote, "rev-parse", "v-new") == git(repo, "rev-parse", "v-new")
    git(repo, "commit", "--allow-empty", "-qm", "neutral contribution")
    assert git_ops.push_to_remote()[0]
    assert refs(remote, "refs/tags") == ["refs/tags/v-new"]


@pytest.mark.parametrize("kind", ["absent", "lightweight", "wrong_commit", "off_branch", "no_evidence"])
def test_invalid_or_unadopted_tag_keeps_branch_success_but_reports_partial(origin, kind):
    repo, remote = origin
    commit = git(repo, "rev-parse", "HEAD")
    if kind == "lightweight":
        git(repo, "tag", "v-new")
    elif kind in {"wrong_commit", "no_evidence"}:
        git(repo, "tag", "-a", "v-new", "-m", "tag")
        if kind == "wrong_commit":
            git(repo, "commit", "--allow-empty", "-qm", "later")
            commit = git(repo, "rev-parse", "HEAD")
        else:
            commit = ""
    elif kind == "off_branch":
        git(repo, "checkout", "-qb", "candidate/test")
        git(repo, "commit", "--allow-empty", "-qm", "unadopted")
        commit = git(repo, "rev-parse", "HEAD")
        git(repo, "tag", "-a", "v-new", "-m", "candidate tag")
        git(repo, "checkout", "-q", "ouroboros")
    ok, message = git_ops.push_to_remote(created_tag="v-new", expected_commit_sha=commit)
    assert ok and "partial publication" in message and "no automatic retry" in message
    assert refs(remote) == ["refs/heads/ouroboros"]


def test_ref_movement_cannot_replace_the_validated_tag_object(origin, monkeypatch):
    repo, remote = origin
    commit = git(repo, "rev-parse", "HEAD")
    git(repo, "tag", "-a", "v-new", "-m", "validated")
    original = git(repo, "rev-parse", "v-new")
    network = git_ops._git_network_bounded

    def swap(command, **kwargs):
        if any(arg.endswith(":refs/tags/v-new") for arg in command):
            git(repo, "tag", "-fa", "v-new", "-m", "substituted")
        return network(command, **kwargs)

    monkeypatch.setattr(git_ops, "_git_network_bounded", swap)
    assert git_ops.push_to_remote(created_tag="v-new", expected_commit_sha=commit)[0]
    assert git(remote, "rev-parse", "v-new") == original
    assert git(repo, "rev-parse", "v-new") != original


@pytest.mark.skipif(os.name == "nt", reason="bare Git hook uses a POSIX shell; other real-ref cases are portable")
def test_tag_failure_is_visible_without_force_retry_or_later_backfill(origin):
    repo, remote = origin
    git(repo, "tag", "-a", "v-new", "-m", "new")
    hook = remote / "hooks" / "update"
    hook.write_text('#!/bin/sh\ncase "$1" in refs/tags/*) echo rejected >> "$(git rev-parse --git-dir)/tag-attempts"; exit 1;; esac\n',
                    encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    ok, message = git_ops.push_to_remote(created_tag="v-new", expected_commit_sha=git(repo, "rev-parse", "HEAD"))
    assert ok and "partial publication: tag v-new failed or unconfirmed" in message
    assert "no automatic retry" in message
    assert (remote / "tag-attempts").read_text().splitlines() == ["rejected"]
    assert git_ops.push_to_remote()[0]
    assert refs(remote) == ["refs/heads/ouroboros"]
    assert (remote / "tag-attempts").read_text().splitlines() == ["rejected"]


def test_explicit_push_tags_false_keeps_tag_local(origin):
    repo, remote = origin
    git(repo, "tag", "-a", "v-new", "-m", "new")
    assert git_ops.push_to_remote(push_tags=False, created_tag="v-new",
                                  expected_commit_sha=git(repo, "rev-parse", "HEAD"))[0]
    assert refs(remote) == ["refs/heads/ouroboros"]
