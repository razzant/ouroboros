from __future__ import annotations

import pathlib
import subprocess


def _git(repo: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Binding Test")
    _git(repo, "config", "user.email", "binding@example.test")
    (repo / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    (repo / "app.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    return repo


def test_existing_expected_tag_is_a_pre_review_collision(tmp_path):
    from ouroboros.tools.git import (
        _fingerprint_staged_diff,
        _review_binding_precondition_error,
    )

    repo = _repo(tmp_path)
    old_target = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v1.1.0", old_target)
    (repo / "VERSION").write_text("1.1.0\n", encoding="utf-8")
    _git(repo, "add", "VERSION")

    fingerprint = _fingerprint_staged_diff(repo)
    message = _review_binding_precondition_error(fingerprint)

    assert fingerprint["binding"]["existing_tag_target"] == old_target
    assert "already targets" in message
    assert "immutable" in message


def test_commit_and_release_tag_must_match_reviewed_binding(tmp_path):
    from ouroboros.tools.git import (
        _auto_tag_on_version_bump,
        _fingerprint_staged_diff,
        _verify_reviewed_commit_binding,
    )

    repo = _repo(tmp_path)
    (repo / "VERSION").write_text("1.1.0\n", encoding="utf-8")
    (repo / "app.txt").write_text("release\n", encoding="utf-8")
    _git(repo, "add", "-A")
    fingerprint = _fingerprint_staged_diff(repo)
    _git(repo, "commit", "-m", "release")
    commit_sha = _git(repo, "rev-parse", "HEAD")

    ok, detail = _verify_reviewed_commit_binding(
        repo, commit_sha, fingerprint, verify_expected_tag=False
    )
    assert ok is True, detail
    ok, detail = _verify_reviewed_commit_binding(
        repo, commit_sha, fingerprint, verify_expected_tag=True
    )
    assert ok is False
    assert "expected tag" in detail

    tag_info = _auto_tag_on_version_bump(
        repo,
        "release",
        expected_commit_sha=commit_sha,
        expected_tag="v1.1.0",
    )
    assert "tagged: v1.1.0" in tag_info
    ok, detail = _verify_reviewed_commit_binding(
        repo, commit_sha, fingerprint, verify_expected_tag=True
    )
    assert ok is True, detail


def test_auto_tag_refuses_to_accept_existing_tag_on_other_commit(tmp_path):
    from ouroboros.tools.git import _auto_tag_on_version_bump

    repo = _repo(tmp_path)
    old_target = _git(repo, "rev-parse", "HEAD")
    (repo / "VERSION").write_text("1.1.0\n", encoding="utf-8")
    _git(repo, "add", "VERSION")
    _git(repo, "commit", "-m", "release")
    commit_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v1.1.0", old_target)

    result = _auto_tag_on_version_bump(
        repo,
        "release",
        expected_commit_sha=commit_sha,
        expected_tag="v1.1.0",
    )

    assert "tag target mismatch" in result
    assert _git(repo, "rev-parse", "v1.1.0^{commit}") == old_target
