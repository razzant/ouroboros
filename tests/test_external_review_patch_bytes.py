"""The runtime's isolated checkout replays exact Git patch bytes across newline
conversion policies, and the operator lane surfaces a drifted reviewed tree."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.tools import git as review_git
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_subject import ReviewSubjectSpec, isolated_checkout
from scripts import run_external_review as runner


def _git_bytes(cwd: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True).stdout


def _staged_repo(tmp_path: Path, *, eol: bytes, autocrlf: str) -> tuple[Path, bytes, bytes]:
    """A repo with ``change.py`` staged to ``value = 2<eol>``; ``(repo, proposed bytes, index tree)``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (("init",), ("config", "user.name", "Test"),
                 ("config", "user.email", "test@example.invalid"),
                 ("config", "core.autocrlf", autocrlf)):
        _git_bytes(repo, *args)
    # Explicit raw text permits either Git blob spelling under either user
    # conversion preference. The ordinary no-attributes fixture is covered by
    # test_external_review_pending_checkout.
    (repo / ".gitattributes").write_bytes(b"change.py -text\n")
    (repo / "VERSION").write_bytes(b"1.0.0\n")
    (repo / "change.py").write_bytes(b"value = 1" + eol)
    _git_bytes(repo, "add", ".")
    _git_bytes(repo, "commit", "-m", "base")
    proposed = b"value = 2" + eol
    (repo / "change.py").write_bytes(proposed)
    _git_bytes(repo, "add", "change.py")
    return repo, proposed, _git_bytes(repo, "write-tree").strip()


@pytest.mark.parametrize("eol", [b"\n", b"\r\n"], ids=["lf-blob", "crlf-blob"])
@pytest.mark.parametrize("autocrlf", ["false", "true"])
@pytest.mark.parametrize("windows_text", [False, True], ids=["native-stdio", "windows-text-stdio"])
def test_isolated_checkout_replays_the_staged_bytes(tmp_path, monkeypatch, eol, autocrlf, windows_text):
    repo, proposed, expected_tree = _staged_repo(tmp_path, eol=eol, autocrlf=autocrlf)
    original_run = subprocess.run

    def run(args, *positional, **kwargs):
        # POSIX normally masks Windows' text-pipe translation. Exercise that
        # exact transformation only if a regression reintroduces text stdin;
        # native Windows already performs it and must not translate twice.
        if (windows_text and os.name != "nt" and list(args)[:2] == ["git", "apply"]
                and kwargs.get("text") and isinstance(kwargs.get("input"), str)):
            kwargs["input"] = kwargs["input"].replace("\n", "\r\n")
        return original_run(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data")
    spec = ReviewSubjectSpec(root_kind="system_repo", root=str(repo), kind="index", surface="commit_gate")

    with isolated_checkout(ctx, spec) as frozen:
        checkout = Path(frozen.checkout)
        assert checkout != repo and checkout.parent.parent == tmp_path / "data" / "state" / "review_checkouts"
        assert frozen.tree_sha.encode() == expected_tree
        assert _git_bytes(checkout, "write-tree").strip() == expected_tree
        assert (checkout / "change.py").read_bytes() == proposed
        # The primary worktree keeps its index; the checkout is a worktree of the same repo.
        assert _git_bytes(repo, "write-tree").strip() == expected_tree
        assert _git_bytes(repo, "worktree", "list", "--porcelain").count(b"worktree ") == 2

    assert not checkout.exists() and not checkout.parent.exists()
    assert _git_bytes(repo, "worktree", "list", "--porcelain").count(b"worktree ") == 1


@pytest.mark.parametrize("eol", [b"\n", b"\r\n"], ids=["lf-blob", "crlf-blob"])
@pytest.mark.parametrize("drift", [False, True], ids=["same-tree", "drift-bytes"])
def test_operator_lane_reviews_the_replayed_bytes_and_surfaces_drift(tmp_path, monkeypatch, eol, drift):
    repo, proposed, expected_tree = _staged_repo(tmp_path, eol=eol, autocrlf="false")
    output = tmp_path / "output"
    observed = {}

    def cycle(ctx, _message, **kwargs):
        assert kwargs["preflight_reviewer"] == ""
        assert _git_bytes(ctx.repo_dir, "write-tree").strip() == expected_tree
        assert (ctx.repo_dir / "change.py").read_bytes() == proposed
        observed["cycle"] = ctx.repo_dir
        if drift:
            (ctx.repo_dir / "change.py").write_bytes(b"value = 3" + eol)
            observed["drift"] = _git_bytes(ctx.repo_dir, "diff", "HEAD", "--binary")
        return {"status": "blocked", "block_reason": "preflight", "message": "fixture"}

    monkeypatch.setattr(runner, "REPO", repo)
    monkeypatch.setattr(runner, "_parse_args", lambda: SimpleNamespace(
        contributor=False, commit_message="candidate", goal="", scope="", preflight_reviewer="",
        output=str(output), drive_root=str(tmp_path / "data"), no_isolated_checkout=False))
    monkeypatch.setattr(runner, "_prepare_review_configuration", lambda _args: (None, {}))
    monkeypatch.setattr(review_git, "_run_non_committing_review_cycle", cycle)

    assert runner.main() == 3
    assert observed["cycle"] != repo, "Patch replay must reach the existing review cycle in the checkout"
    artifact = output / "reviewed-tree-drift.diff"
    assert artifact.exists() is drift
    if drift:
        assert artifact.read_bytes() == observed["drift"]
    assert not (tmp_path / "data" / "state" / "review_checkouts").exists() or not any(
        (tmp_path / "data" / "state" / "review_checkouts").iterdir())
    assert _git_bytes(repo, "worktree", "list", "--porcelain").count(b"worktree ") == 1
