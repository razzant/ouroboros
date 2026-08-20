"""The rescue snapshot: the recoverable ref and the bytes it must not lose.

Split verbatim out of ``tests/test_git_ops_recovery.py`` by theme. This module owns the
recoverable ref the snapshot writes, the merge topology it captures from an unmerged index,
the non-UTF-8 bytes its diff preserves, the untracked-only case that raises no stash error,
and the hook that treats an unreadable status as dirty rather than clean.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

import supervisor.git_ops as git_ops


def test_create_rescue_snapshot_writes_recoverable_ref(monkeypatch, tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "f.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "v1"], cwd=repo, check=True, capture_output=True)
    # Tracked, uncommitted modification — the kind a rescue_and_reset would wipe.
    (repo / "f.txt").write_text("v2-uncommitted\n", encoding="utf-8")

    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    info = git_ops._create_rescue_snapshot(
        "ouroboros", "test",
        {"current_branch": "ouroboros", "dirty_lines": [" M f.txt"], "unpushed_lines": [], "warnings": []},
    )

    ref = info.get("rescue_ref")
    assert ref and ref.startswith("refs/rescue/")
    assert info.get("rescue_commit")
    # The ref is a real, recoverable git object.
    assert subprocess.run(["git", "rev-parse", "--verify", ref], cwd=repo, capture_output=True).returncode == 0
    # Simulate the wipe, then recover the uncommitted change from the ref.
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo, check=True, capture_output=True)
    assert (repo / "f.txt").read_text(encoding="utf-8") == "v1\n"
    subprocess.run(["git", "checkout", ref, "--", "f.txt"], cwd=repo, check=True, capture_output=True)
    assert (repo / "f.txt").read_text(encoding="utf-8") == "v2-uncommitted\n"

def _rescue_fixture_repo(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "f.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    branch = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    return repo, branch

def _conflicted_rescue_repo(tmp_path):
    """A fixture repo parked on a real conflicted merge (MERGE_HEAD + unmerged index)."""
    repo, branch = _rescue_fixture_repo(tmp_path)
    subprocess.run(["git", "checkout", "-q", "-b", "theirs"], cwd=repo, check=True,
                   capture_output=True)
    (repo / "f.txt").write_text("theirs\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "theirs"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "-q", branch], cwd=repo, check=True, capture_output=True)
    (repo / "f.txt").write_text("ours\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "ours"], cwd=repo, check=True, capture_output=True)
    assert subprocess.run(["git", "merge", "theirs"], cwd=repo,
                          capture_output=True).returncode != 0
    return repo, branch

def test_create_rescue_snapshot_captures_merge_topology_on_unmerged_index(monkeypatch, tmp_path):
    """On an in-progress conflicted merge the snapshot must keep the uncommitted
    resolution content, disclose the stash failure, and record the merge topology."""
    import pathlib

    repo, branch = _conflicted_rescue_repo(tmp_path)
    (repo / "f.txt").write_text("agent resolution\n", encoding="utf-8")  # uncommitted resolution

    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    info = git_ops._create_rescue_snapshot(
        branch, "merge-test",
        {"current_branch": branch, "dirty_lines": ["UU f.txt"], "unpushed_lines": [], "warnings": []},
    )

    rescue_dir = pathlib.Path(info["path"])
    assert "agent resolution" in (rescue_dir / "changes.diff").read_text(encoding="utf-8")
    # The stash failure ("needs merge") is disclosed instead of silently dropped.
    assert info.get("rescue_stash_error")
    assert "rescue_ref" not in info
    merge_head = subprocess.run(
        ["git", "rev-parse", "MERGE_HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert info.get("merge_head") == merge_head
    # Unique conflicted PATHS (one file), not its three stage-1/2/3 index rows.
    assert int(info.get("unmerged_count") or 0) == 1
    assert (rescue_dir / "unmerged.txt").read_text(encoding="utf-8").strip()
    assert (rescue_dir / "merge_msg.txt").exists()

def test_rescue_diff_uses_shared_binary_bounded_runner(monkeypatch, tmp_path):
    from ouroboros import update_channels

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(update_channels, "get_rescue_git_timeout_sec", lambda: 41)

    def fake_rescue_capture(cmd):
        if cmd == ["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"]:
            return 1, "", ""
        return 0, "", ""

    bounded = []

    def fake_bounded(cmd, *, timeout, cwd=None, env=None, text=True):
        bounded.append((cmd, timeout, cwd, text))
        return 0, b"raw diff\n", b""

    monkeypatch.setattr(git_ops, "rescue_git_capture", fake_rescue_capture)
    monkeypatch.setattr(git_ops, "_run_git_process_bounded", fake_bounded)

    info = git_ops._create_rescue_snapshot(
        "ouroboros", "bounded-diff",
        {
            "current_branch": "ouroboros", "dirty_lines": [],
            "unpushed_lines": [], "warnings": [],
        },
    )

    assert len(bounded) == 1
    assert bounded[0][1:] == (41, repo, False)
    assert (pathlib.Path(info["path"]) / "changes.diff").read_bytes() == b"raw diff\n"

def test_rescue_changes_diff_preserves_non_utf8_bytes(monkeypatch, tmp_path):
    """changes.diff must survive BYTES end-to-end: on an unmerged index it is the only
    carrier of resolutions, and a text-mode decode would corrupt latin-1 content into
    U+FFFD replacement characters."""
    import pathlib

    repo, branch = _conflicted_rescue_repo(tmp_path)
    # The agent's resolution carries a latin-1 byte (0xE9) — NOT valid UTF-8.
    (repo / "f.txt").write_bytes(b"agent r\xe9solution\n")

    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    info = git_ops._create_rescue_snapshot(
        branch, "bytes-test",
        {"current_branch": branch, "dirty_lines": ["UU f.txt"], "unpushed_lines": [], "warnings": []},
    )

    data = (pathlib.Path(info["path"]) / "changes.diff").read_bytes()
    assert b"r\xe9solution" in data  # raw byte preserved
    assert b"\xef\xbf\xbd" not in data  # no U+FFFD replacement corruption

def test_create_rescue_snapshot_untracked_only_has_no_stash_error(monkeypatch, tmp_path):
    """rc==0 with an empty stash sha (nothing tracked to stash) is legitimate, not an error."""
    import pathlib

    repo, branch = _rescue_fixture_repo(tmp_path)
    (repo / "loose.txt").write_text("untracked only\n", encoding="utf-8")

    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    info = git_ops._create_rescue_snapshot(
        branch, "untracked-test",
        {"current_branch": branch, "dirty_lines": ["?? loose.txt"], "unpushed_lines": [], "warnings": []},
    )

    assert "rescue_stash_error" not in info
    assert "rescue_ref" not in info  # nothing stashable — and no error either
    assert "merge_head" not in info
    assert info["untracked"]["copied_files"] == 1
    copied = pathlib.Path(info["path"]) / "untracked" / "loose.txt"
    assert copied.read_text(encoding="utf-8") == "untracked only\n"

def test_rescue_hook_treats_unreadable_status_as_dirty(monkeypatch, tmp_path):
    """A failing `git status` must be treated as a DIRTY tree: the hook attempts the
    rescue (a clean-shortcut on an unreadable tree would silently skip work it could
    not even see), takes the snapshot WITHOUT the evolution link, and writes the
    durable supervisor.jsonl line before returning the pointer."""
    calls = []

    def fake_git_capture(cmd, *, timeout=None):
        if cmd == ["git", "status", "--porcelain"]:
            return 128, "", "fatal: unreadable index"
        if cmd[:3] == ["git", "rev-parse", "-q"]:
            return 1, "", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)
    monkeypatch.setattr(git_ops, "_collect_repo_sync_state", lambda: {"current_branch": "b"})
    monkeypatch.setattr(
        git_ops, "_create_rescue_snapshot",
        lambda branch, reason, state, link_evolution=True: calls.append(
            (branch, reason, link_evolution)
        ) or {"path": "/r", "ts": "T", "untracked": {}},
    )
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")

    result = git_ops.rescue_before_destructive_rollback("status_unreadable")

    assert calls == [("b", "managed_update_rollback:status_unreadable", False)]
    assert result == {"path": "/r", "ref": "", "ts": "T"}
    log_lines = (tmp_path / "data" / "logs" / "supervisor.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    rows = [json.loads(line) for line in log_lines if line.strip()]
    assert rows[-1]["type"] == "managed_update_rescue_captured"
    assert rows[-1]["rescue_path"] == "/r"

@pytest.mark.parametrize(
    ("merge_rc", "merge_error"),
    [
        pytest.param(git_ops.FETCH_TIMEOUT_RC, "merge probe timed out", id="timeout"),
        pytest.param(1, "merge probe failed", id="rc-one-with-diagnostic"),
    ],
)
def test_rescue_hook_does_not_false_clean_on_unreadable_merge_head(
    monkeypatch, tmp_path, merge_rc, merge_error,
):
    """A failed MERGE_HEAD probe is unknown, not proof that no merge exists."""
    captured_states = []

    def fake_rescue_capture(cmd):
        if cmd == ["git", "status", "--porcelain"]:
            return 0, "", ""
        if cmd == ["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"]:
            return merge_rc, "", merge_error
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "rescue_git_capture", fake_rescue_capture)
    monkeypatch.setattr(
        git_ops, "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros", "dirty_lines": [],
            "unpushed_lines": [], "warnings": [],
        },
    )
    monkeypatch.setattr(
        git_ops, "_create_rescue_snapshot",
        lambda branch, reason, state, link_evolution=True: (
            captured_states.append(dict(state))
            or {"path": "/r", "ts": "T", "warnings": list(state["warnings"])}
        ),
    )
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")

    result = git_ops.rescue_before_destructive_rollback("merge_probe_failed")

    assert result == {"path": "/r", "ref": "", "ts": "T"}
    assert captured_states
    assert captured_states[0]["warnings"] == [
        f"merge_head_error:{merge_error}"
    ]
    rows = [
        json.loads(line)
        for line in (tmp_path / "data" / "logs" / "supervisor.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert rows[-1]["warnings"] == [f"merge_head_error:{merge_error}"]

def test_rescue_hook_clean_tree_without_merge_returns_empty(monkeypatch, tmp_path):
    """Clean tree + no MERGE_HEAD → nothing to rescue: no snapshot, no durable line,
    so a replayed rolling_back boot stays idempotent."""
    repo, _branch = _rescue_fixture_repo(tmp_path)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        git_ops, "_create_rescue_snapshot",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("a clean tree must not be snapshotted")
        ),
    )

    assert git_ops.rescue_before_destructive_rollback("clean") == {}
    assert not (tmp_path / "data" / "logs" / "supervisor.jsonl").exists()

def test_ensure_local_version_tag_accepts_rc_versions(monkeypatch, tmp_path):
    (tmp_path / "VERSION").write_text("4.50.0-rc.2\n", encoding="utf-8")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "_ensure_git_identity", lambda: None)

    calls = []

    def fake_git_capture(cmd):
        calls.append(cmd)
        if cmd == ["git", "tag", "-l", "v4.50.0-rc.2"]:
            return 0, "", ""
        if cmd == ["git", "tag", "-l"]:
            return 0, "", ""
        if cmd == ["git", "rev-parse", "HEAD"]:
            return 0, "abc123", ""
        if cmd == ["git", "tag", "-a", "v4.50.0-rc.2", "-m", "Release v4.50.0-rc.2"]:
            return 0, "", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    git_ops._ensure_local_version_tag()

    assert ["git", "tag", "-a", "v4.50.0-rc.2", "-m", "Release v4.50.0-rc.2"] in calls
