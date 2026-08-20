"""Checkout and reset: what it is allowed to destroy, and when it must block instead.

Split verbatim out of ``tests/test_git_ops_recovery.py`` by theme. This module owns the
stale index lock it clears, the fetch failure it survives, the rescue snapshot it will not
proceed without, the local head it preserves across a managed restart, the merge states and
unreadable reads it blocks on, and the explicit update intent it applies without ever
falling back to a branch tip.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

import supervisor.git_ops as git_ops

from tests._git_ops_recovery_shared import _git


def test_checkout_and_reset_removes_stale_index_lock(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock_path = git_dir / "index.lock"
    lock_path.write_text("lock", encoding="utf-8")
    stale_ts = time.time() - 60
    os.utime(lock_path, (stale_ts, stale_ts))

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: False)
    monkeypatch.setattr(git_ops, "load_state", lambda: {})

    saved_state = {}
    monkeypatch.setattr(git_ops, "save_state", lambda state: saved_state.update(state))

    calls = {"checkout": 0}

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            calls["checkout"] += 1
            if calls["checkout"] == 1:
                return subprocess.CompletedProcess(
                    cmd,
                    128,
                    stdout="",
                    stderr=f"fatal: Unable to create '{git_dir / 'index.lock'}': File exists.\n",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "reset"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset("ouroboros", unsynced_policy="ignore")

    assert ok
    assert message == "ok"
    assert calls["checkout"] == 2
    assert not lock_path.exists()
    assert saved_state["current_branch"] == "ouroboros"
    assert saved_state["current_sha"] == "abc123"

def test_checkout_and_reset_continues_when_fetch_fails(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: name in (None, "origin"))
    monkeypatch.setattr(git_ops, "load_state", lambda: {})

    saved_state = {}
    monkeypatch.setattr(git_ops, "save_state", lambda state: saved_state.update(state))

    events = []
    monkeypatch.setattr(git_ops, "append_jsonl", lambda path, payload: events.append(payload))

    def fake_git_capture(cmd):
        if cmd == ["git", "fetch", "origin"]:
            return 1, "", "network down"
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "reset"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="def456\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset("ouroboros", reason="restart", unsynced_policy="ignore")

    assert ok
    assert message == "ok"
    assert saved_state["current_branch"] == "ouroboros"
    assert saved_state["current_sha"] == "def456"
    assert events
    assert events[0]["type"] == "reset_fetch_failed"
    assert events[0]["continuing_local_reset"] is True

def test_checkout_and_reset_blocks_when_rescue_snapshot_fails(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: False)
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros",
            "dirty_lines": [" M BIBLE.md"],
            "unpushed_lines": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
    )
    events = []
    monkeypatch.setattr(git_ops, "append_jsonl", lambda path, payload: events.append(payload))

    reset_calls = []

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        if cmd[:2] == ["git", "reset"]:
            reset_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros",
        reason="restart",
        unsynced_policy="rescue_and_reset",
    )

    assert ok is False
    assert "rescue snapshot failed" in message
    assert reset_calls == []
    assert events and events[-1]["type"] == "reset_blocked_rescue_failed"
    assert events[-1]["incomplete_reason"] == "snapshot_error"

def test_checkout_and_reset_blocks_when_untracked_rescue_is_truncated(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: False)
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros",
            "dirty_lines": ["?? large.bin"],
            "unpushed_lines": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda **_kwargs: {
            "path": str(tmp_path / "data" / "archive" / "rescue" / "x"),
            "untracked": {"copied_files": 0, "skipped_files": 0, "truncated": True},
        },
    )
    events = []
    monkeypatch.setattr(git_ops, "append_jsonl", lambda path, payload: events.append(payload))
    reset_calls = []

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        if cmd[:2] == ["git", "reset"]:
            reset_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros",
        reason="restart",
        unsynced_policy="rescue_and_reset",
    )

    assert ok is False
    assert "untracked-file rescue was incomplete" in message
    assert reset_calls == []
    assert events and events[-1]["type"] == "reset_blocked_rescue_incomplete"
    assert events[-1]["incomplete_reason"] == "untracked_rescue"
    assert events[-1]["incomplete_detail"] == "untracked rescue copy was truncated"

def test_checkout_and_reset_preserves_local_head_on_managed_restart(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: name in (None, "managed"))
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "managed",
            "managed_remote_branch": "ouroboros",
            "managed_remote_stable_branch": "ouroboros-stable",
        },
    )
    monkeypatch.setattr(git_ops, "load_state", lambda: {})

    saved_state = {}
    monkeypatch.setattr(git_ops, "save_state", lambda state: saved_state.update(state))

    def fake_git_capture(cmd):
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    calls = []

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--verify", "ouroboros"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="local-sha\n", stderr="")
        if cmd == ["git", "checkout", "ouroboros"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "reset", "--hard", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="local-sha\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset("ouroboros", reason="restart", unsynced_policy="ignore")

    assert ok
    assert message == "ok"
    assert ["git", "fetch", "managed"] not in calls
    assert ["git", "checkout", "-B", "ouroboros", "managed/ouroboros"] not in calls
    assert ["git", "checkout", "ouroboros"] in calls
    assert saved_state["current_branch"] == "ouroboros"
    assert saved_state["current_sha"] == "local-sha"

def test_checkout_and_reset_cleans_untracked_after_managed_restart_rescue(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "managed",
            "managed_remote_branch": "ouroboros",
        },
    )
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    monkeypatch.setattr(git_ops, "save_state", lambda _state: None)
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros",
            "dirty_lines": ["?? scratch.py"],
            "unpushed_lines": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda **_kwargs: {
            "path": str(tmp_path / "data" / "archive" / "rescue" / "x"),
            "untracked": {"copied_files": 1, "skipped_files": 0, "truncated": False},
        },
    )
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, _payload: None)
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: (_ for _ in ()).throw(AssertionError(cmd)))

    calls = []

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--verify", "ouroboros"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="local-sha\n", stderr="")
        if cmd == ["git", "checkout", "ouroboros"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "reset", "--hard", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "clean", "-fd"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="local-sha\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros",
        reason="restart",
        unsynced_policy="rescue_and_reset",
    )

    assert ok
    assert message == "ok"
    assert ["git", "clean", "-fd"] in calls
    assert calls.index(["git", "clean", "-fd"]) < calls.index(["git", "checkout", "ouroboros"])

def test_checkout_and_reset_does_not_rescue_for_only_managed_ahead_commits(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {"managed_remote_name": "managed", "managed_remote_branch": "ouroboros"},
    )
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    monkeypatch.setattr(git_ops, "save_state", lambda _state: None)
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros",
            "dirty_lines": [],
            "unpushed_lines": ["abc123 local self-modification"],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("ahead-only restart should not rescue")),
    )
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: (_ for _ in ()).throw(AssertionError(cmd)))

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        if cmd == ["git", "rev-parse", "--verify", "ouroboros"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="local-sha\n", stderr="")
        if cmd[:2] in (["git", "reset"], ["git", "clean"], ["git", "checkout"]):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="local-sha\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros",
        reason="restart",
        unsynced_policy="rescue_and_reset",
    )

    assert ok
    assert message == "ok"

def test_checkout_and_reset_blocks_when_status_read_is_unreadable(monkeypatch, tmp_path):
    """A `git status` failure must not read as a clean tree: the admission gate treats
    it the same as a genuinely dirty tree, even though dirty_lines itself is empty."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: False)
    events = []
    monkeypatch.setattr(git_ops, "append_jsonl", lambda path, payload: events.append(payload))

    def fake_capture(cmd, *, timeout=None):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "status", "--porcelain"]:
            return -9, "", ""
        if cmd == ["git", "remote"]:
            return 0, "", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_capture)
    monkeypatch.setattr(
        git_ops,
        "_run_git_resilient",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError((args, kwargs))),
    )

    ok, message = git_ops.checkout_and_reset(
        "ouroboros", reason="restart", unsynced_policy="block",
    )

    assert ok is False
    assert "status_unreadable" in message
    assert events and events[-1]["type"] == "reset_blocked_unsynced_state"
    assert events[-1]["dirty_count"] == 0
    assert events[-1]["warnings"] == ["status_error:git status exited -9 without stderr"]

@pytest.mark.serial
def test_checkout_and_reset_blocks_clean_merge_in_linked_worktree(monkeypatch, tmp_path):
    """A linked worktree stores MERGE_HEAD outside its .git pointer file."""
    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@ouroboros")
    _git(repo, "commit", "--allow-empty", "-qm", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "branch", "side")
    _git(repo, "commit", "--allow-empty", "-qm", "main")
    _git(repo, "worktree", "add", "-q", str(linked), "side")
    _git(linked, "commit", "--allow-empty", "-qm", "side")
    _git(linked, "merge", "--no-commit", "--no-ff", "main")

    assert _git(linked, "status", "--porcelain") == ""
    merge_head = linked / _git(linked, "rev-parse", "--git-path", "MERGE_HEAD")
    assert merge_head.is_file()

    monkeypatch.setattr(git_ops, "REPO_DIR", linked)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")

    ok, message = git_ops.checkout_and_reset(
        "side", reason="restart", unsynced_policy="block",
    )

    assert ok is False
    assert "merge_in_progress" in message
    assert merge_head.is_file()

def test_checkout_and_reset_blocks_on_unreadable_merge_head(monkeypatch, tmp_path):
    """MERGE_HEAD present but not a resolvable SHA must force the block branch too,
    not just a clean git-status read."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "MERGE_HEAD").write_text("not-a-sha\n", encoding="utf-8")

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: False)
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros",
            "dirty_lines": [],
            "unpushed_lines": [],
            "warnings": [],
        },
    )
    events = []
    monkeypatch.setattr(git_ops, "append_jsonl", lambda path, payload: events.append(payload))

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros", reason="restart", unsynced_policy="block",
    )

    assert ok is False
    assert "merge_head_unreadable" in message
    assert events and events[-1]["type"] == "reset_blocked_unsynced_state"
    assert events[-1]["dirty_count"] == 0

def test_checkout_and_reset_blocks_on_merge_in_progress(monkeypatch, tmp_path):
    """A resolvable MERGE_HEAD (an actual in-progress merge) was never consulted by
    this admission gate before; it must now force the block branch."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "MERGE_HEAD").write_text("a" * 40 + "\n", encoding="utf-8")

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: False)
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros",
            "dirty_lines": [],
            "unpushed_lines": [],
            "warnings": [],
        },
    )
    events = []
    monkeypatch.setattr(git_ops, "append_jsonl", lambda path, payload: events.append(payload))

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros", reason="restart", unsynced_policy="block",
    )

    assert ok is False
    assert "merge_in_progress" in message
    assert events and events[-1]["type"] == "reset_blocked_unsynced_state"
    assert events[-1]["dirty_count"] == 0

def test_checkout_and_reset_applies_explicit_update_intent(monkeypatch, tmp_path):
    import supervisor.update_merge as update_merge

    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "managed",
            "managed_remote_branch": "ouroboros",
            "managed_remote_stable_branch": "ouroboros-stable",
        },
    )
    monkeypatch.setattr(
        git_ops,
        "_read_update_intent",
        lambda: {"branch": "ouroboros", "target_sha": "remote-sha"},
    )
    monkeypatch.setattr(
        update_merge,
        "read_update_tx_strict",
        lambda: ("valid", {"phase": "applying_replace", "target_sha": "remote-sha"}),
    )
    monkeypatch.setattr(git_ops._update_source, "official_ref_has_constitution", lambda *_a, **_k: True)
    monkeypatch.setattr(git_ops, "load_state", lambda: {})

    saved_state = {}
    monkeypatch.setattr(git_ops, "save_state", lambda state: saved_state.update(state))

    def fake_git_capture(cmd):
        if cmd == ["git", "rev-parse", "--verify", "remote-sha^{commit}"]:
            return 0, "remote-sha", ""
        if cmd == ["git", "rev-list", "--left-right", "--count", "ouroboros...remote-sha"]:
            return 0, "0 1", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    calls = []

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--verify", "remote-sha"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="remote-sha\n", stderr="")
        if cmd[:4] == ["git", "checkout", "-B", "ouroboros"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "reset"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "clean"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="remote-sha\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros",
        reason="ui_update_apply",
        unsynced_policy="ignore",
    )

    assert ok
    assert message == "ok"
    assert ["git", "checkout", "-B", "ouroboros", "remote-sha"] in calls
    assert saved_state["current_branch"] == "ouroboros"
    assert saved_state["current_sha"] == "remote-sha"

def test_checkout_and_reset_preserves_ahead_head_before_update_intent(monkeypatch, tmp_path):
    import supervisor.update_merge as update_merge

    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "managed",
            "managed_remote_branch": "ouroboros",
        },
    )
    monkeypatch.setattr(
        git_ops,
        "_read_update_intent",
        lambda: {"branch": "ouroboros", "target_sha": "remote-sha"},
    )
    monkeypatch.setattr(
        update_merge,
        "read_update_tx_strict",
        lambda: ("valid", {"phase": "applying_replace", "target_sha": "remote-sha"}),
    )
    monkeypatch.setattr(git_ops._update_source, "official_ref_has_constitution", lambda *_a, **_k: True)
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    monkeypatch.setattr(git_ops, "save_state", lambda _state: None)
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, _payload: None)

    capture_calls = []

    def fake_git_capture(cmd):
        capture_calls.append(cmd)
        if cmd == ["git", "rev-parse", "--verify", "remote-sha^{commit}"]:
            return 0, "remote-sha", ""
        if cmd == ["git", "rev-list", "--left-right", "--count", "ouroboros...remote-sha"]:
            return 0, "2 1", ""
        if cmd[:2] == ["git", "branch"] and cmd[-1] == "ouroboros":
            return 0, "", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        if cmd == ["git", "rev-parse", "--verify", "remote-sha"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="remote-sha\n", stderr="")
        if cmd[:2] in (["git", "reset"], ["git", "clean"], ["git", "checkout"]):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="remote-sha\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros",
        reason="ui_update_apply",
        unsynced_policy="ignore",
    )

    assert ok
    assert message == "ok"
    assert any(cmd[:2] == ["git", "branch"] and cmd[-1] == "ouroboros" for cmd in capture_calls)

def test_checkout_and_reset_blocks_when_update_ahead_check_fails(monkeypatch, tmp_path):
    import supervisor.update_merge as update_merge

    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {"managed_remote_name": "managed", "managed_remote_branch": "ouroboros"},
    )
    monkeypatch.setattr(
        git_ops,
        "_read_update_intent",
        lambda: {"branch": "ouroboros", "target_sha": "remote-sha"},
    )
    monkeypatch.setattr(
        update_merge,
        "read_update_tx_strict",
        lambda: ("valid", {"phase": "applying_replace", "target_sha": "remote-sha"}),
    )
    monkeypatch.setattr(git_ops._update_source, "official_ref_has_constitution", lambda *_a, **_k: True)

    def fake_git_capture(cmd):
        if cmd == ["git", "rev-parse", "--verify", "remote-sha^{commit}"]:
            return 0, "remote-sha", ""
        if cmd == ["git", "rev-list", "--left-right", "--count", "ouroboros...remote-sha"]:
            return 128, "", "bad revision"
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    checkout_calls = []

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        checkout_calls.append(cmd)
        if cmd == ["git", "rev-parse", "--verify", "remote-sha"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="remote-sha\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros",
        reason="ui_update_apply",
        unsynced_policy="ignore",
    )

    assert ok is False
    assert "Could not preserve local branch before official update" in message
    assert ["git", "checkout", "-B", "ouroboros", "remote-sha"] not in checkout_calls

def test_checkout_and_reset_invalid_update_intent_never_falls_back_to_branch_tip(
    monkeypatch, tmp_path
):
    import supervisor.update_merge as update_merge

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {"managed_remote_name": "managed", "managed_remote_branch": "ouroboros"},
    )
    monkeypatch.setattr(
        git_ops,
        "_read_update_intent",
        lambda: {"branch": "ouroboros", "target_sha": "missing-sha"},
    )
    monkeypatch.setattr(
        update_merge,
        "read_update_tx_strict",
        lambda: ("valid", {"phase": "applying_replace", "target_sha": "missing-sha"}),
    )
    cleared = []
    monkeypatch.setattr(git_ops, "_clear_update_intent", lambda: cleared.append(True) or True)
    monkeypatch.setattr(git_ops, "append_jsonl", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        git_ops,
        "git_capture",
        lambda cmd: (1, "", "unknown revision")
        if cmd == ["git", "rev-parse", "--verify", "missing-sha^{commit}"]
        else (_ for _ in ()).throw(AssertionError(cmd)),
    )
    monkeypatch.setattr(
        git_ops.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid intent must not touch the checkout")
        ),
    )

    ok, message = git_ops.checkout_and_reset(
        "ouroboros",
        reason="ui_update_apply",
        unsynced_policy="ignore",
    )

    assert ok is False
    assert "checkout was left unchanged" in message
    assert cleared == [True]

def test_checkout_and_reset_rejects_orphan_or_mismatched_update_intent(
    monkeypatch, tmp_path
):
    import supervisor.update_merge as update_merge

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {"managed_remote_name": "managed"},
    )
    monkeypatch.setattr(
        git_ops,
        "_read_update_intent",
        lambda: {"branch": "ouroboros", "target_sha": "intent-sha"},
    )
    monkeypatch.setattr(
        git_ops,
        "git_capture",
        lambda cmd: (0, "intent-sha", "")
        if cmd == ["git", "rev-parse", "--verify", "intent-sha^{commit}"]
        else (_ for _ in ()).throw(AssertionError(cmd)),
    )
    monkeypatch.setattr(
        git_ops._update_source,
        "official_ref_has_constitution",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(git_ops, "append_jsonl", lambda *_a, **_k: None)
    monkeypatch.setattr(git_ops, "_clear_update_intent", lambda: True)
    monkeypatch.setattr(
        git_ops.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("orphan intent must not touch the checkout")
        ),
    )

    for tx in (
        ("absent", {}),
        ("valid", {"phase": "applying_replace", "target_sha": "other-sha"}),
    ):
        monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda tx=tx: tx)
        ok, message = git_ops.checkout_and_reset(
            "ouroboros", reason="ui_update_apply", unsynced_policy="ignore"
        )
        assert ok is False
        assert "checkout was left unchanged" in message

def test_checkout_and_reset_rejects_target_without_constitution(monkeypatch, tmp_path):
    import supervisor.update_merge as update_merge

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {"managed": True})
    monkeypatch.setattr(
        git_ops,
        "_read_update_intent",
        lambda: {"branch": "ouroboros", "target_sha": "target-sha"},
    )
    monkeypatch.setattr(
        update_merge,
        "read_update_tx_strict",
        lambda: ("valid", {"phase": "applying_replace", "target_sha": "target-sha"}),
    )
    monkeypatch.setattr(
        git_ops,
        "git_capture",
        lambda cmd: (0, "target-sha", "")
        if cmd == ["git", "rev-parse", "--verify", "target-sha^{commit}"]
        else (_ for _ in ()).throw(AssertionError(cmd)),
    )
    monkeypatch.setattr(
        git_ops._update_source,
        "official_ref_has_constitution",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(git_ops, "append_jsonl", lambda *_a, **_k: None)
    monkeypatch.setattr(git_ops, "_clear_update_intent", lambda: True)

    ok, message = git_ops.checkout_and_reset(
        "ouroboros", reason="ui_update_apply", unsynced_policy="ignore"
    )

    assert ok is False
    assert "checkout was left unchanged" in message
