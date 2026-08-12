import json
import os
import subprocess
import time
from types import SimpleNamespace

import pytest

import supervisor.git_ops as git_ops


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


def test_manual_rollback_pins_previous_head_before_reset(tmp_path, monkeypatch):
    repo, first, second = _history_repo(tmp_path)
    git_ops.init(repo, tmp_path / "data", "")
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name=None: False)
    monkeypatch.setattr(git_ops, "load_state", lambda: {})
    monkeypatch.setattr(git_ops, "save_state", lambda _state: None)

    ok, message = git_ops.rollback_to_version(first, reason="test")

    assert ok, message
    assert _git(repo, "rev-parse", "HEAD") == first
    keep_branches = _git(repo, "branch", "--list", "rollback-keep-*").splitlines()
    assert len(keep_branches) == 1
    keep_branch = keep_branches[0].lstrip("* ")
    assert _git(repo, "rev-parse", keep_branch) == second
    assert keep_branch in message


def test_promotion_push_uses_captured_sha_when_dev_advances(tmp_path, monkeypatch):
    repo, first, second = _history_repo(tmp_path)
    _git(repo, "branch", "ouroboros-stable", first)
    git_ops.init(repo, tmp_path / "data", "")
    monkeypatch.setattr(git_ops, "_has_remote", lambda _name=None: True)
    pushed = []

    def fake_push(args, **_kwargs):
        pushed.append(list(args))
        (repo / "value.txt").write_text("three\n", encoding="utf-8")
        _git(repo, "commit", "-qam", "three")
        return 0, "", ""

    monkeypatch.setattr(git_ops, "_git_network_bounded", fake_push)

    ok, result = git_ops.promote_branch_exact(
        "ouroboros", "ouroboros-stable", push_remote=True
    )

    assert ok, result
    assert result["sha"] == second
    assert _git(repo, "rev-parse", "ouroboros-stable") == second
    assert _git(repo, "rev-parse", "ouroboros") != second
    assert pushed == [[
        "push", "origin", f"{second}:refs/heads/ouroboros-stable"
    ]]


def test_event_promotion_refuses_while_managed_update_is_active(monkeypatch):
    import supervisor.events as events
    import supervisor.update_merge as update_merge

    token = object()
    released = []
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: token)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {"phase": "pending_boot_smoke"})
    monkeypatch.setattr(update_merge, "release_update_lock", lambda value: released.append(value))
    monkeypatch.setattr(
        git_ops,
        "promote_branch_exact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("promotion must be fenced")),
    )
    ctx = SimpleNamespace(
        BRANCH_DEV="ouroboros", BRANCH_STABLE="ouroboros-stable", load_state=lambda: {}
    )

    events._handle_promote_to_stable({}, ctx)

    assert released == [token]


def test_git_capture_repairs_corrupt_index(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "index").write_text("broken", encoding="utf-8")
    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)

    calls = {"status": 0, "rebuild": 0}

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        if cmd == ["git", "status", "--porcelain"]:
            calls["status"] += 1
            if calls["status"] == 1:
                return subprocess.CompletedProcess(
                    cmd,
                    128,
                    stdout="",
                    stderr="fatal: .git/index: index file smaller than expected\n",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout=" M changed.py\n", stderr="")
        if cmd == ["git", "reset", "--mixed", "HEAD"]:
            calls["rebuild"] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    rc, stdout, stderr = git_ops.git_capture(["git", "status", "--porcelain"])

    assert rc == 0
    assert stdout == "M changed.py"
    assert stderr == ""
    assert calls["status"] == 2
    assert calls["rebuild"] == 1
    assert any(path.name.startswith("index.corrupt.") for path in git_dir.iterdir())


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

    def fake_capture(cmd):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "status", "--porcelain"]:
            return -9, "", ""
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


def test_compute_managed_update_status_passive_does_not_ensure_remote(monkeypatch):
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
        "ensure_official_update_remote",
        lambda: (_ for _ in ()).throw(AssertionError("passive status mutated remotes")),
    )
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: ("", "", "no cached official tags"),
    )

    def fake_git_capture(cmd):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "rev-parse", "HEAD"]:
            return 0, "abc123", ""
        if cmd == ["git", "status", "--porcelain"]:
            return 0, "", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    status = git_ops.compute_managed_update_status(fetch=False)

    assert status["managed"] is True
    assert "official_status_requires_check" in status["warnings"]


def test_official_fetch_timeout_kills_the_process_tree(monkeypatch):
    import ouroboros.platform_layer as platform_layer
    from ouroboros.tools import shell

    calls = []

    class FakeProcess:
        returncode = 1

        def __init__(self):
            self.communicates = 0

        def communicate(self, timeout=None):
            assert self in shell._active_subprocesses
            self.communicates += 1
            if self.communicates == 1:
                raise subprocess.TimeoutExpired(["git", "fetch"], timeout)
            return "", "still running"

    proc = FakeProcess()
    monkeypatch.setattr(git_ops.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(
        platform_layer,
        "kill_process_tree",
        lambda child: calls.append(child),
    )

    rc, out, error = git_ops.git_fetch_bounded("managed", timeout=0.01)

    assert rc == git_ops.FETCH_TIMEOUT_RC
    assert out == ""
    assert "exceeded" in error
    assert calls == [proc]
    assert proc not in shell._active_subprocesses


def test_dependency_sync_is_panic_tracked_and_killed_on_timeout(monkeypatch):
    import ouroboros.platform_layer as platform_layer
    from ouroboros.tools import shell

    killed = []

    class HungProcess:
        returncode = 1

        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            assert self in shell._active_subprocesses
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["pip", "install"], timeout)
            return -9

    proc = HungProcess()
    monkeypatch.setattr(git_ops.subprocess, "Popen", lambda *_a, **_k: proc)
    monkeypatch.setattr(platform_layer, "kill_process_tree", lambda value: killed.append(value))

    ok, _message = git_ops.sync_runtime_dependencies("managed_update_test")

    assert ok is False
    assert killed == [proc]
    assert proc not in shell._active_subprocesses


def test_managed_update_target_uses_manifest_remote_name(monkeypatch):
    import ouroboros.update_channels as update_channels

    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "official",
            "managed_remote_branch": "ouroboros",
        },
    )
    monkeypatch.setattr(update_channels, "get_update_branch", lambda settings=None: "main")

    remote_name, remote_branch, target_ref = git_ops._managed_update_target()

    assert remote_name == "official"
    assert remote_branch == "main"
    assert target_ref == "official/main"


def test_prepare_managed_update_preserves_dev_branch_not_current_head(monkeypatch, tmp_path):
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {"managed_remote_name": "managed"})
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda: ("managed", "main", "managed/main"))
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: ("refs/ouroboros-managed/tags/v6.87.5", "remote-sha", ""),
    )
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {"current_branch": "ouroboros", "dirty_lines": [], "unpushed_lines": [], "warnings": []},
    )
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda **_kwargs: {
            "path": str(tmp_path / "data" / "archive" / "rescue" / "x"),
            "untracked": {"copied_files": 0, "skipped_files": 0, "truncated": False},
        },
    )
    intent_writes = []
    monkeypatch.setattr(git_ops, "_write_update_intent", lambda payload: intent_writes.append(payload))
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, _payload: None)

    capture_calls = []

    def fake_git_capture(cmd):
        capture_calls.append(cmd)
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "rev-parse", "--verify", "HEAD"]:
            return 0, "base-sha", ""
        if cmd == ["git", "rev-parse", "--verify", "managed/main^{commit}"]:
            return 0, "remote-sha", ""
        if cmd == ["git", "rev-list", "--left-right", "--count", "ouroboros...remote-sha"]:
            return 0, "1 0", ""
        if cmd[:2] == ["git", "branch"] and cmd[-1] == "ouroboros":
            return 0, "", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    ok, payload = git_ops.prepare_managed_update(
        "replace", expected_base_sha="base-sha", expected_target_sha="remote-sha",
        arm_intent=False,
    )

    assert ok is True
    assert payload["keep_branch"].startswith("local-keep-")
    assert payload["update_intent"]["target_sha"] == "remote-sha"
    assert intent_writes == []
    assert any(cmd[:2] == ["git", "branch"] and cmd[-1] == "ouroboros" for cmd in capture_calls)


def test_prepare_managed_update_blocks_when_ahead_check_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {"managed_remote_name": "managed"})
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda: ("managed", "main", "managed/main"))
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: ("refs/ouroboros-managed/tags/v6.87.5", "remote-sha", ""),
    )
    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {"current_branch": "ouroboros", "dirty_lines": [], "unpushed_lines": [], "warnings": []},
    )
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda **_kwargs: {
            "path": str(tmp_path / "data" / "archive" / "rescue" / "x"),
            "untracked": {"copied_files": 0, "skipped_files": 0, "truncated": False},
        },
    )

    def fake_git_capture(cmd):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "rev-parse", "--verify", "HEAD"]:
            return 0, "base-sha", ""
        if cmd == ["git", "rev-parse", "--verify", "managed/main^{commit}"]:
            return 0, "remote-sha", ""
        if cmd == ["git", "rev-list", "--left-right", "--count", "ouroboros...remote-sha"]:
            return 128, "", "bad revision"
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    ok, payload = git_ops.prepare_managed_update(
        "replace", expected_base_sha="base-sha", expected_target_sha="remote-sha"
    )

    assert ok is False
    assert "Could not compare local branch with managed update target" in payload["error"]


def test_safe_restart_fallback_does_not_rewrite_dev_branch(monkeypatch):
    checkout_calls = []

    def fake_checkout(branch, reason="unspecified", unsynced_policy="ignore"):
        checkout_calls.append((branch, reason, unsynced_policy))
        return True, "ok"

    import_results = [
        {"ok": False, "stdout": "", "stderr": "broken dev", "returncode": 1},
        {"ok": True, "stdout": "import_ok", "stderr": "", "returncode": 0},
    ]

    monkeypatch.setattr(git_ops, "checkout_and_reset", fake_checkout)
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda reason: (True, reason))
    monkeypatch.setattr(git_ops, "import_test", lambda: import_results.pop(0))
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, _payload: None)

    ok, message = git_ops.safe_restart(reason="owner_restart", unsynced_policy="rescue_and_reset")

    assert ok is True
    assert message == "OK: fell back to ouroboros-stable"
    assert checkout_calls == [
        ("ouroboros", "owner_restart", "rescue_and_reset"),
        ("ouroboros-stable", "owner_restart_fallback_stable", "rescue_and_reset"),
    ]


def test_a_stand_can_keep_its_pinned_checkout_across_restarts(monkeypatch):
    """OUROBOROS_DISABLE_MANAGED_UPDATES=1 is the lever for running a stand.

    A test stand launched against a PINNED checkout had that checkout moved under
    the operator mid-test: the launcher-managed path resets the repo onto the
    managed dev branch on every start (reflog "checkout: moving from <sha> to
    ouroboros", version 6.89.0 -> 6.87.5). server.py already had a local-dev
    branch that skips the BOOTSTRAP reset, but bootstrap is only one of three
    callers — the owner restart and the agent restart reset the tree too. The
    lever therefore sits at `safe_restart`, the choke point all three share, and
    keeps the parts that are not a tree move: deps sync and the import test.
    """
    monkeypatch.setenv("OUROBOROS_DISABLE_MANAGED_UPDATES", "1")

    def fail_checkout(*_args, **_kwargs):
        raise AssertionError("a stand with managed updates disabled must not be checked out")

    events = []
    deps = []
    monkeypatch.setattr(git_ops, "checkout_and_reset", fail_checkout)
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies",
                        lambda reason: deps.append(reason) or (True, reason))
    monkeypatch.setattr(git_ops, "import_test",
                        lambda: {"ok": True, "stdout": "", "stderr": "", "returncode": 0})
    monkeypatch.setattr(git_ops, "append_jsonl", lambda _path, payload: events.append(payload))

    ok, message = git_ops.safe_restart(reason="bootstrap", unsynced_policy="rescue_and_reset")
    assert ok is True
    assert "managed checkout disabled" in message
    assert deps == ["bootstrap"], "the deps sync is not a tree move and must still run"
    assert [e["type"] for e in events] == ["managed_checkout_disabled"], \
        "a suppressed checkout is disclosed, never silent"

    # A broken tree still fails closed — the lever pins the checkout, it does not
    # promise the pinned checkout imports.
    monkeypatch.setattr(git_ops, "import_test",
                        lambda: {"ok": False, "stdout": "", "stderr": "boom", "returncode": 1})
    ok_broken, message_broken = git_ops.safe_restart(reason="owner_restart")
    assert ok_broken is False
    assert "Import test failed" in message_broken

    # Without the lever nothing changes: the ordinary managed path still runs.
    monkeypatch.delenv("OUROBOROS_DISABLE_MANAGED_UPDATES")
    checkouts = []
    monkeypatch.setattr(git_ops, "checkout_and_reset",
                        lambda branch, reason="unspecified", unsynced_policy="ignore":
                        checkouts.append(branch) or (True, "ok"))
    monkeypatch.setattr(git_ops, "import_test",
                        lambda: {"ok": True, "stdout": "", "stderr": "", "returncode": 0})
    assert git_ops.safe_restart(reason="bootstrap")[0] is True
    assert checkouts == [git_ops.BRANCH_DEV]


def test_configure_remote_adds_origin_even_when_managed_remote_exists(monkeypatch):
    calls = []

    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: name in (None, "managed"))
    monkeypatch.setattr(
        git_ops,
        "git_capture",
        lambda cmd: calls.append(cmd) or (0, "", ""),
    )
    monkeypatch.setattr(
        git_ops,
        "_configure_credential_helper",
        lambda repo_slug, token: calls.append(("helper", repo_slug, token)),
    )

    ok, message = git_ops.configure_remote("razzant/ouroboros", "ghp_test")

    assert ok
    assert message == "ok"
    assert ["git", "remote", "add", "origin", "https://github.com/razzant/ouroboros.git"] in calls


def test_collect_repo_sync_state_prefers_managed_remote(monkeypatch):
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "managed",
            "managed_remote_branch": "ouroboros",
        },
    )
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: name in (None, "managed"))

    def fake_git_capture(cmd):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "status", "--porcelain"]:
            return 0, "", ""
        if cmd == ["git", "log", "--oneline", "managed/ouroboros..HEAD"]:
            return 0, "abc123 local commit\n", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    state = git_ops._collect_repo_sync_state()

    assert state["current_branch"] == "ouroboros"
    assert state["unpushed_lines"] == ["abc123 local commit"]


def test_checkout_and_reset_keeps_bundled_sha_on_first_managed_bootstrap(monkeypatch, tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / git_ops.BOOTSTRAP_PIN_MARKER_NAME).write_text("pending\n", encoding="utf-8")

    monkeypatch.setattr(git_ops, "REPO_DIR", tmp_path)
    monkeypatch.setattr(git_ops, "_has_remote", lambda name=None: name in (None, "managed"))
    monkeypatch.setattr(
        git_ops,
        "_read_managed_repo_meta",
        lambda: {
            "managed_remote_name": "managed",
            "managed_remote_branch": "ouroboros",
            "source_sha": "bundle123",
        },
    )
    monkeypatch.setattr(git_ops, "load_state", lambda: {"current_sha": "bundle123"})

    saved_state = {}
    monkeypatch.setattr(git_ops, "save_state", lambda state: saved_state.update(state))

    def fake_git_capture(cmd):
        if cmd == ["git", "rev-parse", "HEAD"]:
            return 0, "bundle123", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)

    calls = []

    def fake_run(cmd, cwd=None, capture_output=False, text=False, check=False, env=None):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--verify", "ouroboros"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="bundle123\n", stderr="")
        if cmd[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "reset"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="bundle123\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops.subprocess, "run", fake_run)

    ok, message = git_ops.checkout_and_reset("ouroboros", reason="bootstrap", unsynced_policy="ignore")

    assert ok
    assert message == "ok"
    assert ["git", "fetch", "managed"] not in calls
    assert saved_state["current_sha"] == "bundle123"
    assert not (git_dir / git_ops.BOOTSTRAP_PIN_MARKER_NAME).exists()


def test_ensure_official_update_remote_uses_manifest_remote_name(monkeypatch):
    captured = []
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {"managed_remote_name": "official"})
    monkeypatch.setattr(git_ops, "_list_remotes", lambda: [])
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: captured.append(cmd) or (0, "", ""))
    ok, _msg = git_ops.ensure_official_update_remote()
    assert ok
    assert ["git", "remote", "add", "official", git_ops.OFFICIAL_UPDATE_REMOTE_URL] in captured


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

    def fake_git_capture(cmd):
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
