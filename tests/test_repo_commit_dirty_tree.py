"""Tests for the 2026-05-04 _repo_commit_push checkout-on-dirty-tree fix.

The original code unconditionally ran ``git checkout <branch_dev> --``
before staging, which fails on dirty trees — including the very files
the agent is trying to commit. Ouro hit this loop on 2026-05-04 while
landing the OBC movement tools and burned 16+ rounds working around it
before the gate retry cap stopped him.

Preparation now skips checkout on the target branch, preserving a staged
merge and dirty work. A required branch switch still reports checkout
failure and never reaches staging.

Pinned tests so the regression can't reopen.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import patch



def _fake_run_cmd_sequence(*returns):
    """Build a side_effect list that pops returns in order."""
    return list(returns)


def _ctx(tmp_path: pathlib.Path) -> SimpleNamespace:
    drive = tmp_path / "drive"
    drive.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=drive,
        branch_dev="ouroboros",
        last_push_succeeded=True,
        pending_events=[],
        pending_restart_reason=None,
        pending_restart_policy=None,
        current_task_type="task",
    )


def test_checkout_already_on_branch_proceeds_after_failure(tmp_path):
    """When already on branch_dev and checkout fails (e.g. dirty tree
    no-op-but-complained), the cycle proceeds to staging. The original
    behavior would have aborted — the fix makes the failure incidental."""
    from ouroboros.tools import git as git_module

    ctx = _ctx(tmp_path)
    captured: list[str] = []

    def fake_run(cmd, cwd=None, **_):
        captured.append(" ".join(map(str, cmd)))
        if cmd[:2] == ["git", "checkout"]:
            raise Exception("would overwrite local changes")
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "ouroboros\n"  # we're already on branch_dev
        return ""

    # Stub _run_reviewed_stage_cycle to short-circuit after our checkout
    # path runs — we only care about whether the cycle was *reached*.
    cycle_called = []

    def fake_stage_cycle(ctx, msg, start, **_):
        cycle_called.append(True)
        return {"status": "passed", "message": "",
                "pre_fingerprint": {"fingerprint": "x"},
                "post_fingerprint": {"fingerprint": "x"}}

    with patch.object(git_module, "run_cmd", side_effect=fake_run), \
         patch.object(git_module, "_run_reviewed_stage_cycle",
                      side_effect=fake_stage_cycle), \
         patch.object(git_module, "_acquire_git_lock", return_value=None), \
         patch.object(git_module, "_release_git_lock"), \
         patch.object(git_module, "_record_commit_attempt"), \
         patch.object(git_module, "_post_commit_result", return_value=None), \
         patch.object(git_module, "_auto_tag_on_version_bump", return_value={}), \
         patch.object(git_module, "_auto_push", return_value="ok"):
        try:
            git_module._repo_commit_push(ctx, "test commit")
        except Exception:
            pass  # other downstream failures are out of scope for this test

    assert cycle_called, (
        "stage cycle was not reached — checkout failure aborted commit even "
        "though we were already on branch_dev (regression of 2026-05-04 fix)"
    )


def test_checkout_on_different_branch_with_failure_aborts_cleanly(tmp_path):
    """When checkout fails AND we're on a different branch, abort with the
    original GIT_ERROR (checkout) message — preserves the legitimate
    failure path."""
    from ouroboros.tools import git as git_module

    ctx = _ctx(tmp_path)

    def fake_run(cmd, cwd=None, **_):
        if cmd[:2] == ["git", "checkout"]:
            raise Exception("would overwrite local changes")
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "feature/some-other-branch\n"
        if cmd == ["git", "rev-parse", "--git-path", "MERGE_HEAD"]:
            return ".git/MERGE_HEAD"
        return ""

    cycle_called = []

    def fake_stage_cycle(*args, **kwargs):
        cycle_called.append(True)
        return {"status": "passed"}

    with patch.object(git_module, "run_cmd", side_effect=fake_run), \
         patch.object(git_module, "_run_reviewed_stage_cycle",
                      side_effect=fake_stage_cycle), \
         patch.object(git_module, "_acquire_git_lock", return_value=None), \
         patch.object(git_module, "_release_git_lock"), \
         patch.object(git_module, "_record_commit_attempt"):
        result = git_module._repo_commit_push(ctx, "test commit")

    assert "GIT_ERROR" in result and "checkout" in result
    assert not cycle_called, (
        "stage cycle should NOT be reached when on a different branch with "
        "checkout failure — that's the legitimate failure path"
    )


def test_checkout_failure_on_target_branch_blocks_unmerged_index(tmp_path):
    """Already being on branch_dev is not enough when the index is unmerged."""
    from ouroboros.tools import git as git_module

    ctx = _ctx(tmp_path)
    cycle_called = []

    def fake_run(cmd, cwd=None, **_):
        if cmd[:2] == ["git", "checkout"]:
            raise Exception("would overwrite local changes")
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "ouroboros\n"
        if cmd[:4] == ["git", "diff", "--name-only", "--diff-filter=U"]:
            return "ouroboros/tools/git.py\n"
        return ""

    def fake_stage_cycle(*args, **kwargs):
        cycle_called.append(True)
        return {"status": "passed"}

    with patch.object(git_module, "run_cmd", side_effect=fake_run), \
         patch.object(git_module, "_run_reviewed_stage_cycle",
                      side_effect=fake_stage_cycle), \
         patch.object(git_module, "_acquire_git_lock", return_value=None), \
         patch.object(git_module, "_release_git_lock"), \
         patch.object(git_module, "_record_commit_attempt"):
        result = git_module._repo_commit_push(ctx, "test commit")

    assert "GIT_ERROR" in result and "unmerged paths" in result
    assert "ouroboros/tools/git.py" in result
    assert not cycle_called


def test_checkout_failure_on_target_branch_blocks_unknown_index_state(tmp_path):
    """If index state cannot be verified, keep checkout failure blocking."""
    from ouroboros.tools import git as git_module

    ctx = _ctx(tmp_path)
    cycle_called = []

    def fake_run(cmd, cwd=None, **_):
        if cmd[:2] == ["git", "checkout"]:
            raise Exception("would overwrite local changes")
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "ouroboros\n"
        if cmd[:4] == ["git", "diff", "--name-only", "--diff-filter=U"]:
            raise Exception("index.lock exists")
        return ""

    def fake_stage_cycle(*args, **kwargs):
        cycle_called.append(True)
        return {"status": "passed"}

    with patch.object(git_module, "run_cmd", side_effect=fake_run), \
         patch.object(git_module, "_run_reviewed_stage_cycle",
                      side_effect=fake_stage_cycle), \
         patch.object(git_module, "_acquire_git_lock", return_value=None), \
         patch.object(git_module, "_release_git_lock"), \
         patch.object(git_module, "_record_commit_attempt"):
        result = git_module._repo_commit_push(ctx, "test commit")

    assert "GIT_ERROR" in result and "Could not verify index state" in result
    assert "index.lock exists" in result
    assert not cycle_called


def test_checkout_succeeds_normally_when_clean(tmp_path):
    """Happy path: checkout succeeds, cycle proceeds. The fix must not
    change behavior when checkout works the first time."""
    from ouroboros.tools import git as git_module

    ctx = _ctx(tmp_path)

    captured = []

    def fake_run(cmd, cwd=None, **_):
        captured.append(cmd)
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "feature\n"
        if cmd == ["git", "rev-parse", "--git-path", "MERGE_HEAD"]:
            return ".git/MERGE_HEAD"
        return ""

    cycle_called = []

    def fake_stage_cycle(*args, **kwargs):
        cycle_called.append(True)
        return {"status": "passed",
                "pre_fingerprint": {"fingerprint": "x"},
                "post_fingerprint": {"fingerprint": "x"}}

    with patch.object(git_module, "run_cmd", side_effect=fake_run), \
         patch.object(git_module, "_run_reviewed_stage_cycle",
                      side_effect=fake_stage_cycle), \
         patch.object(git_module, "_acquire_git_lock", return_value=None), \
         patch.object(git_module, "_release_git_lock"), \
         patch.object(git_module, "_record_commit_attempt"), \
         patch.object(git_module, "_post_commit_result", return_value=None), \
         patch.object(git_module, "_auto_tag_on_version_bump", return_value={}), \
         patch.object(git_module, "_auto_push", return_value="ok"):
        try:
            git_module._repo_commit_push(ctx, "test commit")
        except Exception:
            pass

    assert cycle_called, "stage cycle must run on the happy path"
    assert ["git", "checkout", "--no-guess", "ouroboros", "--"] in captured
