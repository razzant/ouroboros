"""(a) Integration test of the ACTUAL _repo_commit_push code path for the BUG1 detached-HEAD fix.

fix_verification.sh checks the pure-git LOGIC; this exercises the real ouroboros.tools.git code:
on a detached HEAD it must issue `git checkout -B ctx.branch_dev HEAD` (preserve the in-flight
commit) and NEVER the plain `git checkout ctx.branch_dev` that orphans it, and flow
came_from_detached_checkout=True into the stage cycle; on a normal branch the path is unchanged.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import patch


def _ctx(tmp_path: pathlib.Path) -> SimpleNamespace:
    drive = tmp_path / "drive"
    drive.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        repo_dir=tmp_path, drive_root=drive, branch_dev="ouroboros",
        last_push_succeeded=True, pending_events=[], pending_restart_reason=None,
        pending_restart_policy=None, current_task_type="task",
    )


def _drive_commit(tmp_path: pathlib.Path, head_branch: str):
    """Run _repo_commit_push with run_cmd mocked; return (checkout_cmds, stage_kwargs)."""
    from ouroboros.tools import git as git_module

    ctx = _ctx(tmp_path)
    captured: list[str] = []
    stage_kwargs: dict = {}

    def fake_run(cmd, cwd=None, **_):
        captured.append(" ".join(map(str, cmd)))
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return head_branch + "\n"   # "HEAD" => detached; a branch name => on-branch
        if cmd[:2] == ["git", "rev-parse"]:
            return "deadbeefdeadbeef\n"
        if cmd[:2] == ["git", "status"]:
            return " M f.txt\n"          # non-empty tree -> proceeds past the GIT_NO_CHANGES guard
        return ""

    def fake_stage_cycle(ctx, msg, start, **kw):
        stage_kwargs.update(kw)
        return {"status": "passed", "message": "",
                "pre_fingerprint": {"fingerprint": "x"},
                "post_fingerprint": {"fingerprint": "x"}}

    with patch.object(git_module, "run_cmd", side_effect=fake_run), \
         patch.object(git_module, "_run_reviewed_stage_cycle", side_effect=fake_stage_cycle), \
         patch.object(git_module, "_acquire_git_lock", return_value=None), \
         patch.object(git_module, "_release_git_lock"), \
         patch.object(git_module, "_record_commit_attempt"), \
         patch.object(git_module, "_post_commit_result"), \
         patch.object(git_module, "_auto_tag_on_version_bump", return_value=""), \
         patch.object(git_module, "_auto_push", return_value="ok"):
        try:
            git_module._repo_commit_push(ctx, "test commit")
        except Exception:
            pass  # downstream (review/push) is out of scope; we assert the checkout decision
    checkout_cmds = [c for c in captured if c.startswith("git checkout")]
    return checkout_cmds, stage_kwargs


def test_detached_head_force_moves_with_B_and_never_orphans(tmp_path):
    checkout_cmds, stage_kwargs = _drive_commit(tmp_path, head_branch="HEAD")  # detached
    assert "git checkout -B ouroboros HEAD" in checkout_cmds, \
        f"detached HEAD must force-move the branch with -B; got {checkout_cmds}"
    assert "git checkout ouroboros" not in checkout_cmds, \
        f"the orphaning plain checkout must NOT be issued on a detached HEAD; got {checkout_cmds}"
    assert stage_kwargs.get("came_from_detached_checkout") is True, \
        "the detached-reconcile flag must flow into the stage cycle (for the GIT_LOST diagnostic)"


def test_on_branch_uses_plain_checkout_no_force_move(tmp_path):
    checkout_cmds, stage_kwargs = _drive_commit(tmp_path, head_branch="ouroboros")  # on-branch
    assert "git checkout ouroboros" in checkout_cmds, \
        f"on-branch must use the plain checkout (byte-identical to pre-fix); got {checkout_cmds}"
    assert not any("-B" in c for c in checkout_cmds), \
        f"on-branch must NOT force-move; got {checkout_cmds}"
    assert not stage_kwargs.get("came_from_detached_checkout"), \
        "on-branch must NOT set the detached-reconcile flag"
