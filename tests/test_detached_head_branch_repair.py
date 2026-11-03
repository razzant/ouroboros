"""The startup repair that attaches a detached HEAD to the missing local dev branch.

Cycle adoption and the managed update both move the local dev branch, so a
checkout left on a detached HEAD (a source install checked out at a tag or
SHA) could never accept a cycle or pull an update, and the refusals did not
name the repair. ``ensure_repo_present`` now attaches the branch only at a
clean HEAD a remote release line contains; every other case keeps the tree
and says the command. The adoption refusal names it too.
"""

from __future__ import annotations

import json
import logging

import pytest

import supervisor.git_ops as git_ops


def _git(repo, *args):
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.strip()


def _rev_soft(repo, *args):
    """stdout of a git query that may legitimately fail (a missing ref): "" then."""
    import subprocess

    done = subprocess.run(
        ["git", *args], cwd=str(repo), check=False, capture_output=True, text=True
    )
    return done.stdout.strip() if done.returncode == 0 else ""


def _release_checkout(tmp_path, *, unpushed_second_commit=False):
    """A clone-shaped repo whose ``origin`` holds branch ``ouroboros`` at the first commit.

    ``unpushed_second_commit`` leaves a second local commit the remote does not
    have — the off-the-release-line case.
    """
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "-q", "--bare")
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
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "ouroboros")
    if unpushed_second_commit:
        (repo / "value.txt").write_text("two\n", encoding="utf-8")
        _git(repo, "commit", "-qam", "two")
    return repo, first, origin


def _bind(monkeypatch, repo, tmp_path):
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER", raising=False)


def _repair_rows(tmp_path):
    log_file = tmp_path / "data" / "logs" / "supervisor.jsonl"
    if not log_file.exists():
        return []
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line]


def test_detached_head_on_release_line_gets_branch_at_head(monkeypatch, tmp_path):
    """The live defect: detached at a commit origin/ouroboros contains, no local
    branch. Startup creates the branch at HEAD, so adoption is no longer blocked."""
    repo, first, _origin = _release_checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    git_ops.ensure_repo_present()

    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    assert branch == "ouroboros"
    assert _git(repo, "rev-parse", "HEAD") == first  # attached exactly where it sat
    rows = [r for r in _repair_rows(tmp_path) if r.get("type") == "detached_head_branch_repaired"]
    assert len(rows) == 1
    assert rows[0]["branch"] == "ouroboros"
    assert rows[0]["head"] == first
    assert rows[0]["on_line"] == "refs/remotes/origin/ouroboros"


def test_detached_head_off_release_line_is_left_alone_with_the_command(monkeypatch, tmp_path, caplog):
    """A HEAD no remote release line contains is the owner's call to make: the
    tree is untouched and the log names the exact command."""
    repo, _first, _origin = _release_checkout(tmp_path, unpushed_second_commit=True)
    second = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _rev_soft(repo, "rev-parse", "--verify", "refs/heads/ouroboros") == ""  # no branch created
    assert _git(repo, "rev-parse", "HEAD") == second
    assert "git checkout -B ouroboros" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_named_branch_checkout_is_unchanged(monkeypatch, tmp_path, caplog):
    """A checkout already on the branch behaves exactly as before: no repair row."""
    repo, first, _origin = _release_checkout(tmp_path)
    _bind(monkeypatch, repo, tmp_path)

    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "ouroboros"
    assert _git(repo, "rev-parse", "HEAD") == first
    assert _repair_rows(tmp_path) == []
    assert "detached" not in caplog.text.lower()


def test_detached_head_with_branch_elsewhere_keeps_both(monkeypatch, tmp_path, caplog):
    """The name already exists at another commit: startup never force-moves it."""
    repo, _first, _origin = _release_checkout(tmp_path, unpushed_second_commit=True)
    second = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", "HEAD~1")  # detached at first, branch stays at second
    _bind(monkeypatch, repo, tmp_path)

    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _git(repo, "rev-parse", "refs/heads/ouroboros") == second  # branch untouched
    assert "git checkout -B ouroboros" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_detached_head_dirty_tree_is_left_alone(monkeypatch, tmp_path, caplog):
    """Uncommitted owner work is never built on silently: no branch, and the log
    names the command for after the owner settles the tree."""
    repo, first, _origin = _release_checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    (repo / "value.txt").write_text("dirty\n", encoding="utf-8")
    _bind(monkeypatch, repo, tmp_path)

    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _rev_soft(repo, "rev-parse", "--verify", "refs/heads/ouroboros") == ""
    assert _git(repo, "rev-parse", "HEAD") == first
    assert "git checkout -B ouroboros" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_untracked_file_is_left_alone(monkeypatch, tmp_path, caplog):
    """Untracked files count as a non-clean tree: no branch is created over
    work the owner has not settled, and the log names the command."""
    repo, first, _origin = _release_checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert "??" in _git(repo, "status", "--porcelain")  # porcelain sees untracked files
    _bind(monkeypatch, repo, tmp_path)

    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _rev_soft(repo, "rev-parse", "--verify", "refs/heads/ouroboros") == ""
    assert _git(repo, "rev-parse", "HEAD") == first
    assert "git checkout -B ouroboros" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_adoption_transition_blocks_repair(monkeypatch, tmp_path):
    """An open adoption transition owns this tree's refs: even the safe-case
    topology is left untouched, and no repair row is written."""
    import pathlib

    repo, first, _origin = _release_checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    helper = pathlib.Path(tmp_path) / "data" / "state" / "body_adoption"
    helper.mkdir(parents=True)
    (helper / "handoff.json").write_text(json.dumps({
        "id": "adopt-test", "phase": "switching", "repo_dir": str(repo),
    }), encoding="utf-8")

    git_ops.ensure_repo_present()

    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _rev_soft(repo, "rev-parse", "--verify", "refs/heads/ouroboros") == ""
    assert _repair_rows(tmp_path) == []  # the transition owns the tree; nothing recorded


def test_git_failure_means_no_repair_and_boot_continues(monkeypatch, tmp_path, caplog):
    """A Git probe that errors (a timeout returns the same nonzero shape)
    fails closed: nothing is touched, the command is named, and boot continues."""
    repo, _first, _origin = _release_checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    def broken(cmd, **_kwargs):
        return 128, "", "fatal: not a git repository (or any parent up to mount point)"

    monkeypatch.setattr(git_ops, "rescue_git_capture", broken)
    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()  # must not raise

    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _rev_soft(repo, "rev-parse", "--verify", "refs/heads/ouroboros") == ""
    assert "git checkout -B ouroboros" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_branch_created_between_check_and_creation_is_never_reset(monkeypatch, tmp_path, caplog):
    """The race the non-forcing creation closes: a branch that appears after
    the existence check is refused by `git branch <name> <sha>`, never reset."""
    repo, first, _origin = _release_checkout(tmp_path, unpushed_second_commit=True)
    second = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", first)
    _git(repo, "branch", "-q", "-D", "ouroboros")  # the check must see the name absent
    _bind(monkeypatch, repo, tmp_path)

    real_capture = git_ops.rescue_git_capture
    hook_fired = []

    def racing(cmd, **kwargs):
        if cmd[1:3] == ["branch", "ouroboros"]:  # the creation this test races against
            hook_fired.append(True)
            _git(repo, "branch", "ouroboros", second)  # a concurrent creator wins the name
        return real_capture(cmd, **kwargs)

    monkeypatch.setattr(git_ops, "rescue_git_capture", racing)
    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert hook_fired == [True]  # the creation command actually ran: the race was reached
    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _git(repo, "rev-parse", "refs/heads/ouroboros") == second  # never reset to HEAD
    assert "git checkout -B ouroboros" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_unreadable_adoption_handoff_blocks_repair(monkeypatch, tmp_path, caplog):
    """A handoff that exists but will not parse may be a transition this boot
    cannot see: no repair, the command named, nothing recorded."""
    import pathlib

    repo, first, _origin = _release_checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    helper = pathlib.Path(tmp_path) / "data" / "state" / "body_adoption"
    helper.mkdir(parents=True)
    (helper / "handoff.json").write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _rev_soft(repo, "rev-parse", "--verify", "refs/heads/ouroboros") == ""
    assert "could not be read" in caplog.text
    assert "git checkout -B ouroboros" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_head_moved_after_creation_leaves_branch_and_detached_head(monkeypatch, tmp_path, caplog):
    """Placement never depends on HEAD staying put: the branch is created at
    the validated commit; a HEAD that moved meanwhile keeps the branch and
    stays detached where its mover put it."""
    repo, first, _origin = _release_checkout(tmp_path, unpushed_second_commit=True)
    _git(repo, "checkout", "-q", "--detach", first)
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    real_capture = git_ops.rescue_git_capture
    hook_fired = []

    def moving_head(cmd, **kwargs):
        if cmd[1:3] == ["branch", "ouroboros"]:  # after the checks, before the attach
            hook_fired.append(True)
            (repo / "value.txt").write_text("three\n", encoding="utf-8")
            _git(repo, "commit", "-qam", "three")  # a concurrent writer moves HEAD
            hook_fired.append(_git(repo, "rev-parse", "HEAD"))
        return real_capture(cmd, **kwargs)

    monkeypatch.setattr(git_ops, "rescue_git_capture", moving_head)
    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert len(hook_fired) == 2  # the creation ran and the mover's sha was captured
    third = hook_fired[1]
    assert third != first
    assert _git(repo, "rev-parse", "refs/heads/ouroboros") == first  # branch at the validated commit
    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # HEAD left detached
    assert _git(repo, "rev-parse", "HEAD") == third  # where its mover put it
    assert "git checkout ouroboros" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_tree_dirtied_after_creation_leaves_head_detached(monkeypatch, tmp_path, caplog):
    """The attach also requires a still-clean tree: dirt that appears after the
    branch was created leaves the branch at the validated commit and HEAD
    detached, never attaching over unsettled work."""
    repo, first, _origin = _release_checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    real_capture = git_ops.rescue_git_capture
    hook_fired = []

    def dirtying(cmd, **kwargs):
        if cmd[1:3] == ["branch", "ouroboros"]:
            hook_fired.append(True)
            (repo / "late.txt").write_text("dirt\n", encoding="utf-8")  # after the clean check
        return real_capture(cmd, **kwargs)

    monkeypatch.setattr(git_ops, "rescue_git_capture", dirtying)
    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert hook_fired == [True]
    assert _git(repo, "rev-parse", "refs/heads/ouroboros") == first
    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _git(repo, "rev-parse", "HEAD") == first
    assert "git checkout ouroboros" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_local_dev_boot_path_runs_the_repair(monkeypatch, tmp_path):
    """The real boot caller: ``_bootstrap_supervisor_repo`` reaches the repair
    through ``ensure_repo_present`` (server.py:520) before the managed/local-dev
    fork; on the local-dev path the branch is attached and boot reports OK."""
    import server
    import supervisor.git_ops as real_git_ops

    repo, first, _origin = _release_checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    monkeypatch.setattr(server, "REPO_DIR", repo)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", False)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED_REPO_DIR", "")
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *a, **k: None)
    monkeypatch.setattr(real_git_ops, "sync_runtime_dependencies", lambda **k: (True, "ok"))
    monkeypatch.setattr(real_git_ops, "import_test", lambda: {"ok": True})

    ok, message = server._bootstrap_supervisor_repo({}, real_git_ops)

    assert ok is True
    assert "local-dev" in message
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "ouroboros"
    assert _git(repo, "rev-parse", "HEAD") == first
    rows = [r for r in _repair_rows(tmp_path) if r.get("type") == "detached_head_branch_repaired"]
    assert len(rows) == 1


def test_adoption_detached_refusal_names_the_command(tmp_path):
    """The refusal the owner actually met says what to run, not just what failed."""
    from ouroboros.body_adoption import AdoptionRefused, authorize_row

    cand_repo = tmp_path / "candidate"
    cand_repo.mkdir()
    _git(cand_repo, "init", "-q")
    _git(cand_repo, "config", "user.name", "Test")
    _git(cand_repo, "config", "user.email", "test@ouroboros")
    (cand_repo / "value.txt").write_text("one\n", encoding="utf-8")
    _git(cand_repo, "add", "value.txt")
    _git(cand_repo, "commit", "-qm", "one")
    base = _git(cand_repo, "rev-parse", "HEAD")
    _git(cand_repo, "branch", "-M", "ouroboros")
    (cand_repo / "value.txt").write_text("two\n", encoding="utf-8")
    _git(cand_repo, "commit", "-qam", "two")
    cand = _git(cand_repo, "rev-parse", "HEAD")

    serving = tmp_path / "serving"
    _git(tmp_path, "clone", "-q", str(cand_repo), "serving")
    _git(serving, "checkout", "-q", "--detach", base)  # the defect topology
    _git(serving, "branch", "-q", "-D", "ouroboros")

    row = {"repo_dir": str(serving), "path": str(cand_repo), "base_sha": base}
    with pytest.raises(AdoptionRefused) as exc:
        authorize_row(tmp_path / "data", row, cand, reason="test")

    assert exc.value.code == "ADOPTION_DETACHED"
    assert "git checkout -B ouroboros" in exc.value.text
