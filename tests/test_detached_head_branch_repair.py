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


@pytest.fixture(autouse=True)
def _own_tracked_registry(monkeypatch):
    """Each test gets its own tracked registry (#2841 routes these Git reads through tracked custody).

    The runners keep naming a child whose tree they could not measure; in the worker's shared registry such a
    record outlived its test and left later readers in the worker (a scope teardown, an owned-work stop) a child
    that never settles: tests/test_task_scope_lifetime.py went red beside this module."""
    from ouroboros.tools import shell_process

    monkeypatch.setattr(shell_process, "_active_subprocesses", set())


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


def test_launcher_managed_identity_rejects_copied_or_mismatched_markers(monkeypatch, tmp_path):
    repo, _first, _origin = _release_checkout(tmp_path)
    _bind(monkeypatch, repo, tmp_path)

    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_MANAGED_REPO_DIR", str(tmp_path / "another-repo"))
    assert git_ops._is_launcher_managed_repo() is False

    monkeypatch.setenv("OUROBOROS_MANAGED_REPO_DIR", str(repo))
    assert git_ops._is_launcher_managed_repo() is True

    monkeypatch.delenv("OUROBOROS_MANAGED_REPO_DIR")
    assert git_ops._is_launcher_managed_repo() is False
    (repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text("{}", encoding="utf-8")
    assert git_ops._is_launcher_managed_repo() is False
    legacy = {"schema_version": 1, "app_version": "7.4.0", "managed_local_branch": "ouroboros"}  # no path before v7.4.1
    (repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(json.dumps(legacy), encoding="utf-8")
    assert git_ops._is_launcher_managed_repo() is True  # a launcher that passes no path: its own record
    monkeypatch.setenv("OUROBOROS_MANAGED_REPO_DIR", str(tmp_path / "another-repo"))
    assert git_ops._is_launcher_managed_repo() is False  # a launcher that names a path is judged by it
    monkeypatch.delenv("OUROBOROS_MANAGED_REPO_DIR")
    (repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(
        json.dumps({**legacy, "managed_repo_dir": str(tmp_path / "another-repo")}), encoding="utf-8")
    assert git_ops._is_launcher_managed_repo() is False  # a record copied from another checkout
    (repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(
        json.dumps({"managed_repo_dir": str(repo.resolve())}), encoding="utf-8")
    assert git_ops._is_launcher_managed_repo() is True

    monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER")
    assert git_ops._is_launcher_managed_repo() is False


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
    assert "git switch" in caplog.text
    assert "checkout -B" not in caplog.text
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
    assert "git switch" in caplog.text
    assert "checkout -B" not in caplog.text
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
    assert "git switch" in caplog.text
    assert "checkout -B" not in caplog.text
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
    assert "git switch" in caplog.text
    assert "checkout -B" not in caplog.text
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
    assert "git switch" in caplog.text
    assert "checkout -B" not in caplog.text
    assert _repair_rows(tmp_path) == []


def test_branch_created_between_check_and_creation_is_never_reset(monkeypatch, tmp_path, caplog):
    """The race the non-forcing creation closes: a branch that appears after
    the existence check is refused by `git branch <name> <sha>`, never reset."""
    repo, first, _origin = _release_checkout(tmp_path, unpushed_second_commit=True)
    second = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", first)
    _git(repo, "branch", "-q", "-D", "ouroboros")  # the check must see the name absent
    _bind(monkeypatch, repo, tmp_path)

    real_attach = git_ops._attach_detached_head
    hook_fired = []

    def racing(branch, head):
        hook_fired.append(True)
        _git(repo, "branch", "ouroboros", second)  # a concurrent creator wins the name
        return real_attach(branch, head)

    monkeypatch.setattr(git_ops, "_attach_detached_head", racing)
    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert hook_fired == [True]  # the creation command actually ran: the race was reached
    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _git(repo, "rev-parse", "refs/heads/ouroboros") == second  # never reset to HEAD
    assert "git switch" in caplog.text
    assert "checkout -B" not in caplog.text
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
    assert "git switch" in caplog.text
    assert "checkout -B" not in caplog.text
    assert _repair_rows(tmp_path) == []


def test_head_moved_after_creation_leaves_branch_and_detached_head(monkeypatch, tmp_path, caplog):
    """A HEAD that moved after validation aborts BOTH transactional writes."""
    repo, first, _origin = _release_checkout(tmp_path, unpushed_second_commit=True)
    _git(repo, "checkout", "-q", "--detach", first)
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    real_attach = git_ops._attach_detached_head
    hook_fired = []

    def moving_head(branch, head):
        hook_fired.append(True)
        (repo / "value.txt").write_text("three\n", encoding="utf-8")
        _git(repo, "commit", "-qam", "three")  # a concurrent writer moves HEAD
        hook_fired.append(_git(repo, "rev-parse", "HEAD"))
        return real_attach(branch, head)

    monkeypatch.setattr(git_ops, "_attach_detached_head", moving_head)
    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert len(hook_fired) == 2  # the creation ran and the mover's sha was captured
    third = hook_fired[1]
    assert third != first
    assert _rev_soft(repo, "rev-parse", "--verify", "refs/heads/ouroboros") == ""
    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # HEAD left detached
    assert _git(repo, "rev-parse", "HEAD") == third  # where its mover put it
    assert "git switch" in caplog.text
    assert _repair_rows(tmp_path) == []


def test_tree_dirtied_after_creation_leaves_head_detached(monkeypatch, tmp_path, caplog):
    """Late untracked work aborts the transaction without even creating a branch."""
    repo, first, _origin = _release_checkout(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")
    _bind(monkeypatch, repo, tmp_path)

    real_attach = git_ops._attach_detached_head
    hook_fired = []

    def dirtying(branch, head):
        hook_fired.append(True)
        (repo / "late.txt").write_text("dirt\n", encoding="utf-8")
        return real_attach(branch, head)

    monkeypatch.setattr(git_ops, "_attach_detached_head", dirtying)
    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert hook_fired == [True]
    assert _rev_soft(repo, "rev-parse", "--verify", "refs/heads/ouroboros") == ""
    assert (repo / "late.txt").read_text() == "dirt\n"
    assert not (repo / ".git" / "HEAD.lock").exists()
    assert not (repo / ".git" / "refs" / "heads" / "ouroboros.lock").exists()
    assert _rev_soft(repo, "symbolic-ref", "-q", "--short", "HEAD") == ""  # still detached
    assert _git(repo, "rev-parse", "HEAD") == first
    assert "git switch" in caplog.text
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
    assert "git switch -c <new-branch> HEAD" in exc.value.text
    assert "git switch <existing-branch>" in exc.value.text
    assert "checkout -B" not in exc.value.text


def _detach_missing(repo):
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    _git(repo, "branch", "-q", "-D", "ouroboros")


def _preserved(repo):
    """Addressed state, including raw index bytes (no status-induced refresh)."""
    import pathlib

    directory = pathlib.Path(_git(repo, "rev-parse", "--absolute-git-dir"))
    index = directory / "index"
    files = {str(p.relative_to(repo)): p.read_bytes() for p in repo.rglob("*")
             if p.is_file() and ".git" not in p.relative_to(repo).parts}
    return ((directory / "HEAD").read_bytes(), index.read_bytes() if index.exists() else None,
            _git(repo, "for-each-ref", "--format=%(refname) %(objectname) %(symref)"), files)


@pytest.mark.parametrize("marker", [
    "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-apply", "rebase-merge",
    "sequencer", "BISECT_START", "ouroboros-bootstrap-pending", "ouroboros-update-intent.json",
])
def test_checkout_operation_markers_refuse_without_altering_state(monkeypatch, tmp_path, marker):
    repo, first, _ = _release_checkout(tmp_path)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    path = repo / ".git" / marker
    if marker in {"rebase-apply", "rebase-merge", "sequencer"}:
        path.mkdir()
    else:
        path.write_text(first + "\n")
    before = _preserved(repo)

    git_ops.ensure_repo_present()

    assert _preserved(repo) == before
    assert path.exists()
    assert _repair_rows(tmp_path) == []


def test_real_bisect_clean_detached_head_is_not_repaired(monkeypatch, tmp_path):
    repo, first, _ = _release_checkout(tmp_path, unpushed_second_commit=True)
    for text in ("three\n", "four\n"):
        (repo / "value.txt").write_text(text)
        _git(repo, "commit", "-qam", text.strip())
    _git(repo, "push", "-q", "origin", "ouroboros")
    _git(repo, "branch", "-m", "topic")
    _git(repo, "bisect", "start", "HEAD", first)
    assert _rev_soft(repo, "symbolic-ref", "-q", "HEAD") == ""
    assert (repo / ".git" / "BISECT_START").exists()
    _bind(monkeypatch, repo, tmp_path)
    before = _preserved(repo)
    markers = {p.name: p.read_bytes() for p in (repo / ".git").glob("BISECT_*") if p.is_file()}

    git_ops.ensure_repo_present()

    assert _preserved(repo) == before
    assert {p.name: p.read_bytes() for p in (repo / ".git").glob("BISECT_*") if p.is_file()} == markers


@pytest.mark.parametrize("kind", ["tracked", "staged", "untracked", "deleted"])
def test_refusal_preserves_index_worktree_and_untracked_bytes(monkeypatch, tmp_path, kind):
    repo, _, _ = _release_checkout(tmp_path)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    target = repo / ("new.txt" if kind == "untracked" else "value.txt")
    if kind == "deleted":
        target.unlink()
    else:
        target.write_bytes(b"owner work\x00\xff\n")
        if kind == "staged":
            _git(repo, "add", "value.txt")
            target.write_bytes(b"later work\n")
    before = _preserved(repo)

    git_ops.ensure_repo_present()

    assert _preserved(repo) == before
    assert _repair_rows(tmp_path) == []


@pytest.mark.parametrize("managed", ["marker", "metadata"])
def test_managed_entry_never_calls_source_repair(monkeypatch, tmp_path, managed):
    repo, _, _ = _release_checkout(tmp_path)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    # The shared managed identity requires the launcher marker plus either an
    # exact repo-path witness or the managed metadata witness.  A copied marker
    # and metadata-only rows are negative cases covered separately above.
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    if managed == "marker":
        monkeypatch.setenv("OUROBOROS_MANAGED_REPO_DIR", str(repo))
    else:
        (repo / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(
            json.dumps({"source_sha": "known", "managed_repo_dir": str(repo.resolve())}))
    monkeypatch.setattr(git_ops, "_repair_detached_head_branch", lambda: pytest.fail("source repair on managed entry"))
    before = _preserved(repo)

    git_ops.ensure_repo_present()

    assert _preserved(repo) == before


@pytest.mark.parametrize("phase", ["armed", "stuck", "switched", "abandoned"])
def test_adoption_handoff_keeps_its_bytes_and_refs(monkeypatch, tmp_path, phase):
    repo, _, _ = _release_checkout(tmp_path)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    handoff = tmp_path / "data" / "state" / "body_adoption" / "handoff.json"
    handoff.parent.mkdir(parents=True)
    content = json.dumps({"id": "owned-transition", "repo_dir": str(repo), "phase": phase})
    handoff.write_text(content)
    before = _preserved(repo)

    git_ops.ensure_repo_present()

    assert _preserved(repo) == before
    assert handoff.read_text() == content


def test_named_head_chosen_at_same_oid_before_prepare_wins(monkeypatch, tmp_path):
    repo, first, _ = _release_checkout(tmp_path)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    real_attach = git_ops._attach_detached_head
    chosen = []

    def choose(branch, head):
        _git(repo, "switch", "-c", "chosen", first)
        chosen.append(_preserved(repo))
        return real_attach(branch, head)

    monkeypatch.setattr(git_ops, "_attach_detached_head", choose)
    git_ops.ensure_repo_present()

    assert len(chosen) == 1
    assert _preserved(repo) == chosen[0]
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "chosen"
    assert not (repo / ".git" / "HEAD.lock").exists()
    assert not (repo / ".git" / "refs" / "heads" / "ouroboros.lock").exists()
    assert _repair_rows(tmp_path) == []


def test_prepared_git_locks_protect_head_and_target_branch(monkeypatch, tmp_path):
    import subprocess

    repo, first, _ = _release_checkout(tmp_path)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    real_blocked = git_ops._detached_repair_blocked
    probes = []

    def while_locked():
        if (repo / ".git" / "HEAD.lock").exists():
            assert (repo / ".git" / "refs" / "heads" / "ouroboros.lock").exists()
            for args in (("symbolic-ref", "HEAD", "refs/heads/chosen"),
                         ("update-ref", "refs/heads/ouroboros", first)):
                result = subprocess.run(["git", *args], cwd=repo, capture_output=True)
                probes.append(result.returncode)
                assert result.returncode != 0, result.stdout + result.stderr
        return real_blocked()

    monkeypatch.setattr(git_ops, "_detached_repair_blocked", while_locked)
    git_ops.ensure_repo_present()

    assert len(probes) == 2
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "ouroboros"
    assert _git(repo, "rev-parse", "HEAD") == first
    assert len(_repair_rows(tmp_path)) == 1


@pytest.mark.parametrize("change", ["handoff", "marker"])
def test_transition_arriving_after_prepare_aborts_both_writes(monkeypatch, tmp_path, change):
    repo, _, _ = _release_checkout(tmp_path)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    before = _preserved(repo)
    real_blocked = git_ops._detached_repair_blocked
    injected = []

    def arriving():
        if (repo / ".git" / "HEAD.lock").exists():
            if change == "marker":
                path = repo / ".git" / "rebase-merge"
                path.mkdir()
            else:
                path = tmp_path / "data" / "state" / "body_adoption" / "handoff.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"repo_dir": str(repo), "phase": "switching"}))
            injected.append(path)
        return real_blocked()

    monkeypatch.setattr(git_ops, "_detached_repair_blocked", arriving)
    git_ops.ensure_repo_present()

    assert len(injected) == 1 and injected[0].exists()
    assert _preserved(repo) == before
    assert not (repo / ".git" / "HEAD.lock").exists()
    assert _repair_rows(tmp_path) == []


def test_old_git_without_symref_update_does_not_fall_back(monkeypatch, tmp_path, caplog):
    import sys

    repo, _, _ = _release_checkout(tmp_path)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    before = _preserved(repo)
    popen = git_ops.subprocess.Popen
    attempted = []

    def old_git(argv, **kwargs):
        if argv[1:4] == ["update-ref", "--no-deref", "--stdin"]:
            attempted.append(argv)
            return popen([sys.executable, "-c", "import sys; print('start: ok', flush=True); print('fatal: unknown command: symref-update HEAD', flush=True); sys.exit(128)"], **kwargs)
        return popen(argv, **kwargs)

    monkeypatch.setattr(git_ops.subprocess, "Popen", old_git)
    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()

    assert len(attempted) == 1
    assert _preserved(repo) == before
    assert "symref-update" in caplog.text and "git switch" in caplog.text
    assert "checkout -B" not in caplog.text
    assert _repair_rows(tmp_path) == []


def test_missing_commit_ack_is_unknown_not_rollback_or_retry(monkeypatch, tmp_path, caplog):
    repo, first, _ = _release_checkout(tmp_path)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    popen = git_ops.subprocess.Popen
    commits = []

    def lost_ack(argv, **kwargs):
        proc = popen(argv, **kwargs)
        if argv[1:4] == ["update-ref", "--no-deref", "--stdin"]:
            communicate = proc.communicate

            def without_ack(input=None, **kw):
                result = communicate(input=input, **kw)
                if input == b"commit\n":
                    commits.append(True)
                    return b"", result[1]
                return result

            proc.communicate = without_ack
        return proc

    monkeypatch.setattr(git_ops.subprocess, "Popen", lost_ack)
    with caplog.at_level(logging.WARNING, logger="supervisor.git_ops"):
        git_ops.ensure_repo_present()
        git_ops.ensure_repo_present()  # named HEAD now; no second transaction

    assert commits == [True]
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "ouroboros"
    assert _git(repo, "rev-parse", "HEAD") == first
    assert "unconfirmed" in caplog.text
    assert _repair_rows(tmp_path) == []


@pytest.mark.parametrize("consumer", ["pull", "adoption"])
def test_source_bootstrap_reaches_real_pull_and_adoption_consumers(monkeypatch, tmp_path, consumer):
    import server
    from ouroboros import body_adoption
    from ouroboros.tools.git_vcs_ops import _ff_pull
    from tests.body_candidate_support import candidate_commit, isolate, make_serving

    repo = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "ouroboros")
    first = _git(repo, "rev-parse", "HEAD")
    candidate = tmp_path / "candidate"
    _git(tmp_path, "clone", "-q", str(repo), str(candidate))
    _git(candidate, "config", "user.name", "Test")
    _git(candidate, "config", "user.email", "test@ouroboros")
    cand = candidate_commit(candidate)
    _git(repo, "fetch", "-q", str(candidate), cand)
    _detach_missing(repo)
    _bind(monkeypatch, repo, tmp_path)
    monkeypatch.setattr(server, "REPO_DIR", repo)
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", False)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED_REPO_DIR", "")
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *a, **k: None)
    monkeypatch.setattr(git_ops, "sync_runtime_dependencies", lambda **k: (True, "test: no pip"))
    monkeypatch.setattr(git_ops, "import_test", lambda: {"ok": True})
    monkeypatch.setattr(git_ops, "_ensure_local_version_tag", lambda: None)
    before = _preserved(repo)

    assert server._bootstrap_supervisor_repo({}, git_ops) == (True, "OK: local-dev bootstrap")
    after = _preserved(repo)
    assert after[1] == before[1] and after[3] == before[3]  # raw index and owner files
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "ouroboros"
    assert _git(repo, "rev-parse", "HEAD") == first
    if consumer == "pull":
        _git(candidate, "remote", "add", "release", str(origin))
        _git(candidate, "push", "-q", "release", "ouroboros")
        result = _ff_pull(repo)
        assert "Pulled origin/ouroboros" in result, result
        assert _git(repo, "rev-parse", "HEAD") == cand
    else:
        handoff = body_adoption.authorize_row(data, {"repo_dir": str(repo), "path": str(candidate),
                                                   "base_sha": first}, cand, reason="fixture", runtime_mode="cyber_pro")
        assert handoff["branch"] == "ouroboros" and handoff["old"] == first and handoff["cand"] == cand
        assert body_adoption.read(data)["phase"] == "authorized"
        assert _git(repo, "rev-parse", "HEAD") == first  # authorization does not adopt
