"""Q1=C: a clean auto-update with a DIRTY worktree never commits local work into
history — it is stashed before the apply and restored as uncommitted content
(boot finalize on success, rollback path on failure), with a disclosed, kept
stash entry when an automatic restore would conflict."""

import subprocess

import supervisor.git_ops as git_ops
import supervisor.update_merge as update_merge


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "a.txt").write_text("base\n")
    (repo / "BIBLE.md").write_text("constitution\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    head = _git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip()
    return repo, head


def _point_at(monkeypatch, tmp_path, repo, head):
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", head)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda branch=None: ("", "", "remote-sim"))
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: (
            "remote-sim",
            _git(repo, "rev-parse", "remote-sim").stdout.strip(),
            "",
        ),
    )
    (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)


def _diverged_clean_repo(tmp_path, monkeypatch):
    """Official target and local commit touch different files; dirty edits on top."""
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "remote.txt").write_text("official\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official")
    _git(repo, "checkout", "-q", head)
    (repo / "local.txt").write_text("local commit\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "local")
    _point_at(monkeypatch, tmp_path, repo, head)
    return repo, head


def test_clean_dirty_plan_builds_history_without_local_work(tmp_path, monkeypatch):
    repo, head = _diverged_clean_repo(tmp_path, monkeypatch)
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    target = _git(repo, "rev-parse", "remote-sim").stdout.strip()
    (repo / "a.txt").write_text("dirty local edit\n")
    (repo / "untracked.txt").write_text("scratch\n")

    plan = update_merge.plan_managed_update_merge(build=True)

    assert plan["kind"] == "clean", plan
    merge_commit = plan["merge_commit"]
    assert merge_commit
    parents = _git(repo, "log", "-1", "--format=%P", merge_commit).stdout.split()
    assert parents == [base, target], parents
    # The committed tree carries NO local dirty work.
    committed_a = _git(repo, "show", f"{merge_commit}:a.txt").stdout
    assert committed_a == "base\n"
    ls = _git(repo, "ls-tree", "-r", "--name-only", merge_commit).stdout
    assert "untracked.txt" not in ls
    # The worktree itself was never touched by planning.
    assert (repo / "a.txt").read_text() == "dirty local edit\n"


def test_clean_dirty_fast_forward_plan_lands_official_history(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "remote.txt").write_text("official\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official")
    _git(repo, "checkout", "-q", head)
    _point_at(monkeypatch, tmp_path, repo, head)
    target = _git(repo, "rev-parse", "remote-sim").stdout.strip()
    (repo / "a.txt").write_text("dirty local edit\n")

    plan = update_merge.plan_managed_update_merge(build=True)

    assert plan["kind"] == "clean", plan
    assert plan["merge_commit"] == target


def test_stash_roundtrip_restores_local_work(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    (repo / "a.txt").write_text("dirty\n")
    (repo / "untracked.txt").write_text("scratch\n")

    status, stash_sha, error = update_merge.stash_local_changes_for_update("t1")
    assert status == "ok" and stash_sha, error
    assert not _git(repo, "status", "--porcelain").stdout.strip()

    result = update_merge.restore_update_stash(stash_sha, context="test")
    assert result.status == "restored", result
    assert (repo / "a.txt").read_text() == "dirty\n"
    assert (repo / "untracked.txt").read_text() == "scratch\n"
    assert stash_sha in _git(repo, "stash", "list", "--format=%H").stdout


def test_stash_on_clean_tree_is_a_noop(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)

    status, stash_sha, error = update_merge.stash_local_changes_for_update("t2")
    assert status == "ok" and stash_sha == "", (stash_sha, error)
    assert update_merge.restore_update_stash("", context="test").status == "not_needed"


def test_conflicting_restore_keeps_stash_and_discloses(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    (repo / "a.txt").write_text("dirty conflicting edit\n")
    status, stash_sha, _error = update_merge.stash_local_changes_for_update("t3")
    assert status == "ok" and stash_sha
    # The updated tree rewrites the same lines the stash carries.
    (repo / "a.txt").write_text("official rewrite\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official rewrite")

    result = update_merge.restore_update_stash(stash_sha, context="test")

    assert result.status == "preserved"
    assert stash_sha[:12] in result.note and "git stash apply" in result.note
    # Worktree left clean; the stash entry survives for manual recovery.
    assert not _git(repo, "status", "--porcelain").stdout.strip()
    assert stash_sha in _git(repo, "stash", "list", "--format=%H").stdout


def test_boot_finalize_restores_stash_and_is_replay_safe(tmp_path, monkeypatch):
    repo, head = _diverged_clean_repo(tmp_path, monkeypatch)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    target = _git(repo, "rev-parse", "remote-sim").stdout.strip()
    (repo / "a.txt").write_text("owner work in flight\n")
    status, stash_sha, _error = update_merge.stash_local_changes_for_update("boot1")
    assert status == "ok" and stash_sha
    _git(repo, "merge", "-q", "--no-ff", "-m", "landed update", target)
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke", "pre_update_sha": pre,
        "pre_update_branch": head, "target_sha": target,
        "merge_commit": _git(repo, "rev-parse", "HEAD").stdout.strip(),
        "pre_restart_smoke": "passed", "stash_sha": stash_sha,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("finalized") is True, result
    assert (repo / "a.txt").read_text() == "owner work in flight\n"
    assert stash_sha in _git(repo, "stash", "list", "--format=%H").stdout
    # Replay (e.g. double boot task): no tx left, restored content untouched.
    replay = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert replay.get("finalized") is False
    assert (repo / "a.txt").read_text() == "owner work in flight\n"


def test_boot_recovers_pre_apply_stash_crash(tmp_path, monkeypatch):
    """Crash after the durable stashing_local_work phase but before the apply:
    boot restores the stash (even when stash_sha never reached the tx) and
    clears the marker so the update can simply be retried."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    (repo / "a.txt").write_text("mid-stash crash work\n")
    status, stash_sha, _error = update_merge.stash_local_changes_for_update("crash1")
    assert status == "ok" and stash_sha
    update_merge.write_update_tx({
        "phase": "stashing_local_work", "pre_update_sha": "x" * 40,
        "pre_update_branch": head, "target_sha": "y" * 40,
        "attempt_id": "crash1", "stash_sha": "",
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("reason") == "recovered pre-apply stash crash", result
    assert (repo / "a.txt").read_text() == "mid-stash crash work\n"
    assert update_merge.read_update_tx() == {}


def test_rollback_resume_after_restore_keeps_owner_work(tmp_path, monkeypatch):
    """A rollback replay that already restored the stash (durable stash_restored
    marker) must NOT reset the recovered work away."""
    import supervisor.workers as workers

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("restored owner work\n")
    update_merge.write_update_tx({
        "phase": "rolling_back", "pre_update_sha": pre,
        "pre_update_branch": head, "target_sha": "y" * 40,
        "stash_sha": "z" * 40, "stash_restored": True,
    })
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda reason: True)
    monkeypatch.setattr(workers, "open_repo_writer_admission", lambda expected_reason="": True)

    ok, message = update_merge.rollback_managed_update("resume-test")

    assert ok, message
    assert (repo / "a.txt").read_text() == "restored owner work\n"


def test_stash_push_with_unreadable_list_fails_closed(tmp_path, monkeypatch):
    """A successful push whose entry cannot be LISTED must refuse the update:
    proceeding with an empty sha would orphan a real stash entry that no later
    restore would ever find."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    (repo / "work.txt").write_text("dirty\n")

    real_capture = git_ops.git_capture

    def flaky_list(cmd):
        if cmd[:3] == ["git", "stash", "list"]:
            return 1, "", "stash storage unreadable"
        return real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", flaky_list)

    status, sha, error = update_merge.stash_local_changes_for_update("flaky-list")

    assert status == "lookup_unknown" and sha == ""
    assert "recover" in error or "git stash list" in error


def test_managed_post_commit_gate_ignores_skip_tests_and_env(monkeypatch):
    from ouroboros.tools import git as git_mod

    calls = []

    def fake_post_commit_result(ctx, commit_message, skip_tests, tw_ref, force=False):
        calls.append({"skip_tests": skip_tests, "force": force})
        return None

    monkeypatch.setattr(git_mod, "_post_commit_result", fake_post_commit_result)
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")
    result = git_mod._managed_post_commit_tests_gate(
        object(), "msg", 0.0, True, [""], {"phase": "committing_assisted"},
    )
    assert result is None
    assert calls == [{"skip_tests": False, "force": True}]
    # Ordinary commits (no managed tx) never enter the gate.
    assert git_mod._managed_post_commit_tests_gate(object(), "msg", 0.0, True, [""], {}) is None


def test_file_directory_collision_routes_to_assisted(tmp_path, monkeypatch):
    """clean(snapshot,target) does NOT imply clean(base,target) for file/dir
    type collisions — the plan must route to the assisted lane instead of
    refusing forever with kind=unknown."""
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "node").write_text("official file\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "target adds FILE node")
    _git(repo, "checkout", "-q", head)
    # The local COMMIT adds a directory of the same name -> D/F conflict vs base.
    (repo / "node").mkdir()
    (repo / "node" / "child.txt").write_text("child\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "local adds DIRECTORY node/")
    # Dirty state mirrors the target: drop the dir, add the identical file, so
    # the SNAPSHOT merge is a clean same-content add/add while the BASE is not.
    _git(repo, "rm", "-q", "-r", "node")
    (repo / "node").write_text("official file\n")
    _point_at(monkeypatch, tmp_path, repo, head)

    plan = update_merge.plan_managed_update_merge(build=True)

    assert plan["kind"] != "clean" or plan["recommended_strategy"] == "assisted", plan
    assert not plan.get("merge_commit"), plan
    assert plan.get("recommended_strategy") == "assisted", plan


def test_rollback_restores_stashed_local_work(tmp_path, monkeypatch):
    import supervisor.workers as workers

    repo, head = _diverged_clean_repo(tmp_path, monkeypatch)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    target = _git(repo, "rev-parse", "remote-sim").stdout.strip()
    (repo / "a.txt").write_text("keep this local work\n")
    status, stash_sha, _error = update_merge.stash_local_changes_for_update("t4")
    assert status == "ok" and stash_sha
    _git(repo, "checkout", "-q", "-B", head, target)
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke", "pre_update_sha": pre,
        "pre_update_branch": head, "target_sha": target,
        "merge_commit": target, "stash_sha": stash_sha,
    })
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda reason: True)
    monkeypatch.setattr(workers, "open_repo_writer_admission", lambda expected_reason="": True)

    ok, message = update_merge.rollback_managed_update("test")

    assert ok, message
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == pre
    assert (repo / "a.txt").read_text() == "keep this local work\n"
    assert "restored" in message
