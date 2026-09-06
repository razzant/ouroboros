"""Tests for the managed-update merge planner (P2) — real 3-way merge in a temp repo."""

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


def _point_at(monkeypatch, repo):
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    # DRIVE_ROOT too: restore_update_stash(context="test") logs stash_restored
    # through _log_supervisor — unpatched it leaked into the live
    # data/logs/supervisor.jsonl (issue #455 repro).
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", repo.parent / "data")
    current = _git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip()
    monkeypatch.setattr(git_ops, "BRANCH_DEV", current)
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda: ("", "ouroboros", "remote-sim"))
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: (
            "remote-sim",
            _git(repo, "rev-parse", "remote-sim").stdout.strip(),
            "",
        ),
    )


def test_plan_clean_when_disjoint(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "b.txt").write_text("remote\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "remote adds b")
    _git(repo, "checkout", "-q", head)
    _point_at(monkeypatch, repo)

    plan = update_merge.plan_managed_update_merge(fetch=False)
    assert plan["available"] is True, plan
    assert plan["kind"] == "clean", plan
    assert plan["auto_mergeable"] is True
    assert plan["recommended_strategy"] == "auto_merge"


def test_clean_divergence_preflight_recommends_automatic_git_merge(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "remote.txt").write_text("remote\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "remote")
    _git(repo, "checkout", "-q", head)
    (repo / "local.txt").write_text("local\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "local")
    _point_at(monkeypatch, repo)

    plan = update_merge.plan_managed_update_merge(fetch=False, build=False)

    assert plan["kind"] == "clean"
    assert plan["local_dirty_count"] == 0
    assert plan["recommended_strategy"] == "auto_merge"


def test_clean_fast_forward_lands_exact_official_sha(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "b.txt").write_text("remote\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "remote adds b")
    target = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", head)
    _point_at(monkeypatch, repo)

    plan = update_merge.plan_managed_update_merge(fetch=False, build=True)

    assert plan["kind"] == "clean"
    assert plan["local_snapshot"] == plan["base_sha"]
    assert plan["merge_commit"] == target
    ok, message = update_merge.apply_managed_merge_update(head, plan["merge_commit"])
    assert ok, message
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == target
    assert _git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip() == head


def test_plan_rejects_official_target_that_deletes_constitution(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    _git(repo, "rm", "-q", "BIBLE.md")
    _git(repo, "commit", "-q", "-m", "delete constitution")
    _git(repo, "checkout", "-q", head)
    _point_at(monkeypatch, repo)

    plan = update_merge.plan_managed_update_merge(fetch=False, build=True)

    assert plan["available"] is False
    assert "does not preserve BIBLE.md" in plan["error"]


def test_plan_conflicting_on_code(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "a.txt").write_text("remote change\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "remote edits a")
    _git(repo, "checkout", "-q", head)
    (repo / "a.txt").write_text("local change\n")  # uncommitted local edit collides
    _point_at(monkeypatch, repo)

    plan = update_merge.plan_managed_update_merge(fetch=False)
    assert plan["available"] is True, plan
    assert plan["kind"] == "conflicting", plan
    assert "a.txt" in plan["code_conflict_paths"]
    assert plan["recommended_strategy"] == "assisted"


def test_plan_document_conflict_uses_assisted_route(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    (repo / "README.md").write_text("base readme\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add readme")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "README.md").write_text("remote readme\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "remote edits readme")
    _git(repo, "checkout", "-q", head)
    (repo / "README.md").write_text("local readme\n")  # uncommitted local doc edit collides
    _point_at(monkeypatch, repo)

    plan = update_merge.plan_managed_update_merge(fetch=False)
    assert plan["available"] is True, plan
    assert plan["kind"] == "conflicting", plan
    assert "README.md" in plan["doc_conflict_paths"]


def test_plan_current_when_no_divergence(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _git(repo, "branch", "remote-sim")  # identical to HEAD
    _point_at(monkeypatch, repo)

    plan = update_merge.plan_managed_update_merge(fetch=False)
    assert plan["available"] is False
    assert plan["kind"] == "current"


def test_build_and_apply_clean_merge(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "b.txt").write_text("remote\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "remote adds b")
    _git(repo, "checkout", "-q", head)
    (repo / "c.txt").write_text("local untracked\n")  # local dirty work to preserve
    _point_at(monkeypatch, repo)

    plan = update_merge.plan_managed_update_merge(fetch=False, build=True)
    assert plan["kind"] == "clean", plan
    assert plan["merge_commit"], plan

    # Q1=C: local dirty work rides a stash through the apply — it never becomes
    # part of committed history — and is restored as uncommitted content after.
    status, stash_sha, stash_error = update_merge.stash_local_changes_for_update("plan-test")
    assert status == "ok" and stash_sha, stash_error
    ok, msg = update_merge.apply_managed_merge_update(head, plan["merge_commit"])
    assert ok, msg
    assert (repo / "b.txt").exists()
    restored, note = update_merge.restore_update_stash(stash_sha, context="test")
    assert restored, note
    assert (repo / "c.txt").read_text() == "local untracked\n"
    # Base was fast-forwardable, so official history lands as-is: HEAD is the
    # target itself, with no synthetic merge commit carrying local work.
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == plan["target_sha"]


def test_rollback_managed_update(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data_dir)
    monkeypatch.setattr(git_ops, "_git_dir", lambda: repo / ".git")
    import supervisor.workers as workers
    gate_calls = []
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda reason: gate_calls.append(("close", reason)))
    monkeypatch.setattr(workers, "open_repo_writer_admission", lambda expected_reason="": gate_calls.append(("open", expected_reason)))
    # simulate a bad update landed on top.
    (repo / "bad.txt").write_text("bad\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "bad update")
    update_merge.write_update_tx({"pre_update_sha": pre, "pre_update_branch": head})

    ok, msg = update_merge.rollback_managed_update("test")
    assert ok, msg
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == pre
    assert not (repo / "bad.txt").exists()
    assert update_merge.read_update_tx() == {}  # marker cleared
    assert gate_calls == [
        ("close", "managed_update:rollback"),
        ("open", "managed_update:rollback"),
    ]


def _wire_git_ops(monkeypatch, repo, data_dir):
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data_dir)
    monkeypatch.setattr(git_ops, "_git_dir", lambda: repo / ".git")


def test_finalize_clears_marker_on_healthy_boot(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _wire_git_ops(monkeypatch, repo, tmp_path / "data")
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    update_merge.write_update_tx(
        {"phase": "pending_boot_smoke", "merge_commit": cur, "pre_update_sha": cur, "pre_update_branch": head}
    )
    res = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert res["finalized"] is True, res
    assert update_merge.read_update_tx() == {}


def test_finalize_rolls_back_after_unhealthy_boot(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _wire_git_ops(monkeypatch, repo, tmp_path / "data")
    import supervisor.workers as workers
    gate_calls = []
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda reason: gate_calls.append(("close", reason)))
    monkeypatch.setattr(workers, "open_repo_writer_admission", lambda expected_reason="": gate_calls.append(("open", expected_reason)))
    (repo / "bad.txt").write_text("bad\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "bad update")
    # merge_commit points at a sha that is NOT HEAD -> health check fails; attempts 1 -> 2 -> rollback.
    update_merge.write_update_tx(
        {"phase": "pending_boot_smoke", "merge_commit": "0" * 40, "pre_update_sha": pre,
         "pre_update_branch": head, "boot_attempts": 1}
    )
    res = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert res.get("rolled_back") is True, res
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == pre
    assert gate_calls == [("close", "managed_update:rollback")]


def test_rollback_still_resets_when_the_forensics_ref_cannot_be_written(tmp_path, monkeypatch):
    """Recovery must not be traded away for a forensics branch name.

    `rollback_managed_update` publishes `failed-update-<target12>` (falling back to
    the candidate's own sha when the tx names no target) before it resets, so the
    rejected candidate keeps a name. That write is BEST-EFFORT on purpose: this
    function's job is to get the machine back onto a working revision, and every
    caller but the commit gate is a recovery path that ignores the boolean and
    relies on the reset having happened. An earlier revision returned early when the
    ref could not be written, which left the box still running the bad update — and
    `_finalize_pending_boot_smoke` returned before persisting `boot_attempts`, so the
    next boot repeated the same failing attempt forever.

    The failure is made real rather than stubbed: an existing `failed-update-<sha>/child`
    ref makes `git branch -f failed-update-<sha>` impossible (a ref cannot be both a
    directory and a file), which is exactly the name-collision case in the finding.
    """
    import supervisor.workers as workers

    repo, head = _init_repo(tmp_path)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _wire_git_ops(monkeypatch, repo, tmp_path / "data")
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda reason: True)
    monkeypatch.setattr(workers, "open_repo_writer_admission", lambda expected_reason="": True)
    (repo / "bad.txt").write_text("bad\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "bad update")

    short = _git(repo, "rev-parse", "HEAD").stdout.strip()[:12]
    blocked = _git(repo, "branch", f"failed-update-{short}/child", "HEAD")
    assert blocked.returncode == 0, blocked.stderr
    assert _git(repo, "branch", "-f", f"failed-update-{short}", "HEAD").returncode != 0, (
        "the collision did not actually make the forensics ref unwritable, so this "
        "test would pass without exercising the failure at all"
    )

    update_merge.write_update_tx({"pre_update_sha": pre, "pre_update_branch": head})
    ok, msg = update_merge.rollback_managed_update("test")

    assert ok, f"a forensics-ref failure aborted the recovery: {msg}"
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == pre, (
        "the machine was left on the bad update because a branch name could not be written"
    )
    assert not (repo / "bad.txt").exists()
    assert update_merge.read_update_tx() == {}, (
        "the tx marker survived, so the next boot resumes the update that was just rolled back"
    )


def test_mark_update_tx_gate_blocked_does_not_invent_a_transaction(tmp_path, monkeypatch):
    """No live tx means nothing to re-phase — writing one would CREATE a blocking marker.

    The caller reaches this helper on a failed rollback, and a rollback fails for
    reasons that include "the marker was already cleared". Writing a `gate_blocked`
    tx from nothing would leave a permanent phantom transaction that blocks every
    later managed update on a machine whose update had actually finished.
    """
    repo, _head = _init_repo(tmp_path)
    _wire_git_ops(monkeypatch, repo, tmp_path / "data")

    update_merge.mark_update_tx_gate_blocked("post_commit_gate_failed", "detail")
    assert update_merge.read_update_tx() == {}, "a gate-blocked marker was invented from no tx"
