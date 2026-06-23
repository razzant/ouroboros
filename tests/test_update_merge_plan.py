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
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    head = _git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip()
    return repo, head


def _point_at(monkeypatch, repo):
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda branch=None: ("", "", "remote-sim"))


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


def test_plan_doc_reconcile(tmp_path, monkeypatch):
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
    assert plan["kind"] == "doc_reconcile", plan
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

    ok, msg = update_merge.apply_managed_merge_update(head, plan["merge_commit"])
    assert ok, msg
    # the live repo now has BOTH the remote's new file AND the local dirty work.
    assert (repo / "b.txt").exists()
    assert (repo / "c.txt").read_text() == "local untracked\n"
    # HEAD is a merge commit (self + 2 parents = local snapshot + target).
    parents = _git(repo, "rev-list", "--parents", "-n", "1", "HEAD").stdout.strip().split()
    assert len(parents) == 3


def test_rollback_managed_update(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data_dir)
    monkeypatch.setattr(git_ops, "_git_dir", lambda: repo / ".git")
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
