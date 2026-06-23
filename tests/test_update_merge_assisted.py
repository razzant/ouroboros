"""Tests for the AUTOMATED assisted managed-update merge (P2/SC2) — native MERGE_HEAD staged
in a real temp repo, the tx authorization gate, the conflict-marker gate, merge-state
classification, and non-destructive boot recovery."""

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


def _point_at(monkeypatch, tmp_path, repo, head):
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", head)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda branch=None: ("", "", "remote-sim"))
    (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)


def _conflict_repo(tmp_path, monkeypatch):
    """A repo where the official target and a local uncommitted edit collide on a.txt."""
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "a.txt").write_text("remote change\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "remote edits a")
    _git(repo, "checkout", "-q", head)
    (repo / "a.txt").write_text("local change\n")  # uncommitted local edit collides
    _point_at(monkeypatch, tmp_path, repo, head)
    plan = update_merge.plan_managed_update_merge(fetch=False)
    return repo, head, plan


def test_materialize_sets_merge_head_and_markers(tmp_path, monkeypatch):
    repo, head, plan = _conflict_repo(tmp_path, monkeypatch)
    assert plan["kind"] == "conflicting", plan
    ok, msg = update_merge.materialize_assisted_merge_live(
        head, plan["local_snapshot"], plan["target_sha"], plan["base_sha"]
    )
    assert ok, msg
    # MERGE_HEAD points at the official target; HEAD is re-based to the REVIEWED pre-update
    # base (so the reviewed diff includes the owner's dirty work); a.txt carries markers.
    assert update_merge._merge_head_sha() == plan["target_sha"]
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == plan["base_sha"]
    body = (repo / "a.txt").read_text()
    assert "<<<<<<<" in body and ">>>>>>>" in body
    # The marker gate (after `git add`) must REJECT the unresolved markers.
    _git(repo, "add", "-A")
    ok2, err = update_merge.managed_assisted_marker_check()
    assert not ok2 and "conflict markers" in err
    # Resolve the conflict → the gate passes.
    (repo / "a.txt").write_text("reconciled\n")
    _git(repo, "add", "-A")
    ok3, _e = update_merge.managed_assisted_marker_check()
    assert ok3


def test_assisted_head_state_in_progress_then_committed(tmp_path, monkeypatch):
    repo, head, plan = _conflict_repo(tmp_path, monkeypatch)
    update_merge.materialize_assisted_merge_live(
        head, plan["local_snapshot"], plan["target_sha"], plan["base_sha"]
    )
    tx = {"pre_update_sha": plan["base_sha"], "target_sha": plan["target_sha"]}
    # Before commit HEAD == pre_update_sha (the reviewed base) → in_progress.
    assert update_merge._assisted_head_state(tx) == "in_progress"
    # Resolve + commit (MERGE_HEAD makes it a real 2-parent merge) → committed.
    (repo / "a.txt").write_text("reconciled\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "merge resolved")
    parents = _git(repo, "rev-list", "--parents", "-n", "1", "HEAD").stdout.split()
    assert plan["target_sha"] in parents[1:]  # the official target is a real parent
    assert update_merge._assisted_head_state(tx) == "committed"


def test_managed_assisted_tx_for_authorizes_only_owner(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    update_merge.write_update_tx({"phase": "assisted_resolution", "task_id": "owner-task"})
    # The authorized task is allowed (no block); any other task is blocked.
    tx, block = update_merge.managed_assisted_tx_for("owner-task")
    assert tx and not block
    _tx2, block2 = update_merge.managed_assisted_tx_for("some-other-task")
    assert not _tx2 and "MANAGED_UPDATE_IN_PROGRESS" in block2
    # No managed tx → never blocks.
    update_merge.clear_update_tx()
    assert update_merge.managed_assisted_tx_for("any") == ({}, "")


def test_read_update_tx_strict_distinguishes_corrupt(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    assert update_merge.read_update_tx_strict()[0] == "absent"
    update_merge.write_update_tx({"phase": "assisted_resolution", "task_id": "x"})
    assert update_merge.read_update_tx_strict()[0] == "valid"
    update_merge._update_tx_marker_path().write_text("{ not json", encoding="utf-8")
    assert update_merge.read_update_tx_strict()[0] == "corrupt"
    # A corrupt marker counts as an ACTIVE tx (fail-closed) and blocks other tasks.
    assert update_merge.managed_assisted_tx_for("anyone")[1]


def test_pending_boot_smoke_not_finalized_on_failed_supervisor(tmp_path, monkeypatch):
    """A failed supervisor boot (supervisor_ready=False) must NOT clear a pending update as
    finalized, even when HEAD contains the merge — the boot-loop rollback must still fire later."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke", "merge_commit": cur,
        "pre_update_sha": cur, "pre_update_branch": head,
    })
    res = update_merge.finalize_managed_update_on_boot(supervisor_ready=False)
    assert res.get("finalized") is not True, res
    assert update_merge.read_update_tx_strict()[0] == "valid"  # survives for the next boot


def test_boot_recovery_diverged_keeps_worker_commit(tmp_path, monkeypatch):
    """A real reviewed commit that landed on top during resolution is NEVER reset away."""
    repo, head, plan = _conflict_repo(tmp_path, monkeypatch)
    # A worker landed a real reviewed commit on top of the pre-update base during resolution.
    _git(repo, "reset", "--hard", plan["base_sha"])
    (repo / "a.txt").write_text("a worker's reviewed change\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "unrelated reviewed commit")
    worker_head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    update_merge.write_update_tx({
        "phase": "assisted_resolution", "task_id": "t",
        "pre_update_sha": plan["base_sha"], "pre_update_branch": head,
        "local_snapshot": plan["local_snapshot"], "target_sha": plan["target_sha"],
    })
    res = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert res.get("abandoned") is True, res
    # The worker's commit survives; the tx is cleared (no destructive rollback).
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == worker_head
    assert update_merge.read_update_tx_strict()[0] == "absent"


def test_dirty_local_work_is_in_the_reviewed_diff(tmp_path, monkeypatch):
    """P3 regression: the owner's uncommitted/untracked local work must be part of the staged
    diff reviewed against pre_update_sha — never reachable in history as an unreviewed parent."""
    repo, head = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "b.txt").write_text("official addition\n")  # disjoint official change (clean merge)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official adds b")
    _git(repo, "checkout", "-q", head)
    (repo / "secret_local.txt").write_text("owner uncommitted work\n")  # untracked dirty work
    _point_at(monkeypatch, tmp_path, repo, head)
    plan = update_merge.plan_managed_update_merge(fetch=False)
    assert int(plan["local_dirty_count"]) > 0

    ok, msg = update_merge.materialize_assisted_merge_live(
        head, plan["local_snapshot"], plan["target_sha"], plan["base_sha"]
    )
    assert ok, msg
    _git(repo, "add", "-A")
    # The reviewed baseline is pre_update_sha — the dirty/untracked file appears in the diff,
    # so commit_reviewed's triad/scope WILL see it (it cannot slip in unreviewed).
    staged = _git(repo, "diff", "--cached", "--name-only", plan["base_sha"]).stdout.split()
    assert "secret_local.txt" in staged, staged
    assert "b.txt" in staged  # the official change is in the same reviewed diff
