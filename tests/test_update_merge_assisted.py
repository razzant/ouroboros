"""Tests for the AUTOMATED assisted managed-update merge (P2/SC2) — native MERGE_HEAD staged
in a real temp repo, the tx authorization gate, the conflict-marker gate, merge-state
classification, non-destructive boot recovery, and the rescue-before-rollback hook."""

import json
import subprocess
from types import SimpleNamespace

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


def _authority_metadata(tx):
    return {
        "managed_update": {
            "authority_fingerprint": update_merge.assisted_authority_fingerprint(tx),
        }
    }


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
    ok, msg, _m0 = update_merge.materialize_assisted_merge_live(
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


def test_marker_gate_accepts_staged_deletion_and_binary_blob(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    (repo / "binary.bin").write_bytes(b"\x00<<<<<<< ours\n>>>>>>> theirs\n")
    _git(repo, "add", "binary.bin")
    _git(repo, "rm", "a.txt")

    ok, message = update_merge.managed_assisted_marker_check()

    assert ok, message


def test_materialize_projects_version_to_target_and_pins_m0(tmp_path, monkeypatch):
    """P9 projection (Q8): a conflicted VERSION file is mechanically resolved
    to the official target's side BEFORE the resolver sees the tree, and the
    pinned M0 baseline already includes that projection (the resolution delta
    must show only the resolver's own work). The local token is deliberately
    NOT a valid version: a well-formed token conflict is already resolved by
    the carrier-span engine at the PLANNER (D34) and never reaches this lane,
    while a degraded anchor falls past the span resolver to the projection —
    which projects VERSION conflicted-or-drifted alike."""
    repo, head = _init_repo(tmp_path)
    (repo / "VERSION").write_text("1.0.0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "version base")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "VERSION").write_text("2.0.0\n")
    (repo / "official.txt").write_text("official\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official release")
    _git(repo, "checkout", "-q", head)
    (repo / "VERSION").write_text("not-a-version\n")
    (repo / "local.txt").write_text("local\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "local fork release")
    _point_at(monkeypatch, tmp_path, repo, head)
    plan = update_merge.plan_managed_update_merge(fetch=False)
    assert plan["kind"] == "conflicting", plan
    assert "VERSION" in (plan["doc_conflict_paths"] + plan["code_conflict_paths"])

    ok, msg, m0 = update_merge.materialize_assisted_merge_live(
        head, plan["base_sha"], plan["target_sha"], plan["base_sha"]
    )

    assert ok, msg
    assert "VERSION projected to the target's version" in msg
    assert update_merge.live_unmerged_paths() == []  # VERSION was the only conflict
    assert _git(repo, "show", ":VERSION").stdout == "2.0.0\n"  # staged = target's token
    assert (repo / "VERSION").read_text() == "2.0.0\n"  # worktree matches
    assert _git(repo, "show", f"{m0}:VERSION").stdout == "2.0.0\n"  # M0 includes projection


def test_materialize_is_immune_to_a_poisoned_rerere_cache(tmp_path, monkeypatch):
    """The mechanical merge must not replay remembered resolutions: with
    rerere.enabled=true and a poisoned rr-cache (a prior merge of the SAME
    conflict resolved to garbage), materialization still yields conflict
    markers as content — the M0 baseline never inherits rerere state."""
    repo, head, plan = _conflict_repo(tmp_path, monkeypatch)
    _git(repo, "config", "rerere.enabled", "true")
    # Poison the rr-cache: resolve this exact conflict once WITH rerere on.
    _git(repo, "-c", "rerere.enabled=true", "merge", "--no-commit", "--no-ff", plan["target_sha"])
    (repo / "a.txt").write_text("poisoned resolution\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "rerere.enabled=true", "rerere")  # record the resolution
    _git(repo, "merge", "--abort")
    (repo / "a.txt").write_text("local change\n")  # restore the dirty local edit

    ok, msg, _m0 = update_merge.materialize_assisted_merge_live(
        head, plan["local_snapshot"], plan["target_sha"], plan["base_sha"]
    )

    assert ok, msg
    content = (repo / "a.txt").read_text()
    assert "<<<<<<<" in content and ">>>>>>>" in content, (
        "rerere replayed a remembered resolution into the mechanical merge"
    )
    assert "poisoned resolution" not in content


def test_materialize_token_syncs_clean_carriers_and_leaves_conflicted_ones(tmp_path, monkeypatch):
    """Q8/Q24 carrier projection: a NON-conflicted carrier whose merged token
    drifted from the projected VERSION is token-synced and staged; a conflicted
    carrier keeps its markers for the resolver."""
    repo, head = _init_repo(tmp_path)
    (repo / "VERSION").write_text("1.0.0\n")
    (repo / "web").mkdir()
    (repo / "web" / "package.json").write_text('{\n  "version": "1.0.0"\n}\n')
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.0.0"\ndescription = "base"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "carrier base")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "VERSION").write_text("2.0.0\n")
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "2.0.0"\ndescription = "official side"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official bumps VERSION and pyproject")
    _git(repo, "checkout", "-q", head)
    (repo / "web" / "package.json").write_text('{\n  "version": "1.0.1"\n}\n')
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.0.1"\ndescription = "fork side"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork bumps package.json and pyproject")
    _point_at(monkeypatch, tmp_path, repo, head)
    plan = update_merge.plan_managed_update_merge(fetch=False)

    ok, msg, m0 = update_merge.materialize_assisted_merge_live(
        head, plan["base_sha"], plan["target_sha"], plan["base_sha"]
    )

    assert ok, msg
    assert "carrier file(s) token-synced" in msg
    assert '"version": "2.0.0"' in _git(repo, "show", ":web/package.json").stdout
    assert '"version": "2.0.0"' in _git(repo, "show", f"{m0}:web/package.json").stdout
    # The CONFLICTED carrier is never auto-resolved: it keeps its unmerged
    # stages and its markers for the resolver (the delta's key safety filter).
    assert "pyproject.toml" in _git(repo, "ls-files", "-u").stdout
    assert _git(repo, "rev-parse", ":pyproject.toml").returncode != 0
    assert "<<<<<<<" in (repo / "pyproject.toml").read_text()


def test_destructive_apply_guard_catches_late_mutations(tmp_path, monkeypatch):
    """The guard re-verifies the EXACT planned state immediately before the
    first destructive command: late edits, late commits, a moved branch, and a
    live merge all refuse; the pristine state passes."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()

    assert update_merge.destructive_apply_guard(head, pre) == ""
    (repo / "late.txt").write_text("late edit\n")
    assert "local changes" in update_merge.destructive_apply_guard(head, pre)
    (repo / "late.txt").unlink()
    assert update_merge.destructive_apply_guard(head, pre) == ""
    (repo / "a.txt").write_text("late committed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "late commit")
    assert "HEAD moved" in update_merge.destructive_apply_guard(head, pre)


def test_clean_diverged_build_projects_version_to_target(tmp_path, monkeypatch):
    """Q8 on the CLEAN lane, end-to-end through plan(build=True): a fork-only
    VERSION bump that merges cleanly must still land under the official
    target's version blob inside the built merge commit."""
    repo, head = _init_repo(tmp_path)
    (repo / "VERSION").write_text("1.0.0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "version base")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "official.txt").write_text("official\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official touches another file")
    _git(repo, "checkout", "-q", head)
    (repo / "VERSION").write_text("1.5.0\n")  # fork-only bump, merges clean
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork bump")
    _point_at(monkeypatch, tmp_path, repo, head)

    plan = update_merge.plan_managed_update_merge(fetch=False, build=True)

    assert plan["kind"] == "clean", plan
    merge_commit = plan["merge_commit"]
    assert merge_commit
    target_version = _git(repo, "show", f"{plan['target_sha']}:VERSION").stdout
    assert _git(repo, "show", f"{merge_commit}:VERSION").stdout == target_version


def test_failed_projection_aborts_materialization(tmp_path, monkeypatch):
    """A projection that cannot complete must abort materialization (typed
    failure) — never freeze a half-projected tree as the M0 baseline."""
    import ouroboros.tools.release_sync as release_sync

    repo, head = _init_repo(tmp_path)
    (repo / "VERSION").write_text("1.0.0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "version base")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "VERSION").write_text("2.0.0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official bump")
    _git(repo, "checkout", "-q", head)
    (repo / "local.txt").write_text("fork work\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork work")
    _point_at(monkeypatch, tmp_path, repo, head)
    plan = update_merge.plan_managed_update_merge(fetch=False)
    monkeypatch.setattr(
        release_sync, "sync_release_metadata",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("sync exploded")),
    )

    ok, msg, m0 = update_merge.materialize_assisted_merge_live(
        head, plan["base_sha"], plan["target_sha"], plan["base_sha"]
    )

    assert not ok
    assert "carrier projection failed" in msg
    assert m0 == ""


def test_cleanup_only_gate_block_retries_the_marker_and_never_rolls_back(tmp_path, monkeypatch):
    """A failed marker unlink after a HEALTHY outcome must not become a boot
    rollback: the cleanup-only gate_blocked branch retries ONLY the cleanup."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "landed.txt").write_text("healthy update result\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "healthy landed state")
    landed = _git(repo, "rev-parse", "HEAD").stdout.strip()
    update_merge.write_update_tx({
        "phase": update_merge.MARKER_CLEANUP_RETRY_PHASE,
        "gate_blocked_reason": "finalize_marker_cleanup_failed",
        "pre_update_sha": pre, "pre_update_branch": head,
    })
    _stub_worker_gates(monkeypatch)

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("finalized") is True, result
    assert update_merge.read_update_tx() == {}
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == landed, (
        "the cleanup-only recovery rolled back a healthy update"
    )


def test_restore_skips_drop_when_the_stash_list_changed_mid_restore(tmp_path, monkeypatch):
    """A concurrent stash push between the apply and the drop shifts every
    selector: the restore must then KEEP the entry (litter) rather than drop a
    selector that may now name someone else's work."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    (repo / "work.txt").write_text("ours\n")
    status, our_sha, error = update_merge.stash_local_changes_for_update("race-test")
    assert status == "ok" and our_sha, error

    real_capture = git_ops.git_capture
    state = {"list_calls": 0}

    def racing_capture(cmd):
        if cmd[:3] == ["git", "stash", "list"] and "--format=%H %gd" in cmd:
            state["list_calls"] += 1
            if state["list_calls"] == 2:
                # Interleave a foreign push right before the post-apply re-list.
                (repo / "foreign.txt").write_text("someone else\n")
                subprocess.run(["git", "-C", str(repo), "stash", "push",
                                "--include-untracked", "-m", "foreign",
                                "--", "foreign.txt"],
                               capture_output=True, text=True)
        return real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", racing_capture)

    restored, note = update_merge.restore_update_stash(our_sha, context="race")

    assert restored, note
    assert (repo / "work.txt").read_text() == "ours\n"
    shas = _git(repo, "stash", "list", "--format=%H").stdout.split()
    assert our_sha in shas, "our entry was dropped despite the shifted list"
    assert len(shas) == 2  # the foreign entry survived too


def test_boot_backfill_reprojects_before_pinning_m0(tmp_path, monkeypatch):
    """A crash between the merge and the mandatory projection must not let boot
    freeze the unprojected tree as M0: the typed projection re-runs first and a
    failure rolls back instead of pinning."""
    repo, head, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    tx["phase"] = "materializing_assisted"
    tx.pop("m0_tree", None)
    tx["local_work_carrier"] = "none"
    update_merge.write_update_tx(tx)
    _stub_worker_gates(monkeypatch)
    monkeypatch.setattr(
        update_merge, "project_version_carriers",
        lambda *_a, **_k: (False, "", "sync exploded"),
    )

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("rolled_back") is True, result
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == plan["base_sha"]


def test_projection_postcondition_catches_an_unsyncable_carrier(tmp_path, monkeypatch):
    """The sync SSOT silently skips token shapes it does not recognize: the
    postcondition must turn that into a typed failure, never a half-projected
    M0 (single-quoted pyproject version = a shape the rewriter skips)."""
    repo, head = _init_repo(tmp_path)
    (repo / "VERSION").write_text("1.0.0\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '1.0.0'\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base with single-quoted pyproject")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "VERSION").write_text("2.0.0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official bumps VERSION only")
    _git(repo, "checkout", "-q", head)
    (repo / "local.txt").write_text("fork\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork work")
    _point_at(monkeypatch, tmp_path, repo, head)
    plan = update_merge.plan_managed_update_merge(fetch=False)

    ok, msg, m0 = update_merge.materialize_assisted_merge_live(
        head, plan["base_sha"], plan["target_sha"], plan["base_sha"]
    )

    assert not ok
    assert "desynced" in msg or "carrier" in msg
    assert m0 == ""


def test_restore_with_marker_refuses_a_dirty_tree(tmp_path, monkeypatch):
    """Restoring onto a dirty tree is never safe (a conflicting apply's cleanup
    would wipe whatever made it dirty): the helper preserves the entry and
    discloses the manual recovery command instead."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    (repo / "work.txt").write_text("stashed work\n")
    status, stash_sha, error = update_merge.stash_local_changes_for_update("dirty-guard")
    assert status == "ok" and stash_sha, error
    (repo / "late.txt").write_text("late human edit\n")  # tree dirty again
    tx = {"stash_sha": stash_sha}

    note = update_merge.restore_stash_with_marker(tx, "unwind-test")

    assert "NOT auto-applied" in note and stash_sha[:12] in note
    assert (repo / "late.txt").read_text() == "late human edit\n"
    assert not (repo / "work.txt").exists()  # stayed in the stash, not half-applied
    assert stash_sha in _git(repo, "stash", "list", "--format=%H").stdout


def test_precommit_verify_requires_m0_or_its_reason_on_new_txs(tmp_path, monkeypatch):
    repo, head, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    base = {
        "pre_update_branch": head, "target_sha": plan["target_sha"],
        "pre_update_sha": plan["base_sha"], "local_work_carrier": "none",
    }
    ok, message = update_merge.managed_assisted_precommit_verify(dict(base))
    assert not ok and "m0_tree" in message
    ok2, message2 = update_merge.managed_assisted_precommit_verify(
        dict(base, m0_missing_reason="resumed_with_progress_before_m0_pin")
    )
    assert ok2, message2
    # Legacy tx (no carrier field) is untouched by the requirement.
    legacy = dict(base)
    legacy.pop("local_work_carrier")
    ok3, message3 = update_merge.managed_assisted_precommit_verify(legacy)
    assert ok3, message3


def test_boot_stash_recovery_fails_safe_when_stash_storage_is_unreadable(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    update_merge.write_update_tx({
        "phase": "stashing_local_work", "attempt_id": "boot-unreadable",
        "pre_update_sha": "x" * 40, "pre_update_branch": head, "stash_sha": "",
    })
    real_capture = git_ops.git_capture

    def flaky(cmd):
        if cmd[:3] == ["git", "stash", "list"]:
            return 1, "", "storage down"
        return real_capture(cmd)

    monkeypatch.setattr(git_ops, "git_capture", flaky)

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert "unreadable" in str(result.get("reason") or "")
    assert update_merge.read_update_tx() != {}  # the pointer survives for a later boot


def test_proof_check_never_fires_for_non_managed_commits():
    from ouroboros.tools import git as git_tool

    ctx = SimpleNamespace(task_id="ordinary", task_metadata=None, repo_dir="/nonexistent")
    assert git_tool._managed_candidate_needs_proof(ctx) is False


def test_live_unmerged_paths_error_is_not_no_conflicts(monkeypatch):
    from supervisor import update_candidate

    monkeypatch.setattr(
        update_candidate._g, "git_capture", lambda cmd: (1, "", "boom")
    )
    assert update_candidate.live_unmerged_paths() is None


def test_marker_guarded_restore_replay_does_not_wipe_restored_work(tmp_path, monkeypatch):
    """Crash between the stash apply and its drop: the tx carries
    stash_restored=True, so a replayed restore must be a no-op — never a
    conflicting re-apply whose cleanup resets the already-restored copy."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    (repo / "work.txt").write_text("owner work\n")
    status, stash_sha, error = update_merge.stash_local_changes_for_update("replay-test")
    assert status == "ok" and stash_sha, error
    tx = {"stash_sha": stash_sha, "pre_update_sha": "x" * 40, "pre_update_branch": head}
    update_merge.write_update_tx(tx)

    note1 = update_merge.restore_stash_with_marker(tx, "first")
    assert (repo / "work.txt").read_text() == "owner work\n"
    assert tx.get("stash_restored") is True

    # Simulate post-restore progress that a naive re-apply would clobber.
    (repo / "work.txt").write_text("owner work + more\n")
    note2 = update_merge.restore_stash_with_marker(tx, "replay")
    assert note2 == ""  # marker short-circuits: no re-apply, no reset
    assert (repo / "work.txt").read_text() == "owner work + more\n", (note1, note2)


def test_carrier_guidance_reaches_the_resolver_objective():
    from supervisor.update_merge_policy import assisted_objective

    tx = {"target_sha": "t" * 40, "conflict_paths": ["README.md", "ouroboros/loop.py"]}
    objective = assisted_objective(tx)
    assert "Version carriers" in objective
    assert "never delete this fork's local history rows" in objective
    # No carrier in the conflict set -> no carrier lecture.
    assert "Version carriers" not in assisted_objective(
        {"target_sha": "t" * 40, "conflict_paths": ["ouroboros/loop.py"]}
    )


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
    tx_data = {"phase": "assisted_resolution", "task_id": "owner-task"}
    metadata = _authority_metadata(tx_data)
    update_merge.write_update_tx(tx_data)
    # The authorized task is allowed (no block); any other task is blocked.
    tx, block = update_merge.managed_assisted_tx_for("owner-task", metadata)
    assert tx and not block
    _tx2, block2 = update_merge.managed_assisted_tx_for("some-other-task", metadata)
    assert not _tx2 and "MANAGED_UPDATE_IN_PROGRESS" in block2
    for phase in ("pending_boot_smoke", "rolling_back"):
        update_merge.write_update_tx({"phase": phase, "task_id": "owner-task"})
        _tx3, block3 = update_merge.managed_assisted_tx_for("owner-task", metadata)
        assert not _tx3 and "MANAGED_UPDATE_IN_PROGRESS" in block3
    # No managed tx → never blocks.
    update_merge.clear_update_tx()
    assert update_merge.managed_assisted_tx_for("any") == ({}, "")


def test_managed_update_tool_gate_fails_closed_when_state_is_unavailable(monkeypatch):
    from ouroboros.tools.registry import _managed_update_code_tool_block

    monkeypatch.setattr(
        update_merge,
        "managed_assisted_tx_for",
        lambda *_args: (_ for _ in ()).throw(OSError("state unavailable")),
    )
    ctx = type("Context", (), {"task_id": "task", "task_metadata": {}})()

    block = _managed_update_code_tool_block(ctx, "write_file")

    assert "MANAGED_UPDATE_STATE_UNAVAILABLE" in block
    assert "write_file" in block


def test_authorized_resolver_can_edit_any_conflicting_official_file(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tool
    from ouroboros.tools.registry import ToolContext

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    (repo / "BIBLE.md").write_text("local\n", encoding="utf-8")
    tx_data = {
        "phase": "assisted_resolution",
        "task_id": "update-resolver",
    }
    metadata = _authority_metadata(tx_data)
    update_merge.write_update_tx(tx_data)
    monkeypatch.setattr(git_tool, "_current_runtime_mode", lambda: "advanced")

    authorized = ToolContext(
        repo_dir=repo,
        drive_root=tmp_path / "data",
        task_id="update-resolver",
        task_metadata=metadata,
    )
    other = ToolContext(
        repo_dir=repo,
        drive_root=tmp_path / "data",
        task_id="unrelated-task",
    )

    result = git_tool._repo_write(authorized, path="BIBLE.md", content="reconciled\n")
    blocked = git_tool._repo_write(other, path="BIBLE.md", content="unrelated\n")

    assert "Written 1 file" in result
    assert (repo / "BIBLE.md").read_text(encoding="utf-8") == "reconciled\n"
    assert "CORE_PROTECTION_BLOCKED" in blocked


def test_forged_marker_without_host_metadata_cannot_authorize_or_rollback(
    tmp_path, monkeypatch
):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "keep.txt").write_text("keep\n", encoding="utf-8")
    update_merge.write_update_tx({
        "phase": "assisted_resolution",
        "task_id": "ordinary-task",
        "pre_update_sha": pre,
        "pre_update_branch": head,
        "target_sha": "b" * 40,
    })

    assert not update_merge.authorized_assisted_task("ordinary-task", {})
    managed, block = update_merge.managed_assisted_tx_for("ordinary-task", {})
    assert not managed and "MANAGED_UPDATE_IN_PROGRESS" in block
    result = update_merge.abort_orphaned_assisted_tx("ordinary-task", {})

    assert result == {"acted": False, "reason": "resolver authority mismatch"}
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == pre
    assert (repo / "keep.txt").read_text(encoding="utf-8") == "keep\n"


def test_cancelled_resolver_task_done_keeps_event_authority(tmp_path, monkeypatch):
    from supervisor.events import _handle_task_done

    metadata = {"managed_update": {"authority_fingerprint": "host-bound"}}
    calls = {}
    monkeypatch.setattr(
        update_merge,
        "abort_orphaned_assisted_tx",
        lambda task_id, task_metadata: calls.setdefault(
            "abort", (task_id, task_metadata)
        ),
    )
    monkeypatch.setattr(
        update_merge,
        "release_assisted_writer_gate_after_task",
        lambda task_metadata: calls.setdefault("release", task_metadata),
    )
    ctx = SimpleNamespace(
        RUNNING={},
        WORKERS={},
        DRIVE_ROOT=tmp_path,
        REPO_DIR=tmp_path,
        persist_queue_snapshot=lambda reason="": None,
        bridge=SimpleNamespace(push_log=lambda event: None),
    )

    _handle_task_done(
        {
            "type": "task_done",
            "task_id": "update-resolver",
            "task_type": "task",
            "status": "cancelled",
            "metadata": metadata,
        },
        ctx,
    )

    assert calls == {
        "abort": ("update-resolver", metadata),
        "release": metadata,
    }


def test_assisted_objective_is_truthful_for_any_conflict_free_reviewed_merge():
    from supervisor.update_merge_policy import assisted_objective

    objective = assisted_objective({
        "target_sha": "b" * 40,
        "conflict_paths": [],
    })

    assert "merge itself is clean" in objective
    assert "combines local and official history" in objective
    assert "conflicts are marked" not in objective
    assert "see `git status` for unmerged paths" not in objective
    # No prior rescue on the tx → the objective must not invent one.
    assert "was rescued to" not in objective


def test_boot_resume_does_not_enqueue_a_duplicate_assisted_resolver(monkeypatch):
    import supervisor.queue as queue
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "ensure_worker_pool_started", lambda **_kwargs: True)
    pending = [{"id": "resolver-task", "type": "task", "legacy_field": "preserved"}]
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(
        queue,
        "enqueue_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing resolver must not be enqueued again")
        ),
    )

    tx = {
        "task_id": "resolver-task",
        "target_sha": "b" * 40,
        "owner_chat_id": 0,
    }
    task_id = update_merge.enqueue_assisted_resolution_task(tx)

    assert task_id == "resolver-task"
    assert pending[0]["legacy_field"] == "preserved"
    assert update_merge.assisted_task_metadata_authorizes(tx, pending[0]["metadata"])


def test_assisted_resolver_readiness_waits_for_clean_tree_boot(tmp_path, monkeypatch):
    import supervisor.workers as workers

    proc = SimpleNamespace(pid=1234, is_alive=lambda: True)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(workers, "WORKERS", {})

    def start_pool(**_kwargs):
        workers.WORKERS[0] = SimpleNamespace(proc=proc)
        return True

    monkeypatch.setattr(workers, "ensure_worker_pool_started", start_pool)
    monkeypatch.setattr(
        workers,
        "_first_worker_event_since",
        lambda *_args: {"pid": 1234, "git_sha": "base-sha"},
    )

    assert update_merge.ensure_assisted_resolver_ready("base-sha", timeout_sec=0.1) is True


def test_assisted_resolver_readiness_rejects_wrong_sha(tmp_path, monkeypatch):
    import supervisor.workers as workers

    proc = SimpleNamespace(pid=1234, is_alive=lambda: True)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(
        workers,
        "ensure_worker_pool_started",
        lambda **_kwargs: workers.WORKERS.setdefault(0, SimpleNamespace(proc=proc)) is not None,
    )
    monkeypatch.setattr(
        workers,
        "_first_worker_event_since",
        lambda *_args: {"pid": 1234, "git_sha": "stale-sha"},
    )

    assert update_merge.ensure_assisted_resolver_ready("base-sha", timeout_sec=0.1) is False


def test_worker_ready_follows_update_authority_preload():
    import inspect
    import supervisor.workers as workers

    source = inspect.getsource(workers.worker_main)
    assert source.index("_prepare_worker_task_runtime()") < source.index('"worker_ready"')


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
    assert update_merge.read_update_tx()["boot_attempts"] == 1
    res2 = update_merge.finalize_managed_update_on_boot(supervisor_ready=False)
    assert res2.get("rolled_back") is True, res2
    assert update_merge.read_update_tx_strict()[0] == "absent"


def test_healthy_boot_clears_replace_intent_before_finalizing(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    git_ops._write_update_intent({"target_sha": cur})
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke", "merge_commit": cur,
        "pre_update_sha": cur, "pre_update_branch": head,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result["finalized"] is True
    assert not git_ops._update_intent_marker_path().exists()
    assert update_merge.read_update_tx_strict()[0] == "absent"


def test_boot_replays_unproven_pre_restart_smoke_before_finalizing(
    tmp_path, monkeypatch
):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    calls = []
    monkeypatch.setattr(
        update_merge,
        "update_restart_smoke",
        lambda: calls.append("smoke") or {"ok": True},
    )
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke",
        "pre_restart_smoke": "pending",
        "merge_commit": cur,
        "pre_update_sha": cur,
        "pre_update_branch": head,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result["finalized"] is True
    assert calls == ["smoke"]
    assert update_merge.read_update_tx_strict()[0] == "absent"


def test_boot_rolls_back_when_recovered_pre_restart_smoke_fails(
    tmp_path, monkeypatch
):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(
        update_merge,
        "update_restart_smoke",
        lambda: {"ok": False, "stderr": "broken", "returncode": 1},
    )
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke",
        "pre_restart_smoke": "pending",
        "merge_commit": cur,
        "pre_update_sha": cur,
        "pre_update_branch": head,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result["rolled_back"] is True
    assert update_merge.read_update_tx_strict()[0] == "absent"


def test_assisted_commit_publishes_smoke_proof_only_after_pass(monkeypatch):
    writes = []
    monkeypatch.setattr(update_merge, "write_update_tx", lambda tx: writes.append(dict(tx)))
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": True})

    ok, _message = update_merge.managed_assisted_postcommit(
        {"phase": "committing_assisted", "task_id": "resolver"},
        "c" * 40,
    )

    assert ok is True
    assert [tx["pre_restart_smoke"] for tx in writes] == ["pending", "passed"]


def test_assisted_commit_crash_before_gates_rolls_back(tmp_path, monkeypatch):
    repo, head, plan = _conflict_repo(tmp_path, monkeypatch)
    update_merge.materialize_assisted_merge_live(
        head, plan["local_snapshot"], plan["target_sha"], plan["base_sha"]
    )
    (repo / "a.txt").write_text("resolved but unproven\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "unproven merge")
    update_merge.write_update_tx({
        "phase": "committing_assisted", "task_id": "resolver",
        "pre_update_sha": plan["base_sha"], "pre_update_branch": head,
        "target_sha": plan["target_sha"],
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("rolled_back") is True, result
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == plan["base_sha"]
    assert update_merge.read_update_tx_strict()[0] == "absent"


def test_replace_crash_before_checkout_preserves_dirty_tree(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("owner dirty work\n")
    git_ops._write_update_intent({"target_sha": "b" * 40})
    update_merge.write_update_tx({
        "phase": "applying_replace", "pre_update_sha": pre,
        "pre_update_branch": head, "target_sha": "b" * 40,
        "merge_commit": "b" * 40, "pre_update_dirty_count": 1,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("abandoned") is True, result
    assert (repo / "a.txt").read_text() == "owner dirty work\n"
    assert update_merge.read_update_tx_strict()[0] == "absent"
    assert not git_ops._update_intent_marker_path().exists()


def test_replace_target_with_dirty_tree_is_rolled_back(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("target\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "target")
    target = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("partial checkout dirt\n")
    git_ops._write_update_intent({"target_sha": target})
    update_merge.write_update_tx({
        "phase": "applying_replace", "pre_update_sha": pre,
        "pre_update_branch": head, "target_sha": target,
        "merge_commit": target, "pre_update_dirty_count": 0,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("rolled_back") is True, result
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == pre
    assert not _git(repo, "status", "--porcelain").stdout.strip()


def test_rollback_disarms_replay_before_touching_dirty_tree(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("keep me\n")
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke", "pre_update_sha": pre,
        "pre_update_branch": head, "target_sha": "b" * 40,
    })
    monkeypatch.setattr(git_ops, "_clear_update_intent", lambda: False)

    ok, _message = update_merge.rollback_managed_update("test")

    assert ok is False
    assert (repo / "a.txt").read_text() == "keep me\n"
    assert update_merge.read_update_tx()["phase"] == "rolling_back"
    detail = "rollback evidence " * 200
    assert update_merge.mark_update_tx_gate_blocked("test", detail) is True
    blocked = update_merge.read_update_tx()
    # The pre-gate phase is taken OFF the marker (a refused merge left in its
    # original phase reads as an interrupted step and gets resumed/promoted);
    # boot's gate_blocked branch retries the rollback, so recovery is preserved.
    assert blocked["phase"] == update_merge.GATE_BLOCKED_PHASE
    assert blocked["gate_blocked_from_phase"] == "rolling_back"
    assert blocked["gate_blocked_detail"] == detail

    monkeypatch.setattr(git_ops, "_clear_update_intent", lambda: True)
    recovered = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert recovered["rolled_back"] is True
    assert update_merge.read_update_tx_strict()[0] == "absent"


def test_restart_smoke_syncs_dependencies_before_code_checks(monkeypatch):
    calls = []
    monkeypatch.setattr(update_merge, "managed_update_constitution_present", lambda _ref: True)
    monkeypatch.setattr(git_ops, "git_capture", lambda _cmd: (0, "", ""))
    monkeypatch.setattr(
        git_ops, "sync_runtime_dependencies",
        lambda reason: (calls.append(("deps", reason)) or (True, "ok")),
    )
    monkeypatch.setattr(
        update_merge, "_run_update_smoke",
        lambda cmd, timeout_sec=120.0: (calls.append(("smoke", cmd)) or {
            "ok": True, "stdout": "", "stderr": "", "returncode": 0,
        }),
    )

    result = update_merge.update_restart_smoke()

    assert result["ok"] is True
    assert calls[0] == ("deps", "managed_update_pre_restart")
    assert [kind for kind, _payload in calls] == ["deps", "smoke", "smoke"]


def test_restart_smoke_timeout_kills_process_tree(monkeypatch):
    import ouroboros.platform_layer as platform_layer
    from ouroboros.tools import shell

    killed = []

    class HungProcess:
        returncode = 1

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            assert self in shell._active_subprocesses
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["python"], timeout)
            return "", ""

    proc = HungProcess()
    monkeypatch.setattr(update_merge.subprocess, "Popen", lambda *_a, **_k: proc)
    monkeypatch.setattr(platform_layer, "kill_process_tree", lambda value: killed.append(value))

    result = update_merge._run_update_smoke(["python"], timeout_sec=0.01)

    assert result["returncode"] == 124
    assert killed == [proc]
    assert proc not in shell._active_subprocesses


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


def test_boot_recovery_rolls_back_interrupted_materialization(tmp_path, monkeypatch):
    repo, head, plan = _conflict_repo(tmp_path, monkeypatch)
    import supervisor.workers as workers

    gate_calls = []
    monkeypatch.setattr(
        workers,
        "close_repo_writer_admission",
        lambda reason: gate_calls.append(("close", reason)),
    )
    monkeypatch.setattr(
        workers,
        "open_repo_writer_admission",
        lambda expected_reason="": gate_calls.append(("open", expected_reason)),
    )
    _git(repo, "reset", "--hard", "HEAD")
    _git(repo, "clean", "-fd")
    _git(repo, "checkout", "-B", head, plan["local_snapshot"])
    _git(repo, "merge", "--no-commit", "--no-ff", plan["target_sha"])
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == plan["local_snapshot"]
    update_merge.write_update_tx({
        "phase": "materializing_assisted", "task_id": "t",
        "pre_update_sha": plan["base_sha"], "pre_update_branch": head,
        "local_snapshot": plan["local_snapshot"], "target_sha": plan["target_sha"],
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("rolled_back") is True, result
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == plan["base_sha"]
    assert update_merge._merge_head_sha() == ""
    assert update_merge.read_update_tx_strict()[0] == "absent"
    assert gate_calls == [("close", "managed_update:rollback")]


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

    ok, msg, _m0 = update_merge.materialize_assisted_merge_live(
        head, plan["local_snapshot"], plan["target_sha"], plan["base_sha"]
    )
    assert ok, msg
    _git(repo, "add", "-A")
    # The reviewed baseline is pre_update_sha — the dirty/untracked file appears in the diff,
    # so commit_reviewed's triad/scope WILL see it (it cannot slip in unreviewed).
    staged = _git(repo, "diff", "--cached", "--name-only", plan["base_sha"]).stdout.split()
    assert "secret_local.txt" in staged, staged
    assert "b.txt" in staged  # the official change is in the same reviewed diff


def _stub_worker_gates(monkeypatch):
    """Neutral worker-pool/admission stubs for rollback paths (parallel-safe)."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "ensure_worker_pool_started", lambda **_kwargs: True)
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda reason: None)
    monkeypatch.setattr(workers, "open_repo_writer_admission", lambda expected_reason="": None)


def _supervisor_events(tmp_path, event_type):
    path = tmp_path / "data" / "logs" / "supervisor.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == event_type]


def _materialized_conflict_tx(tmp_path, monkeypatch):
    """A live materialized assisted merge with an UNCOMMITTED resolution in the worktree."""
    repo, head, plan = _conflict_repo(tmp_path, monkeypatch)
    ok, msg, _m0 = update_merge.materialize_assisted_merge_live(
        head, plan["local_snapshot"], plan["target_sha"], plan["base_sha"]
    )
    assert ok, msg
    (repo / "a.txt").write_text("the resolver's precious resolution\n")
    tx = {
        "phase": "assisted_resolution", "task_id": "resolver",
        "pre_update_sha": plan["base_sha"], "pre_update_branch": head,
        "local_snapshot": plan["local_snapshot"], "target_sha": plan["target_sha"],
    }
    update_merge.write_update_tx(tx)
    return repo, head, plan, tx


def test_orphan_rollback_rescues_uncommitted_resolutions(tmp_path, monkeypatch):
    repo, head, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    _stub_worker_gates(monkeypatch)
    # A rollback rescue must never flip an active evolution transaction to "abandoned".
    monkeypatch.setattr(
        git_ops, "_link_rescue_to_evolution_transaction",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("rollback rescue must not link to the evolution tx")
        ),
    )

    result = update_merge.abort_orphaned_assisted_tx("resolver", _authority_metadata(tx))

    assert result.get("rolled_back") is True, result
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == plan["base_sha"]
    rescue_dirs = list((tmp_path / "data" / "archive" / "rescue").iterdir())
    assert len(rescue_dirs) == 1
    assert "the resolver's precious resolution" in (
        rescue_dirs[0] / "changes.diff"
    ).read_text(encoding="utf-8")
    meta = json.loads((rescue_dirs[0] / "rescue_meta.json").read_text(encoding="utf-8"))
    assert meta["reason"] == "managed_update_rollback:assisted_resolution_orphaned"
    assert meta["merge_head"] == plan["target_sha"]
    assert int(meta["unmerged_count"]) > 0
    assert meta["rescue_stash_error"]  # stash create fails on an unmerged index — disclosed
    assert (rescue_dirs[0] / "unmerged.txt").read_text(encoding="utf-8").strip()
    # The hook writes its own durable line BEFORE the destructive reset — a crash
    # between clear_update_tx and the terminal event cannot hide the rescue.
    captured = _supervisor_events(tmp_path, "managed_update_rescue_captured")
    assert captured and captured[-1]["rescue_path"] == str(rescue_dirs[0])
    rolled = _supervisor_events(tmp_path, "managed_update_rolled_back")
    assert rolled and rolled[-1]["rescue_path"] == str(rescue_dirs[0])
    assert rolled[-1]["reason"] == "assisted_resolution_orphaned"
    assert rolled[-1].get("rescue_ts")


def test_tests_evidence_records_only_for_authorized_resolver_and_live_suite(tmp_path, monkeypatch):
    """Single-run contract (Q10): the pre-commit proof is recorded only by the
    AUTHORIZED resolver and only when the suite actually ran (an env-disabled
    suite must not forge a proof)."""
    repo, head, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    meta = _authority_metadata(tx)

    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")
    assert update_merge.record_managed_tests_evidence("resolver", meta) == ""
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    assert update_merge.record_managed_tests_evidence("other-task", meta) == ""

    # The proof covers what the hermetic runner actually tests: the live
    # worktree projection INCLUDING unstaged edits and untracked files.
    (repo / "untracked_helper.py").write_text("VALUE = 1\n")
    tree = update_merge.record_managed_tests_evidence("resolver", meta)
    assert tree
    assert _git(repo, "show", f"{tree}:untracked_helper.py").stdout == "VALUE = 1\n"
    assert "the resolver's precious resolution" in _git(repo, "show", f"{tree}:a.txt").stdout
    assert update_merge.managed_tests_evidence_covers(tree)
    assert not update_merge.managed_tests_evidence_covers("0" * 40)
    assert not update_merge.managed_tests_evidence_covers("")

    # Fidelity guard: an untracked SYMLINK is not faithfully reproduced by the
    # runner's untracked copy — no proof may be recorded for such a candidate.
    (repo / "sneaky_link").symlink_to("a.txt")
    assert update_merge.record_managed_tests_evidence("resolver", meta) == ""


def test_managed_post_commit_gate_reuses_exact_tree_proof(tmp_path, monkeypatch):
    """The managed gate's mandate is 'the suite provably ran green on the exact
    committed tree' — proof-by-identity skips the duplicate run; any mismatch
    still pays a fresh mandatory run. Synthesis F2: the AUTHORITY is the
    process-held ctx record — FORGED durable ``tests_evidence`` (the tx marker
    is a plain resolver-writable file) never suppresses the mandatory run."""
    from ouroboros.tools import git as git_tool

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    committed_tree = _git(repo, "rev-parse", "HEAD^{tree}").stdout.strip()
    ctx = SimpleNamespace(repo_dir=str(repo), emit_progress_fn=lambda *_a, **_k: None)

    # Process-held proof for the exact committed tree -> no duplicate run.
    ctx._managed_tests_proof_trees = {committed_tree}
    monkeypatch.setattr(
        git_tool, "_post_commit_result",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("duplicate suite run")),
    )
    assert git_tool._managed_post_commit_tests_gate(
        ctx, "msg", 0.0, True, [""], {"target_sha": "x" * 40},
    ) is None

    # F2(a): a FORGED durable evidence file (tree matches the committed tree)
    # WITHOUT a ctx record -> the gate still runs the mandatory suite.
    update_merge.write_update_tx({
        "phase": "assisted_resolution", "task_id": "resolver",
        "tests_evidence": {"tree": committed_tree},
    })
    forged_ctx = SimpleNamespace(repo_dir=str(repo), emit_progress_fn=lambda *_a, **_k: None)
    ran = []
    monkeypatch.setattr(
        git_tool, "_post_commit_result",
        lambda *_a, **_k: ran.append("suite") and None,
    )
    assert git_tool._managed_post_commit_tests_gate(
        forged_ctx, "msg", 0.0, True, [""], {"target_sha": "x" * 40},
    ) is None
    assert ran == ["suite"], "forged durable tests_evidence suppressed the mandatory run"

    # Mismatched ctx proof -> the mandatory run still happens too.
    ctx._managed_tests_proof_trees = {"0" * 40}
    ran.clear()
    assert git_tool._managed_post_commit_tests_gate(
        ctx, "msg", 0.0, True, [""], {"target_sha": "x" * 40},
    ) is None
    assert ran == ["suite"]


def test_rollback_preserves_uncommitted_resolution_on_deterministic_branch(tmp_path, monkeypatch):
    """An aborted resolution survives on ``failed-update-<target12>`` as a synthetic
    commit (private index — plain write-tree is fatal on an unmerged index) with the
    natural [pre, target] parents, and a retry can find it by the target alone."""
    repo, head, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    _stub_worker_gates(monkeypatch)

    ok, msg = update_merge.rollback_managed_update("preserve_test")

    assert ok, msg
    name = f"failed-update-{plan['target_sha'][:12]}"
    assert name in msg
    kept = _git(repo, "rev-parse", name).stdout.strip()
    parents = _git(repo, "rev-list", "--parents", "-n", "1", kept).stdout.split()
    assert parents[1:] == [plan["base_sha"], plan["target_sha"]]
    assert "the resolver's precious resolution" in _git(repo, "show", f"{name}:a.txt").stdout
    rolled = _supervisor_events(tmp_path, "managed_update_rolled_back")
    assert rolled and rolled[-1]["failed_update_ref"] == name
    # A later retry of the SAME target finds the preserved attempt by target alone.
    assert update_merge.existing_failed_update_ref(plan["target_sha"]) == name
    # And the branch name reaches the resolver's objective text on that retry.
    retry_tx = dict(tx)
    retry_tx["failed_update_ref"] = name
    from supervisor.update_merge_policy import assisted_objective

    assert name in assisted_objective(retry_tx)


def test_replayed_rollback_does_not_clobber_the_preserved_attempt(tmp_path, monkeypatch):
    """A rollback replay (crash after the destructive reset) re-enters with a clean
    tree at the pre-update sha: it must NOT overwrite the deterministic branch —
    which holds the real attempt — with the bare base, and a rollback that never
    materialized anything must not mint a junk branch at all."""
    repo, head, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    _stub_worker_gates(monkeypatch)
    name = f"failed-update-{plan['target_sha'][:12]}"

    ok, msg = update_merge.rollback_managed_update("first")
    assert ok, msg
    kept = _git(repo, "rev-parse", name).stdout.strip()
    assert kept != plan["base_sha"]  # a real attempt, not the base

    # Replay: boot re-enters rolling_back with a fresh tx, tree clean at pre.
    update_merge.write_update_tx({
        "pre_update_sha": plan["base_sha"], "pre_update_branch": head,
        "target_sha": plan["target_sha"],
    })
    ok2, msg2 = update_merge.rollback_managed_update("replay")
    assert ok2, msg2
    assert _git(repo, "rev-parse", name).stdout.strip() == kept, (
        "the replayed rollback overwrote the preserved attempt with the pre-update base"
    )
    # Never-materialized rollback of a DIFFERENT target mints no junk branch.
    other_target = "f" * 40
    update_merge.write_update_tx({
        "pre_update_sha": plan["base_sha"], "pre_update_branch": head,
        "target_sha": other_target,
    })
    ok3, _msg3 = update_merge.rollback_managed_update("no_mutation")
    assert ok3
    assert _git(repo, "rev-parse", f"failed-update-{other_target[:12]}").returncode != 0
    assert update_merge.existing_failed_update_ref(other_target) == ""
    # And the retry pointer filters a branch that merely sits on the base.
    _git(repo, "branch", "-f", f"failed-update-{other_target[:12]}", plan["base_sha"])
    assert update_merge.existing_failed_update_ref(other_target, not_at=plan["base_sha"]) == ""


def test_boot_diverged_abandon_restores_the_stash(tmp_path, monkeypatch):
    """The diverged-abandon branch clears the tx — the only pointer that a stash
    restore is owed — so it must bring the owner's stashed work back first."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "precious.txt").write_text("owner work\n")
    status, stash_sha, error = update_merge.stash_local_changes_for_update("diverge-test")
    assert status == "ok" and stash_sha, error
    # A real reviewed commit lands on top while the update is in flight.
    (repo / "landed.txt").write_text("reviewed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "landed on top")
    update_merge.write_update_tx({
        "phase": "assisted_resolution", "task_id": "resolver",
        "pre_update_sha": pre, "pre_update_branch": head,
        "local_snapshot": pre, "target_sha": "e" * 40,
        "stash_sha": stash_sha, "local_work_carrier": "stash",
    })
    _stub_worker_gates(monkeypatch)

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("abandoned") is True, result
    assert (repo / "precious.txt").read_text() == "owner work\n", (
        "the abandoned update dropped the tx without restoring the owner's stash"
    )
    assert update_merge.read_update_tx() == {}


def test_materialize_projects_fork_only_version_bump_to_target(tmp_path, monkeypatch):
    """Q8 is unconditional: a fork-only VERSION bump (target side unchanged, so no
    conflict) still lands under the official target's version."""
    repo, head = _init_repo(tmp_path)
    (repo / "VERSION").write_text("1.0.0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "version base")
    _git(repo, "checkout", "-q", "-b", "remote-sim")
    (repo / "official.txt").write_text("official\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "official change, VERSION untouched")
    _git(repo, "checkout", "-q", head)
    (repo / "VERSION").write_text("1.0.1\n")  # fork-only bump
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fork bump")
    _point_at(monkeypatch, tmp_path, repo, head)
    plan = update_merge.plan_managed_update_merge(fetch=False)

    ok, msg, _m0 = update_merge.materialize_assisted_merge_live(
        head, plan["base_sha"], plan["target_sha"], plan["base_sha"]
    )

    assert ok, msg
    assert "VERSION projected to the target's version" in msg
    assert _git(repo, "show", ":VERSION").stdout == "1.0.0\n"
    assert (repo / "VERSION").read_text() == "1.0.0\n"


def test_boot_cap_rollback_rescues_before_reset(tmp_path, monkeypatch):
    repo, head, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    tx["resolution_attempts"] = 4  # past _ASSISTED_BOOT_ATTEMPT_CAP on the next boot
    update_merge.write_update_tx(tx)
    _stub_worker_gates(monkeypatch)

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("rolled_back") is True, result
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == plan["base_sha"]
    rescue_dirs = list((tmp_path / "data" / "archive" / "rescue").iterdir())
    assert len(rescue_dirs) == 1
    assert "the resolver's precious resolution" in (
        rescue_dirs[0] / "changes.diff"
    ).read_text(encoding="utf-8")
    rolled = _supervisor_events(tmp_path, "managed_update_rolled_back")
    assert rolled and rolled[-1]["rescue_path"] == str(rescue_dirs[0])
    assert rolled[-1]["reason"] == "assisted_resolution_expired"


def test_rollback_on_clean_tree_creates_no_rescue(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke", "pre_update_sha": pre, "pre_update_branch": head,
    })
    _stub_worker_gates(monkeypatch)

    ok, _message = update_merge.rollback_managed_update("clean_tree_test")

    assert ok is True
    assert not (tmp_path / "data" / "archive" / "rescue").exists()
    rolled = _supervisor_events(tmp_path, "managed_update_rolled_back")
    assert rolled
    assert "rescue_path" not in rolled[-1]
    assert "rescue_error" not in rolled[-1]


def test_rollback_replay_does_not_duplicate_rescue(tmp_path, monkeypatch):
    """No second snapshot when the tx ALREADY CARRIES a written rollback_rescue marker.

    Honest scope (accepted residual): the guarantee is at-least-once, not exactly-once —
    a crash in the window between creating the rescue dir and writing the tx marker
    replays the rescue and can leave one extra rescue dir on disk. That duplicate is
    cheap and durable; a two-phase planned/captured protocol was explicitly declined
    (Proportionality). This test pins the replay-with-marker case only."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("dirt from the attempt that was already rescued\n")
    update_merge.write_update_tx({
        "phase": "rolling_back", "pre_update_sha": pre, "pre_update_branch": head,
        "rollback_rescue": {"path": "/rescued/earlier", "ref": "refs/rescue/x", "reason": "first"},
    })
    _stub_worker_gates(monkeypatch)
    monkeypatch.setattr(
        git_ops, "rescue_before_destructive_rollback",
        lambda reason, **_kw: (_ for _ in ()).throw(
            AssertionError("a rollback replay must not take a second rescue")
        ),
    )

    ok, _message = update_merge.rollback_managed_update("replay")

    assert ok is True
    rolled = _supervisor_events(tmp_path, "managed_update_rolled_back")
    assert rolled and rolled[-1]["rescue_path"] == "/rescued/earlier"
    assert rolled[-1]["rescue_ref"] == "refs/rescue/x"


def test_rescue_failure_is_fail_open_and_disclosed(tmp_path, monkeypatch):
    """Owner decision 4=A: a failed rescue never blocks the rollback — it is disclosed."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("dirty work the rescue could not save\n")
    update_merge.write_update_tx({
        "phase": "pending_boot_smoke", "pre_update_sha": pre, "pre_update_branch": head,
    })
    _stub_worker_gates(monkeypatch)
    monkeypatch.setattr(
        git_ops, "_create_rescue_snapshot",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("disk full")),
    )

    ok, _message = update_merge.rollback_managed_update("rescue_fail")

    assert ok is True
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == pre
    rolled = _supervisor_events(tmp_path, "managed_update_rolled_back")
    assert rolled and "disk full" in rolled[-1]["rescue_error"]
    assert "rescue_path" not in rolled[-1]
    # The hook also wrote its own durable failure line before the reset.
    failed = _supervisor_events(tmp_path, "managed_update_rescue_failed")
    assert failed and "disk full" in failed[-1]["error"]
    assert failed[-1]["reason"] == "rescue_fail"


def test_failed_rollback_attempt_drops_marker_and_retry_rescues_fresh_tree(tmp_path, monkeypatch):
    """The rescue marker is per-ATTEMPT, not per-tx. A transient failure of the first
    destructive step must drop the just-written marker so the retry re-rescues the
    tree it actually finds — including second-generation work written in between."""
    import pathlib

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("first-generation resolution\n")
    update_merge.write_update_tx({
        "phase": "assisted_resolution", "task_id": "resolver",
        "pre_update_sha": pre, "pre_update_branch": head,
    })
    _stub_worker_gates(monkeypatch)
    real_git_capture = git_ops.git_capture
    armed = {"on": True}

    def flaky(cmd, *, timeout=None):  # one transient failure (index.lock class) on the first reset
        if armed["on"] and cmd == ["git", "reset", "--hard", "HEAD"]:
            armed["on"] = False
            return 1, "", "fatal: Unable to create '.git/index.lock': File exists."
        return real_git_capture(cmd, timeout=timeout)

    monkeypatch.setattr(git_ops, "git_capture", flaky)

    ok1, msg1 = update_merge.rollback_managed_update("attempt_one")

    assert ok1 is False and "reset failed" in msg1
    assert len(list((tmp_path / "data" / "archive" / "rescue").iterdir())) == 1
    # The stale first-attempt marker is gone — the retry re-runs the hook.
    assert "rollback_rescue" not in update_merge.read_update_tx()

    # The tree keeps moving before the retry (second-generation work).
    (repo / "a.txt").write_text("SECOND-GENERATION resolution\n")
    (repo / "brand_new_untracked.txt").write_text("also new\n")

    ok2, _msg2 = update_merge.rollback_managed_update("attempt_two")

    assert ok2 is True
    rescue_dirs = sorted((tmp_path / "data" / "archive" / "rescue").iterdir())
    assert len(rescue_dirs) == 2, "the retry must take a FRESH rescue of the moved tree"
    rolled = _supervisor_events(tmp_path, "managed_update_rolled_back")
    latest = pathlib.Path(rolled[-1]["rescue_path"])
    assert "SECOND-GENERATION resolution" in (latest / "changes.diff").read_text(encoding="utf-8")
    assert (latest / "untracked" / "brand_new_untracked.txt").exists()


def test_boot_rematerialize_rescues_dirty_work_and_points_resolver_at_it(tmp_path, monkeypatch):
    """The re-materialization reset (boot resume, has_progress=False) rescues surviving
    dirty resolutions, persists the tx pointer BEFORE materialize runs (a crash inside
    it must not lose the pointer), and the resumed resolver's objective points at it.
    A further boot keeps that pointer, because `materialize_assisted_merge_live` sets
    MERGE_HEAD and dirties the tree WITHOUT replaying the rescued edits — dropping the
    pointer on those two signals would lose the rescue nobody has read yet."""
    import supervisor.queue as queue
    import supervisor.workers as workers

    repo, head, plan, _tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    (repo / "a.txt").write_text("half-finished resolution\n")
    # The residual class: MERGE_HEAD lost while dirty resolution work survives.
    (repo / ".git" / "MERGE_HEAD").unlink()
    assert update_merge._merge_head_sha() == ""
    _stub_worker_gates(monkeypatch)
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    captured = []
    monkeypatch.setattr(queue, "enqueue_task", lambda task, front=False: captured.append(task))
    persisted_before_materialize = []
    real_materialize = update_merge.materialize_assisted_merge_live

    def spying_materialize(*args, **kwargs):
        persisted_before_materialize.append(
            (update_merge.read_update_tx().get("progress_rescue") or {}).get("path")
        )
        return real_materialize(*args, **kwargs)

    monkeypatch.setattr(update_merge, "materialize_assisted_merge_live", spying_materialize)

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("resumed") is True, result
    rescue_dirs = list((tmp_path / "data" / "archive" / "rescue").iterdir())
    assert len(rescue_dirs) == 1
    assert "half-finished resolution" in (
        rescue_dirs[0] / "changes.diff"
    ).read_text(encoding="utf-8")
    # The durable pointer was already on disk when materialize started.
    assert persisted_before_materialize == [str(rescue_dirs[0])]
    stored = update_merge.read_update_tx()
    assert stored["progress_rescue"]["path"] == str(rescue_dirs[0])
    meta = json.loads((rescue_dirs[0] / "rescue_meta.json").read_text(encoding="utf-8"))
    assert meta["reason"] == "managed_update_rescue:assisted_rematerialize"  # not rollback:*
    assert captured, "the resumed resolver task must be enqueued"
    assert str(rescue_dirs[0]) in captured[0]["text"]
    assert "do not run git commands" in captured[0]["text"]
    # Second boot with the merge state intact (has_progress=True). "MERGE_HEAD +
    # dirty" is exactly what the materialize above just produced, and materialize
    # never re-applies the rescued edits — so this state is NOT evidence that the
    # work came back, and the pointer must survive into the next objective.
    result2 = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert result2.get("resumed") is True, result2
    assert update_merge.read_update_tx()["progress_rescue"]["path"] == str(rescue_dirs[0])
    assert str(rescue_dirs[0]) in captured[-1]["text"]
    assert "was rescued to" in captured[-1]["text"]
