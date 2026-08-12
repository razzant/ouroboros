"""Durable thread worktrees: no age GC, no force reset, inspected removal.

The subagent worktree machinery cannot back these. Its provisioning
force-removes a stale checkout and branch, its removal is unconditional
``--force``, and its startup sweep deletes on retention age alone — every one
of which would silently destroy an owner's branched-off work. These tests pin
the inverted guarantees and the registry separation that makes the age sweep
structurally unable to see a thread worktree.
"""

from __future__ import annotations

import subprocess

import pytest

from ouroboros.thread_worktrees import (
    get_thread_worktree,
    inspect_thread_worktree,
    list_thread_worktrees,
    provision_thread_worktree,
    remove_thread_worktree,
)


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(root, "add", "seed.txt")
    _git(root, "commit", "-m", "seed")
    return root


@pytest.fixture()
def wt_root(tmp_path):
    return tmp_path / "thread_worktrees"


def _provision(repo, tmp_path, wt_root, thread_id=1, base_ref=""):
    return provision_thread_worktree(
        repo_dir=repo,
        project_id="racer",
        thread_id=thread_id,
        base_ref=base_ref,
        data_dir=tmp_path / "data",
        worktree_root=wt_root,
    )


def test_provision_registers_a_durable_checkout(repo, tmp_path, wt_root):
    handle = _provision(repo, tmp_path, wt_root)

    from pathlib import Path

    assert Path(handle.path).is_dir()
    assert (Path(handle.path) / "seed.txt").read_text(encoding="utf-8") == "seed\n"
    assert handle.branch.startswith("thread/")
    stored = get_thread_worktree(tmp_path / "data", "racer", 1)
    assert stored["path"] == handle.path
    assert stored["base_sha"] == handle.base_sha
    assert len(list_thread_worktrees(tmp_path / "data")) == 1


def test_provision_refuses_instead_of_force_resetting(repo, tmp_path, wt_root):
    """The subagent path clears a stale checkout+branch before creating. Doing
    that here would delete an owner's uncommitted work without a word."""
    handle = _provision(repo, tmp_path, wt_root)
    (tmp_path / "unrelated").mkdir()

    from pathlib import Path

    (Path(handle.path) / "work.txt").write_text("precious\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already has a worktree"):
        _provision(repo, tmp_path, wt_root)
    assert (Path(handle.path) / "work.txt").read_text(encoding="utf-8") == "precious\n"


def test_provision_refuses_an_existing_branch(repo, tmp_path, wt_root):
    _git(repo, "branch", "thread/racer__2")
    with pytest.raises(ValueError, match="already exists"):
        _provision(repo, tmp_path, wt_root, thread_id=2)
    assert list_thread_worktrees(tmp_path / "data") == []


def test_removal_refuses_dirty_or_unmerged_work_until_acknowledged(repo, tmp_path, wt_root):
    handle = _provision(repo, tmp_path, wt_root)

    from pathlib import Path

    (Path(handle.path) / "draft.txt").write_text("unsaved\n", encoding="utf-8")

    refused = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        worktree_root=wt_root,
    )
    assert refused["removed"] is False
    assert refused["reason"] == "unmerged_work"
    assert refused["inspection"]["dirty"] is True
    assert Path(handle.path).is_dir()

    # Commit it: now the tree is clean but the commits never reached the base.
    _git(Path(handle.path), "config", "user.email", "t@example.com")
    _git(Path(handle.path), "config", "user.name", "T")
    _git(Path(handle.path), "add", "draft.txt")
    _git(Path(handle.path), "commit", "-m", "work")

    still_refused = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        worktree_root=wt_root,
    )
    assert still_refused["removed"] is False
    assert still_refused["inspection"]["dirty"] is False
    assert still_refused["inspection"]["unmerged_commits"] == 1

    done = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )
    assert done["removed"] is True
    assert not Path(handle.path).exists()
    assert list_thread_worktrees(tmp_path / "data") == []
    # The owner acknowledged losing the CHECKOUT. Its commits are a separate
    # thing, and this branch is now the last copy of them (T3R-5).
    assert done["branch_removed"] is False
    assert "unmerged work" in done["branch_kept_reason"]
    assert handle.branch in _git(repo, "branch", "--format=%(refname:short)").stdout.split()


def test_clean_fully_merged_worktree_removes_without_ceremony(repo, tmp_path, wt_root):
    handle = _provision(repo, tmp_path, wt_root)

    from pathlib import Path

    result = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        worktree_root=wt_root,
    )
    assert result["removed"] is True
    assert result["inspection"]["unmerged_commits"] == 0
    assert not Path(handle.path).exists()
    assert get_thread_worktree(tmp_path / "data", "racer", 1) is None


def test_A10s_evidence_counts_against_the_PROJECTs_HEAD_not_the_branch_point(repo, tmp_path, wt_root):
    """T3R-4. "Commits the project folder never received" is a question about the
    PROJECT's HEAD, and the answer moves every time that HEAD does.

    Counted against the frozen ``base_sha`` instead, a checkout whose work had
    ALREADY been merged back still reported every one of those commits as
    unmerged — so the owner was asked to acknowledge destroying work that was
    already safe in their folder. Evidence that cries wolf is worse than none,
    because the owner learns to click through it.
    """
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)
    checkout = Path(handle.path)
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "T")
    (checkout / "work.txt").write_text("thread work\n", encoding="utf-8")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-m", "thread work")

    row = get_thread_worktree(tmp_path / "data", "racer", 1)
    before = inspect_thread_worktree(row)
    assert before["unmerged_commits"] == 1, "before the merge it really IS unmerged"
    assert before["unmerged_against"] == _git(repo, "rev-parse", "HEAD").stdout.strip()

    _git(repo, "merge", "--no-ff", "--no-edit", handle.branch)

    after = inspect_thread_worktree(row)
    assert after["unmerged_commits"] == 0, "the project folder HAS this work now"
    assert after["dirty"] is False
    # The branch point has not moved, and is no longer what the count is about.
    assert after["unmerged_against"] != row["base_sha"]
    assert after["unmerged_against"] == _git(repo, "rev-parse", "HEAD").stdout.strip()
    # Which is what makes this a CLEAN removal instead of a warning.
    removed = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1, worktree_root=wt_root,
    )
    assert removed["removed"] is True
    assert removed["reason"] == ""


def test_an_unreadable_project_falls_back_to_the_branch_point(repo, tmp_path, wt_root):
    """The fallback direction is the conservative one: counting from the branch
    point can only OVER-report, which refuses a removal rather than permitting
    one."""
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)
    checkout = Path(handle.path)
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "T")
    (checkout / "work.txt").write_text("thread work\n", encoding="utf-8")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-m", "thread work")
    row = dict(get_thread_worktree(tmp_path / "data", "racer", 1))
    row["repo_dir"] = str(tmp_path / "gone")

    out = inspect_thread_worktree(row)

    assert out["unmerged_against"] == row["base_sha"]
    assert out["unmerged_commits"] == 1


def test_a_clean_removal_makes_the_branch_round_trip_repeatable(repo, tmp_path, wt_root):
    """T3R-5. ``provision_thread_worktree`` refuses to reuse an existing branch —
    deliberately, so an owner's work is never clobbered. Leaving the branch behind
    after a clean removal therefore made branch → merge → remove a ONE-SHOT trip:
    the second branch-off failed with "branch already exists" and no owner surface
    could delete it.

    A clean removal now deletes the branch, and "clean" is judged twice: by this
    module's inspection AND by ``git branch -d``, which refuses on its own account
    if the branch holds anything the repository would not still have.
    """
    from pathlib import Path

    first = _provision(repo, tmp_path, wt_root)
    checkout = Path(first.path)
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "T")
    (checkout / "work.txt").write_text("thread work\n", encoding="utf-8")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-m", "thread work")
    _git(repo, "merge", "--no-ff", "--no-edit", first.branch)

    removed = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1, worktree_root=wt_root,
    )

    assert removed["removed"] is True
    assert removed["branch_removed"] is True
    assert removed["branch_kept_reason"] == ""
    assert first.branch not in _git(repo, "branch", "--format=%(refname:short)").stdout.split()
    # The whole point: the same thread can branch off again.
    second = _provision(repo, tmp_path, wt_root)
    assert second.branch == first.branch
    assert Path(second.path).is_dir()
    # Deleting the branch destroyed nothing — which is exactly what `git branch -d`
    # refusing would have told us if it had.
    assert (repo / "work.txt").read_text(encoding="utf-8") == "thread work\n"


def test_removal_of_an_unknown_thread_is_a_typed_no_op(tmp_path):
    result = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=9,
        worktree_root=tmp_path / "thread_worktrees",
    )
    assert result == {"removed": False, "reason": "unknown", "inspection": {}}


def test_a_survived_checkout_is_reported_as_not_removed_and_keeps_its_row(
    repo, tmp_path, wt_root, monkeypatch,
):
    """``git worktree remove`` runs with ``check=False`` and ``force_rmtree``
    swallows its errors, so a checkout that CANNOT be deleted (git lock, busy
    file, a filesystem that refuses) used to be reported ``removed: True`` while
    its registry row was dropped — an orphaned checkout holding the branch,
    invisible to the registry, and impossible to re-provision or remove again.

    The obstruction is a read-only PARENT (git's own removal fails on it, which
    is the real observed failure) with the rmtree fallback neutered. Neutering
    it is not artifice: `_force_rmtree`'s onerror hook now repairs the failing
    entry's parent permissions and would clear this particular obstruction on
    its own, so a read-only parent alone no longer reproduces anything. What it
    cannot repair — a file another process holds open, a refusing filesystem —
    has no portable spelling in a test, and this stands in for it. The
    GUARANTEE under test is about what the function reports when the checkout
    outlives both attempts, not about which obstruction produced that.
    """
    import os
    import stat
    from pathlib import Path

    import ouroboros.thread_worktrees as twt

    handle = _provision(repo, tmp_path, wt_root)
    monkeypatch.setattr(twt, "force_rmtree", lambda _path: None)
    original = stat.S_IMODE(os.stat(wt_root).st_mode)
    os.chmod(wt_root, 0o555)
    try:
        result = remove_thread_worktree(
            data_dir=tmp_path / "data", project_id="racer", thread_id=1,
            acknowledge_unmerged=True, worktree_root=wt_root,
        )
    finally:
        os.chmod(wt_root, original)
        if Path(handle.path).exists():
            os.chmod(Path(handle.path), 0o755)
    monkeypatch.undo()

    assert Path(handle.path).exists(), "precondition: the checkout survived"
    assert result["removed"] is False
    assert result["reason"] == "removal_failed"
    assert result["inspection"]["exists"] is True
    # The row is RETAINED: the orphan stays visible and re-removable.
    assert get_thread_worktree(tmp_path / "data", "racer", 1) is not None

    # ...and once the obstruction is gone the same call actually removes it.
    done = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )
    assert done["removed"] is True
    assert not Path(handle.path).exists()
    assert list_thread_worktrees(tmp_path / "data") == []


def test_a_malformed_row_can_never_delete_an_outside_path(repo, tmp_path, wt_root):
    import json

    from ouroboros.thread_worktrees import _registry_path

    victim = tmp_path / "not-a-worktree"
    victim.mkdir()
    (victim / "keepme.txt").write_text("keep\n", encoding="utf-8")
    path = _registry_path(tmp_path / "data")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"worktrees": [{
        "project_id": "racer", "thread_id": 1, "path": str(victim),
        "branch": "thread/x", "base_sha": "", "repo_dir": str(repo), "created_at": 0,
    }]}), encoding="utf-8")

    result = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )
    assert result["removed"] is False
    assert result["reason"] == "path_outside_root"
    assert (victim / "keepme.txt").exists()


def test_inspection_treats_unreadable_as_unsafe(tmp_path):
    """"Cannot tell" must never read as "nothing to lose"."""
    plain = tmp_path / "plain"
    plain.mkdir()
    report = inspect_thread_worktree({"path": str(plain), "base_sha": "deadbeef"})
    assert report["exists"] is True
    assert report["dirty"] is True
    assert report["error"]


def test_an_acknowledged_removal_KEEPS_the_branch_that_holds_the_commits(
    repo, tmp_path, wt_root,
):
    """A DELIBERATE reversal of T0's "a permitted removal always deletes the
    branch", decided in T3's round 2 and kept at synthesis.

    T0 deleted the branch with ``-D`` on every permitted removal, reasoning that
    provisioning refuses to reuse an existing ``thread/<name>`` so leaving one
    behind blocks branching that thread off again. But an ACKNOWLEDGED removal is
    exactly the case where the checkout's branch holds commits the project folder
    never received — its last copy — and ``-D`` destroys them to save the owner a
    ``git branch -d``. The removal is asked with ``-d``, so git is a second
    independent judge, and a branch that stays is DISCLOSED rather than silent.

    The cost T0 named is real and stated here rather than hidden: this thread
    cannot be branched off again until the owner deletes that branch themselves,
    and the refusal says so by name.
    """
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)
    (Path(handle.path) / "draft.txt").write_text("unsaved\n", encoding="utf-8")
    _git(Path(handle.path), "config", "user.email", "t@example.com")
    _git(Path(handle.path), "config", "user.name", "T")
    _git(Path(handle.path), "add", "draft.txt")
    _git(Path(handle.path), "commit", "-m", "work")

    done = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )
    assert done["removed"] is True
    assert done["branch_removed"] is False
    assert "unmerged work" in done["branch_kept_reason"]
    listed = subprocess.run(
        ["git", "branch", "--list", handle.branch],
        cwd=str(repo), capture_output=True, text=True, check=True,
    )
    assert listed.stdout.strip().lstrip("* ") == handle.branch
    assert (repo / "draft.txt").exists() is False        # the commit is only there

    # The stated cost: branching this thread off again refuses, by name.
    with pytest.raises(ValueError, match=handle.branch):
        _provision(repo, tmp_path, wt_root)


def test_a_clean_removal_frees_the_branch_so_reprovisioning_works(repo, tmp_path, wt_root):
    """The round trip that must not be one-shot: branch off -> (merge back) ->
    remove -> branch off again. A CLEAN checkout has nothing on its branch the
    repository would not still have, so the branch goes with it."""
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)
    result = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        worktree_root=wt_root,
    )
    assert result["removed"] is True and result["branch_removed"] is True
    again = _provision(repo, tmp_path, wt_root)
    assert Path(again.path).is_dir()
    assert again.branch == handle.branch


def test_worktree_ops_lock_is_keyed_on_the_repo(repo, tmp_path, wt_root):
    """T0-8: `git worktree add|remove|prune` all rewrite the SAME
    <repo>/.git/worktrees metadata. Keying the cross-process lockfile on each
    registry's own worktree ROOT gave the subagent owner and the thread owner
    two different lockfiles over one .git — they never actually serialized."""
    from pathlib import Path

    from ouroboros.subagent_worktrees import _ops_lock_path

    expected = repo / ".git" / ".worktree_ops.lock"
    assert _ops_lock_path(repo) == expected
    # Both registries resolve to that ONE file...
    assert _ops_lock_path(str(repo)) == expected
    # ...while a plain (non-repo) directory keeps a lock of its own: those ops
    # contend for a NAME under the root, not for git metadata.
    assert _ops_lock_path(wt_root) == wt_root / ".worktree_ops.lock"

    # A linked worktree hands us a .git FILE; it must still meet the main repo.
    handle = _provision(repo, tmp_path, wt_root)
    assert (Path(handle.path) / ".git").is_file()
    assert _ops_lock_path(handle.path) == expected


def test_subagent_age_sweep_cannot_see_a_thread_worktree(repo, tmp_path, wt_root):
    """R2/X3 structurally: the sweep iterates the SUBAGENT registry, so a
    separate registry file is what makes a thread worktree unreachable by it."""
    from ouroboros.subagent_worktrees import prune_orphans
    from ouroboros.thread_worktrees import _registry_path

    handle = _provision(repo, tmp_path, wt_root)
    from pathlib import Path

    summary = prune_orphans(
        worktree_root=tmp_path / "subagent_worktrees",
        data_dir=tmp_path / "data",
        retention_days=0,
    )

    assert summary == {"removed": 0, "kept": 0}
    assert Path(handle.path).is_dir()
    assert _registry_path(tmp_path / "data").name == "thread_worktrees.json"
    assert list_thread_worktrees(tmp_path / "data")


def test_force_rmtree_repairs_a_directory_instead_of_bricking_it(tmp_path):
    """The shared teardown hook must not turn a recoverable failure permanent.

    ``os.chmod(p, stat.S_IWRITE)`` REPLACED a directory's mode with ``0o200`` —
    write-only, no execute — so nothing inside it could be listed or unlinked
    afterwards. The tree survived the "force" removal AND could no longer be
    removed by anything, including the owner. The repair must be additive and
    give a directory its ``+x`` back.
    """
    import os
    import stat as stat_mod

    from ouroboros.subagent_worktrees import force_rmtree

    tree = tmp_path / "brick"
    (tree / "inner").mkdir(parents=True)
    (tree / "inner" / "file.txt").write_text("x", encoding="utf-8")
    # The exact shape the hook exists for: a locked-down directory whose
    # contents cannot be reached until permission is restored.
    os.chmod(tree / "inner", 0o500)
    os.chmod(tree, 0o500)
    try:
        force_rmtree(tree)
        assert not tree.exists(), "force_rmtree left the tree behind"
    finally:
        if tree.exists():  # keep tmp_path teardown from failing on our own bricking
            for path in sorted(tree.rglob("*"), reverse=True):
                os.chmod(path, stat_mod.S_IRWXU)
            os.chmod(tree, stat_mod.S_IRWXU)


def test_removal_validates_against_the_provisioning_root(repo, tmp_path, wt_root):
    """T0R2-9: a relocated configuration must not strand a real checkout.

    Containment is checked against the root the row was PROVISIONED under. When
    it was resolved at removal time instead, moving the configured root turned
    every existing row into `path_outside_root` — the owner's own worktree
    became permanently unremovable through the API, with no way back short of
    editing the registry by hand.
    """
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)
    assert handle.worktree_root == str(Path(wt_root).resolve())

    outcome = remove_thread_worktree(
        data_dir=tmp_path / "data",
        project_id="racer",
        thread_id=1,
        # The configuration moved after provisioning; the checkout did not.
        worktree_root=tmp_path / "relocated_worktrees",
    )

    assert outcome["removed"] is True, outcome
    assert not Path(handle.path).exists()
    assert get_thread_worktree(tmp_path / "data", "racer", 1) is None


def test_a_row_pointing_outside_its_provisioning_root_is_still_refused(repo, tmp_path, wt_root):
    """The guard is narrowed to the stored root, never dropped: a hand-edited
    row must not turn removal into `rm -rf` on an arbitrary path."""
    from pathlib import Path

    import ouroboros.thread_worktrees as twt

    _provision(repo, tmp_path, wt_root)
    outsider = tmp_path / "not_a_worktree"
    outsider.mkdir()
    (outsider / "precious.txt").write_text("owner data\n", encoding="utf-8")
    rows = twt._load(tmp_path / "data")
    rows[0]["path"] = str(outsider)
    twt._save(tmp_path / "data", rows)

    outcome = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True,
    )

    assert outcome["removed"] is False
    assert outcome["reason"] == "path_outside_root"
    assert Path(outsider / "precious.txt").exists()


def test_a_DETACHED_checkout_does_not_hide_its_branchs_commits(repo, tmp_path, wt_root):
    """A10's evidence is counted from BOTH tips, because they come apart.

    A checkout standing on a detached HEAD — or moved onto another branch — still
    has a `thread/<name>` branch holding every commit made in it. Asking only
    where HEAD points reported ZERO, and the owner was told the removal "deletes
    only the folder". Nothing was actually lost, because `git branch -d` refuses
    an unmerged branch, but evidence has to be true when it is READ, not merely
    harmless.
    """
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)
    checkout = Path(handle.path)
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "T")
    (checkout / "the_only_copy.txt").write_text("thread work\n", encoding="utf-8")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-m", "the only copy")
    branch_tip = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    # Detach onto the project's HEAD: the tree is clean and HEAD is level with
    # the project, while the branch is one commit ahead of it.
    _git(checkout, "checkout", "-q", "--detach", _git(repo, "rev-parse", "HEAD").stdout.strip())

    inspection = inspect_thread_worktree(get_thread_worktree(tmp_path / "data", "racer", 1))

    assert inspection["dirty"] is False
    assert inspection["unmerged_commits"] == 1, "the branch still holds that commit"

    refused = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1, worktree_root=wt_root,
    )

    assert refused["removed"] is False
    assert refused["reason"] == "unmerged_work"
    assert Path(handle.path).is_dir()
    # And an acknowledged removal keeps the branch, so the commit survives.
    done = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )
    assert done["removed"] is True
    assert done["branch_removed"] is False
    assert _git(repo, "cat-file", "-e", branch_tip).returncode == 0


def test_ignored_files_count_as_work_a_removal_would_destroy(repo, tmp_path, wt_root):
    """T3R2-H3: `git status --porcelain` alone hides exactly the files a thread's
    checkout is most likely to be the only copy of.

    A `.env`, a `local.db`, a `build/` — all gitignored, none listed, so the
    checkout read `dirty: false` and one-click removal force-deleted them with no
    prompt. The same `.env` the branch-off snapshot works hard to keep OUT of
    history is what this deleted from disk.
    """
    from pathlib import Path

    (repo / ".gitignore").write_text(".env\nlocal.db\nbuild/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore rules")
    handle = _provision(repo, tmp_path, wt_root)
    checkout = Path(handle.path)
    (checkout / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    (checkout / "local.db").write_text("rows\n", encoding="utf-8")
    (checkout / "build").mkdir()
    (checkout / "build" / "out.js").write_text("built\n", encoding="utf-8")

    report = inspect_thread_worktree({**handle.__dict__})

    assert report["dirty"] is True, "an ignored file is still a file the removal deletes"
    listed = "\n".join(report["dirty_files"])
    assert ".env" in listed and "local.db" in listed
    # ...so removal refuses until the owner has actually been shown them.
    refused = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1, worktree_root=wt_root,
    )
    assert refused["removed"] is False
    assert refused["reason"] == "unmerged_work"
    assert (checkout / ".env").exists()
    # And the acknowledgement is still the way through — nothing became a dead end.
    allowed = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )
    assert allowed["removed"] is True, allowed


def test_a_checkout_that_is_not_on_disk_is_cannot_tell_not_nothing_to_lose(tmp_path):
    """T3R2-M3: an unmounted volume, a folder moved out from under the registry,
    a `git worktree remove` run by hand — all answered
    `{exists: False, dirty: False, unmerged_commits: 0}`, which the removal
    prompt reads as a clean checkout and offers to delete with one click."""
    report = inspect_thread_worktree({
        "path": str(tmp_path / "gone"), "base_sha": "deadbeef", "branch": "thread/x",
    })

    assert report["exists"] is False
    assert report["dirty"] is True
    assert "not on disk" in report["error"]


def test_removal_refuses_while_the_project_is_busy(repo, tmp_path, wt_root):
    """T3R2-H5: removal deletes a folder something may be WRITING in.

    `project_lease.running_project_ids`, `thread_branching.project_is_busy` and
    ARCHITECTURE all describe this precondition; merge-back was its only caller.
    Reproduced: a running task in the checkout, merge-back correctly refuses
    `project_busy`, and removal answered `removed: True` and deleted the folder
    under the live worker.
    """
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)

    refused = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        worktree_root=wt_root, busy=True,
    )

    assert refused["removed"] is False
    assert refused["reason"] == "project_busy"
    assert Path(handle.path).is_dir()
    # It is a WAIT, not a dead end.
    allowed = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        worktree_root=wt_root, busy=False,
    )
    assert allowed["removed"] is True, allowed


def test_removal_reads_the_live_activity_query_when_no_answer_is_supplied(repo, tmp_path, wt_root, monkeypatch):
    """The default is the LIVE query — the same judge merge-back uses — not an
    argument a caller has to remember to pass."""
    import ouroboros.thread_branching as branching
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)
    monkeypatch.setattr(branching, "project_is_busy", lambda pid, repo=None: pid == "racer")

    refused = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1, worktree_root=wt_root,
    )

    assert refused["reason"] == "project_busy"
    assert Path(handle.path).is_dir()


def test_a_row_that_nominates_its_own_boundary_cannot_delete_the_owners_folder(
    repo, tmp_path, wt_root, monkeypatch,
):
    """T3R2-M7: T0R2-9 moved the containment boundary onto the row, and the row is
    untrusted input. A malformed one carrying `worktree_root=<documents>` and
    `path=<documents>/important_project` passed containment trivially and was
    deleted with `removed: True` — while the guard's own comment promised the
    opposite. Two INDEPENDENT facts are required now: a root this process would
    itself accept, and the path this thread's checkout is derived to have."""
    from pathlib import Path

    import ouroboros.thread_worktrees as twt

    documents = tmp_path / "Documents"
    victim = documents / "important_project"
    victim.mkdir(parents=True)
    (victim / "thesis.txt").write_text("years of work\n", encoding="utf-8")
    _provision(repo, tmp_path, wt_root)
    rows = twt._load(tmp_path / "data")
    rows[0]["path"] = str(victim)
    rows[0]["worktree_root"] = str(documents)
    twt._save(tmp_path / "data", rows)

    outcome = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )

    assert outcome["removed"] is False
    assert outcome["reason"] == "path_outside_root"
    assert (victim / "thesis.txt").read_text(encoding="utf-8") == "years of work\n"
    assert Path(documents).is_dir()


def test_removal_does_not_hold_the_registry_lock_across_its_git_calls(repo, tmp_path, wt_root, monkeypatch):
    """T3R2-L2: `_LOCK` guards the registry read and the final save, nothing
    between. Held across two `run_git` calls, `force_rmtree`, a prune and a
    `git branch -d`, it blocked every `thread_location`/`get_thread_worktree`
    read — and with `run_git`'s known missing timeout, for an unbounded time."""
    import ouroboros.thread_worktrees as twt

    _provision(repo, tmp_path, wt_root)
    seen = []
    real_run_git = twt.run_git

    def _probe(*args, **kwargs):
        # Mid-removal, from THIS thread's point of view the RLock is re-entrant,
        # so the honest probe is whether the lock is held at all.
        seen.append(twt._LOCK._is_owned())
        return real_run_git(*args, **kwargs)

    monkeypatch.setattr(twt, "run_git", _probe)

    outcome = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )

    assert outcome["removed"] is True, outcome
    assert seen, "the removal must actually have called git"
    assert not any(seen), "the registry lock was held across a git call"


def test_checkout_work_at_risk_separates_the_unlosable_from_the_rebuildable():
    """The question DELETION asks, which is not the question REMOVAL asks.

    `inspect_thread_worktree` answers "what would removing this DESTROY", and an
    ignored `node_modules/` belongs in that answer (H3). Deletion asks the
    narrower "what here cannot be rebuilt", because only that may block a gesture
    aimed at the thread rather than at the folder — refusing a delete over a build
    directory made "delete the thread and its folder" a three-step detour.

    A pure read over an existing inspection: no git, no disk.
    """
    from ouroboros.thread_worktrees import checkout_work_at_risk

    clean = checkout_work_at_risk(
        {"dirty": False, "dirty_files": [], "unmerged_commits": 0, "error": ""},
    )
    assert clean["at_risk"] is False
    assert clean == {
        "at_risk": False, "unmerged_commits": 0, "tracked_files": [],
        "untracked_files": [], "ignored_files": [], "omitted_files": 0,
        "unreadable": "",
    }

    rebuildable = checkout_work_at_risk({
        "dirty": True,
        "dirty_files": ["!! node_modules/", "!! build.log", "?? scratch.txt"],
        "unmerged_commits": 0,
        "error": "",
    })
    assert rebuildable["at_risk"] is False
    assert rebuildable["ignored_files"] == ["!! node_modules/", "!! build.log"]
    assert rebuildable["untracked_files"] == ["?? scratch.txt"]
    assert rebuildable["tracked_files"] == []

    # A TRACKED modification is at risk: its previous contents are in history,
    # this edit is nowhere at all. Every porcelain code that is not ?? or !!
    # counts, including the staged, renamed and conflicted spellings.
    for line in (" M app.txt", "M  app.txt", "A  new.py", "D  gone.txt",
                 "R  old.txt -> new.txt", "UU merged.txt", "AM half.txt"):
        risk = checkout_work_at_risk(
            {"dirty": True, "dirty_files": [line], "unmerged_commits": 0, "error": ""},
        )
        assert risk["at_risk"] is True, line
        assert risk["tracked_files"] == [line]

    # Commits the project folder never received are at risk on their own, even in
    # a spotlessly clean tree.
    commits = checkout_work_at_risk(
        {"dirty": False, "dirty_files": [], "unmerged_commits": 3, "error": ""},
    )
    assert commits["at_risk"] is True
    assert commits["unmerged_commits"] == 3

    # "Cannot tell" must never read as "nothing to lose".
    unreadable = checkout_work_at_risk(
        {"dirty": True, "dirty_files": [], "unmerged_commits": 0,
         "error": "the checkout is not on disk: /gone"},
    )
    assert unreadable["at_risk"] is True
    assert unreadable["unreadable"].startswith("the checkout is not on disk")


def test_an_acknowledged_removal_keeps_the_branch_only_for_COMMITS(repo, tmp_path, wt_root):
    """A branch is kept because it holds HISTORY, never because a folder was dirty.

    An acknowledged removal whose only dirt was an ignored `node_modules/` has
    nothing on its branch the repository would not still have, and keeping it left
    a `thread/<name>` behind that the next branch-off refuses on — and that
    nothing can reach at all once the thread it belonged to is tombstoned.
    `git branch -d` is still the second judge, so this only ever asks.
    """
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)
    checkout = Path(handle.path)
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "ignore logs")
    (checkout / "build.log").write_text("noise\n", encoding="utf-8")

    refused = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1, worktree_root=wt_root,
    )
    # A10 is UNCHANGED: removal still refuses over ignored files until acknowledged.
    assert refused["removed"] is False
    assert refused["reason"] == "unmerged_work"

    done = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )

    assert done["removed"] is True
    assert done["branch_removed"] is True, done["branch_kept_reason"]
    assert done["branch_kept_reason"] == ""
    assert handle.branch not in _git(repo, "branch", "--format=%(refname:short)").stdout.split()


# --------------------------------------------------------------------------- #
# P4 — every git call on an owner-facing worktree path is BOUNDED
# --------------------------------------------------------------------------- #

def test_every_git_call_in_this_module_goes_through_the_bounded_seam():
    """P4: `subagent_worktrees.run_git` passes NO timeout to `subprocess.run`.

    That is correct for its own callers (background provisioning, the startup
    orphan sweep — nothing waits on them), but this module is reached from six
    routes that did not exist before T3: `GET`/`POST` on a thread's worktree,
    branch-off, merge-back, thread delete and `DELETE /api/projects/{id}`. A
    wedged git there holds the owner's request and a thread-pool thread forever.

    Read from the AST so prose about the unbounded helper cannot satisfy it: the
    ONLY function allowed to call `run_git` is the bounded `_git` wrapper, and it
    must pass a `timeout`.
    """
    import ast
    import inspect

    import ouroboros.thread_worktrees as twt

    tree = ast.parse(inspect.getsource(twt))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "run_git"
                and node.name != "_git"
            ):
                offenders.append(node.name)
    assert not offenders, f"unbounded git calls remain in: {sorted(set(offenders))}"

    seam = ast.parse(inspect.getsource(twt._git))
    passes_timeout = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_git"
        and any(kw.arg == "timeout" for kw in node.keywords)
        for node in ast.walk(seam)
    )
    assert passes_timeout, "the seam must pass a timeout through to run_git"
    # ...and the ceiling is the SSOT knob, not a module-local number.
    from ouroboros.config import get_thread_git_timeout_sec

    assert twt._git_timeout_sec() == get_thread_git_timeout_sec()


def test_the_shared_helper_accepts_a_timeout_and_defaults_to_unbounded():
    """The pass-through is added WITHOUT changing the pre-existing surface: the
    subagent path keeps its unbounded default, which is what its own callers were
    written under."""
    import inspect

    from ouroboros.subagent_worktrees import run_git

    signature = inspect.signature(run_git)
    assert "timeout" in signature.parameters
    assert signature.parameters["timeout"].default is None


def test_a_timed_out_git_is_a_typed_outcome_not_a_traceback(repo, tmp_path, wt_root, monkeypatch):
    """An expiry has to be legible on both call shapes.

    `check=False` (every read, and both deletions) must come back as rc=124 with a
    sentence naming the ceiling, so the inspection reports it as "cannot tell" —
    which already counts as UNSAFE. `check=True` (provisioning, which must refuse
    rather than continue) must raise into the channel `branch_off_thread` already
    turns into a typed `branch_failed`.
    """
    import ouroboros.thread_worktrees as twt

    def _wedged(root, *args, **kwargs):
        raise subprocess.TimeoutExpired(["git", *args], kwargs.get("timeout") or 1)

    monkeypatch.setattr(twt, "run_git", _wedged)

    soft = twt._git(repo, "status", "--porcelain", check=False)
    assert soft.returncode == 124
    assert "OUROBOROS_THREAD_GIT_TIMEOUT_SEC" in soft.stderr

    with pytest.raises(ValueError) as raised:
        twt._git(repo, "rev-parse", "HEAD")
    assert "OUROBOROS_THREAD_GIT_TIMEOUT_SEC" in str(raised.value)

    # The inspection folds the soft form into its own "unsafe by construction"
    # answer rather than raising out of an HTTP handler.
    out = inspect_thread_worktree(
        {"path": str(repo), "repo_dir": str(tmp_path / "nope"), "branch": "thread/x",
         "base_sha": "deadbeef"},
    )
    assert out["dirty"] is True
    assert "did not finish within" in out["error"]


def test_a_wedged_git_lets_the_inspection_return(tmp_path, monkeypatch):
    """The reproduction itself: with a `git` that never exits, the owner-facing
    inspection used to hang forever. It now returns inside the ceiling with a
    typed error. Bounded by the SSOT knob's own minimum (5s)."""
    import os
    import threading

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    fake = fakebin / "git"
    # `exec`, not a bare `sleep`: without it the shell forks the sleep as a CHILD,
    # and when the timeout kills the shell the sleep is re-parented to init and
    # survives the whole session. The preflight container reports exactly that as
    # PREFLIGHT_CONTAINMENT_FAILED and refuses the pass verdict — a leaked tree
    # outlives the run that made it. With `exec` the process the timeout kills IS
    # the sleep, so this test owns no process once it returns.
    fake.write_text("#!/bin/sh\nexec sleep 3600\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fakebin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", "5")

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    row = {
        "path": str(checkout),
        # A repo_dir that is not a directory, so `_project_head` costs no git call
        # and the test spends exactly one ceiling.
        "repo_dir": str(tmp_path / "absent"),
        "branch": "thread/x",
        "base_sha": "deadbeef",
    }

    result = {}

    def _call():
        result["out"] = inspect_thread_worktree(row)

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    worker.join(timeout=45.0)
    assert not worker.is_alive(), "inspect_thread_worktree never returned — unbounded"
    assert result["out"]["dirty"] is True
    assert "did not finish within" in result["out"]["error"]


# --------------------------------------------------------------------------- #
# P5(i) — the removal HOLDS the folder it is deleting
# --------------------------------------------------------------------------- #

def test_removal_reserves_the_checkout_lane_for_its_whole_window(repo, tmp_path, wt_root, monkeypatch):
    """P5(i): `_project_is_busy` is a bare READ.

    Merge-back HOLDS the folder it rewrites (`reserved_folder_lane`); removal held
    nothing, so between "nothing is running in this checkout" and the `rmtree` the
    scheduler could admit a task straight into the folder being deleted. The
    reservation is observed from INSIDE the operation — at the busy check, at the
    inspection and at the deletion — because a claim taken and dropped before the
    destructive part would prove nothing.

    Ordering is deliberately NOT changed: this is a reservation, not a routing
    fence, so a REFUSED removal still touches nothing (86aaf2b1).
    """
    from ouroboros.project_lease import (
        candidate_is_leasable,
        normalize_workspace_root,
        reserved_folder_lanes,
        running_project_lanes,
    )
    import ouroboros.thread_worktrees as twt

    handle = _provision(repo, tmp_path, wt_root)
    lane = ("", normalize_workspace_root(handle.path))
    seen: list[tuple[str, bool]] = []

    real_inspect = twt.inspect_thread_worktree
    real_rmtree = twt.force_rmtree

    def _inspect(row):
        seen.append(("inspect", lane in reserved_folder_lanes()))
        return real_inspect(row)

    def _rmtree(path):
        seen.append(("rmtree", lane in reserved_folder_lanes()))
        return real_rmtree(path)

    monkeypatch.setattr(twt, "inspect_thread_worktree", _inspect)
    monkeypatch.setattr(twt, "force_rmtree", _rmtree)
    # `busy` is answered by the module's own judge so the reservation is exercised
    # exactly where the check-then-act gap used to be.
    monkeypatch.setattr(
        twt, "_project_is_busy",
        lambda pid, row: seen.append(("busy", lane in reserved_folder_lanes())) or False,
    )

    outcome = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1,
        acknowledge_unmerged=True, worktree_root=wt_root,
    )
    assert outcome["removed"] is True, outcome

    stages = dict(seen)
    assert "busy" in stages and "inspect" in stages, seen
    assert all(held for _stage, held in seen), seen
    # A task naming that checkout could not have been admitted during the window.
    with_lane = running_project_lanes([], {}) | {lane}
    assert candidate_is_leasable(
        {"id": "t9", "workspace_root": handle.path}, with_lane, {},
    ) is False
    # And the folder is free again the moment the removal returns.
    assert lane not in reserved_folder_lanes()


def test_a_refused_removal_still_destroys_nothing(repo, tmp_path, wt_root):
    """The reservation must not have reordered anything: 86aaf2b1 put the
    inspection and every refusal BEFORE any destruction on purpose."""
    from pathlib import Path

    handle = _provision(repo, tmp_path, wt_root)
    checkout = Path(handle.path)
    (checkout / "unsaved.txt").write_text("only copy\n", encoding="utf-8")

    refused = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1, worktree_root=wt_root,
    )
    assert refused["removed"] is False
    assert refused["reason"] == "unmerged_work"
    assert (checkout / "unsaved.txt").read_text(encoding="utf-8") == "only copy\n"
    assert get_thread_worktree(tmp_path / "data", "racer", 1) is not None


def test_the_removal_refusal_states_the_TRUE_number_of_dirty_files(repo, tmp_path, wt_root):
    """The count in the sentence immediately before an irreversible removal.

    `inspect_thread_worktree` bounds `dirty_files` at 200 — an unbounded list on
    an owner-facing envelope is its own problem — and the removal refusal counted
    the SLICE. A long-running agent leaves ordinary modified TRACKED files, so
    800 of them were announced to the owner as "200 uncommitted file changes":
    the qualitative claim stayed true and the acknowledgement was still required,
    but the magnitude the owner decided on was wrong by a factor of four.

    A wholly-ignored DIRECTORY does not reproduce it — git collapses
    `node_modules/` to one entry. It takes plain files, which is the ordinary
    case.
    """
    from pathlib import Path

    from ouroboros.gateway.project_threads import _removal_message

    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    for i in range(400):
        (repo / f"f{i}.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "400 tracked files")

    handle = _provision(repo, tmp_path, wt_root)
    checkout = Path(handle.path)
    for i in range(400):
        (checkout / f"f{i}.txt").write_text("edited by the agent\n", encoding="utf-8")
        (checkout / f"a{i}.log").write_text("noise\n", encoding="utf-8")

    inspection = inspect_thread_worktree(get_thread_worktree(tmp_path / "data", "racer", 1))

    assert inspection["dirty"] is True

    # The sentence the owner reads, asserted FIRST: against f05d429d it says
    # "200 uncommitted file changes" about 800 of them.
    message = _removal_message("unmerged_work", inspection)

    assert "800 uncommitted file changes" in message, message
    assert "200 uncommitted file changes" not in message, message
    # ...and the shorter list is disclosed rather than left to imply the count.
    assert "Only the first 200 of those files are listed here." in message, message

    assert inspection["dirty_files_total"] == 800, inspection["dirty_files_total"]
    assert len(inspection["dirty_files"]) == 200, "the listing itself stays bounded"

    # The refusal itself is unchanged: this was never a hole in the safety gate,
    # only in what the gate SAID.
    refused = remove_thread_worktree(
        data_dir=tmp_path / "data", project_id="racer", thread_id=1, worktree_root=wt_root,
    )
    assert refused["removed"] is False
    assert refused["reason"] == "unmerged_work"
    # And the inspection the ROUTE builds its copy from carries the same total,
    # so the sentence the owner actually receives is this one.
    assert _removal_message(refused["reason"], refused["inspection"]) == message


def test_the_removal_sentence_reads_correctly_at_every_boundary():
    """0, 1, exactly at the cap, and one past it — a pure read, no git."""
    from ouroboros.gateway.project_threads import _removal_message

    none_dirty = _removal_message(
        "unmerged_work",
        {"dirty_files": [], "dirty_files_total": 0, "unmerged_commits": 2, "error": ""},
    )
    assert "uncommitted file change" not in none_dirty, none_dirty
    assert "Only the first" not in none_dirty, none_dirty

    one = _removal_message(
        "unmerged_work",
        {"dirty_files": [" M a.txt"], "dirty_files_total": 1, "unmerged_commits": 0, "error": ""},
    )
    assert "1 uncommitted file change." in one, one
    assert "1 uncommitted file changes" not in one, one
    assert "Only the first" not in one, one

    exactly = _removal_message(
        "unmerged_work",
        {"dirty_files": [f" M f{i}.txt" for i in range(200)], "dirty_files_total": 200,
         "unmerged_commits": 0, "error": ""},
    )
    assert "200 uncommitted file changes" in exactly, exactly
    assert "Only the first" not in exactly, "nothing was left out at exactly the cap"

    one_past = _removal_message(
        "unmerged_work",
        {"dirty_files": [f" M f{i}.txt" for i in range(200)], "dirty_files_total": 201,
         "unmerged_commits": 0, "error": ""},
    )
    assert "201 uncommitted file changes" in one_past, one_past
    assert "Only the first 200 of those files are listed here." in one_past, one_past

    # A count with an EMPTY listing never renders "Only the first 0 …". The
    # producer cannot make this shape, but a sentence that reads as nonsense is
    # still a sentence the owner was shown.
    listless = _removal_message(
        "unmerged_work",
        {"dirty_files": [], "dirty_files_total": 5, "unmerged_commits": 0, "error": ""},
    )
    assert "5 uncommitted file changes" in listless, listless
    assert "None of those files are listed here." in listless, listless
    assert "Only the first 0" not in listless, listless

    # An inspection that never carried the field reads as "the listing IS the
    # set" — the old behaviour, never an under-count of what is in hand.
    legacy = _removal_message(
        "unmerged_work",
        {"dirty_files": [" M a.txt", " M b.txt"], "unmerged_commits": 0, "error": ""},
    )
    assert "2 uncommitted file changes" in legacy, legacy
    assert "Only the first" not in legacy, legacy


def test_the_delete_copy_discloses_what_its_per_category_counts_left_out():
    """`checkout_work_at_risk` splits a BOUNDED listing, so its category lengths
    count what was SHOWN. Which category the unlisted entries fall into is not
    knowable from the listing, so the copy states the one thing that is true:
    how many were left out. Without it the delete refusal would have gone on
    saying "200 files" in the same release that taught the removal refusal 800.
    """
    from ouroboros.gateway.project_threads import (
        _delete_confirm_message,
        _delete_refusal_message,
    )
    from ouroboros.thread_worktrees import checkout_work_at_risk

    tracked = checkout_work_at_risk({
        "dirty": True,
        "dirty_files": [f" M f{i}.txt" for i in range(200)],
        "dirty_files_total": 800,
        "unmerged_commits": 0,
        "error": "",
    })
    assert tracked["omitted_files"] == 600
    refusal = _delete_refusal_message(tracked)
    assert "changes to 200 files git is tracking" in refusal, refusal
    assert "600 further changed files in that checkout are not listed here." in refusal, refusal

    ignored = checkout_work_at_risk({
        "dirty": True,
        "dirty_files": [f"!! a{i}.log" for i in range(200)],
        "dirty_files_total": 201,
        "unmerged_commits": 0,
        "error": "",
    })
    assert ignored["at_risk"] is False
    confirm = _delete_confirm_message(ignored)
    assert "1 further changed file in that checkout is not listed here." in confirm, confirm

    # Nothing omitted says nothing — never "0 more files".
    whole = checkout_work_at_risk({
        "dirty": True, "dirty_files": ["!! a.log"], "dirty_files_total": 1,
        "unmerged_commits": 0, "error": "",
    })
    assert whole["omitted_files"] == 0
    assert "not listed here" not in _delete_confirm_message(whole)
