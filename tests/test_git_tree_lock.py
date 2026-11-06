"""The commit tools' git lock belongs to one working tree and to a live holder (audit of early stops, 06.11).

One install-wide path lock, closed right after it was taken and evicted after 600 s by any waiter, made every
copy of the body wait for every other during a whole review and let a waiter remove a live holder. Real git
and a real finished process throughout; the commit continuation's defaults and the attribution of a body copy
younger than its task baseline sit beside it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ouroboros.tools import git_plumbing


def _repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def _ctx(tmp_path, repo, task="task-a"):
    drive = tmp_path / "drive"
    return SimpleNamespace(repo_dir=str(repo), task_id=task, drive_path=lambda name: drive / name)


def test_two_working_trees_hold_their_own_locks_and_one_tree_serializes(tmp_path):
    one, two = _repo(tmp_path / "one"), _repo(tmp_path / "two")
    first = git_plumbing._acquire_git_lock(_ctx(tmp_path, one), timeout_sec=1)
    second = git_plumbing._acquire_git_lock(_ctx(tmp_path, two), timeout_sec=1)  # another tree never waits
    assert first.path != second.path
    with pytest.raises(TimeoutError) as busy:
        git_plumbing._acquire_git_lock(_ctx(tmp_path, one, "task-b"), timeout_sec=1)
    assert "task task-a" in str(busy.value) and "nothing was staged, reviewed or committed" in str(busy.value)
    git_plumbing._release_git_lock(first)
    assert not first.path.exists()
    again = git_plumbing._acquire_git_lock(_ctx(tmp_path, one, "task-b"), timeout_sec=1)
    git_plumbing._release_git_lock(again)
    git_plumbing._release_git_lock(second)


def test_a_live_holder_older_than_the_stale_age_is_never_evicted_and_a_dead_ones_lock_is_taken(tmp_path):
    repo = _repo(tmp_path / "one")
    held = git_plumbing._acquire_git_lock(_ctx(tmp_path, repo), timeout_sec=1)
    old = time.time() - 3_600
    os.utime(held.path, (old, old))  # an hour old: its age alone once let any waiter remove it
    with pytest.raises(TimeoutError):
        git_plumbing._acquire_git_lock(_ctx(tmp_path, repo, "task-b"), timeout_sec=1)
    assert held.path.exists()
    git_plumbing._release_git_lock(held)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    held.path.write_text(f"pid={dead.pid} task=gone locked_at=2026-01-01T00:00:00+00:00\n", encoding="utf-8")
    taken = git_plumbing._acquire_git_lock(_ctx(tmp_path, repo), timeout_sec=2)  # a dead holder's lock is taken
    git_plumbing._release_git_lock(taken)


def test_a_decision_without_a_reference_takes_this_copys_latest_returned_review(tmp_path, monkeypatch):
    from ouroboros import review_state
    from ouroboros.tools import review_collection as rc

    attempts = []
    monkeypatch.setattr(review_state, "_load_state_unlocked", lambda *_a, **_kw: SimpleNamespace(
        filter_attempts=lambda **_kw: list(attempts)))
    ctx = SimpleNamespace(repo_dir=str(tmp_path), drive_root=str(tmp_path), task_id="t1", _commit_call_notes=[])
    decision = {"disposition": "accepted", "rationale": "Read the panel."}
    assert rc.default_commit_continuation(ctx, None, decision, None) == (None, None)  # none returned: ask first
    assert "no review of this copy was returned yet" in ctx._commit_call_notes[0]
    attempts.extend(SimpleNamespace(tool_name="commit_reviewed", attempt=n, phase=phase, pre_review_fingerprint=fp)
                    for n, phase, fp in ((1, "review_only", "a"), (2, "committed", "b"), (3, "late_wait", "c")))
    reference, kept = rc.default_commit_continuation(ctx, None, decision, None)
    assert kept == decision and (reference["attempt"], reference["pre_review_fingerprint"]) == (3, "c")
    named = {"surface": "commit", "attempt": 1}
    assert rc.default_commit_continuation(ctx, named, decision, None) == (named, decision)  # a named one stays


def test_a_body_copy_younger_than_its_task_baseline_is_a_moved_base_that_names_its_paths(tmp_path):
    from ouroboros.mutation_attribution import resolve_attributed_git_paths
    from ouroboros.task_results import write_task_result

    repo = _repo(tmp_path / "copy")
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "base"], check=True)
    (repo / "a.txt").write_text("task work\n", encoding="utf-8")
    drive = tmp_path / "drive"
    write_task_result(drive, "t1", "running", mutation_evidence={"baseline": {"surfaces": []}})
    selected, _facts, error = resolve_attributed_git_paths(drive, "t1", repo, None, candidate_root=True)
    assert selected == [] and "baseline_surface_missing" in error and 'paths=["a.txt"]' in error
    selected, facts, error = resolve_attributed_git_paths(drive, "t1", repo, ["a.txt"], candidate_root=True)
    assert not error and selected == ["a.txt"] and facts["selection"]["gaps"] == ["baseline_surface_missing"]
    selected, _facts, error = resolve_attributed_git_paths(drive, "t1", repo, ["a.txt"])  # not a body copy: as before
    assert selected == [] and "GIT_ATTRIBUTION_BLOCKED" in error
