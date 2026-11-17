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


def _attempt(n, phase, fingerprint):
    return SimpleNamespace(tool_name="commit_reviewed", attempt=n, phase=phase, pre_review_fingerprint=fingerprint)


def _continuation_ctx(tmp_path, monkeypatch, attempts):
    from ouroboros import config, review_state

    monkeypatch.setattr(review_state, "_load_state_unlocked", lambda *_a, **_kw: SimpleNamespace(
        filter_attempts=lambda **_kw: list(attempts)))
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    return SimpleNamespace(repo_dir=str(tmp_path), drive_root=str(tmp_path), task_id="t1", _commit_call_notes=[])


DECISION = {"disposition": "accepted", "rationale": "Read the panel."}


def test_a_decision_without_a_reference_asks_for_the_review_or_names_the_exact_reference(tmp_path, monkeypatch):
    """Owner 11.10 (review 06.11), both directions: with no outcome of this copy returned, the decision is set aside
    and the call asks for the review; with one returned, nothing is staged and the refusal names that outcome's exact
    reference, since the reference a decision quotes is what shows the outcome reached the mind. A named one passes."""
    import json
    from ouroboros.tools import review_collection as rc

    attempts = []
    ctx = _continuation_ctx(tmp_path, monkeypatch, attempts)
    call = {"author_disposition": DECISION}
    assert rc.default_commit_continuation(ctx, call) == "" and call["author_disposition"] is None
    assert "no review of this copy was returned yet" in ctx._commit_call_notes[0]
    attempts.extend(_attempt(n, phase, fp) for n, phase, fp in ((1, "review_only", "a"), (2, "committed", "b"),
                                                                  (3, "late_wait", "c")))
    call = {"author_disposition": DECISION}
    refusal = rc.default_commit_continuation(ctx, call)
    assert "Nothing was staged or committed" in refusal and call == {"author_disposition": DECISION}
    named = json.loads(refusal.split("review_reference=", 1)[1].split(" and no author_disposition", 1)[0])
    assert (named["attempt"], named["pre_review_fingerprint"], named["task_id"]) == (3, "c", "t1")
    call = {"author_disposition": DECISION, "review_reference": named}
    assert rc.default_commit_continuation(ctx, call) == "" and call["review_reference"] is named


def test_the_continuation_texts_name_the_step_their_cause_leaves(tmp_path, monkeypatch):
    """06.11: a running wave names collecting it, else the returned outcome; an unreadable state names the
    reference instead of buying a review. A leftover Blocking value of the retired setting changes no text."""
    from ouroboros import review_state
    from ouroboros.tools import review_collection as rc

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")  # retired: read by nothing
    ctx = _continuation_ctx(tmp_path, monkeypatch, [])
    assert "collect that outcome" in rc.continuation_refusal(rc.ContinuationWaits("reviewers are still running"))
    assert "decide with the review_reference an outcome you read returned" in rc.continuation_refusal(ValueError("x"))
    assert "a passing review lands the commit" not in rc.continuation_refusal(ValueError("x"))

    def unreadable(*_a, **_kw):
        raise OSError("state locked away")
    monkeypatch.setattr(review_state, "_load_state_unlocked", unreadable)
    text = rc.default_commit_continuation(ctx, {"author_disposition": DECISION})
    assert "unreadable (state locked away)" in text and "Next: name review_reference" in text
    assert rc.default_commit_continuation(ctx, {"author_disposition": DECISION, "review_reference": {}}) == ""


def test_a_continuation_sets_aside_the_fields_of_another_step_and_a_new_ask_keeps_them():
    from ouroboros.tools import review_collection as rc

    ctx = SimpleNamespace(_commit_call_notes=[], _review_since="record-1")
    assert rc.set_aside_wrong_step_fields(ctx, False, "mail") == "mail" and ctx._review_since == "record-1"
    assert rc.set_aside_wrong_step_fields(ctx, True, "mail") is None and ctx._review_since == ""
    assert [note.split(" was set aside")[0] for note in ctx._commit_call_notes] == ["wake_on", "since_review"]


def test_a_revert_takes_the_lock_of_the_tree_it_writes(tmp_path, monkeypatch):
    """Review 06.11, both directions with real git: a revert of a linked worktree waits for that tree's lock and
    lands nothing, while a held lock of the serving checkout does not stop it."""
    from functools import partial
    from ouroboros.tools import git, git_vcs_ops

    def run(repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

    serving = _repo(tmp_path / "serving")
    run(serving, "config", "user.email", "t@t")
    run(serving, "config", "user.name", "t")
    (serving / "f.txt").write_text("one\n", encoding="utf-8")
    run(serving, "add", "f.txt")
    run(serving, "commit", "-qm", "one")
    target = tmp_path / "linked"
    run(serving, "worktree", "add", "-qb", "target", str(target))
    (target / "f.txt").write_text("two\n", encoding="utf-8")
    run(target, "commit", "-qam", "two")
    head = run(target, "rev-parse", "HEAD")
    ctx = _ctx(tmp_path, serving)
    ctx.task_metadata, ctx.drive_root = {}, tmp_path / "drive"
    monkeypatch.setattr(git, "_acquire_git_lock", partial(git_plumbing._acquire_git_lock, timeout_sec=1))
    binding = SimpleNamespace(operation="vcs", root="active_workspace", base_path=target, target_path=target,
                              repository_path=target)
    held = git_plumbing._acquire_git_lock(ctx, repo_dir=target, timeout_sec=1)
    try:
        with pytest.raises(TimeoutError):
            git_vcs_ops._revert_commit(ctx, head, confirm=True, _resolved_binding=binding)
        assert run(target, "rev-parse", "HEAD") == head  # nothing landed under another holder
    finally:
        git_plumbing._release_git_lock(held)
    serving_lock = git_plumbing._acquire_git_lock(ctx, timeout_sec=1)
    try:
        assert "New revert commit created" in git_vcs_ops._revert_commit(ctx, head, confirm=True,
                                                                           _resolved_binding=binding)
        assert run(target, "rev-parse", "HEAD") != head
    finally:
        git_plumbing._release_git_lock(serving_lock)


def test_a_body_copy_younger_than_its_task_baseline_is_a_moved_base_that_names_its_paths(tmp_path):
    from ouroboros.mutation_attribution import resolve_attributed_git_paths, selection_disclosure
    from ouroboros.task_results import write_task_result

    repo = _repo(tmp_path / "copy")
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "base"], check=True)
    (repo / "a.txt").write_text("task work\n", encoding="utf-8")
    drive = tmp_path / "drive"
    write_task_result(drive, "t1", "running", mutation_evidence={"baseline": {"surfaces": []}})
    selected, _facts, error = resolve_attributed_git_paths(drive, "t1", repo, None, candidate_root=True)
    assert selected == [] and "baseline_surface_missing" in error and 'paths=["a.txt"]' in error
    assert "moved base" not in error and "with this gap disclosed" in error
    selected, facts, error = resolve_attributed_git_paths(drive, "t1", repo, ["a.txt"], candidate_root=True)
    assert not error and selected == ["a.txt"] and facts["selection"]["gaps"] == ["baseline_surface_missing"]
    disclosed = selection_disclosure(facts["selection"])  # the reviewers read a missing baseline, not a moved HEAD
    assert "no recorded task baseline" in disclosed and "HEAD moved" not in disclosed
    moved = selection_disclosure({"gaps": ["baseline_stale"], "base_commit": "abcdef1234567890"})
    assert "HEAD moved from the recorded base abcdef123456" in moved and "no recorded task baseline" not in moved
    selected, _facts, error = resolve_attributed_git_paths(drive, "t1", repo, ["a.txt"])  # not a body copy: as before
    assert selected == [] and "GIT_ATTRIBUTION_BLOCKED" in error
