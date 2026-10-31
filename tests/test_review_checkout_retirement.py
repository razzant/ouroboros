"""Review checkouts nothing could retire (owner 3A).

C8: an isolated checkout (``state/review_checkouts/<token>/repo``) that its wave kept for open custody,
or that a crash left, is retired on the 600 s custody pass once no review can still read it: a marked
one (``checkout.json``) when its task is terminal and not live, no active review custody names its
round, no live review operation runs for the task and no retained seat checks from it; an unmarked
older one under the same conditions only past the GC retention wait and only when a recorded round
names its task. C9: a dead controller's open review pointer is removed by the maintenance pass, never
settled; a live controller keeps it.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ouroboros import obligations, review_checking_prune as rp, review_operation
from ouroboros.force_remove import force_rmtree
from ouroboros.startup_migrations import prepare_startup_state
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.review_subject import CHECKOUT_MARKER, ReviewSubjectSpec, checkout_token, isolated_checkout

TASK = "t-reviewer"
RETRY_KEY = "review:round-1"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _registered(repo) -> set:
    """The worktree paths Git registers for REPO, as paths: Git prints them with forward slashes on Windows."""
    lines = _git(repo, "worktree", "list", "--porcelain").splitlines()
    return {pathlib.Path(line[len("worktree "):]) for line in lines if line.startswith("worktree ")}


def _rewrite(path: pathlib.Path, data: bytes) -> None:
    """Rewrite a file in place. Git for Windows hides a ``.git`` it creates, and Windows refuses a
    create-always open (``write_text``/``write_bytes``) of a hidden file with EACCES."""
    with path.open("r+b") as handle:
        handle.write(data)
        handle.truncate()


def _repo(path):
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    for key, value in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git(path, "config", key, value)
    (path / "a.txt").write_text("one\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "base")
    (path / "a.txt").write_text("two\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "proposal")
    return path


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _alive_controller() -> dict:
    from ouroboros.platform_layer import process_start_time

    return {"pid": os.getppid(), "birth": process_start_time(os.getppid()), "session": "another-process"}


@pytest.fixture
def subject(tmp_path):
    """A prepared data root, a foreign repository with a proposal, and a reviewer task context."""
    data = tmp_path / "data"
    prepare_startup_state(data)
    repo = _repo(tmp_path / "subject")
    system = _repo(tmp_path / "system")
    ctx = SimpleNamespace(repo_dir=str(system), drive_root=str(data), task_id=TASK, task_metadata=None)
    spec = ReviewSubjectSpec(root_kind="active_workspace", root=str(repo), kind="base..head", base="HEAD~1",
                             head="HEAD", governance_root=str(system), surface="change")
    return SimpleNamespace(data=data.resolve(), repo=repo, ctx=ctx, spec=spec,
                           name=checkout_token(RETRY_KEY, task_id=TASK))


def _retain(subject, *, task_status="completed"):
    """The wave of TASK's round exits with custody open: its checkout stays, marked."""
    with isolated_checkout(subject.ctx, subject.spec, retain=lambda: {"seats": ["late seat"]},
                           token=lambda _identity: subject.name) as frozen:
        checkout = pathlib.Path(frozen.checkout)
    write_task_result(subject.data, TASK, task_status, result="done")
    assert checkout.is_dir() and len(_git(subject.repo, "worktree", "list").splitlines()) == 2
    return checkout.parent


def _attempt(subject, status):
    """One review-state attempt row of TASK's round (``reviewing`` keeps custody active)."""
    path = subject.data / "state" / "advisory_review.json"
    path.write_text(json.dumps({"attempts": [{"task_id": TASK, "review_retry_key": RETRY_KEY, "status": status,
                                              "tool_name": "review_change", "repo_key": "subject",
                                              "ts": "2026-10-17T00:00:00+00:00"}]}), encoding="utf-8")


def _prune(subject, live=()):
    return rp.prune_review_checkouts(subject.data, set(live))


def test_the_marker_names_the_task_and_a_reused_checkout_keeps_it(subject):
    entry = _retain(subject)
    marker = json.loads((entry / CHECKOUT_MARKER).read_text(encoding="utf-8"))
    assert marker["task_id"] == TASK and marker["schema_version"] == 1 and marker["created_at"]
    with isolated_checkout(subject.ctx, subject.spec, retain=lambda: {"seats": ["late seat"]},
                           token=lambda _identity: subject.name) as again:  # a rejoin reads it in place
        assert pathlib.Path(again.checkout).parent == entry
    assert json.loads((entry / CHECKOUT_MARKER).read_text(encoding="utf-8")) == marker


def test_a_marked_checkout_no_review_can_read_is_removed_and_its_worktree_pruned(subject):
    entry = _retain(subject)
    report = _prune(subject)
    assert not entry.exists() and report["checkouts_removed"] == 1, report
    assert len(_git(subject.repo, "worktree", "list").splitlines()) == 1, "the subject's worktree entry is pruned"


@pytest.mark.parametrize("hold", ["task_live", "task_running", "custody_active", "seat_checking", "operation_live",
                                  "no_owner"])
def test_a_checkout_a_review_may_still_read_stays(subject, hold):
    entry = _retain(subject, task_status="running" if hold == "task_running" else "completed")
    live = {TASK} if hold == "task_live" else set()
    if hold == "custody_active":
        _attempt(subject, "reviewing")
    elif hold == "seat_checking":
        seat = subject.data / "state" / "review_checkouts" / "seats" / "seat-1"
        (seat / "area.noindex").mkdir(parents=True)
        (seat / "checking.json").write_text(json.dumps({"task_id": TASK, "operation_id": "op-1", "source": {
            "kind": "git_tree", "root": str(entry / "repo.noindex")}}), encoding="utf-8")
    elif hold == "operation_live":
        write_task_result(subject.data, TASK, "completed", review_operations={"op-live": {
            "state": "dispatched", "surface": "multi_model_review", "retry_key": RETRY_KEY,
            "controller": _alive_controller()}})
    elif hold == "no_owner":
        marker = entry / CHECKOUT_MARKER
        marker.write_text(json.dumps({**json.loads(marker.read_text(encoding="utf-8")), "task_id": ""}),
                          encoding="utf-8")
    report = _prune(subject, live)
    assert entry.is_dir() and report["checkouts_removed"] == 0 and not report["errors"], report


def test_an_unmarked_checkout_goes_only_past_the_wait_when_a_recorded_round_names_its_task(subject):
    entry = _retain(subject)
    (entry / CHECKOUT_MARKER).unlink()  # made by a body before the marker
    assert _prune(subject)["checkouts_removed"] == 0 and entry.is_dir(), "no recorded round: owner unknown"
    _attempt(subject, "succeeded")
    assert _prune(subject)["checkouts_removed"] == 0 and entry.is_dir(), "younger than the retention wait"
    later = rp.prune_review_checkouts(subject.data, set(), now=time.time() + 30 * 86400)  # the wait has passed
    assert later["checkouts_removed"] == 1 and not entry.exists(), later


def test_an_unreadable_review_state_removes_no_checkout(subject):
    entry = _retain(subject)
    (subject.data / "state" / "advisory_review.json").write_text("{torn", encoding="utf-8")
    report = _prune(subject)
    assert entry.is_dir() and report["checkouts_removed"] == 0 and report["errors"], report


def test_a_dead_controllers_pointer_is_removed_never_settled(tmp_path):
    """Both directions: an open pointer whose controller is proven dead is removed (the retired task
    acceptance's open pointer too: nothing settles it any more), a live controller's stays, and a closed
    pointer is history the pass never touches."""
    prepare_startup_state(tmp_path)
    dead = {"pid": _dead_pid(), "birth": "gone", "session": "gone"}
    pointers = {owner: {"state": state, "surface": surface, "retry_key": f"{surface}:{owner}", "controller": controller}
                for owner, surface, state, controller in (
                    ("op-dead", "multi_model_review", "dispatched", dead),
                    ("op-live", "plan_review", "dispatched", _alive_controller()),
                    ("op-acceptance", "task_acceptance", "retained", dead),
                    ("op-closed", "task_acceptance", "closed", dead))}
    write_task_result(tmp_path, TASK, "completed", result="done", review_operations=pointers)
    assert obligations.review_collection_pending(load_task_result(tmp_path, TASK)), "an open liveness pointer is found"

    report = review_operation.retire_dead_review_pointers(tmp_path)
    assert report["retired"] == [{"task_id": TASK, "owner_id": "op-dead"},
                                 {"task_id": TASK, "owner_id": "op-acceptance"}], report
    left = load_task_result(tmp_path, TASK)["review_operations"]
    assert set(left) == {"op-live", "op-closed"}, "a live controller keeps its pointer, a closed one stays history"
    assert left["op-closed"] == pointers["op-closed"]
    assert review_operation.retire_dead_review_pointers(tmp_path)["retired"] == []
    assert obligations.review_collection_pending(load_task_result(tmp_path, TASK)), "the live pointer still holds"


def test_a_row_an_older_body_flagged_with_nothing_open_is_written_back_and_packs(tmp_path):
    """Both directions: a terminal row whose stored review_collection_pending is 1 only because an older body
    flagged it for its acceptance collection (here a closed acceptance pointer) is written back to 0 by one
    pass and becomes packable, its body untouched; a row with an open live pointer keeps its 1."""
    from ouroboros import store
    from ouroboros.evidence_packs import task_packable

    prepare_startup_state(tmp_path)
    closed = {"op-closed": {"state": "closed", "surface": "task_acceptance", "retry_key": "task_acceptance:closed",
                            "controller": {"pid": _dead_pid(), "birth": "gone", "session": "gone"}}}
    live = {"op-live": {"state": "dispatched", "surface": "plan_review", "retry_key": "plan_review:live",
                        "controller": _alive_controller()}}
    write_task_result(tmp_path, "old-flag", "completed", result="done", review_operations=closed)
    write_task_result(tmp_path, "open-review", "completed", result="done", review_operations=live)
    with store.open_store(tmp_path, write=True) as txn:  # the column as the older body derived it
        txn.conn.execute("UPDATE tasks SET review_collection_pending=1 WHERE task_id='old-flag'")

    def column(task):
        with store.open_store(tmp_path, write=False) as txn:
            return txn.row(task, "review_collection_pending")

    assert (column("old-flag"), column("open-review")) == (1, 1)
    report = review_operation.retire_dead_review_pointers(tmp_path)
    assert report["reclassified"] == ["old-flag"] and report["retired"] == [], report
    assert (column("old-flag"), column("open-review")) == (0, 1)
    with store.open_store(tmp_path, write=False) as txn:
        assert task_packable(txn, "old-flag") and not task_packable(txn, "open-review")
    assert review_operation.retire_dead_review_pointers(tmp_path)["reclassified"] == []
    assert load_task_result(tmp_path, "old-flag")["review_operations"] == closed


def test_checkout_retirement_targets_only_its_registration_and_keeps_an_offline_owner_worktree(subject):
    entry = _retain(subject)
    offline = subject.repo.parent / "owners-offline-checkout"
    _git(subject.repo, "worktree", "add", "--detach", str(offline), "HEAD")
    offline.rename(offline.with_name("temporarily-unmounted"))
    before = _registered(subject.repo)
    assert offline in before and entry / "repo.noindex" in before

    report = _prune(subject)
    after = _registered(subject.repo)
    assert report["checkouts_removed"] == 1 and not entry.exists(), report
    assert entry / "repo.noindex" not in after
    assert offline in after, "another owner's unavailable checkout must retain its registration"


def test_checkout_retirement_keeps_a_git_locked_checkout_until_its_registration_can_be_removed(subject, monkeypatch):
    from ouroboros import server_maintenance as sm

    entry = _retain(subject)
    _git(subject.repo, "worktree", "lock", "--reason", "reader retains this copy", str(entry / "repo.noindex"))
    with pytest.raises(subprocess.CalledProcessError) as refusal:
        _git(subject.repo, "worktree", "remove", "--force", str(entry / "repo.noindex"))
    message = refusal.value.stderr.strip()
    assert "reader retains this copy" in message
    monkeypatch.setattr(sm, "DATA_DIR", subject.data)
    monkeypatch.setattr(sm, "_REVIEW_VIEWS_OWED", [False])
    monkeypatch.setattr(sm, "_startup_live_task_ids", lambda *_args, **_kwargs: set())
    sm._prune_review_checkouts()
    events = [json.loads(line) for line in (subject.data / "logs" / "events.jsonl").read_text().splitlines()]
    report = [event for event in events if event["type"] == "review_checkout_prune"][-1]["report"]
    assert report["errors"] == [f"{entry.name}: Git worktree removal failed: {message}"]
    assert report["checkouts_removed"] == 0
    assert (entry / "repo.noindex" / "a.txt").read_text() == "two\n"
    _git(subject.repo, "worktree", "unlock", str(entry / "repo.noindex"))
    report = _prune(subject)
    assert not report["errors"] and report["checkouts_removed"] == 1 and not entry.exists()


@pytest.mark.parametrize("replacement", [False, True], ids=["deleted", "recloned"])
def test_checkout_of_a_deleted_subject_retires_only_after_review_custody_ends(subject, replacement):
    entry = _retain(subject)
    force_rmtree(subject.repo)  # its Git objects are read-only (Windows refuses a plain rmtree)
    assert not subject.repo.exists()
    if replacement:
        _repo(subject.repo)  # the new repository has no registration for this old checkout
    _attempt(subject, "reviewing")
    held = _prune(subject)
    assert held["checkouts_kept"] == 1 and not held["errors"] and entry.is_dir(), held

    _attempt(subject, "succeeded")
    report = _prune(subject)
    assert not report["errors"] and report["checkouts_removed"] == 1 and not entry.exists(), report
    if replacement:
        assert len(_git(subject.repo, "worktree", "list").splitlines()) == 1
        assert (subject.repo / "a.txt").read_text() == "two\n"


@pytest.mark.parametrize("damage", ["common_dir", "git_pointer"])
def test_existing_broken_git_metadata_keeps_the_checkout_and_reports_the_git_error(subject, damage):
    entry = _retain(subject)
    gitfile = entry / "repo.noindex" / ".git"
    registration = pathlib.Path(gitfile.read_text().removeprefix("gitdir: ").strip())
    metadata = registration / "commondir" if damage == "common_dir" else gitfile
    original = metadata.read_bytes()
    _rewrite(metadata, b"unavailable-metadata\n")
    with pytest.raises(subprocess.CalledProcessError) as refusal:
        _git(entry / "repo.noindex", "rev-parse", "--git-common-dir")

    report = _prune(subject)
    assert report["errors"] == [f"{entry.name}: Git common directory unavailable: {refusal.value.stderr.strip()}"]
    assert report["checkouts_removed"] == 0 and (entry / "repo.noindex" / "a.txt").read_text() == "two\n"
    assert registration.is_dir(), "a Git refusal alone does not prove the registration is gone"
    _rewrite(metadata, original)
    report = _prune(subject)
    assert not report["errors"] and report["checkouts_removed"] == 1 and not entry.exists(), report


def test_published_rejoin_custody_keeps_the_copy_and_a_retired_rejoin_rebuilds_the_frozen_tree(subject):
    entry = _retain(subject)
    with isolated_checkout(subject.ctx, subject.spec, retain=lambda: {"seats": ["rejoined"]},
                           token=lambda _identity: subject.name) as rejoined:
        _attempt(subject, "reviewing")  # published before the retirement judgment
        report = _prune(subject)
        assert report["checkouts_removed"] == 0 and report["checkouts_kept"] == 1, report
        assert pathlib.Path(rejoined.checkout) == entry / "repo.noindex"
        assert (entry / "repo.noindex" / "a.txt").read_text() == "two\n"
        tree = rejoined.tree_sha
    _attempt(subject, "succeeded")
    assert _prune(subject)["checkouts_removed"] == 1 and not entry.exists()
    with isolated_checkout(subject.ctx, subject.spec, token=lambda _identity: subject.name) as rebuilt:
        assert pathlib.Path(rebuilt.checkout) == entry / "repo.noindex"
        assert (entry / "repo.noindex" / "a.txt").read_text() == "two\n"
        assert rebuilt.tree_sha == tree == _git(rebuilt.checkout, "write-tree")
    assert not entry.exists()
