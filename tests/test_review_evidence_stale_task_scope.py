"""Stale provenance survives the original contributor's persistence paths.

Every task on a shared checkout sees its stale marker, with the writer named
separately. Unknown legacy ownership never changes freshness. The marker reads
against the checkout's preflight look; one no look precedes is history.
"""
from __future__ import annotations

import pathlib

from ouroboros import review_ledger
from ouroboros.review_evidence import collect_review_evidence
from ouroboros.review_state import (
    AdvisoryReviewState,
    invalidate_advisory_after_mutation,
    load_state,
    make_repo_key,
    save_state,
)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _seed_state_with_stale(drive_root, repo, *, stale_task_id):
    review_ledger.write_record(drive_root, {  # the checkout's preflight look
        "record_id": review_ledger.new_record_id(), "state": "settled", "surface": "preflight",
        "ts": "2026-08-31T00:00:00+00:00", "verdict": {"aggregate": "PASS"}, "subject": {"root": str(repo)}})
    st = AdvisoryReviewState()
    st.mark_look_stale(  # the marker a mutation writes over that look
        "2026-08-31T00:00:00", reason_ts="2026-08-31T12:00:00",
        reason="edit_text mutated the worktree", stale_repo_key=make_repo_key(repo),
        stale_task_id=stale_task_id,
    )
    save_state(drive_root, st)


def test_stale_marker_attributes_a_different_task_without_hiding_the_edit(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    repo = _repo(tmp_path)
    _seed_state_with_stale(drive, repo, stale_task_id="taskA")

    ev = collect_review_evidence(drive, task_id="taskB", repo_dir=repo)
    assert ev["current_repo"]["stale_reason"] == "edit_text mutated the worktree"
    assert ev["current_repo"]["stale_ts"] == "2026-08-31T12:00:00"
    assert ev["current_repo"]["stale_task_id"] == "taskA"
    assert ev["current_repo"]["stale_attribution"] == "other_task"


def test_stale_marker_visible_to_the_task_that_set_it(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    repo = _repo(tmp_path)
    _seed_state_with_stale(drive, repo, stale_task_id="taskA")

    ev = collect_review_evidence(drive, task_id="taskA", repo_dir=repo)
    assert ev["current_repo"]["stale_reason"] == "edit_text mutated the worktree"
    assert ev["current_repo"]["stale_ts"] == "2026-08-31T12:00:00"


def test_legacy_unattributed_stale_marker_still_shows_for_everyone(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    repo = _repo(tmp_path)
    _seed_state_with_stale(drive, repo, stale_task_id="")

    for tid in ("taskA", "taskB", ""):
        ev = collect_review_evidence(drive, task_id=tid, repo_dir=repo)
        assert ev["current_repo"]["stale_reason"] == "edit_text mutated the worktree", tid


def test_stale_task_id_round_trips_through_save_load(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    st = AdvisoryReviewState()
    st.mark_look_stale(
        "2026-08-31T08:00:00", reason_ts="2026-08-31T09:00:00", reason="r",
        stale_repo_key="k", stale_task_id="t-123",
    )
    save_state(drive, st)
    reloaded = load_state(drive)
    assert reloaded.last_stale_task_id == "t-123"


def test_invalidate_advisory_after_mutation_records_the_task_id(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    repo = _repo(tmp_path)
    review_ledger.write_record(drive, {  # the checkout's preflight look, which the mutation makes stale
        "record_id": review_ledger.new_record_id(), "state": "settled", "surface": "preflight",
        "ts": "2026-08-31T00:00:00+00:00", "verdict": {"aggregate": "PASS"}, "subject": {"root": str(repo)}})

    invalidate_advisory_after_mutation(
        pathlib.Path(drive),
        mutation_root=pathlib.Path(repo),
        changed_paths=["x.py"],
        mutating_task_id="t-abc",
    )
    assert load_state(drive).last_stale_task_id == "t-abc"
