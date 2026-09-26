"""ibl-00615dbd1a16: `review_evidence.current_repo.stale_reason` / `.stale_ts`
must be scoped to the querying task. On the shared /opt/ouroboros tree a task
that recorded its own stale marker used to leak into every other task's
`current_repo` panel (only `last_stale_repo_key` gated it). A `last_stale_task_id`
now gates the displayed marker; an empty stored id (legacy / unattributed)
still matches every caller so pre-existing records stay effective."""
from __future__ import annotations

import pathlib

from ouroboros.review_evidence import collect_review_evidence
from ouroboros.review_state import (
    AdvisoryRunRecord,
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


def _seed_state_with_stale(drive_root, repo_key, *, stale_task_id):
    st = AdvisoryReviewState()
    st.add_run(AdvisoryRunRecord(
        snapshot_hash="deadbeef", commit_message="m", status="fresh",
        ts="2026-08-31T00:00:00", repo_key=repo_key,
    ))
    st.mark_repo_stale(
        repo_key=repo_key, reason_ts="2026-08-31T12:00:00",
        reason="edit_text mutated the worktree", stale_repo_key=repo_key,
        stale_task_id=stale_task_id,
    )
    save_state(drive_root, st)


def test_stale_marker_hidden_from_a_different_task(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    repo = _repo(tmp_path)
    rk = make_repo_key(repo)
    _seed_state_with_stale(drive, rk, stale_task_id="taskA")

    ev = collect_review_evidence(drive, task_id="taskB", repo_dir=repo)
    assert ev["current_repo"]["stale_reason"] == ""
    assert ev["current_repo"]["stale_ts"] == ""


def test_stale_marker_visible_to_the_task_that_set_it(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    repo = _repo(tmp_path)
    rk = make_repo_key(repo)
    _seed_state_with_stale(drive, rk, stale_task_id="taskA")

    ev = collect_review_evidence(drive, task_id="taskA", repo_dir=repo)
    assert ev["current_repo"]["stale_reason"] == "edit_text mutated the worktree"
    assert ev["current_repo"]["stale_ts"] == "2026-08-31T12:00:00"


def test_legacy_unattributed_stale_marker_still_shows_for_everyone(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    repo = _repo(tmp_path)
    rk = make_repo_key(repo)
    _seed_state_with_stale(drive, rk, stale_task_id="")

    for tid in ("taskA", "taskB", ""):
        ev = collect_review_evidence(drive, task_id=tid, repo_dir=repo)
        assert ev["current_repo"]["stale_reason"] == "edit_text mutated the worktree", tid


def test_stale_task_id_round_trips_through_save_load(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    st = AdvisoryReviewState()
    # mark_repo_stale only records the marker when there is an invalidatable run.
    st.add_run(AdvisoryRunRecord(
        snapshot_hash="0ff1ce", commit_message="m", status="fresh",
        ts="2026-08-31T08:00:00", repo_key="k",
    ))
    st.mark_repo_stale(
        repo_key="k", reason_ts="2026-08-31T09:00:00", reason="r",
        stale_repo_key="k", stale_task_id="t-123",
    )
    save_state(drive, st)
    reloaded = load_state(drive)
    assert reloaded.last_stale_task_id == "t-123"


def test_invalidate_advisory_after_mutation_records_the_task_id(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    repo = _repo(tmp_path)
    st = AdvisoryReviewState()
    st.add_run(AdvisoryRunRecord(
        snapshot_hash="cafef00d", commit_message="m", status="fresh",
        ts="2026-08-31T00:00:00", repo_key=make_repo_key(repo),
    ))
    save_state(drive, st)

    invalidate_advisory_after_mutation(
        pathlib.Path(drive),
        mutation_root=pathlib.Path(repo),
        changed_paths=["x.py"],
        mutating_task_id="t-abc",
    )
    assert load_state(drive).last_stale_task_id == "t-abc"


def test_add_run_clears_the_stale_task_id(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    repo = _repo(tmp_path)
    rk = make_repo_key(repo)
    st = AdvisoryReviewState()
    st.mark_repo_stale(
        repo_key=rk, reason_ts="2026-08-31T12:00:00", reason="r",
        stale_repo_key=rk, stale_task_id="taskA",
    )
    st.add_run(AdvisoryRunRecord(
        snapshot_hash="feedface", commit_message="m", status="fresh",
        ts="2026-08-31T13:00:00", repo_key=rk,
    ))
    assert st.last_stale_task_id == ""
