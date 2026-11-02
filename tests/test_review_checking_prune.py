"""The prune retires review checking copies only on positive evidence that their run ended.

Custody is a stub (``task_snapshot`` answers which runs executed in which copy); copies,
receipts, locks and deletions are real. Runs with unknown execution, failed captures and
their frozen views stay past retention; complete captures release their copies.
"""
from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody_current as current
from ouroboros import review_checking as rc
from ouroboros import review_checking_prune as rp
from ouroboros import server_maintenance as sm
from ouroboros.utils import utc_now_iso
from tests.test_review_checkout_lifecycle import (
    EDIT, LOG, TASK, _boom, _frozen, _seat, _thread, _work, directory_source,
)

LATER = 400 * 86400  # past any retention period


def _custody(monkeypatch, *, running=(), settled=(), pending=(), unreadable=False, calls=None, started=None):
    """Runs that executed in ``running`` (not settled) or ``settled`` copies (started now, or at
    ``started``), and invocations pending for ``pending`` copies, whatever the task id."""
    runs = {f"run-{index}": SimpleNamespace(execution_root=str(path), settled=False, terminal_state="", started_ts="")
            for index, path in enumerate(running)}
    runs.update({f"done-{index}": SimpleNamespace(execution_root=str(path), settled=True, terminal_state="succeeded",
                                                  started_ts=started or utc_now_iso())
                 for index, path in enumerate(settled)})

    def snapshot(_root, task_id):
        if calls is not None:
            calls.append(task_id)
        return {"runs": dict(runs), "pending": [{"execution_root": str(path)} for path in pending]}

    monkeypatch.setattr(current, "unreadable", lambda _root: unreadable)
    monkeypatch.setattr(current, "task_snapshot", snapshot)


def test_only_positive_evidence_releases_a_copy(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    running, pending, done, unknown = (_seat(root, source, f"op-{name}")
                                       for name in ("running", "pending", "done", "unknown"))
    skill = _seat(root, source, "op-skill", task="api_skill_review")  # an owner that is never a live task
    _custody(monkeypatch, running=[running, skill], pending=[pending], settled=[done])
    report = rp.prune_review_checkouts(root, set())
    assert running.is_dir() and pending.is_dir() and skill.is_dir() and unknown.is_dir() and not done.exists()
    assert (report["released"], report["errors"]) == (1, [])
    receipt = rc._receipt(root, TASK, "op-done")
    assert receipt["status"] == "retained"
    assert (Path(receipt["evidence_dir"]) / "files" / "check.py").read_bytes() == EDIT
    rp.prune_review_checkouts(root, set(), now=time.time() + LATER)  # a run that never started
    assert unknown.is_dir() and running.is_dir() and pending.is_dir() and skill.is_dir()


def test_custody_is_read_again_for_each_seat(tmp_path, monkeypatch):
    """A run registered after the pass began is seen when the pass reaches its seat."""
    root, source = _frozen(tmp_path)
    late = [_seat(root, source, f"op-{index}") for index in range(3)]
    calls = []
    _custody(monkeypatch, pending=late, calls=calls)
    rp.prune_review_checkouts(root, set())
    assert len(calls) == 3 and all(area.is_dir() for area in late)


def test_a_seat_a_worker_holds_is_skipped(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    done = _seat(root, source, "op-done")
    _custody(monkeypatch, settled=[done])
    with rc._seat_locked(done.parent):
        report = rp.prune_review_checkouts(root, set())
    assert report["busy"] == 1 and done.is_dir()
    assert rp.prune_review_checkouts(root, set())["released"] == 1 and not done.exists()


def test_custody_that_cannot_be_read_removes_nothing(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-done")
    _custody(monkeypatch, settled=[area], unreadable=True)
    report = rp.prune_review_checkouts(root, set(), views_since=time.time(), now=time.time() + LATER)
    assert report["skipped"] == "custody_unreadable" and "views_done" not in report
    assert area.is_dir() and Path(source["view"]).is_dir()


def test_a_copy_an_older_host_kept_after_its_receipt_is_released(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-old")
    evidence = area.parent / "evidence"
    evidence.mkdir()
    older = {"status": "retained", "evidence_dir": str(evidence), "gaps": [], "area": str(area),
             "refresh_safe": False, "retention_reason": "writer_quiescence_unverified"}
    (evidence / "summary.json").write_text(json.dumps(older), encoding="utf-8")
    _custody(monkeypatch)
    assert rp.prune_review_checkouts(root, set())["released"] == 1 and not area.exists()
    assert json.loads((evidence / "summary.json").read_text(encoding="utf-8")) == older


def test_a_failed_capture_retries_and_keeps_unique_work_beyond_retention(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-failed")
    (area / rc.EVIDENCE_DIRNAME / "summary.json").write_bytes(b'{"test": "failed"}')
    _custody(monkeypatch, settled=[area])
    compare = rc._directory_delta
    monkeypatch.setattr(rc, "_directory_delta", _boom)
    assert rp.prune_review_checkouts(root, set())["capture_failed"] == 1 and area.is_dir()
    assert rp.prune_review_checkouts(root, set(), retention_days=7)["capture_failed"] == 1
    later = rp.prune_review_checkouts(root, set(), retention_days=7, now=time.time() + 8 * 86400)
    assert later["removed"] == 0 and area.is_dir()
    assert rc._receipt(root, TASK, "op-failed")["status"] == "capture_failed"
    assert len(list(area.parent.glob("evidence*"))) == 1, "identical failures reuse their retained generation"
    monkeypatch.setattr(rc, "_directory_delta", compare)
    assert rp.prune_review_checkouts(root, set())["released"] == 1
    receipt = rc._receipt(root, TASK, "op-failed")
    assert receipt["status"] == "retained" and not area.exists()
    assert (Path(receipt["evidence_dir"]) / "files" / "check.py").read_bytes() == EDIT


def test_leftovers_without_a_marker_expire_and_older_staging_goes_at_once(tmp_path, monkeypatch):
    seats = tmp_path / "data" / "state" / "review_checkouts" / "seats"
    bare, documented = seats / "bare", seats / "documented"
    for home in (bare, documented):
        (home / "area").mkdir(parents=True)
        (home / "area" / "big.bin").write_bytes(b"x" * 4096)
    (bare / ".next-0123").mkdir()  # an older body's first thread materialization: a pristine copy
    (documented / "evidence").mkdir()
    _custody(monkeypatch)
    rp.prune_review_checkouts(tmp_path / "data", set(), views_since=time.time())
    assert (bare / "area").is_dir() and not (bare / ".next-0123").exists() and (documented / "area").is_dir()
    rp.prune_review_checkouts(tmp_path / "data", set(), views_since=time.time(), now=time.time() + LATER)
    assert not (bare / "area").exists()
    assert (documented / "evidence").is_dir() and not (documented / "area").exists()


@pytest.mark.parametrize("torn", ["{torn", "[]"], ids=["not-json", "not-an-object"])
def test_a_marker_that_cannot_be_read_keeps_the_copy_and_every_view(tmp_path, monkeypatch, torn):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-torn")
    (area.parent / "checking.json").write_text(torn, encoding="utf-8")
    finished = _seat(root, source, "op-done")
    _custody(monkeypatch, settled=[finished])
    report = rp.prune_review_checkouts(root, set(), views_since=time.time() + 1)
    assert area.is_dir() and report["errors"] and "views_done" not in report
    assert Path(source["view"]).is_dir()
    # Unknown original sources stay recoverable even after the retention period.
    later = rp.prune_review_checkouts(root, set(), views_since=time.time() + 1, now=time.time() + LATER)
    assert "views_done" not in later and area.is_dir() and Path(source["view"]).exists()
    assert later["errors"] and rp.prune_review_checkouts(root, set(), now=time.time() + LATER)["errors"]


def test_a_thread_copy_lives_while_its_task_is_live(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _thread(root, source, "turn-1")
    _work(area)
    (area.parent / ".next-0123").mkdir()  # an interrupted refresh
    asked = []
    _custody(monkeypatch, settled=[area])
    rp.prune_review_checkouts(root, lambda: asked.append(True) or {TASK})
    assert area.is_dir() and asked == [True]
    rp.prune_review_checkouts(root, set())
    assert not area.exists() and not (area.parent / ".next-0123").exists()
    assert rc._receipt(root, TASK, "turn-1")["status"] == "retained"
    assert _thread(root, source, "turn-2") == area and (area / "check.py").read_bytes() == b"# candidate\n"


def test_a_turn_prepared_before_the_pass_keeps_its_copy(tmp_path, monkeypatch):
    """Turn 1 ended; turn 2 was prepared but has no receipt or custody yet: nothing proves it ended."""
    root, source = _frozen(tmp_path)
    area = _thread(root, source, "turn-1")
    turn_one_started = utc_now_iso()
    _work(area)
    assert rc.capture_seat_evidence(root, TASK, "turn-1")["status"] == "retained"
    time.sleep(0.01)
    assert _thread(root, source, "turn-2") == area
    (area / "turn-2.log").write_bytes(LOG)
    _custody(monkeypatch, settled=[area], started=turn_one_started)  # only turn 1's run, settled
    rp.prune_review_checkouts(root, set())
    assert (area / "turn-2.log").read_bytes() == LOG


def test_views_used_since_the_process_began_are_kept(tmp_path, monkeypatch):
    root, subject = tmp_path / "data", tmp_path / "subject"
    subject.mkdir()
    (subject / "a.txt").write_bytes(b"1")
    reused = directory_source(subject, root)
    done = _seat(root, reused, "op-reused-done")
    (subject / "a.txt").write_bytes(b"2")
    finished = directory_source(subject, root)
    gone = _seat(root, finished, "op-done")
    views = root / "state" / "review_checkouts" / "views"
    (views / ".discarded-x.1234").mkdir()
    _custody(monkeypatch, settled=[done, gone])
    rp.prune_review_checkouts(root, set())  # the periodic pass releases copies, never views
    assert not done.exists() and not gone.exists()
    since = time.time()
    time.sleep(0.05)
    os.utime(Path(reused["view"]))  # a new wave's freeze reused it after the process began
    report = rp.prune_review_checkouts(root, set(), views_since=since)
    assert report["views_done"] is True
    assert Path(reused["view"]).is_dir() and not Path(finished["view"]).exists()
    assert not (views / ".discarded-x.1234").exists() and (views.parent / rc.VIEWS_LOCK_NAME).exists()


def test_an_unnamed_view_goes_only_unused_and_past_retention(tmp_path, monkeypatch):
    root, subject = tmp_path / "data", tmp_path / "subject"
    subject.mkdir()
    (subject / "a.txt").write_bytes(b"1")
    inventory = Path(directory_source(subject, root)["view"])  # no seat names it
    _custody(monkeypatch)
    rp.prune_review_checkouts(root, set(), views_since=time.time() + LATER)
    assert inventory.is_dir()  # unused since the process began, but younger than the retention period
    rp.prune_review_checkouts(root, set(), views_since=time.time() + LATER, now=time.time() + LATER)
    assert not inventory.exists()


def test_the_views_pass_waits_for_a_freeze_holding_the_views_lock(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    gone = _seat(root, source, "op-done")
    _custody(monkeypatch, settled=[gone])
    views = root / "state" / "review_checkouts" / "views"
    with rc._seat_locked(views.parent, name=rc.VIEWS_LOCK_NAME):
        busy = rp.prune_review_checkouts(root, set(), views_since=time.time() + 1)
    assert busy["skipped"] == "views_busy" and "views_done" not in busy and Path(source["view"]).is_dir()
    assert rp.prune_review_checkouts(root, set(), views_since=time.time() + 1)["views_done"] is True
    assert not Path(source["view"]).exists()


def test_a_reuse_touches_the_view(tmp_path):
    root, subject = tmp_path / "data", tmp_path / "subject"
    subject.mkdir()
    (subject / "a.txt").write_bytes(b"1")
    view = Path(directory_source(subject, root)["view"])
    os.utime(view, (1, 1))
    assert directory_source(subject, root)["view"] == str(view)
    assert view.stat().st_mtime > time.time() - 60


def test_stop_ends_the_pass_before_the_next_seat(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    done = _seat(root, source, "op-done")
    _custody(monkeypatch, settled=[done])
    report = rp.prune_review_checkouts(root, set(), views_since=time.time() + 1, stop=lambda: True)
    assert report["skipped"] == "stopped" and done.is_dir() and Path(source["view"]).is_dir()


def test_the_periodic_pass_reads_only_seats_that_still_hold_a_copy(tmp_path, monkeypatch):
    """Released seats keep their evidence; a pass every ten minutes must not reread them all."""
    root, source = _frozen(tmp_path)
    _seat(root, source, "op-done")
    assert rc.terminal_seat_evidence(root, TASK, "op-done", {"summary": {"state": "succeeded"}})["status"] == "retained"
    read = []
    real = rp._view_containers
    monkeypatch.setattr(rp, "_view_containers", lambda *args: read.append(True) or real(*args))
    _custody(monkeypatch)
    rp.prune_review_checkouts(root, set())
    assert read == []
    rp.prune_review_checkouts(root, set(), views_since=time.time())
    assert read == [True]


def test_stray_files_never_block_the_views_pass(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    done = _seat(root, source, "op-done")
    views = root / "state" / "review_checkouts" / "views"
    for folder in (done.parent.parent, views):
        (folder / ".DS_Store").write_bytes(b"finder")
    _custody(monkeypatch, settled=[done])
    report = rp.prune_review_checkouts(root, set(), views_since=time.time() + 1, now=time.time() + LATER)
    assert report["views_done"] is True and report["errors"] == [] and not Path(source["view"]).exists()
    assert (views / ".DS_Store").read_bytes() == b"finder" and not list(views.glob(".discarded-*"))


def test_a_busy_seat_defers_the_views_pass(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    busy = _seat(root, source, "op-busy")
    _custody(monkeypatch, settled=[busy])
    with rc._seat_locked(busy.parent):  # a worker preparing or capturing it
        report = rp.prune_review_checkouts(root, set(), views_since=time.time() + 1)
    assert report["busy"] == 1 and report["skipped"] == "views_busy" and "views_done" not in report
    assert busy.is_dir() and Path(source["view"]).is_dir()
    assert rp.prune_review_checkouts(root, set(), views_since=time.time() + 1)["views_done"] is True
    assert not busy.exists() and not Path(source["view"]).exists()


def test_a_settled_run_whose_capture_saves_nothing_keeps_its_copy(tmp_path, monkeypatch):
    """A full volume: the capture writes no receipt, and an old marker is no reason to delete the work."""
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-old")
    _custody(monkeypatch, settled=[area])
    real = rc.atomic_write_json
    with monkeypatch.context() as patch:
        patch.setattr(rc, "atomic_write_json", lambda path, *args, **kwargs: _boom() if Path(
            path).name == "summary.json" else real(path, *args, **kwargs))
        report = rp.prune_review_checkouts(root, set(), now=time.time() + LATER)
    assert report["capture_failed"] == 1 and (area / "check.py").read_bytes() == EDIT
    assert rp.prune_review_checkouts(root, set(), now=time.time() + LATER)["released"] == 1 and not area.exists()


def test_a_reused_failed_copy_is_not_released_on_its_previous_run(tmp_path, monkeypatch):
    """A re-dispatch reuses the copy its failed capture kept; until the new run registers,
    custody shows only the previous run, which proves nothing about this use."""
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    previous_run = utc_now_iso()
    with monkeypatch.context() as patch:
        patch.setattr(rc, "_directory_delta", _boom)
        assert rc.capture_seat_evidence(root, TASK, "op-1")["status"] == "capture_failed"
    time.sleep(0.01)
    assert Path(rc.prepare_seat_area(root, TASK, "op-1", source)) == area
    _custody(monkeypatch, settled=[area], started=previous_run)
    rp.prune_review_checkouts(root, set())
    assert (area / "check.py").read_bytes() == EDIT


def test_a_failed_capture_stays_recoverable_after_the_data_root_moves(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-failed")
    monkeypatch.setattr(rc, "_directory_delta", _boom)
    assert rc.capture_seat_evidence(root, TASK, "op-failed")["status"] == "capture_failed"
    moved = tmp_path / "bigger-disk"
    root.rename(moved)  # the receipt's stored evidence_dir now names nothing
    area = moved / area.relative_to(root)
    _custody(monkeypatch, settled=[area])
    assert rp.prune_review_checkouts(moved, set(), now=time.time() + LATER)["removed"] == 0 and area.is_dir()
    assert (area / "check.py").read_bytes() == EDIT


def test_an_old_copy_whose_capture_fails_keeps_its_unique_bytes(tmp_path, monkeypatch):
    """Neither the materialization age nor the failure's age expires uncaptured work."""
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-old")
    _custody(monkeypatch, settled=[area])
    real = rp._last_use
    monkeypatch.setattr(rp, "_last_use", lambda path: 0.0 if path.name == "checking.json" else real(path))
    monkeypatch.setattr(rc, "_directory_delta", _boom)
    assert rp.prune_review_checkouts(root, set(), retention_days=7)["capture_failed"] == 1 and area.is_dir()
    rp.prune_review_checkouts(root, set(), retention_days=7, now=time.time() + 8 * 86400)
    assert (area / "check.py").read_bytes() == EDIT


def test_a_failing_ownership_read_keeps_every_view_its_seat_names(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _thread(root, source, "turn-1")
    (tmp_path / "subject" / "check.py").write_bytes(b"# another wave\n")
    other = directory_source(tmp_path / "subject", root)
    done = _seat(root, other, "op-done")
    _custody(monkeypatch, settled=[done])
    monkeypatch.setattr(rp, "_last_use", lambda _path: 0.0)  # every marker and view looks long unused

    def unreadable():
        raise ValueError("live ownership unreadable")

    report = rp.prune_review_checkouts(root, unreadable, views_since=time.time() + 1)
    assert report["errors"] and area.is_dir() and Path(source["view"]).is_dir()
    # Its views are known, so it holds back only those: the finished wave's view still goes.
    assert report["views_done"] is True and not done.exists() and not Path(other["view"]).exists()


def test_a_refresh_staging_left_behind_goes_in_the_periodic_pass(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _thread(root, source, "turn-1")
    staging = area.parent.parent / f".next-{area.parent.name}.0123"
    staging.mkdir()
    (staging / "big.bin").write_bytes(b"x" * 4096)
    _custody(monkeypatch)
    with rc._seat_locked(area.parent):  # a refresh building it right now
        rp.prune_review_checkouts(root, {TASK})
    assert staging.is_dir()
    rp.prune_review_checkouts(root, {TASK})
    assert not staging.exists() and area.is_dir()


def test_a_second_concurrent_pass_skips(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-done")
    _custody(monkeypatch, settled=[area])
    with rp._PASS_LOCK:
        assert rp.prune_review_checkouts(root, set())["skipped"] == "pass_in_progress"
    assert area.is_dir()


def test_repeated_folder_reviews_leave_only_evidence(tmp_path, monkeypatch):
    """The 2026-10-16 class: 14 acceptance waves of two writable seats over a working folder
    left 14 folder views and 28 copies. The folder is now read in place: no wave copies it,
    and each seat keeps only what it publishes."""
    root, subject = tmp_path / "data", tmp_path / "subject"
    subject.mkdir()
    (subject / "model.bin").write_bytes(os.urandom(2 * 1024 * 1024))
    for wave in range(14):
        (subject / "notes.txt").write_text(f"wave {wave}", encoding="utf-8")
        source = rc.freeze_checking_source(subject, data_root=root)
        for seat in (f"op-{wave}-a", f"op-{wave}-b"):
            area = Path(rc.prepare_seat_area(root, TASK, seat, source))
            (area / rc.EVIDENCE_DIRNAME).mkdir()
            (area / rc.EVIDENCE_DIRNAME / "verdict.log").write_bytes(LOG)
            receipt = rc.terminal_seat_evidence(root, TASK, seat, {"summary": {"state": "succeeded"}})
            assert receipt["status"] == "retained" and not area.exists()
    checkouts = root / "state" / "review_checkouts"
    assert not (checkouts / "views").exists()
    assert sum(path.stat().st_size for path in checkouts.rglob("*") if path.is_file()) < 1024 * 1024

def test_the_custody_pass_runs_the_prune_and_owes_views_until_done(monkeypatch, tmp_path):
    calls, answers = [], [{"skipped": "views_busy"}, {"views_done": True}, {}]
    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sm, "_REVIEW_VIEWS_OWED", [True])
    monkeypatch.setattr(sm, "_startup_live_task_ids", lambda _root, include_pending=False: {"task-live"})
    monkeypatch.setattr(rp, "prune_review_checkouts", lambda root, live, **kwargs: calls.append(
        (root, live(), kwargs["views_since"])) or answers.pop(0))
    for _pass in range(3):
        sm._prune_review_checkouts()
    assert calls == [(tmp_path, {"task-live"}, sm._PROCESS_STARTED)] * 2 + [(tmp_path, {"task-live"}, None)]
    assert "_prune_review_checkouts(stop_event=stop_event)" in inspect.getsource(sm._run_periodic_custody_sweep)


def test_unreadable_live_ownership_prunes_nothing(monkeypatch, tmp_path):
    def unreadable(_root, include_pending=False):
        raise ValueError("startup pending ownership is unreadable")

    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sm, "_startup_live_task_ids", unreadable)
    monkeypatch.setattr(rp, "prune_review_checkouts", lambda *_args, **_kwargs: pytest.fail("pruned blind"))
    sm._prune_review_checkouts()
