"""A reviewer seat's scratch belongs to its copy, not to its task (#2706, #2707).

``review_session_preparation.seat_service_directories`` records the seat's scratch in the seat's own
home. The two places that release the seat's copy, its own terminal capture
(``review_checking.terminal_seat_evidence``) and the prune pass on the seat's custody proof
(``review_checking_prune.prune_review_checkouts``), hand the record to the custody trash, whose remover
deletes the scratch: while the task still runs, and for the Skills-page and marketplace pseudo task ids
that never get a task row and create no task drive. A seat with a pending or running run, a copy held
for its continuation while its task lives, and the owner's Keep keep the copy and its scratch. A seat
home that retires whole (Storage -> Free) releases its scratch through the same trash remover, and a
handoff a crash interrupted is finished when the prune reads the seat without its copy. Isolated roots
and a custody stub only; no engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ouroboros import owner_keep, review_checking as rc, review_checking_prune as rp, task_custody
from ouroboros.review_session_preparation import seat_service_directories
from ouroboros.subagents import delegated_run_shape
from ouroboros.workspace_preflight import OPEN_SCRATCH_DIR, SESSION_SCRATCH_DIRNAME
from tests._store_shared import seed_result
from tests.test_review_checking_prune import _custody
from tests.test_review_checkout_lifecycle import TASK, _frozen, _seat
from tests.test_session_service_directories import GRANTED, _Catalog, _synthetic_temp_parent  # noqa: F401


def _seated(root: Path, source, operation: str, task: str = TASK, access: str = "full"):
    """One writable seat with its copy and its scratch holding an environment."""
    area = _seat(root, source, operation, task)
    service = seat_service_directories(_Catalog(GRANTED), "route", delegated_run_shape(False, access), root, task,
                                       operation, str(area))
    scratch = Path(service.scratch)
    assert scratch.is_dir(), service
    (scratch / "env").mkdir()
    (scratch / "env" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    home = rc.seat_dir(root, task, operation)
    record = home / SESSION_SCRATCH_DIRNAME / f"seat-{home.name}" / "scratch.json"
    assert json.loads(record.read_text(encoding="utf-8"))["path"] == str(scratch)
    return area, scratch, record


def _trash(root: Path) -> list:
    trash = root / "state" / "custody_trash"
    return sorted(trash.iterdir()) if trash.is_dir() else []


def test_a_seats_scratch_goes_with_its_copy_at_its_own_terminal_while_its_task_runs(tmp_path):
    root, source = _frozen(tmp_path)
    area, scratch, record = _seated(root, source, "op-1")
    seed_result(root, TASK, {"status": "running"})  # the task is live and will be for days (#2706)
    assert not (root / "task_drives").exists() and not (root / OPEN_SCRATCH_DIR).exists()  # nothing held under the task
    receipt = rc.terminal_seat_evidence(root, TASK, "op-1", {"summary": {"state": "succeeded"}})
    assert receipt["status"] == "retained" and not area.exists()
    assert not record.exists() and not (record.parents[1]).exists()  # the home holds no scratch record any more
    [entry] = _trash(root)
    assert entry.name.startswith(f"{TASK}+scratch-")
    assert (entry / SESSION_SCRATCH_DIRNAME / record.parent.name / "scratch.json").is_file()
    assert scratch.is_dir()  # the delete is the maintenance child's, off the serving process
    report = task_custody.remove_custody_trash(root)
    assert report["removed"] == 1 and not scratch.exists()


def test_a_pending_or_running_run_keeps_the_copy_and_its_scratch_until_the_seat_proof_holds(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    pending, running = (_seated(root, source, name) for name in ("op-pending", "op-running"))
    _custody(monkeypatch, pending=[pending[0]], running=[running[0]])
    report = rp.prune_review_checkouts(root, set())
    assert report["released"] == 0 and _trash(root) == []
    for area, scratch, record in (pending, running):
        assert area.is_dir() and record.is_file() and (scratch / "env" / "pyvenv.cfg").is_file()
    _custody(monkeypatch, settled=[pending[0], running[0]])
    report = rp.prune_review_checkouts(root, set())
    assert (report["released"], report["errors"]) == (2, [])
    assert not any(record.exists() for _area, _scratch, record in (pending, running)) and len(_trash(root)) == 2
    assert task_custody.remove_custody_trash(root)["removed"] == 2
    assert not any(scratch.exists() for _area, scratch, _record in (pending, running))


def test_a_skills_page_seat_releases_on_its_proof_and_its_pseudo_owner_creates_no_drive(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area, scratch, record = _seated(root, source, "op-skill", task="api_skill_review")
    assert not (root / "task_drives").exists() and not (root / OPEN_SCRATCH_DIR).exists()
    _custody(monkeypatch, running=[area])
    rp.prune_review_checkouts(root, set())
    assert record.is_file() and area.is_dir()
    _custody(monkeypatch, settled=[area])
    report = rp.prune_review_checkouts(root, set())
    assert report["released"] == 1 and not record.exists()
    [entry] = _trash(root)
    assert entry.name.startswith("api_skill_review+scratch-")
    assert task_custody.remove_custody_trash(root)["removed"] == 1 and not scratch.exists()
    assert not (root / "task_drives").exists()  # a pseudo owner never had a drive: it holds nothing


@pytest.mark.parametrize("task_live", [True, False], ids=["task-live", "task-over"])
def test_a_copy_held_for_its_continuation_keeps_its_scratch_while_its_task_lives(tmp_path, monkeypatch, task_live):
    root, source = _frozen(tmp_path)
    area, scratch, record = _seated(root, source, "op-held")
    receipt = rc.terminal_seat_evidence(root, TASK, "op-held", {"summary": {"state": "interrupted"}}, hold=True)
    assert receipt["status"] == "retained" and area.is_dir() and record.is_file()  # kept for an explicit successor
    _custody(monkeypatch, settled=[area])
    report = rp.prune_review_checkouts(root, {TASK} if task_live else set())
    if task_live:
        assert report["released"] == 0 and area.is_dir() and record.is_file() and _trash(root) == []
    else:
        assert report["released"] == 1 and not area.exists() and not record.exists() and len(_trash(root)) == 1


def test_the_owners_keep_holds_the_seats_copy_and_scratch_and_names_the_scratch(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area, scratch, record = _seated(root, source, "op-kept")
    seed_result(root, TASK, {"status": "running"})
    places, _unknown, pins = owner_keep.inventory(root, TASK, {"status": "running", "task_id": TASK})
    assert str(scratch) in places and pins[str(scratch)] == str(record)
    assert owner_keep.set_owner_keep(root, TASK, keep=True, source={"basis": "test"})["keep"] is True
    receipt = rc.terminal_seat_evidence(root, TASK, "op-kept", {"summary": {"state": "succeeded"}})
    assert receipt["status"] == "retained" and area.is_dir() and record.is_file() and _trash(root) == []
    assert owner_keep.set_owner_keep(root, TASK, keep=False, source={"basis": "test"})["keep"] is False
    _custody(monkeypatch, settled=[area])
    report = rp.prune_review_checkouts(root, set())
    assert report["released"] == 1 and not record.exists()
    assert task_custody.remove_custody_trash(root)["removed"] == 1 and not scratch.exists()


def test_a_seat_home_retired_whole_releases_its_scratch_through_the_trash_remover(tmp_path):
    root, source = _frozen(tmp_path)
    area, scratch, record = _seated(root, source, "op-home")
    home = record.parents[2]
    assert home == rc.seat_dir(root, TASK, "op-home")
    trash = root / "state" / "custody_trash" / f"{home.name}-0123456789ab"  # as retire_seat_home moves it
    trash.parent.mkdir(parents=True)
    home.replace(trash)
    report = task_custody.remove_custody_trash(root)
    assert report["removed"] == 1 and not report["stuck"] and not scratch.exists() and not trash.exists()


def test_a_handoff_a_crash_interrupted_is_finished_when_the_prune_reads_the_seat_without_its_copy(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area, scratch, record = _seated(root, source, "op-crash")
    rc._release_area(area, {})  # the copy went; the process died before the scratch record followed
    assert not area.exists() and record.is_file()
    _custody(monkeypatch)
    report = rp.prune_review_checkouts(root, set())
    assert report["errors"] == [] and not record.exists() and len(_trash(root)) == 1
    assert task_custody.remove_custody_trash(root)["removed"] == 1 and not scratch.exists()
    rp.prune_review_checkouts(root, set())
    assert _trash(root) == [] and not (record.parents[2] / SESSION_SCRATCH_DIRNAME).exists()  # a second pass writes nothing
