"""A finished task's session scratch goes once the task is over, not a retention term later (#2586).

``workspace_preflight.session_service_directories`` notes each scratch record under its task in the
canonical root; ``release_finished_scratch`` (the custody pass) puts the record of a task proven over,
with no delegated run still open, into the custody trash whole, whose remover deletes the scratch. A
live, unknown, unfinished or owner-kept task, or one whose spared run still works, keeps its scratch;
a note whose record already retired with its drive, that names no record of its task, or that sits
behind a link touches nothing; the owner's Keep never reports a released scratch as kept; a producer's
retention keeps its scratch in place without holding the trash; a release or handoff that fails keeps the
trash entry and its records. Isolated roots only.
"""
from __future__ import annotations

import errno
import json
import os
import pathlib
import tempfile

import pytest

from ouroboros import delegate_custody, headless, owner_keep, session_scratch, task_custody, test_environment
from ouroboros.workspace_preflight import (
    OPEN_SCRATCH_DIR, SESSION_SCRATCH_DIRNAME, release_finished_scratch, session_service_directories,
)
from tests._shared import retire_synthetic_retention
from tests._store_shared import seed_result
from tests.test_session_service_directories import GRANTED, _Catalog, _prepare, _synthetic_temp_parent  # noqa: F401

OVER = lambda _task_id: False  # noqa: E731 -- the supervisor's probe: proven not live


def _started(tmp_path, task_id="task-1", status="completed", **kwargs):
    """One start's scratch holding an environment, then the task's durable row in STATUS."""
    data = tmp_path / "data"
    made = _prepare(tmp_path, _Catalog(GRANTED), task_id=task_id, **kwargs)
    scratch = pathlib.Path(made.scratch)
    (scratch / "env").mkdir()
    (scratch / "env" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    seed_result(data, task_id, {"status": status})
    return data, scratch


def _notes(data):
    return sorted(path.relative_to(data / OPEN_SCRATCH_DIR).as_posix()
                  for path in (data / OPEN_SCRATCH_DIR).rglob("*.json"))


def _record(data, task_id="task-1", key="snap-1"):
    return data / "task_drives" / task_id / SESSION_SCRATCH_DIRNAME / key / "scratch.json"


def _hold(data, reason, task_id="task-1", key="snap-1"):
    """The typed hold one kept record reports: its class, the record folder and since the note was written."""
    from ouroboros.session_scratch import _iso

    note = data / OPEN_SCRATCH_DIR / task_id / f"{key}.json"
    return {"class": reason, "count": 1, "example": str(_record(data, task_id, key).parent),
            "since": _iso(note.lstat().st_mtime)}


def test_a_finished_task_releases_its_scratch_on_the_custody_pass_long_before_its_drive_retires(tmp_path):
    data, scratch = _started(tmp_path)
    drive = data / "task_drives" / "task-1"
    report = release_finished_scratch(data, OVER)
    assert report == {"released": [str(_record(data).parent.resolve())], "holds": [], "errors": []}
    assert _notes(data) == [] and not (data / OPEN_SCRATCH_DIR / "task-1").exists()
    assert drive.is_dir() and not _record(data).parent.exists()  # the drive itself waits for its own term
    [entry] = (data / "state" / "custody_trash").iterdir()
    assert entry.name.startswith("task-1+scratch-")
    assert (entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").is_file()
    assert not any((data / "state" / "custody_staging").iterdir())
    assert scratch.is_dir()  # the delete is the maintenance child's, off the serving process
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "holds": []}
    assert not scratch.exists() and drive.is_dir()
    assert release_finished_scratch(data, OVER) == {"released": [], "holds": [], "errors": []}


@pytest.mark.parametrize("still_open", ["run", "start_request"])
def test_a_spared_delegated_run_keeps_its_finished_tasks_scratch_until_it_settles(tmp_path, still_open):
    data, scratch = _started(tmp_path)  # the task failed undeliberately; its run was left live (B1-A)
    try:
        if still_open == "run":
            assert delegate_custody.emit(data, delegate_custody.STARTED, {"run_id": "run-1", "task_id": "task-1"})
        else:
            assert delegate_custody.emit(data, delegate_custody.START_REQUESTED, {
                "invocation_id": "inv-1", "task_id": "task-1", "request": {"prompt": "x"}})
        assert release_finished_scratch(data, OVER) == {"released": [], "holds": [_hold(data, "run_open")], "errors": []}
        assert _record(data).is_file() and (scratch / "env" / "pyvenv.cfg").is_file()
        if still_open == "run":
            assert delegate_custody.emit(data, delegate_custody.SETTLED, {
                "run_id": "run-1", "task_id": "task-1", "state": "succeeded"})
        else:
            assert delegate_custody.emit(data, delegate_custody.START_FAILED, {
                "invocation_id": "inv-1", "task_id": "task-1", "definite": True, "error": "refused"})
        assert release_finished_scratch(data, OVER)["released"]
    finally:
        delegate_custody._CUSTODY.clear()


def test_the_trash_never_meets_a_half_made_scratch_entry(tmp_path, monkeypatch):
    data, scratch = _started(tmp_path)
    trash, real_replace, published = data / "state" / "custody_trash", os.replace, []

    def replace(src, dst):
        if pathlib.Path(dst).parent == trash:  # what the remover may see: complete at once, or nothing
            assert not trash.exists() or not any(trash.iterdir())
            assert (pathlib.Path(src) / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").is_file()
            published.append(dst)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    assert release_finished_scratch(data, OVER)["released"] and len(published) == 1
    monkeypatch.setattr(os, "replace", real_replace)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_an_interrupted_release_keeps_the_record_and_its_note(tmp_path, monkeypatch):
    data, scratch = _started(tmp_path)
    real_replace = os.replace

    def refused(src, dst):
        raise PermissionError("the trash could not take it")

    monkeypatch.setattr(os, "replace", refused)
    report = release_finished_scratch(data, OVER)
    assert report["released"] == [] and len(report["errors"]) == 1
    assert _record(data).is_file() and _notes(data) == ["task-1/snap-1.json"]
    assert not (data / "state" / "custody_trash").exists() or not any((data / "state" / "custody_trash").iterdir())
    monkeypatch.setattr(os, "replace", real_replace)
    assert release_finished_scratch(data, OVER)["released"]
    task_custody.remove_custody_trash(data)
    assert not scratch.exists()


def test_a_pass_stopped_after_the_record_left_the_drive_is_finished_by_the_next(tmp_path, monkeypatch):
    data, scratch = _started(tmp_path)
    trash, real_replace = data / "state" / "custody_trash", os.replace

    def unpublished(src, dst):
        if pathlib.Path(dst).parent == trash:
            raise PermissionError("the trash could not take it")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", unpublished)
    report = release_finished_scratch(data, OVER)
    assert report["released"] == [] and len(report["errors"]) == 1 and _notes(data) == ["task-1/snap-1.json"]
    _places, _unknown, pins = owner_keep.inventory(data, "task-1", {"status": "completed"})
    assert str(scratch) not in pins  # the record is in staging now: the Keep counts the scratch released
    monkeypatch.setattr(os, "replace", real_replace)
    release_finished_scratch(data, OVER)  # publishes the staged record, then drops the empty record folder
    assert _notes(data) == [] and not any((data / "state" / "custody_staging").iterdir())
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_an_aged_staged_release_outlives_the_leftover_sweep_and_is_published(tmp_path, monkeypatch):
    data, scratch = _started(tmp_path)
    staging, trash, real_replace = data / "state" / "custody_staging", data / "state" / "custody_trash", os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(PermissionError("no trash"))
                        if pathlib.Path(dst).parent == trash else real_replace(src, dst))
    assert release_finished_scratch(data, OVER)["errors"]
    monkeypatch.setattr(os, "replace", real_replace)
    settlement = staging / "task-7-0123456789ab"  # a crashed drive settlement's own leftover
    settlement.mkdir()
    for entry in staging.iterdir():
        os.utime(entry, (1, 1))  # both two hours (and more) old
    task_custody.sweep_custody_leftovers(data)
    [kept] = list(staging.iterdir())
    assert "+scratch-" in kept.name, "the settlement's leftover goes, the release's sole record stays"
    release_finished_scratch(data, OVER)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


@pytest.mark.parametrize("junction", ["note_folder", "record_folder"])
def test_a_windows_junction_is_never_followed(tmp_path, monkeypatch, junction):
    data, scratch = _started(tmp_path)
    target = str(data / OPEN_SCRATCH_DIR / "task-1") if junction == "note_folder" else str(_record(data).parent)
    real_lstat = os.lstat

    class _Junction:  # what Windows reports for a directory junction: a directory with the reparse attribute
        def __init__(self, seen):
            self.st_mode, self.st_file_attributes = seen.st_mode, 0x400

    monkeypatch.setattr(os, "lstat", lambda path, *a, **k: _Junction(real_lstat(path, *a, **k))
                        if os.fspath(path) == target else real_lstat(path, *a, **k))
    report = release_finished_scratch(data, OVER)
    assert report["released"] == [] and _record(data).is_file() and scratch.is_dir()
    assert not (data / "state" / "custody_trash").exists() or not any((data / "state" / "custody_trash").iterdir())


def test_a_record_folder_that_outlives_its_release_names_no_scratch(tmp_path, monkeypatch):
    from ouroboros import force_remove

    data, scratch = _started(tmp_path)
    stale = owner_keep.inventory(data, "task-1", {"status": "completed", "task_id": "task-1"})
    folder, original = _record(data).parent, force_remove.force_rmtree
    monkeypatch.setattr(force_remove, "force_rmtree",  # a Windows sharing violation keeps the folder
                        lambda path: None if pathlib.Path(path) == folder else original(path))
    assert release_finished_scratch(data, OVER)["released"]
    assert folder.is_dir() and not _record(data).exists()
    monkeypatch.setattr(owner_keep, "inventory", lambda *_args: stale)
    answer = owner_keep.set_owner_keep(data, "task-1", keep=True, source={"basis": "test"})
    assert str(scratch) in answer["retired"] and str(scratch) not in answer["retained"]
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_an_unreadable_custody_store_keeps_the_scratch(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current

    data, scratch = _started(tmp_path)

    def unreadable(*_args, **_kwargs):
        raise OSError("the store could not be read")

    monkeypatch.setattr(delegate_custody_current, "task_snapshot", unreadable)
    assert release_finished_scratch(data, OVER) == {"released": [], "holds": [_hold(data, "custody_unreadable")], "errors": []}
    assert _record(data).is_file() and scratch.is_dir()


def test_an_unreadable_record_is_unknown_to_the_keep_never_kept_or_retired(tmp_path, monkeypatch):
    data, scratch = _started(tmp_path)
    stale = owner_keep.inventory(data, "task-1", {"status": "completed", "task_id": "task-1"})
    _record(data).write_text("{torn", encoding="utf-8")
    monkeypatch.setattr(owner_keep, "inventory", lambda *_args: stale)
    answer = owner_keep.set_owner_keep(data, "task-1", keep=True, source={"basis": "test"})
    assert answer["unknown"] == ["environments"]
    assert str(scratch) not in answer["retained"] and str(scratch) not in answer["retired"]


def test_a_keep_racing_the_release_never_reports_the_released_scratch_as_kept(tmp_path, monkeypatch):
    data, scratch = _started(tmp_path)
    row = {"status": "completed", "task_id": "task-1"}
    stale = owner_keep.inventory(data, "task-1", row)  # read before the release, as set_owner_keep does unlocked
    places, _unknown, pins = stale
    assert str(scratch) in places and owner_keep._present(data, "task-1", str(scratch), pins) is True
    assert release_finished_scratch(data, OVER)["released"]  # accepted for deletion; the child has not run yet
    assert scratch.is_dir() and owner_keep._present(data, "task-1", str(scratch), pins) is False
    monkeypatch.setattr(owner_keep, "inventory", lambda *_args: stale)
    answer = owner_keep.set_owner_keep(data, "task-1", keep=True, source={"basis": "test"})
    assert str(scratch) in answer["retired"] and str(scratch) not in answer["retained"]
    assert answer["status"] == "partly_retired"


def test_a_linked_note_folder_is_never_followed(tmp_path):
    data, _scratch = _started(tmp_path, task_id="task-9")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "document.json").write_text('{"unrelated": true}\n', encoding="utf-8")
    linked = data / OPEN_SCRATCH_DIR / "task-1"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")
    seed_result(data, "task-1", {"status": "completed"})
    release_finished_scratch(data, OVER)
    assert (outside / "document.json").read_text(encoding="utf-8") == '{"unrelated": true}\n'
    assert linked.is_symlink()


def test_a_linked_note_is_never_read_or_removed(tmp_path):
    data, scratch = _started(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"unrelated": true}\n', encoding="utf-8")
    try:
        (data / OPEN_SCRATCH_DIR / "task-1" / "linked.json").symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")
    report = release_finished_scratch(data, OVER)
    assert report["errors"] == [] and len(report["released"]) == 1
    assert outside.read_text(encoding="utf-8") == '{"unrelated": true}\n'


@pytest.mark.parametrize("status, live, reason", [("completed", True, "task_live"), ("completed", None, "liveness_unknown"),
                                                  ("running", False, "task_not_terminal")],
                         ids=["live", "unknown", "unfinished"])
def test_a_task_not_proven_over_keeps_its_scratch_for_its_next_start(tmp_path, status, live, reason):
    data, scratch = _started(tmp_path, status=status)
    assert release_finished_scratch(data, lambda _task_id: live) == {"released": [], "holds": [_hold(data, reason)],
                                                                      "errors": []}
    assert _notes(data) == ["task-1/snap-1.json"] and _record(data).is_file()
    assert not (data / "state" / "custody_trash").exists()
    assert (scratch / "env" / "pyvenv.cfg").is_file()
    assert _prepare(tmp_path, _Catalog(GRANTED)).scratch == str(scratch)  # the same task and key reuse it


def test_a_busy_probe_holds_only_its_own_task_and_the_pass_goes_on(tmp_path):
    """The supervisor's probe raises ``StoreBusy`` when the store is contended (its contract): the task it was
    asked about holds its scratch as ``store_busy`` and the pass goes on, so a later finished task still
    releases; once the store answers, the held one releases too."""
    from ouroboros.store import StoreBusy

    data, held = _started(tmp_path)
    _data, other = _started(tmp_path, task_id="task-2")

    def probe(task_id):
        if task_id == "task-1":
            raise StoreBusy("ownership re-check found the store busy", reason="contention")
        return False

    report = release_finished_scratch(data, probe)
    assert report == {"released": [str(_record(data, "task-2").parent.resolve())], "holds": [_hold(data, "store_busy")],
                      "errors": []}
    assert _notes(data) == ["task-1/snap-1.json"] and _record(data).is_file()
    assert release_finished_scratch(data, OVER)["released"] == [str(_record(data).parent.resolve())]
    assert task_custody.remove_custody_trash(data)["removed"] == 2 and not held.exists() and not other.exists()


def test_the_owner_keep_holds_a_finished_tasks_scratch_until_released(tmp_path, monkeypatch):
    from contextlib import contextmanager

    data, scratch = _started(tmp_path)
    assert owner_keep.set_owner_keep(data, "task-1", keep=True, source={"basis": "test"})["keep"] is True
    assert release_finished_scratch(data, OVER) == {"released": [], "holds": [_hold(data, "owner_keep")], "errors": []}
    assert _notes(data) == ["task-1/snap-1.json"] and scratch.is_dir()
    assert owner_keep.set_owner_keep(data, "task-1", keep=False, source={"basis": "test"})["keep"] is False

    @contextmanager
    def busy(*args, **kwargs):
        yield False

    with monkeypatch.context() as patch:
        patch.setattr(task_custody, "task_custody_lock", busy)
        assert release_finished_scratch(data, OVER) == {"released": [], "holds": [_hold(data, "busy")], "errors": []}
        assert _notes(data) == ["task-1/snap-1.json"] and scratch.is_dir()
    report = release_finished_scratch(data, OVER)
    assert report["released"] and report["holds"] == []
    task_custody.remove_custody_trash(data)
    assert not scratch.exists()


@pytest.mark.parametrize("task", ["api_skill_review", "marketplace_install"])
def test_a_legacy_seat_note_releases_only_after_its_copy_is_gone(tmp_path, task):
    """A-4: legacy pseudo-owner drive records follow the seat proof, without a terminal task row."""
    key = "seat-legacy"
    data = tmp_path / "data"
    scratch = pathlib.Path(_prepare(tmp_path, _Catalog(GRANTED), task_id=task, key=key).scratch)
    area = data / "state" / "review_checkouts" / "seats" / "legacy" / "area"
    area.mkdir(parents=True)
    assert release_finished_scratch(data, OVER) == {
        "released": [], "holds": [_hold(data, "seat_open", task, key)], "errors": []}
    assert scratch.is_dir() and _record(data, task, key).is_file()
    area.rmdir()
    assert release_finished_scratch(data, OVER) == {
        "released": [str(_record(data, task, key).parent.resolve())], "holds": [], "errors": []}
    assert _notes(data) == []
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_counting_scratch_skips_hold_proofs_and_keeps_other_roots_after_an_unreadable_record(tmp_path, monkeypatch):
    data, good = _started(tmp_path, "good")
    _data, bad = _started(tmp_path, "bad")
    read = pathlib.Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == _record(data, "bad"):
            raise PermissionError("record unreadable")
        return read(path, *args, **kwargs)

    def no_hold_reads(*args, **kwargs):
        pytest.fail("counting must not compute discarded holds")

    monkeypatch.setattr(pathlib.Path, "read_text", unreadable)
    with monkeypatch.context() as patch:
        for name in ("task_hold", "seat_hold", "retention_hold"):
            patch.setattr(session_scratch, name, no_hold_reads)
        patch.setattr(owner_keep, "is_kept", no_hold_reads)
        assert session_scratch.recorded_scratch_roots(data) == [str(good)]
        assert bad.is_dir()  # an unreadable record is not retirement authority
    rows = {row["task_id"]: row for row in session_scratch.scratch_records(data, live=lambda _: True)}
    assert rows["good"]["valid"] and rows["good"]["hold"] == "task_live"  # the full view still judges holds
    assert not rows["bad"]["valid"] and rows["bad"]["problem"].startswith("PermissionError:")


def test_a_note_whose_record_retired_with_its_drive_is_dropped(tmp_path):
    data, scratch = _started(tmp_path)
    assert task_custody.settle_child_drive(data, "task-1", data / "task_drives" / "task-1", live=OVER,
                                           admission_rollback=True)["status"] == "removed"
    assert release_finished_scratch(data, OVER) == {"released": [], "holds": [], "errors": []}
    assert _notes(data) == []
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_a_note_that_names_no_record_of_its_task_is_dropped_and_touches_nothing(tmp_path):
    data, _scratch = _started(tmp_path)
    other = pathlib.Path(_prepare(tmp_path, _Catalog(GRANTED), task_id="task-2").scratch)
    forged = data / OPEN_SCRATCH_DIR / "task-1" / "snap-1.json"
    forged.write_text(json.dumps({"schema_version": 1, "owner": str(_record(data, "task-2").parent)}),
                      encoding="utf-8")
    report = release_finished_scratch(data, OVER)
    assert report["released"] == [] and len(report["errors"]) == 1 and "names no scratch record" in report["errors"][0]
    assert _notes(data) == ["task-2/snap-1.json"]  # task-2 is not over: its own note stays
    assert _record(data, "task-2").is_file() and other.is_dir()
    assert not (data / "state" / "custody_trash").exists()


def test_a_headless_tasks_scratch_is_released_from_its_own_data_root(tmp_path):
    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, "child-1", "empty")
    target = tmp_path / "target"
    target.mkdir()
    made = session_service_directories(_Catalog(GRANTED), "route", access="full", mode="agent", data_root=data,
                                       drive_root=drive, task_id="child-1", key="in-place-inv",
                                       execution_root=target)
    scratch = pathlib.Path(made.scratch)
    assert scratch.is_dir() and _notes(data) == ["child-1/in-place-inv.json"]
    seed_result(data, "child-1", {"status": "completed"})
    record = drive / "task_drives" / "child-1" / SESSION_SCRATCH_DIRNAME / "in-place-inv"
    assert release_finished_scratch(data, OVER)["released"] == [str(record.resolve())]
    assert task_custody.remove_custody_trash(data)["removed"] == 1
    assert not scratch.exists() and drive.is_dir()


def test_the_custody_pass_releases_only_on_the_supervisors_negative_probe(tmp_path, monkeypatch):
    import supervisor.queue
    from ouroboros import server_maintenance

    data, scratch = _started(tmp_path)
    monkeypatch.setattr(server_maintenance, "DATA_DIR", data)
    monkeypatch.setattr(supervisor.queue, "task_settlement_liveness", lambda _task_id: None)
    server_maintenance._release_finished_scratch()
    assert _notes(data) == ["task-1/snap-1.json"] and not (data / "state" / "custody_trash").exists()
    monkeypatch.setattr(supervisor.queue, "task_settlement_liveness", OVER)
    server_maintenance._release_finished_scratch()
    assert _notes(data) == [] and len(list((data / "state" / "custody_trash").iterdir())) == 1
    assert scratch.is_dir()  # deleted by the maintenance child, not on this pass


def test_a_record_under_the_legacy_tmp_parent_still_releases_and_a_foreign_parent_never(tmp_path, monkeypatch):
    """O3: new scratch allocates under ``session_scratch_parent``; a record made before that parent existed
    names ``default_temp_parent`` and still releases; a record naming any other parent is refused, not deleted."""
    legacy = tmp_path / "legacy-tmp"
    legacy.mkdir()
    monkeypatch.setattr(test_environment, "default_temp_parent", lambda: legacy)
    data, scratch = _started(tmp_path)
    assert scratch.parent == test_environment.session_scratch_parent().resolve() != legacy.resolve()
    old = pathlib.Path(tempfile.mkdtemp(prefix="obs-", dir=legacy))
    (old / "env").mkdir()
    seen = old.lstat()
    _record(data).write_text(json.dumps({"schema_version": 1, "path": str(old), "identity": [seen.st_dev, seen.st_ino]}),
                             encoding="utf-8")
    assert release_finished_scratch(data, OVER)["released"]
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not old.exists() and scratch.is_dir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    stray = pathlib.Path(tempfile.mkdtemp(prefix="obs-", dir=foreign))
    seen = stray.lstat()
    _record(data, "task-2").parent.mkdir(parents=True)
    _record(data, "task-2").write_text(json.dumps({"schema_version": 1, "path": str(stray),
                                                   "identity": [seen.st_dev, seen.st_ino]}), encoding="utf-8")
    with pytest.raises(OSError, match="invalid scratch ownership record"):
        session_scratch.release_session_scratch(data / "task_drives" / "task-2")
    assert stray.is_dir()


@pytest.mark.parametrize("reason", ["the live process table could not be enumerated", "debug retention"],
                         ids=["process-proof-failed", "debug-keep"])
def test_a_retained_scratch_keeps_its_record_as_a_visible_hold_until_its_marker_goes(tmp_path, reason, monkeypatch):
    """#2708: a producer's retention marker never deletes the scratch, and no longer drops its owner: the
    record stays in the trash as the hold ``producer_retained`` (the marker's words, since it was written),
    re-read every run, and the scratch goes only once the marker is gone, a failed process proof and a
    deliberate debug keep alike; nothing else distinguishes them here."""
    from ouroboros.session_scratch import _iso

    data, scratch = _started(tmp_path)
    nested = scratch / "t" / "ob-run.noindex"  # a test run inside the scratch that could not prove its exit
    nested.mkdir(parents=True)
    assert test_environment.retain_tree(nested, reason)
    marker = nested / test_environment.RETENTION_MARKER
    assert release_finished_scratch(data, OVER)["released"]
    hold = {"class": "producer_retained", "count": 1, "example": f"{scratch}: {reason}", "since": _iso(marker.lstat().st_mtime)}
    report = task_custody.remove_custody_trash(data)
    assert report["removed"] == 0 and report["stuck"] == {} and report["holds"] == [hold]
    assert (scratch / "env" / "pyvenv.cfg").is_file() and marker.is_file()
    [entry] = (data / "state" / "custody_trash").iterdir()  # the record survives, in its own handoff entry
    assert entry.name.startswith("task-1+scratch-") and (entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").is_file()
    assert report["held"][entry.name] == [entry.stat().st_ino, entry.stat().st_mtime_ns]
    assert task_custody.remove_custody_trash(data)["holds"] == [hold]  # unchanged: silent
    from ouroboros import maintenance, maintenance_jobs

    monkeypatch.setattr(maintenance, "JOBS", {"custody_trash": maintenance.JOBS["custody_trash"]})
    assert maintenance.run_job(data, "custody_trash") == {"state": "done", "items": 0}
    assert maintenance.due(data) == [], "an unchanged held entry is at rest"
    row = maintenance_jobs.job(data, "custody_trash")
    assert row["held"][entry.name] == [entry.stat().st_ino, entry.stat().st_mtime_ns]
    stamp = entry.stat().st_mtime_ns + 1_000_000
    os.utime(entry, ns=(stamp, stamp))
    assert maintenance.due(data) == ["custody_trash"], "a changed held entry is due again"
    assert retire_synthetic_retention(nested, reason)  # the producer's proof holds (or the owner removed the marker)
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "holds": []}
    assert not scratch.exists() and not list((data / "state" / "custody_trash").iterdir())


def test_a_retained_sibling_does_not_pin_the_retired_drive_and_keeps_only_its_own_record(tmp_path):
    """A drive whose settlement retired it goes at once; the one marked scratch inside it keeps its record in
    a handoff entry of its own, and the drive's other scratch is deleted."""
    data, held = _started(tmp_path)
    plain = pathlib.Path(_prepare(tmp_path, _Catalog(GRANTED), key="snap-2").scratch)
    reason = "producer could not prove process exit"
    assert test_environment.retain_tree(held, reason)
    assert task_custody.settle_child_drive(data, "task-1", data / "task_drives" / "task-1", live=OVER,
                                           admission_rollback=True)["status"] == "removed"
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], report["stuck"], [hold["class"] for hold in report["holds"]]) == (1, {}, ["producer_retained"])
    assert not plain.exists() and (held / "env" / "pyvenv.cfg").is_file()
    [entry] = (data / "state" / "custody_trash").iterdir()
    assert entry.name.startswith("task-1+scratch-") and (entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").is_file()
    assert not (data / "task_drives" / "task-1").exists()
    assert retire_synthetic_retention(held, reason)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not held.exists()


@pytest.mark.parametrize("fault", ["invalid_record", "unreadable_record", "failed_handoff"])
def test_a_failed_release_or_handoff_keeps_the_trash_entry_and_every_record_in_it(tmp_path, monkeypatch, fault):
    """A trash entry goes only once its scratch is released and each marked record has moved into its own
    entry: an invalid or unreadable ownership record, or a record that could not move, keeps the moved drive
    whole with its records, held as ``removal_failed``, and leaves the trash as it was (the unchanged entry
    spawns no child), with a failed handoff's marked scratch still held; once the fault is gone the next run
    releases the drive and hands the marked record on."""
    from ouroboros import maintenance, maintenance_jobs
    from ouroboros.session_scratch import _iso

    if fault == "unreadable_record":  # a platform or user that searches a folder without its bits cannot fault here
        (tmp_path / "probe" / "folder").mkdir(parents=True)
        (tmp_path / "probe" / "folder" / "file").write_text("x", encoding="utf-8")
        os.chmod(tmp_path / "probe" / "folder", 0o600)
        try:
            os.lstat(tmp_path / "probe" / "folder" / "file")
            pytest.skip("this platform does not deny a folder's search permission")
        except PermissionError:
            pass
        finally:
            os.chmod(tmp_path / "probe" / "folder", 0o700)
    data, held = _started(tmp_path)
    plain = pathlib.Path(_prepare(tmp_path, _Catalog(GRANTED), key="snap-2").scratch)
    reason = "producer could not prove process exit"
    assert test_environment.retain_tree(held, reason)
    assert task_custody.settle_child_drive(data, "task-1", data / "task_drives" / "task-1", live=OVER,
                                           admission_rollback=True)["status"] == "removed"
    trash = data / "state" / "custody_trash"
    [entry] = trash.iterdir()
    records, real_replace = entry / SESSION_SCRATCH_DIRNAME, os.replace
    good = (records / "snap-2" / "scratch.json").read_text(encoding="utf-8")
    if fault == "invalid_record":
        (records / "snap-2" / "scratch.json").write_text("{torn", encoding="utf-8")
    elif fault == "unreadable_record":
        os.chmod(records / "snap-2", 0o600)  # listed, but its record cannot be looked at
    else:
        def refused(src, dst):
            if pathlib.Path(src).name == "scratch.json":
                raise OSError(errno.ENOSPC, "No space left on device")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", refused)
    monkeypatch.setattr(maintenance, "JOBS", {"custody_trash": maintenance.JOBS["custody_trash"]})
    try:
        assert maintenance.run_job(data, "custody_trash") == {"state": "failed", "items": 0}
    finally:
        if fault == "unreadable_record":
            os.chmod(records / "snap-2", 0o700)
    assert list(trash.iterdir()) == [entry] and (records / "snap-1" / "scratch.json").is_file()
    assert not list((data / "state" / "custody_staging").glob("*+scratch-*")) and held.is_dir()
    row, seen = maintenance_jobs.job(data, "custody_trash"), entry.lstat()
    failure = "PermissionError" if fault == "unreadable_record" else "OSError"
    assert row["stuck"] == {entry.name: [seen.st_ino, seen.st_mtime_ns]}
    assert [hold for hold in row["holds"] if hold["class"] == "removal_failed"] == [
        {"class": "removal_failed", "count": 1, "example": f"{entry.name}: {failure}", "since": _iso(seen.st_mtime)}]
    assert [hold["class"] for hold in row["holds"] if hold["class"] != "removal_failed"] == (
        ["producer_retained"] if fault == "failed_handoff" else []), "a marked scratch stands whether its record moved"
    assert maintenance.due(data) == [], "an unchanged kept entry spawns no child"
    monkeypatch.setattr(os, "replace", real_replace)
    (records / "snap-2" / "scratch.json").write_text(good, encoding="utf-8")
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], report["stuck"], [hold["class"] for hold in report["holds"]]) == (1, {}, ["producer_retained"])
    [handoff] = trash.iterdir()
    assert handoff.name.startswith("task-1+scratch-") and (handoff / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").is_file()
    assert not plain.exists() and (held / "env" / "pyvenv.cfg").is_file()
    assert retire_synthetic_retention(held, reason)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not held.exists()


@pytest.mark.parametrize("task,missing_note", [("long-parent", False), ("api_skill_review", True), ("marketplace_install", False)])
def test_review_scratch_follows_finished_seat_not_parent_task(tmp_path, monkeypatch, task, missing_note):
    from types import SimpleNamespace
    from ouroboros import review_checking as rc, review_checking_prune as rp
    from ouroboros.review_session_preparation import seat_service_directories
    from tests.test_review_checkout_lifecycle import _frozen, _seat
    root, source = _frozen(tmp_path)
    if task == "long-parent":
        seed_result(root, task, {"status": "running"})
    areas = [_seat(root, source, op, task=task) for op in ("done", "working")]
    services = [seat_service_directories(_Catalog(GRANTED), "route", SimpleNamespace(access="full", mode="agent"),
                root, task, op, str(area)) for op, area in zip(("done", "working"), areas)]
    for service in services:
        (pathlib.Path(service.scratch) / 'unique-env').write_bytes(b'ENVIRONMENT')
    if missing_note:
        for note in (root / OPEN_SCRATCH_DIR / task).glob('*.json'):
            note.unlink()
        import time
        rp.prune_review_checkouts(root, {task}, views_since=time.time())
    assert rc.capture_seat_evidence(root, task, "done")["status"] == "retained"
    # Reviewer seats own these records beside their copy.  Capturing the finished
    # seat releases that holder directly; the task scratch pass has no task note.
    report = release_finished_scratch(root, lambda _: True)
    assert report['released'] == [] and report['errors'] == []
    task_custody.remove_custody_trash(root)
    assert not pathlib.Path(services[0].scratch).exists()
    assert (pathlib.Path(services[1].scratch) / 'unique-env').read_bytes() == b'ENVIRONMENT'
    receipt = rc._receipt(root, task, "done")
    assert (pathlib.Path(receipt['evidence_dir']) / 'files' / 'check.py').read_bytes() == b'# reviewer reproduction\n'


def test_continuation_keeps_original_environment_through_maintenance_then_releases(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import review_checking as rc, review_checking_prune as rp, review_continuation as continuation
    from ouroboros.review_session_preparation import seat_service_directories
    from tests.test_review_checkout_lifecycle import _frozen, _seat
    from tests.test_review_checking_prune import _custody
    root, source = _frozen(tmp_path)
    task = 'api_skill_review'  # the Skills page can continue without a live managed parent
    area = _seat(root, source, 'stopped', task=task)
    service = seat_service_directories(_Catalog(GRANTED), 'route', SimpleNamespace(access='full', mode='agent'), root, task, 'stopped', str(area))
    env = pathlib.Path(service.scratch) / 'environment'
    env.write_bytes(b'CONTINUABLE-ENV')
    first = rc.capture_seat_evidence(root, task, 'stopped', hold=True)
    _custody(monkeypatch, settled=[area])
    assert rp.prune_review_checkouts(root, set())['released'] == 0
    assert release_finished_scratch(root, OVER)['released'] == []
    assert env.read_bytes() == b'CONTINUABLE-ENV'
    continuation.transfer_checking_copy(root, task, 'stopped', 'successor')
    assert continuation.checking_copy_holder(root, task, 'successor') == (True, '')
    (area / 'check.py').write_bytes(b'AFTER-CONTINUATION')
    second = rc.capture_seat_evidence(root, task, 'successor')
    assert second['status'] == 'retained' and not area.exists()
    assert (pathlib.Path(first['evidence_dir']) / 'files' / 'check.py').read_bytes() == b'# reviewer reproduction\n'
    assert (pathlib.Path(second['evidence_dir']) / 'files' / 'check.py').read_bytes() == b'AFTER-CONTINUATION'
    assert release_finished_scratch(root, OVER)['released'] == []
    task_custody.remove_custody_trash(root)
    assert not env.exists()


@pytest.mark.parametrize("order", ["release_first", "retire_first", "cached_drive_first"])
def test_finished_seat_scratch_release_survives_either_cleanup_order(tmp_path, monkeypatch, order):
    """Seat-home retirement and seat-owned scratch release are independent; neither may strand the other."""
    from types import SimpleNamespace
    from ouroboros import review_checking as rc, review_checking_prune as rp
    from ouroboros.review_session_preparation import seat_service_directories
    from tests.test_review_checkout_lifecycle import _frozen, _seat
    from tests.test_review_checking_prune import _custody

    root, source = _frozen(tmp_path)
    task, operation = "finished-parent", "review-operation"
    seed_result(root, task, {"status": "completed", "finished_at": "2020-01-01T00:00:00+00:00"})
    area = _seat(root, source, operation, task=task)
    service = seat_service_directories(None, "unused-full-route", SimpleNamespace(access="full", mode="agent"),
                                       root, task, operation, str(area))
    scratch = pathlib.Path(service.scratch)
    (scratch / "environment").write_bytes(b"DISPOSABLE-ENV")
    _custody(monkeypatch, settled=[area])

    if order == "release_first":
        assert rc.capture_seat_evidence(root, task, operation)["status"] == "retained"
        assert rp.retire_seat_home(root, area.parent, OVER, {}) == ""
    elif order == "retire_first":
        assert rp.retire_seat_home(root, area.parent, OVER, {}) == "in_use"
        assert rc.capture_seat_evidence(root, task, operation)["status"] == "retained"
        assert rp.retire_seat_home(root, area.parent, OVER, {}) == ""
    elif order == "whole_drive_first":
        assert rp.retire_seat_home(root, area.parent, OVER, {}) == "in_use"
        assert rc.capture_seat_evidence(root, task, operation)["status"] == "retained"
        assert rp.retire_seat_home(root, area.parent, OVER, {}) == ""
    else:
        assert task_custody.remove_custody_trash(root)["removed"] == 0
        assert rc.capture_seat_evidence(root, task, operation)["status"] == "retained"
        assert rp.retire_seat_home(root, area.parent, OVER, {}) == ""
    assert task_custody.remove_custody_trash(root)["removed"] >= 1
    assert not scratch.exists()


@pytest.mark.parametrize('fault', ['keep', 'capture_failed', 'seat_in_trash', 'seat_retirement'])
def test_seat_environment_survives_negative_proofs_and_drive_retirement(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    from ouroboros import review_checking as rc, review_checking_prune as rp
    from ouroboros.review_session_preparation import seat_service_directories
    from tests.test_review_checkout_lifecycle import _frozen, _seat
    root, source = _frozen(tmp_path)
    task = 'parent'
    seed_result(root, task, {'status': 'completed'})
    area = _seat(root, source, 'op', task=task)
    service = seat_service_directories(_Catalog(GRANTED), 'route', SimpleNamespace(access='full', mode='agent'), root, task, 'op', str(area))
    env = pathlib.Path(service.scratch) / 'environment'
    env.write_bytes(b'KEEP-ENV')
    if fault == 'keep':
        owner_keep.set_owner_keep(root, task, keep=True, source={'basis':'test'})
        assert rc.capture_seat_evidence(root, task, 'op')['status'] == 'retained'
    elif fault == 'capture_failed':
        monkeypatch.setattr(rc, '_directory_delta', lambda *_: (_ for _ in ()).throw(OSError('capture refused')))
        assert rc.capture_seat_evidence(root, task, 'op')['status'] == 'capture_failed'
    elif fault == 'seat_retirement':
        assert rp.retire_seat_home(root, area.parent, OVER, {}) == 'in_use'
    else:
        trash = root / 'state' / 'custody_trash' / 'parent-0123456789ab'
        trash.parent.mkdir(parents=True)
        area.parent.rename(trash)
        assert (trash / area.name).is_dir()
    assert release_finished_scratch(root, OVER)['released'] == []
    assert env.read_bytes() == b'KEEP-ENV'
    assert (trash / area.name).is_dir() if fault == 'seat_in_trash' else area.is_dir()
