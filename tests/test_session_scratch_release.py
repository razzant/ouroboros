"""A finished task's session scratch goes once the task is over, not a retention term later (#2586).

``workspace_preflight.session_service_directories`` notes each scratch record under its task in the
canonical root; ``release_finished_scratch`` (the custody pass) puts the record of a task proven over,
with no delegated run still open, into the custody trash whole, whose remover deletes the scratch. A
live, unknown, unfinished or owner-kept task, or one whose spared run still works, keeps its scratch;
a note whose record already retired with its drive, that names no record of its task, or that sits
behind a link touches nothing; the owner's Keep never reports a released scratch as kept; a producer's
retention keeps its scratch in place without holding the trash. Isolated roots only.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from ouroboros import delegate_custody, headless, owner_keep, task_custody, test_environment
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


def test_a_finished_task_releases_its_scratch_on_the_custody_pass_long_before_its_drive_retires(tmp_path):
    data, scratch = _started(tmp_path)
    drive = data / "task_drives" / "task-1"
    report = release_finished_scratch(data, OVER)
    assert report == {"released": [str(_record(data).parent.resolve())], "kept": 0, "errors": []}
    assert _notes(data) == [] and not (data / OPEN_SCRATCH_DIR / "task-1").exists()
    assert drive.is_dir() and not _record(data).parent.exists()  # the drive itself waits for its own term
    [entry] = (data / "state" / "custody_trash").iterdir()
    assert entry.name.startswith("task-1+scratch-")
    assert (entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").is_file()
    assert not any((data / "state" / "custody_staging").iterdir())
    assert scratch.is_dir()  # the delete is the maintenance child's, off the serving process
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "retained": []}
    assert not scratch.exists() and drive.is_dir()
    assert release_finished_scratch(data, OVER) == {"released": [], "kept": 0, "errors": []}


@pytest.mark.parametrize("still_open", ["run", "start_request"])
def test_a_spared_delegated_run_keeps_its_finished_tasks_scratch_until_it_settles(tmp_path, still_open):
    data, scratch = _started(tmp_path)  # the task failed undeliberately; its run was left live (B1-A)
    try:
        if still_open == "run":
            assert delegate_custody.emit(data, delegate_custody.STARTED, {"run_id": "run-1", "task_id": "task-1"})
        else:
            assert delegate_custody.emit(data, delegate_custody.START_REQUESTED, {
                "invocation_id": "inv-1", "task_id": "task-1", "request": {"prompt": "x"}})
        assert release_finished_scratch(data, OVER) == {"released": [], "kept": 1, "errors": []}
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
    assert release_finished_scratch(data, OVER) == {"released": [], "kept": 1, "errors": []}
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


@pytest.mark.parametrize("status, live", [("completed", True), ("completed", None), ("running", False)],
                         ids=["live", "unknown", "unfinished"])
def test_a_task_not_proven_over_keeps_its_scratch_for_its_next_start(tmp_path, status, live):
    data, scratch = _started(tmp_path, status=status)
    assert release_finished_scratch(data, lambda _task_id: live) == {"released": [], "kept": 1, "errors": []}
    assert _notes(data) == ["task-1/snap-1.json"] and _record(data).is_file()
    assert not (data / "state" / "custody_trash").exists()
    assert (scratch / "env" / "pyvenv.cfg").is_file()
    assert _prepare(tmp_path, _Catalog(GRANTED)).scratch == str(scratch)  # the same task and key reuse it


def test_the_owner_keep_holds_a_finished_tasks_scratch_until_released(tmp_path):
    data, scratch = _started(tmp_path)
    assert owner_keep.set_owner_keep(data, "task-1", keep=True, source={"basis": "test"})["keep"] is True
    assert release_finished_scratch(data, OVER) == {"released": [], "kept": 1, "errors": []}
    assert _notes(data) == ["task-1/snap-1.json"] and scratch.is_dir()
    assert owner_keep.set_owner_keep(data, "task-1", keep=False, source={"basis": "test"})["keep"] is False
    assert release_finished_scratch(data, OVER)["released"]
    task_custody.remove_custody_trash(data)
    assert not scratch.exists()


def test_a_note_whose_record_retired_with_its_drive_is_dropped(tmp_path):
    data, scratch = _started(tmp_path)
    assert task_custody.settle_child_drive(data, "task-1", data / "task_drives" / "task-1", live=OVER,
                                           admission_rollback=True)["status"] == "removed"
    assert release_finished_scratch(data, OVER) == {"released": [], "kept": 0, "errors": []}
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


def test_a_retained_scratch_stays_in_place_and_holds_no_trash(tmp_path):
    data, scratch = _started(tmp_path)
    nested = scratch / "t" / "ob-run.noindex"  # a test run inside the scratch that could not prove its exit
    nested.mkdir(parents=True)
    reason = "the live process table could not be enumerated"
    assert test_environment.retain_tree(nested, reason)
    assert release_finished_scratch(data, OVER)["released"]
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "retained": [str(scratch)]}
    assert (scratch / "env" / "pyvenv.cfg").is_file() and (nested / test_environment.RETENTION_MARKER).is_file()
    assert not list((data / "state" / "custody_trash").iterdir())
    assert retire_synthetic_retention(nested, reason)
