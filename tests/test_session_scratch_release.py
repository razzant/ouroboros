"""A finished task's session scratch goes once the task is over, not a retention term later (#2586).

``workspace_preflight.session_service_directories`` notes each scratch record under its task in the
canonical root; ``release_finished_scratch`` (the custody pass) moves the record folder of a task proven
over into the custody trash, whose remover deletes the scratch. A live, unknown, unfinished or
owner-kept task keeps its scratch for a later start of the same task; a note whose record already
retired with its drive, or that names no record of its task, is dropped and touches nothing; a
producer's retention keeps its scratch in place without holding the trash. Isolated roots only.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros import headless, owner_keep, task_custody, test_environment
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
    assert scratch.is_dir()  # the delete is the maintenance child's, off the serving process
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "retained": []}
    assert not scratch.exists() and drive.is_dir()
    assert release_finished_scratch(data, OVER) == {"released": [], "kept": 0, "errors": []}


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
