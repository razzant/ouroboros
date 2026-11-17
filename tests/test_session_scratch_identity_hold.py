"""A recorded scratch whose real ``obs-*`` directory has another device/inode is kept, never repaired.

A reboot can renumber a device, so a valid ownership record may name a real scratch directory under an
identity it no longer has; another run may still use that directory. Only that case becomes the visible
hold ``identity_unverified``: the directory and its markers stay untouched, the record's exact bytes move
into a ``<task>+scratch-<token>`` handoff entry of its own, and the rest of the retired drive (its other,
proven scratch included) is released. A malformed, unreadable, vanished, foreign, linked or non-directory
record stays an error that keeps the whole entry. A new start refuses the directory; Storage never counts
or offers it. Isolated roots only.
"""
from __future__ import annotations

import errno
import json
import os
import pathlib
import tempfile

import pytest

from ouroboros import maintenance, maintenance_jobs, session_scratch, storage_removers, task_custody, test_environment
from ouroboros import workspace_preflight
from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME, release_finished_scratch
from tests._shared import retire_synthetic_retention
from tests.test_session_scratch_release import OVER, _record, _started
from tests.test_session_service_directories import GRANTED, _Catalog, _prepare, _synthetic_temp_parent  # noqa: F401

TRASH = ("state", "custody_trash")


def _unverify(record: pathlib.Path) -> list:
    """Renumber the recorded device as a reboot can: same real directory, another recorded identity."""
    row = json.loads(record.read_text(encoding="utf-8"))
    row["identity"] = [row["identity"][0] + 1, row["identity"][1]]
    record.write_text(json.dumps(row), encoding="utf-8")
    return row["identity"]


def _tree(path: pathlib.Path) -> dict:
    """Every entry under PATH with its inode and bytes: what 'untouched' means here."""
    seen = {".": [path.lstat().st_ino, b""]}
    for item in sorted(path.rglob("*")):
        seen[item.relative_to(path).as_posix()] = [item.lstat().st_ino, item.read_bytes() if item.is_file() else b""]
    return seen


def _settled(data, task_id="task-1"):
    assert task_custody.settle_child_drive(data, task_id, data / "task_drives" / task_id, live=OVER,
                                           admission_rollback=True)["status"] == "removed"
    [entry] = data.joinpath(*TRASH).iterdir()
    return entry


def _two_scratch_drive(tmp_path):
    """task-1's drive: snap-1 (an environment, its record renumbered) and snap-2 (proven, disposable)."""
    data, held = _started(tmp_path)
    plain = pathlib.Path(_prepare(tmp_path, _Catalog(GRANTED), key="snap-2").scratch)
    (plain / "work.txt").write_text("disposable\n", encoding="utf-8")
    recorded = _unverify(_record(data))
    return data, held, plain, recorded, _record(data).read_bytes()


def test_an_unverified_scratch_keeps_its_record_and_releases_its_proven_sibling(tmp_path, monkeypatch):
    data, held, plain, recorded, record_bytes = _two_scratch_drive(tmp_path)
    before = _tree(held)
    found = [held.lstat().st_dev, held.lstat().st_ino]
    _settled(data)
    monkeypatch.setattr(maintenance, "JOBS", {"custody_trash": maintenance.JOBS["custody_trash"]})
    assert maintenance.run_job(data, "custody_trash") == {"state": "done", "items": 1}
    row = maintenance_jobs.job(data, "custody_trash")
    assert row["stuck"] == {}
    [hold] = row["holds"]
    assert hold["class"] == "identity_unverified" and hold["count"] == 1 and hold["since"]
    assert hold["example"] == f"{held} [dev, ino] {recorded} != {found}; record snap-1"[:160]
    assert not plain.exists(), "the proven sibling is released, not pinned by the unverified one"
    assert _tree(held) == before, "the unverified environment is never renamed, copied or deleted"
    [entry] = data.joinpath(*TRASH).iterdir()
    assert entry.name.startswith("task-1+scratch-")  # made by this pass: listed as held from the next one
    moved = entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json"
    assert moved.read_bytes() == record_bytes, "the ownership declaration moves byte for byte"
    layout = sorted(item.relative_to(entry).as_posix() for item in entry.rglob("*"))
    assert layout == [SESSION_SCRATCH_DIRNAME, f"{SESSION_SCRATCH_DIRNAME}/snap-1", f"{SESSION_SCRATCH_DIRNAME}/snap-1/scratch.json"]
    # The Storage remover fact carries the same typed hold.
    [fact] = [fact for fact in storage_removers.place_removers(data, engine=False)["session_scratch"]
              if fact["id"] == "custody_trash"]
    assert [h["class"] for h in fact["holds"]] == ["identity_unverified"] and fact["held"] == 1
    # Repeating the pass makes no further hold, entry or nesting.
    for _ in range(2):
        again = task_custody.remove_custody_trash(data)
        assert (again["removed"], again["stuck"], again["holds"], list(again["held"])) == (0, {}, [hold], [entry.name])
        assert list(data.joinpath(*TRASH).iterdir()) == [entry]
        assert sorted(item.relative_to(entry).as_posix() for item in entry.rglob("*")) == layout
    assert moved.read_bytes() == record_bytes and _tree(held) == before
    # Storage: never counted or offered, the exact record detail reported.
    assert str(held) not in session_scratch.recorded_scratch_roots(data)
    [listed] = session_scratch.scratch_records(data)
    assert (listed["holder"], listed["valid"], listed["missing"], listed["hold"]) == ("trash", False, False, "identity_unverified")
    assert (listed["recorded_identity"], listed["found_identity"], listed["path"]) == (recorded, found, str(held))
    assert listed["record"] == str(moved) and str(moved) in listed["problem"]


def test_negative_control_the_former_contract_pinned_the_whole_entry(tmp_path, monkeypatch):
    """The defect this repairs: treating the renumbered identity as an invalid record kept the whole retired
    drive (removal_failed) and with it the proven sibling's scratch. Reproduced by restoring that contract."""
    data, held, plain, _recorded, _bytes = _two_scratch_drive(tmp_path)
    entry = _settled(data)
    real = session_scratch.recorded_scratch

    def former(record):
        try:
            return real(record)
        except session_scratch.ScratchIdentityMismatch as exc:
            raise ValueError(str(exc)) from exc  # what the record check raised before the hold existed

    monkeypatch.setattr(session_scratch, "recorded_scratch", former)
    report = task_custody.remove_custody_trash(data)
    assert report["removed"] == 0 and list(report["stuck"]) == [entry.name]
    assert [hold["class"] for hold in report["holds"]] == ["removal_failed"]
    assert plain.is_dir() and held.is_dir(), "the disposable sibling stayed pinned by the unverified record"
    monkeypatch.setattr(session_scratch, "recorded_scratch", real)
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], [h["class"] for h in report["holds"]]) == (1, ["identity_unverified"])
    assert not plain.exists() and held.is_dir()


def _fault(tmp_path, fault):
    """task-1's drive with one record broken by FAULT; returns (data, record, the directory to keep)."""
    data, scratch = _started(tmp_path)
    record = _record(data)
    row = json.loads(record.read_text(encoding="utf-8"))
    parent = test_environment.session_scratch_parent()
    keep = scratch
    if fault == "malformed_json":
        record.write_text("{torn", encoding="utf-8")
    elif fault == "no_identity":
        record.write_text(json.dumps({"schema_version": 1, "path": row["path"]}), encoding="utf-8")
    elif fault in ("bool_identity", "short_identity", "text_identity"):
        identity = {"bool_identity": [True, row["identity"][1]], "short_identity": row["identity"][:1],
                    "text_identity": [str(part) for part in row["identity"]]}[fault]
        record.write_text(json.dumps({**row, "identity": identity}), encoding="utf-8")
    elif fault == "number_path":
        record.write_text(json.dumps({**row, "path": 7}), encoding="utf-8")
    elif fault == "malformed_and_scratch_gone":  # a missing directory is no licence for a malformed record
        keep = pathlib.Path(tempfile.mkdtemp(prefix="obs-", dir=parent))
        record.write_text(json.dumps({**row, "path": str(parent / "obs-gone"), "identity": "x"}), encoding="utf-8")
    elif fault == "foreign_parent":
        keep = tmp_path / "elsewhere" / "obs-foreign"
        keep.mkdir(parents=True)
        seen = keep.lstat()
        record.write_text(json.dumps({**row, "path": str(keep), "identity": [seen.st_dev + 1, seen.st_ino]}), encoding="utf-8")
    elif fault == "linked_path":
        link = parent / "obs-link"
        try:
            link.symlink_to(scratch, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        seen = scratch.lstat()
        record.write_text(json.dumps({**row, "path": str(link), "identity": [seen.st_dev + 1, seen.st_ino]}), encoding="utf-8")
    elif fault == "not_directory":
        keep = parent / "obs-file"
        keep.write_text("a file\n", encoding="utf-8")
        seen = keep.lstat()
        record.write_text(json.dumps({**row, "path": str(keep), "identity": [seen.st_dev + 1, seen.st_ino]}), encoding="utf-8")
    elif fault == "linked_record":
        real = record.with_name("real.json")
        record.rename(real)
        try:
            record.symlink_to(real)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
    return data, record, keep


FAULTS = ["malformed_json", "no_identity", "bool_identity", "short_identity", "text_identity", "number_path",
          "malformed_and_scratch_gone", "foreign_parent", "linked_path", "not_directory", "linked_record"]


@pytest.mark.parametrize("fault", FAULTS)
def test_an_invalid_record_stays_an_error_that_keeps_everything(tmp_path, fault):
    data, record, keep = _fault(tmp_path, fault)
    drive = data / "task_drives" / "task-1"
    before = _tree(keep) if keep.is_dir() else keep.read_bytes()
    with pytest.raises(OSError, match="invalid scratch ownership record"):
        session_scratch.release_session_scratch(drive)
    assert (_tree(keep) if keep.is_dir() else keep.read_bytes()) == before
    [row] = session_scratch.scratch_records(data, live=lambda _: True)
    assert not row["valid"] and not row["missing"] and row["hold"] != "identity_unverified" and row["problem"]
    entry = _settled(data)
    report = task_custody.remove_custody_trash(data)
    assert list(report["stuck"]) == [entry.name] and [h["class"] for h in report["holds"]] == ["removal_failed"]
    assert (_tree(keep) if keep.is_dir() else keep.read_bytes()) == before and entry.is_dir()


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_a_record_that_cannot_be_read_is_no_evidence_its_scratch_is_gone(tmp_path, monkeypatch, error):
    """A vanished record (FileNotFoundError while reading it) is not the scratch's absence: the release fails
    and keeps the scratch, Storage reports the record unreadable, never ``missing``."""
    data, scratch = _started(tmp_path)
    record, read = _record(data), pathlib.Path.read_text
    before = _tree(scratch)

    def unreadable(path, *args, **kwargs):
        if path == record:
            raise error(errno.ENOENT if error is FileNotFoundError else errno.EACCES, "record read failed", str(path))
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", unreadable)
    with pytest.raises(OSError) as raised:
        session_scratch.release_session_scratch(data / "task_drives" / "task-1")
    assert not isinstance(raised.value, FileNotFoundError)
    assert _tree(scratch) == before
    [row] = session_scratch.scratch_records(data, live=lambda _: True)
    assert not row["valid"] and not row["missing"] and row["hold"] == ""
    assert session_scratch.recorded_scratch_roots(data) == []


def test_a_new_start_still_refuses_the_unverified_directory(tmp_path):
    first = _prepare(tmp_path, _Catalog(GRANTED))
    scratch = pathlib.Path(first.scratch)
    data = tmp_path / "data"
    _unverify(_record(data))
    record_bytes, before = _record(data).read_bytes(), _tree(scratch)
    again = _prepare(tmp_path, _Catalog(GRANTED))
    assert again.roots == () and "not the recorded scratch directory" in again.limitation
    assert _record(data).read_bytes() == record_bytes and _tree(scratch) == before
    [row] = session_scratch.scratch_records(data, live=lambda _: True)
    assert row["holder"] == "task" and row["hold"] == "identity_unverified" and not row["valid"]


def test_a_live_or_kept_tasks_record_waits_for_its_task_whatever_its_identity(tmp_path):
    data, scratch = _started(tmp_path, status="running")
    _unverify(_record(data))
    record_bytes = _record(data).read_bytes()
    report = release_finished_scratch(data, lambda _task: True)
    assert report["released"] == [] and [h["class"] for h in report["holds"]] == ["task_not_terminal"]
    assert _record(data).read_bytes() == record_bytes and scratch.is_dir()


def _nested_entry(tmp_path):
    """A retired drive with a headless child's records: a renumbered one and a proven one."""
    data = tmp_path / "data"
    parent = test_environment.session_scratch_parent()
    entry = data.joinpath(*TRASH, "task-1-0123456789ab")
    rows = {}
    for key, renumber in (("held", True), ("plain", False)):
        scratch = pathlib.Path(tempfile.mkdtemp(prefix="obs-", dir=parent))
        (scratch / "file.txt").write_text(key, encoding="utf-8")
        seen = scratch.lstat()
        owner = entry / "task_drives" / "child-1" / SESSION_SCRATCH_DIRNAME / key
        owner.mkdir(parents=True)
        (owner / "scratch.json").write_text(json.dumps({"schema_version": 1, "path": str(scratch),
                                                       "identity": [seen.st_dev + renumber, seen.st_ino]}), encoding="utf-8")
        rows[key] = (scratch, (owner / "scratch.json").read_bytes())
    return data, entry, rows


def test_a_nested_drives_unverified_record_is_held_and_its_sibling_released(tmp_path):
    data, entry, rows = _nested_entry(tmp_path)
    held, held_bytes = rows["held"]
    before = _tree(held)
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], report["stuck"], [h["class"] for h in report["holds"]]) == (1, {}, ["identity_unverified"])
    assert not rows["plain"][0].exists() and _tree(held) == before and not entry.exists()
    [handoff] = data.joinpath(*TRASH).iterdir()
    assert handoff.name.startswith("task-1+scratch-")
    assert (handoff / SESSION_SCRATCH_DIRNAME / "held" / "scratch.json").read_bytes() == held_bytes
    assert task_custody.remove_custody_trash(data)["holds"] == report["holds"]
    assert list(data.joinpath(*TRASH).iterdir()) == [handoff]


def test_a_failed_handoff_keeps_the_record_where_it_was(tmp_path, monkeypatch):
    data, held, plain, _recorded, record_bytes = _two_scratch_drive(tmp_path)
    entry = _settled(data)
    before, real_replace = _tree(held), os.replace

    def refused(src, dst):
        if pathlib.Path(src).name == "scratch.json":
            raise OSError(errno.ENOSPC, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", refused)
    report = task_custody.remove_custody_trash(data)
    assert list(report["stuck"]) == [entry.name]
    assert sorted(h["class"] for h in report["holds"]) == ["identity_unverified", "removal_failed"]
    assert (entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").read_bytes() == record_bytes
    assert not list(data.joinpath("state", "custody_staging").glob("*")) and _tree(held) == before
    monkeypatch.setattr(os, "replace", real_replace)
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], [h["class"] for h in report["holds"]]) == (1, ["identity_unverified"])
    [handoff] = data.joinpath(*TRASH).iterdir()
    assert (handoff / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").read_bytes() == record_bytes
    assert _tree(held) == before and not plain.exists()


def test_a_transfer_stopped_in_staging_stays_recoverable_and_is_published_once(tmp_path, monkeypatch):
    data, held, _plain, _recorded, record_bytes = _two_scratch_drive(tmp_path)
    entry = _settled(data)
    before, real_publish = _tree(held), workspace_preflight._publish_released

    def stopped(root, staged):
        raise OSError(errno.EIO, "stopped before publication")

    monkeypatch.setattr(workspace_preflight, "_publish_released", stopped)
    report = task_custody.remove_custody_trash(data)
    assert list(report["stuck"]) == [entry.name]
    [staged] = data.joinpath("state", "custody_staging").iterdir()
    assert staged.name.startswith("task-1+scratch-")
    assert (staged / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").read_bytes() == record_bytes
    task_custody.sweep_custody_leftovers(data, min_age_sec=0)
    assert staged.is_dir(), "a scratch release's staged entry is never swept as a leftover"
    monkeypatch.setattr(workspace_preflight, "_publish_released", real_publish)
    assert task_custody.remove_custody_trash(data)["removed"] == 1  # the drive no longer names the record
    release_finished_scratch(data, OVER)  # the next release pass publishes what the stopped one staged
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], [h["class"] for h in report["holds"]]) == (0, ["identity_unverified"])
    [handoff] = data.joinpath(*TRASH).iterdir()
    assert handoff.name == staged.name
    assert (handoff / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json").read_bytes() == record_bytes
    assert _tree(held) == before


def test_a_hold_whose_directory_a_person_removed_goes_with_its_entry(tmp_path):
    data, held, _plain, _recorded, _bytes = _two_scratch_drive(tmp_path)
    _settled(data)
    assert task_custody.remove_custody_trash(data)["holds"][0]["class"] == "identity_unverified"
    for item in sorted(held.rglob("*"), reverse=True):
        item.rmdir() if item.is_dir() else item.unlink()
    held.rmdir()
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "holds": []}
    assert not list(data.joinpath(*TRASH).iterdir())


def test_free_never_offers_an_unverified_scratch_even_one_its_producer_marked(tmp_path, monkeypatch):
    """Free's producer-retained discard reads only valid records: an unverified handoff is not offered, while
    the same handoff with its identity intact is (the control)."""
    from tests.test_storage_free import _fake_sizer, _owner, _preview

    data, scratch = _started(tmp_path)
    nested = scratch / "held" / "run.noindex"
    nested.mkdir(parents=True)
    reason = "producer kept this evidence"
    assert test_environment.retain_tree(nested, reason)
    assert release_finished_scratch(data, OVER)["released"]
    assert [h["class"] for h in task_custody.remove_custody_trash(data)["holds"]] == ["producer_retained"]
    [entry] = data.joinpath(*TRASH).iterdir()
    _fake_sizer(monkeypatch)

    def offered():
        owner = _owner({}, offered=("service_folders",))
        _preview(owner, data, data.parent)
        view = owner.projection(data)
        return view["pool"]["manual"]["count"], owner.items(data, view["plan"], "manual", 0, 500, None)["items"]

    count, items = offered()
    assert count == 1 and [item["name"] for item in items] == ["task-1"], "control: an intact handoff is offered"
    _unverify(entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json")
    count, items = offered()
    assert count == 0 and items == []
    assert [h["class"] for h in task_custody.remove_custody_trash(data)["holds"]] == ["identity_unverified"]
    assert (nested / test_environment.RETENTION_MARKER).is_file() and scratch.is_dir()
    assert retire_synthetic_retention(nested, reason)  # the marker going does not release an unverified scratch
    assert [h["class"] for h in task_custody.remove_custody_trash(data)["holds"]] == ["identity_unverified"]
    assert scratch.is_dir()
