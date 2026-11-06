"""Absence after a failed read: Windows obstructions and dangling entries are unknown."""
from __future__ import annotations

import errno
import os
import pathlib
import stat

import pytest

from supervisor import evolution_lifecycle, queue, queue_schedules, state, state_initialization
from tests.test_state_authority import _windows_shaped_not_found


@pytest.mark.parametrize(("errors", "consumer"), [("native", "panic"), ("windows", "schedule")])
def test_panic_and_schedule_readers_require_proven_absence(tmp_path, monkeypatch, consumer, errors):
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setitem(evolution_lifecycle._STOP_LATCH, "stopped", False)
    if errors == "windows":
        _windows_shaped_not_found(monkeypatch)
    path = tmp_path / "state" / ("panic_stop.flag" if consumer == "panic" else "scheduled_tasks.json")

    def unknown():
        if consumer == "panic":
            assert evolution_lifecycle.evolution_stop_reason({}) == "panic_flag_unknown"
        else:
            with pytest.raises(queue_schedules.ScheduleStoreUnreadable):
                queue_schedules.load_schedule_store(tmp_path)

    def absent():
        if consumer == "panic":
            assert evolution_lifecycle.evolution_stop_reason({}) == ""
        else:
            assert queue_schedules.load_schedule_store(tmp_path) == {"schema_version": 1, "tasks": []}

    path.parent.write_bytes(b"obstruction")
    unknown()
    assert path.parent.read_bytes() == b"obstruction"
    path.parent.unlink()
    absent()  # genuinely missing directory
    path.parent.mkdir()
    absent()  # genuinely missing leaf below a directory
    raw = b"panic" if consumer == "panic" else b'{"schema_version":1,"tasks":[{"id":"kept"}]}'
    path.write_bytes(raw)
    if consumer == "panic":
        assert evolution_lifecycle.evolution_stop_reason({}) == "panic_flag"
    else:
        assert queue_schedules.load_schedule_store(tmp_path)["tasks"] == [{"id": "kept"}]
    real_lstat = os.lstat

    def denied(candidate, *args, **kwargs):
        if pathlib.Path(candidate) == path:
            raise PermissionError(errno.EACCES, "unreadable marker/table", str(path))
        return real_lstat(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", denied)
    monkeypatch.setattr(pathlib.Path, "lstat", denied)
    unknown()
    assert path.read_bytes() == raw


@pytest.mark.parametrize(("depth", "errors"), [(0, "native"), (1, "windows"), (3, "native")])
def test_nondirectory_at_or_above_data_root_is_unknown(tmp_path, monkeypatch, errors, depth):
    obstruction = tmp_path / "file"
    obstruction.write_bytes(b"retained")
    root = obstruction.joinpath(*[f"missing{i}" for i in range(depth)])
    if errors == "windows":
        _windows_shaped_not_found(monkeypatch)
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setitem(evolution_lifecycle._STOP_LATCH, "stopped", False)
    assert state.read_state_copy(root / "state/state.json")[0] == "unreadable"
    assert state_initialization.read_witness(root)[0] == "unreadable"
    assert state_initialization.supervisor_evidence(root)[0] == "unknown"
    assert evolution_lifecycle.evolution_stop_reason({}) == "panic_flag_unknown"
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable):
        queue_schedules.load_schedule_store(root)
    assert obstruction.read_bytes() == b"retained"


@pytest.mark.parametrize("entry", ["state", "logs", "task_results"])
def test_dangling_directory_entry_cannot_prove_no_supervisor_history(tmp_path, monkeypatch, entry):
    # A synthetic dangling directory entry also exercises this on Windows without
    # requiring permission to create symlinks. All reads/stat calls still see ENOENT.
    link = tmp_path / entry
    real_lstat = os.lstat

    def dangling(candidate, *args, **kwargs):
        if pathlib.Path(candidate) == link:
            return os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 1, 0, 0, 0))
        return real_lstat(candidate, *args, **kwargs)

    assert state_initialization.supervisor_evidence(tmp_path) == ("none", "")
    monkeypatch.setattr(os, "lstat", dangling)
    assert state_initialization.supervisor_evidence(tmp_path)[0] == "unknown"
    assert state_initialization.initialization_decision(tmp_path)["create"] is False
    assert not (tmp_path / "state").exists()  # no witness was minted
    if entry == "state":
        assert state.read_state_copy(tmp_path / "state/state.json")[0] == "unreadable"
        assert state_initialization.read_witness(tmp_path)[0] == "unreadable"
        monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
        monkeypatch.setitem(evolution_lifecycle._STOP_LATCH, "stopped", False)
        assert evolution_lifecycle.evolution_stop_reason({}) == "panic_flag_unknown"
        with pytest.raises(queue_schedules.ScheduleStoreUnreadable):
            queue_schedules.load_schedule_store(tmp_path)


@pytest.mark.parametrize("entry", ["state.json", "state.initialized.json"])
def test_dangling_leaf_is_unreadable_not_missing(tmp_path, monkeypatch, entry):
    directory = tmp_path / "state"
    directory.mkdir()
    path = directory / entry
    real_lstat = os.lstat

    def dangling(candidate, *args, **kwargs):
        if pathlib.Path(candidate) == path:
            return os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 1, 0, 0, 0))
        return real_lstat(candidate, *args, **kwargs)

    assert state.read_state_copy(path)[0] == "missing"
    monkeypatch.setattr(os, "lstat", dangling)
    assert state.read_state_copy(path)[0] == "unreadable"
    if entry == "state.initialized.json":
        assert state_initialization.read_witness(tmp_path)[0] == "unreadable"
        assert state_initialization.initialization_decision(tmp_path)["create"] is False
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(("errors", "entry"), [("native", "logs"), ("windows", "task_results")])
def test_obstructed_evidence_directory_is_unknown(tmp_path, monkeypatch, entry, errors):
    path = tmp_path / entry
    path.write_bytes(b"retained")
    if errors == "windows":
        _windows_shaped_not_found(monkeypatch)
        # Even if scandir reports FileNotFoundError for the leaf obstruction,
        # evidence is unknown. Absence proof must inspect that entry too.
        real_scandir = os.scandir

        def not_found(candidate):
            if pathlib.Path(candidate) == path:
                raise FileNotFoundError(errno.ENOENT, "path not found", str(path))
            return real_scandir(candidate)

        monkeypatch.setattr(os, "scandir", not_found)
    assert state_initialization.supervisor_evidence(tmp_path)[0] == "unknown"
    assert state_initialization.initialization_decision(tmp_path)["create"] is False
    assert path.read_bytes() == b"retained" and not (tmp_path / "state").exists()
    path.unlink()
    assert state_initialization.supervisor_evidence(tmp_path) == ("none", "")


def test_directory_symlink_allows_proven_missing_children(tmp_path, monkeypatch):
    directory = tmp_path / "state"
    directory.mkdir()
    real_lstat = os.lstat

    def linked_directory(candidate, *args, **kwargs):
        if pathlib.Path(candidate) == directory:
            return os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 1, 0, 0, 0))
        return real_lstat(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", linked_directory)
    assert state.read_state_copy(directory / "state.json")[0] == "missing"
    assert state_initialization.read_witness(tmp_path) == ("missing", {})
    assert state_initialization.supervisor_evidence(tmp_path) == ("none", "")


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_failed_ancestor_proof_stays_unknown(tmp_path, monkeypatch, error):
    real_stat = os.stat

    def unreadable(candidate, *args, **kwargs):
        if pathlib.Path(candidate) == tmp_path / "state":
            raise error(errno.EACCES if error is PermissionError else errno.EIO, "cannot examine")
        return real_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "stat", unreadable)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setitem(evolution_lifecycle._STOP_LATCH, "stopped", False)
    assert state.read_state_copy(tmp_path / "state/state.json")[0] == "unreadable"
    assert state_initialization.read_witness(tmp_path)[0] == "unreadable"
    assert state_initialization.supervisor_evidence(tmp_path)[0] == "unknown"
    assert evolution_lifecycle.evolution_stop_reason({}) == "panic_flag_unknown"
    with pytest.raises(queue_schedules.ScheduleStoreUnreadable):
        queue_schedules.load_schedule_store(tmp_path)
