"""Inherited executor records are indexed once; the exit finds records through the ownership set."""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.serial
NAME = "workspace_executor_processes"


def _record(folder, name="service", kind="service"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + ".json")
    path.write_text(json.dumps({
        "id": name, "schema_version": 1, "owner": "ouroboros_workspace_executor",
        "record_type": kind, "executor_type": "docker_exec", "executor_id": "fixture",
        "container_name": "never-contact-this-fixture", "backend_pid": "42",
        "backend_pidfile": "/tmp/ouroboros-exec-fixture.pid", "service_id": name,
    }), encoding="utf-8")
    return path


def _indexed(root):
    from ouroboros import owned_shutdown

    return {Path(entry["record_path"]) for entry in owned_shutdown.owned_records(root)}


def test_import_matches_recursive_roots_and_directory_symlinks(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown

    root = tmp_path / "data"
    state = root / "state"
    expected = {
        _record(state / NAME, "canonical"),
        _record(state / "headless_tasks/old/data/state" / NAME, "child"),
        _record(state / "arbitrary/.hidden/evolved" / NAME, "nonlayout"),
        _record(state / "arbitrary/.hidden/evolved" / NAME / "nested" / NAME, "nested"),
        _record(root / "task_drives/t1/state" / NAME, "task-drive"),
    }
    target = tmp_path / "record-target"
    _record(target, "linked")
    parent_target = tmp_path / "parent-target"
    _record(parent_target / "nested" / NAME, "not-descended")
    named = state / "linked" / NAME
    named.parent.mkdir()
    (state / "broken").mkdir()
    try:
        named.symlink_to(target, target_is_directory=True)
        (state / "parent-link").symlink_to(parent_target, target_is_directory=True)
        (state / "loop-link").symlink_to(state, target_is_directory=True)
        (state / "broken" / NAME).symlink_to(tmp_path / "absent", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable on this host")
    expected.add(named / "linked.json")
    assert owned_shutdown.import_inherited_records(root) == len(expected)
    assert _indexed(root) == expected
    # The retired exit walk's matching contract over data/state, plus data/task_drives.
    previous = {path for base in (state, root / "task_drives") for folder in base.rglob(NAME)
                if folder.is_dir() for path in folder.glob("*.json")}
    assert previous == expected


def test_import_runs_once_and_never_walks_again(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown

    root = tmp_path / "data"
    first = _record(root / "state" / NAME, "first")
    assert owned_shutdown.import_inherited_records(root) == 1
    _record(root / "task_drives/t2/state" / NAME, "late")
    monkeypatch.setattr(os, "walk", lambda *_a, **_k: pytest.fail("the import walks only once"))
    assert owned_shutdown.import_inherited_records(root) is None
    assert owned_shutdown.start_inherited_import(root) is None
    assert owned_shutdown.finish_unconfirmed_stops(root)["retried"] == 0
    assert _indexed(root) == {first}


def test_an_import_landing_after_the_stop_began_is_born_stamped(tmp_path, monkeypatch):
    """The background import publishes after the generation's stop selected its targets: the inherited
    targets are stored stamped, so the next start finishes them instead of skipping them."""
    from ouroboros import owned_shutdown

    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    root = tmp_path / "data"
    inherited = _record(root / "task_drives/t1/state" / NAME, "inherited")
    owned_shutdown.begin_owned_stop(root)
    assert owned_shutdown.import_inherited_records(root) == 1
    stored = json.loads(owned_shutdown.owned_processes_path(root).read_text(encoding="utf-8"))["records"]
    [entry] = stored.values()
    assert entry["record_path"] == str(inherited) and entry["stop_requested_at"]


def test_the_inherited_walk_runs_off_the_ready_path(tmp_path, monkeypatch):
    """A first start answers before the walk ends: the retry never walks, the background import walks once."""
    import threading

    from ouroboros import owned_shutdown

    root = tmp_path / "data"
    first = _record(root / "task_drives/t1/state" / NAME, "inherited")
    release, walking = threading.Event(), threading.Event()
    original = os.walk

    def slow_walk(*args, **kwargs):
        walking.set()
        assert release.wait(10), "the walk was waited on"
        return original(*args, **kwargs)

    monkeypatch.setattr(os, "walk", slow_walk)
    assert owned_shutdown.finish_unconfirmed_stops(root)["retried"] == 0
    assert not walking.is_set(), "the stamped-stop retry walked the disk"
    thread = owned_shutdown.start_inherited_import(root)
    assert walking.wait(10) and thread.is_alive()  # the caller already returned
    release.set()
    thread.join(10)
    assert _indexed(root) == {first}
    row = json.loads((root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["type"] == "owned_records_imported" and row["records"] == 1
    assert owned_shutdown.start_inherited_import(root) is None


def test_import_does_not_stat_a_candidate_in_every_unrelated_directory(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown

    plain, busy = tmp_path / "plain", tmp_path / "busy"
    for root in (plain, busy):
        _record(root / "state" / NAME)
    for index in range(100):
        (busy / "state" / "payload" / str(index) / "cache").mkdir(parents=True)
    original = Path.stat
    candidates = []

    def stat(path, *args, **kwargs):
        if path.name == NAME:
            candidates.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    counts = []
    for root in (plain, busy):
        candidates.clear()
        assert owned_shutdown.import_inherited_records(root) == 1
        counts.append(len(candidates))
    assert counts[0] == counts[1]


def test_service_cleanup_finds_a_record_published_after_the_foreground_stop(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown, workspace_executor as executor

    root = tmp_path / "data"
    foreground = _record(root / "state" / NAME, "foreground", "foreground")
    owned_shutdown.record_executor_process(foreground, json.loads(foreground.read_text(encoding="utf-8")))
    monkeypatch.setattr(executor, "_FOREGROUND", {})
    monkeypatch.setattr(executor, "_SERVICES", {})
    calls = []

    def stop(row, *, wait):
        calls.append(row["id"])
        if row["record_type"] == "foreground":
            late = _record(root / "task_drives/late-owner/state" / NAME, "late-service")  # a task drive
            owned_shutdown.record_executor_process(late, json.loads(late.read_text(encoding="utf-8")))
        return True

    monkeypatch.setattr(executor, "_kill_docker_record", stop)
    monkeypatch.setattr(executor, "_retire_docker_completion", lambda path: True)
    monkeypatch.setattr(os, "walk", lambda *_a, **_k: pytest.fail("exit cleanup never walks data/state"))
    assert len(executor.kill_all_foreground(root, wait=False)) == 1
    stopped = executor.kill_all_services(root, wait=False)
    assert calls == ["foreground", "late-service"]
    assert [row["service_id"] for row in stopped] == ["late-service"]
    assert not [path for path in root.rglob("*.json") if path.parent.name == NAME]
    assert owned_shutdown.owned_records(root) == []
