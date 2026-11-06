"""T5: a promote that names an explicit ``workspace_root`` for a project whose
registry ``working_dir`` is EMPTY records that folder — through the existing
``validate_workspace_root`` canonicalization and an atomic empty-field update
under the projects-registry lock — so the project room is not blind forever.
A non-empty ``working_dir`` is never overwritten (the v6.58.0 invariant), an
invalid path is refused as before, and two concurrent promotes elect one winner.
"""
from __future__ import annotations

import pathlib
import threading
import types

import pytest


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path_factory.mktemp("projects_root")))


def _drive(tmp_path: pathlib.Path, monkeypatch) -> pathlib.Path:
    """A data drive apart from the folders: a workspace may not overlap the drive."""
    import supervisor.workers as workers

    drive = tmp_path / "data"
    drive.mkdir()
    monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
    return drive


def _promote_ctx(enqueued: list) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
        send_with_budget=lambda *args, **kwargs: None,
    )


def _promote(workers, task_id: str, project_id: str, workspace_root: str, enqueued: list) -> dict:
    return workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": task_id,
        "objective": "Continue the project",
        "project_id": project_id,
        "workspace_root": workspace_root,
        "chat_id": 1,
    }, _promote_ctx(enqueued))


def test_explicit_workspace_root_is_recorded_into_an_empty_working_dir(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, get_project

    drive = _drive(tmp_path, monkeypatch)
    create_project(drive, "blind-room", name="Blind Room")
    folder = tmp_path / "somewhere" / "folder"
    folder.mkdir(parents=True)
    enqueued: list = []

    outcome = _promote(workers, "explicit1", "blind-room", str(folder), enqueued)

    assert outcome["status"] == "scheduled", outcome
    assert enqueued[0]["workspace_root"] == str(folder.resolve())
    # The room now knows its folder: canonical (resolved) spelling, not the raw text.
    assert get_project(drive, "blind-room")["working_dir"] == str(folder.resolve())


def test_non_empty_working_dir_is_never_overwritten_by_an_explicit_root(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, get_project

    drive = _drive(tmp_path, monkeypatch)
    registered = tmp_path / "registered"
    registered.mkdir()
    create_project(drive, "settled-room", name="Settled", working_dir=str(registered))
    other = tmp_path / "other"
    other.mkdir()
    enqueued: list = []

    outcome = _promote(workers, "explicit2", "settled-room", str(other), enqueued)

    assert outcome["status"] == "scheduled", outcome
    assert enqueued[0]["workspace_root"] == str(other.resolve())  # the task uses what it asked for
    assert get_project(drive, "settled-room")["working_dir"] == str(registered)  # untouched


def test_invalid_explicit_root_is_refused_and_nothing_is_recorded(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, get_project

    drive = _drive(tmp_path, monkeypatch)
    create_project(drive, "blind-room-2", name="Blind Room 2")
    enqueued: list = []

    outcome = _promote(workers, "explicit3", "blind-room-2", str(tmp_path / "does-not-exist"), enqueued)

    assert outcome["status"] == "needs_manual_target" and outcome["reason"] == "workspace_unusable", outcome
    assert enqueued == []
    assert str(get_project(drive, "blind-room-2").get("working_dir") or "") == ""


def test_two_concurrent_promotes_elect_one_working_dir(tmp_path, monkeypatch):
    """The empty-field update is atomic under the registry lock: two promotes
    racing with different explicit folders leave exactly one of them recorded."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, get_project

    drive = _drive(tmp_path, monkeypatch)
    create_project(drive, "raced-room", name="Raced")
    folders = [tmp_path / f"folder{i}" for i in range(2)]
    for folder in folders:
        folder.mkdir()
    enqueued: list = []
    errors: list = []
    gate = threading.Barrier(2)

    def run(index: int) -> None:
        try:
            gate.wait(timeout=5)
            _promote(workers, f"race{index}", "raced-room", str(folders[index]), enqueued)
        except Exception as exc:  # pragma: no cover - surfaces as a failure below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors, errors
    assert len(enqueued) == 2
    recorded = get_project(drive, "raced-room")["working_dir"]
    assert recorded in {str(f.resolve()) for f in folders}, recorded
    # Each task still runs in the folder it asked for; only the room's record was contested.
    assert {t["workspace_root"] for t in enqueued} == {str(f.resolve()) for f in folders}


def test_update_project_only_if_empty_is_a_compare_and_set(tmp_path):
    from ouroboros.projects_registry import create_project, update_project

    create_project(tmp_path, "cas-room", name="CAS")
    first = update_project(tmp_path, "cas-room", working_dir="/a", only_if_empty=("working_dir",))
    assert first["working_dir"] == "/a"
    second = update_project(tmp_path, "cas-room", working_dir="/b", only_if_empty=("working_dir",))
    assert second["working_dir"] == "/a"  # the loser reads the winner's value back
    plain = update_project(tmp_path, "cas-room", working_dir="/c")
    assert plain["working_dir"] == "/c"  # an ordinary update still overwrites


@pytest.mark.parametrize(("choice", "stage"), [
    ("explicit", "create_project"),
    ("none", "bind_task_to_project"),
    ("room_default", "create_project"),
])
def test_room_default_drift_respects_the_promoted_resource_choice(tmp_path, monkeypatch, stage, choice):
    import supervisor.workers as workers
    from ouroboros import projects_registry as registry

    drive = _drive(tmp_path, monkeypatch)
    registry.create_project(drive, "changing-room")
    explicit, other = tmp_path / "explicit", tmp_path / "other"
    explicit.mkdir()
    other.mkdir()
    original = getattr(registry, stage)

    def changed(*args, **kwargs):
        registry.update_project(drive, "changing-room", working_dir=str(other))
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, stage, changed)
    enqueued = []
    outcome = workers.promote_chat_to_task({
        "task_id": "drift", "objective": "Use the selected resource", "project_id": "changing-room",
        "workspace_root": str(explicit) if choice == "explicit" else "",
        "workspace": "none" if choice == "none" else "", "chat_id": 1,
    }, _promote_ctx(enqueued))
    if choice == "room_default":
        assert outcome["status"] == "needs_manual_target" and not enqueued
    else:
        assert outcome["status"] == "scheduled" and len(enqueued) == 1
        assert enqueued[0].get("workspace_root", "") == (str(explicit.resolve()) if choice == "explicit" else "")
        assert enqueued[0]["_project_admission"]["frozen"] is True
    assert registry.get_project(drive, "changing-room")["working_dir"] == str(other)
