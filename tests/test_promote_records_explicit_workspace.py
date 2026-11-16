"""T5: a promote that names an explicit ``workspace_root`` for a project whose
registry ``working_dir`` is EMPTY records that folder — through the existing
``validate_workspace_root`` canonicalization and an atomic empty-field update
under the projects-registry lock — so the project room is not blind forever.
A non-empty ``working_dir`` is never overwritten (the v6.58.0 invariant), an
invalid path is refused as before, a host-managed temporary copy runs its task but
is never recorded, and two concurrent promotes elect one winner.
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


@pytest.mark.parametrize("kind", [
    "copies_root",      # a helper's copy under the host's working-copies root
    "inside_recorded",  # a subfolder of a recorded copy that lives outside that root
    "third_spelling",   # the same, its record written under another case spelling of that root
    "unknown_record",   # a folder whose own record cannot be read: unknown, never called a copy
])
def test_a_host_managed_copy_runs_the_task_but_never_becomes_the_room_folder(tmp_path, monkeypatch, kind):
    """A helper's copy is a valid task workspace; recording it would bind the room to
    a folder the host removes when the helper's work ends."""
    import supervisor.workers as workers
    from ouroboros import subagent_worktrees as sw
    from ouroboros.projects_registry import create_project, get_project

    drive = _drive(tmp_path, monkeypatch)
    copies = tmp_path / "copies"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(copies))
    if kind == "copies_root":
        copy = copies / "copies.noindex" / "dlg_helper_run"
    elif kind == "unknown_record":
        copy = tmp_path / "older_root" / "dlg_helper_run"
        sw._record_file(copy, drive).parent.mkdir(parents=True)
        sw._record_file(copy, drive).write_text("{", encoding="utf-8")
    else:  # kept as Older_Root, recorded as OLDER_ROOT, asked as older_root where the volume reads them as one
        recorded, copy = tmp_path / "older_root" / "dlg_helper_run", tmp_path / "older_root" / "dlg_helper_run" / "src"
        if kind == "third_spelling":
            (tmp_path / "Older_Root").mkdir()
            if not (tmp_path / "OLDER_ROOT").exists() or not (tmp_path / "older_root").exists():
                pytest.skip("this volume keeps these case spellings apart: they name different folders")
            recorded = tmp_path / "OLDER_ROOT" / "dlg_helper_run"
        copy.mkdir(parents=True)
        sw._put_record({"path": str(recorded.resolve()), "task_id": "helper", "kind": "delegated_exec"}, data_dir=drive)
    copy.mkdir(parents=True, exist_ok=True)
    create_project(drive, "weekly-room", name="Weekly")
    generation = get_project(drive, "weekly-room")["routing_generation"]
    enqueued: list = []

    outcome = _promote(workers, "fromcopy", "weekly-room", str(copy), enqueued)

    assert outcome["status"] == "scheduled", outcome
    assert enqueued[0]["workspace_root"] == str(copy.resolve()) and "_room_folder_note" not in enqueued[0]
    room = get_project(drive, "weekly-room")
    assert room["working_dir"] == "" and room["routing_generation"] == generation
    assert outcome["source_note"].startswith("the task's workspace stays; the Project's folder stays unset: ")
    assert ("removed when" in outcome["source_note"]) is (kind != "unknown_record")
    assert ("is unknown" in outcome["source_note"]) is (kind == "unknown_record")


def test_non_empty_working_dir_is_never_overwritten_by_an_explicit_root(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, get_project, update_project

    drive = _drive(tmp_path, monkeypatch)
    registered = tmp_path / "registered"
    registered.mkdir()
    create_project(drive, "settled-room", name="Settled", working_dir=str(registered))
    before = update_project(drive, "settled-room", provenance="cloned", clone_url="https://example.test/old.git",
                            trusted_at="2026-11-01T00:00:00+00:00", pinned=True)
    other = tmp_path / "other"
    other.mkdir()
    enqueued: list = []

    outcome = _promote(workers, "explicit2", "settled-room", str(other), enqueued)

    assert outcome["status"] == "scheduled", outcome
    assert enqueued[0]["workspace_root"] == str(other.resolve())  # the task uses what it asked for
    after = get_project(drive, "settled-room")
    # Promotion touches activity; a skipped CAS preserves the folder, trust, provenance and other metadata.
    assert {key: value for key, value in after.items() if key != "last_active_at"} == {
        key: value for key, value in before.items() if key != "last_active_at"}


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
