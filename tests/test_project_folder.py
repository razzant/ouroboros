"""A Project's working folder: the recorded No folder and the one door that reads and repairs it.

The registry row is the authority (``no_folder`` beside ``working_dir``); ``project_folder`` is the one
view and change the Project menu's route and the mind's ``manage_project_folder`` both call. A recorded
No folder stands until a folder is chosen for the Project itself: neither the first promoted task nor a
task's own ``workspace_root`` replaces it, while a row that made no choice keeps provisioning one.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ORIGIN = {"absent": "mid_task_no_origin"}


@pytest.fixture(autouse=True)
def _isolated_work_roots(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path_factory.mktemp("projects_root")))


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A data drive apart from the folders (a folder may not overlap the drive), as the supervisor's own."""
    import supervisor.workers as workers

    drive = tmp_path / "data"
    drive.mkdir()
    monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
    return drive


def _folder(tmp_path, name: str) -> pathlib.Path:
    folder = tmp_path / "folders" / name
    folder.mkdir(parents=True)
    return folder.resolve()


def _view(data, project_id="room"):
    from ouroboros import project_folder

    return project_folder.folder_view(data, project_id, repo_dir=REPO)


def _change(data, action, *, expected=None, path="", project_id="room", actor="owner"):
    from ouroboros import project_folder

    expected = _view(data, project_id)["expected"] if expected is None else expected
    return project_folder.change_folder(data, project_id, action=action, expected=expected, actor=actor,
                                        repo_dir=REPO, path=path)


def _events(data, kind="project_folder_changed"):
    log = data / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return [row for row in rows if row.get("type") == kind]


def _promote_ctx(enqueued: list) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
        send_with_budget=lambda *args, **kwargs: None,
    )


def _promote(task_id: str, enqueued: list, **event) -> dict:
    import supervisor.workers as workers

    return workers.promote_chat_to_task({"type": "promote_chat_to_task", "task_id": task_id, "objective": "Do the work",
                                         "project_id": "room", "chat_id": 1, **event}, _promote_ctx(enqueued))


# --- the registry's transition ---------------------------------------------------------------------------------

def test_no_folder_is_its_own_record_apart_from_provenance_and_survives_implicit_writers(data, tmp_path):
    from ouroboros.project_sources import NoFolderChosen
    from ouroboros.projects_registry import create_project, ensure_project_workspace, get_project, update_project

    create_project(data, "room", name="Room")
    update_project(data, "room", provenance="none")  # how every source-less room was always stamped
    legacy = get_project(data, "room")
    assert "no_folder" not in legacy and legacy["routing_generation"] == 0

    chosen = update_project(data, "room", no_folder={"at": "2026-11-15T12:00:00+00:00", "by": "owner"})
    assert chosen["no_folder"] == {"at": "2026-11-15T12:00:00+00:00", "by": "owner"}
    assert (chosen["working_dir"], chosen["routing_generation"], chosen["provenance"]) == ("", 1, "none")
    # A repeat keeps the first record and starts no new generation.
    again = update_project(data, "room", no_folder={"at": "2026-11-16T00:00:00+00:00", "by": "ouroboros:t"})
    assert again["no_folder"]["by"] == "owner" and again["routing_generation"] == 1

    folder = _folder(tmp_path, "task-only")
    before = (data / "state" / "projects.json").read_bytes()
    # The compare-and-set a task's own folder uses reads the choice as set; a plain writer is refused.
    kept = update_project(data, "room", working_dir=str(folder), only_if_empty=("working_dir",))
    assert kept["working_dir"] == "" and "no_folder" in kept
    with pytest.raises(NoFolderChosen):
        update_project(data, "room", working_dir=str(folder))
    assert ensure_project_workspace(data, "room", tmp_path / "repo") == ""  # nothing is provisioned
    assert (data / "state" / "projects.json").read_bytes() == before

    # Choosing the Project's folder names the choice too: it ends No folder in one generation.
    bound = update_project(data, "room", working_dir=str(folder), no_folder=None)
    assert bound["working_dir"] == str(folder) and "no_folder" not in bound and bound["routing_generation"] == 2
    # The low-level removal stays what it was: a room with no choice, not a recorded No folder.
    bare = update_project(data, "room", working_dir="")
    assert bare["working_dir"] == "" and "no_folder" not in bare and bare["routing_generation"] == 3
    for invalid in ({"at": "now"}, {"at": "", "by": "owner"}, "none", {"at": "now", "by": "owner", "note": "x"}):
        with pytest.raises(ValueError):
            update_project(data, "room", no_folder=invalid)


# --- the view ------------------------------------------------------------------------------------------------

def test_view_names_what_one_look_found_and_what_that_state_offers(data, tmp_path, monkeypatch):
    from ouroboros.projects_registry import create_project, update_project

    usable, gone = _folder(tmp_path, "usable"), tmp_path / "folders" / "gone"
    plain_file = tmp_path / "folders" / "a-file"
    plain_file.write_text("not a folder", encoding="utf-8")
    copy = tmp_path / "copies" / "copies.noindex" / "dlg_helper_run"
    copy.mkdir(parents=True)
    for project_id, working_dir in (("usable", usable), ("missing", gone), ("file", plain_file), ("copy", copy), ("unset", "")):
        create_project(data, project_id, name=project_id.title(), working_dir=str(working_dir))
    create_project(data, "none", name="None")
    update_project(data, "none", no_folder={"at": "2026-11-15T12:00:00+00:00", "by": "owner"})
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "copies"))  # bound before it was the copies root

    seen = {pid: _view(data, pid) for pid in ("usable", "missing", "file", "copy", "unset", "none")}
    assert {pid: (view["state"], view["actions"]) for pid, view in seen.items()} == {
        "usable": ("usable", []), "missing": ("missing", ["clear", "choose"]),
        "file": ("unusable", ["clear", "choose"]), "copy": ("unusable", ["clear", "choose"]),
        "unset": ("unset", ["clear", "choose"]), "none": ("none", ["choose"])}
    assert seen["usable"]["reason"] == "" and seen["usable"]["folder"] == str(usable)
    assert "not a directory" in seen["missing"]["reason"] and seen["missing"]["folder"] == str(gone)
    assert "not a directory" in seen["file"]["reason"]
    assert "removed when" in seen["copy"]["reason"]
    assert seen["none"]["no_folder"] == {"at": "2026-11-15T12:00:00+00:00", "by": "owner"} and seen["unset"]["no_folder"] is None
    assert len({view["expected"] for view in seen.values()}) == 6
    assert _view(data, "absent") is None
    # Nothing was written by looking, and a bare drive publishes no queue: its work is unknown, never none.
    assert not _events(data) and seen["usable"]["work"] == {"observed": False, "kept": 0, "moved": [], "moved_total": 0}


# --- the change ----------------------------------------------------------------------------------------------

def test_clear_and_choose_repair_a_missing_folder_with_a_receipt_and_strict_expected(data, tmp_path):
    from ouroboros.projects_registry import create_project, get_project

    gone = tmp_path / "folders" / "gone"
    create_project(data, "room", name="Weekly", working_dir=str(gone))
    seen = _view(data)
    cleared = _change(data, "clear", expected=seen["expected"], actor="ouroboros:task-1")
    assert cleared["status"] == "changed" and cleared["view"]["state"] == "none" and cleared["view"]["actions"] == ["choose"]
    row = get_project(data, "room")
    assert row["working_dir"] == "" and row["no_folder"]["by"] == "ouroboros:task-1" and row["routing_generation"] == 1
    assert not gone.exists()  # no placeholder was created for the lost address
    [receipt] = _events(data)
    assert receipt == {key: value for key, value in cleared["receipt"].items() if key != "recorded"}
    assert cleared["receipt"]["recorded"] is True
    assert (receipt["actor"], receipt["action"]) == ("ouroboros:task-1", "clear")
    assert receipt["old"]["working_dir"] == str(gone) and receipt["old"]["state"] == "missing"
    assert receipt["new"] == {"working_dir": "", "no_folder": row["no_folder"]}

    # The word read before the change no longer names the choice: nothing is written.
    before = (data / "state" / "projects.json").read_bytes()
    stale = _change(data, "choose", expected=seen["expected"], path=str(_folder(tmp_path, "new")))
    assert (stale["status"], stale["reason"]) == ("refused", "stale") and stale["view"]["state"] == "none"
    assert (data / "state" / "projects.json").read_bytes() == before and len(_events(data)) == 1

    new = tmp_path / "folders" / "new"
    (new / "kept.txt").write_text("the owner's file", encoding="utf-8")
    chosen = _change(data, "choose", path=str(new))
    assert chosen["status"] == "changed" and chosen["view"]["state"] == "usable" and chosen["view"]["actions"] == []
    row = get_project(data, "room")
    assert row["working_dir"] == str(new.resolve()) and "no_folder" not in row and row["routing_generation"] == 2
    assert (row["provenance"], row["clone_url"]) == ("attached", "") and row["trusted_at"] == chosen["receipt"]["ts"]
    assert (new / "kept.txt").read_text(encoding="utf-8") == "the owner's file" and not (new / ".git").exists()
    assert [event["action"] for event in _events(data)] == ["clear", "choose"]
    # A folder in place is shown and never moved from here.
    healthy = _change(data, "clear")
    assert (healthy["status"], healthy["reason"]) == ("refused", "not_offered")
    assert get_project(data, "room")["working_dir"] == str(new.resolve())


@pytest.mark.parametrize("case", ["absent", "file", "home", "system_repo", "data", "host_copy", "empty"])
def test_choose_refuses_a_folder_that_cannot_last_and_writes_nothing(data, tmp_path, monkeypatch, case):
    from ouroboros.projects_registry import create_project

    create_project(data, "room", name="Room")
    copies = tmp_path / "copies"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(copies))
    copy = copies / "copies.noindex" / "dlg_helper_run"
    copy.mkdir(parents=True)
    plain_file = tmp_path / "a-file"
    plain_file.write_text("x", encoding="utf-8")
    path = {"absent": tmp_path / "nowhere", "file": plain_file, "home": pathlib.Path.home(), "system_repo": REPO / "docs",
            "data": data / "state", "host_copy": copy, "empty": ""}[case]
    before = (data / "state" / "projects.json").read_bytes()
    refused = _change(data, "choose", path=str(path))
    assert (refused["status"], refused["reason"]) == ("refused", "invalid_folder"), refused
    assert refused["detail"] and refused["view"]["state"] == "unset"
    assert (data / "state" / "projects.json").read_bytes() == before and not _events(data)


def test_the_writers_own_durable_folder_check_has_the_last_word(data, tmp_path, monkeypatch):
    """The attach validation and the registry writer share one durable-folder owner; whatever the writer refuses
    (a spelling only its check resolves) answers ``invalid_folder`` and writes nothing, never an error."""
    from ouroboros import project_sources
    from ouroboros.projects_registry import create_project

    create_project(data, "room", name="Room")
    folder = _folder(tmp_path, "looks-fine")

    def refuse(path, *, drive_root):
        raise project_sources.TemporaryCopyRefused(f"{path} is a host-managed private copy")

    monkeypatch.setattr(project_sources, "require_durable_folder", refuse)
    before = (data / "state" / "projects.json").read_bytes()
    refused = _change(data, "choose", path=str(folder))
    assert (refused["status"], refused["reason"]) == ("refused", "invalid_folder")
    assert "host-managed private copy" in refused["detail"] and refused["view"]["state"] == "unset"
    assert (data / "state" / "projects.json").read_bytes() == before and not _events(data)


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="a folder its owner may not search needs POSIX modes, and root searches it anyway")
@pytest.mark.parametrize("fault", ["no_search", "symlink_loop"])
def test_choosing_a_folder_one_may_not_look_into_is_refused_and_writes_nothing(client, data, tmp_path, fault):
    """A chosen folder whose contents may not be searched, or that resolves to itself, cannot serve: choose answers
    ``invalid_folder`` with the error (the route's 400), writes nothing and leaves the folder as it was, never a
    server error. Python 3.14's pathlib answers a look it may not take as absence: ``no_search`` skips there."""
    from ouroboros.projects_registry import create_project

    create_project(data, "room", name="Room")
    folder = _folder(tmp_path, "closed")
    if fault == "no_search":
        folder.chmod(0o000)
    else:
        folder.rmdir()
        folder.symlink_to(folder)  # the address now resolves to itself
    try:
        if fault == "no_search" and _look_reads_as_absent(folder / ".git"):
            pytest.skip("this Python's pathlib reads a folder it may not search as absent: no error reaches the look")
        before = (data / "state" / "projects.json").read_bytes()
        refused = _change(data, "choose", path=str(folder))
        assert (refused["status"], refused["reason"]) == ("refused", "invalid_folder"), refused
        assert refused["detail"] and refused["view"]["state"] == "unset"
        routed = client.post("/api/projects/room/folder",
                             json={"action": "choose", "expected": refused["view"]["expected"], "path": str(folder)})
        assert routed.status_code == 400 and routed.json()["reason_code"] == "invalid_folder", routed.text
        assert (data / "state" / "projects.json").read_bytes() == before and not _events(data)
    finally:
        if fault == "no_search":
            folder.chmod(0o755)
    if fault == "no_search":
        assert folder.is_dir() and not any(folder.iterdir())
    else:
        assert folder.is_symlink() and pathlib.Path(os.readlink(folder)) == folder


def test_a_row_that_moves_while_the_change_writes_is_refused_as_changed(data, tmp_path, monkeypatch):
    from ouroboros import project_folder, projects_registry
    from ouroboros.projects_registry import create_project, get_project

    create_project(data, "room", name="Room")
    other = _folder(tmp_path, "other")
    real = projects_registry.update_project

    def racing(drive_root, project_id, **kwargs):
        if "admission_basis" in kwargs and "no_folder" in kwargs:
            real(drive_root, project_id, working_dir=str(other))  # another writer lands first
        return real(drive_root, project_id, **kwargs)

    monkeypatch.setattr(projects_registry, "update_project", racing)
    refused = _change(data, "clear")
    assert (refused["status"], refused["reason"]) == ("refused", "changed")
    assert refused["view"]["folder"] == str(other) and refused["view"]["state"] == "usable"
    row = get_project(data, "room")
    assert row["working_dir"] == str(other) and "no_folder" not in row and not _events(data)
    unknown = project_folder.change_folder(data, "absent", action="clear", expected="x", actor="owner", repo_dir=REPO)
    assert (unknown["status"], unknown["reason"], unknown["view"]) == ("refused", "unknown_project", None)


# --- promoted work -------------------------------------------------------------------------------------------

def test_a_recorded_no_folder_room_starts_ordinary_work_in_its_own_scratch_and_provisions_nothing(data, tmp_path):
    import os

    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.tool_access_roots import folderless_scratch_dir

    create_project(data, "room", name="Room")
    assert _change(data, "clear")["status"] == "changed"
    def choice():
        return {key: get_project(data, "room").get(key) for key in ("working_dir", "no_folder", "routing_generation")}

    before = choice()
    enqueued: list = []
    outcome = _promote("ordinary", enqueued)
    assert outcome["status"] == "scheduled", outcome
    task = enqueued[0]
    assert not task.get("workspace_root") and "workspace_autoprovisioned" not in task["metadata"]
    assert task["metadata"]["resource_intent"] == {"kind": "room_default", "project_id": "room"}
    assert choice() == before and before["working_dir"] == "" and before["no_folder"]["by"] == "owner"
    assert not os.listdir(os.environ["OUROBOROS_SUBAGENT_PROJECTS_ROOT"])  # no shared folder appeared
    # The worker's default cwd is this task's own scratch; body work asked for by intent stays with the body.
    ctx = types.SimpleNamespace(workspace_root=None, task_metadata=task["metadata"], project_id="room",
                                drive_root=data, task_id="ordinary")
    assert folderless_scratch_dir(ctx) == (data / "task_drives" / "ordinary").resolve()
    body: list = []
    assert _promote("body", body, resource_intent={"kind": "system_repo"})["status"] == "scheduled"
    assert body[0]["metadata"]["resource_intent"] == {"kind": "system_repo"} and not body[0].get("workspace_root")
    assert folderless_scratch_dir(types.SimpleNamespace(
        workspace_root=None, task_metadata=body[0]["metadata"], project_id="room", drive_root=data, task_id="body")) is None
    assert choice() == before


def test_a_room_that_made_no_choice_still_provisions_its_first_promoted_task_a_folder(data):
    from ouroboros.projects_registry import create_project, get_project, update_project

    create_project(data, "room", name="Room")
    update_project(data, "room", provenance="none")
    enqueued: list = []
    assert _promote("legacy", enqueued)["status"] == "scheduled"
    folder = enqueued[0]["workspace_root"]
    assert folder and enqueued[0]["metadata"]["workspace_autoprovisioned"] is True
    row = get_project(data, "room")
    assert row["working_dir"] == folder and "no_folder" not in row


def test_a_tasks_own_workspace_root_never_ends_the_rooms_recorded_no_folder(data, tmp_path):
    from ouroboros.projects_registry import create_project, get_project, update_project

    create_project(data, "room", name="Room")
    update_project(data, "room", provenance="none", pinned=True)
    assert _change(data, "clear")["status"] == "changed"
    before = get_project(data, "room")
    folder = _folder(tmp_path, "one-off")
    enqueued: list = []
    outcome = _promote("oneoff", enqueued, workspace_root=str(folder))
    assert outcome["status"] == "scheduled", outcome
    assert enqueued[0]["workspace_root"] == str(folder) and "_room_folder_note" not in enqueued[0]
    assert enqueued[0]["metadata"]["resource_intent"] == {"kind": "explicit_resource", "root": str(folder)}
    row = get_project(data, "room")
    # Promotion touches activity; the skipped folder CAS changes none of the other facts.
    assert {key: value for key, value in row.items() if key != "last_active_at"} == {
        key: value for key, value in before.items() if key != "last_active_at"}
    assert row["provenance"] == "none" and not row.get("trusted_at") and row["pinned"] is True
    assert "keeps No folder" in outcome["source_note"] and "this task only" in outcome["source_note"]
    # The same one-off folder into a room that made no choice still becomes that room's folder.
    create_project(data, "open-room", name="Open")
    again: list = []
    assert _promote("oneoff2", again, project_id="open-room", workspace_root=str(folder))["status"] == "scheduled"
    assert get_project(data, "open-room")["working_dir"] == str(folder)


def test_a_promoted_source_chooses_the_projects_folder_and_ends_no_folder(data, tmp_path):
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.promotion_source import resolve_promote_source

    create_project(data, "room", name="Room")
    assert _change(data, "clear")["status"] == "changed"
    folder = _folder(tmp_path, "supplied")
    ctx = types.SimpleNamespace(DRIVE_ROOT=data, REPO_DIR=REPO)
    root, note, error, project_id, created = resolve_promote_source(ctx, str(folder), "room")
    assert (root, error, project_id, created) == (str(folder), "", "room", False), (note, error)
    row = get_project(data, "room")
    assert row["working_dir"] == str(folder) and "no_folder" not in row and row["provenance"] == "attached"


# --- admitted and moved work ---------------------------------------------------------------------------------

def _queue(data, *, running=(), pending=()):
    """The queue's publication as its readers find it."""
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text(json.dumps({
        "running": [{"id": task["id"], "task": task} for task in running],
        "pending": [{"id": task["id"], "task": task} for task in pending]}), encoding="utf-8")


def _moved(data, task, monkeypatch, project_id="room"):
    """The real move of one pending row into the room; no start has applied it yet."""
    from ouroboros.project_placement import move_task_to_project, placement_basis

    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda **kw: True)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: types.SimpleNamespace(broadcast=lambda event: None))
    move_task_to_project(data, task["id"], project_id, origin=ORIGIN, basis=placement_basis(data, task["id"], project_id),
                         running={}, pending=[task])


def test_view_tells_admitted_work_from_work_a_recorded_move_will_retarget(data, tmp_path, monkeypatch):
    from ouroboros import project_folder
    from ouroboros.projects_registry import create_project

    gone = tmp_path / "folders" / "gone"
    create_project(data, "room", name="Weekly", working_dir=str(gone))
    weekly = {"id": "weekly", "type": "task", "title": "Weekly defaults review", "text": "Review", "chat_id": 1,
              "metadata": {"resource_intent": {"kind": "system_repo"}}}
    notes = {"id": "notes", "type": "task", "title": "Reading notes", "text": "Read", "chat_id": 1, "metadata": {}}
    pinned = {"id": "pinned", "type": "task", "title": "Research", "text": "Read", "chat_id": 1,
              "metadata": {"resource_intent": {"kind": "explicit_none"}}}
    for task in (weekly, notes, pinned):
        _moved(data, task, monkeypatch)
    admitted = {"id": "admitted", "type": "task", "title": "Earlier work", "project_id": "room",
                "workspace_root": str(gone), "workspace_mode": "external", "metadata": {}}
    helper = {**admitted, "id": "helper", "delegation_role": "subagent", "parent_task_id": "admitted"}
    elsewhere = {"id": "elsewhere", "type": "task", "title": "Another room", "project_id": "other", "metadata": {}}
    _queue(data, running=[admitted, helper], pending=[weekly, notes, pinned, elsewhere])

    work = _view(data)["work"]
    assert work == {"observed": True, "kept": 2, "moved_total": 2, "moved": [
        {"task_id": "weekly", "name": "Weekly defaults review", "system_repo": True},
        {"task_id": "notes", "name": "Reading notes", "system_repo": False}]}
    view = _view(data)
    cleared, chosen = project_folder.effects(view, "clear"), project_folder.effects(view, "choose", "/new/folder")
    assert any("2 admitted work(s) keep the folder they were admitted with" in line for line in cleared)
    # Work on Ouroboros's own repository is told apart: No folder keeps it there, a chosen folder would take it away.
    assert any("Weekly defaults review" in line and "even though it already ran" in line
               and line.endswith("it stays on Ouroboros's own repository.") for line in cleared)
    assert any("Reading notes" in line and line.endswith("works in its own temporary area.") for line in cleared)
    assert any("Weekly defaults review" in line and "would work in /new/folder instead of Ouroboros's own repository" in line
               and line.endswith("clear keeps it there.") for line in chosen)
    assert any("Reading notes" in line and line.endswith("it works in /new/folder.") for line in chosen)
    assert chosen[0] == "New work in this Project starts in /new/folder."
    crowded = {**view, "work": {**work, "moved_total": 5}}
    assert project_folder.effects(crowded, "clear")[-1] == "3 more moved work(s) are not listed; the same holds for them."
    # An unreadable publication is unknown, never "no work".
    (data / "state" / "queue_snapshot.json").write_text("{not json", encoding="utf-8")
    unknown = _view(data)
    assert unknown["work"]["observed"] is False
    assert any("could not be read" in line for line in project_folder.effects(unknown, "clear"))


@pytest.mark.parametrize("repair", ["none", "clear", "choose"])
def test_moved_work_takes_the_rooms_default_at_its_next_start_and_admitted_work_keeps_its_own(
        data, tmp_path, monkeypatch, repair):
    """The weekly shape: body work moved into a room whose folder is gone. Readmission reads the room's
    default as it then stands: it fails on the lost folder, runs over the body once No folder is recorded,
    and runs in a chosen folder (the placement contract). A row admitted with the old folder is not retargeted."""
    from ouroboros.project_placement import readmit_placement
    from ouroboros.projects_registry import create_project
    from ouroboros.workspace_admission import WorkspaceRootError

    gone = tmp_path / "folders" / "gone"
    create_project(data, "room", name="Weekly", working_dir=str(gone))
    weekly = {"id": "weekly", "type": "task", "text": "Review", "chat_id": 1,
              "metadata": {"resource_intent": {"kind": "system_repo"}}}
    admitted = {"id": "admitted", "type": "task", "project_id": "room", "workspace_root": str(gone),
                "workspace_mode": "external", "metadata": {"resource_intent": {"kind": "room_default", "project_id": "room"}}}
    _moved(data, weekly, monkeypatch)
    replacement = _folder(tmp_path, "replacement")
    if repair == "clear":
        assert _change(data, "clear")["status"] == "changed"
    elif repair == "choose":
        assert _change(data, "choose", path=str(replacement))["status"] == "changed"

    if repair == "none":
        with pytest.raises(WorkspaceRootError):
            readmit_placement(data, weekly, repo_dir=REPO, running={}, pending=[weekly])
        assert "placement_applied_at" not in weekly
    else:
        assert readmit_placement(data, weekly, repo_dir=REPO, running={}, pending=[weekly])
        assert weekly["project_id"] == "room" and weekly["metadata"]["resource_intent"] == {"kind": "system_repo"}
        assert weekly["workspace_root"] == ("" if repair == "clear" else str(replacement))
        assert weekly["workspace_mode"] == ("" if repair == "clear" else "external")
    # No move is recorded for the admitted row: readmission leaves its folder where admission put it.
    assert readmit_placement(data, admitted, repo_dir=REPO, running={}, pending=[admitted])
    assert admitted["workspace_root"] == str(gone) and admitted["workspace_mode"] == "external"


# --- the owner's route and the mind's tool: one owner ---------------------------------------------------------

@pytest.fixture
def client(data, monkeypatch):
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    import ouroboros.config as cfg
    from ouroboros.gateway.router import collect_routes

    monkeypatch.setattr(cfg, "SETTINGS_PATH", data.parent / "settings.json")
    app = Starlette(routes=collect_routes(data_dir=data))
    app.state.drive_root, app.state.repo_dir = data, REPO
    with TestClient(app) as test_client:
        yield test_client


def test_the_new_project_dialogs_no_folder_is_recorded_and_an_older_client_keeps_provisioning(client, data):
    from ouroboros.projects_registry import get_project

    chosen = client.post("/api/projects", json={"name": "Research", "no_folder": True})
    assert chosen.status_code == 200, chosen.text
    project = chosen.json()["project"]
    assert project["no_folder"]["by"] == "owner" and project["working_dir"] == "" and project["provenance"] == "none"
    older = client.post("/api/projects", json={"name": "Older client"})
    assert older.status_code == 200 and "no_folder" not in older.json()["project"]
    assert older.json()["project"]["provenance"] == "none"
    # One source: the recorded choice excludes a folder; an existing id refuses a new explicit source.
    both = client.post("/api/projects", json={"name": "Both", "no_folder": True, "with_workspace": True})
    assert both.status_code == 400 and "choose ONE source" in both.json()["error"]
    repeat = client.post("/api/projects", json={"id": older.json()["project"]["id"], "no_folder": True})
    assert repeat.status_code == 409 and repeat.json()["error_code"] == "project_exists"
    assert "no_folder" not in get_project(data, older.json()["project"]["id"])
    assert client.post("/api/projects", json={"name": "Typed", "no_folder": "yes"}).status_code == 400
    # The list stays the registry's compact projection: no folder is inspected and no new field is published.
    listed = client.get("/api/projects").json()["projects"]
    assert {row["id"] for row in listed} == {project["id"], older.json()["project"]["id"]}
    assert all("no_folder" not in row for row in listed)


@pytest.mark.parametrize("source", ["attached", "unset", "none"])
@pytest.mark.parametrize("identity", ["name", "id"])
def test_no_folder_create_collision_is_a_conflict_and_sourceless_replay_is_unchanged(client, data, tmp_path, source, identity):
    from ouroboros.projects_registry import get_project

    body = {"name": "Existing room"}
    if source == "attached":
        body["path"] = str(_folder(tmp_path, "attached"))
    elif source == "none":
        body["no_folder"] = True
    created = client.post("/api/projects", json=body)
    assert created.status_code == 200, created.text
    project = created.json()["project"]
    before = get_project(data, project["id"])
    raw = (data / "state" / "projects.json").read_bytes()
    key = {identity: project[identity]}
    refused = client.post("/api/projects", json={**key, "no_folder": True})
    assert refused.status_code == 409 and refused.json()["error_code"] == "project_exists", refused.text
    replay = client.post("/api/projects", json=key)
    assert replay.status_code == 200 and replay.json()["project"]["created"] is False
    assert {k: replay.json()["project"].get(k) for k in before} == before
    assert get_project(data, project["id"]) == before
    assert (data / "state" / "projects.json").read_bytes() == raw


def test_the_route_and_the_tool_read_and_change_through_the_same_owner(client, data, tmp_path):
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.tools.control import _manage_project_folder

    gone = tmp_path / "folders" / "gone"
    create_project(data, "room", name="Weekly", working_dir=str(gone))
    ctx = types.SimpleNamespace(task_id="mind-1", task_metadata={}, task_contract={}, drive_root=data, repo_dir=REPO)

    seen = client.get("/api/projects/room/folder")
    assert seen.status_code == 200, seen.text
    view = seen.json()["folder"]
    told = json.loads(_manage_project_folder(ctx, "room"))
    assert told["ok"] is True and told["status"] == "view" and told["view"] == view
    assert (view["state"], view["actions"], view["folder"]) == ("missing", ["clear", "choose"], str(gone))
    assert set(told["effects"]) == {"clear", "choose"} and "Nothing changed" in told["note"]
    assert client.get("/api/projects/absent/folder").status_code == 404
    assert json.loads(_manage_project_folder(ctx, "absent"))["reason"] == "unknown_project"

    # The mind clears with the word the owner's route read; the owner's stale word is then refused.
    cleared = json.loads(_manage_project_folder(ctx, "room", action="clear", expected=view["expected"],
                                                reason="The helper's copy is gone; the review runs over the body."))
    assert cleared["ok"] is True and cleared["status"] == "changed" and cleared["view"]["state"] == "none"
    assert cleared["receipt"]["actor"] == "ouroboros:mind-1" and cleared["receipt"]["note"].startswith("The helper's copy")
    assert "say what you changed and why" in cleared["note"] and cleared["effects"][0].startswith("New work in this Project")
    assert get_project(data, "room")["no_folder"]["by"] == "ouroboros:mind-1"
    stale = client.post("/api/projects/room/folder", json={"action": "clear", "expected": view["expected"]})
    assert stale.status_code == 409 and stale.json()["reason_code"] == "stale"

    current = client.get("/api/projects/room/folder").json()["folder"]
    assert current == cleared["view"]
    new = _folder(tmp_path, "new")
    for body, status, code in (({"action": "choose", "expected": current["expected"], "path": str(gone)}, 400, "invalid_folder"),
                               ({"action": "clear", "expected": current["expected"]}, 409, "not_offered"),
                               ({"action": "move", "expected": current["expected"]}, 400, None),
                               ({"action": "choose"}, 400, None)):
        refused = client.post("/api/projects/room/folder", json=body)
        assert refused.status_code == status and refused.json().get("reason_code") == code, refused.text
    chosen = client.post("/api/projects/room/folder", json={"action": "choose", "expected": current["expected"], "path": str(new)})
    assert chosen.status_code == 200, chosen.text
    assert chosen.json()["folder"]["state"] == "usable" and chosen.json()["receipt"]["actor"] == "owner"
    assert get_project(data, "room")["working_dir"] == str(new)
    assert [(event["actor"], event["action"]) for event in _events(data)] == [("ouroboros:mind-1", "clear"), ("owner", "choose")]

    # The tool's own refusals write nothing: a change needs the view's word and a reason; a helper may only look.
    count = len(_events(data))
    assert "TOOL_ARG_ERROR" in _manage_project_folder(ctx, "room", action="clear", expected="", reason="why")
    assert "TOOL_ARG_ERROR" in _manage_project_folder(ctx, "room", action="clear", expected="word", reason=" ")
    assert "TOOL_ARG_ERROR" in _manage_project_folder(ctx, "Not Clean!", action="view")
    assert "TOOL_ARG_ERROR" in _manage_project_folder(ctx, "room", action="move")
    helper = types.SimpleNamespace(**{**vars(ctx), "task_metadata": {"delegation_role": "subagent"}})
    assert "subagents" in _manage_project_folder(helper, "room", action="clear", expected="word", reason="why")
    assert json.loads(_manage_project_folder(helper, "room"))["status"] == "view"
    healthy = json.loads(_manage_project_folder(ctx, "room"))
    assert healthy["view"]["actions"] == [] and healthy["effects"] == {} and "in place" in healthy["note"]
    assert len(_events(data)) == count


@pytest.mark.parametrize("failure", ["look", "queue"])
@pytest.mark.parametrize("action", ["clear", "choose"])
@pytest.mark.parametrize("door", ["tool", "route"])
def test_a_change_whose_reading_after_it_fails_answers_its_receipt_and_invites_no_second_write(
        client, data, tmp_path, monkeypatch, door, action, failure):
    """The row is written and its receipt appended; then the reading after it fails: the look at the folder (an
    unreadable path), or the queue (which the view already reads as unknown work). Each door answers the change it
    made, never PROJECTS_ERROR or a 500 that reads as nothing written; a resend of the same word is stale."""
    from ouroboros import project_folder, projects_registry, queue_observation
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.tools import control

    create_project(data, "room", name="Weekly", working_dir=str(tmp_path / "folders" / "gone"))
    _queue(data)  # a readable publication: the room's work reads as observed until the failure
    new = _folder(tmp_path, "new")
    assert _view(data)["work"]["observed"] is True
    body = {"action": action, "expected": _view(data)["expected"], **({"path": str(new)} if action == "choose" else {})}
    written, armed = [], [True]
    real_update, real_observed, real_queue = (projects_registry.update_project, project_folder._observed,
                                              queue_observation.load_queue_document)

    def update(*args, **kwargs):  # the real writer: the failure arms only once it has written
        written.append(real_update(*args, **kwargs))
        return written[-1]

    def observed(row, **kwargs):
        if written and armed[0] and failure == "look":
            raise PermissionError(13, "Permission denied", str(row.get("working_dir") or data))
        return real_observed(row, **kwargs)

    def queue_document(*args, **kwargs):
        if written and armed[0] and failure == "queue":
            raise OSError(5, "Input/output error", "queue_snapshot.json")
        return real_queue(*args, **kwargs)

    monkeypatch.setattr(projects_registry, "update_project", update)
    monkeypatch.setattr(project_folder, "_observed", observed)
    monkeypatch.setattr(queue_observation, "load_queue_document", queue_document)
    entry = next(item for item in control.get_tools() if item.name == "manage_project_folder")
    ctx = types.SimpleNamespace(task_id="mind-1", task_metadata={}, task_contract={}, drive_root=data, repo_dir=REPO)

    def send():
        if door == "tool":
            answer = json.loads(entry.handler(ctx, "room", reason="The helper's copy is gone.", **body))
            return answer, answer.get("view"), answer.get("view_error")
        reply = client.post("/api/projects/room/folder", json=body)
        assert reply.status_code in (200, 409), reply.text
        answer = reply.json()
        return answer, answer.get("folder"), answer.get("folder_error")

    answer, view, error = send()
    assert len(written) == 1 and "PROJECTS_ERROR" not in json.dumps(answer)
    receipt = answer["receipt"]
    assert _events(data) == [{key: value for key, value in receipt.items() if key != "recorded"}]
    if failure == "look":
        assert view is None and error.startswith("PermissionError")
        assert door == "route" or ("do not repeat it" in answer["note"] and answer["ok"] is True)
    else:  # the view stands; only its work is unknown
        assert error is None and view["work"]["observed"] is False
        assert view["state"] == ("none" if action == "clear" else "usable")
    row = get_project(data, "room")
    assert (row.get("no_folder", {}).get("by"), row["working_dir"]) == (
        (receipt["actor"], "") if action == "clear" else (None, str(new)))
    armed[0] = False  # the reading works again: a client that resends the word it held writes nothing
    again = send()[0]
    assert (again.get("reason") or again.get("reason_code")) == "stale" and len(written) == 1 and len(_events(data)) == 1
    assert _view(data)["state"] == ("none" if action == "clear" else "usable")


def _look_reads_as_absent(path: pathlib.Path) -> bool:
    """Whether this interpreter's pathlib answers a look it may not take as absence (3.14) rather than raising."""
    try:
        return not path.exists()
    except PermissionError:
        return False


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="a folder its owner may not search needs POSIX modes, and root searches it anyway")
@pytest.mark.parametrize("fault", ["no_search", "symlink_loop"])
def test_a_folder_the_filesystem_will_not_let_one_look_at_reads_as_unusable_and_clear_repairs_it(
        client, data, tmp_path, fault):
    """Astra 15.11.2026: the registered folder's way loses search permission, or the folder became a symlink loop.
    One look raises (``PermissionError``; ``RuntimeError`` on Python 3.10-3.12) where admission's checks expected a
    path: the view and every door over it read the folder as unusable, with that error as its reason, at its
    recorded address and with its own ``expected``. Clear records No folder; the folder itself is left as it was.
    Python 3.14's pathlib answers a look it may not take as absence and raises nothing: ``no_search`` skips there."""
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.tools.control import _manage_project_folder

    folder = _folder(tmp_path, "weekly")
    (folder / "kept.txt").write_text("the owner's file", encoding="utf-8")
    create_project(data, "room", name="Weekly", working_dir=str(folder))
    healthy = _view(data)
    assert (healthy["state"], healthy["actions"]) == ("usable", [])  # positive: the same folder in place
    way = folder.parent
    if fault == "no_search":
        way.chmod(0o000)
    else:
        (folder / "kept.txt").rename(tmp_path / "kept.txt")
        folder.rmdir()
        folder.symlink_to(folder)  # the address now resolves to itself
    try:
        if fault == "no_search" and _look_reads_as_absent(folder):
            pytest.skip("this Python's pathlib reads a folder it may not search as absent: no error reaches the look")
        view = _view(data)
        assert (view["state"], view["actions"], view["folder"]) == ("unusable", ["clear", "choose"], str(folder))
        assert view["expected"] == healthy["expected"] == _view(data)["expected"]  # the choice did not move
        if fault == "no_search":
            assert "PermissionError" in view["reason"], view["reason"]
        assert str(folder) in view["reason"]
        seen = client.get("/api/projects/room/folder")
        assert seen.status_code == 200 and seen.json()["folder"] == view, seen.text
        ctx = types.SimpleNamespace(task_id="mind-1", task_metadata={}, task_contract={}, drive_root=data, repo_dir=REPO)
        told = json.loads(_manage_project_folder(ctx, "room"))
        assert told["ok"] is True and told["view"] == view and set(told["effects"]) == {"clear", "choose"}
        stale = client.post("/api/projects/room/folder", json={"action": "clear", "expected": "a-word-never-read"})
        assert stale.status_code == 409 and stale.json()["reason_code"] == "stale" and not _events(data)
        cleared = client.post("/api/projects/room/folder", json={"action": "clear", "expected": view["expected"]})
        assert cleared.status_code == 200, cleared.text
        assert cleared.json()["folder"]["state"] == "none" and cleared.json()["receipt"]["old"] == {
            "working_dir": str(folder), "no_folder": None, "state": "unusable", "reason": view["reason"]}
        row = get_project(data, "room")
        assert row["working_dir"] == "" and row["no_folder"]["by"] == "owner"
    finally:
        way.chmod(0o755)
    if fault == "no_search":
        assert (folder / "kept.txt").read_text(encoding="utf-8") == "the owner's file"
    else:
        assert folder.is_symlink() and pathlib.Path(os.readlink(folder)) == folder


@pytest.mark.parametrize("moment", ["before_registration", "before_choice"])
def test_a_no_folder_create_never_clears_a_folder_a_promoted_source_attached_meanwhile(
        client, data, tmp_path, monkeypatch, moment):
    """Astra 15.11.2026: the supported promote ``source=`` path registers and attaches the same id after the route's
    absent-row check (``before_registration``: the route's registration then finds the row), or between the route's
    own registration and its folder choice (``before_choice``). Either way the route answers 409 and writes nothing
    more: the attached folder, its provenance and its generation stand, no No folder is recorded. Without such
    interleaving a No-folder create records its choice and a source-less create stays a lookup."""
    from ouroboros import projects_registry
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.projects_registry import get_project
    from ouroboros.promotion_source import resolve_promote_source

    pid = project_id_from_display_name("Research")
    attached = _folder(tmp_path, "attached")
    ctx = types.SimpleNamespace(DRIVE_ROOT=data, REPO_DIR=REPO)
    armed = [True]
    real_create, real_update = projects_registry.create_project, projects_registry.update_project

    def promote_first():
        armed[0] = False
        folder, _note, error, effective, created = resolve_promote_source(ctx, str(attached), pid)
        assert (folder, error, effective, created) == (str(attached), "", pid, moment == "before_registration"), error

    def create(drive_root, project_id, **kwargs):
        if armed[0] and moment == "before_registration" and kwargs.get("origin") == "owner_ui":
            promote_first()
        return real_create(drive_root, project_id, **kwargs)

    def update(drive_root, project_id, **kwargs):
        if armed[0] and moment == "before_choice" and "no_folder" in kwargs:
            promote_first()
        return real_update(drive_root, project_id, **kwargs)

    monkeypatch.setattr(projects_registry, "create_project", create)
    monkeypatch.setattr(projects_registry, "update_project", update)
    refused = client.post("/api/projects", json={"name": "Research", "no_folder": True})
    assert not armed[0], "the promoted source ran inside the route"
    assert refused.status_code == 409, refused.text
    row = get_project(data, pid)
    assert (row["working_dir"], row["provenance"], row["routing_generation"]) == (str(attached), "attached", 1)
    assert "no_folder" not in row and _view(data, pid)["state"] == "usable"
    raw = (data / "state" / "projects.json").read_bytes()
    # Positive: a source-less create of that name is a lookup that writes nothing.
    lookup = client.post("/api/projects", json={"name": "Research"})
    assert lookup.status_code == 200 and lookup.json()["project"]["created"] is False, lookup.text
    assert (data / "state" / "projects.json").read_bytes() == raw
    # Positive: with no one interleaving, a No-folder create records its choice.
    alone = client.post("/api/projects", json={"name": "Reading", "no_folder": True})
    assert alone.status_code == 200, alone.text
    assert alone.json()["project"]["no_folder"]["by"] == "owner" and alone.json()["project"]["working_dir"] == ""
    assert _view(data, alone.json()["project"]["id"])["state"] == "none"


def test_the_folder_listing_types_live_in_the_folder_leaf_and_stay_importable_from_contracts():
    """The folder browser's listing (``GET /api/fs/dirs``), shared by the New Project picker and the folder dialog,
    lives in the folder contract leaf; ``gateway/contracts.py`` re-exports it, and the schema finds one type."""
    from ouroboros.gateway import contracts, project_folder_contracts
    from ouroboros.gateway.schema import contract_typed_dict

    for name in ("FsDirsEntry", "FsDirsResponse"):
        leaf = getattr(project_folder_contracts, name)
        assert leaf.__module__ == "ouroboros.gateway.project_folder_contracts"
        assert getattr(contracts, name) is leaf and contract_typed_dict(name) is leaf
        assert name in contracts.__all__ and name in project_folder_contracts.__all__


def test_the_tool_is_registered_once_with_its_policy_and_classes():
    from ouroboros import tool_capabilities
    from ouroboros.tools import control

    entries = [entry for entry in control.get_tools() if entry.name == "manage_project_folder"]
    assert len(entries) == 1 and entries[0].safety_policy == "skip"
    schema = entries[0].schema["parameters"]
    assert schema["required"] == ["project_id"] and schema["properties"]["action"]["enum"] == ["view", "clear", "choose"]
    assert "manage_project_folder" in tool_capabilities.CORE_TOOL_NAMES
    assert "manage_project_folder" in tool_capabilities.OBSERVE_WORLD_MUTATION_TOOLS


def test_a_view_skips_the_safety_reading_and_every_change_is_read(monkeypatch):
    """A view reads one row, one folder and the queue's publication, as ``list_projects`` reads; clear and choose
    change a room's default and are read as before (``safety.CHECKED_ACTIONS``, beside ``manage_schedules``)."""
    from ouroboros import safety
    from ouroboros.tools import control

    class Read(Exception):
        pass

    declared = next(entry.safety_policy for entry in control.get_tools() if entry.name == "manage_project_folder")
    monkeypatch.setattr(safety, "get_safety_mode", lambda *_a, **_k: (_ for _ in ()).throw(Read()))
    for arguments in ({"project_id": "room"}, {"project_id": "room", "action": "view"}):
        assert safety.check_safety("manage_project_folder", arguments, declared_policy=declared) == (True, "")
    for action in ("clear", "choose"):
        with pytest.raises(Read):
            safety.check_safety("manage_project_folder", {"project_id": "room", "action": action, "expected": "w"},
                                declared_policy=declared)
    assert safety.check_safety("manage_schedules", {"action": "list"}) == (True, "")
    with pytest.raises(Read):  # the action table keeps the schedules' changes read as before
        safety.check_safety("manage_schedules", {"action": "create"})


def test_a_place_turn_changes_only_its_own_rooms_folder_and_may_view_any(data, tmp_path):
    """A Presence speaker (or work acting for its binding) writes only its current room, as its journal writes do;
    an arbitrary project_id does not widen that. Its refusal and a refused change are typed failures."""
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.tools.control import _manage_project_folder
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    for room in ("room", "other"):
        create_project(data, room, name=room, working_dir=str(tmp_path / "folders" / f"gone-{room}"))
    place = types.SimpleNamespace(task_id="place-turn", task_metadata={"presence": {"binding_id": "b-1"}},
                                  task_contract={}, project_id="room", drive_root=data, repo_dir=REPO)
    seen = json.loads(_manage_project_folder(place, "other"))
    assert seen["status"] == "view" and seen["view"]["state"] == "missing", "a place's work may read any room"
    refused = _manage_project_folder(place, "other", action="clear", expected=seen["view"]["expected"], reason="why")
    assert refused.startswith("⚠️ TOOL_FORBIDDEN") and refused.endswith("(manage_project_folder)"), refused
    assert LegacyTextResultAdapter.from_text("manage_project_folder", refused).status == "blocked"
    assert "no_folder" not in get_project(data, "other") and _events(data) == []
    own = json.loads(_manage_project_folder(place, "room", action="clear", expected=_view(data)["expected"], reason="gone"))
    assert own["status"] == "changed" and own["receipt"]["actor"] == "ouroboros:place-turn"
    stale = _manage_project_folder(place, "room", action="clear", expected="old-word", reason="again")
    assert json.loads(stale)["reason"] == "stale"
    typed = LegacyTextResultAdapter.from_text("manage_project_folder", stale)
    assert (typed.status, typed.code) == ("error", "TOOL_REPORTED_FAILURE"), "ok:false is a typed failure"


# --- one queue per recorded folder (owner 15.11.2026) ----------------------------------------------------------

def _admitted_in(data, tmp_path, project_id: str):
    """Room ``project_id``'s promoted work, admitted with the room's folder, then that folder goes missing, the room
    is set to No folder and the folder returns: the admitted work keeps it (``project_admission.admitted_folder``)."""
    import shutil

    work: list = []
    assert _promote(f"{project_id}-work", work, project_id=project_id)["status"] == "scheduled"
    shared = pathlib.Path(work[0]["workspace_root"])
    shutil.rmtree(shared)
    assert _change(data, "clear", project_id=project_id)["status"] == "changed"
    shared.mkdir()
    return work[0], shared


def _waits_beside(row: dict, holder: dict) -> bool:
    from ouroboros.project_lease import candidate_is_leasable, lane_holders, running_project_ids

    running = [{"task": holder}]
    return not candidate_is_leasable(row, running_project_ids(running), holders=lane_holders(running, [row]))


def test_a_bare_folder_task_derived_into_another_room_after_a_clear_waits_for_the_work_admitted_there(data, tmp_path):
    """Counterexample (a): rooms P and Q both name folder A, P first; P's work keeps A; A goes missing, P is cleared
    and A returns. A bare workspace task for A is now derived into Q (it was P, and waited on P's lane): it still
    waits for P's work in A, by the folder."""
    from ouroboros.projects_registry import create_project

    shared = _folder(tmp_path, "shared")
    create_project(data, "p-room", name="P", working_dir=str(shared))
    create_project(data, "q-room", name="Q", working_dir=str(shared))
    frozen, shared = _admitted_in(data, tmp_path, "p-room")
    bare: list = []
    assert _promote("bare", bare, project_id="", workspace_root=str(shared))["status"] == "scheduled"
    assert (bare[0]["project_id"], bare[0]["workspace_root"]) == ("q-room", frozen["workspace_root"])
    assert _waits_beside(bare[0], frozen)


def test_a_folder_attached_to_another_room_after_a_clear_waits_for_the_work_admitted_there(data, tmp_path):
    """Counterexample (b): after P's clear, A returns and is attached to Q, a room that made no choice, by a promoted
    task's own folder (compare-and-set while empty): both are room work, and Q's waits for P's work in A."""
    from ouroboros.projects_registry import create_project, get_project

    create_project(data, "p-room", name="P", working_dir=str(_folder(tmp_path, "shared")))
    frozen, shared = _admitted_in(data, tmp_path, "p-room")
    create_project(data, "q-room", name="Q")
    attached: list = []
    assert _promote("q-work", attached, project_id="q-room", workspace_root=str(shared))["status"] == "scheduled"
    assert get_project(data, "q-room")["working_dir"] == str(shared) == attached[0]["workspace_root"]
    assert _waits_beside(attached[0], frozen)
    elsewhere: list = []
    assert _promote("q-other", elsewhere, project_id="", workspace_root=str(_folder(tmp_path, "elsewhere")))[
        "status"] == "scheduled"
    assert not _waits_beside(elsewhere[0], frozen), "another folder starts beside it"
