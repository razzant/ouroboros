"""One id moves; running resources change only at a leased execution boundary."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.project_placement import (
    append_placement_notice, move_task_to_project, placement_basis,
    readmit_placement, refresh_pending_placement,
)
from ouroboros.projects_registry import (
    bind_task_to_project, create_project, get_project, project_binding_for_task, update_project,
)

ORIGIN = {"absent": "mid_task_no_origin"}
from tests.test_swarm_host_admission import host  # noqa: E402, F401


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from supervisor import queue, workers

    root = tmp_path / "data"
    repo = tmp_path / "repo"
    root.mkdir()
    repo.mkdir()
    for module in (queue, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", root)
        monkeypatch.setattr(module, "RUNNING", {})
        monkeypatch.setattr(module, "PENDING", [])
    monkeypatch.setattr(workers, "REPO_DIR", repo)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kw: True)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    return root, repo, workers


def place(root, task_id, pid, workers, **kw):
    return move_task_to_project(root, task_id, pid, origin=ORIGIN,
                               basis=placement_basis(root, task_id, pid),
                               running=workers.RUNNING, pending=workers.PENDING, **kw)


@pytest.mark.parametrize("entrance", ["mind", "owner"])
def test_same_work_moves_without_pulling_siblings(runtime, entrance):
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.tools.control import _move_to_project
    from ouroboros.utils import append_jsonl
    from supervisor.events_project_routing import _handle_ensure_project_scope

    root, repo, workers = runtime
    original = {"id": "one", "project_id": "", "workspace_root": str(repo)}
    workers.RUNNING.update(one={"task": original}, sibling={"task": {"id": "sibling"}})
    ctx = SimpleNamespace(task_id="one", task_metadata={}, task_contract={}, project_id="",
                          drive_root=root, budget_drive_root=root, pending_events=[])
    if entrance == "mind":
        host = SimpleNamespace(DRIVE_ROOT=root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                               append_jsonl=append_jsonl)
        ctx.event_queue = SimpleNamespace(put_nowait=lambda event: _handle_ensure_project_scope(event, host))
        assert _move_to_project(ctx, project_name="Room").startswith("OK:")
    else:
        async def body():
            return {"task_id": "one", "project_id": "room", "name": "Room"}
        request = SimpleNamespace(json=body, app=SimpleNamespace(state=SimpleNamespace(drive_root=root)))
        response = asyncio.run(api_project_from_task(request))
        assert response.status_code == 200, response.body
        assert json.loads(response.body)["binding"]["task_id"] == "one"
    assert project_binding_for_task(root, "one")["project_id"] == "room"
    assert project_binding_for_task(root, "sibling") is None
    assert original == {"id": "one", "project_id": "", "workspace_root": str(repo)}
    assert ctx.project_id == ""
    messages = []
    assert append_placement_notice(ctx, messages)
    assert not append_placement_notice(ctx, messages)
    assert len(messages) == 1 and "Knowledge is not automatically reloaded" in messages[0]["content"]


def test_bound_elsewhere_is_rename_and_creates_nothing(runtime):
    root, _, workers = runtime
    create_project(root, "first", name="First")
    bind_task_to_project(root, "one", "first", origin=ORIGIN)
    outcome = place(root, "one", "second", workers, name="Renamed")
    assert outcome["status"] == "delivered" and outcome["reason"] == "renamed"
    assert get_project(root, "first")["name"] == "Renamed"
    assert get_project(root, "second") is None
    assert project_binding_for_task(root, "one")["project_id"] == "first"


def test_generation_change_between_prepare_and_move_refuses(runtime, tmp_path):
    root, _, workers = runtime
    create_project(root, "room")
    basis = placement_basis(root, "one", "room")
    folder = tmp_path / "new-folder"
    folder.mkdir()
    update_project(root, "room", working_dir=str(folder))
    from ouroboros.project_admission import ProjectAdmissionError

    with pytest.raises(ProjectAdmissionError, match="changed"):
        move_task_to_project(root, "one", "room", origin=ORIGIN, basis=basis,
                             running=workers.RUNNING, pending=workers.PENDING)
    assert project_binding_for_task(root, "one") is None


@pytest.mark.parametrize("preselect", [False, True])
def test_running_move_waits_for_boundary_and_destination_lease(runtime, tmp_path, preselect):
    root, repo, workers = runtime
    folder = tmp_path / "room-folder"
    folder.mkdir()
    create_project(root, "room")
    update_project(root, "room", working_dir=str(folder))
    work = {"id": "one", "project_id": "old", "workspace_root": str(repo), "metadata": {}}
    workers.RUNNING["one"] = {"task": work}
    before = copy.deepcopy(work)
    place(root, "one", "room", workers)
    assert work == before
    del workers.RUNNING["one"]
    workers.PENDING.append(work)
    current_folder = tmp_path / "current-room-folder"
    current_folder.mkdir()
    update_project(root, "room", working_dir=str(current_folder))
    if preselect:
        assert refresh_pending_placement(root, work)
        assert work["project_id"] == "room" and work["workspace_root"] == str(repo)
    workers.RUNNING["other"] = {"task": {"id": "other", "project_id": "room"}}
    assert not readmit_placement(root, work, repo_dir=repo, running=workers.RUNNING, pending=workers.PENDING)
    assert work["workspace_root"] == str(repo)
    workers.RUNNING.clear()
    assert readmit_placement(root, work, repo_dir=repo, running=workers.RUNNING, pending=workers.PENDING)
    assert work["workspace_root"] == str(current_folder) and work["workspace_mode"] == "external"
    assert work["_project_admission"]["project"]["routing_generation"] == get_project(root, "room")["routing_generation"]


def test_a_move_into_a_folder_another_room_holds_waits_before_selection_and_changes_nothing(tmp_path, monkeypatch):
    """Owner 15.11.2026, one queue per folder: the folder a pending move's readmission would select is its lane before
    the chooser picks it (its first pass reads the work's own resource off Q, then decides: ``dispatch_choice``) and
    before readmission writes it, and a launch claimed on the admission lane, not yet RUNNING, holds that folder too
    (``worker_assignment._started_projection``)."""
    from supervisor import dispatch_choice, evolution_lifecycle, state
    from tests._budget_pause_exact_helpers import _install_queue

    data = tmp_path / "data"  # folders live beside the data drive, never inside it
    data.mkdir()
    queue, _state, workers = _install_queue(data, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    shared, old, repo = ((tmp_path / name).resolve() for name in ("shared", "old", "repo"))
    for folder in (shared, old, repo):
        folder.mkdir()
    create_project(data, "room")
    update_project(data, "room", working_dir=str(shared))
    work = {"id": "one", "type": "task", "chat_id": 0, "_attempt": 1, "project_id": "old-room", "root_task_id": "one",
            "workspace_root": str(old), "workspace_mode": "external", "metadata": {}}
    workers.PENDING.append(work)
    place(data, "one", "room", workers)
    holder = {"id": "other", "type": "task", "project_id": "elsewhere", "root_task_id": "other",
              "workspace_root": str(shared), "workspace_mode": "external"}
    workers.RUNNING["other"] = {"task": holder, "worker_id": 0}
    assert dispatch_choice.choose(queue, 5.0, ()) == [], "the chooser leaves it queued"
    assert work["_placement_folder"] == str(shared) and work["workspace_root"] == str(old)
    assert dispatch_choice.choose(queue, 5.0, ()) == [], "and its next pass names who keeps the folder"
    assert work["_folder_wait"] == {"task_id": "other", "project_id": "elsewhere", "folder": str(shared)}
    del workers.RUNNING["other"]
    assert dispatch_choice.choose(queue, 5.0, ()) == [work], "nothing else held it"
    assert "_folder_wait" not in work
    claimed = [{"task": holder}]  # that work's launch is claimed on the admission lane and not RUNNING yet
    assert not readmit_placement(data, work, repo_dir=repo, running=claimed, pending=workers.PENDING)
    assert work["workspace_root"] == str(old) and "placement_applied_at" not in work, "a refusal changes nothing"
    assert readmit_placement(data, work, repo_dir=repo, running=[], pending=workers.PENDING)
    assert work["workspace_root"] == str(shared) and "_placement_folder" not in work


@pytest.mark.parametrize("own_body", [True, False])
def test_a_moved_body_work_keeps_its_own_resource_and_never_waits_for_the_rooms_busy_folder(tmp_path, monkeypatch,
                                                                                          own_body):
    """The folder a pending move waits for is the one its readmission selects (``project_placement.placement_folder``):
    work that keeps Ouroboros's own body (``keeps_own_resource``) gets no folder from the room, so another room's work
    in that room's folder never holds it, at the chooser or at readmission; work without a resource of its own waits."""
    from ouroboros import subagent_worktrees as sw
    from supervisor import dispatch_choice, evolution_lifecycle, state
    from tests._budget_pause_exact_helpers import _install_queue

    data = tmp_path / "data"
    data.mkdir()
    queue, _state, workers = _install_queue(data, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    shared, repo, body = ((tmp_path / name).resolve() for name in ("shared", "repo", "body-copy"))
    for folder in (shared, repo, body):
        folder.mkdir()
    create_project(data, "room")
    update_project(data, "room", working_dir=str(shared))
    if own_body:
        sw._put_record({"path": str(body), "task_id": "one", "kind": "body_candidate"}, data_dir=data)
    work = {"id": "one", "type": "task", "chat_id": 0, "_attempt": 1, "project_id": "old-room", "root_task_id": "one",
            "workspace_root": "", "workspace_mode": "", "metadata": {}}
    workers.PENDING.append(work)
    place(data, "one", "room", workers)
    holder = {"id": "other", "type": "task", "project_id": "elsewhere", "root_task_id": "other",
              "workspace_root": str(shared), "workspace_mode": "external"}
    workers.RUNNING["other"] = {"task": holder, "worker_id": 0}
    started = list(workers.RUNNING.values())
    assert [row["id"] for row in dispatch_choice.choose(queue, 5.0, ())] == (["one"] if own_body else [])
    assert work["_placement_folder"] == ("" if own_body else str(shared))
    assert readmit_placement(data, work, repo_dir=repo, running=started, pending=workers.PENDING) is own_body
    if own_body:  # readmitted beside the busy folder: it stays on the body, recorded where every resolver reads it
        assert (work["workspace_root"], work["metadata"]["resource_intent"]) == ("", {"kind": "system_repo"})
        assert work["project_id"] == "room" and "_placement_folder" not in work
    else:  # refused: nothing of it changed
        assert work["workspace_root"] == "" and "placement_applied_at" not in work


def test_the_chooser_reads_a_moved_works_copy_records_once_and_its_launch_reads_them_as_before(tmp_path, monkeypatch):
    """The folder a moved body work waits for needs ``keeps_own_resource``, a read of every copy record: the chooser
    reads it once per move, off Q (``dispatch_choice._screen``), and carries it, never reading it under Q; the
    launch's readmission reads it under Q once, as before the folder lane. A candidate retired meanwhile moves the
    lane at once: the launch refuses beside the busy folder, and the next walks wait for it without reading."""
    from ouroboros import subagent_worktrees as sw
    from supervisor import dispatch_choice, evolution_lifecycle, state
    from tests._budget_pause_exact_helpers import _install_queue

    data = tmp_path / "data"
    data.mkdir()
    queue, _state, workers = _install_queue(data, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    shared, repo, body = ((tmp_path / name).resolve() for name in ("shared", "repo", "body-copy"))
    for folder in (shared, repo, body):
        folder.mkdir()
    create_project(data, "room")
    update_project(data, "room", working_dir=str(shared))
    sw._put_record({"path": str(body), "task_id": "one", "kind": "body_candidate"}, data_dir=data)
    work = {"id": "one", "type": "task", "chat_id": 0, "_attempt": 1, "project_id": "", "root_task_id": "one",
            "text": "work", "workspace_root": "", "workspace_mode": "",
            "metadata": {"resource_intent": {"kind": "system_repo"}}}
    workers.PENDING.append(work)
    place(data, "one", "room", workers)
    workers.RUNNING["other"] = {"task": {"id": "other", "type": "task", "project_id": "elsewhere",
                                         "root_task_id": "other", "workspace_root": str(shared),
                                         "workspace_mode": "external"}, "worker_id": 0}
    reads, real = [], sw._records
    monkeypatch.setattr(sw, "_records", lambda *a, **kw: reads.append(queue._queue_lock._is_owned()) or real(*a, **kw))
    walks = [[row["id"] for row in dispatch_choice.choose(queue, 100.0, ())] for _ in range(4)]
    assert walks == [["one"]] * 4 and reads == [False], "body work never waits for the busy folder; one read off Q"
    sw._drop_record(str(body), data)  # its candidate is retired before the launch
    with queue._queue_lock:
        assert not readmit_placement(data, work, repo_dir=repo, running=workers.RUNNING, pending=workers.PENDING)
    assert reads == [False, True] and (work["workspace_root"], work["_placement_folder"]) == ("", str(shared))
    assert [dispatch_choice.choose(queue, 100.0, ()) for _ in range(3)] == [[]] * 3 and len(reads) == 2
    assert work["_folder_wait"] == {"task_id": "other", "project_id": "elsewhere", "folder": str(shared)}


def test_bind_failure_never_changes_pending_or_running(runtime, monkeypatch):
    root, _, workers = runtime
    work = {"id": "one", "project_id": ""}
    workers.PENDING.append(work)
    monkeypatch.setattr("ouroboros.projects_registry.bind_task_to_project",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("write failed")))
    with pytest.raises(OSError, match="write failed"):
        place(root, "one", "room", workers)
    assert work == {"id": "one", "project_id": ""}


def test_deletion_after_prepare_blocks_move_and_boundary(runtime):
    from ouroboros.project_admission import ProjectAdmissionError
    from ouroboros.projects_registry import begin_project_deletion

    root, repo, workers = runtime
    create_project(root, "room")
    basis = placement_basis(root, "one", "room")
    place(root, "already-bound", "room", workers)
    begin_project_deletion(root, "room")
    with pytest.raises(ProjectAdmissionError):
        move_task_to_project(root, "one", "room", origin=ORIGIN, basis=basis,
                             running=workers.RUNNING, pending=workers.PENDING)
    assert project_binding_for_task(root, "one") is None
    pending = {"id": "already-bound", "project_id": "", "workspace_root": str(repo)}
    assert not refresh_pending_placement(root, pending)
    assert pending == {"id": "already-bound", "project_id": "", "workspace_root": str(repo)}


def test_live_memory_reads_follow_binding_without_reloading_or_rescoping(runtime):
    from ouroboros.tools.knowledge import _address
    from ouroboros.tools.project_journal import _resolve_project_id

    root, _, workers = runtime
    ctx = SimpleNamespace(task_id="one", project_id="", drive_root=root, budget_drive_root=root,
                          task_metadata={}, task_contract={}, is_direct_chat=True)
    assert _address(ctx, "notes").scope == "global"
    assert _resolve_project_id(ctx, "", write=True) == ("", "")
    place(root, "one", "room", workers)
    assert _address(ctx, "notes").scope == "project:room"
    assert _resolve_project_id(ctx, "", write=True) == ("room", "")
    assert "TOOL_FORBIDDEN" in _resolve_project_id(ctx, "other", write=True)[1]
    assert _address(ctx, "notes", "global").scope == "global"
    assert ctx.project_id == ""


def test_real_assignment_acquires_destination_before_changing_folder(host, tmp_path, monkeypatch):  # noqa: F811
    from tests.test_main_project_consistency import admit_main
    from tests.test_project_hold_recovery import worker
    from supervisor import workers

    task = admit_main(host, tid="one")
    folder = tmp_path / "destination"
    folder.mkdir()
    create_project(host.root, "room", working_dir=str(folder))
    before = task.get("workspace_root")
    move_task_to_project(host.root, "one", "room", origin=ORIGIN,
                         basis=placement_basis(host.root, "one", "room"),
                         running=workers.RUNNING, pending=host.pending)
    sent = worker(host, monkeypatch)
    workers.RUNNING["occupant"] = {"task": {"id": "occupant", "project_id": "room"}}
    workers.assign_tasks()
    assert not sent and task.get("workspace_root") == before
    workers.RUNNING.clear()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["one"]
    assert sent[0]["workspace_root"] == str(folder)
    assert sent[0]["task_contract"]["workspace"]["root"] == str(folder)
    assert workers.RUNNING["one"]["task"]["project_id"] == "room"


def test_real_admission_never_adopts_shared_origin(host):  # noqa: F811
    from ouroboros.project_dialogue import build_owner_message_ref
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    text = "One owner message"
    ref = build_owner_message_ref(chat_id=1, client_message_id="owner", ts="2026-10-16T12:00:00Z", text=text)
    create_project(host.root, "sibling-room")
    bind_task_to_project(host.root, "sibling", "sibling-room", origin={"ref": ref, "text": text})
    outcome = _handle_promote_chat_to_task({
        "type": "promote_chat_to_task", "task_id": "separate", "routing_token": "new-token",
        "objective": "Separate work", "title": "Separate", "source_ref": ref, "source_text": text,
        "chat_id": 1, "workspace": "none",
    }, host.ctx)
    assert outcome["status"] == "scheduled"
    assert len(host.pending) == 1 and host.pending[0]["id"] == "separate"
    assert not host.pending[0].get("project_id")
    assert project_binding_for_task(host.root, "separate") is None


def test_restore_of_old_execution_scope_readmits_committed_move(host, tmp_path, monkeypatch):  # noqa: F811
    from tests.test_main_project_consistency import admit_main
    from tests.test_project_hold_recovery import resume_after_app_stop, worker
    from ouroboros import subagent_worktrees as sw
    from supervisor import queue, workers

    original = admit_main(host, tid="one")
    assert queue.persist_queue_snapshot()
    saved = queue.QUEUE_SNAPSHOT_PATH.read_bytes()
    folder = tmp_path / "destination"
    folder.mkdir()
    create_project(host.root, "room", working_dir=str(folder))
    move_task_to_project(host.root, "one", "room", origin=ORIGIN,
                         basis=placement_basis(host.root, "one", "room"),
                         running=workers.RUNNING, pending=host.pending)
    queue.QUEUE_SNAPSHOT_PATH.write_bytes(saved)
    host.pending.clear()
    reads, real = [], sw._records  # the copy records: whether a read held Q
    monkeypatch.setattr(sw, "_records", lambda *a, **kw: reads.append(queue._queue_lock._is_owned()) or real(*a, **kw))
    assert queue.restore_pending_from_snapshot() == 1
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent  # A move never grants automatic resume after app stop.
    resume_after_app_stop(host, "one")
    workers.assign_tasks()
    assert [r["id"] for r in sent] == [original["id"]]
    assert sent[0]["workspace_root"] == str(folder) and sent[0]["project_id"] == "room"
    # Restore and the passes read no copy record under Q: the chooser once off Q, the launch's readmission as before.
    assert reads == [False, True]


def test_a_folder_wait_names_only_work_holding_the_folder_now_never_a_row_a_walk_picked(tmp_path, monkeypatch):
    """Opus 16.11.2026: two rooms share one recorded folder and a store gate (here a schedule's control) refuses both
    rows every pass. The row a walk picks holds the folder only for the rest of that walk and may never start, so the
    later row's wait never names it and a pass marks nothing (``dispatch_choice._collect``'s ``waits``). Work that
    does hold the folder is named once and stays named."""
    from supervisor import dispatch_choice, evolution_lifecycle, state
    from tests._budget_pause_exact_helpers import _install_queue

    data = tmp_path / "data"
    data.mkdir()
    queue, _state, workers = _install_queue(data, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    shared = (tmp_path / "shared").resolve()
    shared.mkdir()
    first, second = ({"id": tid, "type": "task", "chat_id": 0, "_attempt": 1, "project_id": room, "root_task_id": tid,
                      "workspace_root": str(shared), "workspace_mode": "external", "metadata": {}}
                     for tid, room in (("first", "room-a"), ("second", "room-b")))
    workers.PENDING.extend([first, second])
    monkeypatch.setattr(dispatch_choice, "_own_refusal", lambda queue, row, held: "schedule_control")
    marks = []
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kw: marks.append(kw.get("reason")) or True)

    def one_pass():  # ``run_pass``'s rounds with two idle workers: a walk, then one without what the store refused
        assert dispatch_choice.choose(queue, 5.0, (), limit=2) == []
        assert dispatch_choice.choose(queue, 5.0, (), limit=2, excluded={"first"}) == []

    for _ in range(3):
        one_pass()
    assert marks.count("folder_wait") == 0 and "_folder_wait" not in second, marks
    workers.RUNNING["other"] = {"task": {"id": "other", "type": "task", "project_id": "elsewhere",
                                         "root_task_id": "other", "workspace_root": str(shared),
                                         "workspace_mode": "external"}, "worker_id": 0}
    one_pass()
    one_pass()
    wait = {"task_id": "other", "project_id": "elsewhere", "folder": str(shared)}
    assert first["_folder_wait"] == second["_folder_wait"] == wait
    assert marks.count("folder_wait") == 2, "each named once, then unchanged"


def test_a_moved_rows_unreadable_own_resource_is_said_once_a_move_and_the_row_waits(tmp_path, monkeypatch, caplog):
    """Opus 16.11.2026: the chooser reads a moved row's own resource off Q (``dispatch_choice._screen``). A failed
    read keeps the row waiting and says why once a move at warning level, as readmission's error said it before
    the folder lane; the next passes say it only at debug."""
    import logging

    from ouroboros import project_placement
    from supervisor import dispatch_choice, evolution_lifecycle, state
    from tests._budget_pause_exact_helpers import _install_queue

    data = tmp_path / "data"
    data.mkdir()
    queue, _state, workers = _install_queue(data, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    monkeypatch.setattr(dispatch_choice, "_UNREAD_MOVES", {}, raising=False)
    shared = (tmp_path / "shared").resolve()
    shared.mkdir()
    create_project(data, "room")
    update_project(data, "room", working_dir=str(shared))
    work = {"id": "one", "type": "task", "chat_id": 0, "_attempt": 1, "project_id": "", "root_task_id": "one",
            "text": "work", "workspace_root": "", "workspace_mode": "",
            "metadata": {"resource_intent": {"kind": "system_repo"}}}
    workers.PENDING.append(work)
    place(data, "one", "room", workers)

    def unreadable(*_args, **_kwargs):
        raise OSError("a copy record is unreadable")

    monkeypatch.setattr(project_placement, "keeps_own_resource", unreadable)
    with caplog.at_level(logging.DEBUG, logger="supervisor.dispatch_choice"):
        assert [dispatch_choice.choose(queue, 100.0, ()) for _ in range(3)] == [[]] * 3
    assert project_placement.pending_resource(work) is not None, "it still waits for its own resource"
    said = [record.levelno for record in caplog.records if "Own resource of moved one" in record.getMessage()]
    assert said == [logging.WARNING, logging.DEBUG, logging.DEBUG]
