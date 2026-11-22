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
    assert dispatch_choice.choose(queue, 5.0, ()) == [], "and its next pass leaves it queued"
    with queue._queue_lock:  # the queue says who keeps the folder now; the chooser wrote nothing on the row
        assert dispatch_choice.lane_waits() == {"one": {"lane": "folder", "task_id": "other", "project_id": "elsewhere",
                                                        "folder": str(shared)}}
    assert "_lane_wait" not in work and "lane_wait" not in work
    del workers.RUNNING["other"]
    with queue._queue_lock:
        assert dispatch_choice.lane_waits() == {}
    assert dispatch_choice.choose(queue, 5.0, ()) == [work], "nothing else held it"
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
    with queue._queue_lock:
        assert dispatch_choice.lane_waits() == {"one": {"lane": "folder", "task_id": "other", "project_id": "elsewhere",
                                                        "folder": str(shared)}}


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


def test_a_lane_wait_names_only_work_holding_the_folder_now_never_a_row_a_walk_picked(tmp_path, monkeypatch):
    """Opus 16.11.2026: two rooms share one recorded folder and a store gate (here a schedule's control) refuses both
    rows every pass. The row a walk picks holds the folder only for the rest of that walk and may never start, so no
    wait names it (``dispatch_choice.lane_waits`` counts what runs or is claimed). Work that does hold the folder is
    named for both rows; the chooser writes nothing on a row and persists nothing for a wait."""
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
    with queue._queue_lock:
        assert dispatch_choice.lane_waits() == {}, "a walk's pick holds no lane"
    fields = {row["id"]: set(row) for row in (first, second)}  # as the chooser's own passes left them
    workers.RUNNING["other"] = {"task": {"id": "other", "type": "task", "project_id": "elsewhere",
                                         "root_task_id": "other", "workspace_root": str(shared),
                                         "workspace_mode": "external"}, "worker_id": 0}
    one_pass()
    one_pass()
    wait = {"lane": "folder", "task_id": "other", "project_id": "elsewhere", "folder": str(shared)}
    with queue._queue_lock:
        assert dispatch_choice.lane_waits() == {"first": wait, "second": wait}
    # The chooser writes nothing for a wait: no field on either row (the old ``_folder_wait`` note included) and no
    # publication for it (the old ``folder_wait`` reason included).
    assert {row["id"]: set(row) for row in (first, second)} == fields
    assert not {"folder_wait", "lane_wait"} & set(marks), marks


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


def test_a_queued_roots_lane_wait_is_what_holds_its_lane_now_whatever_the_chooser_saw(tmp_path, monkeypatch):
    """Owner 21.11.2026 («5A») and review round 1 (a note the chooser stored went stale and was never written on a full
    pool): the wait is computed under Q from the live rows by the lanes scheduling uses (``dispatch_choice.lane_waits``),
    with no walk at all, so it holds on a full pool and ends the moment the holder lets the lane go."""
    from ouroboros.project_placement import _OWN
    from supervisor import conversation_lane, dispatch_choice
    from tests._budget_pause_exact_helpers import _install_queue

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    shared = str((tmp_path / "shared").resolve())

    def root(tid, project="p", **fields):
        return {"id": tid, "type": "task", "chat_id": 0, "_attempt": 1, "project_id": project, "root_task_id": tid,
                "metadata": {}, **fields}

    def waits(running=(), pending=()):
        workers.RUNNING.clear()
        workers.RUNNING.update({meta["task"]["id"]: {"worker_id": 0, **meta} for meta in running})
        workers.PENDING[:] = list(pending)
        with queue._queue_lock:
            return dispatch_choice.lane_waits()

    queued, holder = root("q"), root("x")
    behind_x = {"lane": "project", "task_id": "x", "project_id": "p"}
    assert waits([{"task": holder}], [queued]) == {"q": behind_x}, "the pool is full and nothing walked: it is said"
    for reason in ("owner", "budget"):  # the owner's Pause or the spending limit gives the lane back
        assert waits([], [root("x", _budget_pause={"reason": reason}), queued]) == {}, reason
    assert waits([], [queued]) == {}, "the holder finished"
    asleep = root("x", _budget_pause={"reason": "sleep"})
    assert waits([], [asleep, queued]) == {}, "a solitary cold sleep releases its lane"
    member = {"task": root("x-kid", delegation_role="subagent", root_task_id="x")}
    assert waits([member], [asleep, queued]) == {"q": behind_x}, "a sleeping tree with a running member keeps it"
    assert waits([{"task": holder, "lane": "conversation"}], [queued]) == {}, "a conversation turn holds no lane"
    assert waits([], [root("x", _launch_claim={"token": "t"}), queued]) == {"q": behind_x}, "a claimed launch holds"
    for row in (root("q-kid", delegation_role="subagent", root_task_id="y"), root("q", project="")):
        assert waits([{"task": holder}], [row]) == {}, row
    monkeypatch.setattr(conversation_lane, "resumes_on_lane", lambda row: row.get("id") == "q")
    assert waits([{"task": holder}], [queued]) == {}, "a returned turn takes no project lane"
    monkeypatch.undo()
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    # Folder lane: another Project's tree in the recorded folder; the Project's lane wins when both hold.
    folder = {"workspace_root": shared, "workspace_mode": "external"}
    other = {"task": root("o", project="other", **folder)}
    assert waits([other], [root("q", **folder)]) == {"q": {"lane": "folder", "task_id": "o", "project_id": "other",
                                                           "folder": shared}}
    assert waits([other, {"task": root("x", **folder)}], [root("q", **folder)]) == {"q": behind_x}
    # A sleeper granted back to a folder it already holds waits only for those acting in it (one still parked waits
    # for its sleep, below).
    sleeper = root("s", project="s", _budget_pause_resume={"pause": {"reason": "sleep"}}, **folder)
    kid = {"task": root("s-kid", project="s", delegation_role="subagent", root_task_id="s")}
    assert waits([kid, other], [sleeper]) == {"s": {"lane": "folder", "task_id": "o", "project_id": "other",
                                                    "folder": shared}}
    assert waits([kid], [sleeper]) == {}
    # Two exclusively cold trees in one folder never wait for each other; once the second root acts, the first does.
    other_sleeper = root("t", project="t", _budget_pause={"reason": "sleep"}, **folder)
    other_kid = {"task": root("t-kid", project="t", delegation_role="subagent", root_task_id="t")}
    assert waits([kid, other_kid], [sleeper, other_sleeper]) == {}
    woken = {"task": root("t", project="t", **folder)}
    assert waits([kid, woken], [sleeper]) == {"s": {"lane": "folder", "task_id": "t", "project_id": "t",
                                                    "folder": shared}}
    # A moved row whose own resource is unread keeps its Project's answer; its folder lane reads none meanwhile.
    unread = {_OWN: {"bound_at": "2026-11-21T00:00:00Z", "kind": "move"}, "_placement_folder": ""}
    assert waits([{"task": root("x")}], [root("q", **unread)]) == {"q": behind_x}
    assert waits([other], [root("q", **folder, **unread)]) == {}
    # A folder is said while it answers for the room memory admitted the row to: that room's recorded folder, or an
    # intent that keeps its own; a row whose room changed and whose folder facts did not yet says none of it.
    def admitted(working_dir, kind="room_default"):
        return {"_project_admission": {"project_id": "p", "project": {"id": "p", "working_dir": working_dir}},
                "metadata": {"resource_intent": {"kind": kind, "project_id": "p"}}}

    by_folder = {"q": {"lane": "folder", "task_id": "o", "project_id": "other", "folder": shared}}
    assert waits([other], [root("q", **folder, **admitted(shared))]) == by_folder
    assert waits([other], [root("q", **folder, **admitted("/elsewhere", "explicit_resource"))]) == by_folder
    assert waits([other], [root("q", **folder, **admitted("/elsewhere"))]) == {}
    assert waits([other, {"task": root("x")}], [root("q", **folder, **admitted("/elsewhere"))]) == {"q": behind_x}
    # Only a row the chooser's own memory gates would admit waits for its lane (``_candidate_ineligible`` with no
    # lanes and no stat reads): its pause, sleep, hold, the owner's hold, a launch claim or its root's fence first.
    running = [{"task": root("x")}]
    for fields in ({"_budget_pause": {"reason": "owner"}}, {"_budget_pause": {"reason": "sleep"}},
                   {"_budget_pause_hold": {"reason": "panic_hold"}}, {"_owner_hold": {"reason": "owner"}},
                   {"_launch_claim": {"token": "t"}}):
        assert waits(running, [root("q", **fields)]) == {}, fields
    queue.BUDGET_ROOT_FENCES["q"] = {"root_task_id": "q", "fence_id": "f1", "status": "paused", "cause": "owner_pause"}
    assert waits(running, [root("q")]) == {}, "its root's spending fence, not selected against it"
    grant = {"pause_id": "pz", "grant_id": "g1", "authority": "explicit_resume", "root_fence_id": "f1",
             "root_grant_id": "rg", "root_resume_generation": 1}  # Opus S1: a member selected against the fence
    assert waits(running, [root("q", _budget_pause_resume=grant)]) == {"q": behind_x}
    assert waits(running, [root("q", _resume_selection={"fence_id": "f1"})]) == {"q": behind_x}
    queue.BUDGET_ROOT_FENCES.clear()
    resumed = root("q", _budget_pause_resume={"pause_id": "pz", "grant_id": "g1", "authority": "explicit_resume"})
    assert waits(running, [resumed]) == {"q": behind_x}, "Astra B3: its Resume was granted"


def test_a_moved_row_names_no_holder_of_the_folder_it_left_until_the_chooser_reads_its_new_one(tmp_path, monkeypatch):
    """Review round 3 (Astra B1) and round 4: a move gives a queued row its new room at once and its folder facts
    only at the chooser's next look (``project_placement`` writes no lane fact), so meanwhile its folder is said
    only while it answers for that room (``dispatch_choice._folder_answers``): it names no holder of the folder it
    left, its Project's answer stays, and after the chooser's look it names the holder of its new room's folder."""
    from supervisor import dispatch_choice, evolution_lifecycle, state
    from tests._budget_pause_exact_helpers import _install_queue

    data = tmp_path / "data"
    data.mkdir()
    queue, _state, workers = _install_queue(data, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    old, new = ((tmp_path / name).resolve() for name in ("old", "new"))
    for folder in (old, new):
        folder.mkdir()
    create_project(data, "dest")
    update_project(data, "dest", working_dir=str(new))
    workers.PENDING.append({"id": "q", "type": "task", "chat_id": 0, "_attempt": 1, "project_id": "prior",
                            "root_task_id": "q", "workspace_root": str(old), "workspace_mode": "external",
                            "metadata": {"resource_intent": {"kind": "room_default", "project_id": "prior"}}})

    def holder(tid, project, folder=None):
        workers.RUNNING[tid] = {"worker_id": 0, "task": {"id": tid, "type": "task", "project_id": project,
                                                         "root_task_id": tid, **({"workspace_root": str(folder),
                                                         "workspace_mode": "external"} if folder else {})}}

    def said():
        with queue._queue_lock:
            return dispatch_choice.lane_waits()

    holder("x", "other", old)
    assert said() == {"q": {"lane": "folder", "task_id": "x", "project_id": "other", "folder": str(old)}}
    place(data, "q", "dest", workers)
    assert said() == {}, "right after the move it names no holder of the folder it left"
    holder("z", "dest")
    assert said() == {"q": {"lane": "project", "task_id": "z", "project_id": "dest"}}, "its Project's answer stays"
    del workers.RUNNING["z"], workers.RUNNING["x"]
    holder("y", "other", new)
    assert dispatch_choice.choose(queue, 100.0, ()) == [], "the chooser's look: its new room's folder is held"
    assert said() == {"q": {"lane": "folder", "task_id": "y", "project_id": "other", "folder": str(new)}}


def test_a_moved_sleeping_root_keeps_its_helpers_folder_from_the_chooser(tmp_path, monkeypatch):
    """Review round 3 (Astra B1), the invariant round 3 broke: a sleeping root whose helper still writes in its
    folder is moved to a room with another folder; the chooser still keeps the old folder for that tree, so of
    two independent works it admits the one in the new folder, as before the move."""
    from tests.test_sleep_tree_lease import _run, _task, cold_park
    from supervisor import dispatch_choice, evolution_lifecycle, state
    from tests._budget_pause_exact_helpers import _install_queue

    data = tmp_path / "data"
    data.mkdir()
    queue, _state, workers = _install_queue(data, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda frame: None))
    old, new = tmp_path / "old", tmp_path / "new"
    for folder in (old, new):
        folder.mkdir()
    create_project(data, "dest", working_dir=str(new))
    external = {"workspace_mode": "external"}
    _run(workers, {**_task(data, "x", project="prior"), "workspace_root": str(old), **external,
                   "metadata": {"resource_intent": {"kind": "room_default", "project_id": "prior"}}})
    _run(workers, {**_task(data, "kid", tree="x", project="prior", launch_handoffs={"op": {"tool": "write_file"}}),
                   "workspace_root": str(old), **external})
    cold_park(data, monkeypatch, workers, queue, "x", tasks=["kid"])
    workers.PENDING += [{"id": tid, "type": "task", "chat_id": 1, "_attempt": 1, "project_id": tid,
                         "root_task_id": tid, "metadata": {}, "workspace_root": str(folder), **external}
                        for tid, folder in (("old-writer", old), ("new-writer", new))]
    place(data, "x", "dest", workers)
    assert [row["id"] for row in dispatch_choice.choose(queue, 100, (), limit=1)] == ["new-writer"]


def test_a_failed_lane_wait_reads_as_none_and_is_warned_once_per_distinct_error(tmp_path, monkeypatch, caplog):
    """Review round 2 (Opus A2, Grok, Astra A2): a display fact never stops a snapshot or the census, so a failure
    says no wait, and an operator sees it: one warning per distinct error, not one per call."""
    import logging

    from ouroboros import project_lease
    from supervisor import dispatch_choice
    from tests._budget_pause_exact_helpers import _install_queue

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatch_choice, "_LANE_WAIT_ERRORS", set())
    failures = iter(["first", "first", "second"])

    def broken(*_args, **_kwargs):
        raise RuntimeError(next(failures))

    monkeypatch.setattr(project_lease, "lane_holders", broken)
    with caplog.at_level(logging.WARNING, logger="supervisor.dispatch_choice"), queue._queue_lock:
        assert [dispatch_choice.lane_waits() for _ in range(3)] == [{}, {}, {}]
    said = [record.getMessage() for record in caplog.records if "Lane waits unavailable" in record.getMessage()]
    assert len(said) == 2 and "first" in said[0] and "second" in said[1], said
