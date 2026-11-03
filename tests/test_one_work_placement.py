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
    assert queue.restore_pending_from_snapshot() == 1
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent  # A move never grants automatic resume after app stop.
    resume_after_app_stop(host, "one")
    workers.assign_tasks()
    assert [r["id"] for r in sent] == [original["id"]]
    assert sent[0]["workspace_root"] == str(folder) and sent[0]["project_id"] == "room"
